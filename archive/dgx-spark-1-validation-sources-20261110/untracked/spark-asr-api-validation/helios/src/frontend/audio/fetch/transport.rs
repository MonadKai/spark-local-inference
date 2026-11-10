/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Count received TCP bytes before TLS/HTTP parsing and retain an absolute limit.

use super::*;
use ureq::unversioned::transport::{Buffers, ConnectionDetails, Connector, Transport};

const RECEIVED_LIMIT: usize = 32 * 1024 * 1024;

#[derive(Debug)]
pub(super) struct Budget {
    pub(super) deadline: Instant,
    pub(super) cancelled: Arc<AtomicBool>,
    received: AtomicUsize,
}

impl Budget {
    pub(super) fn new(deadline: Instant, cancelled: Arc<AtomicBool>) -> Arc<Self> {
        Arc::new(Self { deadline, cancelled, received: AtomicUsize::new(0) })
    }
    pub(super) fn check(&self) -> std::result::Result<(), ureq::Error> {
        if self.cancelled.load(Ordering::Acquire) {
            // Rustls and compression readers retry Interrupted indefinitely;
            // cancellation is a terminal request failure, not a signal retry.
            return Err(ureq::Error::Io(io::Error::other("audio fetch cancelled")));
        }
        if Instant::now() >= self.deadline { return Err(ureq::Error::Timeout(ureq::Timeout::Global)); }
        Ok(())
    }
    fn timeout(&self, requested: NextTimeout) -> std::result::Result<NextTimeout, ureq::Error> {
        self.check()?;
        if requested.after.is_zero() { return Err(ureq::Error::Timeout(requested.reason)); }
        let remaining = self.deadline.saturating_duration_since(Instant::now());
        let after = remaining.min(*requested.after).min(STEP_TIMEOUT);
        if after.is_zero() { return Err(ureq::Error::Timeout(ureq::Timeout::Global)); }
        Ok(NextTimeout { after: after.into(), reason: requested.reason })
    }
    fn account(&self, bytes: usize) -> std::result::Result<(), ureq::Error> {
        self.received.fetch_update(Ordering::AcqRel, Ordering::Acquire,
            |received| received.checked_add(bytes).filter(|&total| total <= RECEIVED_LIMIT))
            .map(|_| ()).map_err(|_| ureq::Error::Io(io::Error::new(io::ErrorKind::InvalidData,
                "audio HTTPS received byte budget exceeded")))
    }
}

#[derive(Debug)]
pub(super) struct GuardConnector(pub(super) Arc<Budget>);

impl<T: Transport> Connector<T> for GuardConnector {
    type Out = GuardTransport<T>;
    fn connect(&self, _: &ConnectionDetails, chained: Option<T>) -> std::result::Result<Option<Self::Out>, ureq::Error> {
        self.0.check()?;
        match chained {
            Some(mut inner) => {
                self.0.account(inner.buffers().input().len())?;
                Ok(Some(GuardTransport { inner, budget: self.0.clone() }))
            }
            None => Ok(None),
        }
    }
}

#[derive(Debug)]
pub(super) struct GuardTransport<T> { inner: T, budget: Arc<Budget> }

impl<T: Transport> Transport for GuardTransport<T> {
    fn buffers(&mut self) -> &mut dyn Buffers { self.inner.buffers() }
    fn transmit_output(&mut self, amount: usize, timeout: NextTimeout) -> std::result::Result<(), ureq::Error> {
        let timeout = self.budget.timeout(timeout)?;
        self.inner.transmit_output(amount, timeout)?;
        self.budget.check()
    }
    fn await_input(&mut self, timeout: NextTimeout) -> std::result::Result<bool, ureq::Error> {
        self.budget.check()?;
        let now = Instant::now();
        let local_deadline = now + self.budget.deadline.saturating_duration_since(now).min(*timeout.after);
        loop {
            let requested = NextTimeout { after: local_deadline.saturating_duration_since(Instant::now()).into(), reason: timeout.reason };
            let step = self.budget.timeout(requested)?;
            let before = self.inner.buffers().input().len();
            let result = self.inner.await_input(step);
            let after = self.inner.buffers().input().len();
            let received = after.checked_sub(before).ok_or_else(|| ureq::Error::Io(io::Error::new(
                io::ErrorKind::InvalidData, "audio transport discarded uncounted input")))?;
            self.budget.account(received)?;
            self.budget.check()?;
            match result {
                Err(ureq::Error::Timeout(_)) if Instant::now() < local_deadline => continue,
                result => return result,
            }
        }
    }
    fn is_open(&mut self) -> bool { self.budget.check().is_ok() && self.inner.is_open() }
    fn is_tls(&self) -> bool { self.inner.is_tls() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, net::{TcpListener, SocketAddr}};
    use ureq::unversioned::resolver::Resolver;
    use ureq::unversioned::transport::{TcpConnector, RustlsConnector};

    // Only these transport tests pin a local address. Production uses the
    // independently tested PublicResolver and HTTPS-only connector chain.
    #[derive(Debug)]
    struct LocalResolver(SocketAddr);
    impl Resolver for LocalResolver {
        fn resolve(&self, _: &ureq::http::Uri, _: &ureq::config::Config, _: NextTimeout)
            -> std::result::Result<ResolvedSocketAddrs, ureq::Error> {
            let mut addresses = self.empty();
            addresses.push(self.0);
            Ok(addresses)
        }
    }

    fn client(address: SocketAddr, budget: Arc<Budget>) -> ureq::Agent {
        ureq::Agent::with_parts(ureq::Agent::config_builder().proxy(None)
            .input_buffer_size(16384).timeout_global(Some(Duration::from_secs(10))).build(),
            ().chain(TcpConnector::default()).chain(GuardConnector(budget)), LocalResolver(address))
    }

    fn request_headers(stream: &mut std::net::TcpStream) {
        stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut headers = Vec::new();
        let mut byte = [0];
        while !headers.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).unwrap();
            headers.push(byte[0]);
            assert!(headers.len() <= 8192);
        }
    }

    #[test]
    fn actual_tcp_input_hits_the_cumulative_received_budget() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
            request_headers(&mut stream);
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 33554433\r\n\r\n").unwrap();
            let block = [b'a'; 16384];
            for _ in 0..2048 {
                if stream.write_all(&block).is_err() { return; }
            }
            let _ = stream.write_all(b"a");
        });
        let budget = Budget::new(Instant::now() + Duration::from_secs(10), Arc::new(AtomicBool::new(false)));
        let mut response = client(address, budget.clone()).get("http://audio.test/input").call().unwrap();
        let mut data = Vec::new();
        let error = response.body_mut().as_reader().read_to_end(&mut data).unwrap_err();
        assert!(error.to_string().contains("received byte budget exceeded"), "{error}");
        assert!(budget.received.load(Ordering::Acquire) <= RECEIVED_LIMIT);
        drop(response);
        server.join().unwrap();
    }

    #[test]
    fn a_stalled_tcp_body_uses_the_absolute_fetch_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (release, wait) = mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            request_headers(&mut stream);
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n").unwrap();
            let _ = wait.recv_timeout(Duration::from_secs(2));
        });
        let budget = Budget::new(Instant::now() + Duration::from_millis(100), Arc::new(AtomicBool::new(false)));
        let result = client(address, budget).get("http://audio.test/input").call();
        match result {
            Ok(mut response) => assert!(response.body_mut().as_reader().read_exact(&mut [0]).is_err()),
            Err(ureq::Error::Timeout(_)) => {},
            Err(error) => panic!("unexpected transport failure: {error}"),
        }
        let _ = release.send(());
        server.join().unwrap();
    }

    #[test]
    fn cancelling_an_actual_tls_handshake_is_terminal_instead_of_interrupted_retry() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let server_cancelled = cancelled.clone();
        let (release, wait) = mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let mut hello = [0; 4096];
            assert!(stream.read(&mut hello).unwrap() > 0);
            // Let the client's initial TLS write finish before cancellation.
            std::thread::sleep(Duration::from_millis(20));
            server_cancelled.store(true, Ordering::Release);
            let _ = wait.recv_timeout(Duration::from_secs(5));
        });
        let (send, result) = mpsc::channel();
        let client = std::thread::spawn(move || {
            let budget = Budget::new(Instant::now() + Duration::from_secs(4), cancelled);
            let connector = ().chain(TcpConnector::default()).chain(GuardConnector(budget))
                .chain(RustlsConnector::default());
            let agent = ureq::Agent::with_parts(ureq::Agent::config_builder().proxy(None)
                .timeout_global(Some(Duration::from_secs(4))).build(), connector, LocalResolver(address));
            let _ = send.send(agent.get("https://audio.test/input").call().map(|_| ()));
        });
        let response = result.recv_timeout(Duration::from_secs(3)).expect("TLS cancellation entered a retry loop");
        assert!(response.is_err());
        let _ = release.send(());
        client.join().unwrap();
        server.join().unwrap();
    }

    #[cfg(feature = "server")]
    #[test]
    fn actual_gzip_delivery_preserves_audio_bytes_and_enforces_the_decoded_file_limit() {
        use flate2::{Compression, write::GzEncoder};
        let maximum = 25 * 1024 * 1024;
        for (length, damage, succeeds) in [(44, 0, true), (maximum, 0, true),
            (maximum + 1, 0, false), (0, 0, false), (1024, 1, false), (1024, 2, false)] {
            let original = vec![0x52; length];
            let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(&original).unwrap();
            let mut compressed = encoder.finish().unwrap();
            if damage == 1 {
                let index = compressed.len() - 8;
                compressed[index] ^= 1;
            } else if damage == 2 {
                compressed.truncate(compressed.len() - 4);
            }
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                stream.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
                request_headers(&mut stream);
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: {}\r\n\r\n", compressed.len()).unwrap();
                let _ = stream.write_all(&compressed);
            });
            let cancelled = Arc::new(AtomicBool::new(false));
            let deadline = Instant::now() + Duration::from_secs(10);
            let budget = Budget::new(deadline, cancelled.clone());
            let mut response = client(address, budget).get("http://audio.test/input")
                .header("Accept-Encoding", "identity").call().unwrap();
            assert!(response.headers().get("content-encoding").is_none());
            let result = read_body(response.body_mut().with_config().limit(HTTP_ENTITY_READ_LIMIT).reader(),
                maximum, deadline, &cancelled);
            if succeeds {
                assert_eq!(result.unwrap(), original, "length={length} damage={damage}");
            } else {
                assert!(result.is_err(), "length={length} damage={damage}");
            }
            drop(response);
            server.join().unwrap();
        }
    }
}
