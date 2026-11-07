/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Bounded public HTTPS audio acquisition with connection-bound address checks.

use crate::{Result, frontend::deepseek::invalid};
use std::{io::{self, Read}, net::{IpAddr, SocketAddr, ToSocketAddrs},
    sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}, mpsc}, time::{Duration, Instant}};
use ureq::unversioned::{resolver::{Resolver, ResolvedSocketAddrs, DefaultResolver},
    transport::{Connector, TcpConnector, RustlsConnector, NextTimeout}};

mod transport;
use transport::{Budget, GuardConnector};

const STEP_TIMEOUT: Duration = Duration::from_secs(2);
const FETCH_TIMEOUT: Duration = Duration::from_secs(30);
const REDIRECTS: usize = 3;
// An HTTP content-coding envelope has a distinct budget from audio file bytes.
const HTTP_ENTITY_READ_LIMIT: u64 = 32 * 1024 * 1024 + 1;

#[derive(Debug)]
struct LookupSlots { capacity: usize, used: AtomicUsize }
struct LookupLease(Arc<LookupSlots>);
impl Drop for LookupLease {
    fn drop(&mut self) { self.0.used.fetch_sub(1, Ordering::Release); }
}
impl LookupSlots {
    fn acquire(self: &Arc<Self>) -> io::Result<LookupLease> {
        self.used.fetch_update(Ordering::AcqRel, Ordering::Acquire,
            |used| (used < self.capacity).then_some(used + 1))
            .map(|_| LookupLease(self.clone()))
            .map_err(|_| io::Error::new(io::ErrorKind::WouldBlock, "audio name lookup capacity is full"))
    }
}

#[derive(Debug)]
struct PublicResolver { slots: Arc<LookupSlots>, budget: Arc<Budget> }
impl Resolver for PublicResolver {
    fn resolve(&self, uri: &ureq::http::Uri, _: &ureq::config::Config, timeout: NextTimeout)
        -> std::result::Result<ResolvedSocketAddrs, ureq::Error> {
        // Validate both at the frontend URL boundary and at the socket source.
        self.budget.check()?;
        normalized_url(&uri.to_string()).map_err(|_| ureq::Error::BadUri("audio requires public HTTPS".into()))?;
        let address = DefaultResolver::host_and_port(uri.scheme().unwrap(), uri.authority().unwrap())
            .ok_or(ureq::Error::HostNotFound)?;
        let lease = self.slots.acquire().map_err(ureq::Error::Io)?;
        let deadline = self.budget.deadline.min(Instant::now() + STEP_TIMEOUT.min(*timeout.after));
        let (sender, receiver) = mpsc::sync_channel(1);
        std::thread::Builder::new().name("audio-dns".into()).spawn(move || {
            // An expired HTTP waiter does not release this reservation. An OS
            // lookup can finish late; only its own final use releases the slot.
            let _lease = lease;
            let result = address.to_socket_addrs().and_then(|addresses| {
                let mut checked = Vec::new();
                for address in addresses {
                    if address.port() != 443 || !public_address(address.ip()) {
                        return Err(io::Error::new(io::ErrorKind::PermissionDenied, "nonpublic audio address"));
                    }
                    if checked.len() < 16 { checked.push(address); }
                }
                if checked.is_empty() { return Err(io::Error::new(io::ErrorKind::NotFound, "no public audio address")); }
                Ok(checked)
            });
            let _ = sender.send(result);
        }).map_err(ureq::Error::Io)?;
        let addresses = wait_for_addresses(receiver, &self.budget, deadline, timeout.reason)?;
        let mut result = self.empty();
        for address in addresses { result.push(address); }
        // The connector consumes exactly this checked list. Keep the hostname
        // in the URI for normal certificate/SNI verification; never resolve it a
        // second time in a separate connection path.
        Ok(result)
    }
}

fn wait_for_addresses(receiver: mpsc::Receiver<io::Result<Vec<SocketAddr>>>, budget: &Budget,
    deadline: Instant, reason: ureq::Timeout) -> std::result::Result<Vec<SocketAddr>, ureq::Error> {
    loop {
        budget.check()?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() { return Err(ureq::Error::Timeout(reason)); }
        match receiver.recv_timeout(remaining.min(Duration::from_millis(50))) {
            Ok(result) => { budget.check()?; return result.map_err(ureq::Error::Io); }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err(ureq::Error::HostNotFound),
        }
    }
}

fn public_address(address: IpAddr) -> bool {
    // https://www.iana.org/assignments/iana-ipv4-special-registry/
    // https://www.iana.org/assignments/iana-ipv6-special-registry/
    match address {
        IpAddr::V4(address) => {
            let [a, b, c, _] = address.octets();
            // Conservative exclusions from the IANA special-purpose registry;
            // protocol anycast exceptions inside excluded blocks stay excluded.
            !(a == 0 || a == 10 || a == 127 || a >= 224
                || a == 100 && (64..=127).contains(&b)
                || a == 169 && b == 254 || a == 172 && (16..=31).contains(&b)
                || a == 192 && (b == 168 || b == 0 && (c == 0 || c == 2) || b == 88 && c == 99)
                || a == 198 && (b == 18 || b == 19 || b == 51 && c == 100)
                || a == 203 && b == 0 && c == 113)
        }
        IpAddr::V6(address) => {
            if let Some(mapped) = address.to_ipv4_mapped() { return public_address(IpAddr::V4(mapped)); }
            let bytes = address.octets();
            // Accept ordinary global unicast, excluding special assignments,
            // documentation and transition ranges. NAT64 and scoped addresses
            // are outside this deliberately restricted public input subset.
            bytes[0] & 0xe0 == 0x20
                && !(bytes[..2] == [0x20, 0x01] && bytes[2] < 2
                    || bytes[..4] == [0x20, 0x01, 0x0d, 0xb8]
                    || bytes[..2] == [0x20, 0x02]
                    || bytes[..2] == [0x3f, 0xff] && bytes[2] & 0xf0 == 0)
        }
    }
}

fn normalized_url(input: &str) -> Result<url::Url> {
    if input.len() > 4096 { return Err(invalid("audio HTTPS URL exceeds its length limit")); }
    let url = url::Url::parse(input).map_err(|_| invalid("invalid audio HTTPS URL"))?;
    if url.scheme() != "https" || url.port_or_known_default() != Some(443) || url.host_str().is_none()
        || !url.username().is_empty() || url.password().is_some() || url.fragment().is_some() {
        return Err(invalid("audio URL requires HTTPS port443 without credentials or fragment"));
    }
    match url.host() {
        Some(url::Host::Ipv4(address)) if !public_address(IpAddr::V4(address)) => return Err(invalid("audio URL is not public")),
        Some(url::Host::Ipv6(address)) if !public_address(IpAddr::V6(address)) => return Err(invalid("audio URL is not public")),
        _ => {}
    }
    Ok(url)
}

pub(super) struct Fetcher { slots: Arc<LookupSlots> }
impl Fetcher {
    pub(super) fn new(workers: usize) -> Self {
        Self { slots: Arc::new(LookupSlots { capacity: workers, used: AtomicUsize::new(0) }) }
    }

    pub(super) fn fetch(&self, input: &str, maximum: usize, cancelled: Arc<AtomicBool>) -> Result<Vec<u8>> {
        let deadline = Instant::now() + FETCH_TIMEOUT;
        let budget = Budget::new(deadline, cancelled.clone());
        let mut url = normalized_url(input)?;
        for redirect in 0..=REDIRECTS {
            if cancelled.load(Ordering::Acquire) { return Err(invalid("audio fetch cancelled")); }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() { return Err(invalid("audio fetch deadline exceeded")); }
            let config = ureq::Agent::config_builder().proxy(None).https_only(true)
                .max_redirects(0).max_redirects_will_error(false).http_status_as_error(false)
                .max_idle_connections(0).input_buffer_size(16384).timeout_global(Some(remaining))
                .timeout_resolve(Some(STEP_TIMEOUT)).timeout_connect(Some(STEP_TIMEOUT))
                .timeout_send_request(Some(STEP_TIMEOUT)).timeout_recv_response(Some(STEP_TIMEOUT))
                .timeout_recv_body(Some(remaining)).build();
            // Count cumulative ciphertext before TLS, including handshake and
            // HTTP framing overhead that a dechunked-body limit cannot see.
            // One fixed TCP input buffer may be received before a newly
            // exceeded byte budget is detected.
            let connector = ().chain(TcpConnector::default()).chain(GuardConnector(budget.clone()))
                .chain(RustlsConnector::default());
            let agent = ureq::Agent::with_parts(config, connector,
                PublicResolver { slots: self.slots.clone(), budget: budget.clone() });
            let mut response = agent.get(url.as_str()).header("Accept-Encoding", "identity").call()
                .map_err(|error| crate::Error::frontend(crate::FrontendError::Input(
                    crate::Failure::caused_by("audio HTTPS fetch failed", error))))?;
            if cancelled.load(Ordering::Acquire) { return Err(invalid("audio fetch cancelled")); }
            let status = response.status().as_u16();
            if [301, 302, 303, 307, 308].contains(&status) {
                if redirect == REDIRECTS { return Err(invalid("audio HTTPS redirect limit exceeded")); }
                let location = response.headers().get("location").and_then(|value| value.to_str().ok())
                    .ok_or_else(|| invalid("audio HTTPS redirect has no valid location"))?;
                url = normalized_url(url.join(location).map_err(|_| invalid("invalid audio HTTPS redirect"))?.as_str())?;
                continue;
            }
            if status != 200 { return Err(invalid("audio HTTPS source did not return status200")); }
            if response.headers().get("content-encoding").is_some_and(|value| value != "identity") {
                return Err(invalid("unsupported audio HTTPS content encoding"));
            }
            if response.headers().get("content-length").and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok()).is_some_and(|length| length > maximum as u64) {
                return Err(invalid("encoded audio exceeds its file limit"));
            }
            // ureq's gzip feature can be unified by another binary dependency.
            // It strips the response coding/length headers, so do not claim
            // that requesting identity proves identity delivery. The configured
            // reader bounds unchunked HTTP entity bytes before decompression;
            // read_body separately bounds the resulting audio file bytes.
            return read_body(response.body_mut().with_config().limit(HTTP_ENTITY_READ_LIMIT).reader(),
                maximum, deadline, &cancelled);
        }
        unreachable!()
    }
}

fn read_body(mut reader: impl Read, maximum: usize, deadline: Instant, cancelled: &AtomicBool) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    let mut scratch = [0u8; 16384];
    loop {
        if cancelled.load(Ordering::Acquire) { return Err(invalid("audio fetch cancelled")); }
        if Instant::now() >= deadline { return Err(invalid("audio fetch deadline exceeded")); }
        let requested = scratch.len().min((maximum - result.len()).saturating_add(1));
        let count = reader.read(&mut scratch[..requested]).map_err(|_| invalid("audio HTTPS body read failed"))?;
        if cancelled.load(Ordering::Acquire) { return Err(invalid("audio fetch cancelled")); }
        if Instant::now() >= deadline { return Err(invalid("audio fetch deadline exceeded")); }
        if count == 0 {
            if result.is_empty() { return Err(invalid("audio HTTPS response is empty")); }
            return Ok(result);
        }
        if count > maximum - result.len() { return Err(invalid("encoded audio exceeds its file limit")); }
        result.extend_from_slice(&scratch[..count]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_normalize_before_private_and_special_address_checks() {
        for url in ["http://8.8.8.8/audio", "https://8.8.8.8:8443/audio", "https://user:secret@example.com/audio",
            "https://example.com/audio#fragment", "file:///audio.wav", "https://127.0.0.1/audio",
            "https://2130706433/audio", "https://0177.0.0.1/audio", "https://[::1]/audio",
            "https://[::ffff:127.0.0.1]/audio", "https://10.0.0.1/audio", "https://192.0.2.1/audio",
            "https://[64:ff9b::7f00:1]/audio", "https://[2002:7f00:1::]/audio"] {
            assert!(normalized_url(url).is_err(), "{url}");
        }
        for url in ["https://example.com/audio", "https://8.8.8.8/audio", "https://[2606:4700:4700::1111]/audio"] {
            assert!(normalized_url(url).is_ok(), "{url}");
        }
        for address in ["0.1.2.3", "100.64.0.1", "169.254.169.254", "172.31.255.255", "192.168.0.1",
            "192.88.99.2", "198.18.0.1", "198.51.100.1", "203.0.113.1", "224.0.0.1", "255.255.255.255",
            "::", "fc00::1", "fe80::1", "2001::1", "2001:db8::1", "3fff::1", "5f00::1", "ff02::1"] {
            assert!(!public_address(address.parse().unwrap()), "{address}");
        }
    }

    #[test]
    fn exact_file_byte_limit_allows_eof_and_rejects_one_extra_byte() {
        let maximum = 25 * 1024 * 1024;
        let deadline = Instant::now() + Duration::from_secs(30);
        let cancelled = AtomicBool::new(false);
        assert_eq!(read_body(io::Cursor::new(vec![1; maximum]), maximum, deadline, &cancelled).unwrap().len(), maximum);
        assert!(read_body(io::Cursor::new(vec![1; maximum + 1]), maximum, deadline, &cancelled).is_err());
        assert!(read_body(io::Cursor::new(Vec::<u8>::new()), maximum, deadline, &cancelled).is_err());
    }

    #[test]
    fn a_final_eof_cannot_hide_cancellation_or_a_crossed_deadline() {
        struct CancelAtEof { cancelled: Arc<AtomicBool>, sent: bool }
        impl Read for CancelAtEof {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                if !self.sent { self.sent = true; buffer[0] = b'a'; return Ok(1); }
                self.cancelled.store(true, Ordering::Release);
                Ok(0)
            }
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        assert!(read_body(CancelAtEof { cancelled: cancelled.clone(), sent: false }, 10,
            Instant::now() + Duration::from_secs(1), &cancelled).is_err());
        assert!(read_body(io::Cursor::new(b"audio"), 10, Instant::now(), &AtomicBool::new(false)).is_err());
    }

    #[test]
    fn timed_out_lookup_reservations_remain_owned_until_the_thread_really_ends() {
        let slots = Arc::new(LookupSlots { capacity: 1, used: AtomicUsize::new(0) });
        let lease = slots.acquire().unwrap();
        let (release, wait) = mpsc::channel();
        let (sender, receiver) = mpsc::sync_channel(1);
        let thread = std::thread::spawn(move || {
            let _lease = lease;
            let _ = wait.recv();
            sender.send(Ok(vec!["8.8.8.8:443".parse().unwrap()])).is_err()
        });
        let budget = Budget::new(Instant::now() + Duration::from_secs(1), Arc::new(AtomicBool::new(false)));
        assert!(wait_for_addresses(receiver, &budget, Instant::now(), ureq::Timeout::Resolve).is_err());
        assert_eq!(slots.used.load(Ordering::Acquire), 1);
        assert!(slots.acquire().is_err());
        release.send(()).unwrap();
        assert!(thread.join().unwrap(), "late DNS result was not discarded");
        assert_eq!(slots.used.load(Ordering::Acquire), 0);
        drop(slots.acquire().unwrap());
        assert_eq!(slots.used.load(Ordering::Acquire), 0);
    }

    #[test]
    fn actual_localhost_resolution_cannot_publish_a_connection_candidate() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let resolver = PublicResolver { slots: Arc::new(LookupSlots { capacity: 1, used: AtomicUsize::new(0) }),
            budget: Budget::new(Instant::now() + Duration::from_secs(2), cancelled) };
        let result = resolver.resolve(&"https://localhost/audio".parse().unwrap(), &ureq::config::Config::default(),
            NextTimeout { after: Duration::from_secs(2).into(), reason: ureq::Timeout::Resolve });
        assert!(result.is_err());
    }
}
