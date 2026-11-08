/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Real multipart/FFmpeg/frontend and host queue ownership; no GPU model claim.

use super::*;
use axum::{Extension, extract::State};
use crate::{frontend::{AudioSettings, FrontendSettings}, model::contract::{
    AudioCapacity, AudioSamplePadding, AudioTokenLayout, ControlMarkerSyntax, StateView,
    TranscriptionFormat, TranscriptionLanguage,
}};

fn fixture() -> Fixture {
    fixture_with_template(b"{% mark %}{{ messages[0].content }}")
}

fn fixture_with_template(template: &[u8]) -> Fixture {
    let settings = FrontendSettings {
        prompt_encoder: Default::default(), image_placeholder_token_id: None,
        tokenizer: Default::default(), eos_token_ids: vec![], template_encoding: Default::default(),
        image: None,
        audio: Some(AudioSettings { sample_rate: 16000, mel_bins: 128, fft_size: 400, hop_samples: 160,
            max_file_bytes: 25000, max_samples: 20000, decode_timeout_millis: 10000,
            service_chunk_samples: 3200, silence_search_samples: 1600, silence_window_samples: 160,
            sample_padding: AudioSamplePadding::HopMultiple, numerics: Default::default() }),
    };
    let mut fixture = Fixture::with_template_settings(template,
        &["<eos>", "<audio>", "<feature>", "</audio>"], &[], Some(&settings));
    let tokenizer = fixture.frontend.tokenizer();
    fixture.config.contract.frontend.audio_tokens = Some(AudioTokenLayout {
        start_token: tokenizer.token_to_id("<audio>").unwrap(),
        feature_token: tokenizer.token_to_id("<feature>").unwrap(),
        end_token: tokenizer.token_to_id("</audio>").unwrap(),
    });
    let view = || StateView { buffer: String::new(), offset: 0, stride: 1, nbytes: 1 };
    fixture.config.contract.audio = Some(AudioCapacity { feature_slots: 4,
        features_per_request: 104, feature_width: 2048, frame_capacities: vec![100, 800],
        mel_bins: 128, frames_per_chunk: 100, features_per_chunk: 13, subsampling_factor: 8,
        attention_window_frames: 800, features: view(), generations: view(), counts: view() });
    fixture.config.contract.frontend.transcription = Some(TranscriptionFormat {
        context_prefix: "b".into(), context_suffix: "c".into(), audio_prefix: "a".into(),
        audio_suffix: "d".into(), assistant_prefix: "x".into(), language_prefix: "b".into(),
        text_delimiter: "<asr_text>".into(), default_separator: " ".into(),
        languages: vec![TranscriptionLanguage { code: "en".into(), label: "English".into(), chunk_separator: " ".into() },
                        TranscriptionLanguage { code: "zh".into(), label: "Chinese".into(), chunk_separator: "".into() }],
        sanitization: ControlMarkerSyntax { prefix: "<|".into(), suffix: "|>".into(), excluded_body_character: "|".into() },
    });
    fixture
}

fn wav(samples: u32) -> Vec<u8> {
    let mut bytes = b"RIFF".to_vec();
    bytes.extend((36 + samples * 2).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(16000u32.to_le_bytes());
    bytes.extend(32000u32.to_le_bytes());
    bytes.extend(2u16.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend((samples * 2).to_le_bytes());
    bytes.resize(44 + samples as usize * 2, 0);
    bytes
}

fn request(fields: &[(&str, &str)], files: &[Vec<u8>]) -> axum::extract::Request {
    let mut body = Vec::new();
    for (name, text) in fields {
        body.extend(format!("--asr-boundary\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{text}\r\n").as_bytes());
    }
    for file in files {
        body.extend(b"--asr-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n");
        body.extend(file);
        body.extend(b"\r\n");
    }
    body.extend(b"--asr-boundary--\r\n");
    axum::http::Request::builder().method("POST").uri("/v1/audio/transcriptions")
        .header("content-type", "multipart/form-data; boundary=asr-boundary")
        .body(axum::body::Body::from(body)).unwrap()
}

async fn start(state: Arc<super::super::super::HttpState>, request: axum::extract::Request)
    -> tokio::task::JoinHandle<axum::response::Response> {
    let ingress = Arc::new(state.ingress_slots.clone().acquire_owned().await.unwrap());
    tokio::spawn(super::super::super::transcription::generate(State(state), Extension(ingress), request))
}

#[tokio::test]
async fn late_multipart_tail_and_body_cancellation_return_credits_without_queueing_children() {
    use axum::body::{Body, Bytes};
    use futures_util::stream;

    for cancel in [false, true] {
        let (sender, receiver) = mpsc::sync_channel(4);
        let mut state = http_state_for(fixture(), sender, None);
        let (parts, body) = request(&[], &[wav(400)]).into_parts();
        let envelope = axum::body::to_bytes(body, 65536).await.unwrap();
        Arc::get_mut(&mut state).unwrap().max_body_bytes = envelope.len();
        let (tail, input) = tokio::sync::mpsc::unbounded_channel::<std::result::Result<Bytes, std::io::Error>>();
        tail.send(Ok(envelope)).unwrap();
        let body = Body::from_stream(stream::unfold(input, |mut input| async {
            input.recv().await.map(|value| (value, input))
        }));
        let request = axum::extract::Request::from_parts(parts, body);
        let ingress = Arc::new(state.ingress_slots.clone().acquire_owned().await.unwrap());
        let mut response = Box::pin(super::super::super::transcription::generate(
            State(state.clone()), Extension(ingress), request));
        assert!(futures_util::poll!(response.as_mut()).is_pending());
        assert_eq!(state.ingress_slots.available_permits(), 3);
        assert_eq!(state.frontend_slots.available_permits(), 1);
        assert!(receiver.try_recv().is_err());
        if cancel {
            drop(response);
            assert!(tail.is_closed());
        } else {
            tail.send(Ok(Bytes::from_static(b" "))).unwrap();
            drop(tail);
            let response = response.await;
            assert_eq!(response.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
            let bytes = axum::body::to_bytes(response.into_body(), 65536).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["error"]["code"], "request_too_large");
            assert!(body.get("text").is_none());
        }
        assert!(receiver.try_recv().is_err());
        assert_eq!(state.model.capacity.available_permits(), 4);
        assert_eq!(state.model.queue_capacity.available_permits(), 4);
        assert_eq!(state.ingress_slots.available_permits(), 4);
        assert_eq!(state.frontend_slots.available_permits(), 1);
    }
}

async fn receive(receiver: &mpsc::Receiver<Vec<Incoming>>) -> Incoming {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match receiver.try_recv() {
                Ok(mut group) => { assert_eq!(group.len(), 1); return group.remove(0); }
                Err(mpsc::TryRecvError::Empty) => tokio::task::yield_now().await,
                Err(error) => panic!("transcription queue closed: {error}"),
            }
        }
    }).await.expect("child was not submitted")
}

fn finish(incoming: &Incoming, token: u32) {
    incoming.emit_token(token);
    incoming.emit(GenerationEvent::Finished { prompt_tokens: incoming.prompt.tokens.len(), generated_tokens: 1,
        reused_prompt_tokens: 0, reason: "stop", termination: crate::output::Termination::Eos, stop_sequence: None });
}

#[tokio::test]
async fn multipart_children_keep_seed_order_and_replenish_after_admission() {
    for language in ["en", "zh"] {
        let (sender, receiver) = mpsc::sync_channel(4);
        let state = http_state_for(fixture(), sender, None);
        let response = start(state.clone(), request(&[("language", language), ("seed", "19")], &[wav(6400)])).await;
        let first = receive(&receiver).await;
        // No second submission before the first child reaches native admission.
        assert!(receiver.try_recv().is_err());
        first.notify_admission(Ok(()));
        let second = receive(&receiver).await;
        second.notify_admission(Ok(()));
        let third = receive(&receiver).await;
        third.notify_admission(Ok(()));
        for child in [&first, &second, &third] {
            assert_eq!(child.prompt.sampling.seed, 19);
            assert_eq!(child.prompt.conditioned_audio.as_ref().unwrap().audio().feature_count(),
                       child.prompt.data.audio_token_indices.iter().filter(|&&index| index >= 0).count());
        }
        assert!(!response.is_finished());
        finish(&third, 2);
        finish(&first, 0);
        finish(&second, 1);
        drop((first, second, third));
        let response = response.await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(response.headers()["x-flare-generated-tokens"], "3");
        let bytes = axum::body::to_bytes(response.into_body(), 65536).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["text"], if language == "en" { "a b c" } else { "abc" });
        assert_eq!(body["usage"]["seconds"], 1);
        assert_eq!(state.model.capacity.available_permits(), 4);
        assert_eq!(state.model.queue_capacity.available_permits(), 4);
        assert_eq!(state.ingress_slots.available_permits(), 4);
    }
}

#[tokio::test]
async fn failed_transcription_cancels_siblings_and_never_returns_partial_text() {
    let (sender, receiver) = mpsc::sync_channel(4);
    let state = http_state_for(fixture(), sender, None);
    let response = start(state.clone(), request(&[], &[wav(6400)])).await;
    let first = receive(&receiver).await;
    first.notify_admission(Ok(()));
    let second = receive(&receiver).await;
    first.emit_token(0);
    second.notify_admission(Err(Error::admission(AdmissionError::Capacity, "child rejected")));
    let response = response.await.unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    assert!(first.cancelled.load(Ordering::Acquire));
    assert!(second.cancelled.load(Ordering::Acquire));
    let bytes = axum::body::to_bytes(response.into_body(), 65536).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(body.get("text").is_none());
    drop((first, second));
    assert_eq!(state.model.capacity.available_permits(), 4);
    assert_eq!(state.ingress_slots.available_permits(), 4);
}

#[tokio::test]
async fn disconnected_transcription_cancels_admitted_and_pending_children() {
    let (sender, receiver) = mpsc::sync_channel(4);
    let state = http_state_for(fixture(), sender, None);
    let response = start(state.clone(), request(&[], &[wav(6400)])).await;
    let first = receive(&receiver).await;
    first.notify_admission(Ok(()));
    let second = receive(&receiver).await;
    response.abort();
    assert!(response.await.unwrap_err().is_cancelled());
    assert!(first.cancelled.load(Ordering::Acquire));
    assert!(second.cancelled.load(Ordering::Acquire));
    drop((first, second));
    assert_eq!(state.model.capacity.available_permits(), 4);
    assert_eq!(state.ingress_slots.available_permits(), 4);
}

#[tokio::test]
async fn actual_tcp_disconnect_cancels_admitted_and_pending_transcription_children() {
    use std::io::Write;
    struct ListenerTask(Option<tokio::task::JoinHandle<()>>);
    impl Drop for ListenerTask {
        fn drop(&mut self) {
            if let Some(task) = &self.0 { task.abort(); }
        }
    }
    let (sender, receiver) = mpsc::sync_channel(4);
    let state = http_state_for(fixture(), sender, None);
    let router = super::super::super::router(state.clone(), &state.served_names, state.max_body_bytes);
    let tcp = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = tcp.local_addr().unwrap();
    let listener = super::super::super::listener::HttpListener::new(tcp);
    let mut server = ListenerTask(Some(tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    })));
    let mut client = std::net::TcpStream::connect(address).unwrap();
    client.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
    let body = axum::body::to_bytes(request(&[], &[wav(6400)]).into_body(), 65536).await.unwrap();
    write!(client, "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\nContent-Type: multipart/form-data; boundary=asr-boundary\r\nContent-Length: {}\r\n\r\n", body.len()).unwrap();
    client.write_all(&body).unwrap();
    let first = receive(&receiver).await;
    first.notify_admission(Ok(()));
    let second = receive(&receiver).await;
    drop(client);
    tokio::time::timeout(Duration::from_secs(3), async {
        while !first.cancelled.load(Ordering::Acquire) || !second.cancelled.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    }).await.expect("TCP disconnect did not cancel the transcription parent");
    drop((first, second));
    tokio::time::timeout(Duration::from_secs(3), async {
        while state.ingress_slots.available_permits() != 4 || state.frontend_slots.available_permits() != 1 {
            tokio::task::yield_now().await;
        }
    }).await.expect("disconnected TCP request retained host frontend owners");
    assert_eq!(state.model.capacity.available_permits(), 4);
    assert_eq!(state.model.queue_capacity.available_permits(), 4);
    assert!(receiver.try_recv().is_err());
    server.0.as_ref().unwrap().abort();
    assert!(server.0.take().unwrap().await.unwrap_err().is_cancelled());
    assert!(std::net::TcpStream::connect_timeout(&address, Duration::from_secs(1)).is_err());
}

#[tokio::test]
async fn malformed_transcription_sampling_is_invalid_without_native_submission() {
    for fields in [[("top_k", "1.5")], [("seed", "1.5")]] {
        let (sender, receiver) = mpsc::sync_channel(4);
        let state = http_state_for(fixture(), sender, None);
        let response = start(state.clone(), request(&fields, &[wav(6400)])).await.await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST, "{fields:?}");
        let body = axum::body::to_bytes(response.into_body(), 65536).await.unwrap();
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert!(receiver.try_recv().is_err());
        assert_eq!(state.model.capacity.available_permits(), 4);
        assert_eq!(state.model.queue_capacity.available_permits(), 4);
        assert_eq!(state.frontend_slots.available_permits(), 1);
        assert_eq!(state.ingress_slots.available_permits(), 4);
    }
}

#[tokio::test]
async fn signed_transcription_seed_is_resolved_once_and_shared_by_children() {
    use crate::model::contract::RequestSeedSyntax;
    for (field, expected) in [(Some("-1"), None), (None, None),
        (Some("-2"), Some(u64::MAX - 1)), (Some("-9223372036854775808"), Some(1u64 << 63)),
        (Some("1.0"), Some(1))] {
        let mut fixture = fixture();
        fixture.config.contract.frontend.request_seed_syntax = Some(RequestSeedSyntax::Signed64Optional);
        let (sender, receiver) = mpsc::sync_channel(4);
        let state = http_state_for(fixture, sender, None);
        let fields = field.map(|value| vec![("seed", value)]).unwrap_or_default();
        let response = start(state.clone(), request(&fields, &[wav(6400)])).await;
        let first = receive(&receiver).await;
        first.notify_admission(Ok(()));
        let second = receive(&receiver).await;
        second.notify_admission(Ok(()));
        let third = receive(&receiver).await;
        third.notify_admission(Ok(()));
        let key = first.prompt.sampling.seed;
        assert_eq!(second.prompt.sampling.seed, key);
        assert_eq!(third.prompt.sampling.seed, key);
        if let Some(expected) = expected { assert_eq!(key, expected); }
        for child in [&first, &second, &third] { finish(child, 0); }
        drop((first, second, third));
        assert_eq!(response.await.unwrap().status(), axum::http::StatusCode::OK);
        assert_eq!(state.model.capacity.available_permits(), 4);
        assert_eq!(state.model.queue_capacity.available_permits(), 4);
    }
}

#[tokio::test]
async fn signed_transcription_seed_rejects_literal_null_and_out_of_range_form_values() {
    use crate::model::contract::RequestSeedSyntax;
    for value in ["null", "9223372036854775808", "-9223372036854775809", "1.5", "true"] {
        let mut fixture = fixture();
        fixture.config.contract.frontend.request_seed_syntax = Some(RequestSeedSyntax::Signed64Optional);
        let (sender, receiver) = mpsc::sync_channel(4);
        let state = http_state_for(fixture, sender, None);
        let response = start(state.clone(), request(&[("seed", value)], &[wav(6400)])).await.await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST, "{value}");
        assert!(receiver.try_recv().is_err());
        assert_eq!(state.ingress_slots.available_permits(), 4);
    }
}

#[tokio::test]
async fn signed_chat_seed_normalizes_before_generic_protocol_decoding() {
    use base64::Engine;
    use crate::model::contract::RequestSeedSyntax;
    for (seed, expected) in [(serde_json::json!(-2), Some(u64::MAX - 1)),
        (serde_json::json!(i64::MIN), Some(1u64 << 63)), (serde_json::json!("1.0"), Some(1)),
        (serde_json::json!(-1), None), (serde_json::Value::Null, None)] {
        let mut fixture = fixture_with_template(b"{% mark %}<audio><feature></audio>");
        fixture.config.contract.frontend.request_seed_syntax = Some(RequestSeedSyntax::Signed64Optional);
        let (sender, receiver) = mpsc::sync_channel(4);
        let state = http_state_for(fixture, sender, None);
        let model = state.model.config().model_id.clone();
        let body = serde_json::json!({"model":model,"seed":seed,"temperature":0,"max_tokens":2,
            "messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{
                "format":"wav","data":base64::engine::general_purpose::STANDARD.encode(wav(6400))}}]}]});
        let request = axum::http::Request::builder().method("POST").uri("/v1/chat/completions")
            .header("content-type", "application/json").body(axum::body::Body::from(body.to_string())).unwrap();
        let ingress = Arc::new(state.ingress_slots.clone().acquire_owned().await.unwrap());
        let mut response = tokio::spawn(super::super::super::http::generate(State(state.clone()), Extension(ingress), request));
        let incoming = tokio::select! {
            incoming = receive(&receiver) => incoming,
            result = &mut response => {
                let response = result.unwrap();
                let status = response.status();
                let body = axum::body::to_bytes(response.into_body(), 65536).await.unwrap();
                panic!("audio chat failed before native admission: {status} {}", String::from_utf8_lossy(&body));
            }
        };
        if let Some(expected) = expected { assert_eq!(incoming.prompt.sampling.seed, expected); }
        incoming.notify_admission(Ok(()));
        finish(&incoming, 0);
        drop(incoming);
        assert_eq!(response.await.unwrap().status(), axum::http::StatusCode::OK);
        assert_eq!(state.model.capacity.available_permits(), 4);
        assert_eq!(state.model.queue_capacity.available_permits(), 4);
    }
}

#[tokio::test]
async fn multipart_rejects_unsupported_duplicates_size_and_decode_failures_before_queueing() {
    let cases = [
        (vec![("stream", "true")], vec![wav(6400)], 400),
        (vec![("timestamps", "true")], vec![wav(6400)], 400),
        (vec![("language", "en"), ("language", "zh")], vec![wav(6400)], 400),
        (vec![("seed", "\"19\"")], vec![wav(6400)], 400),
        (vec![("n", "2")], vec![wav(6400)], 400),
        (vec![("response_format", "verbose_json")], vec![wav(6400)], 400),
        (vec![("language", "invalid")], vec![wav(6400)], 400),
        (vec![("model", "another-model")], vec![wav(6400)], 404),
        (vec![], vec![wav(6400), wav(6400)], 400),
        (vec![], vec![], 400),
        (vec![], vec![vec![0; 25001]], 413),
        (vec![], vec![b"broken-audio".to_vec()], 400),
    ];
    for (fields, files, status) in cases {
        let (sender, receiver) = mpsc::sync_channel(4);
        let state = http_state_for(fixture(), sender, None);
        let response = start(state.clone(), request(&fields, &files)).await.await.unwrap();
        assert_eq!(response.status().as_u16(), status, "{fields:?}");
        assert!(receiver.try_recv().is_err());
        assert_eq!(state.model.capacity.available_permits(), 4);
        assert_eq!(state.ingress_slots.available_permits(), 4);
    }
}

#[tokio::test]
async fn small_parent_replenishes_after_native_retirement_without_releasing_draining_credits() {
    let (sender, receiver) = mpsc::sync_channel(1);
    let mut state = http_state_for(fixture(), sender, None);
    let model = Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().model).unwrap();
    // One active plus one queued native request, matching A=1/Q=1 host budget.
    model.capacity = Arc::new(tokio::sync::Semaphore::new(2));
    model.queue_capacity = Arc::new(tokio::sync::Semaphore::new(1));
    let worker_fixture = Fixture::new();
    let mut worker = worker_fixture.worker(mpsc::channel().1);
    let response = start(state.clone(), request(&[("language", "en")], &[wav(6400)])).await;
    worker.enqueue(vec![receive(&receiver).await]);
    let first = worker.pending.pop_front().unwrap();
    first.notify_admission(Ok(()));
    worker.enqueue(vec![receive(&receiver).await]);
    let second = worker.pending.pop_front().unwrap();
    second.notify_admission(Ok(()));
    finish(&first, 0);
    // A finished response reader cannot release the capacity still owned by
    // the native request. Its final retirement is a separate lifecycle event.
    assert_eq!(state.model.capacity.available_permits(), 0);
    assert!(receiver.try_recv().is_err());
    drop(first);
    worker.enqueue(vec![receive(&receiver).await]);
    let third = worker.pending.pop_front().unwrap();
    third.notify_admission(Ok(()));
    finish(&third, 2);
    finish(&second, 1);
    drop((second, third));
    let response = response.await.unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 65536).await.unwrap();
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["text"], "a b c");
    assert_eq!(state.model.capacity.available_permits(), 2);
    assert_eq!(state.model.queue_capacity.available_permits(), 1);
}

#[tokio::test]
async fn sibling_failure_cancels_a_parent_waiting_for_native_capacity() {
    let (sender, receiver) = mpsc::sync_channel(1);
    let mut state = http_state_for(fixture(), sender, None);
    let model = Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().model).unwrap();
    model.capacity = Arc::new(tokio::sync::Semaphore::new(2));
    model.queue_capacity = Arc::new(tokio::sync::Semaphore::new(1));
    let worker_fixture = Fixture::new();
    let mut worker = worker_fixture.worker(mpsc::channel().1);
    let response = start(state.clone(), request(&[], &[wav(6400)])).await;
    worker.enqueue(vec![receive(&receiver).await]);
    let first = worker.pending.pop_front().unwrap();
    first.notify_admission(Ok(()));
    worker.enqueue(vec![receive(&receiver).await]);
    let second = worker.pending.pop_front().unwrap();
    second.notify_admission(Ok(()));
    first.fail(Error::execution(ExecutionError::Worker, "child failed while sibling waits"));
    let response = response.await.unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    assert!(second.cancelled.load(Ordering::Acquire));
    assert_eq!(state.model.capacity.available_permits(), 0);
    assert!(receiver.try_recv().is_err());
    drop((first, second));
    assert_eq!(state.model.capacity.available_permits(), 2);
    assert_eq!(state.model.queue_capacity.available_permits(), 1);
    // A cancelled spawn_blocking preparation retains its ingress/worker owners
    // until it observes cancellation and exits. Native host-credit release is
    // separate; an already returned parent must not free CPU owners early.
    tokio::time::timeout(Duration::from_secs(10), async {
        while state.ingress_slots.available_permits() != 4 || state.frontend_slots.available_permits() != 1 {
            tokio::task::yield_now().await;
        }
    }).await.expect("cancelled CPU preparation did not release its final owners");
    assert!(receiver.try_recv().is_err());
    assert_eq!(state.ingress_slots.available_permits(), 4);
}
