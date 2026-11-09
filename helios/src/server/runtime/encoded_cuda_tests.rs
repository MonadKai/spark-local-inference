/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Actual model worker/upload cancellation. HTTP disconnect propagation is a
//! separate gate. Test inputs and publisher authorization are explicit.

use super::*;
use crate::{cuda::{PinnedHost, Stream}, runtime_package::{KvCacheLimit, LoadOptions,
    NativeModel, Publisher, VerifiedPackage}};
use std::{panic::{AssertUnwindSafe, catch_unwind}, sync::atomic::AtomicU32, time::Instant};

struct Release(usize);
impl Drop for Release {
    fn drop(&mut self) {
        // The caller keeps the mapped allocation through join and GPU drain.
        unsafe { AtomicU32::from_ptr(self.0 as *mut u32).store(1, Ordering::Release); }
    }
}

fn prompt(fixture: &Fixture, bytes: &[u8]) -> Prompt {
    let config = &fixture.config;
    let cancelled = AtomicBool::new(false);
    let settings = fixture.frontend.audio_settings().unwrap().clone();
    let processor = crate::frontend::audio::AudioProcessor::new(settings, 1).unwrap();
    let pcm = processor.decode(bytes, &cancelled).unwrap();
    let (audio, identity) = processor.prepare_pcm(&pcm, config.contract.audio.as_ref().unwrap(), &cancelled).unwrap();
    let frontend = &config.contract.frontend;
    let renderer = crate::frontend::transcription::Renderer::new(&fixture.frontend,
        frontend.transcription.as_ref().unwrap(), frontend.audio_tokens.as_ref().unwrap()).unwrap();
    let rendered = renderer.render(&fixture.frontend, "", Some("en"), &cancelled).unwrap();
    let (prepared, input) = crate::model::prepare_audio_input(config, &rendered, audio, identity).unwrap();
    let mut result = Fixture::prompt_for(prepared.tokens, 1024, prepared.data);
    result.conditioned_audio = Some(input);
    result.conditioning = prepared.conditioning;
    result.sampling = config.sampling_defaults.clone();
    result.sampling.seed = 0;
    result
}

pub(super) fn incoming(fixture: &Fixture, prompt: Prompt, scope: u8) -> (Incoming, Generation) {
    let (mut incoming, generation) = fixture.incoming_for(Vec::new(), 1024, Default::default());
    incoming.initial_prompt_tokens = prompt.tokens.len();
    incoming.prompt = prompt;
    incoming.prefix_scope = Some([scope; 24]);
    (incoming, generation)
}

#[test]
fn native_audio_fixture_freezes_expanded_prompt_before_admission() {
    let fixture = Fixture::new();
    let prepared = Fixture::prompt_for(vec![7; 19], 4, PromptData {
        positions: (0..19).map(|position| [position; 3]).collect(),
        vision_token_indices: vec![-1; 19], ..Default::default()
    });
    let (request, _generation) = incoming(&fixture, prepared, 1);
    assert_eq!(request.initial_prompt_tokens, 19);
    let (sender, receiver) = mpsc::sync_channel(1);
    let mut worker = fixture.worker(receiver);
    sender.send(vec![request]).unwrap();
    assert!(matches!(worker.admit().unwrap(), ControlFlow::Continue(_)));
    let handle = worker.active.iter().flatten().next().unwrap().request;
    assert_eq!(worker.scheduler.request(handle).unwrap().initial_prompt_tokens(), 19);
    worker.scheduler.cancel(handle).unwrap();
}

fn tick(worker: &mut Worker<'_>, executor: &mut RankExecutor) {
    assert!(matches!(worker.admit().unwrap(), ControlFlow::Continue(_)));
    if let Some(completion) = executor.poll().unwrap() {
        worker.complete(executor.manifest().graphs[completion.graph_index].kind, completion).unwrap();
    }
    worker.publish().unwrap();
    worker.service_encoded_inputs(executor).unwrap();
    if let Some(plan) = worker.plan(executor.outstanding()).unwrap().0 {
        worker.submit(executor, &plan).unwrap();
    }
    std::thread::yield_now();
}

fn finish(worker: &mut Worker<'_>, executor: &mut RankExecutor, generation: &mut Generation) -> Vec<u32> {
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut tokens = Vec::new();
    let mut finished = false;
    loop {
        assert!(Instant::now() < deadline, "actual worker did not complete");
        tick(worker, executor);
        while let Ok(event) = generation.events.try_recv() {
            match event {
                GenerationEvent::Token(token) => tokens.push(token),
                GenerationEvent::Finished { reason, .. } => { assert_eq!(reason, "stop"); finished = true; }
                GenerationEvent::Error(error) | GenerationEvent::AdmissionRejected(error) => panic!("{error}"),
                _ => {}
            }
        }
        if finished && worker.active.iter().all(Option::is_none) && executor.outstanding() == 0 {
            assert!(!tokens.is_empty());
            return tokens;
        }
    }
}

#[test]
#[ignore = "requires exclusive CUDA, FLARE_PROMPT_TEST_PACKAGE, FLARE_PROMPT_TEST_PUBLISHERS and FLARE_ENCODED_WORKER_TEST_AUDIO"]
fn cancellation_during_actual_feature_upload_cannot_publish_or_release_its_owner() {
    let package = std::env::var("FLARE_PROMPT_TEST_PACKAGE").unwrap();
    let publishers = std::env::var("FLARE_PROMPT_TEST_PUBLISHERS").unwrap();
    let bytes = std::fs::read(std::env::var("FLARE_ENCODED_WORKER_TEST_AUDIO").unwrap()).unwrap();
    let publishers: Vec<Publisher> = serde_json::from_slice(&std::fs::read(publishers).unwrap()).unwrap();
    let verified = VerifiedPackage::open(std::path::Path::new(&package), &publishers, None).unwrap();
    let namespace = verified.publisher().namespace.clone();
    let authorized = verified.authorize(&namespace, true).unwrap();
    let options = LoadOptions { max_concurrency: Some(1), kv_cache_limit: KvCacheLimit::Concurrency,
        embedding_table_storage: Some(crate::engram::Storage::Ram),
        prepared_input_cache: Some(crate::offload::PreparedInputCacheMode::Encoded),
        prefix_cache: crate::offload::PrefixCacheOffload { memory_bytes: Some(64 << 20),
            disk_limit: Some(0), ..Default::default() }, ..Default::default() };
    let owner = unsafe { NativeModel::load(authorized, &[0], &options,
        &crate::startup::StartupProgress::silent()).unwrap() };
    let manifest = owner.manifests().unwrap().remove(0);
    let bootstrap = crate::server::Bootstrap::load(&owner.server_config().to_string(), &manifest).unwrap();
    assert!(bootstrap.model.speculative_widths.is_empty());
    assert!(bootstrap.model.contract.audio.is_some());
    let store = manifest.offload.as_ref().unwrap().inputs().unwrap();
    let fixture = Fixture { config: bootstrap.model, frontend: bootstrap.frontend,
        stop: AtomicBool::new(false), metrics: Arc::new(super::super::super::metrics::Metrics::new(0, 0, 0, 16)) };
    let context = manifest.context;
    let stream = Stream::open(context, 0).unwrap();
    let mut flag: PinnedHost = stream.mapped_host_buffer(4).unwrap();
    flag.bytes_mut().copy_from_slice(&0u32.to_ne_bytes());
    let mut executor = unsafe { RankExecutor::new(manifest, None).unwrap() };
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        executor.bind_feature_storage(fixture.config.contract.audio.as_ref().unwrap()).unwrap();
        let mut scheduler = RequestScheduler::new(fixture.config.identity, fixture.config.limits,
            crate::model::scheduler_capabilities_with_manifest(&fixture.config, executor.manifest()).unwrap()).unwrap();
        scheduler.graph_durations(&executor.manifest().graphs).unwrap();
        scheduler.configure_policy(Default::default()).unwrap();
        if let Some(cache) = &executor.manifest().cache {
            if !cache.snapshot_pages.is_empty() { scheduler.share_snapshot_pages(&cache.snapshot_pages).unwrap(); }
        }
        scheduler.prefill_graphs(&executor.manifest().graphs.iter().filter(|graph| graph.kind == GraphKind::Prefill)
            .map(|graph| graph.token_capacity).collect::<Vec<_>>()).unwrap();
        let (sender, receiver) = mpsc::sync_channel(4);
        let mut worker = Worker::new(&fixture.config, &fixture.frontend, &fixture.stop,
            &fixture.metrics, scheduler, receiver, 4, super::super::super::PrefixCacheReuse::Shared);
        worker.inputs = Some(crate::model::Inputs::new(&fixture.config, executor.manifest()).unwrap());
        worker.encoded_inputs = Some(super::super::encoded::EncodedInputs::new(store));
        let (first, mut baseline_generation) = incoming(&fixture, prompt(&fixture, &bytes), 1);
        sender.send(vec![first]).unwrap();
        let baseline = finish(&mut worker, &mut executor, &mut baseline_generation);
        assert!(fixture.metrics.snapshot(false).encoded_input_writes > 0);

        let (second, cancelled_generation) = incoming(&fixture, prompt(&fixture, &bytes), 2);
        let cancellation = second.cancelled.clone();
        sender.send(vec![second]).unwrap();
        assert!(matches!(worker.admit().unwrap(), ControlFlow::Continue(_)));
        let handle = worker.active.iter().flatten().next().unwrap().request;
        let deadline = Instant::now() + Duration::from_secs(15);
        while !worker.encoded_inputs.as_ref().unwrap().ready(handle) {
            assert!(Instant::now() < deadline, "retained model features were not ready");
            worker.service_encoded_inputs(&mut executor).unwrap();
            std::thread::yield_now();
        }
        let plan = worker.plan(0).unwrap().0.unwrap();
        assert_eq!(plan.kind, GraphKind::Init);
        assert!(worker.submit(&mut executor, &plan).unwrap());
        loop {
            assert!(Instant::now() < deadline, "real Init did not complete");
            if let Some(completion) = executor.poll().unwrap() {
                assert_eq!(executor.manifest().graphs[completion.graph_index].kind, GraphKind::Init);
                worker.complete(GraphKind::Init, completion).unwrap();
                break;
            }
            std::thread::yield_now();
        }
        let before = worker.scheduler.conditioned_prerequisite(handle).unwrap();
        let arena = before.feature_slot();
        let hits = fixture.metrics.snapshot(false).encoded_input_hits;
        let (next, mut next_generation) = incoming(&fixture, prompt(&fixture, &bytes), 3);
        sender.send(vec![next]).unwrap();
        let release = Release(flag.address());
        let probe = unsafe { executor.hold_feature_copy(&flag).unwrap() };
        assert!(unsafe { probe.completed().unwrap() });
        let returned = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let returned = &returned;
            let admission = &mut next_generation.admission;
            let address = flag.address();
            scope.spawn(move || {
                let _release_on_all_exits = Release(address);
                let deadline = Instant::now() + Duration::from_secs(15);
                while unsafe { probe.completed().unwrap() } {
                    assert!(Instant::now() < deadline, "real feature-copy fence did not become pending");
                    std::thread::yield_now();
                }
                assert!(!returned.load(Ordering::Acquire), "upload returned before its GPU fence");
                assert!(matches!(admission.try_recv(), Err(oneshot::error::TryRecvError::Empty)));
                cancellation.store(true, Ordering::Release);
            });
            worker.service_encoded_inputs(&mut executor).unwrap();
            returned.store(true, Ordering::Release);
        });
        drop(release);
        let after = worker.scheduler.conditioned_prerequisite(handle).unwrap();
        assert!(before.same_as(&after), "cancelled upload published a changed owner, epoch or prerequisite");
        assert_eq!(fixture.metrics.snapshot(false).encoded_input_hits, hits);
        assert_eq!(worker.active[handle.slot as usize].as_ref().unwrap().next_audio_window, 0);
        assert!(matches!(worker.admit().unwrap(), ControlFlow::Continue(_)));
        assert!(worker.scheduler.request(handle).is_err());
        let next = worker.active.iter().flatten().next().unwrap().request;
        assert_eq!(next.slot, handle.slot);
        assert!(next.generation > handle.generation);
        assert_eq!(worker.scheduler.request(next).unwrap().audio_slot, Some(arena));
        let tokens = finish(&mut worker, &mut executor, &mut next_generation);
        assert_eq!(tokens, baseline);
        assert_eq!(fixture.metrics.snapshot(false).encoded_input_hits, hits + 1);
        drop(cancelled_generation);
    }));
    // This release also covers failures before the supervisor was spawned.
    unsafe { AtomicU32::from_ptr(flag.address() as *mut u32).store(1, Ordering::Release); }
    let shutdown = executor.close();
    if !shutdown.drained {
        std::mem::forget(executor);
        std::mem::forget(flag);
        std::mem::forget(stream);
        std::mem::forget(owner);
        panic!("actual feature-copy worker did not confirm drain; GPU owners retained");
    }
    assert!(shutdown.error.is_none(), "{shutdown:?}");
    if let Err(panic) = outcome { std::panic::resume_unwind(panic); }
}
