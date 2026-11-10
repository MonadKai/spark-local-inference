/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Actual native worker logits; the preparer is absent. This diagnostic forces
//! no token: it verifies each observed prefix against the frozen teacher history.

use super::*;
use super::encoded_cuda_tests::incoming;
use crate::runtime_package::{KvCacheLimit, LoadOptions, NativeModel, Publisher, VerifiedPackage,
    test_support::sampling};
use std::{collections::HashMap, io::Write, panic::{AssertUnwindSafe, catch_unwind}, path::PathBuf,
    time::Instant};

#[derive(serde::Deserialize)]
struct Case {
    id: String,
    path: PathBuf,
    wav_sha256: String,
    language: String,
    prompt_ids: Vec<u32>,
    teacher_history: Vec<u32>,
    #[serde(default)]
    endpoint: PromptEndpoint,
}

#[derive(Clone, Copy, Default, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum PromptEndpoint {
    #[default]
    OriginalSdkChat,
    TranscriptionEmptyContext,
    TranscriptionWholePromptDiagnostic,
}

#[test]
fn native_numerical_endpoint_defaults_preserve_original_sdk_inputs() {
    let mut value = serde_json::json!({"id":"legacy", "path":"sample.wav",
        "wav_sha256":"", "language":"zh", "prompt_ids":[], "teacher_history":[]});
    let case: Case = serde_json::from_value(value.clone()).unwrap();
    assert!(case.endpoint == PromptEndpoint::OriginalSdkChat);
    for endpoint in ["original_sdk_chat", "transcription_empty_context", "transcription_whole_prompt_diagnostic"] {
        value["endpoint"] = endpoint.into();
        let case: Case = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(case.endpoint).unwrap(), endpoint);
    }
    value["endpoint"] = "unknown_endpoint".into();
    assert!(serde_json::from_value::<Case>(value).is_err());
}

fn reference_prompt(fixture: &Fixture, bytes: &[u8], language: &str, endpoint: PromptEndpoint) -> Prompt {
    let config = &fixture.config;
    let cancelled = AtomicBool::new(false);
    let processor = crate::frontend::audio::AudioProcessor::new(
        fixture.frontend.audio_settings().unwrap().clone(), 1).unwrap();
    let pcm = processor.decode(bytes, &cancelled).unwrap();
    let (audio, identity) = processor.prepare_pcm(&pcm, config.contract.audio.as_ref().unwrap(), &cancelled).unwrap();
    let syntax = config.contract.frontend.transcription.as_ref().unwrap();
    let rendered = match endpoint {
        PromptEndpoint::OriginalSdkChat => {
            // Frozen SDK numerical inputs retain the empty system turn.
            let options = serde_json::json!({"add_generation_prompt":true}).as_object().unwrap().clone();
            let mut rendered = fixture.frontend.render_chat(&[
                serde_json::json!({"role":"system","content":""}),
                serde_json::json!({"role":"user","content":[{"type":"audio","audio":""}]})], &options).unwrap();
            let label = &syntax.languages.iter().find(|item| item.code == language).unwrap().label;
            rendered.text.push_str(&syntax.language_prefix);
            rendered.text.push_str(label);
            rendered.text.push_str(&syntax.text_delimiter);
            rendered.token_ids = fixture.frontend.encode(&rendered.text, false).unwrap();
            rendered
        }
        PromptEndpoint::TranscriptionEmptyContext | PromptEndpoint::TranscriptionWholePromptDiagnostic => {
            let renderer = crate::frontend::transcription::Renderer::new(&fixture.frontend, syntax,
                config.contract.frontend.audio_tokens.as_ref().unwrap()).unwrap();
            renderer.render(&fixture.frontend, "", Some(language), &cancelled).unwrap()
        }
    };
    let (prepared, input) = crate::model::prepare_audio_input(config, &rendered, audio, identity).unwrap();
    // Preserve the original SDK whole-prompt numerical input. The explicit
    // transcription diagnostic retains the real endpoint's message partitions.
    let mut result = Fixture::prompt_for(prepared.tokens, 1024, prepared.data);
    if endpoint == PromptEndpoint::TranscriptionEmptyContext {
        result.message_boundaries = rendered.message_boundaries;
        crate::server::request::expand_boundaries(&mut result.message_boundaries, &prepared.token_expansions);
    }
    result.conditioned_audio = Some(input);
    result.conditioning = prepared.conditioning;
    result.sampling = config.sampling_defaults.clone();
    result.sampling.seed = 0;
    result
}

fn i32s(bytes: &[u8]) -> Vec<i32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect()
}
fn i64s(bytes: &[u8]) -> Vec<i64> {
    assert_eq!(bytes.len() % 8, 0);
    bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect()
}

fn save(output: &std::path::Path, report: &serde_json::Value) {
    std::fs::write(output.join("result.pending"), serde_json::to_vec_pretty(report).unwrap()).unwrap();
    std::fs::rename(output.join("result.pending"), output.join("result.json")).unwrap();
}

#[test]
#[ignore = "requires exclusive CUDA, explicit package/publishers, FLARE_NATIVE_LOGITS_CASES and FLARE_NATIVE_LOGITS_OUTPUT"]
fn raw_logits_and_actual_histories_survive_native_worker_batch_and_slot_changes() {
    exercise_native_logits(false);
}

#[test]
#[ignore = "requires exclusive CUDA, explicit original-count ASR package/publishers and native logits inputs/output"]
fn generated_history_replays_after_real_snapshot_parking_and_checkpoint_eviction() {
    exercise_native_logits(true);
}

fn kv_prefix_matches(actual: &[Vec<u8>], reference: &[Vec<u8>]) -> bool {
    actual.len() == reference.len() && actual.iter().zip(reference).all(|(a, b)| a == b)
}

fn observation_scope(cohort: usize, request: usize) -> [u8; 24] {
    let mut scope = [0; 24];
    scope[..8].copy_from_slice(b"ASRLOGIT");
    scope[8..16].copy_from_slice(&u64::try_from(cohort).unwrap().to_le_bytes());
    scope[16..].copy_from_slice(&u64::try_from(request).unwrap().to_le_bytes());
    scope
}

// The four-input pilot retains its historical schedule. The complete frozen
// set visits every input in each full logical batch and in a seven-row B8.
fn observation_cohorts(count: usize, history: bool) -> Vec<(Vec<usize>, Option<bool>, Option<usize>)> {
    assert!(matches!(count, 4 | 20));
    if history {
        return (0..count).map(|case| (vec![case], None, None))
            .chain((0..count).flat_map(|case| [false, true]
                .map(move |evict| (vec![case], Some(evict), None)))).collect();
    }
    if count == 4 {
        return [vec![0], vec![1], vec![2], vec![3], vec![0,1], vec![2,3], vec![0,1,2,3],
            vec![3,2,1,0], vec![0,1,2,3,0,1,2,3], vec![3,2,1,0,3,2,1,0], vec![0,1,2,3,0,1,2]]
            .into_iter().map(|order| {
                let capacity = (order.len() >= 7).then_some(8);
                (order, None, capacity)
            }).collect();
    }
    let mut result = (0..count).map(|case| (vec![case], None, None)).collect::<Vec<_>>();
    for capacity in [2, 4, 8] {
        for start in (0..count).step_by(capacity) {
            let order = (0..capacity).map(|row| (start + row) % count).collect::<Vec<_>>();
            result.push((order.clone(), None, Some(capacity)));
            result.push((order.into_iter().rev().collect(), None, Some(capacity)));
        }
    }
    for start in (0..count).step_by(7) {
        let order = (0..7).map(|row| (start + row) % count).collect::<Vec<_>>();
        result.push((order.clone(), None, Some(8)));
        result.push((order.into_iter().rev().collect(), None, Some(8)));
    }
    result
}

#[test]
fn full_observation_schedule_has_unique_scopes_and_per_case_batch_and_row_coverage() {
    let mut scopes = std::collections::HashSet::new();
    let mut coverage = vec![std::collections::HashSet::new(); 20];
    for history in [false, true] {
        let cohorts = observation_cohorts(20, history);
        for (cohort, (order, _, capacity)) in cohorts.iter().enumerate() {
            for (row, &case) in order.iter().enumerate() {
                assert!(scopes.insert((history, observation_scope(cohort, row))));
                if !history { coverage[case].insert((capacity.unwrap_or(1), order.len(), row)); }
            }
        }
    }
    assert_ne!(observation_scope(0, 0), observation_scope(32, 0));
    for seen in coverage {
        for capacity in [1, 2, 4, 8] {
            assert!(seen.iter().any(|&(b, live, _)| b == capacity && live == capacity));
        }
        assert!(seen.iter().any(|&(b, live, _)| b == 8 && live == 7));
        assert!(seen.iter().any(|&(_, _, row)| row > 0));
    }
}

fn exercise_native_logits(replay_history: bool) {
    let path = std::env::var("FLARE_PROMPT_TEST_PACKAGE").unwrap();
    let publishers = std::env::var("FLARE_PROMPT_TEST_PUBLISHERS").unwrap();
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(std::env::var("FLARE_NATIVE_LOGITS_CASES").unwrap()).unwrap()).unwrap();
    assert!(matches!(cases.len(), 4 | 20));
    assert_eq!(cases.iter().map(|case| &case.id).collect::<std::collections::HashSet<_>>().len(), cases.len());
    let output = PathBuf::from(std::env::var("FLARE_NATIVE_LOGITS_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let publishers: Vec<Publisher> = serde_json::from_slice(&std::fs::read(publishers).unwrap()).unwrap();
    let verified = VerifiedPackage::open(std::path::Path::new(&path), &publishers, None).unwrap();
    let namespace = verified.publisher().namespace.clone();
    let authorized = verified.authorize(&namespace, true).unwrap();
    let mut views = HashMap::new();
    for name in sampling::graph_names(&authorized, 0).unwrap() {
        views.insert(name.clone(), sampling::views(&authorized, 0, &name, 151936,
            ["row_slot_ids", "row_generations", "live_rows"]).unwrap());
    }
    let pending = sampling::named_buffer(&authorized, 0, "request_pending_token").unwrap();
    // These are the measured package's buffer names. After loading, every
    // observed input is checked against its signed typed graph binding.
    let mut audio_views = HashMap::new();
    for capacity in [100, 800] {
        for suffix in ["mel", "frames", "total_input_frames", "convolution_width", "gather_indices",
            "cu_seqlens", "feature_count", "feature_offset", "request_slot", "generation"] {
            let name = format!("audio{capacity}_{suffix}");
            audio_views.insert(name.clone(), sampling::named_buffer(&authorized, 0, &name).unwrap());
        }
    }
    for name in ["audio_features", "audio_generations", "audio_counts", "audio_feature_slot"] {
        audio_views.insert(name.into(), sampling::named_buffer(&authorized, 0, name).unwrap());
    }
    let endpoint_diagnostic = cases.iter().any(|case| case.endpoint != PromptEndpoint::OriginalSdkChat);
    let observe_history = replay_history || cases.len() == 20 || endpoint_diagnostic;
    let original_count = observe_history.then(|| sampling::named_buffer(&authorized, 0,
        "original_prompt_tokens").unwrap());
    let history_package = observe_history.then(|| VerifiedPackage::open(std::path::Path::new(&path),
        &publishers, None).unwrap().authorize(&namespace, true).unwrap());
    let envelope = authorized.envelope_digest();
    let options = LoadOptions { max_concurrency: Some(8), kv_cache_limit: KvCacheLimit::Concurrency,
        embedding_table_storage: Some(crate::engram::Storage::Ram),
        prefix_cache: crate::offload::PrefixCacheOffload { memory_bytes: Some(0), disk_limit: Some(0),
            ..Default::default() }, ..Default::default() };
    let owner = unsafe { NativeModel::load(authorized, &[0], &options,
        &crate::startup::StartupProgress::silent()).unwrap() };
    let manifest = owner.manifests().unwrap().remove(0);
    let pending_layout = manifest.buffer("request_pending_token").unwrap();
    assert_eq!(pending_layout.dtype, crate::execution::Dtype::Int64);
    assert_eq!(pending_layout.shape, [8]);
    assert_eq!(pending_layout.strides, [1]);
    assert_eq!(pending_layout.nbytes, 8 * 8);
    let bootstrap = crate::server::Bootstrap::load(&owner.server_config().to_string(), &manifest).unwrap();
    assert!(bootstrap.model.speculative_widths.is_empty());
    assert_eq!(bootstrap.model.model_id, "Qwen/Qwen3-ASR-1.7B");
    let fixture = Fixture { config: bootstrap.model, frontend: bootstrap.frontend, stop: AtomicBool::new(false),
        metrics: Arc::new(super::super::super::metrics::Metrics::new(0, 0, 0, 16)) };
    let mut diagnostic_inputs = HashMap::new();
    if endpoint_diagnostic {
        let package = history_package.as_ref().unwrap();
        for graph in &fixture.config.contract.graphs {
            if matches!(graph.kind, crate::model::contract::GraphOperation::Prefill
                | crate::model::contract::GraphOperation::Snapshot) {
                diagnostic_inputs.insert(graph.graph.clone(), graph.inputs.iter().map(|binding|
                    (binding.buffer.clone(), sampling::named_buffer(package, 0, &binding.buffer).unwrap()))
                    .collect::<HashMap<_, _>>());
            }
        }
    }
    let kv_views = history_package.as_ref().map(|package| {
        assert_eq!(package.envelope_digest(), envelope);
        fixture.config.contract.cache.regions.iter().filter(|region|
            matches!(region.kind, crate::model::contract::HistoryKind::PagedKeyValue))
            .map(|region| sampling::kv_history_view(package, 0, region,
                fixture.config.contract.cache.page_tokens).unwrap()).collect::<Vec<_>>()
    }).unwrap_or_default();
    let page_table_view = history_package.as_ref().map(|package| {
        let [table] = fixture.config.contract.cache.page_tables.as_slice() else {
            panic!("history probe requires one authenticated full-attention page table");
        };
        assert_eq!(table.page_offset, 0);
        let layout = manifest.buffer(&table.buffer).unwrap();
        assert_eq!(layout.dtype, crate::execution::Dtype::Int32);
        assert_eq!(layout.shape.len(), 2);
        assert_eq!(layout.strides, [layout.shape[1], 1]);
        (sampling::named_buffer(package, 0, &table.buffer).unwrap(), layout.shape[1])
    });
    if history_package.is_some() {
        assert_eq!(kv_views.len(), 56);
        assert_eq!(fixture.config.contract.cache.immutable_request_values,
            [crate::model::contract::ValueSource::InitialPromptTokens]);
    }
    let audio = fixture.config.contract.audio.as_ref().unwrap();
    for graph in fixture.config.contract.graphs.iter().filter(|g|
        g.kind == crate::model::contract::GraphOperation::AudioEncode) {
        let descriptor = manifest.graphs.iter().find(|g| g.name == graph.graph).unwrap();
        for input in &graph.inputs {
            let view = audio_views.get(&input.buffer).unwrap();
            let layout = manifest.buffer(&input.buffer).unwrap();
            assert_eq!(view.nbytes, layout.nbytes);
            if let Some(kind) = input.audio_tensor {
                let (dtype, shape) = audio.tensor_layout(kind, descriptor.token_capacity).unwrap();
                assert_eq!((layout.dtype, &layout.shape), (dtype, &shape));
            }
        }
    }
    let mut executor = unsafe { RankExecutor::new(manifest, None).unwrap() };
    let mut report = serde_json::json!({"scope":"Native package BF16 logits and actual natural histories for the supplied inputs; not complete operator, lifecycle or performance qualification",
        "input_count":cases.len(),
        "prompt_endpoint":if cases.iter().all(|case| case.endpoint == PromptEndpoint::OriginalSdkChat) {
            "Original SDK/chat template and forced-language suffix; separate from transcription API semantics"
        } else { "Explicit per-case endpoint; transcription uses its actual renderer, with message partitions preserved unless the whole-prompt diagnostic is explicitly selected" },
        "case_endpoints":cases.iter().map(|case| serde_json::json!({"id":case.id,"endpoint":case.endpoint})).collect::<Vec<_>>(),
        "prepared_identity":owner.server_config()["identity"],
        "runtime_cache_identity":fixture.config.identity.iter().map(|v| format!("{v:02x}")).collect::<String>(),
        "authenticated_envelope_digest":envelope.iter().map(|v| format!("{v:02x}")).collect::<String>(),
        "audio_storage":audio, "status":"running", "single_inflight":true,
        "frames":[], "audio_frames":[], "cohorts":[], "errors":[], "kv_frames":[],
        "endpoint_input_frames":[], "snapshot_kv_frames":[],
        "generated_history_probe":replay_history, "preemption_policy_qualified":false,
        "capacity_schedule":"Declared batch cohorts defer decode until every real request has naturally sampled its first token. The test selects legal production plans through test-only phase selection; this is not automatic scheduling or performance qualification.",
        "decode_readiness_barriers":[],
        "parking":[]} );
    save(&output, &report);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let mut scheduler = RequestScheduler::new(fixture.config.identity, fixture.config.limits,
            crate::model::scheduler_capabilities_with_manifest(&fixture.config, executor.manifest()).unwrap()).unwrap();
        scheduler.graph_durations(&executor.manifest().graphs).unwrap();
        scheduler.configure_policy(Default::default()).unwrap();
        if let Some(cache) = &executor.manifest().cache {
            if !cache.snapshot_pages.is_empty() { scheduler.share_snapshot_pages(&cache.snapshot_pages).unwrap(); }
        }
        let prefills = crate::model::catalog::prefill_fragment_capacities(&fixture.config,
            executor.manifest(), GraphKind::Prefill).unwrap();
        scheduler.prefill_graphs(&prefills).unwrap();
        let (sender, receiver) = mpsc::sync_channel(4);
        let mut worker = Worker::new(&fixture.config, &fixture.frontend, &fixture.stop,
            &fixture.metrics, scheduler, receiver, 32, super::super::super::PrefixCacheReuse::Shared);
        worker.inputs = Some(crate::model::Inputs::new(&fixture.config, executor.manifest()).unwrap());
        let mut baseline: HashMap<usize, Vec<Vec<u8>>> = HashMap::new();
        let mut baseline_kv: HashMap<usize, Vec<Vec<u8>>> = HashMap::new();
        let cohorts = observation_cohorts(cases.len(), replay_history);
        let mut seen_decode = std::collections::HashSet::new();
        let mut seen_by_case = vec![std::collections::HashSet::new(); cases.len()];
        for (cohort, (order, eviction, expected_capacity)) in cohorts.iter().enumerate() {
            assert_eq!(executor.outstanding(), 0);
            let mut generations = Vec::new();
            let mut arrivals = Vec::new();
            let mut files = Vec::new();
            let mut request_indices = HashMap::new();
            for (index, &case) in order.iter().enumerate() {
                let bytes = std::fs::read(&cases[case].path).unwrap();
                assert_eq!(format!("{:x}", Sha256::digest(&bytes)), cases[case].wav_sha256);
                let prepared = reference_prompt(&fixture, &bytes, &cases[case].language, cases[case].endpoint);
                assert_eq!(prepared.tokens, cases[case].prompt_ids, "actual native prompt differs for {}", cases[case].id);
                let (mut request, generation) = incoming(&fixture, prepared, 0);
                let scope = observation_scope(cohort, index);
                request.prefix_scope = Some(scope);
                assert!(request_indices.insert(scope, index).is_none());
                arrivals.push(request); generations.push(generation);
                files.push(std::fs::OpenOptions::new().create_new(true).write(true)
                    .open(output.join(format!("cohort{cohort}-request{index}-logits.bf16"))).unwrap());
            }
            sender.send(arrivals).unwrap();
            let mut histories = vec![Vec::<u32>::new(); order.len()];
            let mut delivered = vec![Vec::<u32>::new(); order.len()];
            let mut observed = vec![Vec::<Vec<u8>>::new(); order.len()];
            let mut finished = vec![false; order.len()];
            let mut snapshot_completed = false;
            let mut park_started = false;
            let mut parked_once = false;
            let mut readmitted_once = false;
            let mut history_prefill = false;
            let mut pressure_reservations = 0;
            let mut poisoned_prefill_submissions = 0;
            let mut submitted: Option<crate::scheduler::ScheduledPlan> = None;
            let mut decode_barrier = expected_capacity.is_some();
            let deadline = Instant::now() + Duration::from_secs(300);
            loop {
                assert!(Instant::now() < deadline, "native logits cohort did not finish");
                assert!(matches!(worker.admit().unwrap(), ControlFlow::Continue(_)));
                if let Some(completion) = executor.poll().unwrap() {
                    assert_eq!(executor.outstanding(), 0, "a later graph may overwrite observed logits");
                    let plan = submitted.take().unwrap();
                    let graph = &executor.manifest().graphs[completion.graph_index];
                    if let Some(inputs) = diagnostic_inputs.get(&graph.name) {
                        let contract = fixture.config.contract.graph(&graph.name).unwrap();
                        let mut buffers = serde_json::Map::new();
                        for binding in &contract.inputs {
                            let view = &inputs[&binding.buffer];
                            let bytes = sampling::download(&owner, 0, view).unwrap();
                            let file = format!("cohort{cohort}-fence{}-endpoint-{}.bin", completion.fence.0, binding.buffer);
                            std::fs::write(output.join(&file), bytes).unwrap();
                            let layout = executor.manifest().buffer(&binding.buffer).unwrap();
                            buffers.insert(binding.buffer.clone(), serde_json::json!({"file":file,
                                "source":binding.source,"indexing":binding.indexing,
                                "dtype":format!("{:?}",layout.dtype),"shape":layout.shape,
                                "strides":layout.strides,"nbytes":layout.nbytes}));
                        }
                        report["endpoint_input_frames"].as_array_mut().unwrap().push(serde_json::json!({
                            "cohort":cohort,"graph":graph.name,"fence":completion.fence.0,"buffers":buffers,
                            "completion_records":completion.records.iter().map(|r| r.fields()).collect::<Vec<_>>()}));
                        save(&output, &report);
                    }
                    if endpoint_diagnostic && graph.kind == GraphKind::Snapshot {
                        let geometry = executor.manifest().cache.as_ref().unwrap();
                        let (table, _) = page_table_view.as_ref().unwrap();
                        let table_bytes = sampling::download(&owner, 0, table).unwrap();
                        let table_file = format!("cohort{cohort}-fence{}-snapshot-pages.i32", completion.fence.0);
                        std::fs::write(output.join(&table_file), table_bytes).unwrap();
                        let pending_tokens = i64s(&sampling::download(&owner, 0, &pending).unwrap());
                        for lane in &plan.lanes {
                            let slot = lane.request.slot as usize;
                            let record = &completion.records[slot];
                            let consumed = usize::try_from(record.consumed_tokens).unwrap();
                            let active = worker.active[slot].as_ref().unwrap();
                            let index = request_indices[&active.incoming.prefix_scope.unwrap()];
                            let request = worker.scheduler.request(lane.request).unwrap();
                            let pages = request.pages().map(|page| page.0).collect::<Vec<_>>();
                            let mut planes = Vec::new();
                            for view in &kv_views {
                                let bytes = sampling::download_kv_prefix(&owner, 0, view, geometry, &pages, consumed).unwrap();
                                let file = format!("cohort{cohort}-request{index}-fence{}-snapshot-{}.kv", completion.fence.0, view.buffer);
                                std::fs::write(output.join(&file), &bytes).unwrap();
                                planes.push(serde_json::json!({"buffer":view.buffer,"file":file,"nbytes":bytes.len()}));
                            }
                            report["snapshot_kv_frames"].as_array_mut().unwrap().push(serde_json::json!({
                                "cohort":cohort,"request":index,"case":cases[order[index]].id,
                                "graph":graph.name,"fence":completion.fence.0,"slot":slot,
                                "generation":lane.request.generation,"consumed":consumed,
                                "pending_token":pending_tokens[slot],"page_table":table_file,
                                "pages":pages,"planes":planes,"completion_record":record.fields()}));
                            save(&output, &report);
                            assert_eq!(active.request, lane.request);
                            assert_eq!(record.generation, lane.request.generation as i64);
                            assert_eq!(record.submission_sequence, completion.fence.0 as i64);
                            assert_eq!(record.error_code, 0);
                            assert_eq!(lane.tokens, 0);
                            assert_eq!(consumed as u64, lane.start_position);
                        }
                    }
                    if graph.kind == GraphKind::Snapshot && park_started {
                        assert_eq!(plan.lanes.len(), 1);
                        let lane = &plan.lanes[0];
                        let record = &completion.records[lane.request.slot as usize];
                        assert_eq!(record.generation, lane.request.generation as i64);
                        assert_eq!(record.submission_sequence, completion.fence.0 as i64);
                        assert_eq!(record.error_code, 0);
                        // This fixture has one in-flight plan. Readmission can
                        // schedule another prompt-boundary snapshot after G2.
                        assert_eq!(lane.tokens, 0);
                        assert_eq!(u64::try_from(record.consumed_tokens).unwrap(), lane.start_position);
                        let initial = !snapshot_completed;
                        if initial {
                            assert!(!parked_once && !readmitted_once);
                            assert_eq!(record.consumed_tokens as usize, cases[order[0]].prompt_ids.len() + 1);
                            snapshot_completed = true;
                        }
                        report["parking"].as_array_mut().unwrap().push(serde_json::json!({
                            "cohort":cohort,"event":if initial { "actual_snapshot_completed" }
                                else { "subsequent_snapshot_completed" },"graph":graph.name,
                            "request_slot":lane.request.slot,"generation":lane.request.generation,
                            "fence":completion.fence.0,"consumed":record.consumed_tokens,
                            "lane_start":lane.start_position,"lane_tokens":lane.tokens,
                            "completion_record":record.fields()}));
                        save(&output, &report);
                    }
                    if graph.kind == GraphKind::AudioEncode {
                        let contract = fixture.config.contract.graph(&graph.name).unwrap();
                        let mut inputs = serde_json::Map::new();
                        let prefix = format!("cohort{cohort}-fence{}-audio", completion.fence.0);
                        for input in &contract.inputs {
                            let bytes = sampling::download(&owner, 0, &audio_views[&input.buffer]).unwrap();
                            let file = format!("{prefix}-{}.bin", input.buffer);
                            std::fs::write(output.join(&file), &bytes).unwrap();
                            let layout = executor.manifest().buffer(&input.buffer).unwrap();
                            inputs.insert(input.buffer.clone(), serde_json::json!({"file":file,
                                "source":input.source,"audio_tensor":input.audio_tensor,
                                "dtype":format!("{:?}",layout.dtype),"shape":layout.shape,
                                "strides":layout.strides,"nbytes":layout.nbytes}));
                        }
                        let storage = fixture.config.contract.audio.as_ref().unwrap();
                        for name in [&storage.features.buffer, &storage.generations.buffer, &storage.counts.buffer] {
                            let bytes = sampling::download(&owner, 0, &audio_views[name]).unwrap();
                            let file = format!("{prefix}-{name}.bin");
                            std::fs::write(output.join(&file), &bytes).unwrap();
                            inputs.insert(name.clone(), serde_json::json!({"file":file}));
                        }
                        let request = plan.lanes[0].request;
                        let slot = request.slot as usize;
                        let active = worker.active[slot].as_ref().unwrap();
                        assert_eq!(active.request, request);
                        let index = request_indices[&active.incoming.prefix_scope.unwrap()];
                        report["audio_frames"].as_array_mut().unwrap().push(serde_json::json!({"cohort":cohort,
                            "graph":graph.name,"fence":completion.fence.0,"buffers":inputs,
                            "request":index,"case":cases[order[index]].id,
                            "request_slot":slot,"request_generation":request.generation}));
                        save(&output, &report);
                    }
                    if let Some(view) = views.get(graph.name.as_str()) {
                        let logits = sampling::download(&owner, 0, &view.logits).unwrap();
                        // Preserve the complete physical frame before trusting
                        // its row map or any completion/nonfinite assertion.
                        let raw_name = format!("cohort{cohort}-fence{}-frame.bf16", completion.fence.0);
                        std::fs::write(output.join(&raw_name), &logits).unwrap();
                        let slots = i32s(&sampling::download(&owner, 0, &view.slots).unwrap());
                        let epochs = i64s(&sampling::download(&owner, 0, &view.epochs).unwrap());
                        let live = i32s(&sampling::download(&owner, 0, &view.live).unwrap())[0] as usize;
                        report["frames"].as_array_mut().unwrap().push(serde_json::json!({"cohort":cohort,
                            "graph":graph.name,"kind":format!("{:?}", graph.kind),
                            "initial_prompt_capacity":graph.initial_prompt_capacity,"token_capacity":graph.token_capacity,
                            "lanes":plan.lanes.iter().enumerate().map(|(row, lane)| {
                                let active = worker.active[lane.request.slot as usize].as_ref().unwrap();
                                assert_eq!(active.request, lane.request);
                                let scope = active.incoming.prefix_scope.unwrap();
                                let index = request_indices[&scope];
                                serde_json::json!({"row":row,"request":index,"case":cases[order[index]].id,
                                    "scope":scope,"slot":lane.request.slot,"generation":lane.request.generation,
                                    "start":lane.start_position,"tokens":lane.tokens,
                                    "history_step":histories[index].len()})
                            }).collect::<Vec<_>>(),
                            "capacity":graph.batch_capacity,"live":live,"slots":slots,
                            "epochs":epochs,"fence":completion.fence.0,"raw_frame":raw_name,
                            "completion_records":completion.records.iter().map(|r| r.fields()).collect::<Vec<_>>()}));
                        save(&output, &report);
                        assert_eq!(live, plan.lanes.len());
                        if graph.kind == GraphKind::Decode { seen_decode.insert((graph.batch_capacity, live)); }
                        for (row, lane) in plan.lanes.iter().enumerate() {
                            let slot = lane.request.slot as usize;
                            assert_eq!((slots[row], epochs[row]), (slot as i32, lane.request.generation as i64));
                            let record = &completion.records[slot];
                            assert_eq!(record.generation, epochs[row]);
                            assert_eq!(record.submission_sequence, completion.fence.0 as i64);
                            let active = worker.active[slot].as_ref().unwrap();
                            assert_eq!(active.request, lane.request);
                            let index = request_indices[&active.incoming.prefix_scope.unwrap()];
                            assert_eq!(record.error_code, 0);
                            let count = record.vector.as_ref().unwrap().emitted_count;
                            let case = order[index];
                            if let Some(original_count) = original_count.as_ref().filter(|_| graph.kind == GraphKind::Prefill) {
                                let original = i32s(&sampling::download(&owner, 0, original_count).unwrap());
                                assert_eq!(original[0] as usize, cases[case].prompt_ids.len());
                                if lane.start_position + lane.tokens as u64 > original[0] as u64 {
                                    assert!(readmitted_once);
                                    history_prefill = true;
                                }
                            }
                            if graph.kind == GraphKind::Decode {
                                seen_by_case[case].insert((graph.batch_capacity, live, row, slot, lane.request.generation));
                            }
                            let step = histories[index].len();
                            let observe_kv = history_package.is_some() && (graph.kind == GraphKind::Prefill
                                || graph.kind == GraphKind::ScoredPrefill
                                || (count != 0 && step < cases[case].teacher_history.len())
                                || (replay_history && step == 2));
                            if count == 0 { assert!(graph.kind.is_prefill()); }
                            if count > 0 { assert_eq!(count, 1); }
                            if observe_kv {
                                assert!((1..=2).contains(&record.status));
                                assert_eq!(record.consumed_tokens as u64,
                                    lane.start_position.checked_add(lane.tokens as u64).unwrap());
                                let consumed = record.consumed_tokens as usize;
                                let request = worker.scheduler.request(lane.request).unwrap();
                                let pages = request.pages().map(|page| page.0).collect::<Vec<_>>();
                                let geometry = executor.manifest().cache.as_ref().unwrap();
                                let (table, columns) = page_table_view.as_ref().unwrap();
                                let actual_table = i32s(&sampling::download(&owner, 0, table).unwrap());
                                let needed = consumed.div_ceil(fixture.config.contract.cache.page_tokens);
                                assert_eq!(&actual_table[slot * columns..slot * columns + needed],
                                    pages[..needed].iter().map(|page| *page as i32).collect::<Vec<_>>());
                                let compared_with_baseline = cohort >= cases.len();
                                let mut values = Vec::new();
                                let mut planes = Vec::new();
                                for view in &kv_views {
                                    assert!(sampling::download_kv_prefix(&owner, 0, view, geometry,
                                        &[], consumed).is_err());
                                    assert!(sampling::download_kv_prefix(&owner, 0, view, geometry,
                                        &pages, 0).is_err());
                                    let mut bad_pages = pages.clone();
                                    bad_pages[0] = u32::try_from(geometry.page_capacity).unwrap();
                                    assert!(sampling::download_kv_prefix(&owner, 0, view, geometry,
                                        &bad_pages, consumed).is_err());
                                    let value = sampling::download_kv_prefix(&owner, 0, view, geometry,
                                        &pages, consumed).unwrap();
                                    let file = format!("cohort{cohort}-request{index}-fence{}-{}.kv",
                                        completion.fence.0, view.buffer);
                                    std::fs::write(output.join(&file), &value).unwrap();
                                    planes.push(serde_json::json!({"buffer":view.buffer,"file":file,"nbytes":value.len()}));
                                    values.push(value);
                                }
                                if compared_with_baseline {
                                    let expected = &baseline_kv[&case];
                                    let prefix = expected.iter().map(|plane| plane[..consumed * 2048].to_vec()).collect::<Vec<_>>();
                                    assert!(kv_prefix_matches(&values, &prefix), "KV changed across actual batch, slot or history replay");
                                } else {
                                    if let Some(previous) = baseline_kv.get(&case) {
                                        for (value, before) in values.iter().zip(previous) {
                                            assert!(value.starts_with(before), "earlier consumed KV changed within B1");
                                        }
                                    }
                                    baseline_kv.insert(case, values);
                                }
                                report["kv_frames"].as_array_mut().unwrap().push(serde_json::json!({
                                    "cohort":cohort,"request":index,"case":cases[case].id,"consumed":consumed,
                                    "fence":completion.fence.0,"kind":format!("{:?}",graph.kind),
                                    "initial_prompt_capacity":graph.initial_prompt_capacity,"token_capacity":graph.token_capacity,
                                    "lane_start":lane.start_position,"lane_tokens":lane.tokens,
                                    "request_slot":slot,"generation":lane.request.generation,"pages":pages,
                                    "graph":graph.name,"original_prompt_tokens":cases[case].prompt_ids.len(),
                                    "history_step":step,"planes":planes,"actual_page_table_matches":true,
                                    "compared_with_baseline":compared_with_baseline,
                                    "exact_if_compared":compared_with_baseline.then_some(true)}));
                                save(&output, &report);
                            }
                            if count == 0 { continue; }

                            assert_eq!(record.consumed_tokens as usize, cases[case].prompt_ids.len() + histories[index].len());
                            let bytes = logits[row * 151936 * 2..(row + 1) * 151936 * 2].to_vec();
                            files[index].write_all(&bytes).unwrap();
                            files[index].flush().unwrap();
                            let mut maximum = f32::NEG_INFINITY;
                            let mut argmax = 0;
                            for (token, value) in bytes.chunks_exact(2).enumerate() {
                                let value = f32::from_bits((u16::from_le_bytes(value.try_into().unwrap()) as u32) << 16);
                                assert!(value.is_finite(), "native logits contain a nonfinite value");
                                if value > maximum { maximum = value; argmax = token; }
                            }
                            assert_eq!(record.token_id, argmax as i64, "native sampler skipped the observed BF16 node");
                            if step < cases[case].teacher_history.len() {
                                assert_eq!(record.token_id as u32, cases[case].teacher_history[step],
                                    "actual natural history differs from the frozen teacher history");
                            }
                            if let Some(reference) = baseline.get(&case) {
                                assert!(reference.get(step).is_some_and(|value| value == &bytes),
                                    "native logits changed across batch/slot at step {step}");
                            }
                            histories[index].push(record.token_id as u32);
                            observed[index].push(bytes);
                        }
                    }
                    worker.complete(graph.kind, completion).unwrap();
                }
                worker.publish().unwrap();
                if let Some(evict) = eviction {
                    if !park_started && histories[0].len() == 2 && executor.outstanding() == 0 {
                        let handle = worker.active.iter().flatten().next().unwrap().request;
                        crate::scheduler::tests::preempt_completed_decode(&mut worker.scheduler, handle).unwrap();
                        park_started = true;
                    }
                    if park_started && !parked_once && !worker.parked.is_empty() {
                        assert!(snapshot_completed, "no actual terminal Snapshot graph completed");
                        assert_eq!(worker.parked.len(), 1);
                        let incoming = &worker.parked.front().unwrap().0;
                        assert_eq!(incoming.initial_prompt_tokens, cases[order[0]].prompt_ids.len());
                        assert_eq!(incoming.prompt.tokens.len(), incoming.initial_prompt_tokens + 2);
                        assert!(incoming.prompt.data.audio_token_indices[incoming.initial_prompt_tokens..]
                            .iter().all(|index| *index == -1));
                        let parked_spec = incoming.spec();
                        let resident_before = worker.scheduler.plan_admission(&parked_spec).unwrap().resident_tokens();
                        assert_eq!(resident_before as usize, cases[order[0]].prompt_ids.len() + 1);
                        if *evict {
                            let mut handles = Vec::new();
                            let per_request = fixture.config.limits.max_sequence_tokens
                                / fixture.config.contract.cache.page_tokens;
                            let mut remaining = usize::try_from(fixture.config.limits.pages).unwrap();
                            while remaining != 0 {
                                let pages = remaining.min(per_request);
                                let tokens = pages * fixture.config.contract.cache.page_tokens - 1;
                                handles.push(worker.scheduler.admit(RequestSpec {
                                    initial_prompt_tokens: tokens, prompt: vec![7 + handles.len() as u32; tokens],
                                    prefix_scope: None, message_boundaries: vec![], conditioning: vec![],
                                    max_new_tokens: 1, growable_output: false, grammar_constrained: false,
                                    host_stops: false, prompt_logprobs: false,
                                }).unwrap());
                                remaining -= pages;
                            }
                            assert_eq!(worker.scheduler.free_page_count(), 0);
                            pressure_reservations = handles.len();
                            for handle in handles { worker.scheduler.cancel(handle).unwrap(); }
                        }
                        let resident_after = worker.scheduler.plan_admission(&parked_spec).unwrap().resident_tokens();
                        assert_eq!(resident_after, if *evict { 0 } else { resident_before });
                        report["parking"].as_array_mut().unwrap().push(serde_json::json!({
                            "cohort":cohort,"event":"checkpoint_residency_verified","evicted":evict,
                            "resident_before":resident_before,"resident_after":resident_after,
                            "pressure_reservations":pressure_reservations}));
                        save(&output, &report);
                        parked_once = true;
                    }
                    if parked_once && !readmitted_once {
                        if let Some(active) = worker.active.iter().flatten().next() {
                            assert!(active.incoming.readmitted);
                            let request = worker.scheduler.request(active.request).unwrap();
                            let expected = if *evict { 0 } else { cases[order[0]].prompt_ids.len() + 1 };
                            assert_eq!(request.consumed_tokens as usize, expected);
                            assert_eq!(request.initial_prompt_tokens(), cases[order[0]].prompt_ids.len());
                            readmitted_once = true;
                        }
                    }
                }
                worker.service_encoded_inputs(&mut executor).unwrap();
                for (index, generation) in generations.iter_mut().enumerate() {
                    while let Ok(event) = generation.events.try_recv() {
                        match event {
                            GenerationEvent::Token(token) => delivered[index].push(token),
                            GenerationEvent::Finished { reason, .. } => { assert_eq!(reason, "stop"); finished[index] = true; }
                            GenerationEvent::Error(error) | GenerationEvent::AdmissionRejected(error) => panic!("{error}"),
                            _ => {}
                        }
                    }
                }
                if finished.iter().all(|v| *v) && worker.active.iter().all(Option::is_none) && executor.outstanding() == 0 { break; }
                if executor.outstanding() == 0 {
                    let normal = worker.plan(0).unwrap().0;
                    let plan = if decode_barrier {
                        let active = worker.active.iter().flatten().collect::<Vec<_>>();
                        let ready = active.len() == order.len()
                            && active.iter().all(|request| worker.scheduler.request(request.request).unwrap().phase
                                == crate::scheduler::RequestPhase::Decode)
                            && histories.iter().all(|history| history.len() == 1);
                        if ready {
                            assert!(finished.iter().all(|finished| !*finished));
                            let batch = expected_capacity.unwrap();
                            let plan = worker.scheduler.try_phase_plan(GraphKind::Decode, None, Some(batch))
                                .expect("all real requests must fit their declared decode capacity");
                            assert_eq!((plan.batch_capacity, plan.lanes.len()), (batch, order.len()));
                            report["decode_readiness_barriers"].as_array_mut().unwrap().push(serde_json::json!({
                                "cohort":cohort,"batch_capacity":batch,"live":plan.lanes.len(),
                                "natural_tokens_per_request":histories.iter().map(Vec::len).collect::<Vec<_>>(),
                                "slots":plan.lanes.iter().map(|lane| lane.request.slot).collect::<Vec<_>>() }));
                            save(&output, &report);
                            decode_barrier = false;
                            Some(plan)
                        } else {
                            [GraphKind::Init, GraphKind::Restore, GraphKind::Vision,
                                GraphKind::AudioEncode, GraphKind::Prefill, GraphKind::ScoredPrefill,
                                GraphKind::Snapshot, GraphKind::Resume].into_iter().find_map(|kind|
                                    worker.scheduler.try_phase_plan(kind, None, None))
                        }
                    } else { normal };
                    if let Some(plan) = plan {
                        if plan.kind == GraphKind::Decode {
                            let actual = i64s(&sampling::download(&owner, 0, &pending).unwrap());
                            for lane in &plan.lanes {
                                let slot = lane.request.slot as usize;
                                let active = worker.active[slot].as_ref().unwrap();
                                assert_eq!(active.request, lane.request);
                                let index = request_indices[&active.incoming.prefix_scope.unwrap()];
                                assert_eq!(actual[slot], i64::from(*histories[index].last().unwrap()),
                                    "packed pending input differs from actual history");
                            }
                        }
                        let poison_event = if replay_history && readmitted_once && plan.kind == GraphKind::Prefill {
                            assert_eq!(plan.lanes.len(), 1);
                            let lane = &plan.lanes[0];
                            let start = usize::try_from(lane.start_position).unwrap();
                            let end = start.checked_add(lane.tokens).unwrap();
                            let request = worker.scheduler.request(lane.request).unwrap();
                            assert_eq!(request.consumed_tokens as usize, start);
                            let pages = crate::scheduler::tests::owned_execution_pages(&worker.scheduler,
                                lane.request).unwrap();
                            let geometry = executor.manifest().cache.as_ref().unwrap();
                            let mut poisoned = Vec::new();
                            for view in &kv_views {
                                let bytes = sampling::poison_kv_range(&owner, 0, view, geometry, &pages,
                                    start..end).unwrap();
                                assert_eq!(bytes.len(), end * 2048);
                                std::fs::write(output.join(format!("cohort{cohort}-start{start}-{}.poison",
                                    view.buffer)), &bytes).unwrap();
                                poisoned.push(bytes);
                            }
                            let reference = baseline_kv[&order[0]].iter().map(|plane|
                                plane[..end * 2048].to_vec()).collect::<Vec<_>>();
                            assert!(!kv_prefix_matches(&poisoned, &reference), "KV comparison missed actual device poison");
                            let events = report["parking"].as_array_mut().unwrap();
                            let event = events.len();
                            events.push(serde_json::json!({"cohort":cohort,"event":"device_KV_poison",
                                "slot":lane.request.slot,"generation":lane.request.generation,"pages":pages,
                                "start":start,"end":end,"planes":56,"readback_verified":true,
                                "earlier_prefix_unchanged":true,"same_comparator_rejected_poison":true,
                                "submitted":false}));
                            save(&output, &report);
                            Some(event)
                        } else { None };
                        let expected_fence = executor.next_fence().unwrap();
                        let accepted = worker.submit(&mut executor, &plan).unwrap();
                        if let Some(event) = poison_event {
                            report["parking"][event]["submitted"] = accepted.into();
                            if accepted {
                                report["parking"][event]["fence"] = expected_fence.0.into();
                                poisoned_prefill_submissions += 1;
                            }
                            save(&output, &report);
                        }
                        if accepted { submitted = Some(plan); }
                    }
                }
                std::thread::yield_now();
            }
            for (index, &case) in order.iter().enumerate() {
                assert_eq!(histories[index], delivered[index], "worker delivery differs from actual sampled tokens");
                assert!(histories[index].len() >= cases[case].teacher_history.len());
                if let Some(reference) = baseline.get(&case) { assert_eq!(observed[index].len(), reference.len()); }
                else { baseline.insert(case, observed[index].clone()); }
            }
            if eviction.is_some() {
                assert!(snapshot_completed && park_started && parked_once && readmitted_once && history_prefill
                    && poisoned_prefill_submissions > 0,
                    "no actual generated-history PREFILL completed");
            }
            report["cohorts"].as_array_mut().unwrap().push(serde_json::json!({"cohort":cohort,"order":order,"expected_decode_capacity":expected_capacity,
                "scopes":(0..order.len()).map(|row| observation_scope(cohort, row)).collect::<Vec<_>>(),
                "actual_tokens":histories,"natural_stop":finished,"checkpoint_evicted":eviction,
                "real_snapshot_parking":snapshot_completed && parked_once,"readmitted":readmitted_once,
                "actual_history_prefill":history_prefill,"pressure_reservations":pressure_reservations,
                "poisoned_prefill_submissions":poisoned_prefill_submissions}));
            save(&output, &report);
        }
        report["per_case_decode_witnesses"] = serde_json::json!(cases.iter().enumerate().map(|(case, input)|
            serde_json::json!({"case":input.id,"witnesses":seen_by_case[case]})).collect::<Vec<_>>());
        if !replay_history && cases.len() == 20 {
            for seen in &seen_by_case {
                for capacity in [1usize, 2, 4, 8] {
                    assert!(seen.iter().any(|&(b, live, _, _, _)| b == capacity && live == capacity),
                        "one input lacks an actual full-batch decode completion");
                }
                assert!(seen.iter().any(|&(b, live, _, _, _)| b == 8 && live == 7));
                assert!(seen.iter().any(|&(_, _, row, _, _)| row > 0));
                assert!(seen.iter().any(|&(_, _, _, slot, _)| slot > 0));
            }
        }
        report["actual_decode_capacity_and_live_rows"] = serde_json::json!(seen_decode);
        if !replay_history {
        assert!([1usize, 2, 4, 8].iter().all(|b| seen_decode.contains(&(*b, *b))),
            "native scheduling did not exercise every required full logical decode batch");
        assert!(seen_decode.iter().any(|(capacity, live)| 0 < *live && live < capacity),
            "native scheduling did not exercise a partially occupied decode graph");
        }
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        assert!(!maps.contains("libpython") && !maps.contains("libtorch"));
    }));
    let shutdown = executor.close();
    report["shutdown"] = serde_json::json!({"drained":shutdown.drained,
        "error":shutdown.error.as_ref().map(ToString::to_string)});
    if let Err(error) = &outcome {
        let message = error.downcast_ref::<String>().map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied()).unwrap_or("non-string native fixture panic");
        report["errors"].as_array_mut().unwrap().push(message.into());
    }
    if let Some(error) = &shutdown.error { report["errors"].as_array_mut().unwrap().push(error.to_string().into()); }
    if !shutdown.drained {
        std::mem::forget(executor);
        std::mem::forget(owner);
        report["status"] = "undrained_owners_retained".into(); save(&output, &report);
        panic!("native logits worker did not confirm drain; owners retained");
    }
    if outcome.is_err() || shutdown.error.is_some() {
        report["status"] = "failed_or_interrupted".into(); save(&output, &report);
    }
    if let Err(error) = outcome {
        std::panic::resume_unwind(error);
    }
    assert!(shutdown.error.is_none(), "{shutdown:?}");
    report["status"] = if cases.len() == 4 { "complete_native_four_input_diagnostic" }
        else { "complete_native_twenty_input_diagnostic" }.into(); save(&output, &report);
}
