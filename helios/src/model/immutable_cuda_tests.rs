/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Exercise real semantic packing, pinned uploads and captured consumption.

use super::*;
use crate::{
    execution::{GraphKind, RankExecutor, SamplingMode},
    runtime_package::{KvCacheLimit, LoadOptions, NativeModel, test_support::sampling},
    scheduler::{RequestScheduler, RequestSpec},
};
use std::{panic::{AssertUnwindSafe, catch_unwind}, time::{Duration, Instant}};

#[test]
#[ignore = "requires exclusive CUDA and FLARE_IMMUTABLE_INPUT_TEST_PACKAGE from the immutable generation fixture"]
fn immutable_values_reach_captured_programs_after_native_packing() {
    let path = std::env::var_os("FLARE_IMMUTABLE_INPUT_TEST_PACKAGE").unwrap();
    let authorized = crate::runtime_package::test_support::authorize(std::path::Path::new(&path));
    let input = sampling::named_buffer(&authorized, 0, "original_prompt_tokens").unwrap();
    let observed = sampling::named_buffer(&authorized, 0, "observed_original_prompt_tokens").unwrap();
    let options = LoadOptions { max_concurrency: Some(8), kv_cache_limit: KvCacheLimit::Concurrency,
        embedding_table_storage: Some(crate::engram::Storage::Ram),
        prefix_cache: crate::offload::PrefixCacheOffload { memory_bytes: Some(0), disk_limit: Some(0),
            ..Default::default() }, ..Default::default() };
    let owner = unsafe { NativeModel::load(authorized, &[0], &options,
        &crate::startup::StartupProgress::silent()).unwrap() };
    let manifest = owner.manifests().unwrap().remove(0);
    let config = crate::server::Bootstrap::load(&owner.server_config().to_string(), &manifest).unwrap().model;
    assert_eq!(config.limits.max_active, 8);
    assert_eq!(config.contract.cache.immutable_request_values, [contract::ValueSource::InitialPromptTokens]);
    assert_eq!(manifest.buffer("original_prompt_tokens").unwrap().shape, [8]);
    let mut executor = unsafe { RankExecutor::new(manifest, None).unwrap() };
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let mut scheduler = RequestScheduler::new(config.identity, config.limits,
            scheduler_capabilities_with_manifest(&config, executor.manifest()).unwrap()).unwrap();
        scheduler.graph_durations(&executor.manifest().graphs).unwrap();
        scheduler.configure_policy(Default::default()).unwrap();
        if let Some(cache) = &executor.manifest().cache {
            if !cache.snapshot_pages.is_empty() { scheduler.share_snapshot_pages(&cache.snapshot_pages).unwrap(); }
        }
        scheduler.prefill_graphs(&[128]).unwrap();
        let mut inputs = Inputs::new(&config, executor.manifest()).unwrap();
        let mut admissions: [Option<Admission>; 8] = std::array::from_fn(|_| None);
        let mut generations = [0; 8];
        let mut covered = std::collections::HashSet::new();
        let cohorts = [vec![7], vec![0, 7], vec![0, 2, 5, 7], (0..8).collect(),
            vec![1, 4, 7], (0..8).collect(), (0..8).collect()];
        for (cohort, keep) in cohorts.iter().enumerate() {
            let mut handles = Vec::new();
            let mut originals = [0usize; 8];
            for identity in 0..8 {
                let original = 19 + 64 * cohort + if cohort % 2 == 0 { identity } else { 7 - identity };
                let prompt = vec![2; original + 17];
                let request = RequestSpec { prefix_scope: None, initial_prompt_tokens: original,
                    prompt: prompt.clone(), message_boundaries: vec![], conditioning: vec![],
                    max_new_tokens: 4, grammar_constrained: false, host_stops: false,
                    prompt_logprobs: false, growable_output: false };
                let handle = scheduler.admit(request).unwrap();
                let slot = handle.slot as usize;
                assert!(handle.generation > generations[slot]);
                generations[slot] = handle.generation;
                originals[slot] = original;
                admissions[slot] = Some(Admission { handle, sampling: config.sampling_defaults.clone(),
                    max_new_tokens: 4, data: PromptData {
                        positions: (0..prompt.len()).map(|p| [p as i64; 3]).collect(),
                        vision_token_indices: vec![-1; prompt.len()], ..Default::default()
                    }, prompt, grammar_mask: None, ignore_eos: true, prompt_logprobs: false });
                handles.push(handle);
            }
            for handle in &handles {
                if !keep.contains(&(handle.slot as usize)) {
                    scheduler.cancel(*handle).unwrap();
                    admissions[handle.slot as usize] = None;
                }
            }
            let plan = scheduler.plan().unwrap();
            assert_eq!(plan.kind, GraphKind::Init);
            assert_eq!(plan.lanes.len(), keep.len());
            let graph = catalog::select_shape_with_drafts(&config, executor.manifest(), plan.kind,
                plan.batch_capacity, plan.token_capacity, plan.prefill_capacity(),
                plan.packed_token_capacity, plan.draft_tokens, SamplingMode::General).unwrap();
            let mut expected = vec![0i32; 8];
            for (row, lane) in plan.lanes.iter().enumerate() {
                let live = scheduler.request(lane.request).unwrap();
                assert_eq!(live.initial_prompt_tokens(), originals[lane.request.slot as usize]);
                expected[row] = originals[lane.request.slot as usize] as i32;
                assert_ne!(expected[row] as usize, live.prompt_tokens);
            }
            covered.insert((plan.batch_capacity, plan.lanes.len()));
            let fence = executor.next_fence().unwrap();
            scheduler.commit(&plan, fence).unwrap();
            pack_inputs(&mut inputs, &config, &scheduler, &plan, &admissions).unwrap();
            assert_eq!(executor.submit(graph, &inputs.uploads(), false).unwrap(), Some(fence));
            let cancelled = if cohort == cohorts.len() - 2 {
                let handle = plan.lanes[0].request;
                scheduler.cancel(handle).unwrap();
                assert!(scheduler.admit(RequestSpec { initial_prompt_tokens: 19, prompt: vec![2; 36],
                    prefix_scope: None, message_boundaries: vec![], conditioning: vec![], max_new_tokens: 4,
                    grammar_constrained: false, host_stops: false, prompt_logprobs: false,
                    growable_output: false }).is_err(), "queued initialization released its cancelled slot early");
                Some(handle)
            } else { None };
            let deadline = Instant::now() + Duration::from_secs(120);
            let completion = loop {
                assert!(Instant::now() < deadline, "native immutable input replay timed out");
                if let Some(completion) = executor.poll().unwrap() { break completion; }
                std::thread::yield_now();
            };
            assert_eq!(executor.outstanding(), 0);
            for view in [&input, &observed] {
                let bytes = sampling::download(&owner, 0, view).unwrap();
                let actual: Vec<_> = bytes.chunks_exact(4)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
                assert_eq!(actual, expected, "native input or captured observation differs at cohort {cohort}");
            }
            scheduler.complete(completion.fence, &completion.records, completion.timing).unwrap();
            for handle in handles {
                if keep.contains(&(handle.slot as usize)) && Some(handle) != cancelled {
                    scheduler.cancel(handle).unwrap();
                }
                admissions[handle.slot as usize] = None;
            }
        }
        assert!([1, 2, 4, 8].iter().all(|b| covered.contains(&(*b, *b))));
        assert!(covered.contains(&(4, 3)));
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        assert!(!maps.contains("libpython") && !maps.contains("libtorch"));
    }));
    let shutdown = executor.close();
    if !shutdown.drained {
        std::mem::forget(executor);
        std::mem::forget(owner);
        panic!("native immutable input replay did not drain; owners retained");
    }
    if let Err(error) = outcome { std::panic::resume_unwind(error); }
    assert!(shutdown.error.is_none(), "{shutdown:?}");
}
