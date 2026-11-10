/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! CPU ownership checks; scripted host events do not establish CUDA copy fences.

use super::*;
use super::super::super::PrefixCacheReuse;

#[test]
fn request_cache_reclaims_idle_full_partial_and_zero_tail_checkpoints() {
    for mode in [PrefixCacheReuse::Shared, PrefixCacheReuse::Request] {
        let mut fixture = Fixture::new();
        fixture.config.limits.pages = 128;
        let (_sender, receiver) = mpsc::channel();
        let mut worker = fixture.worker(receiver);
        worker.prefix_cache_reuse = mode;
        let before = worker.scheduler.stats();
        let ids = [1, 64, 70].map(|end| unpinned_checkpoint(&mut worker, end as u32, end));
        assert_eq!(worker.scheduler.cache().stats().published_checkpoints, 3);
        assert!(worker.scheduler.stats().cached_pages > 0);
        assert_eq!(worker.trim_request_cache(0).unwrap(), mode == PrefixCacheReuse::Request);
        if mode == PrefixCacheReuse::Request {
            let stats = worker.scheduler.stats();
            let cache = worker.scheduler.cache().stats();
            assert_eq!((cache.published_checkpoints, cache.pending_checkpoints,
                cache.owned_pages, cache.owned_snapshots, cache.retired_pages, cache.retired_snapshots),
                (0, 0, 0, 0, 0, 0));
            assert_eq!((stats.free_pages, stats.free_snapshots), (before.free_pages, before.free_snapshots));
            assert!(!worker.trim_request_cache(0).unwrap());
        } else {
            assert!(ids.iter().all(|id| worker.scheduler.cache().checkpoint(*id).is_ok()));
            let spec = RequestSpec { prefix_scope: None, prompt: vec![70; 71], initial_prompt_tokens: 71,
                message_boundaries: vec![], conditioning: vec![], max_new_tokens: 1,
                grammar_constrained: false, host_stops: false, prompt_logprobs: false, growable_output: false };
            assert_eq!(worker.scheduler.plan_admission(&spec).unwrap().resident_tokens(), 70);
        }
    }
}

#[test]
fn request_cache_preserves_queued_pending_and_active_consumers() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    worker.prefix_cache_reuse = PrefixCacheReuse::Request;
    let id = unpinned_checkpoint(&mut worker, 0, 64);
    assert!(!worker.trim_request_cache(1).unwrap());
    let (mut incoming, _generation) = fixture.incoming(vec![1; 70], 4);
    incoming.prefix_scope = Some([1; 24]);
    sender.send(vec![incoming]).unwrap();
    // The same path used by the idle worker observes the real receiver first.
    assert!(worker.trim_request_cache(0).unwrap());
    assert_eq!(worker.pending.len(), 1);
    assert!(worker.scheduler.cache().checkpoint(id).is_ok());
    assert!(!worker.trim_request_cache(0).unwrap());
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert!(worker.active.iter().any(Option::is_some));
    assert!(!worker.trim_request_cache(0).unwrap());
    assert!(worker.scheduler.cache().checkpoint(id).is_ok());
}

#[test]
fn request_cache_keeps_parked_history_until_the_same_generation_finishes() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (mut incoming, mut generation) = fixture.incoming(vec![0], 3);
    incoming.prefix_scope = Some([7; 24]);
    incoming.prompt.reasoning_cut = Some(budget_cut(4, 4));
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    worker.prefix_cache_reuse = PrefixCacheReuse::Request;
    let _ = worker.admit().unwrap();
    let mut sequence = 0;
    budget_pass(&mut worker, &mut sequence, &[1, 1, 1]);
    assert!(worker.active.iter().all(Option::is_none));
    assert_eq!(worker.parked.len(), 1);
    assert!(worker.scheduler.cache().stats().published_checkpoints > 0);
    assert!(!worker.trim_request_cache(0).unwrap());
    assert_eq!(worker.parked.front().unwrap().0.prefix_scope, Some([7; 24]));
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let handle = worker.active.iter().flatten().next().unwrap().request;
    assert!(worker.scheduler.request(handle).unwrap().reused_prompt_tokens() > 0);
    budget_pass(&mut worker, &mut sequence, &[2, 2, 2]);
    let (tokens, terminal, _) = budget_events(&mut generation);
    assert_eq!(tokens, [1, 1, 1, 4, 2, 2, 2]);
    assert_eq!(terminal, [(1, 7, "length")]);
    assert!(worker.parked.is_empty() && worker.active.iter().all(Option::is_none));
    assert!(worker.trim_request_cache(0).unwrap());
    assert_eq!(worker.scheduler.cache().stats().owned_snapshots, 0);
}

#[test]
fn request_cache_waits_for_cancelled_submission_retirement() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    worker.prefix_cache_reuse = PrefixCacheReuse::Request;
    let id = unpinned_checkpoint(&mut worker, 0, 64);
    let (mut incoming, generation) = fixture.incoming(vec![1; 70], 4);
    incoming.prefix_scope = Some([1; 24]);
    sender.send(vec![incoming]).unwrap();
    let _ = worker.admit().unwrap();
    let step = planned(&mut worker, GraphKind::Init, 1);
    let frame = completion(&worker, &step, 1, None);
    drop(generation);
    let _ = worker.admit().unwrap();
    assert!(worker.active.iter().all(Option::is_none));
    assert_eq!(worker.scheduler.stats().retiring_slots, 1);
    assert!(!worker.trim_request_cache(0).unwrap());
    assert!(worker.scheduler.cache().checkpoint(id).is_ok());
    worker.complete(GraphKind::Init, frame).unwrap();
    assert_eq!(worker.scheduler.stats().retiring_slots, 0);
    assert!(worker.trim_request_cache(0).unwrap());
}

#[test]
fn request_cache_waits_for_publication_copy_pin_release() {
    for success in [false, true] {
        let fixture = Fixture::new();
        let (_sender, receiver) = mpsc::channel();
        let mut worker = fixture.worker(receiver);
        worker.prefix_cache_reuse = PrefixCacheReuse::Request;
        let mut store = FakeStore::new(None);
        store.store_outcomes.push_back(flare_selene::StoreOutcome::Queued);
        worker.attach_store(Some(Box::new(store)));
        let id = unpinned_checkpoint(&mut worker, 0, 64);
        worker.service_store().unwrap();
        assert_eq!(pins(&worker, id), 1);
        assert!(!worker.trim_request_cache(0).unwrap());
        complete_store(&mut worker, id, success);
        worker.service_store().unwrap();
        assert_eq!(pins(&worker, id), 0);
        assert!(worker.trim_request_cache(0).unwrap());
        assert_eq!(worker.scheduler.cache().stats().owned_snapshots, 0);
    }
}
