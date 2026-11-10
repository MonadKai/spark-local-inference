/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

use super::*;
use crate::model::contract::{AudioCapacity, StateView};

pub(super) fn scheduler(slots: usize) -> RequestScheduler {
    let mut scheduler = tests::scheduler(2, 128, 8);
    let view = StateView { buffer: "unused-by-scheduler".into(), offset: 0, stride: 1, nbytes: 1 };
    scheduler.capabilities.audio = Some(AudioCapacity {
        feature_slots: slots, features_per_request: 7800, feature_width: 2048,
        frame_capacities: vec![100, 800], mel_bins: 128, frames_per_chunk: 100,
        features_per_chunk: 13, subsampling_factor: 8, attention_window_frames: 800,
        features: view.clone(), generations: view.clone(), counts: view,
    });
    scheduler.free_audio_slots = (0..slots).collect();
    let mut graphs = tests::duration_graphs(2);
    let template = graphs.iter().find(|graph| graph.kind == GraphKind::Vision).unwrap().clone();
    for frames in [100, 800] {
        let mut graph = template.clone();
        graph.name = format!("audio{frames}");
        graph.kind = GraphKind::AudioEncode;
        graph.token_capacity = frames;
        graphs.push(graph);
    }
    scheduler.measured_graph_durations(&graphs).unwrap();
    scheduler
}

pub(super) fn input() -> ConditionedInput {
    ConditionedInput { token_span: 1..106, feature_range: 0..105,
        encodings: vec![EncoderJob::Audio { frames: 800, features: 104 },
                        EncoderJob::Audio { frames: 1, features: 1 }] }
}

fn admit(scheduler: &mut RequestScheduler) -> RequestHandle {
    let handle = scheduler.admit(tests::spec(120, 2)).unwrap();
    scheduler.set_conditioned_inputs(handle, &[input()]).unwrap();
    scheduler.conditioned_inputs_ready(handle, true).unwrap();
    handle
}

fn commit(scheduler: &mut RequestScheduler) -> (ScheduledPlan, Fence, Vec<CompletionRecord>) {
    let plan = scheduler.plan().unwrap();
    let fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&plan, fence).unwrap();
    let mut records = vec![CompletionRecord::default(); scheduler.limits.request_slots];
    for lane in &plan.lanes {
        let live = scheduler.request(lane.request).unwrap();
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64, submission_sequence: fence.0 as i64,
            consumed_tokens: live.consumed_tokens as i64, generated_tokens: live.generated_tokens as i64,
            status: 1, token_id: -1, ..Default::default()
        };
    }
    (plan, fence, records)
}

fn complete(scheduler: &mut RequestScheduler, plan: &ScheduledPlan, fence: Fence, records: &[CompletionRecord]) {
    scheduler.advance_to(scheduler.now_ns + plan.estimated_duration_ns()).unwrap();
    scheduler.complete(fence, records, Default::default()).unwrap();
}

#[test]
fn audio_windows_finish_before_text_without_advancing_tokens_or_generation() {
    let mut scheduler = scheduler(1);
    let handle = admit(&mut scheduler);
    let arena = scheduler.request(handle).unwrap().audio_slot;
    for kind in [GraphKind::Init, GraphKind::AudioEncode, GraphKind::AudioEncode] {
        let (plan, fence, records) = commit(&mut scheduler);
        assert_eq!(plan.kind, kind);
        assert_eq!(plan.lanes[0].tokens, 0);
        complete(&mut scheduler, &plan, fence, &records);
        let live = scheduler.request(handle).unwrap();
        assert_eq!(live.handle, handle);
        assert_eq!(live.audio_slot, arena);
        assert_eq!((live.consumed_tokens, live.generated_tokens), (0, 0));
    }
    let live = scheduler.request(handle).unwrap();
    assert!(live.encoder_jobs.is_empty());
    assert_eq!(live.phase, RequestPhase::Prefill);
    assert_eq!(scheduler.plan().unwrap().kind, GraphKind::Prefill);
    assert_eq!(live.feature_index(104).unwrap(), 104);
}

#[test]
fn cancellation_holds_audio_arena_until_last_encoder_fence_and_reuses_with_new_epoch() {
    let mut scheduler = scheduler(1);
    let first = admit(&mut scheduler);
    let (plan, fence, records) = commit(&mut scheduler);
    complete(&mut scheduler, &plan, fence, &records);
    let (plan, fence, records) = commit(&mut scheduler);
    assert_eq!(plan.kind, GraphKind::AudioEncode);
    scheduler.cancel(first).unwrap();
    assert!(scheduler.free_audio_slots.is_empty());
    complete(&mut scheduler, &plan, fence, &records);
    assert_eq!(scheduler.free_audio_slots.len(), 1);
    assert!(scheduler.conditioned_inputs_ready(first, true).is_err());
    let second = admit(&mut scheduler);
    assert_eq!(second.slot, first.slot);
    assert!(second.generation > first.generation);
    assert!(scheduler.request(first).is_err());
    assert_eq!(scheduler.request(second).unwrap().audio_slot, Some(0));
}

#[test]
fn malformed_audio_jobs_do_not_reserve_an_arena() {
    let mut scheduler = scheduler(1);
    let handle = scheduler.admit(tests::spec(300, 2)).unwrap();
    let good = input();
    let mut second = input();
    second.token_span = 130..235;
    second.feature_range = 105..210;
    assert!(scheduler.set_conditioned_inputs(handle, &[good.clone(), second]).is_err());
    for job in [EncoderJob::Audio { frames: 801, features: 104 },
                EncoderJob::Audio { frames: 0, features: 104 },
                EncoderJob::Audio { frames: 800, features: 0 },
                EncoderJob::Audio { frames: 800, features: 103 },
                EncoderJob::Vision { patches: 800 }] {
        let mut bad = good.clone();
        bad.encodings[0] = job;
        assert!(scheduler.set_conditioned_inputs(handle, &[bad]).is_err());
        assert_eq!(scheduler.free_audio_slots.len(), 1);
    }
    assert_eq!(scheduler.free_audio_slots.len(), 1);
}

#[test]
fn encoder_failure_stops_remaining_work_and_releases_only_after_retirement() {
    let mut scheduler = scheduler(1);
    let handle = admit(&mut scheduler);
    let (plan, fence, records) = commit(&mut scheduler);
    complete(&mut scheduler, &plan, fence, &records);
    let (plan, fence, mut records) = commit(&mut scheduler);
    assert_eq!(plan.kind, GraphKind::AudioEncode);
    records[handle.slot as usize].status = 2;
    assert!(scheduler.complete(fence, &records, Default::default()).is_err());
    records[handle.slot as usize].status = 3;
    records[handle.slot as usize].error_code = 64;
    complete(&mut scheduler, &plan, fence, &records);
    let live = scheduler.request(handle).unwrap();
    assert_eq!(live.phase, RequestPhase::Finishing);
    assert_eq!(live.error_code, Some(64));
    assert_eq!(live.audio_slot, Some(0));
    assert_eq!((live.consumed_tokens, live.generated_tokens), (0, 0));
    assert!(scheduler.plan().is_none());
    assert!(scheduler.free_audio_slots.is_empty());
    scheduler.cancel(handle).unwrap();
    assert!(scheduler.request(handle).is_err());
    assert_eq!(scheduler.free_audio_slots.len(), 1);
}
