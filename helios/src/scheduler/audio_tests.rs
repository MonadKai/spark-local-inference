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
fn test_phase_selection_uses_real_encoder_capacities_and_readiness() {
    let mut scheduler = scheduler(1);
    assert!(scheduler.try_phase_plan(GraphKind::AudioEncode, None, None).is_none());
    let handle = scheduler.admit(tests::spec(120, 2)).unwrap();
    scheduler.set_conditioned_inputs(handle, &[input()]).unwrap();
    let (plan, fence, records) = commit(&mut scheduler);
    assert_eq!(plan.kind, GraphKind::Init);
    complete(&mut scheduler, &plan, fence, &records);
    assert!(scheduler.try_phase_plan(GraphKind::AudioEncode, Some(handle), None).is_none());
    scheduler.conditioned_inputs_ready(handle, true).unwrap();
    assert!(scheduler.try_phase_plan(GraphKind::AudioEncode, Some(handle), Some(2)).is_none());
    for frames in [800, 1] {
        let plan = scheduler.try_phase_plan(GraphKind::AudioEncode, Some(handle), Some(1)).unwrap();
        // Both catalog entries have equal synthetic durations here. The
        // selector promises a fitting graph, not the smallest frame capacity.
        assert_eq!(plan.batch_capacity, 1);
        assert!([100, 800].contains(&plan.token_capacity));
        assert!(plan.token_capacity >= frames);
        assert_eq!(plan.lanes.len(), 1);
        assert_eq!((plan.lanes[0].request, plan.lanes[0].tokens), (handle, 0));
        let fence = Fence(scheduler.submitted.0 + 1);
        scheduler.commit(&plan, fence).unwrap();
        let mut records = vec![CompletionRecord::default(); scheduler.limits.request_slots];
        records[handle.slot as usize] = CompletionRecord {
            generation: handle.generation as i64, submission_sequence: fence.0 as i64,
            consumed_tokens: 0, generated_tokens: 0, status: 1, token_id: -1,
            ..Default::default()
        };
        complete(&mut scheduler, &plan, fence, &records);
    }
    assert!(scheduler.try_phase_plan(GraphKind::AudioEncode, Some(handle), None).is_none());
    assert_eq!(scheduler.request(handle).unwrap().phase, RequestPhase::Prefill);
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

fn initialized(scheduler: &mut RequestScheduler) -> RequestHandle {
    let handle = admit(scheduler);
    let (plan, fence, records) = commit(scheduler);
    complete(scheduler, &plan, fence, &records);
    handle
}

#[test]
fn retained_input_retires_only_its_prerequisite_without_fabricating_a_completion() {
    let mut scheduler = scheduler(1);
    let handle = initialized(&mut scheduler);
    let plan = scheduler.plan().unwrap();
    let live = scheduler.request(handle).unwrap();
    let before = (live.consumed_tokens, live.generated_tokens, live.scheduled_consumed,
        live.reused_prompt_tokens(), live.last_use, scheduler.submitted, scheduler.completed);
    let ticket = scheduler.conditioned_prerequisite(handle).unwrap();
    assert_eq!(ticket.input().token_span, 1..106);
    assert_ne!(ticket.input().token_span.end, 120); // window includes trailing text
    assert_eq!(ticket.feature_slot(), 0);
    scheduler.complete_conditioned_prerequisite(ticket).unwrap();
    let live = scheduler.request(handle).unwrap();
    assert_eq!(before, (live.consumed_tokens, live.generated_tokens, live.scheduled_consumed,
        live.reused_prompt_tokens(), live.last_use, scheduler.submitted, scheduler.completed));
    assert_eq!(live.phase, RequestPhase::Prefill);
    assert!(live.encoder_jobs.is_empty());
    assert_eq!(live.audio_slot, Some(0));
    assert_eq!(scheduler.plan().unwrap().kind, GraphKind::Prefill);
    assert!(scheduler.commit(&plan, Fence(scheduler.submitted.0 + 1)).is_err());
    assert!(scheduler.conditioned_prerequisite(handle).is_err());
    scheduler.cancel(handle).unwrap();
    let next = initialized(&mut scheduler);
    assert_eq!(next.slot, handle.slot);
    assert!(next.generation > handle.generation);
    assert!(scheduler.conditioned_prerequisite(handle).is_err());
    assert_eq!(scheduler.conditioned_prerequisite(next).unwrap().feature_slot(), 0);
}

#[test]
fn stale_or_mutated_prerequisites_do_not_change_epochs_jobs_or_plans() {
    let mut scheduler = scheduler(1);
    let handle = initialized(&mut scheduler);
    for mutation in 0..7 {
        let mut ticket = scheduler.conditioned_prerequisite(handle).unwrap();
        match mutation {
            0 => ticket.owner = Arc::new(()),
            1 => ticket.state.epoch += 1,
            2 => ticket.state.feature_slot += 1,
            3 => ticket.state.input.token_span.end += 1,
            4 => ticket.state.input.feature_range.end += 1,
            5 => { ticket.state.input.encodings.pop(); },
            6 => ticket.state.window_inputs.end += 1,
            _ => unreachable!(),
        }
        let before = scheduler.prerequisite_state(handle).unwrap();
        assert!(scheduler.complete_conditioned_prerequisite(ticket).is_err(), "mutation {mutation}");
        assert_eq!(scheduler.prerequisite_state(handle).unwrap(), before);
    }
    let ticket = scheduler.conditioned_prerequisite(handle).unwrap();
    scheduler.conditioned_inputs_ready(handle, false).unwrap();
    scheduler.conditioned_inputs_ready(handle, true).unwrap();
    let before = scheduler.prerequisite_state(handle).unwrap();
    assert!(scheduler.complete_conditioned_prerequisite(ticket).is_err());
    assert_eq!(scheduler.prerequisite_state(handle).unwrap(), before);
}

#[test]
fn prerequisite_rejects_uninitialized_pending_partial_and_retired_work() {
    let mut scheduler = scheduler(1);
    let handle = admit(&mut scheduler);
    assert!(scheduler.conditioned_prerequisite(handle).is_err());
    let (plan, fence, records) = commit(&mut scheduler);
    assert!(scheduler.conditioned_prerequisite(handle).is_err());
    complete(&mut scheduler, &plan, fence, &records);
    let ticket = scheduler.conditioned_prerequisite(handle).unwrap();
    let (plan, fence, records) = commit(&mut scheduler);
    let before = scheduler.request(handle).unwrap().epoch;
    assert!(scheduler.complete_conditioned_prerequisite(ticket).is_err());
    assert_eq!(scheduler.request(handle).unwrap().epoch, before);
    assert!(scheduler.conditioned_prerequisite(handle).is_err());
    complete(&mut scheduler, &plan, fence, &records);
    assert_eq!(scheduler.request(handle).unwrap().encoder_jobs.len(), 1);
    assert!(scheduler.conditioned_prerequisite(handle).is_err());
    scheduler.cancel(handle).unwrap();
    assert!(scheduler.conditioned_prerequisite(handle).is_err());
}

#[test]
fn whole_input_prerequisite_preserves_later_conditioned_windows() {
    let mut scheduler = tests::scheduler(2, 128, 8);
    scheduler.capabilities.vision.features_per_request = 64;
    let handle = scheduler.admit(tests::spec(150, 2)).unwrap();
    let images = [ConditionedInput { token_span: 1..65, feature_range: 0..64,
        encodings: vec![EncoderJob::Vision { patches: 256 }] },
        ConditionedInput { token_span: 70..134, feature_range: 64..128,
        encodings: vec![EncoderJob::Vision { patches: 256 }] }];
    scheduler.set_conditioned_inputs(handle, &images).unwrap();
    scheduler.conditioned_inputs_ready(handle, true).unwrap();
    let (plan, fence, records) = commit(&mut scheduler);
    complete(&mut scheduler, &plan, fence, &records);
    let ticket = scheduler.conditioned_prerequisite(handle).unwrap();
    scheduler.complete_conditioned_prerequisite(ticket).unwrap();
    let live = scheduler.request(handle).unwrap();
    assert_eq!(live.conditioned_windows.len(), 2);
    assert!(live.encoder_jobs.is_empty());
    assert_eq!(live.conditioned_windows[1].encodings.iter().copied().collect::<Vec<_>>(), images[1].encodings);
    assert!(live.window_snapshot.is_some());
    assert_eq!(live.phase, RequestPhase::Prefill);
}
