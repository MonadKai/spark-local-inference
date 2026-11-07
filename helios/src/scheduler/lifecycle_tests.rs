/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use std::collections::HashSet;

type Projection = [u64; 14];
type Action = [u64; 3];

const MAX_SUBMISSIONS: u64 = 4;
const MAX_DEPTH: usize = 12;

struct Fixture {
    scheduler: RequestScheduler,
    fence_origin: u64,
    last_plan: Option<ScheduledPlan>,
    kind: FixtureKind,
    rejected_admission: Option<Projection>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FixtureKind { Text, AudioStart, AudioDecode }

fn fixture_spec(kind: FixtureKind, slot: usize, generation: u64) -> RequestSpec {
    let mut spec = request_spec(slot, generation);
    if kind != FixtureKind::Text {
        spec.prompt.resize(120, 7);
        spec.conditioning.push(ConditioningEvent { position: 1, identity: [generation as u8; 32] });
    }
    spec
}

fn request_spec(slot: usize, generation: u64) -> RequestSpec {
    let mut spec = tests::spec(63, 32);
    spec.prompt[0] = 100 + slot as u32 + generation as u32 * 10;
    spec
}

fn completion_records(scheduler: &RequestScheduler, fence: Fence) -> Vec<CompletionRecord> {
    let submission = &scheduler.submissions[&fence];
    let mut records = vec![CompletionRecord::default(); 2];
    for (row, lane) in submission.lanes.iter().enumerate() {
        let Ok(request) = scheduler.request(lane.request) else {
            continue;
        };
        let kind = submission.lane_kind(row);
        let transfer = matches!(
            kind,
            GraphKind::Init | GraphKind::Restore | GraphKind::Snapshot | GraphKind::AudioEncode
        );
        assert!(!matches!(
            kind,
            GraphKind::Speculative | GraphKind::Mixed | GraphKind::Vision
        ));
        let consumed = if transfer {
            request.consumed_tokens
        } else {
            lane.start_position + lane.tokens as u64
        };
        let emits = !transfer && (!kind.is_prefill() || consumed >= request.prompt_tokens as u64);
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            status: 1,
            consumed_tokens: consumed as i64,
            generated_tokens: i64::from(request.generated_tokens) + i64::from(emits),
            token_id: if emits { 99 } else { -1 },
            ..Default::default()
        };
    }
    records
}

impl Fixture {
    fn new() -> Self {
        Self::new_kind(FixtureKind::Text)
    }

    fn new_kind(kind: FixtureKind) -> Self {
        let mut scheduler = if kind == FixtureKind::Text { tests::scheduler(2, 64, 16) }
                            else { audio_tests::scheduler(1) };
        if kind == FixtureKind::AudioDecode {
            // Seed a valid ownership state with the public policy API. Uniform
            // synthetic microsecond costs do not model serving fairness.
            scheduler.configure_policy(PolicyConfig { output_token_value: 1., token_fairness: 0.,
                delivery_fairness: 0., decode_age_price: 0., ..Default::default() }).unwrap();
        }
        for slot in 0..2 {
            let handle = scheduler.admit(fixture_spec(kind, slot, 1)).unwrap();
            assert_eq!(handle.slot as usize, slot);
            if kind != FixtureKind::Text && slot == 0 {
                scheduler.set_conditioned_inputs(handle, &[audio_tests::input()]).unwrap();
                scheduler.conditioned_inputs_ready(handle, true).unwrap();
            }
        }
        if kind == FixtureKind::AudioStart {
            return Self { scheduler, fence_origin: 0, last_plan: None, kind, rejected_admission: None };
        }
        for _ in 0..16 {
            if scheduler
                .slots
                .iter()
                .all(|slot| slot.request.as_ref().unwrap().phase == RequestPhase::Decode)
            {
                let fence_origin = scheduler.submitted.0;
                assert_eq!(scheduler.completed.0, fence_origin);
                return Self {
                    scheduler,
                    fence_origin,
                    last_plan: None,
                    kind, rejected_admission: None,
                };
            }
            let plan = scheduler
                .plan()
                .expect("warmup must reach decode through real scheduler plans");
            let fence = Fence(scheduler.submitted.0 + 1);
            scheduler.commit(&plan, fence).unwrap();
            let records = completion_records(&scheduler, fence);
            let mut timing = crate::execution::ExecutionTiming::default();
            if kind != FixtureKind::Text {
                scheduler.advance_to(scheduler.now_ns + plan.estimated_duration_ns()).unwrap();
                timing.gpu_ns[0] = Some(plan.estimated_duration_ns());
                timing.wall_ns = plan.estimated_duration_ns();
            }
            let generated = scheduler
                .complete(fence, &records, timing)
                .unwrap();
            if kind != FixtureKind::Text {
                for token in generated { scheduler.published_output(token.request, fence).unwrap(); }
            }
        }
        panic!("scheduler warmup did not reach two decode requests: {kind:?}");
    }

    fn projection(&self) -> Projection {
        Self::project(&self.scheduler, self.fence_origin)
    }

    fn project(s: &RequestScheduler, origin: u64) -> Projection {
        let logical = |fence: Fence| fence.0.saturating_sub(origin);
        let mut result = [0; 14];
        result[0] = logical(s.submitted);
        result[1] = logical(s.completed);
        for (index, slot) in s.slots.iter().enumerate() {
            result[2 + index] = if slot.request.is_some() {
                1
            } else if slot.retiring {
                2
            } else {
                0
            };
            result[4 + index] = slot.generation;
            result[6 + index] = slot
                .request
                .as_ref()
                .map(|request| request.last_use)
                .or_else(|| {
                    s.retired
                        .iter()
                        .find(|retired| retired.slot == index)
                        .map(|retired| retired.through)
                })
                .map_or(0, logical);
        }
        for (index, (fence, submission)) in s.submissions.iter().enumerate() {
            assert!(index < 2);
            result[8 + index] = logical(*fence);
            result[10 + index] = submission
                .lanes
                .iter()
                .fold(0, |mask, lane| mask | (1 << lane.request.slot));
            result[12 + index] = u64::from(submission.requires_feedback);
        }
        result
    }

    fn state_key(&self) -> String {
        Self::scheduler_key(&self.scheduler, self.fence_origin)
    }

    fn scheduler_key(s: &RequestScheduler, fence_origin: u64) -> String {
        let live = s
            .slots
            .iter()
            .map(|slot| {
                slot.request.as_ref().map(|request| {
                    (
                        request.handle,
                        request.phase,
                        request.epoch,
                        request.consumed_tokens,
                        request.generated_tokens,
                        request.scheduled_consumed,
                        request.pending,
                        (request.audio_slot, &request.encoder_jobs, request.conditioned_inputs_ready,
                         &request.conditioned_windows, &request.conditioned_features, request.window_restore),
                        (
                            request.private_pages.clone(),
                            request.reserved_pages.clone(),
                            request.terminal_snapshot,
                            request.prompt_snapshot,
                            request.prefix,
                            request.admission_order,
                        ),
                    )
                })
            })
            .collect::<Vec<_>>();
        let submitted = s
            .submissions
            .iter()
            .map(|(fence, submission)| (*fence, &submission.step))
            .collect::<Vec<_>>();
        let retired = s
            .retired
            .iter()
            .map(|retired| {
                (
                    retired.slot,
                    retired.through,
                    &retired.pages,
                    retired.snapshots,
                    retired.audio_slot,
                )
            })
            .collect::<Vec<_>>();
        format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
            Self::project(s, fence_origin),
            live,
            submitted,
            retired,
            s.stats(),
            s.free_pages,
            s.free_snapshots,
            s.free_audio_slots
        )
    }

    fn assert_ownership(&self) {
        Self::assert_scheduler_ownership(&self.scheduler);
    }

    fn assert_scheduler_ownership(s: &RequestScheduler) {
        for (fence, submission) in &s.submissions {
            for lane in &submission.lanes {
                let slot = &s.slots[lane.request.slot as usize];
                assert_eq!(
                    slot.generation, lane.request.generation,
                    "queued request generation was reused before completion"
                );
                assert!(
                    slot.request
                        .as_ref()
                        .is_some_and(|request| request.handle == lane.request)
                        || s.retired.iter().any(|retired| {
                            retired.slot == lane.request.slot as usize && retired.through >= *fence
                        }),
                    "queued request lost ownership before completion"
                );
            }
        }
        let mut pages = s.free_pages.iter().copied().collect::<HashSet<_>>();
        let mut snapshots = s.free_snapshots.iter().copied().collect::<HashSet<_>>();
        let capacity = s.capabilities.audio.as_ref().map_or(0, |audio| audio.feature_slots);
        let mut arenas = s.free_audio_slots.iter().copied().collect::<HashSet<_>>();
        assert!(arenas.iter().all(|arena| *arena < capacity));
        for (index, slot) in s.slots.iter().enumerate() {
            assert!(!(slot.retiring && slot.request.is_some()));
            assert_eq!(
                slot.retiring,
                s.retired.iter().any(|retired| retired.slot == index)
            );
            if let Some(request) = &slot.request {
                if let Some(arena) = request.audio_slot {
                    assert!(arena < capacity && arenas.insert(arena), "live audio arena aliases free or owned storage");
                }
                assert_eq!(request.handle.generation, slot.generation);
                assert_eq!(
                    request.pending,
                    s.submissions
                        .values()
                        .filter(|submission| submission
                            .lanes
                            .iter()
                            .any(|lane| lane.request == request.handle))
                        .count()
                );
                for page in request.private_pages.iter().chain(&request.reserved_pages) {
                    assert!(
                        pages.insert(*page),
                        "live private page aliases free or owned storage"
                    );
                }
                for snapshot in [
                    request.terminal_snapshot,
                    request.message_snapshot,
                    request.prompt_snapshot,
                    request.window_snapshot,
                ]
                .into_iter()
                .flatten()
                {
                    assert!(
                        snapshots.insert(snapshot),
                        "live snapshot aliases free or owned storage"
                    );
                }
            }
        }
        for retired in &s.retired {
            if let Some(arena) = retired.audio_slot {
                assert!(arena < capacity && arenas.insert(arena), "retired audio arena became reusable before its fence");
            }
            assert!(
                retired.through > s.completed,
                "completed retirement still reserves a slot"
            );
            for page in &retired.pages {
                assert!(
                    pages.insert(*page),
                    "retired page became reusable before its fence"
                );
            }
            for snapshot in retired.snapshots.into_iter().flatten() {
                assert!(
                    snapshots.insert(snapshot),
                    "retired snapshot became reusable before its fence"
                );
            }
        }
        assert_eq!(arenas.len(), capacity, "audio arena ownership was lost");
    }

    fn actions(&self) -> Vec<Action> {
        let s = &self.scheduler;
        let mut actions = vec![[5, 0, 0]];
        if let Some(slot) = s
            .slots
            .iter()
            .position(|slot| slot.request.is_none() && !slot.retiring)
            && s.slots[slot].generation < 2
        {
            actions.push([1, slot as u64, 0]);
        }
        if self.projection()[0] < MAX_SUBMISSIONS
            && let Some(plan) = s.plan()
        {
            actions.push([
                2,
                plan.lanes
                    .iter()
                    .fold(0, |mask, lane| mask | (1 << lane.request.slot)),
                u64::from(s.plan_requires_feedback(&plan)),
            ]);
        }
        if !s.submissions.is_empty() {
            actions.push([3, 0, 0]);
        }
        for (index, slot) in s.slots.iter().enumerate() {
            if let Some(request) = &slot.request {
                actions.push([4, index as u64, 0]);
                if s.conditioned_prerequisite(request.handle).is_ok() {
                    actions.push([7, index as u64, 0]);
                }
            }
        }
        if s.slots
            .iter()
            .any(|slot| slot.request.is_some() || slot.retiring)
            || !s.submissions.is_empty()
        {
            actions.push([6, 0, 0]);
        }
        actions
    }

    fn apply(&mut self, action: Action) {
        self.rejected_admission = None;
        let before = self.state_key();
        let s = &mut self.scheduler;
        match action[0] {
            1 => {
                let slot = action[1] as usize;
                let handle = s
                    .admit(fixture_spec(self.kind, slot, s.slots[slot].generation + 1))
                    .unwrap();
                assert_eq!(handle.slot as usize, slot);
                if self.kind != FixtureKind::Text {
                    let mid = Self::project(s, self.fence_origin);
                    let before_binding = Self::scheduler_key(s, self.fence_origin);
                    // The single worker binds input immediately after admission.
                    // Failed binding cancels in the same worker turn; its two
                    // observable ownership transitions are checked separately.
                    if let Err(error) = s.set_conditioned_inputs(handle, &[audio_tests::input()]) {
                        assert!(matches!(error, crate::Error::Admission(crate::AdmissionError::Capacity(_))));
                        assert_eq!(Self::scheduler_key(s, self.fence_origin), before_binding,
                            "failed audio binding changed resource ownership or input state");
                        Self::assert_scheduler_ownership(s);
                        s.cancel(handle).unwrap();
                        self.rejected_admission = Some(mid);
                    } else {
                        s.conditioned_inputs_ready(handle, true).unwrap();
                    }
                }
            }
            2 => {
                let plan = s.plan().unwrap();
                assert_eq!(
                    action[1],
                    plan.lanes
                        .iter()
                        .fold(0, |mask, lane| mask | (1 << lane.request.slot))
                );
                assert_eq!(action[2], u64::from(s.plan_requires_feedback(&plan)));
                s.commit(&plan, Fence(s.submitted.0 + 1)).unwrap();
                self.last_plan = Some(plan);
            }
            3 => {
                let fence = *s.submissions.first_key_value().unwrap().0;
                let records = completion_records(s, fence);
                let protected = s
                    .retired
                    .iter()
                    .filter(|retired| retired.through > fence)
                    .map(|retired| {
                        (
                            retired.slot,
                            retired.through,
                            retired.pages.clone(),
                            retired.snapshots,
                            retired.audio_slot,
                        )
                    })
                    .collect::<Vec<_>>();
                let audio = (s.submissions[&fence].kind == GraphKind::AudioEncode).then(|| {
                    let lane = &s.submissions[&fence].lanes[0];
                    s.request(lane.request).ok().map(|request| (request.handle, request.consumed_tokens,
                        request.generated_tokens, request.scheduled_consumed, request.encoder_jobs.len(), request.audio_slot, request.epoch))
                }).flatten();
                let generated = s.complete(fence, &records, Default::default()).unwrap();
                if let Some((handle, consumed, emitted, scheduled, jobs, arena, epoch)) = audio {
                    let live = s.request(handle).unwrap();
                    assert!(generated.is_empty(), "audio encoding emitted a token");
                    assert_eq!((live.consumed_tokens, live.generated_tokens, live.scheduled_consumed), (consumed, emitted, scheduled));
                    assert_eq!(live.audio_slot, arena, "audio encoding released features before text prefill");
                    assert_eq!(live.handle, handle, "audio completion changed the request generation");
                    assert!(live.epoch > epoch, "audio completion did not invalidate prior scheduler plans");
                    assert_eq!(live.encoder_jobs.len() + 1, jobs, "audio completion did not retire exactly one encoder job");
                }
                for (slot, through, pages, snapshots, arena) in protected {
                    let retired = s
                        .retired
                        .iter()
                        .find(|retired| retired.slot == slot)
                        .expect(
                            "first completion released a slot still referenced by a later frame",
                        );
                    assert_eq!(
                        (retired.through, &retired.pages, retired.snapshots),
                        (through, &pages, snapshots)
                    );
                    assert_eq!(retired.audio_slot, arena, "first completion lost a later frame's audio arena");
                    assert!(
                        pages.iter().all(|page| !s.free_pages.contains(page)),
                        "first completion released pages referenced by a later frame"
                    );
                    assert!(
                        snapshots
                            .into_iter()
                            .flatten()
                            .all(|snapshot| !s.free_snapshots.contains(&snapshot)),
                        "first completion released snapshots referenced by a later frame"
                    );
                }
            }
            4 => {
                let slot = action[1] as usize;
                let request = s.slots[slot].request.as_ref().unwrap();
                let handle = request.handle;
                let through = request.last_use;
                let arena = request.audio_slot;
                let pages = request
                    .private_pages
                    .iter()
                    .chain(&request.reserved_pages)
                    .copied()
                    .collect::<Vec<_>>();
                let snapshots = [
                    request.terminal_snapshot,
                    request.message_snapshot,
                    request.prompt_snapshot,
                    request.window_snapshot,
                ];
                s.cancel(handle).unwrap();
                if through > s.completed {
                    let retired = s
                        .retired
                        .iter()
                        .find(|retired| retired.slot == slot)
                        .expect("cancel released a slot referenced by queued work");
                    assert_eq!(
                        retired.through, through,
                        "cancel shortened the last-use fence"
                    );
                    assert_eq!(retired.pages, pages, "cancel lost private page ownership");
                    assert_eq!(retired.audio_slot, arena, "cancel lost audio arena ownership");
                    assert_eq!(
                        retired.snapshots, snapshots,
                        "cancel lost snapshot ownership"
                    );
                    assert!(
                        pages.iter().all(|page| !s.free_pages.contains(page)),
                        "cancel released pages referenced by queued work"
                    );
                    assert!(
                        snapshots
                            .into_iter()
                            .flatten()
                            .all(|snapshot| !s.free_snapshots.contains(&snapshot)),
                        "cancel released snapshots referenced by queued work"
                    );
                }
            }
            5 => {
                assert!(
                    s.release_after_shutdown(&Shutdown {
                        drained: false,
                        error: Some("uncertain stream drain".into())
                    })
                    .is_err(),
                    "unconfirmed drain accepted resource release"
                );
                assert_eq!(
                    self.state_key(),
                    before,
                    "unconfirmed drain released ownership"
                );
            }
            6 => {
                let mut pages = Vec::new();
                let mut snapshots = Vec::new();
                for request in s.slots.iter().filter_map(|slot| slot.request.as_ref()) {
                    pages.extend(
                        request
                            .private_pages
                            .iter()
                            .chain(&request.reserved_pages)
                            .copied(),
                    );
                    snapshots.extend(
                        [
                            request.terminal_snapshot,
                            request.message_snapshot,
                            request.prompt_snapshot,
                            request.window_snapshot,
                        ]
                        .into_iter()
                        .flatten(),
                    );
                }
                for retired in &s.retired {
                    pages.extend(retired.pages.iter().copied());
                    snapshots.extend(retired.snapshots.into_iter().flatten());
                }
                s.release_after_shutdown(&Shutdown {
                    drained: true,
                    error: Some("executor failure after drain".into()),
                })
                .unwrap();
                assert!(
                    pages.iter().all(|page| s.free_pages.contains(page)),
                    "confirmed drain lost private pages"
                );
                assert!(
                    snapshots
                        .iter()
                        .all(|snapshot| s.free_snapshots.contains(snapshot)),
                    "confirmed drain lost private snapshots"
                );
            }
            7 => {
                let handle = s.slots[action[1] as usize].request.as_ref().unwrap().handle;
                let live = s.request(handle).unwrap();
                let preserved = (live.handle, live.audio_slot, live.image_slot, live.consumed_tokens,
                    live.generated_tokens, live.scheduled_consumed, live.reused_prompt_tokens(), live.last_use,
                    live.pending, live.private_pages.clone(), live.reserved_pages.clone());
                let epoch = live.epoch;
                let ownership = (Self::project(s, self.fence_origin), s.stats(), s.free_pages.clone(),
                    s.free_snapshots.clone(), s.free_audio_slots.clone());
                let ticket = s.conditioned_prerequisite(handle).unwrap();
                // The external copy has completed; it creates no graph fence.
                s.complete_conditioned_prerequisite(ticket).unwrap();
                let live = s.request(handle).unwrap();
                assert_eq!(preserved, (live.handle, live.audio_slot, live.image_slot, live.consumed_tokens,
                    live.generated_tokens, live.scheduled_consumed, live.reused_prompt_tokens(), live.last_use,
                    live.pending, live.private_pages.clone(), live.reserved_pages.clone()),
                    "conditioned upload changed request ownership or token state");
                assert_eq!(ownership, (Self::project(s, self.fence_origin), s.stats(), s.free_pages.clone(),
                    s.free_snapshots.clone(), s.free_audio_slots.clone()),
                    "conditioned upload changed shared ownership or graph fences");
                assert!(live.epoch > epoch, "conditioned upload did not invalidate prior scheduler plans");
                assert!(live.encoder_jobs.is_empty(), "conditioned upload did not retire its encoder jobs");
                assert!(matches!(live.phase, RequestPhase::Prefill | RequestPhase::Resume),
                    "conditioned upload did not advance to text work");
            }
            _ => unreachable!(),
        }
        self.assert_ownership();
    }
}

#[derive(Default, Debug)]
struct Coverage {
    two_pending: bool,
    cancel_two_pending: bool,
    first_completion_retains_slot: bool,
    stale_handle: bool,
    stale_plan: bool,
    reused_generation_plan: bool,
    stale_completion: bool,
    partially_valid_completion: bool,
    duplicate_completion: bool,
    out_of_order: bool,
    uncertain_drain: bool,
    feedback_barrier: bool,
    full_window_rejection: bool,
    feedback_rejection: bool,
    audio_rejections: bool,
}

fn check_rejections(fixture: &mut Fixture, coverage: &mut Coverage) {
    let before = fixture.state_key();
    let s = &mut fixture.scheduler;
    if let Some(plan) = &fixture.last_plan {
        let full = s.submissions.len() == 2;
        let feedback = s
            .submissions
            .values()
            .any(|submission| submission.requires_feedback);
        if full || feedback {
            let error = s
                .commit(plan, Fence(s.submitted.0 + 1))
                .expect_err("commit accepted work beyond the window or feedback barrier");
            assert_eq!(
                error.to_string(),
                "submission window or feedback barrier is full",
                "commit bypassed the submission window or feedback barrier"
            );
            coverage.full_window_rejection |= full;
            coverage.feedback_rejection |= feedback;
        }
    }
    if s.submissions.len() == 2 {
        coverage.two_pending = true;
        let fence = *s.submissions.last_key_value().unwrap().0;
        let records = completion_records(s, fence);
        assert!(
            s.complete(fence, &records, Default::default()).is_err(),
            "out-of-order completion accepted"
        );
        coverage.out_of_order = true;
    }
    if let Some((fence, submission)) = s.submissions.first_key_value() {
        let fence = *fence;
        if submission.kind == GraphKind::AudioEncode {
            let slot = submission.lanes[0].request.slot as usize;
            if s.slots[slot].request.is_some() {
                for control in 0..6 {
                    let mut invalid = completion_records(s, fence);
                    let record = &mut invalid[slot];
                    match control {
                        0 => record.submission_sequence += 1,
                        1 => record.consumed_tokens += 1,
                        2 => record.generated_tokens += 1,
                        3 => record.token_id = 99,
                        4 => {
                            let mut vector = crate::execution::TokenVector::default();
                            vector.emitted_count = 1;
                            vector.tokens[0] = 99;
                            record.token_id = 99;
                            record.vector = Some(vector);
                        }
                        5 => record.status = 2,
                        _ => unreachable!(),
                    }
                    assert!(s.complete(fence, &invalid, Default::default()).is_err(),
                        "audio encoder accepted invalid completion metadata");
                }
                coverage.audio_rejections = true;
            }
        }
        let mut records = completion_records(s, fence);
        coverage.partially_valid_completion |=
            records.iter().filter(|record| record.status != 0).count() > 1;
        if let Some(record) = records.iter_mut().rfind(|record| record.status != 0) {
            record.generation += 1;
            assert!(
                s.complete(fence, &records, Default::default()).is_err(),
                "stale generation completion accepted"
            );
            coverage.stale_completion = true;
        }
    }
    if s.completed.0 > fixture.fence_origin {
        assert!(
            s.complete(
                s.completed,
                &[CompletionRecord::default(), CompletionRecord::default()],
                Default::default()
            )
            .is_err(),
            "duplicate completion accepted"
        );
        coverage.duplicate_completion = true;
    }
    let stale_handles = s
        .slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.generation > 1)
        .map(|(index, slot)| RequestHandle {
            slot: index as u32,
            generation: slot.generation - 1,
        })
        .collect::<Vec<_>>();
    for stale in stale_handles {
        assert!(
            s.request(stale).is_err(),
            "reused slot accepted an old request handle"
        );
        assert!(
            s.cancel(stale).is_err(),
            "old request handle cancelled a reused slot"
        );
        coverage.stale_handle = true;
    }
    assert_eq!(
        fixture.state_key(),
        before,
        "rejected completion mutated scheduler state"
    );
}

#[test]
fn scheduler_matches_lifecycle_model() {
    let mut pending = VecDeque::from([Vec::<Action>::new()]);
    let mut visited = HashSet::new();
    let mut edges = BTreeSet::new();
    let mut coverage = Coverage::default();
    let mut reused = Fixture::new();
    let stale_plan = reused.scheduler.plan().unwrap();
    let handle = stale_plan.lanes[0].request;
    reused.apply([4, u64::from(handle.slot), 0]);
    reused.apply([1, u64::from(handle.slot), 0]);
    let replacement = reused.scheduler.slots[handle.slot as usize]
        .request
        .as_ref()
        .unwrap()
        .handle;
    assert!(replacement.generation > handle.generation);
    let before = reused.state_key();
    assert!(
        reused
            .scheduler
            .commit(&stale_plan, Fence(reused.scheduler.submitted.0 + 1))
            .is_err(),
        "old generation execution plan accepted after slot reuse"
    );
    assert_eq!(
        reused.state_key(),
        before,
        "stale plan changed a reused slot"
    );
    coverage.reused_generation_plan = true;
    while let Some(path) = pending.pop_front() {
        let mut fixture = Fixture::new();
        for action in &path {
            fixture.apply(*action);
        }
        if !visited.insert(fixture.state_key()) {
            continue;
        }
        fixture.assert_ownership();
        check_rejections(&mut fixture, &mut coverage);
        if path.len() == MAX_DEPTH {
            continue;
        }
        let before = fixture.projection();
        for action in fixture.actions() {
            let mut candidate = Fixture::new();
            for previous in &path {
                candidate.apply(*previous);
            }
            let stale_plan = (action[0] == 4)
                .then(|| candidate.scheduler.plan())
                .flatten()
                .filter(|plan| {
                    plan.lanes
                        .iter()
                        .any(|lane| u64::from(lane.request.slot) == action[1])
                });
            if action[0] == 4 && candidate.scheduler.submissions.len() == 2 {
                coverage.cancel_two_pending = true;
            }
            candidate.apply(action);
            let after = candidate.projection();
            if let Some(plan) = stale_plan {
                let state = candidate.state_key();
                let fence = Fence(candidate.scheduler.submitted.0 + 1);
                assert!(
                    candidate.scheduler.commit(&plan, fence).is_err(),
                    "cancelled request plan accepted"
                );
                assert_eq!(candidate.state_key(), state);
                coverage.stale_plan = true;
            }
            if action[0] == 3 && before[2..4].contains(&2) && after[2..4].contains(&2) {
                coverage.first_completion_retains_slot = true;
            }
            coverage.uncertain_drain |= action[0] == 5 && before[8] != 0;
            coverage.feedback_barrier |= action[0] == 2 && action[2] == 1;
            edges.insert((before, action, after));
            let mut successor = path.clone();
            successor.push(action);
            pending.push_back(successor);
        }
    }
    assert!(
        coverage.two_pending
            && coverage.cancel_two_pending
            && coverage.first_completion_retains_slot,
        "missing deferred-retirement coverage: {coverage:?}"
    );
    assert!(
        coverage.stale_handle
            && coverage.stale_plan
            && coverage.reused_generation_plan
            && coverage.stale_completion
            && coverage.partially_valid_completion
            && coverage.duplicate_completion
            && coverage.out_of_order,
        "missing rejection coverage: {coverage:?}"
    );
    assert!(
        coverage.uncertain_drain
            && coverage.feedback_barrier
            && coverage.full_window_rejection
            && coverage.feedback_rejection,
        "missing drain or barrier coverage: {coverage:?}"
    );
    for (before, action, after) in &edges {
        println!("LIFECYCLE_EDGE [{before:?}, {action:?}, {after:?}]");
    }
    println!(
        "LIFECYCLE_COVERAGE states={} edges={} depth={MAX_DEPTH} {coverage:?}",
        visited.len(),
        edges.len()
    );
}

#[test]
fn scheduler_audio_matches_lifecycle_model() {
    let mut edges = BTreeSet::new();
    let mut coverage = Coverage::default();
    let (mut initial_cancel, mut encoder_cancel, mut between_encoders, mut encoded_cancel) = (false, false, false, false);
    let (mut prefill_cancel, mut arena_pressure, mut reused_arena, mut total_states) = (false, false, false, 0usize);
    let mut conditioned_upload = false;
    let (mut upload_cancel, mut upload_text, mut upload_reuse, mut upload_failed_drain) = (false, false, false, false);
    for kind in [FixtureKind::AudioStart, FixtureKind::AudioDecode] {
        let mut pending = VecDeque::from([Vec::<Action>::new()]);
        let mut visited = HashSet::new();
        while let Some(path) = pending.pop_front() {
            let mut fixture = Fixture::new_kind(kind);
            for action in &path { fixture.apply(*action); }
            if !visited.insert(fixture.state_key()) { continue; }
            fixture.assert_ownership();
            check_rejections(&mut fixture, &mut coverage);
            if path.len() == MAX_DEPTH { continue; }
            let before = fixture.projection();
            for action in fixture.actions() {
                let after_upload = path.iter().any(|previous| previous[0] == 7);
                let mut candidate = Fixture::new_kind(kind);
                for previous in &path { candidate.apply(*previous); }
                upload_text |= after_upload && action[0] == 2 && candidate.scheduler.plan()
                    .is_some_and(|plan| matches!(plan.kind, GraphKind::Prefill | GraphKind::Decode));
                let stale_plan = (action[0] == 4).then(|| candidate.scheduler.plan()).flatten()
                    .filter(|plan| plan.lanes.iter().any(|lane| lane.request.slot as u64 == action[1]));
                if action[0] == 4 {
                    let live = candidate.scheduler.slots[action[1] as usize].request.as_ref().unwrap();
                    if live.audio_slot.is_some() {
                        initial_cancel |= live.phase == RequestPhase::Initialize;
                        encoder_cancel |= live.phase == RequestPhase::AudioEncode && live.pending != 0;
                        between_encoders |= live.phase == RequestPhase::AudioEncode && live.pending == 0 && live.encoder_jobs.len() == 1;
                        encoded_cancel |= live.phase == RequestPhase::Prefill && live.pending == 0;
                        prefill_cancel |= live.phase == RequestPhase::Prefill && live.pending != 0;
                        coverage.cancel_two_pending |= live.phase == RequestPhase::Decode && live.pending == 2;
                    }
                }
                candidate.apply(action);
                conditioned_upload |= action[0] == 7;
                upload_cancel |= after_upload && action[0] == 4;
                upload_failed_drain |= after_upload && action[0] == 5;
                let after = candidate.projection();
                if let Some(mid) = candidate.rejected_admission {
                    arena_pressure = true;
                    edges.insert((before, action, mid));
                    edges.insert((mid, [4, action[1], 0], after));
                } else {
                    edges.insert((before, action, after));
                    if action[0] == 1 {
                        let live = candidate.scheduler.slots[action[1] as usize].request.as_ref().unwrap();
                        reused_arena |= live.handle.generation == 2 && live.audio_slot.is_some();
                        upload_reuse |= after_upload && live.handle.generation == 2 && live.audio_slot.is_some();
                    }
                }
                if let Some(plan) = stale_plan {
                    let state = candidate.state_key();
                    assert!(candidate.scheduler.commit(&plan, Fence(candidate.scheduler.submitted.0 + 1)).is_err(),
                        "cancelled audio request plan accepted");
                    assert_eq!(candidate.state_key(), state);
                    coverage.stale_plan = true;
                }
                coverage.first_completion_retains_slot |= action[0] == 3 && before[2..4].contains(&2) && after[2..4].contains(&2);
                coverage.uncertain_drain |= action[0] == 5 && before[8] != 0;
                coverage.feedback_barrier |= action[0] == 2 && action[2] == 1;
                let mut successor = path.clone();
                successor.push(action);
                pending.push_back(successor);
            }
        }
        total_states += visited.len();
    }
    assert!(initial_cancel && encoder_cancel && between_encoders && encoded_cancel && prefill_cancel,
        "missing audio phase cancellation: {initial_cancel} {encoder_cancel} {between_encoders} {encoded_cancel} {prefill_cancel}");
    assert!(arena_pressure && reused_arena && coverage.audio_rejections,
        "missing audio capacity/reuse/completion controls: {arena_pressure} {reused_arena} {coverage:?}");
    assert!(conditioned_upload, "missing completed feature upload transition");
    assert!(upload_cancel && upload_text && upload_reuse && upload_failed_drain,
        "missing upload successor coverage: cancel={upload_cancel} text={upload_text} reuse={upload_reuse} failed_drain={upload_failed_drain}");
    assert!(coverage.two_pending && coverage.cancel_two_pending && coverage.first_completion_retains_slot
        && coverage.stale_handle && coverage.stale_plan && coverage.stale_completion && coverage.out_of_order
        && coverage.duplicate_completion && coverage.uncertain_drain && coverage.feedback_barrier
        && coverage.full_window_rejection && coverage.feedback_rejection, "missing shared lifecycle checks: {coverage:?}");
    for (before, action, after) in &edges {
        println!("LIFECYCLE_EDGE [{before:?}, {action:?}, {after:?}]");
    }
    println!("AUDIO_LIFECYCLE_COVERAGE states={total_states} edges={} depth={MAX_DEPTH} {coverage:?}", edges.len());
}

fn initialize_prerequisite(fixture: &mut Fixture, handle: RequestHandle) {
    for _ in 0..8 {
        if fixture.scheduler.conditioned_prerequisite(handle).is_ok() { return; }
        let plan = fixture.scheduler.plan().unwrap();
        // Shared policy may advance another live request before this Init.
        let mask = plan.lanes.iter().fold(0, |mask, lane| mask | (1 << lane.request.slot));
        let barrier = u64::from(fixture.scheduler.plan_requires_feedback(&plan));
        fixture.apply([2, mask, barrier]);
        fixture.apply([3, 0, 0]);
    }
    panic!("initialization did not expose a complete prerequisite");
}

#[test]
fn late_prerequisite_ticket_cannot_publish_into_a_reused_slot_and_arena() {
    let mut fixture = Fixture::new_kind(FixtureKind::AudioStart);
    fixture.apply([4, 1, 0]);
    let first = fixture.scheduler.slots[0].request.as_ref().unwrap().handle;
    initialize_prerequisite(&mut fixture, first);
    let ticket = fixture.scheduler.conditioned_prerequisite(first).unwrap();
    let arena = ticket.feature_slot();
    fixture.apply([4, 0, 0]);
    fixture.apply([1, 0, 0]);
    let next = fixture.scheduler.slots[0].request.as_ref().unwrap().handle;
    assert_eq!(next.slot, first.slot);
    assert!(next.generation > first.generation);
    initialize_prerequisite(&mut fixture, next);
    assert_eq!(fixture.scheduler.conditioned_prerequisite(next).unwrap().feature_slot(), arena);
    let before = fixture.state_key();
    assert!(fixture.scheduler.complete_conditioned_prerequisite(ticket).is_err());
    assert_eq!(fixture.state_key(), before, "late ticket changed a reused request or its arena");
    let current = fixture.scheduler.conditioned_prerequisite(next).unwrap();
    fixture.scheduler.complete_conditioned_prerequisite(current).unwrap();
    fixture.assert_ownership();
}

#[test]
fn duplicate_prerequisite_ticket_cannot_advance_the_epoch_twice() {
    let mut fixture = Fixture::new_kind(FixtureKind::AudioStart);
    let handle = fixture.scheduler.slots[0].request.as_ref().unwrap().handle;
    initialize_prerequisite(&mut fixture, handle);
    let first = fixture.scheduler.conditioned_prerequisite(handle).unwrap();
    let duplicate = fixture.scheduler.conditioned_prerequisite(handle).unwrap();
    fixture.scheduler.complete_conditioned_prerequisite(first).unwrap();
    let before = fixture.state_key();
    assert!(fixture.scheduler.complete_conditioned_prerequisite(duplicate).is_err());
    assert_eq!(fixture.state_key(), before, "duplicate ticket changed epochs, jobs or ownership");
    fixture.assert_ownership();
}
