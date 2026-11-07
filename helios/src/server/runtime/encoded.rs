/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Bounded host retention of graph-produced features. Lookups never block the
//! model worker. Only a completed copy can satisfy a current prerequisite.

use super::*;
use crate::frontend::audio::encoded::{Geometry, Lookup};
use flare_selene::input::{InputStore, PreparedInputRecord};
use std::task::Poll;

enum Entry {
    Looking(Lookup),
    Ready(Arc<PreparedInputRecord>),
    Miss,
    Finished,
}

pub(super) struct EncodedInputs {
    store: InputStore,
    entries: HashMap<RequestHandle, Entry>,
}

impl EncodedInputs {
    pub(super) fn new(store: InputStore) -> Self {
        Self { store, entries: HashMap::new() }
    }

    fn service(&mut self, worker: &mut Worker<'_>, executor: &mut RankExecutor) -> Result<bool> {
        self.entries.retain(|handle, _| worker.active.get(handle.slot as usize)
            .and_then(Option::as_ref).is_some_and(|active| active.request == *handle
                && !active.incoming.cancelled.load(Ordering::Acquire)));
        let mut progress = false;
        for slot in 0..worker.active.len() {
            let Some(active) = worker.active[slot].as_ref() else { continue; };
            if active.incoming.cancelled.load(Ordering::Acquire) { continue; }
            let Some(input) = active.incoming.prompt.conditioned_audio.as_ref() else { continue; };
            let handle = active.request;
            let live = worker.scheduler.request(handle)?;
            if live.audio_slot.is_none() || live.error_code.is_some() { continue; }
            let audio = input.audio();
            let count = audio.feature_count();
            let capacity = worker.config.contract.audio.as_ref().ok_or("missing feature storage contract")?;
            let geometry = Geometry::new(count, capacity.feature_width)?;
            let entry = self.entries.entry(handle).or_insert_with(|| {
                geometry.lookup(&self.store, worker.config.identity, input.identity())
                    .map(Entry::Looking).unwrap_or(Entry::Miss)
            });
            if let Entry::Looking(lookup) = entry {
                if active.next_audio_window != 0 {
                    // A slow lookup cannot delay normal encoding or replace a
                    // partially consumed input. Dropping it cancels no GPU work.
                    *entry = Entry::Miss;
                    progress = true;
                } else if let Poll::Ready(record) = lookup.poll() {
                    *entry = record.map(Entry::Ready).unwrap_or(Entry::Miss);
                    progress = true;
                }
            }
            match entry {
                Entry::Ready(record) if live.phase == RequestPhase::AudioEncode
                    && active.next_audio_window == 0 && executor.outstanding() == 0 => {
                    let ticket = worker.scheduler.conditioned_prerequisite(handle)?;
                    if ticket.input().token_span != *input.token_span()
                        || ticket.input().feature_range != (0..count)
                    {
                        return Err("retained input differs from its current prerequisite".into());
                    }
                    let bytes = record.payload.as_ref().ok_or("retained features have no payload")?;
                    let started = std::time::Instant::now();
                    executor.load_features(handle, ticket.feature_slot(), count, bytes)?;
                    // The client may disconnect while the synchronous transfer
                    // sleeps. The arena is still owned; do not publish cancelled
                    // work. The ordinary worker cancellation path retires it.
                    if !active.incoming.cancelled.load(Ordering::Acquire) {
                        worker.scheduler.complete_conditioned_prerequisite(ticket)?;
                        worker.active[slot].as_mut().unwrap().next_audio_window = audio.windows().len();
                        worker.metrics.encoded_inputs_hit(bytes.len(), started.elapsed());
                    }
                    *entry = Entry::Finished;
                    progress = true;
                }
                Entry::Ready(_) if active.next_audio_window != 0 || live.encoder_job().is_none() => {
                    // Another request may have supplied a prefix, or normal
                    // encoding started before this lookup became ready.
                    *entry = Entry::Finished;
                    worker.metrics.encoded_inputs_skip();
                    progress = true;
                }
                Entry::Miss if live.encoder_job().is_none()
                    && active.next_audio_window == audio.windows().len()
                    && matches!(live.phase, RequestPhase::Prefill | RequestPhase::Resume)
                    && executor.outstanding() == 0 => {
                    let feature_slot = live.audio_slot.unwrap();
                    let key = geometry.key(worker.config.identity, input.identity());
                    let mut transfer_error = None;
                    let started = std::time::Instant::now();
                    let retained = self.store.insert_with(key, geometry.serialized_bytes()?, || {
                        // The reservation bounds CPU allocation before readback.
                        // CUDA failures retain their original type and poison
                        // the executor; they are never converted to cache misses.
                        let mut bytes = vec![0; geometry.nbytes()];
                        if let Err(error) = executor.read_features(handle, feature_slot, count, &mut bytes) {
                            transfer_error = Some(error);
                            return Err("encoded feature readback failed".into());
                        }
                        geometry.record(bytes).map_err(|error| error.to_string().into())
                    });
                    if let Some(error) = transfer_error { return Err(error); }
                    if retained? {
                        worker.metrics.encoded_inputs_retained(geometry.nbytes(), started.elapsed());
                    } else { worker.metrics.encoded_inputs_skip(); }
                    *entry = Entry::Finished;
                    progress = true;
                }
                Entry::Miss if live.consumed_tokens > 0 || live.generated_tokens > 0 => {
                    // Do not hold or rewind a generation to obtain a retention
                    // opportunity after prefill has started.
                    *entry = Entry::Finished;
                    worker.metrics.encoded_inputs_skip();
                    progress = true;
                }
                _ => {}
            }
        }
        Ok(progress)
    }
}

impl Worker<'_> {
    pub(super) fn service_encoded_inputs(&mut self, executor: &mut RankExecutor) -> Result<bool> {
        let Some(mut encoded) = self.encoded_inputs.take() else { return Ok(false); };
        let result = encoded.service(self, executor);
        self.encoded_inputs = Some(encoded);
        result
    }
}
