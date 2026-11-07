/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Upload one declared audio encoder window without text/KV input packing.

use super::*;
use crate::{frontend::audio::PreparedAudio, scheduler::{RequestHandle, ScheduledStep}};

impl Inputs {
    pub fn pack_audio(&mut self, step: &ScheduledStep, config: &ModelConfig, handle: RequestHandle,
                      feature_slot: usize, audio: &PreparedAudio, window: usize) -> Result<()> {
        let capacity = config.contract.audio.as_ref().ok_or("audio input has no prepared storage")?;
        if step.kind != GraphKind::AudioEncode || step.batch_capacity != 1 || step.lanes.len() != 1
            || step.lanes[0].request != handle || step.lanes[0].tokens != 0
            || handle.slot as usize >= config.limits.max_active || handle.generation == 0
            || handle.generation > i64::MAX as u64 || feature_slot >= capacity.feature_slots
            || !audio.matches_layout(capacity)
        {
            return Err("audio submission differs from its request, storage or immutable input layout".into());
        }
        let graph = self.contract.graphs.iter().find(|graph| graph.kind == GraphKind::AudioEncode
            && graph.batch == 1 && graph.tokens == step.token_capacity)
            .ok_or("audio submission lacks a prepared packing plan")?;
        let values = audio.window_buffers(window, step.token_capacity)?;
        self.staging.clear();
        self.unconstrained_masks.retain(|(name, _, _)| graph.inputs.iter()
            .any(|input| input.cache_mask && input.binding.buffer == *name));
        for input in &graph.inputs {
            let integers: &[i32] = match input.binding.source {
                ValueSource::RequestSlot => {
                    write_integers(&mut self.staging, input, 0, &[handle.slot as i64])?;
                    continue;
                }
                ValueSource::RequestGeneration => {
                    write_integers(&mut self.staging, input, 0, &[handle.generation as i64])?;
                    continue;
                }
                ValueSource::AudioFeatureSlot => {
                    write_integers(&mut self.staging, input, 0, &[feature_slot as i64])?;
                    continue;
                }
                ValueSource::AudioTensor => match input.binding.audio_tensor.ok_or("audio tensor kind is absent")? {
                    AudioTensorKind::LogMel => {
                        if input.dtype != Dtype::Float32 || input.elements != values.mel.len() {
                            return Err("audio Mel upload differs from captured storage".into());
                        }
                        self.staging.put(&input.binding.buffer, 0, values.mel.iter().flat_map(|value| value.to_le_bytes()));
                        continue;
                    }
                    AudioTensorKind::ValidFrames => std::slice::from_ref(&values.frames),
                    AudioTensorKind::TotalInputFrames => std::slice::from_ref(&values.total_input_frames),
                    AudioTensorKind::ConvolutionWidth => std::slice::from_ref(&values.convolution_width),
                    AudioTensorKind::GatherIndices => &values.gather_indices,
                    AudioTensorKind::CumulativeLengths => &values.cumulative_lengths,
                    AudioTensorKind::FeatureCount => std::slice::from_ref(&values.feature_count),
                    AudioTensorKind::FeatureOffset => std::slice::from_ref(&values.feature_offset),
                },
                _ => return Err("audio encoder graph declares a non-audio input source".into()),
            };
            if input.dtype != Dtype::Int32 || input.elements != integers.len() {
                return Err("audio metadata upload differs from captured storage".into());
            }
            self.staging.i32(&input.binding.buffer, 0, integers.iter().copied());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{BufferDescriptor, GraphDescriptor, InputBinding, SamplingMode};
    use crate::scheduler::{ScheduledLane, ScheduledStep};

    fn buffer(manifest: &mut ExecutionManifest, name: &str, dtype: Dtype, shape: Vec<usize>) {
        let address = manifest.buffers.iter().map(|buffer| buffer.address + buffer.nbytes).max().unwrap().next_multiple_of(256);
        let nbytes = shape.iter().product::<usize>() * dtype.size();
        let mut stride = 1;
        let mut strides: Vec<_> = shape.iter().rev().map(|extent| { let value = stride; stride *= extent; value }).collect();
        strides.reverse();
        manifest.buffers.push(BufferDescriptor { name: name.into(), address, nbytes, dtype, shape, strides });
    }

    fn fixture() -> (ModelConfig, ExecutionManifest) {
        fixture_with_total_input_frames(false)
    }

    fn fixture_with_total_input_frames(include_total: bool) -> (ModelConfig, ExecutionManifest) {
        let mut config = crate::model::tests::config();
        let mut manifest = crate::model::testing::manifest(&config);
        let view = |name: &str, stride| StateView { buffer: name.into(), offset: 0, stride, nbytes: stride };
        let audio = AudioCapacity { feature_slots: 1, features_per_request: 208, feature_width: 2048,
            frame_capacities: vec![100, 800], mel_bins: 128, frames_per_chunk: 100, features_per_chunk: 13,
            subsampling_factor: 8, attention_window_frames: 800, features: view("audio_features", 208 * 2048 * 2),
            generations: view("audio_generations", 8), counts: view("audio_counts", 4) };
        buffer(&mut manifest, "audio_features", Dtype::Bfloat16, vec![1, 208, 2048]);
        buffer(&mut manifest, "audio_generations", Dtype::Int64, vec![4]);
        buffer(&mut manifest, "audio_counts", Dtype::Int32, vec![4]);
        buffer(&mut manifest, "audio_slot", Dtype::Int32, vec![1]);
        buffer(&mut manifest, "audio_indices", Dtype::Int32, vec![config.limits.prefill_chunk_size]);
        let states = || [&audio.features, &audio.generations, &audio.counts].map(|view|
            StateAccess { view: view.clone(), access: StateAccessKind::ReadWrite }).to_vec();
        for graph in &mut config.contract.graphs {
            if graph.kind == GraphOperation::Prefill {
                graph.state.extend(states());
                graph.inputs.extend([
                    crate::model::testing::binding("audio_slot", ValueSource::AudioFeatureSlot, InputIndexing::Scalar, 0, -1),
                    crate::model::testing::binding("audio_indices", ValueSource::AudioFeatureIndex, InputIndexing::PackedToken, 0, -1),
                ]);
            }
        }
        for frames in [100, 800] {
            let mut inputs = vec![];
            for kind in [AudioTensorKind::LogMel, AudioTensorKind::ValidFrames, AudioTensorKind::ConvolutionWidth,
                         AudioTensorKind::GatherIndices, AudioTensorKind::CumulativeLengths,
                         AudioTensorKind::FeatureCount, AudioTensorKind::FeatureOffset].into_iter()
                .chain(include_total.then_some(AudioTensorKind::TotalInputFrames)) {
                let name = format!("audio{frames}_{kind:?}");
                let (dtype, shape) = audio.tensor_layout(kind, frames).unwrap();
                buffer(&mut manifest, &name, dtype, shape);
                let mut binding = crate::model::testing::binding(&name, ValueSource::AudioTensor, InputIndexing::Scalar, 0, 0);
                binding.audio_tensor = Some(kind);
                inputs.push(binding);
            }
            for (field, source, dtype) in [("slot", ValueSource::RequestSlot, Dtype::Int32),
                                           ("generation", ValueSource::RequestGeneration, Dtype::Int64)] {
                let name = format!("audio{frames}_{field}");
                buffer(&mut manifest, &name, dtype, vec![1]);
                inputs.push(crate::model::testing::binding(&name, source, InputIndexing::Scalar, 0, 0));
            }
            inputs.push(crate::model::testing::binding("audio_slot", ValueSource::AudioFeatureSlot, InputIndexing::Scalar, 0, -1));
            let name = format!("audio{frames}");
            manifest.graphs.push(GraphDescriptor { name: name.clone(), kind: GraphKind::AudioEncode, handle: 1,
                sampling_mode: SamplingMode::General, batch_capacity: 1, token_capacity: frames,
                prefill_capacity: 0, draft_tokens: 0, packed_token_capacity: 0, confidence_width: 0,
                estimated_duration_ns: 1000, input_bindings: vec![], tuned_bindings: vec![] });
            config.contract.graphs.push(GraphContract { graph: name, kind: GraphOperation::AudioEncode, inputs,
                state: states(), state_rows_per_request: 1, draft_cursor: None, embedding_tables: vec![],
                pending_token: None, tuned_bindings: vec![] });
        }
        config.contract.frontend.audio_preparation = Some(AudioPreparationKind::CenteredSlaneyLogMel);
        config.contract.frontend.audio_tokens = Some(AudioTokenLayout { feature_token: 151676, start_token: 151669, end_token: 151670 });
        config.contract.audio = Some(audio);
        for graph in &mut manifest.graphs {
            let contract = config.contract.graph(&graph.name).unwrap();
            for binding in &contract.inputs {
                if graph.input_bindings.iter().any(|input| input.buffer_name == binding.buffer) { continue; }
                let nbytes = manifest.buffers.iter().find(|buffer| buffer.name == binding.buffer).unwrap().nbytes;
                let offset = manifest.input_frame_bytes.next_multiple_of(8);
                graph.input_bindings.push(InputBinding { buffer_name: binding.buffer.clone(), host_offset: offset, nbytes });
                manifest.input_frame_bytes = offset + nbytes;
            }
        }
        (config, manifest)
    }

    #[test]
    fn optional_total_input_frames_preserves_old_layout_and_distinguishes_short_tails() {
        for include_total in [false, true] {
            let (config, manifest) = fixture_with_total_input_frames(include_total);
            config.contract.validate(&manifest).unwrap();
            let mut inputs = Inputs::new(&config, &manifest).unwrap();
            let handle = RequestHandle { slot: 2, generation: 7 };
            for (frames, window, capacity) in [(150, 0, 800), (950, 1, 800),
                                               (801, 1, 100), (801, 1, 800),
                                               (99, 0, 100), (99, 0, 800)] {
                let audio = PreparedAudio::new(vec![0.; 128 * frames], frames,
                    config.contract.audio.as_ref().unwrap()).unwrap();
                let step = ScheduledStep { kind: GraphKind::AudioEncode, batch_capacity: 1,
                    token_capacity: capacity, draft_tokens: 0, packed_token_capacity: 0,
                    mixed: None, requires_feedback: true,
                    lanes: vec![ScheduledLane { request: handle, start_position: 0, tokens: 0 }] };
                inputs.pack_audio(&step, &config, handle, 0, &audio, window).unwrap();
                let uploads = inputs.uploads();
                let total = uploads.iter().find(|upload|
                    upload.buffer == format!("audio{capacity}_TotalInputFrames"));
                if include_total {
                    assert_eq!(i32::from_le_bytes(total.unwrap().bytes.try_into().unwrap()), frames as i32);
                } else {
                    assert!(total.is_none(), "old graphs must not acquire a new binding");
                }
                let valid = uploads.iter().find(|upload|
                    upload.buffer == format!("audio{capacity}_ValidFrames")).unwrap();
                assert_eq!(i32::from_le_bytes(valid.bytes.try_into().unwrap()),
                    (frames - window * 800).min(800) as i32);
            }
        }
    }

    #[test]
    fn total_input_frames_does_not_make_required_inputs_or_layout_checks_optional() {
        let (config, manifest) = fixture_with_total_input_frames(true);
        config.contract.validate(&manifest).unwrap();
        let index = config.contract.graphs.iter().position(|graph|
            graph.kind == GraphOperation::AudioEncode).unwrap();
        let total = config.contract.graphs[index].inputs.iter().position(|input|
            input.audio_tensor == Some(AudioTensorKind::TotalInputFrames)).unwrap();
        let mut duplicate = config.clone();
        duplicate.contract.graphs[index].inputs.push(config.contract.graphs[index].inputs[total].clone());
        assert!(duplicate.contract.validate(&manifest).is_err());
        for required in 0..config.contract.graphs[index].inputs.len() {
            if required == total { continue; }
            let mut missing = config.clone();
            missing.contract.graphs[index].inputs.remove(required);
            assert!(missing.contract.validate(&manifest).is_err());
        }
        let name = &config.contract.graphs[index].inputs[total].buffer;
        let mut wrong_dtype = manifest.clone();
        wrong_dtype.buffers.iter_mut().find(|buffer| &buffer.name == name).unwrap().dtype = Dtype::Float32;
        let error = Inputs::new(&config, &wrong_dtype).err().expect("wrong dtype must fail input-plan construction");
        assert!(error.to_string().contains("audio tensor type, extent or offsets"));
        let mut wrong_shape = manifest.clone();
        let buffer = wrong_shape.buffers.iter_mut().find(|buffer| &buffer.name == name).unwrap();
        buffer.shape = vec![2];
        buffer.nbytes = 8;
        for input in wrong_shape.graphs.iter_mut().flat_map(|graph| &mut graph.input_bindings)
            .filter(|input| &input.buffer_name == name) { input.nbytes = 8; }
        let error = Inputs::new(&config, &wrong_shape).err().expect("wrong shape must fail input-plan construction");
        assert!(error.to_string().contains("audio tensor type, extent or offsets"));
    }

    #[test]
    fn missing_window_or_prefill_inputs_are_rejected_before_replay() {
        let (config, manifest) = fixture();
        for kind in [GraphOperation::AudioEncode, GraphOperation::Prefill] {
            let original = config.contract.graphs.iter().position(|graph| graph.kind == kind).unwrap();
            for index in 0..config.contract.graphs[original].inputs.len() {
                let source = config.contract.graphs[original].inputs[index].source;
                if kind == GraphOperation::Prefill
                    && !matches!(source, ValueSource::AudioFeatureIndex | ValueSource::AudioFeatureSlot) { continue; }
                let mut bad = config.clone();
                bad.contract.graphs[original].inputs.remove(index);
                assert!(bad.contract.validate(&manifest).is_err());
            }
        }
    }

    #[test]
    fn encoder_generation_and_count_views_require_read_write_dependencies() {
        let (config, manifest) = fixture();
        config.contract.validate(&manifest).unwrap();
        for buffer in ["audio_generations", "audio_counts"] {
            let mut bad = config.clone();
            for state in bad.contract.graphs.iter_mut().filter(|graph| graph.kind == GraphOperation::AudioEncode)
                .flat_map(|graph| &mut graph.state).filter(|state| state.view.buffer == buffer) {
                state.access = StateAccessKind::Write;
            }
            assert!(bad.contract.validate(&manifest).unwrap_err().to_string().contains("producer or consumer state access"));
        }
        let mut write_only_features = config.clone();
        for state in write_only_features.contract.graphs.iter_mut().filter(|graph| graph.kind == GraphOperation::AudioEncode)
            .flat_map(|graph| &mut graph.state).filter(|state| state.view.buffer == "audio_features") {
            state.access = StateAccessKind::Write;
        }
        write_only_features.contract.validate(&manifest).unwrap();
    }

    #[test]
    fn feature_bindings_reject_offsets_before_the_integer_writer_can_shift_bounds() {
        let (config, manifest) = fixture();
        Inputs::new(&config, &manifest).unwrap();
        for source in [ValueSource::AudioFeatureSlot, ValueSource::AudioFeatureIndex] {
            for offset in [-1, 1] {
                let mut bad = config.clone();
                for input in bad.contract.graphs.iter_mut().flat_map(|graph| &mut graph.inputs).filter(|input| input.source == source) {
                    input.value_offset = offset;
                }
                assert!(Inputs::new(&bad, &manifest).is_err());
            }
        }
    }

    #[test]
    fn native_uploads_preserve_801_frame_windows_offsets_and_padding() {
        let (config, manifest) = fixture();
        let mut inputs = Inputs::new(&config, &manifest).unwrap();
        let source: Vec<f32> = (0..128 * 801).map(|value| value as f32 / 100.).collect();
        let audio = PreparedAudio::new(source.clone(), 801, config.contract.audio.as_ref().unwrap()).unwrap();
        let handle = RequestHandle { slot: 2, generation: 7 };
        let mut step = ScheduledStep { kind: GraphKind::AudioEncode, batch_capacity: 1, token_capacity: 800,
            draft_tokens: 0, packed_token_capacity: 0, mixed: None, requires_feedback: true,
            lanes: vec![ScheduledLane { request: handle, start_position: 0, tokens: 0 }] };
        for (window, frames, count, offset) in [(0, 800, 104, 0), (1, 1, 1, 104)] {
            step.token_capacity = if window == 0 { 800 } else { 100 };
            inputs.pack_audio(&step, &config, handle, 0, &audio, window).unwrap();
            let uploads = inputs.uploads();
            let get = |suffix: &str| uploads.iter().find(|upload| upload.buffer == format!("audio{}_{}", step.token_capacity, suffix)).unwrap().bytes;
            let integer = |suffix: &str| i32::from_le_bytes(get(suffix).try_into().unwrap());
            assert_eq!(integer("ValidFrames"), frames);
            assert_eq!(integer("FeatureCount"), count);
            assert_eq!(integer("FeatureOffset"), offset);
            assert_eq!(integer("ConvolutionWidth"), 100);
            let mel: Vec<f32> = get("LogMel").chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect();
            for bin in 0..128 {
                for column in 0..step.token_capacity {
                    assert_eq!(mel[bin * step.token_capacity + column], if column < frames as usize {
                        source[bin * 801 + window * 800 + column]
                    } else { 0. });
                }
            }
            let gather: Vec<i32> = get("GatherIndices").chunks_exact(4).map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap())).collect();
            assert!(gather.iter().enumerate().all(|(index, value)| *value == if index < count as usize { index as i32 } else { -1 }));
        }
        assert!(inputs.pack_audio(&step, &config, handle, 1, &audio, 1).is_err());
    }
}
