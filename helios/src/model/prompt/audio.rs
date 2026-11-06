/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

use crate::{Result, frontend::{RenderedPrompt, audio::{PreparedAudio, PreparedAudioInput}},
    model::{ModelConfig, PreparedInput, PromptData}};
use std::sync::Arc;

/// Expand one framed placeholder while retaining the actual audio identity at
/// its first feature token. All text positions remain ordinary causal positions.
pub fn prepare_audio_input(config: &ModelConfig, rendered: &RenderedPrompt,
                           audio: Arc<PreparedAudio>, identity: [u8; 32])
    -> Result<(PreparedInput, PreparedAudioInput)> {
    let capacity = config.contract.audio.as_ref().ok_or("prepared model has no audio feature storage")?;
    let layout = config.contract.frontend.audio_tokens.as_ref().ok_or("prepared audio token layout is missing")?;
    if !audio.matches_layout(capacity) || !rendered.image_sources.is_empty() {
        return Err("audio input does not match the prepared frontend layout".into());
    }
    let markers = &rendered.token_ids;
    let starts: Vec<_> = markers.iter().enumerate().filter(|(_, token)| **token == layout.feature_token)
        .map(|(index, _)| index).collect();
    let [start] = starts.as_slice() else { return Err("audio prompt requires exactly one placeholder".into()); };
    let start = *start;
    if start == 0 || markers.get(start - 1) != Some(&layout.start_token)
        || markers.get(start + 1) != Some(&layout.end_token)
        || markers.iter().filter(|token| **token == layout.start_token).count() != 1
        || markers.iter().filter(|token| **token == layout.end_token).count() != 1
        || markers.iter().any(|token| config.contract.frontend.rejected_tokens.contains(token)) {
        return Err("audio prompt contains invalid or additional audio framing".into());
    }
    let features = audio.feature_count();
    let extent = markers.len().checked_add(features - 1)
        .filter(|extent| *extent <= config.limits.max_sequence_tokens)
        .ok_or("expanded audio prompt exceeds the prepared context")?;
    let mut tokens = Vec::with_capacity(extent);
    tokens.extend_from_slice(&markers[..start]);
    tokens.resize(start + features, layout.feature_token);
    tokens.extend_from_slice(&markers[start + 1..]);
    let mut indices = vec![-1; extent];
    for (index, value) in indices[start..start + features].iter_mut().enumerate() {
        *value = i32::try_from(index).map_err(|_| "audio feature index exceeds Int32")?;
    }
    let data = PromptData {
        positions: if config.contract.frontend.position_dimensions == 3 {
            (0..extent).map(|position| [position as i64; 3]).collect()
        } else { Vec::new() },
        audio_token_indices: indices,
        ..Default::default()
    };
    super::super::validate_prompt_data(config, &tokens, &data)?;
    let input = PreparedAudioInput::new(audio, start..start + features, identity)?;
    Ok((PreparedInput { tokens, token_expansions: vec![(start, features - 1)], data,
        conditioning: vec![crate::cache::ConditioningEvent { position: start as u64, identity }],
        conditioned_inputs: Vec::new() }, input))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::contract::{AudioCapacity, AudioTokenLayout, StateView};

    fn config() -> ModelConfig {
        let mut config = crate::model::tests::config();
        let view = || StateView { buffer: String::new(), offset: 0, stride: 1, nbytes: 1 };
        config.contract.audio = Some(AudioCapacity {
            feature_slots: 1, features_per_request: 7800, feature_width: 2048,
            frame_capacities: vec![100, 800], mel_bins: 128, frames_per_chunk: 100,
            features_per_chunk: 13, subsampling_factor: 8, attention_window_frames: 800,
            features: view(), generations: view(), counts: view(),
        });
        config.contract.frontend.audio_tokens = Some(AudioTokenLayout {
            feature_token: 151676, start_token: 151669, end_token: 151670 });
        config
    }

    fn rendered(ids: Vec<u32>) -> RenderedPrompt {
        RenderedPrompt { text: String::new(), token_ids: ids, message_boundaries: vec![5],
            image_sources: vec![], image_checkpoint_hints: vec![] }
    }

    #[test]
    fn audio_expansion_uses_actual_features_positions_and_content_identity() {
        let config = config();
        let audio = Arc::new(PreparedAudio::new(vec![0.0; 128 * 801], 801, config.contract.audio.as_ref().unwrap()).unwrap());
        let raw = rendered(vec![7, 151669, 151676, 151670, 8, 9]);
        let (prepared, owner) = prepare_audio_input(&config, &raw, audio.clone(), [1; 32]).unwrap();
        assert_eq!(prepared.tokens.len(), 110);
        assert_eq!(prepared.token_expansions, vec![(2, 104)]);
        assert_eq!(&prepared.tokens[..2], &[7, 151669]);
        assert!(prepared.tokens[2..107].iter().all(|id| *id == 151676));
        assert_eq!(&prepared.tokens[107..], &[151670, 8, 9]);
        assert_eq!(&prepared.data.audio_token_indices[2..107], &(0..105).collect::<Vec<i32>>());
        assert!(prepared.data.audio_token_indices[..2].iter().chain(&prepared.data.audio_token_indices[107..]).all(|index| *index == -1));
        assert_eq!(prepared.data.positions, (0..110).map(|i| [i; 3]).collect::<Vec<_>>());
        assert_eq!(prepared.conditioning[0].position, 2);
        assert_eq!(prepared.conditioning[0].identity, [1; 32]);
        assert_eq!(*owner.token_span(), 2..107);
        let (different, _) = prepare_audio_input(&config, &raw, audio, [2; 32]).unwrap();
        assert_eq!(prepared.tokens, different.tokens);
        assert_ne!(prepared.conditioning[0].identity, different.conditioning[0].identity);
    }

    #[test]
    fn malformed_duplicate_and_context_overflow_audio_placeholders_are_rejected() {
        let mut config = config();
        let audio = Arc::new(PreparedAudio::new(vec![0.0; 128 * 99], 99, config.contract.audio.as_ref().unwrap()).unwrap());
        for ids in [vec![7], vec![151676, 151670], vec![7, 151676, 151670],
            vec![151669, 151676, 7], vec![151669, 151676, 151670, 151669, 151676, 151670],
            vec![151669, 151676, 151670, 151669]] {
            assert!(prepare_audio_input(&config, &rendered(ids), audio.clone(), [1; 32]).is_err());
        }
        config.limits.max_sequence_tokens = 14;
        assert!(prepare_audio_input(&config, &rendered(vec![7, 151669, 151676, 151670]), audio, [1; 32]).is_err());
    }
}
