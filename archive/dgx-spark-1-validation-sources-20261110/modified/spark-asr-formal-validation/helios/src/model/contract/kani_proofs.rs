/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

#[kani::proof]
#[kani::unwind(16)]
fn state_view_containment_matches_wide_arithmetic() {
    let same_buffer: bool = kani::any();
    let outer = StateView {
        buffer: "state".into(),
        offset: kani::any(),
        stride: kani::any(),
        nbytes: kani::any(),
    };
    let inner = StateView {
        buffer: if same_buffer { "state" } else { "other" }.into(),
        offset: kani::any(),
        stride: kani::any(),
        nbytes: kani::any(),
    };
    let outer_end = outer.offset as u128 + outer.nbytes as u128;
    let inner_end = inner.offset as u128 + inner.nbytes as u128;
    let expected = same_buffer
        && outer.stride == inner.stride
        && outer.offset <= inner.offset
        && outer_end <= usize::MAX as u128
        && inner_end <= usize::MAX as u128
        && inner_end <= outer_end;
    assert_eq!(outer.contains(&inner), expected);
}

#[kani::proof]
#[kani::unwind(16)]
fn state_view_storage_validation_matches_wide_arithmetic() {
    use crate::execution::Dtype;
    let (dtype, element_bytes) = match kani::any::<u8>() % 8 {
        0 => (Dtype::Bool, 1u128),
        1 => (Dtype::Uint8, 1),
        2 => (Dtype::Float8E4m3fn, 1),
        3 => (Dtype::Float16, 2),
        4 => (Dtype::Bfloat16, 2),
        5 => (Dtype::Int32, 4),
        6 => (Dtype::Float32, 4),
        _ => (Dtype::Int64, 8),
    };
    let rows: usize = kani::any();
    let elements: usize = kani::any();
    let nbytes: usize = kani::any();
    let has_shape: bool = kani::any();
    let has_stride: bool = kani::any();
    let mut shape = vec![rows];
    let mut strides = vec![elements];
    if !has_shape {
        shape.clear();
    }
    if !has_stride {
        strides.clear();
    }
    let view = StateView {
        buffer: String::new(),
        offset: kani::any(),
        stride: kani::any(),
        nbytes: kani::any(),
    };
    let buffer = BufferDescriptor {
        name: String::new(),
        address: 4096,
        nbytes,
        dtype,
        shape,
        strides,
    };
    let expected = view.nbytes != 0
        && has_shape
        && has_stride
        && elements as u128 * element_bytes == view.stride as u128
        && view.offset as u128 + view.nbytes as u128 <= view.stride as u128
        && rows as u128 * view.stride as u128 <= nbytes as u128;
    assert_eq!(view.fits_storage(&buffer), expected);
}

fn check_audio_stride(capacities: Vec<usize>) {
    let view = StateView { buffer: String::new(), offset: 0, stride: 1, nbytes: 1 };
    let audio = AudioCapacity {
        feature_slots: kani::any(), features_per_request: kani::any(), feature_width: kani::any(),
        frame_capacities: capacities, mel_bins: kani::any(), frames_per_chunk: kani::any(),
        features_per_chunk: kani::any(), subsampling_factor: kani::any(), attention_window_frames: kani::any(),
        features: view.clone(), generations: view.clone(), counts: view,
    };
    let chunk = audio.frames_per_chunk;
    let factor = audio.subsampling_factor;
    let window = audio.attention_window_frames;
    // Only overflow arithmetic needs wider integers. The division is performed
    // after proving that its widened numerator is representable as usize.
    let ceiling = if factor == 0 { None } else {
        let numerator = chunk as u128 + factor as u128 - 1;
        if numerator <= usize::MAX as u128 { Some(numerator as usize / factor) } else { None }
    };
    let expected = (1..=crate::execution::REQUEST_LANES).contains(&audio.feature_slots)
        && (1..=i32::MAX as usize).contains(&audio.features_per_request)
        && audio.feature_width > 0 && audio.mel_bins > 0 && chunk > 0 && factor > 0
        && audio.features_per_chunk > 0 && ceiling == Some(audio.features_per_chunk)
        && window > 0 && window <= i32::MAX as usize && window % chunk == 0
        && (window / chunk) as u128 * audio.features_per_chunk as u128 <= audio.features_per_request as u128
        && match audio.frame_capacities.as_slice() {
            [last] => *last == window,
            [first, last] => *first > 0 && *first < *last && *last == window && first % chunk == 0,
            _ => false,
        }
        && audio.features_per_request as u128 * audio.feature_width as u128 * 2 <= usize::MAX as u128;
    let actual = audio.feature_stride();
    assert_eq!(actual.is_ok(), expected);
    if let Ok(stride) = actual {
        assert_eq!(stride as u128, audio.features_per_request as u128 * audio.feature_width as u128 * 2);
    }
}

#[kani::proof]
#[kani::unwind(16)]
fn audio_capacity_empty_catalog_matches_wide_arithmetic() {
    check_audio_stride(Vec::new());
}

#[kani::proof]
#[kani::unwind(16)]
fn audio_capacity_single_catalog_matches_wide_arithmetic() {
    check_audio_stride(vec![kani::any()]);
}

#[kani::proof]
#[kani::unwind(16)]
fn audio_capacity_two_entry_catalog_matches_wide_arithmetic() {
    check_audio_stride(vec![kani::any(), kani::any()]);
}
