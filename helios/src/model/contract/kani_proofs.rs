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

// Arithmetic first-failure cases plus an independent catalog-rejection
// property cover every assignment of the original usize inputs. The catalog
// property has no unnecessary valid-arithmetic precondition. Every active case
// invokes the production validator; no reduced-width domain is used.
fn arbitrary_audio_capacity(capacities: Vec<usize>) -> AudioCapacity {
    let view = StateView { buffer: String::new(), offset: 0, stride: 1, nbytes: 1 };
    AudioCapacity {
        feature_slots: kani::any(), features_per_request: kani::any(), feature_width: kani::any(),
        frame_capacities: capacities, mel_bins: kani::any(), frames_per_chunk: kani::any(),
        features_per_chunk: kani::any(), subsampling_factor: kani::any(), attention_window_frames: kani::any(),
        features: view.clone(), generations: view.clone(), counts: view,
    }
}

#[kani::proof]
#[kani::unwind(16)]
fn audio_capacity_two_entry_catalog_last_mismatch_matches_wide_arithmetic() {
    let audio = arbitrary_audio_capacity(vec![kani::any(), kani::any()]);
    if audio.frame_capacities[1] != audio.attention_window_frames {
        let result = std::mem::ManuallyDrop::new(audio.feature_stride());
        assert!(result.is_err());
    }
}

#[kani::proof]
fn audio_ceiling_large_factor_has_unit_quotient() {
    let chunk: usize = kani::any();
    let factor: usize = kani::any();
    if chunk > 0 && factor >= chunk {
        if let Some(numerator) = chunk.checked_add(factor - 1) {
            // This independent lemma uses the real division, rather than the
            // shortcut used by the wide geometry oracle.
            assert_eq!(numerator / factor, 1);
        }
    }
}

#[kani::proof]
fn audio_positive_window_multiple_contains_chunk() {
    let chunk: usize = kani::any();
    let window: usize = kani::any();
    if chunk > 0 && window > 0 && window.is_multiple_of(chunk) {
        assert!(chunk <= window);
    }
}

// Independent arithmetic lemmas for a possible compositional proof. These do
// not stub the validator or replace its full-usize acceptance/rejection oracle.
#[kani::proof]
fn audio_arithmetic_u32_division_product_interval() {
    let numerator: u32 = kani::any();
    let denominator: u32 = kani::any();
    if denominator != 0 {
        let quotient = numerator / denominator;
        assert!(u64::from(quotient) * u64::from(denominator) <= u64::from(numerator));
        assert!(u64::from(numerator) < (u64::from(quotient) + 1) * u64::from(denominator));
    }
}

#[kani::proof]
fn audio_arithmetic_bounded_division_widens_exactly() {
    let numerator: u32 = kani::any();
    let denominator: u32 = kani::any();
    if denominator != 0 {
        assert_eq!(numerator as usize / denominator as usize, (numerator / denominator) as usize);
    }
}

#[kani::proof]
fn audio_arithmetic_small_ceiling_numerator_fits_u32() {
    let chunk: usize = kani::any();
    let factor: usize = kani::any();
    if chunk > 0 && chunk <= i32::MAX as usize && factor > 0 && factor < chunk {
        let numerator = chunk.checked_add(factor - 1).unwrap();
        assert!(numerator <= u32::MAX as usize);
        assert_eq!(numerator as u32 as usize, numerator);
        assert_eq!(factor as u32 as usize, factor);
    }
}

fn check_audio_stride_partition<const CASE: u8, const PARTITION: u8>(capacities: Vec<usize>) {
    let audio = arbitrary_audio_capacity(capacities);
    let chunk = audio.frames_per_chunk;
    let factor = audio.subsampling_factor;
    let window = audio.attention_window_frames;
    // Reject cheap scalar geometry before forming any symbolic remainder.
    // CASE 0 and CASE 3 both invoke the real validator on this full-usize
    // partition. This strengthens their rejection claims without excluding an
    // input, stubbing production code, or changing the wide arithmetic oracle.
    let scalar_bad = !(1..=crate::execution::REQUEST_LANES).contains(&audio.feature_slots)
        || !(1..=i32::MAX as usize).contains(&audio.features_per_request)
        || audio.feature_width == 0 || audio.mel_bins == 0 || chunk == 0 || factor == 0
        || audio.features_per_chunk == 0 || window == 0 || window > i32::MAX as usize
        || chunk > window;
    if scalar_bad {
        if CASE == 0 || CASE == 3 { assert!(std::mem::ManuallyDrop::new(audio.feature_stride()).is_err()); }
        return;
    }
    // Expand the bounded proof catalogs independently of the validator's
    // generic iteration. Every entry retains its original full usize domain.
    let catalog_bad = match audio.frame_capacities.as_slice() {
        [] => true,
        [last] => *last != window || window % chunk != 0,
        [first, last] => *last != window || *first == 0 || *first >= window
            || first % chunk != 0 || window % chunk != 0,
        _ => audio.frame_capacities.last() != Some(&window)
            || audio.frame_capacities.windows(2).any(|pair| pair[0] >= pair[1])
            || audio.frame_capacities.iter().any(|frames| *frames == 0 || frames % chunk != 0),
    };
    if CASE == 3 {
        if catalog_bad {
            // Prove the actual returned rejection, excluding caller-side
            // destruction of its diagnostic Error after return.
            let result = std::mem::ManuallyDrop::new(audio.feature_stride());
            assert!(result.is_err());
        }
        return;
    }
    // CASE 3 covers the remaining catalog-rejection partition. Together with
    // the scalar rejection above it still covers every original catalog error.
    if catalog_bad { return; }
    assert_eq!(window % chunk, 0);
    if CASE == 0 { return; }
    assert!(chunk <= window && chunk <= i32::MAX as usize);
    // The two nonzero partitions are complementary over the original usize
    // inputs. Keep the unsplit proofs as well; these isolate solver cost and
    // do not replace the complete acceptance/rejection obligations.
    if PARTITION == 1 && factor < chunk || PARTITION == 2 && factor >= chunk { return; }
    let numerator = chunk as u128 + factor as u128 - 1;
    let checked_numerator = chunk.checked_add(factor - 1);
    // Prove the narrowing bridge before forming the quotient. This preserves
    // the independent wide addition oracle while giving both divisions the
    // identical, proven-equal target-width operand expression.
    assert_eq!(checked_numerator.map(|value| value as u128),
        if numerator > usize::MAX as u128 { None } else { Some(numerator) });
    let Some(value) = checked_numerator else {
        if CASE == 1 { assert!(std::mem::ManuallyDrop::new(audio.feature_stride()).is_err()); }
        return;
    };
    let ceiling = if factor >= chunk { 1 } else { value / factor };
    if ceiling != audio.features_per_chunk {
        if CASE == 1 { assert!(std::mem::ManuallyDrop::new(audio.feature_stride()).is_err()); }
        return;
    }
    if CASE == 1 { return; }
    if window % chunk != 0
        || (window / chunk) as u128 * audio.features_per_chunk as u128 > audio.features_per_request as u128 {
        if CASE == 2 { assert!(std::mem::ManuallyDrop::new(audio.feature_stride()).is_err()); }
        return;
    }
    if CASE == 2 { return; }
    if !match audio.frame_capacities.as_slice() {
        [last] => *last == window,
        [first, last] => *first > 0 && *first < *last && *last == window && first % chunk == 0,
        _ => false,
    } {
        return;
    }
    let product = audio.features_per_request as u128 * audio.feature_width as u128;
    if product > usize::MAX as u128 / 2 {
        if CASE == 4 { assert!(std::mem::ManuallyDrop::new(audio.feature_stride()).is_err()); }
        return;
    }
    if CASE == 5 {
        let result = std::mem::ManuallyDrop::new(audio.feature_stride());
        match &*result {
            Ok(stride) => assert!(*stride as u128 == product * 2),
            Err(_) => assert!(false, "valid wide-arithmetic geometry was rejected"),
        }
    }

}

fn check_audio_stride<const CASE: u8>(capacities: Vec<usize>) {
    check_audio_stride_partition::<CASE, 0>(capacities);
}

macro_rules! audio_stride_proof {
    ($name:ident, $case:literal, $($entry:expr),* $(,)?) => {
        #[kani::proof]
        #[kani::unwind(16)]
        fn $name() { check_audio_stride::<$case>(vec![$($entry),*]); }
    };
}

audio_stride_proof!(audio_capacity_empty_scalar_matches_wide_arithmetic, 0, );
audio_stride_proof!(audio_capacity_empty_ceiling_matches_wide_arithmetic, 1, );
audio_stride_proof!(audio_capacity_empty_window_matches_wide_arithmetic, 2, );
audio_stride_proof!(audio_capacity_empty_catalog_matches_wide_arithmetic, 3, );
audio_stride_proof!(audio_capacity_empty_extent_matches_wide_arithmetic, 4, );
audio_stride_proof!(audio_capacity_empty_accepted_matches_wide_arithmetic, 5, );
audio_stride_proof!(audio_capacity_single_scalar_matches_wide_arithmetic, 0, kani::any());
audio_stride_proof!(audio_capacity_single_ceiling_matches_wide_arithmetic, 1, kani::any());
audio_stride_proof!(audio_capacity_single_window_matches_wide_arithmetic, 2, kani::any());
audio_stride_proof!(audio_capacity_single_catalog_matches_wide_arithmetic, 3, kani::any());
audio_stride_proof!(audio_capacity_single_extent_matches_wide_arithmetic, 4, kani::any());
audio_stride_proof!(audio_capacity_single_accepted_matches_wide_arithmetic, 5, kani::any());
audio_stride_proof!(audio_capacity_two_entry_scalar_matches_wide_arithmetic, 0, kani::any(), kani::any());
audio_stride_proof!(audio_capacity_two_entry_ceiling_matches_wide_arithmetic, 1, kani::any(), kani::any());
audio_stride_proof!(audio_capacity_two_entry_window_matches_wide_arithmetic, 2, kani::any(), kani::any());
audio_stride_proof!(audio_capacity_two_entry_catalog_matches_wide_arithmetic, 3, kani::any(), kani::any());
audio_stride_proof!(audio_capacity_two_entry_extent_matches_wide_arithmetic, 4, kani::any(), kani::any());
audio_stride_proof!(audio_capacity_two_entry_accepted_matches_wide_arithmetic, 5, kani::any(), kani::any());

macro_rules! audio_stride_partition_proof {
    ($name:ident, $case:literal, $partition:literal, $($entry:expr),* $(,)?) => {
        #[kani::proof]
        #[kani::unwind(16)]
        fn $name() { check_audio_stride_partition::<$case, $partition>(vec![$($entry),*]); }
    };
}

audio_stride_partition_proof!(audio_capacity_single_ceiling_factor_ge_chunk, 1, 1, kani::any());
audio_stride_partition_proof!(audio_capacity_single_ceiling_factor_lt_chunk, 1, 2, kani::any());
audio_stride_partition_proof!(audio_capacity_single_accepted_factor_ge_chunk, 5, 1, kani::any());
audio_stride_partition_proof!(audio_capacity_single_accepted_factor_lt_chunk, 5, 2, kani::any());
audio_stride_partition_proof!(audio_capacity_two_entry_ceiling_factor_ge_chunk, 1, 1, kani::any(), kani::any());
audio_stride_partition_proof!(audio_capacity_two_entry_ceiling_factor_lt_chunk, 1, 2, kani::any(), kani::any());
audio_stride_partition_proof!(audio_capacity_two_entry_accepted_factor_ge_chunk, 5, 1, kani::any(), kani::any());
audio_stride_partition_proof!(audio_capacity_two_entry_accepted_factor_lt_chunk, 5, 2, kani::any(), kani::any());
