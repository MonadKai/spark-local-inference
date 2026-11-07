/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host retention of published prefix checkpoints in registered memory and on disk.
//!
//! A GPU prefix checkpoint owns its tail pages, the complete cache pages since its
//! parent, and one continuation-state row. The host store copies those bytes out
//! at publication and copies them back into free GPU
//! ids when a later request matches a checkpoint the GPU no longer holds. The
//! restored bytes are published as an ordinary GPU checkpoint, so the model
//! worker, the scheduler, and the recipes see no difference between a checkpoint
//! produced by prefill and one restored from the host.
//!
#[cfg(all(test, feature = "runtime-package"))]
mod gpu_tests;
#[cfg(feature = "runtime-package")]
pub(crate) mod pool;
mod registrar;
pub mod shard;
pub mod transfer;

pub use flare_selene::pool::{PoolOptions, parse_pool_address};
use flare_selene::{CacheGeometry, HostStore};

use crate::Result;

pub use flare_selene::offload::{
    Binding, PrefixCacheOffload, Preparation, Prepared, Resolved, Shared, input_memory_bytes,
    parse_drain_on_shutdown, parse_max_age, parse_size,
};

pub fn prepare(
    options: &PrefixCacheOffload,
    package: &std::path::Path,
    identity: [u8; 32],
    geometry: Vec<CacheGeometry>,
    contexts: Vec<(usize, i32)>,
) -> Result<Preparation> {
    prepare_with_inputs(options, package, identity, geometry, contexts, 1, 0, None)
}

pub fn prepare_with_inputs(
    options: &PrefixCacheOffload,
    package: &std::path::Path,
    identity: [u8; 32],
    geometry: Vec<CacheGeometry>,
    contexts: Vec<(usize, i32)>,
    max_active: usize,
    input_staging: u64,
    shard: Option<(usize, usize)>,
) -> Result<Preparation> {
    Ok(flare_selene::offload::prepare_with_inputs(
        options,
        package,
        identity,
        geometry,
        contexts,
        max_active,
        input_staging,
        shard,
        &registrar::CudaRegistrar,
    )?)
}

pub fn open(prepared: Prepared) -> Result<Box<dyn HostStore>> {
    Ok(flare_selene::offload::open(prepared, &transfer::Copiers)?)
}

/// One bounded ordinary-memory window for serialized prepared inputs and disk-read leases.
pub fn input_staging_bytes(
    contract: &crate::model::PreparedContract,
    buffer_bytes: impl Fn(&str) -> Option<u64>,
) -> Result<u64> {
    use crate::model::contract::ValueSource;
    let mut maximum = 0;
    for graph in &contract.graphs {
        let mut bytes = 0u64;
        for binding in graph
            .inputs
            .iter()
            .filter(|binding| binding.source == ValueSource::ImageTensor)
        {
            let input =
                buffer_bytes(&binding.buffer).ok_or("prepared image input buffer is absent")?;
            bytes = bytes
                .checked_add(input)
                .ok_or("prepared image input extent overflows")?;
        }
        maximum = maximum.max(bytes);
    }
    if let Some(audio) = &contract.audio {
        // A retained record contains the whole normalized input, not one
        // encoder window. A feature covers at most one convolution chunk or
        // subsampling_factor frames, whichever is smaller.
        let bytes = (audio.features_per_request as u64)
            .checked_mul(audio.subsampling_factor.min(audio.frames_per_chunk) as u64)
            .and_then(|frames| frames.checked_mul(audio.mel_bins as u64))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or("prepared audio input extent overflows")?;
        maximum = maximum.max(bytes);
    }
    if maximum == 0 {
        Ok(0)
    } else {
        maximum
            .checked_add(4096)
            .ok_or_else(|| "prepared input staging extent overflows".into())
    }
}

#[derive(Clone, Copy)]
pub enum LookupOutcome {
    NoMatch,
    CostRejected,
    CapacityRejected,
    EnqueueRejected,
    Admitted,
}

#[cfg(all(test, feature = "frontend"))]
mod input_tests {
    use super::input_staging_bytes;
    use crate::model::contract::{AudioCapacity, GraphContract, GraphOperation, InputIndexing,
        PreparedContract, StateView, ValueBinding, ValueSource};

    #[test]
    fn audio_records_reserve_whole_input_bytes_and_preserve_image_accounting() {
        let mut contract: PreparedContract = serde_json::from_str(include_str!(
            "../../tests/support/contracts/qwen38.json")).unwrap();
        assert_eq!(input_staging_bytes(&contract, |_| None).unwrap(), 0);
        contract.graphs.push(GraphContract {
            graph: "vision256".into(), kind: GraphOperation::Vision, state: vec![], state_rows_per_request: 1,
            draft_cursor: None, embedding_tables: vec![], pending_token: None, tuned_bindings: vec![],
            inputs: vec![ValueBinding { buffer: "pixels".into(), source: ValueSource::ImageTensor,
                indexing: InputIndexing::Scalar, value_offset: 0, padding: 0, tensor: "pixel_values".into(), audio_tensor: None }],
        });
        let image = input_staging_bytes(&contract, |_| Some(8192)).unwrap();
        assert_eq!(image, 8192 + 4096);
        assert!(input_staging_bytes(&contract, |_| None).is_err());
        let view = || StateView { buffer: String::new(), offset: 0, stride: 1, nbytes: 1 };
        contract.audio = Some(AudioCapacity {
            feature_slots: 2, features_per_request: 8192, feature_width: 2048,
            frame_capacities: vec![100, 800], mel_bins: 128, frames_per_chunk: 100,
            features_per_chunk: 13, subsampling_factor: 8, attention_window_frames: 800,
            features: view(), generations: view(), counts: view(),
        });
        assert_eq!(input_staging_bytes(&contract, |_| Some(8192)).unwrap(), 8192 * 8 * 128 * 4 + 4096);
        contract.audio.as_mut().unwrap().features_per_request = usize::MAX;
        assert!(input_staging_bytes(&contract, |_| Some(8192)).is_err());
        let audio = contract.audio.as_mut().unwrap();
        audio.features_per_request = 1;
        audio.feature_width = 1;
        audio.frames_per_chunk = 1;
        audio.features_per_chunk = 1;
        audio.subsampling_factor = usize::MAX;
        audio.attention_window_frames = 1;
        audio.frame_capacities = vec![1];
        assert_eq!(audio.feature_stride().unwrap(), 2);
        assert_eq!(input_staging_bytes(&contract, |_| Some(8192)).unwrap(), image);
        contract.graphs.clear();
        assert_eq!(input_staging_bytes(&contract, |_| None).unwrap(), 128 * 4 + 4096);
        contract.audio = None;
        assert_eq!(input_staging_bytes(&contract, |_| None).unwrap(), 0);
    }
}
