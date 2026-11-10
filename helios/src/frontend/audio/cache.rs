/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Whole-input FP32 features in the existing bounded prepared-input store.

use super::{MELS, PreparedAudio};
use flare_selene::input::{InputStore, PreparedInputRecord};
use std::sync::Arc;

const FORMAT: &[u8] = b"flare-centered-slaney-f32-v1\0";

pub(super) fn key(identity: [u8; 32]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(FORMAT);
    hash.update(identity);
    hash.finalize().into()
}

pub(super) fn lookup(store: &InputStore, key: [u8; 32], frames: usize) -> Option<Vec<f32>> {
    let record = store.get_blocking(key, true)?;
    let expected = frames.checked_mul(MELS)?.checked_mul(4)?;
    let valid = record.metadata.as_ref() == FORMAT
        && record.payload.as_ref().is_some_and(|payload| payload.len() == expected);
    if valid {
        let mel: Vec<_> = record.payload.as_ref().unwrap().chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect();
        if mel.iter().all(|value| value.is_finite()) { return Some(mel); }
    }
    store.invalidate(key);
    None
}

pub(super) fn insert(store: &InputStore, key: [u8; 32], audio: &PreparedAudio) -> crate::Result<()> {
    let payload_bytes = audio.mel.len().checked_mul(4).ok_or("audio feature record extent overflows")?;
    let serialized = payload_bytes.checked_add(FORMAT.len()).ok_or("audio feature record extent overflows")?;
    store.insert_with(key, serialized as u64, || {
        let mut payload = Vec::with_capacity(payload_bytes);
        for value in audio.mel.iter() { payload.extend_from_slice(&value.to_le_bytes()); }
        Ok(PreparedInputRecord { metadata: Arc::from(FORMAT), payload: Some(Arc::new(payload)) })
    })?;
    Ok(())
}
