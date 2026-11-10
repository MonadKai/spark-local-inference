/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Immutable whole-input BF16 features, separate from FP32 frontend records.
//!
//! The producer is the effective authenticated package/deployment identity, not
//! a recipe name. The input identity covers actual PCM and frontend settings.
//! Slot generations are supplied by the current publication, never retained in
//! these records. A corrupt retained value is a miss; a nonfinite live producer
//! value is an error and cannot be converted to a successful cache insertion.

use std::{sync::Arc, task::Poll};

use flare_selene::input::{InputKey, InputLookup, InputStore, PreparedInputRecord};
use sha2::{Digest, Sha256};

use crate::Result;

const FORMAT: &[u8] = b"flare-encoded-feature-bf16-v1\0";

#[derive(Clone, Copy)]
pub(crate) struct Geometry {
    count: usize,
    width: usize,
    nbytes: usize,
}

impl Geometry {
    pub fn new(count: usize, width: usize) -> Result<Self> {
        if count == 0 || width == 0 {
            return Err("encoded features require a positive count and width".into());
        }
        let nbytes = count.checked_mul(width).and_then(|elements| elements.checked_mul(2))
            .ok_or("encoded feature extent overflows")?;
        Ok(Self { count, width, nbytes })
    }

    pub fn nbytes(self) -> usize { self.nbytes }

    fn metadata(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(FORMAT.len() + 16);
        bytes.extend_from_slice(FORMAT);
        bytes.extend_from_slice(&(self.count as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.width as u64).to_le_bytes());
        bytes
    }

    pub fn key(self, producer: [u8; 32], input: [u8; 32]) -> InputKey {
        let mut hash = Sha256::new();
        hash.update(self.metadata());
        hash.update(producer);
        hash.update(input);
        hash.finalize().into()
    }

    fn valid_payload(self, bytes: &[u8]) -> bool {
        bytes.len() == self.nbytes && bytes.chunks_exact(2).all(|value|
            u16::from_le_bytes([value[0], value[1]]) & 0x7f80 != 0x7f80)
    }

    /// Call inside InputStore::insert_with, after its reservation succeeds.
    pub fn record(self, bytes: Vec<u8>) -> Result<PreparedInputRecord> {
        if !self.valid_payload(&bytes) {
            return Err("live encoded features have an invalid extent or nonfinite value".into());
        }
        Ok(PreparedInputRecord { metadata: self.metadata().into(), payload: Some(Arc::new(bytes)) })
    }

    pub fn lookup(self, store: &InputStore, producer: [u8; 32], input: [u8; 32]) -> Option<Lookup> {
        let key = self.key(producer, input);
        let pending = store.lookup(key, true)?;
        Some(Lookup { store: store.clone(), geometry: self, key, pending })
    }

    pub fn serialized_bytes(self) -> Result<u64> {
        self.nbytes.checked_add(FORMAT.len() + 16)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| "encoded feature record extent overflows".into())
    }
}

pub(crate) struct Lookup {
    store: InputStore,
    geometry: Geometry,
    key: InputKey,
    pending: InputLookup,
}

impl Lookup {
    /// Poll on the model worker without waiting for a cache thread or disk.
    /// A ready record retains the store's existing memory/read-lease accounting.
    pub fn poll(&self) -> Poll<Option<Arc<PreparedInputRecord>>> {
        let record = match self.pending.poll() {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(None) => return Poll::Ready(None),
            Poll::Ready(Some(record)) => record,
        };
        if record.metadata.as_ref() != self.geometry.metadata()
            || record.payload.as_ref().is_none_or(|payload| !self.geometry.valid_payload(payload))
        {
            self.store.invalidate(self.key);
            return Poll::Ready(None);
        }
        Poll::Ready(Some(record))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flare_selene::input::DiskBudget;
    use std::time::{Duration, Instant};

    fn payload() -> Vec<u8> {
        [0u16, 0x8000, 0x3f80, 0x7f7f].into_iter().flat_map(u16::to_le_bytes).collect()
    }

    fn ready(lookup: Lookup) -> Option<Arc<PreparedInputRecord>> {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            match lookup.poll() {
                Poll::Ready(value) => return value,
                Poll::Pending => {
                    assert!(Instant::now() < deadline);
                    std::thread::yield_now();
                }
            }
        }
    }

    #[test]
    fn encoded_keys_bind_the_real_input_producer_and_geometry() {
        let geometry = Geometry::new(2, 2).unwrap();
        let key = geometry.key([1; 32], [2; 32]);
        assert_eq!(key, geometry.key([1; 32], [2; 32]));
        assert_ne!(key, geometry.key([3; 32], [2; 32]));
        assert_ne!(key, geometry.key([1; 32], [4; 32]));
        assert_ne!(key, Geometry::new(1, 4).unwrap().key([1; 32], [2; 32]));
        assert_ne!(key, Geometry::new(2, 3).unwrap().key([1; 32], [2; 32]));
        assert!(Geometry::new(0, 2).is_err());
        assert!(Geometry::new(1, 0).is_err());
        assert!(Geometry::new(usize::MAX, 2).is_err());
    }

    #[test]
    fn live_records_preserve_bytes_and_reject_nonfinite_or_bad_extents() {
        let geometry = Geometry::new(2, 2).unwrap();
        let record = geometry.record(payload()).unwrap();
        assert_eq!(record.payload.as_ref().unwrap().as_slice(), payload());
        assert_eq!(geometry.serialized_bytes().unwrap(), (record.metadata.len() + 8) as u64);
        assert!(geometry.record(vec![0; 7]).is_err());
        assert!(geometry.record(vec![0; 9]).is_err());
        for bits in [0x7f80u16, 0xff80, 0x7f81, 0x7fc0, 0xffff] {
            let mut bytes = payload();
            bytes[..2].copy_from_slice(&bits.to_le_bytes());
            assert!(geometry.record(bytes).is_err());
        }
    }

    #[test]
    fn retained_corruption_is_invalidated_and_a_dropped_waiter_does_not_poison_reuse() {
        let store = InputStore::start(4096, None, DiskBudget::new(0, 0)).unwrap();
        let geometry = Geometry::new(2, 2).unwrap();
        let key = geometry.key([1; 32], [2; 32]);
        for kind in 0..4 {
            let mut record = geometry.record(payload()).unwrap();
            match kind {
                0 => record.metadata = Arc::from(&b"other frontend format"[..]),
                1 => record.payload = None,
                2 => record.payload = Some(Arc::new(vec![0; 7])),
                3 => {
                    let mut bytes = payload();
                    bytes[..2].copy_from_slice(&0x7f80u16.to_le_bytes());
                    record.payload = Some(Arc::new(bytes));
                }
                _ => unreachable!(),
            }
            store.insert(key, Arc::new(record));
            assert!(ready(geometry.lookup(&store, [1; 32], [2; 32]).unwrap()).is_none());
            assert!(store.get_blocking(key, true).is_none());
        }
        store.insert(key, Arc::new(geometry.record(payload()).unwrap()));
        drop(geometry.lookup(&store, [1; 32], [2; 32]));
        let found = ready(geometry.lookup(&store, [1; 32], [2; 32]).unwrap()).unwrap();
        assert_eq!(found.payload.as_ref().unwrap().as_slice(), payload());
        assert!(ready(geometry.lookup(&store, [1; 32], [9; 32]).unwrap()).is_none());
    }
}
