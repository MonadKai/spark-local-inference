/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Copies immutable encoded bytes into declared, generation-owned storage.
//! Computation and encoder state updates remain in the prepared graphs.

use crate::{Result, cuda::{Event, PinnedHost, Stream}, model::contract::{AudioCapacity, StateView},
    scheduler::RequestHandle};
use super::{BufferDescriptor, ExecutionManifest};

#[cfg(kani)]
mod kani_proofs;

#[derive(Clone, Copy, Debug)]
struct StridedBytes {
    address: usize,
    stride: usize,
    nbytes: usize,
    slots: usize,
    end: usize,
}

impl StridedBytes {
    fn resolve(manifest: &ExecutionManifest, view: &StateView, slots: usize) -> Result<Self> {
        let buffer = manifest.buffer(&view.buffer)?;
        Self::from_storage(buffer, view, slots).ok_or_else(|| "invalid resolved feature storage".into())
    }

    fn from_storage(buffer: &BufferDescriptor, view: &StateView, slots: usize) -> Option<Self> {
        let extent = slots.checked_mul(view.stride)?;
        let address = buffer.address.checked_add(view.offset)?;
        let end = buffer.address.checked_add(extent)?;
        if slots == 0 || buffer.address == 0 || extent > buffer.nbytes
            || buffer.address.checked_add(buffer.nbytes).is_none()
            || view.nbytes == 0 || view.offset.checked_add(view.nbytes).is_none_or(|v| v > view.stride)
            || !address.is_multiple_of(buffer.dtype.size())
        {
            return None;
        }
        Some(Self { address, stride: view.stride, nbytes: view.nbytes, slots, end })
    }

    fn at(self, slot: usize, nbytes: usize) -> Option<usize> {
        let address = slot.checked_mul(self.stride).and_then(|v| self.address.checked_add(v))?;
        if slot >= self.slots || nbytes == 0 || nbytes > self.nbytes
            || address.checked_add(nbytes).is_none_or(|v| v > self.end)
        {
            return None;
        }
        Some(address)
    }
}

pub(super) struct FeatureStorageLayout {
    payload: StridedBytes,
    generations: StridedBytes,
    counts: StridedBytes,
    max_count: usize,
    width: usize,
    request_slots: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct Addresses {
    payload: usize,
    generation: usize,
    count: usize,
    nbytes: usize,
}

impl FeatureStorageLayout {
    pub(super) fn new(audio: &AudioCapacity, manifest: &ExecutionManifest) -> Result<Self> {
        audio.validate(manifest)?;
        let payload = StridedBytes::resolve(manifest, &audio.features, audio.feature_slots)?;
        let generations = StridedBytes::resolve(manifest, &audio.generations, manifest.request_capacity())?;
        let counts = StridedBytes::resolve(manifest, &audio.counts, manifest.request_capacity())?;
        if !disjoint_ranges([payload, generations, counts]) {
            return Err("encoded feature storage views overlap".into());
        }
        Ok(Self { payload, generations, counts, max_count: audio.features_per_request,
            width: audio.feature_width, request_slots: manifest.request_capacity() })
    }

    fn addresses(&self, owner: RequestHandle, feature_slot: usize, count: usize) -> Option<Addresses> {
        if !(1..=i64::MAX as u64).contains(&owner.generation)
            || owner.slot as usize >= self.request_slots || count == 0 || count > self.max_count
            || count > i32::MAX as usize
        {
            return None;
        }
        let nbytes = count.checked_mul(self.width).and_then(|v| v.checked_mul(2))?;
        Some(Addresses {
            payload: self.payload.at(feature_slot, nbytes)?,
            generation: self.generations.at(owner.slot as usize, 8)?,
            count: self.counts.at(owner.slot as usize, 4)?, nbytes,
        })
    }

    fn staging_bytes(&self) -> Result<usize> {
        self.payload.nbytes.checked_add(12).ok_or_else(|| "feature staging extent overflows".into())
    }
}

fn disjoint_ranges(ranges: [StridedBytes; 3]) -> bool {
    for (index, left) in ranges.iter().enumerate() {
        if ranges[index + 1..].iter().any(|right|
            left.address < right.end && right.address < left.end)
        { return false; }
    }
    true
}

pub(crate) fn finite_bf16(bytes: &[u8]) -> bool {
    bytes.len().is_multiple_of(2) && bytes.chunks_exact(2).all(|v|
        u16::from_le_bytes([v[0], v[1]]) & 0x7f80 != 0x7f80)
}

/// Startup-owned pinned staging; retained with executor streams through drain.
pub(super) struct FeatureTransfer {
    layout: FeatureStorageLayout,
    slab: PinnedHost,
    completed: Event,
}

impl FeatureTransfer {
    #[cfg(test)]
    pub(super) fn probe(&self, context: usize) -> crate::cuda::EventProbe {
        self.completed.probe(context)
    }

    pub(super) fn new(layout: FeatureStorageLayout, stream: &Stream) -> Result<Self> {
        Ok(Self { slab: stream.host_buffer(layout.staging_bytes()?)?,
            completed: stream.blocking_event()?, layout })
    }

    pub(super) fn load(&mut self, stream: &Stream, owner: RequestHandle, slot: usize,
        count: usize, bytes: &[u8]) -> Result<()> {
        let addresses = self.layout.addresses(owner, slot, count)
            .ok_or("feature upload exceeds its declared owner, slot or extent")?;
        if bytes.len() != addresses.nbytes || !finite_bf16(bytes) {
            return Err("encoded feature upload is incomplete or nonfinite".into());
        }
        let metadata = self.layout.payload.nbytes;
        let slab = self.slab.bytes_mut();
        slab[..bytes.len()].copy_from_slice(bytes);
        slab[metadata..metadata + 8].copy_from_slice(&owner.generation.to_le_bytes());
        slab[metadata + 8..metadata + 12].copy_from_slice(&(count as i32).to_le_bytes());
        // SAFETY: All addresses are checked declared slots. The slab is fixed
        // and retained by Resources until both streams confirm drain. No graph
        // is outstanding; the event precedes scheduler publication and reuse.
        unsafe {
            stream.upload(addresses.payload, &self.slab, 0, addresses.nbytes)?;
            stream.upload(addresses.generation, &self.slab, metadata, 8)?;
            stream.upload(addresses.count, &self.slab, metadata + 8, 4)?;
        }
        stream.record(&self.completed)?;
        self.completed.synchronize()
    }

    pub(super) fn read(&mut self, stream: &Stream, owner: RequestHandle, slot: usize,
        count: usize, destination: &mut [u8]) -> Result<()> {
        let addresses = self.layout.addresses(owner, slot, count)
            .ok_or("feature readback exceeds its declared owner, slot or extent")?;
        if destination.len() != addresses.nbytes {
            return Err("encoded feature readback has the wrong extent".into());
        }
        // SAFETY: The graph has completed and the checked arena remains owned.
        // Resources retains this slab through a failed event/stream drain.
        let metadata = self.layout.payload.nbytes;
        unsafe {
            stream.download(&self.slab, addresses.payload, addresses.nbytes)?;
            stream.download_at(&self.slab, metadata, addresses.generation, 8)?;
            stream.download_at(&self.slab, metadata + 8, addresses.count, 4)?;
        }
        stream.record(&self.completed)?;
        self.completed.synchronize()?;
        let slab = self.slab.bytes();
        if slab[metadata..metadata + 8] != owner.generation.to_le_bytes()
            || slab[metadata + 8..metadata + 12] != (count as i32).to_le_bytes()
        {
            return Err("encoded feature readback owner or count differs from its completion".into());
        }
        let bytes = &slab[..addresses.nbytes];
        if !finite_bf16(bytes) { return Err("live encoder produced nonfinite features".into()); }
        destination.copy_from_slice(bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{BufferDescriptor, Dtype};

    pub(super) fn fixture() -> (AudioCapacity, ExecutionManifest) {
        let mut manifest = crate::execution::tests::manifest();
        let mut view = |name: &str, address, dtype: Dtype, shape: Vec<usize>, strides: Vec<usize>| {
            let stride = strides[0] * dtype.size();
            manifest.buffers.push(BufferDescriptor { name: name.into(), address,
                nbytes: shape[0] * stride, dtype, shape, strides });
            StateView { buffer: name.into(), offset: 0, stride, nbytes: stride }
        };
        let features = view("retained.features", 0x100000, Dtype::Bfloat16, vec![2, 52, 4], vec![208, 4, 1]);
        let generations = view("retained.generations", 0x200000, Dtype::Int64, vec![64], vec![1]);
        let counts = view("retained.counts", 0x300000, Dtype::Int32, vec![64], vec![1]);
        (AudioCapacity { feature_slots: 2, features_per_request: 52, feature_width: 4,
            frame_capacities: vec![100, 400], mel_bins: 128, frames_per_chunk: 100,
            features_per_chunk: 13, subsampling_factor: 8, attention_window_frames: 400,
            features, generations, counts }, manifest)
    }

    #[test]
    fn copy_layout_distinguishes_feature_and_request_slots_and_full_generations() {
        let (audio, manifest) = fixture();
        let layout = FeatureStorageLayout::new(&audio, &manifest).unwrap();
        let owner = RequestHandle { slot: 1, generation: i64::MAX as u64 };
        assert_eq!(layout.addresses(owner, 0, 13).unwrap(), Addresses {
            payload: 0x100000, generation: 0x200008, count: 0x300004, nbytes: 104 });
        assert_eq!(layout.addresses(owner, 1, 52).unwrap().payload, 0x100000 + 416);
        assert_eq!(layout.staging_bytes().unwrap(), 428);
        for count in [0, 53, usize::MAX] { assert!(layout.addresses(owner, 0, count).is_none()); }
        assert!(layout.addresses(owner, 2, 13).is_none());
        for generation in [0, i64::MAX as u64 + 1, u64::MAX] {
            assert!(layout.addresses(RequestHandle { generation, ..owner }, 0, 13).is_none());
        }
        assert!(layout.addresses(RequestHandle { slot: 64, ..owner }, 0, 13).is_none());
    }

    #[test]
    fn copy_layout_rejects_alias_overflow_bad_type_and_partial_storage() {
        let (audio, manifest) = fixture();
        for kind in 0..5 {
            let mut bad = manifest.clone();
            let target = bad.buffers.iter_mut().find(|b| b.name == audio.counts.buffer).unwrap();
            match kind {
                0 => target.address = 0x100000,
                1 => target.address = usize::MAX - 3,
                2 => target.dtype = Dtype::Float32,
                3 => target.nbytes -= 1,
                4 => target.address += 1,
                _ => unreachable!(),
            }
            assert!(FeatureStorageLayout::new(&audio, &bad).is_err(), "accepted mutation {kind}");
        }
        let bad = StridedBytes { address: usize::MAX - 7, stride: 16, nbytes: 8, slots: 2, end: usize::MAX };
        assert!(bad.at(1, 8).is_none());
        assert!(!finite_bf16(&[0]));
        assert!(finite_bf16(&[0, 0, 0, 128, 127, 127]));
        assert!(!finite_bf16(&[128, 127]));
    }
}

#[cfg(all(test, feature = "runtime-package"))]
mod cuda_tests {
    use super::*;
    use crate::runtime_package::driver::Driver;
    use std::sync::{Arc, atomic::{AtomicBool, AtomicU32, Ordering}};

    struct CopyOwners {
        stream: Stream,
        transfer: FeatureTransfer,
        guards: PinnedHost,
        metadata_guards: Vec<(BufferDescriptor, PinnedHost)>,
        flag: PinnedHost,
        producer: PinnedHost,
    }

    struct Device {
        driver: Driver,
        context: *mut std::ffi::c_void,
        allocations: Vec<u64>,
        owners: Option<std::mem::ManuallyDrop<CopyOwners>>,
    }

    impl Device {
        fn new() -> Self {
            let driver = Driver::load().unwrap();
            let mut context = std::ptr::null_mut();
            unsafe {
                driver.check("cuInit", (driver.init)(0)).unwrap();
                let mut device = 0;
                driver.check("cuDeviceGet", (driver.device)(&mut device, 0)).unwrap();
                driver.check("cuCtxCreate_v2", (driver.context_create)(&mut context, 0, device)).unwrap();
            }
            Self { driver, context, allocations: Vec::new(), owners: None }
        }

        fn allocate(&mut self, bytes: usize) -> usize {
            let mut address = 0;
            unsafe { self.driver.check("cuMemAlloc_v2", (self.driver.allocate)(&mut address, bytes)).unwrap(); }
            self.allocations.push(address);
            address as usize
        }
    }

    impl Drop for Device {
        fn drop(&mut self) {
            unsafe {
                if self.driver.check("cuCtxSynchronize", (self.driver.context_sync)()).is_err() {
                    // All pinned allocations, streams, events and context-owned
                    // device addresses survive an unconfirmed test drain.
                    if let Some(owners) = self.owners.take() { std::mem::forget(owners); }
                    return;
                }
                if let Some(mut owners) = self.owners.take() { std::mem::ManuallyDrop::drop(&mut owners); }
                for address in &self.allocations {
                    let _ = self.driver.check("cuMemFree_v2", (self.driver.free)(*address));
                }
                let _ = self.driver.check("cuCtxDestroy_v2", (self.driver.context_destroy)(self.context));
            }
        }
    }

    #[test]
    #[ignore = "requires an exclusive CUDA device selected through CUDA_VISIBLE_DEVICES"]
    fn feature_copies_wait_for_the_fence_and_preserve_owner_slots_and_guards() {
        let mut device = Device::new();
        let stream = Stream::open(device.context as usize, 0).unwrap();
        let (audio, mut manifest) = tests::fixture();
        for name in [&audio.features.buffer, &audio.generations.buffer, &audio.counts.buffer] {
            let buffer = manifest.buffers.iter_mut().find(|b| &b.name == name).unwrap();
            buffer.address = device.allocate(buffer.nbytes);
        }
        let layout = FeatureStorageLayout::new(&audio, &manifest).unwrap();
        let transfer = FeatureTransfer::new(layout, &stream).unwrap();
        let payload_buffer = manifest.buffer(&audio.features.buffer).unwrap();
        let mut guards = stream.host_buffer(payload_buffer.nbytes).unwrap();
        for bytes in guards.bytes_mut().chunks_exact_mut(2) { bytes.copy_from_slice(&0x3f00u16.to_le_bytes()); }
        let mut metadata_guards = Vec::new();
        for name in [&audio.generations.buffer, &audio.counts.buffer] {
            let buffer = manifest.buffer(name).unwrap();
            let mut host = stream.host_buffer(buffer.nbytes).unwrap();
            host.bytes_mut().fill(0xa5);
            metadata_guards.push((buffer.clone(), host));
        }
        let owner = RequestHandle { slot: 3, generation: i64::MAX as u64 - 2 };
        let payload: Vec<u8> = (0..52).map(|index| if index % 2 == 0 { 0x3f80u16 } else { 0x8000 })
            .flat_map(u16::to_le_bytes).collect();
        let flag = stream.mapped_host_buffer(4).unwrap();
        let mut producer = stream.host_buffer(payload.len()).unwrap();
        for bytes in producer.bytes_mut().chunks_exact_mut(2) { bytes.copy_from_slice(&0x4000u16.to_le_bytes()); }
        // Register every asynchronous owner before the first queued copy. The
        // Device drop boundary drains before releasing any of these fields.
        device.owners = Some(std::mem::ManuallyDrop::new(CopyOwners {
            stream, transfer, guards, metadata_guards, flag, producer,
        }));
        let CopyOwners { stream, transfer, guards, metadata_guards, flag, producer } =
            &mut **device.owners.as_mut().unwrap();
        unsafe { stream.upload(payload_buffer.address, guards, 0, payload_buffer.nbytes).unwrap(); }
        for (buffer, host) in metadata_guards.iter() {
            unsafe { stream.upload(buffer.address, host, 0, buffer.nbytes).unwrap(); }
        }
        stream.synchronize().unwrap();
        flag.bytes_mut().copy_from_slice(&0u32.to_ne_bytes());
        let address = stream.mapped_address(&flag).unwrap();
        let host_address = flag.address();
        let released = Arc::new(AtomicBool::new(false));
        transfer.load(&stream, owner, 1, 13, &payload).unwrap();
        // SAFETY: The mapped flag is retained until this stream's fence and the
        // releasing thread complete. No CPU byte-slice access overlaps the store.
        std::thread::scope(|scope| {
            let release = released.clone();
            scope.spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                release.store(true, Ordering::Release);
                unsafe { AtomicU32::from_ptr(host_address as *mut u32).store(1, Ordering::Release); }
            });
            unsafe { stream.wait_value(address, 1).unwrap(); }
            transfer.load(stream, owner, 1, 13, &payload).unwrap();
            assert!(released.load(Ordering::Acquire), "feature publication returned before its GPU fence");
        });
        let mut actual = vec![0; payload.len()];
        // Read back different GPU-produced data into a poisoned slab. Reusing
        // the upload bytes would conceal missing D2H work or an omitted wait.
        transfer.slab.bytes_mut().fill(0xcd);
        released.store(false, Ordering::Release);
        std::thread::scope(|scope| {
            let release = released.clone();
            scope.spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                release.store(true, Ordering::Release);
                unsafe { AtomicU32::from_ptr(host_address as *mut u32).store(2, Ordering::Release); }
            });
            unsafe {
                stream.wait_value(address, 2).unwrap();
                stream.upload(payload_buffer.address + audio.features.stride, producer, 0, payload.len()).unwrap();
            }
            transfer.read(stream, owner, 1, 13, &mut actual).unwrap();
            assert!(released.load(Ordering::Acquire), "readback returned before its GPU fence");
            assert_eq!(actual, producer.bytes());
        });
        let slab_address = transfer.slab.address();
        for _ in 0..100 {
            transfer.load(&stream, owner, 1, 13, &payload).unwrap();
            transfer.read(&stream, owner, 1, 13, &mut actual).unwrap();
            assert_eq!(actual, payload);
            assert_eq!(transfer.slab.address(), slab_address);
        }
        assert!(transfer.read(&stream, RequestHandle { generation: owner.generation - 1, ..owner },
            1, 13, &mut actual).is_err());
        assert!(transfer.read(&stream, owner, 1, 12, &mut vec![0; 96]).is_err());
        assert!(transfer.load(&stream, owner, 2, 13, &payload).is_err());
        let mut nonfinite = payload.clone();
        nonfinite[..2].copy_from_slice(&0x7f80u16.to_le_bytes());
        assert!(transfer.load(&stream, owner, 1, 13, &nonfinite).is_err());
        assert!(transfer.load(&stream, owner, 1, 13, &payload[..100]).is_err());
        unsafe { stream.download(&guards, payload_buffer.address, payload_buffer.nbytes).unwrap(); }
        stream.synchronize().unwrap();
        let start = audio.features.stride;
        assert_eq!(&guards.bytes()[start..start + payload.len()], payload);
        assert!(guards.bytes()[..start].chunks_exact(2).chain(guards.bytes()[start + payload.len()..].chunks_exact(2))
            .all(|bytes| bytes == 0x3f00u16.to_le_bytes()));
        for (buffer, host) in metadata_guards.iter_mut() {
            unsafe { stream.download(host, buffer.address, buffer.nbytes).unwrap(); }
            stream.synchronize().unwrap();
            let expected = if buffer.dtype == super::super::Dtype::Int64 {
                owner.generation.to_le_bytes().to_vec()
            } else { 13i32.to_le_bytes().to_vec() };
            for (index, bytes) in host.bytes().chunks_exact(expected.len()).enumerate() {
                if index == owner.slot as usize { assert_eq!(bytes, expected); }
                else { assert!(bytes.iter().all(|byte| *byte == 0xa5)); }
            }
        }
    }
}
