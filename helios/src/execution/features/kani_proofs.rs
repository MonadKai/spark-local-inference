/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

use super::*;

#[kani::proof]
fn resolved_storage_and_slot_never_overflow_or_escape_the_buffer() {
    use crate::execution::Dtype;
    let dtype = match kani::any::<u8>() % 8 {
        0 => Dtype::Bool, 1 => Dtype::Uint8, 2 => Dtype::Float8E4m3fn,
        3 => Dtype::Int32, 4 => Dtype::Int64, 5 => Dtype::Float32,
        6 => Dtype::Float16, _ => Dtype::Bfloat16,
    };
    let buffer = BufferDescriptor { name: String::new(), address: kani::any(),
        nbytes: kani::any(), dtype,
        shape: Vec::new(), strides: Vec::new() };
    let view = StateView { buffer: String::new(), offset: kani::any(),
        stride: kani::any(), nbytes: kani::any() };
    let slots: usize = kani::any();
    let index: usize = kani::any();
    let nbytes: usize = kani::any();
    if let Some(storage) = StridedBytes::from_storage(&buffer, &view, slots) {
        assert!(storage.address >= buffer.address && storage.address != 0);
        assert_eq!(storage.address % dtype.size(), 0);
        assert!(storage.end <= buffer.address.checked_add(buffer.nbytes).unwrap());
        if let Some(address) = storage.at(index, nbytes) {
            assert!(index < slots && nbytes > 0 && nbytes <= view.nbytes);
            assert_eq!(address, buffer.address.checked_add(view.offset).unwrap()
                .checked_add(index.checked_mul(view.stride).unwrap()).unwrap());
            assert!(address.checked_add(nbytes).unwrap() <= storage.end);
        }
    }
}

#[kani::proof]
fn every_valid_copy_is_accepted_including_the_last_slot_and_byte() {
    let address: usize = kani::any();
    let stride: usize = kani::any();
    let slots: usize = kani::any();
    let index: usize = kani::any();
    let offset: usize = kani::any();
    let nbytes: usize = kani::any();
    kani::assume(address != 0 && slots > 0 && stride > 0 && index < slots);
    kani::assume(nbytes > 0 && offset.checked_add(nbytes).is_some_and(|end| end <= stride));
    kani::assume(address.checked_add(offset).is_some_and(|v| v.is_multiple_of(2)));
    let extent = slots.checked_mul(stride);
    kani::assume(extent.is_some_and(|bytes| address.checked_add(bytes).is_some()));
    let buffer = BufferDescriptor { name: String::new(), address, nbytes: extent.unwrap(),
        dtype: super::super::Dtype::Bfloat16, shape: Vec::new(), strides: Vec::new() };
    let view = StateView { buffer: String::new(), offset, stride, nbytes };
    let storage = StridedBytes::from_storage(&buffer, &view, slots).unwrap();
    let resolved = storage.at(index, nbytes).unwrap();
    assert_eq!(resolved, address + index * stride + offset);
    if index == slots - 1 && offset + nbytes == stride {
        assert_eq!(resolved + nbytes, address + extent.unwrap());
    }
}

// These five conjuncts preserve every assertion of the combined owner/extent
// obligation. Independent storage views widen the old correlated-view domain.
fn symbolic_storage() -> StridedBytes {
    StridedBytes { address: kani::any(), stride: kani::any(), nbytes: kani::any(),
        slots: kani::any(), end: kani::any() }
}

fn symbolic_copy() -> (FeatureStorageLayout, RequestHandle, usize, usize) {
    (FeatureStorageLayout { payload: symbolic_storage(), generations: symbolic_storage(),
        counts: symbolic_storage(), max_count: kani::any(), width: kani::any(), request_slots: kani::any() },
        RequestHandle { slot: kani::any(), generation: kani::any() }, kani::any(), kani::any())
}

#[kani::proof]
fn feature_owner_and_count_are_valid_before_copy() {
    let (layout, owner, slot, count) = symbolic_copy();
    if layout.addresses(owner, slot, count).is_some() {
        assert!(owner.generation > 0 && owner.generation <= i64::MAX as u64);
        assert!((owner.slot as usize) < layout.request_slots);
        assert!(count > 0 && count <= layout.max_count && count <= i32::MAX as usize);
    }
}

#[kani::proof]
fn feature_payload_extent_is_exact_and_bounded() {
    let (layout, owner, slot, count) = symbolic_copy();
    if let Some(addresses) = layout.addresses(owner, slot, count) {
        assert_eq!(addresses.nbytes, count.checked_mul(layout.width).unwrap().checked_mul(2).unwrap());
        assert!(addresses.nbytes > 0 && addresses.nbytes <= layout.payload.nbytes);
    }
}

#[kani::proof]
fn feature_payload_uses_its_feature_slot() {
    let (layout, owner, slot, count) = symbolic_copy();
    if let Some(addresses) = layout.addresses(owner, slot, count) {
        assert_eq!(addresses.payload, layout.payload.at(slot, addresses.nbytes).unwrap());
    }
}

#[kani::proof]
fn feature_generation_uses_its_request_slot() {
    let (layout, owner, slot, count) = symbolic_copy();
    if let Some(addresses) = layout.addresses(owner, slot, count) {
        assert_eq!(addresses.generation, layout.generations.at(owner.slot as usize, 8).unwrap());
    }
}

#[kani::proof]
fn feature_count_uses_its_request_slot() {
    let (layout, owner, slot, count) = symbolic_copy();
    if let Some(addresses) = layout.addresses(owner, slot, count) {
        assert_eq!(addresses.count, layout.counts.at(owner.slot as usize, 4).unwrap());
    }
}

#[kani::proof]
#[kani::unwind(5)]
fn disjoint_storage_cannot_alias_any_contained_nonempty_copies() {
    let ranges = [symbolic_storage(), symbolic_storage(), symbolic_storage()];
    let starts: [usize; 3] = kani::any();
    let ends: [usize; 3] = kani::any();
    if disjoint_ranges(ranges) {
        for first in 0..3 {
            for second in first + 1..3 {
                if ranges[first].address <= starts[first] && starts[first] < ends[first] && ends[first] <= ranges[first].end
                    && ranges[second].address <= starts[second] && starts[second] < ends[second] && ends[second] <= ranges[second].end
                {
                    assert!(ends[first] <= starts[second] || ends[second] <= starts[first]);
                }
            }
        }
    }
}
