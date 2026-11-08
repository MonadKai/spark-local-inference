/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native callbacks used by the preparation process's PyTorch memory pool.

use std::{
    collections::BTreeMap,
    ffi::{CString, c_char, c_void},
    ptr::null_mut,
    sync::Mutex,
};

use crate::runtime_package::driver::{Access, AllocationProperties, Driver, Location};
use crate::{Error, ExecutionError, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Manifest, message)
}

// PyTorch's CPU allocator is mimalloc, which places its 1 GiB arenas at random addresses
// in 2-6 TiB; a peer rank allocates CPU tensors before reserving these ranges.
#[cfg(target_os = "linux")]
const BASE: u64 = 0x80000000000;
#[cfg(target_os = "windows")]
const BASE: u64 = 0x40000000000;
const EXTENT: usize = 1 << 40;
// Keep small Linux exports within the Windows WDDM address range as well.
const HOST_BASE: u64 = 0xc0000000000;

fn host_base(device_base: u64, world: u32) -> u64 {
    if device_base + u64::from(world) * EXTENT as u64 <= HOST_BASE {
        HOST_BASE
    } else {
        0x200000000000
    }
}

#[derive(Default)]
struct RetainedBacking {
    active: bool,
    aborted: bool,
    finished: bool,
    reference_bytes: usize,
    handles: Vec<(u64, usize)>,
}

impl RetainedBacking {
    fn begin(&mut self) -> Result<()> {
        if self.active || self.aborted || !self.handles.is_empty() {
            return Err(invalid(
                "physical retention scope is already active or abandoned",
            ));
        }
        self.active = true;
        self.finished = false;
        Ok(())
    }

    fn defer_free(&self) -> Result<bool> {
        if self.aborted || (!self.active && !self.finished) {
            return Err(invalid(
                "transient free outside a healthy physical retention scope",
            ));
        }
        Ok(self.active)
    }

    fn may_reuse(&self) -> Result<bool> {
        if !self.active || self.aborted {
            return Err(invalid(
                "transient allocation outside an active physical retention scope",
            ));
        }
        Ok(true)
    }

    fn retain(&mut self, size: usize, acquire: impl FnOnce() -> Result<u64>) -> Result<()> {
        if !self.active || self.aborted {
            return Err(invalid(
                "transient retention outside an active physical retention scope",
            ));
        }
        let reference_bytes = self
            .reference_bytes
            .checked_add(size)
            .ok_or_else(|| invalid("retained physical byte count overflow"))?;
        self.handles
            .try_reserve(1)
            .map_err(|e| invalid(format!("retained handle storage: {e}")))?;
        let handle = acquire()?;
        self.handles.push((handle, size));
        self.reference_bytes = reference_bytes;
        Ok(())
    }

    fn check_finishable(&self) -> Result<()> {
        if !self.active || self.aborted {
            return Err(invalid("physical retention scope is not active"));
        }
        Ok(())
    }

    fn finish(&mut self, mut release: impl FnMut(u64) -> Result<()>) -> Result<()> {
        if !self.active || self.aborted {
            return Err(invalid("physical retention scope is not active"));
        }
        while let Some(&(handle, size)) = self.handles.last() {
            release(handle)?;
            self.handles.pop();
            self.reference_bytes -= size;
        }
        self.active = false;
        self.finished = true;
        Ok(())
    }

    fn abort_on_free_error(&mut self, error: Error) -> Option<Error> {
        if self.aborted {
            return None;
        }
        self.abort();
        Some(invalid(format!(
            "{error}; retained reference bytes={}, handles={}",
            self.reference_bytes,
            self.handles.len()
        )))
    }

    fn abort(&mut self) {
        self.active = false;
        self.aborted = true;
    }
}

struct RetainedMapping {
    address: u64,
    size: usize,
}

#[derive(Default)]
struct RetainedMappingPool {
    // Buckets are never empty: insert creates the vector it pushes into, and
    // every removal drops a bucket when it empties.
    mappings: BTreeMap<usize, Vec<RetainedMapping>>,
}

impl RetainedMappingPool {
    fn take(&mut self, size: usize) -> Option<RetainedMapping> {
        let mappings = self.mappings.get_mut(&size)?;
        debug_assert!(!mappings.is_empty());
        let mapping = mappings.pop()?;
        if mappings.is_empty() {
            self.mappings.remove(&size);
        }
        Some(mapping)
    }

    fn insert(&mut self, mapping: RetainedMapping) {
        self.mappings.entry(mapping.size).or_default().push(mapping);
    }

    fn pop(&mut self) -> Option<RetainedMapping> {
        let size = self.mappings.keys().next().copied()?;
        let mappings = self.mappings.get_mut(&size)?;
        debug_assert!(!mappings.is_empty());
        let mapping = mappings.pop()?;
        if mappings.is_empty() {
            self.mappings.remove(&size);
        }
        Some(mapping)
    }

    fn len(&self) -> usize {
        self.mappings.values().map(Vec::len).sum()
    }

    fn bytes(&self) -> usize {
        self.mappings
            .iter()
            .map(|(&size, handles)| size.saturating_mul(handles.len()))
            .sum()
    }
}

// A failed driver call may report an earlier asynchronous failure. Do not retry
// a possibly released handle or reuse a possibly mapped address. The process
// retains this owner and the rest of the arena until process exit.
struct AbandonedAllocation {
    address: u64,
    size: usize,
    creator_reference: Option<u64>,
    mapping_confirmed: bool,
}

struct Arena {
    driver: Driver,
    device: i32,
    granularity: usize,
    total_memory: usize,
    gpudirect: bool,
    host_base: u64,
    free: Vec<(u64, usize)>,
    mappings: BTreeMap<u64, usize>,
    retained: RetainedBacking,
    retained_mappings: RetainedMappingPool,
    abandoned: Option<AbandonedAllocation>,
}

impl Arena {
    fn check_healthy(&self) -> Result<()> {
        if let Some(owner) = &self.abandoned {
            return Err(invalid(format!(
                "native package allocator abandoned allocation {:#x}+{:#x}, creator reference={:?}, mapping confirmed={}",
                owner.address, owner.size, owner.creator_reference, owner.mapping_confirmed
            )));
        }
        Ok(())
    }
}
static ARENA: Mutex<Option<Arena>> = Mutex::new(None);
static ERROR: Mutex<Option<CString>> = Mutex::new(None);

pub(crate) struct MemoryInventory {
    pub device: i32,
    pub free: usize,
    pub total: usize,
    pub mappings: Vec<(u64, usize)>,
}

pub(crate) fn memory_inventory() -> Result<MemoryInventory> {
    let arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let a = arena
        .as_ref()
        .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
    a.check_healthy()?;
    let (_, device) = crate::current_context()?;
    if device != a.device {
        return Err(invalid(
            "memory inventory requires the package rank's CUDA context",
        ));
    }
    let (mut free, mut total) = (0, 0);
    // SAFETY: Preparation retains the current context and both outputs are valid.
    unsafe {
        a.driver
            .check("cuCtxSynchronize", (a.driver.context_sync)())?;
        a.driver.check(
            "cuMemGetInfo_v2",
            (a.driver.memory_info)(&mut free, &mut total),
        )?;
    }
    Ok(MemoryInventory {
        device,
        free,
        total,
        mappings: a
            .mappings
            .iter()
            .map(|(&address, &size)| (address, size))
            .collect(),
    })
}

fn properties(device: i32, gpudirect: bool) -> AllocationProperties {
    AllocationProperties {
        kind: 1,
        handles: i32::from(cfg!(target_os = "linux")),
        location: Location { kind: 1, device },
        windows: null_mut(),
        // cuFile registration requires gpuDirectRDMACapable on VMM allocations.
        flags: [0, u8::from(gpudirect), 0, 0, 0, 0, 0, 0],
    }
}

fn failure(error: impl ToString) {
    ERROR
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(|| CString::new(error.to_string().replace('\0', " ")).unwrap());
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_initialize(rank: u32, world: u32, device: i32) -> i32 {
    let initialize = || -> Result<()> {
        let mut arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        if arena.is_some() {
            return Err(invalid("native package allocator is already initialized"));
        }
        if world == 0 || rank >= world || device < 0 {
            return Err(invalid("invalid package rank or device"));
        }
        let host_base = if cfg!(target_os = "windows") {
            HOST_BASE
        } else {
            host_base(BASE, world)
        };
        if u64::from(world) * EXTENT as u64 > BASE.abs_diff(host_base) {
            return Err(invalid(
                "package host reservations would overlap device reservations",
            ));
        }
        let driver = Driver::load()?;
        let (mut address, mut granularity, mut gpudirect, mut total_memory) = (0, 0, 0, 0);
        unsafe {
            driver.check(
                "cuDeviceTotalMem_v2",
                (driver.device_total_memory)(&mut total_memory, device),
            )?;
            driver.check(
                "cuDeviceGetAttribute(GPU_DIRECT_RDMA_SUPPORTED)",
                (driver.device_attribute)(&mut gpudirect, 116, device),
            )?;
            driver.check(
                "cuMemGetAllocationGranularity",
                (driver.granularity)(&mut granularity, &properties(device, gpudirect != 0), 0),
            )?;
            let mut reserved = Vec::new();
            for peer in 0..world {
                let base = BASE + u64::from(peer) * EXTENT as u64;
                let status = (driver.reserve)(&mut address, EXTENT, 0, base, 0);
                if status != 0 || address != base {
                    if status == 0 {
                        (driver.release_address)(address, EXTENT);
                    }
                    for address in reserved {
                        (driver.release_address)(address, EXTENT);
                    }
                    driver.check("cuMemAddressReserve", status)?;
                    return Err(invalid(format!(
                        "native package device address range {base:#x}+{EXTENT:#x} is unavailable; CUDA returned {address:#x}"
                    )));
                }
                reserved.push(address);
            }
        }
        *arena = Some(Arena {
            driver,
            device,
            granularity,
            total_memory,
            gpudirect: gpudirect != 0,
            host_base: host_base + u64::from(rank) * EXTENT as u64,
            free: vec![(BASE + u64::from(rank) * EXTENT as u64, EXTENT / 2)],
            mappings: BTreeMap::new(),
            retained: RetainedBacking::default(),
            retained_mappings: RetainedMappingPool::default(),
            abandoned: None,
        });
        Ok(())
    };
    match initialize() {
        Ok(()) => 0,
        Err(error) => {
            failure(error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_allocate(
    size: usize,
    device: i32,
    _stream: *mut c_void,
) -> *mut c_void {
    package_allocate(size, device, false)
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_transient_allocate(
    size: usize,
    device: i32,
    _stream: *mut c_void,
) -> *mut c_void {
    package_allocate(size, device, true)
}

fn package_allocate(size: usize, device: i32, transient: bool) -> *mut c_void {
    let allocate = || -> Result<u64> {
        let mut guard = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        let a = guard
            .as_mut()
            .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
        a.check_healthy()?;
        if device != a.device {
            return Err(invalid("allocation device differs from package rank"));
        }
        let size = size
            .checked_add(a.granularity - 1)
            .ok_or_else(|| invalid("allocation size overflow"))?
            / a.granularity
            * a.granularity;
        if transient && a.retained.may_reuse()? {
            if let Some(mapping) = a.retained_mappings.take(size) {
                a.mappings.insert(mapping.address, mapping.size);
                return Ok(mapping.address);
            }
        }
        // Impossible allocations can make an integrated GPU's driver reclaim
        // host memory for a long time before returning OOM. Return the same
        // retryable null result without entering that driver allocation.
        if size > a.total_memory {
            return Ok(0);
        }
        let index = a
            .free
            .iter()
            .position(|(_, n)| *n >= size)
            .ok_or_else(|| invalid("native package address range exhausted"))?;
        let (address, available) = a.free[index];
        let mut physical = 0;
        unsafe {
            a.driver.check(
                "cuMemCreate",
                (a.driver.physical_create)(
                    &mut physical,
                    size,
                    &properties(device, a.gpudirect),
                    0,
                ),
            )?;
        }
        // OOM above remains retryable. After successful creation, reserve this
        // VA even if a later call fails; PyTorch must never retry over it.
        if size == available {
            a.free.remove(index);
        } else {
            a.free[index] = (address + size as u64, available - size);
        }
        let mut owner = AbandonedAllocation {
            address,
            size,
            creator_reference: Some(physical),
            mapping_confirmed: false,
        };
        let mapped = (|| -> Result<()> {
            unsafe {
                a.driver
                    .check("cuMemMap", (a.driver.map)(address, size, 0, physical, 0))?;
                owner.mapping_confirmed = true;
                a.driver
                    .check("cuMemRelease", (a.driver.physical_release)(physical))?;
                owner.creator_reference = None;
                let access = Access {
                    location: Location { kind: 1, device },
                    flags: 3,
                };
                a.driver.check(
                    "cuMemSetAccess",
                    (a.driver.access)(address, size, &access, 1),
                )?;
            }
            Ok(())
        })();
        if let Err(error) = mapped {
            a.abandoned = Some(owner);
            // Publish the originating error while ARENA still excludes other
            // callbacks; their quarantine errors must not win ERROR's first slot.
            failure(&error);
            return Err(error);
        }
        a.mappings.insert(address, size);
        Ok(address)
    };
    match allocate() {
        Ok(address) => address as *mut c_void,
        // PyTorch may release cached segments and retry a physical allocation.
        Err(Error::Execution(ExecutionError::Driver(error)))
            if error.operation == "cuMemCreate" && error.code == 2 =>
        {
            null_mut()
        }
        Err(error) => {
            failure(error);
            null_mut()
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_free(
    pointer: *mut c_void,
    _size: usize,
    _device: i32,
    _stream: *mut c_void,
) {
    package_free(pointer, false);
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_transient_free(
    pointer: *mut c_void,
    _size: usize,
    _device: i32,
    _stream: *mut c_void,
) {
    package_free(pointer, true);
}

fn package_free(pointer: *mut c_void, transient: bool) {
    let release = || -> Result<()> {
        let mut guard = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        let a = guard
            .as_mut()
            .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
        a.check_healthy()?;
        let address = pointer as u64;
        let size = *a
            .mappings
            .get(&address)
            .ok_or_else(|| invalid("unknown native package allocation"))?;
        let retain = transient && a.retained.defer_free()?;
        if retain {
            let driver = &a.driver;
            a.retained.retain(size, || {
                let mut handle = 0;
                unsafe {
                    driver.check(
                        "cuMemRetainAllocationHandle",
                        (driver.physical_retain)(&mut handle, pointer),
                    )?;
                }
                Ok(handle)
            })?;
        }
        if retain {
            a.mappings.remove(&address);
            a.retained_mappings
                .insert(RetainedMapping { address, size });
            return Ok(());
        }
        unsafe {
            a.driver
                .check("cuMemUnmap", (a.driver.unmap)(address, size))?;
        }
        a.mappings.remove(&address);
        release_free_range(&mut a.free, address, size);
        Ok(())
    };
    if let Err(error) = release() {
        if transient {
            let mut arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(a) = arena.as_mut() {
                if let Some(error) = a.retained.abort_on_free_error(error) {
                    failure(error);
                }
                return;
            }
        }
        failure(error);
    }
}

fn retention_operation(operation: impl FnOnce(&mut Arena) -> Result<()>) -> i32 {
    let mut guard = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let Some(a) = guard.as_mut() else {
        failure(invalid("native package allocator is uninitialized"));
        return -1;
    };
    match a.check_healthy().and_then(|()| operation(a)) {
        Ok(()) => 0,
        Err(error) => {
            failure(invalid(format!(
                "{error}; retained reference bytes={}, handles={}",
                a.retained.reference_bytes,
                a.retained.handles.len()
            )));
            -1
        }
    }
}

fn release_free_range(free: &mut Vec<(u64, usize)>, address: u64, size: usize) {
    free.push((address, size));
    free.sort_unstable();
    let mut merged: Vec<(u64, usize)> = Vec::new();
    for (address, size) in free.drain(..) {
        if let Some(previous) = merged.last_mut().filter(|p| p.0 + p.1 as u64 == address) {
            previous.1 += size;
        } else {
            merged.push((address, size));
        }
    }
    *free = merged;
}

fn finish_retention(
    retained: &mut RetainedBacking,
    retained_mappings: &mut RetainedMappingPool,
    free: &mut Vec<(u64, usize)>,
    mut unmap: impl FnMut(u64, usize) -> Result<()>,
    mut release: impl FnMut(u64) -> Result<()>,
) -> Result<()> {
    retained.check_finishable()?;
    while let Some(mapping) = retained_mappings.pop() {
        let (address, size) = (mapping.address, mapping.size);
        if let Err(error) = unmap(address, size) {
            retained_mappings.insert(mapping);
            return Err(error);
        }
        release_free_range(free, address, size);
    }
    retained.finish(|handle| release(handle))
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_retention_begin() -> i32 {
    retention_operation(|a| a.retained.begin())
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_retention_finish() -> i32 {
    retention_operation(|a| {
        let driver = &a.driver;
        finish_retention(
            &mut a.retained,
            &mut a.retained_mappings,
            &mut a.free,
            |address, size| unsafe { driver.check("cuMemUnmap", (driver.unmap)(address, size)) },
            |handle| unsafe { driver.check("cuMemRelease", (driver.physical_release)(handle)) },
        )
    })
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_retention_abort() -> i32 {
    retention_operation(|a| {
        a.retained.abort();
        Ok(())
    })
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_snapshot() -> *mut c_char {
    let arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let Some(a) = arena.as_ref() else {
        return null_mut();
    };
    if let Err(error) = a.check_healthy() {
        failure(error);
        return null_mut();
    }
    let value = serde_json::json!({
        "granularity": a.granularity, "mappings": a.mappings, "host_base": a.host_base,
        "retained_reference_bytes": a.retained.reference_bytes,
        "retained_handles": a.retained.handles.len(),
        "retained_reusable_handles": a.retained_mappings.len(),
        "retained_reusable_bytes": a.retained_mappings.bytes(),
    });
    CString::new(value.to_string()).unwrap().into_raw()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn flare_package_string_free(pointer: *mut c_char) {
    if !pointer.is_null() {
        unsafe {
            drop(CString::from_raw(pointer));
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_error() -> *const c_char {
    ERROR
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(std::ptr::null(), |s| s.as_ptr())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    // Tests that install the global arena must not interleave with each other.
    static ARENA_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn failure_preserves_first_fatal_allocator_error() {
        let _guard = ARENA_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *ERROR.lock().unwrap_or_else(|error| error.into_inner()) = None;
        failure("first allocator failure");
        failure("secondary allocator failure");
        let recorded = ERROR
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
            .unwrap();
        assert_eq!(recorded.to_str().unwrap(), "first allocator failure");
    }

    #[test]
    fn secondary_aborted_free_preserves_first_driver_error() {
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.retain(4096, || Ok(7)).unwrap();
        let mut published = retained
            .abort_on_free_error(invalid("cuMemUnmap returned CUDA error 715"))
            .unwrap()
            .to_string();
        let first = published.clone();
        let secondary = retained.defer_free().unwrap_err();
        if let Some(error) = retained.abort_on_free_error(secondary) {
            published = error.to_string();
        }
        assert_eq!(published, first);
        assert!(published.contains("cuMemUnmap returned CUDA error 715"));
        assert!(published.contains("retained reference bytes=4096, handles=1"));
        assert_eq!(retained.handles, vec![(7, 4096)]);
    }

    #[test]
    fn physical_references_preserve_duplicates_and_release_once() {
        let mut retained = RetainedBacking::default();
        assert!(retained.retain(4, || Ok(7)).is_err());
        retained.begin().unwrap();
        assert!(retained.begin().is_err());
        retained.retain(4, || Ok(7)).unwrap();
        retained.retain(8, || Ok(7)).unwrap();
        assert_eq!(retained.reference_bytes, 12);
        let mut released = Vec::new();
        retained
            .finish(|handle| {
                released.push(handle);
                Ok(())
            })
            .unwrap();
        assert_eq!(released, vec![7, 7]);
        assert_eq!(retained.reference_bytes, 0);
        assert!(retained.handles.is_empty());
        assert!(retained.retain(4, || Ok(8)).is_err());
        assert!(!retained.defer_free().unwrap());
    }

    #[test]
    fn physical_reference_errors_preserve_accounting_and_abort_never_releases() {
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        assert!(
            retained
                .retain(4, || Err(invalid("retain failed")))
                .is_err()
        );
        assert_eq!(retained.reference_bytes, 0);
        retained.retain(4, || Ok(7)).unwrap();
        assert!(retained.finish(|_| Err(invalid("release failed"))).is_err());
        assert_eq!(retained.handles, vec![(7, 4)]);
        assert_eq!(retained.reference_bytes, 4);
        retained.abort();
        assert!(retained.defer_free().is_err());
        assert!(
            retained
                .finish(|_| panic!("must not call CUDA after abort"))
                .is_err()
        );
        assert!(retained.begin().is_err());
        assert_eq!(retained.handles, vec![(7, 4)]);
    }

    #[test]
    fn retained_byte_overflow_precedes_driver_call() {
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.reference_bytes = usize::MAX;
        assert!(
            retained
                .retain(1, || panic!("overflow must precede retain"))
                .is_err()
        );
    }

    #[test]
    fn retention_finish_unmaps_every_pooled_mapping_once() {
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.retain(0x1000, || Ok(7)).unwrap();
        retained.retain(0x1000, || Ok(8)).unwrap();
        let mut retained_mappings = RetainedMappingPool::default();
        retained_mappings.insert(RetainedMapping {
            address: 0x10000,
            size: 0x1000,
        });
        retained_mappings.insert(RetainedMapping {
            address: 0x20000,
            size: 0x1000,
        });
        let mut free = Vec::new();
        let mut unmapped = Vec::new();

        finish_retention(
            &mut retained,
            &mut retained_mappings,
            &mut free,
            |address, size| {
                unmapped.push((address, size));
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();

        unmapped.sort_unstable();
        assert_eq!(unmapped, vec![(0x10000, 0x1000), (0x20000, 0x1000)]);
        assert_eq!(retained_mappings.len(), 0);
        assert_eq!(free, vec![(0x10000, 0x1000), (0x20000, 0x1000)]);
        assert!(retained.handles.is_empty());
    }

    #[test]
    fn retention_finish_drains_each_pool_shape_once() {
        let cases: Vec<(&str, Vec<RetainedMapping>, Option<usize>, Vec<(u64, usize)>)> = vec![
            ("empty pool", Vec::new(), None, Vec::new()),
            (
                "one bucket with one mapping",
                vec![RetainedMapping {
                    address: 0x10000,
                    size: 0x1000,
                }],
                None,
                vec![(0x10000, 0x1000)],
            ),
            (
                "three buckets with one mapping each",
                vec![
                    RetainedMapping {
                        address: 0x10000,
                        size: 0x1000,
                    },
                    RetainedMapping {
                        address: 0x30000,
                        size: 0x2000,
                    },
                    RetainedMapping {
                        address: 0x50000,
                        size: 0x3000,
                    },
                ],
                None,
                vec![(0x10000, 0x1000), (0x30000, 0x2000), (0x50000, 0x3000)],
            ),
            (
                "two buckets with several mappings each",
                vec![
                    RetainedMapping {
                        address: 0x10000,
                        size: 0x1000,
                    },
                    RetainedMapping {
                        address: 0x30000,
                        size: 0x1000,
                    },
                    RetainedMapping {
                        address: 0x50000,
                        size: 0x2000,
                    },
                    RetainedMapping {
                        address: 0x70000,
                        size: 0x2000,
                    },
                    RetainedMapping {
                        address: 0x90000,
                        size: 0x1000,
                    },
                    RetainedMapping {
                        address: 0xb0000,
                        size: 0x2000,
                    },
                ],
                None,
                vec![
                    (0x90000, 0x1000),
                    (0x30000, 0x1000),
                    (0x10000, 0x1000),
                    (0xb0000, 0x2000),
                    (0x70000, 0x2000),
                    (0x50000, 0x2000),
                ],
            ),
            (
                "pool after take emptied a bucket",
                vec![
                    RetainedMapping {
                        address: 0x10000,
                        size: 0x1000,
                    },
                    RetainedMapping {
                        address: 0x30000,
                        size: 0x2000,
                    },
                    RetainedMapping {
                        address: 0x50000,
                        size: 0x2000,
                    },
                ],
                Some(0x1000),
                vec![(0x50000, 0x2000), (0x30000, 0x2000)],
            ),
        ];

        for (name, mappings, take_size, expected) in cases {
            let mut retained = RetainedBacking::default();
            retained.begin().unwrap();
            let mut retained_mappings = RetainedMappingPool::default();
            for mapping in mappings {
                retained_mappings.insert(mapping);
            }
            if let Some(size) = take_size {
                retained_mappings.take(size).unwrap();
            }
            let mut free = Vec::new();
            let mut unmapped = Vec::new();

            finish_retention(
                &mut retained,
                &mut retained_mappings,
                &mut free,
                |address, size| {
                    unmapped.push((address, size));
                    Ok(())
                },
                |_| Ok(()),
            )
            .unwrap();

            assert_eq!(unmapped, expected, "{name}");
            let mut expected_free = expected.clone();
            expected_free.sort_unstable();
            assert_eq!(free, expected_free, "{name}");
            assert_eq!(retained_mappings.len(), 0, "{name}");
            assert_eq!(retained_mappings.bytes(), 0, "{name}");
        }
    }

    #[test]
    fn retention_finish_keeps_mapping_when_unmap_fails() {
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.retain(4096, || Ok(7)).unwrap();
        let mut retained_mappings = RetainedMappingPool::default();
        retained_mappings.insert(RetainedMapping {
            address: 0x100,
            size: 4096,
        });
        let mut free = Vec::new();

        let result = finish_retention(
            &mut retained,
            &mut retained_mappings,
            &mut free,
            |_, _| Err(invalid("unmap failed")),
            |_| panic!("release must not run while unmap fails"),
        );

        assert!(result.is_err());
        assert_eq!(retained_mappings.len(), 1);
        assert_eq!(retained.handles, vec![(7, 4096)]);
        assert!(free.is_empty());
    }

    #[test]
    fn retention_finish_restores_pool_when_unmap_fails() {
        fn drain_order(pool: &mut RetainedMappingPool) -> Vec<(u64, usize)> {
            let mut order = Vec::new();
            while let Some(mapping) = pool.pop() {
                order.push((mapping.address, mapping.size));
            }
            order
        }

        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.retain(0x1000, || Ok(7)).unwrap();
        retained.retain(0x2000, || Ok(8)).unwrap();
        let mut pool = RetainedMappingPool::default();
        pool.insert(RetainedMapping {
            address: 0x10000,
            size: 0x1000,
        });
        pool.insert(RetainedMapping {
            address: 0x30000,
            size: 0x1000,
        });
        pool.insert(RetainedMapping {
            address: 0x50000,
            size: 0x2000,
        });
        let mut free = Vec::new();
        let mut calls = 0;

        let result = finish_retention(
            &mut retained,
            &mut pool,
            &mut free,
            |address, _| {
                calls += 1;
                if address == 0x10000 {
                    Err(invalid("unmap failed"))
                } else {
                    Ok(())
                }
            },
            |_| panic!("release must not run while unmap fails"),
        );

        assert!(result.is_err());
        assert_eq!(calls, 2);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.bytes(), 0x3000);
        assert_eq!(
            drain_order(&mut pool),
            vec![(0x10000, 0x1000), (0x50000, 0x2000)]
        );
        assert_eq!(free, vec![(0x30000, 0x1000)]);
        assert_eq!(retained.handles, vec![(7, 0x1000), (8, 0x2000)]);
        assert_eq!(retained.reference_bytes, 0x3000);

        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.retain(0x1000, || Ok(9)).unwrap();
        let mut pool = RetainedMappingPool::default();
        pool.insert(RetainedMapping {
            address: 0x10000,
            size: 0x1000,
        });
        pool.insert(RetainedMapping {
            address: 0x50000,
            size: 0x2000,
        });
        let mut free = Vec::new();
        let mut calls = 0;

        let result = finish_retention(
            &mut retained,
            &mut pool,
            &mut free,
            |_, _| {
                calls += 1;
                Err(invalid("unmap failed"))
            },
            |_| panic!("release must not run while unmap fails"),
        );

        assert!(result.is_err());
        assert_eq!(calls, 1);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.bytes(), 0x3000);
        assert_eq!(
            drain_order(&mut pool),
            vec![(0x10000, 0x1000), (0x50000, 0x2000)]
        );
        assert!(free.is_empty());
        assert_eq!(retained.handles, vec![(9, 0x1000)]);
        assert_eq!(retained.reference_bytes, 0x1000);
    }

    #[test]
    fn retained_mapping_pool_requires_exact_size() {
        let mut pool = RetainedMappingPool::default();
        pool.insert(RetainedMapping {
            address: 0x100,
            size: 4096,
        });
        pool.insert(RetainedMapping {
            address: 0x180,
            size: 4096,
        });
        pool.insert(RetainedMapping {
            address: 0x200,
            size: 8192,
        });
        pool.insert(RetainedMapping {
            address: 0x300,
            size: 16384,
        });
        assert_eq!(pool.len(), 4);
        assert_eq!(pool.bytes(), 32768);
        for size in [2048, 6144, 12288, 32768] {
            assert!(pool.take(size).is_none());
            assert_eq!(pool.len(), 4);
            assert_eq!(pool.bytes(), 32768);
        }
        let popped = pool.pop().unwrap();
        assert_eq!((popped.address, popped.size), (0x180, 4096));
        pool.insert(popped);
        let first = pool.take(4096).unwrap();
        assert_eq!((first.address, first.size), (0x180, 4096));
        assert_eq!(pool.len(), 3);
        assert_eq!(pool.bytes(), 28672);
        let second = pool.take(4096).unwrap();
        assert_eq!((second.address, second.size), (0x100, 4096));
        assert!(pool.take(4096).is_none());
        let larger = pool.take(8192).unwrap();
        assert_eq!((larger.address, larger.size), (0x200, 8192));
        let largest = pool.take(16384).unwrap();
        assert_eq!((largest.address, largest.size), (0x300, 16384));
        assert_eq!(pool.len(), 0);
        assert_eq!(pool.bytes(), 0);
        assert!(pool.take(4096).is_none());
    }

    unsafe extern "C" fn abort_test_unmap(_address: u64, _size: usize) -> i32 {
        panic!("driver unmap called after abort");
    }

    unsafe extern "C" fn abort_test_release(_handle: u64) -> i32 {
        panic!("driver release called after abort");
    }

    unsafe extern "C" fn failing_retain(_handle: *mut u64, _pointer: *mut c_void) -> i32 {
        715
    }

    #[test]
    fn transient_allocation_after_abort_preserves_first_driver_error() {
        let _guard = ARENA_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Ok(mut driver) = Driver::load() else {
            return;
        };
        driver.physical_retain = failing_retain;
        driver.unmap = abort_test_unmap;
        driver.physical_release = abort_test_release;
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.retain(4096, || Ok(7)).unwrap();
        let mut retained_mappings = RetainedMappingPool::default();
        retained_mappings.insert(RetainedMapping {
            address: 0x10000,
            size: 4096,
        });
        *ERROR.lock().unwrap_or_else(|error| error.into_inner()) = None;
        {
            let mut arena = ARENA.lock().unwrap_or_else(|error| error.into_inner());
            *arena = Some(Arena {
                driver,
                device: 0,
                granularity: 4096,
                total_memory: usize::MAX,
                gpudirect: false,
                host_base: 0,
                free: Vec::new(),
                mappings: BTreeMap::new(),
                retained,
                retained_mappings,
                abandoned: None,
            });
        }
        let pointer = flare_package_transient_allocate(4095, 0, null_mut());
        assert_eq!(pointer as u64, 0x10000);
        flare_package_transient_free(pointer, 4095, 0, null_mut());
        let first = ERROR
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
            .unwrap();
        assert!(
            first
                .to_str()
                .unwrap()
                .contains("cuMemRetainAllocationHandle")
        );
        assert!(first.to_str().unwrap().contains("715"));
        assert!(
            first
                .to_str()
                .unwrap()
                .contains("retained reference bytes=4096, handles=1")
        );

        assert!(flare_package_transient_allocate(4095, 0, null_mut()).is_null());
        let after_allocate = ERROR
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        assert_eq!(flare_package_retention_finish(), -1);
        let after_finish = ERROR
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        {
            let mut arena = ARENA.lock().unwrap_or_else(|error| error.into_inner());
            let allocator = arena.as_ref().unwrap();
            assert!(allocator.retained.aborted);
            assert_eq!(allocator.mappings.get(&0x10000), Some(&4096));
            assert!(allocator.retained_mappings.mappings.is_empty());
            assert_eq!(allocator.retained.handles, vec![(7, 4096)]);
            *arena = None;
        }
        assert_eq!(after_allocate.as_ref(), Some(&first));
        assert_eq!(after_finish.as_ref(), Some(&first));
    }

    #[test]
    fn retention_finish_after_abort_makes_no_driver_calls() {
        let _guard = ARENA_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Ok(mut driver) = Driver::load() else {
            return;
        };
        driver.unmap = abort_test_unmap;
        driver.physical_release = abort_test_release;
        let mut retained = RetainedBacking::default();
        retained.begin().unwrap();
        retained.retain(4096, || Ok(7)).unwrap();
        let mut retained_mappings = RetainedMappingPool::default();
        retained_mappings.insert(RetainedMapping {
            address: 0x100,
            size: 4096,
        });
        retained.abort();
        {
            let mut arena = ARENA.lock().unwrap_or_else(|error| error.into_inner());
            *arena = Some(Arena {
                driver,
                device: 0,
                granularity: 4096,
                total_memory: usize::MAX,
                gpudirect: false,
                host_base: 0,
                free: Vec::new(),
                mappings: BTreeMap::new(),
                retained,
                retained_mappings,
                abandoned: None,
            });
        }
        assert_eq!(flare_package_retention_finish(), -1);
        let mut arena = ARENA.lock().unwrap_or_else(|error| error.into_inner());
        let allocator = arena.as_mut().unwrap();
        assert_eq!(allocator.retained_mappings.len(), 1);
        assert!(allocator.free.is_empty());
        assert_eq!(allocator.retained.handles, vec![(7, 4096)]);
        *arena = None;
        *ERROR.lock().unwrap_or_else(|error| error.into_inner()) = None;
    }

    static FREE_TEST_UNMAP_CALLS: AtomicUsize = AtomicUsize::new(0);

    static PHYSICAL_CREATE_CALLS: AtomicUsize = AtomicUsize::new(0);

    static CREATOR_RELEASE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static ALLOCATION_MAP_CALLS: AtomicUsize = AtomicUsize::new(0);
    static ALLOCATION_ACCESS_CALLS: AtomicUsize = AtomicUsize::new(0);
    static ALLOCATION_MAP_STATUS: AtomicUsize = AtomicUsize::new(0);
    static CREATOR_RELEASE_STATUS: AtomicUsize = AtomicUsize::new(0);
    static ALLOCATION_ACCESS_STATUS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn allocation_test_create(
        handle: *mut u64,
        _size: usize,
        _properties: *const AllocationProperties,
        _flags: u64,
    ) -> i32 {
        PHYSICAL_CREATE_CALLS.fetch_add(1, Ordering::SeqCst);
        unsafe { *handle = 7 };
        0
    }

    unsafe extern "C" fn allocation_test_map(
        _address: u64,
        _size: usize,
        _offset: usize,
        _handle: u64,
        _flags: u64,
    ) -> i32 {
        ALLOCATION_MAP_CALLS.fetch_add(1, Ordering::SeqCst);
        ALLOCATION_MAP_STATUS.load(Ordering::SeqCst) as i32
    }

    unsafe extern "C" fn allocation_test_release(_handle: u64) -> i32 {
        CREATOR_RELEASE_CALLS.fetch_add(1, Ordering::SeqCst);
        CREATOR_RELEASE_STATUS.load(Ordering::SeqCst) as i32
    }

    unsafe extern "C" fn allocation_test_access(
        _address: u64,
        _size: usize,
        _access: *const Access,
        _count: usize,
    ) -> i32 {
        ALLOCATION_ACCESS_CALLS.fetch_add(1, Ordering::SeqCst);
        ALLOCATION_ACCESS_STATUS.load(Ordering::SeqCst) as i32
    }

    unsafe extern "C" fn allocation_test_unmap(_address: u64, _size: usize) -> i32 {
        FREE_TEST_UNMAP_CALLS.fetch_add(1, Ordering::SeqCst);
        0
    }

    fn allocation_test_arena(transient: bool) -> Arena {
        let mut driver =
            Driver::load().expect("allocator callback control requires the NVIDIA driver library");
        driver.physical_create = allocation_test_create;
        driver.map = allocation_test_map;
        driver.physical_release = allocation_test_release;
        driver.access = allocation_test_access;
        driver.unmap = allocation_test_unmap;
        for counter in [
            &PHYSICAL_CREATE_CALLS,
            &CREATOR_RELEASE_CALLS,
            &ALLOCATION_MAP_CALLS,
            &ALLOCATION_ACCESS_CALLS,
            &FREE_TEST_UNMAP_CALLS,
            &ALLOCATION_MAP_STATUS,
            &CREATOR_RELEASE_STATUS,
            &ALLOCATION_ACCESS_STATUS,
        ] {
            counter.store(0, Ordering::SeqCst);
        }
        *ERROR.lock().unwrap_or_else(|error| error.into_inner()) = None;
        let mut retained = RetainedBacking::default();
        if transient {
            retained.begin().unwrap();
        }
        Arena {
            driver,
            device: 0,
            granularity: 4096,
            total_memory: 8192,
            gpudirect: false,
            host_base: 0,
            free: vec![(0x10000, 8192)],
            mappings: BTreeMap::new(),
            retained,
            retained_mappings: RetainedMappingPool::default(),
            abandoned: None,
        }
    }

    #[test]
    fn allocation_rejects_failed_creator_handle_release() {
        let _guard = ARENA_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for transient in [false, true] {
            for code in [1, 715] {
                let allocator = allocation_test_arena(transient);
                CREATOR_RELEASE_STATUS.store(code, Ordering::SeqCst);
                *ARENA.lock().unwrap_or_else(|error| error.into_inner()) = Some(allocator);
                let pointer = if transient {
                    flare_package_transient_allocate(4096, 0, null_mut())
                } else {
                    flare_package_allocate(4096, 0, null_mut())
                };
                let error = ERROR
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take()
                    .map(|error| error.to_string_lossy().into_owned());
                *ARENA.lock().unwrap_or_else(|error| error.into_inner()) = None;
                assert!(
                    pointer.is_null(),
                    "published {pointer:p} after cuMemRelease failed"
                );
                assert_eq!(PHYSICAL_CREATE_CALLS.load(Ordering::SeqCst), 1);
                assert_eq!(CREATOR_RELEASE_CALLS.load(Ordering::SeqCst), 1);
                assert_eq!(ALLOCATION_ACCESS_CALLS.load(Ordering::SeqCst), 0);
                assert!(
                    error
                        .as_deref()
                        .is_some_and(|error| error.contains("cuMemRelease")
                            && error.contains(&code.to_string())),
                    "{error:?}"
                );
            }
        }
    }

    #[test]
    fn failed_allocation_preserves_resources_and_stops_driver_calls() {
        let _guard = ARENA_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for transient in [false, true] {
            for (operation, stage) in [("cuMemMap", 0), ("cuMemRelease", 1), ("cuMemSetAccess", 2)]
            {
                for code in [1, 715] {
                    let mut allocator = allocation_test_arena(transient);
                    allocator.mappings.insert(0x30000, 4096);
                    if transient {
                        allocator.retained.retain(4096, || Ok(9)).unwrap();
                        allocator.retained_mappings.insert(RetainedMapping {
                            address: 0x40000,
                            size: 8192,
                        });
                    }
                    [
                        &ALLOCATION_MAP_STATUS,
                        &CREATOR_RELEASE_STATUS,
                        &ALLOCATION_ACCESS_STATUS,
                    ][stage]
                        .store(code, Ordering::SeqCst);
                    *ARENA.lock().unwrap_or_else(|error| error.into_inner()) = Some(allocator);
                    let pointer = if transient {
                        flare_package_transient_allocate(4096, 0, null_mut())
                    } else {
                        flare_package_allocate(4096, 0, null_mut())
                    };
                    let first = ERROR
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .clone();
                    let retry = flare_package_allocate(4096, 0, null_mut());
                    let transient_retry = flare_package_transient_allocate(4096, 0, null_mut());
                    flare_package_free(0x30000 as *mut c_void, 4096, 0, null_mut());
                    flare_package_transient_free(0x30000 as *mut c_void, 4096, 0, null_mut());
                    let begin = flare_package_retention_begin();
                    let finish = flare_package_retention_finish();
                    let abort = flare_package_retention_abort();
                    let snapshot = flare_package_snapshot();
                    // Health rejection must occur before asking for a CUDA context.
                    let inventory_failed = memory_inventory().is_err();
                    let last = ERROR
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .take();
                    let allocator = ARENA
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .take()
                        .unwrap();
                    assert!(pointer.is_null() && retry.is_null() && transient_retry.is_null());
                    assert_eq!((begin, finish, abort), (-1, -1, -1));
                    assert!(snapshot.is_null() && inventory_failed);
                    assert_eq!(first, last);
                    let error = first.unwrap().into_string().unwrap();
                    assert!(
                        error.contains(operation) && error.contains(&code.to_string()),
                        "{error}"
                    );
                    assert_eq!(allocator.free, [(0x11000, 4096)]);
                    assert_eq!(allocator.mappings, BTreeMap::from([(0x30000, 4096)]));
                    let owner = allocator.abandoned.unwrap();
                    assert_eq!((owner.address, owner.size), (0x10000, 4096));
                    assert_eq!(
                        owner.creator_reference,
                        if stage == 2 { None } else { Some(7) }
                    );
                    assert_eq!(owner.mapping_confirmed, stage != 0);
                    assert_eq!(
                        allocator.retained.reference_bytes,
                        if transient { 4096 } else { 0 }
                    );
                    assert_eq!(
                        allocator.retained.handles,
                        if transient { vec![(9, 4096)] } else { vec![] }
                    );
                    assert_eq!(
                        allocator.retained_mappings.bytes(),
                        if transient { 8192 } else { 0 }
                    );
                    assert_eq!(PHYSICAL_CREATE_CALLS.load(Ordering::SeqCst), 1);
                    assert_eq!(ALLOCATION_MAP_CALLS.load(Ordering::SeqCst), 1);
                    assert_eq!(
                        CREATOR_RELEASE_CALLS.load(Ordering::SeqCst),
                        usize::from(stage > 0)
                    );
                    assert_eq!(
                        ALLOCATION_ACCESS_CALLS.load(Ordering::SeqCst),
                        usize::from(stage > 1)
                    );
                    assert_eq!(FREE_TEST_UNMAP_CALLS.load(Ordering::SeqCst), 0);
                }
            }
        }
    }

    #[test]
    fn successful_allocation_releases_creator_once_and_publishes_mapping() {
        let _guard = ARENA_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for transient in [false, true] {
            let allocator = allocation_test_arena(transient);
            *ARENA.lock().unwrap_or_else(|error| error.into_inner()) = Some(allocator);
            let pointer = if transient {
                flare_package_transient_allocate(4096, 0, null_mut())
            } else {
                flare_package_allocate(4096, 0, null_mut())
            };
            // Use the ordinary callback so no artificial retained handle is needed.
            flare_package_free(pointer, 4096, 0, null_mut());
            let allocator = ARENA
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take()
                .unwrap();
            let error = ERROR
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take();
            assert_eq!(pointer as u64, 0x10000);
            assert!(error.is_none());
            assert!(allocator.abandoned.is_none() && allocator.mappings.is_empty());
            assert_eq!(allocator.free, [(0x10000, 8192)]);
            assert_eq!(PHYSICAL_CREATE_CALLS.load(Ordering::SeqCst), 1);
            assert_eq!(ALLOCATION_MAP_CALLS.load(Ordering::SeqCst), 1);
            assert_eq!(CREATOR_RELEASE_CALLS.load(Ordering::SeqCst), 1);
            assert_eq!(ALLOCATION_ACCESS_CALLS.load(Ordering::SeqCst), 1);
            assert_eq!(FREE_TEST_UNMAP_CALLS.load(Ordering::SeqCst), 1);
        }
    }

    unsafe extern "C" fn oom_create(
        _handle: *mut u64,
        _size: usize,
        _properties: *const AllocationProperties,
        _flags: u64,
    ) -> i32 {
        PHYSICAL_CREATE_CALLS.fetch_add(1, Ordering::SeqCst);
        2
    }

    #[test]
    fn create_oom_can_retry_without_poisoning_or_losing_address_space() {
        let _guard = ARENA_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for transient in [false, true] {
            let mut allocator = allocation_test_arena(transient);
            allocator.driver.physical_create = oom_create;
            *ARENA.lock().unwrap_or_else(|error| error.into_inner()) = Some(allocator);
            let first = if transient {
                flare_package_transient_allocate(8192, 0, null_mut())
            } else {
                flare_package_allocate(8192, 0, null_mut())
            };
            {
                let mut arena = ARENA.lock().unwrap_or_else(|error| error.into_inner());
                let allocator = arena.as_mut().unwrap();
                assert!(allocator.abandoned.is_none() && allocator.mappings.is_empty());
                assert_eq!(allocator.free, [(0x10000, 8192)]);
                allocator.driver.physical_create = allocation_test_create;
            }
            let retry = if transient {
                flare_package_transient_allocate(8192, 0, null_mut())
            } else {
                flare_package_allocate(8192, 0, null_mut())
            };
            flare_package_free(retry, 8192, 0, null_mut());
            let allocator = ARENA
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take()
                .unwrap();
            let error = ERROR
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take();
            assert!(first.is_null());
            assert_eq!(retry as u64, 0x10000);
            assert!(error.is_none());
            assert!(allocator.abandoned.is_none() && allocator.mappings.is_empty());
            assert_eq!(allocator.free, [(0x10000, 8192)]);
            assert_eq!(PHYSICAL_CREATE_CALLS.load(Ordering::SeqCst), 2);
            assert_eq!(ALLOCATION_MAP_CALLS.load(Ordering::SeqCst), 1);
            assert_eq!(CREATOR_RELEASE_CALLS.load(Ordering::SeqCst), 1);
            assert_eq!(ALLOCATION_ACCESS_CALLS.load(Ordering::SeqCst), 1);
            assert_eq!(FREE_TEST_UNMAP_CALLS.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn impossible_allocation_is_retryable_without_entering_driver() {
        let _guard = ARENA_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut driver =
            Driver::load().expect("allocator callback control requires the NVIDIA driver library");
        driver.physical_create = oom_create;
        PHYSICAL_CREATE_CALLS.store(0, Ordering::SeqCst);
        *ERROR.lock().unwrap_or_else(|error| error.into_inner()) = None;
        *ARENA.lock().unwrap_or_else(|error| error.into_inner()) = Some(Arena {
            driver,
            device: 0,
            granularity: 4096,
            total_memory: 8192,
            gpudirect: false,
            host_base: 0,
            free: vec![(0x10000, 32768)],
            mappings: BTreeMap::new(),
            retained: RetainedBacking::default(),
            retained_mappings: RetainedMappingPool::default(),
            abandoned: None,
        });
        // Rounding the first byte above the limit must not enter cuMemCreate.
        assert!(flare_package_allocate(8193, 0, null_mut()).is_null());
        assert_eq!(PHYSICAL_CREATE_CALLS.load(Ordering::SeqCst), 0);
        assert!(
            ERROR
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_none()
        );
        // A within-limit retry still reaches the driver and preserves ordinary
        // driver OOM as retryable, rather than poisoning the package allocator.
        assert!(flare_package_allocate(8192, 0, null_mut()).is_null());
        assert_eq!(PHYSICAL_CREATE_CALLS.load(Ordering::SeqCst), 1);
        assert!(
            ERROR
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_none()
        );
        let mut arena = ARENA.lock().unwrap_or_else(|error| error.into_inner());
        let allocator = arena.as_ref().unwrap();
        assert_eq!(allocator.free, vec![(0x10000, 32768)]);
        assert!(allocator.mappings.is_empty());
        *arena = None;
    }

    unsafe extern "C" fn failing_once_unmap(_address: u64, _size: usize) -> i32 {
        if FREE_TEST_UNMAP_CALLS.fetch_add(1, Ordering::SeqCst) == 0 {
            1
        } else {
            0
        }
    }

    #[test]
    fn package_free_keeps_mapping_when_unmap_fails() {
        let _guard = ARENA_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Ok(mut driver) = Driver::load() else {
            return;
        };
        driver.unmap = failing_once_unmap;
        {
            let mut arena = ARENA.lock().unwrap_or_else(|error| error.into_inner());
            *arena = Some(Arena {
                driver,
                device: 0,
                granularity: 4096,
                total_memory: usize::MAX,
                gpudirect: false,
                host_base: 0,
                free: Vec::new(),
                mappings: BTreeMap::from([(0x10000, 4096)]),
                retained: RetainedBacking::default(),
                retained_mappings: RetainedMappingPool::default(),
                abandoned: None,
            });
        }
        let pointer = 0x10000 as *mut c_void;

        flare_package_free(pointer, 4096, 0, null_mut());
        let error = ERROR
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .map(|error| error.to_string_lossy().into_owned());
        assert!(
            error
                .as_deref()
                .is_some_and(|error| error.contains("cuMemUnmap")),
            "{error:?}"
        );
        {
            let mut arena = ARENA.lock().unwrap_or_else(|error| error.into_inner());
            let allocator = arena.as_mut().unwrap();
            assert_eq!(allocator.mappings.get(&0x10000), Some(&4096));
            assert!(allocator.free.is_empty());
        }

        flare_package_free(pointer, 4096, 0, null_mut());
        {
            let mut arena = ARENA.lock().unwrap_or_else(|error| error.into_inner());
            let allocator = arena.as_mut().unwrap();
            assert!(allocator.mappings.is_empty());
            assert_eq!(allocator.free, vec![(0x10000, 4096)]);
            *arena = None;
        }
        assert_eq!(FREE_TEST_UNMAP_CALLS.load(Ordering::SeqCst), 2);
        *ERROR.lock().unwrap_or_else(|error| error.into_inner()) = None;
    }

    #[test]
    fn small_linux_exports_keep_every_host_range_within_windows_addresses() {
        let device_base = 8 << 40;
        for world in 1..=4 {
            let host = host_base(device_base, world);
            assert_eq!(host, 12 << 40);
            assert!(device_base + u64::from(world) * EXTENT as u64 <= host);
            assert!(host + u64::from(world) * EXTENT as u64 <= 16 << 40);
        }
    }

    #[test]
    fn larger_linux_exports_keep_host_ranges_clear_of_all_device_ranks() {
        let device_base = 8 << 40;
        for world in 5..=24 {
            let host = host_base(device_base, world);
            assert_eq!(host, 32 << 40);
            assert!(device_base + u64::from(world) * EXTENT as u64 <= host);
        }
    }
}
