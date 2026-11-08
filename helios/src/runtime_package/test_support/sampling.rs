/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Inspect the shared sampler's actual graph operands for native qualification.
//! No manifest views, graph arguments, or serving behavior are changed.

use super::super::{
    AuthorizedPackage, NativeModel, format::{Memory, Package, invalid}, payload::Files,
    protection::AuthorizedKind, protobuf::{self, wire},
};
use crate::Result;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceView {
    pub allocation: String,
    pub offset: usize,
    pub nbytes: usize,
    base: u64,
}

#[derive(Clone, Debug)]
pub struct SamplingViews {
    pub logits: DeviceView,
    pub slots: DeviceView,
    pub epochs: DeviceView,
    pub live: DeviceView,
    pub rows: usize,
    pub vocabulary: usize,
    pub node: usize,
}

fn extent(package: &Package, name: &str, offset: usize, nbytes: usize) -> Result<DeviceView> {
    let allocation = package.allocations.iter().find(|a| a.name == name)
        .ok_or_else(|| invalid("sampler operand names an absent allocation"))?;
    let base = allocation.address.ok_or_else(|| invalid("sampler inspection requires fixed addresses"))?;
    if !matches!(allocation.memory, Memory::Device) || allocation.sizing.is_some()
        || nbytes == 0 || offset.checked_add(nbytes).is_none_or(|end| end > allocation.nbytes)
        || base.checked_add(allocation.nbytes as u64).is_none() {
        return Err(invalid("sampler operand must fit fixed device storage"));
    }
    Ok(DeviceView { allocation: name.into(), offset, nbytes, base })
}

fn pointer(package: &Package, argument: &wire::Blob, nbytes: usize) -> Result<DeviceView> {
    if argument.data.len() != 8 {
        return Err(invalid("sampler pointer argument must contain eight bytes"));
    }
    if !argument.relocations.is_empty() {
        let [relocation] = argument.relocations.as_slice() else {
            return Err(invalid("sampler pointer has multiple relocations"));
        };
        if relocation.offset != 0 || relocation.kind != wire::RelocationKind::Pointer as i32 {
            return Err(invalid("sampler pointer relocation has an incompatible layout"));
        }
        let target = relocation.target.as_ref().ok_or_else(|| invalid("sampler pointer has no target"))?;
        return extent(package, &target.allocation,
            usize::try_from(target.offset).map_err(|_| invalid("sampler offset overflow"))?, nbytes);
    }
    let address = u64::from_le_bytes(argument.data.as_slice().try_into().unwrap());
    let mut matches = package.allocations.iter().filter_map(|a| {
        let offset = usize::try_from(address.checked_sub(a.address?)?).ok()?;
        extent(package, &a.name, offset, nbytes).ok()
    });
    let view = matches.next().ok_or_else(|| invalid("sampler address is outside fixed device storage"))?;
    if matches.next().is_some() {
        return Err(invalid("sampler address has ambiguous storage"));
    }
    Ok(view)
}

fn locate(package: &Package, files: &dyn Files, graph_name: &str, vocabulary: usize,
    names: [&str; 3]) -> Result<SamplingViews> {
    let execution = package.execution.as_ref().ok_or_else(|| invalid("sampler inspection needs execution metadata"))?;
    // This diagnostic recognizes one measured shared-sampler ABI, without
    // imposing a new loading gate on packages with another execution layout.
    if execution.abi_version != 6 || vocabulary == 0 {
        return Err(invalid("sampler inspection requires the qualified ABI6 geometry"));
    }
    let descriptor = execution.graphs.iter().find(|g| g.name == graph_name)
        .ok_or_else(|| invalid("sampler graph is absent from execution metadata"))?;
    if !matches!(descriptor.kind.as_str(), "prefill" | "decode" | "resume") || descriptor.sampling_mode != "general" {
        return Err(invalid("sampler inspection requires ordinary prefill, decode or resume"));
    }
    let rows = usize::try_from(descriptor.batch_capacity).map_err(|_| invalid("sampler rows overflow"))?;
    if !(1..=64).contains(&rows) { return Err(invalid("sampler row capacity is invalid")); }
    let logits_bytes = rows.checked_mul(vocabulary).and_then(|n| n.checked_mul(2))
        .ok_or_else(|| invalid("BF16 sampler extent overflow"))?;
    let graph = package.graphs.iter().find(|g| g.name == graph_name)
        .ok_or_else(|| invalid("sampler graph body is absent"))?;
    let mut records = files.records(package, &graph.resource)?;
    let mut found = None;
    for node in 0..graph.node_count {
        let value = records.read::<wire::Node>()?;
        let Some(wire::node::Operation::Kernel(kernel)) = value.operation else { continue; };
        let function = package.functions.iter().find(|f| f.id as u64 == kernel.function)
            .ok_or_else(|| invalid("graph kernel names an absent function"))?;
        if function.symbol != "_scores" { continue; }
        if found.is_some() || function.parameter_sizes != vec![8; 13]
            || kernel.arguments.len() != 13 || kernel.arguments.iter().any(|a| a.data.len() != 8)
            || kernel.grid.first().copied() != Some(rows as u32) {
            return Err(invalid("sampler node is ambiguous or has an unqualified ABI"));
        }
        // Slots/Epochs/Live tie this kernel to the signed transport inputs;
        // its name alone is not sufficient to identify a logits allocation.
        let mut anchors = Vec::new();
        for ((argument, name), (dtype, count, width)) in [2usize, 3, 4].into_iter().zip(names)
            .zip([("int32", rows, 4), ("int64", rows, 8), ("int32", 1, 4)]) {
            let buffer = execution.buffers.iter().find(|b| b.name == name)
                .ok_or_else(|| invalid("sampler transport anchor is absent"))?;
            let reference = buffer.reference.as_ref().ok_or_else(|| invalid("sampler anchor has no reference"))?;
            let nbytes = count * width;
            if buffer.dtype != dtype || buffer.shape.len() != 1 || buffer.shape[0] < count as u64
                || buffer.strides != [1] || buffer.shape[0].checked_mul(width as u64) != Some(buffer.nbytes)
                || !descriptor.input_bindings.iter().any(|b| b.buffer_name == name) {
                return Err(invalid("sampler transport anchor has an incompatible layout"));
            }
            let actual = pointer(package, &kernel.arguments[argument], nbytes)?;
            let expected = extent(package, &reference.allocation,
                usize::try_from(reference.offset).map_err(|_| invalid("sampler anchor offset overflow"))?, nbytes)?;
            if actual != expected { return Err(invalid("sampler pointer differs from its transport anchor")); }
            anchors.push(actual);
        }
        // The two extra ABI slots in the qualified Triton artifact are zero;
        // they are not inferred model parameters or executable metadata.
        if kernel.arguments[11..].iter().any(|a| a.data != [0u8; 8] || !a.relocations.is_empty()) {
            return Err(invalid("sampler trailing ABI slots differ from the qualified artifact"));
        }
        found = Some(SamplingViews { logits: pointer(package, &kernel.arguments[0], logits_bytes)?,
            slots: anchors.remove(0), epochs: anchors.remove(0), live: anchors.remove(0),
            rows, vocabulary, node });
    }
    for _ in 0..graph.edge_count { records.read::<wire::Edge>()?; }
    records.finish()?;
    found.ok_or_else(|| invalid("graph contains no qualified sampler node"))
}

/// Read through the already authorized package's authenticated graph reader.
/// BF16 is a qualification precondition, checked against the pinned recipe and
/// reference tensors by the caller; anonymous graph storage has no dtype label.
pub fn views(package: &AuthorizedPackage, rank: usize, graph: &str, vocabulary: usize,
    names: [&str; 3]) -> Result<SamplingViews> {
    match &package.kind {
        AuthorizedKind::Payload(payload) => locate(payload.rank_packages().get(rank)
            .ok_or_else(|| invalid("sampler rank is absent"))?, payload.source(), graph, vocabulary, names),
        AuthorizedKind::Legacy { archive, index } => {
            let rank = protobuf::decode(index.ranks.get(rank)
                .ok_or_else(|| invalid("sampler rank is absent"))?.clone())?;
            locate(&rank, archive, graph, vocabulary, names)
        }
    }
}

/// Enumerate authenticated sampler graph names without assuming a preparation catalog.
pub fn graph_names(package: &AuthorizedPackage, rank: usize) -> Result<Vec<String>> {
    fn names(package: &Package) -> Result<Vec<String>> {
        let execution = package.execution.as_ref().ok_or_else(|| invalid("sampler execution metadata is absent"))?;
        Ok(execution.graphs.iter().filter(|graph| matches!(graph.kind.as_str(), "prefill" | "decode" | "resume"))
            .map(|graph| graph.name.clone()).collect())
    }
    match &package.kind {
        AuthorizedKind::Payload(payload) => names(payload.rank_packages().get(rank)
            .ok_or_else(|| invalid("sampler rank is absent"))?),
        AuthorizedKind::Legacy { index, .. } => names(&protobuf::decode(index.ranks.get(rank)
            .ok_or_else(|| invalid("sampler rank is absent"))?.clone())?),
    }
}

/// Resolve an existing signed execution buffer for a quiescent diagnostic read.
pub fn named_buffer(package: &AuthorizedPackage, rank: usize, name: &str) -> Result<DeviceView> {
    fn resolve(package: &Package, name: &str) -> Result<DeviceView> {
        let buffer = package.execution.as_ref().and_then(|e| e.buffers.iter().find(|b| b.name == name))
            .ok_or_else(|| invalid("diagnostic execution buffer is absent"))?;
        let reference = buffer.reference.as_ref().ok_or_else(|| invalid("diagnostic buffer has no reference"))?;
        extent(package, &reference.allocation,
            usize::try_from(reference.offset).map_err(|_| invalid("diagnostic offset overflow"))?,
            usize::try_from(buffer.nbytes).map_err(|_| invalid("diagnostic extent overflow"))?)
    }
    match &package.kind {
        AuthorizedKind::Payload(payload) => resolve(payload.rank_packages().get(rank)
            .ok_or_else(|| invalid("diagnostic rank is absent"))?, name),
        AuthorizedKind::Legacy { index, .. } => resolve(&protobuf::decode(index.ranks.get(rank)
            .ok_or_else(|| invalid("diagnostic rank is absent"))?.clone())?, name),
    }
}

/// Download only after the executor reports zero outstanding work, before the
/// next submission. Context synchronization cannot protect an overwritten
/// result if a second frame was submitted before this observer was called.
pub fn download(model: &NativeModel, rank: usize, view: &DeviceView) -> Result<Vec<u8>> {
    let resources = model.ranks.get(rank).ok_or_else(|| invalid("sampler local rank is absent"))?;
    let allocation = resources.allocations.get(&view.allocation)
        .ok_or_else(|| invalid("sampler runtime allocation is absent"))?;
    if !allocation.fixed || !matches!(allocation.memory, Memory::Device)
        || allocation.address != view.base
        || view.offset.checked_add(view.nbytes).is_none_or(|end| end > allocation.nbytes) {
        return Err(invalid("sampler runtime storage differs from its authenticated view"));
    }
    let bytes = resources.download(&view.allocation)?;
    Ok(bytes[view.offset..view.offset + view.nbytes].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::{Publisher, VerifiedPackage, format::{Allocation, Context}};
    use std::path::Path;

    fn package() -> Package {
        Package { resources: Default::default(), architecture: [12, 1],
            allocations: vec![Allocation { name: "storage".into(), nbytes: 64, alignment: 8,
                file: None, data_spans: vec![], sources: vec![], relocations: vec![],
                address: Some(4096), memory: Memory::Device, sizing: None }],
            modules: vec![], functions: vec![], graphs: vec![], reservations: vec![],
            execution: None, context: Context::default(), events: vec![] }
    }

    fn raw(address: u64) -> wire::Blob {
        wire::Blob { data: address.to_le_bytes().to_vec(), relocations: vec![] }
    }

    fn relocated(offset: u64) -> wire::Blob {
        wire::Blob { data: vec![0; 8], relocations: vec![wire::Relocation { offset: 0,
            target: Some(wire::Reference { allocation: "storage".into(), offset }),
            kind: wire::RelocationKind::Pointer as i32 }] }
    }

    #[test]
    fn raw_and_relocated_pointers_resolve_the_same_complete_span() {
        let p = package();
        assert_eq!(pointer(&p, &raw(4104), 56).unwrap(), pointer(&p, &relocated(8), 56).unwrap());
        let v = pointer(&p, &raw(4096), 64).unwrap();
        assert_eq!((v.allocation.as_str(), v.offset, v.nbytes), ("storage", 0, 64));
        for (address, nbytes) in [(4095, 1), (4096, 65), (4160, 1), (4104, usize::MAX), (4096, 0)] {
            assert!(pointer(&p, &raw(address), nbytes).is_err(), "{address}/{nbytes}");
        }
        assert!(pointer(&p, &relocated(u64::MAX), 1).is_err());
    }

    #[test]
    fn inspection_rejects_nonfixed_dynamic_and_nondevice_storage() {
        let mut p = package();
        p.allocations[0].address = None;
        assert!(pointer(&p, &relocated(0), 8).is_err());
        p.allocations[0].address = Some(4096);
        p.allocations[0].sizing = Some(wire::Sizing::default());
        assert!(pointer(&p, &raw(4096), 8).is_err());
        p.allocations[0].sizing = None;
        for memory in [Memory::MappedHost, Memory::Peer] {
            p.allocations[0].memory = memory;
            assert!(pointer(&p, &raw(4096), 8).is_err());
            assert!(pointer(&p, &relocated(0), 8).is_err());
        }
        p.allocations[0].memory = Memory::Device;
        p.allocations[0].address = Some(u64::MAX - 31);
        assert!(pointer(&p, &raw(u64::MAX - 15), 8).is_err());
    }

    #[test]
    fn inspection_rejects_ambiguous_addresses_and_malformed_relocations() {
        let mut p = package();
        let mut second = package().allocations.remove(0);
        second.name = "alias".into();
        p.allocations.push(second);
        assert!(pointer(&p, &raw(4096), 8).unwrap_err().to_string().contains("ambiguous"));
        p.allocations.pop();
        let mut a = relocated(0);
        a.relocations[0].offset = 1;
        assert!(pointer(&p, &a, 8).is_err());
        a.relocations[0].offset = 0;
        a.relocations[0].kind = wire::RelocationKind::TensorMap as i32;
        assert!(pointer(&p, &a, 8).is_err());
        a.relocations[0].kind = wire::RelocationKind::Pointer as i32;
        a.relocations[0].target = None;
        assert!(pointer(&p, &a, 8).is_err());
        a = relocated(0);
        a.relocations.push(a.relocations[0].clone());
        assert!(pointer(&p, &a, 8).is_err());
        a = raw(4096); a.data.pop();
        assert!(pointer(&p, &a, 8).is_err());
    }

    #[test]
    #[ignore = "CPU package inspection; requires FLARE_PROMPT_TEST_PACKAGE and FLARE_PROMPT_TEST_PUBLISHERS"]
    fn actual_authorized_package_exposes_anchored_bf16_sampler_views() {
        let path = std::env::var("FLARE_PROMPT_TEST_PACKAGE").unwrap();
        let publishers = std::env::var("FLARE_PROMPT_TEST_PUBLISHERS").unwrap();
        let publishers: Vec<Publisher> = serde_json::from_slice(&std::fs::read(publishers).unwrap()).unwrap();
        let verified = VerifiedPackage::open(Path::new(&path), &publishers, None).unwrap();
        let namespace = verified.publisher().namespace.clone();
        let package = verified.authorize(&namespace, false).unwrap();
        let catalog = graph_names(&package, 0).unwrap();
        assert!(catalog.iter().any(|name| name.starts_with("resume")));
        for graph in catalog {
            let view = views(&package, 0, &graph, 151936,
                ["row_slot_ids", "row_generations", "live_rows"]).unwrap();
            assert_eq!(view.logits.nbytes, view.rows * 151936 * 2);
        }
        for (graph, rows) in [("prefill64", 1), ("decode1", 1), ("decode2", 2), ("decode4", 4), ("decode8", 8)] {
            let view = views(&package, 0, graph, 151936,
                ["row_slot_ids", "row_generations", "live_rows"]).unwrap();
            assert_eq!(view.rows, rows);
            assert_eq!(view.logits.nbytes, rows * 151936 * 2);
            println!("{graph}: {view:?}");
        }
        assert!(views(&package, 0, "audio100", 151936,
            ["row_slot_ids", "row_generations", "live_rows"]).is_err());
        assert!(views(&package, 0, "decode8", 151936,
            ["row_generations", "row_slot_ids", "live_rows"]).is_err());
        assert!(views(&package, 0, "decode8", usize::MAX,
            ["row_slot_ids", "row_generations", "live_rows"]).is_err());
    }
}

// Requires a quiescent executor and request-owned
// pages. Resolves the actual signed byte-plane layout before any device read.
#[derive(Clone, Debug)]
pub struct KvHistoryView {
    pub buffer: String,
    allocation: String,
    base: u64,
    maximum: u64,
    page_bytes: usize,
    page_tokens: usize,
}

pub fn kv_history_view(package: &AuthorizedPackage, rank: usize,
    region: &crate::model::contract::HistoryRegion, page_tokens: usize) -> Result<KvHistoryView> {
    fn resolve(package: &Package, region: &crate::model::contract::HistoryRegion,
        page_tokens: usize) -> Result<KvHistoryView> {
        let buffer = package.execution.as_ref().and_then(|e| e.buffers.iter()
            .find(|b| b.name == region.storage.buffer))
            .ok_or_else(|| invalid("KV diagnostic signed buffer is absent"))?;
        let reference = buffer.reference.as_ref().ok_or_else(|| invalid("KV buffer lacks reference"))?;
        let allocation = package.allocations.iter().find(|a| a.name == reference.allocation)
            .ok_or_else(|| invalid("KV buffer allocation is absent"))?;
        let sizing = allocation.sizing.as_ref().ok_or_else(|| invalid("KV allocation is not sized"))?;
        let base = allocation.address.ok_or_else(|| invalid("KV allocation is not fixed"))?;
        let page_bytes = usize::try_from(sizing.unit_bytes).map_err(|_| invalid("KV stride overflow"))?;
        let extent = sizing.unit_bytes.checked_mul(sizing.maximum)
            .ok_or_else(|| invalid("KV maximum extent overflow"))?;
        if !matches!(region.kind, crate::model::contract::HistoryKind::PagedKeyValue)
            || region.tokens_per_record != 1 || region.window_tokens != 0
            || region.storage.offset != 0 || region.storage.stride != page_bytes
            || region.storage.nbytes != page_bytes || sizing.capacity != wire::Capacity::KvPages as i32
            || sizing.maximum == 0 || page_tokens == 0 || !page_bytes.is_multiple_of(page_tokens)
            || !matches!(allocation.memory, Memory::Device) || reference.offset != 0
            || buffer.dtype != "uint8" || buffer.shape != [sizing.maximum, sizing.unit_bytes]
            || buffer.strides != [sizing.unit_bytes, 1] || buffer.nbytes != extent
            || base.checked_add(extent).is_none() {
            return Err(invalid("KV diagnostic requires an authenticated complete dense byte plane"));
        }
        Ok(KvHistoryView { buffer: buffer.name.clone(), allocation: allocation.name.clone(),
            base, maximum: sizing.maximum, page_bytes, page_tokens })
    }
    match &package.kind {
        AuthorizedKind::Payload(payload) => resolve(payload.rank_packages().get(rank)
            .ok_or_else(|| invalid("KV rank is absent"))?, region, page_tokens),
        AuthorizedKind::Legacy { index, .. } => resolve(&protobuf::decode(index.ranks.get(rank)
            .ok_or_else(|| invalid("KV rank is absent"))?.clone())?, region, page_tokens),
    }
}

pub fn download_kv_prefix(model: &NativeModel, rank: usize, view: &KvHistoryView,
    expected: &flare_selene::CacheGeometry, pages: &[u32], consumed: usize) -> Result<Vec<u8>> {
    let resources = model.ranks.get(rank).ok_or_else(|| invalid("KV local rank is absent"))?;
    let geometry = resources.cache_geometry().ok_or_else(|| invalid("KV geometry is absent"))?;
    let plane = geometry.planes.iter().find(|p| p.name == view.allocation)
        .ok_or_else(|| invalid("KV runtime plane is absent"))?;
    let allocation = resources.allocations.get(&view.allocation)
        .ok_or_else(|| invalid("KV runtime allocation is absent"))?;
    let needed = consumed.div_ceil(view.page_tokens);
    if geometry != expected || consumed == 0 || needed > pages.len() || geometry.page_capacity > view.maximum
        || geometry.page_tokens != view.page_tokens as u64 || plane.address != view.base
        || plane.unit_bytes != view.page_bytes as u64 || !allocation.fixed
        || allocation.address != view.base || !matches!(allocation.memory, Memory::Device)
        || pages.iter().any(|p| u64::from(*p) >= geometry.page_capacity)
        || pages.iter().collect::<std::collections::HashSet<_>>().len() != pages.len() {
        return Err(invalid("KV runtime geometry or owned page chain differs from signed layout"));
    }
    let token_bytes = view.page_bytes / view.page_tokens;
    let bytes = consumed.checked_mul(token_bytes).ok_or_else(|| invalid("KV read size overflow"))?;
    let mut output = vec![0u8; bytes];
    let mut mappings = allocation.mappings.clone();
    mappings.sort_unstable_by_key(|(start, _)| *start);
    // All ranges are validated before the first driver call. The virtual
    // maximum alone cannot establish physical mapping coverage.
    let mut reads = Vec::new();
    for (logical, &page) in pages[..needed].iter().enumerate() {
        let first = logical.checked_mul(view.page_tokens).ok_or_else(|| invalid("KV token offset overflow"))?;
        let count = (consumed - first).min(view.page_tokens);
        let size = count.checked_mul(token_bytes).ok_or_else(|| invalid("KV page read overflow"))?;
        let offset = usize::try_from(page).ok().and_then(|p| p.checked_mul(view.page_bytes))
            .ok_or_else(|| invalid("KV physical page offset overflow"))?;
        if offset.checked_add(size).is_none_or(|end| end > allocation.nbytes) {
            return Err(invalid("KV read exceeds the loaded allocation"));
        }
        let start = view.base.checked_add(offset as u64).ok_or_else(|| invalid("KV address overflow"))?;
        let end = start.checked_add(size as u64).ok_or_else(|| invalid("KV end overflow"))?;
        let mut covered = start;
        for &(address, length) in &mappings {
            let finish = address.checked_add(length as u64).ok_or_else(|| invalid("KV mapping overflow"))?;
            if address <= covered && covered < finish { covered = finish.min(end); }
            if covered == end { break; }
        }
        if covered != end { return Err(invalid("KV read touches an unmapped physical range")); }
        reads.push((start, first * token_bytes, size));
    }
    // SAFETY: the request still owns every page and the caller has confirmed
    // zero outstanding graph work. Owner, mappings and output remain alive.
    unsafe {
        resources.driver.check("cuCtxSetCurrent", (resources.driver.context_set)(resources.context))?;
        resources.driver.check("cuCtxSynchronize", (resources.driver.context_sync)())?;
        for (address, offset, length) in reads {
            resources.driver.check("cuMemcpyDtoH_v2", (resources.driver.download)(
                output[offset..offset + length].as_mut_ptr().cast(), address, length))?;
        }
    }
    Ok(output)
}

/// Poison only a quiescent request-owned range which the next graph must rewrite.
/// The caller supplies its complete logical owned/reserved page chain.
pub fn poison_kv_range(model: &NativeModel, rank: usize, view: &KvHistoryView,
    expected: &flare_selene::CacheGeometry, pages: &[u32], range: std::ops::Range<usize>) -> Result<Vec<u8>> {
    if range.start >= range.end || !(view.page_bytes / view.page_tokens).is_multiple_of(2) {
        return Err(invalid("KV poison requires a nonempty BF16 token range"));
    }
    // This validates the complete authenticated layout and mapped prefix before
    // the first write, and retains the untouched committed prefix for comparison.
    let before = download_kv_prefix(model, rank, view, expected, pages, range.end)?;
    let token_bytes = view.page_bytes / view.page_tokens;
    let start_bytes = range.start.checked_mul(token_bytes).ok_or_else(|| invalid("KV poison start overflow"))?;
    let size = (range.end - range.start).checked_mul(token_bytes).ok_or_else(|| invalid("KV poison extent overflow"))?;
    let pattern = [0xc1u8, 0x7f]; // BF16 quiet NaN, retained as bytes.
    let payload = pattern.into_iter().cycle().take(size).collect::<Vec<_>>();
    let resources = &model.ranks[rank];
    let mut writes = Vec::new();
    for logical in range.start / view.page_tokens..range.end.div_ceil(view.page_tokens) {
        let first = range.start.max(logical * view.page_tokens);
        let end = range.end.min((logical + 1) * view.page_tokens);
        let page_offset = usize::try_from(pages[logical]).ok().and_then(|p| p.checked_mul(view.page_bytes))
            .and_then(|offset| offset.checked_add((first % view.page_tokens) * token_bytes))
            .ok_or_else(|| invalid("KV poison page offset overflow"))?;
        let address = view.base.checked_add(page_offset as u64).ok_or_else(|| invalid("KV poison address overflow"))?;
        writes.push((address, (first - range.start) * token_bytes, (end - first) * token_bytes));
    }
    // SAFETY: all ranges were mapped and authenticated by the prefix read.
    // No submitted graph or transfer refers to these request-owned future bytes.
    unsafe {
        resources.driver.check("cuCtxSetCurrent", (resources.driver.context_set)(resources.context))?;
        resources.driver.check("cuCtxSynchronize", (resources.driver.context_sync)())?;
        for (address, offset, length) in writes {
            resources.driver.check("cuMemcpyHtoD_v2", (resources.driver.upload)(address,
                payload[offset..offset + length].as_ptr().cast(), length))?;
        }
    }
    let after = download_kv_prefix(model, rank, view, expected, pages, range.end)?;
    if after[..start_bytes] != before[..start_bytes] || after[start_bytes..] != payload {
        return Err(invalid("KV poison changed the committed prefix or failed its actual readback"));
    }
    Ok(after)
}
