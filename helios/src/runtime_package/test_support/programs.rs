/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Read authenticated launch geometry for matching native kernel timelines.
//! This does not instantiate graphs, load CUDA, or supply operator semantics.

use super::super::{AuthorizedPackage, format::{Dependencies, Node, Package, invalid},
    payload::Files, protection::AuthorizedKind, protobuf};
use crate::Result;


fn argument_storage(package: &Package, arguments: &[super::super::format::Blob]) -> Vec<serde_json::Value> {
    let mut values=Vec::new();
    for (argument,blob) in arguments.iter().enumerate() {
        for relocation in &blob.relocations {
            let kind=match relocation.kind {
                super::super::format::RelocationKind::Pointer=>"pointer",
                super::super::format::RelocationKind::TensorMap=>"tensor_map",
            };
            values.push(serde_json::json!({"argument":argument,"argument_offset":relocation.offset,
                "encoding":kind,"allocation":relocation.target.allocation,"offset":relocation.target.offset}));
        }
        if blob.relocations.is_empty() && blob.data.len()==8 {
            let address=u64::from_le_bytes(blob.data.as_slice().try_into().unwrap());
            for allocation in &package.allocations {
                if let Some(offset)=allocation.address.and_then(|base| address.checked_sub(base))
                    .and_then(|offset| usize::try_from(offset).ok()).filter(|offset| *offset<allocation.extent()) {
                    values.push(serde_json::json!({"argument":argument,"argument_offset":0,
                        "encoding":"untyped_fixed_address_candidate","allocation":allocation.name,"offset":offset}));
                }
            }
        }
    }
    values
}

fn program(package: &Package, graph: &super::super::format::Graph,
    mut records: super::super::payload::RecordReader<'_>) -> Result<serde_json::Value> {
    let execution = package.execution.as_ref().ok_or_else(|| invalid("program observation requires execution metadata"))?;
    let serving = execution.graphs.iter().filter(|entry| entry.name == graph.name).collect::<Vec<_>>();
    let tuning = execution.tuning_graphs.iter().filter(|entry| entry.name == graph.name).count();
    let role = match (serving.len(), tuning) {
        (1, 0) => "serving",
        (0, 1) => "startup_tuning",
        _ => return Err(invalid("program needs one known serving or startup-tuning descriptor")),
    };
    let serving=serving.first().copied();
    let mut nodes = Vec::new();
    for ordinal in 0..graph.node_count {
        let node = protobuf::node(records.read()?)?;
        package.validate_node(&node)?;
        let operation = match node {
            Node::Kernel { function, grid, block, shared, arguments, .. } => {
                if !graph.functions.contains(&function) { return Err(invalid("kernel is outside graph dependency index")); }
                let function = package.functions.iter().find(|entry| entry.id == function)
                    .ok_or_else(|| invalid("program names an absent function"))?;
                serde_json::json!({"type":"kernel","module":function.module,"symbol":function.symbol,
                    "grid":grid,"block":block,"shared":shared,"argument_storage":argument_storage(package,&arguments)})
            }
            Node::Memset { pitch, value, element_size, width, height, .. } =>
                serde_json::json!({"type":"memset","pitch":pitch,"value":value,
                    "element_size":element_size,"width":width,"height":height}),
            Node::Copy { nbytes, .. } => serde_json::json!({"type":"copy","nbytes":nbytes}),
            Node::Empty => serde_json::json!({"type":"empty"}),
            Node::EventWait { event } => serde_json::json!({"type":"event_wait","event":event}),
        };
        nodes.push(serde_json::json!({"ordinal":ordinal,"operation":operation}));
    }
    let mut dependencies = Dependencies::new(graph.node_count);
    let mut edges = Vec::new();
    for _ in 0..graph.edge_count {
        let edge = protobuf::edge(records.read()?)?;
        dependencies.add(&edge)?;
        edges.push(serde_json::json!({"source":edge.source,"destination":edge.destination,"data":edge.data}));
    }
    records.finish()?;
    dependencies.finish()?;
    Ok(serde_json::json!({"name":graph.name,"role":role,"kind":serving.map(|g| &g.kind),
        "batch_capacity":serving.map(|g| g.batch_capacity),"token_capacity":serving.map(|g| g.token_capacity),
        "node_count":graph.node_count,"edge_count":graph.edge_count,"nodes":nodes,"edges":edges}))
}

fn read(package: &Package, files: &dyn Files) -> Result<serde_json::Value> {
    let graphs = package.graphs.iter().map(|graph| program(package, graph,
        files.records(package, &graph.resource)?)).collect::<Result<Vec<_>>>()?;
    Ok(serde_json::json!({"scope":"Authenticated package launch geometry only; CUDA node IDs, runtime changes, complete trace coverage and layer/operator semantics require separate observations.",
        "graphs":graphs,"function_uses_cuda_api":false,"operator_attribution_qualified":false,"performance_qualified":false}))
}

/// Follow the existing authenticated resource reader for each package layout.
pub fn observe(package: &AuthorizedPackage, rank: usize) -> Result<serde_json::Value> {
    match &package.kind {
        AuthorizedKind::Payload(payload) => read(payload.rank_packages().get(rank)
            .ok_or_else(|| invalid("program rank is absent"))?, payload.source()),
        AuthorizedKind::Legacy { archive, index } => read(&protobuf::decode(index.ranks.get(rank)
            .ok_or_else(|| invalid("program rank is absent"))?.clone())?, archive),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_package::{Publisher, VerifiedPackage};

    #[test]
    #[ignore = "requires explicit signed package/publisher and fresh program-output path; CPU only"]
    fn observe_explicit_signed_program_without_cuda_loading() {
        let path = std::env::var("FLARE_PROMPT_TEST_PACKAGE").unwrap();
        let publishers: Vec<Publisher> = serde_json::from_slice(&std::fs::read(
            std::env::var("FLARE_PROMPT_TEST_PUBLISHERS").unwrap()).unwrap()).unwrap();
        let verified = VerifiedPackage::open(std::path::Path::new(&path), &publishers, None).unwrap();
        let namespace = verified.publisher().namespace.clone();
        let package = verified.authorize(&namespace, false).unwrap();
        let mut result = observe(&package, 0).unwrap();
        assert!(observe(&package, usize::MAX).is_err());
        if matches!(&package.kind,AuthorizedKind::Payload(_)) {
            use crate::package::{reader::Metadata,protection::{verify,Selection}};
            let metadata=Metadata::open(std::path::Path::new(&path)).unwrap();
            let verified=verify(&metadata.envelope,&publishers,Selection::Deployment {deployment:0,rank:Some(0)}).unwrap();
            assert_eq!(verified.envelope_digest(),package.envelope_digest());
            let rank=verified.rank(0).unwrap();
            let views=rank.prepared_weights.as_ref().map(|catalog| catalog.views.iter().map(|view| {
                let allocation=&rank.allocations[usize::try_from(view.allocation).unwrap()];
                serde_json::json!({"name":view.name,"group":view.group,"dtype":view.dtype,"element_bytes":view.element_bytes,
                    "shape":view.shape,"stride":view.stride,"layout_type":view.layout_type,
                    "allocation":allocation.name,"offset":view.offset,"nbytes":view.nbytes})
            }).collect::<Vec<_>>());
            result["prepared_weight_views"]=serde_json::json!(views);
            result["prepared_weight_catalog_version"]=serde_json::json!(rank.prepared_weights.as_ref().map(|catalog| catalog.version));
            result["prepared_weight_unavailable"]=serde_json::json!(rank.prepared_weights.as_ref().map(|catalog| &catalog.unavailable));
            result["weight_catalog_envelope_matches_program_reader"]=true.into();
        }
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        assert!(!maps.contains("libcuda.so") && !maps.contains("libcudart.so")
            && !maps.contains("libtorch") && !maps.contains("libpython"));
        let mut file = std::fs::OpenOptions::new().create_new(true).write(true)
            .open(std::env::var("FLARE_PROGRAM_OBSERVATION_OUTPUT").unwrap()).unwrap();
        serde_json::to_writer(&mut file, &result).unwrap();
    }

    fn fixture() -> (Package, Vec<protobuf::wire::Node>, Vec<protobuf::wire::Edge>) {
        use super::super::super::format::{Allocation, Context, Function, Graph, Memory};
        use protobuf::wire as w;
        let reference = || Some(w::Reference { allocation:"scratch".into(), offset:0 });
        let nodes = vec![
            w::Node { operation:Some(w::node::Operation::Kernel(w::Kernel {
                function:0, grid:vec![2,1,1], block:vec![128,1,1], shared:32, ..Default::default() })) },
            w::Node { operation:Some(w::node::Operation::Memset(w::Memset {
                destination:reference(), pitch:0, value:37, element_size:1, width:8, height:1 })) },
            w::Node { operation:Some(w::node::Operation::Copy(w::Copy {
                source:reference(), destination:reference(), nbytes:8 })) },
            w::Node { operation:Some(w::node::Operation::Empty(w::Empty {})) },
            w::Node { operation:Some(w::node::Operation::EventWait(w::EventWait { event:"ready".into() })) }];
        let edges = (0..4).map(|i| w::Edge { source:i, destination:i+1, data:vec![0;8] }).collect();
        let package = Package { resources:Default::default(), architecture:[12,1], modules:vec![],
            allocations:vec![Allocation { name:"scratch".into(), nbytes:64, alignment:16, file:None,
                data_spans:vec![],sources:vec![],relocations:vec![],address:None,memory:Memory::Device,sizing:None }],
            functions:vec![Function { id:0,module:"module".into(),symbol:"observed".into(),parameter_sizes:vec![],attributes:vec![] }],
            graphs:vec![Graph { name:"decode1".into(),flags:0,resource:"program".into(),node_count:5,edge_count:4,functions:vec![0] }],
            reservations:vec![],context:Context::default(),events:vec!["ready".into()],
            execution:Some(w::Execution { graphs:vec![w::ExecutionGraph { name:"decode1".into(),kind:"decode".into(),
                batch_capacity:1,token_capacity:1,..Default::default() }],..Default::default() }) };
        (package, nodes, edges)
    }

    fn bytes(nodes: &[protobuf::wire::Node], edges: &[protobuf::wire::Edge]) -> Vec<u8> {
        use prost::Message;
        let mut data = Vec::new();
        for item in nodes.iter().map(Message::encode_to_vec).chain(edges.iter().map(Message::encode_to_vec)) {
            data.extend(u32::try_from(item.len()).unwrap().to_le_bytes());data.extend(item);
        }
        data
    }

    fn check(package: &Package, data: &[u8]) -> Result<serde_json::Value> {
        program(package,&package.graphs[0],super::super::super::payload::RecordReader::Slice { bytes:data.to_vec(),position:0 })
    }

    #[test]
    fn decoded_program_retains_all_node_geometry_and_edges() {
        let (package,nodes,edges) = fixture();
        let value = check(&package,&bytes(&nodes,&edges)).unwrap();
        assert_eq!(value["role"],"serving");assert_eq!(value["nodes"].as_array().unwrap().len(),5);
        assert_eq!(value["nodes"][0]["operation"],serde_json::json!({"type":"kernel","module":"module",
            "symbol":"observed","grid":[2,1,1],"block":[128,1,1],"shared":32,"argument_storage":[]}));
        assert_eq!(value["nodes"][1]["operation"]["width"],8);
        assert_eq!(value["nodes"][2]["operation"]["nbytes"],8);
        assert_eq!(value["nodes"][3]["operation"]["type"],"empty");
        assert_eq!(value["nodes"][4]["operation"]["event"],"ready");
        assert_eq!(value["edges"],serde_json::json!([
            {"source":0,"destination":1,"data":[0,0,0,0,0,0,0,0]},
            {"source":1,"destination":2,"data":[0,0,0,0,0,0,0,0]},
            {"source":2,"destination":3,"data":[0,0,0,0,0,0,0,0]},
            {"source":3,"destination":4,"data":[0,0,0,0,0,0,0,0]}]));
    }

    #[test]
    fn startup_tuning_program_is_distinct_and_unknown_descriptor_is_rejected() {
        let (mut package,nodes,edges) = fixture();let data=bytes(&nodes,&edges);
        let execution=package.execution.as_mut().unwrap();execution.graphs.clear();
        assert!(check(&package,&data).is_err());
        package.execution.as_mut().unwrap().tuning_graphs.push(protobuf::wire::TuningGraph {
            name:"decode1".into(),..Default::default() });
        let value=check(&package,&data).unwrap();assert_eq!(value["role"],"startup_tuning");
        assert!(value["kind"].is_null() && value["batch_capacity"].is_null() && value["token_capacity"].is_null());
        let duplicate=package.execution.as_ref().unwrap().tuning_graphs[0].clone();
        package.execution.as_mut().unwrap().tuning_graphs.push(duplicate);
        assert!(check(&package,&data).is_err());
    }

    #[test]
    fn malformed_records_launches_function_index_and_dag_are_rejected() {
        for defect in ["trailing","truncated","unknown-function","outside-index","bad-grid","bad-copy",
            "bad-memset","unknown-event","duplicate-edge","cycle","out-of-range-edge"] {
            let (mut package,mut nodes,mut edges)=fixture();
            match defect {
                "unknown-function" => if let Some(protobuf::wire::node::Operation::Kernel(k))=&mut nodes[0].operation { k.function=1; },
                "outside-index" => package.graphs[0].functions.clear(),
                "bad-grid" => if let Some(protobuf::wire::node::Operation::Kernel(k))=&mut nodes[0].operation { k.grid[0]=0; },
                "bad-copy" => if let Some(protobuf::wire::node::Operation::Copy(c))=&mut nodes[2].operation { c.nbytes=65; },
                "bad-memset" => if let Some(protobuf::wire::node::Operation::Memset(m))=&mut nodes[1].operation { m.width=0; },
                "unknown-event" => package.events.clear(),
                "duplicate-edge" => edges[1]=edges[0].clone(),
                "cycle" => edges.push(protobuf::wire::Edge {source:4,destination:0,data:vec![0;8]}),
                "out-of-range-edge" => edges[0].source=5,
                _ => (),
            }
            package.graphs[0].edge_count=edges.len();let mut data=bytes(&nodes,&edges);
            if defect=="trailing" { data.push(0); }
            if defect=="truncated" { data.pop(); }
            assert!(check(&package,&data).is_err(),"missed {defect}");
        }
    }


    #[test]
    fn storage_references_retain_exact_offsets_and_do_not_claim_scalar_pointer_types() {
        use protobuf::wire as w;
        let (mut package,mut nodes,edges)=fixture();
        package.allocations[0].address=Some(0x1000);package.functions[0].parameter_sizes=vec![8];
        let blob=w::Blob {data:vec![0;8],relocations:vec![w::Relocation {offset:0,
            target:Some(w::Reference {allocation:"scratch".into(),offset:8}),kind:w::RelocationKind::Pointer as i32}]};
        if let Some(w::node::Operation::Kernel(kernel))=&mut nodes[0].operation {kernel.arguments=vec![blob];}
        let value=check(&package,&bytes(&nodes,&edges)).unwrap();
        assert_eq!(value["nodes"][0]["operation"]["argument_storage"],serde_json::json!([
            {"argument":0,"argument_offset":0,"encoding":"pointer","allocation":"scratch","offset":8}]));
        if let Some(w::node::Operation::Kernel(kernel))=&mut nodes[0].operation {
            kernel.arguments=vec![w::Blob {data:0x1008u64.to_le_bytes().to_vec(),relocations:vec![]}];
        }
        let value=check(&package,&bytes(&nodes,&edges)).unwrap();
        assert_eq!(value["nodes"][0]["operation"]["argument_storage"][0]["encoding"],"untyped_fixed_address_candidate");
        assert_eq!(value["nodes"][0]["operation"]["argument_storage"][0]["offset"],8);
        if let Some(w::node::Operation::Kernel(kernel))=&mut nodes[0].operation {kernel.arguments[0].data=7u64.to_le_bytes().to_vec();}
        let value=check(&package,&bytes(&nodes,&edges)).unwrap();
        assert_eq!(value["nodes"][0]["operation"]["argument_storage"],serde_json::json!([]));
    }

}
