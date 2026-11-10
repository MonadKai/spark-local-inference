# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare full-eight-row and logical-live-row preparation on real ASR matrices.

Both use the same legal fixed-reduction candidates and physical eight-row
storage. All compilation/tuning/capture finishes before randomized timings.
This measures selected projection plans, not a complete ASR serving workload.
Packed QKV/gate-up and reduction LM head are diagnostics, not released recipe
composition. In particular this does not replace the production BF16 LM head.
"""

import argparse
from dataclasses import asdict
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
import random


OPERATIONS = (
    ("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
    ("attention_output", ("self_attn.o_proj",)),
    ("gate_up", ("mlp.gate_proj", "mlp.up_proj")), ("down", ("mlp.down_proj",)),
    ("head", ("head",)), ("query", ("self_attn.q_proj",)),
    ("key", ("self_attn.k_proj",)), ("value", ("self_attn.v_proj",)),
    ("gate", ("mlp.gate_proj",)), ("up", ("mlp.up_proj",)))


def component_interval(pairs):
    """Descriptive nine-block paired ratio; never qualifies service performance."""
    if not isinstance(pairs,(list,tuple)) or len(pairs)!=9 \
            or any(not isinstance(pair,(list,tuple)) or len(pair)!=2 for pair in pairs) \
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for pair in pairs for v in pair):
        raise ValueError("nine finite positive paired block timings required")
    with localcontext() as context:
        context.prec=80
        logs=[(Decimal(old)/Decimal(new)).ln() for old,new in pairs]
        mean=sum(logs)/9;variance=sum((value-mean)**2 for value in logs)/8
        critical=Decimal("2.306004135204166")
        width=critical*(variance/9).sqrt()
        exact=[value.exp() for value in (mean,mean-width,mean+width)]
        estimate,lower,upper=map(float,exact)
        if Decimal(lower)>exact[1]:lower=math.nextafter(lower,-math.inf)
        if Decimal(upper)<exact[2]:upper=math.nextafter(upper,math.inf)
    if not all(math.isfinite(value) and value>0 for value in (estimate,lower,upper)):
        raise ValueError("component interval is outside representable range")
    return dict(blocks=9,degrees_of_freedom=8,t_critical=float(critical),
        legacy_over_logical_geometric_ratio=estimate,lower_95=lower,upper_95=upper,
        service_performance_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("retain prior attempts and choose a new output")
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.planning import discover_gpus
    apply(Configuration.load(args.config), discover_gpus())

    import torch
    from safetensors import safe_open
    from ignition.components.reduction_projection import PreparedReductionProjection
    from ignition.preparation.failure import retain_on_failed_drain
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.qwen3_asr.checkpoint import inspect_original_checkpoint
    from ignition.recipes.qwen3_asr.numerics import fp32_reductions

    torch.set_num_threads(2)
    inspect_original_checkpoint(args.checkpoint)
    reference = json.loads((args.reference / "result.json").read_bytes())
    if reference["errors"] or reference["candidate_executed"] \
            or reference["transformers_revision"] != "1d9654e78a23fb167ef539cbd699dfbd3e4830cf":
        raise ValueError("complete independent pinned operands required")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_bytes())["weight_map"]
    record = dict(scope=__doc__.strip(), status="preparing", errors=[], candidates=[],
        source_case="en-422-122949-0013-first-2f", physical_rows=8, logical_rows=[1,2,4,8],
        planned_operations=[name for name,_ in OPERATIONS], planned_preparations=["legacy-eight","logical1","logical2","logical4","logical8"],
        expected_checkpoint_revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
        checkpoint_content_identity_verified_by_this_tool=False,
        timing_blocks=9, graph_replays_per_sample=10, order_seed=20261106,
        component_interval="paired block log-ratio Student t, df8/tcrit2.306004135204166, two-sided95%; separate from formal service statistics",
        torch=torch.__version__, cuda_runtime=torch.version.cuda,
        numerical_control="all bytes versus legacy selected fixed-reduction plan on identical real weights/operands, all layers; independent arithmetic and whole-model gates remain separate",
        uses_standard_preparation_autotuning=True, legal_candidate_set_unchanged=True,
        production_selection_qualified=False,
        activation_scope="Same eight frozen layer-zero represented inputs for every rotated layer weight; not each layer actual activation or full-network precision",
        rotated_outputs="Distinct per-layer guarded matrices for both variants, all actual captured outputs checked",
        full_model_numerical_qualification=False, service_performance_qualified=False,active_cell=None)
    args.output.mkdir()
    def save():
        pending=args.output/"result.pending"
        pending.write_text(json.dumps(record,indent=2)+"\n");pending.replace(args.output/"result.json")
    save()
    owners, graphs = [], []
    with retain_on_failed_drain(0,owners,graphs), torch.inference_mode(), fp32_reductions(), ModuleImages() as images:
        owners.append(images)
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(12,1):
            raise ValueError("one assigned SM121 GPU required")
        session=RecordedPreparationSession(device=torch.device("cuda:0"),checkpoint="7278e1e70fe206f11671096ffdd38061171dd6e5",
            workload_schema="asr-logical-row-projection-v1",numerical_policy="qwen3-asr.bf16-fp32.v1",compile_workers=2)
        owners.append(session)
        cases=[]
        try:
            with safe_open(args.reference/(record["source_case"]+"-prefill.safetensors"),framework="pt") as frozen:
                for name,parts in OPERATIONS:
                    weights=[]
                    for layer in range(1 if name=="head" else 28):
                        pieces=[]
                        for part in parts:
                            key="thinker.lm_head.weight" if part=="head" else f"thinker.model.layers.{layer}.{part}.weight"
                            with safe_open(args.checkpoint/index[key],framework="pt") as shard:
                                pieces.append(shard.get_tensor(key).clone())
                        weight=torch.cat(pieces) if len(pieces)>1 else pieces[0]
                        if weight.dtype!=torch.bfloat16 or not torch.isfinite(weight).all():
                            raise ValueError("invalid original weight")
                        weights.append(weight.cuda());owners.append(weights[-1])
                    n,k=weights[0].shape
                    module="thinker.lm_head" if name=="head" else "thinker.model.layers.0."+parts[0]
                    represented=frozen.get_tensor(module+".input.0").reshape(-1,k)
                    rows=list(range(len(represented)-1,len(represented)-9,-1))
                    if len(represented)<8 or represented.dtype!=torch.bfloat16:
                        raise ValueError("eight actual represented BF16 operands required")
                    basis=represented[rows].contiguous()
                    for part in parts[1:]:
                        other=frozen.get_tensor("thinker.model.layers.0."+part+".input.0").reshape(-1,k)
                        if not torch.equal(other[rows].contiguous().view(torch.uint8),basis.view(torch.uint8)):
                            raise ValueError("packed projections do not share represented inputs")
                    source=basis.cuda();count=torch.full((1,),8,dtype=torch.int32,device="cuda")
                    owners.extend((source,count));operations=[]
                    for live in (None,1,2,4,8):
                        guard=torch.full((8*n+16,),137.,dtype=torch.bfloat16,device="cuda")
                        output=guard[8:-8].view(8,n)
                        operation=PreparedReductionProjection(source,weights[0],output,count,session=session,
                            name=f"{name}-{'legacy' if live is None else live}",benchmark_rows=live)
                        owners.extend((guard,operation));operations.append((operation,guard,live))
                    cases.append(dict(name=name,weights=weights,source=source,basis=basis,count=count,operations=operations,source_rows=rows))
            record["tuning_report"]=session.tune();record["status"]="checking";save()
            from b12x.preparation.types import require_prepared
            for case in cases:
                states=[]
                for operation,guard,live in case["operations"]:
                    state=require_prepared(operation.plan,operation.plan.component_id);states.append(state);owners.append(state)
                    row=dict(operation=case["name"],benchmark_rows=8 if live is None else live,
                        preparation="legacy-eight" if live is None else "logical-live",
                        query=asdict(operation.plan.query),invocation=operation.plan.invocation.to_dict(),
                        selected=asdict(operation.plan.selection.config),source_rows=case["source_rows"],
                        rotated_weight_sets=len(case["weights"]),weight_payload_bytes=sum(w.nbytes for w in case["weights"]),
                        composition_matches_current_decode=case["name"] not in ("qkv","gate_up","head"),
                        includes_prefill_first_token_input=case["name"]=="head",observations=[],timings=[])
                    record["candidates"].append(row);case.setdefault("rows",[]).append(row)
                    record["active_cell"]=dict(operation=case["name"],preparation=row["preparation"],benchmark_rows=row["benchmark_rows"],stage="capture");save()
                    captured=[]
                    for weight in case["weights"]:
                        storage=torch.full((operation.output.numel()+16,),137.,dtype=torch.bfloat16,device="cuda")
                        target=storage[8:-8].view_as(operation.output);captured.append((storage,target));owners.append(storage)
                        state.run(case["source"],weight,target,case["count"])
                    torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph();graphs.append(graph)
                    with session.capture(),torch.cuda.graph(graph):
                        for weight,(_,target) in zip(case["weights"],captured,strict=True):
                            state.run(case["source"],weight,target,case["count"])
                    case.setdefault("captured_outputs",[]).append(captured)
                    case.setdefault("graphs",[]).append(graph)
                case["states"]=states
                for live in (1,2,4,8,3,7,0):
                    case["source"].copy_(case["basis"]);case["source"][live:].fill_(float("nan"));case["count"].fill_(live)
                    baseline=[]
                    for i,(state,(operation,guard,_)) in enumerate(zip(states,case["operations"],strict=True)):
                        record["active_cell"]=dict(operation=case["name"],preparation=case["rows"][i]["preparation"],live_rows=live,stage="direct-byte-check")
                        for layer,weight in enumerate(case["weights"]):
                            record["active_cell"]["layer"]=layer
                            guard.fill_(137);state.run(case["source"],weight,operation.output,case["count"]);torch.cuda.synchronize()
                            actual=operation.output.cpu().clone()
                            if i==0:baseline.append(actual)
                            if not torch.equal(actual.view(torch.uint8),baseline[layer].view(torch.uint8)) \
                                    or not torch.isfinite(actual).all() or not actual[live:].eq(0).all() \
                                    or not guard[:8].eq(137).all() or not guard[-8:].eq(137).all():
                                raise ValueError("selected projection bytes/finite/guards differ")
                        for repeat in range(5):
                            record["active_cell"]=dict(operation=case["name"],preparation=case["rows"][i]["preparation"],live_rows=live,repeat=repeat,stage="captured-byte-check")
                            for storage,_ in case["captured_outputs"][i]:storage.fill_(137)
                            torch.cuda.synchronize();before=torch.cuda.memory_stats();case["graphs"][i].replay();torch.cuda.synchronize();after=torch.cuda.memory_stats()
                            stable=all(before[key]==after[key] for key in ("allocation.all.allocated","allocation.all.freed","segment.all.allocated","segment.all.freed"))
                            if not stable:raise ValueError("selected captured graph allocations differ")
                            for layer,(storage,target) in enumerate(case["captured_outputs"][i]):
                                record["active_cell"]["layer"]=layer
                                if not torch.equal(target.cpu().view(torch.uint8),baseline[layer].view(torch.uint8)) \
                                        or not storage[:8].eq(137).all() or not storage[-8:].eq(137).all():
                                    raise ValueError("selected captured graph bytes/guards differ")
                        case["rows"][i]["observations"].append(dict(live_rows=live,all_weight_direct_and_captured_bytes_exact=True,captured_repeats=5,torch_allocation_stable=True))
                save()
            # Capture and correctness finish for the entire catalog before sampling.
            record["status"]="timing";save();order=random.Random(record["order_seed"])
            for case in cases:
                case["source"].copy_(case["basis"])
                cells=[(live,index) for live in (1,2,4,8) for index in (0,(1,2,4,8).index(live)+1)]
                for block in range(record["timing_blocks"]):
                    order.shuffle(cells)
                    for live,index in cells:
                        record["active_cell"]=dict(operation=case["name"],preparation=case["rows"][index]["preparation"],live_rows=live,block=block,stage="timing")
                        case["count"].fill_(live);graph=case["graphs"][index]
                        for _ in range(3):graph.replay()
                        torch.cuda.synchronize()
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);owners.extend((begin,end))
                        before=torch.cuda.memory_stats();begin.record()
                        for _ in range(record["graph_replays_per_sample"]):graph.replay()
                        end.record();end.synchronize();after=torch.cuda.memory_stats()
                        if any(before[key]!=after[key] for key in ("allocation.all.allocated","allocation.all.freed","segment.all.allocated","segment.all.freed")):
                            raise ValueError("timed replay changed allocations")
                        case["rows"][index]["timings"].append(dict(block=block,logical_rows=live,
                            microseconds_per_projection=begin.elapsed_time(end)*1000/(record["graph_replays_per_sample"]*len(case["weights"]))))
                    save()
            record["component_intervals"]=[]
            for case in cases:
                for live in (1,2,4,8):
                    legacy_rows=[item for item in case["rows"][0]["timings"] if item["logical_rows"]==live]
                    current_rows=case["rows"][(1,2,4,8).index(live)+1]["timings"]
                    legacy={item["block"]:item["microseconds_per_projection"] for item in legacy_rows}
                    current={item["block"]:item["microseconds_per_projection"] for item in current_rows}
                    if len(case["rows"][0]["timings"])!=36 or len(legacy_rows)!=9 or len(current_rows)!=9 \
                            or any(item["logical_rows"]!=live for item in current_rows) \
                            or set(legacy)!=set(range(9)) or set(current)!=set(range(9)):
                        raise ValueError("missing/duplicate fixed timing blocks")
                    record["component_intervals"].append(dict(operation=case["name"],logical_rows=live,
                        **component_interval([(legacy[block],current[block]) for block in range(9)])))
            torch.cuda.synchronize()
        except BaseException as error:
            record.update(status="failed_or_interrupted");record["errors"].append(dict(type=type(error).__name__,message=str(error),active_cell=record["active_cell"]))
            try:save()
            except BaseException as reporting_error:error.add_note(f"could not publish projection failure: {reporting_error!r}")
            raise
        finally:
            from ignition.preparation.failure import resources_retained, drain_or_retain, retain_undrained
            import sys
            error=sys.exception()
            if error is None or not resources_retained(error):
                try:
                    torch.cuda.synchronize()
                    for graph in graphs:graph.reset()
                    session.close()
                except BaseException as cleanup_error:
                    if error is not None:
                        if not drain_or_retain(cleanup_error,0,owners,graphs):
                            retain_undrained(error,owners,graphs)
                        error.add_note(f"projection cleanup failed: {cleanup_error!r}")
                    else:
                        drain_or_retain(cleanup_error,0,owners,graphs)
                        raise
    record.update(status="complete_selected_projection_diagnostic_only",active_cell=None);save()


if __name__=="__main__":
    main()
