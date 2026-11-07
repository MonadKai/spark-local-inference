# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU original-format Qwen3-ASR preparation declaration."""

from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults

from .checkpoint import MODEL_ID, REVISION


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if plan.world_size != 1 or ranks is not None or plan.speculation is not None:
        raise ValueError("Qwen3-ASR prepares one local GPU without speculative decoding")
    config = PreparationConfig(max_active=plan.concurrency, context=plan.context,
        prefill_chunk_size=plan.prefill_chunk, snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, compile_workers=plan.compile_workers,
        collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica)
    return prepare_model(snapshot, config, progress=progress, package_storage=package_storage)


def usage(prepared):
    return (RankUsage(prepared.memory_report["pages"], prepared.memory_report["free_device_bytes"]),)


RECIPE = Recipe(name="qwen3_asr", served_model_name=MODEL_ID,
    prepare=prepare, usage=usage, context=8192, page_tokens=256, concurrency=8,
    max_concurrency=8, max_draft_tokens=0, prefill_chunk=4096,
    max_body_bytes=32 * 1024 * 1024,
    # Positive-temperature request defaults only: inclusive top-k=1 is still
    # stochastic at ties. This recipe's graph uses the approved fixed argmax;
    # see docs/qwen3-asr-design.md, D2. Native positive-temperature checks remain.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=1, min_p=0.,
        presence_penalty=0., repetition_penalty=1., frequency_penalty=0.),
    checkpoints=(Checkpoint(alias="qwen3-asr-1.7b", repository=MODEL_ID, revision=REVISION,
        architecture="Qwen3ASRForConditionalGeneration", stored_bytes=4_698_434_816,
        device_bytes=4_698_434_816),),
    # Initial capacity budgeting; the prepared package records actual storage.
    ranks=MappingProxyType({1: RankCosts(29_364_480, 29_360_128, 2 << 30)}))
