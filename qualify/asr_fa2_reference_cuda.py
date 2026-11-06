# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualify the unmodified SDK FA2 window contract using exact analytic controls.

Run in the independent original-format SDK environment. This loads checkpoint
configuration only, uses identity/zero control weights, and never establishes
checkpoint-forward accuracy, candidate accuracy or serving performance.
"""

import argparse
import inspect
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import flash_attn
    import transformers
    from qwen_asr.core.transformers_backend import configuration_qwen3_asr as configuration
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr as modeling

    if (torch.__version__, transformers.__version__, flash_attn.__version__) != (
            "2.13.0+cu130", "4.57.6", "2.8.3"):
        raise ValueError("independent reference versions changed")
    for module in (configuration, modeling):
        expected = args.sdk_source / "qwen_asr/core/transformers_backend" / Path(inspect.getfile(module)).name
        if Path(inspect.getfile(module)).read_bytes() != expected.read_bytes():
            raise ValueError("installed SDK module differs from supplied pinned source")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("this qualification is assigned to one SM121 GPU")
    config = configuration.Qwen3ASRAudioEncoderConfig(**json.loads(
        (args.checkpoint / "config.json").read_text())["thinker_config"]["audio_config"])
    assert (config.d_model, config.encoder_attention_heads) == (1024, 16)
    config._attn_implementation = "flash_attention_2"
    torch.manual_seed(0)
    module = modeling.Qwen3ASRAudioAttention(config).eval().to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for projection in (module.q_proj, module.k_proj, module.v_proj, module.out_proj):
            projection.weight.zero_()
            projection.bias.zero_()
        identity = torch.eye(config.d_model, device="cuda", dtype=torch.bfloat16)
        module.v_proj.weight.copy_(identity)
        module.out_proj.weight.copy_(identity)
        lengths = [1, 13, 26, 104]
        values = [.5, 1., 2., 4.]
        hidden = torch.cat([torch.full((length, config.d_model), value, device="cuda", dtype=torch.bfloat16)
                            for length, value in zip(lengths, values)])
        cumulative = torch.tensor([0, 1, 14, 40, 144], device="cuda", dtype=torch.int32)
        first = module(hidden, cu_seqlens=cumulative)
        assert torch.isfinite(first).all() and first.dtype == torch.bfloat16
        # Zero Q/K gives uniform attention; a constant V in each independent
        # window must return exactly that representable BF16 value.
        assert torch.equal(first, hidden), "SDK FA2 did not preserve independent analytic windows"
        for _ in range(5):
            assert torch.equal(module(hidden, cu_seqlens=cumulative), first), "FA2 reference is not repeatable"
        perturbed = hidden.clone()
        perturbed[40:] = 16.
        changed = module(perturbed, cu_seqlens=cumulative)
        assert torch.equal(changed[:40], first[:40]), "another window changed earlier outputs"
        assert torch.equal(changed[40:], perturbed[40:])
        merged = module(hidden, cu_seqlens=torch.tensor([0, 144], device="cuda", dtype=torch.int32))
        assert not torch.equal(merged, first), "window-isolation negative control failed to distinguish merged attention"
        torch.cuda.synchronize()
    result = dict(scope="SDK FA2 analytic window/repeatability controls only; no checkpoint forward or performance",
        sdk_revision="7c6daf77a2421100f5fb066495372c00129d39ff", torch=torch.__version__,
        transformers=transformers.__version__, flash_attention=flash_attn.__version__,
        gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        lengths=lengths, constant_values=values, repeated_calls=5, independent_windows=True,
        other_window_perturbation_isolated=True, merged_window_negative_control_detected=True,
        installed_sdk_modules_match_supplied_source=True, checkpoint_weights_loaded=False)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
