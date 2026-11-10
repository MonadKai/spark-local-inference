# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Network budget acceptance controls only; these do not execute an ASR model."""

from types import SimpleNamespace
from functools import partial

import pytest
import torch

from qualify.asr_prefill_observation import check_decoder_nodes, check_head_nodes
from qualify.asr_audio_network_nodes import check_encoder_nodes, convolution_selection

NAME = "thinker.model.layers.0.input_layernorm.output"


class RotaryPair(torch.nn.Module):
    def forward(self, value):
        return value, value + 1


def fixture():
    reference = torch.tensor([[[1., -1.]]], dtype=torch.bfloat16)
    return ({NAME: reference.reshape(1, 2).clone()},
        SimpleNamespace(get_tensor=lambda name: reference),
        dict(stages={NAME: dict(dtype="BF16", shape=[1, 1, 2], max_abs=0., relative_l2=0.)}))


def test_network_budgets_retain_missing_extra_and_excess_error_failures():
    values, reference, case = fixture()
    result = check_decoder_nodes(values, reference, case)
    assert result["passed"] and not result["local_operator_qualification"]
    with pytest.raises(ValueError, match="coverage"):
        check_decoder_nodes({}, reference, case)
    with pytest.raises(ValueError, match="coverage"):
        check_decoder_nodes({**values, "unexpected": values[NAME]}, reference, case)
    values[NAME][0, 0] = 2
    result = check_decoder_nodes(values, reference, case)
    assert not result["passed"] and not result["stages"][NAME]["passed"]
    assert result["stages"][NAME]["max_abs"] == 1


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_unbudgeted_activation_nodes_cannot_hide_nonfinite_values(number):
    values, reference, case = fixture()
    name = "thinker.model.layers.0.mlp.act_fn.output"
    values[name] = torch.ones((1, 2), dtype=torch.bfloat16)
    result = check_decoder_nodes(values, reference, case)
    assert result["passed"] and result["unbudgeted_activation_nodes"] == [name]
    values[name][0, 0] = number
    with pytest.raises(ValueError, match="nonfinite"):
        check_decoder_nodes(values, reference, case)


def test_network_node_gate_rejects_dtype_and_extent_changes_before_comparison():
    values, reference, case = fixture()
    with pytest.raises(ValueError, match="non-BF16"):
        check_decoder_nodes({NAME: values[NAME].float()}, reference, case)
    with pytest.raises(ValueError, match="geometry"):
        check_decoder_nodes({NAME: values[NAME][:, :1]}, reference, case)


def test_same_element_count_cannot_conceal_reference_or_candidate_layout_drift():
    values, reference, case = fixture()
    changed = reference.get_tensor(NAME).reshape(1, 2, 1)
    with pytest.raises(ValueError, match="geometry"):
        check_decoder_nodes(values, SimpleNamespace(get_tensor=lambda name: changed), case)
    with pytest.raises(ValueError, match="geometry"):
        check_decoder_nodes({NAME: values[NAME].reshape(2, 1)}, reference, case)


@pytest.mark.parametrize("capacity,frames,expected", [
    (100, 2, ((1, 40), (1, 20), (1, 10))),
    (800, 13, ((1, 40), (1, 20), (1, 10))),
    (100, 99, ((1, 100), (1, 50), (1, 25))),
    (800, 99, ((8, 100), (1, 50), (8, 25))),
    (100, 101, ((1, 100), (2, 50), (2, 25))),
    (800, 101, ((8, 100), (8, 50), (4, 25))),
    (800, 401, ((8, 100), (8, 50), (8, 25))),
    (100, 801, ((1, 100), (2, 50), (1, 25))),
    (800, 801, ((8, 100), (8, 50), (8, 25))),
    (100, 3265, ((16, 100), (2, 50), (1, 25))),
    (800, 3265, ((16, 100), (8, 50), (8, 25))),
])
def test_cnn_observation_selects_declared_whole_input_calls(capacity, frames, expected):
    # In particular, a final one-chunk window of a long input keeps whole-N
    # arithmetic. Selection never examines candidate values or scalar errors.
    assert convolution_selection(capacity, frames) == expected


def test_encoder_nodes_require_exact_frozen_axes_and_preserve_failed_budgets():
    name = "thinker.audio_tower.conv_out.output"
    target = torch.ones((2, 13, 1024), dtype=torch.bfloat16)
    reference = SimpleNamespace(get_tensor=lambda key: target)
    case = dict(stages={name: dict(dtype="BF16", shape=list(target.shape), max_abs=0., relative_l2=0.)})
    actual = {name: target.clone()}
    assert check_encoder_nodes(actual, reference, case)["passed"]
    with pytest.raises(ValueError, match="coverage"):
        check_encoder_nodes({}, reference, case)
    with pytest.raises(ValueError, match="geometry"):
        check_encoder_nodes({name: target.reshape(13, 2, 1024)}, reference, case)
    with pytest.raises(ValueError, match="dtype"):
        check_encoder_nodes({name: target.float()}, reference, case)
    actual[name][0, 0, 0] = 2
    assert not check_encoder_nodes(actual, reference, case)["passed"]
    actual[name][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finiteness"):
        check_encoder_nodes(actual, reference, case)


def test_rotary_observation_hook_executes_after_compiler_metadata_roundtrip():
    from ignition.preparation.forward_metadata import dumps, loads
    from qualify.asr_prefill_observation import _copy_rotary

    cosine, sine = torch.empty((3, 128)), torch.empty((3, 128))
    module = RotaryPair()
    module.register_forward_hook(partial(_copy_rotary, cosine=cosine, sine=sine))
    restored = loads(dumps(dict(module=module, source=torch.ones((3, 128)))))
    # This executes the actual registered hook on worker-style fake tensors.
    # The former local closure becomes _Unavailable and fails at this boundary.
    result = restored["module"](restored["source"])
    assert result[0].shape == result[1].shape == (3, 128)
    broken = RotaryPair()
    def local_hook(module, arguments, result):
        pass
    broken.register_forward_hook(local_hook)
    restored = loads(dumps(dict(module=broken, source=torch.ones((3, 128)))))
    with pytest.raises(RuntimeError, match="process-local resource"):
        restored["module"](restored["source"])


def test_convolution_observation_hooks_execute_after_compiler_metadata_roundtrip():
    from ignition.preparation.forward_metadata import dumps, loads
    from qualify.asr_audio_network_nodes import _convolution_input, _convolution_output

    convolution = torch.nn.Conv2d(1, 2, 3, stride=2, padding=1)
    destinations = {(1, 40): [torch.empty((1, 1, 128, 40)), torch.empty((1, 2, 64, 20))]}
    convolution.register_forward_pre_hook(partial(_convolution_input, destinations=destinations))
    convolution.register_forward_hook(partial(_convolution_output, destinations=destinations))
    restored = loads(dumps(dict(module=convolution, source=torch.ones((1, 1, 128, 40)))))
    assert restored["module"](restored["source"]).shape == (1, 2, 64, 20)


def test_head_nodes_gate_actual_hidden_and_logits_separately_without_shape_inference():
    tensors = {"thinker.lm_head.input.0": torch.ones((1, 3, 2), dtype=torch.bfloat16),
        "thinker.lm_head.output": torch.ones((1, 3, 5), dtype=torch.bfloat16)}
    reference = SimpleNamespace(get_tensor=tensors.__getitem__)
    case = dict(stages={name: dict(dtype="BF16", shape=list(value.shape), max_abs=0., relative_l2=0.)
        for name, value in tensors.items()})
    actual = {name: value[0].clone() for name, value in tensors.items()}
    assert check_head_nodes(actual, reference, case)["passed"]
    for name in tensors:
        changed = {key: value.clone() for key, value in actual.items()}
        changed[name][0, 0] = 2
        result = check_head_nodes(changed, reference, case)
        assert not result["passed"] and not result["stages"][name]["passed"]
        changed[name][0, 0] = float("inf")
        with pytest.raises(ValueError, match="finiteness"):
            check_head_nodes(changed, reference, case)
    with pytest.raises(ValueError, match="coverage"):
        check_head_nodes({"thinker.lm_head.output": actual["thinker.lm_head.output"]}, reference, case)
    changed = dict(actual)
    changed["thinker.lm_head.input.0"] = changed["thinker.lm_head.input.0"].reshape(2, 3)
    with pytest.raises(ValueError, match="geometry"):
        check_head_nodes(changed, reference, case)
