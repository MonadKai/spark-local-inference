# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Original checkpoint structural settings, independently encoded prompt IDs."""

import json
import os
from pathlib import Path
import shutil

import pytest

from ignition.preparation.checkpoint import CheckpointError
from ignition.recipes.qwen3_asr.config import canonical_config


@pytest.fixture(scope="module")
def snapshot():
    configured = os.environ.get("FLARE_QWEN3_ASR_SNAPSHOT")
    if configured is None:
        pytest.skip("set FLARE_QWEN3_ASR_SNAPSHOT to a complete original checkpoint")
    path = Path(configured)
    assert path.is_dir()
    return path


def _metadata(snapshot, target):
    for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
        shutil.copyfile(snapshot / name, target / name)
    return target


def test_preparation_costs_match_physical_page_and_continuation_layout():
    from ignition.recipes.qwen3_asr.pool import PAGE_TOKENS, continuation_layout, page_bytes
    from ignition.recipes.qwen3_asr.recipe import RECIPE

    assert RECIPE.page_tokens == PAGE_TOKENS
    assert RECIPE.ranks[1].page_bytes == page_bytes()
    assert RECIPE.ranks[1].state_row_bytes == continuation_layout().row_bytes


def test_original_padding_and_full_attention_override_native_format_defaults(snapshot):
    config = canonical_config(snapshot)
    assert config.pad_token_id == config.text_config.pad_token_id == 151643
    assert config.eos_token_id == [151643, 151645]
    assert config.audio_config.max_position_embeddings == 13
    assert config.audio_config.output_dim == config.text_config.hidden_size == 2048
    assert config.text_config.layer_types == ["full_attention"] * 28
    assert config.text_config.sliding_window is None
    assert config.text_config.rope_parameters == {"rope_type": "default", "rope_theta": 1000000}


def test_generation_distribution_fields_do_not_define_preparation_sampling(snapshot, tmp_path):
    target = _metadata(snapshot, tmp_path)
    original = canonical_config(target).to_dict()
    path = target / "generation_config.json"
    values = json.loads(path.read_text())
    values.update(do_sample=True, temperature=1.9, top_k=71, top_p=0.3)
    path.write_text(json.dumps(values))
    assert canonical_config(target).to_dict() == original


@pytest.mark.parametrize("field,value", [("eos_token_id", [151645]), ("pad_token_id", 151645)])
def test_reject_changed_generation_structure(snapshot, tmp_path, field, value):
    target = _metadata(snapshot, tmp_path)
    path = target / "generation_config.json"
    values = json.loads(path.read_text()); values[field] = value
    path.write_text(json.dumps(values))
    with pytest.raises(CheckpointError, match="EOS|pad"):
        canonical_config(target)


def test_asr_delimiter_is_not_a_special_or_stop_token(snapshot, tmp_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False, local_files_only=True)
    assert tokenizer.encode("<asr_text>", add_special_tokens=False) == [151704]
    assert tokenizer.added_tokens_decoder[151704].special is False
    assert 151705 not in tokenizer.added_tokens_decoder
    target = _metadata(snapshot, tmp_path)
    path = target / "tokenizer_config.json"
    values = json.loads(path.read_text()); values["added_tokens_decoder"]["151704"]["special"] = True
    path.write_text(json.dumps(values))
    with pytest.raises(CheckpointError, match="151704"):
        canonical_config(target)


def test_real_original_template_has_an_empty_system_turn(snapshot):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False, local_files_only=True)
    source = json.loads((snapshot / "chat_template.json").read_text())["chat_template"]
    ids = tokenizer.apply_chat_template([{"role": "user", "content": [{"type": "audio"}]}],
        chat_template=source, tokenize=True, add_generation_prompt=True, return_dict=False)
    assert ids == [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669,
                   151676, 151670, 151645, 198, 151644, 77091, 198]
    assert tokenizer.encode("language English<asr_text>", add_special_tokens=False) == [11528, 6364, 151704]
    assert tokenizer.encode("language Chinese<asr_text>", add_special_tokens=False) == [11528, 8453, 151704]


def test_canonical_meta_tree_has_every_original_parameter(snapshot):
    from ignition.recipes.qwen3_asr.checkpoint import inspect_original_checkpoint
    from ignition.recipes.qwen3_asr.model import create_model_tree

    checkpoint = inspect_original_checkpoint(snapshot)
    tree = create_model_tree(checkpoint)
    assert len(dict(tree.named_parameters(remove_duplicate=False))) == 708
    assert all(parameter.is_meta for parameter in tree.parameters())
    assert tree.model.language_model.layers[0].self_attn.q_norm.weight.shape == (128,)
    assert tree.model.language_model.layers[0].self_attn.k_norm.weight.shape == (128,)
