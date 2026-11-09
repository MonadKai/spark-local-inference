# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only boundary metadata and bounded graph-submission diagnostics.

Only FULL/PIECEWISE top-level successful submissions are counted; NONE is
unobserved, and a PIECEWISE call is not a count of segment graph replays.
The segment registry is process-wide.
Diagnostic instrumentation requires separate overhead calibration before any
formal performance claim. A successful submission is not a GPU busy interval.
"""
from collections import Counter
from functools import wraps
from asr_vllm_encoder_trial_worker import EncoderTrialWorker


def descriptor(value):
    return {name: getattr(value, name) for name in
        ('num_tokens', 'num_reqs', 'uniform_token_count', 'max_query_len', 'num_active_loras')
        if hasattr(value, name)}


class AttentionTrialWorker(EncoderTrialWorker):
    def attention_execution_configuration(self):
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention import Attention
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.compilation.counter import compilation_counter
        runner = self.model_runner
        config = runner.model_state.vllm_config
        groups = []
        for row in runner.attn_groups:
            for group in row:
                builder = group.get_metadata_builder(0)
                layers = get_layers_from_vllm_config(config, Attention, group.layer_names)
                scalar = {}
                for name in ('use_trtllm_decode_attention', 'use_dedicated_xqa',
                        'num_qo_heads', 'num_kv_heads', 'head_dim', 'page_size'):
                    if hasattr(builder, name): scalar[name] = getattr(builder, name)
                selected = getattr(builder, 'flashinfer_trtllm_api_decode_kernel', None)
                scalar['selected_decode_kernel'] = None if selected is None else selected.value
                for name in ('q_data_type_prefill', 'q_data_type_decode'):
                    if hasattr(builder, name): scalar[name] = str(getattr(builder, name))
                groups.append(dict(backend=group.backend.get_name(),
                    builder=type(builder).__module__ + '.' + type(builder).__qualname__,
                    selected=scalar, layer_count=len(layers),
                    implementation_classes=sorted({type(layer.impl).__module__ + '.' + type(layer.impl).__qualname__ for layer in layers.values()}),
                    flash_attention_versions=sorted({layer.impl.vllm_flash_attn_version for layer in layers.values() if hasattr(layer.impl, 'vllm_flash_attn_version')}),
                    manager_block_size=group.kv_cache_spec.block_size,
                    kernel_block_size=runner.kernel_block_sizes[group.kv_cache_group_id],
                    cache_dtype=str(group.kv_cache_spec.dtype)))
        manager = runner.cudagraph_manager
        full = [] if manager is None else [dict(mode=value.cg_mode.name, **descriptor(value)) for value in manager.graphs]
        piecewise = []
        for wrapper in list(CUDAGraphWrapper._all_instances):
            piecewise.append(dict(wrapper_id=id(wrapper), mode=wrapper.runtime_mode.name,
                captured=[descriptor(key) for key, value in wrapper.concrete_cudagraph_entries.items() if value.cudagraph is not None]))
        return dict(scope=__doc__, groups=groups,
            cache_layout=config.cache_config.get_resolved_kv_cache_layout().name,
            cache_blocks=runner.kv_cache_config.num_blocks,
            compilation=dict(mode=config.compilation_config.mode.name,
                cudagraph_mode=config.compilation_config.cudagraph_mode.name,
                capture_sizes=config.compilation_config.cudagraph_capture_sizes),
            compilation_counters={name: getattr(compilation_counter, name) for name in
                ('num_cudagraph_captured', 'num_inductor_compiles', 'num_aot_compiles', 'num_backend_compilations')},
            full_graphs=full, piecewise_wrappers=piecewise)

    def install_graph_submission_observations(self):
        if hasattr(self, '_graph_submission_counts'):
            raise ValueError('graph submission observer already installed')
        runner = self.model_runner
        manager = runner.cudagraph_manager
        if manager is None: raise ValueError('no actual CUDA graph manager')
        counts, errors = Counter(), Counter()
        self._graph_submission_counts, self._graph_submission_errors = counts, errors
        def wrap(original, mode):
            @wraps(original)
            def observed(*args, **kwargs):
                # record_batch has already recorded CPU input geometry.
                batch = runner.step_timing._batch
                if mode == 'FULL':
                    padded = descriptor(args[0])
                    padded_tokens, padded_requests = padded['num_tokens'], padded['num_reqs']
                else:
                    from vllm.forward_context import get_forward_context
                    context = get_forward_context()
                    padded_tokens = context.batch_descriptor.num_tokens
                    padded_requests = None
                key = (mode, int(batch[1]), int(batch[2]), padded_tokens, padded_requests)
                try:
                    value = original(*args, **kwargs)
                except BaseException:
                    errors['submission_exception'] += 1
                    raise
                if key in counts or len(counts) < 1024: counts[key] += 1
                else: errors['capacity'] += 1
                return value
            return observed
        manager.run_fullgraph = wrap(manager.run_fullgraph, 'FULL')
        manager.run_pw_graph = wrap(manager.run_pw_graph, 'PIECEWISE')
        return self.get_graph_submission_observations()

    def get_graph_submission_observations(self):
        return dict(scope=__doc__, errors=dict(self._graph_submission_errors),
            rows=[dict(mode=key[0], actual_tokens=key[1], actual_requests=key[2],
                padded_tokens=key[3], padded_requests=key[4], successful_submissions=count)
                for key, count in self._graph_submission_counts.items()])
