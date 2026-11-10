# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

from asr_vllm_worker_observations import WorkerObservations

class EncoderTrialWorker(WorkerObservations):
    def encoder_execution_configuration(self):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
        from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
        state = self.model_runner.model_state
        model, runner = state.model, state.encoder_runner
        manager = runner.cudagraph_manager
        cc = state.vllm_config.compilation_config
        def modules(root):
            rows = []
            for name, module in root.named_modules():
                wrapped = isinstance(module, TorchCompileWithNoGuardsWrapper)
                row = dict(name=name, module=type(module).__module__, class_name=type(module).__qualname__,
                    vllm_compile_wrapper=wrapped)
                if wrapped:
                    d = module.__dict__
                    row.update(do_not_compile=d.get('do_not_compile'), compiled=d.get('compiled'),
                        is_encoder=d.get('_is_encoder'), compile_prefix=d.get('_compile_prefix'),
                        first_compile=d.get('first_compile'), aot_function_present=d.get('aot_compiled_fn') is not None,
                        aot_loaded_from_disk=d.get('was_aot_compile_fn_loaded_from_disk'),
                        compiled_bytecode_present=d.get('_compiled_bytecode') is not None)
                rows.append(row)
            return rows
        return dict(scope='CPU metadata only at cohort boundaries; no model method or tensor changed',
            configuration=dict(compile_mm_encoder=cc.compile_mm_encoder,
                cudagraph_mm_encoder=cc.cudagraph_mm_encoder, mode=cc.mode.name, backend=cc.backend),
            graphs=dict(protocol=supports_encoder_cudagraph(model), runner_has_cudagraph=runner.has_cudagraph(),
                manager_present=manager is not None, captured=None if manager is None else manager.is_captured(),
                supports_audio=None if manager is None else manager.supports_modality('audio'),
                stats=None if manager is None else manager.get_cumulative_stats()),
            audio_modules=modules(model.audio_tower), decoder_modules=modules(model.language_model))
