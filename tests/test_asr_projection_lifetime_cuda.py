# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Raw diagnostic preparation keeps first-prime owners after failed fences."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


def test_raw_projection_first_prime_retains_native_call_and_program_lease():
    script = r'''
import gc,sys,weakref
import torch
from b12x.gemm import bf16_gemv
from b12x.preparation import PreparationSession
from b12x._lib.compile_plan import retain_compiled_programs
import b12x.preparation.session as preparation
from ignition.preparation.failure import retain_on_failed_drain
from ignition.preparation.module_images import ModuleImages
from qualify.asr_projection_candidates import retained_call
torch.set_num_threads(2)
real_sync=torch.cuda.synchronize
real_prime=preparation._prime
refs={}
def failed_sync(*args,**kwargs):raise RuntimeError("injected raw projection fence")
def failed_prime(call):
    real_prime(call)
    call.run()
    torch.cuda.synchronize=failed_sync
    raise RuntimeError("injected raw projection first prime")
preparation._prime=failed_prime
def prepare():
    owners=[]
    with retain_on_failed_drain(0,owners), retain_compiled_programs() as programs, ModuleImages() as images:
        owners.append(images)
        refs['programs']=weakref.ref(programs)
        if sys.argv[1]=='enabled':owners.append(programs)
        torch.cuda.set_device(0)
        x=torch.ones((8,2048),device='cuda',dtype=torch.bfloat16)
        w=torch.full((4096,2048),.5,device='cuda',dtype=torch.bfloat16)
        y=torch.zeros((8,4096),device='cuda',dtype=torch.bfloat16)
        owners.extend((x,w,y))
        session=PreparationSession(device=torch.device('cuda:0'),autotune=False,compile_workers=1)
        owners.append(session)
        query=bf16_gemv.query_from_call(x,w,out=y)
        plan=bf16_gemv.plan(query,override=bf16_gemv.GemvConfig(backend='simt',rows_per_tile=8))
        owners.append(plan)
        def bind(state):
            refs['state']=weakref.ref(state)
            call=retained_call(state,x,[w],y,owners)
            refs['call']=weakref.ref(call)
            return call
        session.prepare((plan.request(name='raw-lifetime',prepare_call=bind),),autotune=False)
try:prepare()
except RuntimeError as error:
    assert 'injected raw projection first prime' in str(error),str(error)
    error.__traceback__=None;error.__cause__=None;error.__context__=None;error.__dict__.clear()
else:raise AssertionError('raw first-prime failure was not injected')
real_sync()
gc.collect()
assert set(refs)=={'state','call','programs'}
assert refs['state']() is not None and refs['call']() is not None,'raw state/call lost after failed fence'
assert refs['programs']() is not None,'raw compiled-program lease lost after failed fence'
assert refs['programs']().owners,'raw compiled-program lease contains no compiled owners'
print('raw state/call/program owners survived exception disposal',flush=True)
'''
    root = Path(__file__).resolve().parents[1]
    for control in ("enabled", "disabled"):
        result = subprocess.run([sys.executable, "-c", script, control], cwd=root,
            capture_output=True, text=True, timeout=900)
        if control == "enabled":
            assert result.returncode == 0, result.stdout + result.stderr
            assert "owners survived exception disposal" in result.stdout
        else:
            assert result.returncode != 0
            assert "raw compiled-program lease lost after failed fence" in result.stdout + result.stderr
