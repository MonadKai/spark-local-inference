# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Transport evidence controls; these do not execute or qualify an ASR model."""

from pathlib import Path
import subprocess
import sys

import pytest


CONTROL = r'''
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util, json, os, signal, sys, threading, time
import httpx

spec = importlib.util.spec_from_file_location('collector', sys.argv[1])
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
fault = sys.argv[2]
request_started = threading.Event()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_POST(self):
        value = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        request_started.set()
        time.sleep(.02 * (3 - value['id']))
        body = json.dumps(dict(transport_only=value['id'])).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

class FaultPool(ThreadPoolExecutor):
    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)
        if fault != 'none':
            assert request_started.wait(3), 'HTTP request never started'
        if fault == 'signal': os.kill(os.getpid(), signal.SIGINT)
        elif fault == 'submit': raise RuntimeError('planted failure after enqueue')
        return future

collector.ThreadPoolExecutor = FaultPool
signal.signal(signal.SIGINT, collector.interrupted)
server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
thread = threading.Thread(target=server.serve_forever)
thread.start()
cohort = dict(status='running')
published = []
error = None
try:
    with httpx.Client(base_url='http://127.0.0.1:' + str(server.server_port), timeout=3) as client:
        try:
            collector.collect_cohort(client, [dict(id=i) for i in range(3)],
                [2, 0, 1], cohort, lambda: published.append(json.loads(json.dumps(cohort))))
        except BaseException as caught:
            error = type(caught).__name__
    assert error == (None if fault == 'none' else 'KeyboardInterrupt' if fault == 'signal' else 'RuntimeError')
    issued = 3 if fault == 'none' else 1
    assert len(cohort['responses']) == 3
    for row in cohort['responses'][:issued]:
        assert row.get('attempted') and row.get('http_status') == 200 and row.get('body') == dict(transport_only=row['index']), 'issued response was lost'
    assert all(row['attempted'] is False for row in cohort['responses'][issued:])
    assert published[-1]['responses'] == cohort['responses']
    assert len(cohort['submission_errors']) == int(fault == 'submit')
finally:
    server.shutdown()
    thread.join(timeout=3)
    server.server_close()
assert not thread.is_alive()
print('all issued HTTP outcomes retained; transport control only')
'''


@pytest.mark.parametrize('fault', ['none', 'signal', 'submit'])
def test_concurrent_transport_results_survive_interruption_and_submit_failure(fault):
    helper = Path(__file__).resolve().parents[1] / 'qualify/asr_native_input_cache.py'
    result = subprocess.run([sys.executable, '-c', CONTROL, str(helper), fault],
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_interruption_control_detects_a_missing_completion_channel(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'qualify/asr_native_input_cache.py'
    mutated = tmp_path / 'collector.py'
    text = source.read_text()
    needle = 'completed.put_nowait((ordinal, value))'
    assert text.count(needle) == 1
    mutated.write_text(text.replace(needle, 'pass  # planted lost completion', 1))
    result = subprocess.run([sys.executable, '-c', CONTROL, str(mutated), 'signal'],
        capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'issued response was lost' in result.stderr
