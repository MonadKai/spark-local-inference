# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualify independent native input-cache tiers on a supplied exported ASR package.

Run on an exclusively assigned GPU after preparation and compilation finish.
This finite functional cohort does not establish performance, quality, in-flight
worker cancellation, disk retention or cache eviction/readmission. HTTP
concurrency is recorded separately from actual GPU batch qualification.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import base64, copy, hashlib, io, json, os, queue, re, signal, socket, subprocess, time, wave
import httpx

SOURCE = Path(__file__).resolve().parents[1]

def now(): return datetime.now(timezone.utc).isoformat()
cleanup = False
created_output = False
signals = []
def interrupted(number, frame):
    signals.append(number)
    if not cleanup: raise KeyboardInterrupt('signal ' + str(number))

def call(arguments, **kwargs):
    return subprocess.run(arguments, check=True, **kwargs)

def isolated():
    value = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'], text=True)
    if value.strip(): raise RuntimeError('GPU is not isolated')
    return value

def padded(row, samples):
    path = PCM_ROOT / row['path']
    with wave.open(str(path), 'rb') as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) == (1, 2, 16000, row['samples'])
        pcm = stream.readframes(stream.getnframes())
    assert hashlib.sha256(pcm).hexdigest() == row['pcm_s16le_sha256']
    assert samples >= row['samples']
    output = io.BytesIO()
    with wave.open(output, 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
        stream.writeframes(pcm + bytes((samples - row['samples']) * 2))
    return output.getvalue()

def signature(body):
    choice = body['choices'][0]
    tokens = choice['token_ids']
    assert choice['finish_reason'] == 'stop' and tokens
    assert len(tokens) == body['usage']['completion_tokens']
    return dict(prompt_token_ids=body['prompt_token_ids'], token_ids=tokens,
        content=choice['message']['content'], usage=body['usage'])

def metrics(client):
    response = client.get('/metrics?format=json'); response.raise_for_status()
    value = response.json()
    text = client.get('/metrics'); text.raise_for_status()
    match = re.search(r'^flare_graph_launches_total\{kind="audio_encode"\} (\d+)$', text.text, re.M)
    if not match: raise ValueError('missing real encoder submission counter')
    value['observed_encoder_submissions'] = int(match[1])
    catalog = re.search(r'^flare_graph_batch_catalog_entries (\d+)$', text.text, re.M)
    value['observed_graph_catalog_entries'] = None if catalog is None else int(catalog[1])
    value['observed_graph_batches'] = [dict(graph_index=int(index), capacity=int(capacity),
        requests=int(requests), submissions=int(count)) for index, capacity, requests, count in re.findall(
        r'^flare_graph_batch_submissions_total\{graph_index="(\d+)",capacity="(\d+)",requests="(\d+)"\} (\d+)$', text.text, re.M)]
    return value

def settled(client):
    deadline = time.monotonic() + 20
    while True:
        value = metrics(client)
        if value['input_cache_queued'] == value['input_cache_pending_writes'] == 0: return value
        if time.monotonic() > deadline: raise TimeoutError('host cache did not settle')
        time.sleep(.01)

def collect_cohort(client, requests, indices, cohort, publish):
    """Retain issued HTTP outcomes even if submission or the parent is interrupted."""
    global cleanup
    if not 1 <= len(indices) <= 8: raise ValueError('bounded native cache cohort required')
    completed = queue.Queue(maxsize=len(indices))
    cohort['responses'] = [dict(index=index, attempted=False, status='not_attempted') for index in indices]
    cohort['submission_errors'] = []
    publish()
    def submit(ordinal, index):
        value = dict(index=index, attempted=True)
        try:
            response = client.post('/v1/chat/completions', json=requests[index])
            value.update(http_status=response.status_code, response_text=response.text)
            if response.status_code == 200: value['body'] = response.json()
        except BaseException as error:
            value['error'] = dict(type=type(error).__name__, message=str(error))
        finally:
            # This bounded completion channel also covers submit() failing
            # after enqueue, before returning its Future to the caller.
            completed.put_nowait((ordinal, value))
        return value
    pool = ThreadPoolExecutor(max_workers=len(indices))
    futures = []
    try:
        for ordinal, index in enumerate(indices):
            cleanup = True
            try:
                future = pool.submit(submit, ordinal, index)
                futures.append(future)
            except BaseException as error:
                cohort['submission_errors'].append(dict(ordinal=ordinal,
                    type=type(error).__name__, message=str(error)))
                cohort['responses'][ordinal].update(attempted=None, status='submission_outcome_uncertain')
                raise
            finally:
                cleanup = False
            if signals: raise KeyboardInterrupt('interrupted after registering an issued request')
        for ordinal, future in enumerate(futures):
            cohort['responses'][ordinal] = future.result()
            publish()
    finally:
        cleanup = True
        try:
            pool.shutdown(wait=True, cancel_futures=True)
            seen = set()
            while True:
                try: ordinal, value = completed.get_nowait()
                except queue.Empty: break
                if ordinal in seen: raise RuntimeError('duplicate HTTP completion')
                seen.add(ordinal)
                cohort['responses'][ordinal] = value
            publish()
        finally:
            cleanup = False
    if signals: raise KeyboardInterrupt('cohort interrupted; all issued attempts retained')


def main():
    global cleanup, created_output
    OUTPUT.mkdir()
    created_output = True
    path = OUTPUT / 'result.json'
    if path.exists(): raise FileExistsError(path)
    package = args.package.resolve()
    with package.open('rb') as stream: package_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    with BINARY.open('rb') as stream: binary_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    binary_identity = dict(sha256=binary_sha256, source_binding='supplied binary; source/package release qualification is separate')
    development = json.loads(args.manifest.read_text())
    if development['split'] != 'development': raise ValueError('independent development manifest required')
    rows = [next(row for row in development['utterances'] if row['language'] == lang) for lang in ('en', 'zh')]
    length = max(row['samples'] for row in rows)
    audio = [padded(row, length) for row in rows]
    assert audio[0] != audio[1]
    requests = []
    for index, content in enumerate(audio):
        body = dict(model='Qwen/Qwen3-ASR-1.7B', messages=[dict(role='user', content=[dict(type='input_audio',
            input_audio=dict(data=base64.b64encode(content).decode(), format='wav'))])],
            temperature=0, seed=0, max_completion_tokens=1024, return_token_ids=True)
        requests.append(body)
        (OUTPUT / ('equal-placeholder-' + str(index) + '.wav')).write_bytes(content)
    control = copy.deepcopy(requests[0]); control['messages'].insert(0, dict(role='system', content='Vocabulary context: café, local speech.'))
    requests.append(control)
    record = dict(scope=__doc__, started_at=now(), binary=binary_identity, package=str(package),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        package_primary_sha256=package_sha256, samples_per_input=length,
        padded_development_inputs=[row['id'] for row in rows], status='starting', modes=[], errors=[],
        planned_modes=['off', 'preprocessed', 'encoded', 'both'], performance_qualified=False,
        new_attention_candidate_qualified=False)
    def save():
        temporary = path.with_suffix('.pending')
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(path)
    save()
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert gpu and '\n' not in gpu
    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = gpu
    publisher_path = PUBLISHER
    publisher = json.loads(publisher_path.read_text())
    assert len(publisher) == 1 and publisher[0]['namespace'] == 'local-development' and publisher[0]['development']
    fingerprint = hashlib.sha256(bytes(publisher[0]['public_key'])).hexdigest()
    inspection = call([str(BINARY), 'inspect', str(package), '--publisher-keys', str(publisher_path),
        '--all', '--json'], capture_output=True, text=True)
    view = json.loads(inspection.stdout)
    if not view['package']['signatures_verified'] or len(view['ranks']) != 1:
        raise ValueError('one verified rank required')
    record['inspection_graph_catalog'] = view['ranks'][0]['execution_graphs']
    record['graph_index_mapping_verified'] = False
    record['graph_index_mapping_scope'] = 'inspection may include shapes filtered during native loading; counter capacities/requests are direct observations but no graph-kind mapping is asserted'
    save()
    baselines = {}
    server = None
    try:
        for mode in record['planned_modes']:
            isolated()
            cleanup = False
            out = OUTPUT / mode; out.mkdir()
            cell = dict(mode=mode, prefix_cache_reuse='request', requests=[], cohorts=[], errors=[], status='starting')
            record['modes'].append(cell); save()
            argv = [str(BINARY), 'run', str(package), '--config', str(CONFIG),
                '--publisher-keys', str(publisher_path), '--trust-store', str(out / 'trust.toml'),
                '--dangerously-approve-key', fingerprint, '--dangerously-accept-publisher', 'local-development',
                '--verify-weights', '--offline', '--host', '127.0.0.1', '--port', str(PORT), '--gpus', '0',
                '--max-concurrency', '8', '--embedding-table-storage', 'ram', '--kv-cache-limit', 'concurrency',
                '--prefix-cache-memory', '64M', '--prefix-cache-disk-limit', '0', '--prefix-cache-reuse', 'request',
                '--prepared-input-cache', mode]
            cell['argv'] = argv; save()
            try:
                with socket.socket() as probe: probe.bind(('127.0.0.1', PORT))
                with (out / 'server.log').open('x') as log:
                    server = subprocess.Popen(argv, cwd=SOURCE, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    cell['pid'] = server.pid; save()
                    with httpx.Client(base_url=f'http://127.0.0.1:{PORT}', timeout=300) as client:
                        deadline = time.monotonic() + 300
                        while True:
                            if server.poll() is not None: raise RuntimeError('native server exited before readiness')
                            try:
                                if client.get('/health', timeout=2).status_code == 200: break
                            except httpx.TransportError: pass
                            if time.monotonic() > deadline: raise TimeoutError('native readiness')
                            time.sleep(.25)
                        maps = Path(f'/proc/{server.pid}/maps').read_text().lower()
                        assert 'libtorch' not in maps and 'libpython' not in maps
                        cell['python_torch_mappings_absent'] = True
                        for repeat in range(5):
                            for index, body in enumerate(requests):
                                before = settled(client)
                                request = dict(index=index, repeat=repeat, before=before, attempted=True)
                                cell['requests'].append(request); save()
                                response = client.post('/v1/chat/completions', json=body)
                                request.update(http_status=response.status_code, body=response.text)
                                save(); response.raise_for_status()
                                value = signature(response.json()); request['observed'] = value
                                after = settled(client); request['after'] = after
                                encoder_delta = after['observed_encoder_submissions'] - before['observed_encoder_submissions']
                                hit_delta = after['encoded_input_hits'] - before['encoded_input_hits']
                                request.update(encoder_delta=encoder_delta, encoded_hit_delta=hit_delta)
                                payload_hits = after['input_cache_payload_hits_total'] - before['input_cache_payload_hits_total']
                                request['input_store_payload_hit_delta'] = payload_hits
                                if mode == 'off':
                                    if index in baselines: assert baselines[index] == value
                                    baselines[index] = value
                                else: assert baselines[index] == value, 'cache tier changed raw tokens, prompt or usage'
                                eligible = repeat > 0 or index == 2
                                request['encoded_skip_delta'] = after['encoded_input_skipped'] - before['encoded_input_skipped']
                                assert hit_delta in (0, 1)
                                if hit_delta:
                                    assert mode in ('encoded', 'both') and eligible and encoder_delta == 0, 'applied hit did not bypass the whole encoder'
                                else:
                                    assert encoder_delta > 0, 'neither an encoded upload nor real encoding supplied the prerequisite'
                                # Async lookups may miss their normal execution opportunity.
                                # Store hits and applied uploads are separately observable.
                                if mode in ('preprocessed', 'both') and eligible:
                                    assert payload_hits >= hit_delta + 1, 'frontend tier did not perform its independent cache hit'
                                if mode == 'encoded':
                                    assert hit_delta <= payload_hits <= 1, 'encoded-only mode reused an extra frontend record'
                                if mode == 'off': assert payload_hits == 0
                                request['passed'] = True; save()
                        if mode in ('encoded', 'both'):
                            cell['applied_hits_by_input'] = {str(index): sum(row['encoded_hit_delta'] for row in cell['requests'] if row['index'] == index) for index in range(len(requests))}
                            assert all(count > 0 for count in cell['applied_hits_by_input'].values()), 'no actual applied hit within five fixed repetitions'
                        assert baselines[2]['prompt_token_ids'] != baselines[0]['prompt_token_ids'], 'context control did not change the actual prompt'
                        assert baselines[0]['prompt_token_ids'] == baselines[1]['prompt_token_ids']
                        assert baselines[0]['token_ids'] != baselines[1]['token_ids'], 'equal-placeholder control lacks distinct audio results'
                        for batch in (1, 2, 4, 8):
                            for reverse in (False, True):
                                indices = [index % 2 for index in range(batch)]
                                if reverse: indices.reverse()
                                cohort = dict(concurrency=batch, actual_gpu_batch_qualified=False,
                                    reversed=reverse, indices=indices, responses=[], status='running',
                                    metrics_before=settled(client))
                                cell['cohorts'].append(cohort); save()
                                collect_cohort(client, requests, indices, cohort, save)
                                for response in cohort['responses']:
                                    assert response.get('http_status') == 200 and 'error' not in response, 'concurrent response failed; every sibling attempt retained'
                                    assert signature(response['body']) == baselines[response['index']]
                                cohort['metrics_after'] = settled(client)
                                cohort['status'] = 'passed'; save()
                        malformed = copy.deepcopy(requests[0])
                        malformed['messages'][-1]['content'][0]['input_audio']['data'] = '!'
                        rejected = client.post('/v1/chat/completions', json=malformed)
                        cell['invalid_input'] = dict(status=rejected.status_code, body=rejected.text)
                        assert rejected.status_code == 400
                        recovered = client.post('/v1/chat/completions', json=requests[0]); recovered.raise_for_status()
                        cell['after_error_body'] = recovered.json()
                        assert signature(recovered.json()) == baselines[0]
                        cell.update(status='complete_functional_cells_only', final_metrics=settled(client))
            finally:
                cleanup = True
                if server is not None:
                    if server.poll() is None:
                        server.send_signal(signal.SIGTERM)
                        try: server.wait(timeout=90)
                        except subprocess.TimeoutExpired:
                            server.kill(); server.wait(timeout=20); cell['forced_stop'] = True
                    cell.update(server_exit_code=server.returncode, server_reaped=server.poll() is not None)
                    server = None
                cell['gpu_after_exit'] = isolated()
                with socket.socket() as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); probe.bind(('127.0.0.1', PORT))
                cell['listener_released'] = True
                save()
            assert cell.get('server_exit_code') == 0 and not cell.get('forced_stop')
            if signals: raise InterruptedError('deferred cleanup signals')
        record['status'] = 'complete_retained_package_cache_controls_only'
    except BaseException as error:
        record['status'] = 'failed_or_interrupted'
        record['errors'].append(dict(type=type(error).__name__, message=str(error)))
        save(); raise
    finally:
        record['finished_at'] = now(); record['signals'] = signals; save()
    print(record['status'], flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--publisher-keys', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=SOURCE / 'benchmarks/qwen3_asr/development.json')
    parser.add_argument('--pcm-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18474)
    args = parser.parse_args()
    BINARY, OUTPUT, PORT = args.binary.resolve(), args.output.resolve(), args.port
    PCM_ROOT = args.pcm_root.resolve()
    CONFIG, PUBLISHER = args.config.resolve(), args.publisher_keys.resolve()
    for number in (signal.SIGINT, signal.SIGTERM): signal.signal(number, interrupted)
    try:
        main()
    except BaseException as error:
        if created_output:
            path = OUTPUT / 'result.json'
            value = json.loads(path.read_text()) if path.exists() else dict(scope=__doc__, errors=[])
            if value.get('status') != 'failed_or_interrupted':
                value['status'] = 'failed_or_interrupted'
                value['errors'].append(dict(stage='collector_preflight_or_outer', type=type(error).__name__, message=str(error)))
            value['finished_at'] = now()
            value['signals'] = signals
            temporary = path.with_suffix('.pending')
            temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
            temporary.replace(path)
        raise
