# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fault the actual ASR decoder's private KV in already prepared graphs.

This is a synchronous captured-model lifecycle diagnostic. It does not replace
independent arithmetic/recognition gates or establish native HTTP cancellation.
"""

from contextlib import contextmanager
import json


def check(prepared, path):
    import torch
    from ignition.components.sampling import ERROR, RequestAdmission
    from ignition.preparation.failure import drain_or_retain, retain_on_failed_drain
    from ignition.recipes.qwen3_asr.audio_generation import PreparedAudioGeneration

    catalog, = (value for value in prepared.resources if isinstance(value, dict)
        and isinstance(value.get("audio100"), PreparedAudioGeneration))
    exemplar = catalog["prefill64"]
    requests, uploads, pool = exemplar.batch.requests, exemplar.batch.inputs, exemplar.decoder.pool
    # Preparation reserves more continuation rows than it maps physically.
    # A tensor's virtual shape must never be used as its readable extent.
    backed_rows = int(pool.memory_capacity[0].item())
    hidden = pool.last_hidden[:backed_rows]
    assert backed_rows >= requests.status.numel() + 1
    features = exemplar.decoder.feature_state
    buffers = prepared.buffers
    record = dict(scope=__doc__.strip(), prepared_identity=prepared.server_config["identity"],
        status="running", cases=[])
    with path.open("x") as file:
        file.write(json.dumps(record, indent=2) + "\n")

    def publish():
        path.write_text(json.dumps(record, indent=2) + "\n")

    @contextmanager
    def report_failure(row):
        try:
            yield
        except BaseException as error:
            row["error"] = dict(type=type(error).__name__, message=str(error))
            record["status"] = "failed_or_interrupted"
            try:
                publish()
            except BaseException as reporting_error:
                error.add_note(f"could not save ASR fault diagnostic: {reporting_error!r}")
            raise

    def same_bytes(a, b):
        assert a.dtype == b.dtype and a.shape == b.shape
        return torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
            b.contiguous().reshape(-1).view(torch.uint8))

    def replay(name):
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        prepared.graphs[name].replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        for field in ("allocation.all.allocated", "allocation.all.freed",
                "segment.all.allocated", "segment.all.freed"):
            assert before[field] == after[field], field

    fields = ("consumed_tokens", "generated_tokens", "rng_counter", "histogram",
        "pending_token", "status", "error_code")
    epoch = 1000
    for capacity in (1, 2, 4, 8):
        for reverse, partial in ((False, False), (True, False), (True, True)):
            for sample in (False, True):
                epoch += 1
                live = max(1, capacity - 1) if partial else capacity
                slots = list(range(live)) if not reverse else list(range(7, 7 - live, -1))
                row = dict(capacity=capacity, slots=slots, sample=sample, passed=False)
                record["cases"].append(row)
                publish()
                owners = [prepared, requests, uploads, pool]
                with report_failure(row), retain_on_failed_drain(0, owners):
                    pool.clear_for_preparation()
                    requests.status.zero_()
                    uploads.request_state_rows.zero_()
                    uploads.page_tables.fill_(-1)
                    features.indices.fill_(-1)
                    buffers["frame_index"].zero_()
                    buffers["submission_sequence"].fill_(epoch)
                    for page, slot in enumerate(slots):
                        prompt = [1000 + page] * 32
                        requests.admit(slot, RequestAdmission(epoch, 0, max_new_tokens=16,
                            temperature=1., top_k=1, top_p=1.), prompt)
                        uploads.page_tables[slot, 0] = page
                        uploads.request_state_rows[slot, 0] = slot + 1
                        uploads.live_rows.fill_(1)
                        uploads.row_slot_ids.fill_(-1)
                        uploads.row_slot_ids[0] = slot
                        uploads.row_generations.fill_(epoch)
                        uploads.query_lengths.fill_(32)
                        prepared.buffers["original_prompt_tokens"].zero_()
                        prepared.buffers["original_prompt_tokens"][:1].fill_(len(prompt))
                        uploads.token_ids.zero_()
                        uploads.token_ids[:32].fill_(1000 + page)
                        uploads.sample_enabled.zero_()
                        replay("prefill64")
                        assert requests.error_code[slot] == 0 and requests.consumed_tokens[slot] == 32
                        requests.pending_token[slot] = 1000 + page
                    uploads.live_rows.fill_(live)
                    uploads.row_slot_ids.fill_(-1)
                    uploads.row_slot_ids[:live].copy_(torch.tensor(slots, dtype=torch.int32))
                    uploads.query_lengths.fill_(1)
                    uploads.sample_enabled.fill_(int(sample))
                    before = {name: getattr(requests, name).clone() for name in fields}
                    saved_hidden = hidden.clone()
                    owners.extend((*before.values(), saved_hidden))
                    graph = f"decode{capacity}"
                    replay(graph)
                    assert requests.error_code[slots].eq(0).all()
                    control = {name: getattr(requests, name).cpu().clone() for name in fields}
                    control_hidden = hidden.cpu().clone()
                    control_logits = catalog[graph].decoder.scratch.logits.cpu().clone()

                    def restore_request_state():
                        for name, value in before.items():
                            getattr(requests, name).copy_(value)
                        hidden.copy_(saved_hidden)

                    restore_request_state()
                    # Only the first packed request's past value is poisoned;
                    # this is not the position rewritten by the next KV write.
                    _, value = pool.attention(0)
                    physical_page = int(uploads.page_tables[slots[0], 0].item())
                    original = value[physical_page, 0, 0, 0].clone()
                    owners.append(original)
                    try:
                        value[physical_page, 0, 0, 0] = float("nan")
                        replay(graph)
                        decoder = catalog[graph].decoder
                        assert decoder.execution_errors.tolist() == [1] + [0] * (capacity - 1)
                        failed = slots[0]
                        assert requests.status[failed] == ERROR and requests.error_code[failed] == 1
                        for name in fields[:-2]:
                            assert same_bytes(getattr(requests, name)[failed], before[name][failed]), name
                        assert same_bytes(hidden[failed + 1], saved_hidden[failed + 1])
                        completion = buffers["completion_records"][0, failed]
                        assert completion[0] == epoch and completion[1] == epoch
                        assert completion[2] == before["consumed_tokens"][failed]
                        assert completion[3] == before["generated_tokens"][failed]
                        assert completion[5] == ERROR and completion[6] == 1
                        assert completion[4] == -1 and completion[8] == 0 and completion[16:].eq(-1).all()
                        for packed, slot in enumerate(slots[1:], 1):
                            for name in fields:
                                assert same_bytes(getattr(requests, name)[slot].cpu(), control[name][slot]), name
                            assert same_bytes(hidden[slot + 1].cpu(), control_hidden[slot + 1])
                            assert same_bytes(decoder.scratch.logits[packed].cpu(), control_logits[packed])
                    except BaseException as error:
                        if drain_or_retain(error, 0, owners):
                            value[physical_page, 0, 0, 0].copy_(original)
                            torch.cuda.synchronize()
                        raise
                    else:
                        value[physical_page, 0, 0, 0].copy_(original)
                        torch.cuda.synchronize()
                    restore_request_state()
                    replay(graph)
                    assert catalog[graph].decoder.execution_errors.eq(0).all()
                    for name in fields:
                        assert same_bytes(getattr(requests, name).cpu(), control[name]), name
                    assert same_bytes(hidden.cpu(), control_hidden)
                    row.update(passed=True, failed_slot=slots[0], failed_physical_page=physical_page,
                        physical_decoder_rows=8, healthy_rows=live - 1,
                        physically_backed_state_rows=backed_rows,
                        healthy_rows_exact=True, failed_state_preserved=True, recovery_exact=True)
                    publish()
    assert len(record["cases"]) == 24 and all(row["passed"] for row in record["cases"])
    record["status"] = "complete_captured_model_error_diagnostic_only"
    publish()
