# Canonical recognition reference

`canonical-recognition.json` contains the raw pinned Transformers reference
outputs for the unchanged [qualification corpus](../qualification.json): 256
utterances and 261 service chunks, all naturally stopped within the 1024-token
chunk cap. It records actual generated IDs, prompt IDs, EOS outcomes and raw
text. It contains no candidate recognition or performance result.

`canonical-transcriptions.json` replays those IDs through the actual pinned
vLLM incremental detokenizer, transcription cleanup and language-dependent
joiner. Empty chunks are preserved. Whole-tokenizer decode is retained only as
a diagnostic: it can replace an incomplete UTF-8 sequence while the real
incremental decoder retains it without emitting text.

The inputs were prepared by `qualify/asr_recognition_inputs.py` in the isolated
vLLM 0.29.0 environment, including its real decoder, splitter, prompt expansion
and registered processor. Its configured model-dtype boundary is BF16. The
canonical environment runs `qualify/asr_canonical_recognition.py` with Python
isolation and the unmodified upstream original-format weight converter. The
existing original-layout convolution pre-hook handles the one 24-frame tail.
`qualify/asr_recognition_clean.py` performs the separate CPU text replay.

These are reference hypotheses for the frozen paired scoring protocol, not an
ASR model acceptance or an all-language quality claim. Candidate numerical,
recognition, lifecycle and native-serving gates remain separate.

`projection-candidates-sm121.json` preserves every result from
`qualify/asr_projection_candidates.py`: 104 applicable plans, four inapplicable
configurations, 1124 observations and 420 decode timing cells, including all
rejected candidates and raw event samples. The five actual ASR matrix geometries
include the first-prefill-token LM-head input. Decode timing rotates all 28
decoder-layer weight sets; the head has one weight set. B1/2/4/8 logical and
fixed-eight physical rows are both measured. Prefill has numerical observations
only, with no timing or performance claim.

The complete eight represented rows are compared across shapes, row offsets and
poisoned unrelated rows. The conditional FP64 error envelope covers only six
output channels of layer zero. Weight rotation is a timing condition, not
all-layer arithmetic qualification. In particular, MMA/TC/prefill arithmetic has
not been proven to satisfy the frozen RNE update model. These short repeated
CUDA-graph event timings are diagnostic kernel measurements; they do not
establish an admissible production configuration, a full ASR throughput gain or
a vLLM serving comparison.

The historical projection result's `checkpoint_revision` names the expected
source. This diagnostic checks represented tensor geometry/precision/finite
values, not the supplied checkpoint's remote content identity. The original
snapshot was checked separately; newer runs label that field
`expected_checkpoint_revision` explicitly.

`vllm-recognition.json` contains the pinned vLLM 0.29.0 reference for the same
256 requests and 261 naturally stopped chunks. The non-streaming HTTP server
retained its compiled/CUDA-graph configuration. The observer in
`qualify/asr_vllm_recognition_entry.py` delegates engine calls unchanged and
records actual prompt IDs, generated IDs, text and stop reasons; its file IO
excludes this run from performance comparisons.

The second complete run records the actual multipart fields, uploaded WAV/PCM
identities and selected sampling fields at `AsyncLLM.generate` entry. Later
InputProcessor cloning and generation-config normalization were not observed.
A separate CPU
validator checks these against the frozen corpus, including prompt IDs, actual
generated IDs, EOS, cleanup/joining and original HTTP response text. All 256
responses and 261 chunks pass. Cross-request cache keys were not observed;
processor/prefix cache configuration does not itself establish GPU cache state.

`vllm-recognition-initial.json` preserves the first complete run. Its original
client compared completion order with input order and flagged five long
requests. The validator uses the pinned API's engine request ordinals and checks
the original response text. All 256 responses pass; the five original client
statuses/errors and their order-check reinterpretations remain explicit. This
older run did not observe the actual multipart fields or upload/sampling data
required from the second run.

Fifteen result controls accept valid reversed completion order and reject
fourteen deliberate corruptions: missing/duplicate chunks, wrong prompt IDs,
length stop, missing EOS, wrong join, HTTP failure, malformed JSON, missing or
malformed chunk identity/data, delta output, wrong seed, wrong form and wrong
PCM. The affected request remains in the 256-request denominator. These are
reference-result controls, not candidate quality or HTTP failure-recovery
qualification.

`native-recognition-initial.json` preserves the first complete Rust-package
qualification attempt: 256 requests, including 21 HTTP 500/device-error-72
failures. All failures remain in the scoring denominator. The ASR recipe
declared 256-token pages but omitted that argument to its maintenance graphs,
which captured the shared 128-token default. Development long audio reproduced
the failure three times with successful short-audio controls. Native component
tests additionally reproduce silent partial-page loss at endpoint 128 and the
reported error at 129 under the wrong binding.

`native-recognition.json` records a fresh export after correcting the binding.
The preparer exited before a separate Rust `flare run` process loaded the signed
package. All 256 qualification requests naturally stopped. Three repetitions
each of development short/long controls also stopped before qualification. The native
client checks uploaded PCM, actual forms, HTTP status/MIME, duration usage,
generated-token-count headers and terminal reasons. It does not observe native
generated token IDs. The per-run configuration identifies the binary/package,
dirty integration source and limited cache observations; this is not final
clean-source qualification or a performance measurement.

The frozen paired scoring passes against both canonical and vLLM references.
Native English WER is 0.024, English CER 0.006034650574265135 and Mandarin CER
0.01934443847393874. All point gates and both primary bootstrap upper bounds
pass. Reproduce `quality_observation` with `qualify.asr_quality.evaluate`, the
unchanged `../qualification.json`, the candidate's `cases`, and respectively
`canonical-transcriptions.json` or `vllm-recognition.json`'s `cases`.
The earlier failed attempt stays separate. These results do not establish
full scalar numerical acceptance, concurrent determinism, cache/lifecycle
acceptance, other API transports, long meeting-speech quality or a throughput
advantage over vLLM.

`native-concurrency.json` retains a separate finite development schedule:
eight fixed mixed-language inputs, client concurrency 1/2/4/8, forward/reverse
submission order and two repetitions, followed by eight concurrent requests
after a parent disconnect. All 144 responses naturally stopped and matched
their B1 response bodies, usage and actual token-count headers. The disconnect
occurred with eight active and one pending child; the observed cancelled count
then increased by nine, and active/pending/retiring gauges became zero. The
native process subsequently drained on SIGTERM and exited zero.

The client requires an increasing cancellation count and empty ownership gauges;
it does not independently require a nine-child delta in every future run.
The raw metrics establish that specific delta here. Client concurrency and
submission order do not establish actual GPU batch rows or physical slot IDs.
Native token IDs and teacher logits were not observed. The unchanged earlier
package and dirty binary exclude this run from final candidate qualification.
The failed initial attempt is retained with all eight successful baseline
responses and its incorrect JSON-metric-key error. Neither attempt is a
performance measurement or a complete lifecycle/cache qualification.

`native-recognition-repeat.json` reloads the identical repaired package and
binary after review tightened the collector: PCM validation, WAV hashing and
multipart upload now use one immutable byte object. The prior collector opened
the file separately; no mutation was observed, and that observation limit is
retained in the earlier results. The repeat completes all 256 requests with the
same response bodies and actual token-count headers as the repaired run. Both
frozen reference quality gates pass again. These are sequential B1 observations
across separate server processes, not concurrent determinism or equality of
unobserved token IDs.

`native-http-initial.json` and `native-http.json` retain two 69-case black-box
HTTP runs against the same repaired recognition package and native binary.
The initial run has one incorrect client expectation: an empty request without
a multipart boundary returns JSON 400, whereas the client expected 415. The
corrected assertion passes on an unchanged server/package; both attempts remain.
These observations cover model discovery, multipart and inline/data-URL audio,
JSON output for both transcription format names, duration, short sample counts,
signed seeds, actual output caps and chat alias precedence, explicit refusals,
and recovery to the same transcription bytes/counts after invalid requests.

Run `qualify/asr_http.py --base-url URL --wav WAV --output RESULT` against a
separately started package server, using the frozen
`en-61-70968-0058` qualification WAV. Each attempt is saved before sending and
retains the response status, headers and body. Cap diagnostics require the
uncapped fixture to generate more tokens and check actual counts/length stops.
The launcher, rather than this client, checks the recorded binary/package.
The observed source HEAD is distinct from that binary's earlier dirty build.

This is a finite native subset check, not complete vLLM API parity or evidence
for the new projection candidate. Public HTTPS/MP3 success, system-context
handling, exact encoded/body/duration/context limits, long-parent cancellation,
concurrent slot reuse and internal prompt/token IDs remain separate checks.
Repeated content and cache hits here are functional observations, not cold
performance workloads or cache-isolation acceptance.

The three `native-http-boundaries-*.json` files retain every attempt of the
extended client, including failures. Reproduce with `qualify/asr_http.py
--extended --public-revision b8bf56e72accd8e9538a1c02a4595be0c6932ccd`,
the public WAV fixture and a separately started native server. Configurations
identify the actual binary and unchanged earlier graph package; these are WIP
builds, not final clean-source model qualification.

The initial 87 attempts include two unavailable public fixture downloads and
an incorrectly constructed large whitespace-context positive. Whitespace is
preserved by the reference renderer, so that request properly exceeded context.
The corrected 89-attempt workload uses removable control markers for the large
multipart field. It exercises exact 25 MiB encoded WAV, 32 MiB multipart/JSON
body, and 9,600,000 decoded-sample limits with adjacent refusals. The 600-second
silence case caps each child at one token; it is a duration control, never speech
quality, complete transcription throughput or a substitute for real long audio.
Transport cases cover multipart WAV, inline/data-URL WAV/MP3 and fixed public
HTTPS WAV/MP3; different codecs retain different waveforms.
Counts include two client-side immutable-file checks; only successful downloads
enable the two corresponding server-side public audio-URL requests.

Before the body fix, 88/89 attempts pass: a one-byte over-limit chunked epilogue
arriving after the multipart terminal boundary incorrectly returns HTTP 200.
After the fix, that request returns 413, as do the ordinary excess-body controls,
and their exact-limit positives pass. The latter run still fails 1/89 attempts
because the public MP3 fetch returns HTTP 400; that failure remains in the result.
It is not an all-pass HTTP qualification. Controlled Pending-stream Rust tests
separately cover EOF, true size rejection, late transport errors and cancellation
with no child queueing and complete host-credit return. The one-second HTTP tail
delay is a finite observation, not a proof of every ordering.

`legacy-qwen38-native.json` records a real previously published text package
loaded by the original and candidate Rust binaries. All 70 published payloads
were independently checked at the recorded immutable revision; both binaries
verify the publisher signatures. The package was not re-exported or re-signed.
Each process serves the same three requests twice: English, Mandarin and a
10,025-token prompt spanning multiple 4096-token prefills. All twelve requests
naturally stop. Returned prompt/generated token IDs, full messages, terminal
reasons and token usage match within each process and between the binaries.
Both actual native PIDs drain and exit with status zero; neither is forced down.

The file contains exact request bodies and raw response bodies, plus binary and
package identities. Use the publisher profile revision recorded separately in
the configuration. Run each binary against the unchanged package with
`--verify-weights --max-concurrency 1 --mtp-draft-tokens 0 --kv-cache-limit concurrency
--memory-limit 48GiB --embedding-table-storage ram --prefix-cache-memory 0
--prefix-cache-disk-limit 0`, submitting the recorded request order twice.
The legal package sampling defaults remain unchanged; seed zero does not turn
that distribution into greedy sampling. Ignore response IDs/timestamps when
comparing the actual token arrays and logical usage counts.

This finite compatibility result uses earlier binaries, not a final clean-source
build. It does not qualify every old model, vision, MTP, concurrent old-model
execution, complete old-model numerical behavior or serving performance.

The official [Qwen3.8-Flash-Next GB10 package](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-FLARE/tree/cd5b82260c6922d629b0d97d66d16bc4658ed2c2)
has four real VISION graphs in its range-read metadata, with 256/1024/4096/16384
feature capacities, SM121 and one rank. Its 106,886,249,924-byte payload was not
fully acquired, signature-verified or loaded. The metadata declares envelope
version 4; both the unchanged historical source and this branch require version
6 in their archive/protection readers. This static inspection identifies an
existing format limitation, not an executed load failure or a new regression.
The metadata signer's match to the fixed public publisher profile does not
verify the missing payload. The smaller 27B package above has no VISION graphs;
its successful text replay cannot qualify old VISION loading or execution.

`native-prefix-reuse.json` records ten serial native ASR requests through two
separately started Rust processes loading the unchanged earlier audio package.
Two different PCM contents have the same 39-token prompt extent. For A/A/B/A/B,
shared reuse reports cached counts 0/39/5/39/39 and two AudioEncode submissions;
request reuse reports all-zero cached counts and five submissions. All responses
stop naturally with matching text/counts/usage, and both processes drain cleanly.
The build is explicitly a modified worktree, not final clean-source qualification.
This finite control does not establish host-cache combinations, CPU image stores,
asynchronous preemption, long-child isolation or performance. Native prompt and
generated token IDs were not observed; the SDK prompt supplies only the stated
first-audio-token bound.

`long-speech-observer.json` retains nine frozen long requests against pinned
vLLM with decode/prefill CUDA Graph capacities through 4096. The actual bounded
observer records 95 natural terminal chunks/6,170 tokens, agreeing with official
counters; all 95 unique audio identifiers associate with successful encoder
execution, with no pending work or unknown cache observations. The reference
API container exits 0 and its task container is removed after the complete
cohort has drained. The default abort/zero-timeout shutdown force-kills the
EngineCore worker; this is not a graceful worker-teardown check. Its image has no
repository digest; the actual local image ID and installed selected-source
checks are explicit. Encoder CUDA Graphs remain unavailable for this model in
the pinned implementation; decoder graphs, compilation and batching are active.

Paired descriptive long scores versus the earlier native package are retained
per crop and in aggregate. Native/reference English WER is .323510/.326785,
English CER .241497/.243916 and Mandarin CER .149927/.150414. These nine crops
come from three recordings and overlap; frozen annotation gaps/overlap/order
limits remain. They do not establish the primary qualification margin, a
reliable P95 or all-language quality. The native side uses the previously
identified modified-worktree binary and unchanged package from
`native-prefix-reuse.json`, not the new projection candidate. Instrumentation
overhead, sustained performance and final clean-source qualification remain open.
