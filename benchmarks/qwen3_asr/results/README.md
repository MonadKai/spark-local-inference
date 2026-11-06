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
