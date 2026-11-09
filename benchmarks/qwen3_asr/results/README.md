# Canonical recognition reference

`native-worker-logits-initial.json.gz` preserves the initial failed native
Worker observation through an authenticated exported package. Four B1 requests
naturally stop; the next cohort fails full-row BF16 equality when its long
request is processed in different prefill chunks. The package and test binary are
identified separately. Raw BF16 frame/row files remain at the validation site;
this JSON retains completion metadata and the failing frame without claiming
completion of the B2/4/8 schedule.

`native-worker-logit-budgets-initial.json` applies the unchanged frozen scalar
and top-1 checks to every planned request, retaining missing and partial traces.
Three B1 teacher rows fail scalar bounds despite matching top-1 IDs. Subsequent
unexecuted requests remain failures/missing. Recognition quality passes do not
override these numerical and byte-stability failures.

`native-audio-boundaries.json` compares all nine encoder windows in the four B1
baselines against SDK BF16 Mel and retained prepared encoder features. It keeps
the original diagnostic's misnamed identity field and explicitly clarifies its
runtime-cache meaning. `native-audio-observer-b1-bridge.json` checks complete
natural B1 logits before/after extra audio reads: 40,718,848 bytes match exactly.
The extra reads affect multi-request scheduling; they are not a performance
instrumentation path or proof that every later cohort executes the same plans.

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

The [multipart MP3 follow-up](native-http-multipart-mp3.json) retains all 89
historical attempts from the byte-identical current HTTP collector
(`011a01beb194218a19e68455bb5518a2ab4ad8e9c32667b26e5126014a809d1a`).
Its new multipart MP3 case requires nonempty text, natural stop and an actual
token-count header; it passed with seven generated tokens. Overall 87/89 passed:
the public WAV fixture download failed with a network error, and the public MP3
server fetch returned HTTP 400. Both failures remain. This run used the earlier
`5a038ffab9e4846f8ec8fac7bd91e5853f468f82c92b576295290e07c035c67d` binary
and `f5dd799ab4e18f015955b5e25630140a26dcc26f2a7dee8e3beaf20e570e129d`
package file. It does not qualify the new norm package, all HTTP cases or performance.

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
checks are explicit. Encoder compilation and encoder CUDA Graphs were disabled
in this run; their availability and deployment tuning remain open. Decoder
graphs, compilation and batching are active.

Paired descriptive long scores versus the earlier native package are retained
per crop and in aggregate. Native/reference English WER is .323510/.326785,
English CER .241497/.243916 and Mandarin CER .149927/.150414. These nine crops
come from three recordings and overlap; frozen annotation gaps/overlap/order
limits remain. They do not establish the primary qualification margin, a
reliable P95 or all-language quality. The native side uses the previously
identified modified-worktree binary and unchanged package from
`native-prefix-reuse.json`, not the new projection candidate. Instrumentation
overhead, sustained performance and final clean-source qualification remain open.

`request-attention-sm121.json` retains independent actual-GQA and right-aligned
prefill operand observations, including the split/partition controls that reject
uniform or repeated-KV arithmetic. This is component diagnosis, not serving
performance. `request-attention-partition-extent.json` adds actual CUDA kernel
names and max-key/split controls to the same frozen operands. It identifies the
165-key split-KV difference and retains failures of the fixed 128-key partition
control at other lengths; it does not qualify a model configuration.
`shared-generation-errors-sm121.json` retains seven native fixture
attempts and the successful Rust HTTP/SSE metric/body observations. Six earlier
attempts fail during fixture construction/startup; none establishes native ERROR
handling. The final attempt exports real fault-producing attention computation,
exits Python, verifies the signed package and stored weights in Rust, checks
failed-state publication/prefix reuse/slot recovery and drains cleanly. Concurrent
HTTP arrival does not establish a shared B2 GPU graph. Full ASR qualification and
comparative performance remain separate gates.

`vllm-encoder-availability.json` records actual encoder compile/graph flags off
and on in the rebuilt pinned image, with two development inputs repeated five
times per configuration. Each run has ten natural stops and 70 actual tokens;
the off/on token arrays match. With both flags true, the model still fails the
official encoder-graph protocol check, its runner has no encoder graph manager,
and the audio tower has no official vLLM compile wrapper. A decoder wrapper is
present as a positive control. Startup succeeds, so this is a limitation of the
observed native model path, not a failed-startup claim or a universal claim about
other encoder implementations. This small trial has no official-counter or
performance calibration claim; deployment backend and partial-batch tuning
remain open. Both task containers exit normally and are removed.

`native-input-cache-tiers.json` retains four native cache-mode cohorts on an
unchanged earlier ASR package, with 184 successful raw responses and four explicit
bad-input refusals. The two different development utterances are padded to equal
sample counts for placeholder isolation; this is not the quality/performance
corpus. Each mode contains 15 serial and 30 concurrent responses plus recovery
after a refusal. Encoder-only and combined modes each apply 13 serial hits with
zero encoder submissions for those hits. Raw IDs, text, prompts and usage match
the off mode; the context control actually changes prompt IDs. CPU store lookup
hits and successfully applied uploads are separate counters. The recorded HTTP
concurrency does not establish GPU batch size. This modified native source and
earlier package do not qualify the new attention package, in-flight worker
cancellation, eviction/readmission, disk reuse, final source or performance.

`native-input-cache-batches.json` retains the subsequent portable collector run
on the same earlier ASR package and a new native binary. Its four modes again
complete 184 successful requests and four explicit refusals with unchanged raw
tokens, prompts and usage. Each encoded mode applies 13 serial hits. Successful
shared graph-submission counters now observe logical request counts 1 through 8
separately from physical capacities 1/2/4/8, including partial batches. Counter
indices name the loaded manifest; their mapping to the inspection catalog's graph
kinds is not asserted. These are submissions, not completions or GPU busy time.
The new counters supplement the earlier HTTP observations; this remains finite
functional qualification on modified source, with the same open lifecycle,
new-attention-package, final-source and performance gates.

`native-recognition-batched-attention.json` retains a subsequent fresh normal CLI
export with the batched Q1/GQA request-attention candidate. All 21 forwards compile
and all 33 graph variants capture; the preparer exits before the independent Rust
server verifies and loads the package. Two development inputs each repeat five
times exactly, and all 256 primary qualification requests stop naturally. Actual
token counts reconcile as 5,149 primary + 575 development = 5,724 official counter
increments. Every primary text and token count equals the retained reduction
package observation; token IDs are not observed by this transcription collector.
The English WER 70/2875, English CER 94/15411 and Mandarin CER 35/1861 pass the
unchanged finite-cohort margins against canonical and both retained vLLM records.
The native process drains with exit 0 and releases its listener. This package
uses frozen modified preparation source and a separately identified modified
native build. Complete network-node/operator gates, new-package concurrency and
lifecycle checks, final clean-source qualification and sustained performance
remain open. An earlier compiler-worker CUDA-context OOM is retained separately;
its absence in this attempt does not establish its root cause or a general fix.

`native-feature-upload-cancellation.json` retains actual Rust Worker test logs
on the earlier signed ASR package. The real encoder/readback seeds retention;
an actual mapped CUDA wait and pending copy event bound cancellation during
upload. Unmodified execution passes before and after deliberate omission of the
cancellation recheck and upload fence. Both mutants fail at the required state
or fence boundary. The original collector rejected the omitted-fence failure
because it expected only the earlier fence assertion; that failed attempt is
retained and the later predicate explicitly names the following prerequisite
check. These finite controls do not qualify HTTP disconnect, all cache
lifecycles, the new attention package, final source or performance.

The gzip JSON files `batched-attention-teacher-small-audio.json.gz` and
`batched-attention-teacher-large-audio.json.gz` retain the complete earlier
preparation-only teacher/state reports. Each contains twelve B1/2/4/8 schedule
cells and 240 all-position prefill/production-state bridges with the largest
prefill capacity, using a different audio-capacity selection. Their scalar and
meaningful-state gates pass; their full-model numerical flag remains false.
Compression preserves the original JSON bytes.

`network-node-observations-sm121.json.gz` retains the subsequent focused B1
report: all twenty frozen inputs pass 423 audio and 650 decoder node budgets,
audio/prefill production bridges and the saved teacher rows. It uses smaller
legal prefill capacities and the largest audio bucket. This is one schedule
cell; complete teacher/state and full-model qualification remain false. A comment
change alters the conservative all-source preparation hash from `72a5c9da…` to
`b646d970…` and the cached legal projection tactics differ. The complete CPU
comparison in `network-node-candidate-comparison.json` finds all 120 retained B1
files, 3,660 tensors and 1,434,607,040 bytes equal across those runs.

`network-head-node-budgets-sm121.json` applies both unchanged vocabulary-node
budgets to the same retained GPU observations. The actual BF16 `scratch.hidden`
inputs were already saved as the norm-output observation after each captured
position-head call; the collector does not recompute them. Full-position BF16
head results come from the corresponding saved logits. All twenty input/output
checks pass. Together these records cover the 1,075 named network stages per
input, while independent local arithmetic, the new configuration's complete
schedule/native lifecycles, final-source and performance gates remain open.

`reduction-column-tile-candidates-sm121.json.gz` retains a component tuning
negative result. Ten geometries cover the five required packed dimensions and
the actual individual decoder projections; logical B1/2/4/8 use eight physical
rows. All 100 tactic/geometry programs were compiled and captured before the
nine randomized timing blocks (3,600 raw samples). The 2,400 preceding relative
checks include actual captured-output bytes, finite values, guards, inactive
rows and allocation stability. Each timed sample checks allocation counters;
its output is not independently revalidated after that sample. Existing
`(4,4)` arithmetic is the byte reference, not an independent operator oracle.

For all forty operation/batch cells, the best wider-column candidate's median
is slower than the best existing candidate's median (about 2.1%–29.9%). No wider
tile is admitted to production. Layer weights rotate across all 28 original
layers; the vocabulary head has one large weight. The source checkpoint's
names, shapes, dtypes and finite values are checked, while this tool explicitly
does not establish its full content identity. These measurements are limited
to the recorded SM121 component conditions; they establish neither complete
ASR service performance nor a comparison against vLLM.

`prefill-prefix-projection-rejections.json.gz` retains all650 node observations per candidate,
head checks and scalar probes for six rejected existing BF16 projection
configurations on one frozen19-token input. No candidate passes its network-node
gates; this is a rejection subset of the unchanged20-input contract.

`original-prompt-attention-fragments.json` retains the six same-operands
automatic/fixed-family observations for full165 and actual128+37 fragments,
compiled capture repeat/allocation checks, future/active KV poison and recovery.
Fixed original partition geometry removes the 25,629-element discrepancy of
the automatic current-chunk path without extending the actual consumed KV bound.
This is one synthetic geometry, not a passing model.

`positioned-projection-fragments.json` retains all196 full/prefix/tail comparisons
on the independent frozen operands of one actual165-token input, all28 layers
and seven projections. Fixed8192 physical projection rows preserve bytes across
fragments and reproduce every independent same-operands output. The observation
includes actual pre-fix integer-overflow guard failures and the earlier collector
identity/reshape failure before any projection. Captured component repetitions
are not full-network, native request, quality or performance qualification.

`positioned-projection-native.json` records the complete29-test canonical and
positioned CUDA/native selection. All four new full/prefix/tail/empty component
packages replay five times in Rust after preparation exits; every replay first
poisons output. This does not validate N/position request uploads or cache
dependencies because the fixture captures those metadata values. The previous
25-pass/4-failure missing-worker-module attempt and private-wheel correction
remain explicit, with no whole-model or performance claim.


`original-prompt-teacher-state.json.gz` preserves the complete captured-network
run for the original-prompt/generated-history recipe. The accompanying
`original-prompt-teacher-state-run.json` records its source correspondence,
fixed references, inputs and capacities. All twelve B1/2/4/8 schedule cells
contain twenty cases: 1,783 required teacher rows pass the unchanged scalar
bounds with zero top-1 differences; all 240 same-path case traces, 240
full-prefill probes and 960 state probes pass. Thirty-eight case traces are
planned two-row retirement prefixes, not complete natural transcriptions.
Prefix and first-decode KV observations cover all 28 layers.

Budgeted network-node checks pass for 8,460 encoder, 13,000 prefill and 40 head
observations; all 27 encoder observation bridges pass. Another 1,120 activation
hook entries remain unbudgeted, so these are not complete local-operator
qualification. Teacher maximum absolute error is 0.125 and maximum relative L2
is 0.006978388081540694, within the unchanged per-case bounds. All recorded
floating-point values are finite.

This run retains the preparer, uses SDK BF16 Mel widened to FP32 and enables
inference mode. Its largest prepared text capacity is 512; context is 8192.
Ordinary CLI export uses a different ambient inference mode and requires its
own native checks. Between-replay retirement is not asynchronous cancellation.
The raw `full_model_numerical_qualification=false` and
`native_package_serving=false` remain intentional: this run does not qualify
native readmission, the PCM frontend, recognition quality, larger prefill
capacities, API/lifecycle/old-package behavior or serving performance.


`prompt-capacity-components.json.gz` preserves the complete experiment on fixed
operands, with source and geometry in `prompt-capacity-components-run.json`.
All 20 frozen inputs, 28 layers and seven layer projections contribute 3,920
observations and 6,664 fragments. Candidate prompt staging uses 256/512 rows
instead of 8192, while each captured fragment has 64 rows. Every candidate and
baseline matches its independent reference on the same BF16 operands; all
published bytes, five repeated replays, output guards and observed PyTorch
allocation counters pass. The collector's completion state enforces equality
between candidate and baseline; independent-reference differences are recorded
without changing its exit status. All 3,920 reference counts were checked
individually as zero for this result. Completion alone does not establish
canonical qualification. Prompt token IDs and checkpoint payloads are not
revalidated by this tool. It does not observe the LM head, full network,
frontend, API or native package replay.

Timing covers only the first 64-row fragment of one 165-token input, across all
196 layer projections: nine randomized paired blocks with ten replays per
sample. The per-cell intervals are descriptive within one session; independent
repeats and simultaneous coverage are not established. These are uncompiled
captured components, not exported-package serving measurements or a vLLM
comparison. The full-model, native-serving and service-performance qualification
flags remain false. The typed native C×M catalog is a separate implementation
and requires its own numerical, lifecycle, export and serving checks.


`native-rmsnorm-twenty-replay-run.json` records the new canonical RMSNorm
ordinary CLI package: 43 compiled forwards, 55 captured graphs and independent
Rust replay after preparation exits. The two complete scalar archives retain
60 generated-history and 190 batch/slot requests over all 20 frozen inputs.
All observed same-path KV planes, BF16 logits and tokens are exact; unchanged
per-case scalar budgets pass. Snapshot parking, eviction/readmission, B1/2/4/8,
row permutations, partial batches and slot reuse are covered. Test-only
readiness barriers do not qualify automatic HTTP batching.

The three recognition archives retain all 256 requests and both score results.
Every request naturally stops, all text matches canonical, and actual response
headers report 5,147 generated tokens; raw token IDs are not collected here.
Against the actual pinned vLLM0.29 image, English WER is 70/2,875 versus69,
English CER is 91/15,411 versus89, and Chinese CER is 35/1,861 for both. The
10,000-pair bootstrap uses the unchanged frozen seed and gates: WER upper
95% degradation is 0.0010183299389002 and Chinese CER upper is
0.0011074197120709. All required point and interval gates pass. Historical
package failures and measurements remain unchanged.

This is finite native state/scalar/recognition qualification. Complete new-norm
intermediate-node/operator, full CUDA/native/lifecycle, actual current HTTP/cache,
old VISION, strict full storage and sustained performance gates remain open.
Formal sustained comparative cells remain zero, with no vLLM performance claim.


`native-rmsnorm-current-api-run.json` retains all attempted current-package
HTTP, annotated long speech and finite concurrency clients. The HTTP raw has
88 checks:86 pass, while both immutable public fixture prefetches fail with
network-unreachable errors. Their two dependent model URL requests are unrun;
the full extended plan would contain90 checks. An earlier guard's HTTP89 label
came from a historical run and is not this attempt's count.

All nine fixed annotated long inputs remain included: eight naturally stop,
while `L_R003S01C02-540s` contains repeated filler and a service child reaches
its1024-token output cap (2483 aggregate tokens). The long suite fails; neither
its input nor output limit is changed. The separate144-request B1/2/4/8
text/count schedule passes, including active-and-queued parent disconnect,
drain and eight subsequent requests. It does not observe raw token/slot/fence
IDs or an unrelated short request during cancellation. Full ownership metrics
drain at every suite boundary; the native process exits normally. These finite
results do not qualify complete API/lifecycle/model behavior or performance.


`native-rmsnorm-long-child-diagnostic.json.gz` retains all19 actual service-child
responses derived from the unchanged frozen long PCM. Eighteen naturally stop;
child10 (468800 samples,29.3 seconds) repeats499 filler phrases and reaches the
unchanged1024-token limit. The parent failure remains in the long denominator.
Native decoded PCM is exact. Original-SDK and native FP32 Mel differ in26
values (maximum2.384185791015625e-7), while all375040 BF16 model inputs match.
Both independently repeated SDK prompts naturally stop at94 tokens: original
SDK/chat has399 prompt tokens; the matching empty-context transcription prompt
has394. Their actual prompt bytes/IDs, token IDs/text and repeat observations
are retained separately. Matching394 renderer bytes/base IDs were checked on
the native CPU path; expansion is derived, pending actual runtime observation.
These controls localize the regression. They do not establish that it is fixed,
that the parent HTTP lifecycle is equivalent, or that any performance gate passed.


`native-rmsnorm-long-sdk399.json.gz` then observes the same failing long child
through the existing native worker's original-SDK399 whole-prompt path. The
four diagnostic inputs complete11 cohorts/39 natural requests, including the
long child's exact94 reference tokens and byte-exact batch/slot logit repeats.
This explicitly different diagnostic input set does not replace the frozen
four/twenty-input acceptance sets. It does not exercise transcription394
message partitioning or establish a fix for the actual HTTP regression.


`native-rmsnorm-current-http-retry-run.json` and its raw retry archive retain
the unchanged extended HTTP client after public transport recovered. All90
checks pass, including the original immutable public HTTPS WAV/MP3 model
requests. The package and native runtime identities match the preceding attempt;
full ownership metrics drain before/after, and the process exits normally with
its listener/GPU released. The earlier86/88 network failure stays retained.
Long and finite concurrency clients were not repeated in this retry. The
394-token transcription long-child regression remains open; this HTTP coverage
does not establish its repair, complete lifecycle/model qualification or a
sustained performance comparison. Formal sustained comparative cells remain0.
