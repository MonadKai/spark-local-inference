# Qwen3-ASR qualification inputs

`qualification.json` fixes 128 English and 128 Mandarin utterances, with 16
speakers per language, before any recognizer execution. It records the actual
labels, sample counts, canonical PCM identities, source acquisition scope and
primary requests. Audio files are not redistributed here.

Acquire official LibriSpeech test-clean and AISHELL-1 test audio from the linked
publishers. The Chinese annotation file actually used is pinned separately in
the configuration. Its equality to the transcript inside the complete official
outer archive has not been established. Selected complete official speaker
archives passed their inner gzip CRC checks; the complete outer AISHELL archive
was not acquired or checksum-verified. These limits remain part of this input
identity. The English archives passed their publisher full-object MD5 checks.

`python -m qualify.asr_corpus` fully decodes every labeled candidate before
ranking, then writes canonical mono 16 kHz PCM16 WAVs and a manifest. Supply
`--source-details` with the actual acquisition/check descriptions; publisher
expected checksums alone are not verification. Match the resulting IDs, labels,
sample counts and PCM identities against the fixed configuration. Reordering
files cannot change quantile selection, duplicate candidates fail, and a
missing reference is declared before selection. Do not replace samples after
recognizer results.

Scoring and admission margins are fixed by issue #1 and `qualify/asr_quality.py`:
English WER and Mandarin CER absolute degradation at most 0.002, English CER
degradation at most 0.001, and a one-sided 95% paired-bootstrap upper degradation
bound at most 0.005 for each primary metric (10,000 resamples, seed 20261106).
Failed or truncated requests remain in the result set and invalidate admission.
A separate development split is used for tuning. These finite-set gates do not
establish quality in every language.

[The current native recognition result](results/native-recognition-current.json)
retains all 256 responses and both complete paired quality comparisons for a
modified-worktree arithmetic-baseline export served by Rust after Python exited.
Its frozen-set quality gates pass. Earlier larger-text-prefill numerical failures
remain recorded separately; this result establishes neither final-source support
nor comparative performance. Actual generated counts come from server headers; raw token IDs
were not observed in that run.

Independent CPU reference tools in `qualify/asr_*_reference.py` check original
SDK/canonical tensor loading, slow tokenizer behavior and actual vLLM frontend
semantics. Their supplied model/source content identities must be established
separately where stated; successful loading or frontend comparison is not a
model forward, exported-package, HTTP, GPU, quality or performance result.

`long-speech-sources.json` describes supplemental real meeting speech. Its
acquisition and annotation-boundary qualification is separate from the primary
256 utterances. Missing safe boundaries must be recorded as acquisition failures,
without inventing a cut through speech.

`long-speech.json` freezes the resulting nine continuous meeting crops, original
annotations, exact sample boundaries and PCM identities before ASR execution.
Run `python -m qualify.asr_long_corpus --config
benchmarks/qwen3_asr/long-speech-sources.json --source-dir INPUTS --output OUTPUT`
with the acquired sources. All recordings are fully decoded; AISHELL-4 uses
channel 0 of eight, not a channel average. RTTM is only a rounded cross-check of
the original TextGrid. Three ES2002a acquisition failures remain explicit. AMI
annotation gaps and Chinese partial/invalid/overlap markers remain visible;
deterministic speaker ordering is not a word-interleaving claim.
`regression_inputs` contains a separately labelled repeated-content 600-second
stress input, excluded from real long-speech quality and unique-content throughput.

The native collector also accepts `--suite long-speech --manifest
benchmarks/qwen3_asr/long-speech.json --pcm-root INPUTS --base-url URL --output
RESULT`. It requires the exact frozen nine-request manifest and verifies each
uploaded PCM before submission. Its result explicitly separates supplemental
long speech from the 256-request primary qualification. Natural stop and actual
generated-token headers are checked for every parent; failures stay in the
nine-request denominator. Observation IO excludes these runs from timing claims.

For the pinned vLLM image, `qualify/asr_vllm_performance_entry.py` records bounded
actual terminal tokens and frontend media identifiers in memory. Pass the pinned
source directory with `--vllm-source`, a new `--observations` file and an explicit
`--observation-capacity`. Keep the normal non-streaming transcription endpoint;
verbose JSON changes logprob work and is not a token-count workaround.

The optional `asr_vllm_worker_observations.WorkerObservations` official worker
extension exposes `install_asr_encoder_observations` and
`get_asr_encoder_observations` through vLLM's official collective RPC. Install
after startup, snapshot after all HTTP work drains, and use a bounded capacity.
The pinned vLLM dev RPC route requires its documented dev-mode setting; that
reference-only configuration does not introduce a Flare environment switch.
The entry explicitly adds its own module directory for spawned workers.
`validate_cohort` requires the externally planned complete chunk-ID set;
`validate_encoder_cohort` reconciles actual successful dispatches with frontend
identifiers. Unknown KV reuse, reused identifiers in cold mode, missing chunks,
non-natural stops, outstanding dispatches and every engine/collector failure
invalidate the cohort. Recompute counts within a request are retained.

These observers do not establish timing eligibility. Calibrate their combined
overhead, retain all HTTP attempts including failures before engine admission,
reconcile official generated-token/terminal counters and record actual cache and
compile conditions before using a cohort for performance. Successful dispatch
is not a GPU busy timer. `results/long-speech-observer.json` is a finite actual
integration/long-quality observation with performance qualification false.

`python -m qualify.asr_native_observation` submits eight frozen development
inputs under sequential, reversed, concurrent and partially occupied HTTP
schedules, then checks raw prompt IDs, raw generated IDs, EOS, text and usage.
Every generation/refusal attempt is saved before submission, and every response
before comparison, including failed siblings.
The ordinary collector controls deliberately exercise transport, response,
comparison, EOS, prompt and refusal failures; they do not substitute for model execution.

[The raw schedule observation](results/native-chat-schedules.json) retains
83 generation responses and four expected refusals. The [run configuration](results/native-chat-schedules-run.json)
binds the actual modified-worktree binary and exported package. Of 871 actual
generated tokens, 788 IDs were observed in 75 responses; eight observation-disabled
controls account for the remaining tokens. HTTP concurrency does not prove the
actual GPU batch or request slot. Worker timestamps include CPU completion and
delivery, and cannot establish pure GPU TTFT or word timestamps. This finite
automatic-language, empty-context diagnostic is separate from forced-language
quality, cancellation, cache, capacity and performance qualification.

`python -m qualify.asr_sdk_native_prompts` runs the original SDK prompt builder
and processor in an isolated CPU environment. [Its comparison](results/native-chat-sdk-prompts.json)
matches the same eight observed native prompt-ID sequences and reconstructed
UTF-8 expansions. It checks installed Python bytes against the supplied pinned
SDK source; that source's revision identity is an external precondition, not
proved by this helper. It loads no model weights and executes no model forward.
Decoded IDs reconstruct renderer bytes; they are not a direct observation of
the native renderer's allocated string.

[Fixed-reduction component observations](results/reduction-projection-components.json)
retain all six declared tuning candidates. Each candidate matches all 784 frozen
decoder operators, four cached-head observations and 1,582 mixed-row observations
byte-for-byte, with five replays per fixed case. The eight selected inputs have
only seven distinct layer-zero QKV vectors; this is retained rather than changing
samples. Every candidate's prefill-first head differs from its frozen reference,
and its prototype head timing regresses. The existing head path is retained.

The reusable mechanism declares one b12x tuning catalog, compiles before capture,
and computes eight physical rows in one kernel using fixed FP32 accumulation
chains and one terminal BF16 conversion. Its normal-input precision checks reuse
the frozen independent FP64 interval envelope on fixed output channels; separate
controls classify subnormal, overflow and nonfinite values. Guard corruption,
wrong BF16 output, invalid indexing/count/alignment and alias controls must fail.
The changed paths pass 18 focused CUDA/native tests and 27 CPU controls; three new
and five re-exported prior-path fixtures each replay independently in Rust five
times after preparation exits. The prior-path fixtures are not historical-package
compatibility evidence. Actual compiler targets, registers, spills and SM121
observations are retained in the raw measurements.

The timing records describe the earlier prototype before its final inactive-row
safety mask, under a fixed eight-row physical capacity. They guide candidate
selection; the prepared component itself and complete ASR service need separate
latency measurements. Neither these component observations nor unchanged head
selection establish complete-model admission, native ASR, recognition quality,
cache/lifecycle completion or a vLLM serving-performance win.

`python -m qualify.asr_workload` runs one complete frozen short, long or mixed
HTTP cohort against an independently managed server. It verifies and loads PCM
before sampling, keeps bounded response/timestamp records in memory, then writes
all attempts after drain or interruption. Short cells consume the entire 8,169
request list; repeated-content cells cycle the fixed 256 development inputs.
Mixed short latency includes waiting for its four client credits, while long
requests retain their independent frozen arrival schedule. Abbreviated
`--diagnostic-limit` runs are explicitly ineligible for formal measurement.

Ten CPU/real local HTTP controls cover concurrency, failed siblings, truncation,
interruption, PCM identity, mixed admission and latency accounting. They do not
run an ASR model. The launcher still supplies warmup, actual engine/package/image
identity, resource isolation, cache observations and observer-overhead calibration.
For vLLM, HTTP success leaves tokens and complete-ASR throughput unknown until
the independent engine observer reconciles every planned chunk and official
counter. The client always leaves overall performance qualification false;
the frozen block comparison and all model/lifecycle gates remain separate.

Build `test_audio` with `audio,runtime-package,test-tools` and use
`test_audio plan PACKAGE PUBLISHERS_JSON TP INPUTS_JSON OUTPUT_JSONL ACCEPTED_NAMESPACE`
to obtain native CPU frontend plans. The input manifest is an array of
`id`, encoded audio `path`, `samples` and `pcm_f32le_sha256` records. The tool
authenticates the package catalog and the unique requested deployment, checks
the accepted publisher namespace, and consumes its actual typed audio settings
and capacity. It verifies real decoded FP32 PCM, splits service chunks, then
prepares each whole chunk's Log-Mel once before enumerating encoder windows.
Service children and encoder windows therefore have separate counts.

JSON schema errors reject the entire manifest before execution. For structurally
valid rows, semantic identity, decoding and preparation failures retain their
ordinal and do not suppress independent following rows. Read every JSONL row
and require the final zero-failure summary; a partial file is not a complete
plan. Each chunk's Mel storage drops before the next chunk is prepared. This
CPU tool does not load executable payloads, validate the full execution contract,
initialize CUDA or establish actual encoder/model execution. Its native
regressions cover both declared padding modes, non-hop tails, silence split
ties, distinct window counts, authentication rejection and failed-row retention.

`python -m qualify.asr_vllm_phases` collects one run from an independently
managed pinned reference server with the encoder observer already installed.
It loads and verifies all required PCM before warmup, binds the pinned processor
chunk plans to the frozen audio, runs the complete development split followed
by at least120 additional active request seconds, then consumes the complete
timed workload. Unique-content runs call all three official cache reset routes;
their responses are retained, and an HTTP200 from the processor or encoder
route alone does not establish successful cache clearing. Counter and encoder
snapshots occur outside request timing. The launcher must still verify server
identity, cache conditions, stable captured execution, resource isolation and
collector overhead, and shut down normally to publish terminal observations.

`python -m qualify.asr_flare_phases` runs the same complete warmup and timed
workloads through native transcription, using each input set's JSONL native
plan. Supply the authenticated served envelope identity, server PID and binary
SHA256. It checks the live executable, start ticks and argv before and after
phases. RAM delivery is explicit; cold runs also require explicit zero host
prefix budgets, Request prefix scope and disabled prepared-input retention.
The launcher still binds that process to the HTTP listener and actual loaded
package, establishes isolation and records a normal shutdown.

Raw JSON/Prometheus/JSON snapshots retain every attempt. Required fields,
cumulative counters, histogram components, graph batches and uptime must remain
consistent across observations; the warmup-to-timed boundary permits no observed
counter work. Drain covers HTTP/frontend reservations, retiring model slots,
host transfers, input retention and, in the cold regime, final KV reclamation.
Complete child and actual header token totals are reconciled separately from
the native encoder-window plan. Same-request readmission can increase prefix
reuse and encoder execution: aggregate reuse is retained, and extra windows
require observed preemption while their individual request attribution remains
unknown. These observations cannot establish cache equivalence across engines.
Unavailable raw recognition IDs, model TTFT, per-request queue time, stage GPU
timings and physical compute padding remain null with reasons. Reconciliation
does not qualify quality, observer overhead or the frozen performance comparison.

`qualify.asr_performance_run.reconcile_vllm` checks those client phases against
every planned terminal chunk, actual token IDs, encoder dispatches and official
counter deltas. Failed warmup, truncated or missing chunks, and invalid input,
arrival or token identity prevent complete-audio credit. Reconciliation leaves
overall performance qualification false; it does not establish quality,
independent blocks or the final throughput and latency criteria.

`qualify.asr_performance_statistics.paired_interval` calculates the frozen
five-block log-ratio Student t interval from already qualified scalar measurements.
It uses 80-digit decimal arithmetic and rounds binary interval endpoints outward;
strict throughput and latency boundaries must not gain a passing result from
conversion error. Its CPU regressions include exact geometric equality, one
unusually fast block and a latency violation hidden by nearest binary rounding.
The helper does not establish independence, complete transcription, cache
equivalence or any model/service gate.

[The decoder-reduction native observation](results/native-recognition-reduction.json)
retains a fresh exported package served independently by Rust after Python exited:
256/256 primary requests stop naturally; two development inputs each match
across five repetitions. All fixed quality margins pass against canonical and both the historical
and rebuilt vLLM references. English WER is 70/2875, English CER 94/15411 and
Mandarin CER 35/1861. This uses modified-worktree model wiring, with actual terminal
counts and no raw generated IDs; it is not final clean-source qualification.

[The rebuilt vLLM observation](results/vllm-recognition-rebuilt.json) binds a new
actual image and manifest digest to 256 primary plus nine long requests. All
356 chunks stop naturally; 11,318 actual tokens reconcile with official counters,
and every unique media identifier is associated with successful encoder dispatch.
Prefill/decode compile and graphs are enabled with capacities through 4096;
encoder compilation and encoder CUDA Graphs were disabled in this run; their
availability and deployment tuning remain open. Raw
responses, actual token IDs, counters and two historical primary output
differences remain. The image and expanded graph configuration both differ from
the historical run; no single cause is assigned. Sampling after engine cloning
and actual prompt IDs were not recorded in this run. API-container normal exit
is reported separately from any worker-shutdown inference.

These are finite recognition/reference observations with uncalibrated collector
overhead. Complete state and full-prefill numerical gates, native batch/slot and
cancel/cache qualification, API/old-package gates, observer calibration and
the frozen sustained comparisons remain outstanding. Neither file establishes
a serving-performance gain or a vLLM win.

## Prompt query geometry

[Raw query-layout observations](results/prompt-query-layout.json) compare frozen
synthetic Q/K/V under the original, rounded and merged physical query layouts.
The geometry catalog uses the actual paged key capacity and pinned FA2 SM
heuristic; each fragment still reads only its consumed keys. Reproduce with:

```sh
python -m qualify.asr_query_layout --config CONFIG.toml --output NEW_DIRECTORY
```

All 81 observed rows match full SDPA bytes and their same-geometry full result.
The collector checks five identical captured replays, allocation counters,
unread future KV poison and active-poison recovery. These direct ATen
observations explain reduction geometry; they do not measure native serving,
qualify the complete model or establish a performance gain.


The native original-prompt measurements use an ordinary installed CLI export
and separate Rust replay after the Python preparer exits. The
[run configuration](results/native-original-prompt-run.json) preserves inputs,
source/package identities, the two preceding failed attempts and unqualified
scopes. [Generated-history raw observations](results/native-original-prompt-g2.json.gz)
and [their per-row scalar results](results/native-original-prompt-g2-scalar.json.gz)
record 12 natural requests, 129 frozen teacher comparisons and 402 observed
natural logit rows. Eight retained/evicted replays reproduce all 448 consumed KV
planes. The KV comparator detects the injected future-range poison; real prefill
then rewrites it without changing earlier committed bytes. Two independent Rust
runs also reproduce the four B1 traces.

[Batch/slot raw observations](results/native-original-prompt-batch-slot.json.gz)
and [their scalar results](results/native-original-prompt-batch-slot-scalar.json.gz)
record the original 11 cohorts/39 requests, 414 teacher comparisons and 1,321
observed natural rows. Full traces are byte-identical to B1, with zero frozen
teacher top-1 gaps. These JSON files publish all natural token IDs, frame and
completion records, comparison results and per-teacher-row metrics. Complete
BF16, KV and poison files remain locally retained; their relative filenames in
the JSON do not embed the tensor vectors in these gzip files.
Actual frames include full B1/2/4/8 and partial B8. Seven/eight-request cohorts
use a test-only readiness barrier and legal production plans to obtain capacity
coverage; this is a controlled arithmetic/row-placement check, not evidence
that automatic production scheduling forms B8. All runs confirm native drain.

The preceding snapshot-fixture and ordinary-scheduler B8-coverage failures
remain raw failed results. No numerical budget, reference, request order or
natural token was changed. Complete-model/operator, all-input, asynchronous
cancellation, API/quality, old-package and sustained performance qualification
remain separate; these observations establish no vLLM performance win.

[The original-prompt native recognition result](results/native-recognition-original-prompt.json)
retains all256 frozen primary responses from the same ordinary CLI package in
an independent Rust server. All requests stop naturally; actual terminal counts
sum to5,147 tokens, with no raw generated IDs observed by this client. All256
raw texts equal the fixed canonical reference. English WER70/2875, English
CER91/15411 and Mandarin CER35/1861 satisfy all three unchanged point margins.
English WER and Mandarin CER also pass the paired-bootstrap margins; English
CER has no bootstrap interval in the frozen scoring contract. The historical vLLM reference also passes; its result does
not transfer GPU qualification to a newly rebuilt image. This primary-set
recognition result does not establish complete numerical/operator, real-long,
API/lifecycle, old-package or sustained-performance qualification.


`runtime_package::test_support::programs` reads launch geometry through the
existing authenticated package readers. Its explicit ignored CPU collector
also verifies the existing signed prepared-weight catalog against the same
envelope. Four ordinary protobuf/geometry/DAG rejection controls and one real
signed-package CPU read pass; the complete native CPU suite passes1,391 tests,
with32 ignored. This does not load CUDA or establish historical-model serving.
Bare eight-byte values are labelled untyped address candidates. Matching them
to named storage does not prove pointer ABI or operator semantics. Current
native trace copy-byte discrepancies remain unresolved; kernel-sequence matching
does not establish complete node coverage or calibrate profiler overhead.

[Logical-live-row tuning measurements](results/projection-logical-workload-selection.json.gz)
compare legacy count8 and B1/2/4/8 preparation with the same six legal arithmetic
candidates and physical8 rows. [The run configuration](results/projection-logical-workload-run.json)
preserves the complete27-test CUDA/native gate, prior fixture failure, and the
[zero-cell reference-directory failure](results/projection-logical-workload-launch-failure.json).
All28 rotated layer weights reuse the same frozen layer0 activation rows;
every actual captured per-layer output has independent guards and five exact
replays. All compilation/tuning/capture finishes before nine randomized timing
blocks. These are descriptive within-session component intervals. Down improves
about2.4–2.9%, most intervals include1 and some gate/gate-up cells regress.
No overall benefit, full-model selection or serving-performance win is established.
Packed projections and the reduction head remain diagnostics; the production
BF16 LM head is unchanged.

[The current rebuilt vLLM image result](results/vllm-recognition-rebuilt-553.json)
binds image553/manifestfb9789ff to actual GPU execution:256 primary and nine long
parents,356 naturally stopped chunks and11,318 actual tokens reconcile with
engine counters and encoder dispatch. The actual backend is FlashAttention;
compile mode3 and full/piecewise CUDA Graphs are enabled. Encoder compile/graphs
remain disabled and their availability was not tested in this run. Flare also
passes the unchanged primary quality margins against this new reference.
Actual prompt IDs and sampling after engine cloning were not captured here;
API parity, observer overhead and reasonable deployment tuning remain separate.

[The CPU media-plan configuration](results/vllm-cpu-chunks-run.json) binds actual
pinned splitting results for8,169 performance parents/8,176 chunks,256 development
parents/258 chunks and nine long parents/95 chunks. Its three raw files retain
every input identity and chunk sample count. These precomputed expected sets
support later observer reconciliation; they are not ASR or performance results.


[Current-image deployment exploration](results/vllm-development-attention-configurations.json.gz)
retains four fixed-order configurations: FlashAttention/FlashInfer and default/
additional partial-batch graph capacities. [The run configuration](results/vllm-development-attention-run.json)
records eight fixed development inputs, real arguments and296 complete requests.
All terminal and encoder identities/counts reconcile. FlashInfer actually selects
TRTLLM XQA with16-token pages; both backends retain compile and CUDA Graphs.
The observations use the published reference-worker helpers. All task containers
exit normally and are removed. These short, startup-separated, uncalibrated
trials establish functional availability. Their fixed order and three cohorts
per principal batch do not establish a statistical ranking or formal ASR throughput.
Encoder compile/graph availability, observer overhead and sustained cells remain
separate; the default backend is not presented as a qualified global optimum.
The run configuration links all four raw terminal ledgers, including actual
token IDs and encoder media identities: each configuration naturally emits
1,358 tokens across74 requests. Counts come from engine terminal output.

[The positive prompt-capacity package observation](results/native-prompt-capacity-run.json)
uses clean committed model source, ordinary CLI preparation, 43 compiled forwards
and 55 captured graphs. After the preparer exited, Rust independently loaded the
package: 12 G2 and 39 batch/slot pilot requests passed the unchanged teacher
budgets; actual graph rows were constrained by test-only barriers. The fixed 256
recognition requests stopped naturally, with headers reporting 5,147 tokens in
total and no raw recognition token IDs. Text matches canonical on all 256 inputs; all frozen
quality point/bootstrap gates pass against canonical and the actual rebuilt
vLLM reference. This is four numerical inputs and one finite recognition set.
All 20 native inputs, complete operator/state/API/lifecycle and strict allocation
gates, historical VISION service, observer calibration and sustained performance
comparisons remain separate. No throughput win follows from this observation.

[The subsequent full 20-input package diagnostic](results/native-twenty-replay-run.json)
retains a failed G2 run and its smaller four-input reproduction. Both produce
the same six changed KV planes on the Chinese 100-frame boundary input after
checkpoint eviction; original prompt KV is exact, but generated-history KV and
final BF16 logits differ. Equal EOS argmax does not satisfy the byte gate.
The hidden-state root cause is not established by those KV observations.

A separate unmodified-package batch/slot run completes 62 cohorts and 190 natural
requests over all 20 inputs. Its [complete raw scalar checks](results/native-twenty-batch-slot-scalar.json.gz)
pass the unchanged frozen budgets, actual per-case batch/row witnesses and
complete same-path KV checks. The readiness barriers are test-only; these are
not automatic HTTP batching or serving-throughput observations. The G2 failure,
full model/lifecycle/operator qualification and strict allocation gate remain
open. Formal sustained comparison cells remain zero.


[The matched 394-token long-child configuration](results/native-transcription394-diagnostics-run.json)
retains both actual native attempts and the independent SDK attribution controls.
The whole prompt and the 388+6 message partition both fail the original
natural-teacher assertion at generated token 16: actual 72448, expected 42140.
They have identical 381 encoder features, all 56 final-prefill KV planes and
all 16 observed BF16 logit rows. The first 388 live KV rows are also unchanged
across the partition's Snapshot and suffix. These reads observe live request
pages after GPU completion, not immutable checkpoint destinations; device
position bytes were not observed. Both attempts drain normally. Neither
completes the 94-token request; their other three configured inputs are unrun.

[The original SDK encoder-node control](results/sdk-native-feature-control.json.gz)
first preserves all original 394-token prefill tensors, 64 teacher rows and 94 natural
tokens, twice. Its 447 encoder nodes are read-only observations. Replacing only
its encoder output with the actual native 381 features then naturally stops
at 97 tokens, twice; this does not reproduce the candidate loop or qualify a
candidate fix. Actual native versus independent SDK encoder relative L2 is
0.030207190959377794. The input PCM is identical; 26 FP32 Mel elements differ
by at most 2.384185791015625e-7, and every BF16 model Mel value is identical.

[Same-feature scalar comparisons](results/native-transcription394-same-feature-comparisons.json.gz)
localize a separate decoder discrepancy: layer 0 K/V are entirely byte-identical,
but layer 1 differs already at pure-text token 1, before audio positions 4..384.
At generated step 15, actual 42140/72448 logits are 32.5/32.75; the SDK with those
same features produces 33.0/32.75. The previous 15 generated IDs match.
[The decoder node collection](results/sdk-native-feature-decoder-nodes.json.gz)
adds 38 original SDK prefill nodes and preserves every tensor byte, 64 teacher
rows and natural generation from the preceding same-feature control.
These causal controls deliberately share candidate features and cannot serve
as independent encoder oracles. They change no frozen reference or budget.

The current finite primary qualification and same-path batch/slot checks do
not waive this long-child failure. Complete intermediate/operator, strict
storage release, lifecycle/cache and old-package serving gates, formal
sustained comparisons and final whole-task reviews remain open. Formal
comparative cells remain zero.


[An audio-only complete-forward capture](results/captured-encoder-component.json.gz)
reproduces all 381 actual package features exactly across the four windows.
Each observed window also preserves all declared persistent views of its
unobserved graph. Only after these bridges pass are the nodes interpreted.
[All 1,692 scalar node comparisons](results/captured-encoder-SDK-node-comparisons.json.gz)
retain the first difference at conv2d2 output: 275/260/269/199 BF16 elements
in the four windows. Conv1 input/output and conv2 input are entirely exact.
At equal BF16 preactivation elements, GELU2 independently differs in
686/746/746/499 outputs; encoder layer0 GELU differs in 29/29/22/17 outputs.
The last window retains six padded CNN chunks but only 69 valid encoder
features. Neither those differences nor later propagation establish the
complete long-loop root cause. This is one captured encoder component,
not the full frozen numerical matrix, a re-exported package or a qualified fix.


[A single complete prefill capture](results/captured-prefill-component.json.gz)
first reproduces all 56 actual native KV planes and last-position BF16 logits.
Its observed graph preserves every compared state view and the same KV/head bytes,
then records 31 nodes. [The 29 common node comparisons](results/captured-prefill-SDK-node-comparisons.json.gz)
against the original SDK with the same features find the layer0 input norm,
Q/K/V projections, Q/K norms and rotary cos/sin entirely exact. The first
observed difference is attended, before o_proj: 51/806,912 BF16 elements,
maxabs 0.0009765625, relative L2 3.7019087652413166e-5. This does not yet
separate post-RoPE Q, attention library, paging and physical query geometry.

The same-operand [original SDK](results/encoder-same-operands-SDK.json.gz) and
[project](results/encoder-same-operands-project.json.gz) eager controls preserve
all mandatory original SDK/R301 operator bytes. In each environment, N8 and
N30 are byte-identical for convolution pre/postbias and GELU. Project eager N8
matches the actual captured component exactly; the earlier 275 convolution
and 941 GELU2 output differences remain across environments. At the same
SDK BF16 preactivation, GELU alone differs at 686 elements. The actual builds
are Torch2.13.0+cu130/cf30153c with cuDNN92000 and Torch2.14.1+cu132/5c488690
with cuDNN92400. This excludes N8/N30 and compilation for these controls,
not every shape or a specific library component. No shared driver or project
lock was changed. These are numerical diagnostics without timing claims;
bounded task CPU acquisition could share UMA during this operator run.


[Actual SDK GPU RoPE and attention controls](results/attention-same-operands-SDK.json.gz)
subsequently bridge post-Q/K to the candidate, K/V to the actual SDK cache,
and dense output to the observed SDK attended bytes. SDK varlen is identical
to its dense output. [Project same-QKV controls](results/attention-same-operands-project.json.gz)
bridge the actual request family to the captured native attended bytes.
Nonpaged Q394, paged Q394 with maxK394 or8192, and the original right-aligned
physical-Q8192 family are all identical to each other and the native output;
all retain the same 51-element difference from SDK. The original family is
not a Q512 call merely because its input buffer has512 rows. These controls
exclude paging and query padding as the source of this particular difference;
they do not identify a specific build instruction or explain the whole loop.
No model, quality, allocation or service-performance gate follows from them.
