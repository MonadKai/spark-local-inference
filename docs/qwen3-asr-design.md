# Qwen3-ASR native execution design

This is the proposed contract for [fork issue #1](https://github.com/MonadKai/spark-local-inference/issues/1).
It does not register a supported recipe. D1 has the downstream decision below;
D2 follows the greedy-first decision below. All M0–M8 acceptance requirements
remain open.

## Fixed references

| Role | Repository | Revision |
| --- | --- | --- |
| Original checkpoint | Qwen/Qwen3-ASR-1.7B | `7278e1e70fe206f11671096ffdd38061171dd6e5` |
| Canonical structure | huggingface/transformers | `1d9654e78a23fb167ef539cbd699dfbd3e4830cf` |
| Independent original-format reference | QwenLM/Qwen3-ASR | `7c6daf77a2421100f5fb066495372c00129d39ff` |
| Deployment/API comparison | vllm-project/vllm v0.29.0 | `98dff2a81d747d1dba01a47f939f48c3526d4206` |

The `-hf` checkpoint is a different input format. The original checkpoint has
708 BF16 tensors, including separately stored embedding and LM-head tensors.
The preparation adapter must consume every source tensor exactly once and retain
both tensors unless their complete stored values have been checked equal.
Shape equality and `tie_word_embeddings` do not establish that equality.

The pinned [Transformers converter](https://github.com/huggingface/transformers/blob/1d9654e78a23fb167ef539cbd699dfbd3e4830cf/src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py)
defines the original-to-canonical naming changes. Preparation validates the
original geometry before mapping disjoint prefixes. It must preserve encoder
LayerNorm, GELU, biases and residuals; ordinary Qwen3 full attention, Q/K norm,
RMSNorm and full rotary semantics; and the BF16 LM-head rounding before any
FP32 sampling conversion. Qwen3.8 hybrid attention is not this decoder.

The original tokenizer pads inputs with 151643, has no BOS, and declares
`<asr_text>` (151704) as a non-special token. Its generation stop set is
{151643,151645}; these structural values override newer native-format defaults.
The original template always renders a system turn, including empty context.
The pinned vLLM transcription frontend omits that turn for empty sanitized
context. Preserve endpoint-specific prompt bytes and IDs rather than treating
the SDK wrapper's prompt as the transcription API oracle.

The original SDK wrapper invokes its encoder separately for each audio. The
encoder's eager/SDPA paths do not consume cumulative attention window boundaries,
so a long single audio can attend across windows. The at-most-one-window eager
path can provide an initial independent numerical check; longer SDK comparisons
require a separately qualified backend that actually honors the boundaries.
Do not modify the SDK reference or report these distinct paths as interchangeable.

For single audio, the original SDK Whisper frontend produces `floor(N/160)` Mel
frames. The pinned vLLM processor first zero-pads PCM to a hop multiple and
therefore produces `ceil(N/160)` frames. For 16001 samples this changes audio
placeholder expansion from 13 to 14 tokens. Native service preprocessing adopts
the pinned vLLM hop-multiple padding explicitly in typed settings. Duration and
file limits use the original decoded sample count. Independent SDK numerical
checks receive PCM padded by this independently verified, declared rule; raw
SDK-wrapper comparisons remain separate. Normalization is still request-local.

## D1: audio encoding operation (downstream decision)

[AGENTS.md](../AGENTS.md#architecture-and-ownership) and
[design review principle 1](design-review.md#principles) require prior explicit
approval from Luke Alonso for a new captured-graph operation family. No approval
was present in issue #1 at the initial inspection. On 2026-11-06 the fork
maintainer explicitly directed this task to proceed within the fork without
strictly applying that upstream-owner approval requirement. This is a
downstream authorization, not an assertion of Luke's approval. Other
architecture, package, precision and resource-lifetime requirements remain.

The proposed operation authorized for downstream implementation is:

> Approve an `AUDIO_ENCODE` captured operation accepting one bounded FP32
> Log-Mel input and fixed-capacity valid-length, gather and attention-window
> metadata, and publishing generation-owned BF16 features. It returns a normal
> zero-token completion before ordinary text prefill. It does not change token
> history, KV, sampling or RNG state.

The initial encoder submission has one input audio segment. Different requests
can share ordinary decode batches of 1/2/4/8. Encoder and text graphs use shared
sequential scratch with explicit completion dependencies. This does not assume
multi-request prefill: the present prefill batch capacity remains one.

The input has 128 Mel bins, FFT 400, hop 160, and mono 16 kHz PCM provenance.
One convolution chunk has 100 Mel frames; a full chunk produces 13 vectors.
An encoder attention window covers 800 Mel frames. Neither quantity specifies
the service-level long-audio splitter. Preparation declares measured capacity
buckets and bounded valid-tail/gather/window metadata; padded rows cannot
participate in attention or publish features. Host `.item()`, list and dynamic
index construction must finish before capture.

Output is a BF16 feature window of width 2048 with an explicit valid vector
count. The owner is the request slot plus generation, retained across encoder
completion and every consuming prefill chunk. Publication becomes visible only
after successful encoder completion. A cancelled submission drains before
input, upload staging, scratch, feature storage or events can be reused.
Generation checks reject stale publication. Failed encoding cannot publish a
cache entry or permit text prefill.

The existing image path in `helios/src/scheduler.rs` couples feature-range
validation to image patch geometry. Add an audio input path and share ownership
and retirement mechanisms from their actual storage dependencies. Audio must
not enter `set_vision_images`, `IMAGE_TENSOR`, or image patch-count validation.
Ignition owns audio model computation; Helios owns declared storage and replay.

The decoder's seven bias-free layer projections use the shared fixed-reduction
component on eight physical rows. One column tile reads its weight operands for
all live rows; the sixteen FP32 accumulation chains and terminal BF16 conversion
remain fixed across logical B1/2/4/8. Prepared row packing preserves request
positions and generations, defines inactive rows as zero and maps results back
to the original request rows. Text prefill retains its canonical projection
groups. The vocabulary projection, including the first token after prefill,
retains the separate fixed-eight-row canonical BF16 result before sampling.

One-query GQA batches independent requests within each original FA2 catalog
family. Each request still selects its original split, partition and key bound;
all families execute and device metadata selects represented output bits.
General queries retain their right-aligned layout. Preparation compiles and
captures these programs; Rust serves their exported CUDA nodes. The
[native recognition observation](../benchmarks/qwen3_asr/results/native-recognition-batched-attention.json)
qualifies finite primary recognition on modified source. Complete numerical,
lifecycle, final-source and comparative performance gates remain separate.

## D3: additive typed data

In `helios/proto/runtime_package.proto`, `GraphOperation` value 14 is currently
free; 7 remains `VISION`, and reserved 10/11/13 remain reserved. Proposed
`AUDIO_ENCODE = 14` has the operation above. Proposed additions are:

- `ModelContract.audio` at field 8: input/feature capacities, feature width,
  storage views and the captured encoder capacity catalog.
- `FrontendContract.audio_preparation` at field 11 and `audio_tokens` at 12:
  independent typed preparation/token-layout identifiers and concrete syntax.
- `FrontendSettings.audio` at field 7: typed decode/resample/Log-Mel settings,
  accepted media, sample/file/duration bounds and bounded service chunk settings.
- Value sources `AUDIO_TENSOR=41`, `AUDIO_FEATURE_INDEX=42` and
  `AUDIO_FEATURE_SLOT=43`; `ValueBinding.audio_tensor=7` selects a typed audio
  tensor kind, preserving image `tensor=6`. Each source defines dtype, extent,
  indexing and inactive padding. Names cannot select an algorithm or policy.

For a captured capacity M (a multiple of 100), let C=M/100 and Ecap=13*C.
The initial catalog captures M=100 and M=800, each within one attention
window. Inputs are FP32 Log-Mel `[128,M]`, int32 valid frame count `[1]`,
whole-input convolution width `[1]`, gather indices `[Ecap]`, cumulative
attention boundaries `[2]`, and feature count/offset `[1]` each. The common
convolution width is `min(total_input_frames,100)`, including for a short last
attention window of a longer input. Convolution padding is masked after each
stride-2 convolution and GELU. Unused gathers are -1; boundaries are `[0,count]`.
Empty attention is used only to guard invalid requests and still requires
actual backend qualification.
For L valid frames, the feature count is
`13*floor(L/100) + ceil((L mod 100)/8)`, rather than `ceil(L/8)` for a clip.
Storage is BF16 `[feature_slots,features_per_request,2048]`; publication has
int64 generation and int32 count per request slot. Text-prefill feature
indices are int32 `[prefill_capacity]`, -1 for ordinary text.

These field choices must be rechecked against the implementation base before
editing the schema. The exact buffer bindings and extent arithmetic must be
specified and tested before implementation; this proposal is not a schema PASS.
Keep current contract version 3 and its reader behavior for absent audio fields.
If a semantic version change proves necessary, retain an explicit version-3
reader. Execution ABI numbers remain descriptive. Preserve signed wire bytes,
existing assets, completion defaults (zero means eight), `VISION` geometry,
legacy readers and all structural/resource/signature checks.

New audio metadata is inert when absent. No raw processor/config JSON is a
runtime asset. Decoder prefill uses typed feature indices and generation checks;
audio preparation identifiers cannot select scheduler/cache policy.

### CPU feature retention

The native frontend can retain whole-input FP32 Log-Mel in the existing bounded
prepared-input store. Its key separates the CPU processing format from the
conditioning identity and includes the actual decoded PCM and typed processing
settings. It does not use a filename, URL or placeholder-token count. A hit skips
FFT/Mel computation, then validates the exact frame/payload extent, finite values,
current feature capacity and window layout. The request retains the original
conditioning identity. Decode and encoder execution still run.

The record contains one complete normalization domain, including inputs spanning
multiple encoder windows. `insert_with` reserves serialized storage before
encoding; existing store leases, budgets and eviction remain in force. Input
staging accounts for the whole-input feature bound rather than the largest
encoder graph. Absent audio metadata retains the image accounting path. Invalid
records are discarded and recomputed; a valid record rejected by the current
capacity remains valid in the store. Cancellation is checked around blocking
store/plan boundaries; waiting for a store lookup or source guard is not itself
interruptible.

### Encoded feature retention

The implementation adds a separate BF16 record format keyed by the effective
authenticated package/deployment producer, actual decoded-PCM/frontend identity,
feature count and width. Language/context stay in the prompt and prefix-KV key;
they do not condition this encoder. Generation and slot indices are never stored
in retained records. Corrupt retained records are invalidated and recomputed;
nonfinite live graph outputs and CUDA errors fail execution.

The worker polls the existing bounded host store without waiting for disk. A
late lookup cannot delay normal encoding or replace a partially executed input.
An eligible hit receives a scheduler ticket for the exact owner, epoch, feature
slot, original input span, feature range, complete jobs and current window.
Only one whole input per current window is eligible. A startup-owned pinned slab
copies payload and current int64 generation/int32 count into declared storage;
its event must complete before the ticket retires the prerequisite. No graph
completion, token, RNG or KV progress is fabricated. Revalidation rejects stale
tickets without changing the epoch. Future conditioned windows keep their jobs.

Readback likewise requires a quiescent executor and checks the actual generation
and count. CPU serialization is reserved before allocating the retained payload.
The fixed pinned slab is a separate startup memory charge; graph resources and
the slab stay owned until confirmed executor drain, including transfer failure.
Explicit encoded/both modes currently reject multiple ranks because the package
does not declare replicated or sharded feature transfer semantics.

This path is under qualification. The retained-package
[native tier controls](../benchmarks/qwen3_asr/results/native-input-cache-tiers.json)
record four independent Rust servers, unchanged raw tokens across all tiers,
equal-placeholder/different-audio and context controls, and successful encoded
uploads with no encoder submissions. HTTP concurrency is not an actual GPU batch
observation. These finite results use an earlier ASR package and modified native
source; they do not establish in-flight worker cancellation, eviction/readmission,
disk retention, final-source quality or independent cache-tier performance.

`qualify/asr_native_input_cache.py` provides the portable collector with explicit
binary, package, configuration, publisher keys, development PCM and output paths.
The subsequent [portable run](../benchmarks/qwen3_asr/results/native-input-cache-batches.json)
again completes all four modes on the earlier ASR package. The newer native
binary records successful shared submissions with logical request counts 1–8
and separate physical capacities; counter indices refer to the loaded manifest,
so no graph-kind mapping from the inspection catalog is asserted. This adds
actual submission observations without qualifying worker cancellation during
copies or the new attention package. CPU layout/ticket tests do not establish
CUDA copy fencing or native request lifetimes.

The subsequent [actual Worker upload controls](../benchmarks/qwen3_asr/results/native-feature-upload-cancellation.json)
load that retained signed ASR package and run its actual frontend, encoder,
readback and generation graphs. A mapped CUDA wait holds the real upload event
pending while another thread cancels the request. The unchanged prerequisite,
hit counter and window state precede ordinary slot/arena retirement and reuse;
the next generation's natural tokens equal baseline. Unmodified tests pass before
and after two source mutations: omitting the cancellation recheck or the upload
fence fails the upload's subsequent prerequisite-state invariant in both runs.
The first collector's overly narrow expected-error text is retained as a separate
failed attempt.
Test-only probes do not enter production builds. This closes that finite upload
control, with HTTP disconnect propagation, other cache lifecycles, the new
attention package and final-source qualification still separate.

Ordinary CPU tests cover hits with no FFT plan available, identity isolation,
corrupt records, short-input rules, capacity rejection and cross-window equality.
Native HTTP qualification remains required. When the prepared-input option is
omitted, request prefix reuse also disables the input store. Explicit
`--prepared-input-cache` now permits independent tier ablations. Encoder-embedding
retention and its lifecycle validation remain separate gates; CPU feature hits
do not establish encoder-cache reuse.

### Whole encoder-input length

Convolution batch geometry belongs to the complete normalized model input,
whereas `AUDIO_VALID_FRAMES` describes one attention window. For example, a
150-frame input and the 150-frame tail of a 950-frame input both have convolution
width 100, but their original convolution batches contain two and ten chunks.
Adding a 200-frame graph alone cannot distinguish these cases; the shared
scheduler may also legally choose the existing 800-frame graph.

The next additive input is `AUDIO_TOTAL_INPUT_FRAMES`, an int32 scalar containing
`PreparedAudio.frames`. It means the complete encoder input after any service
split, not the parent transcription's duration. Native packing copies that
existing value; it does not select a convolution algorithm. Ignition validates
the whole-input feature bound, window length and 800-frame-aligned feature
offset before publishing features. It can then use measured convolution
geometry without confusing a short input with a long tail.

This is an optional binding for package validation: retain the seven existing
required audio bindings and their old layout semantics. The extra scalar does
not by itself qualify any convolution geometry. Old captured graphs do not read this
new value and must load and serve without re-export. New graphs declare the
additional buffer explicitly. The operation, attention-window size, completion,
generation ownership and shared graph-selection policy stay unchanged.

The added transcription field 13 declares prompt/output syntax, language labels,
joining and control-marker removal. Its finite marker grammar requires the
excluded body character exactly once at the end of the prefix and at the start
of the suffix. Each round removes the entry string's leftmost nonoverlapping
controls, then its literal text delimiters; new matches in the same phase wait
for the next round. This preserves the pinned frontend's fixed point without
quadratic full-string rescanning. It does not introduce executable metadata.
This field and grammar are new in this feature; historical absent-field packages
retain their existing frontend. No additional request prompt byte limit is used
to conceal expensive sanitization.

Required compatibility evidence includes real previously exported image and
text packages loading and serving unchanged, plus a new audio package loaded
by Rust after the preparing Python process exits. Schema round trips alone do
not establish either property.

## D2: greedy first; broader sampling remains open

The pinned [model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/7278e1e70fe206f11671096ffdd38061171dd6e5/README.md)
documents BF16 greedy evaluation with 1024 output tokens and no forced language.
Its offline sampling example uses temperature 0.01 and 256 tokens; that example
does not recommend an inclusive range for temperature, top-p, top-k or min-p.
The pinned [SDK](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/qwen3_asr.py)
uses temperature zero in its vLLM wrapper.

Under [model development](model-development.md#sampling-and-frontend-metadata),
absent author ranges fix parameters at defaults. Required ASR policy is explicit
greedy, top-p 1, disabled min-p and neutral penalties, with request-controlled
seed. The earlier claim that internal temperature 1 and top-k 1 implement this
greedy distribution was incorrect. Existing FlashInfer filtering retains every
token tied at the cutoff, then samples them with RNG. The existing
`test_sampling_flashinfer.py` threshold-tie test requires that behavior. BF16
LM-head rounding can create a maximum tie even when FP32 products differ.
Those stochastic semantics must remain unchanged for existing models/packages.
On 2026-11-06 the fork maintainer selected greedy first, prioritizing existing
Flare mechanisms. The broader sampling-range exception is conditional on
demonstrated need and remains inactive. Exact greedy selection also conflicts
with the [core design sampling rule](core-design.md), which requires one probability
sampling path, rejects zero temperature and preserves cutoff ties, and with
the [manifest](../helios/manifest.md), which specifies the same stochastic
executable. The new tie finding requires an explicit downstream decision on the
following minimal preparation-only exception; the previous design description
did not identify that conflict accurately. On 2026-11-06 the fork maintainer
explicitly approved: “批准固定 argmax 例外，但一定要加文档说明”. This authorizes the
fixed selection below for this ASR recipe only. It is not upstream-owner
approval, broader sampling-range approval, or a seed=-1 exclusion. Implementation
and qualification remain separate from the design authorization.
The decision is recorded in [issue #1](https://github.com/MonadKai/spark-local-inference/issues/1#issuecomment-6675677496).

### Approved downstream exact-greedy exception

Use an explicitly prepared fixed argmax selection for the ASR recipe,
using PyTorch's existing first-maximum operator. This does not activate broader
sampling ranges or a seed=-1 exclusion, and does not reinterpret old stochastic
top-k graphs. Internal native parameter validation remains positive-temperature;
there is no native Greedy mode, model allowlist, runtime selection switch or new
captured operation family.

The prepared selection is a constant in the recipe's graph program, passed
through both `PreparedGeneration` and `PreparedResume`. It covers the first
prefill token, ordinary decode and cached-hidden resume. Keep the BF16 LM-head
rounding before FP32 score processing. Declare an Int64 `[rows]` argmax output
through the existing scratch planner, call `torch.argmax(scores, dim=-1, out=...)`,
and copy the selected IDs into the existing Int32 sampling output. Maximum ties
choose the lowest token ID, independently of seed, slot and row arrangement.

Reuse existing penalties/grammar score preparation, completion, broadcast,
logprob and record mechanisms. Preserve the maximum before sanitization so
nonfinite and all-masked rows fail through the same guards. Supply the declared
point-mass validity/mass to completion; preserve generation, terminal, pending,
budget and inactive-row checks, and the successful counter+1 state reservation.
The selection adds bounded scratch, not serving allocation or recapture.

The default prepared selection remains the current categorical sampler for all
existing call sites. Old packages retain their captured executable and loading
semantics without re-export or signing. This proposal does not change Rust
scheduling, admission, sampling-mode parsing or graph/state/completion contracts.
The downstream core-design and manifest notes state this distinction explicitly.

Qualification must cover unique maxima, all-equal logits and maximum ties across
word/tile/vocabulary boundaries; BF16-rounding-created ties; B=1/2/4/8, partial
rows, permutations, slot generations and seed/counter changes; inactive/invalid
state and sample-disabled prefill; resume; repeated capture/native replay and
allocation/storage stability. Old categorical threshold-tie and bytewise gates
remain intact. Full package, numerical and recognition gates remain required.
Non-greedy parity remains open. A sampling change cannot repair a failed greedy
numerical/quality gate by substituting a different evaluation distribution.

A concrete full-parity alternative is an explicitly approved exception for
this recipe: temperature [0,2], top-p (0,1], top-k integers >=-1 (-1 and 0
disable), min-p [0,1], with the pinned reference's validation and normalization
order. In that reference, `0<temperature<0.01` becomes 0.01; greedy validates
filters before resetting them. Seed -1 means unseeded in vLLM, so its handling
also needs an explicit compatibility/determinism decision. These are observed
vLLM API ranges, not author recommendations. The proposal does not grant a
general exception for other recipes or alter the repository rule.

### Signed request seed syntax

The native frontend declares nullable signed-64-bit seed syntax as typed data,
independently of model identity and the distribution sampler. An absent declaration
retains every old package's unsigned request syntax and packaged seed default.
For the new declaration, explicit fixed integers retain their 64 bits; for example,
-2 becomes the native unsigned key 18446744073709551614. Omission, JSON null and
-1 resolve once per parent to a fresh host key. Every transcription child receives
that same key. Native uint64 storage, Philox counters and prepared graph arithmetic
remain unchanged; greedy argmax output is seed-independent.

Pinned vLLM/Pydantic request-model probes establish signed bounds and numeric/text
coercion cases before native comparison. Native seed text cleanup retains the
Pydantic MIT notice. A form field containing literal `null` is not JSON null and
is rejected, matching the request model. This design does not activate stochastic
sampling ranges or exclude -1. CPU request-model comparisons do not establish
multipart HTTP or GPU sampling equivalence.

## Native API input boundaries

Audio chat reuses Flare's existing `return_token_ids` and
`return_token_timestamps` observation options. These expose actual prompt/raw
generated IDs and worker completion observation times; they do not request
word-aligned ASR timestamps. The shared parser retains strict boolean/null
validation and requires IDs when timestamps are enabled. The audio subset still
rejects `timestamps`, streaming and multiple choices. Ingress/enqueue-to-first
raw-token differences include native delivery/polling; they are server-observed
latencies, not GPU-only model TTFT. The ordinary parser tests pass; real native
audio-chat observation qualification remains required.

The implementation remains unregistered and requires real HTTP/model
qualification. Transcription declares the pinned endpoint's 100 accepted
Whisper language codes. Its 57 model-supported codes use the published language
labels; the other 43 use the literal code in the prompt, matching the reference
validator and constructor. This is syntax compatibility, not a quality claim for
those languages. Empty strings, uppercase codes, language names and locale
aliases are rejected rather than normalized silently.

Public audio fetching accepts HTTPS port 443 with ordinary certificate and SNI
verification, URLs up to 4096 bytes, and no credentials or fragment. It ignores
proxy environment settings. Every resolved connection address must be public;
mixed public/nonpublic answers are rejected together. The connector consumes
that checked address list without another lookup. Special-use, documentation,
NAT64 and transition ranges are conservatively excluded. Redirects are limited
to three hops, statuses 301/302/303/307/308, with URL and destination validation
on each hop; the final response must be 200.

Fetching requests identity encoding, but accepts ureq's transparent gzip
delivery. The 25 MiB limit applies to the resulting encoded audio file, before
FFmpeg decoding. The HTTP entity reader has a separate 32 MiB plus one-byte
probe allowance before gzip decoding. A cumulative 32 MiB TCP budget includes
TLS and HTTP framing across all hops; the failing read can receive at most one
additional 16 KiB buffer. These are different domains, so a file below 25 MiB
can still exceed a transport budget. Decoded PCM has the independent 600-second
limit. These are downstream input limits, not universal vLLM defaults.

One absolute 30-second fetch deadline spans redirects. DNS waiters check
cancellation at intervals up to 50 ms; blocked network operations use steps of
at most two seconds and recheck cancellation/deadline after I/O. An OS lookup
cannot be cancelled: its worker reservation remains held until that lookup
actually exits, even after its HTTP waiter has gone. Lookup capacity is bounded
by the configured frontend worker count; late results cannot enter the model.
A real TCP disconnect test cancels admitted and pending transcription children
through the HTTP listener and returns host/frontend reservations. This uses
controlled native queue inputs, not a GPU model. Public redirect chains and
model-backed HTTP cancellation/draining remain separate qualification gates.

## Qualification boundaries

Native audio decoding and resampling, identical-PCM Log-Mel comparisons, and
feature-to-model comparisons are separate gates. CPU plans use per-input
normalization. The SDK's long-clip splitter and minimum-tail handling differ
from the newer Transformers feature extractor; service behavior must be frozen
against the pinned runnable vLLM rather than inferred from either processor.

The issue's quality selection and scoring remain binding. The frozen corpus
configuration records actual revisions and selected utterances. The
[arithmetic contract](qwen3-asr-numerics.md) fixes development stage/logit
thresholds from two independent references before candidate inference and
records their separate repeatability requirements. These are acceptance gates;
they do not establish a passing candidate or recognition quality.

A fresh exported arithmetic-baseline package subsequently completed all 256
frozen qualification requests in an independent Rust process after its preparer
exited. Every response stopped naturally; none failed or truncated. English WER
was 70/2875, English CER 94/15411 and Chinese CER 35/1861. Both frozen canonical
and vLLM comparisons pass the unchanged point budgets and paired bootstrap
gates. Against vLLM, the English WER degradation was 0.0003478261 with upper
95% bound 0.0010183299; Chinese CER degradation and its upper bound were zero.
The complete responses, configuration and both score objects are in
[the ordinary recognition result](../benchmarks/qwen3_asr/results/native-recognition-current.json).
This is a modified-worktree arithmetic baseline, not final source or performance
acceptance. Raw generated token IDs were not observed in this run. That baseline's
separate largest text-prefill diagnostic failed scalar logit gates on two
development inputs; its recognition result did not waive that capacity requirement.
The subsequent fixed-reduction/batched-attention candidate
[reports](../benchmarks/qwen3_asr/results/README.md) retain 240 full-prefill
production bridges and frozen scalar checks in each of two audio-capacity runs.
The later B1-only observation also passes all 423 audio and 650 decoder nodes on
every frozen input. Independent operators, the complete schedule with these new
observations, native lifecycles and final-source qualification remain open.

The package in that historical recognition record used an arithmetic baseline
for decoder projections. Its decode metadata and pointwise computation had eight
physical rows, while each layer projection issued eight separate M=1 ATen matrix
products through `PreparedRowwiseProjection`. Its vocabulary projection, including
the first prefill token and cached-hidden resume, used one eight-row ATen matrix
product. That record does not measure cross-row layer GEMM. The subsequent shared
fixed-reduction path described above has separate model observations; its complete
serving-step performance qualification remains open.

Prefill captures the 64..4096 token catalog. Its fixed subrow projection extents
19/27/32, plus 64 in larger capacities, all execute and select represented BF16
outputs by device count. Encoder
projections likewise execute the regular 104-row and small 13-row geometries;
`conv_out` also executes a 512-row geometry for measured whole-input classes.
These choices address arithmetic differences, not reduced GPU work. Their
unused branches must not be credited as execution savings.

Request attention selects the original request's split count and partition
geometry using an immutable device lookup. The current baseline executes every
prepared call group, using zero key length for unselected groups, then selects
the result by bits. It preserves the real GQA shortcut and right-aligned causal
rows without native model policy. Its launch, copying and scratch costs require
measurement and optimization; the catalog itself is not a performance gain.

Projection padding alone did not remove cross-capacity differences in captured
decoder histories. The current candidate also prepares decode arithmetic,
including paged attention and final normalization, at eight physical rows.
A shared metadata copy preserves the original packed token order and appends
empty requests: their query lengths are zero, positions and pages are invalid,
and KV writers must leave state untouched. Sampling, hidden-state publication
and completion still use the original logical request rows. This is a graph
shape choice inside preparation, not a new native scheduling policy. Its full
numerical, lifecycle and performance acceptance remains open.

The primary performance comparison is full, naturally stopped ASR at B=4 and
B=8; each throughput-ratio 95% lower confidence bound must exceed one. B=1
request latency and mixed-load P95 have a 5% maximum regression budget, assessed
with uncertainty. The frozen [performance protocol](../benchmarks/qwen3_asr/performance-protocol.json)
requires five independent A/B/A blocks, with each primary run lasting at least
five minutes and completing at least 100 requests. Raw failures and warmups
remain visible. The fixed corpus, cache regimes and statistical units are
recorded before comparative measurements; diagnostic kernel timings cannot
establish a service speedup.

Lifecycle changes require the affected TLA+/Rust conformance, Kani geometry and
real CUDA fence/allocation checks described in [verification](verification.md)
and [Helios lifecycle](../verification/helios-lifecycle/README.md). Neither an
abstract model nor checkpoint inspection establishes native GPU correctness.

### Proposed immutable original-prompt geometry

This is a proposal, not an implemented native contract or qualified path.
The native 165-token versus `128+37` replay control exposed different decoder
histories. Independent canonical forwards also use length-dependent projection
and attention reductions. Six fixed projection rejection controls have not
passed the frozen node gates. None of those observations proves that every
uniform arithmetic implementation is impossible.

The proposed input `INITIAL_PROMPT_TOKENS` is the immutable number of tokens in
the first fully validated, expanded prompt, including actual image/audio feature
tokens. Freeze it before the first queue admission or cache lookup. Carry it
through parking, readmission, branching and offload; never reconstruct it from
the enlarged replay history. Existing `LiveRequest.prompt_tokens` continues to
mean the current admission's replay endpoint. The original usage count should
read the same authoritative fact rather than maintaining a second independent
initial count. The source is ordinary request data, not a recipe identifier or
an instruction for native arithmetic.

An additive `ValueSource` would bind that count as a positive int32 per active
submission row, zero for inactive rows, using ordinary submission staging and
completion fences. No native callback, expression evaluator or per-model policy
is introduced. The recipe owns every use of this input. In particular, rows
before that boundary may use prepared original-prompt arithmetic; generated
history at or after it must reproduce the entire ordinary decode chain during
readmission, including projection, Q/K normalization, RoPE, KV writes, attention,
FFN, final normalization and all required BF16 rounding nodes.

If graph variants need different physical original-prompt storage extents,
declare a separate `ExecutionGraph.initial_prompt_capacity`. It is a real bound
on captured buffers and operations, not an arbitrary tactic identifier. Resolve
the smallest covering capacity once from the complete immutable package catalog
using the initial count; retain that capacity through readmission. Then select
the current token/batch shape only within that exact capacity. Queue pressure,
cost estimates, current chunk length and generated-history length cannot change
the resolved capacity. A missing combination rejects explicitly instead of
falling back to a larger capacity or legacy graph. Runtime configuration must
not filter this dimension. Descriptor validation, startup binding, cost keys,
completion dependencies, threaded agreement and wire fingerprints must preserve
it. The dimension belongs to the sole prefill lane in Prefill, ScoredPrefill or
Mixed; independent decode lanes can have different original lengths. Decode,
Resume and input-encoding graphs do not acquire a meaningless prefill dimension.
Existing Prefill remains B=1 and Mixed contains at most one prefill lane; this
proposal does not claim batched prefill for several original lengths. Package
validation must require a closed catalog for every advertised capacity and all
prefill/mixed shapes reachable in that package and configuration before admission.
Another request joining or finishing cannot cause a missing-variant failure. Native scheduling
must not split queues by original capacity to compensate for an incomplete
catalog. The first implementation should use one fixed physical extent throughout
the package and bind the immutable input without a new native graph dimension.
That extent must cover all advertised original lengths and belong to the existing
captured-program producer identity; it cannot vary with runtime configuration.
Only measured need for distinct physical extents would justify the extra axis.

The captured program must never read future KV to reproduce a reference launch
geometry. Its first 128 queries use only 128 consumed keys even when its original
prompt has 165 tokens. Fixed physical query padding and split/partition bounds
are preparation choices; actual lengths and owned pages still bound access.
The focused proof compares identical full and fragmented Q/K/V, poisons unused
future keys, primes a larger used-key extent before the shorter replay, and
checks active poison rejection, recovery, repeatability and allocation stability.
A passing component proof would still require full-model and native qualification.

Arithmetic depending on the immutable count creates a real cache dependency.
Add an explicit list of immutable request-value dependencies to `CacheContract`;
initially accept only this concrete source. Its consumers canonically hash the
exact count and resolved physical capacity as a domain/version/source-tagged
condition before every nonempty history boundary is constructed or looked up.
In the initial single-extent design, the existing producer identity binds that
extent and the request dependency encodes the exact count; no synthetic capacity
field is needed.
Do not add fake audio events to `Prompt.conditioning`, change scopes to hide the
dependency, or use only the capacity in the key. Apply the same effective history
to GPU/host/disk publication, lookup, eviction and readmission, including partial
pages and saved hidden state. PCM/Mel/encoder identities do not depend on this
decoder-only input. Validation must reject unknown or duplicate dependencies and
any history-writing graph that reads the source without declaring the dependency.
A nonzero original-prompt capacity also requires the dependency even when the
captured graph does not bind the count explicitly; capacity selection itself can
affect its history arithmetic. Every such capacity must agree with the source's
declared count bounds and the prefill-lane binding. Graphs without a prefill lane
cannot declare a nonzero value. Cache dependencies cover decode publication and
saved-hidden resume too, even when those graphs do not read the original count.

Absent dependency lists, absent value bindings and zero original-prompt capacity
retain the existing behavior. The empty-dependency path preserves previous hash,
producer and offload identity bytes; zero-capacity graphs are not a wildcard for
new variants. Historical packages require no re-export, re-signing or ABI gate.
New validation must cover legacy/new catalog ambiguity, count/shape bounds,
same-prefix different-count isolation, partial pages, cancellation, slot/epoch
reuse, offload and generated-history readmission. Actual old-package serving,
all affected shared cache schemes and formal lifecycle checks remain required.

### Native frontend numerical variants

`AudioSettings.numerics` is typed preprocessing data. Absent/zero retains the
original FFmpeg FFT, analytic FP64-to-FP32 Hann and separate FP32 Mel reduction.
Legacy settings omit this field when serialized, preserving their feature-cache
identity bytes. `FP32_FMA` uses explicit FP32 Hann nodes, pinned PocketFFT over
the complete STFT frame extent before dropping its last frame, sequential FP32
FMA Mel accumulation and SLEEF log10. Both normalize each input independently.
The numerical variant participates in audio cache identity; it selects no GPU
operation, scheduler policy or model-specific native algorithm.

The fused variant currently requires Linux AArch64 SVE and rejects unsupported
hosts during plan creation. Previously exported packages keep their old typed
settings and need no re-export or signature change. The recipe declares the
new variant for newly prepared packages. Native dependency source and licenses
are pinned with the code; serving does not link Torch or Python.

Twenty frozen CPU inputs matched the independent SDK BF16 frontend boundary in
the isolated standalone-library experiment, including native Hann generation.
That is not native model acceptance. The integrated Rust plan, package metadata,
legacy bytes/cache identity, cancellation and full model gates must pass before
this variant is described as qualified.
