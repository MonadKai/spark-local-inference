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

## Native API input boundaries

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

The issue's quality selection, scoring and numerical requirements remain
binding. Stage/logit scalar thresholds, independent reference repeatability,
actual corpus revisions and selected utterances still require freezing before
candidate inference. No threshold can be inferred from a matching transcript.

The primary performance comparison is full, naturally stopped ASR at B=4 and
B=8; each throughput-ratio 95% lower confidence bound must exceed one. B=1
request latency and mixed-load P95 have a 5% maximum regression budget, assessed
with uncertainty. Use independent interleaved repetitions, at least three
valid runs of five minutes and 100 complete requests each; raw failures and
warmups remain visible. Corpus/cache/configuration identity and statistical
resampling units must be frozen before candidate results. Until that complete
freeze exists, diagnostic kernel timings cannot establish a service speedup.

Lifecycle changes require the affected TLA+/Rust conformance, Kani geometry and
real CUDA fence/allocation checks described in [verification](verification.md)
and [Helios lifecycle](../verification/helios-lifecycle/README.md). Neither an
abstract model nor checkpoint inspection establishes native GPU correctness.
