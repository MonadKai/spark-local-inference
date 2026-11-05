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
Proposed input bindings are FP32 Log-Mel `[128,M]`, int32 valid frame count
`[1]`, convolution lengths `[C]`, gather indices `[Ecap]`, cumulative encoder
attention boundaries `[ceil(C/8)+1]`, and feature count/offset `[1]` each.
Metadata after the valid prefix is initialized: unused gathers are -1 and
unused cumulative boundaries repeat the terminal feature count. Eligibility
of zero-length padded attention windows in a selected b12x plan is unverified.
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
absent author ranges fix parameters at defaults. Proposed policy is explicit
greedy (temperature 0), top-p 1, disabled top-k/min-p and neutral penalties,
with request-controlled seed. This policy cannot claim non-greedy vLLM parity.
On 2026-11-06 the fork maintainer selected greedy first, prioritizing existing
Flare mechanisms. The broader exception is conditional on demonstrated need;
it is not activated by this design. Implement the author-backed greedy default
with Flare's existing fixed-default policy, and disclose effective parameters.
Non-greedy parity remains open. The primary quality qualification stays greedy;
a sampling change cannot repair a failed greedy numerical/quality gate.

A concrete full-parity alternative is an explicitly approved exception for
this recipe: temperature [0,2], top-p (0,1], top-k integers >=-1 (-1 and 0
disable), min-p [0,1], with the pinned reference's validation and normalization
order. In that reference, `0<temperature<0.01` becomes 0.01; greedy validates
filters before resetting them. Seed -1 means unseeded in vLLM, so its handling
also needs an explicit compatibility/determinism decision. These are observed
vLLM API ranges, not author recommendations. The proposal does not grant a
general exception for other recipes or alter the repository rule.

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
