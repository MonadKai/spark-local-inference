# Qwen3-ASR arithmetic and qualification

The original `Qwen/Qwen3-ASR-1.7B` revision
`7278e1e70fe206f11671096ffdd38061171dd6e5` uses the arithmetic identity
`qwen3-asr.bf16-fp32.v1`. This document defines acceptance criteria, not a claim
that the native model has passed them. Keep the
[recognition gates](../benchmarks/qwen3_asr/qualification.json) separate from
these development probes.

## References and input boundaries

The primary model reference is unprepared Transformers
`1d9654e78a23fb167ef539cbd699dfbd3e4830cf`, loaded through its unmodified upstream
original-format converter. The independent original SDK reference is
`7c6daf77a2421100f5fb066495372c00129d39ff`. Both loading paths compare all 708
logical tensors to the original files. Neither reference imports Ignition or
Heliopause. They retain separate environments and their actual Torch versions;
cross-environment floating-point byte identity is not assumed.

`qualify.asr_sdk_stages --cached-state --frozen-sdk ORIGINAL_OBSERVATIONS`
supplements the SDK's complete cached KV and actual FP32 rotary positions only
after matching the original input, full prefill tensors, natural token IDs and
teacher logits/targets exactly. The first four whole-audio observations predate
`source_id`: their complete source is the existing `id`. Accept that layout only
for a whole-audio input whose current `id == source_id`. Boundary observations
require their explicit source and identical prefix PCM hash. This preserves the
original archives and numerical budgets.

[The reconstructed SDK CPU observation](../benchmarks/qwen3_asr/results/sdk-reconstructed-frontend.json)
matches all twenty original processor BF16 feature tensors and actual SDK prompt
bytes/IDs. The new local reference image has a different identity from the
unavailable historical image. These CPU checks do not establish the required
frozen GPU prefill, cached state or logits equality.

The original SDK retains 1500 audio positions; each convolution chunk uses at
most thirteen. Compare that entire used FP32 prefix and both rotary-frequency
buffers exactly. Canonical stores thirteen positions. The three original
mRoPE axes are identical sequential positions; require all three to agree
before comparing them with the canonical ordinary-position view.

Original-format F<100 inputs use their true common convolution width. The
canonical top-level encoder normally pads to 100. The reference adapter only
slices the `conv2d1` input to F in a pre-hook for those inputs; subsequent
convolutions, activations, encoder layers, projector and decoder remain the
canonical forwards. The unmodified padded-100 path is retained as a wrong-layout
control. This reference hook is never part of native serving.

Development probes comprise the shortest and longest selected utterance per
language plus first-2/13/99/100/101/799/800/801-frame prefixes of the longest
utterance per language. These prefixes were fixed before candidate execution.
They cover common-width and attention-window boundaries, not recognition
accuracy of those fragments. The >30-second whole-clip probe is a model-stage
diagnostic, not evidence about transcription service splitting.

Separate file-to-PCM, identical-PCM frontend, identical-feature model, and full
native file-to-text checks. Identical-feature probes may supply the reference's
represented BF16 features through FP32 graph inputs; that does not qualify the
native FP32 Log-Mel computation or resampler. Full recognition uses the actual
native frontend on the frozen PCM files.

For identical-PCM FP32 Log-Mel, freeze maximum absolute error at 2^-13 and RMS
error at 2^-17 against the pinned canonical frontend, with no nonfinite values.
These downstream engineering budgets reserve respectively 1/32 and 1/512 of the
BF16 unit roundoff (2^-8); they are not FFT stability theorems. Independently
characterize FP32 versus FP64 periodic-Hann/STFT/Slaney/clamp/log/normalization
through `qualify/asr_frontend_precision.py`, and preserve the actual discrepancies.
Earlier unqualified frontend bringup diagnostics do not establish acceptance.
Exact frame/sample/padding boundaries and per-input normalization remain separate
requirements. File resampler differences cannot be charged to this same-PCM gate.

## Required arithmetic

All checkpoint parameters stay BF16. Nonpersistent positional constants remain
FP32 until their declared conversion. Disable Torch's BF16 reduced-precision
reduction before constructing, tuning, warming and capturing ASR programs;
restore the caller's setting on exit, including failure. b12x's legal projection
candidates include Torch GEMMs, so the policy covers those too. Record the policy
in preparation identity and memory/tuning output. Native replay uses the captured
executable and has no numerical mode selected at runtime.

Preserve the separate split-K permission when changing this reduction setting,
and restore both values on exit. Compiler submissions carry both values to
spawned workers, including reused workers, and both enter the forward-cache key.
Record split-K permission in the preparation identity and memory report so two
different arithmetic configurations cannot share host prefix records.

| Boundary | Required evaluation and BF16 nodes |
|---|---|
| Input | Per-input FP32 Log-Mel normalization, then BF16 before convolution |
| Convolution | FP32 compute; BF16 convolution result; separate bias addition and BF16 result; exact GELU and BF16 result at each of three stages |
| Convolution projection | BF16 `conv_out`; FP32 position table converted to BF16; position addition rounded to BF16 |
| Encoder norms | Native LayerNorm, FP32 statistics/normalization/affine, eps 1e-5, BF16 output |
| Encoder attention | Q/K/V including bias each BF16; bidirectional window isolation; attention output BF16; output projection/bias BF16; residual addition BF16 |
| Encoder FFN/projector | First linear/bias BF16, exact GELU BF16, second linear/bias BF16; layer residual additions BF16 |
| Embedding injection | BF16 feature values copied to the declared placeholder positions; exact generation/count/index checks |
| Text RMSNorm | FP32 square/mean, eps 1e-6, rsqrt and normalization; normalized value rounded to BF16 **before** BF16 weight multiplication and its BF16 rounding |
| Q/K and RoPE | Q/K RMSNorm as above; FP32 frequencies/cos/sin then BF16; separate BF16 products and BF16 rotary addition; full 128-dimensional rotary geometry |
| KV and attention | Exact rotated BF16 K and original BF16 V publication; causal full attention, Q16/KV8/head128; attention result BF16 |
| Text MLP/residual | Separate gate/up BF16 outputs; FP32 SiLU evaluation then BF16; multiply with up then BF16; down BF16; residual addition BF16 |
| Final head | Final RMSNorm/hidden BF16; LM head result **BF16**; only then FP32 sampling scores |

Contiguous encoder linears use the canonical
[ATen addmm bias epilogue](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/Linear.cpp#L97-L115),
with one BF16 linear/bias output. They do not introduce a separate pre-bias BF16
node. An independent FP64 matrix-product-plus-bias check is an end-value sanity
check; it does not prove the compiled intermediate nodes by itself.

The compiler preserves intermediate precision casts and native mean, LayerNorm,
exact GELU and FP32 sigmoid boundaries. This is a required implementation
mechanism, not runtime numerical evidence by itself.

The pinned ATen CUDNN path allocates its convolution output in the input dtype,
then adds bias separately: see [Convolution.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/Convolution.cpp#L1505-L1512)
and [ConvShared.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cudnn/ConvShared.cpp#L243-L246).
The BF16 result before bias must survive any replacement using an unfolded
projection. Check the reduction before bias, then the separate BF16 bias and
GELU nodes. This corrects the description of the existing arithmetic; the
captured path and frozen stage/logit budgets are unchanged. The FP32 compute
descriptor does not establish cuDNN's internal reduction order or RNE model.

## Local operator gates

Use the same represented input/history/weights for each local comparison.
Independent reference arithmetic must not consume candidate-packed weights or
candidate-generated activations as its only input source.

| Probe | Scalar acceptance |
|---|---|
| Finite linear/conv reduction with a qualified RNE FP32 update model | Maximum absolute error divided by the independently computed full-interval envelope <=1 |
| Unchanged native ATen norm/activation/rotary nodes under the same pinned Torch stack | Exact BF16 node results against the unprepared canonical operation, including adversarial rounding controls |
| Shared paged/varlen attention | Preserve existing component gates, including atol .002 / rtol .02 where declared; independently check the actual same-Q/K/V geometry against FP64 and retain every maximum/nonfinite failure |
| Constants, packing, embedding copies, KV/state transfer, inactive-row and completion metadata | Exact bytes/integers; poisoned inactive values cannot influence active rows |

The linear envelope in `qualify/asr_linear_bounds.py` admits normal-or-zero finite
BF16 operands and exact normal-or-zero FP32 products. For n=K plus one bias update,
it includes gamma32(n) times the absolute product/bias sum, an explicit
`n*FP32tiny/(1-n*u32)` allowance for partial-sum FTZ, separate FP64 reference error,
and the supremum of BF16 rounding error across the **entire** resulting interval.
It rejects possible FP32 partial overflow and BF16 output interval overflow.
Subnormal operands/product underflow are separate classes, not silently accepted.

This envelope is conditional on at most n RNE FP32 updates and final RNE BF16
conversion with gradual subnormals. Closing Torch's reduced-reduction option
does not prove those properties. In particular, the [PTX MMA
specification](https://docs.nvidia.com/cuda/archive/13.1.1/pdf/ptx_isa_9.1.pdf)
does not specify every accumulation rounding/order/subnormal detail. Qualify the
actual candidate arithmetic and retain unsupported classes explicitly; a Tensor
Core name or one calibration set is not a universal arithmetic proof.

Required controls include binade-crossing intervals, zero/subnormals, premature
rounding before bias, FTZ cancellation of normal products, both overflow classes,
mixed finite/NaN/Inf outputs, wrong dtype and wrong row routing. A nonfinite value
fails before any norm, ratio or aggregate is calculated.

## Network and decision gates

### Small-batch normalization and history replay

The original canonical RMSNorm's FP32 mean can select a different reduction
layout for eight physical decode rows and 64 prefill rows. An unchanged native
package reproduced a Chinese 100-frame readmission failure: the original prompt
KV remained exact, but generated-history KV first differed at layer 25. Copies
of existing compiled nodes reproduced all 56 KV planes and logits exactly both
before and after observation. At layer 24 post-attention normalization, equal
BF16 inputs and FP32 squares produced means differing by `6.103515625e-5`,
21 different BF16 normalized values and 20 different affine values. Earlier
means already had smaller FP32 differences; this is a demonstrated BF16
divergence, not the first FP32 difference in the network.

New ASR preparation composes `CanonicalRMSNorm` for hidden-state and final
normalization. It pads only an FP32 square operand with fewer than 64 physical
rows, computes the row-local mean and selects the original rows. It preserves
the checkpoint gain object, epsilon, FP32 normalization and both BF16 rounding
nodes. Larger extents retain the original expression; Q/K normalization and
resume's already-normalized hidden state retain their existing composition.
This does not change previous packages' captured graphs.

For the actual failing operand, padding to 64 rows reproduced the observed
prefill mean, FP32 normalized values, BF16 normalized values and affine output
exactly. The small operator fixture preserves that operand and its unchanged
gain. [Component CUDA controls](../benchmarks/qwen3_asr/results/canonical-rmsnorm-cuda.json)
pass original-implementation negative control, eager and complete-forward
compiled capture over 17 physical row capacities, poisoned neighboring rows,
guards and repeated replay without allocator counter changes. Coupled prepared
attention/projection controls also pass.

A fresh ordinary CLI export subsequently compiled 43 forwards and captured 55
graphs. After the preparer exited, independent Rust package serving passed all
20 frozen inputs in 60 generated-history cohorts and 62 batch/slot cohorts
(190 natural requests). Complete same-path KV, BF16 logits and tokens, B1/2/4/8,
row permutations, partial batches and slot reuse pass the unchanged frozen
scalar bounds. Readiness barriers in these controls are test-only; they do not
qualify automatic HTTP batching. The fixed 256-request recognition set also
passes with natural stops and no failed or truncated requests; all texts equal
the independent canonical reference. Actual response token counts total 5,147;
this recognition collector does not observe raw token IDs. The
[raw measurements and scope](../benchmarks/qwen3_asr/results/native-rmsnorm-twenty-replay-run.json)
retain the package, source and reference identities.

These finite results close the reproduced generated-history failure for this
package and corpus. Complete new-norm intermediate-node/operator, full
CUDA/native/lifecycle, current API/cache, historical VISION and sustained
performance gates remain separate and open. Strict physical-release diagnostic
failures remain failures. No reference or budget changes are authorized by this
normalization change.

Other registered recipes retain canonical or ATen-mean normalization in some
decoder, draft or vision paths: Gemma4, Muse Spark, MiMo V2, Qwen3.8 and dense,
DeepSeek V4.1 heads, MiniMax M3 draft, Nemotron draft/vision, Kimi DSpark and
GLM vision. These paths are untested for this particular row-geometry effect;
the ASR result does not establish their failure or authorize replacing their
different rounding contracts. Their existing prepared norms and other pooling
reductions are separate mechanisms. Investigate equal actual operands and each
model's existing cross-capacity contract before deciding a local composition.

### Frozen network comparisons

[The numerical fixture configuration](../benchmarks/qwen3_asr/numerical-contract.json)
freezes a max-absolute and relative-L2 scalar for every named BF16 development
probe, and separately for every teacher-forced logit row. L2 always uses the
canonical denominator. Keep all valid elements, including near-zero outputs.

The empirical budget is the independent canonical/SDK discrepancy plus **one
final BF16 adjacent grid step**: the maximum step for the absolute budget and
the L2 norm of pointwise steps for the relative budget. The larger adjacent step
is used at a binade; zero/subnormal spacing is explicit. A wholly zero canonical
and independent stage retains a zero budget. This additional quantization unit
is an explicit downstream empirical allowance for a third valid reduction path;
it is not a network error theorem or permission to skip a rounding node.
The grid term is evaluated at canonical values, not as a supremum across the
canonical–SDK interval; it does not promise to admit every other legal path.
Local operator, state, determinism and recognition gates still apply independently.
No candidate output participates in generating this configuration.

`qualify/asr_candidate_teacher.py` keeps its historical inference-mode default
for reproducibility. To diagnose an ordinary CLI package, pass
`--preparation-mode no-grad`, the package's `--snapshot-budget-bytes`,
`--preparation-headroom-bytes` and `--retain-graphs`. The collector records actual
inference/grad state, thread count, reduction settings and budgets. No-grad
explicitly disables any outer inference mode, since tensor dispatch identity
must match compiled bindings. These options change diagnostic preparation,
never the independent references or frozen thresholds.

`qualify/asr_candidate_teacher.py --full-prefill --prefill-network-nodes` adds
preparation-only copies of the actual decoder module nodes. The collector remains
under GPU qualification. Every observed graph must first reproduce the production
graph's outputs and complete bounded persistent state exactly. The node check
requires every frozen `thinker.model.*` stage, its original BF16 dtype and shape;
only removal of the reference's first singleton batch dimension is allowed.
Missing nodes, additional undeclared names and nonfinite values fail. SiLU module
inputs/outputs absent from the original named-stage contract are retained as
unbudgeted attribution, with finiteness checks; they do not acquire new empirical
budgets or replace independent local operator checks. Observers execute serially,
share bounded diagnostic destinations across capacities and read before reuse.
These graphs are never exported or timed, and their node checks do not set the
full-model numerical qualification flag.

`--encoder-network-nodes` separately adds preparation-only copies of all 423
frozen audio stages. CNN candidates are retained by declared physical batch and
width, selected using the existing whole-input geometry, then cropped to the
original node axes. CNN/conv_out concatenation retains padded tail chunk positions;
encoder-layer and projector concatenation uses valid window feature counts.
The additional audio graph must bridge complete bounded persistent state,
publication records and the existing projection observations before its nodes
are accepted. Failed fences retain owners and snapshots without restoring shared
storage. These observers are under GPU qualification, with no new scalar budgets.
`--focus-network-nodes` visits every frozen input on the B1 baseline to diagnose
observers before repeating the full schedule matrix. It explicitly leaves complete
teacher/state and full-prefill qualification false; final acceptance requires the
ordinary B1/2/4/8, row-order and retirement matrix.

Require **zero** top-1 differences on the last prefill row and all declared
teacher rows. Compare every row directly, with lowest-ID BF16 ties; do not remove
small-margin rows. Report every full-prefill top-1 difference too, including audio
and template positions that are not actual generation decisions, with both
engines' logits and margins. Those positions retain their floating-point gates.
Full natural token sequences, first divergence and recognition differences are
separate observations, never inferred from a few teacher rows.

Same-path repeated/fresh-process values and released Flare schedules are held to
byte identity across B=1/2/4/8, partial batches, row orders, slot/generation reuse,
unrelated completion and cancellation. Cross-engine tolerance cannot excuse a
failure of this determinism requirement. Preserve every existing model's
bitwise, package and numerical regression gates.

## Bounded encoder attention component

`PreparedFlashWindow` captures the canonical ATen variable-length flash operation
over one bidirectional BF16 window. Prepared Q/K/V/output storage has a fixed
capacity; CUDA `cu[2]` supplies `[0, N]`. The caller validates `0 <= N <= capacity`.
The component defines every unused output row as zero, including `N=0`, and
rejects output/input storage overlap. It adds no serving-time Python or plan
selection.

Real compiled replay tests cover capacities 13 and 104, empty/full/interior live
lengths, poisoned tails, repeated transitions and allocation counters. Valid
rows match the pinned canonical SDPA operation exactly. A deliberately wrong
length exposes nonfinite values and the same correct replay recovers. Two
synthetic exported packages also replay in Rust after their preparer exits.
Failure controls enqueue real graph work before rejecting its fence at replay,
storage-context exit and recorder-context exit; retained owners survive exception
disposal and GC, while disabling retention triggers the expected owner failure.

These are component and package-boundary checks. They do not establish complete
encoder arithmetic, ASR recognition, asynchronous request lifetime, performance
or compatibility with every previously published package. Full model stage and
teacher-logit scalar gates remain separate.

## Whole-input encoder geometry

The whole encoder-input length and the current attention-window length are
distinct typed values. The prepared CNN candidate preserves its BF16 convolution,
BF16 bias-add and BF16 GELU nodes while selecting measured physical batch
results from the whole chunk count. A tail with one current chunk does not
become an original one-chunk convolution. Both fixed graph capacities retain
the same width masks and preserve the original short-width path.

The singleton linear candidate is selected only when the whole input has one
encoder feature. An 801-frame clip has 105 features: its last one-feature
attention window keeps the regular projection result. Both capacities use the
whole-input feature count, including a legal small input in the larger bucket.
All declared projection products execute; their byte-preserving selection is
described in [canonical projections](canonical-projections.md).

These are numerical candidates requiring the unchanged stage, complete-model,
capacity-transition and native-package gates. Window padding still need not
reproduce every original flattened linear shape. No numerical, recognition or
performance acceptance follows from a local convolution or singleton control.

## Paged causal attention component

`PreparedPagedFlashAttention` captures ATen flash attention over BF16 paged K/V
and explicit device cumulative query/key lengths. Page size must be a multiple
of 256. Output storage is separate from every input; inactive packed output
rows are exactly zero. Compilation, scratch allocation and capture finish in
preparation, and exported executables replay in Rust after Python exits.

The pinned [ATen GQA shortcut](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp)
for `max_seqlen_q == 1` reshapes queries as one row per request and sets the
query cumulative-length pointer to null. That precondition fails for packed
batches with empty request slots. The component uses a conservative launch
capacity of at least two for GQA, retaining the ordinary varlen path and its
actual device lengths. It does not invent a second query.
The decode empty-slot regression fails with the shortcut and passes with these
explicit lengths.

The same pinned implementation's
[multi-split combine kernel](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/csrc/flash_attn/src/flash_fwd_kernel.h#L1156)
addresses output by batch and row rather than cumulative packed-query offsets.
Its varlen setup does not populate the dense output batch stride. Multiple
requests can therefore write overlapping rows; promoting a one-row GQA launch
to two also makes its dense row bound exceed a one-row output allocation.
The component uses a single split throughout its catalog. In addition to the
packed-row restriction, automatic splitting changes the represented BF16 result
at 122/123 live prefill queries on SM121; the frozen square-SDPA controls agree
exactly with one split. This component's uniform arithmetic is not the current
ASR request-independent attention path described below, or a performance choice.

The upstream varlen interface documents exact packed query storage. Flare's
fixed-capacity query tails and conservative GQA launch are a separately tested
use of this pinned implementation, not an upstream compatibility guarantee.
Tail output is cleared explicitly; that does not establish defined values for
every unused internal accumulator. A dependency change must requalify these
geometries, including physical B1/2/4/8 and the smallest one-row allocation.

An exact-total-query control separately uses two requests with two queries
each, without empty slots or capacity tails. Five single-split replays pass an
independent FP64 bottom-right-causal target; five forced two-split replays leave
the second request's raw return rows at their prefilled sentinel and fail the
same arithmetic gate. This demonstrates missing output writes, not a complete
memory-safety proof of ATen's internal allocation.

CUDA tests cover physical and logical B1/2/4/8, changing slots/row order, empty/valid
transitions, poisoned unrelated storage, guards, repeatability and allocation
counters. Random prefill tests include unequal query/key lengths, real cache
prefixes, permuted pages and 128/4096 query capacities, with a full contiguous
canonical lower-right-causal comparison. Explicit FP64 softmax independently
checks every row of short cases and selected boundary rows at capacity 4096;
it is not an all-element FP64 qualification at that capacity. Wrong key lengths
expose poisoned values and must fail the finite gate before exact recovery.
Three native package cases separately cover empty decode, separated live slots
and a prefill tail after preparation exits. Their constant-value outputs test
export/replay mechanisms; random causal alignment is checked separately.

Shared `PreparedBatch` must receive the same page size as the recipe's KV
writer, pool and contract. Real captured validation tests exercise both 128 and
256 token pages, one-page 129–256-token prefills with 256-token pages, the decode page boundary,
the 8192-token endpoint and required-page rejection. These component gates do
not establish complete ASR scalar, quality, asynchronous lifecycle or serving
performance acceptance. Existing recipes and historical packages retain their
own prepared page geometry and operations.

## Request-independent attention and execution errors

`PreparedRequestFlashAttention` preserves each request's original FA2 split and
partition geometry through a preparation-time catalog of fixed calls. One-query
GQA uses the native Q16/KV8 shortcut rather than repeating KV heads. General
queries use right-aligned physical padding and explicit valid query rows, so
every split partial is written before combination. Device metadata selects
represented BF16 output bits; all catalog calls execute, including unselected
ones with zero used keys. This is an arithmetic baseline with substantial
launch/storage cost, not an optimized attention implementation.

The catalog reproduces the pinned float32 occupancy heuristic. Local tests
cover B1/2/4/8, holes and reversed rows, actual cached lengths through 8192,
query/key asymmetry, nonfinite operands, malformed lengths/pages, recovery and
allocation stability. Three synthetic packages replay after preparation exits.
These checks do not qualify every general-query geometry near context 8192.

An explicitly prepared `num_splits=1` variant uses one full-key-capacity family
and at least two physical query rows, with dummy queries preceding the live
rows. This avoids the one-query GQA shortcut while preserving the live query's
lower-right causal position. The default catalog and its existing bytewise
regressions remain unchanged. This is a separate arithmetic configuration;
neither variant promises bitwise equality to the other.

The first four continuation tests incorrectly required byte equality to
standard fixed-length SDPA. All four batch cases failed first at 165 keys.
The retained [path attribution](../benchmarks/qwen3_asr/results/request-attention-path-attribution.json)
shows 588 differing BF16 values there, with maximum difference 0.000244140625.
Requested nonpaged varlen split counts 0/1/2, paged split1, padded queries and
capture agree with each other and satisfy the unchanged FP64 bounds. That
initial evidence did not isolate the cause: the pinned nonpaged varlen entry
ignores the requested split count for these multi-query inputs.

The subsequent [partition observations](../benchmarks/qwen3_asr/results/request-attention-partition-extent.json)
retain actual CUDA profiler kernel names and all four frozen operand cases.
The fixed-length 165-key reference executes an N128 split-KV kernel and FP32
combine. Paged max-key capacity 165 or 256 with split2 reproduces its final row
exactly; max-key 8192 with split2 differs by 322 BF16 values because its partitions
span 4096 keys. Fixed max-key 8192/split64 also reproduces the 165-key row, while
differing by 418/817/837 values at 123/443/8192 keys. Five captured Q2 replays per
variant preserve bytes and allocation counters. These controls identify the
partition mechanism; they do not qualify a model configuration or establish
fragment, readmission, slot or recognition guarantees. No model threshold or
reference changes as a result.

The corrected tests separately require byte identity against captured prefill
with the same fixed-split configuration and numerical agreement with independent
standard SDPA and FP64 softmax. Both numerical comparisons retain absolute
tolerance .002 and relative tolerance .02. B1/2/4/8 share CPU baseline rows;
prefill capacities 512/4096/8192, repeated replay, row permutations, holes,
inactive poison, storage guards, input immutability and allocation counters are
checked. The 22 CUDA tests pass. Six synthetic packages each load in a separate Rust
process for five replays after their preparer exits. These results do not establish
model logits, full request scheduling, recognition quality or service performance.
The original four failures remain failures of that first comparison contract;
no existing default-path or model scalar threshold was relaxed.

The component clears and writes global/per-row execution errors on every
replay. Shared `PreparedGeneration` optionally reduces current-generation row
errors before publishing hidden state, consuming tokens or sampling. Failed
rows retain consumed/generated counters, RNG, pending token, histogram and saved
hidden state; already written private KV is not rolled back. Existing consumers
without an error operand retain their previous computation. Draft-context writes
are explicitly rejected with this operand until their failure semantics are
integrated. The tests' scaled four-rank reduction input is a mapping control,
not execution of a distributed collective.

Real attention fault tests cover sampling and nonsampling prefill/decode,
reversed request slots, stale generations and preserved prior errors. A separate
signed synthetic package runs through Rust after its preparer exits: token
operands trigger prefill/decode attention faults, HTTP 500 and SSE error frames,
no failed-state checkpoint publication, valid-prefix reuse and request-slot
recovery. Concurrent HTTP arrivals also preserve a healthy request; they do not
prove the two requests shared one B2 GPU graph. The native process drains after
SIGTERM. CPU scheduler tests separately hold a failed slot's storage until its
last submitted decode completes and reject a late completion after reuse.

The focused 75-test CUDA/native selection and this native error fixture pass.
Earlier failed fault-injection and fixture-start attempts remain retained. These
results are component/lifecycle evidence, not final ASR quality, whole-change
qualification, every recipe's execution, or comparative serving performance.

## Original-prompt fragment geometry

The [six fixed-projection rejection controls](../benchmarks/qwen3_asr/results/prefill-prefix-projection-rejections.json.gz)
retain every node/head/scalar observation on the same frozen 19-token input.
Existing BF16 SIMT128/SIMT256/MMA/TC, fixed-eight canonical and singleton canonical
prefill arithmetic fail 54/69/112/122/42/58 network nodes respectively. Original
decode/head paths and thresholds remain unchanged. Scalar logit success in five
of these candidates does not waive their node failures. This rejects those
measured configurations, not every possible fixed arithmetic implementation.

The [same-operands attention control](../benchmarks/qwen3_asr/results/original-prompt-attention-fragments.json)
compares a complete 165-query invocation with its actual `128+37` fragments.
Automatic current-chunk geometry changes 25,629 BF16 elements. Keeping the
original `[192 query rows, 2 splits, 256 key bound]` family makes all three
outputs byte-equal to the full independent SDPA result. Actual consumed key
lengths remain 128 and 165; this does not read future KV. Five compiled/captured
replays are byte-identical and allocation-stable. Poisoning future keys after
priming a larger used-key extent leaves the prefix unchanged; poisoning an
active key reaches the error/nonfinite guard, and restoring it recovers exact
bytes. This is one frozen synthetic operand geometry, not a model configuration.

`PreparedPositionedProjection` initializes a full physical staging matrix from
only the current contiguous absolute-position fragment. It reuses
`PreparedCanonicalProjection`, selects small extents from the immutable sequence
length, and gathers back only the live fragment. All full extents execute; this
is an arithmetic mechanism, not reduced GEMM work. The
[196-projection observation](../benchmarks/qwen3_asr/results/positioned-projection-fragments.json)
uses one frozen 165-token input's independent operands for all seven projections
in all 28 layers. Fixed 8192-row full and `128+37` calls are mutually byte-equal
and equal the corresponding independent outputs, with five allocation-stable
captured replays each. These are uncompiled component captures; no complete
network, generated-history readmission or native request determinism is inferred.

The [complete affected CUDA/native selection](../benchmarks/qwen3_asr/results/positioned-projection-native.json)
passes all 29 tests, including every previous canonical fixture and four new
full/prefix/tail/empty component packages. Their Python preparers exit before
Rust reconstructs the graphs; five independent replays poison and verify the
complete output. These synthetic packages keep N and positions as captured
constants, so they do not test the proposed native value source or cache binding.
The preceding 25-pass/4-failure run is retained: its private compiler-worker
wheel lacked the new component, and all four preparations failed before export.
Rebuilding only that private wheel resolves the import; no shared dependency
or numerical threshold was changed.

Independent review found a real metadata-guard defect: a zero-dimensional int64
start plus an int32 count can narrow and wrap. Retained pre-fix CUDA execution
returns finite zero for invalid starts `2^31-1`, `2^32` and `2^63-1`, so the
intended finite gate fails. Explicit int64 remaining-length comparisons reject
these inputs in the regression. An earlier collector's wrong 40-token identity
failed its 165-row reshape before the first projection; that failed attempt is
retained separately and is not counted as coverage.

The [design proposal](qwen3-asr-design.md#immutable-original-prompt-values-and-proposed-graph-geometry)
describes the additional native immutable-input and real cache dependency needed
before using this mechanism in serving. Neither is implemented by these component
controls. Existing full-model, native batch/slot and performance gates remain open.

## Native logits and frontend boundary diagnosis

The ignored native Worker test authenticates an exported package, verifies its
weights and resolves the actual shared sampler's BF16 operand from signed graph
records. After a completion and zero outstanding work, it saves physical logits
before inspecting row/epoch metadata or testing finiteness, argmax and history.
It uses natural generation, with no teacher-token injection. Complete frozen
SDK/chat prompt IDs are required; the transcription endpoint has its separately
documented empty-system-turn contract.

The planned matrix has four original PCM inputs, B1 baselines, B2/4/8 cohorts,
permutations and partial rows. Actual full logical decode batches are required.
The initial package run fails before completing that matrix: the first mixed
cohort changes a long input's first BF16 logit row. Two minimized repeats fail
at the same boundary. Identical top-1 decisions do not satisfy byte stability.
Three B1 teacher rows also exceed unchanged scalar budgets. Missing subsequent
cohorts remain missing, rather than passing on the completed subset.

Audio boundary observation saves typed GPU Mel uploads and published feature,
generation and count storage before another submission. Its four complete B1
logit traces are byte-identical to the original observer, but its extra reads
change multi-request scheduling timing. It is a diagnostic, never a serving
performance path. B1 input Mel has small BF16 differences from the independent
SDK input despite passing the separate FP32 frontend budget; the published
encoder features differ too. This identifies a boundary for further isolation,
not a complete causal explanation or a waived model gate.

The test requires `FLARE_PROMPT_TEST_PACKAGE`, `FLARE_PROMPT_TEST_PUBLISHERS`,
`FLARE_NATIVE_LOGITS_CASES` and a new `FLARE_NATIVE_LOGITS_OUTPUT` directory.
Cases contain `id`, WAV `path` and `wav_sha256`, `language`, frozen `prompt_ids`
and `teacher_history`. The package's signed recipe identity, runtime cache
identity and authenticated envelope digest are distinct report fields. The
reader recognizes the measured ordinary ABI6 sampler only; this diagnostic
restriction does not change engine loading or historical package compatibility.

The observer also accepts the complete twenty-input frozen set. Its separate
schedule repeats each input at full B1/2/4/8, reversed rows and seven live rows
in B8, using explicit test-only readiness barriers. Complete 24-byte request
scopes survive slot changes and readmission. The CPU reader reconstructs every
request trace from physical completed logit rows and requires per-input batch,
row and slot/generation witnesses. The four-input pilot retains its original
schedule and status; its earlier results do not qualify the twenty-input path.

The complete path records all 56 authenticated KV planes for every Prefill
fragment, including fragments that emit no token, and every frozen teacher
row. The reader requires continuous consumed ranges from cold admission or the
actual G2 restore endpoint, finite complete planes and exact B1 prefix bytes.
Snapshot completion records, residency before/after eviction, readmitted
generations and every device poison file are reconciled with actual completed
Prefill fences. Poison must leave the committed prefix unchanged and contain
the declared BF16 NaN pattern before the submitted graph rewrites it. These
are diagnostic state checks; there is no new cross-engine `cache.*` budget.
CPU corruption controls and successful compilation do not establish that the
twenty inputs have passed native GPU, complete operator, API or performance gates.

The explicit `FP32_FMA` CPU frontend has a separate finite qualification:
[40 native process measurements](../benchmarks/qwen3_asr/results/native-frontend-fma.json)
cover the same twenty frozen inputs, in forward and reverse order. Every call
passes the unchanged FP32 budgets and matches the independent SDK BF16 input
boundary. [Legacy byte controls](../benchmarks/qwen3_asr/results/native-frontend-legacy-bytes.json)
show the newly linked binary preserves all twenty original legacy outputs.
These checks execute Rust with standalone native FFT/math dependencies, with
no Torch runtime linkage. They do not waive the outstanding native model scalar,
batch/slot equality, package, quality or performance gates.

Reproduce the CPU gate with `python -m qualify.asr_native_frontend`, selecting
`--numerics fp32-fma`, the native `test_audio` binary, the frozen FP32 reference,
development manifest/PCM root and all three SDK observation directories. The
default `legacy` retains the prior gate. BF16 differences are reported separately;
they are not a substitute for model-level thresholds.
