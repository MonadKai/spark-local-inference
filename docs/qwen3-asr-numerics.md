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
