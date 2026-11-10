# Captured canonical BF16 projections

The ASR numerical diagnostic found two distinct projection problems on SM121:
padding small matrices changes some BF16 outputs, and the vector-bias library
path loses a represented residual in a sparse bias control. The independent
checkpoint inputs, references and numerical limits remain unchanged.

`PreparedCanonicalProjection` binds contiguous BF16 source, checkpoint weights,
optional bias and caller-owned output. The no-bias operation uses `F.linear`.
The bias operation uses a two-dimensional row bias in `addmm`, retaining one
linear-plus-bias BF16 output node. In the K=7680 sparse control, the vector-bias
operation returns zero where the represented result is 1/256; the row-bias
operation returns 1/256. These observed end values do not establish the NVIDIA
library's internal sequence of rounding instructions. FP32 reduction and the
separate split-K policy are recorded with the measurements.
The component does not set global backend policies; its caller fixes and
qualifies those policies before preparation.

Preparation may declare ascending smaller row extents. Every declared matrix
product executes in the captured graph; a device count selects the smallest
extent that covers the live rows. Selection copies BF16 values and clears
inactive output rows. There is no host count read, runtime compilation or
serving-time algorithm selection. Source/weight/output geometry and writable
storage aliases are checked when binding, and again for an actual eager call.
The caller owns validation of live-count ranges before replay.

The measured ASR prefill candidate declares extents 19, 27 and 32 below the
ordinary physical capacity. Seven frozen development cases and all 196 decoder
projections supply 1,372 exact logical-shape controls and 17,052 eager physical
extent observations. These measurements motivate those extents; they do not
prove canonical equality for every possible live count. Four represented
layer-2 V inputs with 19/20/31/32 rows reproduce their independent references
exactly through `ForwardCache` and CUDA capture. Direct padding to 64 rows fails
the same four controls with 1/1/3/1 differing elements. The captured boundary
test also covers changing counts, poisoned tails, active-row nonfinite failure
and recovery, guards, repeated replays and allocation counters.

Optional zero-filled physical staging keeps a declared matrix extent across
request capacities. `PreparedRowwiseProjection` instead issues an independent
singleton product for each physical row. This is an arithmetic attribution
baseline: it deliberately gives up cross-row weight reuse and is not a batched
GEMM optimization. Neither component establishes native ASR support, complete
model numerical acceptance, recognition quality or a performance improvement.

The complete raw scalar observations, including unsupported bias-operation
attempts and negative controls, are in
[canonical-projection-sm121.json](../benchmarks/qwen3_asr/results/canonical-projection-sm121.json).
The ordinary CUDA regression lives in
`tests/components/test_canonical_projection_cuda.py`. All 21 of its cases passed
in the combined 31-case projection/paged-attention selection. Five synthetic
projection packages load and replay in an independent Rust process after their
preparers exit: empty output, three subrow/full bias cases at K7680/N1024, and
one rowwise case. Each requires five byte-exact replays, poisoned output writes,
fixed addresses, no Python/Torch mappings and a successful native drain. This
does not establish an ASR model package, HTTP or shared scheduler behavior.

A second fresh probe process repeated all four real layer-2 V controls exactly,
including explicit repeatability assertions. Full model, native model-package,
determinism and serving gates remain independent requirements.
