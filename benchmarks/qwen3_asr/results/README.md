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
