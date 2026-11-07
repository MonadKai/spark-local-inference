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
