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

Independent CPU reference tools in `qualify/asr_*_reference.py` check original
SDK/canonical tensor loading, slow tokenizer behavior and actual vLLM frontend
semantics. Their supplied model/source content identities must be established
separately where stated; successful loading or frontend comparison is not a
model forward, exported-package, HTTP, GPU, quality or performance result.

`long-speech-sources.json` describes supplemental real meeting speech. Its
acquisition and annotation-boundary qualification is separate from the primary
256 utterances. Missing safe boundaries must be recorded as acquisition failures,
without inventing a cut through speech.
