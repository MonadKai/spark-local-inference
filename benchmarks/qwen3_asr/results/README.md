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
