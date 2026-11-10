# Audio transport fixture

`en-61-70968-0058.wav` is the existing frozen qualification utterance
`61-70968-0058` from LibriSpeech `test-clean`, speaker 61, chapter 70968.
It contains 29,120 mono 16-kHz signed 16-bit samples. The official transcript
is `WILL YOU FORGIVE ME NOW`.

LibriSpeech was prepared by Vassil Panayotov with assistance from Daniel Povey,
from LibriVox recordings. Attribution and dataset details:
[OpenSLR SLR12](https://www.openslr.org/12/). The audio fixture is distributed
under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
It is a conversion from the corpus FLAC to PCM WAV, without editing the samples.
The associated MP3 is a lossy transport conversion, not sample-identical PCM:

```sh
ffmpeg -nostdin -v error -n -i en-61-70968-0058.wav -map_metadata -1 \
  -ac 1 -ar 16000 -c:a libmp3lame -b:a 64k en-61-70968-0058.mp3
```

These small files provide immutable public HTTPS and inline/multipart inputs
for actual endpoint checks. They do not replace the frozen recognition set or
establish codec-quality equivalence. Decode, feature and model comparisons
remain separate. The local native decoder observes 29,120 samples for each file;
matching sample counts do not imply matching waveform values.
