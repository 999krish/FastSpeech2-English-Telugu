# Changes w.r.t. ming024/FastSpeech2
* `train.py`, `preprocess.py`, `inference.py` rewritten: one combined `config.yaml` (sections `data`, `preprocess`, `model`, `train`).
* `preprocessor/preprocessor.py`: explicit utterance-disjoint train/val lists; pitch/energy statistics from the TRAIN split only;
  MFA-3 silence labels ('') mapped to `sp`; durations forced consistent with mel frames; NaN / missing-TextGrid handling
  (the original crashed when a TextGrid was missing); `max_frames` filter.
* `audio/stft.py`, `audio/tools.py`: re-implemented with `torch.stft` + `librosa.filters.mel(..., keyword args)` (original breaks on librosa>=0.10). `audio_processing.py` removed.
* `utils/model.py`: `torch.load(..., map_location='cpu', weights_only=False)` for recent PyTorch.
* `train.py`: CSV loss logging (window-averaged train loss, full validation loss), best / final / periodic checkpoints, wall-clock limit, no in-loop vocoder synthesis.
* `inference.py`: replaces `synthesize.py`; exports audio, phoneme-level duration/pitch/energy plots and CSVs.
* Removed: AISHELL3/LJSpeech/LibriTTS preprocessors, Mandarin front-end, tensorboard logging, demo files.
* `text/symbols.py`: added the Telugu Unicode block as phone symbols.
* `telugu_text.py`: Telugu normaliser, number/ordinal/time expansion and rule-based grapheme->phone parser.
