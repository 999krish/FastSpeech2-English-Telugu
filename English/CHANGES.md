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
* English data: Svarah speaker-level metadata is used to construct speaker IDs; speaker-stratified, utterance-disjoint train/validation split.
* Audio quality checks reject clipped recordings; denoising and fixed-percentage noisy-recording removal are disabled for this run.
* Feature extraction follows the Telugu notebook's baseline implementation: MFA silence handling, robust missing-TextGrid handling, duration/frame consistency, and train-only pitch/energy statistics.
* Model uses 4 encoder + 6 decoder FFT blocks, matching the Telugu notebook.
* Training uses the Telugu notebook's full-step loop: 30,000 steps, validation every 1,000 steps, no early stopping.
* `english_text.py`: English normalization + pronunciation lexicon with `g2p_en` fallback; English MFA dictionary builder.
* `english_text.py`: normaliser (numbers, times, years, AM/PM) + librispeech lexicon with `g2p_en` fall-back; MFA dictionary builder.
