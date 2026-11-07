# FastSpeech 2 TTS - English (Svarah) and Telugu (IndicTTS)

Two separately trained FastSpeech 2 models built on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2).
See `Report/Technical_Report.pdf` for the full write-up, `English/CHANGES.md` and `Telugu/CHANGES.md` for code changes.

```
English/  Telugu/   train.py  preprocess.py  inference.py  config.yaml  checkpoints/  (+ model code, frontends, test_sentences.json)
Results/<lang>/     training_loss.png  validation_loss.png  test_generated_audios/  variance_adaptor_plots/{duration,pitch,energy}/
Report/             Technical_Report.pdf
```

## Setup
```
pip install -r requirements.txt
# Montreal Forced Aligner (conda):  conda create -n mfa -c conda-forge python=3.11 montreal-forced-aligner
export MFA_CMD="conda run -n mfa mfa"        # or  micromamba run -n mfa mfa
export HF_TOKEN=...                           # Svarah is gated on the Hugging Face Hub (accept its terms first)
python -c "import nltk; [nltk.download(p) for p in ['averaged_perceptron_tagger_eng','cmudict','punkt']]"
```

## Reproduce (run inside `English/` or `Telugu/`)
```
python preprocess.py --config config.yaml --stage all        # prepare -> MFA dictionary -> MFA alignment -> mel/pitch/energy/duration
python train.py      --config config.yaml                    # losses -> logs/*.csv, checkpoints/ (best.pth.tar, final.pth.tar)
python inference.py  --config config.yaml --out ../Results/<English|Telugu> --speaker <speaker>
```
Data selection (English: **6.0 h train + 1.2 h validation raw audio ≈ 5 h / 1 h after MFA removes leading/trailing silence**; speaker-stratified and utterance-disjoint, seed in `config.yaml`) is recorded in `split/manifest.csv`. The English recipe additionally uses speaker ids derived from Svarah metadata, SNR filtering, optional denoising with post-MFA usable hours recorded in `preprocessed_data/summary.json`; early stopping is disabled for the 20k-step run.

## Checkpoints
`checkpoints/fastspeech2_<lang>_best_fp16.pth.tar` (best validation loss, fp16 weights, ~70 MB; `inference.py` loads it automatically when the fp32 file is absent).
The universal HiFi-GAN vocoder (`hifigan/generator_universal.pth.tar.zip`, from the reference repo) must be unzipped inside `hifigan/` before inference.
If you add the fp32 checkpoints, use Git LFS.
