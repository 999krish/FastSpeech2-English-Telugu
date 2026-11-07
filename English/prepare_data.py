"""Dataset selection / sub-sampling / corpus writing (shared by English and Telugu).

Reads the `data:` section of config.yaml, loads the source corpus (HF hub or a local manifest),
keeps clean utterances, draws a random duration-budgeted TRAIN and VAL subset (disjoint
utterances), resamples to the model sampling rate and writes
    raw_data/<speaker>/<utt>.wav|.lab        (MFA corpus layout)
    split/train_list.txt, split/val_list.txt ("speaker|utt")
    split/manifest.csv                       (full record of what was selected and why)
"""
import csv
import importlib
import io
import json
import os
import random
import re

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def get_frontend_module(cfg):
    return importlib.import_module({"English": "english_text", "Telugu": "telugu_text"}[cfg["language"]])


def _safe(s):
    return re.sub(r"[^A-Za-z0-9]", "", str(s)) or "spk"


# ------------------------------------------------------------------ sources
class HFSource:
    def __init__(self, d):
        from datasets import Audio, load_dataset
        self.ds = load_dataset(d["hf_name"], split=d.get("hf_split", "train"), token=os.environ.get("HF_TOKEN"))
        self.audio_col = [k for k, v in self.ds.features.items() if isinstance(v, Audio)][0]
        self.ds = self.ds.cast_column(self.audio_col, Audio(decode=False))
        cols = self.ds.column_names
        print("columns:", cols)
        self.text_col = d.get("text_column") or next(
            (c for c in ("text", "transcript", "transcription", "sentence", "normalized_text") if c in cols), None)
        if self.text_col is None:
            raise RuntimeError(f"no transcript column found in {cols}. Set data.text_column in config.yaml, or download "
                               "Svarah from github.com/AI4Bharat/Svarah and set data.local_dir (SVARAH_LOCAL_DIR).")
        self.spk_cols = [c for c in (d.get("speaker_columns") or []) if c in cols]
        self.spk_col = d.get("speaker_column") or next(
            (c for c in ("speaker_id", "speaker", "spk_id", "client_id") if c in cols), None)
        print("audio:", self.audio_col, "| text:", self.text_col, "| speaker:", self.spk_col, "| speaker metadata columns:", self.spk_cols)

    def __len__(self):
        return len(self.ds)

    def row(self, i):
        r = self.ds[i]
        a = r[self.audio_col]
        b = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
        spk = r[self.spk_col] if self.spk_col else None
        if spk is None and self.spk_cols:                      # Svarah: speaker-level metadata tuple == speaker identity
            spk = "|".join(str(r[c]) for c in self.spk_cols)
        return {"bytes": b, "text": r[self.text_col], "spk": spk, "path": a.get("path") or ""}


class LocalManifestSource:
    """Svarah-style folder: svarah_manifest.json (one json object per line) + audio files."""

    def __init__(self, d):
        root = d["local_dir"]
        self.items = []
        with open(os.path.join(root, d.get("local_manifest", "svarah_manifest.json")), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    p = o["audio_filepath"]
                    self.items.append((p if os.path.isabs(p) else os.path.join(root, p), o["text"]))

    def __len__(self):
        return len(self.items)

    def row(self, i):
        p, t = self.items[i]
        return {"bytes": open(p, "rb").read(), "text": t, "spk": None, "path": p}


def load_source(d):
    if d.get("local_dir") and os.path.isdir(d["local_dir"]):
        return LocalManifestSource(d)
    return HFSource(d)


def infer_speakers(metas):
    """Svarah has no explicit speaker column on the hub: try to recover it from file names."""
    firsts = [re.split(r"[_\-]", os.path.splitext(os.path.basename(m["path"]))[0])[0] for m in metas]
    n = len(set(firsts))
    if 5 <= n <= 400:
        print(f"speaker ids recovered from file-name prefix: {n} speakers")
        return firsts
    print("WARNING: could not recover speaker ids -> treating the corpus as ONE speaker")
    return ["spk"] * len(metas)



# ------------------------------------------------------------------ audio quality / cleaning
def _frame_db(y, sr, win=0.025, hop=0.010):
    fl, hl = int(win * sr), int(hop * sr)
    if len(y) < fl * 3:
        return np.array([-100.0])
    n = 1 + (len(y) - fl) // hl
    idx = np.arange(fl)[None, :] + hl * np.arange(n)[:, None]
    rms = np.sqrt((y[idx].astype(np.float64) ** 2).mean(1)) + 1e-8
    return 20 * np.log10(rms)


def quality(y, sr):
    """(snr_db, clip_ratio): speech level (p90 of frame dB) minus noise floor (p10); fraction of clipped samples."""
    db = _frame_db(y, sr)
    return float(np.percentile(db, 90) - np.percentile(db, 10)), float((np.abs(y) > 0.99).mean())


def denoise(y, sr, strength=0.85):
    """Mild stationary spectral gating. The noise profile is the quietest 300 ms of the clip; the clip is left
    untouched when it has no real noise-only region (<10 dB below speech) so speech is never used as 'noise'."""
    try:
        import noisereduce as nr
        db = _frame_db(y, sr)
        k = 25                                                   # 250 ms
        if len(db) < 2 * k:
            return y
        # window whose LOUDEST frame is as quiet as possible -> guaranteed to contain no speech
        wmax = np.lib.stride_tricks.sliding_window_view(db, k).max(axis=1)
        i = int(np.argmin(wmax))
        if np.percentile(db, 90) - wmax[i] < 12:                 # no real noise-only region -> leave the clip alone
            return y
        hop = int(0.010 * sr)
        noise = y[i * hop: i * hop + int(0.25 * sr)]
        out = nr.reduce_noise(y=y, sr=sr, y_noise=noise, stationary=True, prop_decrease=strength, n_fft=1024)
        return out.astype(np.float32) if np.isfinite(out).all() else y
    except Exception:
        return y


def level_norm(y, sr, target_db=-22.0):
    """Same loudness for every utterance (p90 of frame RMS -> target dBFS) instead of peak normalisation,
    then a safety peak limit. Removes the per-utterance gain offsets the model cannot predict from text."""
    db = _frame_db(y, sr)
    gain_db = float(np.clip(target_db - np.percentile(db, 90), -20, 30))
    y = y * (10 ** (gain_db / 20))
    peak = float(np.abs(y).max())
    return (y * (0.97 / peak) if peak > 0.97 else y).astype(np.float32)

# ------------------------------------------------------------------ selection
def select_split(meta, train_h, val_h, per_speaker, seed, drop_noisy_frac=0.0):
    """Quality-aware, speaker-stratified, utterance-disjoint train/val selection.
      1. drop the noisiest `drop_noisy_frac` of the pool (lowest SNR estimate)
      2. if the pool is larger than the budget, keep the highest-SNR utterances
      3. every speaker with >=4 utterances gives ~val share of its audio to VAL (the rest to TRAIN) so that every
         validation speaker is also a training speaker; speakers with <4 utterances are train-only."""
    rng = random.Random(seed)
    pool = sorted(meta, key=lambda m: m.get("snr", 0.0))
    if drop_noisy_frac > 0:
        pool = pool[int(len(pool) * drop_noisy_frac):]
    budget = (train_h + val_h) * 3600
    if sum(m["dur"] for m in pool) > budget:
        keep, tot = [], 0.0
        for m in sorted(pool, key=lambda m: -m.get("snr", 0.0)):
            if tot >= budget:
                break
            keep.append(m); tot += m["dur"]
        pool = keep
    vshare = val_h / (train_h + val_h)
    groups = {}
    for m in pool:
        groups.setdefault(m["spk"], []).append(m)
    for name in sorted(groups):
        items = groups[name]
        rng.shuffle(items)
        for m in items:
            m["split"] = "train"
        if len(items) >= 4:
            tot, v = sum(m["dur"] for m in items), 0.0
            for m in items[:-2]:                         # always leave >=2 utterances in train
                if v == 0.0 or v < vshare * tot:
                    m["split"], v = "val", v + m["dur"]
    # top-up (small pools / smoke test): make sure validation is not (nearly) empty, still keeping >=2 train utts per speaker
    need = vshare * sum(m["dur"] for m in pool) * 0.6
    have = sum(m["dur"] for m in pool if m["split"] == "val")
    if have < need:
        for name in sorted(groups):
            tr = [m for m in groups[name] if m["split"] == "train"]
            for m in (tr[:-2] if len(tr) >= 3 else []):
                if have >= need:
                    break
                m["split"], have = "val", have + m["dur"]
    return [m for m in pool if m.get("split")]


def prepare(cfg):
    d, pre = cfg["data"], cfg["preprocess"]
    P = pre["path"]
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    fe = get_frontend_module(cfg)
    raw, split_dir = P["raw_path"], P["split_path"]
    if os.path.exists(os.path.join(split_dir, "manifest.csv")):
        print("split/manifest.csv exists -> prepare stage already done (delete split/ to redo)")
        return
    os.makedirs(raw, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    src = load_source(d)
    print(f"SOURCE AUDIT: dataset={d.get('hf_name')} split={d.get('hf_split')} rows={len(src):,}")
    meta = []
    rejected = {"text_regex": 0, "duration_or_word_count": 0, "clipped": 0}
    for i in tqdm(range(len(src)), desc="scan"):
        r = src.row(i)
        y0, sr0 = sf.read(io.BytesIO(r["bytes"]), dtype="float32", always_2d=True)
        y0 = y0.mean(1)
        dur = len(y0) / sr0
        raw_text = str(r["text"])
        if d.get("reject_regex") and re.search(d["reject_regex"], raw_text):
            rejected["text_regex"] += 1
            continue
        txt = fe.clean_for_corpus(raw_text)
        n_words = len(txt.split())
        if not (d["min_dur"] <= dur <= d["max_dur"] and d["min_words"] <= n_words <= d["max_words"]):
            rejected["duration_or_word_count"] += 1
            continue
        snr, clip = quality(y0, sr0)
        if clip > 0.002:                                   # audibly clipped recording
            rejected["clipped"] += 1
            continue
        meta.append({"idx": i, "dur": dur, "text": txt, "raw_text": raw_text, "spk_raw": r["spk"], "path": r["path"],
                     "snr": snr})
        del r, y0
    print(f"FILTER AUDIT: {len(meta):,} / {len(src):,} utterances pass; accepted raw duration={sum(m['dur'] for m in meta)/3600:.3f} h")
    print("Rejected counts:", rejected)
    if sum(m["dur"] for m in meta) < (d["train_hours"] + d["val_hours"]) * 3600 * 0.90:
        print("WARNING: the source pool itself may not contain enough usable audio for the requested split budgets.")

    spks = [m["spk_raw"] for m in meta]
    if all(s is None for s in spks):
        spks = infer_speakers(meta)
    desc = sorted({str(s) for s in spks})
    ids = {s: f"s{i:03d}" for i, s in enumerate(desc)}      # short folder-safe ids; description kept in the manifest
    for m, s in zip(meta, spks):
        m["spk"], m["spk_desc"] = ids[str(s)], str(s)
    print(f"speakers recovered: {len(ids)}  (expect roughly 100-120 for Svarah)")
    if len(ids) < 20 or len(ids) > 400:
        print("WARNING: unusual number of speakers - check data.speaker_columns in config.yaml")
    snrs = np.array([m["snr"] for m in meta])
    print(f"SNR estimate (dB): p10 {np.percentile(snrs,10):.1f} | median {np.median(snrs):.1f} | p90 {np.percentile(snrs,90):.1f}")
    sel = select_split(meta, d["train_hours"], d["val_hours"], d["per_speaker_budget"], d["seed"],
                       drop_noisy_frac=d.get("drop_noisy_frac", 0.0))

    rows = []
    n_prev = 0
    for m in tqdm(sel, desc="write"):
        r = src.row(m["idx"])
        y, in_sr = sf.read(io.BytesIO(r["bytes"]), dtype="float32", always_2d=True)
        y = y.mean(1)
        if in_sr != sr:
            y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
        if d.get("denoise", False):
            y_clean = denoise(y, sr)
            if n_prev < 3 and y_clean is not y:               # keep a few before/after pairs to listen to
                pv = os.path.join(os.path.dirname(os.path.normpath(raw)), "denoise_preview")
                os.makedirs(pv, exist_ok=True)
                sf.write(os.path.join(pv, f"{n_prev}_before.wav"), level_norm(y, sr), sr, subtype="PCM_16")
                sf.write(os.path.join(pv, f"{n_prev}_after.wav"), level_norm(y_clean, sr), sr, subtype="PCM_16")
                n_prev += 1
            y = y_clean
        y = level_norm(y, sr)
        uid = f"{m['spk']}_{m['idx']:05d}"
        os.makedirs(os.path.join(raw, m["spk"]), exist_ok=True)
        sf.write(os.path.join(raw, m["spk"], uid + ".wav"), y, sr, subtype="PCM_16")
        with open(os.path.join(raw, m["spk"], uid + ".lab"), "w", encoding="utf-8") as f:
            f.write(m["text"] + "\n")
        rows.append([uid, m["spk"], m["split"], round(m["dur"], 3), m["idx"], m["text"], round(m["snr"], 2), m["spk_desc"]])

    for s in ("train", "val"):
        with open(os.path.join(split_dir, f"{s}_list.txt"), "w", encoding="utf-8") as f:
            for uid, spk, sp, *_rest in rows:
                if sp == s:
                    f.write(f"{spk}|{uid}\n")
    with open(os.path.join(split_dir, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "speaker", "split", "duration_s", "source_index", "text", "snr_db", "speaker_desc"])
        w.writerows(rows)
    for s in ("train", "val"):
        sub = [r for r in rows if r[2] == s]
        print(f"{s}: {len(sub)} utts, {sum(r[3] for r in sub)/3600:.2f} h, {len({r[1] for r in sub})} speakers")
