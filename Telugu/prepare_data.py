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
        self.spk_col = d.get("speaker_column") or next(
            (c for c in ("speaker_id", "speaker", "spk_id", "client_id") if c in cols), None)
        print("audio:", self.audio_col, "| text:", self.text_col, "| speaker:", self.spk_col)

    def __len__(self):
        return len(self.ds)

    def row(self, i):
        r = self.ds[i]
        a = r[self.audio_col]
        b = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
        return {"bytes": b, "text": r[self.text_col],
                "spk": r[self.spk_col] if self.spk_col else None, "path": a.get("path") or ""}


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


# ------------------------------------------------------------------ selection
def select_split(meta, train_h, val_h, per_speaker, seed):
    rng = random.Random(seed)
    groups = {}
    for m in meta:
        groups.setdefault(m["spk"] if per_speaker else "all", []).append(m)
    ng = len(groups)
    for name in sorted(groups):
        items = groups[name]
        rng.shuffle(items)
        vb, tb = val_h * 3600 / ng, train_h * 3600 / ng
        v = t = 0.0
        for m in items:
            if v < vb:
                m["split"], v = "val", v + m["dur"]
            elif t < tb:
                m["split"], t = "train", t + m["dur"]
            else:
                m["split"] = ""
    return [m for m in meta if m["split"]]


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
    meta = []
    for i in tqdm(range(len(src)), desc="scan"):
        r = src.row(i)
        info = sf.info(io.BytesIO(r["bytes"]))
        dur = info.frames / info.samplerate
        raw_text = str(r["text"])
        if d.get("reject_regex") and re.search(d["reject_regex"], raw_text):
            continue
        txt = fe.clean_for_corpus(raw_text)
        n_words = len(txt.split())
        if not (d["min_dur"] <= dur <= d["max_dur"] and d["min_words"] <= n_words <= d["max_words"]):
            continue
        meta.append({"idx": i, "dur": dur, "text": txt, "raw_text": raw_text, "spk_raw": r["spk"], "path": r["path"]})
        del r
    print(f"{len(meta)} / {len(src)} utterances pass the filters, {sum(m['dur'] for m in meta)/3600:.2f} h")

    spks = [m["spk_raw"] for m in meta]
    if all(s is None for s in spks):
        spks = infer_speakers(meta)
    for m, s in zip(meta, spks):
        m["spk"] = _safe(s)
    sel = select_split(meta, d["train_hours"], d["val_hours"], d["per_speaker_budget"], d["seed"])

    rows = []
    for m in tqdm(sel, desc="write"):
        r = src.row(m["idx"])
        y, in_sr = sf.read(io.BytesIO(r["bytes"]), dtype="float32", always_2d=True)
        y = y.mean(1)
        if in_sr != sr:
            y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
        peak = float(np.abs(y).max())
        if peak > 0:
            y = y / peak * 0.95
        uid = f"{m['spk']}_{m['idx']:05d}"
        os.makedirs(os.path.join(raw, m["spk"]), exist_ok=True)
        sf.write(os.path.join(raw, m["spk"], uid + ".wav"), y, sr, subtype="PCM_16")
        with open(os.path.join(raw, m["spk"], uid + ".lab"), "w", encoding="utf-8") as f:
            f.write(m["text"] + "\n")
        rows.append([uid, m["spk"], m["split"], round(m["dur"], 3), m["idx"], m["text"]])

    for s in ("train", "val"):
        with open(os.path.join(split_dir, f"{s}_list.txt"), "w", encoding="utf-8") as f:
            for uid, spk, sp, *_ in rows:
                if sp == s:
                    f.write(f"{spk}|{uid}\n")
    with open(os.path.join(split_dir, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "speaker", "split", "duration_s", "source_index", "text"])
        w.writerows(rows)
    for s in ("train", "val"):
        sub = [r for r in rows if r[2] == s]
        print(f"{s}: {len(sub)} utts, {sum(r[3] for r in sub)/3600:.2f} h, {len({r[1] for r in sub})} speakers")
