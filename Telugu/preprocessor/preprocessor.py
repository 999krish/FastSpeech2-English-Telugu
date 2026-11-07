"""Feature extraction (mel / pitch / energy / duration).  Patched version of
ming024/FastSpeech2 preprocessor/preprocessor.py:
  * explicit, leakage-free train / val split from split/{train,val}_list.txt
  * pitch / energy statistics computed on the TRAIN split only
  * MFA-3 silence intervals ("") are mapped to "sp"
  * robust to missing TextGrids, duration/frame mismatches and NaNs
"""
import json
import os

import librosa
import numpy as np
import pyworld as pw
import tgt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio

SIL = ["sil", "sp", "spn"]


class Preprocessor:
    def __init__(self, config):
        self.config = config
        p = config["preprocessing"]
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.split_dir = config["path"]["split_path"]
        self.sampling_rate = p["audio"]["sampling_rate"]
        self.hop_length = p["stft"]["hop_length"]
        self.pitch_avg = p["pitch"]["feature"] == "phoneme_level"
        self.energy_avg = p["energy"]["feature"] == "phoneme_level"
        self.pitch_norm = p["pitch"]["normalization"]
        self.energy_norm = p["energy"]["normalization"]
        self.max_frames = p.get("max_frames", 1000)
        self.STFT = Audio.stft.TacotronSTFT(
            p["stft"]["filter_length"], p["stft"]["hop_length"], p["stft"]["win_length"],
            p["mel"]["n_mel_channels"], p["audio"]["sampling_rate"],
            p["mel"]["mel_fmin"], p["mel"]["mel_fmax"])

    # ------------------------------------------------------------------
    def _read_list(self, name):
        with open(os.path.join(self.split_dir, name), encoding="utf-8") as f:
            return [tuple(l.strip().split("|")) for l in f if l.strip()]

    def build_from_path(self):
        for d in ["mel", "pitch", "energy", "duration"]:
            os.makedirs(os.path.join(self.out_dir, d), exist_ok=True)
        splits = {"train": self._read_list("train_list.txt"), "val": self._read_list("val_list.txt")}
        assert not (set(splits["train"]) & set(splits["val"])), "train/val overlap!"
        speakers = {s: i for i, s in enumerate(sorted({s for v in splits.values() for s, _ in v}))}

        out = {"train": [], "val": []}
        frames = {"train": 0, "val": 0}
        skipped = []
        pitch_scaler, energy_scaler = StandardScaler(), StandardScaler()
        train_files = set()

        for split, items in splits.items():
            for spk, base in tqdm(items, desc="features-" + split):
                tg = os.path.join(self.out_dir, "TextGrid", spk, base + ".TextGrid")
                if not os.path.exists(tg):
                    skipped.append((split, spk, base, "no TextGrid (alignment failed)"))
                    continue
                ret = self.process_utterance(spk, base)
                if ret is None:
                    skipped.append((split, spk, base, "rejected"))
                    continue
                info, pitch, energy, n = ret
                out[split].append(info)
                frames[split] += n
                if split == "train":
                    train_files.add(spk + "-%s-" + base + ".npy")
                    if len(pitch):
                        pitch_scaler.partial_fit(pitch.reshape(-1, 1))
                    if len(energy):
                        energy_scaler.partial_fit(energy.reshape(-1, 1))

        pm, ps = (pitch_scaler.mean_[0], pitch_scaler.scale_[0]) if self.pitch_norm else (0, 1)
        em, es = (energy_scaler.mean_[0], energy_scaler.scale_[0]) if self.energy_norm else (0, 1)
        pmin, pmax = self.normalize(os.path.join(self.out_dir, "pitch"), pm, ps, train_files, "pitch")
        emin, emax = self.normalize(os.path.join(self.out_dir, "energy"), em, es, train_files, "energy")

        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            json.dump(speakers, f)
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            json.dump({"pitch": [float(pmin), float(pmax), float(pm), float(ps)],
                       "energy": [float(emin), float(emax), float(em), float(es)]}, f)
        for split in ["train", "val"]:
            with open(os.path.join(self.out_dir, split + ".txt"), "w", encoding="utf-8") as f:
                for m in out[split]:
                    f.write(m + "\n")
        summary = {s: {"utterances": len(out[s]),
                       "hours": frames[s] * self.hop_length / self.sampling_rate / 3600} for s in out}
        summary["speakers"] = len(speakers)
        summary["skipped"] = len(skipped)
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(self.out_dir, "skipped.txt"), "w", encoding="utf-8") as f:
            for s in skipped:
                f.write("|".join(s) + "\n")
        print(json.dumps(summary, indent=2))
        return summary

    # ------------------------------------------------------------------
    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, basename + ".wav")
        text_path = os.path.join(self.in_dir, speaker, basename + ".lab")
        tg_path = os.path.join(self.out_dir, "TextGrid", speaker, basename + ".TextGrid")

        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(textgrid.get_tier_by_name("phones"))
        if start >= end or len(phone) == 0:
            return None
        text = "{" + " ".join(phone) + "}"

        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[int(self.sampling_rate * start): int(self.sampling_rate * end)].astype(np.float32)
        if len(wav) < self.hop_length * 4:
            return None
        with open(text_path, encoding="utf-8") as f:
            raw_text = f.readline().strip("\n")

        mel, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        n_frames = mel.shape[1]
        # make durations consistent with the number of mel frames
        diff = sum(duration) - n_frames
        i = len(duration) - 1
        while diff > 0 and i >= 0:
            cut = min(duration[i], diff)
            duration[i] -= cut
            diff -= cut
            i -= 1
        total = sum(duration)
        if total == 0 or total > self.max_frames:
            return None
        mel, energy = mel[:, :total], energy[:total]

        pitch, t = pw.dio(wav.astype(np.float64), self.sampling_rate,
                          frame_period=self.hop_length / self.sampling_rate * 1000)
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        if len(pitch) < total:
            pitch = np.pad(pitch, (0, total - len(pitch)), mode="edge")
        pitch = pitch[:total]
        if np.sum(pitch != 0) <= 1:
            return None

        if self.pitch_avg:
            nz = np.where(pitch != 0)[0]
            fn = interp1d(nz, pitch[nz], fill_value=(pitch[nz[0]], pitch[nz[-1]]), bounds_error=False)
            pitch = fn(np.arange(len(pitch)))
            pos = 0
            for i, d in enumerate(duration):
                pitch[i] = np.mean(pitch[pos: pos + d]) if d > 0 else 0
                pos += d
            pitch = pitch[: len(duration)]
        if self.energy_avg:
            pos = 0
            for i, d in enumerate(duration):
                energy[i] = np.mean(energy[pos: pos + d]) if d > 0 else 0
                pos += d
            energy = energy[: len(duration)]
        if not (np.isfinite(pitch).all() and np.isfinite(energy).all() and np.isfinite(mel).all()):
            return None

        np.save(os.path.join(self.out_dir, "duration", f"{speaker}-duration-{basename}.npy"), duration)
        np.save(os.path.join(self.out_dir, "pitch", f"{speaker}-pitch-{basename}.npy"), pitch)
        np.save(os.path.join(self.out_dir, "energy", f"{speaker}-energy-{basename}.npy"), energy)
        np.save(os.path.join(self.out_dir, "mel", f"{speaker}-mel-{basename}.npy"), mel.T)
        return ("|".join([basename, speaker, text, raw_text]),
                self.remove_outlier(pitch), self.remove_outlier(energy), mel.shape[1])

    def get_alignment(self, tier):
        phones, durations = [], []
        start_time = end_time = 0
        end_idx = 0
        for t in tier.intervals:
            s, e, p = t.start_time, t.end_time, t.text.strip()
            if p == "":
                p = "sp"                      # MFA >= 3 writes silences as empty labels
            if phones == []:
                if p in SIL:
                    continue
                start_time = s
            if p not in SIL:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                phones.append("sp")
            durations.append(int(np.round(e * self.sampling_rate / self.hop_length)
                                 - np.round(s * self.sampling_rate / self.hop_length)))
        return phones[:end_idx], durations[:end_idx], start_time, end_time

    @staticmethod
    def remove_outlier(values):
        values = np.array(values)
        p25, p75 = np.percentile(values, 25), np.percentile(values, 75)
        lo, hi = p25 - 1.5 * (p75 - p25), p75 + 1.5 * (p75 - p25)
        return values[np.logical_and(values > lo, values < hi)]

    @staticmethod
    def normalize(in_dir, mean, std, train_files, kind):
        mx, mn = np.finfo(np.float64).min, np.finfo(np.float64).max
        train_names = {f % kind for f in train_files}
        for fn in os.listdir(in_dir):
            path = os.path.join(in_dir, fn)
            v = (np.load(path) - mean) / std
            np.save(path, v.astype(np.float32))
            if fn in train_names:
                mx, mn = max(mx, v.max()), min(mn, v.min())
        return mn, mx
