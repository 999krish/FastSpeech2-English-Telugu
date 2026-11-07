"""Loss-curve and variance-adaptor plots."""
import argparse
import os
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

COLS = ["total", "mel", "postnet", "pitch", "energy", "duration"]


def setup_fonts():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans Telugu", "Noto Serif Telugu", "Lohit Telugu", "Gautami", "Pothana2000"):
        if cand in names:
            plt.rcParams["font.family"] = [cand, "DejaVu Sans"]
            return cand
    print("WARNING: no Telugu font found - Telugu labels may show as boxes (apt-get install fonts-noto-core)")
    return None


def label(p):
    if p in ("sp", "spn", "sil"):
        return "·"
    if len(p) == 1 and unicodedata.category(p) in ("Mn", "Mc"):
        return "\u25cc" + p
    return p


# ------------------------------------------------------------------ loss curves
def plot_loss_curves(log_dir, out_dir, name=""):
    os.makedirs(out_dir, exist_ok=True)
    tr = pd.read_csv(os.path.join(log_dir, "train_loss.csv"))
    va = pd.read_csv(os.path.join(log_dir, "val_loss.csv"))
    for df, fn, ttl in [(tr, "training_loss.png", "Training"), (va, "validation_loss.png", "Validation")]:
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
        ax[0].plot(df["step"], df["total"], lw=2)
        ax[0].set_title(f"{name} {ttl} total loss"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].grid(alpha=.3)
        for c in COLS[1:]:
            ax[1].plot(df["step"], df[c], label=c)
        ax[1].set_yscale("log"); ax[1].legend(ncol=3); ax[1].grid(alpha=.3)
        ax[1].set_title(f"{name} {ttl} loss components"); ax[1].set_xlabel("step")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, fn), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tr["step"], tr["total"], label="train"); ax.plot(va["step"], va["total"], "o-", label="validation")
    ax.set_xlabel("step"); ax.set_ylabel("total loss"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title(f"{name} train vs validation total loss")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "train_vs_val_total.png"), dpi=150); plt.close(fig)


# ------------------------------------------------------------------ variance adaptor
def _axes_with_boundaries(ax, bounds, labels, fs):
    centers = (bounds[:-1] + bounds[1:]) / 2
    for b in bounds:
        ax.axvline(b, color="grey", lw=.3, alpha=.6)
    ax.set_xticks(centers)
    ax.set_xticklabels([label(p) for p in labels], fontsize=fs)


def _fs(n):
    return max(5, min(11, 420 / max(n, 1)))


def plot_duration(labels, dur_frames, ms_per_frame, title, path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(9, 0.28 * n), 4.2))
    ms = np.asarray(dur_frames) * ms_per_frame
    colors = ["#c9c9c9" if p in ("sp", "spn", "sil") else "#2a6fdb" for p in labels]
    ax.bar(range(n), ms, color=colors)
    ax.set_xticks(range(n)); ax.set_xticklabels([label(p) for p in labels], fontsize=_fs(n))
    ax.set_ylabel("predicted duration (ms)"); ax.set_xlabel("phoneme (grey = pause)")
    ax.set_title(f"{title}\nTotal = {ms.sum()/1000:.2f} s, {n} symbols")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_curve(kind, mel, labels, dur_frames, values, unit, ms_per_frame, title, path, color):
    bounds = np.concatenate([[0], np.cumsum(dur_frames)]).astype(float)
    n = len(labels)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(max(10, 0.3 * n), 7), sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, 1]})
    a0.imshow(mel.T, origin="lower", aspect="auto", extent=[0, mel.shape[0], 0, mel.shape[1]], cmap="magma")
    a0.set_ylabel("mel bin"); a0.set_title(f"{title}\nPredicted {kind} (phoneme level)")
    a1.stairs(values, bounds, color=color, lw=2, baseline=None)
    a1.set_ylabel(f"{kind} ({unit})"); a1.grid(alpha=.3)
    _axes_with_boundaries(a1, bounds, labels, _fs(n))
    a1.set_xlabel(f"time (frames of {ms_per_frame:.1f} ms) - phoneme boundaries from predicted durations")
    a1.set_xlim(0, bounds[-1])
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_pitch(mel, labels, dur, pitch_hz, ms, title, path):
    _plot_curve("pitch F0", mel, labels, dur, pitch_hz, "Hz", ms, title, path, "#d62728")


def plot_energy(mel, labels, dur, energy, ms, title, path):
    _plot_curve("energy", mel, labels, dur, energy, "L2 norm of STFT frame", ms, title, path, "#2ca02c")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    plot_loss_curves(a.logs, a.out, a.name)
