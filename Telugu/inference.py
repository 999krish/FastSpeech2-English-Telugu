"""Synthesis + variance-adaptor analysis for the assignment test sentences.
Writes  <out>/test_generated_audios/<ID>.wav
        <out>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png
        <out>/variance_adaptor_values/<ID>.csv   (phoneme, duration, pitch Hz, energy)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile

import plotting
from config_utils import load_config, split_config
from model import FastSpeech2
from preprocess import get_frontend
from text import text_to_sequence
from utils.model import get_vocoder, vocoder_infer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(pre, mod, ckpt):
    model = FastSpeech2(pre, mod).to(device)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)   # fp16 release files are cast back to fp32
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best.pth.tar")
    ap.add_argument("--sentences", default="test_sentences.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speaker", default=None, help="speaker name from speakers.json (default: first)")
    ap.add_argument("--pitch_control", type=float, default=1.0)
    ap.add_argument("--energy_control", type=float, default=1.0)
    ap.add_argument("--duration_control", type=float, default=1.0)
    ap.add_argument("--no_plots", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):                       # e.g. a fresh clone: use the fp16 release checkpoint
        import glob
        a.ckpt = sorted(glob.glob("checkpoints/*fp16*.pth.tar"))[0]
    print("checkpoint:", a.ckpt)
    cfg = load_config(a.config)
    pre, mod, tr = split_config(cfg)
    P = pre["path"]["preprocessed_path"]
    hop = pre["preprocessing"]["stft"]["hop_length"]
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    ms = hop / sr * 1000
    speakers = json.load(open(os.path.join(P, "speakers.json")))
    spk_name = a.speaker or sorted(speakers)[0]
    spk_id = speakers[spk_name]
    stats = json.load(open(os.path.join(P, "stats.json")))
    p_mean, p_std = stats["pitch"][2:]
    e_mean, e_std = stats["energy"][2:]
    print("speaker:", spk_name, spk_id)

    model = load_model(pre, mod, a.ckpt)
    vocoder = get_vocoder(mod, device)
    frontend = get_frontend(cfg)
    if cfg["language"] == "Telugu":
        plotting.setup_fonts()
    sentences = json.load(open(a.sentences, encoding="utf-8"))

    d_audio = os.path.join(a.out, "test_generated_audios")
    d_vals = os.path.join(a.out, "variance_adaptor_values")
    d_plot = {k: os.path.join(a.out, "variance_adaptor_plots", k) for k in ("duration", "pitch", "energy")}
    for d in [d_audio, d_vals, *d_plot.values()]:
        os.makedirs(d, exist_ok=True)

    for sid, text in sentences.items():
        norm, ph = frontend(text)
        labels = ph.strip("{}").split()
        seq = np.array(text_to_sequence(ph, pre["preprocessing"]["text"]["text_cleaners"]))
        if len(seq) != len(labels):
            print(f"[{sid}] WARNING: {len(labels) - len(seq)} phones not in symbol table were dropped")
            labels = [l for l in labels if text_to_sequence("{" + l + "}", [])]
        print(f"[{sid}] {text}\n      normalised: {norm}\n      phones: {ph}")
        with torch.no_grad():
            out = model(torch.LongTensor([spk_id]).to(device), torch.from_numpy(seq).long()[None].to(device),
                         torch.LongTensor([len(seq)]).to(device), len(seq),
                         p_control=a.pitch_control, e_control=a.energy_control, d_control=a.duration_control)
            mel = out[1]
            p, e, d, mel_len = out[2][0].cpu().numpy(), out[3][0].cpu().numpy(), out[5][0].cpu().numpy(), out[9][0].item()
            wav = vocoder_infer(mel.transpose(1, 2), vocoder, mod, pre, lengths=[mel_len * hop])[0]
        wavfile.write(os.path.join(d_audio, f"{sid}.wav"), sr, wav)

        pitch_hz, energy = p * p_std + p_mean, e * e_std + e_mean
        pd.DataFrame({"phoneme": labels, "duration_frames": d, "duration_ms": d * ms,
                      "pitch_hz": pitch_hz, "energy": energy}).to_csv(
            os.path.join(d_vals, f"{sid}.csv"), index=False, encoding="utf-8")
        if not a.no_plots:
            title = f"{sid}: {text}"
            plotting.plot_duration(labels, d, ms, title, os.path.join(d_plot["duration"], f"{sid}.png"))
            plotting.plot_pitch(mel[0].cpu().numpy()[:mel_len], labels, d, pitch_hz, ms, title,
                                os.path.join(d_plot["pitch"], f"{sid}.png"))
            plotting.plot_energy(mel[0].cpu().numpy()[:mel_len], labels, d, energy, ms, title,
                                 os.path.join(d_plot["energy"], f"{sid}.png"))
        print(f"      -> {mel_len * ms / 1000:.2f} s audio")


if __name__ == "__main__":
    main()
