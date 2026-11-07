"""Stage-based preprocessing.  Usage:  python preprocess.py --config config.yaml --stage all
stages: prepare (select+write corpus) -> dict (MFA lexicon) -> align (MFA) -> features (mel/pitch/energy/duration)
"""
import argparse
import os
import shutil
import subprocess
import sys

from config_utils import load_config
from prepare_data import get_frontend_module


def get_frontend(cfg):
    mod = get_frontend_module(cfg)
    if cfg["language"] == "English":
        return mod.Frontend(cfg["preprocess"]["path"]["lexicon_path"])
    return mod.Frontend()


def stage_dict(cfg):
    P = cfg["preprocess"]["path"]
    words = set()
    for spk in os.listdir(P["raw_path"]):
        for fn in os.listdir(os.path.join(P["raw_path"], spk)):
            if fn.endswith(".lab"):
                with open(os.path.join(P["raw_path"], spk, fn), encoding="utf-8") as f:
                    words.update(f.read().split())
    d = get_frontend(cfg).build_dictionary(sorted(words))
    os.makedirs(os.path.dirname(P["dict_path"]), exist_ok=True)
    with open(P["dict_path"], "w", encoding="utf-8") as f:
        for w in sorted(d):
            f.write(w + " " + " ".join(d[w]) + "\n")
    print(f"dictionary: {len(d)} words -> {P['dict_path']}")


def _run(cmd):
    print(">>", cmd)
    return subprocess.call(cmd, shell=True)


def stage_align(cfg):
    P, m = cfg["preprocess"]["path"], cfg["data"]["mfa"]
    mfa = os.environ.get("MFA_CMD", "mfa")
    tg = os.path.join(P["preprocessed_path"], "TextGrid")
    shutil.rmtree(tg, ignore_errors=True)
    os.makedirs(P["preprocessed_path"], exist_ok=True)
    raw, dic, j = P["raw_path"], P["dict_path"], m.get("jobs", 4)
    ok = 1
    if m["mode"] == "pretrained":
        _run(f"{mfa} model download acoustic {m['acoustic_model']}")
        ok = _run(f"{mfa} align {raw} {dic} {m['acoustic_model']} {tg} --clean -j {j} "
                  f"--beam 100 --retry_beam 400 --output_format long_textgrid")
        if ok != 0:
            print("pretrained alignment failed -> falling back to training MFA from scratch")
    if ok != 0:
        zipf = os.path.join(P["preprocessed_path"], "mfa_acoustic_model.zip")
        shutil.rmtree(tg, ignore_errors=True)
        ok = _run(f"{mfa} train {raw} {dic} {zipf} --output_directory {tg} --clean -j {j} "
                  f"--output_format long_textgrid")
    assert ok == 0 and os.path.isdir(tg), "MFA failed - see log above"
    n = sum(len(fs) for _, _, fs in os.walk(tg))
    print("TextGrids written:", n)


def stage_features(cfg):
    from preprocessor.preprocessor import Preprocessor
    Preprocessor(cfg["preprocess"]).build_from_path()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", default="all", choices=["all", "prepare", "dict", "align", "features"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    sys.path.insert(0, os.getcwd())
    if a.stage in ("all", "prepare"):
        from prepare_data import prepare
        prepare(cfg)
    if a.stage in ("all", "dict"):
        stage_dict(cfg)
    if a.stage in ("all", "align"):
        stage_align(cfg)
    if a.stage in ("all", "features"):
        stage_features(cfg)
