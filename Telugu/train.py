"""FastSpeech2 training (adapted from ming024/FastSpeech2 train.py).
Changes: single combined config.yaml, CSV loss logging (window-averaged train loss + full validation
loss), best/final/periodic checkpoints, wall-clock limit for Kaggle sessions, no in-loop vocoder.
"""
import argparse
import csv
import glob
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_utils import load_config, split_config
from dataset import Dataset
from model import FastSpeech2Loss
from utils.model import get_model, get_param_num
from utils.tools import to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEAD = ["step", "total", "mel", "postnet", "pitch", "energy", "duration"]


@torch.no_grad()
def validate(model, configs, Loss):
    pre, mod, tr = configs
    ds = Dataset("val.txt", pre, tr, sort=False, drop_last=False)
    loader = DataLoader(ds, batch_size=tr["optimizer"]["batch_size"], shuffle=False, collate_fn=ds.collate_fn)
    sums, n = np.zeros(6), 0
    model.eval()
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            losses = Loss(batch, model(*(batch[2:])))
            b = len(batch[0])
            sums += np.array([l.item() for l in losses]) * b
            n += b
    model.train()
    return sums / n


def append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEAD)
        w.writerow(row)


def main(args):
    cfg = load_config(args.config)
    configs = split_config(cfg)
    pre, mod, tr = configs
    seed = cfg["train"].get("seed", 1234)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # ---- auto-resume: --restore_step auto -> newest numbered checkpoint in ckpt_path
    ck_dir = tr["path"]["ckpt_path"]
    os.makedirs(ck_dir, exist_ok=True)
    if str(args.restore_step) == "auto":
        found = glob.glob(os.path.join(ck_dir, "[0-9]*.pth.tar"))
        args.restore_step = max((int(os.path.basename(p).split(".")[0]) for p in found), default=0)
        print("auto-resume from step", args.restore_step, flush=True)
    args.restore_step = int(args.restore_step)
    if args.restore_step >= tr["step"]["total_step"]:
        sd = torch.load(os.path.join(ck_dir, f"{args.restore_step}.pth.tar"), map_location="cpu", weights_only=False)
        torch.save({"model": sd["model"], "step": args.restore_step}, os.path.join(ck_dir, "final.pth.tar"))
        print("training already finished at step", args.restore_step, flush=True)
        return

    ds = Dataset("train.txt", pre, tr, sort=True, drop_last=True)
    bs, group = tr["optimizer"]["batch_size"], 4
    assert bs * group < len(ds), "train set too small for batch_size*4"
    loader = DataLoader(ds, batch_size=bs * group, shuffle=True, collate_fn=ds.collate_fn)

    model, optimizer = get_model(args, configs, device, train=True)
    Loss = FastSpeech2Loss(pre, mod).to(device)
    print("train utts:", len(ds), "| params:", get_param_num(model))

    ck_dir, log_dir = tr["path"]["ckpt_path"], tr["path"]["log_path"]
    os.makedirs(ck_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)
    S = tr["step"]
    total, log_step, val_step, save_step = S["total_step"], S["log_step"], S["val_step"], S["save_step"]
    acc, clip = tr["optimizer"]["grad_acc_step"], tr["optimizer"]["grad_clip_thresh"]
    step, best = args.restore_step + 1, float("inf")
    if args.restore_step > 0:               # continue the CSV logs / best-loss tracking from the restored step
        import pandas as _pd
        for _n in ("train_loss.csv", "val_loss.csv"):
            _p = os.path.join(log_dir, _n)
            if os.path.exists(_p):
                _d = _pd.read_csv(_p); _d = _d[_d["step"] <= args.restore_step]; _d.to_csv(_p, index=False)
                if _n == "val_loss.csv" and len(_d):
                    best = float(_d["total"].min())
    win, win_n = np.zeros(6), 0
    t0 = time.time()
    bar = tqdm(total=total, initial=args.restore_step, desc="train", mininterval=60)
    done = False

    def save(name, with_opt=True):
        obj = {"model": model.state_dict(), "step": step}
        if with_opt:
            obj["optimizer"] = optimizer._optimizer.state_dict()
        torch.save(obj, os.path.join(ck_dir, name))

    while not done:
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                losses = Loss(batch, model(*(batch[2:])))
                (losses[0] / acc).backward()
                if step % acc == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()
                win += np.array([l.item() for l in losses]); win_n += 1

                if step % log_step == 0:
                    m = win / win_n
                    append_csv(os.path.join(log_dir, "train_loss.csv"), [step] + [f"{v:.5f}" for v in m])
                    bar.write(f"step {step}/{total} | total {m[0]:.4f} mel {m[1]:.4f} post {m[2]:.4f} "
                              f"pitch {m[3]:.4f} energy {m[4]:.4f} dur {m[5]:.4f}")
                    win, win_n = np.zeros(6), 0
                if step % val_step == 0:
                    v = validate(model, configs, Loss)
                    append_csv(os.path.join(log_dir, "val_loss.csv"), [step] + [f"{x:.5f}" for x in v])
                    bar.write(f"VAL step {step} | total {v[0]:.4f} mel {v[1]:.4f} post {v[2]:.4f} "
                              f"pitch {v[3]:.4f} energy {v[4]:.4f} dur {v[5]:.4f}")
                    if v[0] < best:
                        best = v[0]
                        save("best.pth.tar", with_opt=False)
                if step % save_step == 0:
                    save(f"{step}.pth.tar")
                    old = sorted(glob.glob(os.path.join(ck_dir, "[0-9]*.pth.tar")),
                                 key=lambda p: int(os.path.basename(p).split(".")[0]))[:-2]
                    for p in old:
                        os.remove(p)
                limit = args.max_minutes and (time.time() - t0) / 60 > args.max_minutes
                if step >= total or limit:
                    save(f"{step}.pth.tar"); save("final.pth.tar", with_opt=False)
                    print("stopping at step", step, "(time limit)" if limit else "(done)")
                    done = True
                    break
                step += 1
                bar.update(1)
            if done:
                break

    from plotting import plot_loss_curves
    try:
        plot_loss_curves(log_dir, log_dir, cfg["language"])
    except Exception as e:
        print("plotting failed:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--restore_step", default="0", help="int, or 'auto' = newest checkpoint")
    ap.add_argument("--max_minutes", type=float, default=None, help="stop (and checkpoint) after N minutes")
    main(ap.parse_args())
