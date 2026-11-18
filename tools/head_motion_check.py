"""Interpretable head-motion statistics: do test videos assigned Looking_Around / Repeated_Peeking resemble
the train videos of the same behaviour (per angle)?"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

V2, ST = Path("outputs/v2"), Path("outputs/structure")


def head_stats(P):
    kp = P[:, :51].reshape(-1, 17, 3)
    x, y, c = kp[..., 0] * 640, kp[..., 1] * 360, kp[..., 2]
    bh = np.median(np.maximum((P[:, 54] - P[:, 52]) * 360, 1))
    shx, shy = (x[:, 5] + x[:, 6]) / 2, (y[:, 5] + y[:, 6]) / 2
    ok = (c[:, 0] > 0.3) & (c[:, 5] > 0.3) & (c[:, 6] > 0.3)
    nx = np.where(ok, (x[:, 0] - shx) / bh, np.nan); ny = np.where(ok, (y[:, 0] - shy) / bh, np.nan)
    nx = pd.Series(nx).interpolate(limit_direction="both").rolling(9, center=True, min_periods=1).median().values
    ny = pd.Series(ny).interpolate(limit_direction="both").rolling(9, center=True, min_periods=1).median().values
    dx, dy = nx - np.nanmedian(nx), ny - np.nanmedian(ny)
    sp = np.abs(np.diff(nx)) + np.abs(np.diff(ny))
    ex = np.abs(dx) > 0.06
    n_exc = int(((ex[1:].astype(int) - ex[:-1].astype(int)) == 1).sum())
    return dict(yaw_std=np.nanstd(dx), yaw_range=np.nanpercentile(dx, 95) - np.nanpercentile(dx, 5),
                pitch_std=np.nanstd(dy), up=np.nanpercentile(-dy, 95), n_excursions=n_exc / len(P) * 450,
                frac_away=float(ex.mean()), head_speed=np.nanmean(sp) * 30,
                skew_lr=float(np.nanmean(np.sign(dx[ex]))) if ex.any() else 0.0)


vdf = pd.read_csv(V2 / "train_videos.csv")
tp = np.load(V2 / "train_pose.npy")
tr = pd.DataFrame([head_stats(tp[s:s + n]) for s, n in zip(vdf.start, vdf.n)])
tr[["behavior", "angle"]] = vdf[["behavior", "angle"]]

st = json.load(open(ST / "test_videos.json"))
te_pose = np.load(V2 / "test_pose.npy")
tv = pd.read_csv("outputs/v4/test_videos.csv")
B = ["Writing_Answers", "Thinking", "Looking_Around", "Checking_Smartwatch",
     "Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]
te = pd.DataFrame([head_stats(np.concatenate([te_pose[w] for w in ch])) for ch in st["chains"]])
te["behavior"] = [B[b] for b in tv.behaviour]
te["angle"] = st["angle"]

cols = ["yaw_std", "yaw_range", "pitch_std", "up", "n_excursions", "frac_away", "head_speed"]
for ang in ["front", "side"]:
    a = tr[(tr.angle == ang) & tr.behavior.isin(["Looking_Around", "Repeated_Peeking", "Writing_Answers"])]
    b = te[(te.angle == ang) & te.behavior.isin(["Looking_Around", "Repeated_Peeking", "Writing_Answers"])]
    print(f"\n=== {ang}: median per behaviour (train true label | test assigned label)")
    print(pd.concat([a.groupby("behavior")[cols].median().add_prefix(""), b.groupby("behavior")[cols].median()],
                    keys=["train", "test"]).round(3).to_string())
