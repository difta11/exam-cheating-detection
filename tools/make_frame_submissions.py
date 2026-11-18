"""Test submissions from frame-resolution onsets (tools/frame_onset2.py) under two boundary conventions.

usage: python tools/make_frame_submissions.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "tools")
from onset_model import NORMAL, BEH, best_lag
from frame_onset import deviation
from frame_onset2 import onset_frame, to_window
from pair_views import signals

st = json.load(open("outputs/structure/test_videos.json")); chains = st["chains"]
sub = pd.read_csv("dataset/sample_submission.csv")
inst = pd.read_csv("outputs/v5/test_inst_scores.csv")
beh = pd.read_csv("outputs/v5_post/test_videos_posterior.csv").behaviour.values
tp = np.load("outputs/v2/test_pose.npy")
P = lambda v: np.concatenate([tp[w] for w in chains[v]])
det = lambda v, b: (inst.loc[chains[v], f"m_{b}"].values - inst.loc[chains[v], NORMAL].max(axis=1).values) / 10.0
out = Path("outputs/v5_post")
fix3 = pd.read_csv(out / "submission_onset_fix3.csv").label.values
m1 = pd.read_csv(out / "submission_onset_m1.csv").label.values
labs = {c: np.zeros(len(sub), int) for c in ["majority", "any"]}
rows = []
for f, s in st["pairs"]:
    b = beh[f]
    if BEH.index(b) < 4:
        continue
    lag = best_lag(signals(P(f)), signals(P(s)))
    df_, ds_, pf_, ps_ = det(f, b), det(s, b), deviation(P(f)), deviation(P(s))
    ff, _ = onset_frame(df_, pf_, ds_, ps_, lag)
    fs, _ = onset_frame(ds_, ps_, df_, pf_, -lag)
    for v, fo in [(f, ff), (s, fs)]:
        for c in labs:
            w = min(max(to_window(fo, c), 0), len(chains[v]))
            labs[c][np.array(chains[v])[w:]] = 1
        rows.append(dict(video=v, onset_frame=fo, win_majority=to_window(fo, "majority"), win_any=to_window(fo, "any"),
                         frac_in_window=round((fo % 30) / 30, 2)))
r = pd.DataFrame(rows)
r.to_csv(out / "test_onset_frames.csv", index=False)
print("fraction of onsets in the first half of their window:", round(float((r.frac_in_window < 0.5).mean()), 2))
for c, lab in labs.items():
    sub.assign(label=lab).to_csv(out / f"submission_frame_{c}.csv", index=False)
    print(f"frame_{c}: ones {lab.sum()} | vs fix3 changed {int((lab != fix3).sum())} | vs m1 changed {int((lab != m1).sum())}")
