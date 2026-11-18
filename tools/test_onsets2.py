"""Test onsets under several onset-estimator configs (tools/onset_model2.py), compared with the submitted
fix3 onsets; writes selective-shift submissions.

usage: python tools/test_onsets2.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "tools")
from onset_model import NORMAL, BEH, best_lag
from onset_model2 import onset_scores, pose_profile, posterior
from pair_views import signals

st = json.load(open("outputs/structure/test_videos.json")); chains = st["chains"]
sub = pd.read_csv("dataset/sample_submission.csv")
inst = pd.read_csv("outputs/v5/test_inst_scores.csv")
beh = pd.read_csv("outputs/v5_post/test_videos_posterior.csv").behaviour.values
tp = np.load("outputs/v2/test_pose.npy")
cur = pd.read_csv("outputs/v5_post/test_onsets.csv")
cur_on = {**dict(zip(cur.front, cur.onset_front)), **dict(zip(cur.side, cur.onset_side))}

P = lambda v: np.concatenate([tp[w] for w in chains[v]])
det = lambda v, b: (inst.loc[chains[v], f"m_{b}"].values - inst.loc[chains[v], NORMAL].max(axis=1).values) / 10.0
configs = {"v1": dict(w_pose=0.0), "pose1": dict(), "pose2": dict(w_pose=2.0)}
res = {k: {} for k in configs}
post_prev = {}
for f, s in st["pairs"]:
    b = beh[f]
    if BEH.index(b) < 4:
        continue
    lag = best_lag(signals(P(f)), signals(P(s)))
    df_, ds_, pf_, ps_ = det(f, b), det(s, b), pose_profile(P(f)), pose_profile(P(s))
    for k, kw in configs.items():
        scf = onset_scores(df_, pf_, ds_, ps_, lag / 30, **kw)
        scs = onset_scores(ds_, ps_, df_, pf_, -lag / 30, **kw)
        res[k][f], res[k][s] = int(np.argmax(scf)), int(np.argmax(scs))
        if k == "pose1":
            for v, sc in [(f, scf), (s, scs)]:
                post = posterior(sc)
                o = cur_on[v]
                post_prev[v] = float(post[:o].sum())        # P(onset earlier than the submitted one)
check = all(res["v1"][v] == cur_on[v] for v in cur_on)
print("v1 reproduces submitted onsets:", check)
for k in ["pose1", "pose2"]:
    d = {v: res[k][v] - cur_on[v] for v in cur_on}
    print(k, "earlier:", sum(x < 0 for x in d.values()), "later:", sum(x > 0 for x in d.values()),
          "| videos earlier:", sorted(v for v, x in d.items() if x < 0))
pp = pd.Series(post_prev).sort_values(ascending=False)
print("P(onset earlier than submitted), top 15:", pp.head(15).round(2).to_dict())

out = Path("outputs/v5_post")
base = pd.read_csv(out / "submission_onset_fix3.csv").label.values
for k in ["pose1", "pose2"]:
    lab = np.zeros(len(sub), int)
    for v, o in res[k].items():
        lab[np.array(chains[v])[o:]] = 1
    sub.assign(label=lab).to_csv(out / f"submission_onset_{k}.csv", index=False)
    print(k, "windows changed vs fix3:", int((lab != base).sum()), "| 0->1:", int(((lab == 1) & (base == 0)).sum()),
          "1->0:", int(((lab == 0) & (base == 1)).sum()))
