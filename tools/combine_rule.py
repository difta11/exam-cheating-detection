"""Combine position-in-video and detector evidence for per-second labels inside cheat videos.
Leave-one-video-out on the manual train annotation (tools/moment_eval.py).

usage: python tools/combine_rule.py <oof_inst_scores.csv>
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "tools")
from moment_eval import ANN, BEH

NORMAL = [f"m_{b}" for b in BEH[:4]]
oof = pd.read_csv(sys.argv[1]); oof["w"] = oof.off // 30
vdf = pd.read_csv("outputs/v2/train_videos.csv")


def feats(g, beh):
    n = len(g)
    s = (g[f"m_{beh}"] - g[NORMAL].max(axis=1)).values / 10.0
    pos = np.arange(n) / n
    sm = pd.Series(s).rolling(3, center=True, min_periods=1).mean().values      # neighbouring seconds
    after = np.array([s[: i + 1].max() for i in range(n)])                      # evidence seen so far
    return np.c_[s, sm, pos, after]


data = []
for (subj, ang, beh), wins in ANN.items():
    v = vdf.index[(vdf.subject == subj) & (vdf.angle == ang) & (vdf.behavior == beh)][0]
    g = oof[oof.video == v].sort_values("w")
    data.append((feats(g, beh), g.w.isin(wins).values.astype(int)))

sets = {"pos only": [2], "detector only": [0], "detector smoothed": [1], "pos + detector": [0, 2],
        "pos + smoothed": [1, 2], "pos + smoothed + seen-so-far": [1, 2, 3]}
for name, cols in sets.items():
    accs = []
    for k in range(len(data)):
        Xtr = np.concatenate([d[0][:, cols] for i, d in enumerate(data) if i != k])
        ytr = np.concatenate([d[1] for i, d in enumerate(data) if i != k])
        clf = LogisticRegression(C=1.0).fit(Xtr, ytr)
        accs.append((clf.predict(data[k][0][:, cols]) == data[k][1]).mean())
    full = LogisticRegression(C=1.0).fit(np.concatenate([d[0][:, cols] for d in data]), np.concatenate([d[1] for d in data]))
    print(f"{name:30s} leave-one-video-out accuracy {np.mean(accs):.3f}  coef {np.round(full.coef_[0], 2).tolist()} b {full.intercept_[0]:.2f}")
