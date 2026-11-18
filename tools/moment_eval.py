"""Evaluate per-second cheating-moment scores against a small manual annotation of TRAIN cheat videos.

Annotation (1-s windows w = frames [30w, 30w+30), judged from contact sheets, tools/moment_sheets.py):
cheating visible in the listed windows; the remaining windows of the same video show ordinary writing.
usage: python tools/moment_eval.py   (needs outputs/v5/oof_inst_scores.csv)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ANN = {  # (subject, angle, behaviour): windows with visible cheating
    (7, "front", "Using_Smartwatch"): [9, 10, 11, 12, 13, 14],
    (7, "front", "Reading_Hidden_Notes"): [8, 9, 10, 11, 12, 13, 14],
    (7, "front", "Repeated_Peeking"): [7, 8, 11, 12, 13],
    (22, "front", "Using_Smartwatch"): [7, 8, 9, 10, 11, 13, 14],
    (22, "front", "Reading_Hidden_Notes"): [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    (15, "side", "Repeated_Peeking"): [8, 9, 10, 12, 13, 16],
    (15, "side", "Using_Smartwatch"): [8, 9, 10, 11, 16],
}
BEH = ["Writing_Answers", "Thinking", "Looking_Around", "Checking_Smartwatch",
       "Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]


def segment(s, tau):
    """best contiguous segment (max subarray of s - tau); returns boolean mask"""
    best, cur, start, bs, be = 0.0, 0.0, 0, 0, -1
    for i, x in enumerate(s - tau):
        if cur <= 0:
            cur, start = x, i
        else:
            cur += x
        if cur > best:
            best, bs, be = cur, start, i
    m = np.zeros(len(s), bool); m[bs:be + 1] = True
    return m


if __name__ == "__main__":
    vdf = pd.read_csv("outputs/v2/train_videos.csv")
    oof = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "outputs/v5/oof_inst_scores.csv")
    oof["w"] = oof.off // 30
    rows = []
    for (subj, ang, beh), wins in ANN.items():
        v = vdf.index[(vdf.subject == subj) & (vdf.angle == ang) & (vdf.behavior == beh)][0]
        d = oof[oof.video == v].sort_values("w").copy()
        d["truth"] = d.w.isin(wins).astype(int)
        d["s_generic"] = d.moment
        d["s_beh"] = d[f"m_{beh}"] - d[[f"m_{b}" for b in BEH[:4]]].max(1)
        d["s_pos"] = d.w / d.w.max()
        d["key"] = f"s{subj}-{ang}-{beh[:8]}"
        rows.append(d)
    d = pd.concat(rows)
    print("pooled AUC (visible-cheat seconds vs writing seconds inside the same cheat videos):")
    for s in ["s_generic", "s_beh", "s_pos"]:
        within = np.mean([roc_auc_score(g.truth, g[s]) for _, g in d.groupby("key")])
        print(f"  {s:10s} pooled {roc_auc_score(d.truth, d[s]):.3f} | mean within-video {within:.3f}")
    print("\nper video: truth / generic score rank")
    for k, g in d.groupby("key"):
        print(f"{k:28s}", "".join("X" if t else "." for t in g.truth), "|",
              " ".join(f"{x:+.1f}" for x in g.s_beh))
    # decision rules: fraction of seconds right inside annotated cheat videos
    for name, fn in [("all windows = 1", lambda g: np.ones(len(g), bool)),
                     ("second half", lambda g: g.s_pos.values >= 0.45),
                     ("s_beh > 0", lambda g: g.s_beh.values > 0),
                     ("segment(s_beh, 0)", lambda g: segment(g.s_beh.values, 0.0)),
                     ("segment(s_beh, -0.5)", lambda g: segment(g.s_beh.values, -0.5))]:
        acc = np.mean(np.concatenate([fn(g) == g.truth.values.astype(bool) for _, g in d.groupby("key")]))
        print(f"rule {name:22s} accuracy inside cheat videos {acc:.3f}")
