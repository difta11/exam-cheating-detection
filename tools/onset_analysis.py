"""Where does the behaviour start inside a video? Single change-point on per-second detector scores (OOF),
for all train videos, plus onset-based labelling rules evaluated on the manual annotation (tools/moment_eval.py).

usage: python tools/onset_analysis.py <oof_inst_scores.csv>
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "tools")
from moment_eval import ANN, BEH

oof = pd.read_csv(sys.argv[1]); oof["w"] = oof.off // 30
vdf = pd.read_csv("outputs/v2/train_videos.csv")
NORMAL = [f"m_{b}" for b in BEH[:4]]


def change_point(s, lo=0.2, hi=0.8, prior=0.0):
    """t* maximising mean(after) - mean(before) (+ prior * closeness to the middle); returns t*, contrast"""
    n = len(s); best = (-1e9, n // 2)
    for t in range(max(2, int(lo * n)), min(n - 1, int(np.ceil(hi * n))) + 1):
        c = s[t:].mean() - s[:t].mean() - prior * abs(t / n - 0.5)
        best = max(best, (c, t))
    return best[1], best[0]


def score(g, beh):
    return (g[f"m_{beh}"] - g[NORMAL].max(axis=1)).values


rows = []
for v, g in oof.groupby("video"):
    g = g.sort_values("w")
    beh = vdf.behavior[v]
    s = score(g, beh) if vdf.label[v] == 1 else g[f"m_{beh}"].values - g[[c for c in NORMAL if c != f"m_{beh}"]].max(axis=1).values
    t, c = change_point(s)
    rows.append(dict(video=v, behavior=beh, label=vdf.label[v], angle=vdf.angle[v], n=len(g), onset=t, onset_pos=t / len(g), contrast=c))
r = pd.DataFrame(rows)
print("onset position (change point of own-behaviour evidence), quantiles per behaviour:")
print(r.groupby("behavior").onset_pos.describe(percentiles=[.1, .25, .5, .75, .9])[["10%", "25%", "50%", "75%", "90%"]].round(2))
print("\nall cheat videos: onset position histogram", np.histogram(r[r.label == 1].onset_pos, bins=[0, .2, .3, .4, .5, .6, .7, .8, 1])[0].tolist())

# labelling rules on the manual annotation
res = {}
for (subj, ang, beh), wins in ANN.items():
    v = vdf.index[(vdf.subject == subj) & (vdf.angle == ang) & (vdf.behavior == beh)][0]
    g = oof[oof.video == v].sort_values("w")
    truth = g.w.isin(wins).values
    n = len(g); pos = np.arange(n) / n
    s = score(g, beh)
    t_cp, _ = change_point(s)
    t_cp_prior, _ = change_point(s, prior=2.0)
    for name, lab in [("all = 1", np.ones(n, bool)), ("pos >= 0.40", pos >= 0.40), ("pos >= 0.45", pos >= 0.45),
                      ("pos >= 0.50", pos >= 0.50), ("change point", np.arange(n) >= t_cp),
                      ("change point + middle prior", np.arange(n) >= t_cp_prior)]:
        res.setdefault(name, []).append((lab == truth).mean())
    print(f"s{subj}-{ang}-{beh[:10]:10s} truth onset {min(wins):2d} | cp {t_cp:2d} | cp+prior {t_cp_prior:2d} | n {n}")
print()
for k, v in res.items():
    print(f"{k:28s} mean accuracy inside annotated cheat videos {np.mean(v):.3f}")
