"""Onset estimator v2: position prior + detector change point + pose-deviation change point, own view + paired view.
Validated on the manual train onsets; also reports per-window posterior P(window positive) for F1-style decisions.

usage: python tools/onset_model2.py
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "tools")
from onset_model import ONSET, NORMAL, best_lag, cp_curve
from pair_views import signals
from frame_onset import deviation


def pose_profile(P):
    d = deviation(P)
    nwin = (len(P) - 30) // 30 + 1
    return np.log(np.array([d[30 * w:30 * w + 30].mean() for w in range(nwin)]) + 0.05)


def score_curve(det, prof, n, w_det=1.0, w_pose=1.0):
    c = np.full(n + 1, -50.0)
    cd, cp = cp_curve(det), cp_curve(prof)
    ok = np.isfinite(cd) & np.isfinite(cp)
    c[ok] = w_det * cd[ok] + w_pose * cp[ok]
    return c


def onset_scores(det_o, prof_o, det_x=None, prof_x=None, shift=0.0, w_det=1.0, w_pose=1.0, w_other=0.6,
                 w_prior=1.0, mu=0.49, sd=0.06):
    n = len(det_o); t = np.arange(n + 1)
    sc = -w_prior * 0.5 * ((t / n - mu) / sd) ** 2 + score_curve(det_o, prof_o, n, w_det, w_pose)
    if det_x is not None:
        cx = score_curve(det_x, prof_x, len(det_x), w_det, w_pose)
        to = np.clip(np.round(t - shift).astype(int), 0, len(det_x))
        sc = sc + w_other * cx[to]
    return sc


def posterior(sc, temp=1.0):
    p = np.exp((sc - sc.max()) / temp); return p / p.sum()


if __name__ == "__main__":
    vdf = pd.read_csv("outputs/v2/train_videos.csv")
    pose = np.load("outputs/v2/train_pose.npy")
    oof = pd.read_csv("outputs/v5/oof_inst_scores.csv"); oof["w"] = oof.off // 30
    det = {v: ((g.sort_values("w")[f"m_{vdf.behavior[v]}"] - g.sort_values("w")[NORMAL].max(axis=1)).values / 10.0)
           for v, g in oof.groupby("video")}
    P = lambda v: pose[vdf.start[v]:vdf.start[v] + vdf.n[v]]
    cases = []
    for (subj, ang, beh), on in ONSET.items():
        other = "side" if ang == "front" else "front"
        v = vdf.index[(vdf.subject == subj) & (vdf.angle == ang) & (vdf.behavior == beh)][0]
        u = vdf.index[(vdf.subject == subj) & (vdf.angle == other) & (vdf.behavior == beh)][0]
        lag = best_lag(signals(P(v)), signals(P(u)))
        cases.append(dict(on=on, det_o=det[v], prof_o=pose_profile(P(v)), det_x=det[u], prof_x=pose_profile(P(u)), shift=lag / 30))
    configs = {"v1: prior+det+other(det)": dict(w_pose=0.0),
               "prior+pose": dict(w_det=0.0, w_other=0.0),
               "prior+det+pose": dict(w_other=0.0),
               "prior+det+pose+other": dict(),
               "prior+det+pose+other, w_pose 0.5": dict(w_pose=0.5),
               "prior+det+pose+other, w_pose 2": dict(w_pose=2.0)}
    for name, kw in configs.items():
        err = []
        for c in cases:
            sc = onset_scores(c["det_o"], c["prof_o"], c["det_x"], c["prof_x"], c["shift"], **kw)
            err.append(int(np.argmax(sc)) - c["on"])
        err = np.array(err)
        print(f"{name:36s} exact {np.mean(err == 0):.2f} | within1 {np.mean(np.abs(err) <= 1):.2f} | "
              f"early {np.sum(err < 0)} late {np.sum(err > 0)}")
