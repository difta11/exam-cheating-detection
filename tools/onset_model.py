"""Behaviour-onset estimation for cheat videos, validated on manual TRAIN onsets.

Evidence: (1) position prior (onset ~ 0.49 n), (2) change point of per-second detector scores (v5 instance MIL,
own behaviour vs best normal behaviour), (3) the paired view of the same take (front/side, aligned with the
pose cross-correlation lag) — both cameras see the behaviour start at the same moment.

usage: python tools/onset_model.py   (project root)
"""
import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "tools")
from pair_views import signals

BEH = ["Writing_Answers", "Thinking", "Looking_Around", "Checking_Smartwatch",
       "Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]
NORMAL = [f"m_{b}" for b in BEH[:4]]
# manual onsets (first 1-s window showing the behaviour), front unless noted; from tools/onset_sheets.py and
# tools/moment_sheets.py contact sheets
ONSET = {(2, "front", "Looking_Under_the_Desk"): 8, (2, "front", "Reading_Hidden_Notes"): 7,
         (2, "front", "Repeated_Peeking"): 7, (2, "front", "Using_Smartwatch"): 8,
         (5, "front", "Looking_Under_the_Desk"): 8, (5, "front", "Reading_Hidden_Notes"): 7,
         (5, "front", "Repeated_Peeking"): 7, (5, "front", "Using_Smartwatch"): 7,
         (11, "front", "Looking_Under_the_Desk"): 8, (11, "front", "Reading_Hidden_Notes"): 8,
         (11, "front", "Using_Smartwatch"): 8,
         (24, "front", "Looking_Under_the_Desk"): 7, (24, "front", "Reading_Hidden_Notes"): 7,
         (24, "front", "Repeated_Peeking"): 7, (24, "front", "Using_Smartwatch"): 8,
         (7, "front", "Using_Smartwatch"): 9, (7, "front", "Reading_Hidden_Notes"): 8,
         (7, "front", "Repeated_Peeking"): 7, (22, "front", "Using_Smartwatch"): 7,
         (22, "front", "Reading_Hidden_Notes"): 7, (15, "side", "Repeated_Peeking"): 8,
         (15, "side", "Using_Smartwatch"): 8}


def best_lag(A, B, maxlag=150, minov=150):
    """lag maximising correlation; front frame f <-> side frame f - lag"""
    best = (-1, 0)
    for lag in range(-maxlag, maxlag + 1, 1):
        a = A[max(lag, 0):]; b = B[max(-lag, 0):]
        n = min(len(a), len(b))
        if n < minov:
            continue
        a, b = a[:n], b[:n]
        a = (a - a.mean(0)) / (a.std(0) + 1e-6); b = (b - b.mean(0)) / (b.std(0) + 1e-6)
        c = (a * b).mean(0)
        best = max(best, (float(np.mean(c[:3]) * 0.6 + np.mean(c[3:]) * 0.4), lag))
    return best[1]


def cp_curve(s):
    """contrast(t) = mean(s[t:]) - mean(s[:t]) for every split t (onset = first window of the 2nd part)"""
    n = len(s); c = np.full(n + 1, -np.inf)
    for t in range(2, n - 1):
        c[t] = s[t:].mean() - s[:t].mean()
    return c


def estimate(s_own, n_own, s_other=None, n_other=None, shift=0.0, w_prior=1.0, w_det=1.0, w_other=0.6, mu=0.49, sd=0.06):
    """onset window index for the own view; shift = other-view window offset (own window w ~ other window w - shift)"""
    t = np.arange(n_own + 1)
    score = -w_prior * 0.5 * ((t / n_own - mu) / sd) ** 2
    c = cp_curve(s_own); score = score + w_det * np.where(np.isfinite(c), c, -50)
    if s_other is not None:
        co = cp_curve(s_other)
        to = np.clip(np.round(t - shift).astype(int), 0, n_other)
        score = score + w_other * np.where(np.isfinite(co[to]), co[to], -50)
    return int(np.argmax(score))


if __name__ == "__main__":
    vdf = pd.read_csv("outputs/v2/train_videos.csv")
    pose = np.load("outputs/v2/train_pose.npy")
    oof = pd.read_csv("outputs/v5/oof_inst_scores.csv"); oof["w"] = oof.off // 30
    sc = {}
    for v, g in oof.groupby("video"):
        g = g.sort_values("w"); beh = vdf.behavior[v]
        sc[v] = ((g[f"m_{beh}"] - g[NORMAL].max(axis=1)).values / 10.0)
    rows = []
    for (subj, ang, beh), on in ONSET.items():
        other = "side" if ang == "front" else "front"
        v = vdf.index[(vdf.subject == subj) & (vdf.angle == ang) & (vdf.behavior == beh)][0]
        u = vdf.index[(vdf.subject == subj) & (vdf.angle == other) & (vdf.behavior == beh)][0]
        Sv = signals(pose[vdf.start[v]:vdf.start[v] + vdf.n[v]]); Su = signals(pose[vdf.start[u]:vdf.start[u] + vdf.n[u]])
        lag = best_lag(Sv, Su)                     # own frame f <-> other frame f - lag
        shift = lag / 30.0
        n = len(sc[v])
        est = {"pos 0.45": int(np.ceil(0.45 * n)), "pos 0.49": int(round(0.49 * n)),
               "detector cp": estimate(sc[v], n, w_prior=0.0),
               "prior + detector": estimate(sc[v], n),
               "prior + detector + other view": estimate(sc[v], n, sc[u], len(sc[u]), shift)}
        rows.append(dict(key=f"s{subj}-{ang}-{beh[:12]}", n=n, truth=on, lag=lag, **est))
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(r.to_string(index=False))
    print()
    for k in ["pos 0.45", "pos 0.49", "detector cp", "prior + detector", "prior + detector + other view"]:
        err = r[k] - r.truth
        # per-window accuracy inside the video if all windows >= onset are labelled 1 (onset-to-end labels)
        acc = 1 - (err.abs() / r.n).mean()
        print(f"{k:30s} exact {np.mean(err == 0):.2f} | within 1 {np.mean(err.abs() <= 1):.2f} | "
              f"mean |err| {err.abs().mean():.2f} | onset-to-end window accuracy {acc:.3f}")
