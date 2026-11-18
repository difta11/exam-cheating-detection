"""Frame-resolution onset: prior on relative position + pose-deviation contrast (per frame) + detector
change-point (window level, interpolated) for the own and the paired view (aligned by the pose lag).
Window labels can then follow any boundary convention:
  majority : window w is 1 if onset_frame <= 30w + 15
  any      : window w is 1 if onset_frame <  30w + 30

usage: python tools/frame_onset2.py            (validation on the manual train onsets)
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "tools")
from onset_model import ONSET, NORMAL, best_lag, cp_curve
from frame_onset import deviation
from pair_views import signals


def frame_curve(det, d, w_det=1.0, w_pose=1.0):
    """score per frame boundary f (onset = first frame of the behaviour part)"""
    n = len(d)
    ld = np.log(d + 0.05)
    cs = np.concatenate([[0], np.cumsum(ld)])
    f = np.arange(n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pose_c = (cs[-1] - cs[f]) / np.maximum(n - f, 1) - cs[f] / np.maximum(f, 1)
    pose_c[:30] = -50; pose_c[-30:] = -50
    cd = cp_curve(det)                                   # index = window boundary t
    t = np.clip(np.round(f / 30).astype(int), 0, len(det))
    det_c = np.where(np.isfinite(cd[t]), cd[t], -50)
    return w_pose * pose_c + w_det * det_c


def onset_frame(det_o, d_o, det_x=None, d_x=None, lag=0, mu=0.484, sd=0.045, w_other=0.6, **kw):
    n = len(d_o); f = np.arange(n + 1)
    sc = -0.5 * ((f / n - mu) / sd) ** 2 + frame_curve(det_o, d_o, **kw)
    if det_x is not None:
        cx = frame_curve(det_x, d_x, **kw)
        fx = np.clip(f - lag, 0, len(d_x))                # own frame f <-> other frame f - lag
        sc = sc + w_other * cx[fx]
    return int(np.argmax(sc)), sc


def to_window(f_on, convention):
    return int(np.ceil((f_on - 15) / 30)) if convention == "majority" else int(f_on // 30)


if __name__ == "__main__":
    vdf = pd.read_csv("outputs/v2/train_videos.csv"); pose = np.load("outputs/v2/train_pose.npy")
    oof = pd.read_csv("outputs/v5/oof_inst_scores.csv"); oof["w"] = oof.off // 30
    det = {v: ((g.sort_values("w")[f"m_{vdf.behavior[v]}"] - g.sort_values("w")[NORMAL].max(axis=1)).values / 10.0)
           for v, g in oof.groupby("video")}
    P = lambda v: pose[vdf.start[v]:vdf.start[v] + vdf.n[v]]
    for kw in [dict(w_pose=1.0), dict(w_pose=0.5), dict(w_pose=0.0)]:
        err = []
        for (subj, ang, beh), on in ONSET.items():
            other = "side" if ang == "front" else "front"
            v = vdf.index[(vdf.subject == subj) & (vdf.angle == ang) & (vdf.behavior == beh)][0]
            u = vdf.index[(vdf.subject == subj) & (vdf.angle == other) & (vdf.behavior == beh)][0]
            lag = best_lag(signals(P(v)), signals(P(u)))
            f_on, _ = onset_frame(det[v], deviation(P(v)), det[u], deviation(P(u)), lag, **kw)
            err.append(to_window(f_on, "majority") - on)
        e = np.array(err)
        print(kw, f"majority-window vs manual onset: exact {np.mean(e == 0):.2f} within1 {np.mean(abs(e) <= 1):.2f} "
                  f"early {np.sum(e < 0)} late {np.sum(e > 0)}")
