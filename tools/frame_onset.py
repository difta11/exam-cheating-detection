"""Frame-level onset cue from pose: deviation of each frame from the subject's own 'writing state'
(first 30% of the same video), validated against the manual window onsets (tools/onset_model.py ONSET).

usage: python tools/frame_onset.py
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "tools")
from onset_model import ONSET

UPPER = list(range(11))


def frame_feats(P):
    kp = P[:, :51].reshape(-1, 17, 3)
    x, y, c = kp[..., 0] * 640, kp[..., 1] * 360, kp[..., 2]
    bh = np.median(np.maximum((P[:, 54] - P[:, 52]) * 360, 1))
    shx, shy = (x[:, 5] + x[:, 6]) / 2, (y[:, 5] + y[:, 6]) / 2
    rel = np.concatenate([(x[:, UPPER] - shx[:, None]) / bh, (y[:, UPPER] - shy[:, None]) / bh], 1)
    rel = np.where(np.concatenate([c[:, UPPER], c[:, UPPER]], 1) > 0.3, rel, np.nan)
    rel = pd.DataFrame(rel).interpolate(limit_direction="both").fillna(0).values
    box = np.c_[(P[:, 51] + P[:, 53]) / 2, (P[:, 52] + P[:, 54]) / 2, P[:, 54] - P[:, 52]]
    return np.c_[rel, box * 3]


def deviation(P, base_frac=0.3, smooth=15):
    F = frame_feats(P)
    nb = max(30, int(base_frac * len(F)))
    mu, sd = F[:nb].mean(0), F[:nb].std(0) + 0.02
    z = np.abs((F - mu) / sd)
    d = np.sqrt((np.minimum(z, 8) ** 2).mean(1))
    return pd.Series(d).rolling(smooth, center=True, min_periods=1).median().values


if __name__ == "__main__":
    vdf = pd.read_csv("outputs/v2/train_videos.csv")
    pose = np.load("outputs/v2/train_pose.npy")
    rows = []
    for (subj, ang, beh), on in ONSET.items():
        v = vdf.index[(vdf.subject == subj) & (vdf.angle == ang) & (vdf.behavior == beh)][0]
        d = deviation(pose[vdf.start[v]:vdf.start[v] + vdf.n[v]])
        nwin = (vdf.n[v] - 30) // 30 + 1
        per_win = [d[30 * w:30 * w + 30].mean() for w in range(nwin)]
        thr = np.percentile(d[: int(0.3 * len(d))], 95) * 1.5
        lo = int(0.3 * len(d))
        above = np.where(pd.Series(d[lo:] > thr).rolling(20, min_periods=20).sum().values >= 18)[0]
        f_on = lo + above[0] - 19 if len(above) else -1
        rows.append(dict(key=f"s{subj}-{ang}-{beh[:12]}", truth=on, onset_frame=f_on,
                         win_majority=int(np.ceil((f_on - 15) / 30)) if f_on >= 0 else -1,
                         win_anyframe=f_on // 30 if f_on >= 0 else -1,
                         profile=" ".join(f"{x:.1f}" for x in per_win)))
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(r.to_string(index=False))
    for k in ["win_majority", "win_anyframe"]:
        e = r[k] - r.truth
        print(f"{k}: exact {np.mean(e == 0):.2f} within1 {np.mean(e.abs() <= 1):.2f} mean err {e.mean():+.2f}")
