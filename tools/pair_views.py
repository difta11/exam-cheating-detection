"""Pair front/side videos of the same take by synchronised body motion (pose), validate on train.

usage: python tools/pair_views.py   (project root)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

V2, ST = Path("outputs/v2"), Path("outputs/structure")
UPPER = list(range(11))          # nose, eyes, ears, shoulders, elbows, wrists


def signals(P):
    """P: (n_frames, 56) pose of one video -> (n_frames, k) z-scored motion signals"""
    xy = P[:, :51].reshape(-1, 17, 3)
    px = xy[..., 0] * 640; py = xy[..., 1] * 360; cf = xy[..., 2]
    bh = np.median(np.maximum((P[:, 54] - P[:, 52]) * 360, 1))
    d = np.sqrt(np.diff(px, axis=0, prepend=px[:1]) ** 2 + np.diff(py, axis=0, prepend=py[:1]) ** 2) / bh
    ok = (cf > 0.3) & (np.roll(cf, 1, axis=0) > 0.3)
    d = np.where(ok, np.minimum(d, 0.2), 0)
    sh_y = (py[:, 5] + py[:, 6]) / 2
    sigs = [d[:, UPPER].sum(1), d[:, [9, 10]].sum(1), d[:, [0, 1, 2, 3, 4]].sum(1),
            (py[:, 0] - sh_y) / bh, (py[:, 9] + py[:, 10]) / 2 / bh]
    S = np.stack(sigs, 1)
    k = np.ones(7) / 7
    S = np.stack([np.convolve(s, k, mode="same") for s in S.T], 1)
    return (S - S.mean(0)) / (S.std(0) + 1e-6)


def xcorr(A, B, maxlag=150, minov=150):
    """max over lags of mean (over signals) Pearson correlation of overlapping parts"""
    best = -1
    for lag in range(-maxlag, maxlag + 1, 2):
        a = A[max(lag, 0):]; b = B[max(-lag, 0):]
        n = min(len(a), len(b))
        if n < minov:
            continue
        a, b = a[:n], b[:n]
        a = (a - a.mean(0)) / (a.std(0) + 1e-6); b = (b - b.mean(0)) / (b.std(0) + 1e-6)
        c = (a * b).mean(0)
        best = max(best, float(np.mean(c[:3]) * 0.6 + np.mean(c[3:]) * 0.4))
    return best


def pair(front_sigs, side_sigs):
    C = np.array([[xcorr(f, s) for s in side_sigs] for f in front_sigs])
    r, c = linear_sum_assignment(-C)
    return dict(zip(r, c)), C


if __name__ == "__main__":
    vdf = pd.read_csv(V2 / "train_videos.csv")
    tp_all = np.load(V2 / "train_pose.npy")
    sig = [signals(tp_all[s:s + n]) for s, n in zip(vdf.start, vdf.n)]
    ok_total, n_total, margins = 0, 0, []
    for subj, g in vdf.groupby("subject"):
        fr = g[g.angle == "front"].sort_values("behavior"); sd = g[g.angle == "side"].sort_values("behavior")
        m, C = pair([sig[i] for i in fr.index], [sig[i] for i in sd.index])
        ok = sum(fr.behavior.values[i] == sd.behavior.values[j] for i, j in m.items())
        ok_total += ok; n_total += len(m)
        diag = np.diag(C); off = np.where(~np.eye(8, dtype=bool), C, -9).max(1)
        margins += list(diag - off)
    print(f"train pairing accuracy: {ok_total}/{n_total} = {ok_total / n_total:.3f}; median true-pair margin {np.median(margins):.3f}")

    # test: build per-video pose sequences from the re-linked chains
    st = json.load(open(ST / "test_videos.json"))
    te = np.load(V2 / "test_pose.npy")
    tsig = [signals(np.concatenate([te[w] for w in ch])) for ch in st["chains"]]
    groups, angles = np.array(st["group"]), np.array(st["angle"])
    pairs = {}
    for gf in sorted(set(groups[angles == "front"])):
        fidx = np.where(groups == gf)[0]
        # side group of the same person: best total correlation over side groups
        best = None
        for gs in sorted(set(groups[angles == "side"])):
            sidx = np.where(groups == gs)[0]
            m, C = pair([tsig[i] for i in fidx], [tsig[j] for j in sidx])
            score = np.mean([C[i, j] for i, j in m.items()])
            if best is None or score > best[0]:
                best = (score, gs, sidx, m, C)
        score, gs, sidx, m, C = best
        print(f"front group {gf} <-> side group {gs}: mean paired corr {score:.3f}")
        for i, j in m.items():
            pairs[int(fidx[i])] = int(sidx[j])
    json.dump(pairs, open(ST / "test_view_pairs.json", "w"))
