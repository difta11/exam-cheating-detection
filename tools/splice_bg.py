"""Detect the clip boundary (splice) inside a video from BACKGROUND change only: the camera is static within a
clip, so the background (outside the person box) changes abruptly only where two clips were joined.

Train:  python tools/splice_bg.py train <out_csv> [subject ...]
Test:   python tools/splice_bg.py test  <out_csv>
Output per video: splice frame (index of the first frame of the 2nd clip), its strength (jump / median) and
the 2nd-best jump strength.
"""
import json, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import cv2

W, H = 160, 90


def bg_jumps(frames, boxes):
    """frames: list of (H,W) float32 gray; boxes: (n,4) normalised person boxes -> per-transition bg difference"""
    d = np.zeros(len(frames))
    for t in range(1, len(frames)):
        m = np.ones((H, W), bool)
        for b in (boxes[t - 1], boxes[t]):
            if b[2] > b[0]:
                x1, y1, x2, y2 = b
                bw, bh = x2 - x1, y2 - y1
                x1, x2 = int(max(0, (x1 - 0.1 * bw) * W)), int(min(W, (x2 + 0.1 * bw) * W))
                y1, y2 = int(max(0, (y1 - 0.1 * bh) * H)), int(min(H, (y2 + 0.1 * bh) * H))
                m[y1:y2, x1:x2] = False
        if m.sum() < 200:
            m[:] = True
        d[t] = np.abs(frames[t] - frames[t - 1])[m].mean()
    return d


def summarise(d, lo=0.15, hi=0.85):
    """strongest background jump in the interior of the video (start/stop handling of the phone excluded)"""
    med = np.median(d[1:]) + 1e-3
    a, b = int(lo * len(d)), int(hi * len(d))
    k = a + int(np.argmax(d[a:b]))
    rest = d[a:b].copy(); rest[max(0, k - a - 3):k - a + 4] = 0
    return dict(splice=k, strength=d[k] / med, second=rest.max() / med, n_frames=len(d),
                profile=";".join(f"{x / med:.1f}" for x in d))


def gray(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)


def train_video(args):
    i, path, P = args
    cv2.setNumThreads(1)
    cap, fr = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(gray(f))
    n = min(len(fr), len(P))
    return i, summarise(bg_jumps(fr[:n], P[:n, 51:55]))


def test_video(args):
    v, seqs, P = args
    cv2.setNumThreads(1)
    fr = [gray(cv2.imread(str(Path("dataset/test") / s / f"{k:02d}.jpg"))) for s in seqs for k in range(1, 31)]
    return v, summarise(bg_jumps(fr, P.reshape(-1, 56)[:, 51:55]))


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "train":
        vdf = pd.read_csv("outputs/v2/train_videos.csv")
        pose = np.load("outputs/v2/train_pose.npy")
        subs = [int(s) for s in sys.argv[3:]]
        sel = vdf[vdf.subject.isin(subs)] if subs else vdf
        jobs = [(i, Path("dataset/train") / Path(r.path).parts[-3] / Path(r.path).parts[-2] / Path(r.path).name,
                 pose[r.start:r.start + r.n]) for i, r in sel.iterrows()]
        with ProcessPoolExecutor(8) as ex:
            res = dict(ex.map(train_video, jobs))
        df = sel[["subject", "angle", "behavior", "label", "n"]].join(pd.DataFrame(res).T)
        df["splice_win"] = df.splice / 30
    else:
        st = json.load(open("outputs/structure/test_videos.json"))
        ids = pd.read_csv("dataset/sample_submission.csv").id.tolist()
        tp = np.load("outputs/v2/test_pose.npy")
        jobs = [(v, [ids[w] for w in ch], tp[ch]) for v, ch in enumerate(st["chains"])]
        with ProcessPoolExecutor(8) as ex:
            res = dict(ex.map(test_video, jobs))
        df = pd.DataFrame(res).T
        df["angle"] = st["angle"]; df["group"] = st["group"]
        df["splice_win"] = df.splice / 30
    df.to_csv(out)
    pd.set_option("display.width", 220)
    print(df.round(2).to_string())
