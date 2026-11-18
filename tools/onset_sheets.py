"""Compact onset-annotation sheets for TRAIN videos: one row per video, one person-cropped frame per second
(frame 15 of each 1-s window), 4 videos per sheet.

usage: python tools/onset_sheets.py <out_dir> <angle> <subject> [<subject> ...]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

V2 = Path("outputs/v2")
CHEAT = ["Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]
out, angle, subjects = Path(sys.argv[1]), sys.argv[2], [int(s) for s in sys.argv[3:]]
out.mkdir(parents=True, exist_ok=True)
vdf = pd.read_csv(V2 / "train_videos.csv")
pose = np.load(V2 / "train_pose.npy")
S = 112


def row(r):
    path = Path("dataset/train") / Path(r.path).parts[-3] / Path(r.path).parts[-2] / Path(r.path).name
    P = pose[r.start:r.start + r.n]
    b = P[P[:, 55] > 0, 51:55] * [1280, 720, 1280, 720]
    x1, y1, x2, y2 = np.percentile(b[:, 0], 5), np.percentile(b[:, 1], 5), np.percentile(b[:, 2], 95), np.percentile(b[:, 3], 95)
    side = int(min(max(x2 - x1, y2 - y1) * 1.15, 720)); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X0, Y0 = int(np.clip(cx - side / 2, 0, 1280 - side)), int(np.clip(cy - side / 2, 0, 720 - side))
    cap, i, tiles = cv2.VideoCapture(str(path)), 0, []
    n_win = (r.n - 30) // 30 + 1
    while len(tiles) < n_win:
        ok = cap.grab()
        if not ok:
            break
        if i % 30 == 15:
            _, f = cap.retrieve()
            t = cv2.resize(f[Y0:Y0 + side, X0:X0 + side], (S, S), interpolation=cv2.INTER_AREA)
            cv2.putText(t, str(len(tiles)), (3, 16), 0, 0.5, (0, 0, 255), 2)
            tiles.append(t)
        i += 1
    tiles += [np.zeros((S, S, 3), np.uint8)] * (19 - len(tiles))
    lab = np.full((S, 70, 3), 255, np.uint8)
    cv2.putText(lab, f"s{r.subject}", (2, 40), 0, 0.5, (0, 0, 0), 1); cv2.putText(lab, r.behavior[:7], (2, 70), 0, 0.4, (0, 0, 0), 1)
    return np.hstack([lab] + tiles[:19])


for subj in subjects:
    rows = [row(r) for _, r in vdf[(vdf.subject == subj) & (vdf.angle == angle) & vdf.behavior.isin(CHEAT)]
            .sort_values("behavior").iterrows()]
    cv2.imwrite(str(out / f"onset_s{subj}_{angle}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("subject", subj, "done")
