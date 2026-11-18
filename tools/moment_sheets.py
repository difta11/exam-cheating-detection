"""Contact sheets of 1-s windows of TRAIN videos (person crop, frames 0/10/20 of each second) for manual
per-second annotation used to validate the cheating-moment detector.

usage: python tools/moment_sheets.py <out_dir> <subject> <angle> [behaviour ...]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

V2 = Path("outputs/v2")
out, subj, angle = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
behs = sys.argv[4:] or ["Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]
out.mkdir(parents=True, exist_ok=True)
vdf = pd.read_csv(V2 / "train_videos.csv")
pose = np.load(V2 / "train_pose.npy")
for beh in behs:
    r = vdf[(vdf.subject == subj) & (vdf.angle == angle) & (vdf.behavior == beh)].iloc[0]
    path = Path("dataset/train") / Path(r.path).parts[-3] / Path(r.path).parts[-2] / Path(r.path).name
    P = pose[r.start:r.start + r.n]
    v = P[:, 55] > 0
    b = P[v, 51:55] * [1280, 720, 1280, 720]
    x1, y1, x2, y2 = np.percentile(b[:, 0], 5), np.percentile(b[:, 1], 5), np.percentile(b[:, 2], 95), np.percentile(b[:, 3], 95)
    side = int(min(max(x2 - x1, y2 - y1) * 1.2, 720)); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X0, Y0 = int(np.clip(cx - side / 2, 0, 1280 - side)), int(np.clip(cy - side / 2, 0, 720 - side))
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f[Y0:Y0 + side, X0:X0 + side], (150, 150), interpolation=cv2.INTER_AREA))
    n_win = (len(frames) - 30) // 30 + 1
    tiles = []
    for w in range(n_win):
        t = np.hstack([frames[w * 30 + o] for o in (0, 10, 20)])
        cv2.putText(t, str(w), (4, 22), 0, 0.8, (0, 0, 255), 2)
        tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(out / f"s{subj}_{angle}_{beh}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(beh, "windows:", n_win)
