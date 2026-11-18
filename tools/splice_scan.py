"""Look for an abrupt cut (splice) inside TRAIN videos: per-frame difference of small grayscale frames.

usage: python tools/splice_scan.py <out_csv> [subject ...]
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import cv2


def diff_signal(path):
    cv2.setNumThreads(1)
    cap, prev, d = cv2.VideoCapture(str(path)), None, []
    while True:
        ok = cap.grab()
        if not ok:
            break
        ok, f = cap.retrieve()
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
        d.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    return np.array(d)


def summarise(args):
    i, path = args
    d = diff_signal(path)
    med = np.median(d[1:]) + 1e-3
    k = int(np.argmax(d))
    top2 = np.sort(d)[-2]
    return dict(i=i, n=len(d), peak_frame=k, peak_pos=k / len(d), peak_ratio=d[k] / med, second_ratio=top2 / med)


if __name__ == "__main__":
    vdf = pd.read_csv("outputs/v2/train_videos.csv")
    subs = [int(s) for s in sys.argv[2:]]
    if subs:
        vdf = vdf[vdf.subject.isin(subs)]
    paths = [(i, Path("dataset/train") / Path(p).parts[-3] / Path(p).parts[-2] / Path(p).name) for i, p in zip(vdf.index, vdf.path)]
    with ProcessPoolExecutor(8) as ex:
        res = pd.DataFrame(list(ex.map(summarise, paths))).set_index("i")
    out = vdf[["subject", "angle", "behavior", "label"]].join(res)
    out.to_csv(sys.argv[1])
    pd.set_option("display.width", 200)
    print(out.sort_values(["subject", "angle", "behavior"]).round(2).to_string())
