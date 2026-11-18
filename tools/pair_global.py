"""Global front<->side pairing of all test videos (no group prior); side subject = subject of its paired front video."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "tools")
from pair_views import signals, xcorr
st = json.load(open("outputs/structure/test_videos.json"))
te = np.load("outputs/v2/test_pose.npy")
sig = [signals(np.concatenate([te[w] for w in ch])) for ch in st["chains"]]
ang, grp = np.array(st["angle"]), np.array(st["group"])
F_, S_ = np.where(ang == "front")[0], np.where(ang == "side")[0]
C = np.array([[xcorr(sig[f], sig[s]) for s in S_] for f in F_])
r, c = linear_sum_assignment(-C)
pairs = {int(F_[i]): int(S_[j]) for i, j in zip(r, c)}
side_from_front = {s: int(grp[f]) for f, s in pairs.items()}
# consistency with the clothing/hand-corrected groups: each front group must map to one side group
m = {}
for f, s in pairs.items():
    m.setdefault(int(grp[f]), []).append(int(grp[s]))
print({k: sorted(v) for k, v in m.items()})
marg = [C[i, j] - np.delete(C[i], j).max() for i, j in zip(r, c)]
print("pair corr median %.3f, min margin %.3f, #margin<0.05: %d" % (np.median(C[r, c]), min(marg), sum(np.array(marg) < 0.05)))
json.dump({"pairs": pairs}, open("outputs/structure/test_view_pairs_global.json", "w"))
