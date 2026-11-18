"""Per-second test labels: inside videos assigned to a cheat behaviour (v5 joint front+side assignment), every
1-s window from the estimated behaviour onset to the end is 1; everything else is 0.
Onset = position prior + detector change point + paired view of the same take (tools/onset_model.py).

usage: python tools/make_onset_submission.py   (project root)
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "tools")
from pair_views import signals
from onset_model import best_lag, estimate, BEH, NORMAL

st = json.load(open("outputs/structure/test_videos.json"))
chains = st["chains"]
sub = pd.read_csv("dataset/sample_submission.csv")
inst = pd.read_csv("outputs/v5/test_inst_scores.csv")
assert (inst.id == sub.id).all()
post = pd.read_csv("outputs/v5_post/test_videos_posterior.csv")
beh = post.behaviour.values
tp = np.load("outputs/v2/test_pose.npy")

lab = np.zeros(len(sub), int)
rows = []
for f, s in st["pairs"]:
    assert beh[f] == beh[s]
    b = beh[f]
    if BEH.index(b) < 4:
        continue
    sf = ((inst.loc[chains[f], f"m_{b}"].values - inst.loc[chains[f], NORMAL].max(axis=1).values) / 10.0)
    ss = ((inst.loc[chains[s], f"m_{b}"].values - inst.loc[chains[s], NORMAL].max(axis=1).values) / 10.0)
    lag = best_lag(signals(np.concatenate([tp[w] for w in chains[f]])), signals(np.concatenate([tp[w] for w in chains[s]])))
    on_f = estimate(sf, len(sf), ss, len(ss), lag / 30.0)
    on_s = estimate(ss, len(ss), sf, len(sf), -lag / 30.0)
    lab[np.array(chains[f])[on_f:]] = 1
    lab[np.array(chains[s])[on_s:]] = 1
    rows.append(dict(front=f, side=s, behaviour=b, n_front=len(sf), n_side=len(ss), lag=lag,
                     onset_front=on_f, onset_side=on_s, pos_front=round(on_f / len(sf), 2), pos_side=round(on_s / len(ss), 2)))
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(r.to_string(index=False))
print("onset position quantiles:", np.round(np.quantile(np.r_[r.pos_front, r.pos_side], [0, .1, .5, .9, 1]), 2).tolist())
out = Path("outputs/v5_post")
sub.assign(label=lab).to_csv(out / "submission_onset.csv", index=False)
r.to_csv(out / "test_onsets.csv", index=False)
p45 = pd.read_csv(out / "submission_pos45.csv")
print("ones:", lab.sum(), "| differs from submission_pos45.csv on", int((p45.label != lab).sum()), "windows")
