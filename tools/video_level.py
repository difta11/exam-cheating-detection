"""Video-level decisions on top of v3 window features (runs locally on CPU).

Uses the 1-s window features saved by `notebooks/cheat_video_mil.py` (V-JEPA 2, VideoMAE, pose stats),
trains the same MIL head (GroupKFold by subject, identical folds), and scores whole videos:
  thr      : video cheat-probability > 0.5
  rank     : top-4 of the 8 videos of each (subject, angle) = cheat
  assign   : Hungarian assignment of the 8 videos of each (subject, angle) to the 8 behaviours
Test videos = re-linked chains (tools/test_link_chains.py), groups = tools/test_subject_groups.py.

usage: python tools/video_level.py  (from the project root)
"""
import json, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

torch.set_num_threads(8)
V2, V3, ST, OUT = Path("outputs/v2"), Path("outputs/v3"), Path("outputs/structure"), Path("outputs/v4")
OUT.mkdir(parents=True, exist_ok=True)
T = 30
SEEDS, EPOCHS = [42, 7], 40

vdf = pd.read_csv(V2 / "train_videos.csv")
wdf = pd.read_csv(V3 / "train_windows.csv")
comp = json.load(open(ST / "test_components.json"))
grp = json.load(open(ST / "test_subject_groups.json"))
test_ids = comp["ids"]
chains = [c["members"] for c in comp["chains"]]
n_te = len(test_ids)

# ---------------- features ----------------
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def featurize(W):
    B, Tn, _ = W.shape
    kp = W[..., :51].reshape(B, Tn, 17, 3)
    xy, cf = kp[..., :2], kp[..., 2]
    box, bc = W[..., 51:55], W[..., 55:]
    valid = (bc > 0).float()
    nv = valid.sum(1).clamp(min=1)
    sh = (xy[:, :, 5] + xy[:, :, 6]) / 2
    ref = (sh * valid).sum(1) / nv
    bh = ((box[..., 3] - box[..., 1]) * valid[..., 0]).sum(1) / nv[:, 0]
    scale = bh.clamp(min=0.05)[:, None, None, None]
    rel = (xy - ref[:, None, None]) / scale * valid[..., None]
    vel = torch.diff(rel, dim=1, prepend=rel[:, :1])
    return torch.cat([xy.flatten(2), cf, rel.flatten(2), vel.flatten(2) * 10, box, bc], -1)


def window_stats(f):
    return torch.cat([f.mean(1), f.std(1), f.amin(1), f.amax(1), (f[:, 1:] - f[:, :-1]).abs().mean(1)], 1)


@torch.no_grad()
def pose_stats(Wall, bs=2048):
    Wall = torch.from_numpy(np.ascontiguousarray(Wall)).float()
    return torch.cat([window_stats(featurize(Wall[i:i + bs])) for i in range(0, len(Wall), bs)]).numpy()


train_pose, test_pose = np.load(V2 / "train_pose.npy"), np.load(V2 / "test_pose.npy")
P_TR = pose_stats(train_pose[wdf.gstart.values[:, None] + np.arange(T)])
P_TE = pose_stats(test_pose)
FE = {k: (np.load(V3 / f"train_{k}_fp16.npy"), np.load(V3 / f"test_{k}_fp16.npy")) for k in ["vjepa", "vmae"]}
D = {"vjepa": 1024, "vmae": 768}
NL = {"vjepa": FE["vjepa"][0].shape[1] - 6 * 1024, "vmae": FE["vmae"][0].shape[1] - 6 * 768}


def block(name, which):
    if name == "pose":
        return P_TR, P_TE
    d = D[name]
    sl = dict(pooled=np.arange(0, d), grid=np.arange(2 * d, 6 * d), logits=np.arange(6 * d, 6 * d + NL[name]))
    cols = np.concatenate([sl[w] for w in which])
    return FE[name][0][:, cols].astype(np.float32), FE[name][1][:, cols].astype(np.float32)


CONFIGS = {"vjepa": [("vjepa", ["pooled", "logits"])], "pose": [("pose", None)],
           "vjepa_grid": [("vjepa", ["pooled", "grid", "logits"])], "vmae": [("vmae", ["pooled", "logits"])]}

# ---------------- MIL ----------------
class MIL(nn.Module):
    def __init__(self, din, d=128, p=0.4):
        super().__init__()
        self.enc = nn.Sequential(nn.Dropout(0.25), nn.Linear(din, d), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(d, d), nn.GELU())
        self.att = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
        self.drop = nn.Dropout(p)
        self.out_bin, self.out_beh = nn.Linear(2 * d, 1), nn.Linear(2 * d, 8)

    def forward(self, x, mask):
        h = self.enc(x)
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(1)
        z = torch.cat([(a[..., None] * h).sum(1), h.masked_fill(~mask[..., None], -1e4).amax(1)], 1)
        z = self.drop(z)
        return self.out_bin(z).squeeze(1), self.out_beh(z)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


V_LAB, V_BEH = vdf.label.values, vdf.beh_id.values
VID_ROWS = {v: np.where(wdf.video.values == v)[0] for v in range(len(vdf))}
VID_EVAL = {v: r[wdf.off.values[r] % T == 0] for v, r in VID_ROWS.items()}   # 1-s aligned windows


def pad(bags, X):
    L = max(len(b) for b in bags)
    idx = np.zeros((len(bags), L), np.int64); mask = np.zeros((len(bags), L), bool)
    for i, b in enumerate(bags):
        idx[i, :len(b)] = b; mask[i, :len(b)] = True
    return X[torch.from_numpy(idx)], torch.from_numpy(mask)


def train_mil(X, videos, seed):
    seed_all(seed)
    model = MIL(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    steps = EPOCHS * math.ceil(len(videos) * 8 / 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    for _ in range(EPOCHS):
        model.train()
        order = np.random.permutation(np.repeat(videos, 8))
        for i in range(0, len(order), 64):
            L = random.choice([1, 2, 4, 8, 12, 15, 30])
            bags = []
            for v in order[i:i + 64]:
                seq = VID_ROWS[v][random.randint(0, 1)::2]
                l = min(L, len(seq)); s0 = random.randint(0, len(seq) - l)
                bags.append(seq[s0:s0 + l])
            x, m = pad(bags, X)
            lb, lh = model(x + 0.1 * torch.randn_like(x), m)
            vv = order[i:i + 64]
            loss = F.binary_cross_entropy_with_logits(lb, torch.as_tensor(V_LAB[vv]).float() * 0.9 + 0.05) \
                + 0.5 * F.cross_entropy(lh, torch.as_tensor(V_BEH[vv]), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    return model.eval()


@torch.no_grad()
def predict(model, bags, X):
    x, m = pad(bags, X)
    lb, lh = model(x, m)
    return torch.sigmoid(lb).numpy(), lh.softmax(1).numpy()


folds = list(GroupKFold(n_splits=5).split(vdf, groups=vdf.subject))
res_va = {}      # config -> (bin (n_videos,), beh (n_videos, 8))
res_te = {}
t0 = time.time()
for cname, parts in CONFIGS.items():
    bl = [block(n, w) for n, w in parts]
    Xtr_np, Xte_np = np.concatenate([b[0] for b in bl], 1), np.concatenate([b[1] for b in bl], 1)
    pb, ph = np.zeros(len(vdf)), np.zeros((len(vdf), 8))
    tb, th = np.zeros(len(chains)), np.zeros((len(chains), 8))
    for fold, (tr, va) in enumerate(folds):
        rows = np.concatenate([VID_ROWS[v] for v in tr])
        mu, sd = Xtr_np[rows].mean(0), Xtr_np[rows].std(0) + 1e-3
        Xtr = torch.from_numpy(np.clip((Xtr_np - mu) / sd, -6, 6)).float()
        Xte = torch.from_numpy(np.clip((Xte_np - mu) / sd, -6, 6)).float()
        for seed in SEEDS:
            m = train_mil(Xtr, tr, seed + 100 * fold)
            b, h = predict(m, [VID_EVAL[v] for v in va], Xtr)
            pb[va] += b / len(SEEDS); ph[va] += h / len(SEEDS)
            b, h = predict(m, chains, Xte)
            tb += b / (len(SEEDS) * len(folds)); th += h / (len(SEEDS) * len(folds))
    res_va[cname], res_te[cname] = (pb, ph), (tb, th)
    print(f"{cname:11s} video auc bin={roc_auc_score(V_LAB, pb):.4f} beh-cheat={roc_auc_score(V_LAB, ph[:, 4:].sum(1)):.4f} "
          f"beh-acc={(ph.argmax(1) == V_BEH).mean():.4f} {time.time() - t0:.0f}s", flush=True)

# ---------------- decisions ----------------
def assign(beh_p, n_extra_penalty=3.0):
    """rows = videos of one (subject, angle) group; each behaviour used once (a 2nd use costs extra)"""
    cost = -np.log(np.clip(beh_p, 1e-6, 1))
    cost = np.concatenate([cost, cost + n_extra_penalty], 1)
    r, c = linear_sum_assignment(cost)
    out = np.zeros(len(beh_p), int); out[r] = c % 8
    return out


def decide(pbin, pbeh, groups):
    p = 0.5 * pbin + 0.5 * pbeh[:, 4:].sum(1)
    d = {"thr": (p > 0.5).astype(int)}
    rank = np.zeros(len(p), int); asg = np.zeros(len(p), int); beh = np.zeros(len(p), int)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        k = int(round(len(idx) / 2))
        rank[idx[np.argsort(-p[idx])[:k]]] = 1
        beh[idx] = assign(pbeh[idx])
    d["rank"], d["assign"], d["beh"] = rank, (beh >= 4).astype(int), beh
    return p, d


w_vid = np.array([len(VID_EVAL[v]) for v in range(len(vdf))])   # windows per video -> window-level accuracy
g_tr = (vdf.subject.astype(str) + "_" + vdf.angle).values
report = {}
combos = {c: [c] for c in CONFIGS} | {"vjepa+pose": ["vjepa", "pose"], "vjepa+pose+grid": ["vjepa", "pose", "vjepa_grid"],
                                      "all4": list(CONFIGS)}
for name, cs in combos.items():
    pb = np.mean([res_va[c][0] for c in cs], 0); ph = np.mean([res_va[c][1] for c in cs], 0)
    p, d = decide(pb, ph, g_tr)
    report[name] = {k: round(float(np.average(d[k] == V_LAB, weights=w_vid)), 4) for k in ["thr", "rank", "assign"]}
    report[name]["beh_acc"] = round(float((d["beh"] == V_BEH).mean()), 4)
    report[name]["auc"] = round(float(roc_auc_score(V_LAB, p)), 4)
    print(f"{name:16s}", report[name])
json.dump(report, open(OUT / "video_level_cv.json", "w"), indent=1)
np.savez(OUT / "video_level_preds.npz", **{f"va_{c}_bin": res_va[c][0] for c in CONFIGS},
         **{f"va_{c}_beh": res_va[c][1] for c in CONFIGS}, **{f"te_{c}_bin": res_te[c][0] for c in CONFIGS},
         **{f"te_{c}_beh": res_te[c][1] for c in CONFIGS})
print("saved", OUT)
