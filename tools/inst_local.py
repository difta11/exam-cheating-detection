"""Quick local (CPU) instance-level MIL on v3 window features -> OOF per-second scores (same model as v5).

usage: python tools/inst_local.py <out_csv> [config] [epochs] [seeds]
"""
import math, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

torch.set_num_threads(8)
OUT = sys.argv[1]
CFG = sys.argv[2] if len(sys.argv) > 2 else "vjepa"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 60
SEEDS = list(range(int(sys.argv[4]) if len(sys.argv) > 4 else 1))
V2, V3 = Path("outputs/v2"), Path("outputs/v3")
vdf = pd.read_csv(V2 / "train_videos.csv"); wdf = pd.read_csv(V3 / "train_windows.csv")
BEH = ["Writing_Answers", "Thinking", "Looking_Around", "Checking_Smartwatch",
       "Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]

F_ = np.load(V3 / "train_vjepa_fp16.npy")
d = 1024
cols = {"vjepa": np.r_[0:d, 6 * d:F_.shape[1]], "vjepa_grid": np.r_[0:d, 2 * d:F_.shape[1]]}[CFG]
X_np = F_[:, cols].astype(np.float32)
V_LAB, V_BEH = vdf.label.values, vdf.beh_id.values
VID_ROWS = {v: np.where(wdf.video.values == v)[0] for v in range(len(vdf))}


class InstMIL(nn.Module):
    def __init__(self, din, d=128, p=0.3, r=2.0):
        super().__init__()
        self.enc = nn.Sequential(nn.Dropout(0.2), nn.Linear(din, d), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(d, d), nn.GELU(), nn.Dropout(p))
        self.head = nn.Linear(d, 9); self.r = r

    def forward(self, x, mask):
        s = self.head(self.enc(x))
        sm = s.masked_fill(~mask[..., None], -1e4)
        n = mask.sum(1, keepdim=True).float()
        return (torch.logsumexp(self.r * sm, 1) - torch.log(n)) / self.r, s


def pad(bags, X):
    L = max(len(b) for b in bags)
    idx = np.zeros((len(bags), L), np.int64); mask = np.zeros((len(bags), L), bool)
    for i, b in enumerate(bags):
        idx[i, :len(b)] = b; mask[i, :len(b)] = True
    return X[torch.from_numpy(idx)], torch.from_numpy(mask)


def train(X, videos, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = InstMIL(X.shape[1]); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    reps, bs = 4, 32
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EPOCHS * math.ceil(len(videos) * reps / bs), pct_start=0.1)
    for _ in range(EPOCHS):
        m.train(); order = np.random.permutation(np.repeat(videos, reps))
        for i in range(0, len(order), bs):
            bags = []
            for v in order[i:i + bs]:
                seq = VID_ROWS[v][random.randint(0, 1)::2]
                keep = seq[np.random.rand(len(seq)) > 0.15]
                bags.append(keep if len(keep) else seq)
            x, mk = pad(bags, X)
            pooled, _ = m(x + 0.1 * torch.randn_like(x), mk)
            vv = order[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(pooled[:, 0], torch.as_tensor(V_LAB[vv]).float()) \
                + 0.5 * F.cross_entropy(pooled[:, 1:], torch.as_tensor(V_BEH[vv]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    return m.eval()


EV = np.where(wdf.off.values % 30 == 0)[0]
EVV = wdf.video.values[EV]
S = np.zeros((len(EV), 9))
t0 = time.time()
for fold, (tr, va) in enumerate(GroupKFold(5).split(vdf, groups=vdf.subject)):
    rows = np.concatenate([VID_ROWS[v] for v in tr])
    mu, sd = X_np[rows].mean(0), X_np[rows].std(0) + 1e-3
    X = torch.from_numpy(np.clip((X_np - mu) / sd, -6, 6)).float()
    sel = np.isin(EVV, va)
    for seed in SEEDS:
        m = train(X, tr, seed + 100 * fold)
        with torch.no_grad():
            S[sel] += m.head(m.enc(X[torch.from_numpy(EV[sel])])).numpy() / len(SEEDS)
    print(f"fold {fold} {time.time() - t0:.0f}s", flush=True)
print("window AUC cheat-video vs normal-video:", round(roc_auc_score(V_LAB[EVV], S[:, 0]), 4))
pd.DataFrame(S, columns=["moment"] + [f"m_{b}" for b in BEH]).assign(
    video=EVV, off=wdf.off.values[EV], label=V_LAB[EVV], behavior=vdf.behavior.values[EVV]).to_csv(OUT, index=False)
