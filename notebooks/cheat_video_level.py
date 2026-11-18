# %% [markdown]
# # Exam Cheating Detection v4 — video-level decisions on v3 features
#
# The test set consists of ~96 source videos cut into shuffled 1-s windows. Re-linking windows by frame
# continuity (tools/test_link_chains.py) recovers the videos; clustering them by angle (pose) and clothing
# (tools/test_subject_groups.py, side groups checked by eye) gives 6 subjects x 2 angles x 8 videos —
# the same layout as every training subject (one video per behaviour and angle, 4 normal + 4 cheat).
#
# This notebook trains the multiple-instance head on whole-video bags of the v3 window features
# (V-JEPA 2 / VideoMAE / pose), GroupKFold by subject, and compares three decisions per video:
# * `thr`    – cheat probability > 0.5 (needs only the re-linked videos)
# * `rank`   – top-4 of the 8 videos of a (subject, angle) group are cheat
# * `assign` – Hungarian assignment of the 8 videos of a group to the 8 behaviours
#
# Outputs: `submission_pure.csv` (v3 per-window, no test structure), `submission_video.csv` (thr),
# `submission_group.csv` (best of rank/assign), `cv_report.json`.

# %%
import os, json, math, random, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./work")
WORK.mkdir(parents=True, exist_ok=True)


def find_dir(required, bases=("/kaggle/input", "./dataset", "./outputs")):
    for base in map(Path, bases):
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            if all(r in filenames or r in dirnames for r in required):
                return Path(dirpath)
            dirnames[:] = [d for d in dirnames if not d.startswith(("test_seq_", "subject_"))]
    raise FileNotFoundError(required)


DATA = find_dir(["sample_submission.csv", "train"])
V2 = find_dir(["train_pose.npy", "test_pose.npy", "train_videos.csv"])
V3 = find_dir(["train_vjepa_fp16.npy", "train_windows.csv", "submission.csv"])
print("DATA", DATA, "| V2", V2, "| V3", V3, "| device", DEV)

_LOCAL = Path("outputs/structure/test_videos.json")   # embedded at build time for Kaggle (tools/build_v4.py)
STRUCTURE = json.load(open(_LOCAL)) if _LOCAL.exists() else json.loads(r'''__STRUCTURE__''')
test_ids = pd.read_csv(DATA / "sample_submission.csv").id.tolist()
chains, groups = STRUCTURE["chains"], np.array(STRUCTURE["group"])
assert sorted(sum(chains, [])) == list(range(len(test_ids)))
print(len(chains), "test videos;", np.bincount(groups).tolist(), "videos per (subject, angle) group")

T = 30
vdf = pd.read_csv(V2 / "train_videos.csv")
wdf = pd.read_csv(V3 / "train_windows.csv")

# %% [markdown]
# ## Features (same as v3)

# %%
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
def pose_stats(Wall, bs=4096):
    Wall = torch.from_numpy(np.ascontiguousarray(Wall)).float()
    return torch.cat([window_stats(featurize(Wall[i:i + bs].to(DEV))).cpu() for i in range(0, len(Wall), bs)]).numpy()


train_pose, test_pose = np.load(V2 / "train_pose.npy"), np.load(V2 / "test_pose.npy")
P_TR = pose_stats(train_pose[wdf.gstart.values[:, None] + np.arange(T)])
P_TE = pose_stats(test_pose)
FE = {k: (np.load(V3 / f"train_{k}_fp16.npy"), np.load(V3 / f"test_{k}_fp16.npy")) for k in ["vjepa", "vmae"]}
HID = {"vjepa": 1024, "vmae": 768}


def block(name, which):
    if name == "pose":
        return P_TR, P_TE
    d = HID[name]
    nl = FE[name][0].shape[1] - 6 * d
    sl = dict(pooled=np.arange(0, d), grid=np.arange(2 * d, 6 * d), logits=np.arange(6 * d, 6 * d + nl))
    cols = np.concatenate([sl[w] for w in which])
    return FE[name][0][:, cols].astype(np.float32), FE[name][1][:, cols].astype(np.float32)


CONFIGS = {"vjepa": [("vjepa", ["pooled", "logits"])], "pose": [("pose", None)],
           "vjepa_grid": [("vjepa", ["pooled", "grid", "logits"])], "vmae": [("vmae", ["pooled", "logits"])],
           "all": [("vjepa", ["pooled", "logits"]), ("vmae", ["pooled", "logits"]), ("pose", None)]}

# %% [markdown]
# ## Multiple-instance head on whole-video bags

# %%
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
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


V_LAB, V_BEH = vdf.label.values, vdf.beh_id.values
VID_ROWS = {v: np.where(wdf.video.values == v)[0] for v in range(len(vdf))}
VID_EVAL = {v: r[wdf.off.values[r] % T == 0] for v, r in VID_ROWS.items()}   # 1-s windows aligned like the test
SEEDS, EPOCHS = ([42], 1) if os.environ.get("SMOKE") == "1" else ([42, 7, 2026], 60)


def pad(bags, X):
    L = max(len(b) for b in bags)
    idx = np.zeros((len(bags), L), np.int64); mask = np.zeros((len(bags), L), bool)
    for i, b in enumerate(bags):
        idx[i, :len(b)] = b; mask[i, :len(b)] = True
    return X[torch.from_numpy(idx).to(DEV)], torch.from_numpy(mask).to(DEV)


def train_mil(X, videos, seed):
    seed_all(seed)
    model = MIL(X.shape[1]).to(DEV)
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
            loss = F.binary_cross_entropy_with_logits(lb, torch.as_tensor(V_LAB[vv], device=DEV).float() * 0.9 + 0.05) \
                + 0.5 * F.cross_entropy(lh, torch.as_tensor(V_BEH[vv], device=DEV), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    return model.eval()


@torch.no_grad()
def predict(model, bags, X):
    x, m = pad(bags, X)
    lb, lh = model(x, m)
    return torch.sigmoid(lb).cpu().numpy(), lh.softmax(1).cpu().numpy()


folds = list(GroupKFold(n_splits=5).split(vdf, groups=vdf.subject))
res_va, res_te = {}, {}
t0 = time.time()
for cname, parts in CONFIGS.items():
    bl = [block(n, w) for n, w in parts]
    Xtr_np, Xte_np = np.concatenate([b[0] for b in bl], 1), np.concatenate([b[1] for b in bl], 1)
    pb, ph = np.zeros(len(vdf)), np.zeros((len(vdf), 8))
    tb, th = np.zeros(len(chains)), np.zeros((len(chains), 8))
    for fold, (tr, va) in enumerate(folds):
        rows = np.concatenate([VID_ROWS[v] for v in tr])
        mu, sd = Xtr_np[rows].mean(0), Xtr_np[rows].std(0) + 1e-3
        Xtr = torch.from_numpy(np.clip((Xtr_np - mu) / sd, -6, 6)).float().to(DEV)
        Xte = torch.from_numpy(np.clip((Xte_np - mu) / sd, -6, 6)).float().to(DEV)
        for seed in SEEDS:
            m = train_mil(Xtr, tr, seed + 100 * fold)
            b, h = predict(m, [VID_EVAL[v] for v in va], Xtr)
            pb[va] += b / len(SEEDS); ph[va] += h / len(SEEDS)
            b, h = predict(m, chains, Xte)
            tb += b / (len(SEEDS) * len(folds)); th += h / (len(SEEDS) * len(folds))
    res_va[cname], res_te[cname] = (pb, ph), (tb, th)
    print(f"{cname:11s} video auc bin={roc_auc_score(V_LAB, pb):.4f} beh-cheat={roc_auc_score(V_LAB, ph[:, 4:].sum(1)):.4f} "
          f"behaviour acc={(ph.argmax(1) == V_BEH).mean():.4f} {time.time() - t0:.0f}s", flush=True)

# %% [markdown]
# ## Decisions, CV report (window-weighted accuracy = accuracy on 1-s test-like windows)

# %%
def assign(beh_p):
    cost = -np.log(np.clip(beh_p, 1e-6, 1))
    cost = np.concatenate([cost, cost + 3.0], 1)          # a behaviour may be reused only at a high cost
    r, c = linear_sum_assignment(cost)
    out = np.zeros(len(beh_p), int); out[r] = c % 8
    return out


def decide(pbin, pbeh, grp):
    p = 0.5 * pbin + 0.5 * pbeh[:, 4:].sum(1)
    rank, beh = np.zeros(len(p), int), np.zeros(len(p), int)
    for g in np.unique(grp):
        idx = np.where(grp == g)[0]
        rank[idx[np.argsort(-p[idx])[:int(round(len(idx) / 2))]]] = 1
        beh[idx] = assign(pbeh[idx])
    return p, {"thr": (p > 0.5).astype(int), "rank": rank, "assign": (beh >= 4).astype(int), "beh": beh}


w_vid = np.array([len(VID_EVAL[v]) for v in range(len(vdf))])
g_tr = pd.factorize(vdf.subject.astype(str) + "_" + vdf.angle)[0]
combos = {c: [c] for c in CONFIGS}
combos.update({"vjepa+pose": ["vjepa", "pose"], "vjepa+pose+grid": ["vjepa", "pose", "vjepa_grid"],
               "vjepa+vmae+pose": ["vjepa", "vmae", "pose"], "all5": list(CONFIGS)})
report = {}
for name, cs in combos.items():
    pb = np.mean([res_va[c][0] for c in cs], 0); ph = np.mean([res_va[c][1] for c in cs], 0)
    p, d = decide(pb, ph, g_tr)
    report[name] = {k: round(float(np.average(d[k] == V_LAB, weights=w_vid)), 4) for k in ["thr", "rank", "assign"]}
    report[name].update(auc=round(float(roc_auc_score(V_LAB, p)), 4), behaviour_acc=round(float((d["beh"] == V_BEH).mean()), 4))
    print(f"{name:16s}", report[name])

best_thr = max(report, key=lambda k: report[k]["thr"])
best_grp = max(((k, m) for k in report for m in ["rank", "assign"]), key=lambda km: report[km[0]][km[1]])
print("best video-only:", best_thr, report[best_thr]["thr"], "| best group:", best_grp, report[best_grp[0]][best_grp[1]])

pb = np.mean([res_va[c][0] for c in combos[best_grp[0]]], 0); ph = np.mean([res_va[c][1] for c in combos[best_grp[0]]], 0)
_, d = decide(pb, ph, g_tr)
cm = pd.crosstab(vdf.behavior, pd.Series(np.array(["Writing_Answers", "Thinking", "Looking_Around", "Checking_Smartwatch",
                 "Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"])[d["beh"]], name="assigned"))
print(cm)

# %% [markdown]
# ## Submissions

# %%
def to_windows(video_labels):
    lab = np.zeros(len(test_ids), int)
    for ch, y in zip(chains, video_labels):
        lab[ch] = y
    return lab


sub = pd.read_csv(DATA / "sample_submission.csv")
shutil.copy(V3 / "submission.csv", WORK / "submission_pure.csv")

pb = np.mean([res_te[c][0] for c in combos[best_thr]], 0); ph = np.mean([res_te[c][1] for c in combos[best_thr]], 0)
_, d = decide(pb, ph, groups)
sub.assign(label=to_windows(d["thr"])).to_csv(WORK / "submission_video.csv", index=False)

pb = np.mean([res_te[c][0] for c in combos[best_grp[0]]], 0); ph = np.mean([res_te[c][1] for c in combos[best_grp[0]]], 0)
p_te, d = decide(pb, ph, groups)
sub.assign(label=to_windows(d[best_grp[1]])).to_csv(WORK / "submission_group.csv", index=False)
pd.DataFrame({"video": range(len(chains)), "group": groups, "n_windows": [len(c) for c in chains], "prob": p_te,
              "behaviour": d["beh"], "thr": d["thr"], "rank": d["rank"], "assign": d["assign"]}).to_csv(WORK / "test_videos.csv", index=False)

for f in ["submission_pure.csv", "submission_video.csv", "submission_group.csv"]:
    s = pd.read_csv(WORK / f)
    print(f, s.label.value_counts().to_dict())
json.dump({"cv": report, "best_video_only": [best_thr, report[best_thr]["thr"]],
           "best_group": [*best_grp, report[best_grp[0]][best_grp[1]]]}, open(WORK / "cv_report.json", "w"), indent=1)
