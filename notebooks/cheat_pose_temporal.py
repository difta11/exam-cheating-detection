# %% [markdown]
# # Exam Cheating Detection v2 — pose keypoints + temporal models
#
# v1 (frozen DINOv2 features) reached only ~0.59 window AUC under subject-wise CV: it memorised
# subject appearance. Cheating behaviours are body-motion patterns (head down under the desk, head turns,
# wrist-to-wrist smartwatch use), so v2 uses appearance-free inputs:
# 1. YOLO11-pose on every frame (640x360) -> 17 COCO keypoints + person box of the main subject.
# 2. Per 1-s window (30 frames): absolute + body-normalised keypoints and velocities.
# 3. Models: bidirectional GRU (binary + 8-behaviour aux head) and LightGBM on window statistics,
#    5-fold GroupKFold by subject. Optional blend with v1 probabilities, weights chosen on OOF.
# 4. `submission.csv` (per window) and `submission_smooth.csv` (re-linked test windows, +-7).

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

# %%
import os, json, math, time, random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./work")
WORK.mkdir(parents=True, exist_ok=True)


def find_data_root():
    for base in [Path("/kaggle/input"), Path("../dataset"), Path("./dataset")]:
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            if "sample_submission.csv" in filenames and "train" in dirnames:
                return Path(dirpath)
            dirnames[:] = [d for d in dirnames if not d.startswith("test_seq_")]
    raise FileNotFoundError("dataset root with sample_submission.csv + train/ not found")


DATA = find_data_root()
PW, PH = 640, 360
T = 30
BEHAVIORS = ["Writing_Answers", "Thinking", "Looking_Around", "Checking_Smartwatch",
             "Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]
print("DATA:", DATA, "| device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

vids = []
for p in sorted((DATA / "train").glob("*/*/*")):
    if p.suffix.lower() not in (".mov", ".mp4", ".avi", ".mkv"):
        continue
    angle, beh = p.stem.split("_", 1)
    vids.append(dict(path=str(p), label=int(p.parent.parent.name == "cheat"),
                     subject=int(p.parent.name.split("_")[-1]), angle=angle,
                     behavior=beh, beh_id=BEHAVIORS.index(beh)))
vdf = pd.DataFrame(vids)
print(vdf.shape, vdf.label.value_counts().to_dict(), "subjects:", vdf.subject.nunique())

# %% [markdown]
# ## 1. Pose extraction (every frame)

# %%
def decode_video(path):
    cv2.setNumThreads(1)
    cap, fr = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.resize(f, (PW, PH), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(fr) < 50:   # fallback if this OpenCV build cannot decode HEVC
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={PW}:{PH}:flags=area",
               "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(-1, PH, PW, 3)
    return np.stack(fr)


def bounded_map(ex, fn, items, depth=6):
    """ordered map that keeps at most `depth` results in flight (bounds RAM)"""
    it, futs = iter(items), []
    for x in it:
        futs.append(ex.submit(fn, x))
        if len(futs) >= depth:
            break
    while futs:
        f = futs.pop(0)
        nxt = next(it, None)
        if nxt is not None:
            futs.append(ex.submit(fn, nxt))
        yield f.result()


pose_model = YOLO("yolo11l-pose.pt")
NP = 56   # 17*(x,y,conf) + box(4) + box conf


def run_pose(frames, bs=64):
    """BGR uint8 (N,H,W,3) -> (N,56): main person's keypoints (normalised to [0,1]) + box"""
    out = np.zeros((len(frames), NP), np.float32)
    for i in range(0, len(frames), bs):
        res = pose_model.predict(list(frames[i:i + bs]), imgsz=640, half=True, conf=0.2, verbose=False)
        for j, r in enumerate(res):
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                continue
            b, c = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
            k = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) * c))   # largest confident person
            kp = r.keypoints.data[k].cpu().numpy().copy()
            kp[:, 0] /= PW; kp[:, 1] /= PH
            out[i + j, :51] = kp.ravel()
            out[i + j, 51:55] = b[k] / [PW, PH, PW, PH]
            out[i + j, 55] = c[k]
    return out


t0 = time.time()
train_pose, offsets = [], [0]
with ProcessPoolExecutor(4) as ex:
    for i, frames in enumerate(bounded_map(ex, decode_video, vdf.path.tolist())):
        train_pose.append(run_pose(frames))
        offsets.append(offsets[-1] + len(frames))
        if i % 50 == 0:
            print(f"video {i}/{len(vdf)} frames={len(frames)} {time.time() - t0:.0f}s", flush=True)
train_pose = np.concatenate(train_pose)
vdf["start"], vdf["n"] = offsets[:-1], np.diff(offsets)
print("train pose", train_pose.shape, "detected:", (train_pose[:, 55] > 0).mean().round(4), f"{time.time() - t0:.0f}s")

sub = pd.read_csv(DATA / "sample_submission.csv")
test_ids = sub["id"].tolist()


def read_seq(sid):
    files = sorted((DATA / "test" / sid).glob("*.jpg"))[:T]
    fr = [cv2.resize(cv2.imread(str(f)), (PW, PH), interpolation=cv2.INTER_AREA) for f in files]
    while len(fr) < T:
        fr.append(fr[-1])
    return np.stack(fr)


t0 = time.time()
test_pose = []
with ThreadPoolExecutor(8) as ex:
    for i, frames in enumerate(bounded_map(ex, read_seq, test_ids, 32)):
        test_pose.append(run_pose(frames))
        if i % 300 == 0:
            print(f"test {i}/{len(test_ids)} {time.time() - t0:.0f}s", flush=True)
test_pose = np.stack(test_pose)        # (N, 30, 56)
print("test pose", test_pose.shape, "detected:", (test_pose[..., 55] > 0).mean().round(4))
np.save(WORK / "train_pose.npy", train_pose)
np.save(WORK / "test_pose.npy", test_pose)
vdf.to_csv(WORK / "train_videos.csv", index=False)
del pose_model; torch.cuda.empty_cache()

# %% [markdown]
# ## 2. Window features

# %%
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def hflip(W):
    """mirror raw pose windows (B,T,56): x -> 1-x, swap left/right keypoints"""
    valid = (W[..., 55:56] > 0).float()
    kp = W[..., :51].reshape(*W.shape[:2], 17, 3)[:, :, FLIP].clone()
    kp[..., 0] = (1 - kp[..., 0]) * valid
    box = W[..., 51:55].clone()
    box[..., 0], box[..., 2] = (1 - W[..., 53]) * valid[..., 0], (1 - W[..., 51]) * valid[..., 0]
    return torch.cat([kp.flatten(2), box, W[..., 55:]], -1)


def featurize(W):
    """(B,T,56) -> (B,T,124): abs xy, conf, body-normalised xy, velocity, box"""
    B, Tn, _ = W.shape
    kp = W[..., :51].reshape(B, Tn, 17, 3)
    xy, cf = kp[..., :2], kp[..., 2]
    box, bc = W[..., 51:55], W[..., 55:]
    valid = (bc > 0).float()
    nv = valid.sum(1).clamp(min=1)                                    # (B,1)
    sh = (xy[:, :, 5] + xy[:, :, 6]) / 2
    ref = (sh * valid).sum(1) / nv                                    # (B,2)
    bh = ((box[..., 3] - box[..., 1]) * valid[..., 0]).sum(1) / nv[:, 0]
    scale = bh.clamp(min=0.05)[:, None, None, None]
    rel = (xy - ref[:, None, None]) / scale * valid[..., None]
    vel = torch.diff(rel, dim=1, prepend=rel[:, :1])
    return torch.cat([xy.flatten(2), cf, rel.flatten(2), vel.flatten(2) * 10, box, bc], -1)


def window_stats(f):
    """(B,T,F) -> (B,4F+F) mean/std/min/max + mean |diff|"""
    return torch.cat([f.mean(1), f.std(1), f.amin(1), f.amax(1),
                      (f[:, 1:] - f[:, :-1]).abs().mean(1)], 1)


POSE = torch.from_numpy(train_pose).to(DEV)
TIDX = torch.arange(T, device=DEV)
V_START, V_N = vdf.start.values, vdf.n.values
V_LAB, V_BEH = vdf.label.values, vdf.beh_id.values


def gather(starts):
    return POSE[torch.as_tensor(starts, device=DEV)[:, None] + TIDX]


def windows(vid_idx, stride):
    s, v_of = [], []
    for v in vid_idx:
        for o in range(0, V_N[v] - T + 1, stride):
            s.append(V_START[v] + o); v_of.append(v)
    return np.array(s), np.array(v_of)


TEST_W = torch.from_numpy(test_pose).to(DEV)
FDIM = featurize(TEST_W[:2]).shape[-1]
print("frame feature dim", FDIM)

# %% [markdown]
# ## 3. Models: BiGRU and LightGBM, GroupKFold by subject

# %%
class PoseGRU(nn.Module):
    def __init__(self, fin=FDIM, d=128, p=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(fin, d), nn.GELU(), nn.Dropout(p))
        self.gru = nn.GRU(d, d, 2, batch_first=True, bidirectional=True, dropout=p)
        self.drop = nn.Dropout(p)
        self.out_bin, self.out_beh = nn.Linear(4 * d, 1), nn.Linear(4 * d, len(BEHAVIORS))

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        z = self.drop(torch.cat([h.mean(1), h.amax(1)], 1))
        return self.out_bin(z).squeeze(1), self.out_beh(z)


def prob_from(lb, lh):
    return 0.5 * torch.sigmoid(lb) + 0.5 * lh.softmax(1)[:, 4:].sum(1)


EPOCHS, PER_VIDEO, BATCH, LR = 30, 32, 256, 1e-3


def train_gru(tr_idx, seed, mu, sd):
    seed_all(seed)
    model = PoseGRU().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
    steps = EPOCHS * math.ceil(len(tr_idx) * PER_VIDEO / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    for ep in range(EPOCHS):
        model.train()
        s, y, yb = [], [], []
        for v in tr_idx:
            off = np.random.randint(0, V_N[v] - T + 1, PER_VIDEO)
            s.append(V_START[v] + off); y += [V_LAB[v]] * PER_VIDEO; yb += [V_BEH[v]] * PER_VIDEO
        s = np.concatenate(s); perm = np.random.permutation(len(s))
        s, y, yb = s[perm], np.array(y)[perm], np.array(yb)[perm]
        for i in range(0, len(s), BATCH):
            W = gather(s[i:i + BATCH])
            fl = torch.rand(len(W), device=DEV) < 0.5
            W = torch.where(fl[:, None, None], hflip(W), W)
            # keypoint dropout + coordinate jitter
            m = (torch.rand(*W.shape[:2], 17, 1, device=DEV) > 0.1).float()
            kp = W[..., :51].reshape(*W.shape[:2], 17, 3)
            kp = torch.cat([kp[..., :2] + 0.005 * torch.randn_like(kp[..., :2]), kp[..., 2:]], -1) * m
            W = torch.cat([kp.flatten(2), W[..., 51:]], -1)
            x = (featurize(W) - mu) / sd
            lb, lh = model(x)
            yt = torch.as_tensor(y[i:i + BATCH], device=DEV).float()
            loss = F.binary_cross_entropy_with_logits(lb, yt * 0.9 + 0.05) \
                + 0.5 * F.cross_entropy(lh, torch.as_tensor(yb[i:i + BATCH], device=DEV), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    return model


@torch.no_grad()
def predict_gru(model, Wall, mu, sd, bs=1024):
    model.eval(); out = []
    for i in range(0, len(Wall), bs):
        W = Wall[i:i + bs]
        p = 0
        for Wv in (W, hflip(W)):     # flip TTA
            p = p + prob_from(*model((featurize(Wv) - mu) / sd)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()


LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=60,
                  feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1)


@torch.no_grad()
def stats_np(Wall, bs=4096):
    return torch.cat([window_stats(featurize(Wall[i:i + bs])).cpu() for i in range(0, len(Wall), bs)]).numpy()


SEEDS = [42, 7, 2026]
X_test = stats_np(TEST_W)
oof_rows = []
test_gru, test_lgb = np.zeros(len(test_ids)), np.zeros(len(test_ids))
t0 = time.time()
for fold, (tr, va) in enumerate(GroupKFold(n_splits=5).split(vdf, groups=vdf.subject)):
    s_tr, v_tr = windows(tr, 10)
    s_va, v_va = windows(va, T)
    F_tr = featurize(gather(s_tr))
    mu, sd = F_tr.mean((0, 1)), F_tr.std((0, 1)) + 1e-4
    del F_tr
    W_va = gather(s_va)
    p_gru = np.zeros(len(s_va))
    for seed in SEEDS:
        m = train_gru(tr, seed + fold, mu, sd)
        p_gru += predict_gru(m, W_va, mu, sd) / len(SEEDS)
        test_gru += predict_gru(m, TEST_W, mu, sd) / (5 * len(SEEDS))
        torch.save(m.state_dict(), WORK / f"gru_f{fold}_s{seed}.pt")
    # LightGBM on window statistics (flip-augmented train set)
    W_tr = gather(s_tr)
    X_tr = np.concatenate([stats_np(W_tr), stats_np(hflip(W_tr))])
    y_tr = np.tile(V_LAB[v_tr], 2)
    X_va = stats_np(W_va)
    p_lgb = np.zeros(len(s_va))
    for seed in SEEDS:
        bst = lgb.train({**LGB_PARAMS, "seed": seed}, lgb.Dataset(X_tr, y_tr), num_boost_round=500)
        p_lgb += bst.predict(X_va) / len(SEEDS)
        test_lgb += bst.predict(X_test) / (5 * len(SEEDS))
    y_va = V_LAB[v_va]
    print(f"fold {fold} gru acc={accuracy_score(y_va, p_gru > 0.5):.4f} auc={roc_auc_score(y_va, p_gru):.4f} | "
          f"lgb acc={accuracy_score(y_va, p_lgb > 0.5):.4f} auc={roc_auc_score(y_va, p_lgb):.4f} "
          f"{time.time() - t0:.0f}s", flush=True)
    for s_, v_, a_, b_ in zip(s_va, v_va, p_gru, p_lgb):
        oof_rows.append(dict(video=v_, win=(s_ - V_START[v_]) // T, fold=fold, gru=a_, lgb=b_,
                             label=V_LAB[v_], behavior=vdf.behavior.values[v_], angle=vdf.angle.values[v_]))
oof = pd.DataFrame(oof_rows)
test_df = pd.DataFrame({"id": test_ids, "gru": test_gru, "lgb": test_lgb})

# %% [markdown]
# ## 4. Optional blend with v1 (DINOv2) and CV report

# %%
def find_v1():
    for dirpath, _, files in os.walk("/kaggle/input"):
        if {"oof.csv", "test_probs.csv", "cv_report.json"} <= set(files):
            return Path(dirpath)
    return None


V1 = find_v1()
MODELS = ["gru", "lgb"]
if V1 is not None:
    o1 = pd.read_csv(V1 / "oof.csv")
    o1["win"] = (o1.start - o1.groupby("video").start.transform("min")) // 15
    oof = oof.merge(o1[["video", "win", "prob"]].rename(columns={"prob": "dino"}), on=["video", "win"], how="left")
    oof["dino"] = oof.dino.fillna(0.5)
    test_df = test_df.merge(pd.read_csv(V1 / "test_probs.csv").rename(columns={"prob": "dino"}), on="id")
    MODELS.append("dino")
print("v1 found:", V1)


def report(y, p):
    return dict(acc=round(accuracy_score(y, p > 0.5), 4), f1=round(f1_score(y, p > 0.5), 4),
                auc=round(roc_auc_score(y, p), 4))


# blend weights: grid on OOF AUC (step 0.1)
best = (-1, None)
grid = [w for w in np.array(np.meshgrid(*[np.arange(0, 1.01, 0.1)] * len(MODELS))).reshape(len(MODELS), -1).T
        if abs(w.sum() - 1) < 1e-6]
for w in grid:
    a = roc_auc_score(oof.label, oof[MODELS].values @ w)
    if a > best[0]:
        best = (a, w)
W_BLEND = best[1]
oof["blend"] = oof[MODELS].values @ W_BLEND
test_df["blend"] = test_df[MODELS].values @ W_BLEND
oof = oof.sort_values(["video", "win"]).reset_index(drop=True)
oof["blend_smooth"] = oof.groupby("video")["blend"].transform(lambda s: s.rolling(15, center=True, min_periods=1).mean())
cv = {m: report(oof.label, oof[m]) for m in MODELS + ["blend", "blend_smooth"]}
cv["blend_weights"] = dict(zip(MODELS, map(float, W_BLEND)))
for a in ["front", "side"]:
    s = oof[oof.angle == a]
    cv[f"blend_{a}"] = report(s.label, s.blend)
print(json.dumps(cv, indent=1))
print("\nper-behaviour window accuracy (blend):")
print(oof.assign(ok=((oof.blend > 0.5) == oof.label)).groupby(["behavior", "angle"]).ok.mean().unstack().round(3))
oof.to_csv(WORK / "oof.csv", index=False)

# %% [markdown]
# ## 5. Submissions

# %%
sub["label"] = (test_df.blend.values > 0.5).astype(int)
sub.to_csv(WORK / "submission.csv", index=False)
test_df.rename(columns={"blend": "prob"}).to_csv(WORK / "test_probs.csv", index=False)
print(sub.label.value_counts())

# re-link shuffled test windows by frame continuity (frame 30 -> frame 01), smooth over +-7 links
TW, TH = 96, 54
def thumb(p):
    im = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8)
    return cv2.resize(im, (TW, TH), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def seq_thumbs(sid):
    d = DATA / "test" / sid
    return [thumb(d / f) for f in ("01.jpg", "02.jpg", "29.jpg", "30.jpg")]


with ThreadPoolExecutor(8) as ex:
    th = list(ex.map(seq_thumbs, test_ids))
f01, f02, f29, f30 = (np.stack([t[i] for t in th]) for i in range(4))
n = len(test_ids)
D = np.stack([np.abs(f30[k][None] - f01).mean(1) for k in range(n)])
np.fill_diagonal(D, 1e6)
scale = np.maximum(0.5 * (np.abs(f30 - f29).mean(1)[:, None] + np.abs(f02 - f01).mean(1)[None]), 0.05)
R = np.minimum(D / scale, 1e3)
NOLINK = 4.0
big = np.zeros((2 * n, 2 * n))
big[:n, :n] = R
big[:n, n:] = np.where(np.eye(n, dtype=bool), NOLINK, 1e3)
big[n:, :n] = np.where(np.eye(n, dtype=bool), NOLINK, 1e3)
r, c = linear_sum_assignment(big)
links = [(int(i), int(j)) for i, j in zip(r, c) if i < n and j < n and R[i, j] < 2.5]
nbr = [[] for _ in range(n)]
for i, j in links:
    nbr[i].append(j); nbr[j].append(i)
print("links kept:", len(links), "/", n)

tp = test_df.blend.values
smooth = np.zeros(n)
for k in range(n):
    seen, frontier = {k}, [k]
    for _ in range(7):
        frontier = [m for f in frontier for m in nbr[f] if m not in seen]
        seen.update(frontier)
    smooth[k] = tp[list(seen)].mean()
sub_s = sub.copy()
sub_s["label"] = (smooth > 0.5).astype(int)
sub_s.to_csv(WORK / "submission_smooth.csv", index=False)
print("smoothed label changes vs raw:", int((sub_s.label != sub.label).sum()))
json.dump({"cv": cv, "n_links": len(links), "changes_smooth_vs_raw": int((sub_s.label != sub.label).sum())},
          open(WORK / "cv_report.json", "w"), indent=1)
