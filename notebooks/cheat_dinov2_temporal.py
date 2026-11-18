# %% [markdown]
# # Exam Cheating Detection — DINOv2 frame features + temporal transformer
#
# Label: 0 = normal, 1 = cheat. Test unit = 30 consecutive frames (1 s @ 30 FPS) from one camera.
#
# Pipeline
# 1. Decode every train video (HEVC; front camera is 10-bit HLG HDR, side is SDR) with OpenCV —
#    plain cv2 decoding matches the colour of the provided test JPGs (checked locally).
# 2. Keep every 2nd frame (15 per 1-second window), resize to 392x224, run a frozen
#    DINOv2 ViT-B/14 (registers) and keep 10 tokens per frame: CLS, mean patch, 2x4 pooled patch grid.
# 3. Train a small temporal transformer on 15x10 token windows, binary head + 8-behaviour auxiliary head,
#    5-fold GroupKFold by subject (test subjects are unseen), 3 seeds.
# 4. Predict test windows (frames 01,03,...,29); write `submission.csv`.
# 5. Optional: test sequences are shuffled consecutive windows of the same source videos. Re-link them by
#    frame continuity and smooth probabilities over linked neighbours -> `submission_smooth.csv`.

# %%
import os, json, math, time, random, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.optimize import linear_sum_assignment

SEED = 42
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)

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
IMG_W, IMG_H = 392, 224          # multiples of 14 -> 28x16 patch grid, keeps 16:9
STEP = 2                          # every 2nd frame
T = 15                            # features per 1-second window
BEHAVIORS = ["Writing_Answers", "Thinking", "Looking_Around", "Checking_Smartwatch",
             "Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]
print("DATA:", DATA, "| device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

# %%
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
print(vdf.groupby(["behavior", "label"]).size())

# %% [markdown]
# ## 1. Frame decoding + frozen DINOv2 features

# %%
def decode_cv2(path):
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while True:
        if not cap.grab():
            break
        if i % STEP == 0:
            ok, fr = cap.retrieve()
            if ok:
                fr = cv2.resize(fr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def decode_ffmpeg(path):  # fallback if this OpenCV build cannot decode HEVC
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf",
           f"select='not(mod(n\\,{STEP}))',scale={IMG_W}:{IMG_H}:flags=area",
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, IMG_H, IMG_W, 3))


def bounded_map(ex, fn, items, depth=8):
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


def decode_video(path):
    cv2.setNumThreads(1)
    frames = decode_cv2(path)
    if len(frames) < 50:
        frames = decode_ffmpeg(path)
    return np.stack(frames)


backbone = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True,
                             num_classes=0, img_size=(IMG_H, IMG_W)).to(DEV).eval().half()
GH, GW = IMG_H // 14, IMG_W // 14
NPRE = backbone.num_prefix_tokens
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
NTOK, DIM = 10, backbone.num_features
BS = 96


@torch.no_grad()
def extract(frames):
    """frames: uint8 (N,H,W,3) RGB -> fp16 (N,10,768)"""
    out = []
    for i in range(0, len(frames), BS):
        x = torch.from_numpy(np.ascontiguousarray(frames[i:i + BS])).to(DEV, non_blocking=True)
        x = ((x.permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD).half()
        h = backbone.forward_features(x).float()
        cls = h[:, 0:1]
        p = h[:, NPRE:].reshape(len(x), GH, GW, -1).permute(0, 3, 1, 2)
        grid = F.adaptive_avg_pool2d(p, (2, 4)).flatten(2).transpose(1, 2)
        mean = p.mean((2, 3))[:, None]
        out.append(torch.cat([cls, mean, grid], 1).half().cpu())
    return torch.cat(out).numpy()


t0 = time.time()
train_feats, offsets = [], [0]
with ProcessPoolExecutor(4) as ex:
    for i, frames in enumerate(bounded_map(ex, decode_video, vdf.path.tolist(), 8)):
        train_feats.append(extract(frames))
        offsets.append(offsets[-1] + len(frames))
        if i % 50 == 0:
            print(f"video {i}/{len(vdf)} frames={len(frames)} {time.time() - t0:.0f}s", flush=True)
train_feats = np.concatenate(train_feats)
vdf["start"], vdf["n"] = offsets[:-1], np.diff(offsets)
print("train feats", train_feats.shape, f"{time.time() - t0:.0f}s")

# %%
sub = pd.read_csv(DATA / "sample_submission.csv")
test_ids = sub["id"].tolist()


def read_seq(sid):
    files = sorted((DATA / "test" / sid).glob("*.jpg"))[::STEP][:T]
    fr = [cv2.cvtColor(cv2.resize(cv2.imread(str(f)), (IMG_W, IMG_H), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB) for f in files]
    while len(fr) < T:
        fr.append(fr[-1])
    return np.stack(fr)


t0 = time.time()
test_feats = []
with ThreadPoolExecutor(8) as ex:
    for i, frames in enumerate(bounded_map(ex, read_seq, test_ids, 32)):
        test_feats.append(extract(frames))
        if i % 300 == 0:
            print(f"test {i}/{len(test_ids)} {time.time() - t0:.0f}s", flush=True)
test_feats = np.stack(test_feats)          # (N, T, 10, 768)
print("test feats", test_feats.shape)

np.save(WORK / "train_feats_fp16.npy", train_feats)
np.save(WORK / "test_feats_fp16.npy", test_feats)
vdf.to_csv(WORK / "train_videos.csv", index=False)
del backbone; torch.cuda.empty_cache()

# %% [markdown]
# ## 2. Temporal transformer head, GroupKFold by subject

# %%
class TemporalHead(nn.Module):
    def __init__(self, d_in=DIM, d=256, ntok=NTOK, t=T, layers=2, heads=4, p=0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(0.1), nn.Linear(d_in, d))
        self.tpos = nn.Parameter(torch.randn(1, t, 1, d) * 0.02)
        self.spos = nn.Parameter(torch.randn(1, 1, ntok, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, p, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm, self.drop = nn.LayerNorm(d), nn.Dropout(p)
        self.out_bin, self.out_beh = nn.Linear(d, 1), nn.Linear(d, len(BEHAVIORS))

    def forward(self, x):                       # (B, T, K, D)
        b, t, k, _ = x.shape
        h = (self.proj(x) + self.tpos[:, :t] + self.spos).reshape(b, t * k, -1)
        h = self.enc(torch.cat([self.cls.expand(b, -1, -1), h], 1))
        z = self.drop(self.norm(h[:, 0]))
        return self.out_bin(z).squeeze(1), self.out_beh(z)


def prob_from(logit_bin, logit_beh):
    return 0.5 * torch.sigmoid(logit_bin) + 0.5 * logit_beh.softmax(1)[:, 4:].sum(1)


FEATS = torch.from_numpy(train_feats).to(DEV)           # fp16 on GPU (~1.5 GB)
TIDX = torch.arange(T, device=DEV)
V_START, V_N = vdf.start.values, vdf.n.values
V_LAB, V_BEH = vdf.label.values, vdf.beh_id.values


def gather(starts):
    return FEATS[torch.as_tensor(starts, device=DEV)[:, None] + TIDX].float()


def sample_windows(vid_idx, per_video):
    s, y, yb = [], [], []
    for v in vid_idx:
        off = np.random.randint(0, V_N[v] - T + 1, per_video)
        s.append(V_START[v] + off); y += [V_LAB[v]] * per_video; yb += [V_BEH[v]] * per_video
    s = np.concatenate(s); perm = np.random.permutation(len(s))
    return s[perm], np.array(y)[perm], np.array(yb)[perm]


def eval_windows(vid_idx):
    """non-overlapping consecutive 1-s windows, like the test set"""
    s, v_of = [], []
    for v in vid_idx:
        for o in range(0, V_N[v] - T + 1, T):
            s.append(V_START[v] + o); v_of.append(v)
    return np.array(s), np.array(v_of)


@torch.no_grad()
def predict(model, x_fn, n, bs=512):
    model.eval(); out = []
    for i in range(0, n, bs):
        with torch.autocast("cuda", dtype=torch.float16):
            lb, lh = model(x_fn(i, min(i + bs, n)))
        out.append(prob_from(lb.float(), lh.float()).cpu())
    return torch.cat(out).numpy()


EPOCHS, PER_VIDEO, BATCH, LR = 20, 32, 128, 3e-4


def train_one(tr_idx, seed):
    seed_all(seed)
    model = TemporalHead().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    steps = EPOCHS * math.ceil(len(tr_idx) * PER_VIDEO / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(EPOCHS):
        model.train()
        s, y, yb = sample_windows(tr_idx, PER_VIDEO)
        for i in range(0, len(s), BATCH):
            x = gather(s[i:i + BATCH])
            # augmentation: token dropout + feature noise + brightness-like global scale
            x = x * (torch.rand(x.shape[0], 1, x.shape[2], 1, device=DEV) > 0.1)
            x = x * (1 + 0.05 * torch.randn(x.shape[0], 1, 1, 1, device=DEV)) + 0.05 * torch.randn_like(x)
            if random.random() < 0.5:   # random temporal jitter: repeat/skip one frame
                j = random.randint(1, T - 1); x = torch.cat([x[:, :j], x[:, j - 1:j], x[:, j:-1]], 1)
            yt = torch.as_tensor(y[i:i + BATCH], device=DEV).float()
            ybt = torch.as_tensor(yb[i:i + BATCH], device=DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                lb, lh = model(x)
                loss = F.binary_cross_entropy_with_logits(lb.float(), yt * 0.95 + 0.025) \
                    + 0.5 * F.cross_entropy(lh.float(), ybt, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
    return model


SEEDS = [42, 7, 2026]
gkf = GroupKFold(n_splits=5)
oof_rows = []
test_prob = np.zeros(len(test_ids))
TEST_T = torch.from_numpy(test_feats).to(DEV)
n_models = 0
t0 = time.time()
for fold, (tr, va) in enumerate(gkf.split(vdf, groups=vdf.subject)):
    s_va, v_va = eval_windows(va)
    p_va = np.zeros(len(s_va))
    for seed in SEEDS:
        m = train_one(tr, seed + fold)
        p_va += predict(m, lambda a, b: gather(s_va[a:b]), len(s_va)) / len(SEEDS)
        test_prob += predict(m, lambda a, b: TEST_T[a:b].float(), len(test_ids))
        n_models += 1
        torch.save(m.state_dict(), WORK / f"head_f{fold}_s{seed}.pt")
    y_va = V_LAB[v_va]
    print(f"fold {fold} subjects={sorted(vdf.subject.values[va].tolist())[:3]}.. "
          f"acc={accuracy_score(y_va, p_va > 0.5):.4f} auc={roc_auc_score(y_va, p_va):.4f} "
          f"{time.time() - t0:.0f}s", flush=True)
    for s_, v_, p_ in zip(s_va, v_va, p_va):
        oof_rows.append(dict(video=v_, fold=fold, start=s_, prob=p_, label=V_LAB[v_],
                             behavior=vdf.behavior.values[v_], angle=vdf.angle.values[v_]))
test_prob /= n_models
oof = pd.DataFrame(oof_rows)

# %% [markdown]
# ## 3. Cross-validation report

# %%
def report(df, col):
    y, p = df.label.values, df[col].values
    return dict(acc=round(accuracy_score(y, p > 0.5), 4), f1=round(f1_score(y, p > 0.5), 4),
                auc=round(roc_auc_score(y, p), 4))


# simulate temporal smoothing with +-7 neighbouring windows of the same source video
oof = oof.sort_values(["video", "start"]).reset_index(drop=True)
oof["prob_smooth"] = oof.groupby("video")["prob"].transform(
    lambda s: s.rolling(15, center=True, min_periods=1).mean())
cv = {"window": report(oof, "prob"), "window_smoothed(+-7)": report(oof, "prob_smooth")}
print(json.dumps(cv, indent=1))
print("\nper-behaviour window accuracy (target label shown):")
print(oof.assign(ok=((oof.prob > 0.5) == oof.label)).groupby(["behavior", "angle"]).ok.mean().unstack().round(3))
oof.to_csv(WORK / "oof.csv", index=False)

# %% [markdown]
# ## 4. Submission (pure per-window model)

# %%
sub["label"] = (test_prob > 0.5).astype(int)
sub.to_csv(WORK / "submission.csv", index=False)
pd.DataFrame({"id": test_ids, "prob": test_prob}).to_csv(WORK / "test_probs.csv", index=False)
print(sub.label.value_counts())

# %% [markdown]
# ## 5. Optional: re-link shuffled test windows by frame continuity, smooth over neighbours
# Frame 30 of a window is followed by frame 01 of exactly one other window of the same video.
# Links are solved with an assignment problem; each window's probability is averaged with windows
# up to 7 links away (≈ the same 15-s video), mirroring `window_smoothed(+-7)` in the CV report.

# %%
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

smooth = np.zeros(n)
for k in range(n):
    seen, frontier = {k}, [k]
    for _ in range(7):
        frontier = [m for f in frontier for m in nbr[f] if m not in seen]
        seen.update(frontier)
    smooth[k] = test_prob[list(seen)].mean()
sub_s = sub.copy()
sub_s["label"] = (smooth > 0.5).astype(int)
sub_s.to_csv(WORK / "submission_smooth.csv", index=False)
print("smoothed label changes vs raw:", int((sub_s.label != sub.label).sum()))
json.dump({"cv": cv, "n_links": len(links), "n_models": n_models,
           "changes_smooth_vs_raw": int((sub_s.label != sub.label).sum())},
          open(WORK / "cv_report.json", "w"), indent=1)
