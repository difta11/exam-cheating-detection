# %% [markdown]
# # Exam Cheating Detection v3 — motion video features on person crops + multiple-instance head
#
# Findings so far: in every training video the subject spends most of the time writing; the behaviour that
# defines the label (tapping a smartwatch, peeking, reading a hidden note) appears only in short moments.
# Window labels are therefore noisy, and v1 (DINOv2 frames) / v2 (pose) reach only ~0.60 window accuracy.
#
# v3:
# 1. Square person-centred crops (box from v2 YOLO11-pose) of 16 frames per 1-s window, 256 px.
# 2. Frozen motion-centric video encoders fine-tuned on Something-Something-v2 (hand/object motions):
#    VideoMAE-B (224 px) and V-JEPA 2 ViT-L (256 px). Keep pooled embedding + SSv2 logits per window.
# 3. Multiple-instance (MIL) head: bags of 1..15 consecutive windows of a video, attention + max pooling,
#    binary + 8-behaviour heads. GroupKFold by subject. Bag of 1 = pure per-window prediction;
#    bag of re-linked neighbouring test windows = context prediction.
# 4. Blend with v2 (pose GRU/LightGBM + DINOv2) on OOF; write `submission.csv` and `submission_smooth.csv`.

# %%
import os, sys, json, math, time, random, subprocess
from pathlib import Path

SMOKE = os.environ.get("SMOKE") == "1"   # local CPU dry run on a tiny subset with random tiny backbones
if not SMOKE and subprocess.run([sys.executable, "-c", "from transformers import VJEPA2ForVideoClassification"],
                                capture_output=True).returncode != 0:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"], check=True)

# %%
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.optimize import linear_sum_assignment
import transformers
from transformers import VideoMAEForVideoClassification, VJEPA2ForVideoClassification


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(42)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./work")
WORK.mkdir(parents=True, exist_ok=True)


def find_dir(required, bases=("/kaggle/input", "./dataset", "./outputs/v2")):
    for base in map(Path, bases):
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            if all(r in filenames or r in dirnames for r in required):
                return Path(dirpath)
            dirnames[:] = [d for d in dirnames if not d.startswith("test_seq_") and not d.startswith("subject_")]
    raise FileNotFoundError(required)


DATA = find_dir(["sample_submission.csv", "train"])
V2 = find_dir(["train_pose.npy", "test_pose.npy", "train_videos.csv", "oof.csv", "test_probs.csv"])
T, NF, S = 30, 16, 256                       # window frames, frames fed to encoders, crop size
IDX16 = np.round(np.linspace(0, T - 1, NF)).astype(int)
TR_STRIDE = 15                               # training windows every 0.5 s
BEHAVIORS = ["Writing_Answers", "Thinking", "Looking_Around", "Checking_Smartwatch",
             "Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]
print("DATA:", DATA, "| V2:", V2, "| device:", DEV, "| transformers", transformers.__version__)

vdf = pd.read_csv(V2 / "train_videos.csv")
vdf["path"] = [str(DATA / "train" / Path(p).parts[-3] / Path(p).parts[-2] / Path(p).name) for p in vdf.path]
train_pose = np.load(V2 / "train_pose.npy")
test_pose = np.load(V2 / "test_pose.npy")
sub = pd.read_csv(DATA / "sample_submission.csv")
test_ids = sub["id"].tolist()
if SMOKE:   # 5 subjects, first 100 frames of each video, 24 test sequences
    vdf = vdf[vdf.subject <= 5].copy()
    vdf["n"] = np.minimum(vdf.n, 100)
    test_ids = test_ids[:24]; test_pose = test_pose[:24]; sub = sub.iloc[:24].copy()
vdf = vdf.reset_index(drop=True)
print(vdf.shape, vdf.label.value_counts().to_dict(), "subjects:", vdf.subject.nunique())

# %% [markdown]
# ## 1. Person-centred crops + frozen video encoders

# %%
def crop_box(win_pose, W, H):
    """square crop (x0, y0, side) around the main person over one 30-frame window"""
    valid = win_pose[:, 55] > 0
    if valid.sum() == 0:
        side, cx, cy = H, W / 2, H / 2
    else:
        b = win_pose[valid, 51:55] * [W, H, W, H]
        x1, y1 = np.percentile(b[:, 0], 5), np.percentile(b[:, 1], 5)
        x2, y2 = np.percentile(b[:, 2], 95), np.percentile(b[:, 3], 95)
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.15
    side = int(min(max(side, 160), H, W))
    x0 = int(np.clip(cx - side / 2, 0, W - side)); y0 = int(np.clip(cy - side / 2, 0, H - side))
    return x0, y0, side


def cut(frame, box):
    x0, y0, s = box
    return cv2.cvtColor(cv2.resize(frame[y0:y0 + s, x0:x0 + s], (S, S), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB)


def read_all_ffmpeg(path, W, H):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def crop_video(args):
    """decode one train video once, return (n_windows, 16, S, S, 3) RGB crops"""
    path, starts, boxes = args
    cv2.setNumThreads(1)
    need = {}
    for wi, st in enumerate(starts):
        for j, o in enumerate(IDX16):
            need.setdefault(int(st + o), []).append((wi, j))
    out = np.zeros((len(starts), NF, S, S, 3), np.uint8)
    cap, i, last, done = cv2.VideoCapture(path), 0, max(need), set()
    while i <= last:
        if i in need:
            ok, f = cap.read()
            if not ok:
                break
            for wi, j in need[i]:
                out[wi, j] = cut(f, boxes[wi])
            done.add(i)
        elif not cap.grab():
            break
        i += 1
    cap.release()
    if len(done) < len(need):           # this OpenCV build could not decode the video -> ffmpeg
        W, H = 1280, 720
        frames = read_all_ffmpeg(path, W, H)
        for fi, lst in need.items():
            for wi, j in lst:
                out[wi, j] = cut(frames[min(fi, len(frames) - 1)], boxes[wi])
    return out


def video_size(path):
    cap = cv2.VideoCapture(path)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (W, H) if W > 0 else (1280, 720)


def bounded_map(ex, fn, items, depth=6):
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


# train windows (stride 0.5 s)
win_rows = []
for v, r in vdf.iterrows():
    for o in range(0, r.n - T + 1, TR_STRIDE):
        win_rows.append(dict(video=v, off=o, gstart=r.start + o))
wdf = pd.DataFrame(win_rows)
print("train windows:", len(wdf))

# %%
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 1, 3, 1, 1)


def load_backbones():
    if SMOKE:
        from transformers import VideoMAEConfig, VJEPA2Config
        vm = VideoMAEForVideoClassification(VideoMAEConfig(
            image_size=224, num_frames=NF, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=64, use_mean_pooling=True, num_labels=7))
        vj = VJEPA2ForVideoClassification(VJEPA2Config(
            crop_size=S, frames_per_clip=NF, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            pred_hidden_size=16, pred_num_hidden_layers=1, pred_num_attention_heads=2, num_labels=5))
    else:
        vm = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
        vj = VJEPA2ForVideoClassification.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2")
    return {"vmae": vm.to(DEV).eval().half() if AMP else vm.eval(),
            "vjepa": vj.to(DEV).eval().half() if AMP else vj.eval()}


BB = load_backbones()


def grid_pool(seq, t):
    b, n, d = seq.shape
    hw = int(round(math.sqrt(n // t)))
    g = seq[:, :t * hw * hw].reshape(b, t, hw, hw, d).mean(1).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(g, (2, 2)).flatten(2).transpose(1, 2).flatten(1)   # (b, 4d)


@torch.no_grad()
def encode(clips):
    """uint8 (B,16,S,S,3) -> dict of fp16 arrays: [pooled, token-mean, 2x2 grid, ssv2 logits]"""
    x = torch.from_numpy(clips).to(DEV).permute(0, 1, 4, 2, 3).float() / 255.0
    x = (x - MEAN) / STD
    dt = torch.float16 if AMP else torch.float32
    x224 = F.interpolate(x.flatten(0, 1), size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
    x224 = x224.view(x.shape[0], NF, 3, 224, 224).to(dt)
    x = x.to(dt)
    out = {}
    m = BB["vmae"]
    seq = m.videomae(pixel_values=x224).last_hidden_state
    pooled = m.fc_norm(seq.mean(1)) if m.fc_norm is not None else seq.mean(1)
    out["vmae"] = torch.cat([pooled, seq.mean(1), grid_pool(seq, NF // 2), m.classifier(pooled)], 1)
    m = BB["vjepa"]
    seq = m.vjepa2(pixel_values_videos=x, skip_predictor=True).last_hidden_state
    pooled = m.pooler(seq)
    out["vjepa"] = torch.cat([pooled, seq.mean(1), grid_pool(seq, NF // 2), m.classifier(pooled)], 1)
    return {k: v.float().cpu().numpy().astype(np.float16) for k, v in out.items()}


def dims(name):
    m = BB[name]
    d = m.config.hidden_size
    return dict(pooled=slice(0, d), mean=slice(d, 2 * d), grid=slice(2 * d, 6 * d),
                logits=slice(6 * d, 6 * d + m.config.num_labels))


def encode_all(clip_iter, n_total, bs=16):
    feats = {k: [] for k in BB}
    buf, t0, seen = [], time.time(), 0

    def flush():
        nonlocal buf
        if buf:
            c = np.concatenate(buf)
            for i in range(0, len(c), bs):
                for k, v in encode(c[i:i + bs]).items():
                    feats[k].append(v)
            buf = []
    for clips in clip_iter:
        buf.append(clips); seen += len(clips)
        if sum(len(b) for b in buf) >= 4 * bs:
            flush()
        if seen % 1000 < len(clips):
            print(f"  {seen}/{n_total} clips {time.time() - t0:.0f}s", flush=True)
    flush()
    return {k: np.concatenate(v) for k, v in feats.items()}


t0 = time.time()
sizes = {}
jobs = []
for v, r in vdf.iterrows():
    W, H = sizes.setdefault(r.path, video_size(r.path))
    sw = wdf[wdf.video == v]
    boxes = [crop_box(train_pose[g:g + T], W, H) for g in sw.gstart]
    jobs.append((r.path, sw.off.values, boxes))


def crop_test(k):
    files = sorted((DATA / "test" / test_ids[k]).glob("*.jpg"))
    files = (files + files[-1:] * T)[:T]
    first = cv2.imread(str(files[0]))
    H, W = first.shape[:2]
    box = crop_box(test_pose[k], W, H)
    return np.stack([cut(cv2.imread(str(files[o])) if o else first, box) for o in IDX16])[None]


# visual check of the crops: 4 train videos (window 0) + 4 test sequences, frames 0/5/10/15 of 16
prev = [crop_video((j[0], j[1][:1], j[2][:1]))[0] for j in [jobs[i] for i in np.linspace(0, len(jobs) - 1, 4).astype(int)]]
prev += [crop_test(k)[0] for k in range(4)]
cv2.imwrite(str(WORK / "crop_preview.jpg"),
            np.vstack([np.hstack([cv2.resize(c[f], (160, 160))[..., ::-1] for f in (0, 5, 10, 15)]) for c in prev]))

Pool =ThreadPoolExecutor if os.name == "nt" else ProcessPoolExecutor   # spawn on Windows would re-run the script
with Pool(4) as ex:
    train_feats = encode_all(bounded_map(ex, crop_video, jobs), len(wdf))
print("train feats", {k: v.shape for k, v in train_feats.items()}, f"{time.time() - t0:.0f}s")

t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    test_feats = encode_all(bounded_map(ex, crop_test, range(len(test_ids)), 64), len(test_ids))
print("test feats", {k: v.shape for k, v in test_feats.items()}, f"{time.time() - t0:.0f}s")
for k in BB:
    np.save(WORK / f"train_{k}_fp16.npy", train_feats[k]); np.save(WORK / f"test_{k}_fp16.npy", test_feats[k])
wdf.to_csv(WORK / "train_windows.csv", index=False)
DIMS = {k: dims(k) for k in BB}
del BB
if DEV == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# ## 2. Pose window statistics (same featurisation as v2)

# %%
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
def pose_stats(Wall, bs=4096):
    Wall = torch.from_numpy(np.ascontiguousarray(Wall)).float()
    return torch.cat([window_stats(featurize(Wall[i:i + bs].to(DEV))).cpu()
                      for i in range(0, len(Wall), bs)]).numpy()


tr_pose_w = train_pose[wdf.gstart.values[:, None] + np.arange(T)]
P_TR, P_TE = pose_stats(tr_pose_w), pose_stats(test_pose)
del tr_pose_w
print("pose stats", P_TR.shape, P_TE.shape)

# %% [markdown]
# ## 3. Re-link shuffled test windows (frame 30 -> frame 01) for the context submission

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
n_te = len(test_ids)
D = np.stack([np.abs(f30[k][None] - f01).mean(1) for k in range(n_te)])
np.fill_diagonal(D, 1e6)
scale = np.maximum(0.5 * (np.abs(f30 - f29).mean(1)[:, None] + np.abs(f02 - f01).mean(1)[None]), 0.05)
R = np.minimum(D / scale, 1e3)
big = np.full((2 * n_te, 2 * n_te), 1e3)
big[:n_te, :n_te] = R
big[:n_te, n_te:][np.eye(n_te, dtype=bool)] = 4.0
big[n_te:, :n_te][np.eye(n_te, dtype=bool)] = 4.0
big[n_te:, n_te:] = 0
r_, c_ = linear_sum_assignment(big)
links = [(int(i), int(j)) for i, j in zip(r_, c_) if i < n_te and j < n_te and R[i, j] < 2.5]
nbr = [[] for _ in range(n_te)]
for i, j in links:
    nbr[i].append(j); nbr[j].append(i)


def neighbourhood(k, hops=7):
    seen, frontier = {k}, [k]
    for _ in range(hops):
        frontier = [m for f in frontier for m in nbr[f] if m not in seen]
        seen.update(frontier)
    return sorted(seen)


TEST_NB = [neighbourhood(k) for k in range(n_te)]
print("links kept:", len(links), "/", n_te, "| mean neighbourhood size", np.mean([len(b) for b in TEST_NB]).round(2))

# %% [markdown]
# ## 4. Multiple-instance head, GroupKFold by subject

# %%
def feature_block(name, which):
    if name == "pose":
        return P_TR, P_TE
    sl = DIMS[name]
    cols = np.r_[tuple(np.arange(sl[w].start, sl[w].stop) for w in which)]
    return train_feats[name][:, cols].astype(np.float32), test_feats[name][:, cols].astype(np.float32)


CONFIGS = {
    "vmae": [("vmae", ["pooled", "logits"])],
    "vjepa": [("vjepa", ["pooled", "logits"])],
    "vjepa_grid": [("vjepa", ["pooled", "grid", "logits"])],
    "pose": [("pose", None)],
    "all": [("vmae", ["pooled", "logits"]), ("vjepa", ["pooled", "logits"]), ("pose", None)],
}


class MIL(nn.Module):
    def __init__(self, din, d=128, p=0.4):
        super().__init__()
        self.enc = nn.Sequential(nn.Dropout(0.25), nn.Linear(din, d), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(d, d), nn.GELU())
        self.att = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
        self.drop = nn.Dropout(p)
        self.out_bin, self.out_beh = nn.Linear(2 * d, 1), nn.Linear(2 * d, len(BEHAVIORS))

    def forward(self, x, mask):                       # (B,L,din), (B,L) bool
        h = self.enc(x)
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(1)
        z = torch.cat([(a[..., None] * h).sum(1), h.masked_fill(~mask[..., None], -1e4).amax(1)], 1)
        z = self.drop(z)
        return self.out_bin(z).squeeze(1), self.out_beh(z)


def prob_from(lb, lh):
    return 0.5 * torch.sigmoid(lb) + 0.5 * lh.softmax(1)[:, 4:].sum(1)


V_LAB, V_BEH = vdf.label.values, vdf.beh_id.values
VID_ROWS = {v: np.where(wdf.video.values == v)[0] for v in range(len(vdf))}   # stride-15 rows, ordered


def sample_bags(videos, n_per_video, L):
    """contiguous bags of L windows spaced 1 s apart (stride-15 rows, step 2)"""
    bags, ys, yb = [], [], []
    for v in videos:
        rows = VID_ROWS[v]
        for _ in range(n_per_video):
            ph = random.randint(0, 1)
            seq = rows[ph::2]
            l = min(L, len(seq))
            s0 = random.randint(0, len(seq) - l)
            bags.append(seq[s0:s0 + l]); ys.append(V_LAB[v]); yb.append(V_BEH[v])
    return bags, np.array(ys), np.array(yb)


def pad(bags, X):
    L = max(len(b) for b in bags)
    idx = np.zeros((len(bags), L), np.int64); mask = np.zeros((len(bags), L), bool)
    for i, b in enumerate(bags):
        idx[i, :len(b)] = b; mask[i, :len(b)] = True
    return X[torch.from_numpy(idx).to(DEV)], torch.from_numpy(mask).to(DEV)


EPOCHS = 3 if SMOKE else 60
BAGS_PER_VIDEO, BATCH = 8, 64


def train_mil(Xtr, tr_videos, seed):
    seed_all(seed)
    model = MIL(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    steps = EPOCHS * math.ceil(len(tr_videos) * BAGS_PER_VIDEO / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    for ep in range(EPOCHS):
        model.train()
        order = np.random.permutation(np.repeat(tr_videos, BAGS_PER_VIDEO))
        for i in range(0, len(order), BATCH):
            L = random.choice([1, 1, 2, 4, 8, 12, 15])
            bags, y, yb = sample_bags(order[i:i + BATCH], 1, L)
            x, m = pad(bags, Xtr)
            x = x + 0.1 * torch.randn_like(x)
            lb, lh = model(x, m)
            yt = torch.as_tensor(y, device=DEV).float()
            loss = F.binary_cross_entropy_with_logits(lb, yt * 0.9 + 0.05) \
                + 0.5 * F.cross_entropy(lh, torch.as_tensor(yb, device=DEV), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    return model


@torch.no_grad()
def predict_bags(model, bags, X, bs=512):
    model.eval(); out = []
    for i in range(0, len(bags), bs):
        x, m = pad(bags[i:i + bs], X)
        out.append(prob_from(*model(x, m)).cpu())
    return torch.cat(out).numpy()


# evaluation windows = 1-s windows aligned like the test set (offset % 30 == 0)
EV = wdf.index[(wdf.off % T) == 0].values
EV_VID = wdf.video.values[EV]
EV_WIN = wdf.off.values[EV] // T


def eval_bags(rows):
    single = [[r] for r in rows]
    ctx = []
    for r in rows:
        v, w = wdf.video.values[r], wdf.off.values[r] // T
        same = [q for q in VID_ROWS[v] if wdf.off.values[q] % T == 0 and abs(wdf.off.values[q] // T - w) <= 7]
        ctx.append(same)
    return single, ctx


SEEDS = [42] if SMOKE else [42, 7, 2026]
oof = pd.DataFrame({"video": EV_VID, "win": EV_WIN, "label": V_LAB[EV_VID],
                    "behavior": vdf.behavior.values[EV_VID], "angle": vdf.angle.values[EV_VID],
                    "subject": vdf.subject.values[EV_VID]})
test_out = pd.DataFrame({"id": test_ids})
TEST_SINGLE = [[k] for k in range(n_te)]
folds = list(GroupKFold(n_splits=5).split(vdf, groups=vdf.subject))
t0 = time.time()
for cname, parts in CONFIGS.items():
    blocks = [feature_block(n, w) for n, w in parts]
    Xtr_np = np.concatenate([b[0] for b in blocks], 1); Xte_np = np.concatenate([b[1] for b in blocks], 1)
    p1, pc = np.zeros(len(EV)), np.zeros(len(EV))
    t1, tc = np.zeros(n_te), np.zeros(n_te)
    for fold, (tr, va) in enumerate(folds):
        tr_rows = np.concatenate([VID_ROWS[v] for v in tr])
        mu, sd = Xtr_np[tr_rows].mean(0), Xtr_np[tr_rows].std(0) + 1e-3
        Xtr = torch.from_numpy(np.clip((Xtr_np - mu) / sd, -6, 6)).float().to(DEV)
        Xte = torch.from_numpy(np.clip((Xte_np - mu) / sd, -6, 6)).float().to(DEV)
        va_mask = np.isin(EV_VID, va)
        single, ctx = eval_bags(EV[va_mask])
        for seed in SEEDS:
            m = train_mil(Xtr, tr, seed + 100 * fold)
            p1[va_mask] += predict_bags(m, single, Xtr) / len(SEEDS)
            pc[va_mask] += predict_bags(m, ctx, Xtr) / len(SEEDS)
            t1 += predict_bags(m, TEST_SINGLE, Xte) / (len(SEEDS) * len(folds))
            tc += predict_bags(m, TEST_NB, Xte) / (len(SEEDS) * len(folds))
    oof[f"{cname}_1"], oof[f"{cname}_ctx"] = p1, pc
    test_out[f"{cname}_1"], test_out[f"{cname}_ctx"] = t1, tc
    print(f"{cname:11s} window auc={roc_auc_score(oof.label, p1):.4f} acc={accuracy_score(oof.label, p1 > .5):.4f} | "
          f"context auc={roc_auc_score(oof.label, pc):.4f} acc={accuracy_score(oof.label, pc > .5):.4f} "
          f"{time.time() - t0:.0f}s", flush=True)

# %% [markdown]
# ## 5. Blend with v2 on OOF, CV report, submissions

# %%
o2 = pd.read_csv(V2 / "oof.csv")[["video", "win", "blend"]].rename(columns={"blend": "v2_1"})
oof = oof.merge(o2, on=["video", "win"], how="left")
oof["v2_1"] = oof.v2_1.fillna(0.5)
oof = oof.sort_values(["video", "win"]).reset_index(drop=True)
oof["v2_ctx"] = oof.groupby("video")["v2_1"].transform(lambda s: s.rolling(15, center=True, min_periods=1).mean())
t2 = pd.read_csv(V2 / "test_probs.csv").set_index("id").loc[test_ids, "prob"].values
test_out["v2_1"] = t2
test_out["v2_ctx"] = [t2[b].mean() for b in TEST_NB]


def report(y, p):
    return dict(acc=round(accuracy_score(y, p > 0.5), 4), f1=round(f1_score(y, p > 0.5), 4),
                auc=round(roc_auc_score(y, p), 4))


def best_blend(cols, step=0.1):
    """greedy forward selection of averaged columns (Caruana), scored by OOF AUC"""
    chosen, best = [], -1
    for _ in range(int(1 / step)):
        scores = {c: roc_auc_score(oof.label, oof[chosen + [c]].mean(1)) for c in cols}
        c, s = max(scores.items(), key=lambda kv: kv[1])
        if s <= best + 1e-5 and chosen:
            break
        chosen.append(c); best = s
    w = pd.Series(chosen).value_counts(normalize=True)
    return w.to_dict(), best


cv = {}
for kind in ["1", "ctx"]:
    cols = [c for c in oof.columns if c.endswith(f"_{kind}")]
    for c in cols:
        cv[c] = report(oof.label, oof[c])
    w, _ = best_blend(cols)
    oof[f"final_{kind}"] = sum(oof[c] * wt for c, wt in w.items())
    test_out[f"final_{kind}"] = sum(test_out[c] * wt for c, wt in w.items())
    cv[f"final_{kind}"] = report(oof.label, oof[f"final_{kind}"])
    cv[f"final_{kind}_weights"] = {k: round(float(v), 3) for k, v in w.items()}
    for a in ["front", "side"]:
        s = oof[oof.angle == a]
        cv[f"final_{kind}_{a}"] = report(s.label, s[f"final_{kind}"])
print(json.dumps(cv, indent=1))
print("\nper-behaviour window accuracy (final_ctx):")
print(oof.assign(ok=((oof.final_ctx > 0.5) == oof.label)).groupby(["behavior", "angle"]).ok.mean().unstack().round(3))
oof.to_csv(WORK / "oof.csv", index=False)
test_out.to_csv(WORK / "test_probs.csv", index=False)

sub["label"] = (test_out.final_1.values > 0.5).astype(int)
sub.to_csv(WORK / "submission.csv", index=False)
sub_s = sub.copy()
sub_s["label"] = (test_out.final_ctx.values > 0.5).astype(int)
sub_s.to_csv(WORK / "submission_smooth.csv", index=False)
print("pure:", sub.label.value_counts().to_dict(), "| context:", sub_s.label.value_counts().to_dict(),
      "| differ:", int((sub.label != sub_s.label).sum()))
json.dump({"cv": cv, "n_links": len(links)}, open(WORK / "cv_report.json", "w"), indent=1)
