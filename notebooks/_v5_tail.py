# %% [markdown]
# ## Per-second cheating-moment detector (instance-level MIL)
# Training labels exist only per video. With log-mean-exp pooling over the windows of a video, the model has to
# give high scores to the windows that make a cheat video a cheat video and low scores to windows that also
# occur in normal videos (writing, sitting still). Each window is then scored on its own.

# %%
class InstMIL(nn.Module):
    def __init__(self, din, d=128, p=0.3, r=2.0):
        super().__init__()
        self.enc = nn.Sequential(nn.Dropout(0.2), nn.Linear(din, d), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(d, d), nn.GELU(), nn.Dropout(p))
        self.head = nn.Linear(d, 9)          # 0: cheat moment, 1..8: behaviour moment
        self.r = r

    def forward(self, x, mask):
        s = self.head(self.enc(x))
        sm = s.masked_fill(~mask[..., None], -1e4)
        n = mask.sum(1, keepdim=True).float()
        return (torch.logsumexp(self.r * sm, 1) - torch.log(n)) / self.r, s


IEPOCHS = 1 if SMOKE else 80


def train_inst(X, videos, seed):
    seed_all(seed)
    model = InstMIL(X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    reps, bs = 4, 32
    steps = IEPOCHS * math.ceil(len(videos) * reps / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    for _ in range(IEPOCHS):
        model.train()
        order = np.random.permutation(np.repeat(videos, reps))
        for i in range(0, len(order), bs):
            bags = []
            for v in order[i:i + bs]:
                seq = VID_ROWS[v][random.randint(0, 1)::2]
                keep = seq[np.random.rand(len(seq)) > 0.15]
                bags.append(keep if len(keep) else seq)
            x, m = pad(bags, X)
            pooled, _ = model(x + 0.1 * torch.randn_like(x), m)
            vv = order[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(pooled[:, 0], torch.as_tensor(V_LAB[vv], device=DEV).float()) \
                + 0.5 * F.cross_entropy(pooled[:, 1:], torch.as_tensor(V_BEH[vv], device=DEV))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    return model.eval()


@torch.no_grad()
def inst_scores(model, X, bs=8192):
    return torch.cat([model.head(model.enc(X[i:i + bs])).cpu() for i in range(0, len(X), bs)]).numpy()


INST_CONFIGS = ["vjepa", "pose", "vjepa_grid"]
EV_ROWS = np.where(wdf.off.values % T == 0)[0]
EV_VID = wdf.video.values[EV_ROWS]
inst_oof = {c: np.zeros((len(EV_ROWS), 9)) for c in INST_CONFIGS}
inst_te = {c: np.zeros((len(test_ids), 9)) for c in INST_CONFIGS}
t0 = time.time()
for cname in INST_CONFIGS:
    bl = [block(n, w) for n, w in CONFIGS[cname]]
    Xtr_np, Xte_np = np.concatenate([b[0] for b in bl], 1), np.concatenate([b[1] for b in bl], 1)
    for fold, (tr, va) in enumerate(folds):
        rows = np.concatenate([VID_ROWS[v] for v in tr])
        mu, sd = Xtr_np[rows].mean(0), Xtr_np[rows].std(0) + 1e-3
        Xtr = torch.from_numpy(np.clip((Xtr_np - mu) / sd, -6, 6)).float().to(DEV)
        Xte = torch.from_numpy(np.clip((Xte_np - mu) / sd, -6, 6)).float().to(DEV)
        sel = np.isin(EV_VID, va)
        for seed in SEEDS:
            m = train_inst(Xtr, tr, seed + 100 * fold)
            inst_oof[cname][sel] += inst_scores(m, Xtr[torch.as_tensor(EV_ROWS[sel], device=DEV)]) / len(SEEDS)
            inst_te[cname] += inst_scores(m, Xte) / (len(SEEDS) * len(folds))
    print(f"inst {cname:11s} window auc (cheat-video vs normal-video windows) = "
          f"{roc_auc_score(V_LAB[EV_VID], inst_oof[cname][:, 0]):.4f} {time.time() - t0:.0f}s", flush=True)
INST_OOF = np.mean([inst_oof[c] for c in INST_CONFIGS], 0)
INST_TE = np.mean([inst_te[c] for c in INST_CONFIGS], 0)

# %% [markdown]
# ## Video-level decisions: per-angle vs joint front+side assignment

# %%
BEH_NAMES = ["Writing_Answers", "Thinking", "Looking_Around", "Checking_Smartwatch",
             "Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]


def hungarian(cost):
    r, c = linear_sum_assignment(cost)
    out = np.zeros(len(cost), int); out[r] = c
    return out


def nll(p):
    return -np.log(np.clip(p, 1e-6, 1))


def per_angle(pbeh, grp):
    beh = np.zeros(len(pbeh), int)
    for g in np.unique(grp):
        idx = np.where(grp == g)[0]
        beh[idx] = hungarian(np.concatenate([nll(pbeh[idx]), nll(pbeh[idx]) + 3.0], 1)) % 8
    return beh


def joint(pbeh, pairs_by_subject):
    """pairs_by_subject: list of [(front_idx, side_idx), ...] with 8 pairs each"""
    beh = np.full(len(pbeh), -1)
    for prs in pairs_by_subject:
        f = np.array([a for a, _ in prs]); s = np.array([b for _, b in prs])
        b = hungarian(nll(pbeh[f]) + nll(pbeh[s]))
        beh[f] = b; beh[s] = b
    return beh


w_vid = np.array([len(VID_EVAL[v]) for v in range(len(vdf))])
g_tr = pd.factorize(vdf.subject.astype(str) + "_" + vdf.angle)[0]
tr_pairs = []
for subj, g in vdf.reset_index().groupby("subject"):
    fr = g[g.angle == "front"].set_index("behavior")["index"]
    sd = g[g.angle == "side"].set_index("behavior")["index"]
    tr_pairs.append([(int(fr[b]), int(sd[b])) for b in fr.index])

combos = {"vjepa+pose+grid": ["vjepa", "pose", "vjepa_grid"], "all5": list(CONFIGS)}
report = {}
for name, cs in combos.items():
    ph = np.mean([res_va[c][1] for c in cs], 0)
    for mode, beh in [("angle", per_angle(ph, g_tr)), ("joint", joint(ph, tr_pairs))]:
        acc = float(np.average((beh >= 4).astype(int) == V_LAB, weights=w_vid))
        report[f"{name}_{mode}"] = dict(acc=round(acc, 4), behaviour_acc=round(float((beh == V_BEH).mean()), 4))
        print(f"{name:16s} {mode:5s} cheat acc={acc:.4f} behaviour acc={(beh == V_BEH).mean():.4f}")

# %% [markdown]
# ## How many seconds of a cheat video show the cheating? (OOF instance scores)

# %%
ev_lab, ev_beh = V_LAB[EV_VID], V_BEH[EV_VID]
s0 = INST_OOF[:, 0]
moment = {}
for pct in [80, 90, 95]:
    tau = np.percentile(s0[ev_lab == 0], pct)
    moment[f"p{pct}"] = tau
    frac = pd.Series(s0[ev_lab == 1] > tau).groupby(ev_beh[ev_lab == 1]).mean()
    print(f"threshold = {pct}th pct of normal-video windows: cheat-video windows above = "
          f"{(s0[ev_lab == 1] > tau).mean():.3f}", {BEH_NAMES[k]: round(v, 2) for k, v in frac.items()})
sb = INST_OOF[:, 1:]
own = sb[np.arange(len(sb)), ev_beh]
best_normal = sb[:, :4].max(1)
beh_rule = own > best_normal
print("behaviour rule (own behaviour score > best normal-behaviour score):")
print("  cheat-video windows flagged:", round(float(beh_rule[ev_lab == 1].mean()), 3),
      pd.Series(beh_rule[ev_lab == 1]).groupby(ev_beh[ev_lab == 1]).mean().round(2).rename(lambda k: BEH_NAMES[k]).to_dict())
print("  normal-video windows where some cheat behaviour beats every normal one:",
      round(float((sb[:, 4:].max(1) > best_normal)[ev_lab == 0].mean()), 3))
report["moment_thresholds"] = {k: float(v) for k, v in moment.items()}
report["moment_beh_rule_fraction_cheat_windows"] = float(beh_rule[ev_lab == 1].mean())

# %% [markdown]
# ## Test: joint assignment, per-second labels, submissions

# %%
pairs = STRUCTURE["pairs"]
pairs_by_subject = {}
for f, s in pairs:
    pairs_by_subject.setdefault(int(groups[f]), []).append((f, s))
pairs_by_subject = [v for _, v in sorted(pairs_by_subject.items())]
assert all(len(p) == 8 for p in pairs_by_subject), [len(p) for p in pairs_by_subject]

cs = combos["all5"]
ph_te = np.mean([res_te[c][1] for c in cs], 0)
pb_te = np.mean([res_te[c][0] for c in cs], 0)
beh_angle = per_angle(ph_te, groups)
beh_joint = joint(ph_te, pairs_by_subject)
print("test videos whose behaviour changes angle->joint:", int((beh_angle != beh_joint).sum()),
      "| cheat label changes:", int(((beh_angle >= 4) != (beh_joint >= 4)).sum()))

# cheapest normal<->cheat swap per subject (diagnostic for a systematic confusion)
swap_rows = []
for si, prs in enumerate(pairs_by_subject):
    f = np.array([a for a, _ in prs]); s = np.array([b for _, b in prs])
    C = nll(ph_te[f]) + nll(ph_te[s]); b = beh_joint[f]
    base = C[np.arange(8), b].sum()
    for i in range(8):
        for j in range(8):
            if b[i] < 4 <= b[j]:
                bb = b.copy(); bb[i], bb[j] = b[j], b[i]
                swap_rows.append(dict(subject=si, normal=BEH_NAMES[b[i]], cheat=BEH_NAMES[b[j]],
                                      extra_cost=C[np.arange(8), bb].sum() - base))
sw = pd.DataFrame(swap_rows).sort_values("extra_cost").groupby("subject").head(2)
print(sw.round(2).to_string(index=False))

sub = pd.read_csv(DATA / "sample_submission.csv")
win_video = np.zeros(len(test_ids), int)
for v, ch in enumerate(chains):
    win_video[ch] = v
vid_cheat = (beh_joint >= 4).astype(int)
sub.assign(label=vid_cheat[win_video]).to_csv(WORK / "submission_joint.csv", index=False)

wb = beh_joint[win_video]
own_te = INST_TE[np.arange(len(test_ids)), 1 + wb]
rule_te = own_te > INST_TE[:, 1:5].max(1)
lab_beh = (vid_cheat[win_video] == 1) & rule_te
sub.assign(label=lab_beh.astype(int)).to_csv(WORK / "submission_moment_beh.csv", index=False)
for pct, tau in moment.items():
    lab = (vid_cheat[win_video] == 1) & (INST_TE[:, 0] > tau)
    sub.assign(label=lab.astype(int)).to_csv(WORK / f"submission_moment_{pct}.csv", index=False)

pd.DataFrame({"video": range(len(chains)), "group": groups, "angle": STRUCTURE["angle"],
              "n_windows": [len(c) for c in chains], "p_cheat": pb_te,
              "behaviour_angle": [BEH_NAMES[b] for b in beh_angle],
              "behaviour_joint": [BEH_NAMES[b] for b in beh_joint]}).to_csv(WORK / "test_videos.csv", index=False)
pd.DataFrame(INST_TE, columns=["moment"] + [f"m_{b}" for b in BEH_NAMES]).assign(
    id=test_ids, video=win_video).to_csv(WORK / "test_inst_scores.csv", index=False)
pd.DataFrame(INST_OOF, columns=["moment"] + [f"m_{b}" for b in BEH_NAMES]).assign(
    video=EV_VID, off=wdf.off.values[EV_ROWS], label=ev_lab, behavior=vdf.behavior.values[EV_VID]).to_csv(
    WORK / "oof_inst_scores.csv", index=False)
np.savez(WORK / "video_preds.npz", **{f"va_{c}_bin": res_va[c][0] for c in CONFIGS},
         **{f"va_{c}_beh": res_va[c][1] for c in CONFIGS}, **{f"te_{c}_bin": res_te[c][0] for c in CONFIGS},
         **{f"te_{c}_beh": res_te[c][1] for c in CONFIGS})
for f in sorted(WORK.glob("submission_*.csv")):
    print(f.name, pd.read_csv(f).label.value_counts().to_dict())
json.dump(report, open(WORK / "cv_report.json", "w"), indent=1)
