"""Link shuffled test windows (same rule as the notebooks) and summarise the components."""
import sys, json, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, pandas as pd
from scipy.optimize import linear_sum_assignment
DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2])
ids = pd.read_csv(DATA / "sample_submission.csv").id.tolist()
def thumb(p):
    im = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8)
    return cv2.resize(im, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
def seq(sid):
    d = DATA / "test" / sid
    return [thumb(d / f) for f in ("01.jpg", "02.jpg", "29.jpg", "30.jpg")]
with ThreadPoolExecutor(8) as ex:
    th = list(ex.map(seq, ids))
f01, f02, f29, f30 = (np.stack([t[i] for t in th]) for i in range(4))
n = len(ids)
D = np.stack([np.abs(f30[k][None] - f01).mean(1) for k in range(n)]); np.fill_diagonal(D, 1e6)
scale = np.maximum(0.5 * (np.abs(f30 - f29).mean(1)[:, None] + np.abs(f02 - f01).mean(1)[None]), 0.05)
R = np.minimum(D / scale, 1e3)
big = np.full((2 * n, 2 * n), 1e3); big[:n, :n] = R
big[:n, n:][np.eye(n, dtype=bool)] = 4.0; big[n:, :n][np.eye(n, dtype=bool)] = 4.0; big[n:, n:] = 0
r, c = linear_sum_assignment(big)
nxt = {int(i): int(j) for i, j in zip(r, c) if i < n and j < n and R[i, j] < 2.5}
prv = {j: i for i, j in nxt.items()}
# components via union-find
par = list(range(n))
def f(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for i, j in nxt.items(): par[f(i)] = f(j)
comp = [f(i) for i in range(n)]
sizes = collections.Counter(comp)
cyc = set()
for s in set(comp):
    members = [i for i in range(n) if comp[i] == s]
    if all(i in nxt for i in members): cyc.add(s)
print("links", len(nxt), "components", len(sizes), "cycles", len(cyc))
print("size histogram", sorted(collections.Counter(sizes.values()).items()))
# order chains from head
chains = []
for s in set(comp):
    members = [i for i in range(n) if comp[i] == s]
    head = [i for i in members if i not in prv]
    if head:
        order, x = [], head[0]
        while x is not None and x not in order:
            order.append(x); x = nxt.get(x)
    else:
        order = members
    chains.append(dict(members=order, cycle=s in cyc))
json.dump(dict(ids=ids, chains=chains), open(OUT, "w"))
