"""Group test chains into (angle, subject) with ~8 videos each, by clothing colour + scene."""
import sys, json, collections
from pathlib import Path
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
DATA = Path(sys.argv[1]); comp = json.load(open(sys.argv[2])); tp = np.load(sys.argv[3]); out = sys.argv[4]
ids, chains = comp["ids"], comp["chains"]

def ratio(P):
    sw = np.abs(P[..., 15] - P[..., 18]); bh = P[..., 54] - P[..., 52]
    return np.median(sw / np.maximum(bh, 1e-3))

def appearance(m):
    im = cv2.imread(str(DATA / "test" / ids[m] / "15.jpg")); H, W = im.shape[:2]
    x1, y1, x2, y2 = (tp[m, 14, 51:55] * [W, H, W, H]).astype(int)
    crop = im[max(y1, 0):y2, max(x1, 0):x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    small = cv2.resize(im, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
    return np.sqrt(h / h.sum()), small, cv2.resize(crop, (60, 90))

feat, tiles, angle = [], [], []
for ch in chains:
    ms = ch["members"][:: max(1, len(ch["members"]) // 5)]
    a = [appearance(m) for m in ms]
    feat.append(np.concatenate([np.mean([x[0] for x in a], 0) * 3, np.mean([x[1] for x in a], 0)]))
    tiles.append(a[len(a) // 2][2])
    angle.append("front" if ratio(tp[ch["members"]]) > 0.12 else "side")
feat = np.array(feat); angle = np.array(angle)

groups = np.zeros(len(chains), int); rows = []
for ai, a in enumerate(["front", "side"]):
    idx = np.where(angle == a)[0]; X = feat[idx]
    cen = KMeans(6, n_init=50, random_state=0).fit(X).cluster_centers_
    cap = int(np.ceil(len(idx) / 6))
    for _ in range(30):   # capacity-constrained k-means
        cost = ((X[:, None] - cen[None]) ** 2).sum(-1)
        slots = np.repeat(np.arange(6), cap)
        r, c = linear_sum_assignment(cost[:, slots])
        lab = slots[c][np.argsort(r)]
        cen = np.stack([X[lab == k].mean(0) for k in range(6)])
    groups[idx] = ai * 6 + lab
    for k in range(6):
        mem = [tiles[i] for i in idx[lab == k]]
        mem = (mem + [np.zeros_like(tiles[0])] * 9)[:9]
        rows.append(np.hstack(mem))
    print(a, len(idx), sorted(collections.Counter(lab).values()))
cv2.imwrite(out, np.vstack(rows))
json.dump(dict(angle=angle.tolist(), group=groups.tolist()), open(out + ".json", "w"))
