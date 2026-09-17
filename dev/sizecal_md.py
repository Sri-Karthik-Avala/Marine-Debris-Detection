import sys; sys.path.insert(0, 'dev'); sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd
import solution as S
train = pd.read_csv('train.csv')
LG = [S.parse_boxes(s) for s in train.logged_boxes]; NW = [S.parse_boxes(s) for s in train.new_boxes]; GRP = train.survey_id.values
C = pickle.load(open(sys.argv[1], 'rb')); idx = sorted(C)
def pairs(ids):
    X, Y = [], []
    for i in ids:
        bx, sc, rb, q = C[i]; g = NW[i]
        allg = np.concatenate([LG[i], g]) if len(LG[i]) else g
        if not len(rb) or not len(allg): continue
        M = S.iou_matrix(rb, allg); k = M.argmax(1); ok = (M.max(1) >= 0.5) & (sc >= 0.3)
        b = rb[ok]; gg = allg[k[ok]]
        w = b[:, 2]-b[:, 0]; h = b[:, 3]-b[:, 1]; gw = gg[:, 2]-gg[:, 0]; gh = gg[:, 3]-gg[:, 1]
        X.append(np.stack([np.log(np.sqrt(w*h)), np.log(w/h)], 1)); Y.append(np.stack([np.log(gw/w), np.log(gh/h), ((gg[:, 0]+gg[:, 2])-(b[:, 0]+b[:, 2]))/2/w, ((gg[:, 1]+gg[:, 3])-(b[:, 1]+b[:, 3]))/2/h], 1))
    return np.concatenate(X), np.concatenate(Y)
def fit(ids, deg):
    X, Y = pairs(ids); A = np.stack([X[:, 0]**d for d in range(deg+1)], 1)
    coef = np.linalg.lstsq(A, Y[:, :2], rcond=None)[0]; return coef
def apply(b, coef, deg):
    if not len(b): return b
    w = b[:, 2]-b[:, 0]; h = b[:, 3]-b[:, 1]; s = np.log(np.sqrt(w*h)); A = np.stack([s**d for d in range(deg+1)], 1)
    d = A @ coef; cx = (b[:, 0]+b[:, 2])/2; cy = (b[:, 1]+b[:, 3])/2; nw = w*np.exp(d[:, 0]); nh = h*np.exp(d[:, 1])
    return np.clip(np.stack([cx-nw/2, cy-nh/2, cx+nw/2, cy+nh/2], 1), 0, 640)
def f1(ids, coef, deg, thr):
    tp = p = t = 0
    for i in ids:
        bx, sc, rb, q = C[i]; rb2 = apply(rb, coef, deg) if coef is not None else rb
        pr = S.select(rb2, sc, LG[i], 0.3, thr, 1); p += len(pr); tp += S.match_count(pr, NW[i]); t += len(NW[i])
    return 2*tp/(p+t)
thrs = np.arange(0.5, 0.9, 0.025)
ug = np.unique(GRP); rng = np.random.default_rng(0); out = {'none': [], 'deg1': [], 'deg2': []}
for rep in range(3):
    perm = rng.permutation(ug); fold = {g: k % 2 for k, g in enumerate(perm)}
    for fk in (0, 1):
        tr = [i for i in idx if fold[GRP[i]] != fk]; te = [i for i in idx if fold[GRP[i]] == fk]
        for name, deg in (('none', None), ('deg1', 1), ('deg2', 2)):
            coef = fit(tr, deg) if deg else None
            thr = max((f1(tr, coef, deg, t), t) for t in thrs)[1]
            out[name].append(f1(te, coef, deg, thr))
print({k: round(np.mean(v), 4) for k, v in out.items()})
X, Y = pairs(idx); print('median center offsets', np.median(Y[:, 2:], 0).round(4))
