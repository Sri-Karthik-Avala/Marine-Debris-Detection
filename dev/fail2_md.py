import sys; sys.path.insert(0, 'dev'); sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd, json
import solution as S
from metric_md import group_score
train = pd.read_csv('train.csv')
LG = [S.parse_boxes(s) for s in train.logged_boxes]; NW = [S.parse_boxes(s) for s in train.new_boxes]; GRP = train.survey_id.values
C = pickle.load(open(sys.argv[1], 'rb')); idx = sorted(C)
SUP, THR, FORCE = 0.3, float(sys.argv[2]), 1
pred = {i: S.select(C[i][2], C[i][1], LG[i], SUP, THR, FORCE) for i in idx}
# per-GT table
rows = []
for i in idx:
    bx, sc, rb, q = C[i]; g = NW[i]; p = pred[i]
    Mref = S.iou_matrix(rb, g) if len(rb) else np.zeros((0, len(g))); Mraw = S.iou_matrix(bx, g) if len(bx) else np.zeros((0, len(g)))
    Mp = S.iou_matrix(p, g) if len(p) else np.zeros((0, len(g)))
    for j in range(len(g)):
        w = g[j, 2]-g[j, 0]; h = g[j, 3]-g[j, 1]
        edge = g[j, 0] <= 0.5 or g[j, 1] <= 0.5 or g[j, 2] >= 639.5 or g[j, 3] >= 639.5
        best_ref = Mref[:, j].max() if len(rb) else 0; best_raw = Mraw[:, j].max() if len(bx) else 0
        sc_best = sc[Mref[:, j].argmax()] if len(rb) else 0
        rows.append(dict(i=i, side=np.sqrt(w*h), edge=edge, nlog=len(LG[i]), ntot=len(LG[i])+len(g), raw=best_raw, ref=best_ref, sc_best=sc_best, hit=(Mp[:, j].max() >= 0.81) if len(p) else False))
D = pd.DataFrame(rows)
D['size_bin'] = pd.cut(D.side, [0, 25, 35, 50, 80, 700])
D['nlog_bin'] = pd.cut(D.nlog, [-1, 0, 2, 5, 100])
agg = lambda d: pd.Series({'n': len(d), 'raw>=.5': (d.raw >= .5).mean(), 'ref>=.5': (d.ref >= .5).mean(), 'ref>=.81': (d.ref >= .81).mean(), 'hit': d.hit.mean(), 'ref81_but_lowscore': ((d.ref >= .81) & ~d.hit).mean()})
pd.set_option('display.width', 200)
print(D.groupby('size_bin', observed=True).apply(agg).round(3))
print(D.groupby('edge').apply(agg).round(3))
print(D.groupby('nlog_bin', observed=True).apply(agg).round(3))
# near-miss geometry bias: best refined cand with IoU in [0.5,0.81)
dl = []
for i in idx:
    bx, sc, rb, q = C[i]; g = NW[i]
    if not len(rb) or not len(g): continue
    M = S.iou_matrix(rb, g)
    for j in range(len(g)):
        k = M[:, j].argmax()
        if 0.5 <= M[k, j]:
            b = rb[k]; gg = g[j]; w = gg[2]-gg[0]; h = gg[3]-gg[1]
            dl.append([(b[0]-gg[0])/w, (b[1]-gg[1])/h, (b[2]-gg[2])/w, (b[3]-gg[3])/h, np.log((b[2]-b[0])/w), np.log((b[3]-b[1])/h), M[k, j], np.sqrt(w*h)])
dl = np.array(dl)
print('refined vs GT (IoU>=0.5): median edge offsets /size [l,t,r,b]', np.median(dl[:, :4], 0).round(4), 'median log w,h ratio', np.median(dl[:, 4:6], 0).round(4))
print('mean abs edge err /size', np.abs(dl[:, :4]).mean(0).round(4), ' by size bins:')
for lo, hi in ((0, 25), (25, 35), (35, 50), (50, 80), (80, 700)):
    m = (dl[:, 7] >= lo) & (dl[:, 7] < hi); print(f'  side {lo}-{hi}: n {m.sum()} mean abs edge err/size {np.abs(dl[m, :4]).mean():.4f}  in px {np.abs(dl[m, :4]).mean()*(lo+hi)/2 if hi<700 else 0:.2f}  median logw {np.median(dl[m, 4]):.3f}')
# FP analysis
cat = {'tp': 0, 'near(0.5-0.81)': 0, 'weak(0.1-0.5)': 0, 'logged_near(0.1-0.3)': 0, 'none': 0}
for i in idx:
    p = pred[i]
    if not len(p): continue
    Mg = S.iou_matrix(p, NW[i]).max(1) if len(NW[i]) else np.zeros(len(p)); Ml = S.iou_matrix(p, LG[i]).max(1) if len(LG[i]) else np.zeros(len(p))
    for a, b in zip(Mg, Ml):
        if a >= .81: cat['tp'] += 1
        elif a >= .5: cat['near(0.5-0.81)'] += 1
        elif a >= .1: cat['weak(0.1-0.5)'] += 1
        elif b >= .1: cat['logged_near(0.1-0.3)'] += 1
        else: cat['none'] += 1
print('pred categories', cat)
# selection rules
T = lambda ids: sum(len(NW[i]) for i in ids)
def score_rule(ids, sup, thr, thr0, cap0):
    tp = p = 0; preds = []
    for i in ids:
        th = thr0 if len(LG[i]) == 0 else thr
        pr = S.select(C[i][2], C[i][1], LG[i], sup, th, 1)
        if len(LG[i]) == 0 and cap0 and len(pr) > cap0:
            s = C[i][1]; keep = S.iou_matrix(C[i][2], LG[i]).max(1) < sup if len(LG[i]) else np.ones(len(s), bool)
            o = np.argsort(-s[keep]); pr = C[i][2][keep][o][:cap0]
        preds.append(pr); p += len(pr); tp += S.match_count(pr, NW[i])
    return 2*tp/(p+T(ids)), preds
thrs = np.arange(0.4, 0.95, 0.025)
for sup in (0.1, 0.2, 0.3, 0.4, 0.5):
    b = max((score_rule(idx, sup, t, t, 0)[0], t) for t in thrs); print(f'sup {sup}: pooled {b[0]:.4f} thr {b[1]:.3f}')
ug = np.unique(GRP); rng = np.random.default_rng(0)
res = {'global': [], 'ledger0_thr': [], 'ledger0_top1': []}
gres = {k: [] for k in res}
for rep in range(4):
    perm = rng.permutation(ug); fold = {g: k % 2 for k, g in enumerate(perm)}
    for fk in (0, 1):
        tr = [i for i in idx if fold[GRP[i]] != fk]; te = [i for i in idx if fold[GRP[i]] == fk]
        cfgs = {'global': [(0.3, t, t, 0) for t in thrs], 'ledger0_thr': [(0.3, t, t0, 0) for t in thrs[::2] for t0 in thrs[::2]], 'ledger0_top1': [(0.3, t, t0, 1) for t in thrs for t0 in (t,)]}
        for name, cl in cfgs.items():
            best = max((score_rule(tr, *c)[0], c) for c in cl)[1]
            f, pr = score_rule(te, *best)
            res[name].append(f); gres[name].append(group_score(pr, [NW[i] for i in te], GRP[te]))
print({k: round(np.mean(v), 4) for k, v in res.items()}, 'group', {k: round(np.mean(v), 4) for k, v in gres.items()})
