import sys; sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd, itertools
import solution as S
from sklearn.metrics import roc_auc_score
train = pd.read_csv('train.csv'); D = pickle.load(open('dev/ho_dets_v1.pkl', 'rb')); ho = D['ho_idx']
rows = pickle.load(open('dev/feat_ho.pkl', 'rb'))
hl = [S.parse_boxes(train.logged_boxes[i]) for i in ho]; hn = [S.parse_boxes(train.new_boxes[i]) for i in ho]; grp = train.survey_id.values[ho]
# candidate-level AUC after ledger suppression
F, Y = [], []
for r, l, g in zip(rows, hl, hn):
    if r is None: continue
    keep = S.iou_matrix(r['rb'], l).max(1) < 0.3 if len(l) else np.ones(len(r['rb']), bool)
    lab = (S.iou_matrix(r['rb'], g).max(1) >= 0.81) if len(g) else np.zeros(len(r['rb']), bool)
    w = r['rb'][:, 2]-r['rb'][:, 0]; h = r['rb'][:, 3]-r['rb'][:, 1]
    f = np.stack([r['sc'], r['q2'], -r['std2'], -r['std1'], -r['mv2'], -r['qstd2'], np.log(w*h)], 1)
    F.append(f[keep & (r['sc'] >= 0.3)]); Y.append(lab[keep & (r['sc'] >= 0.3)])
F = np.concatenate(F); Y = np.concatenate(Y)
print('cands sc>=0.3', len(Y), 'pos', Y.sum())
for k, nm in enumerate(['sc', 'q2', '-std2', '-std1', '-mv2', '-qstd2', 'logarea']):
    print(nm, 'AUC %.3f' % roc_auc_score(Y, F[:, k]))
def score_fn(r, a, b, c):
    return r['sc']**a * r['q2']**b * np.exp(-c*r['std2'])
def evalset(idx, a, b, c, thr, sup=0.3, force=1):
    tp = p = t = 0
    for j in idx:
        r, l, g = rows[j], hl[j], hn[j]; t += len(g)
        if r is None: continue
        pr = S.select(r['rb'], score_fn(r, a, b, c), l, sup, thr, force); p += len(pr); tp += S.match_count(pr, g)
    return 2*tp/max(p+t, 1), tp, p
A = (1.0,); B = (0.0, 0.5, 1.0, 2.0); C = (0.0, 5.0, 10.0, 20.0, 40.0)
def search(idx):
    best = None
    for a, b, c in itertools.product(A, B, C):
        s_all = np.concatenate([score_fn(rows[j], a, b, c) for j in idx if rows[j] is not None])
        for thr in np.quantile(s_all, np.linspace(0.5, 0.995, 60)):
            f = evalset(idx, a, b, c, thr)[0]
            if best is None or f > best[0]: best = (f, a, b, c, thr)
    return best
allidx = list(range(len(ho)))
print('in-sample baseline (b0 c0):', max(evalset(allidx, 1, 0, 0, t)[0] for t in np.arange(0.3, 0.95, 0.025)))
print('in-sample best:', search(allidx))
ug = np.unique(grp); rng = np.random.default_rng(0)
for rep in range(3):
    perm = rng.permutation(ug); fold = {g: k % 2 for k, g in enumerate(perm)}
    tot = {'base': [0, 0, 0], 'resc': [0, 0, 0]}
    for fk in (0, 1):
        tr = [j for j in allidx if fold[grp[j]] != fk]; te = [j for j in allidx if fold[grp[j]] == fk]
        bb = search(tr); f, tp, p = evalset(te, *bb[1:])
        base_best = None
        for a, b, c in ((1, 0, 0),):
            s_all = np.concatenate([score_fn(rows[j], 1, 0, 0) for j in tr if rows[j] is not None])
            for thr in np.quantile(s_all, np.linspace(0.5, 0.995, 60)):
                ff = evalset(tr, 1, 0, 0, thr)[0]
                if base_best is None or ff > base_best[0]: base_best = (ff, thr)
        _, tp0, p0 = evalset(te, 1, 0, 0, base_best[1]); t = sum(len(hn[j]) for j in te)
        tot['resc'] = [x+y for x, y in zip(tot['resc'], (tp, p, t))]; tot['base'] = [x+y for x, y in zip(tot['base'], (tp0, p0, t))]
        print(' rep', rep, 'fold', fk, 'chosen', bb[1:4])
    print('rep', rep, {k: round(2*v[0]/(v[1]+v[2]), 4) for k, v in tot.items()})
