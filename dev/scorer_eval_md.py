import sys; sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd
import solution as S
from sklearn.metrics import roc_auc_score
train = pd.read_csv('train.csv'); GRP = train.survey_id.values
LG = [S.parse_boxes(s) for s in train.logged_boxes]; NW = [S.parse_boxes(s) for s in train.new_boxes]
OOF = pickle.load(open(sys.argv[1], 'rb')); S2 = pickle.load(open(sys.argv[2], 'rb'))
lab, a_sc, a_p81, a_p50 = [], [], [], []
for i in OOF:
    bx, sc, rb, q = OOF[i]
    if not len(rb): continue
    keep = (S.iou_matrix(rb, LG[i]).max(1) < 0.3 if len(LG[i]) else np.ones(len(rb), bool)) & (sc >= 0.3)
    l = (S.iou_matrix(rb, NW[i]).max(1) >= 0.81) if len(NW[i]) else np.zeros(len(rb), bool)
    lab.append(l[keep]); a_sc.append(sc[keep]); a_p81.append(S2[i][keep, 0]); a_p50.append(S2[i][keep, 1])
lab = np.concatenate(lab); a_sc = np.concatenate(a_sc); a_p81 = np.concatenate(a_p81); a_p50 = np.concatenate(a_p50)
print('AUC sc %.3f p81 %.3f p50 %.3f sc*p81 %.3f sc*p50 %.3f' % (roc_auc_score(lab, a_sc), roc_auc_score(lab, a_p81), roc_auc_score(lab, a_p50), roc_auc_score(lab, a_sc*a_p81), roc_auc_score(lab, a_sc*a_p50)))
idx = sorted(OOF); T = sum(len(NW[i]) for i in idx)
def rule_score(i, a, b, c):
    sc = OOF[i][1]
    if not len(sc): return sc
    return (sc**a)*(S2[i][:, 0]**b)*(S2[i][:, 1]**c)
def f1(ids, a, b, c, thr):
    tp = p = t = 0
    for i in ids:
        pr = S.select(OOF[i][2], rule_score(i, a, b, c), LG[i], 0.3, thr, 1); p += len(pr); tp += S.match_count(pr, NW[i]); t += len(NW[i])
    return 2*tp/(p+t)
rules = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0.5, 1, 0), (1, 0.5, 0), (1, 0, 1), (1, 0.5, 0.5), (0, 0.5, 0.5)]
qs = np.linspace(0.55, 0.995, 60)
for r in rules:
    allsc = np.concatenate([rule_score(i, *r) for i in idx if len(OOF[i][1])])
    best = max((f1(idx, *r, t), t) for t in np.quantile(allsc, qs))
    print('rule', r, 'in-sample F1 %.4f' % best[0], flush=True)
ug = np.unique(GRP); rng = np.random.default_rng(0); res = {r: [] for r in [(1, 0, 0), (1, 1, 0), (0.5, 1, 0), (1, 0.5, 0.5)]}
for rep in range(2):
    perm = rng.permutation(ug); fold = {g: k % 2 for k, g in enumerate(perm)}
    for fk in (0, 1):
        tr = [i for i in idx if fold[GRP[i]] != fk]; te = [i for i in idx if fold[GRP[i]] == fk]
        for r in res:
            allsc = np.concatenate([rule_score(i, *r) for i in tr if len(OOF[i][1])])
            thr = max((f1(tr, *r, t), t) for t in np.quantile(allsc, qs[::2]))[1]
            res[r].append(f1(te, *r, thr))
print('honest', {str(k): round(np.mean(v), 4) for k, v in res.items()})
