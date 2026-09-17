import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'dev')
import pickle, numpy as np, pandas as pd
import solution as S
from metric_md import group_score
train = pd.read_csv('train.csv'); D = pickle.load(open('dev/ho_dets_v1.pkl', 'rb')); ho = D['ho_idx']
rows = pickle.load(open('dev/feat_ho.pkl', 'rb'))
hl = [S.parse_boxes(train.logged_boxes[i]) for i in ho]; hn = [S.parse_boxes(train.new_boxes[i]) for i in ho]; grp = train.survey_id.values[ho]
def preds(idx, thr, cap, force, sup=0.3):
    out = []
    for j in idx:
        r = rows[j]
        if r is None: out.append(np.zeros((0, 4))); continue
        b, s = r['rb'], r['sc']
        if len(hl[j]):
            k = S.iou_matrix(b, hl[j]).max(1) < sup; b, s = b[k], s[k]
        o = np.argsort(-s); b, s = b[o], s[o]
        m = s >= thr
        if m.sum() == 0 and force and len(s): m[0] = True
        out.append(b[m][:cap])
    return out
def pooled(idx, pr):
    tp = sum(S.match_count(p, hn[j]) for p, j in zip(pr, idx)); P = sum(len(p) for p in pr); T = sum(len(hn[j]) for j in idx)
    return 2*tp/(P+T)
def rowmean(idx, pr):
    v = []
    for p, j in zip(pr, idx):
        tp = S.match_count(p, hn[j]); v.append(2*tp/(len(p)+len(hn[j])))
    return np.mean(v)
allidx = list(range(len(ho))); thrs = np.arange(0.3, 0.96, 0.025)
for cap in (1, 2, 3, 5, 100):
    for force in (0, 1):
        bp = max((pooled(allidx, preds(allidx, t, cap, force)), t) for t in thrs)
        bg = max((group_score(preds(allidx, t, cap, force), [hn[j] for j in allidx], grp), t) for t in thrs)
        br = max((rowmean(allidx, preds(allidx, t, cap, force)), t) for t in thrs)
        print(f'cap {cap} force {force}: pooled {bp[0]:.4f}@{bp[1]:.3f}  group {bg[0]:.4f}@{bg[1]:.3f}  rowmean {br[0]:.4f}@{br[1]:.3f}  group-at-pooled-thr {group_score(preds(allidx, bp[1], cap, force), [hn[j] for j in allidx], grp):.4f}  group-at-rowmean-thr {group_score(preds(allidx, br[1], cap, force), [hn[j] for j in allidx], grp):.4f}')
# 2-fold honest: select (cap, force, thr) by pooled vs by rowmean on one half, score group metric on other half
ug = np.unique(grp); rng = np.random.default_rng(0)
for crit in ('pooled', 'rowmean'):
    res = []
    for rep in range(5):
        perm = rng.permutation(ug); fold = {g: k % 2 for k, g in enumerate(perm)}
        sc = []
        for fk in (0, 1):
            tr = [j for j in allidx if fold[grp[j]] != fk]; te = [j for j in allidx if fold[grp[j]] == fk]
            fn = pooled if crit == 'pooled' else rowmean
            best = max((fn(tr, preds(tr, t, c, f)), t, c, f) for c in (1, 2, 3, 100) for f in (0, 1) for t in thrs)
            sc.append(group_score(preds(te, best[1], best[2], best[3]), [hn[j] for j in te], grp[te]))
        res.append(np.mean(sc))
    print(crit, 'honest group score mean over reps', np.round(res, 4), np.mean(res).round(4))
