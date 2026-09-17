import sys; sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd
import solution as S
train = pd.read_csv('train.csv'); D = pickle.load(open('dev/ho_dets_v1.pkl', 'rb')); ho = D['ho_idx']
cand = pickle.load(open(sys.argv[1], 'rb'))
hl = [S.parse_boxes(train.logged_boxes[i]) for i in ho]; hn = [S.parse_boxes(train.new_boxes[i]) for i in ho]
raw_best, ref_best = [], []
for (bx, sc, rb, q), l, g in zip(cand, hl, hn):
    for arr, store in ((bx, raw_best), (rb, ref_best)):
        if len(g) == 0: continue
        if len(arr) == 0: store += [0.0]*len(g); continue
        store += list(S.iou_matrix(arr, g).max(0))
for nm, v in (('raw', raw_best), ('ref', ref_best)):
    v = np.array(v); print(nm, 'GT covered by any candidate (score>=0.1): IoU>=0.5 %.3f >=0.7 %.3f >=0.81 %.3f >=0.9 %.3f' % tuple((v >= t).mean() for t in (0.5, 0.7, 0.81, 0.9)))
# selected set analysis at sup .3 thr .725 force 1
cats = {'tp': 0, 'loc_miss(0.5-0.81 to new)': 0, 'dup_of_matched': 0, 'logged_overlap>=0.5': 0, 'no_gt': 0}
tot_sel = 0
for (bx, sc, rb, q), l, g in zip(cand, hl, hn):
    pr = S.select(rb, sc, l, 0.3, 0.725, 1); tot_sel += len(pr)
    if len(pr) == 0: continue
    Mg = S.iou_matrix(pr, g) if len(g) else np.zeros((len(pr), 0)); Ml = S.iou_matrix(pr, l) if len(l) else np.zeros((len(pr), 0))
    used = set()
    for p in np.argsort(-(Mg.max(1) if Mg.shape[1] else np.zeros(len(pr)))):
        mg = Mg[p].max() if Mg.shape[1] else 0; j = int(Mg[p].argmax()) if Mg.shape[1] else -1
        if mg >= 0.81 and j not in used: cats['tp'] += 1; used.add(j)
        elif mg >= 0.81: cats['dup_of_matched'] += 1
        elif mg >= 0.5: cats['loc_miss(0.5-0.81 to new)'] += 1
        elif Ml.shape[1] and Ml[p].max() >= 0.5: cats['logged_overlap>=0.5'] += 1
        else: cats['no_gt'] += 1
print('selected', tot_sel, cats, 'targets', sum(len(g) for g in hn))
# score distributions: how many GTs with a >=0.81 refined candidate have that candidate's score below thr
lost = 0; have = 0
for (bx, sc, rb, q), l, g in zip(cand, hl, hn):
    if len(g) == 0 or len(rb) == 0: continue
    M = S.iou_matrix(rb, g)
    for j in range(len(g)):
        ok = M[:, j] >= 0.81
        if ok.any():
            have += 1
            if sc[ok].max() < 0.725: lost += 1
print('GT with a >=0.81 refined candidate', have, 'of which best-score below thr', lost)
