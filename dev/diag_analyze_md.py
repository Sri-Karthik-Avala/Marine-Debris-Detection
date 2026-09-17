import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'dev')
import pickle, numpy as np, pandas as pd
import solution as S
from metric_md import group_score, tp_count, iou_m
D = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else 'dev/ho_dets_v1.pkl', 'rb'))
train = pd.read_csv('train.csv'); ho = D['ho_idx']
lg = [S.parse_boxes(train.logged_boxes[i]) for i in ho]; gt = [S.parse_boxes(train.new_boxes[i]) for i in ho]; grp = train.survey_id.values[ho]
T = sum(len(g) for g in gt)
for nv in (1, 2, 4):
    fused = [S.fuse(d, nv) for d in D['dets']]
    best = None
    for sup in (0.3, 0.5):
        for force in (0, 1):
            for thr in np.arange(0.1, 0.96, 0.025):
                pr = [S.select(fb, fs, l, sup, thr, force) for (fb, fs), l in zip(fused, lg)]
                P = sum(len(p) for p in pr); tp = sum(tp_count(p, g) for p, g in zip(pr, gt))
                f = 2*tp/(P+T)
                if best is None or f > best[0]: best = (f, sup, force, thr, pr)
    f, sup, force, thr, pr = best
    P = sum(len(p) for p in pr)
    msg = [f'nv{nv} sup{sup} force{force} thr{thr:.3f} P{P} T{T}']
    for it in (0.5, 0.7, 0.75, 0.81, 0.9):
        tp = sum(tp_count(p, g, it) for p, g in zip(pr, gt)); msg.append(f'F1@{it} {2*tp/(P+T):.3f}')
    msg.append(f'group@0.81 {group_score(pr, gt, grp):.4f}')
    print(' '.join(msg))
    ious = []
    for (fb, fs), l, g in zip(fused, lg, gt):
        if len(g) and len(fb):
            ious += list(iou_m(fb, g).max(0))
        else: ious += [0]*len(g)
    ious = np.array(ious)
    print('  best-candidate IoU per GT (any score): >=0.5 %.3f >=0.7 %.3f >=0.81 %.3f >=0.9 %.3f' % tuple((ious >= t).mean() for t in (0.5, 0.7, 0.81, 0.9)))
