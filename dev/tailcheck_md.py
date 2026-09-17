import sys; sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd, json
import solution as S
train = pd.read_csv('train.csv'); D = pickle.load(open('dev/ho_dets_v1.pkl', 'rb')); ho = D['ho_idx']
cand = pickle.load(open('dev/cand_det3ep_1000.pkl', 'rb'))
hl = [S.parse_boxes(train.logged_boxes[i]) for i in ho]; hn = [S.parse_boxes(train.new_boxes[i]) for i in ho]; T = sum(len(g) for g in hn)
for use_ref in (0, 1):
    vs = [S.variant(c, use_ref) for c in cand]
    tp = p = 0
    for (fb, fs), l, g in zip(vs, hl, hn):
        pr = S.select(fb, fs, l, 0.3, 0.725, 1); p += len(pr); tp += S.match_count(pr, g)
    print(use_ref, 2*tp/(p+T))
pr = np.clip(pr, 0, S.RES); print(json.dumps([[round(float(v), 4) for v in b] for b in pr]))
