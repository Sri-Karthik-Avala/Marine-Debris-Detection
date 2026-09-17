import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import solution as S
train = pd.read_csv('train.csv')
lg = [S.parse_boxes(s) for s in train.logged_boxes]; nw = [S.parse_boxes(s) for s in train.new_boxes]
groups = train.survey_id.unique(); grng = np.random.default_rng(S.SEED + 1); perm = grng.permutation(len(groups))
hold, cnt = set(), 0
for gi in perm:
    g = groups[gi]; sz = int((train.survey_id == g).sum())
    if sz > S.HOLD_FRAC * len(train) / 4: continue
    hold.add(g); cnt += sz
    if cnt >= S.HOLD_FRAC * len(train): break
tr_idx = np.flatnonzero(~train.survey_id.isin(hold).to_numpy())
b = np.concatenate([S.clean_boxes(np.concatenate([lg[i], nw[i]])) for i in tr_idx])
side = np.sqrt((b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1]))
p5 = np.percentile(side, 5); base = int(2 ** np.floor(np.log2(p5)))
print('p5 side', round(p5, 2), 'anchor base', base, 'anchors', tuple(base * 2 ** k for k in range(5)))
