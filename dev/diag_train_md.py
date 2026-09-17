import sys; sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd, torch, time
from pathlib import Path
import solution as S
S.DEVICE = torch.device('cuda')
pub = Path('.')
train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
tr_logged = [S.parse_boxes(s) for s in train.logged_boxes]; tr_new = [S.parse_boxes(s) for s in train.new_boxes]
groups = train.survey_id.unique(); grng = np.random.default_rng(S.SEED + 1); perm = grng.permutation(len(groups))
hold, cnt = set(), 0
for gi in perm:
    g = groups[gi]; sz = int((train.survey_id == g).sum())
    if sz > S.HOLD_FRAC * len(train) / 4: continue
    hold.add(g); cnt += sz
    if cnt >= S.HOLD_FRAC * len(train): break
is_hold = train.survey_id.isin(hold).to_numpy(); tr_idx = np.flatnonzero(~is_hold); ho_idx = np.flatnonzero(is_hold)
imgs = S.load_images(pub, train.image.tolist())
allb = [S.clean_boxes(np.concatenate([tr_logged[i], tr_new[i]])) for i in range(len(train))]
torch.manual_seed(0)
t0 = time.time()
m = S.build_model()
m = S.train_model(m, imgs[tr_idx], [allb[i] for i in tr_idx], np.random.default_rng(0))
print('train', time.time() - t0)
torch.save(m.state_dict(), 'dev/det_v1_gpu.pt')
d = S.predict(m, imgs[ho_idx], [0, 3, 6, 5])
pickle.dump({'ho_idx': ho_idx, 'tr_idx': tr_idx, 'dets': d}, open('dev/ho_dets_v1.pkl', 'wb'))
print('done', time.time() - t0)
