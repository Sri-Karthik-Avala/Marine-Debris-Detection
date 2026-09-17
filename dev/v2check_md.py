import sys; sys.path.insert(0, '.')
import pickle, time, numpy as np, pandas as pd, torch
from pathlib import Path
import solution as S
S.DEVICE = torch.device('cuda')
S.REF_STEPS = int(sys.argv[1])
train = pd.read_csv('train.csv')
D = pickle.load(open('dev/ho_dets_v1.pkl', 'rb')); ho = D['ho_idx']; tri = D['tr_idx']
lg = [S.parse_boxes(s) for s in train.logged_boxes]; nw = [S.parse_boxes(s) for s in train.new_boxes]
allb = [S.clean_boxes(np.concatenate([lg[i], nw[i]])) for i in range(len(train))]
imgs = S.load_images(Path('.'), train.image.tolist())
t0 = time.time()
torch.manual_seed(0)
S.EPOCHS = 2
det = S.train_model(S.build_model(), imgs[tri], [allb[i] for i in tri], np.random.default_rng(0))
dets2 = S.predict(det, imgs[ho]); del det
print('det2 time', time.time() - t0, flush=True)
dets3 = [d[0] for d in D['dets']]
t0 = time.time()
ref = S.train_refiner(imgs[tri], [allb[i] for i in tri], np.random.default_rng(1))
print('refiner time', time.time() - t0, flush=True)
torch.save(ref.state_dict(), f'dev/ref_v2_{S.REF_STEPS}.pt')
hl = [lg[i] for i in ho]; hn = [nw[i] for i in ho]; T = sum(len(g) for g in hn)
thrs = np.round(np.arange(0.05, 0.96, 0.025), 3)
for name, dets in (('det3ep', dets3), ('det2ep', dets2)):
    cand = [S.candidates(ref, imgs[i], d) for i, d in zip(ho, dets)]
    pickle.dump(cand, open(f'dev/cand_{name}_{S.REF_STEPS}.pkl', 'wb'))
    for use_ref in (0, 1):
        for sm in (0, 1):
            for pn in (1.0, 0.7, 0.5):
                if not use_ref and (sm or pn < 1): continue
                vs = [S.variant(c, use_ref, sm, pn) for c in cand]
                best = None
                for sup in (0.3, 0.5, 0.7):
                    for force in (0, 1):
                        for thr in thrs:
                            tp = p = 0
                            for (fb, fs), l, g in zip(vs, hl, hn):
                                pr = S.select(fb, fs, l, sup, thr, force); p += len(pr); tp += S.match_count(pr, g)
                            f = 2*tp/(p+T)
                            if best is None or f > best[0]: best = (f, sup, force, float(thr), tp, p)
                print(f'{name} ref{use_ref} sm{sm} nms{pn}: {best}', flush=True)
