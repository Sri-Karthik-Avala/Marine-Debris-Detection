import sys; sys.path.insert(0, '.')
import pickle, time, numpy as np, pandas as pd, torch
from pathlib import Path
import solution as S
S.DEVICE = torch.device('cuda')
steps = sys.argv[1]
train = pd.read_csv('train.csv'); D = pickle.load(open('dev/ho_dets_v1.pkl', 'rb')); ho = D['ho_idx']
hl = [S.parse_boxes(train.logged_boxes[i]) for i in ho]; hn = [S.parse_boxes(train.new_boxes[i]) for i in ho]; T = sum(len(g) for g in hn)
imgs = S.load_images(Path('.'), [train.image[i] for i in ho])
ref = S.BoxRefiner().to(S.DEVICE); ref.load_state_dict(torch.load(f'dev/ref_v2_{steps}.pt')); ref.eval()
sys.path.insert(0, 'dev'); import importlib
v1 = importlib.import_module('solution_v1')
thrs = np.round(np.arange(0.05, 0.96, 0.025), 3)
def grid(vs):
    best = None
    for sup in (0.3, 0.5):
        for force in (0, 1):
            for thr in thrs:
                tp = p = 0
                for (fb, fs), l, g in zip(vs, hl, hn):
                    pr = S.select(fb, fs, l, sup, thr, force); p += len(pr); tp += S.match_count(pr, g)
                f = 2*tp/(p+T)
                if best is None or f > best[0]: best = (round(f, 4), sup, force, float(thr), tp, p)
    return best
for nv in (1, 4):
    dets = [v1.fuse(d, nv) for d in D['dets']]
    for iters, var in ((2, (0, 3, 6, 5)), (3, (0, 3, 6, 5)), (2, tuple(range(8))), (1, (0, 3, 6, 5))):
        S.REF_ITERS = iters; S.REF_VARIANTS = var
        cand = [S.candidates(ref, imgs[j], d) for j, d in enumerate(dets)]
        print(f'steps{steps} nv{nv} iters{iters} nvar{len(var)}: ref {grid([S.variant(c, 1, 0, 1.0) for c in cand])} sq {grid([S.variant(c, 1, 1, 1.0) for c in cand])}', flush=True)
