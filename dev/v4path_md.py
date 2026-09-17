import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, torch, json
from pathlib import Path
torch.set_num_threads(2)
import solution as S
S.REF_STEPS = 3; S.SCO_STEPS = 3
tr = pd.read_csv('train.csv').iloc[:12]
imgs = S.load_images(Path('.'), tr.image.tolist())
allb = [S.clean_boxes(np.concatenate([S.parse_boxes(a), S.parse_boxes(c)])) for a, c in zip(tr.logged_boxes, tr.new_boxes)]
m = S.build_model(); m.load_state_dict(torch.load('dev/oof_det_base_f0.pt')); dets = S.predict(m, imgs)
rng = np.random.default_rng(0)
ref = S.train_refiner(imgs, allb, rng)
pool = [(j, bx[sc >= S.REF_MIN_SCORE]) for j, (bx, sc) in enumerate(dets) if (sc >= S.REF_MIN_SCORE).any()]
sco = S.train_scorer(imgs, pool, allb, rng)
c = [S.candidates(ref, sco, imgs[i], d) for i, d in enumerate(dets)]
for use_ref in (0, 1):
    for pw in S.SCORE_POWERS:
        vs = [S.variant(x, use_ref, pw) for x in c]
        pr = [S.select(fb, fs, S.parse_boxes(tr.logged_boxes[i]), 0.3, 0.3, 1) for i, (fb, fs) in enumerate(vs)]
        print(use_ref, pw, sum(len(p) for p in pr), sum(S.match_count(p, S.parse_boxes(tr.new_boxes[i])) for i, p in enumerate(pr)))
x = torch.randn(3, 3, 96, 96); o = ref(x); print('refiner out', o.shape, o[:, :4].abs().max().item() <= 1.0, 'scorer', sco(x).shape, c[0][3][:3])
