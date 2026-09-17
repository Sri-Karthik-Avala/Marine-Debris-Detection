import sys; sys.path.insert(0, 'dev')
import time, numpy as np, pandas as pd, torch
from pathlib import Path
import solution_v3 as S
tr = pd.read_csv('train.csv').iloc[:48]
imgs = S.load_images(Path('.'), tr.image.tolist())
b = [S.clean_boxes(np.concatenate([S.parse_boxes(a), S.parse_boxes(c)])) for a, c in zip(tr.logged_boxes, tr.new_boxes)]
for bs in (512, 128, 512, 128):
    torch.manual_seed(0); m = S.build_model(); m.roi_heads.fg_bg_sampler.batch_size_per_image = bs
    S.EPOCHS = 1
    t = time.time(); S.train_model(m, imgs[:40], b[:40], np.random.default_rng(0)); dt = (time.time()-t)/10
    m.eval(); t = time.time(); S.predict(m, imgs[40:48]); it = (time.time()-t)/8
    m.rpn._post_nms_top_n['testing'] = 300; t = time.time(); S.predict(m, imgs[40:48]); it3 = (time.time()-t)/8
    print(f'box_batch {bs}: train s/step {dt:.2f}  infer s/img (600 props) {it:.3f}  (300 props) {it3:.3f}', flush=True)
