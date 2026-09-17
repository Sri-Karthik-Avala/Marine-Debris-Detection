import sys; sys.path.insert(0, '.')
import time, numpy as np, pandas as pd, torch
from pathlib import Path
import solution as S
train = pd.read_csv('train.csv').iloc[:200]
imgs = S.load_images(Path('.'), train.image.tolist())
b = [S.clean_boxes(np.concatenate([S.parse_boxes(a), S.parse_boxes(c)])) for a, c in zip(train.logged_boxes, train.new_boxes)]
S.REF_STEPS = 50
t = time.time()
net = S.train_refiner(imgs, b, np.random.default_rng(0))
print('v3 refiner CPU s/step incl sampling (50 steps incl warm start):', (time.time() - t) / 50)
print('trainable params', sum(p.numel() for p in net.parameters() if p.requires_grad), 'total', sum(p.numel() for p in net.parameters()))
