import numpy as np, pandas as pd, json
from PIL import Image
tr = pd.read_csv('train.csv'); te = pd.read_csv('test.csv')
def st(p):
    a = np.asarray(Image.open(p).convert('RGB')); blk = a.max(2) <= 10
    rows = blk.all(1); cols = blk.all(0)
    return blk.mean(), rows.mean(), cols.mean(), rows[:5].all() or rows[-5:].all() or cols[:5].all() or cols[-5:].all()
for nm, d in (('train', tr), ('test', te)):
    S = np.array([st(p) for p in d.image])
    print(nm, 'frac imgs black>1%', (S[:, 0] > 0.01).mean().round(3), '>10%', (S[:, 0] > 0.1).mean().round(3), 'mean black', S[:, 0].mean().round(4), 'full black rows/cols band at border', S[:, 3].mean().round(3), 'p90 black', np.percentile(S[:, 0], 90).round(3))
    if nm == 'train': Str = S
# boxes adjacent to black: fraction of GT boxes whose expanded region contains >20% black, and model hit rates later
blk_near = []
for p, l, n in zip(tr.image, tr.logged_boxes, tr.new_boxes):
    a = np.asarray(Image.open(p).convert('RGB')).max(2) <= 10
    for b in json.loads(n):
        x0, y0, x1, y1 = [int(round(v)) for v in b]; w = x1-x0; h = y1-y0
        r = a[max(0, y0-h//2):y1+h//2, max(0, x0-w//2):x1+w//2]
        blk_near.append(r.mean() if r.size else 0)
blk_near = np.array(blk_near); print('new GT boxes with >5% black in 2x region', (blk_near > 0.05).mean().round(3))
