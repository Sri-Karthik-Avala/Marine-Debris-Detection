import sys; sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd
from PIL import Image, ImageDraw
import solution as S
train = pd.read_csv('train.csv')
LG = [S.parse_boxes(s) for s in train.logged_boxes]; NW = [S.parse_boxes(s) for s in train.new_boxes]
C = pickle.load(open('dev/cands_r_base_s0.pkl', 'rb'))
rng = np.random.default_rng(3)
fps, nears = [], []
for i in sorted(C):
    p = S.select(C[i][2], C[i][1], LG[i], 0.3, 0.675, 1)
    if not len(p): continue
    allg = np.concatenate([LG[i], NW[i]])
    Mg = S.iou_matrix(p, NW[i]); Ma = S.iou_matrix(p, allg)
    for k in range(len(p)):
        if Ma[k].max() < 0.1: fps.append((i, p[k], None))
        elif 0.5 <= Mg[k].max() < 0.81: nears.append((i, p[k], NW[i][Mg[k].argmax()], Mg[k].max()))
def tile(items, fn, near=False):
    sel = [items[j] for j in rng.choice(len(items), 30, replace=False)]
    canvas = Image.new('RGB', (6*160, 5*160), 'black')
    for t, it in enumerate(sel):
        i, b = it[0], it[1]; im = Image.open(train.image[i]).convert('RGB'); d = ImageDraw.Draw(im)
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2; s = max(b[2]-b[0], b[3]-b[1])*1.6+20
        for g in np.concatenate([LG[i], NW[i]]): d.rectangle(list(g), outline=(0, 255, 0))
        d.rectangle(list(b), outline=(255, 0, 0))
        crop = im.crop((cx-s, cy-s, cx+s, cy+s)).resize((160, 160))
        if near: ImageDraw.Draw(crop).text((2, 2), f'{it[3]:.2f}', fill=(255, 255, 0))
        canvas.paste(crop, ((t % 6)*160, (t//6)*160))
    canvas.save(fn)
tile(fps, 'dev/viz_fp.png'); tile(nears, 'dev/viz_near.png', True)
print(len(fps), len(nears))
