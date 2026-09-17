import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from PIL import Image
tr = pd.read_csv('train.csv'); te = pd.read_csv('test.csv')
L = tr.logged_boxes.apply(json.loads); N = tr.new_boxes.apply(json.loads)
feats, y = [], []
suffix = {'x': 0, 'y': 0, 'area': 0, 'json': 0}; nboth = 0
frac = []
for l, n in zip(L, N):
    allb = np.array(l + n, float); lab = np.array([0]*len(l) + [1]*len(n))
    frac.append(len(n)/len(allb))
    w = allb[:, 2]-allb[:, 0]; h = allb[:, 3]-allb[:, 1]; cx = (allb[:, 0]+allb[:, 2])/2; cy = (allb[:, 1]+allb[:, 3])/2
    edge = ((allb[:, 0] <= 0.5) | (allb[:, 1] <= 0.5) | (allb[:, 2] >= 639.5) | (allb[:, 3] >= 639.5)).astype(float)
    k = len(allb)
    if k > 1:
        d = np.sqrt((cx[:, None]-cx[None])**2 + (cy[:, None]-cy[None])**2); np.fill_diagonal(d, 1e9); nn = d.min(1)
    else: nn = np.full(k, 1e3)
    arank = np.argsort(np.argsort(w*h))/max(k-1, 1)
    feats.append(np.stack([w*h, w/h, cx, cy, edge, nn, arank, np.full(k, k)], 1)); y.append(lab)
    if len(l) and len(n):
        nboth += 1
        for key, order in (('x', np.argsort(cx)), ('y', np.argsort(cy)), ('area', np.argsort(w*h))):
            s = lab[order]
            if np.all(np.diff(s) >= 0) or np.all(np.diff(s) <= 0): suffix[key] += 1
F = np.concatenate(feats); Y = np.concatenate(y)
for i, nm in enumerate(['area', 'aspect', 'cx', 'cy', 'edge', 'nn_dist', 'area_rank_in_img', 'n_boxes_img']):
    print(f'new-vs-logged AUC {nm}: {roc_auc_score(Y, F[:, i]):.3f}')
print('rows with both', nboth, 'contiguous split by order:', suffix)
frac = np.array(frac); tot = (L.apply(len)+N.apply(len)).values
for lo, hi in ((1, 1), (2, 3), (4, 7), (8, 100)):
    m = (tot >= lo) & (tot <= hi); print(f'total {lo}-{hi}: rows {m.sum()} mean new frac {frac[m].mean():.3f} mean new {N.apply(len).values[m].mean():.2f}')
# ledger len vs new count
nl = L.apply(len).values; nn_ = N.apply(len).values
for a in range(0, 8): 
    m = nl == a
    if m.sum(): print('ledger', a, 'rows', m.sum(), 'new mean', nn_[m].mean().round(2), 'new==1 frac', (nn_[m] == 1).mean().round(2))
# image stats train vs test
def stats(p):
    a = np.asarray(Image.open(p).convert('RGB')).astype(np.float64); g = a.mean(2)
    lap = (g[1:-1, 1:-1]*4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    F2 = np.abs(np.fft.rfft2(g - g.mean())); fy = np.fft.fftfreq(640)[:, None]; fx = np.fft.rfftfreq(640)[None]; r = np.sqrt(fy**2+fx**2)
    hi = F2[(r > 0.45*576/640) & (r < 0.5)].mean() / F2[(r > 0.3) & (r < 0.4)].mean()
    im = Image.open(p); q = im.quantization[0][:8] if hasattr(im, 'quantization') and im.quantization else None
    return [a.mean(), a.std(), lap.var(), (a.max(2) < 8).mean(), a[..., 0].mean()-a[..., 2].mean(), hi, q[0] if q else -1, sum(q) if q else -1]
rng = np.random.default_rng(0)
S1 = np.array([stats(p) for p in tr.image.sample(150, random_state=0)]); S2 = np.array([stats(p) for p in te.image.sample(150, random_state=0)])
for i, nm in enumerate(['mean', 'std', 'lapvar', 'black', 'R-B', 'hf_ratio(576 cutoff)', 'q0', 'qsum8']):
    print(f'{nm}: train med {np.median(S1[:, i]):.4g} [{np.percentile(S1[:, i], 10):.4g},{np.percentile(S1[:, i], 90):.4g}]  test med {np.median(S2[:, i]):.4g} [{np.percentile(S2[:, i], 10):.4g},{np.percentile(S2[:, i], 90):.4g}]')
print('unique q0 train', np.unique(S1[:, 6])[:10], 'test', np.unique(S2[:, 6])[:10])
print('train sizes', Image.open(tr.image[0]).size, 'modes', Image.open(tr.image[0]).mode)
