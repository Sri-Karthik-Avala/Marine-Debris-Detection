import sys; sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd, torch
from pathlib import Path
import solution as S
S.DEVICE = torch.device('cuda')
train = pd.read_csv('train.csv'); D = pickle.load(open('dev/ho_dets_v1.pkl', 'rb')); ho = D['ho_idx']
imgs = S.load_images(Path('.'), [train.image[i] for i in ho])
ref = S.BoxRefiner().to(S.DEVICE); ref.load_state_dict(torch.load('dev/ref_v2_1000.pt')); ref.eval()
@torch.no_grad()
def refine_stats(img, b):
    b = b.astype(np.float64).copy(); n = len(b); stats = []
    img_t = S.to_tensor(img)
    for it in range(2):
        x = S.crop_regions(img_t, b)
        out = ref(torch.cat([S.dihedral_crops(x, k) for k in S.REF_VARIANTS]).to(S.DEVICE)).cpu().numpy().astype(np.float64)
        offs = np.stack([S.undo_dihedral_offsets(out[vi*n:(vi+1)*n, :4], k) for vi, k in enumerate(S.REF_VARIANTS)])
        qs = (1/(1+np.exp(-out[:, 4]))).reshape(4, n)
        acc = offs.mean(0)
        stats.append((offs.std(0).mean(1), np.abs(acc).mean(1), qs.mean(0), qs.std(0)))
        bw = np.maximum(b[:, 2]-b[:, 0], 4); bh = np.maximum(b[:, 3]-b[:, 1], 4)
        b = b + acc*np.stack([bw, bh, bw, bh], 1)
    return np.clip(b, 0, 640), stats
rows = []
for j, (d, i) in enumerate(zip(D['dets'], ho)):
    bx, sc = d[0]; m = sc >= 0.1; bx, sc = bx[m], sc[m]
    if len(bx) == 0: rows.append(None); continue
    rb, st = refine_stats(imgs[j], bx)
    rows.append(dict(bx=bx, sc=sc, rb=rb, std1=st[0][0], mv1=st[0][1], q1=st[0][2], qstd1=st[0][3], std2=st[1][0], mv2=st[1][1], q2=st[1][2], qstd2=st[1][3]))
pickle.dump(rows, open('dev/feat_ho.pkl', 'wb')); print('saved')
