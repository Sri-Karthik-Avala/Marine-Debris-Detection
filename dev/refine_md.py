import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'dev')
import pickle, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from torchvision.ops import roi_align
import solution as S
from metric_md import group_score, tp_count, iou_m
DEV = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda')
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
R = int(sys.argv[3]) if len(sys.argv) > 3 else 96
BS = 128; EXP = 2.0
torch.manual_seed(0); rng = np.random.default_rng(0)
train = pd.read_csv('train.csv')
D = pickle.load(open('dev/ho_dets_v1.pkl', 'rb')); ho = D['ho_idx']; tri = D['tr_idx']
lg = [S.parse_boxes(s) for s in train.logged_boxes]; nw = [S.parse_boxes(s) for s in train.new_boxes]
allb = [S.clean_boxes(np.concatenate([lg[i], nw[i]])) for i in range(len(train))]
imgs = S.load_images(__import__('pathlib').Path('.'), train.image.tolist())
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
sizes = np.concatenate([np.stack([b[:, 2]-b[:, 0], b[:, 3]-b[:, 1]], 1) for b in allb])

class Ref(nn.Module):
    def __init__(s):
        super().__init__()
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        s.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        k = (R // 32)
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(512*k*k, 512), nn.ReLU(), nn.Linear(512, 5))
    def forward(s, x): return s.head(s.body(x))

def regions(b):
    cx = (b[:, 0]+b[:, 2])/2; cy = (b[:, 1]+b[:, 3])/2
    w = np.maximum(b[:, 2]-b[:, 0], 4)*EXP; h = np.maximum(b[:, 3]-b[:, 1], 4)*EXP
    return np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1)

def crops(img_t, bxs):
    rois = torch.from_numpy(regions(bxs)).float()
    x = roi_align(img_t, [rois], output_size=(R, R), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    return (x - MEAN) / STD

def sample_batch(idx_pool):
    ii = rng.choice(idx_pool, 16)
    X, Y, Q, W = [], [], [], []
    for i in ii:
        g = allb[i]; n = BS // 16
        gi = rng.integers(len(g), size=n); gb = g[gi]
        w = gb[:, 2]-gb[:, 0]; h = gb[:, 3]-gb[:, 1]
        lvl = rng.choice([0.04, 0.08, 0.15, 0.3], size=n)
        cx = (gb[:, 0]+gb[:, 2])/2 + rng.normal(0, 1, n)*lvl*w; cy = (gb[:, 1]+gb[:, 3])/2 + rng.normal(0, 1, n)*lvl*h
        nw_ = w*np.exp(rng.normal(0, 1, n)*lvl*1.2); nh_ = h*np.exp(rng.normal(0, 1, n)*lvl*1.2)
        b = np.stack([cx-nw_/2, cy-nh_/2, cx+nw_/2, cy+nh_/2], 1)
        neg = rng.random(n) < 0.15
        if neg.any():
            k = int(neg.sum()); sz = sizes[rng.integers(len(sizes), size=k)]
            px = rng.uniform(0, 640-sz[:, 0]); py = rng.uniform(0, 640-sz[:, 1])
            b[neg] = np.stack([px, py, px+sz[:, 0], py+sz[:, 1]], 1)
        b = np.clip(b, -20, 660)
        M = iou_m(b, g); j = M.argmax(1); q = M.max(1); tg = g[j]
        bw = np.maximum(b[:, 2]-b[:, 0], 4); bh = np.maximum(b[:, 3]-b[:, 1], 4)
        off = np.stack([(tg[:, 0]-b[:, 0])/bw, (tg[:, 1]-b[:, 1])/bh, (tg[:, 2]-b[:, 2])/bw, (tg[:, 3]-b[:, 3])/bh], 1)
        k = int(rng.integers(8))
        a = S.dihedral_image(imgs[i], k); bb = S.dihedral_boxes(b, k)
        if k & 1: off = off[:, [1, 0, 3, 2]]
        if k & 2: off = np.stack([-off[:, 2], off[:, 1], -off[:, 0], off[:, 3]], 1)
        if k & 4: off = np.stack([off[:, 0], -off[:, 3], off[:, 2], -off[:, 1]], 1)
        it = S.to_tensor(a).unsqueeze(0)
        X.append(crops(it, bb.astype(np.float32))); Y.append(off); Q.append(q)
    return torch.cat(X), torch.from_numpy(np.concatenate(Y)).float(), torch.from_numpy(np.concatenate(Q)).float()

net = Ref().to(DEV)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
t0 = time.time()
for st in range(STEPS):
    lr = 1e-3*min(1, (st+1)/100)*0.5*(1+np.cos(np.pi*st/STEPS))
    for pg in opt.param_groups: pg['lr'] = lr
    X, Y, Q = sample_batch(tri)
    X, Y, Q = X.to(DEV), Y.to(DEV), Q.to(DEV)
    out = net(X)
    m = (Q > 0.3).float()
    lreg = ((out[:, :4]-Y.clamp(-1, 1)).abs().sum(1)*m).sum()/m.sum().clamp(min=1)
    lq = F.binary_cross_entropy_with_logits(out[:, 4], Q)
    loss = lreg + lq
    opt.zero_grad(); loss.backward(); opt.step()
    if st % 200 == 0: print(st, f'{lreg.item():.4f} {lq.item():.4f} {time.time()-t0:.0f}s', flush=True)
print('train time', time.time()-t0)
torch.save(net.state_dict(), f'dev/ref_R{R}.pt')

@torch.no_grad()
def refine(i, b, iters=2):
    net.eval(); b = b.astype(np.float64).copy(); q = np.zeros(len(b))
    if len(b) == 0: return b, q
    for _ in range(iters):
        acc = np.zeros((len(b), 4)); qs = np.zeros(len(b))
        for k in (0, 3, 6, 5):
            a = S.dihedral_image(imgs[i], k); bb = S.dihedral_boxes(b, k)
            out = net(crops(S.to_tensor(a).unsqueeze(0), bb.astype(np.float32)).to(DEV)).cpu().numpy()
            off = out[:, :4]
            if k & 4: off = np.stack([off[:, 0], -off[:, 3], off[:, 2], -off[:, 1]], 1)
            if k & 2: off = np.stack([-off[:, 2], off[:, 1], -off[:, 0], off[:, 3]], 1)
            if k & 1: off = off[:, [1, 0, 3, 2]]
            acc += off/4; qs += 1/(1+np.exp(-out[:, 4]))/4
        bw = np.maximum(b[:, 2]-b[:, 0], 4); bh = np.maximum(b[:, 3]-b[:, 1], 4)
        b = b + acc*np.stack([bw, bh, bw, bh], 1); q = qs
    return np.clip(b, 0, 640), q

# synthetic check on holdout GT
bef, aft = [], []
for i in ho[:80]:
    g = allb[i]; n = len(g)
    b = g + rng.normal(0, 1, (n, 4))*0.08*np.stack([g[:, 2]-g[:, 0], g[:, 3]-g[:, 1]]*2, 1)
    rb, _ = refine(i, b)
    bef += list(np.diag(iou_m(b, g))); aft += list(np.diag(iou_m(rb, g)))
bef, aft = np.array(bef), np.array(aft)
print('synthetic IoU>=0.81 before %.3f after %.3f ; mean IoU %.3f -> %.3f' % ((bef >= 0.81).mean(), (aft >= 0.81).mean(), bef.mean(), aft.mean()))

lgh = [lg[i] for i in ho]; gth = [nw[i] for i in ho]; grp = train.survey_id.values[ho]; T = sum(len(g) for g in gth)
res = {}
for nv in (1, 4):
    fused = [S.fuse(d, nv) for d in D['dets']]
    ref = []
    for (fb, fs), i in zip(fused, ho):
        m = fs >= 0.1
        rb, q = refine(i, fb[m]); ref.append((fb[m], fs[m], rb, q))
    for mode in ('raw', 'ref', 'ref_q', 'ref_sq'):
        best = None
        for sup in (0.3, 0.5):
            for thr in np.arange(0.1, 0.96, 0.025):
                pr = []
                for (fb, fs, rb, q), l in zip(ref, lgh):
                    bx = fb if mode == 'raw' else rb
                    sc = fs if mode in ('raw', 'ref') else (q if mode == 'ref_q' else np.sqrt(fs*q))
                    pr.append(S.select(bx, sc, l, sup, thr, 1))
                P = sum(len(p) for p in pr); tp = sum(tp_count(p, g) for p, g in zip(pr, gth)); f = 2*tp/(P+T)
                if best is None or f > best[0]: best = (f, sup, thr, P, tp, pr)
        print(f'nv{nv} {mode}: F1@0.81 {best[0]:.4f} sup {best[1]} thr {best[2]:.3f} P {best[3]} tp {best[4]} group {group_score(best[5], gth, grp):.4f}', flush=True)
