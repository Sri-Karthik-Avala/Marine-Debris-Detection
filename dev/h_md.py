import sys; sys.path.insert(0, '.')
import pickle, math, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'dev'); import solution_v3 as S
S.DEVICE = torch.device('cuda')
train = pd.read_csv('train.csv')
LG = [S.parse_boxes(s) for s in train.logged_boxes]; NW = [S.parse_boxes(s) for s in train.new_boxes]
ALLB = [S.clean_boxes(np.concatenate([LG[i], NW[i]])) for i in range(len(train))]
GRP = train.survey_id.values
def folds(k=3, seed=0):
    ug = np.unique(GRP); rng = np.random.default_rng(seed); perm = rng.permutation(ug)
    sizes = {g: (GRP == g).sum() for g in ug}; load = np.zeros(k); fmap = {}
    for g in sorted(perm, key=lambda g: -sizes[g]):
        j = int(np.argmin(load)); fmap[g] = j; load[j] += sizes[g]
    return np.array([fmap[g] for g in GRP])
FOLD = folds()
_IM = None
def imgs():
    global _IM
    if _IM is None: _IM = S.load_images(Path('.'), train.image.tolist())
    return _IM
_IMG_GPU = None
def imgs_gpu():
    global _IMG_GPU
    if _IMG_GPU is None: _IMG_GPU = torch.from_numpy(imgs()).cuda()
    return _IMG_GPU
def train_refiner_gpu(idx, steps, seed=0, lr=None, exp=None, R=None, lvls=(0.04, 0.08, 0.15, 0.3), neg=None, loss='l1', freeze=True, arch='r18', imgs_per=16, per_img=8):
    if exp is not None: S.REF_EXP = exp
    if R is not None: S.REF_R = R
    lr = lr or S.REF_LR; neg = S.REF_NEG if neg is None else neg
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = make_net(arch, freeze).cuda(); net.train()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    G = imgs_gpu(); bl = [ALLB[i] for i in idx]
    sizes = np.concatenate([np.stack([b[:, 2]-b[:, 0], b[:, 3]-b[:, 1]], 1) for b in bl])
    mean = S.MEAN.cuda(); std = S.STD.cuda()
    for st in range(steps):
        cur = lr*min(1.0, (st+1)/S.WARM)*0.5*(1+math.cos(math.pi*st/steps))
        for pg in opt.param_groups: pg['lr'] = cur
        X, Y, Q = [], [], []
        for ii in rng.integers(len(idx), size=imgs_per):
            i = idx[ii]; g = ALLB[i]; n = per_img
            gb = g[rng.integers(len(g), size=n)].astype(np.float64); w = gb[:, 2]-gb[:, 0]; h = gb[:, 3]-gb[:, 1]
            lvl = rng.choice(lvls, size=n)
            cx = (gb[:, 0]+gb[:, 2])/2 + rng.normal(0, 1, n)*lvl*w; cy = (gb[:, 1]+gb[:, 3])/2 + rng.normal(0, 1, n)*lvl*h
            nw = w*np.exp(rng.normal(0, 1, n)*lvl*1.2); nh = h*np.exp(rng.normal(0, 1, n)*lvl*1.2)
            b = np.stack([cx-nw/2, cy-nh/2, cx+nw/2, cy+nh/2], 1)
            ng = rng.random(n) < neg
            if ng.any():
                sz = sizes[rng.integers(len(sizes), size=int(ng.sum()))]; px = rng.uniform(0, 640-sz[:, 0]); py = rng.uniform(0, 640-sz[:, 1])
                b[ng] = np.stack([px, py, px+sz[:, 0], py+sz[:, 1]], 1)
            b = np.clip(b, -20, 660); M = S.iou_matrix(b, g); q = M.max(1); off = S.box_offsets(b, g[M.argmax(1)].astype(np.float64))
            k = int(rng.integers(8))
            t = G[i].permute(2, 0, 1).float().div(255)
            c = float(rng.uniform(0.8, 1.2)); br = float(rng.uniform(-0.08, 0.08)); gm = float(rng.uniform(0.8, 1.25))
            t = (t.clamp(1e-4, 1).pow(gm)*c + br).clamp(0, 1)
            cxr = (b[:, 0]+b[:, 2])/2; cyr = (b[:, 1]+b[:, 3])/2; wr = np.maximum(b[:, 2]-b[:, 0], 4)*S.REF_EXP; hr = np.maximum(b[:, 3]-b[:, 1], 4)*S.REF_EXP
            rois = torch.from_numpy(np.stack([cxr-wr/2, cyr-hr/2, cxr+wr/2, cyr+hr/2], 1)).float().cuda()
            x = S.roi_align(t.unsqueeze(0), [rois], output_size=(S.REF_R, S.REF_R), spatial_scale=1.0, sampling_ratio=2, aligned=True)
            X.append(S.dihedral_crops((x-mean)/std, k)); Y.append(S.dihedral_offsets(off, k)); Q.append(q)
        X = torch.cat(X); Yt = torch.from_numpy(np.concatenate(Y)).float().clamp(-1, 1).cuda(); Qt = torch.from_numpy(np.concatenate(Q)).float().cuda()
        m = (Qt > 0.3).float()
        if arch == 'bins':
            lgb, qb = net.raw(X); lreg = bin_loss(lgb, Yt, m); out = torch.cat([torch.zeros_like(qb)[:, None].repeat(1, 4), qb[:, None]], 1); loss = 'bins'
        else:
            out = net(X)
        if loss == 'bins': pass
        elif loss == 'l1': lreg = ((out[:, :4]-Yt).abs().sum(1)*m).sum()/m.sum().clamp(min=1)
        elif loss == 'sl1': lreg = (F.smooth_l1_loss(out[:, :4], Yt, reduction='none', beta=0.05).sum(1)*m).sum()/m.sum().clamp(min=1)
        lq = F.binary_cross_entropy_with_logits(out[:, 4], Qt)
        l = lreg + lq; opt.zero_grad(); l.backward(); opt.step()
    net.eval(); return net
def make_net(arch, freeze):
    import torchvision, torch.nn as nn
    if arch == 'r18':
        net = S.BoxRefiner()
        if not freeze:
            for p in net.parameters(): p.requires_grad = True
        return net
    if arch == 'bins':
        return BinRef(freeze)
    if arch == 'r34':
        m = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        net = S.BoxRefiner(); net.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        if freeze:
            for mod in (m.conv1, m.bn1, m.layer1):
                for p in mod.parameters(): p.requires_grad = False
        return net
def cands(net, idx, dets, min_score=None):
    ms = S.REF_MIN_SCORE if min_score is None else min_score
    IM = imgs(); out = {}
    for i in idx:
        bx, sc = dets[i]; m = sc >= ms
        rb, q = S.refine_boxes(net, IM[i], bx[m]); out[i] = (bx[m], sc[m], rb, q)
    return out
THRS = np.round(np.arange(0.3, 0.96, 0.025), 3)
def evaluate(C, idx=None, sups=(0.3,), forces=(1,), use_ref=True, verbose=True):
    idx = sorted(C) if idx is None else idx
    T = sum(len(NW[i]) for i in idx); best = None
    for sup in sups:
        for force in forces:
            for thr in THRS:
                tp = p = 0
                for i in idx:
                    bx, sc, rb, q = C[i]
                    pr = S.select(rb if use_ref else bx, sc, LG[i], sup, thr, force); p += len(pr); tp += S.match_count(pr, NW[i])
                f = 2*tp/(p+T)
                if best is None or f > best[0]: best = (round(f, 4), sup, force, float(thr), tp, p, T)
    return best
import torch.nn as nn
NB = 41
BINS = torch.linspace(-1, 1, NB)
class BinRef(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        base = S.BoxRefiner()
        self.body = base.body
        k = S.REF_R // 32
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512*k*k, 512), nn.ReLU(), nn.Linear(512, 4*NB+1))
        if not freeze:
            for p in self.body.parameters(): p.requires_grad = True
    def raw(self, x):
        o = self.head(self.body(x)); return o[:, :4*NB].view(-1, 4, NB), o[:, 4*NB]
    def forward(self, x):
        lg, q = self.raw(x); p = lg.softmax(-1)
        return torch.cat([(p*BINS.to(x.device)).sum(-1), q[:, None]], 1)
def bin_loss(lg, y, m):
    pos = (y.clamp(-1, 1)+1)/2*(NB-1); lo = pos.floor().clamp(0, NB-2); wl = 1-(pos-lo); lo = lo.long()
    logp = lg.log_softmax(-1)
    l = -(logp.gather(-1, lo[..., None]).squeeze(-1)*wl + logp.gather(-1, (lo+1)[..., None]).squeeze(-1)*(1-wl)).sum(1)
    return (l*m).sum()/m.sum().clamp(min=1)
