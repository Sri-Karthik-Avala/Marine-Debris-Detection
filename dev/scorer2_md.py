import sys; sys.path.insert(0, 'dev')
from h_md import *
steps = int(sys.argv[1]); tag = sys.argv[2]; oof_tag = sys.argv[3]
IM = imgs(); G = imgs_gpu(); mean = S.MEAN.cuda(); std = S.STD.cuda()
OOF = pickle.load(open(f'dev/cands_{oof_tag}.pkl', 'rb'))
def crops_gpu(i, b, photo=None):
    t = G[i].permute(2, 0, 1).float().div(255)
    if photo is not None:
        c, br, gm = photo; t = (t.clamp(1e-4, 1).pow(gm)*c + br).clamp(0, 1)
    cx = (b[:, 0]+b[:, 2])/2; cy = (b[:, 1]+b[:, 3])/2; w = np.maximum(b[:, 2]-b[:, 0], 4)*S.REF_EXP; h = np.maximum(b[:, 3]-b[:, 1], 4)*S.REF_EXP
    rois = torch.from_numpy(np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1)).float().cuda()
    return (S.roi_align(t.unsqueeze(0), [rois], output_size=(S.REF_R, S.REF_R), spatial_scale=1.0, sampling_ratio=2, aligned=True)-mean)/std
S2 = {}
for k in range(3):
    tri = np.flatnonzero(FOLD != k); tei = np.flatnonzero(FOLD == k)
    det = S.build_model(); det.load_state_dict(torch.load(f'dev/oof_det_base_f{k}.pt')); det.eval()
    dets_in = dict(zip(tri, S.predict(det, IM[tri]))); del det; torch.cuda.empty_cache()
    pool = []
    for i in tri:
        bx, sc = dets_in[i]; m = sc >= S.REF_MIN_SCORE; bx = bx[m]
        if len(bx) == 0: continue
        pool.append((i, bx))
    torch.manual_seed(k); rng = np.random.default_rng(k)
    net = S.BoxRefiner().cuda(); net.head[-1] = nn.Linear(512, 1).cuda(); net.train()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    for st in range(steps):
        cur = 1e-3*min(1, (st+1)/100)*0.5*(1+math.cos(math.pi*st/steps))
        for pg in opt.param_groups: pg['lr'] = cur
        X, Y = [], []
        for pj in rng.integers(len(pool), size=16):
            i, bx = pool[pj]; b = bx[rng.integers(len(bx), size=8)].astype(np.float64)
            wh = np.maximum(b[:, 2:]-b[:, :2], 4); b = b + rng.normal(0, 0.05, b.shape)*np.concatenate([wh, wh], 1)
            it = S.iou_matrix(b, ALLB[i]).max(1)
            x = crops_gpu(i, b, (rng.uniform(0.8, 1.2), rng.uniform(-0.08, 0.08), rng.uniform(0.8, 1.25)))
            X.append(S.dihedral_crops(x, int(rng.integers(8)))); Y.append(it >= 0.5)
        X = torch.cat(X); Yt = torch.from_numpy(np.concatenate(Y)).float().cuda()
        l = F.binary_cross_entropy_with_logits(net(X)[:, 0], Yt); opt.zero_grad(); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        for i in tei:
            bx, sc, rb, q = OOF[i]
            if len(rb) == 0: S2[i] = np.zeros((0, 2)); continue
            x = crops_gpu(i, rb); o = torch.stack([torch.sigmoid(net(S.dihedral_crops(x, kk))) for kk in (0, 3, 6, 5)]).mean(0)[:, 0].cpu().numpy()
            S2[i] = np.stack([o, o], 1)
    torch.save(net.state_dict(), f'dev/scorer_{tag}_f{k}.pt'); del net; torch.cuda.empty_cache()
pickle.dump(S2, open(f'dev/scorer_{tag}.pkl', 'wb')); print('saved')
