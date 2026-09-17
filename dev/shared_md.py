import sys, json; sys.path.insert(0, 'dev')
from h_md import *
tag = sys.argv[1]; cfg = json.loads(sys.argv[2])
steps = cfg['steps']; real_imgs = cfg.get('real_imgs', 5); real_reg = cfg.get('real_reg', True); seed = cfg.get('seed', 0)
dets_oof = pickle.load(open('dev/oof_dets_base.pkl', 'rb'))
IM = imgs(); G = imgs_gpu(); mean = S.MEAN.cuda(); std = S.STD.cuda()
class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        base = S.BoxRefiner(); self.body = base.body; k = S.REF_R // 32
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512*k*k, 512), nn.ReLU(), nn.Linear(512, 4*NB+2))
    def raw(self, x):
        o = self.head(self.body(x)); return o[:, :4*NB].view(-1, 4, NB), o[:, 4*NB], o[:, 4*NB+1]
    def forward(self, x):
        lg, q, ob = self.raw(x); p = lg.softmax(-1)
        return torch.cat([(p*BINS.to(x.device)).sum(-1), q[:, None]], 1)
def crops(i, b, photo):
    t = G[i].permute(2, 0, 1).float().div(255)
    if photo is not None:
        c, br, gm = photo; t = (t.clamp(1e-4, 1).pow(gm)*c + br).clamp(0, 1)
    cx = (b[:, 0]+b[:, 2])/2; cy = (b[:, 1]+b[:, 3])/2; w = np.maximum(b[:, 2]-b[:, 0], 4)*S.REF_EXP; h = np.maximum(b[:, 3]-b[:, 1], 4)*S.REF_EXP
    rois = torch.from_numpy(np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1)).float().cuda()
    return (S.roi_align(t.unsqueeze(0), [rois], output_size=(S.REF_R, S.REF_R), spatial_scale=1.0, sampling_ratio=2, aligned=True)-mean)/std
C, S2 = {}, {}
t0 = time.time()
for k in range(3):
    tri = np.flatnonzero(FOLD != k); tei = np.flatnonzero(FOLD == k)
    det = S.build_model(); det.load_state_dict(torch.load(f'dev/oof_det_base_f{k}.pt')); det.eval()
    dets_in = dict(zip(tri, S.predict(det, IM[tri]))); del det; torch.cuda.empty_cache()
    pool = [(i, dets_in[i][0][dets_in[i][1] >= S.REF_MIN_SCORE]) for i in tri if (dets_in[i][1] >= S.REF_MIN_SCORE).any()]
    torch.manual_seed(seed+k); rng = np.random.default_rng(seed+k)
    net = Shared().cuda(); net.train()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=S.REF_LR, weight_decay=1e-4)
    sizes = np.concatenate([np.stack([ALLB[i][:, 2]-ALLB[i][:, 0], ALLB[i][:, 3]-ALLB[i][:, 1]], 1) for i in tri])
    for st in range(steps):
        cur = S.REF_LR*min(1.0, (st+1)/S.WARM)*0.5*(1+math.cos(math.pi*st/steps))
        for pg in opt.param_groups: pg['lr'] = cur
        X, Y, Q, R, O = [], [], [], [], []
        for s_ in range(16):
            photo = (rng.uniform(0.8, 1.2), rng.uniform(-0.08, 0.08), rng.uniform(0.8, 1.25)); kk = int(rng.integers(8))
            if s_ < 16 - real_imgs:
                i = tri[rng.integers(len(tri))]; g = ALLB[i]; n = 8
                gb = g[rng.integers(len(g), size=n)].astype(np.float64); w = gb[:, 2]-gb[:, 0]; h = gb[:, 3]-gb[:, 1]
                lvl = rng.choice([0.04, 0.08, 0.15, 0.3], size=n)
                cx = (gb[:, 0]+gb[:, 2])/2 + rng.normal(0, 1, n)*lvl*w; cy = (gb[:, 1]+gb[:, 3])/2 + rng.normal(0, 1, n)*lvl*h
                nw = w*np.exp(rng.normal(0, 1, n)*lvl*1.2); nh = h*np.exp(rng.normal(0, 1, n)*lvl*1.2)
                b = np.stack([cx-nw/2, cy-nh/2, cx+nw/2, cy+nh/2], 1)
                ng = rng.random(n) < S.REF_NEG
                if ng.any():
                    sz = sizes[rng.integers(len(sizes), size=int(ng.sum()))]; px = rng.uniform(0, 640-sz[:, 0]); py = rng.uniform(0, 640-sz[:, 1])
                    b[ng] = np.stack([px, py, px+sz[:, 0], py+sz[:, 1]], 1)
                real = 0.0
            else:
                i, bx = pool[rng.integers(len(pool))]; g = ALLB[i]
                b = bx[rng.integers(len(bx), size=8)].astype(np.float64); wh = np.maximum(b[:, 2:]-b[:, :2], 4); b = b + rng.normal(0, 0.05, b.shape)*np.concatenate([wh, wh], 1)
                real = 1.0
            b = np.clip(b, -20, 660); M = S.iou_matrix(b, g); q = M.max(1); off = S.box_offsets(b, g[M.argmax(1)].astype(np.float64))
            X.append(S.dihedral_crops(crops(i, b, photo), kk)); Y.append(S.dihedral_offsets(off, kk)); Q.append(q); R.append(np.full(len(b), real))
        X = torch.cat(X); Yt = torch.from_numpy(np.concatenate(Y)).float().clamp(-1, 1).cuda(); Qt = torch.from_numpy(np.concatenate(Q)).float().cuda(); Rt = torch.from_numpy(np.concatenate(R)).float().cuda()
        lg, ql, ol = net.raw(X)
        mreg = ((Qt > 0.3) & ((Rt == 0) | torch.tensor(real_reg, device='cuda'))).float()
        l = bin_loss(lg, Yt, mreg) + (F.binary_cross_entropy_with_logits(ql, Qt, reduction='none')*(1-Rt)).sum()/(1-Rt).sum().clamp(min=1) + (F.binary_cross_entropy_with_logits(ol, (Qt >= 0.5).float(), reduction='none')*Rt).sum()/Rt.sum().clamp(min=1)
        opt.zero_grad(); l.backward(); opt.step()
    net.eval()
    Ck = cands(net, tei, dets_oof); C.update(Ck)
    with torch.no_grad():
        for i in tei:
            rb = C[i][2]
            if len(rb) == 0: S2[i] = np.zeros((0, 2)); continue
            x = crops(i, rb, None); o = torch.stack([torch.sigmoid(net.raw(S.dihedral_crops(x, kk))[2]) for kk in (0, 3, 6, 5)]).mean(0).cpu().numpy()
            S2[i] = np.stack([o, o], 1)
    del net; torch.cuda.empty_cache()
    print('fold', k, round(time.time()-t0), flush=True)
pickle.dump(C, open(f'dev/cands_{tag}.pkl', 'wb')); pickle.dump(S2, open(f'dev/scorer_{tag}.pkl', 'wb'))
b = evaluate(C); print(f'RESULT {tag} {sys.argv[2]} refine-only F1={b[0]} tp={b[4]} P={b[5]}', flush=True)
