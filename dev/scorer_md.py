import sys; sys.path.insert(0, 'dev')
from h_md import *
from sklearn.metrics import roc_auc_score
steps = int(sys.argv[1]); tag = sys.argv[2]
IM = imgs(); G = imgs_gpu(); mean = S.MEAN.cuda(); std = S.STD.cuda()
OOF = pickle.load(open('dev/cands_r_base_s0.pkl', 'rb'))
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
    ref = S.BoxRefiner().cuda(); ref.load_state_dict(torch.load(f'dev/refx_r_base_s0_f{k}.pt')); ref.eval()
    cin = cands(ref, tri, dets_in); del ref
    pool = []
    for i in tri:
        bx, sc, rb, q = cin[i]
        if len(rb) == 0: continue
        iou = S.iou_matrix(rb, ALLB[i]).max(1); pool.append((i, rb, iou))
    y81 = np.concatenate([p[2] for p in pool]) >= 0.81
    print('fold', k, 'in-sample cands', sum(len(p[2]) for p in pool), 'pos81 frac', y81.mean().round(3), flush=True)
    torch.manual_seed(k); rng = np.random.default_rng(k)
    net = S.BoxRefiner().cuda(); net.head[-1] = nn.Linear(512, 2).cuda(); net.train()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    for st in range(steps):
        cur = 1e-3*min(1, (st+1)/100)*0.5*(1+math.cos(math.pi*st/steps))
        for pg in opt.param_groups: pg['lr'] = cur
        X, Y = [], []
        for pj in rng.integers(len(pool), size=16):
            i, rb, iou = pool[pj]; sel = rng.integers(len(rb), size=8)
            b = rb[sel].copy(); b += rng.normal(0, 0.01, b.shape)*np.repeat(np.maximum(b[:, 2:]-b[:, :2], 4), 2, 1)
            it = S.iou_matrix(b, ALLB[i]).max(1)
            x = crops_gpu(i, b, (rng.uniform(0.8, 1.2), rng.uniform(-0.08, 0.08), rng.uniform(0.8, 1.25)))
            X.append(S.dihedral_crops(x, int(rng.integers(8)))); Y.append(np.stack([it >= 0.81, it >= 0.5], 1))
        X = torch.cat(X); Yt = torch.from_numpy(np.concatenate(Y)).float().cuda()
        l = F.binary_cross_entropy_with_logits(net(X), Yt); opt.zero_grad(); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        for i in tei:
            bx, sc, rb, q = OOF[i]
            if len(rb) == 0: S2[i] = np.zeros((0, 2)); continue
            x = crops_gpu(i, rb); o = torch.stack([torch.sigmoid(net(S.dihedral_crops(x, kk))) for kk in (0, 3, 6, 5)]).mean(0)
            S2[i] = o.cpu().numpy()
    del net; torch.cuda.empty_cache()
pickle.dump(S2, open(f'dev/scorer_{tag}.pkl', 'wb'))
lab, a_sc, a_p81, a_p50 = [], [], [], []
for i in OOF:
    bx, sc, rb, q = OOF[i]
    if not len(rb): continue
    keep = (S.iou_matrix(rb, LG[i]).max(1) < 0.3 if len(LG[i]) else np.ones(len(rb), bool)) & (sc >= 0.3)
    l = (S.iou_matrix(rb, NW[i]).max(1) >= 0.81) if len(NW[i]) else np.zeros(len(rb), bool)
    lab.append(l[keep]); a_sc.append(sc[keep]); a_p81.append(S2[i][keep, 0]); a_p50.append(S2[i][keep, 1])
lab = np.concatenate(lab); a_sc = np.concatenate(a_sc); a_p81 = np.concatenate(a_p81); a_p50 = np.concatenate(a_p50)
print('AUC sc', roc_auc_score(lab, a_sc).round(3), 'p81', roc_auc_score(lab, a_p81).round(3), 'p50', roc_auc_score(lab, a_p50).round(3), 'sc*p81', roc_auc_score(lab, a_sc*a_p81).round(3), 'sqrt', roc_auc_score(lab, np.sqrt(a_sc*a_p81)).round(3))
for a, b in ((1, 0), (0, 1), (1, 1), (0.5, 1), (1, 0.5), (1, 2), (0, 0)):
    Cm = {i: (OOF[i][0], (OOF[i][1]**a)*(S2[i][:, 0]**b if b else 1) if len(OOF[i][1]) else OOF[i][1], OOF[i][2], OOF[i][3]) for i in OOF}
    if a == 0 and b == 0:
        Cm = {i: (OOF[i][0], OOF[i][1]*S2[i][:, 1] if len(OOF[i][1]) else OOF[i][1], OOF[i][2], OOF[i][3]) for i in OOF}
    ev = evaluate_generic = None
    best = None
    for thr in np.quantile(np.concatenate([Cm[i][1] for i in Cm if len(Cm[i][1])]), np.linspace(0.6, 0.995, 80)):
        tp = p = 0
        for i in Cm:
            pr = S.select(Cm[i][2], Cm[i][1], LG[i], 0.3, thr, 1); p += len(pr); tp += S.match_count(pr, NW[i])
        f = 2*tp/(p+2326)
        if best is None or f > best[0]: best = (round(f, 4), tp, p)
    print(f'score=sc^{a}*p81^{b}' if (a, b) != (0, 0) else 'score=sc*p50', best, flush=True)
