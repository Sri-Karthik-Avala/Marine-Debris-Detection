import sys; sys.path.insert(0, 'dev')
from h_md import *
from torchvision.ops import nms as tv_nms
mode = sys.argv[1]; tag = sys.argv[2]
IM = imgs()
if mode == 'dets':
    dets = pickle.load(open(f'dev/oof_dets_{tag}.pkl', 'rb'))
elif mode == 'multiscale':
    base = pickle.load(open('dev/oof_dets_base.pkl', 'rb')); size = int(sys.argv[3]); dets = {}
    for k in range(3):
        tei = np.flatnonzero(FOLD == k); m = S.build_model(); m.load_state_dict(torch.load(f'dev/oof_det_base_f{k}.pt')); m.eval()
        m.transform.min_size = (size,); m.transform.max_size = size
        d = S.predict(m, IM[tei])
        for i, dd in zip(tei, d):
            b = np.concatenate([base[i][0], dd[0]]); s = np.concatenate([base[i][1], dd[1]])
            keep = tv_nms(torch.from_numpy(b).float(), torch.from_numpy(s).float(), 0.5).numpy()
            dets[i] = (b[keep], s[keep])
        del m; torch.cuda.empty_cache()
C = {}
for k in range(3):
    tei = np.flatnonzero(FOLD == k); net = S.BoxRefiner().cuda(); net.load_state_dict(torch.load(f'dev/refx_r_base_s0_f{k}.pt')); net.eval()
    C.update(cands(net, tei, dets))
cov = np.array(sum([list(S.iou_matrix(C[i][2], NW[i]).max(0)) if len(C[i][2]) else [0]*len(NW[i]) for i in C], []))
covraw = np.array(sum([list(S.iou_matrix(C[i][0], NW[i]).max(0)) if len(C[i][0]) else [0]*len(NW[i]) for i in C], []))
b = evaluate(C)
print(f'DET {mode} {tag} {sys.argv[3:]} F1={b[0]} thr={b[3]} tp={b[4]} P={b[5]} raw_cov5={np.mean(covraw>=0.5):.3f} cov81={np.mean(cov>=0.81):.3f}', flush=True)
