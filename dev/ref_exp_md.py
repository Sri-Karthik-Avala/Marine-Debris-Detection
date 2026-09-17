import sys, json; sys.path.insert(0, 'dev')
from h_md import *
tag = sys.argv[1]; cfg = json.loads(sys.argv[2]); dets_tag = cfg.pop('dets', 'base'); iters = cfg.pop('iters', 2)
dets = pickle.load(open(f'dev/oof_dets_{dets_tag}.pkl', 'rb'))
S.REF_ITERS = iters
t0 = time.time(); C = {}
for k in range(3):
    tri = np.flatnonzero(FOLD != k); tei = np.flatnonzero(FOLD == k)
    net = train_refiner_gpu(tri, **cfg)
    C.update(cands(net, tei, dets))
    if k == 0: print('fold0 time', round(time.time()-t0), flush=True)
    torch.save(net.state_dict(), f'dev/refx_{tag}_f{k}.pt')
    del net; torch.cuda.empty_cache()
pickle.dump(C, open(f'dev/cands_{tag}.pkl', 'wb'))
cov = []
for i in C:
    bx, sc, rb, q = C[i]; g = NW[i]
    cov += list(S.iou_matrix(rb, g).max(0)) if len(rb) else [0]*len(g)
cov = np.array(cov)
b = evaluate(C)
print(f'RESULT {tag} cfg={sys.argv[2]} F1={b[0]} thr={b[3]} tp={b[4]} P={b[5]} T={b[6]} cov81={np.mean(cov>=0.81):.3f} cov5={np.mean(cov>=0.5):.3f} time={time.time()-t0:.0f}', flush=True)
