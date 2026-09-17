import sys; sys.path.insert(0, 'dev')
from h_md import *
tag = sys.argv[1]; nms_t = float(sys.argv[2]); epochs = int(sys.argv[3]); res = int(sys.argv[4]); roi_bs = int(sys.argv[5]) if len(sys.argv) > 5 else 512
S.EPOCHS = epochs
dets = {}
IM = imgs()
for k in range(3):
    tri = np.flatnonzero(FOLD != k); tei = np.flatnonzero(FOLD == k)
    torch.manual_seed(k)
    old = S.RES
    m = S.build_model(); m.roi_heads.nms_thresh = nms_t; m.roi_heads.fg_bg_sampler.batch_size_per_image = roi_bs
    if res != 640:
        m.transform.min_size = (res,); m.transform.max_size = res
    m = S.train_model(m, IM[tri], [ALLB[i] for i in tri], np.random.default_rng(k))
    d = S.predict(m, IM[tei])
    for i, dd in zip(tei, d): dets[i] = dd
    torch.save(m.state_dict(), f'dev/oof_det_{tag}_f{k}.pt'); del m; torch.cuda.empty_cache()
    print('fold', k, 'done', flush=True)
pickle.dump(dets, open(f'dev/oof_dets_{tag}.pkl', 'wb')); print('saved', tag)
