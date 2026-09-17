import numpy as np
from scipy.optimize import linear_sum_assignment
def iou_m(a, b):
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy; aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter)
def tp_count(p, g, thr=0.81):
    M = iou_m(p, g) >= thr - 1e-12
    if M.size == 0 or not M.any(): return 0
    r, c = linear_sum_assignment(-M.astype(float)); return int(M[r, c].sum())
def group_score(preds, gts, groups, thr=0.81):
    agg = {}
    for p, g, s in zip(preds, gts, groups):
        a = agg.setdefault(s, [0, 0, 0]); a[0] += tp_count(p, g, thr); a[1] += len(p); a[2] += len(g)
    return float(np.mean([1.0 if P + T == 0 else 2*TP/(P+T) for TP, P, T in agg.values()]))
