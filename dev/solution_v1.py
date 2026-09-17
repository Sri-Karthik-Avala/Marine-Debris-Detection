# made by - Karthik
import sys
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

SEED = 0
EPOCHS = 3
BATCH = 4
LR = 0.01
WD = 1e-4
WARM = 100
HOLD_FRAC = 0.15
RES = 640
MATCH_IOU = 0.81
FUSE_IOU = 0.55
PRE_SCORE = 0.02
DEVICE = torch.device("cpu")


def load_images(public_dir, paths):
    arr = np.zeros((len(paths), RES, RES, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(public_dir / p).convert("RGB")
        if im.size != (RES, RES):
            im = im.resize((RES, RES), Image.BILINEAR)
        arr[i] = np.asarray(im)
    return arr


def parse_boxes(s):
    b = np.array(json.loads(s), dtype=np.float32).reshape(-1, 4)
    return b


def dihedral_image(a, k):
    if k & 1:
        a = a.transpose(1, 0, 2)
    if k & 2:
        a = a[:, ::-1]
    if k & 4:
        a = a[::-1]
    return a


def dihedral_boxes(b, k):
    b = b.copy()
    if k & 1:
        b = b[:, [1, 0, 3, 2]]
    if k & 2:
        b = np.stack([RES - b[:, 2], b[:, 1], RES - b[:, 0], b[:, 3]], 1)
    if k & 4:
        b = np.stack([b[:, 0], RES - b[:, 3], b[:, 2], RES - b[:, 1]], 1)
    return b


def undo_dihedral_boxes(b, k):
    b = b.copy()
    if k & 4:
        b = np.stack([b[:, 0], RES - b[:, 3], b[:, 2], RES - b[:, 1]], 1)
    if k & 2:
        b = np.stack([RES - b[:, 2], b[:, 1], RES - b[:, 0], b[:, 3]], 1)
    if k & 1:
        b = b[:, [1, 0, 3, 2]]
    return b


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def build_model():
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
        min_size=RES, max_size=RES,
        box_detections_per_img=150, box_score_thresh=0.01,
        rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=600,
        rpn_pre_nms_top_n_train=2000, rpn_post_nms_top_n_train=1000,
    )
    model.rpn.anchor_generator = AnchorGenerator(((16, 32, 64, 128, 256),) * 3, ((0.5, 1.0, 2.0),) * 3)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    return model.to(DEVICE)


def to_tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div_(255.0)


def clean_boxes(b):
    b = np.clip(b, 0, RES)
    keep = ((b[:, 2] - b[:, 0]) >= 1.0) & ((b[:, 3] - b[:, 1]) >= 1.0)
    return b[keep]


def train_model(model, imgs, boxes_list, rng):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=WD)
    n = len(imgs)
    spe = n // BATCH
    total = EPOCHS * spe
    step = 0
    for ep in range(EPOCHS):
        order = rng.permutation(n)
        run = 0.0
        for bi in range(spe):
            idx = order[bi * BATCH:(bi + 1) * BATCH]
            ims, tg = [], []
            for i in idx:
                k = int(rng.integers(8))
                a = dihedral_image(imgs[i], k)
                b = clean_boxes(dihedral_boxes(boxes_list[i], k))
                t = to_tensor(a)
                c = float(rng.uniform(0.8, 1.2))
                br = float(rng.uniform(-0.08, 0.08))
                g = float(rng.uniform(0.8, 1.25))
                t = (t.clamp(1e-4, 1).pow(g) * c + br).clamp(0, 1)
                ims.append(t.to(DEVICE))
                tg.append({"boxes": torch.from_numpy(b).float().to(DEVICE),
                           "labels": torch.ones(len(b), dtype=torch.int64, device=DEVICE)})
            lr = LR * min(1.0, (step + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * step / total))
            for pg in opt.param_groups:
                pg["lr"] = lr
            losses = model(ims, tg)
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            run += float(loss)
            step += 1
            if step % 25 == 0:
                print(f"ep {ep} step {step}/{total} loss {run / (bi + 1):.4f} lr {lr:.5f}", flush=True)
        print(f"epoch {ep} done loss {run / spe:.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, imgs, variants):
    model.eval()
    out = [[] for _ in range(len(imgs))]
    for k in variants:
        for s in range(0, len(imgs), BATCH):
            batch = [to_tensor(dihedral_image(imgs[i], k)).to(DEVICE) for i in range(s, min(s + BATCH, len(imgs)))]
            res = model(batch)
            for j, r in enumerate(res):
                sc = r["scores"].cpu().numpy()
                bx = r["boxes"].cpu().numpy()
                m = sc >= PRE_SCORE
                out[s + j].append((undo_dihedral_boxes(bx[m], k), sc[m]))
    return out


def fuse(dets, nv):
    parts = dets[:nv]
    B = np.concatenate([p[0] for p in parts]) if parts else np.zeros((0, 4))
    S = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0)
    if len(B) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    if nv == 1:
        return B.astype(np.float64), S.astype(np.float64)
    order = np.argsort(-S)
    fb, mem = [], []
    for i in order:
        if fb:
            ious = iou_matrix(B[i:i + 1], np.array(fb))[0]
            j = int(np.argmax(ious))
            if ious[j] >= FUSE_IOU:
                mem[j].append(i)
                w = S[mem[j]].astype(np.float64)
                fb[j] = (B[mem[j]].astype(np.float64) * w[:, None]).sum(0) / w.sum()
                continue
        fb.append(B[i].astype(np.float64))
        mem.append([i])
    fs = np.array([S[m].astype(np.float64).sum() / max(nv, len(m)) for m in mem])
    return np.array(fb), fs


def select(fb, fs, logged, sup, thr, force):
    if len(fb) == 0:
        return np.zeros((0, 4))
    if len(logged):
        keep = iou_matrix(fb, logged).max(1) < sup
        fb, fs = fb[keep], fs[keep]
    if len(fb) == 0:
        return np.zeros((0, 4))
    m = fs >= thr
    if m.sum() == 0 and force:
        m = fs >= fs.max()
        m[np.flatnonzero(m)[1:]] = False
    return fb[m]


def match_count(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return 0
    iou = iou_matrix(pred, gt)
    pi, gi = np.nonzero(iou >= MATCH_IOU)
    if len(pi) == 0:
        return 0
    order = np.argsort(-iou[pi, gi])
    up, ug, c = set(), set(), 0
    for o in order:
        if pi[o] in up or gi[o] in ug:
            continue
        up.add(pi[o])
        ug.add(gi[o])
        c += 1
    return c


def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    tr_logged = [parse_boxes(s) for s in train["logged_boxes"]]
    tr_new = [parse_boxes(s) for s in train["new_boxes"]]
    te_logged = [parse_boxes(s) for s in test["logged_boxes"]]

    groups = train["survey_id"].unique()
    grng = np.random.default_rng(SEED + 1)
    perm = grng.permutation(len(groups))
    hold_groups, cnt = set(), 0
    for gi in perm:
        g = groups[gi]
        sz = int((train["survey_id"] == g).sum())
        if sz > HOLD_FRAC * len(train) / 4:
            continue
        hold_groups.add(g)
        cnt += sz
        if cnt >= HOLD_FRAC * len(train):
            break
    is_hold = train["survey_id"].isin(hold_groups).to_numpy()
    tr_idx = np.flatnonzero(~is_hold)
    ho_idx = np.flatnonzero(is_hold)
    print(f"train rows {len(tr_idx)} holdout rows {len(ho_idx)} holdout groups {len(hold_groups)}", flush=True)

    tr_imgs = load_images(public_dir, train["image"].tolist())
    te_imgs = load_images(public_dir, test["image"].tolist())
    print("images loaded", flush=True)

    all_boxes = [clean_boxes(np.concatenate([tr_logged[i], tr_new[i]])) for i in range(len(train))]
    model = build_model()
    model = train_model(model, tr_imgs[tr_idx], [all_boxes[i] for i in tr_idx], rng)
    print("training done", flush=True)

    ho_dets = predict(model, tr_imgs[ho_idx], [0, 3, 6, 5])
    print("holdout inference done", flush=True)
    ho_logged = [tr_logged[i] for i in ho_idx]
    ho_new = [tr_new[i] for i in ho_idx]
    total_t = sum(len(g) for g in ho_new)
    best = None
    thrs = np.round(np.arange(0.05, 0.96, 0.025), 3)
    for nv in (1, 2, 4):
        fused = [fuse(d, nv) for d in ho_dets]
        for sup in (0.3, 0.4, 0.5, 0.6, 0.7):
            for force in (0, 1):
                for thr in thrs:
                    tp = p = 0
                    for (fb, fs), lg, gt in zip(fused, ho_logged, ho_new):
                        pr = select(fb, fs, lg, sup, thr, force)
                        p += len(pr)
                        tp += match_count(pr, gt)
                    f1 = 2 * tp / max(p + total_t, 1)
                    if best is None or f1 > best[0] + 1e-12:
                        best = (f1, nv, sup, force, float(thr), tp, p)
        print(f"variants {nv} best so far f1 {best[0]:.4f} cfg {best[1:]}", flush=True)
    f1, nv, sup, force, thr, tp, p = best
    print(f"holdout detection F1@{MATCH_IOU} {f1:.4f} variants {nv} suppress {sup} force {force} thr {thr} tp {tp} pred {p} targets {total_t}", flush=True)

    variants = [0, 3, 6, 5][:nv]
    te_dets = predict(model, te_imgs, variants)
    rows = []
    for i in range(len(test)):
        fb, fs = fuse(te_dets[i], nv)
        pr = select(fb, fs, te_logged[i], sup, thr, force)
        pr = np.clip(pr, 0, RES)
        pr = pr[((pr[:, 2] - pr[:, 0]) > 0) & ((pr[:, 3] - pr[:, 1]) > 0)] if len(pr) else pr
        rows.append(json.dumps([[round(float(v), 4) for v in b] for b in pr]))
    submission = pd.DataFrame({"id": test["id"], "new_boxes": rows})
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)
    print(f"wrote {len(submission)} rows, mean boxes {np.mean([len(json.loads(r)) for r in rows]):.2f}", flush=True)


if __name__ == "__main__":
    main()
