# made by - Karthik
import sys
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import roi_align

SEED = 0
EPOCHS = 3
BATCH = 4
LR = 0.01
WD = 1e-4
WARM = 100
HOLD_FRAC = 0.15
RES = 640
MATCH_IOU = 0.81
PRE_SCORE = 0.02
REF_STEPS = 1300
REF_R = 96
REF_IMGS = 16
REF_PER_IMG = 8
REF_EXP = 2.0
REF_LR = 1e-3
REF_NEG = 0.15
REF_ITERS = 2
REF_MIN_SCORE = 0.1
REF_VARIANTS = (0, 3, 6, 5)
DEVICE = torch.device("cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_images(public_dir, paths):
    arr = np.zeros((len(paths), RES, RES, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(public_dir / p).convert("RGB")
        if im.size != (RES, RES):
            im = im.resize((RES, RES), Image.BILINEAR)
        arr[i] = np.asarray(im)
    return arr


def parse_boxes(s):
    return np.array(json.loads(s), dtype=np.float32).reshape(-1, 4)


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


def dihedral_crops(x, k):
    if k & 1:
        x = x.transpose(2, 3)
    if k & 2:
        x = x.flip(3)
    if k & 4:
        x = x.flip(2)
    return x


def dihedral_offsets(o, k):
    if k & 1:
        o = o[:, [1, 0, 3, 2]]
    if k & 2:
        o = np.stack([-o[:, 2], o[:, 1], -o[:, 0], o[:, 3]], 1)
    if k & 4:
        o = np.stack([o[:, 0], -o[:, 3], o[:, 2], -o[:, 1]], 1)
    return o


def undo_dihedral_offsets(o, k):
    if k & 4:
        o = np.stack([o[:, 0], -o[:, 3], o[:, 2], -o[:, 1]], 1)
    if k & 2:
        o = np.stack([-o[:, 2], o[:, 1], -o[:, 0], o[:, 3]], 1)
    if k & 1:
        o = o[:, [1, 0, 3, 2]]
    return o


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


def photometric(t, rng):
    c = float(rng.uniform(0.8, 1.2))
    br = float(rng.uniform(-0.08, 0.08))
    g = float(rng.uniform(0.8, 1.25))
    return (t.clamp(1e-4, 1).pow(g) * c + br).clamp(0, 1)


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
                t = photometric(to_tensor(a), rng)
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
def predict(model, imgs):
    model.eval()
    out = []
    for s in range(0, len(imgs), BATCH):
        batch = [to_tensor(imgs[i]).to(DEVICE) for i in range(s, min(s + BATCH, len(imgs)))]
        for r in model(batch):
            sc = r["scores"].cpu().numpy().astype(np.float64)
            bx = r["boxes"].cpu().numpy().astype(np.float64)
            m = sc >= PRE_SCORE
            out.append((bx[m], sc[m]))
    return out


class BoxRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        for mod in (m.conv1, m.bn1, m.layer1):
            for p in mod.parameters():
                p.requires_grad = False
        k = REF_R // 32
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512 * k * k, 512), nn.ReLU(), nn.Linear(512, 5))

    def forward(self, x):
        return self.head(self.body(x))


def crop_regions(img_t, b):
    b = np.asarray(b, dtype=np.float64)
    cx = (b[:, 0] + b[:, 2]) / 2
    cy = (b[:, 1] + b[:, 3]) / 2
    w = np.maximum(b[:, 2] - b[:, 0], 4) * REF_EXP
    h = np.maximum(b[:, 3] - b[:, 1], 4) * REF_EXP
    rois = torch.from_numpy(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)).float()
    x = roi_align(img_t.unsqueeze(0), [rois], output_size=(REF_R, REF_R), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    return (x - MEAN) / STD


def box_offsets(b, t):
    bw = np.maximum(b[:, 2] - b[:, 0], 4)
    bh = np.maximum(b[:, 3] - b[:, 1], 4)
    return np.stack([(t[:, 0] - b[:, 0]) / bw, (t[:, 1] - b[:, 1]) / bh, (t[:, 2] - b[:, 2]) / bw, (t[:, 3] - b[:, 3]) / bh], 1)


def train_refiner(imgs, boxes_list, rng):
    net = BoxRefiner().to(DEVICE)
    net.train()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=REF_LR, weight_decay=1e-4)
    sizes = np.concatenate([np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1) for b in boxes_list])
    run_r = run_q = 0.0
    for st in range(REF_STEPS):
        lr = REF_LR * min(1.0, (st + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * st / REF_STEPS))
        for pg in opt.param_groups:
            pg["lr"] = lr
        X, Y, Q = [], [], []
        for i in rng.integers(len(imgs), size=REF_IMGS):
            g = boxes_list[i]
            n = REF_PER_IMG
            gb = g[rng.integers(len(g), size=n)].astype(np.float64)
            w = gb[:, 2] - gb[:, 0]
            h = gb[:, 3] - gb[:, 1]
            lvl = rng.choice([0.04, 0.08, 0.15, 0.3], size=n)
            cx = (gb[:, 0] + gb[:, 2]) / 2 + rng.normal(0, 1, n) * lvl * w
            cy = (gb[:, 1] + gb[:, 3]) / 2 + rng.normal(0, 1, n) * lvl * h
            nw = w * np.exp(rng.normal(0, 1, n) * lvl * 1.2)
            nh = h * np.exp(rng.normal(0, 1, n) * lvl * 1.2)
            b = np.stack([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2], 1)
            neg = rng.random(n) < REF_NEG
            if neg.any():
                sz = sizes[rng.integers(len(sizes), size=int(neg.sum()))]
                px = rng.uniform(0, RES - sz[:, 0])
                py = rng.uniform(0, RES - sz[:, 1])
                b[neg] = np.stack([px, py, px + sz[:, 0], py + sz[:, 1]], 1)
            b = np.clip(b, -20, RES + 20)
            M = iou_matrix(b, g)
            q = M.max(1)
            off = box_offsets(b, g[M.argmax(1)].astype(np.float64))
            k = int(rng.integers(8))
            x = dihedral_crops(crop_regions(photometric(to_tensor(imgs[i]), rng), b), k)
            X.append(x)
            Y.append(dihedral_offsets(off, k))
            Q.append(q)
        X = torch.cat(X).to(DEVICE)
        Y = torch.from_numpy(np.concatenate(Y)).float().clamp(-1, 1).to(DEVICE)
        Q = torch.from_numpy(np.concatenate(Q)).float().to(DEVICE)
        out = net(X)
        m = (Q > 0.3).float()
        lreg = ((out[:, :4] - Y).abs().sum(1) * m).sum() / m.sum().clamp(min=1)
        lq = F.binary_cross_entropy_with_logits(out[:, 4], Q)
        loss = lreg + lq
        opt.zero_grad()
        loss.backward()
        opt.step()
        run_r += float(lreg)
        run_q += float(lq)
        if (st + 1) % 50 == 0:
            print(f"refiner step {st + 1}/{REF_STEPS} reg {run_r / 50:.4f} qual {run_q / 50:.4f}", flush=True)
            run_r = run_q = 0.0
    net.eval()
    return net


@torch.no_grad()
def refine_boxes(net, img, b):
    b = np.asarray(b, dtype=np.float64).copy()
    q = np.zeros(len(b))
    if len(b) == 0:
        return b, q
    img_t = to_tensor(img)
    n = len(b)
    nvar = len(REF_VARIANTS)
    for _ in range(REF_ITERS):
        x = crop_regions(img_t, b)
        out = net(torch.cat([dihedral_crops(x, k) for k in REF_VARIANTS]).to(DEVICE)).cpu().numpy().astype(np.float64)
        acc = np.zeros((n, 4))
        qs = np.zeros(n)
        for vi, k in enumerate(REF_VARIANTS):
            o = out[vi * n:(vi + 1) * n]
            acc += undo_dihedral_offsets(o[:, :4], k) / nvar
            qs += 1.0 / (1.0 + np.exp(-o[:, 4])) / nvar
        bw = np.maximum(b[:, 2] - b[:, 0], 4)
        bh = np.maximum(b[:, 3] - b[:, 1], 4)
        b = b + acc * np.stack([bw, bh, bw, bh], 1)
        q = qs
    b = np.clip(b, 0, RES)
    b[:, 2] = np.maximum(b[:, 2], b[:, 0] + 1.0)
    b[:, 3] = np.maximum(b[:, 3], b[:, 1] + 1.0)
    return b, q


def candidates(net, img, det):
    bx, sc = det
    m = sc >= REF_MIN_SCORE
    bx, sc = bx[m], sc[m]
    rb, q = refine_boxes(net, img, bx)
    return bx, sc, rb, q


def variant(cand, use_ref):
    bx, sc, rb, q = cand
    return (rb if use_ref else bx), sc


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
        m = np.zeros(len(fs), dtype=bool)
        m[int(np.argmax(fs))] = True
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
    print("detector training done", flush=True)
    ho_dets = predict(model, tr_imgs[ho_idx])
    te_dets = predict(model, te_imgs)
    del model
    print("detector inference done", flush=True)

    refiner = train_refiner(tr_imgs[tr_idx], [all_boxes[i] for i in tr_idx], rng)
    print("refiner training done", flush=True)
    ho_cand = [candidates(refiner, tr_imgs[i], d) for i, d in zip(ho_idx, ho_dets)]
    te_cand = [candidates(refiner, te_imgs[i], d) for i, d in enumerate(te_dets)]
    print("refinement done", flush=True)

    ho_logged = [tr_logged[i] for i in ho_idx]
    ho_new = [tr_new[i] for i in ho_idx]
    total_t = sum(len(g) for g in ho_new)
    thrs = np.round(np.arange(0.05, 0.96, 0.025), 3)
    best = None
    for use_ref in (0, 1):
        vs = [variant(c, use_ref) for c in ho_cand]
        local = None
        for sup in (0.3, 0.5, 0.7):
            for force in (0, 1):
                for thr in thrs:
                    tp = p = 0
                    for (fb, fs), lg, gt in zip(vs, ho_logged, ho_new):
                        pr = select(fb, fs, lg, sup, thr, force)
                        p += len(pr)
                        tp += match_count(pr, gt)
                    f1 = 2 * tp / max(p + total_t, 1)
                    if local is None or f1 > local[0] + 1e-12:
                        local = (f1, use_ref, sup, force, float(thr), tp, p)
        print(f"refined {use_ref}: F1@{MATCH_IOU} {local[0]:.4f} cfg {local[2:]}", flush=True)
        if best is None or local[0] > best[0] + 1e-12:
            best = local
    f1, use_ref, sup, force, thr, tp, p = best
    print(f"holdout detection F1@{MATCH_IOU} {f1:.4f} refined {use_ref} suppress {sup} force {force} thr {thr} tp {tp} pred {p} targets {total_t}", flush=True)

    rows = []
    for i in range(len(test)):
        fb, fs = variant(te_cand[i], use_ref)
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
