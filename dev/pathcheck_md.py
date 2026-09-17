import sys; sys.path.insert(0, '.')
import sys, numpy as np, pandas as pd, torch, json
sys.argv=['x','.','dev/tmp.csv']
torch.set_num_threads(2)
import solution as S
from pathlib import Path
tr=pd.read_csv('train.csv').iloc[:4]
imgs=S.load_images(Path('.'), tr.image.tolist())
m=S.build_model()
S.EPOCHS=1
b=[S.clean_boxes(np.concatenate([S.parse_boxes(a),S.parse_boxes(c)])) for a,c in zip(tr.logged_boxes,tr.new_boxes)]
S.train_model(m, imgs, b, np.random.default_rng(0))
m.roi_heads.score_thresh=0.0
d=S.predict(m, imgs, [0,3])
for nv in (1,2):
    fb,fs=S.fuse(d[0],nv); print(nv, fb.shape, fs[:3])
    pr=S.select(fb,fs,S.parse_boxes(tr.logged_boxes[1]),0.5,0.5,1); print(pr.shape, S.match_count(pr,b[0]))
# dihedral roundtrip check
bx=b[2]
for k in range(8):
    assert np.allclose(S.undo_dihedral_boxes(S.dihedral_boxes(bx,k),k),bx)
a=np.zeros((640,640,3),np.uint8); a[100:120,300:340]=255
for k in range(8):
    aa=S.dihedral_image(a,k); ys,xs=np.nonzero(aa[...,0]); bb=S.dihedral_boxes(np.array([[300,100,340,120]],np.float32),k)[0]
    assert (xs.min(),ys.min(),xs.max()+1,ys.max()+1)==tuple(bb.astype(int)), (k,bb)
print('ok')
