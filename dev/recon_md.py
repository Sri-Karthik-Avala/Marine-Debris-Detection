import pandas as pd, numpy as np, json
from PIL import Image
tr=pd.read_csv('train.csv'); te=pd.read_csv('test.csv')
for d in (tr,te):
    d['lg']=d.logged_boxes.apply(json.loads)
tr['nw']=tr.new_boxes.apply(json.loads)
print('groups train',tr.survey_id.nunique(),'test',te.survey_id.nunique())
print('rows/group train', tr.groupby('survey_id').size().describe().to_dict())
print('rows/group test', te.groupby('survey_id').size().describe().to_dict())
nl=tr.lg.apply(len); nn=tr.nw.apply(len)
print('logged per row', nl.describe().to_dict(), 'empty ledger frac', (nl==0).mean())
print('new per row', nn.describe().to_dict())
print('test logged', te.lg.apply(len).describe().to_dict(), (te.lg.apply(len)==0).mean())
allb=np.array([b for bs in tr.lg for b in bs]+[b for bs in tr.nw for b in bs])
newb=np.array([b for bs in tr.nw for b in bs]); logb=np.array([b for bs in tr.lg for b in bs])
for nm,B in [('all',allb),('new',newb),('log',logb)]:
    w=B[:,2]-B[:,0]; h=B[:,3]-B[:,1]
    print(nm,len(B),'w q',np.percentile(w,[5,25,50,75,95]).round(1),'h q',np.percentile(h,[5,25,50,75,95]).round(1), 'area sqrt med', np.sqrt(w*h).mean().round(1))
g=allb*0.9
print('grid 0.9 residual', np.abs(g-np.round(g)).max(), 'frac int', (np.abs(g-np.round(g))<1e-3).mean())
# per-row total boxes
tot=nl+nn; print('total per row', tot.describe().to_dict())
# image stats
import random
ims=tr.image.tolist()[:60]+te.image.tolist()[:60]
res=[]
for p in ims:
    a=np.asarray(Image.open(p)); res.append((a.shape, a.mean(), (a.max(-1)<8).mean()))
print(set(r[0] for r in res))
print('train mean/black', np.mean([r[1] for r in res[:60]]), np.mean([r[2] for r in res[:60]]))
print('test mean/black', np.mean([r[1] for r in res[60:]]), np.mean([r[2] for r in res[60:]]))
# duplicated images across rows?
print('dup images', tr.image.duplicated().sum())
