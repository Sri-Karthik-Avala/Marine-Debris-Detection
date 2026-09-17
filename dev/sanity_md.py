import pandas as pd, json, numpy as np
s=pd.read_csv('working/submission.csv'); t=pd.read_csv('test.csv')
n=s.new_boxes.apply(lambda x: len(json.loads(x)))
print('boxes/row dist', n.value_counts().sort_index().to_dict())
B=np.array([b for x in s.new_boxes for b in json.loads(x)])
print('coord range',B.min(),B.max(),'min w',(B[:,2]-B[:,0]).min(),'min h',(B[:,3]-B[:,1]).min())
m=t.set_index('id').logged_boxes.apply(json.loads)
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1])); i=ix*iy
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
mx=[max([iou(b,l) for l in m[r.id]] or [0]) for r in s.itertuples() for b in json.loads(r.new_boxes)]
print('max IoU of any pred with its ledger', max(mx))
