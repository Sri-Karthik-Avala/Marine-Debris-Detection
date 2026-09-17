import sys; sys.path.insert(0, 'dev')
from h_md import *
cand_tag = sys.argv[1]; sc_tag = sys.argv[2]
G = imgs_gpu(); mean = S.MEAN.cuda(); std = S.STD.cuda()
OOF = pickle.load(open(f'dev/cands_{cand_tag}.pkl', 'rb')); S2 = {}
for k in range(3):
    net = S.BoxRefiner().cuda(); net.head[-1] = nn.Linear(512, 1).cuda(); net.load_state_dict(torch.load(f'dev/scorer_{sc_tag}_f{k}.pt')); net.eval()
    with torch.no_grad():
        for i in np.flatnonzero(FOLD == k):
            rb = OOF[i][2]
            if len(rb) == 0: S2[i] = np.zeros((0, 2)); continue
            t = G[i].permute(2, 0, 1).float().div(255)
            cx = (rb[:, 0]+rb[:, 2])/2; cy = (rb[:, 1]+rb[:, 3])/2; w = np.maximum(rb[:, 2]-rb[:, 0], 4)*S.REF_EXP; h = np.maximum(rb[:, 3]-rb[:, 1], 4)*S.REF_EXP
            rois = torch.from_numpy(np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1)).float().cuda()
            x = (S.roi_align(t.unsqueeze(0), [rois], output_size=(S.REF_R, S.REF_R), spatial_scale=1.0, sampling_ratio=2, aligned=True)-mean)/std
            o = torch.stack([torch.sigmoid(net(S.dihedral_crops(x, kk))) for kk in (0, 3, 6, 5)]).mean(0)[:, 0].cpu().numpy()
            S2[i] = np.stack([o, o], 1)
pickle.dump(S2, open(f'dev/scorer_{sc_tag}_on_{cand_tag}.pkl', 'wb')); print('saved')
