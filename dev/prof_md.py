import torch, torchvision, time, torch.nn as nn
for R in (96, 64):
    m = torchvision.models.resnet18(); body = nn.Sequential(*list(m.children())[:-2]); head = nn.Sequential(nn.Flatten(), nn.Linear(512*(R//32)**2, 512), nn.ReLU(), nn.Linear(512, 5))
    for frz in (0, 1):
        for p in list(body[0].parameters()) + list(body[1].parameters()) + list(body[4].parameters()): p.requires_grad = not frz
        params = [p for p in list(body.parameters()) + list(head.parameters()) if p.requires_grad]
        opt = torch.optim.AdamW(params, 1e-3); x = torch.randn(128, 3, R, R)
        for i in range(8):
            if i == 3: t = time.time()
            out = head(body(x)); loss = out.abs().mean(); opt.zero_grad(); loss.backward(); opt.step()
        print('R', R, 'freeze_stem_l1', frz, 'fwd+bwd s/step', (time.time()-t)/5, 'threads', torch.get_num_threads(), flush=True)
