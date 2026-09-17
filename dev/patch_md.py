s = open('dev/h_md.py').read()
s = s.replace("""    if arch == 'r34':""", """    if arch == 'bins':
        return BinRef(freeze)
    if arch == 'r34':""")
s = s.replace("""        out = net(X); m = (Qt > 0.3).float()
        if loss == 'l1':""", """        m = (Qt > 0.3).float()
        if arch == 'bins':
            lgb, qb = net.raw(X); lreg = bin_loss(lgb, Yt, m); out = torch.cat([torch.zeros_like(qb)[:, None].repeat(1, 4), qb[:, None]], 1); loss = 'bins'
        else:
            out = net(X)
        if loss == 'bins': pass
        elif loss == 'l1':""")
open('dev/h_md.py', 'w').write(s)
