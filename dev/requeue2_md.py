p = 'dev/queue_ref_md.sh'; L = open(p, newline='').read().split('\n')
i = [k for k, l in enumerate(L) if l.startswith('run r_bins_1000')][0]
new = L[:i+1] + [
 "run r_bins_nofreeze '{\"steps\": 1300, \"seed\": 0, \"arch\": \"bins\", \"freeze\": false}'",
 "run r_bins_exp15 '{\"steps\": 1300, \"seed\": 0, \"arch\": \"bins\", \"exp\": 1.5}'",
 "run r_bins_lvlwide '{\"steps\": 1300, \"seed\": 0, \"arch\": \"bins\", \"lvls\": [0.08, 0.15, 0.3, 0.45]}'",
 "echo QUEUE_DONE >> dev/ref_queue.log", ""]
open(p, 'w', newline='').write('\n'.join(new)); print('\n'.join(new[i-1:]))
