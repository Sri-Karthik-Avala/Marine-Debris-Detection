p = 'dev/queue_ref_md.sh'; L = open(p, newline='').read().split('\n')
head = L[:4]
assert head[3].startswith('run r_steps2600'), head
new = head + [
 "run r_bins_s1 '{\"steps\": 1300, \"seed\": 1, \"arch\": \"bins\"}'",
 "run r_bins_1000 '{\"steps\": 1000, \"seed\": 0, \"arch\": \"bins\"}'",
 "run r_bins_exp15 '{\"steps\": 1300, \"seed\": 0, \"arch\": \"bins\", \"exp\": 1.5}'",
 "run r_bins_exp30 '{\"steps\": 1300, \"seed\": 0, \"arch\": \"bins\", \"exp\": 3.0}'",
 "run r_bins_lvlwide '{\"steps\": 1300, \"seed\": 0, \"arch\": \"bins\", \"lvls\": [0.08, 0.15, 0.3, 0.45]}'",
 "echo QUEUE_DONE >> dev/ref_queue.log", ""]
open(p, 'w', newline='').write('\n'.join(new))
p = 'dev/queue_det_md.sh'; L = open(p, newline='').read().split('\n')
assert L[0].startswith('until grep -q QUEUE_DONE'), L[0]
new = [L[0],
 "conda run -n max --no-capture-output python -u dev/det_eval_md.py dets base 2>&1 | grep -E \"DET|Error\" >> dev/det_queue.log",
 "for cfg in \"nms06 0.6 3 640\" \"e2 0.5 2 640\"; do",
 "  set -- $cfg",
 "  conda run -n max --no-capture-output python -u dev/oof_det_md.py $1 $2 $3 $4 2>&1 | grep -E \"saved|Error\" >> dev/det_queue.log",
 "  conda run -n max --no-capture-output python -u dev/det_eval_md.py dets $1 2>&1 | grep -E \"DET|Error\" >> dev/det_queue.log",
 "done",
 "echo DET_DONE >> dev/det_queue.log", ""]
open(p, 'w', newline='').write('\n'.join(new))
print(open('dev/queue_ref_md.sh').read()); print(open('dev/queue_det_md.sh').read())
