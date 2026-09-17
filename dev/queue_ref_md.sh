run() { conda run -n max --no-capture-output python -u dev/ref_exp_md.py "$1" "$2" 2>&1 | grep -E "RESULT|Error|error" >> dev/ref_queue.log; }
run r_base_s1 '{"steps": 1300, "seed": 1}'
run r_bins '{"steps": 1300, "seed": 0, "arch": "bins"}'
run r_steps2600 '{"steps": 2600, "seed": 0}'
run r_bins_s1 '{"steps": 1300, "seed": 1, "arch": "bins"}'
run r_bins_1000 '{"steps": 1000, "seed": 0, "arch": "bins"}'
run r_bins_nofreeze '{"steps": 1300, "seed": 0, "arch": "bins", "freeze": false}'
run r_bins_exp15 '{"steps": 1300, "seed": 0, "arch": "bins", "exp": 1.5}'
run r_bins_lvlwide '{"steps": 1300, "seed": 0, "arch": "bins", "lvls": [0.08, 0.15, 0.3, 0.45]}'
echo QUEUE_DONE >> dev/ref_queue.log
