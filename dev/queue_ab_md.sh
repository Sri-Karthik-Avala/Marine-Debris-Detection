until grep -q EXIT dev/det_e2.log 2>/dev/null; do sleep 20; done
conda run -n max --no-capture-output python -u dev/apply_scorer_md.py r_bins raw50_500 2>&1 | grep -E "saved|Error" > dev/ab.log
conda run -n max --no-capture-output python -u dev/scorer_eval_md.py dev/cands_r_bins.pkl dev/scorer_raw50_500_on_r_bins.pkl 2>&1 | grep -E "AUC|honest|rule" >> dev/ab.log
conda run -n max --no-capture-output python -u dev/shared_md.py sh1800 '{"steps": 1800, "real_imgs": 5}' 2>&1 | grep -E "RESULT|Error|fold" >> dev/ab.log
conda run -n max --no-capture-output python -u dev/scorer_eval_md.py dev/cands_sh1800.pkl dev/scorer_sh1800.pkl 2>&1 | grep -E "AUC|honest|rule" >> dev/ab.log
echo AB_DONE >> dev/ab.log
