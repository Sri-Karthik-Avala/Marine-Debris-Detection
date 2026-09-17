until grep -q AB_DONE dev/ab.log 2>/dev/null; do sleep 20; done
for cfg in "bs128e3 0.5 3 640 128" "bs128e4 0.5 4 640 128"; do
  set -- $cfg
  conda run -n max --no-capture-output python -u dev/oof_det_md.py $1 $2 $3 $4 $5 2>&1 | grep -E "saved|Error" >> dev/det2_queue.log
  conda run -n max --no-capture-output python -u dev/det_eval_md.py dets $1 2>&1 | grep -E "DET|Error" >> dev/det2_queue.log
done
echo DET2_DONE >> dev/det2_queue.log
