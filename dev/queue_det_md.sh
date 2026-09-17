until grep -q QUEUE_DONE dev/ref_queue.log 2>/dev/null; do sleep 20; done
conda run -n max --no-capture-output python -u dev/det_eval_md.py dets base 2>&1 | grep -E "DET|Error" >> dev/det_queue.log
for cfg in "nms06 0.6 3 640"; do
  set -- $cfg
  conda run -n max --no-capture-output python -u dev/oof_det_md.py $1 $2 $3 $4 2>&1 | grep -E "saved|Error" >> dev/det_queue.log
  conda run -n max --no-capture-output python -u dev/det_eval_md.py dets $1 2>&1 | grep -E "DET|Error" >> dev/det_queue.log
done
echo DET_DONE >> dev/det_queue.log
