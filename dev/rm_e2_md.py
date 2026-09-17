p = 'dev/queue_det_md.sh'; s = open(p, newline='').read()
assert '"nms06 0.6 3 640" "e2 0.5 2 640"' in s
s = s.replace('"nms06 0.6 3 640" "e2 0.5 2 640"', '"nms06 0.6 3 640" "e2x 0.5 2 640"')
open(p, 'w', newline='').write(s); print('ok')
