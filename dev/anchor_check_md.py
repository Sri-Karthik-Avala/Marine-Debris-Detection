import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, torch
import solution as S
train = pd.read_csv('train.csv').iloc[:50]
b = [S.clean_boxes(np.concatenate([S.parse_boxes(a), S.parse_boxes(c)])) for a, c in zip(train.logged_boxes, train.new_boxes)]
a = S.anchor_sizes(b); m = S.build_model(a)
print('anchors', a, m.rpn.anchor_generator.sizes, 'rpn head anchors/loc', m.rpn.anchor_generator.num_anchors_per_location())
