import sys; sys.path.insert(0, '.')
import torch
import solution as S
S.DEVICE = torch.device('cuda')
sys.argv = ['solution.py', '.', 'working/submission.csv']
S.main()
