import sys, time
src = open('dev/refine_md.py').read().split("print('train time'")[0]
R = sys.argv[1]
sys.argv = ['x', 'cpu', '21', R]
t = time.time()
exec(compile(src, 'refine_bench', 'exec'))
print('R', R, 'total incl load', time.time() - t, 'train-only', time.time() - t0)
t1 = time.time()
for _ in range(10): sample_batch(tri)
print('sampling per step', (time.time() - t1) / 10)
