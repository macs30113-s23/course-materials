from numba import vectorize, cuda
import numpy as np
import time

@vectorize(['i4(f4, f4)'], target='cuda')
def in_circle(x, y):
  '''
  Vectorized function takes in x, y coordinates (float32, float32) within an
  array and returns a boolean indication of whether these values are (1) or
  are not (0) in the unit circle (int32). All computation is done on the GPU.
  '''
  in_circle = x**2 + y**2 <= 1
  return in_circle

@cuda.reduce
def gpu_sum(a, b):
  '''
  Sums values in an array together on the GPU.

  `numba` comes with a built-in `reduce` algorithm
  '''
  return a + b

if __name__ == '__main__':
  n_runs = 10 ** 8
  ran = np.random.uniform(low=-1, high=1, size=(2, n_runs)).astype(np.float32)
  x, y = ran[0], ran[1]

  times = []
  for _ in range(100):
    t0 = time.time()
    in_circle_dev = cuda.device_array(shape=(n_runs,),
                                    dtype=np.float32)
    in_circle(x, y, out=in_circle_dev)
    result = 4 * gpu_sum(in_circle_dev) / n_runs
    t1 = time.time()
    times.append(t1 - t0)

  print('avg. time (100 runs): ', sum(times) / 100)
