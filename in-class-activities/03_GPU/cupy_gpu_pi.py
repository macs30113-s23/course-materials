import cupy as cp
from cupyx.profiler import benchmark

# Generate random numbers on GPU 
n_runs = 10 ** 8
x, y = cp.random.uniform(low=-1, high=1, size=(2, n_runs))

def cupy_pi_est(x, y):
  # Compute on GPU via CuPy arrays:
  result = x ** 2 + y ** 2 <= 1
  n_in_circle = cp.sum(result)
  pi = 4 * n_in_circle / n_runs

  # Get pi back to host CPU from GPU:
  pi_cpu = pi.get()
  
  return pi_cpu

print(benchmark(cupy_pi_est, (x, y), n_repeat=100))
