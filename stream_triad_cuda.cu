
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>


const int block_size = 512;
const int chunk_size = 1;

__global__ void compute_triad(const int    N,
                              const float  a,
                              const float *x,
                              const float *y,
                              float *      z)
{
  const int idx_base = threadIdx.x + blockIdx.x * (blockDim.x * chunk_size);
  for (unsigned int i = 0; i < chunk_size; ++i)
    {
      const int idx = idx_base + i * block_size;
      if (idx < N)
        z[idx] = a * x[idx] + y[idx];
    }
}


__global__ void set_vector(const int N, const float val, float *x)
{
  const int idx_base = threadIdx.x + blockIdx.x * (blockDim.x * chunk_size);
  for (unsigned int i = 0; i < chunk_size; ++i)
    {
      const int idx = idx_base + i * block_size;
      if (idx < N)
        x[idx] = val;
    }
}


// Run the actual benchmark
void benchmark_triad(const bool        align,
                     const std::size_t N,
                     const long long   repeat)
{
  float *v1, *v2, *v3;
  // allocate memory on the device
  cudaMalloc(&v1, N * sizeof(float));
  cudaMalloc(&v2, N * sizeof(float));
  cudaMalloc(&v3, N * sizeof(float));

  const unsigned int n_blocks = (N + block_size - 1) / block_size;

  set_vector<<<n_blocks, block_size>>>(N, 17.f, v1);
  set_vector<<<n_blocks, block_size>>>(N, 42.f, v2);
  set_vector<<<n_blocks, block_size>>>(N, 0.f, v3);

  std::vector<float> result_host(N);

  const unsigned int           n_tests = 20;
  const unsigned long long int n_repeat =
    repeat > 0 ? repeat : std::max(1UL, 100000000U / N);
  double best = 1e10, worst = 0, avg = 0;
  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();

      for (unsigned int rep = 0; rep < n_repeat; ++rep)
        compute_triad<<<n_blocks, block_size>>>(N, 13.f, v1, v2, v3);

      cudaDeviceSynchronize();
      // measure the time by taking the difference between the time point
      // before starting and now
      const double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t1)
          .count();

      best  = std::min(best, time / n_repeat);
      worst = std::max(worst, time / n_repeat);
      avg += time / n_repeat;
    }

  // Copy the result back to the host
  cudaMemcpy(result_host.data(), v3, N * sizeof(float), cudaMemcpyDeviceToHost);
  if ((result_host[0] + result_host[N - 1]) != 526.f)
    std::cout << "Error in computation, got "
              << (result_host[0] + result_host[N - 1]) << " instead of 526"
              << std::endl;

  // Free the memory on the device
  cudaFree(v1);
  cudaFree(v2);
  cudaFree(v3);

  std::cout << "STREAM triad of size " << std::setw(8) << N
            << " : min/avg/max: " << std::setw(11) << best << " "
            << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
            << " seconds or " << std::setw(8) << 1e-6 * N / best
            << " MUPD/s or " << std::setw(8)
            << 1e-9 * 3 * sizeof(float) * N / best << " GB/s" << std::endl;
}

int main(int argc, char **argv)
{
  if (argc % 2 == 0)
    {
      std::cout << "Error, expected odd number of common line arguments"
                << std::endl
                << "Expected line of the form" << std::endl
                << "-min 100 -max 1e8 -repeat -1" << std::endl;
      std::abort();
    }

  long N_min  = 8;
  long N_max  = -1;
  bool align  = false;
  long repeat = -1;
  // parse from the command line
  for (int l = 1; l < argc; l += 2)
    {
      std::string option = argv[l];
      if (option == "-min")
        N_min = static_cast<long>(std::stod(argv[l + 1]));
      else if (option == "-max")
        N_max = static_cast<long>(std::stod(argv[l + 1]));
      else if (option == "-repeat")
        repeat = std::atoll(argv[l + 1]);
      else if (option == "-align")
        align = std::atoi(argv[l + 1]);
      else
        std::cout << "Unknown option " << option << " - ignored!" << std::endl;
    }
  if (N_min < 1)
    {
      std::cout << "Expected positive size for min argument, got " << N_min
                << std::endl;
      return 0;
    }

  if (N_max < N_min)
    N_max = N_min;

  for (long n = N_min; n <= N_max; n = (1 + n * 1.1))
    {
      // round up to nearest multiple of 8
      n = (n + 7) / 8 * 8;
      benchmark_triad(align, n, repeat);
    }

  return 0;
}
