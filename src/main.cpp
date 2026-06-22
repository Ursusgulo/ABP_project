#include "lancoz.cuh"




// const int N = 2; //size in one dimension
//     int N3 = N * N * N;
//     int nnz = N3 * 3 -2;
//     int m = 20 * N; 
//     using T = float;
//     SparseMatrixCRS <T> result(m*m, m*3-2); //EBBA FIX changed to N3 -> m*m and nnz -> m*3-2
//     lancoz_gpu<T>(N, m, &result);

void benchmark_lancoz(const unsigned long N, const long long repeat, int gpu)
{
  int m = 20 * N; 
  if(m > N*N*N) {
      m = N*N*N;
  }
  using T = float;
  Timings timings;

    // TODO insidof loop??
  SparseMatrixCRS <T> result_gpu(m, m*3-2); // should be  N*N*N and N*N*N*3-2 ?
  SparseMatrixCRS <T> result_cpu(m, m*3-2);

  const unsigned int           n_tests = 20;
  const unsigned long long int n_repeat =
    repeat > 0 ? repeat : std::max(1UL, 100000000U / N);


  for (unsigned int t = 0; t < n_repeat; ++t)
    {
      if (gpu) lancoz_gpu(N, m, &result_gpu, &timings);
      else lancoz<T>(N, m, &result_cpu, &timings);
    }
   
  float spmv_avg_s = timings.spmv_s / (n_repeat * (m-1)); // TODO change to m?
  float h2d_avg_s = timings.h2d_s / (n_repeat);

  // N*N*N*(3*3) + N*N*N * 3 - 2*3   saknas parentes?
  float nnz_A = timings.nnz_a;
  float bytes_per_spmv = 

  float gbytes = 1.0e-9 * sizeof(float);
  float flops_per_spmv = nnz_A * 2; // 2 operations (mul + add) per non-zero
  float memops_per_spmv = nnz_A * 3 + 3 * N*N*N; // 3 memory ops (read val, read col, write res) per non-zero read
  float gflops = flops_per_spmv * 1.0e-9 / spmv_avg_s; // 7 flops per non-zero
  float bandwidth = memops_per_spmv * gbytes / spmv_avg_s; // in GB/s

  if(gpu)std::cout << N << ", " << m << ", " << gflops << ", " << bandwidth <<", " << h2d_avg_s << "\n";
  else std::cout << N << ", " << m << ", " << gflops << ", " << bandwidth <<"\n";


}

template <typename Number>
void benchmark_spmv(const unsigned long N, const long long repeat, int gpu)
{
  int m = 20 * N; 
  if(m > N*N*N) {
      m = N*N*N;
  }

  SparseMatrixCRS <Number> A;
  generate_laplacian3D<Number>(N, A);

  int new_N = A.N;

  CUDA_CHECK(cudaMalloc(&d_A_val, A.row_starts[new_N]*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_A_row_starts, (new_N + 1)*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_A_col, A.row_starts[new_N]*sizeof(int)));

  //allocate result matrix on device

  //allocate vectors on device
  CUDA_CHECK(cudaMalloc(&d_v, new_N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_w, new_N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp, new_N*sizeof(float)));

  //Copy Laplacian3D matrix to device
  const auto t1 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaMemcpy(d_A_val, A.val.data(), A.nnz*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_A_row_starts, A.row_starts.data(), (new_N + 1)*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_A_col, A.col.data(), A.nnz*sizeof(int), cudaMemcpyHostToDevice));
  const double host_to_dev_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - t1)
        .count();
  timings->h2d_s += host_to_dev_time;

  int n_blocks = (new_N + block_size - 1) / (block_size);

  // Measure time for spmv
  const unsigned long long int n_repeat =
    repeat > 0 ? repeat : std::max(1UL, 100000000U / N);

  CUDA_CHECK(cudaDeviceSynchronize());
  const auto t1 = std::chrono::steady_clock::now();
  for (unsigned int rep = 0; rep < n_repeat; ++rep)
  {
    d_compute_spmv<<<n_blocks, block_size>>>(new_N, d_A_row_starts, d_A_col, d_A_val, d_v, d_w);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  const double time =
    std::chrono::duration_cast<std::chrono::duration<double>>(
      std::chrono::steady_clock::now() - t1)
      .count();
  
  float spmv_avg_s = time / n_repeat;
  float nnz_A = A.nnz;
  // float bytes_per_spmv = nnz_A( 2 * sizeof(float) + sizeof(int)) + N*N*N(sizeof(float) + 2 * sizeof(int)); //new
  float gbytes = 1.0e-9 * sizeof(float);
  float flops_per_spmv = nnz_A * 2; // 2 operations (mul + add) per non-zero
  float memops_per_spmv = nnz_A * 3 + 3 * N*N*N; // 3 memory ops (read val, read col, write res) per non-zero read
  float gflops = flops_per_spmv * 1.0e-9 / spmv_avg_s; // 7 flops per non-zero
  float bandwidth = memops_per_spmv * gbytes / spmv_avg_s; // in GB/s

  if(gpu)std::cout << N << ", " << m << ", " << gflops << ", " << bandwidth <<", " << h2d_avg_s << "\n";
  else std::cout << N << ", " << m << ", " << gflops << ", " << bandwidth <<"\n";


}

int main(int argc, char **argv)
{
  long long          N           = -1;
  long long          n_repeat    = 100;
  int                gpu         = 0;

  if (argc < 4)
    {
      
        std::cout << "Error, 2 arguments"
                  << std::endl
                  << "Expected line of the form" << std::endl
                  << "-N 100 -repeat 100 -number double" << std::endl;
      std::abort();
    }

  // parse from the command line
  for (unsigned l = 1; l < argc; l += 2)
    {
      std::string option = argv[l];
      if (option == "-N")
        N = std::atoll(argv[l + 1]);
      else if (option == "-repeat")
        n_repeat = std::atoll(argv[l + 1]);
      else if (option == "-gpu")
        gpu = std::atoi(argv[l + 1]);
      else
        std::cout << "Unknown option " << option << " - ignored!" << std::endl;
    }

    if (gpu) std::cout << "N, m, glfops, bandwitdh(GB/S), h2d\n";
    else std::cout << "N, m, glfops, bandwitdh(GB/S)\n";

    if (N == -1)
        for (unsigned long long NN = 8; NN < 160; NN += 8)
        {
            benchmark_lancoz(NN,n_repeat, gpu);
        }
    else
        {
            benchmark_lancoz(N,n_repeat, gpu);
        }

}