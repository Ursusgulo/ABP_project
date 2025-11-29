#include "lancoz.cuh"




// const int N = 2; //size in one dimension
//     int N3 = N * N * N;
//     int nnz = N3 * 3 -2;
//     int m = 20 * N; 
//     using T = float;
//     SparseMatrixCRS <T> result(m*m, m*3-2); //EBBA FIX changed to N3 -> m*m and nnz -> m*3-2
//     lancoz_gpu<T>(N, m, &result);

void benchmark_triad(const unsigned long N, const long long repeat)
{
  int m = 20 * N; 
  if(m > N*N*N) {
      m = N*N*N;
  }
  using T = float;
  Timings gpu_timings;
  Timings cpu_timings;

    // TODO insidof loop??
  SparseMatrixCRS <T> result_gpu(m, m*3-2);
  SparseMatrixCRS <T> result_cpu(m, m*3-2);

  const unsigned int           n_tests = 20;
  const unsigned long long int n_repeat =
    repeat > 0 ? repeat : std::max(1UL, 100000000U / N);

  double best = 1e10, worst = 0, avg = 0;
  for (unsigned int t = 0; t < n_repeat; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      

      // for (unsigned int rep = 0; rep < n_repeat; ++rep)
      lancoz_gpu(N, m, &result_gpu, &gpu_timings);
      lancoz<T>(N, m, &result_cpu, &cpu_timings);

    }
   
  float spmv_avg_s_gpu = gpu_timings.spmv_s / (n_repeat * (m-1)); // TODO change to m?
  float h2d_avg_s = gpu_timings.h2d_s / (n_repeat);
  float spmv_avg_s_cpu = cpu_timings.spmv_s / (n_repeat * (m-1));


  // TODO 
  // This is the times of the parts from within the lancoz function
  // The timings struct holds the total times, calculate average same way as for total
  std::cout << "==== Benchmark results ====\n";
  std::cout << "HostToDevice: " << h2d_avg_s << " s\n";
  std::cout << "SpMV avg GPU: " << spmv_avg_s_gpu << " s\n";
  std::cout << "SpMV avg CPU: " << spmv_avg_s_cpu << " s\n";

  printf("Resulting Lancoz matrix gpu:\n");
  for(int i = 0; i < 10; i++) {
      std::cout << "Row " << i << ": ";
      for(int j = result_gpu.row_starts[i]; j < result_gpu.row_starts[i+1]; j++) {
          std::cout << "(" << result_gpu.col[j] << ", " << result_gpu.val[j] << ") ";
      }
      std::cout << std::endl;
  }
  printf("Resulting Lancoz matrix cpu:\n");
  for(int i = 0; i < 10; i++) {
      std::cout << "Row " << i << ": ";
      for(int j = result_cpu.row_starts[i]; j < result_cpu.row_starts[i+1]; j++) {
          std::cout << "(" << result_cpu.col[j] << ", " << result_cpu.val[j] << ") ";
      }
      std::cout << std::endl;
  }
}

int main(int argc, char **argv)
{
  long long          N           = -1;
  long long          n_repeat    = 100;

  if (argc < 3)
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
      else
        std::cout << "Unknown option " << option << " - ignored!" << std::endl;
    }


    benchmark_triad(N,n_repeat);

}