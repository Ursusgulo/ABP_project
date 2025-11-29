#include "lancoz.hpp"
#include "lancoz.cu"



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
    using T = float;
    Timings timings;

    // TODO inside of loop??
    SparseMatrixCRS <T> result(m*m, m*3-2);

  const unsigned int           n_tests = 20;
  const unsigned long long int n_repeat =
    repeat > 0 ? repeat : std::max(1UL, 100000000U / N);
  double best = 1e10, worst = 0, avg = 0;
  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();

      for (unsigned int rep = 0; rep < n_repeat; ++rep)
            lancoz_gpu<T>(N, m, &result, &timings);

      
      // Measure time of entire lancoz function
      const double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t1)
          .count();

      best  = std::min(best, time / n_repeat);
      worst = std::max(worst, time / n_repeat);
      avg += time / n_repeat;
    }

    // TODO 
    // This is the times of the parts from within the lancoz function
    // The timings struct holds the total times, calculate average same way as for total
    std::cout << "==== Benchmark results ====\n";
    std::cout << "HostToDevice: " << timings.h2d_s << " s\n";
    std::cout << "HostToDevice: " << timings.h2d_s << " s\n";
    std::cout << "SpMV avg:     " << timings.spmv_avg_s << " s\n";
    std::cout << "Lanczos loop:" << timings.lanczos_s << " s\n";
    std::cout << "Lanczos total:" << best << " s\n";
}

// int main(int argc, char **argv)
// {


//   long long          N           = -1;
//   long long          n_repeat    = 100;

//   if (argc < 3)
//     {
      
//         std::cout << "Error, expected odd number of common line arguments"
//                   << std::endl
//                   << "Expected line of the form" << std::endl
//                   << "-N 100 -repeat 100 -number double" << std::endl;
//       std::abort();
//     }

//   // parse from the command line
//   for (unsigned l = 1; l < argc; l += 2)
//     {
//       std::string option = argv[l];
//       if (option == "-N")
//         N = std::atoll(argv[l + 1]);
//       else if (option == "-repeat")
//         n_repeat = std::atoll(argv[l + 1]);
//       else
//         std::cout << "Unknown option " << option << " - ignored!" << std::endl;
//     }

    


// }