// #include "lancoz.cuh"
// #include <cmath>



// #define CUDA_CHECK(call)                                                     \
//     do {                                                                     \
//         cudaError_t err = call;                                              \
//         if (err != cudaSuccess) {                                            \
//             fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
//                     __FILE__, __LINE__, cudaGetErrorString(err));            \
//             exit(EXIT_FAILURE);                                              \
//         }                                                                    \
//     } while (0)

// #define CHECK_CUBLAS(status) \
//     if (status != CUBLAS_STATUS_SUCCESS) { \
//         fprintf(stderr, "cuBLAS error: %d\n", status); \
//         exit(EXIT_FAILURE); \
//     }

// __global__
// void d_compute_spmv(const int N,
//                              const int *row_starts,
//                              const int *column_indices,
//                              const float *values,
//                              const float *x, 
//                              float *y)
// {
//   int row = threadIdx.x + blockIdx.x * blockDim.x;
//   if (row < N)
//   {
//     float sum = 0;
//     for (int idx = row_starts[row]; idx < row_starts[row + 1]; ++idx)
//       sum += values[idx] * x[column_indices[idx]];
//     y[row] = sum;
//   }
  
// }





// __global__ 
// void new_vector_v(const int N, const float *w, const float *beta, float *v, int counter) {
//     int idx = threadIdx.x + blockDim.x*blockIdx.x;
//     if (idx < N) {
//         if (*beta == 0) {
//             v[idx] = (idx == counter) ? 1.0 : 0.0;
//         } else {
//             v[idx] = w[idx] / (*beta);
//         }
//     }
// }




// void lancoz_gpu(const int N, const int m, SparseMatrixCRS <float> *result, Timings *timings) {    
//     int *d_A_row_starts;
//     int *d_A_col;
//     float *h_v, *h_w;
//     float *d_A_val;
//     float *d_v, *d_w, *d_tmp;
//     float alpha, h_tmp;
//     int spmv_total_time;
//     float axpy_scalar = -1;


//     SparseMatrixCRS <float> A;
//     generate_laplacian3D<float>(N, A);

//     int new_N = A.N;
//     h_v = new float[new_N];
//     generate_unit_vector<float>(new_N, h_v, 0);
//     h_w = new float[new_N];

//     compute_spmv<float>(new_N, &A, h_v, h_w);
//     //alpha = w*v
//     alpha = dot_product(new_N, h_w, h_v);
    
//     //w' ||Ax - alpha*v||
//     //beta = ||w'||
//     float beta = gemv_norm<float>(new_N, -alpha, h_w, h_v);
    
//     // store in result matrix
//     result->row_starts[0] = 0;
//     result->val[0] = alpha;
//     result->col[0] = 0;
//     result->row_starts[1] = 2;

//     //move data to device and do remaining iterations there
//     CUDA_CHECK(cudaMalloc(&d_A_val, A.row_starts[new_N]*sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_A_row_starts, (new_N + 1)*sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_A_col, A.row_starts[new_N]*sizeof(int)));

//     CUDA_CHECK(cudaMalloc(&d_v, new_N*sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_w, new_N*sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_tmp, new_N*sizeof(float)));

//     const auto t1 = std::chrono::steady_clock::now();
//     CUDA_CHECK(cudaMemcpy(d_A_val, A.val.data(), A.nnz*sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_A_row_starts, A.row_starts.data(), (new_N + 1)*sizeof(int), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_A_col, A.col.data(), A.nnz*sizeof(int), cudaMemcpyHostToDevice));

//     //copy vectors to device   
//     CUDA_CHECK(cudaMemcpy(d_v, h_v, new_N*sizeof(float), cudaMemcpyHostToDevice)); 
//     CUDA_CHECK(cudaMemcpy(d_w, h_w, new_N*sizeof(float), cudaMemcpyHostToDevice));

//     const double host_to_dev_time =
//         std::chrono::duration_cast<std::chrono::duration<double>>(
//           std::chrono::steady_clock::now() - t1)
//           .count();
//     timings->h2d_s += host_to_dev_time;

//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     int n_blocks = (new_N + block_size - 1) / (block_size);

//     // First iteration (j = 0)

//     for(int j = 1; j < m; j++) {


//         //-b * v_{j-1}
//         float scale_beta = -beta;

//         CHECK_CUBLAS(cublasScopy(
//             handle,
//             new_N,                
//             d_v,               
//             1,        
//             d_tmp,     
//             1         
//         ));

//         CHECK_CUBLAS(cublasSscal(
//             handle,   
//             new_N,                 
//             &scale_beta,     
//             d_tmp,         
//             1         
//         ));
        
//         new_vector_v<<<n_blocks, block_size>>>(new_N, d_w, &beta, d_v, j);
//         // new_vector_v<float><<<n_blocks, block_size>>>(new_N, d_w, float beta, d_v, j); // EBBAS FIX
//         CUDA_CHECK(cudaDeviceSynchronize());
//         const auto spmv_start = std::chrono::steady_clock::now();
//         d_compute_spmv<<<n_blocks, block_size>>>(new_N, d_A_row_starts, d_A_col, d_A_val, d_v, d_w);
//         CUDA_CHECK(cudaDeviceSynchronize());
//         const double spmv_time =
//         std::chrono::duration_cast<std::chrono::duration<double>>(
//           std::chrono::steady_clock::now() - spmv_start)
//           .count();
//         spmv_total_time += spmv_time;


//         CHECK_CUBLAS(cublasSaxpy(
//             handle,
//             new_N,
//             &axpy_scalar,
//             d_tmp, 1,
//             d_w, 1
//         ));

//         // gemv<float><<<n_blocks, block_size>>>(new_N, &axpy_scalar, d_w, d_tmp);

//         CHECK_CUBLAS(cublasSdot(
//             handle,
//             new_N,
//             d_w, 1,
//             d_v, 1,
//             &alpha
//         ));

//         result->val[result->row_starts[j]+1] = alpha;
//         result->val[result->row_starts[j]-1] = beta;
//         result->val[result->row_starts[j]] = beta;

//         result->col[result->row_starts[j]-1] = j;
//         result->col[result->row_starts[j]] = j - 1;
//         result->col[result->row_starts[j] +1 ] = j ;

//         result->row_starts[j + 1] = result->row_starts[j] + 3;
//         if(j == new_N -1) {
//             result->row_starts[j + 1] = result->row_starts[j] + 2;
//         }

//         h_tmp = -alpha;
//         CHECK_CUBLAS(cublasSaxpy(
//             handle,
//             new_N,
//             &h_tmp,
//             d_v, 1,
//             d_w, 1
//         ));

//         CHECK_CUBLAS(cublasSdot(
//             handle,
//             new_N,
//             d_w, 1,
//             d_w, 1,
//             &beta
//         ));
//         beta = std::sqrt(beta);

//     }    
//     cudaDeviceSynchronize();
//     const double lanczos_time =
//         std::chrono::duration_cast<std::chrono::duration<double>>(
//           std::chrono::steady_clock::now() - t1)
//           .count();
//     timings->lanczos_s += lanczos_time;

//     // Add spmv to struct
//     timings->spmv_s += spmv_total_time;
    

//     cudaFree(d_A_val);
//     cudaFree(d_A_row_starts);
//     cudaFree(d_A_col);
   
//     cudaFree(d_v);
//     cudaFree(d_w);
//     cudaFree(d_tmp);
//     cublasDestroy(handle);
//     delete [] h_v;
//     delete [] h_w;
// }


// int main() {
//     const int N = 10; //size in one dimension
//     // int N3 = N * N * N;
//     //int nnz = N3 * 3 -2;
//     int m = 20 * N; 
//     Timings timings;
//     SparseMatrixCRS <float> result(m*m, m*3-2); //TODO time 
//     lancoz_gpu(N, m, &result, &timings);

//     printf("Resulting Lancoz matrix:\n");
//     for(int i = 0; i < m; i++) {
//         std::cout << "Row " << i << ": ";
//         for(int j = result.row_starts[i]; j < result.row_starts[i+1]; j++) {
//             std::cout << "(" << result.col[j] << ", " << result.val[j] << ") ";
//         }
//         std::cout << std::endl;
//     }

//     std::cout << "==== Benchmark results ====\n";
//     std::cout << "HostToDevice: " << timings.h2d_s << " s\n";
//     std::cout << "SpMV avg:     " << timings.spmv_s << " s\n";
//     std::cout << "Lanczos total:" << timings.lanczos_s << " s\n";

//     return 0;
// }