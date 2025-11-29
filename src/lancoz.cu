
#include "lancoz.cuh"



#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

__global__
void d_compute_spmv(const int N,
                             const int *row_starts,
                             const int *column_indices,
                             const float *values,
                             const float *x, 
                             float *y)
{
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row < N)
  {
    float sum = 0;
    for (int idx = row_starts[row]; idx < row_starts[row + 1]; ++idx)
      sum += values[idx] * x[column_indices[idx]];
    y[row] = sum;
  }
}

__global__ 
void new_vector_v(const int N, const float *w, const float beta, float *v, int counter) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < N) {
        if (beta == 0.f) {
            v[idx] = (idx == counter) ? 1.f : 0.f;
        } else {
            v[idx] = w[idx] / (beta);
        }
    }
}



void lancoz_gpu(const int N, const int m, SparseMatrixCRS <float> *result, Timings *timings) {    
    int *d_A_row_starts;
    int *d_A_col;
    float *d_A_val;
    float *d_v, *d_w, *d_tmp;
    float alpha, h_tmp;
    double spmv_total_time = 0;
    float beta = 0;

    SparseMatrixCRS <float> A;
    generate_laplacian3D<float>(N, A);

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


    cublasHandle_t handle;
    cublasCreate(&handle);
    int n_blocks = (new_N + block_size - 1) / (block_size);
    float axpy_scalar = -1;

    new_vector_v<<<n_blocks, block_size>>>(new_N, d_w, beta, d_v, 0);

    d_compute_spmv<<<n_blocks, block_size>>>(new_N, d_A_row_starts, d_A_col, d_A_val, d_v, d_w);


    cublasSdot(
        handle,
        new_N,
        d_w, 1,
        d_v, 1,
        &alpha
    );
    


    h_tmp = -alpha;

    cublasSaxpy(
        handle,
        new_N,
        &h_tmp,
        d_v, 1,
        d_w, 1
    );
    
    // store in result matrix
    result->row_starts[0] = 0;
    result->val[0] = alpha;
    result->col[0] = 0;
    result->row_starts[1] = 2;

    
    for(int j = 1; j < m; j++) {

        cublasSdot(
            handle,
            new_N,
            d_w, 1,
            d_w, 1,
            &beta
        );
        beta = std::sqrt(beta);
        beta = is_zero(beta) ? 0.f : beta;

        CUDA_CHECK(cudaDeviceSynchronize());

        cublasScopy(
            handle,
            new_N,                
            d_v,               
            1,        
            d_tmp,     
            1         
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        cublasSscal(
            handle,   
            new_N,                 
            &beta,     
            d_tmp,         
            1         
        );
        
        new_vector_v<<<n_blocks, block_size>>>(new_N, d_w, beta, d_v, j);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        const auto spmv_start = std::chrono::steady_clock::now();
        d_compute_spmv<<<n_blocks, block_size>>>(new_N, d_A_row_starts, d_A_col, d_A_val, d_v, d_w);
        CUDA_CHECK(cudaDeviceSynchronize());
        const double spmv_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - spmv_start)
          .count();
        spmv_total_time += spmv_time;

        cublasSaxpy(
            handle,
            new_N,
            &axpy_scalar,
            d_tmp, 1,
            d_w, 1
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        cublasSdot(
            handle,
            new_N,
            d_w, 1,
            d_v, 1,
            &alpha
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        if(!is_zero(alpha)) {
            result->val[result->row_starts[j]+1] = alpha;
            result->col[result->row_starts[j] +1 ] = j ;
        }
        if(!is_zero(beta)) {
            result->val[result->row_starts[j]-1] = beta;
            result->val[result->row_starts[j]] = beta;
            result->col[result->row_starts[j]-1] = j;
            result->col[result->row_starts[j]] = j - 1;
        }



        result->row_starts[j + 1] = result->row_starts[j] + 3;
        if(j == new_N -1) {
            result->row_starts[j + 1] = result->row_starts[j] + 2;
        }

        h_tmp = -alpha;
        cublasSaxpy(
            handle,
            new_N,
            &h_tmp,
            d_v, 1,
            d_w, 1
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }    
    timings->spmv_s += spmv_total_time;
    CUDA_CHECK(cudaFree(d_A_val));
    CUDA_CHECK(cudaFree(d_A_row_starts));
    CUDA_CHECK(cudaFree(d_A_col));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_tmp));
    cublasDestroy(handle);
}

// int main() {
//     const int N = 2; //size in one dimension
//     // int N3 = N * N * N;
//     //int nnz = N3 * 3 -2;
//     int m = 20 * N; 
//     if(m > N*N*N) {
//         m = N*N*N;
//     }
//     using T = float;
//     Timings timings;
//     SparseMatrixCRS <float> result(m, m*3-2); //floatODO time 
//     lancoz_gpu(N, m, &result, &timings);

//     printf("Resulting Lancoz matrix:\n");
//     for(int i = 0; i < m; i++) {
//         std::cout << "Row " << i << ": ";
//         for(int j = result.row_starts[i]; j < result.row_starts[i+1]; j++) {
//             std::cout << "(" << result.col[j] << ", " << result.val[j] << ") ";
//         }
//         std::cout << std::endl;
//     }
//     printf("result->row_starts[8]: %d\n", result.row_starts[7]);
//     printf("result->val[result->row_starts[7]+1]: %f\n", result.val[result.row_starts[7]+1]);
//     return 0;
// }