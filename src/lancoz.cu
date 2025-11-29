#include "lancoz.hpp"
#include <cublas_v2.h>
#include <cmath>
#include <chrono>

const int block_size = 512;


#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

template <typename T>
__global__
void d_compute_spmv(const std::size_t N,
                             const std::size_t *row_starts,
                             const int *column_indices,
                             const T *values,
                             const T *x, 
                             T *y)
{
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row < N)
  {
    T sum = 0;
    for (std::size_t idx = row_starts[row]; idx < row_starts[row + 1]; ++idx)
      sum += values[idx] * x[column_indices[idx]];
    y[row] = sum;
  }
  
}


template <typename T>
__global__ 
void d_scale_vector(const std::size_t N, const T *scalar, const T *x, T *y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        y[idx] = -(*scalar)  * x[idx];
    }
}



template <typename T>
__global__ 
void new_vector_v(const std::size_t N, const T *w, const T *beta, T *v, int counter) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < N) {
        if (*beta == 0) {
            v[idx] = (idx == counter) ? 1.0 : 0.0;
        } else {
            v[idx] = w[idx] / (*beta);
        }
    }
}

struct Timings { 
    float h2d_s = 0.0;
    float spmv_avg_s = 0.0;
    float lanczos_s = 0.0;
 };

template <typename T>
void lancoz_gpu(const std::size_t N, const std::size_t m, SparseMatrixCRS <T> *result, Timings *timings) {    
    std::size_t *d_A_row_starts;
    int *d_A_col;
    T *h_v, *h_w;
    T *d_A_val, d_A_N;
    T *d_v, *d_w, *d_tmp;
    T alpha, h_tmp;
    int spmv_total_time;

    SparseMatrixCRS <T> A;
    generate_laplacian3D<T>(N, A);

    std::size_t new_N = A.N;

    h_v = new T[new_N];
    generate_unit_vector<T>(new_N, h_v, 0);


    // TODO move first iteration to device?
    
    //Do iteration one on host
    h_w = new T[new_N];
    //Ax

    compute_spmv<T>(new_N, &A, h_v, h_w);

    //alpha = w*v
    alpha = dot_product(new_N, h_w, h_v);
    

    //w' ||Ax - alpha*v||
    //beta = ||w'||
    T beta = gemv_norm<T>(new_N, -alpha, h_w, h_v);
    
    // store in result matrix
    result->row_starts[0] = 0;
    result->val[0] = alpha;
    result->col[0] = 0;
    result->row_starts[1] = 2;


    //move data to device and do remaining iterations there
    //Allocate Laplacian3D matrix on device

    cudaMalloc(&d_A_val, A.row_starts[new_N]*sizeof(T));
    cudaMalloc(&d_A_row_starts, (new_N + 1)*sizeof(std::size_t));
    cudaMalloc(&d_A_col, A.row_starts[new_N]*sizeof(int));

    //allocate result matrix on device

    //allocate vectors on device
    cudaMalloc(&d_v, new_N*sizeof(T));
    cudaMalloc(&d_w, new_N*sizeof(T));
    cudaMalloc(&d_tmp, new_N*sizeof(T));

    //Copy Laplacian3D matrix to device
    const auto t1 = std::chrono::steady_clock::now();
    cudaMemcpy(d_A_val, A.val.data(), A.nnz*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_row_starts, A.row_starts.data(), (new_N + 1)*sizeof(std::size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_col, A.col.data(), A.nnz*sizeof(int), cudaMemcpyHostToDevice);

    //copy vectors to device   
    cudaMemcpy(d_v, h_v, new_N*sizeof(T), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_w, h_w, new_N*sizeof(T), cudaMemcpyHostToDevice);
    const double host_to_dev_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t1)
          .count();

    timings->h2d_s += host_to_dev_time;
    cublasHandle_t handle;
    cublasCreate(&handle);
    int n_blocks = (new_N + block_size - 1) / (block_size);

    T axpy_scalar = -1;
    const auto loop_start = std::chrono::steady_clock::now();
    for(int j = 1; j < m; j++) {

        //-b * v_{j-1}
        T scale_beta = -beta;
        cublasScopy(
            handle,
            new_N,                
            d_v,               
            1,        
            d_tmp,     
            1         
        );
        cublasSscal(
            handle,   
            new_N,                 
            &scale_beta,     
            d_tmp,         
            1         
        );
        
        //TODO kanske flytta ut logiken i new_vector_v till hosten
        // if(is_zero(beta)) {
        //     new_vector_v<T><<<n_blocks, block_size>>>(new_N, d_v, j);
        // } else {
        //     scale_beta = 1/beta;
        //     scale_vector<T><<<n_blocks, block_size>>>(new_N, &scale_beta, d_w, d_v);
        // }
        new_vector_v<T><<<n_blocks, block_size>>>(new_N, d_w, &beta, d_v, j);
        // new_vector_v<T><<<n_blocks, block_size>>>(new_N, d_w, T beta, d_v, j); // EBBAS FIX
        cudaDeviceSynchronize();
        
        const auto spmv_start = std::chrono::steady_clock::now();
        d_compute_spmv<T><<<n_blocks, block_size>>>(new_N, d_A_row_starts, d_A_col, d_A_val, d_v, d_w);
        cudaDeviceSynchronize();
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

        // gemv<T><<<n_blocks, block_size>>>(new_N, &axpy_scalar, d_w, d_tmp);

        cublasSdot(
            handle,
            new_N,
            d_w, 1,
            d_v, 1,
            &alpha
        );

        result->val[result->row_starts[j]+1] = alpha;
        result->val[result->row_starts[j]-1] = beta;
        result->val[result->row_starts[j]] = beta;

        result->col[result->row_starts[j]-1] = j;
        result->col[result->row_starts[j]] = j - 1;
        result->col[result->row_starts[j] +1 ] = j ;

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

        cublasSdot(
            handle,
            new_N,
            d_w, 1,
            d_w, 1,
            &beta
        );
        beta = std::sqrt(beta);

    }    
    cudaDeviceSynchronize();
    const double lanczos_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t1)
          .count();
    timings->lanczos_s += lanczos_time;

    // Average SpMV time and add to struct
    timings -> spmv_avg_s = spmv_total_time / (float)(m-1);

    cudaFree(d_A_val);
    cudaFree(d_A_row_starts);
    cudaFree(d_A_col);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_tmp);
    cublasDestroy(handle);
    delete [] h_v;
    delete [] h_w;
}

// int main() {
//     const int N = 2; //size in one dimension
//     // int N3 = N * N * N;
//     //int nnz = N3 * 3 -2;
//     int m = 20 * N; 
//     using T = float;
//     Timings timings;
//     SparseMatrixCRS <T> result(m*m, m*3-2); //TODO time 
//     lancoz_gpu<T>(N, m, &result, &timings);

//     // printf("Resulting Lancoz matrix:\n");
//     // for(int i = 0; i < m; i++) {
//     //     std::cout << "Row " << i << ": ";
//     //     for(int j = result.row_starts[i]; j < result.row_starts[i+1]; j++) {
//     //         std::cout << "(" << result.col[j] << ", " << result.val[j] << ") ";
//     //     }
//     //     std::cout << std::endl;
//     // }

//     std::cout << "==== Benchmark results ====\n";
//     std::cout << "HostToDevice: " << timings.h2d_s << " s\n";
//     std::cout << "SpMV avg:     " << timings.spmv_avg_s << " s\n";
//     std::cout << "Lanczos total:" << timings.lanczos_s << " s\n";

//     return 0;
// }