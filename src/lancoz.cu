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
void d_compute_spmv(const int N,
                             const int *row_starts,
                             const int *column_indices,
                             const T *values,
                             const T *x, 
                             T *y)
{
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row < N)
  {
    T sum = 0;
    for (int idx = row_starts[row]; idx < row_starts[row + 1]; ++idx)
      sum += values[idx] * x[column_indices[idx]];
    y[row] = sum;
  }
}






template <typename T>
__global__ 
void new_vector_v(const int N, const T *w, const T beta, T *v, int counter) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < N) {
        if (fabs(beta) < 1e-6) {
            v[idx] = (idx == counter) ? 1.0 : 0.0;
        } else {
            v[idx] = w[idx] / (beta);
        }
    }
}

struct Timings { 
    float h2d_s = 0.0;
    float spmv_avg_s = 0.0;
    float lanczos_s = 0.0;
 };

template <typename T>
void lancoz_gpu(const int N, const int m, SparseMatrixCRS <T> *result, Timings *timings) {    
    int *d_A_row_starts;
    int *d_A_col;
    T *d_A_val, d_A_N;
    T *d_v, *d_w, *d_tmp;
    T alpha, h_tmp;
    int spmv_total_time;
    T beta = 0;

    SparseMatrixCRS <T> A;
    generate_laplacian3D<T>(N, A);

    int new_N = A.N;



    // TODO move first iteration to device?
    
    //Do iteration one on host
    //Ax




    //move data to device and do remaining iterations there
    //Allocate Laplacian3D matrix on device

    CUDA_CHECK(cudaMalloc(&d_A_val, A.row_starts[new_N]*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_A_row_starts, (new_N + 1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_A_col, A.row_starts[new_N]*sizeof(int)));

    //allocate result matrix on device

    //allocate vectors on device
    CUDA_CHECK(cudaMalloc(&d_v, new_N*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_w, new_N*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tmp, new_N*sizeof(T)));

    //Copy Laplacian3D matrix to device
    const auto t1 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(d_A_val, A.val.data(), A.nnz*sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_row_starts, A.row_starts.data(), (new_N + 1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_col, A.col.data(), A.nnz*sizeof(int), cudaMemcpyHostToDevice));


    cublasHandle_t handle;
    cublasCreate(&handle);
    int n_blocks = (new_N + block_size - 1) / (block_size);
    T axpy_scalar = -1;

    new_vector_v<T><<<n_blocks, block_size>>>(new_N, d_w, beta, d_v, 0);

    // compute_spmv<T>(new_N, &A, h_v, h_w);
    d_compute_spmv<T><<<n_blocks, block_size>>>(new_N, d_A_row_starts, d_A_col, d_A_val, d_v, d_w);

    //alpha = w*v
    // alpha = dot_product(new_N, h_w, h_v);
    cublasSdot(
        handle,
        new_N,
        d_w, 1,
        d_v, 1,
        &alpha
    );
    
    //w' ||Ax - alpha*v||
    //beta = ||w'||
    // T beta = gemv_norm<T>(new_N, -alpha, h_w, h_v);
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

    
    // store in result matrix
    result->row_starts[0] = 0;
    result->val[0] = alpha;
    result->col[0] = 0;
    result->row_starts[1] = 2;


    for(int j = 1; j < m; j++) {

        T scale_beta = -beta;
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
            &scale_beta,     
            d_tmp,         
            1         
        );
        
        new_vector_v<T><<<n_blocks, block_size>>>(new_N, d_w, beta, d_v, j);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        d_compute_spmv<T><<<n_blocks, block_size>>>(new_N, d_A_row_starts, d_A_col, d_A_val, d_v, d_w);
        CUDA_CHECK(cudaDeviceSynchronize());

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
        CUDA_CHECK(cudaDeviceSynchronize());

        cublasSaxpy(
            handle,
            new_N,
            &h_tmp,
            d_v, 1,
            d_w, 1
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        cublasSdot(
            handle,
            new_N,
            d_w, 1,
            d_w, 1,
            &beta
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        beta = std::sqrt(beta);
        CUDA_CHECK(cudaDeviceSynchronize());
    }    

    CUDA_CHECK(cudaFree(d_A_val));
    CUDA_CHECK(cudaFree(d_A_row_starts));
    CUDA_CHECK(cudaFree(d_A_col));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_tmp));
    cublasDestroy(handle);
}

int main() {
    const int N = 2; //size in one dimension
    // int N3 = N * N * N;
    //int nnz = N3 * 3 -2;
    int m = 20 * N; 
    if(m > N*N*N) {
        m = N*N*N;
    }
    using T = float;
    Timings timings;
    SparseMatrixCRS <T> result(m, m*3-2); //TODO time 
    lancoz_gpu<T>(N, m, &result, &timings);

    printf("Resulting Lancoz matrix:\n");
    for(int i = 0; i < m; i++) {
        std::cout << "Row " << i << ": ";
        for(int j = result.row_starts[i]; j < result.row_starts[i+1]; j++) {
            std::cout << "(" << result.col[j] << ", " << result.val[j] << ") ";
        }
        std::cout << std::endl;
    }
    printf("result->row_starts[8]: %d\n", result.row_starts[7]);
    printf("result->val[result->row_starts[7]+1]: %f\n", result.val[result.row_starts[7]+1]);
    // std::cout << "==== Benchmark results ====\n";
    // std::cout << "HostToDevice: " << timings.h2d_s << " s\n";
    // std::cout << "SpMV avg:     " << timings.spmv_avg_s << " s\n";
    // std::cout << "Lanczos total:" << timings.lanczos_s << " s\n";

    return 0;
}