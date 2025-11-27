#include "lancoz.hpp"
#include <cmath>

const int block_size = 512;

template <typename T>
__global__
void d_compute_spmv(const std::size_t N,
                             const std::size_t *row_starts,
                             const int *column_indices,
                             const T *values,
                             const T *x, 
                             T *y)
{
  // TODO implement for GPU
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row < N)
  {
    T sum = 0;
    for (std::size_t idx = row_starts[row]; idx < row_starts[row + 1]; ++idx)
      sum += values[idx] * x[column_indices[idx]];
    y[row] = sum;
  }
  
}

template <typename T, unsigned int n_threads>
__device__ 
void warp_reduce(volatile T *s_data, unsigned int tid) {
    if (n_threads >= 64) s_data[tid] += s_data[tid + 32];
    if (n_threads >= 32) s_data[tid] += s_data[tid + 16];
    if (n_threads >= 16) s_data[tid] += s_data[tid + 8];
    if (n_threads >= 8) s_data[tid] += s_data[tid + 4];
    if (n_threads >= 4) s_data[tid] += s_data[tid + 2];
    if (n_threads >= 2) s_data[tid] += s_data[tid + 1];
}

template <typename T, unsigned int n_threads>
__global__ 
void d_dot_product(const std::size_t N, const T *x, const T *y, T *result) {
    extern __shared__ T s[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x *(n_threads * 2) + tid;
    unsigned int grid_size =  blockDim.x * 2 * gridDim.x;
    s[tid] = 0;

    while (i < N) {
        s[tid] += x[i] * y[i] + x[i + n_threads] * y[i + n_threads];
        i += grid_size;
    }  
    __syncthreads();

    if (n_threads >= 512) {if (tid < 256) {s[tid] += s[tid + 256];} __syncthreads();}
    if (n_threads >= 256) {if (tid < 128) {s[tid] += s[tid + 128];} __syncthreads();}
    if (n_threads >= 128) {if (tid < 64) {s[tid] += s[tid + 64];} __syncthreads();}
    
    if (tid < 32) warp_reduce<T, n_threads>(s, tid);
    if (tid == 0) atomicAdd(result, s[0]);
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
void gemv(const std::size_t N, const T *alpha, T* Ax, const T* y) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < N) {
        Ax[idx] = Ax[idx] -  (*alpha) * y[idx];
    }
}

template <typename T>
__global__ 
void new_vector_v(const std::size_t N, const T *w, const T *beta, T *v, int counter) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < N) {
        if (*beta == 0) {
            v[idx] = (idx == counter) ? 1.0 : 0.0;
            if(idx == counter) counter++;
        } else {
            v[idx] = w[idx] / (*beta);
        }
    }
}

template <typename T>
__global__
void update_result_matrix(T *d_result_val, const T *d_beta) {
    d_result_val[0] = *d_beta;
    d_result_val[1] = *d_beta;
}



template <typename T>
void lancoz_gpu(const std::size_t N, SparseMatrixCRS <T> *result) {
    int counter = 0;
    std::size_t *d_A_row_starts;
    int *d_A_col;
    T *h_v, *h_w;
    T *d_A_val, d_A_N;
    // T *d_result_row_starts, *d_result_col, 
    T *d_result_val;
    T *d_v, *d_w, *d_tmp, *d_beta;

    SparseMatrixCRS <T> A;
    generate_laplacian3D<T>(N, A);

    std::size_t new_N = A.N;

    h_v = new T[new_N];
    generate_unit_vector<T>(new_N, h_v, counter);


    int n_blocks = (new_N + block_size - 1) / (block_size);
    const unsigned int reduce_blocks = (N + (2 * block_size) - 1) / (2 * block_size);

    //Do iteration one on host
    h_w = new T[new_N];
    //Ax
    printf("A.row_starts[new_N]: %d\n", A.row_starts[new_N]);
    printf("A.nz: %d\n", A.nnz);
    compute_spmv<T>(new_N, &A, h_v, h_w);

    //alpha = w*v
    T alpha = dot_product(new_N, h_w, h_v);

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
    printf("N: %zu\n", new_N);

    cudaMalloc(&d_A_val, A.row_starts[new_N]*sizeof(T));
    cudaMalloc(&d_A_row_starts, (new_N + 1)*sizeof(std::size_t));
    cudaMalloc(&d_A_col, A.row_starts[new_N]*sizeof(int));

    //allocate result matrix on device
    cudaMalloc(&d_result_val, result->nnz * sizeof(T));
    //cudaMalloc(&d_result_row_starts, (result->N + 1)*sizeof(int));
    //cudaMalloc(&d_result_col, result->nnz*sizeof(int));

    //allocate vectors on device
    cudaMalloc(&d_v, new_N*sizeof(T));
    cudaMalloc(&d_w, new_N*sizeof(T));
    cudaMalloc(&d_tmp, new_N*sizeof(T));

    //Copy Laplacian3D matrix to device
    cudaMemcpy(d_A_val, A.val, A.row_starts[new_N]*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_row_starts, A.row_starts, (new_N + 1)*sizeof(std::size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_col, A.col, A.row_starts[new_N]*sizeof(int), cudaMemcpyHostToDevice);

    //copy result matrix to device
    cudaMemcpy(d_result_val, result->val, result->nnz*sizeof(T), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_result_row_starts, result->row_starts, (result->N + 1)*sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_result_col, result->col, result->nnz*sizeof(int), cudaMemcpyHostToDevice);
    
    //copy vectors to device   
    cudaMemcpy(d_v, h_v, new_N*sizeof(T), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_w, h_w, new_N*sizeof(T), cudaMemcpyHostToDevice);

    //allocate scalars on device
    cudaMalloc(&d_beta, sizeof(T));

    T identity = -1;
    for(int j = 1; j < new_N; j++) {
        
        d_scale_vector<T><<<n_blocks, block_size>>>(new_N, d_beta, d_v, d_tmp);
        cudaDeviceSynchronize();

        new_vector_v<T><<<n_blocks, block_size>>>(N, d_w, d_beta, d_v, counter);
        cudaDeviceSynchronize();
        
        d_compute_spmv<T><<<n_blocks, block_size>>>(new_N, d_A_row_starts, d_A_col, d_A_val, d_v, d_w);
        cudaDeviceSynchronize();

        gemv<T><<<n_blocks, block_size>>>(new_N, &identity, d_w, d_tmp);
        cudaDeviceSynchronize();
        
        d_dot_product<T, block_size><<<reduce_blocks, block_size, block_size * sizeof(T)>>>
                                    (new_N, d_w, d_v, d_result_val + result->row_starts[j] + 1);
        cudaDeviceSynchronize();

        result->col[result->row_starts[j]-1] = j;
        result->col[result->row_starts[j]] = j - 1;
        result->col[result->row_starts[j] +1 ] = j ;

        result->row_starts[j + 1] = result->row_starts[j] + 3;
        if(j == new_N -1) {
            result->row_starts[j + 1] = result->row_starts[j] + 2;
        }
        gemv<T><<<n_blocks, block_size>>>(new_N, d_result_val + result->row_starts[j] + 1, d_w, d_v);
        cudaDeviceSynchronize();

        d_dot_product<T, block_size><<<reduce_blocks, block_size, block_size * sizeof(T)>>>
                            (new_N, d_w, d_v, d_beta);
        cudaDeviceSynchronize();
        update_result_matrix<T><<<1,1>>>(d_result_val + result->row_starts[j] - 1, d_beta);
    }
    cudaMemcpy(result->val, d_result_val, result->nnz*sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_result_val);
    cudaFree(d_A_val);
    cudaFree(d_A_row_starts);
    cudaFree(d_A_col);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_tmp);
    cudaFree(d_beta);
}

// int main(){
//     const int N = 2; //size in one dimension
//     int N3 = N * N * N;
//     int nnz = N3 * 3 -2;
//     using T = float;
//     SparseMatrixCRS <T> result(N3, nnz);
//     lancoz_gpu<T>(N, &result);

//     // printf("Resulting Lancoz matrix:\n");
//     // for(int i = 0; i < result.N; i++) {
//     //     std::cout << "Row " << i << ": ";
//     //     for(int j = result.row_starts[i]; j < result.row_starts[i+1]; j++) {
//     //         std::cout << "(" << result.col[j] << ", " << result.val[j] << ") ";
//     //     }
//     //     std::cout << std::endl;
//     // }

//     return 0;
// }