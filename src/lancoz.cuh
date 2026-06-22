#ifndef LANCOZ_CUH
#define LANCOZ_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include "lancoz.hpp" 

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




void lancoz_gpu(const int N, const int m, SparseMatrixCRS<float>* result, Timings* timings);
void d_compute_spmv(const int N,
                             const int *row_starts,
                             const int *column_indices,
                             const float *values,
                             const float *x, 
                             float *y);
#endif // LANCOZ_CUH