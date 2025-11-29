#ifndef LANCOZ_CUH
#define LANCOZ_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include "lancoz.hpp" 

const int block_size = 512;

struct Timings {
    float h2d_s = 0.0f;
    float spmv_s = 0.0f;
    float lanczos_s = 0.0f;
};


void lancoz_gpu(const std::size_t N, const std::size_t m, SparseMatrixCRS<float>* result, Timings* timings);

#endif // LANCOZ_CUH