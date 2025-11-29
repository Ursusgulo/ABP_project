#ifndef LANCOZ_CUH
#define LANCOZ_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include "lancoz.hpp" 

const int block_size = 512;




void lancoz_gpu(const int N, const int m, SparseMatrixCRS<float>* result, Timings* timings);

#endif // LANCOZ_CUH