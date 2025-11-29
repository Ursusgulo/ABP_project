#include "laplacian3D.hpp"
#include "math_utils.hpp"
#include <cmath>

struct Timings {
    float h2d_s = 0.0f;
    float spmv_s = 0.0f;
};

template <typename T>
void compute_spmv(const int N,
                    const SparseMatrixCRS<T> *matrix,
                    const T *vec, 
                    T *result){

    for (int i = 0; i < N; i++) {
        result[i] = 0;
        for(int j = matrix->row_starts[i]; j < matrix->row_starts[i + 1]; j++) {
            result[i] += matrix->val[j] * vec[matrix->col[j]];
        }
    }
}

template <typename T>
void lancoz(const int N,const int m, SparseMatrixCRS <T> *result, Timings* timings) {
    const int nnz = N * 3 - 2;
    double spmv_total_time = 0;
    
    //generate laplacian matrix in 3D
    SparseMatrixCRS <T> A;
    generate_laplacian3D<T>(N, A);

    //generate unit vector
    T *v = new T[A.N];
    T *tmp = new T[A.N];
    generate_unit_vector<T>(A.N, v, 0);

 
    // iteration one
    T *w = new T[A.N];
    compute_spmv<T>(A.N, &A, v, w);

    T alpha = dot_product(A.N, w, v);
    T beta = gemv_norm<T>(A.N, -alpha, w, v);
    
    // store in T matrix
    result->row_starts[0] = 0;
    result->val[0] = alpha;
    result->col[0] = 0;
    result->row_starts[1] = 2;
    //remining iterations

    for(int j = 1; j < m; j++) {
        //beta * v_{j-1}
        scale_vector<T>(A.N, -beta, v, tmp);
        if(is_zero(beta)) {

            generate_unit_vector<T>(A.N, v, j);
        } else {
            scale_vector<T>(A.N, 1/beta, w, v);
        }

        const auto spmv_start = std::chrono::steady_clock::now();
        compute_spmv<T>(A.N, &A, v, w);
        const auto spmv_end = std::chrono::steady_clock::now();
        spmv_total_time += std::chrono::duration<float>(spmv_end - spmv_start).count();

        gemv_norm<T>(A.N, 1, w, tmp);
        
        
        alpha = dot_product(A.N, w, v);
        
        alpha = is_zero(alpha) ? 0 : alpha;

        if(beta != 0.f) {
            result->val[result->row_starts[j]-1] = beta;
            result->val[result->row_starts[j]] = beta;
            result->col[result->row_starts[j]-1] = j;
            result->col[result->row_starts[j]] = j - 1;
        }
        if(alpha != 0.f) {
            result->val[result->row_starts[j]+1] = alpha;
            result->col[result->row_starts[j] +1 ] = j ;
        }


        result->row_starts[j + 1] = result->row_starts[j] + 3;
        if(j == A.N -1) {
            result->row_starts[j + 1] = result->row_starts[j] + 2;
        }
        beta = gemv_norm<T>(A.N, -alpha, w, v);
        beta = is_zero(beta) ? 0.f : beta;

    }
    timings->spmv_s = spmv_total_time;
    delete[] tmp;
    delete[] v;
    delete[] w; 
    
}

// int main(){
//     const int N = 10; //size in one dimension
//     int N3 = N * N * N;
//     int nnz = N3 * 3 -2;
//     using T = double;
//     SparseMatrixCRS <T> result(N3, nnz);
//     lancoz<double>(N, &result);

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

// int main() {
//     const int N = 2; //size in one dimension
//     // int N3 = N * N * N;
//     //int nnz = N3 * 3 -2;
//     int m = 20 * N; 
//     if(m > N*N*N) {
//         m = N*N*N;
//     }
//     using T = float;
//     SparseMatrixCRS <T> result(m, m*3-2); //TODO time 
//     lancoz<T>(N, m, &result);

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
//     // std::cout << "==== Benchmark results ====\n";
//     // std::cout << "HostToDevice: " << timings.h2d_s << " s\n";
//     // std::cout << "SpMV avg:     " << timings.spmv_avg_s << " s\n";
//     // std::cout << "Lanczos total:" << timings.lanczos_s << " s\n";

//     return 0;
// }