#include "laplacian3D.hpp"
#include "math_utils.hpp"
#include <cmath>


template <typename T>
void compute_spmv(const std::size_t N,
                    const SparseMatrixCRS<T> *matrix,
                    const T *vec, 
                    T *result){

    for (std::size_t i = 0; i < N; i++) {
        result[i] = 0;
        for(std::size_t j = matrix->row_starts[i]; j < matrix->row_starts[i + 1]; j++) {
            result[i] += matrix->val[j] * vec[matrix->col[j]];
        }
    }
}

template <typename T>
void lancoz(const int N, SparseMatrixCRS <T> *result) {
    const int nnz = N * 3 - 2;
    int counter = 0;
    
    //generate laplacian matrix in 3D
    SparseMatrixCRS <T> A;
    generate_laplacian3D<T>(N, A);
    printf("A.N = %d\n", A.N);
    printf("Laplacian 3D matrix generated with nnz: %d\n", A.row_starts[A.N]);
    //generate unit vector
    T *v = new T[A.N];
    T *tmp = new T[A.N];
    generate_unit_vector<T>(A.N, v, counter);

 
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

    for(int j = 1; j < A.N; j++) {
        scale_vector<T>(A.N, -beta, v, tmp);
        if(is_zero(beta)) {
            counter++;
            //orthogonal to all previous unit vectors
            generate_unit_vector<T>(A.N, v, counter);
        } else {
            scale_vector<T>(A.N, 1/beta, w, v);
        }

        compute_spmv<T>(A.N, &A, v, w);

        gemv_norm<T>(A.N, 1, w, tmp);
        
        
        alpha = dot_product(A.N, w, v);
        
        alpha = is_zero(alpha) ? 0 : alpha;

        result->val[result->row_starts[j]-1] = beta;
        result->val[result->row_starts[j]] = beta;
        result->val[result->row_starts[j]+1] = alpha;

        result->col[result->row_starts[j]-1] = j;
        result->col[result->row_starts[j]] = j - 1;
        result->col[result->row_starts[j] +1 ] = j ;

        result->row_starts[j + 1] = result->row_starts[j] + 3;
        if(j == A.N -1) {
            result->row_starts[j + 1] = result->row_starts[j] + 2;
        }
        beta = gemv_norm<T>(A.N, -alpha, w, v);
        beta = is_zero(beta) ? 0 : beta;

    }
    delete[] v;
    delete[] w; 
    
}

int main(){
    const int N = 10; //size in one dimension
    int N3 = N * N * N;
    int nnz = N3 * 3 -2;
    using T = double;
    SparseMatrixCRS <T> result(N3, nnz);
    lancoz<double>(N, &result);

    // printf("Resulting Lancoz matrix:\n");
    // for(int i = 0; i < result.N; i++) {
    //     std::cout << "Row " << i << ": ";
    //     for(int j = result.row_starts[i]; j < result.row_starts[i+1]; j++) {
    //         std::cout << "(" << result.col[j] << ", " << result.val[j] << ") ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}