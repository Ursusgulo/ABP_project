#include <iostream>
#include <vector>
#include <numeric>
#include <climits>

template <typename T>
struct SparseMatrixCRS {
    int N = 0;
    int nnz = 0;

    std::vector<int> col;
    std::vector<int> row_starts;
    std::vector<T> val;

    SparseMatrixCRS() = default;

    SparseMatrixCRS(int N_, int nnz_) :
        N(N_), nnz(nnz_),
        col(nnz_),
        row_starts(N_ + 1),
        val(nnz_)
    {}
};


template <typename T>
void create_1D_laplacian_CRS(int N, int *col, int *row_starts, T *val) {
    row_starts[0] = 0;
    float seq[3] = {1.f,-2.f,1.f};
    val[0] = -2;
    val[1] = 1;
    col[0] = 0;
    col[1] = 1;
    row_starts[1] = row_starts[0] + 2;

    for(int i = 1; i < N; i += 1){ 
        int from_seq = 3;
        if(i == N -1) from_seq = 2;
        for (int j = 0; j < from_seq; j++) {
            col[row_starts[i] + j] = (i-1) + j;
            val[row_starts[i] + j] = seq[j] * 1/((float(N)+1.f)*(float(N)+1.f));
        }
        row_starts[i + 1] = row_starts[i] + from_seq;
    }
}

template <typename T>
void create_identity_matrix_CRS(int N, int *col, int *row_starts, T *val) {
    for(int i = 0; i < N; i++) {
        row_starts[i] = i;
        col[i] = i;
        val[i] = 1;
    }
    row_starts[N] = N;
}

template <typename T>
void sparse_kronecker_product_CRS(SparseMatrixCRS <T> *A, SparseMatrixCRS <T> *B, SparseMatrixCRS <T> *C) {
    int idx = 0; // index in C->val and C->col
    C->row_starts[0] = 0;
    // Iterate over rows of A
    for (int i = 0; i < A->N; i++) {
        // Iterate over rows of B
        for (int j = 0; j < B->N; j++) {
            // Current row in C
            int C_row = i * B->N + j;
            // Iterate over non-zero elements in row i of A and row j of B
            for (int a_idx = A->row_starts[i]; a_idx < A->row_starts[i+1]; a_idx++) {
                for (int b_idx = B->row_starts[j]; b_idx < B->row_starts[j+1]; b_idx++) {
                    C->val[idx] = A->val[a_idx] * B->val[b_idx];
                    C->col[idx] = A->col[a_idx] * B->N + B->col[b_idx];
                    idx++;
                }
            }
            C->row_starts[C_row + 1] = idx;

        }
    }
}

//TODO gågnra med grejen
template <typename T>
void sparse_matrix_addition_CRS(
    const SparseMatrixCRS <T> *A,
    const SparseMatrixCRS <T> *B,
    const SparseMatrixCRS <T> *C,
    SparseMatrixCRS <T> *result
) {
    int idx = 0;
    result->row_starts[0] = 0;
    T n_plus_one = result->N + 1;
    // T scalar = 1/(n_plus_one * n_plus_one);
    T scalar = 1;

    for (int i = 0; i < A->N; i++) {
        int index_A = A->row_starts[i], row_end_A = A->row_starts[i+1];
        int index_B = B->row_starts[i], row_end_B = B->row_starts[i+1];
        int index_C = C->row_starts[i], row_end_C = C->row_starts[i+1];

        while (index_A < row_end_A || index_B < row_end_B || index_C < row_end_C) {
            //rerieve current column indices from each row. If there are no more 
            //elements, set to int_max to avoid minimum
            int col_a = (index_A < row_end_A ? A->col[index_A] : INT_MAX);
            int col_b = (index_B < row_end_B ? B->col[index_B] : INT_MAX);
            int col_c = (index_C < row_end_C ? C->col[index_C] : INT_MAX);
            //retrieve minimum column index with element
            int min_d = std::min(col_a, std::min(col_b, col_c));

            //add value to sum if column index matches minimum

            T sum = 0;
            if (index_A < row_end_A && A->col[index_A] == min_d) sum += A->val[index_A++];
            if (index_B < row_end_B && B->col[index_B] == min_d) sum += B->val[index_B++];
            if (index_C < row_end_C && C->col[index_C] == min_d) sum += C->val[index_C++];
            sum *= scalar;
            //update col index and value to result
            //increment row index
            if (sum != 0.0) {
                result->col[idx] = min_d;
                result->val[idx] = sum;
                idx++;
            }
        }
        result->row_starts[i+1] = idx;
    }
    result->nnz = idx;
}

template <typename T>
void generate_laplacian3D(const int N, SparseMatrixCRS <T> &laplacian_3d) {
    const int nnz = N * 3 - 2;

    //Laplacian in 1D and Identity matrix in CRS format
    SparseMatrixCRS <T> laplacian_1d(N, nnz);
    create_1D_laplacian_CRS(laplacian_1d.N, laplacian_1d.col.data(), 
                            laplacian_1d.row_starts.data(), laplacian_1d.val.data());

    SparseMatrixCRS <T> identity_mat(N, N);
    create_identity_matrix_CRS(identity_mat.N, identity_mat.col.data(), 
                                identity_mat.row_starts.data(), identity_mat.val.data());

    //Temporary matrices for 3D lapalcian Kronecker products
    SparseMatrixCRS <T> temp_L_I(N * N, laplacian_1d.nnz * identity_mat.nnz);
    sparse_kronecker_product_CRS(&laplacian_1d, &identity_mat, &temp_L_I);

    SparseMatrixCRS <T> temp_I_I(N * N, identity_mat.nnz * identity_mat.nnz);
    sparse_kronecker_product_CRS(&identity_mat, &identity_mat, &temp_I_I);

    //For laplacian in 3D: L_I_I + I_L_I + I_I_L
    SparseMatrixCRS <T> L_I_I(N * N * N, temp_L_I.nnz * identity_mat.nnz);
    sparse_kronecker_product_CRS(&temp_L_I, &identity_mat, &L_I_I);

    SparseMatrixCRS <T> I_L_I(N * N * N, identity_mat.nnz * temp_L_I.nnz);
    sparse_kronecker_product_CRS(&identity_mat, &temp_L_I, &I_L_I);

    SparseMatrixCRS <T> I_I_L(N * N * N, identity_mat.nnz * identity_mat.nnz * laplacian_1d.nnz);
    sparse_kronecker_product_CRS(&temp_I_I, &laplacian_1d, &I_I_L);


    //Final laplacian in 3D matrix
    laplacian_3d = SparseMatrixCRS <T>(N * N * N, L_I_I.nnz + I_L_I.nnz + I_I_L.nnz);
    sparse_matrix_addition_CRS(&L_I_I, &I_L_I, &I_I_L, &laplacian_3d);


    
    //print values for each row of final laplacian

}



