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
    int seq[3] = {1,-2,1};
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
            val[row_starts[i] + j] = seq[j];
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
void sparse_kronecker_product_CRS(
    const SparseMatrixCRS<T> *A,
    const SparseMatrixCRS<T> *B,
    SparseMatrixCRS<T> *C)
{
    int idx = 0;
    int* C_row = C->row_starts.data();
    int* C_col = C->col.data();
    T*   C_val = C->val.data();

    const int* A_row = A->row_starts.data();
    const int* A_col = A->col.data();
    const T*   A_val = A->val.data();

    const int* B_row = B->row_starts.data();
    const int* B_col = B->col.data();
    const T*   B_val = B->val.data();

    C_row[0] = 0;
    // Loop rows of A
    for (int i = 0; i < A->N; i++) {
        // Loop rows of B
        for (int j = 0; j < B->N; j++) {
            int C_row_index = i * B->N + j;
            // Loop non-zeros in row i of A and row j of B
            for (int a_idx = A_row[i]; a_idx < A_row[i+1]; a_idx++) {
                for (int b_idx = B_row[j]; b_idx < B_row[j+1]; b_idx++) {

                    C_val[idx] = A_val[a_idx] * B_val[b_idx];
                    C_col[idx] = A_col[a_idx] * B->N + B_col[b_idx];
                    idx++;
                }
            }
            C_row[C_row_index + 1] = idx;
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
    const int* A_row = A->row_starts.data();
    const int* A_col = A->col.data();
    const T* A_val = A->val.data();

    const int* B_row = B->row_starts.data();
    const int* B_col = B->col.data();
    const T* B_val = B->val.data();

    const int* C_row = C->row_starts.data();
    const int* C_col = C->col.data();
    const T* C_val = C->val.data();

    int* result_row = result->row_starts.data();
    int* result_col = result->col.data();
    T* result_val = result->val.data();

    for (int i = 0; i < A->N; i++) {
        int index_A = A_row[i], row_end_A = A_row[i+1];
        int index_B = B_row[i], row_end_B = B_row[i+1];
        int index_C = C_row[i], row_end_C = C_row[i+1];

        while (index_A < row_end_A || index_B < row_end_B || index_C < row_end_C) {
            //rerieve current column indices from each row. If there are no more 
            //elements, set to int_max to avoid minimum
            int col_a = (index_A < row_end_A ? A_col[index_A] : INT_MAX);
            int col_b = (index_B < row_end_B ? B_col[index_B] : INT_MAX);
            int col_c = (index_C < row_end_C ? C_col[index_C] : INT_MAX);
            //retrieve minimum column index with element
            int min_d = std::min(col_a, std::min(col_b, col_c));

            //add value to sum if column index matches minimum

            T sum = 0;
            if (index_A < row_end_A && A_col[index_A] == min_d) sum += A_val[index_A++];
            if (index_B < row_end_B && B_col[index_B] == min_d) sum += B_val[index_B++];
            if (index_C < row_end_C && C_col[index_C] == min_d) sum += C_val[index_C++];
            sum *= scalar;
            //update col index and value to result
            //increment row index
            if (sum != 0.0) {
                result_col[idx] = min_d;
                result_val[idx] = sum;
                idx++;
            }
        }
        result_row[i+1] = idx;
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
    printf("Laplacian 1D created with nnz: %d\n", laplacian_1d.row_starts[N]);

    SparseMatrixCRS <T> identity_mat(N, N);
    create_identity_matrix_CRS(identity_mat.N, identity_mat.col.data(), 
                                identity_mat.row_starts.data(), identity_mat.val.data());
    printf("identity created with nnz: %d\n", identity_mat.row_starts[N]);

    //Temporary matrices for 3D lapalcian Kronecker products
    SparseMatrixCRS <T> temp_L_I(N * N, laplacian_1d.nnz * identity_mat.nnz);
    sparse_kronecker_product_CRS(&laplacian_1d, &identity_mat, &temp_L_I);
    printf("temp_L_I created with nnz: %d\n", temp_L_I.row_starts[N * N]);

    SparseMatrixCRS <T> temp_I_I(N * N, identity_mat.nnz * identity_mat.nnz);
    sparse_kronecker_product_CRS(&identity_mat, &identity_mat, &temp_I_I);
    printf("hej\n");
    printf("temp_I_I created with nnz: %d\n", temp_I_I.row_starts[N * N]);

    //For laplacian in 3D: L_I_I + I_L_I + I_I_L
    SparseMatrixCRS <T> L_I_I(N * N * N, temp_L_I.nnz * identity_mat.nnz);
    sparse_kronecker_product_CRS(&temp_L_I, &identity_mat, &L_I_I);
    printf("L_I_I created with nnz: %d\n", L_I_I.row_starts[N * N * N]);

    SparseMatrixCRS <T> I_L_I(N * N * N, identity_mat.nnz * temp_L_I.nnz);
    sparse_kronecker_product_CRS(&identity_mat, &temp_L_I, &I_L_I);
    printf("I_L_I created with nnz: %d\n", I_L_I.row_starts[N * N * N]);

    SparseMatrixCRS <T> I_I_L(N * N * N, identity_mat.nnz * identity_mat.nnz * laplacian_1d.nnz);
    sparse_kronecker_product_CRS(&temp_I_I, &laplacian_1d, &I_I_L);
    printf("I_I_L created with nnz: %d\n", I_I_L.row_starts[N * N * N]);


    //Final laplacian in 3D matrix
    laplacian_3d = SparseMatrixCRS <T>(N * N * N, L_I_I.nnz + I_L_I.nnz + I_I_L.nnz);
    sparse_matrix_addition_CRS(&L_I_I, &I_L_I, &I_I_L, &laplacian_3d);
    printf("3D Laplacian created with nnz: %d\n", laplacian_3d.row_starts[laplacian_3d.N]);
    printf("hej\n");

    
    //print values for each row of final laplacian

}



