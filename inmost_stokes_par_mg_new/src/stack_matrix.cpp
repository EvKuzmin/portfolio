#include "stack_matrix.hpp"

#include<vector>
#include<algorithm>

auto get_inner_arrays(Mat A) {
    // void *A_raw_data = A->data;
    // Mat_MPIAIJ* A_mpi_data = (Mat_MPIAIJ*)A_raw_data;

    // Mat A_diag = A_mpi_data->A;
    // Mat A_off_diag = A_mpi_data->B;

    // PetscInt *A_garray = A_mpi_data->garray;

    Mat A_diag, A_off_diag;
    const int *A_garray;

    MatMPIAIJGetSeqAIJ(A, &A_diag, &A_off_diag, &A_garray);

    PetscInt N_A_diag;
    const PetscInt *ia_diag;
    const PetscInt *ja_diag;
    PetscScalar *data_diag;
    PetscBool success_diag;

    MatGetRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
    MatSeqAIJGetArray(A_diag, &data_diag);

    PetscInt N_A_off_diag;
    const PetscInt *ia_off_diag;
    const PetscInt *ja_off_diag;
    PetscScalar *data_off_diag;
    PetscBool success_off_diag;

    MatGetRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);
    MatSeqAIJGetArray(A_off_diag, &data_off_diag);

    return std::make_tuple(A_diag, A_off_diag, A_garray, N_A_diag, ia_diag, ja_diag, data_diag, success_diag, N_A_off_diag, ia_off_diag, ja_off_diag, data_off_diag, success_off_diag);
}

void restore_inner_arrays(auto inner_arrays) {
    auto [A_diag, A_off_diag, A_garray, N_A_diag, ia_diag, ja_diag, data_diag, success_diag, N_A_off_diag, ia_off_diag, ja_off_diag, data_off_diag, success_off_diag] = inner_arrays;

    MatRestoreRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
    MatRestoreRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);
}

Mat petsc_stack_h(Mat A, Mat B, int nnz) {
    int A_m, A_n, A_owner_range0, A_owner_range1;
    int B_owner_range0, B_owner_range1;
    MatGetSize(A, &A_m, &A_n);
    MatGetOwnershipRange(A, &A_owner_range0, &A_owner_range1);
    MatGetOwnershipRange(B, &B_owner_range0, &B_owner_range1);
    int A_col_owner_range0, A_col_owner_range1;
    MatGetOwnershipRangeColumn(A, &A_col_owner_range0, &A_col_owner_range1);
    int B_col_owner_range0, B_col_owner_range1;
    MatGetOwnershipRangeColumn(B, &B_col_owner_range0, &B_col_owner_range1);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> A_owner_range0_vec(size), A_owner_range1_vec(size), B_owner_range0_vec(size), B_owner_range1_vec(size);

    MPI_Allgather(
    &A_col_owner_range0,
    1,
    MPI_INTEGER,
    A_owner_range0_vec.data(),
    1,
    MPI_INTEGER,
    MPI_COMM_WORLD);

    MPI_Allgather(
    &A_col_owner_range1,
    1,
    MPI_INTEGER,
    A_owner_range1_vec.data(),
    1,
    MPI_INTEGER,
    MPI_COMM_WORLD);

    MPI_Allgather(
    &B_col_owner_range0,
    1,
    MPI_INTEGER,
    B_owner_range0_vec.data(),
    1,
    MPI_INTEGER,
    MPI_COMM_WORLD);

    MPI_Allgather(
    &B_col_owner_range1,
    1,
    MPI_INTEGER,
    B_owner_range1_vec.data(),
    1,
    MPI_INTEGER,
    MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    int A_m_local, A_n_local;
    MatGetLocalSize(A, &A_m_local, &A_n_local);
    int B_m_local, B_n_local;
    MatGetLocalSize(B, &B_m_local, &B_n_local);

    int B_m, B_n;
    MatGetSize(B, &B_m, &B_n);

    Mat C;

    MatCreate( MPI_COMM_WORLD, &C );
    MatSetSizes( C, A_m_local, A_n_local + B_n_local, A_m, A_n + B_n );

    MatMPIAIJSetPreallocation(C, nnz, NULL, nnz, NULL);

    auto A_inner_stuff = get_inner_arrays( A );
    auto B_inner_stuff = get_inner_arrays( B );

    auto [A_diag, A_off_diag, A_garray, N_A_diag, A_ia_diag, A_ja_diag, A_data_diag, A_success_diag, N_A_off_diag, A_ia_off_diag, A_ja_off_diag, A_data_off_diag, A_success_off_diag] = A_inner_stuff;

    auto [B_diag, B_off_diag, B_garray, N_B_diag, B_ia_diag, B_ja_diag, B_data_diag, B_success_diag, N_B_off_diag, B_ia_off_diag, B_ja_off_diag, B_data_off_diag, B_success_off_diag] = B_inner_stuff;

    for( int i = 0 ; i < N_A_diag ; i++ ) {

        std::vector<int> off_diag_j;
        std::vector<double> off_diag_val;
        for( int j = A_ia_off_diag[i] ; j < A_ia_off_diag[i+1] ; j++) {
            off_diag_j.push_back(A_garray[A_ja_off_diag[j]]);
            off_diag_val.push_back(A_data_off_diag[j]);
        }
        for( int j = 0 ; j < off_diag_j.size() ; j++ ) {
            double value = off_diag_val[j];
            int i_id = i + A_owner_range0;
            // int j_id = off_diag_j[j] + B_col_owner_range0;

            int it = std::upper_bound(A_owner_range1_vec.begin(), A_owner_range1_vec.end(), off_diag_j[j]) - A_owner_range1_vec.begin();
            int j_id = off_diag_j[j] + B_owner_range0_vec[it];
            MatSetValues(C,1,&i_id,1,&j_id,&value,INSERT_VALUES);
        }
        for( int j = A_ia_diag[i] ; j < A_ia_diag[i+1] ; j++) {
            double value = A_data_diag[j];
            int i_id = i + A_owner_range0;
            int j_id = A_ja_diag[j] + A_col_owner_range0 + B_col_owner_range0;
            MatSetValues(C,1,&i_id,1,&j_id,&value,INSERT_VALUES);
        }
        off_diag_j.clear();
        off_diag_val.clear();
        for( int j = B_ia_off_diag[i] ; j < B_ia_off_diag[i+1] ; j++) {
            off_diag_j.push_back(B_garray[B_ja_off_diag[j]]);
            off_diag_val.push_back(B_data_off_diag[j]);
        }
        for( int j = 0 ; j < off_diag_j.size() ; j++ ) {
            double value = off_diag_val[j];
            int i_id = i + A_owner_range0;
            int it = std::upper_bound(B_owner_range1_vec.begin(), B_owner_range1_vec.end(), off_diag_j[j]) - B_owner_range1_vec.begin();
            int j_id = off_diag_j[j] + A_owner_range1_vec[it];
            
            MatSetValues(C,1,&i_id,1,&j_id,&value,INSERT_VALUES);
        }
        for( int j = B_ia_diag[i] ; j < B_ia_diag[i+1] ; j++) {
            double value = B_data_diag[j];
            int i_id = i + A_owner_range0;
            int j_id = B_ja_diag[j] + A_col_owner_range1 + B_col_owner_range0;
            MatSetValues(C,1,&i_id,1,&j_id,&value,INSERT_VALUES);
        }
    }

    restore_inner_arrays(A_inner_stuff);
    restore_inner_arrays(B_inner_stuff);

    MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); //MAT_FINAL_ASSEMBLY
    MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);

    return C;
}

Mat petsc_stack_v(Mat A, Mat B, int nnz) {
    int A_m, A_n, A_owner_range0, A_owner_range1;
    MatGetSize(A, &A_m, &A_n);
    MatGetOwnershipRange(A, &A_owner_range0, &A_owner_range1);
    int A_col_owner_range0, A_col_owner_range1;
    MatGetOwnershipRangeColumn(A, &A_col_owner_range0, &A_col_owner_range1);
    int B_col_owner_range0, B_col_owner_range1;
    MatGetOwnershipRangeColumn(B, &B_col_owner_range0, &B_col_owner_range1);

    int A_m_local, A_n_local;
    MatGetLocalSize(A, &A_m_local, &A_n_local);
    int B_m_local, B_n_local;
    MatGetLocalSize(B, &B_m_local, &B_n_local);

    int B_m, B_n, B_owner_range0, B_owner_range1;
    MatGetSize(B, &B_m, &B_n);
    MatGetOwnershipRange(B, &B_owner_range0, &B_owner_range1);

    Mat C;

    MatCreate( MPI_COMM_WORLD, &C );
    MatSetSizes( C, A_m_local + B_m_local, A_n_local, A_m + B_m, A_n );

    MatMPIAIJSetPreallocation(C, nnz, NULL, nnz, NULL);

    auto A_inner_stuff = get_inner_arrays( A );
    auto B_inner_stuff = get_inner_arrays( B );

    auto [A_diag, A_off_diag, A_garray, N_A_diag, A_ia_diag, A_ja_diag, A_data_diag, A_success_diag, N_A_off_diag, A_ia_off_diag, A_ja_off_diag, A_data_off_diag, A_success_off_diag] = A_inner_stuff;

    auto [B_diag, B_off_diag, B_garray, N_B_diag, B_ia_diag, B_ja_diag, B_data_diag, B_success_diag, N_B_off_diag, B_ia_off_diag, B_ja_off_diag, B_data_off_diag, B_success_off_diag] = B_inner_stuff;

    for( int i = 0 ; i < N_A_diag ; i++ ) {

        std::vector<int> off_diag_j;
        std::vector<double> off_diag_val;
        for( int j = A_ia_off_diag[i] ; j < A_ia_off_diag[i+1] ; j++) {
            off_diag_j.push_back(A_garray[A_ja_off_diag[j]]);
            off_diag_val.push_back(A_data_off_diag[j]);
        }
        for( int j = 0 ; j < off_diag_j.size() ; j++ ) {
            double value = off_diag_val[j];
            int i_id = i + A_owner_range0 + B_owner_range0;
            int j_id = off_diag_j[j];
            MatSetValues(C,1,&i_id,1,&j_id,&value,INSERT_VALUES);
        }
        for( int j = A_ia_diag[i] ; j < A_ia_diag[i+1] ; j++) {
            double value = A_data_diag[j];
            int i_id = i + A_owner_range0 + B_owner_range0;
            int j_id = A_ja_diag[j] + A_col_owner_range0;
            MatSetValues(C,1,&i_id,1,&j_id,&value,INSERT_VALUES);
        }
    }

    for( int i = 0 ; i < N_B_diag ; i++ ) {
        std::vector<int> off_diag_j;
        std::vector<double> off_diag_val;
        for( int j = B_ia_off_diag[i] ; j < B_ia_off_diag[i+1] ; j++) {
            off_diag_j.push_back(B_garray[B_ja_off_diag[j]]);
            off_diag_val.push_back(B_data_off_diag[j]);
        }
        for( int j = 0 ; j < off_diag_j.size() ; j++ ) {
            double value = off_diag_val[j];
            int i_id = i + A_owner_range1 + B_owner_range0;
            int j_id = off_diag_j[j];
            MatSetValues(C,1,&i_id,1,&j_id,&value,INSERT_VALUES);
        }
        for( int j = B_ia_diag[i] ; j < B_ia_diag[i+1] ; j++) {
            double value = B_data_diag[j];
            int i_id = i + A_owner_range1 + B_owner_range0;
            int j_id = B_ja_diag[j] + B_col_owner_range0;
            MatSetValues(C,1,&i_id,1,&j_id,&value,INSERT_VALUES);
        }
    }

    restore_inner_arrays(A_inner_stuff);
    restore_inner_arrays(B_inner_stuff);

    MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); //MAT_FINAL_ASSEMBLY
    MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);

    return C;
}

Mat petsc_stack_main_diag(Mat A, Mat B, int nnz) {
    int A_m, A_n, A_owner_range0, A_owner_range1;
    MatGetSize(A, &A_m, &A_n);
    MatGetOwnershipRange(A, &A_owner_range0, &A_owner_range1);
    int A_col_owner_range0, A_col_owner_range1;
    MatGetOwnershipRangeColumn(A, &A_col_owner_range0, &A_col_owner_range1);
    int B_col_owner_range0, B_col_owner_range1;
    MatGetOwnershipRangeColumn(B, &B_col_owner_range0, &B_col_owner_range1);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> A_owner_range0_vec(size), A_owner_range1_vec(size), B_owner_range0_vec(size), B_owner_range1_vec(size);

    MPI_Allgather(
    &A_col_owner_range0,
    1,
    MPI_INTEGER,
    A_owner_range0_vec.data(),
    1,
    MPI_INTEGER,
    MPI_COMM_WORLD);

    MPI_Allgather(
    &A_col_owner_range1,
    1,
    MPI_INTEGER,
    A_owner_range1_vec.data(),
    1,
    MPI_INTEGER,
    MPI_COMM_WORLD);

    MPI_Allgather(
    &B_col_owner_range0,
    1,
    MPI_INTEGER,
    B_owner_range0_vec.data(),
    1,
    MPI_INTEGER,
    MPI_COMM_WORLD);

    MPI_Allgather(
    &B_col_owner_range1,
    1,
    MPI_INTEGER,
    B_owner_range1_vec.data(),
    1,
    MPI_INTEGER,
    MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    int A_m_local, A_n_local;
    MatGetLocalSize(A, &A_m_local, &A_n_local);
    int B_m_local, B_n_local;
    MatGetLocalSize(B, &B_m_local, &B_n_local);

    int B_m, B_n, B_owner_range0, B_owner_range1;
    MatGetSize(B, &B_m, &B_n);
    MatGetOwnershipRange(B, &B_owner_range0, &B_owner_range1);

    Mat C;

    MatCreate( MPI_COMM_WORLD, &C );
    MatSetSizes( C, A_m_local + B_m_local, A_n_local + B_n_local, A_m + B_m, A_n + B_n );

    MatMPIAIJSetPreallocation(C, nnz, NULL, nnz, NULL);

    auto A_inner_stuff = get_inner_arrays( A );
    auto B_inner_stuff = get_inner_arrays( B );

    auto [A_diag, A_off_diag, A_garray, N_A_diag, A_ia_diag, A_ja_diag, A_data_diag, A_success_diag, N_A_off_diag, A_ia_off_diag, A_ja_off_diag, A_data_off_diag, A_success_off_diag] = A_inner_stuff;

    auto [B_diag, B_off_diag, B_garray, N_B_diag, B_ia_diag, B_ja_diag, B_data_diag, B_success_diag, N_B_off_diag, B_ia_off_diag, B_ja_off_diag, B_data_off_diag, B_success_off_diag] = B_inner_stuff;

    for( int i = 0 ; i < N_A_diag ; i++ ) {

        std::vector<int> off_diag_j;
        std::vector<double> off_diag_val;
        for( int j = A_ia_off_diag[i] ; j < A_ia_off_diag[i+1] ; j++) {
            off_diag_j.push_back(A_garray[A_ja_off_diag[j]]);
            off_diag_val.push_back(A_data_off_diag[j]);
        }
        for( int j = 0 ; j < off_diag_j.size() ; j++ ) {
            double value = off_diag_val[j];
            int i_id = i + A_owner_range0 + B_owner_range0;

            int it = std::upper_bound(A_owner_range1_vec.begin(), A_owner_range1_vec.end(), off_diag_j[j]) - A_owner_range1_vec.begin();

            int j_id = off_diag_j[j] + B_owner_range0_vec[it];
            MatSetValues(C,1,&i_id,1,&j_id,&value,INSERT_VALUES);
        }
        for( int j = A_ia_diag[i] ; j < A_ia_diag[i+1] ; j++) {
            double value = A_data_diag[j];
            int i_id = i + A_owner_range0 + B_owner_range0;
            int j_id = A_ja_diag[j] + A_col_owner_range0 + B_col_owner_range0;
            MatSetValues(C,1,&i_id,1,&j_id,&value,INSERT_VALUES);
        }
    }

    for( int i = 0 ; i < N_B_diag ; i++ ) {
        std::vector<int> off_diag_j;
        std::vector<double> off_diag_val;
        for( int j = B_ia_off_diag[i] ; j < B_ia_off_diag[i+1] ; j++) {
            off_diag_j.push_back(B_garray[B_ja_off_diag[j]]);
            off_diag_val.push_back(B_data_off_diag[j]);
        }
        for( int j = 0 ; j < off_diag_j.size() ; j++ ) {
            double value = off_diag_val[j];
            int i_id = i + A_owner_range1 + B_owner_range0;
            int it = std::upper_bound(B_owner_range1_vec.begin(), B_owner_range1_vec.end(), off_diag_j[j]) - B_owner_range1_vec.begin();
            int j_id = off_diag_j[j] + A_owner_range1_vec[it];
            MatSetValues(C,1,&i_id,1,&j_id,&value,INSERT_VALUES);
        }
        for( int j = B_ia_diag[i] ; j < B_ia_diag[i+1] ; j++) {
            double value = B_data_diag[j];
            int i_id = i + A_owner_range1 + B_owner_range0;
            int j_id = B_ja_diag[j] + A_col_owner_range1 + B_col_owner_range0;
            MatSetValues(C,1,&i_id,1,&j_id,&value,INSERT_VALUES);
        }
    }

    restore_inner_arrays(A_inner_stuff);
    restore_inner_arrays(B_inner_stuff);

    MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); //MAT_FINAL_ASSEMBLY
    MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);

    return C;
}