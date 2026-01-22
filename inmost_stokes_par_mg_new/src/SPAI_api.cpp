#include"SPAI_api.hpp"

#include "aij.h"
#include "mpiaij.h"
#include "dvecimpl.h"
#include "vecimpl.h"

#include "Sparse_alpha.hpp"

#include <vector>

PetscErrorCode convergence_test_diag_compens(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
{
    *reason = KSP_CONVERGED_ITERATING;
    // if(it > 1) {
        Mat A, P;
        KSPGetOperators(ksp, &A, &P);
        Vec rhs;
        KSPGetRhs(ksp, &rhs);
        Vec x;
        // KSPGetSolution(ksp, &x);
        KSPBuildSolution(ksp, NULL, &x);
        Vec Ax;
        VecDuplicate(rhs, &Ax);
        VecSet(Ax, 0.0);
        MatMult(A, x, Ax);

        VecAXPY(Ax, -1.0, rhs);

        double residual_Ax = 0.0;

        VecNorm(Ax, NORM_2, &residual_Ax);

        double f_norm = 0.0;
        VecNorm(rhs, NORM_2, &f_norm);

        double relative_residual = 0.0;

        if( f_norm != 0.0 ) {
            relative_residual = residual_Ax / f_norm;
            std::cout << "   it = " << it << " diag compens A f_norm = " << f_norm << "\n";
            std::cout << "   it = " << it << " diag compens A relative = " << relative_residual << "\n";
            if (relative_residual < 1e-7)
            {
                *reason = KSP_CONVERGED_ITS;
            }
        }

        // if(res_history.size() > 10) {
        //     auto iter = res_history.rbegin();
        //     if(*iter != 0.0){
        //         if(std::fabs(*iter - *(iter+1)) / *iter < 1e-8 && std::fabs(*(iter+1) - *(iter+2)) / *(iter+1) < 1e-8 ) {
        //             if(relative_residual < 1e-5) {
        //                 *reason = KSP_CONVERGED_ITS;
        //             }
        //         }
        //     }
        // }
        // if (residual_Ax < 1e-8)
        // {
        //     *reason = KSP_CONVERGED_ITS;
        // }
        if (it > 10000) {
            *reason = KSP_DIVERGED_ITS;
        }

        std::cout << "   it = " << it << " diag compens A = " << residual_Ax << "\n";
        VecDestroy(&Ax);
    // }
    return 0;
}

void diagonal_compensation(Mat A, Mat A_inv) {
    KSP ksp_A;
    PetscOptionsClear(NULL);

    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
    PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_2", "1.0e-15");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    // PetscOptionsSetValue(NULL, "-pc_mg_levels", "2");
    // PetscOptionsSetValue(NULL, "-pc_type", "none");
    // PetscOptionsSetValue(NULL, "-pc_type", "gamg");
    PetscOptionsSetValue(NULL, "-pc_type", "hypre");
    PetscOptionsSetValue(NULL, "-pc_hypre_type", "boomeramg");
    // PetscOptionsSetValue(NULL, "-pc_type", "parms");
    // PetscOptionsSetValue(NULL, "-pc_parms_global", "schur");
    // PetscOptionsSetValue(NULL, "-pc_parms_local", "ilut");//"ilu0");
    // PetscOptionsSetValue(NULL, "-pc_parms_levels", "25");
    // PetscOptionsSetValue(NULL, "-pc_parms_blocksize", "100");
    // PetscOptionsSetValue(NULL, "-pc_parms_ind_tol", "1.0");
    // PetscOptionsSetValue(NULL, "-pc_parms_solve_tol", "1e-6");
    // PetscOptionsSetValue(NULL, "-pc_parms_max_dim", "10000");
    // PetscOptionsSetValue(NULL, "-pc_parms_max_it", "10000");
    // PetscOptionsSetValue(NULL, "-pc_parms_lfil_ilu_arms", "100");
    // PetscOptionsSetValue(NULL, "-pc_parms_lfil_schur", "100");
    // PetscOptionsSetValue(NULL, "-pc_parms_lfil_ilut_L_U", "100");
    // PetscOptionsSetValue(NULL, "-pc_parms_inter_nonsymmetric_perm", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_inter_column_perm", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_inter_row_scaling", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_inter_column_scaling", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_last_nonsymmetric_perm", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_last_column_perm", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_last_row_scaling", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_last_column_scaling", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_droptol_factors", "0.001");
    // PetscOptionsSetValue(NULL, "-pc_parms_droptol_schur_compl", "0.001");
    // PetscOptionsSetValue(NULL, "-pc_parms_droptol_last_schur", "0.001");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_stong_threshold", "0.2");
    PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_statistics", "");
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "modifiedRuge-Stueben");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+e-mm");
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "1");
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor", "0.2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+i-mm");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_relax_type_down", "l1-Gauss-Seidel");//l1-Gauss-Seidel");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_relax_type_up", "l1-Gauss-Seidel");//l1-Gauss-Seidel");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_min_coarse_size", "50");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_relax_type_coarse", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_cycle_type", "W");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_down", "2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_up", "2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_coarse", "2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_num_paths", "2");
    // PetscOptionsSetValue(NULL, "-ksp_monitor", "");
    // PetscOptionsSetValue(NULL, "-ksp_view", "");

    KSPCreate(PETSC_COMM_WORLD, &ksp_A);
    KSPSetOperators(ksp_A, A, A);
    KSPSetType(ksp_A, KSPGMRES);
    KSPSetErrorIfNotConverged(ksp_A, PETSC_TRUE);
    KSPSetConvergenceTest(ksp_A, convergence_test_diag_compens, nullptr, nullptr);
    KSPSetFromOptions(ksp_A);
    KSPSetUp(ksp_A);

    Vec A_e, A_SPAI_e, e;
    MatCreateVecs(A, nullptr, &A_e);
    MatCreateVecs(A, nullptr, &A_SPAI_e);
    MatCreateVecs(A, nullptr, &e);
    // VecSet(A_e, 0.0);
    // VecSet(A_SPAI_e, 0.0);
    VecSet(e, 1.0);

    KSPSolve( ksp_A, e, A_e );

    MatMult( A_inv, e, A_SPAI_e );

    // VecView( A_SPAI_e, PETSC_VIEWER_STDOUT_WORLD );

    VecAXPY( A_e, -1.0, A_SPAI_e );

    MatDiagonalSet(A_inv, A_e, ADD_VALUES);
}

void schur_diagonal_compensation(Mat Auu, Mat Aup, Mat Apu, Mat Schur) {
    KSP ksp_A;
    PetscOptionsClear(NULL);

    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
    PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_2", "1.0e-15");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    // PetscOptionsSetValue(NULL, "-pc_mg_levels", "2");
    // PetscOptionsSetValue(NULL, "-pc_type", "none");
    // PetscOptionsSetValue(NULL, "-pc_type", "gamg");
    PetscOptionsSetValue(NULL, "-pc_type", "hypre");
    PetscOptionsSetValue(NULL, "-pc_hypre_type", "boomeramg");
    // PetscOptionsSetValue(NULL, "-pc_type", "parms");
    // PetscOptionsSetValue(NULL, "-pc_parms_global", "schur");
    // PetscOptionsSetValue(NULL, "-pc_parms_local", "ilut");//"ilu0");
    // PetscOptionsSetValue(NULL, "-pc_parms_levels", "25");
    // PetscOptionsSetValue(NULL, "-pc_parms_blocksize", "100");
    // PetscOptionsSetValue(NULL, "-pc_parms_ind_tol", "1.0");
    // PetscOptionsSetValue(NULL, "-pc_parms_solve_tol", "1e-6");
    // PetscOptionsSetValue(NULL, "-pc_parms_max_dim", "10000");
    // PetscOptionsSetValue(NULL, "-pc_parms_max_it", "10000");
    // PetscOptionsSetValue(NULL, "-pc_parms_lfil_ilu_arms", "100");
    // PetscOptionsSetValue(NULL, "-pc_parms_lfil_schur", "100");
    // PetscOptionsSetValue(NULL, "-pc_parms_lfil_ilut_L_U", "100");
    // PetscOptionsSetValue(NULL, "-pc_parms_inter_nonsymmetric_perm", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_inter_column_perm", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_inter_row_scaling", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_inter_column_scaling", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_last_nonsymmetric_perm", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_last_column_perm", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_last_row_scaling", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_last_column_scaling", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_droptol_factors", "0.001");
    // PetscOptionsSetValue(NULL, "-pc_parms_droptol_schur_compl", "0.001");
    // PetscOptionsSetValue(NULL, "-pc_parms_droptol_last_schur", "0.001");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_stong_threshold", "0.2");
    PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_statistics", "");
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "modifiedRuge-Stueben");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+e-mm");
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "1");
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor", "0.2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+i-mm");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_relax_type_down", "l1-Gauss-Seidel");//l1-Gauss-Seidel");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_relax_type_up", "l1-Gauss-Seidel");//l1-Gauss-Seidel");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_min_coarse_size", "50");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_relax_type_coarse", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_cycle_type", "W");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_down", "2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_up", "2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_coarse", "2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_num_paths", "2");
    // PetscOptionsSetValue(NULL, "-ksp_monitor", "");
    // PetscOptionsSetValue(NULL, "-ksp_view", "");

    KSPCreate(PETSC_COMM_WORLD, &ksp_A);
    KSPSetOperators(ksp_A, Auu, Auu);
    KSPSetType(ksp_A, KSPGMRES);
    KSPSetErrorIfNotConverged(ksp_A, PETSC_TRUE);
    KSPSetConvergenceTest(ksp_A, convergence_test_diag_compens, nullptr, nullptr);
    KSPSetFromOptions(ksp_A);
    KSPSetUp(ksp_A);

    Vec Apu_Auu_Aup_e, Auu_Aup_e, Aup_e, Schur_e, e;
    MatCreateVecs(Aup, nullptr, &Aup_e);
    MatCreateVecs(Auu, nullptr, &Auu_Aup_e);
    MatCreateVecs(Apu, nullptr, &Apu_Auu_Aup_e);
    MatCreateVecs(Schur, nullptr, &Schur_e);
    MatCreateVecs(Schur, nullptr, &e);
    // VecSet(A_e, 0.0);
    // VecSet(A_SPAI_e, 0.0);
    VecSet(e, 1.0);

    MatMult( Aup, e, Aup_e );

    // MPI_Barrier( MPI_COMM_WORLD );

    // VecView( Aup_e, PETSC_VIEWER_STDOUT_WORLD );

    // MPI_Barrier( MPI_COMM_WORLD );

    // std::exit(0);

    KSPSolve( ksp_A, Aup_e, Auu_Aup_e );

    MatMult( Apu, Auu_Aup_e, Apu_Auu_Aup_e );

    MatMult( Schur, e, Schur_e );

    // VecView( A_SPAI_e, PETSC_VIEWER_STDOUT_WORLD );

    VecAXPY( Apu_Auu_Aup_e, -1.0, Schur_e );

    MatDiagonalSet(Schur, Apu_Auu_Aup_e, ADD_VALUES);
}

void build_SPAI( Mat A, Mat *A_inv_p )
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MatTranspose(A, MAT_INPLACE_MATRIX, &A);
    MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, A_inv_p);

    Mat A_inv = *A_inv_p;

    


    Mat A_diag, A_off_diag;
    const int *garray;

    MatMPIAIJGetSeqAIJ(A, &A_diag, &A_off_diag, &garray);

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

    PetscInt N_diag, M_diag;
    MatGetSize(A_diag, &N_diag, &M_diag);

    PetscInt N_off, M_off;
    MatGetSize(A_off_diag, &N_off, &M_off);

    std::vector<int> N_for_rank;
    N_for_rank.resize(size);
    MPI_Allgather(&N_diag, 1, MPI_INT, N_for_rank.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> narray;
    narray.resize(size + 1, 0); 
    for (int hk = 0; hk < size; hk++){
        narray[hk + 1] = narray[hk] +  N_for_rank[hk];
    }

    CRS_like_petsc A_like(N_diag, data_diag, ia_diag, ja_diag, 
                            M_off, data_off_diag, ia_off_diag, ja_off_diag, 
                            garray, narray[rank], narray[size], narray[size], rank);
    CRS_like_petsc M_like = A_like.SPAI(A_like);
    {
        void *A_raw_data = A_inv->data;
        Mat_MPIAIJ* A_mpi_data = (Mat_MPIAIJ*)A_raw_data;

        PetscInt *garray = A_mpi_data->garray;

        Mat A_diag = A_mpi_data->A;
        Mat A_off_diag = A_mpi_data->B;

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

        std::copy(M_like.Diag->a.begin(), M_like.Diag->a.end(), data_diag);
        std::copy(M_like.Off_diag_Cpr->a.begin(), M_like.Off_diag_Cpr->a.end(), data_off_diag);

        MatRestoreRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
        MatRestoreRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);
    }
    MatRestoreRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
    MatRestoreRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);

    Vec A_diagonal;
    MatCreateVecs(A, &A_diagonal, nullptr);

    MatGetDiagonal(A, A_diagonal);
    VecReciprocal(A_diagonal);

    Mat A_A_SPAI;
    // MatTranspose(Aff_SPAI, MAT_INPLACE_MATRIX, &Aff_SPAI);
    MatMatMult(A, A_inv, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &A_A_SPAI);
    MatShift(A_A_SPAI, -1.0);
    double residual = 0.0;
    MatNorm( A_A_SPAI, NORM_FROBENIUS, &residual );
    std::cout << "spai right residual = " << residual << "\n";
    MatDestroy( &A_A_SPAI );

    MatMatMult( A_inv, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &A_A_SPAI);
    MatShift(A_A_SPAI, -1.0);
    residual = 0.0;
    MatNorm( A_A_SPAI, NORM_FROBENIUS, &residual );
    std::cout << "spai left residual = " << residual << "\n";
    MatDestroy( &A_A_SPAI );

    MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, &A_A_SPAI);
    MatDiagonalScale( A_A_SPAI, nullptr, A_diagonal );
    MatShift(A_A_SPAI, -1.0);
    residual = 0.0;
    MatNorm( A_A_SPAI, NORM_FROBENIUS, &residual );
    std::cout << "spai right diag residual = " << residual << "\n";
    MatDestroy( &A_A_SPAI );

    MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, &A_A_SPAI);
    MatDiagonalScale( A_A_SPAI, A_diagonal, nullptr );
    MatShift(A_A_SPAI, -1.0);
    residual = 0.0;
    MatNorm( A_A_SPAI, NORM_FROBENIUS, &residual );
    std::cout << "spai left diag residual = " << residual << "\n";
    MatDestroy( &A_A_SPAI );
    // MatTranspose(Aff_SPAI, MAT_INPLACE_MATRIX, &Aff_SPAI);
    // MatTranspose(A, MAT_INPLACE_MATRIX, &A);
}

void build_SPAI_pattern( Mat A, Mat pattern, Mat *A_inv_p )
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MatTranspose(A, MAT_INPLACE_MATRIX, &A);
    MatConvert(pattern, MATSAME, MAT_INITIAL_MATRIX, A_inv_p);

    Mat A_inv = *A_inv_p;

    
    
    void *A_raw_data = A->data;
    Mat_MPIAIJ* A_mpi_data = (Mat_MPIAIJ*)A_raw_data;

    Mat A_diag = A_mpi_data->A;
    Mat A_off_diag = A_mpi_data->B;

    PetscInt *garray = A_mpi_data->garray;

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

    PetscInt N_diag, M_diag;
    MatGetSize(A_diag, &N_diag, &M_diag);

    PetscInt N_off, M_off;
    MatGetSize(A_off_diag, &N_off, &M_off);

    std::vector<int> N_for_rank;
    N_for_rank.resize(size);
    MPI_Allgather(&N_diag, 1, MPI_INT, N_for_rank.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> narray;
    narray.resize(size + 1, 0); 
    for (int hk = 0; hk < size; hk++){
        narray[hk + 1] = narray[hk] +  N_for_rank[hk];
    }

    CRS_like_petsc A_like(N_diag, data_diag, ia_diag, ja_diag, 
                            M_off, data_off_diag, ia_off_diag, ja_off_diag, 
                            garray, narray[rank], narray[size], narray[size], rank);

    Mat A_diag_pattern;
    Mat A_off_diag_pattern;
    const PetscInt *garray_pattern;

    MatMPIAIJGetSeqAIJ(pattern, &A_diag_pattern, &A_off_diag_pattern, &garray_pattern);

    // write_func(pattern);

    void *A_diag_raw_data_pattern = A_diag_pattern->data;
    Mat_SeqAIJ *A_diag_data_pattern = (Mat_SeqAIJ *)A_diag_raw_data_pattern;
    void *A_off_diag_raw_data_pattern = A_off_diag_pattern->data;
    Mat_SeqAIJ *A_off_diag_data_pattern = (Mat_SeqAIJ *)A_off_diag_raw_data_pattern;

    PetscInt *ia_diag_pattern = A_diag_data_pattern->i;
    PetscInt *ja_diag_pattern = A_diag_data_pattern->j;
    PetscScalar *data_diag_pattern = A_diag_data_pattern->a;

    PetscInt *ia_off_diag_pattern = A_off_diag_data_pattern->i;
    PetscInt *ja_off_diag_pattern = A_off_diag_data_pattern->j;
    PetscScalar *data_off_diag_pattern = A_off_diag_data_pattern->a;

    PetscInt N_diag_pattern, M_diag_pattern;
    MatGetSize(A_diag_pattern, &N_diag_pattern, &M_diag_pattern);

    PetscInt N_off_pattern, M_off_pattern;
    MatGetSize(A_off_diag_pattern, &N_off_pattern, &M_off_pattern);

    std::vector<int> N_for_rank_pattern;
    N_for_rank_pattern.resize(size);
    MPI_Allgather(&N_diag_pattern, 1, MPI_INT, N_for_rank_pattern.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> narray_pattern;
    narray_pattern.resize(size + 1, 0); 
    for (int hk_pattern = 0; hk_pattern < size; hk_pattern++){
        narray_pattern[hk_pattern + 1] = narray_pattern[hk_pattern] +  N_for_rank_pattern[hk_pattern];
    }

    CRS_like_petsc A_like_pattern(N_diag_pattern, data_diag_pattern, ia_diag_pattern, ja_diag_pattern, 
                                M_off_pattern, data_off_diag_pattern, ia_off_diag_pattern, ja_off_diag_pattern, 
                                garray_pattern, narray_pattern[rank], narray_pattern[size], narray_pattern[size], rank);
    // CRS_like_petsc M_like = A_like.SPAI(A_like);
    CRS_like_petsc M_like = A_like.SPAI(A_like_pattern);
    {
        void *A_raw_data = A_inv->data;
        Mat_MPIAIJ* A_mpi_data = (Mat_MPIAIJ*)A_raw_data;

        PetscInt *garray = A_mpi_data->garray;

        Mat A_diag = A_mpi_data->A;
        Mat A_off_diag = A_mpi_data->B;

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

        std::copy(M_like.Diag->a.begin(), M_like.Diag->a.end(), data_diag);
        std::copy(M_like.Off_diag_Cpr->a.begin(), M_like.Off_diag_Cpr->a.end(), data_off_diag);

        MatRestoreRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
        MatRestoreRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);
    }
    MatRestoreRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
    MatRestoreRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);

    Vec A_diagonal;
    MatCreateVecs(A, &A_diagonal, nullptr);

    MatGetDiagonal(A, A_diagonal);
    VecReciprocal(A_diagonal);

    Mat A_A_SPAI;
    // MatTranspose(Aff_SPAI, MAT_INPLACE_MATRIX, &Aff_SPAI);
    MatMatMult(A, A_inv, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &A_A_SPAI);
    MatShift(A_A_SPAI, -1.0);
    double residual = 0.0;
    MatNorm( A_A_SPAI, NORM_FROBENIUS, &residual );
    std::cout << "spai right residual = " << residual << "\n";
    MatDestroy( &A_A_SPAI );

    MatMatMult( A_inv, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &A_A_SPAI);
    MatShift(A_A_SPAI, -1.0);
    residual = 0.0;
    MatNorm( A_A_SPAI, NORM_FROBENIUS, &residual );
    std::cout << "spai left residual = " << residual << "\n";
    MatDestroy( &A_A_SPAI );

    MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, &A_A_SPAI);
    MatDiagonalScale( A_A_SPAI, nullptr, A_diagonal );
    MatShift(A_A_SPAI, -1.0);
    residual = 0.0;
    MatNorm( A_A_SPAI, NORM_FROBENIUS, &residual );
    std::cout << "spai right diag residual = " << residual << "\n";
    MatDestroy( &A_A_SPAI );

    MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, &A_A_SPAI);
    MatDiagonalScale( A_A_SPAI, A_diagonal, nullptr );
    MatShift(A_A_SPAI, -1.0);
    residual = 0.0;
    MatNorm( A_A_SPAI, NORM_FROBENIUS, &residual );
    std::cout << "spai left diag residual = " << residual << "\n";
    MatDestroy( &A_A_SPAI );

    // MatTranspose(A, MAT_INPLACE_MATRIX, &A);
}