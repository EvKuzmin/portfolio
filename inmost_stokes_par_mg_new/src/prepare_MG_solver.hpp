#pragma once

#include "petsc.h"

#include <map>
#include <string>
#include <iostream>
#include "prepare_diag_preconditioner.hpp"
#include "stack_matrix.hpp"
#include "petsc_matrix_write.hpp"
// #include "prepare_multigrid_preconditioner_prec_Aff.hpp"

PetscErrorCode convergence_test_MG(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
{
    *reason = KSP_CONVERGED_ITERATING;

    if (it > 15) {
        *reason = KSP_CONVERGED_ITS;
    }

    return 0;
}

PetscErrorCode convergence_test_full(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
{
    *reason = KSP_CONVERGED_ITERATING;
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
    
    // VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    VecAXPY(Ax, -1.0, rhs);

    double residual_Ax = 0.0;

    VecNorm(Ax, NORM_2, &residual_Ax);

    double f_norm = 0.0;
    VecNorm(rhs, NORM_2, &f_norm);

    double relative_residual = 0.0;

    if( f_norm != 0.0 ) {
        relative_residual = residual_Ax / f_norm;
        std::cout << "it = " << it << " solve_matrix convergence_test_full f_norm = " << f_norm << "\n";
        std::cout << "it = " << it << " solve_matrix convergence_test_full relative = " << relative_residual << "\n";
        if (relative_residual < 1e-8)
        {
            *reason = KSP_CONVERGED_ITS;
        }
    }

    // if (residual_Ax < 1e-1)
    // {
    //     *reason = KSP_CONVERGED_ITS;
    // }

    if (it > 25000) {
        *reason = KSP_DIVERGED_ITS;
    }

    PC             pc;
    KSPGetPC(ksp, &pc);

    PCFailedReason pcreason;
    PCGetFailedReason(pc,&pcreason);

    if(pcreason != PC_NOERROR) {
        *reason = KSP_DIVERGED_ITS;
    }

    std::cout << "it = " << it << " solve_matrix convergence_test_full = " << residual_Ax << "\n";
    VecDestroy(&Ax);
    return 0;
}

auto prepare_solver_MG(auto &submatrix_map, std::map< std::string, Mat > &submatrices, 
    std::map< std::string, IS > &index_sequences, auto write_func_immediate) {
    std::cout << "prepare_solver_multigrid\n";

    Vec x;

    MatCreateVecs(submatrices.at("without_dirichlet"), NULL, &x);

    // VecCreate(PETSC_COMM_WORLD, &x);
    // VecSetSizes(x, N, N);
    VecSetFromOptions(x);
    VecSet(x,0.0);

    Mat A_main_diag;

    A_main_diag = petsc_stack_main_diag(submatrices.at("Auu"), submatrices.at("App"), 9);
    // {
    //     MatNullSpace nullspace_full;

    //     Vec nsp_full1, nsp_full2;
    //     double *nsp_raw;

    //     MatCreateVecs(submatrices.at("without_dirichlet"), NULL, &nsp_full1);
    //     MatCreateVecs(submatrices.at("without_dirichlet"), NULL, &nsp_full2);

    //     VecSet(nsp_full1, 0.0);

    //     VecGetArray(nsp_full1, &nsp_raw);

    //     // const auto& Ap_Aup_map = submatrix_map.find_nodes("Ap")[0]->map_to_parent;

    //     for( auto idx : submatrix_map.find_nodes("Ap")[0]->map_to_parent ){
    //         nsp_raw[idx] = 1.0;
    //     }

    //     VecRestoreArray(nsp_full1, &nsp_raw);

    //     VecSet(nsp_full2, 0.0);

    //     VecGetArray(nsp_full2, &nsp_raw);

    //     // const auto& Au_Aup_map = submatrix_map.find_nodes("Au")[0]->map_to_parent;

    //     for( auto idx : submatrix_map.find_nodes("Au")[0]->map_to_parent ){
    //         nsp_raw[idx] = 1.0;
    //     }

    //     VecRestoreArray(nsp_full2, &nsp_raw);

    //     MPI_Barrier(MPI_COMM_WORLD);

    //     VecNormalize(nsp_full1, nullptr);
    //     VecNormalize(nsp_full2, nullptr);

    //     Vec nsp_vectors[2];
    //     nsp_vectors[0] = nsp_full1;
    //     nsp_vectors[1] = nsp_full2;

    //     MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 2, nsp_vectors, &nullspace_full);
    //     MatSetNearNullSpace(A_main_diag, nullspace_full);
    // }

    // MatNullSpace nullspace;
    // MatGetNullSpace(submatrices.at("without_dirichlet"), &nullspace);
    // MatSetNullSpace(A_main_diag, nullspace);
    // MatSetNearNullSpace(A_main_diag, nullspace);
    // MatSetNullSpace(A_main_diag, nullspace_full);
    // MatSetNearNullSpace(A_main_diag, nullspace_full);

    // write_func_immediate(A_main_diag);
    // MPI_Barrier(MPI_COMM_WORLD);
    // std::exit(0);

    KSP            ksp_lapl;
    KSPCreate(PETSC_COMM_WORLD,&ksp_lapl);

    PetscOptionsClear(NULL);
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-8");
    PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    // PetscOptionsSetValue(NULL, "-pc_mg_levels", "3");
    // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_2", "1.0e-15");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    PetscOptionsSetValue(NULL, "-pc_type", "gamg");
    // PetscOptionsSetValue(NULL, "-pc_gamg_agg_nsmooths", "3");
    // PetscOptionsSetValue(NULL, "-pc_gamg_type", "agg");
    PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_coarsening", "1");
    // PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_mis_k", "1");
    // PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_square_graph", "false");
    // PetscOptionsSetValue(NULL, "-mat_coarsen_view","");
    PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "150");
    // PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
    PetscOptionsSetValue(NULL, "-ksp_monitor", "");
    PetscOptionsSetValue(NULL, "-ksp_view", "");

    KSPSetFromOptions(ksp_lapl);
    KSPSetOperators(ksp_lapl,A_main_diag,A_main_diag);
    KSPSetType(ksp_lapl,KSPGMRES);

    KSPSetUp(ksp_lapl);

    PC pc_lapl;
    KSPGetPC(ksp_lapl, &pc_lapl);

    int N_levels;

    PCMGGetLevels(pc_lapl, &N_levels);

    std::vector<Mat> A_lapl_interp_GAMG(N_levels-1);
    std::vector<Mat> A_lapl_restrict_GAMG(N_levels-1);

    for( int l = 0 ; l < N_levels-1 ; l++ ){
        PCMGGetInterpolation(pc_lapl, l+1, &(A_lapl_interp_GAMG[l]));
        MatTranspose(A_lapl_interp_GAMG[l], MAT_INITIAL_MATRIX, &(A_lapl_restrict_GAMG[l]));
    }

    std::vector<Mat> A_lapl_interp(N_levels-1);
    std::vector<Mat> A_lapl_restrict(N_levels-1);

    for( int l = 0 ; l < N_levels-1 ; l++ ){
        MatConvert(A_lapl_interp_GAMG[l], MATSAME, MAT_INITIAL_MATRIX, &(A_lapl_interp[l]));
        MatConvert(A_lapl_restrict_GAMG[l], MATSAME, MAT_INITIAL_MATRIX, &(A_lapl_restrict[l]));
    }
    KSPDestroy(&ksp_lapl);
    MatDestroy(&A_main_diag);

    // KSPSolve(ksp_lapl,b_wo,x_wo);




    KSP            ksp_lapl2;
    KSPCreate(PETSC_COMM_WORLD,&ksp_lapl2);

    PetscOptionsClear(NULL);
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-8");
    PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    // PetscOptionsSetValue(NULL, "-pc_mg_levels", "6");
    // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_2", "1.0e-15");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    PetscOptionsSetValue(NULL, "-pc_type", "mg");
    // PetscOptionsSetValue(NULL, "-pc_gamg_agg_nsmooths", "3");
    // PetscOptionsSetValue(NULL, "-pc_gamg_type", "agg");
    PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_coarsening", "0");
    // PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_mis_k", "1");
    // PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_square_graph", "false");
    // PetscOptionsSetValue(NULL, "-mat_coarsen_view","");
    PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "150");

    // PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
    PetscOptionsSetValue(NULL, "-ksp_monitor", "");
    PetscOptionsSetValue(NULL, "-ksp_view", "");

    KSPSetFromOptions(ksp_lapl2);
    
    KSPSetOperators(ksp_lapl2,submatrices.at("without_dirichlet"),submatrices.at("without_dirichlet"));
    KSPSetType(ksp_lapl2,KSPGMRES);

    
    KSPSetConvergenceTest(ksp_lapl2, convergence_test_full, nullptr, nullptr);

    PC pc_lapl2;
    KSPGetPC(ksp_lapl2, &pc_lapl2);
    PCMGSetLevels(pc_lapl2, N_levels, NULL);

    for( int l = 0 ; l < N_levels-1 ; l++ ){
        PCMGSetInterpolation(pc_lapl2, l+1, A_lapl_interp[l]);
    }

    Vec temp;
    MatCreateVecs(A_lapl_interp[0], &temp, nullptr);
    PCMGSetX(pc_lapl2, 0, temp);
    MatCreateVecs(A_lapl_interp[0], &temp, nullptr);
    PCMGSetRhs(pc_lapl2, 0, temp);
    // MatCreateVecs(A_lapl_interp[0], &temp, nullptr);
    // PCMGSetR(pc_lapl2, 0, temp);

    for( int l = 1 ; l < N_levels-1 ; l++ ){
        MatCreateVecs(A_lapl_interp[l], &temp, nullptr);
        PCMGSetX(pc_lapl2, l, temp);
        MatCreateVecs(A_lapl_interp[l], &temp, nullptr);
        PCMGSetRhs(pc_lapl2, l, temp);
        MatCreateVecs(A_lapl_interp[l], &temp, nullptr);
        PCMGSetR(pc_lapl2, l, temp);
    }

    // MatCreateVecs(A_lapl_interp[4], nullptr, &temp);
    // PCMGSetX(pc_lapl2, 5, temp);
    // MatCreateVecs(A_lapl_interp[4], nullptr, &temp);
    // PCMGSetRhs(pc_lapl2, 5, temp);
    MatCreateVecs(A_lapl_interp[N_levels-2], nullptr, &temp);
    PCMGSetR(pc_lapl2, N_levels-1, temp);


    KSP ksp_level;
    PC pc_level;
    PCMGGetSmoother(pc_lapl2,N_levels-1,&ksp_level);
    KSPSetOperators(ksp_level, submatrices.at("without_dirichlet"), submatrices.at("without_dirichlet"));
    // KSPSetType(ksp_level,KSPGMRES);
    // KSPSetTolerances(ksp_level, 0.4, 1e-5, 1000.0, 10);
    KSPSetConvergenceTest(ksp_level, convergence_test_MG, nullptr, nullptr);
    // KSPGetPC(ksp_level, &pc_level);
    // PCSetType(pc_level, PCNONE);
    
    Mat mat_level, mat_level2;
    mat_level = submatrices.at("without_dirichlet");
    
    for( int l = N_levels - 2 ; l >= 0 ; l-- ){
        PCMGGetSmoother(pc_lapl2,l,&ksp_level);
        MatMatMatMult(A_lapl_restrict[l], mat_level, A_lapl_interp[l], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &mat_level2);
        write_func_immediate(mat_level2);
        MPI_Barrier(MPI_COMM_WORLD);
        std::exit(0);
        KSPSetOperators(ksp_level, mat_level2, mat_level2);
        // KSPSetType(ksp_level,KSPGMRES);
        KSPSetTolerances(ksp_level, 0.4, 1e-5, 1000.0, 10);
        KSPSetConvergenceTest(ksp_level, convergence_test_MG, nullptr, nullptr);
        // KSPGetPC(ksp_level, &pc_level);
        // PCSetType(pc_level, PCNONE);
        mat_level = mat_level2;
    }
    
    return std::make_tuple( ksp_lapl2, x );
}