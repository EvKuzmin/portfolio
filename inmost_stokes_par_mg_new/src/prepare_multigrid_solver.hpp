#pragma once

#include "petsc.h"

#include <map>
#include <string>
#include <iostream>
#include "prepare_multigrid_preconditioner.hpp"
// #include "prepare_multigrid_preconditioner_prec_Aff.hpp"

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

auto prepare_solver_multigrid(std::map< std::string, Mat > &submatrices, 
    std::map< std::string, IS > &index_sequences) {
    std::cout << "prepare_solver_multigrid\n";

    // PetscInt N;
    // ISGetSize(index_sequences.at("without_dirichlet"), &N);

    KSP            ksp;
    PC             pc;
    Vec            x;

    MatCreateVecs(submatrices.at("without_dirichlet"), NULL, &x);

    // VecCreate(PETSC_COMM_WORLD, &x);
    // VecSetSizes(x, N, N);
    VecSetFromOptions(x);
    VecSet(x,0.0);

    auto [ctx_full] = prepare_multigrid_preconditioner_context( submatrices, index_sequences );

    PetscOptionsClear(NULL);
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
    PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_2", "1.0e-15");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    PetscOptionsSetValue(NULL, "-pc_type", "shell");
    // PetscOptionsSetValue(NULL, "-pc_type", "parms");
    // PetscOptionsSetValue(NULL, "-pc_parms_global", "schur");
    // PetscOptionsSetValue(NULL, "-pc_parms_local", "arms");//"ilu0");
    // PetscOptionsSetValue(NULL, "-pc_parms_levels", "25");
    // PetscOptionsSetValue(NULL, "-pc_parms_blocksize", "100");
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
    PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "150");
    // PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
    // PetscOptionsSetValue(NULL, "-ksp_monitor", "");

    KSPCreate(PETSC_COMM_WORLD,&ksp);

    KSPSetOperators(ksp,submatrices.at("without_dirichlet"),submatrices.at("without_dirichlet"));
    KSPSetType(ksp,KSPFGMRES);

    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCSHELL);
    KSPSetConvergenceTest(ksp, convergence_test_full, ctx_full, nullptr);
    // PCShellSetApply(pc, precond_Full_apply);
    PCShellSetApply(pc, precond_Aup_apply);
    PCShellSetContext(pc, ctx_full);
    PCSetUp(pc);
    KSPSetFromOptions(ksp);
    KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
    KSPSetErrorIfNotConverged(ksp, PETSC_TRUE);
    // KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);

    return std::make_tuple( ksp, x );
}