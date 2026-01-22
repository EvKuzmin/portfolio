#pragma once

#include "petsc.h"
#include <map>
#include <string>

#include "aij.h"
#include "dvecimpl.h"
#include "vecimpl.h"

#include "Aup_SPAI.hpp"

PetscErrorCode ConvergenceDestroy(void *ctx)
{
    PetscFunctionBegin;
    PetscCall(PetscInfo(NULL, "User provided convergence destroy called\n"));
    // PetscCall(PetscFree(ctx));
    PetscFunctionReturn(0);
}

PetscErrorCode convergence_test_Auu(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
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
            std::cout << "   it = " << it << " convergence_test_Auu f_norm = " << f_norm << "\n";
            std::cout << "   it = " << it << " convergence_test_Auu relative = " << relative_residual << "\n";
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

        std::cout << "   it = " << it << " convergence_test_Auu = " << residual_Ax << "\n";
        VecDestroy(&Ax);
    // }
    return 0;
}

PetscErrorCode convergence_test_Apu_Auu_inv_Aup(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
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
            std::cout << "   it = " << it << " Apu_Auu_inv_Aup f_norm = " << f_norm << "\n";
            std::cout << "   it = " << it << " Apu_Auu_inv_Aup relative = " << relative_residual << "\n";
            if (relative_residual < 1e-6)
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

        // if (residual_Ax < 1e-5)
        // {
        //     *reason = KSP_CONVERGED_ITS;
        // }
        if (it > 1000) {
            *reason = KSP_DIVERGED_ITS;
        }

        std::cout << "   it = " << it << " Apu_Auu_inv_Aup = " << residual_Ax << "\n";
        VecDestroy(&Ax);
    // }
    return 0;
}

auto prepare_preconditioner_context(
    std::map<std::string, Mat> &submatrices,
    std::map<std::string, IS> &index_sequences)
{
    Mat Auu = submatrices.at("Auu");
    Mat Aup = submatrices.at("Aup");
    Mat Apu = submatrices.at("Apu");
    Mat App = submatrices.at("App");

    PrecondContextAup *ctx;
    ctx = new PrecondContextAup;

    ctx->Aup_ = submatrices.at("without_dirichlet");
    ctx->Auu = Auu;
    ctx->Aup = Aup;
    ctx->Apu = Apu;
    ctx->Apu_Auu_inv_Aup = App;
    // ctx->Apu_Auu_inv_Aup_SPAI = 

    // int App_N, App_M;
    // MatGetSize(App, &App_N, &App_M);

    // MatCreateShell(PETSC_COMM_WORLD, App_N, App_M, App_N, App_M, ctx, &(ctx->Apu_Auu_inv_Aup));
    // MatShellSetOperation(ctx->Apu_Auu_inv_Aup, MATOP_MULT, (void (*)(void))Stokes_Schur_Mult);

    ctx->IS_U = index_sequences.at("Au");
    ctx->IS_P = index_sequences.at("Ap");

    {
        // PetscInt nU, nP;
        // ISGetSize(ctx->IS_U, &nU);
        // ISGetSize(ctx->IS_P, &nP);

        MatCreateVecs(Auu, NULL, &(ctx->Vu1));
        MatCreateVecs(Auu, NULL, &(ctx->Vu2));
        MatCreateVecs(Auu, NULL, &(ctx->Vu3));
        MatCreateVecs(App, NULL, &(ctx->Vp1));
        MatCreateVecs(App, NULL, &(ctx->Vp2));
        MatCreateVecs(App, NULL, &(ctx->Vp3));

        // VecCreate(PETSC_COMM_WORLD, &(ctx->Vu1));
        // VecCreate(PETSC_COMM_WORLD, &(ctx->Vu2));
        // VecCreate(PETSC_COMM_WORLD, &(ctx->Vu3));
        // VecCreate(PETSC_COMM_WORLD, &(ctx->Vp1));
        // VecCreate(PETSC_COMM_WORLD, &(ctx->Vp2));
        // VecCreate(PETSC_COMM_WORLD, &(ctx->Vp3));
        // VecSetSizes(ctx->Vu1, nU, nU);
        // VecSetSizes(ctx->Vu2, nU, nU);
        // VecSetSizes(ctx->Vu3, nU, nU);
        // VecSetSizes(ctx->Vp1, nP, nP);
        // VecSetSizes(ctx->Vp2, nP, nP);
        // VecSetSizes(ctx->Vp3, nP, nP);
        VecSetFromOptions(ctx->Vu1);
        VecSetFromOptions(ctx->Vu2);
        VecSetFromOptions(ctx->Vu3);
        VecSetFromOptions(ctx->Vp1);
        VecSetFromOptions(ctx->Vp2);
        VecSetFromOptions(ctx->Vp3);
        // Vec bu, bp;
        // VecGetSubVector(ctx->b, ctx->IS_U, &bu);
        // VecGetSubVector(ctx->b, ctx->IS_P, &bp);

        // VecDuplicate(bu, &(ctx->Vu1));
        // VecDuplicate(bu, &(ctx->Vu2));
        // VecDuplicate(bu, &(ctx->Vu3));
        // VecDuplicate(bp, &(ctx->Vp1));
        // VecDuplicate(bp, &(ctx->Vp2));
        // VecDuplicate(bp, &(ctx->Vp3));

        // VecRestoreSubVector(ctx->b, ctx->IS_U, &bu);
        // VecRestoreSubVector(ctx->b, ctx->IS_P, &bp);

        PetscOptionsClear(NULL);

        PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
        PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
        PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
        // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_2", "1.0e-15");
        PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
        PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
        // PetscOptionsSetValue(NULL, "-pc_mg_levels", "2");
        // PetscOptionsSetValue(NULL, "-pc_type", "ilu");
        PetscOptionsSetValue(NULL, "-pc_type", "hypre");
        PetscOptionsSetValue(NULL, "-pc_hypre_type", "boomeramg");
        // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_stong_threshold", "0.2");
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
        PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
        // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_statistics", "");
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "modifiedRuge-Stueben");
        PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+e-mm");
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "1");
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor", "0.5");
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
        // PetscOptionsSetValue(NULL, "-ksp_constant_null_space", "");
        // PetscOptionsSetValue(NULL, "-ksp_monitor", "");

        KSPCreate(PETSC_COMM_WORLD, &(ctx->kspAuu));
        KSPSetOperators(ctx->kspAuu, ctx->Auu, ctx->Auu);
        KSPSetType(ctx->kspAuu, KSPGMRES);
        KSPSetErrorIfNotConverged(ctx->kspAuu, PETSC_TRUE);
        KSPSetConvergenceTest(ctx->kspAuu, convergence_test_Auu, nullptr, ConvergenceDestroy);
        KSPSetFromOptions(ctx->kspAuu);
        KSPSetUp(ctx->kspAuu);
        // KSPView(ctx->kspAuu,PETSC_VIEWER_STDOUT_WORLD);

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

        MatNullSpace nullspace;

        MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
        MatSetNearNullSpace(ctx->Apu_Auu_inv_Aup, nullspace);
        MatSetNullSpace(ctx->Apu_Auu_inv_Aup, nullspace);

        KSPCreate(PETSC_COMM_WORLD, &(ctx->kspApu_Auu_inv_Aup));
        KSPSetOperators(ctx->kspApu_Auu_inv_Aup, ctx->Apu_Auu_inv_Aup, ctx->Apu_Auu_inv_Aup);
        KSPSetType(ctx->kspApu_Auu_inv_Aup, KSPGMRES);
        KSPSetErrorIfNotConverged(ctx->kspApu_Auu_inv_Aup, PETSC_TRUE);
        KSPSetConvergenceTest(ctx->kspApu_Auu_inv_Aup, convergence_test_Apu_Auu_inv_Aup, nullptr, nullptr);
        KSPSetFromOptions(ctx->kspApu_Auu_inv_Aup);
        KSPSetUp(ctx->kspApu_Auu_inv_Aup);
        // KSPView(ctx->kspApu_Auu_inv_Aup,PETSC_VIEWER_STDOUT_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    return std::make_tuple(ctx);
}