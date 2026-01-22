#pragma once

#include "inmost.h"
#include "petsc.h"
#include <map>
#include <string>
#include <chrono>

#include "submatrix_manager.hpp"

#include "Aup_diag.hpp"

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

PetscErrorCode convergence_test_ASchur_precond(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
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
            std::cout << "   it = " << it << " convergence_test_ASchur_precond f_norm = " << f_norm << "\n";
            std::cout << "   it = " << it << " convergence_test_ASchur_precond relative = " << relative_residual << "\n";
            if (relative_residual < 1e-8)
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

        std::cout << "   it = " << it << " convergence_test_ASchur_precond = " << residual_Ax << "\n";
        VecDestroy(&Ax);
    // }
    return 0;
}

PetscErrorCode Stokes_Schur_Mult(Mat A, Vec x, Vec y) {
    auto calc_residual = [](Mat A, Vec x, Vec b, std::string name){
        Vec Ax;
        VecDuplicate(b, &Ax);
        VecSet(Ax, 0.0);
        MatMult(A, x, Ax);

        VecAXPY(Ax, -1.0, b);

        double residual_Ax = 0.0;

        VecNorm(Ax, NORM_2, &residual_Ax);

        std::cout << "solve_matrix " << name << " = " << residual_Ax << "\n";
        VecDestroy(&Ax);
    };
    // double x_sum = 0.0;
    // VecSum(x, &x_sum);

    PrecondContextAup* ctx_full;
    MatShellGetContext(A, &ctx_full);

    KSPConvergedReason reason;

    Vec tmp1 = ctx_full->tmp1;
    Vec tmp2 = ctx_full->tmp2;
    Vec tmp3 = ctx_full->tmp3;

    VecZeroEntries(tmp1);
    VecZeroEntries(tmp2);
    VecZeroEntries(tmp3);

    MatMult(ctx_full->Aup, x, tmp1);
    // KSPView(ctx_full->kspAuu,PETSC_VIEWER_STDOUT_WORLD);
    // std::exit(0);
    double tmp1_norm = 0.0;

    VecNorm(tmp1, NORM_2, &tmp1_norm);
    if(tmp1_norm != 0) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        KSPSolve(ctx_full->kspAuu, tmp1, tmp2);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Auu in Schur solve time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " milliseconds" << std::endl;
        KSPGetConvergedReason(ctx_full->kspAuu, &reason);
        // std::cout << "Stokes_Schur_Mult convergence reason = " << reason << "\n";
        if(reason < 0) {
            std::cerr << "Stokes_Schur_Mult matrix application failed\n";
            // VecRestoreSubVector(x, ctx->IS_UP, &bup);
            // VecRestoreSubVector(x, ctx->IS_PhiMu, &bphimu);
            // VecRestoreSubVector(y, ctx->IS_UP, &yup);
            // VecRestoreSubVector(y, ctx->IS_PhiMu, &yphimu);
            // throw std::runtime_error("something not converged\n");
            // PCSetFailedReason(pc, PC_FACTOR_OTHER);
            return 0;
        }
    }
    // calc_residual( ctx_full->Auu, tmp2, tmp1, "Stokes_Schur_Mult" );
    MatMult(ctx_full->Apu, tmp2, tmp3);

    VecScale(tmp3, -1.0);

    // MatNullSpace nullspace;
    // MatGetNullSpace(ctx_full->Apu_Auu_inv_Aup, &nullspace);
    // MatNullSpaceRemove(nullspace, tmp3);

    // VecShift(tmp3, 0.1*x_sum);

    VecCopy(tmp3, y);

    // VecDestroy(&tmp1);
    // VecDestroy(&tmp2);
    // VecDestroy(&tmp3);

    return 0;
}

PetscErrorCode Stokes_Schur_Mult_precond_apply(PC pc, Vec x, Vec y){
    PrecondContextAup *ctx;
    PCShellGetContext(pc,&ctx);
    KSPConvergedReason reason;
    KSPSolve(ctx->kspASchur_precond, x, y); 
    KSPGetConvergedReason(ctx->kspASchur_precond, &reason);
    if(reason < 0) {
        std::cerr << "kspASchur_precond diverged\n";
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }
    PCSetFailedReason(pc, PC_NOERROR);
    // std::exit(0);
    return 0;
}

auto prepare_diag_preconditioner_context(
    std::map<std::string, Mat> &submatrices,
    std::map<std::string, IS> &index_sequences)
{
    Mat Auu = submatrices.at("Auu");
    Mat App = submatrices.at("App");
    Mat Aup = submatrices.at("Aup");
    Mat Apu = submatrices.at("Apu");

    PrecondContextAup *ctx;
    ctx = new PrecondContextAup;

    MatCreateVecs(Auu, NULL, &(ctx->tmp1));
    MatCreateVecs(Auu, NULL, &(ctx->tmp2));
    MatCreateVecs(App, NULL, &(ctx->tmp3));

    ctx->Aup_ = submatrices.at("without_dirichlet");
    ctx->Auu = Auu;
    //ctx->App = App;
    ctx->Aup = Aup;
    ctx->Apu = Apu;


    {
        Mat A_R = submatrices.at("Aup");
        Mat A_L = submatrices.at("Apu");
        Mat Auu_inv;
        build_SPAI(submatrices.at("Auu"), &Auu_inv);

        // { // Spai symm
        //     Mat Auu_inv_T; 
        //     MatTranspose(Auu_inv, MAT_INITIAL_MATRIX, &Auu_inv_T);

        //     MatAXPY(Auu_inv, 1.0, Auu_inv_T, SAME_NONZERO_PATTERN);
        //     MatScale(Auu_inv, 0.5);
        //     MatDestroy(&Auu_inv_T);
        // }

        // diagonal_compensation(submatrices.at("Auu"), Auu_inv); // Diagonal compensation actually make matrix worse in porous medium case

        Mat Apu_Auu_inv;

        MatMatMult(A_L, Auu_inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Apu_Auu_inv);
        MatMatMult(Apu_Auu_inv, A_R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->ASchur_precond));
        // MatMatMult(A_L, A_R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->ASchur_precond));
        // schur_diagonal_compensation(submatrices.at("Auu"), A_R, A_L, ctx->ASchur_precond);
        MatScale(ctx->ASchur_precond, -1.0);

        // MatDestroy(&Auu_inv);
        // MatDestroy(&Apu_Auu_inv);

        PetscOptionsClear(NULL);

        PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
        PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
        PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
        PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
        PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
        // PetscOptionsSetValue(NULL, "-pc_type", "none");
        // PetscOptionsSetValue(NULL, "-pc_type", "bjacobi");
        PetscOptionsSetValue(NULL, "-pc_type", "hypre");
        PetscOptionsSetValue(NULL, "-pc_hypre_type", "boomeramg");
        PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
        // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_statistics", "");
        // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "modifiedRuge-Stueben");
        PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "1");
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_strong_threshold", "0.7");
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "HMIS");
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+i");
        PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor", "0.2");
        // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+e-mm");
        // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "1");
        // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor", "0.5");
        // PetscOptionsSetValue(NULL, "-ksp_monitor", "");

        MatNullSpace nullspace;

        MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
        MatSetNearNullSpace(ctx->ASchur_precond, nullspace);
        MatSetNullSpace(ctx->ASchur_precond, nullspace);
        MatNullSpaceDestroy(&nullspace);

        KSPCreate(PETSC_COMM_WORLD, &(ctx->kspASchur_precond));
        KSPSetFromOptions(ctx->kspASchur_precond);
        KSPSetOperators(ctx->kspASchur_precond, ctx->ASchur_precond, ctx->ASchur_precond);
        KSPSetType(ctx->kspASchur_precond, KSPGMRES);
        KSPSetErrorIfNotConverged(ctx->kspASchur_precond, PETSC_TRUE);
        KSPSetConvergenceTest(ctx->kspASchur_precond, convergence_test_ASchur_precond, nullptr, nullptr);
        KSPSetFromOptions(ctx->kspASchur_precond);
        KSPSetUp(ctx->kspASchur_precond);
    }


    int App_n, App_m, App_N, App_M;
    MatGetSize(App, &App_N, &App_M);
    MatGetLocalSize(App, &App_n, &App_m);
    

    std::cout << "App_N = " << App_N << " App_N = " << App_M << "\n";
    // std::exit(0);

    MatCreateShell(PETSC_COMM_WORLD, App_n, App_m, App_N, App_M, ctx, &(ctx->App));
    MatShellSetOperation(ctx->App, MATOP_MULT, (void (*)(void))Stokes_Schur_Mult);

    ctx->IS_U = index_sequences.at("Au");
    ctx->IS_P = index_sequences.at("Ap");

    PetscOptionsClear(NULL);

    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
    PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    // PetscOptionsSetValue(NULL, "-pc_type", "none");
    // PetscOptionsSetValue(NULL, "-pc_type", "bjacobi");
    PetscOptionsSetValue(NULL, "-pc_type", "hypre");
    PetscOptionsSetValue(NULL, "-pc_hypre_type", "boomeramg");
    PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_statistics", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "modifiedRuge-Stueben");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+e-mm");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "1");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor", "0.5");
    // PetscOptionsSetValue(NULL, "-ksp_monitor", "");

    KSPCreate(PETSC_COMM_WORLD, &(ctx->kspAuu));
    KSPSetOperators(ctx->kspAuu, ctx->Auu, ctx->Auu);
    KSPSetType(ctx->kspAuu, KSPGMRES);
    KSPSetErrorIfNotConverged(ctx->kspAuu, PETSC_TRUE);
    KSPSetConvergenceTest(ctx->kspAuu, convergence_test_Auu, nullptr, nullptr);
    KSPSetFromOptions(ctx->kspAuu);
    KSPSetUp(ctx->kspAuu);

    PC             pc;

    PetscOptionsClear(NULL);

    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
    PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    // PetscOptionsSetValue(NULL, "-pc_type", "lu");
    // PetscOptionsSetValue(NULL, "-pc_type", "none");
    PetscOptionsSetValue(NULL, "-pc_type", "shell");
    // PetscOptionsSetValue(NULL, "-pc_type", "bjacobi");
    // PetscOptionsSetValue(NULL, "-sub_pc_type", "eisenstat");
    // PetscOptionsSetValue(NULL, "-pc_type", "hypre");
    // PetscOptionsSetValue(NULL, "-pc_hypre_type", "boomeramg");
    PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_statistics", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "modifiedRuge-Stueben");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+e-mm");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "1");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor", "0.2");
    // PetscOptionsSetValue(NULL, "-ksp_monitor", "");
    // PetscOptionsSetValue(NULL, "-ksp_view", "");

    MatNullSpace nullspace;

    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
    MatSetNearNullSpace(ctx->App, nullspace);
    MatSetNullSpace(ctx->App, nullspace);
    MatNullSpaceDestroy(&nullspace);

    KSPCreate(PETSC_COMM_WORLD, &(ctx->kspApp));
    KSPSetOperators(ctx->kspApp, ctx->App, ctx->App);
    KSPSetType(ctx->kspApp, KSPGMRES);
    KSPGetPC(ctx->kspApp, &pc);
    PCSetType(pc, PCSHELL);
    PCShellSetApply(pc, Stokes_Schur_Mult_precond_apply);
    PCShellSetContext(pc, ctx);
    PCSetUp(pc);
    KSPSetErrorIfNotConverged(ctx->kspApp, PETSC_TRUE);
    KSPSetConvergenceTest(ctx->kspApp, convergence_test_Apu_Auu_inv_Aup, nullptr, nullptr);
    KSPSetFromOptions(ctx->kspApp);
    KSPSetUp(ctx->kspApp);

    return std::make_tuple(ctx);
}