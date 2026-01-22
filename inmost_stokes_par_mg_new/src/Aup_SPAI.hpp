#pragma once

#include <iostream>
#include "petsc.h"
#include "common.hpp"

struct PrecondContextAup {
    Mat Aup_;
    Mat Auu;
    Mat Aup;
    Mat Apu;
    Mat Apu_Auu_inv_Aup;
    Mat Apu_Auu_inv_Aup_SPAI;
    Vec b;
    IS IS_U;
    IS IS_P;
    Vec Vu1, Vu2, Vu3;
    Vec Vp1, Vp2, Vp3;
    // std::vector<double> Auu_res;
    // std::vector<double> Apu_Auu_inv_Aup_res;
    KSP kspAuu;
    KSP kspApu_Auu_inv_Aup;
    ~PrecondContextAup() {
        // MatDestroy(&Apu_Auu_inv_Aup);
        KSPDestroy(&kspAuu);
        KSPDestroy(&kspApu_Auu_inv_Aup);
        VecDestroy(&Vu1);
        VecDestroy(&Vu2);
        VecDestroy(&Vu3);
        VecDestroy(&Vp1);
        VecDestroy(&Vp2);
        VecDestroy(&Vp3);
    }
};

PetscErrorCode precond_Aup_apply(PC pc, Vec x, Vec y){

    std::cout << "precond_Aup_apply\n";

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

    auto calc_residual_block = [](Mat Auu, Mat Aup, Mat Apu, IS IS_u, IS IS_p, Vec x, Vec b, std::string name){
        Vec Ax;
        VecDuplicate(b, &Ax);
        VecSet(Ax, 0.0);

        Vec Ax_u, Ax_p;
        Vec x_u, x_p;

        VecGetSubVector(Ax, IS_u, &Ax_u);
        VecGetSubVector(Ax, IS_p, &Ax_p);
        VecGetSubVector(x, IS_u, &x_u);
        VecGetSubVector(x, IS_p, &x_p);

        Vec vu, vp;

        VecDuplicate(Ax_u, &vu);
        VecSet(vu, 0.0);
        VecDuplicate(Ax_p, &vp);
        VecSet(vp, 0.0);

        MatMult(Auu, x_u, vu);
        MatMult(Aup, x_p, Ax_u);
        VecAXPY(Ax_u, 1.0, vu);

        MatMult(Apu, x_u, Ax_p);

        VecRestoreSubVector(Ax, IS_u, &Ax_u);
        VecRestoreSubVector(Ax, IS_p, &Ax_p);
        VecRestoreSubVector(x, IS_u, &x_u);
        VecRestoreSubVector(x, IS_p, &x_p);

        VecAXPY(Ax, -1.0, b);

        double residual_Ax = 0.0;

        VecNorm(Ax, NORM_2, &residual_Ax);

        std::cout << "residual_block " << name << " = " << residual_Ax << "\n";
        VecDestroy(&Ax);
        VecDestroy(&vu);
        VecDestroy(&vp);
    };

    auto check_schur = [](Mat Auu, Mat Aup, Mat Apu, Vec x, Vec b, std::string name){
        KSP ksp;
        PetscOptionsClear(NULL);
        PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
        PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
        PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
        // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_2", "1.0e-15");
        PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
        PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
        // PetscOptionsSetValue(NULL, "-pc_mg_levels", "2");
        PetscOptionsSetValue(NULL, "-pc_type", "ilu");

        KSPCreate(PETSC_COMM_WORLD,&ksp);
        KSPSetOperators(ksp,Auu,Auu);
        KSPSetType(ksp,KSPPREONLY);
        KSPSetFromOptions(ksp);

        Vec v1, v2;

        MatCreateVecs(Aup, &v1, &v2);

        // VecDuplicate(b, &v1);
        // VecSet(vu, 0.0);
        // VecDuplicate(b, &v2);
        // VecSet(vp, 0.0);

        MatMult(Aup, x, v2);

        Vec v3;
        VecDuplicate(v2, &v3);
        VecSet(v3, 0.0);

        KSPSolve(ksp, v2, v3);

        MatMult(Apu, v3, v1);

        VecAXPY(v1, -1.0, b);

        // VecView(v1, PETSC_VIEWER_STDOUT_SELF);

        double residual_Ax = 0.0;

        VecNorm(v1, NORM_2, &residual_Ax);

        std::cout << "check_schur " << name << " = " << residual_Ax << "\n";
        KSPDestroy(&ksp);
        VecDestroy(&v1);
        VecDestroy(&v2);
        VecDestroy(&v3);
    };

    auto calc_mul_residual = [](Mat A, Vec x, Vec b, std::string name){
        Vec x_new;
        VecDuplicate(b, &x_new);
        Vec Ax;
        VecDuplicate(b, &Ax);
        KSP ksp;
        PetscOptionsClear(NULL);
        PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
        PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
        PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
        // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_2", "1.0e-15");
        PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
        PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
        // PetscOptionsSetValue(NULL, "-pc_mg_levels", "2");
        PetscOptionsSetValue(NULL, "-pc_type", "ilu");

        KSPCreate(PETSC_COMM_WORLD,&ksp);
        KSPSetOperators(ksp,A,A);
        KSPSetType(ksp,KSPPREONLY);
        KSPSetFromOptions(ksp);
        // KSPSetConvergenceTest(ksp,convergence_test_Auu,ctx,ConvergenceDestroy);

        KSPSolve(ksp, b, x_new);

        VecAXPY(x_new, -1.0, x);

        double residual_Ax = 0.0;

        VecNorm(x_new, NORM_2, &residual_Ax);

        std::cout << "mul_residual " << name << " = " << residual_Ax << "\n";

        KSPDestroy(&ksp);
        VecDestroy(&x_new);
        VecDestroy(&Ax);
    };

    auto calc_vec_norm = [](Vec x, std::string name){
        double residual_Ax = 0.0;

        VecNorm(x, NORM_2, &residual_Ax);

        std::cout << "Vec " << name << " norm = " << residual_Ax << "\n";
    };

    PrecondContextAup *ctx;
    PCShellGetContext(pc,&ctx);

    VecZeroEntries(ctx->Vu1);
    VecZeroEntries(ctx->Vu2);
    VecZeroEntries(ctx->Vu3);
    VecZeroEntries(ctx->Vp1);
    VecZeroEntries(ctx->Vp2);
    VecZeroEntries(ctx->Vp3);

    Vec bu, bp;
    Vec yu, yp;
    VecGetSubVector(x, ctx->IS_U, &bu);
    VecGetSubVector(x, ctx->IS_P, &bp);
    VecGetSubVector(y, ctx->IS_U, &yu);
    VecGetSubVector(y, ctx->IS_P, &yp);

    KSPConvergedReason reason;

    // write_local_PetSc_matrix(ctx->Auu);
    // write_local_PetSc_vector(bu);
    
    // std::exit(0);

    // KSPSetOperators(ctx->ksp1,ctx->Auu,ctx->Auu);
    // KSPSetUp(ctx->ksp1);
    KSPSolve(ctx->kspAuu, bu, ctx->Vu1); //Vu1 = A^-1 bu
    KSPGetConvergedReason(ctx->kspAuu, &reason);
    if(reason < 0) {
        std::cerr << "Auu diverged\n";
        VecRestoreSubVector(x, ctx->IS_U, &bu);
        VecRestoreSubVector(x, ctx->IS_P, &bp);
        VecRestoreSubVector(y, ctx->IS_U, &yu);
        VecRestoreSubVector(y, ctx->IS_P, &yp);
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }

    // calc_residual( ctx->Auu, ctx->Vu1, bu, "A^-1 bu" );
    // calc_vec_norm(ctx->Vu1, "A^-1 bu");
    // VecScale(ctx->Vu1, -1.0);
    MatMult(ctx->Apu, ctx->Vu1, ctx->Vp2); //Vp2 = B^T A^-1 bu

    VecCopy(bp, ctx->Vp1);
    VecAXPY(ctx->Vp1, -1.0, ctx->Vp2); //Vp1 = -B^T A^-1 bu + bp

    Vec vec_before_schur;
    VecDuplicate(ctx->Vp1, &vec_before_schur);
    VecCopy(ctx->Vp1, vec_before_schur);

    // calc_vec_norm(ctx->Vp2, "B^T A^-1 bu");
    // calc_vec_norm(bp, "bp");
                                        

    // KSPSetOperators(ctx->ksp1,ctx->Auu,ctx->Auu);
    // KSPSetUp(ctx->ksp1);
    // KSPSolve(ctx->kspAuu, bu, ctx->Vu1);
    VecZeroEntries(ctx->Vp2);

    MatNullSpace nullspace;
    MatGetNullSpace(ctx->Apu_Auu_inv_Aup, &nullspace);
    MatNullSpaceRemove(nullspace, ctx->Vp1);

    KSPSolve(ctx->kspApu_Auu_inv_Aup, ctx->Vp1, yp); //Vp2 = (B^T A^-1 B)^-1 (-B^T A^-1 bu + bp)
    KSPGetConvergedReason(ctx->kspApu_Auu_inv_Aup, &reason);
    if(reason < 0) {
        std::cerr << "Aup Schur diverged\n";
        VecRestoreSubVector(x, ctx->IS_U, &bu);
        VecRestoreSubVector(x, ctx->IS_P, &bp);
        VecRestoreSubVector(y, ctx->IS_U, &yu);
        VecRestoreSubVector(y, ctx->IS_P, &yp);
        VecDestroy(&vec_before_schur);
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }
    // calc_vec_norm(yp, "(B^T A^-1 B)^-1 (-B^T A^-1 bu + bp)");
    // calc_residual( ctx->Apu_Auu_inv_Aup, yp, ctx->Vp1, "(B^T A^-1 B)^-1 (-B^T A^-1 bu + bp)" );

    // // VERROU_START_INSTRUMENTATION;
    // KSPSetConvergenceTest(ctx->kspApu_Aup,convergence_test_Apu_Aup_1,ctx,ConvergenceDestroy);
    // KSPSolve(ctx->kspApu_Aup, ctx->Vp1, ctx->Vp2); //Vp2 = (B^T B)^-1 (-B^T A^-1 bu + bp)
    // calc_vec_norm(ctx->Vp2, "(B^T B)^-1 (-B^T A^-1 bu + bp)");
    // // std::exit(0);
    // // VERROU_STOP_INSTRUMENTATION;
    // calc_residual( ctx->Apu_Aup, ctx->Vp2, ctx->Vp1, "(B^T B)^-1 (-B^T A^-1 bu + bp)" );


    // // VecZeroEntries(ctx->Vu2);
    // // MatMult(ctx->Aup, ctx->Vp2, ctx->Vu2); //Vu2 = B(B^T B)^-1 (-B^T A^-1 bu + bp)
    // // VecZeroEntries(ctx->Vu3);
    // // MatMult(ctx->Auu, ctx->Vu2, ctx->Vu3); //Vu3 = A B(B^T B)^-1 (-B^T A^-1 bu + bp)
    // // VecZeroEntries(ctx->Vp1);
    // // MatMult(ctx->Apu, ctx->Vu3, ctx->Vp1); //Vp1 = B^T A B(B^T B)^-1 (-B^T A^-1 bu + bp)

    // VecZeroEntries(ctx->Vp1);
    // MatMult(ctx->Apu_Auu_Aup, ctx->Vp2, ctx->Vp1); //Vp1 = B^T A B(B^T B)^-1 (-B^T A^-1 bu + bp)
    // calc_vec_norm(ctx->Vp1, "B^T A B(B^T B)^-1 (-B^T A^-1 bu + bp)");

    // calc_mul_residual(ctx->Apu_Auu_Aup, ctx->Vp2, ctx->Vp1, "B^T A B(B^T B)^-1 (-B^T A^-1 bu + bp)");
    // // std::exit(0);

    // KSPSetConvergenceTest(ctx->kspApu_Aup,convergence_test_Apu_Aup_2,ctx,ConvergenceDestroy);
    // KSPSolve(ctx->kspApu_Aup, ctx->Vp1, yp); //yp = (B^T B)^-1 B^T A B(B^T B)^-1 (-B^T A^-1 bu + bp)
    // calc_vec_norm(yp, "(B^T B)^-1 B^T A B(B^T B)^-1 (-B^T A^-1 bu + bp)");
    // calc_residual( ctx->Apu_Aup, yp, ctx->Vp1, "(B^T B)^-1 B^T A B(B^T B)^-1 (-B^T A^-1 bu + bp)" );
    // check_schur(ctx->Auu, ctx->Aup, ctx->Apu, yp, vec_before_schur, "(B^T B)^-1 B^T A B(B^T B)^-1 (-B^T A^-1 bu + bp)");
    // // std::exit(0);
    VecScale(yp, -1.0); //yp = -(B^T B)^-1 B^T A B(B^T B)^-1 (-B^T A^-1 bu + bp)

    VecZeroEntries(ctx->Vu3);
    MatMult(ctx->Aup, yp, ctx->Vu3); //Vu3 = B S^-1(-B^T A^-1 bu + bp)
    // calc_vec_norm(ctx->Vu3, "B S^-1(-B^T A^-1 bu + bp)");
    VecZeroEntries(ctx->Vu2);
    KSPSolve(ctx->kspAuu, ctx->Vu3, ctx->Vu2); // Vu2 = A^-1 B S^-1(-B^T A^-1 bu + bp)
    KSPGetConvergedReason(ctx->kspAuu, &reason);
    if(reason < 0) {
        std::cerr << "Vu2 = A^-1 B S^-1(-B^T A^-1 bu + bp) diverged\n";
        VecRestoreSubVector(x, ctx->IS_U, &bu);
        VecRestoreSubVector(x, ctx->IS_P, &bp);
        VecRestoreSubVector(y, ctx->IS_U, &yu);
        VecRestoreSubVector(y, ctx->IS_P, &yp);
        VecDestroy(&vec_before_schur);
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }
    // calc_vec_norm(ctx->Vu2, "A^-1 B S^-1(-B^T A^-1 bu + bp)");
    // calc_residual( ctx->Auu, ctx->Vu2, ctx->Vu3, "A^-1 B S^-1(-B^T A^-1 bu + bp)" );
    // VecZeroEntries(ctx->Vu2);
    VecCopy(ctx->Vu1, yu); // yu = Vu1 = A^-1 bu
    VecAXPY(yu, -1.0, ctx->Vu2); // yu = A^-1 bu - A^-1 B S^-1(-B^T A^-1 bu + bp)        

    VecRestoreSubVector(x, ctx->IS_U, &bu);
    VecRestoreSubVector(x, ctx->IS_P, &bp);
    VecRestoreSubVector(y, ctx->IS_U, &yu);
    VecRestoreSubVector(y, ctx->IS_P, &yp);

    // calc_residual( ctx->Aup_, y, x, "full precond residual" );
    // calc_residual_block(ctx->Auu, ctx->Aup, ctx->Apu, ctx->IS_U, ctx->IS_P, y, x, "full precond residual");
    VecDestroy(&vec_before_schur);

    PCSetFailedReason(pc, PC_NOERROR);
    // std::exit(0);
    return 0;
}