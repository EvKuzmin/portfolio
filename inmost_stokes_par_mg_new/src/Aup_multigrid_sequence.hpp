#pragma once

#include <iostream>
#include "petsc.h"
#include "common.hpp"

struct PrecondContextAup {
    Mat Aup_;
    Mat Aff;
    Mat Acf_C;
    Mat Afc_P;
    Mat Ac;
    Vec b;
    IS IS_F;
    IS IS_C;
    IS IS_C_U;
    IS IS_C_P;
    IS IS_U;
    IS IS_P;
    Vec Vf1, Vf2, Vf3;
    Vec Vc1, Vc2, Vc3;
    // std::vector<double> Auu_res;
    // std::vector<double> Apu_Auu_inv_Aup_res;
    KSP kspAff;
    KSP kspAc;
    KSP kspFull;
    ~PrecondContextAup() {
        // MatDestroy(&Apu_Auu_inv_Aup);
        KSPDestroy(&kspAff);
        KSPDestroy(&kspAc);
        KSPDestroy(&kspFull);
        VecDestroy(&Vf1);
        VecDestroy(&Vf2);
        VecDestroy(&Vf3);
        VecDestroy(&Vc1);
        VecDestroy(&Vc2);
        VecDestroy(&Vc3);
    }
};

PetscErrorCode precond_Aup_apply(PC pc, Vec x, Vec y){
    PrecondContextAup *ctx;
    PCShellGetContext(pc,&ctx);

    VecZeroEntries(ctx->Vf1);
    VecZeroEntries(ctx->Vf2);
    VecZeroEntries(ctx->Vf3);
    VecZeroEntries(ctx->Vc1);
    VecZeroEntries(ctx->Vc2);
    VecZeroEntries(ctx->Vc3);
    Vec residual;
    Vec residual_solve;
    MatCreateVecs(ctx->Aup_, nullptr, &residual);
    MatCreateVecs(ctx->Aup_, nullptr, &residual_solve);

    KSPSolve(ctx->kspFull, x, y);

    MatMult( ctx->Aup_, y, residual );
    VecAYPX( residual, -1.0, x );

    Vec bf, bc;
    Vec yf, yc;
    VecGetSubVector(residual, ctx->IS_F, &bf);
    VecGetSubVector(residual, ctx->IS_C, &bc);
    VecGetSubVector(residual_solve, ctx->IS_F, &yf);
    VecGetSubVector(residual_solve, ctx->IS_C, &yc);

    KSPConvergedReason reason;

    KSPSolve(ctx->kspAff, bf, yf); // Vf1 = Aff^-1 bf
    KSPGetConvergedReason(ctx->kspAff, &reason);
    if(reason < 0) {
        std::cerr << "Aff diverged\n";
        VecRestoreSubVector(x, ctx->IS_F, &bf);
        VecRestoreSubVector(x, ctx->IS_C, &bc);
        VecRestoreSubVector(y, ctx->IS_F, &yf);
        VecRestoreSubVector(y, ctx->IS_C, &yc);
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }

    VecZeroEntries(yc);

    VecRestoreSubVector(residual, ctx->IS_F, &bf);
    VecRestoreSubVector(residual, ctx->IS_C, &bc);
    VecRestoreSubVector(residual_solve, ctx->IS_F, &yf);
    VecRestoreSubVector(residual_solve, ctx->IS_C, &yc);

    VecAXPY( y, 1.0, residual_solve );

    VecZeroEntries(residual);
    VecZeroEntries(residual_solve);

    MatMult( ctx->Aup_, y, residual );
    VecAYPX( residual, -1.0, x );

    VecGetSubVector(residual, ctx->IS_F, &bf);
    VecGetSubVector(residual, ctx->IS_C, &bc);
    VecGetSubVector(residual_solve, ctx->IS_F, &yf);
    VecGetSubVector(residual_solve, ctx->IS_C, &yc);
    
    MatMult(ctx->Acf_C, bf, ctx->Vc1); // Vc1 = Acf Aff^-1 bf

    VecCopy(bc, ctx->Vc2);
    VecAXPY(ctx->Vc2, 1.0, ctx->Vc1); // Vc2 = bc - Acf Aff^-1 bf

    KSPSolve(ctx->kspAc, ctx->Vc2, ctx->Vc3); // Vc3 = Ac^{-1} (bc - Acf Aff^-1 bf)
    KSPGetConvergedReason(ctx->kspAc, &reason);
    if(reason < 0) {
        std::cerr << "Ac diverged\n";
        VecRestoreSubVector(x, ctx->IS_F, &bf);
        VecRestoreSubVector(x, ctx->IS_C, &bc);
        VecRestoreSubVector(y, ctx->IS_F, &yf);
        VecRestoreSubVector(y, ctx->IS_C, &yc);
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }

    Vec vc3_p;

    VecGetSubVector(ctx->Vc3, ctx->IS_C_P, &vc3_p);

    MatNullSpace nullspace;
    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
    MatNullSpaceRemove(nullspace, vc3_p);
    MatNullSpaceDestroy(&nullspace);

    VecRestoreSubVector(ctx->Vc3, ctx->IS_C_P, &vc3_p);

    MatMult(ctx->Afc_P, ctx->Vc3, ctx->Vf2); // Vf2 = Afc Ac^{-1} (bc - Acf Aff^-1 bf)
    // KSPSolve(ctx->kspAff, ctx->Vf2, ctx->Vf3); // Vf3 = Aff^-1 Afc Ac^{-1} (bc - Acf Aff^-1 bf)
    // KSPGetConvergedReason(ctx->kspAff, &reason);
    // if(reason < 0) {
    //     std::cerr << "Aff diverged\n";
    //     VecRestoreSubVector(x, ctx->IS_F, &bf);
    //     VecRestoreSubVector(x, ctx->IS_C, &bc);
    //     VecRestoreSubVector(y, ctx->IS_F, &yf);
    //     VecRestoreSubVector(y, ctx->IS_C, &yc);
    //     // throw std::runtime_error("something not converged\n");
    //     PCSetFailedReason(pc, PC_FACTOR_OTHER);
    //     return 0;
    // }

    VecCopy( ctx->Vc3, yc );

    VecCopy( ctx->Vf2, yf );
    // VecAXPY(yf, 1.0, ctx->Vf2);

    VecRestoreSubVector(residual, ctx->IS_F, &bf);
    VecRestoreSubVector(residual, ctx->IS_C, &bc);
    VecRestoreSubVector(residual_solve, ctx->IS_F, &yf);
    VecRestoreSubVector(residual_solve, ctx->IS_C, &yc);

    VecAXPY( y, 1.0, residual_solve );

    // VecGetSubVector(y, ctx->IS_P, &yc);

    // MatNullSpace nullspace;
    // MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
    // MatNullSpaceRemove(nullspace, yc);
    // MatNullSpaceDestroy(&nullspace);

    // VecRestoreSubVector(y, ctx->IS_P, &yc);

    PCSetFailedReason(pc, PC_NOERROR);
    VecDestroy(&residual);
    VecDestroy(&residual_solve);
    // std::exit(0);
    return 0;
}