#pragma once

#pragma once

#include <iostream>
#include "petsc.h"
#include "common.hpp"

struct PrecondContextAup {
    Mat Aup_;
    Mat Auu;
    Mat App;
    Mat Aup;
    Mat Apu;
    Mat ASchur_precond;
    Vec b;
    IS IS_U;
    IS IS_P;
    // Vec Vf1, Vf2, Vf3;
    // Vec Vc1, Vc2, Vc3;
    Vec tmp1, tmp2, tmp3;
    // std::vector<double> Auu_res;
    // std::vector<double> Apu_Auu_inv_Aup_res;
    KSP kspAuu;
    KSP kspApp;
    KSP kspASchur_precond;
    ~PrecondContextAup() {
        MatDestroy(&ASchur_precond);
        KSPDestroy(&kspAuu);
        KSPDestroy(&kspApp);
        KSPDestroy(&kspASchur_precond);
        VecDestroy(&tmp1);
        VecDestroy(&tmp2);
        VecDestroy(&tmp3);
        // VecDestroy(&Vf1);
        // VecDestroy(&Vf2);
        // VecDestroy(&Vf3);
        // VecDestroy(&Vc1);
        // VecDestroy(&Vc2);
        // VecDestroy(&Vc3);
    }
};

PetscErrorCode precond_Aup_apply(PC pc, Vec x, Vec y){
    PrecondContextAup *ctx;
    PCShellGetContext(pc,&ctx);


    Vec bu, bp;
    Vec yu, yp;
    VecGetSubVector(x, ctx->IS_U, &bu);
    VecGetSubVector(x, ctx->IS_P, &bp);
    VecGetSubVector(y, ctx->IS_U, &yu);
    VecGetSubVector(y, ctx->IS_P, &yp);

    KSPConvergedReason reason;

    KSPSolve(ctx->kspAuu, bu, yu); 
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

    

    KSPSolve(ctx->kspApp, bp, yp); 
    KSPGetConvergedReason(ctx->kspApp, &reason);
    if(reason < 0) {
        std::cerr << "App diverged\n";
        VecRestoreSubVector(x, ctx->IS_U, &bu);
        VecRestoreSubVector(x, ctx->IS_P, &bp);
        VecRestoreSubVector(y, ctx->IS_U, &yu);
        VecRestoreSubVector(y, ctx->IS_P, &yp);
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }

    MatNullSpace nullspace;
    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
    MatNullSpaceRemove(nullspace, yp);
    MatNullSpaceDestroy(&nullspace);

    VecRestoreSubVector(x, ctx->IS_U, &bu);
    VecRestoreSubVector(x, ctx->IS_P, &bp);
    VecRestoreSubVector(y, ctx->IS_U, &yu);
    VecRestoreSubVector(y, ctx->IS_P, &yp);

    PCSetFailedReason(pc, PC_NOERROR);
    // std::exit(0);
    return 0;
}