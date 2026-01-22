
#include "prepare_solver_multilevel.hpp"
#include "convergence_tests.hpp"


#include <iostream>
#include <vector>

PrecondContextMultilevel::~PrecondContextMultilevel() {
    // MatDestroy(&Apu_Aup);
    KSPDestroy(&ksp_main_block);
    KSPDestroy(&ksp_schur);
    // MatNullSpaceDestroy(&nullspace);
    // KSPDestroy(&kspApu_Aup);
    VecDestroy(&V1_1);
    VecDestroy(&V1_2);
    VecDestroy(&V1_3);
    VecDestroy(&V2_1);
    VecDestroy(&V2_2);
    VecDestroy(&V2_3);
}

PrecondContextMultilevel::PrecondContextMultilevel( Mat A11, Mat A12, Mat A21, Mat A22, MatNullSpace nullspace, IS IS_1, IS IS_2 ) : A11(A11), A12(A12), A21(A21), A22(A22), nullspace(nullspace), IS_1(IS_1), IS_2(IS_2) {
    MatCreateVecs(A11, NULL, &V1_1);
    MatCreateVecs(A11, NULL, &V1_2);
    MatCreateVecs(A11, NULL, &V1_3);
    MatCreateVecs(A22, NULL, &V2_1);
    MatCreateVecs(A22, NULL, &V2_2);
    MatCreateVecs(A22, NULL, &V2_3);
}

PetscErrorCode precond_multilevel11_apply(PC pc, Vec x, Vec y){

    PrecondContextMultilevel *ctx;
    PCShellGetContext(pc,&ctx);

    VecZeroEntries(ctx->V1_1);
    VecZeroEntries(ctx->V1_2);
    VecZeroEntries(ctx->V1_3);
    VecZeroEntries(ctx->V2_1);
    VecZeroEntries(ctx->V2_2);
    VecZeroEntries(ctx->V2_3);

    Vec b1, b2;
    Vec y1, y2;
    VecGetSubVector(x, ctx->IS_1, &b1);
    VecGetSubVector(x, ctx->IS_2, &b2);
    VecGetSubVector(y, ctx->IS_1, &y1);
    VecGetSubVector(y, ctx->IS_2, &y2);

    KSPConvergedReason reason;
    
    KSPSolve(ctx->ksp_main_block, b1, ctx->V1_1); //V1_1 = A11^-1 b1
    KSPGetConvergedReason(ctx->ksp_main_block, &reason);
    if(reason < 0) {
        std::cerr << ctx->name << " A11 diverged\n";
        VecRestoreSubVector(x, ctx->IS_1, &b1);
        VecRestoreSubVector(x, ctx->IS_2, &b2);
        VecRestoreSubVector(y, ctx->IS_1, &y1);
        VecRestoreSubVector(y, ctx->IS_2, &y2);
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }

    MatMult(ctx->A21, ctx->V1_1, ctx->V2_2); //V2_2 = A21 A11^-1 b1

    VecCopy(b2, ctx->V2_1);
    VecAXPY(ctx->V2_1, -1.0, ctx->V2_2); //V2_1 = -A21 A11^-1 b1 + b2

    VecZeroEntries(ctx->V2_2);

    if(ctx->nullspace) {
        MatNullSpaceRemove(ctx->nullspace, ctx->V2_1);
    }
    
    KSPSolve(ctx->ksp_schur, ctx->V2_1, y2); //y2 = A22^-1 (-A21 A11^-1 b1 + b2)

    // MPI_Barrier(MPI_COMM_WORLD);
    // std::exit(0);
    KSPGetConvergedReason(ctx->ksp_schur, &reason);
    if(reason < 0) {
        std::cerr << "Aup Schur diverged\n";
        VecRestoreSubVector(x, ctx->IS_1, &b1);
        VecRestoreSubVector(x, ctx->IS_2, &b2);
        VecRestoreSubVector(y, ctx->IS_1, &y1);
        VecRestoreSubVector(y, ctx->IS_2, &y2);
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }

    VecZeroEntries(ctx->V1_3);
    MatMult(ctx->A12, y2, ctx->V1_3); //V1_3 = A12 A22^-1(-A21 A11^-1 b1 + b2)
    // calc_vec_norm(ctx->Vu3, "B S^-1(-B^T A^-1 bu + bp)");
    VecZeroEntries(ctx->V1_2);
    KSPSolve(ctx->ksp_main_block, ctx->V1_3, ctx->V1_2); // V1_2 = A11^-1 A12 A22^-1(-A21 A11^-1 b1 + b2)
    // VecScale(ctx->V1_2, -1.0); //
    KSPGetConvergedReason(ctx->ksp_main_block, &reason);
    if(reason < 0) {
        std::cerr << "Vu2 = A^-1 B S^-1(-B^T A^-1 bu + bp) diverged\n";
        VecRestoreSubVector(x, ctx->IS_1, &b1);
        VecRestoreSubVector(x, ctx->IS_2, &b2);
        VecRestoreSubVector(y, ctx->IS_1, &y1);
        VecRestoreSubVector(y, ctx->IS_2, &y2);
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }
    VecCopy(ctx->V1_1, y1); // y1 = V1_1 = A11^-1 b1
    VecAXPY(y1, -1.0, ctx->V1_2); // y1 = A11^-1 b1 - A11^-1 A12 A22^-1(-A21 A11^-1 b1 + b2)

    VecRestoreSubVector(x, ctx->IS_1, &b1);
    VecRestoreSubVector(x, ctx->IS_2, &b2);
    VecRestoreSubVector(y, ctx->IS_1, &y1);
    VecRestoreSubVector(y, ctx->IS_2, &y2);


    PCSetFailedReason(pc, PC_NOERROR);
    return 0;
}

PetscErrorCode precond_multilevel21_apply(PC pc, Vec x, Vec y){

    PrecondContextMultilevel *ctx;
    PCShellGetContext(pc,&ctx);

    VecZeroEntries(ctx->V1_1);
    VecZeroEntries(ctx->V1_2);
    VecZeroEntries(ctx->V1_3);
    VecZeroEntries(ctx->V2_1);
    VecZeroEntries(ctx->V2_2);
    VecZeroEntries(ctx->V2_3);

    Vec b1, b2;
    Vec y1, y2;
    VecGetSubVector(x, ctx->IS_1, &b1);
    VecGetSubVector(x, ctx->IS_2, &b2);
    VecGetSubVector(y, ctx->IS_1, &y1);
    VecGetSubVector(y, ctx->IS_2, &y2);

    KSPConvergedReason reason;
    
    KSPSolve(ctx->ksp_main_block, b2, ctx->V1_1); //V1_1 = A21^-1 b2
    KSPGetConvergedReason(ctx->ksp_main_block, &reason);
    if(reason < 0) {
        std::cerr << ctx->name << " A11 diverged\n";
        VecRestoreSubVector(x, ctx->IS_1, &b1);
        VecRestoreSubVector(x, ctx->IS_2, &b2);
        VecRestoreSubVector(y, ctx->IS_1, &y1);
        VecRestoreSubVector(y, ctx->IS_2, &y2);
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }

    MatMult(ctx->A11, ctx->V1_1, ctx->V2_2); //V2_2 = A11 A21^-1 b2

    VecCopy(b1, ctx->V2_1);
    VecAXPY(ctx->V2_1, -1.0, ctx->V2_2); //V2_1 = -A11 A21^-1 b2 + b1

    VecZeroEntries(ctx->V2_2);

    if(ctx->nullspace) {
        MatNullSpaceRemove(ctx->nullspace, ctx->V2_1);
    }
    
    KSPSolve(ctx->ksp_schur, ctx->V2_1, y2); //y2 = A12^-1 (-A11 A21^-1 b2 + b1)

    // MPI_Barrier(MPI_COMM_WORLD);
    // std::exit(0);
    KSPGetConvergedReason(ctx->ksp_schur, &reason);
    if(reason < 0) {
        std::cerr << "Aup Schur diverged\n";
        VecRestoreSubVector(x, ctx->IS_1, &b1);
        VecRestoreSubVector(x, ctx->IS_2, &b2);
        VecRestoreSubVector(y, ctx->IS_1, &y1);
        VecRestoreSubVector(y, ctx->IS_2, &y2);
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }

    VecZeroEntries(ctx->V1_3);
    MatMult(ctx->A22, y2, ctx->V1_3); //V1_3 = A12 A22^-1(-A21 A11^-1 b1 + b2)
    // calc_vec_norm(ctx->Vu3, "B S^-1(-B^T A^-1 bu + bp)");
    VecZeroEntries(ctx->V1_2);
    KSPSolve(ctx->ksp_main_block, ctx->V1_3, ctx->V1_2); // V1_2 = A11^-1 A12 A22^-1(-A21 A11^-1 b1 + b2)
    // VecScale(ctx->V1_2, -1.0); //
    KSPGetConvergedReason(ctx->ksp_main_block, &reason);
    if(reason < 0) {
        std::cerr << "Vu2 = A^-1 B S^-1(-B^T A^-1 bu + bp) diverged\n";
        VecRestoreSubVector(x, ctx->IS_1, &b1);
        VecRestoreSubVector(x, ctx->IS_2, &b2);
        VecRestoreSubVector(y, ctx->IS_1, &y1);
        VecRestoreSubVector(y, ctx->IS_2, &y2);
        // throw std::runtime_error("something not converged\n");
        PCSetFailedReason(pc, PC_FACTOR_OTHER);
        return 0;
    }
    VecCopy(ctx->V1_1, y1); // y1 = V1_1 = A11^-1 b1
    VecAXPY(y1, -1.0, ctx->V1_2); // y1 = A11^-1 b1 - A11^-1 A12 A22^-1(-A21 A11^-1 b1 + b2)

    VecRestoreSubVector(x, ctx->IS_1, &b1);
    VecRestoreSubVector(x, ctx->IS_2, &b2);
    VecRestoreSubVector(y, ctx->IS_1, &y1);
    VecRestoreSubVector(y, ctx->IS_2, &y2);


    PCSetFailedReason(pc, PC_NOERROR);
    return 0;
}

// std::tuple< KSP > prepare_multilevel_solver( Mat A )
// {
//     KSP            ksp;
//     PC             pc;
//     // Vec            x;
//     // VecCreate(PETSC_COMM_WORLD, &x);
//     // VecSetSizes(x, N, N);
//     // VecSetFromOptions(x);
//     // VecSet(x,0.0);

//     auto [ctx_full] = prepare_preconditioner_context( submatrices, index_sequences, submatrix_map, cell_center_coordinates, cell_to_face_local, func );

//     PetscOptionsClear(NULL);
//     PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
//     PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
//     PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
//     PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
//     PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
//     PetscOptionsSetValue(NULL, "-pc_type", "shell");
//     PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
//     PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "70");
//     // PetscOptionsSetValue(NULL, "-ksp_monitor", "");

//     KSPCreate(PETSC_COMM_WORLD,&ksp);

//     KSPSetOperators(ksp,A,A);
//     KSPSetType(ksp,KSPFGMRES);

//     KSPGetPC(ksp, &pc);
//     PCSetType(pc, PCSHELL);
//     KSPSetConvergenceTest(ksp, convergence_test_full, ctx_full, NULL);
//     // PCShellSetApply(pc, precond_Full_apply);
//     PCShellSetApply(pc, precond_Full_CHB_apply);
//     PCShellSetContext(pc, ctx_full);
//     PCSetUp(pc);
//     KSPSetFromOptions(ksp);
//     KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
//     KSPSetErrorIfNotConverged(ksp, PETSC_FALSE);
//     KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);

//     PetscOptionsClear(NULL);
//     PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
//     PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
//     PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
//     // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_2", "1.0e-15");
//     PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
//     PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
//     PetscOptionsSetValue(NULL, "-pc_type", "none");
//     PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
//     // PetscOptionsSetValue(NULL, "-ksp_monitor", "");


//     return std::make_tuple( ksp );
// }