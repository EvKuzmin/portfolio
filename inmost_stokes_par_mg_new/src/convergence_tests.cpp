
#include "convergence_tests.hpp"

#include <iostream>

PetscErrorCode convergence_test_smoother(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
{
    ConvergenceTestParameters *params = (ConvergenceTestParameters *) mctx;
    *reason = KSP_CONVERGED_ITERATING;

    if (it > params->i_max) {
        *reason = KSP_CONVERGED_ITS;
    }

    return 0;
}

PetscErrorCode convergence_test_full(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
{
    ConvergenceTestParameters *params = (ConvergenceTestParameters *) mctx;
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
    

    VecAXPY(Ax, -1.0, rhs);

    double residual_Ax = 0.0;

    VecNorm(Ax, NORM_2, &residual_Ax);

    double f_norm = 0.0;
    VecNorm(rhs, NORM_2, &f_norm);

    double relative_residual = 0.0;

    if( params->view_enabled ) std::cout << params->indent << params->matrix_name << " rnorm = " << rnorm << "\n";

    if( rnorm == 0.0 ) *reason = KSP_CONVERGED_ITS;

    if( f_norm != 0.0 ) {
        relative_residual = residual_Ax / f_norm;
        if( params->view_enabled ) std::cout << params->indent << "it = " << it << " " << params->matrix_name << " f_norm = " << f_norm << "\n";
        if( params->view_enabled ) std::cout << params->indent << "it = " << it << " " << params->matrix_name << " relative = " << relative_residual << "\n";
        if (relative_residual < params->relative_residual)
        {
            if( params->view_enabled ) std::cout << params->indent << params->matrix_name << " iters = " << it << "\n";
            *reason = KSP_CONVERGED_ITS;
        }
    }

    if (it > 0 && residual_Ax < params->absolute_residual)
        {
            if( params->view_enabled ) std::cout << params->indent << params->matrix_name << " iters = " << it << "\n";
            *reason = KSP_CONVERGED_ITS;
        }

    // if (residual_Ax < 1e-1)
    // {
    //     *reason = KSP_CONVERGED_ITS;
    // }

    // VecView(x, PETSC_VIEWER_STDOUT_WORLD);
        // write_func_immediate(A);
        // MPI_Barrier(MPI_COMM_WORLD);

    if (it > params->i_max) {
        // VecView(x, PETSC_VIEWER_STDOUT_WORLD);
        // write_func_immediate(A);
        // MPI_Barrier(MPI_COMM_WORLD);
        // std::exit(0);
        if(params->error_if_over_i_max) {
            *reason = KSP_DIVERGED_ITS;
        } else {
            // if( params->matrix_name == "smoother coarse level 3" ) {
            //     // VecView(x, PETSC_VIEWER_STDOUT_WORLD);
            //     VecView(rhs, PETSC_VIEWER_STDOUT_WORLD);
            //     MPI_Barrier(MPI_COMM_WORLD);
            //     std::exit(0);
            // }
            *reason = KSP_CONVERGED_ITS;
        }
        
    }

    PC             pc;
    KSPGetPC(ksp, &pc);

    PCFailedReason pcreason;
    PCGetFailedReason(pc,&pcreason);

    if(pcreason != PC_NOERROR) {
        *reason = KSP_DIVERGED_ITS;
    }

    if( params->view_enabled ) std::cout << params->indent << "it = " << it << " " << params->matrix_name << " absolute = " << residual_Ax << "\n";
    VecDestroy(&Ax);
    return 0;
}