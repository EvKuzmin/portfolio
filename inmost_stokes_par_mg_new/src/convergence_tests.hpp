#pragma once

#include "petsc.h"

struct ConvergenceTestParameters {
    int i_max;
    double relative_residual;
    double absolute_residual;
    std::string matrix_name;
    std::string indent;
    bool view_enabled;
    bool error_if_over_i_max;
};

PetscErrorCode convergence_test_smoother(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx);
PetscErrorCode convergence_test_full(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx);