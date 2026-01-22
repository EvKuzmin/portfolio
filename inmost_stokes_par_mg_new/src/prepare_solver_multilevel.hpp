#pragma once

#include"petsc.h"
#include "solver_tree.hpp"
#include<tuple>

struct PrecondContextMultilevel : public SolverContext {
    Mat A11;
    Mat A12;
    Mat A21;
    Mat A22;
    MatNullSpace nullspace;
    // Mat Apu_Aup;
    Vec b;
    IS IS_1;
    IS IS_2;
    Vec V1_1, V1_2, V1_3;
    Vec V2_1, V2_2, V2_3;
    KSP ksp_main_block;
    KSP ksp_schur;
    std::string name;
    PrecondContextMultilevel( Mat A11, Mat A12, Mat A21, Mat A22, MatNullSpace nullspace, IS IS_1, IS IS_2 );
    // KSP kspApu_Aup;
    virtual ~PrecondContextMultilevel();
};

// std::tuple< KSP > prepare_multilevel_solver( Mat A );
PetscErrorCode precond_multilevel11_apply(PC pc, Vec x, Vec y);
PetscErrorCode precond_multilevel21_apply(PC pc, Vec x, Vec y);