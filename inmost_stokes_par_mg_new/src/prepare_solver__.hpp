#pragma once

#include"petsc.h"
#include<string>
#include"solver_tree.hpp"

void prepare_nested_solver(SolverTree *node, KSP parent_ksp, PC parent_pc, std::string parent_pc_type);
void prepare_pc(SolverTree *node, PC pc);