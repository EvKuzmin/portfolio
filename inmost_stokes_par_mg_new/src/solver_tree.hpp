#pragma once

#include"petsc.h"
#include<vector>
#include<functional>
#include<map>
#include<string>
#include<any>
#include"submatrix_manager.hpp"

struct SolverContext {
    virtual ~SolverContext();
};

struct SolverTree {
    Mat A;
    std::vector<SolverTree *> child;
    SubmatrixManager *submatrix_map;
    std::map<std::string, Mat> *submatrices;
    std::map<std::string, IS> *index_sequences;
    std::string solver_type;
    std::string precond_type;
    std::map<std::string, std::string> petsc_options;
    std::map<std::string, std::any> properties;
    SolverContext *ctx;
    KSP ksp;
    static void visit_nodes( SolverTree *node, std::function< void(SolverTree*) > visitor );
};