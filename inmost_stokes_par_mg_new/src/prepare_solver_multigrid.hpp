#pragma once

#include "petsc.h"

#include<tuple>

#include"solver_tree.hpp"

struct PrecondContextMultigrid : public SolverContext {

    std::map< std::string, Mat > matrices;
    std::map< std::string, Vec > vectors;

    std::vector< std::map< std::string, IS > > index_sequences_sub_list;
    std::vector< std::map< std::string, Mat > > submatrices_sub_list;

    virtual ~PrecondContextMultigrid();
};

// std::tuple< Mat, Vec, Vec > prepare_matrix(int matrix_size_local, int matrix_size_sum, int matrix_size_global, int prealloc_diag, int prealloc_off_diag);