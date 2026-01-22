#pragma once
#include<vector>
#include<tuple>
#include<string>

#include "petsc.h"

std::vector<int> find_dirichlet_csr( Mat A );

void apply_dirichlet_csr(
    Mat A, 
    Vec rhs,
    const std::vector<int>& dirichlet_rows
    );