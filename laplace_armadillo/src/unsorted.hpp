#pragma once

#include "armadillo"
#include <vector>

std::vector<int> coarsening_MIS(arma::sp_mat A, int dist);

arma::sp_mat SPAI(const arma::sp_mat& A_wo_d);

struct MatrixBlocks {
    arma::sp_mat Aff;
    arma::sp_mat Afc;
    arma::sp_mat Acf;
    arma::sp_mat Acc;
};

MatrixBlocks extract_sparse_block(arma::sp_mat &A, const arma::uvec &fine_elements_b, const arma::uvec &coarse_elements_b, const arma::uvec &fine_elements_prefix_sum, const arma::uvec &coarse_elements_prefix_sum, int N_fine, int N_coarse);