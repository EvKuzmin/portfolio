#include "petsc.h"
#include <map>
#include <vector>
#include <string>
#include <tuple>
#include "Sparse_alpha.hpp"
std::tuple<std::vector<int>, std::vector<int>> get_coarse_A(std::vector<int> coarse_local_index_wo, std::vector<int> J, std::vector<int> narray);
std::tuple<Mat, Mat, int, int, int, int> prolongx( Mat A_wo_main_diag, int face_unknowns_local_wo, std::vector<int> &coarse_local_index_wo, int overflow);
CRSMatrix prolongx_change(CRS_like_petsc A_com, std::vector<int> &coarse_local_index_wo, int overflow);
void set_value(COOMatrix& A, double val, int i, int j);
void print_prolong(CRSMatrix Prolong, int rank);
void print_prolong_petsc(Mat A_wo_main_diag);