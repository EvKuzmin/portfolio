#pragma once

#include"petsc.h"
#include<map>
#include<string>
#include"submatrix_manager.hpp"
#include<tuple>

std::tuple<Mat, Mat, int, int, int, int> simple_interp_prolong_kuzmin( Mat A_wo_main_diag, int face_unknowns_local_wo, std::vector<int> &coarse_local_index_wo );