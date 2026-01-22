#include"simple_interp_prolong_kuzmin.hpp"

#include<vector>
#include<iostream>
#include "prolongx.hpp"

std::tuple<Mat, Mat, int, int, int, int> simple_interp_prolong_kuzmin( Mat A_wo_main_diag, int face_unknowns_local_wo, std::vector<int> &coarse_local_index_wo )
{
    return prolongx(A_wo_main_diag, 0, coarse_local_index_wo, 0);
}