#pragma once

#include "petsc.h"
#include <vector>
#include <tuple>
#include <map>

std::tuple<Mat, Mat, int, int, int, int> interp_prolong_distance_scaled( Mat A_wo_main_diag, std::vector<int> &coarse_local_index_wo, std::vector<int> &unknowns_local, std::vector< std::tuple< double, double, double > >& all_center_coordinates, std::vector< std::tuple< double, double, double > >& coarse_all_center_coordinates );

std::tuple<Mat, Mat, int, int, int, int> interp_prolong_distance_scaled_cached( std::string name, Mat A_wo_main_diag, std::vector<int> &coarse_local_index_wo, std::vector<int> &unknowns_local, std::map< std::string, std::tuple<Mat, Mat, int, int, int, int> > &cache, std::vector< std::tuple< double, double, double > >& all_center_coordinates, std::vector< std::tuple< double, double, double > >& coarse_all_center_coordinates );