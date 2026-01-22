#pragma once

#include"petsc.h"
#include<map>
#include<string>
#include"submatrix_manager.hpp"
#include<tuple>

std::tuple<Mat, Mat> simple_interp_prolongation( std::map< std::string, IS > &index_sequences, std::map< std::string, Mat > &submatrices, std::vector<int> &coarse_local_index_wo, SubmatrixManager &submatrix_map );