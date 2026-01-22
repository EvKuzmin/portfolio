#pragma once

#include<map>
#include<string>
#include<tuple>
#include"petsc.h"
#include"submatrix_manager.hpp"

std::tuple< Mat, Mat > SPAI_prolongation( std::map< std::string, Mat > &submatrices, std::map< std::string, IS > &index_sequences, SubmatrixManager &submatrix_map );

void check_prolongation_operator(std::map< std::string, Mat > &submatrices);