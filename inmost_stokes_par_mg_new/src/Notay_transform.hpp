#pragma once
#include "petsc.h"
#include <tuple>
#include "submatrix_manager.hpp"

std::tuple<Mat, Mat, Mat, Mat, Mat, Mat> notay_transform(Mat Auu, Mat Aup, Mat Apu, Mat App, SubmatrixManager &submatrix_map);

void notay_transform_submatrices(std::map< std::string, Mat > &submatrices, std::map< std::string, IS > &index_sequences, SubmatrixManager &submatrix_map );

void prepare_main_block_diag(std::map< std::string, Mat > &submatrices, std::map< std::string, IS > &index_sequences);

std::tuple<Mat, Mat, Mat, Mat, Mat, Mat> notay_transform_right(Mat Auu, Mat Aup, Mat Apu, Mat App, SubmatrixManager &submatrix_map);