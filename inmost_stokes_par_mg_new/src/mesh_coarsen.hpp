#pragma once

#include<tuple>
#include<vector>
#include<map>
#include<string>
#include"petsc.h"
#include"submatrix_manager.hpp"
#include "inmost.h"

using namespace INMOST;

std::tuple< std::vector<int>, std::vector<int> > mesh_coarsen(Mesh *m, Mat A, std::map< std::string, Mat > &submatrices, SubmatrixManager &submatrix_map, std::vector<unsigned int> &cell_id_local_to_handle, int face_unknowns_local, int lapl_matrix_size_sum);