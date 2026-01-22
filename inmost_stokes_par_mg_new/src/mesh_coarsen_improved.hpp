#pragma once

#include<tuple>
#include<vector>
#include"petsc.h"
#include"submatrix_manager.hpp"
#include "inmost.h"

std::tuple< std::vector<int>, 
            std::vector<int>, 
            std::vector< std::tuple< double, double, double > >, 
            std::vector< std::tuple< int, int, int, int, int, int > >,
            int > 
            mesh_coarsen_i(Mat A_full, 
                           Mat A_lapl, 
                           int face_unknowns_local, 
                           std::vector< std::tuple< double, double, double > > &cell_center_coordinates, 
                           std::vector< std::tuple< int, int, int, int, int, int > > &cell_to_face_local,
                           int MISKDistance);