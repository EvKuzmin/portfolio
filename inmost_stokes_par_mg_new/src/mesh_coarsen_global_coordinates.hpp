#pragma once

#include<tuple>
#include<vector>
#include"petsc.h"
#include"submatrix_manager.hpp"
#include "inmost.h"
#include "make_ids.hpp"

std::tuple< std::vector<int>, 
            std::vector<int>, 
            std::vector< std::tuple< double, double, double > >, 
            std::vector< std::tuple< double, double, double > >, 
            std::vector< std::tuple< int, int, int, int, int, int > >,
            int,
            int,
            std::vector<int> >
            mesh_coarsen_global_coordinates(Mat A_full, 
                           Mat A_lapl, 
                           int face_unknowns_local, 
                           int cell_unknowns_local, 
                           std::vector< std::tuple< double, double, double > > &cell_center_coordinates, 
                           std::vector< std::tuple< int, int, int, int, int, int > > &cell_to_face_local,
                           std::vector<VariableOnMesh> &variables, 
                           std::vector<int> unknowns_local,
                           int MISKDistance);