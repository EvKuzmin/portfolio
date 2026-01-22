#pragma once

#include "inmost.h"
#include <utility>
#include <any>
#include <tuple>
#include "submatrix_manager.hpp"

struct VariableOnMesh {
    std::string name;
    std::string type;
    std::map< std::string, std::any > properties;
};

int make_face_ids(INMOST::Mesh *m);
int make_cell_ids(INMOST::Mesh *m);
std::tuple< std::vector<int>, int, int, int, SubmatrixManager > make_variable_ids( INMOST::Mesh *m, int cell_unknowns_local, int face_unknowns_local, std::vector<VariableOnMesh> &variables );
std::tuple< std::vector<int>, int, int, int, SubmatrixManager > make_submatrix_map( int cell_unknowns_local, int face_unknowns_local, std::vector<VariableOnMesh> &variables );