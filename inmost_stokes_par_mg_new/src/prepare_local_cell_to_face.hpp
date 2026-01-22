#pragma once

#include<vector>
#include<tuple>
#include"inmost.h"

std::vector< std::tuple< int, int, int, int, int, int > > prepare_local_cell_to_face( INMOST::Mesh *m );