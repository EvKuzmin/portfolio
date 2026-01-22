#pragma once

#include "inmost.h"

using namespace INMOST;

void make_face_adj_cells( Mesh *m, 
                          MarkerType u_mrk, 
                          MarkerType v_mrk, 
                          MarkerType w_mrk, 
                          MarkerType inner_face_mrk );

void make_face_lapl_adj( Mesh *m, 
                         MarkerType u_mrk, 
                         MarkerType v_mrk, 
                         MarkerType w_mrk, 
                         MarkerType inner_face_mrk);

void make_cell_lapl_adj(Mesh *m );