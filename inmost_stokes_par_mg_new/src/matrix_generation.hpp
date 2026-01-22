#pragma once

#include "inmost.h"
#include "petsc.h"
#include <map>
#include <variant>

using namespace INMOST;

using MapVariant = std::map< std::string, std::variant<int, double, bool, std::string, std::vector<double> > >;

void Stokes_matrix( Mesh *m, 
                   Mat A,
                   double hx,
                   double hy,
                   double hz,
                   MarkerType u_mrk, 
                   MarkerType v_mrk, 
                   MarkerType w_mrk, 
                   MarkerType inner_face_mrk, 
                   MapVariant parameters );

// void matrix_Laplace_row(Mesh *m, 
//                    Mat A,
//                    double hx,
//                    double hy,
//                    double hz,
//                    MapVariant parameters);