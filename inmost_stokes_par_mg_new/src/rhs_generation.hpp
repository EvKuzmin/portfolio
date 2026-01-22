#pragma once

#include "inmost.h"
#include <petsc.h>
#include <map>
#include <variant>

using namespace INMOST;

using MapVariant = std::map< std::string, std::variant<int, double, bool, std::string, std::vector<double> > >;

void Stokes_rhs(Mesh *m,
           Vec rhs,
           double hx,
           double hy,
           double hz,
           MarkerType u_mrk, 
           MarkerType v_mrk, 
           MarkerType w_mrk, 
           MarkerType inner_face_mrk,
           MapVariant parameters);