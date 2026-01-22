#pragma once

#include "inmost.h"
#include <petsc.h>
#include <map>
#include <variant>

using namespace INMOST;

using MapVariant = std::map< std::string, std::variant<int, double, bool, std::string, std::vector<double> > >;

void boundary_conditions_U(Mesh *m,
           Mat A,
           Vec rhs,
           double hx,
           double hy,
           double hz,
           MarkerType inlet,
           MarkerType outlet,
           MarkerType wall,
           MarkerType u_mrk, 
           MarkerType v_mrk, 
           MarkerType w_mrk, 
           MarkerType inner_face_mrk,
           MapVariant parameters);

void boundary_conditions_P(Mesh *m,
           Mat A,
           Vec rhs,
           double hx,
           double hy,
           double hz,
           MarkerType inlet,
           MarkerType outlet,
           MarkerType wall,
           MarkerType u_mrk, 
           MarkerType v_mrk, 
           MarkerType w_mrk, 
           MarkerType inner_face_mrk,
           MapVariant parameters);

void boundary_conditions(Mesh *m,
           Mat A,
           Vec rhs,
           double hx,
           double hy,
           double hz,
           MarkerType inlet,
           MarkerType outlet,
           MarkerType wall,
           MarkerType u_mrk, 
           MarkerType v_mrk, 
           MarkerType w_mrk, 
           MarkerType inner_face_mrk,
           MapVariant parameters);