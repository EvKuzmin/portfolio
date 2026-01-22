#pragma once

#include"inmost.h"
#include<string>
#include<array>

void mark_inlet_outlet(INMOST::Mesh *m, std::array<double, 6> bounding_box, std::map< std::string, INMOST::MarkerType > &markers);
void mark_velocities(INMOST::Mesh *m, std::map< std::string, INMOST::MarkerType > &markers);
void mark_submicro(INMOST::Mesh *m, std::map< std::string, INMOST::MarkerType > &markers);