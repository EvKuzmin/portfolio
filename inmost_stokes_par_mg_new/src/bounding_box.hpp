#pragma once

#include "inmost.h"
#include <array>
#include <variant>
#include <vector>


using MapVariant = std::map< std::string, std::variant<int, double, bool, std::string, std::vector<double> > >;

std::array<double, 6> prepare_bounding_box(INMOST::Mesh *m);