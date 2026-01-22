#pragma once

#include<map>
#include<string>
#include<variant>
#include<vector>

using MapVariant = std::map< std::string, std::variant<int, double, bool, std::string, std::vector<double> > >;

void dimless_parameters( MapVariant &parameters );