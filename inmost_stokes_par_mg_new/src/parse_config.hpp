#pragma once

#include<vector>
#include<string>
#include<variant>
#include<map>

using MapVariant = std::map< std::string, std::variant<int, double, bool, std::string, std::vector<double> > >;

std::vector<std::string> read_file( std::string name );
void parse_config( std::vector<std::string> lines, MapVariant &parameters, MapVariant &config );