
#include<vector>
#include<string>
#include<fstream>
#include<regex>
#include<variant>
#include<iostream>

#include"parse_config.hpp"

std::vector<std::string> read_file( std::string name ) {
    std::vector<std::string> lines;
    std::ifstream file(name);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            // using printf() in all tests for consistency
            lines.push_back(line);
        }
        file.close();
    }
    return lines;
}

void parse_config( std::vector<std::string> lines, MapVariant &parameters, MapVariant &config ) {
    double umD_to_m2 = 9.869233e-19;
    std::regex option( R"blabla(([\w]+)[\s]*=[\s]*(.+))blabla" );
    std::regex array3( R"blabla([\s]*([\w\.e-]+)[\s]*,[\s]*([\w\.e-]+)[\s]*,[\s]*([\w\.e-]+)[\s]*)blabla" );
    std::map< std::string, std::string > options_map;
    for( auto s:lines ) {
        std::cmatch match;
        bool success = std::regex_match( s.c_str(), match, option );
        if( success ) {
            options_map[match[1]] = match[2];
        }
    }

    std::cmatch match;
    
    parameters["dt"] = std::stod( options_map.at("dt") );
    parameters["dt_max"] = std::stod( options_map.at("dt_max") );
    parameters["t_max"] = std::stod( options_map.at("t_max") );
    parameters["N_t"] = std::stoi( options_map.at("N_t") );
    parameters["mu"] = std::stod( options_map.at("mu") );
    parameters["rho"] = std::stod( options_map.at("rho") );
    parameters["inlet_velocity"] = std::stod( options_map.at("inlet_velocity") );
    parameters["L_d"] = std::stod( options_map.at("L_d") );
    parameters["U_d"] = std::stod( options_map.at("U_d") );
    parameters["Rho_d"] = std::stod( options_map.at("Rho_d") );
    parameters["Mu_d"] = std::stod( options_map.at("Mu_d") );

    config["mesh_path"] = options_map.at("mesh_path");

    config["ksp_i_max"] = std::stoi(options_map.at("ksp_i_max"));
    config["ksp_relative_residual"] = std::stod(options_map.at("ksp_relative_residual"));
    config["ksp_absolute_residual"] = std::stod(options_map.at("ksp_absolute_residual"));
    config["fine_smoother_i_max"] = std::stoi(options_map.at("fine_smoother_i_max"));
    config["coarse_smoother_i_max"] = std::stoi(options_map.at("coarse_smoother_i_max"));
    config["N_levels"] = std::stoi(options_map.at("N_levels"));
    config["cycle_type"] = options_map.at("cycle_type");
    config["keep_N_elements"] = std::stoi(options_map.at("keep_N_elements"));
    config["first_N_smoothers_SOR"] = std::stoi(options_map.at("first_N_smoothers_SOR"));
    config["SOR_omega"] = std::stod(options_map.at("SOR_omega"));
    config["coarsest_pc"] = options_map.at("coarsest_pc");
    config["coarsest_pc_iters"] = std::stoi( options_map.at("coarsest_pc_iters") );
    
    if( options_map.at("view_enabled") == std::string{"true"} ) {
        config["view_enabled"] = true;
    } else if( options_map.at("view_enabled") == std::string{"false"} ) {
        config["view_enabled"] = false;
    } else {
        std::cerr << "view_enabled should be \"false\" or \"true\"" << std::endl;
        std::exit(0);
    }
    if( options_map.at("coarse_view_enabled") == std::string{"true"} ) {
        config["coarse_view_enabled"] = true;
    } else if( options_map.at("coarse_view_enabled") == std::string{"false"} ) {
        config["coarse_view_enabled"] = false;
    } else {
        std::cerr << "coarse_view_enabled should be \"false\" or \"true\"" << std::endl;
        std::exit(0);
    }
    config["save_on_each"] = std::stoi(options_map.at("save_on_each"));
    if( options_map.at("agg_coarsening") == std::string{"true"} ) {
        config["agg_coarsening"] = true;
    } else if( options_map.at("agg_coarsening") == std::string{"false"} ) {
        config["agg_coarsening"] = false;
    } else {
        std::cerr << "agg_coarsening should be \"false\" or \"true\"" << std::endl;
        std::exit(0);
    }
    config["coarse_ksp"] = options_map.at("coarse_ksp");
}
