#pragma once

#include "petsc.h"
#include <map>
#include <string>
#include <any>
#include "submatrix_manager.hpp"

struct IndexTreeStruct {
    IndexTreeStruct *parent;
    std::string name;
    std::string type;
    std::map< std::string, std::any > additional_data;
    std::vector<std::string> variable_list;
    std::vector<IndexTreeStruct*> child;
    IndexTreeStruct( IndexTreeStruct *parent, std::string name, std::vector<std::string> variable_list ) : parent(parent), name(name), variable_list(variable_list) {
        type = "recombine";
    }
    IndexTreeStruct( IndexTreeStruct *parent, std::string name, std::vector<int> *local_dirichlet_rows_p ) : parent(parent), name(name) {
        additional_data["local_dirichlet_rows"] = local_dirichlet_rows_p;
        type = "remove";
    }
};

struct SubmatrixTreeStruct {
    SubmatrixTreeStruct *parent;
    std::string name;
    std::pair< std::string, std::string > IS_names;
    std::vector<SubmatrixTreeStruct*> child;
    SubmatrixTreeStruct( SubmatrixTreeStruct *parent, std::string name, std::pair< std::string, std::string > IS_names ) : parent(parent), name(name), IS_names(IS_names) {
        
    }
};

void create_IS(SubmatrixManager &submatrix_map, std::map< std::string, IS > &index_sequences, std::string name);
void create_IS_tree( IndexTreeStruct *node, SubmatrixManager &submatrix_map, std::map< std::string, IS > &index_sequences );
void recombine_index( IndexTreeStruct *node, SubmatrixManager &submatrix_map );
void create_submatrices( SubmatrixTreeStruct *node, Mat A, std::map< std::string, Mat > &submatrices, std::map< std::string, IS > &index_sequences );