#include "decompose_matrix.hpp"

#include<iostream>

void create_IS( SubmatrixManager &submatrix_map, std::map< std::string, IS > &index_sequences, std::string name ) {
    auto node = submatrix_map.find_nodes(name)[0];
    int parent_size_ex_sum = 0;
    if( node->parent ) {
        parent_size_ex_sum = node->parent->size_ex_sum;
        std::vector<int> map_to_parent_global = node->map_to_parent;
        for( auto &v:map_to_parent_global ) {
            v+=parent_size_ex_sum;
        }
        ISCreateGeneral(
        PETSC_COMM_WORLD, 
        map_to_parent_global.size(), 
        map_to_parent_global.data(), 
        PETSC_COPY_VALUES, 
        &(index_sequences[name]));
    }
}

void create_IS_tree( IndexTreeStruct *node, SubmatrixManager &submatrix_map, std::map< std::string, IS > &index_sequences ) {
    create_IS(submatrix_map, index_sequences, node->name);
    for(auto v : node->child) {
        create_IS_tree( v, submatrix_map, index_sequences );
    }
}

void recombine_index( IndexTreeStruct *node, SubmatrixManager &submatrix_map ) {
    std::string parent_name;
    if( node->parent == nullptr ) {
        parent_name = "initial";
    } else {
        parent_name = node->parent->name;
    }
    if( node->type == "recombine" ) {
        submatrix_map.recombine_blocks( 
            parent_name, 
            node->name, 
            node->variable_list );
    } else if( node->type == "remove" ) {
        std::vector<int> *local_dirichlet_rows_p = std::any_cast< std::vector<int>* >(node->additional_data.at( "local_dirichlet_rows" ));
        submatrix_map.remove_index( parent_name, node->name, *local_dirichlet_rows_p );
    } else {
        std::cerr << "unknown command" << std::endl;
        std::exit(1);
    }
    for( auto v : node->child ) {
        recombine_index( v, submatrix_map );
    }
}

void create_submatrices( SubmatrixTreeStruct *node, Mat A, std::map< std::string, Mat > &submatrices, std::map< std::string, IS > &index_sequences ) {
    Mat V;
    if( node->parent == nullptr ) {
        V = A;
    } else {
        V = submatrices.at(node->parent->name);
    }
    MatCreateSubMatrix(
        V,
        index_sequences.at(node->IS_names.first),
        index_sequences.at(node->IS_names.second),
        MAT_INITIAL_MATRIX,
        &(submatrices[node->name]));
    for( auto v : node->child ) {
        create_submatrices( v, A, submatrices, index_sequences );
    }
}