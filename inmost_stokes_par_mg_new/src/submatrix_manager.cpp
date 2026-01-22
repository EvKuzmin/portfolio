#include"submatrix_manager.hpp"
#include<numeric>
#include<iostream>
#include"mpi.h"
#include<fstream>
#include<algorithm>

SubmatrixManager::Node::Node() : parent(nullptr),
name(""),
size(0),
size_ex_sum(0),
map_to_parent_ib(nullptr)
{

}

SubmatrixManager::Node::Node( std::string name, int size, int size_ex_sum ) :
parent(nullptr),
name(name),
size(size),
size_ex_sum(size_ex_sum),
map_to_parent_ib(nullptr)
{
    
}

SubmatrixManager::Node::Node( std::string name, int size, int size_ex_sum, index_block ib ) :
parent(nullptr),
name(name),
size(size),
size_ex_sum(size_ex_sum),
map_to_parent_ib(nullptr),
ib(ib)
{
    
}

SubmatrixManager::Node::~Node() {
    if( map_to_parent_ib != nullptr ) {
        delete map_to_parent_ib;
    }
}

void SubmatrixManager::Node::write() const {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::ofstream file("Node_" + name + "_rank_" + std::to_string(rank) + ".txt");
    file << "size = " << size << "\n";
    file << "size_ex_sum = " << size_ex_sum << "\n";
    file << "map_to_parent = \n";
    for(auto v : map_to_parent) {
        file << v << "\n";
    }
    file << "map_from_parent = \n";
    for(auto v : map_from_parent) {
        file << v << "\n";
    }
    file << "map_to_parent_ib = \n";
    if( map_to_parent_ib != nullptr ) {
        for( auto const& [key, val] : *map_to_parent_ib ) {
            file << "   " << key << " = \n";
            for(auto const v : val) {
                file << "   " << v << "\n";
            }
        }
    }
    file << "ib = \n";
    for( auto const& [key, val] : ib ) {
        file << "   " << key << " = \n";
        for(auto const v : val) {
            file << "   " << v << "\n";
        }
    }
    file.close();
}

SubmatrixManager::SubmatrixManager( int size, int size_ex_sum, index_block initial_index_map ) {
    root = new Node("initial", size, size_ex_sum, initial_index_map);
}

SubmatrixManager::SubmatrixManager( const SubmatrixManager &s ) {
    root = new Node;
    recreate_node( root, s.root );
}

SubmatrixManager::~SubmatrixManager() {
    auto delete_visitor = [](Node *node){
        delete node;
    };
    visit_nodes( root, delete_visitor );
}

void SubmatrixManager::remove_index( Node *node, std::string name, std::vector< int > index_to_delete ) {
    int new_size = node->size - index_to_delete.size();
    int new_size_ex_sum = 0;

    MPI_Exscan(&new_size,
                   &new_size_ex_sum,
                   1,
               MPI_INT,
               MPI_SUM,
               MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    Node *new_node = new Node(name, new_size, new_size_ex_sum, index_block());
    std::vector< int > parent_index(node->size);
    std::iota( parent_index.begin(), parent_index.end(), 0 );
    std::set_difference( 
        parent_index.begin(), 
        parent_index.end(), 
        index_to_delete.begin(), 
        index_to_delete.end(),
        std::back_inserter(new_node->map_to_parent) );

    std::vector<int> map_to_child(node->size);
    std::fill( map_to_child.begin(), map_to_child.end(), -1 );
    for( int i = 0 ; i < new_node->map_to_parent.size() ; i++ ) {
        map_to_child[ new_node->map_to_parent[i] ] = i;
    }

    new_node->map_from_parent = map_to_child;

    new_node->map_to_parent_ib = new index_block();
    for( auto &[k, v] : node->ib ) {
        std::vector< int > new_map_to_parent_ids;
        std::set_difference( 
            v.begin(), 
            v.end(), 
            index_to_delete.begin(), 
            index_to_delete.end(),
            std::back_inserter(new_map_to_parent_ids) );
        new_node->map_to_parent_ib->operator[](k) = new_map_to_parent_ids;
        std::vector< int > new_ib(new_map_to_parent_ids.size(),0);
        for(int i = 0 ; i < new_ib.size() ; i++) {
            new_ib[i] = map_to_child[ new_map_to_parent_ids[i] ];
        }
        new_node->ib[k] = new_ib;
    }
    new_node->parent = node;
    node->child.push_back(new_node);
}

void SubmatrixManager::remove_index( std::string node_name, std::string name, std::vector< int > index ) {
    std::vector<Node*> found_nodes = find_nodes( node_name );
    if( found_nodes.size() == 1 ) {
        Node *node = found_nodes[0];
        remove_index( node, name, index );
    } else if (found_nodes.size() > 1) {
        std::cerr << "More than one nodes \"" << node_name << "\"\n";
        std::exit(2);
    } else if ( found_nodes.size() == 0 ) {
        std::cerr << "There is no \"" << node_name << "\" node in SubmatrixManager\n";
        std::exit(3);
    }
}

void SubmatrixManager::recombine_blocks( std::string node_name, 
                        std::string name, 
                        std::vector< std::string > combine_blocks ) {
    Node* parent_node = find_nodes( node_name )[0];
    int new_size = 0;
    for( auto v2 : combine_blocks ) {
        new_size += parent_node->ib.at(v2).size();
    }
    int new_size_ex_sum = 0;

    MPI_Exscan(&new_size,
                   &new_size_ex_sum,
                   1,
               MPI_INT,
               MPI_SUM,
               MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    Node* new_node = new Node( name, new_size, new_size_ex_sum );
    std::vector<int> map_to_parent_new;
    for( auto v2: combine_blocks ) {
        for( auto id : parent_node->ib.at(v2) ) {
            map_to_parent_new.push_back(id);
        }
    }
    std::sort(map_to_parent_new.begin(), map_to_parent_new.end());
    new_node->map_to_parent = map_to_parent_new;
    std::vector<int> map_to_child_new(parent_node->size);
    std::fill(map_to_child_new.begin(), map_to_child_new.end(), -1);
    for( int i = 0 ; i < map_to_parent_new.size() ; i++ ) {
        map_to_child_new[ map_to_parent_new[i] ] = i;
    }
    new_node->map_from_parent = map_to_child_new;
    index_block new_ib;
    new_node->map_to_parent_ib = new index_block;
    for( auto v2 : combine_blocks ) {
        std::vector< int > new_map_to_parent_ib;
        std::copy( parent_node->ib.at(v2).begin(), parent_node->ib.at(v2).end(), std::back_inserter( new_map_to_parent_ib ) );
        (*(new_node->map_to_parent_ib))[v2] = new_map_to_parent_ib;
        new_node->ib[v2].resize(new_map_to_parent_ib.size());
        for( int i = 0 ; i < new_map_to_parent_ib.size() ; i++ ) {
            new_node->ib.at(v2)[i] = map_to_child_new[new_map_to_parent_ib[i]];
        }
        
    }
    new_node->parent = parent_node;
    parent_node->child.push_back(new_node);
}

// void SubmatrixManager::split_block(std::string node_name, 
//                         std::string name, 
//                         std::vector< std::pair< std::string, std::vector< int > > > block_ids) {
//     Node* parent_node = find_nodes( node_name )[0];
//     int new_size = 0;
//     for( auto &[k, v] : block_ids ) {
//         for( auto v2 : v ) {
//             new_size += v.size();
//         }
//     }
//     Node* new_node = new Node( name, new_size );
//     std::vector<int> map_to_parent_new;
//     for( auto &[k, v] : block_ids ) {
//         for( auto id : v ) {
//             map_to_parent_new.push_back(id);
//         }
//     }
//     new_node->map_to_parent = map_to_parent_new;
//     std::vector<int> map_to_child_new(parent_node->size);
//     std::fill(map_to_child_new.begin(), map_to_child_new.end(), -1);
//     for( int i = 0 ; i < map_to_parent_new.size() ; i++ ) {
//         map_to_child_new[ map_to_parent_new[i] ] = i;
//     }
//     new_node->map_from_parent = map_to_child_new;
//     index_block new_ib;
//     new_node->map_to_parent_ib = new index_block;
//     for( auto &[k, v] : block_ids ) {
//         // std::vector< int > new_map_to_parent_ib;
//         (*(new_node->map_to_parent_ib))[k] = v;
//         new_node->ib[k].resize(v.size());
//         for( int i = 0 ; i < v.size() ; i++ ) {
//             new_node->ib.at(k)[i] = map_to_child_new[v[i]];
//         }
//     }
//     new_node->parent = parent_node;
//     parent_node->child.push_back(new_node);
// }

void SubmatrixManager::visit_nodes( Node *node, std::function< void(Node*) > visitor ) {
    for( int i = 0 ; i < node->child.size() ; i++ ) {
        visit_nodes( node->child[i], visitor );
    }
    visitor(node);
}

void SubmatrixManager::recreate_node( Node *new_node, Node *node ) {
    new_node->name = node->name;
    new_node->size = node->size;
    new_node->size_ex_sum = node->size_ex_sum;
    new_node->ib = node->ib;
    new_node->map_to_parent = node->map_to_parent;
    new_node->map_from_parent = node->map_from_parent;
    if(node->map_to_parent_ib) {
        new_node->map_to_parent_ib = new index_block(*(node->map_to_parent_ib));
    }
    for( auto c : node->child ) {
        Node *new_child = new Node;
        recreate_node( new_child, c );
        new_child->parent = new_node;
    }
}

std::vector<SubmatrixManager::Node*> SubmatrixManager::find_nodes( std::string name ) {
    std::vector<Node*> found_nodes;
    auto find_visitor = [&found_nodes, name](Node * node){
        if(node->name == name) {
            found_nodes.push_back(node);
        }
    };
    visit_nodes(root, find_visitor);
    return found_nodes;
}

void SubmatrixManager::write() const {
    std::vector<Node *> queue;
    queue.push_back(root);
    while( queue.size() > 0 ) {
        std::vector<Node *> new_queue;
        for( Node *n : queue ) {
            n->write();
            for( auto c : n->child ) {
                new_queue.push_back(c);
            }
        }
        queue = new_queue;
    }
}