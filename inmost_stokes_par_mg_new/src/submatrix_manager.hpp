#pragma once
#include<map>
#include<vector>
#include<utility>
#include<string>
#include<functional>

// Local indices for submatrices
struct SubmatrixManager {
    typedef std::map< std::string, std::vector< int > > index_block;
    struct Node {
        Node *parent;
        std::string name;
        std::vector< Node * > child;
        int size;
        int size_ex_sum;
        std::vector< int > map_to_parent;
        std::vector< int > map_from_parent;
        index_block *map_to_parent_ib;
        index_block ib;
        Node();
        Node( std::string name, int size, int size_ex_sum);
        Node( std::string name, int size, int size_ex_sum, index_block ib);
        ~Node();
        void write() const;
    };
    Node *root;
    SubmatrixManager( int size, int size_ex_sum, index_block initial_index_map );
    SubmatrixManager( const SubmatrixManager &s );
    ~SubmatrixManager();
    void remove_index( Node *node, std::string name, std::vector< int > index_to_delete );
    void remove_index( std::string node_name, std::string name, std::vector< int > index );
    void visit_nodes( Node *node, std::function< void(Node*) > visitor );
    void recreate_node( Node *new_parent, Node *child );
    void recombine_blocks( std::string node_name, 
                        std::string name, 
                        std::vector< std::string > combine_blocks );
    // void split_block(std::string node_name, 
    //                     std::string name, 
    //                     std::vector< std::pair< std::string, std::vector< int > > > block_ids);
    std::vector<Node*> find_nodes( std::string name );
    void write() const;
};