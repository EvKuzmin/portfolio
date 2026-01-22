#include"solver_tree.hpp"

SolverContext::~SolverContext() {
    
}

void SolverTree::visit_nodes( SolverTree *node, std::function< void(SolverTree*) > visitor ) {
    for( int i = 0 ; i < node->child.size() ; i++ ) {
        visit_nodes( node->child[i], visitor );
    }
    visitor(node);
}