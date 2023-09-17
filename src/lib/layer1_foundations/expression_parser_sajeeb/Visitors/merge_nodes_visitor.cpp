
/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.22.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/
    
#include "merge_nodes_visitor.h"
#include "dispatcher.h"
#include <iostream>

using std::cout;
using std::endl;

#define BACKUP_CURRENT_AND_PREVIOUS_NODE_POINTERS \
			pointer backup_current_output_node_ptr = current_output_node_ptr; \
			pointer backup_previous_output_node_ptr = previous_output_node_ptr;
			
#define RESTORE_CURRENT_AND_PREVIOUS_NODE_POINTERS \
			current_output_node_ptr = backup_current_output_node_ptr; \
			previous_output_node_ptr = backup_previous_output_node_ptr;

#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;

using std::cout;
using std::endl;



void merge_nodes_visitor::visit(plus_node* node) {
    for (auto it = node->children.begin(); it != node->children.end(); ) {
	    if ((*it)->type == node_type::PLUS_NODE) {
			// make all grandchild nodes as child nodes
			make_child_nodes(it->get(), node);
			// 
			it = node->children.erase(it);
	    } else ++it;
	}
    for (shared_ptr<irtree_node>& child : node->children) {
        dispatcher::visit(child, *this);
    }
}

void merge_nodes_visitor::visit(multiply_node* node) {
	for (auto it = node->children.begin(); it != node->children.end(); ) {
	    if ((*it)->type == node_type::MULTIPLY_NODE) {
			// make all grandchild nodes as child nodes
			make_child_nodes(it->get(), node);
			it = node->children.erase(it);
	    } else ++it;
	}
    for (shared_ptr<irtree_node>& child : node->children) {
        dispatcher::visit(child, *this);
    }
}
