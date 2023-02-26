
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
    bool found_plus_chain = true;
    auto it_start = node->children.begin();
    while (found_plus_chain) {
        found_plus_chain = false;
        bool set_start_iterator = true;
        for (auto it = it_start; it != node->children.end(); ++it) {
            if ((*it)->type == node_type::PLUS_NODE) {
                found_plus_chain = true;
                plus_node* child_raw_ptr = static_cast<plus_node*>(it->get());
                for (shared_ptr<irtree_node>& grandchild : child_raw_ptr->children) {
                    auto new_iter = node->children.insert(it, grandchild);
                    if (grandchild->type == node_type::PLUS_NODE && set_start_iterator)
                        it_start = new_iter, set_start_iterator = false;
                }
                node->children.erase(it--);
            }
            if (set_start_iterator) it_start = it;
        }
    }
    for (shared_ptr<irtree_node>& child : node->children) {
        dispatcher::visit(child, *this);
    }
}

void merge_nodes_visitor::visit(multiply_node* node) {
    bool found_multiplication_chain = true;
    auto it_start = node->children.begin();
    while (found_multiplication_chain) {
        found_multiplication_chain = false;
        bool set_start_iterator = true;
        for (auto it = it_start; it != node->children.end(); ++it) {
            if ((*it)->type == node_type::MULTIPLY_NODE) {
                found_multiplication_chain = true;
                multiply_node* child_raw_ptr = static_cast<multiply_node*>(it->get());
                for (shared_ptr<irtree_node>& grandchild : child_raw_ptr->children) {
                    auto new_iter = node->children.insert(it, grandchild);
                    if (grandchild->type == node_type::MULTIPLY_NODE && set_start_iterator)
                        it_start = new_iter, set_start_iterator = false;
                }
                node->children.erase(it--);
            }
            if (set_start_iterator) it_start = it;
        }
    }
    for (shared_ptr<irtree_node>& child : node->children) {
        dispatcher::visit(child, *this);
    }
}
