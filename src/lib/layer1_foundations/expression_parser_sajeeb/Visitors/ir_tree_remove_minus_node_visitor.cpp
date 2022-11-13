
/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.22.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/
    
#include "ir_tree_remove_minus_node_visitor.h"


void ir_tree_remove_minus_node_visitor::merge_visited_minus_node(non_terminal_node* root,
    list<shared_ptr<irtree_node>>::iterator& it) 
{
    if (visited_minus_node == it->get()) {
        for (shared_ptr<irtree_node>& child : visited_minus_node->children)
            root->children.insert(it, child);
        root->children.erase(it--);
        visited_minus_node = NULL;
    }
}

void ir_tree_remove_minus_node_visitor::visit(plus_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) {
        it->get()->accept(this);
        merge_visited_minus_node(op_node, it);
    }
}

void ir_tree_remove_minus_node_visitor::visit(minus_node* op_node) {
    for (auto it=++op_node->children.begin(); it != op_node->children.end(); ++it) {
        shared_ptr<non_terminal_node> uminus_child = make_shared<unary_negate_node>();
        uminus_child->children.emplace_back(std::move(*it));
        *it = std::move(uminus_child);
    }
    visited_minus_node = op_node;
}

void ir_tree_remove_minus_node_visitor::visit(multiply_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) {
        it->get()->accept(this);
        merge_visited_minus_node(op_node, it);
    }
}

void ir_tree_remove_minus_node_visitor::visit(exponent_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) {
        it->get()->accept(this);
        merge_visited_minus_node(op_node, it);
    }
}

void ir_tree_remove_minus_node_visitor::visit(unary_negate_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) {
        it->get()->accept(this);
        if (visited_minus_node == it->get()) {
            shared_ptr<non_terminal_node> intermediate_plus_node = make_shared<plus_node>();
            intermediate_plus_node->children = 
                        std::move(static_cast<non_terminal_node*>(it->get())->children);
            *it = std::move(intermediate_plus_node);
            visited_minus_node = NULL;
        }
    }
}
