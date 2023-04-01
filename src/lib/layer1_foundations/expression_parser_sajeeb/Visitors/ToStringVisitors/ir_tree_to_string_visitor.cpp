/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.26.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "ir_tree_to_string_visitor.h"

using type = irtree_node::node_type;

void ir_tree_to_string_visitor::visit(plus_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ) {
        const shared_ptr<irtree_node>& child = *it;
        dispatcher::visit(child, *this);
        if (++it != op_node->children.end())
            rep.push_back(" + ");
    }
}

void ir_tree_to_string_visitor::visit(multiply_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ) {
        const shared_ptr<irtree_node>& child = *it;
        if (dynamic_cast<non_terminal_node*>(child.get()) && child->type != irtree_node::node_type::EXPONENT_NODE) {
            rep.push_back("(");
            dispatcher::visit(child, *this);
            rep.push_back(")");
        } else dispatcher::visit(child, *this);
        if (++it != op_node->children.end())
            rep.push_back(" * ");
    }
}

void ir_tree_to_string_visitor::visit(exponent_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ) {
        const shared_ptr<irtree_node>& child = *it;
        if (dynamic_cast<non_terminal_node*>(child.get())) {
            rep.push_back("(");
            dispatcher::visit(child, *this);
            rep.push_back(")");
        } else dispatcher::visit(child, *this);
        if (++it != op_node->children.end())
            rep.push_back(" ^ ");
    }
}

void ir_tree_to_string_visitor::visit(unary_negate_node* op_node) {
    if (dynamic_cast<terminal_node*>(op_node->children.front().get())) {
        rep.push_back("(-");
        dispatcher::visit(op_node->children.front(), *this);
        rep.push_back(")");
    } else {
        rep.push_back("-(");
        dispatcher::visit(op_node->children.front(), *this);
        rep.push_back(")");
    }
}

void ir_tree_to_string_visitor::visit(variable_node* num_node) {
    rep.push_back(num_node->name);
}

void ir_tree_to_string_visitor::visit(parameter_node* op_node) {
    rep.push_back(op_node->name);
}

void ir_tree_to_string_visitor::visit(number_node* op_node) {
    rep.push_back(std::to_string(op_node->value));
}

void ir_tree_to_string_visitor::visit(sentinel_node* op_node) {
    rep.clear();
    loc = left = rep.begin();
    for (const shared_ptr<irtree_node>& child : op_node->children) {
        dispatcher::visit(child, *this);
    }
}
