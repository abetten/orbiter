//
// Created by Sajeeb Roy Chowdhury on 10/1/22.
//

#include "remove_minus_nodes_visitor.h"
#include <iostream>

using std::shared_ptr;
using std::make_shared;

#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;

void remove_minus_nodes_visitor::visit(minus_node* op_node, list<shared_ptr<irtree_node>>::iterator& link) {
    for (auto it=++op_node->children.begin(); it != op_node->children.end(); ++it) {
        shared_ptr<non_terminal_node> uminus = make_shared<unary_negate_node>();
        uminus->add_child(*it);
        *it = uminus;
    }

    shared_ptr<non_terminal_node> plus = make_shared<plus_node>();
    plus->children = std::move(op_node->children);
    *link = plus;

    for (auto it=plus->children.begin(); it != plus->children.end(); ++it) {
        shared_ptr<irtree_node>& child = *it;
        child->accept(this, it);
    }
}

