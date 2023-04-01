/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/2/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "IRTreeChildLinkArgumentVisitor.h"
#include "dispatcher.h"
#include "../IRTree/node.h"


void IRTreeChildLinkArgumentVisitor::visit(plus_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, it);
}

void IRTreeChildLinkArgumentVisitor::visit(minus_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, it);
}

void IRTreeChildLinkArgumentVisitor::visit(multiply_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, it);
}

void IRTreeChildLinkArgumentVisitor::visit(exponent_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, it);
}

void IRTreeChildLinkArgumentVisitor::visit(unary_negate_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, it);
}

void IRTreeChildLinkArgumentVisitor::visit(variable_node* num_node) {}

void IRTreeChildLinkArgumentVisitor::visit(parameter_node* node) {}

void IRTreeChildLinkArgumentVisitor::visit(number_node* op_node) {}

void IRTreeChildLinkArgumentVisitor::visit(sentinel_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, it);
}




void IRTreeChildLinkArgumentVisitor::visit(plus_node* op_node,
                                           list<shared_ptr<irtree_node> >::iterator& link) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, it);
}

void IRTreeChildLinkArgumentVisitor::visit(minus_node* op_node,
                                           list<shared_ptr<irtree_node> >::iterator& link) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, it);
}

void IRTreeChildLinkArgumentVisitor::visit(multiply_node* op_node,
                                           list<shared_ptr<irtree_node> >::iterator& link) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, it);
}

void IRTreeChildLinkArgumentVisitor::visit(exponent_node* op_node,
                                           list<shared_ptr<irtree_node> >::iterator& link) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, it);
}

void IRTreeChildLinkArgumentVisitor::visit(unary_negate_node* op_node,
                                           list<shared_ptr<irtree_node> >::iterator& link) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, it);
}

void IRTreeChildLinkArgumentVisitor::visit(variable_node* num_node,
                                           list<shared_ptr<irtree_node> >::iterator& link) {}

void IRTreeChildLinkArgumentVisitor::visit(parameter_node* node,
                                           list<shared_ptr<irtree_node> >::iterator& link) {}

void IRTreeChildLinkArgumentVisitor::visit(number_node* op_node,
                                           list<shared_ptr<irtree_node> >::iterator& link) {}

void IRTreeChildLinkArgumentVisitor::visit(sentinel_node* op_node,
                                           list<shared_ptr<irtree_node> >::iterator& link) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, it);
}
