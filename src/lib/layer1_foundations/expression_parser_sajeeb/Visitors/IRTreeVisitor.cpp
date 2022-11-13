/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/15/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "IRTreeVisitor.h"
#include "../IRTree/node.h"


using std::shared_ptr;
using std::cout;
using std::endl;

void IRTreeVoidReturnTypeVisitor::visit(plus_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this);
}

void IRTreeVoidReturnTypeVisitor::visit(minus_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this);
}

void IRTreeVoidReturnTypeVisitor::visit(multiply_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this);
}

void IRTreeVoidReturnTypeVisitor::visit(exponent_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this);
}

void IRTreeVoidReturnTypeVisitor::visit(unary_negate_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this);
}

void IRTreeVoidReturnTypeVisitor::visit(sentinel_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this);
}

void IRTreeVoidReturnTypeVisitor::visit(variable_node* num_node) {}
void IRTreeVoidReturnTypeVisitor::visit(parameter_node* node) {}
void IRTreeVoidReturnTypeVisitor::visit(number_node* op_node) {}

