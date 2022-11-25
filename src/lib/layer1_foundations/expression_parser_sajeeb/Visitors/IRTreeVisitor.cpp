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

void IRTreeVoidReturnTypeVisitorInterface::visit(plus_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this);
}

void IRTreeVoidReturnTypeVisitorInterface::visit(minus_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this);
}

void IRTreeVoidReturnTypeVisitorInterface::visit(multiply_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this);
}

void IRTreeVoidReturnTypeVisitorInterface::visit(exponent_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this);
}

void IRTreeVoidReturnTypeVisitorInterface::visit(unary_negate_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this);
}

void IRTreeVoidReturnTypeVisitorInterface::visit(sentinel_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this);
}

void IRTreeVoidReturnTypeVisitorInterface::visit(variable_node* num_node) {}
void IRTreeVoidReturnTypeVisitorInterface::visit(parameter_node* node) {}
void IRTreeVoidReturnTypeVisitorInterface::visit(number_node* op_node) {}

