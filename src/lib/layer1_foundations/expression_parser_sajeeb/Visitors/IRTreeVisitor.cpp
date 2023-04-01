/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/15/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "IRTreeVisitor.h"
#include "dispatcher.h"
#include "../IRTree/node.h"


void IRTreeVoidReturnTypeVisitorInterface::visit(plus_node* node) {
    for (shared_ptr<irtree_node>& child : node->children)
        dispatcher::visit(child, *this);
}

void IRTreeVoidReturnTypeVisitorInterface::visit(minus_node* node) {
    for (shared_ptr<irtree_node>& child : node->children)
        dispatcher::visit(child, *this);
}

void IRTreeVoidReturnTypeVisitorInterface::visit(multiply_node* node) {
    for (shared_ptr<irtree_node>& child : node->children)
        dispatcher::visit(child, *this);
}

void IRTreeVoidReturnTypeVisitorInterface::visit(exponent_node* node) {
    for (shared_ptr<irtree_node>& child : node->children)
        dispatcher::visit(child, *this);
}

void IRTreeVoidReturnTypeVisitorInterface::visit(unary_negate_node* node) {
    for (shared_ptr<irtree_node>& child : node->children)
        dispatcher::visit(child, *this);
}

void IRTreeVoidReturnTypeVisitorInterface::visit(sentinel_node* node) {
    for (shared_ptr<irtree_node>& child : node->children)
        dispatcher::visit(child, *this);
}

void IRTreeVoidReturnTypeVisitorInterface::visit(variable_node*) {}
void IRTreeVoidReturnTypeVisitorInterface::visit(parameter_node*) {}
void IRTreeVoidReturnTypeVisitorInterface::visit(number_node*) {}