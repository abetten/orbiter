/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/7/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/


#include "../IRTree/node_forward_declaration.h"
#include <memory.h>
#include "dispatcher.h"
#include <iostream>

#ifndef IRTREETEMPLATERETURNVISITOR_H
#define IRTREETEMPLATERETURNVISITOR_H

#ifndef LOG
#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;
#endif

template<class return_t, class... Args>
class IRTreeTemplateReturnTypeVariadicArgumentVisitorInterface {
public:
    return_t visit(plus_node& node, Args... args);
    return_t visit(minus_node* node, Args... args);
    return_t visit(multiply_node* node, Args... args);
    return_t visit(exponent_node* node, Args... args);
    return_t visit(unary_negate_node* node, Args... args);
    return_t visit(variable_node* node, Args... args);
    return_t visit(parameter_node* node, Args... args);
    return_t visit(number_node* node, Args... args);
    return_t visit(sentinel_node* node, Args... args);
};


template<class return_t, class... Args>
class IRTreeTemplateReturnTypeVariadicArgumentConstantVisitorInterface {
public:
    return_t visit(const plus_node* node, Args... args);
    return_t visit(const minus_node* node, Args... args);
    return_t visit(const multiply_node* node, Args... args);
    return_t visit(const exponent_node* node, Args... args);
    return_t visit(const unary_negate_node* node, Args... args);
    return_t visit(const variable_node* node, Args... args);
    return_t visit(const parameter_node* node, Args... args);
    return_t visit(const number_node* node, Args... args);
    return_t visit(const sentinel_node* node, Args... args);
};


template<class... Args>
class IRTreeVoidReturnTypeVariadicArgumentVisitorInterface {
public:
    void visit(plus_node* node, Args... args);
    void visit(minus_node* node, Args... args);
    void visit(multiply_node* node, Args... args);
    void visit(exponent_node* node, Args... args);
    void visit(unary_negate_node* node, Args... args);
    void visit(variable_node* node, Args... args);
    void visit(parameter_node* node, Args... args);
    void visit(number_node* node, Args... args);
    void visit(sentinel_node* node, Args... args);
};



template<class... Args>
class IRTreeVoidReturnTypeVariadicArgumentConstantVisitorInterface {
public:
    void visit(const plus_node* node, Args... args);
    void visit(const minus_node* node, Args... args);
    void visit(const multiply_node* node, Args... args);
    void visit(const exponent_node* node, Args... args);
    void visit(const unary_negate_node* node, Args... args);
    void visit(const variable_node* node, Args... args);
    void visit(const parameter_node* node, Args... args);
    void visit(const number_node* node, Args... args);
    void visit(const sentinel_node* node, Args... args);
};


template<class return_t>
class IRTreeTemplateReturnTypeVisitorInterface {
public:
    return_t visit(plus_node* node);
    return_t visit(minus_node* node);
    return_t visit(multiply_node* node);
    return_t visit(exponent_node* node);
    return_t visit(unary_negate_node* node);
    return_t visit(variable_node* node);
    return_t visit(parameter_node* node);
    return_t visit(number_node* node);
    return_t visit(sentinel_node* node);
};


template<class return_t>
class IRTreeTemplateReturnTypeConstantVisitorInterface {
public:
    return_t visit(const plus_node* node);
    return_t visit(const minus_node* node);
    return_t visit(const multiply_node* node);
    return_t visit(const exponent_node* node);
    return_t visit(const unary_negate_node* node);
    return_t visit(const variable_node* node);
    return_t visit(const parameter_node* node);
    return_t visit(const number_node* node);
    return_t visit(const sentinel_node* node);
};


class IRTreeVoidReturnTypeVisitorInterface {
public:
    void visit(plus_node* node);
    void visit(minus_node* node);
    void visit(multiply_node* node);
    void visit(exponent_node* node);
    void visit(unary_negate_node* node);
    void visit(variable_node* node);
    void visit(parameter_node* node);
    void visit(number_node* node);
    void visit(sentinel_node* node);
};

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


class IRTreeVoidReturnTypeConstantVisitorInterface {
public:
    void visit(const plus_node* node);
    void visit(const minus_node* node);
    void visit(const multiply_node* node);
    void visit(const exponent_node* node);
    void visit(const unary_negate_node* node);
    void visit(const variable_node* node);
    void visit(const parameter_node* node);
    void visit(const number_node* node);
    void visit(const sentinel_node* node);
};

#endif // IRTREETEMPLATERETURNVISITOR_H
