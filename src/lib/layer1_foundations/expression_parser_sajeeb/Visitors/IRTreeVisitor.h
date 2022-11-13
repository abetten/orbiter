/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/7/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/


#include "../IRTree/node_forward_declaration.h"
#include <memory.h>
#include <iostream>

#ifndef IRTREETEMPLATERETURNVISITOR_H
#define IRTREETEMPLATERETURNVISITOR_H

#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;

template<class return_t, class... Args>
class IRTreeTemplateReturnTypeVariadicArgumentVisitor {
public:
    virtual return_t visit(plus_node* op_node, Args... args) = 0;
    virtual return_t visit(minus_node* op_node, Args... args) = 0;
    virtual return_t visit(multiply_node* op_node, Args... args) = 0;
    virtual return_t visit(exponent_node* op_node, Args... args) = 0;
    virtual return_t visit(unary_negate_node* op_node, Args... args) = 0;
    virtual return_t visit(variable_node* num_node, Args... args) = 0;
    virtual return_t visit(parameter_node* node, Args... args) = 0;
    virtual return_t visit(number_node* op_node, Args... args) = 0;
    virtual return_t visit(sentinel_node* op_node, Args... args) = 0;
};


template<class return_t, class... Args>
class IRTreeTemplateReturnTypeVariadicArgumentConstantVisitor {
public:
    virtual return_t visit(const plus_node* op_node, Args... args) = 0;
    virtual return_t visit(const minus_node* op_node, Args... args) = 0;
    virtual return_t visit(const multiply_node* op_node, Args... args) = 0;
    virtual return_t visit(const exponent_node* op_node, Args... args) = 0;
    virtual return_t visit(const unary_negate_node* op_node, Args... args) = 0;
    virtual return_t visit(const variable_node* num_node, Args... args) = 0;
    virtual return_t visit(const parameter_node* node, Args... args) = 0;
    virtual return_t visit(const number_node* op_node, Args... args) = 0;
    virtual return_t visit(const sentinel_node* op_node, Args... args) = 0;
};



template<class... Args>
class IRTreeVoidReturnTypeVariadicArgumentReferenceVisitor {
public:
    virtual void visit(plus_node* op_node, Args... args) = 0;
    virtual void visit(minus_node* op_node, Args... args) = 0;
    virtual void visit(multiply_node* op_node, Args... args) = 0;
    virtual void visit(exponent_node* op_node, Args... args) = 0;
    virtual void visit(unary_negate_node* op_node, Args... args) = 0;
    virtual void visit(variable_node* num_node, Args... args) = 0;
    virtual void visit(parameter_node* node, Args... args) = 0;
    virtual void visit(number_node* op_node, Args... args) = 0;
    virtual void visit(sentinel_node* op_node, Args... args) = 0;
};



template<class... Args>
class IRTreeVoidReturnTypeVariadicArgumentVisitor {
public:
    virtual void visit(plus_node* op_node, Args... args) = 0;
    virtual void visit(minus_node* op_node, Args... args) = 0;
    virtual void visit(multiply_node* op_node, Args... args) = 0;
    virtual void visit(exponent_node* op_node, Args... args) = 0;
    virtual void visit(unary_negate_node* op_node, Args... args) = 0;
    virtual void visit(variable_node* num_node, Args... args) = 0;
    virtual void visit(parameter_node* node, Args... args) = 0;
    virtual void visit(number_node* op_node, Args... args) = 0;
    virtual void visit(sentinel_node* op_node, Args... args) = 0;
};



template<class... Args>
class IRTreeVoidReturnTypeVariadicArgumentConstantVisitor {
public:
    virtual void visit(const plus_node* op_node, Args... args) = 0;
    virtual void visit(const minus_node* op_node, Args... args) = 0;
    virtual void visit(const multiply_node* op_node, Args... args) = 0;
    virtual void visit(const exponent_node* op_node, Args... args) = 0;
    virtual void visit(const unary_negate_node* op_node, Args... args) = 0;
    virtual void visit(const variable_node* num_node, Args... args) = 0;
    virtual void visit(const parameter_node* node, Args... args) = 0;
    virtual void visit(const number_node* op_node, Args... args) = 0;
    virtual void visit(const sentinel_node* op_node, Args... args) = 0;
};


template<class return_t>
class IRTreeTemplateReturnTypeVisitor {
public:
    virtual return_t visit(plus_node* op_node) = 0;
    virtual return_t visit(minus_node* op_node) = 0;
    virtual return_t visit(multiply_node* op_node) = 0;
    virtual return_t visit(exponent_node* op_node) = 0;
    virtual return_t visit(unary_negate_node* op_node) = 0;
    virtual return_t visit(variable_node* num_node) = 0;
    virtual return_t visit(parameter_node* node) = 0;
    virtual return_t visit(number_node* op_node) = 0;
    virtual return_t visit(sentinel_node* op_node) = 0;
};


template<class return_t>
class IRTreeTemplateReturnTypeConstantVisitor {
public:
    // base template function set
    virtual return_t visit(const plus_node* op_node) = 0;
    virtual return_t visit(const minus_node* op_node) = 0;
    virtual return_t visit(const multiply_node* op_node) = 0;
    virtual return_t visit(const exponent_node* op_node) = 0;
    virtual return_t visit(const unary_negate_node* op_node) = 0;
    virtual return_t visit(const variable_node* num_node) = 0;
    virtual return_t visit(const parameter_node* node) = 0;
    virtual return_t visit(const number_node* op_node) = 0;
    virtual return_t visit(const sentinel_node* op_node) = 0;
};


class IRTreeVoidReturnTypeVisitor {
public:
    virtual void visit(plus_node* op_node);
    virtual void visit(minus_node* op_node);
    virtual void visit(multiply_node* op_node);
    virtual void visit(exponent_node* op_node);
    virtual void visit(unary_negate_node* op_node);
    virtual void visit(variable_node* num_node);
    virtual void visit(parameter_node* node);
    virtual void visit(number_node* op_node);
    virtual void visit(sentinel_node* op_node);
};


class IRTreeVoidReturnTypeConstantVisitor {
public:
    virtual void visit(const plus_node* op_node) = 0;
    virtual void visit(const minus_node* op_node) = 0;
    virtual void visit(const multiply_node* op_node) = 0;
    virtual void visit(const exponent_node* op_node) = 0;
    virtual void visit(const unary_negate_node* op_node) = 0;
    virtual void visit(const variable_node* num_node) = 0;
    virtual void visit(const parameter_node* node) = 0;
    virtual void visit(const number_node* op_node) = 0;
    virtual void visit(const sentinel_node* op_node) = 0;
};

#endif // IRTREETEMPLATERETURNVISITOR_H
