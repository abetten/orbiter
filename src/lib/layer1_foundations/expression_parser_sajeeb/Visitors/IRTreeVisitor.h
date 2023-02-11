/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/7/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/


#include "../IRTree/node_forward_declaration.h"
#include "dispatcher.h"
#include <memory.h>
#include <iostream>

#ifndef IRTREETEMPLATERETURNVISITOR_H
#define IRTREETEMPLATERETURNVISITOR_H

#ifndef LOG
#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;
#endif

using std::shared_ptr;

template<class return_t, class... Args>
class IRTreeTemplateReturnTypeVariadicArgumentVisitorInterface {
public:
    template<class Type, typename FAKE=void> return_t visit(shared_ptr<irtree_node>&, Args...) = delete;
    template<> return_t visit<plus_node, void>(shared_ptr<irtree_node>& node, Args... args) = 0;
    template<> return_t visit<minus_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<multiply_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<exponent_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<unary_negate_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<variable_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<parameter_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<number_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<sentinel_node>(shared_ptr<irtree_node>& node, Args... args);
};


template<class return_t, class... Args>
class IRTreeTemplateReturnTypeVariadicArgumentConstantVisitorInterface {
public:
    template<class Type> return_t visit(const shared_ptr<irtree_node>&, Args...) = delete;
    template<> return_t visit<plus_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<minus_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<multiply_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<exponent_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<unary_negate_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<variable_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<parameter_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<number_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> return_t visit<sentinel_node>(const shared_ptr<irtree_node>& node, Args... args);
};


template<class... Args>
class IRTreeVoidReturnTypeVariadicArgumentVisitorInterface {
public:
    template<class Type> void visit(shared_ptr<irtree_node>&, Args...) = delete;
    template<> void visit<plus_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<minus_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<multiply_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<exponent_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<unary_negate_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<variable_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<parameter_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<number_node>(shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<sentinel_node>(shared_ptr<irtree_node>& node, Args... args);
};



template<class... Args>
class IRTreeVoidReturnTypeVariadicArgumentConstantVisitorInterface {
public:
    template<class Type> void visit(const shared_ptr<irtree_node>&, Args...) = delete;
    template<> void visit<plus_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<minus_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<multiply_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<exponent_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<unary_negate_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<variable_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<parameter_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<number_node>(const shared_ptr<irtree_node>& node, Args... args);
    template<> void visit<sentinel_node>(const shared_ptr<irtree_node>& node, Args... args);
};


template<class return_t>
class IRTreeTemplateReturnTypeVisitorInterface {
public:
    template<class Type> return_t visit(shared_ptr<irtree_node>&) = delete;
    template<> return_t visit<plus_node>(shared_ptr<irtree_node>& node);
    template<> return_t visit<minus_node>(shared_ptr<irtree_node>& node);
    template<> return_t visit<multiply_node>(shared_ptr<irtree_node>& node);
    template<> return_t visit<exponent_node>(shared_ptr<irtree_node>& node);
    template<> return_t visit<unary_negate_node>(shared_ptr<irtree_node>& node);
    template<> return_t visit<variable_node>(shared_ptr<irtree_node>& node);
    template<> return_t visit<parameter_node>(shared_ptr<irtree_node>& node);
    template<> return_t visit<number_node>(shared_ptr<irtree_node>& node);
    template<> return_t visit<sentinel_node>(shared_ptr<irtree_node>& node);
};


template<class return_t>
class IRTreeTemplateReturnTypeConstantVisitorInterface {
public:
    template<class Type> return_t visit(const shared_ptr<irtree_node>&) = delete;
    template<> return_t visit<plus_node>(const shared_ptr<irtree_node>& node);
    template<> return_t visit<minus_node>(const shared_ptr<irtree_node>& node);
    template<> return_t visit<multiply_node>(const shared_ptr<irtree_node>& node);
    template<> return_t visit<exponent_node>(const shared_ptr<irtree_node>& node);
    template<> return_t visit<unary_negate_node>(const shared_ptr<irtree_node>& node);
    template<> return_t visit<variable_node>(const shared_ptr<irtree_node>& node);
    template<> return_t visit<parameter_node>(const shared_ptr<irtree_node>& node);
    template<> return_t visit<number_node>(const shared_ptr<irtree_node>& node);
    template<> return_t visit<sentinel_node>(const shared_ptr<irtree_node>& node);
};


class IRTreeVoidReturnTypeVisitorInterface {
public:
    template<class Type> void visit(shared_ptr<irtree_node>&) = delete;
    template<> void visit<plus_node>(shared_ptr<irtree_node>& node);
    template<> void visit<minus_node>(shared_ptr<irtree_node>& node);
    template<> void visit<multiply_node>(shared_ptr<irtree_node>& node);
    template<> void visit<exponent_node>(shared_ptr<irtree_node>& node);
    template<> void visit<unary_negate_node>(shared_ptr<irtree_node>& node);
    template<> void visit<variable_node>(shared_ptr<irtree_node>& node);
    template<> void visit<parameter_node>(shared_ptr<irtree_node>& node);
    template<> void visit<number_node>(shared_ptr<irtree_node>& node);
    template<> void visit<sentinel_node>(shared_ptr<irtree_node>& node);
};

template<>
void IRTreeVoidReturnTypeVisitorInterface::visit<plus_node>(shared_ptr<irtree_node>& node) {
    non_terminal_node* node_cast = static_cast<non_terminal_node*>(node.get());
    for (shared_ptr<irtree_node>& child : node_cast->children)
        dispatcher::visit(child, *this);
}
template<>
void IRTreeVoidReturnTypeVisitorInterface::visit<minus_node>(shared_ptr<irtree_node>& node) {
    non_terminal_node* node_cast = static_cast<non_terminal_node*>(node.get());
    for (shared_ptr<irtree_node>& child : node_cast->children)
        dispatcher::visit(child, *this);
}
template<>
void IRTreeVoidReturnTypeVisitorInterface::visit<multiply_node>(shared_ptr<irtree_node>& node) {
    non_terminal_node* node_cast = static_cast<non_terminal_node*>(node.get());
    for (shared_ptr<irtree_node>& child : node_cast->children)
        dispatcher::visit(child, *this);
}
template<>
void IRTreeVoidReturnTypeVisitorInterface::visit<exponent_node>(shared_ptr<irtree_node>& node) {
    non_terminal_node* node_cast = static_cast<non_terminal_node*>(node.get());
    for (shared_ptr<irtree_node>& child : node_cast->children)
        dispatcher::visit(child, *this);
}
template<>
void IRTreeVoidReturnTypeVisitorInterface::visit<unary_negate_node>(shared_ptr<irtree_node>& node) {
    non_terminal_node* node_cast = static_cast<non_terminal_node*>(node.get());
    for (shared_ptr<irtree_node>& child : node_cast->children)
        dispatcher::visit(child, *this);
}
template<>
void IRTreeVoidReturnTypeVisitorInterface::visit<sentinel_node>(shared_ptr<irtree_node>& node) {
    non_terminal_node* node_cast = static_cast<non_terminal_node*>(node.get());
    for (shared_ptr<irtree_node>& child : node_cast->children) dispatcher::visit(child, *this);
}

template<> void IRTreeVoidReturnTypeVisitorInterface::visit<variable_node>(shared_ptr<irtree_node>& node) {}
template<> void IRTreeVoidReturnTypeVisitorInterface::visit<parameter_node>(shared_ptr<irtree_node>& node) {}
template<> void IRTreeVoidReturnTypeVisitorInterface::visit<number_node>(shared_ptr<irtree_node>& node) {}


class IRTreeVoidReturnTypeConstantVisitorInterface {
public:
    template<class Type> void visit(const shared_ptr<irtree_node>&) = delete;
    template<> void visit<plus_node>(const shared_ptr<irtree_node>& node);
    template<> void visit<minus_node>(const shared_ptr<irtree_node>& node);
    template<> void visit<multiply_node>(const shared_ptr<irtree_node>& node);
    template<> void visit<exponent_node>(const shared_ptr<irtree_node>& node);
    template<> void visit<unary_negate_node>(const shared_ptr<irtree_node>& node);
    template<> void visit<variable_node>(const shared_ptr<irtree_node>& node);
    template<> void visit<parameter_node>(const shared_ptr<irtree_node>& node);
    template<> void visit<number_node>(const shared_ptr<irtree_node>& node);
    template<> void visit<sentinel_node>(const shared_ptr<irtree_node>& node);
};

#endif // IRTREETEMPLATERETURNVISITOR_H
