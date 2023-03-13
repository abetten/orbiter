/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/21/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/




#include "simplify_visitor.h"
#include "../dispatcher.h"
#include <functional>
#include "../../IRTree/node.h"
#include <memory>


#ifndef SIMPLIFY_NUMERICAL_VISITOR_H
#define SIMPLIFY_NUMERICAL_VISITOR_H

using std::shared_ptr;

class simplify_numerical_visitor : public simplify_visitor,
                                   public IRTreeTemplateReturnTypeVariadicArgumentVisitorInterface<decltype(number_node::value),
                                           irtree_node*> {

    inline 
    void create_grandchild_link(shared_ptr<irtree_node>& child,
                                const std::function<void(void)>& fn1 = [](){},
                                const std::function<void(void)>& fn2 = [](){}) const {
        if ((!child->is_terminal()) && (static_cast<non_terminal_node*>(child.get())->children.size() == 1)) {
            fn1();
            child = static_cast<non_terminal_node*>(child.get())->children.front();
        }
        fn2();
    }

public:
    using simplify_visitor::visit;
    typedef decltype(number_node::value) return_t;

    return_t visit(plus_node* node, irtree_node* parent_node) override;
    return_t visit(minus_node* node, irtree_node* parent_node) override;
    return_t visit(multiply_node* node, irtree_node* parent_node) override;
    return_t visit(exponent_node* node, irtree_node* parent_node) override;
    return_t visit(unary_negate_node* node, irtree_node* parent_node) override;
    return_t visit(variable_node* num_node, irtree_node* parent_node) override;
    return_t visit(parameter_node* node, irtree_node* parent_node) override;
    return_t visit(number_node* node, irtree_node* parent_node) override;
    return_t visit(sentinel_node* node, irtree_node* parent_node) override;

    void visit(plus_node* node) override;
    void visit(minus_node* node) override;
    void visit(multiply_node* node) override;
    void visit(exponent_node* node) override;
    void visit(unary_negate_node* node) override;
    void visit(variable_node* num_node) override;
    void visit(parameter_node* node) override;
    void visit(number_node* node) override;
    void visit(sentinel_node* node) override;

    friend dispatcher;
};


#endif //SIMPLIFY_NUMERICAL_VISITOR_H
