/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/6/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "../IRTreeVisitor.h"
#include <memory>

#ifndef DEEP_COPY_VISITOR_H
#define DEEP_COPY_VISITOR_H

using std::shared_ptr;

class deep_copy_visitor : public IRTreeTemplateReturnTypeConstantVisitor<shared_ptr<irtree_node>>,
                          public IRTreeVoidReturnTypeVariadicArgumentConstantVisitor<shared_ptr<irtree_node>> {
    typedef shared_ptr<irtree_node> return_t;

    void visit(const plus_node* op_node, return_t _root) override;
    void visit(const minus_node* op_node, return_t _root) override;
    void visit(const multiply_node* op_node, return_t _root) override;
    void visit(const exponent_node* op_node, return_t _root) override;
    void visit(const unary_negate_node* op_node, return_t _root) override;
    void visit(const variable_node* num_node, return_t _root) override;
    void visit(const parameter_node* node, return_t _root) override;
    void visit(const number_node* op_node, return_t _root) override;
    void visit(const sentinel_node* op_node, return_t _root) override;

public:
    return_t visit(const plus_node* op_node) override;
    return_t visit(const minus_node* op_node) override;
    return_t visit(const multiply_node* op_node) override;
    return_t visit(const exponent_node* op_node) override;
    return_t visit(const unary_negate_node* op_node) override;
    return_t visit(const variable_node* num_node) override;
    return_t visit(const parameter_node* node) override;
    return_t visit(const number_node* op_node) override;
    return_t visit(const sentinel_node* op_node) override;

    friend class irtree_node;
    friend class plus_node;
    friend class minus_node;
    friend class multiply_node;
    friend class exponent_node;
    friend class unary_negate_node;
    friend class variable_node;
    friend class parameter_node;
    friend class number_node;
    friend class sentinel_node;

};

#endif //DEEP_COPY_VISITOR_H
