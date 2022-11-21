/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/8/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "../IRTreeVisitor.h"

#ifndef GRAMMAR_TEST_EVAL_VISITOR_H
#define GRAMMAR_TEST_EVAL_VISITOR_H

class eval_visitor : public IRTreeTemplateReturnTypeVariadicArgumentConstantVisitorInterface<int, void*> {
    typedef int return_t;

public:
    return_t visit(const plus_node* op_node, void* Fq) override;
    return_t visit(const minus_node* op_node, void* Fq) override;
    return_t visit(const multiply_node* op_node, void* Fq) override;
    return_t visit(const exponent_node* op_node, void* Fq) override;
    return_t visit(const unary_negate_node* op_node, void* Fq) override;
    return_t visit(const variable_node* num_node, void* Fq) override;
    return_t visit(const parameter_node* node, void* Fq) override;
    return_t visit(const number_node* op_node, void* Fq) override;
    return_t visit(const sentinel_node* op_node, void* Fq) override;
};

#endif //GRAMMAR_TEST_EVAL_VISITOR_H
