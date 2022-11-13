/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.22.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "ir_tree_latex_visitor.h"
#include "../../IRTree/node_forward_declaration.h"
#include "../../IRTree/node.h"
#include "../IRTreeVisitor.h"
#include <iostream>
#include <memory.h>

#ifndef IR_TREE_LATEX_VISITOR_SIMPLE_TREE
#define IR_TREE_LATEX_VISITOR_SIMPLE_TREE

#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;

class ir_tree_latex_visitor_simple_tree final : public ir_tree_latex_visitor, public IRTreeVoidReturnTypeVisitor {
    void add_epilogue() override;
    void add_prologue() override;

public:
    ir_tree_latex_visitor_simple_tree();
    ir_tree_latex_visitor_simple_tree(std::ostream& ostream);
    virtual ~ir_tree_latex_visitor_simple_tree();

    void visit(plus_node* op_node) override;
    void visit(minus_node* op_node) override;
    void visit(multiply_node* op_node) override;
    void visit(exponent_node* op_node) override;
    void visit(unary_negate_node* op_node) override;
    void visit(variable_node* num_node) override;
    void visit(parameter_node* node) override;
    void visit(number_node* op_node) override;
    void visit(sentinel_node* op_node) override;
};

#endif /* IR_TREE_LATEX_VISITOR_SIMPLE_TREE */
