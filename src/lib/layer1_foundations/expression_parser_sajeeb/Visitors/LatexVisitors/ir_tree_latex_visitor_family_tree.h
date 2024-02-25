//
// Created by Sajeeb Roy Chowdhury on 9/29/22.
//

#include "ir_tree_latex_visitor.h"
#include "../../IRTree/node_forward_declaration.h"
#include "../../IRTree/node.h"
#include "../IRTreeVisitor.h"
#include <iostream>
#include <memory.h>

#ifndef IR_TREE_LATEX_VISITOR_FAMILY_TREE
#define IR_TREE_LATEX_VISITOR_FAMILY_TREE

//! a visitor is a class that realizes an activity for processing an abstract syntax tree of Sajeeb type.


class ir_tree_latex_visitor_family_tree : public IRTreeVoidReturnTypeVisitorInterface, public ir_tree_latex_visitor {
    void add_epilogue() override;
    void add_prologue() override;

public:
    ir_tree_latex_visitor_family_tree();
    ir_tree_latex_visitor_family_tree(std::ostream& ostream);
    virtual ~ir_tree_latex_visitor_family_tree();

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


#endif //IR_TREE_LATEX_VISITOR_FAMILY_TREE
