/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.26.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "../../IRTree/node_forward_declaration.h"
#include "../../IRTree/node.h"
#include "../IRTreeVisitor.h"
#include <iostream>
#include <string>
#include <memory.h>

#ifndef IR_TREE_TO_STRING_VISITOR
#define IR_TREE_TO_STRING_VISITOR

using std::string;
using std::list;

//! a visitor is a class that realizes an activity for processing an abstract syntax tree of Sajeeb type.

class ir_tree_to_string_visitor : public IRTreeVoidReturnTypeVisitorInterface {
    list<string> rep;
    list<string>::iterator loc;
    list<string>::iterator left;

public:
    using IRTreeVoidReturnTypeVisitorInterface::visit;

    string get_string_representation() const {
        string res;
        for (auto it=rep.begin(); it != rep.end(); ++it)
            res.append(*it);
        return res;
    }

    void visit(plus_node* op_node) override;
    void visit(multiply_node* op_node) override;
    void visit(exponent_node* op_node) override;
    void visit(unary_negate_node* op_node) override;
    void visit(variable_node* num_node) override;
    void visit(parameter_node* node) override;
    void visit(number_node* op_node) override;
    void visit(sentinel_node* op_node) override;
};


#endif //IR_TREE_TO_STRING_VISITOR
