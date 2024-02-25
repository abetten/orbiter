/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/2/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "IRTreeVisitor.h"
#include "../IRTree/node_forward_declaration.h"
#include <list>
#include <memory>

#ifndef IRTREECHILDLINKARGUMENTVISITOR_H
#define IRTREECHILDLINKARGUMENTVISITOR_H

using std::list;
using std::shared_ptr;


//! a visitor is a class that realizes an activity for processing an abstract syntax tree of Sajeeb type.

class IRTreeChildLinkArgumentVisitor : public IRTreeVoidReturnTypeVisitorInterface,
                                       public IRTreeVoidReturnTypeVariadicArgumentVisitorInterface<
                                                list<shared_ptr<irtree_node> >::iterator&> {
public:
    void visit(plus_node* op_node) override;
    void visit(minus_node* op_node) override;
    void visit(multiply_node* op_node) override;
    void visit(exponent_node* op_node) override;
    void visit(unary_negate_node* op_node) override;
    void visit(variable_node* num_node) override;
    void visit(parameter_node* node) override;
    void visit(number_node* op_node) override;
    void visit(sentinel_node* op_node) override;

    void visit(plus_node* op_node, list<shared_ptr<irtree_node> >::iterator& link) override;
    void visit(minus_node* op_node, list<shared_ptr<irtree_node> >::iterator& link) override;
    void visit(multiply_node* op_node, list<shared_ptr<irtree_node> >::iterator& link) override;
    void visit(exponent_node* op_node, list<shared_ptr<irtree_node> >::iterator& link) override;
    void visit(unary_negate_node* op_node, list<shared_ptr<irtree_node> >::iterator& link) override;
    void visit(variable_node* num_node, list<shared_ptr<irtree_node> >::iterator& link) override;
    void visit(parameter_node* node, list<shared_ptr<irtree_node> >::iterator& link) override;
    void visit(number_node* op_node, list<shared_ptr<irtree_node> >::iterator& link) override;
    void visit(sentinel_node* op_node, list<shared_ptr<irtree_node> >::iterator& link) override;
};


#endif //IRTREECHILDLINKARGUMENTVISITOR_H
