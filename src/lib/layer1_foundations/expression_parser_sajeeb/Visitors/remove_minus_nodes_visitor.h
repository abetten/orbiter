//
// Created by Sajeeb Roy Chowdhury on 10/1/22.
//

#include "IRTreeChildLinkArgumentVisitor.h"
#include "../IRTree/node.h"
#include <memory>

#ifndef REMOVE_MINUS_NODES_VISITOR_H
#define REMOVE_MINUS_NODES_VISITOR_H

//! a visitor is a class that realizes an activity for processing an abstract syntax tree of Sajeeb type.


class remove_minus_nodes_visitor : public IRTreeChildLinkArgumentVisitor {
public:
    using IRTreeChildLinkArgumentVisitor::visit;

    bool found_operator_chain = false;
    void visit(minus_node* op_node, list<shared_ptr<irtree_node>>::iterator& link) override;
};

#endif // REMOVE_MINUS_NODES_VISITOR_H
