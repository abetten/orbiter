//
// Created by Sajeeb Roy Chowdhury on 10/1/22.
//

#include "IRTreeVisitor.h"
#include "../IRTree/node.h"
#include <memory>

#ifndef REMOVE_MINUS_NODES_VISITOR_H
#define REMOVE_MINUS_NODES_VISITOR_H

class remove_minus_nodes_visitor : public IRTreeChildLinkArgumentVisitor {
public:
    bool found_operator_chain = false;
    void visit(minus_node* op_node, list<shared_ptr<irtree_node>>::iterator& link) override;
};

#endif // REMOVE_MINUS_NODES_VISITOR_H
