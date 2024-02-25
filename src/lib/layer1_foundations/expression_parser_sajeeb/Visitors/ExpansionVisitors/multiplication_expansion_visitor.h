/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10.04.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "../IRTreeChildLinkArgumentVisitor.h"
#include "../../exception.h"
#include <iostream>
#include <string>

#ifndef MULTIPLICATION_EXPANSION_VISITOR
#define MULTIPLICATION_EXPANSION_VISITOR


//! a visitor is a class that realizes an activity for processing an abstract syntax tree of Sajeeb type.

class multiplication_expansion_visitor : public IRTreeChildLinkArgumentVisitor {
    typedef list<shared_ptr<irtree_node>>::iterator iterator_t;

    void expand_multiplication_node(multiply_node*& op_node, iterator_t& link);

public:
    using IRTreeChildLinkArgumentVisitor::visit;

    void visit(multiply_node* op_node, list<shared_ptr<irtree_node> >::iterator& link) override;
};


#endif
