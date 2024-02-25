#include "../IRTree/node_forward_declaration.h"
#include "IRTreeVisitor.h"
#include <iostream>
#include <memory>
#include <list>

#ifndef UMINUS_DISTRIBUTE_AND_REDUCE_VISITOR
#define UMINUS_DISTRIBUTE_AND_REDUCE_VISITOR

using std::list;
using std::cout;
using std::endl;
using std::shared_ptr;
using std::make_shared;

//! a visitor is a class that realizes an activity for processing an abstract syntax tree of Sajeeb type.

class uminus_distribute_and_reduce_visitor : public IRTreeVoidReturnTypeVisitorInterface,
											 public IRTreeVoidReturnTypeVariadicArgumentVisitorInterface<irtree_node*, 
											 			list<shared_ptr<irtree_node> >::iterator&> {

	void visit(plus_node* op_node, irtree_node* parent_node, list<shared_ptr<irtree_node> >::iterator& link) override;
	void visit(minus_node* op_node, irtree_node* parent_node, list<shared_ptr<irtree_node> >::iterator& link) override;
	void visit(multiply_node* op_node, irtree_node* parent_node, list<shared_ptr<irtree_node> >::iterator& link) override;
	void visit(exponent_node* op_node, irtree_node* parent_node, list<shared_ptr<irtree_node> >::iterator& link) override;
	void visit(unary_negate_node* op_node, irtree_node* parent_node, list<shared_ptr<irtree_node> >::iterator& link) override;
	void visit(variable_node* num_node, irtree_node* parent_node, list<shared_ptr<irtree_node> >::iterator& link) override;
    void visit(parameter_node* node, irtree_node* parent_node, list<shared_ptr<irtree_node> >::iterator& link) override;
    void visit(number_node* op_node, irtree_node* parent_node, list<shared_ptr<irtree_node> >::iterator& link) override;
	void visit(sentinel_node* op_node, irtree_node* parent_node, list<shared_ptr<irtree_node> >::iterator& link) override;

public:
	using IRTreeVoidReturnTypeVisitorInterface::visit;
	void visit(sentinel_node* op_node) override;

    friend dispatcher;
};

#endif /* UMINUS_DISTRIBUTE_AND_REDUCE_VISITOR */
