#include "../IRTree/node_forward_declaration.h"
#include "../IRTree/node.h"
#include "IRTreeVisitor.h"
#include <iostream>
#include <memory.h>

#ifndef UMINUS_DISTRIBUTE_AND_REDUCE_VISITOR
#define UMINUS_DISTRIBUTE_AND_REDUCE_VISITOR

using std::cout;
using std::endl;
using std::shared_ptr;
using std::make_shared;

class uminus_distribute_and_reduce_visitor : public IRTreeVoidReturnTypeVisitorInterface {
    using node_type = irtree_node::node_type;

    node_type previous_node;
	unary_negate_node* previous_uminus_node = NULL;
	shared_ptr<irtree_node>* previous_node_visiting_child;

public:
	void visit(plus_node* op_node) override;
	void visit(minus_node* op_node) override;
	void visit(multiply_node* op_node) override;
	void visit(exponent_node* op_node) override;
	void visit(unary_negate_node* op_node) override;
	void visit(sentinel_node* op_node) override;
};

#endif /* UMINUS_DISTRIBUTE_AND_REDUCE_VISITOR */