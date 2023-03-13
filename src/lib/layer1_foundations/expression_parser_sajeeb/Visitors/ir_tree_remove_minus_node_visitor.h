#include "../IRTree/node_forward_declaration.h"
#include "../IRTree/node.h"
#include "IRTreeVisitor.h"

#include <iostream>
#include <memory>
#include <algorithm>
#include <list>

#ifndef IR_TREE_REMOVE_MINUS_NODE_VISITOR
#define IR_TREE_REMOVE_MINUS_NODE_VISITOR

using std::shared_ptr;
using std::make_shared;
using std::list;
using std::cout;
using std::endl;

class ir_tree_remove_minus_node_visitor : public IRTreeVoidReturnTypeVisitorInterface {
public:
    using IRTreeVoidReturnTypeVisitorInterface::visit;

    void visit(plus_node* op_node) override;
	void visit(minus_node* op_node) override;
	void visit(multiply_node* op_node) override;
	void visit(exponent_node* op_node) override;
	void visit(unary_negate_node* op_node) override;

private:
	non_terminal_node* visited_minus_node;

	void merge_visited_minus_node(non_terminal_node* root, 
	list<shared_ptr<irtree_node>>::iterator& it);
};


#endif /* IR_TREE_REMOVE_MINUS_NODE_VISITOR */