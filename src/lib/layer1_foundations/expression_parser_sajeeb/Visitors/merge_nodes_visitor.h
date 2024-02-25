#include "../IRTree/node_forward_declaration.h"
#include "../IRTree/node.h"
#include "IRTreeVisitor.h"
#include <iostream>
#include <memory.h>

#ifndef MERGE_NODES_VISITOR
#define MERGE_NODES_VISITOR

using std::cout;
using std::endl;
using std::shared_ptr;
using std::make_shared;

//! a visitor is a class that realizes an activity for processing an abstract syntax tree of Sajeeb type.

class merge_nodes_visitor final : public IRTreeVoidReturnTypeVisitorInterface {
public:
    using IRTreeVoidReturnTypeVisitorInterface::visit;

    void visit(plus_node* node) override;
	void visit(multiply_node* node) override;

private:
    using node_type = irtree_node::node_type;
	template <typename T>
	inline void make_child_nodes(irtree_node* child, T* node) {
		for (shared_ptr<irtree_node>& grandchild : static_cast<non_terminal_node*>(child)->children) 
			node->children.push_back(grandchild);
	}
};

#endif /* MERGE_NODES_VISITOR */
