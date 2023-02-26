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

class merge_nodes_visitor final : public IRTreeVoidReturnTypeVisitorInterface {
public:
    void visit(plus_node* node);
	void visit(multiply_node* node);

private:
    using node_type = irtree_node::node_type;
};

#endif /* MERGE_NODES_VISITOR */