#include "../IRTreeVisitor.h"
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <iostream>

#ifndef IR_TREE_PRETTY_PRINT_VISITOR
#define IR_TREE_PRETTY_PRINT_VISITOR

using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::shared_ptr;

class ir_tree_pretty_print_visitor : public IRTreeVoidReturnTypeVisitorInterface {
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

private:
	string indentation = "";
	string delimiter = "\t";
	
	void add_indentation() {indentation.append(delimiter);}
	void remove_indentation() {
		for (size_t i=0; i<delimiter.size(); ++i) 
			indentation.pop_back();
	}
};

#endif /* IR_TREE_PRETTY_PRINT_VISITOR */