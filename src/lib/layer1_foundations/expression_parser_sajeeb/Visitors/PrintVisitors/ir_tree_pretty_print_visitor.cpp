
/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.22.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/
    
#include "ir_tree_pretty_print_visitor.h"
#include "../../IRTree/node.h"


void ir_tree_pretty_print_visitor::visit(plus_node* op_node) {
	cout << indentation << "PLUS\n";
	add_indentation();
	for (shared_ptr<irtree_node>& child : op_node->children) 
		child->accept(this);
	remove_indentation();
}

void ir_tree_pretty_print_visitor::visit(minus_node* op_node) {
	cout << indentation << "MINUS\n";
	add_indentation();
	for (shared_ptr<irtree_node>& child : op_node->children) 
		child->accept(this);
	remove_indentation();
}

void ir_tree_pretty_print_visitor::visit(multiply_node* op_node) {
	cout << indentation << "TIMES\n";
	add_indentation();
	for (shared_ptr<irtree_node>& child : op_node->children) 
		child->accept(this);
	remove_indentation();
}

void ir_tree_pretty_print_visitor::visit(exponent_node* op_node) {
	cout << indentation << "EXPONENT\n";
	add_indentation();
	for (shared_ptr<irtree_node>& child : op_node->children) 
		child->accept(this);
	remove_indentation();
}

void ir_tree_pretty_print_visitor::visit(unary_negate_node* op_node) {
	cout << indentation << "(-)\n";
	add_indentation();
	for (shared_ptr<irtree_node>& child : op_node->children)
		child->accept(this);
	remove_indentation();
}

void ir_tree_pretty_print_visitor::visit(sentinel_node* op_node) {
	cout << indentation << "START\n";
	add_indentation();
	for (shared_ptr<irtree_node>& child : op_node->children) 
		child->accept(this);
	remove_indentation();
}

void ir_tree_pretty_print_visitor::visit(variable_node* num_node) {
	cout << indentation << "CONSTANT (" << num_node->name << ")\n";
}

void ir_tree_pretty_print_visitor::visit(parameter_node* op_node) {
	cout << indentation << "PARAMETER (" << op_node->name << ")\n";
}

void ir_tree_pretty_print_visitor::visit(number_node* op_node) {
	cout << indentation << "NUMBER (" << op_node->value<< ")\n";
}
