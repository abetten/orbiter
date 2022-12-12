/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.22.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "ir_tree_latex_visitor_simple_tree.h"

using std::ostream;
using node_type = irtree_node::node_type;


#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;
#define OUT (*output_stream)

ir_tree_latex_visitor_simple_tree::ir_tree_latex_visitor_simple_tree() {}

ir_tree_latex_visitor_simple_tree::ir_tree_latex_visitor_simple_tree(std::ostream& ostream)
: ir_tree_latex_visitor(ostream) {}

ir_tree_latex_visitor_simple_tree::~ir_tree_latex_visitor_simple_tree() {}

void ir_tree_latex_visitor_simple_tree::visit(plus_node* op_node) {
    OUT << "\\fbox{+}, arn_n" << "\n";
    add_indentation();
    for (shared_ptr<irtree_node>& child : op_node->children) {
        OUT << indentation << "[";
        child->accept(this);
        if (!dynamic_cast<terminal_node*>(child.get())) OUT << indentation;
        OUT  << "]\n";
    }
    remove_indentation();
}

void ir_tree_latex_visitor_simple_tree::visit(minus_node* op_node) {
    OUT << "\\fbox{-}, arn_n" << "\n";
    add_indentation();
    for (shared_ptr<irtree_node>& child : op_node->children) {
        OUT << indentation << "[";
        child->accept(this);
        if (!dynamic_cast<terminal_node*>(child.get())) OUT << indentation;
        OUT  << "]\n";
    }
    remove_indentation();
}

void ir_tree_latex_visitor_simple_tree::visit(multiply_node* op_node) {
    OUT << "\\fbox{*}, arn_n" << "\n";
    add_indentation();
    for (shared_ptr<irtree_node>& child : op_node->children) {
        OUT << indentation << "[";
        child->accept(this);
        if (!dynamic_cast<terminal_node*>(child.get())) OUT << indentation;
        OUT  << "]\n";
    }
    remove_indentation();
}

void ir_tree_latex_visitor_simple_tree::visit(exponent_node* op_node) {
    OUT << "\\fbox{\\^{}}, arn_n" << "\n";
    add_indentation();
    for (shared_ptr<irtree_node>& child : op_node->children) {
        OUT << indentation << "[";
        child->accept(this);
        if (!dynamic_cast<terminal_node*>(child.get())) OUT << indentation;
        OUT  << "]\n";
    }
    remove_indentation();
}

void ir_tree_latex_visitor_simple_tree::visit(unary_negate_node* op_node) {
    OUT << "\\fbox{(-)}, arn_n" << "\n";
    add_indentation();
    for (shared_ptr<irtree_node>& child : op_node->children) {
        OUT << indentation << "[";
        child->accept(this);
        if (!dynamic_cast<terminal_node*>(child.get())) OUT << indentation;
        OUT  << "]\n";
    }
    remove_indentation();
}

void ir_tree_latex_visitor_simple_tree::visit(variable_node* num_node) {
    OUT << num_node->name << ", arn_x";
}

void ir_tree_latex_visitor_simple_tree::visit(parameter_node* op_node) {
    OUT << op_node->name << ", arn_x";
}

void ir_tree_latex_visitor_simple_tree::visit(number_node* op_node) {
    OUT << op_node->value << ", arn_x";
}

void ir_tree_latex_visitor_simple_tree::visit(sentinel_node* op_node) {
    add_epilogue();
    add_indentation();
    for (shared_ptr<irtree_node>& child : op_node->children) {
        OUT << indentation << "[";
        child->accept(this);
        OUT << indentation << "]\n";
    }
    remove_indentation();
    add_prologue();
}

void ir_tree_latex_visitor_simple_tree::add_epilogue() {
    OUT << "\\documentclass{standalone}" << "\n"
        << "\\usepackage{forest}" << "\n"
        << "\\usetikzlibrary{arrows.meta}" << "\n"
        << "\\begin{document}" << "\n"
        << "\\begin{forest}" << "\n"
        << "for tree={" << "\n"
        << "grow'=0, treenode/.style = {align=center, inner sep=2.5pt," << "\n"
        << "    text centered, font=\\sffamily}," << "\n"
        << "arn_n/.style = {treenode, rectangle, text width=1.5em}," << "\n"
        << "arn_x/.style = {treenode}," << "\n"
        << "gray-arrow/.style = {draw=gray}," << "\n"
        << "edge=-{Stealth}}" << "\n"
        << "[\\fbox{S}, arn_n" << "\n";
}

void ir_tree_latex_visitor_simple_tree::add_prologue() {
    OUT << "]" << "\n"
        << "\\end{forest}" << "\n"
        <<  "\\end{document}" << "\n";
}