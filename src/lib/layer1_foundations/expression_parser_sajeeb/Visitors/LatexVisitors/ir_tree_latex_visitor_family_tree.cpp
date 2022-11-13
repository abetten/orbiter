//
// Created by Sajeeb Roy Chowdhury on 9/29/22.
//

#include "ir_tree_latex_visitor_family_tree.h"

using std::ostream;
using node_type = irtree_node::node_type;


#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;
#define OUT (*output_stream)

ir_tree_latex_visitor_family_tree::ir_tree_latex_visitor_family_tree() {}

ir_tree_latex_visitor_family_tree::ir_tree_latex_visitor_family_tree(std::ostream& ostream)
: ir_tree_latex_visitor(ostream) {}

ir_tree_latex_visitor_family_tree::~ir_tree_latex_visitor_family_tree() {}

void ir_tree_latex_visitor_family_tree::visit(plus_node* op_node) {
    OUT << "+" << "\n";
    add_indentation();
    for (auto it = op_node->children.rbegin(); it != op_node->children.rend(); ++it) {
        shared_ptr<irtree_node>& child = *it;
        OUT << indentation << "[";
        child->accept(this);
        if (!dynamic_cast<terminal_node*>(child.get())) OUT << indentation;
        OUT  << "]\n";
    }
    remove_indentation();
}

void ir_tree_latex_visitor_family_tree::visit(minus_node* op_node) {
    OUT << "-" << "\n";
    add_indentation();
    for (auto it = op_node->children.rbegin(); it != op_node->children.rend(); ++it) {
        shared_ptr<irtree_node>& child = *it;
        OUT << indentation << "[";
        child->accept(this);
        if (!dynamic_cast<terminal_node*>(child.get())) OUT << indentation;
        OUT  << "]\n";
    }
    remove_indentation();
}

void ir_tree_latex_visitor_family_tree::visit(multiply_node* op_node) {
    OUT << "*" << "\n";
    add_indentation();
    for (auto it = op_node->children.rbegin(); it != op_node->children.rend(); ++it) {
        shared_ptr<irtree_node>& child = *it;
        OUT << indentation << "[";
        child->accept(this);
        if (!dynamic_cast<terminal_node*>(child.get())) OUT << indentation;
        OUT  << "]\n";
    }
    remove_indentation();
}

void ir_tree_latex_visitor_family_tree::visit(exponent_node* op_node) {
    OUT << "\\^{}" << "\n";
    add_indentation();
    for (auto it = op_node->children.rbegin(); it != op_node->children.rend(); ++it) {
        shared_ptr<irtree_node>& child = *it;
        OUT << indentation << "[";
        child->accept(this);
        if (!dynamic_cast<terminal_node*>(child.get())) OUT << indentation;
        OUT  << "]\n";
    }
    remove_indentation();
}

void ir_tree_latex_visitor_family_tree::visit(unary_negate_node* op_node) {
    OUT << "(-)" << "\n";
    add_indentation();
    for (auto it = op_node->children.rbegin(); it != op_node->children.rend(); ++it) {
        shared_ptr<irtree_node>& child = *it;
        OUT << indentation << "[";
        child->accept(this);
        if (!dynamic_cast<terminal_node*>(child.get())) OUT << indentation;
        OUT  << "]\n";
    }
    remove_indentation();
}

void ir_tree_latex_visitor_family_tree::visit(variable_node* num_node) {
    OUT << num_node->name;
}

void ir_tree_latex_visitor_family_tree::visit(parameter_node* op_node) {
    OUT << op_node->name;
}

void ir_tree_latex_visitor_family_tree::visit(number_node* op_node) {
    OUT << op_node->value;
}

void ir_tree_latex_visitor_family_tree::visit(sentinel_node* op_node) {
    add_epilogue();
    OUT << indentation << "[S";
    add_indentation();
    for (auto it = op_node->children.rbegin(); it != op_node->children.rend(); ++it) {
        shared_ptr<irtree_node>& child = *it;
        OUT << indentation << "[";
        child->accept(this);
        OUT << indentation << "]\n";
    }
    remove_indentation();
    OUT << indentation << "]";
    add_prologue();
}

void ir_tree_latex_visitor_family_tree::add_epilogue() {
    OUT << "\\documentclass[tikz,12pt]{standalone}" << "\n"
        << "\\usetikzlibrary{calc,positioning,backgrounds,arrows.meta}" << "\n"
        << "\\usepackage{forest}" << "\n"
        << "\\pagestyle{empty}" << "\n"
        << "\\begin{document}" << "\n"
        << "\\begin{forest}" << "\n"
        << "for tree={" << "\n"
        << "    child anchor=west," << "\n"
        << "    parent anchor=east," << "\n"
        << "    grow=east," << "\n"
        << "    draw," << "\n"
        << "    anchor=west," << "\n"
        << "    edge path={" << "\n"
        << "        \\noexpand\\path[\\forestoption{edge}]" << "\n"
        << "        (.child anchor) -| +(-5pt,0) -- +(-5pt,0) |-" << "\n"
        << "                (!u.parent anchor)\\forestoption{edge label};" << "\n"
        << "    }," << "\n"
        << "}" << "\n";
}

void ir_tree_latex_visitor_family_tree::add_prologue() {
    OUT << "]" << "\n"
        << "\\end{forest}" << "\n"
        <<  "\\end{document}" << "\n";
}
