/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/8/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "eval_visitor.h"
#include "../../IRTree/node.h"
#include <memory>

using std::shared_ptr;

int eval_visitor::visit(const plus_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) {
    int return_value = 0;
    for (shared_ptr<irtree_node> child : op_node->children) {
        return_value = Fq->add(return_value, child->accept(this, Fq, assignment_table));
    }
    return return_value;
}
int eval_visitor::visit(const minus_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) {
    return 0;
}
int eval_visitor::visit(const multiply_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) {
    int return_value = 1;
    for (shared_ptr<irtree_node> child : op_node->children) {
        return_value = Fq->mult(return_value, child->accept(this, Fq, assignment_table));
    }
    return return_value;
}
int eval_visitor::visit(const exponent_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) {
    auto power_iter = op_node->children.begin(), base_iter = power_iter++;
    int base_eval = base_iter->get()->accept(this, Fq, assignment_table);
    int power_eval = power_iter->get()->accept(this, Fq, assignment_table);
    return Fq->power(base_eval, power_eval);
}
int eval_visitor::visit(const unary_negate_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) {
    return Fq->negate(op_node->children.front()->accept(this, Fq, assignment_table));
}
int eval_visitor::visit(const variable_node* num_node, finite_field* Fq, unordered_map<string, int>& assignment_table) {
    return 1;
}
int eval_visitor::visit(const parameter_node* node, finite_field* Fq, unordered_map<string, int>& assignment_table) {
    return assignment_table.at(node->name);
}
int eval_visitor::visit(const number_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) {
    return op_node->value;
}

int eval_visitor::visit(const sentinel_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) {
    int return_value = 0;
    for (shared_ptr<irtree_node> child : op_node->children) {
        return_value = child->accept(this, Fq, assignment_table);
    }
    return return_value;
}