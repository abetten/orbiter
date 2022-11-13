/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/8/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "exponent_vector_visitor.h"
#include "../IRTree/node.h"


typedef unsigned int uint32_t;

using std::shared_ptr;
using std::cout;
using std::endl;

/**
 * f0  -> "+"
 *    |-> "-"
 *    |-> "*"
 *    |-> "^"
 *    |-> "NUMBER"
 *    |-> "PARAMETER"
 *    |-> "VARIABLE"
 */
void exponent_vector_visitor::visit(multiply_node* op_node) {
    vector<unsigned int> exponent_vector;

    for (auto it=op_node->children.begin(); it!=op_node->children.end(); ++it)
        (*it)->accept(this, exponent_vector);



    monomial_coefficient_table_.insert({std::move(exponent_vector), op_node});
}

void exponent_vector_visitor::visit(plus_node* op_node,vector<uint32_t>& exponent_vector) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this, exponent_vector);
}

void exponent_vector_visitor::visit(minus_node* op_node,vector<uint32_t>& exponent_vector) {
    throw not_implemented("not implemented.");
}

/**
 * f0  -> "+"
 *    |-> "-"
 *    |-> "*"
 *    |-> "^"
 *    |-> "NUMBER"
 *    |-> "PARAMETER"
 *    |-> "VARIABLE"
 */
void exponent_vector_visitor::visit(multiply_node* op_node,vector<uint32_t>& exponent_vector) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        child->accept(this, exponent_vector);
}

/**
 * base, exponent  -> "+"
 *                |-> "-"
 *                |-> "*"
 *                |-> "^"
 *                |-> "NUMBER"
 *                |-> "PARAMETER"
 *                |-> "VARIABLE"
 */
void exponent_vector_visitor::visit(exponent_node* op_node,vector<uint32_t>& exponent_vector) {
    auto power_iter = op_node->children.begin(), base_iter = power_iter++;
    assert((*power_iter)->is_terminal() == true);
    assert((*base_iter)->is_terminal() == true);
    if ((*base_iter)->type == irtree_node::node_type::VARIABLE_NODE)
        if ((*power_iter)->type == irtree_node::node_type::NUMBER_NODE) {
            variable_node* base_node = static_cast<variable_node*>(base_iter->get());
            number_node* power_node = static_cast<number_node*>(power_iter->get());
            exponent_vector.at(symbol_table->index(base_node->name)) += power_node->value;
        }
}

void exponent_vector_visitor::visit(unary_negate_node* op_node,vector<uint32_t>& exponent_vector) {}

void exponent_vector_visitor::visit(variable_node* num_node,vector<uint32_t>& exponent_vector) {
    exponent_vector.at(symbol_table->index(num_node->name)) += 1;
}

void exponent_vector_visitor::visit(parameter_node* node,vector<uint32_t>& exponent_vector) {
    // NO-OP
}

void exponent_vector_visitor::visit(number_node* op_node,vector<uint32_t>& exponent_vector) {
    // NO-OP
}

void exponent_vector_visitor::visit(sentinel_node* op_node,vector<uint32_t>& exponent_vector) {
    // NO-OP
    throw not_implemented("not implemented.");
}
