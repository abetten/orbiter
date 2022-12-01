/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/8/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "exponent_vector_visitor.h"
#include "../IRTree/node.h"
#include <cassert>

typedef unsigned int uint32_t;

using std::shared_ptr;
using std::cout;
using std::endl;
using std::make_shared;

std::ostream& operator<< (std::ostream& os, const managed_variables_index_table& obj) {
    for (auto it=obj.begin(); it!=obj.end(); ++it)
        os << it->first << ": " << it->second << '\n';
    return os;
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
void exponent_vector_visitor::visit(multiply_node* op_node) {
	vector<unsigned int> exponent_vector(symbol_table->size(), 0);
    for (auto it=op_node->children.begin(); it!=op_node->children.end(); ++it) {
        (*it)->accept(this, exponent_vector, it, op_node);
    }
    if (op_node->children.size() == 0)
        op_node->children.push_back(make_shared<number_node>(1));
    monomial_coefficient_table_[exponent_vector].push_back(op_node);
}

void exponent_vector_visitor::visit(plus_node* op_node) {
    for (auto it=op_node->children.begin(); it!=op_node->children.end(); ++it) {
        switch (it->get()->type) {
            case irtree_node::node_type::VARIABLE_NODE:
            case irtree_node::node_type::EXPONENT_NODE: {
                shared_ptr<multiply_node> new_multiply_node = std::make_shared<multiply_node>();
                new_multiply_node->children.push_back(make_shared<number_node>(1));
                new_multiply_node->children.push_back(*it);
                *it = new_multiply_node;
                break;
            }
            default: break;
        }
        it->get()->accept(this);
    }
}

void exponent_vector_visitor::visit(sentinel_node* op_node) {
    LOG("");
    for (auto it=op_node->children.begin(); it!=op_node->children.end(); ++it) {
        switch (it->get()->type) {
            case irtree_node::node_type::VARIABLE_NODE:
            case irtree_node::node_type::EXPONENT_NODE: {
                shared_ptr<multiply_node> new_multiply_node = std::make_shared<multiply_node>();
                new_multiply_node->children.push_back(make_shared<number_node>(1));
                new_multiply_node->children.push_back(*it);
                *it = new_multiply_node;
                break;
            }
            default: break;
        }
        it->get()->accept(this);
    }
}

void exponent_vector_visitor::visit(plus_node* op_node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    irtree_node* parent_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) {
        it->get()->accept(this, exponent_vector, link, op_node);
    }
}

void exponent_vector_visitor::visit(minus_node* op_node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    irtree_node* parent_node) {
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
void exponent_vector_visitor::visit(multiply_node* op_node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    irtree_node* parent_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) {
        it->get()->accept(this, exponent_vector, link, op_node);
    }
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
void exponent_vector_visitor::visit(exponent_node* op_node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    irtree_node* parent_node) {
    auto power_iter = op_node->children.begin(), base_iter = power_iter++;
    assert((*power_iter)->is_terminal() == true);
    assert((*base_iter)->is_terminal() == true);
    if ((*base_iter)->type == irtree_node::node_type::VARIABLE_NODE) {
        if ((*power_iter)->type == irtree_node::node_type::NUMBER_NODE) {
            variable_node *base_node = static_cast<variable_node *>(base_iter->get());
            number_node *power_node = static_cast<number_node *>(power_iter->get());
            exponent_vector.at(symbol_table->index(base_node->name)) += power_node->value;
            non_terminal_node *non_terminal_parent_node = static_cast<non_terminal_node *>(parent_node);
            link = --(non_terminal_parent_node->children.erase(link));
        }
    }
}

void exponent_vector_visitor::visit(unary_negate_node* op_node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    irtree_node* parent_node) {

}

void exponent_vector_visitor::visit(variable_node* num_node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    irtree_node* parent_node) {
    exponent_vector.at(symbol_table->index(num_node->name)) += 1;
    link = --(static_cast<non_terminal_node*>(parent_node)->children.erase(link));
}

void exponent_vector_visitor::visit(parameter_node* node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    irtree_node* parent_node) {
    // NO-OP
}

void exponent_vector_visitor::visit(number_node* op_node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    irtree_node* parent_node) {
    // NO-OP
}

void exponent_vector_visitor::visit(sentinel_node* op_node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    irtree_node* parent_node) {
    // NO-OP
    throw not_implemented("not implemented.");
}