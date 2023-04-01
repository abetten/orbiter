/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/8/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include <string>
#include "exponent_vector_visitor.h"
#include "dispatcher.h"
#include "../IRTree/node.h"
#include <cassert>


typedef unsigned int uint32_t;

using std::string;
using std::shared_ptr;
using std::cout;
using std::endl;
using std::make_shared;

std::ostream& operator<< (std::ostream& os, const managed_variables_index_table& obj) {
    for (auto it=obj.begin(); it!=obj.end(); ++it)
        os << it->first << ": " << it->second << '\n';
    return os;
}


void exponent_vector_visitor::visit(sentinel_node* node) {
    LOG("");
    for (auto it=node->children.begin(); it!=node->children.end(); ++it) {
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
        dispatcher::visit(*it, *this, *it);
    }
}

void exponent_vector_visitor::visit(plus_node* node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    shared_ptr<irtree_node>& node_self,
                                    shared_ptr<irtree_node>& parent_node) {
    for (auto it=node->children.begin(); it != node->children.end(); ++it) {
        dispatcher::visit(*it, *this, exponent_vector, link, *it, node_self);
    }
}

void exponent_vector_visitor::visit(minus_node* node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    shared_ptr<irtree_node>& node_self,
                                    shared_ptr<irtree_node>& parent_node) {
    throw not_implemented(string(__FILE__) + "::" + std::to_string(__LINE__) + "::not implemented.");
}

void exponent_vector_visitor::visit(multiply_node* node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    shared_ptr<irtree_node>& node_self,
                                    shared_ptr<irtree_node>& parent_node) {
    for (auto it=node->children.begin(); it != node->children.end(); ++it) {
        dispatcher::visit(*it, *this, exponent_vector, link, *it,node_self);
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
void exponent_vector_visitor::visit(exponent_node* node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    shared_ptr<irtree_node>& node_self,
                                    shared_ptr<irtree_node>& parent_node) {
    auto power_iter = node->children.begin(), base_iter = power_iter++;
    assert((*power_iter)->is_terminal() == true);
    assert((*base_iter)->is_terminal() == true);
    if ((*base_iter)->type == irtree_node::node_type::VARIABLE_NODE) {
        if ((*power_iter)->type == irtree_node::node_type::NUMBER_NODE) {
            variable_node *base_node = static_cast<variable_node *>(base_iter->get());
            number_node *power_node = static_cast<number_node *>(power_iter->get());
            exponent_vector.at(symbol_table->index(base_node->name)) += power_node->value;
            non_terminal_node *non_terminal_parent_node = static_cast<non_terminal_node *>(parent_node.get());
            link = --(non_terminal_parent_node->children.erase(link));
        }
    }
}

void exponent_vector_visitor::visit(unary_negate_node* node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    shared_ptr<irtree_node>& node_self,
                                    shared_ptr<irtree_node>& parent_node) {

}

void exponent_vector_visitor::visit(variable_node* num_node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    shared_ptr<irtree_node>& node_self,
                                    shared_ptr<irtree_node>& parent_node) {
    exponent_vector.at(symbol_table->index(num_node->name)) += 1;
    link = --(static_cast<non_terminal_node*>(parent_node.get())->children.erase(link));
}

void exponent_vector_visitor::visit(parameter_node* node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    shared_ptr<irtree_node>& node_self,
                                    shared_ptr<irtree_node>& parent_node) {
    // NO-OP
}

void exponent_vector_visitor::visit(number_node* node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    shared_ptr<irtree_node>& node_self,
                                    shared_ptr<irtree_node>& parent_node) {
    // NO-OP
}

void exponent_vector_visitor::visit(sentinel_node* node,
                                    vector<uint32_t>& exponent_vector,
                                    list<shared_ptr<irtree_node> >::iterator& link,
                                    shared_ptr<irtree_node>& node_self,
                                    shared_ptr<irtree_node>& parent_node) {
    // NO-OP
    throw not_implemented(string(__FILE__) + "::" + std::to_string(__LINE__) + "::not implemented.");
}

void exponent_vector_visitor::visit(plus_node *node, shared_ptr<irtree_node> &node_self) {
    for (auto it=node->children.begin(); it!=node->children.end(); ++it) {
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
        dispatcher::visit(*it, *this, *it);
    }
}

void exponent_vector_visitor::visit(minus_node *node, shared_ptr<irtree_node> &node_self) {
    throw not_implemented(string(__FILE__) + "::" + std::to_string(__LINE__) + "::not implemented.");
}

void exponent_vector_visitor::visit(multiply_node *node, shared_ptr<irtree_node> &node_self) {
    vector<unsigned int> exponent_vector(symbol_table->size(), 0);
    for (auto it=node->children.begin(); it!=node->children.end(); ++it) {
        dispatcher::visit(*it, *this, exponent_vector, it, *it, node_self);
    }
    if (node->children.size() == 0)
        node->children.push_back(make_shared<number_node>(1));
    monomial_coefficient_table_[exponent_vector].push_back(node_self);
}

void exponent_vector_visitor::visit(exponent_node *node, shared_ptr<irtree_node> &node_self) {
    for (shared_ptr<irtree_node>& child : node->children)
        dispatcher::visit(child, *this, child);
}

void exponent_vector_visitor::visit(unary_negate_node *node, shared_ptr<irtree_node> &node_self) {
    for (shared_ptr<irtree_node>& child : node->children)
        dispatcher::visit(child, *this, child);
}

//void exponent_vector_visitor::visit(variable_node *node, shared_ptr<irtree_node> &node_self) {
//    // NO_OP
//}
//
//void exponent_vector_visitor::visit(parameter_node *node, shared_ptr<irtree_node> &node_self) {
//    // NO_OP
//}
//
//void exponent_vector_visitor::visit(number_node *node, shared_ptr<irtree_node> &node_self) {
//    // NO_OP
//}

void exponent_vector_visitor::visit(sentinel_node *node, shared_ptr<irtree_node> &node_self) {
    throw not_implemented(string(__FILE__) + "::" + std::to_string(__LINE__) + "::not implemented.");;
}
