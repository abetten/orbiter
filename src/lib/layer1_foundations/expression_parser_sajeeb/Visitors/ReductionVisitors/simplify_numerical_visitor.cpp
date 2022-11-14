/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/21/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "simplify_numerical_visitor.h"
#include <memory>
#include <math.h>
#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;


typedef decltype(number_node::value) return_t;
using std::shared_ptr;
using std::make_shared;
using node_type = irtree_node::node_type;


void simplify_numerical_visitor::visit(plus_node* op_node) {}

void simplify_numerical_visitor::visit(minus_node* op_node) {}

void simplify_numerical_visitor::visit(multiply_node* op_node) {}

void simplify_numerical_visitor::visit(exponent_node* op_node) {}

void simplify_numerical_visitor::visit(unary_negate_node* op_node) {}

void simplify_numerical_visitor::visit(variable_node* num_node) {}

void simplify_numerical_visitor::visit(parameter_node* node) {}

void simplify_numerical_visitor::visit(number_node* op_node) {}

void simplify_numerical_visitor::visit(sentinel_node* op_node) {
    for (shared_ptr<irtree_node>& child : op_node->children)
        static_cast<void>(child->accept_simplify_numerical_visitor(this, op_node));
    if (!op_node->children.front()->is_terminal()) {
        non_terminal_node* child_node_raw_ptr = static_cast<non_terminal_node*>(op_node->children.front().get());
        if (child_node_raw_ptr->children.size() == 1)
            op_node->children.front() = child_node_raw_ptr->children.front();
    }
}



return_t simplify_numerical_visitor::visit(plus_node* op_node, irtree_node* parent_node) {
    return_t return_val = 0;
    bool append_node = false;
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) {
        shared_ptr<irtree_node>& child = *it;
        if (child->type == node_type::NUMBER_NODE) {
            return_val += child->accept_simplify_numerical_visitor(this, op_node);
            it = --op_node->children.erase(it);
            append_node = true;
        } else if (child->type == node_type::UNARY_NEGATE_NODE) {
            return_val -= child->accept_simplify_numerical_visitor(this, op_node);
            it = --op_node->children.erase(it);
            append_node = true;
        } else {
            static_cast<void>(child->accept_simplify_numerical_visitor(this, op_node));
            create_grandchild_link(child, [&it](){--it;});
        }
    }
    if (append_node) op_node->add_child(make_shared<number_node>(return_val));
    return return_val;
}

return_t simplify_numerical_visitor::visit(minus_node* op_node, irtree_node* parent_node) {
    LOG("minus node operation not implemented");
    return -1;
}

return_t simplify_numerical_visitor::visit(multiply_node* op_node, irtree_node* parent_node) {
    return_t return_val = 1;
    bool append_node = false;
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) {
        shared_ptr<irtree_node>& child = *it;
        if (child->type == irtree_node::node_type::NUMBER_NODE) {
            return_val *= child->accept_simplify_numerical_visitor(this, op_node);
            it = --op_node->children.erase(it);
            append_node = true;
        } else if (child->type == node_type::UNARY_NEGATE_NODE) {
            return_val *= -child->accept_simplify_numerical_visitor(this, op_node);
            it = --op_node->children.erase(it);
            append_node = true;
        } else {
            static_cast<void>(child->accept_simplify_numerical_visitor(this, op_node));
            create_grandchild_link(child, [&it](){--it;});
        }
    }
    if (append_node) op_node->add_child(make_shared<number_node>(return_val));
    return return_val;
}

return_t simplify_numerical_visitor::visit(exponent_node* op_node, irtree_node* parent_node) {
    return_t return_val = parent_node->type != node_type::PLUS_NODE;
    auto it2=op_node->children.begin(), it1=it2++;
    auto evaluate_fn = [&return_val, &it1, &it2, &op_node](simplify_numerical_visitor* self){
        auto base = (*it1)->accept_simplify_numerical_visitor(self, op_node);
        auto exponent = (*it2)->accept_simplify_numerical_visitor(self, op_node);
        return_val = pow(base, exponent);
        it2 = --op_node->children.erase(it2);
        static_cast<number_node*>(it1->get())->value = return_val;
    };
    if ((*it1)->type == node_type::NUMBER_NODE && (*it2)->type == node_type::NUMBER_NODE) 
        evaluate_fn(this);
    else {
        static_cast<void>((*it1)->accept_simplify_numerical_visitor(this, op_node));
        static_cast<void>((*it2)->accept_simplify_numerical_visitor(this, op_node));
        create_grandchild_link(*it1);
        create_grandchild_link(*it2);
        if ((*it1)->type == node_type::NUMBER_NODE && (*it2)->type == node_type::NUMBER_NODE) 
            evaluate_fn(this);
    }
    return return_val;
}

return_t simplify_numerical_visitor::visit(unary_negate_node* op_node, irtree_node* parent_node) {
    return_t return_val = parent_node->type == node_type::PLUS_NODE ? 0 :
                          parent_node->type == node_type::MULTIPLY_NODE ? 1 : 0;
    shared_ptr<irtree_node>& child = op_node->children.front();
    if (child->type == node_type::NUMBER_NODE) {
        return_val = child->accept_simplify_numerical_visitor(this, op_node);
        return return_val;
    }
    static_cast<void>(child->accept_simplify_numerical_visitor(this, op_node));
    create_grandchild_link(child);
    if (child->type == node_type::NUMBER_NODE)
        return_val = child->accept_simplify_numerical_visitor(this, op_node);
    return return_val;
}

return_t simplify_numerical_visitor::visit(variable_node* num_node, irtree_node* parent_node) {
    return 1;
}

return_t simplify_numerical_visitor::visit(parameter_node* node, irtree_node* parent_node) { return 1; }

return_t simplify_numerical_visitor::visit(number_node* op_node, irtree_node* parent_node) { return op_node->value; }

return_t simplify_numerical_visitor::visit(sentinel_node* op_node, irtree_node* parent_node) {
    return_t return_val = 0;
    for (shared_ptr<irtree_node>& child : op_node->children)
        return_val = child->accept_simplify_numerical_visitor(this, op_node);
    if (!op_node->children.front()->is_terminal()) {
        if (static_cast<non_terminal_node*>(op_node->children.front().get())->children.size() == 1)
            op_node->children.front() =
                    static_cast<non_terminal_node*>(op_node->children.front().get())->children.front();
    }
    return return_val;
}
