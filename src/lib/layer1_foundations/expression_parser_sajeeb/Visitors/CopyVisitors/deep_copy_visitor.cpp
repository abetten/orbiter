/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/6/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "deep_copy_visitor.h"
#include "../../IRTree/node.h"

#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << x << std::endl;

#define RETURN_T shared_ptr<irtree_node>

using std::make_shared;

/*
 * Entry point functions
 */

RETURN_T deep_copy_visitor::visit(const plus_node* op_node) {
    RETURN_T root = make_shared<plus_node>(*op_node);
    for (const shared_ptr<irtree_node>& child : op_node->children) {
        dispatcher::visit(child, *this, root);
    }
    return root;
}

RETURN_T deep_copy_visitor::visit(const minus_node* op_node) {
    RETURN_T root = make_shared<minus_node>(*op_node);
    for (const shared_ptr<irtree_node>& child : op_node->children) {
        dispatcher::visit(child, *this, root);
    }
    return root;
}

RETURN_T deep_copy_visitor::visit(const multiply_node* op_node) {
    RETURN_T root = make_shared<multiply_node>(*op_node);
    for (const shared_ptr<irtree_node>& child : op_node->children) {
        dispatcher::visit(child, *this, root);
    }
    return root; }

RETURN_T deep_copy_visitor::visit(const exponent_node* op_node) {
    RETURN_T root = make_shared<exponent_node>(*op_node);
    for (const shared_ptr<irtree_node>& child : op_node->children) {
        dispatcher::visit(child, *this, root);
    }
    return root; }

RETURN_T deep_copy_visitor::visit(const unary_negate_node* op_node) {
    RETURN_T root = make_shared<unary_negate_node>(*op_node);
    for (const shared_ptr<irtree_node>& child : op_node->children) {
        dispatcher::visit(child, *this, root);
    }
    return root; }

RETURN_T deep_copy_visitor::visit(const variable_node* num_node) {
    return make_shared<variable_node>(*num_node);
}

RETURN_T deep_copy_visitor::visit(const parameter_node* node) {
    return make_shared<parameter_node>(*node);
}

RETURN_T deep_copy_visitor::visit(const number_node* op_node) {
    return make_shared<number_node>(*op_node);
}

RETURN_T deep_copy_visitor::visit(const sentinel_node* op_node) {
    RETURN_T root = make_shared<sentinel_node>(*op_node);
    for (const shared_ptr<irtree_node>& child : op_node->children) {
        dispatcher::visit(child, *this, root);
    }
    return root; }


/*
 *
 */

void deep_copy_visitor::visit(const plus_node* op_node, RETURN_T root) {
    shared_ptr<irtree_node> cpy = make_shared<plus_node>(*op_node);
    static_cast<non_terminal_node*>(root.get())->add_child(cpy);
    for (const shared_ptr<irtree_node>& child : op_node->children)
        dispatcher::visit(child, *this, cpy);
}

void deep_copy_visitor::visit(const minus_node* op_node, RETURN_T root) {
    shared_ptr<irtree_node> cpy = make_shared<minus_node>(*op_node);
    static_cast<non_terminal_node*>(root.get())->add_child(cpy);
    for (const shared_ptr<irtree_node>& child : op_node->children)
        dispatcher::visit(child, *this, cpy);
}

void deep_copy_visitor::visit(const multiply_node* op_node, RETURN_T root) {
    shared_ptr<irtree_node> cpy = make_shared<multiply_node>(*op_node);
    static_cast<non_terminal_node*>(root.get())->add_child(cpy);
    for (const shared_ptr<irtree_node>& child : op_node->children)
        dispatcher::visit(child, *this, cpy);
}

void deep_copy_visitor::visit(const exponent_node* op_node, RETURN_T root) {
    shared_ptr<irtree_node> cpy = make_shared<exponent_node>(*op_node);
    static_cast<non_terminal_node*>(root.get())->add_child(cpy);
    for (const shared_ptr<irtree_node>& child : op_node->children)
        dispatcher::visit(child, *this, cpy);
}

void deep_copy_visitor::visit(const unary_negate_node* op_node, RETURN_T root) {
    shared_ptr<irtree_node> cpy = make_shared<unary_negate_node>(*op_node);
    static_cast<non_terminal_node*>(root.get())->add_child(cpy);
    for (const shared_ptr<irtree_node>& child : op_node->children)
        dispatcher::visit(child, *this, cpy);
}

void deep_copy_visitor::visit(const variable_node* num_node, RETURN_T root) {
    non_terminal_node* raw_ptr = static_cast<non_terminal_node*>(root.get());
    raw_ptr->add_child(make_shared<variable_node>(*num_node));
}

void deep_copy_visitor::visit(const parameter_node* node, RETURN_T root) {
    non_terminal_node* raw_ptr = static_cast<non_terminal_node*>(root.get());
    raw_ptr->add_child(make_shared<parameter_node>(*node));
}

void deep_copy_visitor::visit(const number_node* op_node, RETURN_T root) {
    non_terminal_node* raw_ptr = static_cast<non_terminal_node*>(root.get());
    raw_ptr->add_child(make_shared<number_node>(*op_node));
}

void deep_copy_visitor::visit(const sentinel_node* op_node, RETURN_T root) {
    shared_ptr<irtree_node> cpy = make_shared<sentinel_node>(*op_node);
    static_cast<non_terminal_node*>(root.get())->add_child(cpy);
    for (const shared_ptr<irtree_node>& child : op_node->children)
        dispatcher::visit(child, *this, cpy);
}
