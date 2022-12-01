
/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.22.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/
    
#include "uminus_distribute_and_reduce_visitor.h"

#define BACKUP_PREVIOUS_BOOKKEEPING_INFO \
			node_type backup_previous_node = previous_node; \
			shared_ptr<irtree_node>* backup_previous_node_visiting_child = previous_node_visiting_child

#define RESTORE_PREVIOUS_BOOKKEEPING_INFO \
			previous_node = backup_previous_node; \
			previous_node_visiting_child = backup_previous_node_visiting_child

#define STORE_BOOKKEEPING_INFO(PREVIOUS_NODE, VISITING_CHILD) \
			previous_node = PREVIOUS_NODE; \
			previous_node_visiting_child = VISITING_CHILD

void uminus_distribute_and_reduce_visitor::visit(plus_node* op_node) {
    BACKUP_PREVIOUS_BOOKKEEPING_INFO;

    LOG("");
    if (previous_node == node_type::UNARY_NEGATE_NODE) {
        // bypass uminus node
        LOG("");
        *previous_node_visiting_child = previous_uminus_node->children.front();
        // distribute uminus to children
        LOG("");
        for (shared_ptr<irtree_node>& child : op_node->children) {
            LOG("");
            shared_ptr<non_terminal_node> uminus = make_shared<unary_negate_node>();
            LOG("");
            uminus->children.emplace_back(child);
            LOG("");
            child = uminus;
            LOG("");
        }
        LOG("");
    }
    LOG("");

    for (shared_ptr<irtree_node>& child : op_node->children) {
        previous_node = op_node->type;
        previous_node_visiting_child = &child;
        LOG("");
        child->accept(this);
        LOG("");
    }
    LOG("");

    RESTORE_PREVIOUS_BOOKKEEPING_INFO;
    LOG("");
}

void uminus_distribute_and_reduce_visitor::visit(minus_node* op_node) {
    BACKUP_PREVIOUS_BOOKKEEPING_INFO;
    for (shared_ptr<irtree_node>& child : op_node->children) {
        previous_node = op_node->type;
        previous_node_visiting_child = &child;
        child->accept(this);
    }    
    RESTORE_PREVIOUS_BOOKKEEPING_INFO;
}

void uminus_distribute_and_reduce_visitor::visit(multiply_node* op_node) {
    BACKUP_PREVIOUS_BOOKKEEPING_INFO;

    LOG("");
   size_t negative_children_count = 0;
    for (shared_ptr<irtree_node>& child : op_node->children) {
        LOG("");
        negative_children_count += child->type == node_type::UNARY_NEGATE_NODE;
    }
        LOG("");
    // function to drop all the following uminus nodes and update the negative children counter
    auto drop_following_uminus_nodes = [&]()->void {
        LOG("");
        if (negative_children_count > 0)
            LOG("");
            for (shared_ptr<irtree_node>& child : op_node->children) { // drop proceeding uminus nodes
                if (child->type == node_type::UNARY_NEGATE_NODE) {
                    LOG("");
                    non_terminal_node* child_raw_ptr = static_cast<non_terminal_node*>(child.get());
                    LOG("");
                    shared_ptr<irtree_node> grandchild = child_raw_ptr->children.front();
                    LOG("");
                    child = grandchild;
                    negative_children_count -= 1;
                    LOG("");
                }
                LOG("");
            }
            LOG("");
    };
    LOG("");

    // 
    if (previous_node == node_type::UNARY_NEGATE_NODE) {     
        LOG("");
        auto prior_negative_children_count = negative_children_count;
        drop_following_uminus_nodes();
        LOG("");
        if (prior_negative_children_count % 2 != 0) { // odd
            *previous_node_visiting_child = previous_uminus_node->children.front();
        } else { // even
            LOG("");
            *previous_node_visiting_child = previous_uminus_node->children.front();
            shared_ptr<non_terminal_node> uminus = make_shared<unary_negate_node>();
            uminus->children.emplace_back(op_node->children.front());
            op_node->children.front() = uminus;
            negative_children_count += 1;
            LOG("");
        }
        LOG("");
    }
    LOG("");
    if (negative_children_count % 2 == 0) { // even number of proceeding negative nodes
        LOG("");
        drop_following_uminus_nodes();
        LOG("");
    }
    LOG("");

    for (shared_ptr<irtree_node>& child : op_node->children) {
        previous_node = op_node->type;
        previous_node_visiting_child = &child;
        LOG("");
        child->accept(this);
        LOG("");
    }
    LOG("");
    RESTORE_PREVIOUS_BOOKKEEPING_INFO;
}

void uminus_distribute_and_reduce_visitor::visit(exponent_node* op_node) {
    BACKUP_PREVIOUS_BOOKKEEPING_INFO;
    LOG("");
    for (shared_ptr<irtree_node>& child : op_node->children) {
        previous_node = op_node->type;
        previous_node_visiting_child = &child;
        LOG("");
        child->accept(this);
        LOG("");
    }
    LOG("");
    RESTORE_PREVIOUS_BOOKKEEPING_INFO;
}

void uminus_distribute_and_reduce_visitor::visit(unary_negate_node* op_node) {
    auto backup_previous_uminus_node = previous_uminus_node;
    previous_uminus_node = op_node;
    non_terminal_node* child_raw_ptr = static_cast<non_terminal_node*>(op_node->children.front().get());
    switch (child_raw_ptr->type) {
        case node_type::UNARY_NEGATE_NODE: {
            shared_ptr<irtree_node>& grandchild = child_raw_ptr->children.front();
            *previous_node_visiting_child = grandchild;
            grandchild->accept(this);
            break;
        }
        
        case node_type::MINUS_NODE:
        case node_type::PLUS_NODE: {
            auto backup_previous_node = previous_node;
            previous_node = op_node->type;
            child_raw_ptr->accept(this);
            previous_node = backup_previous_node;
            break;
        }

        default: {
            auto backup_previous_node = previous_node;
            previous_node = op_node->type;
            op_node->children.front()->accept(this);
            previous_node = backup_previous_node;
            break;
        }
    }
    previous_uminus_node = backup_previous_uminus_node;
}

void uminus_distribute_and_reduce_visitor::visit(sentinel_node* op_node) {
	LOG("");
    for (shared_ptr<irtree_node>& child : op_node->children) {
        previous_node = op_node->type;
        previous_node_visiting_child = &child;
    	LOG("");
        child->accept(this);
    }
	LOG("");
}
