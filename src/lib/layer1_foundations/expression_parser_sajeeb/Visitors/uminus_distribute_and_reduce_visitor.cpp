/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.22.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/
    
#include "uminus_distribute_and_reduce_visitor.h"
#include "../IRTree/node.h"

#ifndef LOG
#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;
#endif


using node_type = irtree_node::node_type;


void uminus_distribute_and_reduce_visitor::visit(variable_node* num_node, 
                                                 irtree_node* parent_node, 
                                                 list<shared_ptr<irtree_node> >::iterator& link) {/* NO-OP */}
void uminus_distribute_and_reduce_visitor::visit(parameter_node* node, 
                                                 irtree_node* parent_node, 
                                                 list<shared_ptr<irtree_node> >::iterator& link) {/* NO-OP */}
void uminus_distribute_and_reduce_visitor::visit(number_node* op_node, 
                                                 irtree_node* parent_node, 
                                                 list<shared_ptr<irtree_node> >::iterator& link) {/* NO-OP */}
void uminus_distribute_and_reduce_visitor::visit(sentinel_node* op_node, 
                                                 irtree_node* parent_node, 
                                                 list<shared_ptr<irtree_node> >::iterator& link) {/* NO-OP */}


void uminus_distribute_and_reduce_visitor::visit(plus_node* op_node,
                                                 irtree_node* parent_node, 
                                                 list<shared_ptr<irtree_node> >::iterator& link) {

    if (parent_node->type == node_type::UNARY_NEGATE_NODE) {
        // bypass uminus node
        *link = static_cast<unary_negate_node*>(parent_node)->children.front();
        // distribute uminus to children
        for (shared_ptr<irtree_node>& child : op_node->children) {
            shared_ptr<non_terminal_node> uminus = make_shared<unary_negate_node>();
            uminus->children.emplace_back(child);
            child = uminus;
        }
    }

    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) 
        dispatcher::visit(*it, *this, op_node, it);

}

void uminus_distribute_and_reduce_visitor::visit(minus_node* op_node,
                                                 irtree_node* parent_node, 
                                                 list<shared_ptr<irtree_node> >::iterator& link) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) 
        dispatcher::visit(*it, *this, op_node, it);
}

void uminus_distribute_and_reduce_visitor::visit(multiply_node* op_node,
                                                 irtree_node* parent_node, 
                                                 list<shared_ptr<irtree_node> >::iterator& link) {

   int negative_children_count = 0;
    for (shared_ptr<irtree_node>& child : op_node->children) {
        negative_children_count += (child->type == node_type::UNARY_NEGATE_NODE);
    }

    // function to drop all the following uminus nodes and update the negative children counter
    auto drop_following_uminus_nodes = [&]()->void {
        if (negative_children_count > 0)
            for (shared_ptr<irtree_node>& child : op_node->children) { // drop proceeding uminus nodes
                if (child->type == node_type::UNARY_NEGATE_NODE) {
                    non_terminal_node* child_raw_ptr = static_cast<non_terminal_node*>(child.get());
                    shared_ptr<irtree_node> grandchild = child_raw_ptr->children.front();
                    child = grandchild;
                    negative_children_count -= 1;
                }
            }
    };

    // 
    if (parent_node->type == node_type::UNARY_NEGATE_NODE) {     
        auto prior_negative_children_count = negative_children_count;
        drop_following_uminus_nodes();
        if (prior_negative_children_count % 2 != 0) { // odd
            *link = static_cast<unary_negate_node*>(parent_node)->children.front();
        } else { // even
            *link = static_cast<unary_negate_node*>(parent_node)->children.front();
            shared_ptr<non_terminal_node> uminus = make_shared<unary_negate_node>();
            uminus->children.emplace_back(op_node->children.front());
            op_node->children.front() = uminus;
            negative_children_count += 1;
        }
    }
    if (negative_children_count > 0 && negative_children_count % 2 == 0) { // even number of proceeding negative nodes
        drop_following_uminus_nodes();
    }

    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) 
        dispatcher::visit(*it, *this, op_node, it);
}

void uminus_distribute_and_reduce_visitor::visit(exponent_node* op_node,
                                                 irtree_node* parent_node, 
                                                 list<shared_ptr<irtree_node> >::iterator& link) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) 
        dispatcher::visit(*it, *this, op_node, it);
}

void uminus_distribute_and_reduce_visitor::visit(unary_negate_node* op_node,
                                                 irtree_node* parent_node, 
                                                 list<shared_ptr<irtree_node> >::iterator& link) {
    non_terminal_node* child_raw_ptr = static_cast<non_terminal_node*>(op_node->children.front().get());
    switch (child_raw_ptr->type) {
        case node_type::UNARY_NEGATE_NODE: {
            shared_ptr<irtree_node>& grandchild = *link = child_raw_ptr->children.front();
            dispatcher::visit(grandchild, *this, parent_node, link);
            break;
        }

        default: {
            dispatcher::visit(op_node->children.front(), *this, op_node, link);
            break;
        }
    }
}

void uminus_distribute_and_reduce_visitor::visit(sentinel_node* op_node) {
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) 
        dispatcher::visit(*it, *this, op_node, it);
}
