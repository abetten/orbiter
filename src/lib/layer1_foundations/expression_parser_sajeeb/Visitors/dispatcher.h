/**
* Author:    Sajeeb Roy Chowdhury
* Created:   12.13.2022
* Email:     sajeeb.roy.chow@gmail.com
* 
**/

/**
 * @file 
 * @brief 
*/


#include "../IRTree/node.h"
#include <memory>

#ifndef _DISPATCHER_H_
#define _DISPATCHER_H_

using std::shared_ptr;

//! auxiliary struct to realize the visitor programming paradigm for Sajeeb's abstract syntax trees

struct dispatcher {

    template <typename node_t, typename visitor_t, typename... visitor_args_t>
    static auto visit(node_t& node, visitor_t&& visitor, visitor_args_t&&... args) {
        switch (node->type) {
            case irtree_node::node_type::SENTINEL_NODE:
                return visitor.visit(static_cast<sentinel_node*>(node.get()),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::NUMBER_NODE:
                return visitor.visit(static_cast<number_node*>(node.get()),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::PARAMETER_NODE:
                return visitor.visit(static_cast<parameter_node*>(node.get()),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::VARIABLE_NODE:
                return visitor.visit(static_cast<variable_node*>(node.get()),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::UNARY_NEGATE_NODE:
                return visitor.visit(static_cast<unary_negate_node*>(node.get()),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::EXPONENT_NODE:
                return visitor.visit(static_cast<exponent_node*>(node.get()),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::MULTIPLY_NODE:
                return visitor.visit(static_cast<multiply_node*>(node.get()),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::MINUS_NODE:
                return visitor.visit(static_cast<minus_node*>(node.get()),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::PLUS_NODE:
                return visitor.visit(static_cast<plus_node*>(node.get()),
                                                std::forward<decltype(args)>(args)...);
            default:
                break;
        };
    }

};

#endif /* _DISPATCHER_H_ */
