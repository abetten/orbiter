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
#include <unordered_map>
#include <iostream>
#include <variant>

#ifndef _DISPATCHER_H_
#define _DISPATCHER_H_

using std::unordered_map;
using std::cout;
using std::endl;
using std::shared_ptr;

namespace dispatcher {

    template <typename node_t, typename visitor_t, typename... visitor_args_t>
    auto visit(shared_ptr<node_t>& node, visitor_t& visitor, visitor_args_t... args) {
        switch (node->type) {
            case irtree_node::node_type::SENTINEL_NODE:
                return visitor.template visit<sentinel_node>(std::forward<decltype(node)>(node),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::NUMBER_NODE:
                return visitor.template visit<number_node>(std::forward<decltype(node)>(node),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::PARAMETER_NODE:
                return visitor.template visit<parameter_node>(std::forward<decltype(node)>(node),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::VARIABLE_NODE:
                return visitor.template visit<variable_node>(std::forward<decltype(node)>(node),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::UNARY_NEGATE_NODE:
                return visitor.template visit<unary_negate_node>(std::forward<decltype(node)>(node),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::EXPONENT_NODE:
                return visitor.template visit<exponent_node>(std::forward<decltype(node)>(node),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::MULTIPLY_NODE:
                return visitor.template visit<multiply_node>(std::forward<decltype(node)>(node),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::MINUS_NODE:
                return visitor.template visit<minus_node>(std::forward<decltype(node)>(node),
                                                std::forward<decltype(args)>(args)...);
            case irtree_node::node_type::PLUS_NODE:
                return visitor.template visit<plus_node>(std::forward<decltype(node)>(node),
                                                std::forward<decltype(args)>(args)...);
            default:
                break;
        };
    }

}

#endif /* _DISPATCHER_H_ */