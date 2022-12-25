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

#include "../IRTree/node_forward_declaration.h"

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

    std::variant<plus_node, minus_node, multiply_node, exponent_node, unary_negate_node, variable_node, parameter_node, 
    number_node, sentinel_node> nodes;

    template<typename visitor_t, typename node_t, class... Args_t>
    void visit(shared_ptr<visitor_t>& visitor, shared_ptr<node_t>& node, Args_t... args) {
        cout << "visit: " << node->type << endl;
    }

    template<typename visitor_t, typename node_t, class... Args_t>
    void visit(const shared_ptr<visitor_t>& visitor, shared_ptr<node_t>& node, Args_t... args) {
        std::visit(visitor.get(), node.get(), args...);
    }

}

#endif /* _DISPATCHER_H_ */