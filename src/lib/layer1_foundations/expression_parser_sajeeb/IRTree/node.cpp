
/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.22.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/
    
#include "node.h"

irtree_node::~irtree_node() {}

non_terminal_node::~non_terminal_node() {}

terminal_node::~terminal_node() {}

plus_node::~plus_node() {}

minus_node::~minus_node() {}

multiply_node::~multiply_node() {}

exponent_node::~exponent_node() {}

unary_negate_node::~unary_negate_node() {}

variable_node::~variable_node() {}

parameter_node::~parameter_node() {}

number_node::~number_node() {}

sentinel_node::~sentinel_node() {}

std::ostream& operator<< (std::ostream& os, const irtree_node::node_type& obj) {
    switch (obj) {
        case irtree_node::node_type::SENTINEL_NODE: {
            os << "SENTINEL_NODE";
            break;
        }
        case irtree_node::node_type::PLUS_NODE: {
            os << "PLUS_NODE";
            break;
        }
        case irtree_node::node_type::EXPONENT_NODE: {
            os << "EXPONENT_NODE";
            break;
        }
        case irtree_node::node_type::MULTIPLY_NODE: {
            os << "MULTIPLY_NODE";
            break;
        }
        case irtree_node::node_type::MINUS_NODE: {
            os << "MINUS_NODE";
            break;
        }
        case irtree_node::node_type::UNARY_NEGATE_NODE: {
            os << "UNARY_NEGATE_NODE";
            break;
        }
        case irtree_node::node_type::VARIABLE_NODE: {
            os << "VARIABLE_NODE";
            break;
        }
        case irtree_node::node_type::NUMBER_NODE: {
            os << "NUMBER_NODE";
            break;
        }
        case irtree_node::node_type::PARAMETER_NODE: {
            os << "PARAMETER_NODE";
            break;
        }
        default:
            os << "INVALID";
    }
    return os;
}
