/**
* Author:    Sajeeb Roy Chowdhury
* Created:   12/20/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include <string>
#include <memory>

#include "IRTree/node_forward_declaration.h"
#include "Visitors/exponent_vector_visitor.h"

using std::shared_ptr;


#ifndef _PARSER_
#define _PARSER_

namespace parser {

    shared_ptr<irtree_node> parse_expression(std::string& exp, managed_variables_index_table managed_variables_table);

}

#endif /* _PARSER_ */