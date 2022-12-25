#include "parser.h"

#include "parser.tab.hpp"
#include "lexer.yy.h"


shared_ptr<irtree_node> parser::parse_expression(std::string& exp, managed_variables_index_table managed_variables_table) {
    shared_ptr<irtree_node> ir_tree_root;
    YY_BUFFER_STATE buffer = yy_scan_string( exp.c_str() );
    yy_switch_to_buffer(buffer);
    int result = yyparse(ir_tree_root, managed_variables_table);
    yy_delete_buffer(buffer);
    yylex_destroy();
    return ir_tree_root;
}