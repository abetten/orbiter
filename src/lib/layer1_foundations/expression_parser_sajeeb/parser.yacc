%parse-param {shared_ptr<irtree_node>& root}
%parse-param {managed_variables_index_table& managed_variables_map}

%code requires{
	#include <cstring>	
	#include <memory>
    #include "Visitors/exponent_vector_visitor.h"
    #include <string>
	#include "IRTree/node.h"
}

%code provides {
	// function prototypes, we need the yylex return prototype so C++ won't complain
	extern "C" int yylex(void);
	extern "C" void yyerror(shared_ptr<irtree_node>& root,
                            managed_variables_index_table& managed_variables_map,
                            const char* s);
}

%start GOAL

%union {
	int num;
	char* id;	// for identifier nodes
    irtree_node* expression_node;
    sentinel_node* ast_root;
}

%define parse.error verbose

%token <num> NUMBER
%token <id> IDENTIFIER
%token NEWLINE
%left PLUS MINUS 
%left TIMES
%nonassoc UMINUS
%right EXPONENT
%left LPARENTHESIS 
%left RPARENTHESIS
%type <expression_node> EXPRESSION 
%type <ast_root> GOAL

%%

GOAL: EXPRESSION {
            auto* node = new sentinel_node();
            node->add_child($1);

            root = std::shared_ptr<irtree_node>(node);
		}
;

EXPRESSION: MINUS EXPRESSION %prec UMINUS {
                auto* node = new unary_negate_node();
                node->add_child($2);
                $$ = node;
			}
			| EXPRESSION PLUS EXPRESSION {
				auto* node = new plus_node();
                node->add_child($1);
                node->add_child($3);
                $$ = node;
			}
			| EXPRESSION MINUS EXPRESSION {
                auto* node = new minus_node();
                node->add_child($1);
                node->add_child($3);
                $$ = node;
			}
			| EXPRESSION TIMES EXPRESSION {
                auto* node = new multiply_node();
                node->add_child($1);
                node->add_child($3);
                $$ = node;
			}
			| EXPRESSION EXPONENT EXPRESSION {
                auto* node = new exponent_node();
                node->add_child($1);
                node->add_child($3);
                $$ = node;
			}
			| LPARENTHESIS EXPRESSION RPARENTHESIS {
				$$ = $2;
			}
			| NUMBER {
				auto* node = new number_node($1);
                $$ = node;
			}
			| IDENTIFIER {
                irtree_node* node;
                if (managed_variables_map.find(string$1) != managed_variables_map.end())
                    node = new variable_node($1);
                else
                    node = new parameter_node($1);
                $$ = node;
			}
;

%%

void yyerror (shared_ptr<irtree_node>& root,
              managed_variables_index_table& managed_variables_map,
              const char* s) {
	fprintf (stderr, "%s\n", s);
}