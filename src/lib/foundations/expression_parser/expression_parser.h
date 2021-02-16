/*
 * parser.h
 *
 *  Created on: Feb 14, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_EXPRESSION_PARSER_EXPRESSION_PARSER_H_
#define SRC_LIB_FOUNDATIONS_EXPRESSION_PARSER_EXPRESSION_PARSER_H_



namespace orbiter {
namespace foundations {

// #############################################################################
// expression_parser.cpp
// #############################################################################



//! class to parse expressions




class expression_parser {



  private:

	lexer *Lexer;

  public:

	  // symbol table - can be accessed directly (eg. to copy a batch in)
	  std::map<std::string, double> symbols_;

	  syntax_tree *Tree;

	  expression_parser();
	  ~expression_parser();


  // access symbols with operator []
  double & operator[] (std::string & key) { return symbols_ [key]; }



  syntax_tree_node *Primary (int verbose_level,
		  int &f_single_literal, std::string &single_literal, int &f_has_seen_minus,
		  const bool get);
  syntax_tree_node *Term (int verbose_level, const bool get);
  syntax_tree_node *AddSubtract (int verbose_level, const bool get);
  syntax_tree_node *Comparison (int verbose_level, const bool get);
  syntax_tree_node *Expression (int verbose_level, const bool get);
  syntax_tree_node *CommaList (int verbose_level, const bool get);
  void parse(syntax_tree *tree, std::string & program, int verbose_level);

};


// #############################################################################
// lexer.cpp
// #############################################################################

//! lexical analysis of expressions



class lexer {
public:
	  std::string program_;

	  const char * pWord_;
	  const char * pWordStart_;
	  // last token parsed
	  TokenType type_;
	  std::string word_;
	  double value_;
	  syntax_tree_node_terminal *T;

	  lexer();
	  void print_token(std::ostream &ost, TokenType t);
	  void token_as_string(std::string &s, TokenType t);
	  TokenType GetToken (int verbose_level, const bool ignoreSign = false);
	  void create_text_token(std::string &txt);
	  void create_double_token(double dbl);
	  void CheckToken (TokenType wanted);

};

// #############################################################################
// syntax_tree_node_terminal.cpp
// #############################################################################

//! terminal note in the syntax tree of an expression



class syntax_tree_node_terminal {
public:
	int f_int;
	int f_double;
	int f_text;
	int value_int;
	double value_double;
	std::string value_text;

	syntax_tree_node_terminal();
	void print(std::ostream &ost);
	void print_expression(std::ostream &ost);
	int evaluate(std::map<std::string, std::string> &symbol_table,
			finite_field *F, int verbose_level);

};



// #############################################################################
// syntax_tree_node.cpp
// #############################################################################

#define MAX_NODES_SYNTAX_TREE 1000


//! interior node in a syntax tree


class syntax_tree_node {
public:
	syntax_tree *Tree;
	int idx;

	int f_terminal;
	syntax_tree_node_terminal *T;

	enum syntax_tree_node_operation_type type;

	int nb_nodes;
	syntax_tree_node *Nodes[MAX_NODES_SYNTAX_TREE];

	int f_has_monomial;
	int *monomial;

	int f_has_minus;

	syntax_tree_node();
	~syntax_tree_node();
	void null();
	void split_by_monomials(homogeneous_polynomial_domain *Poly,
			syntax_tree_node **Subtrees, int verbose_level);
	int is_homogeneous(int &degree);
	void print(std::ostream &ost);
	int evaluate(std::map<std::string, std::string> &symbol_table,
			finite_field *F, int verbose_level);
	void print_expression(std::ostream &ost);
	void push_a_minus_sign();
	void print_without_recursion(std::ostream &ost);
	void export_graphviz(std::string &name, std::ostream &ost);
	void export_graphviz_recursion(std::ostream &ost);
};


// #############################################################################
// syntax_tree.cpp
// #############################################################################

//! the syntax tree of an expression



class syntax_tree {
public:
	int f_has_managed_variables;
	std::vector<std::string> managed_variables;

	syntax_tree_node *Root;

	syntax_tree();
	void print(std::ostream &ost);
	void print_monomial(std::ostream &ost, int *monomial);
	int identify_single_literal(std::string &single_literal);
	int is_homogeneous(int &degree);
	void split_by_monomials(homogeneous_polynomial_domain *Poly,
			syntax_tree_node **&Subtrees, int verbose_level);

};







}}


#endif /* SRC_LIB_FOUNDATIONS_EXPRESSION_PARSER_EXPRESSION_PARSER_H_ */
