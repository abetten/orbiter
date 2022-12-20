/*
 * parser.h
 *
 *  Created on: Feb 14, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_EXPRESSION_PARSER_EXPRESSION_PARSER_H_
#define SRC_LIB_FOUNDATIONS_EXPRESSION_PARSER_EXPRESSION_PARSER_H_



namespace orbiter {
namespace layer1_foundations {
namespace expression_parser {


// #############################################################################
// expression_parser_domain.cpp
// #############################################################################



//! a domain for things related to expression parsing

class expression_parser_domain {
public:
	expression_parser_domain();
	~expression_parser_domain();
	void parse_and_evaluate(
			field_theory::finite_field *F,
			std::string &name_of_formula,
			std::string &formula_text,
			std::string &managed_variables,
			int f_evaluate,
			std::string &parameters,
			int verbose_level);
	void evaluate(
			field_theory::finite_field *Fq,
			std::string &formula_label,
			std::string &parameters,
			int verbose_level);
	int evaluate_formula(
			formula *F,
			field_theory::finite_field *Fq,
			std::string &parameters,
			int verbose_level);
	void evaluate_managed_formula(
			formula *F,
			field_theory::finite_field *Fq,
			std::string &parameters,
			int *&Values, int &nb_monomials,
			int verbose_level);

};





// #############################################################################
// expression_parser.cpp
// #############################################################################



//! class to parse expressions




class expression_parser {



	private:

		lexer *Lexer;

	public:

		// symbol table
		std::map<std::string, double> symbols;

	syntax_tree *Tree;

	expression_parser();
	~expression_parser();


	// access symbols with operator []
	double & operator[] (std::string & key) { return symbols [key]; }



	syntax_tree_node *Primary(int verbose_level,
		  int &f_single_literal, std::string &single_literal,
		  int &f_has_seen_minus, const bool get);
	syntax_tree_node *Term(int verbose_level, const bool get);
	syntax_tree_node *AddSubtract(int verbose_level, const bool get);
	syntax_tree_node *Comparison(int verbose_level, const bool get);
	syntax_tree_node *Expression(int verbose_level, const bool get);
	syntax_tree_node *CommaList(int verbose_level, const bool get);
	void parse(syntax_tree *tree, std::string & program, int verbose_level);

};


// #############################################################################
// formula_activity_description.cpp
// #############################################################################


//! description of an activity involving a formula

class formula_activity_description {
public:

	int f_export;

	int f_evaluate;
	std::string evaluate_finite_field_label;
	std::string evaluate_assignment;

	int f_print_over_Fq;
	std::string print_over_Fq_field_label;

	int f_sweep;
	std::string sweep_field_label;
	std::string sweep_variables;

	int f_sweep_affine;
	std::string sweep_affine_field_label;
	std::string sweep_affine_variables;



	formula_activity_description();
	~formula_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// formula_activity.cpp
// #############################################################################


//! an activity involving a formula

class formula_activity {
public:


	formula_activity_description *Descr;
	formula *f;

	formula_activity();
	~formula_activity();
	void init(formula_activity_description *Descr,
			formula *f,
			int verbose_level);
	void perform_activity(int verbose_level);
	void do_sweep(int f_affine,
			formula *f,
			field_theory::finite_field *F, std::string &sweep_variables,
			int verbose_level);

};


// #############################################################################
// formula.cpp
// #############################################################################



//! front-end to expression




class formula {


public:

	std::string name_of_formula;
	std::string name_of_formula_latex;
	std::string managed_variables;
	std::string formula_text;
	syntax_tree *tree;


	int nb_managed_vars;

	int f_is_homogeneous;
	int degree;


	formula();
	~formula();
	void print();
	void init(std::string &label, std::string &label_tex,
			std::string &managed_variables, std::string &formula_text,
			int verbose_level);
#if 0
	void init_Sajeeb(std::string &label, std::string &label_tex,
			std::string &managed_variables, std::string &formula_text,
			int verbose_level);
#endif
	int is_homogeneous(int &degree, int verbose_level);
	void get_subtrees(ring_theory::homogeneous_polynomial_domain *Poly,
			syntax_tree_node **&Subtrees, int &nb_monomials,
			int verbose_level);
	void evaluate(ring_theory::homogeneous_polynomial_domain *Poly,
			syntax_tree_node **Subtrees, std::string &evaluate_text, int *Values,
			int verbose_level);
	void print(std::ostream &ost);
	void print_easy(field_theory::finite_field *F, std::ostream &ost);

};


// #############################################################################
// lexer.cpp
// #############################################################################

//! lexical analysis of expressions



class lexer {
public:
	  std::string program;

	  const char * pWord;
	  const char * pWordStart;
	  // last token parsed
	  TokenType type;
	  std::string word;
	  double value;
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

//! terminal node in the syntax tree of an expression



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
	void print_easy(std::ostream &ost);
	void print_expression(std::ostream &ost);
	void print_graphviz(std::ostream &ost);
	int evaluate(
			std::map<std::string, std::string> &symbol_table,
			field_theory::finite_field *F,
			int verbose_level);

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

	//! multiplication or addition
	enum syntax_tree_node_operation_type type;

	// ! if we are not a terminal node, we can have any number of nodes
	int nb_nodes;
	syntax_tree_node *Nodes[MAX_NODES_SYNTAX_TREE];

	int f_has_monomial; // only for multiplication nodes
	int *monomial;

	int f_has_minus;

	syntax_tree_node();
	~syntax_tree_node();
	void null();
	void split_by_monomials(
			ring_theory::homogeneous_polynomial_domain *Poly,
			syntax_tree_node **Subtrees, int verbose_level);
	int is_homogeneous(int &degree, int verbose_level);
	void print(std::ostream &ost);
	void print_easy(std::ostream &ost);
	void print_easy_without_monomial(std::ostream &ost);
	int is_mult();
	int is_add();
	int evaluate(std::map<std::string, std::string> &symbol_table,
			field_theory::finite_field *F, int verbose_level);
	void print_expression(std::ostream &ost);
	void push_a_minus_sign();
	void print_without_recursion(std::ostream &ost);
	void export_graphviz(std::string &name, std::ostream &ost);
	void export_graphviz_recursion(std::ostream &ost);
};


// #############################################################################
// syntax_tree.cpp
// #############################################################################

//! the syntax tree of an expression, possibly with managed variables



class syntax_tree {
public:
	int f_has_managed_variables;
	std::vector<std::string> managed_variables;

	syntax_tree_node *Root;

	syntax_tree();
	void print(std::ostream &ost);
	void print_easy(std::ostream &ost);
	void print_monomial(std::ostream &ost, int *monomial);
	int identify_single_literal(std::string &single_literal);
	int is_homogeneous(int &degree, int verbose_level);
	void split_by_monomials(
			ring_theory::homogeneous_polynomial_domain *Poly,
			syntax_tree_node **&Subtrees, int verbose_level);

};







}}}



#endif /* SRC_LIB_FOUNDATIONS_EXPRESSION_PARSER_EXPRESSION_PARSER_H_ */
