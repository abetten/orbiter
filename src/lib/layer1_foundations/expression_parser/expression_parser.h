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
			field_theory::finite_field *Fq,
			std::string &name_of_formula,
			std::string &formula_text,
			std::string &managed_variables,
			int f_evaluate,
			std::string &parameters,
			int verbose_level);
	// uses the old parser
	void evaluate(
			std::string &formula_label,
			std::string &parameters,
			int verbose_level);
	// ToDo: use symbolic object instead
	int evaluate_formula(
			formula *F,
			std::string &parameters,
			int verbose_level);
	// creates a homogeneous_polynomial_domain object
	// if the formula is homogeneous
	void evaluate_managed_formula(
			formula *F,
			std::string &parameters,
			int *&Values, int &nb_monomials,
			int verbose_level);
	// creates a homogeneous_polynomial_domain object

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
	//std::string evaluate_finite_field_label;
	std::string evaluate_assignment;

	int f_print;
	//std::string print_over_Fq_field_label;

	int f_sweep;
	std::string sweep_variables;

	int f_sweep_affine;
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


//! an activity involving a symbolic expression

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
			std::string &sweep_variables,
			int verbose_level);

};


// #############################################################################
// formula_vector.cpp
// #############################################################################



//! matrices and vectors of symbolic objects


class formula_vector {
public:

	std::string label_txt;
	std::string label_tex;

	int f_has_managed_variables;
	std::string managed_variables_text;

	formula *V;
	int len;

	int f_matrix;
	int nb_rows;
	int nb_cols;


	formula_vector();
	~formula_vector();
	void init_from_text(
			std::string &label_txt, std::string &label_tex,
			std::string &text,
			field_theory::finite_field *Fq,
			int f_managed_variables,
			std::string &managed_variables_text,
			int f_matrix, int nb_rows,
			int verbose_level);
	void init_and_allocate(
			std::string &label_txt, std::string &label_tex,
			int f_has_managed_variables,
			std::string managed_variables_text,
			int len, int verbose_level);
	int is_integer_matrix();
	void get_integer_matrix(int *&M, int verbose_level);
	void get_string_representation_Sajeeb(std::vector<std::string> &S);
	void get_string_representation_formula(
			std::vector<std::string> &S, int verbose_level);
	void print_Sajeeb(std::ostream &ost);
	void print_formula_size(
			std::ostream &ost, int verbose_level);
	void print_formula(
			std::ostream &ost, int verbose_level);
	void print_matrix(
			std::vector<std::string> &S, std::ostream &ost);
	void print_vector(
			std::vector<std::string> &S, std::ostream &ost);
	void print_vector_latex(
			std::vector<std::string> &S, std::ostream &ost);
	void print_latex(std::ostream &ost, std::string &label);
	void make_A_minus_lambda_Identity(
			formula_vector *A,
			field_theory::finite_field *Fq,
			std::string &variable,
			std::string &label_txt,
			std::string &label_tex,
			int f_managed_variables,
			std::string &managed_variables_text,
			int verbose_level);
	void substitute(formula_vector *Source,
			formula_vector *Target,
			std::string &substitution_variables,
			std::string &label_txt,
			std::string &label_tex,
			std::string &managed_variables,
			int verbose_level);
	void simplify(
			formula_vector *A,
			field_theory::finite_field *Fq,
			std::string &label_txt,
			std::string &label_tex,
			int verbose_level);
	void expand(
			formula_vector *A,
			field_theory::finite_field *Fq,
			std::string &label_txt,
			std::string &label_tex,
			std::string &managed_variables,
			int f_write_trees,
			int verbose_level);
	void characteristic_polynomial(
			formula_vector *A,
			field_theory::finite_field *Fq,
			std::string &variable,
			std::string &label_txt,
			std::string &label_tex,
			std::string &managed_variables_text,
			int verbose_level);
	void determinant(
			formula_vector *A,
			field_theory::finite_field *Fq,
			std::string &label_txt,
			std::string &label_tex,
			std::string &managed_variables_text,
			int verbose_level);
	void right_nullspace(
			formula_vector *A,
			field_theory::finite_field *Fq,
			std::string &label_txt,
			std::string &label_tex,
			std::string &managed_variables,
			int verbose_level);
	void matrix_minor(
			formula_vector *A,
			field_theory::finite_field *Fq,
			int i, int j,
			std::string &label_txt,
			std::string &label_tex,
			std::string &managed_variables_text,
			int verbose_level);
	void symbolic_nullspace(
			formula_vector *A,
			field_theory::finite_field *Fq,
			std::string &label_txt,
			std::string &label_tex,
			std::string &managed_variables,
			int verbose_level);
	void multiply_2by2_from_the_left(
			formula_vector *M,
			formula_vector *A2,
			int i, int j,
			field_theory::finite_field *Fq,
			std::string &label_txt,
			std::string &label_tex,
			std::string &managed_variables_text,
			int verbose_level);
	void latex_tree(int verbose_level);
	void export_tree(int verbose_level);
	void collect_variables(int verbose_level);
	void print_variables(std::ostream &ost);

};



// #############################################################################
// formula.cpp
// #############################################################################



//! a single symbolic expression represented as a tree, consisting of multiplication nodes, addition nodes, and terminal nodes




class formula {


public:

	std::string name_of_formula;
	std::string name_of_formula_latex;
	std::string managed_variables;
	std::string formula_text;
	field_theory::finite_field *Fq;
	syntax_tree *tree;


	int nb_managed_vars;

	int f_is_homogeneous;
	int degree;

	int f_Sajeeb;
	l1_interfaces::expression_parser_sajeeb *Expression_parser_sajeeb;


	formula();
	~formula();
	std::string string_representation(
			int f_latex, int verbose_level);
	std::string string_representation_Sajeeb();
	std::string string_representation_formula(
			int f_latex, int verbose_level);
	void print(std::ostream &ost);
	void init_empty_plus_node(
			std::string &label, std::string &label_tex,
			std::string &managed_variables_text,
			field_theory::finite_field *Fq,
			int verbose_level);
	void init_formula(
			std::string &label, std::string &label_tex,
			std::string &managed_variables, std::string &formula_text,
			field_theory::finite_field *Fq,
			int verbose_level);
	// using the old parser
	void init_formula_from_tree(
			std::string &label, std::string &label_tex,
			field_theory::finite_field *Fq,
			syntax_tree *Tree,
			int verbose_level);
	void init_formula_int(
			std::string &label, std::string &label_tex,
			int value,
			field_theory::finite_field *Fq,
			std::string &managed_variables,
			int verbose_level);
	void init_formula_monopoly(
			std::string &label, std::string &label_tex,
			field_theory::finite_field *Fq,
			std::string &managed_variables,
			std::string &variable,
			int *coeffs, int nb_coeffs,
			int verbose_level);
	void init_formula_Sajeeb(
			std::string &label, std::string &label_tex,
			int f_managed_variables,
			std::string &managed_variables,
			std::string &formula_text,
			field_theory::finite_field *Fq,
			int verbose_level);
	int is_homogeneous(
			int &degree, int verbose_level);
	void get_subtrees(
			ring_theory::homogeneous_polynomial_domain *Poly,
			syntax_tree_node **&Subtrees, int &nb_monomials,
			int verbose_level);
	void evaluate_with_symbol_table(
			std::map<std::string, std::string> &symbol_table,
			int *Values,
			int verbose_level);
	void evaluate(
			ring_theory::homogeneous_polynomial_domain *Poly,
			syntax_tree_node **Subtrees,
			std::string &evaluate_text, int *Values,
			int verbose_level);
	void export_graphviz(
			std::string &name);
	void print_easy(
			std::ostream &ost);
	void substitute(
			std::vector<std::string> &variables,
			std::string &managed_variables_text,
			formula **S,
			formula *output,
			int verbose_level);
	void copy_to(
			formula *output,
			int verbose_level);
	void make_linear_combination(
			formula *input_1a,
			formula *input_1b,
			formula *input_2a,
			formula *input_2b,
			field_theory::finite_field *Fq,
			std::string &label_txt,
			std::string &label_tex,
			std::string &managed_variables_text,
			int verbose_level);
	void simplify(
			int verbose_level);
	void expand_in_place(int f_write_trees,
			int verbose_level);
	int highest_order_term(
			std::string &variable, int verbose_level);
	void get_monopoly(
			std::string &variable, int *&coeff, int &nb_coeff,
			int verbose_level);
	void latex_tree(
			std::string &label, int verbose_level);
	void export_tree(
			std::string &label, int verbose_level);
	void latex_tree_split(
			std::string &label, int split_level, int split_mod,
			int verbose_level);
	void collect_variables(int verbose_level);
	void collect_monomial_terms(
			data_structures::int_matrix *&I, int *&Coeff,
			int verbose_level);
	void collect_coefficients_of_equation(
			ring_theory::homogeneous_polynomial_domain *Poly,
			int *&coeffs, int &nb_coeffs,
			int verbose_level);

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
	~syntax_tree_node_terminal();
	void print_to_vector(
			std::vector<std::string> &rep, int verbose_level);
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



//! interior node in a syntax tree


class syntax_tree_node {
public:
	syntax_tree *Tree;
	int idx;

	int f_has_exponent;
	int exponent;

	int f_terminal;
	syntax_tree_node_terminal *T;

	//! multiplication or addition
	enum syntax_tree_node_operation_type type;

	//! if we are not a terminal node, we can have any number of nodes
	int nb_nodes;
	int nb_nodes_allocated;
	syntax_tree_node **Nodes;

	int f_has_monomial; // only for multiplication nodes
	int *monomial;

	int f_has_minus;

	syntax_tree_node();
	~syntax_tree_node();
	void null();
	void add_numerical_factor(
			int value, int verbose_level);
	void add_numerical_summand(
			int value, int verbose_level);
	int text_value_match(std::string &factor);
	void add_factor(
			std::string &factor, int exponent,
			int verbose_level);
	void add_summand(
			std::string &summand, int verbose_level);
	void init_terminal_node_int(
			syntax_tree *Tree,
			int value,
			int verbose_level);
	void init_monopoly(
			syntax_tree *Tree,
			std::string &variable,
			int *coeffs, int nb_coeffs,
			int verbose_level);
	void init_terminal_node_text(
			syntax_tree *Tree,
			std::string &value_text,
			int verbose_level);
	void init_terminal_node_text_with_exponent(
			syntax_tree *Tree,
			std::string &value_text,
			int exp,
			int verbose_level);
	void add_empty_plus_node_with_exponent(
			syntax_tree *Tree,
			int exponent, int verbose_level);
	void init_empty_plus_node_with_exponent(
			syntax_tree *Tree,
			int exponent,
			int verbose_level);
	void add_empty_multiplication_node(
			syntax_tree *Tree,
			int verbose_level);
	void init_empty_multiplication_node(
			syntax_tree *Tree,
			int verbose_level);
	void add_empty_node(
			syntax_tree *Tree,
			int verbose_level);
	void init_empty_node(
			syntax_tree *Tree,
			int verbose_level);
	void split_by_monomials(
			ring_theory::homogeneous_polynomial_domain *Poly,
			syntax_tree_node **Subtrees,
			int verbose_level);
	int is_homogeneous(
			int &degree, int verbose_level);
	int is_mult_node();
	int is_add_node();
	int is_int_node();
	int is_text_node();
	int is_monomial();
	int is_this_variable(
			std::string &variable);
	int highest_order_term(
			std::string &variable, int verbose_level);
	void get_monopoly(
			std::string &variable,
			std::vector<int> &Coeff, std::vector<int> &Exp,
			int verbose_level);
	void get_exponent_and_coefficient_of_variable(
			std::string &variable, int &coeff, int &exp,
			int verbose_level);
	int exponent_of_variable(
			std::string &variable, int verbose_level);
	int exponent_of_variable_destructive(
			std::string &variable);
	int get_exponent();
	int evaluate(
			std::map<std::string, std::string> &symbol_table,
			int verbose_level);
	void push_a_minus_sign();
	void copy_to(
			syntax_tree *Output_tree,
			syntax_tree_node *Output_node,
			int verbose_level);
	void substitute(
			std::vector<std::string> &variables,
			formula *Target,
			formula **Substitutions,
			syntax_tree *Input_tree,
			syntax_tree *Output_tree,
			syntax_tree_node *Output_node,
			int verbose_level);
	void simplify(
			int verbose_level);
	void expand_in_place(
			int verbose_level);
	void expand_in_place_handle_multiplication_node(
			int verbose_level);
	void expand_in_place_handle_exponents(
			int verbose_level);
	void simplify_exponents(int verbose_level);
	void sort_terms(int verbose_level);
	void collect_like_terms(int verbose_level);
	void collect_like_terms_addition(
			int verbose_level);
	void collect_monomial_terms(
			data_structures::int_matrix *&I, int *&Coeff,
			int verbose_level);
	void collect_terms_and_coefficients(
			data_structures::int_matrix *&I, int *&Coeff,
			int verbose_level);
	void simplify_constants(int verbose_level);
	void flatten(
			int verbose_level);
	void flatten_with_depth(int depth,
			int verbose_level);
	void flatten_post_process(int depth,
			int verbose_level);
	void flatten_at(int i, int depth, int verbose_level);
	void delete_all_but_one_child(
			int i, int verbose_level);
	void delete_one_child(
			int i, int verbose_level);
	int is_constant_one(int verbose_level);
	int is_constant_zero(int verbose_level);
	void collect_variables(int verbose_level);
	int terminal_node_get_variable_index();
	void count_nodes(
			int &nb_add, int &nb_mult, int &nb_int,
			int &nb_text, int &max_degree);
	void reallocate(
			int nb_nodes_needed, int verbose_level);
	void append_node(
			syntax_tree_node *child, int verbose_level);
	void insert_nodes_at(
			int idx, int nb_to_insert, int verbose_level);
	int needs_to_be_expanded();
	void make_determinant(
			syntax_tree *Output_tree,
			field_theory::finite_field *Fq,
			formula *V_in,
			int n,
			int verbose_level);
	void make_linear_combination(
			syntax_tree *Output_tree,
			syntax_tree_node *Node1a,
			syntax_tree_node *Node1b,
			syntax_tree_node *Node2a,
			syntax_tree_node *Node2b,
			int verbose_level);
	// we assume that the node is an empty plus node


	// syntax_tree_node_io.cpp:
	void print_subtree_to_vector(
			std::vector<std::string> &rep,
			int f_latex,
			int verbose_level);
	void print_subtree(
			std::ostream &ost);
	void print_subtree_easy(
			std::ostream &ost);
	void print_subtree_easy_no_lf(
			std::ostream &ost);
	void print_subtree_easy_without_monomial(
			std::ostream &ost);
	void print_node_type(
			std::ostream &ost);
	void print_expression(
			std::ostream &ost);
	void print_without_recursion(
			std::ostream &ost);
	void export_graphviz(
			std::string &name, std::ostream &ost);
	void export_graphviz_recursion(
			std::ostream &ost);
	void display_children_by_type();


};


// #############################################################################
// syntax_tree.cpp
// #############################################################################

//! the syntax tree of an expression, possibly with managed variables



class syntax_tree {
public:
	int f_has_managed_variables;
	std::string managed_variables_text;
	std::vector<std::string> managed_variables;

	field_theory::finite_field *Fq;

	syntax_tree_node *Root;

	std::vector<std::string> variables;

	syntax_tree();
	~syntax_tree();

	void init(
			field_theory::finite_field *Fq,
			int f_managed_variables, std::string &managed_variables_text,
			int verbose_level);
	void init_root_node(int verbose_level);
	void init_int(
			field_theory::finite_field *Fq, int value,
			int verbose_level);
	void init_monopoly(
			field_theory::finite_field *Fq,
			std::string &variable,
			int *coeffs, int nb_coeffs,
			int verbose_level);
	void print_to_vector(
			std::vector<std::string> &rep, int f_latex,
			int verbose_level);
	void count_nodes(
			int &nb_add, int &nb_mult, int &nb_int,
			int &nb_text, int &max_degree);
	int nb_nodes_total();
	void print(std::ostream &ost);
	void print_easy(std::ostream &ost);
	void print_monomial(
			std::ostream &ost, int *monomial);
	void export_graphviz(
			std::string &name, std::ostream &ost);
	int identify_single_literal(
			std::string &single_literal);
	int is_homogeneous(
			int &degree, int verbose_level);
	void split_by_monomials(
			ring_theory::homogeneous_polynomial_domain *Poly,
			syntax_tree_node **&Subtrees, int verbose_level);
	void substitute(
			std::vector<std::string> &variables,
			formula *Target,
			formula **Substitutions,
			syntax_tree *Output_tree,
			syntax_tree_node *Output_root_node,
			int verbose_level);
	void copy_to(
			syntax_tree *Output_tree,
			syntax_tree_node *Output_root_node,
			int verbose_level);
	void simplify(
			int verbose_level);
	void expand_in_place(
			int verbose_level);
	int highest_order_term(
			std::string &variable,
			int verbose_level);
	void get_monopoly(
			std::string &variable, int *&coeff, int &nb_coeff,
			int verbose_level);
	void multiply_by_minus_one(
			field_theory::finite_field *Fq,
			int verbose_level);
	void make_determinant(
			field_theory::finite_field *Fq,
			formula *V_in,
			int n,
			int verbose_level);
	int compare_nodes(
			syntax_tree_node *Node1,
			syntax_tree_node *Node2,
			int verbose_level);
	void make_linear_combination(
			syntax_tree_node *Node1a,
			syntax_tree_node *Node1b,
			syntax_tree_node *Node2a,
			syntax_tree_node *Node2b,
			int verbose_level);
	// Creates Output_node = Node1a * Node1b + Node2a * Node2b
	// All input nodes are copied.
	void latex_tree(
			std::string &name, int verbose_level);
	void export_tree(
			std::string &name, int verbose_level);
	void latex_tree_split(
			std::string &name, int split_level, int split_mod,
			int verbose_level);
	void collect_variables(
			int verbose_level);
	void print_variables(
			std::ostream &ost,
			int verbose_level);
	void print_variables_in_line(
			std::ostream &ost);
	int find_variable(
			std::string &var,
			int verbose_level);
	int find_managed_variable(
			std::string &var,
			int verbose_level);
	void add_variable(std::string &var);
	int get_number_of_variables();
	std::string &get_variable_name(int index);
	int needs_to_be_expanded();
	long int evaluate(
			std::map<std::string, std::string> &symbol_table,
			int verbose_level);

};



class syntax_tree_latex {
public:
	std::ostream* output_stream;
	std::string indentation;
	std::string delimiter;
	int f_split;
	int split_level;
	int split_r;
	int split_mod;

	void latex_tree(
			std::string &name, syntax_tree_node *Root,
			int verbose_level);
	void export_tree(
    		std::string &name, syntax_tree_node *Root,
    		int verbose_level);
	void latex_tree_split(
			std::string &name, syntax_tree_node *Root,
			int split_level, int split_mod,
			int verbose_level);
	void add_prologue();
	void add_epilogue();
	void add_indentation();
	void remove_indentation();
	void latex_tree_recursion(
			syntax_tree_node *node, int depth,
			int verbose_level);
	void export_tree_recursion(
			syntax_tree_node *node, int depth,
			std::vector<int> &path,
			int verbose_level);

};




}}}



#endif /* SRC_LIB_FOUNDATIONS_EXPRESSION_PARSER_EXPRESSION_PARSER_H_ */
