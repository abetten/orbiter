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
	void get_integer_matrix(
			int *&M, int verbose_level);
	void get_string_representation_Sajeeb(
			std::vector<std::string> &S);
	void get_string_representation_formula(
			std::vector<std::string> &S, int verbose_level);
	void print_Sajeeb(
			std::ostream &ost);
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
	void print_latex(
			std::ostream &ost, std::string &label);
	void make_A_minus_lambda_Identity(
			formula_vector *A,
			field_theory::finite_field *Fq,
			std::string &variable,
			std::string &label_txt,
			std::string &label_tex,
			int f_managed_variables,
			std::string &managed_variables_text,
			int verbose_level);
	void substitute(
			formula_vector *Source,
			formula_vector *Target,
			std::string &substitution_variables,
			std::string &label_txt,
			std::string &label_tex,
			int f_has_managed_variables,
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
			int f_has_managed_variables,
			std::string &managed_variables,
			int f_write_trees,
			int verbose_level);
	void characteristic_polynomial(
			formula_vector *A,
			field_theory::finite_field *Fq,
			std::string &variable,
			std::string &label_txt,
			std::string &label_tex,
			int f_has_managed_variables,
			std::string &managed_variables_text,
			int verbose_level);
	void determinant(
			formula_vector *A,
			field_theory::finite_field *Fq,
			std::string &label_txt,
			std::string &label_tex,
			int f_has_managed_variables,
			std::string &managed_variables_text,
			int verbose_level);
	void right_nullspace(
			formula_vector *A,
			field_theory::finite_field *Fq,
			std::string &label_txt,
			std::string &label_tex,
			int f_has_managed_variables,
			std::string &managed_variables,
			int verbose_level);
	void matrix_minor(
			formula_vector *A,
			field_theory::finite_field *Fq,
			int i, int j,
			std::string &label_txt,
			std::string &label_tex,
			int f_has_managed_variables,
			std::string &managed_variables_text,
			int verbose_level);
	void symbolic_nullspace(
			formula_vector *A,
			field_theory::finite_field *Fq,
			std::string &label_txt,
			std::string &label_tex,
			int f_has_managed_variables,
			std::string &managed_variables,
			int verbose_level);
	void multiply_2by2_from_the_left(
			formula_vector *M,
			formula_vector *A2,
			int i, int j,
			field_theory::finite_field *Fq,
			std::string &label_txt,
			std::string &label_tex,
			int f_has_managed_variables,
			std::string &managed_variables_text,
			int verbose_level);
	void latex_tree(
			int verbose_level);
	void export_tree(
			int verbose_level);
	void collect_variables(
			int verbose_level);
	void print_variables(
			std::ostream &ost);

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

	int f_has_managed_variables;
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
	void print(
			std::ostream &ost);
	void init_empty_plus_node(
			std::string &label, std::string &label_tex,
			int f_has_managed_variables,
			std::string &managed_variables_text,
			field_theory::finite_field *Fq,
			int verbose_level);
	void init_formula_from_tree(
			std::string &label, std::string &label_tex,
			field_theory::finite_field *Fq,
			syntax_tree *Tree,
			int verbose_level);
	void init_formula_int(
			std::string &label, std::string &label_tex,
			int value,
			field_theory::finite_field *Fq,
			int f_has_managed_variables,
			std::string &managed_variables,
			int verbose_level);
	void init_formula_monopoly(
			std::string &label, std::string &label_tex,
			field_theory::finite_field *Fq,
			int f_has_managed_variables,
			std::string &managed_variables,
			std::string &variable,
			int *coeffs, int nb_coeffs,
			int verbose_level);
	void init_formula_Sajeeb(
			std::string &label, std::string &label_tex,
			int f_has_managed_variables,
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
	void expand_in_place(
			int f_write_trees,
			int verbose_level);
	int highest_order_term(
			std::string &variable, int verbose_level);
	void get_monopoly(
			std::string &variable, int *&coeff, int &nb_coeff,
			int verbose_level);
	void get_multipoly(
			ring_theory::homogeneous_polynomial_domain *HPD,
			int *&eqn, int &sz,
			int verbose_level);
	void latex_tree(
			std::string &label, int verbose_level);
	void export_tree(
			std::string &label, int verbose_level);
	void latex_tree_split(
			std::string &label, int split_level, int split_mod,
			int verbose_level);
	void collect_variables(
			int verbose_level);
	void collect_monomial_terms(
			data_structures::int_matrix *&I, int *&Coeff,
			int verbose_level);
	void collect_coefficients_of_equation(
			ring_theory::homogeneous_polynomial_domain *Poly,
			int *&coeffs, int &nb_coeffs,
			int verbose_level);

};


// #############################################################################
// symbolic_object_activity_description.cpp
// #############################################################################


//! description of an activity involving a symbolic object

class symbolic_object_activity_description {
public:

	int f_export;

#if 0
	int f_evaluate;
	//std::string evaluate_finite_field_label;
	std::string evaluate_assignment;
#endif

	int f_print;
	//std::string print_over_Fq_field_label;

#if 0
	int f_sweep;
	std::string sweep_variables;

	int f_sweep_affine;
	std::string sweep_affine_variables;
#endif


	symbolic_object_activity_description();
	~symbolic_object_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// symbolic_object_activity.cpp
// #############################################################################


//! an activity involving a symbolic object

class symbolic_object_activity {
public:


	symbolic_object_activity_description *Descr;
	symbolic_object_builder *f;

	symbolic_object_activity();
	~symbolic_object_activity();
	void init(
			symbolic_object_activity_description *Descr,
			symbolic_object_builder *f,
			int verbose_level);
	void perform_activity(int verbose_level);
#if 0
	void do_sweep(int f_affine,
			formula *f,
			std::string &sweep_variables,
			int verbose_level);
#endif

};



// #############################################################################
// symbolic_object_builder_description.cpp
// #############################################################################



//! to define a symbolic object


class symbolic_object_builder_description {
public:

	int f_label_txt;
	std::string label_txt;

	int f_label_tex;
	std::string label_tex;

	int f_managed_variables;
	std::string managed_variables;

	int f_text;
	std::string text_txt;

	int f_field;
	std::string field_label;

	int f_field_pointer;
	field_theory::finite_field *field_pointer;


	int f_ring;
	std::string ring_label;

	int f_file;
	std::string file_name;




	int f_matrix;
	int nb_rows;

	int f_determinant;
	std::string determinant_source;

	int f_characteristic_polynomial;
	std::string characteristic_polynomial_variable;
	std::string characteristic_polynomial_source;

	int f_substitute;
	std::string substitute_variables;
	std::string substitute_target;
	std::string substitute_source;


	int f_simplify;
	std::string simplify_source;

	int f_expand;
	std::string expand_source;

	int f_right_nullspace;
	std::string right_nullspace_source;

	int f_minor;
	std::string minor_source;
	int minor_i;
	int minor_j;

	int f_symbolic_nullspace;
	std::string symbolic_nullspace_source;

	int f_stack_matrices_vertically;
	int f_stack_matrices_horizontally;
	int f_stack_matrices_z_shape;
	std::string stack_matrices_label;

	int f_multiply_2x2_from_the_left;
	std::string multiply_2x2_from_the_left_source;
	std::string multiply_2x2_from_the_left_A2;
	int multiply_2x2_from_the_left_i;
	int multiply_2x2_from_the_left_j;

	int f_matrix_entry;
	std::string matrix_entry_source;
	int matrix_entry_i;
	int matrix_entry_j;

	int f_vector_entry;
	std::string vector_entry_source;
	int vector_entry_i;

	int f_collect;
	std::string collect_source;
	std::string collect_by;

	int f_collect_by;
	std::string collect_by_source;
	std::string collect_by_ring;

	int f_encode_CRC;
	int encode_CRC_block_length;
	std::string encode_CRC_data_polynomial;
	std::string encode_CRC_check_polynomial;

	int f_decode_CRC;
	int decode_CRC_block_length;
	std::string decode_CRC_data_polynomial;
	std::string decode_CRC_check_polynomial;

	int f_submatrix;
	std::string submatrix_source;
	int submatrix_row_first;
	int submatrix_nb_rows;
	int submatrix_col_first;
	int submatrix_nb_cols;


	int f_do_not_simplify;

	int f_write_trees_during_expand;

	symbolic_object_builder_description();
	~symbolic_object_builder_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();



};


// #############################################################################
// symbolic_object_builder.cpp
// #############################################################################



//! to create a vector of symbolic objects from class symbolic_object_builder_description


class symbolic_object_builder {
public:

	symbolic_object_builder_description *Descr;
	std::string label;

	field_theory::finite_field *Fq;

	ring_theory::homogeneous_polynomial_domain *Ring;


	expression_parser::formula_vector *Formula_vector;

	symbolic_object_builder();
	~symbolic_object_builder();
	void init(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void process_arguments(
			int verbose_level);
	void do_determinant(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_characteristic_polynomial(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_substitute(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_simplify(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_expand(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_right_nullspace(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_minor(
			symbolic_object_builder_description *Descr,
			int minor_i, int minor_j,
			std::string &label,
			int verbose_level);
	void do_symbolic_nullspace(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_stack(
			symbolic_object_builder_description *Descr,
			int verbose_level);
	void do_multiply_2x2_from_the_left(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_matrix_entry(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_vector_entry(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_collect(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_collect_by(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_CRC_encode(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_CRC_decode(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void do_submatrix(
			symbolic_object_builder_description *Descr,
			std::string &label,
			int verbose_level);
	void multiply_terms(
			expression_parser::formula **terms,
			int n,
			int &stage_counter,
			int verbose_level);

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
	void print(
			std::ostream &ost);
	void print_easy(
			std::ostream &ost);
	void print_expression(
			std::ostream &ost);
	void print_graphviz(
			std::ostream &ost);
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
	int text_value_match(
			std::string &factor);
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
	void get_multipoly(
			ring_theory::homogeneous_polynomial_domain *HPD,
			int *&eqn, int &sz,
			int verbose_level);
	void get_monomial(
			std::map<std::string,int> &variable_idx, int nb_variables,
			int *exponent_vector, int &coeff,
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
	void simplify_exponents(
			int verbose_level);
	void sort_terms(
			int verbose_level);
	void collect_like_terms(
			int verbose_level);
	void collect_like_terms_addition(
			int verbose_level);
	void collect_monomial_terms(
			data_structures::int_matrix *&I, int *&Coeff,
			int verbose_level);
	void collect_terms_and_coefficients(
			data_structures::int_matrix *&I, int *&Coeff,
			int verbose_level);
	void simplify_constants(
			int verbose_level);
	void flatten(
			int verbose_level);
	void flatten_with_depth(
			int depth,
			int verbose_level);
	void flatten_post_process(
			int depth,
			int verbose_level);
	void flatten_at(
			int i, int depth, int verbose_level);
	void delete_all_but_one_child(
			int i, int verbose_level);
	void delete_one_child(
			int i, int verbose_level);
	int is_constant_one(
			int verbose_level);
	int is_constant_zero(
			int verbose_level);
	void collect_variables(
			int verbose_level);
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
	void init_root_node(
			int verbose_level);
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
	void print(
			std::ostream &ost);
	void print_easy(
			std::ostream &ost);
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
	void get_multipoly(
			ring_theory::homogeneous_polynomial_domain *HPD,
			int *&eqn, int &sz,
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
	void add_variable(
			std::string &var, int verbose_level);
	int get_number_of_variables();
	std::string &get_variable_name(
			int index);
	int needs_to_be_expanded();
	long int evaluate(
			std::map<std::string, std::string> &symbol_table,
			int verbose_level);

};

// #############################################################################
// syntax_tree_latex.cpp
// #############################################################################

//! auxiliary class to latex print syntax tree nodes


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
