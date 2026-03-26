/*
 * activities_layer1.h
 *
 *  Created on: Mar 22, 2026
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER5_TOP_LEVEL_USER_INTERFACE_ACTIVITIES_LAYER1_ACTIVITIES_LAYER1_H_
#define SRC_LIB_LAYER5_TOP_LEVEL_USER_INTERFACE_ACTIVITIES_LAYER1_ACTIVITIES_LAYER1_H_



namespace orbiter {
namespace layer5_applications {
namespace user_interface {
namespace activities_layer1 {


// #############################################################################
// diophant_activity_description.cpp
// #############################################################################


//! to describe an activity for a diophantine system from command line arguments

class diophant_activity_description {
public:

	// doc/commands/diophant_activity.csv

	int f_input_file;
	std::string input_file;
	int f_print;
	int f_solve_mckay;
	int f_solve_standard;
	int f_solve_DLX;

	int f_draw_as_bitmap;
	int box_width;
	int bit_depth; // 8 or 24

	int f_draw;
	std::string draw_options_label;


	int f_perform_column_reductions;

	int f_project_to_single_equation_and_solve;
	int eqn_idx;
	int solve_case_idx;

	int f_project_to_two_equations_and_solve;
	int eqn1_idx;
	int eqn2_idx;
	int solve_case_idx_r;
	int solve_case_idx_m;

	int f_test_single_equation;
	int max_number_of_coefficients;



	diophant_activity_description();
	~diophant_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();



};


// #############################################################################
// diophant_activity.cpp
// #############################################################################


//! to perform an activity for a diophantine system using diophant_activity_description

class diophant_activity {
public:

	user_interface::activities_layer1::diophant_activity_description *Descr;

	diophant_activity();
	~diophant_activity();
	void init_from_file(
			user_interface::activities_layer1::diophant_activity_description *Descr,
			int verbose_level);
	void perform_activity(
			user_interface::activities_layer1::diophant_activity_description *Descr,
			combinatorics::solvers::diophant *Dio,
			int verbose_level);


};




// #############################################################################
// finite_field_activity_description.cpp
// #############################################################################


//! description of a finite field activity

class finite_field_activity_description {
public:


	// TABLES/finite_field_activities.tex


		int f_cheat_sheet_GF;

		int f_export_tables;


		int f_polynomial_division;
		std::string polynomial_division_A;
		std::string polynomial_division_B;

		int f_extended_gcd_for_polynomials;

		int f_polynomial_mult_mod;
		std::string polynomial_mult_mod_A;
		std::string polynomial_mult_mod_B;
		std::string polynomial_mult_mod_M;

		int f_polynomial_power_mod;
		std::string polynomial_power_mod_A;
		std::string polynomial_power_mod_n;
		std::string polynomial_power_mod_M;

		int f_Berlekamp_matrix;
		std::string Berlekamp_matrix_label;

		int f_polynomial_find_roots;
		std::string polynomial_find_roots_label;


		int f_product_of;
		std::string product_of_elements;

		int f_sum_of;
		std::string sum_of_elements;

		int f_negate;
		std::string negate_elements;

		int f_inverse;
		std::string inverse_elements;

		int f_power_map;
		int power_map_k;
		std::string power_map_elements;


	// Section 3.3:
	// Extension fields:
	// TABLES/finite_field_activities_2.tex

		int f_trace;

		int f_norm;

		int f_normal_basis;
		int normal_basis_d;

		int f_normal_basis_with_given_polynomial;
		std::string normal_basis_with_given_polynomial_poly_encoded;
		int normal_basis_with_given_polynomial_d;

		int f_nth_roots;
		int nth_roots_n;

		int f_field_reduction;
		std::string field_reduction_label;
		int field_reduction_q;
		int field_reduction_m;
		int field_reduction_n;
		std::string field_reduction_text;



	// Section 3.4:
	// Linear algebra:
	// TABLES/finite_field_activities_linear_algebra.tex

		int f_nullspace;
		std::string nullspace_input_matrix;

		int f_RREF;
		std::string RREF_input_matrix;


		int f_RREF_random_matrix;
		int RREF_random_matrix_m;
		int RREF_random_matrix_n;

		int f_Walsh_matrix;
		int Walsh_matrix_n;

		int f_Vandermonde_matrix;





	// TABLES/finite_field_activities_summary_1.tex


		int f_Walsh_Hadamard_transform;
		std::string Walsh_Hadamard_transform_fname_csv_in;
		int Walsh_Hadamard_transform_n;

		int f_algebraic_normal_form_of_boolean_function;
		std::string algebraic_normal_form_of_boolean_function_fname_csv_in;
		int algebraic_normal_form_of_boolean_function_n;

		int f_algebraic_normal_form;
		int algebraic_normal_form_n;
		std::string algebraic_normal_form_input;

		int f_apply_trace_function;
		std::string apply_trace_function_fname_csv_in;

		int f_apply_power_function;
		std::string apply_power_function_fname_csv_in;
		long int apply_power_function_d;

		int f_identity_function;
		std::string identity_function_fname_csv_out;



	// undocumented:

		int f_search_APN_function;

		int f_make_table_of_irreducible_polynomials;
		int make_table_of_irreducible_polynomials_degree;


	// TABLES/finite_field_activities_polynomials.tex


		int f_get_primitive_polynomial;
		int get_primitive_polynomial_degree;

		int f_get_primitive_polynomial_in_range;
		int get_primitive_polynomial_in_range_min;
		int get_primitive_polynomial_in_range_max;


	// cryptography:

	// TABLES/cryptography_2.tex

		int f_EC_Koblitz_encoding;
		std::string EC_message;
		int EC_s;

		int f_EC_points;
		std::string EC_label;

		int f_EC_add;
		std::string EC_pt1_text;
		std::string EC_pt2_text;

		int f_EC_cyclic_subgroup;
		int EC_b;
		int EC_c;
		std::string EC_pt_text;

		int f_EC_multiple_of;
		int EC_multiple_of_n;


		int f_EC_discrete_log;
		std::string EC_discrete_log_pt_text;


		int f_EC_baby_step_giant_step;
		std::string EC_bsgs_G;
		int EC_bsgs_N;
		std::string EC_bsgs_cipher_text;

		int f_EC_baby_step_giant_step_decode;
		std::string EC_bsgs_A;
		std::string EC_bsgs_keys;

		int f_NTRU_encrypt;
		int NTRU_encrypt_N;
		int NTRU_encrypt_p;
		std::string NTRU_encrypt_H;
		std::string NTRU_encrypt_R;
		std::string NTRU_encrypt_Msg;

		int f_polynomial_center_lift;
		std::string polynomial_center_lift_A;

		int f_polynomial_reduce_mod_p;
		std::string polynomial_reduce_mod_p_A;


	// undocumented:

		int f_cheat_sheet_hermitian;
		int cheat_sheet_hermitian_projective_dimension;

		int f_cheat_sheet_desarguesian_spread;
		int cheat_sheet_desarguesian_spread_m;

		int f_sift_polynomials;
		long int sift_polynomials_r0;
		long int sift_polynomials_r1;

		int f_mult_polynomials;
		long int mult_polynomials_r0;
		long int mult_polynomials_r1;

		int f_polynomial_division_ranked;
		long int polynomial_division_r0;
		long int polynomial_division_r1;

		int f_assemble_monopoly;
		int assemble_monopoly_length;
		std::string assemble_monopoly_coefficient_vector;
		std::string assemble_monopoly_exponent_vector;



	// TABLES/finite_field_activities_summary_2.tex

		int f_transversal;
		std::string transversal_line_1_basis;
		std::string transversal_line_2_basis;
		std::string transversal_point;

		int f_intersection_of_two_lines;
		std::string line_1_basis;
		std::string line_2_basis;

		int f_inverse_isomorphism_klein_quadric;
		std::string inverse_isomorphism_klein_quadric_matrix_A6;

		// ranking and unranking points in PG:

		int f_rank_point_in_PG;
		std::string rank_point_in_PG_label;

		int f_unrank_point_in_PG;
		int unrank_point_in_PG_n;
		std::string unrank_point_in_PG_text;





	finite_field_activity_description();
	~finite_field_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// finite_field_activity.cpp
// #############################################################################


//! perform a finite field activity

class finite_field_activity {
public:
	user_interface::activities_layer1::finite_field_activity_description *Descr;
	algebra::field_theory::finite_field *F;
	algebra::field_theory::finite_field *F_secondary;

	finite_field_activity();
	~finite_field_activity();
	void init(
			user_interface::activities_layer1::finite_field_activity_description *Descr,
			algebra::field_theory::finite_field *F,
			int verbose_level);
	void perform_activity(
			other::orbiter_kernel_system::activity_output *&AO,
			int verbose_level);

};




// #############################################################################
// polynomial_ring_activity_description.cpp
// #############################################################################


//! description of a polynomial ring activity

class polynomial_ring_activity_description {
public:


	// used as -ring_theoretic_activity


	// TABLES/polynomial_ring_activity.tex


	int f_cheat_sheet;

	int f_export_partials;

	int f_ideal;
	std::string ideal_label_txt;
	std::string ideal_label_tex;
	std::string ideal_point_set_label;

	int f_apply_transformation;
	std::string apply_transformation_space_label;
	std::string apply_transformation_Eqn_in_label;
	std::string apply_transformation_vector_ge_label;

	int f_set_variable_names;
	std::string set_variable_names_txt;
	std::string set_variable_names_tex;

	int f_print_equation;
	std::string print_equation_input;

	int f_parse_equation_wo_parameters;
	std::string parse_equation_wo_parameters_name_of_formula;
	std::string parse_equation_wo_parameters_name_of_formula_tex;
	std::string parse_equation_wo_parameters_equation_text;

	int f_parse_equation;
	std::string parse_equation_name_of_formula;
	std::string parse_equation_name_of_formula_tex;
	std::string parse_equation_equation_text;
	std::string parse_equation_equation_parameters;
	std::string parse_equation_equation_parameter_values;

	int f_table_of_monomials_write_csv;
	std::string table_of_monomials_write_csv_label;



	polynomial_ring_activity_description();
	~polynomial_ring_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};




// #############################################################################
// symbolic_object_activity_description.cpp
// #############################################################################


//! description of an activity involving a symbolic object

class symbolic_object_activity_description {
public:

	// TABLES/symbolic_object_activity.tex

	int f_export;

#if 0
	int f_evaluate;
	//std::string evaluate_finite_field_label;
	std::string evaluate_assignment;
#endif

	int f_print;
	//std::string print_over_Fq_field_label;

	int f_as_vector;

	int f_homogenize;

	int f_latex;

	int f_evaluate_affine;

	int f_collect_monomials_binary;

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
	algebra::expression_parser::symbolic_object_builder *f;

	symbolic_object_activity();
	~symbolic_object_activity();
	void init(
			user_interface::activities_layer1::symbolic_object_activity_description *Descr,
			algebra::expression_parser::symbolic_object_builder *f,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void print(
			int verbose_level);
	void as_vector(
			int *&v, int &len,
			int verbose_level);
	void stringify(
			std::vector<std::string> &String_rep,
			int verbose_level);
	void latex(
			int verbose_level);
	void evaluate_affine(
			int verbose_level);
	void collect_monomials_binary(
			int verbose_level);
#if 0
	void do_sweep(
			int f_affine,
			formula *f,
			std::string &sweep_variables,
			int verbose_level);
#endif

};





}}}}



#endif /* SRC_LIB_LAYER5_TOP_LEVEL_USER_INTERFACE_ACTIVITIES_LAYER1_ACTIVITIES_LAYER1_H_ */
