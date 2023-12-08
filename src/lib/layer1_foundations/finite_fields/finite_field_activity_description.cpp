/*
 * finite_field_activity_description.cpp
 *
 *  Created on: Nov 4, 2020
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace field_theory {


finite_field_activity_description::finite_field_activity_description()
{

	f_cheat_sheet_GF = false;

	f_export_tables = false;

	f_polynomial_division = false;
	//polynomial_division_A;
	//polynomial_division_B;

	f_extended_gcd_for_polynomials = false;

	f_polynomial_mult_mod = false;
	//std::string polynomial_mult_mod_A;
	//std::string polynomial_mult_mod_B;
	//std::string polynomial_mult_mod_M;

	f_polynomial_power_mod = false;
	//std::string polynomial_power_mod_A;
	//std::string polynomial_power_mod_n;
	//std::string polynomial_power_mod_M;


	f_Berlekamp_matrix = false;
	//Berlekamp_matrix_label;


	f_polynomial_find_roots = false;
	//polynomial_find_roots_label;

	f_product_of = false;
	//std::string product_of_elements;

	f_sum_of = false;
	//std::string sum_of_elements;

	f_negate = false;
	//std::string negate_elements;

	f_inverse = false;
	//std::string inverse_elements;

	f_power_map = false;
	power_map_k = 0;
	//std::string power_map_elements;

	// Extension fields:

	f_trace = false;

	f_norm = false;

	f_normal_basis = false;
	normal_basis_d = 0;

	f_normal_basis_with_given_polynomial = false;
	//std::string normal_basis_with_given_polynomial_poly_encoded;
	normal_basis_with_given_polynomial_d = 0;



	f_nth_roots = false;
	nth_roots_n = 0;

	f_field_reduction = false;
	//field_reduction_label
	field_reduction_q = 0;
	field_reduction_m = 0;
	field_reduction_n = 0;
	// field_reduction_text;



	// Linear Algebra

	f_nullspace = false;
	//nullspace_input_matrix = NULL;

	f_RREF = false;
	//RREF_input_matrix

	//f_normalize_from_the_right = false;
	//f_normalize_from_the_left = false;

	f_RREF_random_matrix = false;
	RREF_random_matrix_m = 0;
	RREF_random_matrix_n = 0;

	f_Walsh_matrix = false;
	Walsh_matrix_n = 0;

	f_Vandermonde_matrix = false;






	//cout << "finite_field_activity_description::finite_field_activity_description 3" << endl;
	//RREF_text = NULL;

	f_Walsh_Hadamard_transform = false;
	//std::string Walsh_Hadamard_transform_fname_csv_in;
	Walsh_Hadamard_transform_n = 0;

	f_algebraic_normal_form_of_boolean_function = false;
	//std::string algebraic_normal_form_of_boolean_function_fname_csv_in;
	algebraic_normal_form_of_boolean_function_n = 0;

	f_algebraic_normal_form = false;
	algebraic_normal_form_n = 0;
	//std::string algebraic_normal_form_input;


	f_apply_trace_function = false;
	//std::string apply_trace_function_fname_csv_in;

	f_apply_power_function = false;
	//std::string apply_power_function_fname_csv_in;
	apply_power_function_d = 0;

	f_identity_function = false;
	//std::string identity_function_fname_csv_out;


	f_search_APN_function = false;

	f_make_table_of_irreducible_polynomials = false;
	make_table_of_irreducible_polynomials_degree = 0;

	f_get_primitive_polynomial = false;
	get_primitive_polynomial_degree = 0;

	f_get_primitive_polynomial_in_range = false;
	get_primitive_polynomial_in_range_min = 0;
	get_primitive_polynomial_in_range_max = 0;


	// cryptography:

	f_EC_Koblitz_encoding = false;
	EC_s = 0;
	//EC_message = NULL;
	f_EC_points = false;
	f_EC_add = false;
	//EC_pt1_text = NULL;
	//EC_pt2_text = NULL;
	f_EC_cyclic_subgroup = false;
	EC_b = 0;
	EC_c = 0;
	//EC_pt_text = NULL;
	f_EC_multiple_of = false;
	EC_multiple_of_n = 0;
	f_EC_discrete_log = false;
	//EC_discrete_log_pt_text = NULL;
	f_EC_baby_step_giant_step = false;
	//EC_bsgs_G = NULL;
	EC_bsgs_N = 0;
	//EC_bsgs_cipher_text = NULL;
	f_EC_baby_step_giant_step_decode = false;
	//EC_bsgs_A = NULL;
	//EC_bsgs_keys = NULL;


	//cout << "finite_field_activity_description::finite_field_activity_description done" << endl;
	f_NTRU_encrypt = false;
	NTRU_encrypt_N = 0;
	NTRU_encrypt_p = 0;
	//NTRU_encrypt_H, NTRU_encrypt_R, NTRU_encrypt_Msg
	f_polynomial_center_lift = false;
	//polynomial_center_lift_A
	f_polynomial_reduce_mod_p = false;
	//polynomial_reduce_mod_p_A;


	f_cheat_sheet_hermitian = false;
	cheat_sheet_hermitian_projective_dimension = 0;

	f_cheat_sheet_desarguesian_spread = false;
	cheat_sheet_desarguesian_spread_m = 0;

	f_sift_polynomials = false;
	sift_polynomials_r0 = 0;
	sift_polynomials_r1 = 0;

	f_mult_polynomials = false;
	mult_polynomials_r0 = 0;
	mult_polynomials_r1 = 0;

	f_polynomial_division_ranked = false;
	polynomial_division_r0 = 0;
	polynomial_division_r1 = 0;

	f_assemble_monopoly = false;
	assemble_monopoly_length = 0;
	//std::string assemble_monopoly_coefficient_vector;
	//std::string assemble_monopoly_exponent_vector;



	f_transversal = false;
	//transversal_line_1_basis = NULL;
	//transversal_line_2_basis = NULL;
	//transversal_point = NULL;
	f_intersection_of_two_lines = false;
	//line_1_basis = NULL;
	//line_2_basis = NULL;


	f_inverse_isomorphism_klein_quadric = false;
	// std::string inverse_isomorphism_klein_quadric_matrix_A6;

	f_rank_point_in_PG = false;
	//rank_point_in_PG_label;

	f_unrank_point_in_PG = false;
	unrank_point_in_PG_n = 0;
	//std::string unrank_point_in_PG_text;


}


finite_field_activity_description::~finite_field_activity_description()
{

}

int finite_field_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "finite_field_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-cheat_sheet_GF") == 0) {
			f_cheat_sheet_GF = true;
			if (f_v) {
				cout << "-cheat_sheet_GF " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_tables") == 0) {
			f_export_tables = true;
			if (f_v) {
				cout << "-export_tables " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-polynomial_division") == 0) {
			f_polynomial_division = true;
			polynomial_division_A.assign(argv[++i]);
			polynomial_division_B.assign(argv[++i]);
			if (f_v) {
				cout << "-polynomial_division " << polynomial_division_A << " "
						<< polynomial_division_B << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-extended_gcd_for_polynomials") == 0) {
			f_extended_gcd_for_polynomials = true;
			polynomial_division_A.assign(argv[++i]);
			polynomial_division_B.assign(argv[++i]);
			if (f_v) {
				cout << "-extended_gcd_for_polynomials " << polynomial_division_A
					<< " " << polynomial_division_B << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-polynomial_mult_mod") == 0) {
			f_polynomial_mult_mod = true;
			polynomial_mult_mod_A.assign(argv[++i]);
			polynomial_mult_mod_B.assign(argv[++i]);
			polynomial_mult_mod_M.assign(argv[++i]);
			if (f_v) {
				cout << "-polynomial_mult_mod "
						<< " " << polynomial_mult_mod_A
						<< " " << polynomial_mult_mod_B
						<< " " << polynomial_mult_mod_M << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-polynomial_power_mod") == 0) {
			f_polynomial_power_mod = true;
			polynomial_power_mod_A.assign(argv[++i]);
			polynomial_power_mod_n.assign(argv[++i]);
			polynomial_power_mod_M.assign(argv[++i]);
			if (f_v) {
				cout << "-polynomial_power_mod "
						<< " " << polynomial_power_mod_A
						<< " " << polynomial_power_mod_n
						<< " " << polynomial_power_mod_M << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-Berlekamp_matrix") == 0) {
			f_Berlekamp_matrix = true;
			Berlekamp_matrix_label.assign(argv[++i]);
			if (f_v) {
				cout << "-Berlekamp_matrix " << Berlekamp_matrix_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-nullspace") == 0) {
			f_nullspace = true;
			nullspace_input_matrix.assign(argv[++i]);
			if (f_v) {
				cout << "-nullspace " << nullspace_input_matrix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-polynomial_find_roots") == 0) {
			f_polynomial_find_roots = true;
			polynomial_find_roots_label.assign(argv[++i]);
			if (f_v) {
				cout << "-polynomial_find_roots " << polynomial_find_roots_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-product_of") == 0) {
			f_product_of = true;
			product_of_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-product_of " << product_of_elements
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-sum_of") == 0) {
			f_sum_of = true;
			sum_of_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-sum_of " << sum_of_elements
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-negate") == 0) {
			f_negate = true;
			negate_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-negate " << negate_elements
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-inverse") == 0) {
			f_inverse = true;
			inverse_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-inverse " << inverse_elements
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-power_map") == 0) {
			f_power_map = true;
			power_map_k = ST.strtoi(argv[++i]);
			power_map_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-power_map " << power_map_k
						<< " " << power_map_elements
					<< endl;
			}
		}

		// Extension fields:

		else if (ST.stringcmp(argv[i], "-trace") == 0) {
			f_trace = true;
			if (f_v) {
				cout << "-trace " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-norm") == 0) {
			f_norm = true;
			if (f_v) {
				cout << "-norm " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-normal_basis") == 0) {
			f_normal_basis = true;
			normal_basis_d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-normal_basis " << normal_basis_d << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-normal_basis_with_given_polynomial") == 0) {
			f_normal_basis_with_given_polynomial = true;
			normal_basis_with_given_polynomial_poly_encoded.assign(argv[++i]);
			normal_basis_with_given_polynomial_d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-normal_basis_with_given_polynomial "
						<< " " << normal_basis_with_given_polynomial_poly_encoded
						<< " " << normal_basis_with_given_polynomial_d << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-nth_roots") == 0) {
			f_nth_roots = true;
			nth_roots_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-nth_roots " << nth_roots_n << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-field_reduction") == 0) {
			f_field_reduction = true;
			field_reduction_label.assign(argv[++i]);
			field_reduction_q = ST.strtoi(argv[++i]);
			field_reduction_m = ST.strtoi(argv[++i]);
			field_reduction_n = ST.strtoi(argv[++i]);
			field_reduction_text.assign(argv[++i]);
			if (f_v) {
				cout << "-field_reduction "
					<< " " << field_reduction_label
					<< " " << field_reduction_q
					<< " " << field_reduction_m
					<< " " << field_reduction_n
					<< " " << field_reduction_text << endl;
			}
		}





		// Linear Algebra:
		else if (ST.stringcmp(argv[i], "-RREF") == 0) {
			f_RREF = true;
			RREF_input_matrix.assign(argv[++i]);
			if (f_v) {
				cout << "-RREF " << RREF_input_matrix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-RREF_random_matrix") == 0) {
			f_RREF_random_matrix = true;
			RREF_random_matrix_m = ST.strtolint(argv[++i]);
			RREF_random_matrix_n = ST.strtolint(argv[++i]);
			if (f_v) {
				cout << "-RREF_random_matrix "
					<< " " << RREF_random_matrix_m
					<< " " << RREF_random_matrix_n
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Walsh_matrix") == 0) {
			f_Walsh_matrix = true;
			Walsh_matrix_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Walsh_matrix " << Walsh_matrix_n << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-Vandermonde_matrix") == 0) {
			f_Vandermonde_matrix = true;
			if (f_v) {
				cout << "-Vandermonde_matrix " << endl;
			}
		}






		else if (ST.stringcmp(argv[i], "-Walsh_Hadamard_transform") == 0) {
			f_Walsh_Hadamard_transform = true;
			Walsh_Hadamard_transform_fname_csv_in.assign(argv[++i]);
			Walsh_Hadamard_transform_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Walsh_Hadamard_transform "
					<< Walsh_Hadamard_transform_fname_csv_in
					<< " " << Walsh_Hadamard_transform_n << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-algebraic_normal_form_of_boolean_function") == 0) {
			f_algebraic_normal_form_of_boolean_function = true;
			algebraic_normal_form_of_boolean_function_fname_csv_in.assign(argv[++i]);
			algebraic_normal_form_of_boolean_function_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-algebraic_normal_form_of_boolean_function "
					<< algebraic_normal_form_of_boolean_function_fname_csv_in
					<< " " << algebraic_normal_form_of_boolean_function_n << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-algebraic_normal_form") == 0) {
			f_algebraic_normal_form = true;
			algebraic_normal_form_n = ST.strtoi(argv[++i]);
			algebraic_normal_form_input.assign(argv[++i]);
			if (f_v) {
				cout << "-algebraic_normal_form "
					<< algebraic_normal_form_n
					<< " " << algebraic_normal_form_input << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-apply_trace_function") == 0) {
			f_apply_trace_function = true;
			apply_trace_function_fname_csv_in.assign(argv[++i]);
			if (f_v) {
				cout << "-apply_trace_function "
					<< apply_trace_function_fname_csv_in
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-apply_power_function") == 0) {
			f_apply_power_function = true;
			apply_power_function_fname_csv_in.assign(argv[++i]);
			apply_power_function_d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-apply_power_function "
					<< apply_power_function_fname_csv_in
					<< " " << apply_power_function_d
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-identity_function") == 0) {
			f_identity_function = true;
			identity_function_fname_csv_out.assign(argv[++i]);
			if (f_v) {
				cout << "-identity_function "
					<< identity_function_fname_csv_out
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-search_APN_function") == 0) {
			f_search_APN_function = true;
			if (f_v) {
				cout << "-search_APN_function " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-make_table_of_irreducible_polynomials") == 0) {
			f_make_table_of_irreducible_polynomials = true;
			make_table_of_irreducible_polynomials_degree = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-make_table_of_irreducible_polynomials "
					<< make_table_of_irreducible_polynomials_degree << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-get_primitive_polynomial") == 0) {
			f_get_primitive_polynomial = true;
			get_primitive_polynomial_degree = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-get_primitive_polynomial " << get_primitive_polynomial_degree << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-get_primitive_polynomial_in_range") == 0) {
			f_get_primitive_polynomial_in_range = true;
			get_primitive_polynomial_in_range_min = ST.strtoi(argv[++i]);
			get_primitive_polynomial_in_range_max = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-get_primitive_polynomial_in_range"
						<< " " << get_primitive_polynomial_in_range_min
						<< " " << get_primitive_polynomial_in_range_max
						<< endl;
			}
		}

		// cryptography stuff:

		else if (ST.stringcmp(argv[i], "-EC_Koblitz_encoding") == 0) {
			f_EC_Koblitz_encoding = true;
			EC_b = ST.strtoi(argv[++i]);
			EC_c = ST.strtoi(argv[++i]);
			EC_s = ST.strtoi(argv[++i]);
			EC_pt_text.assign(argv[++i]);
			EC_message.assign(argv[++i]);
			if (f_v) {
				cout << "-EC_Koblitz_encoding "
					<< EC_b << " " << EC_c << " " << EC_s << " "
					<< EC_pt_text << " " << EC_message << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-EC_points") == 0) {
			f_EC_points = true;
			EC_label.assign(argv[++i]);
			EC_b = ST.strtoi(argv[++i]);
			EC_c = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-EC_points " << " " << EC_label << " " << EC_b << " " << EC_c << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-EC_add") == 0) {
			f_EC_add = true;
			EC_b = ST.strtoi(argv[++i]);
			EC_c = ST.strtoi(argv[++i]);
			EC_pt1_text.assign(argv[++i]);
			EC_pt2_text.assign(argv[++i]);
			if (f_v) {
				cout << "-EC_add " << EC_b << " " << EC_c << " " << EC_pt1_text << " " << EC_pt2_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-EC_cyclic_subgroup") == 0) {
			f_EC_cyclic_subgroup = true;
			EC_b = ST.strtoi(argv[++i]);
			EC_c = ST.strtoi(argv[++i]);
			EC_pt_text.assign(argv[++i]);
			if (f_v) {
				cout << "-EC_cyclic_subgroup " << " " << EC_b << " " << EC_c << " " << EC_pt_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-EC_multiple_of") == 0) {
			f_EC_multiple_of = true;
			EC_b = ST.strtoi(argv[++i]);
			EC_c = ST.strtoi(argv[++i]);
			EC_pt_text.assign(argv[++i]);
			EC_multiple_of_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-EC_multiple_of " << " " << EC_b << " " << EC_c << " " << EC_pt_text
					<< " " << EC_multiple_of_n << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-EC_discrete_log") == 0) {
			f_EC_discrete_log = true;
			EC_b = ST.strtoi(argv[++i]);
			EC_c = ST.strtoi(argv[++i]);
			EC_pt_text.assign(argv[++i]);
			EC_discrete_log_pt_text.assign(argv[++i]);
			if (f_v) {
				cout << "-EC_discrete_log " << " " << EC_b << " " << EC_c << " " << EC_pt_text << " "
					<< EC_discrete_log_pt_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-EC_bsgs") == 0) {
			f_EC_baby_step_giant_step = true;
			EC_b = ST.strtoi(argv[++i]);
			EC_c = ST.strtoi(argv[++i]);
			EC_bsgs_G.assign(argv[++i]);
			EC_bsgs_N = ST.strtoi(argv[++i]);
			EC_bsgs_cipher_text.assign(argv[++i]);
			if (f_v) {
				cout << "-EC_baby_step_giant_step " << " " << EC_b << " " << EC_c << " "
					<< EC_bsgs_G << " "
					<< EC_bsgs_N << " "
					<< EC_bsgs_cipher_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-EC_bsgs_decode") == 0) {
			f_EC_baby_step_giant_step_decode = true;
			EC_b = ST.strtoi(argv[++i]);
			EC_c = ST.strtoi(argv[++i]);
			EC_bsgs_A.assign(argv[++i]);
			EC_bsgs_N = ST.strtoi(argv[++i]);
			EC_bsgs_cipher_text.assign(argv[++i]);
			EC_bsgs_keys.assign(argv[++i]);
			if (f_v) {
				cout << "-EC_baby_step_giant_step_decode "
					<< EC_b << " " << EC_c << " "
					<< EC_bsgs_A << " "
					<< EC_bsgs_N << " "
					<< EC_bsgs_cipher_text << " "
					<< EC_bsgs_keys << " "
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-NTRU_encrypt") == 0) {
			f_NTRU_encrypt = true;
			NTRU_encrypt_N = ST.strtoi(argv[++i]);
			NTRU_encrypt_p = ST.strtoi(argv[++i]);
			NTRU_encrypt_H.assign(argv[++i]);
			NTRU_encrypt_R.assign(argv[++i]);
			NTRU_encrypt_Msg.assign(argv[++i]);
			if (f_v) {
				cout << "-polynomial_mult_mod " << NTRU_encrypt_N
					<< " " << NTRU_encrypt_p
					<< " " << NTRU_encrypt_H
					<< " " << NTRU_encrypt_R
					<< " " << NTRU_encrypt_Msg << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-polynomial_center_lift") == 0) {
			f_polynomial_center_lift = true;
			polynomial_center_lift_A.assign(argv[++i]);
			if (f_v) {
				cout << "-polynomial_center_lift " << polynomial_center_lift_A << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-polynomial_reduce_mod_p") == 0) {
			f_polynomial_reduce_mod_p = true;
			polynomial_reduce_mod_p_A.assign(argv[++i]);
			if (f_v) {
				cout << "-polynomial_reduce_mod_p " << polynomial_reduce_mod_p_A << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-cheat_sheet_hermitian") == 0) {
			f_cheat_sheet_hermitian = true;
			cheat_sheet_hermitian_projective_dimension = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-cheat_sheet_hermitian " << cheat_sheet_hermitian_projective_dimension << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-cheat_sheet_desarguesian_spread") == 0) {
			f_cheat_sheet_desarguesian_spread = true;
			cheat_sheet_desarguesian_spread_m = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-cheat_sheet_desarguesian_spread " << cheat_sheet_desarguesian_spread_m << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-sift_polynomials") == 0) {
			f_sift_polynomials = true;
			sift_polynomials_r0 = ST.strtolint(argv[++i]);
			sift_polynomials_r1 = ST.strtolint(argv[++i]);
			if (f_v) {
				cout << "-sift_polynomials "
					<< " " << sift_polynomials_r0
					<< " " << sift_polynomials_r1
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-mult_polynomials") == 0) {
			f_mult_polynomials = true;
			mult_polynomials_r0 = ST.strtolint(argv[++i]);
			mult_polynomials_r1 = ST.strtolint(argv[++i]);
			if (f_v) {
				cout << "-mult_polynomials "
					<< " " << mult_polynomials_r0
					<< " " << mult_polynomials_r1
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-polynomial_division_ranked") == 0) {
			f_polynomial_division_ranked = true;
			polynomial_division_r0 = ST.strtolint(argv[++i]);
			polynomial_division_r1 = ST.strtolint(argv[++i]);
			if (f_v) {
				cout << "-polynomial_division_ranked "
					<< " " << polynomial_division_r0
					<< " " << polynomial_division_r1
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-assemble_monopoly") == 0) {
			f_assemble_monopoly = true;
			assemble_monopoly_length = ST.strtolint(argv[++i]);
			assemble_monopoly_coefficient_vector.assign(argv[++i]);
			assemble_monopoly_exponent_vector.assign(argv[++i]);
			if (f_v) {
				cout << "-assemble_monopoly "
						<< " " << assemble_monopoly_length
						<< " " << assemble_monopoly_coefficient_vector
						<< " " << assemble_monopoly_exponent_vector << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-transversal") == 0) {
			f_transversal = true;
			transversal_line_1_basis.assign(argv[++i]);
			transversal_line_2_basis.assign(argv[++i]);
			transversal_point.assign(argv[++i]);
			if (f_v) {
				cout << "-transversal "
					<< " " << transversal_line_1_basis
					<< " " << transversal_line_2_basis
					<< " " << transversal_point << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-intersection_of_two_lines") == 0) {
			f_intersection_of_two_lines = true;
			line_1_basis.assign(argv[++i]);
			line_2_basis.assign(argv[++i]);
			if (f_v) {
				cout << "-intersection_of_two_lines "
					<< " " << line_1_basis
					<< " " << line_2_basis
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-inverse_isomorphism_klein_quadric") == 0) {
			f_inverse_isomorphism_klein_quadric = true;
			inverse_isomorphism_klein_quadric_matrix_A6.assign(argv[++i]);
			if (f_v) {
				cout << "-inverse_isomorphism_klein_quadric "
					<< inverse_isomorphism_klein_quadric_matrix_A6 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-rank_point_in_PG") == 0) {
			f_rank_point_in_PG = true;
			rank_point_in_PG_label.assign(argv[++i]);
			if (f_v) {
				cout << "-rank_point_in_PG " << rank_point_in_PG_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-unrank_point_in_PG") == 0) {
			f_unrank_point_in_PG = true;
			unrank_point_in_PG_n = ST.strtolint(argv[++i]);
			unrank_point_in_PG_text.assign(argv[++i]);
			if (f_v) {
				cout << "-unrank_point_in_PG " << unrank_point_in_PG_n << " " << unrank_point_in_PG_text << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "finite_field_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "finite_field_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void finite_field_activity_description::print()
{
	if (f_cheat_sheet_GF) {
		cout << "-cheat_sheet_GF " << endl;
	}
	if (f_export_tables) {
		cout << "-export_tables " << endl;
	}
	if (f_polynomial_division) {
		cout << "-polynomial_division " << polynomial_division_A << " "
				<< polynomial_division_B << endl;
	}
	if (f_extended_gcd_for_polynomials) {
		cout << "-extended_gcd_for_polynomials " << polynomial_division_A
				<< " " << polynomial_division_B << endl;
	}
	if (f_polynomial_mult_mod) {
		cout << "-polynomial_mult_mod "
				<< " " << polynomial_mult_mod_A
				<< " " << polynomial_mult_mod_B
				<< " " << polynomial_mult_mod_M << endl;
	}
	if (f_polynomial_power_mod) {
		cout << "-polynomial_power_mod "
				<< " " << polynomial_power_mod_A
				<< " " << polynomial_power_mod_n
				<< " " << polynomial_power_mod_M << endl;
	}
	if (f_Berlekamp_matrix) {
		cout << "-Berlekamp_matrix " << Berlekamp_matrix_label << endl;
	}
	if (f_polynomial_find_roots) {
		cout << "-polynomial_find_roots " << polynomial_find_roots_label << endl;
	}
	if (f_product_of) {
			cout << "-product_of " << product_of_elements
				<< endl;
	}
	if (f_sum_of) {
			cout << "-sum_of " << sum_of_elements
				<< endl;
	}
	if (f_negate) {
			cout << "-negate " << negate_elements
				<< endl;
	}
	if (f_inverse) {
			cout << "-inverse " << inverse_elements
				<< endl;
	}
	if (f_power_map) {
			cout << "-power_map " << power_map_k
					<< " " << power_map_elements
				<< endl;
	}

	// Extension fields:

	if (f_trace) {
		cout << "-trace " << endl;
	}
	if (f_norm) {
		cout << "-norm " << endl;
	}
	if (f_normal_basis) {
		cout << "-normal_basis " << normal_basis_d << endl;
	}
	if (f_normal_basis_with_given_polynomial) {
			cout << "-normal_basis_with_given_polynomial "
					<< " " << normal_basis_with_given_polynomial_poly_encoded
					<< " " << normal_basis_with_given_polynomial_d << endl;
	}
	if (f_nth_roots) {
		cout << "-nth_roots " << nth_roots_n << endl;
	}
	if (f_field_reduction) {
		cout << "-field_reduction "
				<< " " << field_reduction_label
				<< " " << field_reduction_q
				<< " " << field_reduction_m
				<< " " << field_reduction_n
				<< " " << field_reduction_text << endl;
	}


	// Linear Algebra:

	if (f_RREF) {
		cout << "-RREF " << RREF_input_matrix << endl;
	}
	if (f_nullspace) {
		cout << "-nullspace " << nullspace_input_matrix << endl;
	}
	if (f_RREF_random_matrix) {
		cout << "-RREF_random_matrix "
				<< " " << RREF_random_matrix_m
				<< " " << RREF_random_matrix_n
				<< endl;
	}
	if (f_Walsh_matrix) {
		cout << "-Walsh_matrix " << Walsh_matrix_n << endl;
	}
	if (f_Vandermonde_matrix) {
		cout << "-Vandermonde_matrix " << endl;
	}





	if (f_Walsh_Hadamard_transform) {
		cout << "-Walsh_Hadamard_transform "
				<< Walsh_Hadamard_transform_fname_csv_in
				<< " " << Walsh_Hadamard_transform_n << endl;
	}

	if (f_algebraic_normal_form_of_boolean_function) {
		cout << "-algebraic_normal_form_of_boolean_function "
				<< algebraic_normal_form_of_boolean_function_fname_csv_in
				<< " " << algebraic_normal_form_of_boolean_function_n << endl;
	}
	if (f_algebraic_normal_form) {
			cout << "-algebraic_normal_form "
				<< algebraic_normal_form_n
				<< " " << algebraic_normal_form_input << endl;
	}

	if (f_apply_trace_function) {
		cout << "-apply_trace_function "
				<< apply_trace_function_fname_csv_in
				<< endl;
	}

	if (f_apply_power_function) {
		cout << "-apply_power_function "
				<< apply_power_function_fname_csv_in
				<< " " << apply_power_function_d
				<< endl;
	}

	if (f_identity_function) {
		cout << "-identity_function "
				<< identity_function_fname_csv_out
				<< endl;
	}

	if (f_search_APN_function) {
		cout << "-search_APN_function " << endl;
	}

	if (f_make_table_of_irreducible_polynomials) {
		cout << "-make_table_of_irreducible_polynomials "
				<< make_table_of_irreducible_polynomials_degree << endl;
	}
	if (f_get_primitive_polynomial) {
		cout << "-get_primitive_polynomial " << get_primitive_polynomial_degree << endl;
	}
	if (f_get_primitive_polynomial_in_range) {
		cout << "-get_primitive_polynomial_in_range"
				<< " " << get_primitive_polynomial_in_range_min
				<< " " << get_primitive_polynomial_in_range_max
				<< endl;
	}

	// cryptography stuff:

	if (f_EC_Koblitz_encoding) {
		cout << "-EC_Koblitz_encoding "
				<< EC_b << " " << EC_c << " " << EC_s << " "
				<< EC_pt_text << " " << EC_message << endl;
	}
	if (f_EC_points) {
		cout << "-EC_points " << " " << EC_label << " " << EC_b << " " << EC_c << endl;
	}
	if (f_EC_add) {
		cout << "-EC_add " << EC_b << " " << EC_c << " " << EC_pt1_text << " " << EC_pt2_text << endl;
	}
	if (f_EC_cyclic_subgroup) {
		cout << "-EC_cyclic_subgroup " << " " << EC_b << " " << EC_c << " " << EC_pt_text << endl;
	}
	if (f_EC_multiple_of) {
		cout << "-EC_multiple_of " << " " << EC_b << " " << EC_c << " " << EC_pt_text
				<< " " << EC_multiple_of_n << endl;
	}
	if (f_EC_discrete_log) {
		cout << "-EC_discrete_log "
				<< " " << EC_b
				<< " " << EC_c
				<< " " << EC_pt_text
				<< " " << EC_discrete_log_pt_text
				<< endl;
	}
	if (f_EC_baby_step_giant_step) {
		cout << "-EC_baby_step_giant_step "
				<< " " << EC_b
				<< " " << EC_c
				<< " " << EC_bsgs_G
				<< " " << EC_bsgs_N
				<< " " << EC_bsgs_cipher_text
				<< endl;
	}
	if (f_EC_baby_step_giant_step_decode) {
		cout << "-EC_baby_step_giant_step_decode "
				<< EC_b << " " << EC_c << " "
				<< EC_bsgs_A << " "
				<< EC_bsgs_N << " "
				<< EC_bsgs_cipher_text << " "
				<< EC_bsgs_keys << " "
				<< endl;
	}
	if (f_NTRU_encrypt) {
		cout << "-NTRU_encrypt " << NTRU_encrypt_N
				<< " " << NTRU_encrypt_p
				<< " " << NTRU_encrypt_H
				<< " " << NTRU_encrypt_R
				<< " " << NTRU_encrypt_Msg << endl;
	}
	if (f_polynomial_center_lift) {
		cout << "-polynomial_center_lift " << polynomial_center_lift_A << endl;
	}
	if (f_polynomial_reduce_mod_p) {
		cout << "-polynomial_reduce_mod_p " << polynomial_reduce_mod_p_A << endl;
	}

	if (f_cheat_sheet_hermitian) {
		cout << "-cheat_sheet_hermitian " << cheat_sheet_hermitian_projective_dimension << endl;
	}
	if (f_cheat_sheet_desarguesian_spread) {
		cout << "-cheat_sheet_desarguesian_spread " << cheat_sheet_desarguesian_spread_m << endl;
	}
	if (f_sift_polynomials) {
		cout << "-sift_polynomials "
				<< " " << sift_polynomials_r0
				<< " " << sift_polynomials_r1
				<< endl;
	}
	if (f_mult_polynomials) {
		cout << "-mult_polynomials "
				<< " " << mult_polynomials_r0
				<< " " << mult_polynomials_r1
				<< endl;
	}
	if (f_polynomial_division_ranked) {
		cout << "-polynomial_division_ranked "
				<< " " << polynomial_division_r0
				<< " " << polynomial_division_r1
				<< endl;
	}
	if (f_assemble_monopoly) {
		cout << "-assemble_monopoly "
				<< " " << assemble_monopoly_length
				<< " " << assemble_monopoly_coefficient_vector
				<< " " << assemble_monopoly_exponent_vector << endl;
	}

	if (f_transversal) {
		cout << "-transversal "
				<< " " << transversal_line_1_basis
				<< " " << transversal_line_2_basis
				<< " " << transversal_point << endl;
	}
	if (f_intersection_of_two_lines) {
		cout << "-intersection_of_two_lines "
				<< " " << line_1_basis
				<< " " << line_2_basis
				<< endl;
	}
	if (f_inverse_isomorphism_klein_quadric) {
		cout << "-inverse_isomorphism_klein_quadric "
				<< inverse_isomorphism_klein_quadric_matrix_A6 << endl;
	}
	if (f_rank_point_in_PG) {
		cout << "-rank_point_in_PG " << rank_point_in_PG_label << endl;
	}
	if (f_unrank_point_in_PG) {
		cout << "-unrank_point_in_PG " << unrank_point_in_PG_n << " " << unrank_point_in_PG_text << endl;
	}

}


}}}





