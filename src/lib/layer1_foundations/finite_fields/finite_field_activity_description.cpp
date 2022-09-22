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

	f_cheat_sheet_GF = FALSE;

	f_polynomial_division = FALSE;
	//polynomial_division_A;
	//polynomial_division_B;

	f_extended_gcd_for_polynomials = FALSE;

	f_polynomial_mult_mod = FALSE;
	//std::string polynomial_mult_mod_A;
	//std::string polynomial_mult_mod_B;
	//std::string polynomial_mult_mod_M;

	f_polynomial_power_mod = FALSE;
	//std::string polynomial_power_mod_A;
	//std::string polynomial_power_mod_n;
	//std::string polynomial_power_mod_M;




	f_Berlekamp_matrix = FALSE;
	//Berlekamp_matrix_label;

	f_normal_basis = FALSE;
	normal_basis_d = 0;

	f_polynomial_find_roots = FALSE;
	//polynomial_find_roots_label;

	f_normalize_from_the_right = FALSE;
	f_normalize_from_the_left = FALSE;


	f_nullspace = FALSE;
	//nullspace_input_matrix = NULL;

	f_RREF = FALSE;
	//RREF_input_matrix

	//cout << "finite_field_activity_description::finite_field_activity_description 3" << endl;
	//RREF_text = NULL;

	f_Walsh_Hadamard_transform = FALSE;
	//std::string Walsh_Hadamard_transform_fname_csv_in;
	Walsh_Hadamard_transform_n = 0;

	f_algebraic_normal_form = FALSE;
	//std::string algebraic_normal_form_fname_csv_in;
	algebraic_normal_form_n = 0;



	f_apply_trace_function = FALSE;
	//std::string apply_trace_function_fname_csv_in;

	f_apply_power_function = FALSE;
	//std::string apply_power_function_fname_csv_in;
	apply_power_function_d = 0;

	f_identity_function = FALSE;
	//std::string identity_function_fname_csv_out;

	f_trace = FALSE;
	f_norm = FALSE;

	f_Walsh_matrix = FALSE;
	Walsh_matrix_n = 0;

	f_Vandermonde_matrix = FALSE;

	f_search_APN_function = FALSE;

	f_make_table_of_irreducible_polynomials = FALSE;
	make_table_of_irreducible_polynomials_degree = 0;

	f_EC_Koblitz_encoding = FALSE;
	EC_s = 0;
	//EC_message = NULL;
	f_EC_points = FALSE;
	f_EC_add = FALSE;
	//EC_pt1_text = NULL;
	//EC_pt2_text = NULL;
	f_EC_cyclic_subgroup = FALSE;
	EC_b = 0;
	EC_c = 0;
	//EC_pt_text = NULL;
	f_EC_multiple_of = FALSE;
	EC_multiple_of_n = 0;
	f_EC_discrete_log = FALSE;
	//EC_discrete_log_pt_text = NULL;
	f_EC_baby_step_giant_step = FALSE;
	//EC_bsgs_G = NULL;
	EC_bsgs_N = 0;
	//EC_bsgs_cipher_text = NULL;
	f_EC_baby_step_giant_step_decode = FALSE;
	//EC_bsgs_A = NULL;
	//EC_bsgs_keys = NULL;


	//cout << "finite_field_activity_description::finite_field_activity_description done" << endl;
	f_NTRU_encrypt = FALSE;
	NTRU_encrypt_N = 0;
	NTRU_encrypt_p = 0;
	//NTRU_encrypt_H, NTRU_encrypt_R, NTRU_encrypt_Msg
	f_polynomial_center_lift = FALSE;
	//polynomial_center_lift_A
	f_polynomial_reduce_mod_p = FALSE;
	//polynomial_reduce_mod_p_A;

	f_cheat_sheet_Gr = FALSE;
	cheat_sheet_Gr_n = 0;
	cheat_sheet_Gr_k = 0;

	f_cheat_sheet_hermitian = FALSE;
	cheat_sheet_hermitian_projective_dimension = 0;

	f_cheat_sheet_desarguesian_spread = FALSE;
	cheat_sheet_desarguesian_spread_m = 0;

	f_sift_polynomials = FALSE;
	sift_polynomials_r0 = 0;
	sift_polynomials_r1 = 0;

	f_mult_polynomials = FALSE;
	mult_polynomials_r0 = 0;
	mult_polynomials_r1 = 0;

	f_polynomial_division_ranked = FALSE;
	polynomial_division_r0 = 0;
	polynomial_division_r1 = 0;

	f_RREF_random_matrix = FALSE;
	RREF_random_matrix_m = 0;
	RREF_random_matrix_n = 0;



	f_transversal = FALSE;
	//transversal_line_1_basis = NULL;
	//transversal_line_2_basis = NULL;
	//transversal_point = NULL;
	f_intersection_of_two_lines = FALSE;
	//line_1_basis = NULL;
	//line_2_basis = NULL;

	f_field_reduction = FALSE;
	//field_reduction_label
	field_reduction_q = 0;
	field_reduction_m = 0;
	field_reduction_n = 0;
	// field_reduction_text;

	f_parse_and_evaluate = FALSE;
	//parse_name_of_formula
	//parse_managed_variables
	//std::string parse_text;
	//std::string parse_parameters

	f_product_of = FALSE;
	//std::string product_of_elements;

	f_sum_of = FALSE;
	//std::string sum_of_elements;

	f_negate = FALSE;
	//std::string negate_elements;

	f_inverse = FALSE;
	//std::string inverse_elements;

	f_power_map = FALSE;
	power_map_k = 0;
	//std::string power_map_elements;

	f_evaluate = FALSE;
	//std::string evaluate_formula_label;
	//std::string evaluate_parameters;


	f_inverse_isomorphism_klein_quadric = FALSE;
	// std::string inverse_isomorphism_klein_quadric_matrix_A6;

	f_rank_point_in_PG = FALSE;
	//rank_point_in_PG_label;

	f_unrank_point_in_PG = FALSE;
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
			f_cheat_sheet_GF = TRUE;
			if (f_v) {
				cout << "-cheat_sheet_GF " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-polynomial_division") == 0) {
			f_polynomial_division = TRUE;
			polynomial_division_A.assign(argv[++i]);
			polynomial_division_B.assign(argv[++i]);
			if (f_v) {
				cout << "-polynomial_division " << polynomial_division_A << " "
						<< polynomial_division_B << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-extended_gcd_for_polynomials") == 0) {
			f_extended_gcd_for_polynomials = TRUE;
			polynomial_division_A.assign(argv[++i]);
			polynomial_division_B.assign(argv[++i]);
			if (f_v) {
				cout << "-extended_gcd_for_polynomials " << polynomial_division_A
					<< " " << polynomial_division_B << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-polynomial_mult_mod") == 0) {
			f_polynomial_mult_mod = TRUE;
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
			f_polynomial_power_mod = TRUE;
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
			f_Berlekamp_matrix = TRUE;
			Berlekamp_matrix_label.assign(argv[++i]);
			if (f_v) {
				cout << "-Berlekamp_matrix " << Berlekamp_matrix_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-normal_basis") == 0) {
			f_normal_basis = TRUE;
			normal_basis_d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-normal_basis " << normal_basis_d << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-normalize_from_the_right") == 0) {
			f_normalize_from_the_right = TRUE;
			if (f_v) {
				cout << "-normalize_from_the_right " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-normalize_from_the_left") == 0) {
			f_normalize_from_the_left = TRUE;
			if (f_v) {
				cout << "-normalize_from_the_left " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-nullspace") == 0) {
			f_nullspace = TRUE;
			nullspace_input_matrix.assign(argv[++i]);
			if (f_v) {
				cout << "-nullspace " << nullspace_input_matrix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-polynomial_find_roots") == 0) {
			f_polynomial_find_roots = TRUE;
			polynomial_find_roots_label.assign(argv[++i]);
			if (f_v) {
				cout << "-polynomial_find_roots " << polynomial_find_roots_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-RREF") == 0) {
			f_RREF = TRUE;
			RREF_input_matrix.assign(argv[++i]);
			if (f_v) {
				cout << "-RREF " << RREF_input_matrix << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-Walsh_Hadamard_transform") == 0) {
			f_Walsh_Hadamard_transform = TRUE;
			Walsh_Hadamard_transform_fname_csv_in.assign(argv[++i]);
			Walsh_Hadamard_transform_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Walsh_Hadamard_transform "
					<< Walsh_Hadamard_transform_fname_csv_in
					<< " " << Walsh_Hadamard_transform_n << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-algebraic_normal_form") == 0) {
			f_algebraic_normal_form = TRUE;
			algebraic_normal_form_fname_csv_in.assign(argv[++i]);
			algebraic_normal_form_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-algebraic_normal_form "
					<< algebraic_normal_form_fname_csv_in
					<< " " << algebraic_normal_form_n << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-apply_trace_function") == 0) {
			f_apply_trace_function = TRUE;
			apply_trace_function_fname_csv_in.assign(argv[++i]);
			if (f_v) {
				cout << "-apply_trace_function "
					<< apply_trace_function_fname_csv_in
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-apply_power_function") == 0) {
			f_apply_power_function = TRUE;
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
			f_identity_function = TRUE;
			identity_function_fname_csv_out.assign(argv[++i]);
			if (f_v) {
				cout << "-identity_function "
					<< identity_function_fname_csv_out
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-trace") == 0) {
			f_trace = TRUE;
			if (f_v) {
				cout << "-trace " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-norm") == 0) {
			f_norm = TRUE;
			if (f_v) {
				cout << "-norm " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-Walsh_matrix") == 0) {
			f_Walsh_matrix = TRUE;
			Walsh_matrix_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Walsh_matrix " << Walsh_matrix_n << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-Vandermonde_matrix") == 0) {
			f_Vandermonde_matrix = TRUE;
			if (f_v) {
				cout << "-Vandermonde_matrix " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-search_APN_function") == 0) {
			f_search_APN_function = TRUE;
			if (f_v) {
				cout << "-search_APN_function " << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-make_table_of_irreducible_polynomials") == 0) {
			f_make_table_of_irreducible_polynomials = TRUE;
			make_table_of_irreducible_polynomials_degree = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-make_table_of_irreducible_polynomials "
					<< make_table_of_irreducible_polynomials_degree << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-EC_Koblitz_encoding") == 0) {
			f_EC_Koblitz_encoding = TRUE;
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
			f_EC_points = TRUE;
			EC_label.assign(argv[++i]);
			EC_b = ST.strtoi(argv[++i]);
			EC_c = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-EC_points " << " " << EC_label << " " << EC_b << " " << EC_c << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-EC_add") == 0) {
			f_EC_add = TRUE;
			EC_b = ST.strtoi(argv[++i]);
			EC_c = ST.strtoi(argv[++i]);
			EC_pt1_text.assign(argv[++i]);
			EC_pt2_text.assign(argv[++i]);
			if (f_v) {
				cout << "-EC_add " << EC_b << " " << EC_c << " " << EC_pt1_text << " " << EC_pt2_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-EC_cyclic_subgroup") == 0) {
			f_EC_cyclic_subgroup = TRUE;
			EC_b = ST.strtoi(argv[++i]);
			EC_c = ST.strtoi(argv[++i]);
			EC_pt_text.assign(argv[++i]);
			if (f_v) {
				cout << "-EC_cyclic_subgroup " << " " << EC_b << " " << EC_c << " " << EC_pt_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-EC_multiple_of") == 0) {
			f_EC_multiple_of = TRUE;
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
			f_EC_discrete_log = TRUE;
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
			f_EC_baby_step_giant_step = TRUE;
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
			f_EC_baby_step_giant_step_decode = TRUE;
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
			f_NTRU_encrypt = TRUE;
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
			f_polynomial_center_lift = TRUE;
			polynomial_center_lift_A.assign(argv[++i]);
			if (f_v) {
				cout << "-polynomial_center_lift " << polynomial_center_lift_A << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-polynomial_reduce_mod_p") == 0) {
			f_polynomial_reduce_mod_p = TRUE;
			polynomial_reduce_mod_p_A.assign(argv[++i]);
			if (f_v) {
				cout << "-polynomial_reduce_mod_p " << polynomial_reduce_mod_p_A << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-cheat_sheet_Gr") == 0) {
			f_cheat_sheet_Gr = TRUE;
			cheat_sheet_Gr_n = ST.strtoi(argv[++i]);
			cheat_sheet_Gr_k = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-cheat_sheet_Gr " << cheat_sheet_Gr_n << " " << cheat_sheet_Gr_k << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-cheat_sheet_hermitian") == 0) {
			f_cheat_sheet_hermitian = TRUE;
			cheat_sheet_hermitian_projective_dimension = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-cheat_sheet_hermitian " << cheat_sheet_hermitian_projective_dimension << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-cheat_sheet_desarguesian_spread") == 0) {
			f_cheat_sheet_desarguesian_spread = TRUE;
			cheat_sheet_desarguesian_spread_m = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-cheat_sheet_desarguesian_spread " << cheat_sheet_desarguesian_spread_m << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-sift_polynomials") == 0) {
			f_sift_polynomials = TRUE;
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
			f_mult_polynomials = TRUE;
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
			f_polynomial_division_ranked = TRUE;
			polynomial_division_r0 = ST.strtolint(argv[++i]);
			polynomial_division_r1 = ST.strtolint(argv[++i]);
			if (f_v) {
				cout << "-polynomial_division_ranked "
					<< " " << polynomial_division_r0
					<< " " << polynomial_division_r1
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-RREF_random_matrix") == 0) {
			f_RREF_random_matrix = TRUE;
			RREF_random_matrix_m = ST.strtolint(argv[++i]);
			RREF_random_matrix_n = ST.strtolint(argv[++i]);
			if (f_v) {
				cout << "-RREF_demo "
					<< " " << RREF_random_matrix_m
					<< " " << RREF_random_matrix_n
					<< endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-transversal") == 0) {
			f_transversal = TRUE;
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
			f_intersection_of_two_lines = TRUE;
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
			f_inverse_isomorphism_klein_quadric = TRUE;
			inverse_isomorphism_klein_quadric_matrix_A6.assign(argv[++i]);
			if (f_v) {
				cout << "-inverse_isomorphism_klein_quadric "
					<< inverse_isomorphism_klein_quadric_matrix_A6 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-rank_point_in_PG") == 0) {
			f_rank_point_in_PG = TRUE;
			rank_point_in_PG_label.assign(argv[++i]);
			if (f_v) {
				cout << "-rank_point_in_PG " << rank_point_in_PG_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-unrank_point_in_PG") == 0) {
			f_unrank_point_in_PG = TRUE;
			unrank_point_in_PG_n = ST.strtolint(argv[++i]);
			unrank_point_in_PG_text.assign(argv[++i]);
			if (f_v) {
				cout << "-unrank_point_in_PG " << unrank_point_in_PG_n << " " << unrank_point_in_PG_text << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-field_reduction") == 0) {
			f_field_reduction = TRUE;
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
		else if (ST.stringcmp(argv[i], "-parse_and_evaluate") == 0) {
			f_parse_and_evaluate = TRUE;
			parse_name_of_formula.assign(argv[++i]);
			parse_managed_variables.assign(argv[++i]);
			parse_text.assign(argv[++i]);
			parse_parameters.assign(argv[++i]);
			if (f_v) {
				cout << "-parse_and_evaluate " << parse_name_of_formula
					<< " " << parse_managed_variables
					<< " " << parse_text
					<< " " << parse_parameters
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-product_of") == 0) {
			f_product_of = TRUE;
			product_of_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-product_of " << product_of_elements
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-sum_of") == 0) {
			f_sum_of = TRUE;
			sum_of_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-sum_of " << sum_of_elements
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-negate") == 0) {
			f_negate = TRUE;
			negate_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-negate " << negate_elements
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-inverse") == 0) {
			f_inverse = TRUE;
			inverse_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-inverse " << inverse_elements
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-power_map") == 0) {
			f_power_map = TRUE;
			power_map_k = ST.strtoi(argv[++i]);
			power_map_elements.assign(argv[++i]);
			if (f_v) {
				cout << "-power_map " << power_map_k
						<< " " << power_map_elements
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-evaluate") == 0) {
			f_evaluate = TRUE;
			evaluate_formula_label.assign(argv[++i]);
			evaluate_parameters.assign(argv[++i]);
			if (f_v) {
				cout << "-evaluate " << evaluate_formula_label
					<< " " << evaluate_parameters
					<< endl;
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
	if (f_normal_basis) {
		cout << "-normal_basis " << normal_basis_d << endl;
	}
	if (f_normalize_from_the_right) {
		cout << "-normalize_from_the_right " << endl;
	}
	if (f_normalize_from_the_left) {
		cout << "-normalize_from_the_left " << endl;
	}
	if (f_nullspace) {
		cout << "-nullspace " << nullspace_input_matrix << endl;
	}
	if (f_polynomial_find_roots) {
		cout << "-polynomial_find_roots " << polynomial_find_roots_label << endl;
	}
	if (f_RREF) {
		cout << "-RREF " << RREF_input_matrix << endl;
	}

	if (f_Walsh_Hadamard_transform) {
		cout << "-Walsh_Hadamard_transform "
				<< Walsh_Hadamard_transform_fname_csv_in
				<< " " << Walsh_Hadamard_transform_n << endl;
	}

	if (f_algebraic_normal_form) {
		cout << "-algebraic_normal_form "
				<< algebraic_normal_form_fname_csv_in
				<< " " << algebraic_normal_form_n << endl;
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

	if (f_trace) {
		cout << "-trace " << endl;
	}
	if (f_norm) {
		cout << "-norm " << endl;
	}

	if (f_Walsh_matrix) {
		cout << "-Walsh_matrix " << Walsh_matrix_n << endl;
	}
	if (f_Vandermonde_matrix) {
		cout << "-Vandermonde_matrix " << endl;
	}
	if (f_search_APN_function) {
		cout << "-search_APN_function " << endl;
	}

	if (f_make_table_of_irreducible_polynomials) {
		cout << "-make_table_of_irreducible_polynomials "
				<< make_table_of_irreducible_polynomials_degree << endl;
	}
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

#if 0
	if (f_cheat_sheet_PG) {
		cout << "-cheat_sheet_PG " << cheat_sheet_PG_n << endl;
	}
#endif

	if (f_cheat_sheet_Gr) {
		cout << "-cheat_sheet_Gr " << cheat_sheet_Gr_n << " " << cheat_sheet_Gr_k << endl;
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


	if (f_RREF_random_matrix) {
		cout << "-RREF_demo "
				<< " " << RREF_random_matrix_m
				<< " " << RREF_random_matrix_n
				<< endl;
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
	if (f_field_reduction) {
		cout << "-field_reduction "
				<< " " << field_reduction_label
				<< " " << field_reduction_q
				<< " " << field_reduction_m
				<< " " << field_reduction_n
				<< " " << field_reduction_text << endl;
	}
	if (f_parse_and_evaluate) {
		cout << "-parse_and_evaluate " << parse_name_of_formula
				<< " " << parse_managed_variables
				<< " " << parse_text
				<< " " << parse_parameters
				<< endl;
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

	if (f_evaluate) {
		cout << "-evaluate " << evaluate_formula_label
				<< " " << evaluate_parameters
				<< endl;
	}

}


}}}





