/*
 * finite_field_activity_description.cpp
 *
 *  Created on: Nov 4, 2020
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


finite_field_activity_description::finite_field_activity_description()
{

	f_q = FALSE;
	q = 0;
	f_override_polynomial = FALSE;
	//std::string override_polynomial;

	f_cheat_sheet_GF = FALSE;
	f_all_rational_normal_forms = FALSE;
	d = 0;

	f_polynomial_division = FALSE;
	//polynomial_division_A;
	//polynomial_division_B;
	f_extended_gcd_for_polynomials = FALSE;

	f_polynomial_mult_mod = FALSE;
	//std::string polynomial_mult_mod_A;
	//std::string polynomial_mult_mod_B;
	//std::string polynomial_mult_mod_M;

	f_Berlekamp_matrix = FALSE;
	//Berlekamp_matrix_coeffs;

	f_normal_basis = FALSE;
	normal_basis_d = 0;

	f_polynomial_find_roots = FALSE;
	//polynomial_find_roots_A;

	f_normalize_from_the_right = FALSE;
	f_normalize_from_the_left = FALSE;


	f_nullspace = FALSE;
	nullspace_m = 0;
	nullspace_n = 0;
	//nullspace_text = NULL;

	f_RREF = FALSE;
	RREF_m = 0;
	RREF_n = 0;

	//cout << "interface_cryptography::interface_cryptography 3" << endl;
	//RREF_text = NULL;

	f_weight_enumerator = FALSE;
	f_trace = FALSE;
	f_norm = FALSE;

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


	//cout << "interface_cryptography::interface_cryptography done" << endl;
	f_NTRU_encrypt = FALSE;
	NTRU_encrypt_N = 0;
	NTRU_encrypt_p = 0;
	//NTRU_encrypt_H, NTRU_encrypt_R, NTRU_encrypt_Msg
	f_polynomial_center_lift = FALSE;
	//polynomial_center_lift_A
	f_polynomial_reduce_mod_p = FALSE;
	//polynomial_reduce_mod_p_A;

	f_cheat_sheet_PG = FALSE;
	cheat_sheet_PG_n = 0;

	f_decomposition_by_element = FALSE;
	decomposition_by_element_power = 1;
	//std::string decomposition_by_element_data
	//decomposition_by_element_fname_base


	f_canonical_form_PG = FALSE;
	canonical_form_PG_n = 0;
	Canonical_form_PG_Descr = NULL;

	f_transversal = FALSE;
	//transversal_line_1_basis = NULL;
	//transversal_line_2_basis = NULL;
	//transversal_point = NULL;
	f_intersection_of_two_lines = FALSE;
	//line_1_basis = NULL;
	//line_2_basis = NULL;

	f_move_two_lines_in_hyperplane_stabilizer = FALSE;
	line1_from = 0;
	line2_from = 0;
	line1_to = 0;
	line2_to = 0;

	f_move_two_lines_in_hyperplane_stabilizer_text = FALSE;
	//std:string line1_from_text;
	//std:string line2_from_text;
	//std:string line1_to_text;
	//std:string line2_to_text;

	f_study_surface = FALSE;
	study_surface_nb = 0;

	f_inverse_isomorphism_klein_quadric = FALSE;
	// std::string inverse_isomorphism_klein_quadric_matrix_A6;

	f_rank_point_in_PG = FALSE;
	rank_point_in_PG_n = 0;
	//rank_point_in_PG_text;

	f_eigenstuff = FALSE;
	f_eigenstuff_from_file = FALSE;
	eigenstuff_n = 0;
	//eigenstuff_coeffs = NULL;
	//eigenstuff_fname = NULL;

}


finite_field_activity_description::~finite_field_activity_description()
{

}

int finite_field_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "finite_field_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (stringcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = strtoi(argv[++i]);
			cout << "-q " << q << endl;
		}
		else if (stringcmp(argv[i], "-override_polynomial") == 0) {
			f_override_polynomial = TRUE;
			override_polynomial.assign(argv[++i]);
			cout << "-override_polynomial" << override_polynomial << endl;
		}
		else if (stringcmp(argv[i], "-cheat_sheet_GF") == 0) {
			f_cheat_sheet_GF = TRUE;
			cout << "-cheat_sheet_GF " << endl;
		}
		else if (stringcmp(argv[i], "-all_rational_normal_forms") == 0) {
			f_all_rational_normal_forms = TRUE;
			d = strtoi(argv[++i]);
			cout << "-f_all_rational_normal_forms " << d << endl;
		}
		else if (stringcmp(argv[i], "-polynomial_division") == 0) {
			f_polynomial_division = TRUE;
			polynomial_division_A.assign(argv[++i]);
			polynomial_division_B.assign(argv[++i]);
			cout << "-polynomial_division " << polynomial_division_A << " "
					<< polynomial_division_B << endl;
		}
		else if (stringcmp(argv[i], "-extended_gcd_for_polynomials") == 0) {
			f_extended_gcd_for_polynomials = TRUE;
			polynomial_division_A.assign(argv[++i]);
			polynomial_division_B.assign(argv[++i]);
			cout << "-extended_gcd_for_polynomials " << polynomial_division_A
					<< " " << polynomial_division_B << endl;
		}
		else if (stringcmp(argv[i], "-polynomial_mult_mod") == 0) {
			f_polynomial_mult_mod = TRUE;
			polynomial_mult_mod_A.assign(argv[++i]);
			polynomial_mult_mod_B.assign(argv[++i]);
			polynomial_mult_mod_M.assign(argv[++i]);
			cout << "-polynomial_mult_mod "
					<< " " << polynomial_mult_mod_A
					<< " " << polynomial_mult_mod_B
					<< " " << polynomial_mult_mod_M << endl;
		}
		else if (stringcmp(argv[i], "-Berlekamp_matrix") == 0) {
			f_Berlekamp_matrix = TRUE;
			Berlekamp_matrix_coeffs.assign(argv[++i]);
			cout << "-Berlekamp_matrix " << Berlekamp_matrix_coeffs << endl;
		}
		else if (stringcmp(argv[i], "-normal_basis") == 0) {
			f_normal_basis = TRUE;
			normal_basis_d = strtoi(argv[++i]);
			cout << "-normal_basis " << normal_basis_d << endl;
		}
		else if (stringcmp(argv[i], "-normalize_from_the_right") == 0) {
			f_normalize_from_the_right = TRUE;
			cout << "-normalize_from_the_right " << endl;
		}
		else if (stringcmp(argv[i], "-normalize_from_the_left") == 0) {
			f_normalize_from_the_left = TRUE;
			cout << "-normalize_from_the_left " << endl;
		}
		else if (stringcmp(argv[i], "-nullspace") == 0) {
			f_nullspace = TRUE;
			nullspace_m = strtoi(argv[++i]);
			nullspace_n = strtoi(argv[++i]);
			nullspace_text.assign(argv[++i]);
			cout << "-nullspace " << nullspace_m << " "
					<< nullspace_n << " " << nullspace_text << endl;
		}
		else if (stringcmp(argv[i], "-polynomial_find_roots") == 0) {
			f_polynomial_find_roots = TRUE;
			polynomial_find_roots_A.assign(argv[++i]);
			cout << "-polynomial_find_roots " << polynomial_find_roots_A << endl;
		}
		else if (stringcmp(argv[i], "-RREF") == 0) {
			f_RREF = TRUE;
			RREF_m = strtoi(argv[++i]);
			RREF_n = strtoi(argv[++i]);
			RREF_text.assign(argv[++i]);
			cout << "-RREF " << RREF_m << " " << RREF_n
					<< " " << RREF_text << endl;
		}
		else if (stringcmp(argv[i], "-weight_enumerator") == 0) {
			f_weight_enumerator = TRUE;
			RREF_m = strtoi(argv[++i]);
			RREF_n = strtoi(argv[++i]);
			RREF_text.assign(argv[++i]);
			cout << "-weight_enumerator " << RREF_m << " " << RREF_n
					<< " " << RREF_text << endl;
		}
		else if (stringcmp(argv[i], "-trace") == 0) {
			f_trace = TRUE;
			cout << "-trace " << endl;
		}
		else if (stringcmp(argv[i], "-norm") == 0) {
			f_norm = TRUE;
			cout << "-norm " << endl;
		}
		else if (stringcmp(argv[i], "-make_table_of_irreducible_polynomials") == 0) {
			f_make_table_of_irreducible_polynomials = TRUE;
			make_table_of_irreducible_polynomials_degree = strtoi(argv[++i]);
			cout << "-make_table_of_irreducible_polynomials "
					<< make_table_of_irreducible_polynomials_degree << endl;
		}
		else if (stringcmp(argv[i], "-EC_Koblitz_encoding") == 0) {
			f_EC_Koblitz_encoding = TRUE;
			EC_b = strtoi(argv[++i]);
			EC_c = strtoi(argv[++i]);
			EC_s = strtoi(argv[++i]);
			EC_pt_text.assign(argv[++i]);
			EC_message.assign(argv[++i]);
			cout << "-EC_Koblitz_encoding "
					<< EC_b << " " << EC_c << " " << EC_s << " "
					<< EC_pt_text << " " << EC_message << endl;
		}
		else if (stringcmp(argv[i], "-EC_points") == 0) {
			f_EC_points = TRUE;
			EC_b = strtoi(argv[++i]);
			EC_c = strtoi(argv[++i]);
			cout << "-EC_points " << " " << EC_b << " " << EC_c << endl;
		}
		else if (stringcmp(argv[i], "-EC_add") == 0) {
			f_EC_add = TRUE;
			EC_b = strtoi(argv[++i]);
			EC_c = strtoi(argv[++i]);
			EC_pt1_text.assign(argv[++i]);
			EC_pt2_text.assign(argv[++i]);
			cout << "-EC_add " << EC_b << " " << EC_c << " " << EC_pt1_text << " " << EC_pt2_text << endl;
		}
		else if (stringcmp(argv[i], "-EC_cyclic_subgroup") == 0) {
			f_EC_cyclic_subgroup = TRUE;
			EC_b = strtoi(argv[++i]);
			EC_c = strtoi(argv[++i]);
			EC_pt_text.assign(argv[++i]);
			cout << "-EC_cyclic_subgroup " << " " << EC_b << " " << EC_c << " " << EC_pt_text << endl;
		}
		else if (stringcmp(argv[i], "-EC_multiple_of") == 0) {
			f_EC_multiple_of = TRUE;
			EC_b = strtoi(argv[++i]);
			EC_c = strtoi(argv[++i]);
			EC_pt_text.assign(argv[++i]);
			EC_multiple_of_n = strtoi(argv[++i]);
			cout << "-EC_multiple_of " << " " << EC_b << " " << EC_c << " " << EC_pt_text
					<< " " << EC_multiple_of_n << endl;
		}
		else if (stringcmp(argv[i], "-EC_discrete_log") == 0) {
			f_EC_discrete_log = TRUE;
			EC_b = strtoi(argv[++i]);
			EC_c = strtoi(argv[++i]);
			EC_pt_text.assign(argv[++i]);
			EC_discrete_log_pt_text.assign(argv[++i]);
			cout << "-EC_discrete_log " << " " << EC_b << " " << EC_c << " " << EC_pt_text << " "
					<< EC_discrete_log_pt_text << endl;
		}
		else if (stringcmp(argv[i], "-EC_bsgs") == 0) {
			f_EC_baby_step_giant_step = TRUE;
			EC_b = strtoi(argv[++i]);
			EC_c = strtoi(argv[++i]);
			EC_bsgs_G.assign(argv[++i]);
			EC_bsgs_N = strtoi(argv[++i]);
			EC_bsgs_cipher_text.assign(argv[++i]);
			cout << "-EC_baby_step_giant_step " << " " << EC_b << " " << EC_c << " "
					<< EC_bsgs_G << " "
					<< EC_bsgs_N << " "
					<< EC_bsgs_cipher_text << endl;
		}
		else if (stringcmp(argv[i], "-EC_bsgs_decode") == 0) {
			f_EC_baby_step_giant_step_decode = TRUE;
			EC_b = strtoi(argv[++i]);
			EC_c = strtoi(argv[++i]);
			EC_bsgs_A.assign(argv[++i]);
			EC_bsgs_N = strtoi(argv[++i]);
			EC_bsgs_cipher_text.assign(argv[++i]);
			EC_bsgs_keys.assign(argv[++i]);
			cout << "-EC_baby_step_giant_step_decode "
					<< EC_b << " " << EC_c << " "
					<< EC_bsgs_A << " "
					<< EC_bsgs_N << " "
					<< EC_bsgs_cipher_text << " "
					<< EC_bsgs_keys << " "
					<< endl;
		}
		else if (stringcmp(argv[i], "-NTRU_encrypt") == 0) {
			f_NTRU_encrypt = TRUE;
			NTRU_encrypt_N = strtoi(argv[++i]);
			NTRU_encrypt_p = strtoi(argv[++i]);
			NTRU_encrypt_H.assign(argv[++i]);
			NTRU_encrypt_R.assign(argv[++i]);
			NTRU_encrypt_Msg.assign(argv[++i]);
			cout << "-polynomial_mult_mod " << NTRU_encrypt_N
					<< " " << NTRU_encrypt_p
					<< " " << NTRU_encrypt_H
					<< " " << NTRU_encrypt_R
					<< " " << NTRU_encrypt_Msg << endl;
		}
		else if (stringcmp(argv[i], "-polynomial_center_lift") == 0) {
			f_polynomial_center_lift = TRUE;
			polynomial_center_lift_A.assign(argv[++i]);
			cout << "-polynomial_center_lift " << polynomial_center_lift_A << endl;
		}
		else if (stringcmp(argv[i], "-polynomial_reduce_mod_p") == 0) {
			f_polynomial_reduce_mod_p = TRUE;
			polynomial_reduce_mod_p_A.assign(argv[++i]);
			cout << "-polynomial_reduce_mod_p " << polynomial_reduce_mod_p_A << endl;
		}
		else if (stringcmp(argv[i], "-cheat_sheet_PG") == 0) {
			f_cheat_sheet_PG = TRUE;
			cheat_sheet_PG_n = strtoi(argv[++i]);
			cout << "-cheat_sheet_PG " << cheat_sheet_PG_n << endl;
		}
		else if (stringcmp(argv[i], "-decomposition_by_element") == 0) {
			f_decomposition_by_element = TRUE;
			decomposition_by_element_power = strtoi(argv[++i]);
			decomposition_by_element_data.assign(argv[++i]);
			decomposition_by_element_fname_base.assign(argv[++i]);
			cout << "-decomposition_by_element " <<  decomposition_by_element_power
					<< " " << decomposition_by_element_data
					<< " " << decomposition_by_element_fname_base << endl;
		}
		else if (stringcmp(argv[i], "-canonical_form_PG") == 0) {
			f_canonical_form_PG = TRUE;
			canonical_form_PG_n = strtoi(argv[++i]);
			cout << "-canonical_form_PG " << canonical_form_PG_n << ", reading extra arguments" << endl;

			Canonical_form_PG_Descr = NEW_OBJECT(projective_space_object_classifier_description);

			i += Canonical_form_PG_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			cout << "done reading -Canonical_form_PG_Descr " << canonical_form_PG_n << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-transversal") == 0) {
			f_transversal = TRUE;
			transversal_line_1_basis.assign(argv[++i]);
			transversal_line_2_basis.assign(argv[++i]);
			transversal_point.assign(argv[++i]);
			cout << "-transversal "
					<< " " << transversal_line_1_basis
					<< " " << transversal_line_2_basis
					<< " " << transversal_point << endl;
		}
		else if (stringcmp(argv[i], "-intersection_of_two_lines") == 0) {
			f_intersection_of_two_lines = TRUE;
			line_1_basis.assign(argv[++i]);
			line_2_basis.assign(argv[++i]);
			cout << "-intersection_of_two_lines "
					<< " " << line_1_basis
					<< " " << line_2_basis
					<< endl;
		}
		else if (stringcmp(argv[i], "-move_two_lines_in_hyperplane_stabilizer") == 0) {
			f_move_two_lines_in_hyperplane_stabilizer = TRUE;
			line1_from = strtoi(argv[++i]);
			line2_from = strtoi(argv[++i]);
			line1_to = strtoi(argv[++i]);
			line2_to = strtoi(argv[++i]);
			cout << "-move_two_lines_in_hyperplane_stabilizer"
					<< " " << line1_from
					<< " " << line1_from
					<< " " << line1_to
					<< " " << line2_to
					<< endl;
		}
		else if (stringcmp(argv[i], "-move_two_lines_in_hyperplane_stabilizer_text") == 0) {
			f_move_two_lines_in_hyperplane_stabilizer_text = TRUE;
			line1_from_text.assign(argv[++i]);
			line2_from_text.assign(argv[++i]);
			line1_to_text.assign(argv[++i]);
			line2_to_text.assign(argv[++i]);
			cout << "-move_two_lines_in_hyperplane_stabilizer_text"
					<< " " << line1_from_text
					<< " " << line2_from_text
					<< " " << line1_to_text
					<< " " << line2_to_text
					<< endl;
		}
		else if (stringcmp(argv[i], "-study_surface") == 0) {
			f_study_surface = TRUE;
			study_surface_nb = strtoi(argv[++i]);
			cout << "-study_surface" << study_surface_nb << endl;
		}
		else if (stringcmp(argv[i], "-inverse_isomorphism_klein_quadric") == 0) {
			f_inverse_isomorphism_klein_quadric = TRUE;
			inverse_isomorphism_klein_quadric_matrix_A6.assign(argv[++i]);
			cout << "-inverse_isomorphism_klein_quadric "
					<< inverse_isomorphism_klein_quadric_matrix_A6 << endl;
		}
		else if (stringcmp(argv[i], "-rank_point_in_PG") == 0) {
			f_rank_point_in_PG = TRUE;
			rank_point_in_PG_n = strtoi(argv[++i]);
			rank_point_in_PG_text.assign(argv[++i]);
			cout << "-rank_point_in_PG " << rank_point_in_PG_n << " " << rank_point_in_PG_text << endl;
		}
		else if (stringcmp(argv[i], "-eigenstuff") == 0) {
			f_eigenstuff = TRUE;
			eigenstuff_n = strtoi(argv[++i]);
			eigenstuff_coeffs.assign(argv[++i]);
			cout << "-eigenstuff " << eigenstuff_n
					<< " " << eigenstuff_coeffs << endl;
		}
		else if (stringcmp(argv[i], "-eigenstuff_matrix_from_file") == 0) {
			f_eigenstuff_from_file = TRUE;
			eigenstuff_n = strtoi(argv[++i]);
			eigenstuff_fname.assign(argv[++i]);
			cout << "-eigenstuff_from_file " << eigenstuff_n
					<< " " << eigenstuff_fname << endl;
		}


		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "finite_field_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "finite_field_activity_description::read_arguments done" << endl;
	return i + 1;
}



}}




