/*
 * finite_field_activity.cpp
 *
 *  Created on: Nov 4, 2020
 *      Author: betten
 */





#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace field_theory {



finite_field_activity::finite_field_activity()
{
	Descr = NULL;
	F = NULL;
	F_secondary = NULL;
}

finite_field_activity::~finite_field_activity()
{
}

void finite_field_activity::init(finite_field_activity_description *Descr,
		finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_activity::init" << endl;
	}

	finite_field_activity::Descr = Descr;
	finite_field_activity::F = F;
	if (f_v) {
		cout << "finite_field_activity::init done" << endl;
	}
}

void finite_field_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_activity::perform_activity" << endl;
	}


	if (Descr->f_cheat_sheet_GF) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_cheat_sheet_GF" << endl;
		}
		algebra::algebra_global Algebra;

		Algebra.do_cheat_sheet_GF(F, verbose_level);
	}
	else if (Descr->f_polynomial_division) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_polynomial_division" << endl;
		}
		ring_theory::ring_theory_global R;

		R.polynomial_division(F,
				Descr->polynomial_division_A,
				Descr->polynomial_division_B,
				verbose_level);
	}
	else if (Descr->f_extended_gcd_for_polynomials) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_extended_gcd_for_polynomials" << endl;
		}
		ring_theory::ring_theory_global R;

		R.extended_gcd_for_polynomials(F,
				Descr->polynomial_division_A,
				Descr->polynomial_division_B,
				verbose_level);
	}

	else if (Descr->f_polynomial_mult_mod) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_polynomial_mult_mod" << endl;
		}
		ring_theory::ring_theory_global R;

		R.polynomial_mult_mod(F,
				Descr->polynomial_mult_mod_A,
				Descr->polynomial_mult_mod_B,
				Descr->polynomial_mult_mod_M,
				verbose_level);
	}

	else if (Descr->f_polynomial_power_mod) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_polynomial_power_mod" << endl;
		}
		ring_theory::ring_theory_global R;

		R.polynomial_power_mod(F,
				Descr->polynomial_power_mod_A,
				Descr->polynomial_power_mod_n,
				Descr->polynomial_power_mod_M,
				verbose_level);

	}

	else if (Descr->f_Berlekamp_matrix) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_Berlekamp_matrix" << endl;
		}
		linear_algebra::linear_algebra_global LA;

		LA.Berlekamp_matrix(F,
				Descr->Berlekamp_matrix_label, verbose_level);

	}
	else if (Descr->f_normal_basis) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_normal_basis" << endl;
		}
		linear_algebra::linear_algebra_global LA;

		LA.compute_normal_basis(F,
				Descr->normal_basis_d, verbose_level);

	}
	else if (Descr->f_polynomial_find_roots) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_polynomial_find_roots" << endl;
		}
		ring_theory::ring_theory_global R;

		R.polynomial_find_roots(F,
				Descr->polynomial_find_roots_label,
				verbose_level);
	}

	else if (Descr->f_nullspace) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_nullspace" << endl;
		}
		linear_algebra::linear_algebra_global LA;
		int *v;
		int m, n;

		Get_matrix(Descr->nullspace_input_matrix, v, m, n);

		LA.do_nullspace(F,
				v, m, n,
				Descr->f_normalize_from_the_left,
				Descr->f_normalize_from_the_right, verbose_level);

		FREE_int(v);

	}
	else if (Descr->f_RREF) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_RREF" << endl;
		}
		linear_algebra::linear_algebra_global LA;
		int *v;
		int m, n;

		Get_matrix(Descr->RREF_input_matrix, v, m, n);

		LA.do_RREF(F,
				v, m, n,
				Descr->f_normalize_from_the_left,
				Descr->f_normalize_from_the_right,
				verbose_level);

		FREE_int(v);

	}

	else if (Descr->f_Walsh_Hadamard_transform) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_Walsh_Hadamard_transform" << endl;
		}
		algebra::algebra_global Algebra;

		Algebra.apply_Walsh_Hadamard_transform(F,
				Descr->Walsh_Hadamard_transform_fname_csv_in,
				Descr->Walsh_Hadamard_transform_n, verbose_level);
	}


	else if (Descr->f_algebraic_normal_form_of_boolean_function) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_algebraic_normal_form_of_boolean_function" << endl;
		}
		algebra::algebra_global Algebra;

		Algebra.algebraic_normal_form_of_boolean_function(F,
				Descr->algebraic_normal_form_of_boolean_function_fname_csv_in,
				Descr->algebraic_normal_form_of_boolean_function_n, verbose_level);
	}

	else if (Descr->f_algebraic_normal_form) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_algebraic_normal_form" << endl;
		}
		algebra::algebra_global Algebra;

		int *func;
		int len;

		Get_int_vector_from_label(Descr->algebraic_normal_form_input, func, len, 0 /* verbose_level */);

		Algebra.algebraic_normal_form(F,
				Descr->algebraic_normal_form_n,
				func, len, verbose_level);
	}

	else if (Descr->f_apply_trace_function) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_apply_trace_function" << endl;
		}
		algebra::algebra_global Algebra;

		Algebra.apply_trace_function(F,
				Descr->apply_trace_function_fname_csv_in, verbose_level);
	}

	else if (Descr->f_apply_power_function) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_apply_power_function" << endl;
		}
		algebra::algebra_global Algebra;

		Algebra.apply_power_function(F,
				Descr->apply_power_function_fname_csv_in,
				Descr->apply_power_function_d,
				verbose_level);
	}

	else if (Descr->f_identity_function) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_identity_function" << endl;
		}
		algebra::algebra_global Algebra;

		Algebra.identity_function(F,
				Descr->identity_function_fname_csv_out, verbose_level);
	}


	else if (Descr->f_trace) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_trace" << endl;
		}
		algebra::algebra_global Algebra;

		Algebra.do_trace(F, verbose_level);
	}
	else if (Descr->f_norm) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_norm" << endl;
		}
		algebra::algebra_global Algebra;

		Algebra.do_norm(F, verbose_level);
	}
	else if (Descr->f_Walsh_matrix) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_Walsh_matrix" << endl;
		}
		algebra::algebra_global Algebra;
		int *W = NULL;

		Algebra.Walsh_matrix(F, Descr->Walsh_matrix_n, W, verbose_level);
		FREE_int(W);
	}
	else if (Descr->f_Vandermonde_matrix) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_Vandermonde_matrix" << endl;
		}
		algebra::algebra_global Algebra;
		int *W = NULL;
		int *W_inv = NULL;

		Algebra.Vandermonde_matrix(F, W, W_inv, verbose_level);

		if (F->q < 33) {
			cout << "Vandermonde:" << endl;
			Int_matrix_print(W, F->q, F->q);
			cout << "Vandermonde inverse:" << endl;
			Int_matrix_print(W_inv, F->q, F->q);

			orbiter_kernel_system::latex_interface LI;

			cout << "Vandermonde:" << endl;

			cout << "$$" << endl;
			cout << "\\left[" << endl;
			LI.int_matrix_print_tex(cout, W, F->q, F->q);
			cout << "\\right]" << endl;
			cout << "$$" << endl;

			cout << "Vandermonde inverse:" << endl;
			cout << "$$" << endl;
			cout << "\\left[" << endl;
			LI.int_matrix_print_tex(cout, W_inv, F->q, F->q);
			cout << "\\right]" << endl;
			cout << "$$" << endl;


		}
		else {
			cout << "too big to print" << endl;
		}

		FREE_int(W);
		FREE_int(W_inv);
	}
	else if (Descr->f_search_APN_function) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_search_APN_function" << endl;
		}
		algebra::algebra_global Algebra;
		int delta_max = 2;

		Algebra.search_APN(F, delta_max, verbose_level);

	}



	else if (Descr->f_make_table_of_irreducible_polynomials) {


		if (f_v) {
			cout << "finite_field_activity::perform_activity f_make_table_of_irreducible_polynomials" << endl;
		}
		ring_theory::ring_theory_global R;

		R.do_make_table_of_irreducible_polynomials(F,
				Descr->make_table_of_irreducible_polynomials_degree,
				verbose_level);

	}
	else if (Descr->f_EC_Koblitz_encoding) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_EC_Koblitz_encoding" << endl;
		}
		cryptography::cryptography_domain Crypto;

		Crypto.do_EC_Koblitz_encoding(F,
				Descr->EC_b,
				Descr->EC_c,
				Descr->EC_s,
				Descr->EC_pt_text,
				Descr->EC_message,
				verbose_level);
	}
	else if (Descr->f_EC_points) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_EC_points" << endl;
		}
		cryptography::cryptography_domain Crypto;

		Crypto.do_EC_points(F,
				Descr->EC_label,
				Descr->EC_b,
				Descr->EC_c,
				verbose_level);
	}
	else if (Descr->f_EC_add) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_EC_add" << endl;
		}
		cryptography::cryptography_domain Crypto;

		Crypto.do_EC_add(F,
				Descr->EC_b,
				Descr->EC_c,
				Descr->EC_pt1_text,
				Descr->EC_pt2_text,
				verbose_level);
	}
	else if (Descr->f_EC_cyclic_subgroup) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_EC_cyclic_subgroup" << endl;
		}
		cryptography::cryptography_domain Crypto;

		Crypto.do_EC_cyclic_subgroup(F,
				Descr->EC_b,
				Descr->EC_c,
				Descr->EC_pt_text,
				verbose_level);
	}
	else if (Descr->f_EC_multiple_of) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_EC_multiple_of" << endl;
		}
		cryptography::cryptography_domain Crypto;

		Crypto.do_EC_multiple_of(F,
				Descr->EC_b,
				Descr->EC_c,
				Descr->EC_pt_text,
				Descr->EC_multiple_of_n,
				verbose_level);
	}
	else if (Descr->f_EC_discrete_log) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_EC_discrete_log" << endl;
		}
		cryptography::cryptography_domain Crypto;

		Crypto.do_EC_discrete_log(F,
				Descr->EC_b,
				Descr->EC_c,
				Descr->EC_pt_text,
				Descr->EC_discrete_log_pt_text,
				verbose_level);
	}
	else if (Descr->f_EC_baby_step_giant_step) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_EC_baby_step_giant_step" << endl;
		}
		cryptography::cryptography_domain Crypto;

		Crypto.do_EC_baby_step_giant_step(F,
				Descr->EC_b,
				Descr->EC_c,
				Descr->EC_bsgs_G,
				Descr->EC_bsgs_N,
				Descr->EC_bsgs_cipher_text,
				verbose_level);
	}
	else if (Descr->f_EC_baby_step_giant_step_decode) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_EC_baby_step_giant_step_decode" << endl;
		}
		cryptography::cryptography_domain Crypto;

		Crypto.do_EC_baby_step_giant_step_decode(F,
				Descr->EC_b,
				Descr->EC_c,
				Descr->EC_bsgs_A,
				Descr->EC_bsgs_N,
				Descr->EC_bsgs_cipher_text,
				Descr->EC_bsgs_keys,
				verbose_level);
	}
	else if (Descr->f_NTRU_encrypt) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_NTRU_encrypt" << endl;
		}
		cryptography::cryptography_domain Crypto;

		Crypto.NTRU_encrypt(Descr->NTRU_encrypt_N,
				Descr->NTRU_encrypt_p,
				F,
				Descr->NTRU_encrypt_H,
				Descr->NTRU_encrypt_R,
				Descr->NTRU_encrypt_Msg,
				verbose_level);
	}
	else if (Descr->f_polynomial_center_lift) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_polynomial_center_lift" << endl;
		}
		cryptography::cryptography_domain Crypto;

		Crypto.polynomial_center_lift(Descr->polynomial_center_lift_A, F,
				verbose_level);
	}
	else if (Descr->f_polynomial_reduce_mod_p) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_polynomial_reduce_mod_p" << endl;
		}
		cryptography::cryptography_domain Crypto;

		Crypto.polynomial_reduce_mod_p(Descr->polynomial_reduce_mod_p_A, F,
				verbose_level);
	}

	else if (Descr->f_cheat_sheet_Gr) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_cheat_sheet_Gr" << endl;
		}
		geometry::geometry_global Geo;

		Geo.do_cheat_sheet_Gr(F, Descr->cheat_sheet_Gr_n, Descr->cheat_sheet_Gr_k,
				verbose_level);
	}
	else if (Descr->f_cheat_sheet_hermitian) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_cheat_sheet_hermitian" << endl;
		}
		geometry::geometry_global Geo;

		Geo.do_cheat_sheet_hermitian(F,
				Descr->cheat_sheet_hermitian_projective_dimension,
				verbose_level);
	}
	else if (Descr->f_cheat_sheet_desarguesian_spread) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_cheat_sheet_desarguesian_spread" << endl;
		}
		geometry::geometry_global Geo;

		if (F_secondary == NULL) {
			cout << "F_secondary == NULL" << endl;
			exit(1);
		}
		Geo.do_create_desarguesian_spread(F, F_secondary,
				Descr->cheat_sheet_desarguesian_spread_m,
				verbose_level);
	}

	else if (Descr->f_sift_polynomials) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_sift_polynomials" << endl;
		}
		ring_theory::ring_theory_global R;

		R.sift_polynomials(F,
				Descr->sift_polynomials_r0,
				Descr->sift_polynomials_r1,
				verbose_level);
	}

	else if (Descr->f_mult_polynomials) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_mult_polynomials" << endl;
		}
		ring_theory::ring_theory_global R;

		R.mult_polynomials(F,
				Descr->mult_polynomials_r0,
				Descr->mult_polynomials_r1,
				verbose_level);
	}

	else if (Descr->f_polynomial_division_ranked) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_polynomial_division_ranked" << endl;
		}
		ring_theory::ring_theory_global R;

		R.polynomial_division_with_report(F,
				Descr->polynomial_division_r0,
				Descr->polynomial_division_r1,
				verbose_level);
	}


	else if (Descr->f_RREF_random_matrix) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_RREF_random_matrix" << endl;
		}
		linear_algebra::linear_algebra_global LA;
		int *A;
		int m, n;
		int i;

		m = Descr->RREF_random_matrix_m;
		n = Descr->RREF_random_matrix_n;

		orbiter_kernel_system::os_interface Os;

		A = NEW_int(m * n);
		for (i = 0; i < m * n; i++) {
			A[i] = Os.random_integer(F->q);
		}

		LA.RREF_demo(F,
				A,
				m,
				n,
				verbose_level);

		FREE_int(A);
	}

	else if (Descr->f_transversal) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_transversal" << endl;
		}
		geometry::geometry_global GG;

		GG.do_transversal(F,
				Descr->transversal_line_1_basis,
				Descr->transversal_line_2_basis,
				Descr->transversal_point,
				Descr->f_normalize_from_the_left,
				Descr->f_normalize_from_the_right,
				verbose_level);
	}
	else if (Descr->f_intersection_of_two_lines) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_intersection_of_two_lines" << endl;
		}
		geometry::geometry_global GG;

		GG.do_intersection_of_two_lines(F,
				Descr->line_1_basis,
				Descr->line_2_basis,
				Descr->f_normalize_from_the_left,
				Descr->f_normalize_from_the_right,
				verbose_level);
	}
	else if (Descr->f_inverse_isomorphism_klein_quadric) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_inverse_isomorphism_klein_quadric" << endl;
		}
		geometry::geometry_global GG;

		GG.do_inverse_isomorphism_klein_quadric(F,
				Descr->inverse_isomorphism_klein_quadric_matrix_A6,
				verbose_level);
	}
	else if (Descr->f_rank_point_in_PG) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_rank_point_in_PG" << endl;
		}
		geometry::geometry_global GG;

		GG.do_rank_points_in_PG(F,
				Descr->rank_point_in_PG_label,
				verbose_level);
	}

	else if (Descr->f_unrank_point_in_PG) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_unrank_point_in_PG" << endl;
		}
		geometry::geometry_global GG;

		GG.do_unrank_points_in_PG(F,
				Descr->unrank_point_in_PG_n,
				Descr->unrank_point_in_PG_text,
				verbose_level);
	}

	else if (Descr->f_field_reduction) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_field_reduction" << endl;
		}
		coding_theory::coding_theory_domain Coding;
		finite_field *Fq;

		Fq = NEW_OBJECT(finite_field);
		Fq->finite_field_init(Descr->field_reduction_q,
				FALSE /* f_without_tables */, verbose_level);
		Coding.field_reduction(F, Fq,
				Descr->field_reduction_label,
				Descr->field_reduction_m, Descr->field_reduction_n,
				Descr->field_reduction_text,
				verbose_level);

		FREE_OBJECT(Fq);

	}
	else if (Descr->f_parse_and_evaluate) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_parse_and_evaluate" << endl;
		}
		expression_parser::expression_parser_domain ED;

		ED.parse_and_evaluate(F,
				Descr->parse_name_of_formula,
				Descr->parse_text,
				Descr->parse_managed_variables,
				TRUE,
				Descr->parse_parameters,
				verbose_level);

	}


	else if (Descr->f_product_of) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_product_of" << endl;
		}
		int *data;
		int sz;
		int i, a, s;

		if (f_v) {
			cout << "-product_of " << Descr->product_of_elements
				<< endl;
		}

		Get_int_vector_from_label(Descr->product_of_elements,
				data, sz, verbose_level);
		s = 1;
		for (i = 0; i < sz; i++) {
			a = data[i];
			s = F->mult(s, a);
		}
		if (f_v) {
			cout << "the product is " << s << endl;
		}

	}
	else if (Descr->f_sum_of) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_sum_of" << endl;
		}
		int *data;
		int sz;
		int i, a, s;

		if (f_v) {
			cout << "-sum_of " << Descr->sum_of_elements
				<< endl;
		}

		Get_int_vector_from_label(Descr->sum_of_elements,
				data, sz, verbose_level);
		s = 1;
		for (i = 0; i < sz; i++) {
			a = data[i];
			s = F->add(s, a);
		}
		if (f_v) {
			cout << "the sum is " << s << endl;
		}

	}

	else if (Descr->f_negate) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_negate" << endl;
		}
		int *data;
		int sz;
		int i, a, s;

		if (f_v) {
			cout << "-negate " << Descr->negate_elements
				<< endl;
		}

		Get_int_vector_from_label(Descr->negate_elements,
				data, sz, verbose_level);
		for (i = 0; i < sz; i++) {
			a = data[i];
			s = F->negate(a);
			if (f_v) {
				cout << "the negative of " << a << " is " << s << endl;
			}

		}

	}

	else if (Descr->f_inverse) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_inverse" << endl;
		}
		int *data;
		int sz;
		int i, a, s;

		if (f_v) {
			cout << "-inverse " << Descr->inverse_elements
				<< endl;
		}

		Get_int_vector_from_label(Descr->inverse_elements,
				data, sz, verbose_level);
		for (i = 0; i < sz; i++) {
			a = data[i];
			s = F->negate(a);
			if (f_v) {
				cout << "the inverse of " << a << " is " << s << endl;
			}

		}

	}

	else if (Descr->f_power_map) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_power_map" << endl;
		}
		int *data;
		int sz;
		int i, a, s;

		if (f_v) {
			cout << "-power_map " << Descr->power_map_elements
				<< endl;
		}

		Get_int_vector_from_label(Descr->power_map_elements,
				data, sz, verbose_level);

		if (f_v) {
			cout << "a : a^k" << endl;
		}

		for (i = 0; i < sz; i++) {
			a = data[i];
			s = F->power(a, Descr->power_map_k);
			if (f_v) {
				cout << a << " : " << s << endl;
			}

		}

	}

	else if (Descr->f_evaluate) {

		if (f_v) {
			cout << "finite_field_activity::perform_activity f_evaluate" << endl;
		}

		expression_parser::expression_parser_domain ED;

		ED.evaluate(F,
				Descr->evaluate_formula_label,
				Descr->evaluate_parameters,
				verbose_level);

	}





	if (f_v) {
		cout << "finite_field_activity::perform_activity done" << endl;
	}

}






}}}


