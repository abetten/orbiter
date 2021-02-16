/*
 * finite_field_activity.cpp
 *
 *  Created on: Nov 4, 2020
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


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
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_activity::init" << endl;
	}

	finite_field_activity::Descr = Descr;
	F = NEW_OBJECT(finite_field);

	if (Descr->f_override_polynomial) {
		F->init_override_polynomial(Descr->q,
				Descr->override_polynomial, verbose_level);
	}
	else {
		F->finite_field_init(Descr->q, 0 /* verbose_level */);
	}


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

		algebra_global Algebra;

		Algebra.do_cheat_sheet_GF(F, verbose_level);
	}
	else if (Descr->f_all_rational_normal_forms) {

		algebra_global_with_action Algebra;

		Algebra.classes_GL(F, Descr->d,
				FALSE /* f_no_eigenvalue_one */, verbose_level);


	}
	else if (Descr->f_polynomial_division) {
		algebra_global Algebra;

		Algebra.polynomial_division(F,
				Descr->polynomial_division_A, Descr->polynomial_division_B,
				verbose_level);
	}
	else if (Descr->f_extended_gcd_for_polynomials) {
		algebra_global Algebra;

		Algebra.extended_gcd_for_polynomials(F,
				Descr->polynomial_division_A, Descr->polynomial_division_B,
				verbose_level);
	}

	else if (Descr->f_polynomial_mult_mod) {
		algebra_global Algebra;

		Algebra.polynomial_mult_mod(F,
				Descr->polynomial_mult_mod_A, Descr->polynomial_mult_mod_B,
				Descr->polynomial_mult_mod_M, verbose_level);
	}
	else if (Descr->f_Berlekamp_matrix) {
		algebra_global Algebra;

		Algebra.Berlekamp_matrix(F,
				Descr->Berlekamp_matrix_coeffs, verbose_level);

	}
	else if (Descr->f_normal_basis) {

		algebra_global Algebra;


		Algebra.compute_normal_basis(F, Descr->normal_basis_d, verbose_level);

	}
	else if (Descr->f_polynomial_find_roots) {

		algebra_global Algebra;



		Algebra.polynomial_find_roots(F,
				Descr->polynomial_find_roots_A,
				verbose_level);
	}

	else if (Descr->f_nullspace) {

		algebra_global Algebra;

		Algebra.do_nullspace(F, Descr->nullspace_m, Descr->nullspace_n,
				Descr->nullspace_text,
				Descr->f_normalize_from_the_left,
				Descr->f_normalize_from_the_right, verbose_level);
		FREE_OBJECT(F);
	}
	else if (Descr->f_RREF) {

		algebra_global Algebra;


		Algebra.do_RREF(F, Descr->RREF_m, Descr->RREF_n, Descr->RREF_text,
				Descr->f_normalize_from_the_left,
				Descr->f_normalize_from_the_right,
				verbose_level);

	}
	else if (Descr->f_weight_enumerator) {

		coding_theory_domain Codes;

		Codes.do_weight_enumerator(F,
				Descr->RREF_m, Descr->RREF_n, Descr->RREF_text,
				Descr->f_normalize_from_the_left,
				Descr->f_normalize_from_the_right,
				verbose_level);
	}
	else if (Descr->f_trace) {

		algebra_global Algebra;

		Algebra.do_trace(F, verbose_level);
	}
	else if (Descr->f_norm) {

		algebra_global Algebra;

		Algebra.do_norm(F, verbose_level);
	}

	else if (Descr->f_make_table_of_irreducible_polynomials) {

		algebra_global Algebra;

		Algebra.do_make_table_of_irreducible_polynomials(
				Descr->make_table_of_irreducible_polynomials_degree,
				F, verbose_level);

	}
	else if (Descr->f_EC_Koblitz_encoding) {

		cryptography_domain Crypto;

		Crypto.do_EC_Koblitz_encoding(F, Descr->EC_b, Descr->EC_c,
				Descr->EC_s, Descr->EC_pt_text, Descr->EC_message, verbose_level);
	}
	else if (Descr->f_EC_points) {

		cryptography_domain Crypto;

		Crypto.do_EC_points(F, Descr->EC_b, Descr->EC_c, verbose_level);
	}
	else if (Descr->f_EC_add) {

		cryptography_domain Crypto;

		Crypto.do_EC_add(F, Descr->EC_b, Descr->EC_c,
				Descr->EC_pt1_text, Descr->EC_pt2_text, verbose_level);
	}
	else if (Descr->f_EC_cyclic_subgroup) {

		cryptography_domain Crypto;

		Crypto.do_EC_cyclic_subgroup(F, Descr->EC_b, Descr->EC_c,
				Descr->EC_pt_text, verbose_level);
	}
	else if (Descr->f_EC_multiple_of) {

		cryptography_domain Crypto;

		Crypto.do_EC_multiple_of(F, Descr->EC_b, Descr->EC_c,
				Descr->EC_pt_text, Descr->EC_multiple_of_n, verbose_level);
	}
	else if (Descr->f_EC_discrete_log) {

		cryptography_domain Crypto;

		Crypto.do_EC_discrete_log(F, Descr->EC_b, Descr->EC_c, Descr->EC_pt_text,
				Descr->EC_discrete_log_pt_text, verbose_level);
	}
	else if (Descr->f_EC_baby_step_giant_step) {

		cryptography_domain Crypto;

		Crypto.do_EC_baby_step_giant_step(F, Descr->EC_b, Descr->EC_c,
				Descr->EC_bsgs_G, Descr->EC_bsgs_N, Descr->EC_bsgs_cipher_text,
				verbose_level);
	}
	else if (Descr->f_EC_baby_step_giant_step_decode) {

		cryptography_domain Crypto;

		Crypto.do_EC_baby_step_giant_step_decode(F, Descr->EC_b, Descr->EC_c,
				Descr->EC_bsgs_A, Descr->EC_bsgs_N,
				Descr->EC_bsgs_cipher_text, Descr->EC_bsgs_keys,
				verbose_level);
	}
	else if (Descr->f_NTRU_encrypt) {

		cryptography_domain Crypto;

		Crypto.NTRU_encrypt(Descr->NTRU_encrypt_N, Descr->NTRU_encrypt_p, F,
				Descr->NTRU_encrypt_H, Descr->NTRU_encrypt_R,
				Descr->NTRU_encrypt_Msg,
				verbose_level);
	}
	else if (Descr->f_polynomial_center_lift) {

		cryptography_domain Crypto;

		Crypto.polynomial_center_lift(Descr->polynomial_center_lift_A, F,
				verbose_level);
	}
	else if (Descr->f_polynomial_reduce_mod_p) {

		cryptography_domain Crypto;

		Crypto.polynomial_reduce_mod_p(Descr->polynomial_reduce_mod_p_A, F,
				verbose_level);
	}
	else if (Descr->f_cheat_sheet_PG) {

		geometry_global Geo;

		layered_graph_draw_options *O;


		if (!The_Orbiter_session->f_draw_options) {
			cout << "please use option -draw_options .. -end" << endl;
			exit(1);
		}
		O = The_Orbiter_session->draw_options;

		Geo.do_cheat_sheet_PG(F, O, Descr->cheat_sheet_PG_n,
				verbose_level);
	}
	else if (Descr->f_cheat_sheet_Gr) {

		geometry_global Geo;

		Geo.do_cheat_sheet_Gr(F, Descr->cheat_sheet_Gr_n, Descr->cheat_sheet_Gr_k,
				verbose_level);
	}
	else if (Descr->f_cheat_sheet_hermitian) {

		geometry_global Geo;

		Geo.do_cheat_sheet_hermitian(F,
				Descr->cheat_sheet_hermitian_projective_dimension,
				verbose_level);
	}
	else if (Descr->f_cheat_sheet_desarguesian_spread) {

		geometry_global Geo;

		if (F_secondary == NULL) {
			cout << "F_secondary == NULL" << endl;
			exit(1);
		}
		Geo.do_create_desarguesian_spread(F, F_secondary,
				Descr->cheat_sheet_desarguesian_spread_m,
				verbose_level);
	}
	else if (Descr->f_find_CRC_polynomials) {

		algebra_global Algebra;

		Algebra.find_CRC_polynomials(F,
				Descr->find_CRC_polynomials_nb_errors,
				Descr->find_CRC_polynomials_information_bits,
				Descr->find_CRC_polynomials_check_bits,
				verbose_level);
	}

	else if (Descr->f_sift_polynomials) {

		algebra_global Algebra;

		Algebra.sift_polynomials(F,
				Descr->sift_polynomials_r0,
				Descr->sift_polynomials_r1,
				verbose_level);
	}

	else if (Descr->f_RREF_random_matrix) {

		algebra_global Algebra;

		int *A;
		int m, n;
		int i;

		m = Descr->RREF_random_matrix_m;
		n = Descr->RREF_random_matrix_n;

		os_interface Os;

		A = NEW_int(m * n);
		for (i = 0; i < m * n; i++) {
			A[i] = Os.random_integer(F->q);
		}

		Algebra.RREF_demo(F, A,
				m,
				n,
				verbose_level);

		FREE_int(A);
	}

	else if (Descr->f_decomposition_by_element) {

		algebra_global_with_action Algebra;

		Algebra.do_cheat_sheet_for_decomposition_by_element_PG(F,
				Descr->decomposition_by_element_n,
				Descr->decomposition_by_element_power,
				Descr->decomposition_by_element_data,
				Descr->decomposition_by_element_fname_base,
				verbose_level);
	}
	else if (Descr->f_transversal) {

		geometry_global GG;

		GG.do_transversal(F,
				Descr->transversal_line_1_basis,
				Descr->transversal_line_2_basis,
				Descr->transversal_point,
				Descr->f_normalize_from_the_left,
				Descr->f_normalize_from_the_right,
				verbose_level);
	}
	else if (Descr->f_intersection_of_two_lines) {

		geometry_global GG;

		GG.do_intersection_of_two_lines(F,
				Descr->line_1_basis,
				Descr->line_2_basis,
				Descr->f_normalize_from_the_left,
				Descr->f_normalize_from_the_right,
				verbose_level);
	}
	else if (Descr->f_move_two_lines_in_hyperplane_stabilizer) {

		geometry_global GG;


		GG.do_move_two_lines_in_hyperplane_stabilizer(
				F,
				Descr->line1_from, Descr->line2_from,
				Descr->line1_to, Descr->line2_to, verbose_level);
	}
	else if (Descr->f_move_two_lines_in_hyperplane_stabilizer_text) {

		geometry_global GG;


		GG.do_move_two_lines_in_hyperplane_stabilizer_text(
				F,
				Descr->line1_from_text, Descr->line2_from_text,
				Descr->line1_to_text, Descr->line2_to_text,
				verbose_level);
	}
	else if (Descr->f_study_surface) {

		algebra_global_with_action Algebra;

		Algebra.do_study_surface(F,
				Descr->study_surface_nb,
				verbose_level);
	}
	else if (Descr->f_inverse_isomorphism_klein_quadric) {

		geometry_global GG;

		GG.do_inverse_isomorphism_klein_quadric(F,
				Descr->inverse_isomorphism_klein_quadric_matrix_A6,
				verbose_level);
	}
	else if (Descr->f_rank_point_in_PG) {

		geometry_global GG;

		GG.do_rank_point_in_PG(F,
				Descr->rank_point_in_PG_n,
				Descr->rank_point_in_PG_text,
				verbose_level);
	}
	else if (Descr->f_rank_point_in_PG_given_as_pairs) {

		geometry_global GG;

		GG.do_rank_point_in_PG_given_as_pairs(F,
				Descr->rank_point_in_PG_given_as_pairs_n,
				Descr->rank_point_in_PG_given_as_pairs_text,
				verbose_level);
	}
	else if (Descr->f_eigenstuff) {

		algebra_global_with_action Algebra;

		Algebra.do_eigenstuff_with_coefficients(
				F,
				Descr->eigenstuff_n,
				Descr->eigenstuff_coeffs,
				verbose_level);
	}
	else if (Descr->f_eigenstuff_from_file) {

		algebra_global_with_action Algebra;

		Algebra.do_eigenstuff_from_file(
				F,
				Descr->eigenstuff_n,
				Descr->eigenstuff_fname,
				verbose_level);
	}
	else if (Descr->f_parse) {

		expression_parser Parser;
		syntax_tree *tree;
		int i;

		tree = NEW_OBJECT(syntax_tree);

		cout << "Formula " << Descr->parse_name_of_formula << " is " << Descr->parse_text << endl;
		cout << "Managed variables: " << Descr->parse_managed_variables << endl;

		const char *p = Descr->parse_managed_variables.c_str();
		char str[1000];

		while (TRUE) {
			if (!s_scan_token_comma_separated(&p, str)) {
				break;
			}
			string var;

			var.assign(str);
			cout << "adding managed variable " << var << endl;

			tree->managed_variables.push_back(var);
			tree->f_has_managed_variables = TRUE;

		}

		int nb_vars;

		nb_vars = tree->managed_variables.size();

		cout << "Managed variables: " << endl;
		for (i = 0; i < nb_vars; i++) {
			cout << i << " : " << tree->managed_variables[i] << endl;
		}


		cout << "Starting to parse " << Descr->parse_name_of_formula << endl;
		Parser.parse(tree, Descr->parse_text, verbose_level);
		cout << "Parsing " << Descr->parse_name_of_formula << " finished" << endl;


		cout << "Syntax tree:" << endl;
		//tree->print(cout);

		std::string fname;
		fname.assign(Descr->parse_name_of_formula);
		fname.append(".gv");

		{
			std::ofstream ost(fname);
			tree->Root->export_graphviz(Descr->parse_name_of_formula, ost);
		}

		int ret, degree;
		ret = tree->is_homogeneous(degree);
		if (ret) {
			cout << "homogeneous of degree " << degree << endl;

			homogeneous_polynomial_domain *Poly;

			Poly = NEW_OBJECT(homogeneous_polynomial_domain);

			if (f_v) {
				cout << "before Poly->init" << endl;
			}
			Poly->init(F,
					nb_vars /* nb_vars */, degree,
					FALSE /* f_init_incidence_structure */,
					t_PART,
					verbose_level);
			if (f_v) {
				cout << "after Poly->init" << endl;
			}

			syntax_tree_node **Subtrees;
			int nb_monomials;

			nb_monomials = Poly->get_nb_monomials();

			tree->split_by_monomials(Poly, Subtrees, verbose_level);

			for (i = 0; i < nb_monomials; i++) {
				cout << "Monomial " << i << " : ";
				if (Subtrees[i]) {
					Subtrees[i]->print_expression(cout);
					cout << " * ";
					Poly->print_monomial(cout, i);
					cout << endl;
				}
				else {
					cout << "no subtree" << endl;
				}
			}
			if (Descr->f_evaluate) {

				cout << "before evaluate" << endl;

				const char *p = Descr->evaluate_text.c_str();
				//char str[1000];

				std::map<std::string, std::string> symbol_table;
				//vector<string> symbols;
				//vector<string> values;

				while (TRUE) {
					if (!s_scan_token_comma_separated(&p, str)) {
						break;
					}
					string assignment;
					int len;

					assignment.assign(str);
					len = strlen(str);

					std::size_t found;

					found = assignment.find('=');
					if (found == std::string::npos) {
						cout << "did not find '=' in variable assignment" << endl;
						exit(1);
					}
					std::string symb = assignment.substr (0, found);
					std::string val = assignment.substr (found + 1, len - found - 1);



					cout << "adding symbol " << symb << " = " << val << endl;

					symbol_table[symb] = val;
					//symbols.push_back(symb);
					//values.push_back(val);

				}

#if 0
				cout << "symbol table:" << endl;
				for (i = 0; i < symbol_table.size(); i++) {
					cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
				}
#endif
				int a;
				int *Values;

				Values = NEW_int(nb_monomials);

				for (i = 0; i < nb_monomials; i++) {
					cout << "Monomial " << i << " : ";
					if (Subtrees[i]) {
						//Subtrees[i]->print_expression(cout);
						a = Subtrees[i]->evaluate(symbol_table, F, verbose_level);
						Values[i] = a;
						cout << a << " * ";
						Poly->print_monomial(cout, i);
						cout << endl;
					}
					else {
						cout << "no subtree" << endl;
						Values[i] = 0;
					}
				}
				cout << "evaluated polynomial:" << endl;
				for (i = 0; i < nb_monomials; i++) {
					cout << Values[i] << " * ";
					Poly->print_monomial(cout, i);
					cout << endl;
				}
				cout << "coefficient vector: ";
				int_vec_print(cout, Values, nb_monomials);
				cout << endl;

			}


			FREE_OBJECT(Poly);
		}
		else {
			cout << "not homogeneous" << endl;
		}

	}

	if (f_v) {
		cout << "finite_field_activity::perform_activity done" << endl;
	}

}





}}

