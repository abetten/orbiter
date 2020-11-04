/*
 * interface_algebra.cpp
 *
 *  Created on: Apr 26, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace interfaces {


interface_algebra::interface_algebra()
{
	f_linear_group = FALSE;
	Linear_group_description = NULL;
	F = NULL;
	LG = NULL;

	f_group_theoretic_activity = FALSE;
	Group_theoretic_activity_description = NULL;
	f_cheat_sheet_GF = FALSE;
	q = 0;
	f_all_rational_normal_forms = FALSE;
	d = 0;
	f_search_for_primitive_polynomial_in_range = FALSE;
	p_min = 0;
	p_max = 0;
	deg_min = 0;
	deg_max = 0;
	f_make_table_of_irreducible_polynomials = FALSE;
	deg = 0;
	q = 0;
	f_character_table_symmetric_group = FALSE;
	f_make_A5_in_PSL_2_q = FALSE;
	f_eigenstuff = FALSE;
	f_eigenstuff_from_file = FALSE;
	eigenstuff_n = 0;
	eigenstuff_q = 0;
	//eigenstuff_coeffs = NULL;
	//eigenstuff_fname = NULL;
	f_young_symmetrizer = FALSE;
	young_symmetrizer_n = 0;
	f_young_symmetrizer_sym_4 = FALSE;
	f_poset_classification_control = FALSE;
	Control = NULL;
	f_polynomial_division = FALSE;
	polynomial_division_q = 0;
	//polynomial_division_A;
	//polynomial_division_B;
	f_extended_gcd_for_polynomials = FALSE;

	f_polynomial_mult_mod = FALSE;
	polynomial_mult_mod_q = 0;
	//std::string polynomial_mult_mod_A;
	//std::string polynomial_mult_mod_B;
	//std::string polynomial_mult_mod_M;

	f_Berlekamp_matrix = FALSE;
	Berlekamp_matrix_q = 0;
	//Berlekamp_matrix_coeffs;

	f_normal_basis = FALSE;
	normal_basis_q = 0;
	normal_basis_d = 0;

	f_normalize_from_the_right = FALSE;
	f_normalize_from_the_left = FALSE;


	f_nullspace = FALSE;
	nullspace_q = 0;
	nullspace_m = 0;
	nullspace_n = 0;
	//nullspace_text = NULL;

	f_RREF = FALSE;
	RREF_q = 0;
	RREF_m = 0;
	RREF_n = 0;
	//cout << "interface_cryptography::interface_cryptography 3" << endl;
	//RREF_text = NULL;
	f_weight_enumerator = FALSE;
	f_trace = FALSE;
	trace_q = 0;
	f_norm = FALSE;
	norm_q = 0;
	f_count_subprimitive = FALSE;
	count_subprimitive_Q_max = 0;
	count_subprimitive_H_max = 0;
	f_equivalence_class_of_fractions = FALSE;
	equivalence_class_of_fractions_N = 0;
}


void interface_algebra::print_help(int argc,
		const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-linear_group") == 0) {
		cout << "-linear_group <description>" << endl;
	}
	else if (strcmp(argv[i], "-group_theoretic_activity") == 0) {
		cout << "-group_theoretic_activity <description>" << endl;
	}
	else if (strcmp(argv[i], "-cheat_sheet_GF") == 0) {
		cout << "-cheat_sheet_GF <int : q>" << endl;
	}
	else if (strcmp(argv[i], "-all_rational_normal_forms") == 0) {
		cout << "-all_rational_normal_forms <int : d> <int : q>" << endl;
	}
	else if (strcmp(argv[i], "-override_polynomial") == 0) {
		cout << "-override_polynomial <polynomial in decimal>" << endl;
	}
	else if (strcmp(argv[i], "-search_for_primitive_polynomial_in_range") == 0) {
		cout << "-search_for_primitive_polynomial_in_range <int : p_min> <int : p_max> <int : deg_min> <int : deg_max> " << endl;
	}
	else if (strcmp(argv[i], "-make_table_of_irreducible_polynomials") == 0) {
		cout << "-table_of_irreducible_polynomials <int : deg> <int : q> " << endl;
	}
	else if (strcmp(argv[i], "-character_table_symmetric_group") == 0) {
		cout << "-character_table_symmetric_group <int : deg> " << endl;
	}
	else if (strcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		cout << "-make_A5_in_PSL_2_q <int : q> " << endl;
	}
	else if (strcmp(argv[i], "-eigenstuff") == 0) {
		cout << "-eigenstuff <int : m> <int : q> <string : coeffs> " << endl;
	}
	else if (strcmp(argv[i], "-eigenstuff_from_file") == 0) {
		cout << "-eigenstuff_from_file <int : m> <int : q>  <string : fname> " << endl;
	}
	else if (strcmp(argv[i], "-young_symmetrizer") == 0) {
		cout << "-young_symmetrizer <int : n>" << endl;
	}
	else if (strcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		cout << "-young_symmetrizer_sym_4 " << endl;
	}
	else if (strcmp(argv[i], "-poset_classification_control") == 0) {
		cout << "-poset_classification_control <description>" << endl;
	}
	else if (strcmp(argv[i], "-polynomial_division") == 0) {
		cout << "-polynomial_division <int : q> <string : A> <string : B>" << endl;
	}
	else if (strcmp(argv[i], "-extended_gcd_for_polynomials") == 0) {
		cout << "-extended_gcd_for_polynomials <int : q> <string : A> <string : B>" << endl;
	}
	else if (strcmp(argv[i], "-polynomial_mult_mod") == 0) {
		cout << "-polynomial_mult_mod <int : q> <string : A> <string : B> <string : M>" << endl;
	}
	else if (strcmp(argv[i], "-Berlekamp_matrix") == 0) {
		cout << "-Berlekamp_matrix <int : q> <string : polynomial coefficients>" << endl;
	}
	else if (strcmp(argv[i], "-normal_basis") == 0) {
		cout << "-normal_basis <int : q> <int : degree>" << endl;
	}
	else if (strcmp(argv[i], "-normalize_from_the_right") == 0) {
		cout << "-normalize_from_the_right" << endl;
	}
	else if (strcmp(argv[i], "-normalize_from_the_left") == 0) {
		cout << "-normalize_from_the_left" << endl;
	}
	else if (strcmp(argv[i], "-nullspace") == 0) {
		cout << "-nullspace <int : q> <int : m> <int : n> <string : coeff_matrix>" << endl;
	}
	else if (strcmp(argv[i], "-RREF") == 0) {
		cout << "-RREF <int : q> <int : m> <int : n> <string : coeff_matrix>" << endl;
	}
	else if (strcmp(argv[i], "-weight_enumerator") == 0) {
		cout << "-weight_enumerator <int : q> <int : m> <int : n> <string : coeff_matrix>" << endl;
	}
	else if (strcmp(argv[i], "-trace") == 0) {
		cout << "-trace <int : q>" << endl;
	}
	else if (strcmp(argv[i], "-norm") == 0) {
		cout << "-norm <int : q>" << endl;
	}
	else if (strcmp(argv[i], "-count_subprimitive") == 0) {
		cout << "-count_subprimitive <int : Q_max> <int : H_max>" << endl;
	}
	else if (strcmp(argv[i], "-equivalence_class_of_fractions") == 0) {
		cout << "-equivalence_class_of_fractions <int : N> " << endl;
	}
}

int interface_algebra::recognize_keyword(int argc,
		const char **argv, int i, int verbose_level)
{
	if (i >= argc) {
		return false;
	}
	if (strcmp(argv[i], "-linear_group") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-group_theoretic_activity") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-cheat_sheet_GF") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-all_rational_normal_forms") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-override_polynomial") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-search_for_primitive_polynomial_in_range") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-make_table_of_irreducible_polynomials") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-character_table_symmetric_group") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-eigenstuff") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-eigenstuff_from_file") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-young_symmetrizer") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-poset_classification_control") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-polynomial_division") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-extended_gcd_for_polynomials") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-polynomial_mult_mod") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-Berlekamp_matrix") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-normal_basis") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-normalize_from_the_right") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-normalize_from_the_left") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-nullspace") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-RREF") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-weight_enumerator") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-trace") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-norm") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-count_subprimitive") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-equivalence_class_of_fractions") == 0) {
		return true;
	}
	return false;
}

void interface_algebra::read_arguments(int argc,
		const char **argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_algebra::read_arguments" << endl;

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-linear_group") == 0) {
			f_linear_group = TRUE;
			Linear_group_description = NEW_OBJECT(linear_group_description);
			cout << "reading -linear_group" << endl;
			i += Linear_group_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-linear_group" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-group_theoretic_activities") == 0) {
			f_group_theoretic_activity = TRUE;
			Group_theoretic_activity_description =
					NEW_OBJECT(group_theoretic_activity_description);
			cout << "reading -group_theoretic_activities" << endl;
			i += Group_theoretic_activity_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-group_theoretic_activities" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-poset_classification_control") == 0) {
			f_poset_classification_control = TRUE;
			Control = NEW_OBJECT(poset_classification_control);
			cout << "reading -poset_classification_control" << endl;
			i += Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -poset_classification_control " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-cheat_sheet_GF") == 0) {
			f_cheat_sheet_GF = TRUE;
			q = atoi(argv[++i]);
			cout << "-cheat_sheet_GF " << q << endl;
		}
		else if (strcmp(argv[i], "-all_rational_normal_forms") == 0) {
			f_all_rational_normal_forms = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-f_all_rational_normal_forms " << d << " " << q << endl;
		}
		else if (strcmp(argv[i], "-search_for_primitive_polynomial_in_range") == 0) {
			f_search_for_primitive_polynomial_in_range = TRUE;
			p_min = atoi(argv[++i]);
			p_max = atoi(argv[++i]);
			deg_min = atoi(argv[++i]);
			deg_max = atoi(argv[++i]);
			cout << "-search_for_primitive_polynomial_in_range " << p_min
					<< " " << p_max << " " << deg_min << " " << deg_max << " " << endl;
		}
		else if (strcmp(argv[i], "-make_table_of_irreducible_polynomials") == 0) {
			f_make_table_of_irreducible_polynomials = TRUE;
			deg = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-make_table_of_irreducible_polynomials " << deg << " " << q << " " << endl;
		}
		else if (strcmp(argv[i], "-character_table_symmetric_group") == 0) {
			f_character_table_symmetric_group = TRUE;
			deg = atoi(argv[++i]);
			cout << "-character_table_symmetric_group " << deg << endl;
		}
		else if (strcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
			f_make_A5_in_PSL_2_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-make_A5_in_PSL_2_q " << q << endl;
		}
		else if (strcmp(argv[i], "-eigenstuff") == 0) {
			f_eigenstuff = TRUE;
			eigenstuff_n = atoi(argv[++i]);
			eigenstuff_q = atoi(argv[++i]);
			eigenstuff_coeffs.assign(argv[++i]);
			cout << "-eigenstuff " << eigenstuff_n
					<< " " << eigenstuff_q << " " << eigenstuff_coeffs << endl;
		}
		else if (strcmp(argv[i], "-eigenstuff_matrix_from_file") == 0) {
			f_eigenstuff_from_file = TRUE;
			eigenstuff_n = atoi(argv[++i]);
			eigenstuff_q = atoi(argv[++i]);
			eigenstuff_fname.assign(argv[++i]);
			cout << "-eigenstuff_from_file " << eigenstuff_n
					<< " " << eigenstuff_q << " " << eigenstuff_fname << endl;
		}

		else if (strcmp(argv[i], "-young_symmetrizer") == 0) {
			f_young_symmetrizer = TRUE;
			young_symmetrizer_n = atoi(argv[++i]);
			cout << "-young_symmetrizer " << young_symmetrizer_n << endl;
		}
		else if (strcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
			f_young_symmetrizer_sym_4 = TRUE;
			cout << "-young_symmetrizer_sym_4 " << endl;
		}
		else if (strcmp(argv[i], "-polynomial_division") == 0) {
			f_polynomial_division = TRUE;
			polynomial_division_q = atoi(argv[++i]);
			polynomial_division_A.assign(argv[++i]);
			polynomial_division_B.assign(argv[++i]);
			cout << "-polynomial_division " << polynomial_division_q << " " << polynomial_division_A << " " << polynomial_division_B << endl;
		}
		else if (strcmp(argv[i], "-extended_gcd_for_polynomials") == 0) {
			f_extended_gcd_for_polynomials = TRUE;
			polynomial_division_q = atoi(argv[++i]);
			polynomial_division_A.assign(argv[++i]);
			polynomial_division_B.assign(argv[++i]);
			cout << "-extended_gcd_for_polynomials " << polynomial_division_q << " " << polynomial_division_A << " " << polynomial_division_B << endl;
		}
		else if (strcmp(argv[i], "-polynomial_mult_mod") == 0) {
			f_polynomial_mult_mod = TRUE;
			polynomial_mult_mod_q = atoi(argv[++i]);
			polynomial_mult_mod_A.assign(argv[++i]);
			polynomial_mult_mod_B.assign(argv[++i]);
			polynomial_mult_mod_M.assign(argv[++i]);
			cout << "-polynomial_mult_mod " << polynomial_mult_mod_q
					<< " " << polynomial_mult_mod_A
					<< " " << polynomial_mult_mod_B
					<< " " << polynomial_mult_mod_M << endl;
		}
		else if (strcmp(argv[i], "-Berlekamp_matrix") == 0) {
			f_Berlekamp_matrix = TRUE;
			Berlekamp_matrix_q = atoi(argv[++i]);
			Berlekamp_matrix_coeffs.assign(argv[++i]);
			cout << "-Berlekamp_matrix " << Berlekamp_matrix_q
					<< " " << Berlekamp_matrix_coeffs << endl;
		}
		else if (strcmp(argv[i], "-normal_basis") == 0) {
			f_normal_basis = TRUE;
			normal_basis_q = atoi(argv[++i]);
			normal_basis_d = atoi(argv[++i]);
			cout << "-normal_basis " << normal_basis_q
					<< " " << normal_basis_d << endl;
		}
		else if (strcmp(argv[i], "-normalize_from_the_right") == 0) {
			f_normalize_from_the_right = TRUE;
			cout << "-normalize_from_the_right " << endl;
		}
		else if (strcmp(argv[i], "-normalize_from_the_left") == 0) {
			f_normalize_from_the_left = TRUE;
			cout << "-normalize_from_the_left " << endl;
		}
		else if (strcmp(argv[i], "-nullspace") == 0) {
			f_nullspace = TRUE;
			nullspace_q = atoi(argv[++i]);
			nullspace_m = atoi(argv[++i]);
			nullspace_n = atoi(argv[++i]);
			nullspace_text.assign(argv[++i]);
			cout << "-nullspace " << nullspace_q
					<< " " << nullspace_m << " " << nullspace_n << " " << nullspace_text << endl;
		}
		else if (strcmp(argv[i], "-RREF") == 0) {
			f_RREF = TRUE;
			RREF_q = atoi(argv[++i]);
			RREF_m = atoi(argv[++i]);
			RREF_n = atoi(argv[++i]);
			RREF_text.assign(argv[++i]);
			cout << "-RREF " << RREF_q
					<< " " << RREF_m << " " << RREF_n << " " << RREF_text << endl;
		}
		else if (strcmp(argv[i], "-weight_enumerator") == 0) {
			f_weight_enumerator = TRUE;
			RREF_q = atoi(argv[++i]);
			RREF_m = atoi(argv[++i]);
			RREF_n = atoi(argv[++i]);
			RREF_text = argv[++i];
			cout << "-weight_enumerator " << RREF_q
					<< " " << RREF_m << " " << RREF_n << " " << RREF_text << endl;
		}
		else if (strcmp(argv[i], "-trace") == 0) {
			f_trace = TRUE;
			trace_q = atoi(argv[++i]);
			cout << "-trace " << trace_q
					<< endl;
		}
		else if (strcmp(argv[i], "-norm") == 0) {
			f_norm = TRUE;
			norm_q = atoi(argv[++i]);
			cout << "-norm " << norm_q
					<< endl;
		}
		else if (strcmp(argv[i], "-count_subprimitive") == 0) {
			f_count_subprimitive = TRUE;
			count_subprimitive_Q_max = atoi(argv[++i]);
			count_subprimitive_H_max = atoi(argv[++i]);
			cout << "-count_subprimitive " << count_subprimitive_Q_max
					<< " " << count_subprimitive_H_max
					<< endl;
		}
		else if (strcmp(argv[i], "-equivalence_class_of_fractions") == 0) {
			f_equivalence_class_of_fractions = TRUE;
			equivalence_class_of_fractions_N = atoi(argv[++i]);
			cout << "-equivalence_class_of_fractions " << equivalence_class_of_fractions_N
					<< endl;
		}
	}
}


void interface_algebra::worker(orbiter_session *Session, int verbose_level)
{
	if (f_linear_group) {
		do_linear_group(Linear_group_description, verbose_level);
	}

	else if (f_cheat_sheet_GF) {
		do_cheat_sheet_GF(q,
				Session->f_override_polynomial, Session->override_polynomial,
				verbose_level);
	}
	else if (f_all_rational_normal_forms) {
		do_all_rational_normal_forms(d, q,
				Session->f_override_polynomial, Session->override_polynomial,
				verbose_level);
	}
	else if (f_search_for_primitive_polynomial_in_range) {
		do_search_for_primitive_polynomial_in_range(p_min, p_max, deg_min, deg_max, verbose_level);
	}
	else if (f_make_table_of_irreducible_polynomials) {
		do_make_table_of_irreducible_polynomials(deg, q, verbose_level);
	}
	else if (f_character_table_symmetric_group) {
		do_character_table_symmetric_group(deg, verbose_level);
	}
	else if (f_make_A5_in_PSL_2_q) {
		do_make_A5_in_PSL_2_q(q, verbose_level);
	}
	else if (f_eigenstuff) {
		do_eigenstuff(eigenstuff_n, eigenstuff_q, eigenstuff_coeffs, verbose_level);
	}
	else if (f_eigenstuff_from_file) {
		do_eigenstuff_from_file(eigenstuff_n, eigenstuff_q, eigenstuff_fname, verbose_level);
	}
	else if (f_young_symmetrizer) {
		algebra_global_with_action Algebra;

		Algebra.young_symmetrizer(young_symmetrizer_n, verbose_level);
	}
	else if (f_young_symmetrizer_sym_4) {
		algebra_global_with_action Algebra;

		Algebra.young_symmetrizer_sym_4(verbose_level);
	}
	else if (f_polynomial_division) {
		algebra_global Algebra;

		Algebra.polynomial_division(polynomial_division_q,
				polynomial_division_A, polynomial_division_B, verbose_level);
	}
	else if (f_extended_gcd_for_polynomials) {
		algebra_global Algebra;

		Algebra.extended_gcd_for_polynomials(polynomial_division_q,
				polynomial_division_A, polynomial_division_B, verbose_level);
	}

	else if (f_polynomial_mult_mod) {
		algebra_global Algebra;

		Algebra.polynomial_mult_mod(polynomial_mult_mod_q,
				polynomial_mult_mod_A, polynomial_mult_mod_B,
				polynomial_mult_mod_M, verbose_level);
	}
	else if (f_Berlekamp_matrix) {
		algebra_global Algebra;

		Algebra.Berlekamp_matrix(Berlekamp_matrix_q,
				Berlekamp_matrix_coeffs, verbose_level);
	}
	else if (f_normal_basis) {
		algebra_global Algebra;

		F = NEW_OBJECT(finite_field);

		F->init(normal_basis_q, 0);


		Algebra.compute_normal_basis(F, normal_basis_d, verbose_level);

		FREE_OBJECT(F);
	}
	else if (f_nullspace) {

		algebra_global Algebra;

		Algebra.do_nullspace(nullspace_q, nullspace_m, nullspace_n,
				nullspace_text,
				f_normalize_from_the_left,
				f_normalize_from_the_right, verbose_level);
	}
	else if (f_RREF) {

		algebra_global Algebra;


		Algebra.do_RREF(RREF_q, RREF_m, RREF_n, RREF_text,
				f_normalize_from_the_left,
				f_normalize_from_the_right,
				verbose_level);
	}
	else if (f_weight_enumerator) {

		algebra_global Algebra;

		Algebra.do_weight_enumerator(RREF_q, RREF_m, RREF_n, RREF_text,
				f_normalize_from_the_left,
				f_normalize_from_the_right,
				verbose_level);
	}
	else if (f_trace) {

		algebra_global Algebra;

		Algebra.do_trace(trace_q, verbose_level);
	}
	else if (f_norm) {

		algebra_global Algebra;

		Algebra.do_norm(norm_q, verbose_level);
	}
	else if (f_count_subprimitive) {
		algebra_global AG;
		AG.count_subprimitive(count_subprimitive_Q_max, count_subprimitive_H_max);
	}
	else if (f_equivalence_class_of_fractions) {
		algebra_global Algebra;

		Algebra.do_equivalence_class_of_fractions(equivalence_class_of_fractions_N, verbose_level);
	}



}

void interface_algebra::do_eigenstuff(
		int n, int q, std::string &coeffs_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_eigenstuff" << endl;
	}
	int *Data;
	int len;

	int_vec_scan(coeffs_text, Data, len);
	if (len != n * n) {
		cout << "len != n * n " << len << endl;
		exit(1);
	}

	algebra_global_with_action A;

	A.do_eigenstuff(q, n, Data, verbose_level);

	FREE_int(Data);
	if (f_v) {
		cout << "interface_algebra::do_eigenstuff done" << endl;
	}
}

void interface_algebra::do_eigenstuff_from_file(
		int n, int q, std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_eigenstuff_from_file" << endl;
	}

	file_io Fio;
	int *Data;
	int mtx_m, mtx_n;

	Fio.int_matrix_read_csv(fname, Data, mtx_m, mtx_n, verbose_level - 1);
	if (mtx_m != n) {
		cout << "mtx_m != n" << endl;
		exit(1);
	}
	if (mtx_n != n) {
		cout << "mtx_n != n" << endl;
		exit(1);
	}

	algebra_global_with_action A;

	A.do_eigenstuff(q, n, Data, verbose_level);


	if (f_v) {
		cout << "interface_algebra::do_eigenstuff_from_file done" << endl;
	}
}

void interface_algebra::do_linear_group(
		linear_group_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_linear_group" << endl;
	}


	F = NEW_OBJECT(finite_field);

	if (Descr->f_override_polynomial) {
		cout << "creating finite field of order q=" << Descr->input_q
				<< " using override polynomial " << Descr->override_polynomial << endl;
		F->init_override_polynomial(Descr->input_q,
				Descr->override_polynomial, verbose_level - 3);
	}
	else {
		cout << "interface_algebra::do_linear_group creating finite field "
				"of order q=" << Descr->input_q
				<< " using the default polynomial (if necessary)" << endl;
		F->init(Descr->input_q, 0);
	}

	Descr->F = F;
	//q = Descr->input_q;



	LG = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "interface_algebra::do_linear_group before LG->init, "
				"creating the group" << endl;
		}

	LG->init(Descr, verbose_level - 5);

	if (f_v) {
		cout << "interface_algebra::do_linear_group after LG->init" << endl;
		}

	action *A;

	A = LG->A2;

	cout << "interface_algebra::do_linear_group created group " << A->label << endl;



	if (LG->f_has_nice_gens) {
		cout << "interface_algebra::do_linear_group we have nice generators, they are:" << endl;
		LG->nice_gens->print(cout);
		cout << "$$" << endl;

		int i;

		for (i = 0; i < LG->nice_gens->len; i++) {
			//cout << "Generator " << i << " / " << gens->len
			// << " is:" << endl;
			A->element_print_latex(LG->nice_gens->ith(i), cout);
			if (i < LG->nice_gens->len - 1) {
				cout << ", " << endl;
			}
			if (((i + 1) % 3) == 0 && i < LG->nice_gens->len - 1) {
				cout << "$$" << endl;
				cout << "$$" << endl;
				}
			}
		cout << "$$" << endl;
		LG->nice_gens->print_as_permutation(cout);
	}



	int n;

	n = A->matrix_group_dimension();

	cout << "interface_algebra::do_linear_group The group acts on the points of PG(" << n - 1
			<< "," << Descr->input_q << ")" << endl;

#if 0
	if (A->degree < 1000) {
		int i;

		for (i = 0; i < A->degree; i++) {
			cout << i << " & ";
			A->print_point(i, cout);
			cout << "\\\\" << endl;
		}
	}
	else {
		cout << "Too many points to print" << endl;
	}
#endif



	if (f_group_theoretic_activity) {
		perform_group_theoretic_activity(F, LG,
				Group_theoretic_activity_description, verbose_level);
	}

	if (f_v) {
		cout << "interface_algebra::do_linear_group done" << endl;
	}
}

void interface_algebra::perform_group_theoretic_activity(
		finite_field *F, linear_group *LG,
		group_theoretic_activity_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::perform_group_theoretic_activity" << endl;
	}


	action *A;

	A = LG->A2;

	cout << "created group " << A->label << endl;

	{
		group_theoretic_activity Activity;

		Activity.init(Descr, F, LG, verbose_level);

		Activity.perform_activity(verbose_level);

	}


	if (f_v) {
		cout << "interface_algebra::perform_group_theoretic_activity done" << endl;
	}
}

void interface_algebra::do_cheat_sheet_GF(int q, int f_poly, std::string &poly, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_cheat_sheet_GF q=" << q << endl;
	}

	//int i;
	//int f_poly = FALSE;
	//const char *poly = NULL;

	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "GF_%d.tex", q);
	snprintf(title, 1000, "Cheat Sheet GF($%d$)", q);
	//sprintf(author, "");
	author[0] = 0;

	finite_field F;

	if (f_poly) {
		F.init_override_polynomial(q, poly, verbose_level);
	}
	else {
		F.init(q, 0 /* verbose_level */);
	}


	F.addition_table_save_csv();

	F.multiplication_table_save_csv();

	F.addition_table_reordered_save_csv();

	F.multiplication_table_reordered_save_csv();


	{
		ofstream f(fname);


		//algebra_global AG;

		//AG.cheat_sheet_GF(q, f_override_poly, my_override_poly, verbose_level);
		latex_interface L;

		//F.init(q), verbose_level - 2);

		L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
			title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
				TRUE /* f_12pt */,
				TRUE /* f_enlarged_page */,
				TRUE /* f_pagenumbers */,
				NULL /* extra_praeamble */);


		F.cheat_sheet(f, verbose_level);

		F.cheat_sheet_main_table(f, verbose_level);

		F.cheat_sheet_addition_table(f, verbose_level);

		F.cheat_sheet_multiplication_table(f, verbose_level);

		F.cheat_sheet_power_table(f, verbose_level);





		L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "interface_algebra::do_cheat_sheet_GF q=" << q << " done" << endl;
	}
}

void interface_algebra::do_all_rational_normal_forms(int d, int q, int f_poly, std::string &poly, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_all_rational_normal_forms d=" << d << " q=" << q << endl;
	}
	algebra_global_with_action Algebra;

	Algebra.classes_GL(q, d,
			FALSE /* f_no_eigenvalue_one */, verbose_level);

	if (f_v) {
		cout << "interface_algebra::do_all_rational_normal_forms done" << endl;
	}
}


void interface_algebra::do_search_for_primitive_polynomial_in_range(int p_min, int p_max,
		int deg_min, int deg_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_search_for_primitive_polynomial_in_range" << endl;
		cout << "p_min=" << p_min << endl;
		cout << "p_max=" << p_max << endl;
		cout << "deg_min=" << deg_min << endl;
		cout << "deg_max=" << deg_max << endl;
	}


	if (deg_min == deg_max && p_min == p_max) {
		char *poly;

		algebra_global AG;


		poly = AG.search_for_primitive_polynomial_of_given_degree(
				p_min, deg_min, verbose_level);

		cout << "poly = " << poly << endl;

	}
	else {
		algebra_global AG;

		AG.search_for_primitive_polynomials(p_min, p_max,
				deg_min, deg_max,
				verbose_level);
	}

	if (f_v) {
		cout << "interface_algebra::do_search_for_primitive_polynomial_in_range done" << endl;
	}
}

void interface_algebra::do_make_table_of_irreducible_polynomials(int deg, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_make_table_of_irreducible_polynomials" << endl;
		cout << "deg=" << deg << endl;
		cout << "q=" << q << endl;
	}
	int nb;
	//int *Table;
	std::vector<std::vector<int>> Table;
	finite_field F;
	algebra_global Algebra;

	F.init(q, 0);

	Algebra.make_all_irreducible_polynomials_of_degree_d(&F, deg,
			Table, verbose_level);

	nb = Table.size();

	cout << "The " << nb << " irreducible polynomials of "
			"degree " << deg << " over F_" << q << " are:" << endl;

	int_vec_vec_print(Table);


	//int_matrix_print(Table, nb, deg + 1);

	//FREE_int(Table);

	if (f_v) {
		cout << "interface_algebra::do_make_table_of_irreducible_polynomials done" << endl;
	}
}

void interface_algebra::do_character_table_symmetric_group(int deg, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_character_table_symmetric_group" << endl;
		cout << "deg=" << deg << endl;
	}

	character_table_burnside *CTB;

	CTB = NEW_OBJECT(character_table_burnside);

	CTB->do_it(deg, verbose_level);

	FREE_OBJECT(CTB);

	if (f_v) {
		cout << "interface_algebra::do_character_table_symmetric_group done" << endl;
	}
}

void interface_algebra::do_make_A5_in_PSL_2_q(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_make_A5_in_PSL_2_q" << endl;
		cout << "q=" << q << endl;
	}

	algebra_global_with_action A;

	A.A5_in_PSL_(q, verbose_level);

	if (f_v) {
		cout << "interface_algebra::do_make_A5_in_PSL_2_q done" << endl;
	}
}







}}

