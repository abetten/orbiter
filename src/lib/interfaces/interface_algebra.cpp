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
	argc = 0;
	argv = NULL;

	f_linear_group = FALSE;
	Linear_group_description = NULL;
	F = NULL;
	LG = NULL;

	f_group_theoretic_activity = FALSE;
	Group_theoretic_activity_description = NULL;
	f_cheat_sheet_GF = FALSE;
	q = 0;
	f_search_for_primitive_polynomial_in_range = FALSE;
	p_min = 0;
	p_max = 0;
	deg_min = 0;
	deg_max = 0;
	f_make_table_of_irreducible_polynomials = FALSE;
	deg = 0;
	q = 0;
	f_make_character_table_symmetric_group = FALSE;
	f_make_A5_in_PSL_2_q = FALSE;
	f_eigenstuff_matrix_direct = FALSE;
	f_eigenstuff_matrix_from_file = FALSE;
	eigenstuff_n = 0;
	eigenstuff_q = 0;
	eigenstuff_coeffs = NULL;
	eigenstuff_fname = NULL;
	f_surface_classify = FALSE;
	f_surface_report = FALSE;
	f_surface_identify_Sa = FALSE;
	f_surface_isomorphism_testing = FALSE;
		surface_descr_isomorph1 = NULL;
		surface_descr_isomorph2 = NULL;
	f_surface_recognize = FALSE;
		surface_descr = NULL;
	f_young_symmetrizer = FALSE;
	young_symmetrizer_n = 0;
	f_young_symmetrizer_sym_4 = FALSE;
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
	else if (strcmp(argv[i], "-search_for_primitive_polynomial_in_range") == 0) {
		cout << "-search_for_primitive_polynomial_in_range <int : p_min> <int : p_max> <int : deg_min> <int : deg_max> " << endl;
	}
	else if (strcmp(argv[i], "-make_table_of_irreducible_polynomials") == 0) {
		cout << "-table_of_irreducible_polynomials <int : deg> <int : q> " << endl;
	}
	else if (strcmp(argv[i], "-make_character_table_symmetric_group") == 0) {
		cout << "-make_character_table_symmetric_group <int : deg> " << endl;
	}
	else if (strcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		cout << "-make_A5_in_PSL_2_q <int : q> " << endl;
	}
	else if (strcmp(argv[i], "-eigenstuff_matrix_direct") == 0) {
		cout << "-eigenstuff_matrix_direct <int : m> <int : q> <string : coeffs> " << endl;
	}
	else if (strcmp(argv[i], "-eigenstuff_matrix_from_file") == 0) {
		cout << "-eigenstuff_matrix_from_file <int : m> <int : q>  <string : fname> " << endl;
	}
	else if (strcmp(argv[i], "-surface_classify") == 0) {
		cout << "-surface_classify " << endl;
	}
	else if (strcmp(argv[i], "-surface_report") == 0) {
		cout << "-surface_report " << endl;
	}
	else if (strcmp(argv[i], "-surface_identify_Sa") == 0) {
		cout << "-surface_identify_Sa " << endl;
	}
	else if (strcmp(argv[i], "-surface_isomorphism_testing") == 0) {
		cout << "-surface_isomorphism_testing " << endl;
	}
	else if (strcmp(argv[i], "-surface_recognize") == 0) {
		cout << "-surface_recognize " << endl;
	}
	else if (strcmp(argv[i], "-young_symmetrizer") == 0) {
		cout << "-young_symmetrizer <int : n>" << endl;
	}
	else if (strcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		cout << "-young_symmetrizer_sym_4 " << endl;
	}
}

int interface_algebra::recognize_keyword(int argc,
		const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-linear_group") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-group_theoretic_activity") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-cheat_sheet_GF") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-search_for_primitive_polynomial_in_range") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-make_table_of_irreducible_polynomials") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-make_character_table_symmetric_group") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-eigenstuff_matrix_direct") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-eigenstuff_matrix_from_file") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-surface_classify") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-surface_report") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-surface_identify_Sa") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-surface_isomorphism_testing") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-surface_recognize") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-young_symmetrizer") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		return true;
	}
	return false;
}

void interface_algebra::read_arguments(int argc,
		const char **argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_algebra::read_arguments" << endl;
	//return 0;

	interface_algebra::argc = argc;
	interface_algebra::argv = argv;

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-linear_group") == 0) {
			f_linear_group = TRUE;
			Linear_group_description = NEW_OBJECT(linear_group_description);
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
			Group_theoretic_activity_description = NEW_OBJECT(group_theoretic_activity_description);
			i += Group_theoretic_activity_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-group_theoretic_activity" << endl;
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
		else if (strcmp(argv[i], "-search_for_primitive_polynomial_in_range") == 0) {
			f_search_for_primitive_polynomial_in_range = TRUE;
			p_min = atoi(argv[++i]);
			p_max = atoi(argv[++i]);
			deg_min = atoi(argv[++i]);
			deg_max = atoi(argv[++i]);
			cout << "-search_for_primitive_polynomial_in_range " << p_min << " " << p_max << " " << deg_min << " " << deg_max << " " << endl;
		}
		else if (strcmp(argv[i], "-make_table_of_irreducible_polynomials") == 0) {
			f_make_table_of_irreducible_polynomials = TRUE;
			deg = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-make_table_of_irreducible_polynomials " << deg << " " << q << " " << endl;
		}
		else if (strcmp(argv[i], "-make_character_table_symmetric_group") == 0) {
			f_make_character_table_symmetric_group = TRUE;
			deg = atoi(argv[++i]);
			cout << "-make_character_table_symmetric_group " << deg << endl;
		}
		else if (strcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
			f_make_A5_in_PSL_2_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-make_A5_in_PSL_2_q " << q << endl;
		}
		else if (strcmp(argv[i], "-eigenstuff_matrix_direct") == 0) {
			f_eigenstuff_matrix_direct = TRUE;
			eigenstuff_n = atoi(argv[++i]);
			eigenstuff_q = atoi(argv[++i]);
			eigenstuff_coeffs = argv[++i];
			cout << "-eigenstuff_matrix_direct " << eigenstuff_n << " " << eigenstuff_q << " " << eigenstuff_coeffs << endl;
		}
		else if (strcmp(argv[i], "-eigenstuff_matrix_from_file") == 0) {
			f_eigenstuff_matrix_from_file = TRUE;
			eigenstuff_n = atoi(argv[++i]);
			eigenstuff_q = atoi(argv[++i]);
			eigenstuff_fname = argv[++i];
			cout << "-eigenstuff_matrix_from_file " << eigenstuff_n << " " << eigenstuff_q << " " << eigenstuff_fname << endl;
		}
		else if (strcmp(argv[i], "-surface_classify") == 0) {
			f_surface_classify = TRUE;
			cout << "-surface_classify " << endl;
		}
		else if (strcmp(argv[i], "-surface_report") == 0) {
			f_surface_report = TRUE;
			cout << "-surface_report " << endl;
		}
		else if (strcmp(argv[i], "-surface_identify_Sa") == 0) {
			f_surface_identify_Sa = TRUE;
			cout << "-surface_identify_Sa " << endl;
		}
		else if (strcmp(argv[i], "-surface_isomorphism_testing") == 0) {
			f_surface_isomorphism_testing = TRUE;
			cout << "-surface_isomorphism_testing reading description of first surface" << endl;
			surface_descr_isomorph1 = NEW_OBJECT(surface_create_description);
			i += surface_descr_isomorph1->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level) - 1;
			cout << "-isomorph after reading description of first surface" << endl;
			i += 2;
			cout << "the current argument is " << argv[i] << endl;
			cout << "-isomorph reading description of second surface" << endl;
			surface_descr_isomorph2 = NEW_OBJECT(surface_create_description);
			i += surface_descr_isomorph2->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level) - 1;
			cout << "-surface_isomorphism_testing " << endl;
		}
		else if (strcmp(argv[i], "-surface_recognize") == 0) {
			f_surface_recognize = TRUE;
			cout << "-surface_recognize reading description of surface" << endl;
			surface_descr = NEW_OBJECT(surface_create_description);
			i += surface_descr->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level) - 1;
			i += 2;
			cout << "-surface_recognize " << endl;
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
	}
}


void interface_algebra::worker(int verbose_level)
{
	if (f_linear_group) {
		do_linear_group(Linear_group_description, verbose_level);
	}

	if (f_cheat_sheet_GF) {
		do_cheat_sheet_GF(q, verbose_level);
	}
	else if (f_search_for_primitive_polynomial_in_range) {
		do_search_for_primitive_polynomial_in_range(p_min, p_max, deg_min, deg_max, verbose_level);
	}
	else if (f_make_table_of_irreducible_polynomials) {
		do_make_table_of_irreducible_polynomials(deg, q, verbose_level);
	}
	else if (f_make_character_table_symmetric_group) {
		do_make_character_table_symmetric_group(deg, verbose_level);
	}
	else if (f_make_A5_in_PSL_2_q) {
		do_make_A5_in_PSL_2_q(q, verbose_level);
	}
	else if (f_eigenstuff_matrix_direct) {
		do_eigenstuff_matrix_direct(eigenstuff_n, eigenstuff_q, eigenstuff_coeffs, verbose_level);
	}
	else if (f_eigenstuff_matrix_from_file) {
		do_eigenstuff_matrix_from_file(eigenstuff_n, eigenstuff_q, eigenstuff_fname, verbose_level);
	}
	else if (f_surface_classify) {
		if (!f_linear_group) {
			cout << "need a linear group to classify surfaces" << endl;
			exit(1);
		}
		do_surface_classify(verbose_level);
	}
	else if (f_surface_report) {
		if (!f_linear_group) {
			cout << "need a linear group to report surfaces" << endl;
			exit(1);
		}
		do_surface_report(verbose_level);
	}
	else if (f_surface_identify_Sa) {
		if (!f_linear_group) {
			cout << "need a linear group to identify S1 surfaces" << endl;
			exit(1);
		}
		do_surface_identify_Sa(verbose_level);
	}
	else if (f_surface_isomorphism_testing) {
		if (!f_linear_group) {
			cout << "need a linear group to do isomorphism testing for surfaces" << endl;
			exit(1);
		}
		do_surface_isomorphism_testing(surface_descr_isomorph1, surface_descr_isomorph2, verbose_level);
	}
	else if (f_surface_recognize) {
		if (!f_linear_group) {
			cout << "need a linear group to do recognition for surfaces" << endl;
			exit(1);
		}

		do_surface_recognize(surface_descr, verbose_level);
	}
	else if (f_young_symmetrizer) {
		algebra_global_with_action Algebra;

		Algebra.young_symmetrizer(young_symmetrizer_n, verbose_level);
	}
	else if (f_young_symmetrizer_sym_4) {
		algebra_global_with_action Algebra;

		Algebra.young_symmetrizer_sym_4(verbose_level);
	}
}

void interface_algebra::do_eigenstuff_matrix_direct(
		int n, int q, const char *coeffs_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_eigenstuff_matrix_direct" << endl;
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
		cout << "interface_algebra::do_eigenstuff_matrix_direct done" << endl;
	}
}

void interface_algebra::do_eigenstuff_matrix_from_file(
		int n, int q, const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_eigenstuff_matrix_from_file" << endl;
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
		cout << "interface_algebra::do_eigenstuff_matrix_from_file done" << endl;
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
				Descr->override_polynomial, verbose_level);
	}
	else {
		cout << "creating finite field of order q=" << Descr->input_q << endl;
		F->init(Descr->input_q, 0);
	}

	Descr->F = F;
	//q = Descr->input_q;



	LG = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "linear_group before LG->init, "
				"creating the group" << endl;
		}

	LG->init(Descr, verbose_level - 1);

	if (f_v) {
		cout << "linear_group after LG->init" << endl;
		}

	action *A;

	A = LG->A2;

	cout << "created group " << LG->prefix << endl;



	if (LG->f_has_nice_gens) {
		cout << "we have nice generators, they are:" << endl;
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



	cout << "The group acts on the points of PG(" << Descr->n - 1
			<< "," << Descr->input_q << ")" << endl;

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

	cout << "created group " << LG->prefix << endl;

	if (Descr->f_classes) {

		sims *G;

		G = LG->Strong_gens->create_sims(verbose_level);

		A->conjugacy_classes_and_normalizers(G,
				LG->prefix, LG->label_latex, verbose_level);

		FREE_OBJECT(G);
	}

	if (Descr->f_multiply) {
		cout << "multiplying" << endl;
		cout << "A=" << Descr->multiply_a << endl;
		cout << "B=" << Descr->multiply_b << endl;
		int *Elt1;
		int *Elt2;
		int *Elt3;

		Elt1 = NEW_int(A->elt_size_in_int);
		Elt2 = NEW_int(A->elt_size_in_int);
		Elt3 = NEW_int(A->elt_size_in_int);

		A->make_element_from_string(Elt1,
				Descr->multiply_a, verbose_level);
		cout << "A=" << endl;
		A->element_print_quick(Elt1, cout);

		A->make_element_from_string(Elt2,
				Descr->multiply_b, verbose_level);
		cout << "B=" << endl;
		A->element_print_quick(Elt2, cout);

		A->element_mult(Elt1, Elt2, Elt3, 0);
		cout << "A*B=" << endl;
		A->element_print_quick(Elt3, cout);
		A->element_print_for_make_element(Elt3, cout);
		cout << endl;
		FREE_int(Elt1);
		FREE_int(Elt2);
		FREE_int(Elt3);
	}

	if (Descr->f_inverse) {
		cout << "computing the inverse" << endl;
		cout << "A=" << Descr->inverse_a << endl;
		int *Elt1;
		int *Elt2;

		Elt1 = NEW_int(A->elt_size_in_int);
		Elt2 = NEW_int(A->elt_size_in_int);

		A->make_element_from_string(Elt1,
				Descr->inverse_a, verbose_level);
		cout << "A=" << endl;
		A->element_print_quick(Elt1, cout);

		A->element_invert(Elt1, Elt2, 0);
		cout << "A^-1=" << endl;
		A->element_print_quick(Elt2, cout);
		A->element_print_for_make_element(Elt2, cout);
		cout << endl;
		FREE_int(Elt1);
		FREE_int(Elt2);
	}

	if (Descr->f_normalizer) {
		char fname_magma_prefix[1000];
		sims *G;
		sims *H;
		strong_generators *gens_N;
		longinteger_object N_order;


		sprintf(fname_magma_prefix, "%s_normalizer", LG->prefix);

		G = LG->initial_strong_gens->create_sims(verbose_level);
		H = LG->Strong_gens->create_sims(verbose_level);

		cout << "group order G = " << G->group_order_lint() << endl;
		cout << "group order H = " << H->group_order_lint() << endl;
		cout << "before A->normalizer_using_MAGMA" << endl;
		A->normalizer_using_MAGMA(fname_magma_prefix,
				G, H, gens_N, verbose_level);

		cout << "group order G = " << G->group_order_lint() << endl;
		cout << "group order H = " << H->group_order_lint() << endl;
		gens_N->group_order(N_order);
		cout << "group order N = " << N_order << endl;
		cout << "Strong generators for the normalizer of H are:" << endl;
		gens_N->print_generators_tex(cout);
		cout << "Strong generators for the normalizer of H as permutations are:" << endl;
		gens_N->print_generators_as_permutations();

		sims *N;
		int N_goi;

		N = gens_N->create_sims(verbose_level);
		N_goi = N->group_order_lint();
		cout << "The elements of N are:" << endl;
		N->print_all_group_elements();

		if (N_goi < 30) {
			cout << "creating group table:" << endl;

			char fname[1000];
			int *Table;
			long int n;
			N->create_group_table(Table, n, verbose_level);
			cout << "The group table of the normalizer is:" << endl;
			int_matrix_print(Table, n, n, 2);
			sprintf(fname, "normalizer_%ld.tex", n);
			{
				ofstream fp(fname);
				latex_interface L;
				L.head_easy(fp);

				fp << "\\begin{sidewaystable}" << endl;
				fp << "$$" << endl;
				L.int_matrix_print_tex(fp, Table, n, n);
				fp << "$$" << endl;
				fp << "\\end{sidewaystable}" << endl;

				N->print_all_group_elements_tex(fp);

				L.foot(fp);
			}
			FREE_int(Table);
		}
	}

	if (Descr->f_report) {
		char fname[1000];
		char title[1000];
		const char *author = "Orbiter";
		const char *extras_for_preamble = "";

		double tikz_global_scale = 0.3;
		double tikz_global_line_width = 1.;
		int factor1000 = 1000;


		sprintf(fname, "%s_report.tex", LG->prefix);
		sprintf(title, "The group $%s$", LG->label_latex);

		{
			ofstream fp(fname);
			latex_interface L;
			//latex_head_easy(fp);
			L.head(fp,
				FALSE /* f_book */, TRUE /* f_title */,
				title, author,
				FALSE /*f_toc*/, FALSE /* f_landscape*/, FALSE /* f_12pt*/,
				TRUE /*f_enlarged_page*/, TRUE /* f_pagenumbers*/,
				extras_for_preamble);

			LG->report(fp, Descr->f_sylow, Descr->f_group_table,
					tikz_global_scale, tikz_global_line_width, factor1000,
					verbose_level);

			L.foot(fp);
		}
	}

	if (Descr->f_print_elements) {
		sims *H;

		//G = LG->initial_strong_gens->create_sims(verbose_level);
		H = LG->Strong_gens->create_sims(verbose_level);

		//cout << "group order G = " << G->group_order_int() << endl;
		cout << "group order H = " << H->group_order_lint() << endl;

		int *Elt;
		longinteger_object go;
		int i, cnt;

		Elt = NEW_int(A->elt_size_in_int);
		H->group_order(go);


		cnt = 0;
		for (i = 0; i < go.as_lint(); i++) {
			H->element_unrank_lint(i, Elt);

			cout << "Element " << setw(5) << i << " / "
					<< go.as_int() << ":" << endl;
			A->element_print(Elt, cout);
			cout << endl;
			A->element_print_as_permutation(Elt, cout);
			cout << endl;



		}
		FREE_int(Elt);
	}

	if (Descr->f_print_elements_tex) {
		sims *H;

		//G = LG->initial_strong_gens->create_sims(verbose_level);
		H = LG->Strong_gens->create_sims(verbose_level);

		//cout << "group order G = " << G->group_order_int() << endl;
		cout << "group order H = " << H->group_order_lint() << endl;

		int *Elt;
		longinteger_object go;

		Elt = NEW_int(A->elt_size_in_int);
		H->group_order(go);

		action *A_conj;

		A_conj = NEW_OBJECT(action);

		cout << "before A_conj->induced_action_by_conjugation" << endl;
		A_conj->induced_action_by_conjugation(H /* old_G */,
				H /* Base_group */, FALSE /* f_ownership */,
				FALSE /* f_basis */, verbose_level);
		cout << "before A_conj->induced_action_by_conjugation" << endl;

		schreier Schreier;
		cout << "before A_conj->all_point_orbits" << endl;
		A_conj->all_point_orbits_from_generators(Schreier, LG->Strong_gens, verbose_level);
		cout << "after A_conj->all_point_orbits" << endl;

		char fname[1000];

		sprintf(fname, "%s_elements.tex", LG->prefix);


				{
					ofstream fp(fname);
					latex_interface L;
					L.head_easy(fp);

					H->print_all_group_elements_tex(fp);

					Schreier.print_and_list_orbits_tex(fp);

					if (Descr->f_order_of_products) {
						int *elements;
						int nb_elements;
						int *order_table;
						int i;

						int_vec_scan(Descr->order_of_products_elements, elements, nb_elements);

						int j;
						int *Elt1, *Elt2, *Elt3;

						Elt1 = NEW_int(A->elt_size_in_int);
						Elt2 = NEW_int(A->elt_size_in_int);
						Elt3 = NEW_int(A->elt_size_in_int);

						order_table = NEW_int(nb_elements * nb_elements);
						for (i = 0; i < nb_elements; i++) {

							H->element_unrank_lint(elements[i], Elt1);


							for (j = 0; j < nb_elements; j++) {

								H->element_unrank_lint(elements[j], Elt2);

								A->element_mult(Elt1, Elt2, Elt3, 0);

								order_table[i * nb_elements + j] = A->element_order(Elt3);

							}
						}
						FREE_int(Elt1);
						FREE_int(Elt2);
						FREE_int(Elt3);

						latex_interface L;

						fp << "$$" << endl;
						L.print_integer_matrix_with_labels(fp, order_table,
								nb_elements, nb_elements, elements, elements, TRUE /* f_tex */);
						fp << "$$" << endl;
					}

					L.foot(fp);
				}

		FREE_int(Elt);
	}

	if (Descr->f_search_subgroup) {
		sims *H;

		//G = LG->initial_strong_gens->create_sims(verbose_level);
		H = LG->Strong_gens->create_sims(verbose_level);

		//cout << "group order G = " << G->group_order_int() << endl;
		cout << "group order H = " << H->group_order_lint() << endl;

		int *Elt;
		longinteger_object go;
		int i, cnt;

		Elt = NEW_int(A->elt_size_in_int);
		H->group_order(go);

		cnt = 0;
		for (i = 0; i < go.as_int(); i++) {
			H->element_unrank_lint(i, Elt);

#if 0
			cout << "Element " << setw(5) << i << " / "
					<< go.as_int() << ":" << endl;
			A->element_print(Elt, cout);
			cout << endl;
			A->element_print_as_permutation(Elt, cout);
			cout << endl;
#endif
			if (Elt[7] == 0 && Elt[8] == 0 &&
					Elt[11] == 0 && Elt[14] == 0 &&
					Elt[12] == 0 && Elt[19] == 0 &&
					Elt[22] == 0 && Elt[23] == 0) {
				cout << "Element " << setw(5) << i << " / "
						<< go.as_int() << " = " << cnt << ":" << endl;
				A->element_print(Elt, cout);
				cout << endl;
				//A->element_print_as_permutation(Elt, cout);
				//cout << endl;
				cnt++;
			}
		}
		cout << "we found " << cnt << " group elements of the special form" << endl;

		FREE_int(Elt);

	}


	if (Descr->f_orbits_on_set_system_from_file) {
		cout << "computing orbits on set system from file "
				<< Descr->orbits_on_set_system_from_file_fname << ":" << endl;
		file_io Fio;
		int *M;
		int m, n;
		long int *Table;
		int i, j;

		Fio.int_matrix_read_csv(Descr->orbits_on_set_system_from_file_fname, M,
				m, n, verbose_level);
		cout << "read a matrix of size " << m << " x " << n << endl;


		//orbits_on_set_system_first_column = atoi(argv[++i]);
		//orbits_on_set_system_number_of_columns = atoi(argv[++i]);


		Table = NEW_lint(m * Descr->orbits_on_set_system_number_of_columns);
		for (i = 0; i < m; i++) {
			for (j = 0; j < Descr->orbits_on_set_system_number_of_columns; j++) {
				Table[i * Descr->orbits_on_set_system_number_of_columns + j] =
						M[i * n + Descr->orbits_on_set_system_first_column + j];
			}
		}
		action *A_on_sets;
		int set_size;

		set_size = Descr->orbits_on_set_system_number_of_columns;

		cout << "creating action on sets:" << endl;
		A_on_sets = A->create_induced_action_on_sets(m /* nb_sets */,
				set_size, Table,
				verbose_level);

		schreier *Sch;
		int first, a;

		cout << "computing orbits on sets:" << endl;
		A_on_sets->compute_orbits_on_points(Sch,
				LG->Strong_gens->gens, verbose_level);

		cout << "The orbit lengths are:" << endl;
		Sch->print_orbit_lengths(cout);

		cout << "The orbits are:" << endl;
		//Sch->print_and_list_orbits(cout);
		for (i = 0; i < Sch->nb_orbits; i++) {
			cout << " Orbit " << i << " / " << Sch->nb_orbits
					<< " : " << Sch->orbit_first[i] << " : " << Sch->orbit_len[i];
			cout << " : ";

			first = Sch->orbit_first[i];
			a = Sch->orbit[first + 0];
			cout << a << " : ";
			lint_vec_print(cout, Table + a * set_size, set_size);
			cout << endl;
			//Sch->print_and_list_orbit_tex(i, ost);
			}
		char fname[1000];

		strcpy(fname, Descr->orbits_on_set_system_from_file_fname);
		chop_off_extension(fname);
		strcat(fname, "_orbit_reps.txt");

		{
			ofstream ost(fname);

			for (i = 0; i < Sch->nb_orbits; i++) {

				first = Sch->orbit_first[i];
				a = Sch->orbit[first + 0];
				ost << set_size;
				for (j = 0; j < set_size; j++) {
					ost << " " << Table[a * set_size + j];
				}
				ost << endl;
			}
			ost << -1 << " " << Sch->nb_orbits << endl;
		}

	}

	if (Descr->f_orbit_of_set_from_file) {

		cout << "computing orbit of set from file "
				<< Descr->orbit_of_set_from_file_fname << ":" << endl;
		file_io Fio;
		long int *the_set;
		int set_sz;

		Fio.read_set_from_file(Descr->orbit_of_set_from_file_fname,
				the_set, set_sz, verbose_level);
		cout << "read a set of size " << set_sz << endl;

		orbit_of_sets *OS;

		OS = NEW_OBJECT(orbit_of_sets);

		OS->init(A, A, the_set, set_sz,
				LG->Strong_gens->gens, verbose_level);

		//OS->compute(verbose_level);

		cout << "Found an orbit of length " << OS->used_length << endl;

		long int *Table;
		int orbit_length, set_size;

		cout << "before OS->get_table_of_orbits" << endl;
		OS->get_table_of_orbits_and_hash_values(Table,
				orbit_length, set_size, verbose_level);
		cout << "after OS->get_table_of_orbits" << endl;

		char str[1000];
		strcpy(str, Descr->orbit_of_set_from_file_fname);
		chop_off_extension(str);

		char fname[1000];
		sprintf(fname, "orbit_of_%s_under_%s_with_hash.csv", str, LG->prefix);
		cout << "Writing table to file " << fname << endl;
		Fio.lint_matrix_write_csv(fname,
				Table, orbit_length, set_size);
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

		FREE_lint(Table);

		cout << "before OS->get_table_of_orbits" << endl;
		OS->get_table_of_orbits(Table,
				orbit_length, set_size, verbose_level);
		cout << "after OS->get_table_of_orbits" << endl;

		strcpy(str, Descr->orbit_of_set_from_file_fname);
		chop_off_extension(str);
		sprintf(fname, "orbit_of_%s_under_%s.txt", str, LG->prefix);
		cout << "Writing table to file " << fname << endl;
		{
			ofstream ost(fname);
			int i;
			for (i = 0; i < orbit_length; i++) {
				ost << set_size;
				for (int j = 0; j < set_size; j++) {
					ost << " " << Table[i * set_size + j];
				}
				ost << endl;
			}
			ost << -1 << " " << orbit_length << endl;
		}
		//Fio.int_matrix_write_csv(fname,
		//		Table, orbit_length, set_size);
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;


		cout << "before FREE_OBJECT(OS)" << endl;
		FREE_OBJECT(OS);
		cout << "after FREE_OBJECT(OS)" << endl;
	}

	if (Descr->f_orbit_of) {

		schreier *Sch;
		Sch = NEW_OBJECT(schreier);

		cout << "computing orbit of point " << Descr->orbit_of_idx << ":" << endl;

		//A->all_point_orbits(*Sch, verbose_level);

		Sch->init(A, verbose_level - 2);
		if (!A->f_has_strong_generators) {
			cout << "action::all_point_orbits !f_has_strong_generators" << endl;
			exit(1);
			}
		Sch->init_generators(*LG->Strong_gens->gens /* *strong_generators */, verbose_level - 2);
		Sch->initialize_tables();
		Sch->compute_point_orbit(Descr->orbit_of_idx, verbose_level);


		cout << "computing orbit of point done." << endl;

		char fname_tree_mask[1000];

		sprintf(fname_tree_mask, "%s_orbit_of_point_%d.layered_graph",
				LG->prefix, Descr->orbit_of_idx);

		Sch->export_tree_as_layered_graph(0 /* orbit_no */,
				fname_tree_mask,
				verbose_level - 1);

		strong_generators *SG_stab;
		longinteger_object full_group_order;

		LG->Strong_gens->group_order(full_group_order);

		cout << "computing the stabilizer of the orbit rep:" << endl;
		SG_stab = Sch->stabilizer_orbit_rep(
				LG->A_linear,
				full_group_order,
				0 /* orbit_idx */, verbose_level);
		cout << "The stabilizer of the orbit rep has been computed:" << endl;
		SG_stab->print_generators();
		SG_stab->print_generators_tex();


		schreier *shallow_tree;

		cout << "computing shallow Schreier tree:" << endl;

#if 0
		enum shallow_schreier_tree_strategy Shallow_schreier_tree_strategy =
				shallow_schreier_tree_standard;
				//shallow_schreier_tree_Seress_deterministic;
				//shallow_schreier_tree_Seress_randomized;
				//shallow_schreier_tree_Sajeeb;
#endif
		int orbit_idx = 0;
		int f_randomized = TRUE;

		Sch->shallow_tree_generators(orbit_idx,
				f_randomized,
				shallow_tree,
				verbose_level);

		cout << "computing shallow Schreier tree done." << endl;

		sprintf(fname_tree_mask, "%s_%%d_shallow.layered_graph", LG->prefix);

		shallow_tree->export_tree_as_layered_graph(0 /* orbit_no */,
				fname_tree_mask,
				verbose_level - 1);



	} // if (f_orbit_of)


	if (Descr->f_orbits_on_points) {



		cout << "computing orbits on points:" << endl;


		schreier *Sch;
		Sch = NEW_OBJECT(schreier);

		cout << "Strong generators are:" << endl;
		LG->Strong_gens->print_generators();
		cout << "Strong generators in tex are:" << endl;
		LG->Strong_gens->print_generators_tex(cout);
		cout << "Strong generators as permutations are:" << endl;
		LG->Strong_gens->print_generators_as_permutations();




		//A->all_point_orbits(*Sch, verbose_level);
		A->all_point_orbits_from_generators(*Sch,
				LG->Strong_gens,
				verbose_level);

		longinteger_object go;
		int orbit_idx;

		LG->Strong_gens->group_order(go);
		cout << "Computing stabilizers. Group order = " << go << endl;
		if (Descr->f_stabilizer) {
			for (orbit_idx = 0; orbit_idx < Sch->nb_orbits; orbit_idx++) {

				strong_generators *SG;

				SG = Sch->stabilizer_orbit_rep(
						LG->A_linear /*default_action*/,
						go,
						orbit_idx, 0 /*verbose_level*/);

				cout << "orbit " << orbit_idx << " / " << Sch->nb_orbits << ":" << endl;
				SG->print_generators_tex(cout);

			}
		}


		cout << "computing orbits on points done." << endl;


		{
			int *M;
			int *O;
			int h, x, y;
			int idx, m;

			m = Sch->A->degree;

			M = NEW_int(m * m);
			O = NEW_int(m);


			for (idx = 0; idx < Sch->nb_orbits; idx++) {
				int_vec_zero(M, m * m);
				for (h = 0; h < Sch->orbit_len[idx] - 1; h++) {
					x = Sch->orbit[Sch->orbit_first[idx] + h];
					y = Sch->orbit[Sch->orbit_first[idx] + h + 1];
					M[x * m + y] = 1;
				}
				for (h = 0; h < Sch->orbit_len[idx] - 1; h++) {
					x = Sch->orbit[Sch->orbit_first[idx] + h];
					O[h] = x;
				}
				{
				char fname[1000];
				file_io Fio;

				sprintf(fname, "orbit_%d_transition.csv", idx);
				Fio.int_matrix_write_csv(fname, M, m, m);

				cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
				}
				{
				char fname[1000];
				file_io Fio;

				sprintf(fname, "orbit_%d_elts.csv", idx);
				Fio.int_vec_write_csv(O, Sch->orbit_len[idx],
						fname, "Elt");

				cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
				}
			}
			FREE_int(M);
			FREE_int(O);

		}

		Sch->print_and_list_orbits(cout);

		char fname_orbits[1000];
		file_io Fio;

		sprintf(fname_orbits, "%s_orbits.tex", LG->prefix);


		Sch->latex(fname_orbits);
		cout << "Written file " << fname_orbits << " of size "
				<< Fio.file_size(fname_orbits) << endl;


		char fname_tree_mask[1000];

		sprintf(fname_tree_mask, "%s_%%d.layered_graph", LG->prefix);

		if (Descr->f_export_trees) {
			for (orbit_idx = 0; orbit_idx < Sch->nb_orbits; orbit_idx++) {
				cout << "orbit " << orbit_idx << " / " <<  Sch->nb_orbits
						<< " before Sch->export_tree_as_layered_graph" << endl;
				Sch->export_tree_as_layered_graph(0 /* orbit_no */,
						fname_tree_mask,
						verbose_level - 1);
			}
		}

		if (Descr->f_shallow_tree) {
			orbit_idx = 0;
			schreier *shallow_tree;

			cout << "computing shallow Schreier tree for orbit " << orbit_idx << endl;

#if 0
			enum shallow_schreier_tree_strategy Shallow_schreier_tree_strategy =
					shallow_schreier_tree_standard;
					//shallow_schreier_tree_Seress_deterministic;
					//shallow_schreier_tree_Seress_randomized;
					//shallow_schreier_tree_Sajeeb;
#endif
			int f_randomized = TRUE;

			Sch->shallow_tree_generators(orbit_idx,
					f_randomized,
					shallow_tree,
					verbose_level);

			cout << "computing shallow Schreier tree done." << endl;

			sprintf(fname_tree_mask, "%s_%%d_shallow.layered_graph", LG->prefix);

			shallow_tree->export_tree_as_layered_graph(0 /* orbit_no */,
					fname_tree_mask,
					verbose_level - 1);
		}
	}

	if (Descr->f_orbits_on_subsets) {
		cout << "computing orbits on subsets:" << endl;
		poset_classification *PC;
		poset *Poset;

		Poset = NEW_OBJECT(poset);


		Poset->init_subset_lattice(A, A,
				LG->Strong_gens,
				verbose_level);
		PC = Poset->orbits_on_k_sets_compute(
				Descr->orbits_on_subsets_size, verbose_level);


		for (int depth = 0; depth <= Descr->orbits_on_subsets_size; depth++) {
			cout << "There are " << PC->nb_orbits_at_level(depth)
					<< " orbits on subsets of size " << depth << ":" << endl;

			if (depth < Descr->orbits_on_subsets_size) {
				//continue;
			}
			PC->list_all_orbits_at_level(depth,
					FALSE /* f_has_print_function */,
					NULL /* void (*print_function)(ostream &ost, int len, int *S, void *data)*/,
					NULL /* void *print_function_data*/,
					FALSE /* f_show_orbit_decomposition */,
					TRUE /* f_show_stab */,
					FALSE /* f_save_stab */,
					FALSE /* f_show_whole_orbit*/);
		}

		if (Descr->f_draw_poset) {
			{
			char fname_poset[1000];
			sprintf(fname_poset, "%s_poset_%d", LG->prefix, Descr->orbits_on_subsets_size);
			PC->draw_poset(fname_poset,
					Descr->orbits_on_subsets_size /*depth*/, 0 /* data1 */,
					TRUE /* f_embedded */,
					FALSE /* f_sideways */,
					0 /* verbose_level */);
			}
		}
		if (Descr->f_draw_full_poset) {
			{
			char fname_poset[1000];
			sprintf(fname_poset, "%s_poset_%d", LG->prefix, Descr->orbits_on_subsets_size);
			//double x_stretch = 0.4;
			PC->draw_poset_full(fname_poset, Descr->orbits_on_subsets_size,
				0 /* data1 */, Descr->f_embedded, Descr->f_sideways,
				Descr->x_stretch, 0 /*verbose_level */);

			const char *fname_prefix = "flag_orbits";

			PC->make_flag_orbits_on_relations(
					Descr->orbits_on_subsets_size, fname_prefix, verbose_level);

			}
		}




		if (Descr->f_test_if_geometric) {
			int depth = Descr->test_if_geometric_depth;

			//for (depth = 0; depth <= orbits_on_subsets_size; depth++) {

			cout << "Orbits on subsets of size " << depth << ":" << endl;
			PC->list_all_orbits_at_level(depth,
					FALSE /* f_has_print_function */,
					NULL /* void (*print_function)(ostream &ost, int len, int *S, void *data)*/,
					NULL /* void *print_function_data*/,
					TRUE /* f_show_orbit_decomposition */,
					TRUE /* f_show_stab */,
					FALSE /* f_save_stab */,
					TRUE /* f_show_whole_orbit*/);
			int nb_orbits, orbit_idx;

			nb_orbits = PC->nb_orbits_at_level(depth);
			for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {

				int orbit_length;
				long int *Orbit;

				cout << "before PC->get_whole_orbit depth " << depth
						<< " orbit " << orbit_idx
						<< " / " << nb_orbits << ":" << endl;
				PC->get_whole_orbit(
						depth, orbit_idx,
						Orbit, orbit_length, verbose_level);
				cout << "depth " << depth << " orbit " << orbit_idx
						<< " / " << nb_orbits << " has length "
						<< orbit_length << ":" << endl;
				lint_matrix_print(Orbit, orbit_length, depth);

				action *Aut;
				longinteger_object ago;
				nauty_interface Nauty;

				Aut = Nauty.create_automorphism_group_of_block_system(
					A->degree /* nb_points */,
					orbit_length /* nb_blocks */,
					depth /* block_size */, Orbit,
					verbose_level);
				Aut->group_order(ago);
				cout << "The automorphism group of the set system "
						"has order " << ago << endl;

				FREE_OBJECT(Aut);
				FREE_lint(Orbit);
			}
			if (nb_orbits == 2) {
				cout << "the number of orbits at depth " << depth
						<< " is two, we will try create_automorphism_"
						"group_of_collection_of_two_block_systems" << endl;
				long int *Orbit1;
				int orbit_length1;
				long int *Orbit2;
				int orbit_length2;

				cout << "before PC->get_whole_orbit depth " << depth
						<< " orbit " << orbit_idx
						<< " / " << nb_orbits << ":" << endl;
				PC->get_whole_orbit(
						depth, 0 /* orbit_idx*/,
						Orbit1, orbit_length1, verbose_level);
				cout << "depth " << depth << " orbit " << 0
						<< " / " << nb_orbits << " has length "
						<< orbit_length1 << ":" << endl;
				lint_matrix_print(Orbit1, orbit_length1, depth);

				PC->get_whole_orbit(
						depth, 1 /* orbit_idx*/,
						Orbit2, orbit_length2, verbose_level);
				cout << "depth " << depth << " orbit " << 1
						<< " / " << nb_orbits << " has length "
						<< orbit_length2 << ":" << endl;
				lint_matrix_print(Orbit2, orbit_length2, depth);

				action *Aut;
				longinteger_object ago;
				nauty_interface Nauty;

				Aut = Nauty.create_automorphism_group_of_collection_of_two_block_systems(
					A->degree /* nb_points */,
					orbit_length1 /* nb_blocks */,
					depth /* block_size */, Orbit1,
					orbit_length2 /* nb_blocks */,
					depth /* block_size */, Orbit2,
					verbose_level);
				Aut->group_order(ago);
				cout << "The automorphism group of the collection of two set systems "
						"has order " << ago << endl;

				FREE_OBJECT(Aut);
				FREE_lint(Orbit1);
				FREE_lint(Orbit2);

			} // if nb_orbits == 2
		} // if (f_test_if_geometric)


		if (Descr->f_draw_poset) {
			{
			char fname_poset[1000];
			sprintf(fname_poset, "%s_%d", LG->prefix, Descr->orbits_on_subsets_size);
			PC->draw_poset(fname_poset,
					Descr->orbits_on_subsets_size /*depth*/, 0 /* data1 */,
					TRUE /* f_embedded */,
					FALSE /* f_sideways */,
					0 /* verbose_level */);
			}
		}
	}


	if (f_v) {
		cout << "interface_algebra::perform_group_theoretic_activity done" << endl;
	}
}

void interface_algebra::do_cheat_sheet_GF(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_cheat_sheet_GF q=" << q << endl;
	}
	int i;
	int f_poly = FALSE;
	const char *poly = NULL;

	char fname[1000];
	char title[1000];
	char author[1000];

	sprintf(fname, "GF_%d.tex", q);
	sprintf(title, "Cheat Sheet GF($%d$)", q);
	//sprintf(author, "");
	author[0] = 0;

	finite_field F;

	if (f_poly) {
		F.init_override_polynomial(q, poly, verbose_level);
	}
	else {
		F.init(q, 0 /* verbose_level */);
	}

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

	F.cheat_sheet_tables(f, verbose_level);

	int *power_table;
	int t;
	int len = q;

	t = F.primitive_root();

	power_table = NEW_int(len);
	F.power_table(t, power_table, len);

	f << "\\begin{multicols}{2}" << endl;
	f << "\\noindent" << endl;
	for (i = 0; i < len; i++) {
		if (F.e == 1) {
			f << "$" << t << "^{" << i << "} \\equiv " << power_table[i] << "$\\\\" << endl;
		}
		else {
			f << "$" << t << "^{" << i << "} = " << power_table[i] << "$\\\\" << endl;
		}
	}
	f << "\\end{multicols}" << endl;
	FREE_int(power_table);



	L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "interface_algebra::do_cheat_sheet_GF q=" << q << " done" << endl;
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
	int *Table;
	finite_field F;

	F.init(q, 0);

	F.make_all_irreducible_polynomials_of_degree_d(deg,
			nb, Table, verbose_level);

	cout << "The " << nb << " irreducible polynomials of "
			"degree " << deg << " over F_" << q << " are:" << endl;
	int_matrix_print(Table, nb, deg + 1);

	FREE_int(Table);

	if (f_v) {
		cout << "interface_algebra::do_make_table_of_irreducible_polynomials done" << endl;
	}
}

void interface_algebra::do_make_character_table_symmetric_group(int deg, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_make_character_table_symmetric_group" << endl;
		cout << "deg=" << deg << endl;
	}

	character_table_burnside *CTB;

	CTB = NEW_OBJECT(character_table_burnside);

	CTB->do_it(deg, verbose_level);

	FREE_OBJECT(CTB);

	if (f_v) {
		cout << "interface_algebra::do_make_character_table_symmetric_group done" << endl;
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

void interface_algebra::do_surface_classify(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_surface_classify" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;


	if (f_v) {
		cout << "interface_algebra::do_surface_classify before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(argc, argv,
			F, LG,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "interface_algebra::do_surface_classify after Algebra.classify_surfaces" << endl;
	}


	if (f_v) {
		cout << "interface_algebra::do_surface_classify before SCW->generate_source_code" << endl;
	}
	SCW->generate_source_code(verbose_level);
	if (f_v) {
		cout << "interface_algebra::do_surface_classify after SCW->generate_source_code" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "interface_algebra::do_surface_classify done" << endl;
	}
}

void interface_algebra::do_surface_report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_surface_report" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;


	if (f_v) {
		cout << "interface_algebra::do_surface_report before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(argc, argv,
			F, LG,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "interface_algebra::do_surface_report after Algebra.classify_surfaces" << endl;
	}

	int f_with_stabilizers = TRUE;

	if (f_v) {
		cout << "interface_algebra::do_surface_report before SCW->create_report" << endl;
	}
	SCW->create_report(f_with_stabilizers, verbose_level - 1);
	if (f_v) {
		cout << "interface_algebra::do_surface_report after SCW->create_report" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "interface_algebra::do_surface_report done" << endl;
	}
}

void interface_algebra::do_surface_identify_Sa(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_surface_identify_Sa" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;


	if (f_v) {
		cout << "interface_algebra::do_surface_identify_Sa before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(argc, argv,
			F, LG,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "interface_algebra::do_surface_identify_Sa after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "interface_algebra::do_surface_identify_Sa before SCW->identify_Sa_and_print_table" << endl;
	}
	SCW->identify_Sa_and_print_table(verbose_level);
	if (f_v) {
		cout << "interface_algebra::do_surface_identify_Sa after SCW->identify_Sa_and_print_table" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "interface_algebra::do_surface_identify_Sa done" << endl;
	}
}

void interface_algebra::do_surface_isomorphism_testing(
		surface_create_description *surface_descr_isomorph1,
		surface_create_description *surface_descr_isomorph2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_surface_isomorphism_testing" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;


	if (f_v) {
		cout << "interface_algebra::do_surface_isomorphism_testing before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(argc, argv,
			F, LG,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "interface_algebra::do_surface_isomorphism_testing after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "interface_algebra::do_surface_isomorphism_testing before SCW->test_isomorphism" << endl;
	}
	SCW->test_isomorphism(
			surface_descr_isomorph1,
			surface_descr_isomorph2,
			verbose_level);
	if (f_v) {
		cout << "interface_algebra::do_surface_isomorphism_testing after SCW->test_isomorphism" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "interface_algebra::do_surface_isomorphism_testing done" << endl;
	}
}

void interface_algebra::do_surface_recognize(
		surface_create_description *surface_descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_surface_recognize" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;


	if (f_v) {
		cout << "interface_algebra::do_surface_recognize before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(argc, argv,
			F, LG,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "interface_algebra::do_surface_recognize after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "interface_algebra::do_surface_recognize before SCW->recognition" << endl;
	}
	SCW->recognition(
			surface_descr,
			verbose_level);
	if (f_v) {
		cout << "interface_algebra::do_surface_recognize after SCW->recognition" << endl;
	}

	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "interface_algebra::do_surface_recognize done" << endl;
	}
}



}}

