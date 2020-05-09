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
	f_young_symmetrizer = FALSE;
	young_symmetrizer_n = 0;
	f_young_symmetrizer_sym_4 = FALSE;
	f_classify_surfaces_through_arcs_and_trihedral_pairs = FALSE;
	f_poset_classification_control = FALSE;
	Control = NULL;
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
	else if (strcmp(argv[i], "-young_symmetrizer") == 0) {
		cout << "-young_symmetrizer <int : n>" << endl;
	}
	else if (strcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		cout << "-young_symmetrizer_sym_4 " << endl;
	}
	else if (strcmp(argv[i], "-classify_surfaces_through_arcs_and_trihedral_pairs") == 0) {
		cout << "-classify_surfaces_through_arcs_and_trihedral_pairs <int : q>" << endl;
	}
	else if (strcmp(argv[i], "-poset_classification_control") == 0) {
		cout << "-poset_classification_control <description>" << endl;
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
	else if (strcmp(argv[i], "-young_symmetrizer") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-classify_surfaces_through_arcs_and_trihedral_pairs") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-poset_classification_control") == 0) {
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

		else if (strcmp(argv[i], "-young_symmetrizer") == 0) {
			f_young_symmetrizer = TRUE;
			young_symmetrizer_n = atoi(argv[++i]);
			cout << "-young_symmetrizer " << young_symmetrizer_n << endl;
		}
		else if (strcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
			f_young_symmetrizer_sym_4 = TRUE;
			cout << "-young_symmetrizer_sym_4 " << endl;
		}
		else if (strcmp(argv[i], "-classify_surfaces_through_arcs_and_trihedral_pairs") == 0) {
			f_classify_surfaces_through_arcs_and_trihedral_pairs = TRUE;
			q = atoi(argv[++i]);
			cout << "-classify_surfaces_through_arcs_and_trihedral_pairs " << q << endl;
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
	else if (f_young_symmetrizer) {
		algebra_global_with_action Algebra;

		Algebra.young_symmetrizer(young_symmetrizer_n, verbose_level);
	}
	else if (f_young_symmetrizer_sym_4) {
		algebra_global_with_action Algebra;

		Algebra.young_symmetrizer_sym_4(verbose_level);
	}
	else if (f_classify_surfaces_through_arcs_and_trihedral_pairs) {
		classify_surfaces_through_arcs_and_trihedral_pairs(q,
				f_poset_classification_control, Control,
				verbose_level);
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

	{
	group_theoretic_activity Activity;

	Activity.init(Descr, F, LG, verbose_level);

	Activity.perform_activity(verbose_level);

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

void interface_algebra::classify_surfaces_through_arcs_and_trihedral_pairs(
		int q,
		int f_control, poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
		cout << "q=" << q << endl;
	}

	algebra_global_with_action A;
	surface_with_action *Surf_A;
	surface_domain *Surf;
	finite_field *F;
	poset_classification_control *my_control;
	number_theory_domain NT;


	if (f_control) {
		my_control = Control;
	}
	else {
		my_control = NEW_OBJECT(poset_classification_control);
	}
	F = NEW_OBJECT(finite_field);
	Surf = NEW_OBJECT(surface_domain);
	Surf_A = NEW_OBJECT(surface_with_action);

	F->init(q, 0);

	if (f_v) {
		cout << "before Surf->init" << endl;
	}
	Surf->init(F, verbose_level - 5);
	if (f_v) {
		cout << "after Surf->init" << endl;
	}

	int f_semilinear;

	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}


#if 0
	if (f_v) {
		cout << "before Surf->init_large_polynomial_domains" << endl;
	}
	Surf->init_large_polynomial_domains(0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf->init_large_polynomial_domains" << endl;
	}
#endif


	if (f_v) {
		cout << "before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, f_semilinear, 0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf_A->init" << endl;
	}



	if (f_v) {
		cout << "before Surf_A->Classify_trihedral_pairs->classify" << endl;
	}
	Surf_A->Classify_trihedral_pairs->classify(my_control, 0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf_A->Classify_trihedral_pairs->classify" << endl;
	}

	if (f_v) {
		cout << "before A.classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}
	A.classify_surfaces_through_arcs_and_trihedral_pairs(
			Surf_A, verbose_level);
	if (f_v) {
		cout << "after A.classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}


	if (f_v) {
		cout << "interface_algebra::classify_surfaces_through_arcs_and_trihedral_pairs done" << endl;
	}
}





}}

