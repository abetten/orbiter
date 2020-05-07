/*
 * group_theoretic_activity.cpp
 *
 *  Created on: May 5, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


group_theoretic_activity::group_theoretic_activity()
{
	Descr = NULL;
	F = NULL;
	LG = NULL;
	A = NULL;
}

group_theoretic_activity::~group_theoretic_activity()
{

}



void group_theoretic_activity::init(group_theoretic_activity_description *Descr,
		finite_field *F, linear_group *LG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::init" << endl;
	}

	group_theoretic_activity::Descr = Descr;
	group_theoretic_activity::F = F;
	group_theoretic_activity::LG = LG;


	A = LG->A2;

	if (f_v) {
		cout << "group_theoretic_activity::init group = " << LG->prefix << endl;
		cout << "group_theoretic_activity::init action = " << A->label << endl;
	}
	//cout << "created group " << LG->prefix << endl;

	if (f_v) {
		cout << "group_theoretic_activity::init done" << endl;
	}
}

void group_theoretic_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::perform_activity" << endl;
	}


	if (Descr->f_classes) {
		classes(verbose_level);
	}

	if (Descr->f_multiply) {
		multiply(verbose_level);
	}

	if (Descr->f_inverse) {
		inverse(verbose_level);
	}

	if (Descr->f_normalizer) {
		normalizer(verbose_level);
	}

	if (Descr->f_report) {
		report(verbose_level);
	}

	if (Descr->f_print_elements) {
		print_elements(verbose_level);
	}

	if (Descr->f_print_elements_tex) {
		print_elements_tex(verbose_level);
	}

	if (Descr->f_search_subgroup) {
		search_subgroup(verbose_level);
	}


	if (Descr->f_orbits_on_set_system_from_file) {
		orbits_on_set_system_from_file(verbose_level);
	}

	if (Descr->f_orbit_of_set_from_file) {
		orbits_on_set_from_file(verbose_level);
	}

	if (Descr->f_orbit_of) {
		orbit_of(verbose_level);
	} // if (f_orbit_of)


	if (Descr->f_orbits_on_points) {
		orbits_on_points(verbose_level);
	}

	if (Descr->f_orbits_on_subsets) {
		orbits_on_subsets(verbose_level);
	}

	if (Descr->f_classify_arcs) {
		do_classify_arcs(verbose_level);
	}

	else if (Descr->f_surface_classify) {
		do_surface_classify(verbose_level);
	}
	else if (Descr->f_surface_report) {
		do_surface_report(verbose_level);
	}
	else if (Descr->f_surface_identify_Sa) {
		do_surface_identify_Sa(verbose_level);
	}
	else if (Descr->f_surface_isomorphism_testing) {
		do_surface_isomorphism_testing(Descr->surface_descr_isomorph1, Descr->surface_descr_isomorph2, verbose_level);
	}
	else if (Descr->f_surface_recognize) {
		do_surface_recognize(Descr->surface_descr, verbose_level);
	}

	if (f_v) {
		cout << "group_theoretic_activity::perform_activity done" << endl;
	}
}

void group_theoretic_activity::classes(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::classes" << endl;
	}
	sims *G;

	G = LG->Strong_gens->create_sims(verbose_level);

	A->conjugacy_classes_and_normalizers(G,
			LG->prefix, LG->label_latex, verbose_level);

	FREE_OBJECT(G);
	if (f_v) {
		cout << "group_theoretic_activity::classes done" << endl;
	}
}


void group_theoretic_activity::multiply(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::multiply" << endl;
	}
	if (f_v) {
		cout << "multiplying" << endl;
		cout << "A=" << Descr->multiply_a << endl;
		cout << "B=" << Descr->multiply_b << endl;
	}
	int *Elt1;
	int *Elt2;
	int *Elt3;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	A->make_element_from_string(Elt1,
			Descr->multiply_a, verbose_level);
	if (f_v) {
		cout << "A=" << endl;
		A->element_print_quick(Elt1, cout);
	}

	A->make_element_from_string(Elt2,
			Descr->multiply_b, verbose_level);
	if (f_v) {
		cout << "B=" << endl;
		A->element_print_quick(Elt2, cout);
	}

	A->element_mult(Elt1, Elt2, Elt3, 0);
	if (f_v) {
		cout << "A*B=" << endl;
		A->element_print_quick(Elt3, cout);
		A->element_print_for_make_element(Elt3, cout);
		cout << endl;
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_v) {
		cout << "group_theoretic_activity::multiply done" << endl;
	}
}

void group_theoretic_activity::inverse(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::inverse" << endl;
	}
	if (f_v) {
		cout << "computing the inverse" << endl;
		cout << "A=" << Descr->inverse_a << endl;
	}
	int *Elt1;
	int *Elt2;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	A->make_element_from_string(Elt1,
			Descr->inverse_a, verbose_level);
	if (f_v) {
		cout << "A=" << endl;
		A->element_print_quick(Elt1, cout);
	}

	A->element_invert(Elt1, Elt2, 0);
	if (f_v) {
		cout << "A^-1=" << endl;
		A->element_print_quick(Elt2, cout);
		A->element_print_for_make_element(Elt2, cout);
		cout << endl;
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "group_theoretic_activity::inverse done" << endl;
	}
}

void group_theoretic_activity::normalizer(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::normalizer" << endl;
	}
	char fname_magma_prefix[1000];
	sims *G;
	sims *H;
	strong_generators *gens_N;
	longinteger_object N_order;


	sprintf(fname_magma_prefix, "%s_normalizer", LG->prefix);

	G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	if (f_v) {
	cout << "group order G = " << G->group_order_lint() << endl;
		cout << "group order H = " << H->group_order_lint() << endl;
		cout << "before A->normalizer_using_MAGMA" << endl;
	}
	A->normalizer_using_MAGMA(fname_magma_prefix,
			G, H, gens_N, verbose_level);

	if (f_v) {
		cout << "group order G = " << G->group_order_lint() << endl;
		cout << "group order H = " << H->group_order_lint() << endl;
	}
	gens_N->group_order(N_order);
	if (f_v) {
		cout << "group order N = " << N_order << endl;
		cout << "Strong generators for the normalizer of H are:" << endl;
		gens_N->print_generators_tex(cout);
		cout << "Strong generators for the normalizer of H as permutations are:" << endl;
		gens_N->print_generators_as_permutations();
	}

	sims *N;
	int N_goi;

	N = gens_N->create_sims(verbose_level);
	N_goi = N->group_order_lint();
	if (f_v) {
		cout << "The elements of N are:" << endl;
		N->print_all_group_elements();
	}

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
	if (f_v) {
		cout << "group_theoretic_activity::normalizer done" << endl;
	}
}

void group_theoretic_activity::report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::report" << endl;
	}
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
	if (f_v) {
		cout << "group_theoretic_activity::report done" << endl;
	}
}

void group_theoretic_activity::print_elements(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::print_elements" << endl;
	}
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
	if (f_v) {
		cout << "group_theoretic_activity::print_elements done" << endl;
	}
}

void group_theoretic_activity::print_elements_tex(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::print_elements_tex" << endl;
	}
	sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	longinteger_object go;

	Elt = NEW_int(A->elt_size_in_int);
	H->group_order(go);

#if 0
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
#endif

	char fname[1000];

	sprintf(fname, "%s_elements.tex", LG->prefix);


	{
		ofstream fp(fname);
		latex_interface L;
		L.head_easy(fp);

		//H->print_all_group_elements_tex(fp);
		H->print_all_group_elements_with_permutations_tex(fp);

		//Schreier.print_and_list_orbits_tex(fp);

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
	if (f_v) {
		cout << "group_theoretic_activity::print_elements_tex done" << endl;
	}
}

void group_theoretic_activity::search_subgroup(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::search_subgroup" << endl;
	}
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
	if (f_v) {
		cout << "group_theoretic_activity::search_subgroup done" << endl;
	}
}

void group_theoretic_activity::orbits_on_set_system_from_file(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_set_system_from_file" << endl;
	}
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
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_set_system_from_file done" << endl;
	}
}

void group_theoretic_activity::orbits_on_set_from_file(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_set_from_file" << endl;
	}
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
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_set_from_file done" << endl;
	}
}

void group_theoretic_activity::orbit_of(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::orbit_of" << endl;
	}
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
	if (f_v) {
		cout << "group_theoretic_activity::orbit_of done" << endl;
	}
}

void group_theoretic_activity::orbits_on_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_points" << endl;
	}
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
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_points done" << endl;
	}
}


void group_theoretic_activity::orbits_on_subsets(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subsets" << endl;
	}
	cout << "computing orbits on subsets:" << endl;
	poset_classification *PC;
	poset_classification_control *Control;
	poset *Poset;

	Poset = NEW_OBJECT(poset);
	Control = NEW_OBJECT(poset_classification_control);


	Poset->init_subset_lattice(A, A,
			LG->Strong_gens,
			verbose_level);
	PC = Poset->orbits_on_k_sets_compute(
			Control,
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
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subsets done" << endl;
	}
}

void group_theoretic_activity::do_classify_arcs(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_arcs" << endl;
	}

	{
	arc_generator *Gen;

	//finite_field *F;
	//action *A;

	action *A;

	A = LG->A2;

	Gen = NEW_OBJECT(arc_generator);

	//cout << argv[0] << endl;
	//cout << "before Gen->read_arguments" << endl;
	//Gen->read_arguments(argc, argv);


	Gen->f_starter = TRUE;
	Gen->f_target_size = TRUE;
	Gen->target_size = Descr->classify_arcs_target_size;
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_arcs target_size=" << Gen->target_size << endl;
	}


	const char *input_prefix = "";
	const char *base_fname = "";
	//int starter_size = 0;

	Gen->F = LG->F;
	Gen->q = LG->F->q;
	if (Descr->f_classify_arcs_d) {
		Gen->f_d = TRUE;
		Gen->d = Descr->classify_arcs_d;
		cout << "setting condition for no more than "
				<< Gen->d << " points per line" << endl;
	}
	else {
		Gen->f_d = FALSE;
		cout << "no arc condition" << endl;
	}
	Gen->verbose_level = verbose_level;
	if (Descr->f_exact_cover) {
		Gen->ECA = Descr->ECA;
		input_prefix = Gen->ECA->input_prefix;
		base_fname = Gen->ECA->base_fname;
		//starter_size = Gen->ECA->starter_size;
	}
	else {
		cout << "no exact cover" << endl;
		Gen->ECA = NULL;
	}
	if (Descr->f_isomorph_arguments) {
		Gen->IA = Descr->IA;
	}
	else {
		cout << "no isomorph arguments" << endl;
		Gen->IA = NULL;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_arcs before Gen->init" << endl;
	}
	Gen->init(LG->F,
		A, LG->Strong_gens,
		input_prefix,
		base_fname,
		Gen->target_size,
		//argc, argv,
		verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_arcs after Gen->init" << endl;
	}



	if (f_v) {
		cout << "group_theoretic_activity::do_classify_arcs before Gen->main" << endl;
	}
	Gen->main(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_arcs after Gen->main" << endl;
	}

#if 0
	if (Gen->f_starter) {
			cout << "preparing level spreadsheet" << endl;
			{
			spreadsheet *Sp;
			Gen->gen->make_spreadsheet_of_level_info(
					Sp, Gen->ECA->starter_size, Gen->verbose_level);
			char fname_csv[1000];
			sprintf(fname_csv, "arcs_%d_%d_level.csv",
					Gen->q, Gen->ECA->starter_size);
			Sp->save(fname_csv, Gen->verbose_level);
			delete Sp;
			}
			cout << "preparing orbit spreadsheet" << endl;
			{
			spreadsheet *Sp;
			Gen->gen->make_spreadsheet_of_orbit_reps(
					Sp, Gen->ECA->starter_size);
			char fname_csv[1000];
			sprintf(fname_csv, "arcs_%d_%d.csv",
					Gen->q, Gen->ECA->starter_size);
			Sp->save(fname_csv, Gen->verbose_level);
			delete Sp;
			}
			cout << "preparing orbit spreadsheet done" << endl;
	}

	if (f_draw_poset) {
		cout << "f_draw_poset verbose_level=" << verbose_level << endl;
		{
		char fname_poset[1000];

		Gen->gen->draw_poset_fname_base_poset_lvl(fname_poset, Gen->ECA->starter_size);
#if 0
		sprintf(fname_poset, "arcs_%d_poset_%d",
				Gen->q, Gen->ECA->starter_size);
#endif
		Gen->gen->draw_poset(fname_poset,
				Gen->ECA->starter_size /*depth*/,
				0 /* data1 */,
				f_embedded /* f_embedded */,
				FALSE /* f_sideways */,
				verbose_level);
		}
	}
	if (f_draw_full_poset) {
		cout << "f_draw_full_poset verbose_level=" << verbose_level << endl;
		{
		char fname_flag_orbits[1000];

		Gen->gen->draw_poset_fname_base_poset_lvl(fname_flag_orbits, Gen->ECA->starter_size);
		strcat(fname_flag_orbits, "_flag_orbits");

		Gen->gen->make_flag_orbits_on_relations(
				Gen->ECA->starter_size, fname_flag_orbits, verbose_level);
		}
	}
	if (f_report) {
		cout << "doing a report" << endl;

		file_io Fio;

		{
		char fname[1000];
		char title[1000];
		char author[1000];
		//int f_with_stabilizers = TRUE;

		sprintf(title, "Arcs over GF(%d) ", q);
		sprintf(author, "Orbiter");
		sprintf(fname, "Arcs_q%d.tex", q);

			{
			ofstream fp(fname);
			latex_interface L;

			//latex_head_easy(fp);
			L.head(fp,
				FALSE /* f_book */,
				TRUE /* f_title */,
				title, author,
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				TRUE /*f_enlarged_page */,
				TRUE /* f_pagenumbers*/,
				NULL /* extra_praeamble */);

			fp << "\\section{The field of order " << q << "}" << endl;
			fp << "\\noindent The field ${\\mathbb F}_{"
					<< Gen->q
					<< "}$ :\\\\" << endl;
			Gen->F->cheat_sheet(fp, verbose_level);

			fp << "\\section{The plane PG$(2, " << q << ")$}" << endl;

			fp << "The points in the plane PG$(2, " << q << ")$:\\\\" << endl;

			fp << "\\bigskip" << endl;


			Gen->P->cheat_sheet_points(fp, 0 /*verbose_level*/);


			int f_group_table = FALSE;
			double tikz_global_scale = 0.3;
			double tikz_global_line_width = 1.;
			int factor1000 = 1000;

			LG->report(fp, f_sylow, f_group_table,
					tikz_global_scale, tikz_global_line_width, factor1000,
					verbose_level);

			fp << endl;
			fp << "\\section{Poset Classification}" << endl;
			fp << endl;


			Gen->gen->report(fp);

			L.foot(fp);
			}
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
		if (f_recognize) {
			cout << "recognizing the set " << recognize_set_ascii << endl;
			long int *recognize_set;
			int recognize_set_sz;
			int *transporter;
			int *transporter_inv;
			int f_implicit_fusion = TRUE;
			int final_node = 0;

			lint_vec_scan(recognize_set_ascii, recognize_set, recognize_set_sz);
			cout << "set=";
			lint_vec_print(cout, recognize_set, recognize_set_sz);
			cout << endl;

			transporter = NEW_int(A->elt_size_in_int);
			transporter_inv = NEW_int(A->elt_size_in_int);
			Gen->gen->recognize(
					recognize_set, recognize_set_sz, transporter, f_implicit_fusion,
					final_node, verbose_level);
			cout << "final_node = " << final_node << endl;

			A->element_invert(transporter, transporter_inv, 0);

			cout << "transporter=" << endl;
			A->element_print(transporter, cout);
			cout << endl;

			cout << "transporter_inv=" << endl;
			A->element_print(transporter_inv, cout);
			cout << endl;

		}
	}
#endif


	FREE_OBJECT(Gen);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_arcs done" << endl;
	}
}

void group_theoretic_activity::do_surface_classify(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify after Algebra.classify_surfaces" << endl;
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify before SCW->generate_source_code" << endl;
	}
	SCW->generate_source_code(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify after SCW->generate_source_code" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "interface_algebra::do_surface_classify done" << endl;
	}
}

void group_theoretic_activity::do_surface_report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report after Algebra.classify_surfaces" << endl;
	}

	int f_with_stabilizers = TRUE;

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report before SCW->create_report" << endl;
	}
	SCW->create_report(f_with_stabilizers, verbose_level - 1);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report after SCW->create_report" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "interface_algebra::do_surface_report done" << endl;
	}
}

void group_theoretic_activity::do_surface_identify_Sa(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Sa" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Sa before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Sa after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Sa before SCW->identify_Sa_and_print_table" << endl;
	}
	SCW->identify_Sa_and_print_table(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Sa after SCW->identify_Sa_and_print_table" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Sa done" << endl;
	}
}

void group_theoretic_activity::do_surface_isomorphism_testing(
		surface_create_description *surface_descr_isomorph1,
		surface_create_description *surface_descr_isomorph2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing before SCW->test_isomorphism" << endl;
	}
	SCW->test_isomorphism(
			surface_descr_isomorph1,
			surface_descr_isomorph2,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing after SCW->test_isomorphism" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing done" << endl;
	}
}

void group_theoretic_activity::do_surface_recognize(
		surface_create_description *surface_descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize before SCW->recognition" << endl;
	}
	SCW->recognition(
			surface_descr,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize after SCW->recognition" << endl;
	}

	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize done" << endl;
	}
}

}}

