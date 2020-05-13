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

	orbits_on_subspaces_Poset = NULL;
	orbits_on_subspaces_PC = NULL;
	orbits_on_subspaces_VS = NULL;
	orbits_on_subspaces_M = NULL;
	orbits_on_subspaces_base_cols = NULL;

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
		do_surface_isomorphism_testing(Descr->surface_descr_isomorph1,
				Descr->surface_descr_isomorph2, verbose_level);
	}
	else if (Descr->f_surface_recognize) {
		do_surface_recognize(Descr->surface_descr, verbose_level);
	}

	if (Descr->f_orbits_on_subsets) {
		orbits_on_subsets(verbose_level);
	}

	else if (Descr->f_orbits_on_subspaces) {
		orbits_on_subspaces(verbose_level);
	}
	else if (Descr->f_classify_surfaces_through_arcs_and_trihedral_pairs) {
		do_classify_surfaces_through_arcs_and_trihedral_pairs(verbose_level);
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
	poset_classification *PC;
	poset_classification_control *Control;
	poset *Poset;

	Poset = NEW_OBJECT(poset);


	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	Poset->init_subset_lattice(A, A,
			LG->Strong_gens,
			verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subsets before Poset->orbits_on_k_sets_compute" << endl;
	}
	PC = Poset->orbits_on_k_sets_compute(
			Control,
			Descr->orbits_on_subsets_size, verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subsets after Poset->orbits_on_k_sets_compute" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subsets before orbits_on_poset_post_processing" << endl;
	}
	orbits_on_poset_post_processing(
			PC, Descr->orbits_on_subsets_size, verbose_level);


	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subsets done" << endl;
	}
}

void group_theoretic_activity::orbits_on_subspaces(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subspaces" << endl;
	}

	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}

	Control->f_max_depth = TRUE;
	Control->max_depth = Descr->orbits_on_subspaces_depth;
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subspaces Control->max_depth=" << Control->max_depth << endl;
	}

	int n;

	n = LG->n;

	orbits_on_subspaces_PC = NEW_OBJECT(poset_classification);
	orbits_on_subspaces_Poset = NEW_OBJECT(poset);



	orbits_on_subspaces_M = NEW_int(n * n);
	orbits_on_subspaces_base_cols = NEW_int(n);

	orbits_on_subspaces_VS = NEW_OBJECT(vector_space);
	orbits_on_subspaces_VS->init(LG->F, n /* dimension */, verbose_level - 1);
	orbits_on_subspaces_VS->init_rank_functions(
			gta_subspace_orbits_rank_point_func,
			gta_subspace_orbits_unrank_point_func,
			this,
			verbose_level - 1);



	if (Descr->f_print_generators) {
		int f_print_as_permutation = FALSE;
		int f_offset = TRUE;
		int offset = 1;
		int f_do_it_anyway_even_for_big_degree = TRUE;
		int f_print_cycles_of_length_one = TRUE;

		cout << "group_theoretic_activity::orbits_on_subspaces "
				"printing generators "
				"for the group:" << endl;
		LG->Strong_gens->gens->print(cout,
			f_print_as_permutation,
			f_offset, offset,
			f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one);
	}

	orbits_on_subspaces_Poset = NEW_OBJECT(poset);
	orbits_on_subspaces_Poset->init_subspace_lattice(LG->A_linear,
			LG->A2, LG->Strong_gens,
			orbits_on_subspaces_VS,
			verbose_level);
	orbits_on_subspaces_Poset->add_testing_without_group(
			gta_subspace_orbits_early_test_func,
				this /* void *data */,
				verbose_level);



	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subspaces "
				"LG->prefix=" << LG->prefix << endl;
	}

	sprintf(orbits_on_subspaces_PC->fname_base, "%s", LG->prefix);

	orbits_on_subspaces_PC->init(Control, orbits_on_subspaces_Poset,
			Control->max_depth, verbose_level);



	int nb_poset_orbit_nodes = 1000;

	if (f_v) {
		cout << "subspace_orbits->init_subspace_lattice "
				"before Gen->init_poset_orbit_node" << endl;
	}
	orbits_on_subspaces_PC->init_poset_orbit_node(
			nb_poset_orbit_nodes, verbose_level - 1);
	if (f_v) {
		cout << "subspace_orbits->init_subspace_lattice "
				"calling Gen->init_root_node" << endl;
	}
	orbits_on_subspaces_PC->root[0].init_root_node(orbits_on_subspaces_PC, verbose_level - 1);

	int schreier_depth = Control->max_depth;
	int f_use_invariant_subset_if_available = FALSE;
	int f_debug = FALSE;
	int nb_orbits;

	os_interface Os;
	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subspaces "
				"calling generator_main" << endl;
		cout << "A=";
		orbits_on_subspaces_PC->Poset->A->print_info();
		cout << "A2=";
		orbits_on_subspaces_PC->Poset->A2->print_info();
	}
	orbits_on_subspaces_PC->main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 1);


	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subspaces "
				"done with generator_main" << endl;
	}
	nb_orbits = orbits_on_subspaces_PC->nb_orbits_at_level(orbits_on_subspaces_PC->depth);
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subspaces we found "
				<< nb_orbits << " orbits at depth " << orbits_on_subspaces_PC->depth << endl;
	}

	orbits_on_poset_post_processing(
			orbits_on_subspaces_PC, Control->max_depth, verbose_level);


	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subspaces done" << endl;
	}
}

void group_theoretic_activity::orbits_on_poset_post_processing(
		poset_classification *PC,
		int depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_poset_post_processing" << endl;
	}


	for (int d = 0; d <= depth; d++) {
		cout << "There are " << PC->nb_orbits_at_level(d)
				<< " orbits on subsets of size " << d << ":" << endl;

		if (d < Descr->orbits_on_subsets_size) {
			//continue;
		}
		PC->list_all_orbits_at_level(d,
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
		sprintf(fname_poset, "%s_poset_%d", LG->prefix, depth);
		PC->draw_poset(fname_poset,
				depth /*depth*/, 0 /* data1 */,
				TRUE /* f_embedded */,
				FALSE /* f_sideways */,
				0 /* verbose_level */);
		}
	}

	if (Descr->f_draw_full_poset) {
		{
		char fname_poset[1000];
		sprintf(fname_poset, "%s_poset_%d", LG->prefix, depth);
		//double x_stretch = 0.4;
		PC->draw_poset_full(fname_poset, depth,
			0 /* data1 */, Descr->f_embedded, Descr->f_sideways,
			Descr->x_stretch, 0 /*verbose_level */);

		const char *fname_prefix = "flag_orbits";

		PC->make_flag_orbits_on_relations(
				depth, fname_prefix, verbose_level);

		}
	}




	if (Descr->f_test_if_geometric) {
		int d = Descr->test_if_geometric_depth;

		//for (depth = 0; depth <= orbits_on_subsets_size; depth++) {

		cout << "Orbits on subsets of size " << d << ":" << endl;
		PC->list_all_orbits_at_level(d,
				FALSE /* f_has_print_function */,
				NULL /* void (*print_function)(ostream &ost, int len, int *S, void *data)*/,
				NULL /* void *print_function_data*/,
				TRUE /* f_show_orbit_decomposition */,
				TRUE /* f_show_stab */,
				FALSE /* f_save_stab */,
				TRUE /* f_show_whole_orbit*/);
		int nb_orbits, orbit_idx;

		nb_orbits = PC->nb_orbits_at_level(d);
		for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {

			int orbit_length;
			long int *Orbit;

			cout << "before PC->get_whole_orbit depth " << d
					<< " orbit " << orbit_idx
					<< " / " << nb_orbits << ":" << endl;
			PC->get_whole_orbit(
					d, orbit_idx,
					Orbit, orbit_length, verbose_level);
			cout << "depth " << d << " orbit " << orbit_idx
					<< " / " << nb_orbits << " has length "
					<< orbit_length << ":" << endl;
			lint_matrix_print(Orbit, orbit_length, d);

			action *Aut;
			longinteger_object ago;
			nauty_interface_with_group Nauty;

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

			cout << "before PC->get_whole_orbit depth " << d
					<< " orbit " << orbit_idx
					<< " / " << nb_orbits << ":" << endl;
			PC->get_whole_orbit(
					depth, 0 /* orbit_idx*/,
					Orbit1, orbit_length1, verbose_level);
			cout << "depth " << d << " orbit " << 0
					<< " / " << nb_orbits << " has length "
					<< orbit_length1 << ":" << endl;
			lint_matrix_print(Orbit1, orbit_length1, d);

			PC->get_whole_orbit(
					depth, 1 /* orbit_idx*/,
					Orbit2, orbit_length2, verbose_level);
			cout << "depth " << d << " orbit " << 1
					<< " / " << nb_orbits << " has length "
					<< orbit_length2 << ":" << endl;
			lint_matrix_print(Orbit2, orbit_length2, d);

			action *Aut;
			longinteger_object ago;
			nauty_interface_with_group Nauty;

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


	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_poset_post_processing done" << endl;
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
		cout << "group_theoretic_activity::do_classify_arcs "
				"target_size=" << Gen->target_size << endl;
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
		//Gen->ECA = Descr->ECA;
		input_prefix = Descr->ECA->input_prefix;
		base_fname = Descr->ECA->base_fname;
		//starter_size = Gen->ECA->starter_size;
	}
	else {
		cout << "no exact cover" << endl;
		Descr->ECA = NULL;
		input_prefix = "";
		base_fname = "";
	}
	if (Descr->f_isomorph_arguments) {
		//Gen->IA = Descr->IA;
	}
	else {
		cout << "no isomorph arguments" << endl;
		//Gen->IA = NULL;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_arcs before Gen->init" << endl;
	}
	Gen->init(this,
			LG->F,
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

	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
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
	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
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
	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Sa before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
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
	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
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
	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
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



int group_theoretic_activity::subspace_orbits_test_set(
		int len, long int *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret = TRUE;
	int rk;
	int n;
	finite_field *F;

	if (f_v) {
		cout << "group_theoretic_activity::subspace_orbits_test_set" << endl;
		cout << "Testing set ";
		lint_vec_print(cout, S, len);
		cout << endl;
		cout << "LG->n=" << LG->n << endl;
	}
	n = LG->n;
	F = LG->F;

	F->PG_elements_unrank_lint(
			orbits_on_subspaces_M, len, n, S);

	if (f_vv) {
		cout << "coordinate matrix:" << endl;
		print_integer_matrix_width(cout,
				orbits_on_subspaces_M, len, n, n, F->log10_of_q);
	}

	rk = F->Gauss_simple(orbits_on_subspaces_M, len, n,
			orbits_on_subspaces_base_cols, 0 /*verbose_level - 2*/);

	if (f_v) {
		cout << "the matrix has rank " << rk << endl;
	}

	if (rk < len) {
		ret = FALSE;
	}

#if 0
	if (ret) {
		if (f_has_extra_test_func) {
			ret = (*extra_test_func)(this,
					len, S, extra_test_func_data, verbose_level);
		}
	}
#endif

	if (ret) {
		if (f_v) {
			cout << "group_theoretic_activity::subspace_orbits_test_set OK" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "group_theoretic_activity::subspace_orbits_test_set not OK" << endl;
		}
	}
	return ret;
}


void group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	algebra_global_with_action Algebra;
	surface_with_action *Surf_A;
	surface_domain *Surf;
	poset_classification_control *my_control;
	number_theory_domain NT;


	if (Descr->f_poset_classification_control) {
		my_control = Descr->Control;
	}
	else {
		my_control = NEW_OBJECT(poset_classification_control);
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf_A = NEW_OBJECT(surface_with_action);


	if (f_v) {
		cout << "before Surf->init" << endl;
	}
	Surf->init(F, verbose_level - 5);
	if (f_v) {
		cout << "after Surf->init" << endl;
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
	Surf_A->init(Surf, LG, verbose_level - 1);
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
	Algebra.classify_surfaces_through_arcs_and_trihedral_pairs(
			this,
			Surf_A, verbose_level);
	if (f_v) {
		cout << "after A.classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs done" << endl;
	}

}

void group_theoretic_activity::do_create_surface(
		surface_create_description *Surface_Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface" << endl;
	}

	int q;
	int i;
	int f_semilinear;
	finite_field *F;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	number_theory_domain NT;
	sorting Sorting;
	file_io Fio;

	q = Surface_Descr->get_q();
	cout << "q=" << q << endl;

	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}


	F = NEW_OBJECT(finite_field);
	F->init(q, 0);


	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface before Surf->init" << endl;
		}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, 0/*verbose_level - 1*/);
	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface after Surf->init" << endl;
		}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface before Surf_A->init" << endl;
		}
	Surf_A->init(Surf, LG, 0 /*verbose_level*/);
	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface after Surf_A->init" << endl;
		}


	surface_create *SC;
	SC = NEW_OBJECT(surface_create);

	cout << "before SC->init" << endl;
	SC->init(Surface_Descr, Surf_A, verbose_level);
	cout << "after SC->init" << endl;

	if (Descr->nb_transform) {
		cout << "group_theoretic_activity::do_create_surface "
				"before SC->apply_transformations" << endl;
		SC->apply_transformations(Descr->transform_coeffs,
				Descr->f_inverse_transform, Descr->nb_transform, verbose_level);
		cout << "group_theoretic_activity::do_create_surface "
				"after SC->apply_transformations" << endl;
		}

	int coeffs_out[20];
	action *A;
	//int *Elt1;
	int *Elt2;

	A = SC->Surf_A->A;

	Elt2 = NEW_int(A->elt_size_in_int);

	SC->F->init_symbol_for_print("\\omega");

	if (SC->F->e == 1) {
		SC->F->f_print_as_exponentials = FALSE;
	}

	SC->F->PG_element_normalize(SC->coeffs, 1, 20);

	cout << "group_theoretic_activity::do_create_surface "
			"We have created the following surface:" << endl;
	cout << "$$" << endl;
	SC->Surf->print_equation_tex(cout, SC->coeffs);
	cout << endl;
	cout << "$$" << endl;


	if (SC->f_has_group) {
		for (i = 0; i < SC->Sg->gens->len; i++) {
			cout << "Testing generator " << i << " / "
					<< SC->Sg->gens->len << endl;
			A->element_invert(SC->Sg->gens->ith(i),
					Elt2, 0 /*verbose_level*/);



			matrix_group *M;

			M = A->G.matrix_grp;
			M->substitute_surface_equation(Elt2,
					SC->coeffs, coeffs_out, SC->Surf,
					verbose_level - 1);


			if (int_vec_compare(SC->coeffs, coeffs_out, 20)) {
				cout << "error, the transformation does not preserve "
						"the equation of the surface" << endl;
				exit(1);
			}
			cout << "Generator " << i << " / " << SC->Sg->gens->len
					<< " is good" << endl;
		}
	}
	else {
		cout << "We do not have information about "
				"the automorphism group" << endl;
	}


	cout << "We have created the surface " << SC->label_txt << ":" << endl;
	cout << "$$" << endl;
	SC->Surf->print_equation_tex(cout, SC->coeffs);
	cout << endl;
	cout << "$$" << endl;

	if (SC->f_has_group) {
		cout << "The stabilizer is generated by:" << endl;
		SC->Sg->print_generators_tex(cout);

		if (SC->f_has_nice_gens) {
			cout << "The stabilizer is generated by the following nice generators:" << endl;
			SC->nice_gens->print_tex(cout);

		}
	}

	if (SC->f_has_lines) {
		cout << "The lines are:" << endl;
		SC->Surf->Gr->print_set_tex(cout, SC->Lines, 27);


		surface_object *SO;

		SO = NEW_OBJECT(surface_object);
		if (f_v) {
			cout << "before SO->init" << endl;
			}
		SO->init(SC->Surf, SC->Lines, SC->coeffs,
				FALSE /*f_find_double_six_and_rearrange_lines */, verbose_level);
		if (f_v) {
			cout << "after SO->init" << endl;
			}

		char fname_points[1000];

		sprintf(fname_points, "surface_%s_points.txt", SC->label_txt);
		Fio.write_set_to_file(fname_points,
				SO->Pts, SO->nb_pts, 0 /*verbose_level*/);
		cout << "Written file " << fname_points << " of size "
				<< Fio.file_size(fname_points) << endl;
	}
	else {
		cout << "The surface " << SC->label_txt
				<< " does not come with lines" << endl;
	}




	if (SC->f_has_group) {

		cout << "creating surface_object_with_action object" << endl;

		surface_object_with_action *SoA;

		SoA = NEW_OBJECT(surface_object_with_action);

		if (SC->f_has_lines) {
			cout << "creating surface using the known lines (which are "
					"arranged with respect to a double six):" << endl;
			SoA->init(SC->Surf_A,
				SC->Lines,
				SC->coeffs,
				SC->Sg,
				FALSE /*f_find_double_six_and_rearrange_lines*/,
				SC->f_has_nice_gens, SC->nice_gens,
				verbose_level);
			}
		else {
			cout << "creating surface from equation only "
					"(no lines):" << endl;
			SoA->init_equation(SC->Surf_A,
				SC->coeffs,
				SC->Sg,
				verbose_level);
			}
		cout << "The surface has been created." << endl;




		six_arcs_not_on_a_conic *Six_arcs;
		int *transporter;

		Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);


		// classify six arcs not on a conic:

		cout << "Classifying six-arcs not on a conic:" << endl;

		action *A;

		A = NEW_OBJECT(action);

		group_theoretic_activity *GTA;

		GTA = NEW_OBJECT(group_theoretic_activity);

		int f_semilinear = TRUE;
		number_theory_domain NT;

		if (NT.is_prime(F->q)) {
			f_semilinear = FALSE;
			}

		{
			vector_ge *nice_gens;
			A->init_projective_group(3, F,
					f_semilinear, TRUE /*f_basis*/, TRUE /* f_init_sims */,
					nice_gens,
					0 /*verbose_level*/);
			FREE_OBJECT(nice_gens);
		}
		Six_arcs->init(GTA,
				SC->F,
				A,
			SC->Surf->P2,
			//argc, argv,
			verbose_level);
		transporter = NEW_int(Six_arcs->Gen->A->elt_size_in_int);




		char fname[1000];
		char fname_mask[1000];
		char label[1000];
		char label_tex[1000];

		sprintf(fname, "surface_%s.tex", SC->prefix);
		sprintf(label, "surface_%s", SC->label_txt);
		sprintf(label_tex, "surface %s", SC->label_tex);
		sprintf(fname_mask, "surface_%s_orbit_%%d", SC->prefix);
		{
			ofstream fp(fname);
			latex_interface L;

			L.head_easy(fp);


			fp << "\\section{The Finite Field $\\mathbb F_{" << q << "}$}" << endl;
			SC->F->cheat_sheet(fp, verbose_level);

			fp << "\\bigskip" << endl;

			SoA->cheat_sheet(fp,
				label,
				label_tex,
				TRUE /* f_print_orbits */,
				fname_mask /* const char *fname_mask*/,
				verbose_level);

			fp << "\\setlength{\\parindent}{0pt}" << endl;

			if (Descr->f_surface_clebsch) {

				surface_object *SO;
				SO = SoA->SO;

				fp << endl;
				fp << "\\bigskip" << endl;
				fp << endl;
				fp << "\\section{Points on the surface}" << endl;
				fp << endl;

				SO->print_affine_points_in_source_code(fp);


				fp << endl;
				fp << "\\bigskip" << endl;
				fp << endl;

				fp << "\\section{Clebsch maps}" << endl;

				SC->Surf->latex_table_of_clebsch_maps(fp);


				fp << endl;
				fp << "\\clearpage" << endl;
				fp << endl;



				fp << "\\section{Six-arcs not on a conic}" << endl;
				fp << endl;


				//fp << "The six-arcs not on a conic are:\\\\" << endl;
				Six_arcs->report_latex(fp);


				if (Descr->f_surface_codes) {

					homogeneous_polynomial_domain *HPD;

					HPD = NEW_OBJECT(homogeneous_polynomial_domain);

					HPD->init(F, 3, 2 /* degree */,
							TRUE /* f_init_incidence_structure */,
							verbose_level);

					action *A_on_poly;

					A_on_poly = NEW_OBJECT(action);
					A_on_poly->induced_action_on_homogeneous_polynomials(A,
						HPD,
						FALSE /* f_induce_action */, NULL,
						verbose_level);

					cout << "created action A_on_poly" << endl;
					A_on_poly->print_info();

					schreier *Sch;
					longinteger_object full_go;

					//Sch = new schreier;
					//A2->all_point_orbits(*Sch, verbose_level);

					cout << "computing orbits:" << endl;

					Sch = A->Strong_gens->orbits_on_points_schreier(A_on_poly, verbose_level);

					//SC->Sg->
					//Sch = SC->Sg->orbits_on_points_schreier(A_on_poly, verbose_level);

					orbit_transversal *T;

					A->group_order(full_go);
					T = NEW_OBJECT(orbit_transversal);

					cout << "before T->init_from_schreier" << endl;

					T->init_from_schreier(
							Sch,
							A,
							full_go,
							verbose_level);

					cout << "after T->init_from_schreier" << endl;

					Sch->print_orbit_reps(cout);

					cout << "orbit reps:" << endl;

					fp << "\\section{Orbits on conics}" << endl;
					fp << endl;

					T->print_table_latex(
							fp,
							TRUE /* f_has_callback */,
							HPD_callback_print_function2,
							HPD /* callback_data */,
							TRUE /* f_has_callback */,
							HPD_callback_print_function,
							HPD /* callback_data */,
							verbose_level);


				}


#if 0

				int *Arc_iso; // [72]
				int *Clebsch_map; // [nb_pts]
				int *Clebsch_coeff; // [nb_pts * 4]
				//int line_a, line_b;
				//int transversal_line;
				int tritangent_plane_rk;
				int plane_rk_global;
				int ds, ds_row;

				fp << endl;
				fp << "\\clearpage" << endl;
				fp << endl;

				fp << "\\section{Clebsch maps in detail}" << endl;
				fp << endl;




				Arc_iso = NEW_int(72);
				Clebsch_map = NEW_int(SO->nb_pts);
				Clebsch_coeff = NEW_int(SO->nb_pts * 4);

				for (ds = 0; ds < 36; ds++) {
					for (ds_row = 0; ds_row < 2; ds_row++) {
						SC->Surf->prepare_clebsch_map(
								ds, ds_row,
								line_a, line_b,
								transversal_line,
								0 /*verbose_level */);


						fp << endl;
						fp << "\\bigskip" << endl;
						fp << endl;
						fp << "\\subsection{Clebsch map for double six "
								<< ds << ", row " << ds_row << "}" << endl;
						fp << endl;



						cout << "computing clebsch map:" << endl;
						SO->compute_clebsch_map(line_a, line_b,
							transversal_line,
							tritangent_plane_rk,
							Clebsch_map,
							Clebsch_coeff,
							verbose_level);


						plane_rk_global = SO->Tritangent_planes[
							SO->Eckardt_to_Tritangent_plane[
								tritangent_plane_rk]];

						int Arc[6];
						int Arc2[6];
						int Blown_up_lines[6];
						int perm[6];

						SO->clebsch_map_find_arc_and_lines(
								Clebsch_map,
								Arc,
								Blown_up_lines,
								0 /* verbose_level */);

						for (j = 0; j < 6; j++) {
							perm[j] = j;
							}

						int_vec_heapsort_with_log(Blown_up_lines, perm, 6);
						for (j = 0; j < 6; j++) {
							Arc2[j] = Arc[perm[j]];
							}


						fp << endl;
						fp << "\\bigskip" << endl;
						fp << endl;
						//fp << "\\section{Clebsch map}" << endl;
						//fp << endl;
						fp << "Line 1 = $";
						fp << SC->Surf->Line_label_tex[line_a];
						fp << "$\\\\" << endl;
						fp << "Line 2 = $";
						fp << SC->Surf->Line_label_tex[line_b];
						fp << "$\\\\" << endl;
						fp << "Transversal line $";
						fp << SC->Surf->Line_label_tex[transversal_line];
						fp << "$\\\\" << endl;
						fp << "Image plane $\\pi_{" << tritangent_plane_rk
								<< "}=" << plane_rk_global << "=$\\\\" << endl;
						fp << "$$" << endl;

						fp << "\\left[" << endl;
						SC->Surf->Gr3->print_single_generator_matrix_tex(
								fp, plane_rk_global);
						fp << "\\right]," << endl;

						fp << "$$" << endl;
						fp << "Arc $";
						int_set_print_tex(fp, Arc2, 6);
						fp << "$\\\\" << endl;
						fp << "Half double six: $";
						int_set_print_tex(fp, Blown_up_lines, 6);
						fp << "=\\{";
						for (j = 0; j < 6; j++) {
							fp << SC->Surf->Line_label_tex[Blown_up_lines[j]];
							fp << ", ";
							}
						fp << "\\}$\\\\" << endl;

						fp << "The arc consists of the following "
								"points:\\\\" << endl;
						display_table_of_projective_points(fp,
								SC->F, Arc2, 6, 3);

						int orbit_at_level, idx;
						Six_arcs->Gen->gen->identify(Arc2, 6,
								transporter, orbit_at_level,
								0 /*verbose_level */);


						if (!int_vec_search(Six_arcs->Not_on_conic_idx,
							Six_arcs->nb_arcs_not_on_conic,
							orbit_at_level,
							idx)) {
							cout << "could not find orbit" << endl;
							exit(1);
							}

						fp << "The arc is isomorphic to arc " << orbit_at_level
								<< " in the original classification.\\\\" << endl;
						fp << "The arc is isomorphic to arc " << idx
								<< " in the list.\\\\" << endl;
						Arc_iso[2 * ds + ds_row] = idx;



						SO->clebsch_map_latex(fp, Clebsch_map, Clebsch_coeff);

						//SO->clebsch_map_print_fibers(Clebsch_map);
						}
					}



				fp << "The isomorphism type of arc associated with "
						"each half-double six is:" << endl;
				fp << "$$" << endl;
				print_integer_matrix_with_standard_labels(fp,
						Arc_iso, 36, 2, TRUE);
				fp << "$$" << endl;

				FREE_int(Arc_iso);
				FREE_int(Clebsch_map);
				FREE_int(Clebsch_coeff);
#endif


#if 0
				fp << endl;
				fp << "\\clearpage" << endl;
				fp << endl;


				fp << "\\section{Clebsch maps in detail by orbits "
						"on half-double sixes}" << endl;
				fp << endl;



				fp << "There are " << SoA->Orbits_on_single_sixes->nb_orbits
						<< "orbits on half double sixes\\\\" << endl;

				Arc_iso = NEW_int(SoA->Orbits_on_single_sixes->nb_orbits);
				Clebsch_map = NEW_int(SO->nb_pts);
				Clebsch_coeff = NEW_int(SO->nb_pts * 4);

				int j, f, l, k;

				for (j = 0; j < SoA->Orbits_on_single_sixes->nb_orbits; j++) {

					int line1, line2, transversal_line;

					if (f_v) {
						cout << "surface_with_action::arc_lifting_and_classify "
							"orbit on single sixes " << j << " / "
							<< SoA->Orbits_on_single_sixes->nb_orbits << ":" << endl;
					}

					fp << "\\subsection*{Orbit on single sixes " << j << " / "
						<< SoA->Orbits_on_single_sixes->nb_orbits << "}" << endl;

					f = SoA->Orbits_on_single_sixes->orbit_first[j];
					l = SoA->Orbits_on_single_sixes->orbit_len[j];
					if (f_v) {
						cout << "orbit f=" << f <<  " l=" << l << endl;
						}
					k = SoA->Orbits_on_single_sixes->orbit[f];

					if (f_v) {
						cout << "The half double six is no " << k << " : ";
						int_vec_print(cout, SoA->Surf->Half_double_sixes + k * 6, 6);
						cout << endl;
						}

					int h;

					fp << "The half double six is no " << k << "$ = "
							<< Surf->Half_double_six_label_tex[k] << "$ : $";
					int_vec_print(fp, Surf->Half_double_sixes + k * 6, 6);
					fp << " = \\{" << endl;
					for (h = 0; h < 6; h++) {
						fp << Surf->Line_label_tex[
								Surf->Half_double_sixes[k * 6 + h]];
						if (h < 6 - 1) {
							fp << ", ";
							}
						}
					fp << "\\}$\\\\" << endl;

					ds = k / 2;
					ds_row = k % 2;

					SC->Surf->prepare_clebsch_map(
							ds, ds_row,
							line1, line2,
							transversal_line,
							0 /*verbose_level */);

					fp << endl;
					fp << "\\bigskip" << endl;
					fp << endl;
					fp << "\\subsection{Clebsch map for double six "
							<< ds << ", row " << ds_row << "}" << endl;
					fp << endl;



					cout << "computing clebsch map:" << endl;
					SO->compute_clebsch_map(line1, line2,
						transversal_line,
						tritangent_plane_rk,
						Clebsch_map,
						Clebsch_coeff,
						verbose_level);


					plane_rk_global = SO->Tritangent_planes[
						SO->Eckardt_to_Tritangent_plane[
							tritangent_plane_rk]];

					int Arc[6];
					int Arc2[6];
					int Blown_up_lines[6];
					int perm[6];

					SO->clebsch_map_find_arc_and_lines(
							Clebsch_map,
							Arc,
							Blown_up_lines,
							0 /* verbose_level */);

					for (h = 0; h < 6; h++) {
						perm[h] = h;
						}

					Sorting.int_vec_heapsort_with_log(Blown_up_lines, perm, 6);
					for (h = 0; h < 6; h++) {
						Arc2[h] = Arc[perm[h]];
						}


					fp << endl;
					fp << "\\bigskip" << endl;
					fp << endl;
					//fp << "\\section{Clebsch map}" << endl;
					//fp << endl;
					fp << "Line 1 = $";
					fp << SC->Surf->Line_label_tex[line1];
					fp << "$\\\\" << endl;
					fp << "Line 2 = $";
					fp << SC->Surf->Line_label_tex[line2];
					fp << "$\\\\" << endl;
					fp << "Transversal line $";
					fp << SC->Surf->Line_label_tex[transversal_line];
					fp << "$\\\\" << endl;
					fp << "Image plane $\\pi_{" << tritangent_plane_rk
							<< "}=" << plane_rk_global << "=$\\\\" << endl;
					fp << "$$" << endl;

					fp << "\\left[" << endl;
					SC->Surf->Gr3->print_single_generator_matrix_tex(
							fp, plane_rk_global);
					fp << "\\right]," << endl;

					fp << "$$" << endl;
					fp << "Arc $";
					int_set_print_tex(fp, Arc2, 6);
					fp << "$\\\\" << endl;
					fp << "Half double six: $";
					int_set_print_tex(fp, Blown_up_lines, 6);
					fp << "=\\{";
					for (h = 0; h < 6; h++) {
						fp << SC->Surf->Line_label_tex[Blown_up_lines[h]];
						fp << ", ";
						}
					fp << "\\}$\\\\" << endl;

					fp << "The arc consists of the following "
							"points:\\\\" << endl;
					SC->F->display_table_of_projective_points(fp,
							Arc2, 6, 3);

					int orbit_at_level, idx;
					Six_arcs->Gen->gen->identify(Arc2, 6,
							transporter, orbit_at_level,
							0 /*verbose_level */);


					if (!Sorting.int_vec_search(Six_arcs->Not_on_conic_idx,
						Six_arcs->nb_arcs_not_on_conic,
						orbit_at_level,
						idx)) {
						cout << "could not find orbit" << endl;
						exit(1);
						}

					fp << "The arc is isomorphic to arc " << orbit_at_level
							<< " in the original classification.\\\\" << endl;
					fp << "The arc is isomorphic to arc " << idx
							<< " in the list.\\\\" << endl;
					Arc_iso[j] = idx;



					SO->clebsch_map_latex(fp, Clebsch_map, Clebsch_coeff);

				} // next j

				fp << "The isomorphism type of arc associated with "
						"each half-double six is:" << endl;
				fp << "$$" << endl;
				int_vec_print(fp,
						Arc_iso, SoA->Orbits_on_single_sixes->nb_orbits);
				fp << "$$" << endl;



				FREE_int(Arc_iso);
				FREE_int(Clebsch_map);
				FREE_int(Clebsch_coeff);

#endif



				if (Descr->f_surface_quartic) {
					SoA->quartic(fp, verbose_level);
					}


			}


			L.foot(fp);
		}
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;



		FREE_OBJECT(SoA);
		FREE_OBJECT(Six_arcs);
		FREE_int(transporter);


		}



	FREE_int(Elt2);

	FREE_OBJECT(SC);


	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface done" << endl;
	}
}



// #############################################################################
// global functions:
// #############################################################################


long int gta_subspace_orbits_rank_point_func(int *v, void *data)
{
	group_theoretic_activity *G;
	poset_classification *gen;
	long int rk;

	G = (group_theoretic_activity *) data;
	gen = G->orbits_on_subspaces_PC;
	gen->Poset->VS->F->PG_element_rank_modified_lint(v, 1,
			gen->Poset->VS->dimension, rk);
	return rk;
}

void gta_subspace_orbits_unrank_point_func(int *v, long int rk, void *data)
{
	group_theoretic_activity *G;
	poset_classification *gen;

	G = (group_theoretic_activity *) data;
	gen = G->orbits_on_subspaces_PC;
	gen->Poset->VS->F->PG_element_unrank_modified(v, 1,
			gen->Poset->VS->dimension, rk);
}

void gta_subspace_orbits_early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	//verbose_level = 1;

	group_theoretic_activity *G;
	//poset_classification *gen;
	int f_v = (verbose_level >= 1);
	int i;

	G = (group_theoretic_activity *) data;

	//gen = G->orbits_on_subspaces_PC;

	if (f_v) {
		cout << "gta_subspace_orbits_early_test_func" << endl;
		cout << "testing " << nb_candidates << " candidates" << endl;
	}
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		S[len] = candidates[i];
		if (G->subspace_orbits_test_set(len + 1, S, verbose_level - 1)) {
			good_candidates[nb_good_candidates++] = candidates[i];
		}
	}
	if (f_v) {
		cout << "gta_subspace_orbits_early_test_func" << endl;
		cout << "Out of " << nb_candidates << " candidates, "
				<< nb_good_candidates << " survive" << endl;
	}
}








}}

