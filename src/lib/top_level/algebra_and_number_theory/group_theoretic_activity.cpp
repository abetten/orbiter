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
	A1 = NULL;
	A2 = NULL;

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


	A1 = LG->A_linear;
	A2 = LG->A2;

	if (f_v) {
		cout << "group_theoretic_activity::init group = " << A1->label << endl;
		cout << "group_theoretic_activity::init action = " << A2->label << endl;
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


	if (Descr->f_multiply) {
		multiply(verbose_level);
	}

	if (Descr->f_inverse) {
		inverse(verbose_level);
	}

	if (Descr->f_export_gap) {
		do_export_gap(verbose_level);
	}

	if (Descr->f_export_magma) {
		do_export_magma(verbose_level);
	}

	if (Descr->f_classes) {
		classes(verbose_level);
	}

	if (Descr->f_group_table) {
		create_group_table(verbose_level);
	}

	if (Descr->f_normalizer) {
		normalizer(verbose_level);
	}

	if (Descr->f_centralizer_of_element) {
		centralizer(Descr->element_label,
				Descr->element_description_text, verbose_level);
	}

	if (Descr->f_normalizer_of_cyclic_subgroup) {
		normalizer_of_cyclic_subgroup(Descr->element_label,
				Descr->element_description_text, verbose_level);
	}

	if (Descr->f_find_subgroup) {
		do_find_subgroups(Descr->find_subgroup_order, verbose_level);
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
	if (Descr->f_find_singer_cycle) {
		find_singer_cycle(verbose_level);
	}
	if (Descr->f_search_element_of_order) {
		search_element_of_order(Descr->search_element_order, verbose_level);
	}



	if (Descr->f_orbits_on_set_system_from_file) {
		orbits_on_set_system_from_file(verbose_level);
	}

	if (Descr->f_orbit_of_set_from_file) {
		orbits_on_set_from_file(verbose_level);
	}

	if (Descr->f_orbit_of) {
		orbit_of(verbose_level);
	}
	else if (Descr->f_orbits_on_subsets) {
		orbits_on_subsets(verbose_level);
	}

	// generic orbits on points or subspaces:

	else if (Descr->f_orbits_on_points) {
		orbits_on_points(verbose_level);
	}
	else if (Descr->f_orbits_on_subspaces) {
		orbits_on_subspaces(verbose_level);
	}

	// classification of:


	// linear codes:


	if (Descr->f_linear_codes) {
		do_linear_codes(Descr->linear_codes_minimum_distance,
				Descr->linear_codes_target_size, verbose_level);
	}





	// arcs:


	else if (Descr->f_classify_arcs) {
		if (!Descr->f_poset_classification_control) {
			cout << "For classifying arcs, please use -poset_classification_control <descr> -end" << endl;
			exit(1);
		}
		do_classify_arcs(Descr->Arc_generator_description,
				verbose_level);
	}



	// surfaces:


	else if (Descr->f_surface_classify) {
		do_surface_classify(verbose_level);
	}
	else if (Descr->f_surface_report) {
		do_surface_report(verbose_level);
	}
	else if (Descr->f_surface_identify_HCV) {
		do_surface_identify_HCV(verbose_level);
	}
	else if (Descr->f_surface_identify_F13) {
		do_surface_identify_F13(verbose_level);
	}
	else if (Descr->f_surface_isomorphism_testing) {
		do_surface_isomorphism_testing(
				Descr->surface_descr_isomorph1,
				Descr->surface_descr_isomorph2,
				verbose_level);
	}
	else if (Descr->f_surface_recognize) {
		do_surface_recognize(Descr->surface_descr, verbose_level);
	}

	else if (Descr->f_classify_surfaces_through_arcs_and_two_lines) {

		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		do_classify_surfaces_through_arcs_and_two_lines(
				Descr->Control_six_arcs,
				Descr->f_test_nb_Eckardt_points, Descr->nb_E,
				verbose_level);
	}

	else if (Descr->f_classify_surfaces_through_arcs_and_trihedral_pairs) {
		if (!Descr->f_trihedra1_control) {
			cout << "please use option -trihedra1_control <description> -end" << endl;
			exit(1);
		}
		if (!Descr->f_trihedra2_control) {
			cout << "please use option -trihedra2_control <description> -end" << endl;
			exit(1);
		}
		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		do_classify_surfaces_through_arcs_and_trihedral_pairs(
				Descr->Trihedra1_control, Descr->Trihedra2_control,
				Descr->Control_six_arcs,
				Descr->f_test_nb_Eckardt_points, Descr->nb_E,
				verbose_level);
	}
	else if (Descr->f_create_surface) {
		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		do_create_surface(Descr->surface_description, Descr->Control_six_arcs, verbose_level);
	}
	else if (Descr->f_six_arcs) {
		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		do_six_arcs(Descr->Control_six_arcs,
				Descr->f_filter_by_nb_Eckardt_points, Descr->nb_Eckardt_points,
				verbose_level);
	}

	// spreads:

	else if (Descr->f_spread_classify) {
		do_spread_classify(Descr->spread_classify_k, verbose_level);
	}


	// packings:

	else if (Descr->f_packing_with_assumed_symmetry) {
		if (!Descr->f_packing_classify) {
			cout << "packing with symmetry needs packing" << endl;
			exit(1);
		}
		packing_classify *P;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before do_packing_classify" << endl;
		}

		do_packing_classify(Descr->dimension_of_spread_elements,
				Descr->spread_selection_text,
				Descr->spread_tables_prefix,
				0, // starter_size
				P,
				verbose_level);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after do_packing_classify" << endl;
		}

		packing_was *PW;

		PW = NEW_OBJECT(packing_was);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before PW->init" << endl;
		}

		PW->init(Descr->packing_was_descr, P, verbose_level);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after PW->init" << endl;
		}

		packing_was_fixpoints *PWF;

		PWF = NEW_OBJECT(packing_was_fixpoints);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before PWF->init" << endl;
		}

		PWF->init(PW, verbose_level);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after PWF->init" << endl;
		}


		FREE_OBJECT(PWF);
		FREE_OBJECT(PW);
		FREE_OBJECT(P);

	}
	else if (Descr->f_packing_classify) {
		packing_classify *P;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before do_packing_classify" << endl;
		}

		do_packing_classify(Descr->dimension_of_spread_elements,
				Descr->spread_selection_text,
				Descr->spread_tables_prefix,
				0, // starter_size
				P,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after do_packing_classify" << endl;
		}

		FREE_OBJECT(P);
	}



	// tensors:

	else if (Descr->f_tensor_classify) {
		do_tensor_classify(Descr->tensor_classify_depth, verbose_level);
	}
	else if (Descr->f_tensor_permutations) {
		do_tensor_permutations(verbose_level);
	}


	else if (Descr->f_classify_ovoids) {
		do_classify_ovoids(Descr->Control, Descr->Ovoid_classify_description, verbose_level);
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

	A2->conjugacy_classes_and_normalizers(G,
			LG->label, LG->label_tex, verbose_level);

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

	A1->multiply_based_on_text(Descr->multiply_a,
			Descr->multiply_b, verbose_level);

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

	A1->inverse_based_on_text(Descr->inverse_a, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::inverse done" << endl;
	}
}

void group_theoretic_activity::do_export_gap(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_export_gap" << endl;
	}

	char fname[1000];
	file_io Fio;

	sprintf(fname, "%s_generators.gap", LG->label.c_str());
	{
		ofstream fp(fname);
		LG->Strong_gens->print_generators_gap(fp);
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "group_theoretic_activity::do_export_gap done" << endl;
	}
}

void group_theoretic_activity::do_export_magma(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_export_magma" << endl;
	}

	char fname[1000];
	file_io Fio;

	sprintf(fname, "%s_generators.magma", LG->label.c_str());
	{
		ofstream fp(fname);
		LG->Strong_gens->export_magma(LG->A_linear, fp);
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "group_theoretic_activity::do_export_magma done" << endl;
	}
}


void group_theoretic_activity::create_group_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::create_group_table" << endl;
	}
	sims *H;
	int goi;


	H = LG->Strong_gens->create_sims(verbose_level);

	goi = H->group_order_lint();

	if (f_v) {
			cout << "group order H = " << goi << endl;
	}


	char fname[1000];
	int *Table;
	long int n;
	file_io Fio;

	H->create_group_table(Table, n, verbose_level);

	if (n != goi) {
		cout << "group_theoretic_activity::create_group_table n != goi" << endl;
		exit(1);
	}

	snprintf(fname, 1000, "%s_group_table.csv", LG->label.c_str());

	cout << "The group table is:" << endl;
	int_matrix_print(Table, n, n, 2);

	Fio.int_matrix_write_csv(fname, Table, n, n);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	FREE_int(Table);
	FREE_OBJECT(H);

	if (f_v) {
		cout << "group_theoretic_activity::create_group_table done" << endl;
	}
}



void group_theoretic_activity::normalizer(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::normalizer" << endl;
	}
	string fname_magma_prefix;
	sims *G;
	sims *H;
	strong_generators *gens_N;
	longinteger_object N_order;


	fname_magma_prefix.assign(LG->label);
	fname_magma_prefix.append("_normalizer");

	G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	if (f_v) {
		cout << "group order G = " << G->group_order_lint() << endl;
			cout << "group order H = " << H->group_order_lint() << endl;
			cout << "before A->normalizer_using_MAGMA" << endl;
	}
	A2->normalizer_using_MAGMA(fname_magma_prefix,
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

#if 0
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
#endif
	if (f_v) {
		cout << "group_theoretic_activity::normalizer done" << endl;
	}
}

void group_theoretic_activity::centralizer(
		const char *element_label,
		const char *element_description_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::centralizer" << endl;
	}

	algebra_global_with_action Algebra;
	sims *S;

	S = LG->Strong_gens->create_sims(verbose_level);

	Algebra.centralizer_of_element(
			LG->A2, S,
			element_description_text,
			element_label, verbose_level);

	FREE_OBJECT(S);

	if (f_v) {
		cout << "group_theoretic_activity::centralizer done" << endl;
	}
}

void group_theoretic_activity::normalizer_of_cyclic_subgroup(
		const char *element_label,
		const char *element_description_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::normalizer_of_cyclic_subgroup" << endl;
	}

	algebra_global_with_action Algebra;
	sims *S;

	S = LG->Strong_gens->create_sims(verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::normalizer_of_cyclic_subgroup "
				"before Algebra.normalizer_of_cyclic_subgroup" << endl;
	}
	Algebra.normalizer_of_cyclic_subgroup(
			LG->A2, S,
			element_description_text,
			element_label, verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::normalizer_of_cyclic_subgroup "
				"after Algebra.normalizer_of_cyclic_subgroup" << endl;
	}

	FREE_OBJECT(S);

	if (f_v) {
		cout << "group_theoretic_activity::normalizer_of_cyclic_subgroup done" << endl;
	}
}


void group_theoretic_activity::do_find_subgroups(
		int order_of_subgroup,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_find_subgroups" << endl;
	}

	algebra_global_with_action Algebra;
	sims *S;

	int nb_subgroups;
	strong_generators *H_gens;
	strong_generators *N_gens;


	S = LG->Strong_gens->create_sims(verbose_level);

	Algebra.find_subgroups(
			LG->A2, S,
			order_of_subgroup,
			LG->A2->label,
			nb_subgroups,
			H_gens,
			N_gens,
			verbose_level);


	cout << "We found " << nb_subgroups << " subgroups" << endl;


	string fname;
	string title;
	const char *author = "Orbiter";
	const char *extras_for_preamble = "";

	file_io Fio;

	fname.assign(LG->A2->label);
	fname.append("_report.tex");


	char str[1000];

	sprintf(str, "Subgroups of order $%d$ in $", order_of_subgroup);
	title.assign(str);
	title.append(LG->A2->label_tex);
	title.append("$");


	{
		ofstream fp(fname);
		latex_interface L;
		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */, TRUE /* f_title */,
			title.c_str(), author,
			FALSE /*f_toc*/, FALSE /* f_landscape*/, FALSE /* f_12pt*/,
			TRUE /*f_enlarged_page*/, TRUE /* f_pagenumbers*/,
			extras_for_preamble);

		LG->A2->report_groups_and_normalizers(fp,
				nb_subgroups, H_gens, N_gens,
				verbose_level);

		L.foot(fp);
	}

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	FREE_OBJECT(S);
	FREE_OBJECTS(H_gens);
	FREE_OBJECTS(N_gens);


	if (f_v) {
		cout << "group_theoretic_activity::do_find_subgroups done" << endl;
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


	sprintf(fname, "%s_report.tex", LG->label.c_str());
	sprintf(title, "The group $%s$", LG->label_tex.c_str());

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
				Descr->f_classes,
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
	int i; //, cnt;

	Elt = NEW_int(A1->elt_size_in_int);
	H->group_order(go);


	//cnt = 0;
	for (i = 0; i < go.as_lint(); i++) {
		H->element_unrank_lint(i, Elt);

		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		A1->element_print(Elt, cout);
		cout << endl;
		A1->element_print_as_permutation(Elt, cout);
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

	Elt = NEW_int(A1->elt_size_in_int);
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

	sprintf(fname, "%s_elements.tex", LG->label.c_str());


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

			Elt1 = NEW_int(A1->elt_size_in_int);
			Elt2 = NEW_int(A1->elt_size_in_int);
			Elt3 = NEW_int(A1->elt_size_in_int);

			order_table = NEW_int(nb_elements * nb_elements);
			for (i = 0; i < nb_elements; i++) {

				H->element_unrank_lint(elements[i], Elt1);


				for (j = 0; j < nb_elements; j++) {

					H->element_unrank_lint(elements[j], Elt2);

					A1->element_mult(Elt1, Elt2, Elt3, 0);

					order_table[i * nb_elements + j] = A2->element_order(Elt3);

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

	Elt = NEW_int(A1->elt_size_in_int);
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
			A2->element_print(Elt, cout);
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

void group_theoretic_activity::find_singer_cycle(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::find_singer_cycle" << endl;
	}
	sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	longinteger_object go;
	int i, d, q, cnt, ord, order;
	number_theory_domain NT;

	if (!A1->is_matrix_group()) {
		cout << "group_theoretic_activity::find_singer_cycle needs matrix group" << endl;
		exit(1);
	}
	matrix_group *M;

	M = A1->get_matrix_group();
	q = M->GFq->q;
	d = A1->matrix_group_dimension();

	if (A1->is_projective()) {
		order = (NT.i_power_j(q, d) - 1) / (q - 1);
	}
	else {
		order = NT.i_power_j(q, d) - 1;
	}
	if (f_v) {
		cout << "group_theoretic_activity::find_singer_cycle looking for an "
				"element of order " << order << endl;
	}

	Elt = NEW_int(A1->elt_size_in_int);
	H->group_order(go);

	cnt = 0;
	for (i = 0; i < go.as_int(); i++) {
		H->element_unrank_lint(i, Elt);


		ord = A2->element_order(Elt);

	#if 0
		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
	#endif

		if (ord != order) {
			continue;
		}
		if (!M->has_shape_of_singer_cycle(Elt)) {
			continue;
		}
		cout << "Element " << setw(5) << i << " / "
					<< go.as_int() << " = " << cnt << ":" << endl;
		A2->element_print(Elt, cout);
		cout << endl;
		A2->element_print_as_permutation(Elt, cout);
		cout << endl;
		cnt++;
	}
	cout << "we found " << cnt << " group elements of order " << order << endl;

	FREE_int(Elt);
	if (f_v) {
		cout << "group_theoretic_activity::find_singer_cycle done" << endl;
	}
}

void group_theoretic_activity::search_element_of_order(int order, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::search_element_of_order" << endl;
	}
	sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	longinteger_object go;
	int i, cnt, ord;

	Elt = NEW_int(A1->elt_size_in_int);
	H->group_order(go);

	cnt = 0;
	for (i = 0; i < go.as_int(); i++) {
		H->element_unrank_lint(i, Elt);


		ord = A2->element_order(Elt);

	#if 0
		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
	#endif

		if (ord != order) {
			continue;
		}
		cout << "Element " << setw(5) << i << " / "
					<< go.as_int() << " = " << cnt << ":" << endl;
		A2->element_print(Elt, cout);
		cout << endl;
		A2->element_print_as_permutation(Elt, cout);
		cout << endl;
		cnt++;
	}
	cout << "we found " << cnt << " group elements of order " << order << endl;

	FREE_int(Elt);
	if (f_v) {
		cout << "group_theoretic_activity::search_element_of_order done" << endl;
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
	A_on_sets = A2->create_induced_action_on_sets(m /* nb_sets */,
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

	OS->init(A1, A2, the_set, set_sz,
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

	char fname[2000];
	snprintf(fname, 2000, "orbit_of_%s_under_%s_with_hash.csv", str, LG->label.c_str());
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
	sprintf(fname, "orbit_of_%s_under_%s.txt", str, LG->label.c_str());
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

	Sch->init(A2, verbose_level - 2);
	if (!A2->f_has_strong_generators) {
		cout << "action::all_point_orbits !f_has_strong_generators" << endl;
		exit(1);
		}
	Sch->init_generators(*LG->Strong_gens->gens /* *strong_generators */, verbose_level - 2);
	Sch->initialize_tables();
	Sch->compute_point_orbit(Descr->orbit_of_idx, verbose_level);


	cout << "computing orbit of point done." << endl;

	char fname_tree_mask[1000];

	sprintf(fname_tree_mask, "%s_orbit_of_point_%d.layered_graph",
			LG->label.c_str(), Descr->orbit_of_idx);

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
	SG_stab->print_generators(cout);
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

	sprintf(fname_tree_mask, "%s_%%d_shallow.layered_graph", A2->label.c_str());

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
	LG->Strong_gens->print_generators(cout);
	cout << "Strong generators in tex are:" << endl;
	LG->Strong_gens->print_generators_tex(cout);
	cout << "Strong generators as permutations are:" << endl;
	LG->Strong_gens->print_generators_as_permutations();




	//A->all_point_orbits(*Sch, verbose_level);
	A2->all_point_orbits_from_generators(*Sch,
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
		char fname[1000];
		file_io Fio;
		int *orbit_reps;
		int i;


		sprintf(fname, "%s_orbit_reps.csv", A2->label.c_str());

		orbit_reps = NEW_int(Sch->nb_orbits);


		for (i = 0; i < Sch->nb_orbits; i++) {
			orbit_reps[i] = Sch->orbit[Sch->orbit_first[i]];
		}


		Fio.int_vec_write_csv(orbit_reps, Sch->nb_orbits,
				fname, "OrbRep");

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}


	{
		char fname[1000];
		file_io Fio;
		int *orbit_reps;
		int i;


		sprintf(fname, "%s_orbit_length.csv", A2->label.c_str());

		orbit_reps = NEW_int(Sch->nb_orbits);


		for (i = 0; i < Sch->nb_orbits; i++) {
			orbit_reps[i] = Sch->orbit_len[i];
		}


		Fio.int_vec_write_csv(orbit_reps, Sch->nb_orbits,
				fname, "OrbLen");

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}



	cout << "before Sch->print_and_list_orbits." << endl;
	if (A2->degree < 1000) {
		Sch->print_and_list_orbits(cout);
	}
	else {
		cout << "The degree is too large." << endl;
	}

	char fname_orbits[1000];
	file_io Fio;

	sprintf(fname_orbits, "%s_orbits.tex", A2->label.c_str());


	Sch->latex(fname_orbits);
	cout << "Written file " << fname_orbits << " of size "
			<< Fio.file_size(fname_orbits) << endl;



	if (Descr->f_export_trees) {
		char fname_tree_mask[1000];

		sprintf(fname_tree_mask, "%s_%%d.layered_graph", A2->label.c_str());

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
		char fname_schreier_tree_mask[1000];

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

		sprintf(fname_schreier_tree_mask, "%s_%%d_shallow.layered_graph", A2->label.c_str());

		shallow_tree->export_tree_as_layered_graph(0 /* orbit_no */,
				fname_schreier_tree_mask,
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


	Poset->init_subset_lattice(A1, A2,
			LG->Strong_gens,
			verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subsets "
				"before Poset->orbits_on_k_sets_compute" << endl;
	}
	PC = Poset->orbits_on_k_sets_compute(
			Control,
			Descr->orbits_on_subsets_size,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subsets "
				"after Poset->orbits_on_k_sets_compute" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subsets "
				"before orbits_on_poset_post_processing" << endl;
	}
	orbits_on_poset_post_processing(
			PC, Descr->orbits_on_subsets_size,
			verbose_level);


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

	Control->f_depth = TRUE;
	Control->depth = Descr->orbits_on_subspaces_depth;
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subspaces "
				"Control->max_depth=" << Control->depth << endl;
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


#if 0
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
#endif

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
				"LG->label=" << LG->label << endl;
	}

	Control->problem_label.assign(LG->label);
	Control->f_problem_label = TRUE;
	//sprintf(orbits_on_subspaces_PC->fname_base, "%s", LG->prefix);

	orbits_on_subspaces_PC->initialize_and_allocate_root_node(
			Control, orbits_on_subspaces_Poset,
			Control->depth, verbose_level);



	int schreier_depth = Control->depth;
	int f_use_invariant_subset_if_available = FALSE;
	int f_debug = FALSE;
	int nb_orbits;

	os_interface Os;
	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subspaces "
				"calling generator_main" << endl;
		cout << "A=";
		orbits_on_subspaces_PC->get_A()->print_info();
		cout << "A2=";
		orbits_on_subspaces_PC->get_A2()->print_info();
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
	nb_orbits = orbits_on_subspaces_PC->nb_orbits_at_level(Control->depth);
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subspaces we found "
				<< nb_orbits << " orbits at depth "
				<< Control->depth << endl;
	}

	orbits_on_poset_post_processing(
			orbits_on_subspaces_PC, Control->depth, verbose_level);


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


#if 0

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
#endif

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_poset_post_processing "
				"after PC->list_all_orbits_at_level" << endl;
	}

#if 0
	if (Descr->f_report) {

		if (f_v) {
			cout << "group_theoretic_activity::orbits_on_poset_post_processing doing a report" << endl;
		}
		{
			char fname_report[1000];
			sprintf(fname_report, "%s_poset.tex", LG->label.c_str());
			latex_interface L;
			file_io Fio;

			{
				ofstream ost(fname_report);
				L.head_easy(ost);

				if (f_v) {
					cout << "group_theoretic_activity::orbits_on_poset_post_processing "
							"before A1->report" << endl;
				}

				A1 /*LG->A_linear*/->report(ost,
						FALSE /* f_sims */,
						NULL, //A1/*LG->A_linear*/->Sims,
						TRUE /* f_strong_gens */,
						LG->Strong_gens,
						verbose_level - 1);

				if (f_v) {
					cout << "group_theoretic_activity::orbits_on_poset_post_processing "
							"after LG->A_linear->report" << endl;
				}

				L.foot(ost);
			}
			cout << "Written file " << fname_report << " of size "
					<< Fio.file_size(fname_report) << endl;
		}
	}
#endif

#if 0
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
#endif


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
				A2->degree /* nb_points */,
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
				A2->degree /* nb_points */,
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





void group_theoretic_activity::do_classify_arcs(
		arc_generator_description *Arc_generator_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_arcs" << endl;
	}

	if (!Arc_generator_description->f_q) {
		cout << "please use -q <q>" << endl;
		exit(1);
	}

	if (!Arc_generator_description->f_n) {
		cout << "please use -n <n>" << endl;
		exit(1);
	}

	if (Arc_generator_description->q != LG->F->q) {
		cout << "group_theoretic_activity::do_classify_arcs the order of the fields don't match" << endl;
		exit(1);
	}
	Arc_generator_description->F = LG->F;
	Arc_generator_description->LG = LG;
	Arc_generator_description->Control = Descr->Control;

	if (Arc_generator_description->n != LG->A2->matrix_group_dimension()) {
		cout << "group_theoretic_activity::do_classify_arcs the dimensions don't match" << endl;
		exit(1);
	}

	{
	arc_generator *Gen;

	Gen = NEW_OBJECT(arc_generator);



	if (f_v) {
		cout << "group_theoretic_activity::do_classify_arcs before Gen->init" << endl;
	}
	Gen->init_from_description(Arc_generator_description,
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
		cout << "group_theoretic_activity::do_surface_classify "
				"before Algebra.classify_surfaces, control=" << endl;
		Control->print();
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify "
				"after Algebra.classify_surfaces" << endl;
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify "
				"before SCW->generate_source_code" << endl;
	}
	SCW->generate_source_code(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify "
				"after SCW->generate_source_code" << endl;
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
		cout << "group_theoretic_activity::do_surface_report "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report "
				"after Algebra.classify_surfaces" << endl;
	}

	int f_with_stabilizers = TRUE;

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report "
				"before SCW->create_report" << endl;
	}
	SCW->create_report(f_with_stabilizers, verbose_level - 1);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report "
				"after SCW->create_report" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "interface_algebra::do_surface_report done" << endl;
	}
}

void group_theoretic_activity::do_surface_identify_HCV(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_HCV" << endl;
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
		cout << "group_theoretic_activity::do_surface_identify_HCV "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_HCV "
				"after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_HCV "
				"before SCW->identify_HCV_and_print_table" << endl;
	}
	SCW->identify_HCV_and_print_table(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_HCV "
				"after SCW->identify_HCV_and_print_table" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_HCV done" << endl;
	}
}

void group_theoretic_activity::do_surface_identify_F13(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_F13" << endl;
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
		cout << "group_theoretic_activity::do_surface_identify_F13 "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_F13 "
				"after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_F13 "
				"before SCW->identify_HCV_and_print_table" << endl;
	}
	SCW->identify_F13_and_print_table(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_F13 "
				"after SCW->identify_HCV_and_print_table" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_F13 done" << endl;
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
		cout << "group_theoretic_activity::do_surface_isomorphism_testing "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing "
				"after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing "
				"before SCW->test_isomorphism" << endl;
	}
	SCW->test_isomorphism(
			surface_descr_isomorph1,
			surface_descr_isomorph2,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing "
				"after SCW->test_isomorphism" << endl;
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
		cout << "group_theoretic_activity::do_surface_recognize "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize "
				"after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize "
				"before SCW->recognition" << endl;
	}
	SCW->recognition(
			surface_descr,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize "
				"after SCW->recognition" << endl;
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


void group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines(
		poset_classification_control *Control_six_arcs,
		int f_test_nb_Eckardt_points, int nb_E,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines" << endl;
	}

	surface_with_action *Surf_A;
	surface_domain *Surf;
	number_theory_domain NT;



	Surf = NEW_OBJECT(surface_domain);
	Surf_A = NEW_OBJECT(surface_with_action);


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
				"before Surf->init" << endl;
	}
	Surf->init(F, 0 /*verbose_level*/);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
				"after Surf->init" << endl;
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
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, LG, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
				"after Surf_A->init" << endl;
	}


	surfaces_arc_lifting *SAL;

	SAL = NEW_OBJECT(surfaces_arc_lifting);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
				"before SAL->init" << endl;
	}
	SAL->init(
		LG->F, LG /* LG4 */,
		Surf_A,
		Control_six_arcs,
		f_test_nb_Eckardt_points, nb_E,
		verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
				"after SAL->init" << endl;
	}

	if (Descr->f_report) {
		if (f_v) {
			cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
					"before SAL->report" << endl;
		}
		SAL->report(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
					"after SAL->report" << endl;
		}

	}
	FREE_OBJECT(SAL);


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines done" << endl;
	}

}

void group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs(
		poset_classification_control *Control1,
		poset_classification_control *Control2,
		poset_classification_control *Control_six_arcs,
		int f_test_nb_Eckardt_points, int nb_E,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	surface_with_action *Surf_A;
	surface_domain *Surf;
	number_theory_domain NT;



	Surf = NEW_OBJECT(surface_domain);
	Surf_A = NEW_OBJECT(surface_with_action);


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf->init" << endl;
	}
	Surf->init(F, verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf->init" << endl;
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
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, LG, verbose_level - 1);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_A->init" << endl;
	}



	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_A->Classify_trihedral_pairs->classify" << endl;
	}
	Surf_A->Classify_trihedral_pairs->classify(Control1, Control2, verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_A->Classify_trihedral_pairs->classify" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_arc->classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	surface_classify_using_arc *Surf_arc;

	Surf_arc = NEW_OBJECT(surface_classify_using_arc);


	Surf_arc->classify_surfaces_through_arcs_and_trihedral_pairs(
			Control_six_arcs,
			Surf_A,
			f_test_nb_Eckardt_points, nb_E,
			verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_arc->classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_arc->report" << endl;
	}


	Surf_arc->report(verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_arc->report" << endl;
	}

	FREE_OBJECT(Surf_arc);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs done" << endl;
	}

}

void group_theoretic_activity::do_create_surface(
		surface_create_description *Surface_Descr,
		poset_classification_control *Control_six_arcs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface" << endl;
	}

	int q;
	int i;
	finite_field *F;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	number_theory_domain NT;
	sorting Sorting;
	file_io Fio;

	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface before Surface_Descr->get_q" << endl;
	}
	q = Surface_Descr->get_q();
	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface q = " << q << endl;
	}


	F = LG->F;
	if (F->q != q) {
		cout << "F->q != q" << endl;
		exit(1);
	}


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

	//SC->F->init_symbol_for_print("\\omega");

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

	cout << "$$" << endl;
	int_vec_print(cout, SC->coeffs, 20);
	cout << endl;
	cout << "$$" << endl;


	if (SC->f_has_group) {
		for (i = 0; i < SC->Sg->gens->len; i++) {
			cout << "group_theoretic_activity::do_create_surface "
					"Testing generator " << i << " / "
					<< SC->Sg->gens->len << endl;
			A->element_invert(SC->Sg->gens->ith(i),
					Elt2, 0 /*verbose_level*/);



			matrix_group *M;

			M = A->G.matrix_grp;
			M->substitute_surface_equation(Elt2,
					SC->coeffs, coeffs_out, SC->Surf,
					verbose_level - 1);


			if (int_vec_compare(SC->coeffs, coeffs_out, 20)) {
				cout << "group_theoretic_activity::do_create_surface error, "
						"the transformation does not preserve "
						"the equation of the surface" << endl;
				exit(1);
			}
			cout << "group_theoretic_activity::do_create_surface "
					"Generator " << i << " / " << SC->Sg->gens->len
					<< " is good" << endl;
		}
	}
	else {
		cout << "group_theoretic_activity::do_create_surface "
				"We do not have information about "
				"the automorphism group" << endl;
	}


	cout << "group_theoretic_activity::do_create_surface We have created "
			"the surface " << SC->label_txt << ":" << endl;
	cout << "$$" << endl;
	SC->Surf->print_equation_tex(cout, SC->coeffs);
	cout << endl;
	cout << "$$" << endl;

	if (SC->f_has_group) {
		cout << "group_theoretic_activity::do_create_surface "
				"The stabilizer is generated by:" << endl;
		SC->Sg->print_generators_tex(cout);

		if (SC->f_has_nice_gens) {
			cout << "group_theoretic_activity::do_create_surface "
					"The stabilizer is generated by the following nice generators:" << endl;
			SC->nice_gens->print_tex(cout);

		}
	}

	if (SC->f_has_lines) {
		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface "
					"The lines are:" << endl;
			SC->Surf->Gr->print_set_tex(cout, SC->Lines, 27);
		}


		surface_object *SO;

		SO = NEW_OBJECT(surface_object);
		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface before SO->init" << endl;
			}
		SO->init(SC->Surf, SC->Lines, SC->coeffs,
				FALSE /*f_find_double_six_and_rearrange_lines */, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface after SO->init" << endl;
			}

		char fname_points[2000];

		snprintf(fname_points, 2000, "surface_%s_points.txt", SC->label_txt.c_str());
		Fio.write_set_to_file(fname_points,
				SO->Pts, SO->nb_pts, 0 /*verbose_level*/);
		cout << "group_theoretic_activity::do_create_surface "
				"Written file " << fname_points << " of size "
				<< Fio.file_size(fname_points) << endl;
	}
	else {
		cout << "group_theoretic_activity::do_create_surface "
				"The surface " << SC->label_txt
				<< " does not come with lines" << endl;
	}




	if (SC->f_has_group) {

		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface creating "
					"surface_object_with_action object" << endl;
		}

		surface_object_with_action *SoA;

		SoA = NEW_OBJECT(surface_object_with_action);

		if (SC->f_has_lines) {
			if (f_v) {
				cout << "group_theoretic_activity::do_create_surface creating "
						"surface using the known lines (which are "
						"arranged with respect to a double six):" << endl;
			}
			SoA->init(SC->Surf_A,
				SC->Lines,
				SC->coeffs,
				SC->Sg,
				FALSE /*f_find_double_six_and_rearrange_lines*/,
				SC->f_has_nice_gens, SC->nice_gens,
				verbose_level);
			}
		else {
			if (f_v) {
				cout << "group_theoretic_activity::do_create_surface "
						"creating surface from equation only "
						"(no lines):" << endl;
			}
			SoA->init_equation(SC->Surf_A,
				SC->coeffs,
				SC->Sg,
				verbose_level);
			}
		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface "
					"The surface has been created." << endl;
		}



		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface "
					"Classifying non-conical six-arcs." << endl;
		}

		six_arcs_not_on_a_conic *Six_arcs;
		arc_generator_description *Six_arc_descr;

		int *transporter;

		Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);

		Six_arc_descr = NEW_OBJECT(arc_generator_description);
		Six_arc_descr->F = F;
		Six_arc_descr->f_q = TRUE;
		Six_arc_descr->q = F->q;
		Six_arc_descr->f_n = TRUE;
		Six_arc_descr->n = 3;
		Six_arc_descr->f_target_size = TRUE;
		Six_arc_descr->target_size = 6;
		Six_arc_descr->Control = Control_six_arcs;



		// classify six arcs not on a conic:

		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface "
					"Setting up the group of the plane:" << endl;
		}

		action *A;

		A = NEW_OBJECT(action);


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


		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface "
					"before Six_arcs->init:" << endl;
		}


		Six_arcs->init(
				Six_arc_descr,
				A,
				SC->Surf->P2,
				FALSE, 0, NULL,
				verbose_level);

		transporter = NEW_int(Six_arcs->Gen->A->elt_size_in_int);


		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface "
					"before SoA->investigate_surface_and_write_report:" << endl;
		}

		SoA->investigate_surface_and_write_report(
				A,
				SC,
				Six_arcs,
				Descr->f_surface_clebsch,
				Descr->f_surface_codes,
				Descr->f_surface_quartic,
				verbose_level);

		FREE_OBJECT(SoA);
		FREE_OBJECT(Six_arcs);
		FREE_OBJECT(Six_arc_descr);
		FREE_int(transporter);


		}
	else {
		cout << "We don't have the group of the surface" << endl;
	}



	FREE_int(Elt2);

	FREE_OBJECT(SC);


	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface done" << endl;
	}
}

void group_theoretic_activity::do_six_arcs(
		poset_classification_control *Control_six_arcs,
		int f_filter_by_nb_Eckardt_points, int nb_Eckardt_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs" << endl;
	}

	finite_field *F;

	F = LG->F;


	surface_domain *Surf;

	if (f_v) {
			cout << "group_theoretic_activity::do_six_arcs before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, 0/*verbose_level - 1*/);
	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs after Surf->init" << endl;
	}



	six_arcs_not_on_a_conic *Six_arcs;
	arc_generator_description *Six_arc_descr;

	int *transporter;

	Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);

	Six_arc_descr = NEW_OBJECT(arc_generator_description);
	Six_arc_descr->F = F;
	Six_arc_descr->f_q = TRUE;
	Six_arc_descr->q = F->q;
	Six_arc_descr->f_n = TRUE;
	Six_arc_descr->n = 3;
	Six_arc_descr->f_target_size = TRUE;
	Six_arc_descr->target_size = 6;
	Six_arc_descr->Control = Control_six_arcs;



	// classify six arcs not on a conic:

	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs "
				"Setting up the group of the plane:" << endl;
	}



	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs "
				"before Six_arcs->init:" << endl;
	}


	Six_arcs->init(
			Six_arc_descr,
			LG->A_linear,
			Surf->P2,
			FALSE, 0, NULL,
			verbose_level);

	transporter = NEW_int(Six_arcs->Gen->A->elt_size_in_int);

	int nb_orbits;
	int level = 6;

	nb_orbits = Six_arcs->Gen->gen->nb_orbits_at_level(level);

	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs "
				"We found " << nb_orbits << " isomorphism types "
				"of 6-arcs" << endl;
	}



	long int Arc6[6];
	int h, a, b, c, d;
	int v1[3];
	int v2[3];


	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs "
				"testing the arcs" << endl;
	}

	longinteger_object ago;
	int *Abcd;
	int *Nb_E;
	int *Ago;

	Abcd = NEW_int(nb_orbits * 4);
	Nb_E = NEW_int(nb_orbits);
	Ago = NEW_int(nb_orbits);

	for (h = 0; h < nb_orbits; h++) {

		if (f_v) {
			cout << "group_theoretic_activity::do_six_arcs "
					"testing arc " << h << " / " << nb_orbits << endl;
		}


		Six_arcs->Gen->gen->get_set_by_level(level, h, Arc6);

		Six_arcs->Gen->gen->get_stabilizer_order(level, h, ago);

		if (Arc6[0] != 0) {
			cout << "Arc6[0] != 0" << endl;
			exit(1);
		}
		if (Arc6[1] != 1) {
			cout << "Arc6[1] != 1" << endl;
			exit(1);
		}
		if (Arc6[2] != 2) {
			cout << "Arc6[2] != 2" << endl;
			exit(1);
		}
		if (Arc6[3] != 3) {
			cout << "Arc6[3] != 3" << endl;
			exit(1);
		}
		Surf->P2->unrank_point(v1, Arc6[4]);
		Surf->P2->unrank_point(v2, Arc6[5]);
		if (v1[2] != 1) {
			cout << "v1[2] != 1" << endl;
			exit(1);
		}
		if (v2[2] != 1) {
			cout << "v2[2] != 1" << endl;
			exit(1);
		}
		a = v1[0];
		b = v1[1];
		c = v2[0];
		d = v2[1];

		Abcd[h * 4 + 0] = a;
		Abcd[h * 4 + 1] = b;
		Abcd[h * 4 + 2] = c;
		Abcd[h * 4 + 3] = d;

		eckardt_point_info *E;

		E = Surf->P2->compute_eckardt_point_info(Surf, Arc6, 0/*verbose_level*/);


		Nb_E[h] = E->nb_E;
		Ago[h] = ago.as_int();

		//cout << h << " : " << a << "," << b << "," << c << "," << d << " : " << E->nb_E << " : " << ago << endl;

		FREE_OBJECT(E);
	}

#if 0
	cout << "Summary of " << nb_orbits << " arcs:" << endl;
	for (h = 0; h < nb_orbits; h++) {
		a = Abcd[h * 4 + 0];
		b = Abcd[h * 4 + 1];
		c = Abcd[h * 4 + 2];
		d = Abcd[h * 4 + 3];

		cout << h << " : " << a << "," << b << "," << c << "," << d << " : " << Nb_E[h] << " : " << Ago[h] << endl;
	}
#endif

	tally C;

	C.init(Nb_E, nb_orbits, FALSE, 0);

	cout << "nb_E distribution: ";
	C.print_naked_tex(cout, FALSE);
	cout << endl;


	if (f_filter_by_nb_Eckardt_points) {
		cout << "Nonconical six-arcs associated with surfaces with " << nb_Eckardt_points << " Eckardt points in PG(2," << F->q << "):" << endl;

	}
	else {
		cout << "Nonconical six-arcs associated in PG(2," << F->q << "):" << endl;

	}
	int nb_E;
	int cnt = 0;

	cout << "$$" << endl;
	cout << "\\begin{array}{|r|c|r|}" << endl;
	cout << "\\hline" << endl;
	cout << "\\mbox{Orbit} & a,b,c,d & \\mbox{Ago} \\\\" << endl;
	cout << "\\hline" << endl;

	for (h = 0; h < nb_orbits; h++) {
		a = Abcd[h * 4 + 0];
		b = Abcd[h * 4 + 1];
		c = Abcd[h * 4 + 2];
		d = Abcd[h * 4 + 3];

		nb_E = Nb_E[h];

		if (f_filter_by_nb_Eckardt_points) {
			if (nb_E != nb_Eckardt_points) {
				continue;
			}
		}
		cout << h << " & " << a << "," << b << "," << c << "," << d << " & ";
		//<< Nb_E[h] << " & "
		cout << Ago[h] << "\\\\" << endl;

		cnt++;
	}
	cout << "\\hline" << endl;
	cout << "\\end{array}" << endl;
	cout << "$$" << endl;
	cout << "There are " << cnt << " such arcs.\\\\" << endl;


	FREE_int(Abcd);
	FREE_int(Nb_E);
	FREE_int(Ago);


	FREE_OBJECT(Six_arcs);
	FREE_OBJECT(Six_arc_descr);
	FREE_int(transporter);


	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs done" << endl;
	}

}

void group_theoretic_activity::do_spread_classify(int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_spread_classify" << endl;
	}

	poset_classification_control *Control;

	if (!Descr->f_poset_classification_control) {
		cout << "please use -poset_classification_control <descr> -end" << endl;
		exit(1);
	}
	else {
		Control = Descr->Control;
	}


	spread_classify *SC;

	SC = NEW_OBJECT(spread_classify);

	if (f_v) {
		cout << "group_theoretic_activity::do_spread_classify before SC->init" << endl;
	}

	SC->init(
			LG,
			k,
			Control,
			verbose_level - 1);


	if (f_v) {
		cout << "group_theoretic_activity::do_spread_classify after SC->init" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_spread_classify before SC->compute" << endl;
	}

	SC->compute(verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::do_spread_classify after SC->compute" << endl;
	}


	FREE_OBJECT(SC);

	if (f_v) {
		cout << "group_theoretic_activity::do_spread_classify done" << endl;
	}
}

void group_theoretic_activity::do_packing_classify(int dimension_of_spread_elements,
		std::string &spread_selection_text,
		std::string &spread_tables_prefix,
		int starter_size,
		packing_classify *&P,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_packing_classify" << endl;
	}

	poset_classification_control *Control;

	if (!Descr->f_poset_classification_control) {
		cout << "please use -poset_classification_control <descr> -end" << endl;
		exit(1);
	}
	else {
		Control = Descr->Control;
	}


	algebra_global_with_action Algebra;

	Algebra.packing_init(
			Control, LG,
			dimension_of_spread_elements,
			TRUE /* f_select_spread */, spread_selection_text,
			spread_tables_prefix,
			P,
			verbose_level);



	if (f_v) {
		cout << "group_theoretic_activity::do_packing_classify done" << endl;
	}
}

void group_theoretic_activity::do_tensor_classify(int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_tensor_classify" << endl;
	}


	if (!Descr->f_poset_classification_control) {
		cout << "please use option -poset_classification_control descr -end" << endl;
		exit(1);
	}



	tensor_classify *T;

	T = NEW_OBJECT(tensor_classify);

	if (f_v) {
		cout << "group_theoretic_activity::do_tensor_classify before T->init" << endl;
	}
	T->init(F, LG, verbose_level - 1);
	if (f_v) {
		cout << "group_theoretic_activity::do_tensor_classify after T->init" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_tensor_classify before classify_poset" << endl;
	}
	T->classify_poset(depth,
			Descr->Control,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_tensor_classify after classify_poset" << endl;
	}



	FREE_OBJECT(T);

	if (f_v) {
		cout << "group_theoretic_activity::do_tensor_classify done" << endl;
	}
}


void group_theoretic_activity::do_tensor_permutations(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_tensor_permutations" << endl;
	}

#if 0
	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}
#endif



	tensor_classify *T;

	T = NEW_OBJECT(tensor_classify);

	T->init(F, LG, verbose_level - 1);


	FREE_OBJECT(T);

	if (f_v) {
		cout << "group_theoretic_activity::do_tensor_permutations done" << endl;
	}
}

#if 0

tensor_classify *T;

T = NEW_OBJECT(tensor_classify);

T->init(nb_factors, d, q, depth,
		0/*verbose_level*/);

if (f_tensor_ranks) {
	cout << "before T->W->compute_tensor_ranks" << endl;
	T->W->compute_tensor_ranks(verbose_level);
	cout << "after T->W->compute_tensor_ranks" << endl;
}

{
	int *result = NULL;

	cout << "time check: ";
	Os.time_check(cout, t0);
	cout << endl;

	cout << "tensor_classify::init " << __FILE__ << ":" << __LINE__ << endl;

	int nb_gens, degree;

	if (f_permutations) {
		cout << "before T->W->compute_permutations_and_write_to_file" << endl;
		T->W->compute_permutations_and_write_to_file(T->SG, T->A, result,
				nb_gens, degree, nb_factors,
				verbose_level);
		cout << "after T->W->compute_permutations_and_write_to_file" << endl;
	}
	//wreath_product_orbits_CUDA(W, SG, A,
	// result, nb_gens, degree, nb_factors, verbose_level);

	if (f_orbits) {
		cout << "before T->W->orbits_using_files_and_union_find" << endl;
		T->W->orbits_using_files_and_union_find(T->SG, T->A, result, nb_gens, degree, nb_factors,
				verbose_level);
		cout << "after T->W->orbits_using_files_and_union_find" << endl;
	}
	if (f_orbits_restricted) {
		cout << "before T->W->orbits_restricted" << endl;
		T->W->orbits_restricted(T->SG, T->A, result,
				nb_gens, degree, nb_factors, orbits_restricted_fname,
				verbose_level);
		cout << "after T->W->orbits_restricted" << endl;
	}
	if (f_orbits_restricted_compute) {
		cout << "before T->W->orbits_restricted_compute" << endl;
		T->W->orbits_restricted_compute(T->SG, T->A, result,
				nb_gens, degree, nb_factors, orbits_restricted_fname,
				verbose_level);
		cout << "after T->W->orbits_restricted_compute" << endl;
	}
}

#endif


void group_theoretic_activity::do_linear_codes(int minimum_distance,
		int target_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_linear_codes" << endl;
	}

	if (!Descr->f_poset_classification_control) {
		cout << "Please use option -poset_classification_control <descr> -end" << endl;
		exit(1);
	}

	algebra_global_with_action Algebra;

	if (f_v) {
		cout << "group_theoretic_activity::do_linear_codes before "
				"Algebra.linear_codes_with_bounded_minimum_distance" << endl;
	}

	Algebra.linear_codes_with_bounded_minimum_distance(
			Descr->Control, LG,
			minimum_distance, target_size, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::do_linear_codes after "
				"Algebra.linear_codes_with_bounded_minimum_distance" << endl;
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_linear_codes done" << endl;
	}
}

void group_theoretic_activity::do_classify_ovoids(
		poset_classification_control *Control,
		ovoid_classify_description *Ovoid_classify_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_ovoids" << endl;
	}

	ovoid_classify *Ovoid_classify;


	Ovoid_classify = NEW_OBJECT(ovoid_classify);

	Ovoid_classify_description->Control = Control;

	Ovoid_classify->init(Ovoid_classify_description,
			LG,
			verbose_level);

	FREE_OBJECT(Ovoid_classify);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_ovoids done" << endl;
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
	gen->get_VS()->F->PG_element_rank_modified_lint(v, 1,
			gen->get_VS()->dimension, rk);
	return rk;
}

void gta_subspace_orbits_unrank_point_func(int *v, long int rk, void *data)
{
	group_theoretic_activity *G;
	poset_classification *gen;

	G = (group_theoretic_activity *) data;
	gen = G->orbits_on_subspaces_PC;
	gen->get_VS()->F->PG_element_unrank_modified(v, 1,
			gen->get_VS()->dimension, rk);
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

