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

	if (Descr->f_centralizer_of_element) {
		centralizer(Descr->element_label,
				Descr->element_description_text, verbose_level);
	}

	if (Descr->f_normalizer_of_cyclic_subgroup) {
		normalizer_of_cyclic_subgroup(Descr->element_label,
				Descr->element_description_text, verbose_level);
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
	} // if (f_orbit_of)
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
		do_linear_codes(Descr->linear_codes_minimum_distance, Descr->linear_codes_target_size, verbose_level);
	}





	// arcs:


	else if (Descr->f_classify_arcs) {
		if (!Descr->f_poset_classification_control) {
			cout << "please use -poset_classification_control <descr> -end" << endl;
			exit(1);
		}
		do_classify_arcs(Descr->classify_arcs_target_size, Descr->classify_arcs_d, FALSE,
				Descr->Control,
				verbose_level);
	}

	else if (Descr->f_classify_nonconical_arcs) {
		if (!Descr->f_poset_classification_control) {
			cout << "please use -poset_classification_control <descr> -end" << endl;
			exit(1);
		}
		do_classify_arcs(Descr->classify_arcs_target_size, Descr->classify_arcs_d, TRUE,
				Descr->Control,
				verbose_level);
	}


	// surfaces:


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
		do_surface_isomorphism_testing(
				Descr->surface_descr_isomorph1,
				Descr->surface_descr_isomorph2,
				verbose_level);
	}
	else if (Descr->f_surface_recognize) {
		do_surface_recognize(Descr->surface_descr, verbose_level);
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
			cout << "please use option -Control_six_arcs <description> -end" << endl;
			exit(1);
		}
		do_classify_surfaces_through_arcs_and_trihedral_pairs(
				Descr->Trihedra1_control, Descr->Trihedra2_control,
				Descr->Control_six_arcs,
				verbose_level);
	}
	else if (Descr->f_create_surface) {
		if (!Descr->f_control_six_arcs) {
			cout << "please use option -Control_six_arcs <description> -end" << endl;
			exit(1);
		}
		do_create_surface(Descr->surface_descr, Descr->Control_six_arcs, verbose_level);
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

		do_packing_classify(Descr->dimension_of_spread_elements,
				Descr->spread_selection_text,
				Descr->spread_tables_prefix,
				0, // starter_size
				P,
				verbose_level);

		packing_was *PWAS;

		PWAS = NEW_OBJECT(packing_was);

		PWAS->init(Descr->packing_was_descr,
				P, verbose_level);


		FREE_OBJECT(PWAS);
		FREE_OBJECT(P);

	}
	else if (Descr->f_packing_classify) {
		packing_classify *P;

		do_packing_classify(Descr->dimension_of_spread_elements,
				Descr->spread_selection_text,
				Descr->spread_tables_prefix,
				0, // starter_size
				P,
				verbose_level);
		FREE_OBJECT(P);
	}



	// tensors:

	else if (Descr->f_tensor_classify) {
		do_tensor_classify(Descr->tensor_classify_depth, verbose_level);
	}
	else if (Descr->f_tensor_permutations) {
		do_tensor_permutations(verbose_level);
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
			LG->label.c_str(), LG->label_tex.c_str(), verbose_level);

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


	sprintf(fname_magma_prefix, "%s_normalizer", LG->label.c_str());

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
	int i, cnt;

	Elt = NEW_int(A1->elt_size_in_int);
	H->group_order(go);


	cnt = 0;
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
		cout << "group_theoretic_activity::find_singer_cycle looking for an element of order " << order << endl;
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

	char fname[1000];
	sprintf(fname, "orbit_of_%s_under_%s_with_hash.csv", str, LG->label.c_str());
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



#if 0
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

			sprintf(fname, "%s_orbit_%d_transition.csv", A->label, idx);
			Fio.int_matrix_write_csv(fname, M, m, m);

			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
			}
			{
			char fname[1000];
			file_io Fio;

			sprintf(fname, "%s_orbit_%d_elts.csv", A->label, idx);
			Fio.int_vec_write_csv(idx, Sch->orbit_len[idx],
					fname, "Elt");

			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
			}
		}
		FREE_int(M);
		FREE_int(O);

	}
#endif

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

	Control->f_max_depth = TRUE;
	Control->max_depth = Descr->orbits_on_subspaces_depth;
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subspaces "
				"Control->max_depth=" << Control->max_depth << endl;
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
				"LG->label=" << LG->label << endl;
	}

	Control->problem_label.assign(LG->label);
	Control->f_problem_label = TRUE;
	//sprintf(orbits_on_subspaces_PC->fname_base, "%s", LG->prefix);

	orbits_on_subspaces_PC->initialize_and_allocate_root_node(
			Control, orbits_on_subspaces_Poset,
			Control->max_depth, verbose_level);



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
	nb_orbits = orbits_on_subspaces_PC->nb_orbits_at_level(Control->max_depth);
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subspaces we found "
				<< nb_orbits << " orbits at depth "
				<< Control->max_depth << endl;
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

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_poset_post_processing after PC->list_all_orbits_at_level" << endl;
	}

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
					cout << "group_theoretic_activity::orbits_on_poset_post_processing before A1->report" << endl;
				}

				A1 /*LG->A_linear*/->report(ost,
						FALSE /* f_sims */,
						NULL, //A1/*LG->A_linear*/->Sims,
						TRUE /* f_strong_gens */,
						LG->Strong_gens,
						verbose_level - 1);

				if (f_v) {
					cout << "group_theoretic_activity::orbits_on_poset_post_processing after LG->A_linear->report" << endl;
				}

				L.foot(ost);
			}
			cout << "Written file " << fname_report << " of size " << Fio.file_size(fname_report) << endl;
		}
	}

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





void group_theoretic_activity::do_classify_arcs(int arc_size, int arc_d, int f_not_on_conic,
		poset_classification_control *Control,
		int verbose_level)
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
	Gen->target_size = arc_size;
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_arcs target_size=" << arc_size << endl;
		cout << "group_theoretic_activity::do_classify_arcs arc_d=" << arc_d << endl;
		cout << "group_theoretic_activity::do_classify_arcs f_not_on_conic=" << f_not_on_conic << endl;
	}


	//const char *input_prefix = "";
	//const char *base_fname = "";
	//int starter_size = 0;

	Gen->F = LG->F;
	Gen->q = LG->F->q;
	if (arc_d > 0) {
		Gen->f_d = TRUE;
		Gen->d = arc_d;
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
		//input_prefix = Descr->ECA->input_prefix;
		//base_fname = Descr->ECA->base_fname;
		//starter_size = Gen->ECA->starter_size;
	}
	else {
		cout << "no exact cover" << endl;
		Descr->ECA = NULL;
		//input_prefix = "";
		//base_fname = "";
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
			arc_size,
			f_not_on_conic,
			Control,
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
				"before Algebra.classify_surfaces" << endl;
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
		cout << "group_theoretic_activity::do_surface_identify_Sa "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Sa "
				"after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Sa "
				"before SCW->identify_Sa_and_print_table" << endl;
	}
	SCW->identify_Sa_and_print_table(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Sa "
				"after SCW->identify_Sa_and_print_table" << endl;
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


void group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs(
		poset_classification_control *Control1,
		poset_classification_control *Control2,
		poset_classification_control *Control_six_arcs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	algebra_global_with_action Algebra;
	surface_with_action *Surf_A;
	surface_domain *Surf;
	number_theory_domain NT;



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
	Surf_A->Classify_trihedral_pairs->classify(Control1, Control2, 0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf_A->Classify_trihedral_pairs->classify" << endl;
	}

	if (f_v) {
		cout << "before A.classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}
	Algebra.classify_surfaces_through_arcs_and_trihedral_pairs(
			this,
			Surf_A,
			Control_six_arcs,
			verbose_level);
	if (f_v) {
		cout << "after A.classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

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
		Six_arcs->init(this,
				SC->F,
				A,
			SC->Surf->P2,
			Control_six_arcs,
			verbose_level);
		transporter = NEW_int(Six_arcs->Gen->A->elt_size_in_int);

		algebra_global_with_action Algebra;


		Algebra.investigate_surface_and_write_report(
				A,
				SC,
				Six_arcs,
				SoA,
				Descr->f_surface_clebsch,
				Descr->f_surface_codes,
				Descr->f_surface_quartic,
				verbose_level);

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
		const char *spread_selection_text,
		const char *spread_tables_prefix,
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

	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
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
			Control,
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

	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}



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

