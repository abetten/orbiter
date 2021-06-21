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

	if (Descr->f_raise_to_the_power) {
		raise_to_the_power(verbose_level);
	}

	if (Descr->f_export_gap) {
		do_export_gap(verbose_level);
	}

	if (Descr->f_export_magma) {
		do_export_magma(verbose_level);
	}

	if (Descr->f_classes_based_on_normal_form) {
		classes_based_on_normal_form(verbose_level);
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

	if (Descr->f_conjugacy_class_of_element) {
		do_conjugacy_class_of_element(Descr->element_label,
				Descr->element_description_text, verbose_level);
	}
	if (Descr->f_orbits_on_group_elements_under_conjugation) {
		do_orbits_on_group_elements_under_conjugation(
				Descr->orbits_on_group_elements_under_conjugation_fname,
				Descr->orbits_on_group_elements_under_conjugation_transporter_fname,
				verbose_level);
	}



	if (Descr->f_normalizer_of_cyclic_subgroup) {
		normalizer_of_cyclic_subgroup(Descr->element_label,
				Descr->element_description_text, verbose_level);
	}

	if (Descr->f_find_subgroup) {
		do_find_subgroups(Descr->find_subgroup_order, verbose_level);
	}





	if (Descr->f_report) {

		if (!Orbiter->f_draw_options) {
			cout << "for a report of the group, please use -draw_options" << endl;
			exit(1);
		}

		LG->create_latex_report(
				Orbiter->draw_options,
				Descr->f_sylow, Descr->f_group_table, Descr->f_classes,
				verbose_level);

	}

	if (Descr->f_print_elements) {
		print_elements(verbose_level);
	}

	if (Descr->f_print_elements_tex) {
		print_elements_tex(verbose_level);
	}

	if (Descr->f_find_singer_cycle) {
		find_singer_cycle(verbose_level);
	}
	if (Descr->f_search_element_of_order) {
		search_element_of_order(Descr->search_element_order, verbose_level);
	}

	if (Descr->f_find_standard_generators) {
		find_standard_generators(Descr->find_standard_generators_order_a,
				Descr->find_standard_generators_order_b,
				Descr->find_standard_generators_order_ab,
				verbose_level);

	}

	if (Descr->f_element_rank) {
		element_rank(Descr->element_rank_data, verbose_level);
	}
	if (Descr->f_element_unrank) {
		element_unrank(Descr->element_unrank_data, verbose_level);
	}
	if (Descr->f_conjugacy_class_of) {
		conjugacy_class_of(Descr->conjugacy_class_of_data, verbose_level);
	}
	if (Descr->f_isomorphism_Klein_quadric) {
		isomorphism_Klein_quadric(Descr->isomorphism_Klein_quadric_fname, verbose_level);
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
	else if (Descr->f_reverse_isomorphism_exterior_square) {
		do_reverse_isomorphism_exterior_square(verbose_level);
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


	else if (Descr->f_classify_cubic_curves) {
		do_classify_cubic_curves(
				Descr->Arc_generator_description,
				verbose_level);
	}

	else if (Descr->f_orbits_on_polynomials) {

		algebra_global_with_action Algebra;


		if (Descr->f_orbits_on_polynomials_draw_tree &&
				!Orbiter->f_draw_options) {
			cout << "please use -draw_options ... -end" << endl;
			exit(1);
		}

		Algebra.do_orbits_on_polynomials(
				LG,
				Descr->orbits_on_polynomials_degree,
				Descr->f_recognize_orbits_on_polynomials,
				Descr->recognize_orbits_on_polynomials_text,
				Descr->f_orbits_on_polynomials_draw_tree,
				Descr->orbits_on_polynomials_draw_tree_idx,
				Orbiter->draw_options,
				verbose_level);
	}

	else if (Descr->f_representation_on_polynomials) {

		algebra_global_with_action Algebra;

		Algebra.representation_on_polynomials(
				LG,
				Descr->representation_on_polynomials_degree,
				verbose_level);
	}

	else if (Descr->f_Andre_Bruck_Bose_construction) {
		do_Andre_Bruck_Bose_construction(
					Descr->Andre_Bruck_Bose_construction_spread_no,
					FALSE /* f_Fano */, FALSE /* f_arcs */, FALSE /* f_depth */, 0 /* depth */,
					Descr->Andre_Bruck_Bose_construction_label,
					verbose_level);
	}


	else if (Descr->f_BLT_starter) {
		do_BLT_starter(
					LG,
					Descr->BLT_starter_size,
					verbose_level);
	}


	if (f_v) {
		cout << "group_theoretic_activity::perform_activity done" << endl;
	}
}


void group_theoretic_activity::classes_based_on_normal_form(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::classes_based_on_normal_form" << endl;
	}
	sims *G;
	algebra_global_with_action Algebra;

	G = LG->Strong_gens->create_sims(verbose_level);


	Algebra.conjugacy_classes_based_on_normal_forms(LG->A_linear,
			G,
			LG->label,
			LG->label_tex,
			verbose_level);

	FREE_OBJECT(G);
	if (f_v) {
		cout << "group_theoretic_activity::classes_based_on_normal_form done" << endl;
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

	if (f_v) {
		cout << "group_theoretic_activity::classes "
				"before A2->conjugacy_classes_and_normalizers" << endl;
	}
	A2->conjugacy_classes_and_normalizers(G,
			LG->label, LG->label_tex, verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::classes "
				"after A2->conjugacy_classes_and_normalizers" << endl;
	}

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

void group_theoretic_activity::raise_to_the_power(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::raise_to_the_power" << endl;
	}

	A1->raise_to_the_power_based_on_text(Descr->raise_to_the_power_a_text,
			Descr->raise_to_the_power_exponent_text, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::raise_to_the_power done" << endl;
	}
}

void group_theoretic_activity::do_export_gap(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_export_gap" << endl;
	}

	string fname;
	file_io Fio;

	fname.assign(LG->label);
	fname.append("_generators.gap");
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

	string fname;
	file_io Fio;

	fname.assign(LG->label);
	fname.append("_generators.magma");
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


	string fname;
	int *Table;
	long int n;
	file_io Fio;

	H->create_group_table(Table, n, verbose_level);

	if (n != goi) {
		cout << "group_theoretic_activity::create_group_table n != goi" << endl;
		exit(1);
	}

	fname.assign(LG->label);
	fname.append("_group_table.csv");

	cout << "The group table is:" << endl;
	Orbiter->Int_vec.matrix_print(Table, n, n, 2);

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
	longinteger_object G_order;
	longinteger_object H_order;


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

	LG->initial_strong_gens->group_order(G_order);
	LG->Strong_gens->group_order(H_order);
	gens_N->group_order(N_order);
	if (f_v) {
		cout << "group order G = " << G->group_order_lint() << endl;
		cout << "group order H = " << H_order << endl;
	}
	if (f_v) {
		cout << "group order N = " << N_order << endl;
		cout << "Strong generators for the normalizer of H are:" << endl;
		gens_N->print_generators_tex(cout);
		cout << "Strong generators for the normalizer of H as permutations are:" << endl;
		gens_N->print_generators_as_permutations();
	}



	string fname;

	fname.assign(fname_magma_prefix);
	fname.append(".tex");


	{
		char title[1000];
		char author[1000];

		snprintf(title, 1000, "Normalizer of subgroup %s", LG->label_tex.c_str());
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);

			ost << "\\noindent The group $" << LG->label_tex << "$ "
					"of order " << H_order << " is:\\\\" << endl;
			LG->Strong_gens->print_generators_tex(ost);

			ost << "\\bigskip" << endl;

			ost << "Inside the group of order " << G_order << ", "
					"the normalizer has order " << N_order << ":\\\\" << endl;
			if (f_v) {
				cout << "group_theoretic_activity::normalizer before report" << endl;
			}
			gens_N->print_generators_tex(ost);

			if (f_v) {
				cout << "group_theoretic_activity::normalizer after report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	if (f_v) {
		cout << "group_theoretic_activity::normalizer done" << endl;
	}
}

void group_theoretic_activity::centralizer(
		std::string &element_label,
		std::string &element_description_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::centralizer" << endl;
	}

	algebra_global_with_action Algebra;
	sims *S;

	S = LG->Strong_gens->create_sims(verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::centralizer "
				"before Algebra.centralizer_of_element" << endl;
	}
	Algebra.centralizer_of_element(
			LG->A2, S,
			element_description_text,
			element_label, verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::centralizer "
				"after Algebra.centralizer_of_element" << endl;
	}

	FREE_OBJECT(S);

	if (f_v) {
		cout << "group_theoretic_activity::centralizer done" << endl;
	}
}

void group_theoretic_activity::normalizer_of_cyclic_subgroup(
		std::string &element_label,
		std::string &element_description_text,
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

	string fname;

	fname.assign(LG->label);
	fname.append("_elements.tex");


	{
		ofstream fp(fname);
		latex_interface L;
		L.head_easy(fp);

		H->print_all_group_elements_tex(fp);
		//H->print_all_group_elements_with_permutations_tex(fp);

		//Schreier.print_and_list_orbits_tex(fp);

		if (Descr->f_order_of_products) {
			int *elements;
			int nb_elements;
			int *order_table;
			int i;

			Orbiter->Int_vec.scan(Descr->order_of_products_elements, elements, nb_elements);

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

#if 0
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
#endif

void group_theoretic_activity::find_singer_cycle(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::find_singer_cycle" << endl;
	}
	algebra_global_with_action Algebra;

	Algebra.find_singer_cycle(LG,
			A1, A2,
			verbose_level);
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
	algebra_global_with_action Algebra;

	Algebra.search_element_of_order(LG,
			A1, A2,
			order, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::search_element_of_order done" << endl;
	}
}

void group_theoretic_activity::find_standard_generators(int order_a,
		int order_b,
		int order_ab,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::find_standard_generators" << endl;
	}
	algebra_global_with_action Algebra;

	Algebra.find_standard_generators(LG,
			A1, A2,
			order_a, order_b, order_ab, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::find_standard_generators done" << endl;
	}

}
void group_theoretic_activity::element_rank(std::string &elt_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::element_rank" << endl;
	}
	algebra_global_with_action Algebra;

	Algebra.element_rank(LG,
			A1,
			elt_data, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::element_rank done" << endl;
	}
}

void group_theoretic_activity::element_unrank(std::string &rank_string, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::element_unrank" << endl;
	}
	algebra_global_with_action Algebra;

	Algebra.element_unrank(LG,
			A1,
			rank_string, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::element_unrank done" << endl;
	}
}

void group_theoretic_activity::conjugacy_class_of(std::string &elt_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::conjugacy_class_of" << endl;
	}
	sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;

	Elt = NEW_int(A1->elt_size_in_int);

	longinteger_object a, b;

#if 1
	cout << "creating element " << elt_data << endl;

	A1->make_element_from_string(Elt, elt_data, 0);

	H->element_rank(a, Elt);

	a.assign_to(b);


#else


	a.create_from_base_10_string(rank_string.c_str(), 0 /*verbose_level*/);

	cout << "Creating element of rank " << a << endl;

	a.assign_to(b);

	H->element_unrank(a, Elt);

#endif

	cout << "Element :" << endl;
	A1->element_print(Elt, cout);
	cout << endl;


	action *A_conj;

	A_conj = A1->create_induced_action_by_conjugation(
		H /*Base_group*/, FALSE /* f_ownership */,
		verbose_level);


	cout << "created action A_conj of degree " << A_conj->degree << endl;

#if 0
	schreier *Sch;

	Sch = LG->Strong_gens->orbit_of_one_point_schreier(
			A_conj, b.as_lint(), verbose_level);

	cout << "Orbits on itself by conjugation:\\\\" << endl;
	Sch->print_orbit_reps(cout);


	FREE_OBJECT(Sch);
#else


	orbit_of_sets Orb;
	long int set[1];
	file_io Fio;

	set[0] = b.as_lint();

	Orb.init(A1, A_conj,
			set, 1 /* sz */, LG->Strong_gens->gens, verbose_level);
	cout << "Found an orbit of size " << Orb.used_length << endl;

	std::vector<long int> Orbit;

	cout << "before Orb.get_orbit_of_points" << endl;
	Orb.get_orbit_of_points(Orbit, verbose_level);
	cout << "Found an orbit of size " << Orbit.size() << endl;

	int *M;
	int i, j;

	M = NEW_int(Orbit.size() * A1->make_element_size);
	for (i = 0; i < Orbit.size(); i++) {
		H->element_unrank_lint(Orbit[i], Elt);
		for (j = 0; j < A1->make_element_size; j++) {
			M[i * A1->make_element_size + j] = Elt[j];
		}
		//M[i] = Orbit[i];
	}
	string fname;

	fname.assign(LG->label);
	fname.append("_class_of_");
	fname.append(elt_data);
	fname.append(".csv");

	Fio.int_matrix_write_csv(fname, M, Orbit.size(), A1->make_element_size);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	FREE_int(M);

#endif

	FREE_OBJECT(A_conj);
	FREE_OBJECT(H);




	FREE_int(Elt);
	if (f_v) {
		cout << "group_theoretic_activity::conjugacy_class_of done" << endl;
	}
}


void group_theoretic_activity::do_reverse_isomorphism_exterior_square(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 5);

	if (f_v) {
		cout << "group_theoretic_activity::do_reverse_isomorphism_exterior_square" << endl;
	}


	if (LG->f_has_nice_gens) {
		if (f_v) {
			cout << "group_theoretic_activity::do_reverse_isomorphism_exterior_square nice generators are:" << endl;
			LG->nice_gens->print(cout);
		}
		LG->nice_gens->reverse_isomorphism_exterior_square(verbose_level);
	}
	else {
		if (f_v) {
			cout << "group_theoretic_activity::do_reverse_isomorphism_exterior_square strong generators are:" << endl;
			LG->Strong_gens->print_generators_in_latex_individually(cout);
		}
		LG->Strong_gens->reverse_isomorphism_exterior_square(verbose_level);
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_reverse_isomorphism_exterior_square done" << endl;
	}
}

void group_theoretic_activity::isomorphism_Klein_quadric(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);

	if (f_v) {
		cout << "group_theoretic_activity::isomorphism_Klein_quadric" << endl;
	}
	sims *H;
	file_io Fio;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;

	Elt = NEW_int(A1->elt_size_in_int);


	cout << "Reading file " << fname << " of size " << Fio.file_size(fname) << endl;

	int *M;
	int m, n;
	Fio.int_matrix_read_csv(fname, M, m, n, verbose_level);

	cout << "Read a set of size " << m << endl;

	if (n != A1->make_element_size) {
		cout << "n != A1->make_element_size" << endl;
		exit(1);
	}





	int i, j, c;
	int Basis1[] = {
#if 1
			1,0,0,0,0,0,
			0,1,0,0,0,0,
			0,0,1,0,0,0,
			0,0,0,1,0,0,
			0,0,0,0,1,0,
			0,0,0,0,0,1,
#else
			1,0,0,0,0,0,
			0,0,0,0,0,1,
			0,1,0,0,0,0,
			0,0,0,0,-1,0,
			0,0,1,0,0,0,
			0,0,0,1,0,0,
#endif
	};
	//int Basis1b[36];
	int Basis2[36];
	int An2[37];
	int v[6];
	int w[6];
	int C[36];
	int D[36];
	int E[36];
	int B[] = {
			1,0,0,0,0,0,
			0,0,0,2,0,0,
			1,3,0,0,0,0,
			0,0,0,1,3,0,
			1,0,2,0,0,0,
			0,0,0,2,0,4,
	};
	int Target[] = {
			1,0,0,0,0,0,
			3,2,2,0,0,0,
			1,4,2,0,0,0,
			0,0,0,1,0,0,
			0,0,0,3,2,2,
			0,0,0,1,4,2,
	};
	int Bv[36];
	sorting Sorting;

#if 0
	for (i = 0; i < 6; i++) {
		if (Basis1[i] == -1) {
			Basis1b[i] = F->negate(1);
		}
		else {
			Basis1b[i] = Basis1[i];
		}
	}
#endif

	for (i = 0; i < 6; i++) {
		F->klein_to_wedge(Basis1 + i * 6, Basis2 + i * 6);
	}

	F->matrix_inverse(B, Bv, 6, 0 /* verbose_level */);


	for (i = 0; i < m; i++) {

		A1->make_element(Elt, M + i * A1->make_element_size, 0);

		if ((i % 10000) == 0) {
			cout << i << " / " << m << endl;
		}

		if (f_vv) {
			cout << "Element " << i << " / " << m << endl;
			A1->element_print(Elt, cout);
			cout << endl;
		}

		F->exterior_square(Elt, An2, 4, 0 /*verbose_level*/);

		if (f_vv) {
			cout << "Exterior square:" << endl;
			Orbiter->Int_vec.matrix_print(An2, 6, 6);
			cout << endl;
		}

		for (j = 0; j < 6; j++) {
			F->mult_vector_from_the_left(Basis2 + j * 6, An2, v, 6, 6);
					// v[m], A[m][n], vA[n]
			F->wedge_to_klein(v /* W */, w /*K*/);
			Orbiter->Int_vec.copy(w, C + j * 6, 6);
		}

		int Gram[] = {
				0,1,0,0,0,0,
				1,0,0,0,0,0,
				0,0,0,1,0,0,
				0,0,1,0,0,0,
				0,0,0,0,0,1,
				0,0,0,0,1,0,
		};
		int new_Gram[36];

		F->transform_form_matrix(C, Gram,
				new_Gram, 6, 0 /* verbose_level*/);

		if (f_vv) {
			cout << "Transformed Gram matrix:" << endl;
			Orbiter->Int_vec.matrix_print(new_Gram, 6, 6);
			cout << endl;
		}


		if (f_vv) {
			cout << "orthogonal matrix :" << endl;
			Orbiter->Int_vec.matrix_print(C, 6, 6);
			cout << endl;
		}

		F->mult_matrix_matrix(Bv, C, D, 6, 6, 6, 0 /*verbose_level */);
		F->mult_matrix_matrix(D, B, E, 6, 6, 6, 0 /*verbose_level */);

		F->PG_element_normalize_from_front(E, 1, 36);

		if (f_vv) {
			cout << "orthogonal matrix in the special form:" << endl;
			Orbiter->Int_vec.matrix_print(E, 6, 6);
			cout << endl;
		}

		int special_Gram[] = {
				0,0,0,3,4,1,
				0,0,0,4,1,3,
				0,0,0,1,3,4,
				3,4,1,0,0,0,
				4,1,3,0,0,0,
				1,3,4,0,0,0,
		};
		int new_special_Gram[36];

		F->transform_form_matrix(E, special_Gram,
				new_special_Gram, 6, 0 /* verbose_level*/);

		if (f_vv) {
			cout << "Transformed special Gram matrix:" << endl;
			Orbiter->Int_vec.matrix_print(new_special_Gram, 6, 6);
			cout << endl;
		}



		c = Sorting.integer_vec_compare(E, Target, 36);
		if (c == 0) {
			cout << "We found it! i=" << i << " element = ";
			Orbiter->Int_vec.print(cout, M + i * A1->make_element_size, A1->make_element_size);
			cout << endl;

			cout << "Element :" << endl;
			A1->element_print(Elt, cout);
			cout << endl;

			cout << "exterior square :" << endl;
			Orbiter->Int_vec.matrix_print(An2, 6, 6);
			cout << endl;

			cout << "orthogonal matrix :" << endl;
			Orbiter->Int_vec.matrix_print(C, 6, 6);
			cout << endl;

			cout << "orthogonal matrix in the special form:" << endl;
			Orbiter->Int_vec.matrix_print(E, 6, 6);
			cout << endl;

			//exit(1);
		}


	}

	FREE_int(Elt);
	FREE_int(M);
	FREE_OBJECT(H);

	if (f_v) {
		cout << "group_theoretic_activity::isomorphism_Klein_quadric" << endl;
	}
}



void group_theoretic_activity::orbits_on_set_system_from_file(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_set_system_from_file" << endl;
	}
	if (f_v) {
		cout << "computing orbits on set system from file "
			<< Descr->orbits_on_set_system_from_file_fname << ":" << endl;
	}
	file_io Fio;
	int *M;
	int m, n;
	long int *Table;
	int i, j;

	Fio.int_matrix_read_csv(Descr->orbits_on_set_system_from_file_fname, M,
			m, n, verbose_level);
	if (f_v) {
		cout << "read a matrix of size " << m << " x " << n << endl;
	}


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

	if (f_v) {
		cout << "creating action on sets:" << endl;
	}
	A_on_sets = A2->create_induced_action_on_sets(m /* nb_sets */,
			set_size, Table,
			verbose_level);

	schreier *Sch;
	int first, a;

	if (f_v) {
		cout << "computing orbits on sets:" << endl;
	}
	A_on_sets->compute_orbits_on_points(Sch,
			LG->Strong_gens->gens, verbose_level);

	if (f_v) {
		cout << "The orbit lengths are:" << endl;
		Sch->print_orbit_lengths(cout);
	}

	if (f_v) {
		cout << "The orbits are:" << endl;
		//Sch->print_and_list_orbits(cout);
		for (i = 0; i < Sch->nb_orbits; i++) {
			cout << " Orbit " << i << " / " << Sch->nb_orbits
					<< " : " << Sch->orbit_first[i] << " : " << Sch->orbit_len[i];
			cout << " : ";

			first = Sch->orbit_first[i];
			a = Sch->orbit[first + 0];
			cout << a << " : ";
			Orbiter->Lint_vec.print(cout, Table + a * set_size, set_size);
			cout << endl;
			//Sch->print_and_list_orbit_tex(i, ost);
		}
	}
	string fname;
	string_tools ST;

	fname.assign(Descr->orbits_on_set_system_from_file_fname);
	ST.chop_off_extension(fname);
	fname.append("_orbit_reps.txt");

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

	if (f_v) {
		cout << "computing orbit of set from file "
			<< Descr->orbit_of_set_from_file_fname << ":" << endl;
	}
	file_io Fio;
	long int *the_set;
	int set_sz;

	Fio.read_set_from_file(Descr->orbit_of_set_from_file_fname,
			the_set, set_sz, 0 /*verbose_level*/);
	if (f_v) {
		cout << "read a set of size " << set_sz << endl;
	}


	string label_set;
	string_tools ST;

	label_set.assign(Descr->orbit_of_set_from_file_fname);
	ST.chop_off_extension(label_set);

	algebra_global_with_action Algebra;
	long int *Table;
	int size;

	Algebra.orbits_on_set_from_file(
			the_set, set_sz,
			A1, A2,
			LG->Strong_gens->gens,
			label_set,
			LG->label,
			Table, size,
			verbose_level);

	FREE_lint(Table);

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

	string fname_tree_mask;
	char str[1000];

	fname_tree_mask.assign(LG->label);
	sprintf(str, "_orbit_of_point_%d.layered_graph", Descr->orbit_of_idx);
	fname_tree_mask.append(str);


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

	fname_tree_mask.assign(A2->label);
	fname_tree_mask.append("_%d_shallow.layered_graph");

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
	algebra_global_with_action Algebra;

	orbits_on_something *Orb;
	int f_load_save = TRUE;
	string prefix;

	prefix.assign(LG->label);

	Algebra.orbits_on_points(
			LG,
			A2,
			f_load_save,
			prefix,
			Orb,
			verbose_level);

	if (Descr->f_report) {

		Orb->create_latex_report(verbose_level);

	}

	if (Descr->f_export_trees) {
		string fname_tree_mask;
		int orbit_idx;

		fname_tree_mask.assign(A2->label);
		fname_tree_mask.append("_%d.layered_graph");

		for (orbit_idx = 0; orbit_idx < Orb->Sch->nb_orbits; orbit_idx++) {
			cout << "orbit " << orbit_idx << " / " <<  Orb->Sch->nb_orbits
					<< " before Sch->export_tree_as_layered_graph" << endl;
			Orb->Sch->export_tree_as_layered_graph(0 /* orbit_no */,
					fname_tree_mask,
					verbose_level - 1);
		}
	}



	FREE_OBJECT(Orb);

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
	poset_with_group_action *Poset;

	Poset = NEW_OBJECT(poset_with_group_action);


	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		cout << "please use option -poset_classification_control" << endl;
		exit(1);
		//Control = NEW_OBJECT(poset_classification_control);
	}
	if (f_v) {
		cout << "group_theoretic_activity::orbits_on_subsets control=" << endl;
		Control->print();
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
		cout << "please use option -poset_classification_control" << endl;
		exit(1);
		//Control = NEW_OBJECT(poset_classification_control);
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
	orbits_on_subspaces_Poset = NEW_OBJECT(poset_with_group_action);



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

	orbits_on_subspaces_Poset = NEW_OBJECT(poset_with_group_action);
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
			Orbiter->Lint_vec.matrix_print(Orbit, orbit_length, d);

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
			Orbiter->Lint_vec.matrix_print(Orbit1, orbit_length1, d);

			PC->get_whole_orbit(
					depth, 1 /* orbit_idx*/,
					Orbit2, orbit_length2, verbose_level);
			cout << "depth " << d << " orbit " << 1
					<< " / " << nb_orbits << " has length "
					<< orbit_length2 << ":" << endl;
			Orbiter->Lint_vec.matrix_print(Orbit2, orbit_length2, d);

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


	FREE_OBJECT(Gen);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_arcs done" << endl;
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

	tensor_classify *T;

	T = NEW_OBJECT(tensor_classify);

	T->init(F, LG, verbose_level - 1);


	FREE_OBJECT(T);

	if (f_v) {
		cout << "group_theoretic_activity::do_tensor_permutations done" << endl;
	}
}


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


void group_theoretic_activity::do_classify_cubic_curves(
		arc_generator_description *Arc_generator_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves" << endl;
	}



	finite_field *F;

	F = LG->A2->matrix_group_finite_field();

	Arc_generator_description->F = F;
	if (Arc_generator_description->q != F->q) {
		cout << "Arc_generator_description->q != F->q" << endl;
		exit(1);
	}



	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves q = " << F->q << endl;
	}

	cubic_curve *CC;

	CC = NEW_OBJECT(cubic_curve);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves before CC->init" << endl;
	}
	CC->init(F, verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves after CC->init" << endl;
	}


	cubic_curve_with_action *CCA;

	CCA = NEW_OBJECT(cubic_curve_with_action);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves before CCA->init" << endl;
	}
	CCA->init(CC, LG->A2, verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves after CCA->init" << endl;
	}


	classify_cubic_curves *CCC;

	CCC = NEW_OBJECT(classify_cubic_curves);


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves before CCC->init" << endl;
	}
	CCC->init(
			this,
			CCA,
			Arc_generator_description,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves after CCC->init" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves before CCC->compute_starter" << endl;
	}
	CCC->compute_starter(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves after CCC->compute_starter" << endl;
	}

#if 0
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves before CCC->test_orbits" << endl;
	}
	CCC->test_orbits(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves after CCC->test_orbits" << endl;
	}
#endif

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves before CCC->do_classify" << endl;
	}
	CCC->do_classify(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves after CCC->do_classify" << endl;
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves creating cheat sheet" << endl;
	}
	char fname[1000];
	char title[1000];
	char author[1000];
	snprintf(title, 1000, "Cubic Curves in PG$(2,%d)$", F->q);
	strcpy(author, "");
	snprintf(fname, 1000, "Cubic_curves_q%d.tex", F->q);

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

		fp << "\\subsection*{" << title << "}" << endl;

		if (f_v) {
			cout << "group_theoretic_activity::do_classify_cubic_curves before CCC->report" << endl;
		}
		CCC->report(fp, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::do_classify_cubic_curves after CCC->report" << endl;
		}

		L.foot(fp);
	}

	file_io Fio;

	cout << "Written file " << fname << " of size "
		<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves writing cheat sheet on "
				"cubic curves done" << endl;
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_cubic_curves done" << endl;
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
		Orbiter->Lint_vec.print(cout, S, len);
		cout << endl;
		cout << "LG->n=" << LG->n << endl;
	}
	n = LG->n;
	F = LG->F;

	F->PG_elements_unrank_lint(
			orbits_on_subspaces_M, len, n, S);

	if (f_vv) {
		cout << "coordinate matrix:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
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


void group_theoretic_activity::do_conjugacy_class_of_element(
		std::string &elt_label, std::string &elt_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_conjugacy_class_of_element" << endl;
	}


	int *data, sz;

	Orbiter->Int_vec.scan(elt_text, data, sz);

	if (f_v) {
		cout << "computing conjugacy class of ";
		Orbiter->Int_vec.print(cout, data, sz);
		cout << endl;
	}


	int *Elt;

	Elt = NEW_int(A1->elt_size_in_int);
	A1->make_element(Elt, data, 0 /* verbose_level */);

	if (!A1->f_has_sims) {
		if (f_v) {
			cout << "group_theoretic_activity::do_conjugacy_class_of_element "
				"Group does not have a sims object" << endl;
		}
		//exit(1);

		{
			sims *S;

			S = LG->Strong_gens->create_sims(verbose_level);

			if (f_v) {
				cout << "group_theoretic_activity::do_conjugacy_class_of_element before init_sims" << endl;
			}
			A1->init_sims_only(S, 0/*verbose_level - 1*/);
			if (f_v) {
				cout << "group_theoretic_activity::do_conjugacy_class_of_element after init_sims" << endl;
			}
		}

	}
	sims *S;

	S = A1->Sims;

	long int the_set[1];
	int set_size = 1;

	the_set[0] = S->element_rank_lint(Elt);

	if (f_v) {
		cout << "computing conjugacy class of " << endl;
		A1->element_print_latex(Elt, cout);
		cout << "which is the set ";
		Orbiter->Lint_vec.print(cout, the_set, set_size);
		cout << endl;
	}


	action A_conj;
	if (f_v) {
		cout << "group_theoretic_activity::do_conjugacy_class_of_element "
				"before A_conj.induced_action_by_conjugation" << endl;
	}
	A_conj.induced_action_by_conjugation(S, S,
			FALSE /* f_ownership */, FALSE /* f_basis */,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_conjugacy_class_of_element "
				"created action by conjugation" << endl;
	}



	//schreier Classes;
	//Classes.init(&A_conj, verbose_level - 2);
	//Classes.init_generators(*A1->Strong_gens->gens, verbose_level - 2);
	//cout << "Computing orbits:" << endl;
	//Classes.compute_all_point_orbits(1 /*verbose_level - 1*/);
	//cout << "found " << Classes.nb_orbits << " conjugacy classes" << endl;




	algebra_global_with_action Algebra;

	long int *Table;
	int orbit_length;

	Algebra.orbits_on_set_from_file(
			the_set, set_size,
			A1, &A_conj,
			LG->Strong_gens->gens,
			elt_label,
			LG->label,
			Table,
			orbit_length,
			verbose_level);


	// write as txt file:

	string fname;
	file_io Fio;

	fname.assign(elt_label);
	fname.append("_orbit_under_");
	fname.append(LG->label);
	fname.append("_elements_coded.csv");

	if (f_v) {
		cout << "Writing table to file " << fname << endl;
	}
	{
		ofstream ost(fname);
		int i;

		// header line:
		ost << "ROW";
		for (int j = 0; j < A1->make_element_size; j++) {
			ost << ",C" << j;
		}
		ost << endl;

		for (i = 0; i < orbit_length; i++) {

			ost << i;
			S->element_unrank_lint(Table[i], Elt);

			for (int j = 0; j < A1->make_element_size; j++) {
				ost << "," << Elt[j];
			}
			ost << endl;
		}
		ost << "END" << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	FREE_int(Elt);
	FREE_int(data);
	FREE_lint(Table);

	if (f_v) {
		cout << "group_theoretic_activity::do_conjugacy_class_of_element done" << endl;
	}
}


void group_theoretic_activity::do_orbits_on_group_elements_under_conjugation(
		std::string &fname_group_elements_coded,
		std::string &fname_transporter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_orbits_on_group_elements_under_conjugation" << endl;
	}




	if (!A2->f_has_sims) {
		if (f_v) {
			cout << "group_theoretic_activity::do_orbits_on_group_elements_under_conjugation "
				"Group does not have a sims object" << endl;
		}
		//exit(1);

		{
			//sims *S;

			A2->create_sims(verbose_level);

#if 0
			if (f_v) {
				cout << "group_theoretic_activity::do_orbits_on_group_elements_under_conjugation before init_sims" << endl;
			}
			A2->init_sims_only(S, 0/*verbose_level - 1*/);
			if (f_v) {
				cout << "group_theoretic_activity::do_orbits_on_group_elements_under_conjugation after init_sims" << endl;
			}
#endif
		}

	}





	sims *S;

	S = A2->Sims;

	if (f_v) {
		cout << "the group has order " << S->group_order_lint() << endl;
	}
	int *Elt;

	Elt = NEW_int(A1->elt_size_in_int);

	if (f_v) {
		cout << "computing the element ranks:" << endl;
	}

	file_io Fio;
	long int *the_ranks;
	vector_ge *Transporter;
	int m, n;
	int i;

	{
		int *M;
		Fio.int_matrix_read_csv(fname_group_elements_coded,
				M, m, n, 0 /*verbose_level*/);
		if (f_v) {
			cout << "read a set of size " << m << endl;
		}
		the_ranks = NEW_lint(m);
		for (i = 0; i < m; i++) {

			if (FALSE) {
				cout << i << " : ";
				Orbiter->Int_vec.print(cout, M + i * n, n);
				cout << endl;
			}

			LG->A_linear->make_element(Elt, M + i * n, 0 /* verbose_level */);
			if (FALSE) {
				cout << "computing rank of " << endl;
				LG->A_linear->element_print_latex(Elt, cout);
			}

			the_ranks[i] = S->element_rank_lint(Elt);
			if (FALSE) {
				cout << i << " : " << the_ranks[i] << endl;
			}
		}

		FREE_int(M);
	}

	Transporter = NEW_OBJECT(vector_ge);
	Transporter->init(S->A, 0);
	{
		int *M;
		Fio.int_matrix_read_csv(fname_transporter,
				M, m, n, 0 /*verbose_level*/);
		if (f_v) {
			cout << "read a set of size " << m << endl;
		}
		Transporter->allocate(m, 0);
		for (i = 0; i < m; i++) {

			if (FALSE) {
				cout << i << " : ";
				Orbiter->Int_vec.print(cout, M + i * n, n);
				cout << endl;
			}

			LG->A_linear->make_element(Transporter->ith(i), M + i * n, 0 /* verbose_level */);
			if (FALSE) {
				cout << "computing rank of " << endl;
				LG->A_linear->element_print_latex(Elt, cout);
			}

		}

		FREE_int(M);
	}




	if (f_v) {
		cout << "computing conjugacy classes on the set " << endl;
		Orbiter->Lint_vec.print(cout, the_ranks, m);
		cout << endl;
	}

	algebra_global_with_action Algebra;

	if (f_v) {
		cout << "group_theoretic_activity::do_orbits_on_group_elements_under_conjugation "
				"before Algebra.orbits_under_conjugation" << endl;
	}
	Algebra.orbits_under_conjugation(
			the_ranks, m, S,
			LG->Strong_gens,
			Transporter,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_orbits_on_group_elements_under_conjugation "
				"after Algebra.orbits_under_conjugation" << endl;
	}




	FREE_int(Elt);

	if (f_v) {
		cout << "group_theoretic_activity::do_orbits_on_group_elements_under_conjugation done" << endl;
	}
}

void group_theoretic_activity::do_Andre_Bruck_Bose_construction(int spread_no,
		int f_Fano, int f_arcs, int f_depth, int depth,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *spread_elements_numeric; // do not free
	action *An;
	action *An1;
	vector_ge *gens;
	translation_plane_via_andre_model *Andre;
	matrix_group *M; // do not free

	int f_semilinear = FALSE;
	int n, k, q;

	const char *stab_order;
	longinteger_object stab_go;
	int order_of_plane;
	number_theory_domain NT;
	knowledge_base K;

	if (f_v) {
		cout << "group_theoretic_activity::do_Andre_Bruck_Bose_construction" << endl;
	}

	An = A1;
	An1 = A2;
	F = An->matrix_group_finite_field();
	M = An->get_matrix_group();
	f_semilinear = An->is_semilinear_matrix_group();
	n = An->matrix_group_dimension();
	if (ODD(n)) {
		cout << "group_theoretic_activity::do_Andre_Bruck_Bose_construction "
				"dimension must be even" << endl;
		exit(1);
	}
	k = n >> 1;
	q = F->q;

	if (f_v) {
		cout << "group_theoretic_activity::do_Andre_Bruck_Bose_construction n=" << n << " k=" << k << " q=" << q << endl;
	}

	order_of_plane = NT.i_power_j(q, k);


	int sz;
	//vector_ge *nice_gens;

	spread_elements_numeric = K.Spread_representative(q, k, spread_no, sz);




	An->stabilizer_of_spread_representative(q, k, spread_no,
			gens, stab_order, verbose_level);

	stab_go.create_from_base_10_string(stab_order, 0 /* verbose_level */);

	if (f_v) {
		cout << "Spread stabilizer has order " << stab_go << endl;
	}

	Andre = NEW_OBJECT(translation_plane_via_andre_model);

	Andre->init(spread_elements_numeric, k, An, An1,
		gens /*spread_stab_gens*/, stab_go, label, verbose_level);

	Andre->create_latex_report(verbose_level);


	if (f_Fano) {
		char prefix[1000];
		int nb_subplanes;

		sprintf(prefix, "Fano_TP_%d_", spread_no);

		Andre->classify_subplanes(prefix, verbose_level);

		int target_depth;

		if (f_depth) {
			target_depth = depth;
			}
		else {
			target_depth = 7;
			}

		nb_subplanes = Andre->arcs->nb_orbits_at_level(target_depth);

		cout << "Translation plane " << q << "#" << spread_no << " has "
				<<  nb_subplanes << " partial Fano subplanes "
						"(up to isomorphism) at depth "
				<< target_depth << endl;
		}
	else if (f_arcs) {
		char prefix[1000];
		int nb;

		int target_depth;

		if (f_depth) {
			target_depth = depth;
			}
		else {
			target_depth = order_of_plane + 2;
				// we are looking for hyperovals
			}


		sprintf(prefix, "Arcs_TP_%d_", spread_no);

		Andre->classify_arcs(prefix, target_depth, verbose_level);


		nb = Andre->arcs->nb_orbits_at_level(target_depth);

		cout << "Translation plane " << q << "#" << spread_no << " has "
				<<  nb << " Arcs of size " << target_depth
				<< " (up to isomorphism)" << endl;
		}

	FREE_OBJECT(Andre);
	FREE_OBJECT(gens);
	if (f_v) {
		cout << "group_theoretic_activity::do_Andre_Bruck_Bose_construction done" << endl;
	}
}

void group_theoretic_activity::do_BLT_starter(
		linear_group *LG,
		int starter_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_BLT_starter" << endl;
	}

	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		cout << "please use option -poset_classification_control" << endl;
		exit(1);
		//Control = NEW_OBJECT(poset_classification_control);
	}

	blt_set_classify *BLT;

	BLT = NEW_OBJECT(blt_set_classify);

	if (f_v) {
		cout << "group_theoretic_activity::do_BLT_starter before BLT->init_basic" << endl;
	}
	BLT->init_basic(LG->A2,
			LG->Strong_gens,
			starter_size,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_BLT_starter after BLT->init_basic" << endl;
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_BLT_starter before BLT->compute_starter" << endl;
	}
	BLT->compute_starter(
			Control,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_BLT_starter after BLT->compute_starter" << endl;
	}

	FREE_OBJECT(BLT);

	if (f_v) {
		cout << "group_theoretic_activity::do_BLT_starter done" << endl;
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

