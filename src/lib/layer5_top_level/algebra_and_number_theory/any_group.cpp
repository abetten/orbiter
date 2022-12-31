/*
 * any_group.cpp
 *
 *  Created on: Sep 26, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


any_group::any_group()
{
	f_linear_group = FALSE;
	LG = NULL;

	f_permutation_group = FALSE;
	PGC = NULL;

	f_modified_group = FALSE;
	MGC = NULL;

	A_base = NULL;
	A = NULL;

	//std::string label;
	//std::string label_tex;

	Subgroup_gens = NULL;
	Subgroup_sims = NULL;

}

any_group::~any_group()
{
}


void any_group::init_linear_group(groups::linear_group *LG, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::init_linear_group" << endl;
	}

	f_linear_group = TRUE;
	any_group::LG = LG;

	A_base = LG->A_linear;
	A = LG->A2;

	label.assign(LG->label);
	label_tex.assign(LG->label_tex);

	if (!LG->f_has_strong_generators) {
		cout << "any_group::init_linear_group "
				"!LG->f_has_strong_generators" << endl;
		exit(1);
	}
	Subgroup_gens = LG->Strong_gens;

	if (f_v) {
		cout << "any_group::init_linear_group "
				"before Subgroup_gens->create_sims" << endl;
	}
	Subgroup_sims = Subgroup_gens->create_sims(0/*verbose_level*/);
	if (f_v) {
		cout << "any_group::init_linear_group "
				"after Subgroup_gens->create_sims" << endl;
		cout << "any_group::init_linear_group group order is ";
		Subgroup_sims->print_group_order(cout);
		cout << endl;
	}

	if (f_v) {
		cout << "any_group::init_linear_group done" << endl;
	}
}

void any_group::init_permutation_group(groups::permutation_group_create *PGC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::init_permutation_group" << endl;
	}

	f_permutation_group = TRUE;
	any_group::PGC = PGC;

	A_base = PGC->A_initial;
	A = PGC->A2;

	label.assign(PGC->label);
	label_tex.assign(PGC->label_tex);
	if (f_v) {
		cout << "any_group::init_permutation_group label = " << label << endl;
		cout << "any_group::init_permutation_group label_tex = " << label_tex << endl;
	}

	if (!PGC->f_has_strong_generators) {
		cout << "any_group::init_permutation_group "
				"!PGC->f_has_strong_generators" << endl;
		exit(1);
	}
	Subgroup_gens = PGC->Strong_gens;

	if (f_v) {
		cout << "any_group::init_permutation_group "
				"before Subgroup_gens->create_sims_in_different_action" << endl;
	}
	Subgroup_sims = Subgroup_gens->create_sims_in_different_action(
			A_base, 0 /*verbose_level*/);
	//Subgroup_sims = Subgroup_gens->create_sims(verbose_level);
	if (f_v) {
		cout << "any_group::init_permutation_group "
				"after Subgroup_gens->create_sims_in_different_action" << endl;
	}

	if (f_v) {
		cout << "any_group::init_permutation_group done" << endl;
	}
}

void any_group::init_modified_group(modified_group_create *MGC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::init_modified_group" << endl;
	}

	f_modified_group = TRUE;
	any_group::MGC = MGC;

	A_base = MGC->A_base;
	A = MGC->A_modified;

	if (!MGC->f_has_strong_generators) {
		cout << "any_group::init_linear_group "
				"!PGC->f_has_strong_generators" << endl;
		exit(1);
	}
	Subgroup_gens = MGC->Strong_gens;

	label.assign(MGC->label);
	label_tex.assign(MGC->label_tex);

	if (f_v) {
		cout << "any_group::init_modified_group done" << endl;
	}
}



void any_group::create_latex_report(
		graphics::layered_graph_draw_options *O,
		int f_sylow, int f_group_table, int f_classes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "any_group::create_latex_report" << endl;
	}

	if (f_linear_group) {
		if (f_v) {
			cout << "any_group::create_latex_report linear group" << endl;
		}
		LG->create_latex_report(O,
				f_sylow, f_group_table, f_classes,
				verbose_level);
	}
	else if (f_permutation_group) {
		if (f_v) {
			cout << "any_group::create_latex_report permutation group" << endl;
		}
		create_latex_report_for_permutation_group(
					O,
					verbose_level);
	}
	else if (f_modified_group) {
		if (f_v) {
			cout << "any_group::create_latex_report modified group" << endl;
		}
		create_latex_report_for_modified_group(
					O,
					verbose_level);

	}
	else {
		cout << "any_group::create_latex_report unknown type of group" << endl;
		exit(1);
	}
}

void any_group::export_group_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "any_group::export_group_table" << endl;
	}



	string fname;
	int *Table;
	long int n;

	if (f_v) {
		cout << "any_group::export_group_table "
				"before create_group_table" << endl;
	}
	create_group_table(Table, n, verbose_level);
	if (f_v) {
		cout << "any_group::export_group_table "
				"after create_group_table" << endl;
	}


	orbiter_kernel_system::file_io Fio;


	fname.assign(label);
	fname.append("_group_table.csv");

	Fio.int_matrix_write_csv(fname, Table, n, n);
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_int(Table);

	if (f_v) {
		cout << "any_group::export_group_table done" << endl;
	}


}


void any_group::do_export_orbiter(actions::action *A2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_export_orbiter" << endl;
	}

	string fname;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "any_group::do_export_orbiter label=" << label << endl;
	}
	fname.assign(label);
	fname.append(".makefile");
	{
		ofstream fp(fname);

		if (Subgroup_gens) {
			if (f_v) {
				cout << "any_group::do_export_orbiter "
						"using Subgroup_gens" << endl;
			}
			Subgroup_gens->export_to_orbiter_as_bsgs(A2,
					fname, label, label_tex, verbose_level);
		}
		else if (A->f_has_strong_generators) {
			if (f_v) {
				cout << "any_group::do_export_orbiter "
						"using A_base->Strong_gens" << endl;
			}
			A_base->export_to_orbiter_as_bsgs(fname,
					label, label_tex, A_base->Strong_gens, verbose_level);
		}
		else {
			cout << "any_group::do_export_orbiter "
					"no generators to export" << endl;
			exit(1);
		}

	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "any_group::do_export_orbiter done" << endl;
	}
}



void any_group::do_export_gap(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_export_gap" << endl;
	}

	string fname;
	orbiter_kernel_system::file_io Fio;

	fname.assign(label);
	fname.append("_generators.gap");
	{
		ofstream fp(fname);

		if (Subgroup_gens) {
			if (f_v) {
				cout << "any_group::do_export_gap "
						"using Subgroup_gens" << endl;
			}
			Subgroup_gens->print_generators_gap(fp);
		}
		else if (A->f_has_strong_generators) {
			if (f_v) {
				cout << "any_group::do_export_gap "
						"using A_base->Strong_gens" << endl;
			}
			A->Strong_gens->print_generators_gap_in_different_action(fp, A);
		}
		else {
			cout << "any_group::do_export_gap "
					"no generators to export" << endl;
			exit(1);
		}

	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "any_group::do_export_gap done" << endl;
	}
}

void any_group::do_export_magma(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_export_magma" << endl;
	}

	string fname;
	orbiter_kernel_system::file_io Fio;

	fname.assign(label);
	fname.append("_generators.magma");
	{
		ofstream fp(fname);
		groups::strong_generators *SG;

		SG = get_strong_generators();

		SG->export_magma(A, fp, verbose_level);
	}
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "any_group::do_export_magma done" << endl;
	}
}

void any_group::do_canonical_image_GAP(std::string &input_set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_canonical_image_GAP" << endl;
	}

	string fname;
	orbiter_kernel_system::file_io Fio;

	fname.assign(label);
	fname.append("_canonical_image.gap");
	{
		ofstream ost(fname);
		groups::strong_generators *SG;

		SG = get_strong_generators();
		SG->canonical_image_GAP(input_set_text, ost);
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "any_group::do_canonical_image_GAP done" << endl;
	}
}


void any_group::create_group_table(int *&Table, long int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::create_group_table" << endl;
	}
	int goi;



	goi = Subgroup_sims->group_order_lint();

	if (f_v) {
			cout << "any_group::create_group_table group order = " << goi << endl;
	}


	Subgroup_sims->create_group_table(Table, n, verbose_level);

	if (n != goi) {
		cout << "any_group::create_group_table n != goi" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "any_group::create_group_table done" << endl;
	}
}

void any_group::normalizer(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::normalizer" << endl;
	}
	string fname_magma_prefix;
	groups::sims *G;
	groups::sims *H;
	groups::strong_generators *gens_N;
	ring_theory::longinteger_object N_order;
	ring_theory::longinteger_object G_order;
	ring_theory::longinteger_object H_order;



	fname_magma_prefix.assign(label);
	fname_magma_prefix.append("_normalizer");


	H = Subgroup_gens->create_sims(verbose_level);

	if (f_linear_group) {
		G = LG->initial_strong_gens->create_sims(verbose_level);
	}
	else {
		G = PGC->A_initial->Strong_gens->create_sims(verbose_level);
	}

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	//H = LG->Strong_gens->create_sims(verbose_level);


	groups::magma_interface M;


	if (f_v) {
		cout << "group order G = " << G->group_order_lint() << endl;
			cout << "group order H = " << H->group_order_lint() << endl;
			cout << "before M.normalizer_using_MAGMA" << endl;
	}
	M.normalizer_using_MAGMA(A, fname_magma_prefix,
			G, H, gens_N, verbose_level);

	G->group_order(G_order);
	H->group_order(H_order);
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




	{
		string fname, title, author, extra_praeamble;
		char str[1000];

		fname.assign(fname_magma_prefix);
		fname.append(".tex");
		snprintf(str, 1000, "Normalizer of subgroup %s", label_tex.c_str());
		title.assign(str);


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);

			ost << "\\noindent The group $" << label_tex << "$ "
					"of order " << H_order << " is:\\\\" << endl;
			Subgroup_gens->print_generators_tex(ost);

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
		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	if (f_v) {
		cout << "any_group::normalizer done" << endl;
	}
}

void any_group::centralizer(
		std::string &element_label,
		std::string &element_description_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::centralizer" << endl;
	}

	groups::magma_interface M;
	groups::sims *S;
	groups::strong_generators *SG;

	SG = get_strong_generators();

	S = SG->create_sims(verbose_level);

	if (f_v) {
		cout << "any_group::centralizer "
				"before M.centralizer_of_element" << endl;
	}
	M.centralizer_of_element(
			A, S,
			element_description_text,
			element_label, verbose_level);
	if (f_v) {
		cout << "any_group::centralizer "
				"after M.centralizer_of_element" << endl;
	}

	FREE_OBJECT(S);

	if (f_v) {
		cout << "any_group::centralizer done" << endl;
	}
}

void any_group::permutation_representation_of_element(
		std::string &element_description_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::permutation_representation_of_element" << endl;
	}

	algebra_global_with_action Algebra;

	if (f_v) {
		cout << "any_group::permutation_representation_of_element "
				"before Algebra.permutation_representation_of_element" << endl;
	}
	Algebra.permutation_representation_of_element(
			A,
			element_description_text,
			verbose_level);
	if (f_v) {
		cout << "any_group::permutation_representation_of_element "
				"after Algebra.permutation_representation_of_element" << endl;
	}

	if (f_v) {
		cout << "any_group::permutation_representation_of_element done" << endl;
	}
}

void any_group::normalizer_of_cyclic_subgroup(
		std::string &element_label,
		std::string &element_description_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::normalizer_of_cyclic_subgroup" << endl;
	}

	groups::magma_interface M;
	groups::sims *S;
	groups::strong_generators *SG;

	SG = get_strong_generators();
	S = SG->create_sims(verbose_level);

	if (f_v) {
		cout << "any_group::normalizer_of_cyclic_subgroup "
				"before M.normalizer_of_cyclic_subgroup" << endl;
	}
	M.normalizer_of_cyclic_subgroup(
			A, S,
			element_description_text,
			element_label, verbose_level);
	if (f_v) {
		cout << "any_group::normalizer_of_cyclic_subgroup "
				"after M.normalizer_of_cyclic_subgroup" << endl;
	}

	FREE_OBJECT(S);

	if (f_v) {
		cout << "any_group::normalizer_of_cyclic_subgroup done" << endl;
	}
}


void any_group::do_find_subgroups(
		int order_of_subgroup,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_find_subgroups" << endl;
	}

	groups::magma_interface M;
	groups::sims *S;
	groups::strong_generators *SG;

	SG = get_strong_generators();

	int nb_subgroups;
	groups::strong_generators *H_gens;
	groups::strong_generators *N_gens;


	S = SG->create_sims(verbose_level);

	if (f_v) {
		cout << "any_group::do_find_subgroups before M.find_subgroups" << endl;
	}
	M.find_subgroups(
			A, S,
			order_of_subgroup,
			A->label,
			nb_subgroups,
			H_gens,
			N_gens,
			verbose_level);
	if (f_v) {
		cout << "any_group::do_find_subgroups after M.find_subgroups" << endl;
	}


	if (f_v) {
		cout << "any_group::do_find_subgroups We found " << nb_subgroups << " subgroups" << endl;
	}


	string fname;
	string title;
	string author;

	author.assign("Orbiter");
	string extras_for_preamble;

	orbiter_kernel_system::file_io Fio;

	fname.assign(A->label);
	fname.append("_report.tex");


	char str[1000];

	snprintf(str, sizeof(str), "Subgroups of order $%d$ in $", order_of_subgroup);
	title.assign(str);
	title.append(A->label_tex);
	title.append("$");


	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;
		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */, TRUE /* f_title */,
			title, author,
			FALSE /*f_toc*/, FALSE /* f_landscape*/, FALSE /* f_12pt*/,
			TRUE /*f_enlarged_page*/, TRUE /* f_pagenumbers*/,
			extras_for_preamble);

		A->report_groups_and_normalizers(fp,
				nb_subgroups, H_gens, N_gens,
				verbose_level);

		L.foot(fp);
	}

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	FREE_OBJECT(S);
	FREE_OBJECTS(H_gens);
	FREE_OBJECTS(N_gens);


	if (f_v) {
		cout << "any_group::do_find_subgroups done" << endl;
	}
}


void any_group::print_elements(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::print_elements" << endl;
	}

	groups::strong_generators *SG;

	SG = get_strong_generators();


	groups::sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = SG->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	ring_theory::longinteger_object go;
	int i; //, cnt;

	Elt = NEW_int(A->elt_size_in_int);
	H->group_order(go);


	//cnt = 0;
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
		cout << "any_group::print_elements done" << endl;
	}
}

void any_group::print_elements_tex(int f_with_permutation,
		int f_override_action, actions::action *A_special,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::print_elements_tex" << endl;
	}

	orbiter_kernel_system::file_io Fio;

	groups::strong_generators *SG;


	SG = get_strong_generators();

	groups::sims *H;

	H = SG->create_sims(verbose_level);

	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	ring_theory::longinteger_object go;

	Elt = NEW_int(A->elt_size_in_int);
	H->group_order(go);

	string fname;

	fname.assign(label);
	fname.append("_elements.tex");


	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;
		L.head_easy(fp);

		H->print_all_group_elements_tex(fp,
				f_with_permutation,
				f_override_action, A_special);
		//H->print_all_group_elements_tree(fp);
		//H->print_all_group_elements_with_permutations_tex(fp);

		//Schreier.print_and_list_orbits_tex(fp);


		L.foot(fp);
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	fname.assign(label);
	fname.append("_elements_tree.txt");


	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;
		//L.head_easy(fp);

		//H->print_all_group_elements_tex(fp);
		H->print_all_group_elements_tree(fp);
		//H->print_all_group_elements_with_permutations_tex(fp);

		//Schreier.print_and_list_orbits_tex(fp);

		fp << -1 << endl;

		//L.foot(fp);
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	FREE_int(Elt);
	if (f_v) {
		cout << "any_group::print_elements_tex done" << endl;
	}
}

void any_group::order_of_products_of_elements_by_rank(
		std::string &Elements_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::order_of_products_of_elements_by_rank" << endl;
	}

	groups::strong_generators *SG;


	SG = get_strong_generators();

	groups::sims *H;

	H = SG->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	ring_theory::longinteger_object go;

	Elt = NEW_int(A->elt_size_in_int);
	H->group_order(go);

	int *elements;
	int nb_elements;

	Int_vec_scan(Elements_text, elements, nb_elements);


	string fname;

	fname.assign(label);
	fname.append("_elements.tex");


	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;
		L.head_easy(fp);

		int f_override_action = FALSE;

		H->print_all_group_elements_tex(fp, FALSE, f_override_action, NULL);
		//H->print_all_group_elements_with_permutations_tex(fp);

		//Schreier.print_and_list_orbits_tex(fp);

		int *order_table;
		int i;


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

		//latex_interface L;

		fp << "$$" << endl;
		L.print_integer_matrix_with_labels(fp, order_table,
				nb_elements, nb_elements, elements, elements, TRUE /* f_tex */);
		fp << "$$" << endl;

		L.foot(fp);
	}

	FREE_int(Elt);
	if (f_v) {
		cout << "any_group::order_of_products_of_elements_by_rank done" << endl;
	}
}

void any_group::save_elements_csv(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::save_elements_csv" << endl;
	}


	Subgroup_sims->all_elements_save_csv(fname, verbose_level);

	if (f_v) {
		cout << "any_group::save_elements_csv done" << endl;
	}
}

void any_group::export_inversion_graphs(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::export_inversion_graphs" << endl;
	}



	Subgroup_sims->all_elements_export_inversion_graphs(fname, verbose_level);

	if (f_v) {
		cout << "any_group::export_inversion_graphs done" << endl;
	}
}



void any_group::multiply_elements_csv(std::string &fname1,
		std::string &fname2, std::string &fname3,
		int f_column_major_ordering, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::multiply_elements_csv" << endl;
	}

	data_structures_groups::vector_ge V1, V2, V3;
	int n1, n2, n3;
	int i, j, k;

	V1.read_column_csv(fname1, A, 1 /* col_idx */, verbose_level);
	n1 = V1.len;

	V2.read_column_csv(fname2, A, 1 /* col_idx */, verbose_level);
	n2 = V2.len;

	n3 = n1 * n2;

	if (f_v) {
		cout << "any_group::multiply_elements_csv "
				"n1=" << V1.len << " n2=" << V2.len << " n3=" << n3 << endl;
	}

	V3.init(A, 0 /* vl */);
	V3.allocate(n3, 0 /* vl */);

	if (f_column_major_ordering) {
		k = 0;
		for (j = 0; j < n2; j++) {
			for (i = 0; i < n1; i++, k++) {
				A->mult(V1.ith(i), V2.ith(j), V3.ith(k));
			}
		}
	}
	else {
		k = 0;
		for (i = 0; i < n1; i++) {
			for (j = 0; j < n2; j++, k++) {
				A->mult(V1.ith(i), V2.ith(j), V3.ith(k));
			}
		}

	}

	//Subgroup_sims->all_elements_save_csv(fname, verbose_level);
	V3.save_csv(fname3, verbose_level);

	if (f_v) {
		cout << "any_group::multiply_elements_csv done" << endl;
	}
}

void any_group::apply_elements_to_set_csv(std::string &fname1, std::string &fname2,
		std::string &set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::apply_elements_to_set_csv" << endl;
	}

	data_structures_groups::vector_ge V1;
	int n1;
	int i, j;
	long int *set;
	long int *set_image;
	int *set_image_int;
	int sz;
	int *Rk;
	combinatorics::combinatorics_domain Combi;


	Lint_vec_scan(set_text, set, sz);

	V1.read_column_csv(fname1, A, 1 /* col_idx */, verbose_level);
	n1 = V1.len;

	set_image = NEW_lint(sz);
	set_image_int = NEW_int(sz);
	Rk = NEW_int(n1);

	if (f_v) {
		cout << "any_group::apply_elements_to_set_csv "
				"n1=" << V1.len << endl;
	}

	for (i = 0; i < n1; i++) {
		A->map_a_set_and_reorder(set, set_image,
				sz, V1.ith(i), 0 /* verbose_level */);

		for (j = 0; j < sz; j++) {
			set_image_int[j] = set_image[j];
		}
		Rk[i] = Combi.rank_k_subset(set_image_int, A->degree, sz);

		cout << i << " : ";
		Lint_vec_print(cout, set_image, sz);
		cout << i << " : ";
		cout << Rk[i];
		cout << endl;

	}


	cout << "Image sets by rank: ";
	Int_vec_print_fully(cout, Rk, n1);
	cout << endl;


	if (!Combi.is_permutation(Rk, n1)) {
		cout << "any_group::apply_elements_to_set_csv "
				"The set Rk is *not* a permutation" << endl;
		exit(1);
	}
	else {
		cout << "any_group::apply_elements_to_set_csv "
				"The set Rk is a permutation" << endl;
	}

	FREE_lint(set);
	FREE_lint(set_image);
	FREE_int(set_image_int);

	if (f_v) {
		cout << "any_group::apply_elements_to_set_csv done" << endl;
	}
}




void any_group::element_rank(std::string &elt_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::element_rank" << endl;
	}

	actions::action *A1;

	A1 = A;

	groups::sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	groups::strong_generators *SG;

	SG = get_strong_generators();
	H = SG->create_sims(verbose_level);

	if (f_v) {
		//cout << "group order G = " << G->group_order_int() << endl;
		cout << "group order H = " << H->group_order_lint() << endl;
	}

	if (f_v) {
		cout << "creating element " << elt_data << endl;
	}
	int *Elt;

	Elt = NEW_int(A1->elt_size_in_int);
	A1->make_element_from_string(Elt, elt_data, 0);

	if (f_v) {
		cout << "Element :" << endl;
		A1->element_print(Elt, cout);
		cout << endl;
	}

	ring_theory::longinteger_object a;
	H->element_rank(a, Elt);

	if (f_v) {
		cout << "The rank of the element is " << a << endl;
	}


	FREE_int(Elt);
	FREE_OBJECT(H);



	if (f_v) {
		cout << "any_group::element_rank done" << endl;
	}
}

void any_group::element_unrank(std::string &rank_string, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::element_unrank" << endl;
	}

	actions::action *A1;

	A1 = A;

	groups::sims *H;
	groups::strong_generators *SG;

	SG = get_strong_generators();

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = SG->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	if (f_v) {
		cout << "group order H = " << H->group_order_lint() << endl;
	}

	int *Elt;

	Elt = NEW_int(A1->elt_size_in_int);


	ring_theory::longinteger_object a;

	a.create_from_base_10_string(rank_string.c_str(), 0 /*verbose_level*/);

	if (f_v) {
		cout << "Creating element of rank " << a << endl;
	}

	H->element_unrank(a, Elt);

	if (f_v) {
		cout << "Element :" << endl;
		A1->element_print(Elt, cout);
		cout << endl;
	}


	FREE_int(Elt);
	FREE_OBJECT(H);

	if (f_v) {
		cout << "any_group::element_unrank done" << endl;
	}
}



void any_group::conjugacy_class_of(std::string &label,
		std::string &elt_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::conjugacy_class_of" << endl;
	}

#if 0
	sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;
#endif

	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);

	ring_theory::longinteger_object a, b;

#if 1
	if (f_v) {
		cout << "any_group::conjugacy_class_of creating element " << elt_data << endl;
	}

	A->make_element_from_string(Elt, elt_data, 0);

	Subgroup_sims->element_rank(a, Elt);

	a.assign_to(b);


#else


	a.create_from_base_10_string(rank_string.c_str(), 0 /*verbose_level*/);

	cout << "Creating element of rank " << a << endl;

	a.assign_to(b);

	H->element_unrank(a, Elt);

#endif

	if (f_v) {
		cout << "any_group::conjugacy_class_of Element :" << endl;
		A->element_print(Elt, cout);
		cout << endl;
	}


	actions::action *A_conj;

	A_conj = A->create_induced_action_by_conjugation(
			Subgroup_sims /*Base_group*/, FALSE /* f_ownership */,
			verbose_level);


	if (f_v) {
		cout << "created action A_conj of degree " << A_conj->degree << endl;
	}

#if 0
	schreier *Sch;

	Sch = LG->Strong_gens->orbit_of_one_point_schreier(
			A_conj, b.as_lint(), verbose_level);

	cout << "Orbits on itself by conjugation:\\\\" << endl;
	Sch->print_orbit_reps(cout);


	FREE_OBJECT(Sch);
#else


	orbits_schreier::orbit_of_sets Orb;
	long int set[1];
	orbiter_kernel_system::file_io Fio;

	set[0] = b.as_lint();

	Orb.init(A, A_conj,
			set, 1 /* sz */,
			Subgroup_gens->gens, //LG->Strong_gens->gens,
			verbose_level);
	if (f_v) {
		cout << "Found an orbit of size " << Orb.used_length << endl;
	}

	std::vector<long int> Orbit;

	if (f_v) {
		cout << "before Orb.get_orbit_of_points" << endl;
	}
	Orb.get_orbit_of_points(Orbit, verbose_level);
	if (f_v) {
		cout << "Found an orbit of size " << Orbit.size() << endl;
	}

	int *M;
	int i, j;

	M = NEW_int(Orbit.size() * A->make_element_size);
	for (i = 0; i < Orbit.size(); i++) {
		Subgroup_sims->element_unrank_lint(Orbit[i], Elt);
		for (j = 0; j < A->make_element_size; j++) {
			M[i * A->make_element_size + j] = Elt[j];
		}
		//M[i] = Orbit[i];
	}
	string fname;

	fname.assign(LG->label);
	fname.append("_class_of_");
	fname.append(label);
	fname.append(".csv");

	Fio.int_matrix_write_csv(fname, M, Orbit.size(), A->make_element_size);

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_int(M);

#endif

	FREE_OBJECT(A_conj);
	//FREE_OBJECT(H);



	FREE_int(Elt);
	if (f_v) {
		cout << "any_group::conjugacy_class_of done" << endl;
	}
}


void any_group::do_reverse_isomorphism_exterior_square(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 5);

	if (f_v) {
		cout << "any_group::do_reverse_isomorphism_exterior_square" << endl;
	}


	if (LG->f_has_nice_gens) {
		if (f_v) {
			cout << "any_group::do_reverse_isomorphism_exterior_square nice generators are:" << endl;
			LG->nice_gens->print(cout);
		}
		LG->nice_gens->reverse_isomorphism_exterior_square(verbose_level);
	}
	else {
		if (f_v) {
			cout << "any_group::do_reverse_isomorphism_exterior_square strong generators are:" << endl;
			LG->Strong_gens->print_generators_in_latex_individually(cout);
		}
		LG->Strong_gens->reverse_isomorphism_exterior_square(verbose_level);
	}

	if (f_v) {
		cout << "any_group::do_reverse_isomorphism_exterior_square done" << endl;
	}
}



void any_group::orbits_on_set_system_from_file(std::string &fname_csv,
		int number_of_columns, int first_column, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::orbits_on_set_system_from_file" << endl;
	}
	if (f_v) {
		cout << "computing orbits on set system from file "
			<< fname_csv << ":" << endl;
	}
	orbiter_kernel_system::file_io Fio;
	int *M;
	int m, n;
	long int *Table;
	int i, j;

	Fio.int_matrix_read_csv(fname_csv, M,
			m, n, verbose_level);
	if (f_v) {
		cout << "read a matrix of size " << m << " x " << n << endl;
	}


	//orbits_on_set_system_first_column = atoi(argv[++i]);
	//orbits_on_set_system_number_of_columns = atoi(argv[++i]);


	Table = NEW_lint(m * number_of_columns);
	for (i = 0; i < m; i++) {
		for (j = 0; j < number_of_columns; j++) {
			Table[i * number_of_columns + j] =
					M[i * n + first_column + j];
		}
	}
	actions::action *A_on_sets;
	int set_size;

	set_size = number_of_columns;

	if (f_v) {
		cout << "creating action on sets:" << endl;
	}
	A_on_sets = A->create_induced_action_on_sets(m /* nb_sets */,
			set_size, Table,
			verbose_level);

	groups::schreier *Sch;
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
			Lint_vec_print(cout, Table + a * set_size, set_size);
			cout << endl;
			//Sch->print_and_list_orbit_tex(i, ost);
		}
	}
	string fname;
	data_structures::string_tools ST;

	fname.assign(fname_csv);
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
		cout << "any_group::orbits_on_set_system_from_file done" << endl;
	}
}

void any_group::orbits_on_set_from_file(std::string &fname_csv, int verbose_level)
// called from group_theoretic_activity: f_orbit_of_set_from_file
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::orbits_on_set_from_file" << endl;
	}

	if (f_v) {
		cout << "any_group::orbits_on_set_from_file "
				"computing orbit of set from file "
			<< fname_csv << ":" << endl;
	}
	orbiter_kernel_system::file_io Fio;
	long int *the_set;
	int set_sz;

	Fio.read_set_from_file(fname_csv,
			the_set, set_sz, 0 /*verbose_level*/);
	if (f_v) {
		cout << "any_group::orbits_on_set_from_file "
				"read a set of size " << set_sz << endl;
	}


	string label_set;
	data_structures::string_tools ST;

	label_set.assign(fname_csv);
	ST.chop_off_extension(label_set);

	algebra_global_with_action Algebra;
	long int *Table;
	int size;

	if (f_v) {
		cout << "any_group::orbits_on_set_from_file "
				"before Algebra.compute_orbit_of_set" << endl;
	}

	Algebra.compute_orbit_of_set(
			the_set, set_sz,
			A_base, A,
			Subgroup_gens->gens,
			label_set,
			label,
			Table, size,
			verbose_level);

	if (f_v) {
		cout << "any_group::orbits_on_set_from_file "
				"after Algebra.compute_orbit_of_set" << endl;
	}

	FREE_lint(Table);

	if (f_v) {
		cout << "any_group::orbits_on_set_from_file done" << endl;
	}
}


void any_group::orbit_of(int point_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::orbit_of" << endl;
	}
	groups::schreier *Sch;
	Sch = NEW_OBJECT(groups::schreier);

	if (f_v) {
		cout << "any_group::orbit_of computing orbit of point " << point_idx << ":" << endl;
	}

	//A->all_point_orbits(*Sch, verbose_level);

	Sch->init(A_base, verbose_level - 2);

#if 0
	if (!A->f_has_strong_generators) {
		cout << "any_group::orbit_of !f_has_strong_generators" << endl;
		exit(1);
		}
#endif

	Sch->init_generators(*Subgroup_gens->gens /* *strong_generators */, verbose_level - 2);
	Sch->initialize_tables();
	Sch->compute_point_orbit(point_idx, verbose_level);

	int orbit_idx = 0;

	if (f_v) {
		cout << "any_group::orbit_of computing orbit of point " << point_idx << " done" << endl;
	}

	string fname_tree_mask;
	char str[1000];

	fname_tree_mask.assign(label);
	snprintf(str, sizeof(str), "_orbit_of_point_%d.layered_graph", point_idx);
	fname_tree_mask.append(str);


	Sch->export_tree_as_layered_graph(orbit_idx,
			fname_tree_mask,
			verbose_level - 1);

	groups::strong_generators *SG_stab;
	ring_theory::longinteger_object full_group_order;

	Subgroup_gens->group_order(full_group_order);


	if (f_v) {
		cout << "any_group::orbit_of computing the stabilizer "
				"of the rep of orbit " << orbit_idx << endl;
		cout << "any_group::orbit_of orbit length = " << Sch->orbit_len[orbit_idx] << endl;
	}

	if (f_v) {
		cout << "any_group::orbit_of before Sch->stabilizer_orbit_rep" << endl;
	}

	SG_stab = Sch->stabilizer_orbit_rep(
			A_base,
			full_group_order,
			0 /* orbit_idx */, 0 /*verbose_level*/);

	if (f_v) {
		cout << "any_group::orbit_of after Sch->stabilizer_orbit_rep" << endl;
	}


	cout << "any_group::orbit_of "
			"The stabilizer of the orbit rep has been computed:" << endl;
	SG_stab->print_generators(cout);
	SG_stab->print_generators_tex();

#if 0

	groups::schreier *shallow_tree;

	if (f_v) {
		cout << "any_group::orbit_of "
				"computing shallow Schreier tree:" << endl;
	}

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

	if (f_v) {
		cout << "any_group::orbit_of "
				"computing shallow Schreier tree done." << endl;
	}

	fname_tree_mask.assign(label);
	fname_tree_mask.append("_%d_shallow.layered_graph");

	shallow_tree->export_tree_as_layered_graph(0 /* orbit_no */,
			fname_tree_mask,
			verbose_level - 1);
#endif

	if (f_v) {
		cout << "any_group::orbit_of done" << endl;
	}
}

void any_group::orbits_on_points(groups::orbits_on_something *&Orb, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::orbits_on_points" << endl;
	}
	algebra_global_with_action Algebra;


	int f_load_save = TRUE;
	string prefix;

	prefix.assign(label);

	if (f_v) {
		cout << "any_group::orbits_on_points "
				"before Algebra.orbits_on_points" << endl;
	}
	Algebra.orbits_on_points(
			A,
			Subgroup_gens,
			f_load_save,
			prefix,
			Orb,
			verbose_level);
	if (f_v) {
		cout << "any_group::orbits_on_points "
				"after Algebra.orbits_on_points" << endl;
	}


	if (f_v) {
		cout << "any_group::orbits_on_points done" << endl;
	}
}

void any_group::create_latex_report_for_permutation_group(
		graphics::layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "any_group::create_latex_report_for_permutation_group" << endl;
	}

	{
		string fname;
		string title;
		string author;
		string extra_praeamble;

		fname.assign(label);
		fname.append("_report.tex");
		title.assign("The group $");
		title.append(label_tex);
		title.append("$");

		author.assign("");


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


#if 1
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group "
						"before A->report" << endl;
			}
			A->report(ost, A->f_has_sims, A->Sims,
					A->f_has_strong_generators, A->Strong_gens,
					O,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group "
						"after A->report" << endl;
			}
#endif


			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group "
						"before Subgroup_gens->print_generators_in_latex_individually" << endl;
			}
			Subgroup_gens->print_generators_in_latex_individually(ost);
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group "
						"after Subgroup_gens->print_generators_in_latex_individually" << endl;
			}
			//A_initial->print_base();
			//A_initial->print_info();

#if 1
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group "
						"before Subgroup_sims->report" << endl;
			}
			Subgroup_sims->report(ost,
					label,
					O,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group "
						"after Subgroup_sims->report" << endl;
			}
#endif

			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "any_group::create_latex_report_for_permutation_group done" << endl;
	}
}

void any_group::create_latex_report_for_modified_group(
		graphics::layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "any_group::create_latex_report_for_modified_group" << endl;
	}

	{
		string fname;
		string title;
		string author;
		string extra_praeamble;

		fname.assign(label);
		fname.append("_report.tex");
		title.assign("The group $");
		title.append(label_tex);
		title.append("$");

		author.assign("");


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


#if 0
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group before A->report" << endl;
			}
			A->report(ost, A->f_has_sims, A->Sims,
					A->f_has_strong_generators, A->Strong_gens,
					O,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group after A->report" << endl;
			}
#endif


			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group before Subgroup_gens->print_generators_in_latex_individually" << endl;
			}
			Subgroup_gens->print_generators_in_latex_individually(ost);
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group after Subgroup_gens->print_generators_in_latex_individually" << endl;
			}
			//A_initial->print_base();
			//A_initial->print_info();

#if 0
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group before Subgroup_sims->report" << endl;
			}
			Subgroup_sims->report(ost,
					label,
					O,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group after Subgroup_sims->report" << endl;
			}
#endif

			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "any_group::create_latex_report_for_modified_group done" << endl;
	}
}

groups::strong_generators *any_group::get_strong_generators()
{
	int f_v = FALSE;
	groups::strong_generators *SG;

	if (Subgroup_gens) {
		if (f_v) {
			cout << "any_group::get_strong_generators using Subgroup_gens" << endl;
		}
		SG = Subgroup_gens;
	}
	else if (A->f_has_strong_generators) {
		if (f_v) {
			cout << "any_group::get_strong_generators using A_base->Strong_gens" << endl;
		}
		SG = A->Strong_gens;
	}
	else {
		cout << "any_group::get_strong_generators no generators to export" << endl;
		exit(1);
	}
	return SG;
}

int any_group::is_subgroup_of(any_group *AG_secondary, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = FALSE;


	if (f_v) {
		cout << "any_group::is_subgroup_of" << endl;
	}

	//actions::action *A1;
	//actions::action *A2;

	//A1 = A;
	//A2 = AG_secondary->A;

	groups::strong_generators *Subgroup_gens1;
	groups::strong_generators *Subgroup_gens2;

	Subgroup_gens1 = Subgroup_gens;
	Subgroup_gens2 = AG_secondary->Subgroup_gens;

	groups::sims *S;

	S = Subgroup_gens2->create_sims(verbose_level);


	ret = Subgroup_gens1->test_if_subgroup(S, verbose_level);


	FREE_OBJECT(S);

	if (f_v) {
		cout << "any_group::is_subgroup_of done" << endl;
	}
	return ret;
}

void any_group::set_of_coset_representatives(any_group *AG_secondary,
		data_structures_groups::vector_ge *&coset_reps,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::set_of_coset_representatives" << endl;
	}

	//actions::action *A1;
	//actions::action *A2;

	//A1 = A;
	//A2 = AG_secondary->A;

	groups::strong_generators *Subgroup_gens1;
	groups::strong_generators *Subgroup_gens2;

	Subgroup_gens1 = Subgroup_gens;
	Subgroup_gens2 = AG_secondary->Subgroup_gens;

	groups::sims *S;

	S = Subgroup_gens2->create_sims(verbose_level);

	if (f_v) {
		cout << "any_group::set_of_coset_representatives "
				"before Subgroup_gens1->set_of_coset_representatives" << endl;
	}
	Subgroup_gens1->set_of_coset_representatives(S,
			coset_reps,
			verbose_level);
	if (f_v) {
		cout << "any_group::set_of_coset_representatives "
				"after Subgroup_gens1->set_of_coset_representatives" << endl;
		cout << "any_group::set_of_coset_representatives "
				"number of coset reps = " << coset_reps->len << endl;
	}



	FREE_OBJECT(S);

	if (f_v) {
		cout << "any_group::set_of_coset_representatives done" << endl;
	}
}

void any_group::report_coset_reps(
		data_structures_groups::vector_ge *coset_reps,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::report_coset_reps" << endl;
	}



	string fname;

	fname.assign(label);
	fname.append("_coset_reps.tex");


	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;
		L.head_easy(fp);

		//H->print_all_group_elements_tex(fp);
		//H->print_all_group_elements_with_permutations_tex(fp);

		//Schreier.print_and_list_orbits_tex(fp);

		int i;

		for (i = 0; i < coset_reps->len; i++) {


			fp << "coset " << i << " / " << coset_reps->len << ":" << endl;

			fp << "$$" << endl;
			A->element_print_latex(coset_reps->ith(i), fp);
			fp << "$$" << endl;


		}
		//latex_interface L;


		L.foot(fp);
	}

	if (f_v) {
		cout << "any_group::report_coset_reps done" << endl;
	}
}

void any_group::print_given_elements_tex(
		std::string &label_of_elements,
		int *element_data, int nb_elements,
		int f_with_permutation,
		int f_with_fix_structure,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::print_given_elements_tex" << endl;
	}

	orbiter_kernel_system::file_io Fio;



	int *Elt;
	ring_theory::longinteger_object go;

	Elt = NEW_int(A->elt_size_in_int);


	string fname;

	fname.assign(label_of_elements);
	fname.append("_elements.tex");


	{
		ofstream ost(fname);
		orbiter_kernel_system::latex_interface L;
		int i, ord;

		L.head_easy(ost);

		//H->print_all_group_elements_tex(fp, f_with_permutation, f_override_action, A_special);
		//H->print_all_group_elements_tree(fp);
		//H->print_all_group_elements_with_permutations_tex(fp);

		//Schreier.print_and_list_orbits_tex(fp);

		ost << "Action $" << label_tex << "$:\\\\" << endl;
		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;

		for (i = 0; i < nb_elements; i++) {

			A->make_element(Elt,
					element_data + i * A->make_element_size,
					verbose_level);

			ord = A->element_order(Elt);

			ost << "Element " << setw(5) << i << " / "
					<< nb_elements << " of order " << ord << ":" << endl;

			A->print_one_element_tex(ost, Elt, f_with_permutation);

			if (f_with_fix_structure) {
				int f;

				f = A->count_fixed_points(Elt, 0 /* verbose_level */);

				ost << "$f=" << f << "$\\\\" << endl;
			}
		}


		L.foot(ost);
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	FREE_int(Elt);
	if (f_v) {
		cout << "any_group::print_elements_tex done" << endl;
	}
}


void any_group::process_given_elements(
		std::string &label_of_elements,
		int *element_data, int nb_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::process_given_elements" << endl;
	}

	orbiter_kernel_system::file_io Fio;



	int *Elt;
	ring_theory::longinteger_object go;

	Elt = NEW_int(A->elt_size_in_int);


	string fname;

	fname.assign(label_of_elements);
	fname.append("_processing.tex");


	{
		ofstream ost(fname);
		orbiter_kernel_system::latex_interface L;
		int i, ord;

		L.head_easy(ost);

		//H->print_all_group_elements_tex(fp, f_with_permutation, f_override_action, A_special);
		//H->print_all_group_elements_tree(fp);
		//H->print_all_group_elements_with_permutations_tex(fp);

		//Schreier.print_and_list_orbits_tex(fp);

		ost << "Action $" << label_tex << "$:\\\\" << endl;
		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;

		for (i = 0; i < nb_elements; i++) {

			A->make_element(Elt,
					element_data + i * A->make_element_size,
					verbose_level);

			ord = A->element_order(Elt);

			ost << "Element " << setw(5) << i << " / "
					<< nb_elements << " of order " << ord << ":" << endl;

			A->print_one_element_tex(ost, Elt, FALSE /* f_with_permutation */);

		}


		L.foot(ost);
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	FREE_int(Elt);
	if (f_v) {
		cout << "any_group::process_given_elements done" << endl;
	}
}

void any_group::apply_isomorphism_wedge_product_4to6(
		std::string &label_of_elements,
		int *element_data, int nb_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::apply_isomorphism_wedge_product_4to6" << endl;
	}


	orbiter_kernel_system::file_io Fio;



	int *Elt;
	int *Elt_in;
	int *Elt_out;
	int *Output;
	ring_theory::longinteger_object go;

	Elt = NEW_int(A->elt_size_in_int);

	int elt_size_out;

	elt_size_out = 6 * 6 + 1;

	Elt_out = NEW_int(elt_size_out);

	Output = NEW_int(nb_elements * elt_size_out);


	string fname;

	fname.assign(label_of_elements);
	fname.append("_wedge_4to6.csv");

	if (A->type_G != action_on_wedge_product_t) {
		cout << "any_group::apply_isomorphism_wedge_product_4to6 "
				"the action is not of wedge product type" << endl;
		exit(1);
	}

	{
		int i;



		for (i = 0; i < nb_elements; i++) {

			Elt_in = element_data + i * A->make_element_size;

			A->make_element(Elt, Elt_in, verbose_level);


			induced_actions::action_on_wedge_product *AW = A->G.AW;



			AW->create_induced_matrix(
					Elt, Elt_out, verbose_level);

			if (A->is_semilinear_matrix_group()) {
				Elt_out[6 * 6] = Elt_in[4 * 4];
			}

			Int_vec_copy(Elt_out, Output + i * elt_size_out, elt_size_out);

		}

	}

	Fio.int_matrix_write_csv(fname, Output, nb_elements, elt_size_out);

	if (f_v) {
		cout << "any_group::apply_isomorphism_wedge_product_4to6 "
				"Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
	}


	FREE_int(Elt);
	FREE_int(Elt_out);
	FREE_int(Output);



	if (f_v) {
		cout << "any_group::apply_isomorphism_wedge_product_4to6 done" << endl;
	}
}

void any_group::order_of_products_of_pairs(
		std::string &label_of_elements,
		int *element_data, int nb_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::order_of_products_of_pairs" << endl;
	}


	orbiter_kernel_system::file_io Fio;



	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Order_table;
	ring_theory::longinteger_object go;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	Order_table = NEW_int(nb_elements * nb_elements);


	string fname;

	fname.assign(label_of_elements);
	fname.append("_order_of_products_of_pairs.csv");


	{
		int i, j;



		for (i = 0; i < nb_elements; i++) {

			A->make_element(Elt1, element_data + i * A->make_element_size, verbose_level);

			for (j = 0; j < nb_elements; j++) {

				A->make_element(Elt2, element_data + j * A->make_element_size, verbose_level);

				A->element_mult(Elt1, Elt2, Elt3, 0);

				Order_table[i * nb_elements + j] = A->element_order(Elt3);


			}
		}


	}

	Fio.int_matrix_write_csv(fname, Order_table, nb_elements, nb_elements);

	if (f_v) {
		cout << "any_group::order_of_products_of_pairs "
				"Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
	}


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Order_table);



	if (f_v) {
		cout << "any_group::order_of_products_of_pairs done" << endl;
	}
}


void any_group::conjugate(
		std::string &label_of_elements,
		std::string &conjugate_data,
		int *element_data, int nb_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::conjugate" << endl;
	}


	orbiter_kernel_system::file_io Fio;



	int *S;
	int *Sv;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Output_table;
	ring_theory::longinteger_object go;

	S = NEW_int(A->elt_size_in_int);
	Sv = NEW_int(A->elt_size_in_int);
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	Output_table = NEW_int(nb_elements * A->make_element_size);

	A->make_element_from_string(S, conjugate_data, verbose_level);

	A->element_invert(S, Sv, verbose_level);

	string fname;

	fname.assign(label_of_elements);
	fname.append("_conjugate.csv");


	{
		int i;



		for (i = 0; i < nb_elements; i++) {

			A->make_element(Elt1,
					element_data + i * A->make_element_size,
					verbose_level);


			A->element_mult(Sv, Elt1, Elt2, 0);
			A->element_mult(Elt2, S, Elt3, 0);

			A->code_for_make_element(
					Output_table + i * A->make_element_size, Elt3);


			//A->print_one_element_tex(cout, Elt3, FALSE /* f_with_permutation */);


		}


	}

	Fio.int_matrix_write_csv(fname, Output_table, nb_elements, A->make_element_size);

	if (f_v) {
		cout << "any_group::conjugate "
				"Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
	}


	FREE_int(S);
	FREE_int(Sv);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Output_table);



	if (f_v) {
		cout << "any_group::conjugate done" << endl;
	}
}


void any_group::print_action_on_surface(
		std::string &surface_label,
		std::string &label_of_elements,
		int *element_data, int nb_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::print_action_on_surface" << endl;
	}

	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *SC;

	SC = Get_object_of_cubic_surface(surface_label);

	if (f_v) {
		cout << "any_group::print_action_on_surface "
				"before SC->SOA->print_action_on_surface" << endl;
	}
	SC->SOA->print_action_on_surface(
			label_of_elements,
			element_data, nb_elements,
			verbose_level);
	if (f_v) {
		cout << "any_group::print_action_on_surface "
				"after SC->SOA->print_action_on_surface" << endl;
	}

	if (f_v) {
		cout << "any_group::print_action_on_surface done" << endl;
	}

}

void any_group::element_processing(
		element_processing_description *element_processing_descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::element_processing" << endl;
	}

	int *element_data = NULL;
	int nb_elements;
	int n;


	if (element_processing_descr->f_input) {

		if (f_v) {
			cout << "any_group::element_processing getting input" << endl;
		}
		Get_matrix(element_processing_descr->input_label,
				element_data, nb_elements, n);

	}
	else {
		cout << "please use -input <label> to define input elements" << endl;
		exit(1);

	}


	if (element_processing_descr->f_print) {
		if (f_v) {
			cout << "any_group::element_processing f_print" << endl;
		}

		if (f_v) {
			cout << "any_group::element_processing "
					"before print_given_elements_tex" << endl;
		}

		print_given_elements_tex(
				element_processing_descr->input_label,
				element_data, nb_elements,
				element_processing_descr->f_with_permutation,
				element_processing_descr->f_with_fix_structure,
				verbose_level);

		if (f_v) {
			cout << "any_group::element_processing "
					"after print_given_elements_tex" << endl;
		}

	}
	else if (element_processing_descr->f_apply_isomorphism_wedge_product_4to6) {
		if (f_v) {
			cout << "any_group::element_processing "
					"f_apply_isomorphism_wedge_product_4to6" << endl;
		}

		if (f_v) {
			cout << "any_group::element_processing "
					"before apply_isomorphism_wedge_product_4to6" << endl;
		}
		apply_isomorphism_wedge_product_4to6(
				element_processing_descr->input_label,
				element_data, nb_elements,
				verbose_level);
		if (f_v) {
			cout << "any_group::element_processing "
					"after apply_isomorphism_wedge_product_4to6" << endl;
		}


	}
	else if (element_processing_descr->f_order_of_products_of_pairs) {
		if (f_v) {
			cout << "any_group::element_processing "
					"f_order_of_products_of_pairs" << endl;
		}

		if (f_v) {
			cout << "any_group::element_processing "
					"before order_of_products_of_pairs" << endl;
		}
		order_of_products_of_pairs(
				element_processing_descr->input_label,
				element_data, nb_elements,
				verbose_level);
		if (f_v) {
			cout << "any_group::element_processing "
					"after order_of_products_of_pairs" << endl;
		}


	}
	else if (element_processing_descr->f_conjugate) {
		if (f_v) {
			cout << "any_group::element_processing "
					"f_conjugate" << endl;
		}

		if (f_v) {
			cout << "any_group::element_processing "
					"before conjugate" << endl;
		}
		conjugate(
				element_processing_descr->input_label,
				element_processing_descr->conjugate_data,
				element_data, nb_elements,
				verbose_level);
		if (f_v) {
			cout << "any_group::element_processing "
					"after conjugate" << endl;
		}


	}

	else if (element_processing_descr->f_print_action_on_surface) {
		if (f_v) {
			cout << "any_group::element_processing "
					"f_print_action_on_surface" << endl;
		}


		if (f_v) {
			cout << "any_group::element_processing "
					"before print_action_on_surface" << endl;
		}
		print_action_on_surface(
				element_processing_descr->print_action_on_surface_label,
				element_processing_descr->input_label,
				element_data, nb_elements,
				verbose_level);
		if (f_v) {
			cout << "any_group::element_processing "
					"after print_action_on_surface" << endl;
		}


	}



	if (f_v) {
		cout << "any_group::element_processing done" << endl;
	}
}




}}}

