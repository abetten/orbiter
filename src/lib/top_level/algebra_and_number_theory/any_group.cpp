/*
 * any_group.cpp
 *
 *  Created on: Sep 26, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;
using namespace orbiter::foundations;

namespace orbiter {
namespace top_level {


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


void any_group::init_linear_group(linear_group *LG, int verbose_level)
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
		cout << "any_group::init_linear_group !LG->f_has_strong_generators" << endl;
		exit(1);
	}
	Subgroup_gens = LG->Strong_gens;

	if (f_v) {
		cout << "any_group::init_linear_group before Subgroup_gens->create_sims" << endl;
	}
	Subgroup_sims = Subgroup_gens->create_sims(0/*verbose_level*/);
	if (f_v) {
		cout << "any_group::init_linear_group after Subgroup_gens->create_sims" << endl;
	}

	if (f_v) {
		cout << "any_group::init_linear_group done" << endl;
	}
}

void any_group::init_permutation_group(permutation_group_create *PGC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::init_linear_group" << endl;
	}

	f_permutation_group = TRUE;
	any_group::PGC = PGC;

	A_base = PGC->A_initial;
	A = PGC->A2;

	label.assign(PGC->label);
	label_tex.assign(PGC->label_tex);

	if (!PGC->f_has_strong_generators) {
		cout << "any_group::init_linear_group !PGC->f_has_strong_generators" << endl;
		exit(1);
	}
	Subgroup_gens = PGC->Strong_gens;

	if (f_v) {
		cout << "any_group::init_permutation_group before Subgroup_gens->create_sims_in_different_action" << endl;
	}
	Subgroup_sims = Subgroup_gens->create_sims_in_different_action(
			A_base, 0 /*verbose_level*/);
	//Subgroup_sims = Subgroup_gens->create_sims(verbose_level);
	if (f_v) {
		cout << "any_group::init_permutation_group after Subgroup_gens->create_sims_in_different_action" << endl;
	}

	if (f_v) {
		cout << "any_group::init_linear_group done" << endl;
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
		cout << "any_group::init_linear_group !PGC->f_has_strong_generators" << endl;
		exit(1);
	}
	Subgroup_gens = MGC->Strong_gens;

	label.assign(A->label);
	label_tex.assign(A->label_tex);

	if (f_v) {
		cout << "any_group::init_modified_group done" << endl;
	}
}



void any_group::create_latex_report(
		layered_graph_draw_options *O,
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

void any_group::do_export_orbiter(action *A2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_export_orbiter" << endl;
	}

	string fname;
	file_io Fio;

	if (f_v) {
		cout << "any_group::do_export_orbiter label=" << label << endl;
	}
	fname.assign(label);
	fname.append(".makefile");
	{
		ofstream fp(fname);

		if (Subgroup_gens) {
			if (f_v) {
				cout << "any_group::do_export_orbiter using Subgroup_gens" << endl;
			}
			Subgroup_gens->export_to_orbiter_as_bsgs(A2, fname, label, label_tex, verbose_level);
		}
		else if (A->f_has_strong_generators) {
			if (f_v) {
				cout << "any_group::do_export_orbiter using A_base->Strong_gens" << endl;
			}
			A_base->export_to_orbiter_as_bsgs(fname, label, label_tex, A_base->Strong_gens, verbose_level);
		}
		else {
			cout << "any_group::do_export_orbiter no generators to export" << endl;
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
	file_io Fio;

	fname.assign(label);
	fname.append("_generators.gap");
	{
		ofstream fp(fname);
		LG->Strong_gens->print_generators_gap(fp);
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
	file_io Fio;

	fname.assign(label);
	fname.append("_generators.magma");
	{
		ofstream fp(fname);
		LG->Strong_gens->export_magma(LG->A_linear, fp, verbose_level);
	}
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "any_group::do_export_magma done" << endl;
	}
}

void any_group::do_canonical_image_GAP(std::string &input_set_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_canonical_image_GAP" << endl;
	}

	string fname;
	file_io Fio;

	fname.assign(label);
	fname.append("_canonical_image.gap");
	{
		ofstream ost(fname);
		LG->Strong_gens->canonical_image_GAP(input_set_text, ost);
	}
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "any_group::do_canonical_image_GAP done" << endl;
	}
}

void any_group::create_group_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::create_group_table" << endl;
	}
	int goi;



	goi = Subgroup_sims->group_order_lint();

	if (f_v) {
			cout << "group order = " << goi << endl;
	}


	string fname;
	int *Table;
	long int n;
	file_io Fio;

	Subgroup_sims->create_group_table(Table, n, verbose_level);

	if (n != goi) {
		cout << "any_group::create_group_table n != goi" << endl;
		exit(1);
	}

	fname.assign(label);
	fname.append("_group_table.csv");

	cout << "The group table is:" << endl;
	Orbiter->Int_vec.matrix_print(Table, n, n, 2);

	Fio.int_matrix_write_csv(fname, Table, n, n);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	FREE_int(Table);

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
	A->normalizer_using_MAGMA(fname_magma_prefix,
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

	algebra_global_with_action Algebra;
	sims *S;

	S = LG->Strong_gens->create_sims(verbose_level);

	if (f_v) {
		cout << "any_group::centralizer "
				"before Algebra.centralizer_of_element" << endl;
	}
	Algebra.centralizer_of_element(
			LG->A2, S,
			element_description_text,
			element_label, verbose_level);
	if (f_v) {
		cout << "any_group::centralizer "
				"after Algebra.centralizer_of_element" << endl;
	}

	FREE_OBJECT(S);

	if (f_v) {
		cout << "any_group::centralizer done" << endl;
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

	algebra_global_with_action Algebra;
	sims *S;

	S = LG->Strong_gens->create_sims(verbose_level);

	if (f_v) {
		cout << "any_group::normalizer_of_cyclic_subgroup "
				"before Algebra.normalizer_of_cyclic_subgroup" << endl;
	}
	Algebra.normalizer_of_cyclic_subgroup(
			LG->A2, S,
			element_description_text,
			element_label, verbose_level);
	if (f_v) {
		cout << "any_group::normalizer_of_cyclic_subgroup "
				"after Algebra.normalizer_of_cyclic_subgroup" << endl;
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
		cout << "any_group::do_find_subgroups done" << endl;
	}
}


void any_group::print_elements(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::print_elements" << endl;
	}
	sims *H;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	longinteger_object go;
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

void any_group::print_elements_tex(
		int f_order_of_products, std::string &Elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::print_elements_tex" << endl;
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

		if (f_order_of_products) {
			int *elements;
			int nb_elements;
			int *order_table;
			int i;

			Orbiter->Int_vec.scan(Elements, elements, nb_elements);

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
		cout << "any_group::print_elements_tex done" << endl;
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

void any_group::multiply_elements_csv(std::string &fname1,
		std::string &fname2, std::string &fname3,
		int f_column_major_ordering, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::multiply_elements_csv" << endl;
	}

	vector_ge V1, V2, V3;
	int n1, n2, n3;
	int i, j, k;

	V1.read_column_csv(fname1, A, 1 /* col_idx */, verbose_level);
	n1 = V1.len;

	V2.read_column_csv(fname2, A, 1 /* col_idx */, verbose_level);
	n2 = V2.len;

	n3 = n1 * n2;

	if (f_v) {
		cout << "any_group::multiply_elements_csv n1=" << V1.len << " n2=" << V2.len << " n3=" << n3 << endl;
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

	vector_ge V1;
	int n1;
	int i, j;
	long int *set;
	long int *set_image;
	int *set_image_int;
	int sz;
	int *Rk;
	combinatorics_domain Combi;


	Orbiter->Lint_vec.scan(set_text, set, sz);

	V1.read_column_csv(fname1, A, 1 /* col_idx */, verbose_level);
	n1 = V1.len;

	set_image = NEW_lint(sz);
	set_image_int = NEW_int(sz);
	Rk = NEW_int(n1);

	if (f_v) {
		cout << "any_group::apply_elements_to_set_csv n1=" << V1.len << endl;
	}

	for (i = 0; i < n1; i++) {
		A->map_a_set_and_reorder(set, set_image,
				sz, V1.ith(i), 0 /* verbose_level */);

		for (j = 0; j < sz; j++) {
			set_image_int[j] = set_image[j];
		}
		Rk[i] = Combi.rank_k_subset(set_image_int, A->degree, sz);

		cout << i << " : ";
		Orbiter->Lint_vec.print(cout, set_image, sz);
		cout << i << " : ";
		cout << Rk[i];
		cout << endl;

	}


	cout << "Image sets by rank: ";
	Orbiter->Int_vec.print_fully(cout, Rk, n1);
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
	algebra_global_with_action Algebra;

	Algebra.element_rank(LG,
			A,
			elt_data, verbose_level);

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
	algebra_global_with_action Algebra;

	Algebra.element_unrank(LG,
			A,
			rank_string, verbose_level);

	if (f_v) {
		cout << "any_group::element_unrank done" << endl;
	}
}

void any_group::conjugacy_class_of(std::string &elt_data, int verbose_level)
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

	longinteger_object a, b;

#if 1
	cout << "creating element " << elt_data << endl;

	A->make_element_from_string(Elt, elt_data, 0);

	Subgroup_sims->element_rank(a, Elt);

	a.assign_to(b);


#else


	a.create_from_base_10_string(rank_string.c_str(), 0 /*verbose_level*/);

	cout << "Creating element of rank " << a << endl;

	a.assign_to(b);

	H->element_unrank(a, Elt);

#endif

	cout << "Element :" << endl;
	A->element_print(Elt, cout);
	cout << endl;


	action *A_conj;

	A_conj = A->create_induced_action_by_conjugation(
			Subgroup_sims /*Base_group*/, FALSE /* f_ownership */,
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

	Orb.init(A, A_conj,
			set, 1 /* sz */, LG->Strong_gens->gens, verbose_level);
	cout << "Found an orbit of size " << Orb.used_length << endl;

	std::vector<long int> Orbit;

	cout << "before Orb.get_orbit_of_points" << endl;
	Orb.get_orbit_of_points(Orbit, verbose_level);
	cout << "Found an orbit of size " << Orbit.size() << endl;

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
	fname.append(elt_data);
	fname.append(".csv");

	Fio.int_matrix_write_csv(fname, M, Orbit.size(), A->make_element_size);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

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
	file_io Fio;
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
	action *A_on_sets;
	int set_size;

	set_size = number_of_columns;

	if (f_v) {
		cout << "creating action on sets:" << endl;
	}
	A_on_sets = A->create_induced_action_on_sets(m /* nb_sets */,
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
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::orbits_on_set_from_file" << endl;
	}

	if (f_v) {
		cout << "computing orbit of set from file "
			<< fname_csv << ":" << endl;
	}
	file_io Fio;
	long int *the_set;
	int set_sz;

	Fio.read_set_from_file(fname_csv,
			the_set, set_sz, 0 /*verbose_level*/);
	if (f_v) {
		cout << "read a set of size " << set_sz << endl;
	}


	string label_set;
	string_tools ST;

	label_set.assign(fname_csv);
	ST.chop_off_extension(label_set);

	algebra_global_with_action Algebra;
	long int *Table;
	int size;

	Algebra.orbits_on_set_from_file(
			the_set, set_sz,
			A, A,
			LG->Strong_gens->gens,
			label_set,
			LG->label,
			Table, size,
			verbose_level);

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
	schreier *Sch;
	Sch = NEW_OBJECT(schreier);

	cout << "computing orbit of point " << point_idx << ":" << endl;

	//A->all_point_orbits(*Sch, verbose_level);

	Sch->init(A, verbose_level - 2);
	if (!A->f_has_strong_generators) {
		cout << "any_group::orbit_of !f_has_strong_generators" << endl;
		exit(1);
		}
	Sch->init_generators(*LG->Strong_gens->gens /* *strong_generators */, verbose_level - 2);
	Sch->initialize_tables();
	Sch->compute_point_orbit(point_idx, verbose_level);


	cout << "computing orbit of point done." << endl;

	string fname_tree_mask;
	char str[1000];

	fname_tree_mask.assign(LG->label);
	sprintf(str, "_orbit_of_point_%d.layered_graph", point_idx);
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

	fname_tree_mask.assign(label);
	fname_tree_mask.append("_%d_shallow.layered_graph");

	shallow_tree->export_tree_as_layered_graph(0 /* orbit_no */,
			fname_tree_mask,
			verbose_level - 1);
	if (f_v) {
		cout << "any_group::orbit_of done" << endl;
	}
}

void any_group::orbits_on_points(orbits_on_something *&Orb, int verbose_level)
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
		cout << "any_group::orbits_on_points before Algebra.orbits_on_points" << endl;
	}
	Algebra.orbits_on_points(
			A,
			Subgroup_gens,
			f_load_save,
			prefix,
			Orb,
			verbose_level);
	if (f_v) {
		cout << "any_group::orbits_on_points after Algebra.orbits_on_points" << endl;
	}


	if (f_v) {
		cout << "any_group::orbits_on_points done" << endl;
	}
}

void any_group::orbits_on_subsets(poset_classification_control *Control,
		poset_classification *&PC,
		int subset_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::orbits_on_subsets subset_size=" << subset_size << endl;
	}
	//poset_classification *PC;
	poset_with_group_action *Poset;

	Poset = NEW_OBJECT(poset_with_group_action);

	if (f_v) {
		cout << "any_group::orbits_on_subsets control=" << endl;
		Control->print();
	}
	if (f_v) {
		cout << "any_group::orbits_on_subsets label=" << label << endl;
	}
	if (f_v) {
		cout << "any_group::orbits_on_subsets A_base=" << endl;
		A_base->print_info();
	}
	if (f_v) {
		cout << "any_group::orbits_on_subsets A=" << endl;
		A->print_info();
	}
	if (f_v) {
		cout << "any_group::orbits_on_subsets group order" << endl;

		longinteger_object go;

		Subgroup_gens->group_order(go);

		cout << go << endl;
	}


	if (f_v) {
		cout << "any_group::orbits_on_subsets "
				"before Poset->init_subset_lattice" << endl;
	}
	Poset->init_subset_lattice(A_base, A,
			Subgroup_gens,
			verbose_level);

	if (f_v) {
		cout << "any_group::orbits_on_subsets "
				"before Poset->orbits_on_k_sets_compute" << endl;
	}
	PC = Poset->orbits_on_k_sets_compute(
			Control,
			subset_size,
			verbose_level);
	if (f_v) {
		cout << "any_group::orbits_on_subsets "
				"after Poset->orbits_on_k_sets_compute" << endl;
	}

	if (f_v) {
		cout << "any_group::orbits_on_subsets "
				"before orbits_on_poset_post_processing" << endl;
	}
	orbits_on_poset_post_processing(
			PC, subset_size,
			verbose_level);
	if (f_v) {
		cout << "any_group::orbits_on_subsets "
				"after orbits_on_poset_post_processing" << endl;
	}


	if (f_v) {
		cout << "any_group::orbits_on_subsets done" << endl;
	}
}


void any_group::orbits_on_poset_post_processing(
		poset_classification *PC,
		int depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::orbits_on_poset_post_processing" << endl;
	}



#if 0
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
#endif



	if (f_v) {
		cout << "any_group::orbits_on_poset_post_processing done" << endl;
	}
}









void any_group::do_conjugacy_class_of_element(
		std::string &elt_label, std::string &elt_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_conjugacy_class_of_element" << endl;
	}


	int *data, sz;

	Orbiter->Int_vec.scan(elt_text, data, sz);

	if (f_v) {
		cout << "computing conjugacy class of ";
		Orbiter->Int_vec.print(cout, data, sz);
		cout << endl;
	}


	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);
	A->make_element(Elt, data, 0 /* verbose_level */);

	if (!A->f_has_sims) {
		if (f_v) {
			cout << "any_group::do_conjugacy_class_of_element "
				"Group does not have a sims object" << endl;
		}
		//exit(1);

		{
			sims *S;

			S = LG->Strong_gens->create_sims(verbose_level);

			if (f_v) {
				cout << "any_group::do_conjugacy_class_of_element before init_sims" << endl;
			}
			A->init_sims_only(S, 0/*verbose_level - 1*/);
			if (f_v) {
				cout << "any_group::do_conjugacy_class_of_element after init_sims" << endl;
			}
		}

	}
	sims *S;

	S = A->Sims;

	long int the_set[1];
	int set_size = 1;

	the_set[0] = S->element_rank_lint(Elt);

	if (f_v) {
		cout << "computing conjugacy class of " << endl;
		A->element_print_latex(Elt, cout);
		cout << "which is the set ";
		Orbiter->Lint_vec.print(cout, the_set, set_size);
		cout << endl;
	}


	action A_conj;
	if (f_v) {
		cout << "any_group::do_conjugacy_class_of_element "
				"before A_conj.induced_action_by_conjugation" << endl;
	}
	A_conj.induced_action_by_conjugation(S, S,
			FALSE /* f_ownership */, FALSE /* f_basis */,
			verbose_level);
	if (f_v) {
		cout << "any_group::do_conjugacy_class_of_element "
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
			A, &A_conj,
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
		for (int j = 0; j < A->make_element_size; j++) {
			ost << ",C" << j;
		}
		ost << endl;

		for (i = 0; i < orbit_length; i++) {

			ost << i;
			S->element_unrank_lint(Table[i], Elt);

			for (int j = 0; j < A->make_element_size; j++) {
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
		cout << "any_group::do_conjugacy_class_of_element done" << endl;
	}
}


void any_group::do_orbits_on_group_elements_under_conjugation(
		std::string &fname_group_elements_coded,
		std::string &fname_transporter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_orbits_on_group_elements_under_conjugation" << endl;
	}




	if (!A->f_has_sims) {
		if (f_v) {
			cout << "any_group::do_orbits_on_group_elements_under_conjugation "
				"Group does not have a sims object" << endl;
		}
		//exit(1);

		{
			//sims *S;

			A->create_sims(verbose_level);

#if 0
			if (f_v) {
				cout << "any_group::do_orbits_on_group_elements_under_conjugation before init_sims" << endl;
			}
			A2->init_sims_only(S, 0/*verbose_level - 1*/);
			if (f_v) {
				cout << "any_group::do_orbits_on_group_elements_under_conjugation after init_sims" << endl;
			}
#endif
		}

	}





	sims *S;

	S = A->Sims;

	if (f_v) {
		cout << "the group has order " << S->group_order_lint() << endl;
	}
	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);

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
		cout << "any_group::do_orbits_on_group_elements_under_conjugation "
				"before Algebra.orbits_under_conjugation" << endl;
	}
	Algebra.orbits_under_conjugation(
			the_ranks, m, S,
			LG->Strong_gens,
			Transporter,
			verbose_level);
	if (f_v) {
		cout << "any_group::do_orbits_on_group_elements_under_conjugation "
				"after Algebra.orbits_under_conjugation" << endl;
	}




	FREE_int(Elt);

	if (f_v) {
		cout << "any_group::do_orbits_on_group_elements_under_conjugation done" << endl;
	}
}

void any_group::create_latex_report_for_permutation_group(
		layered_graph_draw_options *O,
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

		fname.assign(label);
		fname.append("_report.tex");
		title.assign("The group $");
		title.append(label_tex);
		title.append("$");

		author.assign("");


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title.c_str(), author.c_str(),
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


#if 0
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group before A->report" << endl;
			}
			A->report(ost, A->f_has_sims, A->Sims,
					A->f_has_strong_generators, A->Strong_gens,
					O,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group after A->report" << endl;
			}
#endif


			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group before Subgroup_gens->print_generators_in_latex_individually" << endl;
			}
			Subgroup_gens->print_generators_in_latex_individually(ost);
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group after Subgroup_gens->print_generators_in_latex_individually" << endl;
			}
			//A_initial->print_base();
			//A_initial->print_info();

#if 0
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group before Subgroup_sims->report" << endl;
			}
			Subgroup_sims->report(ost,
					label,
					O,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group after Subgroup_sims->report" << endl;
			}
#endif

			L.foot(ost);

		}
		file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "any_group::create_latex_report_for_permutation_group done" << endl;
	}
}

void any_group::create_latex_report_for_modified_group(
		layered_graph_draw_options *O,
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

		fname.assign(label);
		fname.append("_report.tex");
		title.assign("The group $");
		title.append(label_tex);
		title.append("$");

		author.assign("");


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title.c_str(), author.c_str(),
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


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
		file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "any_group::create_latex_report_for_modified_group done" << endl;
	}
}









}}

