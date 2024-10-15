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
	f_linear_group = false;
	LG = NULL;

	f_permutation_group = false;
	PGC = NULL;

	f_modified_group = false;
	MGC = NULL;

	A_base = NULL;
	A = NULL;

	//std::string label;
	//std::string label_tex;

	Subgroup_gens = NULL;
	Subgroup_sims = NULL;

	Any_group_linear = NULL;

	f_has_subgroup_lattice = false;
	Subgroup_lattice = NULL;

	f_has_class_data = false;
	class_data = NULL;

}

any_group::~any_group()
{
	if (Any_group_linear) {
		FREE_OBJECT(Any_group_linear);
	}
	if (f_has_subgroup_lattice) {
		FREE_OBJECT(Subgroup_lattice);
	}
	if (f_has_class_data) {
		FREE_OBJECT(class_data);
	}
}


void any_group::init_basic(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::init_basic" << endl;
	}

	Any_group_linear = NEW_OBJECT(any_group_linear);
	Any_group_linear->init(this, verbose_level);

	if (f_v) {
		cout << "any_group::init_basic done" << endl;
	}
}


void any_group::init_linear_group(
		group_constructions::linear_group *LG, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::init_linear_group" << endl;
	}


	if (f_v) {
		cout << "any_group::init_linear_group before init_basic" << endl;
	}
	init_basic(verbose_level);
	if (f_v) {
		cout << "any_group::init_linear_group after init_basic" << endl;
	}

	f_linear_group = true;
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
	Subgroup_sims = Subgroup_gens->create_sims(verbose_level - 2);
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

void any_group::init_permutation_group(
		group_constructions::permutation_group_create *PGC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::init_permutation_group" << endl;
	}

	if (f_v) {
		cout << "any_group::init_permutation_group before init_basic" << endl;
	}
	init_basic(verbose_level);
	if (f_v) {
		cout << "any_group::init_permutation_group after init_basic" << endl;
	}

	f_permutation_group = true;
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

void any_group::init_modified_group(
		modified_group_create *MGC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::init_modified_group" << endl;
	}

	if (f_v) {
		cout << "any_group::init_modified_group before init_basic" << endl;
	}
	init_basic(verbose_level);
	if (f_v) {
		cout << "any_group::init_modified_group after init_basic" << endl;
	}

	f_modified_group = true;
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



#if 0
void any_group::create_latex_report(
		graphics::layered_graph_draw_options *O,
		int f_sylow, int f_group_table, //int f_classes,
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
		LG->create_latex_report(
				O,
				f_sylow, f_group_table, //f_classes,
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
#endif




void any_group::create_latex_report(
		graphics::layered_graph_draw_options *LG_Draw_options,
		int f_sylow, int f_group_table, //int f_classes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "any_group::create_latex_report" << endl;
	}
	if (f_v) {
		cout << "any_group::create_latex_report "
				"label = " << label << endl;
		cout << "any_group::create_latex_report "
				"label_tex = " << label_tex << endl;
	}

	{
		string fname;
		string title;
		string author, extra_praeamble;

		fname = label + "_report.tex";
		title = "The group $" + label_tex + "$";
		author = "";


		{
			ofstream ost(fname);
			l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "any_group::create_latex_report before report" << endl;
			}
#if 0
			report(
					ost,
					f_sylow, f_group_table,
					//f_classes,
					LG_Draw_options,
					verbose_level);
#endif

			actions::action_global Action_global;


			if (f_v) {
				cout << "any_group::create_latex_report "
						"before Action_global.report" << endl;
			}
			Action_global.report(
					ost,
					label,
					label_tex,
					A,
					Subgroup_gens,
					f_sylow, f_group_table,
					LG_Draw_options,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report "
						"after Action_global.report" << endl;
			}

			ost << endl;
			ost << "\\subsection*{Strong Generators}" << endl;
			ost << endl;

			if (f_v) {
				cout << "any_group::create_latex_report "
						"before Subgroup_gens->print_generators_tex" << endl;
			}

			cout << "Strong generators:\\\\" << endl;
			Subgroup_gens->print_generators_tex(ost);

			if (f_v) {
				cout << "any_group::create_latex_report "
						"after Subgroup_gens->print_generators_tex" << endl;
			}



#if 0
			actions::action *A_base;
			actions::action *A;

			std::string label;
			std::string label_tex;

			groups::strong_generators *Subgroup_gens;
			groups::sims *Subgroup_sims;
#endif

			if (f_v) {
				cout << "any_group::create_latex_report after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "any_group::create_latex_report done" << endl;
	}
}


void any_group::export_group_table(
		int verbose_level)
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


	fname = label + "_group_table.csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Table, n, n);
	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_int(Table);

	if (f_v) {
		cout << "any_group::export_group_table done" << endl;
	}


}


void any_group::do_export_orbiter(
		actions::action *A2, int verbose_level)
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
	fname = label + ".makefile";
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



void any_group::do_export_gap(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_export_gap" << endl;
	}

	string fname;
	orbiter_kernel_system::file_io Fio;

	fname = label + "_generators.gap";
	{
		ofstream ost(fname);

#if 0
		if (Subgroup_gens) {
			if (f_v) {
				cout << "any_group::do_export_gap "
						"using Subgroup_gens" << endl;
			}
			Subgroup_gens->print_generators_gap(ost);
		}
		else if (A->f_has_strong_generators) {
			if (f_v) {
				cout << "any_group::do_export_gap "
						"using A_base->Strong_gens" << endl;
			}
			A->Strong_gens->print_generators_gap_in_different_action(ost, A);
		}
		else {
			cout << "any_group::do_export_gap "
					"no generators to export" << endl;
			exit(1);
		}
#endif

		ost << "LoadPackage(\"fining\");" << endl;


		groups::strong_generators *SG;

		SG = get_strong_generators();

		if (A->is_matrix_group()) {
			if (f_v) {
				cout << "any_group::do_export_gap "
						"before SG->export_fining" << endl;
			}
			SG->export_fining(A, ost, verbose_level);
			if (f_v) {
				cout << "any_group::do_export_gap "
						"after SG->export_fining" << endl;
			}
		}

	}
	if (f_v) {
		cout << "any_group::do_export_gap "
			"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "any_group::do_export_gap done" << endl;
	}
}

void any_group::do_export_magma(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_export_magma" << endl;
	}

	string fname;
	orbiter_kernel_system::file_io Fio;

	fname = label + "_generators.magma";
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


void any_group::do_canonical_image_GAP(
		std::string &input_set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_canonical_image_GAP" << endl;
	}

	string fname;
	orbiter_kernel_system::file_io Fio;

	fname = label + "_canonical_image.gap";
	{
		ofstream ost(fname);
		groups::strong_generators *SG;

		SG = get_strong_generators();
		SG->canonical_image_GAP(input_set_text, ost, verbose_level);
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "any_group::do_canonical_image_GAP done" << endl;
	}
}

void any_group::do_canonical_image_orbiter(
		std::string &input_set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::do_canonical_image_orbiter" << endl;
	}

	{
		groups::strong_generators *SG;

		SG = get_strong_generators();
		SG->canonical_image_orbiter(input_set_text, verbose_level);
	}


	if (f_v) {
		cout << "any_group::do_canonical_image_orbiter done" << endl;
	}
}


void any_group::create_group_table(
		int *&Table, long int &n, int verbose_level)
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

void any_group::normalizer(
		int verbose_level)
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



	fname_magma_prefix = label + "_normalizer";


	H = Subgroup_gens->create_sims(verbose_level);

	if (f_linear_group) {
		G = LG->initial_strong_gens->create_sims(verbose_level);
	}
	else {
		G = PGC->A_initial->Strong_gens->create_sims(verbose_level);
	}

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	//H = LG->Strong_gens->create_sims(verbose_level);


	interfaces::magma_interface M;


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

		fname = fname_magma_prefix + ".tex";
		title = "Normalizer of subgroup " + label_tex;


		{
			ofstream ost(fname);
			l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
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

	interfaces::magma_interface Magma;
	groups::sims *S;
	groups::strong_generators *SG;

	SG = get_strong_generators();

	S = SG->create_sims(verbose_level);

	if (f_v) {
		cout << "any_group::centralizer "
				"before Magma.centralizer_of_element" << endl;
	}
	Magma.centralizer_of_element(
			A, S,
			element_description_text,
			element_label, verbose_level);
	if (f_v) {
		cout << "any_group::centralizer "
				"after Magma.centralizer_of_element" << endl;
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

	interfaces::magma_interface M;
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

	interfaces::magma_interface M;
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

	fname = A->label + "_report.tex";



	title = "Subgroups of order $" + std::to_string(order_of_subgroup) + "$ in $" + A->label_tex + "$";


	{
		ofstream fp(fname);
		l1_interfaces::latex_interface L;
		//latex_head_easy(fp);
		L.head(fp,
			false /* f_book */, true /* f_title */,
			title, author,
			false /*f_toc*/, false /* f_landscape*/, false /* f_12pt*/,
			true /*f_enlarged_page*/, true /* f_pagenumbers*/,
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


void any_group::print_elements(
		int verbose_level)
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
		A->Group_element->element_print(Elt, cout);
		cout << endl;
		A->Group_element->element_print_as_permutation(Elt, cout);
		cout << endl;



	}
	FREE_int(Elt);
	if (f_v) {
		cout << "any_group::print_elements done" << endl;
	}
}

void any_group::print_elements_tex(
		int f_with_permutation,
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

	fname = label + "_elements.tex";


	{
		ofstream fp(fname);
		l1_interfaces::latex_interface L;
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


	fname = label + "_elements_tree.txt";


	{
		ofstream fp(fname);
		l1_interfaces::latex_interface L;
		//L.head_easy(fp);

		//H->print_all_group_elements_tex(fp);
		H->print_all_group_elements_tree(fp);
		//H->print_all_group_elements_with_permutations_tex(fp);

		//Schreier.print_and_list_orbits_tex(fp);

		fp << -1 << endl;

		//L.foot(fp);
	}
	cout << "Written file " << fname
			<< " of size " << Fio.file_size(fname) << endl;


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

	fname = label + "_elements.tex";


	{
		ofstream fp(fname);
		l1_interfaces::latex_interface L;
		L.head_easy(fp);

		int f_override_action = false;

		H->print_all_group_elements_tex(fp, false, f_override_action, NULL);
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

				A->Group_element->element_mult(Elt1, Elt2, Elt3, 0);

				order_table[i * nb_elements + j] = A->Group_element->element_order(Elt3);

			}
		}
		FREE_int(Elt1);
		FREE_int(Elt2);
		FREE_int(Elt3);

		//latex_interface L;

		fp << "$$" << endl;
		L.print_integer_matrix_with_labels(fp, order_table,
				nb_elements, nb_elements, elements, elements, true /* f_tex */);
		fp << "$$" << endl;

		L.foot(fp);
	}

	FREE_int(Elt);
	if (f_v) {
		cout << "any_group::order_of_products_of_elements_by_rank done" << endl;
	}
}

void any_group::save_elements_csv(
		std::string &fname, int verbose_level)
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

void any_group::export_inversion_graphs(
		std::string &fname, int verbose_level)
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


#if 0
void any_group::multiply_elements_csv(
		std::string &fname1,
		std::string &fname2,
		std::string &fname3,
		int f_column_major_ordering,
		int verbose_level)
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
				A->Group_element->mult(V1.ith(i), V2.ith(j), V3.ith(k));
			}
		}
	}
	else {
		k = 0;
		for (i = 0; i < n1; i++) {
			for (j = 0; j < n2; j++, k++) {
				A->Group_element->mult(V1.ith(i), V2.ith(j), V3.ith(k));
			}
		}

	}

	//Subgroup_sims->all_elements_save_csv(fname, verbose_level);
	V3.save_csv(fname3, verbose_level);

	if (f_v) {
		cout << "any_group::multiply_elements_csv done" << endl;
	}
}

void any_group::apply_elements_to_set_csv(
		std::string &fname1,
		std::string &fname2,
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
		A->Group_element->map_a_set_and_reorder(set, set_image,
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


	if (!Combi.Permutations->is_permutation(Rk, n1)) {
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
#endif

void any_group::random_element(
		std::string &elt_label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::random_element" << endl;
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

	int *Elt;
	int *data;

	Elt = NEW_int(A1->elt_size_in_int);
	data = NEW_int(A1->make_element_size);


	H->random_element(Elt, 0 /* verbose_level */);


	A1->Group_element->code_for_make_element(data, Elt);

	if (f_v) {

		cout << "Element :" << endl;
		A1->Group_element->element_print(Elt, cout);
		cout << endl;

		cout << "coded: ";
		Int_vec_print(cout, data, A1->make_element_size);
		cout << endl;

		cout << "Element as permutation:" << endl;


		A1->Group_element->element_print_as_permutation(Elt, cout);
		cout << endl;

		int *perm;

		perm = NEW_int(A1->degree);

		A1->Group_element->compute_permutation(
				Elt,
				perm, 0 /*verbose_level*/);
		cout << "In list notation:" << endl;
		Int_vec_print(cout, perm, A1->degree);
		cout << endl;

	}

	ring_theory::longinteger_object a;
	H->element_rank(a, Elt);

	if (f_v) {
		cout << "The rank of the element is " << a << endl;
	}


	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = elt_label + ".csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, data, 1, A->make_element_size);

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	// read vector from file:

	data_structures::vector_builder_description *Descr;
	user_interface::symbol_definition *Symbol_definition;

	Descr = NEW_OBJECT(data_structures::vector_builder_description);

	Descr->f_file = true;
	Descr->file_name.assign(fname);

	Symbol_definition = NEW_OBJECT(user_interface::symbol_definition);

	user_interface::interface_symbol_table *Interface_symbol_table;

	Interface_symbol_table = NEW_OBJECT(user_interface::interface_symbol_table);
	Interface_symbol_table->init(user_interface::The_Orbiter_top_level_session, verbose_level);


	Symbol_definition->Sym = Interface_symbol_table;

	Symbol_definition->definition_of_vector(
			elt_label, Descr, verbose_level);



	FREE_int(Elt);
	FREE_OBJECT(H);



	if (f_v) {
		cout << "any_group::random_element done" << endl;
	}
}


void any_group::element_rank(
		std::string &elt_data, int verbose_level)
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
	A1->Group_element->make_element_from_string(Elt, elt_data, 0);

	if (f_v) {
		cout << "Element :" << endl;
		A1->Group_element->element_print(Elt, cout);
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

void any_group::element_unrank(
		std::string &rank_string, int verbose_level)
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
		A1->Group_element->element_print(Elt, cout);
		cout << endl;
	}


	FREE_int(Elt);
	FREE_OBJECT(H);

	if (f_v) {
		cout << "any_group::element_unrank done" << endl;
	}
}



void any_group::conjugacy_class_of(
		std::string &label_of_class,
		std::string &elt_data,
		int verbose_level)
// uses orbits_schreier::orbit_of_sets
// needs Subgroup_sims to set up action by conjugation
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

	A->Group_element->make_element_from_string(Elt, elt_data, 0);

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
		A->Group_element->element_print(Elt, cout);
		cout << endl;
	}


	actions::action *A_conj;

	if (f_v) {
		cout << "any_group::conjugacy_class_of "
				"before A->Induced_action->create_induced_action_by_conjugation" << endl;
	}
	A_conj = A->Induced_action->create_induced_action_by_conjugation(
			Subgroup_sims /*Base_group*/, false /* f_ownership */,
			false /* f_basis */, NULL /* old_G */,
			verbose_level);
	if (f_v) {
		cout << "any_group::conjugacy_class_of "
				"after A->Induced_action->create_induced_action_by_conjugation" << endl;
	}


	if (f_v) {
		cout << "any_group::conjugacy_class_of "
				"created action A_conj of degree " << A_conj->degree << endl;
	}



	orbits_schreier::orbit_of_sets Orb;
	long int set[1];
	orbiter_kernel_system::file_io Fio;

	set[0] = b.as_lint();

	if (f_v) {
		cout << "any_group::conjugacy_class_of "
				"before Orb.init" << endl;
	}
	Orb.init(A, A_conj,
			set, 1 /* sz */,
			Subgroup_gens->gens, //LG->Strong_gens->gens,
			verbose_level);
	if (f_v) {
		cout << "any_group::conjugacy_class_of "
				"Found an orbit of size " << Orb.used_length << endl;
	}

	std::vector<long int> Orbit;

	if (f_v) {
		cout << "any_group::conjugacy_class_of "
				"before Orb.get_orbit_of_points" << endl;
	}
	Orb.get_orbit_of_points(Orbit, verbose_level);
	if (f_v) {
		cout << "any_group::conjugacy_class_of "
				"Found an orbit of size " << Orbit.size() << endl;
	}

	int *data;
	int i;
	int nb_r, nb_c;
	std::string *Header_cols;
	std::string *Table;

	nb_r = Orbit.size();
	nb_c = 1;
	data = NEW_int(A->make_element_size);
	Table = new std::string [nb_r * nb_c];
	Header_cols = new std::string [nb_c];

	Header_cols[0] = "elements";

	if (f_v) {
		cout << "any_group::conjugacy_class_of "
				"collecting orbit elements" << endl;
	}

	for (i = 0; i < Orbit.size(); i++) {
		Subgroup_sims->element_unrank_lint(Orbit[i], Elt);

		A->Group_element->element_code_for_make_element(Elt, data);

		//Int_vec_copy(Elt, M + i * A->make_element_size, A->make_element_size);
		Table[i] = "\"" + Int_vec_stringify(data, A->make_element_size) + "\"";
#if 0
		for (j = 0; j < A->make_element_size; j++) {
			M[i * A->make_element_size + j] = Elt[j];
		}
		//M[i] = Orbit[i];
#endif
	}

	if (f_v) {
		cout << "any_group::conjugacy_class_of "
				"collecting orbit elements done" << endl;
	}

	string fname;

	fname = label + "_class_of_" + label_of_class + ".csv";

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_r, nb_c, Table,
			Header_cols,
			verbose_level);

	delete [] Header_cols;
	delete [] Table;

#if 0
	Fio.Csv_file_support->int_matrix_write_csv(
			fname, M, Orbit.size(), A->make_element_size);
#endif

	if (f_v) {
		cout << "any_group::conjugacy_class_of "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	//FREE_int(M);


	FREE_OBJECT(A_conj);
	//FREE_OBJECT(H);



	FREE_int(Elt);
	if (f_v) {
		cout << "any_group::conjugacy_class_of done" << endl;
	}
}



void any_group::automorphism_by_generator_images(
		std::string &label,
		data_structures_groups::vector_ge *Elements_ge,
		int *Images, int m, int n,
		int *&Perms, long int &go,
		int verbose_level)
// uses orbits_schreier::orbit_of_sets
// needs Subgroup_sims to set up action by right multiplication
// output: Perms[m * go]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = false;

	if (f_v) {
		cout << "any_group::automorphism_by_generator_images" << endl;
	}

	//long int go;

	go = Subgroup_sims->group_order_lint();
	if (f_v) {
		cout << "any_group::automorphism_by_generator_images go = " << go << endl;
	}

	Perms = NEW_int(m * go);

#if 0

	ring_theory::longinteger_object a, b;


	if (f_v) {
		cout << "any_group::automorphism_by_generator_images creating element " << elt_data << endl;
	}

	A->Group_element->make_element_from_string(Elt, elt_data, 0);

	Subgroup_sims->element_rank(a, Elt);

	a.assign_to(b);


	if (f_v) {
		cout << "any_group::automorphism_by_generator_images Element :" << endl;
		A->Group_element->element_print(Elt, cout);
		cout << endl;
	}
#endif

	actions::action *A_rm;
	// action by right multiplication

	if (f_v) {
		cout << "any_group::automorphism_by_generator_images "
				"before A->Induced_action->induced_action_by_right_multiplication" << endl;
	}
	A_rm = A->Induced_action->induced_action_by_right_multiplication(
			false /* f_basis */, NULL /* old_G */,
			Subgroup_sims /*Base_group*/, false /* f_ownership */,
			verbose_level - 2);

	if (f_v) {
		cout << "any_group::automorphism_by_generator_images "
				"after A->Induced_action->induced_action_by_right_multiplication" << endl;
	}


	if (f_v) {
		cout << "any_group::automorphism_by_generator_images "
				"generators:" << endl;
		Subgroup_gens->gens->print_quick(cout);
		cout << endl;
	}

	int nb_gens;

	nb_gens = Subgroup_gens->gens->len;

	if (f_v) {
		cout << "any_group::automorphism_by_generator_images "
				"before Orb.init" << endl;
	}


	orbits_schreier::orbit_of_sets Orb;
	long int set[1];
	//orbiter_kernel_system::file_io Fio;

	set[0] = 0;

	if (f_v) {
		cout << "any_group::automorphism_by_generator_images "
				"before Orb.init" << endl;
	}
	Orb.init(A, A_rm,
			set, 1 /* sz */,
			Subgroup_gens->gens,
			verbose_level - 2);
	if (f_v) {
		cout << "any_group::automorphism_by_generator_images "
				"Found an orbit of size " << Orb.used_length << endl;
	}
	if (Orb.used_length != go) {
		cout << "any_group::automorphism_by_generator_images orbit length != go" << endl;
		exit(1);
	}

	combinatorics::combinatorics_domain Combi;

	int in, out;
	int h, i, a, b, c;
	int *Elt;
	int *perm;
	long int new_set[1];
	int pos;

	Elt = NEW_int(A->elt_size_in_int);
	perm = NEW_int(go);


	for (h = 0; h < m; h++) {

		if (f_vv) {
			cout << "any_group::automorphism_by_generator_images h=" << h << " : ";
			Int_vec_print(cout, Images + h * n, n);
			cout << endl;
		}


		for (in = 0; in < go; in++) {

			uint32_t hash;
			std::vector<int> path;

			new_set[0] = in;

			if (!Orb.find_set(
					new_set, pos, hash)) {
				cout << "any_group::automorphism_by_generator_images !find_set" << endl;
				exit(1);
			}
			Orb.get_path(
					path,
					pos);

			if (f_vvv) {
				cout << "any_group::automorphism_by_generator_images in=" << in << " pos=" << pos << " path=";
				Int_vec_stl_print(cout, path);
				cout << endl;
			}


			int *word;

			word = NEW_int(path.size());

			for (i = 0; i < path.size(); i++) {
				a = path[i];
				b = nb_gens - 1 - a; // reverse ordering because the Coxeter generators are listed in reverse
				c = Images[h * n + b];

				word[i] = c;
			}

			if (f_vvv) {
				cout << "any_group::automorphism_by_generator_images in=" << in << " path=";
				Int_vec_stl_print(cout, path);
				cout << " -> ";
				Int_vec_print(cout, word, path.size());
				cout << endl;
			}


			A->Group_element->evaluate_word(
					Elt, word, path.size(),
					Elements_ge,
					verbose_level - 3);

			if (false) {
				cout << "The word evaluates to" << endl;
				A->Group_element->element_print_quick(Elt, cout);
				cout << endl;
				cout << "in latex:" << endl;
				A->Group_element->element_print_latex(Elt, cout);
				cout << endl;
			}

			ring_theory::longinteger_object rk_out;

			Subgroup_sims->element_rank(rk_out, Elt);

			out = rk_out.as_int();

			perm[in] = out;

			if (f_vvv) {
				cout << "any_group::automorphism_by_generator_images in=" << in << " -> " << out << endl;
			}


			FREE_int(word);

		}


		int c;

		c = Combi.Permutations->is_permutation(
				perm, go);
		if (c) {
			if (f_vv) {
				cout << "any_group::automorphism_by_generator_images h = " << h << " output is a permutation" << endl;
			}
		}
		else {
			cout << "any_group::automorphism_by_generator_images h = " << h << " output is not a permutation" << endl;
			exit(1);
		}

		if (f_vv) {
			cout << "any_group::automorphism_by_generator_images h = " << h << ", perm = ";
			Combi.Permutations->perm_print_list(
						cout, perm, go);
			cout << endl;
		}

		Int_vec_copy(perm, Perms + h * go, go);

	}


	FREE_int(Elt);
	FREE_int(perm);

	FREE_OBJECT(A_rm);



	if (f_v) {
		cout << "any_group::automorphism_by_generator_images done" << endl;
	}
}

void any_group::automorphism_by_generator_images_save(
		int *Images, int m, int n,
		int *Perms, long int go,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::automorphism_by_generator_images_save" << endl;
	}


	int nb_r, nb_c;
	int ord;
	std::string *Table;
	std::string *Col_headings;

	nb_r = m;
	nb_c = 4;
	Table = new std::string [nb_r * nb_c];
	Col_headings = new std::string [nb_c];

	Col_headings[0] = "row";
	Col_headings[1] = "order";
	Col_headings[2] = "images";
	Col_headings[3] = "perm";

	int h;
	combinatorics::combinatorics_domain Combi;

	for (h = 0; h < go; h++) {
		Table[h * nb_c + 0] = std::to_string(h);
		ord = Combi.Permutations->perm_order(
				Perms + h * go, go);
		Table[h * nb_c + 1] = std::to_string(ord);
		Table[h * nb_c + 2] = "\"" + Int_vec_stringify(Images + h * n, n) + "\"";
		Table[h * nb_c + 3] = "\"" + Int_vec_stringify(Perms + h * go, go) + "\"";
	}

	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = label + "_elements.csv";

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_r, nb_c, Table,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "any_group::automorphism_by_generator_images_save "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	delete [] Col_headings;
	delete [] Table;

	if (f_v) {
		cout << "any_group::automorphism_by_generator_images_save done" << endl;
	}
}


void any_group::do_reverse_isomorphism_exterior_square(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 5);

	if (f_v) {
		cout << "any_group::do_reverse_isomorphism_exterior_square" << endl;
	}


	if (LG->f_has_nice_gens) {
		if (f_v) {
			cout << "any_group::do_reverse_isomorphism_exterior_square "
					"nice generators are:" << endl;
			LG->nice_gens->print(cout);
		}
		LG->nice_gens->reverse_isomorphism_exterior_square(verbose_level);
	}
	else {
		if (f_v) {
			cout << "any_group::do_reverse_isomorphism_exterior_square "
					"strong generators are:" << endl;
			LG->Strong_gens->print_generators_in_latex_individually(cout, verbose_level - 1);
		}
		LG->Strong_gens->reverse_isomorphism_exterior_square(verbose_level);
	}

	if (f_v) {
		cout << "any_group::do_reverse_isomorphism_exterior_square done" << endl;
	}
}



#if 0
void any_group::create_latex_report_for_permutation_group(
		graphics::layered_graph_draw_options *LG_Draw_options,
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

		fname = label + "_report.tex";
		title = "The group $" + label_tex + "$";
		author = "";


		{
			ofstream ost(fname);
			l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


#if 1
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group "
						"before A->report" << endl;
			}
			A->report(
					ost,
					false, // A->f_has_sims,
					NULL, //A->Sims,
					false, //A->f_has_strong_generators,
					NULL, //A->Strong_gens,
					LG_Draw_options,
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
			Subgroup_gens->print_generators_in_latex_individually(
					ost, verbose_level - 1);
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
					LG_Draw_options,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group "
						"after Subgroup_sims->report" << endl;
			}
#endif

			actions::action_global Action_global;


			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group "
						"before Action_global.report" << endl;
			}
			Action_global.report(
					ost,
					label,
					label_tex,
					A,
					Subgroup_gens,
					false /*f_sylow*/, false /*f_group_table*/,
					LG_Draw_options,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group "
						"after Action_global.report" << endl;
			}


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
		graphics::layered_graph_draw_options *LG_Draw_options,
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

		fname = label + "_report.tex";
		title = "The group $" + label_tex + "$";
		author = "";


		{
			ofstream ost(fname);
			l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);

			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group "
						"before A->report_group_name_and_degree" << endl;
			}

			A->report_group_name_and_degree(
					ost,
					LG_Draw_options,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group "
						"after A->report_group_name_and_degree" << endl;
			}

			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group "
						"before A->report_type_of_action" << endl;
			}

			A->report_type_of_action(
					ost,
					LG_Draw_options,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group "
						"after A->report_type_of_action" << endl;
			}


			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group "
						"before A->report_what_we_act_on" << endl;
			}

			A->report_what_we_act_on(
					ost,
					LG_Draw_options,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group "
						"after A->report_what_we_act_on" << endl;
			}




#if 0
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group "
						"before A->report" << endl;
			}
			A->report(
					ost,
					A->f_has_sims, A->Sims,
					A->f_has_strong_generators, A->Strong_gens,
					LG_Draw_options,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group "
						"after A->report" << endl;
			}
#endif

			if (Subgroup_gens == NULL) {
				cout << "any_group::create_latex_report_for_modified_group "
						"Subgroup_gens == NULL" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group "
						"before Subgroup_gens->print_generators_in_latex_individually" << endl;
			}
			Subgroup_gens->print_generators_in_latex_individually(ost, verbose_level - 1);
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group "
						"after Subgroup_gens->print_generators_in_latex_individually" << endl;
			}
			//A_initial->print_base();
			//A_initial->print_info();

#if 0
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group "
						"before Subgroup_sims->report" << endl;
			}
			Subgroup_sims->report(ost,
					label,
					LG_Draw_options,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_modified_group "
						"after Subgroup_sims->report" << endl;
			}
#endif


			actions::action_global Action_global;


			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group "
						"before Action_global.report" << endl;
			}
			Action_global.report(
					ost,
					label,
					label_tex,
					A,
					Subgroup_gens,
					false /*f_sylow*/, false /*f_group_table*/,
					LG_Draw_options,
					verbose_level);
			if (f_v) {
				cout << "any_group::create_latex_report_for_permutation_group "
						"after Action_global.report" << endl;
			}




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
#endif

groups::strong_generators *any_group::get_strong_generators()
{
	int f_v = false;
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

int any_group::is_subgroup_of(
		any_group *AG_secondary, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = false;


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

void any_group::set_of_coset_representatives(
		any_group *AG_secondary,
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
	Subgroup_gens1->set_of_coset_representatives(
			S,
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

	fname = label + "_coset_reps.tex";


	{
		ofstream fp(fname);
		l1_interfaces::latex_interface L;
		L.head_easy(fp);

		//H->print_all_group_elements_tex(fp);
		//H->print_all_group_elements_with_permutations_tex(fp);

		//Schreier.print_and_list_orbits_tex(fp);

		int i;

		for (i = 0; i < coset_reps->len; i++) {


			fp << "coset " << i << " / " << coset_reps->len << ":" << endl;

			fp << "$$" << endl;
			A->Group_element->element_print_latex(coset_reps->ith(i), fp);
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

	fname = label_of_elements + "_elements.tex";


	{
		ofstream ost(fname);
		l1_interfaces::latex_interface L;
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

			A->Group_element->make_element(Elt,
					element_data + i * A->make_element_size,
					verbose_level);

			ord = A->Group_element->element_order(Elt);

			ost << "Element " << setw(5) << i << " / "
					<< nb_elements << " of order " << ord << ":" << endl;

			A->print_one_element_tex(ost, Elt, f_with_permutation);
			Int_vec_print(ost, Elt, A->make_element_size);
			ost << "\\\\" << endl;

			if (f_with_fix_structure) {
				int f;

				f = A->Group_element->count_fixed_points(Elt, 0 /* verbose_level */);

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


	data_structures_groups::vector_ge *vec_in;

	vec_in = NEW_OBJECT(data_structures_groups::vector_ge);

	vec_in->init_from_data(A, element_data,
		nb_elements, A->make_element_size, 0 /* verbose_level */);



	int *Elt;
	ring_theory::longinteger_object go;

	//Elt = NEW_int(A->elt_size_in_int);


	string fname;

	fname = label_of_elements + "_processing.tex";


	{
		ofstream ost(fname);
		l1_interfaces::latex_interface L;
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

#if 0
			A->Group_element->make_element(Elt,
					element_data + i * A->make_element_size,
					verbose_level);
#endif

			Elt = vec_in->ith(i);

			ord = A->Group_element->element_order(Elt);

			ost << "Element " << setw(5) << i << " / "
					<< nb_elements << " of order " << ord << ":" << endl;

			A->print_one_element_tex(ost, Elt, false /* f_with_permutation */);

		}


		L.foot(ost);
	}

	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname
			<< " of size " << Fio.file_size(fname) << endl;

	FREE_OBJECT(vec_in);
	//FREE_int(Elt);

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

	//int *Elt;
	//int *Elt_in;
	int *Elt_out;
	//int *Output;

	//ring_theory::longinteger_object go;

	//Elt = NEW_int(A->elt_size_in_int);

	int elt_size_out;

	elt_size_out = 6 * 6 + 1;

	Elt_out = NEW_int(elt_size_out);

	//Output = NEW_int(nb_elements * elt_size_out);


	string fname;

	fname = label_of_elements + "_wedge_4to6.csv";

	if (A->type_G != action_on_wedge_product_t) {
		cout << "any_group::apply_isomorphism_wedge_product_4to6 "
				"the action is not of wedge product type" << endl;
		exit(1);
	}

	data_structures_groups::vector_ge *vec_in;
	data_structures_groups::vector_ge *vec_out;

	vec_in = NEW_OBJECT(data_structures_groups::vector_ge);
	vec_out = NEW_OBJECT(data_structures_groups::vector_ge);

	vec_in->init_from_data(A, element_data,
		nb_elements, A->make_element_size, 0 /* verbose_level */);

	vec_out->init(A, verbose_level);
	vec_out->allocate(nb_elements, 0 /* verbose_level */);



	{
		int i;



		for (i = 0; i < nb_elements; i++) {

			//Elt_in = element_data + i * A->make_element_size;

			//A->Group_element->make_element(Elt, Elt_in, verbose_level);



			induced_actions::action_on_wedge_product *AW = A->G.AW;



			//AW->create_induced_matrix(
			//		Elt, Elt_out, verbose_level);

			AW->F->Linear_algebra->wedge_product(
					vec_in->ith(i), Elt_out,
					AW->n, AW->wedge_dimension,
					0 /* verbose_level */);


			if (A->is_semilinear_matrix_group()) {
				Elt_out[6 * 6] = vec_in->ith(i)[4 * 4];
			}

			A->Group_element->make_element(
					vec_out->ith(i), Elt_out, 0 /* verbose_level */);

			//Int_vec_copy(Elt_out, Output + i * elt_size_out, elt_size_out);

		}

	}

	orbiter_kernel_system::file_io Fio;

#if 0
	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Output, nb_elements, elt_size_out);
#else
	vec_out->save_csv(
			fname, verbose_level - 1);
#endif

	if (f_v) {
		cout << "any_group::apply_isomorphism_wedge_product_4to6 "
				"Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
	}


	//FREE_int(Elt);
	FREE_int(Elt_out);
	//FREE_int(Output);
	FREE_OBJECT(vec_in);
	FREE_OBJECT(vec_out);



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

	fname = label_of_elements + "_order_of_products_of_pairs.csv";


	{
		int i, j;



		for (i = 0; i < nb_elements; i++) {

			A->Group_element->make_element(
					Elt1, element_data + i * A->make_element_size, verbose_level);

			for (j = 0; j < nb_elements; j++) {

				A->Group_element->make_element(
						Elt2, element_data + j * A->make_element_size, verbose_level);

				A->Group_element->element_mult(
						Elt1, Elt2, Elt3, 0);

				Order_table[i * nb_elements + j] = A->Group_element->element_order(
						Elt3);


			}
		}


	}

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Order_table, nb_elements, nb_elements);

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

	A->Group_element->make_element_from_string(
			S, conjugate_data, verbose_level);

	A->Group_element->element_invert(S, Sv, verbose_level);

	string fname;

	fname = label_of_elements + "_conjugate.csv";


	{
		int i;



		for (i = 0; i < nb_elements; i++) {

			A->Group_element->make_element(Elt1,
					element_data + i * A->make_element_size,
					verbose_level);


			A->Group_element->element_mult(Sv, Elt1, Elt2, 0);
			A->Group_element->element_mult(Elt2, S, Elt3, 0);

			A->Group_element->code_for_make_element(
					Output_table + i * A->make_element_size, Elt3);


			//A->print_one_element_tex(cout, Elt3, false /* f_with_permutation */);


		}


	}

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Output_table, nb_elements, A->make_element_size);

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
				"before SC->SOG->print_action_on_surface" << endl;
	}
	SC->SOG->print_action_on_surface(
			label_of_elements,
			element_data, nb_elements,
			verbose_level);
	if (f_v) {
		cout << "any_group::print_action_on_surface "
				"after SC->SOG->print_action_on_surface" << endl;
	}

	if (f_v) {
		cout << "any_group::print_action_on_surface done" << endl;
	}

}

void any_group::subgroup_lattice_compute(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::subgroup_lattice_compute" << endl;
	}

	Subgroup_lattice = NEW_OBJECT(groups::subgroup_lattice);

#if 0
	groups::strong_generators *SG;

	SG = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "any_group::subgroup_lattice_compute before init_from_sims" << endl;
	}
	SG->init_from_sims(Subgroup_sims, verbose_level);
	if (f_v) {
		cout << "any_group::subgroup_lattice_compute after init_from_sims" << endl;
	}
#endif


	if (f_v) {
		cout << "any_group::subgroup_lattice_compute "
				"before Subgroup_lattice->init" << endl;
	}
	Subgroup_lattice->init(
			A_base, Subgroup_sims,
			label,
			label_tex,
			Subgroup_gens,
			verbose_level - 1);
	if (f_v) {
		cout << "any_group::subgroup_lattice_compute "
				"after Subgroup_lattice->init" << endl;
	}


	if (f_v) {
		cout << "any_group::subgroup_lattice_compute "
				"before Subgroup_lattice->compute" << endl;
	}
	Subgroup_lattice->compute(verbose_level - 1);
	if (f_v) {
		cout << "any_group::subgroup_lattice_compute "
				"after Subgroup_lattice->compute" << endl;
	}
	std::string fname2;

	fname2 = Subgroup_lattice->label_txt + "_subgroup_lattice.csv";


	if (f_v) {
		cout << "any_group::subgroup_lattice_compute "
				"before Subgroup_lattice->save_csv" << endl;
	}
	Subgroup_lattice->save_csv(fname2, verbose_level - 1);

	//FREE_OBJECT(Subgroup_lattice);
	f_has_subgroup_lattice = true;

	if (f_v) {
		cout << "any_group::subgroup_lattice_compute done" << endl;
	}
}

void any_group::subgroup_lattice_load(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::subgroup_lattice_load" << endl;
	}


	Subgroup_lattice = NEW_OBJECT(groups::subgroup_lattice);


	if (f_v) {
		cout << "any_group::subgroup_lattice_load "
				"before Subgroup_lattice->init" << endl;
	}
	Subgroup_lattice->init(
			A_base, Subgroup_sims,
			label,
			label_tex,
			Subgroup_gens,
			verbose_level - 1);
	if (f_v) {
		cout << "any_group::subgroup_lattice_load "
				"after Subgroup_lattice->init" << endl;
	}


	if (f_v) {
		cout << "any_group::subgroup_lattice_load "
				"before Subgroup_lattice->load_csv" << endl;
	}
	Subgroup_lattice->load_csv(fname, verbose_level - 1);
	if (f_v) {
		cout << "any_group::subgroup_lattice_load "
				"after Subgroup_lattice->load_csv" << endl;
	}


	if (f_v) {
		cout << "any_group::subgroup_lattice_load "
				"before Subgroup_lattice->conjugacy_classes" << endl;
	}
	Subgroup_lattice->conjugacy_classes(verbose_level);
	if (f_v) {
		cout << "any_group::subgroup_lattice_load "
				"after Subgroup_lattice->conjugacy_classes" << endl;
	}

	std::string fname2;

	fname2 = Subgroup_lattice->label_txt + "_subgroup_lattice_with_conj_classes.csv";

	if (f_v) {
		cout << "any_group::subgroup_lattice_load "
				"before Subgroup_lattice->save_csv" << endl;
	}
	Subgroup_lattice->save_csv(fname2, verbose_level - 1);
	if (f_v) {
		cout << "any_group::subgroup_lattice_load "
				"after Subgroup_lattice->save_csv" << endl;
	}

	if (f_v) {
		cout << "any_group::subgroup_lattice_load "
				"Subgroup_lattice:" << endl;
		Subgroup_lattice->print();
	}

	fname2 = Subgroup_lattice->label_txt + "_subgroup_lattice_with_conj_classes_reordered.csv";

	if (f_v) {
		cout << "any_group::subgroup_lattice_load "
				"before Subgroup_lattice->save_rearranged_by_orbits_csv" << endl;
	}
	Subgroup_lattice->save_rearranged_by_orbits_csv(
			fname2,
			verbose_level - 1);
	if (f_v) {
		cout << "any_group::subgroup_lattice_load "
				"after Subgroup_lattice->save_rearranged_by_orbits_csv" << endl;
	}



#if 0
	if (f_v) {
		cout << "any_group::subgroup_lattice_load "
				"before Subgroup_lattice->save_csv" << endl;
	}
	Subgroup_lattice->save_csv(verbose_level - 1);
#endif

	f_has_subgroup_lattice = true;
	//FREE_OBJECT(Subgroup_lattice);

	if (f_v) {
		cout << "any_group::subgroup_lattice_load done" << endl;
	}
}

void any_group::subgroup_lattice_draw(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::subgroup_lattice_draw" << endl;
	}

	if (!f_has_subgroup_lattice) {
		cout << "any_group::subgroup_lattice_draw subgroup lattice is not available" << endl;
		exit(1);
	}

	graph_theory::layered_graph *LG;

	if (f_v) {
		cout << "any_group::subgroup_lattice_draw "
				"before Subgroup_lattice->create_drawing" << endl;
	}
	Subgroup_lattice->create_drawing(
			LG,
			verbose_level - 1);
	if (f_v) {
		cout << "any_group::subgroup_lattice_draw "
				"after Subgroup_lattice->create_drawing" << endl;
	}


	graphics::layered_graph_draw_options *LG_Draw_options;

	LG_Draw_options = orbiter_kernel_system::Orbiter->draw_options;


	std::string fname_base, fname_layered_graph;

	fname_base = Subgroup_lattice->label_txt + "_drawing";
	fname_layered_graph = fname_base + ".layered_graph";

	LG->write_file(fname_layered_graph, verbose_level);
	LG->draw_with_options(
			fname_base, LG_Draw_options,
			verbose_level);


	FREE_OBJECT(LG);

	if (f_v) {
		cout << "any_group::subgroup_lattice_draw done" << endl;
	}
}



void any_group::subgroup_lattice_draw_by_orbits(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::subgroup_lattice_draw_by_orbits" << endl;
	}

	if (!f_has_subgroup_lattice) {
		cout << "any_group::subgroup_lattice_draw_by_orbits subgroup lattice is not available" << endl;
		exit(1);
	}

	graph_theory::layered_graph *LG;

	if (f_v) {
		cout << "any_group::subgroup_lattice_draw_by_orbits "
				"before Subgroup_lattice->create_drawing_by_orbits" << endl;
	}
	Subgroup_lattice->create_drawing_by_orbits(
			LG,
			verbose_level - 1);
	if (f_v) {
		cout << "any_group::subgroup_lattice_draw_by_orbits "
				"after Subgroup_lattice->create_drawing_by_orbits" << endl;
	}


	graphics::layered_graph_draw_options *LG_Draw_options;

	LG_Draw_options = orbiter_kernel_system::Orbiter->draw_options;


	std::string fname_base, fname_layered_graph;

	fname_base = Subgroup_lattice->label_txt + "_drawing_by_orbits";
	fname_layered_graph = fname_base + ".layered_graph";

	LG->write_file(fname_layered_graph, verbose_level);
	LG->draw_with_options(
			fname_base, LG_Draw_options,
			verbose_level);


	FREE_OBJECT(LG);

	if (f_v) {
		cout << "any_group::subgroup_lattice_draw_by_orbits done" << endl;
	}
}


void any_group::subgroup_lattice_intersection_orbit_orbit(
		int orbit1, int orbit2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::subgroup_lattice_intersection_orbit_orbit" << endl;
	}

	if (!f_has_subgroup_lattice) {
		cout << "any_group::subgroup_lattice_intersection_orbit_orbit subgroup lattice is not available" << endl;
		exit(1);
	}

	int *intersection_matrix;
	int len1, len2;
	// intersection_matrix[len1 * len2]

	if (f_v) {
		cout << "any_group::subgroup_lattice_intersection_orbit_orbit "
				"before Subgroup_lattice->intersection_orbit_orbit" << endl;
	}
	Subgroup_lattice->intersection_orbit_orbit(
			orbit1, orbit2,
			intersection_matrix,
			len1, len2,
			verbose_level);

	if (f_v) {
		cout << "any_group::subgroup_lattice_intersection_orbit_orbit "
				"after Subgroup_lattice->intersection_orbit_orbit" << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice_intersection_orbit_orbit intersection_matrix=" << endl;
		Int_matrix_print(intersection_matrix, len1, len2);
	}

	if (f_v) {
		cout << "any_group::subgroup_lattice_intersection_orbit_orbit done" << endl;
	}
}


void any_group::subgroup_lattice_find_overgroup_in_orbit(
		int orbit_global1, int group1, int orbit_global2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::subgroup_lattice_find_overgroup_in_orbit" << endl;
	}

	if (!f_has_subgroup_lattice) {
		cout << "any_group::subgroup_lattice_find_overgroup_in_orbit "
				"subgroup lattice is not available" << endl;
		exit(1);
	}

	int layer1, orb_local1;
	int layer2, orb_local2, group2;

	Subgroup_lattice->orb_global_to_orb_local(
			orbit_global1, layer1, orb_local1,
			verbose_level - 2);
	Subgroup_lattice->orb_global_to_orb_local(
			orbit_global2, layer2, orb_local2,
			verbose_level - 2);

	if (f_v) {
		cout << "any_group::subgroup_lattice_find_overgroup_in_orbit "
				"orbit_global1 = " << orbit_global1 << " = (" << layer1 << "," << orb_local1 << ")" << endl;
		cout << "any_group::subgroup_lattice_find_overgroup_in_orbit "
				"orbit_global2 = " << orbit_global2 << " = (" << layer2 << "," << orb_local2 << ")" << endl;
	}

	if (f_v) {
		cout << "any_group::subgroup_lattice_find_overgroup_in_orbit "
				"before Subgroup_lattice->find_overgroup_in_orbit" << endl;
	}
	Subgroup_lattice->find_overgroup_in_orbit(
			layer1, orb_local1, group1,
			layer2, orb_local2, group2,
			verbose_level);

	if (f_v) {
		cout << "any_group::subgroup_lattice_find_overgroup_in_orbit "
				"after Subgroup_lattice->find_overgroup_in_orbit" << endl;
	}

	if (f_v) {
		cout << "any_group::subgroup_lattice_find_overgroup_in_orbit "
				"group2=" << group2 << endl;
	}

	if (f_v) {
		cout << "any_group::subgroup_lattice_find_overgroup_in_orbit done" << endl;
	}
}


void any_group::subgroup_lattice_create_flag_transitive_geometry_with_partition(
		int P_orbit_global,
		int Q_orbit_global,
		int R_orbit_global,
		int R_group,
		int intersection_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::subgroup_lattice_create_flag_transitive_geometry_with_partition" << endl;
	}

	if (!f_has_subgroup_lattice) {
		cout << "any_group::subgroup_lattice_create_flag_transitive_geometry_with_partition "
				"subgroup lattice is not available" << endl;
		exit(1);
	}

	int P_layer, P_orb_local;
	int Q_layer, Q_orb_local;
	int R_layer, R_orb_local;

	Subgroup_lattice->orb_global_to_orb_local(
			P_orbit_global, P_layer, P_orb_local,
			verbose_level - 2);
	Subgroup_lattice->orb_global_to_orb_local(
			Q_orbit_global, Q_layer, Q_orb_local,
			verbose_level - 2);
	Subgroup_lattice->orb_global_to_orb_local(
			R_orbit_global, R_layer, R_orb_local,
			verbose_level - 2);

	if (f_v) {
		cout << "any_group::subgroup_lattice_create_flag_transitive_geometry_with_partition "
				"P_orbit_global = " << P_orbit_global << " = (" << P_layer << "," << P_orb_local << ")" << endl;
		cout << "any_group::subgroup_lattice_create_flag_transitive_geometry_with_partition "
				"Q_orbit_global = " << Q_orbit_global << " = (" << Q_layer << "," << Q_orb_local << ")" << endl;
		cout << "any_group::subgroup_lattice_create_flag_transitive_geometry_with_partition "
				"R_orbit_global = " << R_orbit_global << " = (" << R_layer << "," << R_orb_local << ")" << endl;
		cout << "any_group::subgroup_lattice_create_flag_transitive_geometry_with_partition "
				"R_group = " << R_group << endl;
		cout << "any_group::subgroup_lattice_create_flag_transitive_geometry_with_partition "
				"intersection_size = " << intersection_size << endl;
	}

	int *intersection_matrix;
	int nb_r, nb_c;


	if (f_v) {
		cout << "any_group::subgroup_lattice_create_flag_transitive_geometry_with_partition "
				"before Subgroup_lattice->create_flag_transitive_geometry_with_partition" << endl;
	}

	Subgroup_lattice->create_flag_transitive_geometry_with_partition(
			P_layer, P_orb_local,
			Q_layer, Q_orb_local,
			R_layer, R_orb_local, R_group,
			intersection_size,
			intersection_matrix,
			nb_r, nb_c,
			verbose_level);

	if (f_v) {
		cout << "any_group::subgroup_lattice_create_flag_transitive_geometry_with_partition "
				"after Subgroup_lattice->create_flag_transitive_geometry_with_partition" << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice_intersection_orbit_orbit intersection_matrix=" << endl;
		Int_matrix_print_comma_separated(intersection_matrix, nb_r, nb_c);
	}

	int *intersection_matrix_t;
	int i, j;

	intersection_matrix_t = NEW_int(nb_c * nb_r);
	for (i = 0; i < nb_c; i++) {
		for (j = 0; j < nb_r; j++) {
			intersection_matrix_t[i * nb_r + j] = intersection_matrix[j * nb_c + i];
		}
	}

	if (f_v) {
		cout << "subgroup_lattice_intersection_orbit_orbit intersection_matrix transposed=" << endl;
		Int_matrix_print_comma_separated(intersection_matrix_t, nb_c, nb_r);
	}

	FREE_int(intersection_matrix);
	FREE_int(intersection_matrix_t);

	if (f_v) {
		cout << "any_group::subgroup_lattice_create_flag_transitive_geometry_with_partition done" << endl;
	}
}



void any_group::subgroup_lattice_create_coset_geometry(
		int P_orb_global, int P_group,
		int Q_orb_global, int Q_group,
		int intersection_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::subgroup_lattice_create_coset_geometry" << endl;
	}

	if (!f_has_subgroup_lattice) {
		cout << "any_group::subgroup_lattice_create_coset_geometry "
				"subgroup lattice is not available" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "any_group::subgroup_lattice_create_coset_geometry "
				"P_orbit_global = " << P_orb_global << endl;
		cout << "any_group::subgroup_lattice_create_coset_geometry "
				"P_group = " << P_group << endl;
		cout << "any_group::subgroup_lattice_create_coset_geometry "
				"Q_orbit_global = " << Q_orb_global << endl;
		cout << "any_group::subgroup_lattice_create_coset_geometry "
				"Q_group = " << Q_group << endl;
	}

	int *intersection_matrix;
	int nb_r, nb_c;


	if (f_v) {
		cout << "any_group::subgroup_lattice_create_coset_geometry "
				"before Subgroup_lattice->create_coset_geometry" << endl;
	}

	Subgroup_lattice->create_coset_geometry(
			P_orb_global, P_group,
			Q_orb_global, Q_group,
			intersection_size,
			intersection_matrix,
			nb_r, nb_c,
			verbose_level);

	if (f_v) {
		cout << "any_group::subgroup_lattice_create_coset_geometry "
				"after Subgroup_lattice->create_coset_geometry" << endl;
	}

	if (f_v) {
		cout << "any_group::subgroup_lattice_create_coset_geometry intersection_matrix=" << endl;
		Int_matrix_print_comma_separated(intersection_matrix, nb_r, nb_c);
	}

#if 1
	int *intersection_matrix_t;
	int i, j;

	intersection_matrix_t = NEW_int(nb_c * nb_r);
	for (i = 0; i < nb_c; i++) {
		for (j = 0; j < nb_r; j++) {
			intersection_matrix_t[i * nb_r + j] = intersection_matrix[j * nb_c + i];
		}
	}


	cout << "incidences:" << endl;
	for (i = 0; i < nb_c; i++) {
		for (j = 0; j < nb_r; j++) {
			if (intersection_matrix_t[i * nb_r + j]) {
				cout << i * nb_r + j << " ";
			}
		}
	}
	cout << endl;

	if (f_v) {
		cout << "any_group::subgroup_lattice_create_coset_geometry intersection_matrix transposed=" << endl;
		Int_matrix_print_comma_separated(intersection_matrix_t, nb_c, nb_r);
	}
	FREE_int(intersection_matrix_t);
#endif

	FREE_int(intersection_matrix);

	if (f_v) {
		cout << "any_group::subgroup_lattice_create_coset_geometry done" << endl;
	}
}





void any_group::subgroup_lattice_identify_subgroup(
		std::string &group_label,
		int &go, int &layer_idx, int &orb_idx, int &group_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::subgroup_lattice_identify_subgroup" << endl;
	}

	if (!f_has_subgroup_lattice) {
		cout << "any_group::subgroup_lattice_identify_subgroup "
				"subgroup lattice is not available" << endl;
		exit(1);
	}

	any_group *Subgroup;

	groups::strong_generators *Subgroup_strong_gens;

	Subgroup = Get_any_group(group_label);

	Subgroup_strong_gens = Subgroup->Subgroup_gens;

	Subgroup_lattice->identify_subgroup(
			Subgroup_strong_gens,
			go, layer_idx, orb_idx, group_idx,
			verbose_level);

	if (f_v) {
		cout << "subgroup_lattice::identify_subgroup "
				"found subgroup of order " << go << " in layer " << layer_idx
				<< " in orbit " << orb_idx << " at position " << group_idx << endl;
	}


	if (f_v) {
		cout << "any_group::subgroup_lattice_identify_subgroup done" << endl;
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

void any_group::print()
{
	cout << "any_group: " << label << " : " << label_tex << endl;
	cout << "A_base:" << endl;
	A_base->print_info();
	cout << "A:" << endl;
	A->print_info();
}

void any_group::classes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::classes" << endl;
	}



	if (Subgroup_gens == NULL) {
		cout << "any_group::classes Subgroup_gens == NULL" << endl;
		exit(1);
	}

	groups::sims *Sims;
	interfaces::conjugacy_classes_and_normalizers *class_data;

	Sims = Subgroup_gens->create_sims(verbose_level);

	interfaces::magma_interface Magma;

	if (f_v) {
		cout << "any_group::classes "
				"before Magma.get_conjugacy_classes_and_normalizers" << endl;
	}
	Magma.get_conjugacy_classes_and_normalizers(
			A, Sims,
			label, label_tex,
			class_data,
			verbose_level);
	if (f_v) {
		cout << "any_group::classes "
				"after Magma.get_conjugacy_classes_and_normalizers" << endl;
	}


	if (f_v) {
		cout << "any_group::classes "
				"before class_data->report" << endl;
	}
	class_data->report(
			Sims,
			label_tex,
			verbose_level - 1);
	if (f_v) {
		cout << "any_group::classes "
				"after class_data->report" << endl;
	}

	if (f_v) {
		cout << "any_group::classes "
				"before class_data->export_csv" << endl;
	}
	class_data->export_csv(
			Sims,
			verbose_level);
	if (f_v) {
		cout << "any_group::classes "
				"after class_data->export_csv" << endl;
	}


	FREE_OBJECT(class_data);



	FREE_OBJECT(Sims);


	if (f_v) {
		cout << "any_group::classes done" << endl;
	}
}

void any_group::subgroup_lattice_magma(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::subgroup_lattice_magma" << endl;
	}



	if (Subgroup_gens == NULL) {
		cout << "any_group::subgroup_lattice_magma Subgroup_gens == NULL" << endl;
		exit(1);
	}

	groups::sims *Sims;

	Sims = Subgroup_gens->create_sims(verbose_level);

	interfaces::magma_interface Magma;


	if (f_v) {
		cout << "any_group::subgroup_lattice_magma "
				"before Magma.get_subgroup_lattice" << endl;
	}

	Magma.get_subgroup_lattice(
			A, Sims,
			label, label_tex,
			class_data,
			verbose_level);

	if (f_v) {
		cout << "any_group::subgroup_lattice_magma "
				"after Magma.get_subgroup_lattice" << endl;
	}

	f_has_class_data = true;

	FREE_OBJECT(Sims);


	if (f_v) {
		cout << "any_group::subgroup_lattice_magma done" << endl;
	}
}



void any_group::find_standard_generators(
		int order_a,
		int order_b,
		int order_ab,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::find_standard_generators" << endl;
	}
#if 0
	if (!Any_group->f_linear_group) {
		cout << "any_group_linear::find_standard_generators !Any_group->f_linear_group" << endl;
		exit(1);
	}
#endif

	algebra_global_with_action Algebra;

	Algebra.find_standard_generators(
			this,
			A, A,
			order_a, order_b, order_ab, verbose_level);

	if (f_v) {
		cout << "any_group::find_standard_generators done" << endl;
	}

}

void any_group::search_element_of_order(
		int order, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_group::search_element_of_order" << endl;
	}
	algebra_global_with_action Algebra;

	Algebra.search_element_of_order(this,
			A, A,
			order, verbose_level);

	if (f_v) {
		cout << "any_group::search_element_of_order done" << endl;
	}
}


}}}

