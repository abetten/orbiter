/*
 * projective_space_activity.cpp
 *
 *  Created on: Jan 5, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


projective_space_activity::projective_space_activity()
{
	Descr = NULL;
	PA = NULL;
}

projective_space_activity::~projective_space_activity()
{

}

void projective_space_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::perform_activity" << endl;
	}

	if (Descr->f_canonical_form_PG) {

		PA->canonical_form(
				Descr->Canonical_form_PG_Descr,
				verbose_level);
	}

	else if (Descr->f_table_of_cubic_surfaces_compute_properties) {

		surface_domain_high_level SH;

		SH.do_cubic_surface_properties(
				PA,
				Descr->table_of_cubic_surfaces_compute_fname_csv,
				Descr->table_of_cubic_surfaces_compute_defining_q,
				Descr->table_of_cubic_surfaces_compute_column_offset,
				verbose_level);
	}
	else if (Descr->f_cubic_surface_properties_analyze) {


		surface_domain_high_level SH;

		SH.do_cubic_surface_properties_analyze(
				PA,
				Descr->cubic_surface_properties_fname_csv,
				Descr->cubic_surface_properties_defining_q,
				verbose_level);
	}
	else if (Descr->f_canonical_form_of_code) {

		canonical_form_of_code(
				PA,
				Descr->canonical_form_of_code_label,
				Descr->canonical_form_of_code_m, Descr->canonical_form_of_code_n,
				Descr->canonical_form_of_code_text,
				verbose_level);

	}
	else if (Descr->f_map) {

		map(
				PA,
				Descr->map_label,
				Descr->map_parameters,
				verbose_level);

	}
	else if (Descr->f_analyze_del_Pezzo_surface) {

		analyze_del_Pezzo_surface(
				PA,
				Descr->analyze_del_Pezzo_surface_label,
				Descr->analyze_del_Pezzo_surface_parameters,
				verbose_level);

	}

	else if (Descr->f_cheat_sheet_for_decomposition_by_element_PG) {

		PA->do_cheat_sheet_for_decomposition_by_element_PG(
				Descr->decomposition_by_element_power,
				Descr->decomposition_by_element_data,
				Descr->decomposition_by_element_fname,
				verbose_level);

	}

	else if (Descr->f_define_surface) {

		cout << "f_define_surface label = " << Descr->define_surface_label << endl;

		surface_with_action *Surf_A;
		surface_create *SC;

		do_create_surface(
			PA,
			Descr->Surface_Descr,
			Surf_A,
			SC,
			verbose_level);

		orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(orbiter_symbol_table_entry);

		Symb->init_cubic_surface(Descr->define_surface_label, SC, verbose_level);
		if (f_v) {
			cout << "before Orbiter->add_symbol_table_entry " << Descr->define_surface_label << endl;
		}
		Orbiter->add_symbol_table_entry(Descr->define_surface_label, Symb, verbose_level);


		//FREE_OBJECT(SC);
		//FREE_OBJECT(Surf_A);
	}

	// surfaces:


	else if (Descr->f_classify_surfaces_with_double_sixes) {

		surface_domain_high_level SH;
		surface_classify_wedge *SCW;


		SH.classify_surfaces_with_double_sixes(
				PA,
				Descr->classify_surfaces_with_double_sixes_control,
				SCW,
				verbose_level);

		orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(orbiter_symbol_table_entry);

		Symb->init_classification_of_cubic_surfaces_with_double_sixes(Descr->classify_surfaces_with_double_sixes_label, SCW, verbose_level);
		if (f_v) {
			cout << "before Orbiter->add_symbol_table_entry " << Descr->classify_surfaces_with_double_sixes_label << endl;
		}
		Orbiter->add_symbol_table_entry(Descr->classify_surfaces_with_double_sixes_label, Symb, verbose_level);

	}

	else if (Descr->f_classify_surfaces_through_arcs_and_two_lines) {

		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		surface_domain_high_level SH;

		SH.do_classify_surfaces_through_arcs_and_two_lines(
				PA,
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
		surface_domain_high_level SH;

		SH.do_classify_surfaces_through_arcs_and_trihedral_pairs(
				PA,
				Descr->Trihedra1_control, Descr->Trihedra2_control,
				Descr->Control_six_arcs,
				Descr->f_test_nb_Eckardt_points, Descr->nb_E,
				verbose_level);
	}
	else if (Descr->f_sweep_4) {

		surface_domain_high_level SH;


		SH.do_sweep_4(
				PA,
				Descr->sweep_4_surface_description,
				Descr->sweep_4_fname,
				verbose_level);
	}
	else if (Descr->f_create_surface) {
		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}

		surface_domain_high_level SH;

		SH.do_create_surface(
				PA,
				Descr->surface_description, Descr->Control_six_arcs,
				verbose_level);
	}
	else if (Descr->f_six_arcs) {
		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		surface_domain_high_level SH;

		SH.do_six_arcs(
				PA,
				Descr->Control_six_arcs,
				Descr->f_filter_by_nb_Eckardt_points, Descr->nb_Eckardt_points,
				verbose_level);
	}
	else if (Descr->f_make_gilbert_varshamov_code) {

		coding_theory_domain Coding;

		Coding.make_gilbert_varshamov_code(
				Descr->make_gilbert_varshamov_code_n,
				Descr->make_gilbert_varshamov_code_n - (PA->P->n + 1),
				Descr->make_gilbert_varshamov_code_d,
				PA->P->F->q,
				PA->P, verbose_level);
	}

	else if (Descr->f_spread_classify) {
		do_spread_classify(PA,
				Descr->spread_classify_k,
				Descr->spread_classify_Control,
				verbose_level);
	}
	else if (Descr->f_classify_semifields) {
		do_classify_semifields(
				PA,
				Descr->Semifield_classify_description,
				Descr->Semifield_classify_Control,
				verbose_level);

	}
	else if (Descr->f_cheat_sheet) {

		layered_graph_draw_options *O;

		if (Orbiter->f_draw_options) {
			O = Orbiter->draw_options;
		}
		else {
			cout << "please use -draw_options .. -end" << endl;
			exit(1);
		}
		do_cheat_sheet_PG(
				PA,
				O,
				verbose_level);
	}
	else if (Descr->f_classify_quartic_curves) {

		classify_quartic_curves(PA,
				Descr->classify_quartic_curves_fname_mask, Descr->classify_quartic_curves_nb,
				verbose_level);
	}


	if (f_v) {
		cout << "projective_space_activity::perform_activity done" << endl;
	}

}


void projective_space_activity::map(
		projective_space_with_action *PA,
		std::string &label,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::map" << endl;
	}



	int idx;
	idx = The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->find_symbol(label);

	if (idx < 0) {
		cout << "could not find symbol " << label << endl;
		exit(1);
	}
	The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->get_object(idx);

	if (The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].type != t_object) {
		cout << "symbol table entry must be of type t_object" << endl;
		exit(1);
	}
	if (The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].object_type == t_collection) {
		cout << "symbol table entry is a collection" << endl;

		vector<string> *List;

		List = (vector<string> *) The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].ptr;
		int i;

		for (i = 0; i < List->size(); i++) {
			int idx1;

			idx1 = The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->find_symbol((*List)[i]);
			if (idx1 < 0) {
				cout << "could not find symbol " << (*List)[i] << endl;
				exit(1);
			}
			formula *Formula;
			Formula = (formula *) The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx1].ptr;

			PA->map(Formula,
					evaluate_text,
					verbose_level);
		}
	}
	else if (The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].object_type == t_formula) {
		cout << "symbol table entry is a formula" << endl;

		formula *Formula;
		Formula = (formula *) The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].ptr;

		PA->map(Formula,
				evaluate_text,
				verbose_level);
	}
	else {
		cout << "symbol table entry must be either a formula or a collection" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_activity::map done" << endl;
	}
}


void projective_space_activity::analyze_del_Pezzo_surface(
		projective_space_with_action *PA,
		std::string &label,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface" << endl;
	}



	int idx;
	idx = The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->find_symbol(label);

	if (idx < 0) {
		cout << "could not find symbol " << label << endl;
		exit(1);
	}
	The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->get_object(idx);

	if (The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].type != t_object) {
		cout << "symbol table entry must be of type t_object" << endl;
		exit(1);
	}
	if (The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].object_type == t_collection) {
		cout << "symbol table entry is a collection" << endl;

		vector<string> *List;

		List = (vector<string> *) The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].ptr;
		int i;

		for (i = 0; i < List->size(); i++) {
			int idx1;

			idx1 = The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->find_symbol((*List)[i]);
			if (idx1 < 0) {
				cout << "could not find symbol " << (*List)[i] << endl;
				exit(1);
			}
			formula *F;
			F = (formula *) The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx1].ptr;

			analyze_del_Pezzo_surface_formula_given(
					PA,
					F,
					evaluate_text,
					verbose_level);
		}
	}
	else if (The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].object_type == t_formula) {
		cout << "symbol table entry is a formula" << endl;

		formula *F;
		F = (formula *) The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].ptr;

		analyze_del_Pezzo_surface_formula_given(
				PA,
				F,
				evaluate_text,
				verbose_level);
	}
	else {
		cout << "symbol table entry must be either a formula or a collection" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface done" << endl;
	}
}

void projective_space_activity::analyze_del_Pezzo_surface_formula_given(
		projective_space_with_action *PA,
		formula *F,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface_formula_given" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface_formula_given before PA->analyze_del_Pezzo_surface" << endl;
	}

	PA->analyze_del_Pezzo_surface(F, evaluate_text, verbose_level);

	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface_formula_given after PA->analyze_del_Pezzo_surface" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface_formula_given done" << endl;
	}
}




void projective_space_activity::canonical_form_of_code(
		projective_space_with_action *PA,
		std::string &label, int m, int n,
		std::string &data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code" << endl;
		cout << "m=" << m << endl;
		cout << "n=" << n << endl;
		cout << "data=" << data << endl;
	}

	int *genma;
	int sz;
	int i, j;
	int *v;
	long int *set;

	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code before int_vec_scan" << endl;
	}
	Orbiter->Int_vec.scan(data, genma, sz);
	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code after int_vec_scan, sz=" << sz << endl;
	}

	if (sz != m * n) {
		cout << "projective_space_activity::canonical_form_of_code sz != m * n" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "genma: " << endl;
		Orbiter->Int_vec.print(cout, genma, sz);
		cout << endl;
	}
	v = NEW_int(m);
	set = NEW_lint(n);
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			v[i] = genma[i * n + j];
		}
		if (f_v) {
			cout << "projective_space_activity::canonical_form_of_code before PA->P->rank_point" << endl;
			Orbiter->Int_vec.print(cout, v, m);
			cout << endl;
		}
		if (PA->P == NULL) {
			cout << "PA->P == NULL" << endl;
			exit(1);
		}
		set[j] = PA->P->rank_point(v);
	}
	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code set=";
		Orbiter->Lint_vec.print(cout, set, n);
		cout << endl;
	}

	projective_space_object_classifier_description Descr;
	data_input_stream Data;
	string points_as_string;
	char str[1000];

	sprintf(str, "%ld", set[0]);
	points_as_string.assign(str);
	for (i = 1; i < n; i++) {
		points_as_string.append(",");
		sprintf(str, "%ld", set[i]);
		points_as_string.append(str);
	}
	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code points_as_string=" << points_as_string << endl;
	}

	Descr.f_input = TRUE;
	Descr.Data = &Data;

	Descr.f_save_classification = TRUE;
	Descr.save_prefix.assign("code_");

	Descr.f_report = TRUE;
	Descr.report_prefix.assign("code_");
	Descr.report_prefix.append(label);

	Descr.f_classification_prefix = TRUE;
	Descr.classification_prefix.assign("classify_code_");
	Descr.classification_prefix.append(label);

	Data.nb_inputs = 0;
	Data.input_type[Data.nb_inputs] = INPUT_TYPE_SET_OF_POINTS;
	Data.input_string[Data.nb_inputs] = points_as_string;
	Data.nb_inputs++;


	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code before PA->canonical_form" << endl;
	}

	PA->canonical_form(&Descr, verbose_level);

	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code after PA->canonical_form" << endl;
	}


	FREE_int(v);
	FREE_lint(set);

	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code done" << endl;
	}
}



void projective_space_activity::do_create_surface(
		projective_space_with_action *PA,
		surface_create_description *Surface_Descr,
		surface_with_action *&Surf_A,
		surface_create *&SC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::do_create_surface" << endl;
		cout << "projective_space_activity::do_create_surface verbose_level=" << verbose_level << endl;
	}

	int q;
	surface_domain *Surf;

	if (f_v) {
		cout << "projective_space_activity::do_create_surface before Surface_Descr->get_q" << endl;
	}
	q = Surface_Descr->get_q();
	if (f_v) {
		cout << "projective_space_activity::do_create_surface q = " << q << endl;
	}

	if (PA->q != q) {
		cout << "projective_space_activity::do_create_surface PA->q != q" << endl;
		exit(1);
	}
	if (PA->n != 3) {
		cout << "projective_space_activity::do_create_surface we need a three-dimensional projective space" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_activity::do_create_surface before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_activity::do_create_surface after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "projective_space_activity::do_create_surface before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_activity::do_create_surface after Surf_A->init" << endl;
	}


	if (f_v) {
		cout << "projective_space_activity::do_create_surface before Surf_A->create_surface_and_do_report" << endl;
	}

	Surf_A->create_surface(
			Surface_Descr,
			SC,
			verbose_level);

	if (f_v) {
		cout << "projective_space_activity::do_create_surface after Surf_A->create_surface_and_do_report" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::do_create_surface done" << endl;
	}
}


void projective_space_activity::do_spread_classify(
		projective_space_with_action *PA,
		int k,
		poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::do_spread_classify" << endl;
	}


	spread_classify *SC;

	SC = NEW_OBJECT(spread_classify);

	if (f_v) {
		cout << "projective_space_activity::do_spread_classify before SC->init" << endl;
	}

	SC->init(
			PA,
			k,
			TRUE /* f_recoordinatize */,
			verbose_level - 1);
	if (f_v) {
		cout << "projective_space_activity::do_spread_classify after SC->init" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::do_spread_classify before SC->init2" << endl;
	}
	SC->init2(Control, verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_spread_classify after SC->init2" << endl;
	}


	if (f_v) {
		cout << "projective_space_activity::do_spread_classify before SC->compute" << endl;
	}

	SC->compute(verbose_level);

	if (f_v) {
		cout << "projective_space_activity::do_spread_classify after SC->compute" << endl;
	}


	FREE_OBJECT(SC);

	if (f_v) {
		cout << "projective_space_activity::do_spread_classify done" << endl;
	}
}


void projective_space_activity::do_classify_semifields(
		projective_space_with_action *PA,
		semifield_classify_description *Semifield_classify_description,
		poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::do_classify_semifields" << endl;
	}


	semifield_classify_with_substructure *S;

	S = NEW_OBJECT(semifield_classify_with_substructure);

	if (f_v) {
		cout << "projective_space_activity::do_classify_semifields before S->init" << endl;
	}
	S->init(
			Semifield_classify_description,
			PA,
			Control,
			verbose_level);
	if (f_v) {
		cout << "projective_space_activity::do_classify_semifields after S->init" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::do_classify_semifields done" << endl;
	}
}


void projective_space_activity::do_cheat_sheet_PG(
		projective_space_with_action *PA,
		layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_activity::do_cheat_sheet_PG verbose_level="
				<< verbose_level << endl;
	}



	{
		char fname[1000];
		char title[1000];
		char author[1000];

		snprintf(fname, 1000, "PG_%d_%d.tex", PA->n, PA->F->q);
		snprintf(title, 1000, "Cheat Sheet ${\\rm PG}(%d,%d)$", PA->n, PA->F->q);
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


			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG before PA->A->report" << endl;
			}

			PA->A->report(ost, PA->A->f_has_sims, PA->A->Sims,
					PA->A->f_has_strong_generators, PA->A->Strong_gens,
					O,
					verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG after PA->A->report" << endl;
			}

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG before PA->P->report" << endl;
			}



			PA->P->report(ost, O, verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG after PA->P->report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}

	}


	if (f_v) {
		cout << "projective_space_activity::do_cheat_sheet_PG done" << endl;
	}

}


void projective_space_activity::classify_quartic_curves(
		projective_space_with_action *PA,
		std::string &fname_mask, int nb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_activity::classify_quartic_curves" << endl;
	}
	homogeneous_polynomial_domain *Poly4_x123;

	int cnt;

	Poly4_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly4_x123->init(PA->F, 3, 4, FALSE, t_PART, verbose_level);

	action_on_homogeneous_polynomials *AonHPD;

	AonHPD = NEW_OBJECT(action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "projective_space_activity::process_quartic_curve "
				"before AonHPD->init" << endl;
	}
	AonHPD->init(PA->A, Poly4_x123, verbose_level);
	if (f_v) {
		cout << "projective_space_activity::process_quartic_curve "
				"after AonHPD->init" << endl;
	}



	int nb_objects_to_test;

	nb_objects_to_test = 0;

	for (cnt = 0; cnt < nb; cnt++) {
		char str[1000];
		string fname;

		sprintf(str, fname_mask.c_str(), cnt);
		fname.assign(str);

		spreadsheet S;

		S.read_spreadsheet(fname, verbose_level);

		nb_objects_to_test += S.nb_rows - 1;
		if (f_v) {
			cout << "projective_space_activity::process_quartic_curve "
					"file " << cnt << " / " << nb << " has  " << S.nb_rows - 1 << " objects" << endl;
		}
	}

	if (f_v) {
		cout << "projective_space_activity::process_quartic_curve "
				"nb_objects_to_test=" << nb_objects_to_test << endl;
	}


	number_theory_domain NT;

	classify_bitvectors *CB;

	CB = NEW_OBJECT(classify_bitvectors);

	int canonical_labeling_len;
	long int *alpha;
	int *gamma;

	int *Elt;
	int *eqn2;

	Elt = NEW_int(PA->A->elt_size_in_int);
	eqn2 = NEW_int(Poly4_x123->get_nb_monomials());


	for (cnt = 0; cnt < nb; cnt++) {
		char str[1000];
		string fname;
		int row;

		sprintf(str, fname_mask.c_str(), cnt);
		fname.assign(str);

		spreadsheet S;

		S.read_spreadsheet(fname, verbose_level);

		if (f_v) {
			cout << "projective_space_activity::classify_quartic_curves S.nb_rows = " << S.nb_rows << endl;
			cout << "projective_space_activity::classify_quartic_curves S.nb_cols = " << S.nb_cols << endl;
		}

		int j, t;
		string eqn_txt;
		string pts_txt;
		string bitangents_txt;
		int *eqn;
		int sz;
		long int *pts;
		int nb_pts;
		long int *bitangents;
		int nb_bitangents;



		for (row = 0; row < S.nb_rows - 1; row++) {

			if (f_v) {
				cout << "cnt = " << cnt << " / " << nb << " row = " << row << " / " << S.nb_rows - 1 << endl;
			}

			j = 1;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "projective_space_activity::classify_quartic_curves token[t] == NULL" << endl;
			}
			eqn_txt.assign(S.tokens[t]);
			j = 2;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "projective_space_activity::classify_quartic_curves token[t] == NULL" << endl;
			}
			pts_txt.assign(S.tokens[t]);
			j = 3;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "projective_space_activity::classify_quartic_curves token[t] == NULL" << endl;
			}
			bitangents_txt.assign(S.tokens[t]);

			string_tools ST;

			ST.remove_specific_character(eqn_txt, '\"');
			ST.remove_specific_character(pts_txt, '\"');
			ST.remove_specific_character(bitangents_txt, '\"');

			if (f_v) {
				cout << "row = " << row << " eqn=" << eqn_txt << " pts_txt=" << pts_txt << " =" << bitangents_txt << endl;
			}

			Orbiter->Int_vec.scan(eqn_txt, eqn, sz);
			Orbiter->Lint_vec.scan(pts_txt, pts, nb_pts);
			Orbiter->Lint_vec.scan(bitangents_txt, bitangents, nb_bitangents);

			if (f_v) {
				cout << "row = " << row << " eqn=";
				Orbiter->Int_vec.print(cout, eqn, sz);
				cout << " pts=";
				Orbiter->Lint_vec.print(cout, pts, nb_pts);
				cout << " bitangents=";
				Orbiter->Lint_vec.print(cout, bitangents, nb_bitangents);
				cout << endl;
			}


			canonical_form *C;
			longinteger_object go;


			C = NEW_OBJECT(canonical_form);

			C->quartic_curve(
					PA,
					Poly4_x123,
					AonHPD,
					row, eqn, sz,
					pts, nb_pts,
					bitangents, nb_bitangents,
					verbose_level);

			C->Stab_gens_quartic->group_order(go);

			canonical_labeling_len = C->canonical_labeling_len;
			alpha = NEW_lint(canonical_labeling_len);
			gamma = NEW_int(canonical_labeling_len);


			if (CB->n == 0) {
				CB->init(nb_objects_to_test,
						C->Canonical_form->get_allocated_length(),
						verbose_level);
			}
			int f_found;
			int idx;

			CB->search_and_add_if_new(C->Canonical_form->get_data(), C /* void *extra_data */, f_found, idx, verbose_level);


			if (!f_found) {
				if (f_v) {
					cout << "After search_and_add_if_new, cnt = " << cnt << " row = " << row << " The canonical form is new" << endl;
				}
			}
			else {
				if (f_v) {
					cout << "After search_and_add_if_new, cnt = " << cnt << " row = " << row << " We found the canonical form at idx = " << idx << endl;
				}




				long int *alpha_inv;
				long int *beta_inv;
				int i;

				//long int *canonical_labeling;




				int idx1;
				int found_at = -1;

				if (f_v) {
					cout << "starting loop over idx1" << endl;
				}

				for (idx1 = idx; idx1 >= 0; idx1--) {



					// test if entry at idx1 is equal to C.
					// if not, break

					if (f_v) {
						cout << "projective_space_activity::classify_quartic_curves before CB->compare_at idx1 = " << idx1 << endl;
					}
					if (CB->compare_at(C->Canonical_form->get_data(), idx1) != 0) {
						if (f_v) {
							cout << "projective_space_activity::classify_quartic_curves at idx1 = " << idx1 << " is not equal, break" << endl;
						}
						break;
					}
					if (f_v) {
						cout << "projective_space_activity::classify_quartic_curves canonical form at " << idx1 << " is equal" << endl;
					}


					canonical_form *C1;
					C1 = (canonical_form *) CB->Type_extra_data[idx1];

					alpha_inv = C1->canonical_labeling;

					beta_inv = C->canonical_labeling;

					// compute gamma = beta * alpha^-1


					if (f_v) {
						cout << "projective_space_activity::classify_quartic_curves computing alpha" << endl;
					}
					for (i = 0; i < canonical_labeling_len; i++) {
						j = alpha_inv[i];
						alpha[j] = i;
					}

					if (f_v) {
						cout << "projective_space_activity::classify_quartic_curves computing gamma" << endl;
					}
					for (i = 0; i < canonical_labeling_len; i++) {
						gamma[i] = beta_inv[alpha[i]];
					}


					// turn gamma into a matrix


					int Mtx[10];
					int Mtx_inv[10];
					int frobenius;

					if (f_v) {
						cout << "projective_space_activity::classify_quartic_curves before PA->P->reverse_engineer_semilinear_map" << endl;
					}
					PA->P->reverse_engineer_semilinear_map(
						gamma, Mtx, frobenius,
						0 /*verbose_level*/);
					if (f_v) {
						cout << "projective_space_activity::classify_quartic_curves after PA->P->reverse_engineer_semilinear_map" << endl;
					}

					Mtx[9] = frobenius;

					PA->A->make_element(Elt, Mtx, 0 /* verbose_level*/);

					if (f_v) {
						cout << "The isomorphism from C to C1 is given by:" << endl;
						PA->A->element_print(Elt, cout);
					}



					int frobenius_inv;

					frobenius_inv = NT.int_negate(Mtx[3 * 3], PA->F->e);


					PA->F->matrix_inverse(Mtx, Mtx_inv, 3, 0 /* verbose_level*/);

					if (f_v) {
						cout << "projective_space_activity::classify_quartic_curves before substitute_semilinear" << endl;
					}
					Poly4_x123->substitute_semilinear(C->eqn /* coeff_in */, eqn2 /* coeff_out */,
						PA->A->is_semilinear_matrix_group(), frobenius, Mtx, 0/*verbose_level*/);
					if (f_v) {
						cout << "projective_space_activity::classify_quartic_curves after substitute_semilinear" << endl;
					}

					PA->F->PG_element_normalize_from_front(eqn2, 1, Poly4_x123->get_nb_monomials());


					if (f_v) {
						cout << "The mapped equation is:";
						Poly4_x123->print_equation_simple(cout, eqn2);
						cout << endl;
					}




					int idx2;

					if (!C1->Orb->search_equation(eqn2 /*new_object */, idx2, TRUE)) {
						// need to map points and bitangents under gamma:
						if (f_v) {
							cout << "we found the canonical form but we did not find the equation at idx1=" << idx1 << endl;
						}


					}
					else {
						if (f_v) {
							cout << "After search_and_add_if_new, cnt = " << cnt << " row = " << row << " We found the canonical form and the equation at idx2 " << idx2 << ", idx1=" << idx1 << endl;
						}
						found_at = idx1;
						break;
					}


				}


				if (found_at == -1) {

					if (f_v) {
						cout << "we found the canonical form but we did not find the equation" << endl;
					}

					long int *pts2;
					//int nb_pts;
					long int *bitangents2;
					//int nb_bitangents;
					int i;

					pts2 = NEW_lint(nb_pts);
					bitangents2 = NEW_lint(nb_bitangents);

					for (i = 0; i < nb_pts; i++) {
						pts2[i] = PA->A->element_image_of(pts[i], Elt, 0 /* verbose_level */);
					}
					for (i = 0; i < nb_bitangents; i++) {
						bitangents2[i] = PA->A_on_lines->element_image_of(bitangents[i], Elt, 0 /* verbose_level */);
					}

					canonical_form *C2;
					longinteger_object go;


					C2 = NEW_OBJECT(canonical_form);

					if (f_v) {
						cout << "we recompute the quartic curve from the canonical equation." << endl;
					}
					if (f_v) {
						cout << "projective_space_activity::classify_quartic_curves before C2->quartic_curve" << endl;
					}
					C2->quartic_curve(
							PA,
							Poly4_x123,
							AonHPD,
							row, eqn2, sz,
							pts2, nb_pts,
							bitangents2, nb_bitangents,
							verbose_level);
					if (f_v) {
						cout << "projective_space_activity::classify_quartic_curves after C2->quartic_curve" << endl;
					}

					if (f_v) {
						cout << "After search_and_add_if_new, adding at " << idx << endl;
					}
					CB->add_at_idx(C2->Canonical_form->get_data(), C2 /* void *extra_data */, idx, 0 /* verbose_level*/);


				} // if (found_at == -1)
				else {
					if (f_v) {
						cout << "we found the equation at found_at = " << found_at << endl;
					}

				}

			} // if f_found

			FREE_lint(alpha);
			FREE_int(gamma);

		} // next row


	} // next cnt

	cout << "The number of types of canonical forms is " << CB->nb_types << endl;

	FREE_OBJECT(Poly4_x123);
	FREE_OBJECT(AonHPD);

	FREE_int(eqn2);
	FREE_int(Elt);


	if (f_v) {
		cout << "projective_space_activity::classify_quartic_curves done" << endl;
	}
}






}}
