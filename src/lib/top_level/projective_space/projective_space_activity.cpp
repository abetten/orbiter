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

	else if (Descr->f_define_quartic_curve) {

		cout << "f_define_quartic_curve label = " << Descr->f_define_quartic_curve << endl;

		quartic_curve_create *QC;

		do_create_quartic_curve(
			PA,
			Descr->Quartic_curve_descr,
			QC,
			verbose_level);

		orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(orbiter_symbol_table_entry);

		Symb->init_quartic_curve(Descr->define_quartic_curve_label, QC, verbose_level);
		if (f_v) {
			cout << "before Orbiter->add_symbol_table_entry " << Descr->define_surface_label << endl;
		}
		Orbiter->add_symbol_table_entry(Descr->define_quartic_curve_label, Symb, verbose_level);


		//FREE_OBJECT(SC);
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
	else if (Descr->f_sweep_4_27) {

		surface_domain_high_level SH;


		SH.do_sweep_4_27(
				PA,
				Descr->sweep_4_27_surface_description,
				Descr->sweep_4_27_fname,
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
	else if (Descr->f_classify_quartic_curves_nauty) {

		classify_quartic_curves_nauty(PA,
				Descr->classify_quartic_curves_nauty_fname_mask,
				Descr->classify_quartic_curves_nauty_nb,
				Descr->classify_quartic_curves_nauty_fname_classification,
				verbose_level);
	}


	else if (Descr->f_classify_quartic_curves_with_substructure) {

		classify_quartic_curves_with_substructure(PA,
				Descr->classify_quartic_curves_with_substructure_fname_mask,
				Descr->classify_quartic_curves_with_substructure_nb,
				Descr->classify_quartic_curves_with_substructure_size,
				Descr->classify_quartic_curves_with_substructure_degree,
				Descr->classify_quartic_curves_with_substructure_fname_classification,
				verbose_level);
	}
	else if (Descr->f_set_stabilizer) {

		set_stabilizer(PA,
				Descr->set_stabilizer_intermediate_set_size,
				Descr->set_stabilizer_fname_mask, Descr->set_stabilizer_nb,
				verbose_level);
	}

	else if (Descr->f_conic_type) {

		conic_type(PA,
				Descr->conic_type_set_text,
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


void projective_space_activity::do_create_quartic_curve(
		projective_space_with_action *PA,
		quartic_curve_create_description *Quartic_curve_descr,
		quartic_curve_create *&QC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::do_create_quartic_curve" << endl;
		cout << "projective_space_activity::do_create_quartic_curve verbose_level=" << verbose_level << endl;
	}

	int q;

	if (f_v) {
		cout << "projective_space_activity::do_create_quartic_curve before Surface_Descr->get_q" << endl;
	}
	q = Quartic_curve_descr->get_q();
	if (f_v) {
		cout << "projective_space_activity::do_create_quartic_curve q = " << q << endl;
	}

	if (PA->q != q) {
		cout << "projective_space_activity::do_create_quartic_curve PA->q != q" << endl;
		exit(1);
	}
	if (PA->n != 2) {
		cout << "projective_space_activity::do_create_quartic_curve we need a two-dimensional projective space" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_activity::do_create_quartic_curve before PA->create_quartic_curve" << endl;
	}

	PA->create_quartic_curve(
				Quartic_curve_descr,
				QC,
				verbose_level);

	if (f_v) {
		cout << "projective_space_activity::do_create_quartic_curve after PA->create_quartic_curve" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::do_create_quartic_curve done" << endl;
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


void projective_space_activity::classify_quartic_curves_nauty(
		projective_space_with_action *PA,
		std::string &fname_mask, int nb,
		std::string &fname_classification,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_activity::classify_quartic_curves_nauty" << endl;
	}



	canonical_form_classifier_description Descr;

	Descr.fname_mask.assign(fname_mask);
	Descr.f_fname_base_out = TRUE;
	Descr.fname_base_out.assign(fname_classification);
	Descr.PA = PA;
	Descr.f_degree = TRUE;
	Descr.degree = 4;
	Descr.nb_files = nb;
	Descr.f_algorithm_nauty = TRUE;
	Descr.f_algorithm_substructure = FALSE;

	canonical_form_classifier Classifier;

	Classifier.classify(&Descr, verbose_level);

	cout << "The number of types of quartic curves is " << Classifier.CB->nb_types << endl;


	int idx;

	cout << "idx : ago" << endl;
	for (idx = 0; idx < Classifier.CB->nb_types; idx++) {

		canonical_form_nauty *C1;
		longinteger_object go;

		C1 = (canonical_form_nauty *) Classifier.CB->Type_extra_data[idx];

		C1->Stab_gens_quartic->group_order(go);

		cout << idx << " : " << go << endl;


	}



	if (f_v) {
		cout << "projective_space_activity::classify_quartic_curves_nauty done" << endl;
	}
}


void projective_space_activity::classify_quartic_curves_with_substructure(
		projective_space_with_action *PA,
		std::string &fname_mask, int nb, int substructure_size, int degree,
		std::string &fname_classification,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_activity::classify_quartic_curves_with_substructure" << endl;
	}



	canonical_form_classifier_description Descr;

	Descr.fname_mask.assign(fname_mask);
	Descr.f_fname_base_out = TRUE;
	Descr.fname_base_out.assign(fname_classification);
	Descr.PA = PA;
	Descr.f_degree = TRUE;
	Descr.degree = degree;
	Descr.nb_files = nb;
	Descr.f_algorithm_nauty = FALSE;
	Descr.f_algorithm_substructure = TRUE;
	Descr.substructure_size = substructure_size;

	canonical_form_classifier Classifier;

	Classifier.classify(&Descr, verbose_level);


	Classifier.report(fname_classification, verbose_level);

#if 0
	cout << "The number of types of quartic curves is " << Classifier.CB->nb_types << endl;
	int idx;

	cout << "idx : ago" << endl;
	for (idx = 0; idx < Classifier.CB->nb_types; idx++) {

		canonical_form *C1;
		longinteger_object go;

		C1 = (canonical_form *) Classifier.CB->Type_extra_data[idx];

		C1->Stab_gens_quartic->group_order(go);

		cout << idx << " : " << go << endl;


	}
#endif


	if (f_v) {
		cout << "projective_space_activity::classify_quartic_curves_with_substructure done" << endl;
	}
}

void projective_space_activity::set_stabilizer(
		projective_space_with_action *PA,
		int intermediate_subset_size,
		std::string &fname_mask, int nb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_activity::set_stabilizer" << endl;
	}

	poset_classification *PC;
	poset_classification_control *Control;
	poset_with_group_action *Poset;
	int nb_orbits;
	int j;

	Poset = NEW_OBJECT(poset_with_group_action);


	Control = NEW_OBJECT(poset_classification_control);

	Control->f_depth = TRUE;
	Control->depth = intermediate_subset_size;


	if (f_v) {
		cout << "projective_space_activity::set_stabilizer control=" << endl;
		Control->print();
	}


	Poset->init_subset_lattice(PA->A, PA->A,
			PA->A->Strong_gens,
			verbose_level);

	if (f_v) {
		cout << "projective_space_activity::set_stabilizer "
				"before Poset->orbits_on_k_sets_compute" << endl;
	}
	PC = Poset->orbits_on_k_sets_compute(
			Control,
			intermediate_subset_size,
			verbose_level);
	if (f_v) {
		cout << "projective_space_activity::set_stabilizer "
				"after Poset->orbits_on_k_sets_compute" << endl;
	}

	nb_orbits = PC->nb_orbits_at_level(intermediate_subset_size);

	cout << "We found " << nb_orbits << " orbits at level " << intermediate_subset_size << ":" << endl;
	for (j = 0; j < nb_orbits; j++) {


		strong_generators *Strong_gens;

		PC->get_stabilizer_generators(
				Strong_gens,
				intermediate_subset_size, j, 0 /* verbose_level*/);

		longinteger_object go;

		Strong_gens->group_order(go);

		FREE_OBJECT(Strong_gens);

		cout << j << " : " << go << endl;


	}


	int nb_objects_to_test;
	int cnt;
	int row;

	nb_objects_to_test = 0;


	for (cnt = 0; cnt < nb; cnt++) {

		char str[1000];
		string fname;

		sprintf(str, fname_mask.c_str(), cnt);
		fname.assign(str);

		spreadsheet S;

		S.read_spreadsheet(fname, 0 /*verbose_level*/);

		nb_objects_to_test += S.nb_rows - 1;
		if (f_v) {
			cout << "projective_space_activity::set_stabilizer "
					"file " << cnt << " / " << nb << " has  "
					<< S.nb_rows - 1 << " objects" << endl;
		}

	}

	if (f_v) {
		cout << "projective_space_activity::set_stabilizer "
				"nb_objects_to_test = " << nb_objects_to_test << endl;
	}



	for (cnt = 0; cnt < nb; cnt++) {

		char str[1000];
		string fname;

		sprintf(str, fname_mask.c_str(), cnt);
		fname.assign(str);

		spreadsheet S;

		S.read_spreadsheet(fname, verbose_level);

		if (f_v) {
			cout << "projective_space_activity::set_stabilizer S.nb_rows = " << S.nb_rows << endl;
			cout << "projective_space_activity::set_stabilizer S.nb_cols = " << S.nb_cols << endl;
		}

		int j, t;
		string eqn_txt;
		string pts_txt;
		string bitangents_txt;
		int *eqn;
		int sz;
		long int *pts;
		int nb_pts;
		long int *canonical_pts;
		long int *bitangents;
		int nb_bitangents;



		for (row = 0; row < S.nb_rows - 1; row++) {

			if (f_v) {
				cout << "#############################################################################" << endl;
				cout << "cnt = " << cnt << " / " << nb << " row = " << row << " / " << S.nb_rows - 1 << endl;
			}

			j = 1;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "projective_space_activity::set_stabilizer token[t] == NULL" << endl;
			}
			eqn_txt.assign(S.tokens[t]);
			j = 2;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "projective_space_activity::set_stabilizer token[t] == NULL" << endl;
			}
			pts_txt.assign(S.tokens[t]);
			j = 3;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "projective_space_activity::set_stabilizer token[t] == NULL" << endl;
			}
			bitangents_txt.assign(S.tokens[t]);

			string_tools ST;

			ST.remove_specific_character(eqn_txt, '\"');
			ST.remove_specific_character(pts_txt, '\"');
			ST.remove_specific_character(bitangents_txt, '\"');

			if (FALSE) {
				cout << "row = " << row << " eqn=" << eqn_txt << " pts_txt=" << pts_txt << " =" << bitangents_txt << endl;
			}

			Orbiter->Int_vec.scan(eqn_txt, eqn, sz);
			Orbiter->Lint_vec.scan(pts_txt, pts, nb_pts);
			Orbiter->Lint_vec.scan(bitangents_txt, bitangents, nb_bitangents);

			canonical_pts = NEW_lint(nb_pts);


			if (f_v) {
				cout << "row = " << row << " eqn=";
				Orbiter->Int_vec.print(cout, eqn, sz);
				//cout << " pts=";
				//Orbiter->Lint_vec.print(cout, pts, nb_pts);
				//cout << " bitangents=";
				//Orbiter->Lint_vec.print(cout, bitangents, nb_bitangents);
				cout << endl;
			}

			int nCk;
			int *isotype;
			int *orbit_frequencies;
			int nb_orbits;
			tally *T;

			if (f_v) {
				cout << "projective_space_activity::set_stabilizer before PC->trace_all_k_subsets_and_compute_frequencies" << endl;
			}

			PC->trace_all_k_subsets_and_compute_frequencies(
					pts, nb_pts, intermediate_subset_size, nCk, isotype, orbit_frequencies, nb_orbits,
					0 /*verbose_level*/);

			if (f_v) {
				cout << "projective_space_activity::set_stabilizer after PC->trace_all_k_subsets_and_compute_frequencies" << endl;
			}




			T = NEW_OBJECT(tally);

			T->init(orbit_frequencies, nb_orbits, FALSE, 0);


			if (f_v) {
				cout << "cnt = " << cnt << " / " << nb << ", row = " << row << " eqn=";
				Orbiter->Int_vec.print(cout, eqn, sz);
				cout << " pts=";
				Orbiter->Lint_vec.print(cout, pts, nb_pts);
				cout << endl;
				cout << "orbit isotype=";
				Orbiter->Int_vec.print(cout, isotype, nCk);
				cout << endl;
				cout << "orbit frequencies=";
				Orbiter->Int_vec.print(cout, orbit_frequencies, nb_orbits);
				cout << endl;
				cout << "orbit frequency types=";
				T->print_naked(FALSE /* f_backwards */);
				cout << endl;
			}

			set_of_sets *SoS;
			int *types;
			int nb_types;
			int i, f, l, idx;
			int selected_type = -1;
			int selected_orbit = -1;
			int selected_frequency;
			longinteger_domain D;



			SoS = T->get_set_partition_and_types(types, nb_types, verbose_level);

			longinteger_object go_min;


			for (i = 0; i < nb_types; i++) {
				f = T->type_first[i];
				l = T->type_len[i];
				cout << types[i];
				cout << " : ";
				Orbiter->Lint_vec.print(cout, SoS->Sets[i], SoS->Set_size[i]);
				cout << " : ";


				for (j = 0; j < SoS->Set_size[i]; j++) {

					idx = SoS->Sets[i][j];

					longinteger_object go;

					PC->get_stabilizer_order(intermediate_subset_size, idx, go);

					if (types[i]) {

						// types[i] must be greater than zero
						// so the type really appears.

						if (selected_type == -1) {
							selected_type = j;
							selected_orbit = idx;
							selected_frequency = types[i];
							go.assign_to(go_min);
						}
						else {
							if (D.compare_unsigned(go, go_min) < 0) {
								selected_type = j;
								selected_orbit = idx;
								selected_frequency = types[i];
								go.assign_to(go_min);
							}
						}
					}

					cout << go;
					if (j < SoS->Set_size[i] - 1) {
						cout << ", ";
					}
				}
				cout << endl;
			}

			if (f_v) {
				cout << "selected_type = " << selected_type
					<< " selected_orbit = " << selected_orbit
					<< " selected_frequency = " << selected_frequency
					<< " go_min = " << go_min << endl;
			}

			strong_generators *gens;

			PC->get_stabilizer_generators(
				gens,
				intermediate_subset_size, selected_orbit, verbose_level);


			int *transporter_to_canonical_form;
			strong_generators *Gens_stabilizer_original_set;

			if (f_v) {
				cout << "projective_space_activity::set_stabilizer before handle_orbit" << endl;
			}

			transporter_to_canonical_form = NEW_int(PA->A->elt_size_in_int);


			handle_orbit(*T,
					isotype,
					selected_orbit, selected_frequency, nCk,
					intermediate_subset_size,
					PC, PA->A, PA->A,
					pts, nb_pts,
					canonical_pts,
					transporter_to_canonical_form,
					Gens_stabilizer_original_set,
					0 /*verbose_level*/);

			if (f_v) {
				cout << "projective_space_activity::set_stabilizer after handle_orbit" << endl;
				cout << "canonical point set: ";
				Orbiter->Lint_vec.print(cout, canonical_pts, nb_pts);
				longinteger_object go;

				Gens_stabilizer_original_set->group_order(go);
				cout << "_{" << go << "}" << endl;
				cout << endl;
				cout << "transporter to canonical form:" << endl;
				PA->A->element_print(transporter_to_canonical_form, cout);
				cout << "Stabilizer of the original set:" << endl;
				Gens_stabilizer_original_set->print_generators_tex();
			}

			strong_generators *Gens_stabilizer_canonical_form;

			Gens_stabilizer_canonical_form = NEW_OBJECT(strong_generators);

			if (f_v) {
				cout << "projective_space_activity::set_stabilizer before init_generators_for_the_conjugate_group_avGa" << endl;
			}
			Gens_stabilizer_canonical_form->init_generators_for_the_conjugate_group_avGa(
					Gens_stabilizer_original_set, transporter_to_canonical_form,
					verbose_level);
			if (f_v) {
				cout << "projective_space_activity::set_stabilizer after init_generators_for_the_conjugate_group_avGa" << endl;
			}

			if (f_v) {
				cout << "projective_space_activity::set_stabilizer after handle_orbit" << endl;
				cout << "canonical point set: ";
				Orbiter->Lint_vec.print(cout, canonical_pts, nb_pts);
				longinteger_object go;

				Gens_stabilizer_canonical_form->group_order(go);
				cout << "_{" << go << "}" << endl;
				cout << endl;
				cout << "transporter to canonical form:" << endl;
				PA->A->element_print(transporter_to_canonical_form, cout);
				cout << "Stabilizer of the canonical form:" << endl;
				Gens_stabilizer_canonical_form->print_generators_tex();
			}




			FREE_int(transporter_to_canonical_form);
			FREE_OBJECT(gens);
			FREE_OBJECT(Gens_stabilizer_original_set);
			FREE_OBJECT(Gens_stabilizer_canonical_form);
			FREE_OBJECT(SoS);
			FREE_int(types);

			FREE_int(isotype);
			FREE_int(orbit_frequencies);
			FREE_OBJECT(T);

			FREE_int(eqn);
			FREE_lint(pts);
			FREE_lint(bitangents);
			FREE_lint(canonical_pts);


		} // row

	}


	if (f_v) {
		cout << "projective_space_activity::set_stabilizer done" << endl;
	}

}


void projective_space_activity::handle_orbit(tally &C,
		int *isotype,
		int selected_orbit, int selected_frequency, int n_choose_k,
		int intermediate_subset_size,
		poset_classification *PC, action *A, action *A2,
		long int *pts,
		int nb_pts,
		long int *canonical_pts,
		int *transporter_to_canonical_form,
		strong_generators *&Gens_stabilizer_original_set,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	// interesting_subsets are the lvl-subsets of the given set
	// which are of the chosen type.
	// There is nb_interesting_subsets of them.
	long int *interesting_subsets;
	int nb_interesting_subsets;

	int i, j;

	if (f_v) {
		cout << "projective_space_activity::handle_orbit" << endl;
		cout << "selected_orbit = " << selected_orbit << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::handle_orbit we decide to go for subsets of size " << intermediate_subset_size << ", selected_frequency = " << selected_frequency << endl;
	}

	j = 0;
	interesting_subsets = NEW_lint(selected_frequency);
	for (i = 0; i < n_choose_k; i++) {
		if (isotype[i] == selected_orbit) {
			interesting_subsets[j++] = i;
			//cout << "subset of rank " << i << " is isomorphic to orbit " << orb_idx << " j=" << j << endl;
			}
		}
	if (j != selected_frequency) {
		cout << "j != nb_interesting_subsets" << endl;
		exit(1);
		}
	nb_interesting_subsets = selected_frequency;
#if 0
	if (f_vv) {
		print_interesting_subsets(nb_pts, intermediate_subset_size, nb_interesting_subsets, interesting_subsets);
		}
#endif


	//overall_backtrack_nodes = 0;
	if (f_v) {
		cout << "projective_space_activity::handle_orbit calling compute_stabilizer_function" << endl;
		}

	compute_stabilizer *CS;

	CS = NEW_OBJECT(compute_stabilizer);

	if (f_v) {
		cout << "projective_space_activity::handle_orbit before CS->init" << endl;
	}
	CS->init(pts, nb_pts,
			canonical_pts,
			PC, A, A2,
			intermediate_subset_size, selected_orbit,
			nb_interesting_subsets, interesting_subsets,
			verbose_level);
	if (f_v) {
		cout << "projective_space_activity::handle_orbit after CS->init" << endl;
	}


	A->element_move(CS->T1, transporter_to_canonical_form, 0);

	Gens_stabilizer_original_set = NEW_OBJECT(strong_generators);

	Gens_stabilizer_original_set->init_from_sims(CS->Stab, verbose_level);

	if (f_v) {
		cout << "projective_space_activity::handle_orbit done with compute_stabilizer" << endl;
		cout << "projective_space_activity::handle_orbit backtrack_nodes_first_time = " << CS->backtrack_nodes_first_time << endl;
		cout << "projective_space_activity::handle_orbit backtrack_nodes_total_in_loop = " << CS->backtrack_nodes_total_in_loop << endl;
		}


	FREE_OBJECT(CS);

	//overall_backtrack_nodes += CS->nodes;

	FREE_lint(interesting_subsets);

	if (f_v) {
		cout << "projective_space_activity::handle_orbit done" << endl;
	}
}


void projective_space_activity::print_interesting_subsets(int set_size, int lvl, int nb_interesting_subsets, int *interesting_subsets)
{

	cout << "the ranks of the corresponding subsets are:" << endl;
	Orbiter->Int_vec.print(cout, interesting_subsets, nb_interesting_subsets);
	cout << endl;
	int set[1000];
	int i, j, ii;
	combinatorics_domain Combi;

	cout << "the interesting subsets are:" << endl;

	if (nb_interesting_subsets < 50) {
		for (i = 0; i < nb_interesting_subsets; i++) {

			j = interesting_subsets[i];
			Combi.unrank_k_subset(j, set, set_size, lvl);
			cout << setw(3) << i << " : " << setw(6) << j << " : (";
			for (ii = 0; ii < lvl; ii++) {
				cout << setw(3) << set[ii];
				if (ii < lvl - 1)
					cout << ", ";
				}
			//INT_vec_print(cout, set, lvl);
			cout << ")" << endl;
#if 0
			cout << " : (";
			for (ii = 0; ii < lvl; ii++) {
				cout << setw(6) << the_set[set[ii]];
				if (ii < lvl - 1)
					cout << ", ";
				}
			cout << ") : (";
			for (ii = 0; ii < lvl; ii++) {
				A->print_point(the_set[set[ii]], cout);
				if (ii < lvl - 1)
					cout << ", ";
				}
			cout << ")" << endl;
#endif
			}
		}
	else {
		cout << "Too many to print" << endl;
		}
}


void projective_space_activity::conic_type(
		projective_space_with_action *PA,
		std::string &set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_activity::conic_type" << endl;
	}

	long int *Pts;
	int nb_pts;
	long int **Pts_on_conic;
	int **Conic_eqn;
	int *nb_pts_on_conic;
	int len;
	int h;

	Orbiter->Lint_vec.scan(set_text, Pts, nb_pts);

	if (f_v) {
		cout << "projective_space_activity::conic_type before PA->P->conic_type" << endl;
	}

	PA->P->conic_type(Pts, nb_pts,
			Pts_on_conic, Conic_eqn, nb_pts_on_conic, len,
			verbose_level);

	if (f_v) {
		cout << "projective_space_activity::conic_type after PA->P->conic_type" << endl;
	}


	cout << "We found the following conics:" << endl;
	for (h = 0; h < len; h++) {
		cout << h << " : " << nb_pts_on_conic[h] << " : ";
		Orbiter->Int_vec.print(cout, Conic_eqn[h], 6);
		cout << " : ";
		Orbiter->Lint_vec.print(cout, Pts_on_conic[h], nb_pts_on_conic[h]);
		cout << endl;
	}

	cout << "computing intersection types with bisecants of the first 11 points:" << endl;
	int Line_P1[55];
	int Line_P2[55];
	int P1, P2;
	long int p1, p2, line_rk;
	long int *pts_on_line;
	long int pt;
	int *Conic_line_intersection_sz;
	int cnt;
	int i, j, q, u, v;
	int nb_pts_per_line;

	q = PA->P->F->q;
	nb_pts_per_line = q + 1;
	pts_on_line = NEW_lint(55 * nb_pts_per_line);

	cnt = 0;
	for (i = 0; i < 11; i++) {
		for (j = i + 1; j < 11; j++) {
			Line_P1[cnt] = i;
			Line_P2[cnt] = j;
			cnt++;
		}
	}
	if (cnt != 55) {
		cout << "cnt != 55" << endl;
		cout << "cnt = " << cnt << endl;
		exit(1);
	}
	for (u = 0; u < 55; u++) {
		P1 = Line_P1[u];
		P2 = Line_P2[u];
		p1 = Pts[P1];
		p2 = Pts[P2];
		line_rk = PA->P->line_through_two_points(p1, p2);
		PA->P->create_points_on_line(line_rk, pts_on_line + u * nb_pts_per_line, 0 /*verbose_level*/);
	}

	Conic_line_intersection_sz = NEW_int(len * 55);
	Orbiter->Int_vec.zero(Conic_line_intersection_sz, len * 55);

	for (h = 0; h < len; h++) {
		for (u = 0; u < 55; u++) {
			for (v = 0; v < nb_pts_per_line; v++) {
				if (PA->P->test_if_conic_contains_point(Conic_eqn[h], pts_on_line[u * nb_pts_per_line + v])) {
					Conic_line_intersection_sz[h * 55 + u]++;
				}

			}
		}
	}

	sorting Sorting;
	int idx;

	cout << "We found the following conics and their intersections with the 55 bisecants:" << endl;
	for (h = 0; h < len; h++) {
		cout << h << " : " << nb_pts_on_conic[h] << " : ";
		Orbiter->Int_vec.print(cout, Conic_eqn[h], 6);
		cout << " : ";
		Orbiter->Int_vec.print_fully(cout, Conic_line_intersection_sz + h * 55, 55);
		cout << " : ";
		Orbiter->Lint_vec.print(cout, Pts_on_conic[h], nb_pts_on_conic[h]);
		cout << " : ";
		cout << endl;
	}

	for (u = 0; u < 55; u++) {
		cout << "line " << u << " : ";
		int str[55];

		Orbiter->Int_vec.zero(str, 55);
		for (v = 0; v < nb_pts; v++) {
			pt = Pts[v];
			if (Sorting.lint_vec_search_linear(pts_on_line + u * nb_pts_per_line, nb_pts_per_line, pt, idx)) {
				str[v] = 1;
			}
		}
		Orbiter->Int_vec.print_fully(cout, str, 55);
		cout << endl;
	}



	if (f_v) {
		cout << "projective_space_activity::conic_type done" << endl;
	}
}



}}
