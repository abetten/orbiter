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

	else if (Descr->f_define_object) {
		cout << "-define_object " << Descr->define_object_label << endl;
		//Object_Descr->print();

		combinatorial_object_create *CombObj;

		CombObj = NEW_OBJECT(combinatorial_object_create);

		CombObj->init(Descr->Object_Descr, PA->P, verbose_level);

		orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(orbiter_symbol_table_entry);

		Symb->init_combinatorial_object(Descr->define_object_label, CombObj, verbose_level);
		if (f_v) {
			cout << "before Orbiter->add_symbol_table_entry " << Descr->define_surface_label << endl;
		}
		Orbiter->add_symbol_table_entry(Descr->define_surface_label, Symb, verbose_level);

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

	else if (Descr->f_table_of_quartic_curves) {

		cout << "table_of_quartic_curves" << endl;


		table_of_quartic_curves(PA, verbose_level);
	}

	else if (Descr->f_table_of_cubic_surfaces) {

		cout << "table_of_cubic_surfaces" << endl;


		table_of_cubic_surfaces(PA, verbose_level);
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

#if 0
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
#endif
	else if (Descr->f_six_arcs_not_on_conic) {
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
				Descr->set_stabilizer_fname_mask, Descr->set_stabilizer_nb, Descr->set_stabilizer_column_label,
				verbose_level);
	}

	else if (Descr->f_conic_type) {

		conic_type(PA,
				Descr->conic_type_set_text,
				verbose_level);
	}

	else if (Descr->f_lift_skew_hexagon) {

		do_lift_skew_hexagon(PA,
				Descr->lift_skew_hexagon_text,
				verbose_level);
	}

	else if (Descr->f_lift_skew_hexagon_with_polarity) {

		do_lift_skew_hexagon_with_polarity(PA,
				Descr->lift_skew_hexagon_with_polarity_polarity,
				verbose_level);
	}
	else if (Descr->f_arc_with_given_set_as_s_lines_after_dualizing) {
		if (f_v) {
			cout << "perform_job_for_one_set f_arc_with_given_set_as_i_lines_after_dualizing" << endl;
		}
		diophant *D = NULL;
		int f_save_system = TRUE;

		long int *the_set_in;
		int set_size_in;

		Orbiter->Lint_vec.scan(Descr->arc_input_set, the_set_in, set_size_in);

		PA->P->arc_with_given_set_of_s_lines_diophant(
				the_set_in /*one_lines*/, set_size_in /* nb_one_lines */,
				Descr->arc_size /*target_sz*/, Descr->arc_d /* target_d */,
				Descr->arc_d_low, Descr->arc_s /* target_s */,
				TRUE /* f_dualize */,
				D,
				verbose_level);

		if (FALSE) {
			D->print_tight();
		}

		if (f_save_system) {

			string fname_system;

			fname_system.assign(Descr->arc_label);
			fname_system.append(".diophant");
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << endl;
			D->save_in_general_format(fname_system, 0 /* verbose_level */);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << " done" << endl;
			//D->print();
			//D->print_tight();
			}

		long int nb_backtrack_nodes;
		long int *Sol;
		int nb_sol;

		D->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level);

		if (f_v) {
			cout << "before D->get_solutions" << endl;
			}
		D->get_solutions(Sol, nb_sol, verbose_level);
		if (f_v) {
			cout << "after D->get_solutions, nb_sol=" << nb_sol << endl;
			}
		string fname_solutions;

		fname_solutions.assign(Descr->arc_label);
		fname_solutions.append(".solutions");

		{
			ofstream fp(fname_solutions);
			int i, j, a;

			for (i = 0; i < nb_sol; i++) {
				fp << D->sum;
				for (j = 0; j < D->sum; j++) {
					a = Sol[i * D->sum + j];
					fp << " " << a;
					}
				fp << endl;
				}
			fp << -1 << " " << nb_sol << endl;
		}
		file_io Fio;

		cout << "Written file " << fname_solutions << " of size "
				<< Fio.file_size(fname_solutions) << endl;
		FREE_lint(Sol);
		FREE_lint(the_set_in);


	}
	else if (Descr->f_arc_with_two_given_sets_of_lines_after_dualizing) {
		if (f_v) {
			cout << "perform_job_for_one_set f_arc_with_two_given_sets_of_lines_after_dualizing" << endl;
		}

		long int *t_lines;
		int nb_t_lines;

		Orbiter->Lint_vec.scan(Descr->t_lines_string, t_lines, nb_t_lines);

		cout << "The t-lines, t=" << Descr->arc_t << " are ";
		Orbiter->Lint_vec.print(cout, t_lines, nb_t_lines);
		cout << endl;


		long int *the_set_in;
		int set_size_in;

		Orbiter->Lint_vec.scan(Descr->arc_input_set, the_set_in, set_size_in);


		diophant *D = NULL;
		int f_save_system = TRUE;

		PA->P->arc_with_two_given_line_sets_diophant(
				the_set_in /* s_lines */, set_size_in /* nb_s_lines */, Descr->arc_s,
				t_lines, nb_t_lines, Descr->arc_t,
				Descr->arc_size /*target_sz*/, Descr->arc_d, Descr->arc_d_low,
				TRUE /* f_dualize */,
				D,
				verbose_level);

		if (FALSE) {
			D->print_tight();
		}

		if (f_save_system) {
			string fname_system;

			fname_system.assign(Descr->arc_label);
			fname_system.append(".diophant");

			//sprintf(fname_system, "system_%d.diophant", back_end_counter);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << endl;
			D->save_in_general_format(fname_system, 0 /* verbose_level */);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << " done" << endl;
			//D->print();
		}

		long int nb_backtrack_nodes;
		long int *Sol;
		int nb_sol;

		D->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level);

		if (f_v) {
			cout << "before D->get_solutions" << endl;
		}
		D->get_solutions(Sol, nb_sol, verbose_level);
		if (f_v) {
			cout << "after D->get_solutions, nb_sol=" << nb_sol << endl;
		}
		string fname_solutions;

		fname_solutions.assign(Descr->arc_label);
		fname_solutions.append(".solutions");

		{
			ofstream fp(fname_solutions);
			int i, j, a;

			for (i = 0; i < nb_sol; i++) {
				fp << D->sum;
				for (j = 0; j < D->sum; j++) {
					a = Sol[i * D->sum + j];
					fp << " " << a;
				}
				fp << endl;
			}
			fp << -1 << " " << nb_sol << endl;
		}

		file_io Fio;

		cout << "Written file " << fname_solutions << " of size "
				<< Fio.file_size(fname_solutions) << endl;
		FREE_lint(Sol);
		FREE_lint(the_set_in);


	}
	else if (Descr->f_arc_with_three_given_sets_of_lines_after_dualizing) {
		if (f_v) {
			cout << "perform_job_for_one_set f_arc_with_three_given_sets_of_lines_after_dualizing" << endl;
		}
		//int arc_size;
		//int arc_d;
		diophant *D = NULL;
		int f_save_system = TRUE;

		long int *t_lines;
		int nb_t_lines;
		long int *u_lines;
		int nb_u_lines;


		Orbiter->Lint_vec.scan(Descr->t_lines_string, t_lines, nb_t_lines);
		Orbiter->Lint_vec.scan(Descr->u_lines_string, u_lines, nb_u_lines);
		//lint_vec_print(cout, t_lines, nb_t_lines);
		//cout << endl;

		cout << "The t-lines, t=" << Descr->arc_t << " are ";
		Orbiter->Lint_vec.print(cout, t_lines, nb_t_lines);
		cout << endl;
		cout << "The u-lines, u=" << Descr->arc_u << " are ";
		Orbiter->Lint_vec.print(cout, u_lines, nb_u_lines);
		cout << endl;


		long int *the_set_in;
		int set_size_in;

		Orbiter->Lint_vec.scan(Descr->arc_input_set, the_set_in, set_size_in);


		PA->P->arc_with_three_given_line_sets_diophant(
				the_set_in /* s_lines */, set_size_in /* nb_s_lines */, Descr->arc_s,
				t_lines, nb_t_lines, Descr->arc_t,
				u_lines, nb_u_lines, Descr->arc_u,
				Descr->arc_size /*target_sz*/, Descr->arc_d, Descr->arc_d_low,
				TRUE /* f_dualize */,
				D,
				verbose_level);

		if (FALSE) {
			D->print_tight();
		}
		if (f_save_system) {
			string fname_system;

			fname_system.assign(Descr->arc_label);
			fname_system.append(".diophant");

			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << endl;
			D->save_in_general_format(fname_system, 0 /* verbose_level */);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << " done" << endl;
			//D->print();
			//D->print_tight();
		}

		long int nb_backtrack_nodes;
		long int *Sol;
		int nb_sol;

		D->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level);

		if (f_v) {
			cout << "before D->get_solutions" << endl;
		}
		D->get_solutions(Sol, nb_sol, verbose_level);
		if (f_v) {
			cout << "after D->get_solutions, nb_sol=" << nb_sol << endl;
		}
		string fname_solutions;

		fname_solutions.assign(Descr->arc_label);
		fname_solutions.append(".solutions");

		{
			ofstream fp(fname_solutions);
			int i, j, a;

			for (i = 0; i < nb_sol; i++) {
				fp << D->sum;
				for (j = 0; j < D->sum; j++) {
					a = Sol[i * D->sum + j];
					fp << " " << a;
				}
				fp << endl;
			}
			fp << -1 << " " << nb_sol << endl;
		}
		file_io Fio;

		cout << "Written file " << fname_solutions << " of size "
				<< Fio.file_size(fname_solutions) << endl;
		FREE_lint(Sol);
		FREE_lint(the_set_in);


	}
	else if (Descr->f_dualize_hyperplanes_to_points) {
		if (f_v) {
			cout << "projective_space_job_description::perform_job_for_one_set f_dualize_hyperplanes_to_points" << endl;
		}
		long int *the_set_in;
		int set_size_in;
		long int *the_set_out;
		int set_size_out;

		Orbiter->Lint_vec.scan(Descr->dualize_input_set, the_set_in, set_size_in);

		int i;
		long int a;

		set_size_out = set_size_in;
		the_set_out = NEW_lint(set_size_in);
		for (i = 0; i < set_size_in; i++) {
			a = the_set_in[i];
			the_set_out[i] = PA->P->Polarity_hyperplane_to_point[a];
		}

		// only if n = 2:
		//int *Polarity_point_to_hyperplane; // [N_points]
		//int *Polarity_hyperplane_to_point; // [N_points]

		FREE_lint(the_set_in);
		FREE_lint(the_set_out);

	}
	else if (Descr->f_dualize_points_to_hyperplanes) {
		if (f_v) {
			cout << "projective_space_job_description::perform_job_for_one_set f_dualize_points_to_hyperplanes" << endl;
		}
		long int *the_set_in;
		int set_size_in;
		long int *the_set_out;
		int set_size_out;

		Orbiter->Lint_vec.scan(Descr->dualize_input_set, the_set_in, set_size_in);

		int i;
		long int a;

		set_size_out = set_size_in;
		the_set_out = NEW_lint(set_size_in);
		for (i = 0; i < set_size_in; i++) {
			a = the_set_in[i];
			the_set_out[i] = PA->P->Polarity_point_to_hyperplane[a];
		}

		FREE_lint(the_set_in);
		FREE_lint(the_set_out);

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

	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code before PA->canonical_form_of_code" << endl;
	}
	PA->canonical_form_of_code(
				label, m, n,
				data,
				verbose_level);
	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code after PA->canonical_form_of_code" << endl;
	}


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


void projective_space_activity::table_of_quartic_curves(
		projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::table_of_quartic_curves" << endl;
	}

	PA->table_of_quartic_curves(verbose_level);

	if (f_v) {
		cout << "projective_space_activity::table_of_quartic_curves done" << endl;
	}
}

void projective_space_activity::table_of_cubic_surfaces(
		projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::table_of_cubic_surfaces" << endl;
	}

	PA->table_of_cubic_surfaces(verbose_level);

	if (f_v) {
		cout << "projective_space_activity::table_of_cubic_surfaces done" << endl;
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

	PA->do_spread_classify(k,
			Control,
			verbose_level);

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


	PA->cheat_sheet(O, verbose_level);


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
		std::string &fname_mask, int nb, std::string &column_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_activity::set_stabilizer" << endl;
	}

#if 0
	top_level_geometry_global T;

	T.set_stabilizer_projective_space(
				PA,
				intermediate_subset_size,
				fname_mask, nb, column_label,
				verbose_level);
#endif
	substructure_classifier *SubC;

	SubC = NEW_OBJECT(substructure_classifier);

	SubC->set_stabilizer_in_any_space(
			PA->A, PA->A, PA->A->Strong_gens,
			intermediate_subset_size,
			fname_mask, nb, column_label,
			verbose_level);
	FREE_OBJECT(SubC);

	if (f_v) {
		cout << "projective_space_activity::set_stabilizer done" << endl;
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

	Orbiter->Lint_vec.scan(set_text, Pts, nb_pts);


	if (f_v) {
		cout << "projective_space_activity::conic_type before PA->conic_type" << endl;
	}

	PA->conic_type(Pts, nb_pts, verbose_level);

	if (f_v) {
		cout << "projective_space_activity::conic_type after PA->conic_type" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::conic_type done" << endl;
	}
}

void projective_space_activity::do_lift_skew_hexagon(
		projective_space_with_action *PA,
		std::string &text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon" << endl;
	}

	int *Pluecker_coords;
	int sz;

	Orbiter->Int_vec.scan(text, Pluecker_coords, sz);

	long int *Pts;
	int nb_pts;

	nb_pts = sz / 6;

	if (nb_pts * 6 != sz) {
		cout << "projective_space_activity::do_lift_skew_hexagon the number of coordinates must be a multiple of 6" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "Pluecker coordinates of lines:" << endl;
		Orbiter->Int_vec.matrix_print(Pluecker_coords, nb_pts, 6);
	}

	surface_domain *Surf;
	surface_with_action *Surf_A;

	if (PA->n != 3) {
		cout << "projective_space_activity::do_lift_skew_hexagon we need a three-dimensional projective space" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon after Surf_A->init" << endl;
	}




	int i;

	Pts = NEW_lint(nb_pts);

	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Surf_A->Surf->Klein->Pluecker_to_line_rk(Pluecker_coords + i * 6, 0 /*verbose_level*/);
	}

	if (nb_pts != 6) {
		cout << "projective_space_activity::do_lift_skew_hexagon nb_pts != 6" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "lines:" << endl;
		Orbiter->Lint_vec.print(cout, Pts, 6);
		cout << endl;
	}


	std::vector<std::vector<long int> > Double_sixes;

	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon before Surf_A->complete_skew_hexagon" << endl;
	}

	Surf_A->complete_skew_hexagon(Pts, Double_sixes, verbose_level);

	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon after Surf_A->complete_skew_hexagon" << endl;
	}

	cout << "We found " << Double_sixes.size() << " double sixes. They are:" << endl;
	for (i = 0; i < Double_sixes.size(); i++) {
		cout << Double_sixes[i][0] << ",";
		cout << Double_sixes[i][1] << ",";
		cout << Double_sixes[i][2] << ",";
		cout << Double_sixes[i][3] << ",";
		cout << Double_sixes[i][4] << ",";
		cout << Double_sixes[i][5] << ",";
		cout << Double_sixes[i][6] << ",";
		cout << Double_sixes[i][7] << ",";
		cout << Double_sixes[i][8] << ",";
		cout << Double_sixes[i][9] << ",";
		cout << Double_sixes[i][10] << ",";
		cout << Double_sixes[i][11] << "," << endl;

	}

	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon done" << endl;
	}
}


void projective_space_activity::do_lift_skew_hexagon_with_polarity(
		projective_space_with_action *PA,
		std::string &polarity_36,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon_with_polarity" << endl;
	}

	int *Polarity36;
	int sz1;

	Orbiter->Int_vec.scan(polarity_36, Polarity36, sz1);

	if (sz1 != 36) {
		cout << "projective_space_activity::do_lift_skew_hexagon_with_polarity I need exactly 36 coefficients for the polarity" << endl;
		exit(1);
	}


	surface_domain *Surf;
	surface_with_action *Surf_A;

	if (PA->n != 3) {
		cout << "projective_space_activity::do_lift_skew_hexagon_with_polarity we need a three-dimensional projective space" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon_with_polarity before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon_with_polarity after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon_with_polarity before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_activity::do_lift_skew_hexagon_with_polarity after Surf_A->init" << endl;
	}




	std::vector<std::vector<long int> > Double_sixes;

	int Pluecker_coords[36];
	int alpha, beta;
	int i, j;

	Orbiter->Int_vec.zero(Pluecker_coords, 36);
	// a1 = 1,0,0,0,0,0
	Pluecker_coords[0] = 1;

	for (alpha = 1; alpha < PA->F->q; alpha++) {



		for (beta = 1; beta < PA->F->q; beta++) {

			// a2 = 0,beta,0,alpha,alpha,0

			Pluecker_coords[6 + 1] = beta;
			Pluecker_coords[6 + 3] = alpha;
			Pluecker_coords[6 + 4] = alpha;

			// a3 = 0,beta,0,alpha,alpha,0

			Pluecker_coords[12 + 1] = alpha;
			Pluecker_coords[12 + 2] = beta;


			for (j = 0; j < 3; j++) {
				Surf->F->mult_matrix_matrix(Pluecker_coords + j * 6, Polarity36,
						Pluecker_coords + 18 + j * 6, 1, 6, 6, 0 /* verbose_level */);
			}

			int nb_pts;

			nb_pts = 6;

			if (f_v) {
				cout << "Pluecker coordinates of lines:" << endl;
				Orbiter->Int_vec.matrix_print(Pluecker_coords, nb_pts, 6);
			}


			long int *Pts;


			Pts = NEW_lint(nb_pts);

			for (i = 0; i < nb_pts; i++) {
				Pts[i] = Surf_A->Surf->Klein->Pluecker_to_line_rk(Pluecker_coords + i * 6, 0 /*verbose_level*/);
			}

			if (nb_pts != 6) {
				cout << "projective_space_activity::do_lift_skew_hexagon_with_polarity nb_pts != 6" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "lines:" << endl;
				Orbiter->Lint_vec.print(cout, Pts, 6);
				cout << endl;
			}


			string label;
			char str[1000];

			sprintf(str, "alpha=%d beta=%d", alpha, beta);

			label.assign(str);

			if (f_v) {
				cout << "projective_space_activity::do_lift_skew_hexagon_with_polarity before Surf_A->complete_skew_hexagon_with_polarity" << endl;
			}

			Surf_A->complete_skew_hexagon_with_polarity(label, Pts, Polarity36, Double_sixes, verbose_level);

			if (f_v) {
				cout << "projective_space_activity::do_lift_skew_hexagon_with_polarity after Surf_A->complete_skew_hexagon_with_polarity" << endl;
			}

			FREE_lint(Pts);


		}

	}



	cout << "We found " << Double_sixes.size() << " double sixes. They are:" << endl;
	for (i = 0; i < Double_sixes.size(); i++) {
		cout << Double_sixes[i][0] << ",";
		cout << Double_sixes[i][1] << ",";
		cout << Double_sixes[i][2] << ",";
		cout << Double_sixes[i][3] << ",";
		cout << Double_sixes[i][4] << ",";
		cout << Double_sixes[i][5] << ",";
		cout << Double_sixes[i][6] << ",";
		cout << Double_sixes[i][7] << ",";
		cout << Double_sixes[i][8] << ",";
		cout << Double_sixes[i][9] << ",";
		cout << Double_sixes[i][10] << ",";
		cout << Double_sixes[i][11] << "," << endl;

	}

	if (f_v) {
		cout << "projective_space_activity::do_lift_do_lift_skew_hexagon_with_polarityskew_hexagon done" << endl;
	}
}


}}
