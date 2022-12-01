/*
 * projective_space_activity.cpp
 *
 *  Created on: Jan 5, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace projective_geometry {


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

	if (Descr->f_export_point_line_incidence_matrix) {

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before PA->P->export_incidence_matrix_to_csv" << endl;
		}
		PA->P->export_incidence_matrix_to_csv(verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after PA->P->export_incidence_matrix_to_csv" << endl;
		}
	}


	else if (Descr->f_table_of_cubic_surfaces_compute_properties) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before SH.do_cubic_surface_properties" << endl;
		}
		SH.do_cubic_surface_properties(
				PA,
				Descr->table_of_cubic_surfaces_compute_fname_csv,
				Descr->table_of_cubic_surfaces_compute_defining_q,
				Descr->table_of_cubic_surfaces_compute_column_offset,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after SH.do_cubic_surface_properties" << endl;
		}
	}
	else if (Descr->f_cubic_surface_properties_analyze) {


		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before SH.do_cubic_surface_properties_analyze" << endl;
		}
		SH.do_cubic_surface_properties_analyze(
				PA,
				Descr->cubic_surface_properties_fname_csv,
				Descr->cubic_surface_properties_defining_q,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after SH.do_cubic_surface_properties_analyze" << endl;
		}
	}
	else if (Descr->f_canonical_form_of_code) {


		int *genma;
		int k, n;

		Get_matrix(Descr->canonical_form_of_code_generator_matrix, genma, k, n);

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before PA->canonical_form_of_code" << endl;
		}
		PA->canonical_form_of_code(
				Descr->canonical_form_of_code_label,
					genma, k, n,
					Descr->Canonical_form_codes_Descr,
					verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after PA->canonical_form_of_code" << endl;
		}


	}
	else if (Descr->f_map) {

		projective_space_global G;

		if (f_v) {
			cout << "projective_space_activity::perform_activity before G.map" << endl;
		}
		if (f_v) {
			cout << "projective_space_activity::perform_activity n=" << PA->P->n << endl;
		}

		long int *Image_pts;
		int N_points;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before G.map" << endl;
		}
		G.map(
				PA,
				Descr->map_ring_label,
				Descr->map_formula_label,
				Descr->map_parameters,
				Image_pts, N_points,
				verbose_level);

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after G.map" << endl;
		}

		if (f_v) {
			cout << "projective_space_activity::perform_activity permutation:" << endl;
			Lint_vec_print(cout, Image_pts, N_points);
			cout << endl;
		}

		string fname_map;
		orbiter_kernel_system::file_io Fio;

		fname_map.assign(Descr->map_formula_label);
		fname_map.append("_map.csv");


		Fio.lint_matrix_write_csv(fname_map, Image_pts, N_points, 1);
		if (f_v) {
			cout << "Written file " << fname_map
					<< " of size " << Fio.file_size(fname_map) << endl;
		}



	}
	else if (Descr->f_analyze_del_Pezzo_surface) {

		projective_space_global G;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before G.analyze_del_Pezzo_surface" << endl;
		}
		G.analyze_del_Pezzo_surface(
				PA,
				Descr->analyze_del_Pezzo_surface_label,
				Descr->analyze_del_Pezzo_surface_parameters,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after G.analyze_del_Pezzo_surface" << endl;
		}

	}

	else if (Descr->f_cheat_sheet_for_decomposition_by_element_PG) {

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before PA->do_cheat_sheet_for_decomposition_by_element_PG" << endl;
		}
		PA->do_cheat_sheet_for_decomposition_by_element_PG(
				Descr->decomposition_by_element_power,
				Descr->decomposition_by_element_data,
				Descr->decomposition_by_element_fname,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after PA->do_cheat_sheet_for_decomposition_by_element_PG" << endl;
		}

	}

	else if (Descr->f_decomposition_by_subgroup) {

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before PA->do_cheat_sheet_for_decomposition_by_subgroup" << endl;
		}
		PA->do_cheat_sheet_for_decomposition_by_subgroup(
				Descr->decomposition_by_subgroup_label,
				Descr->decomposition_by_subgroup_Descr,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after PA->do_cheat_sheet_for_decomposition_by_subgroup" << endl;
		}

	}



	else if (Descr->f_table_of_quartic_curves) {

		cout << "table_of_quartic_curves" << endl;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before PA->table_of_quartic_curves" << endl;
		}
		PA->table_of_quartic_curves(verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after PA->table_of_quartic_curves" << endl;
		}

	}

	else if (Descr->f_table_of_cubic_surfaces) {

		cout << "table_of_cubic_surfaces" << endl;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before PA->table_of_cubic_surfaces" << endl;
		}
		PA->table_of_cubic_surfaces(verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after PA->table_of_cubic_surfaces" << endl;
		}
	}


	// surfaces:


	else if (Descr->f_classify_surfaces_with_double_sixes) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;
		applications_in_algebraic_geometry::cubic_surfaces_and_double_sixes::surface_classify_wedge *SCW;


		SH.classify_surfaces_with_double_sixes(
				PA,
				Descr->classify_surfaces_with_double_sixes_control_label,
				SCW,
				verbose_level);

		orbiter_kernel_system::orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(orbiter_kernel_system::orbiter_symbol_table_entry);

		Symb->init_classification_of_cubic_surfaces_with_double_sixes(Descr->classify_surfaces_with_double_sixes_label, SCW, verbose_level);
		if (f_v) {
			cout << "before Orbiter->add_symbol_table_entry "
					<< Descr->classify_surfaces_with_double_sixes_label << endl;
		}
		orbiter_kernel_system::Orbiter->add_symbol_table_entry(Descr->classify_surfaces_with_double_sixes_label, Symb, verbose_level);

	}

	else if (Descr->f_classify_surfaces_through_arcs_and_two_lines) {

		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before SH.do_classify_surfaces_through_arcs_and_two_lines" << endl;
		}
		SH.do_classify_surfaces_through_arcs_and_two_lines(
				PA,
				Descr->Control_six_arcs_label,
				Descr->f_test_nb_Eckardt_points, Descr->nb_E,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after SH.do_classify_surfaces_through_arcs_and_two_lines" << endl;
		}
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
			cout << "please use option -control_six_arcs <label> -end" << endl;
			exit(1);
		}
		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before SH.do_classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
		}
		SH.do_classify_surfaces_through_arcs_and_trihedral_pairs(
				PA,
				Descr->Trihedra1_control, Descr->Trihedra2_control,
				Descr->Control_six_arcs_label,
				Descr->f_test_nb_Eckardt_points, Descr->nb_E,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after SH.do_classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
		}
	}
	else if (Descr->f_sweep_4_15_lines) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;


		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before SH.do_sweep_4_15_lines" << endl;
		}
		SH.do_sweep_4_15_lines(
				PA,
				Descr->sweep_4_15_lines_surface_description,
				Descr->sweep_4_15_lines_fname,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after SH.do_sweep_4_15_lines" << endl;
		}
	}
	else if (Descr->f_sweep_F_beta_9_lines) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;


		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before SH.do_sweep_F_beta_9_lines" << endl;
		}
		SH.do_sweep_F_beta_9_lines(
				PA,
				Descr->sweep_F_beta_9_lines_surface_description,
				Descr->sweep_F_beta_9_lines_fname,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after SH.do_sweep_F_beta_9_lines" << endl;
		}
	}
	else if (Descr->f_sweep_6_9_lines) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;


		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before SH.do_sweep_6_9_lines" << endl;
		}
		SH.do_sweep_6_9_lines(
				PA,
				Descr->sweep_6_9_lines_surface_description,
				Descr->sweep_6_9_lines_fname,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after SH.do_sweep_6_9_lines" << endl;
		}
	}
	else if (Descr->f_sweep_4_27) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;


		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before SH.do_sweep_4_27" << endl;
		}
		SH.do_sweep_4_27(
				PA,
				Descr->sweep_4_27_surface_description,
				Descr->sweep_4_27_fname,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after SH.do_sweep_4_27" << endl;
		}
	}

	else if (Descr->f_sweep_4_L9_E4) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;


		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before SH.do_sweep_4_L9_E4" << endl;
		}
		SH.do_sweep_4_L9_E4(
				PA,
				Descr->sweep_4_L9_E4_surface_description,
				Descr->sweep_4_L9_E4_fname,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after SH.do_sweep_4_L9_E4" << endl;
		}
	}

	else if (Descr->f_six_arcs_not_on_conic) {
		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before SH.do_six_arcs" << endl;
		}
		SH.do_six_arcs(
				PA,
				Descr->Control_six_arcs_label,
				Descr->f_filter_by_nb_Eckardt_points,
				Descr->nb_Eckardt_points,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after SH.do_six_arcs" << endl;
		}
	}


	else if (Descr->f_classify_semifields) {

		semifields::semifield_classify_with_substructure *S;

		S = NEW_OBJECT(semifields::semifield_classify_with_substructure);

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before S->init" << endl;
		}
		S->init(
				Descr->Semifield_classify_description,
				PA,
				Descr->Semifield_classify_Control,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after S->init" << endl;
		}

	}
	else if (Descr->f_cheat_sheet) {

		graphics::layered_graph_draw_options *O;

		if (orbiter_kernel_system::Orbiter->f_draw_options) {
			O = orbiter_kernel_system::Orbiter->draw_options;
		}
		else {
			cout << "please use -draw_options .. -end" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before PA->cheat_sheet" << endl;
		}
		PA->cheat_sheet(O, verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after PA->cheat_sheet" << endl;
		}

	}
	else if (Descr->f_classify_quartic_curves_nauty) {

		canonical_form_classifier *Classifier;

		projective_space_global G;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before G.classify_quartic_curves_nauty" << endl;
		}
		G.classify_quartic_curves_nauty(PA,
				Descr->classify_quartic_curves_nauty_fname_mask,
				Descr->classify_quartic_curves_nauty_nb,
				Descr->classify_quartic_curves_nauty_fname_classification,
				Classifier,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after G.classify_quartic_curves_nauty" << endl;
		}
	}


	else if (Descr->f_classify_quartic_curves_with_substructure) {

		projective_space_global G;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before G.classify_quartic_curves" << endl;
		}
		G.classify_quartic_curves(
				PA,
				Descr->classify_quartic_curves_with_substructure_fname_mask,
				Descr->classify_quartic_curves_with_substructure_nb,
				Descr->classify_quartic_curves_with_substructure_size,
				Descr->classify_quartic_curves_with_substructure_degree,
				Descr->classify_quartic_curves_with_substructure_fname_classification,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after G.classify_quartic_curves" << endl;
		}

	}
	else if (Descr->f_set_stabilizer) {

		projective_space_global G;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before G.set_stabilizer" << endl;
		}
		G.set_stabilizer(PA,
				Descr->set_stabilizer_intermediate_set_size,
				Descr->set_stabilizer_fname_mask,
				Descr->set_stabilizer_nb,
				Descr->set_stabilizer_column_label,
				Descr->set_stabilizer_fname_out,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after G.set_stabilizer" << endl;
		}
	}

	else if (Descr->f_conic_type) {

		projective_space_global G;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before G.conic_type" << endl;
		}
		G.conic_type(PA,
				Descr->conic_type_threshold,
				Descr->conic_type_set_text,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after G.conic_type" << endl;
		}
	}

	else if (Descr->f_lift_skew_hexagon) {

		projective_space_global G;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before G.do_lift_skew_hexagon" << endl;
		}
		G.do_lift_skew_hexagon(PA,
				Descr->lift_skew_hexagon_text,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after G.do_lift_skew_hexagon" << endl;
		}
	}

	else if (Descr->f_lift_skew_hexagon_with_polarity) {

		projective_space_global G;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before G.do_lift_skew_hexagon_with_polarity" << endl;
		}
		G.do_lift_skew_hexagon_with_polarity(PA,
				Descr->lift_skew_hexagon_with_polarity_polarity,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after G.do_lift_skew_hexagon_with_polarity" << endl;
		}
	}
	else if (Descr->f_arc_with_given_set_as_s_lines_after_dualizing) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"f_arc_with_given_set_as_i_lines_after_dualizing" << endl;
		}

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before PA->P->Arc_in_projective_space->arc_lifting1" << endl;
		}
		PA->P->Arc_in_projective_space->arc_lifting1(
				Descr->arc_size,
				Descr->arc_d,
				Descr->arc_d_low,
				Descr->arc_s,
				Descr->arc_input_set,
				Descr->arc_label,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after PA->P->Arc_in_projective_space->arc_lifting1" << endl;
		}

	}
	else if (Descr->f_arc_with_two_given_sets_of_lines_after_dualizing) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"f_arc_with_two_given_sets_of_lines_after_dualizing" << endl;
		}

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before PA->P->Arc_in_projective_space->arc_lifting2" << endl;
		}
		PA->P->Arc_in_projective_space->arc_lifting2(
				Descr->arc_size,
				Descr->arc_d,
				Descr->arc_d_low,
				Descr->arc_s,
				Descr->arc_input_set,
				Descr->arc_label,
				Descr->arc_t,
				Descr->t_lines_string,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after PA->P->Arc_in_projective_space->arc_lifting2" << endl;
		}


	}
	else if (Descr->f_arc_with_three_given_sets_of_lines_after_dualizing) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"f_arc_with_three_given_sets_of_lines_after_dualizing" << endl;
		}

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before PA->P->Arc_in_projective_space->arc_lifting3" << endl;
		}
		PA->P->Arc_in_projective_space->arc_lifting3(
				Descr->arc_size,
				Descr->arc_d,
				Descr->arc_d_low,
				Descr->arc_s,
				Descr->arc_input_set,
				Descr->arc_label,
				Descr->arc_t,
				Descr->t_lines_string,
				Descr->arc_u,
				Descr->u_lines_string,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after PA->P->Arc_in_projective_space->arc_lifting3" << endl;
		}

	}
	else if (Descr->f_dualize_hyperplanes_to_points) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"f_dualize_hyperplanes_to_points" << endl;
		}
		long int *the_set_in;
		int set_size_in;
		long int *the_set_out;
		//int set_size_out;

		Lint_vec_scan(Descr->dualize_input_set, the_set_in, set_size_in);

		int i;
		long int a;

		//set_size_out = set_size_in;
		the_set_out = NEW_lint(set_size_in);
		for (i = 0; i < set_size_in; i++) {
			a = the_set_in[i];
			the_set_out[i] = PA->P->Standard_polarity->Hyperplane_to_point[a];
		}

		cout << "output set:" << endl;
		Lint_vec_print(cout, the_set_out, set_size_in);
		cout << endl;

		// only if n = 2:
		//int *Polarity_point_to_hyperplane; // [N_points]
		//int *Polarity_hyperplane_to_point; // [N_points]

		FREE_lint(the_set_in);
		FREE_lint(the_set_out);

	}
	else if (Descr->f_dualize_points_to_hyperplanes) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"f_dualize_points_to_hyperplanes" << endl;
		}
		long int *the_set_in;
		int set_size_in;
		long int *the_set_out;
		//int set_size_out;

		Lint_vec_scan(Descr->dualize_input_set, the_set_in, set_size_in);

		int i;
		long int a;

		//set_size_out = set_size_in;
		the_set_out = NEW_lint(set_size_in);
		for (i = 0; i < set_size_in; i++) {
			a = the_set_in[i];
			the_set_out[i] = PA->P->Standard_polarity->Point_to_hyperplane[a];
		}

		cout << "output set:" << endl;
		Lint_vec_print(cout, the_set_out, set_size_in);
		cout << endl;

		FREE_lint(the_set_in);
		FREE_lint(the_set_out);

	}
	else if (Descr->f_dualize_rank_k_subspaces) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"f_dualize_rank_k_subspaces" << endl;
		}
		long int *the_set_in;
		int set_size_in;
		long int *the_set_out;
		//int set_size_out;

		Lint_vec_scan(Descr->dualize_input_set, the_set_in, set_size_in);

		int i;
		long int a;

		//set_size_out = set_size_in;
		the_set_out = NEW_lint(set_size_in);
		for (i = 0; i < set_size_in; i++) {
			a = the_set_in[i];
			cout << "i=" << i << " in=" << a << endl;
			PA->P->polarity_rank_k_subspace(Descr->dualize_rank_k_subspaces_k,
					a, the_set_out[i], verbose_level);
			cout << "i=" << i << " in=" << a << " out=" << the_set_out[i] << endl;
		}

		cout << "output set:" << endl;
		Lint_vec_print(cout, the_set_out, set_size_in);
		cout << endl;

		FREE_lint(the_set_in);
		FREE_lint(the_set_out);

	}
	else if (Descr->f_classify_arcs) {
		projective_space_global G;

		if (f_v) {
			cout << "projective_space_activity::perform_activity before G.do_classify_arcs" << endl;
		}
		G.do_classify_arcs(
				PA,
				Descr->Arc_generator_description,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after G.do_classify_arcs" << endl;
		}
	}
	else if (Descr->f_classify_cubic_curves) {
		projective_space_global G;

		if (f_v) {
			cout << "projective_space_activity::perform_activity before G.do_classify_cubic_curves" << endl;
		}
		G.do_classify_cubic_curves(
				PA,
				Descr->Arc_generator_description,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after G.do_classify_cubic_curves" << endl;
		}
	}
	else if (Descr->f_lines_on_point_but_within_a_plane) {

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"f_lines_on_point_but_within_a_plane" << endl;
		}

		long int point_rk = Descr->lines_on_point_but_within_a_plane_point_rk;
		long int plane_rk = Descr->lines_on_point_but_within_a_plane_plane_rk;
		long int *line_pencil;
		int q;

		q = PA->F->q;
		line_pencil = NEW_lint(q + 1);

		if (f_v) {
			cout << "projective_space_activity::perform_activity before PA->P->create_lines_on_point_but_inside_a_plane" << endl;
		}
		PA->P->create_lines_on_point_but_inside_a_plane(
				point_rk, plane_rk,
				line_pencil, verbose_level);
			// assumes that line_pencil[q + 1] has been allocated
		if (f_v) {
			cout << "projective_space_activity::perform_activity after PA->P->create_lines_on_point_but_inside_a_plane" << endl;
		}

		cout << "line_pencil: ";
		Lint_vec_print(cout, line_pencil, q + 1);
		cout << endl;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"f_lines_on_point_but_within_a_plane done" << endl;
		}
	}
	else if (Descr->f_rank_lines_in_PG) {


		if (f_v) {
			cout << "projective_space_activity::perform_activity before PA->do_rank_lines_in_PG" << endl;
		}
		PA->do_rank_lines_in_PG(
				Descr->rank_lines_in_PG_label,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after PA->do_rank_lines_in_PG" << endl;
		}
	}
	else if (Descr->f_unrank_lines_in_PG) {


		if (f_v) {
			cout << "projective_space_activity::perform_activity before PA->do_unrank_lines_in_PG" << endl;
		}
		PA->do_unrank_lines_in_PG(
				Descr->unrank_lines_in_PG_text,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after PA->do_unrank_lines_in_PG" << endl;
		}
	}
	else if (Descr->f_move_two_lines_in_hyperplane_stabilizer) {


		geometry::geometry_global Geo;

		if (f_v) {
			cout << "projective_space_activity::perform_activity before Geo.do_move_two_lines_in_hyperplane_stabilizer" << endl;
		}
		Geo.do_move_two_lines_in_hyperplane_stabilizer(PA->P,
				Descr->line1_from, Descr->line2_from,
				Descr->line1_to, Descr->line2_to, verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after Geo.do_move_two_lines_in_hyperplane_stabilizer" << endl;
		}
	}
	else if (Descr->f_move_two_lines_in_hyperplane_stabilizer_text) {

		geometry::geometry_global Geo;

		if (f_v) {
			cout << "projective_space_activity::perform_activity before Geo.do_move_two_lines_in_hyperplane_stabilizer_text" << endl;
		}
		Geo.do_move_two_lines_in_hyperplane_stabilizer_text(PA->P,
				Descr->line1_from_text, Descr->line2_from_text,
				Descr->line1_to_text, Descr->line2_to_text,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after Geo.do_move_two_lines_in_hyperplane_stabilizer_text" << endl;
		}
	}
	else if (Descr->f_planes_through_line) {

		cout << "planes through line:" << endl;


		long int *Lines;
		int nb_lines;
		Lint_vec_scan(Descr->planes_through_line_rank, Lines, nb_lines);

		long int *Plane_ranks;
		int nb_planes_on_one_line;

		if (f_v) {
			cout << "projective_space_activity::perform_activity before PA->P->planes_through_line" << endl;
		}
		PA->P->planes_through_line(Lines, nb_lines,
				Plane_ranks, nb_planes_on_one_line, verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after PA->P->planes_through_line" << endl;
		}

		cout << "Ranks of planes on the given set of lines:" << endl;
		Lint_matrix_print(Plane_ranks, nb_lines, nb_planes_on_one_line);

		FREE_lint(Lines);
		FREE_lint(Plane_ranks);

	}
	else if (Descr->f_restricted_incidence_matrix) {

		cout << "f_restricted_incidence_matrix:" << endl;

		int type_i = Descr->restricted_incidence_matrix_type_row_objects;
		int type_j = Descr->restricted_incidence_matrix_type_col_objects;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before make_restricted_incidence_matrix" << endl;
		}

		projective_space_global G;


		if (f_v) {
			cout << "projective_space_activity::perform_activity before G.make_restricted_incidence_matrix" << endl;
		}
		G.make_restricted_incidence_matrix(
				PA,
				type_i, type_j,
				Descr->restricted_incidence_matrix_row_objects,
				Descr->restricted_incidence_matrix_col_objects,
				Descr->restricted_incidence_matrix_file_name,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after G.make_restricted_incidence_matrix" << endl;
		}

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after make_restricted_incidence_matrix" << endl;
		}


	}




	else if (Descr->f_make_relation) {

		cout << "f_make_relation:" << endl;


		projective_space_global G;

		if (f_v) {
			cout << "projective_space_activity::perform_activity before G.make_relation" << endl;
		}
		G.make_relation(
				PA,
				Descr->make_relation_plane_rk,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after G.make_relation" << endl;
		}


	}
	else if (Descr->f_plane_intersection_type_of_klein_image) {


		if (f_v) {
			cout << "projective_space_activity::perform_activity plane_intersection_type_of_klein_image:" << Descr->plane_intersection_type_of_klein_image_input << endl;
			cout << "projective_space_activity::perform_activity plane_intersection_type_of_klein_image_threshold = " << Descr->plane_intersection_type_of_klein_image_threshold << endl;
		}

		projective_space_global G;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"before G.plane_intersection_type_of_klein_image" << endl;
		}
		G.plane_intersection_type_of_klein_image(
				PA,
				Descr->plane_intersection_type_of_klein_image_input,
				Descr->plane_intersection_type_of_klein_image_threshold,
				verbose_level);

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"after G.plane_intersection_type_of_klein_image" << endl;
		}

	}



	if (f_v) {
		cout << "projective_space_activity::perform_activity done" << endl;
	}

}






}}}

