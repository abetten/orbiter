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

#if 0
	if (Descr->f_canonical_form_PG) {

		PA->canonical_form(
				Descr->Canonical_form_PG_Descr,
				verbose_level);
	}
#endif

	if (Descr->f_export_point_line_incidence_matrix) {

		PA->P->export_incidence_matrix_to_csv(verbose_level);
	}


	else if (Descr->f_table_of_cubic_surfaces_compute_properties) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

		SH.do_cubic_surface_properties(
				PA,
				Descr->table_of_cubic_surfaces_compute_fname_csv,
				Descr->table_of_cubic_surfaces_compute_defining_q,
				Descr->table_of_cubic_surfaces_compute_column_offset,
				verbose_level);
	}
	else if (Descr->f_cubic_surface_properties_analyze) {


		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

		SH.do_cubic_surface_properties_analyze(
				PA,
				Descr->cubic_surface_properties_fname_csv,
				Descr->cubic_surface_properties_defining_q,
				verbose_level);
	}
	else if (Descr->f_canonical_form_of_code) {

		//projective_space_global G;

		int *genma;
		int k, n;

		Orbiter->get_matrix_from_label(Descr->canonical_form_of_code_generator_matrix, genma, k, n);

#if 0
		G.canonical_form_of_code(
				PA,
				Descr->canonical_form_of_code_label,
				v, m, n,
				Descr->Canonical_form_codes_Descr,
				verbose_level);
#endif

		PA->canonical_form_of_code(
				Descr->canonical_form_of_code_label,
					genma, k, n,
					Descr->Canonical_form_codes_Descr,
					verbose_level);


	}
	else if (Descr->f_map) {

		projective_space_global G;

		G.map(
				PA,
				Descr->map_label,
				Descr->map_parameters,
				verbose_level);

	}
	else if (Descr->f_analyze_del_Pezzo_surface) {

		projective_space_global G;

		G.analyze_del_Pezzo_surface(
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

	else if (Descr->f_decomposition_by_subgroup) {

		PA->do_cheat_sheet_for_decomposition_by_subgroup(
				Descr->decomposition_by_subgroup_label,
				Descr->decomposition_by_subgroup_Descr,
				verbose_level);

	}


	else if (Descr->f_define_object) {
		cout << "-define_object " << Descr->define_object_label << endl;
		Descr->Object_Descr->print();

		geometric_object_create *GeoObj;

		GeoObj = NEW_OBJECT(geometric_object_create);

		GeoObj->init(Descr->Object_Descr, PA->P, verbose_level);

		orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(orbiter_symbol_table_entry);

		Symb->init_geometric_object(Descr->define_object_label, GeoObj, verbose_level);
		if (f_v) {
			cout << "before Orbiter->add_symbol_table_entry " << Descr->define_object_label << endl;
		}
		Orbiter->add_symbol_table_entry(Descr->define_object_label, Symb, verbose_level);

	}
	else if (Descr->f_define_surface) {

		cout << "f_define_surface label = " << Descr->define_surface_label << endl;

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action *Surf_A;
		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *SC;

		projective_space_global G;

		G.do_create_surface(
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

		//projective_space_global G;

		//G.table_of_quartic_curves(PA, verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity before PA->table_of_quartic_curves" << endl;
		}
		PA->table_of_quartic_curves(verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after PA->table_of_quartic_curves" << endl;
		}

	}

	else if (Descr->f_table_of_cubic_surfaces) {

		cout << "table_of_cubic_surfaces" << endl;

		//projective_space_global G;

		//G.table_of_cubic_surfaces(PA, verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity before PA->table_of_cubic_surfaces" << endl;
		}
		PA->table_of_cubic_surfaces(verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after PA->table_of_cubic_surfaces" << endl;
		}
	}

	else if (Descr->f_define_quartic_curve) {

		cout << "f_define_quartic_curve label = " << Descr->f_define_quartic_curve << endl;

		applications_in_algebraic_geometry::quartic_curves::quartic_curve_create *QC;

#if 0
		projective_space_global G;

		G.do_create_quartic_curve(
			PA,
			Descr->Quartic_curve_descr,
			QC,
			verbose_level);
#endif

		if (f_v) {
			cout << "projective_space_activity::perform_activity before PA->create_quartic_curve" << endl;
		}
		PA->create_quartic_curve(
				Descr->Quartic_curve_descr,
				QC,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after PA->create_quartic_curve" << endl;
		}

		orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(orbiter_symbol_table_entry);

		Symb->init_quartic_curve(Descr->define_quartic_curve_label, QC, verbose_level);
		if (f_v) {
			cout << "before Orbiter->add_symbol_table_entry "
					<< Descr->define_surface_label << endl;
		}
		Orbiter->add_symbol_table_entry(Descr->define_quartic_curve_label, Symb, verbose_level);


		//FREE_OBJECT(SC);
	}

	// surfaces:


	else if (Descr->f_classify_surfaces_with_double_sixes) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;
		applications_in_algebraic_geometry::cubic_surfaces_and_double_sixes::surface_classify_wedge *SCW;


		SH.classify_surfaces_with_double_sixes(
				PA,
				Descr->classify_surfaces_with_double_sixes_control,
				SCW,
				verbose_level);

		orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(orbiter_symbol_table_entry);

		Symb->init_classification_of_cubic_surfaces_with_double_sixes(Descr->classify_surfaces_with_double_sixes_label, SCW, verbose_level);
		if (f_v) {
			cout << "before Orbiter->add_symbol_table_entry "
					<< Descr->classify_surfaces_with_double_sixes_label << endl;
		}
		Orbiter->add_symbol_table_entry(Descr->classify_surfaces_with_double_sixes_label, Symb, verbose_level);

	}

	else if (Descr->f_classify_surfaces_through_arcs_and_two_lines) {

		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

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
		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

		SH.do_classify_surfaces_through_arcs_and_trihedral_pairs(
				PA,
				Descr->Trihedra1_control, Descr->Trihedra2_control,
				Descr->Control_six_arcs,
				Descr->f_test_nb_Eckardt_points, Descr->nb_E,
				verbose_level);
	}
	else if (Descr->f_sweep_4) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;


		SH.do_sweep_4(
				PA,
				Descr->sweep_4_surface_description,
				Descr->sweep_4_fname,
				verbose_level);
	}
	else if (Descr->f_sweep_4_27) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;


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
		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

		SH.do_six_arcs(
				PA,
				Descr->Control_six_arcs,
				Descr->f_filter_by_nb_Eckardt_points,
				Descr->nb_Eckardt_points,
				verbose_level);
	}
	else if (Descr->f_make_gilbert_varshamov_code) {

		coding_theory::coding_theory_domain Coding;

		Coding.make_gilbert_varshamov_code(
				Descr->make_gilbert_varshamov_code_n,
				Descr->make_gilbert_varshamov_code_n - (PA->P->n + 1),
				Descr->make_gilbert_varshamov_code_d,
				PA->P->F->q,
				PA->P, verbose_level);
	}

	else if (Descr->f_spread_classify) {

#if 0
		projective_space_global G;

		G.do_spread_classify(PA,
				Descr->spread_classify_k,
				Descr->spread_classify_Control,
				verbose_level);
#endif
		PA->do_spread_classify(Descr->spread_classify_k,
				Descr->spread_classify_Control,
				verbose_level);
	}
	else if (Descr->f_classify_semifields) {

#if 0
		projective_space_global G;

		G.do_classify_semifields(
				PA,
				Descr->Semifield_classify_description,
				Descr->Semifield_classify_Control,
				verbose_level);
#endif
		semifields::semifield_classify_with_substructure *S;

		S = NEW_OBJECT(semifields::semifield_classify_with_substructure);

		if (f_v) {
			cout << "projective_space_activity::perform_activity before S->init" << endl;
		}
		S->init(
				Descr->Semifield_classify_description,
				PA,
				Descr->Semifield_classify_Control,
				verbose_level);
		if (f_v) {
			cout << "projective_space_activity::perform_activity after S->init" << endl;
		}

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

#if 0
		projective_space_global G;

		G.do_cheat_sheet_PG(
				PA,
				O,
				verbose_level);
#endif
		PA->cheat_sheet(O, verbose_level);

	}
	else if (Descr->f_classify_quartic_curves_nauty) {

		canonical_form_classifier *Classifier;

		projective_space_global G;

		G.classify_quartic_curves_nauty(PA,
				Descr->classify_quartic_curves_nauty_fname_mask,
				Descr->classify_quartic_curves_nauty_nb,
				Descr->classify_quartic_curves_nauty_fname_classification,
				Classifier,
				verbose_level);
	}


	else if (Descr->f_classify_quartic_curves_with_substructure) {

		projective_space_global G;

		G.classify_quartic_curves(
				PA,
				Descr->classify_quartic_curves_with_substructure_fname_mask,
				Descr->classify_quartic_curves_with_substructure_nb,
				Descr->classify_quartic_curves_with_substructure_size,
				Descr->classify_quartic_curves_with_substructure_degree,
				Descr->classify_quartic_curves_with_substructure_fname_classification,
				verbose_level);

	}
	else if (Descr->f_set_stabilizer) {

		projective_space_global G;

		G.set_stabilizer(PA,
				Descr->set_stabilizer_intermediate_set_size,
				Descr->set_stabilizer_fname_mask,
				Descr->set_stabilizer_nb,
				Descr->set_stabilizer_column_label,
				Descr->set_stabilizer_fname_out,
				verbose_level);
	}

	else if (Descr->f_conic_type) {

		projective_space_global G;

		G.conic_type(PA,
				Descr->conic_type_threshold,
				Descr->conic_type_set_text,
				verbose_level);
	}

	else if (Descr->f_lift_skew_hexagon) {

		projective_space_global G;

		G.do_lift_skew_hexagon(PA,
				Descr->lift_skew_hexagon_text,
				verbose_level);
	}

	else if (Descr->f_lift_skew_hexagon_with_polarity) {

		projective_space_global G;

		G.do_lift_skew_hexagon_with_polarity(PA,
				Descr->lift_skew_hexagon_with_polarity_polarity,
				verbose_level);
	}
	else if (Descr->f_arc_with_given_set_as_s_lines_after_dualizing) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity f_arc_with_given_set_as_i_lines_after_dualizing" << endl;
		}

		PA->P->Arc_in_projective_space->arc_lifting1(
				Descr->arc_size,
				Descr->arc_d,
				Descr->arc_d_low,
				Descr->arc_s,
				Descr->arc_input_set,
				Descr->arc_label,
				verbose_level);

	}
	else if (Descr->f_arc_with_two_given_sets_of_lines_after_dualizing) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity f_arc_with_two_given_sets_of_lines_after_dualizing" << endl;
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


	}
	else if (Descr->f_arc_with_three_given_sets_of_lines_after_dualizing) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity f_arc_with_three_given_sets_of_lines_after_dualizing" << endl;
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

	}
	else if (Descr->f_dualize_hyperplanes_to_points) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity f_dualize_hyperplanes_to_points" << endl;
		}
		long int *the_set_in;
		int set_size_in;
		long int *the_set_out;
		int set_size_out;

		Orbiter->Lint_vec->scan(Descr->dualize_input_set, the_set_in, set_size_in);

		int i;
		long int a;

		set_size_out = set_size_in;
		the_set_out = NEW_lint(set_size_in);
		for (i = 0; i < set_size_in; i++) {
			a = the_set_in[i];
			the_set_out[i] = PA->P->Standard_polarity->Hyperplane_to_point[a];
		}

		cout << "output set:" << endl;
		Orbiter->Lint_vec->print(cout, the_set_out, set_size_in);
		cout << endl;

		// only if n = 2:
		//int *Polarity_point_to_hyperplane; // [N_points]
		//int *Polarity_hyperplane_to_point; // [N_points]

		FREE_lint(the_set_in);
		FREE_lint(the_set_out);

	}
	else if (Descr->f_dualize_points_to_hyperplanes) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity f_dualize_points_to_hyperplanes" << endl;
		}
		long int *the_set_in;
		int set_size_in;
		long int *the_set_out;
		int set_size_out;

		Orbiter->Lint_vec->scan(Descr->dualize_input_set, the_set_in, set_size_in);

		int i;
		long int a;

		set_size_out = set_size_in;
		the_set_out = NEW_lint(set_size_in);
		for (i = 0; i < set_size_in; i++) {
			a = the_set_in[i];
			the_set_out[i] = PA->P->Standard_polarity->Point_to_hyperplane[a];
		}

		cout << "output set:" << endl;
		Orbiter->Lint_vec->print(cout, the_set_out, set_size_in);
		cout << endl;

		FREE_lint(the_set_in);
		FREE_lint(the_set_out);

	}
	else if (Descr->f_dualize_rank_k_subspaces) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity f_dualize_rank_k_subspaces" << endl;
		}
		long int *the_set_in;
		int set_size_in;
		long int *the_set_out;
		int set_size_out;

		Orbiter->Lint_vec->scan(Descr->dualize_input_set, the_set_in, set_size_in);

		int i;
		long int a;

		set_size_out = set_size_in;
		the_set_out = NEW_lint(set_size_in);
		for (i = 0; i < set_size_in; i++) {
			a = the_set_in[i];
			cout << "i=" << i << " in=" << a << endl;
			PA->P->polarity_rank_k_subspace(Descr->dualize_rank_k_subspaces_k,
					a, the_set_out[i], verbose_level);
			cout << "i=" << i << " in=" << a << " out=" << the_set_out[i] << endl;
		}

		cout << "output set:" << endl;
		Orbiter->Lint_vec->print(cout, the_set_out, set_size_in);
		cout << endl;

		FREE_lint(the_set_in);
		FREE_lint(the_set_out);

	}
	else if (Descr->f_classify_arcs) {
		projective_space_global G;

		G.do_classify_arcs(
				PA,
				Descr->Arc_generator_description,
				verbose_level);
	}
	else if (Descr->f_classify_cubic_curves) {
		projective_space_global G;

		G.do_classify_cubic_curves(
				PA,
				Descr->Arc_generator_description,
				verbose_level);
	}
	else if (Descr->f_latex_homogeneous_equation) {

		geometry_global G;
		int degree = Descr->latex_homogeneous_equation_degree;
		int nb_vars = PA->d;

		G.latex_homogeneous_equation(PA->F, degree, nb_vars,
				Descr->latex_homogeneous_equation_text,
				Descr->latex_homogeneous_equation_symbol_txt,
				Descr->latex_homogeneous_equation_symbol_tex,
				verbose_level);


	}
	else if (Descr->f_lines_on_point_but_within_a_plane) {

		if (f_v) {
			cout << "projective_space_activity::perform_activity f_lines_on_point_but_within_a_plane" << endl;
		}

		long int point_rk = Descr->lines_on_point_but_within_a_plane_point_rk;
		long int plane_rk = Descr->lines_on_point_but_within_a_plane_plane_rk;
		long int *line_pencil;
		int q;

		q = PA->F->q;
		line_pencil = NEW_lint(q + 1);

		PA->P->create_lines_on_point_but_inside_a_plane(
				point_rk, plane_rk,
				line_pencil, verbose_level);
			// assumes that line_pencil[q + 1] has been allocated

		cout << "line_pencil: ";
		Orbiter->Lint_vec->print(cout, line_pencil, q + 1);
		cout << endl;

		if (f_v) {
			cout << "projective_space_activity::perform_activity f_lines_on_point_but_within_a_plane done" << endl;
		}
	}



	if (f_v) {
		cout << "projective_space_activity::perform_activity done" << endl;
	}

}


}}}

