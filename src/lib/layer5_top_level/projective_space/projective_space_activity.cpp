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


		int *genma;
		int k, n;

		orbiter_kernel_system::Orbiter->get_matrix_from_label(Descr->canonical_form_of_code_generator_matrix, genma, k, n);

		PA->canonical_form_of_code(
				Descr->canonical_form_of_code_label,
					genma, k, n,
					Descr->Canonical_form_codes_Descr,
					verbose_level);


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

		G.map(
				PA,
				Descr->map_ring_label,
				Descr->map_formula_label,
				Descr->map_parameters,
				Image_pts, N_points,
				verbose_level);

		if (f_v) {
			cout << "projective_space_activity::perform_activity after G.map" << endl;
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
			cout << "Written file " << fname_map << " of size " << Fio.file_size(fname_map) << endl;
		}



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



	else if (Descr->f_table_of_quartic_curves) {

		cout << "table_of_quartic_curves" << endl;

		//projective_space_global G;

		//G.table_of_quartic_curves(PA, verbose_level);
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

		//projective_space_global G;

		//G.table_of_cubic_surfaces(PA, verbose_level);
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
				Descr->classify_surfaces_with_double_sixes_control,
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
	else if (Descr->f_sweep_4_15_lines) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;


		SH.do_sweep_4_15_lines(
				PA,
				Descr->sweep_4_15_lines_surface_description,
				Descr->sweep_4_15_lines_fname,
				verbose_level);
	}
	else if (Descr->f_sweep_F_beta_9_lines) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;


		SH.do_sweep_F_beta_9_lines(
				PA,
				Descr->sweep_F_beta_9_lines_surface_description,
				Descr->sweep_F_beta_9_lines_fname,
				verbose_level);
	}
	else if (Descr->f_sweep_6_9_lines) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;


		SH.do_sweep_6_9_lines(
				PA,
				Descr->sweep_6_9_lines_surface_description,
				Descr->sweep_6_9_lines_fname,
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

	else if (Descr->f_sweep_4_L9_E4) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;


		SH.do_sweep_4_L9_E4(
				PA,
				Descr->sweep_4_L9_E4_surface_description,
				Descr->sweep_4_L9_E4_fname,
				verbose_level);
	}

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
			cout << "projective_space_activity::perform_activity "
					"f_arc_with_given_set_as_i_lines_after_dualizing" << endl;
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
			cout << "projective_space_activity::perform_activity "
					"f_arc_with_two_given_sets_of_lines_after_dualizing" << endl;
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
			cout << "projective_space_activity::perform_activity "
					"f_arc_with_three_given_sets_of_lines_after_dualizing" << endl;
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

		geometry::geometry_global G;
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
			cout << "projective_space_activity::perform_activity "
					"f_lines_on_point_but_within_a_plane" << endl;
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
		Lint_vec_print(cout, line_pencil, q + 1);
		cout << endl;

		if (f_v) {
			cout << "projective_space_activity::perform_activity "
					"f_lines_on_point_but_within_a_plane done" << endl;
		}
	}
	else if (Descr->f_rank_lines_in_PG) {


		PA->do_rank_lines_in_PG(
				Descr->rank_lines_in_PG_label,
				verbose_level);
	}
	else if (Descr->f_unrank_lines_in_PG) {


		PA->do_unrank_lines_in_PG(
				Descr->unrank_lines_in_PG_text,
				verbose_level);
	}
	else if (Descr->f_move_two_lines_in_hyperplane_stabilizer) {


		PA->P->do_move_two_lines_in_hyperplane_stabilizer(
				Descr->line1_from, Descr->line2_from,
				Descr->line1_to, Descr->line2_to, verbose_level);
	}
	else if (Descr->f_move_two_lines_in_hyperplane_stabilizer_text) {


		PA->P->do_move_two_lines_in_hyperplane_stabilizer_text(
				Descr->line1_from_text, Descr->line2_from_text,
				Descr->line1_to_text, Descr->line2_to_text,
				verbose_level);
	}
	else if (Descr->f_planes_through_line) {

		cout << "planes through line:" << endl;
		long int *v;
		int sz, i, j, d;
		int *M;

		Lint_vec_scan(Descr->planes_through_line_rank, v, sz);

		d = PA->P->n + 1;
		M = NEW_int(3 * d);

		for (i = 0; i < sz; i++) {

			std::vector<long int> plane_ranks;

			PA->P->planes_through_a_line(
					v[i], plane_ranks,
					0 /*verbose_level*/);

			cout << "planes through line " << v[i] << " : ";
			for (j = 0; j < plane_ranks.size(); j++) {
				cout << plane_ranks[j];
				if (j < plane_ranks.size() - 1) {
					cout << ",";
				}
			}
			cout << endl;


			cout << "planes through line " << v[i] << endl;
			for (j = 0; j < plane_ranks.size(); j++) {
				cout << j << " : " << plane_ranks[j] << " : " << endl;
				PA->P->Grass_planes->unrank_lint_here(M, plane_ranks[j], 0 /* verbose_level */);
				Int_matrix_print(M, 3, d);

			}
			cout << endl;


		}
		FREE_int(M);

	}
	else if (Descr->f_restricted_incidence_matrix) {

		cout << "f_restricted_incidence_matrix:" << endl;

		long int *Row_objects;
		int nb_row_objects;
		long int *Col_objects;
		int nb_col_objects;
		int i, j;
		int type_i = Descr->restricted_incidence_matrix_type_row_objects;
		int type_j = Descr->restricted_incidence_matrix_type_col_objects;

		int *M;

		Get_vector_or_set(Descr->restricted_incidence_matrix_row_objects, Row_objects, nb_row_objects);
		Get_vector_or_set(Descr->restricted_incidence_matrix_col_objects, Col_objects, nb_col_objects);

		M = NEW_int(nb_row_objects * nb_col_objects);
		Int_vec_zero(M, nb_row_objects * nb_col_objects);

		for (i = 0; i < nb_row_objects; i++) {

			for (j = 0; j < nb_col_objects; j++) {

				if (PA->P->incidence_test_for_objects_of_type_ij(
					type_i, type_j, Row_objects[i], Col_objects[j],
					0 /* verbose_level */)) {
					M[i * nb_col_objects + j] = 1;
				}
			}
		}

		orbiter_kernel_system::file_io Fio;
		string fname_csv;
		string fname_inc;

		fname_csv.assign(Descr->restricted_incidence_matrix_file_name);
		fname_inc.assign(Descr->restricted_incidence_matrix_file_name);

		fname_csv.append(".csv");
		Fio.int_matrix_write_csv(fname_csv, M, nb_row_objects, nb_col_objects);
		cout << "written file " << fname_csv << " of size "
				<< Fio.file_size(fname_csv) << endl;

		fname_inc.append(".inc");
		Fio.write_incidence_matrix_to_file(fname_inc,
			M, nb_row_objects, nb_col_objects, 0 /*verbose_level*/);
		cout << "written file " << fname_inc << " of size "
				<< Fio.file_size(fname_inc) << endl;

		FREE_int(M);

	}




	else if (Descr->f_make_relation) {

		cout << "f_make_relation:" << endl;


		//long int plane_rk = 0;
		long int line_rk = 0;
		long int *the_points; // [7]
		long int *the_outside_points; // [8]
		long int *the_outside_lines; // [28]
		long int *the_inside_lines; // [21]
		long int *points_on_inside_lines; // [21]
		int nb_points;
		int nb_points_outside;
		int nb_lines_outside;
		int nb_lines_inside;
		int pair[2];
		int i;
		long int p1, p2;

		combinatorics::combinatorics_domain Combi;
		data_structures::sorting Sorting;



		PA->P->points_covered_by_plane(Descr->make_relation_plane_rk,
				the_points, nb_points, 0 /* verbose_level */);

		Sorting.lint_vec_heapsort(the_points, nb_points);


		if (nb_points != 7) {
			cout << "f_make_relation wrong projective space, must be PG(3,2)" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "f_make_relation the_points : " << nb_points << " : ";
			Lint_vec_print(cout, the_points, nb_points);
			cout << endl;
		}

		the_outside_points = NEW_lint(8);
		the_outside_lines = NEW_lint(28);
		the_inside_lines = NEW_lint(21);
		points_on_inside_lines = NEW_lint(21);

		Combi.set_complement_lint(the_points, nb_points, the_outside_points,
				nb_points_outside, 15 /* universal_set_size */);

		if (nb_points_outside != 8) {
			cout << "f_make_relation nb_points_outside != 8" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "f_make_relation the_outside_points : " << nb_points_outside << " : ";
			Lint_vec_print(cout, the_outside_points, nb_points_outside);
			cout << endl;
		}

		nb_lines_outside = 28;

		for (i = 0; i < nb_lines_outside; i++) {
			Combi.unrank_k_subset(i, pair, 8, 2);
			p1 = the_outside_points[pair[0]];
			p2 = the_outside_points[pair[1]];
			line_rk = PA->P->line_through_two_points(p1, p2);
			the_outside_lines[i] = line_rk;
		}

		Sorting.lint_vec_heapsort(the_outside_lines, nb_lines_outside);

		if (f_v) {
			cout << "f_make_relation the_outside_lines : " << nb_lines_outside << " : ";
			Lint_vec_print(cout, the_outside_lines, nb_lines_outside);
			cout << endl;
		}


		nb_lines_inside = 21;

		for (i = 0; i < nb_lines_inside; i++) {
			Combi.unrank_k_subset(i, pair, 7, 2);
			p1 = the_points[pair[0]];
			p2 = the_points[pair[1]];
			line_rk = PA->P->line_through_two_points(p1, p2);
			the_inside_lines[i] = line_rk;
		}

		if (f_v) {
			cout << "f_make_relation the_inside_lines : " << nb_lines_inside << " : ";
			Lint_vec_print(cout, the_inside_lines, nb_lines_inside);
			cout << endl;
		}


		Sorting.lint_vec_sort_and_remove_duplicates(the_inside_lines, nb_lines_inside);
		if (nb_lines_inside != 7) {
			cout << "f_make_relation nb_lines_inside != 7" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "f_make_relation the_inside_lines : " << nb_lines_inside << " : ";
			Lint_vec_print(cout, the_inside_lines, nb_lines_inside);
			cout << endl;
		}



		for (i = 0; i < nb_lines_inside; i++) {
			long int *pts;
			int nb;

			PA->P->points_on_line(the_inside_lines[i],
					pts, nb, 0 /* verbose_level */);
			if (nb != 3) {
				cout << "f_make_relation nb != 3" << endl;
			}
			Lint_vec_copy(pts, points_on_inside_lines + i * 3, 3);
			Sorting.lint_vec_heapsort(points_on_inside_lines + i * 3, 3);
			FREE_lint(pts);
		}


		if (f_v) {
			cout << "f_make_relation points_on_inside_lines : " << endl;
			Lint_matrix_print(points_on_inside_lines, nb_lines_inside, 3);
			cout << endl;
		}


		//int j;

		int *M;

		int nb_pts;
		int nb_lines;

		nb_pts = 21;
		nb_lines = 28;

		M = NEW_int(nb_pts * nb_lines);
		Int_vec_zero(M, nb_pts * nb_lines);


		int pt_idx, pt_on_line_idx;
		long int pt, line;

		for (i = 0; i < nb_pts; i++) {

			pt_idx = i / 3;
			pt_on_line_idx = i % 3;
			line = the_inside_lines[pt_idx];
			pt = points_on_inside_lines[pt_idx * 3 + pt_on_line_idx];


#if 0
			for (j = 0; j < nb_lines; j++) {

				if (PA->P->incidence_test_for_objects_of_type_ij(
					type_i, type_j, Pts[i], Lines[j],
					0 /* verbose_level */)) {
					M[i * nb_lines + j] = 1;
				}
			}
#endif

		}


		orbiter_kernel_system::file_io Fio;
		string fname_csv;
		string fname_inc;

		fname_csv.assign("relation");
		fname_inc.assign("relation");

		fname_csv.append(".csv");
		Fio.int_matrix_write_csv(fname_csv, M, nb_pts, nb_lines);
		cout << "written file " << fname_csv << " of size "
				<< Fio.file_size(fname_csv) << endl;

		fname_inc.append(".inc");
		Fio.write_incidence_matrix_to_file(fname_inc,
			M, nb_pts, nb_lines, 0 /*verbose_level*/);
		cout << "written file " << fname_inc << " of size "
				<< Fio.file_size(fname_inc) << endl;

		FREE_int(M);

	}



	if (f_v) {
		cout << "projective_space_activity::perform_activity done" << endl;
	}

}






}}}

