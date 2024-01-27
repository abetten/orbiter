/*
 * projective_space_global.cpp
 *
 *  Created on: Oct 9, 2021
 *      Author: betten
 */

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace projective_geometry {


void projective_space_global::analyze_del_Pezzo_surface(
		projective_space_with_action *PA,
		std::string &label,
		std::string &evaluate_text,
		int verbose_level)
// ToDo use symbolic object instead
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface" << endl;
	}



	int idx;
	idx = user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->find_symbol(label);

	if (idx < 0) {
		cout << "could not find symbol " << label << endl;
		exit(1);
	}
	user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->get_object(idx);

	if (user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].type != orbiter_kernel_system::t_object) {
		cout << "symbol table entry must be of type t_object" << endl;
		exit(1);
	}


	// ToDo use symbolic object instead

#if 0
	if (user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].object_type == t_collection) {
		cout << "symbol table entry is a collection" << endl;

		vector<string> *List;

		List = (vector<string> *) user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].ptr;
		int i;

		for (i = 0; i < List->size(); i++) {
			int idx1;

			idx1 = user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->find_symbol((*List)[i]);
			if (idx1 < 0) {
				cout << "could not find symbol " << (*List)[i] << endl;
				exit(1);
			}
			expression_parser::formula *F;
			F = (expression_parser::formula *) user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx1].ptr;

			analyze_del_Pezzo_surface_formula_given(
					PA,
					F,
					evaluate_text,
					verbose_level);
		}
	}
	else if (user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].object_type == t_formula) {
		cout << "symbol table entry is a formula" << endl;

		expression_parser::formula *F;
		F = (expression_parser::formula *) user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].ptr;

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
#endif


	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface done" << endl;
	}
}


void projective_space_global::analyze_del_Pezzo_surface_formula_given(
		projective_space_with_action *PA,
		expression_parser::formula *F,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given "
				"before PA->analyze_del_Pezzo_surface" << endl;
	}

	algebraic_geometry::algebraic_geometry_global AGG;

	AGG.analyze_del_Pezzo_surface(PA->P, F, evaluate_text, verbose_level);

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given "
				"after PA->analyze_del_Pezzo_surface" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given done" << endl;
	}
}





void projective_space_global::do_lift_skew_hexagon(
		projective_space_with_action *PA,
		std::string &text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon" << endl;
	}

	int *Pluecker_coords;
	int sz;

	Int_vec_scan(text, Pluecker_coords, sz);

	long int *Pts;
	int nb_pts;

	nb_pts = sz / 6;

	if (nb_pts * 6 != sz) {
		cout << "projective_space_global::do_lift_skew_hexagon "
				"the number of coordinates must be a multiple of 6" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "Pluecker coordinates of lines:" << endl;
		Int_matrix_print(Pluecker_coords, nb_pts, 6);
	}

	algebraic_geometry::surface_domain *Surf;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action *Surf_A;

	if (PA->n != 3) {
		cout << "projective_space_global::do_lift_skew_hexagon "
				"we need a three-dimensional projective space" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon "
				"before Surf->init_surface_domain" << endl;
	}
	Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
	Surf->init_surface_domain(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon "
				"after Surf->init_surface_domain" << endl;
	}

	Surf_A = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action);

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, true /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon "
				"after Surf_A->init" << endl;
	}




	int i;

	Pts = NEW_lint(nb_pts);

	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Surf_A->Surf->Klein->Pluecker_to_line_rk(
				Pluecker_coords + i * 6, 0 /*verbose_level*/);
	}

	if (nb_pts != 6) {
		cout << "projective_space_global::do_lift_skew_hexagon "
				"nb_pts != 6" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "lines:" << endl;
		Lint_vec_print(cout, Pts, 6);
		cout << endl;
	}


	std::vector<std::vector<long int> > Double_sixes;

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon "
				"before Surf_A->complete_skew_hexagon" << endl;
	}

	Surf_A->complete_skew_hexagon(Pts, Double_sixes, verbose_level);

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon "
				"after Surf_A->complete_skew_hexagon" << endl;
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
		cout << "projective_space_global::do_lift_skew_hexagon done" << endl;
	}
}


void projective_space_global::do_lift_skew_hexagon_with_polarity(
		projective_space_with_action *PA,
		std::string &polarity_36,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity" << endl;
	}

	int *Polarity36;
	int sz1;

	Int_vec_scan(polarity_36, Polarity36, sz1);

	if (sz1 != 36) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity "
				"We need exactly 36 coefficients for the polarity" << endl;
		exit(1);
	}


	algebraic_geometry::surface_domain *Surf;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action *Surf_A;

	if (PA->n != 3) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity "
				"We need a three-dimensional projective space" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity "
				"before Surf->init_surface_domain" << endl;
	}
	Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
	Surf->init_surface_domain(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity "
				"after Surf->init_surface_domain" << endl;
	}

	Surf_A = NEW_OBJECT(
			applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action);

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, true /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity "
				"after Surf_A->init" << endl;
	}




	std::vector<std::vector<long int> > Double_sixes;

	int Pluecker_coords[36];
	int alpha, beta;
	int i, j;

	Int_vec_zero(Pluecker_coords, 36);
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
				Surf->F->Linear_algebra->mult_matrix_matrix(
						Pluecker_coords + j * 6,
						Polarity36,
						Pluecker_coords + 18 + j * 6,
						1, 6, 6,
						0 /* verbose_level */);
			}

			int nb_pts;

			nb_pts = 6;

			if (f_v) {
				cout << "Pluecker coordinates of lines:" << endl;
				Int_matrix_print(Pluecker_coords, nb_pts, 6);
			}


			long int *Pts;


			Pts = NEW_lint(nb_pts);

			for (i = 0; i < nb_pts; i++) {
				Pts[i] = Surf_A->Surf->Klein->Pluecker_to_line_rk(
						Pluecker_coords + i * 6,
						0 /*verbose_level*/);
			}

			if (nb_pts != 6) {
				cout << "projective_space_global::do_lift_skew_hexagon_with_polarity "
						"nb_pts != 6" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "lines:" << endl;
				Lint_vec_print(cout, Pts, 6);
				cout << endl;
			}


			string label;

			label = "alpha=" + std::to_string(alpha) + " beta=" + std::to_string(beta);

			if (f_v) {
				cout << "projective_space_global::do_lift_skew_hexagon_with_polarity "
						"before Surf_A->complete_skew_hexagon_with_polarity" << endl;
			}

			Surf_A->complete_skew_hexagon_with_polarity(label,
					Pts, Polarity36, Double_sixes,
					verbose_level);

			if (f_v) {
				cout << "projective_space_global::do_lift_skew_hexagon_with_polarity "
						"after Surf_A->complete_skew_hexagon_with_polarity" << endl;
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
		cout << "projective_space_global::do_lift_do_lift_skew_hexagon_with_polarity done" << endl;
	}
}


void projective_space_global::do_classify_arcs(
		projective_space_with_action *PA,
		apps_geometry::arc_generator_description
			*Arc_generator_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::do_classify_arcs" << endl;
	}

#if 0
	Arc_generator_description->F = LG->F;
	Arc_generator_description->LG = LG;
	Arc_generator_description->Control = Descr->Control;

	if (Arc_generator_description->n != LG->A2->matrix_group_dimension()) {
		cout << "projective_space_global::do_classify_arcs the dimensions don't match" << endl;
		exit(1);
	}
#endif


	groups::strong_generators *gens;

	if (Arc_generator_description->f_override_group) {
		if (f_v) {
			cout << "projective_space_global::do_classify_arcs "
					"f_override_group label = " << Arc_generator_description->override_group_label << endl;
		}
		//int idx;
		apps_algebra::any_group *AG;
		//linear_group *LG;

#if 0
		idx = orbiter_kernel_system::Orbiter->find_symbol(Arc_generator_description->override_group_label);
		if (orbiter_kernel_system::Orbiter->get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_any_group) {
			cout << "projective_space_global::do_classify_arcs "
					"The object given must be a group" << endl;
			exit(1);
		}
		AG = (apps_algebra::any_group *) orbiter_kernel_system::Orbiter->get_object(idx);
#endif

		AG = Get_any_group(Arc_generator_description->override_group_label);

#if 0
		if (!LG->f_has_strong_generators) {
			cout << "projective_space_global::do_classify_arcs "
					"the group must have strong generators" << endl;
			exit(1);
		}
#endif

		gens = AG->Subgroup_gens;


	}
	else {
		gens = PA->A->Strong_gens;
	}

	{
		apps_geometry::arc_generator *Gen;

		Gen = NEW_OBJECT(apps_geometry::arc_generator);



		if (f_v) {
			cout << "projective_space_global::do_classify_arcs "
					"before Gen->init" << endl;
		}
		Gen->init(
				Arc_generator_description,
				PA,
				gens,
				verbose_level);

		if (f_v) {
			cout << "projective_space_global::do_classify_arcs "
					"after Gen->init" << endl;
		}



		if (f_v) {
			cout << "projective_space_global::do_classify_arcs "
					"before Gen->main" << endl;
		}
		Gen->main(verbose_level);
		if (f_v) {
			cout << "projective_space_global::do_classify_arcs "
					"after Gen->main" << endl;
		}


		FREE_OBJECT(Gen);
	}


	if (f_v) {
		cout << "projective_space_global::do_classify_arcs done" << endl;
	}
}


#if 0
void projective_space_global::do_classify_cubic_curves(
		projective_space_with_action *PA,
		apps_geometry::arc_generator_description
			*Arc_generator_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves" << endl;
	}



	algebraic_geometry::cubic_curve *CC;

	CC = NEW_OBJECT(algebraic_geometry::cubic_curve);

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"before CC->init" << endl;
	}
	CC->init(PA->F, verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"after CC->init" << endl;
	}


	apps_geometry::cubic_curve_with_action *CCA;

	CCA = NEW_OBJECT(apps_geometry::cubic_curve_with_action);

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"before CCA->init" << endl;
	}
	CCA->init(CC, PA->A, verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"after CCA->init" << endl;
	}


	apps_geometry::classify_cubic_curves *CCC;

	CCC = NEW_OBJECT(apps_geometry::classify_cubic_curves);


	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"before CCC->init" << endl;
	}
	CCC->init(
			PA,
			CCA,
			Arc_generator_description,
			verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"after CCC->init" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"before CCC->compute_starter" << endl;
	}
	CCC->compute_starter(verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"after CCC->compute_starter" << endl;
	}

#if 0
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"before CCC->test_orbits" << endl;
	}
	CCC->test_orbits(verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"after CCC->test_orbits" << endl;
	}
#endif

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"before CCC->do_classify" << endl;
	}
	CCC->do_classify(verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"after CCC->do_classify" << endl;
	}


	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"creating cheat sheet" << endl;
	}
	string fname, title, author, extra_praeamble;

	title = "Cubic Curves in PG$(2," + std::to_string(PA->F->q) + ")$";
	author = "";
	fname = "Cubic_curves_q" + std::to_string(PA->F->q) + ".tex";

	{
		ofstream fp(fname);
		l1_interfaces::latex_interface L;

		//latex_head_easy(fp);
		L.head(fp,
			false /* f_book */,
			true /* f_title */,
			title, author,
			false /*f_toc */,
			false /* f_landscape */,
			false /* f_12pt */,
			true /*f_enlarged_page */,
			true /* f_pagenumbers*/,
			extra_praeamble /* extra_praeamble */);

		fp << "\\subsection*{" << title << "}" << endl;

		if (f_v) {
			cout << "projective_space_global::do_classify_cubic_curves "
					"before CCC->report" << endl;
		}
		CCC->report(fp, verbose_level);
		if (f_v) {
			cout << "projective_space_global::do_classify_cubic_curves "
					"after CCC->report" << endl;
		}

		L.foot(fp);
	}

	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname << " of size "
		<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves "
				"writing cheat sheet on "
				"cubic curves done" << endl;
	}


	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves done" << endl;
	}
}
#endif


void projective_space_global::set_stabilizer(
		projective_space_with_action *PA,
		int intermediate_subset_size,
		std::string &fname_mask, int nb, std::string &column_label,
		std::string &fname_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::set_stabilizer" << endl;
	}

#if 0
	top_level_geometry_global T;

	T.set_stabilizer_projective_space(
				PA,
				intermediate_subset_size,
				fname_mask, nb, column_label,
				verbose_level);
#endif
	set_stabilizer::substructure_classifier *SubC;

	SubC = NEW_OBJECT(set_stabilizer::substructure_classifier);

	SubC->set_stabilizer_in_any_space(
			PA->A, PA->A, PA->A->Strong_gens,
			intermediate_subset_size,
			fname_mask, nb, column_label,
			fname_out,
			verbose_level);
	FREE_OBJECT(SubC);

	if (f_v) {
		cout << "projective_space_global::set_stabilizer done" << endl;
	}

}


#if 0
void projective_space_global::make_relation(
		projective_space_with_action *PA,
		long int plane_rk,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::make_relation" << endl;
	}

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



	PA->P->points_covered_by_plane(plane_rk,
			the_points, nb_points, 0 /* verbose_level */);

	Sorting.lint_vec_heapsort(the_points, nb_points);


	if (nb_points != 7) {
		cout << "projective_space_global::make_relation "
				"wrong projective space, must be PG(3,2)" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "projective_space_global::make_relation "
				"the_points : " << nb_points << " : ";
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
		cout << "projective_space_global::make_relation "
				"nb_points_outside != 8" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "projective_space_global::make_relation "
				"the_outside_points : " << nb_points_outside << " : ";
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
		cout << "projective_space_global::make_relation "
				"the_outside_lines : " << nb_lines_outside << " : ";
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
		cout << "projective_space_global::make_relation "
				"the_inside_lines : " << nb_lines_inside << " : ";
		Lint_vec_print(cout, the_inside_lines, nb_lines_inside);
		cout << endl;
	}


	Sorting.lint_vec_sort_and_remove_duplicates(the_inside_lines, nb_lines_inside);
	if (nb_lines_inside != 7) {
		cout << "projective_space_global::make_relation "
				"nb_lines_inside != 7" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "projective_space_global::make_relation "
				"the_inside_lines : " << nb_lines_inside << " : ";
		Lint_vec_print(cout, the_inside_lines, nb_lines_inside);
		cout << endl;
	}



	for (i = 0; i < nb_lines_inside; i++) {
		long int *pts;
		int nb;

		PA->P->points_on_line(the_inside_lines[i],
				pts, nb, 0 /* verbose_level */);
		if (nb != 3) {
			cout << "projective_space_global::make_relation "
					"nb != 3" << endl;
		}
		Lint_vec_copy(pts, points_on_inside_lines + i * 3, 3);
		Sorting.lint_vec_heapsort(points_on_inside_lines + i * 3, 3);
		FREE_lint(pts);
	}


	if (f_v) {
		cout << "projective_space_global::make_relation "
				"points_on_inside_lines : " << endl;
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

	fname_csv = "relation.csv";
	fname_inc = "relation.inc";

	Fio.int_matrix_write_csv(fname_csv, M, nb_pts, nb_lines);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;

	Fio.write_incidence_matrix_to_file(fname_inc,
		M, nb_pts, nb_lines, 0 /*verbose_level*/);
	cout << "written file " << fname_inc << " of size "
			<< Fio.file_size(fname_inc) << endl;

	FREE_int(M);

	if (f_v) {
		cout << "projective_space_global::make_relation done" << endl;
	}

}
#endif


#if 0
void projective_space_global::classify_bent_functions(
		projective_space_with_action *PA,
		int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::classify_bent_functions" << endl;
	}

	if (PA->P->Subspaces->F->q != 2) {
		cout << "projective_space_global::classify_bent_functions the field must have order 2" << endl;
		exit(1);
	}
	if (PA->A->matrix_group_dimension() != n + 1) {
		cout << "projective_space_global::classify_bent_functions the dimension of the matrix group must be n + 1" << endl;
		exit(1);
	}

	combinatorics::boolean_function_domain *BF;

	BF = NEW_OBJECT(combinatorics::boolean_function_domain);

	if (f_v) {
		cout << "projective_space_global::classify_bent_functions before BF->init" << endl;
	}
	BF->init(PA->P->Subspaces->F, n, verbose_level);
	if (f_v) {
		cout << "projective_space_global::classify_bent_functions after BF->init" << endl;
	}

	apps_combinatorics::boolean_function_classify *BFC;

	BFC = NEW_OBJECT(apps_combinatorics::boolean_function_classify);

	if (f_v) {
		cout << "projective_space_global::classify_bent_functions before BFC->init_group" << endl;
	}
	BFC->init_group(BF, PA->A, verbose_level);
	if (f_v) {
		cout << "projective_space_global::classify_bent_functions after BFC->init_group" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::classify_bent_functions before BFC->search_for_bent_functions" << endl;
	}
	BFC->search_for_bent_functions(verbose_level);
	if (f_v) {
		cout << "projective_space_global::classify_bent_functions after BFC->search_for_bent_functions" << endl;
	}

	FREE_OBJECT(BFC);
	FREE_OBJECT(BF);

	if (f_v) {
		cout << "projective_space_global::classify_bent_functions done" << endl;
	}

}
#endif


}}}





