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



void projective_space_global::map(
		projective_space_with_action *PA,
		std::string &ring_label,
		std::string &formula_label,
		std::string &evaluate_text,
		long int *&Image_pts,
		int &N_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::map" << endl;
	}
	if (f_v) {
		cout << "projective_space_global::map n = " << PA->P->n << endl;
	}



	int idx;
	ring_theory::homogeneous_polynomial_domain *Ring;

	Ring = Get_object_of_type_ring(ring_label);

	idx = user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->find_symbol(formula_label);

	if (idx < 0) {
		cout << "could not find symbol " << formula_label << endl;
		exit(1);
	}
	user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->get_object(idx);

	if (user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].type != orbiter_kernel_system::t_object) {
		cout << "symbol table entry must be of type t_object" << endl;
		exit(1);
	}
	if (user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].object_type == t_collection) {
		cout << "symbol table entry is a collection" << endl;

		vector<string> *List;

		List = (vector<string> *) user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].ptr;
		int i;

		int *coefficient_vector; // [List->size() * Ring->get_nb_monomials()]

		coefficient_vector = NEW_int(List->size() * Ring->get_nb_monomials());

		for (i = 0; i < List->size(); i++) {
			int idx1;

			idx1 = user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->find_symbol((*List)[i]);
			if (idx1 < 0) {
				cout << "could not find symbol " << (*List)[i] << endl;
				exit(1);
			}
			expression_parser::formula *Formula;
			Formula = (expression_parser::formula *) user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx1].ptr;

			if (f_v) {
				cout << "projective_space_global::map i=" << i << " / " << List->size() << " before Ring->get_coefficient_vector" << endl;
			}
			Ring->get_coefficient_vector(Formula,
					evaluate_text,
					coefficient_vector + i * Ring->get_nb_monomials(),
					verbose_level);
			if (f_v) {
				cout << "projective_space_global::map i=" << i << " / " << List->size() << " after Ring->get_coefficient_vector" << endl;
			}
		}

		if (f_v) {
			cout << "projective_space_global::map coefficient_vector:" << endl;
			Int_matrix_print(coefficient_vector, List->size(), Ring->get_nb_monomials());
		}



		Ring->evaluate_regular_map(
				coefficient_vector,
				List->size(),
				PA->P,
				Image_pts, N_points,
				verbose_level);


		if (f_v) {
			cout << "projective_space_global::map permutation:" << endl;
			Lint_vec_print(cout, Image_pts, N_points);
			cout << endl;
		}



	}
	else if (user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].object_type == t_formula) {
		cout << "symbol table entry is a formula" << endl;

		expression_parser::formula *Formula;
		Formula = (expression_parser::formula *) user_interface::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].ptr;

		int *coefficient_vector; // [Ring->get_nb_monomials()]

		coefficient_vector = NEW_int(Ring->get_nb_monomials());

		Ring->get_coefficient_vector(Formula,
				evaluate_text,
				coefficient_vector,
				verbose_level);


		Ring->evaluate_regular_map(
				coefficient_vector,
				1,
				PA->P,
				Image_pts, N_points,
				verbose_level);


		FREE_int(coefficient_vector);



	}
	else {
		cout << "symbol table entry must be either a formula or a collection" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_global::map done" << endl;
	}
}


void projective_space_global::analyze_del_Pezzo_surface(
		projective_space_with_action *PA,
		std::string &label,
		std::string &evaluate_text,
		int verbose_level)
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

	PA->analyze_del_Pezzo_surface(F, evaluate_text, verbose_level);

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given "
				"after PA->analyze_del_Pezzo_surface" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given done" << endl;
	}
}




void projective_space_global::conic_type(
		projective_space_with_action *PA,
		int threshold,
		std::string &set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::conic_type" << endl;
	}

	long int *Pts;
	int nb_pts;

	Lint_vec_scan(set_text, Pts, nb_pts);


	if (f_v) {
		cout << "projective_space_global::conic_type "
				"before PA->conic_type" << endl;
	}

	PA->conic_type(Pts, nb_pts, threshold, verbose_level);

	if (f_v) {
		cout << "projective_space_global::conic_type "
				"after PA->conic_type" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::conic_type done" << endl;
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
				"before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
	Surf->init(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon "
				"after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action);

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
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
				"before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
	Surf->init(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity "
				"after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action);

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
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
			char str[1000];

			snprintf(str, sizeof(str), "alpha=%d beta=%d", alpha, beta);

			label.assign(str);

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
		apps_geometry::arc_generator_description *Arc_generator_description,
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
		int idx;
		apps_algebra::any_group *AG;
		//linear_group *LG;

		idx = orbiter_kernel_system::Orbiter->find_symbol(Arc_generator_description->override_group_label);
		if (orbiter_kernel_system::Orbiter->get_object_type(idx) != t_any_group) {
			cout << "projective_space_global::do_classify_arcs "
					"The object given must be a group" << endl;
			exit(1);
		}
		AG = (apps_algebra::any_group *) orbiter_kernel_system::Orbiter->get_object(idx);

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


void projective_space_global::do_classify_cubic_curves(
		projective_space_with_action *PA,
		apps_geometry::arc_generator_description *Arc_generator_description,
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
	char str[1000];

	snprintf(str, 1000, "Cubic Curves in PG$(2,%d)$", PA->F->q);
	title.assign(str);
	author.assign("");
	snprintf(str, 1000, "Cubic_curves_q%d.tex", PA->F->q);
	fname.assign(str);

	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;

		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */,
			TRUE /* f_title */,
			title, author,
			FALSE /*f_toc */,
			FALSE /* f_landscape */,
			FALSE /* f_12pt */,
			TRUE /*f_enlarged_page */,
			TRUE /* f_pagenumbers*/,
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

void projective_space_global::classify_quartic_curves_nauty(
		projective_space_with_action *PA,
		std::string &fname_mask, int nb,
		std::string &fname_classification,
		canonical_form_classifier *&Classifier,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_nauty" << endl;
	}


	canonical_form_classifier_description *Descr;

	Descr = NEW_OBJECT(canonical_form_classifier_description);

	Descr->fname_mask.assign(fname_mask);
	Descr->f_fname_base_out = TRUE;
	Descr->fname_base_out.assign(fname_classification);
#if 0
	Descr->column_label_eqn.assign("curve");
	Descr->column_label_pts.assign("pts_on_curve");
	Descr->column_label_bitangents.assign("bitangents");
#endif
	Descr->PA = PA;
	Descr->f_degree = TRUE;
	Descr->degree = 4;
	Descr->nb_files = nb;
	Descr->f_algorithm_nauty = TRUE;
	Descr->f_algorithm_substructure = FALSE;

	std::string column_label_eqn;
	std::string column_label_pts;
	std::string column_label_bitangents;

	Classifier = NEW_OBJECT(canonical_form_classifier);

	Classifier->classify(Descr, verbose_level);

	cout << "The number of types of quartic curves is " << Classifier->CB->nb_types << endl;

	Descr->Canon_substructure = Classifier;



	int idx;

	cout << "idx : ago" << endl;
	for (idx = 0; idx < Classifier->CB->nb_types; idx++) {

		canonical_form_nauty *C1;
		ring_theory::longinteger_object go;

		C1 = (canonical_form_nauty *) Classifier->CB->Type_extra_data[idx];

		C1->Stab_gens_quartic->group_order(go);

		cout << idx << " : " << go << endl;


	}



	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_nauty done" << endl;
	}
}

void projective_space_global::classify_quartic_curves_with_substructure(
		projective_space_with_action *PA,
		std::string &fname_mask, int nb, int substructure_size, int degree,
		std::string &fname_classification,
		canonical_form_classifier *&Classifier,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure" << endl;
	}

	canonical_form_classifier_description *Descr;

	Descr = NEW_OBJECT(canonical_form_classifier_description);


	Descr->fname_mask.assign(fname_mask);
	Descr->f_fname_base_out = TRUE;
	Descr->fname_base_out.assign(fname_classification);
#if 0
	Descr->column_label_eqn.assign("curve");
	Descr->column_label_pts.assign("pts_on_curve");
	Descr->column_label_bitangents.assign("bitangents");
#endif
	Descr->PA = PA;
	Descr->f_degree = TRUE;
	Descr->degree = degree;
	Descr->nb_files = nb;
	Descr->f_algorithm_nauty = FALSE;
	Descr->f_algorithm_substructure = TRUE;
	Descr->substructure_size = substructure_size;


	Classifier = NEW_OBJECT(canonical_form_classifier);

	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure "
				"before Classifier.classify" << endl;
	}
	Classifier->classify(Descr, verbose_level);
	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure "
				"after Classifier.classify" << endl;
	}

	Descr->Canon_substructure = Classifier;


	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure "
				"before Classifier.report" << endl;
	}
	Classifier->report(fname_classification, verbose_level);
	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure "
				"after Classifier.report" << endl;
	}


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
		cout << "projective_space_global::classify_quartic_curves_with_substructure done" << endl;
	}
}


void projective_space_global::classify_quartic_curves(
		projective_space_with_action *PA,
		std::string &fname_mask,
		int nb,
		int size,
		int degree,
		std::string &fname_classification,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves" << endl;
	}

	canonical_form_classifier *Classifier;


	classify_quartic_curves_with_substructure(PA,
			fname_mask,
			nb,
			size,
			degree,
			fname_classification,
			Classifier,
			verbose_level);

	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves done" << endl;
	}

}

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

void projective_space_global::make_restricted_incidence_matrix(
		projective_space_with_action *PA,
		int type_i, int type_j,
		std::string &row_objects,
		std::string &col_objects,
		std::string &file_name,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::make_restricted_incidence_matrix" << endl;
	}


	long int *Row_objects;
	int nb_row_objects;
	long int *Col_objects;
	int nb_col_objects;
	int i, j;

	int *M;

	Get_vector_or_set(row_objects, Row_objects, nb_row_objects);
	Get_vector_or_set(col_objects, Col_objects, nb_col_objects);

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

	fname_csv.assign(file_name);
	fname_inc.assign(file_name);

	fname_csv.append(".csv");
	Fio.int_matrix_write_csv(fname_csv, M, nb_row_objects, nb_col_objects);

	if (f_v) {
		cout << "written file " << fname_csv << " of size "
				<< Fio.file_size(fname_csv) << endl;
	}

	fname_inc.append(".inc");
	Fio.write_incidence_matrix_to_file(fname_inc,
		M, nb_row_objects, nb_col_objects, 0 /*verbose_level*/);

	if (f_v) {
		cout << "written file " << fname_inc << " of size "
				<< Fio.file_size(fname_inc) << endl;
	}

	FREE_int(M);

	if (f_v) {
		cout << "projective_space_global::make_restricted_incidence_matrix done" << endl;
	}
}

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

	if (f_v) {
		cout << "projective_space_global::make_relation done" << endl;
	}

}

void projective_space_global::plane_intersection_type_of_klein_image(
		projective_space_with_action *PA,
		std::string &input,
		int threshold,
		int verbose_level)
// creates a projective_space object P5
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::plane_intersection_type_of_klein_image" << endl;
	}
	long int *Lines;
	int nb_lines;

	Get_vector_or_set(input, Lines, nb_lines);

	//int *intersection_type;
	//int highest_intersection_number;

	geometry::projective_space *P5;

	P5 = NEW_OBJECT(geometry::projective_space);

	int f_init_incidence_structure = TRUE;

	if (f_v) {
		cout << "projective_space_global::plane_intersection_type_of_klein_image "
				"before P5->projective_space_init" << endl;
	}
	P5->projective_space_init(5, PA->P->F,
			f_init_incidence_structure,
			verbose_level);
	if (f_v) {
		cout << "projective_space_global::plane_intersection_type_of_klein_image "
				"after P5->projective_space_init" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::plane_intersection_type_of_klein_image "
				"before plane_intersection_type_of_klein_image" << endl;
	}

	geometry::intersection_type *Int_type;

	PA->P->Grass_lines->plane_intersection_type_of_klein_image(
			PA->P /* P3 */,
			P5,
			Lines, nb_lines, threshold,
			Int_type,
			verbose_level);

	if (f_v) {
		cout << "projective_space_global::plane_intersection_type_of_klein_image "
				"after plane_intersection_type_of_klein_image" << endl;
	}

	cout << "projective_space_global::plane_intersection_type_of_klein_image "
			"intersection numbers: ";
	Int_vec_print(cout, Int_type->the_intersection_type, Int_type->highest_intersection_number + 1);
	cout << endl;

	if (f_v) {
		cout << "projective_space_global::plane_intersection_type_of_klein_image "
				"highest weight objects: " << endl;
		Lint_vec_print(cout, Int_type->Highest_weight_objects, Int_type->nb_highest_weight_objects);
		cout << endl;
	}

	if (f_v) {
		cout << "projective_space_global::plane_intersection_type_of_klein_image "
				"Intersection_sets: " << endl;
		Int_matrix_print(Int_type->Intersection_sets, Int_type->nb_highest_weight_objects, Int_type->highest_intersection_number);
	}

	if (f_v) {
		cout << "projective_space_global::plane_intersection_type_of_klein_image "
				"Intersection_sets sorted: " << endl;
		Int_matrix_print(Int_type->M->M, Int_type->nb_highest_weight_objects, Int_type->highest_intersection_number);
	}

	string fname;
	data_structures::string_tools ST;

	fname.assign(input);
	ST.chop_off_extension(fname);
	fname.append("_highest_weight_objects.csv");

	Int_type->M->write_csv(fname, verbose_level);


	FREE_OBJECT(Int_type);

	FREE_OBJECT(P5);

	if (f_v) {
		cout << "projective_space_global::plane_intersection_type_of_klein_image done" << endl;
	}
}


}}}





