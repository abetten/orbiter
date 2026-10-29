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


projective_space_global::projective_space_global()
{
	Record_birth();

}

projective_space_global::~projective_space_global()
{
	Record_death();

}


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
	idx = user_interface::core_system::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->find_symbol(label);

	if (idx < 0) {
		cout << "could not find symbol " << label << endl;
		exit(1);
	}
	user_interface::core_system::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->get_object(idx);

	if (user_interface::core_system::The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].type != other::orbiter_kernel_system::t_object) {
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
		algebra::expression_parser::formula *F,
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

	geometry::algebraic_geometry::algebraic_geometry_global AGG;

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

	geometry::algebraic_geometry::surface_domain *Surf;
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
	Surf = NEW_OBJECT(geometry::algebraic_geometry::surface_domain);
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


	geometry::algebraic_geometry::surface_domain *Surf;
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
	Surf = NEW_OBJECT(geometry::algebraic_geometry::surface_domain);
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



}}}





