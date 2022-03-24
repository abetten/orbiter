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
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::map" << endl;
	}
	if (f_v) {
		cout << "projective_space_global::map PA->P->n = " << PA->P->n << endl;
	}


	int idx;
	ring_theory::homogeneous_polynomial_domain *Ring;
	Ring = user_interface::The_Orbiter_top_level_session->get_object_of_type_ring(ring_label);

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

		long int *Pts;
		int N;

		Ring->evaluate_regular_map(
				coefficient_vector,
				List->size(),
				PA->P,
				Pts, N,
				verbose_level);

		if (f_v) {
			cout << "projective_space_global::map permutation:" << endl;
			Lint_vec_print(cout, Pts, N);
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
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given before PA->analyze_del_Pezzo_surface" << endl;
	}

	PA->analyze_del_Pezzo_surface(F, evaluate_text, verbose_level);

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given after PA->analyze_del_Pezzo_surface" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given done" << endl;
	}
}





void projective_space_global::do_create_surface(
		projective_space_with_action *PA,
		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description *Surface_Descr,
		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action *&Surf_A,
		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *&SC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::do_create_surface" << endl;
		cout << "projective_space_global::do_create_surface verbose_level=" << verbose_level << endl;
	}

	int q;
	algebraic_geometry::surface_domain *Surf;

	if (f_v) {
		cout << "projective_space_global::do_create_surface before Surface_Descr->get_q" << endl;
	}
	q = Surface_Descr->get_q();
	if (f_v) {
		cout << "projective_space_global::do_create_surface q = " << q << endl;
	}

	if (PA->q != q) {
		cout << "projective_space_global::do_create_surface PA->q != q" << endl;
		exit(1);
	}
	if (PA->n != 3) {
		cout << "projective_space_global::do_create_surface we need a three-dimensional projective space" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_global::do_create_surface before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
	Surf->init(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_global::do_create_surface after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action);

	if (f_v) {
		cout << "projective_space_global::do_create_surface before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_global::do_create_surface after Surf_A->init" << endl;
	}


	if (f_v) {
		cout << "projective_space_global::do_create_surface before Surf_A->create_surface_and_do_report" << endl;
	}

	Surf_A->create_surface(
			Surface_Descr,
			SC,
			verbose_level);

	if (f_v) {
		cout << "projective_space_global::do_create_surface after Surf_A->create_surface_and_do_report" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::do_create_surface done" << endl;
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
		cout << "projective_space_global::conic_type before PA->conic_type" << endl;
	}

	PA->conic_type(Pts, nb_pts, threshold, verbose_level);

	if (f_v) {
		cout << "projective_space_global::conic_type after PA->conic_type" << endl;
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
		cout << "projective_space_global::do_lift_skew_hexagon the number of coordinates must be a multiple of 6" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "Pluecker coordinates of lines:" << endl;
		Int_matrix_print(Pluecker_coords, nb_pts, 6);
	}

	algebraic_geometry::surface_domain *Surf;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action *Surf_A;

	if (PA->n != 3) {
		cout << "projective_space_global::do_lift_skew_hexagon we need a three-dimensional projective space" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
	Surf->init(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action);

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon after Surf_A->init" << endl;
	}




	int i;

	Pts = NEW_lint(nb_pts);

	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Surf_A->Surf->Klein->Pluecker_to_line_rk(Pluecker_coords + i * 6, 0 /*verbose_level*/);
	}

	if (nb_pts != 6) {
		cout << "projective_space_global::do_lift_skew_hexagon nb_pts != 6" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "lines:" << endl;
		Lint_vec_print(cout, Pts, 6);
		cout << endl;
	}


	std::vector<std::vector<long int> > Double_sixes;

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon before Surf_A->complete_skew_hexagon" << endl;
	}

	Surf_A->complete_skew_hexagon(Pts, Double_sixes, verbose_level);

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon after Surf_A->complete_skew_hexagon" << endl;
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
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity I need exactly 36 coefficients for the polarity" << endl;
		exit(1);
	}


	algebraic_geometry::surface_domain *Surf;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action *Surf_A;

	if (PA->n != 3) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity we need a three-dimensional projective space" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
	Surf->init(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action);

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity after Surf_A->init" << endl;
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
				Surf->F->Linear_algebra->mult_matrix_matrix(Pluecker_coords + j * 6, Polarity36,
						Pluecker_coords + 18 + j * 6, 1, 6, 6, 0 /* verbose_level */);
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
				Pts[i] = Surf_A->Surf->Klein->Pluecker_to_line_rk(Pluecker_coords + i * 6, 0 /*verbose_level*/);
			}

			if (nb_pts != 6) {
				cout << "projective_space_global::do_lift_skew_hexagon_with_polarity nb_pts != 6" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "lines:" << endl;
				Lint_vec_print(cout, Pts, 6);
				cout << endl;
			}


			string label;
			char str[1000];

			sprintf(str, "alpha=%d beta=%d", alpha, beta);

			label.assign(str);

			if (f_v) {
				cout << "projective_space_global::do_lift_skew_hexagon_with_polarity before Surf_A->complete_skew_hexagon_with_polarity" << endl;
			}

			Surf_A->complete_skew_hexagon_with_polarity(label, Pts, Polarity36, Double_sixes, verbose_level);

			if (f_v) {
				cout << "projective_space_global::do_lift_skew_hexagon_with_polarity after Surf_A->complete_skew_hexagon_with_polarity" << endl;
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
		cout << "projective_space_global::do_lift_do_lift_skew_hexagon_with_polarityskew_hexagon done" << endl;
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
			cout << "projective_space_global::do_classify_arcs The object given must be a group" << endl;
			exit(1);
		}
		AG = (apps_algebra::any_group *) orbiter_kernel_system::Orbiter->get_object(idx);

#if 0
		if (!LG->f_has_strong_generators) {
			cout << "projective_space_global::do_classify_arcs the group must have strong generators" << endl;
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
			cout << "projective_space_global::do_classify_arcs before Gen->init" << endl;
		}
		Gen->init(
				Arc_generator_description,
				PA,
				gens,
				verbose_level);

		if (f_v) {
			cout << "projective_space_global::do_classify_arcs after Gen->init" << endl;
		}



		if (f_v) {
			cout << "projective_space_global::do_classify_arcs before Gen->main" << endl;
		}
		Gen->main(verbose_level);
		if (f_v) {
			cout << "projective_space_global::do_classify_arcs after Gen->main" << endl;
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
		cout << "projective_space_global::do_classify_cubic_curves before CC->init" << endl;
	}
	CC->init(PA->F, verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves after CC->init" << endl;
	}


	apps_geometry::cubic_curve_with_action *CCA;

	CCA = NEW_OBJECT(apps_geometry::cubic_curve_with_action);

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves before CCA->init" << endl;
	}
	CCA->init(CC, PA->A, verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves after CCA->init" << endl;
	}


	apps_geometry::classify_cubic_curves *CCC;

	CCC = NEW_OBJECT(apps_geometry::classify_cubic_curves);


	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves before CCC->init" << endl;
	}
	CCC->init(
			PA,
			CCA,
			Arc_generator_description,
			verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves after CCC->init" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves before CCC->compute_starter" << endl;
	}
	CCC->compute_starter(verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves after CCC->compute_starter" << endl;
	}

#if 0
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves before CCC->test_orbits" << endl;
	}
	CCC->test_orbits(verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves after CCC->test_orbits" << endl;
	}
#endif

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves before CCC->do_classify" << endl;
	}
	CCC->do_classify(verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves after CCC->do_classify" << endl;
	}


	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves creating cheat sheet" << endl;
	}
	char fname[1000];
	char title[1000];
	char author[1000];
	snprintf(title, 1000, "Cubic Curves in PG$(2,%d)$", PA->F->q);
	strcpy(author, "");
	snprintf(fname, 1000, "Cubic_curves_q%d.tex", PA->F->q);

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
			NULL /* extra_praeamble */);

		fp << "\\subsection*{" << title << "}" << endl;

		if (f_v) {
			cout << "projective_space_global::do_classify_cubic_curves before CCC->report" << endl;
		}
		CCC->report(fp, verbose_level);
		if (f_v) {
			cout << "projective_space_global::do_classify_cubic_curves after CCC->report" << endl;
		}

		L.foot(fp);
	}

	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname << " of size "
		<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves writing cheat sheet on "
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
	Descr->PA = PA;
	Descr->f_degree = TRUE;
	Descr->degree = 4;
	Descr->nb_files = nb;
	Descr->f_algorithm_nauty = TRUE;
	Descr->f_algorithm_substructure = FALSE;


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
	Descr->PA = PA;
	Descr->f_degree = TRUE;
	Descr->degree = degree;
	Descr->nb_files = nb;
	Descr->f_algorithm_nauty = FALSE;
	Descr->f_algorithm_substructure = TRUE;
	Descr->substructure_size = substructure_size;


	Classifier = NEW_OBJECT(canonical_form_classifier);

	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure before Classifier.classify" << endl;
	}
	Classifier->classify(Descr, verbose_level);
	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure after Classifier.classify" << endl;
	}

	Descr->Canon_substructure = Classifier;


	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure before Classifier.report" << endl;
	}
	Classifier->report(fname_classification, verbose_level);
	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure after Classifier.report" << endl;
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

#if 0
	cout << "transversal:" << endl;
	Int_vec_print(cout, Classifier->transversal, Classifier->nb_types);
	cout << endl;

	int i, j;

	cout << "orbit frequencies:" << endl;
	for (i = 0; i < Classifier->nb_types; i++) {
		cout << i << " : ";

		j = Classifier->transversal[i];

		cout << j << " : ";

		if (Classifier->CFS_table[j]) {
			Int_vec_print(cout,
					Classifier->CFS_table[j]->SubSt->orbit_frequencies,
					Classifier->CFS_table[j]->SubSt->nb_orbits);
		}
		else {
			cout << "DNE";
		}

		cout << endl;

	}

	int *orbit_frequencies;
	int nb_orbits = 0;

	for (i = 0; i < Classifier->nb_types; i++) {
		cout << i << " : ";

		j = Classifier->transversal[i];

		cout << j << " : ";

		if (Classifier->CFS_table[j]) {
			nb_orbits = Classifier->CFS_table[j]->SubSt->nb_orbits;
			break;
		}
	}
	if (i == Classifier->nb_types) {
		cout << "cannot determine nb_orbits" << endl;
		exit(1);
	}
	orbit_frequencies = NEW_int(Classifier->nb_types * nb_orbits);

	Int_vec_zero(orbit_frequencies, Classifier->nb_types * nb_orbits);

	for (i = 0; i < Classifier->nb_types; i++) {

		j = Classifier->transversal[i];

		if (Classifier->CFS_table[j]) {
			Int_vec_copy(
					Classifier->CFS_table[j]->SubSt->orbit_frequencies,
					orbit_frequencies + i * nb_orbits,
					nb_orbits);
		}

	}

	tally_vector_data *T;
	int *transversal;
	int *frequency;
	int nb_types;

	T = NEW_OBJECT(tally_vector_data);

	T->init(orbit_frequencies, Classifier->nb_types, nb_orbits, verbose_level);



	T->get_transversal(transversal, frequency, nb_types, verbose_level);


	cout << "Classification of types:" << endl;
	cout << "nb_types=" << nb_types << endl;


	cout << "transversal:" << endl;
	Int_vec_print(cout, transversal, nb_types);
	cout << endl;

	cout << "frequency:" << endl;
	Int_vec_print(cout, frequency, nb_types);
	cout << endl;

	T->print_classes_bigger_than_one(verbose_level);


	file_io Fio;
	std::string fname;
	string_tools String;
	char str[1000];

	fname.assign(fname_mask);
	String.chop_off_extension(fname);
	sprintf(str, "_subset%d_types.csv", size);
	fname.append(str);


	cout << "preparing table" << endl;
	int *table;
	int h;

	table = NEW_int(Classifier->nb_types * (nb_orbits + 2));
	for (i = 0; i < Classifier->nb_types; i++) {

		cout << "preparing table i=" << i << endl;

		h = Classifier->transversal[i];

		cout << "preparing table i=" << i << " h=" << h << endl;

		table[i * (nb_orbits + 2) + 0] = i;

		for (j = 0; j < nb_orbits; j++) {
			table[i * (nb_orbits + 2) + 1 + j] = orbit_frequencies[i * nb_orbits + j];
		}

		table[i * (nb_orbits + 2) + 1 + nb_orbits] = Classifier->CFS_table[h]->SubSt->selected_orbit;

	}

	Fio.int_matrix_write_csv(fname, table, Classifier->nb_types, nb_orbits + 2);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	if (Classifier->nb_types == 1) {
		cout << "preparing detailed information:" << endl;

		i = 0;
		int h;

		substructure_stats_and_selection *SubSt;

		h = Classifier->transversal[i];

		SubSt = Classifier->CFS_table[h]->SubSt;

		cout << "nb_interesting_subsets = "
				<< SubSt->nb_interesting_subsets << endl;
		cout << "interesting subsets: ";
		Orbiter->Lint_vec.print(cout, SubSt->interesting_subsets, SubSt->nb_interesting_subsets);
		cout << endl;

		cout << "selected_orbit=" << SubSt->selected_orbit << endl;

		cout << "generators for the canonical subset:" << endl;
		SubSt->gens->print_generators_tex();


		compute_stabilizer *CS;

		CS = Classifier->CFS_table[h]->CS;

		stabilizer_orbits_and_types *Stab_orbits;

		Stab_orbits = CS->Stab_orbits;

		cout << "reduced_set_size=" << Stab_orbits->reduced_set_size << endl;

		cout << "nb_orbits=" << Stab_orbits->Schreier->nb_orbits << endl;

		cout << "Orbit length:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Stab_orbits->Schreier->orbit_len,
				1,
				Stab_orbits->Schreier->nb_orbits,
				Stab_orbits->Schreier->nb_orbits,
				2);

		cout << "Orbit_patterns:" << endl;
#if 0
		Orbiter->Int_vec.print_integer_matrix_width(cout,
					Stab_orbits->Orbit_patterns,
					CS->SubSt->nb_interesting_subsets,
					Stab_orbits->Schreier->nb_orbits,
					Stab_orbits->Schreier->nb_orbits,
					2);
#endif

		cout << "minimal orbit pattern:" << endl;
		Stab_orbits->print_minimal_orbit_pattern();


		tally_vector_data *T_O;
		int *T_O_transversal;
		int *T_O_frequency;
		int T_O_nb_types;

		T_O = NEW_OBJECT(tally_vector_data);

		T_O->init(Stab_orbits->Orbit_patterns, CS->SubSt->nb_interesting_subsets,
				Stab_orbits->Schreier->nb_orbits, verbose_level);



		T_O->get_transversal(T_O_transversal, T_O_frequency, T_O_nb_types, verbose_level);

		cout << "T_O_nb_types = " << T_O_nb_types << endl;

		cout << "T_O_transversal:" << endl;
		Int_vec_print(cout, T_O_transversal, T_O_nb_types);
		cout << endl;

		cout << "T_O_frequency:" << endl;
		Int_vec_print(cout, T_O_frequency, T_O_nb_types);
		cout << endl;

		T_O->print_classes_bigger_than_one(verbose_level);

		cout << "Types classified:" << endl;
		int u, v;

		for (u = 0; u < T_O_nb_types; u++) {
			v = T_O_transversal[u];

			if (v == Stab_orbits->minimal_orbit_pattern_idx) {
				cout << "*";
			}
			else {
				cout << " ";
			}
			cout << setw(3) << u << " : " << setw(3) << v << " : " << setw(3) << T_O_frequency[u] << " : ";

			Orbiter->Int_vec.print_integer_matrix_width(cout,
						Stab_orbits->Orbit_patterns + v * Stab_orbits->Schreier->nb_orbits,
						1,
						Stab_orbits->Schreier->nb_orbits,
						Stab_orbits->Schreier->nb_orbits,
						2);

		}


		cout << "Types classified in lex order:" << endl;

		int *data;

		data = NEW_int(T_O_nb_types * Stab_orbits->Schreier->nb_orbits);
		for (u = 0; u < T_O_nb_types; u++) {

			cout << setw(3) << u << " : " << setw(3) << T_O->Frequency_in_lex_order[u] << " : ";

			Orbiter->Int_vec.print_integer_matrix_width(cout,
					T_O->Reps_in_lex_order[u],
					1,
					Stab_orbits->Schreier->nb_orbits,
					Stab_orbits->Schreier->nb_orbits,
					2);
			Int_vec_copy(T_O->Reps_in_lex_order[u], data + u * Stab_orbits->Schreier->nb_orbits, Stab_orbits->Schreier->nb_orbits);
		}

		fname.assign(fname_mask);
		String.chop_off_extension(fname);
		sprintf(str, "_subset%d_types_classified.csv", size);
		fname.append(str);

		Fio.int_matrix_write_csv(fname, data, T_O_nb_types, Stab_orbits->Schreier->nb_orbits);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;



		cout << "All canonical_forms:" << endl;
		Orbiter->Lint_vec.matrix_print_width(cout,
				CS->Canonical_forms,
				Stab_orbits->nb_interesting_subsets_reduced,
				Stab_orbits->reduced_set_size,
				Stab_orbits->reduced_set_size,
				2);

		cout << "All canonical_forms, with transporter" << endl;
		CS->print_canonical_sets();


		fname.assign(fname_mask);
		String.chop_off_extension(fname);
		sprintf(str, "_subset%d_cf_input.csv", size);
		fname.append(str);

#if 0
		Fio.lint_matrix_write_csv(fname, CS->Canonical_form_input,
				Stab_orbits->nb_interesting_subsets_reduced,
				Stab_orbits->reduced_set_size);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
#endif

		Fio.write_characteristic_matrix(fname,
				CS->Canonical_form_input,
				Stab_orbits->nb_interesting_subsets_reduced,
				Stab_orbits->reduced_set_size,
				Stab_orbits->nb_interesting_points,
				verbose_level);



		fname.assign(fname_mask);
		String.chop_off_extension(fname);
		sprintf(str, "_subset%d_cf_output.csv", size);
		fname.append(str);

#if 0
		Fio.lint_matrix_write_csv(fname, CS->Canonical_forms, Stab_orbits->nb_interesting_subsets_reduced, Stab_orbits->reduced_set_size);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
#endif

		Fio.write_characteristic_matrix(fname,
				CS->Canonical_forms,
				Stab_orbits->nb_interesting_subsets_reduced,
				Stab_orbits->reduced_set_size,
				Stab_orbits->nb_interesting_points,
				verbose_level);

		fname.assign(fname_mask);
		String.chop_off_extension(fname);
		sprintf(str, "_subset%d_cf_transporter.tex", size);
		fname.append(str);


		std::string title;

		title.assign("Transporter");
		PA->A->write_set_of_elements_latex_file(fname, title,
				CS->Canonical_form_transporter,
				Stab_orbits->nb_interesting_subsets_reduced);


		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	}
#endif



#if 0


	substructure_stats_and_selection *SubSt;

#if 0
	long int *interesting_subsets; // [selected_frequency]
	int nb_interesting_subsets;
		// interesting_subsets are the lvl-subsets of the given set
		// which are of the chosen type.
		// There is nb_interesting_subsets of them.

	strong_generators *gens;
#endif



	compute_stabilizer *CS;
#if 0
	action *A_on_the_set;
		// only used to print the induced action on the set
		// of the set stabilizer

	sims *Stab; // the stabilizer of the original set


	longinteger_object stab_order, new_stab_order;
	int nb_times_orbit_count_does_not_match_up;
	int backtrack_nodes_first_time;
	int backtrack_nodes_total_in_loop;

	stabilizer_orbits_and_types *Stab_orbits;
#if 0
	strong_generators *selected_set_stab_gens;
	sims *selected_set_stab;


	int reduced_set_size; // = set_size - level




	long int *reduced_set1; // [set_size]
	long int *reduced_set2; // [set_size]
	long int *reduced_set1_new_labels; // [set_size]
	long int *reduced_set2_new_labels; // [set_size]
	long int *canonical_set1; // [set_size]
	long int *canonical_set2; // [set_size]

	int *elt1, *Elt1, *Elt1_inv, *new_automorphism, *Elt4;
	int *elt2, *Elt2;
	int *transporter0; // = elt1 * elt2

	longinteger_object go_G;

	schreier *Schreier;
	int nb_orbits;
	int *orbit_count1; // [nb_orbits]
	int *orbit_count2; // [nb_orbits]


	int nb_interesting_subsets_reduced;
	long int *interesting_subsets_reduced;

	int *Orbit_patterns; // [nb_interesting_subsets * nb_orbits]


	int *orbit_to_interesting_orbit; // [nb_orbits]

	int nb_interesting_orbits;
	int *interesting_orbits;

	int nb_interesting_points;
	long int *interesting_points;

	int *interesting_orbit_first;
	int *interesting_orbit_len;

	int local_idx1, local_idx2;
#endif






	action *A_induced;
	longinteger_object induced_go, K_go;

	int *transporter_witness;
	int *transporter1;
	int *transporter2;
	int *T1, *T1v;
	int *T2;

	sims *Kernel_original;
	sims *K; // kernel for building up Stab



	sims *Aut;
	sims *Aut_original;
	longinteger_object ago;
	longinteger_object ago1;
	longinteger_object target_go;


	//union_find_on_k_subsets *U;


	long int *Canonical_forms; // [nb_interesting_subsets_reduced * reduced_set_size]
	int nb_interesting_subsets_rr;
	long int *interesting_subsets_rr;
#endif

	strong_generators *Gens_stabilizer_original_set;
	strong_generators *Gens_stabilizer_canonical_form;


	orbit_of_equations *Orb;

	strong_generators *gens_stab_of_canonical_equation;

	int *trans1;
	int *trans2;
	int *intermediate_equation;



	int *Elt;
	int *eqn2;

	int *canonical_equation;
	int *transporter_to_canonical_form;
#endif

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
	substructure_classifier *SubC;

	SubC = NEW_OBJECT(substructure_classifier);

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





