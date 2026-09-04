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


void projective_space_global::create_all_transvections(
		projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::create_all_transvections" << endl;
	}

	int q;
	int d;
	long int N, big_N;
	long int rk1, rk2, a;

	d = PA->P->Subspaces->n + 1;
	q = PA->F->q;

	geometry::other_geometry::geometry_global Geometry_global;

	N = Geometry_global.nb_PG_elements(
			d - 1, q);

	big_N = N * N * (q - 1);

	int *vec_v;
	int *vec_u;

	vec_v = NEW_int(d);
	vec_u = NEW_int(d);



	// First, we count the number of transvections:




	long int cur;
	long int nb_transvections;

	cur = 0;


	for (rk1 = 0; rk1 < N; rk1++) {

		PA->F->Projective_space_basic->PG_element_unrank_modified(
				vec_v, 1 /*stride*/, d, rk1);

		PA->F->Projective_space_basic->PG_element_normalize_from_front(
					vec_v, 1, d);


		for (rk2 = 0; rk2 < N; rk2++) {

			PA->F->Projective_space_basic->PG_element_unrank_modified(
					vec_u, 1 /*stride*/, d, rk2);

			PA->F->Projective_space_basic->PG_element_normalize_from_front(
						vec_u, 1, d);


			for (a = 1; a < q; a++) {

				data_structures_groups::vector_ge *vec;

				if (PA->create_transvection(
						a, vec_v, vec_u, d,
						vec,
						verbose_level - 2)) {

					FREE_OBJECT(vec);

					cur++;
				}



			}
		}
	}

	nb_transvections = cur;




	if (f_v) {
		cout << "projective_space_global::create_all_transvections big_N = " << big_N << endl;
		cout << "projective_space_global::create_all_transvections nb_transvections = " << nb_transvections << endl;
	}



	// Now we create the transvections themselves
	// and save the data to a csv file:



	data_structures_groups::vector_ge *Vec;

	Vec = NEW_OBJECT(data_structures_groups::vector_ge);

	Vec->init(PA->A, 0 /* verbose_level */);
	Vec->allocate(nb_transvections, 0 /* verbose_level */);




	// save the data to a csv file:


	other::orbiter_kernel_system::file_io Fio;
	std::string fname;

	fname = "PG_" + std::to_string(d - 1) + "_" + std::to_string(q) + "_all_transvections.csv";

	std::string *Col_headings;

	int nb_rows = nb_transvections;
	int nb_cols = 8;

	string *Table;


	Table = new string[nb_rows * nb_cols];

	cur = 0;

	for (rk1 = 0; rk1 < N; rk1++) {

		PA->F->Projective_space_basic->PG_element_unrank_modified(
				vec_v, 1 /*stride*/, d, rk1);

		PA->F->Projective_space_basic->PG_element_normalize_from_front(
					vec_v, 1, d);

		for (rk2 = 0; rk2 < N; rk2++) {

			PA->F->Projective_space_basic->PG_element_unrank_modified(
					vec_u, 1 /*stride*/, d, rk2);

			PA->F->Projective_space_basic->PG_element_normalize_from_front(
						vec_u, 1, d);

			for (a = 1; a < q; a++) {


				data_structures_groups::vector_ge *vec;

				if (PA->create_transvection(
						a, vec_v, vec_u, d,
						vec,
						verbose_level - 2)) {

					PA->A->Group_element->element_move(vec->ith(0), Vec->ith(cur), 0 /* verbose_level */);


					Table[cur * nb_cols + 0] = std::to_string(cur);
					Table[cur * nb_cols + 1] = std::to_string(rk1);
					Table[cur * nb_cols + 2] = std::to_string(rk2);
					Table[cur * nb_cols + 3] = std::to_string(a);
					Table[cur * nb_cols + 4] = "\"" + Int_vec_stringify(vec_v, d) + "\"";
					Table[cur * nb_cols + 5] = "\"" + Int_vec_stringify(vec_u, d) + "\"";


					{

						Table[cur * nb_cols + 6] = "\"" + Int_vec_stringify(Vec->ith(cur), d * d) + "\"";
					}
#if 0
					{
					std::string options;
					string s;

					s = PA->A->Group_element->element_stringify(
							Vec->ith(cur), options);


					Table[cur * nb_cols + 7] = "\"" + s + "\"";
					}
#endif

					FREE_OBJECT(vec);


					cur++;
				}




			}
		}
	}


	Col_headings = new string [nb_cols];

	Col_headings[0] = "idx";
	Col_headings[1] = "rk1";
	Col_headings[2] = "rk2";
	Col_headings[3] = "s";
	Col_headings[4] = "v";
	Col_headings[5] = "u";
	Col_headings[6] = "Element";
	Col_headings[7] = "ElementTex";


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "orthogonal_group::create_all_transvections "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	delete [] Col_headings;
	delete [] Table;






	FREE_int(vec_v);
	FREE_int(vec_u);

	FREE_OBJECT(Vec);


	if (f_v) {
		cout << "projective_space_global::create_all_transvections done" << endl;
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





