/*
 * surface_object_properties.cpp
 *
 *  Created on: Oct 13, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;



namespace orbiter {
namespace foundations {


#define MAX_NUMBER_OF_PLANES_FOR_PLANE_TYPE 100000



surface_object_properties::surface_object_properties()
{
	SO = NULL;

	pts_on_lines = NULL;
	lines_on_point = NULL;
	Type_pts_on_lines = NULL;
	Type_lines_on_point = NULL;

	Eckardt_points = NULL;
	Eckardt_points_index = NULL;
	Eckardt_points_schlaefli_labels = NULL;
	Eckardt_point_bitvector_in_Schlaefli_labeling = NULL;
	nb_Eckardt_points = 0;

	Eckardt_points_line_type = NULL;
	Eckardt_points_plane_type = NULL;

	Hesse_planes = NULL;
	nb_Hesse_planes = 0;
	Eckardt_point_Hesse_plane_incidence = NULL;

	nb_axes = 0;
	Axes_index = NULL;
	Axes_Eckardt_points = NULL;
	Axes_line_rank = NULL;



	Double_points = NULL;
	Double_points_index = NULL;
	nb_Double_points = 0;

	Single_points = NULL;
	Single_points_index = NULL;
	nb_Single_points = 0;



	Pts_not_on_lines = NULL;
	nb_pts_not_on_lines = 0;

	nb_planes = 0;

	plane_type_by_lines = NULL;
	plane_type_by_points = NULL;
	C_plane_type_by_points = NULL;

	Tritangent_plane_rk = NULL;
	nb_tritangent_planes = 0;

	Lines_in_tritangent_planes = NULL;

	Trihedral_pairs_as_tritangent_planes = NULL;

	All_Planes = NULL;
	Dual_point_ranks = NULL;

	Adj_line_intersection_graph = NULL;
	Line_neighbors = NULL;
	Line_intersection_pt = NULL;
	Line_intersection_pt_idx = NULL;


	gradient = NULL;

	singular_pts = NULL;

	nb_singular_pts = 0;
	nb_non_singular_pts = 0;

	tangent_plane_rank_global = NULL;
	tangent_plane_rank_dual = NULL;

}

surface_object_properties::~surface_object_properties()
{
	if (Eckardt_points) {
		FREE_lint(Eckardt_points);
	}
	if (Eckardt_points_index) {
		FREE_int(Eckardt_points_index);
	}
	if (Eckardt_points_schlaefli_labels) {
		FREE_int(Eckardt_points_schlaefli_labels);
	}
	if (Eckardt_point_bitvector_in_Schlaefli_labeling) {
		FREE_int(Eckardt_point_bitvector_in_Schlaefli_labeling);
	}
	if (Eckardt_points_line_type) {
		FREE_int(Eckardt_points_line_type);
	}
	if (Eckardt_points_plane_type) {
		FREE_int(Eckardt_points_plane_type);
	}
	if (Hesse_planes) {
		FREE_lint(Hesse_planes);
	}
	if (Eckardt_point_Hesse_plane_incidence) {
		FREE_int(Eckardt_point_Hesse_plane_incidence);
	}
	if (Axes_index) {
		FREE_int(Axes_index);
	}
	if (Axes_Eckardt_points) {
		FREE_lint(Axes_Eckardt_points);
	}
	if (Axes_line_rank) {
		FREE_lint(Axes_line_rank);
	}
	if (Double_points) {
		FREE_lint(Double_points);
	}
	if (Double_points_index) {
		FREE_int(Double_points_index);
	}
	if (Single_points) {
		FREE_lint(Single_points);
	}
	if (Single_points_index) {
		FREE_int(Single_points_index);
	}
	if (Pts_not_on_lines) {
		FREE_lint(Pts_not_on_lines);
	}
	if (pts_on_lines) {
		FREE_OBJECT(pts_on_lines);
	}
	if (lines_on_point) {
		FREE_OBJECT(lines_on_point);
	}
	if (plane_type_by_points) {
		FREE_int(plane_type_by_points);
	}
	if (plane_type_by_lines) {
		FREE_int(plane_type_by_lines);
	}
	if (C_plane_type_by_points) {
		FREE_OBJECT(C_plane_type_by_points);
	}
	if (Type_pts_on_lines) {
		FREE_OBJECT(Type_pts_on_lines);
	}
	if (Type_lines_on_point) {
		FREE_OBJECT(Type_lines_on_point);
	}
	if (Tritangent_plane_rk) {
		FREE_lint(Tritangent_plane_rk);
	}


	if (Lines_in_tritangent_planes) {
		FREE_lint(Lines_in_tritangent_planes);
	}

	if (Trihedral_pairs_as_tritangent_planes) {
		FREE_lint(Trihedral_pairs_as_tritangent_planes);
	}

	if (All_Planes) {
		FREE_lint(All_Planes);
	}
	if (Dual_point_ranks) {
		FREE_int(Dual_point_ranks);
	}
	if (Adj_line_intersection_graph) {
		FREE_int(Adj_line_intersection_graph);
	}
	if (Line_neighbors) {
		FREE_OBJECT(Line_neighbors);
	}
	if (Line_intersection_pt) {
		FREE_int(Line_intersection_pt);
	}
	if (Line_intersection_pt_idx) {
		FREE_int(Line_intersection_pt_idx);
	}

	if (gradient) {
		FREE_int(gradient);
	}

	if (singular_pts) {
		FREE_lint(singular_pts);
	}

	if (tangent_plane_rank_global) {
		FREE_lint(tangent_plane_rank_global);
	}
	if (tangent_plane_rank_dual) {
		FREE_lint(tangent_plane_rank_dual);
	}


}


void surface_object_properties::init(surface_object *SO, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_properties::init" << endl;
	}
	surface_object_properties::SO = SO;



	if (SO->nb_lines == 27) {

		if (f_v) {
			cout << "surface_object_properties::init "
					"before compute_tritangent_planes_by_rank" << endl;
		}
		compute_tritangent_planes_by_rank(0 /*verbose_level*/);
		if (f_v) {
			cout << "surface_object_properties::init "
					"after compute_tritangent_planes_by_rank" << endl;
		}

		if (f_v) {
			cout << "surface_object_properties::init "
					"before compute_Lines_in_tritangent_planes" << endl;
		}
		compute_Lines_in_tritangent_planes(0 /*verbose_level*/);
		if (f_v) {
			cout << "surface_object_properties::init "
					"after compute_Lines_in_tritangent_planes" << endl;
		}

		if (f_v) {
			cout << "surface_object_properties::init "
					"before compute_Trihedral_pairs_as_tritangent_planes" << endl;
		}
		compute_Trihedral_pairs_as_tritangent_planes(0 /*verbose_level*/);
		if (f_v) {
			cout << "surface_object_properties::init "
					"after compute_Trihedral_pairs_as_tritangent_planes" << endl;
		}

	}

	if (f_v) {
		cout << "surface_object_properties::init "
				"before compute_adjacency_matrix_of_line_intersection_graph" << endl;
	}
	compute_adjacency_matrix_of_line_intersection_graph(0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_object_properties::init "
				"after compute_adjacency_matrix_of_line_intersection_graph" << endl;
	}


	if (f_v) {
		cout << "surface_object_properties::init before compute_properties" << endl;
	}
	compute_properties(verbose_level);
	if (f_v) {
		cout << "surface_object_properties::init after compute_properties" << endl;
	}


	if (f_v) {
		cout << "surface_object_properties::init done" << endl;
	}
}

void surface_object_properties::compute_properties(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vvv = (verbose_level >= 3);
	sorting Sorting;
	latex_interface L;

	if (f_v) {
		cout << "surface_object_properties::compute_properties" << endl;
	}



	if (SO->Pts == NULL) {
		if (f_v) {
			cout << "surface_object_properties::compute_properties SO->Pts == NULL" << endl;
		}
		exit(1);
	}


	Sorting.lint_vec_heapsort(SO->Pts, SO->nb_pts);
	if (f_v) {
		cout << "surface_object::compute_properties we found "
				<< SO->nb_pts << " points on the surface" << endl;
	}
	if (f_vvv) {
		cout << "surface_object_properties::compute_properties The points "
				"on the surface are:" << endl;
		L.print_lint_matrix_with_standard_labels(cout,
			SO->Pts, SO->nb_pts, 1, FALSE /* f_tex */);
	}


	if (f_v) {
		cout << "surface_object::compute_properties before compute_singular_points_and_tangent_planes" << endl;
	}
	compute_singular_points_and_tangent_planes(verbose_level);
	if (f_v) {
		cout << "surface_object::compute_properties after compute_singular_points_and_tangent_planes" << endl;
	}



	if (f_v) {
		cout << "surface_object_properties::compute_properties before "
				"Surf->compute_points_on_lines" << endl;
	}
	SO->Surf->compute_points_on_lines(SO->Pts, SO->nb_pts,
		SO->Lines, SO->nb_lines,
		pts_on_lines,
		0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object_properties::compute_properties after "
				"Surf->compute_points_on_lines" << endl;
	}

	pts_on_lines->sort();

	if (f_vvv) {
		cout << "surface_object::compute_properties pts_on_lines:" << endl;
		pts_on_lines->print_table();
	}

	Type_pts_on_lines = NEW_OBJECT(tally);
	Type_pts_on_lines->init_lint(pts_on_lines->Set_size,
		pts_on_lines->nb_sets, FALSE, 0);
	if (f_v) {
		cout << "points on lines:" << endl;
		Type_pts_on_lines->print_naked_tex(cout, TRUE);
		cout << endl;
	}

	pts_on_lines->dualize(lines_on_point, 0 /* verbose_level */);
	if (f_vvv) {
		cout << "surface_object::compute_properties lines_on_point:" << endl;
		lines_on_point->print_table();
	}

	Type_lines_on_point = NEW_OBJECT(tally);
	Type_lines_on_point->init_lint(lines_on_point->Set_size,
		lines_on_point->nb_sets, FALSE, 0);
	if (f_v) {
		cout << "surface_object::compute_properties type of lines_on_point:" << endl;
		Type_lines_on_point->print_naked_tex(cout, TRUE);
		cout << endl;
	}

	if (f_v) {
		cout << "surface_object::compute_properties computing Eckardt points:" << endl;
	}
	Type_lines_on_point->get_class_by_value(Eckardt_points_index,
		nb_Eckardt_points, 3 /* value */, 0 /* verbose_level */);
	Sorting.int_vec_heapsort(Eckardt_points_index, nb_Eckardt_points);
	if (f_v) {
		cout << "surface_object::compute_properties computing Eckardt points done, we found "
				<< nb_Eckardt_points << " Eckardt points" << endl;
	}
	if (f_vvv) {
		cout << "surface_object::compute_properties Eckardt_points_index=";
		int_vec_print(cout, Eckardt_points_index, nb_Eckardt_points);
		cout << endl;
	}
	Eckardt_points = NEW_lint(nb_Eckardt_points);
	int_vec_apply_lint(Eckardt_points_index, SO->Pts,
		Eckardt_points, nb_Eckardt_points);
	if (f_v) {
		cout << "surface_object::compute_properties computing Eckardt points done, we found "
				<< nb_Eckardt_points << " Eckardt points" << endl;
	}
	if (f_vvv) {
		cout << "surface_object::compute_properties Eckardt_points=";
		lint_vec_print(cout, Eckardt_points, nb_Eckardt_points);
		cout << endl;
	}


	if (SO->nb_lines == 27) {
		int p, a, b, c, idx, i;

		Eckardt_points_schlaefli_labels = NEW_int(nb_Eckardt_points);

		for (p = 0; p < nb_Eckardt_points; p++) {

			i = Eckardt_points_index[p];
			if (lines_on_point->Set_size[i] != 3) {
				cout << "surface_object::compute_properties Eckardt point is not on three lines" << endl;
				exit(1);
			}
			a = lines_on_point->Sets[i][0];
			b = lines_on_point->Sets[i][1];
			c = lines_on_point->Sets[i][2];


			idx = SO->Surf->Schlaefli->identify_Eckardt_point(a, b, c, 0 /*verbose_level*/);

			Eckardt_points_schlaefli_labels[p] = idx;
		}
		Eckardt_point_bitvector_in_Schlaefli_labeling = NEW_int(45);
		int_vec_zero(Eckardt_point_bitvector_in_Schlaefli_labeling, 45);
		for (p = 0; p < nb_Eckardt_points; p++) {
			idx = Eckardt_points_schlaefli_labels[p];
			Eckardt_point_bitvector_in_Schlaefli_labeling[idx] = TRUE;
		}

		if (f_v) {
			cout << "surface_object::compute_properties computing axes:" << endl;
		}
		compute_axes(verbose_level);
		if (f_v) {
			cout << "surface_object::compute_properties computing axes done" << endl;
		}

	}
	else {
		Eckardt_points_schlaefli_labels = NULL;
	}




	if (f_v) {
		cout << "computing Double points:" << endl;
	}
	Type_lines_on_point->get_class_by_value(Double_points_index,
		nb_Double_points, 2 /* value */, 0 /* verbose_level */);
	Sorting.int_vec_heapsort(Double_points_index, nb_Double_points);
	if (f_v) {
		cout << "computing Double points done, we found "
				<< nb_Double_points << " Double points" << endl;
	}
	if (f_vvv) {
		cout << "Double_points_index=";
		int_vec_print(cout, Double_points_index, nb_Double_points);
		cout << endl;
	}
	Double_points = NEW_lint(nb_Double_points);
	int_vec_apply_lint(Double_points_index, SO->Pts,
		Double_points, nb_Double_points);
	if (f_v) {
		cout << "computing Double points done, we found "
				<< nb_Double_points << " Double points" << endl;
	}
	if (f_vvv) {
		cout << "Double_points=";
		lint_vec_print(cout, Double_points, nb_Double_points);
		cout << endl;
	}







	if (f_v) {
		cout << "computing Single points:" << endl;
	}
	Type_lines_on_point->get_class_by_value(Single_points_index,
		nb_Single_points, 1 /* value */, 0 /* verbose_level */);
	Sorting.int_vec_heapsort(Single_points_index, nb_Single_points);
	if (f_v) {
		cout << "computing Single points done, we found "
				<< nb_Single_points << " Single points" << endl;
	}
	if (f_vvv) {
		cout << "Single_points_index=";
		int_vec_print(cout, Single_points_index, nb_Single_points);
		cout << endl;
	}
	Single_points = NEW_lint(nb_Single_points);
	int_vec_apply_lint(Single_points_index, SO->Pts,
			Single_points, nb_Single_points);
	if (f_v) {
		cout << "computing Single points done, we found "
				<< nb_Single_points << " Single points" << endl;
	}
	if (f_vvv) {
		cout << "Single_points=";
		lint_vec_print(cout, Single_points, nb_Single_points);
		cout << endl;
	}










	Pts_not_on_lines = NEW_lint(SO->nb_pts);
	lint_vec_copy(SO->Pts, Pts_not_on_lines, SO->nb_pts);
	nb_pts_not_on_lines = SO->nb_pts;

	int i, j, a, b, idx, h;

	for (i = 0; i < pts_on_lines->nb_sets; i++) {

		for (j = 0; j < pts_on_lines->Set_size[i]; j++) {
			a = pts_on_lines->Sets[i][j];
			b = SO->Pts[a];

			if (Sorting.lint_vec_search(Pts_not_on_lines,
				nb_pts_not_on_lines, b, idx, 0)) {
				for (h = idx + 1; h < nb_pts_not_on_lines; h++) {
					Pts_not_on_lines[h - 1] = Pts_not_on_lines[h];
				}
				nb_pts_not_on_lines--;
			}
		}
	}
	if (f_v) {
		cout << "nb_pts_not_on_lines=" << nb_pts_not_on_lines << endl;
	}


	Eckardt_points_line_type = NEW_int(nb_Eckardt_points + 1);

	Eckardt_points_plane_type = NEW_int(SO->Surf->P->Nb_subspaces[2]);

	if (f_v) {
		cout << "surface_object_properties::compute_properties computing line type" << endl;
	}
	SO->Surf->P->line_intersection_type_collected(Eckardt_points, nb_Eckardt_points,
			Eckardt_points_line_type, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object_properties::compute_properties computing line type done" << endl;
	}
	if (f_v) {
		cout << "surface_object_properties::compute_properties computing plane type" << endl;
	}
	SO->Surf->P->plane_intersection_type_basic(
			Eckardt_points, nb_Eckardt_points,
			Eckardt_points_plane_type, 0 /* verbose_level */);
	// type[N_planes]
	if (f_v) {
		cout << "surface_object_properties::compute_properties computing plane type done" << endl;
	}

	{
		tally T_planes;
		int *H_planes;
		sorting Sorting;

		T_planes.init(Eckardt_points_plane_type, SO->Surf->P->Nb_subspaces[2], FALSE, 0);

		T_planes.get_class_by_value(H_planes, nb_Hesse_planes, 9 /* value */,
				0 /* verbose_level */);

		Sorting.int_vec_heapsort(H_planes, nb_Hesse_planes);

		Hesse_planes = NEW_lint(nb_Hesse_planes);
		for (i = 0; i < nb_Hesse_planes; i++) {
			Hesse_planes[i] = H_planes[i];
		}
		FREE_int(H_planes);


		SO->Surf->P->point_plane_incidence_matrix(
					Eckardt_points, nb_Eckardt_points,
					Hesse_planes, nb_Hesse_planes,
					Eckardt_point_Hesse_plane_incidence,
					verbose_level);

		//T_planes.print_file_tex_we_are_in_math_mode(ost, TRUE);
	}


	if (f_v) {
		cout << "surface_object_properties::compute_properties done" << endl;
	}
}

void surface_object_properties::compute_axes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, h, idx;

	if (f_v) {
		cout << "surface_object_properties::compute_axes" << endl;
	}
	nb_axes = 0;
	Axes_index = NEW_int(SO->Surf->Schlaefli->nb_trihedral_pairs * 2);
	Axes_Eckardt_points = NEW_lint(SO->Surf->Schlaefli->nb_trihedral_pairs * 2 * 3);
	for (t = 0; t < SO->Surf->Schlaefli->nb_trihedral_pairs; t++) {
		for (i = 0; i < 2; i++) {
			for (h = 0; h < 3; h++) {
				idx = SO->Surf->Schlaefli->Trihedral_to_Eckardt[6 * t + i * 3 + h];
				if (!Eckardt_point_bitvector_in_Schlaefli_labeling[idx]) {
					break;
				}
			}
			if (h == 3) {
				Axes_index[nb_axes] = 2 * t + i;
				lint_vec_copy(SO->Surf->Schlaefli->Trihedral_to_Eckardt + t * 6 + i * 3,
						Axes_Eckardt_points + nb_axes * 3, 3);
				nb_axes++;
			}
		}
	}
	if (f_v) {
		cout << "we found " << nb_axes << " axes" << endl;
	}
	if (f_v) {
		cout << "surface_object_properties::compute_axes done" << endl;
	}
}

void surface_object_properties::compute_gradient(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "surface_object_properties::compute_gradient" << endl;
	}


	if (f_v) {
		cout << "surface_object_properties::compute_gradient SO->Surf->Poly2_4->get_nb_monomials() = " << SO->Surf->Poly2_4->get_nb_monomials() << endl;
	}

	gradient = NEW_int(4 * SO->Surf->Poly2_4->get_nb_monomials());

	for (i = 0; i < 4; i++) {
		if (f_v) {
			cout << "surface_object_properties::compute_gradient i=" << i << endl;
		}
		if (f_v) {
			cout << "surface_object_properties::compute_gradient eqn_in=";
			int_vec_print(cout, SO->eqn, 20);
			cout << " = " << endl;
			SO->Surf->Poly3_4->print_equation(cout, SO->eqn);
			cout << endl;
		}
		SO->Surf->Partials[i].apply(SO->eqn,
				gradient + i * SO->Surf->Poly2_4->get_nb_monomials(),
				verbose_level - 2);
		if (f_v) {
			cout << "surface_object_properties::compute_gradient "
					"partial=";
			int_vec_print(cout, gradient + i * SO->Surf->Poly2_4->get_nb_monomials(),
					SO->Surf->Poly2_4->get_nb_monomials());
			cout << " = ";
			SO->Surf->Poly2_4->print_equation(cout,
					gradient + i * SO->Surf->Poly2_4->get_nb_monomials());
			cout << endl;
		}
	}


	if (f_v) {
		cout << "surface_object_properties::compute_gradient done" << endl;
	}
}



void surface_object_properties::compute_singular_points_and_tangent_planes(int verbose_level)
// a singular point is a point where all partials vanish
// We compute the set of singular points into Pts[nb_pts]
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);

	if (f_v) {
		cout << "surface_object_properties::compute_singular_points_and_tangent_planes" << endl;
	}
	int h, i;
	long int rk;
	int nb_eqns = 4;
	int v[4];
	int w[4];


	if (f_v) {
		cout << "surface_object_properties::compute_singular_points_and_tangent_planes before compute_gradient" << endl;
	}
	compute_gradient(verbose_level);
	if (f_v) {
		cout << "surface_object_properties::compute_singular_points_and_tangent_planes after compute_gradient" << endl;
	}

	nb_singular_pts = 0;
	nb_non_singular_pts = 0;

	singular_pts = NEW_lint(SO->nb_pts);
	tangent_plane_rank_global = NEW_lint(SO->nb_pts);
	tangent_plane_rank_dual = NEW_lint(SO->nb_pts);
	for (h = 0; h < SO->nb_pts; h++) {
		if (f_vv) {
			cout << "surface_object_properties::compute_singular_points_and_tangent_planes "
					"h=" << h << " / " << SO->nb_pts << endl;
		}
		rk = SO->Pts[h];
		if (f_vv) {
			cout << "surface_object_properties::compute_singular_points_and_tangent_planes "
					"rk=" << rk << endl;
		}
		SO->Surf->unrank_point(v, rk);
		if (f_vv) {
			cout << "surface_object_properties::compute_singular_points_and_tangent_planes "
					"v=";
			int_vec_print(cout, v, 4);
			cout << endl;
		}
		for (i = 0; i < nb_eqns; i++) {
			if (f_vv) {
				cout << "surface_object_properties::compute_singular_points_and_tangent_planes "
						"gradient i=" << i << " / " << nb_eqns << endl;
			}
			if (FALSE) {
				cout << "surface_object_properties::compute_singular_points_and_tangent_planes "
						"gradient " << i << " = ";
				int_vec_print(cout,
						gradient + i * SO->Surf->Poly2_4->get_nb_monomials(),
						SO->Surf->Poly2_4->get_nb_monomials());
				cout << endl;
			}
			w[i] = SO->Surf->Poly2_4->evaluate_at_a_point(
					gradient + i * SO->Surf->Poly2_4->get_nb_monomials(), v);
			if (f_vv) {
				cout << "surface_object_properties::compute_singular_points_and_tangent_planes "
						"value = " << w[i] << endl;
			}
		}
		for (i = 0; i < nb_eqns; i++) {
			if (w[i]) {
				break;
			}
		}

		if (i == nb_eqns) {
			singular_pts[nb_singular_pts++] = rk;
			tangent_plane_rank_global[h] = -1;
		}
		else {
			long int plane_rk;

			plane_rk = SO->Surf->P->plane_rank_using_dual_coordinates_in_three_space(
					w /* eqn4 */,
					0 /* verbose_level*/);
			tangent_plane_rank_global[h] = plane_rk;
			tangent_plane_rank_dual[nb_non_singular_pts++] =
					SO->Surf->P->dual_rank_of_plane_in_three_space(
							plane_rk, 0 /* verbose_level*/);
		}
	}

	sorting Sorting;
	int nb_tangent_planes;

	nb_tangent_planes = nb_non_singular_pts;

	Sorting.lint_vec_sort_and_remove_duplicates(tangent_plane_rank_dual, nb_tangent_planes);


	string fname_tangents;
	file_io Fio;

	fname_tangents.assign("tangents.txt");

	Fio.write_set_to_file_lint(fname_tangents,
			tangent_plane_rank_dual, nb_tangent_planes, verbose_level);

	if (f_v) {
		cout << "Written file " << fname_tangents << " of size " << Fio.file_size(fname_tangents) << endl;
	}


	int *Kernel;
	int *w1;
	int *w2;
	int r, ns;

	Kernel = NEW_int(SO->Surf->Poly3_4->get_nb_monomials() * SO->Surf->Poly3_4->get_nb_monomials());
	w1 = NEW_int(SO->Surf->Poly3_4->get_nb_monomials());
	w2 = NEW_int(SO->Surf->Poly3_4->get_nb_monomials());



	SO->Surf->Poly3_4->vanishing_ideal(tangent_plane_rank_dual, nb_non_singular_pts,
			r, Kernel, 0 /*verbose_level */);

	ns = SO->Surf->Poly3_4->get_nb_monomials() - r; // dimension of null space
	if (f_v) {
		cout << "The system has rank " << r << endl;
		cout << "The ideal has dimension " << ns << endl;
		cout << "and is generated by:" << endl;
		int_matrix_print(Kernel, ns, SO->Surf->Poly3_4->get_nb_monomials());
		cout << "corresponding to the following basis "
				"of polynomials:" << endl;
		for (h = 0; h < ns; h++) {
			SO->Surf->Poly3_4->print_equation(cout, Kernel + h * SO->Surf->Poly3_4->get_nb_monomials());
			cout << endl;
		}
	}

	FREE_int(Kernel);
	FREE_int(w1);
	FREE_int(w2);


	if (f_v) {
		cout << "surface_object_properties::compute_singular_points_and_tangent_planes done" << endl;
	}
}

void surface_object_properties::compute_adjacency_matrix_of_line_intersection_graph(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_properties::compute_adjacency_matrix_of_line_intersection_graph" << endl;
	}

	if (f_v) {
		cout << "surface_object_properties::compute_adjacency_matrix_of_"
				"line_intersection_graph before Surf->compute_adjacency_"
				"matrix_of_line_intersection_graph" << endl;
	}
	SO->Surf->compute_adjacency_matrix_of_line_intersection_graph(
		Adj_line_intersection_graph, SO->Lines, SO->nb_lines, verbose_level - 2);
	if (f_v) {
		cout << "surface_object_properties::compute_adjacency_matrix_of_"
				"line_intersection_graph after Surf->compute_adjacency_"
				"matrix_of_line_intersection_graph" << endl;
	}

	Line_neighbors = NEW_OBJECT(set_of_sets);
	Line_neighbors->init_from_adjacency_matrix(SO->nb_lines,
		Adj_line_intersection_graph, 0 /* verbose_level*/);

	if (f_v) {
		cout << "surface_object_properties::compute_adjacency_matrix_of_"
				"line_intersection_graph before Surf->compute_"
				"intersection_points_and_indices" << endl;
	}
	SO->Surf->compute_intersection_points_and_indices(
		Adj_line_intersection_graph,
		SO->Pts, SO->nb_pts,
		SO->Lines, SO->nb_lines /* nb_lines */,
		Line_intersection_pt, Line_intersection_pt_idx,
		verbose_level);
	if (f_v) {
		cout << "surface_object_properties::compute_adjacency_matrix_of_"
				"line_intersection_graph after Surf->compute_"
				"intersection_points_and_indices" << endl;
	}
#if 0
	Surf->compute_intersection_points(Adj_line_intersection_graph,
			Line_intersection_pt, Line_intersection_pt_idx,
		Lines, SO->nb_lines, verbose_level - 2);
#endif

	if (f_v) {
		cout << "surface_object_properties::compute_adjacency_matrix_of_line_intersection_graph done" << endl;
	}

}

int surface_object_properties::Adj_ij(int i, int j)
{
	return Adj_line_intersection_graph[i * SO->nb_lines + j];
}

void surface_object_properties::compute_plane_type_by_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_properties::compute_plane_type_by_points" << endl;
	}

	nb_planes = SO->Surf->P->Nb_subspaces[2];
	plane_type_by_points = NEW_int(nb_planes);
	plane_type_by_lines = NEW_int(nb_planes);

	if (nb_planes < MAX_NUMBER_OF_PLANES_FOR_PLANE_TYPE) {
		SO->Surf->P->plane_intersection_type_basic(SO->Pts, SO->nb_pts,
			plane_type_by_points, 0 /* verbose_level */);


		C_plane_type_by_points = NEW_OBJECT(tally);

		C_plane_type_by_points->init(plane_type_by_points, nb_planes, FALSE, 0);
		if (f_v) {
			cout << "plane types by points: ";
			C_plane_type_by_points->print_naked(TRUE);
			cout << endl;
		}
	}
	else {
		cout << "surface_object_properties::compute_plane_type_by_points "
				"too many planes, skipping plane type " << endl;
	}


	if (f_v) {
		cout << "surface_object_properties::compute_plane_type_by_points done" << endl;
	}
}


void surface_object_properties::compute_tritangent_planes_by_rank(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "surface_object_properties::compute_tritangent_planes_by_rank" << endl;
	}


	if (SO->nb_lines != 27) {
		cout << "surface_object_properties::compute_tritangent_planes_by_rank "
				"SO->nb_lines != 27 we don't compute the tritangent planes" << endl;
		nb_tritangent_planes = 0;
		return;
	}


	nb_tritangent_planes = 45;
	Tritangent_plane_rk = NEW_lint(45);


	int tritangent_plane_idx;
	int three_lines_idx[3];
	long int three_lines[3];
	int i, r;
	int Basis[6 * 4];
	int base_cols[4];

	for (tritangent_plane_idx = 0;
			tritangent_plane_idx < 45;
			tritangent_plane_idx++) {
		SO->Surf->Schlaefli->Eckardt_points[tritangent_plane_idx].three_lines(
				SO->Surf, three_lines_idx);

		for (i = 0; i < 3; i++) {
			three_lines[i] = SO->Lines[three_lines_idx[i]];
			SO->Surf->Gr->unrank_lint_here(Basis + i * 8,
					three_lines[i], 0 /* verbose_level */);
		}
		r = SO->F->Gauss_simple(Basis, 6, 4,
			base_cols, 0 /* verbose_level */);
		if (r != 3) {
			cout << "surface_object_properties::compute_tritangent_planes_by_rank r != 3" << endl;
			exit(1);
		}
		Tritangent_plane_rk[tritangent_plane_idx] =
				SO->Surf->Gr3->rank_lint_here(Basis, 0 /* verbose_level */);
	}
	if (f_vv) {
		cout << "surface_object_properties::compute_tritangent_planes_by_rank" << endl;
		for (tritangent_plane_idx = 0;
				tritangent_plane_idx < 45;
				tritangent_plane_idx++) {
			cout << tritangent_plane_idx << " : " << Tritangent_plane_rk[tritangent_plane_idx] << endl;
		}
	}
	if (f_v) {
		cout << "surface_object_properties::compute_tritangent_planes_by_rank done" << endl;
	}
}


void surface_object_properties::compute_Lines_in_tritangent_planes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int tritangent_plane_idx, j;

	if (f_v) {
		cout << "surface_object_properties::compute_Lines_in_tritangent_planes" << endl;
	}
	Lines_in_tritangent_planes = NEW_lint(45 * 3);
	for (tritangent_plane_idx = 0;
			tritangent_plane_idx < 45;
			tritangent_plane_idx++) {
		for (j = 0; j < 3; j++) {
			Lines_in_tritangent_planes[tritangent_plane_idx * 3 + j] =
				SO->Lines[SO->Surf->Schlaefli->Lines_in_tritangent_planes[tritangent_plane_idx * 3 + j]];
		}
	}

	if (f_v) {
		cout << "surface_object_properties::compute_Lines_in_tritangent_planes done" << endl;
	}
}

void surface_object_properties::compute_Trihedral_pairs_as_tritangent_planes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "surface_object_properties::compute_Trihedral_pairs_as_tritangent_planes" << endl;
	}
	Trihedral_pairs_as_tritangent_planes = NEW_lint(120 * 6);
	for (i = 0; i < 120; i++) {
		for (j = 0; j < 6; j++) {
			Trihedral_pairs_as_tritangent_planes[i * 6 + j] =
					Tritangent_plane_rk[SO->Surf->Schlaefli->Trihedral_to_Eckardt[i * 6 + j]];
		}
	}

	if (f_v) {
		cout << "surface_object_properties::compute_Trihedral_pairs_as_tritangent_planes done" << endl;
	}
}

void surface_object_properties::compute_planes_and_dual_point_ranks(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "surface_object_properties::compute_planes_and_dual_point_ranks" << endl;
	}

	All_Planes = NEW_lint(SO->Surf->Schlaefli->nb_trihedral_pairs * 6);
	Dual_point_ranks = NEW_int(SO->Surf->Schlaefli->nb_trihedral_pairs * 6);
	//Iso_trihedral_pair = NEW_int(Surf->nb_trihedral_pairs);


	SO->Surf->Trihedral_pairs_to_planes(SO->Lines, All_Planes, 0 /*verbose_level*/);


	for (i = 0; i < SO->Surf->Schlaefli->nb_trihedral_pairs; i++) {
		//cout << "trihedral pair " << i << " / "
		// << Surf->nb_trihedral_pairs << endl;
		for (j = 0; j < 6; j++) {
			Dual_point_ranks[i * 6 + j] =
				SO->Surf->P->dual_rank_of_plane_in_three_space(
						All_Planes[i * 6 + j], 0 /* verbose_level */);
		}

	}
	if (f_v) {
		cout << "surface_object_properties::compute_planes_and_dual_point_ranks done" << endl;
	}
}

void surface_object_properties::print_everything(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_properties::print_everything" << endl;
	}

	if (f_v) {
		cout << "surface_object_properties::print_everything before print_equation" << endl;
	}
	print_equation(ost);
	if (f_v) {
		cout << "surface_object_properties::print_everything after print_equation" << endl;
	}

	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_general" << endl;
	}
	print_general(ost);


	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_lines" << endl;
	}
	print_lines(ost);


	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_points" << endl;
	}
	print_points(ost);


	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_lines_with_points_on_them" << endl;
	}
	print_lines_with_points_on_them(ost);



	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before SO->print_line_intersection_graph" << endl;
	}
	print_line_intersection_graph(ost);

	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_adjacency_matrix_with_intersection_points" << endl;
	}
	print_adjacency_matrix_with_intersection_points(ost);


	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_neighbor_sets" << endl;
	}
	print_neighbor_sets(ost);

	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_tritangent_planes" << endl;
	}
	print_tritangent_planes(ost);


	//SO->print_planes_in_trihedral_pairs(ost);

#if 0
	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_generalized_quadrangle" << endl;
	}
	print_generalized_quadrangle(ost);
#endif

	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_double sixes" << endl;
	}
	print_double_sixes(ost);

	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_half_double_sixes" << endl;
	}
	print_half_double_sixes(ost);

	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_half_double_sixes_numerically" << endl;
	}
	print_half_double_sixes_numerically(ost);

	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_trihedral_pairs" << endl;
	}

	print_trihedral_pairs(ost);

	if (f_v) {
		cout << "surface_object_properties::print_everything "
				"before print_trihedral_pairs_numerically" << endl;
	}

	print_trihedral_pairs_numerically(ost);

	if (f_v) {
		cout << "surface_object_properties::print_everything done" << endl;
	}
}




void surface_object_properties::report_properties(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_properties::report_properties" << endl;
	}


	if (f_v) {
		cout << "surface_object_properties::report_properties before print_general" << endl;
	}
	print_general(ost);


	if (f_v) {
		cout << "surface_object_properties::report_properties before print_lines" << endl;
	}
	print_lines(ost);

	if (f_v) {
		cout << "surface_object_properties::report_properties before print_points" << endl;
	}
	print_points(ost);


	if (f_v) {
		cout << "surface_object_properties::report_properties print_tritangent_planes" << endl;
	}
	print_tritangent_planes(ost);


	if (f_v) {
		cout << "surface_object_properties::report_properties "
				"before print_Steiner_and_Eckardt" << endl;
	}
	print_Steiner_and_Eckardt(ost);

	//SOA->SO->print_planes_in_trihedral_pairs(fp);

#if 0
	if (f_v) {
		cout << "surface_object_properties::report_properties "
				"before print_generalized_quadrangle" << endl;
	}
	print_generalized_quadrangle(ost);
#endif

	if (f_v) {
		cout << "surface_object_properties::report_properties "
				"before print_line_intersection_graph" << endl;
	}
	print_line_intersection_graph(ost);

	if (f_v) {
		cout << "surface_object_properties::report_properties done" << endl;
	}
}

void surface_object_properties::report_properties_simple(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_properties::report_properties_simple" << endl;
	}


	if (f_v) {
		cout << "surface_object_properties::report_properties_simple before print_general" << endl;
	}
	print_general(ost);


	if (f_v) {
		cout << "surface_object_properties::report_properties_simple before print_all_points_on_surface" << endl;
	}
	print_all_points_on_surface(ost);


	ost << "\\subsubsection*{Singular Points}" << endl;
	cout << "surface_object_properties::print_points before print_singular_points" << endl;
	print_singular_points(ost);


	if (f_v) {
		cout << "surface_object_properties::report_properties_simple before print_lines" << endl;
	}
	print_lines(ost);


	ost << "\\subsubsection*{Eckardt Points}" << endl;
	cout << "surface_object_properties::print_points before print_Eckardt_points" << endl;
	print_Eckardt_points(ost);

	ost << "\\subsubsection*{Double Points}" << endl;
	cout << "surface_object_properties::print_points before print_double_points" << endl;
	print_double_points(ost);


	ost << "\\subsubsection*{Single Points}" << endl;
	cout << "surface_object_properties::print_points before print_single_points" << endl;
	print_single_points(ost);

	//ost << "\\subsubsection*{Points on lines}" << endl;
	//cout << "surface_object_properties::print_points before print_points_on_lines" << endl;
	//print_points_on_lines(ost);

	ost << "\\subsubsection*{Points on surface but on no line}" << endl;
	cout << "surface_object_properties::print_points before print_points_on_surface_but_not_on_a_line" << endl;
	print_points_on_surface_but_not_on_a_line(ost);




#if 0
	if (f_v) {
		cout << "surface_object_properties::report_properties_simple print_tritangent_planes" << endl;
	}
	print_tritangent_planes(ost);

	if (f_v) {
		cout << "surface_object_properties::report_properties_simple "
				"before print_Steiner_and_Eckardt" << endl;
	}
	print_Steiner_and_Eckardt(ost);
#endif

	//SOA->SO->print_planes_in_trihedral_pairs(fp);

#if 0
	if (f_v) {
		cout << "surface_object_properties::report_properties_simple "
				"before print_generalized_quadrangle" << endl;
	}
	print_generalized_quadrangle(ost);
#endif

	if (f_v) {
		cout << "surface_object_properties::report_properties_simple "
				"before print_line_intersection_graph" << endl;
	}
	print_line_intersection_graph(ost);

	if (f_v) {
		cout << "surface_object_properties::report_properties_simple done" << endl;
	}
}

void surface_object_properties::print_line_intersection_graph(std::ostream &ost)
{
	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Line Intersection Graph}" << endl;

	//print_adjacency_list(ost);

	print_adjacency_matrix(ost);

	//print_adjacency_matrix_with_intersection_points(ost);

	print_neighbor_sets(ost);
}

void surface_object_properties::print_adjacency_list(std::ostream &ost)
{
	int i, j, m, n, h;
	int *p;
	int *set;


	set = NEW_int(SO->nb_lines);
	m = SO->nb_lines;
	n = SO->nb_lines;
	p = Adj_line_intersection_graph;
	ost << "{\\arraycolsep=1pt" << endl;

	ost << "$$" << endl;
	ost << "\\begin{array}{rr|l|l}" << endl;
	ost << " &  & \\mbox{intersecting} & \\mbox{non-intersecting}";
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << i << " & ";
		if (SO->nb_lines == 27) {
			ost << SO->Surf->Schlaefli->Line_label_tex[i];
		}
		else {
			ost << i;
		}
		ost << " & ";
		h = 0;
		for (j = 0; j < n; j++) {
			if (p[i * n + j]) {
				set[h++] = j;
			}
		}
		for (j = 0; j < h; j++) {
			if (SO->nb_lines == 27) {
				ost << SO->Surf->Schlaefli->Line_label_tex[set[j]];
			}
			else {
				ost << set[j];
			}
			if (j < h - 1) {
				ost << ", ";
			}
		}
		ost << " & ";
		h = 0;
		for (j = 0; j < n; j++) {
			if (p[i * n + j] == 0) {
				set[h++] = j;
			}
		}
		for (j = 0; j < h; j++) {
			if (SO->nb_lines == 27) {
				ost << SO->Surf->Schlaefli->Line_label_tex[set[j]];
			}
			else {
				ost << set[j];
			}
			if (j < h - 1) {
				ost << ", ";
			}
		}
		ost << "\\\\";
		ost << endl;
		}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
	ost << "}%%" << endl;

	FREE_int(set);

}


void surface_object_properties::print_adjacency_matrix(std::ostream &ost)
{
	int i, j, m, n;
	int *p;

	m = SO->nb_lines;
	n = SO->nb_lines;
	p = Adj_line_intersection_graph;
	ost << "{\\arraycolsep=1pt" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{rr|*{" << n << "}r}" << endl;
	ost << " & ";
	for (j = 0; j < n; j++) {
		ost << " & " << j;
	}
	ost << "\\\\" << endl;
	if (SO->nb_lines == 27) {
		ost << " & ";
		for (j = 0; j < n; j++) {
			ost << " & " << SO->Surf->Schlaefli->Line_label_tex[j];
		}
		ost << "\\\\" << endl;
	}
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << i << " & ";
		if (SO->nb_lines == 27) {
			ost << SO->Surf->Schlaefli->Line_label_tex[i];
		}
		for (j = 0; j < n; j++) {
			ost << " & " << p[i * n + j];
		}
		ost << "\\\\";
		ost << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
	ost << "}%%" << endl;
}

void surface_object_properties::print_adjacency_matrix_with_intersection_points(
		std::ostream &ost)
{
	int i, j, m, n, idx;
	int *p;

	m = SO->nb_lines;
	n = SO->nb_lines;
	p = Adj_line_intersection_graph;
	ost << "{\\arraycolsep=1pt" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{rr|*{" << n << "}r}" << endl;
	ost << " & ";
	for (j = 0; j < n; j++) {
		ost << " & " << j;
	}
	ost << "\\\\" << endl;

	if (SO->nb_lines == 27) {
		ost << " & ";
		for (j = 0; j < n; j++) {
			ost << " & " << SO->Surf->Schlaefli->Line_label_tex[j];
		}
		ost << "\\\\" << endl;
	}
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << i;
		if (SO->nb_lines == 27) {
			ost << " & " << SO->Surf->Schlaefli->Line_label_tex[i];
		}
		else {
			ost << " & ";
		}
		for (j = 0; j < n; j++) {
			ost << " & ";
			if (p[i * n + j]) {
				//a = Line_intersection_pt[i * n + j];
				idx = Line_intersection_pt_idx[i * n + j];
				ost << "P_{" << idx << "}";
			}
			else {
				ost << ".";
			}
		}
		ost << "\\\\";
		ost << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
	ost << "}%%" << endl;
}

void surface_object_properties::print_neighbor_sets(std::ostream &ost)
{
	int i, j, h, p;
	sorting Sorting;

	//ost << "\\clearpage" << endl;
	ost << "Neighbor sets in the line intersection graph:\\\\" << endl;
	//Line_neighbors->print_table_tex(ost);
	for (i = 0; i < Line_neighbors->nb_sets; i++) {
		ost << "Line " << i << " intersects " << endl;
		ost << "$$" << endl;
		ost << "\\begin{array}{|r*{"
				<< Line_neighbors->Set_size[i] << "}{|c}|}" << endl;
		//int_set_print_tex(ost, Line_neighbors->Sets[i],
		//Line_neighbors->Set_size[i]);
		ost << "\\hline" << endl;
		ost << "\\mbox{Line} ";
		for (h = 0; h < Line_neighbors->Set_size[i]; h++) {
			j = Line_neighbors->Sets[i][h];
			ost << " & " << j;
		}
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\mbox{in point} ";
		for (h = 0; h < Line_neighbors->Set_size[i]; h++) {
			j = Line_neighbors->Sets[i][h];
			p = Line_intersection_pt[i * SO->nb_lines + j];

#if 0
			if (!Sorting.lint_vec_search_linear(SO->Pts, SO->nb_pts, p, idx)) {
				cout << "surface_object::print_line_intersection_graph "
						"did not find intersection point" << endl;
				exit(1);
			}
			ost << " & " << idx;
#else
			ost << " & " << p;
#endif
		}
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
	}
}

void surface_object_properties::print_planes_in_trihedral_pairs(std::ostream &ost)
{
	latex_interface L;

	ost << "\\clearpage\n\\subsection*{All planes "
			"in trihedral pairs}" << endl;

	ost << "All planes by plane rank:" << endl;

	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
		All_Planes, 30, 6, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		All_Planes + 30 * 6, 30, 6, 30, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		All_Planes + 60 * 6, 30, 6, 60, 0, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		All_Planes + 90 * 6, 30, 6, 90, 0, TRUE /* f_tex */);
	ost << "$$" << endl;



	ost << "All planes by dual point rank:" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
		Dual_point_ranks, 30, 6, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
		Dual_point_ranks + 30 * 6, 30, 6, 30, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
		Dual_point_ranks + 60 * 6, 30, 6, 60, 0, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
		Dual_point_ranks + 90 * 6, 30, 6, 90, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
}

void surface_object_properties::print_tritangent_planes(std::ostream &ost)
{
	int i;
	//int plane_rk, b, v4[4];
	//int Mtx[16];

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Tritangent planes}" << endl;
	ost << "The " << nb_tritangent_planes << " tritangent "
			"planes are:\\\\" << endl;
	for (i = 0; i < nb_tritangent_planes; i++) {
		print_single_tritangent_plane(ost, i);
	}

#if 0
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Tritangent_planes, 9, 5, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "Their dual point ranks are:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Tritangent_plane_dual, 9, 5, TRUE /* f_tex */);
	ost << "$$" << endl;

	for (i = 0; i < nb_tritangent_planes; i++) {
		a = Tritangent_planes[i];
		b = Tritangent_plane_dual[i];
		//b = Surf->P->dual_rank_of_plane_in_three_space(a, 0);
		ost << "plane " << i << " / " << nb_tritangent_planes
				<< " : rank " << a << " is $";
		ost << "\\left[" << endl;
		Surf->Gr3->print_single_generator_matrix_tex(ost, a);
		ost << "\\right]" << endl;
		ost << "$, dual pt rank = $" << b << "$ ";
		PG_element_unrank_modified(*F, v4, 1, 4, b);
		ost << "$=";
		int_vec_print(ost, v4, 4);
		ost << "$\\\\" << endl;
		}

	ost << "The iso types of the tritangent planes are:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			iso_type_of_tritangent_plane, nb_tritangent_planes / 5, 5,
			TRUE /* f_tex */);
	ost << "$$" << endl;

	ost << "Type iso of tritangent planes: ";
	ost << "$$" << endl;
	Type_iso_tritangent_planes->print_naked_tex(ost, TRUE);
	ost << endl;
	ost << "$$" << endl;
	ost << endl;

	ost << "Tritangent\\_plane\\_to\\_Eckardt:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Tritangent_plane_to_Eckardt, nb_tritangent_planes / 5, 5,
			TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "Eckardt\\_to\\_Tritangent\\_plane:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Eckardt_to_Tritangent_plane, nb_tritangent_planes / 5, 5,
			TRUE /* f_tex */);
	ost << "$$" << endl;

	ost << "Trihedral\\_pairs\\_as\\_tritangent\\_planes:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Trihedral_pairs_as_tritangent_planes, 30, 6, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Trihedral_pairs_as_tritangent_planes + 30 * 6, 30, 6, 30, 0,
			TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Trihedral_pairs_as_tritangent_planes + 60 * 6, 30, 6, 60, 0,
			TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Trihedral_pairs_as_tritangent_planes + 90 * 6, 30, 6, 90, 0,
			TRUE /* f_tex */);
	ost << "$$" << endl;
#endif

}

void surface_object_properties::print_single_tritangent_plane(std::ostream &ost, int plane_idx)
{
	long int plane_rk, b;
	int v4[4];
	int Mtx[16];

#if 0
	j = Eckardt_to_Tritangent_plane[plane_idx];
	plane_rk = Tritangent_planes[j];
	b = Tritangent_plane_dual[j];
#else
	plane_rk = Tritangent_plane_rk[plane_idx];
	b = SO->Surf->P->dual_rank_of_plane_in_three_space(
			plane_rk, 0 /* verbose_level */);
#endif
	ost << "$$" << endl;
	ost << "\\pi_{" << SO->Surf->Schlaefli->Eckard_point_label_tex[plane_idx] << "} = ";
	ost << "\\pi_{" << plane_idx << "} = " << plane_rk << " = ";
	//ost << "\\left[" << endl;
	SO->Surf->Gr3->print_single_generator_matrix_tex(ost, plane_rk);
	//ost << "\\right]" << endl;
	ost << " = ";
	SO->Surf->Gr3->print_single_generator_matrix_tex_numerical(ost, plane_rk);

	SO->Surf->Gr3->unrank_lint_here_and_compute_perp(Mtx, plane_rk,
		0 /*verbose_level */);
	SO->F->PG_element_normalize(Mtx + 12, 1, 4);
	ost << "$$" << endl;
	ost << "$$" << endl;
	ost << "=V\\big(" << endl;
	SO->Surf->Poly1_4->print_equation(ost, Mtx + 12);
	ost << "\\big)" << endl;
	ost << "=V\\big(" << endl;
	SO->Surf->Poly1_4->print_equation_numerical(ost, Mtx + 12);
	ost << "\\big)" << endl;
	ost << "$$" << endl;
	ost << "dual pt rank = $" << b << "$ ";
	SO->F->PG_element_unrank_modified(v4, 1, 4, b);
	ost << "$=";
	int_vec_print(ost, v4, 4);
	ost << "$.\\\\" << endl;

}


void surface_object_properties::print_plane_type_by_points(std::ostream &ost)
{
	ost << "\\subsection*{Plane types by points}" << endl;
		//*fp << "$$" << endl;
		//*fp << "\\Big(" << endl;
	C_plane_type_by_points->print_naked_tex(ost, TRUE);
		//*fp << "\\Big)" << endl;
	ost << "\\\\" << endl;
}

void surface_object_properties::print_lines(std::ostream &ost)
{
	ost << "\\subsection*{The " << SO->nb_lines << " Lines}" << endl;
	SO->Surf->print_lines_tex(ost, SO->Lines, SO->nb_lines);
}

void surface_object_properties::print_lines_with_points_on_them(std::ostream &ost)
{
	latex_interface L;

	ost << "\\subsection*{The " << SO->nb_lines << " lines with points on them}" << endl;
	int i, j;
	int pt;


	for (i = 0; i < SO->nb_lines; i++) {
		//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
		SO->Surf->Gr->unrank_lint(SO->Lines[i], 0 /*verbose_level*/);
		ost << "$$" << endl;
		ost << "\\ell_{" << i << "} ";
		if (SO->nb_lines == 27) {
			ost << " = " << SO->Surf->Schlaefli->Line_label_tex[i];
		}
		ost << " = \\left[" << endl;
		//print_integer_matrix_width(cout, Gr->M,
		// k, n, n, F->log10_of_q + 1);
		L.print_integer_matrix_tex(ost, SO->Surf->Gr->M, 2, 4);
		ost << "\\right]_{" << SO->Lines[i] << "}" << endl;
		ost << "$$" << endl;
		ost << "which contains the point set " << endl;
		ost << "$$" << endl;
		ost << "\\{ P_{i} \\mid i \\in ";
		L.lint_set_print_tex(ost, pts_on_lines->Sets[i],
				pts_on_lines->Set_size[i]);
		ost << "\\}." << endl;
		ost << "$$" << endl;

		{
			std::vector<long int> plane_ranks;

			SO->Surf->P->planes_through_a_line(
					SO->Lines[i], plane_ranks,
					0 /*verbose_level*/);

			// print the tangent planes associated with the points on the line:
			ost << "The tangent planes associated with the points on this line are:\\\\" << endl;
			for (j = 0; j < pts_on_lines->Set_size[i]; j++) {

				int w[4];

				pt = pts_on_lines->Sets[i][j];
				ost << j << " : " << pt << " : ";
				SO->Surf->unrank_point(w, SO->Pts[pt]);
				int_vec_print(ost, w, 4);
				ost << " : ";
				if (tangent_plane_rank_global[pt] == -1) {
					ost << " is singular\\\\" << endl;
				}
				else {
					ost << tangent_plane_rank_global[pt] << "\\\\" << endl;
				}
			}
			ost << "The planes in the pencil through the line are:\\\\" << endl;
			for (j = 0; j < plane_ranks.size(); j++) {
				ost << j << " : " << plane_ranks[j] << "\\\\" << endl;

			}
		}
	}
}

void surface_object_properties::print_equation(std::ostream &ost)
{
	ost << "\\subsection*{The equation}" << endl;
	ost << "The equation of the surface ";
	ost << " is :" << endl;
	ost << "$$" << endl;
	SO->Surf->print_equation_tex(ost, SO->eqn);
	ost << endl << "=0\n$$" << endl;
	int_vec_print(ost, SO->eqn, 20);
	ost << "\\\\" << endl;
	ost << "Number of points on the surface " << SO->nb_pts << "\\\\" << endl;
}

void surface_object_properties::print_general(std::ostream &ost)
{
	ost << "\\subsection*{General information}" << endl;



	ost << "Points on lines:" << endl;
	ost << "$$" << endl;
	Type_pts_on_lines->print_naked_tex(ost, TRUE);
	ost << "$$" << endl;
	ost << "Lines on points:" << endl;
	ost << "$$" << endl;
	Type_lines_on_point->print_naked_tex(ost, TRUE);
	ost << "$$" << endl;
}

void surface_object_properties::print_affine_points_in_source_code(std::ostream &ost)
{
	int i, j, cnt;
	int v[4];

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Affine points on surface}" << endl;
	ost << "\\begin{verbatim}" << endl;
	ost << "int Pts[] = {" << endl;
	cnt = 0;
	for (i = 0; i < SO->nb_pts; i++) {
		SO->Surf->unrank_point(v, SO->Pts[i]);
		SO->Surf->F->PG_element_normalize(v, 1, 4);
		if (v[3]) {
			ost << "\t";
			for (j = 0; j < 4; j++) {
				ost << v[j] << ", ";
			}
			ost << endl;
			cnt++;
		}
	}
	ost << "};" << endl;
	ost << "nb_affine_pts = " << cnt << ";" << endl;
	ost << "\\end{verbatim}" << endl;
}

void surface_object_properties::print_points(std::ostream &ost)
{
	ost << "\\subsection*{All Points on surface}" << endl;

	cout << "surface_object_properties::print_points before print_points_on_surface" << endl;
	//print_points_on_surface(ost);
	print_all_points_on_surface(ost);

	ost << "\\subsubsection*{Eckardt Points}" << endl;
	cout << "surface_object_properties::print_points before print_Eckardt_points" << endl;
	print_Eckardt_points(ost);

	ost << "\\subsubsection*{Singular Points}" << endl;
	cout << "surface_object_properties::print_points before print_singular_points" << endl;
	print_singular_points(ost);

	ost << "\\subsubsection*{Double Points}" << endl;
	cout << "surface_object_properties::print_points before print_double_points" << endl;
	print_double_points(ost);

	ost << "\\subsubsection*{Points on lines}" << endl;
	cout << "surface_object_properties::print_points before print_points_on_lines" << endl;
	print_points_on_lines(ost);

	ost << "\\subsubsection*{Points on surface but on no line}" << endl;
	cout << "surface_object_properties::print_points before print_points_on_surface_but_not_on_a_line" << endl;
	print_points_on_surface_but_not_on_a_line(ost);

#if 0
	ost << "\\clearpage" << endl;
	ost << "\\section*{Lines through points}" << endl;
	lines_on_point->print_table_tex(ost);
#endif
}

void surface_object_properties::print_Eckardt_points(std::ostream &ost)
{
	//latex_interface L;
	int i, j, p, a, b, c;
	int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << nb_Eckardt_points
			<< " Eckardt points:\\\\" << endl;





	//ost << "%%\\clearpage" << endl;
	//ost << "The Eckardt points are:\\\\" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	//ost << "\\begin{align*}" << endl;
	for (i = 0; i < nb_Eckardt_points; i++) {

		ost << "$";

		p = Eckardt_points_index[i];

		SO->Surf->unrank_point(v, Eckardt_points[i]);

		ost << i << " : ";
		if (SO->nb_lines == 27) {
			ost << "E_{" << SO->Surf->Schlaefli->Eckard_point_label_tex[Eckardt_points_schlaefli_labels[i]] << "}=";
		}
		if (lines_on_point->Set_size[p] != 3) {
			cout << "surface_object_properties::print_Eckardt_points Eckardt point is not on three lines" << endl;
			exit(1);
		}
		a = lines_on_point->Sets[p][0];
		b = lines_on_point->Sets[p][1];
		c = lines_on_point->Sets[p][2];


		if (SO->nb_lines == 27) {
			//ost << "\\ell_{" << a << "} \\cap ";
			//ost << "\\ell_{" << b << "} \\cap ";
			//ost << "\\ell_{" << c << "}";
			//ost << " = ";
			ost << SO->Surf->Schlaefli->Line_label_tex[a] << " \\cap ";
			ost << SO->Surf->Schlaefli->Line_label_tex[b] << " \\cap ";
			ost << SO->Surf->Schlaefli->Line_label_tex[c];
			ost << " = ";
		}
		//ost << "P_{" << p << "} = ";
		ost << "P_{" << Eckardt_points[i] << "}=";
		ost << "\\bP(";
		//int_vec_print_fully(ost, v, 4);
		for (j = 0; j < 4; j++) {
			SO->F->print_element(ost, v[j]);
			if (j < 4 - 1) {
				ost << ", ";
			}
		}
		ost << ")";
		ost << " = \\bP(";
		for (j = 0; j < 4; j++) {
			ost << v[j];
			if (j < 4 - 1) {
				ost << ", ";
			}
		}
		ost << ")";

		if (i < nb_Eckardt_points - 1) {
			ost << ",";
		}
		else {
			ost << ".";
		}
		ost << "\\; T= " << tangent_plane_rank_global[p];
		ost << "$\\\\" << endl;
		if (tangent_plane_rank_global[p] == -1) {
			cout << "Eckardt point is singular. This should not happen" << endl;
			exit(1);
		}
		}
	//ost << "\\end{align*}" << endl;


#if 0
	{
		//latex_interface L;
		long int *T;

		T = NEW_lint(nb_Eckardt_points);
		for (i = 0; i < nb_Eckardt_points; i++) {
			p = Eckardt_points_index[i];
			T[i] = tangent_plane_rank_global[p];
		}
		ost << "Set of tangent planes: ";
		lint_vec_print(ost, T, nb_Eckardt_points);
		ost << "\\\\" << endl;
		FREE_lint(T);
	}

	latex_interface L;

	ost << "Line type of Eckardt points: $";
	L.print_type_vector_tex(ost, Eckardt_points_line_type, nb_Eckardt_points);
	ost << "$\\\\" << endl;


	{
		ost << "Plane type of Eckardt points: $";
		tally T_planes;

		T_planes.init(Eckardt_points_plane_type, SO->Surf->P->Nb_subspaces[2], FALSE, 0);


		T_planes.print_file_tex_we_are_in_math_mode(ost, TRUE);
		ost << "$\\\\" << endl;
	}

	print_Hesse_planes(ost);

	print_axes(ost);
#endif

}

void surface_object_properties::print_Hesse_planes(std::ostream &ost)
{
	//latex_interface L;
	int i, j;

	ost << "\\subsection*{Hesse planes}" << endl;
	ost << "Number of Hesse planes: " << nb_Hesse_planes << "\\\\" << endl;
	ost << "Set of Hesse planes: ";
	lint_vec_print(ost, Hesse_planes, nb_Hesse_planes);
	ost << "\\\\" << endl;

	SO->Surf->Gr3->print_set_tex(ost, Hesse_planes, nb_Hesse_planes);



	cout << "Hesse plane : rank : Incident Eckardt points\\\\" << endl;

	for (j = 0; j < nb_Hesse_planes; j++) {


		int H[9], cnt, h;

		cnt = 0;
		for (i = 0; i < nb_Eckardt_points; i++) {
			if (Eckardt_point_Hesse_plane_incidence[i * nb_Hesse_planes + j]) {
				if (cnt == 9) {
					cout << "too many points per Hesse plane" << endl;
					exit(1);
				}
				H[cnt++] = i;
			}
		}
		if (cnt != 9) {
			cout << "cnt != 9" << endl;
			exit(1);
		}
		ost << j << " : " << Hesse_planes[j] << " : ";

		for (h = 0; h < 9; h++) {
			i = H[h];
			ost << "$E_{" << SO->Surf->Schlaefli->Eckard_point_label_tex[Eckardt_points_schlaefli_labels[i]] << "}$";
			if (h < 9 - 1) {
				ost << ", ";
			}
		}
		ost << "\\\\" << endl;
	}
}

void surface_object_properties::print_axes(std::ostream &ost)
{
	latex_interface L;
	int i, j, idx, t_idx, t_r, a;

	ost << "\\subsection*{Axes}" << endl;
	ost << "Number of axes: " << nb_axes << "\\\\" << endl;
	ost << "Axes: \\\\" << endl;
	for (i = 0; i < nb_axes; i++) {
		idx = Axes_index[i];
		t_idx = idx / 2;
		t_r = idx % 2;
		ost << i << " : " << idx << " = " << t_idx << "," << t_r << " = " << endl;
		for (j = 0; j < 3; j++) {
			a = Axes_Eckardt_points[i * 3 + j];
			ost << "$E_{" << SO->Surf->Schlaefli->Eckard_point_label_tex[a] << "}$";
			if (j < 3 - 1) {
				ost << ", ";
			}
		}
		ost << "\\\\" << endl;
	}
}

void surface_object_properties::print_singular_points(std::ostream &ost)
{
	latex_interface L;
	int i, j, p;
	int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << nb_singular_pts
			<< " singular points:\\\\" << endl;





	//ost << "%%\\clearpage" << endl;
	//ost << "The Eckardt points are:\\\\" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	ost << "\\begin{align*}" << endl;
	for (i = 0; i < nb_singular_pts; i++) {
		p = singular_pts[i];
		SO->Surf->unrank_point(v, p);
		ost << "S_{" << i << "} &= P_{" << p << "}=\\bP(";
		//int_vec_print_fully(ost, v, 4);
		for (j = 0; j < 4; j++) {
			SO->F->print_element(ost, v[j]);
			if (j < 4 - 1) {
				ost << ", ";
			}
		}
		ost << ")";
		ost << " = \\bP(";
		for (j = 0; j < 4; j++) {
			ost << v[j];
			if (j < 4 - 1) {
				ost << ", ";
			}
		}
		ost << ")";

		ost << "\\\\" << endl;
		}
	ost << "\\end{align*}" << endl;
}



void surface_object_properties::print_double_points(std::ostream &ost)
{
	latex_interface L;
	int i, p, a, b;
	int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << nb_Double_points
			<< " Double points:\\\\" << endl;
	if (nb_Double_points < 1000) {
		ost << "$$" << endl;
		L.lint_vec_print_as_matrix(ost,
				Double_points, nb_Double_points, 10,
				TRUE /* f_tex */);
		ost << "$$" << endl;
		ost << "$$" << endl;
		L.int_vec_print_as_matrix(ost,
				Double_points_index, nb_Double_points, 10,
				TRUE /* f_tex */);
		ost << "$$" << endl;
		//ost << "\\clearpage" << endl;
		ost << "The Double points on the surface are:\\\\" << endl;
		ost << "\\begin{multicols}{2}" << endl;
		ost << "\\noindent" << endl;
		for (i = 0; i < nb_Double_points; i++) {
			SO->Surf->unrank_point(v, Double_points[i]);
			ost << i << " : $P_{" << Double_points_index[i]
					<< "} = P_{" << Double_points[i] << "}=";
			int_vec_print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
		ost << "The double points on the surface are:\\\\" << endl;
		//ost << "\\begin{multicols}{2}" << endl;

		int *pt_idx;

		pt_idx = NEW_int(SO->nb_lines * SO->nb_lines);
		for (i = 0; i < SO->nb_lines * SO->nb_lines; i++) {
			pt_idx[i] = -1;
		}
		for (p = 0; p < SO->nb_pts; p++) {
			if (lines_on_point->Set_size[p] != 2) {
				continue;
				}
			a = lines_on_point->Sets[p][0];
			b = lines_on_point->Sets[p][1];
			if (a > b) {
				a = lines_on_point->Sets[p][1];
				b = lines_on_point->Sets[p][0];
			}
			pt_idx[a * SO->nb_lines + b] = p;
		}
		ost << "\\begin{align*}" << endl;
		for (a = 0; a < SO->nb_lines; a++) {
			for (b = a + 1; b < SO->nb_lines; b++) {
				p = pt_idx[a * SO->nb_lines + b];
				if (p == -1) {
					continue;
				}
				SO->Surf->unrank_point(v, SO->Pts[p]);
				ost << "P_{" << p
						<< "} = P_{" << SO->Pts[p] << "}=";
				int_vec_print_fully(ost, v, 4);


				if (SO->nb_lines == 27) {
					ost << " = ";
					ost << "\\ell_{" << a << "} \\cap ";
					ost << "\\ell_{" << b << "} ";
					ost << " = ";
					ost << SO->Surf->Schlaefli->Line_label_tex[a] << " \\cap ";
					ost << SO->Surf->Schlaefli->Line_label_tex[b];
				}
				ost << "\\\\" << endl;
			}
		}
		ost << "\\end{align*}" << endl;

		FREE_int(pt_idx);
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}
}

void surface_object_properties::print_single_points(std::ostream &ost)
{
	latex_interface L;
	int i, p, a;
	int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << nb_Single_points
			<< " single points:\\\\" << endl;
	if (TRUE /* nb_Single_points < 1000*/) {
		ost << "$$" << endl;
		L.lint_vec_print_as_matrix(ost,
				Single_points, nb_Single_points, 10,
				TRUE /* f_tex */);
		ost << "$$" << endl;
		ost << "$$" << endl;
		L.int_vec_print_as_matrix(ost,
				Single_points_index, nb_Single_points, 10,
				TRUE /* f_tex */);
		ost << "$$" << endl;
		//ost << "\\clearpage" << endl;
		ost << "The single points on the surface are:\\\\" << endl;
		ost << "\\begin{multicols}{2}" << endl;
		ost << "\\noindent" << endl;
		for (i = 0; i < nb_Single_points; i++) {
			SO->Surf->unrank_point(v, Single_points[i]);
			ost << i << " : ";
			// "$P_{" << Single_points_index[i] << "}=";
			p = Single_points_index[i];
			a = lines_on_point->Sets[p][0];
			ost << "$P_{" << Single_points[i] << "}=";
			int_vec_print_fully(ost, v, 4);
			ost << "$";
			if (SO->nb_lines == 27) {
				ost << " lies on line $" << SO->Surf->Schlaefli->Line_label_tex[a] << "$";
			}
			ost << "\\\\" << endl;
		}
		ost << "\\end{multicols}" << endl;
		ost << "The single points on the surface are:\\\\" << endl;
		//ost << "\\begin{multicols}{2}" << endl;
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}
}

void surface_object_properties::print_points_on_surface(std::ostream &ost)
{
	//latex_interface L;
	//int i;
	//int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << SO->nb_pts << " points\\\\" << endl;

#if 0
	if (SO->nb_pts < 1000) {
		ost << "$$" << endl;
		L.lint_vec_print_as_matrix(ost, SO->Pts, SO->nb_pts, 10, TRUE /* f_tex */);
		ost << "$$" << endl;
		//ost << "\\clearpage" << endl;
		ost << "The points on the surface are:\\\\" << endl;
		ost << "\\begin{multicols}{2}" << endl;
		ost << "\\noindent" << endl;
		for (i = 0; i < SO->nb_pts; i++) {
			SO->Surf->unrank_point(v, SO->Pts[i]);
			ost << i << " : $P_{" << i << "} = P_{" << SO->Pts[i] << "}=";
			int_vec_print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}
#endif
}

void surface_object_properties::print_all_points_on_surface(std::ostream &ost)
{
	//latex_interface L;
	//int i;
	//int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << SO->nb_pts << " points:\\\\" << endl;

	if (TRUE /*SO->nb_pts < 1000*/) {
		//ost << "$$" << endl;
		//L.lint_vec_print_as_matrix(ost, SO->Pts, SO->nb_pts, 10, TRUE /* f_tex */);
		//ost << "$$" << endl;
		//ost << "\\clearpage" << endl;
		ost << "The points on the surface are:\\\\" << endl;
		ost << "\\begin{multicols}{3}" << endl;
		ost << "\\noindent" << endl;
		int i;
		int v[4];

		for (i = 0; i < SO->nb_pts; i++) {
			SO->Surf->unrank_point(v, SO->Pts[i]);
			ost << i << " : $P_{" << SO->Pts[i] << "}=";
			int_vec_print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}
}

void surface_object_properties::print_points_on_lines(std::ostream &ost)
{
	latex_interface L;
	int i;

	//ost << "\\clearpage" << endl;
	//pts_on_lines->print_table_tex(ost);
	ost << "\\noindent" << endl;
	for (i = 0; i < pts_on_lines->nb_sets; i++) {
		ost << "Line " << i;

		if (SO->nb_lines == 27) {
			ost << " = "
				"$" << SO->Surf->Schlaefli->Line_label_tex[i]
			<< "$ " << endl;
		}

		ost << "has " << pts_on_lines->Set_size[i]
			<< " points: $\\{ P_{i} \\mid i \\in ";
		L.lint_set_print_tex(ost, pts_on_lines->Sets[i],
				pts_on_lines->Set_size[i]);
		ost << "\\}$\\\\" << endl;
	}

	//ost << "\\clearpage" << endl;
}

void surface_object_properties::print_points_on_surface_but_not_on_a_line(std::ostream &ost)
{
	latex_interface L;
	int i;
	int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << nb_pts_not_on_lines
			<< " points not on any line:\\\\" << endl;
	if (nb_pts_not_on_lines < 1000) {
		ost << "$$" << endl;
		L.lint_vec_print_as_matrix(ost,
				Pts_not_on_lines, nb_pts_not_on_lines, 10,
				TRUE /* f_tex */);
		//print_integer_matrix_with_standard_labels(ost, Pts3,
		//(nb_pts_not_on_lines + 9) / 10, 10, TRUE /* f_tex */);
		ost << "$$" << endl;
		//ost << "%%\\clearpage" << endl;
		ost << "The points on the surface but not "
				"on lines are:\\\\" << endl;
		ost << "\\begin{multicols}{2}" << endl;
		ost << "\\noindent" << endl;
		for (i = 0; i < nb_pts_not_on_lines; i++) {
			SO->Surf->unrank_point(v, Pts_not_on_lines[i]);
			ost << i << " : $P_{" << Pts_not_on_lines[i] << "}=";
			int_vec_print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}
}

void surface_object_properties::print_double_sixes(std::ostream &ost)
{
	//int i, j, a;
	//latex_interface L;

	ost << "\\bigskip" << endl;

	ost << "\\subsection*{Double sixes}" << endl;

	SO->Surf->Schlaefli->latex_table_of_double_sixes(ost);


}

void surface_object_properties::print_half_double_sixes(std::ostream &ost)
{
	//int h, i, j, a;
	//latex_interface L;


	ost << "\\bigskip" << endl;

	ost << "\\subsection*{Half Double sixes}" << endl;

	SO->Surf->Schlaefli->latex_table_of_half_double_sixes(ost);


	//ost << "\\clearpage" << endl;

}

void surface_object_properties::print_half_double_sixes_numerically(std::ostream &ost)
{
	latex_interface L;

	ost << "The half double sixes are:\\\\" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
			SO->Surf->Schlaefli->Half_double_sixes, 36, 6, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
			SO->Surf->Schlaefli->Half_double_sixes + 36 * 6,
		36, 6, 36, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
}

void surface_object_properties::print_trihedral_pairs(std::ostream &ost)
{

	SO->Surf->Schlaefli->print_trihedral_pairs(ost);

	SO->Surf->Schlaefli->latex_triads(ost);
}

void surface_object_properties::print_trihedral_pairs_numerically(std::ostream &ost)
{
	latex_interface L;

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Trihedral pairs}" << endl;
	ost << "The planes in the trihedral pairs in Eckardt "
			"point labeling are:\\\\" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
			SO->Surf->Schlaefli->Trihedral_to_Eckardt, 40, 6, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
			SO->Surf->Schlaefli->Trihedral_to_Eckardt + 40 * 6, 40, 6, 40, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
			SO->Surf->Schlaefli->Trihedral_to_Eckardt + 80 * 6, 40, 6, 80, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
}




void surface_object_properties::latex_table_of_trihedral_pairs_and_clebsch_system(
	std::ostream &ost, int *T, int nb_T)
{
	int t_idx, t;

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Trihedral Pairs and the Clebsch System}" << endl;

	for (t = 0; t < nb_T; t++) {

		t_idx = T[t];


		int F_planes[12];
		int G_planes[12];
		int lambda;
		int equation[20];
		int *system;

		make_equation_in_trihedral_form(t_idx,
			F_planes, G_planes, lambda, equation,
			0 /* verbose_level */);

#if 0
		if (t_idx == 71) {
			int_vec_swap(F_planes, F_planes + 8, 4);
			}
#endif

		SO->Surf->prepare_system_from_FG(F_planes, G_planes,
				lambda, system, 0 /*verbose_level*/);


		ost << "$" << t << " / " << nb_T << "$ ";
		ost << "$T_{" << t_idx << "} = T_{"
			<< SO->Surf->Schlaefli->Trihedral_pair_labels[t_idx]
			<< "} = \\\\" << endl;
		latex_trihedral_pair(ost, t_idx);
		ost << "$\\\\" << endl;
		ost << "$";
		print_equation_in_trihedral_form_equation_only(ost,
				F_planes, G_planes, lambda);
		ost << "$\\\\" << endl;
		//ost << "$";
		SO->Surf->print_system(ost, system);
		//ost << "$\\\\" << endl;
		FREE_int(system);


	}
}


void surface_object_properties::latex_table_of_trihedral_pairs(std::ostream &ost,
		int *T, int nb_T)
{
	int h, i, j, t_idx;

	cout << "surface_object_properties::latex_table_of_trihedral_pairs" << endl;
	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Trihedral Pairs}" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	ost << "\\noindent" << endl;
	for (h = 0; h < nb_T; h++) {
		ost << "$" << h << " / " << nb_T << "$ ";
		t_idx = T[h];
		ost << "$T_{" << t_idx << "} = T_{"
				<< SO->Surf->Schlaefli->Trihedral_pair_labels[t_idx]
				<< "} = \\\\" << endl;
		latex_trihedral_pair(ost, t_idx);
		ost << "$\\\\" << endl;
		ost << "$";
		make_and_print_equation_in_trihedral_form(ost, t_idx);
		ost << "$\\\\" << endl;
	}
	ost << "Dual point ranks: \\\\" << endl;
	for (i = 0; i < SO->Surf->Schlaefli->nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{"
			<< SO->Surf->Schlaefli->Trihedral_pair_labels[i]
			<< "}: \\quad " << endl;
		for (j = 0; j < 6; j++) {
			ost << Dual_point_ranks[i * 6 + j];
			if (j < 6 - 1) {
				ost << ", ";
			}
		}
		ost << "$\\\\" << endl;
	}


#if 0
	ost << "Planes by generator matrix: \\\\" << endl;;
	for (i = 0; i < Surf->nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{"
			<< Surf->Trihedral_pair_labels[i] << "}$" << endl;
		for (j = 0; j < 6; j++) {
			int d;

			d = All_Planes[i * 6 + j];
			ost << "Plane " << j << " has rank " << d << "\\\\" << endl;
			Surf->Gr3->unrank_int(d, 0 /* verbose_level */);
			ost << "$";
			ost << "\\left[";
			print_integer_matrix_tex(ost, Surf->Gr3->M, 3, 4);
			ost << "\\right]";
			ost << "$\\\\" << endl;
		}
	}
#endif
	//ost << "\\end{multicols}" << endl;
	cout << "surface_object_properties::latex_table_of_trihedral_pairs done" << endl;
}

void surface_object_properties::latex_trihedral_pair(std::ostream &ost, int t_idx)
{
	int i, j, e, a;

	//ost << "\\left[" << endl;
	ost << "\\begin{array}{c||ccc|cc}" << endl;
	ost << " & G_0 & G_1 & G_2 & \\mbox{plane} & "
			"\\mbox{dual rank} \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < 3; i++) {
		ost << "F_" << i;
		for (j = 0; j < 3; j++) {
			a = SO->Surf->Schlaefli->Trihedral_pairs[t_idx * 9 + i * 3 + j];
			ost << " & {" << SO->Surf->Schlaefli->Line_label_tex[a] << "=\\atop";
			ost << "\\left[" << endl;
			SO->Surf->Gr->print_single_generator_matrix_tex(ost, SO->Lines[a]);
			ost << "\\right]}" << endl;
		}
		e = SO->Surf->Schlaefli->Trihedral_to_Eckardt[t_idx * 6 + i];
		ost << " & {\\pi_{" << e << "} =\\atop";
#if 0
		t = Eckardt_to_Tritangent_plane[e];
		a = Tritangent_planes[t];
#else
		a = Tritangent_plane_rk[e];
#endif
		ost << "\\left[" << endl;
		SO->Surf->Gr3->print_single_generator_matrix_tex(ost, a);
		ost << "\\right]}" << endl;
		ost << " & ";
		a = Dual_point_ranks[t_idx * 6 + i];
		ost << a << "\\\\" << endl;
	}
	ost << "\\hline" << endl;
	for (j = 0; j < 3; j++) {
		e = SO->Surf->Schlaefli->Trihedral_to_Eckardt[t_idx * 6 + 3 + j];
		ost << " & {\\pi_{" << e << "} =\\atop";
#if 0
		t = Eckardt_to_Tritangent_plane[e];
		a = Tritangent_planes[t];
#else
		a = Tritangent_plane_rk[e];
#endif
		ost << "\\left[" << endl;
		SO->Surf->Gr3->print_single_generator_matrix_tex(ost, a);
		ost << "\\right]}" << endl;
	}
	ost << " & & \\\\" << endl;
	for (j = 0; j < 3; j++) {
		a = Dual_point_ranks[t_idx * 6 + 3 + j];
		ost << " & " << a;
	}
	ost << " & & \\\\" << endl;
	//Surf->latex_trihedral_pair(ost, Surf->Trihedral_pairs + h * 9);
	ost << "\\end{array}" << endl;
	//ost << "\\right]" << endl;
}

void surface_object_properties::make_equation_in_trihedral_form(int t_idx,
	int *F_planes, int *G_planes, int &lambda, int *equation,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, c, h;
	int row_col_Eckardt_points[6];
	int plane_rk[6];
	//int plane_idx[6];

	if (f_v) {
		cout << "surface_object_properties::make_equation_in_trihedral_form t_idx=" << t_idx << endl;
	}

	if (f_v) {
		cout << "Trihedral pair T_{"
			<< SO->Surf->Schlaefli->Trihedral_pair_labels[t_idx] << "}"
			<< endl;
	}

	for (h = 0; h < 6; h++) {
		row_col_Eckardt_points[h] = SO->Surf->Schlaefli->Trihedral_to_Eckardt[t_idx * 6 + h];
	}
	//int_vec_copy(Surf->Trihedral_to_Eckardt + t_idx * 6, row_col_Eckardt_points, 6);
	for (i = 0; i < 6; i++) {
		//plane_idx[i] = Eckardt_to_Tritangent_plane[row_col_Eckardt_points[i]];
		//plane_rk[i] = Tritangent_planes[plane_idx[i]];
		plane_rk[i] = Tritangent_plane_rk[row_col_Eckardt_points[i]];
	}
	for (i = 0; i < 3; i++) {
		c = SO->Surf->P->dual_rank_of_plane_in_three_space(
				plane_rk[i], 0 /* verbose_level */);
		//c = Tritangent_plane_dual[plane_idx[i]];
		SO->F->PG_element_unrank_modified(F_planes + i * 4, 1, 4, c);
	}
	for (i = 0; i < 3; i++) {
		c = SO->Surf->P->dual_rank_of_plane_in_three_space(
				plane_rk[3 + i], 0 /* verbose_level */);
		//c = Tritangent_plane_dual[plane_idx[3 + i]];
		SO->F->PG_element_unrank_modified(G_planes + i * 4, 1, 4, c);
	}
	int evals[6];
	int pt_on_surface[4];
	int a, b, ma, bv, pt;
	int eqn_F[20];
	int eqn_G[20];
	int eqn_G2[20];

	for (h = 0; h < SO->nb_pts; h++) {
		pt = SO->Pts[h];
		SO->F->PG_element_unrank_modified(pt_on_surface, 1, 4, pt);
		for (i = 0; i < 3; i++) {
			evals[i] = SO->Surf->Poly1_4->evaluate_at_a_point(F_planes + i * 4, pt_on_surface);
		}
		for (i = 0; i < 3; i++) {
			evals[3 + i] = SO->Surf->Poly1_4->evaluate_at_a_point(
					G_planes + i * 4, pt_on_surface);
		}
		a = SO->F->mult3(evals[0], evals[1], evals[2]);
		b = SO->F->mult3(evals[3], evals[4], evals[5]);
		if (b) {
			ma = SO->F->negate(a);
			bv = SO->F->inverse(b);
			lambda = SO->F->mult(ma, bv);
			break;
		}
	}
	if (h == SO->nb_pts) {
		cout << "surface_object_properties::make_equation_in_trihedral_form could "
				"not determine lambda" << endl;
		exit(1);
	}

	SO->Surf->multiply_linear_times_linear_times_linear_in_space(F_planes,
		F_planes + 4, F_planes + 8,
		eqn_F, FALSE /* verbose_level */);
	SO->Surf->multiply_linear_times_linear_times_linear_in_space(G_planes,
		G_planes + 4, G_planes + 8,
		eqn_G, FALSE /* verbose_level */);

	int_vec_copy(eqn_G, eqn_G2, 20);
	SO->F->scalar_multiply_vector_in_place(lambda, eqn_G2, 20);
	SO->F->add_vector(eqn_F, eqn_G2, equation, 20);
	SO->F->PG_element_normalize(equation, 1, 20);



	if (f_v) {
		cout << "surface_object_properties::make_equation_in_trihedral_form done" << endl;
	}
}

void surface_object_properties::print_equation_in_trihedral_form(std::ostream &ost,
	int *F_planes, int *G_planes, int lambda)
{

	ost << "\\begin{align*}" << endl;
	ost << "0 & = F_0F_1F_2 + \\lambda G_0G_1G_2\\\\" << endl;
	ost << "& = " << endl;

	print_equation_in_trihedral_form_equation_only(ost, F_planes, G_planes, lambda);
}

void surface_object_properties::print_equation_in_trihedral_form_equation_only(
	std::ostream &ost,
	int *F_planes, int *G_planes, int lambda)
{

	ost << "\\Big(";
	SO->Surf->Poly1_4->print_equation(ost, F_planes);
	ost << "\\Big)";
	ost << "\\Big(";
	SO->Surf->Poly1_4->print_equation(ost, F_planes + 4);
	ost << "\\Big)";
	ost << "\\Big(";
	SO->Surf->Poly1_4->print_equation(ost, F_planes + 8);
	ost << "\\Big)";
	ost << "+ " << lambda;
	ost << "\\Big(";
	SO->Surf->Poly1_4->print_equation(ost, G_planes);
	ost << "\\Big)";
	ost << "\\Big(";
	SO->Surf->Poly1_4->print_equation(ost, G_planes + 4);
	ost << "\\Big)";
	ost << "\\Big(";
	SO->Surf->Poly1_4->print_equation(ost, G_planes + 8);
	ost << "\\Big)";
}

void surface_object_properties::make_and_print_equation_in_trihedral_form(
	std::ostream &ost, int t_idx)
{
	int F_planes[12];
	int G_planes[12];
	int lambda;
	int equation[20];
	//int *system;

	make_equation_in_trihedral_form(t_idx, F_planes, G_planes,
		lambda, equation, 0 /* verbose_level */);
	print_equation_in_trihedral_form_equation_only(ost,
		F_planes, G_planes, lambda);
	//FREE_int(system);
}

int surface_object_properties::compute_transversal_line(
	int line_a, int line_b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "surface_object_properties::compute_transversal_line" << endl;
	}
	if (SO->nb_lines != 27) {
		cout << "surface_object_properties::compute_transversal_line SO->nb_lines != 27" << endl;
		exit(1);
	}
	for (i = 0; i < 27; i++) {
		if (i == line_a) {
			continue;
		}
		if (i == line_b) {
			continue;
		}
		if (Adj_line_intersection_graph[i * 27 + line_a] &&
				Adj_line_intersection_graph[i * 27 + line_b]) {
			break;
		}
	}
	if (i == 27) {
		cout << "surface_object_properties::compute_transversal_line "
				"did not find transversal line" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "surface_object_properties::compute_transversal_line "
				"done" << endl;
	}
	return i;
}

void surface_object_properties::compute_transversal_lines(
	int line_a, int line_b, int *transversals5,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int nb_trans = 0;

	if (f_v) {
		cout << "surface_object_properties::compute_transversal_lines" << endl;
	}
	if (SO->nb_lines != 27) {
		cout << "surface_object_properties::compute_transversal_lines SO->nb_lines != 27" << endl;
		exit(1);
	}
	for (i = 0; i < 27; i++) {
		if (i == line_a) {
			continue;
		}
		if (i == line_b) {
			continue;
		}
		if (Adj_line_intersection_graph[i * 27 + line_a] &&
				Adj_line_intersection_graph[i * 27 + line_b]) {
			transversals5[nb_trans++] = i;
		}
	}
	if (nb_trans != 5) {
		cout << "surface_object_properties::compute_transversal_lines "
				"nb_trans != 5" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "surface_object_properties::compute_transversal_lines "
				"done" << endl;
	}
}




void surface_object_properties::clebsch_map_latex(std::ostream &ost,
	long int *Clebsch_map, int *Clebsch_coeff)
{
	long int i, j, a;
	int v[4];
	int w[3];

	ost << "$$";
	ost << "\\begin{array}{|c|c|c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & P_i & \\mbox{lines} & \\Phi(P_i) & \\Phi(P_i)\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < SO->nb_pts; i++) {
		ost << i;
		ost << " & ";
		a = SO->Pts[i];
		SO->Surf->unrank_point(v, a);
		int_vec_print(ost, v, 4);
		ost << " & ";

		for (j = 0; j < lines_on_point->Set_size[i]; j++) {
			a = lines_on_point->Sets[i][j];
			ost << SO->Surf->Schlaefli->Line_label_tex[a];
			if (j < lines_on_point->Set_size[i] - 1) {
				ost << ", ";
			}
		}
		ost << " & ";


		if (Clebsch_map[i] >= 0) {
			int_vec_print(ost, Clebsch_coeff + i * 4, 4);
		}
		else {
			ost << "\\mbox{undef}";
		}
		ost << " & ";
		if (Clebsch_map[i] >= 0) {
			SO->Surf->P2->unrank_point(w, Clebsch_map[i]);
			int_vec_print(ost, w, 3);
		}
		else {
			ost << "\\mbox{undef}";
		}
		ost << "\\\\" << endl;
		if (((i + 1) % 30) == 0) {
			ost << "\\hline" << endl;
			ost << "\\end{array}" << endl;
			ost << "$$" << endl;
			ost << "$$";
			ost << "\\begin{array}{|c|c|c|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "i & P_i & \\mbox{lines} & \\Phi(P_i) & "
					"\\Phi(P_i)\\\\" << endl;
			ost << "\\hline" << endl;
		}
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
}


void surface_object_properties::print_Steiner_and_Eckardt(std::ostream &ost)
{
#if 0
	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Eckardt Points}" << endl;
	latex_table_of_Eckardt_points(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Tritangent Planes}" << endl;
	latex_table_of_tritangent_planes(ost);
#endif

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Steiner Trihedral Pairs}" << endl;
	latex_table_of_trihedral_pairs(ost);

}

void surface_object_properties::latex_table_of_trihedral_pairs(std::ostream &ost)
{
	int i;

	cout << "surface_object_properties::latex_table_of_trihedral_pairs" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < SO->Surf->Schlaefli->nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{"
				<< SO->Surf->Schlaefli->Trihedral_pair_labels[i]
				<< "} = $\\\\" << endl;
		ost << "$" << endl;
		//ost << "\\left[" << endl;
		//ost << "\\begin{array}" << endl;
		latex_trihedral_pair(ost,
				SO->Surf->Schlaefli->Trihedral_pairs + i * 9,
				SO->Surf->Schlaefli->Trihedral_to_Eckardt + i * 6);
		//ost << "\\end{array}" << endl;
		//ost << "\\right]" << endl;
		ost << "$\\\\" << endl;
#if 0
		ost << "planes: $";
		int_vec_print(ost, Trihedral_to_Eckardt + i * 6, 6);
		ost << "$\\\\" << endl;
#endif
	}
	//ost << "\\end{multicols}" << endl;

	//print_trihedral_pairs(ost);

	cout << "surface_object_properties::latex_table_of_trihedral_pairs done" << endl;
}

void surface_object_properties::latex_trihedral_pair(std::ostream &ost, int *T, long int *TE)
{
	int i, j, plane_rk;
	int Mtx[16];

	ost << "\\begin{array}{*{" << 3 << "}{c}|c}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			SO->Surf->Schlaefli->print_line(ost, T[i * 3 + j]);
			ost << " & ";
		}
		ost << "\\pi_{";
		SO->Surf->Schlaefli->Eckardt_points[TE[i]].latex_index_only(ost);
		ost << "}=" << endl;
#if 0
		t = Eckardt_to_Tritangent_plane[TE[i]];
		plane_rk = Tritangent_planes[t];
#else
		plane_rk = Tritangent_plane_rk[TE[i]];
#endif
		SO->Surf->Gr3->unrank_lint_here_and_compute_perp(Mtx, plane_rk,
			0 /*verbose_level */);
		SO->F->PG_element_normalize(Mtx + 12, 1, 4);
		ost << "V\\big(";
		SO->Surf->Poly1_4->print_equation(ost, Mtx + 12);
		ost << "\\big)=" << plane_rk;
		ost << "\\\\" << endl;
	}
	ost << "\\hline" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\pi_{";
		SO->Surf->Schlaefli->Eckardt_points[TE[3 + j]].latex_index_only(ost);
		ost << "} & ";
	}
	ost << "\\\\" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\multicolumn{4}{l}{" << endl;
#if 0
		t = Eckardt_to_Tritangent_plane[TE[3 + j]];
		plane_rk = Tritangent_planes[t];
#else
		plane_rk = Tritangent_plane_rk[TE[i]];
#endif
		ost << "\\pi_{";
		SO->Surf->Schlaefli->Eckardt_points[TE[3 + j]].latex_index_only(ost);
		ost << "}=" << endl;
		ost << "V\\big(" << endl;
		SO->Surf->Gr3->unrank_lint_here_and_compute_perp(Mtx, plane_rk,
			0 /*verbose_level */);
		SO->F->PG_element_normalize(Mtx + 12, 1, 4);
		SO->Surf->Poly1_4->print_equation(ost, Mtx + 12);
		ost << "\\big)=" << plane_rk << "}\\\\" << endl;
	}
	ost << "\\\\" << endl;
	ost << "\\end{array}" << endl;
}




}}
