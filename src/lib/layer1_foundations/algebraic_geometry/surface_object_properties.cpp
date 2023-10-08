/*
 * surface_object_properties.cpp
 *
 *  Created on: Oct 13, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


#define MAX_NUMBER_OF_PLANES_FOR_PLANE_TYPE 100000



surface_object_properties::surface_object_properties()
{
	SO = NULL;

	// point properties:

	pts_on_lines = NULL;
	f_is_on_line = NULL;


	Pluecker_coordinates = NULL;
	Pluecker_rk = NULL;

	Eckardt_points = NULL;
	Eckardt_points_index = NULL;
	Eckardt_points_schlaefli_labels = NULL;
	Eckardt_point_bitvector_in_Schlaefli_labeling = NULL;
	nb_Eckardt_points = 0;

	Eckardt_points_line_type = NULL;
	Eckardt_points_plane_type = NULL;


	lines_on_point = NULL;
	Type_pts_on_lines = NULL;
	Type_lines_on_point = NULL;

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


	SmoothProperties = NULL;

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

	// point properties:

	if (pts_on_lines) {
		FREE_OBJECT(pts_on_lines);
	}
	if (f_is_on_line) {
		FREE_int(f_is_on_line);
	}

	if (Pluecker_coordinates) {
		FREE_int(Pluecker_coordinates);
	}
	if (Pluecker_rk) {
		FREE_lint(Pluecker_rk);
	}

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



	if (lines_on_point) {
		FREE_OBJECT(lines_on_point);
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

	if (SmoothProperties) {
		FREE_OBJECT(SmoothProperties);
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


		SmoothProperties = NEW_OBJECT(smooth_surface_object_properties);

		SmoothProperties->init(SO, verbose_level);
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
		cout << "surface_object_properties::init "
				"before compute_properties" << endl;
	}
	compute_properties(verbose_level);
	if (f_v) {
		cout << "surface_object_properties::init "
				"after compute_properties" << endl;
	}

	if (SO->nb_lines == 27) {

		if (f_v) {
			cout << "surface_object_properties::init "
					"before SmoothProperties->init_roots" << endl;
		}
		SmoothProperties->init_roots(verbose_level);
		if (f_v) {
			cout << "surface_object_properties::init "
					"after SmoothProperties->init_roots" << endl;
		}
	}

	if (f_v) {
		cout << "surface_object_properties::init done" << endl;
	}
}

void surface_object_properties::compute_properties(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vvv = (verbose_level >= 3);
	data_structures::sorting Sorting;
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "surface_object_properties::compute_properties" << endl;
	}

	int i;

	Pluecker_coordinates = NEW_int(SO->nb_lines * 6);
	Pluecker_rk = NEW_lint(SO->nb_lines);

	for (i = 0; i < SO->nb_lines; i++) {

		int v6[6];

		SO->Surf->Gr->Pluecker_coordinates(
				SO->Lines[i], v6, 0 /* verbose_level */);

		Int_vec_copy(
				v6, Pluecker_coordinates + i * 6, 6);

		Pluecker_rk[i] = SO->Surf->O->Orthogonal_indexing->Qplus_rank(
				v6, 1, 5, 0 /* verbose_level*/);
			// destroys v6[]
	}

	if (SO->Pts == NULL) {
		if (f_v) {
			cout << "surface_object_properties::compute_properties "
					"SO->Pts == NULL" << endl;
		}
		exit(1);
	}



	Sorting.lint_vec_heapsort(SO->Pts, SO->nb_pts);

	if (f_v) {
		cout << "surface_object::compute_properties "
				"we found "
				<< SO->nb_pts << " points on the surface" << endl;
	}
	if (f_vvv) {
		cout << "surface_object_properties::compute_properties "
				"The points on the surface are:" << endl;
		L.print_lint_matrix_with_standard_labels(cout,
			SO->Pts, SO->nb_pts, 1, false /* f_tex */);
	}


	if (f_v) {
		cout << "surface_object::compute_properties "
				"before compute_singular_points_and_tangent_planes" << endl;
	}
	compute_singular_points_and_tangent_planes(verbose_level);
	if (f_v) {
		cout << "surface_object::compute_properties "
				"after compute_singular_points_and_tangent_planes" << endl;
	}



	if (f_v) {
		cout << "surface_object_properties::compute_properties before "
				"Surf->compute_points_on_lines" << endl;
	}
	SO->Surf->compute_points_on_lines(
			SO->Pts, SO->nb_pts,
		SO->Lines, SO->nb_lines,
		pts_on_lines,
		f_is_on_line,
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

	Type_pts_on_lines = NEW_OBJECT(data_structures::tally);
	Type_pts_on_lines->init_lint(pts_on_lines->Set_size,
		pts_on_lines->nb_sets, false, 0);
	if (f_v) {
		cout << "points on lines:" << endl;
		Type_pts_on_lines->print_bare_tex(cout, true);
		cout << endl;
	}

	pts_on_lines->dualize(lines_on_point, 0 /* verbose_level */);
	if (f_vvv) {
		cout << "surface_object::compute_properties lines_on_point:" << endl;
		lines_on_point->print_table();
	}

	Type_lines_on_point = NEW_OBJECT(data_structures::tally);
	Type_lines_on_point->init_lint(lines_on_point->Set_size,
		lines_on_point->nb_sets, false, 0);
	if (f_v) {
		cout << "surface_object::compute_properties "
				"type of lines_on_point:" << endl;
		Type_lines_on_point->print_bare_tex(cout, true);
		cout << endl;
	}

	if (f_v) {
		cout << "surface_object::compute_properties "
				"computing Eckardt points:" << endl;
	}
	Type_lines_on_point->get_class_by_value(Eckardt_points_index,
		nb_Eckardt_points, 3 /* value */, 0 /* verbose_level */);
	Sorting.int_vec_heapsort(Eckardt_points_index, nb_Eckardt_points);
	if (f_v) {
		cout << "surface_object::compute_properties "
				"computing Eckardt points done, we found "
				<< nb_Eckardt_points << " Eckardt points" << endl;
	}
	if (f_vvv) {
		cout << "surface_object::compute_properties Eckardt_points_index=";
		Int_vec_print(cout, Eckardt_points_index, nb_Eckardt_points);
		cout << endl;
	}
	Eckardt_points = NEW_lint(nb_Eckardt_points);
	Int_vec_apply_lint(Eckardt_points_index, SO->Pts,
		Eckardt_points, nb_Eckardt_points);
	if (f_v) {
		cout << "surface_object::compute_properties "
				"computing Eckardt points done, we found "
				<< nb_Eckardt_points << " Eckardt points" << endl;
	}
	if (f_vvv) {
		cout << "surface_object::compute_properties Eckardt_points=";
		Lint_vec_print(cout, Eckardt_points, nb_Eckardt_points);
		cout << endl;
	}


	if (SO->nb_lines == 27) {




		int p, a, b, c, idx;

		Eckardt_points_schlaefli_labels = NEW_int(nb_Eckardt_points);

		for (p = 0; p < nb_Eckardt_points; p++) {

			i = Eckardt_points_index[p];
			if (lines_on_point->Set_size[i] != 3) {
				cout << "surface_object::compute_properties "
						"Eckardt point is not on three lines" << endl;
				exit(1);
			}
			a = lines_on_point->Sets[i][0];
			b = lines_on_point->Sets[i][1];
			c = lines_on_point->Sets[i][2];


			idx = SO->Surf->Schlaefli->identify_Eckardt_point(
					a, b, c, 0 /*verbose_level*/);

			Eckardt_points_schlaefli_labels[p] = idx;
		}
		Eckardt_point_bitvector_in_Schlaefli_labeling = NEW_int(45);
		Int_vec_zero(Eckardt_point_bitvector_in_Schlaefli_labeling, 45);
		for (p = 0; p < nb_Eckardt_points; p++) {
			idx = Eckardt_points_schlaefli_labels[p];
			Eckardt_point_bitvector_in_Schlaefli_labeling[idx] = true;
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
		Int_vec_print(cout, Double_points_index, nb_Double_points);
		cout << endl;
	}
	Double_points = NEW_lint(nb_Double_points);
	Int_vec_apply_lint(Double_points_index, SO->Pts,
		Double_points, nb_Double_points);
	if (f_v) {
		cout << "computing Double points done, we found "
				<< nb_Double_points << " Double points" << endl;
	}
	if (f_vvv) {
		cout << "Double_points=";
		Lint_vec_print(cout, Double_points, nb_Double_points);
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
		Int_vec_print(cout, Single_points_index, nb_Single_points);
		cout << endl;
	}
	Single_points = NEW_lint(nb_Single_points);
	Int_vec_apply_lint(Single_points_index, SO->Pts,
			Single_points, nb_Single_points);
	if (f_v) {
		cout << "computing Single points done, we found "
				<< nb_Single_points << " Single points" << endl;
	}
	if (f_vvv) {
		cout << "Single_points=";
		Lint_vec_print(cout, Single_points, nb_Single_points);
		cout << endl;
	}


	Pts_not_on_lines = NEW_lint(SO->nb_pts);
	Lint_vec_copy(SO->Pts, Pts_not_on_lines, SO->nb_pts);
	nb_pts_not_on_lines = SO->nb_pts;

	int j, a, b, idx, h;

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

	Eckardt_points_plane_type = NEW_int(SO->Surf->P->Subspaces->Nb_subspaces[2]);

	if (f_v) {
		cout << "surface_object_properties::compute_properties "
				"computing line type" << endl;
	}
	SO->Surf->P->Subspaces->line_intersection_type_collected(
			Eckardt_points, nb_Eckardt_points,
			Eckardt_points_line_type, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object_properties::compute_properties "
				"computing line type done" << endl;
	}
	if (f_v) {
		cout << "surface_object_properties::compute_properties "
				"computing plane type" << endl;
	}
	SO->Surf->P->Subspaces->plane_intersection_type_basic(
			Eckardt_points, nb_Eckardt_points,
			Eckardt_points_plane_type, 0 /* verbose_level */);
	// type[N_planes]
	if (f_v) {
		cout << "surface_object_properties::compute_properties "
				"computing plane type done" << endl;
	}

	{
		data_structures::tally T_planes;
		int *H_planes;
		data_structures::sorting Sorting;

		T_planes.init(Eckardt_points_plane_type,
				SO->Surf->P->Subspaces->Nb_subspaces[2], false, 0);

		T_planes.get_class_by_value(H_planes,
				nb_Hesse_planes, 9 /* value */,
				0 /* verbose_level */);

		Sorting.int_vec_heapsort(H_planes, nb_Hesse_planes);

		Hesse_planes = NEW_lint(nb_Hesse_planes);
		for (i = 0; i < nb_Hesse_planes; i++) {
			Hesse_planes[i] = H_planes[i];
		}
		FREE_int(H_planes);


		SO->Surf->P->Subspaces->point_plane_incidence_matrix(
					Eckardt_points, nb_Eckardt_points,
					Hesse_planes, nb_Hesse_planes,
					Eckardt_point_Hesse_plane_incidence,
					verbose_level);

		//T_planes.print_file_tex_we_are_in_math_mode(ost, true);
	}


	if (f_v) {
		cout << "surface_object_properties::compute_properties done" << endl;
	}
}

void surface_object_properties::compute_axes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, h, l, idx;

	if (f_v) {
		cout << "surface_object_properties::compute_axes" << endl;
	}

	Axes_line_rank = NEW_lint(240); // at most 240 axes

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
				long int E[3];
				long int line_rk;
				long int line_rk1;
				long int line_rk2;
				int m;

				Axes_index[nb_axes] = 2 * t + i;
				Lint_vec_copy(SO->Surf->Schlaefli->Trihedral_to_Eckardt + t * 6 + i * 3,
						Axes_Eckardt_points + nb_axes * 3, 3);
				for (l = 0; l < 3; l++) {
					idx = SO->Surf->Schlaefli->Trihedral_to_Eckardt[6 * t + i * 3 + l];
					for (m = 0; m < nb_Eckardt_points; m++) {
						if (Eckardt_points_schlaefli_labels[m] == idx) {
							break;
						}
					}
					if (m == nb_Eckardt_points) {
						cout << "surface_object_properties::compute_axes "
								"cannot find Eckardt point with Schlaefli label" << endl;
						exit(1);
					}
					E[l] = Eckardt_points[m];
				}
				if (f_v) {
					cout << "surface_object_properties::compute_axes found axis "
							"t=" << t << " i=" << i << " : ";
					cout << "E[] = ";
					Lint_vec_print(cout, E, 3);
					cout << endl;
				}
				line_rk = SO->Surf->P->Subspaces->line_through_two_points(E[0], E[1]);
				line_rk1 = SO->Surf->P->Subspaces->line_through_two_points(E[0], E[2]);
				line_rk2 = SO->Surf->P->Subspaces->line_through_two_points(E[1], E[2]);
				if (line_rk1 != line_rk) {
					cout << "surface_object_properties::compute_axes line_rk1 != line_rk" << endl;
					exit(1);
				}
				if (line_rk2 != line_rk) {
					cout << "surface_object_properties::compute_axes line_rk2 != line_rk" << endl;
					exit(1);
				}
				Axes_line_rank[nb_axes] = line_rk;
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

	if (f_v) {
		cout << "surface_object_properties::compute_gradient" << endl;
	}


	if (f_v) {
		cout << "surface_object_properties::compute_gradient "
				"SO->Surf->Poly2_4->get_nb_monomials() = "
				<< SO->Surf->PolynomialDomains->Poly2_4->get_nb_monomials() << endl;
	}

	SO->Surf->PolynomialDomains->compute_gradient(
			SO->eqn, gradient, verbose_level);

	if (f_v) {
		cout << "surface_object_properties::compute_gradient done" << endl;
	}
}



void surface_object_properties::compute_singular_points_and_tangent_planes(int verbose_level)
// a singular point is a point where all partials vanish
// We compute the set of singular points into Pts[nb_pts]
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);

	if (f_v) {
		cout << "surface_object_properties::compute_singular_points_and_tangent_planes" << endl;
	}
	int h, i;
	long int rk;
	int nb_eqns = 4;
	int v[4];
	int w[4];


	if (f_v) {
		cout << "surface_object_properties::compute_singular_points_and_tangent_planes "
				"before compute_gradient" << endl;
	}
	compute_gradient(verbose_level - 2);
	if (f_v) {
		cout << "surface_object_properties::compute_singular_points_and_tangent_planes "
				"after compute_gradient" << endl;
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
			Int_vec_print(cout, v, 4);
			cout << endl;
		}
		for (i = 0; i < nb_eqns; i++) {
			if (f_vv) {
				cout << "surface_object_properties::compute_singular_points_and_tangent_planes "
						"gradient i=" << i << " / " << nb_eqns << endl;
			}
			if (false) {
				cout << "surface_object_properties::compute_singular_points_and_tangent_planes "
						"gradient " << i << " = ";
				Int_vec_print(cout,
						gradient + i * SO->Surf->PolynomialDomains->Poly2_4->get_nb_monomials(),
						SO->Surf->PolynomialDomains->Poly2_4->get_nb_monomials());
				cout << endl;
			}
			w[i] = SO->Surf->PolynomialDomains->Poly2_4->evaluate_at_a_point(
					gradient + i * SO->Surf->PolynomialDomains->Poly2_4->get_nb_monomials(), v);
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

			plane_rk = SO->Surf->P->Solid->plane_rank_using_dual_coordinates_in_three_space(
					w /* eqn4 */,
					0 /* verbose_level*/);
			tangent_plane_rank_global[h] = plane_rk;
			tangent_plane_rank_dual[nb_non_singular_pts++] =
					SO->Surf->P->Solid->dual_rank_of_plane_in_three_space(
							plane_rk, 0 /* verbose_level*/);
		}
	}

	data_structures::sorting Sorting;
	int nb_tangent_planes;

	nb_tangent_planes = nb_non_singular_pts;

	Sorting.lint_vec_sort_and_remove_duplicates(
			tangent_plane_rank_dual, nb_tangent_planes);

#if 0
	string fname_tangents;
	file_io Fio;

	fname_tangents.assign("tangents.txt");

	Fio.write_set_to_file_lint(fname_tangents,
			tangent_plane_rank_dual, nb_tangent_planes, verbose_level);

	if (f_v) {
		cout << "Written file " << fname_tangents << " of size " << Fio.file_size(fname_tangents) << endl;
	}
#endif


	int *Kernel;
	int *w1;
	int *w2;
	int r, ns;

	Kernel = NEW_int(SO->Surf->PolynomialDomains->Poly3_4->get_nb_monomials()
			* SO->Surf->PolynomialDomains->Poly3_4->get_nb_monomials());
	w1 = NEW_int(SO->Surf->PolynomialDomains->Poly3_4->get_nb_monomials());
	w2 = NEW_int(SO->Surf->PolynomialDomains->Poly3_4->get_nb_monomials());



	SO->Surf->PolynomialDomains->Poly3_4->vanishing_ideal(
			tangent_plane_rank_dual, nb_non_singular_pts,
			r, Kernel, 0 /*verbose_level */);

	ns = SO->Surf->PolynomialDomains->Poly3_4->get_nb_monomials() - r; // dimension of null space
	if (f_v) {
		cout << "The system has rank " << r << endl;
		cout << "The ideal has dimension " << ns << endl;
#if 0
		cout << "and is generated by:" << endl;
		int_matrix_print(Kernel, ns, SO->Surf->Poly3_4->get_nb_monomials());
		cout << "corresponding to the following basis "
				"of polynomials:" << endl;
		for (h = 0; h < ns; h++) {
			SO->Surf->Poly3_4->print_equation(cout, Kernel + h * SO->Surf->Poly3_4->get_nb_monomials());
			cout << endl;
		}
#endif
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

	Line_neighbors = NEW_OBJECT(data_structures::set_of_sets);
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

	nb_planes = SO->Surf->P->Subspaces->Nb_subspaces[2];
	plane_type_by_points = NEW_int(nb_planes);
	plane_type_by_lines = NEW_int(nb_planes);

	if (nb_planes < MAX_NUMBER_OF_PLANES_FOR_PLANE_TYPE) {
		SO->Surf->P->Subspaces->plane_intersection_type_basic(SO->Pts, SO->nb_pts,
			plane_type_by_points, 0 /* verbose_level */);


		C_plane_type_by_points = NEW_OBJECT(data_structures::tally);

		C_plane_type_by_points->init(plane_type_by_points, nb_planes, false, 0);
		if (f_v) {
			cout << "plane types by points: ";
			C_plane_type_by_points->print_bare(true);
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





#if 0
void surface_object_properties::print_everything(std::ostream &ost, int verbose_level)
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
#endif


void surface_object_properties::report_properties(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_properties::report_properties" << endl;
	}

	if (f_v) {
		cout << "surface_object_properties::report_properties_simple "
				"before print_equation" << endl;
	}
	print_equation(ost);

	if (f_v) {
		cout << "surface_object_properties::report_properties "
				"before print_summary" << endl;
	}
	print_summary(ost);


	if (f_v) {
		cout << "surface_object_properties::report_properties "
				"before print_lines" << endl;
	}
	print_lines(ost);

	if (f_v) {
		cout << "surface_object_properties::report_properties "
				"before print_points" << endl;
	}
	print_points(ost);


	if (f_v) {
		cout << "surface_object_properties::report_properties "
				"SmoothProperties->print_tritangent_planes" << endl;
	}
	SmoothProperties->print_tritangent_planes(ost);


	if (f_v) {
		cout << "surface_object_properties::report_properties "
				"before SmoothProperties->print_Steiner_and_Eckardt" << endl;
	}
	SmoothProperties->print_Steiner_and_Eckardt(ost);

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
		cout << "surface_object_properties::report_properties_simple "
				"before print_equation" << endl;
	}
	print_equation(ost);

	if (f_v) {
		cout << "surface_object_properties::report_properties_simple "
				"before print_summary" << endl;
	}
	print_summary(ost);




	ost << "\\subsubsection*{Singular Points}" << endl;
	cout << "surface_object_properties::print_points "
			"before print_singular_points" << endl;
	print_singular_points(ost);


	if (f_v) {
		cout << "surface_object_properties::report_properties_simple "
				"before print_lines" << endl;
	}
	print_lines(ost);


	ost << "\\subsubsection*{All Points of the Surface}" << endl;
	print_all_points_on_surface(ost);


	cout << "surface_object_properties::print_points "
			"before print_Eckardt_points" << endl;
	print_Eckardt_points(ost);



	ost << "\\subsubsection*{Eckardt Points}" << endl;
	cout << "surface_object_properties::print_points "
			"before print_Eckardt_points" << endl;
	print_Eckardt_points(ost);

	ost << "\\subsubsection*{Double Points}" << endl;
	cout << "surface_object_properties::print_points "
			"before print_double_points" << endl;
	print_double_points(ost);


	ost << "\\subsubsection*{Single Points}" << endl;
	cout << "surface_object_properties::print_points "
			"before print_single_points" << endl;
	print_single_points(ost);

	//ost << "\\subsubsection*{Points on lines}" << endl;
	//cout << "surface_object_properties::print_points before print_points_on_lines" << endl;
	//print_points_on_lines(ost);

	ost << "\\subsubsection*{Points on surface but on no line}" << endl;
	cout << "surface_object_properties::print_points "
			"before print_points_on_surface_but_not_on_a_line" << endl;
	print_points_on_surface_but_not_on_a_line(ost);


	if (f_v) {
		cout << "surface_object_properties::report_properties_simple "
				"print_Hesse_planes" << endl;
	}
	print_Hesse_planes(ost);

	if (f_v) {
		cout << "surface_object_properties::report_properties_simple "
				"print_axes" << endl;
	}
	print_axes(ost);

	if (SmoothProperties) {

		if (f_v) {
			cout << "surface_object_properties::report_properties_simple "
					"SmoothProperties->print_tritangent_planes" << endl;
		}
		SmoothProperties->print_tritangent_planes(ost);
	}
	else {
		if (f_v) {
			cout << "surface_object_properties::report_properties_simple "
					"SmoothProperties does not exist" << endl;
		}

	}

#if 0
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
		cout << "surface_object_properties::report_properties_simple "
				"after print_line_intersection_graph" << endl;
	}


	if (f_v) {
		cout << "surface_object_properties::report_properties_simple "
				"before print_all_points_on_surface" << endl;
	}
	print_all_points_on_surface(ost);


	if (f_v) {
		cout << "surface_object_properties::report_properties_simple "
				"before SO->Surf->Schlaefli->print_Steiner_and_Eckardt" << endl;
	}

	SO->Surf->Schlaefli->print_Steiner_and_Eckardt(ost);

	if (f_v) {
		cout << "surface_object_properties::report_properties_simple "
				"after SO->Surf->Schlaefli->print_Steiner_and_Eckardt" << endl;
	}

	if (f_v) {
		cout << "surface_object_properties::report_properties_simple "
				"done" << endl;
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
	if (SO->nb_lines < 128) {
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
				ost << SO->Surf->Schlaefli->Labels->Line_label_tex[i];
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
					ost << SO->Surf->Schlaefli->Labels->Line_label_tex[set[j]];
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
					ost << SO->Surf->Schlaefli->Labels->Line_label_tex[set[j]];
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
	else {
		ost << "Too many lines to print.\\\\" << endl;
	}

}


void surface_object_properties::print_adjacency_matrix(std::ostream &ost)
{
	if (SO->nb_lines < 128) {
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
				ost << " & " << SO->Surf->Schlaefli->Labels->Line_label_tex[j];
			}
			ost << "\\\\" << endl;
		}
		ost << "\\hline" << endl;
		for (i = 0; i < m; i++) {
			ost << i << " & ";
			if (SO->nb_lines == 27) {
				ost << SO->Surf->Schlaefli->Labels->Line_label_tex[i];
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
	else {
		ost << "Too many lines to print.\\\\" << endl;
	}
}

void surface_object_properties::print_adjacency_matrix_with_intersection_points(
		std::ostream &ost)
{

	if (SO->nb_lines < 128) {
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
				ost << " & " << SO->Surf->Schlaefli->Labels->Line_label_tex[j];
			}
			ost << "\\\\" << endl;
		}
		ost << "\\hline" << endl;
		for (i = 0; i < m; i++) {
			ost << i;
			if (SO->nb_lines == 27) {
				ost << " & " << SO->Surf->Schlaefli->Labels->Line_label_tex[i];
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
	else {
		ost << "Too many lines to print.\\\\" << endl;
	}
}

void surface_object_properties::print_neighbor_sets(std::ostream &ost)
{
	int i, j, h, p;
	data_structures::sorting Sorting;

	//ost << "\\clearpage" << endl;
	ost << "Neighbor sets in the line intersection graph:\\\\" << endl;
	//Line_neighbors->print_table_tex(ost);
	if (Line_neighbors->nb_sets < 1028) {
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
				ost << " & " << "\\ell_{" << j << "}";
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
				ost << " & " << "P_{" << p << "}";
#endif
			}
			ost << "\\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\end{array}" << endl;
			ost << "$$" << endl;
		}
	}
	else {
		ost << "Too many lines to print.\\\\" << endl;
	}
}



void surface_object_properties::print_plane_type_by_points(std::ostream &ost)
{
	ost << "\\subsection*{Plane types by points}" << endl;
		//*fp << "$$" << endl;
		//*fp << "\\Big(" << endl;
	C_plane_type_by_points->print_bare_tex(ost, true);
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
	l1_interfaces::latex_interface L;

	ost << "\\subsection*{The " << SO->nb_lines << " lines with points on them}" << endl;
	int i, j;
	int pt;

	if (SO->nb_lines < 128) {

		ost << "As lines in ${\\rm PG}(3,q)$:\\\\" << endl;
		Lint_vec_print(ost, SO->Lines, SO->nb_lines);
		ost << "\\\\" << endl;

		ost << "As elements on the Klein quadric (in the same order):\\\\" << endl;
		for (i = 0; i < SO->nb_lines; i++) {

			long int line_rk, a;

			line_rk = SO->Lines[i];

			a = SO->Surf->Klein->line_to_point_on_quadric(line_rk, 0 /* verbose_level*/);
			ost << a;
			if (i < SO->nb_lines - 1) {
				ost << ", ";
			}

		}
		ost << "\\\\" << endl;



		for (i = 0; i < SO->nb_lines; i++) {
			//ost << "Line " << i << " is " << v[i] << ":\\\\" << endl;
			SO->Surf->Gr->unrank_lint(SO->Lines[i], 0 /*verbose_level*/);
			ost << "$$" << endl;
			ost << "\\ell_{" << i << "} ";
			if (SO->nb_lines == 27) {
				ost << " = " << SO->Surf->Schlaefli->Labels->Line_label_tex[i];
			}
			ost << " = \\left[" << endl;
			//print_integer_matrix_width(cout, Gr->M,
			// k, n, n, F->log10_of_q + 1);
			L.print_integer_matrix_tex(ost, SO->Surf->Gr->M, 2, 4);
			ost << "\\right]_{" << SO->Lines[i] << "}" << endl;
			ost << "$$" << endl;
			ost << "which contains the point set " << endl;
			ost << "$$" << endl;
			ost << "\\{ ";
			L.lint_set_print_tex(ost, pts_on_lines->Sets[i],
					pts_on_lines->Set_size[i]);
			ost << "\\}." << endl;
			ost << "$$" << endl;

			{
				std::vector<long int> plane_ranks;

				SO->Surf->P->Subspaces->planes_through_a_line(
						SO->Lines[i], plane_ranks,
						0 /*verbose_level*/);

				// print the tangent planes associated with the points on the line:
				ost << "The tangent planes associated "
						"with the points on this line are:\\\\" << endl;
				for (j = 0; j < pts_on_lines->Set_size[i]; j++) {

					int w[4];

					pt = pts_on_lines->Sets[i][j];
					ost << j << " : " << pt << " : ";
					SO->Surf->unrank_point(w, SO->Pts[pt]);
					Int_vec_print(ost, w, 4);
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
	else {
		ost << "Too many to print.\\\\" << endl;
	}
}

void surface_object_properties::print_equation(std::ostream &ost)
{
	ost << "\\subsection*{The equation}" << endl;
	ost << "The equation of the surface ";
	ost << " is :" << endl;

#if 0
	ost << "$$" << endl;
	SO->Surf->print_equation_tex(ost, SO->eqn);
	ost << endl << "=0\n$$" << endl;
#else
	SO->Surf->print_equation_with_line_breaks_tex(ost, SO->eqn);
#endif
	Int_vec_print(ost, SO->eqn, 20);
	ost << "\\\\" << endl;

	long int rk;

	SO->F->Projective_space_basic->PG_element_rank_modified_lint(
			SO->eqn, 1, 20, rk);
	ost << "The point rank of the equation over GF$(" << SO->F->q << ")$ is " << rk << "\\\\" << endl;

	ost << "\\begin{verbatim}" << endl;
	SO->Surf->PolynomialDomains->Poly3_4->print_equation_relaxed(ost, SO->eqn);
	ost << endl;
	ost << "\\end{verbatim}" << endl;


	//ost << "Number of points on the surface " << SO->nb_pts << "\\\\" << endl;


}

void surface_object_properties::print_summary(std::ostream &ost)
{
	ost << "\\subsection*{Summary}" << endl;


	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|l|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of lines} & " << SO->nb_lines << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of points} & " << SO->nb_pts << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of singular points} & " << nb_singular_pts << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of Eckardt points} & " << nb_Eckardt_points << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of double points} & " << nb_Double_points << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of single points} & " << nb_Single_points << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of points off lines} & " << nb_pts_not_on_lines << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of Hesse planes} & " << nb_Hesse_planes << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of axes} & " << nb_axes << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Type of points on lines} & ";
	Type_pts_on_lines->print_bare_tex(ost, true);
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Type of lines on points} & ";
	Type_lines_on_point->print_bare_tex(ost, true);
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
#if 0
	ost << "Points on lines:" << endl;
	ost << "$$" << endl;
	Type_pts_on_lines->print_bare_tex(ost, true);
	ost << "$$" << endl;
	ost << "Lines on points:" << endl;
	ost << "$$" << endl;
	Type_lines_on_point->print_bare_tex(ost, true);
	ost << "$$" << endl;
#endif
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
		SO->Surf->F->Projective_space_basic->PG_element_normalize(
				v, 1, 4);
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

	cout << "surface_object_properties::print_points "
			"before print_points_on_surface" << endl;
	//print_points_on_surface(ost);
	print_all_points_on_surface(ost);

	ost << "\\subsubsection*{Singular Points}" << endl;
	cout << "surface_object_properties::print_points "
			"before print_singular_points" << endl;
	print_singular_points(ost);

	ost << "\\subsubsection*{Eckardt Points}" << endl;
	cout << "surface_object_properties::print_points "
			"before print_Eckardt_points" << endl;
	print_Eckardt_points(ost);

	ost << "\\subsubsection*{Double Points}" << endl;
	cout << "surface_object_properties::print_points "
			"before print_double_points" << endl;
	print_double_points(ost);

	ost << "\\subsubsection*{Single Points}" << endl;
	cout << "surface_object_properties::print_points "
			"before print_single_points" << endl;
	print_single_points(ost);

	ost << "\\subsubsection*{Points on lines}" << endl;
	cout << "surface_object_properties::print_points "
			"before print_points_on_lines" << endl;
	print_points_on_lines(ost);

	ost << "\\subsubsection*{Points on surface but on no line}" << endl;
	cout << "surface_object_properties::print_points "
			"before print_points_on_surface_but_not_on_a_line" << endl;
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
			cout << "surface_object_properties::print_Eckardt_points "
					"Eckardt point is not on three lines" << endl;
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
			ost << SO->Surf->Schlaefli->Labels->Line_label_tex[a] << " \\cap ";
			ost << SO->Surf->Schlaefli->Labels->Line_label_tex[b] << " \\cap ";
			ost << SO->Surf->Schlaefli->Labels->Line_label_tex[c];
			ost << " = ";
		}
		//ost << "P_{" << p << "} = ";
		ost << "P_{" << Eckardt_points[i] << "}=";
		ost << "\\bP(";
		//int_vec_print_fully(ost, v, 4);
		for (j = 0; j < 4; j++) {
			SO->F->Io->print_element(ost, v[j]);
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

		//ost << "\\; T= " << tangent_plane_rank_global[p];
		ost << "$\\\\" << endl;
#if 0
		if (tangent_plane_rank_global[p] == -1) {
			cout << "Eckardt point is singular. " << endl;
			//exit(1);
		}
#endif

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

		T_planes.init(Eckardt_points_plane_type, SO->Surf->P->Nb_subspaces[2], false, 0);


		T_planes.print_file_tex_we_are_in_math_mode(ost, true);
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
	Lint_vec_print(ost, Hesse_planes, nb_Hesse_planes);
	ost << "\\\\" << endl;

	SO->Surf->Gr3->print_set_tex(ost, Hesse_planes, nb_Hesse_planes, 0 /* verbose_level */);


	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;


	cout << "Hesse plane : rank : Incident Eckardt points\\\\" << endl;

	ost << "\\noindent" << endl;
	for (j = 0; j < nb_Hesse_planes; j++) {


		int H[9], cnt, h;

		cnt = 0;
		for (i = 0; i < nb_Eckardt_points; i++) {
			if (Eckardt_point_Hesse_plane_incidence[i * nb_Hesse_planes + j]) {
				if (cnt == 9) {
					cout << "too many points on the Hesse plane" << endl;
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


	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;


	cout << "Hesse plane : rank : Incident Eckardt points\\\\" << endl;
	ost << "\\noindent" << endl;
	for (j = 0; j < nb_Hesse_planes; j++) {


		int H[9], cnt, h;

		cnt = 0;
		for (i = 0; i < nb_Eckardt_points; i++) {
			if (Eckardt_point_Hesse_plane_incidence[i * nb_Hesse_planes + j]) {
				if (cnt == 9) {
					cout << "too many points on the Hesse plane" << endl;
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
			ost << Eckardt_points_schlaefli_labels[i];
			if (h < 9 - 1) {
				ost << ", ";
			}
		}
		ost << "\\\\" << endl;
	}

}

void surface_object_properties::print_axes(std::ostream &ost)
{
	l1_interfaces::latex_interface L;
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
	l1_interfaces::latex_interface L;
	int i, j, p;
	int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << nb_singular_pts
			<< " singular points:\\\\" << endl;





	//ost << "The Eckardt points are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	//ost << "\\begin{align*}" << endl;
	ost << "\\noindent" << endl;
	for (i = 0; i < nb_singular_pts; i++) {
		p = singular_pts[i];
		SO->Surf->unrank_point(v, p);
		ost << i << " :  $P_{" << p << "}=\\bP(";
		//int_vec_print_fully(ost, v, 4);
		for (j = 0; j < 4; j++) {
			SO->F->Io->print_element(ost, v[j]);
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

		ost << "$\\\\" << endl;
		}
	//ost << "\\end{align*}" << endl;
	ost << "\\end{multicols}" << endl;
}



void surface_object_properties::print_double_points(std::ostream &ost)
{
	l1_interfaces::latex_interface L;
	int i, p, a, b;
	int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << nb_Double_points
			<< " Double points:\\\\" << endl;
	if (nb_Double_points < 1000) {

#if 0
		ost << "$$" << endl;
		L.lint_vec_print_as_matrix(ost,
				Double_points, nb_Double_points, 10,
				true /* f_tex */);
		ost << "$$" << endl;

		ost << "$$" << endl;
		L.int_vec_print_as_matrix(ost,
				Double_points_index, nb_Double_points, 10,
				true /* f_tex */);
		ost << "$$" << endl;
#endif

#if 0
		//ost << "\\clearpage" << endl;
		ost << "The Double points on the surface are:\\\\" << endl;
		ost << "\\begin{multicols}{2}" << endl;
		ost << "\\noindent" << endl;
		for (i = 0; i < nb_Double_points; i++) {
			SO->Surf->unrank_point(v, Double_points[i]);
			ost << i << " : $";
			//ost << P_{" << Double_points_index[i] << "}=";
			ost << "P_{" << Double_points[i] << "}=";
			int_vec_print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
#endif

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
		ost << "\\begin{multicols}{2}" << endl;
		ost << "\\noindent" << endl;
		for (a = 0; a < SO->nb_lines; a++) {
			for (b = a + 1; b < SO->nb_lines; b++) {
				p = pt_idx[a * SO->nb_lines + b];
				if (p == -1) {
					continue;
				}
				SO->Surf->unrank_point(v, SO->Pts[p]);
				//ost << "P_{" << p << "} = ";
				ost << "$P_{" << SO->Pts[p] << "}";
				ost << " = ";
				Int_vec_print_fully(ost, v, 4);


				if (SO->nb_lines == 27) {
					ost << " = ";
					ost << "\\ell_{" << a << "} \\cap ";
					ost << "\\ell_{" << b << "} ";
					ost << " = ";
					ost << SO->Surf->Schlaefli->Labels->Line_label_tex[a] << " \\cap ";
					ost << SO->Surf->Schlaefli->Labels->Line_label_tex[b];
				}
				else {
					ost << " = ";
					ost << "\\ell_{" << a << "} \\cap ";
					ost << "\\ell_{" << b << "} ";
				}
				ost << "$\\\\" << endl;
			}
		}
		ost << "\\end{multicols}" << endl;

		FREE_int(pt_idx);
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}
}

void surface_object_properties::print_single_points(std::ostream &ost)
{
	l1_interfaces::latex_interface L;
	int i, p, a;
	int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << nb_Single_points
			<< " single points:\\\\" << endl;
	if (nb_Single_points < 1000) {

#if 0
		ost << "$$" << endl;
		L.lint_vec_print_as_matrix(ost,
				Single_points, nb_Single_points, 10,
				true /* f_tex */);
		ost << "$$" << endl;
		ost << "$$" << endl;
		L.int_vec_print_as_matrix(ost,
				Single_points_index, nb_Single_points, 10,
				true /* f_tex */);
		ost << "$$" << endl;
#endif

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
			Int_vec_print_fully(ost, v, 4);
			ost << "$";
			if (SO->nb_lines == 27) {
				ost << " lies on line $" << SO->Surf->Schlaefli->Labels->Line_label_tex[a] << "$";
			}
			else {
				ost << " lies on line $\\ell_{" << a << "}$";
			}
			ost << "\\\\" << endl;
		}
		ost << "\\end{multicols}" << endl;
		ost << "The single points on the surface are:\\\\" << endl;
		Lint_vec_print_fully(ost, Single_points, nb_Single_points);
		ost << "\\\\" << endl;
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
		L.lint_vec_print_as_matrix(ost, SO->Pts, SO->nb_pts, 10, true /* f_tex */);
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

	if (SO->nb_pts < 2000) {
		//ost << "$$" << endl;
		//L.lint_vec_print_as_matrix(ost, SO->Pts, SO->nb_pts, 10, true /* f_tex */);
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
			Int_vec_print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
		Lint_vec_print_fully(ost, SO->Pts, SO->nb_pts);
		ost << "\\\\" << endl;
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}
}

void surface_object_properties::print_points_on_lines(std::ostream &ost)
{
	l1_interfaces::latex_interface L;
	int i;

	//ost << "\\clearpage" << endl;
	//pts_on_lines->print_table_tex(ost);
	ost << "\\noindent" << endl;
	if (pts_on_lines->nb_sets < 1000) {
		for (i = 0; i < pts_on_lines->nb_sets; i++) {
			ost << "Line " << i;

			if (SO->nb_lines == 27) {
				ost << " = "
					"$" << SO->Surf->Schlaefli->Labels->Line_label_tex[i]
				<< "$ " << endl;
			}

			ost << "has " << pts_on_lines->Set_size[i]
				<< " points: $\\{ P_{i} \\mid i \\in ";
			L.lint_set_print_tex(ost, pts_on_lines->Sets[i],
					pts_on_lines->Set_size[i]);
			ost << "\\}$\\\\" << endl;
		}
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}

	//ost << "\\clearpage" << endl;
}

void surface_object_properties::print_points_on_surface_but_not_on_a_line(
			std::ostream &ost)
{
	l1_interfaces::latex_interface L;
	int i;
	int v[4];

	//ost << "\\clearpage" << endl;
	ost << "The surface has " << nb_pts_not_on_lines
			<< " points not on any line:\\\\" << endl;
	if (nb_pts_not_on_lines < 1000) {
#if 0
		ost << "$$" << endl;
		L.lint_vec_print_as_matrix(ost,
				Pts_not_on_lines, nb_pts_not_on_lines, 10,
				true /* f_tex */);
		//print_integer_matrix_with_standard_labels(ost, Pts3,
		//(nb_pts_not_on_lines + 9) / 10, 10, true /* f_tex */);
		ost << "$$" << endl;
#endif
		//ost << "%%\\clearpage" << endl;
		ost << "The points on the surface but not "
				"on lines are:\\\\" << endl;
		ost << "\\begin{multicols}{2}" << endl;
		ost << "\\noindent" << endl;
		for (i = 0; i < nb_pts_not_on_lines; i++) {
			SO->Surf->unrank_point(v, Pts_not_on_lines[i]);
			ost << i << " : $P_{" << Pts_not_on_lines[i] << "}=";
			Int_vec_print_fully(ost, v, 4);
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
	l1_interfaces::latex_interface L;

	ost << "The half double sixes are:\\\\" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
			SO->Surf->Schlaefli->Half_double_sixes, 36, 6, true /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
			SO->Surf->Schlaefli->Half_double_sixes + 36 * 6,
		36, 6, 36, 0, true /* f_tex */);
	ost << "$$" << endl;
}

void surface_object_properties::print_trihedral_pairs(std::ostream &ost)
{

	SO->Surf->Schlaefli->print_trihedral_pairs(ost);

	SO->Surf->Schlaefli->latex_triads(ost);
}

void surface_object_properties::print_trihedral_pairs_numerically(std::ostream &ost)
{
	l1_interfaces::latex_interface L;

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Trihedral pairs}" << endl;
	ost << "The planes in the trihedral pairs in Eckardt "
			"point labeling are:\\\\" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
			SO->Surf->Schlaefli->Trihedral_to_Eckardt, 40, 6, true /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
			SO->Surf->Schlaefli->Trihedral_to_Eckardt + 40 * 6, 40, 6, 40, 0, true /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
			SO->Surf->Schlaefli->Trihedral_to_Eckardt + 80 * 6, 40, 6, 80, 0, true /* f_tex */);
	ost << "$$" << endl;
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
		Int_vec_print(ost, v, 4);
		ost << " & ";

		for (j = 0; j < lines_on_point->Set_size[i]; j++) {
			a = lines_on_point->Sets[i][j];
			ost << SO->Surf->Schlaefli->Labels->Line_label_tex[a];
			if (j < lines_on_point->Set_size[i] - 1) {
				ost << ", ";
			}
		}
		ost << " & ";


		if (Clebsch_map[i] >= 0) {
			Int_vec_print(ost, Clebsch_coeff + i * 4, 4);
		}
		else {
			ost << "\\mbox{undef}";
		}
		ost << " & ";
		if (Clebsch_map[i] >= 0) {
			SO->Surf->P2->unrank_point(w, Clebsch_map[i]);
			Int_vec_print(ost, w, 3);
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




void surface_object_properties::compute_reduced_set_of_points_not_on_lines_wrt_P(
		int P_idx,
		int *&f_deleted, int verbose_level)
// P_idx = index into SO->Pts[]
{
	int f_v = (verbose_level >= 1);
	int f_vv = false;
	int i, idx;
	long int P, R, Q;
	int Basis_of_PR[8];

	if (f_v) {
		cout << "surface_object_properties::compute_reduced_set_of_points_not_on_lines_wrt_P" << endl;
	}
	f_deleted = NEW_int(nb_pts_not_on_lines);
	Int_vec_zero(f_deleted, nb_pts_not_on_lines);

	P = SO->Pts[P_idx];
	for (i = 0; i < nb_pts_not_on_lines; i++) {
		R = Pts_not_on_lines[i];
		if (R == P) {
			continue;
		}
		SO->Surf->unrank_point(Basis_of_PR, P);
		SO->Surf->unrank_point(Basis_of_PR + 4, R);

		int v[2];
		int w[4];
		int j;

		for (j = 0; j < SO->F->q + 1; j++) {
			SO->F->Projective_space_basic->PG_element_unrank_modified(
					v, 1, 2, j);
			if (f_vv) {
				cout << "surface_object_properties::compute_reduced_set_of_points_not_on_lines_wrt_P v=" << endl;
				Int_vec_print(cout, v, 2);
				cout << endl;
			}

			SO->F->Linear_algebra->mult_matrix_matrix(v, Basis_of_PR, w, 1, 2, 4,
					0 /* verbose_level */);
			if (f_vv) {
				cout << "surface_object_properties::compute_reduced_set_of_points_not_on_lines_wrt_P w=" << endl;
				Int_vec_print(cout, w, 4);
				cout << endl;
			}

			Q = SO->Surf->rank_point(w);

			if (SO->find_point(Q, idx)) {
				f_deleted[i] = true;
			}

		}


	}

	if (f_v) {
		cout << "surface_object_properties::compute_reduced_set_of_points_not_on_lines_wrt_P done" << endl;
	}
}


int surface_object_properties::test_full_del_pezzo(
		int P_idx, int *f_deleted, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	long int P, R, rk_tangent_plane;
	int Basis[4 * 4];

	if (f_v) {
		cout << "surface_object_properties::test_full_del_pezzo" << endl;
	}

	P = SO->Pts[P_idx];
	rk_tangent_plane = tangent_plane_rank_global[P_idx];

	SO->Surf->P->Solid->dual_rank_of_plane_in_three_space(
			rk_tangent_plane, 0 /* verbose_level*/);


	for (i = 0; i < nb_pts_not_on_lines; i++) {
		R = Pts_not_on_lines[i];
		if (R == P) {
			continue;
		}
		if (f_deleted[i]) {
			continue;
		}
		int rk;

		if (f_v) {
			cout << "projective_space::dual_rank_of_plane_in_three_space" << endl;
		}
		SO->Surf->P->unrank_plane(Basis, rk_tangent_plane);
		SO->Surf->unrank_point(Basis + 12, R);
		rk = SO->F->Linear_algebra->Gauss_easy(Basis, 4, 4);
		if (rk != 3) {
			return false;
		}
	}

	if (f_v) {
		cout << "surface_object_properties::test_full_del_pezzo done" << endl;
	}
	return true;
}


void surface_object_properties::create_summary_file(
		std::string &fname,
		std::string &surface_label,
		std::string &col_postfix,
		int verbose_level)
{
	string col_lab_surface_label;
	string col_lab_nb_lines;
	string col_lab_nb_points;
	string col_lab_nb_singular_points;
	string col_lab_nb_Eckardt_points;
	string col_lab_nb_double_points;
	string col_lab_nb_Single_points;
	string col_lab_nb_pts_not_on_lines;
	string col_lab_nb_Hesse_planes;
	string col_lab_nb_axes;


	col_lab_surface_label.assign("Surface");


	col_lab_nb_lines = "#L" + col_postfix;

	col_lab_nb_points = "#P" + col_postfix;

	col_lab_nb_singular_points = "#S" + col_postfix;

	col_lab_nb_Eckardt_points = "#E" + col_postfix;

	col_lab_nb_double_points = "#D" + col_postfix;

	col_lab_nb_Single_points = "#U" + col_postfix;

	col_lab_nb_pts_not_on_lines = "#OFF" + col_postfix;

	col_lab_nb_Hesse_planes = "#H" + col_postfix;

	col_lab_nb_axes = "#AX" + col_postfix;

#if 0
	SO->nb_lines;

	SO->nb_pts;

	nb_singular_pts;

	nb_Eckardt_points;

	nb_Double_points;

	nb_Single_points;

	nb_pts_not_on_lines;

	nb_Hesse_planes;

	nb_axes;
#endif


	orbiter_kernel_system::file_io Fio;

	{
		ofstream f(fname);

		f << col_lab_surface_label << ",";
		f << col_lab_nb_lines << ",";
		f << col_lab_nb_points << ",";
		f << col_lab_nb_singular_points << ",";
		f << col_lab_nb_Eckardt_points << ",";
		f << col_lab_nb_double_points << ",";
		f << col_lab_nb_Single_points << ",";
		f << col_lab_nb_pts_not_on_lines << ",";
		f << col_lab_nb_Hesse_planes << ",";
		f << col_lab_nb_axes << ",";
		f << endl;

		f << surface_label << ",";
		f << SO->nb_lines << ",";
		f << SO->nb_pts << ",";
		f << nb_singular_pts << ",";
		f << nb_Eckardt_points << ",";
		f << nb_Double_points << ",";
		f << nb_Single_points << ",";
		f << nb_pts_not_on_lines << ",";
		f << nb_Hesse_planes << ",";
		f << nb_axes << ",";
		f << endl;

		f << "END" << endl;
	}
	cout << "Written file " << fname
			<< " of size " << Fio.file_size(fname) << endl;


}


}}}

