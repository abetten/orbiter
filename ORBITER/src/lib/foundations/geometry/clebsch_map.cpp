/*
 * clebsch_map.cpp
 *
 *  Created on: Jul 2, 2019
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


clebsch_map::clebsch_map()
{
	SO = NULL;
	Surf = NULL;
	F = NULL;

	hds = 0;
	ds = 0;
	ds_row = 0;

	line1 = 0;
	line2 = 0;
	transversal = 0;

	tritangent_plane_idx = 0;


	line_idx[0] = -1;
	line_idx[1] = -1;
	plane_rk_global = 0;

	Clebsch_map = NULL;
	Clebsch_coeff = NULL;
}

clebsch_map::~clebsch_map()
{
	freeself();
}


void clebsch_map::freeself()
{
	int f_v = FALSE;

	if (f_v) {
		cout << "clebsch_map::freeself" << endl;
		}
}

void clebsch_map::init_half_double_six(surface_object *SO,
		int hds, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "clebsch_map::init_half_double_six" << endl;
		}


	clebsch_map::SO = SO;
	Surf = SO->Surf;
	F = Surf->F;
	clebsch_map::hds = hds;
	ds = hds / 2;
	ds_row = hds % 2;
	if (f_v) {
		cout << "clebsch_map::init_half_double_six hds = " << hds
				<< " double six = " << ds << " row = " << ds_row << endl;
		}

	if (f_v) {
		cout << "clebsch_map::init_half_double_six "
				"before Surf->prepare_clebsch_map" << endl;
		}
	Surf->prepare_clebsch_map(ds, ds_row, line1, line2,
			transversal, verbose_level);
	if (f_v) {
		cout << "clebsch_map::init_half_double_six "
				"after Surf->prepare_clebsch_map" << endl;
		}

	if (f_v) {
		cout << "clebsch_map::init_half_double_six "
			"line1=" << line1
			<< " = " << Surf->Line_label_tex[line1]
			<< " line2=" << line2
			<< " = " << Surf->Line_label_tex[line2]
			<< " transversal=" << transversal
			<< " = " << Surf->Line_label_tex[transversal]
			<< endl;
		}

	line_idx[0] = line1;
	line_idx[1] = line2;
	//plane_rk = New_clebsch->choose_unitangent_plane(
	//line1, line2, transversal, 0 /* verbose_level */);
	tritangent_plane_idx = SO->choose_tritangent_plane(line1, line2,
			transversal, 0 /* verbose_level */);

	//plane_rk_global = New_clebsch->Unitangent_planes[plane_rk];
	plane_rk_global = SO->Tritangent_planes[
			SO->Eckardt_to_Tritangent_plane[tritangent_plane_idx]];


	int u, a, h;
	int v[4];
	int coefficients[3];


	Surf->P->Grass_planes->unrank_int_here(
			Plane, plane_rk_global, 0);
	F->Gauss_simple(Plane, 3, 4, base_cols,
			0 /* verbose_level */);

	if (f_v) {
		int_matrix_print(Plane, 3, 4);
		cout << "surface_with_action::arc_lifting_and_classify "
				"base_cols: ";
		int_vec_print(cout, base_cols, 3);
		cout << endl;
		}


	if (f_v) {
		cout << "surface_with_action::arc_lifting_and_classify "
				"Lines with points on them:" << endl;
		SO->print_lines_with_points_on_them(cout);
		cout << "The half double six is no " << hds
				<< "$ = " << Surf->Half_double_six_label_tex[hds]
				<< "$ : $";
		int_vec_print(cout, Surf->Half_double_sixes + hds * 6, 6);
		cout << " = \\{" << endl;
		for (h = 0; h < 6; h++) {
			cout << Surf->Line_label_tex[
					Surf->Half_double_sixes[hds * 6 + h]];
			if (h < 6 - 1) {
				cout << ", ";
				}
			}
		cout << "\\}$\\\\" << endl;
		}

	for (u = 0; u < 6; u++) {

		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify u="
					<< u << " / 6" << endl;
			}
		a = SO->Lines[Surf->Half_double_sixes[hds * 6 + u]];
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify "
					"intersecting line " << a << " and plane "
					<< plane_rk_global << endl;
			}
		intersection_points[u] =
				Surf->P->point_of_intersection_of_a_line_and_a_plane_in_three_space(
						a, plane_rk_global,
						0 /* verbose_level */);
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify "
					"intersection point " << intersection_points[u] << endl;
			}
		Surf->P->unrank_point(v, intersection_points[u]);
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify "
					"which is ";
			int_vec_print(cout, v, 4);
			cout << endl;
			}
		F->reduce_mod_subspace_and_get_coefficient_vector(
			3, 4, Plane, base_cols,
			v, coefficients,
			0 /* verbose_level */);
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify "
					"local coefficients ";
			int_vec_print(cout, coefficients, 3);
			cout << endl;
			}
		intersection_points_local[u] = Surf->P2->rank_point(coefficients);
		}


	init(SO, line_idx, plane_rk_global, verbose_level);


	if (f_v) {
		cout << "clebsch_map::init_half_double_six done" << endl;
		}
}

void clebsch_map::init(surface_object *SO,
		int *line_idx, int plane_rk_global, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "clebsch_map::init" << endl;
		}

	clebsch_map::SO = SO;
	Surf = SO->Surf;
	int_vec_copy(line_idx, clebsch_map::line_idx, 2);
	clebsch_map::plane_rk_global = plane_rk_global;


	Clebsch_map = NEW_int(SO->nb_pts);
	Clebsch_coeff = NEW_int(SO->nb_pts * 4);

	if (!Surf->clebsch_map(
			SO->Lines,
			SO->Pts,
			SO->nb_pts,
			line_idx,
			plane_rk_global,
			Clebsch_map,
			Clebsch_coeff,
			verbose_level)) {
		cout << "clebsch_map::init The plane contains one of the lines, "
				"this should not happen" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "clebsch_map::init done" << endl;
		}
}

}}

