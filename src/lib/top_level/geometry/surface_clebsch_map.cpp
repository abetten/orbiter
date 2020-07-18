/*
 * surface_clebsch_map.cpp
 *
 *  Created on: Jul 17, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


surface_clebsch_map::surface_clebsch_map()
{
	SOA = NULL;

	orbit_idx = 0;
	f = l = k = 0;
	line1 = line2 = transversal = 0;

	Clebsch_map = NULL;
	Clebsch_coeff = NULL;

	plane_rk = plane_rk_global = 0;
	//int line_idx[2];
	//long int Arc[6];
	//long int Blown_up_lines[6];

	ds = ds_row = 0;
	//int intersection_points[6];
	//int v[4];
	//int Plane[16];
	//int base_cols[4];
	//int coefficients[3];


}





surface_clebsch_map::~surface_clebsch_map()
{
	if (Clebsch_map) {
		FREE_lint(Clebsch_map);
	}
	if (Clebsch_coeff) {
		FREE_int(Clebsch_coeff);
	}
}

void surface_clebsch_map::report(std::ostream &ost, int verbose_level)
{
	latex_interface L;

	ost << "\\subsection*{Orbit on single sixes " << orbit_idx << " / "
		<< SOA->Orbits_on_single_sixes->nb_orbits << "}" << endl;

	int h;

	ost << "The half double six is no " << k << "$ = "
			<< SOA->Surf->Half_double_six_label_tex[k] << "$ : $";
	lint_vec_print(ost, SOA->Surf->Half_double_sixes + k * 6, 6);
	ost << " = \\{" << endl;
	for (h = 0; h < 6; h++) {
		ost << SOA->Surf->Line_label_tex[SOA->Surf->Half_double_sixes[k * 6 + h]];
		if (h < 6 - 1) {
			ost << ", ";
		}
	}
	ost << "\\}$\\\\" << endl;

	ost << "line1$=" << line1 << " = " << SOA->Surf->Line_label_tex[line1]
			<< "$ line2$=" << line2 << " = " << SOA->Surf->Line_label_tex[line2]
			<< "$ transversal$=" << transversal << " = "
			<< SOA->Surf->Line_label_tex[transversal] << "$\\\\" << endl;

	ost << "transversal = " << transversal << " = $"
			<< SOA->Surf->Line_label_tex[transversal] << "$\\\\" << endl;
	ost << "plane\\_rk = $\\pi_{" << plane_rk << "} = \\pi_{"
			<< SOA->Surf->Eckard_point_label_tex[plane_rk] << "} = "
			<< plane_rk_global << "$\\\\" << endl;


	ost << "The plane is:" << endl;
	SOA->Surf->P->Grass_planes->print_set_tex(ost, &plane_rk_global, 1);

	ost << "Clebsch map for lines $" << line1
			<< " = " << SOA->Surf->Line_label_tex[line1] << ", "
			<< line2 << " = "
			<< SOA->Surf->Line_label_tex[line2]
			<< "$\\\\" << endl;

	//SOA->SO->clebsch_map_latex(fp, Clebsch_map, Clebsch_coeff);

	ost << "Clebsch map for lines $" << line1
		<< " = " << SOA->Surf->Line_label_tex[line1] << ", "
		<< line2 << " = " << SOA->Surf->Line_label_tex[line2]
		<< "$ yields arc = $";
	L.lint_set_print_tex(ost, Arc, 6);
	ost << "$ : blown up lines = ";
	lint_vec_print(ost, Blown_up_lines, 6);
	ost << "\\\\" << endl;

	SOA->SO->clebsch_map_latex(ost, Clebsch_map, Clebsch_coeff);

}

void surface_clebsch_map::init(surface_object_with_action *SOA, int orbit_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int u;
	int h;
	int a;

	if (f_v) {
		cout << "surface_clebsch_map::init orbit "
			"on single sixes " << orbit_idx << " / "
			<< SOA->Orbits_on_single_sixes->nb_orbits << ":" << endl;
	}

	surface_clebsch_map::SOA = SOA;
	surface_clebsch_map::orbit_idx = orbit_idx;


	f = SOA->Orbits_on_single_sixes->orbit_first[orbit_idx];
	l = SOA->Orbits_on_single_sixes->orbit_len[orbit_idx];
	if (f_v) {
		cout << "orbit " << orbit_idx << " has f=" << f <<  " l=" << l << endl;
	}
	k = SOA->Orbits_on_single_sixes->orbit[f];

	if (f_v) {
		cout << "The half double six is no " << k << " : ";
		lint_vec_print(cout, SOA->Surf->Half_double_sixes + k * 6, 6);
		cout << endl;
	}





	ds = k / 2;
	ds_row = k % 2;
	if (f_v) {
		cout << "double six = " << ds << " row = " << ds_row << endl;
	}

	if (f_v) {
		cout << "surface_clebsch_map::init "
				"before Surf->prepare_clebsch_map" << endl;
	}
	SOA->Surf->prepare_clebsch_map(ds, ds_row, line1, line2, transversal, verbose_level);
	if (f_v) {
		cout << "surface_clebsch_map::init "
				"after Surf->prepare_clebsch_map" << endl;
	}

	if (f_v) {
		cout << "surface_clebsch_map::init "
			"line1=" << line1
			<< " = " << SOA->Surf->Line_label_tex[line1]
			<< " line2=" << line2
			<< " = " << SOA->Surf->Line_label_tex[line2]
			<< " transversal=" << transversal
			<< " = " << SOA->Surf->Line_label_tex[transversal]
			<< endl;
	}





	line_idx[0] = line1;
	line_idx[1] = line2;
	//plane_rk = New_clebsch->choose_unitangent_plane(
	//line1, line2, transversal, 0 /* verbose_level */);
	plane_rk = SOA->SO->choose_tritangent_plane(line1, line2,
			transversal, 0 /* verbose_level */);

	//plane_rk_global = New_clebsch->Unitangent_planes[plane_rk];
	plane_rk_global = SOA->SO->Tritangent_planes[
			SOA->SO->Eckardt_to_Tritangent_plane[plane_rk]];

	if (f_v) {
		cout << "surface_with_action::arc_lifting_and_classify "
				"transversal = " << transversal
			<< " = " << SOA->Surf->Line_label_tex[transversal]
			<< endl;
		cout << "plane_rk = " << plane_rk << " = "
				<< plane_rk_global << endl;
	}


	if (f_v) {
		cout << "surface_clebsch_map::init "
			"intersecting blow up lines with plane:" << endl;
	}


	SOA->Surf->P->Grass_planes->unrank_lint_here(
			Plane, plane_rk_global, 0);
	SOA->F->Gauss_simple(Plane, 3, 4, base_cols,
			0 /* verbose_level */);

	if (f_v) {
		int_matrix_print(Plane, 3, 4);
		cout << "surface_clebsch_map::init "
				"base_cols: ";
		int_vec_print(cout, base_cols, 3);
		cout << endl;
	}


	if (f_v) {
		cout << "surface_clebsch_map::init "
				"Lines with points on them:" << endl;
		SOA->SO->print_lines_with_points_on_them(cout);
		cout << "The half double six is no " << k
				<< "$ = " << SOA->Surf->Half_double_six_label_tex[k]
				<< "$ : $";
		lint_vec_print(cout, SOA->Surf->Half_double_sixes + k * 6, 6);
		cout << " = \\{" << endl;
		for (h = 0; h < 6; h++) {
			cout << SOA->Surf->Line_label_tex[SOA->Surf->Half_double_sixes[k * 6 + h]];
			if (h < 6 - 1) {
				cout << ", ";
			}
		}
		cout << "\\}$\\\\" << endl;
	}

	for (u = 0; u < 6; u++) {

		if (f_v) {
			cout << "surface_clebsch_map::init u="
					<< u << " / 6" << endl;
		}
		a = SOA->SO->Lines[SOA->Surf->Half_double_sixes[k * 6 + u]];
		if (f_v) {
			cout << "surface_clebsch_map::init "
					"intersecting line " << a << " and plane "
					<< plane_rk_global << endl;
		}
		intersection_points[u] =
				SOA->Surf->P->point_of_intersection_of_a_line_and_a_plane_in_three_space(
						a, plane_rk_global,
						0 /* verbose_level */);
		if (f_v) {
			cout << "surface_clebsch_map::init "
					"intersection point " << intersection_points[u] << endl;
		}
		SOA->Surf->P->unrank_point(v, intersection_points[u]);
		if (f_v) {
			cout << "surface_clebsch_map::init "
					"which is ";
			int_vec_print(cout, v, 4);
			cout << endl;
		}
		SOA->F->reduce_mod_subspace_and_get_coefficient_vector(
			3, 4, Plane, base_cols,
			v, coefficients,
			0 /* verbose_level */);
		if (f_v) {
			cout << "surface_clebsch_map::init "
					"local coefficients ";
			int_vec_print(cout, coefficients, 3);
			cout << endl;
		}
		//intersection_points_local[u] =
		//Surf->P2->rank_point(coefficients);
	}


	Clebsch_map = NEW_lint(SOA->SO->nb_pts);
	Clebsch_coeff = NEW_int(SOA->SO->nb_pts * 4);

	if (!SOA->Surf->clebsch_map(
			SOA->SO->Lines,
			SOA->SO->Pts,
			SOA->SO->nb_pts,
			line_idx,
			plane_rk_global,
			Clebsch_map,
			Clebsch_coeff,
			verbose_level)) {
		cout << "The plane contains one of the lines, "
				"this should not happen" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "clebsch map for lines " << line1
			<< " = " << SOA->Surf->Line_label_tex[line1] << ", "
			<< line2 << " = " << SOA->Surf->Line_label_tex[line2]
			<< " before clebsch_map_print_fibers:" << endl;
	}
	SOA->SO->clebsch_map_print_fibers(Clebsch_map);

	if (f_v) {
		cout << "clebsch map for lines " << line1
			<< " = " << SOA->Surf->Line_label_tex[line1] << ", "
			<< line2 << " = " << SOA->Surf->Line_label_tex[line2]
			<< "  before clebsch_map_find_arc_and_lines:" << endl;
	}

	SOA->SO->clebsch_map_find_arc_and_lines(Clebsch_map,
			Arc, Blown_up_lines, 0 /* verbose_level */);



	if (f_v) {
		cout << "surface_clebsch_map::init "
				"after clebsch_map_find_arc_and_lines" << endl;
	}
	//clebsch_map_find_arc(Clebsch_map, Pts, nb_pts, Arc,
	//0 /* verbose_level */);

	if (f_v) {
		cout << "surface_clebsch_map::init "
				"Clebsch map for lines " << line1 << ", "
				<< line2 << " yields arc = ";
		lint_vec_print(cout, Arc, 6);
		cout << " : blown up lines = ";
		lint_vec_print(cout, Blown_up_lines, 6);
		cout << endl;
	}







}

}}
