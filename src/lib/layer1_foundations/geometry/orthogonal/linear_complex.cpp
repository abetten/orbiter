/*
 * linear_complex.cpp
 *
 *  Created on: Jan 29, 2023
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace orthogonal_geometry {


linear_complex::linear_complex()
{
	Record_birth();
	Surf = NULL;

	pt0_wedge = 0;
	pt0_line = 0;
	pt0_klein = 0;

	nb_neighbors = 0;
	Neighbors = NULL;
	Neighbor_to_line = NULL;
	Neighbor_to_klein = NULL;
}

linear_complex::~linear_complex()
{
	Record_death();
}

void linear_complex::init(
		geometry::algebraic_geometry::surface_domain *Surf,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_complex::init" << endl;
	}

	linear_complex::Surf = Surf;

	pt0_line = 0; // pt0 = the line spanned by 1000, 0100
		// (we call it point because it is a point on the Klein quadric)
	pt0_wedge = 0; // in wedge coordinates 100000
	pt0_klein = 0; // in klein coordinates 100000

	if (f_v) {
		cout << "linear_complex::init "
				"before compute_neighbors" << endl;
	}
	compute_neighbors(verbose_level);
	if (f_v) {
		cout << "linear_complex::init "
				"after compute_neighbors" << endl;
	}
	{
		other::data_structures::spreadsheet *Sp;
		if (f_v) {
			cout << "linear_complex::init "
					"before make_spreadsheet_of_neighbors" << endl;
		}
		make_spreadsheet_of_neighbors(
				Sp, 0 /* verbose_level */);
		if (f_v) {
			cout << "linear_complex::init "
					"after make_spreadsheet_of_neighbors" << endl;
		}
		FREE_OBJECT(Sp);
	}
	if (f_v) {
		cout << "linear_complex::init "
				"after compute_neighbors "
				"nb_neighbors = " << nb_neighbors << endl;
		cout << "Neighbors=";
		Lint_vec_print(cout, Neighbors, nb_neighbors);
		cout << endl;
	}
	if (f_v) {
		cout << "linear_complex::done" << endl;
	}
}

void linear_complex::compute_neighbors(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, a, b, c;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "linear_complex::compute_neighbors" << endl;
	}

	nb_neighbors = (long int) (Surf->F->q + 1) * Surf->F->q * (Surf->F->q + 1);
	if (f_v) {
		cout << "linear_complex::compute_neighbors "
				"nb_neighbors = " << nb_neighbors << endl;
	}
	Neighbors = NEW_lint(nb_neighbors);
	Neighbor_to_line = NEW_lint(nb_neighbors);
	Neighbor_to_klein = NEW_lint(nb_neighbors);

	int sz;

	// At first, we get the neighbors
	// as points on the Klein quadric:
	// Later, we will change them to wedge ranks:

	if (f_v) {
		cout << "linear_complex::compute_neighbors "
				"before Surf->O->perp" << endl;
		}
	Surf->O->perp(0, Neighbors, sz, 0 /*verbose_level - 3*/);
	if (f_v) {
		cout << "linear_complex::compute_neighbors "
				"after Surf->O->perp" << endl;

		//cout << "Neighbors:" << endl;
		//lint_matrix_print(Neighbors, (sz + 9) / 10, 10);
	}

	if (sz != nb_neighbors) {
		cout << "linear_complex::compute_neighbors "
				"sz != nb_neighbors" << endl;
		cout << "sz = " << sz << endl;
		cout << "nb_neighbors = " << nb_neighbors << endl;
		exit(1);
	}
	if (f_v) {
		cout << "linear_complex::compute_neighbors "
				"nb_neighbors = " << nb_neighbors << endl;
	}

	if (f_v) {
		cout << "linear_complex::compute_neighbors "
				"allocating Line_to_neighbor, "
				"Surf->nb_lines_PG_3=" << Surf->nb_lines_PG_3 << endl;
	}

#if 0
	Line_to_neighbor = NEW_lint(Surf->nb_lines_PG_3);
	for (i = 0; i < Surf->nb_lines_PG_3; i++) {
		Line_to_neighbor[i] = -1;
	}
#endif


	// Convert Neighbors[] from points
	// on the Klein quadric to wedge points:
	if (f_v) {
		cout << "linear_complex::compute_neighbors "
				"before Surf->klein_to_wedge_vec" << endl;
	}
	Surf->klein_to_wedge_vec(Neighbors, Neighbors, nb_neighbors);

	// Sort the set Neighbors:
	Sorting.lint_vec_heapsort(Neighbors, nb_neighbors);




	// Establish the bijection between Neighbors and Lines in PG(3,q)
	// by going through the Klein correspondence.
	// It is important that this be done after we sort Neighbors.
	if (f_v) {
		cout << "linear_complex::compute_neighbors "
				"Establish the bijection between Neighbors and Lines in "
				"PG(3,q), nb_neighbors=" << nb_neighbors << endl;
	}
	int N100;
	int w[6];
	int v[6];

	N100 = nb_neighbors / 100 + 1;

	for (i = 0; i < nb_neighbors; i++) {
		if ((i % N100) == 0) {
			cout << "linear_complex::compute_neighbors i=" << i << " / "
					<< nb_neighbors << " at "
					<< (double)i * 100. / nb_neighbors << "%" << endl;
		}
		a = Neighbors[i];
		//AW->unrank_point(w, a);
		Surf->F->Projective_space_basic->PG_element_unrank_modified_lint(
				w, 1, 6 /* wedge_dimension */, a);

		Surf->wedge_to_klein(w, v);
		if (false) {
			cout << i << " : ";
			Int_vec_print(cout, v, 6);
			cout << endl;
		}
		b = Surf->O->Hyperbolic_pair->rank_point(
				v, 1, 0 /* verbose_level*/);
		if (false) {
			cout << " : " << b;
			cout << endl;
		}
		c = Surf->Klein->point_on_quadric_to_line(
				b, 0 /* verbose_level*/);
		if (false) {
			cout << " : " << c << endl;
			cout << endl;
		}
		Neighbor_to_line[i] = c;
		//Line_to_neighbor[c] = i;
		}

	if (f_v) {
		cout << "linear_complex::compute_neighbors "
				"before int_vec_apply" << endl;
	}
	for (i = 0; i < nb_neighbors; i++) {
		Neighbor_to_klein[i] = Surf->Klein->line_to_point_on_quadric(
				Neighbor_to_line[i], 0 /* verbose_level*/);
	}
#if 0
	lint_vec_apply(Neighbor_to_line,
			Surf->Klein->Line_to_point_on_quadric,
			Neighbor_to_klein, nb_neighbors);
#endif


	if (f_v) {
		cout << "linear_complex::compute_neighbors done" << endl;
	}
}

void linear_complex::make_spreadsheet_of_neighbors(
		other::data_structures::spreadsheet *&Sp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_csv;

	if (f_v) {
		cout << "linear_complex::make_spreadsheet_of_neighbors" << endl;
	}

	fname_csv = "neighbors_" + std::to_string(Surf->F->q) + ".csv";


	Surf->make_spreadsheet_of_lines_in_three_kinds(Sp,
		Neighbors, Neighbor_to_line,
		Neighbor_to_klein, nb_neighbors, 0 /* verbose_level */);

	if (f_v) {
		cout << "before Sp->save " << fname_csv << endl;
	}
	Sp->save(fname_csv, verbose_level);
	if (f_v) {
		cout << "after Sp->save " << fname_csv << endl;
	}





	if (f_v) {
		cout << "linear_complex::make_spreadsheet_of_neighbors done" << endl;
	}
}


}}}}



