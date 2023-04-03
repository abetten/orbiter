// action_by_restriction.cpp
//
// Anton Betten
// February 20, 2010

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_by_restriction::action_by_restriction()
{
	nb_points = 0;
	points = NULL;
	points_sorted = NULL;
	perm_inv = NULL;
	f_single_orbit = false;
	pt = 0;
	idx_of_root_node = 0;
}

action_by_restriction::~action_by_restriction()
{
	if (points) {
		FREE_lint(points);
		}
	if (points_sorted) {
		FREE_lint(points_sorted);
		}
	if (perm_inv) {
		FREE_lint(perm_inv);
		}
}

void action_by_restriction::init_single_orbit_from_schreier_vector(
		data_structures_groups::schreier_vector *Schreier_vector,
		int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_by_restriction::init_single_orbit_from_schreier_vector "
				"pt=" << pt << endl;
	}

	f_single_orbit = true;
	action_by_restriction::pt = pt;

	long int *orbit_elts;
	int orbit_len;


	if (f_v) {
		cout << "action_by_restriction::init_single_orbit_from_schreier_vector "
				"before Schreier_vector->orbit_of_point pt=" << pt << endl;
	}
	Schreier_vector->orbit_of_point(
			pt, orbit_elts, orbit_len, idx_of_root_node,
			verbose_level);
	if (f_v) {
		cout << "action_by_restriction::init_single_orbit_from_schreier_vector "
				"after Schreier_vector->orbit_of_point pt=" << pt << endl;
		cout << "orbit_elts = ";
		Lint_vec_print(cout, orbit_elts, orbit_len);
		cout << endl;
	}

	if (f_v) {
		cout << "action_by_restriction::init_single_orbit_from_schreier_vector "
				"before init" << endl;
	}
	init(orbit_len, orbit_elts, verbose_level);
	if (f_v) {
		cout << "action_by_restriction::init_single_orbit_from_schreier_vector "
				"after init" << endl;
	}

	FREE_lint(orbit_elts);

	if (f_v) {
		cout << "action_by_restriction::init_single_orbit_from_schreier_vector "
				"done" << endl;
	}
}

void action_by_restriction::init(
		int nb_points, long int *points,
		int verbose_level)
// the array points must be ordered
{
	int i;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "action_by_restriction::init nb_points="
				<< nb_points << endl;
		}
	if (f_vv) {
		cout << "action_by_restriction::init points=";
		Lint_vec_print(cout, points, nb_points);
		cout << endl;
	}
	action_by_restriction::nb_points = nb_points;
	action_by_restriction::points = NEW_lint(nb_points);
	action_by_restriction::points_sorted = NEW_lint(nb_points);
	action_by_restriction::perm_inv = NEW_lint(nb_points);
	for (i = 0; i < nb_points; i++) {
		action_by_restriction::points[i] = points[i];
		points_sorted[i] = points[i];
		perm_inv[i] = i;
		}
	Sorting.lint_vec_heapsort_with_log(points_sorted, perm_inv, nb_points);
	if (f_vv) {
		cout << "action_by_restriction::init points after sorting=";
		Lint_vec_print(cout, points_sorted, nb_points);
		cout << endl;
	}
	if (f_v) {
		cout << "action_by_restriction::init finished" << endl;
		}
}

long int action_by_restriction::original_point(long int pt)
{
	return points[pt];
}

long int action_by_restriction::restricted_point_idx(long int pt)
{
	int idx;
	data_structures::sorting Sorting;

	if (!Sorting.lint_vec_search(points_sorted, nb_points, pt, idx, 0 /* verbose_level */)) {
		cout << "action_by_restriction::restricted_point_idx fatal: "
				"point " << pt << " not found" << endl;
		exit(1);
		}
	return perm_inv[idx];
}


long int action_by_restriction::compute_image(
		actions::action *A,
		int *Elt, long int i, int verbose_level)
{
	int idx;
	long int b, c, h;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "action_by_restriction::compute_image i = " << i << endl;
		}
	if (i < 0 || i >= nb_points) {
		cout << "action_by_restriction::compute_image "
				"i = " << i << " out of range" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "under the group element " << endl;
		A->Group_element->element_print_quick(Elt, cout);
		cout << endl;
		}
	if (f_vv) {
		cout << "points[i]=" << points[i] << endl;
		}
	b = A->Group_element->element_image_of(points[i], Elt, verbose_level - 2);
	if (f_vv) {
		cout << "image of " << points[i] << " is " << b << endl;
		}
	if (!Sorting.lint_vec_search(points_sorted, nb_points, b, idx, 0 /* verbose_level */)) {
		cout << "action_by_restriction::compute_image fatal: "
				"image point " << b << " not found" << endl;
		cout << "action: ";
		A->print_info();

		cout << "the element " << endl;
		A->Group_element->element_print_quick(Elt, cout);
		cout << endl;
		//cout << "as permutation:" << endl;
		//A->print_as_permutation(cout, Elt);
		//cout << endl;

		cout << "i=" << i << endl;
		cout << "image of " << points[i] << " is " << b << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "points=";
		Lint_vec_print(cout, points, nb_points);
		cout << endl;
		cout << "points_sorted=" << endl;
		for (h = 0; h < nb_points; h++) {
			cout << h << " : " << points_sorted[h] << endl;
			}
		//int_vec_print(cout, points_sorted, nb_points);
		//cout << endl;

		cout << "We compute the image point again, "
				"this time with more output" << endl;
		b = A->Group_element->element_image_of(points[i],
				Elt, 10 /* verbose_level - 2*/);
		cout << "action_by_restriction::compute_image fatal: "
				"image point " << b << " not found" << endl;
		cout << "action: ";
		A->print_info();
		exit(1);
		}
	if (f_v) {
		cout << "action_on_sets::compute_image idx = " << idx << endl;
		}
	c = perm_inv[idx];
	if (f_v) {
		cout << "action_on_sets::compute_image c = " << c << endl;
		}
	return c;
}

}}}



