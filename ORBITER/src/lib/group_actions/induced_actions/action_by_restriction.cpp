// action_by_restriction.C
//
// Anton Betten
// February 20, 2010

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {

action_by_restriction::action_by_restriction()
{
	null();
}

action_by_restriction::~action_by_restriction()
{
	free();
}

void action_by_restriction::null()
{
	points = NULL;
	points_sorted = NULL;
	perm_inv = NULL;
}

void action_by_restriction::free()
{
	if (points) {
		FREE_int(points);
		}
	if (points_sorted) {
		FREE_int(points_sorted);
		}
	if (perm_inv) {
		FREE_int(perm_inv);
		}
	null();
}

void action_by_restriction::init_from_schreier_vector(
		schreier_vector *Schreier_vector,
		int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_by_restriction::init_from_schreier_vector "
				"pt=" << pt << endl;
		}
	int *orbit_elts;
	int orbit_len;

	Schreier_vector->orbit_of_point(
			pt, orbit_elts, orbit_len,
			verbose_level);

	init(orbit_len, orbit_elts, verbose_level);

	FREE_int(orbit_elts);

	if (f_v) {
		cout << "action_by_restriction::init_from_schreier_vector "
				"done" << endl;
		}
}

void action_by_restriction::init(int nb_points, int *points,
		int verbose_level)
// the array points must be orderd
{
	int i;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_by_restriction::init nb_points="
				<< nb_points << endl;
		}
	action_by_restriction::nb_points = nb_points;
	action_by_restriction::points = NEW_int(nb_points);
	action_by_restriction::points_sorted = NEW_int(nb_points);
	action_by_restriction::perm_inv = NEW_int(nb_points);
	for (i = 0; i < nb_points; i++) {
		action_by_restriction::points[i] = points[i];
		points_sorted[i] = points[i];
		perm_inv[i] = i;
		}
	int_vec_heapsort_with_log(points_sorted, perm_inv, nb_points);
	if (f_v) {
		cout << "action_by_restriction::init finished" << endl;
		}
}

int action_by_restriction::compute_image(
		action *A, int *Elt, int i, int verbose_level)
{
	int idx, b, c, h;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

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
		A->element_print_quick(Elt, cout);
		cout << endl;
		}
	if (f_vv) {
		cout << "points[i]=" << points[i] << endl;
		}
	b = A->element_image_of(points[i], Elt, verbose_level - 2);
	if (f_vv) {
		cout << "image of " << points[i] << " is " << b << endl;
		}
	if (!int_vec_search(points_sorted, nb_points, b, idx)) {
		cout << "action_by_restriction::compute_image fatal: "
				"image point " << b << " not found" << endl;
		cout << "action: ";
		A->print_info();

		cout << "the element " << endl;
		A->element_print_quick(Elt, cout);
		cout << endl;
		//cout << "as permutation:" << endl;
		//A->print_as_permutation(cout, Elt);
		//cout << endl;

		cout << "i=" << i << endl;
		cout << "image of " << points[i] << " is " << b << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "points=";
		int_vec_print(cout, points, nb_points);
		cout << endl;
		cout << "points_sorted=" << endl;
		for (h = 0; h < nb_points; h++) {
			cout << h << " : " << points_sorted[h] << endl;
			}
		//int_vec_print(cout, points_sorted, nb_points);
		//cout << endl;

		cout << "We compute the image point again, "
				"this time with more output" << endl;
		b = A->element_image_of(points[i],
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

}}


