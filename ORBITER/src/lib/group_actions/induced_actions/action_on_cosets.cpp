// action_on_cosets.C
//
// Anton Betten
// Dec 24, 2013

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {

action_on_cosets::action_on_cosets()
{
	null();
}

action_on_cosets::~action_on_cosets()
{
	freeself();
}

void action_on_cosets::null()
{
	v1 = NULL;
	v2 = NULL;
}

void action_on_cosets::freeself()
{
	int f_v = FALSE;
	//int f_vv = FALSE;

	if (f_v) {
		cout << "action_on_cosets::free" << endl;
		}
	if (v1) {
		FREE_int(v1);
		}
	if (v2) {
		FREE_int(v2);
		}
	null();
	if (f_v) {
		cout << "action_on_cosets::free done" << endl;
		}
}

void action_on_cosets::init(int nb_points, int *Points, 
	action *A_linear, 
	finite_field *F, 
	int dimension_of_subspace, 
	int n, 
	int *subspace_basis, 
	int *base_cols, 
	void (*unrank_point)(int *v, int a, void *data), 
	int (*rank_point)(int *v, void *data), 
	void *rank_unrank_data, 
	int verbose_level)
{
	int f_v = FALSE;
	int i;

	if (f_v) {
		cout << "action_on_cosets::init nb_points=" << nb_points
				<< " dimension_of_subspace=" << dimension_of_subspace
				<< " n=" << n << endl;
		}
	action_on_cosets::nb_points = nb_points;
	action_on_cosets::Points = Points;
	action_on_cosets::A_linear = A_linear;
	action_on_cosets::F = F;
	action_on_cosets::dimension_of_subspace = dimension_of_subspace;
	action_on_cosets::n = n;
	action_on_cosets::subspace_basis = subspace_basis;
	action_on_cosets::base_cols = base_cols;
	action_on_cosets::unrank_point = unrank_point;
	action_on_cosets::rank_point = rank_point;
	action_on_cosets::rank_unrank_data = rank_unrank_data;
	v1 = NEW_int(n);
	v2 = NEW_int(n);
	for (i = 0; i < nb_points - 1; i++) {
		if (Points[i] >= Points[i + 1]) {
			cout << "action_on_cosets::init the array Points[] "
					"is not sorted increasingly" << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "action_on_cosets::init done" << endl;
		}
}

void action_on_cosets::reduce_mod_subspace(int *v, int verbose_level)
{
	F->reduce_mod_subspace(dimension_of_subspace, n, 
		subspace_basis, base_cols, v, verbose_level);
}


int action_on_cosets::compute_image(int *Elt, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int r, idx;
	sorting Sorting;
	
	if (f_v) {
		cout << "action_on_cosets::compute_image i = " << i << endl;
		}
	if (i >= nb_points) {
		cout << "action_on_cosets::compute_image "
				"i = " << i << " i >= nb_points" << endl;
		exit(1);
		}
	(*unrank_point)(v1, Points[i], rank_unrank_data);
	if (f_vv) {
		cout << "action_on_cosets::compute_image after unrank:";
		int_vec_print(cout, v1, n);
		cout << endl;
		}
	
	A_linear->element_image_of_low_level(v1, v2,
			Elt, 0/*verbose_level - 1*/);

	if (f_vv) {
		cout << "action_on_cosets::compute_image "
				"after element_image_of_low_level:";
		int_vec_print(cout, v2, n);
		cout << endl;
		}

	reduce_mod_subspace(v2, 0 /* verbose_level */);
	
	if (f_vv) {
		cout << "action_on_cosets::compute_image "
				"after reduce_mod_subspace:";
		int_vec_print(cout, v2, n);
		cout << endl;
		}

	r = (*rank_point)(v2, rank_unrank_data);
	if (f_vv) {
		cout << "action_on_cosets::compute_image after "
				"rank, r = " << r << endl;
		}
	if (!Sorting.int_vec_search(Points, nb_points, r, idx)) {
		cout << "action_on_cosets::compute_image image "
				<< r << " not found in list pf points" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "action_on_cosets::compute_image image "
				"of " << i << " is " << idx << endl;
		}
	return idx;
}



}}

