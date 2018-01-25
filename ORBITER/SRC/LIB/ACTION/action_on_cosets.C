// action_on_cosets.C
//
// Anton Betten
// Dec 24, 2013

#include "galois.h"
#include "action.h"

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
	INT f_v = FALSE;
	//INT f_vv = FALSE;

	if (f_v) {
		cout << "action_on_cosets::free" << endl;
		}
	if (v1) {
		FREE_INT(v1);
		}
	if (v2) {
		FREE_INT(v2);
		}
	null();
	if (f_v) {
		cout << "action_on_cosets::free done" << endl;
		}
}

void action_on_cosets::init(INT nb_points, INT *Points, 
	action *A_linear, 
	finite_field *F, 
	INT dimension_of_subspace, 
	INT n, 
	INT *subspace_basis, 
	INT *base_cols, 
	void (*unrank_point)(INT *v, INT a, void *data), 
	INT (*rank_point)(INT *v, void *data), 
	void *rank_unrank_data, 
	INT verbose_level)
{
	INT f_v = FALSE;
	INT i;

	if (f_v) {
		cout << "action_on_cosets::init nb_points=" << nb_points << " dimension_of_subspace=" << dimension_of_subspace << " n=" << n << endl;
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
	v1 = NEW_INT(n);
	v2 = NEW_INT(n);
	for (i = 0; i < nb_points - 1; i++) {
		if (Points[i] >= Points[i + 1]) {
			cout << "action_on_cosets::init the array Points[] is not sorted increasingly" << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "action_on_cosets::init done" << endl;
		}
}

void action_on_cosets::reduce_mod_subspace(INT *v, INT verbose_level)
{
	F->reduce_mod_subspace(dimension_of_subspace, n, 
		subspace_basis, base_cols, v, verbose_level);
}


INT action_on_cosets::compute_image(INT *Elt, INT i, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT r, idx;
	
	if (f_v) {
		cout << "action_on_cosets::compute_image i = " << i << endl;
		}
	if (i >= nb_points) {
		cout << "action_on_cosets::compute_image i = " << i << " i >= nb_points" << endl;
		exit(1);
		}
	(*unrank_point)(v1, Points[i], rank_unrank_data);
	if (f_vv) {
		cout << "action_on_cosets::compute_image after unrank:";
		INT_vec_print(cout, v1, n);
		cout << endl;
		}
	
	A_linear->element_image_of_low_level(v1, v2, Elt, 0/*verbose_level - 1*/);

	if (f_vv) {
		cout << "action_on_cosets::compute_image after element_image_of_low_level:";
		INT_vec_print(cout, v2, n);
		cout << endl;
		}

	reduce_mod_subspace(v2, 0 /* verbose_level */);
	
	if (f_vv) {
		cout << "action_on_cosets::compute_image after reduce_mod_subspace:";
		INT_vec_print(cout, v2, n);
		cout << endl;
		}

	r = (*rank_point)(v2, rank_unrank_data);
	if (f_vv) {
		cout << "action_on_cosets::compute_image after rank, r = " << r << endl;
		}
	if (!INT_vec_search(Points, nb_points, r, idx)) {
		cout << "action_on_cosets::compute_image image " << r << " not found in list pf points" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "action_on_cosets::compute_image image of " << i << " is " << idx << endl;
		}
	return idx;
}




