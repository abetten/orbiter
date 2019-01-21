// action_by_right_multiplication.C
//
// Anton Betten
// January 10, 2009

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

namespace orbiter {

action_by_right_multiplication::action_by_right_multiplication()
{
	null();
}

action_by_right_multiplication::~action_by_right_multiplication()
{
	free();
}

void action_by_right_multiplication::null()
{
	Base_group = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
}

void action_by_right_multiplication::free()
{
	
	if (Base_group && f_ownership) {
		delete Base_group;
		}
	if (Elt1) {
		FREE_int(Elt1);
		}
	if (Elt2) {
		FREE_int(Elt2);
		}
	null();
}


void action_by_right_multiplication::init(sims *Base_group, int f_ownership, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object go;
	action *A;
	
	if (f_v) {
		cout << "action_by_right_multiplication::init" << endl;
		}
	action_by_right_multiplication::Base_group = Base_group;
	action_by_right_multiplication::f_ownership = f_ownership;
	A = Base_group->A;
	Base_group->group_order(go);
	goi = go.as_int();
	if (f_v) {
		cout << "action_by_right_multiplication::init we are acting on a group of order " << goi << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
}

void action_by_right_multiplication::compute_image(action *A, int *Elt, int i, int &j, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_by_right_multiplication::compute_image i = " << i << endl;
		}
	if (i < 0 || i >= goi) {
		cout << "action_by_right_multiplication::compute_image i = " << i << " out of range" << endl;
		exit(1);
		}
	Base_group->element_unrank_int(i, Elt1);
	A->mult(Elt1, Elt, Elt2);
	j = Base_group->element_rank_int(Elt2);
	if (f_v) {
		cout << "action_by_right_multiplication::compute_image image is " << j << endl;
		}
}

}

