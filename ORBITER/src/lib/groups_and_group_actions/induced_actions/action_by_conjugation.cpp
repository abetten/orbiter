// action_by_conjugation.C
//
// Anton Betten
// January 11, 2009

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

action_by_conjugation::action_by_conjugation()
{
	null();
}

action_by_conjugation::~action_by_conjugation()
{
	free();
}

void action_by_conjugation::null()
{
	f_ownership = FALSE;
	Base_group = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
}

void action_by_conjugation::free()
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
	if (Elt3) {
		FREE_int(Elt3);
		}
	null();
}


void action_by_conjugation::init(sims *Base_group, int f_ownership, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object go;
	action *A;
	
	if (f_v) {
		cout << "action_by_conjugation::init" << endl;
		}
	action_by_conjugation::Base_group = Base_group;
	action_by_conjugation::f_ownership = f_ownership;
	A = Base_group->A;
	Base_group->group_order(go);
	goi = go.as_int();
	if (f_v) {
		cout << "action_by_conjugation::init we are acting on a group of order " << goi << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	if (f_v) {
		cout << "action_by_conjugation::init done" << endl;
		}
}

int action_by_conjugation::compute_image(action *A, int *Elt, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;

	if (f_v) {
		cout << "action_by_conjugation::compute_image i = " << i << endl;
		}
	if (i < 0 || i >= goi) {
		cout << "action_by_conjugation::compute_image i = " << i << " out of range" << endl;
		exit(1);
		}
	A->invert(Elt, Elt2);
	Base_group->element_unrank_int(i, Elt1);
	A->mult(Elt2, Elt1, Elt3);
	A->mult(Elt3, Elt, Elt1);
	j = Base_group->element_rank_int(Elt1);
	if (f_v) {
		cout << "action_by_conjugation::compute_image image is " << j << endl;
		}
	return j;
}

int action_by_conjugation::rank(int *Elt)
{
	int j;

	j = Base_group->element_rank_int(Elt);

	return j;
}

int action_by_conjugation::multiply(action *A, int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int k;

	if (f_v) {
		cout << "action_by_conjugation::multiply" << endl;
		}
	if (i < 0 || i >= goi) {
		cout << "action_by_conjugation::multiply i = " << i << " out of range" << endl;
		exit(1);
		}
	if (j < 0 || j >= goi) {
		cout << "action_by_conjugation::multiply j = " << j << " out of range" << endl;
		exit(1);
		}
	Base_group->element_unrank_int(i, Elt1);
	Base_group->element_unrank_int(j, Elt2);
	A->mult(Elt1, Elt2, Elt3);
	k = Base_group->element_rank_int(Elt3);
	if (f_v) {
		cout << "action_by_conjugation::multiply the product is " << k << endl;
		}
	return k;
}

