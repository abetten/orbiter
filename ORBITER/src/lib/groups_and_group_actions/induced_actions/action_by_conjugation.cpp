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
		FREE_INT(Elt1);
		}
	if (Elt2) {
		FREE_INT(Elt2);
		}
	if (Elt3) {
		FREE_INT(Elt3);
		}
	null();
}


void action_by_conjugation::init(sims *Base_group, INT f_ownership, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_object go;
	action *A;
	
	if (f_v) {
		cout << "action_by_conjugation::init" << endl;
		}
	action_by_conjugation::Base_group = Base_group;
	action_by_conjugation::f_ownership = f_ownership;
	A = Base_group->A;
	Base_group->group_order(go);
	goi = go.as_INT();
	if (f_v) {
		cout << "action_by_conjugation::init we are acting on a group of order " << goi << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	if (f_v) {
		cout << "action_by_conjugation::init done" << endl;
		}
}

INT action_by_conjugation::compute_image(action *A, INT *Elt, INT i, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT j;

	if (f_v) {
		cout << "action_by_conjugation::compute_image i = " << i << endl;
		}
	if (i < 0 || i >= goi) {
		cout << "action_by_conjugation::compute_image i = " << i << " out of range" << endl;
		exit(1);
		}
	A->invert(Elt, Elt2);
	Base_group->element_unrank_INT(i, Elt1);
	A->mult(Elt2, Elt1, Elt3);
	A->mult(Elt3, Elt, Elt1);
	j = Base_group->element_rank_INT(Elt1);
	if (f_v) {
		cout << "action_by_conjugation::compute_image image is " << j << endl;
		}
	return j;
}

INT action_by_conjugation::rank(INT *Elt)
{
	INT j;

	j = Base_group->element_rank_INT(Elt);

	return j;
}

INT action_by_conjugation::multiply(action *A, INT i, INT j, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT k;

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
	Base_group->element_unrank_INT(i, Elt1);
	Base_group->element_unrank_INT(j, Elt2);
	A->mult(Elt1, Elt2, Elt3);
	k = Base_group->element_rank_INT(Elt3);
	if (f_v) {
		cout << "action_by_conjugation::multiply the product is " << k << endl;
		}
	return k;
}

