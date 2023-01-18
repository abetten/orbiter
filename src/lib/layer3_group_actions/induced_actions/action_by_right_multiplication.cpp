// action_by_right_multiplication.cpp
//
// Anton Betten
// January 10, 2009

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_by_right_multiplication::action_by_right_multiplication()
{
	Base_group = NULL;
	f_ownership = FALSE;
	goi = 0;
	Elt1 = NULL;
	Elt2 = NULL;
}


action_by_right_multiplication::~action_by_right_multiplication()
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
}

void action_by_right_multiplication::init(
		groups::sims *Base_group,
		int f_ownership, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go;
	actions::action *A;
	
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

long int action_by_right_multiplication::compute_image(
		actions::action *A, int *Elt,
		long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int j;

	if (f_v) {
		cout << "action_by_right_multiplication::compute_image i = " << i << endl;
		}
	if (i < 0 || i >= goi) {
		cout << "action_by_right_multiplication::compute_image i = " << i << " out of range" << endl;
		exit(1);
		}
	Base_group->element_unrank_lint(i, Elt1);
	A->mult(Elt1, Elt, Elt2);
	j = Base_group->element_rank_lint(Elt2);
	if (f_v) {
		cout << "action_by_right_multiplication::compute_image image is " << j << endl;
		}
	return j;
}

}}}


