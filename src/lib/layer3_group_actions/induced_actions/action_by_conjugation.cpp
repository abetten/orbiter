// action_by_conjugation.cpp
//
// Anton Betten
// January 11, 2009

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_by_conjugation::action_by_conjugation()
{
	Record_birth();
	Base_group = NULL;
	f_ownership = false;
	goi = 0;

	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
}


action_by_conjugation::~action_by_conjugation()
{
	Record_death();
	if (Base_group && f_ownership) {
		FREE_OBJECT(Base_group);
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
}


void action_by_conjugation::init(
		groups::sims *Base_group,
		int f_ownership, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::ring_theory::longinteger_object go;
	actions::action *A;
	
	if (f_v) {
		cout << "action_by_conjugation::init" << endl;
		}
	action_by_conjugation::Base_group = Base_group;
	action_by_conjugation::f_ownership = f_ownership;
	A = Base_group->A;
	Base_group->group_order(go);
	goi = go.as_lint();
	if (f_v) {
		cout << "action_by_conjugation::init we are acting "
				"on a group of order " << goi << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	if (f_v) {
		cout << "action_by_conjugation::init done" << endl;
	}
}

long int action_by_conjugation::compute_image(
		actions::action *A,
		int *Elt, long int i, int verbose_level)
// computes s^-1 * a * s
// where s = Elt and a = group element i from the Base_group
{
	int f_v = (verbose_level >= 1);
	long int j;

	if (f_v) {
		cout << "action_by_conjugation::compute_image "
				"i = " << i << endl;
	}
	if (i < 0 || i >= goi) {
		cout << "action_by_conjugation::compute_image "
				"i = " << i << " out of range" << endl;
		exit(1);
	}
	A->Group_element->invert(Elt, Elt2);
	Base_group->element_unrank_lint(i, Elt1);
	A->Group_element->mult(Elt2, Elt1, Elt3);
	A->Group_element->mult(Elt3, Elt, Elt1);
	j = Base_group->element_rank_lint(Elt1);
	if (f_v) {
		cout << "action_by_conjugation::compute_image "
				"image is " << j << endl;
	}
	return j;
}

long int action_by_conjugation::rank(
		int *Elt)
{
	long int j;

	j = Base_group->element_rank_lint(Elt);

	return j;
}

long int action_by_conjugation::multiply(
		actions::action *A,
		long int i, long int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int k;

	if (f_v) {
		cout << "action_by_conjugation::multiply" << endl;
	}
	if (i < 0 || i >= goi) {
		cout << "action_by_conjugation::multiply "
				"i = " << i << " out of range" << endl;
		exit(1);
	}
	if (j < 0 || j >= goi) {
		cout << "action_by_conjugation::multiply "
				"j = " << j << " out of range" << endl;
		exit(1);
	}
	Base_group->element_unrank_lint(i, Elt1);
	Base_group->element_unrank_lint(j, Elt2);
	A->Group_element->mult(Elt1, Elt2, Elt3);
	k = Base_group->element_rank_lint(Elt3);
	if (f_v) {
		cout << "action_by_conjugation::multiply "
				"the product is " << k << endl;
	}
	return k;
}

}}}

