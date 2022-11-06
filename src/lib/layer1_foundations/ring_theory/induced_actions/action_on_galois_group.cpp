/*
 * action_on_galois_group.cpp
 *
 *  Created on: Mar 23, 2019
 *      Author: betten
 */



#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_galois_group::action_on_galois_group()
{
	null();
}

action_on_galois_group::~action_on_galois_group()
{
	free();
}

void action_on_galois_group::null()
{
	M = NULL;
}

void action_on_galois_group::free()
{

	null();
}


void action_on_galois_group::init(actions::action *A,
		int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "action_on_galois_group::init" << endl;
		cout << "m=" << m << endl;
		}
	action_on_galois_group::A = A;
	action_on_galois_group::m = m;
	if (A->type_G != matrix_group_t) {
		cout << "action_on_galois_group::init action "
				"not of matrix group type" << endl;
		exit(1);
		}
	M = A->G.matrix_grp;
	if (M->f_semilinear == FALSE) {
		cout << "action_on_galois_group::init "
				"M->f_semilinear == FALSE" << endl;
		exit(1);
	}
	q = M->GFq->q;
	degree = M->GFq->e;
	if (f_v) {
		cout << "degree=" << degree << endl;
		}

	if (f_v) {
		cout << "action_on_galois_group::init "
				"field order is " << q << endl;
		}
}

long int action_on_galois_group::compute_image(
		int *Elt, long int i, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	long int a, b, j;

	if (f_v) {
		cout << "action_on_galois_group::compute_image "
				"i = " << i << endl;
		}
	if (i < 0 || i >= degree) {
		cout << "action_on_galois_group::compute_image "
				"i = " << i << " out of range" << endl;
		exit(1);
		}

	a = A->linear_entry_frobenius(Elt);
	b = i + a;
	j = b % degree;

	if (f_v) {
		cout << "action_on_galois_group::compute_image "
				"image of " << i << " is " << j << endl;
		}
	return j;
}

}}}


