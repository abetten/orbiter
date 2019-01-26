// action_on_sign.C
//
// Anton Betten
// August 12, 2016

#include "foundations/foundations.h"
#include "group_actions.h"

namespace orbiter {

action_on_sign::action_on_sign()
{
	null();
}

action_on_sign::~action_on_sign()
{
	free();
}

void action_on_sign::null()
{
	perm = NULL;
}

void action_on_sign::free()
{	
	if (perm) {
		FREE_int(perm);
		}
	null();
}


void action_on_sign::init(action *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object go;
	
	if (f_v) {
		cout << "action_on_sign::init" << endl;
		}
	action_on_sign::A = A;
	perm_degree = A->degree;
	if (f_v) {
		cout << "perm_degree=" << perm_degree << endl;
		}
	perm = NEW_int(perm_degree);
	degree = 2;
	
	if (f_v) {
		cout << "action_on_sign::init" << endl;
		}
}

void action_on_sign::compute_image(int *Elt, int i, int &j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int u, v, sgn;
	
	if (f_v) {
		cout << "action_on_sign::compute_image i = " << i << endl;
		}
	if (i < 0 || i >= degree) {
		cout << "action_on_sign::compute_image i = " << i << " out of range" << endl;
		exit(1);
		}
	for (u = 0; u < perm_degree; u++) {
		v = A->element_image_of(u, Elt, FALSE);
		perm[u] = v;
		}
	sgn = perm_signum(perm, perm_degree);
	if (sgn == -1) {
		j = (i + 1) % 2;
		}
	else {
		j = i;
		}
	
	if (f_v) {
		cout << "action_on_sign::compute_image  image of " << i << " is " << j << endl;
		}
}

}

