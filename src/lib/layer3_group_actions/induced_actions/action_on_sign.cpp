// action_on_sign.cpp
//
// Anton Betten
// August 12, 2016

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_sign::action_on_sign()
{
	Record_birth();
	A = NULL;
	perm_degree = 0;
	perm = NULL;
	degree = 0;
}

action_on_sign::~action_on_sign()
{
	Record_death();
	if (perm) {
		FREE_int(perm);
	}
	//null();
}


void action_on_sign::init(
		actions::action *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::ring_theory::longinteger_object go;
	
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

long int action_on_sign::compute_image(
		int *Elt,
		long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int u, v, sgn, j;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "action_on_sign::compute_image "
				"i = " << i << endl;
	}
	if (i < 0 || i >= degree) {
		cout << "action_on_sign::compute_image "
				"i = " << i << " out of range" << endl;
		exit(1);
	}
	for (u = 0; u < perm_degree; u++) {
		v = A->Group_element->element_image_of(u, Elt, false);
		perm[u] = v;
	}
	sgn = Combi.Permutations->perm_signum(perm, perm_degree);
	if (sgn == -1) {
		j = (i + 1) % 2;
	}
	else {
		j = i;
	}
	
	if (f_v) {
		cout << "action_on_sign::compute_image "
				"image of " << i << " is " << j << endl;
	}
	return j;
}

}}}

