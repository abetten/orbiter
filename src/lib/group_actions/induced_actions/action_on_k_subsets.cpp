// action_on_k_subsets.cpp
//
// Anton Betten
// May 15, 2012

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {

action_on_k_subsets::action_on_k_subsets()
{
	null();
}

action_on_k_subsets::~action_on_k_subsets()
{
	free();
}

void action_on_k_subsets::null()
{
	A = NULL;
	set1 = NULL;
	set2 = NULL;
}

void action_on_k_subsets::free()
{
	if (set1) {
		FREE_int(set1);
		}
	if (set2) {
		FREE_int(set2);
		}
	null();
}

void action_on_k_subsets::init(actions::action *A,
		int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n;
	combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "action_on_k_subsets::init k=" << k << endl;
		}
	action_on_k_subsets::A = A;
	action_on_k_subsets::k = k;
	n = A->degree;
	degree = Combi.int_n_choose_k(n, k);
	set1 = NEW_int(k);
	set2 = NEW_int(k);
	if (f_v) {
		cout << "action_on_k_subsets::init n=" << n << endl;
		cout << "action_on_k_subsets::init "
				"n choose k=" << degree << endl;
		}
}

long int action_on_k_subsets::compute_image(
		int *Elt, long int i, int verbose_level)
{
	long int u, a, b, j;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "action_on_k_subsets::compute_image "
				"i = " << i << endl;
		}
	if (i < 0 || i >= degree) {
		cout << "action_on_k_subsets::compute_image "
				"i = " << i << " out of range" << endl;
		exit(1);
		}
	Combi.unrank_k_subset(i, set1, A->degree, k);
	for (u = 0; u < k; u++) {
		a = set1[u];
		b = A->image_of(Elt, a);
		set2[u] = b;
		}
	Sorting.int_vec_heapsort(set2, k);
	j = Combi.rank_k_subset(set2, A->degree, k);
	if (f_vv) {
		cout << "set " << i << " = ";
		Orbiter->Int_vec->print(cout, set1, k);
		cout << " maps to ";
		Orbiter->Int_vec->print(cout, set2, k);
		cout << " = " << j << endl;
		}
	if (j < 0 || j >= degree) {
		cout << "action_on_k_subsets::compute_image "
				"j = " << j << " out of range" << endl;
		exit(1);
		}
	return j;
}


}}

