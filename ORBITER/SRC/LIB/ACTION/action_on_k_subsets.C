// action_on_k_subsets.C
//
// Anton Betten
// May 15, 2012

#include "galois.h"
#include "action.h"

INT action_on_k_subsets::cntr_new = 0;
INT action_on_k_subsets::cntr_objects = 0;
INT action_on_k_subsets::f_debug_memory = FALSE;

void *action_on_k_subsets::operator new(size_t bytes)
{
	cntr_new++;
	cntr_objects++;
	if (f_debug_memory) {
		cout << "action_on_k_subsets::operator new bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void *action_on_k_subsets::operator new[](size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(action_on_k_subsets);
	cntr_new++;
	cntr_objects += n;
	if (f_debug_memory) {
		cout << "action_on_k_subsets::operator new[] n=" << n 
			<< " bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void action_on_k_subsets::operator delete(void *ptr, size_t bytes)
{
	if (f_debug_memory) {
		cout << "action_on_k_subsets::operator delete bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects--;
	return ::free(ptr);
}

void action_on_k_subsets::operator delete[](void *ptr, size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(action_on_k_subsets);
	if (f_debug_memory) {
		cout << "action_on_k_subsets::operator delete[] n=" << n 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects -= n;
	return ::free(ptr);
}

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
		FREE_INT(set1);
		}
	if (set2) {
		FREE_INT(set2);
		}
	null();
}

void action_on_k_subsets::init(action *A, INT k, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT n;
	
	if (f_v) {
		cout << "action_on_k_subsets::init k=" << k << endl;
		}
	action_on_k_subsets::A = A;
	action_on_k_subsets::k = k;
	n = A->degree;
	degree = INT_n_choose_k(n, k);
	set1 = NEW_INT(k);
	set2 = NEW_INT(k);
	if (f_v) {
		cout << "action_on_k_subsets::init n=" << n << endl;
		cout << "action_on_k_subsets::init n choose k=" << degree << endl;
		}
}

void action_on_k_subsets::compute_image(INT *Elt, INT i, INT &j, INT verbose_level)
{
	INT u, a, b;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "action_on_k_subsets::compute_image i = " << i << endl;
		}
	if (i < 0 || i >= degree) {
		cout << "action_on_k_subsets::compute_image i = " << i << " out of range" << endl;
		exit(1);
		}
	unrank_k_subset(i, set1, A->degree, k);
	for (u = 0; u < k; u++) {
		a = set1[u];
		b = A->image_of(Elt, a);
		set2[u] = b;
		}
	INT_vec_heapsort(set2, k);
	j = rank_k_subset(set2, A->degree, k);
	if (f_vv) {
		cout << "set " << i << " = ";
		INT_vec_print(cout, set1, k);
		cout << " maps to ";
		INT_vec_print(cout, set2, k);
		cout << " = " << j << endl;
		}
	if (j < 0 || j >= degree) {
		cout << "action_on_k_subsets::compute_image j = " << j << " out of range" << endl;
		exit(1);
		}
}



