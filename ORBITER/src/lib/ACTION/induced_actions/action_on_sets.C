// action_on_sets.C
//
// Anton Betten
// November 13, 2007

#include "GALOIS/galois.h"
#include "action.h"

INT action_on_sets::cntr_new = 0;
INT action_on_sets::cntr_objects = 0;
INT action_on_sets::f_debug_memory = FALSE;

void *action_on_sets::operator new(size_t bytes)
{
	cntr_new++;
	cntr_objects++;
	if (f_debug_memory) {
		cout << "action_on_sets::operator new bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void *action_on_sets::operator new[](size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(action_on_sets);
	cntr_new++;
	cntr_objects += n;
	if (f_debug_memory) {
		cout << "action_on_sets::operator new[] n=" << n 
			<< " bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void action_on_sets::operator delete(void *ptr, size_t bytes)
{
	if (f_debug_memory) {
		cout << "action_on_sets::operator delete bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects--;
	return ::free(ptr);
}

void action_on_sets::operator delete[](void *ptr, size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(action_on_sets);
	if (f_debug_memory) {
		cout << "action_on_sets::operator delete[] n=" << n 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects -= n;
	return ::free(ptr);
}

action_on_sets::action_on_sets()
{
	null();
}

action_on_sets::~action_on_sets()
{
	free();
}

void action_on_sets::null()
{
	sets = NULL;
	image_set = NULL;
	perm = NULL;
	perm_inv = NULL;
}

void action_on_sets::free()
{
	INT i;
	
	if (sets) {
		for (i = 0; i < nb_sets; i++) {
			FREE_INT(sets[i]);
			}
		FREE_PINT(sets);
		}
	if (image_set) {
		FREE_INT(image_set);
		}
	if (perm) {
		FREE_INT(perm);
		}
	if (perm_inv) {
		FREE_INT(perm_inv);
		}
	null();
}


void action_on_sets::init(INT nb_sets, INT set_size, INT *input_sets, INT verbose_level)
{
	INT i, j;
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE; //(verbose_level >= 5);
	
	if (f_v) {
		cout << "action_on_sets::init nb_sets=" << nb_sets << " set_size=" << set_size << endl;
		}
	action_on_sets::nb_sets = nb_sets;
	action_on_sets::set_size = set_size;
	sets = NEW_PINT(nb_sets);
	image_set = NEW_INT(set_size);
	perm = NEW_INT(nb_sets);
	perm_inv = NEW_INT(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		perm[i] = i;
		perm_inv[i] = i;
		}
	for (i = 0; i < nb_sets; i++) {
		sets[i] = NEW_INT(set_size);
		for (j = 0; j < set_size; j++) {
			sets[i][j] = input_sets[i * set_size + j];
			}
		INT_vec_quicksort_increasingly(sets[i], set_size);
		if (f_vv) {
			cout << "set " << setw(3) << i << " is ";
			INT_vec_print(cout, sets[i], set_size);
			cout << endl;
			}
		}
	quicksort_array_with_perm(nb_sets, (void **) sets, perm_inv, action_on_sets_compare, this);
	perm_inverse(perm_inv, perm, nb_sets);

	test_sets();


	if (f_vv) {
		cout << "after quicksort_array_with_perm" << endl;
#if 0
		cout << "i : perm[i] : perm_inv[i]" << endl;
		for (i = 0; i < nb_sets; i++) {
			cout << i << " : " << perm[i] << " : " << perm_inv[i] << endl;
			}
#endif

		print_sets_sorted();

		print_sets_in_original_ordering();

#if 0
		cout << "the sets in the perm_inv ordering:" << endl;
		for (i = 0; i < nb_sets; i++) {
			j = perm_inv[i];
			cout << "set " << i << " is set " << j << " : ";
			INT_vec_print(cout, sets[j], set_size);
			cout << endl;
			}
#endif
		}
	if (f_v) {
		cout << "action_on_sets::init finished" << endl;
		}
}

void action_on_sets::compute_image(action *A, INT *Elt, INT i, INT &j, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT idx, res;

	if (f_v) {
		cout << "action_on_sets::compute_image i = " << i << endl;
		cout << "action_on_sets::compute_image perm[i] = " << perm[i] << endl;
		}
	if (i < 0 || i >= nb_sets) {
		cout << "action_on_sets::compute_image i = " << i << " out of range" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "the element " << endl;
		A->print(cout, Elt);
		cout << endl;
		cout << "as permutation:" << endl;
		A->print_as_permutation(cout, Elt);
		cout << endl;
		}
	if (f_vv) {
		cout << "sets[perm[i]]:" << endl;
		INT_vec_print(cout, sets[perm[i]], set_size);
		cout << endl;
		for (j = 0; j < set_size; j++) {
			cout << j << " : " << sets[perm[i]][j] << " : " << endl;
			A->print_point(sets[perm[i]][j], cout);
			cout << endl;
			}
		}
	A->map_a_set_and_reorder(sets[perm[i]], image_set, set_size, Elt, 0);
	if (f_vv) {
		cout << "after map_a_set_and_reorder:" << endl;
		INT_vec_print(cout, image_set, set_size);
		cout << endl;
		for (j = 0; j < set_size; j++) {
			cout << j << " : " << image_set[j] << " : " << endl;
			A->print_point(image_set[j], cout);
			cout << endl;
			}
		}
	if (!vec_search((void **)sets, action_on_sets_compare_inverted, this, nb_sets, image_set, idx, verbose_level)) {
		INT u, a, b;
		cout << "action_on_sets::compute_image image set not found" << endl;
		cout << "action = " << A->label << endl;

		cout << "the element " << endl;
		A->print(cout, Elt);
		cout << endl;
		cout << "as permutation:" << endl;
		A->print_as_permutation(cout, Elt);
		cout << endl;

		cout << "i=" << i << endl;
		cout << "perm[i]=" << perm[i] << endl;
		cout << "sets[perm[i]]:" << endl;
		INT_vec_print_fully(cout, sets[perm[i]], set_size);
		cout << endl;
		cout << "image_set:" << endl;
		INT_vec_print_fully(cout, image_set, set_size);
		cout << endl;
		for (u = 0; u < nb_sets; u++) {
			cout << u << " : ";
			INT_vec_print(cout, sets[u], set_size);
			cout << endl;
			}
		for (u = 0; u < set_size; u++) {
			a = sets[perm[i]][u];
			b = A->image_of(Elt, a);
			cout << setw(3) << u << " : " << setw(3) << a << " : " << setw(3) << b << endl;
			}
		exit(1);
		}
	if (f_v) {
		cout << "action_on_sets::compute_image idx = " << idx << endl;
		}
	res = action_on_sets_compare(image_set, sets[idx], this);
	if (res != 0) {
		cout << "action_on_sets::compute_image the set we found is not the right one" << endl;
		}
	j = perm_inv[idx];
	if (f_v) {
		cout << "action_on_sets::compute_image j = perm_inv[idx] = " << j << endl;
		}
	if (j < 0 || j >= nb_sets) {
		cout << "action_on_sets::compute_image j=" << j << " out of range" << endl;
		exit(1);
		}
}

void action_on_sets::print_sets_sorted()
{
	INT i;
	
	cout << "the sets in the sorted ordering:" << endl;
	for (i = 0; i < nb_sets; i++) {
		cout << "set " << i << " : is " << perm_inv[i] << " : ";
		INT_vec_print(cout, sets[i], set_size);
		cout << endl;
		}
}

void action_on_sets::print_sets_in_original_ordering()
{
	INT i;
	
	cout << "the sets in the original ordering:" << endl;
	for (i = 0; i < nb_sets; i++) {
		cout << "set " << i << " : is " << perm[i] << " : ";
		INT_vec_print(cout, sets[perm[i]], set_size);
		cout << endl;
		}
}

void action_on_sets::test_sets()
{
	INT i, c;
	
	for (i = 0; i < nb_sets - 1; i++) {
		c = action_on_sets_compare(sets[i], sets[i + 1], this);
		if (c == 0) {
			cout << "action_on_sets::test_sets() sorted set " << i << " and sorted set " << i + 1 << " are equal. This should not be" << endl;
			cout << "The original set numbers are " << perm_inv[i] << " and " << perm_inv[i + 1] << " respectively." << endl;
			exit(1);
			}
		}
}



INT action_on_sets_compare(void *a, void *b, void *data)
{
	action_on_sets *AOS = (action_on_sets *) data;
	INT *A = (INT *)a;
	INT *B = (INT *)b;
	INT c;
	
	c = INT_vec_compare(A, B, AOS->set_size);
	return c;
}

INT action_on_sets_compare_inverted(void *a, void *b, void *data)
{
	action_on_sets *AOS = (action_on_sets *) data;
	INT *A = (INT *)a;
	INT *B = (INT *)b;
	INT c;
	
	c = INT_vec_compare(B, A, AOS->set_size);
	return c;
}


