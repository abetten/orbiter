// union_find.C
//
// Anton Betten
// February 2, 2010

#include "galois.h"
#include "action.h"

INT union_find::cntr_new = 0;
INT union_find::cntr_objects = 0;
INT union_find::f_debug_memory = FALSE;




void *union_find::operator new(size_t bytes)
{
	cntr_new++;
	cntr_objects++;
	if (f_debug_memory) {
		cout << "union_find::operator new bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void *union_find::operator new[](size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(union_find);
	cntr_new++;
	cntr_objects += n;
	if (f_debug_memory) {
		cout << "union_find::operator new[] n=" << n 
			<< " bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void union_find::operator delete(void *ptr, size_t bytes)
{
	if (f_debug_memory) {
		cout << "union_find::operator delete bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects--;
	return free(ptr);
}

void union_find::operator delete[](void *ptr, size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(union_find);
	if (f_debug_memory) {
		cout << "union_find::operator delete[] n=" << n 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects -= n;
	return free(ptr);
}

union_find::union_find()
{
	A = NULL;
	prev = NULL;
}

union_find::~union_find()
{
	freeself();
};

void union_find::freeself()
{
	if (prev) {
		FREE_INT(prev);
		}
	null();
}

void union_find::null()
{
	prev = NULL;
}

void union_find::init(action *A, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "union_find::init action=" << A->label << endl;
		}
	freeself();
	union_find::A = A;
	prev = NEW_INT(A->degree);
	for (i = 0; i < A->degree; i++) {
		prev[i] = i;
		}
}

INT union_find::ancestor(INT i)
{
	INT j;
	
	while ((j = prev[i]) != i) {
		i = j;
		}
	return i;
}

INT union_find::count_ancestors()
{
	INT i, nb;

	nb = 0;
	for (i = 0; i < A->degree; i++) {
		if (prev[i] == i)
			nb++;
		}
	return nb;
	
}

INT union_find::count_ancestors_above(INT i0)
{
	INT i, nb;

	nb = 0;
	for (i = i0; i < A->degree; i++) {
		if (prev[i] == i)
			nb++;
		}
	return nb;
	
}

void union_find::do_union(INT a, INT b)
{
	INT A, B;

	A = ancestor(a);
	B = ancestor(b);
	if (A == B)
		return;
	if (A < B) {
		prev[a] = A;
		prev[b] = A;
		}
	else {
		prev[a] = B;
		prev[b] = B;
		}
}

void union_find::print()
{
	INT i, j;
	
	cout << "i : ancestor(i) : prev[i]" << endl;
	for (i = 0; i < A->degree; i++) {
		j = ancestor(i);
		cout << setw(4) << i << " : " << setw(4) << j << " : " << setw(4) << i << endl;
		}
}
	
void union_find::add_generators(vector_ge *gens, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i;

	if (f_v) {
		cout << "union_find::add_generators" << endl;
		}
	if (f_vv) {
		cout << "union_find::add_generators before:" << endl;
		print();
		}
	for (i = 0; i < gens->len; i++) {
		if (f_vv) {
			cout << "union_find::add_generators adding generator " << i << endl;
			}
		add_generator(gens->ith(i), verbose_level - 2);
		}
	if (f_vv) {
		cout << "union_find::add_generators after:" << endl;
		print();
		}
	if (f_v) {
		cout << "union_find::add_generators done" << endl;
		}
}

void union_find::add_generator(INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT *f_seen;
	INT i, i0, j, k;
	INT cnt = 0;

	if (f_v) {
		cout << "union_find::add_generator" << endl;
		}
	if (f_vv) {
		A->print_quick(cout, Elt);
		cout << "as permutation of degree " << A->degree << " (skipped)" << endl;
		//A->print_as_permutation(cout, Elt);
		}
	if (f_vv) {
		cout << "union_find::add_generator degree=" << A->degree << endl;
		print();
		}
	f_seen = NEW_INT(A->degree);
	for (i = 0; i < A->degree; i++) {
		f_seen[i] = FALSE;
		}

	for (i = 0; i < A->degree; i++) {
		if (f_seen[i])
			continue;
		i0 = i;
		f_seen[i0] = TRUE;
		j = i0;

		if (f_vv) {
			cout << "union_find::add_generator i0=" << i0 << endl;
			//print();
			}
		
		while (TRUE) {
			cnt++;
			if (cnt > A->degree) {
				cout << "union_find::add_generator too many iterations" << endl;
				exit(1);
				}
			k = A->element_image_of(j, Elt, 0);
			if (f_vv) {
				cout << "union_find::add_generator i0=" << i0 << " j=" << j << " k=" << k << endl;
				}
			if (k == i0)
				break;
			f_seen[k] = TRUE;
			do_union(i0, k);
			j = k;
			}
		}
	FREE_INT(f_seen);
	if (f_vv) {
		cout << "union_find::add_generator after:" << endl;
		print();
		}
	if (f_v) {
		cout << "union_find::add_generator done" << endl;
		}
}



