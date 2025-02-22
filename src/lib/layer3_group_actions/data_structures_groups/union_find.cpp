// union_find.cpp
//
// Anton Betten
// February 2, 2010

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {


union_find::union_find()
{
	Record_birth();
	A = NULL;
	prev = NULL;
}

union_find::~union_find()
{
	Record_death();
	if (prev) {
		FREE_int(prev);
	}
}

void union_find::init(
		actions::action *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "union_find::init action=" << A->label << endl;
	}
	union_find::A = A;
	prev = NEW_int(A->degree);
	for (i = 0; i < A->degree; i++) {
		prev[i] = i;
	}
}

int union_find::ancestor(
		int i)
{
	int j;
	
	while ((j = prev[i]) != i) {
		i = j;
	}
	return i;
}

int union_find::count_ancestors()
{
	int i, nb;

	nb = 0;
	for (i = 0; i < A->degree; i++) {
		if (prev[i] == i) {
			nb++;
		}
	}
	return nb;
	
}

int union_find::count_ancestors_above(
		int i0)
{
	int i, nb;

	nb = 0;
	for (i = i0; i < A->degree; i++) {
		if (prev[i] == i) {
			nb++;
		}
	}
	return nb;
	
}

void union_find::do_union(
		int a, int b)
{
	int A, B;

	A = ancestor(a);
	B = ancestor(b);
	if (A == B) {
		return;
	}
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
	int i, j;
	
	cout << "i : ancestor(i) : prev[i]" << endl;
	for (i = 0; i < A->degree; i++) {
		j = ancestor(i);
		cout << setw(4) << i << " : " << setw(4)
				<< j << " : " << setw(4) << i << endl;
	}
}
	
void union_find::add_generators(
		vector_ge *gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "union_find::add_generators" << endl;
	}
	if (f_vv) {
		cout << "union_find::add_generators before:" << endl;
		print();
	}
	for (i = 0; i < gens->len; i++) {
		if (f_vv) {
			cout << "union_find::add_generators "
					"adding generator " << i << endl;
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

void union_find::add_generator(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int *f_seen;
	int i, i0, j, k;
	int cnt = 0;

	if (f_v) {
		cout << "union_find::add_generator" << endl;
	}
	if (f_vv) {
		A->Group_element->print_quick(cout, Elt);
		cout << "as permutation of degree "
				<< A->degree << " (skipped)" << endl;
		//A->print_as_permutation(cout, Elt);
	}
	if (f_vv) {
		cout << "union_find::add_generator degree=" << A->degree << endl;
		print();
	}
	f_seen = NEW_int(A->degree);
	for (i = 0; i < A->degree; i++) {
		f_seen[i] = false;
	}

	for (i = 0; i < A->degree; i++) {
		if (f_seen[i]) {
			continue;
		}
		i0 = i;
		f_seen[i0] = true;
		j = i0;

		if (f_vv) {
			cout << "union_find::add_generator i0=" << i0 << endl;
			//print();
		}
		
		while (true) {
			cnt++;
			if (cnt > A->degree) {
				cout << "union_find::add_generator "
						"too many iterations" << endl;
				exit(1);
			}
			k = A->Group_element->element_image_of(j, Elt, 0);
			if (f_vv) {
				cout << "union_find::add_generator "
						"i0=" << i0 << " j=" << j << " k=" << k << endl;
			}
			if (k == i0) {
				break;
			}
			f_seen[k] = true;
			do_union(i0, k);
			j = k;
		}
	}
	FREE_int(f_seen);
	if (f_vv) {
		cout << "union_find::add_generator after:" << endl;
		print();
	}
	if (f_v) {
		cout << "union_find::add_generator done" << endl;
	}
}



}}}


