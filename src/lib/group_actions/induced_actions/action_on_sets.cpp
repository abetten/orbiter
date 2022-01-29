// action_on_sets.cpp
//
// Anton Betten
// November 13, 2007

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


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
	int i;
	
	if (sets) {
		for (i = 0; i < nb_sets; i++) {
			FREE_lint(sets[i]);
			}
		FREE_plint(sets);
		}
	if (image_set) {
		FREE_lint(image_set);
		}
	if (perm) {
		FREE_int(perm);
		}
	if (perm_inv) {
		FREE_int(perm_inv);
		}
	null();
}


void action_on_sets::init(int nb_sets,
		int set_size, long int *input_sets,
		int verbose_level)
{
	int i, j;
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 5);
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "action_on_sets::init "
				"nb_sets=" << nb_sets
				<< " set_size=" << set_size << endl;
		}
	action_on_sets::nb_sets = nb_sets;
	action_on_sets::set_size = set_size;
	sets = NEW_plint(nb_sets);
	image_set = NEW_lint(set_size);
	perm = NEW_int(nb_sets);
	perm_inv = NEW_int(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		perm[i] = i;
		perm_inv[i] = i;
		}
	for (i = 0; i < nb_sets; i++) {
		sets[i] = NEW_lint(set_size);
		for (j = 0; j < set_size; j++) {
			sets[i][j] = input_sets[i * set_size + j];
			}
		Sorting.lint_vec_quicksort_increasingly(sets[i], set_size);
		if (f_vv) {
			cout << "set " << setw(3) << i << " is ";
			Lint_vec_print(cout, sets[i], set_size);
			cout << endl;
			}
		}
	Sorting.quicksort_array_with_perm(nb_sets,
			(void **) sets, perm_inv,
			action_on_sets_compare,
			this);
	Combi.perm_inverse(perm_inv, perm, nb_sets);

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
			int_vec_print(cout, sets[j], set_size);
			cout << endl;
			}
#endif
		}
	if (f_v) {
		cout << "action_on_sets::init finished" << endl;
		}
}

int action_on_sets::find_set(long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;
	int idx, j;

	if (f_v) {
		cout << "action_on_sets::find_set, nb_sets=" << nb_sets << endl;
	}
	if (!Sorting.vec_search(
			(void **)sets,
			action_on_sets_compare_inverted,
			this,
			nb_sets,
			set,
			idx,
			verbose_level)) {
		cout << "action_on_sets::find_set could not find the given set" << endl;
		exit(1);
	}
	j = perm_inv[idx];
	if (f_v) {
		cout << "action_on_sets::find_set done" << endl;
	}
	return j;
}

long int action_on_sets::compute_image(actions::action *A,
		int *Elt, long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int idx, res;
	long int j;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "action_on_sets::compute_image "
				"i = " << i << endl;
		cout << "action_on_sets::compute_image "
				"perm[i] = " << perm[i] << endl;
		}
	if (i < 0 || i >= nb_sets) {
		cout << "action_on_sets::compute_image "
				"i = " << i << " out of range" << endl;
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
		Lint_vec_print(cout, sets[perm[i]], set_size);
		cout << endl;
		for (j = 0; j < set_size; j++) {
			cout << j << " : " << sets[perm[i]][j] << " : " << endl;
			A->print_point(sets[perm[i]][j], cout);
			cout << endl;
			}
		}
	A->map_a_set_and_reorder(
			sets[perm[i]],
			image_set,
			set_size,
			Elt,
			0);
	if (f_vv) {
		cout << "after map_a_set_and_reorder:" << endl;
		Lint_vec_print(cout, image_set, set_size);
		cout << endl;
		for (j = 0; j < set_size; j++) {
			cout << j << " : " << image_set[j] << " : " << endl;
			A->print_point(image_set[j], cout);
			cout << endl;
			}
		}
	if (!Sorting.vec_search(
			(void **)sets,
			action_on_sets_compare_inverted,
			this,
			nb_sets,
			image_set,
			idx,
			verbose_level)) {
		int u, a, b;
		cout << "action_on_sets::compute_image "
				"image set not found" << endl;
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
		Lint_vec_print_fully(cout, sets[perm[i]], set_size);
		cout << endl;
		cout << "image_set:" << endl;
		Lint_vec_print_fully(cout, image_set, set_size);
		cout << endl;
		for (u = 0; u < nb_sets; u++) {
			cout << u << " : ";
			Lint_vec_print(cout, sets[u], set_size);
			cout << endl;
			}
		for (u = 0; u < set_size; u++) {
			a = sets[perm[i]][u];
			b = A->image_of(Elt, a);
			cout << setw(3) << u << " : " << setw(3) << a
					<< " : " << setw(3) << b << endl;
			}
		exit(1);
		}
	if (f_v) {
		cout << "action_on_sets::compute_image "
				"idx = " << idx << endl;
		}
	res = action_on_sets_compare(image_set, sets[idx], this);
	if (res != 0) {
		cout << "action_on_sets::compute_image "
				"the set we found is not the right one" << endl;
		}
	j = perm_inv[idx];
	if (f_v) {
		cout << "action_on_sets::compute_image "
				"j = perm_inv[idx] = " << j << endl;
		}
	if (j < 0 || j >= nb_sets) {
		cout << "action_on_sets::compute_image "
				"j=" << j << " out of range" << endl;
		exit(1);
		}
	return j;
}

void action_on_sets::print_sets_sorted()
{
	int i;
	
	cout << "the sets in the sorted ordering:" << endl;
	for (i = 0; i < nb_sets; i++) {
		cout << "set " << i << " : is " << perm_inv[i] << " : ";
		Lint_vec_print(cout, sets[i], set_size);
		cout << endl;
		}
}

void action_on_sets::print_sets_in_original_ordering()
{
	int i;
	
	cout << "the sets in the original ordering:" << endl;
	for (i = 0; i < nb_sets; i++) {
		cout << "set " << i << " : is " << perm[i] << " : ";
		Lint_vec_print(cout, sets[perm[i]], set_size);
		cout << endl;
		}
}

void action_on_sets::test_sets()
{
	int i, c;
	
	for (i = 0; i < nb_sets - 1; i++) {
		c = action_on_sets_compare(sets[i], sets[i + 1], this);
		if (c == 0) {
			cout << "action_on_sets::test_sets "
					"sorted set " << i << " and sorted set "
					<< i + 1 << " are equal. This should not be" << endl;
			cout << "The original set numbers are " << perm_inv[i]
				<< " and " << perm_inv[i + 1] << " respectively." << endl;
			exit(1);
			}
		}
}



int action_on_sets_compare(void *a, void *b, void *data)
{
	action_on_sets *AOS = (action_on_sets *) data;
	long int *A = (long int *)a;
	long int *B = (long int *)b;
	int c;
	data_structures::sorting Sorting;
	
	c = Sorting.lint_vec_compare(A, B, AOS->set_size);
	return c;
}

int action_on_sets_compare_inverted(void *a, void *b, void *data)
{
	action_on_sets *AOS = (action_on_sets *) data;
	long int *A = (long int *)a;
	long int *B = (long int *)b;
	int c;
	data_structures::sorting Sorting;
	
	c = Sorting.lint_vec_compare(B, A, AOS->set_size);
	return c;
}

}}}



