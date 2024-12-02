// action_on_subgroups.cpp
//
// Anton Betten
// April 29, 2017

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {

#if 0
static int action_on_subgroups_compare(
		void *a, void *b, void *data);
static int action_on_subgroups_compare_inverted(
		void *a, void *b, void *data);
#endif

action_on_subgroups::action_on_subgroups()
{
	Record_birth();
	A = NULL;
	S = NULL;
	Hash_table_subgroups = NULL;
#if 0
	nb_subgroups = 0;
	subgroup_order = 0;
	Subgroups = NULL;
	sets = NULL;
	perm = NULL;
	perm_inv = NULL;
#endif
	max_subgroup_order = 0;
	image_set = NULL;
	Elt1 = NULL;
}


action_on_subgroups::~action_on_subgroups()
{
	Record_death();
#if 0
	int i;
	
	if (sets) {
		for (i = 0; i < nb_subgroups; i++) {
			FREE_int(sets[i]);
		}
		FREE_pint(sets);
	}
#endif
	if (image_set) {
		FREE_int(image_set);
	}
#if 0
	if (perm) {
		FREE_int(perm);
	}
	if (perm_inv) {
		FREE_int(perm_inv);
	}
#endif
	if (Elt1) {
		FREE_int(Elt1);
	}
}

void action_on_subgroups::init(
		actions::action *A,
		groups::sims *S,
		data_structures_groups::hash_table_subgroups *Hash_table_subgroups,
	int verbose_level)
// Subgroups[nb_subgroups]
{
	int i;
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 5);
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "action_on_subgroups::init" << endl;
	}
	action_on_subgroups::A = A;
	action_on_subgroups::S = S;

	action_on_subgroups::Hash_table_subgroups = Hash_table_subgroups;

	max_subgroup_order = 0;
	for (i = 0; i < Hash_table_subgroups->nb_groups(); i++) {
		max_subgroup_order = MAX(max_subgroup_order, Hash_table_subgroups->get_subgroup(i)->group_order);
	}

	if (f_v) {
		cout << "action_on_subgroups::init max_subgroup_order = " << max_subgroup_order << endl;
	}
	image_set = NEW_int(max_subgroup_order);
	Elt1 = NEW_int(A->elt_size_in_int);

#if 0
	action_on_subgroups::nb_subgroups = nb_subgroups;
	action_on_subgroups::subgroup_order = subgroup_order;
	action_on_subgroups::Subgroups = Subgroups;

	sets = NEW_pint(nb_subgroups);
	image_set = NEW_int(subgroup_order);
	perm = NEW_int(nb_subgroups);
	perm_inv = NEW_int(nb_subgroups);
	
	for (i = 0; i < nb_subgroups; i++) {
		perm[i] = i;
		perm_inv[i] = i;
	}
	for (i = 0; i < nb_subgroups; i++) {
		sets[i] = NEW_int(subgroup_order);
		Int_vec_copy(Subgroups[i]->Elements, sets[i], subgroup_order);
		Sorting.int_vec_quicksort_increasingly(sets[i], subgroup_order);
		if (f_vv) {
			cout << "set " << setw(3) << i << " is ";
			Int_vec_print(cout, sets[i], subgroup_order);
			cout << endl;
		}
	}
	Sorting.quicksort_array_with_perm(nb_subgroups,
			(void **) sets, perm_inv, action_on_subgroups_compare,
			this);
	Combi.perm_inverse(perm_inv, perm, nb_subgroups);

	//test_sets();
#endif

#if 0
	if (f_vv) {
		cout << "after quicksort_array_with_perm" << endl;
		cout << "i : perm[i] : perm_inv[i]" << endl;
		for (i = 0; i < nb_sets; i++) {
			cout << i << " : " << perm[i] << " : " << perm_inv[i] << endl;
		}

		//print_sets_sorted();

		//print_sets_in_original_ordering();

		cout << "the sets in the perm_inv ordering:" << endl;
		for (i = 0; i < nb_sets; i++) {
			j = perm_inv[i];
			cout << "set " << i << " is set " << j << " : ";
			int_vec_print(cout, sets[j], set_size);
			cout << endl;
		}
	}
#endif

	if (f_v) {
		cout << "action_on_subgroups::init finished" << endl;
	}
}

long int action_on_subgroups::compute_image(
		int *Elt, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//long int res, j, b, aa, s, t;
	//int idx;
	//data_structures::sorting Sorting;

	if (f_v) {
		cout << "action_on_subgroups::compute_image "
				"a = " << a << endl;
	}
	if (a < 0 || a >= Hash_table_subgroups->nb_groups()) {
		cout << "action_on_subgroups::compute_image "
				"a = " << a << " out of range" << endl;
		exit(1);
	}
#if 0
	aa = perm[a];
	if (f_v) {
		cout << "action_on_subgroups::compute_image "
				"perm[a] = " << aa << endl;
	}
	if (f_vv) {
		cout << "the element " << endl;
		A->Group_element->print(cout, Elt);
		cout << endl;
		cout << "as permutation:" << endl;
		A->Group_element->print_as_permutation(cout, Elt);
		cout << endl;
	}
	if (f_vv) {
		cout << "sets[perm[a]]:" << endl;
		Int_vec_print(cout, sets[aa], subgroup_order);
		cout << endl;
		for (j = 0; j < subgroup_order; j++) {
			cout << j << " : " << sets[aa][j] << " : " << endl;
			A->Group_element->print_point(sets[aa][j], cout);
			cout << endl;
		}
	}
#endif

	if (f_v) {
		cout << "action_on_subgroups::compute_image "
				"group order = " << Hash_table_subgroups->get_subgroup(a)->group_order << endl;
	}


#if 0
	//r = S->element_rank_int(Elt);
	if (f_v) {
		cout << "action_on_subgroups::compute_image "
				"before element_invert" << endl;
	}
	A->Group_element->element_invert(Elt, Elt1, verbose_level - 1);
	if (f_v) {
		cout << "action_on_subgroups::compute_image "
				"after element_invert" << endl;
	}

	int j;
	long int s, t, b;

	for (j = 0; j < Hash_table_subgroups->get_subgroup(a)->group_order; j++) {
		if (f_v) {
			cout << "action_on_subgroups::compute_image "
					"j = " << j << " / " << Hash_table_subgroups->get_subgroup(a)->group_order << endl;
		}
		s = Hash_table_subgroups->get_subgroup(a)->Elements[j];
		//s = sets[aa][j];
		if (f_v) {
			cout << "action_on_subgroups::compute_image "
					"j = " << j << " / " << Hash_table_subgroups->get_subgroup(a)->group_order
					<< " before S->conjugate_by_rank_b_bv_given" << endl;
		}
		t = S->conjugate_by_rank_b_bv_given(s, Elt, Elt1,
				0 /* verbose_level */);
		//t = S->conjugate_by_rank(s, r, 0);
		if (f_v) {
			cout << "action_on_subgroups::compute_image "
					<< s << " -> " << t << endl;
		}
		image_set[j] = t;
	}
	

	Sorting.int_vec_heapsort(image_set, Hash_table_subgroups->get_subgroup(a)->group_order);
#else
	S->conjugate_numerical_set(
			Hash_table_subgroups->get_subgroup(a)->Elements,
			Hash_table_subgroups->get_subgroup(a)->group_order,
			Elt, image_set,
			verbose_level - 2);

#endif
	int f_found;
	int pos;
	uint32_t hash;

	f_found = Hash_table_subgroups->find_subgroup_direct(
			image_set, Hash_table_subgroups->get_subgroup(a)->group_order,
			pos, hash, verbose_level - 2);



	if (!f_found) {
		cout << "action_on_subgroups::compute_image "
				"a = " << a << " image not found" << endl;
		exit(1);
	}

	long int b;

	b = pos;

#if 0
	A->map_a_set_and_reorder(sets[perm[a]], image_set, set_size, Elt, 0);
#endif

#if 0
	if (f_vv) {
		cout << "after map_a_set_and_reorder:" << endl;
		Int_vec_print(cout, image_set, subgroup_order);
		cout << endl;
		for (j = 0; j < subgroup_order; j++) {
			cout << j << " : " << image_set[j] << " : " << endl;
			A->Group_element->print_point(image_set[j], cout);
			cout << endl;
		}
	}


	if (!Sorting.vec_search(
			(void **)sets, action_on_subgroups_compare_inverted,
		this, nb_subgroups, image_set, idx,
		verbose_level)) {

		int u;

		cout << "action_on_subgroups::compute_image "
				"image set not found" << endl;
		cout << "action = " << A->label << endl;

		cout << "the element " << endl;
		A->Group_element->print(cout, Elt);
		cout << endl;

		cout << "as permutation:" << endl;
		A->Group_element->print_as_permutation(cout, Elt);
		cout << endl;

		cout << "a=" << a << endl;
		cout << "perm[a]=" << aa << endl;

		cout << "sets[perm[a]]:" << endl;
		Int_vec_print_fully(cout, sets[aa], subgroup_order);
		cout << endl;

		cout << "image_set:" << endl;
		Int_vec_print_fully(cout, image_set, subgroup_order);
		cout << endl;

		for (u = 0; u < nb_subgroups; u++) {
			cout << u << " : ";
			Int_vec_print(cout, sets[u], subgroup_order);
			cout << endl;
		}

		for (u = 0; u < subgroup_order; u++) {
			s = sets[aa][u];
			t = A->Group_element->image_of(Elt, s);
			cout << setw(3) << u << " : " << setw(3) << s
					<< " : " << setw(3) << t << endl;
		}
		exit(1);
	}
	if (f_v) {
		cout << "action_on_subgroups::compute_image "
				"idx = " << idx << endl;
	}
	res = action_on_subgroups_compare(image_set, sets[idx], this);
	if (res != 0) {
		cout << "action_on_subgroups::compute_image "
				"the set we found is not the right one" << endl;
	}
	b = perm_inv[idx];
	if (f_v) {
		cout << "action_on_subgroups::compute_image "
				"b = perm_inv[idx] = " << b << endl;
	}
	if (b < 0 || b >= nb_subgroups) {
		cout << "action_on_subgroups::compute_image "
				"b=" << b << " out of range" << endl;
		exit(1);
	}
#endif

	return b;
}


#if 0
static int action_on_subgroups_compare(
		void *a, void *b, void *data)
{
	action_on_subgroups *AOS = (action_on_subgroups *) data;
	int *A = (int *)a;
	int *B = (int *)b;
	int c;
	data_structures::sorting Sorting;
	
	c = Sorting.int_vec_compare(A, B, AOS->subgroup_order);
	return c;
}

static int action_on_subgroups_compare_inverted(
		void *a, void *b, void *data)
{
	action_on_subgroups *AOS = (action_on_subgroups *) data;
	int *A = (int *)a;
	int *B = (int *)b;
	int c;
	data_structures::sorting Sorting;
	
	c = Sorting.int_vec_compare(B, A, AOS->subgroup_order);
	return c;
}
#endif


}}}

