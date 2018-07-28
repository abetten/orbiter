// action_on_subgroups.C
//
// Anton Betten
// April 29, 2017

#include "GALOIS/galois.h"
#include "action.h"

action_on_subgroups::action_on_subgroups()
{
	null();
}

action_on_subgroups::~action_on_subgroups()
{
	free();
}

void action_on_subgroups::null()
{
	A = NULL;
	S = NULL;
	Subgroups = NULL;
	sets = NULL;
	image_set = NULL;
	perm = NULL;
	perm_inv = NULL;
	Elt1 = NULL;
}

void action_on_subgroups::free()
{
	INT i;
	
	if (sets) {
		for (i = 0; i < nb_subgroups; i++) {
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
	if (Elt1) {
		FREE_INT(Elt1);
		}
	null();
}

void action_on_subgroups::init(action *A, sims *S, INT nb_subgroups, 
	INT subgroup_order, subgroup **Subgroups, INT verbose_level)
{
	INT i;
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE; //(verbose_level >= 5);
	
	if (f_v) {
		cout << "action_on_subgroups::init nb_subgroups=" << nb_subgroups << " subgroup_order=" << subgroup_order << endl;
		}
	action_on_subgroups::A = A;
	action_on_subgroups::S = S;
	action_on_subgroups::nb_subgroups = nb_subgroups;
	action_on_subgroups::subgroup_order = subgroup_order;
	action_on_subgroups::Subgroups = Subgroups;

	sets = NEW_PINT(nb_subgroups);
	image_set = NEW_INT(subgroup_order);
	perm = NEW_INT(nb_subgroups);
	perm_inv = NEW_INT(nb_subgroups);
	Elt1 = NEW_INT(A->elt_size_in_INT);
	
	for (i = 0; i < nb_subgroups; i++) {
		perm[i] = i;
		perm_inv[i] = i;
		}
	for (i = 0; i < nb_subgroups; i++) {
		sets[i] = NEW_INT(subgroup_order);
		INT_vec_copy(Subgroups[i]->Elements, sets[i], subgroup_order);
		INT_vec_quicksort_increasingly(sets[i], subgroup_order);
		if (f_vv) {
			cout << "set " << setw(3) << i << " is ";
			INT_vec_print(cout, sets[i], subgroup_order);
			cout << endl;
			}
		}
	quicksort_array_with_perm(nb_subgroups, (void **) sets, perm_inv, action_on_subgroups_compare, this);
	perm_inverse(perm_inv, perm, nb_subgroups);

	//test_sets();


	if (f_vv) {
		cout << "after quicksort_array_with_perm" << endl;
#if 0
		cout << "i : perm[i] : perm_inv[i]" << endl;
		for (i = 0; i < nb_sets; i++) {
			cout << i << " : " << perm[i] << " : " << perm_inv[i] << endl;
			}
#endif

		//print_sets_sorted();

		//print_sets_in_original_ordering();

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
		cout << "action_on_subgroups::init finished" << endl;
		}
}

INT action_on_subgroups::compute_image(INT *Elt, INT a, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT idx, res, j, b, aa, s, t;

	if (f_v) {
		cout << "action_on_subgroups::compute_image a = " << a << endl;
		}
	if (a < 0 || a >= nb_subgroups) {
		cout << "action_on_subgroups::compute_image a = " << a << " out of range" << endl;
		exit(1);
		}
	aa = perm[a];
	if (f_v) {
		cout << "action_on_subgroups::compute_image perm[a] = " << aa << endl;
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
		cout << "sets[perm[a]]:" << endl;
		INT_vec_print(cout, sets[aa], subgroup_order);
		cout << endl;
		for (j = 0; j < subgroup_order; j++) {
			cout << j << " : " << sets[aa][j] << " : " << endl;
			A->print_point(sets[aa][j], cout);
			cout << endl;
			}
		}

	//r = S->element_rank_INT(Elt);
	A->element_invert(Elt, Elt1, 0);

	for (j = 0; j < subgroup_order; j++) {
		s = sets[aa][j];
		t = S->conjugate_by_rank_b_bv_given(s, Elt, Elt1, 0 /* verbose_level */);
		//t = S->conjugate_by_rank(s, r, 0);
		image_set[j] = t;
		}
	INT_vec_heapsort(image_set, subgroup_order);
	
#if 0
	A->map_a_set_and_reorder(sets[perm[a]], image_set, set_size, Elt, 0);
#endif
	if (f_vv) {
		cout << "after map_a_set_and_reorder:" << endl;
		INT_vec_print(cout, image_set, subgroup_order);
		cout << endl;
		for (j = 0; j < subgroup_order; j++) {
			cout << j << " : " << image_set[j] << " : " << endl;
			A->print_point(image_set[j], cout);
			cout << endl;
			}
		}


	if (!vec_search((void **)sets, action_on_subgroups_compare_inverted, 
		this, nb_subgroups, image_set, idx, verbose_level)) {

		INT u;
		cout << "action_on_subgroups::compute_image image set not found" << endl;
		cout << "action = " << A->label << endl;

		cout << "the element " << endl;
		A->print(cout, Elt);
		cout << endl;
		cout << "as permutation:" << endl;
		A->print_as_permutation(cout, Elt);
		cout << endl;

		cout << "a=" << a << endl;
		cout << "perm[a]=" << aa << endl;
		cout << "sets[perm[a]]:" << endl;
		INT_vec_print_fully(cout, sets[aa], subgroup_order);
		cout << endl;
		cout << "image_set:" << endl;
		INT_vec_print_fully(cout, image_set, subgroup_order);
		cout << endl;
		for (u = 0; u < nb_subgroups; u++) {
			cout << u << " : ";
			INT_vec_print(cout, sets[u], subgroup_order);
			cout << endl;
			}
		for (u = 0; u < subgroup_order; u++) {
			s = sets[aa][u];
			t = A->image_of(Elt, s);
			cout << setw(3) << u << " : " << setw(3) << s << " : " << setw(3) << t << endl;
			}
		exit(1);
		}
	if (f_v) {
		cout << "action_on_subgroups::compute_image idx = " << idx << endl;
		}
	res = action_on_subgroups_compare(image_set, sets[idx], this);
	if (res != 0) {
		cout << "action_on_subgroups::compute_image the set we found is not the right one" << endl;
		}
	b = perm_inv[idx];
	if (f_v) {
		cout << "action_on_subgroups::compute_image b = perm_inv[idx] = " << b << endl;
		}
	if (b < 0 || b >= nb_subgroups) {
		cout << "action_on_subgroups::compute_image b=" << b << " out of range" << endl;
		exit(1);
		}
	return b;
}



INT action_on_subgroups_compare(void *a, void *b, void *data)
{
	action_on_subgroups *AOS = (action_on_subgroups *) data;
	INT *A = (INT *)a;
	INT *B = (INT *)b;
	INT c;
	
	c = INT_vec_compare(A, B, AOS->subgroup_order);
	return c;
}

INT action_on_subgroups_compare_inverted(void *a, void *b, void *data)
{
	action_on_subgroups *AOS = (action_on_subgroups *) data;
	INT *A = (INT *)a;
	INT *B = (INT *)b;
	INT c;
	
	c = INT_vec_compare(B, A, AOS->subgroup_order);
	return c;
}



