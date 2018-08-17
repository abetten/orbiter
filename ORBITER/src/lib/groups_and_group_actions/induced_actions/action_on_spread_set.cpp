// action_on_spread_set.C
//
// Anton Betten
// October 9, 2013

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

action_on_spread_set::action_on_spread_set()
{
	null();
}

action_on_spread_set::~action_on_spread_set()
{
	free();
}

void action_on_spread_set::null()
{
	mtx1 = NULL;
	mtx2 = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	subspace1 = NULL;
	subspace2 = NULL;
}

void action_on_spread_set::free()
{
	if (mtx1) {
		FREE_INT(mtx1);
		}
	if (mtx2) {
		FREE_INT(mtx2);
		}
	if (Elt1) {
		FREE_INT(Elt1);
		}
	if (Elt2) {
		FREE_INT(Elt2);
		}
	if (subspace1) {
		FREE_INT(subspace1);
		}
	if (subspace2) {
		FREE_INT(subspace2);
		}
	null();
}

void action_on_spread_set::init(action *A_PGL_n_q, action *A_PGL_k_q, sims *G_PGL_k_q, 
	INT k, finite_field *F, INT verbose_level)
// we are acting on the elements of G_PGL_k_q, so the degree of the action 
// is the order of this group.
// A_PGL_k_q in only needed for make_element
{
	INT f_v = (verbose_level >= 1);
	longinteger_object go;
	
	if (f_v) {
		cout << "action_on_spread_set::init" << endl;
		}
	action_on_spread_set::k = k;
	action_on_spread_set::F = F;
	action_on_spread_set::q = F->q;
	action_on_spread_set::A_PGL_n_q = A_PGL_n_q;
	action_on_spread_set::A_PGL_k_q = A_PGL_k_q;
	action_on_spread_set::G_PGL_k_q = G_PGL_k_q;
	n = 2 * k;
	k2 = k * k;
	low_level_point_size = k2;

	if (f_v) {
		cout << "action_on_spread_set::init" << endl;
		cout << "k=" << k << endl;
		cout << "n=" << n << endl;
		cout << "q=" << q << endl;
		cout << "low_level_point_size=" << low_level_point_size << endl;
		}
	
	G_PGL_k_q->group_order(go);
	degree = go.as_INT();
	if (f_v) {
		cout << "action_on_spread_set::init the order of the group of matrices is " << degree << endl;
		}

	Elt1 = NEW_INT(A_PGL_k_q->elt_size_in_INT);
	Elt2 = NEW_INT(A_PGL_k_q->elt_size_in_INT);
	

	mtx1 = NEW_INT(k * k);
	mtx2 = NEW_INT(k * k);
	subspace1 = NEW_INT(k * n);
	subspace2 = NEW_INT(k * n);
	
	if (f_v) {
		cout << "degree = " << degree << endl;
		cout << "low_level_point_size = " << low_level_point_size << endl;
		}
	
	if (f_v) {
		cout << "action_on_spread_set::init done" << endl;
		}
}

INT action_on_spread_set::compute_image_INT(INT *Elt, INT rk, INT verbose_level)
{
	//verbose_level = 2;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, rk2;
	
	if (f_v) {
		cout << "action_on_spread_set::compute_image_INT rk = " << rk << endl;
		}

	unrank_point(rk, mtx1, verbose_level - 1);
	matrix_to_subspace(mtx1, subspace1, verbose_level);

	if (f_vv) {
		cout << "action_on_spread_set::compute_image_INT after unrank_point" << endl;
		print_integer_matrix_width(cout, subspace1, k, n, n, F->log10_of_q);
		cout << "action_on_spread_set::compute_image_INT group element:" << endl;
		INT_matrix_print(Elt, n, n);
		}

	for (i = 0; i < k; i++) {
		A_PGL_n_q->element_image_of_low_level(
			subspace1 + i * n, subspace2 + i * n, Elt, verbose_level - 1);
		}

	if (f_vv) {
		cout << "action_on_spread_set::compute_image_INT after applying group element" << endl;
		print_integer_matrix_width(cout, subspace2, k, n, n, F->log10_of_q);
		}

	subspace_to_matrix(subspace2, mtx2, verbose_level - 1);
	rk2 = rank_point(mtx2, verbose_level - 1);

	if (f_v) {
		cout << "action_on_spread_set::compute_image_INT image of " << rk << " is " << rk2 << endl;
		}
	return rk2;
}

void action_on_spread_set::matrix_to_subspace(INT *mtx, INT *subspace, INT verbose_level)
{
	INT i, j;
	
	INT_vec_zero(subspace, k * n);
#if 0
	for (i = 0; i < k * n; i++) {
		subspace[i] = 0;
		}
#endif
	for (i = 0; i < k; i++) {
		subspace[i * n + i] = 1;
		for (j = 0; j < k; j++) {
			subspace[i * n + k + j] = mtx[i * k + j];
			}
		}
}

void action_on_spread_set::subspace_to_matrix(INT *subspace, INT *mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, r;
	
	if (f_v) {
		cout << "action_on_spread_set::subspace_to_matrix" << endl;
		}
	
	r = F->Gauss_easy(subspace, k, n);
	if (r != k) {
		cout << "action_on_spread_set::subspace_to_matrix r != k" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "action_on_spread_set::subspace_to_matrix after Gauss_easy" << endl;
		}
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			mtx[i * k + j] = subspace[i * n + k + j];
			}
		}
	if (f_v) {
		cout << "action_on_spread_set::subspace_to_matrix done" << endl;
		}
}

void action_on_spread_set::unrank_point(INT rk, INT *mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_spread_set::unrank_point rk = " << rk << endl;
		}
	G_PGL_k_q->element_unrank_INT(rk, Elt1);
	INT_vec_copy(Elt1, mtx, k * k);
	if (f_v) {
		cout << "action_on_spread_set::unrank_point done" << endl;
		}
}

INT action_on_spread_set::rank_point(INT *mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rk;
	
	if (f_v) {
		cout << "action_on_spread_set::rank_point" << endl;
		}
	A_PGL_k_q->make_element(Elt2, mtx, 0 /* verbose_level */);

	rk = G_PGL_k_q->element_rank_INT(Elt2);
	if (f_v) {
		cout << "action_on_spread_set::rank_point done, rk = " << rk << endl;
		}
	return rk;
}

void action_on_spread_set::compute_image_low_level(INT *Elt, INT *input, INT *output, INT verbose_level)
{
	//verbose_level = 2;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i;
	
	if (f_v) {
		cout << "action_on_spread_set::compute_image_low_level" << endl;
		}
	if (f_vv) {
		cout << "action_on_spread_set::compute_image_low_level input=" << endl;
		INT_matrix_print(input, k, k);
		cout << "action_on_spread_set::compute_image_low_level matrix=" << endl;
		INT_matrix_print(Elt, n, n);
		}

	matrix_to_subspace(input, subspace1, verbose_level- 1);


	for (i = 0; i < k; i++) {
		A_PGL_n_q->element_image_of_low_level(
			subspace1 + i * n, subspace2 + i * n, Elt, verbose_level - 2);
		}
	if (f_vv) {
		cout << "action_on_spread_set::compute_image_low_level after mult=" << endl;
		INT_matrix_print(subspace2, k, n);
		}

	subspace_to_matrix(subspace2, output, verbose_level - 1);

	if (f_vv) {
		cout << "action_on_spread_set::compute_image_low_level output=" << endl;
		INT_matrix_print(output, k, k);
		}

	if (f_v) {
		cout << "action_on_spread_set::compute_image_low_level done" << endl;
		}
}


