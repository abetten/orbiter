// action_on_spread_set.C
//
// Anton Betten
// October 9, 2013

#include "foundations/foundations.h"
#include "group_actions.h"

namespace orbiter {

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
		FREE_int(mtx1);
		}
	if (mtx2) {
		FREE_int(mtx2);
		}
	if (Elt1) {
		FREE_int(Elt1);
		}
	if (Elt2) {
		FREE_int(Elt2);
		}
	if (subspace1) {
		FREE_int(subspace1);
		}
	if (subspace2) {
		FREE_int(subspace2);
		}
	null();
}

void action_on_spread_set::init(action *A_PGL_n_q, action *A_PGL_k_q, sims *G_PGL_k_q, 
	int k, finite_field *F, int verbose_level)
// we are acting on the elements of G_PGL_k_q, so the degree of the action 
// is the order of this group.
// A_PGL_k_q in only needed for make_element
{
	int f_v = (verbose_level >= 1);
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
	degree = go.as_int();
	if (f_v) {
		cout << "action_on_spread_set::init the order of the group of matrices is " << degree << endl;
		}

	Elt1 = NEW_int(A_PGL_k_q->elt_size_in_int);
	Elt2 = NEW_int(A_PGL_k_q->elt_size_in_int);
	

	mtx1 = NEW_int(k * k);
	mtx2 = NEW_int(k * k);
	subspace1 = NEW_int(k * n);
	subspace2 = NEW_int(k * n);
	
	if (f_v) {
		cout << "degree = " << degree << endl;
		cout << "low_level_point_size = " << low_level_point_size << endl;
		}
	
	if (f_v) {
		cout << "action_on_spread_set::init done" << endl;
		}
}

int action_on_spread_set::compute_image_int(int *Elt, int rk, int verbose_level)
{
	//verbose_level = 2;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, rk2;
	
	if (f_v) {
		cout << "action_on_spread_set::compute_image_int rk = " << rk << endl;
		}

	unrank_point(rk, mtx1, verbose_level - 1);
	matrix_to_subspace(mtx1, subspace1, verbose_level);

	if (f_vv) {
		cout << "action_on_spread_set::compute_image_int after unrank_point" << endl;
		print_integer_matrix_width(cout, subspace1, k, n, n, F->log10_of_q);
		cout << "action_on_spread_set::compute_image_int group element:" << endl;
		int_matrix_print(Elt, n, n);
		}

	for (i = 0; i < k; i++) {
		A_PGL_n_q->element_image_of_low_level(
			subspace1 + i * n, subspace2 + i * n, Elt, verbose_level - 1);
		}

	if (f_vv) {
		cout << "action_on_spread_set::compute_image_int after applying group element" << endl;
		print_integer_matrix_width(cout, subspace2, k, n, n, F->log10_of_q);
		}

	subspace_to_matrix(subspace2, mtx2, verbose_level - 1);
	rk2 = rank_point(mtx2, verbose_level - 1);

	if (f_v) {
		cout << "action_on_spread_set::compute_image_int image of " << rk << " is " << rk2 << endl;
		}
	return rk2;
}

void action_on_spread_set::matrix_to_subspace(int *mtx, int *subspace, int verbose_level)
{
	int i, j;
	
	int_vec_zero(subspace, k * n);
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

void action_on_spread_set::subspace_to_matrix(int *subspace, int *mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, r;
	
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

void action_on_spread_set::unrank_point(int rk, int *mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_spread_set::unrank_point rk = " << rk << endl;
		}
	G_PGL_k_q->element_unrank_int(rk, Elt1);
	int_vec_copy(Elt1, mtx, k * k);
	if (f_v) {
		cout << "action_on_spread_set::unrank_point done" << endl;
		}
}

int action_on_spread_set::rank_point(int *mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk;
	
	if (f_v) {
		cout << "action_on_spread_set::rank_point" << endl;
		}
	A_PGL_k_q->make_element(Elt2, mtx, 0 /* verbose_level */);

	rk = G_PGL_k_q->element_rank_int(Elt2);
	if (f_v) {
		cout << "action_on_spread_set::rank_point done, rk = " << rk << endl;
		}
	return rk;
}

void action_on_spread_set::compute_image_low_level(int *Elt, int *input, int *output, int verbose_level)
{
	//verbose_level = 2;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	
	if (f_v) {
		cout << "action_on_spread_set::compute_image_low_level" << endl;
		}
	if (f_vv) {
		cout << "action_on_spread_set::compute_image_low_level input=" << endl;
		int_matrix_print(input, k, k);
		cout << "action_on_spread_set::compute_image_low_level matrix=" << endl;
		int_matrix_print(Elt, n, n);
		}

	matrix_to_subspace(input, subspace1, verbose_level- 1);


	for (i = 0; i < k; i++) {
		A_PGL_n_q->element_image_of_low_level(
			subspace1 + i * n, subspace2 + i * n, Elt, verbose_level - 2);
		}
	if (f_vv) {
		cout << "action_on_spread_set::compute_image_low_level after mult=" << endl;
		int_matrix_print(subspace2, k, n);
		}

	subspace_to_matrix(subspace2, output, verbose_level - 1);

	if (f_vv) {
		cout << "action_on_spread_set::compute_image_low_level output=" << endl;
		int_matrix_print(output, k, k);
		}

	if (f_v) {
		cout << "action_on_spread_set::compute_image_low_level done" << endl;
		}
}

}

