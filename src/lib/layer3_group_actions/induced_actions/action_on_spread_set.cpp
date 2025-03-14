// action_on_spread_set.cpp
//
// Anton Betten
// October 9, 2013

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;

namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_spread_set::action_on_spread_set()
{
	Record_birth();
	k = n = k2 = q = 0;
	F = NULL;
	low_level_point_size = 0;
	degree = 0;

	A_PGL_n_q = NULL;
	A_PGL_k_q = NULL;
	G_PGL_k_q = NULL;

	Elt1 = NULL;
	Elt2 = NULL;

	mtx1 = NULL;
	mtx2 = NULL;
	subspace1 = NULL;
	subspace2 = NULL;
}

action_on_spread_set::~action_on_spread_set()
{
	Record_death();
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
}

void action_on_spread_set::init(
		actions::action *A_PGL_n_q,
		actions::action *A_PGL_k_q,
		groups::sims *G_PGL_k_q,
	int k, algebra::field_theory::finite_field *F,
	int verbose_level)
// we are acting on the elements of G_PGL_k_q, so the degree of the action 
// is the order of this group.
// A_PGL_k_q in only needed for make_element
{
	int f_v = (verbose_level >= 1);
	algebra::ring_theory::longinteger_object go;
	
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
		cout << "action_on_spread_set::init the order of "
				"the group of matrices is " << degree << endl;
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

void action_on_spread_set::report(
		std::ostream &ost, int verbose_level)
{
	ost << "Action on spread set has degree = " << degree << "\\\\" << endl;
	ost << "Low-level point size = " << low_level_point_size << "\\\\" << endl;
	algebra::ring_theory::longinteger_object go;
	G_PGL_k_q->group_order(go);
	ost << "PGL$(" << k << "," << q << ")$ has order " << go << "\\\\" << endl;
}

long int action_on_spread_set::compute_image_int(
		int *Elt,
		long int rk, int verbose_level)
{
	//verbose_level = 2;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int i, rk2;
	
	if (f_v) {
		cout << "action_on_spread_set::compute_image_int "
				"rk = " << rk << endl;
		}

	unrank_point(rk, mtx1, verbose_level - 1);
	matrix_to_subspace(mtx1, subspace1, verbose_level);

	if (f_vv) {
		cout << "action_on_spread_set::compute_image_int "
				"after unrank_point" << endl;
		Int_vec_print_integer_matrix_width(
				cout,
				subspace1, k, n, n, F->log10_of_q);
		cout << "action_on_spread_set::compute_image_int "
				"group element:" << endl;
		Int_matrix_print(Elt, n, n);
		}

	for (i = 0; i < k; i++) {
		A_PGL_n_q->Group_element->element_image_of_low_level(
			subspace1 + i * n, subspace2 + i * n, Elt, verbose_level - 1);
		}

	if (f_vv) {
		cout << "action_on_spread_set::compute_image_int "
				"after applying group element" << endl;
		Int_vec_print_integer_matrix_width(
				cout,
				subspace2, k, n, n, F->log10_of_q);
		}

	subspace_to_matrix(subspace2, mtx2, verbose_level - 1);
	rk2 = rank_point(mtx2, verbose_level - 1);

	if (f_v) {
		cout << "action_on_spread_set::compute_image_int "
				"image of " << rk << " is " << rk2 << endl;
		}
	return rk2;
}

void action_on_spread_set::matrix_to_subspace(
		int *mtx, int *subspace, int verbose_level)
{
	int i, j;
	
	Int_vec_zero(subspace, k * n);
	for (i = 0; i < k; i++) {
		subspace[i * n + i] = 1;
		for (j = 0; j < k; j++) {
			subspace[i * n + k + j] = mtx[i * k + j];
			}
		}
}

void action_on_spread_set::subspace_to_matrix(
		int *subspace, int *mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, r;
	
	if (f_v) {
		cout << "action_on_spread_set::subspace_to_matrix" << endl;
		}
	
	r = F->Linear_algebra->Gauss_easy(subspace, k, n);
	if (r != k) {
		cout << "action_on_spread_set::subspace_to_matrix "
				"r != k" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "action_on_spread_set::subspace_to_matrix "
				"after Gauss_easy" << endl;
		}
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			mtx[i * k + j] = subspace[i * n + k + j];
			}
		}
	if (f_v) {
		cout << "action_on_spread_set::subspace_to_matrix "
				"done" << endl;
		}
}

void action_on_spread_set::unrank_point(
		long int rk, int *mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_spread_set::unrank_point "
				"rk = " << rk << endl;
		}
	G_PGL_k_q->element_unrank_lint(rk, Elt1);
	Int_vec_copy(Elt1, mtx, k * k);
	if (f_v) {
		cout << "action_on_spread_set::unrank_point done" << endl;
		}
}

long int action_on_spread_set::rank_point(
		int *mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int rk;
	
	if (f_v) {
		cout << "action_on_spread_set::rank_point" << endl;
		}
	A_PGL_k_q->Group_element->make_element(Elt2, mtx, 0 /* verbose_level */);

	rk = G_PGL_k_q->element_rank_lint(Elt2);
	if (f_v) {
		cout << "action_on_spread_set::rank_point done, rk = " << rk << endl;
		}
	return rk;
}

void action_on_spread_set::compute_image_low_level(
		int *Elt, int *input, int *output, int verbose_level)
{
	//verbose_level = 2;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	
	if (f_v) {
		cout << "action_on_spread_set::compute_image_low_level" << endl;
		}
	if (f_vv) {
		cout << "action_on_spread_set::compute_image_low_level "
				"input=" << endl;
		Int_matrix_print(input, k, k);
		cout << "action_on_spread_set::compute_image_low_level "
				"matrix=" << endl;
		Int_matrix_print(Elt, n, n);
		}

	matrix_to_subspace(input, subspace1, verbose_level- 1);


	for (i = 0; i < k; i++) {
		A_PGL_n_q->Group_element->element_image_of_low_level(
			subspace1 + i * n,
			subspace2 + i * n,
			Elt,
			verbose_level - 2);
		}
	if (f_vv) {
		cout << "action_on_spread_set::compute_image_low_level "
				"after mult=" << endl;
		Int_matrix_print(subspace2, k, n);
		}

	subspace_to_matrix(subspace2, output, verbose_level - 1);

	if (f_vv) {
		cout << "action_on_spread_set::compute_image_low_level "
				"output=" << endl;
		Int_matrix_print(output, k, k);
		}

	if (f_v) {
		cout << "action_on_spread_set::compute_image_low_level "
				"done" << endl;
		}
}

}}}

