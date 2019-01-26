// action_on_grassmannian.C
//
// Anton Betten
// July 20, 2009

#include "foundations/foundations.h"
#include "group_actions.h"

namespace orbiter {

action_on_grassmannian::action_on_grassmannian()
{
	null();
}

action_on_grassmannian::~action_on_grassmannian()
{
	free();
}

void action_on_grassmannian::null()
{
	//M = NULL;
	M1 = NULL;
	M2 = NULL;
	G = NULL;
	GE = NULL;
	subspace_basis = NULL;
	subspace_basis2 = NULL;
	f_embedding = FALSE;
}

void action_on_grassmannian::free()
{
	int f_v = FALSE;

	if (M1) {
		if (f_v) {
			cout << "action_on_grassmannian::free "
					"before free M1" << endl;
			}
		FREE_int(M1);
		}
	if (M2) {
		if (f_v) {
			cout << "action_on_grassmannian::free "
					"before free M2" << endl;
			}
		FREE_int(M2);
		}
	if (GE) {
		if (f_v) {
			cout << "action_on_grassmannian::free "
					"before free GE" << endl;
			}
		FREE_OBJECT(GE);
		}
	if (subspace_basis) {
		if (f_v) {
			cout << "action_on_grassmannian::free "
					"before free subspace_basis" << endl;
			}
		FREE_int(subspace_basis);
		}
	if (subspace_basis2) {
		if (f_v) {
			cout << "action_on_grassmannian::free "
					"before free subspace_basis2" << endl;
			}
		FREE_int(subspace_basis2);
		}
	null();
}

void action_on_grassmannian::init(action &A,
		grassmann *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object go;
	longinteger_domain D;
	
	if (f_v) {
		cout << "action_on_grassmannian::init" << endl;
		}
	action_on_grassmannian::A = &A;
	action_on_grassmannian::G = G;
	n = G->n;
	k = G->k;
	q = G->q;
	F = G->F;
	low_level_point_size = k * n;


	if (f_v) {
		cout << "action_on_grassmannian::init" << endl;
		cout << "n=" << n << endl;
		cout << "k=" << k << endl;
		cout << "q=" << q << endl;
		}
	

	M1 = NEW_int(k * n);
	M2 = NEW_int(k * n);
	
	if (!A.f_is_linear) {
		cout << "action_on_grassmannian::init "
				"action not of linear type" << endl;
		exit(1);
		}

#if 0
	if (A.type_G == matrix_group_t) {
		M = A.G.matrix_grp;
		}
	else {
		action *sub = A.subaction;
		M = sub->G.matrix_grp;
		}
#endif
	
	D.q_binomial(degree, n, k, q, 0);
	max_string_length = degree.len();
	if (f_v) {
		cout << "degree = " << degree << endl;
		cout << "max_string_length = " << max_string_length << endl;
		cout << "low_level_point_size = " << low_level_point_size << endl;
		}
	
	if (f_v) {
		cout << "action_on_grassmannian::init done" << endl;
		}
}

void action_on_grassmannian::init_embedding(int big_n,
		int *ambient_space, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_grassmannian::init_embedding" << endl;
		cout << "big_n=" << big_n << endl;
		cout << "ambient space:" << endl;
		print_integer_matrix_width(cout, ambient_space,
				n, big_n, big_n, F->log10_of_q);
		}
	action_on_grassmannian::big_n = big_n;
	f_embedding = TRUE;
	GE = NEW_OBJECT(grassmann_embedded);
	GE->init(big_n, n, G, ambient_space, verbose_level);
	subspace_basis = NEW_int(n * big_n);
	subspace_basis2 = NEW_int(n * big_n);
}


void action_on_grassmannian::compute_image_longinteger(
	action *A, int *Elt,
	longinteger_object &i, longinteger_object &j,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h;
	
	if (f_v) {
		cout << "action_on_grassmannian::compute_image_longinteger "
				"i = " << i << endl;
		}
	G->unrank_longinteger(i, 0/*verbose_level - 1*/);
	if (f_vv) {
		cout << "after G->unrank_longinteger" << endl;
		print_integer_matrix_width(cout, G->M,
				G->k, G->n, G->n, F->log10_of_q);
		}
	for (h = 0; h < k; h++) {
		A->element_image_of_low_level(G->M + h * n,
				M1 + h * n, Elt, verbose_level - 1);
		}
	//A->element_image_of_low_level(G->M, M1, Elt, verbose_level - 1);
#if 0
	F->mult_matrix_matrix(G->M, Elt, M1, k, n, n);
	
	if (M->f_semilinear) {
		f = Elt[n * n];
		F->vector_frobenius_power_in_place(M1, k * n, f);
		}
#endif
	if (f_vv) {
		cout << "after element_image_of_low_level" << endl;
		print_integer_matrix_width(cout, M1,
				G->k, G->n, G->n, F->log10_of_q);
		}
	
	int_vec_copy(M1, G->M, k * n);
#if 0
	for (h = 0; h < k * n; h++) {
		G->M[h] = M1[h];
		}
#endif
	G->rank_longinteger(j, 0/*verbose_level - 1*/);
	if (f_v) {
		cout << "action_on_grassmannian::compute_image_longinteger "
				"image of " << i << " is " << j << endl;
		}
}

int action_on_grassmannian::compute_image_int(
	action *A, int *Elt,
	int i, int verbose_level)
{
	if (f_embedding) {
		return compute_image_int_embedded(A, Elt, i, verbose_level);
		}
	else {
		return compute_image_int_ordinary(A, Elt, i, verbose_level);
		}
}

int action_on_grassmannian::compute_image_int_ordinary(
	action *A, int *Elt,
	int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, j;
	
	if (f_v) {
		cout << "action_on_grassmannian::compute_image_int_ordinary "
				"i = " << i << endl;
		cout << "A->low_level_point_size="
				<< A->low_level_point_size << endl;
		cout << "using action " << A->label << endl;
		}
	G->unrank_int(i, verbose_level - 1);
	if (f_vv) {
		cout << "action_on_grassmannian::compute_image_int_ordinary "
				"after G->unrank_int" << endl;
		print_integer_matrix_width(cout, G->M,
				G->k, G->n, G->n, 2/* M->GFq->log10_of_q*/);
		}
	for (h = 0; h < k; h++) {
		A->element_image_of_low_level(G->M + h * n,
				M1 + h * n, Elt, verbose_level - 1);
		}
#if 0
	F->mult_matrix_matrix(G->M, Elt, M1, k, n, n);
	
	if (M->f_semilinear) {
		f = Elt[n * n];
		F->vector_frobenius_power_in_place(M1, k * n, f);
		}
#endif
	
	int_vec_copy(M1, G->M, k * n);
#if 0
	for (h = 0; h < k * n; h++) {
		G->M[h] = M1[h];
		}
#endif
	j = G->rank_int(verbose_level - 1);
	if (f_v) {
		cout << "action_on_grassmannian::compute_image_int_ordinary "
				"image of " << i << " is " << j << endl;
		}
	return j;
}

int action_on_grassmannian::compute_image_int_embedded(
	action *A, int *Elt,
	int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int j, h;
	
	if (f_v) {
		cout << "action_on_grassmannian::compute_image_int_embedded "
				"i = " << i << endl;
		cout << "calling GE->unrank_int" << endl;
		}
	GE->unrank_int(subspace_basis, i, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "action_on_grassmannian::compute_image_int_embedded "
				"subspace_basis:" << endl;
		cout << "k=" << k << endl;
		cout << "big_n=" << big_n << endl;
		print_integer_matrix_width(cout, subspace_basis,
				k, big_n, big_n, F->log10_of_q);
		}
	for (h = 0; h < k; h++) {
		A->element_image_of_low_level(
			subspace_basis + h * big_n,
			subspace_basis2 + h * big_n,
			Elt, verbose_level - 1);
		}
	
	//A->element_image_of_low_level(subspace_basis,
	// subspace_basis2, Elt, verbose_level - 1);
#if 0
	F->mult_matrix_matrix(subspace_basis, Elt,
			subspace_basis2, k, big_n, big_n);
	if (f_vv) {
		cout << "action_on_grassmannian::compute_image_int_embedded "
				"after mult_matrix_matrix:" << endl;
		print_integer_matrix_width(cout, subspace_basis2,
				k, big_n, big_n, F->log10_of_q);
		}
	
	if (M->f_semilinear) {
		f = Elt[big_n * big_n];
		if (f_v) {
			cout << "f_semilinear is TRUE, f=" << f << endl;
			}
		F->vector_frobenius_power_in_place(subspace_basis2, k * big_n, f);
		}
#endif
	
	if (f_vv) {
		cout << "action_on_grassmannian::compute_image_int_embedded "
				"subspace_basis after the action:" << endl;
		print_integer_matrix_width(cout, subspace_basis2,
				k, big_n, big_n, F->log10_of_q);
		}
	j = GE->rank_int(subspace_basis2,
			0 /*verbose_level - 1 */);
	if (f_v) {
		cout << "action_on_grassmannian::compute_image_int_embedded "
				"image of " << i << " is " << j << endl;
		}
	return j;
}

void action_on_grassmannian::print_point(int a, ostream &ost)
{
	G->unrank_int(a, 0);
	print_integer_matrix_width(ost, G->M,
			G->k, G->n, G->n, 2 /*M->GFq->log10_of_q*/);
}

}

