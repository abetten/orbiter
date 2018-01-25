// action_on_grassmannian.C
//
// Anton Betten
// July 20, 2009

#include "galois.h"
#include "action.h"

INT action_on_grassmannian::cntr_new = 0;
INT action_on_grassmannian::cntr_objects = 0;
INT action_on_grassmannian::f_debug_memory = FALSE;

void *action_on_grassmannian::operator new(size_t bytes)
{
	cntr_new++;
	cntr_objects++;
	if (f_debug_memory) {
		cout << "action_on_grassmannian::operator new bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void *action_on_grassmannian::operator new[](size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(action_on_grassmannian);
	cntr_new++;
	cntr_objects += n;
	if (f_debug_memory) {
		cout << "action_on_grassmannian::operator new[] n=" << n 
			<< " bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void action_on_grassmannian::operator delete(void *ptr, size_t bytes)
{
	if (f_debug_memory) {
		cout << "action_on_grassmannian::operator delete bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects--;
	return ::free(ptr);
}

void action_on_grassmannian::operator delete[](void *ptr, size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(action_on_grassmannian);
	if (f_debug_memory) {
		cout << "action_on_grassmannian::operator delete[] n=" << n 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects -= n;
	return ::free(ptr);
}

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
	M = NULL;
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
	INT f_v = FALSE;

	if (M1) {
		if (f_v) {
			cout << "action_on_grassmannian::free before free M1" << endl;
			}
		FREE_INT(M1);
		}
	if (M2) {
		if (f_v) {
			cout << "action_on_grassmannian::free before free M2" << endl;
			}
		FREE_INT(M2);
		}
#if 0
	if (G) {
		delete G;
		}
#endif
	if (GE) {
		if (f_v) {
			cout << "action_on_grassmannian::free before free GE" << endl;
			}
		delete GE;
		}
	if (subspace_basis) {
		if (f_v) {
			cout << "action_on_grassmannian::free before free subspace_basis" << endl;
			}
		FREE_INT(subspace_basis);
		}
	if (subspace_basis2) {
		if (f_v) {
			cout << "action_on_grassmannian::free before free subspace_basis2" << endl;
			}
		FREE_INT(subspace_basis2);
		}
	null();
}

void action_on_grassmannian::init(action &A, grassmann *G, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
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
	
#if 0
	G = new grassmann;
	G->init(n, k, q, F, verbose_level - 1);
#endif

	M1 = NEW_INT(k * n);
	M2 = NEW_INT(k * n);
	
	if (A.type_G != matrix_group_t && 
		A.type_G != action_on_wedge_product_t && 
		A.type_G != action_by_subfield_structure_t && 
		A.type_G != action_by_representation_t && 
		A.type_G != action_on_spread_set_t) {
		cout << "action_on_grassmannian::init action not of linear type" << endl;
		exit(1);
		}
	if (A.type_G == matrix_group_t) {
		M = A.G.matrix_grp;
		}
	else {
		action *sub = A.subaction;
		M = sub->G.matrix_grp;
		}
	
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

void action_on_grassmannian::init_embedding(INT big_n, INT *ambient_space, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_grassmannian::init_embedding" << endl;
		cout << "big_n=" << big_n << endl;
		cout << "ambient space:" << endl;
		print_integer_matrix_width(cout, ambient_space, n, big_n, big_n, F->log10_of_q);
		}
	action_on_grassmannian::big_n = big_n;
	f_embedding = TRUE;
	GE = new grassmann_embedded;
	GE->init(big_n, n, G, ambient_space, verbose_level);
	subspace_basis = NEW_INT(n * big_n);
	subspace_basis2 = NEW_INT(n * big_n);
}


void action_on_grassmannian::compute_image_longinteger(action *A, INT *Elt, 
	longinteger_object &i, longinteger_object &j, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT h;
	
	if (f_v) {
		cout << "action_on_grassmannian::compute_image_longinteger i = " << i << endl;
		}
	G->unrank_longinteger(i, 0/*verbose_level - 1*/);
	if (f_vv) {
		cout << "after G->unrank_longinteger" << endl;
		print_integer_matrix_width(cout, G->M, G->k, G->n, G->n, F->log10_of_q);
		}
	for (h = 0; h < k; h++) {
		A->element_image_of_low_level(G->M + h * n, M1 + h * n, Elt, verbose_level - 1);
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
		print_integer_matrix_width(cout, M1, G->k, G->n, G->n, F->log10_of_q);
		}
	
	INT_vec_copy(M1, G->M, k * n);
#if 0
	for (h = 0; h < k * n; h++) {
		G->M[h] = M1[h];
		}
#endif
	G->rank_longinteger(j, 0/*verbose_level - 1*/);
	if (f_v) {
		cout << "action_on_grassmannian::compute_image_longinteger image of " << i << " is " << j << endl;
		}
}

INT action_on_grassmannian::compute_image_INT(action *A, INT *Elt, 
	INT i, INT verbose_level)
{
	if (f_embedding) {
		return compute_image_INT_embedded(A, Elt, i, verbose_level);
		}
	else {
		return compute_image_INT_ordinary(A, Elt, i, verbose_level);
		}
}

INT action_on_grassmannian::compute_image_INT_ordinary(action *A, INT *Elt, 
	INT i, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT h, j;
	
	if (f_v) {
		cout << "action_on_grassmannian::compute_image_INT_ordinary i = " << i << endl;
		cout << "A->low_level_point_size=" << A->low_level_point_size << endl;
		cout << "using action " << A->label << endl;
		}
	G->unrank_INT(i, verbose_level - 1);
	if (f_vv) {
		cout << "action_on_grassmannian::compute_image_INT_ordinary after G->unrank_INT" << endl;
		print_integer_matrix_width(cout, G->M, G->k, G->n, G->n, M->GFq->log10_of_q);
		}
	for (h = 0; h < k; h++) {
		A->element_image_of_low_level(G->M + h * n, M1 + h * n, Elt, verbose_level - 1);
		}
#if 0
	F->mult_matrix_matrix(G->M, Elt, M1, k, n, n);
	
	if (M->f_semilinear) {
		f = Elt[n * n];
		F->vector_frobenius_power_in_place(M1, k * n, f);
		}
#endif
	
	INT_vec_copy(M1, G->M, k * n);
#if 0
	for (h = 0; h < k * n; h++) {
		G->M[h] = M1[h];
		}
#endif
	j = G->rank_INT(verbose_level - 1);
	if (f_v) {
		cout << "action_on_grassmannian::compute_image_INT_ordinary image of " << i << " is " << j << endl;
		}
	return j;
}

INT action_on_grassmannian::compute_image_INT_embedded(action *A, INT *Elt, 
	INT i, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT j, h;
	
	if (f_v) {
		cout << "action_on_grassmannian::compute_image_INT_embedded i = " << i << endl;
		cout << "calling GE->unrank_INT" << endl;
		}
	GE->unrank_INT(subspace_basis, i, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "action_on_grassmannian::compute_image_INT_embedded subspace_basis:" << endl;
		cout << "k=" << k << endl;
		cout << "big_n=" << big_n << endl;
		print_integer_matrix_width(cout, subspace_basis, k, big_n, big_n, F->log10_of_q);
		}
	for (h = 0; h < k; h++) {
		A->element_image_of_low_level(
			subspace_basis + h * big_n, subspace_basis2 + h * big_n, Elt, verbose_level - 1);
		}
	
	//A->element_image_of_low_level(subspace_basis, subspace_basis2, Elt, verbose_level - 1);
#if 0
	F->mult_matrix_matrix(subspace_basis, Elt, subspace_basis2, k, big_n, big_n);
	if (f_vv) {
		cout << "action_on_grassmannian::compute_image_INT_embedded after mult_matrix_matrix:" << endl;
		print_integer_matrix_width(cout, subspace_basis2, k, big_n, big_n, F->log10_of_q);
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
		cout << "action_on_grassmannian::compute_image_INT_embedded subspace_basis after the action:" << endl;
		print_integer_matrix_width(cout, subspace_basis2, k, big_n, big_n, F->log10_of_q);
		}
	j = GE->rank_INT(subspace_basis2, 0 /*verbose_level - 1 */);
	if (f_v) {
		cout << "action_on_grassmannian::compute_image_INT_embedded image of " << i << " is " << j << endl;
		}
	return j;
}

void action_on_grassmannian::print_point(INT a, ostream &ost)
{
	G->unrank_INT(a, 0);
	print_integer_matrix_width(ost, G->M, G->k, G->n, G->n, M->GFq->log10_of_q);
}


