// grassmann_embedded.C
// 
// Anton Betten
// Jan 24, 2010
//
//
// 
//
//

#include "galois.h"

grassmann_embedded::grassmann_embedded()
{
	G = NULL;
	M = NULL;
	M_Gauss = NULL;
	transform = NULL;
	base_cols = NULL;
	embedding = NULL;
	Tmp1 = NULL;
	Tmp2 = NULL;
	Tmp3 = NULL;
	tmp_M1 = NULL;
	tmp_M2 = NULL;
}

grassmann_embedded::~grassmann_embedded()
{
	//if (G) {
		//delete G;
		//}
	if (M) {
		FREE_INT(M);
		}
	if (M_Gauss) {
		FREE_INT(M_Gauss);
		}
	if (transform) {
		FREE_INT(transform);
		}
	if (base_cols) {
		FREE_INT(base_cols);
		}
	if (embedding) {
		FREE_INT(embedding);
		}
	if (Tmp1) {
		FREE_INT(Tmp1);
		}
	if (Tmp2) {
		FREE_INT(Tmp2);
		}
	if (Tmp3) {
		FREE_INT(Tmp3);
		}
	if (tmp_M1) {
		FREE_INT(tmp_M1);
		}
	if (tmp_M2) {
		FREE_INT(tmp_M2);
		}
}

void grassmann_embedded::init(INT big_n, INT n, grassmann *G, INT *M, INT verbose_level)
// M is n x big_n
// G is for k-dimensional subspaces of an n-space.
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, rk, idx;
	longinteger_object deg;
	longinteger_domain D;
	
	grassmann_embedded::big_n = big_n;
	grassmann_embedded::G = G;
	grassmann_embedded::n = n;

	if (G->n != n) {
		cout << "grassmann_embedded::init n != G->n" << endl;
		exit(1);
		}
	grassmann_embedded::k = G->k;
	grassmann_embedded::F = G->F;
	grassmann_embedded::q = G->F->q;
	

	if (f_v) {
		cout << "grassmann_embedded::init big_n = " << big_n << " n=" << n << " k=" << k << " q=" << q << endl;
		}


	base_cols = NEW_INT(big_n);
	embedding = NEW_INT(big_n);
	grassmann_embedded::M = NEW_INT(n * big_n);
	M_Gauss = NEW_INT(n * big_n);
	transform = NEW_INT(n * n);
	tmp_M1 = NEW_INT(n * n);
	tmp_M2 = NEW_INT(n * n);
	Tmp1 = NEW_INT(big_n);
	Tmp2 = NEW_INT(big_n);
	Tmp3 = NEW_INT(big_n);
	for (i = 0; i < n * big_n; i++) {
		grassmann_embedded::M[i] = M[i];
		M_Gauss[i] = M[i];
		}
	// we initialize transform as the identity matrix:
	F->identity_matrix(transform, n);


	if (f_vv) {
		cout << "grassmann_embedded::init subspace basis before Gauss reduction:" << endl;
		print_integer_matrix_width(cout, grassmann_embedded::M, n, big_n, big_n, F->log10_of_q);
		}
	//rk = F->Gauss_simple(M_Gauss, n, big_n, base_cols, verbose_level - 1);
	rk = F->Gauss_INT(M_Gauss, FALSE /*f_special*/, TRUE/* f_complete*/, base_cols, 
		TRUE /* f_P */, transform, n, big_n, n /* Pn */, 0/*INT verbose_level*/);
	if (f_vv) {
		cout << "grassmann_embedded::init subspace basis after reduction:" << endl;
		print_integer_matrix_width(cout, M_Gauss, n, big_n, big_n, F->log10_of_q);
		cout << "grassmann_embedded::init transform:" << endl;
		print_integer_matrix_width(cout, transform, n, n, n, F->log10_of_q);
		}
	if (f_v) {
		cout << "base_cols:" << endl;
		INT_vec_print(cout, base_cols, rk);
		cout << endl;
		}
	if (rk != n) {
		cout << "grassmann_embedded::init rk != n" << endl;
		cout << "rk=" << rk << endl;
		cout << "n=" << n << endl;
		exit(1);
		}
	j = 0;
	for (i = 0; i < big_n; i++) {
		if (!INT_vec_search(base_cols, n, i, idx)) {
			embedding[j++] = i;
			}
		}
	if (j != big_n - n) {
		cout << "j != big_n - n" << endl;
		cout << "j=" << j << endl;
		cout << "big_n - n=" << big_n - n << endl;
		exit(1);
		}
	if (f_v) {
		cout << "embedding: ";
		INT_vec_print(cout, embedding, big_n - n);
		cout << endl;
		}
	D.q_binomial(deg, n, k, q, 0);
	degree = deg.as_INT();
}

void grassmann_embedded::unrank_embedded_INT(INT *subspace_basis_with_embedding, INT rk, INT verbose_level)
// subspace_basis_with_embedding is n x big_n
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "grassmann_embedded::unrank_embedded_INT" << endl;
		cout << "rk=" << rk << endl;
		cout << "calling G->unrank_INT" << endl;
		}
	G->unrank_embedded_subspace_INT(rk, verbose_level);
	if (f_v) {
		cout << "grassmann_embedded::unrank_embedded_INT coefficient matrix:" << endl;
		print_integer_matrix_width(cout, G->M, n /* not k */, n, n, F->log10_of_q);
		}
	if (f_v) {
		cout << "grassmann_embedded::unrank_embedded_INT subspace_basis:" << endl;
		print_integer_matrix_width(cout, M, n, big_n, big_n, F->log10_of_q);
		}
	F->mult_matrix_matrix(G->M, M, subspace_basis_with_embedding, n /* not k */, n, big_n);
	if (f_v) {
		cout << "grassmann_embedded::unrank_embedded_INT subspace_basis:" << endl;
		print_integer_matrix_width(cout, subspace_basis_with_embedding, n /* not k */, big_n, big_n, F->log10_of_q);
		}
}

INT grassmann_embedded::rank_embedded_INT(INT *subspace_basis, INT verbose_level)
// subspace_basis is n x big_n, only the first k x big_n entries are used
{
	INT f_v = (verbose_level >= 1);
	INT rk;

	if (f_v) {
		cout << "grassmann_embedded::rank_embedded_INT" << endl;
		//print_integer_matrix_width(cout, subspace_basis, n, big_n, big_n, F->log10_of_q);
		}
	rk = rank_INT(subspace_basis, verbose_level);
	if (f_v) {
		cout << "grassmann_embedded::rank_embedded_INT done" << endl;
		}
	return rk;
}

void grassmann_embedded::unrank_INT(INT *subspace_basis, INT rk, INT verbose_level)
// subspace_basis is k x big_n
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "grassmann_embedded::unrank_INT" << endl;
		cout << "rk=" << rk << endl;
		cout << "calling G->unrank_INT" << endl;
		}
	G->unrank_INT(rk, verbose_level);
	if (f_v) {
		cout << "grassmann_embedded::unrank_INT coefficient matrix:" << endl;
		print_integer_matrix_width(cout, G->M, k, n, n, F->log10_of_q);
		}
	if (f_v) {
		cout << "grassmann_embedded::rank_INT subspace_basis:" << endl;
		print_integer_matrix_width(cout, M, n, big_n, big_n, F->log10_of_q);
		}
	F->mult_matrix_matrix(G->M, M, subspace_basis, k, n, big_n);
	if (f_v) {
		cout << "grassmann_embedded::unrank_INT subspace_basis:" << endl;
		print_integer_matrix_width(cout, subspace_basis, k, big_n, big_n, F->log10_of_q);
		}
}

INT grassmann_embedded::rank_INT(INT *subspace_basis, INT verbose_level)
// subspace_basis is k x big_n
{
	INT f_v = (verbose_level >= 1);
	INT rk, i, j, a;


	if (f_v) {
		cout << "grassmann_embedded::rank_INT" << endl;
		print_integer_matrix_width(cout, subspace_basis, k, big_n, big_n, F->log10_of_q);
		}
	for (i = 0; i < k; i++) {
		for (j = 0; j < n; j++) {
			a = subspace_basis[i * big_n + base_cols[j]];
			tmp_M1[i * n + j] = a;
			}
		}
	// now tmp_M1 is k x n

	if (f_v) {
		cout << "grassmann_embedded::rank_INT tmp_M1:" << endl;
		print_integer_matrix_width(cout, tmp_M1, k, n, n, F->log10_of_q);
		}
	F->mult_matrix_matrix(tmp_M1, transform, tmp_M2, k, n, n);

	// now tmp_M2 is k x n
	// tmp_M2 is the coefficient matrix that defines the given subspace in terms of the big space M
	// this means: tmp_M2 * M = subspace_basis

	if (f_v) {
		cout << "grassmann_embedded::rank_INT tmp_M2:" << endl;
		print_integer_matrix_width(cout, tmp_M2, k, n, n, F->log10_of_q);
		}

	for (i = 0; i < k; i++) {
		F->mult_vector_from_the_left(tmp_M2 + i * n, M, Tmp2, n, big_n);

			// recall that M is n x big_n

			// now Tmp2 is of length big_n

		if (f_v) {
			cout << "grassmann_embedded::rank_INT i=" << i << " Tmp2=" << endl;
			print_integer_matrix_width(cout, Tmp2, 1, big_n, big_n, F->log10_of_q);
			}
		if (INT_vec_compare(subspace_basis + i * big_n, Tmp2, big_n)) {
			cout << "grassmann_embedded::rank_INT fatal: the i-th vector is not in the space" << endl;
			cout << "i=" << i << endl;
			cout << "subspace:" << endl;
			print_integer_matrix_width(cout, subspace_basis, k, big_n, big_n, F->log10_of_q);
			cout << "space:" << endl;
			print_integer_matrix_width(cout, M, n, big_n, big_n, F->log10_of_q);
			cout << "Tmp1:" << endl;
			INT_vec_print(cout, Tmp1, n);
			cout << endl;
			cout << "Tmp2:" << endl;
			INT_vec_print(cout, Tmp2, big_n);
			cout << endl;
			exit(1);
			}
		for (j = 0; j < n; j++) {
			G->M[i * n + j] = tmp_M2[i * n + j];
			}
		}
	if (f_v) {
		cout << "grassmann_embedded::rank_INT coefficient matrix:" << endl;
		print_integer_matrix_width(cout, G->M, k, n, n, F->log10_of_q);
		}
	rk = G->rank_INT(verbose_level);
	if (f_v) {
		cout << "rk=" << rk << endl;
		}
	return rk;
}

