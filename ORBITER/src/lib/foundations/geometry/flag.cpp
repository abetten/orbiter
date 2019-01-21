// flag.C
// 
// Anton Betten
// May 19, 2016
//
//
// 
//
//

#include "foundations.h"

namespace orbiter {


flag::flag()
{
	null();
}

flag::~flag()
{
	freeself();
}

void flag::null()
{
	Gr = NULL;
	Flag = NULL;
	M = NULL;
	M_Gauss = NULL;
	transform = NULL;
	base_cols = NULL;
	M1 = NULL;
	M2 = NULL;
	M3 = NULL;
}

void flag::freeself()
{
	if (Gr) {
		FREE_OBJECT(Gr);
		}
	if (Flag) {
		FREE_OBJECT(Flag);
		}
	if (M) {
		FREE_int(M);
		}
	if (M_Gauss) {
		FREE_int(M_Gauss);
		}
	if (transform) {
		FREE_int(transform);
		}
	if (base_cols) {
		FREE_int(base_cols);
		}
	if (M1) {
		FREE_int(M1);
		}
	if (M2) {
		FREE_int(M2);
		}
	if (M3) {
		FREE_int(M3);
		}
	null();
}

void flag::init(int n,
		int *type, int type_len, finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag::init type_len = "
				<< type_len << " idx=" << idx << endl;
		}
	init_recursion(n, type, type_len, type_len - 1, F, verbose_level);
	if (f_v) {
		cout << "flag::init done" << endl;
		}
}

void flag::init_recursion(int n,
		int *type, int type_len, int idx, finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "flag::init_recursion type_len = "
				<< type_len << " idx=" << idx << endl;
		}
	flag::F = F;
	flag::n = n;
	flag::type = type;
	flag::type_len = type_len;
	flag::idx = idx;

	Gr = NEW_OBJECT(grassmann);
	K = 0;
	for (i = 0; i < type_len; i++) {
		K += type[i];
		}
	s0 = 0;
	for (i = 0; i < idx; i++) {
		s0 += type[i];
		}
	k = type[idx];
	s1 = s0 + k;
	if (idx < type_len - 1) {
		s2 = s1 + type[idx + 1];
		}
	else {
		s2 = n;
		}
	Gr->init(s2, s1, F, 0 /* verbose_level */);
	if (f_v) {
		cout << "flag::init type_len = " << type_len << " s0=" << s0
			<< " s1=" << s1 << " s2=" << s2 << " k=" << k
			<< " K=" << K << endl;
		}
	M = NEW_int(K * n);
	M_Gauss = NEW_int(K * n);
	transform = NEW_int(K * K);
	base_cols = NEW_int(n);
	M1 = NEW_int(n * n);
	M2 = NEW_int(n * n);
	M3 = NEW_int(n * n);

	if (idx > 0) {	
		Flag = NEW_OBJECT(flag);
	
		Flag->init_recursion(n, type, type_len, idx - 1, F, verbose_level);
		}
	else {
		Flag = NULL;
		}
	if (idx == 0) {
		N0 = 1;
		}
	else {
		N0 = Flag->N;
		}
	N1 = generalized_binomial(s2, s1, F->q);
	N = N0 * N1;
	if (f_v) {
		cout << "flag::init_recursion type_len = " << type_len
				<< " N0=" << N0 << " N1=" << N1 << " N=" << N << endl;
		}

	if (f_v) {
		cout << "flag::init_recursion type_len = " << type_len
				<< " done" << endl;
		}
}

void flag::unrank(int rk, int *subspace, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b;

	if (f_v) {
		cout << "flag::unrank idx=" << idx << " rk=" << rk << endl;
		}

	if (idx != type_len - 1) {
		cout << "flag::unrank idx != type_len - 1" << endl;
		exit(1);
		}

	b = rk % N0;
	a = rk / N0;
	if (f_v) {
		cout << "flag::unrank idx=" << idx << " rk=" << rk
				<< " a=" << a << " b=" << b << endl;
		}

	Gr->unrank_embedded_subspace_int(a, 0 /*verbose_level*/);
	int_vec_copy(Gr->M, M, K * n);
	if (f_v) {
		cout << "flag::unrank M=" << endl;
		int_matrix_print(M, K, n);
		}
	if (Flag) {
		Flag->unrank_recursion(b, M, verbose_level);
		}
	int_vec_copy(M, subspace, K * n);
}

void flag::unrank_recursion(int rk, int *subspace, int verbose_level)
// subspace is K x n
{
	int f_v = (verbose_level >= 1);
	int a, b;

	if (f_v) {
		cout << "flag::unrank_recursion idx=" << idx
				<< " rk=" << rk << endl;
		}
	if (f_v) {
		cout << "flag::unrank_recursion subspace=" << endl;
		int_matrix_print(subspace, K, n);
		}

	b = rk % N0;
	a = rk / N0;

	if (f_v) {
		cout << "flag::unrank_recursion idx=" << idx
				<< " rk=" << rk << " a=" << a << " b=" << b << endl;
		}

	Gr->unrank_embedded_subspace_int(a, 0 /*verbose_level*/);

	// now Gr->M is s2 x s2

	if (f_v) {
		cout << "flag::unrank_recursion after unrank "
				<< a << ":" << endl;
		int_matrix_print(Gr->M, s2, n);
		}

	F->mult_matrix_matrix(Gr->M, subspace, M, s2, s2, n,
			0 /* verbose_level */);
	int_vec_copy(subspace + s2 * n, M + s2 * n, (K - s2) * n);

	if (f_v) {
		cout << "flag::unrank_recursion idx=" << idx
				<< " after mult, subspace=:" << endl;
		int_matrix_print(M, K, n);
		}

	

	if (Flag) {
		Flag->unrank_recursion(b, M, verbose_level);
		}
	int_vec_copy(M, subspace, K * n);
}

int flag::rank(int *input_subspace, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, rk;

	if (f_v) {
		cout << "flag::rank idx=" << idx << endl;
		}

	if (idx != type_len - 1) {
		cout << "flag::rank idx != type_len - 1" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "flag::rank input_subspace:" << endl;
		int_matrix_print(input_subspace, K, n);
		}

	int_vec_copy(input_subspace, M, K * n);
	int_vec_copy(input_subspace, Gr->M, s2 * n);
	a = Gr->rank_int(0 /*verbose_level*/);
	if (f_v) {
		cout << "flag::rank idx=" << idx << " a=" << a << endl;
		}
	Gr->unrank_embedded_subspace_int(a, 0 /*verbose_level*/);
	int_vec_copy(Gr->M, M1, K * n);
	if (f_v) {
		cout << "flag::rank after unrank:" << endl;
		int_matrix_print(M1, K, n);
		}
	if (Flag) {
		b = Flag->rank_recursion(M, M1, verbose_level);
		}
	else {
		b = 0;
		}
	rk = a * N0 + b;
	if (f_v) {
		cout << "flag::rank idx=" << idx << " rk=" << rk
				<< " a=" << a << " b=" << b << endl;
		}
	return rk;
}

int flag::rank_recursion(int *input_subspace,
		int *big_space, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, rk, r, i, j;

	if (f_v) {
		cout << "flag::rank_recursion idx=" << idx << endl;
		}
	if (f_v) {
		cout << "flag::rank_recursion input_subspace:" << endl;
		int_matrix_print(input_subspace, s1, n);
		}
	if (f_v) {
		cout << "flag::rank_recursion big_space:" << endl;
		int_matrix_print(big_space, s2, n);
		}
	int_vec_copy(big_space, M, s2 * n);
	int_vec_copy(big_space, M_Gauss, s2 * n);
	F->identity_matrix(transform, s2);
	r = F->Gauss_int(M_Gauss,
		FALSE /*f_special*/,
		TRUE/* f_complete*/, base_cols,
		TRUE /* f_P */, transform, s2, n,
		s2 /* Pn */,
		0/*int verbose_level*/);
	if (r != s2) {
		cout << "flag::rank_recursion r != s2" << endl;
		cout << "r=" << r << endl;
		cout << "s2=" << s2 << endl;
		exit(1);
		}
	if (f_v) {
		cout << "flag::rank_recursion transform:" << endl;
		int_matrix_print(transform, s2, s2);
		}
	for (i = 0; i < s1; i++) {
		for (j = 0; j < s2; j++) {
			a = input_subspace[i * n + base_cols[j]];
			M1[i * s2 + j] = a;
			}
		}
	// now M1 is s1 x s2
	if (f_v) {
		cout << "flag::rank_recursion input submatrix M1:" << endl;
		int_matrix_print(M1, s1, s2);
		}

	F->mult_matrix_matrix(M1, transform, M2, s1, s2, s2,
			0 /* verbose_level */);

	// now M2 is s1 x s2
	// M2 is the coefficient matrix that defines
	// the given subspace input_subspace
	// in terms of the big space big_space
	// this means: M2 * big_space = input_subspace
	if (f_v) {
		cout << "flag::rank_recursion coefficient matrix M2:" << endl;
		int_matrix_print(M2, s1, s2);
		}

	for (i = 0; i < s1; i++) {
		F->mult_vector_from_the_left(M2 + i * s2,
				M, M3, s2, n);
		if (f_v) {
			cout << "flag::rank_recursion i=" << i << " M3=" << endl;
			print_integer_matrix_width(cout,
					M3, 1, n, n, F->log10_of_q);
			}
		if (int_vec_compare(input_subspace + i * n, M3, n)) {
			cout << "flag::rank_recursion fatal: "
					"the i-th vector is not in the space" << endl;
			cout << "i=" << i << endl;
			exit(1);
			}
		// add one row to Gr->M:
		int_vec_copy(M2 + i * s2, Gr->M + i * s2, s2);
		}


	// now Gr->M is s1 x s2

	if (f_v) {
		cout << "flag::rank_recursion coefficient matrix:" << endl;
		print_integer_matrix_width(cout,
				Gr->M, s1, s2, s2, F->log10_of_q);
		}
	a = Gr->rank_int(verbose_level);
	if (f_v) {
		cout << "a=" << a << endl;
		}
	Gr->unrank_embedded_subspace_int(a, 0 /*verbose_level*/);
		// now Gr->M is s2 x s2
	if (f_v) {
		cout << "after unrank_embedded, coefficient matrix is" << endl;
		int_matrix_print(Gr->M, s2, s2);
		}
	F->mult_matrix_matrix(Gr->M, big_space, M, s2, s2, n,
			0 /* verbose_level */);
	int_vec_copy(big_space + s2 * n, M + s2 * n, (K - s2) * n);
	//now M is K x n
	if (f_v) {
		cout << "after unrank_embedded:" << endl;
		int_matrix_print(M, K, n);
		}

	if (Flag) {
		b = Flag->rank_recursion(input_subspace, M, verbose_level);
		}
	else {
		b = 0;
		}
	rk = a * N0 + b;
	if (f_v) {
		cout << "flag::rank_recursion idx=" << idx
				<< " a=" << a << " b=" << b << " rk=" << rk << endl;
		}
	return rk;
}

}

