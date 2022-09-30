/*
 * linear_algebra.cpp
 *
 *  Created on: Jan 10, 2022
 *      Author: betten
 */





#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace linear_algebra {


linear_algebra::linear_algebra()
{
	F = NULL;
}

linear_algebra::~linear_algebra()
{
}

void linear_algebra::init(field_theory::finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra::init" << endl;
	}
	linear_algebra::F = F;
	if (f_v) {
		cout << "linear_algebra::init done" << endl;
	}
}

void linear_algebra::copy_matrix(int *A, int *B, int ma, int na)
{

	Int_vec_copy(A, B, ma * na);
}

void linear_algebra::reverse_matrix(int *A, int *B, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			B[i * n + j] = A[(m - 1 - i) * n + (n - 1 - j)];
		}
	}
}

void linear_algebra::identity_matrix(int *A, int n)
{
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				A[i * n + j] = 1;
			}
			else {
				A[i * n + j] = 0;
			}
		}
	}
}

int linear_algebra::is_identity_matrix(int *A, int n)
{
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				if (A[i * n + j] != 1) {
					return FALSE;
				}
			}
			else {
				if (A[i * n + j]) {
					return FALSE;
				}
			}
		}
	}
	return TRUE;
}

int linear_algebra::is_diagonal_matrix(int *A, int n)
{
	algebra::algebra_global Algebra;

	return Algebra.is_diagonal_matrix(A, n);
}

int linear_algebra::is_scalar_multiple_of_identity_matrix(
		int *A, int n, int &scalar)
{
	int i;

	if (!is_diagonal_matrix(A, n)) {
		return FALSE;
	}
	scalar = A[0 * n + 0];
	for (i = 1; i < n; i++) {
		if (A[i * n + i] != scalar) {
			return FALSE;
		}
	}
	return TRUE;
}

void linear_algebra::diagonal_matrix(int *A, int n, int alpha)
{
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				A[i * n + j] = alpha;
			}
			else {
				A[i * n + j] = 0;
			}
		}
	}
}

void linear_algebra::matrix_minor(int f_semilinear,
		int *A, int *B, int n, int f, int l)
// initializes B as the l x l minor of A
// (which is n x n) starting from row f.
{
	int i, j;

	if (f + l > n) {
		cout << "linear_algebra::matrix_minor f + l > n" << endl;
		exit(1);
	}
	for (i = 0; i < l; i++) {
		for (j = 0; j < l; j++) {
			B[i * l + j] = A[(f + i) * n + (f + j)];
		}
	}
	if (f_semilinear) {
		B[l * l] = A[n * n];
	}
}

void linear_algebra::mult_vector_from_the_left(int *v,
		int *A, int *vA, int m, int n)
// v[m], A[m][n], vA[n]
{
	mult_matrix_matrix(
			v, A, vA,
			1, m, n, 0 /*verbose_level */);
}

void linear_algebra::mult_vector_from_the_right(int *A,
		int *v, int *Av, int m, int n)
// A[m][n], v[n], Av[m]
{
	mult_matrix_matrix(
			A, v, Av,
			m, n, 1, 0 /*verbose_level */);
}

void linear_algebra::mult_matrix_matrix(
		int *A, int *B, int *C,
		int m, int n, int o, int verbose_level)
// matrix multiplication C := A * B,
// where A is m x n and B is n x o, so that C is m by o
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, k, a, b;

	if (f_v) {
		cout << "linear_algebra::mult_matrix_matrix" << endl;
	}
	if (f_vv) {
		cout << "A=" << endl;
		Int_matrix_print(A, m, n);
		cout << "B=" << endl;
		Int_matrix_print(B, n, o);
	}
	F->nb_calls_to_mult_matrix_matrix++;
	for (i = 0; i < m; i++) {
		for (j = 0; j < o; j++) {
			a = 0;
			for (k = 0; k < n; k++) {
				b = F->mult(A[i * n + k], B[k * o + j]);
				a = F->add(a, b);
			}
			C[i * o + j] = a;
		}
	}
	if (f_v) {
		cout << "linear_algebra::mult_matrix_matrix done" << endl;
	}
}

void linear_algebra::semilinear_matrix_mult(int *A, int *B, int *AB, int n)
// (A,f1) * (B,f2) = (A*B^{\varphi^{-f1}},f1+f2)
{
	int i, j, k, a, b, ab, c, f1, f2, f1inv;
	int *B2;
	number_theory::number_theory_domain NT;

	B2 = NEW_int(n * n);
	f1 = A[n * n];
	f2 = B[n * n];
	f1inv = NT.mod(-f1, F->e);
	Int_vec_copy(B, B2, n * n);
	vector_frobenius_power_in_place(B2, n * n, f1inv);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			c = 0;
			for (k = 0; k < n; k++) {
				//cout << "i=" << i << "j=" << j << "k=" << k;
				a = A[i * n + k];
				//cout << "a=A[" << i << "][" << k << "]=" << a;
				b = B2[k * n + j];
				ab = F->mult(a, b);
				c = F->add(c, ab);
				//cout << "b=" << b << "ab=" << ab << "c=" << c << endl;
			}
			AB[i * n + j] = c;
		}
	}
	AB[n * n] = NT.mod(f1 + f2, F->e);
	//vector_frobenius_power_in_place(B, n * n, f1);
	FREE_int(B2);
}

void linear_algebra::semilinear_matrix_mult_memory_given(
		int *A, int *B, int *AB, int *tmp_B, int n, int verbose_level)
// (A,f1) * (B,f2) = (A*B^{\varphi^{-f1}},f1+f2)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, k, a, b, ab, c, f1, f2, f1inv;
	int *B2 = tmp_B;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "linear_algebra::semilinear_matrix_mult_memory_given" << endl;
	}
	//B2 = NEW_int(n * n);
	f1 = A[n * n];
	f2 = B[n * n];
	f1inv = NT.mod(-f1, F->e);
	if (f_vv) {
		cout << "linear_algebra::semilinear_matrix_mult_memory_given f1=" << f1 << endl;
		cout << "linear_algebra::semilinear_matrix_mult_memory_given f2=" << f2 << endl;
		cout << "linear_algebra::semilinear_matrix_mult_memory_given f1inv=" << f1inv << endl;
	}

	Int_vec_copy(B, B2, n * n);
	vector_frobenius_power_in_place(B2, n * n, f1inv);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			c = 0;
			for (k = 0; k < n; k++) {
				//cout << "i=" << i << "j=" << j << "k=" << k;
				a = A[i * n + k];
				//cout << "a=A[" << i << "][" << k << "]=" << a;
				b = B2[k * n + j];
				ab = F->mult(a, b);
				c = F->add(c, ab);
				//cout << "b=" << b << "ab=" << ab << "c=" << c << endl;
			}
			AB[i * n + j] = c;
		}
	}
	AB[n * n] = NT.mod(f1 + f2, F->e);
	//vector_frobenius_power_in_place(B, n * n, f1);
	//FREE_int(B2);
	if (f_v) {
		cout << "linear_algebra::semilinear_matrix_mult_memory_given done" << endl;
	}
}

void linear_algebra::matrix_mult_affine(int *A, int *B, int *AB,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *b1, *b2, *b3;
	int *A1, *A2, *A3;

	if (f_v) {
		cout << "linear_algebra::matrix_mult_affine" << endl;
	}
	A1 = A;
	A2 = B;
	A3 = AB;
	b1 = A + n * n;
	b2 = B + n * n;
	b3 = AB + n * n;
	if (f_vv) {
		cout << "A1=" << endl;
		Int_matrix_print(A1, n, n);
		cout << "b1=" << endl;
		Int_matrix_print(b1, 1, n);
		cout << "A2=" << endl;
		Int_matrix_print(A2, n, n);
		cout << "b2=" << endl;
		Int_matrix_print(b2, 1, n);
	}

	mult_matrix_matrix(A1, A2, A3, n, n, n, 0 /* verbose_level */);
	if (f_vv) {
		cout << "A3=" << endl;
		Int_matrix_print(A3, n, n);
	}
	mult_matrix_matrix(b1, A2, b3, 1, n, n, 0 /* verbose_level */);
	if (f_vv) {
		cout << "b3=" << endl;
		Int_matrix_print(b3, 1, n);
	}
	add_vector(b3, b2, b3, n);
	if (f_vv) {
		cout << "b3 after adding b2=" << endl;
		Int_matrix_print(b3, 1, n);
	}

	if (f_v) {
		cout << "linear_algebra::matrix_mult_affine done" << endl;
	}
}

void linear_algebra::semilinear_matrix_mult_affine(
		int *A, int *B, int *AB, int n)
{
	int f1, f2, f12, f1inv;
	int *b1, *b2, *b3;
	int *A1, *A2, *A3;
	int *T;
	number_theory::number_theory_domain NT;

	T = NEW_int(n * n);
	A1 = A;
	A2 = B;
	A3 = AB;
	b1 = A + n * n;
	b2 = B + n * n;
	b3 = AB + n * n;

	f1 = A[n * n + n];
	f2 = B[n * n + n];
	f12 = NT.mod(f1 + f2, F->e);
	f1inv = NT.mod(F->e - f1, F->e);

	Int_vec_copy(A2, T, n * n);
	vector_frobenius_power_in_place(T, n * n, f1inv);
	mult_matrix_matrix(A1, T, A3, n, n, n, 0 /* verbose_level */);
	//vector_frobenius_power_in_place(A2, n * n, f1);

	mult_matrix_matrix(b1, A2, b3, 1, n, n, 0 /* verbose_level */);
	vector_frobenius_power_in_place(b3, n, f2);
	add_vector(b3, b2, b3, n);

	AB[n * n + n] = f12;
	FREE_int(T);
}

int linear_algebra::matrix_determinant(int *A, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, eps = 1, a, det, det1, det2;
	int *Tmp, *Tmp1;

	if (f_v) {
		cout << "linear_algebra::matrix_determinant" << endl;
	}
	if (n == 1) {
		return A[0];
	}
	if (f_vv) {
		cout << "linear_algebra::matrix_determinant determinant of " << endl;
		Int_vec_print_integer_matrix_width(cout, A, n, n, n, 2);
	}
	Tmp = NEW_int(n * n);
	Tmp1 = NEW_int(n * n);
	Int_vec_copy(A, Tmp, n * n);

	// search for nonzero element in the first column:
	for (i = 0; i < n; i++) {
		if (Tmp[i * n + 0]) {
			break;
		}
	}
	if (i == n) {
		FREE_int(Tmp);
		FREE_int(Tmp1);
		return 0;
	}

	// possibly permute the row with the nonzero element up front:
	if (i != 0) {
		for (j = 0; j < n; j++) {
			a = Tmp[0 * n + j];
			Tmp[0 * n + j] = Tmp[i * n + j];
			Tmp[i * n + j] = a;
		}
		if (ODD(i)) {
			eps *= -1;
		}
	}

	// pick the pivot element:
	det = Tmp[0 * n + 0];

	// eliminate the first column:
	for (i = 1; i < n; i++) {
		Gauss_step(Tmp, Tmp + i * n, n, 0, 0 /* verbose_level */);
	}


	if (eps < 0) {
		det = F->negate(det);
	}
	if (f_vv) {
		cout << "linear_algebra::matrix_determinant after Gauss " << endl;
		Int_vec_print_integer_matrix_width(cout, Tmp, n, n, n, 2);
		cout << "linear_algebra::matrix_determinant det= " << det << endl;
	}

	// delete the first row and column and form the matrix
	// Tmp1 of size (n - 1) x (n - 1):
	for (i = 1; i < n; i++) {
		for (j = 1; j < n; j++) {
			Tmp1[(i - 1) * (n - 1) + j - 1] = Tmp[i * n + j];
		}
	}
	if (f_vv) {
		cout << "linear_algebra::matrix_determinant computing determinant of " << endl;
		Int_vec_print_integer_matrix_width(cout, Tmp1, n - 1, n - 1, n - 1, 2);
	}
	det1 = matrix_determinant(Tmp1, n - 1, 0/*verbose_level*/);
	if (f_vv) {
		cout << "as " << det1 << endl;
	}

	// multiply the pivot element:
	det2 = F->mult(det, det1);

	FREE_int(Tmp);
	FREE_int(Tmp1);
	if (f_vv) {
		cout << "linear_algebra::matrix_determinant determinant is " << det2 << endl;
	}

	return det2;
#if 0
	int *Tmp, *Tmp_basecols, *P, *perm;
	int rk, det, i, j, eps;

	Tmp = NEW_int(n * n + 1);
	Tmp_basecols = NEW_int(n);
	P = NEW_int(n * n);
	perm = NEW_int(n);

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j)
				P[i * n + j] = 1;
			else
				P[i * n + j] = 0;
			}
		}

	copy_matrix(A, Tmp, n, n);
	cout << "before Gauss:" << endl;
	print_integer_matrix_width(cout, Tmp, n, n, n, 2);
	rk = Gauss_int(Tmp,
		TRUE /* f_special */,
		FALSE /*f_complete */, Tmp_basecols,
		TRUE /* f_P */, P, n, n, n, verbose_level - 2);
	cout << "after Gauss:" << endl;
	print_integer_matrix_width(cout, Tmp, n, n, n, 2);
	cout << "P:" << endl;
	print_integer_matrix_width(cout, P, n, n, n, 2);
	if (rk < n) {
		det = 0;
		}
	else {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				if (P[i * n + j]) {
					perm[i] = j;
					break;
					}
				}
			}
		cout << "permutation : ";
		perm_print_list(cout, perm, n);
		perm_print(cout, perm, n);
		cout << endl;
		eps = perm_signum(perm, n);

		det = 1;
		for (i = 0; i < n; i++) {
			det = mult(det, Tmp[i * n + i]);
			}
		if (eps < 0) {
			det = mult(det, negate(1));
			}
		}
	cout << "det=" << det << endl;

	FREE_int(Tmp);
	FREE_int(Tmp_basecols);
	FREE_int(P);
	FREE_int(perm);
	return det;
#endif
	if (f_v) {
		cout << "linear_algebra::matrix_determinant done" << endl;
	}
}

void linear_algebra::matrix_inverse(int *A, int *Ainv, int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Tmp, *Tmp_basecols;

	if (f_v) {
		cout << "linear_algebra::matrix_inverse" << endl;
	}
	Tmp = NEW_int(n * n + 1);
	Tmp_basecols = NEW_int(n);

	matrix_invert(A, Tmp, Tmp_basecols, Ainv, n, verbose_level);

	FREE_int(Tmp);
	FREE_int(Tmp_basecols);
	if (f_v) {
		cout << "linear_algebra::matrix_inverse done" << endl;
	}
}

void linear_algebra::matrix_invert(int *A, int *Tmp, int *Tmp_basecols,
	int *Ainv, int n, int verbose_level)
// Tmp[n * n + 1]
// Tmp_basecols[n]
{
	int rk;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "linear_algebra::matrix_invert" << endl;
	}
	if (f_vv) {
		Int_vec_print_integer_matrix_width(cout, A, n, n, n, F->log10_of_q + 1);
	}
	copy_matrix(A, Tmp, n, n);
	identity_matrix(Ainv, n);
	rk = Gauss_int(Tmp,
		FALSE /* f_special */, TRUE /*f_complete */, Tmp_basecols,
		TRUE /* f_P */, Ainv, n, n, n, verbose_level - 2);
	if (rk < n) {
		cout << "linear_algebra::matrix_invert not invertible" << endl;
		cout << "input matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, A, n, n, n, F->log10_of_q + 1);
		cout << "Tmp matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, Tmp, n, n, n, F->log10_of_q + 1);
		cout << "rk=" << rk << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "the inverse is" << endl;
		Int_vec_print_integer_matrix_width(cout, Ainv, n, n, n, F->log10_of_q + 1);
	}
	if (f_v) {
		cout << "linear_algebra::matrix_invert done" << endl;
	}
}

void linear_algebra::semilinear_matrix_invert(int *A,
	int *Tmp, int *Tmp_basecols, int *Ainv, int n,
	int verbose_level)
// Tmp[n * n + 1]
// Tmp_basecols[n]
{
	int f, finv;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "linear_algebra::semilinear_matrix_invert" << endl;
	}
	if (f_vv) {
		Int_vec_print_integer_matrix_width(cout, A, n, n, n, F->log10_of_q + 1);
		cout << "frobenius: " << A[n * n] << endl;
	}
	matrix_invert(A, Tmp, Tmp_basecols, Ainv, n, verbose_level - 1);
	f = A[n * n];
	vector_frobenius_power_in_place(Ainv, n * n, f);
	finv = NT.mod(-f, F->e);
	Ainv[n * n] = finv;
	if (f_vv) {
		cout << "the inverse is" << endl;
		Int_vec_print_integer_matrix_width(cout, Ainv, n, n, n, F->log10_of_q + 1);
		cout << "frobenius: " << Ainv[n * n] << endl;
	}
	if (f_v) {
		cout << "linear_algebra::semilinear_matrix_invert done" << endl;
	}
}

void linear_algebra::semilinear_matrix_invert_affine(int *A,
	int *Tmp, int *Tmp_basecols, int *Ainv, int n,
	int verbose_level)
// Tmp[n * n + 1]
// Tmp_basecols[n]
{
	int f, finv;
	int *b1, *b2;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "linear_algebra::semilinear_matrix_invert_affine" << endl;
	}
	if (f_vv) {
		Int_vec_print_integer_matrix_width(cout, A, n, n, n, F->log10_of_q + 1);
		cout << "b: ";
		Int_vec_print(cout, A + n * n, n);
		cout << " frobenius: " << A[n * n + n] << endl;
	}
	b1 = A + n * n;
	b2 = Ainv + n * n;
	matrix_invert(A, Tmp, Tmp_basecols, Ainv, n, verbose_level - 1);
	f = A[n * n + n];
	finv = NT.mod(-f, F->e);
	vector_frobenius_power_in_place(Ainv, n * n, f);

	mult_matrix_matrix(b1, Ainv, b2, 1, n, n, 0 /* verbose_level */);

	vector_frobenius_power_in_place(b2, n, finv);

	negate_vector_in_place(b2, n);

	Ainv[n * n + n] = finv;
	if (f_vv) {
		cout << "the inverse is" << endl;
		Int_vec_print_integer_matrix_width(cout, Ainv, n, n, n, F->log10_of_q + 1);
		cout << "b: ";
		Int_vec_print(cout, Ainv + n * n, n);
		cout << " frobenius: " << Ainv[n * n + n] << endl;
	}
	if (f_v) {
		cout << "linear_algebra::semilinear_matrix_invert_affine done" << endl;
	}
}


void linear_algebra::matrix_invert_affine(int *A,
	int *Tmp, int *Tmp_basecols, int *Ainv, int n,
	int verbose_level)
// Tmp[n * n + 1]
// Tmp_basecols[n]
{
	int *b1, *b2;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "linear_algebra::matrix_invert_affine" << endl;
	}
	if (f_vv) {
		Int_vec_print_integer_matrix_width(cout, A, n, n, n, F->log10_of_q + 1);
		cout << "b: ";
		Int_vec_print(cout, A + n * n, n);
		cout << endl;
	}
	b1 = A + n * n;
	b2 = Ainv + n * n;
	matrix_invert(A, Tmp, Tmp_basecols, Ainv, n, verbose_level - 1);

	mult_matrix_matrix(b1, Ainv, b2, 1, n, n, 0 /* verbose_level */);

	negate_vector_in_place(b2, n);

	if (f_vv) {
		cout << "the inverse is" << endl;
		Int_vec_print_integer_matrix_width(cout, Ainv, n, n, n, F->log10_of_q + 1);
		cout << "b: ";
		Int_vec_print(cout, Ainv + n * n, n);
		cout << endl;
	}
	if (f_v) {
		cout << "linear_algebra::matrix_invert_affine done" << endl;
	}
}


void linear_algebra::projective_action_from_the_right(int f_semilinear,
	int *v, int *A, int *vA, int n,
	int verbose_level)
// vA = (v * A)^{p^f}  if f_semilinear
// (where f = A[n *  n]),   vA = v * A otherwise
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra::projective_action_from_the_right"  << endl;
	}
	if (f_semilinear) {
		semilinear_action_from_the_right(v, A, vA, n);
	}
	else {
		mult_vector_from_the_left(v, A, vA, n, n);
	}
	if (f_v) {
		cout << "linear_algebra::projective_action_from_the_right done"  << endl;
	}
}

void linear_algebra::general_linear_action_from_the_right(int f_semilinear,
	int *v, int *A, int *vA, int n,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra::general_linear_action_from_the_right"
				<< endl;
	}
	if (f_semilinear) {
		semilinear_action_from_the_right(v, A, vA, n);
	}
	else {
		mult_vector_from_the_left(v, A, vA, n, n);
	}
	if (f_v) {
		cout << "linear_algebra::general_linear_action_from_the_right done"
				<< endl;
	}
}


void linear_algebra::semilinear_action_from_the_right(
		int *v, int *A, int *vA, int n)
// vA = (v * A)^{p^f}  (where f = A[n *  n])
{
	int f;

	f = A[n * n];
	mult_vector_from_the_left(v, A, vA, n, n);
	vector_frobenius_power_in_place(vA, n, f);
}

void linear_algebra::semilinear_action_from_the_left(
		int *A, int *v, int *Av, int n)
// Av = A * v^{p^f}
{
	int f;

	f = A[n * n];
	mult_vector_from_the_right(A, v, Av, n, n);
	vector_frobenius_power_in_place(Av, n, f);
}

void linear_algebra::affine_action_from_the_right(
		int f_semilinear, int *v, int *A, int *vA, int n)
// vA = (v * A)^{p^f} + b
{
	mult_vector_from_the_left(v, A, vA, n, n);
	if (f_semilinear) {
		int f;

		f = A[n * n + n];
		vector_frobenius_power_in_place(vA, n, f);
	}
	add_vector(vA, A + n * n, vA, n);
}

void linear_algebra::zero_vector(int *A, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		A[i] = 0;
	}
}

void linear_algebra::all_one_vector(int *A, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		A[i] = 1;
	}
}

void linear_algebra::support(int *A, int m, int *&support, int &size)
{
	int i;

	support = NEW_int(m);
	size = 0;
	for (i = 0; i < m; i++) {
		if (A[i]) {
			support[size++] = i;
		}
	}
}

void linear_algebra::characteristic_vector(int *A, int m, int *set, int size)
{
	int i;

	zero_vector(A, m);
	for (i = 0; i < size; i++) {
		A[set[i]] = 1;
	}
}

int linear_algebra::is_zero_vector(int *A, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		if (A[i]) {
			return FALSE;
		}
	}
	return TRUE;
}

void linear_algebra::add_vector(int *A, int *B, int *C, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		C[i] = F->add(A[i], B[i]);
	}
}

void linear_algebra::linear_combination_of_vectors(
		int a, int *A, int b, int *B, int *C, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		C[i] = F->add(F->mult(a, A[i]), F->mult(b, B[i]));
	}
}

void linear_algebra::linear_combination_of_three_vectors(
		int a, int *A, int b, int *B, int c, int *C, int *D, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		D[i] = F->add3(F->mult(a, A[i]), F->mult(b, B[i]), F->mult(c, C[i]));
	}
}

void linear_algebra::negate_vector(int *A, int *B, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		B[i] = F->negate(A[i]);
	}
}

void linear_algebra::negate_vector_in_place(int *A, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		A[i] = F->negate(A[i]);
	}
}

void linear_algebra::scalar_multiply_vector_in_place(int c, int *A, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		A[i] = F->mult(c, A[i]);
	}
}

void linear_algebra::vector_frobenius_power_in_place(int *A, int m, int f)
{
	int i;

	for (i = 0; i < m; i++) {
		A[i] = F->frobenius_power(A[i], f);
	}
}

int linear_algebra::dot_product(int len, int *v, int *w)
{
	int i, a = 0, b;

	for (i = 0; i < len; i++) {
		b = F->mult(v[i], w[i]);
		a = F->add(a, b);
	}
	return a;
}

void linear_algebra::transpose_matrix(int *A, int *At, int ma, int na)
{
	int i, j;

	for (i = 0; i < ma; i++) {
		for (j = 0; j < na; j++) {
			At[j * ma + i] = A[i * na + j];
		}
	}
}

void linear_algebra::transpose_matrix_in_place(int *A, int m)
{
	int i, j, a;

	for (i = 0; i < m; i++) {
		for (j = i + 1; j < m; j++) {
			a = A[i * m + j];
			A[i * m + j] = A[j * m + i];
			A[j * m + i] = a;
		}
	}
}

void linear_algebra::invert_matrix(int *A, int *A_inv, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *A_tmp;
	int *basecols;

	if (f_v) {
		cout << "linear_algebra::invert_matrix" << endl;
	}
	A_tmp = NEW_int(n * n);
	basecols = NEW_int(n);


	invert_matrix_memory_given(A, A_inv, n, A_tmp, basecols, verbose_level);

	FREE_int(A_tmp);
	FREE_int(basecols);
	if (f_v) {
		cout << "linear_algebra::invert_matrix done" << endl;
	}
}

void linear_algebra::invert_matrix_memory_given(int *A, int *A_inv, int n,
		int *tmp_A, int *tmp_basecols, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, rk;
	int *A_tmp;
	int *base_cols;

	if (f_v) {
		cout << "linear_algebra::invert_matrix_memory_given" << endl;
	}
	A_tmp = tmp_A;
	base_cols = tmp_basecols;


	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				a = 1;
			}
			else {
				a = 0;
			}
			A_inv[i * n + j] = a;
		}
	}
	Int_vec_copy(A, A_tmp, n * n);

	rk = Gauss_int(A_tmp,
			FALSE /* f_special */,
			TRUE /*f_complete */, base_cols,
			TRUE /* f_P */, A_inv, n, n, n, 0 /* verbose_level */);
	if (rk < n) {
		cout << "linear_algebra::invert_matrix "
				"matrix is not invertible, the rank is " << rk << endl;
		exit(1);
	}

	if (f_v) {
		cout << "linear_algebra::invert_matrix_memory_given done" << endl;
	}
}

void linear_algebra::transform_form_matrix(int *A,
		int *Gram, int *new_Gram, int d, int verbose_level)
// computes new_Gram = A * Gram * A^\top
{
	int f_v = (verbose_level >= 1);
	int *Tmp1, *Tmp2;

	if (f_v) {
		cout << "linear_algebra::transform_form_matrix" << endl;
	}
	Tmp1 = NEW_int(d * d);
	Tmp2 = NEW_int(d * d);

	transpose_matrix(A, Tmp1, d, d);
	mult_matrix_matrix(A, Gram, Tmp2, d, d, d, 0 /* verbose_level */);
	mult_matrix_matrix(Tmp2, Tmp1, new_Gram, d, d, d, 0 /* verbose_level */);

	FREE_int(Tmp1);
	FREE_int(Tmp2);
	if (f_v) {
		cout << "linear_algebra::transform_form_matrix done" << endl;
	}
}

int linear_algebra::rank_of_matrix(int *A, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *B, *base_cols, rk;

	if (f_v) {
		cout << "linear_algebra::rank_of_matrix" << endl;
	}
	B = NEW_int(m * m);
	base_cols = NEW_int(m);

	rk = rank_of_matrix_memory_given(A,
			m, B, base_cols, verbose_level);

	FREE_int(base_cols);
	FREE_int(B);
	if (f_v) {
		cout << "linear_algebra::rank_of_matrix done" << endl;
	}
	return rk;
}

int linear_algebra::rank_of_matrix_memory_given(int *A,
		int m, int *B, int *base_cols, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk;

	if (f_v) {
		cout << "linear_algebra::rank_of_matrix_memory_given" << endl;
	}
	Int_vec_copy(A, B, m * m);
	rk = Gauss_int(B, FALSE, FALSE, base_cols, FALSE,
			NULL, m, m, m, 0 /* verbose_level */);
	if (FALSE) {
		cout << "the matrix ";
		if (f_vv) {
			cout << endl;
			Int_vec_print_integer_matrix_width(cout, A, m, m, m, 2);
		}
		cout << "has rank " << rk << endl;
	}
	if (f_v) {
		cout << "linear_algebra::rank_of_matrix_memory_given done" << endl;
	}
	return rk;
}

int linear_algebra::rank_of_rectangular_matrix(int *A,
		int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *B, *base_cols;
	int rk;

	if (f_v) {
		cout << "linear_algebra::rank_of_rectangular_matrix" << endl;
	}
	B = NEW_int(m * n);
	base_cols = NEW_int(n);

	int f_complete = FALSE;


	rk = rank_of_rectangular_matrix_memory_given(
			A, m, n, B, base_cols, f_complete, verbose_level);

	FREE_int(base_cols);
	FREE_int(B);
	if (f_v) {
		cout << "linear_algebra::rank_of_rectangular_matrix done" << endl;
	}
	return rk;
}

int linear_algebra::rank_of_rectangular_matrix_memory_given(
		int *A, int m, int n, int *B, int *base_cols, int f_complete,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk;

	if (f_v) {
		cout << "linear_algebra::rank_of_rectangular_matrix_memory_given" << endl;
	}
	//B = NEW_int(m * n);
	//base_cols = NEW_int(n);
	Int_vec_copy(A, B, m * n);
	rk = Gauss_int(B, FALSE, f_complete, base_cols, FALSE,
			NULL, m, n, n, 0 /* verbose_level */);

	if (FALSE) {
		cout << "the matrix ";
		if (f_vv) {
			cout << endl;
			Int_vec_print_integer_matrix_width(cout, A, m, n, n, 2);
		}
		cout << "has rank " << rk << endl;
	}

	if (f_v) {
		cout << "linear_algebra::rank_of_rectangular_matrix_memory_given done" << endl;
	}
	return rk;
}

int linear_algebra::rank_and_basecols(int *A, int m,
		int *base_cols, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *B, rk;

	if (f_v) {
		cout << "linear_algebra::rank_and_basecols" << endl;
	}
	B = NEW_int(m * m);
	Int_vec_copy(A, B, m * m);
	rk = Gauss_int(B, FALSE, FALSE, base_cols, FALSE,
			NULL, m, m, m, 0 /* verbose_level */);
	if (FALSE) {
		cout << "the matrix ";
		if (f_vv) {
			cout << endl;
			Int_vec_print_integer_matrix_width(cout, A, m, m, m, 2);
		}
		cout << "has rank " << rk << endl;
	}
	FREE_int(B);
	if (f_v) {
		cout << "linear_algebra::rank_and_basecols done" << endl;
	}
	return rk;
}

void linear_algebra::Gauss_step(int *v1, int *v2,
		int len, int idx, int verbose_level)
// afterwards: v2[idx] = 0 and v1,v2 span the same space as before
// v1 is not changed if v1[idx] is nonzero
{
	int i, a;
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;//(verbose_level >= 2);

	if (f_v) {
		cout << "linear_algebra::Gauss_step" << endl;
	}
	if (f_vv) {
		cout << "before:" << endl;
		Int_vec_print(cout, v1, len);
		cout << endl;
		Int_vec_print(cout, v2, len);
		cout << endl;
		cout << "pivot column " << idx << endl;
	}
	if (v2[idx] == 0) {
		goto after;
	}
	if (v1[idx] == 0) {
		// do a swap:
		for (i = 0; i < len; i++) {
			a = v2[i];
			v2[i] = v1[i];
			v1[i] = a;
		}
		goto after;
	}
	a = F->negate(F->mult(F->inverse(v1[idx]), v2[idx]));
	//cout << "Gauss_step a=" << a << endl;
	for (i = 0; i < len; i++) {
		v2[i] = F->add(F->mult(v1[i], a), v2[i]);
	}
after:
	if (f_vv) {
		cout << "linear_algebra::Gauss_step after:" << endl;
		Int_vec_print(cout, v1, len);
		cout << endl;
		Int_vec_print(cout, v2, len);
		cout << endl;
	}
	if (f_v) {
		cout << "linear_algebra::Gauss_step done" << endl;
	}
}

void linear_algebra::Gauss_step_make_pivot_one(int *v1, int *v2,
	int len, int idx, int verbose_level)
// afterwards:  v1,v2 span the same space as before
// v2[idx] = 0, v1[idx] = 1,
{
	int i, a, av;
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;//(verbose_level >= 2);

	if (f_v) {
		cout << "linear_algebra::Gauss_step_make_pivot_one" << endl;
	}
	if (f_vv) {
		cout << "before:" << endl;
		Int_vec_print(cout, v1, len);
		cout << endl;
		Int_vec_print(cout, v2, len);
		cout << endl;
		cout << "pivot column " << idx << endl;
	}
	if (v2[idx] == 0) {
		goto after;
	}
	if (v1[idx] == 0) {
		// do a swap:
		for (i = 0; i < len; i++) {
			a = v2[i];
			v2[i] = v1[i];
			v1[i] = a;
		}
		goto after;
	}
	a = F->negate(F->mult(F->inverse(v1[idx]), v2[idx]));
	//cout << "Gauss_step a=" << a << endl;
	for (i = 0; i < len; i++) {
		v2[i] = F->add(F->mult(v1[i], a), v2[i]);
	}
after:
	if (v1[idx] == 0) {
		cout << "linear_algebra::Gauss_step_make_pivot_one after: v1[idx] == 0" << endl;
		exit(1);
	}
	if (v1[idx] != 1) {
		a = v1[idx];
		av = F->inverse(a);
		for (i = 0; i < len; i++) {
			v1[i] = F->mult(av, v1[i]);
		}
	}
	if (f_vv) {
		cout << "linear_algebra::Gauss_step_make_pivot_one after:" << endl;
		Int_vec_print(cout, v1, len);
		cout << endl;
		Int_vec_print(cout, v2, len);
		cout << endl;
	}
	if (f_v) {
		cout << "linear_algebra::Gauss_step_make_pivot_one done" << endl;
	}
}

void linear_algebra::extend_basis(int m, int n, int *Basis,
	int verbose_level)
// Assumes that Basis is n x n, with the first m rows filled in.
// Assumes that Basis has rank m.
// Fills in the bottom n - m rows of Basis to extend to a Basis of F_q^n
// Does not change the first m rows of Basis.
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; // (verbose_level >= 2);
	int *B;
	int *base_cols;
	int *embedding;
	int i, j, rk;

	if (f_v) {
		cout << "linear_algebra::extend_basis" << endl;
	}
	if (f_vv) {
		cout << "matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, Basis, m, n, n, F->log10_of_q);
	}
	Int_vec_zero(Basis + m * n, (n - m) * n);
	B = NEW_int(n * n);
	base_cols = NEW_int(n);
	embedding = NEW_int(n);
	Int_vec_zero(B, n * n);
	Int_vec_copy(Basis, B, m * n);
	rk = base_cols_and_embedding(m, n, B,
		base_cols, embedding, verbose_level);
	if (rk != m) {
		cout << "linear_algebra::extend_basis rk != m" << endl;
		exit(1);
	}
	for (i = rk; i < n; i++) {
		j = embedding[i - rk];
		Basis[i * n + j] = 1;
	}
	FREE_int(B);
	FREE_int(base_cols);
	FREE_int(embedding);

	if (f_v) {
		cout << "linear_algebra::extend_basis done" << endl;
	}
}

int linear_algebra::base_cols_and_embedding(int m, int n, int *A,
	int *base_cols, int *embedding, int verbose_level)
// returns the rank rk of the matrix.
// It also computes base_cols[rk] and embedding[m - rk]
// It leaves A unchanged
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int *B;
	int i, j, rk, idx;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "linear_algebra::base_cols_and_embedding" << endl;
	}
	if (f_vv) {
		cout << "matrix A:" << endl;
		Int_vec_print_integer_matrix_width(cout, A, m, n, n, F->log10_of_q);
	}
	B = NEW_int(m * n);
	Int_vec_copy(A, B, m * n);
	rk = Gauss_simple(B, m, n, base_cols, verbose_level - 3);
	j = 0;
	for (i = 0; i < n; i++) {
		if (!Sorting.int_vec_search(base_cols, rk, i, idx)) {
			embedding[j++] = i;
		}
	}
	if (j != n - rk) {
		cout << "j != n - rk" << endl;
		cout << "j=" << j << endl;
		cout << "rk=" << rk << endl;
		cout << "n=" << n << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "linear_algebra::base_cols_and_embedding" << endl;
		cout << "rk=" << rk << endl;
		cout << "base_cols:" << endl;
		Int_vec_print(cout, base_cols, rk);
		cout << endl;
		cout << "embedding:" << endl;
		Int_vec_print(cout, embedding, n - rk);
		cout << endl;
	}
	FREE_int(B);
	if (f_v) {
		cout << "linear_algebra::base_cols_and_embedding done" << endl;
	}
	return rk;
}

int linear_algebra::Gauss_easy(int *A, int m, int n)
// returns the rank
{
	int *base_cols, rk;

	base_cols = NEW_int(n);
	rk = Gauss_int(A, FALSE, TRUE, base_cols, FALSE, NULL, m, n, n, 0);
	FREE_int(base_cols);
	return rk;
}

int linear_algebra::Gauss_easy_memory_given(int *A,
		int m, int n, int *base_cols)
// returns the rank
{
	int rk;

	//base_cols = NEW_int(n);
	rk = Gauss_int(A, FALSE, TRUE, base_cols, FALSE, NULL, m, n, n, 0);
	//FREE_int(base_cols);
	return rk;
}

int linear_algebra::Gauss_simple(int *A, int m, int n,
		int *base_cols, int verbose_level)
// A[m * n], base_cols[n]
// returns the rank which is the number of entries in base_cols
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "linear_algebra::Gauss_simple before Gauss_int" << endl;
	}
	ret = Gauss_int(A, FALSE, TRUE, base_cols,
			FALSE, NULL, m, n, n, verbose_level);
	if (f_v) {
		cout << "linear_algebra::Gauss_simple after Gauss_int" << endl;
	}
	return ret;
}

void linear_algebra::kernel_columns(int n, int nb_base_cols,
		int *base_cols, int *kernel_cols)
{
	orbiter_kernel_system::Orbiter->Int_vec->complement(base_cols, kernel_cols, n, nb_base_cols);
#if 0
	int i, j, k;

	j = k = 0;
	for (i = 0; i < n; i++) {
		if (j < nb_base_cols && i == base_cols[j]) {
			j++;
			continue;
			}
		kernel_cols[k++] = i;
		}
#endif
}

void linear_algebra::matrix_get_kernel_as_int_matrix(int *M,
	int m, int n, int *base_cols, int nb_base_cols,
	data_structures::int_matrix *kernel, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *K;
	int kernel_m, kernel_n;

	if (f_v) {
		cout << "linear_algebra::matrix_get_kernel_as_int_matrix" << endl;
	}
	K = NEW_int(n * (n - nb_base_cols));
	matrix_get_kernel(M, m, n, base_cols, nb_base_cols,
		kernel_m, kernel_n, K, verbose_level);
	kernel->allocate_and_init(kernel_m, kernel_n, K);
	FREE_int(K);
	if (f_v) {
		cout << "linear_algebra::matrix_get_kernel_as_int_matrix done" << endl;
	}
}

void linear_algebra::matrix_get_kernel(int *M,
	int m, int n, int *base_cols, int nb_base_cols,
	int &kernel_m, int &kernel_n, int *kernel, int verbose_level)
// kernel[n * (n - nb_base_cols)]
// m is not used!
{
	int f_v = (verbose_level >= 1);
	int r, k, i, j, ii, iii, a, b;
	int *kcol;
	int m_one;

	if (f_v) {
		cout << "linear_algebra::matrix_get_kernel" << endl;
	}
	if (kernel == NULL) {
		cout << "linear_algebra::matrix_get_kernel kernel == NULL" << endl;
		exit(1);
	}
	m_one = F->negate(1);
	r = nb_base_cols;
	k = n - r;
	kernel_m = n;
	kernel_n = k;

	kcol = NEW_int(k);

	ii = 0;
	j = 0;
	if (j < r) {
		b = base_cols[j];
	}
	else {
		b = -1;
	}
	for (i = 0; i < n; i++) {
		if (i == b) {
			j++;
			if (j < r) {
				b = base_cols[j];
			}
			else {
				b = -1;
			}
		}
		else {
			kcol[ii] = i;
			ii++;
		}
	}
	if (ii != k) {
		cout << "linear_algebra::matrix_get_kernel ii != k" << endl;
		exit(1);
	}
	//cout << "kcol = " << kcol << endl;
	ii = 0;
	j = 0;
	if (j < r) {
		b = base_cols[j];
	}
	else {
		b = -1;
	}
	for (i = 0; i < n; i++) {
		if (i == b) {
			for (iii = 0; iii < k; iii++) {
				a = kcol[iii];
				kernel[i * kernel_n + iii] = M[j * n + a];
			}
			j++;
			if (j < r) {
				b = base_cols[j];
			}
			else {
				b = -1;
			}
		}
		else {
			for (iii = 0; iii < k; iii++) {
				if (iii == ii) {
					kernel[i * kernel_n + iii] = m_one;
				}
				else {
					kernel[i * kernel_n + iii] = 0;
				}
			}
			ii++;
		}
	}
	if (f_v) {
		cout << "linear_algebra::matrix_get_kernel done" << endl;
	}
	FREE_int(kcol);
}

int linear_algebra::perp(int n, int k, int *A, int *Gram, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *B;
	int *K;
	int *base_cols;
	int nb_base_cols;
	int kernel_m, kernel_n, i, j;

	if (f_v) {
		cout << "linear_algebra::perp" << endl;
	}
	B = NEW_int(n * n);
	K = NEW_int(n * n);
	base_cols = NEW_int(n);
	mult_matrix_matrix(A, Gram, B, k, n, n, 0 /* verbose_level */);

	nb_base_cols = Gauss_int(B,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, k, n, n,
		0 /* verbose_level */);

	if (nb_base_cols != k) {
		cout << "linear_algebra::perp nb_base_cols != k" << endl;
		cout << "need to copy B back to A to be safe." << endl;
		exit(1);
		}

	matrix_get_kernel(B, k, n, base_cols, nb_base_cols,
		kernel_m, kernel_n, K, 0 /* verbose_level */);

	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < n; i++) {
			A[(k + j) * n + i] = K[i * kernel_n + j];
		}
	}
	//cout << "perp, kernel is a " << kernel_m
	// << " by " << kernel_n << " matrix" << endl;
	FREE_int(B);
	FREE_int(K);
	FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::perp done" << endl;
	}
	return nb_base_cols;
}

int linear_algebra::RREF_and_kernel(int n, int k,
		int *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *B;
	int *K;
	int *base_cols;
	int nb_base_cols;
	int kernel_m, kernel_n, i, j, m;

	if (f_v) {
		cout << "linear_algebra::RREF_and_kernel n=" << n
				<< " k=" << k << endl;
	}
#if 0
	if (k > n) {
		m = k;
	}
	else {
		m = n;
	}
#endif
	m = MAXIMUM(k, n);
	B = NEW_int(m * n);
	K = NEW_int(n * n);
	base_cols = NEW_int(n);
	Int_vec_copy(A, B, k * n);
	//mult_matrix_matrix(A, Gram, B, k, n, n);
	if (f_v) {
		cout << "linear_algebra::RREF_and_kernel before Gauss_int" << endl;
	}
	nb_base_cols = Gauss_int(B,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, k, n, n, 0 /* verbose_level */);
	if (f_v) {
		cout << "linear_algebra::RREF_and_kernel after Gauss_int, "
				"rank = " << nb_base_cols << endl;
	}
	Int_vec_copy(B, A, nb_base_cols * n);
	if (f_v) {
		cout << "linear_algebra::RREF_and_kernel "
				"before matrix_get_kernel" << endl;
	}
	matrix_get_kernel(B, k, n, base_cols, nb_base_cols,
		kernel_m, kernel_n, K, 0 /* verbose_level */);
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < n; i++) {
			A[(nb_base_cols + j) * n + i] = K[i * kernel_n + j];
		}
	}
	//cout << "finite_field::RREF_and_kernel,
	// kernel is a " << kernel_m << " by " << kernel_n << " matrix" << endl;
	FREE_int(B);
	FREE_int(K);
	FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::RREF_and_kernel done" << endl;
	}
	return nb_base_cols;
}

int linear_algebra::perp_standard(int n, int k,
		int *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *B;
	int *K;
	int *base_cols;
	int nb_base_cols;

	if (f_v) {
		cout << "linear_algebra::perp_standard" << endl;
	}
	B = NEW_int(n * n);
	K = NEW_int(n * n);
	base_cols = NEW_int(n);
	nb_base_cols = perp_standard_with_temporary_data(n, k, A,
		B, K, base_cols,
		verbose_level - 3);
	FREE_int(B);
	FREE_int(K);
	FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::perp_standard done" << endl;
	}
	return nb_base_cols;
}

int linear_algebra::perp_standard_with_temporary_data(
	int n, int k, int *A,
	int *B, int *K, int *base_cols,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int *B;
	//int *K;
	//int *base_cols;
	int nb_base_cols;
	int kernel_m, kernel_n, i, j;

	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data" << endl;
	}
	//B = NEW_int(n * n);
	//K = NEW_int(n * n);
	//base_cols = NEW_int(n);

	Int_vec_copy(A, B, k * n);
	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data" << endl;
		cout << "B=" << endl;
		Int_matrix_print(B, k, n);
		cout << "finite_field::perp_standard_temporary_data before Gauss_int" << endl;
	}
	nb_base_cols = Gauss_int(B,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, k, n, n,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data after Gauss_int" << endl;
	}
	matrix_get_kernel(B, k, n, base_cols, nb_base_cols,
		kernel_m, kernel_n, K, 0 /* verbose_level */);
	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data "
				"after matrix_get_kernel" << endl;
		cout << "kernel_m = " << kernel_m << endl;
		cout << "kernel_n = " << kernel_n << endl;
	}

	Int_vec_copy(B, A, nb_base_cols * n);

	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < n; i++) {
			A[(nb_base_cols + j) * n + i] = K[i * kernel_n + j];
		}
	}
	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data" << endl;
		cout << "A=" << endl;
		Int_matrix_print(A, n, n);
	}
	//cout << "perp_standard, kernel is a "
	// << kernel_m << " by " << kernel_n << " matrix" << endl;
	//FREE_int(B);
	//FREE_int(K);
	//FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data done" << endl;
	}
	return nb_base_cols;
}

int linear_algebra::intersect_subspaces(int n, int k1,
	int *A, int k2, int *B,
	int &k3, int *intersection, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *AA, *BB, *CC, r1, r2, r3;
	int *B1;
	int *K;
	int *base_cols;

	AA = NEW_int(n * n);
	BB = NEW_int(n * n);
	B1 = NEW_int(n * n);
	K = NEW_int(n * n);
	base_cols = NEW_int(n);
	Int_vec_copy(A, AA, k1 * n);
	Int_vec_copy(B, BB, k2 * n);
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces AA=" << endl;
		Int_vec_print_integer_matrix_width(cout, AA, k1, n, n, 2);
	}
	r1 = perp_standard_with_temporary_data(n, k1,
			AA, B1, K, base_cols, 0);
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces AA=" << endl;
		Int_vec_print_integer_matrix_width(cout, AA, n, n, n, 2);
	}
	if (r1 != k1) {
		cout << "linear_algebra::intersect_subspaces not a base, "
				"rank is too small" << endl;
		cout << "k1=" << k1 << endl;
		cout << "r1=" << r1 << endl;
		exit(1);
	}
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces BB=" << endl;
		Int_vec_print_integer_matrix_width(cout, BB, k2, n, n, 2);
	}
	r2 = perp_standard_with_temporary_data(n, k2, BB, B1, K, base_cols, 0);
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces BB=" << endl;
		Int_vec_print_integer_matrix_width(cout, BB, n, n, n, 2);
	}
	if (r2 != k2) {
		cout << "linear_algebra::intersect_subspaces not a base, "
				"rank is too small" << endl;
		cout << "k2=" << k2 << endl;
		cout << "r2=" << r2 << endl;
		exit(1);
	}
	CC = NEW_int((3 * n) * n);

	Int_vec_copy(AA + k1 * n, CC, (n - k1) * n);
	Int_vec_copy(BB + k2 * n, CC + (n - k1) * n, (n - k2) * n);
	k3 = (n - k1) + (n - k2);
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces CC=" << endl;
		Int_vec_print_integer_matrix_width(cout, CC, k3, n, n, 2);
		cout << "k3=" << k3 << endl;
	}


	k3 = Gauss_easy(CC, k3, n);

	r3 = perp_standard_with_temporary_data(n, k3, CC, B1, K, base_cols, 0);
	Int_vec_copy(CC + k3 * n, intersection, (n - r3) * n);

	FREE_int(AA);
	FREE_int(BB);
	FREE_int(CC);
	FREE_int(B1);
	FREE_int(K);
	FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces n=" << n
				<< " dim A =" << r1 << " dim B =" << r2
				<< " dim intersection =" << n - r3 << endl;
	}
	k3 = n - r3;
	return n - r3;

}

int linear_algebra::n_choose_k_mod_p(int n, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int n1, k1, c, cc = 0, cv, c1 = 1, c2 = 1, i;

	c = 1;
	while (n || k) {
		n1 = n % F->p;
		k1 = k % F->p;
		c1 = 1;
		c2 = 1;
		for (i = 0; i < k1; i++) {
			c1 = F->mult(c1, (n1 - i) % F->p);
			c2 = F->mult(c2, (1 + i) % F->p);
		}
		if (c1 != 0) {
			cv = F->inverse(c2);
			cc = F->mult(c1, cv);
		}
		if (f_vv) {
			cout << "{" << n1 << "\\atop " << k1 << "} mod "
					<< F->p << " = " << cc << endl;
		}
		c = F->mult(c, cc);
		n = (n - n1) / F->p;
		k = (k - k1) / F->p;
	}
	if (f_v) {
		cout << "{" << n << "\\atop " << k << "} mod "
				<< F->p << " = " << endl;
		cout << c << endl;
	}
	return c;
}

void linear_algebra::Dickson_polynomial(int *map, int *coeffs)
// compute the coefficients of a degree q-1 polynomial
// which interpolates a given map
// from F_q to F_q
{
	int i, x, c, xi, a;

	// coeff[0] = map[0]
	coeffs[0] = map[0];

	// the middle coefficients:
	// coeff[i] = - sum_{x \neq 0} g(x) x^{-i}
	for (i = 1; i <= F->q - 2; i++) {
		c = 0;
		for (x = 1; x < F->q; x++) {
			xi = F->inverse(x);
			xi = F->power(xi, i);
			a = F->mult(map[x], xi);
			c = F->add(c, a);
		}
		coeffs[i] = F->negate(c);
	}

	// coeff[q - 1] = - \sum_x map[x]
	c = 0;
	for (x = 0; x < F->q; x++) {
		c = F->add(c, map[x]);
	}
	coeffs[F->q - 1] = F->negate(c);
}

void linear_algebra::projective_action_on_columns_from_the_left(
		int *A, int *M, int m, int n, int *perm,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *AM, i, j;
	data_structures::sorting Sorting;

	AM = NEW_int(m * n);

	if (f_v) {
		cout << "linear_algebra::projective_action_on_columns_from_the_left" << endl;
	}
	if (f_vv) {
		cout << "A:" << endl;
		Int_vec_print_integer_matrix_width(cout, A, m, m, m, 2);
	}
	mult_matrix_matrix(A, M, AM, m, m, n, 0 /* verbose_level */);
	if (f_vv) {
		cout << "M:" << endl;
		Int_vec_print_integer_matrix_width(cout, M, m, n, n, 2);
		//cout << "A * M:" << endl;
		//print_integer_matrix_width(cout, AM, m, n, n, 2);
	}

	for (j = 0; j < n; j++) {
		F->PG_element_normalize_from_front(AM + j,
				n /* stride */, m /* length */);
	}
	if (f_vv) {
		cout << "A*M:" << endl;
		Int_vec_print_integer_matrix_width(cout, AM, m, n, n, 2);
	}

	for (i = 0; i < n; i++) {
		perm[i] = -1;
		for (j = 0; j < n; j++) {
			if (Sorting.int_vec_compare_stride(AM + i, M + j,
					m /* len */, n /* stride */) == 0) {
				perm[i] = j;
				break;
			}
		}
		if (j == n) {
			cout << "linear_algebra::projective_action_on_columns_"
					"from_the_left could not find image" << endl;
			cout << "i=" << i << endl;
			cout << "M:" << endl;
			Int_vec_print_integer_matrix_width(cout, M, m, n, n, 2);
			cout << "A * M:" << endl;
			Int_vec_print_integer_matrix_width(cout, AM, m, n, n, 2);
			exit(1);
		}
	}
	if (f_v) {
		//cout << "column permutation: ";
		combinatorics::combinatorics_domain Combi;

		Combi.perm_print_with_cycle_length(cout, perm, n);
		cout << endl;
	}
	FREE_int(AM);
}

int linear_algebra::evaluate_bilinear_form(int n,
		int *v1, int *v2, int *Gram)
{
	int *v3, a;

	v3 = NEW_int(n);
	mult_matrix_matrix(v1, Gram, v3, 1, n, n, 0 /* verbose_level */);
	a = dot_product(n, v3, v2);
	FREE_int(v3);
	return a;
}

int linear_algebra::evaluate_standard_hyperbolic_bilinear_form(
		int n, int *v1, int *v2)
{
	int a, b, c, n2, i;

	if (ODD(n)) {
		cout << "linear_algebra::evaluate_standard_hyperbolic_bilinear_form "
				"n must be even" << endl;
		exit(1);
	}
	n2 = n >> 1;
	c = 0;
	for (i = 0; i < n2; i++) {
		a = F->mult(v1[2 * i + 0], v2[2 * i + 1]);
		b = F->mult(v1[2 * i + 1], v2[2 * i + 0]);
		c = F->add(c, a);
		c = F->add(c, b);
	}
	return c;
}

int linear_algebra::evaluate_quadratic_form(int n, int nb_terms,
	int *i, int *j, int *coeff, int *x)
{
	int k, xi, xj, a, c, d;

	a = 0;
	for (k = 0; k < nb_terms; k++) {
		xi = x[i[k]];
		xj = x[j[k]];
		c = coeff[k];
		d = F->mult(F->mult(c, xi), xj);
		a = F->add(a, d);
	}
	return a;
}

void linear_algebra::find_singular_vector_brute_force(int n,
	int form_nb_terms,
	int *form_i, int *form_j, int *form_coeff, int *Gram,
	int *vec, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N, a, i;
	int *v1;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "linear_algebra::find_singular_vector_brute_force" << endl;
	}
	v1 = NEW_int(n);
	N = Gg.nb_AG_elements(n, F->q);
	for (i = 2; i < N; i++) {
		Gg.AG_element_unrank(F->q, v1, 1, n, i);
		a = evaluate_quadratic_form(n, form_nb_terms,
				form_i, form_j, form_coeff, v1);
		if (f_v) {
			cout << "v1=";
			Int_vec_print(cout, v1, n);
			cout << endl;
			cout << "form value a=" << a << endl;
		}
		if (a == 0) {
			Int_vec_copy(v1, vec, n);
			goto finish;
		}
	}
	cout << "linear_algebra::find_singular_vector_brute_force "
			"did not find a singular vector" << endl;
	exit(1);

finish:
	FREE_int(v1);
}

void linear_algebra::find_singular_vector(int n, int form_nb_terms,
	int *form_i, int *form_j, int *form_coeff, int *Gram,
	int *vec, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a, b, c, d, r3, x, y, i, k3;
	int *v1, *v2, *v3, *v2_coords, *v3_coords, *intersection;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "linear_algebra::find_singular_vector" << endl;
	}
	if (n < 3) {
		cout << "linear_algebra::find_singular_vector n < 3" << endl;
		exit(1);
	}
	v1 = NEW_int(n * n);
	v2 = NEW_int(n * n);
	v3 = NEW_int(n * n);
	v2_coords = NEW_int(n);
	v3_coords = NEW_int(n);
	intersection = NEW_int(n * n);

	//N = nb_AG_elements(n, q);
	Gg.AG_element_unrank(F->q, v1, 1, n, 1);
	a = evaluate_quadratic_form(n, form_nb_terms,
			form_i, form_j, form_coeff, v1);
	if (f_vv) {
		cout << "v1=";
		Int_vec_print(cout, v1, n);
		cout << endl;
		cout << "form value a=" << a << endl;
	}
	if (a == 0) {
		Int_vec_copy(v1, vec, n);
		goto finish;
	}
	perp(n, 1, v1, Gram, 0 /* verbose_level */);
	if (f_vv) {
		cout << "v1 perp:" << endl;
		Int_vec_print_integer_matrix_width(cout, v1 + n, n - 1, n, n, 2);
	}
	Gg.AG_element_unrank(F->q, v2_coords, 1, n - 1, 1);
	mult_matrix_matrix(v2_coords, v1 + n, v2, 1, n - 1, n,
			0 /* verbose_level */);
	b = evaluate_quadratic_form(n, form_nb_terms,
			form_i, form_j, form_coeff, v2);
	if (f_vv) {
		cout << "vector v2=";
		Int_vec_print(cout, v2, n);
		cout << endl;
		cout << "form value b=" << b << endl;
	}
	if (b == 0) {
		Int_vec_copy(v2, vec, n);
		goto finish;
	}
	perp(n, 1, v2, Gram, 0 /* verbose_level */);
	if (f_vv) {
		cout << "v2 perp:" << endl;
		Int_vec_print_integer_matrix_width(cout, v2 + n, n - 1, n, n, 2);
	}
	r3 = intersect_subspaces(n, n - 1, v1 + n, n - 1, v2 + n,
		k3, intersection, verbose_level);
	if (f_vv) {
		cout << "intersection has dimension " << r3 << endl;
		Int_vec_print_integer_matrix_width(cout, intersection, r3, n, n, 2);
	}
	if (r3 != n - 2) {
		cout << "r3 = " << r3 << " should be " << n - 2 << endl;
		exit(1);
	}
	Gg.AG_element_unrank(F->q, v3_coords, 1, n - 2, 1);
	mult_matrix_matrix(v3_coords, intersection, v3, 1, n - 2, n,
			0 /* verbose_level */);
	c = evaluate_quadratic_form(n, form_nb_terms,
			form_i, form_j, form_coeff, v3);
	if (f_vv) {
		cout << "v3=";
		Int_vec_print(cout, v3, n);
		cout << endl;
		cout << "form value c=" << c << endl;
	}
	if (c == 0) {
		Int_vec_copy(v3, vec, n);
		goto finish;
	}
	if (f_vv) {
		cout << "calling abc2xy" << endl;
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		cout << "c=" << F->negate(c) << endl;
	}
	F->abc2xy(a, b, F->negate(c), x, y, verbose_level);
	if (f_v) {
		cout << "x=" << x << endl;
		cout << "y=" << y << endl;
	}
	scalar_multiply_vector_in_place(x, v1, n);
	scalar_multiply_vector_in_place(y, v2, n);
	for (i = 0; i < n; i++) {
		vec[i] = F->add(F->add(v1[i], v2[i]), v3[i]);
	}
	if (f_vv) {
		cout << "singular vector vec=";
		Int_vec_print(cout, vec, n);
		cout << endl;
	}
	d = evaluate_quadratic_form(n, form_nb_terms,
			form_i, form_j, form_coeff, vec);
	if (d) {
		cout << "is non-singular, error! d=" << d << endl;
		cout << "singular vector vec=";
		Int_vec_print(cout, vec, n);
		cout << endl;
		exit(1);
	}
finish:
	FREE_int(v1);
	FREE_int(v2);
	FREE_int(v3);
	FREE_int(v2_coords);
	FREE_int(v3_coords);
	FREE_int(intersection);
	if (f_v) {
		cout << "linear_algebra::find_singular_vector done" << endl;
	}
}

void linear_algebra::complete_hyperbolic_pair(
	int n, int form_nb_terms,
	int *form_i, int *form_j, int *form_coeff, int *Gram,
	int *vec1, int *vec2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a, b, c;
	int *v0, *v1;

	v0 = NEW_int(n * n);
	v1 = NEW_int(n * n);

	if (f_v) {
		cout << "linear_algebra::complete_hyperbolic_pair" << endl;
	}
	if (f_vv) {
		cout << "vec1=";
		Int_vec_print(cout, vec1, n);
		cout << endl;
		cout << "Gram=" << endl;
		Int_vec_print_integer_matrix_width(cout, Gram, 4, 4, 4, 2);
	}
	mult_matrix_matrix(vec1, Gram, v0, 1, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "v0=";
		Int_vec_print(cout, v0, n);
		cout << endl;
	}
	Int_vec_zero(v1, n);
	for (i = n - 1; i >= 0; i--) {
		if (v0[i]) {
			v1[i] = 1;
			break;
		}
	}
	if (i == -1) {
		cout << "linear_algebra::complete_hyperbolic_pair i == -1" << endl;
		exit(1);
	}
	a = dot_product(n, v0, v1);
#if 0
	int N;

	N = nb_AG_elements(n, q);
	if (f_v) {
		cout << "number of elements in AG(" << n << ","
				<< q << ")=" << N << endl;
	}
	for (i = 1; i < N; i++) {
		if (f_v) {
			cout << "unranking vector " << i << " / "
					<< N << " in the affine geometry" << endl;
		}
		AG_element_unrank(q, v1, 1, n, i);
		if (f_v) {
			cout << "v1=";
			int_vec_print(cout, v1, n);
			cout << endl;
		}
		a = dot_product(n, v0, v1);
		if (f_v) {
			cout << "i=" << i << " trying vector ";
			int_vec_print(cout, v1, n);
			cout << " form value a=" << a << endl;
		}
		if (a)
			break;
	}
	if (i == N) {
		cout << "linear_algebra::complete_hyperbolic_pair "
			"did not find a vector whose dot product is non-zero " << endl;
	}
#endif

	if (a != 1) {
		scalar_multiply_vector_in_place(F->inverse(a), v1, n);
	}
	if (f_vv) {
		cout << "normalized ";
		Int_vec_print(cout, v1, n);
		cout << endl;
	}
	b = evaluate_quadratic_form(n, form_nb_terms,
			form_i, form_j, form_coeff, v1);
	b = F->negate(b);
	for (i = 0; i < n; i++) {
		vec2[i] = F->add(F->mult(b, vec1[i]), v1[i]);
	}
	if (f_vv) {
		cout << "linear_algebra::complete_hyperbolic_pair" << endl;
		cout << "vec2=";
		Int_vec_print(cout, vec2, n);
		cout << endl;
	}
	c = dot_product(n, v0, vec2);
	if (c != 1) {
		cout << "dot product is not 1, error" << endl;
		cout << "c=" << c << endl;
		cout << "vec1=";
		Int_vec_print(cout, vec1, n);
		cout << endl;
		cout << "vec2=";
		Int_vec_print(cout, vec2, n);
		cout << endl;
	}
	FREE_int(v0);
	FREE_int(v1);
	if (f_v) {
		cout << "linear_algebra::complete_hyperbolic_pair done" << endl;
	}

}

void linear_algebra::find_hyperbolic_pair(int n, int form_nb_terms,
	int *form_i, int *form_j, int *form_coeff, int *Gram,
	int *vec1, int *vec2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "linear_algebra::find_hyperbolic_pair" << endl;
	}
	if (n >= 3) {
		find_singular_vector(n, form_nb_terms,
			form_i, form_j, form_coeff, Gram,
			vec1, verbose_level);
	}
	else {
		find_singular_vector_brute_force(n, form_nb_terms,
			form_i, form_j, form_coeff, Gram,
			vec1, verbose_level);
	}
	if (f_vv) {
		cout << "linear_algebra::find_hyperbolic_pair, "
				"found singular vector" << endl;
		Int_vec_print(cout, vec1, n);
		cout << endl;
		cout << "calling complete_hyperbolic_pair" << endl;
	}
	complete_hyperbolic_pair(n, form_nb_terms,
		form_i, form_j, form_coeff, Gram,
		vec1, vec2, verbose_level);
	if (f_v) {
		cout << "linear_algebra::find_hyperbolic_pair done" << endl;
	}
}

void linear_algebra::restrict_quadratic_form_list_coding(
	int k, int n, int *basis,
	int form_nb_terms, int *form_i, int *form_j, int *form_coeff,
	int &restricted_form_nb_terms, int *&restricted_form_i,
	int *&restricted_form_j, int *&restricted_form_coeff,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *C, *D, h, i, j, c;

	if (f_v) {
		cout << "linear_algebra::restrict_quadratic_form_list_coding" << endl;
	}
	C = NEW_int(n * n);
	D = NEW_int(k * k);
	Int_vec_zero(C, n * n);
	for (h = 0; h < form_nb_terms; h++) {
		i = form_i[h];
		j = form_j[h];
		c = form_coeff[h];
		C[i * n + j] = c;
	}
	if (f_vv) {
		cout << "linear_algebra::restrict_quadratic_form_list_coding "
				"C=" << endl;
		Int_vec_print_integer_matrix_width(cout, C, n, n, n, 2);
	}
	restrict_quadratic_form(k, n, basis, C, D, verbose_level - 1);
	if (f_vv) {
		cout << "linear_algebra::restrict_quadratic_form_list_coding "
				"D=" << endl;
		Int_vec_print_integer_matrix_width(cout, D, k, k, k, 2);
	}
	restricted_form_nb_terms = 0;
	restricted_form_i = NEW_int(k * k);
	restricted_form_j = NEW_int(k * k);
	restricted_form_coeff = NEW_int(k * k);
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			c = D[i * k + j];
			if (c == 0) {
				continue;
			}
			restricted_form_i[restricted_form_nb_terms] = i;
			restricted_form_j[restricted_form_nb_terms] = j;
			restricted_form_coeff[restricted_form_nb_terms] = c;
			restricted_form_nb_terms++;
		}
	}
	FREE_int(C);
	FREE_int(D);
	if (f_v) {
		cout << "linear_algebra::restrict_quadratic_form_list_coding done" << endl;
	}
}

void linear_algebra::restrict_quadratic_form(int k, int n,
		int *basis, int *C, int *D,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, lambda, mu, a1, a2, d, c;

	if (f_v) {
		cout << "linear_algebra::restrict_quadratic_form" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::restrict_quadratic_form" << endl;
		Int_vec_print_integer_matrix_width(cout, C, n, n, n, 2);
	}
	Int_vec_zero(D, k * k);
	for (lambda = 0; lambda < k; lambda++) {
		for (mu = 0; mu < k; mu++) {
			d = 0;
			for (i = 0; i < n; i++) {
				for (j = i; j < n; j++) {
					a1 = basis[lambda * n + i];
					a2 = basis[mu * n + j];
					c = C[i * n + j];
					d = F->add(d, F->mult(c, F->mult(a1, a2)));
				}
			}
			if (mu < lambda) {
				D[mu * k + lambda] = F->add(D[mu * k + lambda], d);
			}
			else {
				D[lambda * k + mu] = F->add(D[lambda * k + mu], d);
			}
		}
	}
	if (f_vv) {
		Int_vec_print_integer_matrix_width(cout, D, k, k, k, 2);
	}
	if (f_v) {
		cout << "linear_algebra::restrict_quadratic_form" << endl;
	}
}

int linear_algebra::compare_subspaces_ranked(
	int *set1, int *set2, int size,
	int vector_space_dimension, int verbose_level)
// Compares the span of two sets of vectors.
// returns 0 if equal, 1 if not
// (this is so that it matches to the result of a compare function)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *M1;
	int *M2;
	int *base_cols1;
	int *base_cols2;
	int i;
	int rk1, rk2, r;

	if (f_v) {
		cout << "linear_algebra::compare_subspaces_ranked" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::compare_subspaces_ranked" << endl;
		cout << "set1: ";
		Int_vec_print(cout, set1, size);
		cout << endl;
		cout << "set2: ";
		Int_vec_print(cout, set2, size);
		cout << endl;
	}
	M1 = NEW_int(size * vector_space_dimension);
	M2 = NEW_int(size * vector_space_dimension);
	base_cols1 = NEW_int(vector_space_dimension);
	base_cols2 = NEW_int(vector_space_dimension);
	for (i = 0; i < size; i++) {
		F->PG_element_unrank_modified(
			M1 + i * vector_space_dimension,
			1, vector_space_dimension, set1[i]);
		F->PG_element_unrank_modified(
			M2 + i * vector_space_dimension,
			1, vector_space_dimension, set2[i]);
	}
	if (f_vv) {
		cout << "matrix1:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
			vector_space_dimension, vector_space_dimension, F->log10_of_q);
		cout << "matrix2:" << endl;
		Int_vec_print_integer_matrix_width(cout, M2, size,
			vector_space_dimension, vector_space_dimension, F->log10_of_q);
	}
	rk1 = Gauss_simple(M1, size, vector_space_dimension,
			base_cols1, 0/*int verbose_level*/);
	rk2 = Gauss_simple(M2, size, vector_space_dimension,
			base_cols2, 0/*int verbose_level*/);
	if (f_vv) {
		cout << "after Gauss" << endl;
		cout << "matrix1:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension, F->log10_of_q);
		cout << "rank1=" << rk1 << endl;
		cout << "base_cols1: ";
		Int_vec_print(cout, base_cols1, rk1);
		cout << endl;
		cout << "matrix2:" << endl;
		Int_vec_print_integer_matrix_width(cout, M2, size,
				vector_space_dimension, vector_space_dimension, F->log10_of_q);
		cout << "rank2=" << rk2 << endl;
		cout << "base_cols2: ";
		Int_vec_print(cout, base_cols2, rk2);
		cout << endl;
	}
	if (rk1 != rk2) {
		if (f_vv) {
			cout << "the ranks differ, so the subspaces are not equal, "
					"we return 1" << endl;
			}
		r = 1;
		goto ret;
	}
	for (i = 0; i < rk1; i++) {
		if (base_cols1[i] != base_cols2[i]) {
			if (f_vv) {
				cout << "the base_cols differ in entry " << i
						<< ", so the subspaces are not equal, "
						"we return 1" << endl;
			}
			r = 1;
			goto ret;
		}
	}
	for (i = 0; i < size * vector_space_dimension; i++) {
		if (M1[i] != M2[i]) {
			if (f_vv) {
				cout << "the matrices differ in entry " << i
						<< ", so the subspaces are not equal, "
						"we return 1" << endl;
			}
			r = 1;
			goto ret;
		}
	}
	if (f_vv) {
		cout << "the subspaces are equal, we return 0" << endl;
	}
	r = 0;
ret:
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(base_cols1);
	FREE_int(base_cols2);
	if (f_v) {
		cout << "linear_algebra::compare_subspaces_ranked done" << endl;
	}
	return r;
}

int linear_algebra::compare_subspaces_ranked_with_unrank_function(
	int *set1, int *set2, int size,
	int vector_space_dimension,
	void (*unrank_point_func)(int *v, int rk, void *data),
	void *rank_point_data,
	int verbose_level)
// Compares the span of two sets of vectors.
// returns 0 if equal, 1 if not
// (this is so that it matches to the result of a compare function)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *M1;
	int *M2;
	int *base_cols1;
	int *base_cols2;
	int i;
	int rk1, rk2, r;

	if (f_v) {
		cout << "linear_algebra::compare_subspaces_ranked_with_unrank_function" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::compare_subspaces_ranked_with_unrank_function" << endl;
		cout << "set1: ";
		Int_vec_print(cout, set1, size);
		cout << endl;
		cout << "set2: ";
		Int_vec_print(cout, set2, size);
		cout << endl;
	}
	M1 = NEW_int(size * vector_space_dimension);
	M2 = NEW_int(size * vector_space_dimension);
	base_cols1 = NEW_int(vector_space_dimension);
	base_cols2 = NEW_int(vector_space_dimension);
	for (i = 0; i < size; i++) {
		(*unrank_point_func)(M1 + i * vector_space_dimension,
				set1[i], rank_point_data);
		(*unrank_point_func)(M2 + i * vector_space_dimension,
				set2[i], rank_point_data);
#if 0
		PG_element_unrank_modified(*this, M1 + i * vector_space_dimension,
			1, vector_space_dimension, set1[i]);
		PG_element_unrank_modified(*this, M2 + i * vector_space_dimension,
			1, vector_space_dimension, set2[i]);
#endif
	}
	if (f_vv) {
		cout << "matrix1:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << "matrix2:" << endl;
		Int_vec_print_integer_matrix_width(cout, M2, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
	}
	rk1 = Gauss_simple(M1, size,
			vector_space_dimension, base_cols1,
			0/*int verbose_level*/);
	rk2 = Gauss_simple(M2, size,
			vector_space_dimension, base_cols2,
			0/*int verbose_level*/);
	if (f_vv) {
		cout << "after Gauss" << endl;
		cout << "matrix1:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << "rank1=" << rk1 << endl;
		cout << "base_cols1: ";
		Int_vec_print(cout, base_cols1, rk1);
		cout << endl;
		cout << "matrix2:" << endl;
		Int_vec_print_integer_matrix_width(cout, M2, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << "rank2=" << rk2 << endl;
		cout << "base_cols2: ";
		Int_vec_print(cout, base_cols2, rk2);
		cout << endl;
	}
	if (rk1 != rk2) {
		if (f_vv) {
			cout << "the ranks differ, so the subspaces are not equal, "
					"we return 1" << endl;
		}
		r = 1;
		goto ret;
	}
	for (i = 0; i < rk1; i++) {
		if (base_cols1[i] != base_cols2[i]) {
			if (f_vv) {
				cout << "the base_cols differ in entry " << i
						<< ", so the subspaces are not equal, "
						"we return 1" << endl;
			}
			r = 1;
			goto ret;
		}
	}
	for (i = 0; i < size * vector_space_dimension; i++) {
		if (M1[i] != M2[i]) {
			if (f_vv) {
				cout << "the matrices differ in entry " << i
						<< ", so the subspaces are not equal, "
						"we return 1" << endl;
			}
			r = 1;
			goto ret;
		}
	}
	if (f_vv) {
		cout << "the subspaces are equal, we return 0" << endl;
	}
	r = 0;
ret:
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(base_cols1);
	FREE_int(base_cols2);
	if (f_v) {
		cout << "linear_algebra::compare_subspaces_ranked_with_unrank_function done" << endl;
	}
	return r;
}

int linear_algebra::Gauss_canonical_form_ranked(
	int *set1, int *set2, int size,
	int vector_space_dimension, int verbose_level)
// Computes the Gauss canonical form for the generating set in set1.
// The result is written to set2.
// Returns the rank of the span of the elements in set1.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *M;
	int *base_cols;
	int i;
	int rk;

	if (f_v) {
		cout << "linear_algebra::Gauss_canonical_form_ranked" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::Gauss_canonical_form_ranked" << endl;
		cout << "set1: ";
		Int_vec_print(cout, set1, size);
		cout << endl;
	}
	M = NEW_int(size * vector_space_dimension);
	base_cols = NEW_int(vector_space_dimension);
	for (i = 0; i < size; i++) {
		F->PG_element_unrank_modified(
			M + i * vector_space_dimension,
			1, vector_space_dimension,
			set1[i]);
	}
	if (f_vv) {
		cout << "matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, M, size,
			vector_space_dimension, vector_space_dimension,
			F->log10_of_q);
	}
	rk = Gauss_simple(M, size,
			vector_space_dimension, base_cols,
			0/*int verbose_level*/);
	if (f_vv) {
		cout << "after Gauss" << endl;
		cout << "matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, M, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << "rank=" << rk << endl;
		cout << "base_cols: ";
		Int_vec_print(cout, base_cols, rk);
		cout << endl;
	}

	for (i = 0; i < rk; i++) {
		F->PG_element_rank_modified(
			M + i * vector_space_dimension,
			1, vector_space_dimension,
			set2[i]);
	}


	FREE_int(M);
	FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::Gauss_canonical_form_ranked done" << endl;
	}
	return rk;

}

int linear_algebra::lexleast_canonical_form_ranked(
	int *set1, int *set2, int size,
	int vector_space_dimension, int verbose_level)
// Computes the lexleast generating set the
// subspace spanned by the elements in set1.
// The result is written to set2.
// Returns the rank of the span of the elements in set1.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *M1, *M2;
	int *v;
	int *w;
	int *base_cols;
	int *f_allowed;
	int *basis_vectors;
	int *list_of_ranks;
	int *list_of_ranks_PG;
	int *list_of_ranks_PG_sorted;
	int size_list, idx;
	int *tmp;
	int i, j, h, N, a, sz, Sz;
	int rk;
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "linear_algebra::lexleast_canonical_form_ranked" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::lexleast_canonical_form_ranked" << endl;
		cout << "set1: ";
		Int_vec_print(cout, set1, size);
		cout << endl;
	}
	tmp = NEW_int(vector_space_dimension);
	M1 = NEW_int(size * vector_space_dimension);
	base_cols = NEW_int(vector_space_dimension);
	for (i = 0; i < size; i++) {
		F->PG_element_unrank_modified(M1 + i * vector_space_dimension,
			1, vector_space_dimension, set1[i]);
	}
	if (f_vv) {
		cout << "matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
	}

	rk = Gauss_simple(M1, size, vector_space_dimension,
			base_cols, 0/*int verbose_level*/);
	v = NEW_int(rk);
	w = NEW_int(rk);
	if (f_vv) {
		cout << "after Gauss" << endl;
		cout << "matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << "rank=" << rk << endl;
		cout << "base_cols: ";
		Int_vec_print(cout, base_cols, rk);
		cout << endl;
	}
	N = NT.i_power_j(F->q, rk);
	M2 = NEW_int(N * vector_space_dimension);
	list_of_ranks = NEW_int(N);
	list_of_ranks_PG = NEW_int(N);
	list_of_ranks_PG_sorted = NEW_int(N);
	basis_vectors = NEW_int(rk);
	size_list = 0;
	list_of_ranks_PG[0] = -1;
	for (a = 0; a < N; a++) {
		Gg.AG_element_unrank(F->q, v, 1, rk, a);
		mult_matrix_matrix(v, M1, M2 + a * vector_space_dimension,
			1, rk, vector_space_dimension,
			0 /* verbose_level */);
		list_of_ranks[a] = Gg.AG_element_rank(F->q, M2 + a * vector_space_dimension, 1,
				vector_space_dimension);
		if (a == 0) {
			continue;
		}
		F->PG_element_rank_modified(
				M2 + a * vector_space_dimension, 1,
				vector_space_dimension, list_of_ranks_PG[a]);
		if (!Sorting.int_vec_search(list_of_ranks_PG_sorted,
				size_list, list_of_ranks_PG[a], idx)) {
			for (h = size_list; h > idx; h--) {
				list_of_ranks_PG_sorted[h] = list_of_ranks_PG_sorted[h - 1];
				}
			list_of_ranks_PG_sorted[idx] = list_of_ranks_PG[a];
			size_list++;
		}
	}
	if (f_vv) {
		cout << "expanded matrix with all elements in the space:" << endl;
		Int_vec_print_integer_matrix_width(cout, M2, N,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << "list_of_ranks:" << endl;
		Int_vec_print(cout, list_of_ranks, N);
		cout << endl;
		cout << "list_of_ranks_PG:" << endl;
		Int_vec_print(cout, list_of_ranks_PG, N);
		cout << endl;
		cout << "list_of_ranks_PG_sorted:" << endl;
		Int_vec_print(cout, list_of_ranks_PG_sorted, size_list);
		cout << endl;
	}
	f_allowed = NEW_int(size_list);
	for (i = 0; i < size_list; i++) {
		f_allowed[i] = TRUE;
	}

	sz = 1;
	for (i = 0; i < rk; i++) {
		if (f_vv) {
			cout << "step " << i << " ";
			cout << " list_of_ranks_PG_sorted=";
			Int_vec_print(cout, list_of_ranks_PG_sorted, size_list);
			cout << " ";
			cout << "f_allowed=";
			Int_vec_print(cout, f_allowed, size_list);
			cout << endl;
		}
		for (a = 0; a < size_list; a++) {
			if (f_allowed[a]) {
				break;
			}
		}
		if (f_vv) {
			cout << "choosing a=" << a << " list_of_ranks_PG_sorted[a]="
					<< list_of_ranks_PG_sorted[a] << endl;
		}
		basis_vectors[i] = list_of_ranks_PG_sorted[a];
		F->PG_element_unrank_modified(M1 + i * vector_space_dimension,
			1, vector_space_dimension, basis_vectors[i]);
		Sz = F->q * sz;
		if (f_vv) {
			cout << "step " << i
					<< " basis_vector=" << basis_vectors[i] << " : ";
			Int_vec_print(cout, M1 + i * vector_space_dimension,
					vector_space_dimension);
			cout << " sz=" << sz << " Sz=" << Sz << endl;
		}
		for (h = 0; h < size_list; h++) {
			if (list_of_ranks_PG_sorted[h] == basis_vectors[i]) {
				if (f_vv) {
					cout << "disallowing " << h << endl;
				}
				f_allowed[h] = FALSE;
				break;
			}
		}
		for (j = sz; j < Sz; j++) {
			Gg.AG_element_unrank(F->q, v, 1, i + 1, j);
			if (f_vv) {
				cout << "j=" << j << " v=";
				Int_vec_print(cout, v, i + 1);
				cout << endl;
			}
#if 0
			for (h = 0; h < i + 1; h++) {
				w[i - h] = v[h];
			}
			if (f_v) {
				cout << " w=";
				int_vec_print(cout, w, i + 1);
				cout << endl;
			}
#endif
			mult_matrix_matrix(v/*w*/, M1, tmp, 1, i + 1,
					vector_space_dimension,
					0 /* verbose_level */);
			if (f_vv) {
				cout << " tmp=";
				Int_vec_print(cout, tmp, vector_space_dimension);
				cout << endl;
			}
			F->PG_element_rank_modified(tmp, 1,
					vector_space_dimension, a);
			if (f_vv) {
				cout << "has rank " << a << endl;
			}
			for (h = 0; h < size_list; h++) {
				if (list_of_ranks_PG_sorted[h] == a) {
					if (f_vv) {
						cout << "disallowing " << h << endl;
					}
					f_allowed[h] = FALSE;
					break;
				}
			}
		}
		sz = Sz;
	}
	if (f_vv) {
		cout << "basis_vectors by rank: ";
		Int_vec_print(cout, basis_vectors, rk);
		cout << endl;
	}
	if (f_vv) {
		cout << "basis_vectors by coordinates: " << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << endl;
	}

	for (i = 0; i < rk; i++) {
		F->PG_element_rank_modified(M1 + i * vector_space_dimension,
			1, vector_space_dimension, set2[i]);
	}
	if (f_vv) {
		cout << "basis_vectors by rank again (double check): ";
		Int_vec_print(cout, set2, rk);
		cout << endl;
	}


	FREE_int(tmp);
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(v);
	FREE_int(w);
	FREE_int(base_cols);
	FREE_int(f_allowed);
	FREE_int(list_of_ranks);
	FREE_int(list_of_ranks_PG);
	FREE_int(list_of_ranks_PG_sorted);
	FREE_int(basis_vectors);
	if (f_v) {
		cout << "linear_algebra::lexleast_canonical_form_ranked done" << endl;
	}
	return rk;

}

void linear_algebra::exterior_square(int *An, int *An2, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_algebra::exterior_square" << endl;
	}
	int i, j, k, l, ij, kl;
	int aki, alj, akj, ali;
	int u, v, w;
	int n2;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "linear_algebra::exterior_square input matrix:" << endl;
		Int_matrix_print(An, n, n);
	}


	n2 = (n * (n - 1)) >> 1;
	// (i,j) = row index
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			ij = Combi.ij2k(i, j, n);

			// (k,l) = column index
			for (k = 0; k < n; k++) {
				for (l = k + 1; l < n; l++) {
					kl = Combi.ij2k(k, l, n);


					// a_{k,i}a_{l,j} - a_{k,j}a_{l,i}
					// = matrix entry at position (ij,kl)
#if 0
					aki = An[k * n + i];
					alj = An[l * n + j];
					akj = An[k * n + j];
					ali = An[l * n + i];
#else
					// transposed:
					aki = An[i * n + k];
					alj = An[j * n + l];
					akj = An[j * n + k];
					ali = An[i * n + l];
#endif
					u = F->mult(aki, alj);
					v = F->mult(akj, ali);
					w = F->add(u, F->negate(v));

					// now w is the matrix entry

					An2[ij * n2 + kl] = w;
					} // next l
				} // next k
			} // next j
		} // next i

	if (f_v) {
		cout << "linear_algebra::exterior_square output matrix:" << endl;
		Int_matrix_print(An2, n2, n2);
	}

	if (f_v) {
		cout << "linear_algebra::exterior_square done" << endl;
	}
}

void linear_algebra::lift_to_Klein_quadric(int *A4, int *A6, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_algebra::lift_to_Klein_quadric" << endl;
	}
	int E[36];

	exterior_square(A4, E, 4, verbose_level);

	int Basis1[] = {
			1,0,0,0,0,0,
			0,1,0,0,0,0,
			0,0,1,0,0,0,
			0,0,0,1,0,0,
			0,0,0,0,1,0,
			0,0,0,0,0,1,
	};
	int Basis2[36];
	int Image[36];
	int i;

	for (i = 0; i < 6; i++) {
		F->klein_to_wedge(Basis1 + i * 6, Basis2 + i * 6);
	}

	mult_matrix_matrix(Basis2, E, Image, 6, 6, 6, 0 /* verbose_level*/);
	for (i = 0; i < 6; i++) {
		F->wedge_to_klein(Image + i * 6, A6 + i * 6);
	}
	if (f_v) {
		cout << "linear_algebra::lift_to_Klein_quadric" << endl;
	}

}



}}}


