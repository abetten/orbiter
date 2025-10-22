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
namespace algebra {
namespace linear_algebra {


linear_algebra::linear_algebra()
{
	Record_birth();
	F = NULL;
}

linear_algebra::~linear_algebra()
{
	Record_death();
}

void linear_algebra::init(
		algebra::field_theory::finite_field *F, int verbose_level)
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

void linear_algebra::copy_matrix(
		int *A, int *B, int ma, int na)
{

	Int_vec_copy(A, B, ma * na);
}

void linear_algebra::reverse_columns_of_matrix(
		int *A, int *B, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			B[i * n + j] = A[i * n + (n - 1 - j)];
		}
	}
}
void linear_algebra::reverse_matrix(
		int *A, int *B, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			B[i * n + j] = A[(m - 1 - i) * n + (n - 1 - j)];
		}
	}
}

void linear_algebra::identity_matrix(
		int *A, int n)
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

int linear_algebra::is_identity_matrix(
		int *A, int n)
{
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				if (A[i * n + j] != 1) {
					return false;
				}
			}
			else {
				if (A[i * n + j]) {
					return false;
				}
			}
		}
	}
	return true;
}

int linear_algebra::is_diagonal_matrix(
		int *A, int n)
{
	algebra::basic_algebra::algebra_global Algebra;

	return Algebra.is_diagonal_matrix(A, n);
}

int linear_algebra::is_scalar_multiple_of_identity_matrix(
		int *A, int n, int &scalar)
{
	int i;

	if (!is_diagonal_matrix(A, n)) {
		return false;
	}
	scalar = A[0 * n + 0];
	for (i = 1; i < n; i++) {
		if (A[i * n + i] != scalar) {
			return false;
		}
	}
	return true;
}

void linear_algebra::diagonal_matrix(
		int *A, int n, int alpha)
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

void linear_algebra::matrix_minor(
		int f_semilinear,
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

int linear_algebra::minor_2x2(
		int *Elt, int n, int i, int j, int k, int l,
		int verbose_level)
{
	int aki, alj, akj, ali, u, v, w;

	aki = Elt[k * n + i]; // A->Group_element->element_linear_entry_ij(Elt, k, i, verbose_level); //Elt[k * n + i];
	alj = Elt[l * n + j]; // A->Group_element->element_linear_entry_ij(Elt, l, j, verbose_level); //Elt[l * n + j];
	akj = Elt[k * n + j]; // A->Group_element->element_linear_entry_ij(Elt, k, j, verbose_level); //Elt[k * n + j];
	ali = Elt[l * n + i]; // A->Group_element->element_linear_entry_ij(Elt, l, i, verbose_level); //Elt[l * n + i];
	u = F->mult(aki, alj);
	v = F->mult(akj, ali);
	w = F->add(u, F->negate(v));
	return w;
}

void linear_algebra::wedge_product(
		int *Elt, int *Mtx2,
		int n, int n2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra::wedge_product" << endl;
	}
	int i, j, ij, k, l, kl;
	int w;
	//combinatorics::other::combinatorics_domain Combi;

	for (i = 0, ij = 0; i < n; i++) {
		for (j = i + 1; j < n; j++, ij++) {

			// (i,j) = row index
			//ij = Combi.ij2k(i, j, n);

			for (k = 0, kl = 0; k < n; k++) {
				for (l = k + 1; l < n; l++, kl++) {

					// (k,l) = column index
					//kl = Combi.ij2k(k, l, n);


					// a_{k,i}a_{l,j} - a_{k,j}a_{l,i} = matrix entry
#if 0

					aki = Elt[k * n + i];
					alj = Elt[l * n + j];
					akj = Elt[k * n + j];
					ali = Elt[l * n + i];
					u = F->mult(aki, alj);
					v = F->mult(akj, ali);
					w = F->add(u, F->negate(v));
#endif

					w = minor_2x2(
							Elt, n, i, j, k, l,
							verbose_level - 3);

					//w = element_entry_ijkl(Elt, i, j, k, l, verbose_level - 3);
					// now w is the matrix entry

					Mtx2[ij * n2 + kl] = w;
				}
			}
		}
	}

	if (f_v) {
		cout << "linear_algebra::wedge_product done" << endl;
	}
}

void linear_algebra::mult_vector_from_the_left(
		int *v,
		int *A, int *vA, int m, int n)
// v[m], A[m][n], vA[n]
{
	mult_matrix_matrix(
			v, A, vA,
			1, m, n, 0 /*verbose_level */);
}

void linear_algebra::mult_vector_from_the_right(
		int *A,
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

void linear_algebra::semilinear_matrix_mult(
		int *A, int *B, int *AB, int n)
// (A,f1) * (B,f2) = (A*B^{\varphi^{-f1}},f1+f2 mod F->e)
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
		int *A, int *B, int *AB, int *tmp_B, int n,
		int verbose_level)
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

void linear_algebra::matrix_mult_affine(
		int *A, int *B, int *AB,
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
#if 0
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
#else
	int f1, f2, f12, f1inv;
	int *b1, *b2, *b3;
	int *A1, *A2, *A3;
	int *T;
	int *v;
	number_theory::number_theory_domain NT;

	T = NEW_int(n * n);
	v = NEW_int(n);
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

	Int_vec_copy(b2, v, n);
	vector_frobenius_power_in_place(v, n, f1inv);


	mult_matrix_matrix(b1, T, b3, 1, n, n, 0 /* verbose_level */);
	//vector_frobenius_power_in_place(b3, n, f2);
	add_vector(b3, v, b3, n);

	AB[n * n + n] = f12;
	FREE_int(T);
#endif
}

int linear_algebra::matrix_determinant(
		int *A, int n, int verbose_level)
// too many memory allocations
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
		//if (ODD(i)) { // mistake A Betten 10/26/2024
			eps *= -1;
		//}
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
		true /* f_special */,
		false /*f_complete */, Tmp_basecols,
		true /* f_P */, P, n, n, n, verbose_level - 2);
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



void linear_algebra::projective_action_from_the_right(
		int f_semilinear,
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
		if (f_v) {
			cout << "linear_algebra::projective_action_from_the_right "
					"before semilinear_action_from_the_right"  << endl;
		}
		semilinear_action_from_the_right(v, A, vA, n, verbose_level - 1);
		if (f_v) {
			cout << "linear_algebra::projective_action_from_the_right "
					"after semilinear_action_from_the_right"  << endl;
		}
	}
	else {
		if (f_v) {
			cout << "linear_algebra::projective_action_from_the_right "
					"before mult_vector_from_the_left"  << endl;
		}
		mult_vector_from_the_left(v, A, vA, n, n);
		if (f_v) {
			cout << "linear_algebra::projective_action_from_the_right "
					"after mult_vector_from_the_left"  << endl;
		}
	}
	if (f_v) {
		cout << "linear_algebra::projective_action_from_the_right done"  << endl;
	}
}

void linear_algebra::general_linear_action_from_the_right(
		int f_semilinear,
	int *v, int *A, int *vA, int n,
	int verbose_level)
// same as projective_action_from_the_right
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra::general_linear_action_from_the_right"
				<< endl;
	}
	if (f_semilinear) {
		semilinear_action_from_the_right(v, A, vA, n, verbose_level);
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
		int *v, int *A, int *vA, int n, int verbose_level)
// vA = (v * A)^{p^f}  (where f = A[n *  n])
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra::semilinear_action_from_the_right"
				<< endl;
	}
	int f;

	f = A[n * n];
	if (f_v) {
		cout << "linear_algebra::semilinear_action_from_the_right"
				" A=" << endl;
		Int_matrix_print(A, n, n);
		cout << "frob=" << f << endl;
	}
	if (f_v) {
		cout << "linear_algebra::semilinear_action_from_the_right"
				" input vector = ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	mult_vector_from_the_left(v, A, vA, n, n);
	if (f_v) {
		cout << "linear_algebra::semilinear_action_from_the_right"
				" after multiplication = ";
		Int_vec_print(cout, vA, n);
		cout << endl;
	}
	vector_frobenius_power_in_place(vA, n, f);
	if (f_v) {
		cout << "linear_algebra::semilinear_action_from_the_right"
				" after vector_frobenius_power_in_place = ";
		Int_vec_print(cout, vA, n);
		cout << endl;
	}
	if (f_v) {
		cout << "linear_algebra::semilinear_action_from_the_right "
				"done" << endl;
	}
}

void linear_algebra::semilinear_action_from_the_left(
		int *A, int *v, int *Av, int n)
// Av = (A * v)^{p^f}
{
	int f;

	f = A[n * n];
	mult_vector_from_the_right(A, v, Av, n, n);
	vector_frobenius_power_in_place(Av, n, f);
}

#if 0
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
#endif

void linear_algebra::affine_action_from_the_right(
		int f_semilinear, int *v, int *A, int *vA, int n)
// vA = (v * A + b)^{p^f}
{
	mult_vector_from_the_left(v, A, vA, n, n);

	add_vector(vA, A + n * n, vA, n);

	if (f_semilinear) {
		int f;

		f = A[n * n + n];
		vector_frobenius_power_in_place(vA, n, f);
	}
}

void linear_algebra::zero_vector(
		int *A, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		A[i] = 0;
	}
}

void linear_algebra::all_one_vector(
		int *A, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		A[i] = 1;
	}
}

void linear_algebra::support(
		int *A, int m, int *&support, int &size)
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

void linear_algebra::characteristic_vector(
		int *A, int m, int *set, int size)
{
	int i;

	zero_vector(A, m);
	for (i = 0; i < size; i++) {
		A[set[i]] = 1;
	}
}

int linear_algebra::is_zero_vector(
		int *A, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		if (A[i]) {
			return false;
		}
	}
	return true;
}

void linear_algebra::add_vector(
		int *A, int *B, int *C, int m)
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
		int a, int *A, int b, int *B,
		int c, int *C, int *D, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		D[i] = F->add3(F->mult(a, A[i]), F->mult(b, B[i]), F->mult(c, C[i]));
	}
}

void linear_algebra::negate_vector(
		int *A, int *B, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		B[i] = F->negate(A[i]);
	}
}

void linear_algebra::negate_vector_in_place(
		int *A, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		A[i] = F->negate(A[i]);
	}
}

void linear_algebra::scalar_multiply_vector_in_place(
		int c, int *A, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		A[i] = F->mult(c, A[i]);
	}
}

void linear_algebra::left_normalize_vector_in_place(
		int *A, int m)
{
	int i;

	for (i = 0; i < m; i++) {
		if (A[i]) {
			break;
		}
	}
	if (i == m) {
		cout << "linear_algebra::left_normalize_vector_in_place the zero vector is not allowed" << endl;
		exit(1);
	}

	int c, cv;

	c = A[i];
	cv = F->inverse(c);

	for (i = 0; i < m; i++) {
		A[i] = F->mult(cv, A[i]);
	}
}


void linear_algebra::vector_frobenius_power_in_place(
		int *A, int m, int f)
{
	int i;

	for (i = 0; i < m; i++) {
		A[i] = F->frobenius_power(A[i], f);
	}
}

int linear_algebra::dot_product(
		int len, int *v, int *w)
{
	int i, a = 0, b;

	for (i = 0; i < len; i++) {
		b = F->mult(v[i], w[i]);
		a = F->add(a, b);
	}
	return a;
}

void linear_algebra::transpose_matrix(
		int *A, int *At, int ma, int na)
{
	int i, j;

	for (i = 0; i < ma; i++) {
		for (j = 0; j < na; j++) {
			At[j * ma + i] = A[i * na + j];
		}
	}
}

void linear_algebra::transpose_square_matrix(
		int *A, int *At, int n)
{
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			At[j * n + i] = A[i * n + j];
		}
	}
}

void linear_algebra::transpose_matrix_in_place(
		int *A, int m)
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

void linear_algebra::transform_form_matrix(
		int *A,
		int *Gram, int *new_Gram, int d,
		int verbose_level)
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

int linear_algebra::n_choose_k_mod_p(
		int n, int k, int verbose_level)
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

void linear_algebra::Dickson_polynomial(
		int *map, int *coeffs)
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
	other::data_structures::sorting Sorting;

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
		F->Projective_space_basic->PG_element_normalize_from_front(
				AM + j,
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
			cout << "linear_algebra::projective_action_on_columns_from_the_left "
					"could not find image" << endl;
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
		combinatorics::other_combinatorics::combinatorics_domain Combi;

		Combi.Permutations->perm_print_with_cycle_length(cout, perm, n);
		cout << endl;
	}
	FREE_int(AM);
}

int linear_algebra::evaluate_bilinear_form(
		int n,
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

int linear_algebra::evaluate_quadratic_form(
		int n, int nb_terms,
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

void linear_algebra::find_singular_vector_brute_force(
		int n,
	int form_nb_terms,
	int *form_i, int *form_j, int *form_coeff, int *Gram,
	int *vec, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N, a, i;
	int *v1;
	geometry::other_geometry::geometry_global Gg;

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

void linear_algebra::find_singular_vector(
		int n, int form_nb_terms,
	int *form_i, int *form_j, int *form_coeff, int *Gram,
	int *vec, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a, b, c, d, r3, x, y, i, k3;
	int *v1, *v2, *v3, *v2_coords, *v3_coords, *intersection;
	geometry::other_geometry::geometry_global Gg;

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
		cout << "linear_algebra::complete_hyperbolic_pair vec1=";
		Int_vec_print(cout, vec1, n);
		cout << endl;
		cout << "linear_algebra::complete_hyperbolic_pair Gram=" << endl;
		Int_vec_print_integer_matrix_width(cout, Gram, 4, 4, 4, 2);
	}
	mult_matrix_matrix(vec1, Gram, v0, 1, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "linear_algebra::complete_hyperbolic_pair v0=";
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
		cout << "linear_algebra::complete_hyperbolic_pair normalized ";
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
		cout << "linear_algebra::complete_hyperbolic_pair "
				"dot product is not 1, error" << endl;
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

void linear_algebra::find_hyperbolic_pair(
		int n, int form_nb_terms,
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

void linear_algebra::restrict_quadratic_form(
		int k, int n,
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
		cout << "linear_algebra::restrict_quadratic_form done" << endl;
	}
}


void linear_algebra::exterior_square(
		int *An, int *An2, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_algebra::exterior_square" << endl;
	}
	int i, j, k, l, ij, kl;
	int aki, alj, akj, ali;
	int u, v, w;
	int n2;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

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


void linear_algebra::exterior_square_4x4(
		int *A4, int *A6, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_algebra::exterior_square_4x4" << endl;
	}
	int i, j, k, l, ij, kl;
	int aki, alj, akj, ali;
	int u, v, w;
	int n = 4;
	int n2 = 6;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "linear_algebra::exterior_square_4x4 input matrix:" << endl;
		Int_matrix_print(A4, n, n);
	}

	int Pairs[] = {
			0,1,
			2,3,
			0,2,
			1,3,
			0,3,
			1,2
	};


	// (i,j) = row index

	for (ij = 0; ij < n2; ij++) {
		i = Pairs[ij * 2 + 0];
		j = Pairs[ij * 2 + 1];


		// (k,l) = column index
		for (kl = 0; kl < 6; kl++) {
			k = Pairs[kl * 2 + 0];
			l = Pairs[kl * 2 + 1];


			// a_{k,i}a_{l,j} - a_{k,j}a_{l,i}
			// = matrix entry at position (ij,kl)
#if 0
			aki = An[k * n + i];
			alj = An[l * n + j];
			akj = An[k * n + j];
			ali = An[l * n + i];
#else
			// transposed:
			aki = A4[i * n + k];
			alj = A4[j * n + l];
			akj = A4[j * n + k];
			ali = A4[i * n + l];
#endif
			u = F->mult(aki, alj);
			v = F->mult(akj, ali);
			w = F->add(u, F->negate(v));

			// now w is the matrix entry

			A6[ij * n2 + kl] = w;
		}
	}

	if (f_v) {
		cout << "linear_algebra::exterior_square_4x4 output matrix:" << endl;
		Int_matrix_print(A6, n2, n2);
	}

	if (f_v) {
		cout << "linear_algebra::exterior_square_4x4 done" << endl;
	}
}




void linear_algebra::lift_to_Klein_quadric(
		int *A4, int *A6, int verbose_level)
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

	geometry::other_geometry::geometry_global Geo;

	for (i = 0; i < 6; i++) {
		Geo.klein_to_wedge(F, Basis1 + i * 6, Basis2 + i * 6);
	}

	mult_matrix_matrix(Basis2, E, Image, 6, 6, 6, 0 /* verbose_level*/);
	for (i = 0; i < 6; i++) {
		Geo.wedge_to_klein(F, Image + i * 6, A6 + i * 6);
	}
	if (f_v) {
		cout << "linear_algebra::lift_to_Klein_quadric done" << endl;
	}

}



}}}}



