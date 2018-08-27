// finite_field_linear_algebra.C
//
// Anton Betten
//
// started:  October 23, 2002
// pulled out of finite_field:  July 5 2007




#include "foundations.h"

void finite_field::copy_matrix(INT *A, INT *B, INT ma, INT na)
{

	INT_vec_copy(A, B, ma * na);
}

void finite_field::reverse_matrix(INT *A, INT *B, INT m, INT n)
{
	INT i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			B[i * n + j] = A[(m - 1 - i) * n + (n - 1 - j)];
			}
		}
}

void finite_field::identity_matrix(INT *A, INT n)
{
	INT i, j;
	
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

INT finite_field::is_identity_matrix(INT *A, INT n)
{
	INT i, j;
	
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

INT finite_field::is_diagonal_matrix(INT *A, INT n)
{

	return ::is_diagonal_matrix(A, n);

#if 0
	INT i, j;
	
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				continue;
				}
			else {
				if (A[i * n + j]) {
					return FALSE;
					}
				}
			}
		}
	return TRUE;
#endif
}

INT finite_field::is_scalar_multiple_of_identity_matrix(
		INT *A, INT n, INT &scalar)
{
	INT i;
	
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

void finite_field::diagonal_matrix(INT *A, INT n, INT alpha)
{
	INT i, j;
	
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

void finite_field::matrix_minor(INT f_semilinear,
		INT *A, INT *B, INT n, INT f, INT l)
// initializes B as the l x l minor of A
// (which is n x n) starting from row f.
{
	INT i, j;
	
	if (f + l > n) {
		cout << "finite_field::matrix_minor f + l > n" << endl;
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

void finite_field::mult_matrix(INT *A, INT *B, INT *C,
		INT ma, INT na, INT nb)
{
	INT i, j, k, a, b, c;
	
	for (i = 0; i < ma; i++) {
		for (k = 0; k < nb; k++) {
			c = 0;
			for (j = 0; j < na; j++) {
				a = A[i * na + j];
				b = B[j * nb + k];
				c = add(c, mult(a, b));
				}
			C[i * nb + k] = c;
			}
		}
}

void finite_field::mult_vector_from_the_left(INT *v,
		INT *A, INT *vA, INT m, INT n)
// v[m], A[m][n], vA[n]
{
	INT j, k, a, b, ab, c;
	
	for (j = 0; j < n; j++) {
		c = 0;
		for (k = 0; k < m; k++) {
			a = v[k];
			b = A[k * n + j];
			ab = mult(a, b);
			c = add(c, ab);
			}
		vA[j] = c;
		}
}

void finite_field::mult_vector_from_the_right(INT *A,
		INT *v, INT *Av, INT m, INT n)
// A[m][n], v[n], Av[m]
{
	INT i, k, a, b, ab, c;
	
	for (i = 0; i < m; i++) {
		c = 0;
		for (k = 0; k < n; k++) {
			a = A[i * n + k];
			b = v[k];
			ab = mult(a, b);
			c = add(c, ab);
			}
		Av[i] = c;
		}
}

void finite_field::mult_matrix_matrix_verbose(INT *A, INT *B, INT *C,
		INT m, INT n, INT o, INT verbose_level)
// multiplies C := A * B, where A is m x n and B is n x o, so that C is m by o
// C must already be allocated
{
	INT f_v = (verbose_level >= 1);
	INT i, j, k, a, b;
	
	if (f_v) {
		cout << "finite_field::mult_matrix_matrix_verbose" << endl;
		cout << "A=" << endl;
		INT_matrix_print(A, m, n);
		cout << "B=" << endl;
		INT_matrix_print(B, n, o);
		}
	for (i = 0; i < m; i++) {
		for (j = 0; j < o; j++) {
			a = 0;
			for (k = 0; k < n; k++) {
				b = mult(A[i * n + k], B[k * o + j]);
				a = add(a, b);
				}
			C[i * o + j] = a;
			}
		}
}

void finite_field::mult_matrix_matrix(INT *A, INT *B, INT *C,
		INT m, INT n, INT o)
// multiplies C := A * B,
// where A is m x n and B is n x o, so that C is m by o
// C must already be allocated
{
	INT i, j, k, a, b;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < o; j++) {
			a = 0;
			for (k = 0; k < n; k++) {
				b = mult(A[i * n + k], B[k * o + j]);
				a = add(a, b);
				}
			C[i * o + j] = a;
			}
		}
}

void finite_field::semilinear_matrix_mult(INT *A, INT *B, INT *AB, INT n)
// (A,f1) * (B,f2) = (A*B^{\varphi^{-f1}},f1+f2)
{
	INT i, j, k, a, b, ab, c, f1, f2, f1inv;
	INT *B2;
	
	B2 = NEW_INT(n * n);
	f1 = A[n * n];
	f2 = B[n * n];
	f1inv = irem(-f1, e);
	INT_vec_copy(B, B2, n * n);
	vector_frobenius_power_in_place(B2, n * n, f1inv);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			c = 0;
			for (k = 0; k < n; k++) {
				//cout << "i=" << i << "j=" << j << "k=" << k;
				a = A[i * n + k];
				//cout << "a=A[" << i << "][" << k << "]=" << a;
				b = B2[k * n + j];
				ab = mult(a, b);
				c = add(c, ab);
				//cout << "b=" << b << "ab=" << ab << "c=" << c << endl;
				}
			AB[i * n + j] = c;
			}
		}
	AB[n * n] = irem(f1 + f2, e);
	//vector_frobenius_power_in_place(B, n * n, f1);
	FREE_INT(B2);
}

void finite_field::semilinear_matrix_mult_memory_given(
		INT *A, INT *B, INT *AB, INT *tmp_B, INT n)
// (A,f1) * (B,f2) = (A*B^{\varphi^{-f1}},f1+f2)
{
	INT i, j, k, a, b, ab, c, f1, f2, f1inv;
	INT *B2 = tmp_B;
	
	//B2 = NEW_INT(n * n);
	f1 = A[n * n];
	f2 = B[n * n];
	f1inv = irem(-f1, e);
	INT_vec_copy(B, B2, n * n);
	vector_frobenius_power_in_place(B2, n * n, f1inv);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			c = 0;
			for (k = 0; k < n; k++) {
				//cout << "i=" << i << "j=" << j << "k=" << k;
				a = A[i * n + k];
				//cout << "a=A[" << i << "][" << k << "]=" << a;
				b = B2[k * n + j];
				ab = mult(a, b);
				c = add(c, ab);
				//cout << "b=" << b << "ab=" << ab << "c=" << c << endl;
				}
			AB[i * n + j] = c;
			}
		}
	AB[n * n] = irem(f1 + f2, e);
	//vector_frobenius_power_in_place(B, n * n, f1);
	//FREE_INT(B2);
}

void finite_field::matrix_mult_affine(INT *A, INT *B, INT *AB,
		INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *b1, *b2, *b3;
	INT *A1, *A2, *A3;
	
	if (f_v) {
		cout << "finite_field::matrix_mult_affine" << endl;
		}
	A1 = A;
	A2 = B;
	A3 = AB;
	b1 = A + n * n;
	b2 = B + n * n;
	b3 = AB + n * n;
	if (f_vv) {
		cout << "A1=" << endl;
		INT_matrix_print(A1, n, n);
		cout << "b1=" << endl;
		INT_matrix_print(b1, 1, n);
		cout << "A2=" << endl;
		INT_matrix_print(A2, n, n);
		cout << "b2=" << endl;
		INT_matrix_print(b2, 1, n);
		}
	
	mult_matrix(A1, A2, A3, n, n, n);
	if (f_vv) {
		cout << "A3=" << endl;
		INT_matrix_print(A3, n, n);
		}
	mult_matrix(b1, A2, b3, 1, n, n);
	if (f_vv) {
		cout << "b3=" << endl;
		INT_matrix_print(b3, 1, n);
		}
	add_vector(b3, b2, b3, n);
	if (f_vv) {
		cout << "b3 after adding b2=" << endl;
		INT_matrix_print(b3, 1, n);
		}
	
	if (f_v) {
		cout << "finite_field::matrix_mult_affine done" << endl;
		}
}

void finite_field::semilinear_matrix_mult_affine(
		INT *A, INT *B, INT *AB, INT n)
{
	INT f1, f2, f12, f1inv;
	INT *b1, *b2, *b3;
	INT *A1, *A2, *A3;
	INT *T;
	
	T = NEW_INT(n * n);
	A1 = A;
	A2 = B;
	A3 = AB;
	b1 = A + n * n;
	b2 = B + n * n;
	b3 = AB + n * n;
	
	f1 = A[n * n + n];
	f2 = B[n * n + n];
	f12 = irem(f1 + f2, e);
	f1inv = irem(-f1, e);
	
	INT_vec_copy(A2, T, n * n);
	vector_frobenius_power_in_place(T, n * n, f1inv);
	mult_matrix(A1, T, A3, n, n, n);
	//vector_frobenius_power_in_place(A2, n * n, f1);
	
	mult_matrix(b1, A2, b3, 1, n, n);
	vector_frobenius_power_in_place(b3, n, f2);
	add_vector(b3, b2, b3, n);

	AB[n * n + n] = f12;
	FREE_INT(T);
}

INT finite_field::matrix_determinant(INT *A, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, eps = 1, a, det, det1, det2;
	INT *Tmp, *Tmp1;
	
	if (n == 1) {
		return A[0];
		}
	if (f_v) {
		cout << "determinant of " << endl;
		print_integer_matrix_width(cout, A, n, n, n, 2);
		}
	Tmp = NEW_INT(n * n);
	Tmp1 = NEW_INT(n * n);
	INT_vec_copy(A, Tmp, n * n);

	// search for nonzero element in the first column:
	for (i = 0; i < n; i++) {
		if (Tmp[i * n + 0]) {
			break;
			}
		}
	if (i == n) {
		FREE_INT(Tmp);
		FREE_INT(Tmp1);
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
		det = negate(det);
		}
	if (f_v) {
		cout << "after Gauss " << endl;
		print_integer_matrix_width(cout, Tmp, n, n, n, 2);
		cout << "det= " << det << endl;
		}

	// delete the first row and column and form the matrix
	// Tmp1 of size (n - 1) x (n - 1):
	for (i = 1; i < n; i++) {
		for (j = 1; j < n; j++) {
			Tmp1[(i - 1) * (n - 1) + j - 1] = Tmp[i * n + j];
			}
		}
	if (f_v) {
		cout << "computing determinant of " << endl;
		print_integer_matrix_width(cout, Tmp1, n - 1, n - 1, n - 1, 2);
		}
	det1 = matrix_determinant(Tmp1, n - 1, 0/*verbose_level*/);
	if (f_v) {
		cout << "as " << det1 << endl;
		}
	
	// multiply the pivot element:
	det2 = mult(det, det1);

	FREE_INT(Tmp);
	FREE_INT(Tmp1);
	if (f_v) {
		cout << "determinant is " << det2 << endl;
		}
	
	return det2;
#if 0
	INT *Tmp, *Tmp_basecols, *P, *perm;
	INT rk, det, i, j, eps;
	
	Tmp = NEW_INT(n * n + 1);
	Tmp_basecols = NEW_INT(n);
	P = NEW_INT(n * n);
	perm = NEW_INT(n);

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
	rk = Gauss_INT(Tmp,
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
	
	FREE_INT(Tmp);
	FREE_INT(Tmp_basecols);
	FREE_INT(P);
	FREE_INT(perm);
	return det;
#endif
}

void finite_field::matrix_inverse(INT *A, INT *Ainv, INT n,
		INT verbose_level)
{
	INT *Tmp, *Tmp_basecols;
	
	Tmp = NEW_INT(n * n + 1);
	Tmp_basecols = NEW_INT(n);
	
	matrix_invert(A, Tmp, Tmp_basecols, Ainv, n, verbose_level);
	
	FREE_INT(Tmp);
	FREE_INT(Tmp_basecols);
}

void finite_field::matrix_invert(INT *A, INT *Tmp, INT *Tmp_basecols,
	INT *Ainv, INT n, INT verbose_level)
// Tmp points to n * n + 1 INT's
// Tmp_basecols points to n INT's
{
	INT rk;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "finite_field::matrix_invert" << endl;
		print_integer_matrix_width(cout, A, n, n, n, log10_of_q + 1);
		}
	copy_matrix(A, Tmp, n, n);
	identity_matrix(Ainv, n);
	rk = Gauss_INT(Tmp,
		FALSE /* f_special */, TRUE /*f_complete */, Tmp_basecols,
		TRUE /* f_P */, Ainv, n, n, n, verbose_level - 2);
	if (rk < n) {
		cout << "finite_field::matrix_invert() not invertible" << endl;
		cout << "input matrix:" << endl;
		print_integer_matrix_width(cout, A, n, n, n, log10_of_q + 1);
		cout << "Tmp matrix:" << endl;
		print_integer_matrix_width(cout, Tmp, n, n, n, log10_of_q + 1);
		cout << "rk=" << rk << endl;
		exit(1);
		}
	if (f_v) {
		cout << "the inverse is" << endl;
		print_integer_matrix_width(cout, Ainv, n, n, n, log10_of_q + 1);
		}
}

void finite_field::semilinear_matrix_invert(INT *A,
	INT *Tmp, INT *Tmp_basecols, INT *Ainv, INT n,
	INT verbose_level)
// Tmp points to n * n + 1 INT's
// Tmp_basecols points to n INT's
{
	INT f, finv;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "finite_field::semilinear_matrix_invert" << endl;
		print_integer_matrix_width(cout, A, n, n, n, log10_of_q + 1);
		cout << "frobenius: " << A[n * n] << endl;
		}
	matrix_invert(A, Tmp, Tmp_basecols, Ainv, n, verbose_level - 1);
	f = A[n * n];
	vector_frobenius_power_in_place(Ainv, n * n, f);
	finv = irem(-f, e);
	Ainv[n * n] = finv;
	if (f_v) {
		cout << "the inverse is" << endl;
		print_integer_matrix_width(cout, Ainv, n, n, n, log10_of_q + 1);
		cout << "frobenius: " << Ainv[n * n] << endl;
		}
}

void finite_field::semilinear_matrix_invert_affine(INT *A,
	INT *Tmp, INT *Tmp_basecols, INT *Ainv, INT n,
	INT verbose_level)
// Tmp points to n * n + 1 INT's
// Tmp_basecols points to n INT's
{
	INT f, finv;
	INT *b1, *b2;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "finite_field::semilinear_matrix_invert_affine" << endl;
		print_integer_matrix_width(cout, A, n, n, n, log10_of_q + 1);
		cout << "b: ";
		INT_vec_print(cout, A + n * n, n);
		cout << " frobenius: " << A[n * n + n] << endl;
		}
	b1 = A + n * n;
	b2 = Ainv + n * n;
	matrix_invert(A, Tmp, Tmp_basecols, Ainv, n, verbose_level - 1);
	f = A[n * n + n];
	finv = irem(-f, e);
	vector_frobenius_power_in_place(Ainv, n * n, f);

	mult_matrix(b1, Ainv, b2, 1, n, n);
	
	vector_frobenius_power_in_place(b2, n, finv);

	negate_vector_in_place(b2, n);

	Ainv[n * n + n] = finv;
	if (f_v) {
		cout << "the inverse is" << endl;
		print_integer_matrix_width(cout, Ainv, n, n, n, log10_of_q + 1);
		cout << "b: ";
		INT_vec_print(cout, Ainv + n * n, n);
		cout << " frobenius: " << Ainv[n * n + n] << endl;
		}
}


void finite_field::matrix_invert_affine(INT *A,
	INT *Tmp, INT *Tmp_basecols, INT *Ainv, INT n,
	INT verbose_level)
// Tmp points to n * n + 1 INT's
// Tmp_basecols points to n INT's
{
	INT *b1, *b2;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "finite_field::matrix_invert_affine" << endl;
		print_integer_matrix_width(cout, A, n, n, n, log10_of_q + 1);
		cout << "b: ";
		INT_vec_print(cout, A + n * n, n);
		cout << endl;
		}
	b1 = A + n * n;
	b2 = Ainv + n * n;
	matrix_invert(A, Tmp, Tmp_basecols, Ainv, n, verbose_level - 1);

	mult_matrix(b1, Ainv, b2, 1, n, n);
	
	negate_vector_in_place(b2, n);

	if (f_v) {
		cout << "the inverse is" << endl;
		print_integer_matrix_width(cout, Ainv, n, n, n, log10_of_q + 1);
		cout << "b: ";
		INT_vec_print(cout, Ainv + n * n, n);
		cout << endl;
		}
}


void finite_field::projective_action_from_the_right(INT f_semilinear,
	INT *v, INT *A, INT *vA, INT n,
	INT verbose_level)
// vA = (v * A)^{p^f}  if f_semilinear
// (where f = A[n *  n]),   vA = v * A otherwise
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::projective_action_from_the_right"  << endl;
		}	
	if (f_semilinear) {
		semilinear_action_from_the_right(v, A, vA, n);
		}
	else {
		mult_vector_from_the_left(v, A, vA, n, n);
		}
	if (f_v) {
		cout << "finite_field::projective_action_from_the_right done"  << endl;
		}	
}

void finite_field::general_linear_action_from_the_right(INT f_semilinear,
	INT *v, INT *A, INT *vA, INT n,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::general_linear_action_from_the_right"
				<< endl;
		}	
	if (f_semilinear) {
		semilinear_action_from_the_right(v, A, vA, n);
		}
	else {
		mult_vector_from_the_left(v, A, vA, n, n);
		}
	if (f_v) {
		cout << "finite_field::general_linear_action_from_the_right done"
				<< endl;
		}	
}


void finite_field::semilinear_action_from_the_right(
		INT *v, INT *A, INT *vA, INT n)
// vA = (v * A)^{p^f}  (where f = A[n *  n])
{
	INT f;
	
	f = A[n * n];
	mult_vector_from_the_left(v, A, vA, n, n);
	vector_frobenius_power_in_place(vA, n, f);
}

void finite_field::semilinear_action_from_the_left(
		INT *A, INT *v, INT *Av, INT n)
// Av = A * v^{p^f}
{
	INT f;
	
	f = A[n * n];
	mult_vector_from_the_right(A, v, Av, n, n);
	vector_frobenius_power_in_place(Av, n, f);
}

void finite_field::affine_action_from_the_right(
		INT f_semilinear, INT *v, INT *A, INT *vA, INT n)
// vA = (v * A)^{p^f} + b
{
	mult_vector_from_the_left(v, A, vA, n, n);
	if (f_semilinear) {
		INT f;
	
		f = A[n * n + n];
		vector_frobenius_power_in_place(vA, n, f);
		}
	add_vector(vA, A + n * n, vA, n);
}

void finite_field::zero_vector(INT *A, INT m)
{
	INT i;
	
	for (i = 0; i < m; i++) {
		A[i] = 0;
		}
}

void finite_field::all_one_vector(INT *A, INT m)
{
	INT i;
	
	for (i = 0; i < m; i++) {
		A[i] = 1;
		}
}

void finite_field::support(INT *A, INT m, INT *&support, INT &size)
{
	INT i;
	
	support = NEW_INT(m);
	size = 0;
	for (i = 0; i < m; i++) {
		if (A[i]) {
			support[size++] = i;
			}
		}
}

void finite_field::characteristic_vector(INT *A, INT m, INT *set, INT size)
{
	INT i;
	
	zero_vector(A, m);
	for (i = 0; i < size; i++) {
		A[set[i]] = 1;
		}
}

INT finite_field::is_zero_vector(INT *A, INT m)
{
	INT i;
	
	for (i = 0; i < m; i++) {
		if (A[i]) {
			return FALSE;
			}
		}
	return TRUE;
}

void finite_field::add_vector(INT *A, INT *B, INT *C, INT m)
{
	INT i;
	
	for (i = 0; i < m; i++) {
		C[i] = add(A[i], B[i]);
		}
}

void finite_field::negate_vector(INT *A, INT *B, INT m)
{
	INT i;
	
	for (i = 0; i < m; i++) {
		B[i] = negate(A[i]);
		}
}

void finite_field::negate_vector_in_place(INT *A, INT m)
{
	INT i;
	
	for (i = 0; i < m; i++) {
		A[i] = negate(A[i]);
		}
}

void finite_field::scalar_multiply_vector_in_place(INT c, INT *A, INT m)
{
	INT i;
	
	for (i = 0; i < m; i++) {
		A[i] = mult(c, A[i]);
		}
}

void finite_field::vector_frobenius_power_in_place(INT *A, INT m, INT f)
{
	INT i;
	
	for (i = 0; i < m; i++) {
		A[i] = frobenius_power(A[i], f);
		}
}

INT finite_field::dot_product(INT len, INT *v, INT *w)
{
	INT i, a = 0, b;
	
	for (i = 0; i < len; i++) {
		b = mult(v[i], w[i]);
		a = add(a, b);
		}
	return a;
}

void finite_field::transpose_matrix(INT *A, INT *At, INT ma, INT na)
{
	INT i, j;
	
	for (i = 0; i < ma; i++) {
		for (j = 0; j < na; j++) {
			At[j * ma + i] = A[i * na + j];
			}
		}
}

void finite_field::transpose_matrix_in_place(INT *A, INT m)
{
	INT i, j, a;
	
	for (i = 0; i < m; i++) {
		for (j = i + 1; j < m; j++) {
			a = A[i * m + j];
			A[i * m + j] = A[j * m + i];
			A[j * m + i] = a;
			}
		}
}

void finite_field::invert_matrix(INT *A, INT *A_inv, INT n)
{
	INT i, j, a, rk;
	INT *A_tmp;
	INT *base_cols;
	
	A_tmp = NEW_INT(n * n);
	base_cols = NEW_INT(n);
	
	
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
	INT_vec_copy(A, A_tmp, n * n);
	
	rk = Gauss_INT(A_tmp,
			FALSE /* f_special */,
			TRUE /*f_complete */, base_cols,
		TRUE /* f_P */, A_inv, n, n, n, 0 /* verbose_level */);
	if (rk < n) {
		cout << "finite_field::invert_matrix() "
				"matrix is not invertible, the rank is " << rk << endl;
		exit(1);
		}
	FREE_INT(A_tmp);
	FREE_INT(base_cols);
}

void finite_field::transform_form_matrix(INT *A,
		INT *Gram, INT *new_Gram, INT d)
// computes new_Gram = A * Gram * A^\top
{
	INT *Tmp1, *Tmp2;
	
	Tmp1 = NEW_INT(d * d);
	Tmp2 = NEW_INT(d * d);

	transpose_matrix(A, Tmp1, d, d);
	mult_matrix(A, Gram, Tmp2, d, d, d);
	mult_matrix(Tmp2, Tmp1, new_Gram, d, d, d);

	FREE_INT(Tmp1);
	FREE_INT(Tmp2);
}

INT finite_field::rank_of_matrix(INT *A, INT m, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *B, *base_cols, rk;
	
	if (f_v) {
		cout << "finite_field::rank_of_matrix" << endl;
		}
	B = NEW_INT(m * m);
	base_cols = NEW_INT(m);
	INT_vec_copy(A, B, m * m);
	rk = Gauss_INT(B, FALSE, FALSE, base_cols,
			FALSE, NULL, m, m, m, 0 /* verbose_level */);
	if (f_v) {
		cout << "the matrix ";
		if (f_vv) {
			cout << endl;
			print_integer_matrix_width(cout, A, m, m, m, 2);
			}
		cout << "has rank " << rk << endl;
		}
	FREE_INT(base_cols);
	FREE_INT(B);
	return rk;
}

INT finite_field::rank_of_matrix_memory_given(INT *A,
		INT m, INT *B, INT *base_cols, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT rk;
	
	if (f_v) {
		cout << "finite_field::rank_of_matrix_memory_given" << endl;
		}
	//B = NEW_INT(m * m);
	//base_cols = NEW_INT(m);
	INT_vec_copy(A, B, m * m);
	rk = Gauss_INT(B, FALSE, FALSE, base_cols, FALSE,
			NULL, m, m, m, 0 /* verbose_level */);
	if (f_v) {
		cout << "the matrix ";
		if (f_vv) {
			cout << endl;
			print_integer_matrix_width(cout, A, m, m, m, 2);
			}
		cout << "has rank " << rk << endl;
		}
	//FREE_INT(base_cols);
	//FREE_INT(B);
	return rk;
}

INT finite_field::rank_of_rectangular_matrix(INT *A,
		INT m, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *B, *base_cols, rk;
	
	if (f_v) {
		cout << "finite_field::rank_of_rectangular_matrix" << endl;
		}
	B = NEW_INT(m * n);
	base_cols = NEW_INT(n);
	INT_vec_copy(A, B, m * n);
	rk = Gauss_INT(B, FALSE, FALSE, base_cols, FALSE,
			NULL, m, n, n, 0 /* verbose_level */);
	if (f_v) {
		cout << "the matrix ";
		if (f_vv) {
			cout << endl;
			print_integer_matrix_width(cout, A, m, n, n, 2);
			}
		cout << "has rank " << rk << endl;
		}
	FREE_INT(base_cols);
	FREE_INT(B);
	return rk;
}

INT finite_field::rank_of_rectangular_matrix_memory_given(
		INT *A, INT m, INT n, INT *B, INT *base_cols,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT rk;
	
	if (f_v) {
		cout << "finite_field::rank_of_rectangular_"
				"matrix_memory_given" << endl;
		}
	//B = NEW_INT(m * n);
	//base_cols = NEW_INT(n);
	INT_vec_copy(A, B, m * n);
	rk = Gauss_INT(B, FALSE, FALSE, base_cols, FALSE,
			NULL, m, n, n, 0 /* verbose_level */);
	if (f_v) {
		cout << "the matrix ";
		if (f_vv) {
			cout << endl;
			print_integer_matrix_width(cout, A, m, n, n, 2);
			}
		cout << "has rank " << rk << endl;
		}
	return rk;
}

INT finite_field::rank_and_basecols(INT *A, INT m,
		INT *base_cols, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *B, rk;
	
	if (f_v) {
		cout << "finite_field::rank_and_basecols" << endl;
		}
	B = NEW_INT(m * m);
	INT_vec_copy(A, B, m * m);
	rk = Gauss_INT(B, FALSE, FALSE, base_cols, FALSE,
			NULL, m, m, m, 0 /* verbose_level */);
	if (f_v) {
		cout << "the matrix ";
		if (f_vv) {
			cout << endl;
			print_integer_matrix_width(cout, A, m, m, m, 2);
			}
		cout << "has rank " << rk << endl;
		}
	FREE_INT(B);
	return rk;
}

void finite_field::Gauss_step(INT *v1, INT *v2,
		INT len, INT idx, INT verbose_level)
// afterwards: v2[idx] = 0 and v1,v2 span the same space as before
// v1 is not changed if v1[idx] is nonzero
{
	INT i, a;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "Gauss_step before:" << endl;
		INT_vec_print(cout, v1, len);
		cout << endl;
		INT_vec_print(cout, v2, len);
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
	a = negate(mult(inverse(v1[idx]), v2[idx]));
	//cout << "Gauss_step a=" << a << endl;
	for (i = 0; i < len; i++) {
		v2[i] = add(mult(v1[i], a), v2[i]);
		}
after:
	if (f_v) {
		cout << "Gauss_step after:" << endl;
		INT_vec_print(cout, v1, len);
		cout << endl;
		INT_vec_print(cout, v2, len);
		cout << endl;
		}
}

void finite_field::Gauss_step_make_pivot_one(INT *v1, INT *v2, 
	INT len, INT idx, INT verbose_level)
// afterwards: v2[idx] = 0 and v1,v2 span the same space as before
// v1[idx] is zero
{
	INT i, a, av;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "Gauss_step_make_pivot_one before:" << endl;
		INT_vec_print(cout, v1, len);
		cout << endl;
		INT_vec_print(cout, v2, len);
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
	a = negate(mult(inverse(v1[idx]), v2[idx]));
	//cout << "Gauss_step a=" << a << endl;
	for (i = 0; i < len; i++) {
		v2[i] = add(mult(v1[i], a), v2[i]);
		}
after:
	if (v1[idx] == 0) {
		cout << "Gauss_step_make_pivot_one after: v1[idx] == 0" << endl;
		exit(1);
		}
	if (v1[idx] != 1) {
		a = v1[idx];
		av = inverse(a);
		for (i = 0; i < len; i++) {
			v1[i] = mult(av, v1[i]);
			}
		}
	if (f_v) {
		cout << "Gauss_step_make_pivot_one after:" << endl;
		INT_vec_print(cout, v1, len);
		cout << endl;
		INT_vec_print(cout, v2, len);
		cout << endl;
		}
}

INT finite_field::base_cols_and_embedding(INT m, INT n, INT *A, 
	INT *base_cols, INT *embedding, INT verbose_level)
// returns the rank rk of the matrix.
// It also computes base_cols[rk] and embedding[m - rk]
// It leaves A unchanged
{
	INT f_v = (verbose_level >= 1);
	INT *B;
	INT i, j, rk, idx;

	if (f_v) {
		cout << "finite_field::base_cols_and_embedding" << endl;
		cout << "matrix A:" << endl;
		print_integer_matrix_width(cout, A, m, n, n, log10_of_q);
		}
	B = NEW_INT(m * n);
	INT_vec_copy(A, B, m * n);
	rk = Gauss_simple(B, m, n, base_cols, verbose_level - 3);
	j = 0;
	for (i = 0; i < n; i++) {
		if (!INT_vec_search(base_cols, rk, i, idx)) {
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
	if (f_v) {
		cout << "finite_field::base_cols_and_embedding" << endl;
		cout << "rk=" << rk << endl;
		cout << "base_cols:" << endl;
		INT_vec_print(cout, base_cols, rk);
		cout << endl;
		cout << "embedding:" << endl;
		INT_vec_print(cout, embedding, n - rk);
		cout << endl;
		}
	FREE_INT(B);
	return rk;
}

INT finite_field::Gauss_easy(INT *A, INT m, INT n)
// returns the rank
{
	INT *base_cols, rk;

	base_cols = NEW_INT(n);
	rk = Gauss_INT(A, FALSE, TRUE, base_cols, FALSE, NULL, m, n, n, 0);
	FREE_INT(base_cols);
	return rk;
}

INT finite_field::Gauss_easy_memory_given(INT *A,
		INT m, INT n, INT *base_cols)
// returns the rank
{
	INT rk;

	//base_cols = NEW_INT(n);
	rk = Gauss_INT(A, FALSE, TRUE, base_cols, FALSE, NULL, m, n, n, 0);
	//FREE_INT(base_cols);
	return rk;
}

INT finite_field::Gauss_simple(INT *A, INT m, INT n,
		INT *base_cols, INT verbose_level)
// returns the rank which is the number of entries in base_cols
{
	return Gauss_INT(A, FALSE, TRUE, base_cols,
			FALSE, NULL, m, n, n, verbose_level);
}

INT finite_field::Gauss_INT(INT *A,
	INT f_special, INT f_complete, INT *base_cols,
	INT f_P, INT *P, INT m, INT n, INT Pn, INT verbose_level)
// returns the rank which is the number of entries in base_cols
// A is a m x n matrix,
// P is a m x Pn matrix (if f_P is TRUE)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT rank, i, j, k, jj;
	INT pivot, pivot_inv = 0, a, b, c, z, f;
	
	if (f_v) {
		cout << "Gauss algorithm for matrix:" << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_tables();
		}
	i = 0;
	for (j = 0; j < n; j++) {
		if (f_vv) {
			cout << "j=" << j << endl;
			}
		/* search for pivot element: */
		for (k = i; k < m; k++) {
			if (A[k * n + j]) {
				if (f_vv) {
					cout << "i=" << i << " pivot found in "
							<< k << "," << j << endl;
					}
				// pivot element found: 
				if (k != i) {
					for (jj = j; jj < n; jj++) {
						INT_swap(A[i * n + jj], A[k * n + jj]);
						}
					if (f_P) {
						for (jj = 0; jj < Pn; jj++) {
							INT_swap(P[i * Pn + jj], P[k * Pn + jj]);
							}
						}
					}
				break;
				} // if != 0 
			} // next k
		
		if (k == m) { // no pivot found 
			if (f_vv) {
				cout << "no pivot found" << endl;
				}
			continue; // increase j, leave i constant
			}
		
		if (f_vv) {
			cout << "row " << i << " pivot in row "
					<< k << " colum " << j << endl;
			}
		
		base_cols[i] = j;
		//if (FALSE) {
		//	cout << "."; cout.flush();
		//	}

		pivot = A[i * n + j];
		if (f_vv) {
			cout << "pivot=" << pivot << endl;
			}
		//pivot_inv = inv_table[pivot];
		pivot_inv = inverse(pivot);
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv="
					<< pivot_inv << endl;
			}
		if (!f_special) {
			// make pivot to 1: 
			for (jj = j; jj < n; jj++) {
				A[i * n + jj] = mult(A[i * n + jj], pivot_inv);
				}
			if (f_P) {
				for (jj = 0; jj < Pn; jj++) {
					P[i * Pn + jj] = mult(P[i * Pn + jj], pivot_inv);
					}
				}
			if (f_vv) {
				cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv 
					<< " made to one: " << A[i * n + j] << endl;
				}
			if (f_vvv) {
				print_integer_matrix_width(cout, A, m, n, n, 5);
				}
			}
		
		// do the gaussian elimination: 

		if (f_vv) {
			cout << "doing elimination in column " << j << " from row "
					<< i + 1 << " to row " << m - 1 << ":" << endl;
			}
		for (k = i + 1; k < m; k++) {
			if (f_vv) {
				cout << "k=" << k << endl;
				}
			z = A[k * n + j];
			if (z == 0) {
				continue;
				}
			if (f_special) {
				f = mult(z, pivot_inv);
				}
			else {
				f = z;
				}
			f = negate(f);
			A[k * n + j] = 0;
			if (f_vv) {
				cout << "eliminating row " << k << endl;
				}
			for (jj = j + 1; jj < n; jj++) {
				a = A[i * n + jj];
				b = A[k * n + jj];
				// c := b + f * a
				//    = b - z * a              if !f_special 
				//      b - z * pivot_inv * a  if f_special 
				c = mult(f, a);
				c = add(c, b);
				A[k * n + jj] = c;
				if (f_vv) {
					cout << A[k * n + jj] << " ";
					}
				}
			if (f_P) {
				for (jj = 0; jj < Pn; jj++) {
					a = P[i * Pn + jj];
					b = P[k * Pn + jj];
					// c := b - z * a
					c = mult(f, a);
					c = add(c, b);
					P[k * Pn + jj] = c;
					}
				}
			if (f_vv) {
				cout << endl;
				}
			if (f_vvv) {
				cout << "A=" << endl;
				print_integer_matrix_width(cout, A, m, n, n, 5);
				}
			}
		i++;
		if (f_vv) {
			cout << "A=" << endl;
			print_integer_matrix_width(cout, A, m, n, n, 5);
			//print_integer_matrix(cout, A, m, n);
			if (f_P) {
				cout << "P=" << endl;
				print_integer_matrix(cout, P, m, Pn);
				}
			}
		} // next j 
	rank = i;
	if (f_complete) {
		//if (FALSE) {
		//	cout << ";"; cout.flush();
		//	}
		for (i = rank - 1; i >= 0; i--) {
			if (f_v) {
				cout << "."; cout.flush();
				}
			j = base_cols[i];
			if (!f_special) {
				a = A[i * n + j];
				}
			else {
				pivot = A[i * n + j];
				pivot_inv = inverse(pivot);
				}
			// do the gaussian elimination in the upper part: 
			for (k = i - 1; k >= 0; k--) {
				z = A[k * n + j];
				if (z == 0) {
					continue;
					}
				A[k * n + j] = 0;
				for (jj = j + 1; jj < n; jj++) {
					a = A[i * n + jj];
					b = A[k * n + jj];
					if (f_special) {
						a = mult(a, pivot_inv);
						}
					c = mult(z, a);
					c = negate(c);
					c = add(c, b);
					A[k * n + jj] = c;
					}
				if (f_P) {
					for (jj = 0; jj < Pn; jj++) {
						a = P[i * Pn + jj];
						b = P[k * Pn + jj];
						if (f_special) {
							a = mult(a, pivot_inv);
							}
						c = mult(z, a);
						c = negate(c);
						c = add(c, b);
						P[k * Pn + jj] = c;
						}
					}
				} // next k
			} // next i
		}
	if (f_v) { 
		cout << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_integer_matrix(cout, A, rank, n);
		cout << "the rank is " << rank << endl;
		}
	return rank;
}

INT finite_field::Gauss_INT_with_pivot_strategy(INT *A, 
	INT f_special, INT f_complete, INT *pivot_perm, 
	INT m, INT n, 
	INT (*find_pivot_function)(INT *A, INT m, INT n, INT r,
			INT *pivot_perm, void *data),
	void *find_pivot_data,  
	INT verbose_level)
// returns the rank which is the number of entries in pivots
// A is a m x n matrix
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT rank, i, j, k, jj;
	INT pivot, pivot_inv = 0, a, b, c, z, f, pi;
	
	if (f_v) {
		cout << "finite_field::Gauss_INT_with_pivot_strategy "
				"Gauss algorithm for matrix:" << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_tables();
		}
	for (i = 0; i < m; i++) {
		if (f_vv) {
			cout << "i=" << i << endl;
			}

		j = (*find_pivot_function)(A, m, n, i, pivot_perm, find_pivot_data);

		if (j == -1) {
			break;
			}

		pi = pivot_perm[i];
		pivot_perm[i] = j;
		pivot_perm[j] = pi;

		// search for pivot element in column j from row i down: 
		for (k = i; k < m; k++) {
			if (A[k * n + j]) {
				break;
				} // if != 0 
			} // next k
		
		if (k == m) { // no pivot found 
			if (f_vv) {
				cout << "finite_field::Gauss_INT_with_pivot_strategy "
						"no pivot found in column " << j << endl;
				}
			exit(1);
			}
		
		if (f_vv) {
			cout << "row " << i << " pivot in row " << k
					<< " colum " << j << endl;
			}
		




		// pivot element found in row k, check if we need to swap rows:
		if (k != i) {
			for (jj = 0; jj < n; jj++) {
				INT_swap(A[i * n + jj], A[k * n + jj]);
				}
			}


		// now, pivot is in row i, column j :

		pivot = A[i * n + j];
		if (f_vv) {
			cout << "pivot=" << pivot << endl;
			}
		pivot_inv = inverse(pivot);
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv << endl;
			}
		if (!f_special) {
			// make pivot to 1: 
			for (jj = 0; jj < n; jj++) {
				A[i * n + jj] = mult(A[i * n + jj], pivot_inv);
				}
			if (f_vv) {
				cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv 
					<< " made to one: " << A[i * n + j] << endl;
				}
			if (f_vvv) {
				print_integer_matrix_width(cout, A, m, n, n, 5);
				}
			}
		
		// do the gaussian elimination: 

		if (f_vv) {
			cout << "doing elimination in column " << j << " from row "
					<< i + 1 << " down to row " << m - 1 << ":" << endl;
			}
		for (k = i + 1; k < m; k++) {
			if (f_vv) {
				cout << "k=" << k << endl;
				}
			z = A[k * n + j];
			if (z == 0) {
				continue;
				}
			if (f_special) {
				f = mult(z, pivot_inv);
				}
			else {
				f = z;
				}
			f = negate(f);
			//A[k * n + j] = 0;
			if (f_vv) {
				cout << "eliminating row " << k << endl;
				}
			for (jj = 0; jj < n; jj++) {
				a = A[i * n + jj];
				b = A[k * n + jj];
				// c := b + f * a
				//    = b - z * a              if !f_special 
				//      b - z * pivot_inv * a  if f_special 
				c = mult(f, a);
				c = add(c, b);
				A[k * n + jj] = c;
				if (f_vv) {
					cout << A[k * n + jj] << " ";
					}
				}
			if (f_vv) {
				cout << endl;
				}
			if (f_vvv) {
				cout << "A=" << endl;
				print_integer_matrix_width(cout, A, m, n, n, 5);
				}
			}
		i++;
		if (f_vv) {
			cout << "A=" << endl;
			print_integer_matrix_width(cout, A, m, n, n, 5);
			//print_integer_matrix(cout, A, m, n);
			}
		} // next j 
	rank = i;
	if (f_complete) {
		for (i = rank - 1; i >= 0; i--) {
			j = pivot_perm[i];
			if (!f_special) {
				a = A[i * n + j];
				}
			else {
				pivot = A[i * n + j];
				pivot_inv = inverse(pivot);
				}
			// do the gaussian elimination in the upper part: 
			for (k = i - 1; k >= 0; k--) {
				z = A[k * n + j];
				if (z == 0) {
					continue;
					}
				//A[k * n + j] = 0;
				for (jj = 0; jj < n; jj++) {
					a = A[i * n + jj];
					b = A[k * n + jj];
					if (f_special) {
						a = mult(a, pivot_inv);
						}
					c = mult(z, a);
					c = negate(c);
					c = add(c, b);
					A[k * n + jj] = c;
					}
				} // next k
			} // next i
		}
	if (f_v) { 
		cout << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_integer_matrix(cout, A, rank, n);
		cout << "the rank is " << rank << endl;
		}
	return rank;
}

void finite_field::Gauss_INT_with_given_pivots(INT *A, 
	INT f_special, INT f_complete, INT *pivots, INT nb_pivots, 
	INT m, INT n, 
	INT verbose_level)
// A is a m x n matrix
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT i, j, k, jj;
	INT pivot, pivot_inv = 0, a, b, c, z, f;
	
	if (f_v) {
		cout << "finite_field::Gauss_INT_with_given_pivots "
				"Gauss algorithm for matrix:" << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		cout << "pivots: ";
		::INT_vec_print(cout, pivots, nb_pivots);
		cout << endl;
		//print_tables();
		}
	for (i = 0; i < nb_pivots; i++) {
		if (f_vv) {
			cout << "i=" << i << endl;
			}

		j = pivots[i];

		// search for pivot element in column j from row i down: 
		for (k = i; k < m; k++) {
			if (A[k * n + j]) {
				break;
				} // if != 0 
			} // next k
		
		if (k == m) { // no pivot found 
			if (f_vv) {
				cout << "finite_field::Gauss_INT_with_given_pivots "
						"no pivot found in column " << j << endl;
				}
			exit(1);
			}
		
		if (f_vv) {
			cout << "row " << i << " pivot in row " << k
					<< " colum " << j << endl;
			}
		




		// pivot element found in row k, check if we need to swap rows:
		if (k != i) {
			for (jj = 0; jj < n; jj++) {
				INT_swap(A[i * n + jj], A[k * n + jj]);
				}
			}


		// now, pivot is in row i, column j :

		pivot = A[i * n + j];
		if (f_vv) {
			cout << "pivot=" << pivot << endl;
			}
		pivot_inv = inverse(pivot);
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv << endl;
			}
		if (!f_special) {
			// make pivot to 1: 
			for (jj = 0; jj < n; jj++) {
				A[i * n + jj] = mult(A[i * n + jj], pivot_inv);
				}
			if (f_vv) {
				cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv 
					<< " made to one: " << A[i * n + j] << endl;
				}
			if (f_vvv) {
				print_integer_matrix_width(cout, A, m, n, n, 5);
				}
			}
		
		// do the gaussian elimination: 

		if (f_vv) {
			cout << "doing elimination in column " << j << " from row "
					<< i + 1 << " down to row " << m - 1 << ":" << endl;
			}
		for (k = i + 1; k < m; k++) {
			if (f_vv) {
				cout << "k=" << k << endl;
				}
			z = A[k * n + j];
			if (z == 0) {
				continue;
				}
			if (f_special) {
				f = mult(z, pivot_inv);
				}
			else {
				f = z;
				}
			f = negate(f);
			//A[k * n + j] = 0;
			if (f_vv) {
				cout << "eliminating row " << k << endl;
				}
			for (jj = 0; jj < n; jj++) {
				a = A[i * n + jj];
				b = A[k * n + jj];
				// c := b + f * a
				//    = b - z * a              if !f_special 
				//      b - z * pivot_inv * a  if f_special 
				c = mult(f, a);
				c = add(c, b);
				A[k * n + jj] = c;
				if (f_vv) {
					cout << A[k * n + jj] << " ";
					}
				}
			if (f_vv) {
				cout << endl;
				}
			if (f_vvv) {
				cout << "A=" << endl;
				print_integer_matrix_width(cout, A, m, n, n, 5);
				}
			}
		if (f_vv) {
			cout << "A=" << endl;
			print_integer_matrix_width(cout, A, m, n, n, 5);
			//print_integer_matrix(cout, A, m, n);
			}
		} // next j 
	if (f_complete) {
		for (i = nb_pivots - 1; i >= 0; i--) {
			j = pivots[i];
			if (!f_special) {
				a = A[i * n + j];
				}
			else {
				pivot = A[i * n + j];
				pivot_inv = inverse(pivot);
				}
			// do the gaussian elimination in the upper part: 
			for (k = i - 1; k >= 0; k--) {
				z = A[k * n + j];
				if (z == 0) {
					continue;
					}
				//A[k * n + j] = 0;
				for (jj = 0; jj < n; jj++) {
					a = A[i * n + jj];
					b = A[k * n + jj];
					if (f_special) {
						a = mult(a, pivot_inv);
						}
					c = mult(z, a);
					c = negate(c);
					c = add(c, b);
					A[k * n + jj] = c;
					}
				} // next k
			} // next i
		}
	if (f_v) { 
		cout << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_integer_matrix(cout, A, rank, n);
		}
}

void finite_field::kernel_columns(INT n, INT nb_base_cols,
		INT *base_cols, INT *kernel_cols)
{
	INT_vec_complement(base_cols, kernel_cols, n, nb_base_cols);
#if 0
	INT i, j, k;
	
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

void finite_field::matrix_get_kernel_as_INT_matrix(INT *M,
	INT m, INT n, INT *base_cols, INT nb_base_cols,
	INT_matrix *kernel)
{
	INT *K;
	INT kernel_m, kernel_n;

	K = NEW_INT(n * (n - nb_base_cols));
	matrix_get_kernel(M, m, n, base_cols, nb_base_cols, 
		kernel_m, kernel_n, K);
	kernel->allocate_and_init(kernel_m, kernel_n, K);
	FREE_INT(K);
}

void finite_field::matrix_get_kernel(INT *M,
	INT m, INT n, INT *base_cols, INT nb_base_cols,
	INT &kernel_m, INT &kernel_n, INT *kernel)
	// kernel must point to the appropriate amount of memory!
	// (at least n * (n - nb_base_cols) INT's)
	// m is not used!
{
	INT r, k, i, j, ii, iii, a, b;
	INT *kcol;
	INT m_one;
	
	if (kernel == NULL) {
		cout << "finite_field::matrix_get_kernel kernel == NULL" << endl;
		exit(1);
		}
	m_one = negate(1);
	r = nb_base_cols;
	k = n - r;
	kernel_m = n;
	kernel_n = k;
	
	kcol = NEW_INT(k);
	
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
		cout << "finite_field::matrix_get_kernel ii != k" << endl;
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
	FREE_INT(kcol);
}

INT finite_field::perp(INT n, INT k, INT *A, INT *Gram)
{
	INT *B;
	INT *K;
	INT *base_cols;
	INT nb_base_cols;
	INT kernel_m, kernel_n, i, j;
	
	B = NEW_INT(n * n);
	K = NEW_INT(n * n);
	base_cols = NEW_INT(n);
	mult_matrix_matrix(A, Gram, B, k, n, n);
	nb_base_cols = Gauss_INT(B,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, k, n, n,
		0 /* verbose_level */);
	if (nb_base_cols != k) {
		cout << "finite_field::perp nb_base_cols != k" << endl;
		cout << "need to copy B back to A to be safe." << endl;
		exit(1);
		}
	matrix_get_kernel(B, k, n, base_cols, nb_base_cols, 
		kernel_m, kernel_n, K);
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < n; i++) {
			A[(k + j) * n + i] = K[i * kernel_n + j];
			}
		}
	//cout << "perp, kernel is a " << kernel_m
	// << " by " << kernel_n << " matrix" << endl;
	FREE_INT(B);
	FREE_INT(K);
	FREE_INT(base_cols);
	return nb_base_cols;
}

INT finite_field::RREF_and_kernel(INT n, INT k,
		INT *A, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *B;
	INT *K;
	INT *base_cols;
	INT nb_base_cols;
	INT kernel_m, kernel_n, i, j, m;
	
	if (f_v) {
		cout << "finite_field::RREF_and_kernel n=" << n
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
	B = NEW_INT(m * n);
	K = NEW_INT(n * n);
	base_cols = NEW_INT(n);
	INT_vec_copy(A, B, k * n);
	//mult_matrix_matrix(A, Gram, B, k, n, n);
	if (f_v) {
		cout << "finite_field::RREF_and_kernel before Gauss_INT" << endl;
		}
	nb_base_cols = Gauss_INT(B,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, k, n, n, verbose_level);
	if (f_v) {
		cout << "finite_field::RREF_and_kernel after Gauss_INT, "
				"rank = " << nb_base_cols << endl;
		}
	INT_vec_copy(B, A, nb_base_cols * n);
	if (f_v) {
		cout << "finite_field::RREF_and_kernel "
				"before matrix_get_kernel" << endl;
		}
	matrix_get_kernel(B, k, n, base_cols, nb_base_cols, 
		kernel_m, kernel_n, K);
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < n; i++) {
			A[(nb_base_cols + j) * n + i] = K[i * kernel_n + j];
			}
		}
	//cout << "finite_field::RREF_and_kernel,
	// kernel is a " << kernel_m << " by " << kernel_n << " matrix" << endl;
	FREE_INT(B);
	FREE_INT(K);
	FREE_INT(base_cols);
	if (f_v) {
		cout << "finite_field::RREF_and_kernel done" << endl;
		}
	return nb_base_cols;
}

INT finite_field::perp_standard(INT n, INT k,
		INT *A, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *B;
	INT *K;
	INT *base_cols;
	INT nb_base_cols;

	if (f_v) {
		cout << "finite_field::perp_standard" << endl;
		}
	B = NEW_INT(n * n);
	K = NEW_INT(n * n);
	base_cols = NEW_INT(n);
	nb_base_cols = perp_standard_with_temporary_data(n, k, A, 
		B, K, base_cols, 
		verbose_level);
	FREE_INT(B);
	FREE_INT(K);
	FREE_INT(base_cols);
	if (f_v) {
		cout << "finite_field::perp_standard done" << endl;
		}
	return nb_base_cols;
}

INT finite_field::perp_standard_with_temporary_data(
	INT n, INT k, INT *A,
	INT *B, INT *K, INT *base_cols, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT *B;
	//INT *K;
	//INT *base_cols;
	INT nb_base_cols;
	INT kernel_m, kernel_n, i, j;
	
	if (f_v) {
		cout << "finite_field::perp_standard_temporary_data" << endl;
		}
	//B = NEW_INT(n * n);
	//K = NEW_INT(n * n);
	//base_cols = NEW_INT(n);

	INT_vec_copy(A, B, k * n);
	if (f_v) {
		cout << "finite_field::perp_standard" << endl;
		cout << "B=" << endl;
		INT_matrix_print(B, k, n);
		cout << "finite_field::perp_standard before Gauss_INT" << endl;
		}
	nb_base_cols = Gauss_INT(B,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, k, n, n,
		verbose_level);
	if (f_v) {
		cout << "finite_field::perp_standard after Gauss_INT" << endl;
		}
	matrix_get_kernel(B, k, n, base_cols, nb_base_cols, 
		kernel_m, kernel_n, K);
	if (f_v) {
		cout << "finite_field::perp_standard "
				"after matrix_get_kernel" << endl;
		cout << "kernel_m = " << kernel_m << endl;
		cout << "kernel_n = " << kernel_n << endl;
		}
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < n; i++) {
			A[(k + j) * n + i] = K[i * kernel_n + j];
			}
		}
	if (f_v) {
		cout << "finite_field::perp_standard" << endl;
		cout << "A=" << endl;
		INT_matrix_print(A, n, n);
		}
	//cout << "perp_standard, kernel is a "
	// << kernel_m << " by " << kernel_n << " matrix" << endl;
	//FREE_INT(B);
	//FREE_INT(K);
	//FREE_INT(base_cols);
	if (f_v) {
		cout << "finite_field::perp_standard_temporary_data done" << endl;
		}
	return nb_base_cols;
}

INT finite_field::intersect_subspaces(INT n, INT k1,
	INT *A, INT k2, INT *B,
	INT &k3, INT *intersection, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *AA, *BB, *CC, r1, r2, r3;
	INT *B1;
	INT *K;
	INT *base_cols;
	
	AA = NEW_INT(n * n);
	BB = NEW_INT(n * n);
	B1 = NEW_INT(n * n);
	K = NEW_INT(n * n);
	base_cols = NEW_INT(n);
	INT_vec_copy(A, AA, k1 * n);
	INT_vec_copy(B, BB, k2 * n);
	if (f_v) {
		cout << "finite_field::intersect_subspaces AA=" << endl;
		print_integer_matrix_width(cout, AA, k1, n, n, 2);
		}
	r1 = perp_standard_with_temporary_data(n, k1,
			AA, B1, K, base_cols, 0);
	if (f_v) {
		cout << "finite_field::intersect_subspaces AA=" << endl;
		print_integer_matrix_width(cout, AA, n, n, n, 2);
		}
	if (r1 != k1) {
		cout << "finite_field::intersect_subspaces not a base, "
				"rank is too small" << endl;
		cout << "k1=" << k1 << endl;
		cout << "r1=" << r1 << endl;
		exit(1);
		}
	if (f_v) {
		cout << "finite_field::intersect_subspaces BB=" << endl;
		print_integer_matrix_width(cout, BB, k2, n, n, 2);
		}
	r2 = perp_standard_with_temporary_data(n, k2, BB, B1, K, base_cols, 0);
	if (f_v) {
		cout << "finite_field::intersect_subspaces BB=" << endl;
		print_integer_matrix_width(cout, BB, n, n, n, 2);
		}
	if (r2 != k2) {
		cout << "finite_field::intersect_subspaces not a base, "
				"rank is too small" << endl;
		cout << "k2=" << k2 << endl;
		cout << "r2=" << r2 << endl;
		exit(1);
		}
	CC = NEW_INT((3 * n) * n);

	INT_vec_copy(AA + k1 * n, CC, (n - k1) * n);
	INT_vec_copy(BB + k2 * n, CC + (n - k1) * n, (n - k2) * n);
	k3 = (n - k1) + (n - k2);
	if (f_v) {
		cout << "finite_field::intersect_subspaces CC=" << endl;
		print_integer_matrix_width(cout, CC, k3, n, n, 2);
		cout << "k3=" << k3 << endl;
		}

	
	k3 = Gauss_easy(CC, k3, n);

	r3 = perp_standard_with_temporary_data(n, k3, CC, B1, K, base_cols, 0);
	INT_vec_copy(CC + k3 * n, intersection, (n - r3) * n);

	FREE_INT(AA);
	FREE_INT(BB);
	FREE_INT(CC);
	FREE_INT(B1);
	FREE_INT(K);
	FREE_INT(base_cols);
	if (f_v) {
		cout << "finite_field::intersect_subspaces n=" << n
				<< " dim A =" << r1 << " dim B =" << r2
				<< " dim intersection =" << n - r3 << endl;
		}
	k3 = n - r3;
	return n - r3;
	
}

INT finite_field::n_choose_k_mod_p(INT n, INT k, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT n1, k1, c, cc = 0, cv, c1 = 1, c2 = 1, i;
	
	c = 1;
	while (n || k) {
		n1 = n % p;
		k1 = k % p;
		c1 = 1;
		c2 = 1;
		for (i = 0; i < k1; i++) {
			c1 = mult(c1, (n1 - i) % p);
			c2 = mult(c2, (1 + i) % p);
			}
		if (c1 != 0) {
			cv = inverse(c2);
			cc = mult(c1, cv);
			}
		if (f_vv) {
			cout << "{" << n1 << "\\atop " << k1 << "} mod "
					<< p << " = " << cc << endl;
			}
		c = mult(c, cc);
		n = (n - n1) / p;
		k = (k - k1) / p;
		}
	if (f_v) {
		cout << "{" << n << "\\atop " << k << "} mod "
				<< p << " = " << endl;
		cout << c << endl;
		}
	return c;
}

void finite_field::Dickson_polynomial(INT *map, INT *coeffs)
// compute the coefficients of a degree q-1 polynomial
// which interpolates a given map
// from F_q to F_q
{
	INT i, x, c, xi, a;
	
	// coeff[0] = map[0]
	coeffs[0] = map[0];
	
	// the middle coefficients:
	// coeff[i] = - sum_{x \neq 0} g(x) x^{-i}
	for (i = 1; i <= q - 2; i++) {
		c = 0;
		for (x = 1; x < q; x++) {
			xi = inverse(x);
			xi = power(xi, i);
			a = mult(map[x], xi);
			c = add(c, a);
			}
		coeffs[i] = negate(c);
		}
	
	// coeff[q - 1] = - \sum_x map[x]
	c = 0;
	for (x = 0; x < q; x++) {
		c = add(c, map[x]);
		}
	coeffs[q - 1] = negate(c);
}

void finite_field::projective_action_on_columns_from_the_left(
		INT *A, INT *M, INT m, INT n, INT *perm,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *AM, i, j;
	
	AM = NEW_INT(m * n);
	
	if (f_v) {
		cout << "projective_action_on_columns_from_the_left" << endl;
		}
	if (f_vv) {
		cout << "A:" << endl;
		print_integer_matrix_width(cout, A, m, m, m, 2);
		}
	mult_matrix_matrix(A, M, AM, m, m, n);
	if (f_vv) {
		cout << "M:" << endl;
		print_integer_matrix_width(cout, M, m, n, n, 2);
		//cout << "A * M:" << endl;
		//print_integer_matrix_width(cout, AM, m, n, n, 2);
		}
						
	for (j = 0; j < n; j++) {
		PG_element_normalize_from_front(*this, AM + j,
				n /* stride */, m /* length */);
		}
	if (f_vv) {
		cout << "A*M:" << endl;
		print_integer_matrix_width(cout, AM, m, n, n, 2);
		}
					
	for (i = 0; i < n; i++) {
		perm[i] = -1;
		for (j = 0; j < n; j++) {
			if (INT_vec_compare_stride(AM + i, M + j,
					m /* len */, n /* stride */) == 0) {
				perm[i] = j;
				break;
				}
			}
		if (j == n) {
			cout << "finite_field::projective_action_on_columns_"
					"from_the_left could not find image" << endl;
			cout << "i=" << i << endl;
			cout << "M:" << endl;
			print_integer_matrix_width(cout, M, m, n, n, 2);
			cout << "A * M:" << endl;
			print_integer_matrix_width(cout, AM, m, n, n, 2);
			exit(1);
			}
		}
	if (f_v) {
		//cout << "column permutation: ";
		perm_print_with_cycle_length(cout, perm, n);
		cout << endl;
		}
	FREE_INT(AM);
}

void finite_field::builtin_transversal_rep_GLnq(INT *A,
		INT n, INT f_semilinear, INT i, INT j,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	INT transversal_length;
	INT ii, jj, i0, a;
	
	if (f_v) {
		cout << "finite_field::builtin_transversal_rep_GLnq  "
				"GL(" << n << "," << q << ") i = " << i
				<< " j = " << j << endl;
		}

	// make the n x n identity matrix:
	for (ii = 0; ii < n * n; ii++) {
		A[ii] = 0;
		}
	for (ii = 0; ii < i; ii++) {
		A[ii * n + ii] = 1;
		}
	if (f_semilinear) {
		A[n * n] = 0;
		}

	if ((i == n + 1 && q > 2) || (i == n && q == 2)) {
		if (!f_semilinear) {
			cout << "finite_field::builtin_transversal_rep_GLnq "
					"must be semilinear to access transversal " << n << endl;
			exit(1);
			}
		A[n * n] = j;
		}
	else if (i == n && q > 2) {
		transversal_length = nb_AG_elements(n - 1, q - 1);
		if (j >= transversal_length) {
			cout << "finite_field::builtin_transversal_rep_GLnq "
					"j = " << j << " >= transversal_length = "
					<< transversal_length << endl;
			exit(1);
			}
		INT *v = NEW_INT(n);
		AG_element_unrank(q - 1, v, 1, n - 1, j);
		A[0] = 1;
		for (jj = 0; jj < n - 1; jj++) {
			A[(jj + 1) * n + (jj + 1)] = v[jj] + 1;
			}
		FREE_INT(v);
		}
	else {
		if (i == 0) {
			PG_element_unrank_modified(*this, A + i, n, n, j);
			}
		else {
			PG_element_unrank_modified_not_in_subspace(*this,
					A + i, n, n, i - 1, j);
			}
		i0 = -1;
		for (ii = 0; ii < n; ii++) {
			a = A[ii * n + i];
			if (ii >= i && i0 == -1 && a != 0) {
				i0 = ii;
				}
			}
		if (f_vv) {
			cout << "i0 = " << i0 << endl;
			}
		for (jj = i; jj < i0; jj++) {
			A[jj * n + jj + 1] = 1;
			}
		for (jj = i0 + 1; jj < n; jj++) {
			A[jj * n + jj] = 1;
			}
		//INT_matrix_transpose(n, A);
		transpose_matrix_in_place(A, n);
		}
	
	if (f_vv) {
		cout << "transversal_rep_GLnq[" << i << "][" << j << "] = \n";
		print_integer_matrix(cout, A, n, n);
		}
}

void finite_field::affine_translation(INT n,
		INT coordinate_idx, INT field_base_idx, INT *perm)
// perm points to q^n INT's
// field_base_idx is the base element whose translation
// we compute, 0 \le field_base_idx < e
// coordinate_idx is the coordinate in which we shift,
// 0 \le coordinate_idx < n
{
	INT i, j, l, a;
	INT *v;
	
	cout << "finite_field::affine_translation "
			"coordinate_idx=" << coordinate_idx
			<< " field_base_idx=" << field_base_idx << endl;
	v = NEW_INT(n);
	l = nb_AG_elements(n, q);
	a = i_power_j(p, field_base_idx);
	for (i = 0; i < l; i++) {
		AG_element_unrank(q, v, 1, l, i);
		v[coordinate_idx] = add(v[coordinate_idx], a);
		AG_element_rank(q, v, 1, l, j);
		perm[i] = j;
		}
	FREE_INT(v);
}

void finite_field::affine_multiplication(INT n,
		INT multiplication_order, INT *perm)
// perm points to q^n INT's
// compute the diagonal multiplication by alpha, i.e. 
// the multiplication by alpha of each component
{
	INT i, j, l, k;
	INT alpha_power, a;
	INT *v;
	
	v = NEW_INT(n);
	alpha_power = (q - 1) / multiplication_order;
	if (alpha_power * multiplication_order != q - 1) {
		cout << "finite_field::affine_multiplication: "
				"multiplication_order does not divide q - 1" << endl;
		exit(1);
		}
	a = power(alpha, alpha_power);
	l = nb_AG_elements(n, q);
	for (i = 0; i < l; i++) {
		AG_element_unrank(q, v, 1, l, i);
		for (k = 0; k < n; k++) {
			v[k] = mult(v[k], a);
			}
		AG_element_rank(q, v, 1, l, j);
		perm[i] = j;
		}
	FREE_INT(v);
}

void finite_field::affine_frobenius(INT n, INT k, INT *perm)
// perm points to q^n INT's
// compute the diagonal action of the Frobenius automorphism
// to the power k, i.e.,
// raises each component to the p^k-th power
{
	INT i, j, l, u;
	INT *v;
	
	v = NEW_INT(n);
	l = nb_AG_elements(n, q);
	for (i = 0; i < l; i++) {
		AG_element_unrank(q, v, 1, l, i);
		for (u = 0; u < n; u++) {
			v[u] = frobenius_power(v[u], k);
			}
		AG_element_rank(q, v, 1, l, j);
		perm[i] = j;
		}
	FREE_INT(v);
}


INT finite_field::all_affine_translations_nb_gens(INT n)
{
	INT nb_gens;
	
	nb_gens = e * n;
	return nb_gens;
}

void finite_field::all_affine_translations(INT n, INT *gens)
{
	INT i, j, k = 0;
	INT degree;
	
	degree = nb_AG_elements(n, q);
	
	for (i = 0; i < n; i++) {
		for (j = 0; j < e; j++, k++) {
			affine_translation(n, i, j, gens + k * degree);
			}
		}
}

void finite_field::affine_generators(INT n,
	INT f_translations,
	INT f_semilinear, INT frobenius_power, 
	INT f_multiplication, INT multiplication_order, 
	INT &nb_gens, INT &degree, INT *&gens, 
	INT &base_len, INT *&the_base)
{
	INT k, h;
	
	degree = nb_AG_elements(n, q);
	nb_gens = 0;
	base_len = 0;
	if (f_translations) {
		nb_gens += all_affine_translations_nb_gens(n);
		base_len++;
		}
	if (f_multiplication) {
		nb_gens++;
		base_len++;
		}
	if (f_semilinear) {
		nb_gens++;
		base_len++;
		}
	
	gens = NEW_INT(nb_gens * degree);
	the_base = NEW_INT(base_len);
	k = 0;
	h = 0;
	if (f_translations) {
		all_affine_translations(n, gens);
		k += all_affine_translations_nb_gens(n);
		the_base[h++] = 0;
		}
	if (f_multiplication) {
		affine_multiplication(n, multiplication_order,
				gens + k * degree);
		k++;
		the_base[h++] = 1;
		}
	if (f_semilinear) {
		affine_frobenius(n, frobenius_power, gens + k * degree);
		k++;
		the_base[h++] = p;
		}
}

INT finite_field::evaluate_bilinear_form(INT n,
		INT *v1, INT *v2, INT *Gram)
{
	INT *v3, a;
	
	v3 = NEW_INT(n);
	mult_matrix(v1, Gram, v3, 1, n, n);
	a = dot_product(n, v3, v2);
	FREE_INT(v3);
	return a;
}
 
INT finite_field::evaluate_standard_hyperbolic_bilinear_form(
		INT n, INT *v1, INT *v2)
{
	INT a, b, c, n2, i;
	
	if (ODD(n)) {
		cout << "finite_field::evaluate_standard_hyperbolic_bilinear_form "
				"n must be even" << endl;
		exit(1);
		}
	n2 = n >> 1;
	c = 0;
	for (i = 0; i < n2; i++) {
		a = mult(v1[2 * i + 0], v2[2 * i + 1]);
		b = mult(v1[2 * i + 1], v2[2 * i + 0]);
		c = add(c, a);
		c = add(c, b);
		}
	return c;
}
 
INT finite_field::evaluate_quadratic_form(INT n, INT nb_terms, 
	INT *i, INT *j, INT *coeff, INT *x)
{
	INT k, xi, xj, a, c, d;
	
	a = 0;
	for (k = 0; k < nb_terms; k++) {
		xi = x[i[k]];
		xj = x[j[k]];
		c = coeff[k];
		d = mult(mult(c, xi), xj);
		a = add(a, d);
		}
	return a;
}

void finite_field::find_singular_vector_brute_force(INT n,
	INT form_nb_terms,
	INT *form_i, INT *form_j, INT *form_coeff, INT *Gram, 
	INT *vec, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT N, a, i;
	INT *v1;
	
	if (f_v) {
		cout << "finite_field::find_singular_vector_brute_force" << endl;
		}
	v1 = NEW_INT(n);
	N = nb_AG_elements(n, q);
	for (i = 2; i < N; i++) {
		AG_element_unrank(q, v1, 1, n, i);
		a = evaluate_quadratic_form(n, form_nb_terms,
				form_i, form_j, form_coeff, v1);
		if (f_v) {
			cout << "v1=";
			INT_vec_print(cout, v1, n);
			cout << endl;
			cout << "form value a=" << a << endl;
			}
		if (a == 0) {
			INT_vec_copy(v1, vec, n);
			goto finish;
			}
		}
	cout << "finite_field::find_singular_vector_brute_force "
			"did not find a singular vector" << endl;
	exit(1);

finish:
	FREE_INT(v1);
}

void finite_field::find_singular_vector(INT n, INT form_nb_terms, 
	INT *form_i, INT *form_j, INT *form_coeff, INT *Gram, 
	INT *vec, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a, b, c, d, r3, x, y, i, k3;
	INT *v1, *v2, *v3, *v2_coords, *v3_coords, *intersection;
	
	if (f_v) {
		cout << "finite_field::find_singular_vector" << endl;
		}
	if (n < 3) {
		cout << "finite_field::find_singular_vector n < 3" << endl;
		exit(1);
		}
	v1 = NEW_INT(n * n);
	v2 = NEW_INT(n * n);
	v3 = NEW_INT(n * n);
	v2_coords = NEW_INT(n);
	v3_coords = NEW_INT(n);
	intersection = NEW_INT(n * n);

	//N = nb_AG_elements(n, q);
	AG_element_unrank(q, v1, 1, n, 1);
	a = evaluate_quadratic_form(n, form_nb_terms,
			form_i, form_j, form_coeff, v1);
	if (f_v) {
		cout << "v1=";
		INT_vec_print(cout, v1, n);
		cout << endl;
		cout << "form value a=" << a << endl;
		}
	if (a == 0) {
		INT_vec_copy(v1, vec, n);
		goto finish;
		}
	perp(n, 1, v1, Gram);
	if (f_v) {
		cout << "v1 perp:" << endl;
		print_integer_matrix_width(cout, v1 + n, n - 1, n, n, 2);
		}
	AG_element_unrank(q, v2_coords, 1, n - 1, 1);
	mult_matrix(v2_coords, v1 + n, v2, 1, n - 1, n);
	b = evaluate_quadratic_form(n, form_nb_terms,
			form_i, form_j, form_coeff, v2);
	if (f_v) {
		cout << "vector v2=";
		INT_vec_print(cout, v2, n);
		cout << endl;
		cout << "form value b=" << b << endl;
		}
	if (b == 0) {
		INT_vec_copy(v2, vec, n);
		goto finish;
		}
	perp(n, 1, v2, Gram);
	if (f_v) {
		cout << "v2 perp:" << endl;
		print_integer_matrix_width(cout, v2 + n, n - 1, n, n, 2);
		}
	r3 = intersect_subspaces(n, n - 1, v1 + n, n - 1, v2 + n, 
		k3, intersection, verbose_level);
	if (f_v) {
		cout << "intersection has dimension " << r3 << endl;
		print_integer_matrix_width(cout, intersection, r3, n, n, 2);
		}
	if (r3 != n - 2) {
		cout << "r3 = " << r3 << " should be " << n - 2 << endl;
		exit(1);
		}
	AG_element_unrank(q, v3_coords, 1, n - 2, 1);
	mult_matrix(v3_coords, intersection, v3, 1, n - 2, n);
	c = evaluate_quadratic_form(n, form_nb_terms,
			form_i, form_j, form_coeff, v3);
	if (f_v) {
		cout << "v3=";
		INT_vec_print(cout, v3, n);
		cout << endl;
		cout << "form value c=" << c << endl;
		}
	if (c == 0) {
		INT_vec_copy(v3, vec, n);
		goto finish;
		}
	if (f_v) {
		cout << "calling abc2xy" << endl;
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		cout << "c=" << negate(c) << endl;
		}
	abc2xy(a, b, negate(c), x, y, verbose_level);
	if (f_v) {
		cout << "x=" << x << endl;
		cout << "y=" << y << endl;
		}
	scalar_multiply_vector_in_place(x, v1, n);
	scalar_multiply_vector_in_place(y, v2, n);
	for (i = 0; i < n; i++) {
		vec[i] = add(add(v1[i], v2[i]), v3[i]);
		}
	if (f_v) {
		cout << "singular vector vec=";
		INT_vec_print(cout, vec, n);
		cout << endl;
		}
	d = evaluate_quadratic_form(n, form_nb_terms,
			form_i, form_j, form_coeff, vec);
	if (d) {
		cout << "is non-singular, error! d=" << d << endl;
		cout << "singular vector vec=";
		INT_vec_print(cout, vec, n);
		cout << endl;
		exit(1);
		}
finish:
	FREE_INT(v1);
	FREE_INT(v2);
	FREE_INT(v3);
	FREE_INT(v2_coords);
	FREE_INT(v3_coords);
	FREE_INT(intersection);
}

void finite_field::complete_hyperbolic_pair(INT n, INT form_nb_terms, 
	INT *form_i, INT *form_j, INT *form_coeff, INT *Gram, 
	INT *vec1, INT *vec2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, a, b, c;
	INT *v0, *v1;
	
	v0 = NEW_INT(n * n);
	v1 = NEW_INT(n * n);

	if (f_v) {
		cout << "finite_field::complete_hyperbolic_pair" << endl;
		cout << "vec1=";
		INT_vec_print(cout, vec1, n);
		cout << endl;
		cout << "Gram=" << endl;
		print_integer_matrix_width(cout, Gram, 4, 4, 4, 2);
		}
	mult_matrix(vec1, Gram, v0, 1, n, n);
	if (f_v) {
		cout << "v0=";
		INT_vec_print(cout, v0, n);
		cout << endl;
		}
	INT_vec_zero(v1, n);
	for (i = n - 1; i >= 0; i--) {
		if (v0[i]) {
			v1[i] = 1;
			break;
			}
		}
	if (i == -1) {
		cout << "finite_field::complete_hyperbolic_pair i == -1" << endl;
		exit(1);
		}
	a = dot_product(n, v0, v1);
#if 0
	INT N;
	
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
			INT_vec_print(cout, v1, n);
			cout << endl;
			}
		a = dot_product(n, v0, v1);
		if (f_v) {
			cout << "i=" << i << " trying vector ";
			INT_vec_print(cout, v1, n);
			cout << " form value a=" << a << endl;
			}
		if (a)
			break;
		}
	if (i == N) {
		cout << "finite_field::complete_hyperbolic_pair "
			"did not find a vector whose dot product is non-zero " << endl;
		}
#endif

	if (a != 1) {
		scalar_multiply_vector_in_place(inverse(a), v1, n);
		}
	if (f_v) {
		cout << "normalized ";
		INT_vec_print(cout, v1, n);
		cout << endl;
		}
	b = evaluate_quadratic_form(n, form_nb_terms,
			form_i, form_j, form_coeff, v1);
	b = negate(b);
	for (i = 0; i < n; i++) {
		vec2[i] = add(mult(b, vec1[i]), v1[i]);
		}
	if (f_v) {
		cout << "finite_field::complete_hyperbolic_pair" << endl;
		cout << "vec2=";
		INT_vec_print(cout, vec2, n);
		cout << endl;
		}
	c = dot_product(n, v0, vec2);
	if (c != 1) {
		cout << "dot product is not 1, error" << endl;
		cout << "c=" << c << endl;
		cout << "vec1=";
		INT_vec_print(cout, vec1, n);
		cout << endl;
		cout << "vec2=";
		INT_vec_print(cout, vec2, n);
		cout << endl;
		}
	FREE_INT(v0);
	FREE_INT(v1);
	
}

void finite_field::find_hyperbolic_pair(INT n, INT form_nb_terms, 
	INT *form_i, INT *form_j, INT *form_coeff, INT *Gram, 
	INT *vec1, INT *vec2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
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
	if (f_v) {
		cout << "finite_field::find_hyperbolic_pair, "
				"found singular vector" << endl;
		INT_vec_print(cout, vec1, n);
		cout << endl;
		cout << "calling complete_hyperbolic_pair" << endl;
		}
	complete_hyperbolic_pair(n, form_nb_terms, 
		form_i, form_j, form_coeff, Gram, 
		vec1, vec2, verbose_level);
}

void finite_field::restrict_quadratic_form_list_coding(
	INT k, INT n, INT *basis,
	INT form_nb_terms, INT *form_i, INT *form_j, INT *form_coeff, 
	INT &restricted_form_nb_terms, INT *&restricted_form_i,
	INT *&restricted_form_j, INT *&restricted_form_coeff,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *C, *D, h, i, j, c;
	
	C = NEW_INT(n * n);
	D = NEW_INT(k * k);
	INT_vec_zero(C, n * n);
	for (h = 0; h < form_nb_terms; h++) {
		i = form_i[h];
		j = form_j[h];
		c = form_coeff[h];
		C[i * n + j] = c;
		}
	if (f_v) {
		cout << "finite_field::restrict_quadratic_form_list_coding "
				"C=" << endl;
		print_integer_matrix_width(cout, C, n, n, n, 2);
		}
	restrict_quadratic_form(k, n, basis, C, D, verbose_level);
	if (f_v) {
		cout << "finite_field::restrict_quadratic_form_list_coding "
				"D=" << endl;
		print_integer_matrix_width(cout, D, k, k, k, 2);
		}
	restricted_form_nb_terms = 0;
	restricted_form_i = NEW_INT(k * k);
	restricted_form_j = NEW_INT(k * k);
	restricted_form_coeff = NEW_INT(k * k);
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
	FREE_INT(C);
	FREE_INT(D);
}

void finite_field::restrict_quadratic_form(INT k, INT n,
		INT *basis, INT *C, INT *D,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, lambda, mu, a1, a2, d, c;
	
	if (f_v) {
		cout << "finite_field::restrict_quadratic_form" << endl;
		print_integer_matrix_width(cout, C, n, n, n, 2);
		}
	INT_vec_zero(D, k * k);
	for (lambda = 0; lambda < k; lambda++) {
		for (mu = 0; mu < k; mu++) {
			d = 0;
			for (i = 0; i < n; i++) {
				for (j = i; j < n; j++) {
					a1 = basis[lambda * n + i];
					a2 = basis[mu * n + j];
					c = C[i * n + j];
					d = add(d, mult(c, mult(a1, a2)));
					}
				}
			if (mu < lambda) {
				D[mu * k + lambda] = add(D[mu * k + lambda], d);
				}
			else {
				D[lambda * k + mu] = add(D[lambda * k + mu], d);
				}
			}
		}
	if (f_v) {
		print_integer_matrix_width(cout, D, k, k, k, 2);
		}
}

INT finite_field::compare_subspaces_ranked(
	INT *set1, INT *set2, INT size,
	INT vector_space_dimension, INT verbose_level)
// Compares the span of two sets of vectors.
// returns 0 if equal, 1 if not
// (this is so that it matches to the result of a compare function)
{
	INT f_v = (verbose_level >= 1);
	INT *M1;
	INT *M2;
	INT *base_cols1;
	INT *base_cols2;
	INT i;
	INT rk1, rk2, r;

	if (f_v) {
		cout << "finite_field::compare_subspaces_ranked" << endl;
		cout << "set1: ";
		INT_vec_print(cout, set1, size);
		cout << endl;
		cout << "set2: ";
		INT_vec_print(cout, set2, size);
		cout << endl;
		}
	M1 = NEW_INT(size * vector_space_dimension);
	M2 = NEW_INT(size * vector_space_dimension);
	base_cols1 = NEW_INT(vector_space_dimension);
	base_cols2 = NEW_INT(vector_space_dimension);
	for (i = 0; i < size; i++) {
		PG_element_unrank_modified(*this,
			M1 + i * vector_space_dimension,
			1, vector_space_dimension, set1[i]);
		PG_element_unrank_modified(*this,
			M2 + i * vector_space_dimension,
			1, vector_space_dimension, set2[i]);
		}
	if (f_v) {
		cout << "matrix1:" << endl;
		print_integer_matrix_width(cout, M1, size,
			vector_space_dimension, vector_space_dimension, log10_of_q);
		cout << "matrix2:" << endl;
		print_integer_matrix_width(cout, M2, size,
			vector_space_dimension, vector_space_dimension, log10_of_q);
		}
	rk1 = Gauss_simple(M1, size, vector_space_dimension,
			base_cols1, 0/*INT verbose_level*/);
	rk2 = Gauss_simple(M2, size, vector_space_dimension,
			base_cols2, 0/*INT verbose_level*/);
	if (f_v) {
		cout << "after Gauss" << endl;
		cout << "matrix1:" << endl;
		print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension, log10_of_q);
		cout << "rank1=" << rk1 << endl;
		cout << "base_cols1: ";
		INT_vec_print(cout, base_cols1, rk1);
		cout << endl;
		cout << "matrix2:" << endl;
		print_integer_matrix_width(cout, M2, size,
				vector_space_dimension, vector_space_dimension, log10_of_q);
		cout << "rank2=" << rk2 << endl;
		cout << "base_cols2: ";
		INT_vec_print(cout, base_cols2, rk2);
		cout << endl;
		}
	if (rk1 != rk2) {
		if (f_v) {
			cout << "the ranks differ, so the subspaces are not equal, "
					"we return 1" << endl;
			}
		r = 1;
		goto ret;
		}
	for (i = 0; i < rk1; i++) {
		if (base_cols1[i] != base_cols2[i]) {
			if (f_v) {
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
			if (f_v) {
				cout << "the matrices differ in entry " << i
						<< ", so the subspaces are not equal, "
						"we return 1" << endl;
				}
			r = 1;
			goto ret;
			}
		}
	if (f_v) {
		cout << "the subspaces are equal, we return 0" << endl;
		}
	r = 0;
ret:
	FREE_INT(M1);
	FREE_INT(M2);
	FREE_INT(base_cols1);
	FREE_INT(base_cols2);
	return r;
}

INT finite_field::compare_subspaces_ranked_with_unrank_function(
	INT *set1, INT *set2, INT size, 
	INT vector_space_dimension, 
	void (*unrank_point_func)(INT *v, INT rk, void *data), 
	void *rank_point_data, 
	INT verbose_level)
// Compares the span of two sets of vectors.
// returns 0 if equal, 1 if not
// (this is so that it matches to the result of a compare function)
{
	INT f_v = (verbose_level >= 1);
	INT *M1;
	INT *M2;
	INT *base_cols1;
	INT *base_cols2;
	INT i;
	INT rk1, rk2, r;

	if (f_v) {
		cout << "finite_field::compare_subspaces_ranked_"
				"with_unrank_function" << endl;
		cout << "set1: ";
		::INT_vec_print(cout, set1, size);
		cout << endl;
		cout << "set2: ";
		::INT_vec_print(cout, set2, size);
		cout << endl;
		}
	M1 = NEW_INT(size * vector_space_dimension);
	M2 = NEW_INT(size * vector_space_dimension);
	base_cols1 = NEW_INT(vector_space_dimension);
	base_cols2 = NEW_INT(vector_space_dimension);
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
	if (f_v) {
		cout << "matrix1:" << endl;
		print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				log10_of_q);
		cout << "matrix2:" << endl;
		print_integer_matrix_width(cout, M2, size,
				vector_space_dimension, vector_space_dimension,
				log10_of_q);
		}
	rk1 = Gauss_simple(M1, size,
			vector_space_dimension, base_cols1,
			0/*INT verbose_level*/);
	rk2 = Gauss_simple(M2, size,
			vector_space_dimension, base_cols2,
			0/*INT verbose_level*/);
	if (f_v) {
		cout << "after Gauss" << endl;
		cout << "matrix1:" << endl;
		print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				log10_of_q);
		cout << "rank1=" << rk1 << endl;
		cout << "base_cols1: ";
		::INT_vec_print(cout, base_cols1, rk1);
		cout << endl;
		cout << "matrix2:" << endl;
		print_integer_matrix_width(cout, M2, size,
				vector_space_dimension, vector_space_dimension,
				log10_of_q);
		cout << "rank2=" << rk2 << endl;
		cout << "base_cols2: ";
		::INT_vec_print(cout, base_cols2, rk2);
		cout << endl;
		}
	if (rk1 != rk2) {
		if (f_v) {
			cout << "the ranks differ, so the subspaces are not equal, "
					"we return 1" << endl;
			}
		r = 1;
		goto ret;
		}
	for (i = 0; i < rk1; i++) {
		if (base_cols1[i] != base_cols2[i]) {
			if (f_v) {
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
			if (f_v) {
				cout << "the matrices differ in entry " << i
						<< ", so the subspaces are not equal, "
						"we return 1" << endl;
				}
			r = 1;
			goto ret;
			}
		}
	if (f_v) {
		cout << "the subspaces are equal, we return 0" << endl;
		}
	r = 0;
ret:
	FREE_INT(M1);
	FREE_INT(M2);
	FREE_INT(base_cols1);
	FREE_INT(base_cols2);
	return r;
}

INT finite_field::Gauss_canonical_form_ranked(
	INT *set1, INT *set2, INT size,
	INT vector_space_dimension, INT verbose_level)
// Computes the Gauss canonical form for the generating set in set1.
// The result is written to set2.
// Returns the rank of the span of the elements in set1.
{
	INT f_v = (verbose_level >= 1);
	INT *M;
	INT *base_cols;
	INT i;
	INT rk;

	if (f_v) {
		cout << "finite_field::Gauss_canonical_form_ranked" << endl;
		cout << "set1: ";
		INT_vec_print(cout, set1, size);
		cout << endl;
		}
	M = NEW_INT(size * vector_space_dimension);
	base_cols = NEW_INT(vector_space_dimension);
	for (i = 0; i < size; i++) {
		PG_element_unrank_modified(*this,
			M + i * vector_space_dimension,
			1, vector_space_dimension,
			set1[i]);
		}
	if (f_v) {
		cout << "matrix:" << endl;
		print_integer_matrix_width(cout, M, size,
			vector_space_dimension, vector_space_dimension,
			log10_of_q);
		}
	rk = Gauss_simple(M, size,
			vector_space_dimension, base_cols,
			0/*INT verbose_level*/);
	if (f_v) {
		cout << "after Gauss" << endl;
		cout << "matrix:" << endl;
		print_integer_matrix_width(cout, M, size,
				vector_space_dimension, vector_space_dimension,
				log10_of_q);
		cout << "rank=" << rk << endl;
		cout << "base_cols: ";
		INT_vec_print(cout, base_cols, rk);
		cout << endl;
		}

	for (i = 0; i < rk; i++) {
		PG_element_rank_modified(*this,
			M + i * vector_space_dimension,
			1, vector_space_dimension,
			set2[i]);
		}


	FREE_INT(M);
	FREE_INT(base_cols);
	return rk;

}

INT finite_field::lexleast_canonical_form_ranked(
	INT *set1, INT *set2, INT size,
	INT vector_space_dimension, INT verbose_level)
// Computes the lexleast generating set the
// subspace spanned by the elements in set1.
// The result is written to set2.
// Returns the rank of the span of the elements in set1.
{
	INT f_v = (verbose_level >= 1);
	INT *M1, *M2;
	INT *v;
	INT *w;
	INT *base_cols;
	INT *f_allowed;
	INT *basis_vectors;
	INT *list_of_ranks;
	INT *list_of_ranks_PG;
	INT *list_of_ranks_PG_sorted;
	INT size_list, idx;
	INT *tmp;
	INT i, j, h, N, a, sz, Sz;
	INT rk;

	if (f_v) {
		cout << "finite_field::lexleast_canonical_form_ranked" << endl;
		cout << "set1: ";
		INT_vec_print(cout, set1, size);
		cout << endl;
		}
	tmp = NEW_INT(vector_space_dimension);
	M1 = NEW_INT(size * vector_space_dimension);
	base_cols = NEW_INT(vector_space_dimension);
	for (i = 0; i < size; i++) {
		PG_element_unrank_modified(*this, M1 + i * vector_space_dimension, 
			1, vector_space_dimension, set1[i]);
		}
	if (f_v) {
		cout << "matrix:" << endl;
		print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				log10_of_q);
		}
	
	rk = Gauss_simple(M1, size, vector_space_dimension,
			base_cols, 0/*INT verbose_level*/);
	v = NEW_INT(rk);
	w = NEW_INT(rk);
	if (f_v) {
		cout << "after Gauss" << endl;
		cout << "matrix:" << endl;
		print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				log10_of_q);
		cout << "rank=" << rk << endl;
		cout << "base_cols: ";
		INT_vec_print(cout, base_cols, rk);
		cout << endl;
		}
	N = i_power_j(q, rk);
	M2 = NEW_INT(N * vector_space_dimension);
	list_of_ranks = NEW_INT(N);
	list_of_ranks_PG = NEW_INT(N);
	list_of_ranks_PG_sorted = NEW_INT(N);
	basis_vectors = NEW_INT(rk);
	size_list = 0;
	list_of_ranks_PG[0] = -1;
	for (a = 0; a < N; a++) {
		AG_element_unrank(q, v, 1, rk, a);
		mult_matrix_matrix(v, M1, M2 + a * vector_space_dimension, 
			1, rk, vector_space_dimension);
		AG_element_rank(q, M2 + a * vector_space_dimension, 1,
				vector_space_dimension, list_of_ranks[a]);
		if (a == 0) {
			continue;
			}
		PG_element_rank_modified(*this,
				M2 + a * vector_space_dimension, 1,
				vector_space_dimension, list_of_ranks_PG[a]);
		if (!INT_vec_search(list_of_ranks_PG_sorted,
				size_list, list_of_ranks_PG[a], idx)) {
			for (h = size_list; h > idx; h--) {
				list_of_ranks_PG_sorted[h] = list_of_ranks_PG_sorted[h - 1];
				}
			list_of_ranks_PG_sorted[idx] = list_of_ranks_PG[a];
			size_list++;
			}
		}
	if (f_v) {
		cout << "expanded matrix with all elements in the space:" << endl;
		print_integer_matrix_width(cout, M2, N,
				vector_space_dimension, vector_space_dimension,
				log10_of_q);
		cout << "list_of_ranks:" << endl;
		INT_vec_print(cout, list_of_ranks, N);
		cout << endl;	
		cout << "list_of_ranks_PG:" << endl;
		INT_vec_print(cout, list_of_ranks_PG, N);
		cout << endl;	
		cout << "list_of_ranks_PG_sorted:" << endl;
		INT_vec_print(cout, list_of_ranks_PG_sorted, size_list);
		cout << endl;	
		}
	f_allowed = NEW_INT(size_list);
	for (i = 0; i < size_list; i++) {
		f_allowed[i] = TRUE;
		}

	sz = 1;
	for (i = 0; i < rk; i++) {
		if (f_v) {
			cout << "step " << i << " ";
			cout << " list_of_ranks_PG_sorted=";
			INT_vec_print(cout, list_of_ranks_PG_sorted, size_list);
			cout << " ";
			cout << "f_allowed=";
			INT_vec_print(cout, f_allowed, size_list);
			cout << endl;
			}
		for (a = 0; a < size_list; a++) {
			if (f_allowed[a]) {
				break;
				}
			}
		if (f_v) {
			cout << "choosing a=" << a << " list_of_ranks_PG_sorted[a]="
					<< list_of_ranks_PG_sorted[a] << endl;
			}
		basis_vectors[i] = list_of_ranks_PG_sorted[a];
		PG_element_unrank_modified(*this, M1 + i * vector_space_dimension, 
			1, vector_space_dimension, basis_vectors[i]);
		Sz = q * sz;
		if (f_v) {
			cout << "step " << i
					<< " basis_vector=" << basis_vectors[i] << " : ";
			INT_vec_print(cout, M1 + i * vector_space_dimension,
					vector_space_dimension);
			cout << " sz=" << sz << " Sz=" << Sz << endl;
			}
		for (h = 0; h < size_list; h++) {
			if (list_of_ranks_PG_sorted[h] == basis_vectors[i]) {
				if (f_v) {
					cout << "disallowing " << h << endl;
					}
				f_allowed[h] = FALSE;
				break;
				}
			}
		for (j = sz; j < Sz; j++) {
			AG_element_unrank(q, v, 1, i + 1, j);
			if (f_v) {
				cout << "j=" << j << " v=";
				INT_vec_print(cout, v, i + 1);
				cout << endl;
				}
#if 0
			for (h = 0; h < i + 1; h++) {
				w[i - h] = v[h];
				}
			if (f_v) {
				cout << " w=";
				INT_vec_print(cout, w, i + 1);
				cout << endl;
				}
#endif
			mult_matrix_matrix(v/*w*/, M1, tmp, 1, i + 1,
					vector_space_dimension);
			if (f_v) {
				cout << " tmp=";
				INT_vec_print(cout, tmp, vector_space_dimension);
				cout << endl;
				}
			PG_element_rank_modified(*this, tmp, 1,
					vector_space_dimension, a);
			if (f_v) {
				cout << "has rank " << a << endl;
				}
			for (h = 0; h < size_list; h++) {
				if (list_of_ranks_PG_sorted[h] == a) {
					if (f_v) {
						cout << "disallowing " << h << endl;
						}
					f_allowed[h] = FALSE;
					break;
					}
				}
			}
		sz = Sz;	
		}
	if (f_v) {
		cout << "basis_vectors by rank: ";
		INT_vec_print(cout, basis_vectors, rk);
		cout << endl;
		}
	if (f_v) {
		cout << "basis_vectors by coordinates: " << endl;
		print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				log10_of_q);
		cout << endl;
		}

	for (i = 0; i < rk; i++) {
		PG_element_rank_modified(*this, M1 + i * vector_space_dimension, 
			1, vector_space_dimension, set2[i]);
		}
	if (f_v) {
		cout << "basis_vectors by rank again (double check): ";
		INT_vec_print(cout, set2, rk);
		cout << endl;
		}


	FREE_INT(tmp);
	FREE_INT(M1);
	FREE_INT(M2);
	FREE_INT(v);
	FREE_INT(w);
	FREE_INT(base_cols);
	FREE_INT(f_allowed);
	FREE_INT(list_of_ranks);
	FREE_INT(list_of_ranks_PG);
	FREE_INT(list_of_ranks_PG_sorted);
	FREE_INT(basis_vectors);
	return rk;

}

void finite_field::reduce_mod_subspace_and_get_coefficient_vector(
	INT k, INT len, INT *basis, INT *base_cols, 
	INT *v, INT *coefficients, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, idx;
	
	if (f_v) {
		cout << "finite_field::reduce_mod_subspace_and_get_"
				"coefficient_vector: v=";
		INT_vec_print(cout, v, len);
		cout << endl;
		}
	if (f_vv) {
		cout << "finite_field::reduce_mod_subspace_and_get_"
				"coefficient_vector subspace basis:" << endl;
		print_integer_matrix_width(cout, basis, k, len, len, log10_of_q);
		}
	for (i = 0; i < k; i++) {
		idx = base_cols[i];
		if (basis[i * len + idx] != 1) {
			cout << "finite_field::reduce_mod_subspace_and_get_"
					"coefficient_vector pivot entry is not one" << endl;
			cout << "i=" << i << endl;
			cout << "idx=" << idx << endl;
			print_integer_matrix_width(cout, basis,
					k, len, len, log10_of_q);
			exit(1);
			}
		coefficients[i] = v[idx];
		if (v[idx]) {
			Gauss_step(basis + i * len, v, len, idx, 0/*verbose_level*/);
			if (v[idx]) {
				cout << "finite_field::reduce_mod_subspace_and_get_"
						"coefficient_vector fatal: v[idx]" << endl;
				exit(1);
				}
			}
		}
	if (f_v) {
		cout << "finite_field::reduce_mod_subspace_and_get_"
				"coefficient_vector after: v=";
		INT_vec_print(cout, v, len);
		cout << endl;
		cout << "coefficients=";
		INT_vec_print(cout, coefficients, k);
		cout << endl;
		}
}

void finite_field::reduce_mod_subspace(INT k,
	INT len, INT *basis, INT *base_cols,
	INT *v, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, idx;
	
	if (f_v) {
		cout << "finite_field::reduce_mod_subspace before: v=";
		INT_vec_print(cout, v, len);
		cout << endl;
		}
	if (f_vv) {
		cout << "finite_field::reduce_mod_subspace subspace basis:" << endl;
		print_integer_matrix_width(cout, basis, k,
				len, len, log10_of_q);
		}
	for (i = 0; i < k; i++) {
		idx = base_cols[i];
		if (v[idx]) {
			Gauss_step(basis + i * len,
					v, len, idx, 0/*verbose_level*/);
			if (v[idx]) {
				cout << "finite_field::reduce_mod_"
						"subspace fatal: v[idx]" << endl;
				exit(1);
				}
			}
		}
	if (f_v) {
		cout << "finite_field::reduce_mod_subspace after: v=";
		INT_vec_print(cout, v, len);
		cout << endl;
		}
}

INT finite_field::is_contained_in_subspace(INT k,
	INT len, INT *basis, INT *base_cols,
	INT *v, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i;

	if (f_v) {
		cout << "finite_field::is_contained_in_subspace testing v=";
		INT_vec_print(cout, v, len);
		cout << endl;
		}
	reduce_mod_subspace(k, len, basis,
			base_cols, v, verbose_level - 1);
	for (i = 0; i < len; i++) {
		if (v[i]) {
			if (f_v) {
				cout << "finite_field::is_contained_in_subspace "
						"is NOT in the subspace" << endl;
				}
			return FALSE;
			}
		}
	if (f_v) {
		cout << "finite_field::is_contained_in_subspace "
				"is contained in the subspace" << endl;
		}
	return TRUE;
}

void finite_field::compute_and_print_projective_weights(
		INT *M, INT n, INT k)
{
	INT i;
	INT *weights;

	weights = NEW_INT(n + 1);
	code_projective_weight_enumerator(n, k, 
		M, // [k * n]
		weights, // [n + 1]
		0 /*verbose_level*/);


	cout << "projective weights: " << endl;
	for (i = 0; i <= n; i++) {
		if (weights[i] == 0) {
			continue;
			}
		cout << i << " : " << weights[i] << endl;
		}
	FREE_INT(weights);
}

INT finite_field::code_minimum_distance(INT n, INT k,
		INT *code, INT verbose_level)
	// code[k * n]
{
	INT f_v = (verbose_level >= 1);
	INT *weight_enumerator;
	INT i;
	
	if (f_v) {
		cout << "finite_field::code_minimum_distance" << endl;
		}
	weight_enumerator = NEW_INT(n + 1);
	INT_vec_zero(weight_enumerator, n + 1);
	code_weight_enumerator_fast(n, k, 
		code, // [k * n]
		weight_enumerator, // [n + 1]
		verbose_level);
	for (i = 1; i <= n; i++) {
		if (weight_enumerator[i]) {
			break;
			}
		}
	if (i == n + 1) {
		cout << "finite_field::code_minimum_distance "
				"the minimum weight is undefined" << endl;
		exit(1);
		}
	FREE_INT(weight_enumerator);
	return i;
}

void finite_field::codewords_affine(INT n, INT k, 
	INT *code, // [k * n]
	INT *codewords, // q^k
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT N, h, rk;
	INT *msg;
	INT *word;

	if (f_v) {
		cout << "finite_field::codewords_affine" << endl;
		}
	N = nb_AG_elements(k, q);
	if (f_v) {
		cout << N << " messages" << endl;
		}
	msg = NEW_INT(k);
	word = NEW_INT(n);

	for (h = 0; h < N; h++) {
		AG_element_unrank(q, msg, 1, k, h);
		mult_vector_from_the_left(msg, code, word, k, n);
		AG_element_rank(q, word, 1, n, rk);
		codewords[h] = rk;
		}
	FREE_INT(msg);
	FREE_INT(word);
	if (f_v) {
		cout << "finite_field::codewords_affine done" << endl;
		}
}

void finite_field::code_projective_weight_enumerator(
	INT n, INT k,
	INT *code, // [k * n]
	INT *weight_enumerator, // [n + 1]
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 1);
	INT N, h, wt, i;
	INT *msg;
	INT *word;
	INT t0, t1, dt;
	
	t0 = os_ticks();
	
	if (f_v) {
		cout << "finite_field::code_projective_weight_enumerator" << endl;
		}
	N = nb_AG_elements(k, q);
	if (f_v) {
		cout << N << " messages" << endl;
		}
	msg = NEW_INT(k);
	word = NEW_INT(n);

	INT_vec_zero(weight_enumerator, n + 1);
	
	for (h = 0; h < N; h++) {
		if (f_v && (h % ONE_MILLION) == 0) {
			t1 = os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			time_check_delta(cout, dt);
			cout << endl;
			if (f_vv) {
				cout << "so far, the weight enumerator is:" << endl;
				for (i = 0; i <= n; i++) {
					if (weight_enumerator[i] == 0) 
						continue;
					cout << setw(5) << i << " : " << setw(10)
							<< weight_enumerator[i] << endl;
					}
				}
			}
		AG_element_unrank(q, msg, 1, k, h);
		mult_vector_from_the_left(msg, code, word, k, n);
		wt = 0;
		for (i = 0; i < n; i++) {
			if (word[i]) {
				wt++;
				}
			}
		weight_enumerator[wt]++;
		}
	if (f_v) {
		cout << "the weight enumerator is:" << endl;
		for (i = 0; i <= n; i++) {
			if (weight_enumerator[i] == 0) {
				continue;
				}
			cout << setw(5) << i << " : " << setw(10)
					<< weight_enumerator[i] << endl;
			}
		}


	FREE_INT(msg);
	FREE_INT(word);
}

void finite_field::code_weight_enumerator(
	INT n, INT k,
	INT *code, // [k * n]
	INT *weight_enumerator, // [n + 1]
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 1);
	INT N, h, wt, i;
	INT *msg;
	INT *word;
	INT t0, t1, dt;
	
	t0 = os_ticks();
	
	if (f_v) {
		cout << "finite_field::code_weight_enumerator" << endl;
		}
	N = nb_AG_elements(k, q);
	if (f_v) {
		cout << N << " messages" << endl;
		}
	msg = NEW_INT(k);
	word = NEW_INT(n);

	INT_vec_zero(weight_enumerator, n + 1);
	
	for (h = 0; h < N; h++) {
		if ((h % ONE_MILLION) == 0) {
			t1 = os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			time_check_delta(cout, dt);
			cout << endl;
			if (f_vv) {
				cout << "so far, the weight enumerator is:" << endl;
				for (i = 0; i <= n; i++) {
					if (weight_enumerator[i] == 0) 
						continue;
					cout << setw(5) << i << " : " << setw(10)
							<< weight_enumerator[i] << endl;
					}
				}
			}
		AG_element_unrank(q, msg, 1, k, h);
		mult_vector_from_the_left(msg, code, word, k, n);
		wt = 0;
		for (i = 0; i < n; i++) {
			if (word[i]) {
				wt++;
				}
			}
		weight_enumerator[wt]++;
		}
	if (f_v) {
		cout << "the weight enumerator is:" << endl;
		for (i = 0; i <= n; i++) {
			if (weight_enumerator[i] == 0) {
				continue;
				}
			cout << setw(5) << i << " : " << setw(10)
					<< weight_enumerator[i] << endl;
			}
		}


	FREE_INT(msg);
	FREE_INT(word);
}


void finite_field::code_weight_enumerator_fast(INT n, INT k, 
	INT *code, // [k * n]
	INT *weight_enumerator, // [n + 1]
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT N, h, wt, i;
	INT *msg;
	INT *word;
	INT t0, t1, dt;
	
	t0 = os_ticks();
	
	if (f_v) {
		cout << "finite_field::code_weight_enumerator" << endl;
		}
	N = nb_PG_elements(k - 1, q);
	if (f_v) {
		cout << N << " projective messages" << endl;
		}
	msg = NEW_INT(k);
	word = NEW_INT(n);


	INT_vec_zero(weight_enumerator, n + 1);
	
	for (h = 0; h < N; h++) {
		if (((h % ONE_MILLION) == 0) && h) {
			t1 = os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			time_check_delta(cout, dt);
			cout << endl;
			if (f_vv) {
				cout << "so far, the weight enumerator is:" << endl;
				for (i = 0; i <= n; i++) {
					if (weight_enumerator[i] == 0) 
						continue;
					cout << setw(5) << i << " : " << setw(10)
							<< (q - 1) * weight_enumerator[i] << endl;
					}
				}
			}
		PG_element_unrank_modified(*this, msg, 1, k, h);
		//AG_element_unrank(q, msg, 1, k, h);
		mult_vector_from_the_left(msg, code, word, k, n);
		wt = 0;
		for (i = 0; i < n; i++) {
			if (word[i]) {
				wt++;
				}
			}
		weight_enumerator[wt]++;
		if (f_vv) {
			cout << h << " / " << N << " msg: ";
			INT_vec_print(cout, msg, k);
			cout << " codeword ";
			INT_vec_print(cout, word, n);
			cout << " weight " << wt << endl;
			}
		}
	weight_enumerator[0] = 1;
	for (i = 1; i <= n; i++) {
		weight_enumerator[i] *= q - 1;
		}
	if (f_v) {
		cout << "the weight enumerator is:" << endl;
		for (i = 0; i <= n; i++) {
			if (weight_enumerator[i] == 0) {
				continue;
				}
			cout << setw(5) << i << " : " << setw(10)
					<< weight_enumerator[i] << endl;
			}
		}


	FREE_INT(msg);
	FREE_INT(word);
}

void finite_field::code_projective_weights(
	INT n, INT k,
	INT *code, // [k * n]
	INT *&weights, // will be allocated [N]
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 1);
	INT N, h, wt, i;
	INT *msg;
	INT *word;
	INT t0, t1, dt;
	
	t0 = os_ticks();
	
	if (f_v) {
		cout << "finite_field::code_projective_weights" << endl;
		}
	N = nb_PG_elements(k - 1, q);
	if (f_v) {
		cout << N << " projective messages" << endl;
		}
	weights = NEW_INT(N);
	msg = NEW_INT(k);
	word = NEW_INT(n);
	
	for (h = 0; h < N; h++) {
		if ((h % ONE_MILLION) == 0) {
			t1 = os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			time_check_delta(cout, dt);
			cout << endl;
			}
		PG_element_unrank_modified(*this, msg, 1, k, h);
		//AG_element_unrank(q, msg, 1, k, h);
		mult_vector_from_the_left(msg, code, word, k, n);
		wt = 0;
		for (i = 0; i < n; i++) {
			if (word[i]) {
				wt++;
				}
			}
		weights[h] = wt;
		}
	if (f_v) {
		cout << "finite_field::code_projective_weights done" << endl;
		}


	FREE_INT(msg);
	FREE_INT(word);
}

INT finite_field::is_subspace(INT d, INT dim_U,
		INT *Basis_U, INT dim_V, INT *Basis_V,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Basis;
	INT h, rk, ret;

	if (f_v) {
		cout << "finite_field::is_subspace" << endl;
		}
	Basis = NEW_INT((dim_V + 1) * d);
	for (h = 0; h < dim_U; h++) {

		INT_vec_copy(Basis_V, Basis, dim_V * d);
		INT_vec_copy(Basis_U + h * d, Basis + dim_V * d, d);
		rk = Gauss_easy(Basis, dim_V + 1, d);
		if (rk > dim_V) {
			ret = FALSE;
			goto done;
			}
		}
	ret = TRUE;
done:
	FREE_INT(Basis);
	return ret;
}

void finite_field::Kronecker_product(INT *A, INT *B, 
	INT n, INT *AB)
{
	INT i, j, I, J, u, v, a, b, c, n2;

	n2 = n * n;
	for (I = 0; I < n; I++) {
		for (J = 0; J < n; J++) {
			b = B[I * n + J];
			for (i = 0; i < n; i++) {
				for (j = 0; j < n; j++) {
					a = A[i * n + j];
					c = mult(a, b);
					u = I * n + i;
					v = J * n + j;
					AB[u * n2 + v] = c;
					}
				}
			}
		}
}

void finite_field::Kronecker_product_square_but_arbitrary(
	INT *A, INT *B,
	INT na, INT nb, INT *AB, INT &N,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, I, J, u, v, a, b, c;

	if (f_v) {
		cout << "finite_field::Kronecker_product_square_but_arbitrary"
				<< endl;
		cout << "na=" << na << endl;
		cout << "nb=" << nb << endl;
		cout << "N=" << N << endl;
		}
	N = na * nb;
	for (I = 0; I < nb; I++) {
		for (J = 0; J < nb; J++) {
			b = B[I * nb + J];
			for (i = 0; i < na; i++) {
				for (j = 0; j < na; j++) {
					a = A[i * na + j];
					c = mult(a, b);
					u = I * na + i;
					v = J * na + j;
					AB[u * N + v] = c;
					}
				}
			}
		}
	if (f_v) {
		cout << "finite_field::Kronecker_product_square_but_arbitrary "
				"done" << endl;
		}
}

INT finite_field::dependency(INT d,
		INT *v, INT *A, INT m, INT *rho,
		INT verbose_level)
// Lueneburg~\cite{Lueneburg87a} p. 104.
// A is a matrix of size d + 1 times d
// v[d]
// rho is a column permutation of degree d
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, k, deg, f_null, c;

	if (f_v) {
		cout << "finite_field::dependency" << endl;
		cout << "m = " << m << endl;
		}
	deg = d;
	if (f_vv) {
		cout << "finite_field::dependency A=" << endl;
		INT_matrix_print(A, m, deg);
		cout << "v = ";
		::INT_vec_print(cout, v, deg);
		cout << endl;
		}
	// fill the m-th row of matrix A with v^rho: 
	for (j = 0; j < deg; j++) {
		A[m * deg + j] = v[rho[j]];
		}
	if (f_vv) {
		cout << "finite_field::dependency "
				"after putting in row " << m << " A=" << endl;
		INT_matrix_print(A, m + 1, deg);
		cout << "rho = ";
		::INT_vec_print(cout, rho, deg);
		cout << endl;
		}
	for (k = 0; k < m; k++) {
	
		if (f_vv) {
			cout << "finite_field::dependency "
					"k=" << k << " A=" << endl;
			INT_matrix_print(A, m + 1, deg);
			}
		
		for (j = k + 1; j < deg; j++) {
		
			if (f_vv) {
				cout << "finite_field::dependency "
						"j=" << j << endl;
				}
		
			A[m * deg + j] = mult(A[k * deg + k], A[m * deg + j]);

			c = negate(mult(A[m * deg + k], A[k * deg + j]));

			A[m * deg + j] = add(A[m * deg + j], c);
			
			if (k > 0) {
				c = inverse(A[(k - 1) * deg + k - 1]);
				A[m * deg + j] = mult(A[m * deg + j], c);
				}
			} // next j 

		if (f_vv) {
			cout << "finite_field::dependency "
					"k=" << k << " done, A=" << endl;
			INT_matrix_print(A, m + 1, deg);
			}

		} // next k 
	if (f_vv) {
		cout << "finite_field::dependency "
				"m=" << m << " after reapply, A=" << endl;
		INT_matrix_print(A, m + 1, deg);
		cout << "rho = ";
		::INT_vec_print(cout, rho, deg);
		cout << endl;
		}
	
	f_null = (m == deg);
	if (!f_null) {
	
		// search for an non-zero entry
		// in row m starting in column m.
		// permute that column into column m,
		// change the col-permutation rho
		j = m;
		while ((A[m * deg + j] == 0) && (j < deg - 1)) {
			j++;
			}
		f_null = (A[m * deg + j] == 0);
		if (!f_null && j > m) {
			if (f_vv) {
				cout << "finite_field::dependency "
						"choosing column " << j << endl;
				}

			// swapping columns i and j:

			for (i = 0; i <= m; i++) {
				c = A[i * deg + m];
				A[i * deg + m] = A[i * deg + j];
				A[i * deg + j] = c;
				} // next i

			// updating the permutation rho:
			c = rho[m];
			rho[m] = rho[j];
			rho[j] = c;
			}
		}
	if (f_vv) {
		cout << "finite_field::dependency m=" << m
				<< " after pivoting, A=" << endl;
		INT_matrix_print(A, m + 1, deg);
		cout << "rho = ";
		::INT_vec_print(cout, rho, deg);
		cout << endl;
		}
	
	if (f_v) {
		cout << "finite_field::dependency "
				"done, f_null = " << f_null << endl;
		}
	return f_null;
}

void finite_field::order_ideal_generator(INT d,
	INT idx, INT *mue, INT &mue_deg,
	INT *A, INT *Frobenius, 
	INT verbose_level)
// Lueneburg~\cite{Lueneburg87a} p. 105.
// Frobenius is a matrix of size d x d
// A is (d + 1) x d
// mue[d + 1]
{
	INT f_v = (verbose_level >= 1);
	INT deg;
	INT *v, *v1, *rho;
	INT i, j, m, a, f_null;
	
	if (f_v) {
		cout << "finite_field::order_ideal_generator "
				"d = " << d << " idx = " << idx << endl;
		}
	deg = d;

	v = NEW_INT(deg);
	v1 = NEW_INT(deg);
	rho = NEW_INT(deg);

	// make v the idx-th unit vector:
	INT_vec_zero(v, deg);
	v[idx] = 1;

	// make rho the identity permutation:
	for (i = 0; i < deg; i++) {
		rho[i] = i;
		}
	
	m = 0;
	f_null = dependency(d, v, A, m, rho, verbose_level - 1);

	while (!f_null) {
	
		// apply frobenius
		// (the images are written in the columns):
		
		mult_vector_from_the_right(Frobenius, v, v1, deg, deg);
		INT_vec_copy(v1, v, deg);
		
		m++;
		f_null = dependency(d, v, A, m, rho, verbose_level - 1);
		
		if (m == deg && !f_null) {
			cout << "finite_field::order_ideal_generator "
					"m == deg && ! f_null" << endl;
			exit(1);
			}
		}
	
	mue_deg = m;
	mue[m] = 1;
	for (j = m - 1; j >= 0; j--) {
		mue[j] = A[m * deg + j];
		if (f_v) {
			cout << "finite_field::order_ideal_generator "
					"mue[" << j << "] = " << mue[j] << endl;
			}
		for (i = m - 1; i >= j + 1; i--) {
			a = mult(mue[i], A[i * deg + j]);
			mue[j] = add(mue[j], a);
			if (f_v) {
				cout << "finite_field::order_ideal_generator "
						"mue[" << j << "] = " << mue[j] << endl;
				}
			}
		a = negate(inverse(A[j * deg + j]));
		mue[j] = mult(mue[j], a);
			//g_asr(- mue[j] * -
			// g_inv_mod(Normal_basis[j * dim_nb + j], chi), chi);
		if (f_v) {
			cout << "finite_field::order_ideal_generator "
					"mue[" << j << "] = " << mue[j] << endl;
			}
		}
	
	if (f_v) {
		cout << "finite_field::order_ideal_generator "
				"after preparing mue:" << endl;
		cout << "mue_deg = " << mue_deg << endl;
		cout << "mue = ";
		::INT_vec_print(cout, mue, mue_deg + 1);
		cout << endl;
		}

	FREE_INT(v);
	FREE_INT(v1);
	FREE_INT(rho);
	if (f_v) {
		cout << "finite_field::order_ideal_generator done" << endl;
		}
}

void finite_field::span_cyclic_module(INT *A,
		INT *v, INT n, INT *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *w1, *w2;
	INT i, j;

	if (f_v) {
		cout << "finite_field::span_cyclic_module" << endl;
		}
	w1 = NEW_INT(n);
	w2 = NEW_INT(n);
	INT_vec_copy(v, w1, n);
	for (j = 0; j < n; j++) {

		// put w1 in the j-th column of A:
		for (i = 0; i < n; i++) {
			A[i * n + j] = w1[i];
			}
		mult_vector_from_the_right(Mtx, w1, w2, n, n);
		INT_vec_copy(w2, w1, n);
		}

	FREE_INT(w1);
	FREE_INT(w2);
	if (f_v) {
		cout << "finite_field::span_cyclic_module done" << endl;
		}
}

void finite_field::random_invertible_matrix(INT *M,
		INT k, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *N;
	INT i, qk, r, rk;

	if (f_v) {
		cout << "finite_field::random_invertible_matrix" << endl;
		}
	qk = i_power_j(q, k);
	N = NEW_INT(k * k);
	for (i = 0; i < k; i++) {
		if (f_vv) {
			cout << "i=" << i << endl;
			}
		while (TRUE) {
			r = random_integer(qk);
			if (f_vv) {
				cout << "r=" << r << endl;
				}
			AG_element_unrank(q, M + i * k, 1, k, r);
			if (f_vv) {
				::INT_matrix_print(M, i + 1, k);
				}
			
			INT_vec_copy(M, N, (i + 1) * k);
			rk = Gauss_easy(N, i + 1, k);
			if (f_vv) {
				cout << "rk=" << rk << endl;
				}
			if (rk == i + 1) {
				if (f_vv) {
					cout << "has full rank" << endl;
					}
				break;
				}
			}
		}
	if (f_v) {
		cout << "finite_field::random_invertible_matrix "
				"Random invertible matrix:" << endl;
		INT_matrix_print(M, k, k);
		}
	FREE_INT(N);
}

void finite_field::make_all_irreducible_polynomials_of_degree_d(
		INT d, INT &nb, INT *&Table, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT p, e, i, Q;
	INT cnt;

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << q << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}

	cnt = count_all_irreducible_polynomials_of_degree_d(d, verbose_level - 2);

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"cnt = " << cnt << endl;
		}

	nb = cnt;

	Table = NEW_INT(nb * (d + 1));


	Q = i_power_j(q, d);
	
	factor_prime_power(q, p, e);
	
	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"p=" << p << " e=" << e << endl;
		}
	
	//finite_field Fp;
	//Fp.init(p, 0 /*verbose_level*/);
	unipoly_domain FX(this);

	const BYTE *poly;

	poly = get_primitive_polynomial(q, d, 0 /* verbose_level */);

	unipoly_object m;
	unipoly_object g;
	unipoly_object minpol;


	FX.create_object_by_rank_string(m, poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial m = ";
		FX.print_object(m, cout);
		cout << endl;
		}

	FX.create_object_by_rank(g, 0);
	FX.create_object_by_rank(minpol, 0);

	INT *Frobenius;
	INT *Normal_basis;
	INT *v;
	INT *w;

	//Frobenius = NEW_INT(d * d);
	Normal_basis = NEW_INT(d * d);
	v = NEW_INT(d);
	w = NEW_INT(d);

	FX.Frobenius_matrix(Frobenius, m, verbose_level - 3);
	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"Frobenius_matrix = " << endl;
		INT_matrix_print(Frobenius, d, d);
		cout << endl;
		}

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"before compute_normal_basis" << endl;
		}
	FX.compute_normal_basis(d, Normal_basis, Frobenius, verbose_level - 3);

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"Normal_basis = " << endl;
		INT_matrix_print(Normal_basis, d, d);
		cout << endl;
		}

	cnt = 0;
	INT_vec_first_regular_word(v, d, Q, q);
	while (TRUE) {
		if (f_vv) {
			cout << "finite_field::make_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : v = ";
			INT_vec_print(cout, v, d);
			cout << endl;
			}

		FX.gfq->mult_vector_from_the_right(Normal_basis, v, w, d, d);
		if (f_vv) {
			cout << "finite_field::make_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : w = ";
			INT_vec_print(cout, w, d);
			cout << endl;
			}

		FX.delete_object(g);
		FX.create_object_of_degree(g, d - 1);
		for (i = 0; i < d; i++) {
			((INT *) g)[1 + i] = w[i];
			}

		FX.minimum_polynomial_extension_field(g, m, minpol, d,
				Frobenius, verbose_level - 3);
		if (f_vv) {
			cout << "finite_field::make_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : v = ";
			INT_vec_print(cout, v, d);
			cout << " irreducible polynomial = ";
			FX.print_object(minpol, cout);
			cout << endl;
			}

		for (i = 0; i <= d; i++) {
			Table[cnt * (d + 1) + i] = ((INT *)minpol)[1 + i];
			}

		
		cnt++;


		if (!INT_vec_next_regular_word(v, d, Q, q)) {
			break;
			}

		}

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_"
				"of_degree_d there are " << cnt
				<< " irreducible polynomials "
				"of degree " << d << " over " << "F_" << q << endl;
		}

	FREE_INT(Frobenius);
	FREE_INT(Normal_basis);
	FREE_INT(v);
	FREE_INT(w);
	FX.delete_object(m);
	FX.delete_object(g);
	FX.delete_object(minpol);


	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << q << " done" << endl;
		}
}

INT finite_field::count_all_irreducible_polynomials_of_degree_d(
		INT d, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT p, e, i, Q;
	INT cnt;

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << q << endl;
		}

	Q = i_power_j(q, d);
	
	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"Q=" << Q << endl;
		}
	factor_prime_power(q, p, e);
	
	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"p=" << p << " e=" << e << endl;
		}
	if (e > 1) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"e=" << e << " is greater than one" << endl;
		}
	
	//finite_field Fp;
	//Fp.init(p, 0 /*verbose_level*/);
	unipoly_domain FX(this);

	const BYTE *poly;

	poly = get_primitive_polynomial(q, d, 0 /* verbose_level */);

	unipoly_object m;
	unipoly_object g;
	unipoly_object minpol;


	FX.create_object_by_rank_string(m,
			poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial m = ";
		FX.print_object(m, cout);
		cout << endl;
		}

	FX.create_object_by_rank(g, 0);
	FX.create_object_by_rank(minpol, 0);

	INT *Frobenius;
	INT *F2;
	INT *Normal_basis;
	INT *v;
	INT *w;

	//Frobenius = NEW_INT(d * d);
	F2 = NEW_INT(d * d);
	Normal_basis = NEW_INT(d * d);
	v = NEW_INT(d);
	w = NEW_INT(d);

	FX.Frobenius_matrix(Frobenius, m, verbose_level - 3);
	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"Frobenius_matrix = " << endl;
		INT_matrix_print(Frobenius, d, d);
		cout << endl;
		}

	mult_matrix(Frobenius, Frobenius, F2, d, d, d);
	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"Frobenius^2 = " << endl;
		INT_matrix_print(F2, d, d);
		cout << endl;
		}
	

	FX.compute_normal_basis(d, Normal_basis, Frobenius, verbose_level - 3);

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"Normal_basis = " << endl;
		INT_matrix_print(Normal_basis, d, d);
		cout << endl;
		}

	cnt = 0;
	INT_vec_first_regular_word(v, d, Q, q);
	while (TRUE) {
		if (f_vv) {
			cout << "finite_field::count_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : v = ";
			INT_vec_print(cout, v, d);
			cout << endl;
			}

		FX.gfq->mult_vector_from_the_right(Normal_basis, v, w, d, d);
		if (f_vv) {
			cout << "finite_field::count_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : w = ";
			INT_vec_print(cout, w, d);
			cout << endl;
			}

		FX.delete_object(g);
		FX.create_object_of_degree(g, d - 1);
		for (i = 0; i < d; i++) {
			((INT *) g)[1 + i] = w[i];
			}

		FX.minimum_polynomial_extension_field(g, m, minpol, d,
				Frobenius, verbose_level - 3);
		if (f_vv) {
			cout << "finite_field::count_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : v = ";
			INT_vec_print(cout, v, d);
			cout << " irreducible polynomial = ";
			FX.print_object(minpol, cout);
			cout << endl;
			}
		if (FX.degree(minpol) != d) {
			cout << "finite_field::count_all_irreducible_polynomials_"
					"of_degree_d The polynomial does not have degree d"
					<< endl;
			FX.print_object(minpol, cout);
			cout << endl;
			exit(1);
			}
		if (!FX.is_irreducible(minpol, verbose_level)) {
			cout << "finite_field::count_all_irreducible_polynomials_"
					"of_degree_d The polynomial is not irreducible" << endl;
			FX.print_object(minpol, cout);
			cout << endl;
			exit(1);
			}

		
		cnt++;

		if (!INT_vec_next_regular_word(v, d, Q, q)) {
			break;
			}

		}

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_"
				"of_degree_d there are " << cnt << " irreducible polynomials "
				"of degree " << d << " over " << "F_" << q << endl;
		}

	FREE_INT(Frobenius);
	FREE_INT(F2);
	FREE_INT(Normal_basis);
	FREE_INT(v);
	FREE_INT(w);
	FX.delete_object(m);
	FX.delete_object(g);
	FX.delete_object(minpol);

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_"
				"of_degree_d done" << endl;
		}
	return cnt;
}

void finite_field::adjust_basis(INT *V, INT *U,
		INT n, INT k, INT d, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, ii, b;
	INT *base_cols;
	INT *M;

	if (f_v) {
		cout << "finite_field::adjust_basis" << endl;
		}
	base_cols = NEW_INT(n);
	M = NEW_INT(k * n);

	INT_vec_copy(U, M, d * n);
	
	if (Gauss_simple(M, d, n, base_cols,
			0 /* verbose_level */) != d) {
		cout << "finite_field::adjust_basis rank "
				"of matrix is not d" << endl;
		exit(1);
		}
	ii = 0;
	for (i = 0; i < k; i++) {
		INT_vec_copy(V + i * n, M + (d + ii) * n, n);
		for (j = 0; j < d; j++) {
			b = base_cols[j];
			Gauss_step(M + b * n, M + (d + ii) * n,
					n, b, 0 /* verbose_level */);
			}
		if (INT_vec_is_zero(M + (d + ii) * n, n)) {
			}
		else {
			ii++;
			}
		}
	if (d + ii != k) {
		cout << "finite_field::adjust_basis d + ii != k" << endl;
		exit(1);
		}
	INT_vec_copy(M, V, k * n);
	

	FREE_INT(M);
	FREE_INT(base_cols);
	if (f_v) {
		cout << "finite_field::adjust_basis done" << endl;
		}
}

void finite_field::choose_vector_in_here_but_not_in_here_column_spaces(
		INT_matrix *V, INT_matrix *W, INT *v,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT n, k, d;
	INT *Gen;
	INT *base_cols;
	INT i, j, ii, b;

	if (f_v) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_"
				"column_spaces" << endl;
		}
	n = V->m;
	if (V->m != W->m) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_"
				"column_spaces V->m != W->m" << endl;
		exit(1);
		}
	k = V->n;
	d = W->n;
	if (d >= k) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_"
				"column_spaces W->n >= V->n" << endl;
		exit(1);
		}
	Gen = NEW_INT(k * n);
	base_cols = NEW_INT(n);
	
	for (i = 0; i < d; i++) {
		for (j = 0; j < n; j++) {
			Gen[i * n + j] = W->s_ij(j, i);
			}
		}
	if (Gauss_simple(Gen, d, n,
			base_cols, 0 /* verbose_level */) != d) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_"
				"column_spaces rank of matrix is not d" << endl;
		exit(1);
		}
	ii = 0;
	for (i = 0; i < k; i++) {
		for (j = 0; j < n; j++) {
			Gen[(d + ii) * n + j] = V->s_ij(j, i);
			}
		b = base_cols[i];
		Gauss_step(Gen + b * n, Gen + (d + ii) * n,
				n, b, 0 /* verbose_level */);
		if (INT_vec_is_zero(Gen + (d + ii) * n, n)) {
			}
		else {
			ii++;
			}
		}
	if (d + ii != k) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_"
				"column_spaces d + ii != k" << endl;
		exit(1);
		}
	INT_vec_copy(Gen + d * n, v, n);
	

	FREE_INT(Gen);
	FREE_INT(base_cols);
	if (f_v) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_"
				"column_spaces done" << endl;
		}
}

void finite_field::choose_vector_in_here_but_not_in_here_or_here_column_spaces(
	INT_matrix *V,
	INT_matrix *W1, INT_matrix *W2, INT *v, 
	INT verbose_level)
{

	INT coset = 0;

	choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
			coset, V, W1, W2, v, verbose_level);

}

INT finite_field::choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
	INT &coset,
	INT_matrix *V, INT_matrix *W1, INT_matrix *W2, INT *v, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT n, k, d1, d2, rk;
	INT *Gen;
	INT *base_cols;
	INT *w;
	INT *z;
	INT i, j, b;
	INT ret = TRUE;

	if (f_v) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_or_here_"
				"column_spaces_coset coset=" << coset << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	if (f_vv) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_or_here_"
				"column_spaces_coset" << endl;
		cout << "V=" << endl;
		V->print();
		cout << "W1=" << endl;
		W1->print();
		cout << "W2=" << endl;
		W2->print();
		}
	n = V->m;
	if (V->m != W1->m) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_or_here_"
				"column_spaces_coset V->m != W1->m" << endl;
		exit(1);
		}
	if (V->m != W2->m) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_or_here_"
				"column_spaces_coset V->m != W2->m" << endl;
		exit(1);
		}
	k = V->n;
	d1 = W1->n;
	d2 = W2->n;
	if (d1 >= k) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_or_here_"
				"column_spaces_coset W1->n >= V->n" << endl;
		exit(1);
		}
	Gen = NEW_INT((d1 + d2 + k) * n);
	base_cols = NEW_INT(n);
	w = NEW_INT(k);
	z = NEW_INT(n);
	
	for (i = 0; i < d1; i++) {
		for (j = 0; j < n; j++) {
			Gen[i * n + j] = W1->s_ij(j, i);
			}
		}
	for (i = 0; i < d2; i++) {
		for (j = 0; j < n; j++) {
			Gen[(d1 + i) * n + j] = W2->s_ij(j, i);
			}
		}
	rk = Gauss_simple(Gen, d1 + d2, n, base_cols, 0 /* verbose_level */);


	INT a;

	while (TRUE) {
		if (coset >= i_power_j(q, k)) {
			if (f_vv) {
				cout << "coset = " << coset << " = " << i_power_j(q, k)
						<< " break" << endl;
				}
			ret = FALSE;
			break;
			}
		AG_element_unrank(q, w, 1, k, coset);

		if (f_vv) {
			cout << "coset=" << coset << " w=";
			INT_vec_print(cout, w, k);
			cout << endl;
			}

		coset++;

		// get a linear combination of the generators of V:
		for (j = 0; j < n; j++) {
			Gen[rk * n + j] = 0;
			for (i = 0; i < k; i++) {
				a = w[i];
				Gen[rk * n + j] = add(Gen[rk * n + j], mult(a, V->s_ij(j, i)));
				}
			}
		INT_vec_copy(Gen + rk * n, z, n);
		if (f_vv) {
			cout << "before reduce=";
			INT_vec_print(cout, Gen + rk * n, n);
			cout << endl;
			}

		// reduce modulo the subspace:
		for (j = 0; j < rk; j++) {
			b = base_cols[j];
			Gauss_step(Gen + j * n, Gen + rk * n, n, b, 0 /* verbose_level */);
			}

		if (f_vv) {
			cout << "after reduce=";
			INT_vec_print(cout, Gen + rk * n, n);
			cout << endl;
			}

		
		// see if we got something nonzero:
		if (!INT_vec_is_zero(Gen + rk * n, n)) {
			break;
			}
		// keep moving on to the next vector

		} // while
	
	INT_vec_copy(z, v, n);
	

	FREE_INT(Gen);
	FREE_INT(base_cols);
	FREE_INT(w);
	FREE_INT(z);
	if (f_v) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_"
				"or_here_column_spaces_coset done ret = " << ret << endl;
		}
	return ret;
}

void finite_field::vector_add_apply(INT *v, INT *w, INT c, INT n)
{
	INT i;

	for (i = 0; i < n; i++) {
		v[i] = add(v[i], mult(c, w[i]));
		}
}

void finite_field::vector_add_apply_with_stride(INT *v, INT *w,
		INT stride, INT c, INT n)
{
	INT i;

	for (i = 0; i < n; i++) {
		v[i] = add(v[i], mult(c, w[i * stride]));
		}
}

INT finite_field::test_if_commute(INT *A, INT *B, INT k, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *M1, *M2;
	INT ret;

	if (f_v) {
		cout << "finite_field::test_if_commute" << endl;
		}
	M1 = NEW_INT(k * k);
	M2 = NEW_INT(k * k);

	mult_matrix_matrix(A, B, M1, k, k, k);
	mult_matrix_matrix(B, A, M2, k, k, k);
	if (INT_vec_compare(M1, M2, k * k) == 0) {
		ret = TRUE;
		}
	else {
		ret = FALSE;
		}

	FREE_INT(M1);
	FREE_INT(M2);
	if (f_v) {
		cout << "finite_field::test_if_commute done" << endl;
		}
	return ret;
}

void finite_field::unrank_point_in_PG(INT *v, INT len, INT rk)
// len is the length of the vector, not the projective dimension
{

	PG_element_unrank_modified(*this, v, 1 /* stride */, len, rk);
}

INT finite_field::rank_point_in_PG(INT *v, INT len)
{
	INT rk;

	PG_element_rank_modified(*this, v, 1 /* stride */, len, rk);
	return rk;
}

INT finite_field::nb_points_in_PG(INT n)
// n is projective dimension
{
	INT N;

	N = nb_PG_elements(n, q);
	return N;
}

void finite_field::Borel_decomposition(INT n, INT *M,
		INT *B1, INT *B2, INT *pivots, INT verbose_level)
{
	//verbose_level = 1;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i, j, a, av, b, c, h, k, d, e, mc;
	INT *f_is_pivot;

	if (f_v) {
		cout << "finite_field::Borel_decomposition" << endl;
		}
	if (f_v) {
		cout << "finite_field::Borel_decomposition input matrix:" << endl;
		cout << "M:" << endl;
		INT_matrix_print(M, n, n);
		}

	identity_matrix(B1, n);
	identity_matrix(B2, n);


	f_is_pivot = NEW_INT(n);
	for (i = 0; i < n; i++) {
		f_is_pivot[i] = FALSE;
		}
	if (f_v) {
		cout << "finite_field::Borel_decomposition going down "
				"from the right" << endl;
		}
	for (j = n - 1; j >= 0; j--) {
		for (i = 0; i < n; i++) {
			if (f_is_pivot[i]) {
				continue;
				}
			if (M[i * n + j]) {
				if (f_v) {
					cout << "finite_field::Borel_decomposition pivot "
							"at (" << i << " " << j << ")" << endl;
					}
				f_is_pivot[i] = TRUE;
				pivots[j] = i;
				a = M[i * n + j];
				av = inverse(a);

				// we can only go down:
				for (h = i + 1; h < n; h++) {
					b = M[h * n + j];
					if (b) {
						c = mult(av, b);
						mc = negate(c);
						for (k = 0; k < n; k++) {
							d = mult(M[i * n + k], mc);
							e = add(M[h * n + k], d);
							M[h * n + k] = e;
							}
						//mc = negate(c);
						//cout << "finite_field::Borel_decomposition "
						// "i=" << i << " h=" << h << " mc=" << mc << endl;
						// multiply the inverse of the elementary matrix
						// to the right of B1:
						for (k = 0; k < n; k++) {
							d = mult(B1[k * n + h], c);
							e = add(B1[k * n + i], d);
							B1[k * n + i] = e;
							}
						//cout << "finite_field::Borel_decomposition B1:"
						//<< endl;
						//INT_matrix_print(B1, n, n);
						}
					}
				if (f_v) {
					cout << "finite_field::Borel_decomposition after going "
							"down in column " << j << endl;
					cout << "M:" << endl;
					INT_matrix_print(M, n, n);
					}

				// now we go to the left from the pivot:
				for (h = 0; h < j; h++) {
					b = M[i * n + h];
					if (b) {
						c = mult(av, b);
						mc = negate(c);
						for (k = i; k < n; k++) {
							d = mult(M[k * n + j], mc);
							e = add(M[k * n + h], d);
							M[k * n + h] = e;
							}
						//mc = negate(c);
						//cout << "finite_field::Borel_decomposition "
						// "j=" << j << " h=" << h << " mc=" << mc << endl;
						// multiply the inverse of the elementary matrix
						// to the left of B2:
						for (k = 0; k < n; k++) {
							d = mult(B2[h * n + k], c);
							e = add(B2[j * n + k], d);
							B2[j * n + k] = e;
							}
						}
					}
				if (f_v) {
					cout << "finite_field::Borel_decomposition after going "
							"across to the left:" << endl;
					cout << "M:" << endl;
					INT_matrix_print(M, n, n);
					}
				break;
				}
			
			}
		}
	FREE_INT(f_is_pivot);
	if (f_v) {
		cout << "finite_field::Borel_decomposition done" << endl;
		}
}

void finite_field::map_to_standard_frame(INT d, INT *A,
		INT *Transform, INT verbose_level)
// d = vector space dimension
// maps d + 1 points to the frame e_1, e_2, ..., e_d, e_1+e_2+..+e_d 
// A is (d + 1) x d
// Transform is d x d
{
	INT f_v = (verbose_level >= 1);
	INT *B;
	INT *A2;
	INT n, xd, x, i, j;

	if (f_v) {
		cout << "finite_field::map_to_standard_frame" << endl;
		}

	if (f_v) {
		cout << "A=" << endl;
		INT_matrix_print(A, d + 1, d);
		}

	n = d + 1;
	B = NEW_INT(n * n);
	A2 = NEW_INT(d * d);
	for (i = 0; i < n; i++) {
		for (j = 0; j < d; j++) {
			B[j * n + i] = A[i * d + j];
			} 
		}
	if (f_v) {
		cout << "B before=" << endl;
		INT_matrix_print(B, d, n);
		}
	RREF_and_kernel(n, d, B, 0 /* verbose_level */);
	if (f_v) {
		cout << "B after=" << endl;
		INT_matrix_print(B, n, n);
		}
	xd = B[d * n + d - 1];
	x = negate(inverse(xd));
	for (i = 0; i < d; i++) {
		B[d * n + i] = mult(x, B[d * n + i]);
		}
	if (f_v) {
		cout << "last row of B after scaling : " << endl;
		INT_matrix_print(B + d * n, 1, n);
		}
	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			A2[i * d + j] = mult(B[d * n + i], A[i * d + j]);
			}
		}
	if (f_v) {
		cout << "A2=" << endl;
		INT_matrix_print(A2, d, d);
		}
	matrix_inverse(A2, Transform, d, 0 /* verbose_level */);

	FREE_INT(B);
	FREE_INT(A2);
	if (f_v) {
		cout << "finite_field::map_to_standard_frame done" << endl;
		}
}

void finite_field::map_frame_to_frame_with_permutation(INT d,
		INT *A, INT *perm, INT *B, INT *Transform,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *T1;
	INT *T2;
	INT *T3;
	INT *A1;
	INT i, j;
	
	if (f_v) {
		cout << "finite_field::map_frame_to_frame_with_permutation" << endl;
		}
	T1 = NEW_INT(d * d);
	T2 = NEW_INT(d * d);
	T3 = NEW_INT(d * d);
	A1 = NEW_INT((d + 1) * d);

	if (f_v) {
		cout << "permutation: ";
		::INT_vec_print(cout, perm, d + 1);
		cout << endl;
		}
	if (f_v) {
		cout << "A=" << endl;
		INT_matrix_print(A, d + 1, d);
		}
	if (f_v) {
		cout << "B=" << endl;
		INT_matrix_print(B, d + 1, d);
		}
	
	for (i = 0; i < d + 1; i++) {
		j = perm[i];
		INT_vec_copy(A + j * d, A1 + i * d, d);
		}

	if (f_v) {
		cout << "A1=" << endl;
		INT_matrix_print(A1, d + 1, d);
		}


	if (f_v) {
		cout << "mapping A1 to standard frame:" << endl;
		}
	map_to_standard_frame(d, A1, T1, verbose_level);
	if (f_v) {
		cout << "T1=" << endl;
		INT_matrix_print(T1, d, d);
		}
	if (f_v) {
		cout << "mapping B to standard frame:" << endl;
		}
	map_to_standard_frame(d, B, T2, 0 /* verbose_level */);
	if (f_v) {
		cout << "T2=" << endl;
		INT_matrix_print(T2, d, d);
		}
	matrix_inverse(T2, T3, d, 0 /* verbose_level */);
	if (f_v) {
		cout << "T3=" << endl;
		INT_matrix_print(T3, d, d);
		}
	mult_matrix(T1, T3, Transform, d, d, d);
	if (f_v) {
		cout << "Transform=" << endl;
		INT_matrix_print(Transform, d, d);
		}

	FREE_INT(T1);
	FREE_INT(T2);
	FREE_INT(T3);
	FREE_INT(A1);
	if (f_v) {
		cout << "finite_field::map_frame_to_frame_with_permutation done"
				<< endl;
		}
}


void finite_field::map_points_to_points_projectively(INT d, INT k,
		INT *A, INT *B, INT *Transform, INT &nb_maps,
		INT verbose_level)
// A and B are (d + k + 1) x d
// Transform is d x d
// returns TRUE if a map exists
{
	INT f_v = (verbose_level >= 1);
	INT *lehmercode;
	INT *perm;
	INT *A1;
	INT *B1;
	INT *B_set;
	INT *image_set;
	INT *v;
	INT h, i, j, a;
	INT *subset; // [d + k + 1]
	INT nCk;
	INT cnt, overall_cnt;
	
	if (f_v) {
		cout << "finite_field::map_points_to_points_projectively" << endl;
		}
	lehmercode = NEW_INT(d + 1);
	perm = NEW_INT(d + 1);
	A1 = NEW_INT((d + k + 1) * d);
	B1 = NEW_INT((d + k + 1) * d);
	B_set = NEW_INT(d + k + 1);
	image_set = NEW_INT(d + k + 1);
	subset = NEW_INT(d + k + 1);
	v = NEW_INT(d);

	INT_vec_copy(B, B1, (d + k + 1) * d);
	for (i = 0; i < d + k + 1; i++) {
		//PG_element_normalize(*this, B1 + i * d, 1, d);
		PG_element_rank_modified(*this, B1 + i * d, 1, d, a);
		B_set[i] = a;
		}
	INT_vec_heapsort(B_set, d + k + 1);
	if (f_v) {
		cout << "B_set = ";
		::INT_vec_print(cout, B_set, d + k + 1);
		cout << endl;
		}
	

	overall_cnt = 0;
	nCk = INT_n_choose_k(d + k + 1, d + 1);
	for (h = 0; h < nCk; h++) {
		unrank_k_subset(h, subset, d + k + 1, d + 1);
		set_complement(subset, d + 1, subset + d + 1, k, d + k + 1);

		if (f_v) {
			cout << "subset " << h << " / " << nCk << " is ";
			::INT_vec_print(cout, subset, d + 1);
			cout << ", the complement is ";
			::INT_vec_print(cout, subset + d + 1, k);
			cout << endl;
			}


		for (i = 0; i < d + k + 1; i++) {
			j = subset[i];
			INT_vec_copy(A + j * d, A1 + i * d, d);
			}
		if (f_v) {
			cout << "A1=" << endl;
			INT_matrix_print(A1, d + k + 1, d);
			}
		
		cnt = 0;
		first_lehmercode(d + 1, lehmercode);
		while (TRUE) {
			if (f_v) {
				cout << "lehmercode: ";
				::INT_vec_print(cout, lehmercode, d + 1);
				cout << endl;
				}
			lehmercode_to_permutation(d + 1, lehmercode, perm);
			if (f_v) {
				cout << "permutation: ";
				::INT_vec_print(cout, perm, d + 1);
				cout << endl;
				}
			map_frame_to_frame_with_permutation(d, A1, perm,
					B1, Transform, verbose_level);

			for (i = 0; i < d + k + 1; i ++) {
				mult_vector_from_the_left(A1 + i * d, Transform, v, d, d);
				PG_element_rank_modified(*this, v, 1, d, a);
				image_set[i] = a;
				}
			if (f_v) {
				cout << "image_set before sorting: ";
				::INT_vec_print(cout, image_set, d + k + 1);
				cout << endl;
				}
			INT_vec_heapsort(image_set, d + k + 1);
			if (f_v) {
				cout << "image_set after sorting: ";
				::INT_vec_print(cout, image_set, d + k + 1);
				cout << endl;
				}

			if (INT_vec_compare(image_set, B_set, d + k + 1) == 0) {
				cnt++;
				}
			
			if (!next_lehmercode(d + 1, lehmercode)) {
				break;
				}
			}

		cout << "subset " << h << " / " << nCk << " we found "
				<< cnt << " mappings" << endl;
		overall_cnt += cnt;
		}

	FREE_INT(perm);
	FREE_INT(lehmercode);
	FREE_INT(A1);
	FREE_INT(B1);
	FREE_INT(B_set);
	FREE_INT(image_set);
	FREE_INT(subset);
	FREE_INT(v);

	nb_maps = overall_cnt;
	
	if (f_v) {
		cout << "finite_field::map_points_to_points_projectively done"
				<< endl;
		}
}

INT finite_field::BallChowdhury_matrix_entry(INT *Coord,
		INT *C, INT *U, INT k, INT sz_U,
	INT *T, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, d, d1, u, a;

	if (f_v) {
		cout << "finite_field::BallChowdhury_matrix_entry" << endl;
		}
	d = 1;
	for (u = 0; u < sz_U; u++) {
		a = U[u];
		INT_vec_copy(Coord + a * k, T + (k - 1) * k, k);
		for (i = 0; i < k - 1; i++) {
			a = C[i];
			INT_vec_copy(Coord + a * k, T + i * k, k);
			}
		if (f_vv) {
			cout << "u=" << u << " / " << sz_U << " the matrix is:" << endl;
			INT_matrix_print(T, k, k);
			}
		d1 = matrix_determinant(T, k, 0 /* verbose_level */);
		if (f_vv) {
			cout << "determinant = " << d1 << endl;
			}
		d = mult(d, d1);
		}
	if (f_v) {
		cout << "finite_field::BallChowdhury_matrix_entry d=" << d << endl;
		}
	return d;
}

void finite_field::cubic_surface_family_24_generators(
	INT f_with_normalizer,
	INT f_semilinear, 
	INT *&gens, INT &nb_gens, INT &data_size,
	INT &group_order, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT m_one;

	if (f_v) {
		cout << "finite_field::cubic_surface_family_24_generators" << endl;
		}
	m_one = minus_one();
	nb_gens = 3;
	data_size = 16;
	if (f_semilinear) {
		data_size++;
		}
	if (EVEN(q)) {
		group_order = 6;
		}
	else {
		group_order = 24;
		}
	if (f_with_normalizer) {
		nb_gens++;
		group_order *= q - 1;
		}
	gens = NEW_INT(nb_gens * data_size);
	INT_vec_zero(gens, nb_gens * data_size);
		// this sets the field automorphism index
		// to zero if we are semilinear

	gens[0 * data_size + 0 * 4 + 0] = 1;
	gens[0 * data_size + 1 * 4 + 2] = 1;
	gens[0 * data_size + 2 * 4 + 1] = 1;
	gens[0 * data_size + 3 * 4 + 3] = 1;
	gens[1 * data_size + 0 * 4 + 1] = 1;
	gens[1 * data_size + 1 * 4 + 0] = 1;
	gens[1 * data_size + 2 * 4 + 2] = 1;
	gens[1 * data_size + 3 * 4 + 3] = 1;
	gens[2 * data_size + 0 * 4 + 0] = m_one;
	gens[2 * data_size + 1 * 4 + 2] = 1;
	gens[2 * data_size + 2 * 4 + 1] = 1;
	gens[2 * data_size + 3 * 4 + 3] = m_one;
	if (f_with_normalizer) {
		gens[3 * data_size + 0 * 4 + 0] = 1;
		gens[3 * data_size + 1 * 4 + 1] = 1;
		gens[3 * data_size + 2 * 4 + 2] = 1;
		gens[3 * data_size + 3 * 4 + 3] = primitive_root();
		}
	if (f_v) {
		cout << "finite_field::cubic_surface_family_24_generators "
				"done" << endl;
		}
}


