/*
 * linear_algebra2.cpp
 *
 *  Created on: Jan 10, 2022
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace linear_algebra {


void linear_algebra::Kronecker_product(
		int *A, int *B, int n, int *AB)
{
	int i, j, I, J, u, v, a, b, c, n2;

	n2 = n * n;
	for (I = 0; I < n; I++) {
		for (J = 0; J < n; J++) {
			b = B[I * n + J];
			for (i = 0; i < n; i++) {
				for (j = 0; j < n; j++) {
					a = A[i * n + j];
					c = F->mult(a, b);
					u = I * n + i;
					v = J * n + j;
					AB[u * n2 + v] = c;
				}
			}
		}
	}
}

void linear_algebra::Kronecker_product_square_but_arbitrary(
	int *A, int *B,
	int na, int nb, int *AB, int &N,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, I, J, u, v, a, b, c;

	if (f_v) {
		cout << "linear_algebra::Kronecker_product_square_but_arbitrary"
				<< endl;
		cout << "na=" << na << endl;
		cout << "nb=" << nb << endl;
	}
	N = na * nb;
	if (f_v) {
		cout << "N=" << N << endl;
	}
	for (I = 0; I < nb; I++) {
		for (J = 0; J < nb; J++) {
			b = B[I * nb + J];
			for (i = 0; i < na; i++) {
				for (j = 0; j < na; j++) {
					a = A[i * na + j];
					c = F->mult(a, b);
					u = I * na + i;
					v = J * na + j;
					AB[u * N + v] = c;
				}
			}
		}
	}
	if (f_v) {
		cout << "linear_algebra::Kronecker_product_square_but_arbitrary "
				"done" << endl;
	}
}

int linear_algebra::dependency(
		int d,
		int *v, int *A, int m, int *rho,
		int verbose_level)
// Lueneburg~\cite{Lueneburg87a} p. 104.
// A is a matrix of size d + 1 times d
// v[d]
// rho is a column permutation of degree d
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, k, f_null, c;

	if (f_v) {
		cout << "linear_algebra::dependency" << endl;
		cout << "m = " << m << endl;
		cout << "d = " << d << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::dependency A=" << endl;
		Int_matrix_print(A, m, d);
		cout << "v = ";
		Int_vec_print(cout, v, d);
		cout << endl;
	}
	// fill the m-th row of matrix A with v^rho:
	for (j = 0; j < d; j++) {
		A[m * d + j] = v[rho[j]];
	}
	if (f_vv) {
		cout << "linear_algebra::dependency "
				"after putting in row " << m << " A=" << endl;
		Int_matrix_print(A, m + 1, d);
		cout << "rho = ";
		Int_vec_print(cout, rho, d);
		cout << endl;
	}
	for (k = 0; k < m; k++) {

		if (f_vv) {
			cout << "linear_algebra::dependency "
					"k=" << k << " / m=" << m << endl;
			cout << "finite_field::dependency A=" << endl;
			Int_matrix_print(A, m + 1, d);
		}

		for (j = k + 1; j < d; j++) {

			if (f_vv) {
				cout << "linear_algebra::dependency "
						"k=" << k << " / m=" << m
						<< ", j=" << j << " / " << d << endl;
			}


			// a_{m,j} := a_{k,k} * a_{m,j}:

			A[m * d + j] = F->mult(A[k * d + k], A[m * d + j]);


			// a_{m,j} = a_{m,j} - a_{m,k} * a_{k,j}:

			c = F->negate(F->mult(A[m * d + k], A[k * d + j]));
			A[m * d + j] = F->add(A[m * d + j], c);

			if (k > 0) {

				// a_{m,j} := a_{m,j} / a_{k-1,k-1}:

				c = F->inverse(A[(k - 1) * d + k - 1]);
				A[m * d + j] = F->mult(A[m * d + j], c);
			}

			if (f_vv) {
				cout << "linear_algebra::dependency "
						"k=" << k << " / m=" << m
						<< ", j=" << j << " / " << d << " done" << endl;
				cout << "linear_algebra::dependency A=" << endl;
				Int_matrix_print(A, m + 1, d);
			}

		} // next j

		if (f_vv) {
			cout << "linear_algebra::dependency "
					"k=" << k << " / m=" << m << " done" << endl;
			cout << "finite_field::dependency A=" << endl;
			Int_matrix_print(A, m + 1, d);
		}

	} // next k

	if (f_vv) {
		cout << "linear_algebra::dependency "
				"m=" << m << " after reapply, A=" << endl;
		Int_matrix_print(A, m + 1, d);
		cout << "rho = ";
		Int_vec_print(cout, rho, d);
		cout << endl;
	}

	if (m == d) {
		f_null = TRUE;
	}
	else {
		f_null = FALSE;
	}

	if (!f_null) {

		// search for a pivot in row m:
		//
		// search for an non-zero entry
		// in row m starting in column m.
		// permute that column into column m,
		// change the col-permutation rho

		j = m;
		while ((A[m * d + j] == 0) && (j < d - 1)) {
			j++;
		}
		f_null = (A[m * d + j] == 0);

		if (!f_null && j > m) {
			if (f_vv) {
				cout << "linear_algebra::dependency "
						"choosing column " << j << endl;
			}

			// swap columns m and j
			// (only the first m + 1 elements in each column):

			for (i = 0; i <= m; i++) {
				c = A[i * d + m];
				A[i * d + m] = A[i * d + j];
				A[i * d + j] = c;
			}

			// update the permutation rho:
			c = rho[m];
			rho[m] = rho[j];
			rho[j] = c;
		}
	}
	if (f_vv) {
		cout << "linear_algebra::dependency m=" << m
				<< " after pivoting, A=" << endl;
		Int_matrix_print(A, m + 1, d);
		cout << "rho = ";
		Int_vec_print(cout, rho, d);
		cout << endl;
	}

	if (f_v) {
		cout << "linear_algebra::dependency "
				"done, f_null = " << f_null << endl;
	}
	return f_null;
}

void linear_algebra::order_ideal_generator(
		int d,
	int idx, int *mue, int &mue_deg,
	int *A, int *Frobenius,
	int verbose_level)
// Lueneburg~\cite{Lueneburg87a} p. 105.
// Frobenius is a matrix of size d x d
// A is (d + 1) x d
// mue[d + 1]
{
	int f_v = (verbose_level >= 1);
	int deg;
	int *v, *v1, *rho;
	int i, j, m, a, f_null;

	if (f_v) {
		cout << "linear_algebra::order_ideal_generator "
				"d = " << d << " idx = " << idx << endl;
	}
	deg = d;

	v = NEW_int(deg);
	v1 = NEW_int(deg);
	rho = NEW_int(deg);

	// make v the idx-th unit vector:
	Int_vec_zero(v, deg);
	v[idx] = 1;

	// make rho the identity permutation:
	for (i = 0; i < deg; i++) {
		rho[i] = i;
	}

	m = 0;
	if (f_v) {
		cout << "linear_algebra::order_ideal_generator "
				"d = " << d << " idx = " << idx
				<< " m=" << m << " before dependency" << endl;
	}
	f_null = dependency(d, v, A, m, rho, verbose_level - 1);
	if (f_v) {
		cout << "linear_algebra::order_ideal_generator "
				"d = " << d << " idx = " << idx
				<< " m=" << m << " after dependency" << endl;
	}

	while (!f_null) {

		// apply frobenius
		// (the images are written in the columns):

		if (f_v) {
			cout << "linear_algebra::order_ideal_generator v=";
			Int_vec_print(cout, v, deg);
			cout << endl;
		}
		mult_vector_from_the_right(Frobenius, v, v1, deg, deg);
		if (f_v) {
			cout << "linear_algebra::order_ideal_generator v1=";
			Int_vec_print(cout, v1, deg);
			cout << endl;
		}
		Int_vec_copy(v1, v, deg);

		m++;
		if (f_v) {
			cout << "linear_algebra::order_ideal_generator "
					"d = " << d << " idx = " << idx
					<< " m=" << m << " before dependency" << endl;
		}
		f_null = dependency(d, v, A, m, rho, verbose_level - 1);
		if (f_v) {
			cout << "linear_algebra::order_ideal_generator "
					"d = " << d << " idx = " << idx
					<< " m=" << m << " after dependency, "
							"f_null=" << f_null << endl;
		}

		if (m == deg && !f_null) {
			cout << "linear_algebra::order_ideal_generator "
					"m == deg && ! f_null" << endl;
			exit(1);
		}
	}

	mue_deg = m;
	mue[m] = 1;
	for (j = m - 1; j >= 0; j--) {
		mue[j] = A[m * deg + j];
		if (f_v) {
			cout << "linear_algebra::order_ideal_generator "
					"mue[" << j << "] = " << mue[j] << endl;
		}
		for (i = m - 1; i >= j + 1; i--) {
			a = F->mult(mue[i], A[i * deg + j]);
			mue[j] = F->add(mue[j], a);
			if (f_v) {
				cout << "linear_algebra::order_ideal_generator "
						"mue[" << j << "] = " << mue[j] << endl;
			}
		}
		a = F->negate(F->inverse(A[j * deg + j]));
		mue[j] = F->mult(mue[j], a);
			//g_asr(- mue[j] * -
			// g_inv_mod(Normal_basis[j * dim_nb + j], chi), chi);
		if (f_v) {
			cout << "linear_algebra::order_ideal_generator "
					"mue[" << j << "] = " << mue[j] << endl;
		}
	}

	if (f_v) {
		cout << "linear_algebra::order_ideal_generator "
				"after preparing mue:" << endl;
		cout << "mue_deg = " << mue_deg << endl;
		cout << "mue = ";
		Int_vec_print(cout, mue, mue_deg + 1);
		cout << endl;
	}

	FREE_int(v);
	FREE_int(v1);
	FREE_int(rho);
	if (f_v) {
		cout << "linear_algebra::order_ideal_generator done" << endl;
	}
}

void linear_algebra::span_cyclic_module(
		int *A,
		int *v, int n, int *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *w1, *w2;
	int i, j;

	if (f_v) {
		cout << "linear_algebra::span_cyclic_module" << endl;
	}
	w1 = NEW_int(n);
	w2 = NEW_int(n);
	Int_vec_copy(v, w1, n);
	for (j = 0; j < n; j++) {

		// put w1 in the j-th column of A:
		for (i = 0; i < n; i++) {
			A[i * n + j] = w1[i];
		}
		mult_vector_from_the_right(Mtx, w1, w2, n, n);
		Int_vec_copy(w2, w1, n);
	}

	FREE_int(w1);
	FREE_int(w2);
	if (f_v) {
		cout << "linear_algebra::span_cyclic_module done" << endl;
	}
}

void linear_algebra::random_invertible_matrix(
		int *M,
		int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *N;
	int i, qk, r, rk;
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "linear_algebra::random_invertible_matrix" << endl;
	}
	qk = NT.i_power_j(F->q, k);
	N = NEW_int(k * k);
	for (i = 0; i < k; i++) {
		if (f_vv) {
			cout << "i=" << i << endl;
		}
		while (TRUE) {
			r = Os.random_integer(qk);
			if (f_vv) {
				cout << "r=" << r << endl;
			}
			Gg.AG_element_unrank(F->q, M + i * k, 1, k, r);
			if (f_vv) {
				Int_matrix_print(M, i + 1, k);
			}

			Int_vec_copy(M, N, (i + 1) * k);
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
		cout << "linear_algebra::random_invertible_matrix "
				"Random invertible matrix:" << endl;
		Int_matrix_print(M, k, k);
	}
	FREE_int(N);
}

void linear_algebra::vector_add_apply(
		int *v, int *w, int c, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		v[i] = F->add(v[i], F->mult(c, w[i]));
	}
}

void linear_algebra::vector_add_apply_with_stride(
		int *v, int *w,
		int stride, int c, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		v[i] = F->add(v[i], F->mult(c, w[i * stride]));
	}
}

int linear_algebra::test_if_commute(
		int *A, int *B, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M1, *M2;
	int ret;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "linear_algebra::test_if_commute" << endl;
	}
	M1 = NEW_int(k * k);
	M2 = NEW_int(k * k);

	mult_matrix_matrix(A, B, M1, k, k, k, 0 /* verbose_level */);
	mult_matrix_matrix(B, A, M2, k, k, k, 0 /* verbose_level */);
	if (Sorting.int_vec_compare(M1, M2, k * k) == 0) {
		ret = TRUE;
	}
	else {
		ret = FALSE;
	}

	FREE_int(M1);
	FREE_int(M2);
	if (f_v) {
		cout << "linear_algebra::test_if_commute done" << endl;
	}
	return ret;
}

void linear_algebra::unrank_point_in_PG(
		int *v, int len, long int rk)
// len is the length of the vector, not the projective dimension
{

	F->Projective_space_basic->PG_element_unrank_modified(
			v, 1 /* stride */, len, rk);
}

long int linear_algebra::rank_point_in_PG(
		int *v, int len)
{
	long int rk;

	F->Projective_space_basic->PG_element_rank_modified(
			v, 1 /* stride */, len, rk);
	return rk;
}

long int linear_algebra::nb_points_in_PG(int n)
// n is projective dimension
{
	long int N;
	geometry::geometry_global Gg;

	N = Gg.nb_PG_elements(n, F->q);
	return N;
}

void linear_algebra::Borel_decomposition(
		int n, int *M,
		int *B1, int *B2, int *pivots, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, j, a, av, b, c, h, k, d, e, mc;
	int *f_is_pivot;

	if (f_v) {
		cout << "linear_algebra::Borel_decomposition" << endl;
	}
	if (f_v) {
		cout << "linear_algebra::Borel_decomposition input matrix:" << endl;
		cout << "M:" << endl;
		Int_matrix_print(M, n, n);
	}

	identity_matrix(B1, n);
	identity_matrix(B2, n);


	f_is_pivot = NEW_int(n);
	for (i = 0; i < n; i++) {
		f_is_pivot[i] = FALSE;
	}
	if (f_v) {
		cout << "linear_algebra::Borel_decomposition going down "
				"from the right" << endl;
	}
	for (j = n - 1; j >= 0; j--) {
		for (i = 0; i < n; i++) {
			if (f_is_pivot[i]) {
				continue;
			}
			if (M[i * n + j]) {
				if (f_v) {
					cout << "linear_algebra::Borel_decomposition pivot "
							"at (" << i << " " << j << ")" << endl;
				}
				f_is_pivot[i] = TRUE;
				pivots[j] = i;
				a = M[i * n + j];
				av = F->inverse(a);

				// we can only go down:
				for (h = i + 1; h < n; h++) {
					b = M[h * n + j];
					if (b) {
						c = F->mult(av, b);
						mc = F->negate(c);
						for (k = 0; k < n; k++) {
							d = F->mult(M[i * n + k], mc);
							e = F->add(M[h * n + k], d);
							M[h * n + k] = e;
						}
						//mc = negate(c);
						//cout << "finite_field::Borel_decomposition "
						// "i=" << i << " h=" << h << " mc=" << mc << endl;
						// multiply the inverse of the elementary matrix
						// to the right of B1:
						for (k = 0; k < n; k++) {
							d = F->mult(B1[k * n + h], c);
							e = F->add(B1[k * n + i], d);
							B1[k * n + i] = e;
						}
						//cout << "finite_field::Borel_decomposition B1:"
						//<< endl;
						//int_matrix_print(B1, n, n);
					}
				}
				if (f_v) {
					cout << "linear_algebra::Borel_decomposition after going "
							"down in column " << j << endl;
					cout << "M:" << endl;
					Int_matrix_print(M, n, n);
				}

				// now we go to the left from the pivot:
				for (h = 0; h < j; h++) {
					b = M[i * n + h];
					if (b) {
						c = F->mult(av, b);
						mc = F->negate(c);
						for (k = i; k < n; k++) {
							d = F->mult(M[k * n + j], mc);
							e = F->add(M[k * n + h], d);
							M[k * n + h] = e;
						}
						//mc = negate(c);
						//cout << "finite_field::Borel_decomposition "
						// "j=" << j << " h=" << h << " mc=" << mc << endl;
						// multiply the inverse of the elementary matrix
						// to the left of B2:
						for (k = 0; k < n; k++) {
							d = F->mult(B2[h * n + k], c);
							e = F->add(B2[j * n + k], d);
							B2[j * n + k] = e;
						}
					}
				}
				if (f_v) {
					cout << "linear_algebra::Borel_decomposition after going "
							"across to the left:" << endl;
					cout << "M:" << endl;
					Int_matrix_print(M, n, n);
				}
				break;
			}

		}
	}
	FREE_int(f_is_pivot);
	if (f_v) {
		cout << "linear_algebra::Borel_decomposition done" << endl;
	}
}

void linear_algebra::map_to_standard_frame(
		int d, int *A,
		int *Transform, int verbose_level)
// d = vector space dimension
// maps d + 1 points to the frame e_1, e_2, ..., e_d, e_1+e_2+..+e_d
// A is (d + 1) x d
// Transform is d x d
{
	int f_v = (verbose_level >= 1);
	int *B;
	int *A2;
	int n, xd, x, i, j;

	if (f_v) {
		cout << "linear_algebra::map_to_standard_frame" << endl;
	}

	if (f_v) {
		cout << "A=" << endl;
		Int_matrix_print(A, d + 1, d);
	}

	n = d + 1;
	B = NEW_int(n * n);
	A2 = NEW_int(d * d);
	for (i = 0; i < n; i++) {
		for (j = 0; j < d; j++) {
			B[j * n + i] = A[i * d + j];
		}
	}
	if (f_v) {
		cout << "B before=" << endl;
		Int_matrix_print(B, d, n);
	}
	RREF_and_kernel(n, d, B, 0 /* verbose_level */);
	if (f_v) {
		cout << "B after=" << endl;
		Int_matrix_print(B, n, n);
	}
	xd = B[d * n + d - 1];
	x = F->negate(F->inverse(xd));
	for (i = 0; i < d; i++) {
		B[d * n + i] = F->mult(x, B[d * n + i]);
	}
	if (f_v) {
		cout << "last row of B after scaling : " << endl;
		Int_matrix_print(B + d * n, 1, n);
	}
	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			A2[i * d + j] = F->mult(B[d * n + i], A[i * d + j]);
		}
	}
	if (f_v) {
		cout << "A2=" << endl;
		Int_matrix_print(A2, d, d);
	}
	matrix_inverse(A2, Transform, d, 0 /* verbose_level */);

	FREE_int(B);
	FREE_int(A2);
	if (f_v) {
		cout << "linear_algebra::map_to_standard_frame done" << endl;
	}
}

void linear_algebra::map_frame_to_frame_with_permutation(
		int d,
		int *A, int *perm, int *B, int *Transform,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *T1;
	int *T2;
	int *T3;
	int *A1;
	int i, j;

	if (f_v) {
		cout << "linear_algebra::map_frame_to_frame_with_permutation" << endl;
	}
	T1 = NEW_int(d * d);
	T2 = NEW_int(d * d);
	T3 = NEW_int(d * d);
	A1 = NEW_int((d + 1) * d);

	if (f_v) {
		cout << "permutation: ";
		Int_vec_print(cout, perm, d + 1);
		cout << endl;
	}
	if (f_v) {
		cout << "A=" << endl;
		Int_matrix_print(A, d + 1, d);
	}
	if (f_v) {
		cout << "B=" << endl;
		Int_matrix_print(B, d + 1, d);
	}

	for (i = 0; i < d + 1; i++) {
		j = perm[i];
		Int_vec_copy(A + j * d, A1 + i * d, d);
	}

	if (f_v) {
		cout << "A1=" << endl;
		Int_matrix_print(A1, d + 1, d);
	}


	if (f_v) {
		cout << "mapping A1 to standard frame:" << endl;
	}
	map_to_standard_frame(d, A1, T1, verbose_level);
	if (f_v) {
		cout << "T1=" << endl;
		Int_matrix_print(T1, d, d);
	}
	if (f_v) {
		cout << "mapping B to standard frame:" << endl;
	}
	map_to_standard_frame(d, B, T2, 0 /* verbose_level */);
	if (f_v) {
		cout << "T2=" << endl;
		Int_matrix_print(T2, d, d);
	}
	matrix_inverse(T2, T3, d, 0 /* verbose_level */);
	if (f_v) {
		cout << "T3=" << endl;
		Int_matrix_print(T3, d, d);
	}
	mult_matrix_matrix(T1, T3, Transform, d, d, d, 0 /* verbose_level */);
	if (f_v) {
		cout << "Transform=" << endl;
		Int_matrix_print(Transform, d, d);
	}

	FREE_int(T1);
	FREE_int(T2);
	FREE_int(T3);
	FREE_int(A1);
	if (f_v) {
		cout << "linear_algebra::map_frame_to_frame_with_permutation done"
				<< endl;
	}
}


void linear_algebra::map_points_to_points_projectively(
		int d, int k,
		int *A, int *B, int *Transform, int &nb_maps,
		int verbose_level)
// A and B are (d + k + 1) x d
// Transform is d x d
// returns TRUE if a map exists
{
	int f_v = (verbose_level >= 1);
	int *lehmercode;
	int *perm;
	int *A1;
	int *B1;
	int *B_set;
	int *image_set;
	int *v;
	long int h, i, j, a;
	int *subset; // [d + k + 1]
	int nCk;
	int cnt, overall_cnt;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "linear_algebra::map_points_to_points_projectively" << endl;
	}
	lehmercode = NEW_int(d + 1);
	perm = NEW_int(d + 1);
	A1 = NEW_int((d + k + 1) * d);
	B1 = NEW_int((d + k + 1) * d);
	B_set = NEW_int(d + k + 1);
	image_set = NEW_int(d + k + 1);
	subset = NEW_int(d + k + 1);
	v = NEW_int(d);

	Int_vec_copy(B, B1, (d + k + 1) * d);
	for (i = 0; i < d + k + 1; i++) {
		//PG_element_normalize(*this, B1 + i * d, 1, d);
		F->Projective_space_basic->PG_element_rank_modified(
				B1 + i * d, 1, d, a);
		B_set[i] = a;
	}
	Sorting.int_vec_heapsort(B_set, d + k + 1);
	if (f_v) {
		cout << "B_set = ";
		Int_vec_print(cout, B_set, d + k + 1);
		cout << endl;
	}


	overall_cnt = 0;
	nCk = Combi.int_n_choose_k(d + k + 1, d + 1);
	for (h = 0; h < nCk; h++) {
		Combi.unrank_k_subset(h, subset, d + k + 1, d + 1);
		Combi.set_complement(subset, d + 1, subset + d + 1, k, d + k + 1);

		if (f_v) {
			cout << "subset " << h << " / " << nCk << " is ";
			Int_vec_print(cout, subset, d + 1);
			cout << ", the complement is ";
			Int_vec_print(cout, subset + d + 1, k);
			cout << endl;
		}


		for (i = 0; i < d + k + 1; i++) {
			j = subset[i];
			Int_vec_copy(A + j * d, A1 + i * d, d);
		}
		if (f_v) {
			cout << "A1=" << endl;
			Int_matrix_print(A1, d + k + 1, d);
		}

		cnt = 0;
		Combi.first_lehmercode(d + 1, lehmercode);
		while (TRUE) {
			if (f_v) {
				cout << "lehmercode: ";
				Int_vec_print(cout, lehmercode, d + 1);
				cout << endl;
			}
			Combi.lehmercode_to_permutation(d + 1, lehmercode, perm);
			if (f_v) {
				cout << "permutation: ";
				Int_vec_print(cout, perm, d + 1);
				cout << endl;
			}
			map_frame_to_frame_with_permutation(d, A1, perm,
					B1, Transform, verbose_level);

			for (i = 0; i < d + k + 1; i ++) {
				mult_vector_from_the_left(
						A1 + i * d, Transform, v, d, d);
				F->Projective_space_basic->PG_element_rank_modified(
						v, 1, d, a);
				image_set[i] = a;
			}
			if (f_v) {
				cout << "image_set before sorting: ";
				Int_vec_print(cout, image_set, d + k + 1);
				cout << endl;
			}
			Sorting.int_vec_heapsort(image_set, d + k + 1);
			if (f_v) {
				cout << "image_set after sorting: ";
				Int_vec_print(cout, image_set, d + k + 1);
				cout << endl;
			}

			if (Sorting.int_vec_compare(image_set, B_set, d + k + 1) == 0) {
				cnt++;
			}

			if (!Combi.next_lehmercode(d + 1, lehmercode)) {
				break;
			}
		}

		cout << "subset " << h << " / " << nCk << " we found "
				<< cnt << " mappings" << endl;
		overall_cnt += cnt;
	}

	FREE_int(perm);
	FREE_int(lehmercode);
	FREE_int(A1);
	FREE_int(B1);
	FREE_int(B_set);
	FREE_int(image_set);
	FREE_int(subset);
	FREE_int(v);

	nb_maps = overall_cnt;

	if (f_v) {
		cout << "linear_algebra::map_points_to_points_projectively done"
				<< endl;
	}
}

int linear_algebra::BallChowdhury_matrix_entry(
		int *Coord,
		int *C, int *U, int k, int sz_U,
	int *T, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, d, d1, u, a;

	if (f_v) {
		cout << "linear_algebra::BallChowdhury_matrix_entry" << endl;
	}
	d = 1;
	for (u = 0; u < sz_U; u++) {
		a = U[u];
		Int_vec_copy(Coord + a * k, T + (k - 1) * k, k);
		for (i = 0; i < k - 1; i++) {
			a = C[i];
			Int_vec_copy(Coord + a * k, T + i * k, k);
		}
		if (f_vv) {
			cout << "u=" << u << " / " << sz_U << " the matrix is:" << endl;
			Int_matrix_print(T, k, k);
		}
		d1 = matrix_determinant(T, k, 0 /* verbose_level */);
		if (f_vv) {
			cout << "determinant = " << d1 << endl;
		}
		d = F->mult(d, d1);
	}
	if (f_v) {
		cout << "linear_algebra::BallChowdhury_matrix_entry d=" << d << endl;
	}
	return d;
}

int linear_algebra::is_unit_vector(
		int *v, int len, int k)
{
	int i;

	for (i = 0; i < len; i++) {
		if (i == k) {
			if (v[i] != 1) {
				return FALSE;
			}
		}
		else {
			if (v[i] != 0) {
				return FALSE;
			}
		}
	}
	return TRUE;
}


void linear_algebra::make_Fourier_matrices(
		int omega, int k, int *N, int **A, int **Av,
		int *Omega, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, i, j, om;

	if (f_v) {
		cout << "linear_algebra::make_Fourier_matrices" << endl;
	}

	Omega[k] = omega;
	for (h = k; h > 0; h--) {
		Omega[h - 1] = F->mult(Omega[h], Omega[h]);
	}

	for (h = k; h >= 0; h--) {
		A[h] = NEW_int(N[h] * N[h]);
		om = Omega[h];
		for (i = 0; i < N[h]; i++) {
			for (j = 0; j < N[h]; j++) {
				A[h][i * N[h] + j] = F->power(om, (i * j) % N[k]);
			}
		}
	}

	for (h = k; h >= 0; h--) {
		Av[h] = NEW_int(N[h] * N[h]);
		om = F->inverse(Omega[h]);
		for (i = 0; i < N[h]; i++) {
			for (j = 0; j < N[h]; j++) {
				Av[h][i * N[h] + j] = F->power(om, (i * j) % N[k]);
			}
		}
	}

	if (f_v) {
		for (h = k; h >= 0; h--) {
			cout << "A_" << N[h] << ":" << endl;
			Int_matrix_print(A[h], N[h], N[h]);
		}

		for (h = k; h >= 0; h--) {
			cout << "Av_" << N[h] << ":" << endl;
			Int_matrix_print(Av[h], N[h], N[h]);
		}
	}
	if (f_v) {
		cout << "linear_algebra::make_Fourier_matrices done" << endl;
	}
}



}}}

