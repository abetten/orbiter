/*
 * finite_field_linear_algebra2.cpp
 *
 *  Created on: Nov 3, 2019
 *      Author: anton
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {



void finite_field::reduce_mod_subspace_and_get_coefficient_vector(
	int k, int len, int *basis, int *base_cols,
	int *v, int *coefficients, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, idx;

	if (f_v) {
		cout << "finite_field::reduce_mod_subspace_and_get_coefficient_vector" << endl;
	}
	if (f_vv) {
		cout << "finite_field::reduce_mod_subspace_and_get_coefficient_vector: v=";
		Orbiter->Int_vec.print(cout, v, len);
		cout << endl;
	}
	if (f_vv) {
		cout << "finite_field::reduce_mod_subspace_and_get_"
				"coefficient_vector subspace basis:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, basis, k, len, len, log10_of_q);
	}
	for (i = 0; i < k; i++) {
		idx = base_cols[i];
		if (basis[i * len + idx] != 1) {
			cout << "finite_field::reduce_mod_subspace_and_get_"
					"coefficient_vector pivot entry is not one" << endl;
			cout << "i=" << i << endl;
			cout << "idx=" << idx << endl;
			Orbiter->Int_vec.print_integer_matrix_width(cout, basis,
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
	if (f_vv) {
		cout << "finite_field::reduce_mod_subspace_and_get_coefficient_vector "
				"after: v=";
		Orbiter->Int_vec.print(cout, v, len);
		cout << endl;
		cout << "coefficients=";
		Orbiter->Int_vec.print(cout, coefficients, k);
		cout << endl;
	}
	if (f_v) {
		cout << "finite_field::reduce_mod_subspace_and_get_coefficient_vector done" << endl;
	}
}

void finite_field::reduce_mod_subspace(int k,
	int len, int *basis, int *base_cols,
	int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, idx;

	if (f_v) {
		cout << "finite_field::reduce_mod_subspace" << endl;
	}
	if (f_vv) {
		cout << "finite_field::reduce_mod_subspace before: v=";
		Orbiter->Int_vec.print(cout, v, len);
		cout << endl;
	}
	if (f_vv) {
		cout << "finite_field::reduce_mod_subspace subspace basis:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, basis, k,
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
	if (f_vv) {
		cout << "finite_field::reduce_mod_subspace after: v=";
		Orbiter->Int_vec.print(cout, v, len);
		cout << endl;
	}
	if (f_v) {
		cout << "finite_field::reduce_mod_subspace done" << endl;
	}
}

int finite_field::is_contained_in_subspace(int k,
	int len, int *basis, int *base_cols,
	int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "finite_field::is_contained_in_subspace" << endl;
	}
	if (f_vv) {
		cout << "finite_field::is_contained_in_subspace testing v=";
		Orbiter->Int_vec.print(cout, v, len);
		cout << endl;
	}
	reduce_mod_subspace(k, len, basis,
			base_cols, v, verbose_level - 1);
	for (i = 0; i < len; i++) {
		if (v[i]) {
			if (f_vv) {
				cout << "finite_field::is_contained_in_subspace "
						"is NOT in the subspace" << endl;
			}
			return FALSE;
		}
	}
	if (f_vv) {
		cout << "finite_field::is_contained_in_subspace "
				"is contained in the subspace" << endl;
	}
	if (f_v) {
		cout << "finite_field::is_contained_in_subspace done" << endl;
	}
	return TRUE;
}

int finite_field::is_subspace(int d, int dim_U,
		int *Basis_U, int dim_V, int *Basis_V,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Basis;
	int h, rk, ret;

	if (f_v) {
		cout << "finite_field::is_subspace" << endl;
	}
	Basis = NEW_int((dim_V + 1) * d);
	for (h = 0; h < dim_U; h++) {

		Orbiter->Int_vec.copy(Basis_V, Basis, dim_V * d);
		Orbiter->Int_vec.copy(Basis_U + h * d, Basis + dim_V * d, d);
		rk = Gauss_easy(Basis, dim_V + 1, d);
		if (rk > dim_V) {
			ret = FALSE;
			goto done;
		}
	}
	ret = TRUE;
done:
	FREE_int(Basis);
	if (f_v) {
		cout << "finite_field::is_subspace done" << endl;
	}
	return ret;
}

void finite_field::Kronecker_product(int *A, int *B, int n, int *AB)
{
	int i, j, I, J, u, v, a, b, c, n2;

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
	int *A, int *B,
	int na, int nb, int *AB, int &N,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, I, J, u, v, a, b, c;

	if (f_v) {
		cout << "finite_field::Kronecker_product_square_but_arbitrary"
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

int finite_field::dependency(int d,
		int *v, int *A, int m, int *rho,
		int verbose_level)
// Lueneburg~\cite{Lueneburg87a} p. 104.
// A is a matrix of size d + 1 times d
// v[d]
// rho is a column permutation of degree d
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, k, deg, f_null, c;

	if (f_v) {
		cout << "finite_field::dependency" << endl;
		cout << "m = " << m << endl;
	}
	deg = d;
	if (f_vv) {
		cout << "finite_field::dependency A=" << endl;
		Orbiter->Int_vec.matrix_print(A, m, deg);
		cout << "v = ";
		Orbiter->Int_vec.print(cout, v, deg);
		cout << endl;
	}
	// fill the m-th row of matrix A with v^rho:
	for (j = 0; j < deg; j++) {
		A[m * deg + j] = v[rho[j]];
	}
	if (f_vv) {
		cout << "finite_field::dependency "
				"after putting in row " << m << " A=" << endl;
		Orbiter->Int_vec.matrix_print(A, m + 1, deg);
		cout << "rho = ";
		Orbiter->Int_vec.print(cout, rho, deg);
		cout << endl;
	}
	for (k = 0; k < m; k++) {

		if (f_vv) {
			cout << "finite_field::dependency "
					"k=" << k << " A=" << endl;
			Orbiter->Int_vec.matrix_print(A, m + 1, deg);
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
			Orbiter->Int_vec.matrix_print(A, m + 1, deg);
		}

	} // next k
	if (f_vv) {
		cout << "finite_field::dependency "
				"m=" << m << " after reapply, A=" << endl;
		Orbiter->Int_vec.matrix_print(A, m + 1, deg);
		cout << "rho = ";
		Orbiter->Int_vec.print(cout, rho, deg);
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
		Orbiter->Int_vec.matrix_print(A, m + 1, deg);
		cout << "rho = ";
		Orbiter->Int_vec.print(cout, rho, deg);
		cout << endl;
	}

	if (f_v) {
		cout << "finite_field::dependency "
				"done, f_null = " << f_null << endl;
	}
	return f_null;
}

void finite_field::order_ideal_generator(int d,
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
		cout << "finite_field::order_ideal_generator "
				"d = " << d << " idx = " << idx << endl;
	}
	deg = d;

	v = NEW_int(deg);
	v1 = NEW_int(deg);
	rho = NEW_int(deg);

	// make v the idx-th unit vector:
	Orbiter->Int_vec.zero(v, deg);
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
		Orbiter->Int_vec.copy(v1, v, deg);

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
		Orbiter->Int_vec.print(cout, mue, mue_deg + 1);
		cout << endl;
	}

	FREE_int(v);
	FREE_int(v1);
	FREE_int(rho);
	if (f_v) {
		cout << "finite_field::order_ideal_generator done" << endl;
	}
}

void finite_field::span_cyclic_module(int *A,
		int *v, int n, int *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *w1, *w2;
	int i, j;

	if (f_v) {
		cout << "finite_field::span_cyclic_module" << endl;
	}
	w1 = NEW_int(n);
	w2 = NEW_int(n);
	Orbiter->Int_vec.copy(v, w1, n);
	for (j = 0; j < n; j++) {

		// put w1 in the j-th column of A:
		for (i = 0; i < n; i++) {
			A[i * n + j] = w1[i];
		}
		mult_vector_from_the_right(Mtx, w1, w2, n, n);
		Orbiter->Int_vec.copy(w2, w1, n);
	}

	FREE_int(w1);
	FREE_int(w2);
	if (f_v) {
		cout << "finite_field::span_cyclic_module done" << endl;
	}
}

void finite_field::random_invertible_matrix(int *M,
		int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *N;
	int i, qk, r, rk;
	number_theory_domain NT;
	geometry_global Gg;
	os_interface Os;

	if (f_v) {
		cout << "finite_field::random_invertible_matrix" << endl;
	}
	qk = NT.i_power_j(q, k);
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
			Gg.AG_element_unrank(q, M + i * k, 1, k, r);
			if (f_vv) {
				Orbiter->Int_vec.matrix_print(M, i + 1, k);
			}

			Orbiter->Int_vec.copy(M, N, (i + 1) * k);
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
		Orbiter->Int_vec.matrix_print(M, k, k);
	}
	FREE_int(N);
}

void finite_field::adjust_basis(int *V, int *U,
		int n, int k, int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, ii, b;
	int *base_cols;
	int *M;
	sorting Sorting;

	if (f_v) {
		cout << "finite_field::adjust_basis" << endl;
	}
	base_cols = NEW_int(n);
	M = NEW_int(k * n);

	Orbiter->Int_vec.copy(U, M, d * n);

	if (Gauss_simple(M, d, n, base_cols,
			0 /* verbose_level */) != d) {
		cout << "finite_field::adjust_basis rank "
				"of matrix is not d" << endl;
		exit(1);
	}
	ii = 0;
	for (i = 0; i < k; i++) {
		Orbiter->Int_vec.copy(V + i * n, M + (d + ii) * n, n);
		for (j = 0; j < d; j++) {
			b = base_cols[j];
			Gauss_step(M + b * n, M + (d + ii) * n,
					n, b, 0 /* verbose_level */);
		}
		if (Sorting.int_vec_is_zero(M + (d + ii) * n, n)) {
		}
		else {
			ii++;
		}
	}
	if (d + ii != k) {
		cout << "finite_field::adjust_basis d + ii != k" << endl;
		exit(1);
	}
	Orbiter->Int_vec.copy(M, V, k * n);


	FREE_int(M);
	FREE_int(base_cols);
	if (f_v) {
		cout << "finite_field::adjust_basis done" << endl;
	}
}

void finite_field::choose_vector_in_here_but_not_in_here_column_spaces(
		int_matrix *V, int_matrix *W, int *v,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n, k, d;
	int *Gen;
	int *base_cols;
	int i, j, ii, b;
	sorting Sorting;

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
	Gen = NEW_int(k * n);
	base_cols = NEW_int(n);

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
		if (Sorting.int_vec_is_zero(Gen + (d + ii) * n, n)) {
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
	Orbiter->Int_vec.copy(Gen + d * n, v, n);


	FREE_int(Gen);
	FREE_int(base_cols);
	if (f_v) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_"
				"column_spaces done" << endl;
	}
}

void finite_field::choose_vector_in_here_but_not_in_here_or_here_column_spaces(
	int_matrix *V,
	int_matrix *W1, int_matrix *W2, int *v,
	int verbose_level)
{

	int coset = 0;

	choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
			coset, V, W1, W2, v, verbose_level);

}

int finite_field::choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
	int &coset,
	int_matrix *V, int_matrix *W1, int_matrix *W2, int *v,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int n, k, d1, d2, rk;
	int *Gen;
	int *base_cols;
	int *w;
	int *z;
	int i, j, b;
	int ret = TRUE;
	number_theory_domain NT;
	geometry_global Gg;
	sorting Sorting;

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
	Gen = NEW_int((d1 + d2 + k) * n);
	base_cols = NEW_int(n);
	w = NEW_int(k);
	z = NEW_int(n);

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


	int a;

	while (TRUE) {
		if (coset >= NT.i_power_j(q, k)) {
			if (f_vv) {
				cout << "coset = " << coset << " = " << NT.i_power_j(q, k)
						<< " break" << endl;
			}
			ret = FALSE;
			break;
		}
		Gg.AG_element_unrank(q, w, 1, k, coset);

		if (f_vv) {
			cout << "coset=" << coset << " w=";
			Orbiter->Int_vec.print(cout, w, k);
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
		Orbiter->Int_vec.copy(Gen + rk * n, z, n);
		if (f_vv) {
			cout << "before reduce=";
			Orbiter->Int_vec.print(cout, Gen + rk * n, n);
			cout << endl;
		}

		// reduce modulo the subspace:
		for (j = 0; j < rk; j++) {
			b = base_cols[j];
			Gauss_step(Gen + j * n, Gen + rk * n, n, b, 0 /* verbose_level */);
		}

		if (f_vv) {
			cout << "after reduce=";
			Orbiter->Int_vec.print(cout, Gen + rk * n, n);
			cout << endl;
		}


		// see if we got something nonzero:
		if (!Sorting.int_vec_is_zero(Gen + rk * n, n)) {
			break;
		}
		// keep moving on to the next vector

	} // while

	Orbiter->Int_vec.copy(z, v, n);


	FREE_int(Gen);
	FREE_int(base_cols);
	FREE_int(w);
	FREE_int(z);
	if (f_v) {
		cout << "finite_field::choose_vector_in_here_but_not_in_here_"
				"or_here_column_spaces_coset done ret = " << ret << endl;
	}
	return ret;
}

void finite_field::vector_add_apply(int *v, int *w, int c, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		v[i] = add(v[i], mult(c, w[i]));
	}
}

void finite_field::vector_add_apply_with_stride(int *v, int *w,
		int stride, int c, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		v[i] = add(v[i], mult(c, w[i * stride]));
	}
}

int finite_field::test_if_commute(int *A, int *B, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M1, *M2;
	int ret;

	if (f_v) {
		cout << "finite_field::test_if_commute" << endl;
	}
	M1 = NEW_int(k * k);
	M2 = NEW_int(k * k);

	mult_matrix_matrix(A, B, M1, k, k, k, 0 /* verbose_level */);
	mult_matrix_matrix(B, A, M2, k, k, k, 0 /* verbose_level */);
	if (int_vec_compare(M1, M2, k * k) == 0) {
		ret = TRUE;
	}
	else {
		ret = FALSE;
	}

	FREE_int(M1);
	FREE_int(M2);
	if (f_v) {
		cout << "finite_field::test_if_commute done" << endl;
	}
	return ret;
}

void finite_field::unrank_point_in_PG(int *v, int len, int rk)
// len is the length of the vector, not the projective dimension
{

	PG_element_unrank_modified(v, 1 /* stride */, len, rk);
}

int finite_field::rank_point_in_PG(int *v, int len)
{
	int rk;

	PG_element_rank_modified(v, 1 /* stride */, len, rk);
	return rk;
}

int finite_field::nb_points_in_PG(int n)
// n is projective dimension
{
	int N;
	geometry_global Gg;

	N = Gg.nb_PG_elements(n, q);
	return N;
}

void finite_field::Borel_decomposition(int n, int *M,
		int *B1, int *B2, int *pivots, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, j, a, av, b, c, h, k, d, e, mc;
	int *f_is_pivot;

	if (f_v) {
		cout << "finite_field::Borel_decomposition" << endl;
	}
	if (f_v) {
		cout << "finite_field::Borel_decomposition input matrix:" << endl;
		cout << "M:" << endl;
		Orbiter->Int_vec.matrix_print(M, n, n);
	}

	identity_matrix(B1, n);
	identity_matrix(B2, n);


	f_is_pivot = NEW_int(n);
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
						//int_matrix_print(B1, n, n);
					}
				}
				if (f_v) {
					cout << "finite_field::Borel_decomposition after going "
							"down in column " << j << endl;
					cout << "M:" << endl;
					Orbiter->Int_vec.matrix_print(M, n, n);
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
					Orbiter->Int_vec.matrix_print(M, n, n);
				}
				break;
			}

		}
	}
	FREE_int(f_is_pivot);
	if (f_v) {
		cout << "finite_field::Borel_decomposition done" << endl;
	}
}

void finite_field::map_to_standard_frame(int d, int *A,
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
		cout << "finite_field::map_to_standard_frame" << endl;
	}

	if (f_v) {
		cout << "A=" << endl;
		Orbiter->Int_vec.matrix_print(A, d + 1, d);
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
		Orbiter->Int_vec.matrix_print(B, d, n);
	}
	RREF_and_kernel(n, d, B, 0 /* verbose_level */);
	if (f_v) {
		cout << "B after=" << endl;
		Orbiter->Int_vec.matrix_print(B, n, n);
	}
	xd = B[d * n + d - 1];
	x = negate(inverse(xd));
	for (i = 0; i < d; i++) {
		B[d * n + i] = mult(x, B[d * n + i]);
	}
	if (f_v) {
		cout << "last row of B after scaling : " << endl;
		Orbiter->Int_vec.matrix_print(B + d * n, 1, n);
	}
	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			A2[i * d + j] = mult(B[d * n + i], A[i * d + j]);
		}
	}
	if (f_v) {
		cout << "A2=" << endl;
		Orbiter->Int_vec.matrix_print(A2, d, d);
	}
	matrix_inverse(A2, Transform, d, 0 /* verbose_level */);

	FREE_int(B);
	FREE_int(A2);
	if (f_v) {
		cout << "finite_field::map_to_standard_frame done" << endl;
	}
}

void finite_field::map_frame_to_frame_with_permutation(int d,
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
		cout << "finite_field::map_frame_to_frame_with_permutation" << endl;
	}
	T1 = NEW_int(d * d);
	T2 = NEW_int(d * d);
	T3 = NEW_int(d * d);
	A1 = NEW_int((d + 1) * d);

	if (f_v) {
		cout << "permutation: ";
		Orbiter->Int_vec.print(cout, perm, d + 1);
		cout << endl;
	}
	if (f_v) {
		cout << "A=" << endl;
		Orbiter->Int_vec.matrix_print(A, d + 1, d);
	}
	if (f_v) {
		cout << "B=" << endl;
		Orbiter->Int_vec.matrix_print(B, d + 1, d);
	}

	for (i = 0; i < d + 1; i++) {
		j = perm[i];
		Orbiter->Int_vec.copy(A + j * d, A1 + i * d, d);
	}

	if (f_v) {
		cout << "A1=" << endl;
		Orbiter->Int_vec.matrix_print(A1, d + 1, d);
	}


	if (f_v) {
		cout << "mapping A1 to standard frame:" << endl;
	}
	map_to_standard_frame(d, A1, T1, verbose_level);
	if (f_v) {
		cout << "T1=" << endl;
		Orbiter->Int_vec.matrix_print(T1, d, d);
	}
	if (f_v) {
		cout << "mapping B to standard frame:" << endl;
	}
	map_to_standard_frame(d, B, T2, 0 /* verbose_level */);
	if (f_v) {
		cout << "T2=" << endl;
		Orbiter->Int_vec.matrix_print(T2, d, d);
	}
	matrix_inverse(T2, T3, d, 0 /* verbose_level */);
	if (f_v) {
		cout << "T3=" << endl;
		Orbiter->Int_vec.matrix_print(T3, d, d);
	}
	mult_matrix_matrix(T1, T3, Transform, d, d, d, 0 /* verbose_level */);
	if (f_v) {
		cout << "Transform=" << endl;
		Orbiter->Int_vec.matrix_print(Transform, d, d);
	}

	FREE_int(T1);
	FREE_int(T2);
	FREE_int(T3);
	FREE_int(A1);
	if (f_v) {
		cout << "finite_field::map_frame_to_frame_with_permutation done"
				<< endl;
	}
}


void finite_field::map_points_to_points_projectively(int d, int k,
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
	int h, i, j, a;
	int *subset; // [d + k + 1]
	int nCk;
	int cnt, overall_cnt;
	combinatorics_domain Combi;
	sorting Sorting;

	if (f_v) {
		cout << "finite_field::map_points_to_points_projectively" << endl;
	}
	lehmercode = NEW_int(d + 1);
	perm = NEW_int(d + 1);
	A1 = NEW_int((d + k + 1) * d);
	B1 = NEW_int((d + k + 1) * d);
	B_set = NEW_int(d + k + 1);
	image_set = NEW_int(d + k + 1);
	subset = NEW_int(d + k + 1);
	v = NEW_int(d);

	Orbiter->Int_vec.copy(B, B1, (d + k + 1) * d);
	for (i = 0; i < d + k + 1; i++) {
		//PG_element_normalize(*this, B1 + i * d, 1, d);
		PG_element_rank_modified(B1 + i * d, 1, d, a);
		B_set[i] = a;
	}
	Sorting.int_vec_heapsort(B_set, d + k + 1);
	if (f_v) {
		cout << "B_set = ";
		Orbiter->Int_vec.print(cout, B_set, d + k + 1);
		cout << endl;
	}


	overall_cnt = 0;
	nCk = Combi.int_n_choose_k(d + k + 1, d + 1);
	for (h = 0; h < nCk; h++) {
		Combi.unrank_k_subset(h, subset, d + k + 1, d + 1);
		Combi.set_complement(subset, d + 1, subset + d + 1, k, d + k + 1);

		if (f_v) {
			cout << "subset " << h << " / " << nCk << " is ";
			Orbiter->Int_vec.print(cout, subset, d + 1);
			cout << ", the complement is ";
			Orbiter->Int_vec.print(cout, subset + d + 1, k);
			cout << endl;
		}


		for (i = 0; i < d + k + 1; i++) {
			j = subset[i];
			Orbiter->Int_vec.copy(A + j * d, A1 + i * d, d);
		}
		if (f_v) {
			cout << "A1=" << endl;
			Orbiter->Int_vec.matrix_print(A1, d + k + 1, d);
		}

		cnt = 0;
		Combi.first_lehmercode(d + 1, lehmercode);
		while (TRUE) {
			if (f_v) {
				cout << "lehmercode: ";
				Orbiter->Int_vec.print(cout, lehmercode, d + 1);
				cout << endl;
			}
			Combi.lehmercode_to_permutation(d + 1, lehmercode, perm);
			if (f_v) {
				cout << "permutation: ";
				Orbiter->Int_vec.print(cout, perm, d + 1);
				cout << endl;
			}
			map_frame_to_frame_with_permutation(d, A1, perm,
					B1, Transform, verbose_level);

			for (i = 0; i < d + k + 1; i ++) {
				mult_vector_from_the_left(A1 + i * d, Transform, v, d, d);
				PG_element_rank_modified(v, 1, d, a);
				image_set[i] = a;
			}
			if (f_v) {
				cout << "image_set before sorting: ";
				Orbiter->Int_vec.print(cout, image_set, d + k + 1);
				cout << endl;
			}
			Sorting.int_vec_heapsort(image_set, d + k + 1);
			if (f_v) {
				cout << "image_set after sorting: ";
				Orbiter->Int_vec.print(cout, image_set, d + k + 1);
				cout << endl;
			}

			if (int_vec_compare(image_set, B_set, d + k + 1) == 0) {
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
		cout << "finite_field::map_points_to_points_projectively done"
				<< endl;
	}
}

int finite_field::BallChowdhury_matrix_entry(int *Coord,
		int *C, int *U, int k, int sz_U,
	int *T, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, d, d1, u, a;

	if (f_v) {
		cout << "finite_field::BallChowdhury_matrix_entry" << endl;
	}
	d = 1;
	for (u = 0; u < sz_U; u++) {
		a = U[u];
		Orbiter->Int_vec.copy(Coord + a * k, T + (k - 1) * k, k);
		for (i = 0; i < k - 1; i++) {
			a = C[i];
			Orbiter->Int_vec.copy(Coord + a * k, T + i * k, k);
		}
		if (f_vv) {
			cout << "u=" << u << " / " << sz_U << " the matrix is:" << endl;
			Orbiter->Int_vec.matrix_print(T, k, k);
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
	int f_with_normalizer,
	int f_semilinear,
	int *&gens, int &nb_gens, int &data_size,
	int &group_order, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m_one;

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
	gens = NEW_int(nb_gens * data_size);
	Orbiter->Int_vec.zero(gens, nb_gens * data_size);
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

void finite_field::cubic_surface_family_G13_generators(
	int a,
	int *&gens, int &nb_gens, int &data_size,
	int &group_order, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int data[] = {
			// A1:
			1,0,0,0,
			0,1,0,0,
			3,2,1,0,
			3,0,0,1,

			// A2:
			1,0,0,0,
			0,1,0,0,
			1,1,1,0,
			1,0,0,1,

			// A3:
			0,1,0,0,
			1,0,0,0,
			0,0,1,0,
			1,1,1,1,

			// A4:
			0,1,0,0,
			1,0,0,0,
			1,1,1,0,
			7,6,1,1,

			// A5:
			2,3,0,0,
			3,2,0,0,
			0,0,1,0,
			3,3,5,1,

			// A6:
			2,2,1,0,
			3,3,1,0,
			1,0,1,0,
			1,4,2,1,

	};

	data_size = 16 + 1;
	nb_gens = 6;
	group_order = 192;

	gens = NEW_int(nb_gens * data_size);
	Orbiter->Int_vec.zero(gens, nb_gens * data_size);

	int h, i, j, c, m, l;
	int *v;
	geometry_global Gg;
	number_theory_domain NT;

	m = Orbiter->Int_vec.maximum(data, nb_gens * data_size);
	l = NT.int_log2(m) + 1;

	v = NEW_int(l);


	for (h = 0; h < nb_gens; h++) {
		for (i = 0; i < 16; i++) {
			Orbiter->Int_vec.zero(v, l);
			Gg.AG_element_unrank(p, v, 1, l, data[h * 16 + i]);
			c = 0;
			for (j = 0; j < l; j++) {
				c = mult(c, a);
				if (v[l - 1 - j]) {
					c = add(c, v[l - 1 - j]);
				}
			}
			gens[h * data_size + i] = c;
		}
		gens[h * data_size + 16] = 0;
	}
	FREE_int(v);

	if (f_v) {
		cout << "finite_field::cubic_surface_family_G13_generators" << endl;
		for (h = 0; h < nb_gens; h++) {
			cout << "generator " << h << ":" << endl;
			Orbiter->Int_vec.matrix_print(gens + h * data_size, 4, 4);
		}
	}
	if (f_v) {
		cout << "finite_field::cubic_surface_family_G13_generators done" << endl;
	}
}

void finite_field::cubic_surface_family_F13_generators(
	int a,
	int *&gens, int &nb_gens, int &data_size,
	int &group_order, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	// 2 = a
	// 3 = a+1
	// 4 = a^2
	// 5 = a^2+1
	// 6 = a^2 + a
	// 7 = a^2 + a + 1
	// 8 = a^3
	// 9 = a^3 + 1
	// 10 = a^3 + a
	// 11 = a^3 + a + 1
	// 12 = a^3 + a^2
	// 13 = a^3 + a^2 + 1
	// 14 = a^3 + a^2 + a
	// 15 = a^3 + a^2 + a + 1 = (a+1)^3
	// 16 = a^4
	// 17 = a^4 + 1 = (a+1)^4
	// 18 = a^4 + a
	// 19 = a^4 + a + 1
	// 20 = a^4 + a^2 = a^2(a+1)^2
	// 23 = (a+1)(a^3+a^2+1)
	// 34 = a(a+1)^4
	// 45 = (a+1)^3(a^2+a+1)
	// 52 = a^2(a^3+a^2+1)
	// 54 = a(a+1)^2(a^2+a+1)
	// 57 = (a+1)^2(a^3+a^2+1)
	// 60 = a^2(a+1)^3
	// 63 = (a+1)(a^2+a+1)^2
	// 75 = (a+1)^3(a^3+a^2+1)
	// 90 = a(a+1)^3(a^2+a+1) = a^6 + a^4 + a^3 + a
	// 170 = a(a+1)^6
	int data[] = {
			// A1:
			10,0,0,0,
			0,10,0,0,
			4,10,10,0,
			0,17,0,10,

			// A2:
			10,0,0,0,
			0,10,0,0,
			2,0,10,0,
			0,15,0,10,

			// A3:
			10,0,0,0,
			2,10,0,0,
			0,0,10,0,
			0,0,15,10,

			// A4:
			60,0,0,0,
			12,60,0,0,
			12,0,60,0,
			54,34,34,60,

			// A5:
			12,0,0,0,
			4,12,0,0,
			0,0,12,0,
			0,0,34,12,

			// A6:
			10,0,0,0,
			4,0,10,0,
			0,10,10,0,
			10,60,17,10,

	};

	data_size = 16 + 1;
	nb_gens = 6;
	group_order = 192;

	gens = NEW_int(nb_gens * data_size);
	Orbiter->Int_vec.zero(gens, nb_gens * data_size);

	int h, i, j, c, m, l;
	int *v;
	geometry_global Gg;
	number_theory_domain NT;

	m = Orbiter->Int_vec.maximum(data, nb_gens * data_size);
	l = NT.int_log2(m) + 1;

	v = NEW_int(l);


	for (h = 0; h < nb_gens; h++) {
		for (i = 0; i < 16; i++) {
			Orbiter->Int_vec.zero(v, l);
			Gg.AG_element_unrank(p, v, 1, l, data[h * 16 + i]);
			c = 0;
			for (j = 0; j < l; j++) {
				c = mult(c, a);
				if (v[l - 1 - j]) {
					c = add(c, v[l - 1 - j]);
				}
			}
			gens[h * data_size + i] = c;
		}
		gens[h * data_size + 16] = 0;
	}
	FREE_int(v);

	if (f_v) {
		cout << "finite_field::cubic_surface_family_F13_generators" << endl;
		for (h = 0; h < nb_gens; h++) {
			cout << "generator " << h << ":" << endl;
			Orbiter->Int_vec.matrix_print(gens + h * data_size, 4, 4);
		}
	}
	if (f_v) {
		cout << "finite_field::cubic_surface_family_F13_generators done" << endl;
	}
}

int finite_field::is_unit_vector(int *v, int len, int k)
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


void finite_field::make_Fourier_matrices(
		int omega, int k, int *N, int **A, int **Av,
		int *Omega, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, i, j, om;

	if (f_v) {
		cout << "finite_field::make_Fourier_matrices" << endl;
	}

	Omega[k] = omega;
	for (h = k; h > 0; h--) {
		Omega[h - 1] = mult(Omega[h], Omega[h]);
	}

	for (h = k; h >= 0; h--) {
		A[h] = NEW_int(N[h] * N[h]);
		om = Omega[h];
		for (i = 0; i < N[h]; i++) {
			for (j = 0; j < N[h]; j++) {
				A[h][i * N[h] + j] = power(om, (i * j) % N[k]);
			}
		}
	}

	for (h = k; h >= 0; h--) {
		Av[h] = NEW_int(N[h] * N[h]);
		om = inverse(Omega[h]);
		for (i = 0; i < N[h]; i++) {
			for (j = 0; j < N[h]; j++) {
				Av[h][i * N[h] + j] = power(om, (i * j) % N[k]);
			}
		}
	}

	if (f_v) {
		for (h = k; h >= 0; h--) {
			cout << "A_" << N[h] << ":" << endl;
			Orbiter->Int_vec.matrix_print(A[h], N[h], N[h]);
		}

		for (h = k; h >= 0; h--) {
			cout << "Av_" << N[h] << ":" << endl;
			Orbiter->Int_vec.matrix_print(Av[h], N[h], N[h]);
		}
	}
	if (f_v) {
		cout << "finite_field::make_Fourier_matrices done" << endl;
	}
}





}}

