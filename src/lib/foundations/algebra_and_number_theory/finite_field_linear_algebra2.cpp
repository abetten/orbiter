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
		cout << "finite_field::reduce_mod_subspace_and_get_"
				"coefficient_vector: v=";
		int_vec_print(cout, v, len);
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
		int_vec_print(cout, v, len);
		cout << endl;
		cout << "coefficients=";
		int_vec_print(cout, coefficients, k);
		cout << endl;
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
		cout << "finite_field::reduce_mod_subspace before: v=";
		int_vec_print(cout, v, len);
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
		int_vec_print(cout, v, len);
		cout << endl;
		}
}

int finite_field::is_contained_in_subspace(int k,
	int len, int *basis, int *base_cols,
	int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "finite_field::is_contained_in_subspace testing v=";
		int_vec_print(cout, v, len);
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
		ostream &ost, int *M, int n, int k)
{
	int i;
	int *weights;

	weights = NEW_int(n + 1);
	code_projective_weight_enumerator(n, k,
		M, // [k * n]
		weights, // [n + 1]
		0 /*verbose_level*/);


	ost << "projective weights: " << endl;
	for (i = 0; i <= n; i++) {
		if (weights[i] == 0) {
			continue;
			}
		ost << i << " : " << weights[i] << endl;
		}
	FREE_int(weights);
}

int finite_field::code_minimum_distance(int n, int k,
		int *code, int verbose_level)
	// code[k * n]
{
	int f_v = (verbose_level >= 1);
	int *weight_enumerator;
	int i;

	if (f_v) {
		cout << "finite_field::code_minimum_distance" << endl;
		}
	weight_enumerator = NEW_int(n + 1);
	int_vec_zero(weight_enumerator, n + 1);
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
	FREE_int(weight_enumerator);
	return i;
}

void finite_field::codewords_affine(int n, int k,
	int *code, // [k * n]
	int *codewords, // q^k
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int N, h, rk;
	int *msg;
	int *word;
	geometry_global Gg;

	if (f_v) {
		cout << "finite_field::codewords_affine" << endl;
		}
	N = Gg.nb_AG_elements(k, q);
	if (f_v) {
		cout << N << " messages" << endl;
		}
	msg = NEW_int(k);
	word = NEW_int(n);

	for (h = 0; h < N; h++) {
		Gg.AG_element_unrank(q, msg, 1, k, h);
		mult_vector_from_the_left(msg, code, word, k, n);
		rk = Gg.AG_element_rank(q, word, 1, n);
		codewords[h] = rk;
		}
	FREE_int(msg);
	FREE_int(word);
	if (f_v) {
		cout << "finite_field::codewords_affine done" << endl;
		}
}

void finite_field::code_projective_weight_enumerator(
	int n, int k,
	int *code, // [k * n]
	int *weight_enumerator, // [n + 1]
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int N, h, wt, i;
	int *msg;
	int *word;
	int t0, t1, dt;
	geometry_global Gg;
	os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "finite_field::code_projective_weight_enumerator" << endl;
		}
	N = Gg.nb_AG_elements(k, q);
	if (f_v) {
		cout << N << " messages" << endl;
		}
	msg = NEW_int(k);
	word = NEW_int(n);

	int_vec_zero(weight_enumerator, n + 1);

	for (h = 0; h < N; h++) {
		if (f_v && (h % ONE_MILLION) == 0) {
			t1 = Os.os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			Os.time_check_delta(cout, dt);
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
		Gg.AG_element_unrank(q, msg, 1, k, h);
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


	FREE_int(msg);
	FREE_int(word);
}

void finite_field::code_weight_enumerator(
	int n, int k,
	int *code, // [k * n]
	int *weight_enumerator, // [n + 1]
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int N, h, wt, i;
	int *msg;
	int *word;
	int t0, t1, dt;
	geometry_global Gg;
	os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "finite_field::code_weight_enumerator" << endl;
		}
	N = Gg.nb_AG_elements(k, q);
	if (f_v) {
		cout << N << " messages" << endl;
		}
	msg = NEW_int(k);
	word = NEW_int(n);

	int_vec_zero(weight_enumerator, n + 1);

	for (h = 0; h < N; h++) {
		if ((h % ONE_MILLION) == 0) {
			t1 = Os.os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			Os.time_check_delta(cout, dt);
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
		Gg.AG_element_unrank(q, msg, 1, k, h);
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


	FREE_int(msg);
	FREE_int(word);
}


void finite_field::code_weight_enumerator_fast(int n, int k,
	int *code, // [k * n]
	int *weight_enumerator, // [n + 1]
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int N, h, wt, i;
	int *msg;
	int *word;
	int t0, t1, dt;
	geometry_global Gg;
	os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "finite_field::code_weight_enumerator" << endl;
		}
	N = Gg.nb_PG_elements(k - 1, q);
	if (f_v) {
		cout << N << " projective messages" << endl;
		}
	msg = NEW_int(k);
	word = NEW_int(n);


	int_vec_zero(weight_enumerator, n + 1);

	for (h = 0; h < N; h++) {
		if (((h % ONE_MILLION) == 0) && h) {
			t1 = Os.os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			Os.time_check_delta(cout, dt);
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
		PG_element_unrank_modified(msg, 1, k, h);
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
			int_vec_print(cout, msg, k);
			cout << " codeword ";
			int_vec_print(cout, word, n);
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


	FREE_int(msg);
	FREE_int(word);
}

void finite_field::code_projective_weights(
	int n, int k,
	int *code, // [k * n]
	int *&weights, // will be allocated [N]
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 1);
	int N, h, wt, i;
	int *msg;
	int *word;
	int t0, t1, dt;
	geometry_global Gg;
	os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "finite_field::code_projective_weights" << endl;
		}
	N = Gg.nb_PG_elements(k - 1, q);
	if (f_v) {
		cout << N << " projective messages" << endl;
		}
	weights = NEW_int(N);
	msg = NEW_int(k);
	word = NEW_int(n);

	for (h = 0; h < N; h++) {
		if ((h % ONE_MILLION) == 0) {
			t1 = Os.os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			Os.time_check_delta(cout, dt);
			cout << endl;
			}
		PG_element_unrank_modified(msg, 1, k, h);
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


	FREE_int(msg);
	FREE_int(word);
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

		int_vec_copy(Basis_V, Basis, dim_V * d);
		int_vec_copy(Basis_U + h * d, Basis + dim_V * d, d);
		rk = Gauss_easy(Basis, dim_V + 1, d);
		if (rk > dim_V) {
			ret = FALSE;
			goto done;
			}
		}
	ret = TRUE;
done:
	FREE_int(Basis);
	return ret;
}

void finite_field::Kronecker_product(int *A, int *B,
	int n, int *AB)
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
		int_matrix_print(A, m, deg);
		cout << "v = ";
		orbiter::foundations::int_vec_print(cout, v, deg);
		cout << endl;
		}
	// fill the m-th row of matrix A with v^rho:
	for (j = 0; j < deg; j++) {
		A[m * deg + j] = v[rho[j]];
		}
	if (f_vv) {
		cout << "finite_field::dependency "
				"after putting in row " << m << " A=" << endl;
		int_matrix_print(A, m + 1, deg);
		cout << "rho = ";
		orbiter::foundations::int_vec_print(cout, rho, deg);
		cout << endl;
		}
	for (k = 0; k < m; k++) {

		if (f_vv) {
			cout << "finite_field::dependency "
					"k=" << k << " A=" << endl;
			int_matrix_print(A, m + 1, deg);
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
			int_matrix_print(A, m + 1, deg);
			}

		} // next k
	if (f_vv) {
		cout << "finite_field::dependency "
				"m=" << m << " after reapply, A=" << endl;
		int_matrix_print(A, m + 1, deg);
		cout << "rho = ";
		orbiter::foundations::int_vec_print(cout, rho, deg);
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
		int_matrix_print(A, m + 1, deg);
		cout << "rho = ";
		orbiter::foundations::int_vec_print(cout, rho, deg);
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
	int_vec_zero(v, deg);
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
		int_vec_copy(v1, v, deg);

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
		orbiter::foundations::int_vec_print(cout, mue, mue_deg + 1);
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
	int_vec_copy(v, w1, n);
	for (j = 0; j < n; j++) {

		// put w1 in the j-th column of A:
		for (i = 0; i < n; i++) {
			A[i * n + j] = w1[i];
			}
		mult_vector_from_the_right(Mtx, w1, w2, n, n);
		int_vec_copy(w2, w1, n);
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
				orbiter::foundations::int_matrix_print(M, i + 1, k);
				}

			int_vec_copy(M, N, (i + 1) * k);
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
		int_matrix_print(M, k, k);
		}
	FREE_int(N);
}

void finite_field::make_all_irreducible_polynomials_of_degree_d(
		int d, int &nb, int *&Table, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int p, e, i, Q;
	int cnt;
	number_theory_domain NT;

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

	Table = NEW_int(nb * (d + 1));


	Q = NT.i_power_j(q, d);

	NT.factor_prime_power(q, p, e);

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"p=" << p << " e=" << e << endl;
		}

	//finite_field Fp;
	//Fp.init(p, 0 /*verbose_level*/);
	unipoly_domain FX(this);

	const char *poly;

	poly = get_primitive_polynomial(q, d, 0 /* verbose_level */);

	unipoly_object m;
	unipoly_object g;
	unipoly_object minpol;
	combinatorics_domain Combi;


	FX.create_object_by_rank_string(m, poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial m = ";
		FX.print_object(m, cout);
		cout << endl;
		}

	FX.create_object_by_rank(g, 0, __FILE__, __LINE__, verbose_level);
	FX.create_object_by_rank(minpol, 0, __FILE__, __LINE__, verbose_level);

	int *Frobenius;
	int *Normal_basis;
	int *v;
	int *w;

	//Frobenius = NEW_int(d * d);
	Normal_basis = NEW_int(d * d);
	v = NEW_int(d);
	w = NEW_int(d);

	FX.Frobenius_matrix(Frobenius, m, verbose_level - 3);
	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"Frobenius_matrix = " << endl;
		int_matrix_print(Frobenius, d, d);
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
		int_matrix_print(Normal_basis, d, d);
		cout << endl;
		}

	cnt = 0;

	Combi.int_vec_first_regular_word(v, d, Q, q);
	while (TRUE) {
		if (f_vv) {
			cout << "finite_field::make_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : v = ";
			int_vec_print(cout, v, d);
			cout << endl;
			}

		FX.gfq->mult_vector_from_the_right(Normal_basis, v, w, d, d);
		if (f_vv) {
			cout << "finite_field::make_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : w = ";
			int_vec_print(cout, w, d);
			cout << endl;
			}

		FX.delete_object(g);
		FX.create_object_of_degree(g, d - 1);
		for (i = 0; i < d; i++) {
			((int *) g)[1 + i] = w[i];
			}

		FX.minimum_polynomial_extension_field(g, m, minpol, d,
				Frobenius, verbose_level - 3);
		if (f_vv) {
			cout << "finite_field::make_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : v = ";
			int_vec_print(cout, v, d);
			cout << " irreducible polynomial = ";
			FX.print_object(minpol, cout);
			cout << endl;
			}

		for (i = 0; i <= d; i++) {
			Table[cnt * (d + 1) + i] = ((int *)minpol)[1 + i];
			}


		cnt++;


		if (!Combi.int_vec_next_regular_word(v, d, Q, q)) {
			break;
			}

		}

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_"
				"of_degree_d there are " << cnt
				<< " irreducible polynomials "
				"of degree " << d << " over " << "F_" << q << endl;
		}

	FREE_int(Frobenius);
	FREE_int(Normal_basis);
	FREE_int(v);
	FREE_int(w);
	FX.delete_object(m);
	FX.delete_object(g);
	FX.delete_object(minpol);


	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << q << " done" << endl;
		}
}

int finite_field::count_all_irreducible_polynomials_of_degree_d(
		int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int p, e, i, Q;
	int cnt;
	number_theory_domain NT;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << q << endl;
		}

	Q = NT.i_power_j(q, d);

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"Q=" << Q << endl;
		}
	NT.factor_prime_power(q, p, e);

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

	const char *poly;

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

	FX.create_object_by_rank(g, 0, __FILE__, __LINE__, verbose_level);
	FX.create_object_by_rank(minpol, 0, __FILE__, __LINE__, verbose_level);

	int *Frobenius;
	int *F2;
	int *Normal_basis;
	int *v;
	int *w;

	//Frobenius = NEW_int(d * d);
	F2 = NEW_int(d * d);
	Normal_basis = NEW_int(d * d);
	v = NEW_int(d);
	w = NEW_int(d);

	FX.Frobenius_matrix(Frobenius, m, verbose_level - 3);
	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"Frobenius_matrix = " << endl;
		int_matrix_print(Frobenius, d, d);
		cout << endl;
		}

	mult_matrix_matrix(Frobenius, Frobenius, F2, d, d, d,
			0 /* verbose_level */);
	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"Frobenius^2 = " << endl;
		int_matrix_print(F2, d, d);
		cout << endl;
		}


	FX.compute_normal_basis(d, Normal_basis, Frobenius, verbose_level - 3);

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"Normal_basis = " << endl;
		int_matrix_print(Normal_basis, d, d);
		cout << endl;
		}

	cnt = 0;
	Combi.int_vec_first_regular_word(v, d, Q, q);
	while (TRUE) {
		if (f_vv) {
			cout << "finite_field::count_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : v = ";
			int_vec_print(cout, v, d);
			cout << endl;
			}

		FX.gfq->mult_vector_from_the_right(Normal_basis, v, w, d, d);
		if (f_vv) {
			cout << "finite_field::count_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : w = ";
			int_vec_print(cout, w, d);
			cout << endl;
			}

		FX.delete_object(g);
		FX.create_object_of_degree(g, d - 1);
		for (i = 0; i < d; i++) {
			((int *) g)[1 + i] = w[i];
			}

		FX.minimum_polynomial_extension_field(g, m, minpol, d,
				Frobenius, verbose_level - 3);
		if (f_vv) {
			cout << "finite_field::count_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : v = ";
			int_vec_print(cout, v, d);
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

		if (!Combi.int_vec_next_regular_word(v, d, Q, q)) {
			break;
			}

		}

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_"
				"of_degree_d there are " << cnt << " irreducible polynomials "
				"of degree " << d << " over " << "F_" << q << endl;
		}

	FREE_int(Frobenius);
	FREE_int(F2);
	FREE_int(Normal_basis);
	FREE_int(v);
	FREE_int(w);
	FX.delete_object(m);
	FX.delete_object(g);
	FX.delete_object(minpol);

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_"
				"of_degree_d done" << endl;
		}
	return cnt;
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

	int_vec_copy(U, M, d * n);

	if (Gauss_simple(M, d, n, base_cols,
			0 /* verbose_level */) != d) {
		cout << "finite_field::adjust_basis rank "
				"of matrix is not d" << endl;
		exit(1);
	}
	ii = 0;
	for (i = 0; i < k; i++) {
		int_vec_copy(V + i * n, M + (d + ii) * n, n);
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
	int_vec_copy(M, V, k * n);


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
	int_vec_copy(Gen + d * n, v, n);


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
			int_vec_print(cout, w, k);
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
		int_vec_copy(Gen + rk * n, z, n);
		if (f_vv) {
			cout << "before reduce=";
			int_vec_print(cout, Gen + rk * n, n);
			cout << endl;
		}

		// reduce modulo the subspace:
		for (j = 0; j < rk; j++) {
			b = base_cols[j];
			Gauss_step(Gen + j * n, Gen + rk * n, n, b, 0 /* verbose_level */);
		}

		if (f_vv) {
			cout << "after reduce=";
			int_vec_print(cout, Gen + rk * n, n);
			cout << endl;
		}


		// see if we got something nonzero:
		if (!Sorting.int_vec_is_zero(Gen + rk * n, n)) {
			break;
		}
		// keep moving on to the next vector

	} // while

	int_vec_copy(z, v, n);


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
		int_matrix_print(M, n, n);
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
					int_matrix_print(M, n, n);
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
					int_matrix_print(M, n, n);
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
		int_matrix_print(A, d + 1, d);
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
		int_matrix_print(B, d, n);
		}
	RREF_and_kernel(n, d, B, 0 /* verbose_level */);
	if (f_v) {
		cout << "B after=" << endl;
		int_matrix_print(B, n, n);
		}
	xd = B[d * n + d - 1];
	x = negate(inverse(xd));
	for (i = 0; i < d; i++) {
		B[d * n + i] = mult(x, B[d * n + i]);
		}
	if (f_v) {
		cout << "last row of B after scaling : " << endl;
		int_matrix_print(B + d * n, 1, n);
		}
	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			A2[i * d + j] = mult(B[d * n + i], A[i * d + j]);
			}
		}
	if (f_v) {
		cout << "A2=" << endl;
		int_matrix_print(A2, d, d);
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
		orbiter::foundations::int_vec_print(cout, perm, d + 1);
		cout << endl;
		}
	if (f_v) {
		cout << "A=" << endl;
		int_matrix_print(A, d + 1, d);
		}
	if (f_v) {
		cout << "B=" << endl;
		int_matrix_print(B, d + 1, d);
		}

	for (i = 0; i < d + 1; i++) {
		j = perm[i];
		int_vec_copy(A + j * d, A1 + i * d, d);
		}

	if (f_v) {
		cout << "A1=" << endl;
		int_matrix_print(A1, d + 1, d);
		}


	if (f_v) {
		cout << "mapping A1 to standard frame:" << endl;
		}
	map_to_standard_frame(d, A1, T1, verbose_level);
	if (f_v) {
		cout << "T1=" << endl;
		int_matrix_print(T1, d, d);
		}
	if (f_v) {
		cout << "mapping B to standard frame:" << endl;
		}
	map_to_standard_frame(d, B, T2, 0 /* verbose_level */);
	if (f_v) {
		cout << "T2=" << endl;
		int_matrix_print(T2, d, d);
		}
	matrix_inverse(T2, T3, d, 0 /* verbose_level */);
	if (f_v) {
		cout << "T3=" << endl;
		int_matrix_print(T3, d, d);
		}
	mult_matrix_matrix(T1, T3, Transform, d, d, d, 0 /* verbose_level */);
	if (f_v) {
		cout << "Transform=" << endl;
		int_matrix_print(Transform, d, d);
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

	int_vec_copy(B, B1, (d + k + 1) * d);
	for (i = 0; i < d + k + 1; i++) {
		//PG_element_normalize(*this, B1 + i * d, 1, d);
		PG_element_rank_modified(B1 + i * d, 1, d, a);
		B_set[i] = a;
		}
	Sorting.int_vec_heapsort(B_set, d + k + 1);
	if (f_v) {
		cout << "B_set = ";
		orbiter::foundations::int_vec_print(cout, B_set, d + k + 1);
		cout << endl;
		}


	overall_cnt = 0;
	nCk = Combi.int_n_choose_k(d + k + 1, d + 1);
	for (h = 0; h < nCk; h++) {
		Combi.unrank_k_subset(h, subset, d + k + 1, d + 1);
		Combi.set_complement(subset, d + 1, subset + d + 1, k, d + k + 1);

		if (f_v) {
			cout << "subset " << h << " / " << nCk << " is ";
			orbiter::foundations::int_vec_print(cout, subset, d + 1);
			cout << ", the complement is ";
			orbiter::foundations::int_vec_print(cout, subset + d + 1, k);
			cout << endl;
			}


		for (i = 0; i < d + k + 1; i++) {
			j = subset[i];
			int_vec_copy(A + j * d, A1 + i * d, d);
			}
		if (f_v) {
			cout << "A1=" << endl;
			int_matrix_print(A1, d + k + 1, d);
			}

		cnt = 0;
		Combi.first_lehmercode(d + 1, lehmercode);
		while (TRUE) {
			if (f_v) {
				cout << "lehmercode: ";
				orbiter::foundations::int_vec_print(cout, lehmercode, d + 1);
				cout << endl;
				}
			Combi.lehmercode_to_permutation(d + 1, lehmercode, perm);
			if (f_v) {
				cout << "permutation: ";
				orbiter::foundations::int_vec_print(cout, perm, d + 1);
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
				orbiter::foundations::int_vec_print(cout, image_set, d + k + 1);
				cout << endl;
				}
			Sorting.int_vec_heapsort(image_set, d + k + 1);
			if (f_v) {
				cout << "image_set after sorting: ";
				orbiter::foundations::int_vec_print(cout, image_set, d + k + 1);
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
		int_vec_copy(Coord + a * k, T + (k - 1) * k, k);
		for (i = 0; i < k - 1; i++) {
			a = C[i];
			int_vec_copy(Coord + a * k, T + i * k, k);
			}
		if (f_vv) {
			cout << "u=" << u << " / " << sz_U << " the matrix is:" << endl;
			int_matrix_print(T, k, k);
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
	int_vec_zero(gens, nb_gens * data_size);
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


}}

