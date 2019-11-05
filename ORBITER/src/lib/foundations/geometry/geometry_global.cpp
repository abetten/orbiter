/*
 * geometry_global.cpp
 *
 *  Created on: Apr 19, 2019
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {



geometry_global::geometry_global()
{

}

geometry_global::~geometry_global()
{

}


long int geometry_global::nb_PG_elements(int n, int q)
// $\frac{q^{n+1} - 1}{q-1} = \sum_{i=0}^{n} q^i $
{
	long int qhl, l, deg;

	l = 0;
	qhl = 1;
	deg = 0;
	while (l <= n) {
		deg += qhl;
		qhl *= q;
		l++;
		}
	return deg;
}

long int geometry_global::nb_PG_elements_not_in_subspace(int n, int m, int q)
// |PG_n(q)| - |PG_m(q)|
{
	int a, b;

	a = nb_PG_elements(n, q);
	b = nb_PG_elements(m, q);
	return a - b;
}

long int geometry_global::nb_AG_elements(int n, int q)
// $q^n$
{
	number_theory_domain NT;

	return NT.i_power_j_lint(q, n);
}

long int geometry_global::AG_element_rank(int q, int *v, int stride, int len)
{
	int i;
	long int a;

	if (len <= 0) {
		cout << "geometry_global::AG_element_rank len <= 0" << endl;
		exit(1);
		}
	a = 0;
	for (i = len - 1; i >= 0; i--) {
		a += v[i * stride];
		if (i > 0) {
			a *= q;
			}
		}
	return a;
}

void geometry_global::AG_element_unrank(int q, int *v, int stride, int len, long int a)
{
	int i, b;

#if 1
	if (len <= 0) {
		cout << "geometry_global::AG_element_unrank len <= 0" << endl;
		exit(1);
		}
#endif
	for (i = 0; i < len; i++) {
		b = a % q;
		v[i * stride] = b;
		a /= q;
		}
}

void geometry_global::AG_element_rank_longinteger(int q,
		int *v, int stride, int len, longinteger_object &a)
{
	longinteger_domain D;
	longinteger_object Q, a1;
	int i;

	if (len <= 0) {
		cout << "geometry_global::AG_element_rank_longinteger len <= 0" << endl;
		exit(1);
	}
	a.create(0);
	Q.create(q);
	for (i = len - 1; i >= 0; i--) {
		a.add_int(v[i * stride]);
		//cout << "AG_element_rank_longinteger
		//after add_int " << a << endl;
		if (i > 0) {
			D.mult(a, Q, a1);
			a.swap_with(a1);
			//cout << "AG_element_rank_longinteger
			//after mult " << a << endl;
		}
	}
}

void geometry_global::AG_element_unrank_longinteger(int q,
		int *v, int stride, int len, longinteger_object &a)
{
	int i, r;
	longinteger_domain D;
	longinteger_object a0, Q, a1;

	a.assign_to(a0);
	if (len <= 0) {
		cout << "geometry_global::AG_element_unrank_longinteger len <= 0" << endl;
		exit(1);
	}
	for (i = 0; i < len; i++) {
		D.integral_division_by_int(a0, q, a1, r);
		//r = a % q;
		v[i * stride] = r;
		//a /= q;
		a0.swap_with(a1);
	}
}


int geometry_global::PG_element_modified_is_in_subspace(int n, int m, int *v)
{
	int j;

	for (j = m + 1; j < n + 1; j++) {
		if (v[j]) {
			return FALSE;
		}
	}
	return TRUE;
}


void geometry_global::test_PG(int n, int q)
{
	finite_field F;
	int m;
	int verbose_level = 1;

	F.init(q, verbose_level);

	cout << "all elements of PG_" << n << "(" << q << ")" << endl;
	F.display_all_PG_elements(n);

	for (m = 0; m < n; m++) {
		cout << "all elements of PG_" << n << "(" << q << "), "
			"not in a subspace of dimension " << m << endl;
		F.display_all_PG_elements_not_in_subspace(n, m);
		}

}

void geometry_global::create_Fisher_BLT_set(int *Fisher_BLT,
		int q, const char *poly_q, const char *poly_Q, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	unusual_model U;

	U.setup(q, poly_q, poly_Q, verbose_level);
	U.create_Fisher_BLT_set(Fisher_BLT, verbose_level);

}

void geometry_global::create_Linear_BLT_set(int *BLT, int q,
		const char *poly_q, const char *poly_Q, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	unusual_model U;

	U.setup(q, poly_q, poly_Q, verbose_level);
	U.create_Linear_BLT_set(BLT, verbose_level);

}

void geometry_global::create_Mondello_BLT_set(int *BLT, int q,
		const char *poly_q, const char *poly_Q, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	unusual_model U;

	U.setup(q, poly_q, poly_Q, verbose_level);
	U.create_Mondello_BLT_set(BLT, verbose_level);

}

void geometry_global::print_quadratic_form_list_coded(int form_nb_terms,
	int *form_i, int *form_j, int *form_coeff)
{
	int k;

	for (k = 0; k < form_nb_terms; k++) {
		cout << "i=" << form_i[k] << " j=" << form_j[k]
			<< " coeff=" << form_coeff[k] << endl;
		}
}

void geometry_global::make_Gram_matrix_from_list_coded_quadratic_form(
	int n, finite_field &F,
	int nb_terms, int *form_i, int *form_j, int *form_coeff, int *Gram)
{
	int k, i, j, c;

	int_vec_zero(Gram, n * n);
#if 0
	for (i = 0; i < n * n; i++)
		Gram[i] = 0;
#endif
	for (k = 0; k < nb_terms; k++) {
		i = form_i[k];
		j = form_j[k];
		c = form_coeff[k];
		if (c == 0) {
			continue;
			}
		Gram[i * n + j] = F.add(Gram[i * n + j], c);
		Gram[j * n + i] = F.add(Gram[j * n + i], c);
		}
}

void geometry_global::add_term(int n, finite_field &F,
	int &nb_terms, int *form_i, int *form_j, int *form_coeff,
	int *Gram,
	int i, int j, int coeff)
{
	form_i[nb_terms] = i;
	form_j[nb_terms] = j;
	form_coeff[nb_terms] = coeff;
	if (i == j) {
		Gram[i * n + j] = F.mult(2, coeff);
		}
	else {
		Gram[i * n + j] = coeff;
		Gram[j * n + i] = coeff;
		}
	nb_terms++;
}


void geometry_global::determine_conic(int q, const char *override_poly,
		long int *input_pts, int nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	finite_field F;
	projective_space *P;
	//int f_basis = TRUE;
	//int f_semilinear = TRUE;
	//int f_with_group = FALSE;
	int v[3];
	int len = 3;
	int six_coeffs[6];
	int i;

	if (f_v) {
		cout << "determine_conic q=" << q << endl;
		cout << "input_pts: ";
		lint_vec_print(cout, input_pts, nb_pts);
		cout << endl;
		}
	F.init_override_polynomial(q, override_poly, verbose_level);

	P = NEW_OBJECT(projective_space);
	if (f_vv) {
		cout << "determine_conic before P->init" << endl;
		}
	P->init(len - 1, &F,
		FALSE,
		verbose_level - 2/*MINIMUM(2, verbose_level)*/);

	if (f_vv) {
		cout << "determine_conic after P->init" << endl;
		}
	P->determine_conic_in_plane(input_pts, nb_pts,
			six_coeffs, verbose_level - 2);

	if (f_v) {
		cout << "determine_conic the six coefficients are ";
		int_vec_print(cout, six_coeffs, 6);
		cout << endl;
		}

	long int points[1000];
	int nb_points;
	//int v[3];

	P->conic_points(input_pts, six_coeffs,
			points, nb_points, verbose_level - 2);
	if (f_v) {
		cout << "the " << nb_points << " conic points are: ";
		lint_vec_print(cout, points, nb_points);
		cout << endl;
		for (i = 0; i < nb_points; i++) {
			P->unrank_point(v, points[i]);
			cout << i << " : " << points[i] << " : ";
			int_vec_print(cout, v, 3);
			cout << endl;
			}
		}
	FREE_OBJECT(P);
}


int geometry_global::test_if_arc(finite_field *Fq, int *pt_coords,
		int *set, int set_sz, int k, int verbose_level)
// Used by Hill_cap56()
{
	int f_v = FALSE; //(verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int subset[3];
	int subset1[3];
	int *Mtx;
	int ret = FALSE;
	int i, j, a, rk;
	combinatorics_domain Combi;
	sorting Sorting;


	if (f_v) {
		cout << "test_if_arc testing set" << endl;
		int_vec_print(cout, set, set_sz);
		cout << endl;
		}
	Mtx = NEW_int(3 * k);

	Combi.first_k_subset(subset, set_sz, 3);
	while (TRUE) {
		for (i = 0; i < 3; i++) {
			subset1[i] = set[subset[i]];
			}
		Sorting.int_vec_heapsort(subset1, 3);
		if (f_vv) {
			cout << "testing subset ";
			int_vec_print(cout, subset1, 3);
			cout << endl;
			}

		for (i = 0; i < 3; i++) {
			a = subset1[i];
			for (j = 0; j < k; j++) {
				Mtx[i * k + j] = pt_coords[a * k + j];
				}
			}
		if (f_vv) {
			cout << "matrix:" << endl;
			print_integer_matrix_width(cout, Mtx, 3, k, k, 1);
			}
		rk = Fq->Gauss_easy(Mtx, 3, k);
		if (rk < 3) {
			if (f_v) {
				cout << "not an arc" << endl;
				}
			goto done;
			}
		if (!Combi.next_k_subset(subset, set_sz, 3)) {
			break;
			}
		}
	if (f_v) {
		cout << "passes the arc test" << endl;
		}
	ret = TRUE;
done:

	FREE_int(Mtx);
	return ret;
}

void geometry_global::create_Buekenhout_Metz(
	finite_field *Fq, finite_field *FQ,
	int f_classical, int f_Uab, int parameter_a, int parameter_b,
	char *fname, int &nb_pts, int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, rk, d = 3;
	int v[3];
	buekenhout_metz *BM;

	if (f_v) {
		cout << "create_Buekenhout_Metz" << endl;
		}


	BM = NEW_OBJECT(buekenhout_metz);

	BM->init(Fq, FQ,
		f_Uab, parameter_a, parameter_b, f_classical, verbose_level);


	if (BM->f_Uab) {
		BM->init_ovoid_Uab_even(
				BM->parameter_a, BM->parameter_b,
				verbose_level);
		}
	else {
		BM->init_ovoid(verbose_level);
		}

	BM->create_unital(verbose_level);

	//BM->write_unital_to_file();

	nb_pts = BM->sz;
	Pts = NEW_int(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Pts[i] = BM->U[i];
		}


	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		rk = Pts[i];
		BM->P2->unrank_point(v, rk);
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << rk << endl;
			}
		}



	strcpy(fname, "unital_");
	BM->get_name(fname + strlen(fname));
	strcat(fname, ".txt");

	FREE_OBJECT(BM);

}


long int geometry_global::count_Sbar(int n, int q)
{
	return count_T1(1, n, q);
}

long int geometry_global::count_S(int n, int q)
{
	return (q - 1) * count_Sbar(n, q) + 1;
}

long int geometry_global::count_N1(int n, int q)
{
	if (n <= 0) {
		return 0;
		}
	return nb_pts_N1(n, q);
}

long int geometry_global::count_T1(int epsilon, int n, int q)
// n = Witt index
{
	number_theory_domain NT;

	if (n < 0) {
		//cout << "count_T1 n is negative. n=" << n << endl;
		return 0;
		}
	if (epsilon == 1) {
		return ((NT.i_power_j_lint(q, n) - 1) *
				(NT.i_power_j_lint(q, n - 1) + 1)) / (q - 1);
		}
	else if (epsilon == 0) {
		return count_T1(1, n, q) + count_N1(n, q);
		}
	else {
		cout << "count_T1 epsilon = " << epsilon
				<< " not yet implemented, returning 0" << endl;
		return 0;
		}
	//exit(1);
}

long int geometry_global::count_T2(int n, int q)
{
	number_theory_domain NT;

	if (n <= 0) {
		return 0;
		}
	return (NT.i_power_j_lint(q, 2 * n - 2) - 1) *
			(NT.i_power_j_lint(q, n) - 1) *
			(NT.i_power_j_lint(q, n - 2) + 1) / ((q - 1) * (NT.i_power_j_lint(q, 2) - 1));
}

long int geometry_global::nb_pts_Qepsilon(int epsilon, int k, int q)
// number of singular points on Q^epsilon(k,q)
{
	if (epsilon == 0) {
		return nb_pts_Q(k, q);
		}
	else if (epsilon == 1) {
		return nb_pts_Qplus(k, q);
		}
	else if (epsilon == -1) {
		return nb_pts_Qminus(k, q);
		}
	else {
		cout << "nb_pts_Qepsilon epsilon must be one of 0,1,-1" << endl;
		exit(1);
		}
}

int geometry_global::dimension_given_Witt_index(int epsilon, int n)
{
	if (epsilon == 0) {
		return 2 * n + 1;
		}
	else if (epsilon == 1) {
		return 2 * n;
		}
	else if (epsilon == -1) {
		return 2 * n + 2;
		}
	else {
		cout << "dimension_given_Witt_index "
				"epsilon must be 0,1,-1" << endl;
		exit(1);
		}
}

int geometry_global::Witt_index(int epsilon, int k)
// k = projective dimension
{
	int n;

	if (epsilon == 0) {
		if (!EVEN(k)) {
			cout << "Witt_index dimension k must be even" << endl;
			cout << "k = " << k << endl;
			cout << "epsilon = " << epsilon << endl;
			exit(1);
			}
		n = k >> 1; // Witt index
		}
	else if (epsilon == 1) {
		if (!ODD(k)) {
			cout << "Witt_index dimension k must be odd" << endl;
			cout << "k = " << k << endl;
			cout << "epsilon = " << epsilon << endl;
			exit(1);
			}
		n = (k >> 1) + 1; // Witt index
		}
	else if (epsilon == -1) {
		if (!ODD(k)) {
			cout << "Witt_index dimension k must be odd" << endl;
			cout << "k = " << k << endl;
			cout << "epsilon = " << epsilon << endl;
			exit(1);
			}
		n = k >> 1; // Witt index
		}
	else {
		cout << "Witt_index epsilon must be one of 0,1,-1" << endl;
		exit(1);
		}
	return n;
}

long int geometry_global::nb_pts_Q(int k, int q)
// number of singular points on Q(k,q), parabolic quadric, so k is even
{
	int n;

	n = Witt_index(0, k);
	return nb_pts_Sbar(n, q) + nb_pts_N1(n, q);
}

long int geometry_global::nb_pts_Qplus(int k, int q)
// number of singular points on Q^+(k,q)
{
	int n;

	n = Witt_index(1, k);
	return nb_pts_Sbar(n, q);
}

long int geometry_global::nb_pts_Qminus(int k, int q)
// number of singular points on Q^-(k,q)
{
	int n;

	n = Witt_index(-1, k);
	return nb_pts_Sbar(n, q) + (q + 1) * nb_pts_N1(n, q);
}


// #############################################################################
// the following functions are for the hyperbolic quadric with Witt index n:
// #############################################################################

long int geometry_global::nb_pts_S(int n, int q)
// Number of singular vectors (including the zero vector)
{
	long int a;

	if (n <= 0) {
		cout << "nb_pts_S n <= 0" << endl;
		exit(1);
		}
	if (n == 1) {
		// q-1 vectors of the form (x,0) for x \neq 0,
		// q-1 vectors of the form (0,x) for x \neq 0
		// 1 vector of the form (0,0)
		// for a total of 2 * q - 1 vectors
		return 2 * q - 1;
		}
	a = nb_pts_S(1, q) * nb_pts_S(n - 1, q);
	a += nb_pts_N(1, q) * nb_pts_N1(n - 1, q);
	return a;
}

long int geometry_global::nb_pts_N(int n, int q)
// Number of non-singular vectors.
// Of course, |N(n,q)| + |S(n,q)| = q^{2n}
// |N(n,q)| = (q - 1) * |N1(n,q)|
{
	long int a;

	if (n <= 0) {
		cout << "nb_pts_N n <= 0" << endl;
		exit(1);
		}
	if (n == 1) {
		return (long int) (q - 1) * (long int) (q - 1);
		}
	a = nb_pts_S(1, q) * nb_pts_N(n - 1, q);
	a += nb_pts_N(1, q) * nb_pts_S(n - 1, q);
	a += nb_pts_N(1, q) * (q - 2) * nb_pts_N1(n - 1, q);
	return a;
}

long int geometry_global::nb_pts_N1(int n, int q)
// Number of non-singular vectors
// for one fixed value of the quadratic form
// i.e. number of solutions of
// \sum_{i=0}^{n-1} x_{2i}x_{2i+1} = s
// for some fixed s \neq 0.
{
	long int a;

	//cout << "nb_pts_N1 n=" << n << " q=" << q << endl;
	if (n <= 0) {
		cout << "nb_pts_N1 n <= 0" << endl;
		exit(1);
		}
	if (n == 1) {
		//cout << "gives " << q - 1 << endl;
		return q - 1;
		}
	a = nb_pts_S(1, q) * nb_pts_N1(n - 1, q);
	a += nb_pts_N1(1, q) * nb_pts_S(n - 1, q);
	a += nb_pts_N1(1, q) * (q - 2) * nb_pts_N1(n - 1, q);
	//cout << "gives " << a << endl;
	return a;
}

long int geometry_global::nb_pts_Sbar(int n, int q)
// number of singular projective points
// |S(n,q)| = (q-1) * |Sbar(n,q)| + 1
{
	long int a;

	if (n <= 0) {
		cout << "nb_pts_Sbar n <= 0" << endl;
		exit(1);
		}
	if (n == 1) {
		return 2;
		// namely (0,1) and (1,0)
		}
	a = nb_pts_Sbar(n - 1, q);
	a += nb_pts_Sbar(1, q) * nb_pts_S(n - 1, q);
	a += nb_pts_Nbar(1, q) * nb_pts_N1(n - 1, q);
	if (a < 0) {
		cout << "geometry_global::nb_pts_Sbar a < 0, overflow" << endl;
		exit(1);
	}
	return a;
}

long int geometry_global::nb_pts_Nbar(int n, int q)
// |Nbar(1,q)| = q - 1
{
	//int a;

	if (n <= 0) {
		cout << "nb_pts_Nbar n <= 0" << endl;
		exit(1);
		}
	if (n == 1) {
		return (q - 1);
		}
	cout << "nb_pts_Nbar should only be called for n = 1" << endl;
	exit(1);
#if 0
	a = nb_pts_Nbar(n - 1, q);
	a += nb_pts_Sbar(1, q) * nb_pts_N(n - 1, q);
	a += nb_pts_Nbar(1, q) * nb_pts_S(n - 1, q);
	a += nb_pts_Nbar(1, q) * (q - 2) * nb_pts_N1(n - 1, q);
	return a;
#endif
}



void geometry_global::test_Orthogonal(int epsilon, int k, int q)
// only works for epsilon = 0
{
	finite_field GFq;
	int *v;
	int i, j, a, stride = 1, /*n,*/ len; //, h, wt;
	int nb;
	int c1 = 0, c2 = 0, c3 = 0;
	int verbose_level = 0;

	cout << "test_Orthogonal" << endl;
	GFq.init(q, verbose_level);
	v = NEW_int(k + 1);
	//n = Witt_index(epsilon, k);
	len = k + 1;
	nb = nb_pts_Qepsilon(epsilon, k, q);
	cout << "Q^" << epsilon << "(" << k << "," << q << ") has "
			<< nb << " singular points" << endl;
	if (epsilon == 0) {
		c1 = 1;
		}
	else if (epsilon == 1) {
		}
	else if (epsilon == -1) {
		GFq.choose_anisotropic_form(c1, c2, c3, TRUE);
		}
	for (i = 0; i < nb; i++) {
		GFq.Q_epsilon_unrank(v, stride, epsilon, k, c1, c2, c3, i, 0 /* verbose_level */);

#if 0
		wt = 0;
		for (h = 0; h < len; h++) {
			if (v[h])
				wt++;
			}
#endif
		cout << i << " : ";
		int_vec_print(cout, v, len);
		cout << " : ";
		a = GFq.evaluate_quadratic_form(v, stride, epsilon, k,
				c1, c2, c3);
		cout << a;
		j = GFq.Q_epsilon_rank(v, stride, epsilon, k, c1, c2, c3, 0 /* verbose_level */);
		cout << " : " << j;
#if 0
		if (wt == 1) {
			cout << " -- unit vector";
			}
		cout << " weight " << wt << " vector";
#endif
		cout << endl;
		if (j != i) {
			cout << "error" << endl;
			exit(1);
			}
		}


	FREE_int(v);
	cout << "test_Orthogonal done" << endl;
}

void geometry_global::test_orthogonal(int n, int q)
{
	int *v;
	finite_field GFq;
	int i, j, a, stride = 1;
	int nb;
	int verbose_level = 0;

	cout << "test_orthogonal" << endl;
	GFq.init(q, verbose_level);
	v = NEW_int(2 * n);
	nb = nb_pts_Sbar(n, q);
	cout << "\\Omega^+(" << 2 * n << "," << q << ") has " << nb
			<< " singular points" << endl;
	for (i = 0; i < nb; i++) {
		GFq.Sbar_unrank(v, stride, n, i, 0 /* verbose_level */);
		cout << i << " : ";
		int_set_print(v, 2 * n);
		cout << " : ";
		a = GFq.evaluate_hyperbolic_quadratic_form(v, stride, n);
		cout << a;
		GFq.Sbar_rank(v, stride, n, j, 0 /* verbose_level */);
		cout << " : " << j << endl;
		if (j != i) {
			cout << "error" << endl;
			exit(1);
			}
		}
	cout << "\\Omega^+(" << 2 * n << "," << q << ") has " << nb
			<< " singular points" << endl;
	FREE_int(v);
	cout << "test_orthogonal done" << endl;
}



void geometry_global::create_BLT(int f_embedded,
	finite_field *FQ, finite_field *Fq,
	int f_Linear,
	int f_Fisher,
	int f_Mondello,
	int f_FTWKB,
	char *fname, int &nb_pts, int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, j;
	int epsilon = 0;
	int n = 4;
	//int c1 = 0, c2 = 0, c3 = 0;
	//int d = 5;
	//int *Pts1;
	orthogonal *O;
	int q = Fq->q;
	//int *v;
	//char BLT_label[1000];

	if (f_v) {
		cout << "create_BLT" << endl;
		}
	O = NEW_OBJECT(orthogonal);
	if (f_v) {
		cout << "create_BLT before O->init" << endl;
		}
	O->init(epsilon, n + 1, Fq, verbose_level - 1);
	nb_pts = q + 1;

	//BLT = BLT_representative(q, BLT_k);

	//v = NEW_int(d);
	//Pts1 = NEW_int(nb_pts);
	Pts = NEW_int(nb_pts);

	cout << "create_BLT currently disabled" << endl;
	exit(1);
#if 0
#if 0
	if (f_Linear) {
		strcpy(BLT_label, "Linear");
		create_Linear_BLT_set(Pts1, FQ, Fq, verbose_level - 1);
		}
	else if (f_Fisher) {
		strcpy(BLT_label, "Fi");
		create_Fisher_BLT_set(Pts1, FQ, Fq, verbose_level - 1);
		}
	else if (f_Mondello) {
		strcpy(BLT_label, "Mondello");
		create_Mondello_BLT_set(Pts1, FQ, Fq, verbose_level - 1);
		}
	else if (f_FTWKB) {
		strcpy(BLT_label, "FTWKB");
		create_FTWKB_BLT_set(O, Pts1, verbose_level - 1);
		}
	else {
		cout << "create_BLT no type" << endl;
		exit(1);
		}
#endif
	if (f_v) {
		cout << "i : orthogonal rank : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		Q_epsilon_unrank(*Fq, v, 1, epsilon, n, c1, c2, c3, Pts1[i]);
		if (f_embedded) {
			PG_element_rank_modified(*Fq, v, 1, d, j);
			}
		else {
			j = Pts1[i];
			}
		// recreate v:
		Q_epsilon_unrank(*Fq, v, 1, epsilon, n, c1, c2, c3, Pts1[i]);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : " << setw(4) << Pts1[i] << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}

#if 0
	cout << "list of points:" << endl;
	cout << nb_pts << endl;
	for (i = 0; i < nb_pts; i++) {
		cout << Pts[i] << " ";
		}
	cout << endl;
#endif

	//char fname[1000];
	if (f_embedded) {
		sprintf(fname, "BLT_%s_%d_embedded.txt", BLT_label, q);
		}
	else {
		sprintf(fname, "BLT_%s_%d.txt", BLT_label, q);
		}
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_int(Pts1);
	FREE_int(v);
	//FREE_int(L);
	FREE_OBJECT(O);
#endif
}



//static int TDO_upper_bounds_v_max_init = 12;
static int TDO_upper_bounds_v_max = -1;
static int *TDO_upper_bounds_table = NULL;
static int *TDO_upper_bounds_table_source = NULL;
	// 0 = nothing
	// 1 = packing number
	// 2 = braun test
	// 3 = maxfit
static int TDO_upper_bounds_initial_data[] = {
3,3,1,
4,3,1,
5,3,2,
6,3,4,
7,3,7,
8,3,8,
9,3,12,
10,3,13,
11,3,17,
12,3,20,
4,4,1,
5,4,1,
6,4,1,
7,4,2,
8,4,2,
9,4,3,
10,4,5,
11,4,6,
12,4,9,
5,5,1,
6,5,1,
7,5,1,
8,5,1,
9,5,2,
10,5,2,
11,5,2,
12,5,3,
6,6,1,
7,6,1,
8,6,1,
9,6,1,
10,6,1,
11,6,2,
12,6,2,
7,7,1,
8,7,1,
9,7,1,
10,7,1,
11,7,1,
12,7,1,
8,8,1,
9,8,1,
10,8,1,
11,8,1,
12,8,1,
9,9,1,
10,9,1,
11,9,1,
12,9,1,
10,10,1,
11,10,1,
12,10,1,
11,11,1,
12,11,1,
12,12,1,
-1
};

int &geometry_global::TDO_upper_bound(int i, int j)
{
	int m, bound;

	if (i <= 0) {
		cout << "TDO_upper_bound i <= 0, i = " << i << endl;
		exit(1);
		}
	if (j <= 0) {
		cout << "TDO_upper_bound j <= 0, j = " << j << endl;
		exit(1);
		}
	m = MAXIMUM(i, j);
	if (TDO_upper_bounds_v_max == -1) {
		TDO_refine_init_upper_bounds(12);
		}
	if (m > TDO_upper_bounds_v_max) {
		//cout << "I need TDO_upper_bound " << i << "," << j << endl;
		TDO_refine_extend_upper_bounds(m);
		}
	if (TDO_upper_bound_source(i, j) != 1) {
		//cout << "I need TDO_upper_bound " << i << "," << j << endl;
		}
	bound = TDO_upper_bound_internal(i, j);
	if (bound == -1) {
		cout << "TDO_upper_bound = -1 i=" << i << " j=" << j << endl;
		exit(1);
		}
	//cout << "PACKING " << i << " " << j << " = " << bound << endl;
	return TDO_upper_bound_internal(i, j);
}

int &geometry_global::TDO_upper_bound_internal(int i, int j)
{
	if (i > TDO_upper_bounds_v_max) {
		cout << "TDO_upper_bound i > v_max" << endl;
		cout << "i=" << i << endl;
		cout << "TDO_upper_bounds_v_max=" << TDO_upper_bounds_v_max << endl;
		exit(1);
		}
	if (i <= 0) {
		cout << "TDO_upper_bound_internal i <= 0, i = " << i << endl;
		exit(1);
		}
	if (j <= 0) {
		cout << "TDO_upper_bound_internal j <= 0, j = " << j << endl;
		exit(1);
		}
	return TDO_upper_bounds_table[(i - 1) * TDO_upper_bounds_v_max + j - 1];
}

int &geometry_global::TDO_upper_bound_source(int i, int j)
{
	if (i > TDO_upper_bounds_v_max) {
		cout << "TDO_upper_bound_source i > v_max" << endl;
		cout << "i=" << i << endl;
		cout << "TDO_upper_bounds_v_max=" << TDO_upper_bounds_v_max << endl;
		exit(1);
		}
	if (i <= 0) {
		cout << "TDO_upper_bound_source i <= 0, i = " << i << endl;
		exit(1);
		}
	if (j <= 0) {
		cout << "TDO_upper_bound_source j <= 0, j = " << j << endl;
		exit(1);
		}
	return TDO_upper_bounds_table_source[(i - 1) * TDO_upper_bounds_v_max + j - 1];
}

int geometry_global::braun_test_single_type(int v, int k, int ak)
{
	int i, l, s, m;

	i = 0;
	s = 0;
	for (l = 1; l <= ak; l++) {
		m = MAXIMUM(k - i, 0);
		s += m;
		if (s > v)
			return FALSE;
		i++;
		}
	return TRUE;
}

int geometry_global::braun_test_upper_bound(int v, int k)
{
	int n, bound, v2, k2;

	//cout << "braun_test_upper_bound v=" << v << " k=" << k << endl;
	if (k == 1) {
		bound = INT_MAX;
		}
	else if (k == 2) {
		bound = ((v * (v - 1)) >> 1);
		}
	else {
		v2 = (v * (v - 1)) >> 1;
		k2 = (k * (k - 1)) >> 1;
		for (n = 1; ; n++) {
			if (braun_test_single_type(v, k, n) == FALSE) {
				bound = n - 1;
				break;
				}
			if (n * k2 > v2) {
				bound = n - 1;
				break;
				}
			}
		}
	//cout << "braun_test_upper_bound v=" << v << " k=" << k
	//<< " bound=" << bound << endl;
	return bound;
}

void geometry_global::TDO_refine_init_upper_bounds(int v_max)
{
	int i, j, bound, bound_braun, bound_maxfit, u;
	//cout << "TDO_refine_init_upper_bounds v_max=" << v_max << endl;

	TDO_upper_bounds_table = NEW_int(v_max * v_max);
	TDO_upper_bounds_table_source = NEW_int(v_max * v_max);
	TDO_upper_bounds_v_max = v_max;
	for (i = 0; i < v_max * v_max; i++) {
		TDO_upper_bounds_table[i] = -1;
		TDO_upper_bounds_table_source[i] = 0;
		}
	for (u = 0;; u++) {
		if (TDO_upper_bounds_initial_data[u * 3 + 0] == -1)
			break;
		i = TDO_upper_bounds_initial_data[u * 3 + 0];
		j = TDO_upper_bounds_initial_data[u * 3 + 1];
		bound = TDO_upper_bounds_initial_data[u * 3 + 2];
		bound_braun = braun_test_upper_bound(i, j);
		if (bound < bound_braun) {
			//cout << "i=" << i << " j=" << j << " bound=" << bound
			//<< " bound_braun=" << bound_braun << endl;
			}
		TDO_upper_bound_internal(i, j) = bound;
		TDO_upper_bound_source(i, j) = 1;
		}
	for (i = 1; i <= v_max; i++) {
		for (j = 1; j <= i; j++) {
			if (TDO_upper_bound_internal(i, j) != -1)
				continue;
			bound_braun = braun_test_upper_bound(i, j);
			TDO_upper_bound_internal(i, j) = bound_braun;
			TDO_upper_bound_source(i, j) = 2;
			bound_maxfit = packing_number_via_maxfit(i, j);
			if (bound_maxfit < bound_braun) {
				//cout << "i=" << i << " j=" << j
				//<< " bound_braun=" << bound_braun
				//	<< " bound_maxfit=" << bound_maxfit << endl;
				TDO_upper_bound_internal(i, j) = bound_maxfit;
				TDO_upper_bound_source(i, j) = 3;
				}
			}
		}
	//print_integer_matrix_width(cout,
	//TDO_upper_bounds_table, v_max, v_max, v_max, 3);
	//print_integer_matrix_width(cout,
	//TDO_upper_bounds_table_source, v_max, v_max, v_max, 3);
}

void geometry_global::TDO_refine_extend_upper_bounds(int new_v_max)
{
	int *new_upper_bounds;
	int *new_upper_bounds_source;
	int i, j, bound, bound_braun, bound_maxfit, src;
	int v_max;

	//cout << "TDO_refine_extend_upper_bounds
	//new_v_max=" << new_v_max << endl;
	v_max = TDO_upper_bounds_v_max;
	new_upper_bounds = NEW_int(new_v_max * new_v_max);
	new_upper_bounds_source = NEW_int(new_v_max * new_v_max);
	for (i = 0; i < new_v_max * new_v_max; i++) {
		new_upper_bounds[i] = -1;
		new_upper_bounds_source[i] = 0;
		}
	for (i = 1; i <= v_max; i++) {
		for (j = 1; j <= v_max; j++) {
			bound = TDO_upper_bound_internal(i, j);
			src = TDO_upper_bound_source(i, j);
			new_upper_bounds[(i - 1) * new_v_max + (j - 1)] = bound;
			new_upper_bounds_source[(i - 1) * new_v_max + (j - 1)] = src;
			}
		}
	FREE_int(TDO_upper_bounds_table);
	FREE_int(TDO_upper_bounds_table_source);
	TDO_upper_bounds_table = new_upper_bounds;
	TDO_upper_bounds_table_source = new_upper_bounds_source;
	TDO_upper_bounds_v_max = new_v_max;
	for (i = v_max + 1; i <= new_v_max; i++) {
		for (j = 1; j <= i; j++) {
			bound_braun = braun_test_upper_bound(i, j);
			TDO_upper_bound_internal(i, j) = bound_braun;
			TDO_upper_bound_source(i, j) = 2;
			bound_maxfit = packing_number_via_maxfit(i, j);
			if (bound_maxfit < bound_braun) {
				//cout << "i=" << i << " j=" << j
				//<< " bound_braun=" << bound_braun
				//	<< " bound_maxfit=" << bound_maxfit << endl;
				TDO_upper_bound_internal(i, j) = bound_maxfit;
				TDO_upper_bound_source(i, j) = 3;
				}
			}
		}
	//print_integer_matrix_width(cout,
	//TDO_upper_bounds_table, new_v_max, new_v_max, new_v_max, 3);
	//print_integer_matrix_width(cout,
	//TDO_upper_bounds_table_source, new_v_max, new_v_max, new_v_max, 3);

}

int geometry_global::braun_test_on_line_type(int v, int *type)
{
	int i, k, ak, l, s, m;

	i = 0;
	s = 0;
	for (k = v; k >= 2; k--) {
		ak = type[k];
		for (l = 0; l < ak; l++) {
			m = MAXIMUM(k - i, 0);
			s += m;
			if (s > v) {
				return FALSE;
				}
			i++;
			}
		}
	return TRUE;
}

static int maxfit_table_v_max = -1;
static int *maxfit_table = NULL;

int &geometry_global::maxfit(int i, int j)
{
	int m;

	m = MAXIMUM(i, j);
	if (maxfit_table_v_max == -1) {
		maxfit_table_init(2 * m);
		}
	if (m > maxfit_table_v_max) {
		maxfit_table_reallocate(2 * m);
		}
	return maxfit_internal(i, j);
}

int &geometry_global::maxfit_internal(int i, int j)
{
	if (i > maxfit_table_v_max) {
		cout << "maxfit_table_v_max i > v_max" << endl;
		cout << "i=" << i << endl;
		cout << "maxfit_table_v_max=" << maxfit_table_v_max << endl;
		exit(1);
		}
	if (j > maxfit_table_v_max) {
		cout << "maxfit_table_v_max j > v_max" << endl;
		cout << "j=" << j << endl;
		cout << "maxfit_table_v_max=" << maxfit_table_v_max << endl;
		exit(1);
		}
	if (i <= 0) {
		cout << "maxfit_table_v_max i <= 0, i = " << i << endl;
		exit(1);
		}
	if (j <= 0) {
		cout << "maxfit_table_v_max j <= 0, j = " << j << endl;
		exit(1);
		}
	return maxfit_table[(i - 1) * maxfit_table_v_max + j - 1];
}

void geometry_global::maxfit_table_init(int v_max)
{
	//cout << "maxfit_table_init v_max=" << v_max << endl;

	maxfit_table = NEW_int(v_max * v_max);
	maxfit_table_v_max = v_max;
	maxfit_table_compute();
	//print_integer_matrix_width(cout, maxfit_table, v_max, v_max, v_max, 3);
}

void geometry_global::maxfit_table_reallocate(int v_max)
{
	cout << "maxfit_table_reallocate v_max=" << v_max << endl;

	FREE_int(maxfit_table);
	maxfit_table = NEW_int(v_max * v_max);
	maxfit_table_v_max = v_max;
	maxfit_table_compute();
	//print_integer_matrix_width(cout, maxfit_table, v_max, v_max, v_max, 3);
}

#define Choose2(x)   ((x*(x-1))/2)

void geometry_global::maxfit_table_compute()
{
	int M = maxfit_table_v_max;
	int *matrix = maxfit_table;
	int m, i, j, inz, gki;

	//cout << "computing maxfit table v_max=" << maxfit_table_v_max << endl;
	for (i=0; i<M*M; i++) {
		matrix[i] = 0;
		}
	m = 0;
	for (i=1; i<=M; i++) {
		//cout << "i=" << i << endl;
		inz = i;
		j = 1;
		while (i>=j) {
			gki = inz/i;
			if (j*(j-1)/2 < i*Choose2(gki)+(inz % i)*gki) {
				j++;
				}
			if (j<=M) {
				//cout << "j=" << j << " inz=" << inz << endl;
				m = MAXIMUM(m, inz);
				matrix[(j-1) * M + i-1]=inz;
				matrix[(i-1) * M + j-1]=inz;
				}
			inz++;
			}
		//print_integer_matrix_width(cout, matrix, M, M, M, 3);
		} // next i
}

int geometry_global::packing_number_via_maxfit(int n, int k)
{
	int m;

	if (k == 1) {
		return INT_MAX;
		}
	//cout << "packing_number_via_maxfit n=" << n << " k=" << k << endl;
	m=1;
	while (maxfit(n, m) >= m*k) {
		m++;
		}
	//cout << "P(" << n << "," << k << ")=" << m - 1 << endl;
	return m - 1;
}



}}

