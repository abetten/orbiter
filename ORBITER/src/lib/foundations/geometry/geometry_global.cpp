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


int geometry_global::nb_PG_elements(int n, int q)
// $\frac{q^{n+1} - 1}{q-1} = \sum_{i=0}^{n} q^i $
{
	int qhl, l, deg;

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

int geometry_global::nb_PG_elements_not_in_subspace(int n, int m, int q)
// |PG_n(q)| - |PG_m(q)|
{
	int a, b;

	a = nb_PG_elements(n, q);
	b = nb_PG_elements(m, q);
	return a - b;
}

int geometry_global::nb_AG_elements(int n, int q)
// $q^n$
{
	number_theory_domain NT;

	return NT.i_power_j(q, n);
}

void geometry_global::AG_element_rank(int q, int *v, int stride, int len, int &a)
{
	int i;

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
}

void geometry_global::AG_element_unrank(int q, int *v, int stride, int len, int a)
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
	longinteger_object Q, a1;

	if (len <= 0) {
		cout << "geometry_global::AG_element_unrank_longinteger len <= 0" << endl;
		exit(1);
		}
	for (i = 0; i < len; i++) {
		D.integral_division_by_int(a, q, a1, r);
		//r = a % q;
		v[i * stride] = r;
		//a /= q;
		a.swap_with(a1);
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
		int *input_pts, int nb_pts, int verbose_level)
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
		int_vec_print(cout, input_pts, nb_pts);
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

	int points[1000];
	int nb_points;
	//int v[3];

	P->conic_points(input_pts, six_coeffs,
			points, nb_points, verbose_level - 2);
	if (f_v) {
		cout << "the " << nb_points << " conic points are: ";
		int_vec_print(cout, points, nb_points);
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
		Sorting.int_vec_sort(3, subset1);
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
		// in TOP_LEVEL/buekenhout_metz.C

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


int geometry_global::count_Sbar(int n, int q)
{
	return count_T1(1, n, q);
}

int geometry_global::count_S(int n, int q)
{
	return (q - 1) * count_Sbar(n, q) + 1;
}

int geometry_global::count_N1(int n, int q)
{
	if (n <= 0) {
		return 0;
		}
	return nb_pts_N1(n, q);
}

int geometry_global::count_T1(int epsilon, int n, int q)
// n = Witt index
{
	number_theory_domain NT;

	if (n < 0) {
		//cout << "count_T1 n is negative. n=" << n << endl;
		return 0;
		}
	if (epsilon == 1) {
		return ((NT.i_power_j(q, n) - 1) *
				(NT.i_power_j(q, n - 1) + 1)) / (q - 1);
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

int geometry_global::count_T2(int n, int q)
{
	number_theory_domain NT;

	if (n <= 0) {
		return 0;
		}
	return (NT.i_power_j(q, 2 * n - 2) - 1) *
			(NT.i_power_j(q, n) - 1) *
			(NT.i_power_j(q, n - 2) + 1) / ((q - 1) * (NT.i_power_j(q, 2) - 1));
}

int geometry_global::nb_pts_Qepsilon(int epsilon, int k, int q)
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

int geometry_global::nb_pts_Q(int k, int q)
// number of singular points on Q(k,q), parabolic quadric, so k is even
{
	int n;

	n = Witt_index(0, k);
	return nb_pts_Sbar(n, q) + nb_pts_N1(n, q);
}

int geometry_global::nb_pts_Qplus(int k, int q)
// number of singular points on Q^+(k,q)
{
	int n;

	n = Witt_index(1, k);
	return nb_pts_Sbar(n, q);
}

int geometry_global::nb_pts_Qminus(int k, int q)
// number of singular points on Q^-(k,q)
{
	int n;

	n = Witt_index(-1, k);
	return nb_pts_Sbar(n, q) + (q + 1) * nb_pts_N1(n, q);
}


// #############################################################################
// the following functions are for the hyperbolic quadric with Witt index n:
// #############################################################################

int geometry_global::nb_pts_S(int n, int q)
// Number of singular vectors (including the zero vector)
{
	int a;

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

int geometry_global::nb_pts_N(int n, int q)
// Number of non-singular vectors.
// Of course, |N(n,q)| + |S(n,q)| = q^{2n}
// |N(n,q)| = (q - 1) * |N1(n,q)|
{
	int a;

	if (n <= 0) {
		cout << "nb_pts_N n <= 0" << endl;
		exit(1);
		}
	if (n == 1) {
		return (q - 1) * (q - 1);
		}
	a = nb_pts_S(1, q) * nb_pts_N(n - 1, q);
	a += nb_pts_N(1, q) * nb_pts_S(n - 1, q);
	a += nb_pts_N(1, q) * (q - 2) * nb_pts_N1(n - 1, q);
	return a;
}

int geometry_global::nb_pts_N1(int n, int q)
// Number of non-singular vectors
// for one fixed value of the quadratic form
// i.e. number of solutions of
// \sum_{i=0}^{n-1} x_{2i}x_{2i+1} = s
// for some fixed s \neq 0.
{
	int a;

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

int geometry_global::nb_pts_Sbar(int n, int q)
// number of singular projective points
// |S(n,q)| = (q-1) * |Sbar(n,q)| + 1
{
	int a;

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
	return a;
}

int geometry_global::nb_pts_Nbar(int n, int q)
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
		GFq.Q_epsilon_unrank(v, stride, epsilon, k, c1, c2, c3, i);

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
		j = GFq.Q_epsilon_rank(v, stride, epsilon, k, c1, c2, c3);
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
		GFq.Sbar_unrank(v, stride, n, i);
		cout << i << " : ";
		int_set_print(v, 2 * n);
		cout << " : ";
		a = GFq.evaluate_hyperbolic_quadratic_form(v, stride, n);
		cout << a;
		GFq.Sbar_rank(v, stride, n, j);
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



}}

