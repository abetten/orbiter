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
	long int a, b;

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

long int geometry_global::nb_affine_lines(int n, int q)
{
	number_theory_domain NT;
	long int qnp1, qn, q2, a, b, denom, res;

	qnp1 = NT.i_power_j_lint(q, n + 1);
	qn = NT.i_power_j_lint(q, n);
	q2 = q * q;
	denom = (q2 - 1) * (q2 - q);
	a = (qnp1 - 1) * (qnp1 - q) / denom;
	b = (qn - 1) * (qn - q) / denom;
	res = a - b;
	return res;
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
	int i;
	long int b;

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


int geometry_global::AG_element_next(int q, int *v, int stride, int len)
{
	int i;

#if 1
	if (len <= 0) {
		cout << "geometry_global::AG_element_next len <= 0" << endl;
		exit(1);
		}
#endif
	for (i = len - 1; i >= 0; i--) {
		if (v[i * stride] < q - 1) {
			v[i * stride]++;
			return TRUE;
		}
		else {
			v[i * stride] = 0;
		}
	}
	return FALSE;
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
	a.create(0, __FILE__, __LINE__);
	Q.create(q, __FILE__, __LINE__);
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

	F.finite_field_init(q, verbose_level);

	cout << "all elements of PG_" << n << "(" << q << ")" << endl;
	F.display_all_PG_elements(n);

	for (m = 0; m < n; m++) {
		cout << "all elements of PG_" << n << "(" << q << "), "
			"not in a subspace of dimension " << m << endl;
		F.display_all_PG_elements_not_in_subspace(n, m);
	}

}

void geometry_global::create_Fisher_BLT_set(long int *Fisher_BLT,
		int q, std::string &poly_q, std::string &poly_Q, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	unusual_model U;

	U.setup(q, poly_q, poly_Q, verbose_level);
	U.create_Fisher_BLT_set(Fisher_BLT, verbose_level);

}

void geometry_global::create_Linear_BLT_set(long int *BLT, int q,
		std::string &poly_q, std::string &poly_Q, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	unusual_model U;

	U.setup(q, poly_q, poly_Q, verbose_level);
	U.create_Linear_BLT_set(BLT, verbose_level);

}

void geometry_global::create_Mondello_BLT_set(long int *BLT, int q,
		std::string &poly_q, std::string &poly_Q, int verbose_level)
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


void geometry_global::determine_conic(int q, std::string &override_poly,
		long int *input_pts, int nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	finite_field F;
	projective_space *P;
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
	std::string &fname, int &nb_pts, long int *&Pts,
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
	Pts = NEW_lint(nb_pts);
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



	string name;

	fname.assign("unital_");
	BM->get_name(name);
	fname.append(name);

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
	int stride = 1, /*n,*/ len; //, h, wt;
	long int i, j, a, nb;
	int c1 = 0, c2 = 0, c3 = 0;
	int verbose_level = 0;

	cout << "test_Orthogonal" << endl;
	GFq.finite_field_init(q, verbose_level);
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
	long int i, j, a;
	int stride = 1;
	long int nb;
	int verbose_level = 0;

	cout << "test_orthogonal" << endl;
	GFq.finite_field_init(q, verbose_level);
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



#if 0
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
	O->finite_field_init(epsilon, n + 1, Fq, verbose_level - 1);
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
#endif


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



void geometry_global::do_inverse_isomorphism_klein_quadric(finite_field *F,
		std::string &inverse_isomorphism_klein_quadric_matrix_A6,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_inverse_isomorphism_klein_quadric" << endl;
	}


	int *A6;
	int sz;

	int_vec_scan(inverse_isomorphism_klein_quadric_matrix_A6.c_str(), A6, sz);
	if (sz != 36) {
		cout << "geometry_global::do_inverse_isomorphism_klein_quadric "
				"The input matrix must be of size 6x6" << endl;
		exit(1);
	}


	cout << "A6:" << endl;
	int_matrix_print(A6, 6, 6);

	klein_correspondence *Klein;
	orthogonal *O;


	Klein = NEW_OBJECT(klein_correspondence);
	O = NEW_OBJECT(orthogonal);

	O->init(1 /* epsilon */, 6, F, 0 /* verbose_level*/);
	Klein->init(F, O, 0 /* verbose_level */);

	int A4[16];
	Klein->reverse_isomorphism(A6, A4, verbose_level);

	cout << "A4:" << endl;
	int_matrix_print(A4, 4, 4);

	FREE_OBJECT(Klein);
	FREE_OBJECT(O);

	if (f_v) {
		cout << "geometry_global::do_inverse_isomorphism_klein_quadric done" << endl;
	}
}

void geometry_global::do_rank_point_in_PG(finite_field *F, int n,
		std::string &coeff_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_rank_point_in_PG" << endl;
	}

	int *coeff;
	int sz;

	int_vec_scan(coeff_text, coeff, sz);
	if (sz != n + 1) {
		cout << "geometry_global::do_rank_point_in_PG sz != n + 1" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "geometry_global::do_rank_point_in_PG coeff: ";
		int_vec_print(cout, coeff, sz);
		cout << endl;
	}

	long int a;

	F->PG_element_rank_modified_lint(coeff, 1, n + 1, a);


	if (f_v) {
		cout << "geometry_global::do_rank_point_in_PG coeff: ";
		int_vec_print(cout, coeff, sz);
		cout << " has rank " << a << endl;
	}

	FREE_int(coeff);

}

void geometry_global::do_rank_point_in_PG_given_as_pairs(finite_field *F, int n,
		std::string &coeff_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_rank_point_in_PG_given_as_pairs" << endl;
	}

	int *coeff;

	{
		int *coeff_pairs;
		int sz, sz2;
		int i, a, b;

		int_vec_scan(coeff_text, coeff_pairs, sz);
		coeff = NEW_int(n + 1);
		int_vec_zero(coeff, n + 1);

		sz2 = sz >> 1;

		for (i = 0; i < sz2; i++) {
			a = coeff_pairs[2 * i + 0];
			b = coeff_pairs[2 * i + 1];
			if (b >= n + 1) {
				cout << "geometry_global::do_rank_point_in_PG_given_as_pairs b >= n + 1" << endl;
				exit(1);
			}
			if (b < 0) {
				cout << "geometry_global::do_rank_point_in_PG_given_as_pairs b < 0" << endl;
				exit(1);
			}
			if (a < 0) {
				cout << "geometry_global::do_rank_point_in_PG_given_as_pairs a < 0" << endl;
				exit(1);
			}
			if (a >= F->q) {
				cout << "geometry_global::do_rank_point_in_PG_given_as_pairs a >= F->q" << endl;
				exit(1);
			}
			coeff[b] = a;
		}
		if (f_v) {
			cout << "geometry_global::do_rank_point_in_PG_given_as_pairs coeff: ";
			int_vec_print(cout, coeff, n + 1);
			cout << endl;
		}
		FREE_int(coeff_pairs);
	}

	long int a;

	F->PG_element_rank_modified_lint(coeff, 1, n + 1, a);


	if (f_v) {
		cout << "geometry_global::do_rank_point_in_PG_given_as_pairs coeff: ";
		int_vec_print(cout, coeff, n + 1);
		cout << " has rank " << a << endl;
	}

	FREE_int(coeff);
}



void geometry_global::do_intersection_of_two_lines(finite_field *F,
		std::string &line_1_basis,
		std::string &line_2_basis,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Line1;
	int *Line2;
	int *A;
	int *B;
	int *C;
	int len, rk, i;

	if (f_v) {
		cout << "geometry_global::do_intersection_of_two_lines" << endl;
	}

	int_vec_scan(line_1_basis, Line1, len);
	if (len != 8) {
		cout << "geometry_global::do_intersection_of_two_lines len != 8" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	int_vec_scan(line_2_basis, Line2, len);
	if (len != 8) {
		cout << "geometry_global::do_intersection_of_two_lines len != 8" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	A = NEW_int(16);
	B = NEW_int(16);
	C = NEW_int(16);

	// Line 1
	int_vec_copy(Line1, A, 8);
	rk = F->perp_standard(4, 2, A, verbose_level);
	if (rk != 2) {
		cout << "geometry_global::do_intersection_of_two_lines rk != 2" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}

	// Line 2
	int_vec_copy(Line2, B, 8);
	rk = F->perp_standard(4, 2, B, verbose_level);
	if (rk != 2) {
		cout << "geometry_global::do_intersection_of_two_lines rk != 2" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}


	int_vec_copy(A + 8, C, 8);
	int_vec_copy(B + 8, C + 8, 8);
	rk = F->perp_standard(4, 4, C, verbose_level);
	if (rk != 3) {
		cout << "geometry_global::do_intersection_of_two_lines rk != 3" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}

	if (f_normalize_from_the_left) {
		cout << "geometry_global::do_intersection_of_two_lines normalizing from the left" << endl;
		for (i = 3; i < 4; i++) {
			F->PG_element_normalize_from_front(
					C + i * 4, 1, 4);
		}

		cout << "geometry_global::do_intersection_of_two_lines after normalize from the left:" << endl;
		int_matrix_print(C + 12, 1, 4);
		cout << "rk=" << rk << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "geometry_global::do_intersection_of_two_lines normalizing from the right" << endl;
		for (i = 3; i < 4; i++) {
			F->PG_element_normalize(
					C + i * 4, 1, 4);
		}

		cout << "geometry_global::do_intersection_of_two_lines after normalize from the right:" << endl;
		int_matrix_print(C + 12, 1, 4);
		cout << "rk=" << rk << endl;

	}


	FREE_int(Line1);
	FREE_int(Line2);
	FREE_int(A);
	FREE_int(B);
	FREE_int(C);

	if (f_v) {
		cout << "geometry_global::do_intersection_of_two_lines done" << endl;
	}

}

void geometry_global::do_transversal(finite_field *F,
		std::string &line_1_basis,
		std::string &line_2_basis,
		std::string &point,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Line1;
	int *Line2;
	int *Pt;
	int *A;
	int *B;
	int len, rk, i;

	if (f_v) {
		cout << "geometry_global::do_transversal" << endl;
	}

	int_vec_scan(line_1_basis, Line1, len);
	if (len != 8) {
		cout << "geometry_global::do_transversal len != 8" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	int_vec_scan(line_2_basis, Line2, len);
	if (len != 8) {
		cout << "geometry_global::do_transversal len != 8" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	int_vec_scan(point, Pt, len);
	if (len != 4) {
		cout << "geometry_global::do_transversal len != 4" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	A = NEW_int(16);
	B = NEW_int(16);

	// Line 1
	int_vec_copy(Line1, A, 8);
	int_vec_copy(Pt, A + 8, 4);
	rk = F->perp_standard(4, 3, A, verbose_level);
	if (rk != 3) {
		cout << "geometry_global::do_transversal rk != 3" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}
	int_vec_copy(A + 12, B, 4);

	// Line 2
	int_vec_copy(Line2, A, 8);
	int_vec_copy(Pt, A + 8, 4);
	rk = F->perp_standard(4, 3, A, verbose_level);
	if (rk != 3) {
		cout << "geometry_global::do_transversal rk != 3" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}
	int_vec_copy(A + 12, B + 4, 4);

	// B
	rk = F->perp_standard(4, 2, B, verbose_level);
	if (rk != 2) {
		cout << "geometry_global::do_transversal rk != 2" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}
	if (f_normalize_from_the_left) {
		cout << "geometry_global::do_transversal normalizing from the left" << endl;
		for (i = 2; i < 4; i++) {
			F->PG_element_normalize_from_front(
					B + i * 4, 1, 4);
		}

		cout << "geometry_global::do_transversal after normalize from the left:" << endl;
		int_matrix_print(B + 8, 2, 4);
		cout << "rk=" << rk << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "geometry_global::do_transversal normalizing from the right" << endl;
		for (i = 2; i < 4; i++) {
			F->PG_element_normalize(
					B + i * 4, 1, 4);
		}

		cout << "geometry_global::do_transversal after normalize from the right:" << endl;
		int_matrix_print(B + 8, 2, 4);
		cout << "rk=" << rk << endl;

	}


	FREE_int(Line1);
	FREE_int(Line2);
	FREE_int(Pt);
	FREE_int(A);
	FREE_int(B);

	if (f_v) {
		cout << "geometry_global::do_transversal done" << endl;
	}
}


void geometry_global::do_move_two_lines_in_hyperplane_stabilizer(
		finite_field *F,
		long int line1_from, long int line2_from,
		long int line1_to, long int line2_to, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_move_two_lines_in_hyperplane_stabilizer" << endl;
	}

	projective_space *P;
	int A4[16];

	P = NEW_OBJECT(projective_space);
	P->init(3, F,
			FALSE /* f_init_incidence_structure */,
			0 /*verbose_level*/);
	P->hyperplane_lifting_with_two_lines_moved(
			line1_from, line1_to,
			line2_from, line2_to,
			A4,
			verbose_level);

	cout << "geometry_global::do_move_two_lines_in_hyperplane_stabilizer A4=" << endl;
	int_matrix_print(A4, 4, 4);

	if (f_v) {
		cout << "geometry_global::do_move_two_lines_in_hyperplane_stabilizer done" << endl;
	}
}

void geometry_global::do_move_two_lines_in_hyperplane_stabilizer_text(
		finite_field *F,
		std::string line1_from_text, std::string line2_from_text,
		std::string line1_to_text, std::string line2_to_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_move_two_lines_in_hyperplane_stabilizer_text" << endl;
	}

	projective_space *P;
	int A4[16];

	P = NEW_OBJECT(projective_space);
	P->init(3, F,
			FALSE /* f_init_incidence_structure */,
			0 /*verbose_level*/);

	int *line1_from_data;
	int *line2_from_data;
	int *line1_to_data;
	int *line2_to_data;
	int sz;

	int_vec_scan(line1_from_text.c_str(), line1_from_data, sz);
	if (sz != 8) {
		cout << "line1_from_text must contain exactly 8 integers" << endl;
		exit(1);
	}
	int_vec_scan(line2_from_text.c_str(), line2_from_data, sz);
	if (sz != 8) {
		cout << "line2_from_text must contain exactly 8 integers" << endl;
		exit(1);
	}
	int_vec_scan(line1_to_text.c_str(), line1_to_data, sz);
	if (sz != 8) {
		cout << "line1_to_text must contain exactly 8 integers" << endl;
		exit(1);
	}
	int_vec_scan(line2_to_text.c_str(), line2_to_data, sz);
	if (sz != 8) {
		cout << "line2_to_text must contain exactly 8 integers" << endl;
		exit(1);
	}

	long int line1_from;
	long int line2_from;
	long int line1_to;
	long int line2_to;

	line1_from = P->rank_line(line1_from_data);
	line2_from = P->rank_line(line2_from_data);
	line1_to = P->rank_line(line1_to_data);
	line2_to = P->rank_line(line2_to_data);


	P->hyperplane_lifting_with_two_lines_moved(
			line1_from, line1_to,
			line2_from, line2_to,
			A4,
			verbose_level);

	cout << "geometry_global::do_move_two_lines_in_hyperplane_stabilizer_text A4=" << endl;
	int_matrix_print(A4, 4, 4);

	if (f_v) {
		cout << "geometry_global::do_move_two_lines_in_hyperplane_stabilizer_text done" << endl;
	}
}

void geometry_global::Walsh_matrix(finite_field *F, int n, int *W, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Q;
	int *v;
	int *w;
	int i, j, a;

	if (f_v) {
		cout << "geometry_global::Walsh_matrix" << endl;
	}
	v = NEW_int(n);
	w = NEW_int(n);
	Q = 1 << n;
	for (i = 0; i < Q; i++) {
		AG_element_unrank(2, v, 1, n, i);
		for (j = 0; j < Q; j++) {
			AG_element_unrank(2, w, 1, n, j);
			a = F->dot_product(n, v, w);
			if (a) {
				W[i * Q + j] = -1;
			}
			else {
				W[i * Q + j] = 1;
			}
		}
	}
	FREE_int(v);
	FREE_int(w);
	if (f_v) {
		cout << "geometry_global::Walsh_matrix done" << endl;
	}
}

void geometry_global::do_cheat_sheet_PG(finite_field *F,
		layered_graph_draw_options *O,
		int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_PG verbose_level="
				<< verbose_level << endl;
	}


	projective_space *P;

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_PG before P->init" << endl;
	}
	P->init(n, F,
		TRUE /*f_init_incidence_structure*/,
		0 /* verbose_level */);
	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_PG after P->init" << endl;
	}

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_PG before P->create_latex_report" << endl;
	}
	P->create_latex_report(O, verbose_level);
	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_PG after P->create_latex_report" << endl;
	}



	FREE_OBJECT(P);

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_PG done" << endl;
	}

}

void geometry_global::do_cheat_sheet_Gr(finite_field *F,
		int n, int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_Gr verbose_level="
				<< verbose_level << endl;
	}


	projective_space *P;

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_Gr before P->init" << endl;
	}
	P->init(n - 1, F,
		TRUE /*f_init_incidence_structure*/,
		0 /* verbose_level */);
	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_Gr after P->init" << endl;
	}

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_Gr before P->create_latex_report_for_Grassmannian" << endl;
	}
	P->create_latex_report_for_Grassmannian(k, verbose_level);
	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_Gr after P->create_latex_report_for_Grassmannian" << endl;
	}



	FREE_OBJECT(P);

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_PG done" << endl;
	}

}

#if 0
void geometry_global::do_cheat_sheet_orthogonal(finite_field *F,
		int epsilon, int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_orthogonal verbose_level="
				<< verbose_level << endl;
	}


	orthogonal *O;

	O = NEW_OBJECT(orthogonal);





	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_orthogonal before O->init" << endl;
	}
	O->init(epsilon, n, F, verbose_level);
	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_orthogonal after O->init" << endl;
	}




	if (F->q == 64 && n == 6) {

		long int p1 = 15447347;
		long int p2 = 15225451;
		long int rk;

		cout << "p1 = " << p1 << endl;
		cout << "p2 = " << p2 << endl;
		rk = O->rank_line(p1, p2, verbose_level);
		cout << "rk = " << rk << endl;
	}



	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_orthogonal before O->create_latex_report" << endl;
	}
	O->create_latex_report(verbose_level);
	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_orthogonal after O->create_latex_report" << endl;
	}


	FREE_OBJECT(O);

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_orthogonal done" << endl;
	}

}
#endif

void geometry_global::do_cheat_sheet_hermitian(finite_field *F,
		int projective_dimension,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_hermitian verbose_level="
				<< verbose_level << endl;
	}

	hermitian *H;

	H = NEW_OBJECT(hermitian);

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_hermitian before H->init" << endl;
	}
	H->init(F, projective_dimension + 1, verbose_level - 2);
	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_hermitian after H->init" << endl;
	}

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_hermitian before H->create_latex_report" << endl;
	}
	H->create_latex_report(verbose_level);
	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_hermitian after H->create_latex_report" << endl;
	}



	FREE_OBJECT(H);

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_hermitian done" << endl;
	}

}

void geometry_global::do_create_desarguesian_spread(finite_field *FQ, finite_field *Fq,
		int m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "geometry_global::do_create_desarguesian_spread" << endl;
		cout << "geometry_global::do_create_desarguesian_spread Q=" << FQ->q << " q=" << Fq->q << " m=" << m << endl;
	}

	int s, n;

	if (FQ->p != Fq->p) {
		cout << "geometry_global::do_create_desarguesian_spread the fields must have the same characteristic" << endl;
		exit(1);
	}
	s = FQ->e / Fq->e;

	if (s * Fq->e != FQ->e) {
		cout << "geometry_global::do_create_desarguesian_spread Fq is not a subfield of FQ" << endl;
		exit(1);
	}

	n = m * s;
	subfield_structure *SubS;
	desarguesian_spread *D;

	SubS = NEW_OBJECT(subfield_structure);
	if (f_v) {
		cout << "geometry_global::do_create_desarguesian_spread before SubS->init" << endl;
	}
	SubS->init(FQ, Fq, verbose_level - 2);

	if (f_v) {
		cout << "Field-basis: ";
		int_vec_print(cout, SubS->Basis, s);
		cout << endl;
	}

	D = NEW_OBJECT(desarguesian_spread);
	if (f_v) {
		cout << "geometry_global::do_create_desarguesian_spread before D->init" << endl;
	}
	D->init(n, m, s,
		SubS,
		verbose_level - 2);
	if (f_v) {
		cout << "geometry_global::do_create_desarguesian_spread after D->init" << endl;
	}


	D->create_latex_report(verbose_level);

	FREE_OBJECT(D);
	FREE_OBJECT(SubS);

	if (f_v) {
		cout << "geometry_global::do_create_desarguesian_spread done" << endl;
	}
}



}}

