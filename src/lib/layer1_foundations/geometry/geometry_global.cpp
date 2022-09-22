/*
 * geometry_global.cpp
 *
 *  Created on: Apr 19, 2019
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {



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
	number_theory::number_theory_domain NT;

	return NT.i_power_j_lint(q, n);
}

long int geometry_global::nb_affine_lines(int n, int q)
{
	number_theory::number_theory_domain NT;
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
		int *v, int stride, int len, ring_theory::longinteger_object &a)
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object Q, a1;
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
		int *v, int stride, int len, ring_theory::longinteger_object &a)
{
	int i, r;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a0, Q, a1;

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
	field_theory::finite_field F;
	int m;
	int verbose_level = 1;

	F.finite_field_init(q, FALSE /* f_without_tables */, verbose_level);

	cout << "all elements of PG_" << n << "(" << q << ")" << endl;
	F.display_all_PG_elements(n);

	for (m = 0; m < n; m++) {
		cout << "all elements of PG_" << n << "(" << q << "), "
			"not in a subspace of dimension " << m << endl;
		F.display_all_PG_elements_not_in_subspace(n, m);
	}

}

void geometry_global::create_Fisher_BLT_set(long int *Fisher_BLT, int *ABC,
		field_theory::finite_field *FQ,
		field_theory::finite_field *Fq, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::create_Fisher_BLT_set" << endl;
	}
	orthogonal_geometry::unusual_model U;

	U.setup(FQ, Fq, verbose_level);
	U.create_Fisher_BLT_set(Fisher_BLT, ABC, verbose_level);
	if (f_v) {
		cout << "geometry_global::create_Fisher_BLT_set done" << endl;
	}

}

void geometry_global::create_Linear_BLT_set(long int *BLT, int *ABC,
		field_theory::finite_field *FQ,
		field_theory::finite_field *Fq, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::create_Linear_BLT_set" << endl;
	}
	orthogonal_geometry::unusual_model U;

	U.setup(FQ, Fq, verbose_level);
	U.create_Linear_BLT_set(BLT, ABC, verbose_level);
	if (f_v) {
		cout << "geometry_global::create_Linear_BLT_set done" << endl;
	}

}

void geometry_global::create_Mondello_BLT_set(long int *BLT, int *ABC,
		field_theory::finite_field *FQ,
		field_theory::finite_field *Fq, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::create_Mondello_BLT_set" << endl;
	}
	orthogonal_geometry::unusual_model U;

	U.setup(FQ, Fq, verbose_level);
	U.create_Mondello_BLT_set(BLT, ABC, verbose_level);
	if (f_v) {
		cout << "geometry_global::create_Mondello_BLT_set done" << endl;
	}

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
	int n, field_theory::finite_field &F,
	int nb_terms, int *form_i, int *form_j, int *form_coeff, int *Gram)
{
	int k, i, j, c;

	Int_vec_zero(Gram, n * n);
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

void geometry_global::add_term(int n,
		field_theory::finite_field &F,
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


#if 0
void geometry_global::determine_conic(int q, std::string &override_poly,
		long int *input_pts, int nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	field_theory::finite_field F;
	projective_space *P;
	int v[3];
	int len = 3;
	int six_coeffs[6];
	int i;

	if (f_v) {
		cout << "determine_conic q=" << q << endl;
		cout << "input_pts: ";
		Lint_vec_print(cout, input_pts, nb_pts);
		cout << endl;
		}
	F.init_override_polynomial(q, override_poly, FALSE /* f_without_tables */, verbose_level);

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
		Int_vec_print(cout, six_coeffs, 6);
		cout << endl;
		}

	long int points[1000];
	int nb_points;
	//int v[3];

	P->conic_points(input_pts, six_coeffs,
			points, nb_points, verbose_level - 2);
	if (f_v) {
		cout << "the " << nb_points << " conic points are: ";
		Lint_vec_print(cout, points, nb_points);
		cout << endl;
		for (i = 0; i < nb_points; i++) {
			P->unrank_point(v, points[i]);
			cout << i << " : " << points[i] << " : ";
			Int_vec_print(cout, v, 3);
			cout << endl;
			}
		}
	FREE_OBJECT(P);
}
#endif

int geometry_global::test_if_arc(field_theory::finite_field *Fq,
		int *pt_coords,
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
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;


	if (f_v) {
		cout << "test_if_arc testing set" << endl;
		Int_vec_print(cout, set, set_sz);
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
			Int_vec_print(cout, subset1, 3);
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
			Int_vec_print_integer_matrix_width(cout, Mtx, 3, k, k, 1);
			}
		rk = Fq->Linear_algebra->Gauss_easy(Mtx, 3, k);
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
		field_theory::finite_field *Fq,
		field_theory::finite_field *FQ,
	int f_classical, int f_Uab, int parameter_a, int parameter_b,
	std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, rk, d = 3;
	int v[3];
	buekenhout_metz *BM;

	if (f_v) {
		cout << "geometry_global::create_Buekenhout_Metz" << endl;
	}


	BM = NEW_OBJECT(buekenhout_metz);

	BM->buekenhout_metz_init(Fq, FQ,
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
			Int_vec_print(cout, v, d);
			cout << " : " << setw(5) << rk << endl;
			}
		}



	string name;

	fname.assign("unital_");
	BM->get_name(name);
	fname.append(name);

	FREE_OBJECT(BM);
	if (f_v) {
		cout << "geometry_global::create_Buekenhout_Metz done" << endl;
	}

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
	number_theory::number_theory_domain NT;

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
	number_theory::number_theory_domain NT;

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
	field_theory::finite_field GFq;
	int *v;
	int stride = 1, /*n,*/ len; //, h, wt;
	long int i, j, a, nb;
	int c1 = 0, c2 = 0, c3 = 0;
	int verbose_level = 0;

	cout << "test_Orthogonal" << endl;
	GFq.finite_field_init(q, FALSE /* f_without_tables */, verbose_level);
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
		GFq.Linear_algebra->choose_anisotropic_form(c1, c2, c3, TRUE);
		}
	for (i = 0; i < nb; i++) {
		GFq.Orthogonal_indexing->Q_epsilon_unrank(v, stride, epsilon, k, c1, c2, c3, i, 0 /* verbose_level */);

#if 0
		wt = 0;
		for (h = 0; h < len; h++) {
			if (v[h])
				wt++;
			}
#endif
		cout << i << " : ";
		Int_vec_print(cout, v, len);
		cout << " : ";
		a = GFq.Linear_algebra->evaluate_quadratic_form(v, stride, epsilon, k,
				c1, c2, c3);
		cout << a;
		j = GFq.Orthogonal_indexing->Q_epsilon_rank(v, stride, epsilon, k, c1, c2, c3, 0 /* verbose_level */);
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
	field_theory::finite_field GFq;
	long int i, j, a;
	int stride = 1;
	long int nb;
	int verbose_level = 0;

	cout << "test_orthogonal" << endl;
	GFq.finite_field_init(q, FALSE /* f_without_tables */, verbose_level);
	v = NEW_int(2 * n);
	nb = nb_pts_Sbar(n, q);
	cout << "\\Omega^+(" << 2 * n << "," << q << ") has " << nb
			<< " singular points" << endl;
	for (i = 0; i < nb; i++) {
		GFq.Orthogonal_indexing->Sbar_unrank(v, stride, n, i, 0 /* verbose_level */);
		cout << i << " : ";
		orbiter_kernel_system::Orbiter->Int_vec->set_print(cout, v, 2 * n);
		cout << " : ";
		a = GFq.Linear_algebra->evaluate_hyperbolic_quadratic_form(v, stride, n);
		cout << a;
		GFq.Orthogonal_indexing->Sbar_rank(v, stride, n, j, 0 /* verbose_level */);
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



void geometry_global::do_inverse_isomorphism_klein_quadric(
		field_theory::finite_field *F,
		std::string &inverse_isomorphism_klein_quadric_matrix_A6,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_inverse_isomorphism_klein_quadric" << endl;
	}


	int *A6;
	int sz;

	Int_vec_scan(inverse_isomorphism_klein_quadric_matrix_A6.c_str(), A6, sz);
	if (sz != 36) {
		cout << "geometry_global::do_inverse_isomorphism_klein_quadric "
				"The input matrix must be of size 6x6" << endl;
		exit(1);
	}


	cout << "A6:" << endl;
	Int_matrix_print(A6, 6, 6);

	klein_correspondence *Klein;
	orthogonal_geometry::orthogonal *O;


	Klein = NEW_OBJECT(klein_correspondence);
	O = NEW_OBJECT(orthogonal_geometry::orthogonal);

	O->init(1 /* epsilon */, 6, F, 0 /* verbose_level*/);
	Klein->init(F, O, 0 /* verbose_level */);

	int A4[16];
	Klein->reverse_isomorphism(A6, A4, verbose_level);

	cout << "A4:" << endl;
	Int_matrix_print(A4, 4, 4);

	FREE_OBJECT(Klein);
	FREE_OBJECT(O);

	if (f_v) {
		cout << "geometry_global::do_inverse_isomorphism_klein_quadric done" << endl;
	}
}

void geometry_global::do_rank_points_in_PG(
		field_theory::finite_field *F,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_rank_points_in_PG" << endl;
	}

	int *v;
	int m, n;

	orbiter_kernel_system::Orbiter->get_matrix_from_label(label, v, m, n);

	if (f_v) {
		cout << "geometry_global::do_rank_points_in_PG coeff: ";
		Int_matrix_print(v, m, n);
		cout << endl;
	}

	long int a;
	int i;

	for (i = 0; i < m; i++) {
		F->PG_element_rank_modified_lint(v + i * n, 1, n, a);

		Int_vec_print(cout, v + i * n, n);
		cout << " : " << a << endl;

	}


	FREE_int(v);

}

void geometry_global::do_unrank_points_in_PG(
		field_theory::finite_field *F,
		int n,
		std::string &text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_unrank_points_in_PG" << endl;
	}

	long int *v;
	int len;

	orbiter_kernel_system::Orbiter->get_lint_vector_from_label(text, v, len, 0 /* verbose_level */);

	if (f_v) {
		cout << "geometry_global::do_unrank_points_in_PG rank values = ";
		Lint_vec_print(cout, v, len);
		cout << endl;
	}

	long int a;
	int *M;
	int i;
	int d;

	d = n + 1;
	M = NEW_int(len * d);

	for (i = 0; i < len; i++) {

		a = v[i];

		F->PG_element_unrank_modified_lint(M + i * d, 1, d, a);

		cout << a << " : ";
		Int_vec_print(cout, M + i * d, d);
		cout << endl;
	}


	FREE_int(M);
	FREE_lint(v);
	if (f_v) {
		cout << "geometry_global::do_unrank_points_in_PG done" << endl;
	}

}






void geometry_global::do_intersection_of_two_lines(
		field_theory::finite_field *F,
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

	Int_vec_scan(line_1_basis, Line1, len);
	if (len != 8) {
		cout << "geometry_global::do_intersection_of_two_lines len != 8" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	Int_vec_scan(line_2_basis, Line2, len);
	if (len != 8) {
		cout << "geometry_global::do_intersection_of_two_lines len != 8" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	A = NEW_int(16);
	B = NEW_int(16);
	C = NEW_int(16);

	// Line 1
	Int_vec_copy(Line1, A, 8);
	rk = F->Linear_algebra->perp_standard(4, 2, A, verbose_level);
	if (rk != 2) {
		cout << "geometry_global::do_intersection_of_two_lines rk != 2" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}

	// Line 2
	Int_vec_copy(Line2, B, 8);
	rk = F->Linear_algebra->perp_standard(4, 2, B, verbose_level);
	if (rk != 2) {
		cout << "geometry_global::do_intersection_of_two_lines rk != 2" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}


	Int_vec_copy(A + 8, C, 8);
	Int_vec_copy(B + 8, C + 8, 8);
	rk = F->Linear_algebra->perp_standard(4, 4, C, verbose_level);
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
		Int_matrix_print(C + 12, 1, 4);
		cout << "rk=" << rk << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "geometry_global::do_intersection_of_two_lines normalizing from the right" << endl;
		for (i = 3; i < 4; i++) {
			F->PG_element_normalize(
					C + i * 4, 1, 4);
		}

		cout << "geometry_global::do_intersection_of_two_lines after normalize from the right:" << endl;
		Int_matrix_print(C + 12, 1, 4);
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

void geometry_global::do_transversal(
		field_theory::finite_field *F,
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

	Int_vec_scan(line_1_basis, Line1, len);
	if (len != 8) {
		cout << "geometry_global::do_transversal len != 8" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	Int_vec_scan(line_2_basis, Line2, len);
	if (len != 8) {
		cout << "geometry_global::do_transversal len != 8" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	Int_vec_scan(point, Pt, len);
	if (len != 4) {
		cout << "geometry_global::do_transversal len != 4" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	A = NEW_int(16);
	B = NEW_int(16);

	// Line 1
	Int_vec_copy(Line1, A, 8);
	Int_vec_copy(Pt, A + 8, 4);
	rk = F->Linear_algebra->perp_standard(4, 3, A, verbose_level);
	if (rk != 3) {
		cout << "geometry_global::do_transversal rk != 3" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}
	Int_vec_copy(A + 12, B, 4);

	// Line 2
	Int_vec_copy(Line2, A, 8);
	Int_vec_copy(Pt, A + 8, 4);
	rk = F->Linear_algebra->perp_standard(4, 3, A, verbose_level);
	if (rk != 3) {
		cout << "geometry_global::do_transversal rk != 3" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}
	Int_vec_copy(A + 12, B + 4, 4);

	// B
	rk = F->Linear_algebra->perp_standard(4, 2, B, verbose_level);
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
		Int_matrix_print(B + 8, 2, 4);
		cout << "rk=" << rk << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "geometry_global::do_transversal normalizing from the right" << endl;
		for (i = 2; i < 4; i++) {
			F->PG_element_normalize(
					B + i * 4, 1, 4);
		}

		cout << "geometry_global::do_transversal after normalize from the right:" << endl;
		Int_matrix_print(B + 8, 2, 4);
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


#if 0
void geometry_global::do_cheat_sheet_PG(field_theory::finite_field *F,
		graphics::layered_graph_draw_options *O,
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
	P->projective_space_init(n, F,
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
#endif

void geometry_global::do_cheat_sheet_Gr(field_theory::finite_field *F,
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
	P->projective_space_init(n - 1, F,
		TRUE /*f_init_incidence_structure*/,
		0 /* verbose_level */);
	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_Gr after P->init" << endl;
	}

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_Gr "
				"before P->create_latex_report_for_Grassmannian" << endl;
	}
	P->create_latex_report_for_Grassmannian(k, verbose_level);
	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_Gr "
				"after P->create_latex_report_for_Grassmannian" << endl;
	}



	FREE_OBJECT(P);

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_PG done" << endl;
	}

}


void geometry_global::do_cheat_sheet_hermitian(field_theory::finite_field *F,
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

void geometry_global::do_create_desarguesian_spread(
		field_theory::finite_field *FQ, field_theory::finite_field *Fq,
		int m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "geometry_global::do_create_desarguesian_spread" << endl;
		cout << "geometry_global::do_create_desarguesian_spread "
				"Q=" << FQ->q << " q=" << Fq->q << " m=" << m << endl;
	}

	int s, n;

	if (FQ->p != Fq->p) {
		cout << "geometry_global::do_create_desarguesian_spread "
				"the fields must have the same characteristic" << endl;
		exit(1);
	}
	s = FQ->e / Fq->e;

	if (s * Fq->e != FQ->e) {
		cout << "geometry_global::do_create_desarguesian_spread "
				"Fq is not a subfield of FQ" << endl;
		exit(1);
	}

	n = m * s;
	field_theory::subfield_structure *SubS;
	desarguesian_spread *D;

	SubS = NEW_OBJECT(field_theory::subfield_structure);
	if (f_v) {
		cout << "geometry_global::do_create_desarguesian_spread before SubS->init" << endl;
	}
	SubS->init(FQ, Fq, verbose_level - 2);

	if (f_v) {
		cout << "Field-basis: ";
		Int_vec_print(cout, SubS->Basis, s);
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

void geometry_global::create_decomposition_of_projective_plane(std::string &fname_base,
		projective_space *P,
		long int *points, int nb_points,
		long int *lines, int nb_lines,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::create_decomposition_of_projective_plane" << endl;
	}
	{
		incidence_structure *I;
		data_structures::partitionstack *Stack;
		int depth = INT_MAX;

		I = NEW_OBJECT(incidence_structure);
		I->init_projective_space(P, verbose_level);

		Stack = NEW_OBJECT(data_structures::partitionstack);
		Stack->allocate(I->nb_rows + I->nb_cols, 0 /* verbose_level */);
		Stack->subset_continguous(I->nb_rows, I->nb_cols);
		Stack->split_cell(0 /* verbose_level */);
		Stack->sort_cells();

		int *the_points;
		int *the_lines;
		int i;
		the_points = NEW_int(nb_points);
		the_lines = NEW_int(nb_lines);

		for (i = 0; i < nb_points; i++) {
			the_points[i] = points[i];
		}
		for (i = 0; i < nb_lines; i++) {
			the_lines[i] = lines[i];
		}

		Stack->split_cell_front_or_back(
				the_points, nb_points, TRUE /* f_front*/, verbose_level);

		Stack->split_line_cell_front_or_back(
				the_lines, nb_lines, TRUE /* f_front*/, verbose_level);


		FREE_int(the_points);
		FREE_int(the_lines);

		if (f_v) {
			cout << "geometry_global::create_decomposition_of_projective_plane before I->compute_TDO_safe" << endl;
		}
		I->compute_TDO_safe(*Stack, depth, verbose_level);
		if (f_v) {
			cout << "geometry_global::create_decomposition_of_projective_plane after I->compute_TDO_safe" << endl;
		}


		I->get_and_print_row_tactical_decomposition_scheme_tex(
			cout, FALSE /* f_enter_math */,
			TRUE /* f_print_subscripts */, *Stack);
		I->get_and_print_column_tactical_decomposition_scheme_tex(
			cout, FALSE /* f_enter_math */,
			TRUE /* f_print_subscripts */, *Stack);



		string fname_row_scheme;
		string fname_col_scheme;


		fname_row_scheme.assign(fname_base);
		fname_row_scheme.append("_row_scheme.tex");
		fname_col_scheme.assign(fname_base);
		fname_col_scheme.append("_col_scheme.tex");
		{
			ofstream fp_row_scheme(fname_row_scheme);
			ofstream fp_col_scheme(fname_col_scheme);
			I->get_and_print_row_tactical_decomposition_scheme_tex(
				fp_row_scheme, FALSE /* f_enter_math */,
				TRUE /* f_print_subscripts */, *Stack);
			I->get_and_print_column_tactical_decomposition_scheme_tex(
				fp_col_scheme, FALSE /* f_enter_math */,
				TRUE /* f_print_subscripts */, *Stack);
		}


		FREE_OBJECT(Stack);
		FREE_OBJECT(I);
	}
	if (f_v) {
		cout << "geometry_global::create_decomposition_of_projective_plane done" << endl;
	}

}



void geometry_global::latex_homogeneous_equation(
		field_theory::finite_field *F, int degree, int nb_vars,
		std::string &equation_text,
		std::string &symbol_txt,
		std::string &symbol_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::latex_homogeneous_equation" << endl;
	}
	int *eqn;
	int sz;
	ring_theory::homogeneous_polynomial_domain *Poly;

	Int_vec_scan(equation_text, eqn, sz);
	Poly = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "geometry_global::latex_homogeneous_equation before Poly->init" << endl;
	}
	Poly->init(F,
			degree /* nb_vars */, degree /* degree */,
			t_PART,
			verbose_level);

	Poly->remake_symbols(0 /* symbol_offset */,
				symbol_txt.c_str(),
				symbol_tex.c_str(),
				verbose_level);


	if (Poly->get_nb_monomials() != sz) {
		cout << "Poly->get_nb_monomials() = " << Poly->get_nb_monomials() << endl;
		cout << "number of coefficients given = " << sz << endl;
		exit(1);
	}
	Poly->print_equation_tex(cout, eqn);
	cout << endl;
	if (f_v) {
		cout << "geometry_global::latex_homogeneous_equation done" << endl;
	}

}

void geometry_global::create_BLT_point(field_theory::finite_field *F,
		int *v5, int a, int b, int c, int verbose_level)
// creates the point (-b/2,-c,a,-(b^2/4-ac),1)
// check if it satisfies x_0^2 + x_1x_2 + x_3x_4:
// b^2/4 + (-c)*a + -(b^2/4-ac)
// = b^2/4 -ac -b^2/4 + ac = 0
{
	int f_v = (verbose_level >= 1);
	int v0, v1, v2, v3, v4;
	int half, four, quarter, minus_one;

	if (f_v) {
		cout << "geometry_global::create_BLT_point" << endl;
	}
	four = 4 % F->p;
	half = F->inverse(2);
	quarter = F->inverse(four);
	minus_one = F->negate(1);
	if (f_v) {
		cout << "geometry_global::create_BLT_point "
				"four=" << four << endl;
		cout << "geometry_global::create_BLT_point "
				"half=" << half << endl;
		cout << "geometry_global::create_BLT_point "
				"quarter=" << quarter << endl;
		cout << "geometry_global::create_BLT_point "
				"minus_one=" << minus_one << endl;
	}

	v0 = F->mult(minus_one, F->mult(b, half));
	v1 = F->mult(minus_one, c);
	v2 = a;
	v3 = F->mult(minus_one, F->add(
			F->mult(F->mult(b, b), quarter), F->negate(F->mult(a, c))));
	v4 = 1;
	orbiter_kernel_system::Orbiter->Int_vec->init5(v5, v0, v1, v2, v3, v4);
	if (f_v) {
		cout << "geometry_global::create_BLT_point done" << endl;
	}
}


algebraic_geometry::eckardt_point_info *geometry_global::compute_eckardt_point_info(
		projective_space *P2,
	long int *arc6,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	algebraic_geometry::eckardt_point_info *E;

	if (f_v) {
		cout << "geometry_global::compute_eckardt_point_info" << endl;
	}
	if (P2->n != 2) {
		cout << "geometry_global::compute_eckardt_point_info "
				"P2->n != 2" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "arc: ";
		Lint_vec_print(cout, arc6, 6);
		cout << endl;
	}

	E = NEW_OBJECT(algebraic_geometry::eckardt_point_info);
	E->init(P2, arc6, verbose_level);

	if (f_v) {
		cout << "geometry_global::compute_eckardt_point_info done" << endl;
	}
	return E;
}

int geometry_global::test_nb_Eckardt_points(
		projective_space *P2,
		long int *S, int len, int pt, int nb_E, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = TRUE;
	long int Arc6[6];

	if (f_v) {
		cout << "geometry_global::test_nb_Eckardt_points" << endl;
	}
	if (len != 5) {
		return TRUE;
	}

	Lint_vec_copy(S, Arc6, 5);
	Arc6[5] = pt;

	algebraic_geometry::eckardt_point_info *E;

	E = compute_eckardt_point_info(P2, Arc6, 0/*verbose_level*/);


	if (E->nb_E != nb_E) {
		ret = FALSE;
	}

	FREE_OBJECT(E);

	if (f_v) {
		cout << "geometry_global::test_nb_Eckardt_points done" << endl;
	}
	return ret;
}

void geometry_global::rearrange_arc_for_lifting(
		long int *Arc6,
		long int P1, long int P2, int partition_rk, long int *arc,
		int verbose_level)
// P1 and P2 are points on the arc.
// Find them and remove them
// so we can find the remaining four point of the arc:
{
	int f_v = (verbose_level >= 1);
	long int i, a, h;
	int part[4];
	long int pts[4];
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "geometry_global::rearrange_arc_for_lifting" << endl;
	}
	arc[0] = P1;
	arc[1] = P2;
	h = 2;
	for (i = 0; i < 6; i++) {
		a = Arc6[i];
		if (a == P1 || a == P2) {
			continue;
		}
		arc[h++] = a;
	}
	if (h != 6) {
		cout << "geometry_global::rearrange_arc_for_lifting "
				"h != 6" << endl;
		exit(1);
	}
	// now arc[2], arc[3], arc[4], arc[5] are the remaining four points
	// of the arc.

	Combi.set_partition_4_into_2_unrank(partition_rk, part);

	Lint_vec_copy(arc + 2, pts, 4);

	for (i = 0; i < 4; i++) {
		a = part[i];
		arc[2 + i] = pts[a];
	}

	if (f_v) {
		cout << "geometry_global::rearrange_arc_for_lifting done" << endl;
	}
}

void geometry_global::find_two_lines_for_arc_lifting(
		projective_space *P,
		long int P1, long int P2, long int &line1, long int &line2,
		int verbose_level)
// P1 and P2 are points on the arc and in the plane W=0.
// Note the points are points in PG(3,q), not in local coordinates in W=0.
// We find two skew lines in space through P1 and P2, respectively.
{
	int f_v = (verbose_level >= 1);
	int Basis[16];
	int Basis2[16];
	int Basis_search[16];
	int Basis_search_copy[16];
	int base_cols[4];
	int i, N, rk;
	geometry_global Gg;

	if (f_v) {
		cout << "geometry_global::find_two_lines_for_arc_lifting" << endl;
	}
	if (P->n != 3) {
		cout << "geometry_global::find_two_lines_for_arc_lifting "
				"P->n != 3" << endl;
		exit(1);
	}
	// unrank points P1 and P2 in the plane W=3:
	// Note the points are points in PG(3,q), not in local coordinates.
	P->unrank_point(Basis, P1);
	P->unrank_point(Basis + 4, P2);
	if (Basis[3]) {
		cout << "geometry_global::find_two_lines_for_arc_lifting "
				"Basis[3] != 0, the point P1 does not lie "
				"in the hyperplane W = 0" << endl;
		exit(1);
	}
	if (Basis[7]) {
		cout << "geometry_global::find_two_lines_for_arc_lifting "
				"Basis[7] != 0, the point P2 does not lie "
				"in the hyperplane W = 0" << endl;
		exit(1);
	}
	Int_vec_zero(Basis + 8, 8);

	N = Gg.nb_PG_elements(3, P->q);
	// N = the number of points in PG(3,q)

	// Find the first line.
	// Loop over all points P.
	// Make sure the point does not belong to the hyperplane,
	// i.e. the last coordinate is nonzero.
	// Make sure the rank of the subspace spanned by P1, P2 and P is three.

	for (i = 0; i < N; i++) {
		Int_vec_copy(Basis, Basis_search, 4);
		Int_vec_copy(Basis + 4, Basis_search + 4, 4);
		P->F->PG_element_unrank_modified(Basis_search + 8, 1, 4, i);
		if (Basis_search[11] == 0) {
			continue;
		}
		Int_vec_copy(Basis_search, Basis_search_copy, 12);
		rk = P->F->Linear_algebra->Gauss_easy_memory_given(Basis_search_copy, 3, 4, base_cols);
		if (rk == 3) {
			break;
		}
	}
	if (i == N) {
		cout << "geometry_global::find_two_lines_for_arc_lifting "
				"i == N, could not find line1" << endl;
		exit(1);
	}
	int p0, p1;

	p0 = i;

	// Find the second line.
	// Loop over all points Q after the first P.
	// Make sure the point does not belong to the hyperplane,
	// i.e. the last coordinate is nonzero.
	// Make sure the rank of the subspace spanned by P1, P2 and P and Q is four.

	for (i = p0 + 1; i < N; i++) {
		Int_vec_copy(Basis, Basis_search, 4);
		Int_vec_copy(Basis + 4, Basis_search + 4, 4);
		P->F->PG_element_unrank_modified(Basis_search + 8, 1, 4, p0);
		P->F->PG_element_unrank_modified(Basis_search + 12, 1, 4, i);
		if (Basis_search[15] == 0) {
			continue;
		}
		Int_vec_copy(Basis_search, Basis_search_copy, 16);
		rk = P->F->Linear_algebra->Gauss_easy_memory_given(
				Basis_search_copy, 4, 4, base_cols);
		if (rk == 4) {
			break;
		}
	}
	if (i == N) {
		cout << "geometry_global::find_two_lines_for_arc_lifting "
				"i == N, could not find line2" << endl;
		exit(1);
	}
	p1 = i;

	if (f_v) {
		cout << "geometry_global::find_two_lines_for_arc_lifting "
				"p0=" << p0 << " p1=" << p1 << endl;
	}
	P->F->PG_element_unrank_modified(Basis + 8, 1, 4, p0);
	P->F->PG_element_unrank_modified(Basis + 12, 1, 4, p1);
	if (f_v) {
		cout << "geometry_global::find_two_lines_for_arc_lifting " << endl;
		cout << "Basis:" << endl;
		Int_matrix_print(Basis, 4, 4);
	}
	Int_vec_copy(Basis, Basis2, 4);
	Int_vec_copy(Basis + 8, Basis2 + 4, 4);
	Int_vec_copy(Basis + 4, Basis2 + 8, 4);
	Int_vec_copy(Basis + 12, Basis2 + 12, 4);
	if (f_v) {
		cout << "geometry_global::find_two_lines_for_arc_lifting "
				"Basis2:" << endl;
		Int_matrix_print(Basis2, 4, 4);
	}
	line1 = P->rank_line(Basis2);
	line2 = P->rank_line(Basis2 + 8);
	if (f_v) {
		cout << "geometry_global::find_two_lines_for_arc_lifting "
				"line1=" << line1 << " line2=" << line2 << endl;
	}
	if (f_v) {
		cout << "geometry_global::find_two_lines_for_arc_lifting "
				"done" << endl;
	}
}

void geometry_global::hyperplane_lifting_with_two_lines_fixed(
		projective_space *P,
		int *A3, int f_semilinear, int frobenius,
		long int line1, long int line2,
		int *A4,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Line1[8];
	int Line2[8];
	int P1A[3];
	int P2A[3];
	int A3t[9];
	int x[3];
	int y[3];
	int xmy[4];
	int Mt[16];
	int M[16];
	int Mv[16];
	int v[4];
	int w[4];
	int lmei[4];
	int m1;
	int M_tmp[16];
	int tmp_basecols[4];
	int lambda, mu; //, epsilon, iota;
	int abgd[4];
	int i, j;
	int f_swap; // does A3 swap P1 and P2?
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed" << endl;
	}
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"A3:" << endl;
		Int_matrix_print(A3, 3, 3);
		cout << "f_semilinear = " << f_semilinear
				<< " frobenius=" << frobenius << endl;
	}
	m1 = P->F->negate(1);
	P->unrank_line(Line1, line1);
	P->unrank_line(Line2, line2);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"input Line1:" << endl;
		Int_matrix_print(Line1, 2, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"input Line2:" << endl;
		Int_matrix_print(Line2, 2, 4);
	}
	P->F->Linear_algebra->Gauss_step_make_pivot_one(Line1 + 4, Line1,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line1[3] = 0 and Line1[7] = 1
	P->F->Linear_algebra->Gauss_step_make_pivot_one(Line2 + 4, Line2,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line2[3] = 0 and Line2[7] = 1
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"modified Line1:" << endl;
		Int_matrix_print(Line1, 2, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"modified Line2:" << endl;
		Int_matrix_print(Line2, 2, 4);
	}

	P->F->PG_element_normalize(Line1, 1, 4);
	P->F->PG_element_normalize(Line2, 1, 4);
	P->F->PG_element_normalize(Line1 + 4, 1, 4);
	P->F->PG_element_normalize(Line2 + 4, 1, 4);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"P1 = first point on Line1:" << endl;
		Int_matrix_print(Line1, 1, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"P2 = first point on Line2:" << endl;
		Int_matrix_print(Line2, 1, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"x = second point on Line1:" << endl;
		Int_matrix_print(Line1 + 4, 1, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"y = second point on Line2:" << endl;
		Int_matrix_print(Line2 + 4, 1, 4);
	}
	// compute P1 * A3 to figure out if A switches P1 and P2 or not:
	P->F->Linear_algebra->mult_vector_from_the_left(Line1, A3, P1A, 3, 3);
	P->F->Linear_algebra->mult_vector_from_the_left(Line2, A3, P2A, 3, 3);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"P1 * A = " << endl;
		Int_matrix_print(P1A, 1, 3);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"P2 * A = " << endl;
		Int_matrix_print(P2A, 1, 3);
	}
	if (f_semilinear) {
		if (f_v) {
			cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
					"applying frobenius" << endl;
		}
		P->F->Linear_algebra->vector_frobenius_power_in_place(P1A, 3, frobenius);
		P->F->Linear_algebra->vector_frobenius_power_in_place(P2A, 3, frobenius);
	}
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"P1 * A ^Phi^frobenius = " << endl;
		Int_matrix_print(P1A, 1, 3);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"P2 * A ^Phi^frobenius = " << endl;
		Int_matrix_print(P2A, 1, 3);
	}
	P->F->PG_element_normalize(P1A, 1, 3);
	P->F->PG_element_normalize(P2A, 1, 3);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"normalized P1 * A = " << endl;
		Int_matrix_print(P1A, 1, 3);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"normalized P2 * A = " << endl;
		Int_matrix_print(P2A, 1, 3);
	}
	if (Sorting.int_vec_compare(P1A, Line1, 3) == 0) {
		f_swap = FALSE;
		if (Sorting.int_vec_compare(P2A, Line2, 3)) {
			cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"We don't have a swap but A3 does not stabilize P2" << endl;
			exit(1);
		}
	}
	else if (Sorting.int_vec_compare(P1A, Line2, 3) == 0) {
		f_swap = TRUE;
		if (Sorting.int_vec_compare(P2A, Line1, 3)) {
			cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"We have a swap but A3 does not map P2 to P1" << endl;
			exit(1);
		}
	}
	else {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"unable to determine if we have a swap or not." << endl;
		exit(1);
	}

	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"f_swap=" << f_swap << endl;
	}

	Int_vec_copy(Line1 + 4, x, 3);
	Int_vec_copy(Line2 + 4, y, 3);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"x:" << endl;
		Int_matrix_print(x, 1, 3);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"y:" << endl;
		Int_matrix_print(y, 1, 3);
	}

	P->F->Linear_algebra->linear_combination_of_vectors(1, x, m1, y, xmy, 3);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"xmy:" << endl;
		Int_matrix_print(xmy, 1, 3);
	}

	P->F->Linear_algebra->transpose_matrix(A3, A3t, 3, 3);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"A3t:" << endl;
		Int_matrix_print(A3t, 3, 3);
	}


	P->F->Linear_algebra->mult_vector_from_the_right(A3t, xmy, v, 3, 3);
	if (f_semilinear) {
		P->F->Linear_algebra->vector_frobenius_power_in_place(v, 3, frobenius);
	}
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"v:" << endl;
		Int_matrix_print(v, 1, 3);
	}
	P->F->Linear_algebra->mult_vector_from_the_right(A3t, x, w, 3, 3);
	if (f_semilinear) {
		P->F->Linear_algebra->vector_frobenius_power_in_place(w, 3, frobenius);
	}
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"w:" << endl;
		Int_matrix_print(w, 1, 3);
	}

	if (f_swap) {
		Int_vec_copy(Line2 + 4, Mt + 0, 4);
		Int_vec_copy(Line2 + 0, Mt + 4, 4);
		Int_vec_copy(Line1 + 4, Mt + 8, 4);
		Int_vec_copy(Line1 + 0, Mt + 12, 4);
	}
	else {
		Int_vec_copy(Line1 + 4, Mt + 0, 4);
		Int_vec_copy(Line1 + 0, Mt + 4, 4);
		Int_vec_copy(Line2 + 4, Mt + 8, 4);
		Int_vec_copy(Line2 + 0, Mt + 12, 4);
	}

	P->F->Linear_algebra->negate_vector_in_place(Mt + 8, 8);
	P->F->Linear_algebra->transpose_matrix(Mt, M, 4, 4);
	//int_vec_copy(Mt, M, 16);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"M:" << endl;
		Int_matrix_print(M, 4, 4);
	}

	P->F->Linear_algebra->invert_matrix_memory_given(M, Mv, 4, M_tmp, tmp_basecols, 0 /* verbose_level */);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"Mv:" << endl;
		Int_matrix_print(Mv, 4, 4);
	}

	v[3] = 0;
	w[3] = 0;
	P->F->Linear_algebra->mult_vector_from_the_right(Mv, v, lmei, 4, 4);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"lmei:" << endl;
		Int_matrix_print(lmei, 1, 4);
	}
	lambda = lmei[0];
	mu = lmei[1];
	//epsilon = lmei[2];
	//iota = lmei[3];

	if (f_swap) {
		P->F->Linear_algebra->linear_combination_of_three_vectors(
				lambda, y, mu, Line2, m1, w, abgd, 3);
	}
	else {
		P->F->Linear_algebra->linear_combination_of_three_vectors(
				lambda, x, mu, Line1, m1, w, abgd, 3);
	}
	abgd[3] = lambda;
	if (f_semilinear) {
		P->F->Linear_algebra->vector_frobenius_power_in_place(abgd, 4, P->F->e - frobenius);
	}
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"abgd:" << endl;
		Int_matrix_print(abgd, 1, 4);
	}
	// make an identity matrix:
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			A4[i * 4 + j] = A3[i * 3 + j];
		}
		A4[i * 4 + 3] = 0;
	}
	// fill in the last row:
	Int_vec_copy(abgd, A4 + 4 * 3, 4);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"A4:" << endl;
		Int_matrix_print(A4, 4, 4);
	}

	if (f_semilinear) {
		A4[16] = frobenius;
	}

	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_fixed done" << endl;
	}
}

void geometry_global::hyperplane_lifting_with_two_lines_moved(
		projective_space *P,
		long int line1_from, long int line1_to,
		long int line2_from, long int line2_to,
		int *A4,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Line1_from[8];
	int Line2_from[8];
	int Line1_to[8];
	int Line2_to[8];
	int P1[4];
	int P2[4];
	int x[4];
	int y[4];
	int u[4];
	int v[4];
	int umv[4];
	int M[16];
	int Mv[16];
	int lmei[4];
	int m1;
	int M_tmp[16];
	int tmp_basecols[4];
	int lambda, mu; //, epsilon, iota;
	int abgd[4];
	int i, j;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved" << endl;
	}
	m1 = P->F->negate(1);

	P->unrank_line(Line1_from, line1_from);
	P->unrank_line(Line2_from, line2_from);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"input Line1_from:" << endl;
		Int_matrix_print(Line1_from, 2, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"input Line2_from:" << endl;
		Int_matrix_print(Line2_from, 2, 4);
	}
	P->F->Linear_algebra->Gauss_step_make_pivot_one(Line1_from + 4, Line1_from,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line1[3] = 0 and Line1[7] = 1
	P->F->Linear_algebra->Gauss_step_make_pivot_one(Line2_from + 4, Line2_from,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line2[3] = 0 and Line2[7] = 1
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"modified Line1_from:" << endl;
		Int_matrix_print(Line1_from, 2, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"modified Line2_from:" << endl;
		Int_matrix_print(Line2_from, 2, 4);
	}

	P->F->PG_element_normalize(Line1_from, 1, 4);
	P->F->PG_element_normalize(Line2_from, 1, 4);
	P->F->PG_element_normalize(Line1_from + 4, 1, 4);
	P->F->PG_element_normalize(Line2_from + 4, 1, 4);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"P1 = first point on Line1_from:" << endl;
		Int_matrix_print(Line1_from, 1, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"P2 = first point on Line2_from:" << endl;
		Int_matrix_print(Line2_from, 1, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"u = second point on Line1_from:" << endl;
		Int_matrix_print(Line1_from + 4, 1, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"v = second point on Line2_from:" << endl;
		Int_matrix_print(Line2_from + 4, 1, 4);
	}
	Int_vec_copy(Line1_from + 4, u, 4);
	Int_vec_copy(Line1_from, P1, 4);
	Int_vec_copy(Line2_from + 4, v, 4);
	Int_vec_copy(Line2_from, P2, 4);


	P->unrank_line(Line1_to, line1_to);
	P->unrank_line(Line2_to, line2_to);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"input Line1_to:" << endl;
		Int_matrix_print(Line1_to, 2, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"input Line2_to:" << endl;
		Int_matrix_print(Line2_to, 2, 4);
	}
	P->F->Linear_algebra->Gauss_step_make_pivot_one(Line1_to + 4, Line1_to,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line1[3] = 0 and Line1[7] = 1
	P->F->Linear_algebra->Gauss_step_make_pivot_one(Line2_to + 4, Line2_to,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line2[3] = 0 and Line2[7] = 1
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"modified Line1_to:" << endl;
		Int_matrix_print(Line1_to, 2, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"modified Line2_to:" << endl;
		Int_matrix_print(Line2_to, 2, 4);
	}

	P->F->PG_element_normalize(Line1_to, 1, 4);
	P->F->PG_element_normalize(Line2_to, 1, 4);
	P->F->PG_element_normalize(Line1_to + 4, 1, 4);
	P->F->PG_element_normalize(Line2_to + 4, 1, 4);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"P1 = first point on Line1_to:" << endl;
		Int_matrix_print(Line1_to, 1, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"P2 = first point on Line2_to:" << endl;
		Int_matrix_print(Line2_to, 1, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"x = second point on Line1_to:" << endl;
		Int_matrix_print(Line1_to + 4, 1, 4);
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"y = second point on Line2_to:" << endl;
		Int_matrix_print(Line2_to + 4, 1, 4);
	}


	Int_vec_copy(Line1_to + 4, x, 4);
	//int_vec_copy(Line1_to, P1, 4);
	if (Sorting.int_vec_compare(P1, Line1_to, 4)) {
		cout << "Line1_from and Line1_to must intersect in W=0" << endl;
		exit(1);
	}
	Int_vec_copy(Line2_to + 4, y, 4);
	//int_vec_copy(Line2_to, P2, 4);
	if (Sorting.int_vec_compare(P2, Line2_to, 4)) {
		cout << "Line2_from and Line2_to must intersect in W=0" << endl;
		exit(1);
	}


	P->F->Linear_algebra->linear_combination_of_vectors(1, u, m1, v, umv, 3);
	umv[3] = 0;
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"umv:" << endl;
		Int_matrix_print(umv, 1, 4);
	}

	Int_vec_copy(x, M + 0, 4);
	Int_vec_copy(P1, M + 4, 4);
	Int_vec_copy(y, M + 8, 4);
	Int_vec_copy(P2, M + 12, 4);

	P->F->Linear_algebra->negate_vector_in_place(M + 8, 8);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"M:" << endl;
		Int_matrix_print(M, 4, 4);
	}

	P->F->Linear_algebra->invert_matrix_memory_given(M, Mv, 4, M_tmp, tmp_basecols, 0 /* verbose_level */);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"Mv:" << endl;
		Int_matrix_print(Mv, 4, 4);
	}

	P->F->Linear_algebra->mult_vector_from_the_left(umv, Mv, lmei, 4, 4);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"lmei=" << endl;
		Int_matrix_print(lmei, 1, 4);
	}
	lambda = lmei[0];
	mu = lmei[1];
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"lambda=" << lambda << " mu=" << mu << endl;
	}

	P->F->Linear_algebra->linear_combination_of_three_vectors(lambda, x, mu, P1, m1, u, abgd, 3);
	// abgd = lambda * x + mu * P1 - u, with a lambda in the 4th coordinate.

	abgd[3] = lambda;

	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"abgd:" << endl;
		Int_matrix_print(abgd, 1, 4);
	}

	// make an identity matrix:
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			if (i == j) {
				A4[i * 4 + j] = 1;
			}
			else {
				A4[i * 4 + j] = 0;
			}
		}
	}
	// fill in the last row:
	Int_vec_copy(abgd, A4 + 3 * 4, 4);
	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved "
				"A4:" << endl;
		Int_matrix_print(A4, 4, 4);

		P->F->print_matrix_latex(cout, A4, 4, 4);

	}

	if (f_v) {
		cout << "geometry_global::hyperplane_lifting_with_two_lines_moved done" << endl;
	}

}

void geometry_global::andre_preimage(
		projective_space *P2, projective_space *P4,
	long int *set2, int sz2, long int *set4, int &sz4, int verbose_level)
// we must be a projective plane
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	field_theory::finite_field *FQ;
	field_theory::finite_field *Fq;
	int /*Q,*/ q;
	int *v, *w1, *w2, *w3, *v2;
	int *components;
	int *embedding;
	int *pair_embedding;
	int i, h, k, a, a0, a1, b, b0, b1, e, alpha;

	if (f_v) {
		cout << "geometry_global::andre_preimage" << endl;
	}
	FQ = P2->F;
	//Q = FQ->q;
	alpha = FQ->p;
	if (f_vv) {
		cout << "alpha=" << alpha << endl;
		//FQ->print(TRUE /* f_add_mult_table */);
	}


	Fq = P4->F;
	q = Fq->q;

	v = NEW_int(3);
	w1 = NEW_int(5);
	w2 = NEW_int(5);
	w3 = NEW_int(5);
	v2 = NEW_int(2);
	e = P2->F->e >> 1;
	if (f_vv) {
		cout << "geometry_global::andre_preimage e=" << e << endl;
	}

	FQ->subfield_embedding_2dimensional(*Fq,
		components, embedding, pair_embedding, verbose_level - 3);

		// we think of FQ as two dimensional vector space
		// over Fq with basis (1,alpha)
		// for i,j \in Fq, with x = i + j * alpha \in FQ, we have
		// pair_embedding[i * q + j] = x;
		// also,
		// components[x * 2 + 0] = i;
		// components[x * 2 + 1] = j;
		// also, for i \in Fq, embedding[i] is the element
		// in FQ that corresponds to i

		// components[Q * 2]
		// embedding[q]
		// pair_embedding[q * q]

	if (f_vv) {
		FQ->print_embedding(*Fq,
			components, embedding, pair_embedding);
	}


	sz4 = 0;
	for (i = 0; i < sz2; i++) {
		if (f_vv) {
			cout << "geometry_global::andre_preimage "
					"input point " << i << " : ";
		}
		P2->unrank_point(v, set2[i]);
		FQ->PG_element_normalize(v, 1, 3);
		if (f_vv) {
			Int_vec_print(cout, v, 3);
			cout << " becomes ";
		}

		if (v[2] == 0) {

			// we are dealing with a point on the
			// line at infinity.
			// Such a point corresponds to a line of the spread.
			// We create the line and then create all
			// q + 1 points on that line.

			if (f_vv) {
				cout << endl;
			}
			// w1[4] is the GF(q)-vector corresponding
			// to the GF(q^2)-vector v[2]
			// w2[4] is the GF(q)-vector corresponding
			// to the GF(q^2)-vector v[2] * alpha
			// where v[2] runs through the points of PG(1,q^2).
			// That way, w1[4] and w2[4] are a GF(q)-basis for the
			// 2-dimensional subspace v[2] (when viewed over GF(q)),
			// which is an element of the regular spread.

			for (h = 0; h < 2; h++) {
				a = v[h];
				a0 = components[a * 2 + 0];
				a1 = components[a * 2 + 1];
				b = FQ->mult(a, alpha);
				b0 = components[b * 2 + 0];
				b1 = components[b * 2 + 1];
				w1[2 * h + 0] = a0;
				w1[2 * h + 1] = a1;
				w2[2 * h + 0] = b0;
				w2[2 * h + 1] = b1;
			}
			if (FALSE) {
				cout << "w1=";
				Int_vec_print(cout, w1, 4);
				cout << "w2=";
				Int_vec_print(cout, w2, 4);
				cout << endl;
			}

			// now we create all points on the line
			// spanned by w1[4] and w2[4]:
			// There are q + 1 of these points.
			// We make sure that the coordinate vectors
			// have a zero in the last spot.

			for (h = 0; h < q + 1; h++) {
				Fq->PG_element_unrank_modified(v2, 1, 2, h);
				if (FALSE) {
					cout << "v2=";
					Int_vec_print(cout, v2, 2);
					cout << " : ";
				}
				for (k = 0; k < 4; k++) {
					w3[k] = Fq->add(Fq->mult(v2[0], w1[k]),
							Fq->mult(v2[1], w2[k]));
				}
				w3[4] = 0;
				if (f_vv) {
					cout << " ";
					Int_vec_print(cout, w3, 5);
				}
				a = P4->rank_point(w3);
				if (f_vv) {
					cout << " rank " << a << endl;
				}
				set4[sz4++] = a;
			}
		}
		else {

			// we are dealing with an affine point:
			// We make sure that the coordinate vector
			// has a zero in the last spot.


			for (h = 0; h < 2; h++) {
				a = v[h];
				a0 = components[a * 2 + 0];
				a1 = components[a * 2 + 1];
				w1[2 * h + 0] = a0;
				w1[2 * h + 1] = a1;
			}
			w1[4] = 1;
			if (f_vv) {
				//cout << "w1=";
				Int_vec_print(cout, w1, 5);
			}
			a = P4->rank_point(w1);
			if (f_vv) {
				cout << " rank " << a << endl;
			}
			set4[sz4++] = a;
		}
	}
	if (f_v) {
		cout << "geometry_global::andre_preimage "
				"we found " << sz4 << " points:" << endl;
		Lint_vec_print(cout, set4, sz4);
		cout << endl;
		P4->print_set(set4, sz4);
		for (i = 0; i < sz4; i++) {
			cout << set4[i] << " ";
		}
		cout << endl;
	}


	FREE_int(components);
	FREE_int(embedding);
	FREE_int(pair_embedding);
	if (f_v) {
		cout << "geometry_global::andre_preimage done" << endl;
	}
}



}}}

