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
namespace other_geometry {



geometry_global::geometry_global()
{
	Record_birth();

}

geometry_global::~geometry_global()
{
	Record_death();

}


long int geometry_global::nb_PG_elements(
		int n, int q)
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

long int geometry_global::nb_PG_elements_not_in_subspace(
		int n, int m, int q)
// |PG_n(q)| - |PG_m(q)|
{
	long int a, b;

	a = nb_PG_elements(n, q);
	b = nb_PG_elements(m, q);
	return a - b;
}

long int geometry_global::nb_AG_elements(
		int n, int q)
// $q^n$
{
	algebra::number_theory::number_theory_domain NT;

	return NT.i_power_j_lint(q, n);
}

long int geometry_global::nb_affine_lines(
		int n, int q)
{
	algebra::number_theory::number_theory_domain NT;
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


long int geometry_global::AG_element_rank(
		int q, int *v, int stride, int len)
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

void geometry_global::AG_element_unrank(
		int q, int *v, int stride, int len, long int a)
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


int geometry_global::AG_element_next(
		int q, int *v, int stride, int len)
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
			return true;
		}
		else {
			v[i * stride] = 0;
		}
	}
	return false;
}



void geometry_global::AG_element_rank_longinteger(
		int q,
		int *v, int stride, int len,
		algebra::ring_theory::longinteger_object &a)
{
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object Q, a1;
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

void geometry_global::AG_element_unrank_longinteger(
		int q,
		int *v, int stride, int len,
		algebra::ring_theory::longinteger_object &a)
{
	int i, r;
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object a0, Q, a1;

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


int geometry_global::PG_element_modified_is_in_subspace(
		int n, int m, int *v)
{
	int j;

	for (j = m + 1; j < n + 1; j++) {
		if (v[j]) {
			return false;
		}
	}
	return true;
}




int geometry_global::test_if_arc(
		algebra::field_theory::finite_field *Fq,
		int *pt_coords,
		int *set, int set_sz, int k, int verbose_level)
// Used by Hill_cap56()
{
	int f_v = false; //(verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int subset[3];
	int subset1[3];
	int *Mtx;
	int ret = false;
	int i, j, a, rk;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;


	if (f_v) {
		cout << "geometry_global::test_if_arc testing set" << endl;
		Int_vec_print(cout, set, set_sz);
		cout << endl;
	}
	Mtx = NEW_int(3 * k);

	Combi.first_k_subset(subset, set_sz, 3);
	while (true) {
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
		cout << "geometry_global::test_if_arc: passes the arc test" << endl;
	}
	ret = true;
done:

	FREE_int(Mtx);
	return ret;
}

void geometry_global::create_Buekenhout_Metz(
		algebra::field_theory::finite_field *Fq,
		algebra::field_theory::finite_field *FQ,
	int f_classical, int f_Uab, int parameter_a, int parameter_b,
	std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, rk, d = 3;
	int v[3];
	finite_geometries::buekenhout_metz *BM;

	if (f_v) {
		cout << "geometry_global::create_Buekenhout_Metz" << endl;
	}


	BM = NEW_OBJECT(finite_geometries::buekenhout_metz);

	BM->buekenhout_metz_init(
			Fq, FQ,
		f_Uab, parameter_a, parameter_b, f_classical,
		verbose_level);


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

	BM->get_name(name);
	fname = "unital_" + name;

	FREE_OBJECT(BM);
	if (f_v) {
		cout << "geometry_global::create_Buekenhout_Metz done" << endl;
	}

}


long int geometry_global::count_Sbar(
		int n, int q)
{
	return count_T1(1, n, q);
}

long int geometry_global::count_S(
		int n, int q)
{
	return (q - 1) * count_Sbar(n, q) + 1;
}

long int geometry_global::count_N1(
		int n, int q)
{
	if (n <= 0) {
		return 0;
		}
	return nb_pts_N1(n, q);
}

long int geometry_global::count_T1(
		int epsilon, int n, int q)
// n = Witt index
{
	algebra::number_theory::number_theory_domain NT;

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
		cout << "geometry_global::count_T1 "
				"epsilon = " << epsilon
				<< " not yet implemented, returning 0" << endl;
		return 0;
		}
	//exit(1);
}

long int geometry_global::count_T2(
		int n, int q)
{
	algebra::number_theory::number_theory_domain NT;

	if (n <= 0) {
		return 0;
		}
	return (NT.i_power_j_lint(q, 2 * n - 2) - 1) *
			(NT.i_power_j_lint(q, n) - 1) *
			(NT.i_power_j_lint(q, n - 2) + 1) / ((q - 1) * (NT.i_power_j_lint(q, 2) - 1));
}

long int geometry_global::nb_pts_Qepsilon(
		int epsilon, int k, int q)
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
		cout << "geometry_global::nb_pts_Qepsilon "
				"epsilon must be one of 0,1,-1" << endl;
		exit(1);
	}
}

int geometry_global::dimension_given_Witt_index(
		int epsilon, int n)
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
		cout << "geometry_global::dimension_given_Witt_index "
				"epsilon must be 0,1,-1" << endl;
		exit(1);
	}
}

int geometry_global::Witt_index(
		int epsilon, int k)
// k = projective dimension
{
	int n;

	if (epsilon == 0) {
		if (!EVEN(k)) {
			cout << "geometry_global::Witt_index "
					"dimension k must be even" << endl;
			cout << "k = " << k << endl;
			cout << "epsilon = " << epsilon << endl;
			exit(1);
			}
		n = k >> 1; // Witt index
	}
	else if (epsilon == 1) {
		if (!ODD(k)) {
			cout << "geometry_global::Witt_index "
					"dimension k must be odd" << endl;
			cout << "k = " << k << endl;
			cout << "epsilon = " << epsilon << endl;
			exit(1);
			}
		n = (k >> 1) + 1; // Witt index
	}
	else if (epsilon == -1) {
		if (!ODD(k)) {
			cout << "geometry_global::Witt_index "
					"dimension k must be odd" << endl;
			cout << "k = " << k << endl;
			cout << "epsilon = " << epsilon << endl;
			exit(1);
			}
		n = k >> 1; // Witt index
	}
	else {
		cout << "geometry_global::Witt_index "
				"epsilon must be one of 0,1,-1" << endl;
		exit(1);
	}
	return n;
}

long int geometry_global::nb_pts_Q(
		int k, int q)
// number of singular points on Q(k,q), parabolic quadric, so k is even
{
	int n;

	n = Witt_index(0, k);
	return nb_pts_Sbar(n, q) + nb_pts_N1(n, q);
}

long int geometry_global::nb_pts_Qplus(
		int k, int q)
// number of singular points on Q^+(k,q)
{
	int n;

	n = Witt_index(1, k);
	return nb_pts_Sbar(n, q);
}

long int geometry_global::nb_pts_Qminus(
		int k, int q)
// number of singular points on Q^-(k,q)
{
	int n;

	n = Witt_index(-1, k);
	return nb_pts_Sbar(n, q) + (q + 1) * nb_pts_N1(n, q);
}


// #############################################################################
// the following functions are for the hyperbolic quadric with Witt index n:
// #############################################################################

long int geometry_global::nb_pts_S(
		int n, int q)
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

long int geometry_global::nb_pts_N(
		int n, int q)
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

long int geometry_global::nb_pts_N1(
		int n, int q)
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

long int geometry_global::nb_pts_Sbar(
		int n, int q)
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

long int geometry_global::nb_pts_Nbar(
		int n, int q)
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


#if 0
void geometry_global::test_Orthogonal(int epsilon, int k, int q)
// only works for epsilon = 0
{
	field_theory::finite_field GFq;
	int *v;
	int stride = 1, /*n,*/ len; //, h, wt;
	long int i, j, a, nb;
	int c1 = 0, c2 = 0, c3 = 0;
	int verbose_level = 0;

	cout << "geometry_global::test_Orthogonal" << endl;
	GFq.finite_field_init(q, false /* f_without_tables */, verbose_level);
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
		GFq.Linear_algebra->choose_anisotropic_form(c1, c2, c3, true);
		}
	for (i = 0; i < nb; i++) {
		GFq.Orthogonal_indexing->Q_epsilon_unrank(v,
				stride, epsilon, k, c1, c2, c3, i, 0 /* verbose_level */);

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
		j = GFq.Orthogonal_indexing->Q_epsilon_rank(v,
				stride, epsilon, k, c1, c2, c3, 0 /* verbose_level */);
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
	cout << "geometry_global::test_Orthogonal done" << endl;
}

void geometry_global::test_orthogonal(int n, int q)
{
	int *v;
	field_theory::finite_field GFq;
	long int i, j, a;
	int stride = 1;
	long int nb;
	int verbose_level = 0;

	cout << "geometry_global::test_orthogonal" << endl;
	GFq.finite_field_init(q, false /* f_without_tables */, verbose_level);
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
	cout << "geometry_global::test_orthogonal done" << endl;
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

int &geometry_global::TDO_upper_bound(
		int i, int j)
{
	int m, bound;

	if (i <= 0) {
		cout << "geometry_global::TDO_upper_bound i <= 0, i = " << i << endl;
		exit(1);
	}
	if (j <= 0) {
		cout << "geometry_global::TDO_upper_bound j <= 0, j = " << j << endl;
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
		cout << "geometry_global::TDO_upper_bound = -1 i=" << i << " j=" << j << endl;
		exit(1);
	}
	//cout << "PACKING " << i << " " << j << " = " << bound << endl;
	return TDO_upper_bound_internal(i, j);
}

int &geometry_global::TDO_upper_bound_internal(
		int i, int j)
{
	if (i > TDO_upper_bounds_v_max) {
		cout << "geometry_global::TDO_upper_bound_internal "
				"i > v_max" << endl;
		cout << "i=" << i << endl;
		cout << "TDO_upper_bounds_v_max=" << TDO_upper_bounds_v_max << endl;
		exit(1);
	}
	if (i <= 0) {
		cout << "geometry_global::TDO_upper_bound_internal "
				"i <= 0, i = " << i << endl;
		exit(1);
	}
	if (j <= 0) {
		cout << "geometry_global::TDO_upper_bound_internal "
				"j <= 0, j = " << j << endl;
		exit(1);
	}
	return TDO_upper_bounds_table[(i - 1) * TDO_upper_bounds_v_max + j - 1];
}

int &geometry_global::TDO_upper_bound_source(
		int i, int j)
{
	if (i > TDO_upper_bounds_v_max) {
		cout << "geometry_global::TDO_upper_bound_source "
				"i > v_max" << endl;
		cout << "i=" << i << endl;
		cout << "TDO_upper_bounds_v_max=" << TDO_upper_bounds_v_max << endl;
		exit(1);
	}
	if (i <= 0) {
		cout << "geometry_global::TDO_upper_bound_source "
				"i <= 0, i = " << i << endl;
		exit(1);
	}
	if (j <= 0) {
		cout << "geometry_global::TDO_upper_bound_source "
				"j <= 0, j = " << j << endl;
		exit(1);
	}
	return TDO_upper_bounds_table_source[(i - 1) * TDO_upper_bounds_v_max + j - 1];
}

int geometry_global::braun_test_single_type(
		int v, int k, int ak)
{
	int i, l, s, m;

	i = 0;
	s = 0;
	for (l = 1; l <= ak; l++) {
		m = MAXIMUM(k - i, 0);
		s += m;
		if (s > v) {
			return false;
		}
		i++;
	}
	return true;
}

int geometry_global::braun_test_upper_bound(
		int v, int k)
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
			if (braun_test_single_type(v, k, n) == false) {
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

void geometry_global::TDO_refine_init_upper_bounds(
		int v_max)
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
		if (TDO_upper_bounds_initial_data[u * 3 + 0] == -1) {
			break;
		}
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
			if (TDO_upper_bound_internal(i, j) != -1) {
				continue;
			}
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

void geometry_global::TDO_refine_extend_upper_bounds(
		int new_v_max)
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

int geometry_global::braun_test_on_line_type(
		int v, int *type)
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
				return false;
			}
			i++;
		}
	}
	return true;
}

static int maxfit_table_v_max = -1;
static int *maxfit_table = NULL;

int &geometry_global::maxfit(
		int i, int j)
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

int &geometry_global::maxfit_internal(
		int i, int j)
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

void geometry_global::maxfit_table_init(
		int v_max)
{
	//cout << "maxfit_table_init v_max=" << v_max << endl;

	maxfit_table = NEW_int(v_max * v_max);
	maxfit_table_v_max = v_max;
	maxfit_table_compute();
	//print_integer_matrix_width(cout, maxfit_table, v_max, v_max, v_max, 3);
}

void geometry_global::maxfit_table_reallocate(
		int v_max)
{
	cout << "geometry_global::maxfit_table_reallocate "
			"v_max=" << v_max << endl;

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
	for (i = 0; i < M * M; i++) {
		matrix[i] = 0;
	}
	m = 0;
	for (i = 1; i <= M; i++) {
		//cout << "i=" << i << endl;
		inz = i;
		j = 1;
		while (i >= j) {
			gki = inz/i;
			if (j * (j - 1) / 2 < i * Choose2(gki) + (inz % i) * gki) {
				j++;
			}
			if (j <= M) {
				//cout << "j=" << j << " inz=" << inz << endl;
				m = MAXIMUM(m, inz);
				matrix[(j - 1) * M + i - 1] = inz;
				matrix[(i - 1) * M + j - 1] = inz;
			}
			inz++;
		}
		//print_integer_matrix_width(cout, matrix, M, M, M, 3);
	} // next i
}

int geometry_global::packing_number_via_maxfit(
		int n, int k)
{
	int m;

	if (k == 1) {
		return INT_MAX;
	}
	//cout << "packing_number_via_maxfit n=" << n << " k=" << k << endl;
	m = 1;
	while (maxfit(n, m) >= m * k) {
		m++;
	}
	//cout << "P(" << n << "," << k << ")=" << m - 1 << endl;
	return m - 1;
}



void geometry_global::do_inverse_isomorphism_klein_quadric(
		algebra::field_theory::finite_field *F,
		std::string &inverse_isomorphism_klein_quadric_matrix_A6,
		int verbose_level)
// creates klein_correspondence and orthogonal_geometry::orthogonal objects
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_inverse_isomorphism_klein_quadric" << endl;
	}


	int *A6;
	int sz;

	Int_vec_scan(inverse_isomorphism_klein_quadric_matrix_A6, A6, sz);
	if (sz != 36) {
		cout << "geometry_global::do_inverse_isomorphism_klein_quadric "
				"The input matrix must be of size 6x6" << endl;
		exit(1);
	}


	cout << "A6:" << endl;
	Int_matrix_print(A6, 6, 6);

	projective_geometry::klein_correspondence *Klein;
	orthogonal_geometry::orthogonal *O;


	Klein = NEW_OBJECT(projective_geometry::klein_correspondence);
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
		algebra::field_theory::finite_field *F,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_rank_points_in_PG" << endl;
	}

	int *v;
	int m, n;

	Get_matrix(label, v, m, n);

	if (f_v) {
		cout << "geometry_global::do_rank_points_in_PG coeff: ";
		Int_matrix_print(v, m, n);
		cout << endl;
	}

	long int a;
	int i;

	for (i = 0; i < m; i++) {
		F->Projective_space_basic->PG_element_rank_modified_lint(
				v + i * n, 1, n, a);

		Int_vec_print(cout, v + i * n, n);
		cout << " : " << a << endl;

	}


	FREE_int(v);

}

void geometry_global::do_unrank_points_in_PG(
		algebra::field_theory::finite_field *F,
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

	Get_lint_vector_from_label(text, v, len, 0 /* verbose_level */);

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

		F->Projective_space_basic->PG_element_unrank_modified_lint(
				M + i * d, 1, d, a);

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
		algebra::field_theory::finite_field *F,
		std::string &line_1_basis,
		std::string &line_2_basis,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Line1;
	int *Line2;
	int *A;
	int *B;
	int *C;
	int len, rk;

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

#if 0
	if (f_normalize_from_the_left) {
		cout << "geometry_global::do_intersection_of_two_lines "
				"normalizing from the left" << endl;
		for (i = 3; i < 4; i++) {
			F->Projective_space_basic->PG_element_normalize_from_front(
					C + i * 4, 1, 4);
		}

		cout << "geometry_global::do_intersection_of_two_lines "
				"after normalize from the left:" << endl;
		Int_matrix_print(C + 12, 1, 4);
		cout << "rk=" << rk << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "geometry_global::do_intersection_of_two_lines "
				"normalizing from the right" << endl;
		for (i = 3; i < 4; i++) {
			F->Projective_space_basic->PG_element_normalize(
					C + i * 4, 1, 4);
		}

		cout << "geometry_global::do_intersection_of_two_lines "
				"after normalize from the right:" << endl;
		Int_matrix_print(C + 12, 1, 4);
		cout << "rk=" << rk << endl;

	}
#endif


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
		algebra::field_theory::finite_field *F,
		std::string &line_1_basis,
		std::string &line_2_basis,
		std::string &point,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Line1;
	int *Line2;
	int *Pt;
	int *A;
	int *B;
	int len, rk;

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

#if 0
	if (f_normalize_from_the_left) {
		cout << "geometry_global::do_transversal "
				"normalizing from the left" << endl;
		for (i = 2; i < 4; i++) {
			F->Projective_space_basic->PG_element_normalize_from_front(
					B + i * 4, 1, 4);
		}

		cout << "geometry_global::do_transversal "
				"after normalize from the left:" << endl;
		Int_matrix_print(B + 8, 2, 4);
		cout << "rk=" << rk << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "geometry_global::do_transversal "
				"normalizing from the right" << endl;
		for (i = 2; i < 4; i++) {
			F->Projective_space_basic->PG_element_normalize(
					B + i * 4, 1, 4);
		}

		cout << "geometry_global::do_transversal "
				"after normalize from the right:" << endl;
		Int_matrix_print(B + 8, 2, 4);
		cout << "rk=" << rk << endl;

	}
#endif


	FREE_int(Line1);
	FREE_int(Line2);
	FREE_int(Pt);
	FREE_int(A);
	FREE_int(B);

	if (f_v) {
		cout << "geometry_global::do_transversal done" << endl;
	}
}




void geometry_global::do_cheat_sheet_hermitian(
		algebra::field_theory::finite_field *F,
		int projective_dimension,
		int verbose_level)
// creates a hermitian object
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_hermitian verbose_level="
				<< verbose_level << endl;
	}

	hermitian *H;

	H = NEW_OBJECT(hermitian);

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_hermitian "
				"before H->init" << endl;
	}
	H->init(F, projective_dimension + 1, verbose_level - 2);
	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_hermitian "
				"after H->init" << endl;
	}

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_hermitian "
				"before H->create_latex_report" << endl;
	}
	H->create_latex_report(verbose_level);
	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_hermitian "
				"after H->create_latex_report" << endl;
	}



	FREE_OBJECT(H);

	if (f_v) {
		cout << "geometry_global::do_cheat_sheet_hermitian done" << endl;
	}

}

void geometry_global::do_create_desarguesian_spread(
		algebra::field_theory::finite_field *FQ,
		algebra::field_theory::finite_field *Fq,
		int m,
		int verbose_level)
// creates field_theory::subfield_structure and desarguesian_spread objects
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
	algebra::field_theory::subfield_structure *SubS;
	finite_geometries::desarguesian_spread *D;

	SubS = NEW_OBJECT(algebra::field_theory::subfield_structure);
	if (f_v) {
		cout << "geometry_global::do_create_desarguesian_spread "
				"before SubS->init" << endl;
	}
	SubS->init(FQ, Fq, verbose_level - 2);

	if (f_v) {
		cout << "Field-basis: ";
		Int_vec_print(cout, SubS->Basis, s);
		cout << endl;
	}

	D = NEW_OBJECT(finite_geometries::desarguesian_spread);
	if (f_v) {
		cout << "geometry_global::do_create_desarguesian_spread "
				"before D->init" << endl;
	}
	D->init(n, m, s,
		SubS,
		verbose_level - 2);
	if (f_v) {
		cout << "geometry_global::do_create_desarguesian_spread "
				"after D->init" << endl;
	}


	D->create_latex_report(verbose_level);

	FREE_OBJECT(D);
	FREE_OBJECT(SubS);

	if (f_v) {
		cout << "geometry_global::do_create_desarguesian_spread done" << endl;
	}
}


void geometry_global::create_BLT_point(
		algebra::field_theory::finite_field *F,
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
	other::orbiter_kernel_system::Orbiter->Int_vec->init5(v5, v0, v1, v2, v3, v4);
	if (f_v) {
		cout << "geometry_global::create_BLT_point done" << endl;
	}
}

void geometry_global::create_BLT_point_from_flock(
		algebra::field_theory::finite_field *F,
		int *v5, int a, int b, int c, int verbose_level)
// creates the point (c/2,-c^2/4-ab,1,b,a)
{
	int f_v = (verbose_level >= 1);
	int v0, v1, v2, v3, v4;
	int half, four, quarter, minus_one;

	if (f_v) {
		cout << "geometry_global::create_BLT_point_from_flock" << endl;
	}
	four = 4 % F->p;
	half = F->inverse(2);
	quarter = F->inverse(four);
	minus_one = F->negate(1);
	if (f_v) {
		cout << "geometry_global::create_BLT_point_from_flock "
				"four=" << four << endl;
		cout << "geometry_global::create_BLT_point_from_flock "
				"half=" << half << endl;
		cout << "geometry_global::create_BLT_point_from_flock "
				"quarter=" << quarter << endl;
		cout << "geometry_global::create_BLT_point_from_flock "
				"minus_one=" << minus_one << endl;
	}

	v0 = F->mult(c, half);
	v1 = F->mult(minus_one, F->add(
			F->mult(F->mult(c, c), quarter), F->mult(a, b)));
	v2 = 1;
	v3 = b;
	v4 = a;
	other::orbiter_kernel_system::Orbiter->Int_vec->init5(v5, v0, v1, v2, v3, v4);
	if (f_v) {
		cout << "geometry_global::create_BLT_point_from_flock done" << endl;
	}
}




#if 0
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
		//FQ->print(true /* f_add_mult_table */);
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
			if (false) {
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
				if (false) {
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
		P4->Reporting->print_set(set4, sz4);
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
#endif

void geometry_global::find_secant_lines(
		projective_geometry::projective_space *P,
		long int *set, int set_size,
		long int *lines, int &nb_lines, int max_lines,
		int verbose_level)
// finds the secant lines as an ordered set (secant variety).
// this is done by looping over all pairs of points and creating the
// line that is spanned by the two points.
{
	int f_v = (verbose_level >= 1);
	int i, j, rk, d, h, idx;
	int *M;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "geometry_global::find_secant_lines "
				"set_size=" << set_size << endl;
	}
	d = P->Subspaces->n + 1;
	M = NEW_int(2 * d);
	nb_lines = 0;
	for (i = 0; i < set_size; i++) {
		for (j = i + 1; j < set_size; j++) {
			P->unrank_point(M, set[i]);
			P->unrank_point(M + d, set[j]);
			rk = P->Subspaces->Grass_lines->rank_lint_here(
					M, 0 /* verbose_level */);

			if (!Sorting.lint_vec_search(
					lines, nb_lines, rk, idx, 0)) {
				if (nb_lines == max_lines) {
					cout << "geometry_global::find_secant_lines "
							"nb_lines == max_lines" << endl;
					exit(1);
				}
				for (h = nb_lines; h > idx; h--) {
					lines[h] = lines[h - 1];
				}
				lines[idx] = rk;
				nb_lines++;
			}
		}
	}
	FREE_int(M);
	if (f_v) {
		cout << "geometry_global::find_secant_lines done" << endl;
	}
}

void geometry_global::find_lines_which_are_contained(
		projective_geometry::projective_space *P,
		std::vector<long int> &Points,
		std::vector<long int> &Lines,
		int verbose_level)
// finds all lines which are completely contained in the set of points
// First, finds all lines in the set which lie
// in the hyperplane x_d = 0.
// Then finds all remaining lines.
// The lines are not arranged according to a double six.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = false;
	long int rk;
	long int h, i, j, d, a, b;
	int idx;
	int *M;
	int *M2;
	int *Pts1;
	int *Pts2;
	long int *set1;
	long int *set2;
	int sz1, sz2;
	int *f_taken;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "geometry_global::find_lines_which_are_contained "
				"set_size=" << Points.size() << endl;
	}
	//nb_lines = 0;
	d = P->Subspaces->n + 1;
	M = NEW_int(3 * d);
	M2 = NEW_int(3 * d);
	set1 = NEW_lint(Points.size());
	set2 = NEW_lint(Points.size());
	sz1 = 0;
	sz2 = 0;
	for (i = 0; i < Points.size(); i++) {
		P->unrank_point(M, Points[i]);
		if (f_vvv) {
			cout << Points[i] << " : ";
			Int_vec_print(cout, M, d);
			cout << endl;
		}
		if (M[d - 1] == 0) {
			set1[sz1++] = Points[i];
		}
		else {
			set2[sz2++] = Points[i];
		}
	}

	// set1 is the set of points whose last coordinate is zero.
	// set2 is the set of points whose last coordinate is nonzero.
	Sorting.lint_vec_heapsort(set1, sz1);
	Sorting.lint_vec_heapsort(set2, sz2);

	if (f_vv) {
		cout << "geometry_global::find_lines_which_are_contained "
				"sz1=" << sz1 << " sz2=" << sz2 << endl;
	}


	// find all secants in the hyperplane:
	long int *secants;
	int n2, nb_secants;

	n2 = (sz1 * (sz1 - 1)) >> 1;
	// n2 is an upper bound on the number of secant lines

	secants = NEW_lint(n2);


	if (f_v) {
		cout << "geometry_global::find_lines_which_are_contained "
				"before find_secant_lines" << endl;
	}

	find_secant_lines(
			P,
			set1, sz1,
			secants, nb_secants, n2,
			0/*verbose_level - 3*/);

	if (f_v) {
		cout << "geometry_global::find_lines_which_are_contained "
				"after find_secant_lines" << endl;
	}

	if (f_vv) {
		cout << "geometry_global::find_lines_which_are_contained "
				"we found " << nb_secants
				<< " secants in the hyperplane" << endl;
	}

	// first we test the secants and
	// find those which are lines on the surface:
	if (f_vv) {
		cout << "geometry_global::find_lines_which_are_contained "
				"testing secants, nb_secants=" << nb_secants << endl;
	}

	//nb_lines = 0;
	for (i = 0; i < nb_secants; i++) {
		rk = secants[i];
		P->Subspaces->Grass_lines->unrank_lint_here(
				M, rk, 0 /* verbose_level */);
		if (f_vvv) {
			cout << "testing secant " << i << " / " << nb_secants
					<< " which is line " << rk << ":" << endl;
			Int_matrix_print(M, 2, d);
		}

		int coeffs[2];

		// loop over all points on the line:
		for (a = 0; a < P->Subspaces->q + 1; a++) {

			// unrank a point on the projective line:
			P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
					coeffs, 1, 2, a);
			Int_vec_copy(M, M2, 2 * d);

			// map the point to the line at hand.
			// form the linear combination:
			// coeffs[0] * row0 of M2 + coeffs[1] * row1 of M2:
			for (h = 0; h < d; h++) {
				M2[2 * d + h] = P->Subspaces->F->add(
						P->Subspaces->F->mult(coeffs[0], M2[0 * d + h]),
						P->Subspaces->F->mult(coeffs[1], M2[1 * d + h]));
			}

			// rank the test point and see
			// if it belongs to the surface:
			P->Subspaces->F->Projective_space_basic->PG_element_rank_modified_lint(
					M2 + 2 * d, 1, d, b);
			if (!Sorting.lint_vec_search(set1, sz1, b, idx, 0)) {
				break;
			}
		}
		if (a == P->Subspaces->q + 1) {
			// all q + 1 points of the secant line
			// belong to the surface, so we
			// found a line on the surface in the hyperplane.
			//lines[nb_lines++] = rk;
			if (f_vv) {
				cout << "secant " << i << " / " << nb_secants
						<< " of rank " << rk << " is contained, adding" << endl;
			}
			Lines.push_back(rk);
		}
	}
	FREE_lint(secants);

	if (f_v) {
		cout << "geometry_global::find_lines_which_are_contained "
				"We found " << Lines.size() << " in the hyperplane" << endl;
		//lint_vec_print(cout, lines, nb_lines);
		cout << endl;
	}



	Pts1 = NEW_int(sz1 * d);
	Pts2 = NEW_int(sz2 * d);

	for (i = 0; i < sz1; i++) {
		P->unrank_point(Pts1 + i * d, set1[i]);
	}
	for (i = 0; i < sz2; i++) {
		P->unrank_point(Pts2 + i * d, set2[i]);
	}

	if (f_vv) {
		cout << "geometry_global::find_lines_which_are_contained "
				"checking lines through points of the hyperplane, sz1=" << sz1 << endl;
	}

	f_taken = NEW_int(sz2);
	for (i = 0; i < sz1; i++) {
		if (f_vvv) {
			cout << "geometry_global::find_lines_which_are_contained "
					"checking lines through hyperplane point " << i
					<< " / " << sz1 << ":" << endl;
		}

		// consider a point P1 on the surface and in the hyperplane

		Int_vec_zero(f_taken, sz2);
		for (j = 0; j < sz2; j++) {
			if (f_taken[j]) {
				continue;
			}
			if (f_vvv) {
				cout << "geometry_global::find_lines_which_are_contained "
						"i=" << i << " j=" << j << " / " << sz2 << ":" << endl;
			}

			// consider a point P2 on the surface
			// but not in the hyperplane:

			Int_vec_copy(Pts1 + i * d, M, d);
			Int_vec_copy(Pts2 + j * d, M + d, d);

			f_taken[j] = true;

			if (f_vvv) {
				Int_matrix_print(M, 2, d);
			}

			rk = P->Subspaces->Grass_lines->rank_lint_here(
					M, 0 /* verbose_level */);
			if (f_vvv) {
				cout << "geometry_global::find_lines_which_are_contained "
						"line rk=" << rk << ":" << endl;
			}

			// test the q-1 points on the line through the P1 and P2
			// (but excluding P1 and P2 themselves):
			for (a = 1; a < P->Subspaces->q; a++) {
				Int_vec_copy(M, M2, 2 * d);

				// form the linear combination P3 = P1 + a * P2:
				for (h = 0; h < d; h++) {
					M2[2 * d + h] =
							P->Subspaces->F->add(
								M2[0 * d + h],
								P->Subspaces->F->mult(a, M2[1 * d + h]));
				}
				// row 2 of M2 contains the coordinates of the point P3:
				P->Subspaces->F->Projective_space_basic->PG_element_rank_modified_lint(
						M2 + 2 * d, 1, d, b);
				if (!Sorting.lint_vec_search(set2, sz2, b, idx, 0)) {
					break;
				}
				else {
					if (f_vvv) {
						cout << "eliminating point " << idx << endl;
					}
					// we don't need to consider this point for P2:
					f_taken[idx] = true;
				}
			}
			if (a == P->Subspaces->q) {
				// The line P1P2 is contained in the surface.
				// Add it to lines[]
#if 0
				if (nb_lines == max_lines) {
					cout << "geometry_global::find_lines_which_are_"
							"contained nb_lines == max_lines" << endl;
					exit(1);
				}
#endif
				//lines[nb_lines++] = rk;
				if (f_vvv) {
					cout << "adding line " << rk << " nb_lines="
							<< Lines.size() << endl;
				}
				Lines.push_back(rk);
				if (f_vvv) {
					cout << "adding line " << rk << " nb_lines="
							<< Lines.size() << " done" << endl;
				}
			}
		}
	}
	FREE_int(M);
	FREE_int(M2);
	FREE_lint(set1);
	FREE_lint(set2);
	FREE_int(Pts1);
	FREE_int(Pts2);
	FREE_int(f_taken);

	if (f_v) {
		cout << "geometry_global::find_lines_which_are_contained done" << endl;
	}
}

void geometry_global::make_restricted_incidence_matrix(
		geometry::projective_geometry::projective_space *P,
		int type_i, int type_j,
		std::string &row_objects,
		std::string &col_objects,
		std::string &file_name,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::make_restricted_incidence_matrix" << endl;
	}


	long int *Row_objects;
	int nb_row_objects;
	long int *Col_objects;
	int nb_col_objects;
	int i, j;

	int *M;

	Get_lint_vector_from_label(row_objects, Row_objects, nb_row_objects, 0 /* verbose_level */);
	Get_lint_vector_from_label(col_objects, Col_objects, nb_col_objects, 0 /* verbose_level */);

	M = NEW_int(nb_row_objects * nb_col_objects);
	Int_vec_zero(M, nb_row_objects * nb_col_objects);

	for (i = 0; i < nb_row_objects; i++) {

		for (j = 0; j < nb_col_objects; j++) {

			if (P->Subspaces->incidence_test_for_objects_of_type_ij(
				type_i, type_j, Row_objects[i], Col_objects[j],
				0 /* verbose_level */)) {
				M[i * nb_col_objects + j] = 1;
			}
		}
	}

	other::orbiter_kernel_system::file_io Fio;
	string fname_csv;
	string fname_inc;

	fname_csv = file_name + ".csv";
	fname_inc = file_name + ".inc";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname_csv, M, nb_row_objects, nb_col_objects);

	if (f_v) {
		cout << "written file " << fname_csv << " of size "
				<< Fio.file_size(fname_csv) << endl;
	}

	Fio.write_incidence_matrix_to_file(
			fname_inc,
		M, nb_row_objects, nb_col_objects, 0 /*verbose_level*/);

	if (f_v) {
		cout << "written file " << fname_inc << " of size "
				<< Fio.file_size(fname_inc) << endl;
	}

	FREE_int(M);

	if (f_v) {
		cout << "geometry_global::make_restricted_incidence_matrix done" << endl;
	}
}


void geometry_global::plane_intersection_type(
		geometry::projective_geometry::projective_space *P,
		std::string &input,
		int threshold,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::plane_intersection_type" << endl;
	}
	long int *Set;
	int sz;

	Get_lint_vector_from_label(input, Set, sz, 0 /* verbose_level */);


	if (f_v) {
		cout << "geometry_global::plane_intersection_type "
				"before plane_intersection_type" << endl;
	}

	geometry::other_geometry::intersection_type *Int_type;

	P->plane_intersection_type(
		Set, sz, threshold,
		Int_type,
		verbose_level);

	if (f_v) {
		cout << "geometry_global::plane_intersection_type "
				"after plane_intersection_type" << endl;
	}

	cout << "geometry_global::plane_intersection_type "
			"intersection numbers: ";
	Int_vec_print(cout,
			Int_type->the_intersection_type,
			Int_type->highest_intersection_number + 1);
	cout << endl;

	if (f_v) {
		cout << "geometry_global::plane_intersection_type "
				"highest weight objects: " << endl;
		Lint_vec_print(cout,
				Int_type->Highest_weight_objects,
				Int_type->nb_highest_weight_objects);
		cout << endl;
	}

	if (f_v) {
		cout << "geometry_global::plane_intersection_type "
				"Intersection_sets: " << endl;
		Int_matrix_print(Int_type->Intersection_sets,
				Int_type->nb_highest_weight_objects,
				Int_type->highest_intersection_number);
	}

	if (f_v) {
		cout << "geometry_global::plane_intersection_type "
				"Intersection_sets sorted: " << endl;
		Int_matrix_print(Int_type->M->M,
				Int_type->nb_highest_weight_objects,
				Int_type->highest_intersection_number);
	}

	string fname;
	other::data_structures::string_tools ST;

	fname.assign(input);
	ST.chop_off_extension(fname);
	fname += "_highest_weight_objects.csv";

	Int_type->M->write_csv(fname, verbose_level);


	FREE_OBJECT(Int_type);

	if (f_v) {
		cout << "geometry_global::plane_intersection_type done" << endl;
	}
}


void geometry_global::plane_intersection_type_of_klein_image(
		geometry::projective_geometry::projective_space *P,
		std::string &input,
		int threshold,
		int verbose_level)
// creates a projective_space object P5
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::plane_intersection_type_of_klein_image" << endl;
	}
	long int *Lines;
	int nb_lines;

	Get_lint_vector_from_label(input, Lines, nb_lines, 0 /* verbose_level */);

	//int *intersection_type;
	//int highest_intersection_number;

	geometry::projective_geometry::projective_space *P5;

	P5 = NEW_OBJECT(geometry::projective_geometry::projective_space);

	int f_init_incidence_structure = true;

	if (f_v) {
		cout << "geometry_global::plane_intersection_type_of_klein_image "
				"before P5->projective_space_init" << endl;
	}
	P5->projective_space_init(5, P->Subspaces->F,
			f_init_incidence_structure,
			verbose_level);
	if (f_v) {
		cout << "geometry_global::plane_intersection_type_of_klein_image "
				"after P5->projective_space_init" << endl;
	}

	if (f_v) {
		cout << "geometry_global::plane_intersection_type_of_klein_image "
				"before plane_intersection_type_of_klein_image" << endl;
	}

	geometry::other_geometry::intersection_type *Int_type;

	P->Subspaces->Grass_lines->plane_intersection_type_of_klein_image(
			P /* P3 */,
			P5,
			Lines, nb_lines, threshold,
			Int_type,
			verbose_level);

	if (f_v) {
		cout << "geometry_global::plane_intersection_type_of_klein_image "
				"after plane_intersection_type_of_klein_image" << endl;
	}

	if (f_v) {
		cout << "geometry_global::plane_intersection_type_of_klein_image "
				"intersection numbers: ";
		Int_vec_print(cout,
				Int_type->the_intersection_type,
				Int_type->highest_intersection_number + 1);
		cout << endl;
	}

	if (f_v) {
		cout << "geometry_global::plane_intersection_type_of_klein_image "
				"highest weight objects: " << endl;
		Lint_vec_print(cout,
				Int_type->Highest_weight_objects,
				Int_type->nb_highest_weight_objects);
		cout << endl;
	}

	if (f_v) {
		cout << "geometry_global::plane_intersection_type_of_klein_image "
				"Intersection_sets: " << endl;
		Int_matrix_print(Int_type->Intersection_sets,
				Int_type->nb_highest_weight_objects,
				Int_type->highest_intersection_number);
	}

	if (f_v) {
		cout << "geometry_global::plane_intersection_type_of_klein_image "
				"Intersection_sets sorted: " << endl;
		Int_matrix_print(Int_type->M->M,
				Int_type->nb_highest_weight_objects,
				Int_type->highest_intersection_number);
	}

	string fname;
	other::data_structures::string_tools ST;

	fname.assign(input);
	ST.chop_off_extension(fname);
	fname += "_highest_weight_objects.csv";

	Int_type->M->write_csv(fname, verbose_level);


	FREE_OBJECT(Int_type);

	FREE_OBJECT(P5);

	if (f_v) {
		cout << "geometry_global::plane_intersection_type_of_klein_image done" << endl;
	}
}

void geometry_global::conic_type(
		geometry::projective_geometry::projective_space *P,
		int threshold,
		std::string &set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "geometry_global::conic_type" << endl;
	}

	long int *Pts;
	int nb_pts;

	Get_lint_vector_from_label(set_text, Pts, nb_pts, 0 /* verbose_level */);


	if (f_v) {
		cout << "geometry_global::conic_type "
				"before PA->conic_type" << endl;
	}

	conic_type2(P, Pts, nb_pts, threshold, verbose_level);

	if (f_v) {
		cout << "geometry_global::conic_type "
				"after PA->conic_type" << endl;
	}

	if (f_v) {
		cout << "geometry_global::conic_type done" << endl;
	}
}


void geometry_global::conic_type2(
		geometry::projective_geometry::projective_space *P,
		long int *Pts, int nb_pts, int threshold,
		int verbose_level)
// this function is too specialized. It assumes 11 points
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::conic_type2 "
				"threshold = " << threshold << endl;
	}


	long int **Pts_on_conic;
	int **Conic_eqn;
	int *nb_pts_on_conic;
	int len;
	int h;


	if (f_v) {
		cout << "geometry_global::conic_type2 "
				"before P->conic_type" << endl;
	}

	P->Plane->conic_type(Pts, nb_pts,
			threshold,
			Pts_on_conic, Conic_eqn, nb_pts_on_conic, len,
			verbose_level);

	if (f_v) {
		cout << "geometry_global::conic_type2 "
				"after P->conic_type" << endl;
	}


	cout << "We found the following conics:" << endl;
	for (h = 0; h < len; h++) {
		cout << h << " : " << nb_pts_on_conic[h] << " : ";
		Int_vec_print(cout, Conic_eqn[h], 6);
		cout << " : ";
		Lint_vec_print(cout, Pts_on_conic[h], nb_pts_on_conic[h]);
		cout << endl;
	}

	if (f_v) {
		cout << "geometry_global::conic_type2 computing intersection "
				"types with bisecants of the first 11 points:" << endl;
	}
	int Line_P1[55];
	int Line_P2[55];
	int P1, P2;
	long int p1, p2, line_rk;
	long int *pts_on_line;
	long int pt;
	int *Conic_line_intersection_sz;
	int cnt;
	int i, j, q, u, v;
	int nb_pts_per_line;

	q = P->Subspaces->F->q;
	nb_pts_per_line = q + 1;
	pts_on_line = NEW_lint(55 * nb_pts_per_line);

	cnt = 0;
	for (i = 0; i < 11; i++) {
		for (j = i + 1; j < 11; j++) {
			Line_P1[cnt] = i;
			Line_P2[cnt] = j;
			cnt++;
		}
	}
	if (cnt != 55) {
		cout << "cnt != 55" << endl;
		cout << "cnt = " << cnt << endl;
		exit(1);
	}
	for (u = 0; u < 55; u++) {
		P1 = Line_P1[u];
		P2 = Line_P2[u];
		p1 = Pts[P1];
		p2 = Pts[P2];
		line_rk = P->Subspaces->line_through_two_points(p1, p2);
		P->Subspaces->create_points_on_line(line_rk,
				pts_on_line + u * nb_pts_per_line,
				0 /*verbose_level*/);
	}

	Conic_line_intersection_sz = NEW_int(len * 55);
	Int_vec_zero(Conic_line_intersection_sz, len * 55);

	for (h = 0; h < len; h++) {
		for (u = 0; u < 55; u++) {
			for (v = 0; v < nb_pts_per_line; v++) {
				if (P->Plane->test_if_conic_contains_point(
						Conic_eqn[h],
						pts_on_line[u * nb_pts_per_line + v])) {

					Conic_line_intersection_sz[h * 55 + u]++;
				}

			}
		}
	}

	other::data_structures::sorting Sorting;
	int idx;

	cout << "We found the following conics and their "
			"intersections with the 55 bisecants:" << endl;
	for (h = 0; h < len; h++) {
		cout << h << " : " << nb_pts_on_conic[h] << " : ";
		Int_vec_print(cout, Conic_eqn[h], 6);
		cout << " : ";
		Int_vec_print_fully(cout, Conic_line_intersection_sz + h * 55, 55);
		cout << " : ";
		Lint_vec_print(cout, Pts_on_conic[h], nb_pts_on_conic[h]);
		cout << " : ";
		cout << endl;
	}

	for (u = 0; u < 55; u++) {
		cout << "line " << u << " : ";
		int str[55];

		Int_vec_zero(str, 55);
		for (v = 0; v < nb_pts; v++) {
			pt = Pts[v];
			if (Sorting.lint_vec_search_linear(
					pts_on_line + u * nb_pts_per_line,
					nb_pts_per_line, pt, idx)) {
				str[v] = 1;
			}
		}
		Int_vec_print_fully(cout, str, 55);
		cout << endl;
	}

	if (f_v) {
		cout << "geometry_global::conic_type2 done" << endl;
	}

}


void geometry_global::do_rank_lines_in_PG(
		geometry::projective_geometry::projective_space *P,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_rank_lines_in_PG" << endl;
	}

	int *v;
	int m, n;

	Get_matrix(label, v, m, n);

	if (f_v) {
		cout << "geometry_global::do_rank_lines_in_PG v: ";
		Int_matrix_print(v, m, n);
		cout << endl;
	}

	if (n != 2 * (P->Subspaces->n + 1)) {
		cout << "geometry_global::do_rank_lines_in_PG "
				"n != 2 * (P->n + 1)" << endl;
		exit(1);
	}

	long int a;
	int i;

	for (i = 0; i < m; i++) {


		a = P->rank_line(v + i * n);

		Int_matrix_print(v + i * n, 2, P->Subspaces->n + 1);
		cout << "has rank " << a << endl;

	}


	FREE_int(v);

}

void geometry_global::do_unrank_lines_in_PG(
		geometry::projective_geometry::projective_space *P,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_unrank_lines_in_PG" << endl;
	}

	int len;
	int *basis;

	long int *v;
	int sz;

	Get_lint_vector_from_label(label, v, sz, 0 /* verbose_level */);

	if (f_v) {
		cout << "geometry_global::do_unrank_lines_in_PG v = ";
		Lint_vec_print(cout, v, sz);
		cout << endl;
	}



	len = 2 * (P->Subspaces->n + 1);

	basis = NEW_int(len);

	int i;

	for (i = 0; i < sz; i++) {


		P->unrank_line(basis, v[i]);


		cout << v[i] << " = " << endl;
		Int_matrix_print(basis, 2, P->Subspaces->n + 1);
		cout << endl;

	}


	FREE_lint(v);
	FREE_int(basis);

	if (f_v) {
		cout << "geometry_global::do_unrank_lines_in_PG done" << endl;
	}
}


void geometry_global::do_points_on_lines_in_PG(
		geometry::projective_geometry::projective_space *P,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_points_on_lines_in_PG" << endl;
	}

	int len;
	int *basis;
	int *w;

	long int *v;
	int sz;

	Get_lint_vector_from_label(label, v, sz, 0 /* verbose_level */);

	if (f_v) {
		cout << "geometry_global::do_points_on_lines_in_PG v = ";
		Lint_vec_print(cout, v, sz);
		cout << endl;
	}


	int d, q;
	long int *Pts;

	q = P->Subspaces->F->q;
	d = P->Subspaces->n + 1;

	len = 2 * d;

	basis = NEW_int(len);
	w = NEW_int(d);
	Pts = NEW_lint(sz * (q + 1));

	int i;

	for (i = 0; i < sz; i++) {


		P->unrank_line(basis, v[i]);


		cout << v[i] << " = " << endl;
		Int_matrix_print(basis, 2, d);
		cout << endl;


		int coeffs[2];
		int a;

		// loop over all points on the line:
		for (a = 0; a < q + 1; a++) {

			// unrank a point on the projective line:
			P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
					coeffs, 1, 2, a);


			P->Subspaces->F->Linear_algebra->mult_vector_from_the_left(
					coeffs, basis, w, 2, d);



			// rank the test point and see
			// if it belongs to the surface:
			P->Subspaces->F->Projective_space_basic->PG_element_rank_modified_lint(
					w, 1, d, Pts[i * (q + 1) + a]);


		}

	}
	if (f_v) {
		cout << "geometry_global::do_points_on_lines_in_PG Pts on lines:" << endl;
		Lint_matrix_print(Pts, sz, q + 1);
	}


	FREE_lint(v);
	FREE_int(basis);
	FREE_int(w);
	FREE_lint(Pts);

	if (f_v) {
		cout << "geometry_global::do_points_on_lines_in_PG done" << endl;
	}
}




void geometry_global::do_cone_over(
		int n,
		algebra::field_theory::finite_field *F,
	long int *set_in, int set_size_in,
	long int *&set_out, int &set_size_out,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_cone_over" << endl;
	}
	//projective_space *P1;
	//projective_space *P2;
	int *v;
	int d = n + 2;
	int h, u, cnt;
	long int a, b;

#if 0
	P1 = NEW_OBJECT(projective_space);
	P2 = NEW_OBJECT(projective_space);

	P1->init(n, this,
		false /* f_init_incidence_structure */,
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);

	P2->init(n + 1, this,
		false /* f_init_incidence_structure */,
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);
#endif

	v = NEW_int(d);

	set_size_out = 1 + F->q * set_size_in;
	set_out = NEW_lint(set_size_out);
	cnt = 0;

	// create the vertex:
	Int_vec_zero(v, d);
	v[d - 1] = 1;
	//b = P2->rank_point(v);
	F->Projective_space_basic->PG_element_rank_modified_lint(
			v, 1, n + 2, b);
	set_out[cnt++] = b;


	// for each point, create the generator
	// which is the line connecting the point and the vertex
	// since we have created the vertex already,
	// we only need to create q points per line:

	for (h = 0; h < set_size_in; h++) {
		a = set_in[h];
		for (u = 0; u < F->q; u++) {
			//P1->unrank_point(v, a);
			F->Projective_space_basic->PG_element_unrank_modified_lint(
					v, 1, n + 1, a);

			v[d - 1] = u;

			//b = P2->rank_point(v);
			F->Projective_space_basic->PG_element_rank_modified_lint(
					v, 1, n + 2, b);

			set_out[cnt++] = b;
		}
	}

	if (cnt != set_size_out) {
		cout << "geometry_global::do_cone_over cnt != set_size_out" << endl;
		exit(1);
	}

	FREE_int(v);
	//FREE_OBJECT(P1);
	//FREE_OBJECT(P2);
	if (f_v) {
		cout << "geometry_global::do_cone_over done" << endl;
	}
}


void geometry_global::do_blocking_set_family_3(
		int n,
		algebra::field_theory::finite_field *F,
	long int *set_in, int set_size,
	long int *&the_set_out, int &set_size_out,
	int verbose_level)
// creates projective_space PG(2,q)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::do_blocking_set_family_3" << endl;
	}
	geometry::projective_geometry::projective_space *P;
	int h;

	if (n != 2) {
		cout << "geometry_global::do_blocking_set_family_3 "
				"we need n = 2" << endl;
		exit(1);
	}
	if (ODD(F->q)) {
		cout << "geometry_global::do_blocking_set_family_3 "
				"we need q even" << endl;
		exit(1);
	}
	if (set_size != F->q + 2) {
		cout << "geometry_global::do_blocking_set_family_3 "
				"we need set_size == q + 2" << endl;
		exit(1);
	}
	P = NEW_OBJECT(geometry::projective_geometry::projective_space);

	P->projective_space_init(n, F,
		false /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);


	int *idx;
	int p_idx[4];
	int line[6];
	int diag_pts[3];
	int diag_line;
	int nb, pt, sz;
	int i, j;
	int basis[6];
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	other::data_structures::fancy_set *S;

	S = NEW_OBJECT(other::data_structures::fancy_set);

	S->init(P->Subspaces->N_lines, 0);
	S->k = 0;

	idx = NEW_int(set_size);

#if 1
	while (true) {
		cout << "choosing random permutation" << endl;
		Combi.Permutations->random_permutation(idx, set_size);

		cout << idx[0] << ", ";
		cout << idx[1] << ", ";
		cout << idx[2] << ", ";
		cout << idx[3] << endl;

		for (i = 0; i < 4; i++) {
			p_idx[i] = set_in[idx[i]];
		}

		line[0] = P->line_through_two_points(p_idx[0], p_idx[1]);
		line[1] = P->line_through_two_points(p_idx[0], p_idx[2]);
		line[2] = P->line_through_two_points(p_idx[0], p_idx[3]);
		line[3] = P->line_through_two_points(p_idx[1], p_idx[2]);
		line[4] = P->line_through_two_points(p_idx[1], p_idx[3]);
		line[5] = P->line_through_two_points(p_idx[2], p_idx[3]);
		diag_pts[0] = P->intersection_of_two_lines(line[0], line[5]);
		diag_pts[1] = P->intersection_of_two_lines(line[1], line[4]);
		diag_pts[2] = P->intersection_of_two_lines(line[2], line[3]);

		diag_line = P->line_through_two_points(diag_pts[0], diag_pts[1]);
		if (diag_line != P->line_through_two_points(diag_pts[0], diag_pts[2])) {
			cout << "diagonal points not collinear!" << endl;
			exit(1);
		}
		P->unrank_line(basis, diag_line);
		Int_matrix_print(basis, 2, 3);
		nb = 0;
		for (i = 0; i < set_size; i++) {
			pt = set_in[i];
			if (P->Subspaces->is_incident(pt, diag_line)) {
				nb++;
			}
		}
		cout << "nb=" << nb << endl;
		if (nb == 0) {
			cout << "the diagonal line is external!" << endl;
			break;
		}
	} // while
#endif

#if 0
	int fundamental_quadrangle[4] = {0,1,2,3};
	int basis[6];

	for (i = 0; i < 4; i++) {
		if (!int_vec_search_linear(set_in, set_size, fundamental_quadrangle[i], j)) {
			cout << "the point " << fundamental_quadrangle[i] << " is not contained in the hyperoval" << endl;
			exit(1);
		}
		idx[i] = j;
	}
	cout << "the fundamental quadrangle is contained, the positions are " << endl;
		cout << idx[0] << ", ";
		cout << idx[1] << ", ";
		cout << idx[2] << ", ";
		cout << idx[3] << endl;

		for (i = 0; i < 4; i++) {
			p_idx[i] = set_in[idx[i]];
		}

		line[0] = P->line_through_two_points(p_idx[0], p_idx[1]);
		line[1] = P->line_through_two_points(p_idx[0], p_idx[2]);
		line[2] = P->line_through_two_points(p_idx[0], p_idx[3]);
		line[3] = P->line_through_two_points(p_idx[1], p_idx[2]);
		line[4] = P->line_through_two_points(p_idx[1], p_idx[3]);
		line[5] = P->line_through_two_points(p_idx[2], p_idx[3]);
		diag_pts[0] = P->line_intersection(line[0], line[5]);
		diag_pts[1] = P->line_intersection(line[1], line[4]);
		diag_pts[2] = P->line_intersection(line[2], line[3]);

		diag_line = P->line_through_two_points(diag_pts[0], diag_pts[1]);
		cout << "The diagonal line is " << diag_line << endl;

		P->unrank_line(basis, diag_line);
		int_matrix_print(basis, 2, 3);

		if (diag_line != P->line_through_two_points(diag_pts[0], diag_pts[2])) {
			cout << "diagonal points not collinear!" << endl;
			exit(1);
		}
		nb = 0;
		for (i = 0; i < set_size; i++) {
			pt = set_in[i];
			if (P->Incidence[pt * P->N_lines + diag_line]) {
				nb++;
			}
		}
		cout << "nb=" << nb << endl;
		if (nb == 0) {
			cout << "the diagonal line is external!" << endl;
		}
		else {
			cout << "error: the diagonal line is not external" << endl;
			exit(1);
		}

#endif

	S->add_element(diag_line);
	for (i = 4; i < set_size; i++) {
		pt = set_in[idx[i]];
		for (j = 0; j < P->Subspaces->r; j++) {
			h = P->Subspaces->Implementation->lines_on_point(pt, j);
			if (!S->is_contained(h)) {
				S->add_element(h);
			}
		}
	}

	cout << "we created a blocking set of lines of "
			"size " << S->k << ":" << endl;
	Lint_vec_print(cout, S->set, S->k);
	cout << endl;


	int *pt_type;

	pt_type = NEW_int(P->Subspaces->N_points);

	P->Subspaces->Implementation->point_types_of_line_set(
			S->set, S->k, pt_type, 0);

	other::data_structures::tally C;

	C.init(pt_type, P->Subspaces->N_points, false, 0);


	cout << "the point types are:" << endl;
	C.print_bare(false /*f_backwards*/);
	cout << endl;

#if 0
	for (i = 0; i <= P->N_points; i++) {
		if (pt_type[i]) {
			cout << i << "^" << pt_type[i] << " ";
			}
		}
	cout << endl;
#endif

	sz = ((F->q * F->q) >> 1) + ((3 * F->q) >> 1) - 4;

	if (S->k != sz) {
		cout << "the size does not match the expected size" << endl;
		exit(1);
	}

	cout << "the size is OK" << endl;

	the_set_out = NEW_lint(sz);
	set_size_out = sz;

	for (i = 0; i < sz; i++) {
		j = S->set[i];
		the_set_out[i] = P->Subspaces->Standard_polarity->Hyperplane_to_point[j];
	}



	FREE_OBJECT(P);
}





void geometry_global::create_orthogonal(
		algebra::field_theory::finite_field *F,
		int epsilon, int n,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
	int verbose_level)
// creates a quadratic form
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::create_orthogonal" << endl;
	}
	long int i, j;
	int d = n + 1;
	int *v;
	geometry::other_geometry::geometry_global Gg;
	orthogonal_geometry::quadratic_form *Quadratic_form;

	nb_pts = Gg.nb_pts_Qepsilon(epsilon, n, F->q);

	v = NEW_int(d);
	Pts = NEW_lint(nb_pts);


	Quadratic_form = NEW_OBJECT(orthogonal_geometry::quadratic_form);

	if (f_v) {
		cout << "geometry_global::create_orthogonal "
				"before Quadratic_form->init" << endl;
	}
	Quadratic_form->init(epsilon, n, F, verbose_level);
	if (f_v) {
		cout << "geometry_global::create_orthogonal "
				"after Quadratic_form->init" << endl;
	}



	if (f_v) {
		cout << "orthogonal rank : point : projective rank" << endl;
	}
	for (i = 0; i < nb_pts; i++) {

		Quadratic_form->unrank_point(v, i, 0 /* verbose_level */);

		F->Projective_space_basic->PG_element_rank_modified(
				v, 1, d, j);

		Pts[i] = j;

		if (f_v) {
			cout << setw(4) << i << " : ";
			Int_vec_print(cout, v, d);
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


	algebra::basic_algebra::algebra_global AG;

	label_txt = "Q" + AG.plus_minus_string(epsilon) + "_" + std::to_string(n) + "_" + std::to_string(F->q);
	label_tex = "Q" + AG.plus_minus_string(epsilon) + "\\_" + std::to_string(n) + "\\_" + std::to_string(F->q);


	FREE_OBJECT(Quadratic_form);
	FREE_int(v);
	//FREE_int(L);
}


void geometry_global::create_hermitian(
		algebra::field_theory::finite_field *F,
		int n,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
	int verbose_level)
// creates hermitian
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::create_hermitian" << endl;
	}
	long int i, j;
	int d = n + 1;
	int *v;
	geometry::other_geometry::hermitian *H;

	H = NEW_OBJECT(geometry::other_geometry::hermitian);
	H->init(F, d, verbose_level - 1);

	nb_pts = H->cnt_Sbar[d];

	v = NEW_int(d);
	Pts = NEW_lint(nb_pts);

	if (f_v) {
		cout << "hermitian rank : point : projective rank" << endl;
	}
	for (i = 0; i < nb_pts; i++) {
		H->Sbar_unrank(v, d, i, 0 /*verbose_level*/);
		F->Projective_space_basic->PG_element_rank_modified(
				v, 1, d, j);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			Int_vec_print(cout, v, d);
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


	label_txt = "H_" + std::to_string(n) + "_" + std::to_string(F->q);
	label_tex = "H\\_" + std::to_string(n) + "\\_" + std::to_string(F->q);


	FREE_int(v);
	FREE_OBJECT(H);
	//FREE_int(L);
}

void geometry_global::create_ttp_code(
		algebra::field_theory::finite_field *FQ,
		algebra::field_theory::finite_field *Fq_subfield,
	int f_construction_A, int f_hyperoval, int f_construction_B,
	std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
// creates a projective_space
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::create_ttp_code" << endl;
	}
	geometry::projective_geometry::projective_space *P;
	long int i, j, d;
	int *v;
	int *H_subfield;
	int m, n;
	int f_elements_exponential = true;
	string symbol_for_print_subfield;
	combinatorics::coding_theory::ttp_codes Ttp_codes;

	if (f_v) {
		cout << "geometry_global::create_ttp_code" << endl;
	}

	symbol_for_print_subfield.assign("\\alpha");

	Ttp_codes.twisted_tensor_product_codes(
		FQ, Fq_subfield,
		f_construction_A, f_hyperoval,
		f_construction_B,
		H_subfield, m, n,
		verbose_level - 2);

	if (f_v) {
		cout << "H_subfield:" << endl;
		cout << "m=" << m << endl;
		cout << "n=" << n << endl;
		Int_vec_print_integer_matrix_width(
				cout, H_subfield, m, n, n, 2);
		//f.latex_matrix(cout, f_elements_exponential,
		//symbol_for_print_subfield, H_subfield, m, n);
	}

	d = m;
	P = NEW_OBJECT(geometry::projective_geometry::projective_space);


	P->projective_space_init(d - 1, Fq_subfield,
		false /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	nb_pts = n;

	if (f_v) {
		cout << "H_subfield:" << endl;
		//print_integer_matrix_width(cout, H_subfield, m, n, n, 2);
		Fq_subfield->Io->latex_matrix(cout, f_elements_exponential,
			symbol_for_print_subfield, H_subfield, m, n);
	}

	v = NEW_int(d);
	Pts = NEW_lint(nb_pts);

	if (f_v) {
		cout << "i : point : projective rank" << endl;
	}
	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < d; j++) {
			v[j] = H_subfield[j * n + i];
		}
		j = P->rank_point(v);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			Int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
		}
	}

#if 0
	cout << "list of points for the ttp code:" << endl;
	cout << N << endl;
	for (i = 0; i < N; i++) {
		cout << L[i] << " ";
		}
	cout << endl;
#endif

	if (f_construction_A) {
		if (f_hyperoval) {
			fname = "ttp_code_Ah_" + std::to_string(Fq_subfield->q) + ".txt";
		}
		else {
			fname = "ttp_code_A_" + std::to_string(Fq_subfield->q) + ".txt";
		}
	}
	else if (f_construction_B) {
		fname = "ttp_code_B_" + std::to_string(Fq_subfield->q) + ".txt";
	}
	//write_set_to_file(fname, L, N, verbose_level);

	FREE_OBJECT(P);
	FREE_int(v);
	FREE_int(H_subfield);
}





void geometry_global::create_segre_variety(
		algebra::field_theory::finite_field *F,
		int a, int b,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
	int verbose_level)
// The Segre map goes from PG(a,q) cross PG(b,q) to PG((a+1)*(b+1)-1,q)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_global::create_segre_variety" << endl;
	}
	int d;
	long int N1, N2;
	long int rk1, rk2, rk3;
	int *v1;
	int *v2;
	int *v3;

	if (f_v) {
		cout << "geometry_global::create_segre_variety" << endl;
		cout << "a=" << a << " (projective)" << endl;
		cout << "b=" << b << " (projective)" << endl;
	}
	d = (a + 1) * (b + 1);
	if (f_v) {
		cout << "d=" << d << " (vector space dimension)" << endl;
	}


	v1 = NEW_int(a + 1);
	v2 = NEW_int(b + 1);
	v3 = NEW_int(d);


	geometry::other_geometry::geometry_global GG;

	N1 = GG.nb_PG_elements(a, F->q);
	N2 = GG.nb_PG_elements(b, F->q);

	Pts = NEW_lint(N1 * N2);
	nb_pts = 0;


	for (rk1 = 0; rk1 < N1; rk1++) {
		//P1->unrank_point(v1, rk1);
		F->Projective_space_basic->PG_element_unrank_modified_lint(
				v1, 1, a + 1, rk1);


		for (rk2 = 0; rk2 < N2; rk2++) {
			//P2->unrank_point(v2, rk2);
			F->Projective_space_basic->PG_element_unrank_modified_lint(
					v2, 1, b + 1, rk2);


			F->Linear_algebra->mult_matrix_matrix(
					v1, v2, v3, a + 1, 1, b + 1,
					0 /* verbose_level */);

			//rk3 = P3->rank_point(v3);
			F->Projective_space_basic->PG_element_rank_modified_lint(v3, 1, d, rk3);

			Pts[nb_pts++] = rk3;

			if (f_v) {
				cout << setw(4) << nb_pts - 1 << " : " << endl;
				Int_matrix_print(v3, a + 1, b + 1);
				cout << " : " << setw(5) << rk3 << endl;
			}
		}
	}


	label_txt = "segre_variety_" + std::to_string(a) + "_" + std::to_string(b) + "_" + std::to_string(F->q);
	label_tex = "segre\\_variety\\_" + std::to_string(a) + "\\_" + std::to_string(b) + "\\_" + std::to_string(F->q);

	FREE_int(v1);
	FREE_int(v2);
	FREE_int(v3);
}


#if 0
void geometry_global::do_andre(
		field_theory::finite_field *FQ,
		field_theory::finite_field *Fq,
		long int *the_set_in, int set_size_in,
		long int *&the_set_out, int &set_size_out,
	int verbose_level)
// creates PG(2,Q) and PG(4,q)
// this functions is not called from anywhere right now
// it needs a pair of finite fields
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "geometry_global::do_andre for a set of size " << set_size_in << endl;
	}

	geometry::projective_space *P2, *P4;
	int a, a0, a1;
	int b, b0, b1;
	int i, h, k, alpha;
	int *v, *w1, *w2, *w3, *v2;
	int *components;
	int *embedding;
	int *pair_embedding;


	P2 = NEW_OBJECT(geometry::projective_space);
	P4 = NEW_OBJECT(geometry::projective_space);


	P2->projective_space_init(2, FQ,
		false /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	P4->projective_space_init(4, Fq,
		false /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	//d = 5;


	if (f_v) {
		cout << "geometry_global::do_andre "
				"before subfield_embedding_2dimensional" << endl;
	}

	FQ->subfield_embedding_2dimensional(*Fq,
		components, embedding, pair_embedding,
		verbose_level);

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

	if (f_v) {
		cout << "geometry_global::do_andre "
				"after subfield_embedding_2dimensional" << endl;
	}
	if (f_vv) {
		FQ->print_embedding(*Fq,
			components, embedding, pair_embedding);
	}
	alpha = FQ->p;
	if (f_vv) {
		cout << "finite_field::do_andre alpha=" << alpha << endl;
		//FQ->print(true /* f_add_mult_table */);
	}


	v = NEW_int(3);
	w1 = NEW_int(5);
	w2 = NEW_int(5);
	w3 = NEW_int(5);
	v2 = NEW_int(2);


	the_set_out = NEW_lint(P4->N_points);
	set_size_out = 0;

	for (i = 0; i < set_size_in; i++) {
		if (f_vv) {
			cout << "geometry_global::do_andre "
					"input point " << i << " is "
					<< the_set_in[i] << " : ";
		}
		P2->unrank_point(v, the_set_in[i]);
		FQ->PG_element_normalize(v, 1, 3);
		if (f_vv) {
			Int_vec_print(cout, v, 3);
			cout << " becomes ";
		}

		if (v[2] == 0) {

			// we are dealing with a point on the line at infinity.
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
			if (false) {
				cout << "w1=";
				Int_vec_print(cout, w1, 4);
				cout << "w2=";
				Int_vec_print(cout, w2, 4);
				cout << endl;
			}

			// now we create all points on the line spanned
			// by w1[4] and w2[4]:
			// There are q + 1 of these points.
			// We make sure that the coordinate vectors have
			// a zero in the last spot.

			for (h = 0; h < Fq->q + 1; h++) {
				Fq->PG_element_unrank_modified(v2, 1, 2, h);
				if (false) {
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
				the_set_out[set_size_out++] = a;
			}
		}
		else {

			// we are dealing with an affine point:
			// We make sure that the coordinate vector
			// has a one in the last spot.


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
			the_set_out[set_size_out++] = a;
		}
	}

	if (f_v) {
		for (i = 0; i < set_size_out; i++) {
			a = the_set_out[i];
			P4->unrank_point(w1, a);
			cout << setw(3) << i << " : " << setw(5) << a << " : ";
			Int_vec_print(cout, w1, 5);
			cout << endl;
		}
	}

	FREE_OBJECT(P2);
	FREE_OBJECT(P4);
	FREE_int(v);
	FREE_int(w1);
	FREE_int(w2);
	FREE_int(w3);
	FREE_int(v2);
	FREE_int(components);
	FREE_int(embedding);
	FREE_int(pair_embedding);

	if (f_v) {
		cout << "geometry_global::do_andre done" << endl;
	}
}
#endif

void geometry_global::do_embed_orthogonal(
		algebra::field_theory::finite_field *F,
	int epsilon, int n,
	long int *set_in, long int *&set_out, int set_size,
	int verbose_level)
// creates a quadratic_form object
{
	int f_v = (verbose_level >= 1);
	int *v;
	int d = n + 1;
	long int h, a, b;
	orthogonal_geometry::quadratic_form *Quadratic_form;

	if (f_v) {
		cout << "geometry_global::do_embed_orthogonal" << endl;
	}


	v = NEW_int(d);
	set_out = NEW_lint(set_size);

	Quadratic_form = NEW_OBJECT(orthogonal_geometry::quadratic_form);

	if (f_v) {
		cout << "geometry_global::do_embed_orthogonal "
				"before Quadratic_form->init" << endl;
	}
	Quadratic_form->init(epsilon, n, F, verbose_level);
	if (f_v) {
		cout << "geometry_global::do_embed_orthogonal "
				"after Quadratic_form->init" << endl;
	}

	for (h = 0; h < set_size; h++) {
		a = set_in[h];

		Quadratic_form->unrank_point(v, a, 0 /* verbose_level */);

		//b = P->rank_point(v);
		F->Projective_space_basic->PG_element_rank_modified_lint(
				v, 1, n + 1, b);
		set_out[h] = b;
	}

	FREE_int(v);
	FREE_OBJECT(Quadratic_form);
	if (f_v) {
		cout << "geometry_global::do_embed_orthogonal done" << endl;
	}

}

void geometry_global::do_embed_points(
		algebra::field_theory::finite_field *F,
		int n,
		long int *set_in, long int *&set_out, int set_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v;
	int d = n + 2;
	int h;
	long int a, b;

	if (f_v) {
		cout << "geometry_global::do_embed_points" << endl;
	}

	v = NEW_int(d);
	set_out = NEW_lint(set_size);

	for (h = 0; h < set_size; h++) {
		a = set_in[h];

		F->Projective_space_basic->PG_element_unrank_modified_lint(
				v, 1, n + 1, a);

		v[d - 1] = 0;

		F->Projective_space_basic->PG_element_rank_modified_lint(
				v, 1, n + 2, b);

		set_out[h] = b;
	}

	FREE_int(v);
	if (f_v) {
		cout << "geometry_global::do_embed_points done" << endl;
	}

}

void geometry_global::print_set_in_affine_plane(
		algebra::field_theory::finite_field *F,
		int len, long int *S)
{
	int *A;
	int i, j, x, y, v[3];


	A = NEW_int(F->q * F->q);
	for (x = 0; x < F->q; x++) {
		for (y = 0; y < F->q; y++) {
			A[(F->q - 1 - y) * F->q + x] = 0;
		}
	}
	for (i = 0; i < len; i++) {
		F->Projective_space_basic->PG_element_unrank_modified(
				v, 1 /* stride */, 3 /* len */, S[i]);
		if (v[2] != 1) {
			//cout << "my_generator::print_set_in_affine_plane
			// not an affine point" << endl;
			cout << "(" << v[0] << "," << v[1]
					<< "," << v[2] << ")" << endl;
			continue;
		}
		x = v[0];
		y = v[1];
		A[(F->q - 1 - y) * F->q + x] = 1;
	}
	for (i = 0; i < F->q; i++) {
		for (j = 0; j < F->q; j++) {
			cout << A[i * F->q + j];
		}
		cout << endl;
	}
	FREE_int(A);
}



void geometry_global::simeon(
		algebra::field_theory::finite_field *F,
		int n, int len, long int *S, int s, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 1);
	int k, nb_rows, nb_cols, nb_r1, nb_r2, row, col;
	int *Coord;
	int *M;
	int *A;
	int *C;
	int *T;
	int *Ac; // no not free
	int *U;
	int *U1;
	int nb_A, nb_U;
	int a, u, ac, i, d, idx, mtx_rank;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "geometry_global::simeon s=" << s << endl;
	}
	k = n + 1;
	nb_cols = Combi.int_n_choose_k(len, k - 1);
	nb_r1 = Combi.int_n_choose_k(len, s);
	nb_r2 = Combi.int_n_choose_k(len - s, k - 2);
	nb_rows = nb_r1 * nb_r2;
	if (f_v) {
		cout << "nb_r1=" << nb_r1 << endl;
		cout << "nb_r2=" << nb_r2 << endl;
		cout << "nb_rows=" << nb_rows << endl;
		cout << "nb_cols=" << nb_cols << endl;
	}

	Coord = NEW_int(len * k);
	M = NEW_int(nb_rows * nb_cols);
	A = NEW_int(len);
	U = NEW_int(k - 2);
	U1 = NEW_int(k - 2);
	C = NEW_int(k - 1);
	T = NEW_int(k * k);

	Int_vec_zero(M, nb_rows * nb_cols);


	// unrank all points of the arc:
	for (i = 0; i < len; i++) {
		//point_unrank(Coord + i * k, S[i]);
		F->Projective_space_basic->PG_element_unrank_modified(
				Coord + i * k, 1 /* stride */, n + 1 /* len */, S[i]);
	}


	nb_A = Combi.int_n_choose_k(len, k - 2);
	nb_U = Combi.int_n_choose_k(len - (k - 2), k - 1);
	if (nb_A * nb_U != nb_rows) {
		cout << "nb_A * nb_U != nb_rows" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "nb_A=" << nb_A << endl;
		cout << "nb_U=" << nb_U << endl;
	}


	Ac = A + k - 2;

	row = 0;
	for (a = 0; a < nb_A; a++) {
		if (f_vv) {
			cout << "a=" << a << " / " << nb_A << ":" << endl;
		}
		Combi.unrank_k_subset(a, A, len, k - 2);
		Combi.set_complement(A, k - 2, Ac, ac, len);
		if (ac != len - (k - 2)) {
			cout << "geometry_global::simeon ac != len - (k - 2)" << endl;
			exit(1);
		}
		if (f_vv) {
			cout << "Ac=";
			Int_vec_print(cout, Ac, ac);
			cout << endl;
		}


		for (u = 0; u < nb_U; u++, row++) {

			Combi.unrank_k_subset(u, U, len - (k - 2), k - 1);
			for (i = 0; i < k - 1; i++) {
				U1[i] = Ac[U[i]];
			}
			if (f_vv) {
				cout << "U1=";
				Int_vec_print(cout, U1, k - 1);
				cout << endl;
			}

			for (col = 0; col < nb_cols; col++) {
				if (f_vv) {
					cout << "row=" << row << " / " << nb_rows
							<< " col=" << col << " / "
							<< nb_cols << ":" << endl;
				}
				Combi.unrank_k_subset(col, C, len, k - 1);
				if (f_vv) {
					cout << "C: ";
					Int_vec_print(cout, C, k - 1);
					cout << endl;
				}


				// test if A is a subset of C:
				for (i = 0; i < k - 2; i++) {
					if (!Sorting.int_vec_search_linear(C, k - 1, A[i], idx)) {
						//cout << "did not find A[" << i << "] in C" << endl;
						break;
					}
				}
				if (i == k - 2) {
					d = F->Linear_algebra->BallChowdhury_matrix_entry(
							Coord, C, U1, k, s /*sz_U */,
						T, 0 /*verbose_level*/);
					if (f_vv) {
						cout << "d=" << d << endl;
					}

					M[row * nb_cols + col] = d;
				} // if
			} // next col
		} // next u
	} // next a

	if (f_v) {
		cout << "simeon, the matrix M is:" << endl;
		//int_matrix_print(M, nb_rows, nb_cols);
	}

	//print_integer_matrix_with_standard_labels(cout, M,
	//nb_rows, nb_cols, true /* f_tex*/);
	//int_matrix_print_tex(cout, M, nb_rows, nb_cols);

	if (f_v) {
		cout << "nb_rows=" << nb_rows << endl;
		cout << "nb_cols=" << nb_cols << endl;
		cout << "s=" << s << endl;
	}

	mtx_rank = F->Linear_algebra->Gauss_easy(M, nb_rows, nb_cols);
	if (f_v) {
		cout << "mtx_rank=" << mtx_rank << endl;
		//cout << "simeon, the reduced matrix M is:" << endl;
		//int_matrix_print(M, mtx_rank, nb_cols);
	}


	FREE_int(Coord);
	FREE_int(M);
	FREE_int(A);
	FREE_int(C);
	//FREE_int(E);
#if 0
	FREE_int(A1);
	FREE_int(C1);
	FREE_int(E1);
	FREE_int(A2);
	FREE_int(C2);
#endif
	FREE_int(T);
	if (f_v) {
		cout << "geometry_global::simeon s=" << s << " done" << endl;
	}
}


void geometry_global::wedge_to_klein(
		algebra::field_theory::finite_field *F,
		int *W, int *K)
{
	K[0] = W[0]; // 12
	K[1] = W[5]; // 34
	K[2] = W[1]; // 13
	K[3] = F->negate(W[4]); // 24
	K[4] = W[2]; // 14
	K[5] = W[3]; // 23
}

void geometry_global::klein_to_wedge(
		algebra::field_theory::finite_field *F,
		int *K, int *W)
{
	W[0] = K[0];
	W[1] = K[2];
	W[2] = K[4];
	W[3] = K[5];
	W[4] = F->negate(K[3]);
	W[5] = K[1];
}


void geometry_global::isomorphism_to_special_orthogonal(
		algebra::field_theory::finite_field *F,
		int *A4, int *A6, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "geometry_global::isomorphism_to_special_orthogonal" << endl;
	}
	int i, j;
	int Basis1[] = {
			1,0,0,0,0,0,
			0,1,0,0,0,0,
			0,0,1,0,0,0,
			0,0,0,1,0,0,
			0,0,0,0,1,0,
			0,0,0,0,0,1,
	};
	int Basis2[36];
	int An2[37];
	int v[6];
	int w[6];
	int C[36];
	int D[36];
	int B[] = {
			1,0,0,0,0,0,
			0,0,0,2,0,0,
			1,3,0,0,0,0,
			0,0,0,1,3,0,
			1,0,2,0,0,0,
			0,0,0,2,0,4,
	};
	int Bv[36];
	other::data_structures::sorting Sorting;

	for (i = 0; i < 6; i++) {
		klein_to_wedge(F, Basis1 + i * 6, Basis2 + i * 6);
	}

	F->Linear_algebra->matrix_inverse(B, Bv, 6,
			0 /* verbose_level */);




	F->Linear_algebra->exterior_square(A4, An2, 4,
			0 /*verbose_level*/);

	if (f_vv) {
		cout << "geometry_global::isomorphism_to_special_orthogonal "
				"exterior_square :" << endl;
		Int_matrix_print(An2, 6, 6);
		cout << endl;
	}


	for (j = 0; j < 6; j++) {
		F->Linear_algebra->mult_vector_from_the_left(
				Basis2 + j * 6, An2, v, 6, 6);
				// v[m], A[m][n], vA[n]
		wedge_to_klein(F, v, w);
		Int_vec_copy(w, C + j * 6, 6);
	}


	if (f_vv) {
		cout << "geometry_global::isomorphism_to_special_orthogonal "
				"orthogonal matrix :" << endl;
		Int_matrix_print(C, 6, 6);
		cout << endl;
	}

	F->Linear_algebra->mult_matrix_matrix(
			Bv, C, D, 6, 6, 6,
			0 /*verbose_level */);
	F->Linear_algebra->mult_matrix_matrix(
			D, B, A6, 6, 6, 6,
			0 /*verbose_level */);

	F->Projective_space_basic->PG_element_normalize_from_front(
			A6, 1, 36);

	if (f_vv) {
		cout << "geometry_global::isomorphism_to_special_orthogonal "
				"orthogonal matrix in the special form:" << endl;
		Int_matrix_print(A6, 6, 6);
		cout << endl;
	}

	if (f_v) {
		cout << "geometry_global::isomorphism_to_special_orthogonal done" << endl;
	}

}

void geometry_global::minimal_orbit_rep_under_stabilizer_of_frame_characteristic_two(
		algebra::field_theory::finite_field *F,
		int x, int y,
		int &a, int &b, int verbose_level)
// used by surface_classify_wedge::identify_general_abcd
{
	int X[6];
	int i, i0;

	X[0] = x;
	X[1] = F->add(x, 1);
	X[2] = F->mult(x, F->inverse(y));
	X[3] = F->mult(F->add(x, y), F->inverse(F->add(y, 1)));
	X[4] = F->mult(1, F->inverse(y));
	X[5] = F->mult(F->add(x, y), F->inverse(x));
	i0 = 0;
	for (i = 1; i < 6; i++) {
		if (X[i] < X[i0]) {
			i0 = i;
		}
	}
	a = X[i0];
	if (i0 == 0) {
		b = y;
	}
	else if (i0 == 1) {
		b = F->add(y, 1);
	}
	else if (i0 == 2) {
		b = F->mult(1, F->inverse(y));
	}
	else if (i0 == 3) {
		b = F->mult(y, F->inverse(F->add(y, 1)));
	}
	else if (i0 == 4) {
		b = F->mult(x, F->inverse(y));
	}
	else if (i0 == 5) {
		b = F->mult(F->add(x, 1), F->inverse(x));
	}
}

int geometry_global::evaluate_Fermat_cubic(
		algebra::field_theory::finite_field *F,
		int *v)
// used to create the Schlaefli graph
{
	int a, i;

	a = 0;
	for (i = 0; i < 4; i++) {
		a = F->add(a, F->power(v[i], 3));
	}
	return a;
}


}}}}

