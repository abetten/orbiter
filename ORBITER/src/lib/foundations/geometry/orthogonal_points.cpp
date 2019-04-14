// orthogonal_points.C
//
// Anton Betten
//
// started:  February 15, 2005
// continued:  August 10, 2006 (in Perth)
// 5/20/07: changed the labeling of points in parabolic type
// 7/9/7 renamed orthogonal_points.C (formerly orthogonal.C)



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


int count_Sbar(int n, int q)
{
	return count_T1(1, n, q);
}

int count_S(int n, int q)
{
	return (q - 1) * count_Sbar(n, q) + 1;
}

int count_N1(int n, int q)
{
	if (n <= 0) {
		return 0;
		}
	return nb_pts_N1(n, q);
}

int count_T1(int epsilon, int n, int q)
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

int count_T2(int n, int q)
{
	number_theory_domain NT;

	if (n <= 0) {
		return 0;
		}
	return (NT.i_power_j(q, 2 * n - 2) - 1) *
			(NT.i_power_j(q, n) - 1) *
			(NT.i_power_j(q, n - 2) + 1) / ((q - 1) * (NT.i_power_j(q, 2) - 1));
}

int nb_pts_Qepsilon(int epsilon, int k, int q)
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

int dimension_given_Witt_index(int epsilon, int n)
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

int Witt_index(int epsilon, int k)
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

int nb_pts_Q(int k, int q)
// number of singular points on Q(k,q), parabolic quadric, so k is even
{
	int n;
	
	n = Witt_index(0, k);
	return nb_pts_Sbar(n, q) + nb_pts_N1(n, q);
}

int nb_pts_Qplus(int k, int q)
// number of singular points on Q^+(k,q)
{
	int n;
	
	n = Witt_index(1, k);
	return nb_pts_Sbar(n, q);
}

int nb_pts_Qminus(int k, int q)
// number of singular points on Q^-(k,q)
{
	int n;
	
	n = Witt_index(-1, k);
	return nb_pts_Sbar(n, q) + (q + 1) * nb_pts_N1(n, q);
}


// #############################################################################
// the following functions are for the hyperbolic quadric with Witt index n:
// #############################################################################

int nb_pts_S(int n, int q)
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

int nb_pts_N(int n, int q)
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

int nb_pts_N1(int n, int q)
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

int nb_pts_Sbar(int n, int q)
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

int nb_pts_Nbar(int n, int q)
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


// #############################################################################
// other stuff:
// #############################################################################


void test_Orthogonal(int epsilon, int k, int q)
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

void test_orthogonal(int n, int q)
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



}
}
