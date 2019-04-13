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


void order_POmega_epsilon(int epsilon, int k, int q,
		longinteger_object &go, int verbose_level)
// k is projective dimension
{
	int w, m;
	
	w = Witt_index(epsilon, k);
	if (epsilon == -1) {
		m = w + 1;
		}
	else {
		m = w;
		}
	order_Pomega(epsilon, m, q, go, verbose_level);
	cout << "order_POmega_epsilon  epsilon=" << epsilon
			<< " k=" << k << " q=" << q << " order=" << go << endl;

#if 0
	int f_v = (verbose_level >= 1);
	int n;
	
	n = Witt_index(epsilon, k);
	if (f_v) {
		cout << "Witt index is " << n << endl;
		}
	if (epsilon == 0) {
		order_Pomega(0, n, q, go, verbose_level);
		}
	else if (epsilon == 1) {
		order_Pomega_plusminus(1, n, q, go, verbose_level);
		}
	else if (epsilon == -1) {
		order_Pomega_plusminus(-1, n, q, go, verbose_level);
		}
#endif
}

#if 0
void order_Pomega_plusminus(int epsilon, int m, int q,
		longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object Q, Qm, A, R, S, T, O, minusone, minusepsilon;
	int i, v, r;

	if (epsilon == -1) {
		m++;
		}
	
	//u = i_power_j(q, m) - epsilon;
	//v = gcd_int(u, 4);
	if (EVEN(q)) {
		v = 1;
		}
	else {
		if (epsilon == 1) {
			if (DOUBLYEVEN(q - 1)) {
				v = 4;
				}
			else {
				if (EVEN(m)) {
					v = 4;
					}
				else {
					v = 2;
					}
				}
			}
		else {
			cout << "order_Pomega_plusminus epsilon == -1" << endl;
			exit(1);
			}
		}

	minusone.create(-1);
	minusepsilon.create(-epsilon);
	Q.create(q);
	D.power_int(Q, m);
	Q.assign_to(Qm);
	D.power_int(Q, m - 1);
	D.add(Qm, minusepsilon, A);
	if (f_v) {
		cout << q << "^" << m << " - " << epsilon << " = " << A << endl;
		cout << q << "^" << m << "*" << m - 1 << " = " << Q << endl;
		}
	O.create(1);
	for (i = 1; i < m; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << q << "^" << 2 * i << " - 1 = " << S << endl;
			}
		D.mult(O, S, T);
		T.assign_to(O);
		}
	D.mult(O, A, S);
	D.mult(S, Q, T);
	D.integral_division_by_int(T, v, o, r);
	if (f_v) {
		cout << "the order of P\\Omega^" << epsilon << "(" << 2 * m
				<< "," << q << ") is " << o << endl;
		}
}
#endif

void order_PO_epsilon(int f_semilinear,
	int epsilon, int k, int q,
	longinteger_object &go, int verbose_level)
// k is projective dimension
{
	int f_v = (verbose_level >= 1);
	int m;
	number_theory_domain NT;
	
	if (f_v) {
		cout << "order_PO_epsilon" << endl;
		}
	m = Witt_index(epsilon, k);
	if (f_v) {
		cout << "Witt index = " << m << endl;
		}
	order_PO(epsilon, m, q, go, verbose_level);
	if (f_semilinear) {
		int p, e;
		longinteger_domain D;
		
		NT.factor_prime_power(q, p, e);
		D.mult_integer_in_place(go, e);
		}
	if (f_v) {
		cout << "order_PO_epsilon  f_semilinear=" << f_semilinear
				<< " epsilon=" << epsilon << " k=" << k
				<< " q=" << q << " order=" << go << endl;
		}
}

void order_PO(int epsilon, int m, int q,
		longinteger_object &o, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "order_PO epsilon = " << epsilon
				<< " m=" << m << " q=" << q << endl;
		}

	if (epsilon == 0) {
		order_PO_parabolic(m, q, o, verbose_level);
		}
	else if (epsilon == 1) {
		order_PO_plus(m, q, o, verbose_level);
		}
	else if (epsilon == -1) {
		order_PO_minus(m, q, o, verbose_level);
		}
	else {
		cout << "order_PO fatal: epsilon = " << epsilon << endl;
		exit(1);
		}
}

void order_Pomega(int epsilon, int m, int q,
		longinteger_object &o, int verbose_level)
{
	if (epsilon == 0) {
		order_Pomega_parabolic(m, q, o, verbose_level);
		}
	else if (epsilon == 1) {
		order_Pomega_plus(m, q, o, verbose_level);
		}
	else if (epsilon == -1) {
		order_Pomega_minus(m, q, o, verbose_level);
		}
	else {
		cout << "order_Pomega fatal: epsilon = " << epsilon << endl;
		exit(1);
		}
}

void order_PO_plus(int m, int q,
		longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object O, Q, R, S, T, Two, minusone;
	int i;


	Two.create(2);
	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * (m - 1));
	if (f_v) {
		cout << "order_PO_plus " << q << "^(" << m << "*"
				<< m - 1 << ") = " << Q << endl;
		}
	// now Q = q^{m(m-1)}

	O.create(1);
	for (i = 1; i <= m - 1; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << "order_PO_plus " << q << "^"
					<< 2 * i << " - 1 = " << S << endl;
			}
		D.mult(O, S, T);
		T.assign_to(O);
		}
	// now O = \prod_{i=1}^{m-1} (q^{2i}-1)
	
	R.create(q);
	D.power_int(R, m);
	D.add(R, minusone, S);
	if (f_v) {
		cout << "order_PO_plus " << q << "^" << m << " - 1 = " << S << endl;
		}
	// now S = q^m-1

	D.mult(O, S, T);
	T.assign_to(O);

	D.mult(O, Q, T);
	if (TRUE /*EVEN(q)*/) {
		D.mult(T, Two, o);
		}
	else {
		T.assign_to(o);
		}


	if (f_v) {
		cout << "order_PO_plus the order of PO" << "("
				<< dimension_given_Witt_index(1, m) << ","
				<< q << ") is " << o << endl;
		}
}

void order_PO_minus(int m, int q,
		longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m+2
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object O, Q, R, S, T, Two, plusone, minusone;
	int i;


	Two.create(2);
	plusone.create(1);
	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * (m + 1));
	if (f_v) {
		cout << "order_PO_minus " << q << "^(" << m << "*"
				<< m + 1 << ") = " << Q << endl;
		}
	// now Q = q^{m(m+1)}

	O.create(1);
	for (i = 1; i <= m; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << "order_PO_minus " << q << "^" << 2 * i
					<< " - 1 = " << S << endl;
			}
		D.mult(O, S, T);
		T.assign_to(O);
		}
	// now O = \prod_{i=1}^{m} (q^{2i}-1)
	
	R.create(q);
	D.power_int(R, m + 1);
	D.add(R, plusone, S);
	if (f_v) {
		cout << "order_PO_minus " << q << "^" << m + 1
				<< " + 1 = " << S << endl;
		}
	// now S = q^{m+1}-1

	D.mult(O, S, T);
	T.assign_to(O);

	D.mult(O, Q, T);
	if (EVEN(q)) {
		D.mult(T, Two, o);
		}
	else {
		T.assign_to(o);
		}


	if (f_v) {
		cout << "order_PO_minus the order of PO^-" << "("
			<< dimension_given_Witt_index(-1, m) << ","
			<< q << ") is " << o << endl;
		}
}

void order_PO_parabolic(int m, int q,
		longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m+1
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object O, Q, R, S, T, minusone;
	int i;


	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * m);
	if (f_v) {
		cout << "order_PO_parabolic " << q << "^(" << m
				<< "^2" << ") = " << Q << endl;
		}
	// now Q = q^{m^2}

	O.create(1);
	for (i = 1; i <= m; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << "order_PO_parabolic " << q << "^"
					<< 2 * i << " - 1 = " << S << endl;
			}
		D.mult(O, S, T);
		T.assign_to(O);
		}
	// now O = \prod_{i=1}^{m} (q^{2i}-1)
	

	D.mult(O, Q, o);


	if (f_v) {
		cout << "order_PO_parabolic the order of PO" << "("
			<< dimension_given_Witt_index(0, m) << ","
			<< q << ") is " << o << endl;
		}
}


void order_Pomega_plus(int m, int q,
		longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object O, Q, R, S, S1, T, minusone;
	int i, r;


	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * (m - 1));
	if (f_v) {
		cout << q << "^(" << m << "*" << m - 1 << ") = " << Q << endl;
		}
	O.create(1);
	for (i = 1; i <= m - 1; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << q << "^" << 2 * i << " - 1 = " << S << endl;
			}
		D.mult(O, S, T);
		T.assign_to(O);
		}
	
	R.create(q);
	D.power_int(R, m);
	D.add(R, minusone, S);
	if (f_v) {
		cout << q << "^" << m << " - 1 = " << S << endl;
		}
	D.integral_division_by_int(S, 2, S1, r);
	if (r == 0) {
		S1.assign_to(S);
		}
	D.integral_division_by_int(S, 2, S1, r);
	if (r == 0) {
		S1.assign_to(S);
		}

	D.mult(O, S, T);
	T.assign_to(O);

	D.mult(O, Q, T);
	T.assign_to(o);


	if (f_v) {
		cout << "the order of P\\Omega^1" << "("
			<< dimension_given_Witt_index(1, m) << ","
			<< q << ") is " << o << endl;
		}
}

void order_Pomega_minus(int m, int q,
		longinteger_object &o, int verbose_level)
// m = half the dimension,
// the dimension is n = 2m, the Witt index is m - 1
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object O, Q, R, S, S1, T, minusone, plusone;
	int i, r;

	if (f_v) {
		cout << "order_Pomega_minus m=" << m << " q=" << q << endl;
		}
	minusone.create(-1);
	plusone.create(1);
	Q.create(q);
	D.power_int(Q, m * (m - 1));
	if (f_v) {
		cout << q << "^(" << m << "*" << m - 1 << ") = " << Q << endl;
		}
	O.create(1);
	for (i = 1; i <= m - 1; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << q << "^" << 2 * i << " - 1 = " << S << endl;
			}
		D.mult(O, S, T);
		T.assign_to(O);
		}
	
	R.create(q);
	D.power_int(R, m);
	D.add(R, plusone, S);
	if (f_v) {
		cout << q << "^" << m << " + 1 = " << S << endl;
		}
	D.integral_division_by_int(S, 2, S1, r);
	if (r == 0) {
		if (f_v) {
			cout << "divide by 2" << endl;
			}
		S1.assign_to(S);
		}
	D.integral_division_by_int(S, 2, S1, r);
	if (r == 0) {
		if (f_v) {
			cout << "divide by 2" << endl;
			}
		S1.assign_to(S);
		}

	D.mult(O, S, T);
	T.assign_to(O);

	D.mult(O, Q, T);
	T.assign_to(o);


	if (f_v) {
		cout << "the order of P\\Omega^-1" << "("
			<< dimension_given_Witt_index(-1, m - 1) << ","
			<< q << ") is " << o << endl;
		}
}

void order_Pomega_parabolic(int m, int q,
		longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m + 1
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object O, Q, R, S, T, minusone;
	int i, r;


	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * m);
	if (f_v) {
		cout << q << "^(" << m << "^2) = " << Q << endl;
		}
	O.create(1);
	for (i = 1; i <= m; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << q << "^" << 2 * i << " - 1 = " << S << endl;
			}
		D.mult(O, S, T);
		T.assign_to(O);
		}
	D.mult(O, Q, T);
	if (EVEN(q)) {
		T.assign_to(o);
		}
	else {
		D.integral_division_by_int(T, 2, o, r);
		}
	if (f_v) {
		cout << "the order of P\\Omega" << "("
			<< dimension_given_Witt_index(0, m) << ","
			<< q << ") is " << o << endl;
		}
}

int index_POmega_in_PO(int epsilon, int m, int q, int verbose_level)
{
	if (epsilon == 0) {
		if (EVEN(q)) {
			return 1;
			}
		else {
			return 2;
			}
		}
	if (epsilon == 1) {
		if (EVEN(q)) {
			return 2;
			}
		else {
			if (DOUBLYEVEN(q - 1)) {
				return 4;
				}
			else {
				if (EVEN(m)) {
					return 4;
					}
				else {
					return 2;
					}
				}
			}
		}
	if (epsilon == -1) {
		if (EVEN(q)) {
			return 2;
			}
		else {
			if (DOUBLYEVEN(q - 1)) {
				return 2;
				}
			else {
				if (EVEN(m + 1)) {
					return 2;
					}
				else {
					return 4;
					}
				}
			}
		}
#if 0
	if (epsilon == -1) {
		cout << "index_POmega_in_PO epsilon = -1 not "
				"yet implemented, returning 1" << endl;
		return 1;
		exit(1);
		}
#endif
	cout << "index_POmega_in_PO epsilon not recognized, "
			"epsilon=" << epsilon << endl;
	exit(1);
}

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
