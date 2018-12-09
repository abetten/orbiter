// orthogonal_points.C
//
// Anton Betten
//
// started:  February 15, 2005
// continued:  August 10, 2006 (in Perth)
// 5/20/07: changed the labeling of points in parabolic type
// 7/9/7 renamed orthogonal_points.C (formerly orthogonal.C)



#include "foundations.h"

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
	if (n < 0) {
		//cout << "count_T1 n is negative. n=" << n << endl;
		return 0;
		}
	if (epsilon == 1) {
		return ((i_power_j(q, n) - 1) *
				(i_power_j(q, n - 1) + 1)) / (q - 1);
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
	if (n <= 0) {
		return 0;
		}
	return (i_power_j(q, 2 * n - 2) - 1) *
			(i_power_j(q, n) - 1) *
			(i_power_j(q, n - 2) + 1) / ((q - 1) * (i_power_j(q, 2) - 1));
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

int evaluate_quadratic_form(finite_field &GFq, int *v, int stride, 
	int epsilon, int k, int form_c1, int form_c2, int form_c3)
{
	int n, a, b, c = 0, d, x, x1, x2;
	
	n = Witt_index(epsilon, k);
	if (epsilon == 0) {
		a = evaluate_hyperbolic_quadratic_form(GFq, v + stride, stride, n);
		x = v[0];
		b = GFq.product3(form_c1, x, x);
		c = GFq.add(a, b);
		}
	else if (epsilon == 1) {
		c = evaluate_hyperbolic_quadratic_form(GFq, v, stride, n);
		}
	else if (epsilon == -1) {
		a = evaluate_hyperbolic_quadratic_form(GFq, v, stride, n);
		x1 = v[2 * n * stride];
		x2 = v[(2 * n + 1) * stride];
		b = GFq.product3(form_c1, x1, x1);
		c = GFq.product3(form_c2, x1, x2);
		d = GFq.product3(form_c3, x2, x2);
		c = GFq.add4(a, b, c, d);
		}
	return c;
}

void Q_epsilon_unrank(finite_field &GFq,
	int *v, int stride, int epsilon, int k,
	int c1, int c2, int c3, int a)
{
	if (epsilon == 0) {
		Q_unrank(GFq, v, stride, k, a);
		}
	else if (epsilon == 1) {
		Qplus_unrank(GFq, v, stride, k, a);
		}
	else if (epsilon == -1) {
		Qminus_unrank(GFq, v, stride, k, a, c1, c2, c3);
		}
	else {
		cout << "Q_epsilon_unrank epsilon is wrong" << endl;
		exit(1);
		}
}

int Q_epsilon_rank(finite_field &GFq,
	int *v, int stride, int epsilon, int k,
	int c1, int c2, int c3)
{
	int a;
	
	if (epsilon == 0) {
		a = Q_rank(GFq, v, stride, k);
		}
	else if (epsilon == 1) {
		a = Qplus_rank(GFq, v, stride, k);
		}
	else if (epsilon == -1) {
		a = Qminus_rank(GFq, v, stride, k, c1, c2, c3);
		}
	else {
		cout << "Q_epsilon_unrank epsilon is wrong" << endl;
		exit(1);
		}
	return a;
}

#if 0
// old version
void Q_unrank(finite_field &GFq, int *v, int stride, int k, int a)
// k = projective dimension, must be even
{
	int n, x, i, minusone;
	
	n = Witt_index(0, k);
	x = nb_pts_Sbar(n, GFq.q);
	if (a < x) {
		v[2 * n * stride] = 0;
		Sbar_unrank(GFq, v, stride, n, a);
		return;
		}
	a -= x;
	v[2 * n * stride] = 1;
	N1_unrank(GFq, v, stride, n, a);
	minusone = GFq.negate(1);
	if (minusone != 1) {
		for (i = 0; i < n; i++) {
			v[2 * i * stride] = GFq.mult(v[2 * i * stride], minusone);
			}
		}
}

int Q_rank(finite_field &GFq, int *v, int stride, int k)
// k = projective dimension, must be even
{
	int n, x, a, b, i, minusone;
	
	n = Witt_index(0, k);
	x = nb_pts_Sbar(n, GFq.q);
	if (v[2 * n * stride] == 0) {
		Sbar_rank(GFq, v, stride, n, a);
		return a;
		}
	a = x;
	if (v[2 * n * stride] != 1) {
		PG_element_normalize(GFq, v, stride, k + 1);
		}
	minusone = GFq.negate(1);
	if (minusone != 1) {
		for (i = 0; i < n; i++) {
			v[2 * i * stride] = GFq.mult(v[2 * i * stride], minusone);
			}
		}
	N1_rank(GFq, v, stride, n, b);
	return a + b;
}
#endif

vector_hashing *Hash_table_parabolic = NULL;
int Hash_table_parabolic_q = 0;
int Hash_table_parabolic_k = 0;

void init_hash_table_parabolic(finite_field &GFq,
		int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q, n, ln2q, N, i, j;
	int *v;
	
	q = GFq.q;
	if (f_v) {
		cout << "init_hash_table_parabolic" << endl;
		cout << "q=" << q << endl;
		cout << "k=" << k << endl;
		}
	
	ln2q = int_log2(q);
	Hash_table_parabolic = NEW_OBJECT(vector_hashing);
	Hash_table_parabolic_q = q;
	Hash_table_parabolic_k = k;
	n = k + 1;
	N = nb_pts_Q(k, q);
	if (f_v) {
		cout << "N=" << N << endl;
		}
	Hash_table_parabolic->allocate(n, N, ln2q);
	for (i = 0; i < N; i++) {
		v = Hash_table_parabolic->vector_data + i * n;
		Q_unrank_directly(GFq, v, 1 /* stride */, k, i);
		for (j = 0; j < k + 1; j++) {
			if (v[j])
				break;
			}
		if (v[j] != 1) {
			cout << "init_hash_table_parabolic vector "
					"is not normalized" << endl;
			cout << "i=" << i << endl;
			int_vec_print(cout, v, k + 1);
			cout << endl;
			exit(1);
			}
		}
	Hash_table_parabolic->compute_tables(verbose_level - 1);
	
}

void Q_unrank(finite_field &GFq,
		int *v, int stride, int k, int a)
{
	if (Hash_table_parabolic) {
		if (Hash_table_parabolic_q == GFq.q &&
				Hash_table_parabolic_k == k) {
			if (stride != 1) {
				cout << "Q_unrank with Hash table "
						"needs stride == 1" << endl;
				exit(1);
				}
			Hash_table_parabolic->unrank(a, v);
			return;
			}
		}
	Q_unrank_directly(GFq, v, stride, k, a);
}

int Q_rank(finite_field &GFq,
		int *v, int stride, int k)
{
	if (Hash_table_parabolic) {
		GFq.PG_element_normalize_from_front(v, stride, k + 1);
		if (Hash_table_parabolic_q == GFq.q &&
				Hash_table_parabolic_k == k) {
			if (stride != 1) {
				cout << "Q_unrank with Hash table "
						"needs stride == 1" << endl;
				exit(1);
				}
			return Hash_table_parabolic->rank(v);
			}
		}
	return Q_rank_directly(GFq, v, stride, k);
}

void Q_unrank_directly(finite_field &GFq,
		int *v, int stride, int k, int a)
// parabolic quadric
// k = projective dimension, must be even
{
	int n, x, i, minusone;
	
	n = Witt_index(0, k);
	x = nb_pts_Sbar(n, GFq.q);
	if (a < x) {
		v[0] = 0;
		Sbar_unrank(GFq, v + stride, stride, n, a);
		GFq.PG_element_normalize_from_front(v + stride, stride, k);
		return;
		}
	a -= x;
	v[0] = 1;
	N1_unrank(GFq, v + stride, stride, n, a);
	minusone = GFq.negate(1);
	if (minusone != 1) {
		for (i = 0; i < n; i++) {
			v[(1 + 2 * i) * stride] =
				GFq.mult(v[(1 + 2 * i) * stride], minusone);
			}
		}
}

int Q_rank_directly(finite_field &GFq,
		int *v, int stride, int k)
// parabolic quadric
// k = projective dimension, must be even
{
	int n, x, a, b, i, minusone;
	
	n = Witt_index(0, k);
	x = nb_pts_Sbar(n, GFq.q);
	if (v[0] == 0) {
		Sbar_rank(GFq, v + stride, stride, n, a);
		return a;
		}
	a = x;
	if (v[0] != 1) {
		GFq.PG_element_normalize_from_front(v, stride, k + 1);
		}
	minusone = GFq.negate(1);
	if (minusone != 1) {
		for (i = 0; i < n; i++) {
			v[(1 + 2 * i) * stride] =
				GFq.mult(v[(1 + 2 * i) * stride], minusone);
			}
		}
	N1_rank(GFq, v + stride, stride, n, b);
	return a + b;
}
void Qplus_unrank(finite_field &GFq,
		int *v, int stride, int k, int a)
// hyperbolic quadric
// k = projective dimension, must be odd
{
	int n;
	
	n = Witt_index(1, k);
	Sbar_unrank(GFq, v, stride, n, a);
}

int Qplus_rank(finite_field &GFq,
		int *v, int stride, int k)
// hyperbolic quadric
// k = projective dimension, must be odd
{
	int n, a;
	
	n = Witt_index(1, k);
	Sbar_rank(GFq, v, stride, n, a);
	return a;
}

void Qminus_unrank(finite_field &GFq,
		int *v, int stride, int k, int a,
		int c1, int c2, int c3)
// elliptic quadric
// k = projective dimension, must be odd
// the form is 
// \sum_{i=0}^n x_{2i}x_{2i+1} + c1 x_{2n}^2 +
// c2 x_{2n} x_{2n+1} + c3 x_{2n+1}^2
{
	int n, x, b, c, minusz, x1, x2, u, vv, w, z, i;
	
	n = Witt_index(-1, k);
	x = nb_pts_Sbar(n, GFq.q);
	if (a < x) {
		v[2 * n * stride] = 0;
		v[(2 * n + 1) * stride] = 0;
		Sbar_unrank(GFq, v, stride, n, a);
		return;
		}
	a -= x;
	x = nb_pts_N1(n, GFq.q);
	b = a / x;
	c = a % x;
	// b determines an element on the projective line
	if (b == 0) {
		x1 = 1;
		x2 = 0;
		}
	else {
		b--;
		x1 = b;
		x2 = 1;
		if (b >= GFq.q) {
			cout << "Qminus_unrank() b >= q, the rank "
					"was too big" << endl;
			exit(1);
			}
		}
	v[2 * n * stride] = x1;
	v[(2 * n + 1) * stride] = x2;
	u = GFq.product3(c1, x1, x1);
	vv = GFq.product3(c2, x1, x2);
	w = GFq.product3(c3, x2, x2);
	z = GFq.add3(u, vv, w);
	if (z == 0) {
		cout << "Qminus_unrank z = 0" << endl;
		cout << "b=" << b << endl;
		cout << "c1=" << c1 << endl;
		cout << "c2=" << c2 << endl;
		cout << "c3=" << c3 << endl;
		cout << "x1=" << x1 << endl;
		cout << "x2=" << x2 << endl;
		cout << "u=c1*x1*x1=" << u << endl;
		cout << "vv=c2*x1*x2=" << vv << endl;
		cout << "w=c3*x2*x2=" << w << endl;
		exit(1);
		}
	N1_unrank(GFq, v, stride, n, c);
	minusz = GFq.negate(z);
	if (minusz != 1) {
		for (i = 0; i < n; i++) {
			v[2 * i * stride] = GFq.mult(v[2 * i * stride], minusz);
			}
		}
}

int Qminus_rank(finite_field &GFq,
		int *v, int stride, int k, int c1, int c2, int c3)
// elliptic quadric
// k = projective dimension, must be odd
// the form is 
// \sum_{i=0}^n x_{2i}x_{2i+1} + c1 x_{2n}^2 +
// c2 x_{2n} x_{2n+1} + c3 x_{2n+1}^2
{
	int a, n, x, b, c, minusz, minuszv;
	int x1, x2, u, vv, w, z, i;
	
	n = Witt_index(-1, k);

	{
	int aa;
	aa = evaluate_quadratic_form(GFq, v, stride, -1, k,
			c1, c2, c3);
	if (aa) {
		cout << "Qminus_rank fatal: the vector "
				"is not zero under the quadratic form" << endl;
		cout << "value=" << aa << endl;
		cout << "stride=" << stride << endl;
		cout << "k=" << k << endl;
		cout << "c1=" << c1 << endl;
		cout << "c2=" << c2 << endl;
		cout << "c3=" << c3 << endl;
		int_vec_print(cout, v, k + 1);
		cout << endl;
		exit(1);
		}
	}
	GFq.PG_element_normalize(v, stride, k + 1);
	x1 = v[2 * n * stride];
	x2 = v[(2 * n + 1) * stride];
	if (x1 == 0 && x2 == 0) {
		Sbar_rank(GFq, v, stride, n, a);
		return a;
		}
	a = nb_pts_Sbar(n, GFq.q);
	// determine b from an element on the projective line
	if (x1 == 1 && x2 == 0) {
		b = 0;
		}
	else {
		if (x2 != 1) {
			cout << "Qminus_rank x2 != 1" << endl;
			exit(1);
			}
		b = x1 + 1;
		}

	x = nb_pts_N1(n, GFq.q);
	//b = a / x;
	//c = a % x;
	u = GFq.product3(c1, x1, x1);
	vv = GFq.product3(c2, x1, x2);
	w = GFq.product3(c3, x2, x2);
	z = GFq.add3(u, vv, w);
	if (z == 0) {
		cout << "Qminus_rank z = 0" << endl;
		cout << "b=" << b << endl;
		exit(1);
		}
	
	minusz = GFq.negate(z);
	minuszv = GFq.inverse(minusz);
	if (minuszv != 1) {
		for (i = 0; i < n; i++) {
			v[2 * i * stride] = GFq.mult(v[2 * i * stride], minuszv);
			}
		}

	N1_rank(GFq, v, stride, n, c);
	a += b * x + c;
	return a;
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
// unrank functions for the hyperbolic quadric:
// #############################################################################



void S_unrank(finite_field &GFq,
		int *v, int stride, int n, int a)
{
	int l, i, j, x, y, u, q = GFq.q;
	int alpha, beta;
	
	if (n == 1) {
		if (a < q) {
			v[0 * stride] = a;
			v[1 * stride] = 0;
			return;
			}
		a -= (q - 1);
		if (a < q) {
			v[0 * stride] = 0;
			v[1 * stride] = a;
			return;
			}
		else {
			cout << "error in S_unrank n = 1 a = " << a << endl;
			exit(1);
			}
		}
	else {
		x = nb_pts_S(1, q);
		y = nb_pts_S(n - 1, q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			S_unrank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			S_unrank(GFq, v, stride, n - 1, j);
			return;
			}
		a -= l;
		//cout << "S_unrank subtracting " << l
		//<< " to bring a down to " << a << endl;
		x = nb_pts_N(1, q);
		y = nb_pts_N1(n - 1, q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			N_unrank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			N1_unrank(GFq, v, stride, n - 1, j);

			alpha = GFq.mult(v[2 * (n - 1) * stride],
					v[(2 * (n - 1) + 1) * stride]);
			beta = GFq.negate(alpha);
			for (u = 0; u < n - 1; u++) {
				v[2 * u * stride] = GFq.mult(v[2 * u * stride], beta);
				}
			return;
			}
		else {
			cout << "error in S_unrank n = " << n << ", a = " << a << endl;
			exit(1);
			}
		}
}

void N_unrank(finite_field &GFq,
		int *v, int stride, int n, int a)
{
	int l, i, j, k, j1, x, y, z, yz, u, q = GFq.q;
	int alpha, beta, gamma, delta, epsilon;
	
	if (n == 1) {
		x = q - 1;
		y = q - 1;
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			v[0 * stride] = 1 + j;
			v[1 * stride] = 1 + i;
			return;
			}
		else {
			cout << "error in N_unrank n = 1 a = " << a << endl;
			exit(1);
			}
		}
	else {
		x = nb_pts_S(1, q);
		y = nb_pts_N(n - 1, q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			S_unrank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			N_unrank(GFq, v, stride, n - 1, j);
			return;
			}
		a -= l;
		x = nb_pts_N(1, q);
		y = nb_pts_S(n - 1, q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			N_unrank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			S_unrank(GFq, v, stride, n - 1, j);
			return;
			}
		a -= l;
		x = nb_pts_N(1, q);
		y = (q - 2);
		z = nb_pts_N1(n - 1, q);
		yz = y * z;
		l = x * yz;
		if (a < l) {
			i = a / yz;
			j1 = a % yz;
			j = j1 / z;
			k = j1 % z;
			N_unrank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			N1_unrank(GFq, v, stride, n - 1, k);
			alpha = primitive_element(GFq);
			
			beta = GFq.power(alpha, j + 1);
			gamma = GFq.mult(v[(n - 1) * 2 * stride],
					v[((n - 1) * 2 + 1) * stride]);
			delta = GFq.negate(gamma);
			epsilon = GFq.mult(delta, beta);
			for (u = 0; u < n - 1; u++) {
				v[2 * u * stride] = GFq.mult(v[2 * u * stride], epsilon);
				}
			return;
			}
		else {
			cout << "error in N_unrank n = " << n << ", a = " << a << endl;
			exit(1);
			}
		}
}

void N1_unrank(finite_field &GFq, int *v, int stride, int n, int a)
{
	int l, i, j, k, j1, x, y, z, yz, u, q = GFq.q;
	int alpha, beta, gamma;
	
	if (n == 1) {
		l = q - 1;
		if (a < l) {
			alpha = a + 1;
			beta = GFq.inverse(alpha);
			//cout << "N1_unrank() n == 1, a = " << a
			// << " alpha = " << alpha << " beta = " << beta << endl;
			v[0 * stride] = alpha;
			v[1 * stride] = beta;
			return;
			}
		else {
			cout << "error in N1_unrank n = 1 a = " << a << endl;
			exit(1);
			}
		}
	else {
		x = nb_pts_S(1, q);
		y = nb_pts_N1(n - 1, q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			S_unrank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			N1_unrank(GFq, v, stride, n - 1, j);
			return;
			}
		a -= l;
		//cout << "N1_unrank subtracting " << l
		// << " to bring a down to " << a << endl;
		x = nb_pts_N1(1, q);
		y = nb_pts_S(n - 1, q);
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			N1_unrank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			S_unrank(GFq, v, stride, n - 1, j);
			return;
			}
		a -= l;
		//cout << "N1_unrank subtracting " << l
		// << " to bring a down to " << a << endl;
		x = nb_pts_N1(1, q);
		y = (q - 2); // zero for q = 2
		z = nb_pts_N1(n - 1, q);
		yz = y * z;
		l = x * yz; // zero for q = 2
		if (a < l) {
			// the case q = 2 does not appear here any more
			i = a / yz;
			j1 = a % yz;
			j = j1 / z;
			k = j1 % z;
			
			//cout << "a = " << a << endl;
			//cout << "y = " << y << endl;
			//cout << "z = " << z << endl;
			//cout << "i = a / yz = " << i << endl;
			//cout << "j1 = a % yz = " << j1 << endl;
			//cout << "j = j1 / z = " << j << endl;
			//cout << "k = j1 % z = " << k << endl;
			
			N1_unrank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			
			//cout << "(" << v[2 * (n - 1) * stride] << ","
			// << v[(2 * (n - 1) + 1) * stride] << ")" << endl;
			
			alpha = 2 + j;
			v[2 * (n - 1) * stride] = GFq.mult(
					v[2 * (n - 1) * stride], alpha);

			N1_unrank(GFq, v, stride, n - 1, k);
			
			//int_set_print(v, 2 * (n - 1));
			//cout << endl;
			
			beta = GFq.negate(alpha);
			gamma = GFq.add(beta, 1);
			
			//cout << "alpha = j + 2 = " << alpha << endl;
			//cout << "beta = - alpha = " << beta << endl;
			//cout << "gamma = beta + 1 = " << gamma << endl;
			
			for (u = 0; u < n - 1; u++) {
				v[2 * u * stride] = GFq.mult(v[2 * u * stride], gamma);
				}

			//int_set_print(v, 2 * n);
			//cout << endl;
			return;
			}
		else {
			cout << "error in N1_unrank n = " << n << ", a = " << a << endl;
			exit(1);
			}
		}
}

void Sbar_unrank(finite_field &GFq,
		int *v, int stride, int n, int a)
{
	int l, i, j, x, y, u, q = GFq.q;
	int alpha, beta;
	
	if (n == 1) {
		if (a == 0) {
			v[0 * stride] = 1;
			v[1 * stride] = 0;
			return;
			}
		if (a == 1) {
			v[0 * stride] = 0;
			v[1 * stride] = 1;
			return;
			}
		else {
			cout << "error in Sbar_unrank n = 1 a = " << a << endl;
			exit(1);
			}
		}
	else {
		y = nb_pts_Sbar(n - 1, q);
		l = y;
		if (a < l) {
			u = n - 1;
			v[2 * u * stride] = 0;
			v[(2 * u + 1) * stride] = 0;
			Sbar_unrank(GFq, v, stride, n - 1, a);
			return;
			}
		a -= l;
		//cout << "subtracting " << l << " to bring a to " << a << endl;
		x = nb_pts_Sbar(1, q);
		y = nb_pts_S(n - 1, q);
		//cout << "nb_pts_S(" << n - 1 << ") = " << y << endl;
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			Sbar_unrank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			S_unrank(GFq, v, stride, n - 1, j);
			return;
			}
		a -= l;
		//cout << "subtracting " << l << " to bring a to " << a << endl;
		x = nb_pts_Nbar(1, q);
		y = nb_pts_N1(n - 1, q);
		//cout << "nb_pts_N1(" << n - 1 << ") = " << y << endl;
		l = x * y;
		if (a < l) {
			i = a / y;
			j = a % y;
			//cout << "i=" << i << " j=" << j << endl;
			Nbar_unrank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			//cout << "(" << v[2 * (n - 1) * stride] << ","
			//<< v[(2 * (n - 1) + 1) * stride] << ")" << endl;
			N1_unrank(GFq, v, stride, n - 1, j);

			alpha = GFq.mult(v[2 * (n - 1) * stride],
					v[(2 * (n - 1) + 1) * stride]);
			beta = GFq.negate(alpha);
			for (u = 0; u < n - 1; u++) {
				v[2 * u * stride] = GFq.mult(v[2 * u * stride], beta);
				}
			//int_set_print(v, 2 * n);
			//cout << endl;
			return;
			}
		else {
			cout << "error in Sbar_unrank n = " << n
					<< ", a = " << a << endl;
			exit(1);
			}
		}
}

void Nbar_unrank(finite_field &GFq,
		int *v, int stride, int n, int a)
{
	int y, l, q = GFq.q;
	
	if (n == 1) {
		y = q - 1;
		l = y;
		if (a < l) {
			v[0 * stride] = 1 + a;
			v[1 * stride] = 1;
			return;
			}
		else {
			cout << "error in Nbar_unrank n = 1 a = " << a << endl;
			exit(1);
			}
		}
	else {
		cout << "Nbar_unrank only defined for n = 1" << endl;
		exit(1);
		}
}


// #############################################################################
// rank functions for the hyperbolic quadric:
// #############################################################################



void S_rank(finite_field &GFq,
		int *v, int stride, int n, int &a)
{
	int l, i, j, x, y, u, q = GFq.q;
	int alpha, beta, gamma, delta, epsilon;
	
	if (n == 1) {
		if (v[1 * stride] == 0) {
			a = v[0 * stride];
			return;
			}
		if (v[0 * stride]) {
			cout << "error in S_rank v[0] not null" << endl;
			exit(1);
			}
		a = q - 1;
		a += v[1 * stride];
		}
	else {
		x = nb_pts_S(1, q);
		y = nb_pts_S(n - 1, q);
		l = x * y;
		alpha = GFq.mult(v[2 * (n - 1) * stride],
				v[(2 * (n - 1) + 1) * stride]);
		if (alpha == 0) {
			S_rank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			S_rank(GFq, v, stride, n - 1, j);
			a = i * y + j;
			return;
			}
		a = l;
		x = nb_pts_N(1, q);
		y = nb_pts_N1(n - 1, q);
		
		N_rank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
		
		
		beta = GFq.negate(alpha);
		gamma = evaluate_hyperbolic_quadratic_form(
				GFq, v, stride, n - 1);
		if (gamma != beta) {
			cout << "error in S_rank gamma != beta" << endl;
			exit(1);
			}
		delta = GFq.inverse(beta);
		for (u = 0; u < n - 1; u++) {
			v[2 * u * stride] = GFq.mult(v[2 * u * stride], delta);
			}
		epsilon = evaluate_hyperbolic_quadratic_form(
				GFq, v, stride, n - 1);
		if (epsilon != 1) {
			cout << "error in S_rank epsilon != 1" << endl;
			exit(1);
			}
		N1_rank(GFq, v, stride, n - 1, j);
		a += i * y + j;
		}
}

void N_rank(finite_field &GFq,
		int *v, int stride, int n, int &a)
{
	int l, i, j, k, x, y, z, yz, u, q = GFq.q;
	int alpha, beta, gamma, delta;
	int epsilon, gamma2, epsilon_inv;
	
	if (n == 1) {
		x = q - 1;
		y = q - 1;
		if (v[0 * stride] == 0 || v[1 * stride] == 0) {
			cout << "N_rank() v[0 * stride] == 0 || "
					"v[1 * stride] == 0" << endl;
			exit(1);
			}
		j = v[0 * stride] - 1;
		i = v[1 * stride] - 1;
		a = i * y + j;
		}
	else {
		gamma = GFq.mult(v[(n - 1) * 2 * stride],
				v[((n - 1) * 2 + 1) * stride]);
		x = nb_pts_S(1, q);
		y = nb_pts_N(n - 1, q);
		l = x * y;
		if (gamma == 0) {
			S_rank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			N_rank(GFq, v, stride, n - 1, j);
			a = i * y + j;
			return;
			}
		a = l;
		x = nb_pts_N(1, q);
		y = nb_pts_S(n - 1, q);
		l = x * y;
		gamma2 = evaluate_hyperbolic_quadratic_form(
				GFq, v, stride, n - 1);
		if (gamma2 == 0) {
			N_rank(GFq, v + (n - 1) * 2, stride, 1, i);
			S_rank(GFq, v, stride, n - 1, j);
			a += i * y + j;
			}
		a += l;

		x = nb_pts_N(1, q);
		y = (q - 2);
		z = nb_pts_N1(n - 1, q);
		yz = y * z;
		l = x * yz;

		N_rank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
		alpha = primitive_element(GFq);
		delta = GFq.negate(gamma);
		for (j = 0; j < q - 2; j++) {
			beta = GFq.power(alpha, j + 1);
			epsilon = GFq.mult(delta, beta);
			if (epsilon == gamma2) {
				epsilon_inv = GFq.inverse(epsilon);
				for (u = 0; u < n - 1; u++) {
					v[2 * u * stride] = GFq.mult(
							v[2 * u * stride], epsilon_inv);
					}
				N1_rank(GFq, v, stride, n - 1, k);
				a += i * yz + j * z + k;
				return;
				}
			}
		cout << "error, gamma2 not found" << endl;
		exit(1);
		}
}

void N1_rank(finite_field &GFq, int *v, int stride, int n, int &a)
{
	int l, i, j, k, x, y, z, yz, u, q = GFq.q;
	int alpha, alpha_inv, beta, gamma, gamma2, gamma_inv;
	
	if (n == 1) {
		alpha = v[0 * stride];
		beta = v[1 * stride];
		if (alpha == 0 || beta == 0) {
			cout << "N1_rank() alpha == 0 || beta == 0" << endl;
			exit(1);
			}
		gamma = GFq.inverse(alpha);
		if (gamma != beta) {
			cout << "error in N1_rank gamma = " << gamma
					<< " != beta = " << beta << endl;
			exit(1);
			}
		a = alpha - 1;
		}
	else {
		a = 0;
		alpha = GFq.mult(v[2 * (n - 1) * stride],
				v[(2 * (n - 1) + 1) * stride]);
		x = nb_pts_S(1, q);
		y = nb_pts_N1(n - 1, q);
		l = x * y;
		if (alpha == 0) {
			S_rank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			N1_rank(GFq, v, stride, n - 1, j);
			a = i * y + j;
			return;
			}
		a += l;
		gamma2 = evaluate_hyperbolic_quadratic_form(GFq,
				v, stride, n - 1);
		x = nb_pts_N1(1, q);
		y = nb_pts_S(n - 1, q);
		l = x * y;
		if (gamma2 == 0) {
			N1_rank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			S_rank(GFq, v, stride, n - 1, j);
			a += i * y + j;
			return;
			}
		a += l;
		// the case q = 2 does not appear here any more
		if (q == 2) {
			cout << "N1_rank the case q=2 should not appear here" << endl;
			exit(1);
			}


		x = nb_pts_N1(1, q);
		y = (q - 2); // zero for q = 2
		z = nb_pts_N1(n - 1, q);
		yz = y * z;
		l = x * yz; // zero for q = 2

		alpha = GFq.mult(v[2 * (n - 1) * stride],
				v[(2 * (n - 1) + 1) * stride]);
		if (alpha == 0) {
			cout << "N1_rank alpha == 0" << endl;
			exit(1);
			}
		if (alpha == 1) {
			cout << "N1_rank alpha == 1" << endl;
			exit(1);
			}
		j = alpha - 2;
		alpha_inv = GFq.inverse(alpha);
		v[2 * (n - 1) * stride] = GFq.mult(
				v[2 * (n - 1) * stride], alpha_inv);
		
		N1_rank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
		
		gamma2 = evaluate_hyperbolic_quadratic_form(GFq,
				v, stride, n - 1);
		if (gamma2 == 0) {
			cout << "N1_rank gamma2 == 0" << endl;
			exit(1);
			}
		if (gamma2 == 1) {
			cout << "N1_rank gamma2 == 1" << endl;
			exit(1);
			}
		gamma_inv = GFq.inverse(gamma2);
		for (u = 0; u < n - 1; u++) {
			v[2 * u * stride] = GFq.mult(v[2 * u * stride], gamma_inv);
			}
		N1_rank(GFq, v, stride, n - 1, k);

		a += i * yz + j * z + k;
		
		}
}

void Sbar_rank(finite_field &GFq,
		int *v, int stride, int n, int &a)
{
	int l, i, j, x, y, u, q = GFq.q;
	int alpha, beta, beta2, beta_inv;
	
	GFq.PG_element_normalize(v, stride, 2 * n);
	if (n == 1) {
		// test for (1,0) or (0,1):
		if (v[0 * stride] == 1 && v[1 * stride] == 0) {
			a = 0;
			return;
			}
		if (v[0 * stride] == 0 && v[1 * stride] == 1) {
			a = 1;
			return;
			}
		else {
			cout << "error in Sbar_rank n = 1 bad vector" << endl;
			if (stride == 1) {
				int_vec_print(cout, v, 2);
				}
			exit(1);
			}
		}
	else {
		a = 0;
		// test for leading (0,0):
		if (v[2 * (n - 1) * stride] == 0 &&
				v[(2 * (n - 1) + 1) * stride] == 0) {
			// rank Sbar for the rest:
			Sbar_rank(GFq, v, stride, n - 1, a);
			return;
			}
		l = nb_pts_Sbar(n - 1, q);
		a += l;

		// alpha = form value for the top two coefficients:
		alpha = GFq.mult(v[2 * (n - 1) * stride],
				v[(2 * (n - 1) + 1) * stride]);
		x = nb_pts_Sbar(1, q); // = 2
		y = nb_pts_S(n - 1, q);

		// test for 0 + 0 
		// (i.e. 0 = alpha = value of the form on
		// the top two coefficients
		// and 0 for value of the form on the rest):
		if (alpha == 0) {
			Sbar_rank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);
			S_rank(GFq, v, stride, n - 1, j);
			a += i * y + j;
			//cout << "i*y+j=" << i << "*" << y << "+" << j << endl;
			return;
			}

		l = x * y;
		a += l;
		// now it must be n + (-n) for some n \neq 0
		// (i.e. n = alpha = value of the form on
		// the top two coefficients and
		// -n for the value of the form on the rest): 
		x = nb_pts_Nbar(1, q);
		y = nb_pts_N1(n - 1, q);
		Nbar_rank(GFq, v + (n - 1) * 2 * stride, stride, 1, i);

		beta = GFq.negate(alpha);
		// beta = - alpha
		beta2 = evaluate_hyperbolic_quadratic_form(GFq,
				v, stride, n - 1);
		// beta2 = value of the quadratic form on the rest
		// must be - alpha (otherwise the vector does
		// not represent a point in Sbar)
		if (beta2 != beta) {
			cout << "error in Sbar_rank beta2 != beta" << endl;
			exit(1);
			}
		beta_inv = GFq.inverse(beta);
		// divide by beta so that the quadratic form
		// on the rest is equal to 1.
		for (u = 0; u < n - 1; u++) {
			v[2 * u * stride] = GFq.mult(
					v[2 * u * stride], beta_inv);
			}
		// rank the N1 part:
		N1_rank(GFq, v, stride, n - 1, j);
		a += i * y + j;
		}
}

void Nbar_rank(finite_field &GFq,
		int *v, int stride, int n, int &a)
{
	if (n == 1) {
		if (v[1 * stride] != 1) {
			cout << "error in Nbar_rank n = 1 v[1 * stride] != 1" << endl;
			exit(1);
			}
		if (v[0 * stride] == 0) {
			cout << "error in Nbar_rank n = 1 v[0 * stride] == 0" << endl;
			exit(1);
			}
		a = v[0 * stride] - 1;
		return;
		}
	else {
		cout << "Nbar_rank only defined for n = 1" << endl;
		exit(1);
		}
}


// #############################################################################
// other stuff:
// #############################################################################


int evaluate_hyperbolic_quadratic_form(finite_field &GFq,
		int *v, int stride, int n)
{
	int alpha = 0, beta, u;
	
	for (u = 0; u < n; u++) {
		beta = GFq.mult(v[2 * u * stride], v[(2 * u + 1) * stride]);
		alpha = GFq.add(alpha, beta);
		}
	return alpha;
}

int evaluate_hyperbolic_bilinear_form(finite_field &GFq,
		int *u, int *v, int n)
{
	int alpha = 0, beta1, beta2, i;
	
	for (i = 0; i < n; i++) {
		beta1 = GFq.mult(u[2 * i], v[2 * i + 1]);
		beta2 = GFq.mult(u[2 * i + 1], v[2 * i]);
		alpha = GFq.add(alpha, beta1);
		alpha = GFq.add(alpha, beta2);
		}
	return alpha;
}

int primitive_element(finite_field &GFq)
{
	if (GFq.e == 1) {
		return primitive_root(GFq.p, FALSE);
		}
	return GFq.p;
}

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
		
		factor_prime_power(q, p, e);
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

void Gram_matrix(finite_field &GFq, int epsilon, int k, 
	int form_c1, int form_c2, int form_c3, 
	int *&Gram)
{
	int d = k + 1;
	int n, i, j, offset = 0;

	Gram = NEW_int(d * d);
	for (i = 0; i < d * d; i++) {
		Gram[i] = 0;
		}
	n = Witt_index(epsilon, k);
	if (epsilon == 0) {
		Gram[0 * d + 0] = GFq.add(form_c1, form_c1);
		offset = 1;
		}
	else if (epsilon == 1) {
		}
	else if (epsilon == -1) {
		Gram[(d - 2) * d + d - 2] = GFq.add(form_c1, form_c1);
		Gram[(d - 2) * d + d - 1] = form_c2;
		Gram[(d - 1) * d + d - 2] = form_c2;
		Gram[(d - 1) * d + d - 1] = GFq.add(form_c3, form_c3);
		}
	for (i = 0; i < n; i++) {
		j = 2 * i;
		Gram[(offset + j) * d + offset + j + 1] = 1;
		Gram[(offset + j + 1) * d + offset + j] = 1;
		}
}

int evaluate_bilinear_form(finite_field &GFq,
		int *u, int *v, int d, int *Gram)
{
	int i, j, a, b, c, e, A;
	
	A = 0;
	for (i = 0; i < d; i++) {
		a = u[i];
		for (j = 0; j < d; j++) {
			b = Gram[i * d + j];
			c = v[j];
			e = GFq.mult(a, b);
			e = GFq.mult(e, c);
			A = GFq.add(A, e);
			}
		}
	return A;
}

void Siegel_Transformation(finite_field &GFq,
	int epsilon, int k,
	int form_c1, int form_c2, int form_c3,
	int *M, int *v, int *u, int verbose_level)
// if u is singular and v \in \la u \ra^\perp, then
// \pho_{u,v}(x) := x + \beta(x,v) u - \beta(x,u) v - Q(v) \beta(x,u) u
// is called the Siegel transform (see Taylor p. 148)
// Here Q is the quadratic form
// and \beta is the corresponding bilinear form
{
	int f_v = (verbose_level >= 1);
	int d = k + 1;
	int i, j, Qv, a, b, c, e;
	int *Gram;
	int *new_Gram;
	int *N1;
	int *N2;
	int *w;
	
	if (f_v) {
		cout << "Siegel_Transformation v=";
		int_vec_print(cout, v, d);
		cout << " u=";
		int_vec_print(cout, u, d);
		cout << endl;
		}
	Gram_matrix(GFq, epsilon, k,
			form_c1, form_c2, form_c3, Gram);
	Qv = evaluate_quadratic_form(GFq, v, 1 /*stride*/,
			epsilon, k, form_c1, form_c2, form_c3);
	if (f_v) {
		cout << "Qv=" << Qv << endl;
		}
	N1 = NEW_int(d * d);
	N2 = NEW_int(d * d);
	new_Gram = NEW_int(d * d);
	w = NEW_int(d);
	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			if (i == j) 
				M[i * d + j] = 1;
			else
				M[i * d + j] = 0;
			}
		}
	// compute w^T := Gram * v^T
	for (i = 0; i < d; i++) {
		a = 0;
		for (j = 0; j < d; j++) {
			b = Gram[i * d + j];
			c = v[j];
			e = GFq.mult(b, c);
			a = GFq.add(a, e);
			}
		w[i] = a;
		}
	// M := M + w^T * u
	for (i = 0; i < d; i++) {
		b = w[i];
		for (j = 0; j < d; j++) {
			c = u[j];
			e = GFq.mult(b, c);
			M[i * d + j] = GFq.add(M[i * d + j], e);
			}
		}
	// compute w^T := Gram * u^T
	for (i = 0; i < d; i++) {
		a = 0;
		for (j = 0; j < d; j++) {
			b = Gram[i * d + j];
			c = u[j];
			e = GFq.mult(b, c);
			a = GFq.add(a, e);
			}
		w[i] = a;
		}
	// M := M - w^T * v
	for (i = 0; i < d; i++) {
		b = w[i];
		for (j = 0; j < d; j++) {
			c = v[j];
			e = GFq.mult(b, c);
			M[i * d + j] = GFq.add(M[i * d + j], GFq.negate(e));
			}
		}
	// M := M - Q(v) * w^T * u
	for (i = 0; i < d; i++) {
		b = w[i];
		for (j = 0; j < d; j++) {
			c = u[j];
			e = GFq.mult(b, c);
			M[i * d + j] =
					GFq.add(M[i * d + j], GFq.mult(GFq.negate(e), Qv));
			}
		}
	if (f_v) {
		cout << "Siegel matrix:" << endl;
		print_integer_matrix_width(cout, M, d, d, d, 2);
		//GFq.transform_form_matrix(M, Gram, new_Gram, N1, N2, d);
		//cout << "transformed Gram matrix:" << endl;
		//print_integer_matrix_width(cout, new_Gram, d, d, d, 2);
		//cout << endl;
		}
	
	FREE_int(Gram);
	FREE_int(new_Gram);
	FREE_int(N1);
	FREE_int(N2);
	FREE_int(w);
}

void choose_anisotropic_form(finite_field &GFq,
		int &c1, int &c2, int &c3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	unipoly_domain FX(&GFq);
	unipoly_object m;
	
	if (f_v) {
		cout << "choose_anisotropic_form over GF(" << GFq.q << ")" << endl;
		}

	if (ODD(GFq.q)) {
		c1 = 1;
		c2 = 0;
		c3 = GFq.negate(primitive_element(GFq));
		}
	else {
		FX.create_object_by_rank_string(m,
				get_primitive_polynomial(GFq.q, 2, 0),
				verbose_level);
	
		//FX.create_object_by_rank_string(m,
		//get_primitive_polynomial(GFq.p, 2 * GFq.e, 0), verbose_level);
	
		if (f_v) {
			cout << "choosing the following primitive polynomial:" << endl;
			FX.print_object(m, cout); cout << endl;
			}
		
		int *rep = (int *) m;
		int *coeff = rep + 1;
		c1 = coeff[2];
		c2 = coeff[1];
		c3 = coeff[0];
		}

#if 0
	finite_field GFQ;

	GFQ.init(GFq.q * GFq.q, 0);
	cout << "choose_anisotropic_form created field GF("
			<< GFQ.q << ")" << endl;

	c1 = 1;
	c2 = GFQ.negate(GFQ.T2(GFQ.p));
	c3 = GFQ.N2(GFQ.p);
	if (f_v) {
		cout << "c1=" << c1 << " c2=" << c2 << " c3=" << c3 << endl;
		}

	c2 = GFQ.retract(GFq, 2, c2, verbose_level);
	c3 = GFQ.retract(GFq, 2, c3, verbose_level);
	if (f_v) {
		cout << "after retract:" << endl;
		cout << "c1=" << c1 << " c2=" << c2 << " c3=" << c3 << endl;
		}
#endif

	if (f_v) {
		cout << "choose_anisotropic_form over GF(" << GFq.q
				<< "): choosing c1=" << c1 << ", c2=" << c2
				<< ", c3=" << c3 << endl;
		}
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
		choose_anisotropic_form(GFq, c1, c2, c3, TRUE);
		}
	for (i = 0; i < nb; i++) {
		Q_epsilon_unrank(GFq, v, stride, epsilon, k, c1, c2, c3, i);
		
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
		a = evaluate_quadratic_form(GFq, v, stride, epsilon, k,
				c1, c2, c3);
		cout << a;
		j = Q_epsilon_rank(GFq, v, stride, epsilon, k, c1, c2, c3);
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
		Sbar_unrank(GFq, v, stride, n, i);
		cout << i << " : ";
		int_set_print(v, 2 * n);
		cout << " : ";
		a = evaluate_hyperbolic_quadratic_form(GFq, v, stride, n);
		cout << a;
		Sbar_rank(GFq, v, stride, n, j);
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

void orthogonal_Siegel_map_between_singular_points(int *T, 
	int rk_from, int rk_to, int root, 
	finite_field &GFq, int epsilon, int algebraic_dimension, 
	int form_c1, int form_c2, int form_c3, int *Gram_matrix, 
	int verbose_level)
// root is not perp to from and to.
{
	int *B, *Bv, *w, *z, *x;
	int i, j, a, b, av, bv, minus_one;
	int d, k; //, epsilon, form_c1, form_c2, form_c3;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "Siegel_map_between_singular_points "
				"rk_from=" << rk_from
				<< " rk_to=" << rk_to
				<< " root=" << root << endl;
		}
	d = algebraic_dimension;
	k = d - 1;
	
	B = NEW_int(d * d);
	Bv = NEW_int(d * d);
	w = NEW_int(d);
	z = NEW_int(d);
	x = NEW_int(d);
	Q_epsilon_unrank(GFq, B, 1, epsilon, k,
			form_c1, form_c2, form_c3, root);
	Q_epsilon_unrank(GFq, B + d, 1, epsilon, k,
			form_c1, form_c2, form_c3, rk_from);
	Q_epsilon_unrank(GFq, w, 1, epsilon, k,
			form_c1, form_c2, form_c3, rk_to);
	if (f_vv) {
		cout << "    root=";
		int_vec_print(cout, B, d);
		cout << endl;
		cout << " rk_from=";
		int_vec_print(cout, B + d, d);
		cout << endl;
		cout << "   rk_to=";
		int_vec_print(cout, w, d);
		cout << endl;
		}
	
	a = evaluate_bilinear_form(GFq, B, B + d, d, Gram_matrix);
	b = evaluate_bilinear_form(GFq, B, w, d, Gram_matrix);
	av = GFq.inverse(a);
	bv = GFq.inverse(b);
	for (i = 0; i < d; i++) {
		B[d + i] = GFq.mult(B[d + i], av);
		w[i] = GFq.mult(w[i], bv);
		}
	if (f_vv) {
		cout << "after scaling:" << endl;
		cout << " rk_from=";
		int_vec_print(cout, B + d, d);
		cout << endl;
		cout << "   rk_to=";
		int_vec_print(cout, w, d);
		cout << endl;
		}
	for (i = 2; i < d; i++) {
		for (j = 0; j < d; j++) {
			B[i * d + j] = 0;
			}
		}
	
	if (f_vv) {
		cout << "before perp, the matrix B is:" << endl;
		print_integer_matrix(cout, B, d, d);
		}
	GFq.perp(d, 2, B, Gram_matrix);
	if (f_vv) {
		cout << "after perp, the matrix B is:" << endl;
		print_integer_matrix(cout, B, d, d);
		}
	GFq.invert_matrix(B, Bv, d);
	if (f_vv) {
		cout << "the matrix Bv = B^{-1} is:" << endl;
		print_integer_matrix(cout, B, d, d);
		}
	GFq.mult_matrix_matrix(w, Bv, z, 1, d, d,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "the coefficient vector z = w * Bv is:" << endl;
		int_vec_print(cout, z, d);
		cout << endl;
		}
	z[0] = 0;
	z[1] = 0;
	if (f_vv) {
		cout << "we zero out the first two coordinates:" << endl;
		int_vec_print(cout, z, d);
		cout << endl;
		}
	GFq.mult_matrix_matrix(z, B, x, 1, d, d,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "the vector x = z * B is:" << endl;
		int_vec_print(cout, x, d);
		cout << endl;
		}
	minus_one = GFq.negate(1);
	for (i = 0; i < d; i++) {
		x[i] = GFq.mult(x[i], minus_one);
		}
	if (f_vv) {
		cout << "the vector -x is:" << endl;
		int_vec_print(cout, x, d);
		cout << endl;
		}
	Siegel_Transformation(GFq, epsilon, d - 1, 
		form_c1, form_c2, form_c3, T, x, B, f_vv);
	if (f_v) {
		cout << "the Siegel transformation is:" << endl;
		print_integer_matrix(cout, T, d, d);
		}
	FREE_int(B);
	FREE_int(Bv);
	FREE_int(w);
	FREE_int(z);
	FREE_int(x);
}

int orthogonal_find_root(int rk2, 
	finite_field &GFq, int epsilon, int algebraic_dimension, 
	int form_c1, int form_c2, int form_c3, int *Gram_matrix, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *x, *y, *z;
	int d, k, i;
	//int epsilon, d, k, form_c1, form_c2, form_c3, i;
	int y2_minus_y3, minus_y1, y3_minus_y2, a, a2, u, v, root;
	
	d = algebraic_dimension;
	k = d - 1;
	if (f_v) {
		cout << "orthogonal_find_root rk2=" << rk2 << endl;
		}
	if (rk2 == 0) {
		cout << "orthogonal_find_root: rk2 must not be 0" << endl;
		exit(1);
		}
	//epsilon = orthogonal_epsilon;
	//d = orthogonal_d;
	//k = d - 1;
	//form_c1 = orthogonal_form_c1;
	//form_c2 = orthogonal_form_c2;
	//form_c3 = orthogonal_form_c3;
	x = NEW_int(d);
	y = NEW_int(d);
	z = NEW_int(d);
	for (i = 0; i < d; i++) {
		x[i] = 0;
		z[i] = 0;
		}
	x[0] = 1;
	
	Q_epsilon_unrank(GFq, y, 1, epsilon, k,
			form_c1, form_c2, form_c3, rk2);
	if (y[0]) {
		z[1] = 1;
		goto finish;
		}
	if (y[1] == 0) {
		for (i = 2; i < d; i++) {
			if (y[i]) {
				if (EVEN(i)) {
					z[1] = 1;
					z[i + 1] = 1;
					goto finish;
					}
				else {
					z[1] = 1;
					z[i - 1] = 1;
					goto finish;
					}
				}
			}
		cout << "error: y is zero vector" << endl;
		}
	y2_minus_y3 = GFq.add(y[2], GFq.negate(y[3]));
	minus_y1 = GFq.negate(y[1]);
	if (minus_y1 != y2_minus_y3) {
		z[0] = 1;
		z[1] = 1;
		z[2] = GFq.negate(1);
		z[3] = 1;
		goto finish;
		}
	y3_minus_y2 = GFq.add(y[3], GFq.negate(y[2]));
	if (minus_y1 != y3_minus_y2) {
		z[0] = 1;
		z[1] = 1;
		z[2] = 1;
		z[3] = GFq.negate(1);
		goto finish;
		}
	// now we are in characteristic 2
	if (GFq.q == 2) {
		if (y[2] == 0) {
			z[1] = 1;
			z[2] = 1;
			goto finish;
			}
		else if (y[3] == 0) {
			z[1] = 1;
			z[3] = 1;
			goto finish;
			}
		cout << "error neither y2 nor y3 is zero" << endl;
		exit(1);
		}
	// now the field has at least 4 elements
	a = 3;
	a2 = GFq.mult(a, a);
	z[0] = a2;
	z[1] = 1;
	z[2] = a;
	z[3] = a;
finish:

	u = evaluate_bilinear_form(GFq, z, x, d, Gram_matrix);
	if (u == 0) {
		cout << "u=" << u << endl;
		exit(1);
		}
	v = evaluate_bilinear_form(GFq, z, y, d, Gram_matrix);
	if (v == 0) {
		cout << "v=" << v << endl;
		exit(1);
		}
	root = Q_epsilon_rank(GFq, z, 1, epsilon, k,
			form_c1, form_c2, form_c3);
	if (f_v) {
		cout << "orthogonal_find_root root=" << root << endl;
		}
	
	FREE_int(x);
	FREE_int(y);
	FREE_int(z);
	
	return root;
}

void orthogonal_points_free_global_data()
{
	cout << "orthogonal_points_free_global_data" << endl;
	if (Hash_table_parabolic) {
		delete Hash_table_parabolic;
		Hash_table_parabolic = NULL;
		}
	cout << "orthogonal_points_free_global_data done" << endl;
}


