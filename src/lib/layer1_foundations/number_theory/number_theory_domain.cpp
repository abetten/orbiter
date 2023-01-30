// number_theory_domain.cpp
//
// Anton Betten
// April 3, 2003

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace number_theory {


//                         The First 1,000 Primes
//                          (the 1,000th is 7919)
//         For more information on primes see http://primes.utm.edu/

static int the_first_thousand_primes[] = {
      2,     3,     5,     7,    11,    13,    17,    19,    23,    29
,    31,    37,    41,    43,    47,    53,    59,    61,    67,    71
,    73,    79,    83,    89,    97,   101,   103,   107,   109,   113
,   127,   131,   137,   139,   149,   151,   157,   163,   167,   173
,   179,   181,   191,   193,   197,   199,   211,   223,   227,   229
,   233,   239,   241,   251,   257,   263,   269,   271,   277,   281
,   283,   293,   307,   311,   313,   317,   331,   337,   347,   349
,   353,   359,   367,   373,   379,   383,   389,   397,   401,   409
,   419,   421,   431,   433,   439,   443,   449,   457,   461,   463
,   467,   479,   487,   491,   499,   503,   509,   521,   523,   541
,   547,   557,   563,   569,   571,   577,   587,   593,   599,   601
,   607,   613,   617,   619,   631,   641,   643,   647,   653,   659
,   661,   673,   677,   683,   691,   701,   709,   719,   727,   733
,   739,   743,   751,   757,   761,   769,   773,   787,   797,   809
,   811,   821,   823,   827,   829,   839,   853,   857,   859,   863
,   877,   881,   883,   887,   907,   911,   919,   929,   937,   941
,   947,   953,   967,   971,   977,   983,   991,   997,  1009,  1013
,  1019,  1021,  1031,  1033,  1039,  1049,  1051,  1061,  1063,  1069
,  1087,  1091,  1093,  1097,  1103,  1109,  1117,  1123,  1129,  1151
,  1153,  1163,  1171,  1181,  1187,  1193,  1201,  1213,  1217,  1223
,  1229,  1231,  1237,  1249,  1259,  1277,  1279,  1283,  1289,  1291
,  1297,  1301,  1303,  1307,  1319,  1321,  1327,  1361,  1367,  1373
,  1381,  1399,  1409,  1423,  1427,  1429,  1433,  1439,  1447,  1451
,  1453,  1459,  1471,  1481,  1483,  1487,  1489,  1493,  1499,  1511
,  1523,  1531,  1543,  1549,  1553,  1559,  1567,  1571,  1579,  1583
,  1597,  1601,  1607,  1609,  1613,  1619,  1621,  1627,  1637,  1657
,  1663,  1667,  1669,  1693,  1697,  1699,  1709,  1721,  1723,  1733
,  1741,  1747,  1753,  1759,  1777,  1783,  1787,  1789,  1801,  1811
,  1823,  1831,  1847,  1861,  1867,  1871,  1873,  1877,  1879,  1889
,  1901,  1907,  1913,  1931,  1933,  1949,  1951,  1973,  1979,  1987
,  1993,  1997,  1999,  2003,  2011,  2017,  2027,  2029,  2039,  2053
,  2063,  2069,  2081,  2083,  2087,  2089,  2099,  2111,  2113,  2129
,  2131,  2137,  2141,  2143,  2153,  2161,  2179,  2203,  2207,  2213
,  2221,  2237,  2239,  2243,  2251,  2267,  2269,  2273,  2281,  2287
,  2293,  2297,  2309,  2311,  2333,  2339,  2341,  2347,  2351,  2357
,  2371,  2377,  2381,  2383,  2389,  2393,  2399,  2411,  2417,  2423
,  2437,  2441,  2447,  2459,  2467,  2473,  2477,  2503,  2521,  2531
,  2539,  2543,  2549,  2551,  2557,  2579,  2591,  2593,  2609,  2617
,  2621,  2633,  2647,  2657,  2659,  2663,  2671,  2677,  2683,  2687
,  2689,  2693,  2699,  2707,  2711,  2713,  2719,  2729,  2731,  2741
,  2749,  2753,  2767,  2777,  2789,  2791,  2797,  2801,  2803,  2819
,  2833,  2837,  2843,  2851,  2857,  2861,  2879,  2887,  2897,  2903
,  2909,  2917,  2927,  2939,  2953,  2957,  2963,  2969,  2971,  2999
,  3001,  3011,  3019,  3023,  3037,  3041,  3049,  3061,  3067,  3079
,  3083,  3089,  3109,  3119,  3121,  3137,  3163,  3167,  3169,  3181
,  3187,  3191,  3203,  3209,  3217,  3221,  3229,  3251,  3253,  3257
,  3259,  3271,  3299,  3301,  3307,  3313,  3319,  3323,  3329,  3331
,  3343,  3347,  3359,  3361,  3371,  3373,  3389,  3391,  3407,  3413
,  3433,  3449,  3457,  3461,  3463,  3467,  3469,  3491,  3499,  3511
,  3517,  3527,  3529,  3533,  3539,  3541,  3547,  3557,  3559,  3571
,  3581,  3583,  3593,  3607,  3613,  3617,  3623,  3631,  3637,  3643
,  3659,  3671,  3673,  3677,  3691,  3697,  3701,  3709,  3719,  3727
,  3733,  3739,  3761,  3767,  3769,  3779,  3793,  3797,  3803,  3821
,  3823,  3833,  3847,  3851,  3853,  3863,  3877,  3881,  3889,  3907
,  3911,  3917,  3919,  3923,  3929,  3931,  3943,  3947,  3967,  3989
,  4001,  4003,  4007,  4013,  4019,  4021,  4027,  4049,  4051,  4057
,  4073,  4079,  4091,  4093,  4099,  4111,  4127,  4129,  4133,  4139
,  4153,  4157,  4159,  4177,  4201,  4211,  4217,  4219,  4229,  4231
,  4241,  4243,  4253,  4259,  4261,  4271,  4273,  4283,  4289,  4297
,  4327,  4337,  4339,  4349,  4357,  4363,  4373,  4391,  4397,  4409
,  4421,  4423,  4441,  4447,  4451,  4457,  4463,  4481,  4483,  4493
,  4507,  4513,  4517,  4519,  4523,  4547,  4549,  4561,  4567,  4583
,  4591,  4597,  4603,  4621,  4637,  4639,  4643,  4649,  4651,  4657
,  4663,  4673,  4679,  4691,  4703,  4721,  4723,  4729,  4733,  4751
,  4759,  4783,  4787,  4789,  4793,  4799,  4801,  4813,  4817,  4831
,  4861,  4871,  4877,  4889,  4903,  4909,  4919,  4931,  4933,  4937
,  4943,  4951,  4957,  4967,  4969,  4973,  4987,  4993,  4999,  5003
,  5009,  5011,  5021,  5023,  5039,  5051,  5059,  5077,  5081,  5087
,  5099,  5101,  5107,  5113,  5119,  5147,  5153,  5167,  5171,  5179
,  5189,  5197,  5209,  5227,  5231,  5233,  5237,  5261,  5273,  5279
,  5281,  5297,  5303,  5309,  5323,  5333,  5347,  5351,  5381,  5387
,  5393,  5399,  5407,  5413,  5417,  5419,  5431,  5437,  5441,  5443
,  5449,  5471,  5477,  5479,  5483,  5501,  5503,  5507,  5519,  5521
,  5527,  5531,  5557,  5563,  5569,  5573,  5581,  5591,  5623,  5639
,  5641,  5647,  5651,  5653,  5657,  5659,  5669,  5683,  5689,  5693
,  5701,  5711,  5717,  5737,  5741,  5743,  5749,  5779,  5783,  5791
,  5801,  5807,  5813,  5821,  5827,  5839,  5843,  5849,  5851,  5857
,  5861,  5867,  5869,  5879,  5881,  5897,  5903,  5923,  5927,  5939
,  5953,  5981,  5987,  6007,  6011,  6029,  6037,  6043,  6047,  6053
,  6067,  6073,  6079,  6089,  6091,  6101,  6113,  6121,  6131,  6133
,  6143,  6151,  6163,  6173,  6197,  6199,  6203,  6211,  6217,  6221
,  6229,  6247,  6257,  6263,  6269,  6271,  6277,  6287,  6299,  6301
,  6311,  6317,  6323,  6329,  6337,  6343,  6353,  6359,  6361,  6367
,  6373,  6379,  6389,  6397,  6421,  6427,  6449,  6451,  6469,  6473
,  6481,  6491,  6521,  6529,  6547,  6551,  6553,  6563,  6569,  6571
,  6577,  6581,  6599,  6607,  6619,  6637,  6653,  6659,  6661,  6673
,  6679,  6689,  6691,  6701,  6703,  6709,  6719,  6733,  6737,  6761
,  6763,  6779,  6781,  6791,  6793,  6803,  6823,  6827,  6829,  6833
,  6841,  6857,  6863,  6869,  6871,  6883,  6899,  6907,  6911,  6917
,  6947,  6949,  6959,  6961,  6967,  6971,  6977,  6983,  6991,  6997
,  7001,  7013,  7019,  7027,  7039,  7043,  7057,  7069,  7079,  7103
,  7109,  7121,  7127,  7129,  7151,  7159,  7177,  7187,  7193,  7207
,  7211,  7213,  7219,  7229,  7237,  7243,  7247,  7253,  7283,  7297
,  7307,  7309,  7321,  7331,  7333,  7349,  7351,  7369,  7393,  7411
,  7417,  7433,  7451,  7457,  7459,  7477,  7481,  7487,  7489,  7499
,  7507,  7517,  7523,  7529,  7537,  7541,  7547,  7549,  7559,  7561
,  7573,  7577,  7583,  7589,  7591,  7603,  7607,  7621,  7639,  7643
,  7649,  7669,  7673,  7681,  7687,  7691,  7699,  7703,  7717,  7723
,  7727,  7741,  7753,  7757,  7759,  7789,  7793,  7817,  7823,  7829
,  7841,  7853,  7867,  7873,  7877,  7879,  7883,  7901,  7907,  7919
};


number_theory_domain::number_theory_domain()
{

}

number_theory_domain::~number_theory_domain()
{

}

long int number_theory_domain::mod(long int a, long int p)
{
	int f_negative = FALSE;
	long int r;

	if (a < 0) {
		a = -1 * a; // 5/6/2020: here was an error: it was a = -1;
		f_negative = TRUE;
	}
	r = a % p;
	if (f_negative && r) {
		r = p - r;
	}
	return r;
}

long int number_theory_domain::int_negate(long int a, long int p)
{
	long int b;

	b = a % p;
	if (b == 0) {
		return 0;
	}
	else {
		return p - b;
	}
}


long int number_theory_domain::power_mod(long int a, long int n, long int p)
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object A, N, M;
	
	A.create(a, __FILE__, __LINE__);
	N.create(n, __FILE__, __LINE__);
	M.create(p, __FILE__, __LINE__);
	D.power_longint_mod(A, N, M, 0 /* verbose_level */);
	return A.as_lint();
}

long int number_theory_domain::inverse_mod(long int a, long int p)
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object A, B, U, V, G;
	long int u;
	
	A.create(a, __FILE__, __LINE__);
	B.create(p, __FILE__, __LINE__);
	D.extended_gcd(A,B, G, U, V, 0 /* verbose_level */);
	if (!G.is_one() && !G.is_mone()) {
		cout << "number_theory_domain::inverse_mod a and p are not coprime" << endl;
		cout << "a=" << a << endl;
		cout << "p=" << p << endl;
		cout << "gcd=" << G << endl;
		exit(1);
	}
	u = U.as_lint();
	while (u < 0) {
		u += p;
		}
	return u;
}

long int number_theory_domain::mult_mod(long int a, long int b, long int p)
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object A, B, C, P;
	
	A.create(a, __FILE__, __LINE__);
	B.create(b, __FILE__, __LINE__);
	P.create(p, __FILE__, __LINE__);
	D.mult_mod(A, B, C, P, 0 /* verbose_level */);
	return C.as_int();
}

long int number_theory_domain::add_mod(long int a, long int b, long int p)
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object A, B, C, P, Q;
	long int r;
	
	A.create(a, __FILE__, __LINE__);
	B.create(b, __FILE__, __LINE__);
	P.create(p, __FILE__, __LINE__);
	D.add(A, B, C);
	D.integral_division_by_lint(C, p, Q, r);
	return r;
}

long int number_theory_domain::ab_over_c(long int a, long int b, long int c)
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object A, B, C, AB, Q;
	long int r;

	A.create(a, __FILE__, __LINE__);
	B.create(b, __FILE__, __LINE__);
	D.mult(A, B, AB);
	D.integral_division_by_lint(AB, c, Q, r);
	return Q.as_lint();
}

long int number_theory_domain::int_abs(long int a)
{
	if (a < 0) {
		return -a;
		}
	else {
		return a;
		}
}

long int number_theory_domain::gcd_lint(long int m, long int n)
{
#if 0
	longinteger_domain D;
	longinteger_object M, N, G, U, V;


	M.create(m);
	N.create(n);
	D.extended_gcd(M, N, G, U, V, 0);
	return G.as_int();
#else
	long int r, s;
	
	if (m < 0) {
		m *= -1;
	}
	if (n < 0) {
		n *= -1;
	}
	if (n > m) {
		r = m;
		m = n;
		n = r;
	}
	if (n == 0) {
		return m;
	}
	while (TRUE) {
		s = m / n;
		r = m - (s * n);
		if (r == 0) {
			return n;
		}
		m = n;
		n = r;
	}
#endif
}

void number_theory_domain::extended_gcd_int(int m, int n, int &g, int &u, int &v)
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object M, N, G, U, V;


	M.create(m, __FILE__, __LINE__);
	N.create(n, __FILE__, __LINE__);
	D.extended_gcd(M, N, G, U, V, 0);
	g = G.as_int();
	u = U.as_int();
	v = V.as_int();
}

void number_theory_domain::extended_gcd_lint(long int m, long int n,
		long int &g, long int &u, long int &v)
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object M, N, G, U, V;


	M.create(m, __FILE__, __LINE__);
	N.create(n, __FILE__, __LINE__);
	D.extended_gcd(M, N, G, U, V, 0);
	g = G.as_lint();
	u = U.as_lint();
	v = V.as_lint();
}

long int number_theory_domain::gcd_with_key_in_latex(std::ostream &ost,
		long int a, long int b, int f_key, int verbose_level)
//Computes gcd(a,b)
{
	int f_v = (verbose_level >= 1);
	long int a1, b1, q1, r1;

	if (f_v) {
		cout << "number_theory_domain::gcd_with_key_in_latex "
				"a=" << a << ", b=" << b << ":" << endl;
	}
	if (a > b) {
		a1 = a;
		b1 = b;
	}
	else {
		a1 = b;
		b1 = a;
	}

	while (TRUE) {


		r1 = a1 % b1;
		q1 = (a1 - r1) / b1;
		if (f_key) {
			ost << "=";
			ost << " \\gcd\\big( " << b1 << ", " << r1 << "\\big) "
					"\\qquad \\mbox{b/c} \\; " << a1 << " = " << q1
					<< " \\cdot " << b1 << " + " << r1 << "\\\\" << endl;
		}
		if (f_v) {
			cout << "number_theory_domain::gcd_with_key_in_latex "
					"a1=" << a1 << " b1=" << b1
					<< " r1=" << r1 << " q1=" << q1
					<< endl;
			}
		if (r1 == 0) {
			break;
		}
		a1 = b1;
		b1 = r1;
	}
	if (f_key) {
		ost << "= " << b1 << "\\\\" << endl;
	}
	if (f_v) {
		cout << "number_theory_domain::gcd_with_key_in_latex done" << endl;
	}
	return b1;
}

int number_theory_domain::i_power_j_safe(int i, int j)
{
	ring_theory::longinteger_domain D;

	ring_theory::longinteger_object a, b, c;
	int res;

	a.create(i, __FILE__, __LINE__);
	D.power_int(a, j);
	res = a.as_int();
	b.create(res, __FILE__, __LINE__);
	b.negate();
	D.add(a, b, c);
	if (!c.is_zero()) {
		cout << "i_power_j_safe int overflow when computing "
				<< i << "^" << j << endl;
		cout << "should be        " << a << endl;
		cout << "but comes out as " << res << endl;
		exit(1);
	}
	return res;
}

long int number_theory_domain::i_power_j_lint_safe(int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;

	ring_theory::longinteger_object a, b, c;
	long int res;

	if (f_v) {
		cout << "number_theory_domain::i_power_j_lint_safe "
				"i=" << i << " j=" << j << endl;
	}
	a.create(i, __FILE__, __LINE__);
	D.power_int(a, j);
	if (f_v) {
		cout << "number_theory_domain::i_power_j_lint_safe "
				"a=" << a << endl;
	}
	res = a.as_lint();
	if (f_v) {
		cout << "number_theory_domain::i_power_j_lint_safe "
				"as_lint=" << res << endl;
	}
	b.create(res, __FILE__, __LINE__);
	if (f_v) {
		cout << "number_theory_domain::i_power_j_lint_safe "
				"b=" << b << endl;
	}
	b.negate();
	D.add(a, b, c);
	if (f_v) {
		cout << "number_theory_domain::i_power_j_lint_safe "
				"c=" << c << endl;
	}
	if (!c.is_zero()) {
		cout << "i_power_j_safe int overflow when computing "
				<< i << "^" << j << endl;
		cout << "should be        " << a << endl;
		cout << "but comes out as " << res << endl;
		exit(1);
	}
	return res;
}

long int number_theory_domain::i_power_j_lint(long int i, long int j)
//Computes $i^j$ as integer.
//There is no checking for overflow.
{
	long int k, r = 1;

	//cout << "i_power_j i=" << i << ", j=" << j << endl;
	for (k = 0; k < j; k++) {
		r *= i;
		}
	//cout << "i_power_j yields" << r << endl;
	return r;
}

int number_theory_domain::i_power_j(int i, int j)
//Computes $i^j$ as integer.
//There is no checking for overflow.
{
	int k, r = 1;

	//cout << "i_power_j i=" << i << ", j=" << j << endl;
	for (k = 0; k < j; k++) {
		r *= i;
		}
	//cout << "i_power_j yields" << r << endl;
	return r;
}


void number_theory_domain::do_eulerfunction_interval(
		long int n_min, long int n_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int n, i;
	long int *T;
	int t0, t1, dt;
	orbiter_kernel_system::os_interface Os;
	orbiter_kernel_system::file_io Fio;
	char str[1000];

	t0 = Os.os_ticks();
	if (f_v) {
		cout << "number_theory_domain::do_eulerfunction_interval "
				"n_min=" << n_min << " n_max=" << n_max << endl;
	}

	std::vector<std::vector<long int>> Table;

	for (n = n_min; n <= n_max; n++) {


		//std::pair<long int, long int> P;
		std::vector<long int> data;

		int nb_prime_factors;
		int nb_dpf;
		long int a;

		a = euler_function(n);
		if (f_v) {
			cout << "number_theory_domain::do_eulerfunction_interval "
					"Euler function of " << n << " is " << a << endl;
		}

		nb_prime_factors = nb_prime_factors_counting_multiplicities(n);
		if (f_v) {
			cout << "number_theory_domain::do_eulerfunction_interval "
					"number of prime factors of " << n << " is " << nb_prime_factors << endl;
		}

		nb_dpf = nb_distinct_prime_factors(n);
		if (f_v) {
			cout << "number_theory_domain::do_eulerfunction_interval "
					"number of distinct prime factors of " << n << " is " << nb_dpf << endl;
		}

		data.push_back(n);
		data.push_back(a);
		data.push_back(nb_prime_factors);
		data.push_back(nb_dpf);
		//P.first = n;
		//P.second = a;

		Table.push_back(data);

	}
	T = NEW_lint(4 * Table.size());
	for (i = 0; i < Table.size(); i++) {
		T[i * 4 + 0] = Table[i][0];
		T[i * 4 + 1] = Table[i][1];
		T[i * 4 + 2] = Table[i][2];
		T[i * 4 + 3] = Table[i][3];
	}
	snprintf(str, sizeof(str), "table_eulerfunction_%ld_%ld.csv", n_min, n_max);
	string fname;

	fname.assign(str);

	string *Headers;

	Headers = new string[4];
	Headers[0].assign("N");
	Headers[1].assign("PHI");
	Headers[2].assign("NBPF");
	Headers[3].assign("NBDPF");

	Fio.lint_matrix_write_csv_override_headers(fname, Headers, T, Table.size(), 4);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	delete [] Headers;

	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;
	if (f_v) {
		cout << "number_theory_domain::do_eulerfunction_interval done" << endl;
	}
}




long int number_theory_domain::euler_function(long int n)
//Computes Euler's $\varphi$-function for $n$.
//Uses the prime factorization of $n$. before: eulerfunc
{
	//int *primes;
	//int *exponents;
	vector<long int> primes;
	vector<int> exponents;
	//int len;
	long int i, k, p1, e1;

	//len = factor_int(n, primes, exponents);
	factor_lint(n, primes, exponents);

	k = 1;
	for (i = 0; i < primes.size(); i++) {
		p1 = primes[i];
		e1 = exponents[i];
		if (e1 > 1) {
			k *= i_power_j_lint(p1, e1 - 1);
		}
		k *= (p1 - 1);
	}
	//FREE_int(primes);
	//FREE_int(exponents);
	return k;
}

long int number_theory_domain::moebius_function(long int n)
//Computes the Moebius $\mu$-function for $n$.
//Uses the prime factorization of $n$.
{
	int *primes;
	int *exponents;
	int len;
	long int i;

	len = factor_int(n, primes, exponents);

	for (i = 0; i < len; i++) {
		if (exponents[i] > 1) {
			return 0;
		}
	}
	FREE_int(primes);
	FREE_int(exponents);
	if (EVEN(len)) {
		return 1;
	}
	else {
		return -1;
	}
}



long int number_theory_domain::order_mod_p(long int a, long int p)
//Computes the order of $a$ mod $p$, i.~e. the smallest $k$ 
//s.~th. $a^k \equiv 1$ mod $p$.
{
	long int o, b;
	
	if (a < 0) {
		cout << "number_theory_domain::order_mod_p a < 0" << endl;
		exit(1);
	}
	a %= p;
	if (a == 0) {
		return 0;
	}
	if (a == 1) {
		return 1;
	}
	o = 1;
	b = a;
	while (b != 1) {
		b *= a;
		b %= p;
		o++;
	}
	return o;
}

int number_theory_domain::int_log2(int n)
// returns $\log_2(n)$ 
{	int i;
	
	if (n <= 0) {
		cout << "int_log2 n <= 0" << endl;
		exit(1);
		}
	for (i = 0; n > 0; i++) {
		n >>= 1;
		}
	return i;
}

int number_theory_domain::int_log10(int n)
// returns $\log_{10}(n)$ 
{
	int j;
	
	if (n <= 0) {
		cout << "int_log10 n <= 0" << endl;
		cout << "n = " << n << endl;
		exit(1);
		}
	j = 0;
	while (n) {
		n /= 10;
		j++;
		}
	return j;
}

int number_theory_domain::lint_log10(long int n)
// returns $\log_{10}(n)$
{
	long int j;

	if (n <= 0) {
		cout << "lint_log10 n <= 0" << endl;
		cout << "n = " << n << endl;
		exit(1);
		}
	j = 0;
	while (n) {
		n /= 10;
		j++;
		}
	return j;
}

int number_theory_domain::int_logq(int n, int q)
// returns the number of digits in base q representation
{	int i;
	
	if (n < 0) {
		cout << "int_logq n < 0" << endl;
		exit(1);
		}
	i = 0;
	do {
		i++;
		n /= q;
		} while (n);
	return i;
}

int number_theory_domain::lint_logq(long int n, int q)
// returns the number of digits in base q representation
{
	int i;

	if (n < 0) {
		cout << "int_logq n < 0" << endl;
		exit(1);
		}
	i = 0;
	do {
		i++;
		n /= q;
		} while (n);
	return i;
}

int number_theory_domain::is_strict_prime_power(int q)
// assuming that q is a prime power, this function tests
// if q is a strict prime power
{
	int p;
	
	p = smallest_primedivisor(q);
	if (q != p)
		return TRUE;
	else 
		return FALSE;
}

int number_theory_domain::is_prime(int p)
{
	int p1;
	
	p1 = smallest_primedivisor(p);
	if (p1 != p)
		return FALSE;
	else 
		return TRUE;
}

int number_theory_domain::is_prime_power(int q)
{
	int p, h;

	return is_prime_power(q, p, h);
}

int number_theory_domain::is_prime_power(int q, int &p, int &h)
{
	int i;
	
	p = smallest_primedivisor(q);
	//cout << "smallest prime in " << q << " is " << p << endl;
	q = q / p;
	h = 1;
	while (q > 1) {
		i = q % p;
		//cout << "q=" << q << " i=" << i << endl;
		if (i) {
			return FALSE;
			}
		q = q / p;
		h++;
		}
	return TRUE;
}

int number_theory_domain::smallest_primedivisor(int n)
//Computes the smallest prime dividing $n$. 
//The algorithm is based on Lueneburg~\cite{Lueneburg87a}.
{
	int flag, i, q;
	
	if (EVEN(n)) {
		return(2);
	}
	if ((n % 3) == 0) {
		return(3);
	}
	i = 5;
	flag = 0;
	while (TRUE) {
		q = n / i;
		if (n == q * i) {
			return(i);
		}
		if (q < i) {
			return(n);
		}
		if (flag) {
			i += 4;
		}
		else {
			i += 2;
		}
		flag = !flag;
	}
}

int number_theory_domain::sp_ge(int n, int p_min)
// Computes the smallest prime dividing $n$
// which is greater than or equal to p\_min. 
{
	int i, q;
	
	if (p_min == 0)
		p_min = 2;
	if (p_min < 0)
		p_min = - p_min;
	if (p_min <= 2) {
		if (EVEN(n))
			return 2;
		p_min = 3;
		}
	if (p_min <= 3) {
		if ((n % 3) == 0)
			return 3;
		p_min = 5;
		}
	if (EVEN(p_min))
		p_min--;
	i = p_min;
	while (TRUE) {
		q = n / i;
		// cout << "n=" << n << " i=" << i << " q=" << q << endl;
		if (n == q * i)
			return(i);
		if (q < i)
			return(n);
		i += 2;
		}
#if 0
	int flag;
	
	if (EVEN((p_min - 1) >> 1))
		/* p_min cong 1 mod 4 ? */
		flag = FALSE;
	else
		flag = TRUE;
	while (TRUE) {
		q = n / i;
		cout << "n=" << n << " i=" << i << " q=" << q << endl;
		if (n == q * i)
			return(i);
		if (q < i)
			return(n);
		if (flag) {
			i += 4;
			flag = FALSE;
			}
		else {
			i += 2;
			flag = TRUE;
			}
		}
#endif
}

int number_theory_domain::factor_int(int a, int *&primes, int *&exponents)
{
	int alloc_len = 10, len = 0;
	int p, i;
	
	primes = NEW_int(alloc_len);
	exponents = NEW_int(alloc_len);
	
	if (a == 1) {
		cout << "factor_int, the number is one" << endl;
		return 0;
		}
	if (a <= 0) {
		cout << "factor_int, the number is <= 0" << endl;
		exit(1);
		}
	while (a > 1) {
		p = smallest_primedivisor(a);
		a /= p;
		if (len == 0) {
			primes[0] = p;
			exponents[0] = 1;
			len = 1;
			}
		else {
			if (p == primes[len - 1]) {
				exponents[len - 1]++;
				}
			else {
				if (len == alloc_len) {
					int *primes2, *exponents2;
					
					alloc_len += 10;
					primes2 = NEW_int(alloc_len);
					exponents2 = NEW_int(alloc_len);
					for (i = 0; i < len; i++) {
						primes2[i] = primes[i];
						exponents2[i] = exponents[i];
						}
					FREE_int(primes);
					FREE_int(exponents);
					primes = primes2;
					exponents = exponents2;
					}
				primes[len] = p;
				exponents[len] = 1;
				len++;
				}
			}
		}
	return len;
}

int number_theory_domain::nb_prime_factors_counting_multiplicities(long int a)
{
	vector<long int> primes;
	vector<int> exponents;
	int cnt = 0;
	int i;

	factor_lint(a, primes, exponents);
	for (i = 0; i < primes.size(); i++) {
		cnt += exponents[i];
	}
	return cnt;
}

int number_theory_domain::nb_distinct_prime_factors(long int a)
{
	vector<long int> primes;
	vector<int> exponents;

	factor_lint(a, primes, exponents);
	return primes.size();
}




void number_theory_domain::factor_lint(
		long int a,
		std::vector<long int> &primes,
		std::vector<int> &exponents)
{
	int p, p0;

#if 0
	if (a == 1) {
		cout << "number_theory_domain::factor_lint, the number is one" << endl;
		exit(1);
		}
#endif
	if (a <= 0) {
		cout << "number_theory_domain::factor_lint, the number is <= 0" << endl;
		exit(1);
		}
	p0 = -1;
	while (a > 1) {
		p = smallest_primedivisor(a);
		a /= p;
		if (p == p0) {
			exponents[exponents.size() - 1]++;
		}
		else {
			primes.push_back(p);
			exponents.push_back(1);
			p0 = p;
		}
	}
}

void number_theory_domain::factor_prime_power(int q, int &p, int &e)
{
	if (q == 1) {
		cout << "factor_prime_power q is one" << endl;
		exit(1);
		}
	p = smallest_primedivisor(q);
	q /= p;
	e = 1;
	while (q != 1) {
		if ((q % p) != 0) {
			cout << "factor_prime_power q is not a prime power" << endl;
			exit(1);
			}
		q /= p;
		e++;
		}
}

long int number_theory_domain::primitive_root_randomized(long int p, int verbose_level)
// Computes a primitive element for $\bbZ_p$, i.~e. an integer $k$
// with $2 \le k \le p - 1$ s. th. the order of $k$ mod $p$ is $p-1$.
{
	int f_v = (verbose_level >= 1);
	long int i, pm1, a, n, b;
	vector<long int> primes;
	vector<int> exponents;
	orbiter_kernel_system::os_interface Os;
	int cnt = 0;

	if (f_v) {
		cout << "number_theory_domain::primitive_root_randomized p=" << p << endl;
	}
	if (p < 2) {
		cout << "number_theory_domain::primitive_root_randomized: p < 2" << endl;
		exit(1);
	}

	pm1 = p - 1;
	if (f_v) {
		cout << "number_theory_domain::primitive_root_randomized before factor_lint " << pm1 << endl;
	}
	factor_lint(pm1, primes, exponents);
	if (f_v) {
		cout << "number_theory_domain::primitive_root_randomized after factor_lint " << pm1 << endl;
		cout << "number_theory_domain::primitive_root_randomized number of factors is " << primes.size() << endl;
	}
	while (TRUE) {
		cnt++;
		a = Os.random_integer(pm1);
		if (a == 0) {
			continue;
		}
		if (f_v) {
			cout << "number_theory_domain::primitive_root_randomized iteration " << cnt << " : trying " << a << endl;
		}
		for (i = 0; i < (long int) primes.size(); i++) {
			n = pm1 / primes[i];
			if (f_v) {
				cout << "number_theory_domain::primitive_root_randomized iteration " << cnt << " : trying " << a
						<< " : prime factor " << i << " / " << primes.size() << " raising to the power " << n << endl;
			}
			b = power_mod(a, n, p);
			if (f_v) {
				cout << "number_theory_domain::primitive_root_randomized iteration " << cnt
						<< " : trying " << a << " : prime factor " << i << " / " << primes.size()
						<< " raising to the power " << n << " yields " << b << endl;
			}
			if (b == 1) {
				// fail
				break;
			}
		}
		if (i == (long int)primes.size()) {
			break;
		}
	}

	if (f_v) {
		cout << "number_theory_domain::primitive_root_randomized done after " << cnt << " trials" << endl;
	}
	return a;
}

long int number_theory_domain::primitive_root(long int p, int verbose_level)
// Computes a primitive element for $\bbZ_p$, i.~e. an integer $k$ 
// with $2 \le k \le p - 1$ s.~th. the order of $k$ mod $p$ is $p-1$.
{
	int f_v = (verbose_level >= 1);
	long int i, o;

	if (p < 2) {
		cout << "primitive_root: p < 2" << endl;
		exit(1);
		}
	if (p == 2) {
		return 1;
	}
	for (i = 2; i < p; i++) {
		o = order_mod_p(i, p);
		if (o == p - 1) {
			if (f_v) {
				cout << i << " is primitive root mod " << p << endl;
			}
			return i;
		}
	}
	cout << "no primitive root found" << endl;
	exit(1);
}

int number_theory_domain::Legendre(long int a, long int p, int verbose_level)
// Computes the Legendre symbol $\left( \frac{a}{p} \right)$.
{
	return Jacobi(a, p, verbose_level);
}

int number_theory_domain::Jacobi(long int a, long int m, int verbose_level)
//Computes the Jacobi symbol $\left( \frac{a}{m} \right)$.
{
	int f_v = (verbose_level >= 1);
	long int a1, m1, ord2, r1;
	long int g;
	int f_negative = FALSE;
	long int t, t1, t2;
	
	if (f_v) {
		cout << "Jacobi(" << a << ", " << m << ")" << endl;
		}
	a1 = a;
	m1 = m;
	r1 = 1;
	g = gcd_lint(a1, m1);
	if (ABS(g) != 1) {
		return 0;
		}
	while (TRUE) {
		/* Invariante: 
		 * r1 enthaelt bereits ausgerechnete Faktoren.
		 * ABS(r1) == 1.
		 * Jacobi(a, m) = r1 * Jacobi(a1, m1) und ggT(a1, m1) == 1. */
		if (a1 == 0) {
			cout << "Jacobi a1 == 0" << endl;
			exit(1);
			}
		a1 = a1 % m1;
		if (f_v) {
			cout << "Jacobi = " << r1
					<< " * Jacobi(" << a1 << ", " << m1 << ")" << endl;
			}
#if 0
		a1 = NormRemainder(a1, m1);
		if (a1 < 0)
			f_negative = TRUE;
		else
			f_negative = FALSE;
#endif
		ord2 = ny2(a1, a1);
		
		/* a1 jetzt immer noch != 0 */
		if (f_negative) {
			t = (m1 - 1) >> 1; /* t := (m1 - 1) / 2 */
			/* Ranmultiplizieren von (-1) hoch t an r1: */
			if (t % 2)
				r1 = -r1; /* Beachte ABS(r1) == 1 */
			/* und a1 wieder positiv machen: */
			a1 = -a1;
			}
		if (ord2 % 2) {
			/* tue nur dann etwas, wenn ord2 ungerade */
			// t = (m1 * m1 - 1) >> 3; /* t = (m1 * m1 - 1) / 8 */
			/* Ranmultiplizieren von (-1) hoch t an r1: */
			if (m1 % 8 == 3 || m1 % 8 == 5)
				r1 = -r1; /* Beachte ABS(r1) == 1L */
			}
		if (ABS(a1) <= 1)
			break;
		/* Reziprozitaet: */
		t1 = (m1 - 1) >> 1; /* t1 = (m1 - 1) / 2 */
		t2 = (a1 - 1) >> 1; /* t1 = (a1 - 1) / 2 */
		if ((t1 % 2) && (t2 % 2)) /* t1 und t2 ungerade */
			r1 = -r1; /* Beachte ABS(r1) == 1 */
		t = m1;
		m1 = a1;
		a1 = t;
		if (f_v) {
			cout << "Jacobi = " << r1
					<< " * Jacobi(" << a1 << ", " << m1 << ")" << endl;
			}
		}
	if (a1 == 1) {
		return r1;
		}
	if (a1 <= 0) {
		cout << "Jacobi a1 == -1 || a1 == 0" << endl;
		exit(1);
		}
	cout << "Jacobi wrong termination" << endl;
	exit(1);
}

int number_theory_domain::Jacobi_with_key_in_latex(std::ostream &ost,
		long int a, long int m, int verbose_level)
//Computes the Jacobi symbol $\left( \frac{a}{m} \right)$.
{
	int f_v = (verbose_level >= 1);
	long int a1, m1, ord2, r1;
	long int g;
	int f_negative = FALSE;
	long int t, t1, t2;
	
	if (f_v) {
		cout << "number_theory_domain::Jacobi_with_key_in_latex(" << a << ", " << m << ")" << endl;
	}


	ost << "$\\Big( \\frac{" << a << " }{ "
			<< m << "}\\Big)$\\\\" << endl;


	a1 = a;
	m1 = m;
	r1 = 1;
	g = gcd_lint(a1, m1);
	if (ABS(g) != 1) {
		return 0;
	}
	while (TRUE) {
		// Invariant:
		// r1 contains partial results.
		// ABS(r1) == 1.
		// Jacobi(a, m) = r1 * Jacobi(a1, m1) and gcd(a1, m1) == 1.
		if (a1 == 0) {
			cout << "number_theory_domain::Jacobi_with_key_in_latex a1 == 0" << endl;
			exit(1);
		}
		if (a1 % m1 < a1) {

			ost << "$=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
			}
			ost << " \\Big( \\frac{" << a1 % m1 << " }{ "
					<< m1 << "}\\Big)$\\\\" << endl;

		}

		a1 = a1 % m1;

		if (f_v) {
			cout << "Jacobi = " << r1 << " * Jacobi("
					<< a1 << ", " << m1 << ")" << endl;
		}
		ord2 = ny2(a1, a1);
		
		// a1 != 0
		if (f_negative) {

			ost << "$=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
			}
			ost << "\\Big( \\frac{-1 }{ " << m1
					<< "}\\Big) \\cdot \\Big( \\frac{"
					<< a1 * i_power_j(2, ord2) << " }{ "
					<< m1 << "}\\Big)$\\\\" << endl;
			ost << "$=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
			}
			ost << "(-1)^{\\frac{" << m1
					<< "-1}{2}} \\cdot \\Big( \\frac{"
					<< a1 * i_power_j(2, ord2) << " }{ "
					<< m1 << "}\\Big)$\\\\" << endl;



			t = (m1 - 1) >> 1;
				// t := (m1 - 1) / 2

			/* Ranmultiplizieren von (-1) hoch t an r1: */

			if (t % 2) {
				r1 = -r1; /* note ABS(r1) == 1 */
			}

			a1 = -a1;

			ost << "$=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
			}
			ost << " \\Big( \\frac{"
					<< a1 * i_power_j(2, ord2)
			<< " }{ " << m1 << "}\\Big)$\\\\" << endl;


			}
		if (ord2 % 2) {
			/* tue nur dann etwas, wenn ord2 ungerade */
			// t = (m1 * m1 - 1) >> 3; /* t = (m1 * m1 - 1) / 8 */
			/* Ranmultiplizieren von (-1) hoch t an r1: */

			if (ord2 > 1) {
				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "\\Big( \\frac{2}{ " << m1
						<< "}\\Big)^{" << ord2
						<< "} \\cdot \\Big( \\frac{" << a1
						<< " }{ " << m1 << "}\\Big)$\\\\" << endl;
				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "\\Big( (-1)^{\\frac{" << m1
						<< "^2-1}{8}} \\Big)^{" << ord2
						<< "} \\cdot \\Big( \\frac{" << a1 << " }{ "
						<< m1 << "}\\Big)$\\\\" << endl;
			}
			else {
				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "\\Big( \\frac{2}{ " << m1
						<< "}\\Big) \\cdot \\Big( \\frac{" << a1
						<< " }{ " << m1 << "}\\Big)$\\\\" << endl;
				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "(-1)^{\\frac{" << m1
						<< "^2-1}{8}} \\cdot \\Big( \\frac{" << a1
						<< " }{ " << m1 << "}\\Big)$\\\\" << endl;
			}

			if (m1 % 8 == 3 || m1 % 8 == 5) {
				r1 = -r1; /* Beachte ABS(r1) == 1L */
			}

			ost << "$=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
			}
			ost << "\\Big( \\frac{" << a1 << " }{ " << m1
					<< "}\\Big)$\\\\" << endl;


		}
		else {
			if (ord2) {
				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "\\Big( \\frac{2}{ " << m1 << "}\\Big)^{"
						<< ord2 << "} \\cdot \\Big( \\frac{" << a1
						<< " }{ " << m1 << "}\\Big)$\\\\" << endl;

				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "\\Big( (-1)^{\\frac{" << m1
						<< "^2-1}{8}} \\Big)^{" << ord2
						<< "} \\cdot \\Big( \\frac{" << a1 << " }{ "
						<< m1 << "}\\Big)$\\\\" << endl;
				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "\\Big( \\frac{" << a1 << " }{ " << m1
						<< "}\\Big)$\\\\" << endl;
			}
		}
		if (ABS(a1) <= 1) {
			break;
		}


		t = m1;
		m1 = a1;
		a1 = t;


		ost << "$=";
		if (r1 == -1) {
			ost << "(-1) \\cdot ";
		}
		ost << "\\Big( \\frac{" << a1 << " }{ " << m1
				<< "}\\Big) \\cdot (-1)^{\\frac{" << m1
				<< "-1}{2} \\cdot \\frac{" << a1
				<< " - 1}{2}}$\\\\" << endl;


		// reciprocity:
		t1 = (m1 - 1) >> 1;
			// t1 = (m1 - 1) / 2
		t2 = (a1 - 1) >> 1;
			// t1 = (a1 - 1) / 2
		if ((t1 % 2) && (t2 % 2)) {
			// t1 and t2 are both odd
			r1 = -r1;
			// note ABS(r1) == 1
		}

		ost << "$=";
		if (r1 == -1) {
			ost << "(-1) \\cdot ";
		}
		ost << "\\Big( \\frac{" << a1 << " }{ " << m1
				<< "}\\Big)$\\\\" << endl;

		if (f_v) {
			cout << "number_theory_domain::Jacobi_with_key_in_latex = " << r1 << " * Jacobi(" << a1
					<< ", " << m1 << ")" << endl;
		}
	}
	if (a1 == 1) {
		ost << "$=" << r1 << "$\\\\" << endl;
		return r1;
	}
	if (a1 <= 0) {
		cout << "number_theory_domain::Jacobi_with_key_in_latex a1 == -1 || a1 == 0" << endl;
		exit(1);
	}
	cout << "number_theory_domain::Jacobi_with_key_in_latex wrong termination" << endl;
	exit(1);
}

int number_theory_domain::Legendre_with_key_in_latex(std::ostream &ost,
		long int a, long int m, int verbose_level)
//Computes the Legendre symbol $\left( \frac{a}{m} \right)$.
{
	int f_v = (verbose_level >= 1);
	long int a1, m1, ord2, r1;
	long int g;
	int f_negative = FALSE;
	long int t, t1, t2;

	if (f_v) {
		cout << "number_theory_domain::Legendre_with_key_in_latex(" << a << ", " << m << ")" << endl;
	}


	ost << "$\\Big( \\frac{" << a << " }{ "
			<< m << "}\\Big)$\\\\" << endl;


	a1 = a;
	m1 = m;
	r1 = 1;
	g = gcd_lint(a1, m1);
	if (ABS(g) != 1) {
		return 0;
	}
	while (TRUE) {
		// invariant:
		// r1 contains partial result.
		// ABS(r1) == 1.
		// Legendre(a, m) = r1 * Legendre(a1, m1) and gcd(a1, m1) == 1. */
		if (a1 == 0) {
			cout << "number_theory_domain::Legendre_with_key_in_latex a1 == 0" << endl;
			exit(1);
		}
		if (a1 % m1 < a1) {

			ost << "$=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
			}
			ost << " \\Big( \\frac{" << a1 % m1 << " }{ "
					<< m1 << "}\\Big)$\\\\" << endl;

		}

		a1 = a1 % m1;

		if (f_v) {
			cout << "Jacobi = " << r1 << " * Jacobi("
					<< a1 << ", " << m1 << ")" << endl;
		}
		ord2 = ny2(a1, a1);

		// a1 is != 0
		if (f_negative) {

			ost << "$=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
			}
			ost << "\\Big( \\frac{-1 }{ " << m1
					<< "}\\Big) \\cdot \\Big( \\frac{"
					<< a1 * i_power_j(2, ord2) << " }{ "
					<< m1 << "}\\Big)$\\\\" << endl;
			ost << "$=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
			}
			ost << "(-1)^{\\frac{" << m1
					<< "-1}{2}} \\cdot \\Big( \\frac{"
					<< a1 * i_power_j(2, ord2) << " }{ "
					<< m1 << "}\\Big)$\\\\" << endl;



			t = (m1 - 1) >> 1; /* t := (m1 - 1) / 2 */

			if (t % 2) {
				r1 = -r1;
			}

			a1 = -a1;

			ost << "$=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
			}
			ost << " \\Big( \\frac{"
					<< a1 * i_power_j(2, ord2)
			<< " }{ " << m1 << "}\\Big)$\\\\" << endl;


			}
		if (ord2 % 2) {
			/* tue nur dann etwas, wenn ord2 ungerade */
			// t = (m1 * m1 - 1) >> 3; /* t = (m1 * m1 - 1) / 8 */
			/* Ranmultiplizieren von (-1) hoch t an r1: */

			if (ord2 > 1) {
				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "\\Big( \\frac{2}{ " << m1
						<< "}\\Big)^{" << ord2
						<< "} \\cdot \\Big( \\frac{" << a1
						<< " }{ " << m1 << "}\\Big)$\\\\" << endl;
				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "\\Big( (-1)^{\\frac{" << m1
						<< "^2-1}{8}} \\Big)^{" << ord2
						<< "} \\cdot \\Big( \\frac{" << a1 << " }{ "
						<< m1 << "}\\Big)$\\\\" << endl;
			}
			else {
				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "\\Big( \\frac{2}{ " << m1
						<< "}\\Big) \\cdot \\Big( \\frac{" << a1
						<< " }{ " << m1 << "}\\Big)$\\\\" << endl;
				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "(-1)^{\\frac{" << m1
						<< "^2-1}{8}} \\cdot \\Big( \\frac{" << a1
						<< " }{ " << m1 << "}\\Big)$\\\\" << endl;
			}

			if (m1 % 8 == 3 || m1 % 8 == 5) {
				r1 = -r1; /* Beachte ABS(r1) == 1L */
			}

			ost << "$=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
			}
			ost << "\\Big( \\frac{" << a1 << " }{ " << m1
					<< "}\\Big)$\\\\" << endl;


		}
		else {
			if (ord2) {
				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "\\Big( \\frac{2}{ " << m1 << "}\\Big)^{"
						<< ord2 << "} \\cdot \\Big( \\frac{" << a1
						<< " }{ " << m1 << "}\\Big)$\\\\" << endl;

				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "\\Big( (-1)^{\\frac{" << m1
						<< "^2-1}{8}} \\Big)^{" << ord2
						<< "} \\cdot \\Big( \\frac{" << a1 << " }{ "
						<< m1 << "}\\Big)$\\\\" << endl;
				ost << "$=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
				}
				ost << "\\Big( \\frac{" << a1 << " }{ " << m1
						<< "}\\Big)$\\\\" << endl;
			}
		}
		if (ABS(a1) <= 1) {
			break;
		}


		t = m1;
		m1 = a1;
		a1 = t;


		ost << "$=";
		if (r1 == -1) {
			ost << "(-1) \\cdot ";
		}
		ost << "\\Big( \\frac{" << a1 << " }{ " << m1
				<< "}\\Big) \\cdot (-1)^{\\frac{" << m1
				<< "-1}{2} \\cdot \\frac{" << a1
				<< " - 1}{2}}$\\\\" << endl;


		// reciprocity:

		t1 = (m1 - 1) >> 1;
			// t1 = (m1 - 1) / 2
		t2 = (a1 - 1) >> 1;
			// t1 = (a1 - 1) / 2

		if ((t1 % 2) && (t2 % 2)) /* t1 and t2 are both odd */ {
			r1 = -r1; /* note: ABS(r1) == 1 */
		}

		ost << "$=";
		if (r1 == -1) {
			ost << "(-1) \\cdot ";
		}
		ost << "\\Big( \\frac{" << a1 << " }{ " << m1
				<< "}\\Big)$\\\\" << endl;

		if (f_v) {
			cout << "number_theory_domain::Jacobi = " << r1 << " * Jacobi(" << a1
					<< ", " << m1 << ")" << endl;
		}
	}
	if (a1 == 1) {
		ost << "$=" << r1 << "$\\\\" << endl;
		return r1;
	}
	if (a1 <= 0) {
		cout << "number_theory_domain::Legendre_with_key_in_latex a1 == -1 || a1 == 0" << endl;
		exit(1);
	}
	cout << "number_theory_domain::Legendre_with_key_in_latex wrong termination" << endl;
	exit(1);
}

int number_theory_domain::ny2(long int x, long int &x1)
//returns $n = \ny_2(x).$ 
//Computes $x1 := \frac{x}{2^n}$. 
{
	int xx = x;
	int n1;
	int f_negative;
	
	n1 = 0;
	if (xx == 0) {
		cout << "number_theory_domain::ny2 x == 0" << endl;
		exit(1);
	}
	if (xx < 0) {
		xx = -xx;
		f_negative = TRUE;
	}
	else {
		f_negative = FALSE;
	}
	while (TRUE) {
		// while xx congruent 0 mod 2:
		if (ODD(xx)) {
			break;
		}
		n1++;
		xx >>= 1;
	}
	if (f_negative) {
		xx = -xx;
	}
	x1 = xx;
	return n1;
}

int number_theory_domain::ny_p(long int n, long int p)
//Returns $\nu_p(n),$ the integer $k$ with $n=p^k n'$ and $p \nmid n'$.
{
	int ny_p;
	
	if (n == 0) {
		cout << "number_theory_domain::ny_p n == 0" << endl;
		exit(1);
	}
	if (n < 0) {
		n = -n;
	}
	ny_p = 0;
	while (n != 1) {
		if ((n % p) != 0) {
			break;
		}
		n /= p;
		ny_p++;
	}
	return ny_p;
}

#if 0
// now use longinteger_domain::square_root_mod(int a, int p, int verbose_level)
long int number_theory_domain::sqrt_mod_simple(long int a, long int p)
// solves x^2 = a mod p. Returns x
{
	long int a1, x;
	
	a1 = a % p;
	for (x = 0; x < p; x++) {
		if ((x * x) % p == a1) {
			return x;
		}
	}
	cout << "number_theory_domain::sqrt_mod_simple the number a is "
			"not a quadratic residue mod p" << endl;
	cout << "a = " << a << " p=" << p << endl;
	exit(1);
}
#endif
void number_theory_domain::print_factorization(
		int nb_primes, int *primes, int *exponents)
{
	int i;
	
	for (i = 0; i < nb_primes; i++) {
		cout << primes[i];
		if (exponents[i] > 1) {
			cout << "^" << exponents[i];
		}
		if (i < nb_primes - 1) {
			cout << " * ";
		}
	}
}

void number_theory_domain::print_longfactorization(
		int nb_primes,
		ring_theory::longinteger_object *primes,
		int *exponents)
{
	int i;
	
	for (i = 0; i < nb_primes; i++) {
		cout << primes[i];
		if (exponents[i] > 1) {
			cout << "^" << exponents[i];
		}
		if (i < nb_primes - 1) {
			cout << " * ";
		}
	}
}


void number_theory_domain::int_add_fractions(int at, int ab,
		int bt, int bb, int &ct, int &cb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int g, a1, b1;
	
	if (at == 0) {
		ct = bt;
		cb = bb;
	}
	else if (bt == 0) {
		ct = at;
		cb = ab;
	}
	else {
		g = gcd_lint(ab, bb);
		a1 = ab / g;
		b1 = bb / g;
		cb = ab * b1;
		ct = at * b1 + bt * a1;
	}
	if (cb < 0) {
		cb *= -1;
		ct *= -1;
	}
	g = gcd_lint(int_abs(ct), cb);
	if (g > 1) {
		ct /= g;
		cb /= g;
	}
	if (f_v) {
		cout << "int_add_fractions " << at <<  "/"
				<< ab << " + " << bt << "/" << bb << " = "
				<< ct << "/" << cb << endl;
	}
}

void number_theory_domain::int_mult_fractions(int at, int ab,
		int bt, int bb, int &ct, int &cb,
		int verbose_level)
{
	long int g;
	
	if (at == 0) {
		ct = 0;
		cb = 1;
	}
	else if (bt == 0) {
		ct = 0;
		cb = 1;
	}
	else {
		g = gcd_lint(at, ab);
		if (g != 1 && g != -1) {
			at /= g;
			ab /= g;
		}
		g = gcd_lint(bt, bb);
		if (g != 1 && g != -1) {
			bt /= g;
			bb /= g;
		}
		g = gcd_lint(at, bb);
		if (g != 1 && g != -1) {
			at /= g;
			bb /= g;
		}
		g = gcd_lint(bt, ab);
		if (g != 1 && g != -1) {
			bt /= g;
			ab /= g;
		}
		ct = at * bt;
		cb = ab * bb;
	}
	if (cb < 0) {
		cb *= -1;
		ct *= -1;
	}
	g = gcd_lint(int_abs(ct), cb);
	if (g > 1) {
		ct /= g;
		cb /= g;
	}
}


int number_theory_domain::choose_a_prime_greater_than(int lower_bound)
{
	int p, r;
	int nb_primes = sizeof(the_first_thousand_primes) / sizeof(int);
	orbiter_kernel_system::os_interface Os;

	while (TRUE) {
		r = Os.random_integer(nb_primes);
		p = the_first_thousand_primes[r];
		if (p > lower_bound) {
			return p;
		}
	}
}

int number_theory_domain::choose_a_prime_in_interval(int lower_bound, int upper_bound)
{
	int p, r;
	int nb_primes = sizeof(the_first_thousand_primes) / sizeof(int);
	orbiter_kernel_system::os_interface Os;

	while (TRUE) {
		r = Os.random_integer(nb_primes);
		p = the_first_thousand_primes[r];
		if (p > lower_bound && p < upper_bound) {
			return p;
		}
	}
}

int number_theory_domain::random_integer_greater_than(int n, int lower_bound)
{
	int r;
	orbiter_kernel_system::os_interface Os;

	while (TRUE) {
		r = Os.random_integer(n);
		if (r > lower_bound) {
			return r;
		}
	}
}

int number_theory_domain::random_integer_in_interval(int lower_bound, int upper_bound)
{
	orbiter_kernel_system::os_interface Os;
	int r, n;

	if (upper_bound <= lower_bound) {
		cout << "random_integer_in_interval upper_bound <= lower_bound" << endl;
		exit(1);
	}
	n = upper_bound - lower_bound;
	r = lower_bound + Os.random_integer(n);
	return r;
}

int number_theory_domain::nb_primes_available()
{
	return sizeof(the_first_thousand_primes) / sizeof(int);
}

int number_theory_domain::get_prime_from_table(int idx)
{
	return the_first_thousand_primes[idx];
}

long int number_theory_domain::Chinese_Remainders(
		std::vector<long int> &Remainders,
		std::vector<long int> &Moduli,
		long int &M, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "number_theory_domain::Chinese_Remainders" << endl;
	}

	long int k, mr1, m1v, x;
	long int r1, r2;
	long int m1, m2;
	int i;

	r1 = Remainders[0];
	m1 = Moduli[0];
	x = r1;

	for (i = 1; i < Remainders.size(); i++) {

		r2 = Remainders[i];
		m2 = Moduli[i];

		mr1 = int_negate(r1, m2);

		m1v = inverse_mod(m1, m2);

		k = mult_mod(m1v, add_mod(r2, mr1, m2), m2);
		x = r1 + k * m1;

		r1 = x;
		m1 *= m2;

	}

	M = m1;

	if (f_v) {
		cout << "number_theory_domain::Chinese_Remainders" << endl;
	}
	return x;
}


long int number_theory_domain::ChineseRemainder2(long int a1, long int a2,
		long int p1, long int p2, int verbose_level)
{
	long int k, ma1, p1v, x;

	ma1 = int_negate(a1, p2);
	p1v = inverse_mod(p1, p2);
	k = mult_mod(p1v, add_mod(a2, ma1, p2), p2);
	x = a1 + k * p1;
	return x;
}


void number_theory_domain::sieve(std::vector<int> &primes,
		int factorbase, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, from, to, l, unit_size = 1000;

	if (f_v) {
		cout << "number_theory_domain::sieve" << endl;
	}
	//primes.m_l(0);
	for (i = 0; ; i++) {
		from = i * unit_size + 1;
		to = from + unit_size - 1;
		sieve_primes(primes, from, to, factorbase, FALSE);
		l = primes.size();
		cout << "[" << from << "," << to
			<< "], total number of primes = "
			<< l << endl;
		if (l >= factorbase) {
			break;
		}
	}

	if (f_v) {
		cout << "number_theory_domain::sieve done" << endl;
	}
}

void number_theory_domain::sieve_primes(std::vector<int> &v,
		int from, int to, int limit, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int x, y, l, k, p, f_prime;

	if (f_v) {
		cout << "number_theory_domain::sieve_primes" << endl;
	}
	l = v.size();
	if (ODD(from)) {
		x = from;
	}
	else {
		x = from + 1;
	}
	for (; x <= to; x++, x++) {
		if (x <= 1) {
			continue;
		}
		f_prime = TRUE;
		for (k = 0; k < l; k++) {
			p = v[k];
			y = x / p;
			// cout << "x=" << x << " p=" << p << " y=" << y << endl;
			if ((x - y * p) == 0) {
				f_prime = FALSE;
				break;
			}
			if (y < p) {
				break;
			}
#if 0
			if ((x % p) == 0)
				break;
#endif
			}
		if (!f_prime) {
			continue;
		}
		if (nb_primes(x) != 1) {
			cout << "error: " << x << " is not prime!" << endl;
		}
		v.push_back(x);
		if (f_v) {
			cout << v.size() << " " << x << endl;
		}
		l++;
		if (l >= limit) {
			return;
		}
	}
	if (f_v) {
		cout << "number_theory_domain::sieve_primes done" << endl;
	}
}

int number_theory_domain::nb_primes(int n)
//Returns the number of primes in the prime factorization
//of $n$ (including multiplicities).
{
	int i = 0;
	int d;

	if (n < 0) {
		n = -n;
	}
	while (n != 1) {
		d = smallest_primedivisor(n);
		i++;
		n /= d;
		}
	return i;
}

void number_theory_domain::cyclotomic_set(
		std::vector<int> &cyclotomic_set,
		int a, int q, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int b, c;

	if (f_v) {
		cout << "number_theory_domain::cyclotomic_set" << endl;
	}
	b = a;
	cyclotomic_set.push_back(b);
	if (f_v) {
		cout << "push " << b << endl;
	}
	while (TRUE) {
		c = (b * q) % n;
		if (c == a) {
			break;
		}
		b = c;
		cyclotomic_set.push_back(b);
		if (f_v) {
			cout << "push " << b << endl;
		}
	}
	if (f_v) {
		cout << "number_theory_domain::cyclotomic_set done" << endl;
	}
}


void number_theory_domain::elliptic_curve_addition(
		field_theory::finite_field *F,
		int b, int c,
	int x1, int x2, int x3,
	int y1, int y2, int y3,
	int &z1, int &z2, int &z3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, two, three, top, bottom, m;

	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_addition" << endl;
	}

	//my_nb_calls_to_elliptic_curve_addition++;
	if (x3 == 0) {
		z1 = y1;
		z2 = y2;
		z3 = y3;
		goto done;
	}
	if (y3 == 0) {
		z1 = x1;
		z2 = x2;
		z3 = x3;
		goto done;
	}
	if (x3 != 1) {
		a = F->inverse(x3);
		x1 = F->mult(x1, a);
		x2 = F->mult(x2, a);
	}
	if (y3 != 1) {
		a = F->inverse(y3);
		y1 = F->mult(y1, a);
		y2 = F->mult(y2, a);
	}
	if (x1 == y1 && x2 != y2) {
		if (F->negate(x2) != y2) {
			cout << "x1 == y1 && x2 != y2 && negate(x2) != y2" << endl;
			exit(1);
		}
		z1 = 0;
		z2 = 1;
		z3 = 0;
		goto done;
	}
	if (x1 == y1 && x2 == 0 && y2 == 0) {
		z1 = 0;
		z2 = 1;
		z3 = 0;
		goto done;
	}
	if (x1 == y1 && x2 == y2) {
		two = F->add(1, 1);
		three = F->add(two, 1);
		top = F->add(F->mult(three, F->mult(x1, x1)), b);
		bottom = F->mult(two, x2);
		a = F->inverse(bottom);
		m = F->mult(top, a);
	}
	else {
		top = F->add(y2, F->negate(x2));
		bottom = F->add(y1, F->negate(x1));
		a = F->inverse(bottom);
		m = F->mult(top, a);
	}
	z1 = F->add(F->add(F->mult(m, m), F->negate(x1)), F->negate(y1));
	z2 = F->add(F->mult(m, F->add(x1, F->negate(z1))), F->negate(x2));
	z3 = 1;
done:
	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_addition done" << endl;
	}
}

void number_theory_domain::elliptic_curve_point_multiple(
		field_theory::finite_field *F,
		int b, int c, int n,
	int x1, int y1, int z1,
	int &x3, int &y3, int &z3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int bx, by, bz;
	int cx, cy, cz;
	int tx, ty, tz;

	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_point_multiple" << endl;
	}
	bx = x1;
	by = y1;
	bz = z1;
	cx = 0;
	cy = 1;
	cz = 0;
	while (n) {
		if (n % 2) {
			//cout << "finite_field::power: mult(" << b << "," << c << ")=";

			elliptic_curve_addition(F,
					b, c,
				bx, by, bz,
				cx, cy, cz,
				tx, ty, tz, verbose_level - 1);
			cx = tx;
			cy = ty;
			cz = tz;
			//c = mult(b, c);
			//cout << c << endl;
		}
		elliptic_curve_addition(F,
				b, c,
			bx, by, bz,
			bx, by, bz,
			tx, ty, tz, verbose_level - 1);
		bx = tx;
		by = ty;
		bz = tz;
		//b = mult(b, b);
		n >>= 1;
		//cout << "finite_field::power: " << b << "^" << n << " * " << c << endl;
	}
	x3 = cx;
	y3 = cy;
	z3 = cz;
	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_point_multiple done" << endl;
	}
}

void number_theory_domain::elliptic_curve_point_multiple_with_log(
		field_theory::finite_field *F,
		int b, int c, int n,
	int x1, int y1, int z1,
	int &x3, int &y3, int &z3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int bx, by, bz;
	int cx, cy, cz;
	int tx, ty, tz;

	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_point_multiple_with_log" << endl;
	}
	bx = x1;
	by = y1;
	bz = z1;
	cx = 0;
	cy = 1;
	cz = 0;
	cout << "ECMultiple$\\Big((" << bx << "," << by << "," << bz << "),";
	cout << "(" << cx << "," << cy << "," << cz << "),"
			<< n << "," << b << "," << c << "," << F->p << "\\Big)$\\\\" << endl;

	while (n) {
		if (n % 2) {
			//cout << "finite_field::power: mult(" << b << "," << c << ")=";

			elliptic_curve_addition(F,
					b, c,
				bx, by, bz,
				cx, cy, cz,
				tx, ty, tz, verbose_level - 1);
			cx = tx;
			cy = ty;
			cz = tz;
			//c = mult(b, c);
			//cout << c << endl;
		}
		elliptic_curve_addition(F,
				b, c,
			bx, by, bz,
			bx, by, bz,
			tx, ty, tz, verbose_level - 1);
		bx = tx;
		by = ty;
		bz = tz;
		//b = mult(b, b);
		n >>= 1;
		cout << "=ECMultiple$\\Big((" << bx << "," << by << "," << bz << "),";
		cout << "(" << cx << "," << cy << "," << cz << "),"
				<< n << "," << b << "," << c << "," << F->p << "\\Big)$\\\\" << endl;
		//cout << "finite_field::power: " << b << "^" << n << " * " << c << endl;
	}
	x3 = cx;
	y3 = cy;
	z3 = cz;
	cout << "$=(" << x3 << "," << y3 << "," << z3 << ")$\\\\" << endl;
	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_point_multiple_with_log done" << endl;
	}
}

int number_theory_domain::elliptic_curve_evaluate_RHS(
		field_theory::finite_field *F,
		int x, int b, int c)
{
	int x2, x3, e;

	x2 = F->mult(x, x);
	x3 = F->mult(x2, x);
	e = F->add(x3, F->mult(b, x));
	e = F->add(e, c);
	return e;
}

void number_theory_domain::elliptic_curve_points(
		field_theory::finite_field *F,
		int b, int c, int &nb, int *&T,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//finite_field F;
	int x, y, n;
	int r, l;
	number_theory_domain NT;
	ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_points" << endl;
	}
	nb = 0;
	//F.init(p, verbose_level);
	for (x = 0; x < F->p; x++) {
		r = elliptic_curve_evaluate_RHS(F, x, b, c);
		if (r == 0) {
			if (f_v) {
				cout << nb << " : (" << x << "," << 0 << ",1)" << endl;
			}
			nb++;
		}
		else {
			if (F->p != 2) {
				if (F->e > 1) {
					cout << "number_theory_domain::elliptic_curve_points odd characteristic and e > 1" << endl;
					exit(1);
				}
				l = Legendre(r, F->p, verbose_level - 1);
				if (l == 1) {
					//y = sqrt_mod_involved(r, p);
					y = D.square_root_mod(r, F->p, 0 /* verbose_level*/);
					//y = NT.sqrt_mod_simple(r, p);
					if (f_v) {
						cout << nb << " : (" << x << "," << y << ",1)" << endl;
						cout << nb + 1 << " : (" << x << "," << F->negate(y) << ",1)" << endl;
					}
					nb += 2;
				}
			}
			else {
				y = F->frobenius_power(r, F->e - 1);
				if (f_v) {
					cout << nb << " : (" << x << "," << y << ",1)" << endl;
				}
				nb += 1;
			}
		}
	}
	if (f_v) {
		cout << nb << " : (0,1,0)" << endl;
	}
	nb++;
	if (f_v) {
		cout << "the curve has " << nb << " points" << endl;
	}
	T = NEW_int(nb * 3);
	n = 0;
	for (x = 0; x < F->p; x++) {
		r = elliptic_curve_evaluate_RHS(F, x, b, c);
		if (r == 0) {
			T[n * 3 + 0] = x;
			T[n * 3 + 1] = 0;
			T[n * 3 + 2] = 1;
			n++;
			//cout << nb++ << " : (" << x << "," << 0 << ",1)" << endl;
		}
		else {
			if (F->p != 2) {
				// odd characteristic:
				l = Legendre(r, F->p, verbose_level - 1);
				if (l == 1) {
					//y = sqrt_mod_involved(r, p);
					//y = NT.sqrt_mod_simple(r, p);
					y = D.square_root_mod(r, F->p, 0 /* verbose_level*/);
					T[n * 3 + 0] = x;
					T[n * 3 + 1] = y;
					T[n * 3 + 2] = 1;
					n++;
					T[n * 3 + 0] = x;
					T[n * 3 + 1] = F->negate(y);
					T[n * 3 + 2] = 1;
					n++;
					//cout << nb++ << " : (" << x << "," << y << ",1)" << endl;
					//cout << nb++ << " : (" << x << "," << F.negate(y) << ",1)" << endl;
				}
			}
			else {
				// even characteristic
				y = F->frobenius_power(r, F->e - 1);
				T[n * 3 + 0] = x;
				T[n * 3 + 1] = y;
				T[n * 3 + 2] = 1;
				n++;
				//cout << nb++ << " : (" << x << "," << y << ",1)" << endl;
			}
		}
	}
	T[n * 3 + 0] = 0;
	T[n * 3 + 1] = 1;
	T[n * 3 + 2] = 0;
	n++;
	//print_integer_matrix_width(cout, T, nb, 3, 3, log10_of_q);
	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_points done" << endl;
		cout << "the curve has " << nb << " points" << endl;
	}
}

void number_theory_domain::elliptic_curve_all_point_multiples(
		field_theory::finite_field *F,
		int b, int c, int &order,
	int x1, int y1, int z1,
	std::vector<std::vector<int> > &Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int x2, y2, z2;
	int x3, y3, z3;

	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_all_point_multiples" << endl;
	}
	order = 1;

	x2 = x1;
	y2 = y1;
	z2 = z1;
	while (TRUE) {
		{
			vector<int> pts;

			pts.push_back(x2);
			pts.push_back(y2);
			pts.push_back(z2);

			Pts.push_back(pts);
		}
		if (z2 == 0) {
			return;
		}

		elliptic_curve_addition(F,
				b, c,
			x1, y1, z1,
			x2, y2, z2,
			x3, y3, z3, 0 /*verbose_level */);

		x2 = x3;
		y2 = y3;
		z2 = z3;

		order++;
	}
	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_all_point_multiples done" << endl;
	}
}

int number_theory_domain::elliptic_curve_discrete_log(
		field_theory::finite_field *F,
		int b, int c,
	int x1, int y1, int z1,
	int x3, int y3, int z3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int x2, y2, z2;
	int a3, b3, c3;
	int n;

	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_discrete_log" << endl;
	}
	n = 1;

	x2 = x1;
	y2 = y1;
	z2 = z1;
	while (TRUE) {
		if (x2 == x3 && y2 == y3 && z2 == z3) {
			break;
		}

		elliptic_curve_addition(F, b, c,
			x1, y1, z1,
			x2, y2, z2,
			a3, b3, c3, 0 /*verbose_level */);

		n++;

		x2 = a3;
		y2 = b3;
		z2 = c3;

	}
	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_discrete_log done" << endl;
	}
	return n;
}

int number_theory_domain::eulers_totient_function(
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_primes, *primes, *exponents;
	int i, p, e;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object N, R, A, B, C;

	if (f_v) {
		cout << "number_theory_domain::eulers_totient_function" << endl;
	}
	N.create(n, __FILE__, __LINE__);
	D.factor(N, nb_primes, primes, exponents, verbose_level);
	R.create(1, __FILE__, __LINE__);
	for (i = 0; i < nb_primes; i++) {
		p = primes[i];
		e = exponents[i];
		A.create(p, __FILE__, __LINE__);
		D.power_int(A, e);
		if (f_v) {
			cout << "p^e=" << A << endl;
		}
		B.create(p, __FILE__, __LINE__);
		D.power_int(B, e - 1);
		if (f_v) {
			cout << "p^{e-1}=" << A << endl;
		}
		B.negate();
		D.add(A, B, C);
		if (f_v) {
			cout << "p^e-p^{e-1}=" << C << endl;
		}
		D.mult(R, C, A);
		A.assign_to(R);
	}
	if (f_v) {
		cout << "number_theory_domain::eulers_totient_function done" << endl;
	}
	return R.as_int();
}

void number_theory_domain::do_jacobi(
		long int jacobi_top,
		long int jacobi_bottom, int verbose_level)
{
	string fname;
	string author;
	string title;
	string extra_praeamble;


	char str[1000];

	snprintf(str, 1000, "jacobi_%ld_%ld.tex", jacobi_top, jacobi_bottom);
	fname.assign(str);
	snprintf(str, 1000, "Jacobi %ld over %ld", jacobi_top, jacobi_bottom);
	title.assign(str);

	{
	ofstream f(fname);


	orbiter_kernel_system::latex_interface L;


	L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			extra_praeamble /* extra_praeamble */);


	number_theory::number_theory_domain NT;
	ring_theory::longinteger_domain D;

	ring_theory::longinteger_object A, B;

	A.create(jacobi_top, __FILE__, __LINE__);

	B.create(jacobi_bottom, __FILE__, __LINE__);

	D.jacobi(A, B, verbose_level);

	NT.Jacobi_with_key_in_latex(f,
			jacobi_top, jacobi_bottom, verbose_level);
	//Computes the Jacobi symbol $\left( \frac{a}{m} \right)$.

	L.foot(f);
	}

	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


}

void number_theory_domain::elliptic_curve_addition_table(
		geometry::projective_space *P2,
	int *A6, int *Pts, int nb_pts, int *&Table,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k;
	int pi, pj, pk;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_addition_table" << endl;
	}
	Table = NEW_int(nb_pts * nb_pts);
	for (i = 0; i < nb_pts; i++) {
		pi = Pts[i];
		for (j = 0; j < nb_pts; j++) {
			pj = Pts[j];
			pk = elliptic_curve_addition(P2, A6, pi, pj,
					0 /* verbose_level */);
			if (!Sorting.int_vec_search(Pts, nb_pts, pk, k)) {
				cout << "number_theory_domain::elliptic_curve_addition_table cannot find point pk" << endl;
				cout << "i=" << i << " pi=" << pi << " j=" << j
						<< " pj=" << pj << " pk=" << pk << endl;
				cout << "Pts: ";
				Int_vec_print(cout, Pts, nb_pts);
				cout << endl;
				exit(1);
			}
			Table[i * nb_pts + j] = k;
		}
	}
	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_addition_table done" << endl;
	}
}

int number_theory_domain::elliptic_curve_addition(
		geometry::projective_space *P2,
	int *A6, int p1_rk, int p2_rk,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int p1[3];
	int p2[3];
	int p3[3];
	int x1, y1, z1;
	int x2, y2, z2;
	int x3, y3, z3;
	int a1, a2, a3, a4, a6;
	int p3_rk;

	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_addition" << endl;
	}

	a1 = A6[0];
	a2 = A6[1];
	a3 = A6[2];
	a4 = A6[3];
	a6 = A6[5];

	P2->unrank_point(p1, p1_rk);
	P2->unrank_point(p2, p2_rk);
	P2->F->Projective_space_basic->PG_element_normalize(
			p1, 1, 3);
	P2->F->Projective_space_basic->PG_element_normalize(
			p2, 1, 3);

	x1 = p1[0];
	y1 = p1[1];
	z1 = p1[2];
	x2 = p2[0];
	y2 = p2[1];
	z2 = p2[2];
	if (f_vv) {
		cout << "number_theory_domain::elliptic_curve_addition "
				"x1=" << x1 << " y1=" << y1 << " z1=" << z1 << endl;
		cout << "number_theory_domain::elliptic_curve_addition "
				"x2=" << x2 << " y2=" << y2 << " z2=" << z2 << endl;
	}
	if (z1 == 0) {
		if (p1_rk != 1) {
			cout << "number_theory_domain::elliptic_curve_addition "
					"z1 == 0 && p1_rk != 1" << endl;
			exit(1);
		}
		x3 = x2;
		y3 = y2;
		z3 = z2;
#if 0
		if (z2 == 0) {
			if (p2_rk != 1) {
				cout << "number_theory_domain::elliptic_curve_addition "
						"z2 == 0 && p2_rk != 1" << endl;
				exit(1);
			}
			x3 = 0;
			y3 = 1;
			z3 = 0;
		}
		else {
			x3 = x2;
			y3 = F->negate(F->add3(y2, F->mult(a1, x2), a3));
			z3 = 1;
		}
#endif

	}
	else if (z2 == 0) {
		if (p2_rk != 1) {
			cout << "number_theory_domain::elliptic_curve_addition "
					"z2 == 0 && p2_rk != 1" << endl;
			exit(1);
		}
		x3 = x1;
		y3 = y1;
		z3 = z1;

#if 0
		// at this point, we know that z1 is not zero.
		x3 = x1;
		y3 = F->negate(F->add3(y1, F->mult(a1, x1), a3));
		z3 = 1;
#endif

	}
	else {
		// now both points are affine.


		int lambda_top, lambda_bottom, lambda, nu_top, nu_bottom, nu;
		int three, two; //, m_one;
		int c;

		c = P2->F->add4(y1, y2, P2->F->mult(a1, x2), a3);

		if (x1 == x2 && c == 0) {
			x3 = 0;
			y3 = 1;
			z3 = 0;
		}
		else {

			two = P2->F->add(1, 1);
			three = P2->F->add(two, 1);
			//m_one = F->negate(1);



			if (x1 == x2) {

				// point duplication:
				lambda_top = P2->F->add4(P2->F->mult3(three, x1, x1),
						P2->F->mult3(two, a2, x1), a4,
						P2->F->negate(P2->F->mult(a1, y1)));
				lambda_bottom = P2->F->add3(P2->F->mult(two, y1),
						P2->F->mult(a1, x1), a3);

				nu_top = P2->F->add4(P2->F->negate(P2->F->mult3(x1, x1, x1)),
						P2->F->mult(a4, x1), P2->F->mult(two, a6),
						P2->F->negate(P2->F->mult(a3, y1)));
				nu_bottom = P2->F->add3(P2->F->mult(two, y1),
						P2->F->mult(a1, x1), a3);

			}
			else {
				// adding different points:
				lambda_top = P2->F->add(y2, P2->F->negate(y1));
				lambda_bottom = P2->F->add(x2, P2->F->negate(x1));

				nu_top = P2->F->add(P2->F->mult(y1, x2), P2->F->negate(P2->F->mult(y2, x1)));
				nu_bottom = lambda_bottom;
			}


			if (lambda_bottom == 0) {
				cout << "number_theory_domain::elliptic_curve_addition "
						"lambda_bottom == 0" << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"x1=" << x1 << " y1=" << y1 << " z1=" << z1 << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"x2=" << x2 << " y2=" << y2 << " z2=" << z2 << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"a1=" << a1 << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"a2=" << a2 << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"a3=" << a3 << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"a4=" << a4 << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"a6=" << a6 << endl;
				exit(1);
			}
			lambda = P2->F->mult(lambda_top, P2->F->inverse(lambda_bottom));

			if (nu_bottom == 0) {
				cout << "number_theory_domain::elliptic_curve_addition "
						"nu_bottom == 0" << endl;
				exit(1);
			}
			nu = P2->F->mult(nu_top, P2->F->inverse(nu_bottom));

			if (f_vv) {
				cout << "number_theory_domain::elliptic_curve_addition "
						"a1=" << a1 << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"a2=" << a2 << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"a3=" << a3 << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"a4=" << a4 << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"a6=" << a6 << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"three=" << three << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"lambda_top=" << lambda_top << endl;
				cout << "number_theory_domain::elliptic_curve_addition "
						"lambda=" << lambda << " nu=" << nu << endl;
			}
			x3 = P2->F->add3(P2->F->mult(lambda, lambda), P2->F->mult(a1, lambda),
					P2->F->negate(P2->F->add3(a2, x1, x2)));
			y3 = P2->F->negate(P2->F->add3(P2->F->mult(P2->F->add(lambda, a1), x3), nu, a3));
			z3 = 1;
		}
	}
	p3[0] = x3;
	p3[1] = y3;
	p3[2] = z3;
	if (f_vv) {
		cout << "number_theory_domain::elliptic_curve_addition "
				"x3=" << x3 << " y3=" << y3 << " z3=" << z3 << endl;
	}
	p3_rk = P2->rank_point(p3);
	if (f_v) {
		cout << "number_theory_domain::elliptic_curve_addition "
				"done" << endl;
	}
	return p3_rk;
}



}}}


