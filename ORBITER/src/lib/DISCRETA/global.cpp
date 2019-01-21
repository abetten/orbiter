// global.C
//
// Anton Betten
// 10.11.1999
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"

//#include <stdio.h>
//#include <stdlib.h> // for rand(), RAND_MAX

namespace orbiter {

#undef DEBUG_CALLOC_NOBJECTS_PLUS_LENGTH
#undef TONELLI_VERBOSE
#undef DEBUG_INVERT_MOD_intEGER



/**** global variables ***/


// printing_mode gl_printing_mode = printing_mode_ascii;
// int printing_mode = PRintING_MODE_ASCII;

#define MAX_PRintING_MODE_STACK 100

int printing_mode_stack_size = 0;
static enum printing_mode_enum printing_mode_stack[MAX_PRintING_MODE_STACK];


const char *discreta_home = NULL;
const char *discreta_arch = NULL;


/**** global functions ***/

static void Binomial_using_table(int n, int k, matrix & T, discreta_base & res);

void discreta_init()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	char str[1000];
	
	discreta_home = getenv("DISCRETA_HOME");
	if (discreta_home == NULL) {
		if (f_v) {
			cout << "discreta_init(): WARNING: $DISCRETA_HOME not set !" << endl;
			}
		discreta_home = ".";
		}
	discreta_arch = getenv("DISCRETA_ARCH");
	if (discreta_arch == NULL) {
		if (f_v) {
			cout << "discreta_init(): WARNING: $DISCRETA_ARCH not set !" << endl;
			}
		discreta_arch = ".";
		}
	if (discreta_home) {
		
		strcpy(str, discreta_home);
		strcat(str, "/lib/this_is");
		
#if 1
		if (file_size(str) <= 0) {
			if (f_v) {
				cout << "discreta_init(): WARNING: can't find my library (DISCRETA_HOME/lib) !" << endl;
				}
			}
#endif
		}
	database_init(verbose_level);
}

discreta_base *callocobject(kind k)
{
	discreta_base *p = new( operator new(sizeof(discreta_base)) ) discreta_base;
	
	p->c_kind(k);
	return p;
}

void freeobject(discreta_base *p)
{
	operator delete(p); /* free(p); */
}

discreta_base *calloc_nobjects(int n, kind k)
{
	int i;
	discreta_base *p;
	
	p = (discreta_base *) operator new(n * sizeof(discreta_base));
	if (p == NULL) {
		cout << "calloc_nobjects() no memory" << endl;
		exit(1);
		}
	for (i = 0; i < n; i++) {
		new( &p[i] ) discreta_base;
		p[i].c_kind(k);
		}
	return p;
}

void free_nobjects(discreta_base *p, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		p[i].freeself();
		}
	operator delete(p);
}

discreta_base *calloc_nobjects_plus_length(int n, kind k)
{
	int i;
	discreta_base *p;
	
#ifdef DEBUG_CALLOC_NOBJECTS_PLUS_LENGTH
	cout << "calloc_nobjects_plus_length() n=" << n << endl;
#endif
	p = (discreta_base *) operator new((n + 1) * sizeof(discreta_base));
	if (p == NULL) {
		cout << "calloc_nobjects_plus_length() no memory" << endl;
		exit(1);
		}
	p++;
	for (i = 0; i < n; i++) {
		new( &p[i] ) discreta_base;
		p[i].c_kind(k);
		}
	p[-1].c_kind(INTEGER);
	p[-1].m_i_i(n);
	return p;
}

void free_nobjects_plus_length(discreta_base *p)
{
	int i, n;

	n = p[-1].s_i_i();
	if (n < 0) {
		cout << "free_nobjects_plus_length() length = " << n << " < 0\n";
		}
#ifdef DEBUG_CALLOC_NOBJECTS_PLUS_LENGTH
	cout << "free_nobjects_plus_length() n=" << n << endl;
#endif
	for (i = 0; i < n; i++) {
		p[i].freeself();
		}
	p[-1].freeself();
	p--;
	operator delete(p);
}

discreta_base *calloc_m_times_n_objects(int m, int n, kind k)
{
	int i;
	discreta_base *p;
	
	p = (discreta_base *) operator new((m * n + 2) * sizeof(discreta_base));
	if (p == NULL) {
		cout << "calloc_m_times_n_objects() no memory" << endl;
		exit(1);
		}
	p++;
	p++;
	for (i = 0; i < m * n; i++) {
		new( &p[i] ) discreta_base;
		p[i].c_kind(k);
		}
	p[-2].c_kind(INTEGER);
	p[-2].m_i_i(m);
	p[-1].c_kind(INTEGER);
	p[-1].m_i_i(n);
	return p;
}

void free_m_times_n_objects(discreta_base *p)
{
	int i, m, n;

	m = p[-2].s_i_i();
	n = p[-1].s_i_i();
	if (m < 0) {
		cout << "free_m_times_n_objects() m = " << m << " < 0\n";
		}
	if (n < 0) {
		cout << "free_m_times_n_objects() n = " << n << " < 0\n";
		}
	for (i = 0; i < m * n; i++) {
		p[i].freeself();
		}
	p[-2].freeself();
	p[-1].freeself();
	p--;
	p--;
	operator delete(p);
}

void printobjectkind(ostream& ost, kind k)
{
	ost << kind_ascii(k);
}

const char *kind_ascii(kind k)
{
	switch(k) {
		case BASE: return "BASE";
		case INTEGER: return "intEGER";
		case VECTOR: return "VECTOR";
		case NUMBER_PARTITION: return "NUMBER_PARTITION";
		case PERMUTATION: return "PERMUTATION";
		
		case MATRIX: return "MATRIX";

		case LONGINTEGER: return "LONGintEGER";
		//case SUBGROUP_LATTICE: return "SUBGROUP_LATTICE";
		//case SUBGROUP_ORBIT: return "SUBGROUP_ORBIT";

		case MEMORY: return "MEMORY";

		case HOLLERITH: return "HOLLERITH";
		case DATABASE: return "DATABASE";
		case BTREE: return "BTREE";

		case PERM_GROUP: return "PERM_GROUP";
		case BT_KEY: return "BT_KEY";

		case DESIGN_PARAMETER: return "DESIGN_PARAMETER";
		
		case GROUP_SELECTION: return "GROUP_SELECTION";
		case UNIPOLY: return "UNIPOLY";
		
		case DESIGN_PARAMETER_SOURCE: return "DESIGN_PARAMETER_SOURCE";
		case SOLID: return "SOLID";

		case BITMATRIX: return "BITMATRIX";
		//case PC_PRESENTATION: return "PC_PRESENTATION";
		//case PC_SUBGROUP: return "PC_SUBGROUP";
		//case GROUP_WORD: return "GROUP_WORD";
		//case GROUP_TABLE: return "GROUP_TABLE";
		//case ACTION: return "ACTION";
		case GEOMETRY: return "GEOMETRY";
		default: return "unknown kind";
		}
}

const char *action_kind_ascii(kind k)
{
	switch(k) {
		case vector_entries: return "vector_entries";
		case vector_positions: return "vector_positions";
		default: return "unknown action_kind";
		}
}

#if 0
void int_swap(int& x, int& y)
{
	int z;
	
	z = x;
	x = y;
	y = z;
}
#endif

void uint4_swap(uint_4& x, uint_4& y)
{
	uint_4 z;
	
	z = x;
	x = y;
	y = z;
}


ostream& operator<<(ostream& ost, discreta_base& p)
{
	// cout << "operator<< starting\n";
	p.print(ost);
	// cout << "operator<< finished\n";
	return ost;
};

#if 0
discreta_base operator * (discreta_base& x, discreta_base &y)
{
	discreta_base z;
	cout << "operator *: calling z.mult(x, y)\n";
	z.mult(x, y);
	cout << "operator *: z=" << z << endl;
	z.printobjectkind(cout) << endl;
	return z;
}

discreta_base operator + (discreta_base& x, discreta_base &y)
{
	discreta_base z;
	z.add(x, y);
	cout << "operator +: z=" << z << endl;
	z.printobjectkind(cout) << endl;
	return z;
}
#endif

int lcm_int(int m, int n)
{
	int g = gcd_int(m, n);
	int r = m / g;
	
	r *= n;
	return r;
}

#if 0
int gcd_int(int m, int n)
{
	integer M, N, U, V, G;
	
	M.m_i(m);
	N.m_i(n);
	M.extended_gcd(N, U, V, G);
	return G.s_i();
}
#endif

#if 0
void extended_gcd_int(int m, int n, int &u, int &v, int &g)
{
	integer M, N, U, V, G;
	
	M.m_i(m);
	N.m_i(n);
	M.extended_gcd(N, U, V, G, 0);
	u = U.s_i();
	v = V.s_i();
	g = G.s_i();
}
#endif

int invert_mod_integer(int i, int p)
{
	integer a, b;
	
#ifdef DEBUG_INVERT_MOD_intEGER
	cout << "invert_mod_integer i=" << i << ", p=" << p << endl;
#endif
	a.m_i(i);
	b.m_i(p);
	a.power_int_mod(p - 2, b);
#ifdef DEBUG_INVERT_MOD_intEGER
	cout << "i^-1=" << a.s_i() << endl;
#endif
	return a.s_i();

#if 0
	with ww(GFp, p);
	integer x;

#ifdef DEBUG_INVERT_MOD_intEGER
	cout << "invert_mod_integer i=" << i << ", p=" << p << endl;
#endif
	x.m_i(i);
	x.invert();
#ifdef DEBUG_INVERT_MOD_intEGER
	cout << "i^-1=" << x.s_i() << endl;
#endif
	return x.s_i();
#endif
}

#if 0
int i_power_j(int i, int j)
//Computes $i^j$ as integer.
//There is no checking for overflow.
{
	int k, r = 1;

	//cout << "i_power_j() i=" << i << ", j=" << j << endl;
	for (k = 0; k < j; k++)
		r *= i;
	//cout << "i_power_j() yields" << r << endl;
	return r;
}
#endif

int remainder_mod(int i, int n)
{
	if (i < 0) {
		i *= -1;
		i %= n;
		if (i == 0)
			return 0;
		else
			return n - i;
		}
	return i % n;
}

#if 0
int smallest_primedivisor(int n)
//Computes the smallest prime dividing $n$. 
//The algorithm is based on Lueneburg~\cite{Lueneburg87a}.
{
	int flag, i, q;
	
	if (EVEN(n))
		return(2);
	if ((n % 3) == 0)
		return(3);
	i = 5;
	flag = 0;
	while (TRUE) {
		q = n / i;
		if (n == q * i)
			return(i);
		if (q < i)
			return(n);
		if (flag)
			i += 4;
		else
			i += 2;
		flag = !flag;
		}
}

int sp_ge(int n, int p_min)
//Computes the smalles prime dividing $n$ 
//which is greater than or equal to p\_min. AB 230594.
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
#endif

void factor_integer(int n, Vector& primes, Vector& exponents)
//Factors the integer $n = \prod_{i=1}^r p_i^{e_i}$. 
//The vector primes holds the primes $p_i$,
//the vector exponents holds the $e_i$.
{
	int d, last_prime = 2, l;
	
	if (n == 0) {
		cout << "factor_integer(): n == 0\n";
		exit(1);
		}
	if (n == 1) {
		primes.m_l(0);
		exponents.m_l(0);
		return;
		}
	d = sp_ge(n, last_prime);
	primes.m_l(1);
	exponents.m_l(1);
	l = 1;
	primes.m_ii(0, d);
	exponents.m_ii(0, 1);
	last_prime = d;
	n /= d;
	while (n != 1) {
		d = sp_ge(n, last_prime);
		// cout << "n=" << n << " last_prime=" << last_prime << " next_prime=" << d << endl;
		if (d == last_prime) {
			exponents.m_ii(l - 1, exponents.s_ii(l - 1) + 1);
			}
		else {
			primes.inc();
			exponents.inc();
			primes.m_ii(l, d);
			exponents.m_ii(l, 1);
			l++;
			last_prime = d;
			}
		n /= d;
		}
}

void print_factorization(Vector& primes, Vector& exponents, ostream &o)
//Prints the factorization.
{
	int i, p, e, l;
	
	l = primes.s_l();
	if (l != exponents.s_l()) {
		cout << "print_factorization() l != exponents.s_l()\n";
		exit(1);
		}
	if (current_printing_mode() == printing_mode_latex) {
		for (i = 0; i < l; i++) {
			p = primes.s_ii(i);
			e = exponents.s_ii(i);
			if (e > 1) {
				o << p << "^";
				if (e >= 10)
					o << "{" << e << "}";
				else
					o << e;
				}
			else
				o << p;
			if (i < l - 1)
				o << "\\cdot ";
			}
		}
	else {
		for (i = 0; i < l; i++) {
			p = primes.s_ii(i);
			e = exponents.s_ii(i);
			if (e > 1)
				o << p << "^" << e;
			else
				o << p;
			if (i < l)
				o << " ";
			}
		}
}

void print_factorization_hollerith(Vector& primes, Vector& exponents, hollerith &h)
//Prints the factorization.
{
	int i, p, e, l;
	
	l = primes.s_l();
	if (l != exponents.s_l()) {
		cout << "print_factorization() l != exponents.s_l()\n";
		exit(1);
		}
	h.init("");
	if (current_printing_mode() == printing_mode_latex) {
		for (i = 0; i < l; i++) {
			p = primes.s_ii(i);
			e = exponents.s_ii(i);
			if (e > 1) {
				h.append_i(p);
				h.append("^");
				if (e >= 10) {
					h.append("{");
					h.append_i(e);
					h.append("}");
					}
				else {
					h.append_i(e);
					}
				}
			else
				h.append_i(p);
			if (i < l - 1)
				h.append("\\cdot ");
			}
		}
	else {
		for (i = 0; i < l; i++) {
			p = primes.s_ii(i);
			e = exponents.s_ii(i);
			if (e > 1) {
				h.append_i(p);
				h.append("^");
				h.append_i(e);
				}
			else
				h.append_i(p);
			if (i < l)
				h.append(" ");
			}
		}
}

int nb_primes(int n)
//Returns the number of primes in the prime factorization 
//of $n$ (including multiplicities).
{
	int i = 0;
	int d;
	
	if (n < 0)
		n = -n;
	while (n != 1) {
		d = smallest_primedivisor(n);
		i++;
		n /= d;
		}
	return i;
}

#if 0
int is_prime(int n)
//TRUE if and only if $n$ is prime.
{
	if (smallest_primedivisor(n) == n)
		return TRUE;
	else
		return FALSE;
}
#endif

#if 0
int is_power_of_prime(int n, int p)
//TRUE if and only if $n$ is a power of $p$.
{
	Vector vp, ve;
	
	factor_integer(n, vp, ve);
	return (vp.s_l() == 1);
}

int is_prime_power(int n, int &p, int &e)
//TRUE if and only if $n$ is a prime power. 
//If true, p and e are set to the prime and the exponent e such that n = p^e
{
	Vector vp, ve;
	
	factor_integer(n, vp, ve);
	if (vp.s_l() != 1)
		return FALSE;
	p = vp.s_ii(0);
	e = ve.s_ii(0);
	return TRUE;
}
#endif

int factor_if_prime_power(int n, int *p, int *e)
//Computes $p$ and $e$ with $n=p^e$. 
//If $n$ is not a prime power, FALSE is returned.
{
	Vector vp, ve;

	factor_integer(n, vp, ve);
	if (vp.s_l() != 1) {
		return FALSE;
		}
	*p = vp.s_ii(0);
	*e = ve.s_ii(0);
	return TRUE;
}

#if 0
void factor_prime_power(int n, int *p, int *e)
//Computes $p$ and $e$ with $n=p^e$. 
//If $n$ is not a prime power, an error is raised.
{
	Vector vp, ve;

	factor_integer(n, vp, ve);
	if (vp.s_l() != 1) {
		cout << "factor_prime_power() the number is not a prime power\n";
		exit(1);
		}
	*p = vp.s_ii(0);
	*e = ve.s_ii(0);
}
#endif

int Euler(int n)
//Computes Eulers $\varphi$-function for $n$.
//Uses the prime factorization of $n$. before: eulerfunc
{
	Vector p, e;
	int i, k, p1, e1, l;
	
	factor_integer(n, p, e);
	k = 1;
	l = p.s_l();
	for (i = 0; i < l; i++) {
		p1 = p.s_ii(i);
		e1 = e.s_ii(i);
		if (e1 > 1)
			k *= i_power_j(p1, e1 - 1);
		k *= (p1 - 1);
		}
	return k;
}

int Moebius(int i)
//Computes the number-theoretic $\mu$ (= moebius) function of $i$. before: moebius.
{
	Vector vp, ve;
	int j, a, l;
	
	factor_integer(i, vp, ve);
	l = vp.s_l();
	for (j = 0; j < l; j++) {
		a = ve.s_ii(j);
		if (a > 1)
			return 0;
		}
	if (EVEN(l))
		return 1;
	else
		return -1;
}

#if 0
int order_mod_p(int a, int p)
//Computes the order of $a$ mod $p$, i.~e. the smallest $k$ 
//s.~th. $a^k \equiv 1$ mod $p$.
{
	int o, b;
	
	if (a < 0) {
		cout << "order_mod_p() a < 0\n";
		exit(1);
		}
	a %= p;
	if (a == 0)
		return 0;
	if (a == 1)
		return 1;
	o = 1;
	b = a;
	while (b != 1) {
		b *= a;
		b %= p;
		o++;
		}
	return o;
}
#endif

#if 0
int primitive_root(int p, int f_v)
//Computes a primitive element for $\EZ_p$, i.~e. an integer $k$ 
//with $2 \le k \le p - 1$ s.~th. the order of $k$ mod $p$ is $p-1$.
{
	int i, o;

	if (p < 2) {
		cout << "primitive_root(): p < 2\n";
		exit(1);
		}
	if (p == 2)
		return 1;
	for (i = 2; i < p; i++) {
		o = order_mod_p(i, p);
		if (o == p - 1) {
			if (f_v) {
				cout << i << " is primitive root mod " << p << endl;
				}
			return i;
			}
		}
	cout << "no primitive root found\n";
	exit(1);
}
#endif


int NormRemainder(int a, int m)
//absolute smallest remainder: Computes $r$ such that 
//$a \equiv r$ mod $m$ and $- \frac{m}{2} < r \le \frac{m}{2}$ holds.
{
	int q, m0, m1, m_halbe;
	
	if (m == 0) {
		cout << "NormRemainder() m == 0\n";
		exit(1);
		}
	m0 = m;
	m_halbe = m0 >> 1;
	q = a / m0;
	m1 = a - q * m0;
	if (m1 > m_halbe)
		m1 -= m0;
	if (ODD(m0)) {
		if (m1 < - m_halbe)
			m1 += m0;
		}
	else {
		if (m1 <= - m_halbe)
			m1 += m0;
		}
	return m1;
}

int log2(int n)
//returns $\log_2(n)$ 
{
	int i;
	
	if (n <= 0) {
		cout << "log2(): n <= 0\n";
		exit(1);
		}
	for (i = -1; n > 0; i++) {
		n >>= 1;
		}
	return i;
}

int sqrt_mod(int a, int p)
// solves x^2 = a mod p. Returns x
{
	int a1, x;
	
	a1 = a % p;
	if (p < 300) {
		if (a1 < 0) {
			a1 = - a1;
			a1 = a1 % p;
			if (a1) {
				a1 = p - a1;
				}
			}
		for (x = 0; x < p; x++) {
			if ((x * x) % p == a1)
				return x;
			}
		cout << "sqrt_mod() a not a quadratic residue\n";
		cout << "a = " << a << " p=" << p <<"\n";
		exit(1);
		}
	else {
		x = sqrt_mod_involved(a1, p);
		longinteger X, Y, P;
		
		X.homo_z(x);
		P.homo_z(p);
		Y.mult_mod(X, X, P);
		if (Y.modp(p) != a1) {
			cout << "sqrt_mod() error in sqrt_mod_invoved\n";
			exit(1);
			}
		return x;
		}
}

int sqrt_mod_involved(int a, int p)
// solves x^2 = a mod p. Returns x
{
	int verbose_level = 0;
	longinteger P, m1;
	longinteger A, X, a2, a4, b, X2;
	int round;
	
	A.homo_z(a);
	P.homo_z(p);
	if (p % 4 == 3) {
		X = A;
		X.power_int_mod((p + 1) >> 2, P);
		return X.s_i();
		}
	if (p % 8 == 5) {
		b = A;
		b.power_int_mod((p - 1) >> 2, P);
		// cout << "A = " << A << endl;
		// cout << "b = A^(p-1)/4=" << b << endl;
		if (b.is_one()) {
			X = A;
			X.power_int_mod((p + 3) >> 3, P);
			return X.s_i();
			}
		m1 = P;
		m1.dec();
		if (b.compare_with(m1) == 0) {
			a2.add_mod(A, A, P);
			a4.add_mod(a2, a2, P);
			a4.power_int_mod((p - 5) >> 3, P);
			X.mult_mod(a2, a4, P);
			return X.s_i();
			}
		else {
			cout << "sqrt_mod() p % 8 = 5 and power neq +-1\n";
			cout << "power = " << b << endl;
			cout << "-1 = " << m1 << endl;
			exit(1);
			}
		}
	// now p % 8 == 1
	// Tonelli / Shanks algorithm:
	int n, r = 0, q, e, m;
	longinteger Z, N, Y, B, T, d, mP, AB, Ypower, Bpower;

#ifdef TONELLI_VERBOSE
	cout << "sqrt_mod(), Tonelli / Shanks:\n";
#endif
	q = p - 1;
	e = 0;
	while (EVEN(q)) {
		q >>= 1;
		e++;
		}
#ifdef TONELLI_VERBOSE
	cout << "p - 1 = 2^" << e << " * " << q << endl;
#endif

#if 0
	do {
		n = (int)(((double)rand() * (double)p / RAND_MAX)) % p;
		r = Legendre(n, p, verbose_level - 1);
		} while (r >= 0);
#else
	for (n = 1; n < p - 1; n++) {
		r = Legendre(n, p, verbose_level - 1);
		if (r == -1) {
			break;
			}
		}
#endif

#ifdef TONELLI_VERBOSE
	cout << "n=" << n << " p=" << p << " Legendre(n,p)=" << r<< endl;
#endif
	cout << "n=" << n << " p=" << p << " Legendre(n,p)=" << r<< endl;
	N.homo_z(n);
	Z = N;
	Z.power_int_mod(q, P);
	Y = Z;
	r = e;
	X = A;
	X.power_int_mod((q - 1) >> 1, P);
	d.mult_mod(X, X, P);
	B.mult_mod(A, d, P);
	X.mult_mod(A, X, P);
#ifdef TONELLI_VERBOSE
	cout << "initialization:\n";
#endif	
	round = 0;
	while (TRUE) {
#ifdef TONELLI_VERBOSE
		cout << "Z=" << Z << endl;
		cout << "Y=" << Y << endl;
		cout << "r=" << r << endl;
		cout << "X=" << X << endl;
		cout << "B=" << B << endl;
#endif


		X2.mult_mod(X, X, P);
		AB.mult_mod(A, B, P);
		Ypower = Y;
		Ypower.power_int_mod(1 << (r - 1), P);
		Bpower = B;
		Bpower.power_int_mod(1 << (r - 1), P);

		d = Y;
		d.power_int_mod(1 << (r - 1), P);
		mP = P;
		mP.negate();
		d += mP;
		if (!d.is_m_one()) {
			cout << "loop invariant violated: Y^{2^{r-1}} != -1\n";
			exit(1);
			}
		
		d.mult_mod(A, B, P);
		//X2.mult_mod(X, X, P);
		if (d.compare_with(X2) != 0) {
			cout << "loop invariant violated: ab != x^2\n";
			cout << "ab=" << d << endl;
			cout << "x^2=" << X2 << endl;
			exit(1);
			}

		d = B;
		d.power_int_mod(1 << (r - 1), P);
		if (!d.is_one()) {
			cout << "loop invariant violated: B^{2^{r-1}} != 1\n";
			exit(1);
			}

		
		if (B.modp(p) == 1) {
			m = -1;
			}
		else {
			for (m = 1; ; m++) {
				d = B;
				d.power_int_mod(1 << m, P);
				if (d.is_one())
					break;
				if (m >= r) {
					cout << "sqrt_mod(), Tonelli / Shanks:\n";
					cout << "error: a is not a quadratic residue mod p\n";
					exit(1);
					}
				}
			}

		
		cout << round << " & " << A << " & " << B << " & " << X << " & " << X2 << " & " << Y << " & " << r << " & " << AB << " & " << Ypower << " & " << Bpower << " & ";

		if (m == -1) {
			cout << " & & & & \\\\" << endl;
			}
		else {
			cout << m;
			}
		
		//cout << "m=" << m << endl;

		if (m == -1) {
			return X.s_i();
			}

#ifdef TONELLI_VERBOSE
		cout << "m=" << m << endl;
#endif
		T = Y;
		T.power_int_mod(1 << (r - m - 1), P);
		Y.mult_mod(T, T, P);
		r = m;
		X.mult_mod(X, T, P);
		B.mult_mod(B, Y, P);

		cout << " & " << Y << " & " << X << " & " << B << " & " << r;
		cout << "\\\\" << endl;
		round++;
		}
	//exit(1);
}

#if 0
void latex_head(ostream& ost, int f_book, int f_title, char *title, char *author, int f_toc, int f_landscape)
{
ost << "\\documentclass[12pt]{";
if (f_book)
	ost << "book";
else
	ost << "article";
ost << "}\n"; 
ost << "% a4paper\n";
ost << endl;
ost << "%\\usepackage[dvips]{epsfig}\n"; 
ost << "%\\usepackage{cours11, cours}\n"; 
ost << "%\\usepackage{fancyheadings}\n"; 
ost << "%\\usepackage{amstex}\n"; 
ost << "%\\usepackage{calc}\n"; 
ost << "\\usepackage{amsmath}\n"; 
ost << "\\usepackage{amssymb}\n"; 
ost << "\\usepackage{latexsym}\n"; 
ost << "\\usepackage{epsf}\n"; 
ost << "\\usepackage{supertabular}\n"; 
ost << "%\\usepackage{wrapfig}\n"; 
ost << "%\\usepackage{blackbrd}\n"; 
ost << "%\\usepackage{epic,eepic}\n"; 
ost << "\\usepackage{rotating}\n"; 
ost << "\\usepackage{multicol}\n"; 
ost << "\\usepackage{multirow}\n"; 
ost << "\\usepackage{makeidx} % additional command see\n"; 
ost << "\\usepackage{epsfig}\n"; 
ost << "%\\usepackage{amsmath,amsfonts} \n"; 
ost << endl;
ost << endl;
ost << "%\\usepackage[mtbold,mtplusscr]{mathtime}\n"; 
ost << "% lucidacal,lucidascr,\n"; 
ost << endl;
ost << "%\\usepackage{mathtimy}\n"; 
ost << "%\\usepackage{bm}\n"; 
ost << "%\\usepackage{avant}\n"; 
ost << "%\\usepackage{basker}\n"; 
ost << "%\\usepackage{bembo}\n"; 
ost << "%\\usepackage{bookman}\n"; 
ost << "%\\usepackage{chancery}\n"; 
ost << "%\\usepackage{garamond}\n"; 
ost << "%\\usepackage{helvet}\n"; 
ost << "%\\usepackage{newcent}\n"; 
ost << "%\\usepackage{palatino}\n"; 
ost << "%\\usepackage{times}\n"; 
ost << "%\\usepackage{pifont}\n"; 
ost << endl;
ost << endl;
ost << endl;
ost << "%\\parindent=0pt\n"; 
ost << endl;
ost << "\\renewcommand{\\baselinestretch}{1.5}\n"; 
ost << endl;
ost << "\\hoffset -1.2cm\n"; 
ost << "\\voffset -3.7cm\n"; 
ost << endl;
ost << "%\\oddsidemargin=15pt\n"; 
ost << endl;
ost << "\\oddsidemargin 0pt\n"; 
ost << "\\evensidemargin 0pt\n"; 
ost << "%\\topmargin 0pt\n"; 
ost << endl;
ost << "%\\topmargin=0pt\n"; 
ost << "%\\headsep=18pt\n"; 
ost << "%\\footskip=45pt\n"; 
ost << "%\\mathsurround=1pt\n"; 
ost << "%\\evensidemargin=0pt\n"; 
ost << "%\\oddsidemargin=15pt\n"; 
ost << endl;
if (f_landscape) {
	ost << "\\textwidth = 25cm\n"; 
	ost << "\\textheight= 17cm\n"; 
	}
else {
	ost << "\\textwidth = 17cm\n"; 
	ost << "\\textheight= 25cm\n"; 
	}
ost << endl;
ost << "%\\setlength{\\textheight}{\\baselineskip*41+\\topskip}\n"; 
ost << endl;

ost << "\\newcommand{\\Aut}{{\\rm Aut}}\n"; 
ost << "\\newcommand{\\Sym}{{\\rm Sym}}\n"; 
ost << "\\newcommand{\\sFix}{{\\cal Fix}}\n"; 
ost << "\\newcommand{\\sOrbits}{{\\cal Orbits}}\n"; 
//ost << "\\newcommand{\\sFix}{{\\mathscr Fix}}\n"; 
//ost << "\\newcommand{\\sOrbits}{{\\mathscr Orbits}}\n"; 
ost << "\\newcommand{\\Stab}{{\\rm Stab}}\n"; 
ost << "\\newcommand{\\Fix}{{\\rm Fix}}\n"; 
ost << "\\newcommand{\\fix}{{\\rm fix}}\n"; 
ost << "\\newcommand{\\Orbits}{{\\rm Orbits}}\n"; 
ost << "\\newcommand{\\PG}{{\\rm PG}}\n"; 
ost << "\\newcommand{\\AG}{{\\rm AG}}\n"; 
ost << "\\newcommand{\\SQS}{{\\rm SQS}}\n"; 
ost << "\\newcommand{\\STS}{{\\rm STS}}\n"; 
//ost << "\\newcommand{\\Sp}{{\\rm Sp}}\n"; 
ost << "\\newcommand{\\PSL}{{\\rm PSL}}\n"; 
ost << "\\newcommand{\\PGL}{{\\rm PGL}}\n"; 
ost << "\\newcommand{\\PSSL}{{\\rm P\\Sigma L}}\n"; 
ost << "\\newcommand{\\PGGL}{{\\rm P\\Gamma L}}\n"; 
ost << "\\newcommand{\\SL}{{\\rm SL}}\n"; 
ost << "\\newcommand{\\GL}{{\\rm GL}}\n"; 
ost << "\\newcommand{\\SSL}{{\\rm \\Sigma L}}\n"; 
ost << "\\newcommand{\\GGL}{{\\rm \\Gamma L}}\n"; 
ost << "\\newcommand{\\ASL}{{\\rm ASL}}\n"; 
ost << "\\newcommand{\\AGL}{{\\rm AGL}}\n"; 
ost << "\\newcommand{\\ASSL}{{\\rm A\\Sigma L}}\n"; 
ost << "\\newcommand{\\AGGL}{{\\rm A\\Gamma L}}\n"; 
ost << "\\newcommand{\\PSU}{{\\rm PSU}}\n"; 
ost << "\\newcommand{\\HS}{{\\rm HS}}\n"; 
ost << "\\newcommand{\\Hol}{{\\rm Hol}}\n"; 
ost << "\\newcommand{\\SO}{{\\rm SO}}\n"; 
ost << "\\newcommand{\\ASO}{{\\rm ASO}}\n"; 

ost << "\\newcommand{\\la}{\\langle}\n"; 
ost << "\\newcommand{\\ra}{\\rangle}\n"; 


ost << "\\newcommand{\\cA}{{\\cal A}}\n"; 
ost << "\\newcommand{\\cB}{{\\cal B}}\n"; 
ost << "\\newcommand{\\cC}{{\\cal C}}\n"; 
ost << "\\newcommand{\\cD}{{\\cal D}}\n"; 
ost << "\\newcommand{\\cE}{{\\cal E}}\n"; 
ost << "\\newcommand{\\cF}{{\\cal F}}\n"; 
ost << "\\newcommand{\\cG}{{\\cal G}}\n"; 
ost << "\\newcommand{\\cH}{{\\cal H}}\n"; 
ost << "\\newcommand{\\cI}{{\\cal I}}\n"; 
ost << "\\newcommand{\\cJ}{{\\cal J}}\n"; 
ost << "\\newcommand{\\cK}{{\\cal K}}\n"; 
ost << "\\newcommand{\\cL}{{\\cal L}}\n"; 
ost << "\\newcommand{\\cM}{{\\cal M}}\n"; 
ost << "\\newcommand{\\cN}{{\\cal N}}\n"; 
ost << "\\newcommand{\\cO}{{\\cal O}}\n"; 
ost << "\\newcommand{\\cP}{{\\cal P}}\n"; 
ost << "\\newcommand{\\cQ}{{\\cal Q}}\n"; 
ost << "\\newcommand{\\cR}{{\\cal R}}\n"; 
ost << "\\newcommand{\\cS}{{\\cal S}}\n"; 
ost << "\\newcommand{\\cT}{{\\cal T}}\n"; 
ost << "\\newcommand{\\cU}{{\\cal U}}\n"; 
ost << "\\newcommand{\\cV}{{\\cal V}}\n"; 
ost << "\\newcommand{\\cW}{{\\cal W}}\n"; 
ost << "\\newcommand{\\cX}{{\\cal X}}\n"; 
ost << "\\newcommand{\\cY}{{\\cal Y}}\n"; 
ost << "\\newcommand{\\cZ}{{\\cal Z}}\n"; 

ost << "\\newcommand{\\rmA}{{\\rm A}}\n"; 
ost << "\\newcommand{\\rmB}{{\\rm B}}\n"; 
ost << "\\newcommand{\\rmC}{{\\rm C}}\n"; 
ost << "\\newcommand{\\rmD}{{\\rm D}}\n"; 
ost << "\\newcommand{\\rmE}{{\\rm E}}\n"; 
ost << "\\newcommand{\\rmF}{{\\rm F}}\n"; 
ost << "\\newcommand{\\rmG}{{\\rm G}}\n"; 
ost << "\\newcommand{\\rmH}{{\\rm H}}\n"; 
ost << "\\newcommand{\\rmI}{{\\rm I}}\n"; 
ost << "\\newcommand{\\rmJ}{{\\rm J}}\n"; 
ost << "\\newcommand{\\rmK}{{\\rm K}}\n"; 
ost << "\\newcommand{\\rmL}{{\\rm L}}\n"; 
ost << "\\newcommand{\\rmM}{{\\rm M}}\n"; 
ost << "\\newcommand{\\rmN}{{\\rm N}}\n"; 
ost << "\\newcommand{\\rmO}{{\\rm O}}\n"; 
ost << "\\newcommand{\\rmP}{{\\rm P}}\n"; 
ost << "\\newcommand{\\rmQ}{{\\rm Q}}\n"; 
ost << "\\newcommand{\\rmR}{{\\rm R}}\n"; 
ost << "\\newcommand{\\rmS}{{\\rm S}}\n"; 
ost << "\\newcommand{\\rmT}{{\\rm T}}\n"; 
ost << "\\newcommand{\\rmU}{{\\rm U}}\n"; 
ost << "\\newcommand{\\rmV}{{\\rm V}}\n"; 
ost << "\\newcommand{\\rmW}{{\\rm W}}\n"; 
ost << "\\newcommand{\\rmX}{{\\rm X}}\n"; 
ost << "\\newcommand{\\rmY}{{\\rm Y}}\n"; 
ost << "\\newcommand{\\rmZ}{{\\rm Z}}\n"; 

ost << "\\newcommand{\\bA}{{\\bf A}}\n"; 
ost << "\\newcommand{\\bB}{{\\bf B}}\n"; 
ost << "\\newcommand{\\bC}{{\\bf C}}\n"; 
ost << "\\newcommand{\\bD}{{\\bf D}}\n"; 
ost << "\\newcommand{\\bE}{{\\bf E}}\n"; 
ost << "\\newcommand{\\bF}{{\\bf F}}\n"; 
ost << "\\newcommand{\\bG}{{\\bf G}}\n"; 
ost << "\\newcommand{\\bH}{{\\bf H}}\n"; 
ost << "\\newcommand{\\bI}{{\\bf I}}\n"; 
ost << "\\newcommand{\\bJ}{{\\bf J}}\n"; 
ost << "\\newcommand{\\bK}{{\\bf K}}\n"; 
ost << "\\newcommand{\\bL}{{\\bf L}}\n"; 
ost << "\\newcommand{\\bM}{{\\bf M}}\n"; 
ost << "\\newcommand{\\bN}{{\\bf N}}\n"; 
ost << "\\newcommand{\\bO}{{\\bf O}}\n"; 
ost << "\\newcommand{\\bP}{{\\bf P}}\n"; 
ost << "\\newcommand{\\bQ}{{\\bf Q}}\n"; 
ost << "\\newcommand{\\bR}{{\\bf R}}\n"; 
ost << "\\newcommand{\\bS}{{\\bf S}}\n"; 
ost << "\\newcommand{\\bT}{{\\bf T}}\n"; 
ost << "\\newcommand{\\bU}{{\\bf U}}\n"; 
ost << "\\newcommand{\\bV}{{\\bf V}}\n"; 
ost << "\\newcommand{\\bW}{{\\bf W}}\n"; 
ost << "\\newcommand{\\bX}{{\\bf X}}\n"; 
ost << "\\newcommand{\\bY}{{\\bf Y}}\n"; 
ost << "\\newcommand{\\bZ}{{\\bf Z}}\n"; 

#if 0
ost << "\\newcommand{\\sA}{{\\mathscr A}}\n"; 
ost << "\\newcommand{\\sB}{{\\mathscr B}}\n"; 
ost << "\\newcommand{\\sC}{{\\mathscr C}}\n"; 
ost << "\\newcommand{\\sD}{{\\mathscr D}}\n"; 
ost << "\\newcommand{\\sE}{{\\mathscr E}}\n"; 
ost << "\\newcommand{\\sF}{{\\mathscr F}}\n"; 
ost << "\\newcommand{\\sG}{{\\mathscr G}}\n"; 
ost << "\\newcommand{\\sH}{{\\mathscr H}}\n"; 
ost << "\\newcommand{\\sI}{{\\mathscr I}}\n"; 
ost << "\\newcommand{\\sJ}{{\\mathscr J}}\n"; 
ost << "\\newcommand{\\sK}{{\\mathscr K}}\n"; 
ost << "\\newcommand{\\sL}{{\\mathscr L}}\n"; 
ost << "\\newcommand{\\sM}{{\\mathscr M}}\n"; 
ost << "\\newcommand{\\sN}{{\\mathscr N}}\n"; 
ost << "\\newcommand{\\sO}{{\\mathscr O}}\n"; 
ost << "\\newcommand{\\sP}{{\\mathscr P}}\n"; 
ost << "\\newcommand{\\sQ}{{\\mathscr Q}}\n"; 
ost << "\\newcommand{\\sR}{{\\mathscr R}}\n"; 
ost << "\\newcommand{\\sS}{{\\mathscr S}}\n"; 
ost << "\\newcommand{\\sT}{{\\mathscr T}}\n"; 
ost << "\\newcommand{\\sU}{{\\mathscr U}}\n"; 
ost << "\\newcommand{\\sV}{{\\mathscr V}}\n"; 
ost << "\\newcommand{\\sW}{{\\mathscr W}}\n"; 
ost << "\\newcommand{\\sX}{{\\mathscr X}}\n"; 
ost << "\\newcommand{\\sY}{{\\mathscr Y}}\n"; 
ost << "\\newcommand{\\sZ}{{\\mathscr Z}}\n"; 
#else
ost << "\\newcommand{\\sA}{{\\cal A}}\n"; 
ost << "\\newcommand{\\sB}{{\\cal B}}\n"; 
ost << "\\newcommand{\\sC}{{\\cal C}}\n"; 
ost << "\\newcommand{\\sD}{{\\cal D}}\n"; 
ost << "\\newcommand{\\sE}{{\\cal E}}\n"; 
ost << "\\newcommand{\\sF}{{\\cal F}}\n"; 
ost << "\\newcommand{\\sG}{{\\cal G}}\n"; 
ost << "\\newcommand{\\sH}{{\\cal H}}\n"; 
ost << "\\newcommand{\\sI}{{\\cal I}}\n"; 
ost << "\\newcommand{\\sJ}{{\\cal J}}\n"; 
ost << "\\newcommand{\\sK}{{\\cal K}}\n"; 
ost << "\\newcommand{\\sL}{{\\cal L}}\n"; 
ost << "\\newcommand{\\sM}{{\\cal M}}\n"; 
ost << "\\newcommand{\\sN}{{\\cal N}}\n"; 
ost << "\\newcommand{\\sO}{{\\cal O}}\n"; 
ost << "\\newcommand{\\sP}{{\\cal P}}\n"; 
ost << "\\newcommand{\\sQ}{{\\cal Q}}\n"; 
ost << "\\newcommand{\\sR}{{\\cal R}}\n"; 
ost << "\\newcommand{\\sS}{{\\cal S}}\n"; 
ost << "\\newcommand{\\sT}{{\\cal T}}\n"; 
ost << "\\newcommand{\\sU}{{\\cal U}}\n"; 
ost << "\\newcommand{\\sV}{{\\cal V}}\n"; 
ost << "\\newcommand{\\sW}{{\\cal W}}\n"; 
ost << "\\newcommand{\\sX}{{\\cal X}}\n"; 
ost << "\\newcommand{\\sY}{{\\cal Y}}\n"; 
ost << "\\newcommand{\\sZ}{{\\cal Z}}\n"; 
#endif

ost << "\\newcommand{\\frakA}{{\\mathfrak A}}\n"; 
ost << "\\newcommand{\\frakB}{{\\mathfrak B}}\n"; 
ost << "\\newcommand{\\frakC}{{\\mathfrak C}}\n"; 
ost << "\\newcommand{\\frakD}{{\\mathfrak D}}\n"; 
ost << "\\newcommand{\\frakE}{{\\mathfrak E}}\n"; 
ost << "\\newcommand{\\frakF}{{\\mathfrak F}}\n"; 
ost << "\\newcommand{\\frakG}{{\\mathfrak G}}\n"; 
ost << "\\newcommand{\\frakH}{{\\mathfrak H}}\n"; 
ost << "\\newcommand{\\frakI}{{\\mathfrak I}}\n"; 
ost << "\\newcommand{\\frakJ}{{\\mathfrak J}}\n"; 
ost << "\\newcommand{\\frakK}{{\\mathfrak K}}\n"; 
ost << "\\newcommand{\\frakL}{{\\mathfrak L}}\n"; 
ost << "\\newcommand{\\frakM}{{\\mathfrak M}}\n"; 
ost << "\\newcommand{\\frakN}{{\\mathfrak N}}\n"; 
ost << "\\newcommand{\\frakO}{{\\mathfrak O}}\n"; 
ost << "\\newcommand{\\frakP}{{\\mathfrak P}}\n"; 
ost << "\\newcommand{\\frakQ}{{\\mathfrak Q}}\n"; 
ost << "\\newcommand{\\frakR}{{\\mathfrak R}}\n"; 
ost << "\\newcommand{\\frakS}{{\\mathfrak S}}\n"; 
ost << "\\newcommand{\\frakT}{{\\mathfrak T}}\n"; 
ost << "\\newcommand{\\frakU}{{\\mathfrak U}}\n"; 
ost << "\\newcommand{\\frakV}{{\\mathfrak V}}\n"; 
ost << "\\newcommand{\\frakW}{{\\mathfrak W}}\n"; 
ost << "\\newcommand{\\frakX}{{\\mathfrak X}}\n"; 
ost << "\\newcommand{\\frakY}{{\\mathfrak Y}}\n"; 
ost << "\\newcommand{\\frakZ}{{\\mathfrak Z}}\n"; 

ost << "\\newcommand{\\fraka}{{\\mathfrak a}}\n"; 
ost << "\\newcommand{\\frakb}{{\\mathfrak b}}\n"; 
ost << "\\newcommand{\\frakc}{{\\mathfrak c}}\n"; 
ost << "\\newcommand{\\frakd}{{\\mathfrak d}}\n"; 
ost << "\\newcommand{\\frake}{{\\mathfrak e}}\n"; 
ost << "\\newcommand{\\frakf}{{\\mathfrak f}}\n"; 
ost << "\\newcommand{\\frakg}{{\\mathfrak g}}\n"; 
ost << "\\newcommand{\\frakh}{{\\mathfrak h}}\n"; 
ost << "\\newcommand{\\fraki}{{\\mathfrak i}}\n"; 
ost << "\\newcommand{\\frakj}{{\\mathfrak j}}\n"; 
ost << "\\newcommand{\\frakk}{{\\mathfrak k}}\n"; 
ost << "\\newcommand{\\frakl}{{\\mathfrak l}}\n"; 
ost << "\\newcommand{\\frakm}{{\\mathfrak m}}\n"; 
ost << "\\newcommand{\\frakn}{{\\mathfrak n}}\n"; 
ost << "\\newcommand{\\frako}{{\\mathfrak o}}\n"; 
ost << "\\newcommand{\\frakp}{{\\mathfrak p}}\n"; 
ost << "\\newcommand{\\frakq}{{\\mathfrak q}}\n"; 
ost << "\\newcommand{\\frakr}{{\\mathfrak r}}\n"; 
ost << "\\newcommand{\\fraks}{{\\mathfrak s}}\n"; 
ost << "\\newcommand{\\frakt}{{\\mathfrak t}}\n"; 
ost << "\\newcommand{\\fraku}{{\\mathfrak u}}\n"; 
ost << "\\newcommand{\\frakv}{{\\mathfrak v}}\n"; 
ost << "\\newcommand{\\frakw}{{\\mathfrak w}}\n"; 
ost << "\\newcommand{\\frakx}{{\\mathfrak x}}\n"; 
ost << "\\newcommand{\\fraky}{{\\mathfrak y}}\n"; 
ost << "\\newcommand{\\frakz}{{\\mathfrak z}}\n"; 


ost << "\\newcommand{\\Tetra}{{\\mathfrak Tetra}}\n"; 
ost << "\\newcommand{\\Cube}{{\\mathfrak Cube}}\n"; 
ost << "\\newcommand{\\Octa}{{\\mathfrak Octa}}\n"; 
ost << "\\newcommand{\\Dode}{{\\mathfrak Dode}}\n"; 
ost << "\\newcommand{\\Ico}{{\\mathfrak Ico}}\n"; 

ost << endl;
ost << endl;
ost << endl;
ost << "%\\makeindex\n"; 
ost << endl;
ost << "\\begin{document} \n"; 
ost << endl;	
ost << "\\bibliographystyle{plain}\n"; 
ost << "%\\large\n"; 
ost << endl;
ost << "{\\allowdisplaybreaks%\n"; 
ost << endl;
ost << endl;
ost << endl;
ost << endl;
ost << "%\\makeindex\n"; 
ost << endl;
ost << "\\renewcommand{\\labelenumi}{(\\roman{enumi})}\n"; 
ost << endl;

if (f_title) {
	ost << "\\title{" << title << "}\n"; 
	ost << "\\author{" << author << "}%end author\n"; 
	ost << "%\\date{}\n"; 
	ost << "\\maketitle%\n"; 
	}
ost << "\\pagenumbering{roman}\n"; 
ost << "%\\thispagestyle{empty}\n"; 
if (f_toc) {
	ost << "\\tableofcontents\n"; 
	}
ost << "%\\input et.tex%\n"; 
ost << "%\\thispagestyle{empty}%\\phantom{page2}%\\clearpage%\n"; 
ost << "%\\addcontentsline{toc}{chapter}{Inhaltsverzeichnis}%\n"; 
ost << "%\\tableofcontents\n"; 
ost << "%\\listofsymbols\n"; 
if (f_toc){
	ost << "\\clearpage\n"; 
	ost << endl;
	}
ost << "\\pagenumbering{arabic}\n"; 
ost << "%\\pagenumbering{roman}\n"; 
ost << endl;
ost << endl;
ost << endl;
}


void latex_foot(ostream& ost)
{
ost << endl;
ost << endl;
ost << "%\\bibliographystyle{gerplain}% wird oben eingestellt\n"; 
ost << "%\\addcontentsline{toc}{section}{References}\n"; 
ost << "%\\bibliography{../MY_BIBLIOGRAPHY/anton}\n"; 
ost << "% ACHTUNG: nicht vergessen:\n"; 
ost << "% die Zeile\n"; 
ost << "%\\addcontentsline{toc}{chapter}{Literaturverzeichnis}\n"; 
ost << "% muss per Hand in d.bbl eingefuegt werden !\n"; 
ost << "% nach \\begin{thebibliography}{100}\n"; 
ost << endl;
ost << "%\\begin{theindex}\n"; 
ost << endl;
ost << "%\\clearpage\n"; 
ost << "%\\addcontentsline{toc}{chapter}{Index}\n"; 
ost << "%\\input{apd.ind}\n"; 
ost << endl;
ost << "%\\printindex\n"; 
ost << "%\\end{theindex}\n"; 
ost << endl;
ost << "}% allowdisplaybreaks\n"; 
ost << endl;
ost << "\\end{document}\n"; 
ost << endl;
ost << endl;
}
#endif

void html_head(ostream& ost, char *title_long, char *title_short)
{
	ost << "<html>\n";
	ost << "<head>\n";
	ost << "<title>\n";
	ost << title_short << "\n";
	ost << "</title>\n";
	ost << "</head>\n";
	
	ost << "<body>\n";
	ost << "<h1>\n";
	ost << title_long << "\n";
	ost << "</h1>\n";
	ost << "<hr>\n";
}

void html_foot(ostream& ost)
{
	hollerith h;
	
	h.get_current_date();
	ost << "<p><hr><p>\n";
	ost << "created: " << h.s() << endl;
	ost << "</body>\n";
	ost << "</html>\n";
}

void sieve(Vector &primes, int factorbase, int f_v)
{
	int i, from, to, l, unit_size = 1000;
	
	primes.m_l(0);
	for (i = 0; ; i++) {
		from = i * unit_size + 1;
		to = from + unit_size - 1;
		sieve_primes(primes, from, to, factorbase, FALSE);
		l = primes.s_l();
		cout << "[" << from << "," << to 
			<< "], total number of primes = " 
			<< l << endl;
		if (l >= factorbase)
			break;
		}
	
	if (f_v) {
		print_intvec_mod_10(primes);
		}
}

void sieve_primes(Vector &v, int from, int to, int limit, int f_v)
{
	int x, y, l, k, p, f_prime;
	
	l = v.s_l();
	if (ODD(from))
		x = from;
	else
		x = from + 1;
	for (; x <= to; x++, x++) {
		if (x <= 1)
			continue;
		f_prime = TRUE;
		for (k = 0; k < l; k++) {
			p = v[k].s_i_i();
			y = x / p;
			// cout << "x=" << x << " p=" << p << " y=" << y << endl;
			if ((x - y * p) == 0) {
				f_prime = FALSE;
				break;
				}
			if (y < p)
				break;
#if 0
			if ((x % p) == 0)
				break;
#endif
			}
		if (!f_prime)
			continue;
		if (nb_primes(x) != 1) {
			cout << "error: " << x << " is not prime!\n";
			}
		v.append_integer(x);
		if (f_v) {
			cout << v.s_l() << " " << x << endl;
			}
		l++;
		if (l >= limit)
			return;
		}
}

void print_intvec_mod_10(Vector &v)
{
	int i, l;
	
	l = v.s_l();
	for (i = 0; i < l; i++) {
		cout << v.s_ii(i) << " ";
		if ((i + 1) % 10 == 0)
			cout << endl;
		}
	cout << endl;
}

#if 0
#include <unistd.h>
	/* for sysconf */
#include <limits.h>
	/* for CLK_TCK */
#include <sys/types.h>
#include <sys/times.h>
	/* for times() */
#include <time.h>
	/* for time() */

#define SYSTEMUNIX


int os_ticks()
{
#ifdef SYSTEMMAC
	clock_t t;
	
	t = clock();
	return((int)t);
#endif
#ifdef SYSTEMUNIX
	struct tms tms_buffer;

	if (-1 == (int) times(&tms_buffer))
		return(-1);
	return(tms_buffer.tms_utime);
#endif
	return(0);
}

static int system_time0 = 0;

int os_ticks_system()
{
#ifdef SYSTEMMAC
	clock_t t;
	
	t = clock();
	return((int)t);
#endif
#ifdef SYSTEMUNIX
#if 0
	struct tms tms_buffer;

	if (-1 == times(&tms_buffer))
		return(-1);
	return(tms_buffer.tms_stime);
#endif
	int t;

	t = time(NULL);
	if (system_time0 == 0) {
		system_time0 = t;
		}
	t -= system_time0;
	t *= os_ticks_per_second();
	return t;
#endif
	return(0);
}

int os_ticks_per_second()
{
	int clk_tck = 1;
	
#ifdef SYSTEMUNIX
	clk_tck = sysconf(_SC_CLK_TCK);
	/* printf("clk_tck = %d\n", clk_tck); */
#endif
#ifdef SYSTEMMAC
	clk_tck = CLOCKS_PER_SEC;
#endif
	return(clk_tck);
}

void os_ticks_to_dhms(int ticks, int tps, int &d, int &h, int &m, int &s)
{
	int l1;

	l1 = ticks / tps;
	s = l1 % 60;
	l1 /= 60;
	m = l1 % 60;
	l1 /= 60;
	h = l1;
	if (h >= 24) {
		d = h / 24;
		h = h % 24;
		}
	else
		d = 0;
}

#undef DEBUG_TRANSFORM_LLUR

void transform_llur(int *in, int *out, int &x, int &y)
{
	int dx, dy;
	double a, b;

#ifdef DEBUG_TRANSFORM_LLUR
	cout << "transform_llur: " << x << "," << y << " -> ";
#endif
	dx = x - in[0];
	dy = y - in[1];
	a = (double) dx / (double)(in[2] - in[0]);
	b = (double) dy / (double)(in[3] - in[1]);
	dx = (int)(a * (double)(out[2] - out[0]));
	dy = (int)(b * (double)(out[3] - out[1]));
	x = dx + out[0];
	y = dy + out[1];
#ifdef DEBUG_TRANSFORM_LLUR
	cout << x << "," << y << endl;
#endif
}

void transform_dist(int *in, int *out, int &x, int &y)
{
	int dx, dy;
	double a, b;

	a = (double) x / (double)(in[2] - in[0]);
	b = (double) y / (double)(in[3] - in[1]);
	dx = (int)(a * (double) (out[2] - out[0]));
	dy = (int)(b * (double) (out[3] - out[1]));
	x = dx;
	y = dy;
}

void transform_dist_x(int *in, int *out, int &x)
{
	int dx;
	double a;

	a = (double) x / (double)(in[2] - in[0]);
	dx = (int)(a * (double) (out[2] - out[0]));
	x = dx;
}

void transform_dist_y(int *in, int *out, int &y)
{
	int dy;
	double b;

	b = (double) y / (double)(in[3] - in[1]);
	dy = (int)(b * (double) (out[3] - out[1]));
	y = dy;
}
#endif

void stirling_second(int n, int k, int f_ordered, discreta_base &res, int f_v)
// number of set partitions of an n-set with exactly k classes
{

	// cout << "stirling_second() partition is currently disabled" << endl;
	discreta_base a, b, c;
	number_partition p;

	if (n == 0) {
		if (k == 0) {
			res.m_i_i(1);
			return;
			}
		else {
			res.m_i_i(0);
			return;
			}
		}
	if (k == 0) {
		res.m_i_i(0);
		return;
		}
	if (k > n) {
		res.m_i_i(0);
		return;
		}
	
	if (f_v) {
		cout << "stirling_second";
		if (f_ordered) {
			cout << " ordered ";
			}
		cout << "(" << n << ", " << k << ")" << endl;
		}
	a.m_i_i(0);
	p.first_into_k_parts(n, k);
	do {
		if (f_v) {
			cout << p << endl;
			}
		p.multinomial_ordered(b, f_v);
		a += b;
		} while (p.next_into_k_parts(n, k));
	if (!f_ordered) {
		b.factorial(k);
		a.integral_division_exact(b, c); 
		c.swap(a);
		}
	a.swap(res);
	if (f_v) {
		cout << "stirling_second";
		if (f_ordered) {
			cout << " ordered ";
			}
		cout << "(" << n << ", " << k << ") = " << res << endl;
		}
}

void stirling_first(int n, int k, int f_signless, discreta_base &res, int f_v)
// $(-1)^{n+k} \cdot$ number of elements in $\Sym_n$ with exactly k cycles
{
	// cout << "stirling_first() partition is currently disabled" << endl;
	discreta_base a, b, c;
	number_partition p;
	int l, x, ax;

	if (n == 0) {
		if (k == 0) {
			res.m_i_i(1);
			return;
			}
		else {
			res.m_i_i(0);
			return;
			}
		}
	if (k == 0) {
		res.m_i_i(0);
		return;
		}
	if (k > n) {
		res.m_i_i(0);
		return;
		}
	
	if (f_v) {
		cout << "stirling_first";
		if (f_signless) {
			cout << " signless ";
			}
		cout << "(" << n << ", " << k << ")" << endl;
		}
	a.m_i_i(0);
	p.first_into_k_parts(n, k);
	do {
		if (f_v) {
			cout << p << endl;
			}
		p.multinomial_ordered(b, f_v);


		l = p.s_l();
		for (x = 1; x <= l; x++) {
			ax = p[x - 1];
			if (ax == 0)
				continue;
			c.factorial(x - 1);
			c.power_int(ax);
			b *= c;
			}

		a += b;

		} while (p.next_into_k_parts(n, k));

	b.factorial(k);
	a.integral_division_exact(b, c);
	
	if (!f_signless) {
		if (ODD(n + k))
			c.negate();
		}
	
	res = c;
	if (f_v) {
		cout << "stirling_first";
		if (f_signless) {
			cout << " signless ";
			}
		cout << "(" << n << ", " << k << ") = " << res << endl;
		}
}

void Catalan(int n, Vector &v, int f_v)
{
	int i;
	
	v.m_l_n(n + 1);
	v[0].m_i_i(1);
	v[1].m_i_i(1);
	for (i = 2; i <= n; i++) {
		Catalan_n(i, v, v[i], f_v);
		}
}

void Catalan_n(int n, Vector &v, discreta_base &res, int f_v)
{
	int i;
	discreta_base a, b;
	
	a.m_i_i(0);
	for (i = 1; i <= n; i++) {
		b = v[i - 1];
		b *= v[n - i];
		a += b;
		}
	if (f_v) {
		cout << "Catalan_n(" << n << ")=" << a << endl;
		}
	a.swap(res);
}

void Catalan_nk_matrix(int n, matrix &Cnk, int f_v)
{
	int i, k;
	discreta_base a;
	
	Catalan_nk_star_matrix(n, Cnk, f_v);
	for (k = n; k > 0; k--) {
		for (i = 0; i <= n; i++) {
			a = Cnk[i][k - 1];
			a.negate();
			Cnk[i][k] += a;
			}
		}
}

void Catalan_nk_star_matrix(int n, matrix &Cnk, int f_v)
{
	int i, k;
	
	Cnk.m_mn_n(n + 1, n + 1);
	
	Cnk[0][0].m_i_i(1);
	for (k = 1; k <= n; k++) {
		Cnk[0][k].m_i_i(1);
		}
	
	for (k = 1; k <= n; k++) {
		Cnk[1][k].m_i_i(1);
		}


	for (i = 2; i <= n; i++) {
		Cnk[i][1].m_i_i(1);
		for (k = 2; k <= n; k++) {
			Catalan_nk_star(i, k, Cnk, Cnk[i][k], f_v);
			}
		}
}

void Catalan_nk_star(int n, int k, matrix &Cnk, discreta_base &res, int f_v)
{
	int i;
	discreta_base a, b;
	
	a.m_i_i(0);
	for (i = 0; i <= n - 1; i++) {
		b = Cnk[i][k - 1];
		b *= Cnk[n - i - 1][k];
		a += b;
		}
	if (f_v) {
		cout << "Catalan_nk_star(" << n << ", " << k << ")=" << a << endl;
		}
	a.swap(res);
}

int atoi(char *p)
{
	int x;
	sscanf(p, "%d", &x);
	return x;
#if 0
	char str[1024];
	int x;
	
	strcpy(str, p);
	istrstream ins(str, sizeof(str));
	ins >> x;
	return x;
#endif
}

#if 0
void itoa(char *p, int len_of_p, int i)
{
	sprintf(p, "%d", i);
#if 0
	ostrstream os(p, len_of_p);
	os << i << ends;
#endif
}

static int f_has_swap_initialized = FALSE;
static int f_has_swap = 0;
	// indicates if char swap is present 
	// i.e., little endian / big endian 

static void test_swap()
{
    unsigned long test_long = 0x11223344L;
    SCHAR *ptr;
    
    ptr = (char *) &test_long;
    f_has_swap = (ptr[0] == 0x44);
    f_has_swap_initialized = TRUE;
}
#endif

#if 0
// block_swap_chars:
// turns round the chars within 
// "no" intervalls of "size" in the 
// buffer pointed to by "ptr" 
// this routine goes back to Roland Grund

void block_swap_chars(SCHAR *ptr, int size, int no)
{
	SCHAR *ptr_end, *ptr_start;
	SCHAR chr;
	int i;
	
	if (!f_has_swap_initialized)
		test_swap();
	if ((f_has_swap) && (size > 1)) {

		for(; no--; ) {
	
			ptr_start = ptr;
			ptr_end = ptr_start + (size - 1);
			for(i = size / 2; i--; ) {
				chr = *ptr_start;
				*ptr_start++ = *ptr_end;
				*ptr_end-- = chr;
				}
			ptr += size;
			}
		}
}
#endif

#if 0
#include <cstdio>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

int file_size(char *name)
{
#ifdef SYSTEMUNIX
	int handle, size;
	
	handle = open(name, O_RDWR/*mode*/);
	size = lseek(handle, 0L, SEEK_END);
	close(handle);
	return(size);
#endif
#ifdef SYSTEMMAC
	int handle, size;
	
	handle = open(name, O_RDONLY);
		/* THINK C Unix Lib */
	size = lseek(handle, 0L, SEEK_END);
		/* THINK C Unix Lib */
	close(handle);
	return(size);
#endif
#ifdef SYSTEMWINDOWS
	int handle = _open (name,_O_RDONLY);
	int size   = _lseek (handle,0,SEEK_END);
	close (handle);
	return (size);
#endif

}

#include <ctype.h>

int s_scan_int(char **s, int *i)
{
	char str1[512];
	
	if (!s_scan_token(s, str1))
		return FALSE;
	if (strcmp(str1, ",") == 0) {
		if (!s_scan_token(s, str1))
			return FALSE;
		}
	*i = atoi(str1);
	return TRUE;
}

int s_scan_token(char **s, char *str)
{
	char c;
	int len;
	
	while (TRUE) {
		c = **s;
		if (c == 0) {
			return(FALSE);
			}
		if (c == ' ' || c == '\t' || 
			c == '\r' || c == 10 || c == 13) {
			(*s)++;
			continue;
			}
		break;
		}
	len = 0;
	c = **s;
	if (isalpha(c)) {
		while (isalnum(c) || c == '_') {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
			}
		str[len] = 0;
		}
	else if (isdigit(c) || c == '-') {
		str[len++] = c;
		(*s)++;
		c = **s;
		while (isdigit(c)) {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
			}
		str[len] = 0;
		}
	else {
		str[0] = c;
		str[1] = 0;
		(*s)++;		
		}
	// printf("token = \"%s\"\n", str);
	return TRUE;
}

int s_scan_token_arbitrary(char **s, char *str)
{
	char c;
	int len;
	
	//cout << "s_scan_token_arbitrary:" << *s << endl;
	str[0] = 0;
	while (TRUE) {
		c = **s;
		if (c == 0) {
			return(FALSE);
			}
		if (c == ' ' || c == '\t' || 
			c == '\r' || c == 10 || c == 13) {
			(*s)++;
			continue;
			}
		break;
		}
	//cout << "s_scan_token_arbitrary:" << *s << endl;
	len = 0;
	c = **s;
	while (c != ' ' && c != '\t' && 
		c != '\r' && c != 10 && c != 13 && c != 0) {
		str[len] = c;
		len++;
		(*s)++;
		c = **s;
		}
	str[len] = 0;
	// printf("token = \"%s\"\n", str);
	return TRUE;
}

int s_scan_str(char **s, char *str)
{
	char c;
	int len, f_break;
	
	while (TRUE) {
		c = **s;
		if (c == 0) {
			return(FALSE);
			}
		if (c == ' ' || c == '\t' || 
			c == '\r' || c == 10 || c == 13) {
			(*s)++;
			continue;
			}
		break;
		}
	if (c != '\"') {
		cout << "s_scan_str() error: c != '\"'" << endl;
		return(FALSE);
		}
	(*s)++;
	len = 0;
	f_break = FALSE;
	while (TRUE) {
		c = **s;
		if (c == 0) {
			break;
			}
		if (c == '\\') {
			(*s)++;
			c = **s;
			str[len] = c;
			len++;
			}
		else if (c == '\"') {
			f_break = TRUE;
			}
		else {
			str[len] = c;
			len++;
			}
		(*s)++;
		if (f_break)
			break;
		}
	str[len] = 0;
	return TRUE;
}
#endif

void N_choose_K(discreta_base & n, int k, discreta_base & res)
// Computes ${n \choose k}$ into res as an {\em object}.
// This function uses a recursion formula.
{
	discreta_base n1, res1, k1, tmp;
	
	if (k < 0) {
		cout << "N_choose_K(): k < 0" << endl;
		exit(1);
		}
	k1.m_i_i(k);
#if 0
	if (k1.gt(n)) {
		res->m_i_i(0);
		}
#endif
	if (k == 0) {
		res.m_i_i(1);
		return;
		}
	if (k == 1) {
		res = n;
		return;
		}
	if (n.is_one()) {
		res.m_i_i(1);
		return;
		}
	n1 = n;
	n1.dec();
	N_choose_K(n1, k - 1, res1);
	res1 *= n;
	// a = n * n_choose_k(n - 1, k - 1);
	k1.m_i_i(k);
	res1.integral_division_exact(k1, res);
	// b = a / k;
}

void Binomial(int n, int k, discreta_base & n_choose_k)
// Computes binomial coefficients as {\em objects} 
// so that large numbers are no problem. 
// This function uses an internal table to remember all 
// previously computed values. This may speed up 
// those computations where Binomial() is heavily involved !
{
	static matrix *T = NULL;
	int tn, i, j;
	
	if (k < 0) {
		cout << "Binomial(): k < 0" << endl;
		exit(1);
		}
	if (k > n) {
		n_choose_k.m_i_i(0);
		return;
		}
	if (n == 0 || k == 0 || k == n) {
		n_choose_k.m_i_i(1);
		return;
		}
	if (T == NULL) {
		T = (matrix *) callocobject(MATRIX);
		T->m_mn_n(10, 10);
		}
	tn = T->s_m();
	if (tn < n + 1) {
		matrix TT;

#if 0
		cout << "reallocating table of binomial coefficients to length " << n + 1 << endl;
#endif
		TT.m_mn_n(n + 1, n + 1);
		for (i = 0; i < tn; i++) {
			for (j = 0; j <= i; j++) {
				(*T)[i][j].swap(TT[i][j]);
				}
			}
		TT.swap(*T);
		}
	Binomial_using_table(n, k, *T, n_choose_k);
}

static void Binomial_using_table(int n, int k, matrix & T, discreta_base & res)
{
	discreta_base tmp1, tmp2;
	int m;
	
	m = T.s_m();
	if (m < n) {
		cout << "Binomial_using_table: m < n" << endl;
		exit(1);
		}
	if (k > n) {
		cout << "Binomial_using_table: k > n" << endl;
		exit(1);
		}
	if (k > (n >> 1) + 1) {
		Binomial_using_table(n, n - k, T, res);
		T[n][k] = T[n][n - k];
		return;
		}
	if (n == 0) {
		cout << "Binomial_using_table: n == 0" << endl;
		exit(1);
		}
	if (n < 0) {
		cout << "Binomial_using_table: n < 0" << endl;
		exit(1);
		}
	if (k < 0) {
		cout << "Binomial_using_table: k < 0" << endl;
		exit(1);
		}
	if (n == k) {
		T.m_iji(n, k, 1);
		res.m_i_i(1);
		return;
		}
	if (k == 0) {
		T.m_iji(n, k, 1);
		res.m_i_i(1);
		return;
		}
	if (k == 1) {
		T.m_iji(n, k, n);
		res.m_i_i(n);
		return;
		}
	if (T.s_ij(n, k).is_zero()) {
		Binomial_using_table(n - 1, k - 1, T, tmp1);
		Binomial_using_table(n - 1, k, T, tmp2);
		tmp1 += tmp2;
		res = tmp1;
		T[n][k] = tmp1;
		}
	else {
		res = T[n][k];
		}
}

void Krawtchouk(int n, int q, int i, int j, discreta_base & a)
// $\sum_{u=0}^{\min(i,j)} (-1)^u \cdot (q-1)^{i-u} \cdot {j \choose u} \cdot $
// ${n - j \choose i - u}$
{
	int u, u_max;
	discreta_base b, c, d;
	
	a.m_i_i(0);
	u_max = MINIMUM(i, j);
	for (u = 0; u <= u_max; u++) {
		b.m_i_i(q - 1);
		b.power_int(i - u);
		if (ODD(u))
			b.negate();
		Binomial(j, u, c);
		Binomial(n - j, i - u, d);
		b *= c;
		b *= d;
		a += b;
		}
}

#if 0
int ij2k(int i, int j, int n)
{
	if (i == j) {
		cout << "ij2k() i == j" << endl;
		exit(1);
		}
	if (i > j)
		return ij2k(j, i, n);
	return ((n - i) * i + ((i * (i - 1)) >> 1) + j - i - 1);
}

void k2ij(int k, int & i, int & j, int n)
{
	int ii;
	
	for (ii = 0; ii < n; ii++) {
		if (k < n - ii - 1) {
			i = ii;
			j = k + ii + 1;
			return;
			}
		k -= (n - ii - 1);
		}
	cout << "k too large" << endl;
	exit(1);
}
#endif

void tuple2_rank(int rank, int &i, int &j, int n, int f_injective)
//enumeration of 2-tuples $(i,j)$ (f_injective TRUE iff $i=j$ forbidden).
//this routine produces the tuple with number ``rank'' into $i$ and $j$. 
//$n$ is the number of points $1 \le i,j \le n$.
{
	int a;
	
	if (f_injective) {
		a = rank % (n - 1);
		i = (rank - a) / (n - 1);
		if (a < i)
			j = a;
		else
			j = a + 1;
		}
	else {
		a = rank % n;
		i = (rank - a) / n;
		j = a;
		}
}

int tuple2_unrank(int i, int j, int n, int f_injective)
//inverse function of tuple2_rank(): 
//returns the rank of a given tuple $(i,j)$.
{
	int rank;
	
	if (f_injective) {
		rank = i * (n - 1);
		if (j < i)
			rank += j;
		else if (j == i) {
			cout << "tuple2_unrank() not injective !" << endl;
			exit(1);
			}
		else
			rank += j - 1;
		}
	else {
		rank = i * n + j;
		}
	return rank;
}

#if 0
void chop_off_extension_if_present(char *p, char *ext)
{
	int l1 = strlen(p);
	int l2 = strlen(ext);
	
	if (l1 > l2 && strcmp(p + l1 - l2, ext) == 0) {
		p[l1 - l2] = 0;
		}
}

void get_extension_if_present(char *p, char *ext)
{
	int i, l = strlen(p);
	
	ext[0] = 0;
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '.') {
			strcpy(ext, p + i);
			}
		}
}
#endif

#include <string.h>

void output_texable_string(ostream & ost, char *in)
{
	int i, l;
	char str[100], c;
	
	l = strlen(in);
	for (i = 0; i < l; i++) {
		c = in[i];
		if (c == '_') {
			ost << "\\_";
			}
		else {
			str[0] = c;
			str[1] = 0;
			ost << str;
			}
		}
}

void texable_string(char *in, char *out)
{
	int i, l;
	char str[100], c;
	
	l = strlen(in);
	out[0] = 0;
	for (i = 0; i < l; i++) {
		c = in[i];
		if (c == '_') {
			strcat(out, "\\_");
			}
		else {
			str[0] = c;
			str[1] = 0;
			strcat(out, str);
			}
		}
}

static int table_of_primes[] = { 
		2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 
		31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 
		73, 79, 83, 89, 97 } ; // the first 25 primes 
static int table_of_primes_len = 25;

void the_first_n_primes(Vector &P, int n)
{
	int i;
	
	if (n <= table_of_primes_len) {
		P.m_l(n);
		for (i = 0; i < n; i++) {
			P.m_ii(i, table_of_primes[i]);
			}
		}
	else {
		cout << "the_first_n_primes: n too large" << endl;
		exit(1);
		}
}

#include <math.h>

#if 0
double cos_grad(double phi)
{
	double x;

	x = (phi * M_PI) / 180.;
	return cos(x);
}

double sin_grad(double phi)
{
	double x;

	x = (phi * M_PI) / 180.;
	return sin(x);
}

double tan_grad(double phi)
{
	double x;

	x = (phi * M_PI) / 180.;
	return tan(x);
}

double atan_grad(double x)
{
	double y, phi;

	y = atan(x);
	phi = (y * 180.) / M_PI;
	return phi;
}

void on_circle_int(int *Px, int *Py, int idx, int angle_in_degree, int rad)
{
	
	Px[idx] = (int)(cos_grad(angle_in_degree) * (double) rad);
	Py[idx] = (int)(sin_grad(angle_in_degree) * (double) rad);
}
#endif

void midpoint_of_2(int *Px, int *Py, int i1, int i2, int idx)
{
	double x, y;
	
	x = (double)(Px[i1] + Px[i2]) * 0.5;
	y = (double)(Py[i1] + Py[i2]) * 0.5;
	Px[idx] = (int) x;
	Py[idx] = (int) y;
}

void midpoint_of_5(int *Px, int *Py, int i1, int i2, int i3, int i4, int i5, int idx)
{
	double x, y;
	
	x = (double)(Px[i1] + Px[i2] + Px[i3] + Px[i4] + Px[i5]) * 0.2;
	y = (double)(Py[i1] + Py[i2] + Py[i3] + Py[i4] + Py[i5]) * 0.2;
	Px[idx] = (int) x;
	Py[idx] = (int) y;
}

void ratio_int(int *Px, int *Py, int idx_from, int idx_to, int idx_result, double r)
{
	int dx, dy;

	dx = (int)((double)(Px[idx_to] - Px[idx_from]) * r);
	dy = (int)((double)(Py[idx_to] - Py[idx_from]) * r);
	Px[idx_result] = Px[idx_from] + dx;
	Py[idx_result] = Py[idx_from] + dy;
}

void time_check_delta(int dt)
{
	int tps, d, h, min, s;

	tps = os_ticks_per_second();
	os_ticks_to_dhms(dt, tps, d, h, min, s);

	if ((dt / tps) >= 1) {
		if (d > 0) {
			cout << d << "-" << h << ":" << min << ":" << s;
			}
		else if (h > 0) {
			cout << h << ":" << min << ":" << s;
			}
		else  {
			cout << min << ":" << s;
			}
		}
	else {
		cout << "0:00";
		}
	cout << endl;
}

void time_check(int t0)
{
	int t1, dt;
	
	t1 = os_ticks();
	dt = t1 - t0;
	time_check_delta(dt);
}

int nb_of_bits()
{
	return sizeof (uint) * 8;
}

void bit_set(uint & g, int k)
{
	g |= (((uint) 1) << k);
}

void bit_clear(uint & g, int k)
{
	g &= ~(((uint) 1) << k);
}

int bit_test(uint & g, int k)
{
	return ( (g & (((uint) 1) << k)) ? TRUE : FALSE );
}

void bitset2vector(uint g, Vector &v)
{
	int i = 0;
	
	v.m_l(0);
	while (g != 0) {
		if (bit_test(g, 0)) {
			v.append_integer(i);
			}
		i++;
		g >>= 1;
		}
}

void frobenius_in_PG(domain *dom, int n, permutation &p)
// n is the projective dimension
{
	with ww(dom);
	int i, j, l;
	Vector v;
	
	int qq = dom->order_int();
	int q = dom->order_subfield_int();
	l = nb_PG_elements(n, qq);
	p.m_l(l);
	v.m_l_n(n + 1);
	for (i = 0; i < l; i++) {
		v.PG_element_unrank(i);
		for (j = 0; j <= n; j++) {
			v[j].power_int(q);
			}
		v.PG_element_rank(j);
		p.m_ii(i, j);
		}
}

void frobenius_in_AG(domain *dom, int n, permutation &p)
// n is the dimension
{
	with ww(dom);
	int i, j, l;
	Vector v;
	
	int qq = dom->order_int();
	int q = dom->order_subfield_int();
	l = nb_AG_elements(n, qq);
	p.m_l(l);
	v.m_l_n(n);
	for (i = 0; i < l; i++) {
		v.AG_element_unrank(i);
		for (j = 0; j < n; j++) {
			v[j].power_int(q);
			}
		v.AG_element_rank(j);
		p.m_ii(i, j);
		}
}

void translation_in_AG(domain *dom, int n, int i, discreta_base & a, permutation &p)
{
	with ww(dom);
	int ii, j, l;
	Vector v;
	
	int q = dom->order_int();
	l = nb_AG_elements(n, q);
	p.m_l(l);
	v.m_l_n(n);
	for (ii = 0; ii < l; ii++) {
		v.AG_element_unrank(ii);
		// cout << "ii=" << ii << " v=" << v;
		v[i] += a;
		v.AG_element_rank(j);
		p.m_ii(ii, j);
		// cout << " j=" << j << endl;
		}
}

enum printing_mode_enum current_printing_mode()
{
	if (printing_mode_stack_size == 0)
		return printing_mode_ascii;
	else
		return printing_mode_stack[printing_mode_stack_size - 1];
}

printing_mode::printing_mode(enum printing_mode_enum printing_mode)
{
	if (printing_mode_stack_size == MAX_PRintING_MODE_STACK) {
		cout << "printing_mode() overflow in printing_mode stack" << endl;
		exit(1);
		}
	printing_mode_stack[printing_mode_stack_size++] = printing_mode;
}

printing_mode::~printing_mode()
{
	if (printing_mode_stack_size == 0) {
		cout << "~printing_mode() underflow in printing_mode stack" << endl;
		exit(1);
		}
	printing_mode_stack_size--;
}

void call_system(char *cmd)
{
	printf("executing: %s\n", cmd);
	system(cmd);
}

void fill_char(void *v, int cnt, int c)
{
	char ch = (char)c;
	char *s;
	
	s = (char *)v;
	cnt++;
	while (--cnt)
		*s++ = ch;
}

#define HASH_PRIME ((int)(1 << 30)-1)

int hash_int(int hash0, int a)
{
	int h = hash0;
	
	do {
		h <<= 1;
		if (ODD(a)) {
			h++;
			}
		h = h % HASH_PRIME;
		a >>= 1;
		} while (a);
	return h;
}

void queue_init(Vector &Q, int elt)
{
	Q.m_l(2);
	Q.m_ii(0, 1);
	Q[1].change_to_vector();
	Q[1].as_vector().m_l_n(1);
	Q[1].as_vector().m_ii(0, elt);
}

int queue_get_and_remove_first_element(Vector &Q)
{
	int elt, i, l, a;
	Vector &v = Q[1].as_vector();
	
	l = Q.s_ii(0);
	if (l <= 0) {
		cout << "queue_get_and_remove_first_element() queue is empty" << endl;
		exit(1);
		}
	elt = v.s_ii(0);
	for (i = 0; i < l - 1; i++) {
		a = v.s_ii(i + 1);
		v.m_ii(i, a);
		}
	Q.m_ii(0, l - 1);
	return elt;
}

int queue_length(Vector &Q)
{
	int l;
	
	l = Q.s_ii(0);
	return l;
}

void queue_append(Vector &Q, int elt)
{
	int l;
	
	Vector &v = Q[1].as_vector();
	l = Q.s_ii(0);
	if (v.s_l() == l) {
		v.append_integer(elt);
		}
	else {
		v.m_ii(l, elt);
		}
	Q.m_ii(0, l + 1);
}

void print_classification_tex(Vector &content, Vector &multiplicities)
{
	print_classification_tex(content, multiplicities, cout);
}

void print_classification_tex(Vector &content, Vector &multiplicities, ostream& ost)
{
	int i, l;
	
	l = content.s_l();
	// ost << "(";
	for (i = 0; i < l; i++) {
		ost << content[i];
		if (!multiplicities[i].is_one()) {
			ost << "^{" << multiplicities[i] << "}";
			}
		if (i < l - 1)
			ost << ",$ $";
		}
	// ost << ")";	
}

#if 0
void perm_move(int *from, int *to, int n)
{
	int i;
	
	for (i = 0; i < n; i++)
		to[i] = from[i];
}

void perm_mult(int *a, int *b, int *c, int n)
{
	int i;
	
	for (i = 0; i < n; i++) {
		c[i] = b[a[i]];
		}
}

void perm_conjugate(int *a, int *b, int *c, int n)
// c := a^b = b^-1 * a * b
{
	int i, j, k;
	
	for (i = 0; i < n; i++) {
		j = b[i];
		// now b^-1(j) = i
		k = a[i];
		k = b[k];
		c[j] = k;
		}
}

void perm_inverse(int *a, int *b, int n)
// b := a^-1
{
	int i, j;
	
	for (i = 0; i < n; i++) {
		j = a[i];
		b[j] = i;
		}
}

void perm_raise(int *a, int *b, int e, int n)
// b := a^e (e >= 0)
{
	int i, j, k;
	
	for (i = 0; i < n; i++) {
		k = i;
		for (j = 0; j < e; j++) {
			k = a[k];
			}
		b[i] = k;
		}
}

void perm_print(int *a, int n)
{
	permutation p;
	perm2permutation(a, n, p);
	cout << p;
}
#endif

void perm2permutation(int *a, int n, permutation &p)
{
	int i;
	
	p.m_l_n(n);
	for (i = 0; i < n; i++) {
		p.s_i(i) = a[i];
		}
}

#if 0
void print_integer_matrix(ostream &ost, int *p, int m, int n)
{
	matrix M;
	int i, j;
	
	M.m_mn_n(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			M.m_iji(i, j, p[i * n + j]);
			}
		}
	ost << M;
}
#endif

#if 0
void print_longinteger_matrix(ostream &ost, LONGint *p, int m, int n)
{
	matrix M;
	int i, j;
	longinteger a;
	
	M.m_mn_n(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a.homo_z(p[i * n + j]);
			M.s_ij(i, j) = a;
			}
		}
	ost << M;
}
#endif

int Gauss_int(int *A, int f_special, int f_complete, int *base_cols, 
	int f_P, int *P, int m, int n, int Pn, 
	int q, int *add_table, int *mult_table, int *negate_table, int *inv_table, int f_v)
// returns the rank which is the number of entries in base_cols
// A is a m x n matrix,
// P is a m x Pn matrix (if f_P is TRUE)
{
	int rank, i, j, k, jj;
	int pivot, pivot_inv = 0, a, b, c, z, f;
	
	if (f_v) {
		cout << "Gauss algorithm for matrix:" << endl;
		print_integer_matrix(cout, A, m, n);
		}
	i = 0;
	for (j = 0; j < n; j++) {
	
		/* search for pivot element: */
		for (k = i; k < m; k++) {
			if (A[k * n + j]) {
				// pivot element found: 
				if (k != i) {
					for (jj = j; jj < n; jj++) {
						int_swap(A[i * n + jj], A[k * n + jj]);
						}
					if (f_P) {
						for (jj = 0; jj < Pn; jj++) {
							int_swap(P[i * Pn + jj], P[k * Pn + jj]);
							}
						}
					}
				break;
				} // if != 0 
			} // next k
		
		if (k == m) // no pivot found 
			continue; // increase j, leave i constant
		
		if (f_v) {
			cout << "row " << i << " pivot in row " << k << " colum " << j << endl;
			}
		
		base_cols[i] = j;

		pivot = A[i * n + j];
		pivot_inv = inv_table[pivot];
		if (!f_special) {
			// make pivot to 1: 
			for (jj = j; jj < n; jj++) {
				A[i * n + jj] = mult_table[A[i * n + jj] * q + pivot_inv];
				}
			if (f_P) {
				for (jj = 0; jj < Pn; jj++) {
					P[i * Pn + jj] = mult_table[P[i * Pn + jj] * q + pivot_inv];
					}
				}
			if (f_v) {
				cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv 
					<< " made to one: " << A[i * n + j] << endl;
				}
			}
		
		/* do the gaussian elimination: */
		for (k = i + 1; k < m; k++) {
			z = A[k * n + j];
			if (z == 0)
				continue;
			if (f_special) {
				f = mult_table[z * q + pivot_inv];
				}
			else {
				f = z;
				}
			f = negate_table[f];
			A[k * n + j] = 0;
			if (f_v) {
				cout << "eliminating row " << k << endl;
				}
			for (jj = j + 1; jj < n; jj++) {
				a = A[i * n + jj];
				b = A[k * n + jj];
				// c := b + f * a
				//    = b - z * a              if !f_special 
				//      b - z * pivot_inv * a  if f_special 
				c = mult_table[f * q + a];
				c = add_table[c * q + b];
				A[k * n + jj] = c;
				if (f_v) {
					cout << A[k * n + jj] << " ";
					}
				}
			if (f_P) {
				for (jj = 0; jj < Pn; jj++) {
					a = P[i * Pn + jj];
					b = P[k * Pn + jj];
					// c := b - z * a
					c = mult_table[f * q + a];
					c = add_table[c * q + b];
					P[k * Pn + jj] = c;
					}
				}
			if (f_v) {
				cout << endl;
				}
			}
		i++;
		if (f_v) {
			cout << "A=" << endl;
			print_integer_matrix(cout, A, m, n);
			if (f_P) {
				cout << "P=" << endl;
				print_integer_matrix(cout, P, m, Pn);
				}
			}
		} // next j 
	rank = i;
	if (f_complete) {
		for (i = rank - 1; i >= 0; i--) {
			j = base_cols[i];
			if (!f_special) {
				a = A[i * n + j];
				}
			else {
				pivot = A[i * n + j];
				pivot_inv = inv_table[pivot];
				}
			// do the gaussian elimination in the upper part: 
			for (k = i - 1; k >= 0; k--) {
				z = A[k * n + j];
				if (z == 0)
					continue;
				A[k * n + j] = 0;
				for (jj = j + 1; jj < n; jj++) {
					a = A[i * n + jj];
					b = A[k * n + jj];
					if (f_special) {
						a = mult_table[a * q + pivot_inv];
						}
					c = mult_table[z * q + a];
					c = negate_table[c];
					c = add_table[c * q + b];
					A[k * n + jj] = c;
					}
				if (f_P) {
					for (jj = 0; jj < Pn; jj++) {
						a = P[i * Pn + jj];
						b = P[k * Pn + jj];
						if (f_special) {
							a = mult_table[a * q + pivot_inv];
							}
						c = mult_table[z * q + a];
						c = negate_table[c];
						c = add_table[c * q + b];
						P[k * Pn + jj] = c;
						}
					}
				} // next k
			} // next i
		}
	if (f_v) {
		cout << "the rank is " << rank << endl;
		}
	return rank;
}

void char_move(char *p, char *q, int len)
{
	int i;
	
	for (i = 0; i < len; i++) 
		*q++ = *p++;
}

#if 0
void char_swap(char *p, char *q, int len)
{
	int i;
	char c;
	
	for (i = 0; i < len; i++) {
		c = *q;
		*q++ = *p;
		*p++ = c;
		}
}

void uchar_move(uchar *p, uchar *q, int len)
{
	int i;
	
	for (i = 0; i < len; i++) 
		*q++ = *p++;
}

void int_matrix_transpose(int n, int *A)
{
	int i, j;
	
	for (i = 0; i < n; i++) {
		for (j = 0; j < i; j++) {
			if (i != j)
				int_swap(A[i * n + j], A[j * n + i]);
			}
		}
}
#endif

void int_vector_realloc(int *&p, int old_length, int new_length)
{
	int *q = new int[new_length];
	int m, i;
	
	m = MINIMUM(old_length, new_length);
	for (i = 0; i < m; i++) {
		q[i] = p[i];
		}
	delete [] p;
	p = q;
}

void int_vector_shorten(int *&p, int new_length)
{
	int *q = new int[new_length];
	int i;

	for (i = 0; i < new_length; i++) {
		q[i] = p[i];
		}
	delete [] p;
	p = q;
}

void int_matrix_realloc(int *&p, int old_m, int new_m, int old_n, int new_n)
{
	int *q = new int[new_m * new_n];
	int m = MINIMUM(old_m, new_m);
	int n = MINIMUM(old_n, new_n);
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			q[i * new_n + j] = p[i * old_n + j];
			}
		}
	delete [] p;
	p = q;
}

#if 0
void int_matrix_shorten_rows(int *&p, int m, int n)
{
	int *q = new int[m * n];
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			q[i * n + j] = p[i * n + j];
			}
		}
	delete [] p;
	p = q;
}

void pint_matrix_shorten_rows(pint *&p, int m, int n)
{
	pint *q = new pint[m * n];
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			q[i * n + j] = p[i * n + j];
			}
		}
	delete [] p;
	p = q;
}
#endif

int code_is_irreducible(int k, int nmk, int idx_zero, int *M)
{
	static int col_reached[1000];
	static int row_reached[1000];
	
	/* 0: as yet not reached
	 * 1: reached, but as yet not processed
	 * 2: reached and processed. */
	int c_reached = 0; 
		/* number of 1's and 2's in c[] */
	int r_reached = 0;
	int c_processed = 0; 
		/* number of 2's in c[] */
	int r_processed = 0;
	int i, j;
	
	if (nmk >= 1000) {
		cout << "code_is_irreducible() nmk > 1000" << endl;
		exit(1);
		}
	if (k >= 1000) {
		cout << "code_is_irreducible() k > 1000" << endl;
		exit(1);
		}
	for (j = 0; j < nmk; j++)
		col_reached[j] = 0;
	for (i = 0; i < k; i++)
		row_reached[i] = 0;
	row_reached[0] = 1;
	r_reached++;
	i = 0;
	for (j = 0; j < nmk; j++) {
		if (M[i * nmk + j] != 0) {
			if (col_reached[j] == 0) {
				col_reached[j] = 1;
				c_reached++;
				}
			}
		}
	row_reached[0] = 2;
	r_processed++;
	while ((c_processed < c_reached) || 
		(r_processed < r_reached)) {
		
		/* do a column if there is 
		 * an unprocessed one: */
		if (c_processed < c_reached) {
			for (j = 0; j < nmk; j++) {
				/* search unprocessed col */
				if (col_reached[j] == 1) {
					/* do column j */
					for (i = 0; i < k; i++) {
						if (M[i * nmk + j] != idx_zero) {
							if (row_reached[i] == 0) {
								row_reached[i] = 1;
								r_reached++;
								}
							}
						}
					/* column j is now processed */
					col_reached[j] = 2;
					c_processed++;
					break; /* for j */
					}
				}
			}
		
		/* do a row if there is an unprocessed one: */
		if (r_processed < r_reached) {
			for (i = 0; i < k; i++) {
				/* search unprocessed row */
				if (row_reached[i] == 1) {
					/* do row i */
					for (j = 0; j < nmk; j++) {
						if (M[i * nmk + j] != idx_zero) {
							if (col_reached[j] == 0) {
								col_reached[j] = 1;
								c_reached++;
								}
							}
						}
					/* row i is now processed */
					row_reached[i] = 2;
					r_processed++;
					break; /* for i */
					}
				}
			}
		
		}
	// printf("code_is_irreducible() c_processed = %d r_processed = %d "
	//	"nmk = %d k = %d\n", c_processed, r_processed, nmk, k);
	if (c_processed < nmk || r_processed < k)
		return FALSE;
	return TRUE;
}

//#include "mindist.C"

#if 0
int int_vec_search(int *v, int len, int a, int &idx)
{
	int l, r, m, res;
	int f_found = FALSE;
	
	if (len == 0) {
		idx = 0;
		return FALSE;
		}
	l = 0;
	r = len;
	// invariant:
	// v[i] <= a for i < l;
	// v[i] >  a for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle
		res = v[m] - a;
		//cout << "search l=" << l << " m=" << m << " r=" 
		//	<< r << "a=" << a << " v[m]=" << v[m] << " res=" << res << endl;
		if (res <= 0) {
			l = m + 1;
			if (res == 0)
				f_found = TRUE;
			}
		else
			r = m;
		}
	// now: l == r; 
	// and f_found is set accordingly */
	if (f_found)
		l--;
	idx = l;
	return f_found;
}

void uchar_print_bitwise(ostream &ost, uchar u)
{
	uchar mask;
	int i;
	
	for (i = 0; i < 8; i++) {
		mask = ((uchar) 1) << i;
		if (u & mask)
			ost << "1";
		else
			ost << "0";
		}
}
#endif


void fine_tune(finite_field *F, int *mtxD, int verbose_level)
// added Dec 28 2009
// This is here because it uses sqrt_mod_involved
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int q = F->q;

	if (f_v) {
		cout << "fine_tune: tuning matrix:" << endl;
		print_integer_matrix_width(cout,
				mtxD, 4, 4, 4, F->log10_of_q);
		}

	int mtxGram[16];
	int mtxDt[16];
	int mtxE[16];
	int mtxF[16];
	int mtxG[16];

	for (i = 0; i < 16; i++) {
		mtxGram[i] = 0;
		}
	mtxGram[0 * 4 + 1] = 1;
	mtxGram[1 * 4 + 0] = 1;
	mtxGram[2 * 4 + 3] = 1;
	mtxGram[3 * 4 + 2] = 1;
	
	F->transpose_matrix(mtxD, mtxDt, 4, 4);
	F->mult_matrix_matrix(mtxDt, mtxGram, mtxE, 4, 4, 4,
			0 /* verbose_level */);
	F->mult_matrix_matrix(mtxE, mtxD, mtxF, 4, 4, 4,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "D^transpose * Gram * D = " << endl;
		print_integer_matrix_width(cout,
				mtxF, 4, 4, 4, F->log10_of_q);
		}

	int c, d, cv;
	
	c = -1;
	for (i = 0; i < 16; i++) {
		if (mtxGram[i] == 0) {
			if (mtxF[i]) {
				cout << "does not preserve the form" << endl;
				}
			}
		else {
			d = mtxF[i];
			if (c == -1) {
				c = d;
				}
			else {
				if (d != c) {
					cout << "does not preserve the form (type 2 error)" << endl;
					}
				}
			}
		}
	if (f_vv) {
		cout << "scalar c = " << c << endl;
		}
	cv = F->inverse(c);
	if (f_vv) {
		cout << "cv = " << cv << endl;
		}
	int p, h, s;
	is_prime_power(q, p, h);
	if (h > 1) {
		cout << "q is not a prime" << endl;
		exit(1);
		}
	s = sqrt_mod_involved(cv, q);
	if (f_vv) {
		cout << "sqrt(cv) = " << s << endl;
		}


	for (i = 0; i < 16; i++) {
		mtxG[i] = F->mult(mtxD[i], s);
		}

	if (f_vv) {
		cout << "mtxG = s * mtxD:" << endl;
		print_integer_matrix_width(cout,
				mtxG, 4, 4, 4, F->log10_of_q);
		}


	int mtxGt[16];
	F->transpose_matrix(mtxG, mtxGt, 4, 4);
	F->mult_matrix_matrix(mtxGt, mtxGram, mtxE, 4, 4, 4,
			0 /* verbose_level */);
	F->mult_matrix_matrix(mtxE, mtxG, mtxF, 4, 4, 4,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "G^transpose * Gram * G = " << endl;
		print_integer_matrix_width(cout, mtxF, 4, 4, 4, F->log10_of_q);
		}

	
	for (i = 0; i < 16; i++) {
		mtxD[i] = mtxG[i];
		}
	if (f_v) {
		cout << "fine_tune: the resulting matrix is" << endl;
		print_integer_matrix_width(cout, mtxD, 4, 4, 4, F->log10_of_q);
		}


	//int At2[4], As2[4], f_switch2;

	//cout << "calling O4_isomorphism_4to2" << endl;
	//O4_isomorphism_4to2(F, At2, As2, f_switch2, mtxG, verbose_level);
}

}

