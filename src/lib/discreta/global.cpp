// global.cpp
//
// Anton Betten
// 10.11.1999
// moved from D2 to ORBI Nov 15, 2007

#include "foundations/foundations.h"
#include "discreta.h"

using namespace std;


namespace orbiter {
namespace layer2_discreta {



#undef DEBUG_CALLOC_NOBJECTS_PLUS_LENGTH
#undef TONELLI_VERBOSE
#undef DEBUG_INVERT_MOD_INTEGER



/**** global variables ***/


// printing_mode gl_printing_mode = printing_mode_ascii;
// int printing_mode = PRintING_MODE_ASCII;

#define MAX_PRINTING_MODE_STACK 100

int printing_mode_stack_size = 0;
static enum printing_mode_enum printing_mode_stack[MAX_PRINTING_MODE_STACK];


const char *discreta_home = NULL;
const char *discreta_arch = NULL;


/**** global functions ***/

static void Binomial_using_table(int n, int k, discreta_matrix & T, discreta_base & res);

void discreta_init()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	char str[1000];
	file_io Fio;
	
	discreta_home = getenv("DISCRETA_HOME");
	if (discreta_home == NULL) {
		if (f_v) {
			cout << "discreta_init WARNING: $DISCRETA_HOME not set !" << endl;
			}
		discreta_home = ".";
		}
	discreta_arch = getenv("DISCRETA_ARCH");
	if (discreta_arch == NULL) {
		if (f_v) {
			cout << "discreta_init WARNING: $DISCRETA_ARCH not set !" << endl;
			}
		discreta_arch = ".";
		}
	if (discreta_home) {
		
		strcpy(str, discreta_home);
		strcat(str, "/lib/this_is");
		
#if 1
		if (Fio.file_size(str) <= 0) {
			if (f_v) {
				cout << "discreta_init WARNING: can't find my library (DISCRETA_HOME/lib) !" << endl;
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
		cout << "calloc_nobjects no memory" << endl;
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
	cout << "calloc_nobjects_plus_length n=" << n << endl;
#endif
	p = (discreta_base *) operator new((n + 1) * sizeof(discreta_base));
	if (p == NULL) {
		cout << "calloc_nobjects_plus_length no memory" << endl;
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
		cout << "free_nobjects_plus_length length = " << n << " < 0" << endl;
		}
#ifdef DEBUG_CALLOC_NOBJECTS_PLUS_LENGTH
	cout << "free_nobjects_plus_length n=" << n << endl;
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
		cout << "calloc_m_times_n_objects no memory" << endl;
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
		cout << "free_m_times_n_objects m = " << m << " < 0" << endl;
		}
	if (n < 0) {
		cout << "free_m_times_n_objects n = " << n << " < 0" << endl;
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
		case INTEGER: return "INTEGER";
		case VECTOR: return "VECTOR";
		case NUMBER_PARTITION: return "NUMBER_PARTITION";
		case PERMUTATION: return "PERMUTATION";
		
		case MATRIX: return "MATRIX";

		case LONGINTEGER: return "LONGINTEGER";
		//case SUBGROUP_LATTICE: return "SUBGROUP_LATTICE";
		//case SUBGROUP_ORBIT: return "SUBGROUP_ORBIT";

		case MEMORY: return "MEMORY";

		case HOLLERITH: return "HOLLERITH";
		case DATABASE: return "DATABASE";
		case BTREE: return "BTREE";

		case PERM_GROUP: return "PERM_GROUP";
		case BT_KEY: return "BT_KEY";

		case DESIGN_PARAMETER: return "DESIGN_PARAMETER";
		
		//case GROUP_SELECTION: return "GROUP_SELECTION";
		case UNIPOLY: return "UNIPOLY";
		
		case DESIGN_PARAMETER_SOURCE: return "DESIGN_PARAMETER_SOURCE";
		//case SOLID: return "SOLID";

		case BITMATRIX: return "BITMATRIX";
		//case PC_PRESENTATION: return "PC_PRESENTATION";
		//case PC_SUBGROUP: return "PC_SUBGROUP";
		//case GROUP_WORD: return "GROUP_WORD";
		//case GROUP_TABLE: return "GROUP_TABLE";
		//case ACTION: return "ACTION";
		//case GEOMETRY: return "GEOMETRY";
		default: return "unknown kind";
		}
}

const char *action_kind_ascii(action_kind k)
{
	switch(k) {
		case vector_entries: return "vector_entries";
		case vector_positions: return "vector_positions";
		default: return "unknown action_kind";
		}
}

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



int invert_mod_integer(int i, int p)
{
	integer a, b;
	
#ifdef DEBUG_INVERT_MOD_INTEGER
	cout << "invert_mod_integer i=" << i << ", p=" << p << endl;
#endif
	a.m_i(i);
	b.m_i(p);
	a.power_int_mod(p - 2, b);
#ifdef DEBUG_INVERT_MOD_INTEGER
	cout << "i^-1=" << a.s_i() << endl;
#endif
	return a.s_i();

#if 0
	with ww(GFp, p);
	integer x;

#ifdef DEBUG_INVERT_MOD_INTEGER
	cout << "invert_mod_integer i=" << i << ", p=" << p << endl;
#endif
	x.m_i(i);
	x.invert();
#ifdef DEBUG_INVERT_MOD_INTEGER
	cout << "i^-1=" << x.s_i() << endl;
#endif
	return x.s_i();
#endif
}


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

void factor_integer(int n, Vector& primes, Vector& exponents)
//Factors the integer $n = \prod_{i=1}^r p_i^{e_i}$. 
//The vector primes holds the primes $p_i$,
//the vector exponents holds the $e_i$.
{
	int d, last_prime = 2, l;
	number_theory::number_theory_domain NT;
	
	if (n == 0) {
		cout << "factor_integer n == 0" << endl;
		exit(1);
		}
	if (n == 1) {
		primes.m_l(0);
		exponents.m_l(0);
		return;
		}
	d = NT.sp_ge(n, last_prime);
	primes.m_l(1);
	exponents.m_l(1);
	l = 1;
	primes.m_ii(0, d);
	exponents.m_ii(0, 1);
	last_prime = d;
	n /= d;
	while (n != 1) {
		d = NT.sp_ge(n, last_prime);
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

void discreta_print_factorization(Vector& primes, Vector& exponents, ostream &o)
//Prints the factorization.
{
	int i, p, e, l;
	
	l = primes.s_l();
	if (l != exponents.s_l()) {
		cout << "print_factorization l != exponents.s_l()" << endl;
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
		cout << "print_factorization l != exponents.s_l()" << endl;
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
	number_theory::number_theory_domain NT;
	
	if (n < 0)
		n = -n;
	while (n != 1) {
		d = NT.smallest_primedivisor(n);
		i++;
		n /= d;
		}
	return i;
}

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

int Euler(int n)
//Computes Eulers $\varphi$-function for $n$.
//Uses the prime factorization of $n$. before: eulerfunc
{
	Vector p, e;
	int i, k, p1, e1, l;
	number_theory::number_theory_domain NT;
	
	factor_integer(n, p, e);
	k = 1;
	l = p.s_l();
	for (i = 0; i < l; i++) {
		p1 = p.s_ii(i);
		e1 = e.s_ii(i);
		if (e1 > 1)
			k *= NT.i_power_j(p1, e1 - 1);
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


int NormRemainder(int a, int m)
//absolute smallest remainder: Computes $r$ such that 
//$a \equiv r$ mod $m$ and $- \frac{m}{2} < r \le \frac{m}{2}$ holds.
{
	int q, m0, m1, m_halbe;
	
	if (m == 0) {
		cout << "NormRemainder m == 0" << endl;
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
		cout << "log2 n <= 0" << endl;
		exit(1);
		}
	for (i = -1; n > 0; i++) {
		n >>= 1;
		}
	return i;
}

int sqrt_mod(int a, int p, int verbose_level)
// solves x^2 = a mod p. Returns x
{
	int f_v = (verbose_level >= 1);
	int a1, x;
	
	if (f_v) {
		cout << "sqrt_mod a=" << a << " p=" << p << endl;
	}
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
			if ((x * x) % p == a1) {
				if (f_v) {
					cout << "sqrt_mod a=" << a << " p=" << p << " done" << endl;
				}
				return x;
			}
		}
		cout << "sqrt_mod() a not a quadratic residue" << endl;
		cout << "a = " << a << " p=" << p << endl;
		exit(1);
		}
	else {
		x = sqrt_mod_involved(a1, p, verbose_level);
		longinteger X, Y, P;
		
		X.homo_z(x);
		P.homo_z(p);
		Y.mult_mod(X, X, P);
		if (Y.modp(p) != a1) {
			cout << "sqrt_mod error in sqrt_mod_invoved" << endl;
			exit(1);
			}
		if (f_v) {
			cout << "sqrt_mod a=" << a << " p=" << p << " done" << endl;
		}
		return x;
		}
}

int sqrt_mod_involved(int a, int p, int verbose_level)
// solves x^2 = a mod p. Returns x
{
	int f_v = (verbose_level >= 1);
	longinteger P, m1;
	longinteger A, X, a2, a4, b, X2;
	int round;
	number_theory::number_theory_domain NT;
	
	if (f_v) {
		cout << "sqrt_mod_involved" << endl;
	}
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
			if (f_v) {
				cout << "sqrt_mod_involved done" << endl;
			}
			return X.s_i();
			}
		m1 = P;
		m1.dec();
		if (b.compare_with(m1) == 0) {
			a2.add_mod(A, A, P);
			a4.add_mod(a2, a2, P);
			a4.power_int_mod((p - 5) >> 3, P);
			X.mult_mod(a2, a4, P);
			if (f_v) {
				cout << "sqrt_mod_involved done" << endl;
			}
			return X.s_i();
			}
		else {
			cout << "sqrt_mod() p % 8 = 5 and power neq +-1" << endl;
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
		r = NT.Legendre(n, p, verbose_level - 1);
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
			cout << "loop invariant violated: Y^{2^{r-1}} != -1" << endl;
			exit(1);
			}
		
		d.mult_mod(A, B, P);
		//X2.mult_mod(X, X, P);
		if (d.compare_with(X2) != 0) {
			cout << "loop invariant violated: ab != x^2" << endl;
			cout << "ab=" << d << endl;
			cout << "x^2=" << X2 << endl;
			exit(1);
			}

		d = B;
		d.power_int_mod(1 << (r - 1), P);
		if (!d.is_one()) {
			cout << "loop invariant violated: B^{2^{r-1}} != 1" << endl;
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
					cout << "sqrt_mod(), Tonelli / Shanks:" << endl;
					cout << "error: a is not a quadratic residue mod p" << endl;
					exit(1);
					}
				}
			}

		
		cout << round << " & " << A << " & " << B << " & " << X << " & "
				<< X2 << " & " << Y << " & " << r << " & " << AB
				<< " & " << Ypower << " & " << Bpower << " & ";

		if (m == -1) {
			cout << " & & & & \\\\" << endl;
			}
		else {
			cout << m;
			}
		
		//cout << "m=" << m << endl;

		if (m == -1) {
			if (f_v) {
				cout << "sqrt_mod_involved done" << endl;
			}
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
	if (f_v) {
		cout << "sqrt_mod_involved done" << endl;
	}
}

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

void sieve(Vector &primes, int factorbase, int verbose_level)
{
	int f_v = (verbose_level >= 1);
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

void sieve_primes(Vector &v, int from, int to, int limit, int verbose_level)
{
	int f_v = (verbose_level >= 1);
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


void stirling_second(int n, int k, int f_ordered,
		discreta_base &res, int verbose_level)
// number of set partitions of an n-set with exactly k classes
{
	int f_v = (verbose_level >= 1);
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

void stirling_first(int n, int k, int f_signless,
		discreta_base &res, int verbose_level)
// $(-1)^{n+k} \cdot$ number of elements in $\Sym_n$ with exactly k cycles
{
	int f_v = (verbose_level >= 1);
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

void Catalan(int n, Vector &v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	v.m_l_n(n + 1);
	v[0].m_i_i(1);
	v[1].m_i_i(1);
	for (i = 2; i <= n; i++) {
		Catalan_n(i, v, v[i], f_v);
		}
}

void Catalan_n(int n, Vector &v, discreta_base &res, int verbose_level)
{
	int f_v = (verbose_level >= 1);
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

void Catalan_nk_matrix(int n, discreta_matrix &Cnk, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int i, k;
	discreta_base a;
	
	Catalan_nk_star_matrix(n, Cnk, verbose_level);
	for (k = n; k > 0; k--) {
		for (i = 0; i <= n; i++) {
			a = Cnk[i][k - 1];
			a.negate();
			Cnk[i][k] += a;
			}
		}
}

void Catalan_nk_star_matrix(int n, discreta_matrix &Cnk, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
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
			Catalan_nk_star(i, k, Cnk, Cnk[i][k], verbose_level - 1);
			}
		}
}

void Catalan_nk_star(int n, int k, discreta_matrix &Cnk, discreta_base &res, int verbose_level)
{
	int f_v = (verbose_level >= 1);
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
	static discreta_matrix *T = NULL;
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
		T = (discreta_matrix *) callocobject(MATRIX);
		T->m_mn_n(10, 10);
		}
	tn = T->s_m();
	if (tn < n + 1) {
		discreta_matrix TT;

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

static void Binomial_using_table(int n, int k, discreta_matrix & T, discreta_base & res)
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


#if 0
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
#endif

void frobenius_in_PG(domain *dom, int n, permutation &p)
// n is the projective dimension
{
	with ww(dom);
	int i, j, l;
	Vector v;
	geometry::geometry_global Gg;
	
	int qq = dom->order_int();
	int q = dom->order_subfield_int();
	l = Gg.nb_PG_elements(n, qq);
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
	geometry::geometry_global Gg;
	
	int qq = dom->order_int();
	int q = dom->order_subfield_int();
	l = Gg.nb_AG_elements(n, qq);
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

void translation_in_AG(domain *dom, int n, int i,
		discreta_base & a, permutation &p)
{
	with ww(dom);
	int ii, j, l;
	Vector v;
	geometry::geometry_global Gg;
	
	int q = dom->order_int();
	l = Gg.nb_AG_elements(n, q);
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
	if (printing_mode_stack_size == MAX_PRINTING_MODE_STACK) {
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

void perm2permutation(int *a, int n, permutation &p)
{
	int i;
	
	p.m_l_n(n);
	for (i = 0; i < n; i++) {
		p.s_i(i) = a[i];
		}
}

int Gauss_int(int *A, int f_special, int f_complete, int *base_cols, 
	int f_P, int *P, int m, int n, int Pn, 
	int q, int *add_table, int *mult_table,
	int *negate_table, int *inv_table, int verbose_level)
// returns the rank which is the number of entries in base_cols
// A is a m x n matrix,
// P is a m x Pn matrix (if f_P is TRUE)
{
	int f_v = (verbose_level >= 1);
	int rank, i, j, k, jj;
	int pivot, pivot_inv = 0, a, b, c, z, f;
	data_structures::algorithms Algo;
	
	if (f_v) {
		cout << "Gauss_int Gauss algorithm for matrix:" << endl;
		Orbiter->Int_vec->print_integer_matrix(cout, A, m, n);
		}
	i = 0;
	for (j = 0; j < n; j++) {
	
		/* search for pivot element: */
		for (k = i; k < m; k++) {
			if (A[k * n + j]) {
				// pivot element found: 
				if (k != i) {
					for (jj = j; jj < n; jj++) {
						Algo.int_swap(A[i * n + jj], A[k * n + jj]);
						}
					if (f_P) {
						for (jj = 0; jj < Pn; jj++) {
							Algo.int_swap(P[i * Pn + jj], P[k * Pn + jj]);
							}
						}
					}
				break;
				} // if != 0 
			} // next k
		
		if (k == m) // no pivot found 
			continue; // increase j, leave i constant
		
		if (f_v) {
			cout << "Gauss_int row " << i << " pivot in row " << k << " colum " << j << endl;
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
				cout << "Gauss_int pivot=" << pivot << " pivot_inv=" << pivot_inv
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
				cout << "Gauss_int eliminating row " << k << endl;
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
			cout << "Gauss_int A=" << endl;
			Orbiter->Int_vec->print_integer_matrix(cout, A, m, n);
			if (f_P) {
				cout << "Gauss_int P=" << endl;
				Orbiter->Int_vec->print_integer_matrix(cout, P, m, Pn);
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
		cout << "Gauss_int the rank is " << rank << endl;
		}
	return rank;
}

void char_move(char *p, char *q, int len)
{
	int i;
	
	for (i = 0; i < len; i++) 
		*q++ = *p++;
}

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



void fine_tune(field_theory::finite_field *F, int *mtxD, int verbose_level)
// added Dec 28 2009
// This is here because it uses sqrt_mod_involved
// used in algebra/create_element.cpp
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int q = F->q;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "fine_tune: tuning matrix:" << endl;
		Orbiter->Int_vec->print_integer_matrix_width(cout,
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
	
	F->Linear_algebra->transpose_matrix(mtxD, mtxDt, 4, 4);
	F->Linear_algebra->mult_matrix_matrix(mtxDt, mtxGram, mtxE, 4, 4, 4,
			0 /* verbose_level */);
	F->Linear_algebra->mult_matrix_matrix(mtxE, mtxD, mtxF, 4, 4, 4,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "D^transpose * Gram * D = " << endl;
		Orbiter->Int_vec->print_integer_matrix_width(cout,
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
	NT.is_prime_power(q, p, h);
	if (h > 1) {
		cout << "q is not a prime" << endl;
		exit(1);
		}
	s = sqrt_mod_involved(cv, q, verbose_level - 2);
	if (f_vv) {
		cout << "sqrt(cv) = " << s << endl;
		}


	for (i = 0; i < 16; i++) {
		mtxG[i] = F->mult(mtxD[i], s);
		}

	if (f_vv) {
		cout << "mtxG = s * mtxD:" << endl;
		Orbiter->Int_vec->print_integer_matrix_width(cout,
				mtxG, 4, 4, 4, F->log10_of_q);
		}


	int mtxGt[16];
	F->Linear_algebra->transpose_matrix(mtxG, mtxGt, 4, 4);
	F->Linear_algebra->mult_matrix_matrix(mtxGt, mtxGram, mtxE, 4, 4, 4,
			0 /* verbose_level */);
	F->Linear_algebra->mult_matrix_matrix(mtxE, mtxG, mtxF, 4, 4, 4,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "G^transpose * Gram * G = " << endl;
		Orbiter->Int_vec->print_integer_matrix_width(cout, mtxF, 4, 4, 4, F->log10_of_q);
		}

	
	for (i = 0; i < 16; i++) {
		mtxD[i] = mtxG[i];
		}
	if (f_v) {
		cout << "fine_tune: the resulting matrix is" << endl;
		Orbiter->Int_vec->print_integer_matrix_width(cout, mtxD, 4, 4, 4, F->log10_of_q);
		}


	//int At2[4], As2[4], f_switch2;

	//cout << "calling O4_isomorphism_4to2" << endl;
	//O4_isomorphism_4to2(F, At2, As2, f_switch2, mtxG, verbose_level);
}


}}

