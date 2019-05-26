// number_theory_domain.cpp
//
// Anton Betten
// April 3, 2003

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


number_theory_domain::number_theory_domain()
{

}

number_theory_domain::~number_theory_domain()
{

}


int number_theory_domain::power_mod(int a, int n, int p)
{
	longinteger_domain D;
	longinteger_object A, N, M;
	
	A.create(a);
	N.create(n);
	M.create(p);
	D.power_longint_mod(A, N, M, 0 /* verbose_level */);
	return A.as_int();
}

int number_theory_domain::inverse_mod(int a, int p)
{
	longinteger_domain D;
	longinteger_object A, B, U, V, G;
	int u;
	
	A.create(a);
	B.create(p);
	D.extended_gcd(A,B, G, U, V, 0 /* verbose_level */);
	u = U.as_int();
	while (u < 0) {
		u += p;
		}
	return u;
}

int number_theory_domain::mult_mod(int a, int b, int p)
{
	longinteger_domain D;
	longinteger_object A, B, C, P;
	
	A.create(a);
	B.create(b);
	P.create(p);
	D.mult_mod(A, B, C, P, 0 /* verbose_level */);
	return C.as_int();
}

int number_theory_domain::add_mod(int a, int b, int p)
{
	longinteger_domain D;
	longinteger_object A, B, C, P, Q;
	int r;
	
	A.create(a);
	B.create(b);
	P.create(p);
	D.add(A, B, C);
	D.integral_division_by_int(C, 
		p, Q, r);
	return r;
}

int number_theory_domain::int_abs(int a)
{
	if (a < 0) {
		return -a;
		}
	else {
		return a;
		}
}

int number_theory_domain::irem(int a, int m)
{
	int b;
	
	if (a < 0) {
		b = irem(-a, m);
		return (m - b) % m;
		}
	return a % m;
}

int number_theory_domain::gcd_int(int m, int n)
{
#if 0
	longinteger_domain D;
	longinteger_object M, N, G, U, V;


	M.create(m);
	N.create(n);
	D.extended_gcd(M, N, G, U, V, 0);
	return G.as_int();
#else
	int r, s;
	
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
	longinteger_domain D;
	longinteger_object M, N, G, U, V;


	M.create(m);
	N.create(n);
	D.extended_gcd(M, N, G, U, V, 0);
	g = G.as_int();
	u = U.as_int();
	v = V.as_int();
}

int number_theory_domain::i_power_j_safe(int i, int j)
{
	longinteger_domain D;

	longinteger_object a, b, c;
	int res;

	a.create(i);
	D.power_int(a, j);
	res = a.as_int();
	b.create(res);
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
	longinteger_domain D;

	longinteger_object a, b, c;
	long int res;

	if (f_v) {
		cout << "number_theory_domain::i_power_j_lint_safe "
				"i=" << i << " j=" << j << endl;
	}
	a.create(i);
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
	b.create(res);
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

int number_theory_domain::order_mod_p(int a, int p)
//Computes the order of $a$ mod $p$, i.~e. the smallest $k$ 
//s.~th. $a^k \equiv 1$ mod $p$.
{
	int o, b;
	
	if (a < 0) {
		cout << "order_mod_p a < 0" << endl;
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

int number_theory_domain::is_strict_prime_power(int q)
// assuming that q is a prime power, this fuction tests 
// whether or not q is a srict prime power
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

int number_theory_domain::sp_ge(int n, int p_min)
// Computes the smalles prime dividing $n$ 
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

int number_theory_domain::primitive_root(int p, int verbose_level)
// Computes a primitive element for $\bbZ_p$, i.~e. an integer $k$ 
// with $2 \le k \le p - 1$ s.~th. the order of $k$ mod $p$ is $p-1$.
{
	int f_v = (verbose_level >= 1);
	int i, o;

	if (p < 2) {
		cout << "primitive_root: p < 2" << endl;
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
	cout << "no primitive root found" << endl;
	exit(1);
}

int number_theory_domain::Legendre(int a, int p, int verbose_level)
// Computes the Legendre symbol $\left( \frac{a}{p} \right)$.
{
	return Jacobi(a, p, verbose_level);
}

int number_theory_domain::Jacobi(int a, int m, int verbose_level)
//Computes the Jacobi symbol $\left( \frac{a}{m} \right)$.
{
	int f_v = (verbose_level >= 1);
	int a1, m1, ord2, r1;
	int g;
	int f_negative = FALSE;
	int t, t1, t2;
	
	if (f_v) {
		cout << "Jacobi(" << a << ", " << m << ")" << endl;
		}
	a1 = a;
	m1 = m;
	r1 = 1;
	g = gcd_int(a1, m1);
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

int number_theory_domain::Jacobi_with_key_in_latex(ostream &ost,
		int a, int m, int verbose_level)
//Computes the Jacobi symbol $\left( \frac{a}{m} \right)$.
{
	int f_v = (verbose_level >= 1);
	int a1, m1, ord2, r1;
	int g;
	int f_negative = FALSE;
	int t, t1, t2;
	
	if (f_v) {
		cout << "Jacobi(" << a << ", " << m << ")" << endl;
		}
	a1 = a;
	m1 = m;
	r1 = 1;
	g = gcd_int(a1, m1);
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
		if (a1 % m1 < a1) {

			ost << "=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
				}
			ost << " \\Big( \\frac{" << a1 % m1 << " }{ "
					<< m1 << "}\\Big)\\\\" << endl;

			}

		a1 = a1 % m1;

		if (f_v) {
			cout << "Jacobi = " << r1 << " * Jacobi("
					<< a1 << ", " << m1 << ")" << endl;
			}
		ord2 = ny2(a1, a1);
		
		/* a1 jetzt immer noch != 0 */
		if (f_negative) {

			ost << "=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
				}
			ost << "\\Big( \\frac{-1 }{ " << m1
					<< "}\\Big) \\cdot \\Big( \\frac{"
					<< a1 * i_power_j(2, ord2) << " }{ "
					<< m1 << "}\\Big)\\\\" << endl;
			ost << "=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
				}
			ost << "(-1)^{\\frac{" << m1
					<< "-1}{2}} \\cdot \\Big( \\frac{"
					<< a1 * i_power_j(2, ord2) << " }{ "
					<< m1 << "}\\Big)\\\\" << endl;



			t = (m1 - 1) >> 1; /* t := (m1 - 1) / 2 */
			/* Ranmultiplizieren von (-1) hoch t an r1: */
			if (t % 2)
				r1 = -r1; /* Beachte ABS(r1) == 1 */
			/* und a1 wieder positiv machen: */
			a1 = -a1;

			ost << "=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
				}
			ost << " \\Big( \\frac{"
					<< a1 * i_power_j(2, ord2)
			<< " }{ " << m1 << "}\\Big)\\\\" << endl;


			}
		if (ord2 % 2) {
			/* tue nur dann etwas, wenn ord2 ungerade */
			// t = (m1 * m1 - 1) >> 3; /* t = (m1 * m1 - 1) / 8 */
			/* Ranmultiplizieren von (-1) hoch t an r1: */

			if (ord2 > 1) {
				ost << "=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
					}
				ost << "\\Big( \\frac{2}{ " << m1
						<< "}\\Big)^{" << ord2
						<< "} \\cdot \\Big( \\frac{" << a1
						<< " }{ " << m1 << "}\\Big)\\\\" << endl;
				ost << "=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
					}
				ost << "\\Big( (-1)^{\\frac{" << m1
						<< "^2-1}{8}} \\Big)^{" << ord2
						<< "} \\cdot \\Big( \\frac{" << a1 << " }{ "
						<< m1 << "}\\Big)\\\\" << endl;
				}
			else {
				ost << "=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
					}
				ost << "\\Big( \\frac{2}{ " << m1
						<< "}\\Big) \\cdot \\Big( \\frac{" << a1
						<< " }{ " << m1 << "}\\Big)\\\\" << endl;
				ost << "=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
					}
				ost << "(-1)^{\\frac{" << m1
						<< "^2-1}{8}} \\cdot \\Big( \\frac{" << a1
						<< " }{ " << m1 << "}\\Big)\\\\" << endl;
				}

			if (m1 % 8 == 3 || m1 % 8 == 5)
				r1 = -r1; /* Beachte ABS(r1) == 1L */

			ost << "=";
			if (r1 == -1) {
				ost << "(-1) \\cdot ";
				}
			ost << "\\Big( \\frac{" << a1 << " }{ " << m1
					<< "}\\Big)\\\\" << endl;


			}
		else {
			if (ord2) {
				ost << "=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
					}
				ost << "\\Big( \\frac{2}{ " << m1 << "}\\Big)^{"
						<< ord2 << "} \\cdot \\Big( \\frac{" << a1
						<< " }{ " << m1 << "}\\Big)\\\\" << endl;

				ost << "=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
					}
				ost << "\\Big( (-1)^{\\frac{" << m1
						<< "^2-1}{8}} \\Big)^{" << ord2
						<< "} \\cdot \\Big( \\frac{" << a1 << " }{ "
						<< m1 << "}\\Big)\\\\" << endl;
				ost << "=";
				if (r1 == -1) {
					ost << "(-1) \\cdot ";
					}
				ost << "\\Big( \\frac{" << a1 << " }{ " << m1
						<< "}\\Big)\\\\" << endl;
				}
			}
		if (ABS(a1) <= 1)
			break;


		t = m1;
		m1 = a1;
		a1 = t;


		ost << "=";
		if (r1 == -1) {
			ost << "(-1) \\cdot ";
			}
		ost << "\\Big( \\frac{" << a1 << " }{ " << m1
				<< "}\\Big) \\cdot (-1)^{\\frac{" << m1
				<< "-1}{2} \\cdot \\frac{" << a1
				<< " - 1}{2}}\\\\" << endl;


		/* Reziprozitaet: */
		t1 = (m1 - 1) >> 1; /* t1 = (m1 - 1) / 2 */
		t2 = (a1 - 1) >> 1; /* t1 = (a1 - 1) / 2 */
		if ((t1 % 2) && (t2 % 2)) /* t1 und t2 ungerade */
			r1 = -r1; /* Beachte ABS(r1) == 1 */


		ost << "=";
		if (r1 == -1) {
			ost << "(-1) \\cdot ";
			}
		ost << "\\Big( \\frac{" << a1 << " }{ " << m1
				<< "}\\Big)\\\\" << endl;

		if (f_v) {
			cout << "Jacobi() = " << r1 << " * Jacobi(" << a1
					<< ", " << m1 << ")" << endl;
			}
		}
	if (a1 == 1) {
		return r1;
		}
	if (a1 <= 0) {
		cout << "Jacobi() a1 == -1 || a1 == 0" << endl;
		exit(1);
		}
	cout << "Jacobi() wrong termination" << endl;
	exit(1);
}

int number_theory_domain::gcd_with_key_in_latex(ostream &ost,
		int a, int b, int f_key, int verbose_level)
//Computes gcd(a,b)
{
	int f_v = (verbose_level >= 1);
	int a1, b1, q1, r1;

	if (f_v) {
		cout << "gcd_with_key_in_latex a=" << a << ", b=" << b << ":" << endl;
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
			cout << "gcd_with_key_in_latex "
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
	ost << "= " << b1 << "\\\\" << endl;
	if (f_v) {
		cout << "gcd_with_key_in_latex done" << endl;
	}
	return b1;
}

int number_theory_domain::ny2(int x, int &x1)
//returns $n = \ny_2(x).$ 
//Computes $x1 := \frac{x}{2^n}$. 
{
	int xx = x;
	int n1;
	int f_negative;
	
	n1 = 0;
	if (xx == 0) {
		cout << "ny2 x == 0" << endl;
		exit(1);
		}
	if (xx < 0) {
		xx = -xx;
		f_negative = TRUE;
		}
	else
		f_negative = FALSE;
	while (TRUE) {
		// while xx congruent 0 mod 2:
		if (ODD(xx))
			break;
		n1++;
		xx >>= 1;
		}
	if (f_negative)
		xx = -xx;
	x1 = xx;
	return n1;
}

int number_theory_domain::ny_p(int n, int p)
//Returns $\nu_p(n),$ the integer $k$ with $n=p^k n'$ and $p \nmid n'$.
{
	int ny_p;
	
	if (n == 0) {
		cout << "ny_p n == 0" << endl;
		exit(1);
		}
	if (n < 0)
		n = -n;
	ny_p = 0;
	while (n != 1) {
		if ((n % p) != 0)
			break;
		n /= p;
		ny_p++;
		}
	return ny_p;
}

int number_theory_domain::sqrt_mod_simple(int a, int p)
// solves x^2 = a mod p. Returns x
{
	int a1, x;
	
	a1 = a % p;
	for (x = 0; x < p; x++) {
		if ((x * x) % p == a1)
			return x;
		}
	cout << "sqrt_mod_simple a not a quadratic residue" << endl;
	cout << "a = " << a << " p=" << p << endl;
	exit(1);
}

void number_theory_domain::print_factorization(int nb_primes, int *primes, int *exponents)
{
	int i;
	
	for (i = 0; i < nb_primes; i++) {
		cout << primes[i];
		if (exponents[i] > 1)
			cout << "^" << exponents[i];
		if (i < nb_primes - 1)
			cout << " * ";
		}
}

void number_theory_domain::print_longfactorization(int nb_primes,
		longinteger_object *primes, int *exponents)
{
	int i;
	
	for (i = 0; i < nb_primes; i++) {
		cout << primes[i];
		if (exponents[i] > 1)
			cout << "^" << exponents[i];
		if (i < nb_primes - 1)
			cout << " * ";
		}
}

int number_theory_domain::euler_function(int n)
//Computes Eulers $\varphi$-function for $n$.
//Uses the prime factorization of $n$. before: eulerfunc
{
	int *primes;
	int *exponents;
	int len;
	int i, k, p1, e1;
			
	len = factor_int(n, primes, exponents);
	
	k = 1;
	for (i = 0; i < len; i++) {
		p1 = primes[i];
		e1 = exponents[i];
		if (e1 > 1) {
			k *= i_power_j(p1, e1 - 1);
			}
		k *= (p1 - 1);
		}
	FREE_int(primes);
	FREE_int(exponents);
	return k;
}

void number_theory_domain::int_add_fractions(int at, int ab,
		int bt, int bb, int &ct, int &cb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int g, a1, b1;
	
	if (at == 0) {
		ct = bt;
		cb = bb;
		}
	else if (bt == 0) {
		ct = at;
		cb = ab;
		}
	else {
		g = gcd_int(ab, bb);
		a1 = ab / g;
		b1 = bb / g;
		cb = ab * b1;
		ct = at * b1 + bt * a1;
		}
	if (cb < 0) {
		cb *= -1;
		ct *= -1;
		}
	g = gcd_int(int_abs(ct), cb);
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
	int g;
	
	if (at == 0) {
		ct = 0;
		cb = 1;
		}
	else if (bt == 0) {
		ct = 0;
		cb = 1;
		}
	else {
		g = gcd_int(at, ab);
		if (g != 1 && g != -1) {
			at /= g;
			ab /= g;
			}
		g = gcd_int(bt, bb);
		if (g != 1 && g != -1) {
			bt /= g;
			bb /= g;
			}
		g = gcd_int(at, bb);
		if (g != 1 && g != -1) {
			at /= g;
			bb /= g;
			}
		g = gcd_int(bt, ab);
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
	g = gcd_int(int_abs(ct), cb);
	if (g > 1) {
		ct /= g;
		cb /= g;
		}
}

}
}


