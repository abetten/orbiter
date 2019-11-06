// subprimitive.cpp
// 
// Anton Betten
//
// 5/2/2007
//
//


#include "orbiter.h"

using namespace std;


using namespace orbiter;

int Phi_of(int n, int verbose_level);
void formula_subprimitive(int d, int q,
	longinteger_object &Rdq, int &g, int verbose_level);
void formula(int d, int q, longinteger_object &Rdq, int verbose_level);
int subprimitive(int q, int h);
int period_of_sequence(int *v, int l);
void subexponent(int q, int Q, int h, int f, int j, int k, int &s, int &c);

int main(int argc, char **argv)
{
	int Q_max, H_max, q, h, p, e, i, g, phi_g, l, cmp;
	int *Q, *Rdq, *G, nb_primes = 0;
	longinteger_domain D;
	longinteger_object r2, r3, A, B;
	number_theory_domain NT;
	
	//formula(2, 64, r2, 1);

	Q_max = atoi(argv[1]);
	H_max = atoi(argv[2]);
	Q = new int[Q_max];
	Rdq = new int[Q_max * H_max];
	G = new int[Q_max * H_max];
	
	for (q = 2; q <= Q_max; q++) {
		if (NT.is_prime_power(q, p, e)) {
			Q[nb_primes] = q;
			
			cout << "studying prime power " << q << endl;
			for (h = 2; h <= H_max; h++) {
				//r1 = subprimitive(q, h);
				formula_subprimitive(h, q, r3, g, 1);
				phi_g = Phi_of(g, 1);
				formula(h, q, r2, 1);
				//cout << "The experiment gives " << r1 << endl;
				cout << "g= " << g << endl;
				cout << "\\Phi(g)= " << phi_g << endl;
				cout << "The formula gives " << r2 << endl;
				cout << "#subprimitive " << r3 << endl;
				l = (phi_g * (q - 1)) / g;
				cout << "l=" << l << endl;
				A.create(l);
				D.mult(r2, A, B);
				cout << "#subprimitive from R_c(d,q)=" << B << endl;
				cmp = D.compare_unsigned(r3, B);
				if (cmp) {
					cout << "cmp=" << cmp << endl;
					exit(1);
					}
				
				//cout << h << " times it = " << h * r2 << endl;
#if 0
				if (r1 != r2 * h) {
					cout << "r1 != r2 * h" << endl;
					exit(1);
					}
#endif
				Rdq[nb_primes * H_max + h - 2] = r2.as_int();
				G[nb_primes * H_max + h - 2] = g;
				}
			nb_primes++;
			}
		}
	for (i = 0; i < nb_primes; i++) {
		cout << setw(10) << Q[i];
		for (h = 2; h <= H_max; h++) {
			cout << " & " << setw(10) << Rdq[i * H_max + h - 2]
				<< "_{" << setw(3) << G[i * H_max + h - 2] << "}";
			}
		cout << "\\\\" << endl;
		}
}

int Phi_of(int n, int verbose_level)
{
	int nb_primes, *primes, *exponents;
	int i, p, e;
	longinteger_domain D;
	longinteger_object N, R, A, B, C;
	
	N.create(n);
	D.factor(N, nb_primes, primes, exponents, verbose_level);
	R.create(1);
	for (i = 0; i < nb_primes; i++) {
		p = primes[i];
		e = exponents[i];
		A.create(p);
		D.power_int(A, e);
		cout << "p^e=" << A << endl;
		B.create(p);
		D.power_int(B, e - 1);
		cout << "p^{e-1}=" << A << endl;
		B.negate();
		D.add(A, B, C);
		cout << "p^e-p^{e-1}=" << C << endl;
		D.mult(R, C, A);
		A.assign_to(R);
		}
	return R.as_int();
}

void formula_subprimitive(int d, int q,
		longinteger_object &Rdq, int &g, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int theta_mod_qm1, p, e, i, rem;
	int nb_primes, *primes, *exponents;
	longinteger_domain D;
	longinteger_object Theta, M1, Qm1, A, B, C, R;
	number_theory_domain NT;
	
	if (f_v) {
		cout << "d=" << d << " q=" << q << endl;
		}
	Theta.create(q);
	M1.create(-1);
	Qm1.create(q-1);
	D.power_int(Theta, d);
	D.add(Theta, M1, A);
	cout << "q^d-1 = " << A << endl;
	D.integral_division(A, Qm1, Theta, C, 0);
	cout << "theta = " << Theta << endl;
	D.integral_division_by_int(Theta, q - 1, C, theta_mod_qm1);
	g = NT.gcd_lint(q - 1, theta_mod_qm1);
	cout << "g = " << g << endl;
	D.factor(Theta, nb_primes, primes, exponents, verbose_level);
	if (f_v) {
		cout << "theta = " << Theta << endl;
		NT.print_factorization(nb_primes, primes, exponents);
		cout << endl;
		}
	R.create(1);
	for (i = 0; i < nb_primes; i++) {
		p = primes[i];
		e = exponents[i];
		if (f_v) {
			cout << "p=" << p << " e=" << e << endl;
			}
			//r = r * (i_power_j(p, e) - i_power_j(p, e - 1));
			A.create(p);
			D.power_int(A, e);
			cout << "p^e=" << A << endl;
			B.create(p);
			D.power_int(B, e - 1);
			cout << "p^{e-1}=" << A << endl;
			B.negate();
			D.add(A, B, C);
			cout << "p^e-p^{e-1}=" << C << endl;
			D.mult(R, C, A);
			A.assign_to(R);
			cout << "R=" << R << endl;
		}
	if (f_v) {
		cout << "\\Phi(theta)=" << R << endl;
		}
	D.mult(R, Qm1, B);
	B.assign_to(R);
	if (f_v) {
		cout << "(q-1)\\Phi(theta)=" << R << endl;
		}
	D.integral_division_by_int(R, d, A, rem);
	if (rem) {
		cout << "R not divisible by d" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "R/d=" << A << endl;
		}
	A.assign_to(Rdq);
}

void formula(int d, int q, longinteger_object &Rdq, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int theta_mod_qm1, g, p, e, i, rem;
	int nb_primes, *primes, *exponents;
	longinteger_domain D;
	longinteger_object Theta, M1, Qm1, A, B, C, R;
	number_theory_domain NT;
	
	if (f_v) {
		cout << "d=" << d << " q=" << q << endl;
		}
	Theta.create(q);
	M1.create(-1);
	Qm1.create(q-1);
	D.power_int(Theta, d);
	D.add(Theta, M1, A);
	cout << "q^d-1 = " << A << endl;
	D.integral_division(A, Qm1, Theta, C, 0);
	cout << "theta = " << Theta << endl;
	D.integral_division_by_int(Theta, 
		q - 1, C, theta_mod_qm1);
	g = NT.gcd_lint(q - 1, theta_mod_qm1);
	cout << "g = " << g << endl;
	D.factor(Theta, nb_primes, primes, exponents, verbose_level);
	if (f_v) {
		cout << "theta = " << Theta << endl;
		NT.print_factorization(nb_primes, primes, exponents);
		cout << endl;
		}
	R.create(1);
	for (i = 0; i < nb_primes; i++) {
		p = primes[i];
		e = exponents[i];
		if (f_v) {
			cout << "p=" << p << " e=" << e << endl;
			}
		if (((q - 1) % p) == 0) {
			A.create(p);
			D.power_int(A, e);
			D.mult(R, A, B);
			B.assign_to(R);
			}
		else {
			//r = r * (i_power_j(p, e) - i_power_j(p, e - 1));
			A.create(p);
			D.power_int(A, e);
			cout << "p^e=" << A << endl;
			B.create(p);
			D.power_int(B, e - 1);
			cout << "p^{e-1}=" << A << endl;
			B.negate();
			D.add(A, B, C);
			cout << "p^e-p^{e-1}=" << C << endl;
			D.mult(R, C, A);
			A.assign_to(R);
			cout << "R=" << R << endl;
			}
		}
	if (f_v) {
		cout << "R=" << R << endl;
		}
	D.integral_division_by_int(R, d, A, rem);
	if (rem) {
		cout << "R not divisible by d" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "R/d=" << A << endl;
		}
	A.assign_to(Rdq);
}

int subprimitive(int q, int h)
{
	int Q, f, i, j, k, s, c, l, r;
	number_theory_domain NT;
	
	Q = NT.i_power_j(q, h);
	f = (Q - 1) / (q - 1);
	cout << "q=" << q << " h=" << h << endl;
	//cout << " Q=" << Q << " f=" << f << endl;
	int *S,*C,*SM,*CM;
	
	
	S = new int[f * (q - 1)];
	C = new int[f * (q - 1)];
	SM = new int[Q];
	CM = new int[q - 1];
	for (k = 0; k < Q; k++)
		SM[k] = 0;
	for (k = 0; k < q - 1; k++)
		CM[k] = 0;
	
	for (k = 0; k < f; k++) {
		for (j = 0; j < q - 1; j++) {
			subexponent(q, Q, h, f, j, k, s, c);
			S[k * (q - 1) + j] = s;
			if (s >= Q) {
				cout << "s=" << s << " >= Q=" << Q << endl;
				exit(1);
				}
			SM[s]++;
			if (s == f) {
				if (c >= q - 1) {
					cout << "c=" << c << " >= q-1=" << q-1 << endl;
					exit(1);
					}
				CM[c]++;
				C[k * (q - 1) + j] = c;
				}
			else {
				C[k * (q - 1) + j] = -1;
				}
			}
		}
#if 0
	cout << "subexponents mod " << q - 1 << " :" << endl;
	print_integer_matrix_width(cout, S, f, q - 1, q - 1, 2);
	cout << "subexponents mod " << f << " :" << endl;
	print_integer_matrix_width(cout, S, q - 1, f, f, 2);
	cout << "integral elements:" << endl;
	print_integer_matrix_width(cout, C, f, q - 1, q - 1, 2);
	cout << "integral elements mod " << f << " :" << endl;
	print_integer_matrix_width(cout, C, q - 1, f, f, 2);

	cout << "multiplicities SM:" << endl;
	for (i = 0; i < Q; i++) {
		if (SM[i]) {
			cout << i << " : " << SM[i] << endl;
			}
		}
#endif
	cout << f << "^" << SM[f] << endl;
	cout << "multiplicities CM:" << endl;
	print_integer_matrix_width(cout, CM, 1, q - 1, q - 1, 2);
	l = period_of_sequence(CM, q - 1);
	cout << "period " << l << endl;
	for (i = 0; i < q - 1; i++) {
		if (CM[i]) {
			r = CM[i];
			}
		}
	
	//cout << "delete S" << endl;
	delete [] S;
	//cout << "delete C" << endl;
	delete [] C;
	//cout << "delete SM" << endl;
	delete [] SM;
	//cout << "delete CM" << endl;
	delete [] CM;
	return r;
}

int period_of_sequence(int *v, int l)
{
	int p, i, j;
	
	for (p = 1; p < l; p++) {
		// test if period p;
		for (i = 0; i < l; i++) {
			j = i + p;
			if (j < l) {
				if (v[i] != v[j])
					break;
				}
			}
		if (i == l) {
			return p;
			}
		}
	return l;
}

void subexponent(int q, int Q, int h, int f, int j, int k, int &s, int &c)
{
	int a, g;
	number_theory_domain NT;
	
	a = j + k * (q - 1);
	g = NT.gcd_lint(a, f);
	s = f / g;
	c = a / g;
	c = c % (q - 1);
#if 0
	for (s = 1; TRUE; s++) {
		b = a * s;
		if ((b % f) == 0) {
			c = b / f;
			c = c % (q - 1);
			return;
			}
		}
#endif
}

