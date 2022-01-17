/*
 * algebra_global.cpp
 *
 *  Created on: Nov 25, 2019
 *      Author: betten
 */






#include "foundations.h"

using namespace std;

namespace orbiter {
namespace foundations {
namespace algebra {



void algebra_global::count_subprimitive(int Q_max, int H_max)
{
	int q, h, p, e, i, g, phi_g, l, cmp;
	int *Q, *Rdq, *G, nb_primes = 0;
	longinteger_domain D;
	longinteger_object r2, r3, A, B;
	number_theory_domain NT;

	//formula(2, 64, r2, 1);

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
				phi_g = NT.eulers_totient_function(g, 1);
				formula(h, q, r2, 1);
				//cout << "The experiment gives " << r1 << endl;
				cout << "g= " << g << endl;
				cout << "\\Phi(g)= " << phi_g << endl;
				cout << "The formula gives " << r2 << endl;
				cout << "#subprimitive " << r3 << endl;
				l = (phi_g * (q - 1)) / g;
				cout << "l=" << l << endl;
				A.create(l, __FILE__, __LINE__);
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



void algebra_global::formula_subprimitive(int d, int q,
		longinteger_object &Rdq, int &g, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int theta_mod_qm1, p, e, i, rem;
	int nb_primes, *primes, *exponents;
	longinteger_domain D;
	longinteger_object Theta, M1, Qm1, A, B, C, R;
	number_theory_domain NT;

	if (f_v) {
		cout << "algebra_global::formula_subprimitive d=" << d << " q=" << q << endl;
	}
	Theta.create(q, __FILE__, __LINE__);
	M1.create(-1, __FILE__, __LINE__);
	Qm1.create(q-1, __FILE__, __LINE__);
	D.power_int(Theta, d);
	D.add(Theta, M1, A);
	if (f_v) {
		cout << "q^d-1 = " << A << endl;
	}
	D.integral_division(A, Qm1, Theta, C, 0);
	if (f_v) {
		cout << "theta = " << Theta << endl;
	}
	D.integral_division_by_int(Theta, q - 1, C, theta_mod_qm1);
	g = NT.gcd_lint(q - 1, theta_mod_qm1);
	if (f_v) {
		cout << "g = " << g << endl;
	}
	D.factor(Theta, nb_primes, primes, exponents, verbose_level);
	if (f_v) {
		cout << "theta = " << Theta << endl;
		NT.print_factorization(nb_primes, primes, exponents);
		cout << endl;
	}
	R.create(1, __FILE__, __LINE__);
	for (i = 0; i < nb_primes; i++) {
		p = primes[i];
		e = exponents[i];
		if (f_v) {
			cout << "p=" << p << " e=" << e << endl;
		}
		//r = r * (i_power_j(p, e) - i_power_j(p, e - 1));
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
		if (f_v) {
			cout << "R=" << R << endl;
		}
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
	if (f_v) {
		cout << "algebra_global::formula_subprimitive done" << endl;
	}
}

void algebra_global::formula(int d, int q, longinteger_object &Rdq, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int theta_mod_qm1, g, p, e, i, rem;
	int nb_primes, *primes, *exponents;
	longinteger_domain D;
	longinteger_object Theta, M1, Qm1, A, B, C, R;
	number_theory_domain NT;

	if (f_v) {
		cout << "algebra_global::formula d=" << d << " q=" << q << endl;
	}
	Theta.create(q, __FILE__, __LINE__);
	M1.create(-1, __FILE__, __LINE__);
	Qm1.create(q - 1, __FILE__, __LINE__);
	D.power_int(Theta, d);
	D.add(Theta, M1, A);
	if (f_v) {
		cout << "q^d-1 = " << A << endl;
	}
	D.integral_division(A, Qm1, Theta, C, 0);
	if (f_v) {
		cout << "theta = " << Theta << endl;
	}
	D.integral_division_by_int(Theta,
		q - 1, C, theta_mod_qm1);
	g = NT.gcd_lint(q - 1, theta_mod_qm1);
	if (f_v) {
		cout << "g = " << g << endl;
	}
	D.factor(Theta, nb_primes, primes, exponents, verbose_level);
	if (f_v) {
		cout << "theta = " << Theta << endl;
		NT.print_factorization(nb_primes, primes, exponents);
		cout << endl;
	}
	R.create(1, __FILE__, __LINE__);
	for (i = 0; i < nb_primes; i++) {
		p = primes[i];
		e = exponents[i];
		if (f_v) {
			cout << "p=" << p << " e=" << e << endl;
		}
		if (((q - 1) % p) == 0) {
			A.create(p, __FILE__, __LINE__);
			D.power_int(A, e);
			D.mult(R, A, B);
			B.assign_to(R);
		}
		else {
			//r = r * (i_power_j(p, e) - i_power_j(p, e - 1));
			A.create(p, __FILE__, __LINE__);
			D.power_int(A, e);
			cout << "p^e=" << A << endl;
			B.create(p, __FILE__, __LINE__);
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

int algebra_global::subprimitive(int q, int h)
{
	int Q, f, i, j, k, s, c, l, r = 0;
	number_theory_domain NT;

	Q = NT.i_power_j(q, h);
	f = (Q - 1) / (q - 1);
	cout << "q=" << q << " h=" << h << endl;
	//cout << " Q=" << Q << " f=" << f << endl;
	int *S, *C, *SM, *CM;


	S = NEW_int(f * (q - 1));
	C = NEW_int(f * (q - 1));
	SM = NEW_int(Q);
	CM = NEW_int(q - 1);
	for (k = 0; k < Q; k++) {
		SM[k] = 0;
	}
	for (k = 0; k < q - 1; k++) {
		CM[k] = 0;
	}

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
	Orbiter->Int_vec->print_integer_matrix_width(cout, CM, 1, q - 1, q - 1, 2);
	l = period_of_sequence(CM, q - 1);
	cout << "period " << l << endl;
	for (i = 0; i < q - 1; i++) {
		if (CM[i]) {
			r = CM[i];
		}
	}

	//cout << "delete S" << endl;
	FREE_int(S);
	//cout << "delete C" << endl;
	FREE_int(C);
	//cout << "delete SM" << endl;
	FREE_int(SM);
	//cout << "delete CM" << endl;
	FREE_int(CM);
	return r;
}

int algebra_global::period_of_sequence(int *v, int l)
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

void algebra_global::subexponent(int q, int Q, int h, int f, int j, int k, int &s, int &c)
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


const char *algebra_global::plus_minus_string(int epsilon)
{
	if (epsilon == 1) {
		return "+";
	}
	if (epsilon == -1) {
		return "-";
	}
	if (epsilon == 0) {
		return "";
	}
	cout << "algebra_global::plus_minus_string epsilon=" << epsilon << endl;
	exit(1);
}

const char *algebra_global::plus_minus_letter(int epsilon)
{
	if (epsilon == 1) {
		return "p";
	}
	if (epsilon == -1) {
		return "m";
	}
	if (epsilon == 0) {
		return "";
	}
	cout << "algebra_global::plus_minus_letter epsilon=" << epsilon << endl;
	exit(1);
}




void algebra_global::display_all_PHG_elements(int n, int q)
{
	int *v = NEW_int(n + 1);
	int l;
	int i, j, a;
	finite_ring R;

	if (!R.f_chain_ring) {
		cout << "algebra_global::display_all_PHG_elements not a chain ring" << endl;
		exit(1);
	}
	R.init(q, 0);
	l = R.nb_PHG_elements(n);
	for (i = 0; i < l; i++) {
		R.PHG_element_unrank(v, 1, n + 1, i);
		cout << i << " : ";
		for (j = 0; j < n + 1; j++) {
			cout << v[j] << " ";
		}
		a = R.PHG_element_rank(v, 1, n + 1);
		cout << " : " << a << endl;
	}
	FREE_int(v);
}

void algebra_global::test_unipoly()
{
	finite_field GFp;
	int p = 2;
	unipoly_object m, a, b, c;
	unipoly_object elts[4];
	int i, j;
	int verbose_level = 0;

	GFp.finite_field_init(p, FALSE /* f_without_tables */, verbose_level);
	unipoly_domain FX(&GFp);

	FX.create_object_by_rank(m, 7, __FILE__, __LINE__, 0);
	FX.create_object_by_rank(a, 5, __FILE__, __LINE__, 0);
	FX.create_object_by_rank(b, 55, __FILE__, __LINE__, 0);
	FX.print_object(a, cout); cout << endl;
	FX.print_object(b, cout); cout << endl;

	unipoly_domain Fq(&GFp, m, verbose_level);
	Fq.create_object_by_rank(c, 2, __FILE__, __LINE__, 0);
	for (i = 0; i < 4; i++) {
		Fq.create_object_by_rank(elts[i], i, __FILE__, __LINE__, 0);
		cout << "elt_" << i << " = ";
		Fq.print_object(elts[i], cout); cout << endl;
	}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			Fq.print_object(elts[i], cout);
			cout << " * ";
			Fq.print_object(elts[j], cout);
			cout << " = ";
			Fq.mult(elts[i], elts[j], c, verbose_level);
			Fq.print_object(c, cout); cout << endl;

			FX.mult(elts[i], elts[j], a, verbose_level);
			FX.print_object(a, cout); cout << endl;
		}
	}

}

void algebra_global::test_unipoly2()
{
	finite_field Fq;
	int q = 4, p = 2, i;
	int verbose_level = 0;

	Fq.finite_field_init(q, FALSE /* f_without_tables */, verbose_level);
	unipoly_domain FX(&Fq);

	unipoly_object a;

	FX.create_object_by_rank(a, 0, __FILE__, __LINE__, 0);
	for (i = 1; i < q; i++) {
		FX.minimum_polynomial(a, i, p, TRUE);
		//cout << "minpoly_" << i << " = ";
		//FX.print_object(a, cout); cout << endl;
		}

}

int algebra_global::is_diagonal_matrix(int *A, int n)
{
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				continue;
			}
			else {
				if (A[i * n + j]) {
					return FALSE;
				}
			}
		}
	}
	return TRUE;
}




void algebra_global::test_longinteger()
{
	longinteger_domain D;
	int x[] = {15, 14, 12, 8};
	longinteger_object a, b, q, r;
	int verbose_level = 0;

	D.multiply_up(a, x, 4, verbose_level);
	cout << "a=" << a << endl;
	b.create(2, __FILE__, __LINE__);
	while (!a.is_zero()) {
		D.integral_division(a, b, q, r, verbose_level);
		//cout << a << " = " << q << " * " << b << " + " << r << endl;
		cout << r << endl;
		q.assign_to(a);
	}

	D.multiply_up(a, x, 4, verbose_level);
	cout << "a=" << a << endl;

	int *rep, len;
	D.base_b_representation(a, 2, rep, len);
	b.create_from_base_b_representation(2, rep, len);
	cout << "b=" << b << endl;
	FREE_int(rep);
}

void algebra_global::test_longinteger2()
{
	longinteger_domain D;
	longinteger_object a, b, c, d, e;
	int r;
	int verbose_level = 0;

	a.create_from_base_10_string("562949953421311", verbose_level);
	D.integral_division_by_int(a, 127, b, r);
	cout << a << " = " << b << " * 127 + " << r << endl;
	c.create_from_base_10_string("270549121", verbose_level);
	D.integral_division(b, c, d, e, verbose_level);
	cout << b << " = " << d << " * " << c << " + " << e << endl;
}

void algebra_global::test_longinteger3()
{
	int i, j;
	combinatorics::combinatorics_domain D;
	longinteger_object a, b, c, d, e;

	for (i = 0; i < 10; i++) {
		for (j = 0; j < 10; j++) {
			D.binomial(a, i, j, FALSE);
			a.print(cout);
			cout << " ";
		}
		cout << endl;
	}
}

void algebra_global::test_longinteger4()
{
	int n = 6, q = 2, k, x, d = 3;
	combinatorics::combinatorics_domain D;
	longinteger_object a;

	for (k = 0; k <= n; k++) {
		for (x = 0; x <= n; x++) {
			if (x > 0 && x < d) {
				continue;
			}
			if (q == 2 && EVEN(d) && ODD(x)) {
				continue;
			}
			D.krawtchouk(a, n, q, k, x);
			a.print(cout);
			cout << " ";
		}
		cout << endl;
	}
}

void algebra_global::test_longinteger5()
{
	longinteger_domain D;
	longinteger_object a, b, u, v, g;
	int verbose_level = 2;

	a.create(9548, __FILE__, __LINE__);
	b.create(254774, __FILE__, __LINE__);
	D.extended_gcd(a, b, g, u, v, verbose_level);

	g.print(cout);
	cout << " = ";
	u.print(cout);
	cout << " * ";
	a.print(cout);
	cout << " + ";
	v.print(cout);
	cout << " * ";
	b.print(cout);
	cout << endl;

}

void algebra_global::test_longinteger6()
{
	int verbose_level = 2;
	longinteger_domain D;
	longinteger_object a, b;

	a.create(7411, __FILE__, __LINE__);
	b.create(9283, __FILE__, __LINE__);
	D.jacobi(a, b, verbose_level);


}

void algebra_global::test_longinteger7()
{
	longinteger_domain D;
	longinteger_object a, b;
	int i, j;
	int mult[15];

	a.create(15, __FILE__, __LINE__);
	for (i = 0; i < 15; i++) {
		mult[i] = 0;
	}
	for (i = 0; i < 10000; i++) {
		D.random_number_less_than_n(a, b);
		j = b.as_int();
		mult[j]++;
		//cout << b << endl;
	}
	for (i = 0; i < 15; i++) {
		cout << i << " : " << mult[i] << endl;
	}

}

void algebra_global::test_longinteger8()
{
	int verbose_level = 2;
	cryptography::cryptography_domain Crypto;
	longinteger_object a, b, one;
	int nb_solovay_strassen_tests = 100;
	int f_miller_rabin_test = TRUE;

	one.create(1, __FILE__, __LINE__);
	a.create(197659, __FILE__, __LINE__);
	Crypto.find_probable_prime_above(a, nb_solovay_strassen_tests,
		f_miller_rabin_test, verbose_level);
}

void algebra_global::longinteger_collect_setup(int &nb_agos,
		longinteger_object *&agos, int *&multiplicities)
{
	nb_agos = 0;
	agos = NULL;
	multiplicities = NULL;
}

void algebra_global::longinteger_collect_free(int &nb_agos,
		longinteger_object *&agos, int *&multiplicities)
{
	if (nb_agos) {
		FREE_OBJECTS(agos);
		FREE_int(multiplicities);
	}
}

void algebra_global::longinteger_collect_add(int &nb_agos,
		longinteger_object *&agos, int *&multiplicities,
		longinteger_object &ago)
{
	int j, c, h, f_added;
	longinteger_object *tmp_agos;
	int *tmp_multiplicities;
	longinteger_domain D;

	f_added = FALSE;
	for (j = 0; j < nb_agos; j++) {
		c = D.compare_unsigned(ago, agos[j]);
		//cout << "comparing " << ago << " with "
		//<< agos[j] << " yields " << c << endl;
		if (c >= 0) {
			if (c == 0) {
				multiplicities[j]++;
			}
			else {
				tmp_agos = agos;
				tmp_multiplicities = multiplicities;
				agos = NEW_OBJECTS(longinteger_object, nb_agos + 1);
				multiplicities = NEW_int(nb_agos + 1);
				for (h = 0; h < j; h++) {
					tmp_agos[h].swap_with(agos[h]);
					multiplicities[h] = tmp_multiplicities[h];
				}
				ago.swap_with(agos[j]);
				multiplicities[j] = 1;
				for (h = j; h < nb_agos; h++) {
					tmp_agos[h].swap_with(agos[h + 1]);
					multiplicities[h + 1] = tmp_multiplicities[h];
				}
				nb_agos++;
				if (tmp_agos) {
					FREE_OBJECTS(tmp_agos);
					FREE_int(tmp_multiplicities);
				}
			}
			f_added = TRUE;
			break;
		}
	}
	if (!f_added) {
		// add at the end (including the case that the list is empty)
		tmp_agos = agos;
		tmp_multiplicities = multiplicities;
		agos = NEW_OBJECTS(longinteger_object, nb_agos + 1);
		multiplicities = NEW_int(nb_agos + 1);
		for (h = 0; h < nb_agos; h++) {
			tmp_agos[h].swap_with(agos[h]);
			multiplicities[h] = tmp_multiplicities[h];
		}
		ago.swap_with(agos[nb_agos]);
		multiplicities[nb_agos] = 1;
		nb_agos++;
		if (tmp_agos) {
			FREE_OBJECTS(tmp_agos);
			FREE_int(tmp_multiplicities);
		}
	}
}

void algebra_global::longinteger_collect_print(ostream &ost,
		int &nb_agos, longinteger_object *&agos, int *&multiplicities)
{
	int j;

	ost << "(";
	for (j = 0; j < nb_agos; j++) {
		ost << agos[j];
		if (multiplicities[j] == 1) {
		}
		else if (multiplicities[j] >= 10) {
			ost << "^{" << multiplicities[j] << "}";
		}
		else  {
			ost << "^" << multiplicities[j];
		}
		if (j < nb_agos - 1) {
			ost << ", ";
		}
	}
	ost << ")" << endl;
}


void algebra_global::do_equivalence_class_of_fractions(int N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h, a, b, ap, bp, g;
	number_theory_domain NT;
	file_io Fio;

	if (f_v) {
		cout << "algebra_global::do_equivalence_class_of_fractions" << endl;
	}

	int *Pairs;
	int *Table;
	int length;

	Table = NEW_int(N * N);
	Pairs = NEW_int(N * N);
	length = 0;

	for (i = 0; i < N; i++) {
		a = i + 1;
		for (j = 0; j < N; j++) {
			b = j + 1;
			g = NT.gcd_lint(a, b);
			ap = a / g;
			bp = b / g;
			for (h = 0; h < length; h++) {
				if (Pairs[h * 2 + 0] == ap && Pairs[h * 2 + 1] == bp) {
					Table[i * N + j] = h;
					break;
				}
			}
			if (h == length) {
				Pairs[h * 2 + 0] = ap;
				Pairs[h * 2 + 1] = bp;
				Table[i * N + j] = h;
				length++;
			}
		}
	}

	char str[1000];
	string fname;

	sprintf(str, "table_fractions_N%d.csv", N);
	fname.assign(str);
	Fio.int_matrix_write_csv(fname, Table, N, N);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	FREE_int(Table);
	FREE_int(Pairs);

	if (f_v) {
		cout << "algebra_global::do_equivalence_class_of_fractions done" << endl;
	}
}





void algebra_global::order_of_q_mod_n(int q, int n_min, int n_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::order_of_q_mod_n" << endl;
	}
	{
		char str[1000];
		string fname;

		snprintf(str, 1000, "order_of_q_mod_n_q%d_%d_%d.csv", q, n_min, n_max);
		fname.assign(str);


		{
			ofstream ost(fname);
			number_theory_domain NT;

			if (f_v) {
				cout << "algebra_global::order_of_q_mod_n writing csv file" << endl;
			}
			//report(ost, verbose_level);

			int n, row;
			long int o;

			row = 0;

			ost << "ROW,N,ORD,PHI,COF" << endl;
			for (n = n_min; n <= n_max; n++) {

				int g, phi, cof;

				g = NT.gcd_lint(q, n);
				if (g > 1) {
					continue;
				}

				ost << row << "," << n;
				cout << "computing n=" << n << " q=" << q << endl;

				o = NT.order_mod_p(q, n);
				phi = NT.euler_function(n);
				cof = phi / o;

				ost << "," << o;
				ost << "," << phi;
				ost << "," << cof;
				ost << endl;
				row++;
			}
			ost << "END" << endl;

			if (f_v) {
				cout << "algebra_global::order_of_q_mod_n writing csv file" << endl;
			}


		}
		file_io Fio;

		cout << "algebra_global::order_of_q_mod_n written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "algebra_global::order_of_q_mod_n done" << endl;
	}
}


void algebra_global::power_function_mod_n(int k, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::power_function_mod_n" << endl;
	}
	{
		char str[1000];
		string fname;

		snprintf(str, 1000, "power_function_k%d_n%d.csv", k, n);
		fname.assign(str);


		{
			ofstream ost(fname);
			number_theory_domain NT;

			if (f_v) {
				cout << "algebra_global::power_function_mod_n computing powers" << endl;
			}
			//report(ost, verbose_level);

			int row;
			long int a;
			long int b;

			row = 0;

			ost << "ROW,A,APOWK" << endl;
			for (a = 0; a < n; a++) {

				b = NT.power_mod(a, k, n);

				ost << row;
				ost << "," << a;
				ost << "," << b;
				ost << endl;
				row++;
			}
			ost << "END" << endl;

			if (f_v) {
				cout << "algebra_global::power_function_mod_n writing csv file" << endl;
			}


		}
		file_io Fio;

		cout << "algebra_global::power_function_mod_n written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "algebra_global::power_mod_interval_n done" << endl;
	}
}

void algebra_global::do_trace(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int s, t;
	int *T = NULL;
	int *T0 = NULL;
	int *T1 = NULL;
	int nb_T0 = 0;
	int nb_T1 = 0;


	if (f_v) {
		cout << "algebra_global::do_trace" << endl;
	}


	T = NEW_int(F->q);
	T0 = NEW_int(F->q);
	T1 = NEW_int(F->q);

	for (s = 0; s < F->q; s++) {

#if 0
		int s2, /*s3,*/ s4, s8, s2p1, s2p1t7, s2p1t6, s2p1t4, f;

		s2 = F->mult(s, s);
		s2p1 = F->add(s2, 1);
		s2p1t7 = F->power(s2p1, 7);
		s2p1t6 = F->power(s2p1, 6);
		s2p1t4 = F->power(s2p1, 4);
		//s3 = F->power(s, 3);
		s4 = F->power(s, 4);
		s8 = F->power(s, 8);

		f = F->add4(F->mult(s, s2p1t7), F->mult(s2, s2p1t6), F->mult(s4, s2p1t4), s8);

		//f = F->mult(top, F->inverse(bot));
		t = F->absolute_trace(f);
#endif

		t = F->absolute_trace(s);
		T[s] = t;
		if (t == 1) {
			T1[nb_T1++] = s;
		}
		else {
			T0[nb_T0++] = s;
		}
	}



	cout << "Trace 0:" << endl;
	Orbiter->Int_vec->print_fully(cout, T0, nb_T0);
	cout << endl;

	cout << "Trace 1:" << endl;
	Orbiter->Int_vec->print_fully(cout, T1, nb_T1);
	cout << endl;

	char str[1000];
	string fname_csv;
	file_io Fio;

	snprintf(str, 1000, "F_q%d_trace.csv", F->q);
	fname_csv.assign(str);
	Fio.int_matrix_write_csv(fname_csv, T, 1, F->q);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	snprintf(str, 1000, "F_q%d_trace_0.csv", F->q);
	fname_csv.assign(str);
	Fio.int_vec_write_csv(T0, nb_T0,
			fname_csv, "Trace_0");
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;

	snprintf(str, 1000, "F_q%d_trace_1.csv", F->q);
	fname_csv.assign(str);
	Fio.int_vec_write_csv(T1, nb_T1,
			fname_csv, "Trace_1");
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;

	FREE_int(T);
	FREE_int(T0);
	FREE_int(T1);

	if (f_v) {
		cout << "algebra_global::do_trace done" << endl;
	}
}

void algebra_global::do_norm(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int s, t;
	int *T0 = NULL;
	int *T1 = NULL;
	int nb_T0 = 0;
	int nb_T1 = 0;

	if (f_v) {
		cout << "algebra_global::do_norm" << endl;
	}

	T0 = NEW_int(F->q);
	T1 = NEW_int(F->q);

	for (s = 0; s < F->q; s++) {
		t = F->absolute_norm(s);
		if (t == 1) {
			T1[nb_T1++] = s;
		}
		else {
			T0[nb_T0++] = s;
		}
	}

	cout << "Norm 0:" << endl;
	Orbiter->Int_vec->print_fully(cout, T0, nb_T0);
	cout << endl;

	cout << "Norm 1:" << endl;
	Orbiter->Int_vec->print_fully(cout, T1, nb_T1);
	cout << endl;


	char str[1000];
	string fname_csv;
	file_io Fio;

	snprintf(str, 1000, "F_q%d_norm_0.csv", F->q);
	fname_csv.assign(str);
	Fio.int_vec_write_csv(T0, nb_T0,
			fname_csv, "Norm_0");
	cout << "written file " << fname_csv << " of size " << Fio.file_size(fname_csv) << endl;

	snprintf(str, 1000, "F_q%d_norm_1.csv", F->q);
	fname_csv.assign(str);
	Fio.int_vec_write_csv(T1, nb_T1,
			fname_csv, "Norm_1");
	cout << "written file " << fname_csv << " of size " << Fio.file_size(fname_csv) << endl;


	if (f_v) {
		cout << "algebra_global::do_norm done" << endl;
	}
}

void algebra_global::do_cheat_sheet_GF(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::do_cheat_sheet_GF q=" << F->q << endl;
	}

	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "%s.tex", F->label.c_str());
	snprintf(title, 1000, "Cheat Sheet $%s$", F->label_tex.c_str());
	//sprintf(author, "");
	author[0] = 0;



	F->addition_table_save_csv(verbose_level);

	F->multiplication_table_save_csv(verbose_level);

	F->addition_table_reordered_save_csv(verbose_level);

	F->multiplication_table_reordered_save_csv(verbose_level);


	{
		ofstream ost(fname);


		latex_interface L;


		L.head(ost, FALSE /* f_book*/, TRUE /* f_title */,
			title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
				TRUE /* f_12pt */,
				TRUE /* f_enlarged_page */,
				TRUE /* f_pagenumbers */,
				NULL /* extra_praeamble */);


		F->cheat_sheet(ost, verbose_level);

		F->cheat_sheet_main_table(ost, verbose_level);

		F->cheat_sheet_addition_table(ost, verbose_level);

		F->cheat_sheet_multiplication_table(ost, verbose_level);

		F->cheat_sheet_power_table(ost, TRUE, verbose_level);

		F->cheat_sheet_power_table(ost, FALSE, verbose_level);





		L.foot(ost);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "algebra_global::do_cheat_sheet_GF q=" << F->q << " done" << endl;
	}
}






void algebra_global::gl_random_matrix(finite_field *F, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	int *M2;
	unipoly_object char_poly;

	if (f_v) {
		cout << "algebra_global::gl_random_matrix" << endl;
		}
	//F.finite_field_init(q, 0 /*verbose_level*/);
	M = NEW_int(k * k);
	M2 = NEW_int(k * k);

	F->Linear_algebra->random_invertible_matrix(M, k, verbose_level - 2);

	cout << "Random invertible matrix:" << endl;
	Orbiter->Int_vec->matrix_print(M, k, k);


	{
		unipoly_domain U(F);



		U.create_object_by_rank(char_poly, 0, __FILE__, __LINE__, verbose_level);

		U.characteristic_polynomial(M, k, char_poly, verbose_level - 2);

		cout << "The characteristic polynomial is ";
		U.print_object(char_poly, cout);
		cout << endl;

		U.substitute_matrix_in_polynomial(char_poly, M, M2, k, verbose_level);
		cout << "After substitution, the matrix is " << endl;
		Orbiter->Int_vec->matrix_print(M2, k, k);

		U.delete_object(char_poly);

	}
	FREE_int(M);
	FREE_int(M2);

}




void algebra_global::apply_Walsh_Hadamard_transform(finite_field *F,
		std::string &fname_csv_in, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::apply_Walsh_Hadamard_transform" << endl;
	}


	combinatorics::boolean_function_domain *BF;

	BF = NEW_OBJECT(combinatorics::boolean_function_domain);

	BF->init(n, verbose_level);


	file_io Fio;
	int *M;
	int m, nb_cols;
	int len;
	string fname_csv_out;
	data_structures::string_tools ST;

	fname_csv_out.assign(fname_csv_in);
	ST.chop_off_extension(fname_csv_out);
	fname_csv_out.append("_transformed.csv");

	Fio.int_matrix_read_csv(fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	if (len != BF->Q) {
		cout << "algebra_global::apply_Walsh_Hadamard_transform len != BF->Q" << endl;
		exit(1);
	}
	BF->raise(M, BF->F);

	BF->apply_Walsh_transform(BF->F, BF->T);

	cout << " : ";
	Orbiter->Int_vec->print(cout, BF->T, BF->Q);
	cout << endl;

	if (EVEN(n)) {
		if (BF->is_bent(BF->T)) {
			cout << "is bent" << endl;
		}
		else {
			cout << "is not bent" << endl;
		}
	}
	else {
		if (BF->is_near_bent(BF->T)) {
			cout << "is near bent" << endl;
		}
		else {
			cout << "is not near bent" << endl;
		}

	}
	Fio.int_matrix_write_csv(fname_csv_out, BF->T, m, nb_cols);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);
	FREE_OBJECT(BF);

	if (f_v) {
		cout << "algebra_global::apply_Walsh_Hadamard_transform done" << endl;
	}
}

void algebra_global::algebraic_normal_form(finite_field *F,
		std::string &fname_csv_in, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::algebraic_normal_form" << endl;
	}


	combinatorics::boolean_function_domain *BF;

	BF = NEW_OBJECT(combinatorics::boolean_function_domain);

	if (f_v) {
		cout << "algebra_global::algebraic_normal_form before BF->init" << endl;
	}
	BF->init(n, verbose_level);
	if (f_v) {
		cout << "algebra_global::algebraic_normal_form after BF->init" << endl;
	}


	file_io Fio;
	int *M;
	int m, nb_cols;
	int len;
	string fname_csv_out;
	data_structures::string_tools ST;

	fname_csv_out.assign(fname_csv_in);
	ST.chop_off_extension(fname_csv_out);
	fname_csv_out.append("_alg_normal_form.csv");

	Fio.int_matrix_read_csv(fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	if (len != BF->Q) {
		cout << "algebra_global::algebraic_normal_form len != BF->Q" << endl;
		exit(1);
	}

	int *coeff;
	int nb_coeff;

	nb_coeff = BF->Poly[n].get_nb_monomials();

	coeff = NEW_int(nb_coeff);

	if (f_v) {
		cout << "algebra_global::algebraic_normal_form before BF->compute_polynomial_representation" << endl;
	}
	BF->compute_polynomial_representation(M, coeff, verbose_level);
	if (f_v) {
		cout << "algebra_global::algebraic_normal_form after BF->compute_polynomial_representation" << endl;
	}

	cout << "algebraic normal form:" << endl;
	BF->Poly[n].print_equation(cout, coeff);
	cout << endl;

	cout << "algebraic normal form in tex:" << endl;
	BF->Poly[n].print_equation_tex(cout, coeff);
	cout << endl;

	cout << "algebraic normal form in numerical form:" << endl;
	BF->Poly[n].print_equation_numerical(cout, coeff);
	cout << endl;



	Fio.int_matrix_write_csv(fname_csv_out, coeff, 1, nb_coeff);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);
	FREE_OBJECT(BF);

	if (f_v) {
		cout << "algebra_global::algebraic_normal_form done" << endl;
	}
}

void algebra_global::apply_trace_function(finite_field *F,
		std::string &fname_csv_in, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::apply_trace_function" << endl;
	}


	file_io Fio;
	int *M;
	int m, nb_cols;
	int len, i;
	string fname_csv_out;
	data_structures::string_tools ST;

	fname_csv_out.assign(fname_csv_in);
	ST.chop_off_extension(fname_csv_out);
	fname_csv_out.append("_trace.csv");

	Fio.int_matrix_read_csv(fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	for (i = 0; i < len; i++) {
		M[i] = F->absolute_trace(M[i]);
	}
	Fio.int_matrix_write_csv(fname_csv_out, M, m, nb_cols);

	FREE_int(M);

	if (f_v) {
		cout << "algebra_global::apply_trace_function done" << endl;
	}
}

void algebra_global::apply_power_function(finite_field *F,
		std::string &fname_csv_in, long int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::apply_power_function" << endl;
	}


	file_io Fio;
	int *M;
	int m, nb_cols;
	int len, i;
	string fname_csv_out;
	data_structures::string_tools ST;

	fname_csv_out.assign(fname_csv_in);
	ST.chop_off_extension(fname_csv_out);

	char str[1000];

	sprintf(str, "_power_%ld.csv", d);
	fname_csv_out.append(str);

	Fio.int_matrix_read_csv(fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	for (i = 0; i < len; i++) {
		M[i] = F->power(M[i], d);
	}
	Fio.int_matrix_write_csv(fname_csv_out, M, m, nb_cols);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);

	if (f_v) {
		cout << "algebra_global::apply_power_function done" << endl;
	}
}

void algebra_global::identity_function(finite_field *F,
		std::string &fname_csv_out, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::identity_function" << endl;
	}


	file_io Fio;
	int *M;
	int i;

	M = NEW_int(F->q);
	for (i = 0; i < F->q; i++) {
		M[i] = i;
	}
	Fio.int_matrix_write_csv(fname_csv_out, M, 1, F->q);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);

	if (f_v) {
		cout << "algebra_global::identity_function done" << endl;
	}
}


void algebra_global::Walsh_matrix(finite_field *F, int n, int *&W, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Q;
	int *v;
	int *w;
	int *W01;
	int i, j, a;
	geometry_global Gg;

	if (f_v) {
		cout << "algebra_global::Walsh_matrix" << endl;
	}
	v = NEW_int(n);
	w = NEW_int(n);
	Q = 1 << n;
	W = NEW_int(Q * Q);
	W01 = NEW_int(Q * Q);
	for (i = 0; i < Q; i++) {
		Gg.AG_element_unrank(2, v, 1, n, i);
		for (j = 0; j < Q; j++) {
			Gg.AG_element_unrank(2, w, 1, n, j);
			a = F->Linear_algebra->dot_product(n, v, w);
			if (a) {
				W[i * Q + j] = -1;
				W01[i * Q + j] = 1;
			}
			else {
				W[i * Q + j] = 1;
				W01[i * Q + j] = 0;
			}
		}
	}
	char str[1000];
	string fname_csv;
	file_io Fio;

	snprintf(str, 1000, "Walsh_pm_%d.csv", n);
	fname_csv.assign(str);
	Fio.int_matrix_write_csv(fname_csv, W, Q, Q);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	snprintf(str, 1000, "Walsh_01_%d.csv", n);
	fname_csv.assign(str);
	Fio.int_matrix_write_csv(fname_csv, W01, Q, Q);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	FREE_int(v);
	FREE_int(w);
	FREE_int(W01);
	if (f_v) {
		cout << "algebra_global::Walsh_matrix done" << endl;
	}
}

void algebra_global::Vandermonde_matrix(finite_field *F, int *&W, int *&W_inv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q;
	int i, j, a;
	geometry_global Gg;

	if (f_v) {
		cout << "algebra_global::Vandermonde_matrix" << endl;
	}
	q = F->q;
	W = NEW_int(q * q);
	W_inv = NEW_int(q * q);
	for (i = 0; i < q; i++) {
		a = 1;
		W[i * q + 0] = 1;
		for (j = 1; j < q; j++) {
			a = F->mult(i, a);
			W[i * q + j] = a;
		}
	}

	if (f_v) {
		cout << "algebra_global::Vandermonde_matrix before invert_matrix" << endl;
	}
	F->Linear_algebra->invert_matrix(W, W_inv, q, verbose_level);
	if (f_v) {
		cout << "algebra_global::Vandermonde_matrix after invert_matrix" << endl;
	}

	char str[1000];
	string fname_csv;
	file_io Fio;

	snprintf(str, 1000, "Vandermonde_%d.csv", q);
	fname_csv.assign(str);
	Fio.int_matrix_write_csv(fname_csv, W, q, q);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	snprintf(str, 1000, "Vandermonde_inv_%d.csv", q);
	fname_csv.assign(str);
	Fio.int_matrix_write_csv(fname_csv, W_inv, q, q);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	if (f_v) {
		cout << "algebra_global::Vandermonde_matrix done" << endl;
	}
}

void algebra_global::search_APN(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q;
	int *f;
	int delta_min, nb_times;

	if (f_v) {
		cout << "algebra_global::search_APN" << endl;
	}
	q = F->q;
	delta_min = INT_MAX;
	nb_times = 0;
	f = NEW_int(q);

	std::vector<std::vector<int> > Solutions;

	search_APN_recursion(F,
			f,
			0 /* depth */,
			delta_min, nb_times, Solutions,
			verbose_level);
	cout << "nb_times = " << nb_times << endl;
	FREE_int(f);

	string fname;
	char str[1000];
	file_io Fio;

	sprintf(str, "APN_functions_q%d.csv", F->q);

	fname.assign(str);
	Fio.vector_matrix_write_csv(fname, Solutions);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "algebra_global::search_APN done" << endl;
	}
}

void algebra_global::search_APN_recursion(finite_field *F,
		int *f, int depth, int &delta_min, int &nb_times,
		std::vector<std::vector<int> > &Solutions, int verbose_level)
{
	if (depth == F->q) {
		int delta;

		delta = non_linearity(F, f, 0 /* verbose_level */);
		if (delta < delta_min) {
			delta_min = delta;
			nb_times = 1;

			Solutions.clear();

			vector<int> S;
			int i;

			for (i = 0; i < F->q; i++) {
				S.push_back(f[i]);
			}
			Solutions.push_back(S);

			Orbiter->Int_vec->print(cout, f, F->q);
			cout << " delta = " << delta << " nb_times=" << nb_times << endl;
		}
		else if (delta == delta_min) {
			nb_times++;
			int f_do_it;

			if (nb_times > 100) {
				if ((nb_times % 1000) == 0) {
					f_do_it = TRUE;
				}
				else {
					f_do_it = FALSE;
				}
			}
			else {
				f_do_it = TRUE;
			}

			vector<int> S;
			int i;

			for (i = 0; i < F->q; i++) {
				S.push_back(f[i]);
			}
			Solutions.push_back(S);

			if (f_do_it) {
				Orbiter->Int_vec->print(cout, f, F->q);
				cout << " delta = " << delta << " nb_times=" << nb_times << endl;
			}
		}
		return;
	}

	int a;

	for (a = 0; a < F->q; a++) {
		f[depth]= a;
		search_APN_recursion(F,
				f, depth + 1, delta_min, nb_times, Solutions, verbose_level);
	}
}

int algebra_global::non_linearity(finite_field *F, int *f, int verbose_level)
// f[q]
{
	int f_v = (verbose_level >= 1);
	int q;
	int a, av, x, b, fx, fxpa, mfx, dy, delta;
	int *nb_times_ab;

	if (f_v) {
		cout << "algebra_global::non_linearity" << endl;
	}
	q = F->q;
	nb_times_ab = NEW_int(q * q);
	Orbiter->Int_vec->zero(nb_times_ab, q * q);
	for (x = 0; x < q; x++) {
		fx = f[x];
		mfx = F->negate(fx);
		for (a = 1; a < q; a++) {
			fxpa = f[F->add(x, a)];
			av = F->inverse(a);
			dy = F->add(fxpa, mfx);
			b = F->mult(dy, av);
			nb_times_ab[a * q + b]++;
		}
	}
	delta = 0;
	for (a = 1; a < q; a++) {
		for (b = 0; b < q; b++) {
			delta = MAXIMUM(delta, nb_times_ab[a * q + b]);
		}
	}
	FREE_int(nb_times_ab);
	return delta;
}

void algebra_global::O4_isomorphism_4to2(finite_field *F,
		int *At, int *As, int &f_switch, int *B,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a, b, c, d, e, f, g, h;
	int ev, fv;
	int P[4], Q[4], R[4], S[4];
	int Rx, Ry, Sx, Sy;
	int /*b11,*/ b12, b13, b14;
	int /*b21,*/ b22, b23, b24;
	int /*b31,*/ b32, b33, b34;
	int /*b41,*/ b42, b43, b44;

	if (f_v) {
		cout << "algebra_global::O4_isomorphism_4to2" << endl;
	}
	//b11 = B[0 * 4 + 0];
	b12 = B[0 * 4 + 1];
	b13 = B[0 * 4 + 2];
	b14 = B[0 * 4 + 3];
	//b21 = B[1 * 4 + 0];
	b22 = B[1 * 4 + 1];
	b23 = B[1 * 4 + 2];
	b24 = B[1 * 4 + 3];
	//b31 = B[2 * 4 + 0];
	b32 = B[2 * 4 + 1];
	b33 = B[2 * 4 + 2];
	b34 = B[2 * 4 + 3];
	//b41 = B[3 * 4 + 0];
	b42 = B[3 * 4 + 1];
	b43 = B[3 * 4 + 2];
	b44 = B[3 * 4 + 3];
	O4_grid_coordinates_unrank(F, P[0], P[1], P[2], P[3],
			0, 0, verbose_level);
	if (f_vv) {
		cout << "grid point (0,0) = ";
		Orbiter->Int_vec->print(cout, P, 4);
		cout << endl;
	}
	O4_grid_coordinates_unrank(F, Q[0], Q[1], Q[2], Q[3],
			1, 0, verbose_level);
	if (f_vv) {
		cout << "grid point (1,0) = ";
		Orbiter->Int_vec->print(cout, Q, 4);
		cout << endl;
	}
	F->Linear_algebra->mult_vector_from_the_left(P, B, R, 4, 4);
	F->Linear_algebra->mult_vector_from_the_left(Q, B, S, 4, 4);
	O4_grid_coordinates_rank(F, R[0], R[1], R[2], R[3],
			Rx, Ry, verbose_level);
	O4_grid_coordinates_rank(F, S[0], S[1], S[2], S[3],
			Sx, Sy, verbose_level);
	if (f_vv) {
		cout << "Rx=" << Rx << " Ry=" << Ry
				<< " Sx=" << Sx << " Sy=" << Sy << endl;
	}
	if (Ry == Sy) {
		f_switch = FALSE;
	}
	else {
		f_switch = TRUE;
	}
	if (f_vv) {
		cout << "f_switch=" << f_switch << endl;
	}
	if (f_switch) {
		if (b22 == 0 && b24 == 0 && b32 == 0 && b34 == 0) {
			a = 0;
			b = 1;
			f = b12;
			h = b14;
			e = b42;
			g = b44;
			if (e == 0) {
				fv = F->inverse(f);
				c = F->mult(fv, b33);
				d = F->negate(F->mult(fv, b13));
			}
			else {
				ev = F->inverse(e);
				c = F->negate(F->mult(ev, b23));
				d = F->negate(F->mult(ev, b43));
			}
		}
		else {
			a = 1;
			e = b22;
			g = b24;
			f = F->negate(b32);
			h = F->negate(b34);
			if (e == 0) {
				fv = F->inverse(f);
				b = F->mult(fv, b12);
				c = F->mult(fv, b33);
				d = F->negate(F->mult(fv, b13));
			}
			else {
				ev = F->inverse(e);
				b = F->mult(ev, b42);
				c = F->negate(F->mult(ev, b23));
				d = F->negate(F->mult(ev, b43));
			}
		}
	}
	else {
		// no switch
		if (b22 == 0 && b24 == 0 && b42 == 0 && b44 == 0) {
			a = 0;
			b = 1;
			f = b12;
			h = b14;
			e = F->negate(b32);
			g = F->negate(b34);
			if (e == 0) {
				fv = F->inverse(f);
				c = F->negate(F->mult(fv, b43));
				d = F->negate(F->mult(fv, b13));
			}
			else {
				ev = F->inverse(e);
				c = F->negate(F->mult(ev, b23));
				d = F->mult(ev, b33);
			}
		}
		else {
			a = 1;
			e = b22;
			g = b24;
			f = b42;
			h = b44;
			if (e == 0) {
				fv = F->inverse(f);
				b = F->mult(fv, b12);
				c = F->negate(F->mult(fv, b43));
				d = F->negate(F->mult(fv, b13));
			}
			else {
				ev = F->inverse(e);
				b = F->negate(F->mult(ev, b32));
				c = F->negate(F->mult(ev, b23));
				d = F->mult(ev, b33);
			}
		}
	}
	if (f_vv) {
		cout << "a=" << a << " b=" << b << " c=" << c << " d=" << d << endl;
		cout << "e=" << e << " f=" << f << " g=" << g << " h=" << h << endl;
	}
	At[0] = d;
	At[1] = b;
	At[2] = c;
	At[3] = a;
	As[0] = h;
	As[1] = f;
	As[2] = g;
	As[3] = e;
	if (f_v) {
		cout << "At:" << endl;
		Orbiter->Int_vec->print_integer_matrix_width(cout, At, 2, 2, 2, F->log10_of_q);
		cout << "As:" << endl;
		Orbiter->Int_vec->print_integer_matrix_width(cout, As, 2, 2, 2, F->log10_of_q);
	}

}

void algebra_global::O4_isomorphism_2to4(finite_field *F,
		int *At, int *As, int f_switch, int *B)
{
	int a, b, c, d, e, f, g, h;

	a = At[3];
	b = At[1];
	c = At[2];
	d = At[0];
	e = As[3];
	f = As[1];
	g = As[2];
	h = As[0];
	if (f_switch) {
		B[0 * 4 + 0] = F->mult(h, d);
		B[0 * 4 + 1] = F->mult(f, b);
		B[0 * 4 + 2] = F->negate(F->mult(f, d));
		B[0 * 4 + 3] = F->mult(h, b);
		B[1 * 4 + 0] = F->mult(g, c);
		B[1 * 4 + 1] = F->mult(e, a);
		B[1 * 4 + 2] = F->negate(F->mult(e, c));
		B[1 * 4 + 3] = F->mult(g, a);
		B[2 * 4 + 0] = F->negate(F->mult(h, c));
		B[2 * 4 + 1] = F->negate(F->mult(f, a));
		B[2 * 4 + 2] = F->mult(f, c);
		B[2 * 4 + 3] = F->negate(F->mult(h, a));
		B[3 * 4 + 0] = F->mult(g, d);
		B[3 * 4 + 1] = F->mult(e, b);
		B[3 * 4 + 2] = F->negate(F->mult(e, d));
		B[3 * 4 + 3] = F->mult(g, b);
	}
	else {
		B[0 * 4 + 0] = F->mult(h, d);
		B[0 * 4 + 1] = F->mult(f, b);
		B[0 * 4 + 2] = F->negate(F->mult(f, d));
		B[0 * 4 + 3] = F->mult(h, b);
		B[1 * 4 + 0] = F->mult(g, c);
		B[1 * 4 + 1] = F->mult(e, a);
		B[1 * 4 + 2] = F->negate(F->mult(e, c));
		B[1 * 4 + 3] = F->mult(g, a);
		B[2 * 4 + 0] = F->negate(F->mult(g, d));
		B[2 * 4 + 1] = F->negate(F->mult(e, b));
		B[2 * 4 + 2] = F->mult(e, d);
		B[2 * 4 + 3] = F->negate(F->mult(g, b));
		B[3 * 4 + 0] = F->mult(h, c);
		B[3 * 4 + 1] = F->mult(f, a);
		B[3 * 4 + 2] = F->negate(F->mult(f, c));
		B[3 * 4 + 3] = F->mult(h, a);
	}
}

void algebra_global::O4_grid_coordinates_rank(finite_field *F,
		int x1, int x2, int x3, int x4, int &grid_x, int &grid_y,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, c, d, av, e;
	int v[2], w[2];

	a = x1;
	b = x4;
	c = F->negate(x3);
	d = x2;

	if (a) {
		if (a != 1) {
			av = F->inverse(a);
			b = F->mult(b, av);
			c = F->mult(c, av);
			d = F->mult(d, av);
		}
		v[0] = 1;
		w[0] = 1;
		v[1] = c;
		w[1] = b;
		e = F->mult(c, b);
		if (e != d) {
			cout << "algebra_global::O4_grid_coordinates_rank "
					"e != d" << endl;
			exit(1);
		}
	}
	else if (b == 0) {
		v[0] = 0;
		v[1] = 1;
		w[0] = c;
		w[1] = d;
	}
	else {
		if (c) {
			cout << "a is zero, b and c are not" << endl;
			exit(1);
		}
		w[0] = 0;
		w[1] = 1;
		v[0] = b;
		v[1] = d;
	}
	F->PG_element_normalize_from_front(v, 1, 2);
	F->PG_element_normalize_from_front(w, 1, 2);
	if (f_v) {
		Orbiter->Int_vec->print(cout, v, 2);
		Orbiter->Int_vec->print(cout, w, 2);
		cout << endl;
	}

	F->PG_element_rank_modified(v, 1, 2, grid_x);
	F->PG_element_rank_modified(w, 1, 2, grid_y);
}

void algebra_global::O4_grid_coordinates_unrank(finite_field *F,
		int &x1, int &x2, int &x3, int &x4,
		int grid_x, int grid_y,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, c, d;
	int v[2], w[2];

	F->PG_element_unrank_modified(v, 1, 2, grid_x);
	F->PG_element_unrank_modified(w, 1, 2, grid_y);
	F->PG_element_normalize_from_front(v, 1, 2);
	F->PG_element_normalize_from_front(w, 1, 2);
	if (f_v) {
		Orbiter->Int_vec->print(cout, v, 2);
		Orbiter->Int_vec->print(cout, w, 2);
		cout << endl;
	}

	a = F->mult(v[0], w[0]);
	b = F->mult(v[0], w[1]);
	c = F->mult(v[1], w[0]);
	d = F->mult(v[1], w[1]);
	x1 = a;
	x2 = d;
	x3 = F->negate(c);
	x4 = b;
}

void algebra_global::O4_find_tangent_plane(finite_field *F,
		int pt_x1, int pt_x2, int pt_x3, int pt_x4,
		int *tangent_plane,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int A[4];
	int C[3 * 4];
	int size, x, y, z, xx, yy, zz, h, k;
	int x1, x2, x3, x4;
	int y1, y2, y3, y4;
	int f_special = FALSE;
	int f_complete = FALSE;
	int base_cols[4];
	int f_P = FALSE;
	int rk, det;
	int vec2[2];


	if (f_v) {
		cout << "algebra_global::O4_find_tangent_plane pt_x1=" << pt_x1
				<< " pt_x2=" << pt_x2
				<< " pt_x3=" << pt_x3
				<< " pt_x4=" << pt_x4 << endl;
	}
	size = F->q + 1;
#if 0
	A[0] = pt_x1;
	A[3] = pt_x2;
	A[2] = negate(pt_x3);
	A[1] = pt_x4;
#endif

	int *secants1;
	int *secants2;
	int nb_secants = 0;
	int *complement;
	int nb_complement = 0;

	secants1 = NEW_int(size * size);
	secants2 = NEW_int(size * size);
	complement = NEW_int(size * size);
	for (x = 0; x < size; x++) {
		for (y = 0; y < size; y++) {
			z = x * size + y;

			//cout << "trying grid point (" << x << "," << y << ")" << endl;
			//cout << "nb_secants=" << nb_secants << endl;
			O4_grid_coordinates_unrank(F, x1, x2, x3, x4, x, y, 0);

			//cout << "x1=" << x1 << " x2=" << x2
			//<< " x3=" << x3 << " x4=" << x4 << endl;




			for (k = 0; k < size; k++) {
				F->PG_element_unrank_modified(vec2, 1, 2, k);
				y1 = F->add(F->mult(pt_x1, vec2[0]), F->mult(x1, vec2[1]));
				y2 = F->add(F->mult(pt_x2, vec2[0]), F->mult(x2, vec2[1]));
				y3 = F->add(F->mult(pt_x3, vec2[0]), F->mult(x3, vec2[1]));
				y4 = F->add(F->mult(pt_x4, vec2[0]), F->mult(x4, vec2[1]));
				det = F->add(F->mult(y1, y2), F->mult(y3, y4));
				if (det != 0) {
					continue;
				}
				O4_grid_coordinates_rank(F, y1, y2, y3, y4, xx, yy, 0);
				zz = xx * size + yy;
				if (zz == z) {
					continue;
				}
				C[0] = pt_x1;
				C[1] = pt_x2;
				C[2] = pt_x3;
				C[3] = pt_x4;

				C[4] = x1;
				C[5] = x2;
				C[6] = x3;
				C[7] = x4;

				C[8] = y1;
				C[9] = y2;
				C[10] = y3;
				C[11] = y4;

				rk = F->Linear_algebra->Gauss_int(C,
						f_special, f_complete, base_cols,
					f_P, NULL, 3, 4, 4, 0);
				if (rk < 3) {
					secants1[nb_secants] = z;
					secants2[nb_secants] = zz;
					nb_secants++;
				}

			}

#if 0

			for (xx = 0; xx < size; xx++) {
				for (yy = 0; yy < size; yy++) {
					zz = xx * size + yy;
					if (zz == z)
						continue;
					O4_grid_coordinates_unrank(F, y1, y2, y3, y4, xx, yy, 0);
					//cout << "y1=" << y1 << " y2=" << y2
					//<< " y3=" << y3 << " y4=" << y4 << endl;
					C[0] = pt_x1;
					C[1] = pt_x2;
					C[2] = pt_x3;
					C[3] = pt_x4;

					C[4] = x1;
					C[5] = x2;
					C[6] = x3;
					C[7] = x4;

					C[8] = y1;
					C[9] = y2;
					C[10] = y3;
					C[11] = y4;

					rk = F.Gauss_int(C, f_special, f_complete, base_cols,
						f_P, NULL, 3, 4, 4, 0);
					if (rk < 3) {
						secants1[nb_secants] = z;
						secants2[nb_secants] = zz;
						nb_secants++;
						}
					}
				}
#endif


		}
	}
	if (f_v) {
		cout << "nb_secants=" << nb_secants << endl;
		Orbiter->Int_vec->print(cout, secants1, nb_secants);
		cout << endl;
		Orbiter->Int_vec->print(cout, secants2, nb_secants);
		cout << endl;
	}
	h = 0;
	for (zz = 0; zz < size * size; zz++) {
		if (secants1[h] > zz) {
			complement[nb_complement++] = zz;
		}
		else {
			h++;
		}
	}
	if (f_v) {
		cout << "complement = tangents:" << endl;
		Orbiter->Int_vec->print(cout, complement, nb_complement);
		cout << endl;
	}

	int *T;
	T = NEW_int(4 * nb_complement);

	for (h = 0; h < nb_complement; h++) {
		z = complement[h];
		x = z / size;
		y = z % size;
		if (f_v) {
			cout << setw(3) << h << " : " << setw(4) << z
					<< " : " << x << "," << y << " : ";
		}
		O4_grid_coordinates_unrank(F, y1, y2, y3, y4,
				x, y, verbose_level);
		if (f_v) {
			cout << "y1=" << y1 << " y2=" << y2
					<< " y3=" << y3 << " y4=" << y4 << endl;
		}
		T[h * 4 + 0] = y1;
		T[h * 4 + 1] = y2;
		T[h * 4 + 2] = y3;
		T[h * 4 + 3] = y4;
	}


	rk = F->Linear_algebra->Gauss_int(T, f_special, f_complete, base_cols,
		f_P, NULL, nb_complement, 4, 4, 0);
	if (f_v) {
		cout << "the rank of the tangent space is " << rk << endl;
		cout << "basis:" << endl;
		Orbiter->Int_vec->print_integer_matrix_width(cout, T, rk, 4, 4, F->log10_of_q);
	}

	if (rk != 3) {
		cout << "rk = " << rk << " not equal to 3" << endl;
		exit(1);
	}
	int i;
	for (i = 0; i < 12; i++) {
		tangent_plane[i] = T[i];
	}
	FREE_int(secants1);
	FREE_int(secants2);
	FREE_int(complement);
	FREE_int(T);

#if 0
	for (h = 0; h < nb_secants; h++) {
		z = secants1[h];
		zz = secants2[h];
		x = z / size;
		y = z % size;
		xx = zz / size;
		yy = zz % size;
		cout << "(" << x << "," << y << "),(" << xx
				<< "," << yy << ")" << endl;
		O4_grid_coordinates_unrank(F, x1, x2, x3, x4,
				x, y, verbose_level);
		cout << "x1=" << x1 << " x2=" << x2
				<< " x3=" << x3 << " x4=" << x4 << endl;
		O4_grid_coordinates_unrank(F, y1, y2, y3, y4, xx, yy, verbose_level);
		cout << "y1=" << y1 << " y2=" << y2
				<< " y3=" << y3 << " y4=" << y4 << endl;
	}
#endif
}





}}}

