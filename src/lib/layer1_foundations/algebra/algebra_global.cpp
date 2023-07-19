/*
 * algebra_global.cpp
 *
 *  Created on: Nov 25, 2019
 *      Author: betten
 */






#include "foundations.h"

using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace algebra {



void algebra_global::count_subprimitive(int Q_max, int H_max)
{
	int q, h, p, e, i, g, phi_g, l, cmp;
	int *Q, *Rdq, *G, nb_primes = 0;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object r2, r3, A, B;
	number_theory::number_theory_domain NT;

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



void algebra_global::formula_subprimitive(int d, int q,
		ring_theory::longinteger_object &Rdq, int &g, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int theta_mod_qm1, p, e, i, rem;
	int nb_primes, *primes, *exponents;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object Theta, M1, Qm1, A, B, C, R;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "algebra_global::formula_subprimitive d=" << d << " q=" << q << endl;
	}
	Theta.create(q);
	M1.create(-1);
	Qm1.create(q - 1);
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
		if (f_v) {
			cout << "p^e=" << A << endl;
		}
		B.create(p);
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

void algebra_global::formula(int d, int q,
		ring_theory::longinteger_object &Rdq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int theta_mod_qm1, g, p, e, i, rem;
	int nb_primes, *primes, *exponents;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object Theta, M1, Qm1, A, B, C, R;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "algebra_global::formula d=" << d << " q=" << q << endl;
	}
	Theta.create(q);
	M1.create(-1);
	Qm1.create(q - 1);
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

int algebra_global::subprimitive(int q, int h)
{
	int Q, f, i, j, k, s, c, l, r = 0;
	number_theory::number_theory_domain NT;

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
	Int_vec_print_integer_matrix_width(cout, CM, 1, q - 1, q - 1, 2);
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

void algebra_global::subexponent(
		int q, int Q, int h, int f, int j, int k, int &s, int &c)
{
	int a, g;
	number_theory::number_theory_domain NT;

	a = j + k * (q - 1);
	g = NT.gcd_lint(a, f);
	s = f / g;
	c = a / g;
	c = c % (q - 1);
#if 0
	for (s = 1; true; s++) {
		b = a * s;
		if ((b % f) == 0) {
			c = b / f;
			c = c % (q - 1);
			return;
			}
		}
#endif
}


std::string algebra_global::plus_minus_string(int epsilon)
{
	string s;

	if (epsilon == 1) {
		s = "p";
	}
	else if (epsilon == -1) {
		s = "m";
	}
	else if (epsilon == 0) {
		s = "";
	}
	else {
		cout << "algebra_global::plus_minus_letter unrecognized, epsilon=" << epsilon << endl;
		exit(1);
	}
	return s;
}




void algebra_global::display_all_PHG_elements(int n, int q)
{
	int *v = NEW_int(n + 1);
	int l;
	int i, j, a;
	ring_theory::finite_ring R;

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

void algebra_global::test_unipoly(field_theory::finite_field *F)
{
	ring_theory::unipoly_object m, a, b, c;
	ring_theory::unipoly_object elts[4];
	int i, j;
	int verbose_level = 0;

	ring_theory::unipoly_domain FX(F);

	FX.create_object_by_rank(m, 7, 0);
	FX.create_object_by_rank(a, 5, 0);
	FX.create_object_by_rank(b, 55, 0);
	FX.print_object(a, cout); cout << endl;
	FX.print_object(b, cout); cout << endl;

	ring_theory::unipoly_domain Fq(F, m, verbose_level);
	Fq.create_object_by_rank(c, 2, 0);
	for (i = 0; i < 4; i++) {
		Fq.create_object_by_rank(elts[i], i, 0);
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

void algebra_global::test_unipoly2(field_theory::finite_field *F)
{
	int i;

	ring_theory::unipoly_domain FX(F);

	ring_theory::unipoly_object a;

	FX.create_object_by_rank(a, 0, 0);
	for (i = 1; i < F->q; i++) {
		FX.minimum_polynomial(a, i, F->p, true);
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
					return false;
				}
			}
		}
	}
	return true;
}




void algebra_global::test_longinteger()
{
	ring_theory::longinteger_domain D;
	int x[] = {15, 14, 12, 8};
	ring_theory::longinteger_object a, b, q, r;
	int verbose_level = 0;

	D.multiply_up(a, x, 4, verbose_level);
	cout << "a=" << a << endl;
	b.create(2);
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
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, b, c, d, e;
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
	ring_theory::longinteger_object a, b, c, d, e;

	for (i = 0; i < 10; i++) {
		for (j = 0; j < 10; j++) {
			D.binomial(a, i, j, false);
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
	ring_theory::longinteger_object a;

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
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, b, u, v, g;
	int verbose_level = 2;

	a.create(9548);
	b.create(254774);
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
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, b;

	a.create(7411);
	b.create(9283);
	D.jacobi(a, b, verbose_level);


}

void algebra_global::test_longinteger7()
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, b;
	int i, j;
	int mult[15];

	a.create(15);
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
	ring_theory::longinteger_object a, b, one;
	int nb_solovay_strassen_tests = 100;
	int f_miller_rabin_test = true;

	one.create(1);
	a.create(197659);
	Crypto.find_probable_prime_above(a, nb_solovay_strassen_tests,
		f_miller_rabin_test, verbose_level);
}

void algebra_global::longinteger_collect_setup(
		int &nb_agos,
		ring_theory::longinteger_object *&agos, int *&multiplicities)
{
	nb_agos = 0;
	agos = NULL;
	multiplicities = NULL;
}

void algebra_global::longinteger_collect_free(
		int &nb_agos,
		ring_theory::longinteger_object *&agos, int *&multiplicities)
{
	if (nb_agos) {
		FREE_OBJECTS(agos);
		FREE_int(multiplicities);
	}
}

void algebra_global::longinteger_collect_add(
		int &nb_agos,
		ring_theory::longinteger_object *&agos, int *&multiplicities,
		ring_theory::longinteger_object &ago)
{
	int j, c, h, f_added;
	ring_theory::longinteger_object *tmp_agos;
	int *tmp_multiplicities;
	ring_theory::longinteger_domain D;

	f_added = false;
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
				agos = NEW_OBJECTS(ring_theory::longinteger_object, nb_agos + 1);
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
			f_added = true;
			break;
		}
	}
	if (!f_added) {
		// add at the end (including the case that the list is empty)
		tmp_agos = agos;
		tmp_multiplicities = multiplicities;
		agos = NEW_OBJECTS(ring_theory::longinteger_object, nb_agos + 1);
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

void algebra_global::longinteger_collect_print(
		std::ostream &ost,
		int &nb_agos,
		ring_theory::longinteger_object *&agos,
		int *&multiplicities)
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






void algebra_global::order_of_q_mod_n(
		int q, int n_min, int n_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::order_of_q_mod_n" << endl;
	}
	{
		string fname;

		fname = "order_of_q_mod_n_q" + std::to_string(q) + "_" + std::to_string(n_min) + "_" + std::to_string(n_max) + ".csv";


		{
			ofstream ost(fname);
			number_theory::number_theory_domain NT;

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
		orbiter_kernel_system::file_io Fio;

		cout << "algebra_global::order_of_q_mod_n written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "algebra_global::order_of_q_mod_n done" << endl;
	}
}


void algebra_global::power_function_mod_n(
		int k, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::power_function_mod_n" << endl;
	}
	{
		string fname;

		fname = "power_function_k" + std::to_string(k) + "_n" + std::to_string(n) + ".csv";


		{
			ofstream ost(fname);
			number_theory::number_theory_domain NT;

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
		orbiter_kernel_system::file_io Fio;

		cout << "algebra_global::power_function_mod_n "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "algebra_global::power_mod_interval_n done" << endl;
	}
}

void algebra_global::do_trace(
		field_theory::finite_field *F, int verbose_level)
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
	Int_vec_print_fully(cout, T0, nb_T0);
	cout << endl;

	cout << "Trace 1:" << endl;
	Int_vec_print_fully(cout, T1, nb_T1);
	cout << endl;

	string fname_csv;
	orbiter_kernel_system::file_io Fio;

	fname_csv = "F_q" + std::to_string(F->q) + "_trace.csv";
	Fio.Csv_file_support->int_matrix_write_csv(fname_csv, T, 1, F->q);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	fname_csv = "F_q" + std::to_string(F->q) + "_trace_0.csv";

	string label;
	label.assign("Trace_0");
	Fio.Csv_file_support->int_vec_write_csv(T0, nb_T0, fname_csv, label);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;

	fname_csv = "F_q" + std::to_string(F->q) + "_trace_1.csv";
	label.assign("Trace_1");
	Fio.Csv_file_support->int_vec_write_csv(T1, nb_T1, fname_csv, label);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;

	FREE_int(T);
	FREE_int(T0);
	FREE_int(T1);

	if (f_v) {
		cout << "algebra_global::do_trace done" << endl;
	}
}

void algebra_global::do_norm(
		field_theory::finite_field *F, int verbose_level)
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
	Int_vec_print_fully(cout, T0, nb_T0);
	cout << endl;

	cout << "Norm 1:" << endl;
	Int_vec_print_fully(cout, T1, nb_T1);
	cout << endl;


	string fname_csv;
	orbiter_kernel_system::file_io Fio;
	string label;

	fname_csv = "F_q" + std::to_string(F->q) + "_norm_0.csv";
	label.assign("Norm_0");
	Fio.Csv_file_support->int_vec_write_csv(T0, nb_T0, fname_csv, label);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;

	fname_csv = "F_q" + std::to_string(F->q) + "_norm_1.csv";
	label.assign("Norm_1");
	Fio.Csv_file_support->int_vec_write_csv(T1, nb_T1, fname_csv, label);
	cout << "written file " << fname_csv << " of size "
		<< Fio.file_size(fname_csv) << endl;


	if (f_v) {
		cout << "algebra_global::do_norm done" << endl;
	}
}

void algebra_global::do_cheat_sheet_GF(
		field_theory::finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::do_cheat_sheet_GF q=" << F->q << endl;
	}

	string fname;
	string author;
	string title;
	string extra_praeamble;

	fname = F->label + ".tex";

	title = "Cheat Sheet $" + F->label_tex + "$";





	{
		ofstream ost(fname);


		l1_interfaces::latex_interface L;


		L.head(ost,
				false /* f_book*/, true /* f_title */,
				title, author,
				false /* f_toc */, false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);


		F->Io->cheat_sheet(ost, verbose_level);

		F->Io->cheat_sheet_main_table(ost, verbose_level);

		F->Io->cheat_sheet_addition_table(ost, verbose_level);

		F->Io->cheat_sheet_multiplication_table(ost, verbose_level);

		F->Io->cheat_sheet_power_table(ost, true, verbose_level);

		F->Io->cheat_sheet_power_table(ost, false, verbose_level);





		L.foot(ost);
	}

	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "algebra_global::do_cheat_sheet_GF q=" << F->q << " done" << endl;
	}
}


void algebra_global::export_tables(
		field_theory::finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::export_tables q=" << F->q << endl;
	}

	F->Io->addition_table_save_csv(verbose_level);

	F->Io->multiplication_table_save_csv(verbose_level);

	F->Io->addition_table_reordered_save_csv(verbose_level);

	F->Io->multiplication_table_reordered_save_csv(verbose_level);


	if (f_v) {
		cout << "algebra_global::export_tables done" << endl;
	}
}


void algebra_global::do_cheat_sheet_ring(
		ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::do_cheat_sheet_ring" << endl;
	}

	string fname;
	string author;
	string title;
	string extra_praeamble;

	fname = "cheat_sheet_ring.tex";

	title = "Cheat Sheet Ring";



	{
		ofstream ost(fname);


		l1_interfaces::latex_interface L;


		L.head(ost,
				false /* f_book*/, true /* f_title */,
				title, author,
				false /* f_toc */, false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);

		HPD->print_latex(ost);

		HPD->print_monomial_ordering_latex(ost);

		L.foot(ost);
	}

	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "algebra_global::do_cheat_sheet_ring done" << endl;
	}
}




void algebra_global::gl_random_matrix(
		field_theory::finite_field *F, int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	int *M2;
	ring_theory::unipoly_object char_poly;

	if (f_v) {
		cout << "algebra_global::gl_random_matrix" << endl;
		}
	M = NEW_int(k * k);
	M2 = NEW_int(k * k);

	F->Linear_algebra->random_invertible_matrix(M, k, verbose_level - 2);

	cout << "Random invertible matrix:" << endl;
	Int_matrix_print(M, k, k);


	{
		ring_theory::unipoly_domain U(F);



		U.create_object_by_rank(char_poly, 0, verbose_level);

		U.characteristic_polynomial(M, k, char_poly, verbose_level - 2);

		cout << "The characteristic polynomial is ";
		U.print_object(char_poly, cout);
		cout << endl;

		U.substitute_matrix_in_polynomial(
				char_poly, M, M2, k, verbose_level);
		cout << "After substitution, the matrix is " << endl;
		Int_matrix_print(M2, k, k);

		U.delete_object(char_poly);

	}
	FREE_int(M);
	FREE_int(M2);

}




void algebra_global::apply_Walsh_Hadamard_transform(
		field_theory::finite_field *F,
		std::string &fname_csv_in, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::apply_Walsh_Hadamard_transform" << endl;
	}


	combinatorics::boolean_function_domain *BF;

	BF = NEW_OBJECT(combinatorics::boolean_function_domain);

	BF->init(F, n, verbose_level);


	orbiter_kernel_system::file_io Fio;
	int *M;
	int m, nb_cols;
	int len;
	string fname_csv_out;
	data_structures::string_tools ST;

	fname_csv_out = fname_csv_in;
	ST.chop_off_extension(fname_csv_out);
	fname_csv_out += "_transformed.csv";

	Fio.Csv_file_support->int_matrix_read_csv(fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	if (len != BF->Q) {
		cout << "algebra_global::apply_Walsh_Hadamard_transform "
				"len != BF->Q" << endl;
		exit(1);
	}
	BF->raise(M, BF->F);

	BF->apply_Walsh_transform(BF->F, BF->T);

	cout << " : ";
	Int_vec_print(cout, BF->T, BF->Q);
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
	Fio.Csv_file_support->int_matrix_write_csv(fname_csv_out, BF->T, m, nb_cols);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);
	FREE_OBJECT(BF);

	if (f_v) {
		cout << "algebra_global::apply_Walsh_Hadamard_transform done" << endl;
	}
}

void algebra_global::algebraic_normal_form(
		field_theory::finite_field *F,
		int n,
		int *func, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::algebraic_normal_form" << endl;
	}


	F->f_print_as_exponentials = false;


	combinatorics::polynomial_function_domain *PF;

	PF = NEW_OBJECT(combinatorics::polynomial_function_domain);

	if (f_v) {
		cout << "algebra_global::algebraic_normal_form "
				"before PF->init" << endl;
	}
	PF->init(F, n, verbose_level);
	if (f_v) {
		cout << "algebra_global::algebraic_normal_form "
				"after PF->init" << endl;
	}


	if (len != PF->Q) {
		cout << "algebra_global::algebraic_normal_form "
				"len should be " << PF->Q << endl;
		exit(1);
	}

#if 0
	orbiter_kernel_system::file_io Fio;
	int *M;
	int m, nb_cols;
	int len;
	string fname_csv_out;
	data_structures::string_tools ST;

	fname_csv_out.assign(fname_csv_in);
	ST.chop_off_extension(fname_csv_out);
	fname_csv_out += "_alg_normal_form.csv";

	Fio.int_matrix_read_csv(fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	if (len != BF->Q) {
		cout << "algebra_global::algebraic_normal_form len != BF->Q" << endl;
		exit(1);
	}
#endif

	int *coeff;
	int nb_coeff;

	nb_coeff = PF->Poly[PF->max_degree].get_nb_monomials();

	coeff = NEW_int(nb_coeff);

	if (f_v) {
		cout << "algebra_global::algebraic_normal_form "
				"before PF->compute_polynomial_representation" << endl;
	}
	PF->compute_polynomial_representation(func, coeff, verbose_level);
	if (f_v) {
		cout << "algebra_global::algebraic_normal_form "
				"after PF->compute_polynomial_representation" << endl;
	}

	cout << "algebraic normal form:" << endl;
	PF->Poly[PF->max_degree].print_equation(cout, coeff);
	cout << endl;

	cout << "algebraic normal form in tex:" << endl;
	PF->Poly[PF->max_degree].print_equation_tex(cout, coeff);
	cout << endl;

	cout << "algebraic normal form in numerical form:" << endl;
	PF->Poly[PF->max_degree].print_equation_numerical(cout, coeff);
	cout << endl;

#if 0

	Fio.int_matrix_write_csv(fname_csv_out, coeff, 1, nb_coeff);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);
#endif

	//FREE_OBJECT(PF);
	// there is a memory error in PF

	if (f_v) {
		cout << "algebra_global::algebraic_normal_form done" << endl;
	}
}

void algebra_global::algebraic_normal_form_of_boolean_function(
		field_theory::finite_field *F,
		std::string &fname_csv_in, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::algebraic_normal_form_of_boolean_function" << endl;
	}


	combinatorics::boolean_function_domain *BF;

	BF = NEW_OBJECT(combinatorics::boolean_function_domain);

	if (f_v) {
		cout << "algebra_global::algebraic_normal_form_of_boolean_function "
				"before BF->init" << endl;
	}
	BF->init(F, n, verbose_level);
	if (f_v) {
		cout << "algebra_global::algebraic_normal_form_of_boolean_function "
				"after BF->init" << endl;
	}


	orbiter_kernel_system::file_io Fio;
	int *M;
	int m, nb_cols;
	int len;
	string fname_csv_out;
	data_structures::string_tools ST;

	fname_csv_out.assign(fname_csv_in);
	ST.chop_off_extension(fname_csv_out);
	fname_csv_out += "_alg_normal_form.csv";

	Fio.Csv_file_support->int_matrix_read_csv(
			fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	if (len != BF->Q) {
		cout << "algebra_global::algebraic_normal_form_of_boolean_function "
				"len != BF->Q" << endl;
		exit(1);
	}

	int *coeff;
	int nb_coeff;

	nb_coeff = BF->Poly[n].get_nb_monomials();

	coeff = NEW_int(nb_coeff);

	if (f_v) {
		cout << "algebra_global::algebraic_normal_form_of_boolean_function "
				"before BF->compute_polynomial_representation" << endl;
	}
	BF->compute_polynomial_representation(M, coeff, verbose_level);
	if (f_v) {
		cout << "algebra_global::algebraic_normal_form_of_boolean_function "
				"after BF->compute_polynomial_representation" << endl;
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



	Fio.Csv_file_support->int_matrix_write_csv(
			fname_csv_out, coeff, 1, nb_coeff);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);
	FREE_OBJECT(BF);

	if (f_v) {
		cout << "algebra_global::algebraic_normal_form_of_boolean_function done" << endl;
	}
}

void algebra_global::apply_trace_function(
		field_theory::finite_field *F,
		std::string &fname_csv_in, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::apply_trace_function" << endl;
	}


	orbiter_kernel_system::file_io Fio;
	int *M;
	int m, nb_cols;
	int len, i;
	string fname_csv_out;
	data_structures::string_tools ST;

	fname_csv_out = fname_csv_in;
	ST.chop_off_extension(fname_csv_out);
	fname_csv_out += "_trace.csv";

	Fio.Csv_file_support->int_matrix_read_csv(
			fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	for (i = 0; i < len; i++) {
		M[i] = F->absolute_trace(M[i]);
	}
	Fio.Csv_file_support->int_matrix_write_csv(fname_csv_out, M, m, nb_cols);

	FREE_int(M);

	if (f_v) {
		cout << "algebra_global::apply_trace_function done" << endl;
	}
}

void algebra_global::apply_power_function(
		field_theory::finite_field *F,
		std::string &fname_csv_in, long int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::apply_power_function" << endl;
	}


	orbiter_kernel_system::file_io Fio;
	int *M;
	int m, nb_cols;
	int len, i;
	string fname_csv_out;
	data_structures::string_tools ST;

	fname_csv_out.assign(fname_csv_in);
	ST.chop_off_extension(fname_csv_out);

	fname_csv_out += "_power_" + std::to_string(d) + ".csv";

	Fio.Csv_file_support->int_matrix_read_csv(
			fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	for (i = 0; i < len; i++) {
		M[i] = F->power(M[i], d);
	}
	Fio.Csv_file_support->int_matrix_write_csv(fname_csv_out, M, m, nb_cols);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);

	if (f_v) {
		cout << "algebra_global::apply_power_function done" << endl;
	}
}

void algebra_global::identity_function(
		field_theory::finite_field *F,
		std::string &fname_csv_out, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::identity_function" << endl;
	}


	orbiter_kernel_system::file_io Fio;
	int *M;
	int i;

	M = NEW_int(F->q);
	for (i = 0; i < F->q; i++) {
		M[i] = i;
	}
	Fio.Csv_file_support->int_matrix_write_csv(fname_csv_out, M, 1, F->q);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);

	if (f_v) {
		cout << "algebra_global::identity_function done" << endl;
	}
}


void algebra_global::Walsh_matrix(
		field_theory::finite_field *F,
		int n, int *&W,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Q;
	int *v;
	int *w;
	int *W01;
	int i, j, a;
	geometry::geometry_global Gg;

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
	string fname_csv;
	orbiter_kernel_system::file_io Fio;

	fname_csv = "Walsh_pm_" + std::to_string(n) + ".csv";
	Fio.Csv_file_support->int_matrix_write_csv(
			fname_csv, W, Q, Q);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	fname_csv = "Walsh_01_" + std::to_string(n) + ".csv";
	Fio.Csv_file_support->int_matrix_write_csv(
			fname_csv, W01, Q, Q);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	FREE_int(v);
	FREE_int(w);
	FREE_int(W01);
	if (f_v) {
		cout << "algebra_global::Walsh_matrix done" << endl;
	}
}

void algebra_global::Vandermonde_matrix(
		field_theory::finite_field *F,
		int *&W, int *&W_inv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q;
	int i, j, a;
	geometry::geometry_global Gg;

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
		cout << "algebra_global::Vandermonde_matrix "
				"before invert_matrix" << endl;
	}
	F->Linear_algebra->invert_matrix(W, W_inv, q, verbose_level);
	if (f_v) {
		cout << "algebra_global::Vandermonde_matrix "
				"after invert_matrix" << endl;
	}

	string fname_csv;
	orbiter_kernel_system::file_io Fio;

	fname_csv = "Vandermonde_q" + std::to_string(q) + ".csv";
	Fio.Csv_file_support->int_matrix_write_csv(
			fname_csv, W, q, q);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	fname_csv = "Vandermonde_q" + std::to_string(q) + "_inv.csv";
	Fio.Csv_file_support->int_matrix_write_csv(
			fname_csv, W_inv, q, q);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	if (f_v) {
		cout << "algebra_global::Vandermonde_matrix done" << endl;
	}
}


void algebra_global::O4_isomorphism_4to2(
		field_theory::finite_field *F,
		int *At, int *As, int &f_switch, int *B,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a, b, c, d, e, f, g, h;
	int ev, fv;
	int P[4], Q[4], R[4], S[4];
	long int Rx, Ry, Sx, Sy;
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
		Int_vec_print(cout, P, 4);
		cout << endl;
	}
	O4_grid_coordinates_unrank(F, Q[0], Q[1], Q[2], Q[3],
			1, 0, verbose_level);
	if (f_vv) {
		cout << "grid point (1,0) = ";
		Int_vec_print(cout, Q, 4);
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
		f_switch = false;
	}
	else {
		f_switch = true;
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
		Int_vec_print_integer_matrix_width(cout, At, 2, 2, 2, F->log10_of_q);
		cout << "As:" << endl;
		Int_vec_print_integer_matrix_width(cout, As, 2, 2, 2, F->log10_of_q);
	}

}

void algebra_global::O4_isomorphism_2to4(
		field_theory::finite_field *F,
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

void algebra_global::O4_grid_coordinates_rank(
		field_theory::finite_field *F,
		int x1, int x2, int x3, int x4, long int &grid_x, long int &grid_y,
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
	F->Projective_space_basic->PG_element_normalize_from_front(v, 1, 2);
	F->Projective_space_basic->PG_element_normalize_from_front(w, 1, 2);
	if (f_v) {
		Int_vec_print(cout, v, 2);
		Int_vec_print(cout, w, 2);
		cout << endl;
	}

	F->Projective_space_basic->PG_element_rank_modified(v, 1, 2, grid_x);
	F->Projective_space_basic->PG_element_rank_modified(w, 1, 2, grid_y);
}

void algebra_global::O4_grid_coordinates_unrank(
		field_theory::finite_field *F,
		int &x1, int &x2, int &x3, int &x4,
		int grid_x, int grid_y,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, c, d;
	int v[2], w[2];

	F->Projective_space_basic->PG_element_unrank_modified(v, 1, 2, grid_x);
	F->Projective_space_basic->PG_element_unrank_modified(w, 1, 2, grid_y);
	F->Projective_space_basic->PG_element_normalize_from_front(v, 1, 2);
	F->Projective_space_basic->PG_element_normalize_from_front(w, 1, 2);
	if (f_v) {
		Int_vec_print(cout, v, 2);
		Int_vec_print(cout, w, 2);
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

void algebra_global::O4_find_tangent_plane(
		field_theory::finite_field *F,
		int pt_x1, int pt_x2, int pt_x3, int pt_x4,
		int *tangent_plane,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int A[4];
	int C[3 * 4];
	int size, x, y, z;
	long int xx, yy;
	int zz, h, k;
	int x1, x2, x3, x4;
	int y1, y2, y3, y4;
	int f_special = false;
	int f_complete = false;
	int base_cols[4];
	int f_P = false;
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
				F->Projective_space_basic->PG_element_unrank_modified(vec2, 1, 2, k);
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
		Int_vec_print(cout, secants1, nb_secants);
		cout << endl;
		Int_vec_print(cout, secants2, nb_secants);
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
		Int_vec_print(cout, complement, nb_complement);
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
		Int_vec_print_integer_matrix_width(cout, T, rk, 4, 4, F->log10_of_q);
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

void algebra_global::create_Nth_roots_and_write_report(
		field_theory::finite_field *F,
		int n, int verbose_level)
{
	field_theory::nth_roots *Nth;

	Nth = NEW_OBJECT(field_theory::nth_roots);

	Nth->init(F, n, verbose_level);

	orbiter_kernel_system::file_io Fio;
	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = "Nth_roots_q" + std::to_string(F->q) + "_n" + std::to_string(n) + ".tex";
		title = "Nth roots";




		{
			ofstream ost(fname);
			number_theory::number_theory_domain NT;



			l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			Nth->report(ost, verbose_level);

			ost << "\\begin{verbatim}" << endl;
			Nth->print_irreducible_polynomials_as_makefile_variables(ost, verbose_level);
			ost << "\\end{verbatim}" << endl;

			L.foot(ost);


		}

		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;

	}

}





}}}

