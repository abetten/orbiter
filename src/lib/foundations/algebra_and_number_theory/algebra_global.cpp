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


char *algebra_global::search_for_primitive_polynomial_of_given_degree(
		int p, int degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field Fp;

	if (f_v) {
		cout << "algebra_global::search_for_primitive_polynomial_of_given_degree" << endl;
	}
	Fp.finite_field_init(p, 0 /*verbose_level*/);
	unipoly_domain FX(&Fp);

	unipoly_object m;
	longinteger_object rk;

	FX.create_object_by_rank(m, 0, __FILE__, __LINE__, verbose_level);

	if (f_v) {
		cout << "search_for_primitive_polynomial_of_given_degree "
				"p=" << p << " degree=" << degree << endl;
	}
	FX.get_a_primitive_polynomial(m, degree, verbose_level - 1);
	FX.rank_longinteger(m, rk);

	char *s;
	int i, j;
	if (f_v) {
		cout << "found a polynomial. It's rank is " << rk << endl;
	}

	s = NEW_char(rk.len() + 1);
	for (i = rk.len() - 1, j = 0; i >= 0; i--, j++) {
		s[j] = '0' + rk.rep()[i];
	}
	s[rk.len()] = 0;

	if (f_v) {
		cout << "created string " << s << endl;
	}

	if (f_v) {
		cout << "algebra_global::search_for_primitive_polynomial_of_given_degree done" << endl;
	}
	return s;
}


void algebra_global::search_for_primitive_polynomials(
		int p_min, int p_max, int n_min, int n_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d, q;
	number_theory_domain NT;


	longinteger_f_print_scientific = FALSE;


	if (f_v) {
		cout << "algebra_global::search_for_primitive_polynomials "
				"p_min=" << p_min << " p_max=" << p_max
				<< " n_min=" << n_min << " n_max=" << n_max << endl;
	}
	for (q = p_min; q <= p_max; q++) {


		if (!NT.is_prime_power(q)) {
			continue;
		}

		if (f_v) {
			cout << "algebra_global::search_for_primitive_polynomials "
					"considering the coefficient field of order " << q << endl;
		}

		{
			finite_field Fq;
			Fq.finite_field_init(q, 0 /*verbose_level*/);
			unipoly_domain FX(&Fq);

			unipoly_object m;
			longinteger_object rk;

			FX.create_object_by_rank(m, 0, __FILE__, __LINE__, verbose_level);

			for (d = n_min; d <= n_max; d++) {
				if (f_v) {
					cout << "d=" << d << endl;
				}
				FX.get_a_primitive_polynomial(m, d, verbose_level);
				FX.rank_longinteger(m, rk);
				//cout << d << " : " << rk << " : ";
				cout << "\"" << rk << "\", // ";
				FX.print_object(m, cout);
				cout << endl;
			}
			if (f_v) {
				cout << "algebra_global::search_for_primitive_polynomials "
						"before FX.delete_object(m)" << endl;
			}
			FX.delete_object(m);
			if (f_v) {
				cout << "algebra_global::search_for_primitive_polynomials "
						"after FX.delete_object(m)" << endl;
			}
		}
	}
	if (f_v) {
		cout << "algebra_global::search_for_primitive_polynomials done" << endl;
	}
}


void algebra_global::factor_cyclotomic(int n, int q, int d,
	int *coeffs, int f_poly, std::string &poly, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int p, e, m, i, j, Q, a, b, c, cv, ccv, t, r1, r2, len;
	int field_degree, subgroup_index;
	finite_field FQ;
	finite_field Fq;
	number_theory_domain NT;

	NT.factor_prime_power(q, p, e);
	if (f_v) {
		cout << "factor_cyclotomic q=" << q << " p=" << q
			<< " e=" << e << " n=" << n << endl;
	}
	m = NT.order_mod_p(q, n);
	if (f_v) {
		cout << "order mod q is m=" << m << endl;
	}
	field_degree = e * m;
	Q = NT.i_power_j(p, field_degree);


	if (f_poly) {
		Fq.init_override_polynomial(q, poly, verbose_level - 1);
	}
	else {
		Fq.finite_field_init(q, verbose_level - 2);
	}
	FQ.finite_field_init(Q, verbose_level - 2);

	FQ.compute_subfields(verbose_level);

	subgroup_index = (Q - 1) / (q - 1);

	unipoly_domain FQX(&FQ);
	unipoly_object quo, rem, h, Xma;

	FQX.create_object_of_degree(h, d);

	if (e > 1) {
		cout << "embedding the coefficients into the larger field" << endl;
		for (i = 0; i <= d; i++) {
			c = coeffs[i];
			if (c == 0) {
				t = 0;
			}
			else {
				a = Fq.log_alpha(c);
				t = a * subgroup_index;
				t = FQ.alpha_power(t);
			}
			FQX.s_i(h, i) = t;
		}
	}
	else {
		for (i = 0; i <= d; i++) {
			FQX.s_i(h, i) = coeffs[i];
		}
	}

	if (f_v) {
		cout << "the polynomial is: ";
		FQX.print_object(h, cout);
		cout << endl;
	}


	FQX.create_object_of_degree(quo, d);
	FQX.create_object_of_degree(rem, d);

	int *roots;
	int *roots2;
	int nb_roots = 0;
	int beta = (Q - 1) / n, Beta;

	Beta = FQ.alpha_power(beta);

	if (f_v) {
		cout << "the primitive n-th root of unity we choose "
				"is beta = alpha^" << beta << " = " << Beta << endl;
	}

	roots = NEW_int(n);
	roots2 = NEW_int(n);
	for (a = 0; a < n; a++) {
		FQX.create_object_of_degree(Xma, 1);
		t = FQ.power(Beta, a);
		FQX.s_i(Xma, 0) = FQ.negate(t);
		FQX.s_i(Xma, 1) = 1;
		FQX.division_with_remainder(h, Xma, quo, rem, 0);
		b = FQX.s_i(rem, 0);
		if (b == 0) {
			cout << "zero Beta^" << a << " log "
				<< FQ.log_alpha(t) << endl;
			roots[nb_roots++] = a;
		}
	}

	exit(1);

	longinteger_domain D;
	longinteger_object C, N, A, B, G, U, V;
	sorting Sorting;

	for (c = 0; c < n; c++) {
		if (NT.gcd_lint(c, n) != 1)
			continue;
		C.create(c, __FILE__, __LINE__);
		N.create(n, __FILE__, __LINE__);
		D.extended_gcd(C, N, G, U, V, FALSE);
		cv = U.as_int();
		ccv= c * cv;
		cout << c << " : " << cv << " : ";
		if (ccv < 0) {
			if ((-ccv % n) != n - 1) {
				cout << "error: c=" << c << " cv=" << cv << endl;
				exit(1);
			}
		}
		else if ((ccv % n) != 1) {
			cout << "error: c=" << c << " cv=" << cv << endl;
			exit(1);
		}
		for (i = 0; i < nb_roots; i++) {
			roots2[i] = (cv * roots[i]) % n;
			while (roots2[i] < 0) {
				roots2[i] += n;
			}
		}
		Sorting.int_vec_quicksort_increasingly(roots2, nb_roots);
		t = 0;
		for (i = 0; i < nb_roots; i++) {
			r1 = roots2[i];
			for (j = i + 1; j < i + nb_roots; j++) {
				r2 = roots2[j % nb_roots];
				if (r2 != r1 + 1) {
					break;
				}
			}
			len = j - i - 1;
			t = MAXIMUM(t, len);
		}
		for (i = 0; i < nb_roots; i++) {
			cout << roots2[i] << " ";
		}
		cout << " : " << t << endl;
	}
}

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
				phi_g = eulers_totient_function(g, 1);
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


int algebra_global::eulers_totient_function(int n, int verbose_level)
{
	int nb_primes, *primes, *exponents;
	int i, p, e;
	longinteger_domain D;
	longinteger_object N, R, A, B, C;

	N.create(n, __FILE__, __LINE__);
	D.factor(N, nb_primes, primes, exponents, verbose_level);
	R.create(1, __FILE__, __LINE__);
	for (i = 0; i < nb_primes; i++) {
		p = primes[i];
		e = exponents[i];
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
	}
	return R.as_int();
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
		cout << "d=" << d << " q=" << q << endl;
	}
	Theta.create(q, __FILE__, __LINE__);
	M1.create(-1, __FILE__, __LINE__);
	Qm1.create(q-1, __FILE__, __LINE__);
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

void algebra_global::formula(int d, int q, longinteger_object &Rdq, int verbose_level)
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
	Theta.create(q, __FILE__, __LINE__);
	M1.create(-1, __FILE__, __LINE__);
	Qm1.create(q - 1, __FILE__, __LINE__);
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

void algebra_global::gl_random_matrix(int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 1);
	int *M;
	int *M2;
	finite_field F;
	unipoly_object char_poly;

	if (f_v) {
		cout << "gl_random_matrix" << endl;
		}
	F.finite_field_init(q, 0 /*verbose_level*/);
	M = NEW_int(k * k);
	M2 = NEW_int(k * k);

	F.random_invertible_matrix(M, k, verbose_level - 2);

	cout << "Random invertible matrix:" << endl;
	int_matrix_print(M, k, k);


	{
		unipoly_domain U(&F);



		U.create_object_by_rank(char_poly, 0, __FILE__, __LINE__, verbose_level);

		U.characteristic_polynomial(M, k, char_poly, verbose_level - 2);

		cout << "The characteristic polynomial is ";
		U.print_object(char_poly, cout);
		cout << endl;

		U.substitute_matrix_in_polynomial(char_poly, M, M2, k, verbose_level);
		cout << "After substitution, the matrix is " << endl;
		int_matrix_print(M2, k, k);

		U.delete_object(char_poly);

	}
	FREE_int(M);
	FREE_int(M2);

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



int algebra_global::PHG_element_normalize(finite_ring &R,
		int *v, int stride, int len)
// last unit element made one
{
	int i, j, a;

	if (!R.f_chain_ring) {
		cout << "algebra_global::PHG_element_normalize not a chain ring" << endl;
		exit(1);
	}
	for (i = len - 1; i >= 0; i--) {
		a = v[i * stride];
		if (R.is_unit(a)) {
			if (a == 1)
				return i;
			a = R.inverse(a);
			for (j = len - 1; j >= 0; j--) {
				v[j * stride] = R.mult(v[j * stride], a);
				}
			return i;
			}
		}
	cout << "algebra_global::PHG_element_normalize "
			"vector is not free" << endl;
	exit(1);
}


int algebra_global::PHG_element_normalize_from_front(finite_ring &R,
		int *v, int stride, int len)
// first non unit element made one
{
	int i, j, a;

	if (!R.f_chain_ring) {
		cout << "algebra_global::PHG_element_normalize_from_front not a chain ring" << endl;
		exit(1);
	}
	for (i = 0; i < len; i++) {
		a = v[i * stride];
		if (R.is_unit(a)) {
			if (a == 1)
				return i;
			a = R.inverse(a);
			for (j = 0; j < len; j++) {
				v[j * stride] = R.mult(v[j * stride], a);
				}
			return i;
			}
		}
	cout << "algebra_global::PHG_element_normalize_from_front "
			"vector is not free" << endl;
	exit(1);
}

int algebra_global::PHG_element_rank(finite_ring &R,
		int *v, int stride, int len)
{
	long int i, j, idx, a, b, r1, r2, rk, N;
	int f_v = FALSE;
	int *w;
	int *embedding;
	geometry_global Gg;

	if (!R.f_chain_ring) {
		cout << "algebra_global::PHG_element_rank not a chain ring" << endl;
		exit(1);
	}
	if (len <= 0) {
		cout << "algebra_global::PHG_element_rank len <= 0" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "the vector before normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	idx = PHG_element_normalize(R, v, stride, len);
	if (f_v) {
		cout << "the vector after normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	w = NEW_int(len - 1);
	embedding = NEW_int(len - 1);
	for (i = 0, j = 0; i < len - 1; i++, j++) {
		if (i == idx) {
			j++;
			}
		embedding[i] = j;
		}
	for (i = 0; i < len - 1; i++) {
		w[i] = v[embedding[i] * stride];
		}
	for (i = 0; i < len - 1; i++) {
		a = w[i];
		b = a % R.get_p();
		v[embedding[i] * stride] = b;
		w[i] = (a - b) / R.get_p();
		}
	if (f_v) {
		cout << "w=";
		int_vec_print(cout, w, len - 1);
		cout << endl;
		}
	r1 = Gg.AG_element_rank(R.get_e(), w, 1, len - 1);
	R.get_Fp()->PG_element_rank_modified_lint(v, stride, len, r2);

	N = Gg.nb_PG_elements(len - 1, R.get_p());
	rk = r1 * N + r2;

	FREE_int(w);
	FREE_int(embedding);

	return rk;
}

void algebra_global::PHG_element_unrank(finite_ring &R,
		int *v, int stride, int len, int rk)
{
	int i, j, idx, r1, r2, N;
	int f_v = FALSE;
	int *w;
	int *embedding;
	geometry_global Gg;

	if (!R.f_chain_ring) {
		cout << "algebra_global::PHG_element_unrank not a chain ring" << endl;
		exit(1);
	}
	if (len <= 0) {
		cout << "algebra_global::PHG_element_unrank len <= 0" << endl;
		exit(1);
	}

	w = NEW_int(len - 1);
	embedding = NEW_int(len - 1);

	N = Gg.nb_PG_elements(len - 1, R.get_p());
	r2 = rk % N;
	r1 = (rk - r2) / N;

	Gg.AG_element_unrank(R.get_e(), w, 1, len - 1, r1);
	R.get_Fp()->PG_element_unrank_modified(v, stride, len, r2);

	if (f_v) {
		cout << "w=";
		int_vec_print(cout, w, len - 1);
		cout << endl;
	}

	idx = PHG_element_normalize(R, v, stride, len);
	for (i = 0, j = 0; i < len - 1; i++, j++) {
		if (i == idx) {
			j++;
		}
		embedding[i] = j;
	}

	for (i = 0; i < len - 1; i++) {
		v[embedding[i] * stride] += w[i] * R.get_p();
	}



	FREE_int(w);
	FREE_int(embedding);

}

int algebra_global::nb_PHG_elements(int n, finite_ring &R)
{
	int N1, N2;
	geometry_global Gg;

	if (!R.f_chain_ring) {
		cout << "algebra_global::nb_PHG_elements not a chain ring" << endl;
		exit(1);
	}
	N1 = Gg.nb_PG_elements(n, R.get_p());
	N2 = Gg.nb_AG_elements(n, R.get_e());
	return N1 * N2;
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
	l = nb_PHG_elements(n, R);
	for (i = 0; i < l; i++) {
		PHG_element_unrank(R, v, 1, n + 1, i);
		cout << i << " : ";
		for (j = 0; j < n + 1; j++) {
			cout << v[j] << " ";
		}
		a = PHG_element_rank(R, v, 1, n + 1);
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

	GFp.finite_field_init(p, verbose_level);
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

	Fq.finite_field_init(q, verbose_level);
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

const char *algebra_global::get_primitive_polynomial(int p, int e, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int idx;
	char *s;
	sorting Sorting;

	if (!Sorting.int_vec_search(finitefield_primes, finitefield_nb_primes, p, idx)) {
		cout << "I don't have prime " << p << " in the tables" << endl;

		exit(1);


		cout << "searching for a polynomial of degree " << e << endl;

		algebra_global AG;

		s = AG.search_for_primitive_polynomial_of_given_degree(p, e, verbose_level);
		cout << "the search came up with a polynomial of degree " << e << ", coded as " << s << endl;
		return s;
	}
#if 0
	for (idx = 0; idx < finitefield_nb_primes; idx++) {
		if (finitefield_primes[idx] == p)
			break;
	}
	if (idx == finitefield_nb_primes) {
		cout << "get_primitive_polynomial() couldn't find prime " << p << endl;
		exit(1);
	}
#endif
	if (e > finitefield_largest_degree_irreducible_polynomial[idx]) {
		cout << "get_primitive_polynomial() I do not have a polynomial\n";
		cout << "of that degree over that field" << endl;
		cout << "requested: degree " << e << " polynomial over GF(" << p << ")" << endl;
		exit(1);
	}
	const char *m = finitefield_primitive_polynomial[idx][e - 2];
	if (strlen(m) == 0) {
		cout << "get_primitive_polynomial() I do not have a polynomial\n";
		cout << "of that degree over that field" << endl;
		cout << "requested: degree " << e << " polynomial over GF(" << p << ")" << endl;
		exit(1);
	}
	return m;
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
	combinatorics_domain D;
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
	combinatorics_domain D;
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
	longinteger_domain D;
	longinteger_object a, b, one;
	int nb_solovay_strassen_tests = 100;
	int f_miller_rabin_test = TRUE;

	one.create(1, __FILE__, __LINE__);
	a.create(197659, __FILE__, __LINE__);
	D.find_probable_prime_above(a, nb_solovay_strassen_tests,
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

void algebra_global::make_all_irreducible_polynomials_of_degree_d(
		finite_field *F, int d, std::vector<std::vector<int> > &Table,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int cnt;
	number_theory_domain NT;

	if (f_v) {
		cout << "algebra_global::make_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << F->q << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}

#if 0
	cnt = count_all_irreducible_polynomials_of_degree_d(F, d, verbose_level - 2);

	if (f_v) {
		cout << "algebra_global::make_all_irreducible_polynomials_of_degree_d "
				"cnt = " << cnt << endl;
	}

	nb = cnt;

	Table = NEW_int(nb * (d + 1));
#endif

	//NT.factor_prime_power(F->q, p, e);

	if (f_v) {
		cout << "algebra_global::make_all_irreducible_polynomials_of_degree_d "
				" q=" << F->q << " p=" << F->p << " e=" << F->e << endl;
	}

	unipoly_domain FX(F);

	const char *poly;

	poly = get_primitive_polynomial(F->q, d, 0 /* verbose_level */);

	if (f_v) {
		cout << "algebra_global::make_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial is " << poly << endl;
	}

	unipoly_object m;
	unipoly_object g;
	unipoly_object minpol;
	combinatorics_domain Combi;


	FX.create_object_by_rank_string(m, poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "algebra_global::make_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial m = ";
		FX.print_object(m, cout);
		cout << endl;
	}

	FX.create_object_by_rank(g, 0, __FILE__, __LINE__, verbose_level);
	FX.create_object_by_rank(minpol, 0, __FILE__, __LINE__, verbose_level);

	int *Frobenius;
	int *Normal_basis;
	int *v;
	int *w;

	//Frobenius = NEW_int(d * d);
	Normal_basis = NEW_int(d * d);
	v = NEW_int(d);
	w = NEW_int(d);

	if (f_v) {
		cout << "algebra_global::make_all_irreducible_polynomials_of_degree_d "
				"before FX.Frobenius_matrix" << endl;
	}
	FX.Frobenius_matrix(Frobenius, m, verbose_level - 2);
	if (f_v) {
		cout << "algebra_global::make_all_irreducible_polynomials_of_degree_d "
				"Frobenius_matrix = " << endl;
		int_matrix_print(Frobenius, d, d);
		cout << endl;
	}

	if (f_v) {
		cout << "algebra_global::make_all_irreducible_polynomials_of_degree_d "
				"before compute_normal_basis" << endl;
	}
	FX.compute_normal_basis(d, Normal_basis, Frobenius, verbose_level - 1);

	if (f_v) {
		cout << "algebra_global::make_all_irreducible_polynomials_of_degree_d "
				"Normal_basis = " << endl;
		int_matrix_print(Normal_basis, d, d);
		cout << endl;
	}

	cnt = 0;

	Combi.int_vec_first_regular_word(v, d, F->q);
	while (TRUE) {
		if (f_vv) {
			cout << "algebra_global::make_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : v = ";
			int_vec_print(cout, v, d);
			cout << endl;
		}

		F->mult_vector_from_the_right(Normal_basis, v, w, d, d);
		if (f_vv) {
			cout << "algebra_global::make_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : w = ";
			int_vec_print(cout, w, d);
			cout << endl;
		}

		FX.delete_object(g);
		FX.create_object_of_degree(g, d - 1);
		for (i = 0; i < d; i++) {
			((int *) g)[1 + i] = w[i];
		}

		FX.minimum_polynomial_extension_field(g, m, minpol, d, Frobenius, verbose_level - 3);
		if (f_vv) {
			cout << "algebra_global::make_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : v = ";
			int_vec_print(cout, v, d);
			cout << " irreducible polynomial = ";
			FX.print_object(minpol, cout);
			cout << endl;
		}



		std::vector<int> T;

		for (i = 0; i <= d; i++) {
			T.push_back(((int *)minpol)[1 + i]);
			//Table[cnt * (d + 1) + i] = ((int *)minpol)[1 + i];
		}
		Table.push_back(T);


		cnt++;


		if (!Combi.int_vec_next_regular_word(v, d, F->q)) {
			break;
		}

	}

	if (f_v) {
		cout << "algebra_global::make_all_irreducible_polynomials_of_degree_d "
				"there are " << cnt
				<< " irreducible polynomials "
				"of degree " << d << " over " << "F_" << F->q << endl;
	}

	FREE_int(Frobenius);
	FREE_int(Normal_basis);
	FREE_int(v);
	FREE_int(w);
	FX.delete_object(m);
	FX.delete_object(g);
	FX.delete_object(minpol);


	if (f_v) {
		cout << "algebra_global::make_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << F->q << " done" << endl;
	}
}

int algebra_global::count_all_irreducible_polynomials_of_degree_d(finite_field *F,
		int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int cnt;
	number_theory_domain NT;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << F->q << endl;
	}


	if (f_v) {
		cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d " << endl;
	}

	if (f_v) {
		cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
				"p=" << F->p << " e=" << F->e << endl;
	}
	if (F->e > 1) {
		cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
				"e=" << F->e << " is greater than one" << endl;
	}

	unipoly_domain FX(F);

	const char *poly;

	poly = get_primitive_polynomial(F->q, d, 0 /* verbose_level */);

	unipoly_object m;
	unipoly_object g;
	unipoly_object minpol;


	FX.create_object_by_rank_string(m, poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial m = ";
		FX.print_object(m, cout);
		cout << endl;
	}

	FX.create_object_by_rank(g, 0, __FILE__, __LINE__, verbose_level);
	FX.create_object_by_rank(minpol, 0, __FILE__, __LINE__, verbose_level);

	int *Frobenius;
	//int *F2;
	int *Normal_basis;
	int *v;
	int *w;

	//Frobenius = NEW_int(d * d);
	//F2 = NEW_int(d * d);
	Normal_basis = NEW_int(d * d);
	v = NEW_int(d);
	w = NEW_int(d);

	FX.Frobenius_matrix(Frobenius, m, verbose_level - 3);
	if (f_v) {
		cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
				"Frobenius_matrix = " << endl;
		int_matrix_print(Frobenius, d, d);
		cout << endl;
	}

#if 0
	F->mult_matrix_matrix(Frobenius, Frobenius, F2, d, d, d,
			0 /* verbose_level */);
	if (f_v) {
		cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
				"Frobenius^2 = " << endl;
		int_matrix_print(F2, d, d);
		cout << endl;
	}
#endif

	FX.compute_normal_basis(d, Normal_basis, Frobenius, verbose_level - 3);

	if (f_v) {
		cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
				"Normal_basis = " << endl;
		int_matrix_print(Normal_basis, d, d);
		cout << endl;
	}

	cnt = 0;
	Combi.int_vec_first_regular_word(v, d, F->q);
	while (TRUE) {
		if (f_vv) {
			cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : v = ";
			int_vec_print(cout, v, d);
			cout << endl;
		}

		F->mult_vector_from_the_right(Normal_basis, v, w, d, d);
		if (f_vv) {
			cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : w = ";
			int_vec_print(cout, w, d);
			cout << endl;
		}

		FX.delete_object(g);
		FX.create_object_of_degree(g, d - 1);
		for (i = 0; i < d; i++) {
			((int *) g)[1 + i] = w[i];
		}

		FX.minimum_polynomial_extension_field(g, m, minpol, d,
				Frobenius, verbose_level - 3);
		if (f_vv) {
			cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : v = ";
			int_vec_print(cout, v, d);
			cout << " irreducible polynomial = ";
			FX.print_object(minpol, cout);
			cout << endl;
		}
		if (FX.degree(minpol) != d) {
			cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
					"The polynomial does not have degree d"
					<< endl;
			FX.print_object(minpol, cout);
			cout << endl;
			exit(1);
		}
		if (!FX.is_irreducible(minpol, verbose_level)) {
			cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
					"The polynomial is not irreducible" << endl;
			FX.print_object(minpol, cout);
			cout << endl;
			exit(1);
		}


		cnt++;

		if (!Combi.int_vec_next_regular_word(v, d, F->q)) {
			break;
		}

	}

	if (f_v) {
		cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d "
				"there are " << cnt << " irreducible polynomials "
				"of degree " << d << " over " << "F_" << F->q << endl;
	}

	FREE_int(Frobenius);
	//FREE_int(F2);
	FREE_int(Normal_basis);
	FREE_int(v);
	FREE_int(w);
	FX.delete_object(m);
	FX.delete_object(g);
	FX.delete_object(minpol);

	if (f_v) {
		cout << "algebra_global::count_all_irreducible_polynomials_of_degree_d done" << endl;
	}
	return cnt;
}

void algebra_global::polynomial_division(finite_field *F,
		std::string &A_coeffs, std::string &B_coeffs, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::polynomial_division" << endl;
	}

	//int q = F->q;

	int *data_A;
	int *data_B;
	int sz_A, sz_B;

	int_vec_scan(A_coeffs, data_A, sz_A);
	int_vec_scan(B_coeffs, data_B, sz_B);



	unipoly_domain FX(F);
	unipoly_object A, B, Q, R;


	int da = sz_A - 1;
	int db = sz_B - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		FX.s_i(A, i) = data_A[i];
	}

	FX.create_object_of_degree(B, da);

	for (i = 0; i <= db; i++) {
		FX.s_i(B, i) = data_B[i];
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;
	}


	if (f_v) {
		cout << "B(X)=";
		FX.print_object(B, cout);
		cout << endl;
		}

	FX.create_object_of_degree(Q, da);

	FX.create_object_of_degree(R, da);


	if (f_v) {
		cout << "algebra_global::polynomial_division before FX.division_with_remainder" << endl;
	}

	FX.division_with_remainder(
		A, B,
		Q, R,
		verbose_level);

	if (f_v) {
		cout << "algebra_global::polynomial_division after FX.division_with_remainder" << endl;
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;

		cout << "Q(X)=";
		FX.print_object(Q, cout);
		cout << endl;

		cout << "R(X)=";
		FX.print_object(R, cout);
		cout << endl;
	}

	if (f_v) {
		cout << "algebra_global::polynomial_division done" << endl;
	}
}

void algebra_global::extended_gcd_for_polynomials(finite_field *F,
		std::string &A_coeffs, std::string &B_coeffs, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::extended_gcd_for_polynomials" << endl;
	}


	int q = F->q;

	int *data_A;
	int *data_B;
	int sz_A, sz_B;

	int_vec_scan(A_coeffs, data_A, sz_A);
	int_vec_scan(B_coeffs, data_B, sz_B);

	number_theory_domain NT;




	unipoly_domain FX(F);
	unipoly_object A, B, U, V, G;


	int da = sz_A - 1;
	int db = sz_B - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= q) {
			data_A[i] = NT.mod(data_A[i], q);
		}
		FX.s_i(A, i) = data_A[i];
	}

	FX.create_object_of_degree(B, da);

	for (i = 0; i <= db; i++) {
		if (data_B[i] < 0 || data_B[i] >= q) {
			data_B[i] = NT.mod(data_B[i], q);
		}
		FX.s_i(B, i) = data_B[i];
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;


		cout << "B(X)=";
		FX.print_object(B, cout);
		cout << endl;
	}

	FX.create_object_of_degree(U, da);

	FX.create_object_of_degree(V, da);

	FX.create_object_of_degree(G, da);


	if (f_v) {
		cout << "algebra_global::extended_gcd_for_polynomials before FX.extended_gcd" << endl;
	}

	{
		FX.extended_gcd(
			A, B,
			U, V, G, verbose_level);
	}

	if (f_v) {
		cout << "algebra_global::extended_gcd_for_polynomials after FX.extended_gcd" << endl;
	}

	if (f_v) {
		cout << "U(X)=";
		FX.print_object(U, cout);
		cout << endl;

		cout << "V(X)=";
		FX.print_object(V, cout);
		cout << endl;

		cout << "G(X)=";
		FX.print_object(G, cout);
		cout << endl;

		cout << "deg G(X) = " << FX.degree(G) << endl;
	}

	if (FX.degree(G) == 0) {
		int c, cv, d;

		c = FX.s_i(G, 0);
		if (c != 1) {
			if (f_v) {
				cout << "normalization:" << endl;
			}
			cv = F->inverse(c);
			if (f_v) {
				cout << "cv=" << cv << endl;
			}

			d = FX.degree(U);
			for (i = 0; i <= d; i++) {
				FX.s_i(U, i) = F->mult(cv, FX.s_i(U, i));
			}

			d = FX.degree(V);
			for (i = 0; i <= d; i++) {
				FX.s_i(V, i) = F->mult(cv, FX.s_i(V, i));
			}

			d = FX.degree(G);
			for (i = 0; i <= d; i++) {
				FX.s_i(G, i) = F->mult(cv, FX.s_i(G, i));
			}



			if (f_v) {
				cout << "after normalization:" << endl;

				cout << "U(X)=";
				FX.print_object(U, cout);
				cout << endl;

				cout << "V(X)=";
				FX.print_object(V, cout);
				cout << endl;

				cout << "G(X)=";
				FX.print_object(G, cout);
				cout << endl;
			}
		}

	}

	if (f_v) {
		cout << "algebra_global::extended_gcd_for_polynomials done" << endl;
	}
}


void algebra_global::polynomial_mult_mod(finite_field *F,
		std::string &A_coeffs, std::string &B_coeffs, std::string &M_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::polynomial_mult_mod" << endl;
	}

	int q = F->q;

	int *data_A;
	int *data_B;
	int *data_M;
	int sz_A, sz_B, sz_M;

	int_vec_scan(A_coeffs, data_A, sz_A);
	int_vec_scan(B_coeffs, data_B, sz_B);
	int_vec_scan(M_coeffs, data_M, sz_M);

	number_theory_domain NT;




	unipoly_domain FX(F);
	unipoly_object A, B, M, C;


	int da = sz_A - 1;
	int db = sz_B - 1;
	int dm = sz_M - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= q) {
			data_A[i] = NT.mod(data_A[i], q);
		}
		FX.s_i(A, i) = data_A[i];
	}

	FX.create_object_of_degree(B, da);

	for (i = 0; i <= db; i++) {
		if (data_B[i] < 0 || data_B[i] >= q) {
			data_B[i] = NT.mod(data_B[i], q);
		}
		FX.s_i(B, i) = data_B[i];
	}

	FX.create_object_of_degree(M, dm);

	for (i = 0; i <= dm; i++) {
		if (data_M[i] < 0 || data_M[i] >= q) {
			data_M[i] = NT.mod(data_M[i], q);
		}
		FX.s_i(M, i) = data_M[i];
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;


		cout << "B(X)=";
		FX.print_object(B, cout);
		cout << endl;

		cout << "M(X)=";
		FX.print_object(M, cout);
		cout << endl;
	}

	FX.create_object_of_degree(C, da + db);



	if (f_v) {
		cout << "algebra_global::polynomial_mult_mod before FX.mult_mod" << endl;
	}

	{
		FX.mult_mod(A, B, C, M, verbose_level);
	}

	if (f_v) {
		cout << "algebra_global::polynomial_mult_mod after FX.mult_mod" << endl;
	}

	if (f_v) {
		cout << "C(X)=";
		FX.print_object(C, cout);
		cout << endl;

		cout << "deg C(X) = " << FX.degree(C) << endl;
	}

	if (f_v) {
		cout << "algebra_global::polynomial_mult_mod done" << endl;
	}
}

void algebra_global::Berlekamp_matrix(finite_field *F,
		std::string &Berlekamp_matrix_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::Berlekamp_matrix" << endl;
	}

	int q = F->q;
	int *data_A;
	int sz_A;

	int_vec_scan(Berlekamp_matrix_coeffs, data_A, sz_A);

	number_theory_domain NT;




	unipoly_domain FX(F);
	unipoly_object A;


	int da = sz_A - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= q) {
			data_A[i] = NT.mod(data_A[i], q);
		}
		FX.s_i(A, i) = data_A[i];
	}



	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;
	}


	int *B;
	int r;



	if (f_v) {
		cout << "algebra_global::Berlekamp_matrix before FX.Berlekamp_matrix" << endl;
	}

	{
		FX.Berlekamp_matrix(B, A, verbose_level);
	}

	if (f_v) {
		cout << "algebra_global::Berlekamp_matrix after FX.Berlekamp_matrix" << endl;
	}

	if (f_v) {
		cout << "B=" << endl;
		int_matrix_print(B, da, da);
		cout << endl;
	}

	r = F->rank_of_matrix(B, da, 0 /* verbose_level */);

	if (f_v) {
		cout << "The matrix B has rank " << r << endl;
	}

	FREE_int(B);

	if (f_v) {
		cout << "algebra_global::Berlekamp_matrix done" << endl;
	}
}




void algebra_global::compute_normal_basis(finite_field *F, int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;

	if (f_v) {
		cout << "algebra_global::compute_normal_basis "
				<< " q=" << F->q << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}


	unipoly_domain FX(F);

	const char *poly;

	poly = get_primitive_polynomial(F->q, d, 0 /* verbose_level */);

	if (f_v) {
		cout << "algebra_global::compute_normal_basis "
				"chosen irreducible polynomial is " << poly << endl;
	}

	unipoly_object m;
	unipoly_object g;
	unipoly_object minpol;
	combinatorics_domain Combi;


	FX.create_object_by_rank_string(m, poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "algebra_global::compute_normal_basis "
				"chosen irreducible polynomial m = ";
		FX.print_object(m, cout);
		cout << endl;
	}

	FX.create_object_by_rank(g, 0, __FILE__, __LINE__, verbose_level);
	FX.create_object_by_rank(minpol, 0, __FILE__, __LINE__, verbose_level);

	int *Frobenius;
	int *Normal_basis;

	//Frobenius = NEW_int(d * d);
	Normal_basis = NEW_int(d * d);

	if (f_v) {
		cout << "algebra_global::compute_normal_basis "
				"before FX.Frobenius_matrix" << endl;
	}
	FX.Frobenius_matrix(Frobenius, m, verbose_level - 2);
	if (f_v) {
		cout << "algebra_global::compute_normal_basis "
				"Frobenius_matrix = " << endl;
		int_matrix_print(Frobenius, d, d);
		cout << endl;
	}

	if (f_v) {
		cout << "algebra_global::compute_normal_basis "
				"before compute_normal_basis" << endl;
	}

	FX.compute_normal_basis(d, Normal_basis, Frobenius, verbose_level - 1);

	if (f_v) {
		cout << "algebra_global::compute_normal_basis "
				"Normal_basis = " << endl;
		int_matrix_print(Normal_basis, d, d);
		cout << endl;
	}

	if (f_v) {
		cout << "algebra_global::compute_normal_basis done" << endl;
	}
}


void algebra_global::do_nullspace(finite_field *F,
		int m, int n, std::string &text,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	int *A;
	int *base_cols;
	int len, rk, i, rk1;

	latex_interface Li;

	if (f_v) {
		cout << "do_nullspace" << endl;
	}


	int_vec_scan(text, M, len);
	if (len != m * n) {
		cout << "number of coordinates received differs from m * n" << endl;
		cout << "received " << len << endl;
		exit(1);
	}

	if (m > n) {
		cout << "nullspace needs m < n" << endl;
		exit(1);
	}

	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	int_vec_copy(M, A, m * n);

	rk = F->perp_standard(n, m, A, verbose_level);


	if (f_v) {
		cout << "after perp_standard:" << endl;
		int_matrix_print(A, n, n);
		cout << "rk=" << rk << endl;
	}

	rk1 = F->Gauss_int(A + rk * n,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, n - rk, n, n,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "after RREF" << endl;
		int_matrix_print(A + rk * n, rk1, n);
		cout << "rank of nullspace = " << rk1 << endl;

		cout << "coefficients:" << endl;
		int_vec_print(cout, A + rk * n, rk1 * n);
		cout << endl;

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;
	}

	if (f_normalize_from_the_left) {
		if (f_v) {
			cout << "normalizing from the left" << endl;
		}
		for (i = rk; i < n; i++) {
			F->PG_element_normalize_from_front(
					A + i * n, 1, n);
		}

		if (f_v) {
			cout << "after normalize from the left:" << endl;
			int_matrix_print(A, n, n);
			cout << "rk=" << rk << endl;

			cout << "$$" << endl;
			cout << "\\left[" << endl;
			Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
			cout << "\\right]" << endl;
			cout << "$$" << endl;
		}
	}

	if (f_normalize_from_the_right) {
		if (f_v) {
			cout << "normalizing from the right" << endl;
		}
		for (i = rk; i < n; i++) {
			F->PG_element_normalize(
					A + i * n, 1, n);
		}

		if (f_v) {
			cout << "after normalize from the right:" << endl;
			int_matrix_print(A, n, n);
			cout << "rk=" << rk << endl;

			cout << "$$" << endl;
			cout << "\\left[" << endl;
			Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
			cout << "\\right]" << endl;
			cout << "$$" << endl;
		}
	}


	FREE_int(M);
	FREE_int(A);
	FREE_int(base_cols);

	if (f_v) {
		cout << "do_nullspace done" << endl;
	}
}

void algebra_global::do_RREF(finite_field *F,
		int m, int n, std::string &text,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	int *A;
	int *base_cols;
	int len, rk, i;
	latex_interface Li;

	if (f_v) {
		cout << "do_RREF" << endl;
	}


	int_vec_scan(text, M, len);
	if (len != m * n) {
		cout << "number of coordinates received differs from m * n" << endl;
		cout << "received " << len << endl;
		exit(1);
	}


	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	int_vec_copy(M, A, m * n);

	rk = F->Gauss_int(A,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, m, n, n,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "after RREF:" << endl;
		int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;

		cout << "coefficients:" << endl;
		int_vec_print(cout, A, rk * n);
		cout << endl;

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.int_matrix_print_tex(cout, A, rk, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;
	}

	if (f_normalize_from_the_left) {
		if (f_v) {
			cout << "normalizing from the left" << endl;
		}
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize_from_front(
					A + i * n, 1, n);
		}

		if (f_v) {
			cout << "after normalize from the left:" << endl;
			int_matrix_print(A, rk, n);
			cout << "rk=" << rk << endl;
		}
	}

	if (f_normalize_from_the_right) {
		if (f_v) {
			cout << "normalizing from the right" << endl;
		}
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize(
					A + i * n, 1, n);
		}

		if (f_v) {
			cout << "after normalize from the right:" << endl;
			int_matrix_print(A, rk, n);
			cout << "rk=" << rk << endl;
		}
	}


	FREE_int(M);
	FREE_int(A);
	FREE_int(base_cols);

	if (f_v) {
		cout << "do_RREF done" << endl;
	}
}


void algebra_global::do_trace(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int s, t;
	int *T0 = NULL;
	int *T1 = NULL;
	int nb_T0 = 0;
	int nb_T1 = 0;


	if (f_v) {
		cout << "do_trace" << endl;
	}

	int q = F->q;

	T0 = NEW_int(q);
	T1 = NEW_int(q);

	for (s = 0; s < q; s++) {

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
		if (t == 1) {
			T1[nb_T1++] = s;
		}
		else {
			T0[nb_T0++] = s;
		}
	}

	cout << "Trace 0:" << endl;
	int_vec_print_fully(cout, T0, nb_T0);
	cout << endl;

	cout << "Trace 1:" << endl;
	int_vec_print_fully(cout, T1, nb_T1);
	cout << endl;

	char str[1000];
	string fname_csv;
	file_io Fio;

	snprintf(str, 1000, "F_q%d_trace_0.csv", q);
	fname_csv.assign(str);
	Fio.int_vec_write_csv(T0, nb_T0,
			fname_csv, "Trace_0");
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;

	snprintf(str, 1000, "F_q%d_trace_1.csv", q);
	fname_csv.assign(str);
	Fio.int_vec_write_csv(T1, nb_T1,
			fname_csv, "Trace_1");
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	if (f_v) {
		cout << "do_trace done" << endl;
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
		cout << "do_norm" << endl;
	}

	int q = F->q;

	T0 = NEW_int(q);
	T1 = NEW_int(q);

	for (s = 0; s < q; s++) {
		t = F->absolute_norm(s);
		if (t == 1) {
			T1[nb_T1++] = s;
		}
		else {
			T0[nb_T0++] = s;
		}
	}

	cout << "Norm 0:" << endl;
	int_vec_print_fully(cout, T0, nb_T0);
	cout << endl;

	cout << "Norm 1:" << endl;
	int_vec_print_fully(cout, T1, nb_T1);
	cout << endl;


	char str[1000];
	string fname_csv;
	file_io Fio;

	snprintf(str, 1000, "F_q%d_norm_0.csv", q);
	fname_csv.assign(str);
	Fio.int_vec_write_csv(T0, nb_T0,
			fname_csv, "Norm_0");
	cout << "written file " << fname_csv << " of size " << Fio.file_size(fname_csv) << endl;

	snprintf(str, 1000, "F_q%d_norm_1.csv", q);
	fname_csv.assign(str);
	Fio.int_vec_write_csv(T1, nb_T1,
			fname_csv, "Norm_1");
	cout << "written file " << fname_csv << " of size " << Fio.file_size(fname_csv) << endl;


	if (f_v) {
		cout << "do_norm done" << endl;
	}
}

void algebra_global::do_equivalence_class_of_fractions(int N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h, a, b, ap, bp, g;
	number_theory_domain NT;
	file_io Fio;

	if (f_v) {
		cout << "do_equivalence_class_of_fractions" << endl;
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
		cout << "do_equivalence_class_of_fractions done" << endl;
	}
}


void algebra_global::do_cheat_sheet_GF(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::do_cheat_sheet_GF q=" << F->q << endl;
	}

	//int i;
	//int f_poly = FALSE;
	//const char *poly = NULL;

	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "GF_%d.tex", F->q);
	snprintf(title, 1000, "Cheat Sheet GF($%d$)", F->q);
	//sprintf(author, "");
	author[0] = 0;



	F->addition_table_save_csv();

	F->multiplication_table_save_csv();

	F->addition_table_reordered_save_csv();

	F->multiplication_table_reordered_save_csv();


	{
		ofstream f(fname);


		//algebra_global AG;

		//AG.cheat_sheet_GF(q, f_override_poly, my_override_poly, verbose_level);
		latex_interface L;

		//F.init(q), verbose_level - 2);

		L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
			title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
				TRUE /* f_12pt */,
				TRUE /* f_enlarged_page */,
				TRUE /* f_pagenumbers */,
				NULL /* extra_praeamble */);


		F->cheat_sheet(f, verbose_level);

		F->cheat_sheet_main_table(f, verbose_level);

		F->cheat_sheet_addition_table(f, verbose_level);

		F->cheat_sheet_multiplication_table(f, verbose_level);

		F->cheat_sheet_power_table(f, verbose_level);





		L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "algebra_global::do_cheat_sheet_GF q=" << F->q << " done" << endl;
	}
}

void algebra_global::do_search_for_primitive_polynomial_in_range(
		int p_min, int p_max,
		int deg_min, int deg_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::do_search_for_primitive_polynomial_in_range" << endl;
		cout << "p_min=" << p_min << endl;
		cout << "p_max=" << p_max << endl;
		cout << "deg_min=" << deg_min << endl;
		cout << "deg_max=" << deg_max << endl;
	}


	if (deg_min == deg_max && p_min == p_max) {
		char *poly;



		poly = search_for_primitive_polynomial_of_given_degree(
				p_min, deg_min, verbose_level);

		cout << "poly = " << poly << endl;

	}
	else {

		search_for_primitive_polynomials(p_min, p_max,
				deg_min, deg_max,
				verbose_level);
	}

	if (f_v) {
		cout << "algebra_global::do_search_for_primitive_polynomial_in_range done" << endl;
	}
}

void algebra_global::do_make_table_of_irreducible_polynomials(int deg, finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::do_make_table_of_irreducible_polynomials" << endl;
		cout << "deg=" << deg << endl;
		cout << "q=" << F->q << endl;
	}
	int nb;
	//int *Table;
	std::vector<std::vector<int>> Table;

	make_all_irreducible_polynomials_of_degree_d(F, deg,
			Table, verbose_level);

	nb = Table.size();

	cout << "The " << nb << " irreducible polynomials of "
			"degree " << deg << " over F_" << F->q << " are:" << endl;

	int_vec_vec_print(Table);


	//int_matrix_print(Table, nb, deg + 1);

	//FREE_int(Table);

	if (f_v) {
		cout << "algebra_global::do_make_table_of_irreducible_polynomials done" << endl;
	}
}

void algebra_global::polynomial_find_roots(finite_field *F,
		std::string &A_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::polynomial_find_roots" << endl;
	}

	int q = F->q;

	int *data_A;
	int sz_A;

	int_vec_scan(A_coeffs, data_A, sz_A);

	number_theory_domain NT;




	unipoly_domain FX(F);
	unipoly_object A;


	int da = sz_A - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= q) {
			data_A[i] = NT.mod(data_A[i], q);
		}
		FX.s_i(A, i) = data_A[i];
	}


	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;
	}



	if (f_v) {
		cout << "algebra_global::polynomial_find_roots before FX.mult_mod" << endl;
	}

	{
		int a, b;

		for (a = 0; a < q; a++) {
			b = FX.substitute_scalar_in_polynomial(
				A, a, 0 /* verbose_level*/);
			if (b == 0) {
				cout << "a root is " << a << endl;
			}
		}
	}

	if (f_v) {
		cout << "algebra_global::polynomial_find_roots after FX.mult_mod" << endl;
	}


	if (f_v) {
		cout << "algebra_global::polynomial_find_roots done" << endl;
	}
}

/*
 * twocoef.cpp
 *
 *  Created on: Oct 22, 2020
 *      Author: alissabrown
 *
 *	Received a lot of help from Anton and the recursive function in the possibleC function is modeled after code found at
 *	https://www.geeksforgeeks.org/print-all-combinations-of-given-length/
 *
 *
 */

void algebra_global::find_CRC_polynomials(finite_field *F,
		int t, int da, int dc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::find_CRC_polynomials t=" << t << " info=" << da << " check=" << dc << endl;
	}

	//int dc = 4; //dc is the number of parity bits & degree of g(x)
	//int da = 4; //da is the degree of the information polynomial
	int A[da + 1 + dc]; //
	long int nb_sol = 0;

	//int dc2 = dc; //This is also the degree of C/ # of parity bits


	int C[dc + 1]; //Array C (what we divide by)
	//int p = 2; //this is the number of possible coefficients (1 & 0 in this case)

	int i = 0;

	for (i = 0; i <= dc; i++) {
		C[i] = 0;
	}


	std::vector<std::vector<int>> Solutions;

	search_for_CRC_polynomial(t, da, A, dc, C, 0, F, nb_sol, Solutions, verbose_level - 1);

	cout << "algebra_global::find_CRC_polynomials info=" << da << " check=" << dc << " nb_sol=" << nb_sol << endl;

	for (i = 0; i < Solutions.size(); i++) {
		cout << i << " : ";
		for (int j = dc; j >= 0; j--) {
			cout << Solutions[i][j];
		}
		cout << endl;
	}
	cout << "algebra_global::find_CRC_polynomials info=" << da << " check=" << dc << " nb_sol=" << nb_sol << endl;

}

void algebra_global::search_for_CRC_polynomial(int t, int da, int *A, int dc, int *C, int i, finite_field *F,
		long int &nb_sol, std::vector<std::vector<int> > &Solutions, int verbose_level)
{

	if (i == dc + 1) {

		int ret;

		if (t >= 2) {
			ret = test_all_two_bit_patterns(da, A, dc, C, F, verbose_level);
			if (ret && t >= 3) {
				ret = test_all_three_bit_patterns(da, A, dc, C, F, verbose_level);
			}
		}
		else {
			cout << "illegal value for t, t=" << t << endl;
			exit(1);
		}
		if (ret) {
			cout << "solution " << nb_sol << " is ";
			for (int j = dc; j >= 0; j--) {
				cout << C[j];
			}
			cout << endl;

			vector<int> sol;

			for (int j = 0; j <= dc; j++) {
				sol.push_back(C[j]);
			}
			Solutions.push_back(sol);


			nb_sol++;
		}

		return;
	}

	if (i == dc) {

		C[i] = 1;
		search_for_CRC_polynomial(t, da, A, dc, C, i + 1, F, nb_sol, Solutions, verbose_level);


	}
	else {
		int c;

		for (c = 0; c < F->q; c++) {

			C[i] = c;

			search_for_CRC_polynomial(t, da, A, dc, C, i + 1, F, nb_sol, Solutions, verbose_level);
		}
	}
}

int algebra_global::test_all_two_bit_patterns(int da, int *A, int dc, int *C, finite_field *F, int verbose_level)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{

	//cout << "choosetwo" << endl;

	int f_v = (verbose_level >= 1);

	int i;
	int j;
	int k;
	int ai, aj;
	int ret;
	int B[da + dc + 1];

	if (f_v) {
		cout << "testing polynomial: ";
		for (k = dc; k >= 0; k--) {
			cout << C[k];
		}
		cout << endl;
	}

	for (i = 0; i <= da; i++) {
		A[i] = 0;
	}

	for (i = 0; i <= da; i++) {

		for (ai = 1; ai < F->q; ai++) {

			A[i] = ai;

			for (j = i + 1; j <= da; j++) {

				for (aj = 1; aj < F->q; aj++) {

					A[j] = aj;

					for (k = 0; k <= da; k++) {
						B[dc + k] = A[k];
					}
					for (k = 0; k < dc; k++) {
						B[k] = 0;
					}

					if (f_v) {
						cout << "testing error pattern: ";
						for (k = dc + da; k >= 0; k--) {
							cout << B[k];
						}
					}
					//cout << endl;



					ret = remainder_is_nonzero (da, B, dc, C, F);

					if (f_v) {
						cout << " : ";
						for (k = dc; k >= 0; k--) {
							cout << B[k];
						}
						cout << endl;
					}

					if (!ret) {
						return false;
					}

				}
				A[j] = 0;
			}

			//cout << endl;
		}
		A[i] = 0;
	}
	return true;
}

int algebra_global::test_all_three_bit_patterns(int da, int *A, int dc, int *C, finite_field *F, int verbose_level)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{

	//cout << "choosetwo" << endl;

	int f_v = (verbose_level >= 1);

	int i1, i2, i3;
	int k;
	int a1, a2, a3;
	int ret;
	int B[da + dc + 1];

	if (f_v) {
		cout << "testing polynomial: ";
		for (k = dc; k >= 0; k--) {
			cout << C[k];
		}
		cout << endl;
	}

	for (int h = 0; h <= da; h++) {
		A[h] = 0;
	}

	for (i1 = 0; i1 <= da; i1++) {

		for (a1 = 1; a1 < F->q; a1++) {

			A[i1] = a1;

			for (i2 = i1 + 1; i2 <= da; i2++) {

				for (a2 = 1; a2 < F->q; a2++) {

					A[i2] = a2;

					for (i3 = i2 + 1; i3 <= da; i3++) {

						for (a3 = 1; a3 < F->q; a3++) {

							A[i3] = a3;

							for (int h = 0; h <= da; h++) {
								B[dc + h] = A[h];
							}
							for (int h = 0; h < dc; h++) {
								B[h] = 0;
							}

							if (f_v) {
								cout << "testing error pattern: ";
								for (int h = dc + da; h >= 0; h--) {
									cout << B[h];
								}
							}



							ret = remainder_is_nonzero (da, B, dc, C, F);

							if (f_v) {
								cout << " : ";
								for (int h = dc; h >= 0; h--) {
									cout << B[h];
								}
								cout << endl;
							}

							if (!ret) {
								return false;
							}

						}
						A[i3] = 0;
					}
				}
				A[i2] = 0;
			}
		}
		A[i1] = 0;
	}
	return true;
}

int algebra_global::remainder_is_nonzero(int da, int *A, int db, int *B, finite_field *F)
// returns true if the remainder of A after division by B is nonzero
{

	int i, j, k, a, mav;

	for (i = da + db; i >= db; i--) {

		a = A[i];

		if (a) {

			mav = F->negate(F->inverse(a));

			for (j = db, k = i; j >= 0; j--, k--) {

				A[k] = F->add(A[k], F->mult(mav, B[j]));
					//A[k] = subtraction(A[k], B[j], p);
					//A[k]=(A[k]+2-B[j])%2;
			}
		}
	}


	for (int k = da + db; k >= 0; k--) {
		if (A[k]) {
			return true;
		}
	}
	return false;
}




}}

