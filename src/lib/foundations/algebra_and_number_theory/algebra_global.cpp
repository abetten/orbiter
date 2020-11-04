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


void algebra_global::cheat_sheet_GF(int q,
		int f_override_polynomial,
		std::string &override_polynomial,
		int verbose_level)
{
	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "GF_%d.tex", q);
	snprintf(title, 1000, "Cheat Sheet GF($%d$)", q);
	author[0] = 0;
	author[0] = 0;
	finite_field F;

	{
	ofstream f(fname);
	latex_interface L;

	//F.init(q), verbose_level - 2);
	if (f_override_polynomial) {
		F.init_override_polynomial(q, override_polynomial, verbose_level);
	}
	else {
		F.init(q, verbose_level);
	}
	L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);


	F.cheat_sheet(f, verbose_level);


	L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


	//F.compute_subfields(verbose_level);
}

char *algebra_global::search_for_primitive_polynomial_of_given_degree(
		int p, int degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field Fp;

	if (f_v) {
		cout << "algebra_global::search_for_primitive_polynomial_of_given_degree" << endl;
	}
	Fp.init(p, 0 /*verbose_level*/);
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
			Fq.init(q, 0 /*verbose_level*/);
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
		Fq.init(q, verbose_level - 2);
	}
	FQ.init(Q, verbose_level - 2);

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
				phi_g = Phi_of(g, 1);
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


int algebra_global::Phi_of(int n, int verbose_level)
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
	F.init(q, 0 /*verbose_level*/);
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

void algebra_global::make_Hamming_graph_and_write_file(int n, int q,
		int f_projective, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int width, height;
	int *v;
	int *w;
	int *Table;
	//int *Adj = NULL;
	geometry_global Gg;
	finite_field *F = NULL;

	if (f_v) {
		cout << "algebra_global::make_Hamming_graph_and_write_file" << endl;
	}

	v = NEW_int(n);
	w = NEW_int(n);

	if (f_projective) {
		width = height = Gg.nb_PG_elements(n - 1, q);
		F = NEW_OBJECT(finite_field);
		F->init(q);
	}
	else {
		width = height = Gg.nb_AG_elements(n, q);
	}

#if 0
	int N;
	N = width;
	if (f_graph) {
		Adj = NEW_int(N * N);
		int_vec_zero(Adj, N * N);
	}
#endif

	cout << "width=" << width << endl;

	int i, j, d, h;

	Table = NEW_int(height * width);
	for (i = 0; i < height; i++) {

		if (f_projective) {
			F->PG_element_unrank_modified(v, 1 /*stride*/, n, i);
		}
		else {
			Gg.AG_element_unrank(q, v, 1, n, i);
		}

		for (j = 0; j < width; j++) {

			if (f_projective) {
				F->PG_element_unrank_modified(w, 1 /*stride*/, n, j);
			}
			else {
				Gg.AG_element_unrank(q, w, 1, n, j);
			}

			d = 0;
			for (h = 0; h < n; h++) {
				if (v[h] != w[h]) {
					d++;
				}
			}

#if 0
			if (f_graph && d == 1) {
				Adj[i * N + j] = 1;
			}
#endif

			Table[i * width + j] = d;

		}
	}

	string fname;
	char str[1000];
	file_io Fio;

	sprintf(str, "Hamming_n%d_q%d.csv", n, q);
	fname.assign(str);

	Fio.int_matrix_write_csv(fname, Table, height, width);

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "algebra_global::make_Hamming_graph_and_write_file" << endl;
	}

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

	GFp.init(p, verbose_level);
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

	Fq.init(q, verbose_level);
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

void algebra_global::mac_williams_equations(longinteger_object *&M, int n, int k, int q)
{
	combinatorics_domain D;
	int i, j;

	M = NEW_OBJECTS(longinteger_object, (n + 1) * (n + 1));

	for (i = 0; i <= n; i++) {
		for (j = 0; j <= n; j++) {
			D.krawtchouk(M[i * (n + 1) + j], n, q, i, j);
		}
	}
}

void algebra_global::determine_weight_enumerator()
{
	int n = 19, k = 7, q = 2;
	longinteger_domain D;
	longinteger_object *M, *A1, *A2, qk;
	int i;

	qk.create(q, __FILE__, __LINE__);
	D.power_int(qk, k);
	cout << q << "^" << k << " = " << qk << endl;

	mac_williams_equations(M, n, k, q);

	D.matrix_print_tex(cout, M, n + 1, n + 1);

	A1 = NEW_OBJECTS(longinteger_object, n + 1);
	A2 = NEW_OBJECTS(longinteger_object, n + 1);
	for (i = 0; i <= n; i++) {
		A1[i].create(0, __FILE__, __LINE__);
	}
	A1[0].create(1, __FILE__, __LINE__);
	A1[8].create(78, __FILE__, __LINE__);
	A1[12].create(48, __FILE__, __LINE__);
	A1[16].create(1, __FILE__, __LINE__);
	D.matrix_print_tex(cout, A1, n + 1, 1);

	D.matrix_product(M, A1, A2, n + 1, n + 1, 1);
	D.matrix_print_tex(cout, A2, n + 1, 1);

	D.matrix_entries_integral_division_exact(A2, qk, n + 1, 1);

	D.matrix_print_tex(cout, A2, n + 1, 1);

	FREE_OBJECTS(M);
	FREE_OBJECTS(A1);
	FREE_OBJECTS(A2);
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

void algebra_global::polynomial_division(int q,
		std::string &A_coeffs, std::string &B_coeffs, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::polynomial_division" << endl;
	}


	int *data_A;
	int *data_B;
	int sz_A, sz_B;

	int_vec_scan(A_coeffs, data_A, sz_A);
	int_vec_scan(B_coeffs, data_B, sz_B);

	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init(q);



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

	cout << "A(X)=";
	FX.print_object(A, cout);
	cout << endl;


	cout << "B(X)=";
	FX.print_object(B, cout);
	cout << endl;

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

	cout << "A(X)=";
	FX.print_object(A, cout);
	cout << endl;

	cout << "Q(X)=";
	FX.print_object(Q, cout);
	cout << endl;

	cout << "R(X)=";
	FX.print_object(R, cout);
	cout << endl;


	FREE_OBJECT(F);

	if (f_v) {
		cout << "algebra_global::polynomial_division done" << endl;
	}
}

void algebra_global::extended_gcd_for_polynomials(int q,
		std::string &A_coeffs, std::string &B_coeffs, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::extended_gcd_for_polynomials" << endl;
	}


	int *data_A;
	int *data_B;
	int sz_A, sz_B;

	int_vec_scan(A_coeffs, data_A, sz_A);
	int_vec_scan(B_coeffs, data_B, sz_B);

	finite_field *F;
	number_theory_domain NT;

	F = NEW_OBJECT(finite_field);
	F->init(q);



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

	cout << "A(X)=";
	FX.print_object(A, cout);
	cout << endl;


	cout << "B(X)=";
	FX.print_object(B, cout);
	cout << endl;

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

	if (FX.degree(G) == 0) {
		int c, cv, d;

		c = FX.s_i(G, 0);
		if (c != 1) {
			cout << "normalization:" << endl;
			cv = F->inverse(c);
			cout << "cv=" << cv << endl;

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


	FREE_OBJECT(F);

	if (f_v) {
		cout << "algebra_global::extended_gcd_for_polynomials done" << endl;
	}
}


void algebra_global::polynomial_mult_mod(int q,
		std::string &A_coeffs, std::string &B_coeffs, std::string &M_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::polynomial_mult_mod" << endl;
	}


	int *data_A;
	int *data_B;
	int *data_M;
	int sz_A, sz_B, sz_M;

	int_vec_scan(A_coeffs, data_A, sz_A);
	int_vec_scan(B_coeffs, data_B, sz_B);
	int_vec_scan(M_coeffs, data_M, sz_M);

	finite_field *F;
	number_theory_domain NT;

	F = NEW_OBJECT(finite_field);
	F->init(q);



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

	cout << "A(X)=";
	FX.print_object(A, cout);
	cout << endl;


	cout << "B(X)=";
	FX.print_object(B, cout);
	cout << endl;

	cout << "M(X)=";
	FX.print_object(M, cout);
	cout << endl;

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

	cout << "C(X)=";
	FX.print_object(C, cout);
	cout << endl;

	cout << "deg C(X) = " << FX.degree(C) << endl;





	FREE_OBJECT(F);

	if (f_v) {
		cout << "algebra_global::polynomial_mult_mod done" << endl;
	}
}

void algebra_global::Berlekamp_matrix(int q,
		std::string &Berlekamp_matrix_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::Berlekamp_matrix" << endl;
	}


	int *data_A;
	int sz_A;

	int_vec_scan(Berlekamp_matrix_coeffs, data_A, sz_A);

	finite_field *F;
	number_theory_domain NT;

	F = NEW_OBJECT(finite_field);
	F->init(q);



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



	cout << "A(X)=";
	FX.print_object(A, cout);
	cout << endl;


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

	cout << "B=" << endl;
	int_matrix_print(B, da, da);
	cout << endl;

	r = F->rank_of_matrix(B, da, 0 /* verbose_level */);

	cout << "The matrix B has rank " << r << endl;


	FREE_int(B);
	FREE_OBJECT(F);

	if (f_v) {
		cout << "algebra_global::Berlekamp_matrix done" << endl;
	}
}




void algebra_global::NTRU_encrypt(int N, int p, int q,
		std::string &H_coeffs, std::string &R_coeffs, std::string &Msg_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::NTRU_encrypt" << endl;
	}


	int *data_H;
	int *data_R;
	int *data_Msg;
	int sz_H, sz_R, sz_Msg;

	int_vec_scan(H_coeffs, data_H, sz_H);
	int_vec_scan(R_coeffs, data_R, sz_R);
	int_vec_scan(Msg_coeffs, data_Msg, sz_Msg);

	finite_field *F;
	number_theory_domain NT;

	F = NEW_OBJECT(finite_field);
	F->init(q);



	unipoly_domain FX(F);
	unipoly_object H, R, Msg, M, C, D;


	int dh = sz_H - 1;
	int dr = sz_R - 1;
	int dm = sz_Msg - 1;
	int i;

	FX.create_object_of_degree(H, dh);

	for (i = 0; i <= dh; i++) {
		if (data_H[i] < 0 || data_H[i] >= q) {
			data_H[i] = NT.mod(data_H[i], q);
		}
		FX.s_i(H, i) = data_H[i];
	}

	FX.create_object_of_degree(R, dr);

	for (i = 0; i <= dr; i++) {
		if (data_R[i] < 0 || data_R[i] >= q) {
			data_R[i] = NT.mod(data_R[i], q);
		}
		FX.s_i(R, i) = data_R[i];
	}

	FX.create_object_of_degree(Msg, dm);

	for (i = 0; i <= dm; i++) {
		if (data_Msg[i] < 0 || data_Msg[i] >= q) {
			data_Msg[i] = NT.mod(data_Msg[i], q);
		}
		FX.s_i(Msg, i) = data_Msg[i];
	}

	FX.create_object_of_degree(M, N);
	for (i = 0; i <= N; i++) {
		FX.s_i(M, i) = 0;
	}
	FX.s_i(M, 0) = F->negate(1);
	FX.s_i(M, N) = 1;

	cout << "H(X)=";
	FX.print_object(H, cout);
	cout << endl;


	cout << "R(X)=";
	FX.print_object(R, cout);
	cout << endl;

	cout << "Msg(X)=";
	FX.print_object(Msg, cout);
	cout << endl;

	FX.create_object_of_degree(C, dh);

	FX.create_object_of_degree(D, dh);



	if (f_v) {
		cout << "algebra_global::NTRU_encrypt before FX.mult_mod" << endl;
	}

	{
		FX.mult_mod(R, H, C, M, verbose_level);
		int d;

		d = FX.degree(C);

		for (i = 0; i <= d; i++) {
			FX.s_i(C, i) = F->mult(p, FX.s_i(C, i));
		}

		FX.add(C, Msg, D);

	}

	if (f_v) {
		cout << "algebra_global::NTRU_encrypt after FX.mult_mod" << endl;
	}

	cout << "D(X)=";
	FX.print_object(D, cout);
	cout << endl;

	cout << "deg D(X) = " << FX.degree(D) << endl;





	FREE_OBJECT(F);

	if (f_v) {
		cout << "algebra_global::NTRU_encrypt done" << endl;
	}
}


void algebra_global::polynomial_center_lift(std::string &A_coeffs, int q,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::polynomial_center_lift" << endl;
	}


	int *data_A;
	int sz_A;

	int_vec_scan(A_coeffs, data_A, sz_A);

	finite_field *F;
	number_theory_domain NT;

	F = NEW_OBJECT(finite_field);
	F->init(q);



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


	cout << "A(X)=";
	FX.print_object(A, cout);
	cout << endl;




	if (f_v) {
		cout << "algebra_global::polynomial_center_lift before FX.mult_mod" << endl;
	}

	{
		FX.center_lift_coordinates(A, q);

	}

	if (f_v) {
		cout << "algebra_global::polynomial_center_lift after FX.mult_mod" << endl;
	}

	cout << "A(X)=";
	FX.print_object(A, cout);
	cout << endl;






	FREE_OBJECT(F);

	if (f_v) {
		cout << "algebra_global::polynomial_center_lift done" << endl;
	}
}


void algebra_global::polynomial_reduce_mod_p(std::string &A_coeffs, int p,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::polynomial_reduce_mod_p" << endl;
	}


	int *data_A;
	int sz_A;

	int_vec_scan(A_coeffs, data_A, sz_A);

	finite_field *F;
	number_theory_domain NT;

	F = NEW_OBJECT(finite_field);
	F->init(p);



	unipoly_domain FX(F);
	unipoly_object A;


	int da = sz_A - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		data_A[i] = NT.mod(data_A[i], p);
		FX.s_i(A, i) = data_A[i];
	}


	cout << "A(X)=";
	FX.print_object(A, cout);
	cout << endl;







	FREE_OBJECT(F);

	if (f_v) {
		cout << "algebra_global::polynomial_reduce_mod_p done" << endl;
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


void algebra_global::do_EC_Koblitz_encoding(int q,
		int EC_b, int EC_c, int EC_s,
		const char *pt_text, const char *EC_message,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int x0, x, y;

	if (f_v) {
		cout << "do_EC_Koblitz_encoding" << endl;
	}
	if (f_v) {
		cout << "do_EC_Koblitz_encoding b = " << EC_b << endl;
		cout << "do_EC_Koblitz_encoding c = " << EC_c << endl;
		cout << "do_EC_Koblitz_encoding s = " << EC_s << endl;
	}

	vector<vector<int>> Encoding;
	vector<int> J;

	int u, i, j, r;

	u = q / 27;
	if (f_v) {
		cout << "do_EC_Koblitz_encoding u = " << u << endl;
	}


	F = NEW_OBJECT(finite_field);
	F->init(q, 0 /*verbose_level*/);
	for (i = 1; i <= 26; i++) {
		x0 = i * u;
		for (j = 0; j < u; j++) {
			x = x0 + j;
			r = EC_evaluate_RHS(F, EC_b, EC_c, x);
			if (F->square_root(r, y)) {
				break;
			}
		}
		if (j < u) {
			{
				vector<int> pt;

				J.push_back(j);
				pt.push_back(x);
				pt.push_back(y);
				pt.push_back(1);
				Encoding.push_back(pt);
			}
		}
		else {
			cout << "failure to encode letter " << i << endl;
			exit(1);
		}
	}
	for (i = 0; i < 26; i++) {


		x = (i + 1) * u + J[i];

		r = EC_evaluate_RHS(F, EC_b, EC_c, x);

		F->square_root(r, y);

		cout << (char)('A' + i) << " & " << i + 1 << " & " << J[i] << " & " << x
				<< " & " << r
				<< " & " << y
				<< " & $(" << Encoding[i][0] << "," << Encoding[i][1] << ")$ "
				<< "\\\\" << endl;

	}

	cout << "without j:" << endl;
	for (i = 0; i < 26; i++) {
		cout << (char)('A' + i) << " & $(" << Encoding[i][0] << "," << Encoding[i][1] << ")$ \\\\" << endl;

	}



	vector<vector<int>> Pts;
	int order;
	int *v;
	int len;
	int Gx, Gy, Gz;
	int Mx, My, Mz;
	int Rx, Ry, Rz;
	int Ax, Ay, Az;
	int Cx, Cy, Cz;
	int Tx, Ty, Tz;
	int Dx, Dy, Dz;
	int msRx, msRy, msRz;
	int m, k, plain;
	os_interface Os;

	int_vec_scan(pt_text, v, len);
	if (len != 2) {
		cout << "point should have just two coordinates" << endl;
		exit(1);
	}
	Gx = v[0];
	Gy = v[1];
	Gz = 1;
	FREE_int(v);
	cout << "G = (" << Gx << "," << Gy << "," << Gz << ")" << endl;


	F->elliptic_curve_all_point_multiples(
			EC_b, EC_c, order,
			Gx, Gy, Gz,
			Pts,
			verbose_level);


	int minus_s;

	minus_s = order - EC_s;

	cout << "order = " << order << endl;
	cout << "minus_s = " << minus_s << endl;

	Ax = Pts[EC_s - 1][0];
	Ay = Pts[EC_s - 1][1];
	Az = 1;
	cout << "A = (" << Ax << "," << Ay << "," << Az << ")" << endl;

	len = strlen(EC_message);

	F->nb_calls_to_elliptic_curve_addition() = 0;

	vector<vector<int>> Ciphertext;

	for (i = 0; i < len; i++) {
		if (EC_message[i] < 'A' || EC_message[i] > 'Z') {
			continue;
		}
		m = EC_message[i] - 'A' + 1;
		k = 1 + Os.random_integer(order - 1);

		Mx = Encoding[m - 1][0];
		My = Encoding[m - 1][1];
		Mz = 1;

		// R := k * G
		//cout << "$R=" << k << "*G$\\\\" << endl;

		F->elliptic_curve_point_multiple /*_with_log*/(
					EC_b, EC_c, k,
					Gx, Gy, Gz,
					Rx, Ry, Rz,
					0 /*verbose_level*/);
		//cout << "$R=" << k << "*G=(" << Rx << "," << Ry << "," << Rz << ")$\\\\" << endl;

		// C := k * A
		//cout << "$C=" << k << "*A$\\\\" << endl;
		F->elliptic_curve_point_multiple /*_with_log*/(
					EC_b, EC_c, k,
					Ax, Ay, Az,
					Cx, Cy, Cz,
					0 /*verbose_level*/);
		//cout << "$C=" << k << "*A=(" << Cx << "," << Cy << "," << Cz << ")$\\\\" << endl;

		// T := C + M
		F->elliptic_curve_addition(EC_b, EC_c,
				Cx, Cy, Cz,
				Mx, My, Mz,
				Tx, Ty, Tz,
				0 /*verbose_level*/);
		//cout << "$T=C+M=(" << Tx << "," << Ty << "," << Tz << ")$\\\\" << endl;
		{
		vector<int> cipher;

		cipher.push_back(Rx);
		cipher.push_back(Ry);
		cipher.push_back(Tx);
		cipher.push_back(Ty);
		Ciphertext.push_back(cipher);
		}

		cout << setw(4) << i << " & " << EC_message[i] << " & " << setw(4) << m << " & " << setw(4) << k
				<< "& (" << setw(4) << Mx << "," << setw(4) << My << "," << setw(4) << Mz << ") "
				<< "& (" << setw(4) << Rx << "," << setw(4) << Ry << "," << setw(4) << Rz << ") "
				<< "& (" << setw(4) << Cx << "," << setw(4) << Cy << "," << setw(4) << Cz << ") "
				<< "& (" << setw(4) << Tx << "," << setw(4) << Ty << "," << setw(4) << Tz << ") "
				<< "\\\\"
				<< endl;

	}

	cout << "Ciphertext:\\\\" << endl;
	for (i = 0; i < (int) Ciphertext.size(); i++) {
		cout << Ciphertext[i][0] << ",";
		cout << Ciphertext[i][1] << ",";
		cout << Ciphertext[i][2] << ",";
		cout << Ciphertext[i][3] << "\\\\" << endl;
	}

	for (i = 0; i < (int) Ciphertext.size(); i++) {
		Rx = Ciphertext[i][0];
		Ry = Ciphertext[i][1];
		Tx = Ciphertext[i][2];
		Ty = Ciphertext[i][3];

		// msR := -s * R
		F->elliptic_curve_point_multiple(
					EC_b, EC_c, minus_s,
					Rx, Ry, Rz,
					msRx, msRy, msRz,
					0 /*verbose_level*/);

		// D := msR + T
		F->elliptic_curve_addition(EC_b, EC_c,
				msRx, msRy, msRz,
				Tx, Ty, Tz,
				Dx, Dy, Dz,
				0 /*verbose_level*/);

		plain = Dx / u;

		cout << setw(4) << i << " & (" << Rx << "," << Ry << "," << Tx << "," << Ty << ") "
				<< "& (" << setw(4) << msRx << "," << setw(4) << msRy << "," << setw(4) << msRz << ") "
				<< "& (" << setw(4) << Dx << "," << setw(4) << Dy << "," << setw(4) << Dz << ") "
				<< " & " << plain << " & " << (char)('A' - 1 + plain)
				<< "\\\\"
				<< endl;

	}

	cout << "nb_calls_to_elliptic_curve_addition="
			<< F->nb_calls_to_elliptic_curve_addition() << endl;


	if (f_v) {
		cout << "do_EC_Koblitz_encoding done" << endl;
	}
}

void algebra_global::do_EC_points(int q,
		int EC_b, int EC_c, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int x, y, r, y1, y2;

	if (f_v) {
		cout << "do_EC_points" << endl;
	}
	vector<vector<int>> Pts;

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	for (x = 0; x < q; x++) {
		r = EC_evaluate_RHS(F, EC_b, EC_c, x);
		if (r == 0) {

			{
				vector<int> pt;

				pt.push_back(x);
				pt.push_back(0);
				pt.push_back(1);
				Pts.push_back(pt);
			}
		}
		else {
			if (F->square_root(r, y)) {
				y1 = y;
				y2 = F->negate(y);
				if (y2 == y1) {
					{
						vector<int> pt;

						pt.push_back(x);
						pt.push_back(y1);
						pt.push_back(1);
						Pts.push_back(pt);
					}
				}
				else {
					if (y2 < y1) {
						y1 = y2;
						y2 = y;
					}
					{
						vector<int> pt;

						pt.push_back(x);
						pt.push_back(y1);
						pt.push_back(1);
						Pts.push_back(pt);
					}
					{
						vector<int> pt;

						pt.push_back(x);
						pt.push_back(y2);
						pt.push_back(1);
						Pts.push_back(pt);
					}
				}
			}
			else {
				// no point for this x coordinate
			}

#if 0
			if (p != 2) {
				l = Legendre(r, q, 0);

				if (l == 1) {
					y = sqrt_mod_involved(r, q);
						// DISCRETA/global.cpp

					if (F->mult(y, y) != r) {
						cout << "There is a problem "
								"with the square root" << endl;
						exit(1);
					}
					y1 = y;
					y2 = F->negate(y);
					if (y2 < y1) {
						y1 = y2;
						y2 = y;
					}
					add_point_to_table(x, y1, 1);
					if (nb == bound) {
						cout << "The number of points "
								"exceeds the bound" << endl;
						exit(1);
					}
					add_point_to_table(x, y2, 1);
					if (nb == bound) {
						cout << "The number of points "
								"exceeds the bound" << endl;
						exit(1);
					}
					//cout << nb++ << " : (" << x << ","
					// << y << ",1)" << endl;
					//cout << nb++ << " : (" << x << ","
					// << F.negate(y) << ",1)" << endl;
				}
			}
			else {
				y = F->frobenius_power(r, e - 1);
				add_point_to_table(x, y, 1);
				if (nb == bound) {
					cout << "The number of points exceeds "
							"the bound" << endl;
					exit(1);
				}
				//cout << nb++ << " : (" << x << ","
				// << y << ",1)" << endl;
			}
#endif

		}
	}
	{
		vector<int> pt;

		pt.push_back(0);
		pt.push_back(1);
		pt.push_back(0);
		Pts.push_back(pt);
	}
	int i;
	cout << "We found " << Pts.size() << " points:" << endl;

	for (i = 0; i < (int) Pts.size(); i++) {
		if (i == (int) Pts.size()) {

			cout << i << " : {\\cal O} : 1\\\\" << endl;

		}
		else {
			{
			vector<vector<int>> Multiples;
			int order;


			F->elliptic_curve_all_point_multiples(
					EC_b, EC_c, order,
					Pts[i][0], Pts[i][1], 1,
					Multiples,
					0 /*verbose_level*/);

			//cout << "we found that the point has order " << order << endl;

			cout << i << " : $(" << Pts[i][0] << "," << Pts[i][1] << ")$ : " << order << "\\\\" << endl;
			}
		}
	}


	if (f_v) {
		cout << "do_EC_points done" << endl;
	}
}

int algebra_global::EC_evaluate_RHS(finite_field *F,
		int EC_b, int EC_c, int x)
// evaluates x^3 + bx + c
{
	int x2, x3, t;

	x2 = F->mult(x, x);
	x3 = F->mult(x2, x);
	t = F->add(x3, F->mult(EC_b, x));
	t = F->add(t, EC_c);
	return t;
}


void algebra_global::do_EC_add(int q,
		int EC_b, int EC_c,
		const char *pt1_text, const char *pt2_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int x1, y1, z1;
	int x2, y2, z2;
	int x3, y3, z3;
	int *v;
	int len;
	//sscanf(p1, "(%d,%d,%d)", &x1, &y1, &z1);

	if (f_v) {
		cout << "do_EC_add" << endl;
	}
	vector<vector<int>> Pts;

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(pt1_text, v, len);
	if (len != 2) {
		cout << "point should have just two ccordinates" << endl;
		exit(1);
	}
	x1 = v[0];
	y1 = v[1];
	z1 = 1;
	FREE_int(v);

	int_vec_scan(pt2_text, v, len);
	if (len != 2) {
		cout << "point should have just two ccordinates" << endl;
		exit(1);
	}
	x2 = v[0];
	y2 = v[1];
	z2 = 1;
	FREE_int(v);


	F->elliptic_curve_addition(EC_b, EC_c,
			x1, y1, z1,
			x2, y2, z2,
			x3, y3, z3,
			verbose_level);
	cout << "(" << x1 << "," << y1 << "," << z1 << ")";
	cout << " + ";
	cout << "(" << x2 << "," << y2 << "," << z2 << ")";
	cout << " = ";
	cout << "(" << x3 << "," << y3 << "," << z3 << ")";
	cout << endl;


	FREE_OBJECT(F);

	if (f_v) {
		cout << "do_EC_add done" << endl;
	}
}

void algebra_global::do_EC_cyclic_subgroup(int q,
		int EC_b, int EC_c, const char *pt_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int x1, y1, z1;
	int *v;
	int len, i;
	//sscanf(p1, "(%d,%d,%d)", &x1, &y1, &z1);

	if (f_v) {
		cout << "do_EC_cyclic_subgroup" << endl;
	}
	vector<vector<int>> Pts;
	int order;

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(pt_text, v, len);
	if (len != 2) {
		cout << "point should have just two ccordinates" << endl;
		exit(1);
	}
	x1 = v[0];
	y1 = v[1];
	z1 = 1;
	FREE_int(v);


	F->elliptic_curve_all_point_multiples(
			EC_b, EC_c, order,
			x1, y1, z1,
			Pts,
			verbose_level);

	cout << "we found that the point has order " << order << endl;
	cout << "The multiples are:" << endl;
	cout << "i : (" << x1 << "," << y1 << ")" << endl;
	for (i = 0; i < (int) Pts.size(); i++) {

		vector<int> pts = Pts[i];

		if (i < (int) Pts.size() - 1) {
			cout << setw(3) << i + 1 << " : ";
			cout << "$(" << pts[0] << "," << pts[1] << ")$";
			cout << "\\\\" << endl;
		}
		else {
			cout << setw(3) << i + 1 << " : ";
			cout << "${\\cal O}$";
			cout << "\\\\" << endl;

		}
	}

	FREE_OBJECT(F);

	if (f_v) {
		cout << "do_EC_cyclic_subgroup done" << endl;
	}
}

void algebra_global::do_EC_multiple_of(int q,
		int EC_b, int EC_c, const char *pt_text, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int x1, y1, z1;
	int x3, y3, z3;
	int *v;
	int len;

	if (f_v) {
		cout << "do_EC_multiple_of" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(pt_text, v, len);
	if (len != 2) {
		cout << "point should have just two ccordinates" << endl;
		exit(1);
	}
	x1 = v[0];
	y1 = v[1];
	z1 = 1;
	FREE_int(v);


	F->elliptic_curve_point_multiple(
			EC_b, EC_c, n,
			x1, y1, z1,
			x3, y3, z3,
			verbose_level);

	cout << "The " << n << "-fold multiple of (" << x1 << "," << y1 << ") is ";
	if (z3 == 0) {

	}
	else {
		if (z3 != 1) {
			cout << "z1 != 1" << endl;
			exit(1);
		}
		cout << "(" << x3 << "," << y3 << ")" << endl;
	}

	FREE_OBJECT(F);

	if (f_v) {
		cout << "do_EC_multiple_of done" << endl;
	}
}

void algebra_global::do_EC_discrete_log(int q,
		int EC_b, int EC_c,
		const char *base_pt_text, const char *pt_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int x1, y1, z1;
	int x3, y3, z3;
	int *v;
	int len;
	int n;

	if (f_v) {
		cout << "do_EC_multiple_of" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(base_pt_text, v, len);
	if (len != 2) {
		cout << "point should have just two ccordinates" << endl;
		exit(1);
	}
	x1 = v[0];
	y1 = v[1];
	z1 = 1;
	FREE_int(v);


	int_vec_scan(pt_text, v, len);
	if (len == 2) {
		x3 = v[0];
		y3 = v[1];
		z3 = 1;
	}
	else if (len == 3) {
		x3 = v[0];
		y3 = v[1];
		z3 = v[2];
	}
	else {
		cout << "the point should have either two or three coordinates" << endl;
		exit(1);
	}
	FREE_int(v);


	n = F->elliptic_curve_discrete_log(
			EC_b, EC_c,
			x1, y1, z1,
			x3, y3, z3,
			verbose_level);


	cout << "The discrete log of (" << x3 << "," << y3 << "," << z3 << ") "
			"w.r.t. (" << x1 << "," << y1 << "," << z1 << ") "
			"is " << n << endl;

	FREE_OBJECT(F);

	if (f_v) {
		cout << "do_EC_multiple_of done" << endl;
	}
}

void algebra_global::do_EC_baby_step_giant_step(int EC_q, int EC_b, int EC_c,
		const char *EC_bsgs_G, int EC_bsgs_N, const char *EC_bsgs_cipher_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int Gx, Gy, Gz;
	int nGx, nGy, nGz;
	int Cx, Cy, Cz;
	int Mx, My, Mz;
	int Ax, Ay, Az;
	int *v;
	int len;
	int n;

	if (f_v) {
		cout << "algebra_global::do_EC_baby_step_giant_step" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(EC_q, 0 /*verbose_level*/);


	int_vec_scan(EC_bsgs_G, v, len);
	if (len != 2) {
		cout << "point should have just two coordinates" << endl;
		exit(1);
	}
	Gx = v[0];
	Gy = v[1];
	Gz = 1;
	FREE_int(v);

	n = (int) sqrt((double) EC_bsgs_N) + 1;
	if (f_v) {
		cout << "algebra_global::do_EC_baby_step_giant_step N = " << EC_bsgs_N << endl;
		cout << "algebra_global::do_EC_baby_step_giant_step n = " << n << endl;
	}

	int_vec_scan(EC_bsgs_cipher_text, v, len);

	int cipher_text_length = len >> 1;
	int h, i;

	if (f_v) {
		cout << "algebra_global::do_EC_baby_step_giant_step "
				"cipher_text_length = " << cipher_text_length << endl;
	}

	F->elliptic_curve_point_multiple(
			EC_b, EC_c, n,
			Gx, Gy, Gz,
			nGx, nGy, nGz,
			0 /*verbose_level*/);

	cout << "$" << n << " * G = (" << nGx << "," << nGy << ")$\\\\" << endl;

	cout << " & ";
	for (h = 0; h < cipher_text_length; h++) {
		Cx = v[2 * h + 0];
		Cy = v[2 * h + 1];
		Cz = 1;
		cout << " & (" << Cx << "," << Cy << ")";
	}
	cout << endl;

	for (i = 1; i <= n + 1; i++) {

		F->elliptic_curve_point_multiple(
				EC_b, EC_c, i,
				Gx, Gy, Gz,
				Mx, My, Mz,
				0 /*verbose_level*/);

		cout << i << " & (" << Mx << "," << My << ")";

		for (h = 0; h < cipher_text_length; h++) {
			Cx = v[2 * h + 0];
			Cy = v[2 * h + 1];
			Cz = 1;

			F->elliptic_curve_point_multiple(
					EC_b, EC_c, i,
					nGx, nGy, nGz,
					Mx, My, Mz,
					0 /*verbose_level*/);

			My = F->negate(My);



			F->elliptic_curve_addition(EC_b, EC_c,
					Cx, Cy, Cz,
					Mx, My, Mz,
					Ax, Ay, Az,
					0 /*verbose_level*/);

			cout << " & (" << Ax << "," << Ay << ")";

		}
		cout << "\\\\" << endl;
	}



	FREE_int(v);

	if (f_v) {
		cout << "algebra_global::do_EC_baby_step_giant_step done" << endl;
	}
}

void algebra_global::do_EC_baby_step_giant_step_decode(
		int EC_q, int EC_b, int EC_c,
		const char *EC_bsgs_A, int EC_bsgs_N,
		const char *EC_bsgs_cipher_text, const char *EC_bsgs_keys,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int Ax, Ay, Az;
	int Tx, Ty, Tz;
	int Cx, Cy, Cz;
	int Mx, My, Mz;
	int *v;
	int len;
	int n;
	int *keys;
	int nb_keys;
	int u, plain;

	if (f_v) {
		cout << "algebra_global::do_EC_baby_step_giant_step_decode" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(EC_q, 0 /*verbose_level*/);

	u = EC_q / 27;
	if (f_v) {
		cout << "algebra_global::do_EC_baby_step_giant_step_decode u = " << u << endl;
	}


	int_vec_scan(EC_bsgs_A, v, len);
	if (len != 2) {
		cout << "point should have just two coordinates" << endl;
		exit(1);
	}
	Ax = v[0];
	Ay = v[1];
	Az = 1;
	FREE_int(v);

	int_vec_scan(EC_bsgs_keys, keys, nb_keys);


	n = (int) sqrt((double) EC_bsgs_N) + 1;
	if (f_v) {
		cout << "algebra_global::do_EC_baby_step_giant_step_decode N = " << EC_bsgs_N << endl;
		cout << "algebra_global::do_EC_baby_step_giant_step_decode n = " << n << endl;
	}

	int_vec_scan(EC_bsgs_cipher_text, v, len);

	int cipher_text_length = len >> 1;
	int h;

	if (f_v) {
		cout << "algebra_global::do_EC_baby_step_giant_step_decode "
				"cipher_text_length = " << cipher_text_length << endl;
		cout << "algebra_global::do_EC_baby_step_giant_step_decode "
				"nb_keys = " << nb_keys << endl;
	}
	if (nb_keys != cipher_text_length) {
		cout << "nb_keys != cipher_text_length" << endl;
		exit(1);
	}


	for (h = 0; h < cipher_text_length; h++) {
		Tx = v[2 * h + 0];
		Ty = v[2 * h + 1];
		Tz = 1;
		cout << h << " & (" << Tx << "," << Ty << ")\\\\" << endl;;
	}
	cout << endl;


	for (h = 0; h < cipher_text_length; h++) {



		Tx = v[2 * h + 0];
		Ty = v[2 * h + 1];
		Tz = 1;


		F->elliptic_curve_point_multiple(
				EC_b, EC_c, keys[h],
				Ax, Ay, Az,
				Cx, Cy, Cz,
				0 /*verbose_level*/);

		Cy = F->negate(Cy);


		cout << h << " & " << keys[h]
			<< " & (" << Tx << "," << Ty << ")"
			<< " & (" << Cx << "," << Cy << ")";


		F->elliptic_curve_addition(EC_b, EC_c,
				Tx, Ty, Tz,
				Cx, Cy, Cz,
				Mx, My, Mz,
				0 /*verbose_level*/);

		cout << " & (" << Mx << "," << My << ")";

		plain = Mx / u;
		cout << " & " << plain << " & " << (char)('A' - 1 + plain) << "\\\\" << endl;

	}


	FREE_int(v);
	FREE_int(keys);

	if (f_v) {
		cout << "algebra_global::do_EC_baby_step_giant_step_decode done" << endl;
	}
}

void algebra_global::do_RSA_encrypt_text(long int RSA_d, long int RSA_m,
		int RSA_block_size, const char * RSA_encrypt_text, int verbose_level)
{
	int i, j, l, nb_blocks;
	long int a;
	char c;
	long int *Data;

	l = strlen(RSA_encrypt_text);
	nb_blocks = (l + RSA_block_size - 1) /  RSA_block_size;
	Data = NEW_lint(nb_blocks);
	for (i = 0; i < nb_blocks; i++) {
		a = 0;
		for (j = 0; j < RSA_block_size; j++) {
			c = RSA_encrypt_text[i * RSA_block_size + j];
			if (c >= 'a' && c <= 'z') {
				a *= 100;
				a += (int) (c - 'a') + 1;
			}
		Data[i] = a;
		}
	}

	longinteger_domain D;
	longinteger_object A, M;

	M.create(RSA_m, __FILE__, __LINE__);

	for (i = 0; i < nb_blocks; i++) {
		A.create(Data[i], __FILE__, __LINE__);
		D.power_int_mod(
				A, RSA_d, M);
		cout << A;
		if (i < nb_blocks - 1) {
			cout << ",";
		}
	}
	cout << endl;
}

void algebra_global::do_RSA(long int RSA_d, long int RSA_m, const char *RSA_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *data;
	int data_sz;
	int i;

	if (f_v) {
		cout << "do_RSA RSA_d=" << RSA_d << " RSA_m=" << RSA_m << endl;
	}
	lint_vec_scan(RSA_text, data, data_sz);
	if (f_v) {
		cout << "text: ";
		lint_vec_print(cout, data, data_sz);
		cout << endl;
	}

	longinteger_domain D;
	longinteger_object A, M;

	M.create(RSA_m, __FILE__, __LINE__);
	for (i = 0; i < data_sz; i++) {
		A.create(data[i], __FILE__, __LINE__);
		D.power_int_mod(
				A, RSA_d, M);
		cout << i << " : " << data[i] << " : " << A << endl;
	}
	for (i = 0; i < data_sz; i++) {
		A.create(data[i], __FILE__, __LINE__);
		D.power_int_mod(
				A, RSA_d, M);
		cout << A;
		if (i < data_sz - 1) {
			cout << ",";
		}
	}
	cout << endl;

	long int a;
	int b, j, h;
	char str[1000];

	for (i = 0; i < data_sz; i++) {
		A.create(data[i], __FILE__, __LINE__);
		D.power_int_mod(
				A, RSA_d, M);
		//cout << A;
		a = A.as_lint();
		j = 0;
		while (a) {
			b = a % 100;
			if (b > 26 || b == 0) {
				cout << "out of range" << endl;
				exit(1);
			}
			str[j] = 'a' + b - 1;
			j++;
			str[j] = 0;
			a -= b;
			a /= 100;
		}
		for (h = j - 1; h >= 0; h--) {
			cout << str[h];
		}
	}
	cout << endl;
}

void algebra_global::do_nullspace(int q,
		int m, int n, std::string &text,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int *M;
	int *A;
	int *base_cols;
	int len, rk, i, rk1;
	latex_interface Li;

	if (f_v) {
		cout << "do_nullspace" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

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


	cout << "after perp_standard:" << endl;
	int_matrix_print(A, n, n);
	cout << "rk=" << rk << endl;

	cout << "after RREF" << endl;
	rk1 = F->Gauss_int(A + rk * n,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, n - rk, n, n,
		0 /*verbose_level*/);


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

	if (f_normalize_from_the_left) {
		cout << "normalizing from the left" << endl;
		for (i = rk; i < n; i++) {
			F->PG_element_normalize_from_front(
					A + i * n, 1, n);
		}

		cout << "after normalize from the left:" << endl;
		int_matrix_print(A, n, n);
		cout << "rk=" << rk << endl;

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "normalizing from the right" << endl;
		for (i = rk; i < n; i++) {
			F->PG_element_normalize(
					A + i * n, 1, n);
		}

		cout << "after normalize from the right:" << endl;
		int_matrix_print(A, n, n);
		cout << "rk=" << rk << endl;

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;
	}


	FREE_int(M);
	FREE_int(A);
	FREE_int(base_cols);

	if (f_v) {
		cout << "do_nullspace done" << endl;
	}
}

void algebra_global::do_RREF(int q,
		int m, int n, std::string &text,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int *M;
	int *A;
	int *base_cols;
	int len, rk, i;
	latex_interface Li;

	if (f_v) {
		cout << "do_RREF" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

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

	if (f_normalize_from_the_left) {
		cout << "normalizing from the left" << endl;
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize_from_front(
					A + i * n, 1, n);
		}

		cout << "after normalize from the left:" << endl;
		int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "normalizing from the right" << endl;
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize(
					A + i * n, 1, n);
		}

		cout << "after normalize from the right:" << endl;
		int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;

	}


	FREE_int(M);
	FREE_int(A);
	FREE_int(base_cols);

	if (f_v) {
		cout << "do_RREF done" << endl;
	}
}

void algebra_global::do_weight_enumerator(int q,
		int m, int n, std::string &text,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int *M;
	int *A;
	int *base_cols;
	int *weight_enumerator;
	int len, rk, i;

	if (f_v) {
		cout << "do_weight_enumerator" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(text, M, len);
	if (len != m * n) {
		cout << "number of coordinates received differs from m * n" << endl;
		cout << "received " << len << endl;
		exit(1);
	}


	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	weight_enumerator = NEW_int(n + 1);
	int_vec_copy(M, A, m * n);

	rk = F->Gauss_int(A,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, m, n, n,
		0 /*verbose_level*/);


	cout << "after RREF:" << endl;
	int_matrix_print(A, rk, n);
	cout << "rk=" << rk << endl;

	cout << "coefficients:" << endl;
	int_vec_print(cout, A, rk * n);
	cout << endl;


	F->code_weight_enumerator(n, rk,
		A /* code */, // [k * n]
		weight_enumerator, // [n + 1]
		verbose_level);

	cout << "The weight enumerator is:" << endl;
	for (i = 0; i <= n; i++) {
		cout << i << " : " << weight_enumerator[i] << endl;
	}

	int f_first = TRUE;

	for (i = 0; i <= n; i++) {
		if (weight_enumerator[i] == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			cout << " + ";
		}
		cout << weight_enumerator[i];
		if (i) {
			cout << "*";
			cout << "x";
			if (i > 1) {
				cout << "^";
				if (i < 10) {
					cout << i;
				}
				else {
					cout << "(" << i << ")";
				}
			}
		}
		if (n - i) {
			cout << "*";
			cout << "y";
			if (n - i > 1) {
				cout << "^";
				if (n - i < 10) {
					cout << n - i;
				}
				else {
					cout << "(" << n - i << ")";
				}
			}
		}

	}
	cout << endl;


	if (f_normalize_from_the_left) {
		cout << "normalizing from the left" << endl;
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize_from_front(
					A + i * n, 1, n);
		}

		cout << "after normalize from the left:" << endl;
		int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "normalizing from the right" << endl;
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize(
					A + i * n, 1, n);
		}

		cout << "after normalize from the right:" << endl;
		int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;

	}


	FREE_int(M);
	FREE_int(A);
	FREE_int(base_cols);
	FREE_int(weight_enumerator);

	if (f_v) {
		cout << "do_weight_enumerator done" << endl;
	}
}

void algebra_global::do_trace(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int s, t;
	int *T0 = NULL;
	int *T1 = NULL;
	int nb_T0 = 0;
	int nb_T1 = 0;

	if (f_v) {
		cout << "do_trace" << endl;
	}
	F = NEW_OBJECT(finite_field);

	T0 = NEW_int(q);
	T1 = NEW_int(q);

	F->init(q, 0);
	for (s = 0; s < q; s++) {
		int s2, s3, s4, s8, s2p1, s2p1t7, s2p1t6, s2p1t4, f;

		s2 = F->mult(s, s);
		s2p1 = F->add(s2, 1);
		s2p1t7 = F->power(s2p1, 7);
		s2p1t6 = F->power(s2p1, 6);
		s2p1t4 = F->power(s2p1, 4);
		s3 = F->power(s, 3);
		s4 = F->power(s, 4);
		s8 = F->power(s, 8);

		f = F->add4(F->mult(s, s2p1t7), F->mult(s2, s2p1t6), F->mult(s4, s2p1t4), s8);

		//f = F->mult(top, F->inverse(bot));

		t = F->absolute_trace(f);
		//t = F->absolute_trace(s);
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


	FREE_OBJECT(F);
	if (f_v) {
		cout << "do_trace done" << endl;
	}
}

void algebra_global::do_norm(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int s, t;
	int *T0 = NULL;
	int *T1 = NULL;
	int nb_T0 = 0;
	int nb_T1 = 0;

	if (f_v) {
		cout << "do_norm" << endl;
	}
	F = NEW_OBJECT(finite_field);

	T0 = NEW_int(q);
	T1 = NEW_int(q);

	F->init(q, 0);
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


	FREE_OBJECT(F);
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





}}

