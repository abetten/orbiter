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
		char *my_override_polynomial,
		int verbose_level)
{
	const char *override_poly;
	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "GF_%d.tex", q);
	snprintf(title, 1000, "Cheat Sheet GF($%d$)", q);
	author[0] = 0;
	author[0] = 0;
	if (f_override_polynomial) {
		override_poly = my_override_polynomial;
	}
	else {
		override_poly = NULL;
	}
	finite_field F;

	{
	ofstream f(fname);
	latex_interface L;

	//F.init(q), verbose_level - 2);
	F.init_override_polynomial(q, override_poly, verbose_level);
	L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);


	F.cheat_sheet(f, verbose_level);

	F.cheat_sheet_tables(f, verbose_level);

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

	return s;
}


void algebra_global::search_for_primitive_polynomials(
		int p_min, int p_max, int n_min, int n_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d, p;
	number_theory_domain NT;


	longinteger_f_print_scientific = FALSE;


	if (f_v) {
		cout << "algebra_global::search_for_primitive_polynomials "
				"p_min=" << p_min << " p_max=" << p_max
				<< " n_min=" << n_min << " n_max=" << n_max << endl;
	}
	for (p = p_min; p <= p_max; p++) {
		if (!NT.is_prime(p)) {
			continue;
		}
		if (f_v) {
			cout << "algebra_global::search_for_primitive_polynomials considering the prime " << p << endl;
		}

		{
			finite_field Fp;
			Fp.init(p, 0 /*verbose_level*/);
			unipoly_domain FX(&Fp);

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
				cout << "algebra_global::search_for_primitive_polynomials before FX.delete_object(m)" << endl;
			}
			FX.delete_object(m);
			if (f_v) {
				cout << "algebra_global::search_for_primitive_polynomials after FX.delete_object(m)" << endl;
			}
		}
	}
	if (f_v) {
		cout << "algebra_global::search_for_primitive_polynomials done" << endl;
	}
}


void algebra_global::factor_cyclotomic(int n, int q, int d,
	int *coeffs, int f_poly, char *poly, int verbose_level)
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

void algebra_global::make_Hamming_graph_and_write_file(int n, int q, int f_projective, int verbose_level)
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

	char fname[1000];
	file_io Fio;

	snprintf(fname, 1000, "Hamming_n%d_q%d.csv", n, q);

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





}}

