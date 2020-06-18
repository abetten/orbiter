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
		FQX.integral_division(h, Xma, quo, rem, 0);
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

void algebra_global::NumberTheoreticTransform(int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global::NumberTheoreticTransform" << endl;
		cout << "algebra_global::NumberTheoreticTransform k = " << k << " q=" << q << endl;
	}

	int alpha, omega; //, omega_power;
	int gamma, minus_gamma, minus_one;
	int idx, i, j, h;
	int **A;
	int **A_log;
	int *Omega;


	int **G;
	int **D;
	int **T;
	int **S;

	G = NEW_pint(k + 1);
	D = NEW_pint(k + 1);
	T = NEW_pint(k + 1);
	S = NEW_pint(k + 1);



	int *N;
	os_interface Os;
	finite_field *F = NULL;

	F = NEW_OBJECT(finite_field);
	F->init(q);


	minus_one = F->negate(1);
	alpha = F->primitive_root();
	if (f_v) {
		cout << "alpha = " << alpha << endl;
	}
	Omega = NEW_int(k + 1);
	A = NEW_pint(k + 1);
	A_log = NEW_pint(k + 1);
	N = NEW_int(k + 1);

	for (h = 0; h <= k; h++) {
		N[h] = 1 << h;
	}

	cout << "N[]:" << endl;
	int_matrix_print(N, k + 1, 1);

	idx = (q - 1) / N[k];
	omega = F->power(alpha, idx);
	if (f_v) {
		cout << "omega = " << omega << endl;
	}


	F->make_Fourier_matrices(omega, k, N, A, Omega, verbose_level);


	cout << "Omega:" << endl;
	int_matrix_print(Omega, k + 1, 1);

#if 0
	omega_power = omega;
	for (h = k; h >= 0; h--) {
		A[h] = NEW_int(N[h] * N[h]);
		for (i = 0; i < N[h]; i++) {
			for (j = 0; j < N[h]; j++) {
				A[h][i * N[h] + j] = F->power(omega_power, (i * j) % N[k]);
			}
		}
		omega_power = F->mult(omega_power, omega_power);
	}
	for (h = k; h >= 0; h--) {
		cout << "A_" << N[h] << ":" << endl;
		int_matrix_print(A[h], N[h], N[h]);
	}
#endif

	for (h = k; h >= 0; h--) {
		A_log[h] = NEW_int(N[h] * N[h]);
		for (i = 0; i < N[h]; i++) {
			for (j = 0; j < N[h]; j++) {
				A_log[h][i * N[h] + j] = F->log_alpha(A[h][i * N[h] + j]) + 1;
			}
		}
		cout << "A_" << N[h] << ":" << endl;
		int_matrix_print(A_log[h], N[h], N[h]);
	}

	int *X, *Y, *Z;
	int *X1, *X2;
	int *Y1, *Y2;

	X = NEW_int(N[k]);
	Y = NEW_int(N[k]);
	Z = NEW_int(N[k]);
	X1 = NEW_int(N[k - 1]);
	X2 = NEW_int(N[k - 1]);
	Y1 = NEW_int(N[k - 1]);
	Y2 = NEW_int(N[k - 1]);

	for (i = 0; i < N[k]; i++) {
		X[i] = Os.random_integer(q);
	}
	cout << "X:" << endl;
	int_matrix_print(X, 1, N[k]);
	cout << endl;


	int nb_m10, nb_m11, nb_a10, nb_a11;
	int nb_m20, nb_m21, nb_a20, nb_a21;
	int nb_m1, nb_a1;
	int nb_m2, nb_a2;

	nb_m10 = F->nb_times_mult_called();
	nb_a10 = F->nb_times_add_called();

	F->mult_vector_from_the_right(A[k], X, Y, N[k], N[k]);

	nb_m11 = F->nb_times_mult_called();
	nb_a11 = F->nb_times_add_called();

	nb_m1 = nb_m11 - nb_m10;
	nb_a1 = nb_a11 - nb_a10;

	cout << "nb_m1 = " << nb_m1 << " nb_a1 = " << nb_a1 << endl;


	cout << "Y:" << endl;
	int_matrix_print(Y, 1, N[k]);
	cout << endl;

	//omega_power = omega; //F->power(omega, 2);

	nb_m20 = F->nb_times_mult_called();
	nb_a20 = F->nb_times_add_called();


	for (i = 0; i < N[k - 1]; i++) {
		X1[i] = X[2 * i + 0];
		X2[i] = X[2 * i + 1];
	}


	F->mult_vector_from_the_right(A[k - 1], X1, Y1, N[k - 1], N[k - 1]);
	F->mult_vector_from_the_right(A[k - 1], X2, Y2, N[k - 1], N[k - 1]);

	gamma = 1;
	minus_gamma = minus_one;

	for (i = 0; i < N[k - 1]; i++) {

		Z[i] = F->add(Y1[i], F->mult(gamma, Y2[i]));
		Z[N[k - 1] + i] = F->add(Y1[i], F->mult(minus_gamma, Y2[i]));

		gamma = F->mult(gamma, omega);
		minus_gamma = F->negate(gamma);
	}

	nb_m21 = F->nb_times_mult_called();
	nb_a21 = F->nb_times_add_called();
	nb_m2 = nb_m21 - nb_m20;
	nb_a2 = nb_a21 - nb_a20;

	cout << "nb_m2 = " << nb_m2 << " nb_a2 = " << nb_a2 << endl;


	cout << "Z:" << endl;
	int_matrix_print(Z, 1, N[k]);
	cout << endl;

	for (i = 0; i < N[k]; i++) {
		 if (Y[i] != Z[i]) {
			 cout << "problem in component " << i << endl;
			 exit(1);
		 }
	}


	G[k] = NEW_int(N[k] * N[k]);
	D[k] = NEW_int(N[k] * N[k]);
	T[k] = NEW_int(N[k] * N[k]);
	S[k] = NEW_int(N[k] * N[k]);
	F->identity_matrix(G[k], N[k]);
	F->identity_matrix(D[k], N[k]);
	int_vec_copy(A[k], T[k], N[k] * N[k]);
	F->identity_matrix(S[k], N[k]);

	G[k - 1] = NEW_int(N[k] * N[k]);
	D[k - 1] = NEW_int(N[k] * N[k]);
	T[k - 1] = NEW_int(N[k] * N[k]);
	S[k - 1] = NEW_int(N[k] * N[k]);

	// G[k - 1]:
	int_vec_zero(G[k - 1], N[k] * N[k]);
	for (i = 0; i < N[k - 1]; i++) {
		G[k - 1][i * N[k] + i] = 1;
		G[k - 1][i * N[k] + N[k - 1] + i] = 1;
		G[k - 1][(N[k - 1] + i) * N[k] + i] = 1;
		G[k - 1][(N[k - 1] + i) * N[k] + N[k - 1] + i] = minus_one;
	}

	// D[k - 1]:
	int_vec_zero(D[k - 1], N[k] * N[k]);
	gamma = 1;
	for (i = 0; i < N[k - 1]; i++) {

		D[k - 1][i * N[k] + i] = 1;
		D[k - 1][(N[k - 1] + i) * N[k] + N[k - 1] + i] = gamma;


		//Z[i] = F->add(Y1[i], F->mult(gamma, Y2[i]));
		//Z[N[n - 1] + i] = F->add(Y1[i], F->mult(minus_gamma, Y2[i]));

		gamma = F->mult(gamma, omega);
		//minus_gamma = F->negate(gamma);
	}
	// T[k - 1]:
	int sz;
	int Id2[] = {1,0,0,1};

	F->Kronecker_product_square_but_arbitrary(A[k - 1], Id2,
			N[k - 1], 2, T[k - 1], sz, 0 /*verbose_level */);
	if (sz != N[k]]) {
		cout << "sz != N[k]" << endl;
		exit(1);
	}

	//S[k - 1]:
	int_vec_zero(S[k - 1], N[k] * N[k]);
	for (i = 0; i < N[k - 1]; i++) {
		S[k - 1][i * N[k] + 2 * i] = 1;
		S[k - 1][(N[k - 1] + i) * N[k] + 2 * i + 1] = 1;
	}


	cout << "G[k-1]:" << endl;
	int_matrix_print(G[k - 1], N[k], N[k]);
	cout << endl;
	cout << "D[k-1]:" << endl;
	int_matrix_print(D[k - 1], N[k], N[k]);
	cout << endl;
	cout << "T[k-1]:" << endl;
	int_matrix_print(T[k - 1], N[k], N[k]);
	cout << endl;
	cout << "S[k-1]:" << endl;
	int_matrix_print(S[k - 1], N[k], N[k]);
	cout << endl;

	file_io Fio;
	const char *fname_F = "ntt_F.csv";
	const char *fname_G = "ntt_G.csv";
	const char *fname_D = "ntt_D.csv";
	const char *fname_T = "ntt_T.csv";
	const char *fname_S = "ntt_S.csv";

	Fio.int_matrix_write_csv(fname_F, A[k], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_G, G[k - 1], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_D, D[k - 1], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_T, T[k - 1], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_S, S[k - 1], N[k], N[k]);

	cout << "Written file " << fname_F << " of size " << Fio.file_size(fname_F) << endl;
	cout << "Written file " << fname_G << " of size " << Fio.file_size(fname_G) << endl;
	cout << "Written file " << fname_D << " of size " << Fio.file_size(fname_D) << endl;
	cout << "Written file " << fname_T << " of size " << Fio.file_size(fname_T) << endl;
	cout << "Written file " << fname_S << " of size " << Fio.file_size(fname_S) << endl;




	int *Tmp1;
	int *Tmp2;

	Tmp1 = NEW_int(N[k] * N[k]);
	Tmp2 = NEW_int(N[k] * N[k]);

	F->mult_matrix_matrix(G[k - 1], D[k - 1], Tmp1, N[k], N[k], N[k], 0 /* verbose_level*/);
	F->mult_matrix_matrix(Tmp1, T[k - 1], Tmp2, N[k], N[k], N[k], 0 /* verbose_level*/);
	F->mult_matrix_matrix(Tmp2, S[k - 1], Tmp1, N[k], N[k], N[k], 0 /* verbose_level*/);

	for (i = 0; i < N[k] * N[k]; i++) {
		 if (A[k][i] != Tmp1[i]) {
			 cout << "matrix product differs from the Fourier matrix, problem in component " << i << endl;
			 exit(1);
		 }
	}


	if (f_v) {
		cout << "algebra_global::NumberTheoreticTransform done" << endl;
	}


}



}}

