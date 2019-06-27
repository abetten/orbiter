// factor_cyclotomic.cpp
//
// Anton Betten
// June 2007

#include "orbiter.h"

using namespace std;


using namespace orbiter;

void print_usage();
void factor_cyclotomic(int n, int q, int d,
	int *coeffs, int f_poly, char *poly, int verbose_level);


void print_usage()
{
	cout << "usage: factor_cyclotomic [options] "
			"n q d a_d a_d-1 ... a_0" << endl;
	cout << "where options can be:" << endl;
	cout << "-v  <n>                "
			": verbose level <n>" << endl;
	cout << "-poly  <m>             "
			": use polynomial <m> to create the field GF(q)" << endl;
}

int main(int argc, char **argv)
{
	//int t0 = os_ticks();
	int verbose_level = 0;
	int f_poly = FALSE;
	char *poly = NULL;
	int i, j;
	int n, q, d;

	if (argc <= 4) {
		print_usage();
		exit(1);
		}
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-f_poly " << poly << endl;
			}
		else if (argv[i][0] != '-')
			break;
		}
	
	n = atoi(argv[i]);
	q = atoi(argv[++i]);
	d = atoi(argv[++i]);
	int *coeffs;
	
	coeffs = NEW_int(d + 1);
	for (j = d; j >= 0; j--) {
		coeffs[j] = atoi(argv[++i]);
		}
	
	factor_cyclotomic(n, q, d, coeffs, f_poly, poly, verbose_level);
	
	FREE_int(coeffs);
}

void factor_cyclotomic(int n, int q, int d,
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

	if (f_v) {
		Beta = FQ.alpha_power(beta);
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
		if (NT.gcd_int(c, n) != 1)
			continue;
		C.create(c);
		N.create(n);
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
			while (roots2[i] < 0)
				roots2[i] += n;
			}
		Sorting.int_vec_quicksort_increasingly(roots2, nb_roots);
		t = 0;
		for (i = 0; i < nb_roots; i++) {
			r1 = roots2[i];
			for (j = i + 1; j < i + nb_roots; j++) {
				r2 = roots2[j % nb_roots];
				if (r2 != r1 + 1)
					break;
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
