// unipoly_domain.cpp
//
// Anton Betten
//
// started:  November 16, 2002




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace ring_theory {


unipoly_domain::unipoly_domain()
{
	F = NULL;
	variable_name.assign("X");
	f_factorring = FALSE;
	factor_degree = 0;
	factor_coeffs = NULL;
	factor_poly = NULL;
	f_print_sub = FALSE;
	//f_use_variable_name = FALSE;
	//std::string variable_name;
}

unipoly_domain::unipoly_domain(field_theory::finite_field *F)
{
	unipoly_domain::F = F;
	variable_name.assign("X");
	f_factorring = FALSE;
	factor_degree = 0;
	factor_coeffs = NULL;
	factor_poly = NULL;
	f_print_sub = FALSE;
	//f_use_variable_name = FALSE;
	//std::string variable_name;
}

void unipoly_domain::init_basic(field_theory::finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "unipoly_domain::init_basic" << endl;
	}
	unipoly_domain::F = F;
	variable_name.assign("X");
	f_factorring = FALSE;
	factor_degree = 0;
	factor_coeffs = NULL;
	factor_poly = NULL;
	f_print_sub = FALSE;
	//f_use_variable_name = FALSE;
	//std::string variable_name;
	if (f_v) {
		cout << "unipoly_domain::init_basic done" << endl;
	}
}

unipoly_domain::unipoly_domain(field_theory::finite_field *F, unipoly_object m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, a, b;
	
	unipoly_domain::F = F;
	variable_name.assign("X");
	f_print_sub = FALSE;
	f_factorring = FALSE;

	if (f_v) {
		cout << "unipoly_domain::unipoly_domain creating factorring modulo " << endl;
		print_object(m, cout);
		cout << endl;
		//cout << " of degree " << ((int *)m)[0] << endl;
	}

#if 0
	variable_name.assign("X");
	f_factorring = TRUE;
	factor_degree = ((int *)m)[0];
	factor_coeffs = NEW_int(factor_degree + 1);
	for (i = 0; i <= factor_degree; i++) {
		factor_coeffs[i] = ((int *)m)[1 + i];
	}
	//factor_coeffs = ((int *)m) + 1;
	if (factor_coeffs[factor_degree] != 1) {
		cout << "unipoly_domain::unipoly_domain "
				"factor polynomial is not monic" << endl;
		exit(1);
	}
	for (i = 0; i < factor_degree; i++) {
		a = factor_coeffs[i];
		b = F->negate(a);
		factor_coeffs[i] = b;
	}
	create_object_of_degree_no_test(factor_poly, factor_degree);
	for (i = 0; i <= factor_degree; i++) {
		((int *)factor_poly)[1 + i] = ((int *)m)[1 + i];
	}
	//factor_poly = m;
	if (f_v) {
		cout << "unipoly_domain::unipoly_domain factor_coeffs = ";
		Orbiter->Int_vec.print(cout, factor_coeffs, factor_degree + 1);
		cout << endl;
	}
	f_print_sub = FALSE;
#else

	init_factorring(F, m, verbose_level);

#endif
	if (f_v) {
		cout << "unipoly_domain::unipoly_domain creating factorring done" << endl;
	}
}

unipoly_domain::~unipoly_domain()
{
	//int i, a, b;

	if (f_factorring) {
		FREE_int(factor_coeffs);
		delete_object(factor_poly);
	}
}

void unipoly_domain::init_variable_name(std::string &label)
{
	variable_name.assign(label);
}

void unipoly_domain::init_factorring(field_theory::finite_field *F, unipoly_object m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, b;

	if (f_v) {
		cout << "unipoly_domain::init_factorring creating factorring modulo ";
		print_object(m, cout);
		cout << " of degree " << ((int *)m)[0] << endl;
	}
	unipoly_domain::F = F;
	variable_name.assign("X");
	f_factorring = TRUE;
	factor_degree = ((int *)m)[0];
	factor_coeffs = NEW_int(factor_degree + 1);
	for (i = 0; i <= factor_degree; i++) {
		factor_coeffs[i] = ((int *)m)[1 + i];
	}
	//factor_coeffs = ((int *)m) + 1;
	if (factor_coeffs[factor_degree] != 1) {
		cout << "unipoly_domain::init_factorring "
				"factor polynomial is not monic" << endl;
		exit(1);
	}
	for (i = 0; i < factor_degree; i++) {
		a = factor_coeffs[i];
		b = F->negate(a);
		factor_coeffs[i] = b;
	}
	create_object_of_degree_no_test(factor_poly, factor_degree);
	for (i = 0; i <= factor_degree; i++) {
		((int *)factor_poly)[1 + i] = ((int *)m)[1 + i];
	}
	//factor_poly = m;
	if (f_v) {
		cout << "unipoly_domain::init_factorring factor_coeffs = ";
		Int_vec_print(cout, factor_coeffs, factor_degree + 1);
		cout << endl;
	}
	f_print_sub = FALSE;
	if (f_v) {
		cout << "unipoly_domain::init_factorring done" << endl;
	}
}




field_theory::finite_field *unipoly_domain::get_F()
{
	return F;
}

void unipoly_domain::create_object_of_degree(
		unipoly_object &p, int d)
{
	if (f_factorring) {
		cout << "unipoly_domain::create_object_of_degree "
				"a factorring" << endl;
		exit(1);
	}
	create_object_of_degree_no_test(p, d);
}

void unipoly_domain::create_object_of_degree_no_test(
		unipoly_object &p, int d)
{
	int *rep = NEW_int(d + 2);
	rep[0] = d;
	int *coeff = rep + 1;
	int i;
	
	for (i = 0; i <= d; i++) {
		coeff[i] = 0;
	}
	rep[0] = d;
	p = (void *) rep;
}

void unipoly_domain::create_object_of_degree_with_coefficients(
		unipoly_object &p, int d, int *coeff)
{
	if (f_factorring) {
		cout << "unipoly_domain::create_object_of_degree_with_coefficients a factorring" << endl;
		exit(1);
	}
	int *rep = NEW_int(d + 2);
	rep[0] = d;
	int *C = rep + 1;
	int i;
	
	for (i = 0; i <= d; i++) {
		C[i] = coeff[i];
	}
	rep[0] = d;
	p = (void *) rep;
}

void unipoly_domain::create_object_by_rank(
	unipoly_object &p, long int rk,
	const char *file, int line, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;
	
	if (f_v) {
		cout << "unipoly_domain::create_object_by_rank rk=" << rk << endl;
		cout << "unipoly_domain::create_object_by_rank f_factorring=" << f_factorring << endl;
	}

	int len = NT.lint_logq(rk, F->q);

	if (f_factorring) {
		if (len > factor_degree) {
			cout << "unipoly_domain::create_object_by_rank "
					"len > factor_degree" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "unipoly_domain::create_object_by_rank modulo - (";
			Int_vec_print(cout,
					factor_coeffs,
					factor_degree + 1);
			cout << ")" << endl;
		}
		len = factor_degree;
		int *rep = NEW_int_with_tracking(len + 1, file, line);
		rep[0] = len - 1;
		int *coeff = rep + 1;
		int i = 0;

		for (i = 0; i < factor_degree; i++) {
			coeff[i] = rk % F->q;
			rk /= F->q;
		}
		rep[0] = factor_degree - 1; //i - 1;
		p = (void *) rep;
	}
	else {
		int *rep = NEW_int_with_tracking(len + 1, file, line);
		rep[0] = len - 1;
		int *coeff = rep + 1;
		int i = 0;

		do {
			coeff[i] = rk % F->q;
			rk /= F->q;
			i++;
		} while (rk);
		rep[0] = i - 1;
		p = (void *) rep;
	}
	if (f_v) {
		cout << "unipoly_domain::create_object_by_rank done" << endl;
	}
}

void unipoly_domain::create_object_from_csv_file(
	unipoly_object &p, std::string &fname,
	const char *file, int line,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;
	orbiter_kernel_system::file_io Fio;
	int m, n, len;
	long int *M;

	if (f_v) {
		cout << "unipoly_domain::create_object_from_csv_file" << endl;
	}
	if (f_factorring) {
		cout << "unipoly_domain::create_object_from_csv_file "
					"f_factorring" << endl;
		exit(1);
	}
	Fio.lint_matrix_read_csv(fname, M, m, n, 0 /* verbose_level */);
	len = m * n;


	int *rep = NEW_int_with_tracking(len + 1, file, line);
	rep[0] = len - 1;
	int *coeff = rep + 1;
	int i = 0;

	for (i = 0; i < len; i++) {
		coeff[len - 1 - i] = M[i];
	}
	//rep[0] = i - 1;
	p = (void *) rep;
	FREE_lint(M);
}

void unipoly_domain::create_object_by_rank_longinteger(
	unipoly_object &p,
	longinteger_object &rank,
	const char *file, int line,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object rk, rk1;
	longinteger_domain D;
	
	int len = D.logarithm_base_b(rank, F->q);
	//cout << "len = " << len << endl;
	
	if (f_v) {
		cout << "unipoly_domain::create_object_by_rank_longinteger rank=" << rank << endl;
	}
	if (f_factorring) {
		if (len > factor_degree) {
			cout << "unipoly_domain::create_object_by_rank_longinteger len > factor_degree" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "unipoly_domain::create_object_by_rank_longinteger modulo - (";
			Int_vec_print(cout,
					factor_coeffs,
					factor_degree + 1);
			cout << ")" << endl;
		}
		len = factor_degree;
	}
	int *rep = NEW_int_with_tracking(len + 1, file, line);
	rep[0] = len - 1;
	int *coeff = rep + 1;
	int i = 0;
	
	rank.assign_to(rk);
	do {
		D.integral_division_by_int(rk, F->q, rk1, coeff[i]);
		//cout << "rk=" << rk << " coeff[" << i
		// << "] = " << coeff[i] << endl;
		// coeff[i] = rk % F->q;
		if (f_vv) {
			cout << "unipoly_domain::create_object_by_rank_longinteger "
					"i=" << i << " rk=" << rk << " quotient " << rk1
					<< " remainder " << coeff[i] << endl;
		}
		rk1.assign_to(rk);
		//rk /= F->q;
		i++;
	} while (!rk.is_zero());
	rep[0] = i - 1;
	p = (void *) rep;
	//print_object(p, cout); cout << endl;
}

void unipoly_domain::create_object_by_rank_string(
	unipoly_object &p, std::string &rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object rank;
	
	rank.create_from_base_10_string(rk);
	
	create_object_by_rank_longinteger(p, rank, __FILE__, __LINE__, verbose_level);
	if (f_v) {
		cout << "unipoly_domain::create_object_by_rank_string ";
		print_object(p, cout);
		cout << endl;
	}
}

void unipoly_domain::create_Dickson_polynomial(
		unipoly_object &p, int *map)
{
	if (f_factorring) {
		cout << "unipoly_domain::create_Dickson_polynomial "
				"a factorring" << endl;
		exit(1);
	}
	int d = F->q - 1;
	int *rep = NEW_int(d + 2);
	rep[0] = d;
	int *coeff = rep + 1;
	
	F->Linear_algebra->Dickson_polynomial(map, coeff);
	rep[0] = d;
	p = (void *) rep;
	degree(p);
}

void unipoly_domain::delete_object(unipoly_object &p)
{
	int *rep = (int *) p;
	FREE_int(rep);
	p = NULL;
}

void unipoly_domain::unrank(unipoly_object p, int rk)
{
	int *rep = (int *) p;
	int *coeff = rep + 1;
	int i = 0;
	
	do {
		coeff[i] = rk % F->q;
		rk /= F->q;
		i++;
		} while (rk);
	rep[0] = i - 1;
}

void unipoly_domain::unrank_longinteger(
		unipoly_object p, longinteger_object &rank)
{
	int *rep = (int *) p;
	int *coeff = rep + 1;
	int i = 0;
	
	longinteger_object rank1, rank2;
	longinteger_domain D;
	
	rank.assign_to(rank1);
	do {
		D.integral_division_by_int(rank1, F->q, rank2, coeff[i]);
		//coeff[i] = rk % F->q;
		//rk /= F->q;
		rank2.assign_to(rank1);
		i++;
	} while (!rank1.is_zero());
	rep[0] = i - 1;
}

int unipoly_domain::rank(unipoly_object p)
{
	int *rep = (int *) p;
	int d = rep[0]; // degree
	int *coeff = rep + 1;
	int rk = 0;
	int i;
	
	for (i = d; i >= 0; i--) {
		rk *= F->q;
		rk += coeff[i];
	}
	return rk;
}

void unipoly_domain::rank_longinteger(
	unipoly_object p, longinteger_object &rank)
{
	int *rep = (int *) p;
	int d = rep[0]; // degree
	int *coeff = rep + 1;
	int i;
	longinteger_object q, rk, rk1, c;
	longinteger_domain D;
	
	rk.create(0, __FILE__, __LINE__);
	q.create(F->q, __FILE__, __LINE__);
	for (i = d; i >= 0; i--) {
		D.mult(rk, q, rk1);
		c.create(coeff[i], __FILE__, __LINE__);
		D.add(rk1, c, rk);
		//rk *= F->q;
		//rk += coeff[i];
	}
	rk.assign_to(rank);
}

int unipoly_domain::degree(unipoly_object p)
{
	int *rep = (int *) p;
	int d = rep[0]; // degree
	int *coeff = rep + 1;
	int i;
	
	for (i = d; i >= 0; i--) {
		if (coeff[i]) {
			break;
		}
	}
	rep[0] = i;
	return i;
}


void unipoly_domain::print_object(unipoly_object p, std::ostream &ost)
{
	int i, k;
	int f_prev = FALSE;
	string x, y;
	int f_nothing_printed_at_all = TRUE;
	int *rep = (int *) p;
	int d = rep[0]; // degree
	int *coeff = rep + 1;
	
	x.assign(variable_name);
	if (f_print_sub) {
		y.assign("_");
	}
	else {
		y.assign("^");
	}
	// ost << "(";
	for (i = d; i >= 0; i--) {
		k = coeff[i];
		if (k == 0) {
			continue;
		}
		f_nothing_printed_at_all = FALSE;
		if (f_prev) {
			ost << " + ";
		}
		if (k != 1 || (i == 0 /*&& !unip_f_use_variable_name*/)) {
			//l = F->log_alpha(k);
			//ost << "\\alpha^{" << l << "}";
			ost << k;
		}
		if (i == 0) {
			//ost << x;
			//ost << y;
			//ost << "0";
		}
		else if (i == 1) {
			ost << x;
			if (f_print_sub) {
				ost << y;
				ost << "1";
			}
		}
		else if (i > 1) {
			ost << x;
			ost << y;
			ost << "{" << i << "}";
		}
		f_prev = TRUE;
	}
	if (f_nothing_printed_at_all) {
		ost << "0";
	}
	// ost << ")";
	//return ost;
}

void unipoly_domain::print_object_tight(unipoly_object p, std::ostream &ost)
{
	int i, k;
	int *rep = (int *) p;
	int d = rep[0]; // degree
	int *coeff = rep + 1;


	for (i = 0; i <= d; i++) {
		k = coeff[i];
		ost << k;
	}
}

void unipoly_domain::print_object_sparse(unipoly_object p, std::ostream &ost)
{
	int i, a;
	int *rep = (int *) p;
	int d = rep[0]; // degree
	int *coeff = rep + 1;
	int f_first = TRUE;


	for (i = 0; i <= d; i++) {
		a = coeff[i];

		if (a) {
			if (f_first) {
				f_first = FALSE;
			}
			else {
				ost << ",";
			}
			ost << a << "," << i;
		}
	}
}

void unipoly_domain::print_object_dense(unipoly_object p, std::ostream &ost)
{
	int i, a;
	int *rep = (int *) p;
	int d = rep[0]; // degree
	int *coeff = rep + 1;


	for (i = 0; i <= d; i++) {
		a = coeff[i];

		ost << a;
		if (i < d) {
			ost << ",";
		}
	}
}



void unipoly_domain::assign(unipoly_object a, unipoly_object &b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "unipoly_domain::assign" << endl;
	}
	if (f_factorring) {
		if (f_v) {
			cout << "unipoly_domain::assign with factorring";
			cout << " modulo - (";
			Int_vec_print(cout,
					factor_coeffs,
					factor_degree + 1);
			cout << ")" << endl;
		}
		if (a == NULL) {
			cout << "unipoly_domain::assign with factorring a == NULL" << endl;
			exit(1);
		}
		if (b == NULL) {
			cout << "unipoly_domain::assign with factorring b == NULL" << endl;
			exit(1);
		}
		int *ra = (int *) a;
		int *rb = (int *) b;
		if (rb[0] < factor_degree - 1) {
			cout << "unipoly_domain::assign rb[0] < factor_degree - 1" << endl;
			cout << "rb[0] = " << rb[0] << endl;
			cout << "factor_degree = " << factor_degree << endl;
			exit(1);
		}
		int *A = ra + 1;
		int *B = rb + 1;
		int i;
		for (i = 0; i < factor_degree; i++) {
			B[i] = A[i];
		}
		rb[0] = ra[0];
		if (f_v) {
			cout << "unipoly_domain::assign after copy" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "unipoly_domain::assign not a factorring" << endl;
		}
		int *ra = (int *) a;
		int *rb = (int *) b;
		int m = ra[0];
		FREE_int(rb);
		rb = NEW_int(m + 2);
		rb[0] = m;
		b = (void *) rb;
		if (f_v) {
			cout << "unipoly_domain::assign m=" << m << endl;
			cout << "a=";
			print_object(a, cout);
			cout << endl;
		}
		int *A = ra + 1;
		int *B = rb + 1;
		int i;
		for (i = 0; i <= m; i++) {
			B[i] = A[i];
			if (f_v) {
				cout << "i=" << i << " A[i]=" << A[i]
					<< " B[i]=" << B[i] << endl;
			}
		}
		if (f_v) {
			cout << "unipoly_domain::assign after copy" << endl;
		}
	}
	if (f_v) {
		cout << "unipoly_domain::assign done" << endl;
	}
}

void unipoly_domain::one(unipoly_object p)
{
	int *rep = (int *) p;
	int d = rep[0]; // degree
	int *coeff = rep + 1;
	int i;
	
	for (i = d; i >= 1; i--) {
		coeff[i] = 0;
	}
	coeff[0] = 1;
	rep[0] = 0;
}

void unipoly_domain::m_one(unipoly_object p)
{
	int *rep = (int *) p;
	int d = rep[0]; // degree
	int *coeff = rep + 1;
	int i;
	
	for (i = d; i >= 1; i--) {
		coeff[i] = 0;
	}
	coeff[0] = F->negate(1);
	rep[0] = 0;
}

void unipoly_domain::zero(unipoly_object p)
{
	int *rep = (int *) p;
	int d = rep[0]; // degree
	int *coeff = rep + 1;
	int i;
	
	for (i = d; i >= 1; i--) {
		coeff[i] = 0;
	}
	coeff[0] = 0;
	rep[0] = 0;
}

int unipoly_domain::is_one(unipoly_object p)
{
	int *rep = (int *) p;
	if (rep == NULL) {
		cout << "unipoly_domain::is_one rep == NULL" << endl;
		exit(1);
	}
	int d = rep[0]; // degree
	int *coeff = rep + 1;
	int i;
	
	for (i = d; i >= 1; i--) {
		if (coeff[i]) {
			return FALSE;
		}
	}
	if (coeff[0] != 1) {
		return FALSE;
	}
	return TRUE;
}

int unipoly_domain::is_zero(unipoly_object p)
{
	int *rep = (int *) p;
	if (rep == NULL) {
		cout << "unipoly_domain::is_zero rep == NULL" << endl;
		exit(1);
	}
	int d = rep[0]; // degree
	int *coeff = rep + 1;
	int i;
	
	for (i = d; i >= 0; i--) {
		if (coeff[i]) {
			return FALSE;
		}
	}
	return TRUE;
}

void unipoly_domain::negate(unipoly_object a)
{
	int *ra = (int *) a;
	if (ra == NULL) {
		cout << "unipoly_domain::negate rep == NULL" << endl;
		exit(1);
	}
	int m = ra[0];
	int *A = ra + 1;
	int i;
	
	for (i = 0; i <= m; i++) {
		A[i] = F->negate(A[i]);
	}
}

void unipoly_domain::make_monic(unipoly_object &a)
{
	int *ra = (int *) a;
	if (ra == NULL) {
		cout << "unipoly_domain::make_monic rep == NULL" << endl;
		exit(1);
	}
	int m = ra[0];
	int *A = ra + 1;
	int i, c, cv;

	while (A[m] == 0 && m > 0) {
		m--;
	}
	if (m == 0 && A[0] == 0) {
		cout << "unipoly_domain::make_monic "
				"the polynomial is zero" << endl;
		exit(1);
	}
	c = A[m];
	if (c != 1) {
		cv = F->inverse(c);
		for (i = 0; i <= m; i++) {
			A[i] = F->mult(A[i], cv);
		}
	}
}

void unipoly_domain::add(unipoly_object a,
		unipoly_object b, unipoly_object &c)
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	int *ra = (int *) a;
	int *rb = (int *) b;
	int m = ra[0];
	int n = rb[0];
	int mn = MAXIMUM(m, n);
	
	int *rc = (int *) c;
	FREE_int(rc);
	rc = NEW_int(mn + 2);
	
	int *A = ra + 1;
	int *B = rb + 1;
	int *C = rc + 1;
	int i, x, y;
	
	rc[0] = mn;
	for (i = 0; i <= MAXIMUM(m, n); i++) {
		if (i <= m) {
			x = A[i];
		}
		else {
			x = 0;
		}
		if (i <= n) {
			y = B[i];
		}
		else {
			y = 0;
		}
		if (f_v) {
			cout << "unipoly_domain::add "
					"x=" << x << " y=" << y << endl;
		}
		C[i] = F->add(x, y);
	}
	c = (void *) rc;
}

void unipoly_domain::mult(unipoly_object a,
		unipoly_object b, unipoly_object &c, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "unipoly_domain::mult" << endl;
	}
	if (f_factorring) {
		if (f_v) {
			cout << "unipoly_domain::mult before mult_mod" << endl;
		}
		mult_mod_negated(a, b, c, factor_degree, factor_coeffs, verbose_level);
		if (f_v) {
			cout << "unipoly_domain::mult after mult_mod" << endl;
		}
	}
	else {
		mult_easy(a, b, c);
	}
	if (f_v) {
		cout << "unipoly_domain::mult done" << endl;
	}
}

void unipoly_domain::mult_mod(unipoly_object a,
	unipoly_object b, unipoly_object &c, unipoly_object m,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "unipoly_domain::mult_mod" << endl;
	}

	int d;
	int *factor_polynomial_coefficients_negated;
	int i;


	d = degree(m);
	factor_polynomial_coefficients_negated = NEW_int(d + 1);
	for (i = 0; i <= d; i++) {
		factor_polynomial_coefficients_negated[i] = F->negate(s_i(m, i));
	}

	mult_mod_negated(a, b, c,
		d,
		factor_polynomial_coefficients_negated,
		verbose_level);

	FREE_int(factor_polynomial_coefficients_negated);

	if (f_v) {
		cout << "unipoly_domain::mult_mod done" << endl;
	}
}

void unipoly_domain::mult_mod_negated(unipoly_object a,
	unipoly_object b, unipoly_object &c,
	int factor_polynomial_degree,
	int *factor_polynomial_coefficients_negated,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *ra = (int *) a;
	int *rb = (int *) b;
	int *rc = (int *) c;
	int m = ra[0];
	int n = rb[0];
	int *A = ra + 1;
	int *B = rb + 1;
	int *C = rc + 1;
	int carry, c1;
	int i, j;
	
	if (f_v) {
		cout << "unipoly_domain::mult_mod_negated" << endl;
	}
	if (f_vv) {
		cout << "unipoly_domain::mult_mod_negated computing ";
		print_object(ra, cout);
		cout << " x ";
		print_object(rb, cout);
		cout << " modulo - (";
		Int_vec_print(cout,
				factor_polynomial_coefficients_negated,
			factor_polynomial_degree + 1);
		cout << ")";
		cout << endl;
	}
#if 0
	if (!f_factorring) {
		cout << "unipoly_domain::mult_mod_negated not a factorring" << endl;
		exit(1);
	}
#endif
	if (rc[0] != factor_polynomial_degree - 1) {
		FREE_int(rc);
		rc = NEW_int(factor_polynomial_degree - 1 + 2);
		rc[0] = factor_polynomial_degree - 1;
		c = rc;
		C = rc + 1;
	}
	for (j = 0 ; j < factor_polynomial_degree; j++) {
		C[j] = 0;
	}
	
	for (i = m; i >= 0; i--) {
		for (j = 0; j <= n; j++) {
			c1 = F->mult(A[i], B[j]);
			C[j] = F->add(C[j], c1);
			if (f_vv) {
				if (c1) {
					cout << A[i] << "x^" << i << " * "
						<< B[j] << " x^" << j << " = "
						<< c1 << " x^" << i + j << " = ";
					print_object(rc, cout);
					cout << endl;
				}
			}
		}
		//cout << "i=" << i << " ";
		//print_object(C, cout);
		//cout << endl;
		
		if (i > 0) {
			carry = C[factor_polynomial_degree - 1];
			for (j = factor_polynomial_degree - 1; j > 0; j--) {
				C[j] = C[j - 1];
			}
			C[0] = 0;
			if (carry) {
				if (carry == 1) {
					for (j = 0; j < factor_polynomial_degree; j++) {
						C[j] = F->add(C[j],
								factor_polynomial_coefficients_negated[j]);
					}
				}
				else {
					for (j = 0; j < factor_polynomial_degree; j++) {
						c1 = F->mult(carry,
								factor_polynomial_coefficients_negated[j]);
						C[j] = F->add(C[j], c1);
					}
				}
			}
		}
	}
	c = rc;
	if (f_v) {
		cout << "unipoly_domain::mult_mod_negated done" << endl;
	}
}

void unipoly_domain::Frobenius_matrix_by_rows(int *&Frob,
	unipoly_object factor_polynomial, int verbose_level)
// the j-th row of Frob is x^{j*q} mod m
{
	int f_v = (verbose_level >= 1);
	int d;

	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix_by_rows" << endl;
	}
	d = degree(factor_polynomial);
	Frobenius_matrix(Frob, factor_polynomial, verbose_level);
	F->Linear_algebra->transpose_matrix_in_place(Frob, d);
	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix_by_rows done" << endl;
	}
}

void unipoly_domain::Frobenius_matrix(int *&Frob,
	unipoly_object factor_polynomial, int verbose_level)
// the j-th column of Frob is x^{j*q} mod m

{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = FALSE;

	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix" << endl;
	}
	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix "
			"q=" << F->q << endl;
	}
#if 0
	if (!f_factorring) {
		cout << "unipoly_domain::Frobenius_matrix "
			"not a factorring" << endl;
		exit(1);
	}
#endif
	unipoly_object a, b, c, m_mod, Q, R;
	int i, j, d1;
	int factor_polynomial_degree;
	
	factor_polynomial_degree = degree(factor_polynomial);
	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix "
			"degree=" << factor_polynomial_degree << endl;
		cout << "unipoly_domain::Frobenius_matrix m = ";
		print_object(factor_polynomial, cout);
		cout << endl;
	}

	Frob = NEW_int(factor_polynomial_degree * factor_polynomial_degree);
	Int_vec_zero(Frob,
			factor_polynomial_degree * factor_polynomial_degree);
	Frob[0] = 1; // the first column of Frob is (1,0,...,0)
	
	create_object_by_rank(a, F->q, __FILE__, __LINE__, 0 /*verbose_level*/); // the polynomial X
	create_object_by_rank(b, 1, __FILE__, __LINE__, 0 /*verbose_level*/); // the polynomial 1
	create_object_by_rank(c, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(m_mod, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(Q, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(R, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	
	assign(factor_polynomial, m_mod, 0 /*verbose_level */);
	negate(m_mod);
	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix m_mod = ";
		print_object(m_mod, cout);
		cout << endl;
	}
	
	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix before power_int" << endl;
	}
	power_int(a, F->q, 0 /*verbose_level*/);
	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix after power_int" << endl;
		cout << "unipoly_domain::Frobenius_matrix a = x^q = ";
		print_object(a, cout);
		cout << endl;
	}
	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix before division_with_remainder" << endl;
	}
	division_with_remainder(a,
			factor_polynomial,
			Q, R, 0 /*verbose_level*/);
	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix after division_with_remainder" << endl;
	}
	assign(R, a, 0 /*verbose_level */);
	if (f_vv) {
		cout << "unipoly_domain::Frobenius_matrix "
				"a(X) = X^q mod m(X) = ";
		print_object(a, cout);
		cout << endl;
	}
	for (j = 1; j < factor_polynomial_degree; j++) {
		if (f_vvv) {
			cout << "unipoly_domain::Frobenius_matrix j = " << j << endl;
			cout << "b = ";
			print_object(b, cout);
			cout << endl;
			cout << "a = ";
			print_object(a, cout);
			cout << endl;
		}
		mult_mod_negated(b, a, c,
				factor_polynomial_degree, ((int *)m_mod) + 1, 0);
		if (f_vvv) {
			cout << "c = ";
			print_object(c, cout);
			cout << endl;
		}
		assign(c, b, 0 /*verbose_level */);
		// now b = X^{j*q}
		if (f_vvv) {
			cout << "unipoly_domain::Frobenius_matrix X^{" << j << "*q}=";
			print_object(b, cout);
			cout << endl;
		}
		d1 = degree(b);
		int *rb = (int *) b;
		int *B = rb + 1;

		// put B in the j-th column of F:
		for (i = 0; i <= d1; i++) {
			Frob[i * factor_polynomial_degree + j] = B[i];
		}
	}
	if (f_vv) {
		cout << "unipoly_domain::Frobenius_matrix=" << endl;
		Int_matrix_print(Frob,
			factor_polynomial_degree, factor_polynomial_degree);
		cout << endl;
	}
	delete_object(a);
	delete_object(b);
	delete_object(c);
	delete_object(m_mod);
	delete_object(Q);
	delete_object(R);
	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix done" << endl;
	}
}

void unipoly_domain::Berlekamp_matrix(int *&B,
	unipoly_object factor_polynomial, int verbose_level)
// subtracts the identity matrix off the Frobenius matrix
{
	int f_v = (verbose_level >= 1);
	int i, m1, a, b;
	int factor_polynomial_degree;
	
	if (f_v) {
		cout << "unipoly_domain::Berlekamp_matrix" << endl;
	}
	factor_polynomial_degree = degree(factor_polynomial);
	if (f_v) {
		cout << "unipoly_domain::Berlekamp_matrix before Frobenius_matrix" << endl;
	}
	Frobenius_matrix(B, factor_polynomial, verbose_level - 2);
	if (f_v) {
		cout << "unipoly_domain::Berlekamp_matrix after Frobenius_matrix" << endl;
		cout << "Frobenius matros:" << endl;
		Int_vec_print_integer_matrix(cout, B,
				factor_polynomial_degree, factor_polynomial_degree);
	}
	m1 = F->negate(1);
	
	for (i = 0; i < factor_polynomial_degree; i++) {
		a = B[i * factor_polynomial_degree + i];
		b = F->add(m1, a);
		B[i * factor_polynomial_degree + i] = b;
	}
	if (f_v) {
		cout << "unipoly_domain::Berlekamp_matrix "
				"of degree " << factor_polynomial_degree
				<< " = " << endl;
		Int_vec_print_integer_matrix(cout, B,
				factor_polynomial_degree, factor_polynomial_degree);
		cout << endl;
	}
}

void unipoly_domain::exact_division(
		unipoly_object a, unipoly_object b, unipoly_object &q,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "unipoly_domain::exact_division" << endl;
	}

	unipoly_object r;

	create_object_by_rank(r, 0, __FILE__, __LINE__, 0 /*verbose_level*/);

	division_with_remainder(a, b, q, r, verbose_level - 1);
	
	delete_object(r);
	
	if (f_v) {
		cout << "unipoly_domain::exact_division done" << endl;
	}
}

void unipoly_domain::division_with_remainder(
	unipoly_object a, unipoly_object b,
	unipoly_object &q, unipoly_object &r,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int *ra = (int *) a;
	int *rb = (int *) b;
	//int *A = ra + 1;
	int *B = rb + 1;

	int da, db;
	
	if (f_v) {
		cout << "unipoly_domain::division_with_remainder" << endl;
	}
	da = degree(a);
	db = degree(b);
	
	if (f_factorring) {
		cout << "unipoly_domain::division_with_remainder "
				"not good for a factorring" << endl;
		exit(1);
	}
	if (db == 0) {
		if (B[0] == 0) {
			cout << "unipoly_domain::division_with_remainder: "
					"division by zero" << endl;
			exit(1);
		}
	}
	if (db > da) {
		int *rq = (int *) q;
		FREE_int(rq);
		rq = NEW_int(2);
		int *Q = rq + 1;
		Q[0] = 0;
		rq[0] = 0;
		assign(a, r, 0 /*verbose_level*/);
		q = rq;
		goto done;
	}

	{
		int dq = da - db;
		int *rq = (int *) q;
		FREE_int(rq);
		rq = NEW_int(dq + 2);
		rq[0] = dq;

		assign(a, r, 0 /*verbose_level*/);

		int *rr = (int *) r;

		int *Q = rq + 1;
		int *R = rr + 1;
	
		int i, j, ii, jj, pivot, pivot_inv, x, c, d;

		pivot = B[db];
		pivot_inv = F->inverse(pivot);
	
		Int_vec_zero(Q, dq + 1);

		for (i = da, j = dq; i >= db; i--, j--) {
			x = R[i];
			c = F->mult(x, pivot_inv);
			Q[j] = c;
			c = F->negate(c);
			//cout << "i=" << i << " c=" << c << endl;
			for (ii = i, jj = db; jj >= 0; ii--, jj--) {
				d = B[jj];
				d = F->mult(c, d);
				R[ii] = F->add(d, R[ii]);
			}
			if (R[i] != 0) {
				cout << "unipoly::division_with_remainder: R[i] != 0" << endl;
				exit(1);
			}
			//cout << "i=" << i << endl;
			//cout << "q="; print_object((unipoly_object)
			// rq, cout); cout << endl;
			//cout << "r="; print_object(r, cout); cout << endl;
		}
		rr[0] = MAXIMUM(db - 1, 0);
		q = rq;
		//cout << "q="; print_object(q, cout); cout << endl;
		//cout << "r="; print_object(r, cout); cout << endl;
	}
done:
	if (f_v) {
		cout << "unipoly_domain::division_with_remainder done" << endl;
	}
}

void unipoly_domain::derivative(unipoly_object a, unipoly_object &b)
{
	int *ra = (int *) a;
	int *A = ra + 1;
	int d = degree(a);
	int *rb = (int *) b;
	FREE_int(rb);
	rb = NEW_int(d - 1 + 2);
	int *B = rb + 1;
	int i, ai, bi;
	
	for (i = 1; i <= d; i++) {
		ai = A[i];
		bi = i % F->p;
		bi = F->mult(ai, bi);
		B[i - 1] = bi;
	}
	rb[0] = d - 1;
	b = rb;
}

int unipoly_domain::compare_euclidean(unipoly_object m, unipoly_object n)
{
	int dm = degree(m);
	int dn = degree(n);
	
	if (dm < dn) {
		return -1;
	}
	else if (dm > dn) {
		return 1;
	}
	return 0;
}

void unipoly_domain::greatest_common_divisor(
	unipoly_object m, unipoly_object n,
	unipoly_object &g, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int c;
	
	if (f_v) {
		cout << "unipoly::greatest_common_divisor m=";
		print_object(m, cout);
		cout << " n=";
		print_object(n, cout);
		cout << endl;
	}
	if (f_v) {
		cout << "unipoly::greatest_common_divisor before compare_euclidean" << endl;
	}
	c = compare_euclidean(m, n);
	if (f_v) {
		cout << "unipoly::greatest_common_divisor compare_euclidean returns " << c << endl;
	}
	if (c < 0) {
		greatest_common_divisor(n, m, g, verbose_level);
		return;
	}
	if (c == 0 && is_zero(n)) {
		assign(m, g, 0 /*verbose_level*/);
		if (f_v) {
			cout << "unipoly::greatest_common_divisor of m=";
			print_object(m, cout);
			cout << " n=";
			print_object(n, cout);
			cout << " is ";
			print_object(g, cout);
			cout << endl;
		}
		return;
	}

	unipoly_object M, N, Q, R;
	
	create_object_by_rank(M, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(N, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(Q, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(R, 0, __FILE__, __LINE__, 0 /*verbose_level*/);

	assign(m, M, 0 /*verbose_level*/);
	assign(n, N, 0 /*verbose_level*/);

	while (TRUE) {
		if (f_vv) {
			cout << "unipoly::greatest_common_divisor M=";
			print_object(M, cout);
			cout << " N=";
			print_object(N, cout);
			cout << endl;
		}
		division_with_remainder(M, N, Q, R, verbose_level - 2);
		if (f_vv) {
			cout << "unipoly::greatest_common_divisor Q=";
			print_object(Q, cout);
			cout << " R=";
			print_object(R, cout);
			cout << endl;
		}
		if (is_zero(R)) {
			break;
		}
		
		negate(Q);

		assign(N, M, 0 /*verbose_level*/);
		assign(R, N, 0 /*verbose_level*/);
	}
	assign(N, g, 0 /*verbose_level*/);
	if (f_v) {
		cout << "unipoly::greatest_common_divisor g=";
		print_object(g, cout);
		cout << endl;
	}

	delete_object(M);
	delete_object(N);
	delete_object(Q);
	delete_object(R);

	if (f_v) {
		cout << "unipoly::greatest_common_divisor of m=";
		print_object(m, cout);
		cout << " n=";
		print_object(n, cout);
		cout << " is ";
		print_object(g, cout);
		cout << endl;
	}
	if (f_v) {
		cout << "unipoly::greatest_common_divisor done" << endl;
	}

}

void unipoly_domain::extended_gcd(
	unipoly_object m, unipoly_object n,
	unipoly_object &u, unipoly_object &v, 
	unipoly_object &g, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int c;
	
	if (f_v) {
		cout << "unipoly::extended_gcd" << endl;
	}
	if (f_vv) {
		cout << "m=";
		print_object(m, cout);
		cout << " n=";
		print_object(n, cout);
		cout << endl;
	}
	c = compare_euclidean(m, n);
	if (c < 0) {
		extended_gcd(n, m, v, u, g, verbose_level);
		if (f_v) {
			cout << "unipoly::extended_gcd done" << endl;
		}
		return;
	}
	assign(m, u, 0 /*verbose_level*/);
	assign(n, v, 0 /*verbose_level*/);
	if (c == 0 || is_zero(n)) {
		one(u);
		zero(v);
		assign(m, g, 0 /*verbose_level*/);
		if (f_v) {
			cout << "unipoly::extended_gcd done" << endl;
		}
		return;
	}

	unipoly_object M, N, Q, R;
	unipoly_object u1, u2, u3, v1, v2, v3, tmp;
	
	create_object_by_rank(M, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(N, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(Q, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(R, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(u1, 1, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(u2, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(u3, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(v1, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(v2, 1, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(v3, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(tmp, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	
	assign(m, M, 0 /*verbose_level*/);
	assign(n, N, 0 /*verbose_level*/);

	while (TRUE) {
		if (f_vv) {
			cout << "M=";
			print_object(M, cout);
			cout << " N=";
			print_object(N, cout);
			cout << endl;
		}
		division_with_remainder(M, N, Q, R, 0);
		if (f_vv) {
			cout << "Q=";
			print_object(Q, cout);
			cout << " R=";
			print_object(R, cout);
			cout << endl;
		}
		if (is_zero(R)) {
			break;
		}
		
		negate(Q);

		// u3 := u1 - Q * u2
		mult(Q, u2, tmp, verbose_level - 1);
		add(u1, tmp, u3);
		
		// v3 := v1 - Q * v2
		mult(Q, v2, tmp, verbose_level - 1);
		add(v1, tmp, v3);
		
		assign(N, M, 0 /*verbose_level*/);
		assign(R, N, 0 /*verbose_level*/);
		assign(u2, u1, 0 /*verbose_level*/);
		assign(u3, u2, 0 /*verbose_level*/);
		assign(v2, v1, 0 /*verbose_level*/);
		assign(v3, v2, 0 /*verbose_level*/);
	}
	assign(u2, u, 0 /*verbose_level*/);
	assign(v2, v, 0 /*verbose_level*/);
	assign(N, g, 0 /*verbose_level*/);
	if (f_vv) {
		cout << "g=";
		print_object(g, cout);
		cout << " u=";
		print_object(u, cout);
		cout << " v=";
		print_object(v, cout);
		cout << endl;
	}

	delete_object(M);
	delete_object(N);
	delete_object(Q);
	delete_object(R);
	delete_object(u1);
	delete_object(u2);
	delete_object(u3);
	delete_object(v1);
	delete_object(v2);
	delete_object(v3);
	delete_object(tmp);
	if (f_v) {
		cout << "unipoly::extended_gcd done" << endl;
	}
}

int unipoly_domain::is_squarefree(unipoly_object p, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	unipoly_object a, b, u, v, g;
	int d;
	
	if (f_v) {
		cout << "unipoly::is_squarefree" << endl;
	}
	create_object_by_rank(a, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(b, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(u, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(v, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(g, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	
	assign(p, a, 0 /*verbose_level*/);
	if (f_v) {
		cout << "unipoly::is_squarefree before derivative" << endl;
	}
	derivative(a, b);
	if (f_v) {
		cout << "unipoly::is_squarefree after derivative" << endl;
	}
	if (f_vv) {
		cout << "unipoly::is_squarefree derivative p' = ";
		print_object(b, cout);
		cout << endl;
	}
	if (f_v) {
		cout << "unipoly::is_squarefree before extended_gcd" << endl;
	}
	extended_gcd(a, b, u, v, g, verbose_level - 1);
	if (f_v) {
		cout << "unipoly::is_squarefree after extended_gcd" << endl;
	}
	if (f_vv) {
		cout << "unipoly::is_squarefree gcd(p, p') = ";
		print_object(g, cout);
		cout << endl;
	}
	d = degree(g);
	
	delete_object(a);
	delete_object(b);
	delete_object(u);
	delete_object(v);
	delete_object(g);
	
	if (f_v) {
		cout << "unipoly::is_squarefree done" << endl;
	}
	if (d >= 1) {
		return FALSE;
	}
	else {
		return TRUE;
	}
}

void unipoly_domain::compute_normal_basis(int d,
	int *Normal_basis, int *Frobenius,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *v, *b, *A, deg;
	int i, j;
	unipoly_object mue, lambda, GCD, Q, R, R1, R2;
	
	if (f_v) {
		cout << "unipoly_domain::compute_normal_basis "
				"d=" << d << endl;
	}
	deg = d;

	v = NEW_int(deg);
	b = NEW_int(deg);
	A = NEW_int((deg + 1) * deg);

	create_object_by_rank(mue, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(lambda, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(GCD, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(Q, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(R, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(R1, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(R2, 0, __FILE__, __LINE__, 0 /*verbose_level*/);

	i = 0;
	if (f_v) {
		cout << "unipoly_domain::compute_normal_basis before order_ideal_generator" << endl;
	}
	order_ideal_generator(d, i, mue, 
		A, Frobenius, 
		verbose_level - 10);
	if (f_v) {
		cout << "unipoly_domain::compute_normal_basis after order_ideal_generator" << endl;
	}
	
	if (f_v) {
		cout << "unipoly_domain::compute_normal_basis "
			"Ideal(e_" << i << ") = (";
		print_object(mue, cout);
		cout << ")" << endl;
	}
	Int_vec_zero(v, deg);
	v[0] = 1;

	while (degree(mue) < deg) {
		i++;
		if (f_v) {
			cout << "unipoly_domain::compute_normal_basis "
					"i = " << i << " / " << deg << endl;
		}

		if (i == deg) {
			cout << "unipoly_domain::compute_normal_basis "
					"error: i == deg" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "unipoly_domain::compute_normal_basis "
					"before order_ideal_generator" << endl;
		}
		order_ideal_generator(d, i, lambda, 
			A, Frobenius, 
			verbose_level - 10);
		if (f_v) {
			cout << "unipoly_domain::compute_normal_basis "
					"after order_ideal_generator" << endl;
		}
		if (f_vv) {
			cout << "unipoly_domain::compute_normal_basis "
					"Ideal(e_" << i << ") = ( lambda ),  "
					"where lambda = ";
			print_object(lambda, cout);
			cout << endl;
			cout << "unipoly_domain::compute_normal_basis "
					"Ideal(e_1,..,e_" << i - 1 << ") = ( mue ), "
					"where mue = ";
			print_object(mue, cout);
			cout << endl;
			cout << "v = ";
			Int_vec_print(cout, v, deg);
			cout << endl;
		}
		
		if (f_vv) {
			cout << "unipoly_domain::compute_normal_basis "
				"computing greatest_common_divisor(mue, lambda):" << endl;
		}
		greatest_common_divisor(mue, lambda, GCD, verbose_level);
	
		if (f_vv) {
			cout << "unipoly_domain::compute_normal_basis "
				"greatest_common_divisor(mue, lambda) = ";
			print_object(GCD, cout);
			cout << endl;
		}
		
		if (degree(GCD) < degree(lambda)) {
		
			// b = (0, 0, \ldots, 0, 1, 0, ..., 0) = X^i 
			Int_vec_zero(b, deg);
			b[i] = 1;
			
			exact_division(lambda, GCD, Q, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis "
						"Q = lambda / GCD = ";
				print_object(Q, cout);
				cout << endl;
			}


			take_away_all_factors_from_b(mue, Q, R, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis "
						"R = take_away_all_factors_from_b(mue, Q) = ";
				print_object(R, cout);
				cout << endl;
			}

			exact_division(mue, R, Q, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis "
						"Q = mue / R = ";
				print_object(Q, cout);
				cout << endl;
			}
			
			// Frobenius Module structure: apply Q to v (q is monic):
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis "
						"before module_structure_apply" << endl;
			}
			module_structure_apply(v, Frobenius, deg, Q, 0 /* verbose_level */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis "
						"after module_structure_apply" << endl;
			}


			// now: Orderideal(v1) = Ideal(r) 
			// v = v *(mue/R)(Frobenius) = v * Q (Frobenius)
			
			exact_division(mue, GCD, Q, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis "
						"Q = mue / GCD = ";
				print_object(Q, cout);
				cout << endl;
			}

			take_away_all_factors_from_b(lambda,
					Q, R1, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis "
						"R1 = take_away_all_factors_from_b(lambda, Q) = ";
				print_object(R, cout);
				cout << endl;
			}

			greatest_common_divisor(R,
					R1, GCD, 0 /* verbose_level */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis "
						"greatest_common_divisor(R, R1) = ";
				print_object(GCD, cout);
				cout << endl;
			}

			exact_division(R1,
					GCD, R2, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis "
						"Q = mue / GCD = ";
				print_object(Q, cout);
				cout << endl;
				}

			// now: greatest_common_divisor(R, R2) = 1
			// R * R2 = lcm(mue, lambda) 
			
			exact_division(lambda,
					R2, Q, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis "
						"Q = lambda / R2 = ";
				print_object(Q, cout);
				cout << endl;
			}

			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis "
						"before module_structure_apply" << endl;
			}
			module_structure_apply(b,
					Frobenius, deg, Q, 0 /* verbose_level */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis "
						"after module_structure_apply" << endl;
			}

			// now: Orderideal(b) = Ideal(r2) 
			// b = b * (lambda/R2)(Frobenius) = v * Q(Frobenius)
			
			for (j = 0; j < deg; j++) {
				v[j] = F->add(v[j], b[j]);
			}

			// Orderideal(v) = Ideal(R * R2), 
			// greatest_common_divisor(R, R2) = 1
			
			mult(R, R2, mue, 0 /* verbose_level */);
		} // if
		if (f_v) {
			cout << "unipoly_domain::compute_normal_basis "
				"Ideal(e_1,..,e_" << i << ") = ( mue ), where mue = ";
			print_object(mue, cout);
			cout << endl;
		}
	} // while
	
	if (f_vv) {
		cout << "unipoly_domain::compute_normal_basis "
				"generator = ";
		Int_vec_print(cout, v, deg);
		cout << endl;
	}

	if (f_v) {
		cout << "unipoly_domain::compute_normal_basis "
			"before span_cyclic_module" << endl;
	}
	F->Linear_algebra->span_cyclic_module(Normal_basis,
			v, deg, Frobenius, 0 /* verbose_level */);
	if (f_v) {
		cout << "unipoly_domain::compute_normal_basis "
			"after span_cyclic_module" << endl;
	}

	if (f_vv) {
		cout << "unipoly_domain::compute_normal_basis "
			"Normal_basis = " << endl;
		Int_matrix_print(Normal_basis, deg, deg);
	}

	FREE_int(v);
	FREE_int(b);
	FREE_int(A);

	delete_object(mue);
	delete_object(lambda);
	delete_object(GCD);
	delete_object(Q);
	delete_object(R);
	delete_object(R1);
	delete_object(R2);

	
	if (f_v) {
		cout << "unipoly_domain::compute_normal_basis done" << endl;
	}
}

void unipoly_domain::order_ideal_generator(
	int d, int idx, unipoly_object &mue,
	int *A, int *Frobenius, 
	int verbose_level)
// Lueneburg~\cite{Lueneburg87a} p. 105.
// Frobenius is a matrix of size d x d
// A is a matrix of size (d + 1) x d
{
	int f_v = (verbose_level >= 1);
	int *my_mue, mue_deg;
	
	if (f_v) {
		cout << "unipoly_domain::order_ideal_generator "
			"d=" << d << " idx = " << idx << endl;
	}

	my_mue = NEW_int(d + 1);
	

	if (f_v) {
		cout << "unipoly_domain::order_ideal_generator "
				"before F->order_ideal_generator" << endl;
	}
	F->Linear_algebra->order_ideal_generator(d, idx, my_mue, mue_deg,
		A, Frobenius, 
		verbose_level - 1);
	if (f_v) {
		cout << "unipoly_domain::order_ideal_generator "
				"after F->order_ideal_generator" << endl;
	}


	int *Mue = (int *) mue;
	FREE_int(Mue);
	Mue = NEW_int(mue_deg + 2);
	Mue[0] = mue_deg;
	int *B = Mue + 1;
	int i;
	for (i = 0; i <= mue_deg; i++) {
		B[i] = my_mue[i];
	}
	mue = (void *) Mue;

	FREE_int(my_mue);


	// testing:
	if (f_v) {
		cout << "unipoly_domain::order_ideal_generator "
			"d=" << d << " idx = " << idx << " testing" << endl;
		cout << "mue=";
		print_object(mue, cout);
		cout << endl;
	}
	int *v;

	v = NEW_int(d);
	Int_vec_zero(v, d);
	v[idx] = 1;
	
	if (f_v) {
		cout << "unipoly_domain::order_ideal_generator "
				"before module_structure_apply" << endl;
	}
	module_structure_apply(v,
			Frobenius, d, mue, 0 /*verbose_level*/);
	if (f_v) {
		cout << "unipoly_domain::order_ideal_generator "
				"after module_structure_apply" << endl;
	}
	for (i = 0; i < d; i++) {
		if (v[i]) {
			cout << "unipoly_domain::order_ideal_generator "
				"d=" << d << " idx = " << idx << " test fails, v=" << endl;
			Int_vec_print(cout, v, d);
			cout << endl;
			exit(1);
		}
	}
	FREE_int(v);
	if (f_v) {
		cout << "unipoly_domain::order_ideal_generator "
			"d=" << d << " idx = " << idx << " test passed" << endl;
	}

	if (f_v) {
		cout << "unipoly_domain::order_ideal_generator done" << endl;
	}
}

void unipoly_domain::matrix_apply(unipoly_object &p,
		int *Mtx, int n, int verbose_level)
// The matrix is applied on the left
{
	int f_v = (verbose_level >= 1);
	int *v1, *v2;
	int i, d;

	if (f_v) {
		cout << "unipoly_domain::matrix_apply" << endl;
	}
	v1 = NEW_int(n);
	v2 = NEW_int(n);
	
	d = degree(p);
	if (d >= n) {
		cout << "unipoly_domain::matrix_apply d >= n" << endl;
		exit(1);
	}
	for (i = 0; i <= d; i++) {
		v1[i] = ((int *)p)[1 + i];
	}
	for ( ; i < n; i++) {
		v1[i] = 0;
	}
	if (f_v) {
		cout << "unipoly_domain::matrix_apply v1 = ";
		Int_vec_print(cout, v1, n);
		cout << endl;
	}
	F->Linear_algebra->mult_vector_from_the_right(Mtx, v1, v2, n, n);
	if (f_v) {
		cout << "unipoly_domain::matrix_apply v2 = ";
		Int_vec_print(cout, v2, n);
		cout << endl;
	}

	delete_object(p);
	create_object_of_degree(p, n);
	for (i = 0; i < n; i++) {
		((int *)p)[1 + i] = v2[i];
	}
	
	
	FREE_int(v1);
	FREE_int(v2);
	
	if (f_v) {
		cout << "unipoly_domain::matrix_apply done" << endl;
	}
}

void unipoly_domain::substitute_matrix_in_polynomial(
	unipoly_object &p, int *Mtx_in, int *Mtx_out, int k,
	int verbose_level)
// The matrix is substituted into the polynomial
{
	int f_v = (verbose_level >= 1);
	int *M1, *M2;
	int i, j, h, c, d, *P, *coeffs;

	if (f_v) {
		cout << "unipoly_domain::substitute_matrix_in_polynomial" << endl;
	}
	M1 = NEW_int(k * k);
	M2 = NEW_int(k * k);
	P = (int *)p;
	d = P[0];
	coeffs = P + 1;
	h = d;
	c = coeffs[h];
	for (i = 0; i < k * k; i++) {
		M1[i] = F->mult(c, Mtx_in[i]);
	}
	for (h--; h >= 0; h--) {
		c = coeffs[h];
		for (i = 0; i < k; i++) {
			for (j = 0; j < k; j++) {
				if (i == j) {
					M2[i * k + j] = F->add(c, M1[i * k + j]);
				}
				else {
					M2[i * k + j] = M1[i * k + j];
				}
			}
		}
		if (h) {
			F->Linear_algebra->mult_matrix_matrix(M2, Mtx_in, M1, k, k, k,
					0 /* verbose_level */);
		}
		else {
			Int_vec_copy(M2, M1, k * k);
		}
	}
	Int_vec_copy(M1, Mtx_out, k * k);

	FREE_int(M1);
	FREE_int(M2);
	if (f_v) {
		cout << "unipoly_domain::substitute_matrix_in_polynomial done" << endl;
	}
}


int unipoly_domain::substitute_scalar_in_polynomial(
	unipoly_object &p, int scalar, int verbose_level)
// The scalar 'scalar' is substituted into the polynomial
{
	int f_v = (verbose_level >= 1);
	int m1, m2;
	int h, c, d, *P, *coeffs;

	if (f_v) {
		cout << "unipoly_domain::substitute_scalar_in_polynomial" << endl;
	}
	P = (int *)p;
	d = P[0];
	coeffs = P + 1;
	h = d;
	c = coeffs[h];
	m1 = F->mult(c, scalar);
	for (h--; h >= 0; h--) {
		c = coeffs[h];
		m2 = F->add(c, m1);
		if (h) {
			m1 = F->mult(m2, scalar);
		}
		else {
			m1 = m2;
		}
	}
	if (f_v) {
		cout << "unipoly_domain::substitute_scalar_in_polynomial done" << endl;
	}
	return m1;
}

void unipoly_domain::module_structure_apply(int *v,
		int *Mtx, int n, unipoly_object p,
		int verbose_level)
// computes the effect of Mtx substituted into p=p(x) applied to the vector v.
// Uses Horner's scheme.
// The result is put back into v.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *v1, *v2;
	int i, j, d, c;

	if (f_v) {
		cout << "unipoly_domain::module_structure_apply" << endl;
	}
	if (f_vv) {
		cout << "unipoly_domain::module_structure_apply "
				"applying p=" << endl;
		print_object(p, cout);
		cout << endl;
	}
	

	v1 = NEW_int(n);
	v2 = NEW_int(n);

	Int_vec_copy(v, v1, n);

	d = degree(p);
	int *pp;

	pp = ((int *)p) + 1;
	c = pp[d];
	if (c != 1) {
		for (j = 0; j < n; j++) {
			v1[j] = F->mult(v1[j], c);
		}
	}
#if 0
	if (!F->is_one(pp[d])) {
		cout << "unipoly_domain::module_structure_apply "
			"p is not monic, leading coefficient is "
			<< pp[d] << endl;
		exit(1);
	}
#endif
	for (i = d - 1; i >= 0; i--) {
		if (f_vv) {
			cout << "unipoly_domain::module_structure_apply "
				"i = " << i << endl;
			cout << "unipoly_domain::module_structure_apply "
				"v1 = ";
			Int_vec_print(cout, v1, n);
			cout << endl;
		}
		
		F->Linear_algebra->mult_vector_from_the_right(Mtx, v1, v2, n, n);

		if (f_vv) {
			cout << "unipoly_domain::module_structure_apply "
				"i = " << i << endl;
			cout << "unipoly_domain::module_structure_apply "
				"v2 = ";
			Int_vec_print(cout, v1, n);
			cout << endl;
		}

		c = pp[i];


		if (f_vv) {
			cout << "unipoly_domain::module_structure_apply "
				"i = " << i;
			cout << " c = " << c << endl;
		}
		for (j = 0; j < n; j++) {
			v1[j] = F->add(F->mult(v[j], c), v2[j]);
		}

		if (f_vv) {
			cout << "unipoly_domain::module_structure_apply "
				"i = " << i << endl;
			cout << "unipoly_domain::module_structure_apply "
				"v1 = ";
			Int_vec_print(cout, v1, n);
			cout << endl;
		}

	} // next i

	for (j = 0; j < n; j++) {
		v[j] = v1[j];
	}

	FREE_int(v1);
	FREE_int(v2);
	
	if (f_v) {
		cout << "unipoly_domain::module_structure_apply done" << endl;
	}
}


void unipoly_domain::take_away_all_factors_from_b(
	unipoly_object a,
	unipoly_object b, unipoly_object &a_without_b,
	int verbose_level)
// Computes the polynomial $r$ with
//\begin{enumerate}
//\item
//$r$ divides $a$
//\item
//$gcd(r,b) = 1$ and
//\item
//each irreducible polynomial dividing $a/r$ divides $b$.
//Lueneburg~\cite{Lueneburg87a}, p. 37.
//\end{enumerate}
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "unipoly_domain::take_away_all_factors_from_b" << endl;
	}
	

	unipoly_object G, A, Q;

	create_object_by_rank(G, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(A, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(Q, 0, __FILE__, __LINE__, 0 /*verbose_level*/);

	assign(a, A, verbose_level);

	greatest_common_divisor(A, b, G, verbose_level - 2);

	while (degree(G)) {

		exact_division(A, G, Q, 0 /* verbose_level - 2 */);

		assign(Q, A, 0 /*verbose_level*/);
		
		greatest_common_divisor(A, b, G, verbose_level - 2);
		
	}
	
	assign(A, a_without_b, 0 /*verbose_level*/);

	delete_object(G);
	delete_object(A);
	delete_object(Q);

	if (f_v) {
		cout << "unipoly_domain::take_away_all_factors_from_b done" << endl;
	}
}

int unipoly_domain::is_irreducible(unipoly_object a,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *B;
	int r, ret;
	int *base_cols;
	int factor_polynomial_degree;
	
	if (f_v) {
		cout << "unipoly_domain::is_irreducible" << endl;
	}
	factor_polynomial_degree = degree(a);
	
	if (f_v) {
		cout << "unipoly_domain::is_irreducible before is_squarefree" << endl;
	}
	if (!is_squarefree(a, verbose_level - 2)) {
		if (f_v) {
			cout << "unipoly_domain::is_irreducible is not squarefree" << endl;
		}
		return FALSE;
	}
	if (f_v) {
		cout << "unipoly_domain::is_irreducible is squarefree" << endl;
	}
	
	//unipoly_domain Fq(F, a);
	
	if (f_v) {
		cout << "unipoly_domain::is_irreducible before Berlekamp_matrix" << endl;
	}
	Berlekamp_matrix(B, a, verbose_level - 2);
	if (f_v) {
		cout << "unipoly_domain::is_irreducible after Berlekamp_matrix" << endl;
	}
	if (f_vv) {
		cout << "unipoly_domain::is_irreducible Berlekamp_matrix=" << endl;
		Int_matrix_print(B, factor_polynomial_degree, factor_polynomial_degree);
	}
	
	base_cols = NEW_int(factor_polynomial_degree);
	
	r = F->Linear_algebra->Gauss_int(B,
		FALSE /* f_special */,
		FALSE /* f_complete */,
		base_cols,
		FALSE /* f_P */, NULL /* P */, 
		factor_polynomial_degree, factor_polynomial_degree,
		0 /* Pn */, 0 /* verbose_level */);

	if (f_v) {
		cout << "unipoly_domain::is_irreducible Berlekamp_matrix has rank " << r << endl;
	}
	
	FREE_int(B);
	FREE_int(base_cols);

	if (r == factor_polynomial_degree - 1) {
		ret = TRUE;
	}
	else {
		ret = FALSE;
	}
	if (f_v) {
		cout << "unipoly_domain::is_irreducible done" << endl;
	}
	return ret;
}

void unipoly_domain::singer_candidate(unipoly_object &m,
		int p, int d, int b, int a)
{
	create_object_of_degree(m, d);
	int *M = ((int *)m) + 1;
	M[d] = 1;
	M[d - 1] = 1;
	M[1] = b;
	M[0] = a;
}

int unipoly_domain::is_primitive(unipoly_object &m, 
	longinteger_object &qm1, 
	int nb_primes, longinteger_object *primes, 
	int verbose_level)
//Returns TRUE iff the polynomial $x$ has order $qm1$ 
//modulo the polynomial m (over GF(p)). 
//The prime factorization of $qm1$
// must be given in primes (only the primes).
//A polynomial $a$ has order $s$ mod $m$ ($q = this$) iff 
//$a^m =1 mod q$ and $a^{s/p_i} \not= 1 mod m$ for all $p_i \mid s.$ 
//In this case, we have $a=x$ and we assume that $a^qm1 = 1 mod q.$
{
	int f_v = (verbose_level >= 1);
	longinteger_object qm1_over_p, r, u;
	longinteger_domain D;
	unipoly_object M;
	int i;
	
	if (f_v) {
		cout << "unipoly_domain::is_primitive" << endl;
	}
	create_object_of_degree(M, ((int*)m)[0]);
	assign(m, M, 0 /*verbose_level*/);
	unipoly_domain Fq(F, M, verbose_level - 1);
	
	if (f_v) {
		cout << "unipoly_domain::is_primitive "
			"q=" << F->q << endl;
		cout << "m=";
		print_object(m, cout);
		cout << endl;
		cout << "M=";
		print_object(M, cout);
		cout << endl;
		cout << "primes:" << endl;
		for (i = 0; i < nb_primes; i++) {
			cout << i << " : " << primes[i] << endl;
		}
	}
	
	for (i = 0; i < nb_primes; i++) {

		if (f_v) {
			cout << "unipoly_domain::is_primitive testing prime " << i << " / "
					<< nb_primes << " which is " << primes[i] << endl;
		}


		D.integral_division(qm1, primes[i], qm1_over_p, r, 0);
		if (f_v) {
			cout << "qm1 / " << primes[i] << " = "
					<< qm1_over_p << " remainder " << r << endl;
		}
		if (!r.is_zero()) {
			cout << "unipoly_domain::is_primitive "
					"the prime does not divide!" << endl;
			cout << "qm1=" << qm1 << endl;
			cout << "primes[i]=" << primes[i] << endl;
			cout << "qm1_over_p=" << qm1_over_p << endl;
			cout << "r=" << r << endl;
			exit(1); 
		}
		
		unipoly_object a;
		
		Fq.create_object_by_rank(a, F->q, __FILE__, __LINE__, 0 /*verbose_level*/); // the polynomial X
		Fq.power_longinteger(a, qm1_over_p, 0 /*verbose_level - 1*/);
		
		if (f_v) {
			cout << "unipoly_domain::is_primitive X^" << qm1_over_p << " mod ";
			print_object(m, cout);
			cout << " = ";
			print_object(a, cout);
			cout << endl;
		}
		
		if (Fq.is_one(a)) {
			if (f_v) {
				cout << "unipoly_domain::is_primitive is one, hence m is not primitive" << endl;
			}
			Fq.delete_object(a);
			return FALSE;
		}
		
		Fq.delete_object(a);
		
	}
	if (f_v) {
		cout << "unipoly_domain::is_primitive ";
		cout << "m=";
		print_object(m, cout);
		cout << " is primitive" << endl;

#if 0
		unipoly_object a;
		for (i = 0; i <= qm1.as_int(); i++) {
			Fq.create_object_by_rank(a, F->q); // the polynomial X
			u.create(i);
			Fq.power_longinteger(a, u);
			cout << "X^" << u << " = ";
			print_object(a, cout);
			cout << endl;
		}
		Fq.delete_object(a);
#endif
	}
	
	if (f_v) {
		cout << "unipoly_domain::is_primitive done" << endl;
	}
	return TRUE;
}

void unipoly_domain::get_a_primitive_polynomial(
	unipoly_object &m,
	int f, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a, b, p, nb_primes, i;
	unipoly_object x;
	longinteger_object q, m1, qm1;
	longinteger_object low, high, current, one, tmp;
	longinteger_domain D;
	longinteger_object *primes;
	int *exponents;
	
	if (f_v) {
		cout << "unipoly::get_a_primitive_polynomial" << endl;
		cout << "Searching for a primitive polynomial of degree " << f <<
			" over GF(" << F->q << ")" << endl;
	}
	p = F->q;
	q.create(p, __FILE__, __LINE__);
	m1.create(-1, __FILE__, __LINE__);
	D.power_int(q, f);
	D.add(q, m1, qm1);
	if (f_vv) {
		cout << "unipoly_domain::get_a_primitive_polynomial factoring " << qm1 << endl;
	}
	D.factor_into_longintegers(qm1, nb_primes,
			primes, exponents, verbose_level - 2);
	if (f_vv) {
		cout << "unipoly_domain::get_a_primitive_polynomial after factoring "
				<< qm1 << " nb_primes=" << nb_primes << endl;
		cout << "primes:" << endl;
		for (i = 0; i < nb_primes; i++) {
			cout << i << " : " << primes[i] << endl;
		}
	}
	if (f_vv) {
		cout << "unipoly_domain::get_a_primitive_polynomial "
				"before F->primitive_root" << endl;
	}
	a = F->primitive_root();
	if (f_vv) {
		cout << "unipoly_domain::get_a_primitive_polynomial "
				"a primitive root is " << a << endl;
	}
	
	for (b = 0; b < p; b++) {
		singer_candidate(x, p, f, b, a);
		if (f_v) {
			cout << "singer candidate ";
			print_object(x, cout);
			cout << endl;
		}
		if (is_irreducible(x, verbose_level - 1)) {
			if (f_v) {
				cout << "IS irreducible" << endl;
			}
			if (is_primitive(x, qm1, nb_primes, primes, verbose_level - 1)) {
				if (f_v) {
					cout << "OK, we found an irreducible "
							"and primitive polynomial ";
					print_object(x, cout);
					cout << endl;
				}
				assign(x, m, 0 /*verbose_level*/);
				if (f_v) {
					cout << "before delete_object(x)" << endl;
				}
				delete_object(x);
				if (f_v) {
					cout << "before FREE_OBJECTS(primes)" << endl;
				}
				FREE_OBJECTS(primes);
				if (f_v) {
					cout << "before FREE_int(exponents)" << endl;
				}
				FREE_int(exponents);
				return;
			}
			else {
				if (f_v) {
					cout << "is not primitive" << endl;
				}
			}
		}
		else {
			if (f_v) {
				cout << "is not irreducible" << endl;
			}
		}
		delete_object(x);
	}

	low.create(F->q, __FILE__, __LINE__);
	one.create(1, __FILE__, __LINE__);
	D.power_int(low, f);

	D.mult(low, low, high); // only monic polynomials 

	low.assign_to(current);
	
	while (TRUE) {
		
		create_object_by_rank_longinteger(x, current, __FILE__, __LINE__, 0 /*verbose_level*/);
		
		if (f_v) {
			cout << "candidate " << current << " : ";
			print_object(x, cout);
			cout << endl;
		}
		if (is_irreducible(x, verbose_level - 1)) {
			if (f_v) {
				cout << "is irreducible" << endl;
			}
			if (is_primitive(x, qm1, nb_primes,
					primes, verbose_level - 1)) {
				if (f_v) {
					cout << "is irreducible and primitive" << endl;
				}
				if (f_v) {
					cout << "unipoly::get_a_primitive_polynomial ";
					print_object(x, cout);
					cout << endl;
				}
				assign(x, m, 0 /*verbose_level*/);
				if (f_v) {
					cout << "before delete_object(x)" << endl;
				}
				delete_object(x);
				if (f_v) {
					cout << "before FREE_OBJECTS(primes)" << endl;
				}
				FREE_OBJECTS(primes);
				if (f_v) {
					cout << "before FREE_int(exponents)" << endl;
				}
				FREE_int(exponents);
				return;
			}
			else {
				if (f_v) {
					cout << "is not primitive" << endl;
				}
			}

		}
		else {
			if (f_v) {
				cout << "is not irreducible" << endl;
			}
		}
		
		delete_object(x);
		
		D.add(current, one, tmp);
		tmp.assign_to(current);
		
		if (D.compare(current, high) == 0) {
			cout << "unipoly::get_an_irreducible_polynomial "
					"did not find an irreducible polynomial" << endl;
			exit(1); 
		}
	}
}

void unipoly_domain::get_an_irreducible_polynomial(
	unipoly_object &m,
	int f, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	unipoly_object x;
	longinteger_object low, high, current, one, tmp;
	longinteger_domain D;
	
	if (f_v) {
		cout << "unipoly::get_an_irreducible_polynomial" << endl;
		cout << "Searching for an irreducible polynomial "
			"of degree " << f <<
			" over GF(" << F->q << ")" << endl;
	}
	low.create(F->q, __FILE__, __LINE__);
	one.create(1, __FILE__, __LINE__);
	D.power_int(low, f);

	D.mult(low, low, high); // only monic polynomials 

	low.assign_to(current);
	
	while (TRUE) {
		
		create_object_by_rank_longinteger(x,
				current, __FILE__, __LINE__, 0 /*verbose_level - 2*/);
		
		if (f_vv) {
			cout << "unipoly::get_an_irreducible_polynomial "
				"candidate " << current << " : ";
			print_object(x, cout);
			cout << endl;
		}
		if (is_irreducible(x, verbose_level - 3)) {
			if (f_vv) {
				cout << "unipoly::get_an_irreducible_polynomial "
					"candidate " << current << " : ";
				print_object(x, cout);
				cout << " is irreducible" << endl;
			}
			assign(x, m, 0 /*verbose_level*/);
			delete_object(x);
			return;
		}
		
		delete_object(x);
		
		D.add(current, one, tmp);
		tmp.assign_to(current);
		
		if (D.compare(current, high) == 0) {
			cout << "unipoly::get_an_irreducible_polynomial "
				"did not find an irreducible polynomial" << endl;
			exit(1); 
		}
	}
}

void unipoly_domain::power_int(unipoly_object &a,
		long int n, int verbose_level)
// does not mod out by factor polynomial
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	unipoly_object b, c, d;
	
	if (f_v) {
		cout << "unipoly_domain::power_int, verbose_level=" << verbose_level << endl;
	}
	if (f_vv) {
		cout << "unipoly_domain::power_int computing a=";
		print_object(a, cout);
		cout << " to the power " << n << endl;
	}
	//cout << "power_int a=";
	//print_object(a, cout);
	//cout << " n=" << n << endl;
	create_object_by_rank(b, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(c, 1, __FILE__, __LINE__, 0 /*verbose_level*/); // c = 1
	create_object_by_rank(d, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	assign(a, b, verbose_level);
	while (n) {
		if (f_vv) {
			cout << "unipoly_domain::power_int n=" << n;
			cout << " b=";
			print_object(b, cout);
			cout << " c=";
			print_object(c, cout);
			cout << endl;
		}
		
		if (n % 2) {
			if (f_vv) {
				cout << "unipoly_domain::power_int n is odd" << endl;
				cout << "unipoly_domain::power_int before mult(b,c,d)" << endl;
			}
			mult(b, c, d, verbose_level - 1);
			if (f_vv) {
				cout << "unipoly_domain::power_int before assign(d,c)" << endl;
			}
			if (f_vv) {
				cout << "b*c=d";
				print_object(d, cout);
				cout << endl;
			}
			assign(d, c, 0 /*verbose_level*/);
		}
		else {
			if (f_vv) {
				cout << "unipoly_domain::power_int n is even" << endl;
			}
		}
		if (f_vv) {
			cout << "unipoly_domain::power_int before mult(b,b,d)" << endl;
		}
		mult(b, b, d, verbose_level - 1);
		if (f_vv) {
			cout << "unipoly_domain::power_int b*b=d";
			print_object(d, cout);
			cout << endl;
		}
		if (f_vv) {
			cout << "unipoly_domain::power_int before assign(d,b)" << endl;
		}
		assign(d, b, 0 /*verbose_level*/);
		n >>= 1;
	}
	if (f_vv) {
		cout << "unipoly_domain::power_int before assign(c,a)" << endl;
	}
	assign(c, a, 0 /*verbose_level*/);
	if (f_vv) {
		cout << "unipoly_domain::power_int before delete_object(b)" << endl;
	}
	delete_object(b);
	delete_object(c);
	delete_object(d);
	if (f_v) {
		cout << "unipoly_domain::power_int done" << endl;
	}
}

void unipoly_domain::power_longinteger(
	unipoly_object &a, longinteger_object &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object m, q;
	longinteger_domain D;
	unipoly_object b, c, d;
	int r;
	
	if (f_v) {
		cout << "unipoly_domain::power_longinteger" << endl;
	}
	//cout << "power_int() a=";
	//print_object(a, cout);
	//cout << " n=" << n << endl;
	create_object_by_rank(b, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(c, 1, __FILE__, __LINE__, 0 /*verbose_level*/); // c = 1
	create_object_by_rank(d, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	n.assign_to(m);
	assign(a, b, 0 /*verbose_level*/);
	while (!m.is_zero()) {
		D.integral_division_by_int(m, 2, q, r);
		if (r) {
			mult(b, c, d, verbose_level - 1);
			assign(d, c, verbose_level);
		}
		mult(b, b, d, verbose_level - 1);
		assign(d, b, 0 /*verbose_level*/);
		q.assign_to(m);
	}
	assign(c, a, 0 /*verbose_level*/);
	delete_object(b);
	delete_object(c);
	delete_object(d);
	if (f_v) {
		cout << "unipoly_domain::power_longinteger done" << endl;
	}
}


void unipoly_domain::power_mod(unipoly_object &a, unipoly_object &m,
		long int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = FALSE; //(verbose_level >= 2);
	unipoly_object b, c, d;

	if (f_v) {
		cout << "unipoly_domain::power_mod, verbose_level=" << verbose_level << endl;
	}
	if (f_vv) {
		cout << "unipoly_domain::power_mod computing a=";
		print_object(a, cout);
		cout << " to the power " << n << " modulo ";
		print_object(m, cout);
		cout << endl;
	}
	//cout << "power_mod a=";
	//print_object(a, cout);
	//cout << " n=" << n << endl;
	create_object_by_rank(b, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(c, 1, __FILE__, __LINE__, 0 /*verbose_level*/); // c = 1
	create_object_by_rank(d, 0, __FILE__, __LINE__, 0 /*verbose_level*/);

	assign(a, b, verbose_level);

	while (n) {
		if (f_vv) {
			cout << "unipoly_domain::power_mod n=" << n << endl;
		}
		if (f_vvv) {
			cout << " b=";
			print_object(b, cout);
			cout << " c=";
			print_object(c, cout);
			cout << endl;
		}

		if (n % 2) {
			if (f_vv) {
				cout << "unipoly_domain::power_mod n is odd" << endl;
				cout << "unipoly_domain::power_mod before mult_mod(b,c,d,m)" << endl;
			}
			mult_mod(b, c, d, m, 0 /*verbose_level - 1*/);

			if (f_vvv) {
				cout << "unipoly_domain::power_mod before assign(d,c)" << endl;
			}
			if (f_vvv) {
				cout << "b * c = d";
				print_object(d, cout);
				cout << endl;
			}
			assign(d, c, 0 /*verbose_level*/);
		}
		else {
			if (f_vv) {
				cout << "unipoly_domain::power_mod n is even" << endl;
			}
		}
		if (f_vv) {
			cout << "unipoly_domain::power_mod before mult(b,b,d)" << endl;
		}
		mult_mod(b, b, d, m, 0 /*verbose_level - 1*/);
		if (f_vvv) {
			cout << "unipoly_domain::power_mod b * b = d";
			print_object(d, cout);
			cout << endl;
		}
		if (f_vvv) {
			cout << "unipoly_domain::power_mod before assign(d,b)" << endl;
		}
		assign(d, b, 0 /*verbose_level*/);
		n >>= 1;
	}
	if (f_vv) {
		cout << "unipoly_domain::power_mod before assign(c,a)" << endl;
	}
	assign(c, a, 0 /*verbose_level*/);
	if (f_vv) {
		cout << "unipoly_domain::power_mod before delete_object(b)" << endl;
	}
	delete_object(b);
	delete_object(c);
	delete_object(d);
	if (f_v) {
		cout << "unipoly_domain::power_mod done" << endl;
	}
}



void unipoly_domain::power_coefficients(
	unipoly_object &a, int n)
{
	int *ra = (int *) a;
	int m = ra[0];
	int *A = ra + 1;
	int i;
	
	for (i = 0; i <= m; i++) {
		A[i] = F->power(A[i], n);
	}
}

void unipoly_domain::minimum_polynomial(
	unipoly_object &a,
	int alpha, int p, int verbose_level)
// computes the minimum polynomial of alpha with respect to the ground 
// field of order p (BTW: p might also be a prime power)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int q, m_alpha, u0, cnt;
	unipoly_object u, v, w;
	
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial" << endl;
	}
	if (f_factorring) {
		cout << "unipoly_domain::minimum_polynomial "
				"does not work for factorring" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial "
				"alpha = " << alpha << endl;
	}
	q = F->q;
	m_alpha = F->negate(alpha);
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial "
				"m_alpha = " << m_alpha << endl;
	}
	create_object_by_rank(u, q + m_alpha, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(v, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(w, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	if (f_vv) {
		cout << "unipoly_domain::minimum_polynomial X - alpha = ";
		print_object(u, cout);
		cout << endl;
	}
	assign(u, v, 0 /*verbose_level*/);
	
	cnt = 0;
	while (TRUE) {
		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial "
					"Iteration " << cnt;
			cout << "u=";
			print_object(u, cout);
			cout << endl;
			cout << "v=";
			print_object(v, cout);
			cout << endl;
		}
		power_coefficients(v, p);
		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial conjugate = ";
			print_object(v, cout);
			cout << endl;
		}
		u0 = ((int *)v)[1];
		if (u0 == m_alpha) {
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial finished" << endl;
			}
			break;
		}
		mult(u, v, w, verbose_level - 1);
		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial product = ";
			print_object(w, cout);
			cout << endl;
		}
		assign(w, u, 0 /*verbose_level*/);
		cnt++;
	}

	if (f_vv) {
		cout << "unipoly_domain::minimum_polynomial "
				"Iteration " << cnt << " done";
		cout << "u=";
		print_object(u, cout);
		cout << endl;
	}
	assign(u, a, 0 /*verbose_level*/);
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial "
				"the minimum polynomial of " << alpha
				<< " over GF(" << p << ") is ";
		print_object(a, cout);
		cout << endl;
	}
	delete_object(u);
	delete_object(v);
	delete_object(w);
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial done" << endl;
	}
}

int unipoly_domain::minimum_polynomial_factorring(
		int alpha, int p, int verbose_level)
// compute the minimum polynomial of alpha over F_p.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk, r;
	
	if (!f_factorring) {
		cout << "unipoly_domain::minimum_polynomial_factorring "
				"must be a factorring" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "factor_degree = " << factor_degree << endl;
	}
	unipoly_object *coeffs =
			NEW_OBJECTS(unipoly_object, factor_degree + 1);
	unipoly_object b, c, d;
	int a0, ai, i, j;

	// create the polynomial Y - alpha:
	for (i = 0; i <= factor_degree; i++) {
		if (i == 1) {
			create_object_by_rank(coeffs[i], 1, __FILE__, __LINE__, 0 /*verbose_level*/);
		}
		else {
			create_object_by_rank(coeffs[i], 0, __FILE__, __LINE__, 0 /*verbose_level*/);
		}
	}
	create_object_by_rank(b, alpha, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(c, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(d, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_factorring "
				"minimum polynomial of ";
		print_object(b, cout);
		cout << " = " << alpha << endl;
	}
	negate(b);
	if (f_vv) {
		cout << "-b = ";
		print_object(b, cout);
		cout << endl;
	}
	a0 = rank(b);
	if (f_vv) {
		cout << "a0 = " << a0 << endl;
	}
	assign(b, coeffs[0], 0 /*verbose_level*/);
	
	i = 1;
	while (TRUE) {
		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_factorring "
					"i=" << i << " b=";
			print_object(b, cout);
			cout << " the polynomial is ";
			for (j = i; j >= 0; j--) {
				print_object(coeffs[j], cout);
				if (j > 0) {
					cout << " Y^" << j << " + ";
				}
			}
			cout << endl;
		}
		
		power_int(b, p, 0 /* verbose_level */);
		
		ai = rank(b);
		if (ai == a0) {
			break;
		}
		
		if (i == factor_degree) {
			cout << "unipoly_domain::minimum_polynomial_factorring "
					"i == factor_degree && ai != a0" << endl;
			exit(1);
		}
		
		unipoly_object tmp = coeffs[i + 1];

		for (j = i; j >= 0; j--) {
			coeffs[j + 1] = coeffs[j];
		}
		coeffs[0] = tmp;

		for (j = 1; j <= i + 1; j++) {
			mult(coeffs[j], b, c, verbose_level - 1);
			add(c, coeffs[j - 1], d);
			assign(d, coeffs[j - 1], verbose_level);

			// coeffs[j - 1] = coeffs[j] * b + coeffs[j - 1]
		}
		
		i++;
	}
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_factorring minimum polynomial is: ";
		for (j = i; j >= 0; j--) {
			print_object(coeffs[j], cout);
			if (j > 0) {
				cout << "Y^" << j << " + ";
			}
		}
		cout << endl;
	}
	rk = 0;
	for (j = i; j >= 0; j--) {
		r = rank(coeffs[j]);
		rk *= p;
		rk += r;
	}
	if (f_v) {
		cout << "the rank of this polynomial over "
				"GF(" << p << ") is " << rk << endl;
	}
	for (j = 0; j <= factor_degree; j++) {
		delete_object(coeffs[j]);
	}
	delete_object(b);
	delete_object(c);
	delete_object(d);
	return rk;
}

void unipoly_domain::minimum_polynomial_factorring_longinteger(
	longinteger_object &alpha, longinteger_object &rk_minpoly, 
	int p, int verbose_level)
// compute the minimum polynomial of alpha over F_p.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (!f_factorring) {
		cout << "unipoly_domain::minimum_polynomial_factorring_longinteger "
				"must be a factorring" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "factor_degree = " << factor_degree << endl;
	}


	// Let M(Y) be the minimum polynomial of alpha over F_p.

	// M(Y) has the form

	// M(Y) = (Y - alpha) * (Y - alpha^\varphi) * (Y - alpha^{\varphi^2}) * ...

	// Using b = -alpha, we write

	// M(Y) = (Y + b) * (Y + b^\varphi)*(Y + b^{\varphi^2}) * ...

	// we will maintain a vector of polynomials to represent M(Y)
	// In the end, all coefficients will be constant polynomials.
	// The constants are the coefficients of the minimum polynomial.

	unipoly_object *coeffs = NEW_OBJECTS(unipoly_object, factor_degree + 1);

	unipoly_object b, c, d;
	int i, j;




	// create the polynomial Y - alpha:


	// we first create (0,1,0,0,...) as vector of polynomials
	for (i = 0; i <= factor_degree; i++) {
		if (i == 1) {
			create_object_by_rank(coeffs[i], 1, __FILE__, __LINE__, 0 /*verbose_level*/);
		}
		else {
			create_object_by_rank(coeffs[i], 0, __FILE__, __LINE__, 0 /*verbose_level*/);
		}
	}

	// b = alpha (constant polynomial)
	create_object_by_rank_longinteger(b, alpha, __FILE__, __LINE__, 0 /*verbose_level*/);

	// c and d are needed later:
	create_object_by_rank(c, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(d, 0, __FILE__, __LINE__, 0 /*verbose_level*/);

	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_factorring_longinteger "
				"minimum polynomial of ";
		print_object(b, cout);
		cout << " = " << alpha << endl;
	}
	negate(b);
	if (f_vv) {
		cout << "-b = ";
		print_object(b, cout);
		cout << endl;
	}
	// now b = -alpha
	
	longinteger_object a0, ai;
	longinteger_domain D;
	
	// we remember the rank of b (=-alpha) so that we can
	// easily detect when the orbit under the Frobenius closes.
	// Let a0 be the rank of b = -alpha

	rank_longinteger(b, a0);
		// here we use rank_longinteger and not rank

	if (f_vv) {
		cout << "a0 = " << a0 << endl;
	}

	assign(b, coeffs[0], 0 /*verbose_level*/);

	// now we have the vector of polynomials (-alpha,1,0,0,...)
	// which represents Y + b = Y - alpha.
	// Y - alpha is the first factor in the minimum polynomial
	// Next, we apply the Frobenius automorphism to b and multiply up.
	// Thus, we loop over the conjugates of b.
	// We do this until the orbit of b closes.
	// Then we have M(Y) = (Y + b) * (Y + b^\varphi) * (Y + b^{\varphi^2}) * ...
	
	i = 1;
	while (TRUE) {

		if (f_vv) {
			cout << "i=" << i << " b=";
			print_object(b, cout);
			cout << " the polynomial is ";
			for (j = i; j >= 0; j--) {
				print_object(coeffs[j], cout);
				if (j > 0) {
					cout << " Y^" << j << " + ";
				}
			}
			cout << endl;
		}
		
		// apply the Frobenius automorphism to b.
		// this gives another conjugate of -alpha

		power_int(b, p, 0 /* verbose_level */);

		// test if the orbit has closed:
		
		rank_longinteger(b, ai);

		if (D.compare(ai, a0) == 0) {

			// yes, the orbit has closed. Now b = -alpha

			break;
		}
		
		if (i == factor_degree) {
			cout << "unipoly_domain::minimum_polynomial_factorring_longinteger "
					"i == factor_degree && ai != a0" << endl;
			exit(1);
		}
		
		// every coefficient is a polynomial
		// move every coefficient up by one,
		// move the next unused coefficient to the constant term at the bottom.
		// We are only switching pointers.
		// There is no copying going on here

		unipoly_object tmp = coeffs[i + 1];
		for (j = i; j >= 0; j--) {
			coeffs[j + 1] = coeffs[j];
		}
		coeffs[0] = tmp;

		for (j = 1; j <= i + 1; j++) {
			mult(coeffs[j], b, c, verbose_level - 1);
			add(c, coeffs[j - 1], d);
			assign(d, coeffs[j - 1], 0 /*verbose_level*/);
		}
		
		i++;
	}
	if (f_v) {
		cout << "is: ";
		for (j = i; j >= 0; j--) {
			print_object(coeffs[j], cout);
			if (j > 0) {
				cout << "Y^" << j << " + ";
			}
		}
		cout << endl;
	}
	
	longinteger_object rk, r, p_object, rk1;
	
	rk.create(0, __FILE__, __LINE__);
	p_object.create(p, __FILE__, __LINE__);
	for (j = i; j >= 0; j--) {
		rank_longinteger(coeffs[j], r);
		D.mult(rk, p_object, rk1);
		D.add(rk1, r, rk);

	}
	if (f_v) {
		cout << "the rank of this polynomial over "
				"GF(" << p << ") is " << rk << endl;
	}
	for (j = 0; j <= factor_degree; j++) {
		delete_object(coeffs[j]);
	}
	delete_object(b);
	delete_object(c);
	delete_object(d);
	rk.assign_to(rk_minpoly);
}

void unipoly_domain::print_vector_of_polynomials(
		unipoly_object *sigma, int deg)
{
	int i;

	for (i = 0; i < deg; i++) {
		cout << i << ": ";
		print_object(sigma[i], cout);
		cout << endl;
	}
}

void unipoly_domain::minimum_polynomial_extension_field(
	unipoly_object &g, unipoly_object m,
	unipoly_object &minpol, int d, int *Frobenius,
	int verbose_level)
// Lueneburg~\cite{Lueneburg87a}, p. 112.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int deg, i, j, k;
	unipoly_object mm, h, h2, *sigma;

	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_extension_field of ";
		print_object(g, cout);
		cout << endl;
	}
	deg = d;
	sigma = NEW_OBJECTS(unipoly_object, deg + 2);
	for (i = 0; i < deg + 2; i++) {
		if (i == 0) {
			create_object_by_rank(sigma[i], 1, __FILE__, __LINE__, 0 /*verbose_level*/);
		}
		else {
			create_object_by_rank(sigma[i], 0, __FILE__, __LINE__, 0 /*verbose_level*/);
		}
	}
	create_object_by_rank(h, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(h2, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(mm, 0, __FILE__, __LINE__, 0 /*verbose_level*/);

	assign(m, mm, 0 /*verbose_level*/);
	negate(mm);
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_extension_field "
				"we are working modulo - (";
		print_object(mm, cout);
		cout << ")" << endl;
	}

	assign(g, sigma[1], 0 /*verbose_level*/);

	i = 1;
	while (TRUE) {
		i++;

		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_extension_field "
					"i = " << i << " : g = ";
			print_object(g, cout);
			cout << endl;
			cout << "sigma=" << endl;
			print_vector_of_polynomials(sigma, deg);
		}

		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_extension_field "
					"i = " << i << " before matrix_apply" << endl;
		}
		matrix_apply(g, Frobenius, deg, verbose_level - 2);
		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_extension_field "
					"i = " << i << " after matrix_apply" << endl;
		}

		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_extension_field "
					"i = " << i << " : g=";
			print_object(g, cout);
			cout << endl;
		}
		
		
		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_extension_field "
					"i = " << i
					<< " before mult_mod" << endl;
		}
		mult_mod_negated(g, sigma[i - 1], sigma[i],
				degree(mm), ((int *)mm) + 1,
				verbose_level - 2);
		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_extension_field "
					"i = " << i
					<< " after mult_mod" << endl;
			cout << "sigma=" << endl;
			print_vector_of_polynomials(sigma, deg);
		}

		for (j = i - 1; j >= 1; j--) {
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field "
						"i = " << i << " j = "
						<< j << endl;
			}
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field "
						"i = " << i << " j = "
						<< j << " before mult_mod" << endl;
			}
			mult_mod_negated(g, sigma[j - 1], h, degree(mm), ((int *)mm) + 1, 0);
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field "
						"i = " << i << " j = "
						<< j << " after mult_mod" << endl;
				cout << "sigma=" << endl;
				print_vector_of_polynomials(sigma, deg);
			}
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field "
						"i = " << i << " j = "
						<< j << " before add" << endl;
			}
			add(sigma[j], h, h2);
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field "
						"i = " << i << " j = "
						<< j << " after add" << endl;
			}
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field "
						"i = " << i << " j = "
						<< j << " before assign" << endl;
				cout << "sigma=" << endl;
				print_vector_of_polynomials(sigma, deg);
			}
			assign(h2, sigma[j], 0 /*verbose_level*/);
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field "
						"i = " << i << " j = "
						<< j << " after assign" << endl;
			}

			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field "
						"i = " << i << " j = "
						<< j << " iteration finished" << endl;
			}
		}
		for (k = i; k >= 0; k--) {
			if (degree(sigma[k]) > 0) {
				break;
			}
		}
		if (k == -1) {
			break;
		}
	}
	delete_object(minpol);
	create_object_of_degree(minpol, i);
	for (j = i; j >= 0; j--) {
		((int *) minpol)[1 + j] = ((int *)sigma[i - j])[1 + 0];
	}
	for (j = 0; j <= i; j += 2) {
		((int *) minpol)[1 + j] = F->negate(
				((int *) minpol)[1 + j]);
	}
	make_monic(minpol);
	if (f_vv) {
		cout << "unipoly_domain::minimum_polynomial_extension_field "
				"after make_monic";
		cout << "minpol=";
		print_object(minpol, cout);
		cout << endl;
	}

	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_extension_field "
				"minpol is ";
		print_object(minpol, cout);
		cout << endl;
	}

	delete_object(h);
	delete_object(h2);
	delete_object(mm);
	for (i = 0; i < deg + 2; i++) {
		delete_object(sigma[i]);
	}
	FREE_OBJECTS(sigma);
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_extension_field done" << endl;
	}
}

void unipoly_domain::characteristic_polynomial(
		int *Mtx, int k, unipoly_object &char_poly,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	unipoly_object *M;
	int i, j, a, m_one;

	if (f_v) {
		cout << "unipoly_domain::characteristic_polynomial" << endl;
	}
	if (f_vv) {
		cout << "unipoly_domain::characteristic_polynomial M=" << endl;
		Int_matrix_print(Mtx, k, k);
	}
	m_one = F->negate(1);
	M = NEW_OBJECTS(unipoly_object, k * k);
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			a = Mtx[i * k + j];
			if (i == j) {
				create_object_of_degree(M[i * k + j], 1);
				((int *)M[i * k + j])[1 + 0] = a;
				((int *)M[i * k + j])[1 + 1] = m_one;
			}
			else {
				create_object_of_degree(M[i * k + j], 0);
				((int *)M[i * k + j])[1 + 0] = a;
			}
		}
	}

	
	if (f_vv) {
		cout << "unipoly_domain::characteristic_polynomial M - X Id=" << endl;
		print_matrix(M, k);
	}

	if (f_vv) {
		cout << "unipoly_domain::characteristic_polynomial before determinant" << endl;
	}
	determinant(M, k, char_poly, verbose_level);
	if (f_vv) {
		cout << "unipoly_domain::characteristic_polynomial after determinant" << endl;
	}

	if (f_vv) {
		cout << "unipoly_domain::characteristic_polynomial before delete_object(M[i * k + j]);" << endl;
	}
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			if (f_vv) {
				cout << "unipoly_domain::characteristic_polynomial i=" << i << " j=" << j << endl;
			}
			delete_object(M[i * k + j]);
		}
	}
	if (f_vv) {
		cout << "unipoly_domain::characteristic_polynomial before FREE_OBJECTS(M);" << endl;
	}
	FREE_OBJECTS(M);
	if (f_v) {
		cout << "unipoly_domain::characteristic_polynomial done" << endl;
	}
}

void unipoly_domain::print_matrix(unipoly_object *M, int k)
// M is a matrix with polynomial entries
{
	int i, j;

	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			print_object(M[i * k + j], cout);
			if (j < k - 1) {
				cout << "; ";
			}
		}
		cout << endl;
	}
}

void unipoly_domain::determinant(
		unipoly_object *M,
		int k, unipoly_object &p,
		int verbose_level)
// M is a matrix with polynomial entries
{
	int f_v = (verbose_level >= 1);
	int i, j;
	unipoly_object p1, p2, p3;
	
	if (f_v) {
		cout << "unipoly_domain::determinant k=" << k << endl;
	}
	if (k == 0) {
		delete_object(p);
		create_object_by_rank(p, 1, __FILE__, __LINE__, 0 /*verbose_level*/);
		if (f_v) {
			cout << "unipoly_domain::determinant done" << endl;
		}
		return;
	}
	delete_object(p);
	create_object_by_rank(p, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(p1, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(p2, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	create_object_by_rank(p3, 0, __FILE__, __LINE__, 0 /*verbose_level*/);
	
	for (i = 0; i < k; i++) {
		unipoly_object *N;
		
		deletion_matrix(M, k,
				i /* delete_row */,
				0 /* delete_column */,
				N,
				0 /*verbose_level - 2*/);

		determinant(N, k - 1, p1, verbose_level - 2);
		if (f_v) {
			cout << "unipoly_domain::determinant "
					"deletion of row " << i << " leads to determinant ";
			print_object(p1, cout);
			cout << endl;
		}

		mult(p1, M[i * k + 0], p2, verbose_level - 1);

		if (ODD(i)) {
			negate(p2);
		}


		add(p, p2, p3);
		assign(p3, p, 0 /*verbose_level*/);
		
		for (j = 0; j < (k - 1) * (k - 1); j++) {
			delete_object(N[j]);
		}
		FREE_OBJECTS(N);
	}

	delete_object(p1);
	delete_object(p2);
	delete_object(p3);
	if (f_v) {
		cout << "unipoly_domain::determinant done" << endl;
	}
}

void unipoly_domain::deletion_matrix(unipoly_object *M,
		int k, int delete_row, int delete_column,
		unipoly_object *&N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int k1;
	int i, j, ii, jj;

	if (f_v) {
		cout << "unipoly_domain::deletion_matrix" << endl;
	}
	k1 = k - 1;
	N = NEW_OBJECTS(unipoly_object, k1 * k1);
	for (i = 0; i < k1 * k1; i++) {
		create_object_of_degree(N[i], 0);
	}

	for (i = 0, ii = 0; i < k; i++) {
		if (i == delete_row) {
			continue;
		}
		for (j = 0, jj = 0; j < k; j++) {
			if (j == delete_column) {
				continue;
			}

			assign(M[i * k + j], N[ii * k1 + jj], 0 /*verbose_level*/);

			jj++;
		}
		ii++;
	}
	if (f_v) {
		cout << "unipoly_domain::deletion_matrix done" << endl;
	}
}

void unipoly_domain::center_lift_coordinates(unipoly_object a, int q)
{
	//int verbose_level = 0;
	//int f_v = (verbose_level >= 1);
	int *ra = (int *) a;
	int m = ra[0];
	int q2;

	q2 = q >> 1;

	int *A = ra + 1;
	int i, x;

	for (i = 0; i <= m; i++) {
		x = A[i];
		if (x > q2) {
			x -= q;
		}
		A[i] = x;
	}
}

void unipoly_domain::reduce_modulo_p(unipoly_object a, int p)
{
	//int verbose_level = 0;
	//int f_v = (verbose_level >= 1);
	int *ra = (int *) a;
	int m = ra[0];

	int *A = ra + 1;
	int i, x;

	for (i = 0; i <= m; i++) {
		x = A[i];
		A[i] = x % p;
	}
}


}}}


