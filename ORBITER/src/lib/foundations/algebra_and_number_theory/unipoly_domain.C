// unipoly_domain.C
//
// Anton Betten
//
// started:  November 16, 2002




#include "foundations.h"


unipoly_domain::unipoly_domain(finite_field *GFq)
{
	gfq = GFq;
	f_factorring = FALSE;
	
}

unipoly_domain::unipoly_domain(finite_field *GFq, unipoly_object m)
{
	INT i, a, b;
	
	gfq = GFq;
	f_factorring = TRUE;
	factor_degree = ((INT *)m)[0];
	factor_coeffs = NEW_INT(factor_degree + 1);
	for (i = 0; i <= factor_degree; i++) {
		factor_coeffs[i] = ((INT *)m)[1 + i];
		}
	//factor_coeffs = ((INT *)m) + 1;
	if (factor_coeffs[factor_degree] != 1) {
		cout << "unipoly_domain::unipoly_domain() factor polynomial is not monic" << endl;
		exit(1);
		}
	for (i = 0; i < factor_degree; i++) {
		a = factor_coeffs[i];
		b = gfq->negate(a);
		factor_coeffs[i] = b;
		}
	factor_poly = m;
}

unipoly_domain::~unipoly_domain()
{
	//INT i, a, b;
	
	if (f_factorring) {
#if 0
		for (i = 0; i < factor_degree; i++) {
			a = factor_coeffs[i];
			b = gfq->negate(a);
			factor_coeffs[i] = b;
			}
#endif
		FREE_INT(factor_coeffs);
		}
}

void unipoly_domain::create_object_of_degree(unipoly_object &p, INT d)
{
	if (f_factorring) {
		cout << "unipoly_domain::create_object_of_degree() a factorring" << endl;
		exit(1);
		}
	INT *rep = NEW_INT(d + 2);
	rep[0] = d;
	INT *coeff = rep + 1;
	INT i;
	
#if 0
	if (p) {
		FREE_INT((INT *)p);
		}
#endif
	for (i = 0; i <= d; i++) {
		coeff[i] = 0;
		}
	rep[0] = d;
	p = (void *) rep;
}

void unipoly_domain::create_object_of_degree_with_coefficients(unipoly_object &p, INT d, INT *coeff)
{
	if (f_factorring) {
		cout << "unipoly_domain::create_object_of_degree_with_coefficients() a factorring" << endl;
		exit(1);
		}
	INT *rep = NEW_INT(d + 2);
	rep[0] = d;
	INT *C = rep + 1;
	INT i;
	
#if 0
	if (p) {
		FREE_INT((INT *)p);
		}
#endif
	for (i = 0; i <= d; i++) {
		C[i] = coeff[i];
		}
	rep[0] = d;
	p = (void *) rep;
}

void unipoly_domain::create_object_by_rank(unipoly_object &p, INT rk)
{
	INT len = INT_logq(rk, gfq->q);
	
	if (f_factorring) {
		if (len > factor_degree) {
			cout << "unipoly_domain::create_object_by_rank() len > factor_degree" << endl;
			exit(1);
			}
		len = factor_degree;
		}
	INT *rep = NEW_INT(len + 1);
	rep[0] = len - 1;
	INT *coeff = rep + 1;
	INT i = 0;
	
	do {
		coeff[i] = rk % gfq->q;
		rk /= gfq->q;
		i++;
		} while (rk);
	rep[0] = i - 1;
	p = (void *) rep;
}

void unipoly_domain::create_object_by_rank_longinteger(unipoly_object &p, 
	longinteger_object &rank, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	longinteger_object rk, rk1;
	longinteger_domain D;
	
	INT len = D.logarithm_base_b(rank, gfq->q);
	//cout << "len = " << len << endl;
	
	if (f_v) {
		cout << "unipoly_domain::create_object_by_rank_longinteger rank=" << rank << endl;
		}
	if (f_factorring) {
		if (len > factor_degree) {
			cout << "unipoly_domain::create_object_by_rank_longinteger() len > factor_degree" << endl;
			exit(1);
			}
		len = factor_degree;
		}
	INT *rep = NEW_INT(len + 1);
	rep[0] = len - 1;
	INT *coeff = rep + 1;
	INT i = 0;
	
	rank.assign_to(rk);
	do {
		D.integral_division_by_INT(rk, gfq->q, rk1, coeff[i]);
		//cout << "rk=" << rk << " coeff[" << i << "] = " << coeff[i] << endl;
		// coeff[i] = rk % gfq->q;
		if (f_vv) {
			cout << "quotient " << rk1 << " remainder " << coeff[i] << endl;
			}
		rk1.assign_to(rk);
		//rk /= gfq->q;
		i++;
		} while (!rk.is_zero());
	rep[0] = i - 1;
	p = (void *) rep;
	//print_object(p, cout); cout << endl;
}

void unipoly_domain::create_object_by_rank_string(unipoly_object &p, const BYTE *rk, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_object rank;
	
	rank.create_from_base_10_string(rk, verbose_level);
	
	create_object_by_rank_longinteger(p, rank, verbose_level);
	if (f_v) {
		cout << "unipoly_domain::create_object_by_rank_string ";
		print_object(p, cout); cout << endl;
		}
}

void unipoly_domain::create_Dickson_polynomial(unipoly_object &p, INT *map)
{
	if (f_factorring) {
		cout << "unipoly_domain::create_Dickson_polynomial() a factorring" << endl;
		exit(1);
		}
	INT d = gfq->q - 1;
	INT *rep = NEW_INT(d + 2);
	rep[0] = d;
	INT *coeff = rep + 1;
	
	gfq->Dickson_polynomial(map, coeff);
	rep[0] = d;
	p = (void *) rep;
	degree(p);
}

void unipoly_domain::delete_object(unipoly_object &p)
{
	INT *rep = (INT *) p;
	FREE_INT(rep);
	p = NULL;
}

void unipoly_domain::unrank(unipoly_object p, INT rk)
{
	INT *rep = (INT *) p;
	INT *coeff = rep + 1;
	INT i = 0;
	
	do {
		coeff[i] = rk % gfq->q;
		rk /= gfq->q;
		i++;
		} while (rk);
	rep[0] = i - 1;
}

void unipoly_domain::unrank_longinteger(unipoly_object p, longinteger_object &rank)
{
	INT *rep = (INT *) p;
	INT *coeff = rep + 1;
	INT i = 0;
	
	longinteger_object rank1, rank2;
	longinteger_domain D;
	
	rank.assign_to(rank1);
	do {
		D.integral_division_by_INT(rank1, gfq->q, rank2, coeff[i]);
		//coeff[i] = rk % gfq->q;
		//rk /= gfq->q;
		rank2.assign_to(rank1);
		i++;
		} while (!rank1.is_zero());
	rep[0] = i - 1;
}

INT unipoly_domain::rank(unipoly_object p)
{
	INT *rep = (INT *) p;
	INT d = rep[0]; // degree
	INT *coeff = rep + 1;
	INT rk = 0, i;
	
	for (i = d; i >= 0; i--) {
		rk *= gfq->q;
		rk += coeff[i];
		}
	return rk;
}

void unipoly_domain::rank_longinteger(unipoly_object p, longinteger_object &rank)
{
	INT *rep = (INT *) p;
	INT d = rep[0]; // degree
	INT *coeff = rep + 1;
	INT i;
	longinteger_object q, rk, rk1, c;
	longinteger_domain D;
	
	rk.create(0);
	q.create(gfq->q);
	for (i = d; i >= 0; i--) {
		D.mult(rk, q, rk1);
		c.create(coeff[i]);
		D.add(rk1, c, rk);
		//rk *= gfq->q;
		//rk += coeff[i];
		}
	rk.assign_to(rank);
}

INT unipoly_domain::degree(unipoly_object p)
{
	INT *rep = (INT *) p;
	INT d = rep[0]; // degree
	INT *coeff = rep + 1;
	INT i;
	
	for (i = d; i >= 0; i--) {
		if (coeff[i]) {
			break;
			}
		}
	rep[0] = i;
	return i;
}

INT unip_f_print_sub = FALSE;
INT unip_f_use_variable_name = FALSE;
BYTE unip_variable_name[128];

ostream& unipoly_domain::print_object(unipoly_object p, ostream& ost)
{
	INT i, k;
	INT f_prev = FALSE;
	BYTE *x, *y;
	INT f_nothing_printed_at_all = TRUE;
	INT *rep = (INT *) p;
	INT d = rep[0]; // degree
	INT *coeff = rep + 1;
	
	if (unip_f_use_variable_name) {
		x = unip_variable_name;
		}
	else {
		x = (BYTE *) "X";
		}
	if (unip_f_print_sub) {
		y = (BYTE *) "_";
		}
	else {
		y = (BYTE *) "^";
		}
	// ost << "(";
	for (i = d; i >= 0; i--) {
		k = coeff[i];
		if (k == 0) {
			if (i == 0 && f_nothing_printed_at_all) {
				ost << "0";
				}
			continue;
			}
		f_nothing_printed_at_all = FALSE;
		if (k < 0) {
			ost << " - ";
			}
		else if (f_prev) {
			ost << " + ";
			}
		if (k < 0)
			k = -k;
		if (k != 1 || (i == 0 && !unip_f_use_variable_name)) {
			//l = gfq->log_alpha(k);
			//ost << "\\alpha^{" << l << "}";
			ost << k;
			}
		if (i == 0) {
			if (unip_f_use_variable_name) {
				ost << x;
				ost << y;
				ost << "0";
				}
			}
		else if (i == 1) {
			ost << x;
			if (unip_f_print_sub) {
				ost << y;
				ost << "1";
				}
			}
		else if (i > 1) {
			ost << x;
			ost << y;
#if 0
			if (current_printing_mode() == printing_mode_latex) {
				if (i < 10) 
					ost << i;
				else
					ost << "{" << i << "}";
				}
			else {
#endif
				ost << "{" << i << "}";
				//}
			}
		f_prev = TRUE;
		}
	// ost << ")";
	return ost;
}

void unipoly_domain::assign(unipoly_object a, unipoly_object &b)
{
	INT f_v = FALSE;
	
	if (f_factorring) {
		if (f_v) {
			cout << "unipoly_domain::assign with factorring" << endl;
			}
		INT *ra = (INT *) a;
		INT *rb = (INT *) b;
		INT *A = ra + 1;
		INT *B = rb + 1;
		INT i;
		for (i = 0; i < factor_degree; i++) {
			B[i] = A[i];
			}
		rb[0] = ra[0];
		}
	else {
		INT *ra = (INT *) a;
		INT *rb = (INT *) b;
		INT m = ra[0];
		FREE_INT(rb);
		rb = NEW_INT(m + 2);
		rb[0] = m;
		b = (void *) rb;
		if (f_v) {
			cout << "unipoly_domain::assign m=" << m << endl;
			cout << "a=";
			print_object(a, cout);
			cout << endl;
			}
		INT *A = ra + 1;
		INT *B = rb + 1;
		INT i;
		for (i = 0; i <= m; i++) {
			B[i] = A[i];
			if (f_v) {
				cout << "i=" << i << " A[i]=" << A[i] << " B[i]=" << B[i] << endl;
				}
			}
		if (f_v) {
			cout << "unipoly_domain::assign finished" << endl;
			}
		}
}

void unipoly_domain::one(unipoly_object p)
{
	INT *rep = (INT *) p;
	INT d = rep[0]; // degree
	INT *coeff = rep + 1;
	INT i;
	
	for (i = d; i >= 1; i--) {
		coeff[i] = 0;
		}
	coeff[0] = 1;
	rep[0] = 0;
}

void unipoly_domain::m_one(unipoly_object p)
{
	INT *rep = (INT *) p;
	INT d = rep[0]; // degree
	INT *coeff = rep + 1;
	INT i;
	
	for (i = d; i >= 1; i--) {
		coeff[i] = 0;
		}
	coeff[0] = gfq->negate(1);
	rep[0] = 0;
}

void unipoly_domain::zero(unipoly_object p)
{
	INT *rep = (INT *) p;
	INT d = rep[0]; // degree
	INT *coeff = rep + 1;
	INT i;
	
	for (i = d; i >= 1; i--) {
		coeff[i] = 0;
		}
	coeff[0] = 0;
	rep[0] = 0;
}

INT unipoly_domain::is_one(unipoly_object p)
{
	INT *rep = (INT *) p;
	INT d = rep[0]; // degree
	INT *coeff = rep + 1;
	INT i;
	
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

INT unipoly_domain::is_zero(unipoly_object p)
{
	INT *rep = (INT *) p;
	INT d = rep[0]; // degree
	INT *coeff = rep + 1;
	INT i;
	
	for (i = d; i >= 0; i--) {
		if (coeff[i]) {
			return FALSE;
			}
		}
	return TRUE;
}

void unipoly_domain::negate(unipoly_object a)
{
	INT *ra = (INT *) a;
	INT m = ra[0];
	INT *A = ra + 1;
	INT i;
	
	for (i = 0; i <= m; i++) {
		A[i] = gfq->negate(A[i]);
		}
}

void unipoly_domain::make_monic(unipoly_object &a)
{
	INT *ra = (INT *) a;
	INT m = ra[0];
	INT *A = ra + 1;
	INT i, c, cv;

	while (A[m] == 0 && m > 0) {
		m--;
		}
	if (m == 0 && A[0] == 0) {
		cout << "unipoly_domain::make_monic the polynomial is zero" << endl;
		exit(1);
		}
	c = A[m];
	if (c != 1) {
		cv = gfq->inverse(c);
		for (i = 0; i <= m; i++) {
			A[i] = gfq->mult(A[i], cv);
			}
		}
}

void unipoly_domain::add(unipoly_object a, unipoly_object b, unipoly_object &c)
{
	INT verbose_level = 0;
	INT f_v = (verbose_level >= 1);
	INT *ra = (INT *) a;
	INT *rb = (INT *) b;
	INT m = ra[0];
	INT n = rb[0];
	INT mn = MAXIMUM(m, n);
	
	INT *rc = (INT *) c;
	FREE_INT(rc);
	rc = NEW_INT(mn + 2);
	
	INT *A = ra + 1;
	INT *B = rb + 1;
	INT *C = rc + 1;
	INT i, x, y;
	
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
			cout << "unipoly_domain::add x=" << x << " y=" << y << endl;
			}
		C[i] = gfq->add(x, y);
		}
	c = (void *) rc;
}

void unipoly_domain::mult(unipoly_object a, unipoly_object b, unipoly_object &c)
{
	if (f_factorring) {
		mult_mod(a, b, c, factor_degree, factor_coeffs, 0);
		return;
		}
	else {
		mult_easy(a, b, c);
		return;
		}
}

void unipoly_domain::mult_easy(unipoly_object a, unipoly_object b, unipoly_object &c)
{
	INT *ra = (INT *) a;
	INT *rb = (INT *) b;
	INT m = ra[0];
	INT n = rb[0];
	INT mn = m + n;
	
	INT *rc = (INT *) c;
	FREE_INT(rc);
	rc = NEW_INT(mn + 2);
	
	INT *A = ra + 1;
	INT *B = rb + 1;
	INT *C = rc + 1;
	INT i, j, k, x, y;
	
	rc[0] = mn;
	for (i = 0; i <= mn; i++) {
		C[i] = 0;
		}
	for (i = m; i >= 0; i--) {
		for (j = n; j >= 0; j--) {
			k = i + j;
			x = C[k];
			y = gfq->mult(A[i], B[j]);
			if (x == 0) {
				C[k] = y;
				}
			else {
				C[k] = gfq->add(x, y);
				}
			}
		}
	c = (void *) rc;
}

void unipoly_domain::mult_mod(unipoly_object a, unipoly_object b, unipoly_object &c, 
	INT factor_polynomial_degree, INT *factor_polynomial_coefficents_negated, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *ra = (INT *) a;
	INT *rb = (INT *) b;
	INT *rc = (INT *) c;
	INT m = ra[0];
	INT n = rb[0];
	INT *A = ra + 1;
	INT *B = rb + 1;
	INT *C = rc + 1;
	INT carry, c1;
	INT i, j;
	
	if (f_v) {
		cout << "unipoly_domain::mult_mod" << endl;
		}
	if (f_vv) {
		cout << "multiplying ";
		print_object(ra, cout);
		cout << " x ";
		print_object(rb, cout);
		cout << " modulo - (";
		INT_vec_print(cout, factor_polynomial_coefficents_negated, factor_polynomial_degree + 1);
		cout << ")";
		cout << endl;
		}
#if 0
	if (!f_factorring) {
		cout << "unipoly_domain::mult_mod not a factorring" << endl;
		exit(1);
		}
#endif
	if (rc[0] != factor_polynomial_degree - 1) {
		FREE_INT(rc);
		rc = NEW_INT(factor_polynomial_degree - 1 + 2);
		rc[0] = factor_polynomial_degree - 1;
		C = rc + 1;
		}
	for (j = 0 ; j < factor_polynomial_degree; j++) {
		C[j] = 0;
		}
	
	for (i = m; i >= 0; i--) {
		for (j = 0; j <= n; j++) {
			c1 = gfq->mult(A[i], B[j]);
			C[j] = gfq->add(C[j], c1);
			if (f_vv) {
				if (c1) {
					cout << A[i] << "x^" << i << " * " << B[j] << " x^" << j << " = " << c1 << " x^" << j << " result = ";
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
						C[j] = gfq->add(C[j], factor_polynomial_coefficents_negated[j]);
						}
					}
				else {
					for (j = 0; j < factor_polynomial_degree; j++) {
						c1 = gfq->mult(carry, factor_polynomial_coefficents_negated[j]);
						C[j] = gfq->add(C[j], c1);
						}
					}
				}
			}
		}
	c = rc;
	if (f_v) {
		cout << "unipoly_domain::mult_mod done" << endl;
		}
}

void unipoly_domain::Frobenius_matrix(INT *&Frob, unipoly_object factor_polynomial, INT verbose_level)
// the j-th column of Frob is x^{j*q} mod m

{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix" << endl;
		}
	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix q=" << gfq->q << endl;
		}
#if 0
	if (!f_factorring) {
		cout << "unipoly_domain::Frobenius_matrix not a factorring" << endl;
		exit(1);
		}
#endif
	unipoly_object a, b, c, m_mod, Q, R;
	INT i, j, d1;
	INT factor_polynomial_degree;
	
	factor_polynomial_degree = degree(factor_polynomial);
	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix degree=" << factor_polynomial_degree << endl;
		cout << "unipoly_domain::Frobenius_matrix m = ";
		print_object(factor_polynomial, cout);
		cout << endl;
		}

	Frob = NEW_INT(factor_polynomial_degree * factor_polynomial_degree);
	INT_vec_zero(Frob, factor_polynomial_degree * factor_polynomial_degree);
#if 0
	for (i = 0; i < factor_polynomial_degree * factor_polynomial_degree; i++) {
		Frob[i] = 0;
		}
#endif
	Frob[0] = 1; // the first column of Frob is (1,0,...,0)
	
	create_object_by_rank(a, gfq->q); // the polynomial X
	create_object_by_rank(b, 1); // the polynomial 1
	create_object_by_rank(c, 0);
	create_object_by_rank(m_mod, 0);
	create_object_by_rank(Q, 0);
	create_object_by_rank(R, 0);
	
	assign(factor_polynomial, m_mod);
	negate(m_mod);
	if (f_v) {
		cout << "unipoly_domain::Frobenius_matrix m_mod = ";
		print_object(m_mod, cout);
		cout << endl;
		}
	
	power_INT(a, gfq->q, 0 /* verbose_level */);
	if (f_vv) {
		cout << "unipoly_domain::Frobenius_matrix a = x^q = ";
		print_object(a, cout);
		cout << endl;
		}
	integral_division(a, factor_polynomial, Q, R, 0 /* verbose_level */);
	assign(R, a);
	if (f_vv) {
		cout << "unipoly_domain::Frobenius_matrix a = x^q mod m = ";
		print_object(a, cout);
		cout << endl;
		}
	for (j = 1; j < factor_polynomial_degree; j++) {
		if (f_vv) {
			cout << "unipoly_domain::Frobenius_matrix j = " << j << endl;
			cout << "b = ";
			print_object(b, cout);
			cout << endl;
			cout << "a = ";
			print_object(a, cout);
			cout << endl;
			}
		mult_mod(b, a, c, factor_polynomial_degree, ((INT *)m_mod) + 1, 0);
		if (f_vv) {
			cout << "c = ";
			print_object(c, cout);
			cout << endl;
			}
		assign(c, b);
		// now b = x^{j*q}
		if (f_vv) {
			cout << "unipoly_domain::Frobenius_matrix x^{" << j << "*q}=";
			print_object(b, cout);
			cout << endl;
			}
		d1 = degree(b);
		INT *rb = (INT *) b;
		INT *B = rb + 1;

		// put B in the j-th column of F:
		for (i = 0; i <= d1; i++) {
			Frob[i * factor_polynomial_degree + j] = B[i];
			}
		}
	if (f_vv) {
		cout << "unipoly_domain::Frobenius_matrix=" << endl;
		INT_matrix_print(Frob, factor_polynomial_degree, factor_polynomial_degree);
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

void unipoly_domain::Berlekamp_matrix(INT *&B, unipoly_object factor_polynomial, INT verbose_level)
// subtracts the identity matrix off the given matrix (which should be the Frobenius matrix)
{
	INT f_v = (verbose_level >= 1);
	INT i, m1, a, b;
	INT factor_polynomial_degree;
	
#if 0
	if (!f_factorring) {
		cout << "unipoly_domain::Berlekamp_matrix() not a factorring" << endl;
		exit(1);
		}
#endif
	factor_polynomial_degree = degree(factor_polynomial);
	Frobenius_matrix(B, factor_polynomial, verbose_level);
	m1 = gfq->negate(1);
	
	for (i = 0; i < factor_polynomial_degree; i++) {
		a = B[i * factor_polynomial_degree + i];
		b = gfq->add(m1, a);
		B[i * factor_polynomial_degree + i] = b;
		}
	if (f_v) {
		cout << "unipoly_domain::Berlekamp_matrix of degree " << factor_polynomial_degree << " = " << endl;
		print_integer_matrix(cout, B, factor_polynomial_degree, factor_polynomial_degree);
		cout << endl;
		}
}

void unipoly_domain::integral_division_exact(unipoly_object a, unipoly_object b, unipoly_object &q, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "unipoly_domain::integral_division_exact" << endl;
		}

	unipoly_object r;

	create_object_by_rank(r, 0);

	integral_division(a, b, q, r, verbose_level - 1);
	
	delete_object(r);
	
	if (f_v) {
		cout << "unipoly_domain::integral_division_exact done" << endl;
		}
}

void unipoly_domain::integral_division(unipoly_object a, unipoly_object b, 
	unipoly_object &q, unipoly_object &r, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT *ra = (INT *) a;
	INT *rb = (INT *) b;
	//INT *A = ra + 1;
	INT *B = rb + 1;

	INT da, db;
	
	if (f_v) {
		cout << "unipoly_domain::integral_division" << endl;
		}
	da = degree(a);
	db = degree(b);
	
	if (f_factorring) {
		cout << "unipoly_domain::integral_division not good for a factorring" << endl;
		exit(1);
		}
	if (db == 0) {
		if (B[0] == 0) {
			cout << "unipoly_domain::integral_division: division by zero" << endl;
			exit(1);
			}
		}
	if (db > da) {
		INT *rq = (INT *) q;
		FREE_INT(rq);
		rq = NEW_INT(2);
		INT *Q = rq + 1;
		Q[0] = 0;
		rq[0] = 0;
		assign(a, r);
		q = rq;
		goto done;
		}

	{
	INT dq = da - db;
	INT *rq = (INT *) q;
	FREE_INT(rq);
	rq = NEW_INT(dq + 2);
	rq[0] = dq;
	
	assign(a, r);
	
	INT *rr = (INT *) r;
	
	INT *Q = rq + 1;
	INT *R = rr + 1;

	INT i, j, ii, jj, pivot, pivot_inv, x, c, d;
	
	pivot = B[db];
	pivot_inv = gfq->inverse(pivot);

	INT_vec_zero(Q, dq + 1);
#if 0
	for (i = 0; i <= dq; i++) {
		Q[i] = 0;
		}
#endif
	
	for (i = da, j = dq; i >= db; i--, j--) {
		x = R[i];
		c = gfq->mult(x, pivot_inv);
		Q[j] = c;
		c = gfq->negate(c);
		//cout << "i=" << i << " c=" << c << endl;
		for (ii = i, jj = db; jj >= 0; ii--, jj--) {
			d = B[jj];
			d = gfq->mult(c, d);
			R[ii] = gfq->add(d, R[ii]);
			}
		if (R[i] != 0) {
			cout << "unipoly::integral_division: R[i] != 0" << endl;
			exit(1);
			}
		//cout << "i=" << i << endl;
		//cout << "q="; print_object((unipoly_object) rq, cout); cout << endl;
		//cout << "r="; print_object(r, cout); cout << endl;
		}
	rr[0] = MAXIMUM(db - 1, 0);
	q = rq;
	//cout << "q="; print_object(q, cout); cout << endl;
	//cout << "r="; print_object(r, cout); cout << endl;
	}
done:
	if (f_v) {
		cout << "unipoly_domain::integral_division done" << endl;
		}
}

void unipoly_domain::derive(unipoly_object a, unipoly_object &b)
{
	INT *ra = (INT *) a;
	INT *A = ra + 1;
	INT d = degree(a);
	INT *rb = (INT *) b;
	FREE_INT(rb);
	rb = NEW_INT(d - 1 + 2);
	INT *B = rb + 1;
	INT i, ai, bi;
	
	for (i = 1; i <= d; i++) {
		ai = A[i];
		bi = i % gfq->p;
		bi = gfq->mult(ai, bi);
		B[i - 1] = bi;
		}
	rb[0] = d - 1;
	b = rb;
}

INT unipoly_domain::compare_euclidean(unipoly_object m, unipoly_object n)
{
	INT dm = degree(m);
	INT dn = degree(n);
	
	if (dm < dn) {
		return -1;
		}
	else if (dm > dn) {
		return 1;
		}
	return 0;
}

void unipoly_domain::greatest_common_divisor(unipoly_object m, unipoly_object n, 
	unipoly_object &g, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT c;
	
	if (f_v) {
		cout << "unipoly::greatest_common_divisor m=";
		print_object(m, cout);
		cout << " n=";
		print_object(n, cout);
		cout << endl;
		}
	c = compare_euclidean(m, n);
	if (c < 0) {
		return greatest_common_divisor(n, m, g, verbose_level);
		}
	if (c == 0 || is_zero(n)) {
		assign(m, g);
		return;
		}

	unipoly_object M, N, Q, R;
	
	create_object_by_rank(M, 0);
	create_object_by_rank(N, 0);
	create_object_by_rank(Q, 0);
	create_object_by_rank(R, 0);

	assign(m, M);
	assign(n, N);

	while (TRUE) {
		if (f_vv) {
			cout << "unipoly::greatest_common_divisor M=";
			print_object(M, cout);
			cout << " N=";
			print_object(N, cout);
			cout << endl;
			}
		integral_division(M, N, Q, R, verbose_level - 2);
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

		assign(N, M);
		assign(R, N);
		}
	assign(N, g);
	if (f_v) {
		cout << "unipoly::greatest_common_divisor g=";
		print_object(g, cout);
		cout << endl;
		}

	delete_object(M);
	delete_object(N);
	delete_object(Q);
	delete_object(R);
}

void unipoly_domain::extended_gcd(unipoly_object m, unipoly_object n, 
	unipoly_object &u, unipoly_object &v, 
	unipoly_object &g, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT c;
	
	if (f_v) {
		cout << "unipoly::extended_gcd() m=";
		print_object(m, cout);
		cout << " n=";
		print_object(n, cout);
		cout << endl;
		}
	c = compare_euclidean(m, n);
	if (c < 0) {
		return extended_gcd(n, m, v, u, g, verbose_level);
		}
	assign(m, u);
	assign(n, v);
	if (c == 0 || is_zero(n)) {
		one(u);
		zero(v);
		assign(m, g);
		return;
		}

	unipoly_object M, N, Q, R;
	unipoly_object u1, u2, u3, v1, v2, v3, tmp;
	
	create_object_by_rank(M, 0);
	create_object_by_rank(N, 0);
	create_object_by_rank(Q, 0);
	create_object_by_rank(R, 0);
	create_object_by_rank(u1, 1);
	create_object_by_rank(u2, 0);
	create_object_by_rank(u3, 0);
	create_object_by_rank(v1, 0);
	create_object_by_rank(v2, 1);
	create_object_by_rank(v3, 0);
	create_object_by_rank(tmp, 0);
	
	assign(m, M);
	assign(n, N);

	while (TRUE) {
		if (f_v) {
			cout << "M=";
			print_object(M, cout);
			cout << " N=";
			print_object(N, cout);
			cout << endl;
			}
		integral_division(M, N, Q, R, 0);
		if (f_v) {
			cout << "Q=";
			print_object(Q, cout);
			cout << " R=";
			print_object(R, cout);
			cout << endl;
			}
		if (is_zero(R))
			break;
		
		negate(Q);

		// u3 := u1 - Q * u2
		mult(Q, u2, tmp);
		add(u1, tmp, u3);
		
		// v3 := v1 - Q * v2
		mult(Q, v2, tmp);
		add(v1, tmp, v3);
		
		assign(N, M);
		assign(R, N);
		assign(u2, u1);
		assign(u3, u2);
		assign(v2, v1);
		assign(v3, v2);
		}
	assign(u2, u);
	assign(v2, v);
	assign(N, g);
	if (f_v) {
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

}

INT unipoly_domain::is_squarefree(unipoly_object p, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	unipoly_object a, b, u, v, g;
	INT d;
	
	create_object_by_rank(a, 0);
	create_object_by_rank(b, 0);
	create_object_by_rank(u, 0);
	create_object_by_rank(v, 0);
	create_object_by_rank(g, 0);
	
	assign(p, a);
	derive(a, b);
	if (f_v) {
		cout << "unipoly::is_squarefree() derivative p' = ";
		print_object(b, cout);
		cout << endl;
		}
	extended_gcd(a, b, u, v, g, verbose_level - 1);
	if (f_v) {
		cout << "unipoly::is_squarefree() gcd(p, p') = ";
		print_object(g, cout);
		cout << endl;
		}
	d = degree(g);
	
	delete_object(a);
	delete_object(b);
	delete_object(u);
	delete_object(v);
	delete_object(g);
	
	if (d >= 1) {
		return FALSE;
		}
	else {
		return TRUE;
		}
}

void unipoly_domain::compute_normal_basis(INT d, INT *Normal_basis, INT *Frobenius, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *v, *b, *A, deg;
	INT i, j;
	unipoly_object mue, lambda, GCD, Q, R, R1, R2;
	
	if (f_v) {
		cout << "unipoly_domain::compute_normal_basis d=" << d << endl;
		}
	deg = d;

	v = NEW_INT(deg);
	b = NEW_INT(deg);
	A = NEW_INT((deg + 1) * deg);

	create_object_by_rank(mue, 0);
	create_object_by_rank(lambda, 0);
	create_object_by_rank(GCD, 0);
	create_object_by_rank(Q, 0);
	create_object_by_rank(R, 0);
	create_object_by_rank(R1, 0);
	create_object_by_rank(R2, 0);

	i = 0;
	order_ideal_generator(d, i, mue, 
		A, Frobenius, 
		verbose_level - 2);
	
	if (f_vv) {
		cout << "unipoly_domain::compute_normal_basis Ideal(e_" << i << ") = (";
		print_object(mue, cout);
		cout << ")" << endl;
		}
	INT_vec_zero(v, deg);
	v[0] = 1;

	while (degree(mue) < deg) {
		i++;
		if (f_vv) {
			cout << "unipoly_domain::compute_normal_basis i = " << i << " / " << deg << endl;
			}

		if (f_vv) {
			cout << "unipoly_domain::compute_normal_basis before order_ideal_generator" << endl;
			}
		if (i == deg) {
			cout << "unipoly_domain::compute_normal_basis error: i == deg" << endl;
			exit(1);
			}
		order_ideal_generator(d, i, lambda, 
			A, Frobenius, 
			verbose_level - 2);
		if (f_vv) {
			cout << "unipoly_domain::compute_normal_basis Ideal(e_" << i << ") = ( lambda ),  where lambda = ";
			print_object(lambda, cout);
			cout << endl;
			cout << "unipoly_domain::compute_normal_basis Ideal(e_1,..,e_" << i - 1 << ") = ( mue ), where mue = ";
			print_object(mue, cout);
			cout << endl;
			cout << "v = ";
			INT_vec_print(cout, v, deg);
			cout << endl;
			}
		
		if (f_vv) {
			cout << "unipoly_domain::compute_normal_basis computing greatest_common_divisor(mue, lambda):" << endl;
			}
		greatest_common_divisor(mue, lambda, GCD, verbose_level - 2);
	
		if (f_vv) {
			cout << "unipoly_domain::compute_normal_basis greatest_common_divisor(mue, lambda) = ";
			print_object(GCD, cout);
			cout << endl;
			}
		
		if (degree(GCD) < degree(lambda)) {
		
			// b = (0, 0, \ldots, 0, 1, 0, ..., 0) = X^i 
			INT_vec_zero(b, deg);
			b[i] = 1;
			
			integral_division_exact(lambda, GCD, Q, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis Q = lambda / GCD = ";
				print_object(Q, cout);
				cout << endl;
				}


			take_away_all_factors_from_b(mue, Q, R, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis R = take_away_all_factors_from_b(mue, Q) = ";
				print_object(R, cout);
				cout << endl;
				}

			integral_division_exact(mue, R, Q, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis Q = mue / R = ";
				print_object(Q, cout);
				cout << endl;
				}
			
			// Frobenius Module structure: apply Q to v (q is monic):
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis before module_structure_apply" << endl;
				}
			module_structure_apply(v, Frobenius, deg, Q, verbose_level - 2);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis after module_structure_apply" << endl;
				}


			// now: Orderideal(v1) = Ideal(r) 
			// v = v *(mue/R)(Frobenius) = v * Q (Frobenius)
			
			integral_division_exact(mue, GCD, Q, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis Q = mue / GCD = ";
				print_object(Q, cout);
				cout << endl;
				}

			take_away_all_factors_from_b(lambda, Q, R1, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis R1 = take_away_all_factors_from_b(lambda, Q) = ";
				print_object(R, cout);
				cout << endl;
				}

			greatest_common_divisor(R, R1, GCD, 0 /* verbose_level */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis greatest_common_divisor(R, R1) = ";
				print_object(GCD, cout);
				cout << endl;
				}

			integral_division_exact(R1, GCD, R2, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis Q = mue / GCD = ";
				print_object(Q, cout);
				cout << endl;
				}

			// now: greatest_common_divisor(R, R2) = 1
			// R * R2 = lcm(mue, lambda) 
			
			integral_division_exact(lambda, R2, Q, 0 /* verbose_level - 2 */);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis Q = lambda / R2 = ";
				print_object(Q, cout);
				cout << endl;
				}

			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis before module_structure_apply" << endl;
				}
			module_structure_apply(b, Frobenius, deg, Q, verbose_level - 2);
			if (f_vv) {
				cout << "unipoly_domain::compute_normal_basis after module_structure_apply" << endl;
				}

			// now: Orderideal(b) = Ideal(r2) 
			// b = b *(lambda/R2)(Frobenius) = v * Q(Frobenius)
			
			for (j = 0; j < deg; j++) {
				v[j] = gfq->add(v[j], b[j]);
				}

			// Orderideal(v) = Ideal(R * R2), 
			// greatest_common_divisor(R, R2) = 1
			
			mult(R, R2, mue);
			} // if 
		if (f_v) {
			cout << "unipoly_domain::compute_normal_basis Ideal(e_1,..,e_" << i << ") = ( mue ), where mue = ";
			print_object(mue, cout);
			cout << endl;
			}
		} // while 
	
	if (f_vv) {
		cout << "unipoly_domain::compute_normal_basis generator = ";
		INT_vec_print(cout, v, deg);
		cout << endl;
		}

	if (f_vv) {
		cout << "unipoly_domain::compute_normal_basis before span_cyclic_module" << endl;
		}
	gfq->span_cyclic_module(Normal_basis, v, deg, Frobenius, verbose_level);
	if (f_vv) {
		cout << "unipoly_domain::compute_normal_basis after span_cyclic_module" << endl;
		}

	if (f_vv) {
		cout << "unipoly_domain::compute_normal_basis Normal_basis = " << endl;
		INT_matrix_print(Normal_basis, deg, deg);
		}

	FREE_INT(v);
	FREE_INT(b);
	FREE_INT(A);

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

void unipoly_domain::order_ideal_generator(INT d, INT idx, unipoly_object &mue, 
	INT *A, INT *Frobenius, 
	INT verbose_level)
// Lueneburg~\cite{Lueneburg87a} p. 105.
// Frobenius is a matrix of size d x d
// A is a matrix of size (d + 1) x d
{
	INT f_v = (verbose_level >= 1);
	INT *my_mue, mue_deg;
	
	if (f_v) {
		cout << "unipoly_domain::order_ideal_generator d=" << d << " idx = " << idx << endl;
		}

	my_mue = NEW_INT(d + 1);
	

	gfq->order_ideal_generator(d, idx, my_mue, mue_deg, 
		A, Frobenius, 
		verbose_level - 1);


	INT *Mue = (INT *) mue;
	FREE_INT(Mue);
	Mue = NEW_INT(mue_deg + 2);
	Mue[0] = mue_deg;
	INT *B = Mue + 1;
	INT i;
	for (i = 0; i <= mue_deg; i++) {
		B[i] = my_mue[i];
		}
	mue = (void *) Mue;

	FREE_INT(my_mue);


	// testing:
	if (f_v) {
		cout << "unipoly_domain::order_ideal_generator d=" << d << " idx = " << idx << " testing" << endl;
		cout << "mue=";
		print_object(mue, cout);
		cout << endl;
		}
	INT *v;

	v = NEW_INT(d);
	INT_vec_zero(v, d);
	v[idx] = 1;
	
	module_structure_apply(v, Frobenius, d, mue, 0 /*verbose_level*/);
	for (i = 0; i < d; i++) {
		if (v[i]) {
			cout << "unipoly_domain::order_ideal_generator d=" << d << " idx = " << idx << " test fails, v=" << endl;
			INT_vec_print(cout, v, d);
			cout << endl;
			exit(1);
			}
		}
	FREE_INT(v);
	if (f_v) {
		cout << "unipoly_domain::order_ideal_generator d=" << d << " idx = " << idx << " test passed" << endl;
		}

	if (f_v) {
		cout << "unipoly_domain::order_ideal_generator done" << endl;
		}
}

void unipoly_domain::matrix_apply(unipoly_object &p, INT *Mtx, INT n, INT verbose_level)
// The matrix is applied on the left
{
	INT f_v = (verbose_level >= 1);
	INT *v1, *v2;
	INT i, d;

	if (f_v) {
		cout << "unipoly_domain::matrix_apply" << endl;
		}
	v1 = NEW_INT(n);
	v2 = NEW_INT(n);
	
	d = degree(p);
	if (d >= n) {
		cout << "unipoly_domain::matrix_apply d >= n" << endl;
		exit(1);
		}
	for (i = 0; i <= d; i++) {
		v1[i] = ((INT *)p)[1 + i];
		}
	for ( ; i < n; i++) {
		v1[i] = 0;
		}
	if (f_v) {
		cout << "unipoly_domain::matrix_apply v1 = ";
		INT_vec_print(cout, v1, n);
		cout << endl;
		}
	gfq->mult_vector_from_the_right(Mtx, v1, v2, n, n);
	if (f_v) {
		cout << "unipoly_domain::matrix_apply v2 = ";
		INT_vec_print(cout, v2, n);
		cout << endl;
		}

	delete_object(p);
	create_object_of_degree(p, n);
	for (i = 0; i < n; i++) {
		((INT *)p)[1 + i] = v2[i];
		}
	
	
	FREE_INT(v1);
	FREE_INT(v2);
	
	if (f_v) {
		cout << "unipoly_domain::matrix_apply done" << endl;
		}
}

void unipoly_domain::substitute_matrix_in_polynomial(unipoly_object &p, INT *Mtx_in, INT *Mtx_out, INT k, INT verbose_level)
// The matrix is substituted into the polynomial
{
	INT f_v = (verbose_level >= 1);
	INT *M1, *M2;
	INT i, j, h, c, d, *P, *coeffs;

	if (f_v) {
		cout << "unipoly_domain::substitute_matrix_in_polynomial" << endl;
		}
	M1 = NEW_INT(k * k);
	M2 = NEW_INT(k * k);
	P = (INT *)p;
	d = P[0];
	coeffs = P + 1;
	h = d;
	c = coeffs[h];
	for (i = 0; i < k * k; i++) {
		M1[i] = gfq->mult(c, Mtx_in[i]);
		}
	for (h--; h >= 0; h--) {
		c = coeffs[h];
		for (i = 0; i < k; i++) {
			for (j = 0; j < k; j++) {
				if (i == j) {
					M2[i * k + j] = gfq->add(c, M1[i * k + j]);
					}
				else {
					M2[i * k + j] = M1[i * k + j];
					}
				}
			}
		if (h) {
			gfq->mult_matrix(M2, Mtx_in, M1, k, k, k);
			}
		else {
			INT_vec_copy(M2, M1, k * k);
			}
		}
	INT_vec_copy(M1, Mtx_out, k * k);

	FREE_INT(M1);
	FREE_INT(M2);
	if (f_v) {
		cout << "unipoly_domain::substitute_matrix_in_polynomial done" << endl;
		}
}


INT unipoly_domain::substitute_scalar_in_polynomial(unipoly_object &p, INT scalar, INT verbose_level)
// The scalar 'scalar' is substituted into the polynomial
{
	INT f_v = (verbose_level >= 1);
	INT m1, m2;
	INT h, c, d, *P, *coeffs;

	if (f_v) {
		cout << "unipoly_domain::substitute_scalar_in_polynomial" << endl;
		}
	P = (INT *)p;
	d = P[0];
	coeffs = P + 1;
	h = d;
	c = coeffs[h];
	m1 = gfq->mult(c, scalar);
	for (h--; h >= 0; h--) {
		c = coeffs[h];
		m2 = gfq->add(c, m1);
		if (h) {
			m1 = gfq->mult(m2, scalar);
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

void unipoly_domain::module_structure_apply(INT *v, INT *Mtx, INT n, unipoly_object p, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *v1, *v2;
	INT i, j, d, c;

	if (f_v) {
		cout << "unipoly_domain::module_structure_apply" << endl;
		}
	if (f_vv) {
		cout << "unipoly_domain::module_structure_apply applying p=" << endl;
		print_object(p, cout);
		cout << endl;
		}
	

	v1 = NEW_INT(n);
	v2 = NEW_INT(n);

	INT_vec_copy(v, v1, n);

	d = degree(p);
	INT *pp;

	pp = ((INT *)p) + 1;
	c = pp[d];
	if (c != 1) {
		for (j = 0; j < n; j++) {
			v1[j] = gfq->mult(v1[j], c);
			}
		}
#if 0
	if (!gfq->is_one(pp[d])) {
		cout << "unipoly_domain::module_structure_apply p is not monic, leading coefficient is " << pp[d] << endl;
		exit(1);
		}
#endif
	for (i = d - 1; i >= 0; i--) {
		if (f_vv) {
			cout << "unipoly_domain::module_structure_apply i = " << i << endl;
			cout << "unipoly_domain::module_structure_apply v1 = ";
			INT_vec_print(cout, v1, n);
			cout << endl;
			}
		
		gfq->mult_vector_from_the_right(Mtx, v1, v2, n, n);

		if (f_vv) {
			cout << "unipoly_domain::module_structure_apply i = " << i << endl;
			cout << "unipoly_domain::module_structure_apply v2 = ";
			INT_vec_print(cout, v1, n);
			cout << endl;
			}

		c = pp[i];


		if (f_vv) {
			cout << "unipoly_domain::module_structure_apply i = " << i;
			cout << " c = " << c << endl;
			}
		for (j = 0; j < n; j++) {
			v1[j] = gfq->add(gfq->mult(v[j], c), v2[j]);
			}

		if (f_vv) {
			cout << "unipoly_domain::module_structure_apply i = " << i << endl;
			cout << "unipoly_domain::module_structure_apply v1 = ";
			INT_vec_print(cout, v1, n);
			cout << endl;
			}

		} // next i

	for (j = 0; j < n; j++) {
		v[j] = v1[j];
		}

	FREE_INT(v1);
	FREE_INT(v2);
	
	if (f_v) {
		cout << "unipoly_domain::module_structure_apply done" << endl;
		}
}


void unipoly_domain::take_away_all_factors_from_b(unipoly_object a, 
	unipoly_object b, unipoly_object &a_without_b, INT verbose_level)
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
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "unipoly_domain::take_away_all_factors_from_b" << endl;
		}
	

	unipoly_object G, A, Q;

	create_object_by_rank(G, 0);
	create_object_by_rank(A, 0);
	create_object_by_rank(Q, 0);

	assign(a, A);

	greatest_common_divisor(A, b, G, verbose_level - 2);

	while (degree(G)) {

		integral_division_exact(A, G, Q, 0 /* verbose_level - 2 */);

		assign(Q, A);
		
		greatest_common_divisor(A, b, G, verbose_level - 2);
		
		}
	
	assign(A, a_without_b);

	delete_object(G);
	delete_object(A);
	delete_object(Q);

	if (f_v) {
		cout << "unipoly_domain::take_away_all_factors_from_b done" << endl;
		}
}

INT unipoly_domain::is_irreducible(unipoly_object a, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *B;
	INT r;
	INT *base_cols;
	INT factor_polynomial_degree;
	
	factor_polynomial_degree = degree(a);
	
	if (!is_squarefree(a, verbose_level)) {
		return FALSE;
		}
	
	//unipoly_domain Fq(gfq, a);
	
	Berlekamp_matrix(B, a, verbose_level);
	
	base_cols = NEW_INT(factor_polynomial_degree);
	
	r = gfq->Gauss_INT(B, FALSE /* f_special */, FALSE /* f_complete */, base_cols, 
		FALSE /* f_P */, NULL /* P */, 
		factor_polynomial_degree, factor_polynomial_degree, 0 /* Pn */, 0 /* verbose_level */);
	if (f_v) {
		cout << "has rank " << r << endl;
		}
	
	FREE_INT(B);
	FREE_INT(base_cols);

	if (r == factor_polynomial_degree - 1) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}

void unipoly_domain::singer_candidate(unipoly_object &m, INT p, INT d, INT b, INT a)
{
	create_object_of_degree(m, d);
	INT *M = ((INT *)m) + 1;
	M[d] = 1;
	M[d - 1] = 1;
	M[1] = b;
	M[0] = a;
}

INT unipoly_domain::is_primitive(unipoly_object &m, 
	longinteger_object &qm1, 
	INT nb_primes, longinteger_object *primes, 
	INT verbose_level)
//Returns TRUE iff the polynomial $x$ has order $qm1$ 
//modulo the polynomial m (over GF(p)). 
//The prime factorization of $qm1$ must be given in primes (only the primes).
//A polynomial $a$ has order $s$ mod $m$ ($q = this$) iff 
//$a^m =1 mod q$ and $a^{s/p_i} \not= 1 mod m$ for all $p_i \mid s.$ 
//In this case, we have $a=x$ and we assume that $a^qm1 = 1 mod q.$
{
	INT f_v = (verbose_level >= 1);
	longinteger_object qm1_over_p, r, u;
	longinteger_domain D;
	unipoly_object M;
	INT i;
	
	create_object_of_degree(M, ((INT*)m)[0]);
	assign(m, M);
	unipoly_domain Fq(gfq, M);
	
	if (f_v) {
		cout << "unipoly_domain::is_primitive q=" << gfq->q << endl;
		cout << "m=";
		print_object(m, cout);
		cout << endl;
		cout << "M=";
		print_object(M, cout);
		cout << endl;
		}
	
	for (i = 0; i < nb_primes; i++) {
		D.integral_division(qm1, primes[i], qm1_over_p, r, 0);
		if (f_v) {
			cout << "qm1 / " << primes[i] << " = " << qm1_over_p << " remainder " << r << endl;
			}
		if (!r.is_zero()) {
			cout << "unipoly_domain::is_primitive the prime does not divide!" << endl;
			exit(1); 
			}
		
		unipoly_object a;
		
		Fq.create_object_by_rank(a, gfq->q); // the polynomial X
		Fq.power_longinteger(a, qm1_over_p);
		
		if (f_v) {
			cout << "X^" << qm1_over_p << " mod ";
			print_object(m, cout);
			cout << " = ";
			print_object(a, cout);
			cout << endl;
			}
		
		if (Fq.is_one(a)) {
			if (f_v) {
				cout << "is one, hence m is not primitive" << endl;
				}
			Fq.delete_object(a);
			return FALSE;
			}
		
		Fq.delete_object(a);
		
		}
	if (f_v) {
		cout << "m is primitive" << endl;
		unipoly_object a;
		for (i = 0; i <= qm1.as_INT(); i++) {
			Fq.create_object_by_rank(a, gfq->q); // the polynomial X
			u.create(i);
			Fq.power_longinteger(a, u);
			cout << "X^" << u << " = ";
			print_object(a, cout);
			cout << endl;
			}
		Fq.delete_object(a);
		}
	
	return TRUE;
}

void unipoly_domain::get_a_primitive_polynomial(unipoly_object &m, 
	INT f, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT a, b, p, nb_primes;
	unipoly_object x;
	longinteger_object q, m1, qm1;
	longinteger_object low, high, current, one, tmp;
	longinteger_domain D;
	longinteger_object *primes;
	INT *exponents;
	
	if (f_v) {
		cout << "unipoly::get_a_primitive_polynomial" << endl;
		cout << "Searching for a primitive polynomial of degree " << f <<
			" over GF(" << gfq->q << ")" << endl;
		}
	p = gfq->q;
	q.create(p);
	m1.create(-1);
	D.power_int(q, f);
	D.add(q, m1, qm1);
	if (f_vv) {
		cout << "factoring " << qm1 << endl;
		}
	D.factor_into_longintegers(qm1, nb_primes, primes, exponents, verbose_level - 2);
	//a = primitive_root(p, f_v);
	a = gfq->primitive_root();
	if (f_vv) {
		cout << "a primitive root is " << a << endl;
		}
	
	for (b = 0; b < p; b++) {
		singer_candidate(x, p, f, b, a);
		if (f_v) {
			cout << "singer candidate ";
			print_object(x, cout);
			cout << endl;
			}
		if (is_irreducible(x, verbose_level - 3) && is_primitive(x, qm1, nb_primes, primes, verbose_level - 3)) {
			if (f_v) {
				cout << "OK, we found an irreducible and primitive polynomial ";
				print_object(x, cout);
				cout << endl;
				}
			assign(x, m);
			delete_object(x);
			//cout << "deleting primes" << endl;
			delete [] primes;
			//cout << "deleting exponents" << endl;
			FREE_INT(exponents);
			return;
			}
		delete_object(x);
		}

	low.create(gfq->q);
	one.create(1);
	D.power_int(low, f);

	D.mult(low, low, high); // only monic polynomials 

	low.assign_to(current);
	
	while (TRUE) {
		
		create_object_by_rank_longinteger(x, current, verbose_level);
		
		if (f_vv) {
			cout << "candidate " << current << " : ";
			print_object(x, cout);
			cout << endl;
			}
		if (is_irreducible(x, verbose_level - 3) && 
			is_primitive(x, qm1, nb_primes, primes, verbose_level - 3)) {
			if (f_vv) {
				cout << "is irreducible and primitive" << endl;
				}
			if (f_v) {
				cout << "unipoly::get_a_primitive_polynomial() ";
				print_object(x, cout);
				cout << endl;
				}
			assign(x, m);
			delete_object(x);
			delete [] primes;
			FREE_INT(exponents);
			return;
			}
		
		delete_object(x);
		
		D.add(current, one, tmp);
		tmp.assign_to(current);
		
		if (D.compare(current, high) == 0) {
			cout << "unipoly::get_an_irreducible_polynomial() did not find an irreducible polynomial" << endl;
			exit(1); 
			}
		}
}

void unipoly_domain::get_an_irreducible_polynomial(unipoly_object &m, 
	INT f, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	unipoly_object x;
	longinteger_object low, high, current, one, tmp;
	longinteger_domain D;
	
	if (f_v) {
		cout << "unipoly::get_an_irreducible_polynomial()" << endl;
		cout << "Searching for an irreducible polynomial of degree " << f <<
			" over GF(" << gfq->q << ")" << endl;
		}
	low.create(gfq->q);
	one.create(1);
	D.power_int(low, f);

	D.mult(low, low, high); // only monic polynomials 

	low.assign_to(current);
	
	while (TRUE) {
		
		create_object_by_rank_longinteger(x, current, 0 /*verbose_level - 2*/);
		
		if (f_vv) {
			cout << "unipoly::get_an_irreducible_polynomial candidate " << current << " : ";
			print_object(x, cout);
			cout << endl;
			}
		if (is_irreducible(x, verbose_level - 3)) {
			if (f_vv) {
				cout << "is irreducible" << endl;
				}
			if (f_v) {
				cout << "unipoly::get_an_irreducible_polynomial() ";
				print_object(x, cout);
				cout << endl;
				}
			assign(x, m);
			delete_object(x);
			return;
			}
		
		delete_object(x);
		
		D.add(current, one, tmp);
		tmp.assign_to(current);
		
		if (D.compare(current, high) == 0) {
			cout << "unipoly::get_an_irreducible_polynomial() did not find an irreducible polynomial" << endl;
			exit(1); 
			}
		}
}

void unipoly_domain::power_INT(unipoly_object &a, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	unipoly_object b, c, d;
	
	if (f_v) {
		cout << "unipoly_domain::power_INT" << endl;
		}
	if (f_vv) {
		cout << "computing a=";
		print_object(a, cout);
		cout << " to the power " << n << ":" << endl;
		}
	//cout << "power_INT() a=";
	//print_object(a, cout);
	//cout << " n=" << n << endl;
	create_object_by_rank(b, 0);
	create_object_by_rank(c, 1); // c = 1
	create_object_by_rank(d, 0);
	assign(a, b);
	while (n) {
		if (f_vv) {
			cout << "n=" << n;
			cout << " b=";
			print_object(b, cout);
			cout << " c=";
			print_object(c, cout);
			cout << endl;
			}
		
		if (n % 2) {
			mult(b, c, d);
			assign(d, c);
			}
		mult(b, b, d);
		assign(d, b);
		n >>= 1;
		}
	assign(c, a);
	delete_object(b);
	delete_object(c);
	delete_object(d);
	if (f_v) {
		cout << "unipoly_domain::power_INT done" << endl;
		}
}

void unipoly_domain::power_longinteger(unipoly_object &a, longinteger_object &n)
{
	longinteger_object m, q;
	longinteger_domain D;
	unipoly_object b, c, d;
	INT r;
	
	//cout << "power_INT() a=";
	//print_object(a, cout);
	//cout << " n=" << n << endl;
	create_object_by_rank(b, 0);
	create_object_by_rank(c, 1); // c = 1
	create_object_by_rank(d, 0);
	n.assign_to(m);
	assign(a, b);
	while (!m.is_zero()) {
		D.integral_division_by_INT(m, 2, q, r);
		if (r) {
			mult(b, c, d);
			assign(d, c);
			}
		mult(b, b, d);
		assign(d, b);
		q.assign_to(m);
		}
	assign(c, a);
	delete_object(b);
	delete_object(c);
	delete_object(d);
}

void unipoly_domain::power_coefficients(unipoly_object &a, INT n)
{
	INT *ra = (INT *) a;
	INT m = ra[0];
	INT *A = ra + 1;
	INT i;
	
	for (i = 0; i <= m; i++) {
		A[i] = gfq->power(A[i], n); // GFq_power_INT(*gfq, A[i], n);
		}
}

void unipoly_domain::minimum_polynomial(unipoly_object &a, 
	INT alpha, INT p, INT verbose_level)
// computes the minimum polynomial of alpha with respect to the ground 
// field of order p (BTW: p might also be a prime power)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT q, m_alpha, u0, cnt;
	unipoly_object u, v, w;
	
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial" << endl;
		}
	if (f_factorring) {
		cout << "unipoly_domain::minimum_polynomial does not work for factorring" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial alpha = " << alpha << endl;
		}
	q = gfq->q;
	m_alpha = gfq->negate(alpha);
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial m_alpha = " << m_alpha << endl;
		}
	create_object_by_rank(u, q + m_alpha);
	create_object_by_rank(v, 0);
	create_object_by_rank(w, 0);
	if (f_vv) {
		cout << "unipoly_domain::minimum_polynomial X - alpha = ";
		print_object(u, cout);
		cout << endl;
		}
	assign(u, v);
	
	cnt = 0;
	while (TRUE) {
		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial Iteration " << cnt;
			cout << "u=";
			print_object(u, cout);
			cout << endl;
			cout << "v=";
			print_object(v, cout);
			cout << endl;
			}
		power_coefficients(v, p);
		if (f_vv) {
			cout << "conjugate = ";
			print_object(v, cout);
			cout << endl;
			}
		u0 = ((INT *)v)[1];
		if (u0 == m_alpha) {
			if (f_vv) {
				cout << "finished" << endl;
				}
			break;
			}
		mult(u, v, w);
		if (f_vv) {
			cout << "product = ";
			print_object(w, cout);
			cout << endl;
			}
		assign(w, u);
		cnt++;
		}

	if (f_vv) {
		cout << "unipoly_domain::minimum_polynomial Iteration " << cnt << " done";
		cout << "u=";
		print_object(u, cout);
		cout << endl;
		}
	assign(u, a);
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial the minimum polynomial of " << alpha << " over GF(" << p << ") is ";
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

INT unipoly_domain::minimum_polynomial_factorring(INT alpha, INT p, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT rk, r;
	
	if (!f_factorring) {
		cout << "unipoly_domain::minimum_polynomial_factorring() must be a factorring" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "factor_degree = " << factor_degree << endl;
		}
	unipoly_object *coeffs = new unipoly_object [factor_degree + 1];
	unipoly_object b, c, d;
	INT a0, ai, i, j;

	// create the polynomial X - a:
	for (i = 0; i <= factor_degree; i++) {
		if (i == 1) {
			create_object_by_rank(coeffs[i], 1);
			}
		else {
			create_object_by_rank(coeffs[i], 0);
			}
		}
	create_object_by_rank(b, alpha);
	create_object_by_rank(c, 0);
	create_object_by_rank(d, 0);
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_factorring() minimum polynomial of ";
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
	assign(b, coeffs[0]);
	
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
		
		power_INT(b, p, 0 /* verbose_level */);
		
		ai = rank(b);
		if (ai == a0)
			break;
		
		if (i == factor_degree) {
			cout << "unipoly_domain::minimum_polynomial_factorring() i == factor_degree && ai != a0" << endl;
			exit(1);
			}
		
		unipoly_object tmp = coeffs[i + 1];
		for (j = i; j >= 0; j--) {
			coeffs[j + 1] = coeffs[j];
			}
		coeffs[0] = tmp;
		for (j = 1; j <= i + 1; j++) {
			mult(coeffs[j], b, c);
			add(c, coeffs[j - 1], d);
			assign(d, coeffs[j - 1]);
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
	rk = 0;
	for (j = i; j >= 0; j--) {
		r = rank(coeffs[j]);
		rk *= p;
		rk += r;
		}
	if (f_v) {
		cout << "the rank of this polynomial over GF(" << p << ") is " << rk << endl;
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
	INT p, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (!f_factorring) {
		cout << "unipoly_domain::minimum_polynomial_factorring_longinteger() must be a factorring" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "factor_degree = " << factor_degree << endl;
		}
	unipoly_object *coeffs = new unipoly_object [factor_degree + 1];
	unipoly_object b, c, d;
	INT i, j;

	// create the polynomial X - a:
	for (i = 0; i <= factor_degree; i++) {
		if (i == 1) {
			create_object_by_rank(coeffs[i], 1);
			}
		else {
			create_object_by_rank(coeffs[i], 0);
			}
		}
	create_object_by_rank_longinteger(b, alpha, verbose_level);
	create_object_by_rank(c, 0);
	create_object_by_rank(d, 0);
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_factorring_longinteger() minimum polynomial of ";
		print_object(b, cout);
		cout << " = " << alpha << endl;
		}
	negate(b);
	if (f_vv) {
		cout << "-b = ";
		print_object(b, cout);
		cout << endl;
		}
	
	longinteger_object a0, ai;
	longinteger_domain D;
	
	rank_longinteger(b, a0);
	if (f_vv) {
		cout << "a0 = " << a0 << endl;
		}
	assign(b, coeffs[0]);
	
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
		
		power_INT(b, p, 0 /* verbose_level */);
		
		rank_longinteger(b, ai);
		if (D.compare(ai, a0) == 0)
			break;
		
		if (i == factor_degree) {
			cout << "unipoly_domain::minimum_polynomial_factorring_longinteger() i == factor_degree && ai != a0" << endl;
			exit(1);
			}
		
		unipoly_object tmp = coeffs[i + 1];
		for (j = i; j >= 0; j--) {
			coeffs[j + 1] = coeffs[j];
			}
		coeffs[0] = tmp;
		for (j = 1; j <= i + 1; j++) {
			mult(coeffs[j], b, c);
			add(c, coeffs[j - 1], d);
			assign(d, coeffs[j - 1]);
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
	
	rk.create(0);
	p_object.create(p);
	for (j = i; j >= 0; j--) {
		rank_longinteger(coeffs[j], r);
		D.mult(rk, p_object, rk1);
		D.add(rk1, r, rk);
		}
	if (f_v) {
		cout << "the rank of this polynomial over GF(" << p << ") is " << rk << endl;
		}
	for (j = 0; j <= factor_degree; j++) {
		delete_object(coeffs[j]);
		}
	delete_object(b);
	delete_object(c);
	delete_object(d);
	rk.assign_to(rk_minpoly);
}

void unipoly_domain::BCH_generator_polynomial(unipoly_object &g, INT n, 
	INT designed_distance, INT &bose_distance, 
	INT &transversal_length, INT *&transversal, 
	longinteger_object *&rank_of_irreducibles, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT p = gfq->q;
	INT e, i, j, r;
	longinteger_object q, b, m1, qm1;
	longinteger_domain D;
	
	if (f_v) {
		cout << "unipoly_domain::BCH_generator_polynomial() n=" << n << " designed_distance=" << designed_distance << " p=" << p << endl;
		}
	e = order_mod_p(p, n);
	q.create(p);
	m1.create(-1);
	D.power_int(q, e);
	D.add(q, m1, qm1);
	// q = i_power_j(p, e);
	// GF(q)=GF(p^e) has n-th roots of unity
	D.integral_division_by_INT(qm1, n, b, r);
	//b = (q - 1) / n;
	if (r != 0) {
		cout << "unipoly_domain::BCH_generator_polynomial() r != 0" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "GF(" << q << ") = GF(" << p << "^" << e << ") has " << n << "-th roots of unity" << endl;
		if (b.is_one()) {
			cout << "this is a primitive BCH code" << endl;
			}
		else {
			cout << "we take as " << n 
			<< "-th root \\beta = \\alpha^" << b << ", where \\alpha is a primitive element of the field" << endl;
			}
		}

	unipoly_object m, M, h1, h2;
		
	create_object_by_rank_string(m, get_primitive_polynomial(p, e, 0), verbose_level - 2);
	create_object_by_rank_string(M, get_primitive_polynomial(p, e, 0), verbose_level - 2);
	create_object_by_rank(g, 1);
	create_object_by_rank(h1, 0);
	create_object_by_rank(h2, 0);
	
	if (f_vv) {
		cout << "choosing the following irreducible and primitive polynomial:" << endl;
		print_object(m, cout); cout << endl;
		}

	unipoly_domain Fq(gfq, M);
	unipoly_object beta, beta_i, c;
	if (f_vvv) {
		cout << "extension field created" << endl;
		}
	Fq.create_object_by_rank(c, 0);
	Fq.create_object_by_rank(beta, p); // the primitive element alpha
	Fq.create_object_by_rank(beta_i, 1);
	if (!b.is_one()) {
		//Fq.power_INT(beta, b, 0 /* verbose_level */);
		if (f_vvv) {
			cout << "\\alpha = ";
			Fq.print_object(beta, cout);
			cout << endl;
			}
		Fq.power_longinteger(beta, b);
#if 0
		if (b.as_INT() == 11) {
			for (i = 1; i <= b.as_INT(); i++) {
				Fq.create_object_by_rank(beta, p); // the element alpha
				Fq.power_INT(beta, i, 0 /* verbose_level */);
				cout << "\\alpha^" << i << " = ";
				Fq.print_object(beta, cout);
				cout << endl;
				}
			}
#endif
		if (f_vvv) {
			cout << "\\beta = \\alpha^" << b << " = ";
			Fq.print_object(beta, cout);
			cout << endl;
			}
		}
	else {
		if (f_vvv) {
			cout << "this is a primitive BCH code" << endl;
			}
		}
	
	// now beta is a primitive n-th root of unity

#if 0
	if (1 + designed_distance - 2 >= q - 1) {
		cout << "unipoly_domain::BCH_generator_polynomial() 1 + designed_distance - 2 >= q - 1" << endl;
		exit(1);
		}
#endif

	longinteger_object *beta_rk_table = new longinteger_object[n];
	longinteger_object ai, bi;


	for (i = 0; i < n; i++) {
		Fq.rank_longinteger(beta_i, beta_rk_table[i]);
		
		if (f_vvv) {
			cout << "\\beta^" << i << " = ";
			Fq.print_object(beta_i, cout);
			cout << " = " << beta_rk_table[i] << endl;
			}
		Fq.mult(beta, beta_i, c);
		Fq.assign(c, beta_i);
		}
	if (f_vvv) {
		for (i = 0; i < n; i++) {
			cout << "\\beta^" << i << " = ";
			//Fq.print_object(beta_i, cout);
			cout << " = " << beta_rk_table[i] << endl;
			}
		}
	
	
	INT *chosen = NEW_INT(n);
	//INT *transversal = NEW_INT(n);
	//INT transversal_length = 0, i0;
	INT i0;

	transversal = NEW_INT(n);
	transversal_length = 0;

	for (i = 0; i < n; i++) {
		chosen[i] = FALSE;
		}
	
	for (i = 1; i <= 1 + designed_distance - 2; i++) {
		Fq.mult(beta, beta_i, c);
		Fq.assign(c, beta_i);
		
		Fq.rank_longinteger(beta_i, ai);
		if (f_vvv) {
			cout << "\\beta^" << i << " = ";
			Fq.print_object(beta_i, cout);
			cout << " = " << ai << endl;
			}
		if (chosen[i])
			continue;
		
		transversal[transversal_length++] = i;
		if (f_vv || f_v) {
			cout << "orbit of conjugate elements (in powers of \\beta):" << endl;
			cout << "{ ";
			}
		ai.assign_to(bi);
		i0 = i;
		do {
			chosen[i] = TRUE;
			Fq.create_object_by_rank_longinteger(c, bi, verbose_level);
			if (f_vvv) {
				cout << bi << " = ";
				Fq.print_object(c, cout);
				}
			else if (f_v) {
				cout << i << " ";
				}
			//power_coefficients(c, p);
			Fq.power_INT(c, p, 0 /* verbose_level */);
			Fq.rank_longinteger(c, bi);
			for (j = 0; j < n; j++) {
				if (D.compare(bi, beta_rk_table[j]) == 0)
					break;
				}
			if (j == n) {
				cout << "couldn't find rank in the table (A)" << endl;
				exit(1);
				}
			if (f_vv) {
				cout << " is \\beta^" << j << endl;
				}
			i = j;
			} while (j != i0);
		if (f_vv || f_v) {
			cout << "}" << endl;
			}
		}

	// compute the bose_distance:
	Fq.create_object_by_rank(beta_i, 1);
	for (i = 1; ; i++) {
		Fq.mult(beta, beta_i, c);
		assign(c, beta_i);
		Fq.rank_longinteger(beta_i, ai);
		for (j = 0; j < n; j++) {
			if (D.compare(ai, beta_rk_table[j]) == 0)
				break;
			}
		if (j == n) {
			cout << "couldn't find rank in the table (B)" << endl;
			exit(1);
			}
		if (!chosen[j]) {
			break;
			}
		}
	bose_distance = i;
	
	longinteger_object rk;
	
	if (f_vv || f_v) {
		cout << "taking the minimum polynomials of { ";
		for (i = 0; i < transversal_length; i++) {
			cout << transversal[i] << " ";
			}
		cout << "}" << endl;
		}

	rank_of_irreducibles = new longinteger_object[transversal_length];

	for (i = 0; i < transversal_length; i++) {
		
		// minimum_polynomial(h1, ai, p, f_vv);
		Fq.minimum_polynomial_factorring_longinteger(beta_rk_table[transversal[i]], rk, p, f_vv);
		create_object_by_rank_longinteger(h1, rk, verbose_level - 2);
		if (f_vv) {
			cout << "minimal polynomial of \\beta^" << transversal[i] << " is ";
			print_object(h1, cout);
			cout << " of rank " << rk << endl;
			}
		rk.assign_to(rank_of_irreducibles[i]);
		mult(g, h1, h2);
		assign(h2, g);
		}

	Fq.delete_object(c);
	Fq.delete_object(beta);
	Fq.delete_object(beta_i);
	delete_object(h1);
	delete_object(h2);
	delete_object(m);
	delete [] beta_rk_table;
	FREE_INT(chosen);
	//delete [] transversal;
	if (f_v) {
		cout << "BCH(" << n << "," << p << "," << designed_distance << ") = ";
		print_object(g, cout);
		cout << " bose_distance = " << bose_distance << endl;
		}
}

void unipoly_domain::compute_generator_matrix(unipoly_object a, INT *&genma, INT n, INT &k, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *r = (INT *) a;
	INT d = r[0];
	INT *A = r + 1;
	
	INT i, j, x;
	
	k = n - d;
	if (k < 0) {
		cout << "unipoly_domain::compute_generator_matrix() k < 0" << endl;
		exit(1);
		}
	genma = NEW_INT(k * n);
	for (i = 0; i < k * n; i++) {
		genma[i] = 0;
		}
	for (i = 0; i < k; i++) {
		for (j = 0; j <= d; j++) {
			x = A[j];
			genma[i * n + i + j] = x;
			}
		}
	if (f_v) {
		cout << "generator matrix:" << endl;
		print_integer_matrix(cout, genma, k, n);
		}
}

void unipoly_domain::print_vector_of_polynomials(unipoly_object *sigma, INT deg)
{
	INT i;

	for (i = 0; i < deg; i++) {
		cout << i << ": ";
		print_object(sigma[i], cout);
		cout << endl;
		}
}

void unipoly_domain::minimum_polynomial_extension_field(unipoly_object &g, unipoly_object m, 
	unipoly_object &minpol, INT d, INT *Frobenius, INT verbose_level)
// Lueneburg~\cite{Lueneburg87a}, p. 112.
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT deg, i, j, k;
	unipoly_object mm, h, h2, *sigma;

	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_extension_field of ";
		print_object(g, cout);
		cout << endl;
		}	
	deg = d;
	sigma = new unipoly_object[deg + 2];
	for (i = 0; i < deg + 2; i++) {
		if (i == 0) {
			create_object_by_rank(sigma[i], 1);
			}
		else {
			create_object_by_rank(sigma[i], 0);
			}
		}
	create_object_by_rank(h, 0);
	create_object_by_rank(h2, 0);
	create_object_by_rank(mm, 0);

	assign(m, mm);
	negate(mm);
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_extension_field we are working modulo - (";
		print_object(mm, cout);
		cout << ")" << endl;
		}	

	assign(g, sigma[1]);

	i = 1;
	while (TRUE) {
		i++;

		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " : g = ";
			print_object(g, cout);
			cout << endl;
			cout << "sigma=" << endl;
			print_vector_of_polynomials(sigma, deg);
			}	

		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " before matrix_apply" << endl;
			}
		matrix_apply(g, Frobenius, deg, verbose_level - 2);
		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " after matrix_apply" << endl;
			}

		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " : g=";
			print_object(g, cout);
			cout << endl;
			}	
		
		
		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " before mult_mod" << endl;
			}
		mult_mod(g, sigma[i - 1], sigma[i], degree(mm), ((INT *)mm) + 1, verbose_level - 2);
		if (f_vv) {
			cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " after mult_mod" << endl;
			cout << "sigma=" << endl;
			print_vector_of_polynomials(sigma, deg);
			}

		for (j = i - 1; j >= 1; j--) {
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " j = " << j << endl;
				}
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " j = " << j << " before mult_mod" << endl;
				}
			mult_mod(g, sigma[j - 1], h, degree(mm), ((INT *)mm) + 1, 0);
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " j = " << j << " after mult_mod" << endl;
				cout << "sigma=" << endl;
				print_vector_of_polynomials(sigma, deg);
				}
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " j = " << j << " before add" << endl;
				}
			add(sigma[j], h, h2);
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " j = " << j << " after add" << endl;
				}
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " j = " << j << " before assign" << endl;
				cout << "sigma=" << endl;
				print_vector_of_polynomials(sigma, deg);
				}
			assign(h2, sigma[j]);
			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " j = " << j << " after assign" << endl;
				}

			if (f_vv) {
				cout << "unipoly_domain::minimum_polynomial_extension_field i = " << i << " j = " << j << " iteration finished" << endl;
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
		((INT *) minpol)[1 + j] = ((INT *)sigma[i - j])[1 + 0];
		}
	for (j = 0; j <= i; j += 2) {
		((INT *) minpol)[1 + j] = gfq->negate(((INT *) minpol)[1 + j]);
		}
	make_monic(minpol);
	if (f_vv) {
		cout << "unipoly_domain::minimum_polynomial_extension_field after make_monic";
		cout << "minpol=";
		print_object(minpol, cout);
		cout << endl;
		}

	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_extension_field minpol is ";
		print_object(minpol, cout);
		cout << endl;
		}	

	delete_object(h);
	delete_object(h2);
	delete_object(mm);
	for (i = 0; i < deg + 2; i++) {
		delete_object(sigma[i]);
		}
	delete [] sigma;
	if (f_v) {
		cout << "unipoly_domain::minimum_polynomial_extension_field done" << endl;
		}	

}

void unipoly_domain::characteristic_polynomial(INT *Mtx, INT k, unipoly_object &char_poly, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	unipoly_object *M;
	INT i, j, a, m_one;

	if (f_v) {
		cout << "unipoly_domain::characteristic_polynomial" << endl;
		}
	if (f_vv) {
		cout << "unipoly_domain::characteristic_polynomial M=" << endl;
		INT_matrix_print(Mtx, k, k);
		}
	m_one = gfq->negate(1);
	M = new unipoly_object[k * k];
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			a = Mtx[i * k + j];
			if (i == j) {
				create_object_of_degree(M[i * k + j], 1);
				((INT *)M[i * k + j])[1 + 0] = a;
				((INT *)M[i * k + j])[1 + 1] = m_one;
				}
			else {
				create_object_of_degree(M[i * k + j], 0);
				((INT *)M[i * k + j])[1 + 0] = a;
				}
			}
		}

	
	if (f_vv) {
		cout << "unipoly_domain::characteristic_polynomial M - X Id" << endl;
		print_matrix(M, k);
		}

	determinant(M, k, char_poly, verbose_level);

	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			delete_object(M[i * k + j]);
			}
		}
	delete [] M;
	if (f_v) {
		cout << "unipoly_domain::characteristic_polynomial done" << endl;
		}
}

void unipoly_domain::print_matrix(unipoly_object *M, INT k)
{
	INT i, j;

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

void unipoly_domain::determinant(unipoly_object *M, INT k, unipoly_object &p, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	unipoly_object p1, p2, p3;
	
	if (k == 0) {
		delete_object(p);
		create_object_by_rank(p, 1);
		return;
		}
	delete_object(p);
	create_object_by_rank(p, 0);
	create_object_by_rank(p1, 0);
	create_object_by_rank(p2, 0);
	create_object_by_rank(p3, 0);
	
	for (i = 0; i < k; i++) {
		unipoly_object *N;
		
		deletion_matrix(M, k, i /* delete_row */, 0 /* delete_column */, N, 0 /*verbose_level - 2*/);

		determinant(N, k - 1, p1, verbose_level - 2);
		if (f_v) {
			cout << "unipoly_domain::determinant deletion of row " << i << " leads to determinant ";
			print_object(p1, cout);
			cout << endl;
			}

		mult(p1, M[i * k + 0], p2);

		if (ODD(i)) {
			negate(p2);
			}


		add(p, p2, p3);
		assign(p3, p);
		
		for (j = 0; j < (k - 1) * (k - 1); j++) {
			delete_object(N[j]);
			}
		delete [] N;
		}

	delete_object(p1);
	delete_object(p2);
	delete_object(p3);
}

void unipoly_domain::deletion_matrix(unipoly_object *M, INT k, INT delete_row, INT delete_column, unipoly_object *&N, INT verbose_level)
{
	INT k1;
	INT i, j, ii, jj;

	k1 = k - 1;
	N = new unipoly_object[k1 * k1];
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

			assign(M[i * k + j], N[ii * k1 + jj]);

			jj++;
			}
		ii++;
		}
	
}


