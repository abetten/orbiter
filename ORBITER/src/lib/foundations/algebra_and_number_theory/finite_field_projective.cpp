// finite_field_projective.C
//
// Anton Betten
//
// started:  April 2, 2003
//
// renamed from projective.cpp Nov 16, 2018



#include "foundations.h"


namespace orbiter {
namespace foundations {

void finite_field::create_projective_variety(
		const char *variety_label,
		int variety_nb_vars, int variety_degree,
		const char *variety_coeffs,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::create_projective_variety" << endl;
	}

	homogeneous_polynomial_domain *HPD;
	int *coeff;

	HPD = NEW_OBJECT(homogeneous_polynomial_domain);

	HPD->init(this, variety_nb_vars, variety_degree,
			FALSE /* f_init_incidence_structure */,
			verbose_level);

	HPD->print_monomial_ordering(cout);

	coeff = NEW_int(HPD->nb_monomials);
	int_vec_zero(coeff, HPD->nb_monomials);

	sprintf(fname, "%s.txt", variety_label);
	int *coeff_pairs;
	int len;
	int a, b, i;

	int_vec_scan(variety_coeffs, coeff_pairs, len);
	for (i = 0; i < len / 2; i++) {
		a = coeff_pairs[2 * i];
		b = coeff_pairs[2 * i + 1];
		if (b >= HPD->nb_monomials) {
			cout << "b >= HPD->nb_monomials" << endl;
			exit(1);
		}
		if (b < 0) {
			cout << "b < 0" << endl;
			exit(1);
		}
		coeff[b] = a;
	}

	Pts = NEW_int(HPD->P->N_points);

	HPD->enumerate_points(coeff, Pts, nb_pts, verbose_level);

	display_table_of_projective_points(
			cout, Pts, nb_pts, variety_nb_vars);

	FREE_int(coeff_pairs);
	FREE_int(coeff);
	FREE_OBJECT(HPD);

	if (f_v) {
		cout << "finite_field::create_projective_variety done" << endl;
	}
}

void finite_field::create_projective_curve(
		const char *variety_label,
		int curve_nb_vars, int curve_degree,
		const char *curve_coeffs,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::create_projective_curve" << endl;
	}

	homogeneous_polynomial_domain *HPD;
	int *coeff;

	HPD = NEW_OBJECT(homogeneous_polynomial_domain);

	HPD->init(this, 2, curve_degree,
			FALSE /* f_init_incidence_structure */,
			verbose_level);

	HPD->print_monomial_ordering(cout);

	coeff = NEW_int(HPD->nb_monomials);
	int_vec_zero(coeff, HPD->nb_monomials);

	sprintf(fname, "%s.txt", variety_label);
	int *coeffs;
	int len, i, j, a, b, c, s, t;
	int *v;
	int v2[2];

	int_vec_scan(curve_coeffs, coeffs, len);
	if (len != curve_degree + 1) {
		cout << "finite_field::create_projective_curve "
				"len != curve_degree + 1" << endl;
		exit(1);
	}

	nb_pts = q + 1;

	v = NEW_int(curve_nb_vars);
	Pts = NEW_int(nb_pts);

	for (i = 0; i < nb_pts; i++) {
		PG_element_unrank_modified(v2, 1, 2, i);
		s = v2[0];
		t = v2[1];
		for (j = 0; j < curve_nb_vars; j++) {
			a = HPD->Monomials[j * 2 + 0];
			b = HPD->Monomials[j * 2 + 1];
			v[j] = mult3(coeffs[j], power(s, a), power(t, b));
		}
		PG_element_rank_modified(v, 1, curve_nb_vars, c);
		Pts[i] = c;
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, curve_nb_vars);
			cout << " : " << setw(5) << c << endl;
			}
		}

	display_table_of_projective_points(
			cout, Pts, nb_pts, curve_nb_vars);


	FREE_int(v);
	FREE_int(coeffs);
	FREE_OBJECT(HPD);

	if (f_v) {
		cout << "finite_field::create_projective_curve done" << endl;
	}
}

void finite_field::PG_element_normalize(
		int *v, int stride, int len)
// last non-zero element made one
{
	int i, j, a;
	
	for (i = len - 1; i >= 0; i--) {
		a = v[i * stride];
		if (a) {
			if (a == 1)
				return;
			a = inverse(a);
			v[i * stride] = 1;
			for (j = i - 1; j >= 0; j--) {
				v[j * stride] = mult(v[j * stride], a);
				}
			return;
			}
		}
	cout << "finite_field::PG_element_normalize zero vector" << endl;
	exit(1);
}

void finite_field::PG_element_normalize_from_front(
		int *v, int stride, int len)
// first non zero element made one
{
	int i, j, a;
	
	for (i = 0; i < len; i++) {
		a = v[i * stride];
		if (a) {
			if (a == 1)
				return;
			a = inverse(a);
			v[i * stride] = 1;
			for (j = i + 1; j < len; j++) {
				v[j * stride] = mult(v[j * stride], a);
				}
			return;
			}
		}
	cout << "finite_field::PG_element_normalize_from_front "
			"zero vector" << endl;
	exit(1);
}

void finite_field::PG_elements_embed(
		int *set_in, int *set_out, int sz,
		int old_length, int new_length, int *v)
{
	int i;

	for (i = 0; i < sz; i++) {
		set_out[i] = PG_element_embed(
				set_in[i], old_length, new_length, v);
	}
}

int finite_field::PG_element_embed(
		int rk, int old_length, int new_length, int *v)
{
	int a;
	PG_element_unrank_modified(
			v, 1, old_length, rk);
	int_vec_zero(v + old_length, new_length - old_length);
	PG_element_rank_modified(
			v, 1, new_length, a);
	return a;
}

void finite_field::PG_element_rank_modified(
		int *v, int stride, int len, int &a)
{
	int i, j, q_power_j, b, sqj;
	int f_v = FALSE;
	
	if (len <= 0) {
		cout << "finite_field::PG_element_rank_modified "
				"len <= 0" << endl;
		exit(1);
		}
	nb_calls_to_PG_element_rank_modified++;
	if (f_v) {
		cout << "the vector before normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	PG_element_normalize(v, stride, len);
	if (f_v) {
		cout << "the vector after normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	for (i = 0; i < len; i++) {
		if (v[i * stride])
			break;
		}
	if (i == len) {
		cout << "finite_field::PG_element_rank_modified "
				"zero vector" << endl;
		exit(1);
		}
	for (j = i + 1; j < len; j++) {
		if (v[j * stride])
			break;
		}
	if (j == len) {
		// we have the unit vector vector e_i
		a = i;
		return;
		}
	
	// test for the all one vector: 
	if (i == 0 && v[i * stride] == 1) {
		for (j = i + 1; j < len; j++) {
			if (v[j * stride] != 1)
				break;
			}
		if (j == len) {
			a = len;
			return;
			}
		}
	
	
	for (i = len - 1; i >= 0; i--) {
		if (v[i * stride])
			break;
		}
	if (i < 0) {
		cout << "finite_field::PG_element_rank_modified "
				"zero vector" << endl;
		exit(1);
		}
	if (v[i * stride] != 1) {
		cout << "finite_field::PG_element_rank_modified "
				"vector not normalized" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "i=" << i << endl;
		}

	b = 0;
	q_power_j = 1;
	sqj = 0;
	for (j = 0; j < i; j++) {
		b += q_power_j - 1;
		sqj += q_power_j;
		q_power_j *= q;
		}
	if (f_v) {
		cout << "b=" << b << endl;
		cout << "sqj=" << sqj << endl;
		}


	a = 0;
	for (j = i - 1; j >= 0; j--) {
		a += v[j * stride];
		if (j > 0)
			a *= q;
		if (f_v) {
			cout << "j=" << j << ", a=" << a << endl;
			}
		}

	if (f_v) {
		cout << "a=" << a << endl;
		}
	
	// take care of 1111 vector being left out
	if (i == len - 1) {
		//cout << "sqj=" << sqj << endl;
		if (a >= sqj)
			a--;
		}
	
	a += b;
	a += len;
}

void finite_field::PG_element_unrank_fining(
		int *v, int len, int a)
{
	int b, c;
	
	if (len != 3) {
		cout << "finite_field::PG_element_unrank_fining "
				"len != 3" << endl;
		exit(1);
		}
	if (a <= 0) {
		cout << "finite_field::PG_element_unrank_fining "
				"a <= 0" << endl;
		exit(1);
		}
	if (a == 1) {
		v[0] = 1;
		v[1] = 0;
		v[2] = 0;
		return;
		}
	a--;
	if (a <= q) {
		if (a == 1) {
			v[0] = 0;
			v[1] = 1;
			v[2] = 0;
			return;
			}
		else {
			v[0] = 1;
			v[1] = a - 1;
			v[2] = 0;
			return;
			}
		}
	a -= q;
	if (a <= q * q) {
		if (a == 1) {
			v[0] = 0;
			v[1] = 0;
			v[2] = 1;
			return;
			}
		a--;
		a--;
		b = a % (q + 1);
		c = a / (q + 1);
		v[2] = c + 1;
		if (b == 0) {
			v[0] = 1;
			v[1] = 0;
			}
		else if (b == 1) {
			v[0] = 0;
			v[1] = 1;
			}
		else {
			v[0] = 1;
			v[1] = b - 1;
			}
		return;
		}
	else {
		cout << "finite_field::PG_element_unrank_fining "
				"a is illegal" << endl;
		exit(1);
		}
}

int finite_field::PG_element_rank_fining(
		int *v, int len)
{
	int a;

	if (len != 3) {
		cout << "finite_field::PG_element_rank_fining "
				"len != 3" << endl;
		exit(1);
		}
	//PG_element_normalize(v, 1, len);

	PG_element_normalize_from_front(v, 1, len);

	if (v[2] == 0) {
		if (v[0] == 1 && v[1] == 0) {
			return 1;
		}
		else if (v[0] == 0 && v[1] == 1) {
			return 2;
		}
		else {
			return 2 + v[1];
		}

	} else {
		if (v[0] == 0 && v[1] == 0) {
			return q + 2;
		}
		else {
			a = (q + 1) * v[2] + 2;
			if (v[0] == 1 && v[1] == 0) {
				return a;
			}
			else if (v[0] == 0 && v[1] == 1) {
				return a + 1;
			}
			else {
				return a + 1 + v[1];
			}
		}
	}
}

void finite_field::PG_element_unrank_gary_cook(
		int *v, int len, int a)
{
	int b, qm1o2, rk, i;
	
	if (len != 3) {
		cout << "finite_field::PG_element_unrank_gary_cook "
				"len != 3" << endl;
		exit(1);
		}
	if (q != 11) {
		cout << "finite_field::PG_element_unrank_gary_cook "
				"q != 11" << endl;
		exit(1);
		}
	qm1o2 = (q - 1) >> 1;
	if (a < 0) {
		cout << "finite_field::PG_element_unrank_gary_cook "
				"a < 0" << endl;
		exit(1);
		}
	if (a == 0) {
		v[0] = 0;
		v[1] = 0;
		v[2] = 1;
		}
	else {
		a--;
		if (a < q) {
			v[0] = 0;
			v[1] = 1;
			v[2] = -qm1o2 + a;
			}
		else {
			a -= q;
			rk = a;
			if (rk < q * q) {
				// (1, a, b) = 11a + b + 72, where a,b in -5..5
				b = rk % 11;
				a = (rk - b) / 11;
				v[0] = 1;
				v[1] = -qm1o2 + a;
				v[2] = -qm1o2 + b;
				}
			else {
				cout << "finite_field::PG_element_unrank_gary_cook "
						"a is illegal" << endl;
				exit(1);
				}
			}
		}
	for (i = 0; i < 3; i++) {
		if (v[i] < 0) {
			v[i] += q;
			}
		}
}

void finite_field::PG_element_unrank_modified(
		int *v, int stride, int len, int a)
{
	int n, l, ql, sql, k, j, r, a1 = a;
	
	n = len;
	if (n <= 0) {
		cout << "finite_field::PG_element_unrank_modified "
				"len <= 0" << endl;
		exit(1);
		}
	nb_calls_to_PG_element_unrank_modified++;
	if (a < n) {
		// unit vector:
		for (k = 0; k < n; k++) {
			if (k == a) {
				v[k * stride] = 1;
				}
			else {
				v[k * stride] = 0;
				}
			}
		return;
		}
	a -= n;
	if (a == 0) {
		// all one vector
		for (k = 0; k < n; k++) {
			v[k * stride] = 1;
			}
		return;
		}
	a--;
	
	l = 1;
	ql = q;
	sql = 1;
	// sql = q^0 + q^1 + \cdots + q^{l-1}
	while (l < n) {
		if (a >= ql - 1) {
			a -= (ql - 1);
			sql += ql;
			ql *= q;
			l++;
			continue;
			}
		v[l * stride] = 1;
		for (k = l + 1; k < n; k++) {
			v[k * stride] = 0;
			}
		a++; // take into account that we do not want 00001000
		if (l == n - 1 && a >= sql) {
			a++;
				// take int account that the
				// vector 11111 has already been listed
			}
		j = 0;
		while (a != 0) {
			r = a % q;
			v[j * stride] = r;
			j++;
			a -= r;
			a /= q;
			}
		for ( ; j < l; j++) {
			v[j * stride] = 0;
			}
		return;
		}
	cout << "finite_field::PG_element_unrank_modified "
			"a too large" << endl;
	cout << "len = " << len << endl;
	cout << "a = " << a1 << endl;
	exit(1);
}

void finite_field::PG_element_rank_modified_not_in_subspace(
		int *v, int stride, int len, int m, int &a)
{
	int s, qq, i;

	qq = 1;
	s = qq;
	for (i = 0; i < m; i++) {
		qq *= q;
		s += qq;
		}
	s -= (m + 1);

	PG_element_rank_modified(v, stride, len, a);
	if (a > len + s) {
		a -= s;
		}
	a -= (m + 1);
}

void finite_field::PG_element_unrank_modified_not_in_subspace(
		int *v, int stride, int len, int m, int a)
{
	int s, qq, i;

	qq = 1;
	s = qq;
	for (i = 0; i < m; i++) {
		qq *= q;
		s += qq;
		}
	s -= (m + 1);

	a += (m + 1);
	if (a > len) {
		a += s;
		}

	PG_element_unrank_modified(v, stride, len, a);
}

int finite_field::evaluate_conic_form(int *six_coeffs, int *v3)
{
	//int a = 2, b = 0, c = 0, d = 4, e = 4, f = 4, val, val1;
	//int a = 3, b = 1, c = 2, d = 4, e = 1, f = 4, val, val1;
	int val, val1;

	val = 0;
	val1 = product3(six_coeffs[0], v3[0], v3[0]);
	val = add(val, val1);
	val1 = product3(six_coeffs[1], v3[1], v3[1]);
	val = add(val, val1);
	val1 = product3(six_coeffs[2], v3[2], v3[2]);
	val = add(val, val1);
	val1 = product3(six_coeffs[3], v3[0], v3[1]);
	val = add(val, val1);
	val1 = product3(six_coeffs[4], v3[0], v3[2]);
	val = add(val, val1);
	val1 = product3(six_coeffs[5], v3[1], v3[2]);
	val = add(val, val1);
	return val;
}

int finite_field::evaluate_quadric_form_in_PG_three(
		int *ten_coeffs, int *v4)
{
	int val, val1;

	val = 0;
	val1 = product3(ten_coeffs[0], v4[0], v4[0]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[1], v4[1], v4[1]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[2], v4[2], v4[2]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[3], v4[3], v4[3]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[4], v4[0], v4[1]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[5], v4[0], v4[2]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[6], v4[0], v4[3]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[7], v4[1], v4[2]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[8], v4[1], v4[3]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[9], v4[2], v4[3]);
	val = add(val, val1);
	return val;
}

int finite_field::Pluecker_12(int *x4, int *y4)
{
	return Pluecker_ij(0, 1, x4, y4);
}

int finite_field::Pluecker_21(int *x4, int *y4)
{
	return Pluecker_ij(1, 0, x4, y4);
}

int finite_field::Pluecker_13(int *x4, int *y4)
{
	return Pluecker_ij(0, 2, x4, y4);
}

int finite_field::Pluecker_31(int *x4, int *y4)
{
	return Pluecker_ij(2, 0, x4, y4);
}

int finite_field::Pluecker_14(int *x4, int *y4)
{
	return Pluecker_ij(0, 3, x4, y4);
}

int finite_field::Pluecker_41(int *x4, int *y4)
{
	return Pluecker_ij(3, 0, x4, y4);
}

int finite_field::Pluecker_23(int *x4, int *y4)
{
	return Pluecker_ij(1, 2, x4, y4);
}

int finite_field::Pluecker_32(int *x4, int *y4)
{
	return Pluecker_ij(2, 1, x4, y4);
}

int finite_field::Pluecker_24(int *x4, int *y4)
{
	return Pluecker_ij(1, 3, x4, y4);
}

int finite_field::Pluecker_42(int *x4, int *y4)
{
	return Pluecker_ij(3, 1, x4, y4);
}

int finite_field::Pluecker_34(int *x4, int *y4)
{
	return Pluecker_ij(2, 3, x4, y4);
}

int finite_field::Pluecker_43(int *x4, int *y4)
{
	return Pluecker_ij(3, 2, x4, y4);
}

int finite_field::Pluecker_ij(int i, int j, int *x4, int *y4)
{
	return add(mult(x4[i], y4[j]), negate(mult(x4[j], y4[i])));
}


int finite_field::evaluate_symplectic_form(int len, int *x, int *y)
{
	int i, n, c;

	if (ODD(len)) {
		cout << "finite_field::evaluate_symplectic_form len must be even"
				<< endl;
		cout << "len=" << len << endl;
		exit(1);
		}
	c = 0;
	n = len >> 1;
	for (i = 0; i < n; i++) {
		c = add(c, add(
				mult(x[2 * i + 0], y[2 * i + 1]),
				negate(mult(x[2 * i + 1], y[2 * i + 0]))
				));
		}
	return c;
}

int finite_field::evaluate_quadratic_form_x0x3mx1x2(int *x)
{
	int a;

	a = add(mult(x[0], x[3]), negate(mult(x[1], x[2])));
	return a;
}

int finite_field::is_totally_isotropic_wrt_symplectic_form(
		int k, int n, int *Basis)
{
	int i, j;

	for (i = 0; i < k; i++) {
		for (j = i + 1; j < k; j++) {
			if (evaluate_symplectic_form(n, Basis + i * n, Basis + j * n)) {
				return FALSE;
				}
			}
		}
	return TRUE;
}

int finite_field::evaluate_monomial(int *monomial,
		int *variables, int nb_vars)
{
	int i, j, a, b, x;

	a = 1;
	for (i = 0; i < nb_vars; i++) {
		b = monomial[i];
		x = variables[i];
		for (j = 0; j < b; j++) {
			a = mult(a, x);
			}
		}
	return a;
}

void finite_field::projective_point_unrank(int n, int *v, int rk)
{
	PG_element_unrank_modified(v, 1 /* stride */,
			n + 1 /* len */, rk);
}

int finite_field::projective_point_rank(int n, int *v)
{
	int rk;

	PG_element_rank_modified(v, 1 /* stride */, n + 1, rk);
	return rk;
}

void finite_field::create_BLT_point(
		int *v5, int a, int b, int c, int verbose_level)
// creates the point (-b/2,-c,a,-(b^2/4-ac),1)
// check if it satisfies x_0^2 + x_1x_2 + x_3x_4:
// b^2/4 + (-c)*a + -(b^2/4-ac)
// = b^2/4 -ac -b^2/4 + ac = 0
{
	int f_v = (verbose_level >= 1);
	int v0, v1, v2, v3, v4;
	int half, four, quarter, minus_one;

	if (f_v) {
		cout << "finite_field::create_BLT_point" << endl;
		}
	four = 4 % p;
	half = inverse(2);
	quarter = inverse(four);
	minus_one = negate(1);
	if (f_v) {
		cout << "finite_field::create_BLT_point "
				"four=" << four << endl;
		cout << "finite_field::create_BLT_point "
				"half=" << half << endl;
		cout << "finite_field::create_BLT_point "
				"quarter=" << quarter << endl;
		cout << "finite_field::create_BLT_point "
				"minus_one=" << minus_one << endl;
		}

	v0 = mult(minus_one, mult(b, half));
	v1 = mult(minus_one, c);
	v2 = a;
	v3 = mult(minus_one, add(
			mult(mult(b, b), quarter), negate(mult(a, c))));
	v4 = 1;
	int_vec_init5(v5, v0, v1, v2, v3, v4);
	if (f_v) {
		cout << "finite_field::create_BLT_point done" << endl;
		}

}

void finite_field::Segre_hyperoval(
		int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N = q + 2;
	int i, t, a, t6;
	int *Mtx;

	if (f_v) {
		cout << "finite_field::Segre_hyperoval q=" << q << endl;
		}
	if (EVEN(e)) {
		cout << "finite_field::Segre_hyperoval needs e odd" << endl;
		exit(1);
		}

	nb_pts = N;

	Pts = NEW_int(N);
	Mtx = NEW_int(N * 3);
	int_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		t6 = power(t, 6);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = t6;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_int(Mtx);
	if (f_v) {
		cout << "finite_field::Segre_hyperoval "
				"q=" << q << " done" << endl;
		}
}


void finite_field::GlynnI_hyperoval(
		int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N = q + 2;
	int i, t, te, a;
	int sigma, gamma = 0, Sigma, /*Gamma,*/ exponent;
	int *Mtx;

	if (f_v) {
		cout << "finite_field::GlynnI_hyperoval q=" << q << endl;
		}
	if (EVEN(e)) {
		cout << "finite_field::GlynnI_hyperoval needs e odd" << endl;
		exit(1);
		}

	sigma = e - 1;
	for (i = 0; i < e; i++) {
		if (((i * i) % e) == sigma) {
			gamma = i;
			break;
			}
		}
	if (i == e) {
		cout << "finite_field::GlynnI_hyperoval "
				"did not find gamma" << endl;
		exit(1);
		}

	cout << "finite_field::GlynnI_hyperoval sigma = " << sigma
			<< " gamma = " << gamma << endl;
	//Gamma = i_power_j(2, gamma);
	Sigma = i_power_j(2, sigma);

	exponent = 3 * Sigma + 4;

	nb_pts = N;

	Pts = NEW_int(N);
	Mtx = NEW_int(N * 3);
	int_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		te = power(t, exponent);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = te;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_int(Mtx);
	if (f_v) {
		cout << "finite_field::GlynnI_hyperoval "
				"q=" << q << " done" << endl;
		}
}

void finite_field::GlynnII_hyperoval(
		int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N = q + 2;
	int i, t, te, a;
	int sigma, gamma = 0, Sigma, Gamma, exponent;
	int *Mtx;

	if (f_v) {
		cout << "finite_field::GlynnII_hyperoval q=" << q << endl;
		}
	if (EVEN(e)) {
		cout << "finite_field::GlynnII_hyperoval "
				"needs e odd" << endl;
		exit(1);
		}

	sigma = e - 1;
	for (i = 0; i < e; i++) {
		if (((i * i) % e) == sigma) {
			gamma = i;
			break;
			}
		}
	if (i == e) {
		cout << "finite_field::GlynnII_hyperoval "
				"did not find gamma" << endl;
		exit(1);
		}

	cout << "finite_field::GlynnII_hyperoval "
			"sigma = " << sigma << " gamma = " << i << endl;
	Gamma = i_power_j(2, gamma);
	Sigma = i_power_j(2, sigma);

	exponent = Sigma + Gamma;

	nb_pts = N;

	Pts = NEW_int(N);
	Mtx = NEW_int(N * 3);
	int_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		te = power(t, exponent);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = te;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_int(Mtx);
	if (f_v) {
		cout << "finite_field::GlynnII_hyperoval "
				"q=" << q << " done" << endl;
		}
}


void finite_field::Subiaco_oval(
		int *&Pts, int &nb_pts, int f_short, int verbose_level)
// following Payne, Penttila, Pinneri:
// Isomorphisms Between Subiaco q-Clan Geometries,
// Bull. Belg. Math. Soc. 2 (1995) 197-222.
// formula (53)
{
	int f_v = (verbose_level >= 1);
	int N = q + 1;
	int i, t, a, b, h, k, top, bottom;
	int omega, omega2;
	int t2, t3, t4, sqrt_t;
	int *Mtx;

	if (f_v) {
		cout << "finite_field::Subiaco_oval "
				"q=" << q << " f_short=" << f_short << endl;
		}

	nb_pts = N;
	k = (q - 1) / 3;
	if (k * 3 != q - 1) {
		cout << "Subiaco_oval k * 3 != q - 1" << endl;
		exit(1);
		}
	omega = power(alpha, k);
	omega2 = mult(omega, omega);
	if (add3(omega2, omega, 1) != 0) {
		cout << "finite_field::Subiaco_oval "
				"add3(omega2, omega, 1) != 0" << endl;
		exit(1);
		}
	Pts = NEW_int(N);
	Mtx = NEW_int(N * 3);
	int_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		t2 = mult(t, t);
		t3 = mult(t2, t);
		t4 = mult(t2, t2);
		sqrt_t = frobenius_power(t, e - 1);
		if (mult(sqrt_t, sqrt_t) != t) {
			cout << "finite_field::Subiaco_oval "
					"mult(sqrt_t, sqrt_t) != t" << endl;
			exit(1);
			}
		bottom = add3(t4, mult(omega2, t2), 1);
		if (f_short) {
			top = mult(omega2, add(t4, t));
			}
		else {
			top = add3(t3, t2, mult(omega2, t));
			}
		if (FALSE) {
			cout << "t=" << t << " top=" << top
					<< " bottom=" << bottom << endl;
			}
		a = mult(top, inverse(bottom));
		if (f_short) {
			b = sqrt_t;
			}
		else {
			b = mult(omega, sqrt_t);
			}
		h = add(a, b);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = h;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_int(Mtx);
	if (f_v) {
		cout << "finite_field::Subiaco_oval "
				"q=" << q << " done" << endl;
		}
}




void finite_field::Subiaco_hyperoval(
		int *&Pts, int &nb_pts, int verbose_level)
// email 12/27/2014
//The o-polynomial of the Subiaco hyperoval is

//t^{1/2}+(d^2t^4 + d^2(1+d+d^2)t^3
// + d^2(1+d+d^2)t^2 + d^2t)/(t^4+d^2t^2+1)

//where d has absolute trace 1.

//Best,
//Tim

//absolute trace of 1/d is 1 not d...

{
	int f_v = (verbose_level >= 1);
	int N = q + 2;
	int i, t, d, dv, d2, one_d_d2, a, h;
	int t2, t3, t4, sqrt_t;
	int top1, top2, top3, top4, top, bottom;
	int *Mtx;

	if (f_v) {
		cout << "finite_field::Subiaco_hyperoval q=" << q << endl;
		}

	nb_pts = N;
	for (d = 1; d < q; d++) {
		dv = inverse(d);
		if (absolute_trace(dv) == 1) {
			break;
			}
		}
	if (d == q) {
		cout << "finite_field::Subiaco_hyperoval "
				"cannot find element d" << endl;
		exit(1);
		}
	d2 = mult(d, d);
	one_d_d2 = add3(1, d, d2);

	Pts = NEW_int(N);
	Mtx = NEW_int(N * 3);
	int_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		t2 = mult(t, t);
		t3 = mult(t2, t);
		t4 = mult(t2, t2);
		sqrt_t = frobenius_power(t, e - 1);
		if (mult(sqrt_t, sqrt_t) != t) {
			cout << "finite_field::Subiaco_hyperoval "
					"mult(sqrt_t, sqrt_t) != t" << endl;
			exit(1);
			}


		bottom = add3(t4, mult(d2, t2), 1);

		//t^{1/2}+(d^2t^4 + d^2(1+d+d^2)t^3 +
		// d^2(1+d+d^2)t^2 + d^2t)/(t^4+d^2t^2+1)

		top1 = mult(d2,t4);
		top2 = mult3(d2, one_d_d2, t3);
		top3 = mult3(d2, one_d_d2, t2);
		top4 = mult(d2, t);
		top = add4(top1, top2, top3, top4);

		if (f_v) {
			cout << "t=" << t << " top=" << top
					<< " bottom=" << bottom << endl;
			}
		a = mult(top, inverse(bottom));
		h = add(a, sqrt_t);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = h;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_int(Mtx);
	if (f_v) {
		cout << "finite_field::Subiaco_hyperoval "
				"q=" << q << " done" << endl;
		}
}




// From Bill Cherowitzo's web page:
// In 1991, O'Keefe and Penttila [OKPe92]
// by means of a detailed investigation
// of the divisibility properties of the orders
// of automorphism groups
// of hypothetical hyperovals in this plane,
// discovered a n e w hyperoval.
// Its o-polynomial is given by:

//f(x) = x4 + x16 + x28 + beta*11(x6 + x10 + x14 + x18 + x22 + x26)
// + beta*20(x8 + x20) + beta*6(x12 + x24),
//where ß is a primitive root of GF(32) satisfying beta^5 = beta^2 + 1.
//The full automorphism group of this hyperoval has order 3.

int finite_field::OKeefe_Penttila_32(int t)
// needs the field generated by beta with beta^5 = beta^2+1
// From Bill Cherowitzo's hyperoval page
{
	int *t_powers;
	int a, b, c, d, e, beta6, beta11, beta20;

	t_powers = NEW_int(31);

	power_table(t, t_powers, 31);
	a = add3(t_powers[4], t_powers[16], t_powers[28]);
	b = add6(t_powers[6], t_powers[10], t_powers[14],
			t_powers[18], t_powers[22], t_powers[26]);
	c = add(t_powers[8], t_powers[20]);
	d = add(t_powers[12], t_powers[24]);

	beta6 = power(2, 6);
	beta11 = power(2, 11);
	beta20 = power(2, 20);

	b = mult(b, beta11);
	c = mult(c, beta20);
	d = mult(d, beta6);

	e = add4(a, b, c, d);

	FREE_int(t_powers);
	return e;
}



int finite_field::Subiaco64_1(int t)
// needs the field generated by beta with beta^6 = beta+1
// The first one from Bill Cherowitzo's hyperoval page
{
	int *t_powers;
	int a, b, c, d, beta21, beta42;

	t_powers = NEW_int(65);

	power_table(t, t_powers, 65);
	a = add6(t_powers[8], t_powers[12], t_powers[20],
			t_powers[22], t_powers[42], t_powers[52]);
	b = add6(t_powers[4], t_powers[10], t_powers[14],
			t_powers[16], t_powers[30], t_powers[38]);
	c = add6(t_powers[44], t_powers[48], t_powers[54],
			t_powers[56], t_powers[58], t_powers[60]);
	b = add3(b, c, t_powers[62]);
	c = add7(t_powers[2], t_powers[6], t_powers[26],
			t_powers[28], t_powers[32], t_powers[36], t_powers[40]);
	beta21 = power(2, 21);
	beta42 = mult(beta21, beta21);
	d = add3(a, mult(beta21, b), mult(beta42, c));
	FREE_int(t_powers);
	return d;
}

int finite_field::Subiaco64_2(int t)
// needs the field generated by beta with beta^6 = beta+1
// The second one from Bill Cherowitzo's hyperoval page
{
	int *t_powers;
	int a, b, c, d, beta21, beta42;

	t_powers = NEW_int(65);

	power_table(t, t_powers, 65);
	a = add3(t_powers[24], t_powers[30], t_powers[62]);
	b = add6(t_powers[4], t_powers[8], t_powers[10],
			t_powers[14], t_powers[16], t_powers[34]);
	c = add6(t_powers[38], t_powers[40], t_powers[44],
			t_powers[46], t_powers[52], t_powers[54]);
	b = add4(b, c, t_powers[58], t_powers[60]);
	c = add5(t_powers[6], t_powers[12], t_powers[18],
			t_powers[20], t_powers[26]);
	d = add5(t_powers[32], t_powers[36], t_powers[42],
			t_powers[48], t_powers[50]);
	c = add(c, d);
	beta21 = power(2, 21);
	beta42 = mult(beta21, beta21);
	d = add3(a, mult(beta21, b), mult(beta42, c));
	FREE_int(t_powers);
	return d;
}

int finite_field::Adelaide64(int t)
// needs the field generated by beta with beta^6 = beta+1
{
	int *t_powers;
	int a, b, c, d, beta21, beta42;

	t_powers = NEW_int(65);

	power_table(t, t_powers, 65);
	a = add7(t_powers[4], t_powers[8], t_powers[14], t_powers[34],
			t_powers[42], t_powers[48], t_powers[62]);
	b = add8(t_powers[6], t_powers[16], t_powers[26], t_powers[28],
			t_powers[30], t_powers[32], t_powers[40], t_powers[58]);
	c = add8(t_powers[10], t_powers[18], t_powers[24], t_powers[36],
			t_powers[44], t_powers[50], t_powers[52], t_powers[60]);
	beta21 = power(2, 21);
	beta42 = mult(beta21, beta21);
	d = add3(a, mult(beta21, b), mult(beta42, c));
	FREE_int(t_powers);
	return d;
}



void finite_field::LunelliSce(int *pts18, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//const char *override_poly = "19";
	//finite_field F;
	//int n = 3;
	//int q = 16;
	int v[3];
	//int w[3];

	if (f_v) {
		cout << "finite_field::LunelliSce" << endl;
		}
	//F.init(q), verbose_level - 2);
	//F.init_override_polynomial(q, override_poly, verbose_level);

#if 0
	int cubic1[100];
	int cubic1_size = 0;
	int cubic2[100];
	int cubic2_size = 0;
	int hoval[100];
	int hoval_size = 0;
#endif

	int a, b, i, sz, N;

	if (q != 16) {
		cout << "finite_field::LunelliSce "
				"field order must be 16" << endl;
		exit(1);
		}
	N = nb_PG_elements(2, 16);
	sz = 0;
	for (i = 0; i < N; i++) {
		PG_element_unrank_modified(v, 1, 3, i);
		//cout << "i=" << i << " v=";
		//int_vec_print(cout, v, 3);
		//cout << endl;

		a = LunelliSce_evaluate_cubic1(v);
		b = LunelliSce_evaluate_cubic2(v);

		// form the symmetric difference of the two cubics:
		if ((a == 0 && b) || (b == 0 && a)) {
			pts18[sz++] = i;
			}
		}
	if (sz != 18) {
		cout << "sz != 18" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "the size of the LinelliSce hyperoval is " << sz << endl;
		cout << "the LinelliSce hyperoval is:" << endl;
		int_vec_print(cout, pts18, sz);
		cout << endl;
		}

#if 0
	cout << "the size of cubic1 is " << cubic1_size << endl;
	cout << "the cubic1 is:" << endl;
	int_vec_print(cout, cubic1, cubic1_size);
	cout << endl;
	cout << "the size of cubic2 is " << cubic2_size << endl;
	cout << "the cubic2 is:" << endl;
	int_vec_print(cout, cubic2, cubic2_size);
	cout << endl;
#endif

}

int finite_field::LunelliSce_evaluate_cubic1(int *v)
// computes X^3 + Y^3 + Z^3 + \eta^3 XYZ
{
	int a, b, c, d, e, eta3;

	eta3 = power(2, 3);
	//eta12 = power(2, 12);
	a = power(v[0], 3);
	b = power(v[1], 3);
	c = power(v[2], 3);
	d = product4(eta3, v[0], v[1], v[2]);
	e = add4(a, b, c, d);
	return e;
}

int finite_field::LunelliSce_evaluate_cubic2(int *v)
// computes X^3 + Y^3 + Z^3 + \eta^{12} XYZ
{
	int a, b, c, d, e, eta12;

	//eta3 = power(2, 3);
	eta12 = power(2, 12);
	a = power(v[0], 3);
	b = power(v[1], 3);
	c = power(v[2], 3);
	d = product4(eta12, v[0], v[1], v[2]);
	e = add4(a, b, c, d);
	return e;
}


void finite_field::O4_isomorphism_4to2(
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
		cout << "finite_field::O4_isomorphism_4to2" << endl;
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
	O4_grid_coordinates_unrank(P[0], P[1], P[2], P[3],
			0, 0, verbose_level);
	if (f_vv) {
		cout << "grid point (0,0) = ";
		int_vec_print(cout, P, 4);
		cout << endl;
		}
	O4_grid_coordinates_unrank(Q[0], Q[1], Q[2], Q[3],
			1, 0, verbose_level);
	if (f_vv) {
		cout << "grid point (1,0) = ";
		int_vec_print(cout, Q, 4);
		cout << endl;
		}
	mult_vector_from_the_left(P, B, R, 4, 4);
	mult_vector_from_the_left(Q, B, S, 4, 4);
	O4_grid_coordinates_rank(R[0], R[1], R[2], R[3],
			Rx, Ry, verbose_level);
	O4_grid_coordinates_rank(S[0], S[1], S[2], S[3],
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
				fv = inverse(f);
				c = mult(fv, b33);
				d = negate(mult(fv, b13));
				}
			else {
				ev = inverse(e);
				c = negate(mult(ev, b23));
				d = negate(mult(ev, b43));
				}
			}
		else {
			a = 1;
			e = b22;
			g = b24;
			f = negate(b32);
			h = negate(b34);
			if (e == 0) {
				fv = inverse(f);
				b = mult(fv, b12);
				c = mult(fv, b33);
				d = negate(mult(fv, b13));
				}
			else {
				ev = inverse(e);
				b = mult(ev, b42);
				c = negate(mult(ev, b23));
				d = negate(mult(ev, b43));
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
			e = negate(b32);
			g = negate(b34);
			if (e == 0) {
				fv = inverse(f);
				c = negate(mult(fv, b43));
				d = negate(mult(fv, b13));
				}
			else {
				ev = inverse(e);
				c = negate(mult(ev, b23));
				d = mult(ev, b33);
				}
			}
		else {
			a = 1;
			e = b22;
			g = b24;
			f = b42;
			h = b44;
			if (e == 0) {
				fv = inverse(f);
				b = mult(fv, b12);
				c = negate(mult(fv, b43));
				d = negate(mult(fv, b13));
				}
			else {
				ev = inverse(e);
				b = negate(mult(ev, b32));
				c = negate(mult(ev, b23));
				d = mult(ev, b33);
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
		print_integer_matrix_width(cout, At, 2, 2, 2, log10_of_q);
		cout << "As:" << endl;
		print_integer_matrix_width(cout, As, 2, 2, 2, log10_of_q);
		}

}

void finite_field::O4_isomorphism_2to4(
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
		B[0 * 4 + 0] = mult(h, d);
		B[0 * 4 + 1] = mult(f, b);
		B[0 * 4 + 2] = negate(mult(f, d));
		B[0 * 4 + 3] = mult(h, b);
		B[1 * 4 + 0] = mult(g, c);
		B[1 * 4 + 1] = mult(e, a);
		B[1 * 4 + 2] = negate(mult(e, c));
		B[1 * 4 + 3] = mult(g, a);
		B[2 * 4 + 0] = negate(mult(h, c));
		B[2 * 4 + 1] = negate(mult(f, a));
		B[2 * 4 + 2] = mult(f, c);
		B[2 * 4 + 3] = negate(mult(h, a));
		B[3 * 4 + 0] = mult(g, d);
		B[3 * 4 + 1] = mult(e, b);
		B[3 * 4 + 2] = negate(mult(e, d));
		B[3 * 4 + 3] = mult(g, b);
		}
	else {
		B[0 * 4 + 0] = mult(h, d);
		B[0 * 4 + 1] = mult(f, b);
		B[0 * 4 + 2] = negate(mult(f, d));
		B[0 * 4 + 3] = mult(h, b);
		B[1 * 4 + 0] = mult(g, c);
		B[1 * 4 + 1] = mult(e, a);
		B[1 * 4 + 2] = negate(mult(e, c));
		B[1 * 4 + 3] = mult(g, a);
		B[2 * 4 + 0] = negate(mult(g, d));
		B[2 * 4 + 1] = negate(mult(e, b));
		B[2 * 4 + 2] = mult(e, d);
		B[2 * 4 + 3] = negate(mult(g, b));
		B[3 * 4 + 0] = mult(h, c);
		B[3 * 4 + 1] = mult(f, a);
		B[3 * 4 + 2] = negate(mult(f, c));
		B[3 * 4 + 3] = mult(h, a);
		}
}

void finite_field::O4_grid_coordinates_rank(
		int x1, int x2, int x3, int x4, int &grid_x, int &grid_y,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, c, d, av, e;
	int v[2], w[2];

	a = x1;
	b = x4;
	c = negate(x3);
	d = x2;

	if (a) {
		if (a != 1) {
			av = inverse(a);
			b = mult(b, av);
			c = mult(c, av);
			d = mult(d, av);
			}
		v[0] = 1;
		w[0] = 1;
		v[1] = c;
		w[1] = b;
		e = mult(c, b);
		if (e != d) {
			cout << "finite_field::O4_grid_coordinates_rank "
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
	PG_element_normalize_from_front(v, 1, 2);
	PG_element_normalize_from_front(w, 1, 2);
	if (f_v) {
		int_vec_print(cout, v, 2);
		int_vec_print(cout, w, 2);
		cout << endl;
		}

	PG_element_rank_modified(v, 1, 2, grid_x);
	PG_element_rank_modified(w, 1, 2, grid_y);
}

void finite_field::O4_grid_coordinates_unrank(
		int &x1, int &x2, int &x3, int &x4,
		int grid_x, int grid_y,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, c, d;
	int v[2], w[2];

	PG_element_unrank_modified(v, 1, 2, grid_x);
	PG_element_unrank_modified(w, 1, 2, grid_y);
	PG_element_normalize_from_front(v, 1, 2);
	PG_element_normalize_from_front(w, 1, 2);
	if (f_v) {
		int_vec_print(cout, v, 2);
		int_vec_print(cout, w, 2);
		cout << endl;
		}

	a = mult(v[0], w[0]);
	b = mult(v[0], w[1]);
	c = mult(v[1], w[0]);
	d = mult(v[1], w[1]);
	x1 = a;
	x2 = d;
	x3 = negate(c);
	x4 = b;
}

void finite_field::O4_find_tangent_plane(
		int pt_x1, int pt_x2, int pt_x3, int pt_x4,
		int *tangent_plane,
		int verbose_level)
{
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


	cout << "O4_find_tangent_plane pt_x1=" << pt_x1
		<< " pt_x2=" << pt_x2
		<< " pt_x3=" << pt_x3
		<< " pt_x4=" << pt_x4 << endl;
	size = q + 1;
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
			O4_grid_coordinates_unrank(x1, x2, x3, x4, x, y, 0);

			//cout << "x1=" << x1 << " x2=" << x2
			//<< " x3=" << x3 << " x4=" << x4 << endl;




			for (k = 0; k < size; k++) {
				PG_element_unrank_modified(vec2, 1, 2, k);
				y1 = add(mult(pt_x1, vec2[0]), mult(x1, vec2[1]));
				y2 = add(mult(pt_x2, vec2[0]), mult(x2, vec2[1]));
				y3 = add(mult(pt_x3, vec2[0]), mult(x3, vec2[1]));
				y4 = add(mult(pt_x4, vec2[0]), mult(x4, vec2[1]));
				det = add(mult(y1, y2), mult(y3, y4));
				if (det != 0) {
					continue;
					}
				O4_grid_coordinates_rank(y1, y2, y3, y4, xx, yy, 0);
				zz = xx * size + yy;
				if (zz == z)
					continue;
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

				rk = Gauss_int(C, f_special, f_complete, base_cols,
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
	cout << "nb_secants=" << nb_secants << endl;
	int_vec_print(cout, secants1, nb_secants);
	cout << endl;
	int_vec_print(cout, secants2, nb_secants);
	cout << endl;
	h = 0;
	for (zz = 0; zz < size * size; zz++) {
		if (secants1[h] > zz) {
			complement[nb_complement++] = zz;
			}
		else {
			h++;
			}
		}
	cout << "complement = tangents:" << endl;
	int_vec_print(cout, complement, nb_complement);
	cout << endl;

	int *T;
	T = NEW_int(4 * nb_complement);

	for (h = 0; h < nb_complement; h++) {
		z = complement[h];
		x = z / size;
		y = z % size;
		cout << setw(3) << h << " : " << setw(4) << z
				<< " : " << x << "," << y << " : ";
		O4_grid_coordinates_unrank(y1, y2, y3, y4,
				x, y, verbose_level);
		cout << "y1=" << y1 << " y2=" << y2
				<< " y3=" << y3 << " y4=" << y4 << endl;
		T[h * 4 + 0] = y1;
		T[h * 4 + 1] = y2;
		T[h * 4 + 2] = y3;
		T[h * 4 + 3] = y4;
		}


	rk = Gauss_int(T, f_special, f_complete, base_cols,
		f_P, NULL, nb_complement, 4, 4, 0);
	cout << "the rank of the tangent space is " << rk << endl;
	cout << "basis:" << endl;
	print_integer_matrix_width(cout, T, rk, 4, 4, log10_of_q);

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

void finite_field::oval_polynomial(
	int *S, unipoly_domain &D, unipoly_object &poly,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, v[3], x; //, y;
	int *map;

	if (f_v) {
		cout << "finite_field::oval_polynomial" << endl;
		}
	map = NEW_int(q);
	for (i = 0; i < q; i++) {
		PG_element_unrank_modified(
				v, 1 /* stride */, 3 /* len */, S[2 + i]);
		if (v[2] != 1) {
			cout << "finite_field::oval_polynomial "
					"not an affine point" << endl;
			exit(1);
			}
		x = v[0];
		//y = v[1];
		//cout << "map[" << i << "] = " << xx << endl;
		map[i] = x;
		}
	if (f_v) {
		cout << "the map" << endl;
		for (i = 0; i < q; i++) {
			cout << map[i] << " ";
			}
		cout << endl;
		}

	D.create_Dickson_polynomial(poly, map);

	FREE_int(map);
	if (f_v) {
		cout << "finite_field::oval_polynomial done" << endl;
		}
}


void finite_field::all_PG_elements_in_subspace(
		int *genma, int k, int n, int *&point_list, int &nb_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int *message;
	int *word;
	int i, j;

	if (f_v) {
		cout << "finite_field::all_PG_elements_in_subspace" << endl;
		}
	message = NEW_int(k);
	word = NEW_int(n);
	nb_points = generalized_binomial(k, 1, q);
	point_list = NEW_int(nb_points);

	for (i = 0; i < nb_points; i++) {
		PG_element_unrank_modified(message, 1, k, i);
		if (f_vv) {
			cout << "message " << i << " / " << nb_points << " is ";
			int_vec_print(cout, message, k);
			cout << endl;
			}
		mult_vector_from_the_left(message, genma, word, k, n);
		if (f_vv) {
			cout << "yields word ";
			int_vec_print(cout, word, n);
			cout << endl;
			}
		PG_element_rank_modified(word, 1, n, j);
		if (f_vv) {
			cout << "which has rank " << j << endl;
			}
		point_list[i] = j;
		}

	FREE_int(message);
	FREE_int(word);
	if (f_v) {
		cout << "finite_field::all_PG_elements_in_subspace "
				"done" << endl;
		}
}

void finite_field::all_PG_elements_in_subspace_array_is_given(
		int *genma, int k, int n, int *point_list, int &nb_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int *message;
	int *word;
	int i, j;

	if (f_v) {
		cout << "finite_field::all_PG_elements_in_"
				"subspace_array_is_given" << endl;
		}
	message = NEW_int(k);
	word = NEW_int(n);
	nb_points = generalized_binomial(k, 1, q);
	//point_list = NEW_int(nb_points);

	for (i = 0; i < nb_points; i++) {
		PG_element_unrank_modified(message, 1, k, i);
		if (f_vv) {
			cout << "message " << i << " / " << nb_points << " is ";
			int_vec_print(cout, message, k);
			cout << endl;
			}
		mult_vector_from_the_left(message, genma, word, k, n);
		if (f_vv) {
			cout << "yields word ";
			int_vec_print(cout, word, n);
			cout << endl;
			}
		PG_element_rank_modified(word, 1, n, j);
		if (f_vv) {
			cout << "which has rank " << j << endl;
			}
		point_list[i] = j;
		}

	FREE_int(message);
	FREE_int(word);
	if (f_v) {
		cout << "finite_field::all_PG_elements_in_"
				"subspace_array_is_given "
				"done" << endl;
		}
}

void finite_field::display_all_PG_elements(int n)
{
	int *v = NEW_int(n + 1);
	int l = nb_PG_elements(n, q);
	int i, j, a;

	for (i = 0; i < l; i++) {
		PG_element_unrank_modified(v, 1, n + 1, i);
		cout << i << " : ";
		for (j = 0; j < n + 1; j++) {
			cout << v[j] << " ";
			}
		PG_element_rank_modified(v, 1, n + 1, a);
		cout << " : " << a << endl;
		}
	FREE_int(v);
}

void finite_field::display_all_PG_elements_not_in_subspace(int n, int m)
{
	int *v = NEW_int(n + 1);
	int l = nb_PG_elements_not_in_subspace(n, m, q);
	int i, j, a;

	for (i = 0; i < l; i++) {
		PG_element_unrank_modified_not_in_subspace(v, 1, n + 1, m, i);
		cout << i << " : ";
		for (j = 0; j < n + 1; j++) {
			cout << v[j] << " ";
			}
		PG_element_rank_modified_not_in_subspace(v, 1, n + 1, m, a);
		cout << " : " << a << endl;
		}
	FREE_int(v);
}

void finite_field::display_all_AG_elements(int n)
{
	int *v = NEW_int(n);
	int l = nb_AG_elements(n, q);
	int i, j;

	for (i = 0; i < l; i++) {
		AG_element_unrank(q, v, 1, n + 1, i);
		cout << i << " : ";
		for (j = 0; j < n; j++) {
			cout << v[j] << " ";
			}
		cout << endl;
		}
	FREE_int(v);
}



// #############################################################################
// globals:
// #############################################################################

int nb_PG_elements(int n, int q)
// $\frac{q^{n+1} - 1}{q-1} = \sum_{i=0}^{n} q^i $
{
	int qhl, l, deg;

	l = 0;
	qhl = 1;
	deg = 0;
	while (l <= n) {
		deg += qhl;
		qhl *= q;
		l++;
		}
	return deg;
}

int nb_PG_elements_not_in_subspace(int n, int m, int q)
// |PG_n(q)| - |PG_m(q)|
{
	int a, b;

	a = nb_PG_elements(n, q);
	b = nb_PG_elements(m, q);
	return a - b;
}

int nb_AG_elements(int n, int q)
// $q^n$
{
	return i_power_j(q, n);
}

void PG_element_apply_frobenius(int n,
		finite_field &GFq, int *v, int f)
{
	int i;

	for (i = 0; i < n; i++) {
		v[i] = GFq.frobenius_power(v[i], f);
		}
}

void AG_element_rank(int q, int *v, int stride, int len, int &a)
{
	int i;
	
	if (len <= 0) {
		cout << "AG_element_rank len <= 0" << endl;
		exit(1);
		}
	a = 0;
	for (i = len - 1; i >= 0; i--) {
		a += v[i * stride];
		if (i > 0) {
			a *= q;
			}
		}
}

void AG_element_unrank(int q, int *v, int stride, int len, int a)
{
	int i, b;
	
#if 1
	if (len <= 0) {
		cout << "AG_element_unrank len <= 0" << endl;
		exit(1);
		}
#endif
	for (i = 0; i < len; i++) {
		b = a % q;
		v[i * stride] = b;
		a /= q;
		}
}

void AG_element_rank_longinteger(int q,
		int *v, int stride, int len, longinteger_object &a)
{
	longinteger_domain D;
	longinteger_object Q, a1;
	int i;
	
	if (len <= 0) {
		cout << "AG_element_rank_longinteger len <= 0" << endl;
		exit(1);
		}
	a.create(0);
	Q.create(q);
	for (i = len - 1; i >= 0; i--) {
		a.add_int(v[i * stride]);
		//cout << "AG_element_rank_longinteger
		//after add_int " << a << endl;
		if (i > 0) {
			D.mult(a, Q, a1);
			a.swap_with(a1);
			//cout << "AG_element_rank_longinteger
			//after mult " << a << endl;
			}
		}
}

void AG_element_unrank_longinteger(int q,
		int *v, int stride, int len, longinteger_object &a)
{
	int i, r;
	longinteger_domain D;
	longinteger_object Q, a1;
	
	if (len <= 0) {
		cout << "AG_element_unrank_longinteger len <= 0" << endl;
		exit(1);
		}
	for (i = 0; i < len; i++) {
		D.integral_division_by_int(a, q, a1, r);
		//r = a % q;
		v[i * stride] = r;
		//a /= q;
		a.swap_with(a1);
		}
}


int PG_element_modified_is_in_subspace(int n, int m, int *v)
{
	int j;
	
	for (j = m + 1; j < n + 1; j++) {
		if (v[j]) {
			return FALSE;
			}
		}
	return TRUE;
}

void PG_element_modified_not_in_subspace_perm(int n, int m, 
	finite_field &GFq,
	int *orbit, int *orbit_inv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v = NEW_int(n + 1);
	int l = nb_PG_elements(n, GFq.q);
	int ll = nb_PG_elements_not_in_subspace(n, m, GFq.q);
	int i, j1 = 0, j2 = ll, f_in, j;
	
	for (i = 0; i < l; i++) {
		GFq.PG_element_unrank_modified(v, 1, n + 1, i);
		f_in = PG_element_modified_is_in_subspace(n, m, v);
		if (f_v) {
			cout << i << " : ";
			for (j = 0; j < n + 1; j++) {
				cout << v[j] << " ";
				}
			}
		if (f_in) {
			if (f_v) {
				cout << " is in the subspace" << endl;
				}
			orbit[j2] = i;
			orbit_inv[i] = j2;
			j2++;
			}
		else {
			if (f_v) {
				cout << " is not in the subspace" << endl;
				}
			orbit[j1] = i;
			orbit_inv[i] = j1;
			j1++;
			}
		}
	if (j1 != ll) {
		cout << "j1 != ll" << endl;
		exit(1);
		}
	if (j2 != l) {
		cout << "j2 != l" << endl;
		exit(1);
		}
	FREE_int(v);
}


void test_PG(int n, int q)
{
	finite_field F;
	int m;
	int verbose_level = 1;
	
	F.init(q, verbose_level);
	
	cout << "all elements of PG_" << n << "(" << q << ")" << endl;
	F.display_all_PG_elements(n);
	
	for (m = 0; m < n; m++) {
		cout << "all elements of PG_" << n << "(" << q << "), "
			"not in a subspace of dimension " << m << endl;
		F.display_all_PG_elements_not_in_subspace(n, m);
		}
	
}


void line_through_two_points(finite_field &GFq,
		int len, int pt1, int pt2, int *line)
{
	int v1[100], v2[100], v3[100], alpha, a, ii;
	
	if (len > 100) {
		cout << "line_through_two_points() len >= 100" << endl;
		exit(1);
		}
	GFq.PG_element_unrank_modified(v1, 1 /* stride */, len, pt1);
	GFq.PG_element_unrank_modified(v2, 1 /* stride */, len, pt2);
	line[0] = pt1;
	line[1] = pt2;
	for (alpha = 1; alpha < GFq.q; alpha++) {
		for (ii = 0; ii < len; ii++) {
			a = GFq.mult(v1[ii], alpha);
			v3[ii] = GFq.add(a, v2[ii]);
			}
		GFq.PG_element_normalize(v3, 1 /* stride */, len);
		GFq.PG_element_rank_modified(
				v3, 1 /* stride */, len, line[1 + alpha]);
		}
}

void print_set_in_affine_plane(finite_field &GFq, int len, int *S)
{
	int *A;
	int i, j, x, y, v[3];
	
	
	A = NEW_int(GFq.q * GFq.q);
	for (x = 0; x < GFq.q; x++) {
		for (y = 0; y < GFq.q; y++) {
			A[(GFq.q - 1 - y) * GFq.q + x] = 0;
			}
		}
	for (i = 0; i < len; i++) {
		GFq.PG_element_unrank_modified(
				v, 1 /* stride */, 3 /* len */, S[i]);
		if (v[2] != 1) {
			//cout << "my_generator::print_set_in_affine_plane
			// not an affine point" << endl;
			cout << "(" << v[0] << "," << v[1]
					<< "," << v[2] << ")" << endl;
			continue;
			}
		x = v[0];
		y = v[1];
		A[(GFq.q - 1 - y) * GFq.q + x] = 1;
		}
	for (i = 0; i < GFq.q; i++) {
		for (j = 0; j < GFq.q; j++) {
			cout << A[i * GFq.q + j];
			}
		cout << endl;
		}
	FREE_int(A);
}

int consecutive_ones_property_in_affine_plane(
		ostream &ost, finite_field &GFq, int len, int *S)
{
	int i, y, v[3];
	
	
	for (i = 0; i < len; i++) {
		GFq.PG_element_unrank_modified(
				v, 1 /* stride */, 3 /* len */, S[i]);
		if (v[2] != 1) {
			cout << "my_generator::consecutive_ones_"
					"property_in_affine_plane "
					"not an affine point" << endl;
			ost << "(" << v[0] << "," << v[1]
				<< "," << v[2] << ")" << endl;
			exit(1);
			}
		y = v[1];
		if (y != i)
			return FALSE;
		}
	return TRUE;
}

int line_intersection_with_oval(finite_field &GFq,
	int *f_oval_point, int line_rk,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int line[3], base_cols[3], K[6];
	int points[6], rk, kernel_m, kernel_n;
	int j, w[2], a, b, pt[3], nb = 0;
	
	if (f_v) {
		cout << "intersecting line " << line_rk
				<< " with the oval" << endl;
		}
	GFq.PG_element_unrank_modified(line, 1, 3, line_rk);
	rk = GFq.Gauss_int(line,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /* P */,
		1 /* m */, 3 /* n */, 0 /* Pn */,
		0 /* verbose_level */);
	if (f_vv) {
		cout << "after Gauss:" << endl;
		print_integer_matrix(cout, line, 1, 3);
		}
	GFq.matrix_get_kernel(line, 1, 3, base_cols, rk, 
		kernel_m, kernel_n, K);
	int_matrix_transpose(K, kernel_m, kernel_n, points);
	if (f_vv) {
		cout << "points:" << endl;
		print_integer_matrix(cout, points, 2, 3);
		}
	for (j = 0; j < GFq.q + 1; j++) {
		GFq.PG_element_unrank_modified(w, 1, 2, j);
		a = GFq.mult(points[0], w[0]);
		b = GFq.mult(points[3], w[1]);
		pt[0] = GFq.add(a, b);
		a = GFq.mult(points[1], w[0]);
		b = GFq.mult(points[4], w[1]);
		pt[1] = GFq.add(a, b);
		a = GFq.mult(points[2], w[0]);
		b = GFq.mult(points[5], w[1]);
		pt[2] = GFq.add(a, b);
		GFq.PG_element_rank_modified(pt, 1, 3, rk);
		//cout << j << " : " << pt[0] << ","
		//<< pt[1] << "," << pt[2] << " : " << rk << " : ";
		if (f_oval_point[rk]) {
			//cout << "yes" << endl;
			if (f_v) {
				cout << "found oval point" << endl;
				}
			nb++;
			}
		else {
			//cout << "no" << endl;
			}
		}
	return nb;
}

int get_base_line(finite_field &GFq,
	int plane1, int plane2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "get_base_line()" << endl;
		}
	int planes[8], base_cols[4], rk;
	int intersection[16], intersection2[16];
	int lines[6], line[3], kernel_m, kernel_n, line_rk;

	AG_element_unrank(GFq.q, planes, 1, 3, plane1);
	planes[3] = 1;
	AG_element_unrank(GFq.q, planes + 4, 1, 3, plane2);
	planes[7] = 1;
	if (f_v) {
		cout << "planes:" << endl;
		print_integer_matrix(cout, planes, 2, 4);
		}
	rk = GFq.Gauss_int(planes,
		FALSE /* f_special */,
		TRUE /* f_complete */,
		base_cols,
		FALSE /* f_P */,
		NULL /* P */, 2 /* m */, 4 /* n */, 0 /* Pn */,
		0 /* verbose_level */);
	if (f_v) {
		cout << "after Gauss:" << endl;
		print_integer_matrix(cout, planes, 2, 4);
		}
	GFq.matrix_get_kernel(planes, 2, 4, base_cols, rk, 
		kernel_m, kernel_n, intersection);
	int_matrix_transpose(intersection,
			kernel_m, kernel_n, intersection2);
	if (f_v) {
		cout << "kernel:" << endl;
		print_integer_matrix(cout,
				intersection2, kernel_n, kernel_m);
		}
	lines[0] = intersection2[0];
	lines[1] = intersection2[1];
	lines[2] = intersection2[2];
	lines[3] = intersection2[4];
	lines[4] = intersection2[5];
	lines[5] = intersection2[6];
	if (f_v) {
		cout << "lines:" << endl;
		print_integer_matrix(cout, lines, 2, 3);
		}
	rk = GFq.Gauss_int(lines,
		FALSE /* f_special */,
		TRUE /* f_complete */,
		base_cols,
		FALSE /* f_P */, NULL /* P */,
		2 /* m */, 3 /* n */, 0 /* Pn */,
		0 /* verbose_level */);
	if (rk != 2) {
		cout << "rk != 2" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "after Gauss:" << endl;
		print_integer_matrix(cout, lines, 2, 3);
		}
	GFq.matrix_get_kernel(lines,
			2, 3, base_cols, rk, kernel_m, kernel_n, line);
	if (f_v) {
		cout << "the line:" << endl;
		print_integer_matrix(cout, line, 1, 3);
		}
	GFq.PG_element_rank_modified(line, 1, 3, line_rk);
	if (f_v) {
		cout << "has rank " << line_rk << endl;
		}
	return line_rk;
}


void create_Fisher_BLT_set(int *Fisher_BLT,
		int q, const char *poly_q, const char *poly_Q, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	unusual_model U;

	U.setup(q, poly_q, poly_Q, verbose_level);
	U.create_Fisher_BLT_set(Fisher_BLT, verbose_level);

}

void create_Linear_BLT_set(int *BLT, int q,
		const char *poly_q, const char *poly_Q, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	unusual_model U;

	U.setup(q, poly_q, poly_Q, verbose_level);
	U.create_Linear_BLT_set(BLT, verbose_level);

}

void create_Mondello_BLT_set(int *BLT, int q,
		const char *poly_q, const char *poly_Q, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	unusual_model U;

	U.setup(q, poly_q, poly_Q, verbose_level);
	U.create_Mondello_BLT_set(BLT, verbose_level);

}

void print_quadratic_form_list_coded(int form_nb_terms,
	int *form_i, int *form_j, int *form_coeff)
{
	int k;

	for (k = 0; k < form_nb_terms; k++) {
		cout << "i=" << form_i[k] << " j=" << form_j[k]
			<< " coeff=" << form_coeff[k] << endl;
		}
}

void make_Gram_matrix_from_list_coded_quadratic_form(
	int n, finite_field &F,
	int nb_terms, int *form_i, int *form_j, int *form_coeff, int *Gram)
{
	int k, i, j, c;

	int_vec_zero(Gram, n * n);
#if 0
	for (i = 0; i < n * n; i++)
		Gram[i] = 0;
#endif
	for (k = 0; k < nb_terms; k++) {
		i = form_i[k];
		j = form_j[k];
		c = form_coeff[k];
		if (c == 0) {
			continue;
			}
		Gram[i * n + j] = F.add(Gram[i * n + j], c);
		Gram[j * n + i] = F.add(Gram[j * n + i], c);
		}
}

void add_term(int n, finite_field &F,
	int &nb_terms, int *form_i, int *form_j, int *form_coeff,
	int *Gram,
	int i, int j, int coeff)
{
	form_i[nb_terms] = i;
	form_j[nb_terms] = j;
	form_coeff[nb_terms] = coeff;
	if (i == j) {
		Gram[i * n + j] = F.mult(2, coeff);
		}
	else {
		Gram[i * n + j] = coeff;
		Gram[j * n + i] = coeff;
		}
	nb_terms++;
}






}}

