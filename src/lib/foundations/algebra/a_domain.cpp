// a_domain.cpp
// 
// Anton Betten
// March 14, 2015
//
//
// 
//
//



#include "foundations.h"

using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace algebra {


a_domain::a_domain()
{
	null();
}

a_domain::~a_domain()
{
	freeself();
}

void a_domain::null()
{
	kind = not_applicable;
}

void a_domain::freeself()
{
	
}

void a_domain::init_integers(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::init_integers" << endl;
	}
	kind = domain_the_integers;
	size_of_instance_in_int = 1;
}

void a_domain::init_integer_fractions(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::init_integer_fractions" << endl;
		}
	kind = domain_integer_fractions;
	size_of_instance_in_int = 2;
}


int a_domain::as_int(int *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "a_domain::as_int" << endl;
	}
	if (kind == domain_the_integers) {
		return elt[0];
	}
	else if (kind == domain_integer_fractions) {
		long int at, ab, g;

		at = elt[0];
		ab = elt[1];
		if (at == 0) {
			return 0;
		}
		g = NT.gcd_lint(at, ab);
		at /= g;
		ab /= g;
		if (ab != 1 && ab != -1) {
			cout << "a_domain::as_int the number is not an integer" << endl;
			exit(1);
		}
		if (ab == -1) {
			at *= -1;
		}
		return at;
	}
	else {
		cout << "a_domain::as_int unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::make_integer(int *elt, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::make_integer" << endl;
	}
	if (kind == domain_the_integers) {
		elt[0] = n;
	}
	else if (kind == domain_integer_fractions) {
		elt[0] = n;
		elt[1] = 1;
	}
	else {
		cout << "a_domain::make_integer unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::make_zero(int *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::make_zero" << endl;
	}
	if (kind == domain_the_integers) {
		elt[0] = 0;
	}
	else if (kind == domain_integer_fractions) {
		elt[0] = 0;
		elt[1] = 1;
	}
	else {
		cout << "a_domain::make_zero unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::make_zero_vector(int *elt, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "a_domain::make_zero_vector" << endl;
	}
	for (i = 0; i < len; i++) {
		make_zero(elt + i * size_of_instance_in_int, 0);
	}
}

int a_domain::is_zero_vector(int *elt, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "a_domain::is_zero_vector" << endl;
	}
	for (i = 0; i < len; i++) {
		if (!is_zero(offset(elt, i), 0)) {

			//cout << "a_domain::is_zero_vector element "
			// << i << " is nonzero" << endl;

			return FALSE;
		}
	}
	return TRUE;
}


int a_domain::is_zero(int *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "a_domain::is_zero" << endl;
	}
	if (kind == domain_the_integers) {
		if (elt[0] == 0) {
			ret = TRUE;
		}
		else {
			ret = FALSE;
		}
	}
	else if (kind == domain_integer_fractions) {
		if (elt[0] == 0) {
			ret = TRUE;
		}
		else {
			ret = FALSE;
		}
	}
	else {
		cout << "a_domain::is_zero unknown domain kind" << endl;
		exit(1);
	}
	return ret;
}

void a_domain::make_one(int *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::make_one" << endl;
	}
	if (kind == domain_the_integers) {
		elt[0] = 1;
	}
	else if (kind == domain_integer_fractions) {
		elt[0] = 1;
		elt[1] = 1;
	}
	else {
		cout << "a_domain::make_one unknown domain kind" << endl;
		exit(1);
	}
}

int a_domain::is_one(int *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = FALSE;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "a_domain::is_one" << endl;
	}
	if (kind == domain_the_integers) {
		if (elt[0] == 1) {
			ret = TRUE;
		}
		else {
			ret = FALSE;
		}
	}
	else if (kind == domain_integer_fractions) {
		long int at, ab, g;
		
		at = elt[0];
		ab = elt[1];
		if (at == 0) {
			ret = FALSE;
		}
		else {
			g = NT.gcd_lint(at, ab);
			at = at / g;
			ab = ab / g;
			if (ab < 0) {
				ab *= -1;
				at *= -1;
			}
			if (at == 1) {
				ret = TRUE;
			}
			else {
				ret = FALSE;
			}
		}
	}
	else {
		cout << "a_domain::is_one unknown domain kind" << endl;
		exit(1);
	}
	return ret;
}

void a_domain::copy(int *elt_from, int *elt_to, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::copy" << endl;
	}
	if (kind == domain_the_integers) {
		elt_to[0] = elt_from[0];
	}
	else if (kind == domain_integer_fractions) {
		elt_to[0] = elt_from[0];
		elt_to[1] = elt_from[1];
	}
	else {
		cout << "a_domain::copy unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::copy_vector(int *elt_from, int *elt_to,
		int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "a_domain::copy_vector" << endl;
	}
	for (i = 0; i < len; i++) {
		copy(offset(elt_from, i), offset(elt_to, i), 0);
	}
}


void a_domain::swap_vector(int *elt1, int *elt2,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "a_domain::swap_vector" << endl;
	}
	for (i = 0; i < n; i++) {
		swap(elt1 + i * size_of_instance_in_int,
				elt2 + i * size_of_instance_in_int, 0);
	}
}

void a_domain::swap(int *elt1, int *elt2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::swap" << endl;
	}
	if (kind == domain_the_integers) {
		int a;
		a = elt2[0];
		elt2[0] = elt1[0];
		elt1[0] = a;
	}
	else if (kind == domain_integer_fractions) {
		int a;
		a = elt2[0];
		elt2[0] = elt1[0];
		elt1[0] = a;
		a = elt2[1];
		elt2[1] = elt1[1];
		elt1[1] = a;
	}
	else {
		cout << "a_domain::copy unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::add(int *elt_a, int *elt_b, int *elt_c, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "a_domain::add" << endl;
	}
	if (kind == domain_the_integers) {
		elt_c[0] = elt_a[0] + elt_b[0];
	}
	else if (kind == domain_integer_fractions) {
		int at, ab, bt, bb, ct, cb;

		at = elt_a[0];
		ab = elt_a[1];
		bt = elt_b[0];
		bb = elt_b[1];
		
		NT.int_add_fractions(at, ab, bt, bb, ct, cb, 0 /* verbose_level */);

		elt_c[0] = ct;
		elt_c[1] = cb;
	}
	else {
		cout << "a_domain::add unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::add_apply(int *elt_a, int *elt_b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "a_domain::add_apply" << endl;
	}
	if (kind == domain_the_integers) {
		elt_a[0] = elt_a[0] + elt_b[0];
	}
	else if (kind == domain_integer_fractions) {
		int at, ab, bt, bb, ct, cb;

		at = elt_a[0];
		ab = elt_a[1];
		bt = elt_b[0];
		bb = elt_b[1];
		
		NT.int_add_fractions(at, ab, bt, bb, ct, cb, 0 /* verbose_level */);

		elt_a[0] = ct;
		elt_a[1] = cb;
	}
	else {
		cout << "a_domain::add_apply unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::subtract(int *elt_a, int *elt_b, int *elt_c, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "a_domain::subtract" << endl;
	}
	if (kind == domain_the_integers) {
		elt_c[0] = elt_a[0] + elt_b[0];
	}
	else if (kind == domain_integer_fractions) {
		long int at, ab, bt, bb, g, a1, b1, ct, cb;

		at = elt_a[0];
		ab = elt_a[1];
		bt = elt_b[0];
		bb = elt_b[1];
		g = NT.gcd_lint(ab, bb);
		a1 = ab / g;
		b1 = bb / g;
		cb = a1 * g;
		ct = at * b1 - bt * a1;
		elt_c[0] = ct;
		elt_c[1] = cb;
	}
	else {
		cout << "a_domain::subtract unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::negate(int *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::negate" << endl;
	}
	if (kind == domain_the_integers) {
		elt[0] = - elt[0];
	}
	else if (kind == domain_integer_fractions) {
		elt[0] = - elt[0];
	}
	else {
		cout << "a_domain::negate unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::negate_vector(int *elt, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "a_domain::negate" << endl;
	}
	for (i = 0; i < len; i++) {
		negate(offset(elt, i), 0);
	}
}

void a_domain::mult(int *elt_a, int *elt_b, int *elt_c, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "a_domain::mult" << endl;
	}
	if (kind == domain_the_integers) {
		elt_c[0] = elt_a[0] * elt_b[0];
	}
	else if (kind == domain_integer_fractions) {
		int at, ab, bt, bb, ct, cb;

		at = elt_a[0];
		ab = elt_a[1];
		bt = elt_b[0];
		bb = elt_b[1];


		NT.int_mult_fractions(at, ab, bt, bb, ct, cb, 0 /* verbose_level */);
		
		elt_c[0] = ct;
		elt_c[1] = cb;
	}
	else {
		cout << "a_domain::mult unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::mult_apply(int *elt_a, int *elt_b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "a_domain::mult_apply" << endl;
	}
	if (kind == domain_the_integers) {
		elt_a[0] = elt_a[0] * elt_b[0];
	}
	else if (kind == domain_integer_fractions) {
		int at, ab, bt, bb, ct, cb;

		at = elt_a[0];
		ab = elt_a[1];
		bt = elt_b[0];
		bb = elt_b[1];


		NT.int_mult_fractions(at, ab, bt, bb, ct, cb, 0 /* verbose_level */);
		
		//cout << "a_domain::mult_apply " << at << "/" << ab
		//<< " * " << bt << "/" << bb << " = " << ct << "/" << bb << endl;
		
		elt_a[0] = ct;
		elt_a[1] = cb;
	}
	else {
		cout << "a_domain::mult_apply unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::power(int *elt_a, int *elt_b, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *tmp1;
	int *tmp2;
	int *tmp3;


	if (f_v) {
		cout << "a_domain::power" << endl;
	}
	tmp1 = NEW_int(size_of_instance_in_int);
	tmp2 = NEW_int(size_of_instance_in_int);
	tmp3 = NEW_int(size_of_instance_in_int);

	if (n < 0) {
		cout << "a_domain::power exponent is negative" << endl;
		exit(1);
	}
	make_one(tmp1, 0);
	copy(elt_a, tmp3, 0);
	while (TRUE) {
		if (n % 2) {
			mult(tmp3, tmp1, tmp2, 0);
			copy(tmp2, tmp1, 0);
		}
		n >>= 1;
		if (n == 0) {
			break;
		}
		mult(tmp3, tmp3, tmp2, 0);
		copy(tmp2, tmp3, 0);
	}
	copy(tmp1, elt_b, 0);
	FREE_int(tmp1);
	FREE_int(tmp2);
	FREE_int(tmp3);
	if (f_v) {
		cout << "a_domain::power done" << endl;
	}
}

void a_domain::mult_by_integer(int *elt, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::mult_by_integer" << endl;
	}
	if (kind == domain_the_integers) {
		elt[0] *= n;
	}
	else if (kind == domain_integer_fractions) {
		elt[0] *= n;
	}
	else {
		cout << "a_domain::mult_by_integer unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::divide_by_integer(int *elt, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "a_domain::divide_by_integer" << endl;
	}
	if (kind == domain_the_integers) {
		int a;

		a = elt[0];
		if (a % n) {
			cout << "a_domain::divide_by_integer n does not divide" << endl;
			exit(1);
		}
		elt[0] /= n;
	}
	else if (kind == domain_integer_fractions) {
		long int a, b, g, n1;

		a = elt[0];
		b = elt[1];
		g = NT.gcd_lint(a, n);
		a /= g;
		n1 = n / g;
		b *= n1;	
		elt[0] = a;
		elt[1] = b;
	}
	else {
		cout << "a_domain::divide_by_integer unknown domain kind" << endl;
		exit(1);
	}
}


void a_domain::divide(int *elt_a, int *elt_b, int *elt_c, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "a_domain::divide" << endl;
	}
	if (kind == domain_the_integers) {
		long int g;

		if (elt_b[0] == 0) {
			cout << "a_domain::divide division by zero" << endl;
			exit(1);
		}
		g = NT.gcd_lint(elt_a[0], elt_b[0]);
		elt_c[0] = elt_a[0] / g;
	}
	else if (kind == domain_integer_fractions) {
		int at, ab, bt, bb, ct, cb;

		at = elt_a[0];
		ab = elt_a[1];
		bt = elt_b[0];
		bb = elt_b[1];

		if (bt == 0) {
			cout << "a_domain::divide division by zero" << endl;
			exit(1);
		}

		NT.int_mult_fractions(at, ab, bb, bt, ct, cb, 0 /* verbose_level */);

		elt_c[0] = ct;
		elt_c[1] = cb;
	}
	else {
		cout << "a_domain::divide unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::inverse(int *elt_a, int *elt_b, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::inverse" << endl;
	}
	if (kind == domain_the_integers) {
		int a, av;
		
		a = elt_a[0];
		if (a == 1) {
			av = 1;
		}
		else if (a == -1) {
			av = -1;
		}
		else {
			cout << "a_domain::inverse cannot invert" << endl;
			exit(1);
		}
		elt_b[0] = av;
	}
	else if (kind == domain_integer_fractions) {
		int at, ab;

		
		at = elt_a[0];
		ab = elt_a[1];
		if (at == 0) {
			cout << "a_domain::inverse cannot invert" << endl;
			exit(1);
		}
		elt_b[0] = ab;
		elt_b[1] = at;
	}
	else {
		cout << "a_domain::inverse unknown domain kind" << endl;
		exit(1);
	}
}


void a_domain::print(int *elt)
{
	if (kind == domain_the_integers) {
		cout << elt[0];
	}
	else if (kind == domain_integer_fractions) {
		int at, ab;

		at = elt[0];
		ab = elt[1];

#if 1
		if (ab == -1) {
			at *= -1;
			ab = 1;
		}
		if (at == 0) {
			cout << 0;
		}
		else if (ab == 1) {
			cout << at;
		}
		else {
			cout << at << "/" << ab;
		}
#else
		cout << at << "/" << ab;
#endif
	}
	else {
		cout << "a_domain::divide unknown domain kind" << endl;
		exit(1);
	}
}

void a_domain::print_vector(int *elt, int n)
{
	int i;
	
	cout << "(";
	for (i = 0; i < n; i++) {
		print(offset(elt, i));
		if (i < n - 1) {
			cout << ", ";
		}
	}
	cout << ")";
}

void a_domain::print_matrix(int *A, int m, int n)
{
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			print(offset(A, i * n + j));
			if (j < n - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}
}

void a_domain::print_matrix_for_maple(int *A, int m, int n)
{
	int i, j;
	
	cout << "[";
	for (i = 0; i < m; i++) {
		cout << "[";
		for (j = 0; j < n; j++) {
			print(offset(A, i * n + j));
			if (j < n - 1) {
				cout << ", ";
			}
		}
		cout << "]";
		if (i < m - 1) {
			cout << "," << endl;
		}
		else {
			cout << "];" << endl;
		}
	}
}

void a_domain::make_element_from_integer(int *elt, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::make_element_from_integer" << endl;
	}

	if (kind == domain_the_integers) {
		elt[0] = n;
	}
	else if (kind == domain_integer_fractions) {
		elt[0] = n;
		elt[1] = 1;
	}
	else {
		cout << "a_domain::make_element_from_integer "
				"unknown domain kind" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "a_domain::make_element_from_integer done" << endl;
	}
}

int *a_domain::offset(int *A, int i)
{
	return A + i * size_of_instance_in_int;
}

int a_domain::Gauss_echelon_form(int *A,
	int f_special, int f_complete, int *base_cols,
	int f_P, int *P, int m, int n, int Pn, int verbose_level)
// returns the rank which is the number of entries in base_cols
// A is a m x n matrix,
// P is a m x Pn matrix (if f_P is TRUE)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int rank, i, j, k, jj;
	int *pivot, *pivot_inv;
	int *a, *b, *c, *z, *f;
	
	if (f_v) {
		cout << "Gauss algorithm for matrix:" << endl;
		//print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_tables();
		print_matrix(A, m, n);
	}

	pivot = NEW_int(size_of_instance_in_int);
	pivot_inv = NEW_int(size_of_instance_in_int);
	a = NEW_int(size_of_instance_in_int);
	b = NEW_int(size_of_instance_in_int);
	c = NEW_int(size_of_instance_in_int);
	z = NEW_int(size_of_instance_in_int);
	f = NEW_int(size_of_instance_in_int);

	i = 0;
	for (j = 0; j < n; j++) {
		if (f_vv) {
			cout << "j=" << j << endl;
		}
		/* search for pivot element: */
		for (k = i; k < m; k++) {
			if (!is_zero(offset(A, k * n + j), 0)) {
				if (f_vv) {
					cout << "i=" << i << " pivot found in "
							<< k << "," << j << endl;
				}
				// pivot element found: 
				if (k != i) {
					swap_vector(offset(A, i * n),
							offset(A, k * n), n, 0);
					if (f_P) {
						swap_vector(offset(P, i * Pn),
								offset(P, k * Pn), Pn, 0);
					}
				}
				break;
			} // if != 0
		} // next k
		
		if (k == m) { // no pivot found 
			if (f_vv) {
				cout << "no pivot found" << endl;
			}
			continue; // increase j, leave i constant
		}
		
		if (f_vv) {
			cout << "row " << i << " pivot in row " << k
					<< " colum " << j << endl;
		}
		
		base_cols[i] = j;
		//if (FALSE) {
		//	cout << "."; cout.flush();
		//	}


		copy(offset(A, i * n + j), pivot, 0);
		if (f_vv) {
			cout << "pivot=";
			print(pivot);
			cout << endl;
		}
		inverse(pivot, pivot_inv, 0);

		if (f_vv) {
			cout << "pivot=";
			print(pivot);
			cout << " pivot_inv=";
			print(pivot_inv);
			cout << endl;
		}
		if (!f_special) {
			// make pivot to 1: 
			for (jj = j; jj < n; jj++) {
				mult_apply(offset(A, i * n + jj), pivot_inv, 0);
			}
			if (f_P) {
				for (jj = 0; jj < Pn; jj++) {
					mult_apply(offset(P, i * Pn + jj), pivot_inv, 0);
				}
			}
			if (f_vv) {
				cout << "pivot=";
				print(pivot);
				cout << " pivot_inv=";
				print(pivot_inv); 
				cout << " made to one: ";
				print(offset(A, i * n + j));
				cout << endl;
			}
			if (f_vvv) {
				print_matrix(A, m, n);
			}
		}
		
		// do the gaussian elimination: 

		if (f_vv) {
			cout << "doing elimination in column " << j
					<< " from row " << i + 1 << " to row "
					<< m - 1 << ":" << endl;
		}
		for (k = i + 1; k < m; k++) {
			if (f_vv) {
				cout << "looking at row k=" << k << endl;
			}
			copy(offset(A, k * n + j), z, 0);
			if (is_zero(z, 0)) {
				continue;
			}
			if (f_special) {
				mult(z, pivot_inv, f, 0);
				//f = mult(z, pivot_inv);
			}
			else {
				copy(z, f, 0);
				//f = z;
			}
			negate(f, 0);
			//f = negate(f);
			make_zero(offset(A, k * n + j), 0);
			//A[k * n + j] = 0;
			if (f_vv) {
				cout << "eliminating row " << k << endl;
			}
			for (jj = j + 1; jj < n; jj++) {
				if (f_vv) {
					cout << "eliminating row " << k
							<< " column " << jj << endl;
				}
				copy(offset(A, i * n + jj), a, 0);
				//a = A[i * n + jj];
				copy(offset(A, k * n + jj), b, 0);
				//b = A[k * n + jj];
				// c := b + f * a
				//    = b - z * a              if !f_special 
				//      b - z * pivot_inv * a  if f_special 
				mult(f, a, c, 0);
				//c = mult(f, a);
				add_apply(c, b, 0);
				//c = add(c, b);
				copy(c, offset(A, k * n + jj), 0);
				//A[k * n + jj] = c;
				if (f_vv) {
					cout << "A=" << endl;
					print_matrix(A, m, n);
					//print(offset(A, k * n + jj));
					//cout << " ";
				}
			}
			if (f_P) {
				for (jj = 0; jj < Pn; jj++) {
					copy(offset(P, i * Pn + jj), a, 0);
					//a = P[i * Pn + jj];
					copy(offset(P, k * Pn + jj), b, 0);
					//b = P[k * Pn + jj];
					// c := b - z * a
					mult(f, a, c, 0);
					//c = mult(f, a);
					add_apply(c, b, 0);
					//c = add(c, b);
					copy(c, offset(P, k * Pn + jj), 0);
					//P[k * Pn + jj] = c;
				}
			}
			if (f_vv) {
				cout << endl;
			}
			if (f_vvv) {
				cout << "A=" << endl;
				print_matrix(A, m, n);
			}
		}
		i++;
		if (f_vv) {
			cout << "A=" << endl;
			print_matrix(A, m, n);
			if (f_P) {
				cout << "P=" << endl;
				print_matrix(P, m, Pn);
			}
		}
	} // next j
	rank = i;
	if (f_complete) {
		//if (FALSE) {
		//	cout << ";"; cout.flush();
		//	}
		for (i = rank - 1; i >= 0; i--) {
			if (f_v) {
				cout << "."; cout.flush();
			}
			j = base_cols[i];
			if (!f_special) {
				copy(offset(A, i * n + j), a, 0);
				//a = A[i * n + j];
			}
			else {
				copy(offset(A, i * n + j), pivot, 0);
				//pivot = A[i * n + j];
				inverse(pivot, pivot_inv, 0);
				//pivot_inv = inverse(pivot);
			}
			// do the gaussian elimination in the upper part: 
			for (k = i - 1; k >= 0; k--) {
				copy(offset(A, k * n + j), z, 0);
				//z = A[k * n + j];
				if (z == 0) {
					continue;
				}
				make_zero(offset(A, k * n + j), 0);
				//A[k * n + j] = 0;
				for (jj = j + 1; jj < n; jj++) {
					copy(offset(A, i * n + jj), a, 0);
					//a = A[i * n + jj];
					copy(offset(A, k * n + jj), b, 0);
					//b = A[k * n + jj];
					if (f_special) {
						mult_apply(a, pivot_inv, 0);
						//a = mult(a, pivot_inv);
					}
					mult(z, a, c, 0);
					//c = mult(z, a);
					negate(c, 0);
					//c = negate(c);
					add_apply(c, b, 0);
					//c = add(c, b);
					copy(c, offset(A, k * n + jj), 0);
					//A[k * n + jj] = c;
				}
				if (f_P) {
					for (jj = 0; jj < Pn; jj++) {
						copy(offset(P, i * Pn + jj), a, 0);
						//a = P[i * Pn + jj];
						copy(offset(P, k * Pn + jj), b, 0);
						//b = P[k * Pn + jj];
						if (f_special) {
							mult_apply(a, pivot_inv, 0);
							//a = mult(a, pivot_inv);
							}
						mult(z, a, c, 0);
						//c = mult(z, a);
						negate(c, 0);
						//c = negate(c);
						add_apply(c, b, 0);
						//c = add(c, b);
						copy(c, offset(P, k * Pn + jj), 0);
						//P[k * Pn + jj] = c;
					}
				}
			} // next k
		} // next i
	}

	FREE_int(pivot);
	FREE_int(pivot_inv);
	FREE_int(a);
	FREE_int(b);
	FREE_int(c);
	FREE_int(z);
	FREE_int(f);
	
	if (f_v) { 
		cout << endl;
		print_matrix(A, m, n);
		cout << "the rank is " << rank << endl;
	}
	return rank;
}


void a_domain::Gauss_step(int *v1, int *v2,
		int len, int idx, int verbose_level)
// afterwards: v2[idx] = 0 and v1,v2 span the same space as before
// v1 is not changed if v1[idx] is nonzero
{
	int i;
	int f_v = (verbose_level >= 1);
	int *tmp1;
	int *tmp2;
	
	if (f_v) {
		cout << "Gauss_step" << endl;
	}

	tmp1 = NEW_int(size_of_instance_in_int);
	tmp2 = NEW_int(size_of_instance_in_int);

	if (is_zero(offset(v2, idx), 0)) {
		goto after;
	}
	if (is_zero(offset(v1, idx), 0)) {
		// do a swap:
		for (i = 0; i < len; i++) {
			swap(offset(v1, i), offset(v2, i), 0);
		}
		goto after;
	}

	copy(offset(v1, idx), tmp1, 0);
	inverse(tmp1, tmp2, 0);
	mult(tmp2, offset(v2, idx), tmp1, 0);
	negate(tmp1, 0);

	//cout << "Gauss_step a=" << a << endl;
	for (i = 0; i < len; i++) {
		mult(tmp1, offset(v1, i), tmp2, 0);
		add_apply(offset(v2, i), tmp2, 0);
	}

after:
	if (f_v) {
		cout << "Gauss_step done" << endl;
	}

	FREE_int(tmp1);
	FREE_int(tmp2);
}

void a_domain::matrix_get_kernel(int *M,
	int m, int n, int *base_cols, int nb_base_cols,
	int &kernel_m, int &kernel_n, int *kernel, int verbose_level)
// kernel must point to the appropriate amount of memory!
// (at least n * (n - nb_base_cols) int's)
// kernel is stored as column vectors,
// i.e. kernel_m = n and kernel_n = n - nb_base_cols.
{
	int f_v = (verbose_level >= 1);
	int r, k, i, j, ii, jj;
	int *kernel_cols;
	int *zero;
	int *m_one;
	
	if (f_v) {
		cout << "a_domain::matrix_get_kernel" << endl;
	}
	zero = NEW_int(size_of_instance_in_int);
	m_one = NEW_int(size_of_instance_in_int);

	make_zero(zero, 0);
	make_one(m_one, 0);
	negate(m_one, 0);


	r = nb_base_cols;
	k = n - r;
	kernel_m = n;
	kernel_n = k;
	
	kernel_cols = NEW_int(k);

	Int_vec_complement_to(base_cols, kernel_cols, n, nb_base_cols);
	

	for (i = 0; i < r; i++) {
		ii = base_cols[i];
		for (j = 0; j < k; j++) {
			jj = kernel_cols[j];

			copy(offset(M, i * n + jj),
					offset(kernel, ii * kernel_n + j), 0);
		}
	}
	for (i = 0; i < k; i++) {
		ii = kernel_cols[i];
		for (j = 0; j < k; j++) {
			if (i == j) {
				copy(m_one, offset(kernel, ii * kernel_n + j), 0);
			}
			else {
				copy(zero, offset(kernel, ii * kernel_n + j), 0);
			}
		}
	}
	

	FREE_int(kernel_cols);
	FREE_int(zero);
	FREE_int(m_one);
	if (f_v) {
		cout << "a_domain::matrix_get_kernel done" << endl;
	}
}

void a_domain::matrix_get_kernel_as_row_vectors(
	int *M, int m, int n, int *base_cols, int nb_base_cols,
	int &kernel_m, int &kernel_n, int *kernel, int verbose_level)
// kernel must point to the appropriate amount of memory!
// (at least n * (n - nb_base_cols) int's)
// kernel is stored as row vectors,
// i.e. kernel_m = n - nb_base_cols and kernel_n = n.
{
	int f_v = (verbose_level >= 1);
	int r, k, i, j, ii, jj;
	int *kernel_cols;
	int *zero;
	int *m_one;
	
	if (f_v) {
		cout << "a_domain::matrix_get_kernel_as_row_vectors" << endl;
	}
	zero = NEW_int(size_of_instance_in_int);
	m_one = NEW_int(size_of_instance_in_int);

	make_zero(zero, 0);
	make_one(m_one, 0);
	negate(m_one, 0);


	r = nb_base_cols;
	k = n - r;
	kernel_m = k;
	kernel_n = n;
	
	kernel_cols = NEW_int(k);

	Int_vec_complement_to(base_cols, kernel_cols, n, nb_base_cols);
	

	for (i = 0; i < r; i++) {
		ii = base_cols[i];
		for (j = 0; j < k; j++) {
			jj = kernel_cols[j];

			copy(offset(M, i * n + jj),
					offset(kernel, j * kernel_n + ii), 0);
		}
	}
	for (i = 0; i < k; i++) {
		ii = kernel_cols[i];
		for (j = 0; j < k; j++) {
			if (i == j) {
				copy(m_one, offset(kernel, j * kernel_n + ii), 0);
			}
			else {
				copy(zero, offset(kernel, j * kernel_n + ii), 0);
			}
		}
	}
	

	FREE_int(kernel_cols);
	FREE_int(zero);
	FREE_int(m_one);
	if (f_v) {
		cout << "a_domain::matrix_get_kernel_as_row_vectors done" << endl;
	}
}

void a_domain::get_image_and_kernel(int *M,
		int n, int &rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_special = FALSE;
	int f_complete = TRUE;
	int *base_cols;
	int f_P = FALSE;
	int kernel_m, kernel_n;
	
	if (f_v) {
		cout << "a_domain::get_image_and_kernel" << endl;
	}

	base_cols = NEW_int(n);
	rk = Gauss_echelon_form(M, f_special, f_complete, base_cols, 
		f_P, NULL, n, n, n, verbose_level);

	matrix_get_kernel_as_row_vectors(M, n, n, base_cols, rk, 
		kernel_m, kernel_n, offset(M, rk * n), verbose_level);

	if (f_v) {
		cout << "a_domain::get_image_and_kernel M=" << endl;
		print_matrix(M, n, n);
	}

	FREE_int(base_cols);
	if (f_v) {
		cout << "a_domain::get_image_and_kernel done" << endl;
	}
}

void a_domain::complete_basis(int *M, int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_special = FALSE;
	int f_complete = TRUE;
	int f_P = FALSE;
	int *M1;
	int *base_cols;
	int *kernel_cols;
	int i, j, k, a, rk;
	
	if (f_v) {
		cout << "a_domain::complete_basis" << endl;
	}

	M1 = NEW_int(m * n * size_of_instance_in_int);
	copy_vector(M, M1, m * n, 0);
	
	base_cols = NEW_int(n);

	rk = Gauss_echelon_form(M1, f_special, f_complete, base_cols, 
		f_P, NULL, m, n, n, verbose_level);

	if (rk != m) {
		cout << "a_domain::complete_basis rk != m" << endl;
		exit(1);
	}
	
	k = n - rk;

	kernel_cols = NEW_int(k);

	Int_vec_complement_to(base_cols, kernel_cols, n, rk);
	for (i = rk; i < n; i++) {
		for (j = 0; j < n; j++) {
			make_zero(offset(M, i * n + j), 0);
		}
	}
	for (i = 0; i < k; i++) {
		a = kernel_cols[i];
		make_one(offset(M, (rk + i) * n + a), 0);
	}

	if (f_v) {
		cout << "a_domain::complete_basis M=" << endl;
		print_matrix(M, n, n);
	}

	FREE_int(base_cols);
	FREE_int(kernel_cols);
	FREE_int(M1);
	if (f_v) {
		cout << "a_domain::complete_basis done" << endl;
	}
}

void a_domain::mult_matrix(int *A, int *B, int *C,
		int ma, int na, int nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, k;
	int *D, *E;
	
	if (f_v) {
		cout << "a_domain::mult_matrix" << endl;
	}
	if (f_vv) {
		cout << "a_domain::mult_matrix A=" << endl;
		print_matrix(A, ma, na);
		cout << "a_domain::mult_matrix B=" << endl;
		print_matrix(B, na, nb);
	}
	D = NEW_int(size_of_instance_in_int);
	E = NEW_int(size_of_instance_in_int);
	for (i = 0; i < ma; i++) {
		for (k = 0; k < nb; k++) {
			make_zero(D, 0);
			for (j = 0; j < na; j++) {
				mult(offset(A, i * na + j),
						offset(B, j * nb + k), E, 0);
				add_apply(D, E, 0);
			}
			copy(D, offset(C, i * nb + k), 0);
		}
	}
	FREE_int(D);
	FREE_int(E);
	if (f_vv) {
		cout << "a_domain::mult_matrix C=" << endl;
		print_matrix(C, ma, nb);
	}
	if (f_v) {
		cout << "a_domain::mult_matrix done" << endl;
	}
}

void a_domain::mult_matrix3(int *A, int *B, int *C, int *D,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *T;
	
	if (f_v) {
		cout << "a_domain::mult_matrix3" << endl;
	}
	T = NEW_int(n * n * size_of_instance_in_int);
	mult_matrix(A, B, T, n, n, n, 0);
	mult_matrix(T, C, D, n, n, n, 0);
	FREE_int(T);
	if (f_v) {
		cout << "a_domain::mult_matrix3 done" << endl;
	}
}

void a_domain::add_apply_matrix(int *A, int *B,
		int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	
	if (f_v) {
		cout << "a_domain::add_apply_matrix" << endl;
	}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			add_apply(offset(A, i * n + j), offset(B, i * n + j), 0);
		}
	}
	if (f_v) {
		cout << "a_domain::add_apply_matrix done" << endl;
	}
}

void a_domain::matrix_mult_apply_scalar(int *A,
		int *s, int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	
	if (f_v) {
		cout << "a_domain::matrix_mult_apply_scalar" << endl;
	}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			mult_apply(offset(A, i * n + j), s, 0);
		}
	}
	if (f_v) {
		cout << "a_domain::matrix_mult_apply_scalar done" << endl;
	}
}

void a_domain::make_block_matrix_2x2(int *Mtx,
		int n, int k, int *A, int *B, int *C, int *D,
		int verbose_level)
// A is k x k,
// B is k x (n - k),
// C is (n - k) x k,
// D is (n - k) x (n - k),
// Mtx is n x n
{
	int f_v = (verbose_level >= 1);
	int i, j, r;
	
	if (f_v) {
		cout << "a_domain::make_block_matrix_2x2" << endl;
	}
	r = n - k;
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			copy(offset(A, i * k + j), offset(Mtx, i * n + j), 0);
		}
	}
	for (i = 0; i < k; i++) {
		for (j = 0; j < r; j++) {
			copy(offset(B, i * r + j), offset(Mtx, i * n + k + j), 0);
		}
	}
	for (i = 0; i < r; i++) {
		for (j = 0; j < k; j++) {
			copy(offset(C, i * k + j), offset(Mtx, (k + i) * n + j), 0);
		}
	}
	for (i = 0; i < r; i++) {
		for (j = 0; j < r; j++) {
			copy(offset(D, i * r + j), offset(Mtx, (k + i) * n + k + j), 0);
		}
	}
	
	if (f_v) {
		cout << "a_domain::make_block_matrix_2x2 done" << endl;
	}
}

void a_domain::make_identity_matrix(int *A,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "a_domain::make_identity_matrix" << endl;
	}
	make_zero_vector(A, n * n, 0);
	for (i = 0; i < n; i++) {
		make_one(offset(A, i * n + i), 0);
	}
	if (f_v) {
		cout << "a_domain::make_identity_matrix done" << endl;
	}
}

void a_domain::matrix_inverse(int *A, int *Ainv,
		int n, int verbose_level)
{
	int *T, *basecols;
	
	T = NEW_int(n * n * size_of_instance_in_int);
	basecols = NEW_int(n * size_of_instance_in_int);
	
	matrix_invert(A, T, basecols, Ainv, n, verbose_level);
	
	FREE_int(T);
	FREE_int(basecols);
}

void a_domain::matrix_invert(int *A,
	int *T, int *basecols, int *Ainv, int n,
	int verbose_level)
// T[n * n]
// basecols[n]
{
	int rk;
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "a_domain::matrix_invert" << endl;
	}
	copy_vector(A, T, n * n, 0);
	make_identity_matrix(Ainv, n, 0);
	rk = Gauss_echelon_form(T,
		FALSE /* f_special */,
		TRUE /* f_complete */,
		basecols,
		TRUE /* f_P */, Ainv, n, n, n,
		verbose_level - 2);
	if (rk < n) {
		cout << "a_domain::matrix_invert not invertible" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "a_domain::matrix_invert done" << endl;
	}
}

}}}


