// a_domain.C
// 
// Anton Betten
// March 14, 2015
//
//
// 
//
//

#include "galois.h"

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

void a_domain::init_integers(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::init_integers" << endl;
		}
	kind = domain_the_integers;
	size_of_instance_in_INT = 1;
}

void a_domain::init_integer_fractions(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::init_integer_fractions" << endl;
		}
	kind = domain_integer_fractions;
	size_of_instance_in_INT = 2;
}


INT a_domain::as_INT(INT *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::as_INT" << endl;
		}
	if (kind == domain_the_integers) {
		return elt[0];
		}
	else if (kind == domain_integer_fractions) {
		INT at, ab, g;

		at = elt[0];
		ab = elt[1];
		if (at == 0) {
			return 0;
			}
		g = gcd_INT(at, ab);
		at /= g;
		ab /= g;
		if (ab != 1 && ab != -1) {
			cout << "a_domain::as_INT the number is not an integer" << endl;
			exit(1);
			}
		if (ab == -1) {
			at *= -1;
			}
		return at;
		}
	else {
		cout << "a_domain::as_INT unknown domain kind" << endl;
		exit(1);
		}
}

void a_domain::make_integer(INT *elt, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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

void a_domain::make_zero(INT *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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

void a_domain::make_zero_vector(INT *elt, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "a_domain::make_zero_vector" << endl;
		}
	for (i = 0; i < len; i++) {
		make_zero(elt + i * size_of_instance_in_INT, 0);
		}
}

INT a_domain::is_zero_vector(INT *elt, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "a_domain::is_zero_vector" << endl;
		}
	for (i = 0; i < len; i++) {
		if (!is_zero(offset(elt, i), 0)) {

			//cout << "a_domain::is_zero_vector element " << i << " is nonzero" << endl;

			return FALSE;
			}
		}
	return TRUE;
}


INT a_domain::is_zero(INT *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT ret;

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

void a_domain::make_one(INT *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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

INT a_domain::is_one(INT *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT ret = FALSE;

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
		INT at, ab, g;
		
		at = elt[0];
		ab = elt[1];
		if (at == 0) {
			ret = FALSE;
			}
		else {
			g = gcd_INT(at, ab);
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

void a_domain::copy(INT *elt_from, INT *elt_to, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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

void a_domain::copy_vector(INT *elt_from, INT *elt_to, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "a_domain::copy_vector" << endl;
		}
	for (i = 0; i < len; i++) {
		copy(offset(elt_from, i), offset(elt_to, i), 0);
		}
}


void a_domain::swap_vector(INT *elt1, INT *elt2, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "a_domain::swap_vector" << endl;
		}
	for (i = 0; i < n; i++) {
		swap(elt1 + i * size_of_instance_in_INT, elt2 + i * size_of_instance_in_INT, 0);
		}
}

void a_domain::swap(INT *elt1, INT *elt2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::swap" << endl;
		}
	if (kind == domain_the_integers) {
		INT a;
		a = elt2[0];
		elt2[0] = elt1[0];
		elt1[0] = a;
		}
	else if (kind == domain_integer_fractions) {
		INT a;
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

void a_domain::add(INT *elt_a, INT *elt_b, INT *elt_c, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::add" << endl;
		}
	if (kind == domain_the_integers) {
		elt_c[0] = elt_a[0] + elt_b[0];
		}
	else if (kind == domain_integer_fractions) {
		INT at, ab, bt, bb, ct, cb;

		at = elt_a[0];
		ab = elt_a[1];
		bt = elt_b[0];
		bb = elt_b[1];
		
		INT_add_fractions(at, ab, bt, bb, ct, cb, 0 /* verbose_level */);

		elt_c[0] = ct;
		elt_c[1] = cb;
		}
	else {
		cout << "a_domain::add unknown domain kind" << endl;
		exit(1);
		}
}

void a_domain::add_apply(INT *elt_a, INT *elt_b, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::add_apply" << endl;
		}
	if (kind == domain_the_integers) {
		elt_a[0] = elt_a[0] + elt_b[0];
		}
	else if (kind == domain_integer_fractions) {
		INT at, ab, bt, bb, ct, cb;

		at = elt_a[0];
		ab = elt_a[1];
		bt = elt_b[0];
		bb = elt_b[1];
		
		INT_add_fractions(at, ab, bt, bb, ct, cb, 0 /* verbose_level */);

		elt_a[0] = ct;
		elt_a[1] = cb;
		}
	else {
		cout << "a_domain::add_apply unknown domain kind" << endl;
		exit(1);
		}
}

void a_domain::subtract(INT *elt_a, INT *elt_b, INT *elt_c, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::subtract" << endl;
		}
	if (kind == domain_the_integers) {
		elt_c[0] = elt_a[0] + elt_b[0];
		}
	else if (kind == domain_integer_fractions) {
		INT at, ab, bt, bb, g, a1, b1, ct, cb;

		at = elt_a[0];
		ab = elt_a[1];
		bt = elt_b[0];
		bb = elt_b[1];
		g = gcd_INT(ab, bb);
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

void a_domain::negate(INT *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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

void a_domain::negate_vector(INT *elt, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "a_domain::negate" << endl;
		}
	for (i = 0; i < len; i++) {
		negate(offset(elt, i), 0);
		}
}

void a_domain::mult(INT *elt_a, INT *elt_b, INT *elt_c, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::mult" << endl;
		}
	if (kind == domain_the_integers) {
		elt_c[0] = elt_a[0] * elt_b[0];
		}
	else if (kind == domain_integer_fractions) {
		INT at, ab, bt, bb, ct, cb;

		at = elt_a[0];
		ab = elt_a[1];
		bt = elt_b[0];
		bb = elt_b[1];


		INT_mult_fractions(at, ab, bt, bb, ct, cb, 0 /* verbose_level */);
		
		elt_c[0] = ct;
		elt_c[1] = cb;
		}
	else {
		cout << "a_domain::mult unknown domain kind" << endl;
		exit(1);
		}
}

void a_domain::mult_apply(INT *elt_a, INT *elt_b, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::mult_apply" << endl;
		}
	if (kind == domain_the_integers) {
		elt_a[0] = elt_a[0] * elt_b[0];
		}
	else if (kind == domain_integer_fractions) {
		INT at, ab, bt, bb, ct, cb;

		at = elt_a[0];
		ab = elt_a[1];
		bt = elt_b[0];
		bb = elt_b[1];


		INT_mult_fractions(at, ab, bt, bb, ct, cb, 0 /* verbose_level */);
		
		//cout << "a_domain::mult_apply " << at << "/" << ab << " * " << bt << "/" << bb << " = " << ct << "/" << bb << endl;
		
		elt_a[0] = ct;
		elt_a[1] = cb;
		}
	else {
		cout << "a_domain::mult_apply unknown domain kind" << endl;
		exit(1);
		}
}

void a_domain::power(INT *elt_a, INT *elt_b, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *tmp1;
	INT *tmp2;
	INT *tmp3;


	if (f_v) {
		cout << "a_domain::power" << endl;
		}
	tmp1 = NEW_INT(size_of_instance_in_INT);
	tmp2 = NEW_INT(size_of_instance_in_INT);
	tmp3 = NEW_INT(size_of_instance_in_INT);

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
	FREE_INT(tmp1);
	FREE_INT(tmp2);
	FREE_INT(tmp3);
	if (f_v) {
		cout << "a_domain::power done" << endl;
		}
}

void a_domain::mult_by_integer(INT *elt, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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

void a_domain::divide_by_integer(INT *elt, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::divide_by_integer" << endl;
		}
	if (kind == domain_the_integers) {
		INT a;

		a = elt[0];
		if (a % n) {
			cout << "a_domain::divide_by_integer n does not divide" << endl;
			exit(1);
			}
		elt[0] /= n;
		}
	else if (kind == domain_integer_fractions) {
		INT a, b, g, n1;

		a = elt[0];
		b = elt[1];
		g = gcd_INT(a, n);
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


void a_domain::divide(INT *elt_a, INT *elt_b, INT *elt_c, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::divide" << endl;
		}
	if (kind == domain_the_integers) {
		INT g;

		if (elt_b[0] == 0) {
			cout << "a_domain::divide division by zero" << endl;
			exit(1);
			}
		g = gcd_INT(elt_a[0], elt_b[0]);
		elt_c[0] = elt_a[0] / g;
		}
	else if (kind == domain_integer_fractions) {
		INT at, ab, bt, bb, ct, cb;

		at = elt_a[0];
		ab = elt_a[1];
		bt = elt_b[0];
		bb = elt_b[1];

		if (bt == 0) {
			cout << "a_domain::divide division by zero" << endl;
			exit(1);
			}

		INT_mult_fractions(at, ab, bb, bt, ct, cb, 0 /* verbose_level */);

		elt_c[0] = ct;
		elt_c[1] = cb;
		}
	else {
		cout << "a_domain::divide unknown domain kind" << endl;
		exit(1);
		}
}

void a_domain::inverse(INT *elt_a, INT *elt_b, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "a_domain::inverse" << endl;
		}
	if (kind == domain_the_integers) {
		INT a, av;
		
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
		INT at, ab;

		
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


void a_domain::print(INT *elt)
{
	if (kind == domain_the_integers) {
		cout << elt[0];
		}
	else if (kind == domain_integer_fractions) {
		INT at, ab;

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

void a_domain::print_vector(INT *elt, INT n)
{
	INT i;
	
	cout << "(";
	for (i = 0; i < n; i++) {
		print(offset(elt, i));
		if (i < n - 1) {
			cout << ", ";
			}
		}
	cout << ")";
}

void a_domain::print_matrix(INT *A, INT m, INT n)
{
	INT i, j;
	
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

void a_domain::print_matrix_for_maple(INT *A, INT m, INT n)
{
	INT i, j;
	
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

void a_domain::make_element_from_integer(INT *elt, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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
		cout << "a_domain::make_element_from_integer unknown domain kind" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "a_domain::make_element_from_integer done" << endl;
		}
}

INT *a_domain::offset(INT *A, INT i)
{
	return A + i * size_of_instance_in_INT;
}

INT a_domain::Gauss_echelon_form(INT *A, INT f_special, INT f_complete, INT *base_cols, 
	INT f_P, INT *P, INT m, INT n, INT Pn, INT verbose_level)
// returns the rank which is the number of entries in base_cols
// A is a m x n matrix,
// P is a m x Pn matrix (if f_P is TRUE)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT rank, i, j, k, jj;
	INT *pivot, *pivot_inv;
	INT *a, *b, *c, *z, *f;
	
	if (f_v) {
		cout << "Gauss algorithm for matrix:" << endl;
		//print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_tables();
		print_matrix(A, m, n);
		}

	pivot = NEW_INT(size_of_instance_in_INT);
	pivot_inv = NEW_INT(size_of_instance_in_INT);
	a = NEW_INT(size_of_instance_in_INT);
	b = NEW_INT(size_of_instance_in_INT);
	c = NEW_INT(size_of_instance_in_INT);
	z = NEW_INT(size_of_instance_in_INT);
	f = NEW_INT(size_of_instance_in_INT);

	i = 0;
	for (j = 0; j < n; j++) {
		if (f_vv) {
			cout << "j=" << j << endl;
			}
		/* search for pivot element: */
		for (k = i; k < m; k++) {
			if (!is_zero(offset(A, k * n + j), 0)) {
				if (f_vv) {
					cout << "i=" << i << " pivot found in " << k << "," << j << endl;
					}
				// pivot element found: 
				if (k != i) {
					swap_vector(offset(A, i * n), offset(A, k * n), n, 0);
					if (f_P) {
						swap_vector(offset(P, i * Pn), offset(P, k * Pn), Pn, 0);
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
			cout << "row " << i << " pivot in row " << k << " colum " << j << endl;
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
			cout << "doing elimination in column " << j << " from row " << i + 1 << " to row " << m - 1 << ":" << endl;
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
					cout << "eliminating row " << k <<  " column " << jj << endl;
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

	FREE_INT(pivot);
	FREE_INT(pivot_inv);
	FREE_INT(a);
	FREE_INT(b);
	FREE_INT(c);
	FREE_INT(z);
	FREE_INT(f);
	
	if (f_v) { 
		cout << endl;
		print_matrix(A, m, n);
		cout << "the rank is " << rank << endl;
		}
	return rank;
}


void a_domain::Gauss_step(INT *v1, INT *v2, INT len, INT idx, INT verbose_level)
// afterwards: v2[idx] = 0 and v1,v2 span the same space as before
// v1 is not changed if v1[idx] is nonzero
{
	INT i;
	INT f_v = (verbose_level >= 1);
	INT *tmp1;
	INT *tmp2;
	
	if (f_v) {
		cout << "Gauss_step" << endl;
		}

	tmp1 = NEW_INT(size_of_instance_in_INT);
	tmp2 = NEW_INT(size_of_instance_in_INT);

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

	FREE_INT(tmp1);
	FREE_INT(tmp2);
}

void a_domain::matrix_get_kernel(INT *M, INT m, INT n, INT *base_cols, INT nb_base_cols, 
	INT &kernel_m, INT &kernel_n, INT *kernel, INT verbose_level)
// kernel must point to the appropriate amount of memory! (at least n * (n - nb_base_cols) INT's)
// kernel is stored as column vectors, i.e. kernel_m = n and kernel_n = n - nb_base_cols.
{
	INT f_v = (verbose_level >= 1);
	INT r, k, i, j, ii, jj;
	INT *kernel_cols;
	INT *zero;
	INT *m_one;
	
	if (f_v) {
		cout << "a_domain::matrix_get_kernel" << endl;
		}
	zero = NEW_INT(size_of_instance_in_INT);
	m_one = NEW_INT(size_of_instance_in_INT);

	make_zero(zero, 0);
	make_one(m_one, 0);
	negate(m_one, 0);


	r = nb_base_cols;
	k = n - r;
	kernel_m = n;
	kernel_n = k;
	
	kernel_cols = NEW_INT(k);

	INT_vec_complement(base_cols, kernel_cols, n, nb_base_cols);
	

	for (i = 0; i < r; i++) {
		ii = base_cols[i];
		for (j = 0; j < k; j++) {
			jj = kernel_cols[j];

			copy(offset(M, i * n + jj), offset(kernel, ii * kernel_n + j), 0);
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
	

	FREE_INT(kernel_cols);
	FREE_INT(zero);
	FREE_INT(m_one);
	if (f_v) {
		cout << "a_domain::matrix_get_kernel done" << endl;
		}
}

void a_domain::matrix_get_kernel_as_row_vectors(INT *M, INT m, INT n, INT *base_cols, INT nb_base_cols, 
	INT &kernel_m, INT &kernel_n, INT *kernel, INT verbose_level)
// kernel must point to the appropriate amount of memory! (at least n * (n - nb_base_cols) INT's)
// kernel is stored as row vectors, i.e. kernel_m = n - nb_base_cols and kernel_n = n.
{
	INT f_v = (verbose_level >= 1);
	INT r, k, i, j, ii, jj;
	INT *kernel_cols;
	INT *zero;
	INT *m_one;
	
	if (f_v) {
		cout << "a_domain::matrix_get_kernel_as_row_vectors" << endl;
		}
	zero = NEW_INT(size_of_instance_in_INT);
	m_one = NEW_INT(size_of_instance_in_INT);

	make_zero(zero, 0);
	make_one(m_one, 0);
	negate(m_one, 0);


	r = nb_base_cols;
	k = n - r;
	kernel_m = k;
	kernel_n = n;
	
	kernel_cols = NEW_INT(k);

	INT_vec_complement(base_cols, kernel_cols, n, nb_base_cols);
	

	for (i = 0; i < r; i++) {
		ii = base_cols[i];
		for (j = 0; j < k; j++) {
			jj = kernel_cols[j];

			copy(offset(M, i * n + jj), offset(kernel, j * kernel_n + ii), 0);
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
	

	FREE_INT(kernel_cols);
	FREE_INT(zero);
	FREE_INT(m_one);
	if (f_v) {
		cout << "a_domain::matrix_get_kernel_as_row_vectors done" << endl;
		}
}

void a_domain::get_image_and_kernel(INT *M, INT n, INT &rk, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_special = FALSE;
	INT f_complete = TRUE;
	INT *base_cols;
	INT f_P = FALSE;
	INT kernel_m, kernel_n;
	
	if (f_v) {
		cout << "a_domain::get_image_and_kernel" << endl;
		}

	base_cols = NEW_INT(n);
	rk = Gauss_echelon_form(M, f_special, f_complete, base_cols, 
		f_P, NULL, n, n, n, verbose_level);

	matrix_get_kernel_as_row_vectors(M, n, n, base_cols, rk, 
		kernel_m, kernel_n, offset(M, rk * n), verbose_level);

	if (f_v) {
		cout << "a_domain::get_image_and_kernel M=" << endl;
		print_matrix(M, n, n);
		}

	FREE_INT(base_cols);
	if (f_v) {
		cout << "a_domain::get_image_and_kernel done" << endl;
		}
}

void a_domain::complete_basis(INT *M, INT m, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_special = FALSE;
	INT f_complete = TRUE;
	INT f_P = FALSE;
	INT *M1;
	INT *base_cols;
	INT *kernel_cols;
	INT i, j, k, a, rk;
	
	if (f_v) {
		cout << "a_domain::complete_basis" << endl;
		}

	M1 = NEW_INT(m * n * size_of_instance_in_INT);
	copy_vector(M, M1, m * n, 0);
	
	base_cols = NEW_INT(n);

	rk = Gauss_echelon_form(M1, f_special, f_complete, base_cols, 
		f_P, NULL, m, n, n, verbose_level);

	if (rk != m) {
		cout << "a_domain::complete_basis rk != m" << endl;
		exit(1);
		}
	
	k = n - rk;

	kernel_cols = NEW_INT(k);

	INT_vec_complement(base_cols, kernel_cols, n, rk);
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

	FREE_INT(base_cols);
	FREE_INT(kernel_cols);
	FREE_INT(M1);
	if (f_v) {
		cout << "a_domain::complete_basis done" << endl;
		}
}

void a_domain::mult_matrix(INT *A, INT *B, INT *C, INT ma, INT na, INT nb, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, k;
	INT *D, *E;
	
	if (f_v) {
		cout << "a_domain::mult_matrix" << endl;
		}
	if (f_vv) {
		cout << "a_domain::mult_matrix A=" << endl;
		print_matrix(A, ma, na);
		cout << "a_domain::mult_matrix B=" << endl;
		print_matrix(B, na, nb);
		}
	D = NEW_INT(size_of_instance_in_INT);
	E = NEW_INT(size_of_instance_in_INT);
	for (i = 0; i < ma; i++) {
		for (k = 0; k < nb; k++) {
			make_zero(D, 0);
			for (j = 0; j < na; j++) {
				mult(offset(A, i * na + j), offset(B, j * nb + k), E, 0);
				add_apply(D, E, 0);
				}
			copy(D, offset(C, i * nb + k), 0);
			}
		}
	FREE_INT(D);
	FREE_INT(E);
	if (f_vv) {
		cout << "a_domain::mult_matrix C=" << endl;
		print_matrix(C, ma, nb);
		}
	if (f_v) {
		cout << "a_domain::mult_matrix done" << endl;
		}
}

void a_domain::mult_matrix3(INT *A, INT *B, INT *C, INT *D, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *T;
	
	if (f_v) {
		cout << "a_domain::mult_matrix3" << endl;
		}
	T = NEW_INT(n * n * size_of_instance_in_INT);
	mult_matrix(A, B, T, n, n, n, 0);
	mult_matrix(T, C, D, n, n, n, 0);
	FREE_INT(T);
	if (f_v) {
		cout << "a_domain::mult_matrix3 done" << endl;
		}
}

void a_domain::add_apply_matrix(INT *A, INT *B, INT m, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	
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

void a_domain::matrix_mult_apply_scalar(INT *A, INT *s, INT m, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	
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

void a_domain::make_block_matrix_2x2(INT *Mtx, INT n, INT k, INT *A, INT *B, INT *C, INT *D, INT verbose_level)
// A is k x k, B is k x (n - k), C is (n - k) x k, D is (n - k) x (n - k), Mtx is n x n
{
	INT f_v = (verbose_level >= 1);
	INT i, j, r;
	
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

void a_domain::make_identity_matrix(INT *A, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
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

void a_domain::matrix_inverse(INT *A, INT *Ainv, INT n, INT verbose_level)
{
	INT *T, *basecols;
	
	T = NEW_INT(n * n * size_of_instance_in_INT);
	basecols = NEW_INT(n * size_of_instance_in_INT);
	
	matrix_invert(A, T, basecols, Ainv, n, verbose_level);
	
	FREE_INT(T);
	FREE_INT(basecols);
}

void a_domain::matrix_invert(INT *A, INT *T, INT *basecols, INT *Ainv, INT n, INT verbose_level)
// T[n * n]
// basecols[n]
{
	INT rk;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "a_domain::matrix_invert" << endl;
		}
	copy_vector(A, T, n * n, 0);
	make_identity_matrix(Ainv, n, 0);
	rk = Gauss_echelon_form(T, FALSE /* f_special */, TRUE /* f_complete */, basecols, 
		TRUE /* f_P */, Ainv, n, n, n, verbose_level - 2);
	if (rk < n) {
		cout << "a_domain::matrix_invert not invertible" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "a_domain::matrix_invert done" << endl;
		}
}


