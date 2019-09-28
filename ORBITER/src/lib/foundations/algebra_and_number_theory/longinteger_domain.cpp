// longinteger_domain.cpp
//
// Anton Betten
//
// started as longinteger.cpp:  October 26, 2002
// moved here: January 23, 2015



#include "foundations.h"

using namespace std;


#define TABLE_Q_BINOMIALS_MAX 200

namespace orbiter {
namespace foundations {



int longinteger_domain::compare(
		longinteger_object &a, longinteger_object &b)
{
	int r;
	
	if (a.sign() != b.sign()) {
		if (a.sign())
			return -1;
		else
			return 1;
		}
	r = compare_unsigned(a, b);
	if (a.sign())
		return -1 * r;
	else
		return r;
}

int longinteger_domain::compare_unsigned(
		longinteger_object &a, longinteger_object &b)
// returns -1 if a < b, 0 if a = b,
// and 1 if a > b, treating a and b as unsigned.
{
	int i, l;
	char ai, bi;
	
	l = MAXIMUM(a.len(), b.len());
	for (i = l - 1; i >= 0; i--) {
		if (i < a.len())
			ai = a.rep()[i];
		else
			ai = 0;
		if (i < b.len())
			bi = b.rep()[i];
		else
			bi = 0;
		if (ai < bi)
			return -1;
		else if (ai > bi)
			return 1;
		}
	return 0;
}

void longinteger_domain::subtract_signless(
		longinteger_object &a, longinteger_object &b,
		longinteger_object &c)
// c = a - b, assuming a > b
{
	int i;
	char ai, bi, carry;
	
	c.freeself();
	c.sign() = FALSE;
	c.len() = a.len();
	c.rep() = NEW_char(c.len());
	for (i = 0; i < c.len(); i++)
		c.rep()[i] = 0;

	carry = 0;
	for (i = 0; i < a.len(); i++) {
		if (i < b.len())
			bi = b.rep()[i];
		else
			bi = 0;
		bi += carry;
		ai = a.rep()[i];
		if (bi > ai) {
			ai += 10;
			carry = 1;
			}
		else
			carry = 0;
		c.rep()[i] = ai - bi;
		}
	c.normalize();
}

void longinteger_domain::subtract_signless_in_place(
		longinteger_object &a, longinteger_object &b)
// a := a - b, assuming a > b
{
	int i;
	char ai, bi, carry;
	
	carry = 0;
	for (i = 0; i < a.len(); i++) {
		if (i < b.len())
			bi = b.rep()[i];
		else
			bi = 0;
		bi += carry;
		ai = a.rep()[i];
		if (bi > ai) {
			ai += 10;
			carry = 1;
			}
		else
			carry = 0;
		a.rep()[i] = ai - bi;
		}
	//c.normalize();
}

void longinteger_domain::add(
		longinteger_object &a, longinteger_object &b,
		longinteger_object &c)
{
	int cmp, carry, i, ai, bi, ci;
	
	c.freeself();
	c.len() = MAXIMUM(a.len(), b.len()) + 1;
	if ((a.sign() && b.sign()) || (!a.sign() && !b.sign())) {
		c.sign() = a.sign();
		}
	else {
		// mixed signs: subtraction 
		cmp = compare_unsigned(a, b);
		if (cmp < 0) {
			// |a| < |b|
			
			subtract_signless(b, a, c);
			c.sign() = b.sign();
			c.normalize();
			return;
			}
		else if (cmp > 0) {  
			// |a| > |b|
		
			subtract_signless(a, b, c);
			c.sign() = a.sign();
			c.normalize();
			return;
			}
		else {
			// |a| = |b|
			c.zero();
			return;
			}
		}
	c.rep() = NEW_char(c.len());
	for (i = 0; i < c.len(); i++) {
		c.rep()[i] = 0;
		}

	carry = 0;
	for (i = 0; i < c.len(); i++) {
		if (i < a.len())
			ai = a.rep()[i];
		else
			ai = 0;
		if (i < b.len())
			bi = b.rep()[i];
		else
			bi = 0;
		ci = ai + bi + carry;
		if (ci >= 10)
			carry = 1;
		else
			carry = 0;
		c.rep()[i] = ci % 10;
		}
	c.normalize();
}

void longinteger_domain::add_in_place(
		longinteger_object &a, longinteger_object &b)
// a := a + b
{
	longinteger_object C;
	
	add(a, b, C);
	C.assign_to(a);
}


void longinteger_domain::mult(
		longinteger_object &a, longinteger_object &b,
		longinteger_object &c)
{
	int i, j;
	char ai, bj, d, carry;
	int f_v = FALSE;
	
	if (a.is_zero() || b.is_zero()) {
		c.create(0);
		return;
		}
	if ((a.sign() && !b.sign()) || (!a.sign() && b.sign())) {
		c.sign() = TRUE;
		}
	else {
		c.sign() = FALSE;
		}
	
	c.freeself();
	c.len() = a.len() + b.len() + 2;
	c.rep() = NEW_char(c.len());
	for (i = 0; i < c.len(); i++) {
		c.rep()[i] = 0;
		}
	
	if (f_v) {
		cout << "longinteger_domain::mult a=";
		longinteger_print_digits(a.rep(), a.len());
		cout << "b=";
		longinteger_print_digits(b.rep(), b.len());
		cout << endl;
		}
	for (j = 0; j < b.len(); j++) {
		bj = b.rep()[j];
		carry = 0;
		for (i = 0; i < a.len(); i++) {
			ai = a.rep()[i];
			d = ai * bj + carry + c.rep()[i + j];
			if (d >= 100) {
				cout << "longinteger:mult error: d >= 100" << endl;
				exit(1);
				}
			carry = d / 10;
			c.rep()[i + j] = d % 10;
			if (f_v) {
				cout << "c[" << i + j << "]=" << d % 10 << "="
						<< (char)('0' + c.rep()[i + j]) << endl;
				}
			}
		if (carry) {
			c.rep()[j + a.len()] = carry;
			if (f_v) {
				cout << "c[" << j + a.len() << "]=" << carry << "="
						<< (char)('0' + carry) << endl;
				}
			}
		}
	if (f_v) {
		cout << "longinteger_domain::mult c=";
		longinteger_print_digits(c.rep(), c.len());
		cout << endl;
		}
	c.normalize();
	if (f_v) {
		cout << "longinteger_domain::mult after normalize, c=";
		longinteger_print_digits(c.rep(), c.len());
		cout << endl;
		}
}

void longinteger_domain::mult_in_place(
		longinteger_object &a, longinteger_object &b)
{
	longinteger_object C;
	
	mult(a, b, C);
	C.assign_to(a);
}


void longinteger_domain::mult_integer_in_place(
		longinteger_object &a, int b)
{
	longinteger_object B, C;
	
	B.create(b);
	mult(a, B, C);
	C.assign_to(a);
}

static int do_division(longinteger_domain &D,
		longinteger_object &r, longinteger_object table[10])
{
	int i, cmp;
	
	for (i = 9; i >= 0; i--) {
		cmp = D.compare(r, table[i]);
		if (cmp >= 0)
			return i;
		}
	cout << "do_division we should never reach this point" << endl;
	exit(1);
}

void longinteger_domain::mult_mod(longinteger_object &a, 
	longinteger_object &b, longinteger_object &c, 
	longinteger_object &m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, l;
	char ai, bj, d, carry;
	//longinteger_object table[10], cc;
	longinteger_object q, r, c0;
	
	if (f_v ) {
		cout << "longinteger_domain::mult_mod "
				"a=" << a << " b=" << b << " m=" << m << endl;
		}
	if (a.is_zero() || b.is_zero()) {
		c.create(0);
		return;
		}
	if (compare_unsigned(a, m) >= 0) {
		cout << a << " * " << b << endl;
		cout << "longinteger_domain::mult_mod a >= m" << endl;
		exit(1);
		}
	if (compare_unsigned(b, m) >= 0) {
		cout << a << " * " << b << endl;
		cout << "longinteger_domain::mult_mod b >= m" << endl;
		exit(1);
		}
	if ((a.sign() && !b.sign()) || (!a.sign() && b.sign()))
		c.sign() = TRUE;
	else
		c.sign() = FALSE;

#if 0
	for (i = 0; i < 10; i++) {
		cc.create(i);
		mult(m, cc, table[i]);
		}
#endif
	
	if (f_v) {
		cout << "longinteger_domain::mult_mod "
				"calling c.freeself" << endl;
		}
	c.freeself();
	if (f_v) {
		cout << "longinteger_domain::mult_mod"
				"after c.freeself" << endl;
		}
	l = m.len();
	c.len() = (int) (l + 2);
	c.rep() = NEW_char(c.len());
	for (i = 0; i < c.len(); i++)
		c.rep()[i] = 0;
	c.assign_to(c0);
	
	for (j = b.len() - 1; j >= 0; j--) {
		if (f_vv) {
			cout << "j=" << j << endl;
			}
		bj = b.rep()[j];
		carry = 0;
		for (i = 0; i < l; i++) {
			if (i < a.len())
				ai = a.rep()[i];
			else 
				ai = 0;
			d = ai * bj + carry + c.rep()[i];
			//cout << (int) ai << " * " << (int) bj << " + "
			// << (int)carry << " + " << (int) c.rep()[i]
			// << " = " << (int)d << endl;
			if (d >= 100) {
				cout << "longinteger:mult_mod error: d >= 100" << endl;
				exit(1);
				}
			carry = d / 10;
			c.rep()[i] = d % 10;
			}
		if (carry) {
			d = c.rep()[i] + carry;
			carry = d / 10;
			c.rep()[i] = d % 10;
			i++;
			}
		if (carry) {
			d = c.rep()[i] + carry;
			carry = d / 10;
			c.rep()[i] = d % 10;
			if (carry) {
				cout << "longinteger_domain::mult_mod "
						"error: carry" << endl;
				exit(1);
				}
			}
		if (f_vv) {
			cout << "longinteger_domain::mult_mod "
					"c=" << c << " len " << c.len() << endl;
			}
		integral_division(c, m, q, r, 0/*verbose_level - 1*/);
		//h = do_division(*this, c, table);
		//subtract_signless_in_place(c, table[h]);
		if (f_vv) {
			cout << "longinteger_domain::mult_mod "
					"r=" << r << " len " << r.len() << endl;
			}
		c0.assign_to(c);
		if (f_vv) {
			cout << "longinteger_domain::mult_mod"
					"c=c0=" << c << " len " << c.len() << endl;
			}
		if (j) {
			c.rep()[0] = 0;
			for (i = 0; i < r.len(); i++) {
				if (i + 1 < c.len()) {
					c.rep()[i + 1] = r.rep()[i];
					}
				}
			i++;
			for (; i < c.len(); i++)
				c.rep()[i] = 0;
			}
		else {
			for (i = 0; i < r.len(); i++) {
				if (i < c.len()) {
					c.rep()[i] = r.rep()[i];
					}
				}
			for (; i < c.len(); i++)
				c.rep()[i] = 0;
			}
		if (f_vv) {
			cout << "c=" << c << " len " << c.len() << endl;
			}
		}
	c.normalize();
	if (f_v) {
		cout << "longinteger_domain::mult_mod " << a << " * " << b
				<< " = " << c << " mod " << m << endl;
		}
}

void longinteger_domain::multiply_up(
		longinteger_object &a, int *x, int len, int verbose_level)
{
	longinteger_object b, c;
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "longinteger_domain::multiply_up" << endl;
		}
	a.one();
	if (f_v) {
		cout << "longinteger_domain::multiply_up "
				"a=" << a << endl;
		}
	for (i = 0; i < len; i++) {
		if (x[i] == 1) {
			continue;
		}
		b.create(x[i]);
		if (f_v) {
			cout << "longinteger_domain::multiply_up "
					"i=" << i << " x[i]=" << x[i]
					<< " b=" << b << endl;
			}
		mult(a, b, c);
		if (f_v) {
			cout << "longinteger_domain::multiply_up "
					"i=" << i << " x[i]=" << x[i]
					<< " c=" << c << endl;
			}
		c.assign_to(a);
		if (f_v) {
			cout << "longinteger_domain::multiply_up "
					"i=" << i << " x[i]=" << x[i]
					<< " a=" << a << endl;
			}
		//cout << "*" << x[i] << "=" << a << endl;
		}
	if (f_v) {
		cout << "longinteger_domain::multiply_up done" << endl;
		}
}

int longinteger_domain::quotient_as_int(
		longinteger_object &a, longinteger_object &b)
{
	longinteger_object q, r;

	integral_division(a, b, q, r, 0);
	return q.as_int();
}

void longinteger_domain::integral_division_exact(
		longinteger_object &a, longinteger_object &b,
		longinteger_object &a_over_b)
{
	longinteger_object r;

	integral_division(a, b, a_over_b, r, 0);
	if (!r.is_zero()) {
		cout << "longinteger_domain::integral_division_exact "
				"b does not divide a" << endl;
		exit(1);
		}
}

void longinteger_domain::integral_division(
	longinteger_object &a, longinteger_object &b, 
	longinteger_object &q, longinteger_object &r,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object table[10], c;
	int i, l, ql;
	
	if (f_v) {
		cout << "longinteger_domain::integral_division "
				"dividing a=" << a << " by b=" << b << endl;
		}
	if (a.sign()) {
		cout << "longinteger_domain::integral_division "
				"a is negative" << endl;
		exit(1);
		}
	if (b.sign()) {
		cout << "longinteger_domain::integral_division "
				"b is negative" << endl;
		exit(1);
		}
	if (compare(a, b) == -1) {
		q.zero();
		a.assign_to(r);
		return;
		}
	
	for (i = 0; i < 10; i++) {
		c.create(i);
		mult(b, c, table[i]);
		}
	q.freeself();
	r.freeself();
	
	// load r with leading b.len() digits of a: 
	r.sign() = FALSE;
	r.len() = b.len() + 1;
	r.rep() = NEW_char(r.len());
	l = a.len() - b.len();
	for (i = 0; i < b.len(); i++) {
		r.rep()[i] = a.rep()[l + i];
		}
	r.rep()[b.len()] = 0;
	
	// allocate q of length a.len() - b.len() + 1
	q.sign() = FALSE;
	q.len() = a.len() - b.len() + 1;
	q.rep() = NEW_char(q.len());
	for (i = 0; i < q.len(); i++) {
		q.rep()[i] = 0;
		}
	
	// main loop containing the divisions:
	for ( ; l >= 0; l--) {
		ql = do_division(*this, r, table);
		q.rep()[l] = ql;
		subtract_signless(r, table[ql], c);
		if (f_v) {
			cout << "l=" << l << " r=" << r
					<< " ql=" << ql << " c=" << c << endl;
			}
		
		if (l == 0)
			break;
		
		// put c into r, shift up by one digit 
		// and append the next digit a[l-1].
		for (i = 0; i < c.len(); i++)
			r.rep()[i + 1] = c.rep()[i];
		for (i++ ; i < r.len(); i++)
			r.rep()[i] = 0;
		r.rep()[0] = a.rep()[l - 1];
		}
	c.assign_to(r);
	r.normalize();
	q.normalize();
}

void longinteger_domain::integral_division_by_int(
	longinteger_object &a,
	int b, longinteger_object &q, int &r)
{
	longinteger_object B, R;
	int verbose_level = 0;

	B.create(b);
	integral_division(a, B, q, R, verbose_level);
	r = R.as_int();
}

void longinteger_domain::extended_gcd(
	longinteger_object &a,
	longinteger_object &b, longinteger_object &g, 
	longinteger_object &u, longinteger_object &v, int verbose_level)
// the gcd computed here is always nonnegative
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object q, rm1, r, rp1, sm1, s, sp1, tm1, t, tp1, x, y;
	int c;
	
	if (f_v) {
		cout << "longinteger_domain::extended_gcd "
				"a=" << a << " b=" << b << endl;
		}
	if (a.is_zero()) {
		b.assign_to(g);
		u.create(0);
		v.create(1);
		if (g.sign()) {
			g.negate();
			u.negate();
			v.negate();
			}
		return;
		}
	if (b.is_zero()) {
		a.assign_to(g);
		u.create(1);
		v.create(0);
		if (g.sign()) {
			g.negate();
			u.negate();
			v.negate();
			}
		return;
		}
	c = compare_unsigned(a, b);
	if (c < 0) { // |a| < |b|
		extended_gcd(b, a, g, v, u, verbose_level);
		return;
		}
	if (a.sign()) {
		a.negate();
		extended_gcd(a, b, g, u, v, verbose_level);
		a.negate();
		u.negate();
		if (g.sign()) {
			g.negate();
			u.negate();
			v.negate();
			}
		return;
		}
	if (b.sign()) {
		b.negate();
		extended_gcd(a, b, g, u, v, verbose_level);
		b.negate();
		v.negate();
		if (g.sign()) {
			g.negate();
			u.negate();
			v.negate();
			}
		return;
		}
	// now a > 0, b > 0 and a >= b
	a.assign_to(rm1);
	b.assign_to(r);
	sm1.create(1);
	tm1.create(0);
	s.create(0);
	t.create(1);
	while (TRUE) {
		integral_division(rm1, r, q, rp1, verbose_level - 1);
		if (rp1.is_zero()) {
			r.assign_to(g);
			s.assign_to(u);
			t.assign_to(v);
			return;
			}
		if (f_vv) {
			rm1.print(cout);
			cout << " = ";
			q.print(cout);
			cout << " * ";
			r.print(cout);
			cout << " + ";
			rp1.print(cout);
			cout << endl;
			}
		q.negate();
		mult(q, s, x);
		mult(q, t, y);
		add(sm1, x, sp1);
		add(tm1, y, tp1);
		r.swap_with(rm1);
		s.swap_with(sm1);
		t.swap_with(tm1);
		rp1.swap_with(r);
		sp1.swap_with(s);
		tp1.swap_with(t);
		if (f_vv) {
			r.print(cout);
			cout << " = ";
			s.print(cout);
			cout << " * ";
			a.print(cout);
			cout << " + ";
			t.print(cout);
			cout << " * ";
			b.print(cout);
			cout << endl;
			}
		}
}

int longinteger_domain::logarithm_base_b(
		longinteger_object &a, int b)
{
	int r, l = 0;
	longinteger_object a1, a2;
	
	a.assign_to(a1);
	a1.normalize();
	while (!a1.is_zero()) {
		integral_division_by_int(a1, b, a2, r);
		l++;
		a2.assign_to(a1);
		}
	return l;
}

void longinteger_domain::base_b_representation(
		longinteger_object &a, int b, int *&rep, int &len)
{
	int i, r;
	longinteger_object a1, a2;
	
	a.assign_to(a1);
	a1.normalize();
	len = 0;
	while (!a1.is_zero()) {
		integral_division_by_int(a1, b, a2, r);
		len++;
		a2.assign_to(a1);
		}
	a.assign_to(a1);
	a1.normalize();
	rep = NEW_int(len);
	for (i = 0; i < len; i++) {
		integral_division_by_int(a1, b, a2, r);
		rep[i] = r;
		a2.assign_to(a1);
		}
	for (i = 0; i < len; i++) {
		cout << rep[i] << " ";
		}
	cout << endl;
}

void longinteger_domain::power_int(
		longinteger_object &a, int n)
{
	longinteger_object b, c, d;
	
	a.assign_to(b);
	c.one();
	while (n) {
		if (n % 2) {
			mult(b, c, d);
			d.assign_to(c);
			}
		mult(b, b, d);
		d.assign_to(b);
		n >>= 1;
		}
	c.assign_to(a);
}

void longinteger_domain::power_int_mod(
		longinteger_object &a, int n, longinteger_object &m)
{
	longinteger_object b, c, d;
	
	a.assign_to(b);
	c.one();
	while (n) {
		if (n % 2) {
			mult_mod(b, c, d, m, 0);
			d.assign_to(c);
			}
		mult_mod(b, b, d, m, 0);
		d.assign_to(b);
		n >>= 1;
		}
	c.assign_to(a);
}

void longinteger_domain::power_longint_mod(
	longinteger_object &a,
	longinteger_object &n, longinteger_object &m,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object b, c, d, n1;
	int r;
	
	a.assign_to(b);
	c.one();
	if (f_v) {
		cout << "longinteger_domain::power_longint_mod" << endl;
		cout << "computing " << b << " to the power "
				<< n << " mod " << m << ":" << endl;
		}
	while (!n.is_zero()) {
		if (f_vv) {
			cout << "n=" << n << " : " << b << "^"
					<< n << " * " << c << endl;
			}
		integral_division_by_int(n, 2, n1, r);
		if (f_vv) {
			cout << "after division by 2, n1=" << n1
					<< " r=" << r << endl;
			}
		if (r) {
			mult_mod(b, c, d, m, verbose_level);
			if (f_vv) {
				cout << b << " * " << c << " = " << d << endl;
				}
			d.assign_to(c);
			}
		mult_mod(b, b, d, m, verbose_level);
		if (f_vv) {
			cout << b << "^2 = " << d << endl;
			}
		d.assign_to(b);
		n1.assign_to(n);
		}
	c.assign_to(a);
}

void longinteger_domain::create_qnm1(
		longinteger_object &a, int q, int n)
// create (q^n - 1)
{
	longinteger_object b, c;
	
	b.create(q);
	power_int(b, n);
	c.create(-1);
	add(b, c, a);
}

#define TABLE_BINOMIALS_MAX 1000

static longinteger_object *tab_binomials = NULL;
static int tab_binomials_size = 0;


static void binomial_with_table(
		longinteger_object &a, int n, int k)
{
	int i, j;
	longinteger_domain D;
	
	if (k < 0 || k > n) {
		a.create(0);
		return;
		}
	if (k == n) {
		a.create(1);
		return;
		}
	if (k == 0) {
		a.create(1);
		return;
		}

	// reallocate table if necessary:
	if (n >= tab_binomials_size) {
		//cout << "binomial_with_table
		// reallocating table to size " << n + 1 << endl;
		longinteger_object *tab_binomials2 =
			NEW_OBJECTS(longinteger_object, (n + 1) * (n + 1));
		for (i = 0; i < tab_binomials_size; i++) {
			for (j = 0; j <= i; j++) {
				tab_binomials[i * tab_binomials_size +
					j].swap_with(tab_binomials2[i * (n + 1) + j]);
				}
			}
		for ( ; i <= n; i++) {
			for (j = 0; j <= i; j++) {
				tab_binomials2[i * (n + 1) + j].create(0);
				}
			}
		if (tab_binomials) {
			FREE_OBJECTS(tab_binomials);
		}
		tab_binomials = tab_binomials2;
		tab_binomials_size = n + 1;
#if 0
		for (i = 0; i < tab_binomials_size; i++) {
			for (j = 0; j <= i; j++) {
				tab_binomials2[i * (n + 1) + j].print(cout);
				cout << " ";
				}
			cout << endl;
			}
		cout << endl;
#endif
		}
	if (tab_binomials[n * tab_binomials_size + k].is_zero()) {
		longinteger_object b, c, d;
		int r;
		
		binomial_with_table(b, n, k - 1);
		//cout << "recursion, binom " << n << ", " << k - 1 << " = ";
		//b.print(cout);
		//cout << endl;
		
		c.create(n - k + 1);
		D.mult(b, c, d);
		D.integral_division_by_int(d, k, a, r);
		if (r != 0) {
			cout << "longinteger_domain.cpp: binomial_with_table k != 0" << endl;
			exit(1);
			}
		a.assign_to(tab_binomials[n * tab_binomials_size + k]);
		//cout << "new table entry n=" << n << " k=" << k << " : ";
		//a.print(cout);
		//cout << " ";
		//tab_binomials[n * tab_binomials_size + k].print(cout);
		//cout << endl;
		}
	else {
		tab_binomials[n * tab_binomials_size + k].assign_to(a);
		}
}

void longinteger_domain::binomial(
	longinteger_object &a, int n, int k,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object b, c, d;
	int r;
	
	if (f_v) {
		cout << "longinteger_domain::binomial "
				"n=" << n << " k=" << k << endl;
		}
	if (k < 0 || k > n) {
		a.create(0);
		return;
		}
	if (k == n) {
		a.create(1);
		return;
		}
	if (k == 0) {
		a.create(1);
		return;
		}
	if (n < TABLE_BINOMIALS_MAX) {
		if (f_v) {
			cout << "longinteger_domain::binomial "
					"using table" << endl;
			}
		binomial_with_table(a, n, k);
		return;
		}
	else {
		binomial(b, n, k - 1, verbose_level);
		}
	c.create(n - k + 1);
	mult(b, c, d);
	integral_division_by_int(d, k, a, r);
	if (r != 0) {
		cout << "longinteger_domain::binomial "
				"k != 0" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "longinteger_domain::binomial "
				"n=" << n << " k=" << k << " done" << endl;
		}
	
}

void longinteger_domain::size_of_conjugacy_class_in_sym_n(
	longinteger_object &a, int n, int *part)
{
	longinteger_object b, c, d;
	int i, ai, j;
	
	factorial(b, n);
	for (i = 1; i <= n; i++) {
		ai = part[i - 1];
		c.create(1);
		for (j = 0; j < ai; j++) {
			mult_integer_in_place(c, i);
			}
		for (j = 1; j <= ai; j++) {
			mult_integer_in_place(c, j);
			}
		integral_division_exact(b, c, d);
		d.assign_to(b);
		}
	b.assign_to(a);
}




static longinteger_object *tab_q_binomials = NULL;
static int tab_q_binomials_size = 0;
static int tab_q_binomials_q = 0;


static void q_binomial_with_table(longinteger_object &a, 
	int n, int k, int q, int verbose_level)
{
	int i, j;
	longinteger_domain D;
	
	//cout << "q_binomial_with_table n=" << n
	// << " k=" << k << " q=" << q << endl;
	if (k < 0 || k > n) {
		a.create(0);
		return;
		}
	if (k == n) {
		a.create(1);
		return;
		}
	if (k == 0) {
		a.create(1);
		return;
		}

	// reallocate table if necessary:
	if (n >= tab_q_binomials_size) {
		if (tab_q_binomials_size > 0 && q != tab_q_binomials_q) {


			D.q_binomial_no_table(a, n, k, q, verbose_level);
			return;
#if 0
			cout << "tab_q_binomials_size > 0 && q != tab_q_binomials_q" << endl;
			cout << "q=" << q << endl;
			cout << "tab_q_binomials_q=" << tab_q_binomials_q << endl;
			exit(1);
#endif
			}
		else {
			tab_q_binomials_q = q;
			}
		//cout << "binomial_with_table
		// reallocating table to size " << n + 1 << endl;
		longinteger_object *tab_q_binomials2 =
			NEW_OBJECTS(longinteger_object, (n + 1) * (n + 1));
		for (i = 0; i < tab_q_binomials_size; i++) {
			for (j = 0; j <= i; j++) {
				tab_q_binomials[i * tab_q_binomials_size +
					j].swap_with(tab_q_binomials2[i * (n + 1) + j]);
				}
			}
		for ( ; i <= n; i++) {
			for (j = 0; j <= i; j++) {
				tab_q_binomials2[i * (n + 1) + j].create(0);
				}
			}
		if (tab_q_binomials) {
			FREE_OBJECTS(tab_q_binomials);
			}
		tab_q_binomials = tab_q_binomials2;
		tab_q_binomials_size = n + 1;
#if 0
		for (i = 0; i < tab_q_binomials_size; i++) {
			for (j = 0; j <= i; j++) {
				tab_q_binomials2[i * (n + 1) + j].print(cout);
				cout << " ";
				}
			cout << endl;
			}
		cout << endl;
#endif
		}
	if (tab_q_binomials[n * tab_q_binomials_size + k].is_zero()) {
		
		D.q_binomial_no_table(a, n, k, q, verbose_level);
		a.assign_to(tab_q_binomials[n * tab_q_binomials_size + k]);
		//cout << "new table entry n=" << n << " k=" << k << " : ";
		//a.print(cout);
		//cout << " ";
		//tab_q_binomials[n * tab_q_binomials_size + k].print(cout);
		//cout << endl;
		}
	else {
		tab_q_binomials[n * tab_q_binomials_size + k].assign_to(a);
		}
}


void longinteger_domain::q_binomial(
	longinteger_object &a,
	int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object b, c, top, bottom, r;
	
	if (f_v) {
		cout << "longinteger_domain::q_binomial "
				"n=" << n << " k=" << k << " q=" << q << endl;
		}
	if (k < 0 || k > n) {
		a.create(0);
		return;
		}
	if (k == n) {
		a.create(1);
		return;
		}
	if (k == 0) {
		a.create(1);
		return;
		}
	//cout << "longinteger_domain::q_binomial
	//n=" << n << " k=" << k << " q=" << q << endl;
	if (n < TABLE_Q_BINOMIALS_MAX) {
		q_binomial_with_table(a, n, k, q, verbose_level);
		}
	else {
		q_binomial_no_table(b, n, k, q, verbose_level);
		}
	if (f_v) {
		cout << "longinteger_domain::q_binomial "
			"n=" << n << " k=" << k << " q=" << q
			<< " yields " << a << endl;
		}
}

void longinteger_domain::q_binomial_no_table(
	longinteger_object &a,
	int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object b, c, top, bottom, r;
	
	if (f_v) {
		cout << "longinteger_domain::q_binomial_no_table "
			"n=" << n << " k=" << k << " q=" << q << endl;
		}
	if (k < 0 || k > n) {
		a.create(0);
		return;
		}
	if (k == n) {
		a.create(1);
		return;
		}
	if (k == 0) {
		a.create(1);
		return;
		}
	q_binomial_no_table(b, n - 1, k - 1, q, verbose_level);
	create_qnm1(c, q, n);
	mult(b, c, top);
	create_qnm1(bottom, q, k);
	integral_division(top, bottom, a, r, verbose_level - 1);
	if (!r.is_zero()) {
		cout << "longinteger_domain::q_binomial_no_table "
				"remainder is not zero" << endl;
		cout << "q=" << q << endl;
		cout << "n-1=" << n-1 << endl;
		cout << "k-1=" << k-1 << endl;
		cout << "top=" << top << endl;
		cout << "bottom=" << bottom << endl;
		exit(1);
		}
	if (f_v) {
		cout << "longinteger_domain::q_binomial_no_table "
			"n=" << n << " k=" << k << " q=" << q
			<< " yields " << a << endl;
		}
	
}

static longinteger_object *tab_krawtchouk = NULL;
static int *tab_krawtchouk_entry_computed = NULL;
static int tab_krawtchouk_size = 0;
static int tab_krawtchouk_n = 0;
static int tab_krawtchouk_q = 0;

static void krawtchouk_with_table(longinteger_object &a, 
	int n, int q, int k, int x)
{
	int i, j, kx;
	longinteger_domain D;
	
	if (tab_krawtchouk_size) { 
		if (n != tab_krawtchouk_n || q != tab_krawtchouk_q) {
			delete [] tab_krawtchouk;
			FREE_int(tab_krawtchouk_entry_computed);
			tab_krawtchouk_size = 0;
			tab_krawtchouk_n = 0;
			tab_krawtchouk_q = 0;
			}
		}
	kx = MAXIMUM(k, x);
	// reallocate table if necessary:
	if (kx >= tab_krawtchouk_size) {
		kx++;
		//cout << "krawtchouk_with_table
		//reallocating table to size " << kx << endl;
		longinteger_object *tab_krawtchouk2 =
				NEW_OBJECTS(longinteger_object, kx * kx);
		int *tab_krawtchouk_entry_computed2 =
				NEW_int(kx * kx);
		for (i = 0; i < kx; i++) {
			for (j = 0; j < kx; j++) {
				tab_krawtchouk_entry_computed2[i * kx + j] = FALSE;
				tab_krawtchouk2[i * kx + j].create(0);
				}
			}
		for (i = 0; i < tab_krawtchouk_size; i++) {
			for (j = 0; j < tab_krawtchouk_size; j++) {
				tab_krawtchouk[i * tab_krawtchouk_size + j
					].swap_with(tab_krawtchouk2[i * kx + j]);
				tab_krawtchouk_entry_computed2[i * kx + j] =
					tab_krawtchouk_entry_computed[
						i * tab_krawtchouk_size + j];
				}
			}
		if (tab_krawtchouk) {
			FREE_OBJECTS(tab_krawtchouk);
			}
		if (tab_krawtchouk_entry_computed) {
			FREE_int(tab_krawtchouk_entry_computed);
			}
		tab_krawtchouk = tab_krawtchouk2;
		tab_krawtchouk_entry_computed = tab_krawtchouk_entry_computed2;
		tab_krawtchouk_size = kx;
		tab_krawtchouk_n = n;
		tab_krawtchouk_q = q;
#if 0
		for (i = 0; i < tab_krawtchouk_size; i++) {
			for (j = 0; j < tab_krawtchouk_size; j++) {
				tab_krawtchouk[i * tab_krawtchouk_size + j].print(cout);
				cout << " ";
				}
			cout << endl;
			}
		cout << endl;
#endif
		}
	if (!tab_krawtchouk_entry_computed[k * tab_krawtchouk_size + x]) {
		longinteger_object n_choose_k, b, c, d, e, f;
		
		if (x < 0) {
			cout << "krawtchouk_with_table() x < 0" << endl;
			exit(1);
			}
		if (k < 0) {
			cout << "krawtchouk_with_table() k < 0" << endl;
			exit(1);
			}
		if (x == 0) {
			D.binomial(n_choose_k, n, k, FALSE);
			if (q != 1) {
				b.create(q - 1);
				D.power_int(b, k);
				D.mult(n_choose_k, b, a);
				}
			else {
				n_choose_k.assign_to(a);
				}
			}
		else if (k == 0) {
			a.create(1);
			}
		else {
			krawtchouk_with_table(b, n, q, k, x - 1);
			//cout << "K_" << k << "(" << x - 1 << ")=" << b << endl;
			c.create(-q + 1);
			krawtchouk_with_table(d, n, q, k - 1, x);
			//cout << "K_" << k - 1<< "(" << x << ")=" << d << endl;
			D.mult(c, d, e);
			//cout << " e=";
			//e.print(cout);
			D.add(b, e, c);
			//cout << " c=";
			//c.print(cout);
			d.create(-1);
			krawtchouk_with_table(e, n, q, k - 1, x - 1);
			//cout << "K_" << k - 1<< "(" << x - 1 << ")=" << e << endl;
			D.mult(d, e, f);
			//cout << " f=";
			//f.print(cout);
			//cout << " c=";
			//c.print(cout);
			D.add(c, f, a);
			//cout << " a=";
			//a.print(cout);
			//cout << endl;
			}
		
		a.assign_to(tab_krawtchouk[k * tab_krawtchouk_size + x]);
		tab_krawtchouk_entry_computed[
				k * tab_krawtchouk_size + x] = TRUE;
		//cout << "new table entry k=" << k
		// << " x=" << x << " : " << a << endl;
		}
	else {
		tab_krawtchouk[k * tab_krawtchouk_size + x].assign_to(a);
		}
}

void longinteger_domain::krawtchouk(longinteger_object &a, 
	int n, int q, int k, int x)
{	
	//cout << "krawtchouk() n=" << n << " q=" << q
	//<< " k=" << k << " x=" << x << endl;
	krawtchouk_with_table(a, n, q, k, x);
}

int longinteger_domain::is_even(longinteger_object &a)
{
	if (((a.rep()[0] % 2)) == 0)
		return TRUE;
	else
		return FALSE;
}

int longinteger_domain::is_odd(longinteger_object &a)
{
	if (is_even(a))
		return FALSE;
	else
		return TRUE;
}

int longinteger_domain::remainder_mod_int(longinteger_object &a, int p)
{
	int r;
	longinteger_object q;
	
	integral_division_by_int(a, p, q, r);
	return r;
}

int longinteger_domain::multiplicity_of_p(longinteger_object &a, 
	longinteger_object &residue, int p)
{
	int r, n = 0;
	longinteger_object q;
	
	if (a.is_zero()) {
		cout << "longinteger_domain::multiplicity_of_p a = 0" << endl;
		exit(1);
		}
	a.assign_to(residue);
	while (!residue.is_one()) {
		integral_division_by_int(residue, p, q, r);
		if (r)
			break;
		n++;
		q.assign_to(residue);
		}
	return n;
}

int longinteger_domain::smallest_primedivisor(
	longinteger_object &a,
	int p_min, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object n, n1, q, pp;
	int p, r, cnt = 0;
	
	if (f_v) {
		cout << "smallest_primedivisor " << a
				<< " p_min=" << p_min << endl;
		}
	a.assign_to(n);
	
	if (p_min == 0)
		p_min = 2;
	if (p_min < 0)
		p_min = - p_min;
	if (p_min <= 2) {
		if (is_even(n))
			return 2;
		p_min = 3;
		}
	if (p_min <= 3) {
		if (remainder_mod_int(n, 3) == 0)
			return 3;
		p_min = 5;
		}
	if (EVEN(p_min))
		p_min--;
	p = p_min;
	while (TRUE) {
		cnt++;
		if (f_vv) {
			cout << "smallest_primedivisor:n=" << n
					<< " trying p=" << p << endl;
			}
		n.assign_to(n1);
		integral_division_by_int(n1, p, q, r);
		if (f_vv && (cnt % 1) == 0) {
			cout << "smallest_primedivisor:n=" << n1
				<< " trying p=" << p << " q=" << q
				<< " r=" << r << endl;
			cnt = 0;
			}
		if (r == 0)
			return p;
		pp.create(p);
		if (compare(q, pp) < 0) {
			break;
			}
		p += 2;
		}
	if (f_v) {
		cout << "smallest_primedivisor "
				"the number is prime" << endl;
		}
	return 0;
}

void longinteger_domain::factor_into_longintegers(
	longinteger_object &a,
	int &nb_primes, longinteger_object *&primes, 
	int *&exponents, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	longinteger_object n, q, pp, r;
	longinteger_domain D;
	int p, last_prime = 0, i;
	number_theory_domain NT;
	
	if (f_v) {
		cout << "factoring " << a << endl;
		}
	if (a.is_zero()) {
		cout << "longinteger_domain::factor_into_longintegers "
				"a is zero" << endl;
		exit(1);
		}
	if (a.is_one()) {
		nb_primes = 0;
		primes = NEW_OBJECTS(longinteger_object, 1);
		exponents = NEW_int(1);
		return;
		}
	a.assign_to(n);
	p = smallest_primedivisor(n, last_prime, verbose_level);
	if (p == 0) {
		p = n.as_int();
		}
	pp.create(p);
	primes = NEW_OBJECTS(longinteger_object, 1);
	exponents = NEW_int(1);
	nb_primes = 1;
	pp.assign_to(primes[0]);
	exponents[0] = 1;
	last_prime = p;
	D.integral_division(n, pp, q, r, verbose_level - 1);
	if (!r.is_zero()) {
		cout << "longinteger_domain::factor_into_longintegers "
				"factor does not divide" << endl;
		}
	q.assign_to(n);
	while (!n.is_one()) {
		if (f_v) {
			cout << "remaining factor: " << n << endl;
			}
		p = smallest_primedivisor(n, last_prime, verbose_level);
		// if p == 0: n is prime
		
		if (p == 0) {
			p = n.as_int();
			}
		if (p == last_prime) {
			exponents[nb_primes - 1]++;
			pp.create(p);
			}
		else {
			longinteger_object *pr = NEW_OBJECTS(
				longinteger_object, nb_primes + 1);
			int *ex = NEW_int(nb_primes + 1);
			for (i = 0; i < nb_primes; i++) {
				primes[i].assign_to(pr[i]);
				ex[i] = exponents[i];
				}
			FREE_OBJECTS(primes);
			FREE_int(exponents);
			primes = pr;
			exponents = ex;
			if (p) {
				pp.create(p);
				}
			else {
				n.assign_to(pp);
				}
			pp.assign_to(primes[nb_primes]);
			exponents[nb_primes] = 1;
			nb_primes++;
			last_prime = p;
			}
		
		D.integral_division(n, pp, q, r, verbose_level - 1);
		if (!r.is_zero()) {
			cout << "longinteger_domain::factor_into_longintegers "
					"factor does not divide" << endl;
			}
		q.assign_to(n);
		if (f_v) {
			cout << "partial factorization: " << a << " = ";
			NT.print_longfactorization(nb_primes, primes, exponents);
			cout << "   * " << n;
			cout << endl;
			}

		}
	if (f_v) {
		cout << "factor(): " << a << " = ";
		NT.print_longfactorization(nb_primes, primes, exponents);
		cout << endl;
		}
}

void longinteger_domain::factor(longinteger_object &a, 
	int &nb_primes, int *&primes, int *&exponents, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object n, q;
	int p, last_prime = 2, i, r;
	number_theory_domain NT;
	
	if (f_v) {
		cout << "factoring " << a << endl;
		}
	if (a.is_zero()) {
		cout << "longinteger_domain::factor a is zero" << endl;
		exit(1);
		}
	if (a.is_one()) {
		nb_primes = 0;
		primes = NEW_int(1);
		exponents = NEW_int(1);
		return;
		}
	a.assign_to(n);
	p = smallest_primedivisor(n, last_prime, verbose_level);
	if (p == 0) {
		p = n.as_int();
		}
	primes = NEW_int(1);
	exponents = NEW_int(1);
	nb_primes = 1;
	primes[0] = p;
	exponents[0] = 1;
	last_prime = p;
	integral_division_by_int(n, p, q, r);
	q.assign_to(n);
	while (!n.is_one()) {
		if (f_v) {
			cout << "remaining factor: " << n << endl;
			}
		p = smallest_primedivisor(n, last_prime, verbose_level);
		// if p == 0: n is prime
		
		if (p == 0) {
			p = n.as_int();
			}
		
		if (p == last_prime) {
			exponents[nb_primes - 1]++;
			}
		else {
			int *pr = NEW_int(nb_primes + 1);
			int *ex = NEW_int(nb_primes + 1);
			for (i = 0; i < nb_primes; i++) {
				pr[i] = primes[i];
				ex[i] = exponents[i];
				}
			FREE_int(primes);
			FREE_int(exponents);
			primes = pr;
			exponents = ex;
			primes[nb_primes] = p;
			exponents[nb_primes] = 1;
			nb_primes++;
			last_prime = p;
			}
		
		if (f_v) {
			cout << "dividing " << n << " by " << p << endl;
			}
		integral_division_by_int(n, p, q, r);
		q.assign_to(n);
		if (f_v) {
			cout << "partial factorization: " << a << " = ";
			NT.print_factorization(nb_primes, primes, exponents);
			cout << "   * " << n;
			cout << endl;
			}

		}
	if (f_v) {
		cout << "factor(): " << a << " = ";
		NT.print_factorization(nb_primes, primes, exponents);
		cout << endl;
		}
}

int longinteger_domain::jacobi(longinteger_object &a, 
	longinteger_object &m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object a1, m1, a2, m2, a3, m3;
	longinteger_object u, v, g, q, r, res, minus_one;
	int n, rr, r1, t1, t2;
	
	if (f_v) {
		cout << "longinteger_domain::jacobi ("
			<< a << " over " << m << ")" << endl;
		}
	a.assign_to(a1);
	m.assign_to(m1);
	r1 = 1;
	
	minus_one.create(-1);
	extended_gcd(a1, m1, g, u, v, verbose_level - 2);
	if (!g.is_one_or_minus_one()) {
		return 0;
		}
	
	while (TRUE) {
		// at this point, we have
		// jacobi(a, m) = r1 * jacobi(a1, m1);
		integral_division(a1, m1, q, r, verbose_level - 2);
		if (f_vv) {
			cout << r1 << " * Jacobi(" << r << ", "
					<< m1 << ")" << endl;
			}
		n = multiplicity_of_p(r, res, 2);
		res.assign_to(a1);
		if (f_vv) {
			cout << r1 << " * Jacobi( 2^" << n << " * "
					<< a1 << ", " << m1 << ")" << endl;
			}
		if (ODD(n)) {
			// t = (m1 * m1 - 1) >> 3; /* t = (m1 * m1 - 1) / 8 */
			/* Ranmultiplizieren von (-1) hoch t an r1: */
			rr = remainder_mod_int(m1, 8);
			if (rr == 3 || rr == 5) {
				r1 = -r1; /* Beachte ABS(r1) == 1L */
				}
			}
		if (f_vv) {
			cout << r1 << " * Jacobi(" << a1 << ", "
					<< m1 << ")" << endl;
			}
		if (a1.is_one_or_minus_one())
			break;
		// reciprocity:
		add(a1, minus_one, a2);
		add(m1, minus_one, m2);
		integral_division_by_int(a2, 2, a3, rr);
		integral_division_by_int(m2, 2, m3, rr);
		integral_division_by_int(a3, 2, a2, t1);
		integral_division_by_int(m3, 2, m2, t2);
		a1.assign_to(a2);
		m1.assign_to(a1);
		a2.assign_to(m1);
		if (ODD(t1) && ODD(t2)) {
			r1 = -r1;
			}
		if (f_vv) {
			cout << r1 << " * Jacobi(" << a1
					<< ", " << m1 << ")" << endl;
			}
		}
	if (f_v) {
		cout << "jacobi(" << a << ", " << m
				<< ") = " << r1 << endl;
		}
	return r1;
}

void longinteger_domain::random_number_less_than_n(
	longinteger_object &n, longinteger_object &r)
{
	int i, l, rr;
	//char *n_rep;
	char *r_rep;
	os_interface Os;
	
	l = n.len();
	n.assign_to(r);
	//n_rep = n.rep();
	r_rep = r.rep();
	while (TRUE) {
		for (i = l - 1; i >= 0; i--) {
			rr = Os.random_integer(10);
			r_rep[i] = (char) rr;
			}
		r.normalize();
		if (compare_unsigned(r, n) < 0)
			break;
		}
}

void longinteger_domain::find_probable_prime_above(
	longinteger_object &a, 
	int nb_solovay_strassen_tests, int f_miller_rabin_test, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object b, one;
	int i = 0;
	
	if (f_v) {
		cout << "longinteger_domain::find_probable_prime_above" << endl;
		}
	one.create(1);
	while (TRUE) {
		if (f_vv) {
			cout << "considering " << a << endl;
			}
		if (!miller_rabin_test(a, verbose_level - 2)) {
			if (f_vv) {
				cout << "is not prime because of Miller Rabin" << endl;
				}
			goto loop;
			}
		if (solovay_strassen_is_prime(a,
				nb_solovay_strassen_tests, verbose_level - 2)) {
			if (f_vv) {
				cout << "may be prime" << endl;
				}
			break;
			}
		else {
			if (f_vv) {
				cout << "is not prime because of "
					"Solovay Strassen" << endl;
				}
			}
loop:
		add(a, one, b);
		b.assign_to(a);
		i++;
		}
	if (f_v) {
		cout << "find_probable_prime_above: probable prime: "
			<< a << " (found after " << i << " tests)" << endl;
		}
}

int longinteger_domain::solovay_strassen_is_prime(
	longinteger_object &n, int nb_tests, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "longinteger_domain::solovay_strassen_is_prime for " 
			<< n << " with " << nb_tests << " tests:" << endl;
			}
	for (i = 0; i < nb_tests; i++) {
		if (!solovay_strassen_is_prime_single_test(
				n, verbose_level - 2)) {
			if (f_v) {
				cout << "is not prime after "
						<< i + 1 << " tests" << endl;
				}
			return FALSE;
			}
		}
	return TRUE;
}

int longinteger_domain::solovay_strassen_is_prime_single_test(
	longinteger_object &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object a, one, b, m_one, n_minus_one;
	int r;

	if (f_v) {
		cout << "longinteger_domain::solovay_strassen_"
				"is_prime_single_test" << endl;
		}
	one.create(1);
	m_one.create(-1);
	add(n, m_one, n_minus_one);
	random_number_less_than_n(n_minus_one, a);
	add(a, one, b);
	b.assign_to(a);
	if (f_vv) {
		cout << "longinteger_domain::solovay_strassen_is_prime "
				"choosing integer " << a
				<< " less than " << n << endl;
		}

	r = solovay_strassen_test(n, a, verbose_level);
	return r;

}

int longinteger_domain::solovay_strassen_test(
	longinteger_object &n, longinteger_object &a,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object b, one, m_one, n2, n_minus_one;
	int x, r;
	
	if (f_v) {
		cout << "longinteger_domain::solovay_strassen_test" << endl;
		}
	one.create(1);
	m_one.create(-1);
	add(n, m_one, n_minus_one);
	if (f_vv) {
		cout << "longinteger_domain::solovay_strassen_test "
			"a = " << a << endl;
		}
	x = jacobi(a, n, verbose_level - 2);
	if (x == 0) {
		if (f_v) {
			cout << "not prime (sure)" << endl;
			}
		return FALSE;
		}
	add(n, m_one, b);
	integral_division_by_int(b, 2, n2, r);
	if (f_vv) {
		cout << "longinteger_domain::solovay_strassen_test "
			"raising to the power " << n2 << endl;
		}
	power_longint_mod(a, n2, n, 0 /*verbose_level - 2*/);
	if (f_vv) {
		cout << "longinteger_domain::solovay_strassen_test "
				"a^((n-1)/2) = " << a << endl;
		}
	if (x == 1) {
		if (a.is_one()) {
			if (f_v) {
				cout << "longinteger_domain::solovay_strassen_test "
					"inconclusive" << endl;
				}
			return TRUE;
			}
		else {
			if (f_v) {
				cout << "longinteger_domain::solovay_strassen_test "
					"not prime (sure)" << endl;
				}
			return FALSE;
			}
		}
	if (x == -1) {
		if (compare_unsigned(a, n_minus_one) == 0) {
			if (f_v) {
				cout << "longinteger_domain::solovay_strassen_test "
					"inconclusive" << endl;
				}
			return TRUE;
			}
		else {
			if (f_v) {
				cout << "longinteger_domain::solovay_strassen_test "
					"not prime (sure)" << endl;
				}
			return FALSE;
			}
		}
	// we should never be here:
	cout << "longinteger_domain::solovay_strassen_test "
			"error" << endl;
	exit(1);
}

int longinteger_domain::miller_rabin_test(
	longinteger_object &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object a, b, c, one, m_one, n_minus_one, m, mm;
	int k, i;
	
	if (f_v) {
		cout << "longinteger_domain::miller_rabin_test "
				"for " << n << endl;
		}
	one.create(1);
	m_one.create(-1);
	add(n, m_one, n_minus_one);
	
#if 0
	// choose a random integer a with 1 <= a <= n - 1
	random_number_less_than_n(n_minus_one, a);
	add(a, one, b);
	b.assign_to(a);
#else
	a.create(2);
#endif
	if (f_vv) {
		cout << "longinteger_domain::miller_rabin_test "
			"choosing integer " << a << " less than " << n << endl;
		}
	
	k = multiplicity_of_p(n_minus_one, m, 2);
	m.assign_to(mm);
	if (f_vv) {
		cout << n_minus_one << " = 2^" << k << " x " << m << endl;
		}
	
	// compute b := a^m mod n
	a.assign_to(b);
	power_longint_mod(b, m, n, FALSE /* f_v */);
	if (f_vv) {
		cout << a << "^" << mm << " = " << b << endl;
		}
	if (b.is_one()) {
		if (f_v) {
			cout << "a^m = 1 mod n, so the test is inconclusive" << endl;
			}
		return TRUE;
		}
	if (compare_unsigned(b, n_minus_one) == 0) {
		if (f_v) {
			cout << "is minus one, so the test is inconclusive" << endl;
			}
		return TRUE;
		}
	for (i = 0; i < k; i++) {
		mult_mod(b, b, c, n, 0); 
		if (f_vv) {
			cout << "b_" << i << "=" << b
					<< " b_" << i + 1 << "=" << c << endl;
			}
		c.assign_to(b);
		if (compare_unsigned(b, n_minus_one) == 0) {
			if (f_v) {
				cout << "is minus one, so the test is inconclusive" << endl;
				}
			return TRUE;
			}
		if (compare_unsigned(b, one) == 0) {
			if (f_v) {
				cout << "is one, we reject as composite" << endl;
				}
			return FALSE;
			}
		//mult(b, b, c);
		}
	if (f_v) {
		cout << "inconclusive, we accept as probably prime" << endl;
		}
	return TRUE;
}

void longinteger_domain::get_k_bit_random_pseudoprime(
	longinteger_object &n, int k, 
	int nb_tests_solovay_strassen, 
	int f_miller_rabin_test, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	int kk = (k * 3) / 10;
	longinteger_object a, b;
	
	if (f_v) {
		cout << "longinteger_domain::get_k_bit_random_pseudoprime "
			"trying to get a " << k << " bit, " << kk
			<< " decimals random pseudoprime" << endl;
		}
	a.create(10);
	D.power_int(a, kk);
	random_number_less_than_n(a, b);
	if (f_v) {
		cout << "choosing integer " << b << " less than " << a << endl;
		}
	add(a, b, n);
	if (f_v) {
		cout << "the sum is " << n << endl;
		}
	
	D.find_probable_prime_above(n,
			nb_tests_solovay_strassen, f_miller_rabin_test,
			verbose_level - 1);
	
}

void longinteger_domain::RSA_setup(
	longinteger_object &n,
	longinteger_object &p, longinteger_object &q, 
	longinteger_object &a, longinteger_object &b, 
	int nb_bits, 
	int nb_tests_solovay_strassen, int f_miller_rabin_test, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_domain D;
	longinteger_object m1, pm1, qm1, phi_n, v, g;
	int half_bits = nb_bits >> 1;
	
	if (f_v) {
		cout << "longinteger_domain::RSA_setup nb_bits=" << nb_bits 
			<< " nb_tests_solovay_strassen=" << nb_tests_solovay_strassen 
			<< " f_miller_rabin_test=" << f_miller_rabin_test << endl;
		}
	m1.create(-1);
	D.get_k_bit_random_pseudoprime(p, half_bits, 
		nb_tests_solovay_strassen,
		f_miller_rabin_test, verbose_level - 2);
	if (f_vv) {
		cout << "choosing p = " << p << endl;
		}
	D.get_k_bit_random_pseudoprime(q, half_bits, 
		nb_tests_solovay_strassen,
		f_miller_rabin_test, verbose_level - 2);
	if (f_v) {
		cout << "choosing p = " << p << endl;
		cout << "choosing q = " << q << endl;
		}
	D.mult(p, q, n);
	if (f_v) {
		cout << "n = pq = " << n << endl;
		}
	D.add(p, m1, pm1);
	D.add(q, m1, qm1);
	D.mult(pm1, qm1, phi_n);
	if (f_v) {
		cout << "phi(n) = (p - 1)(q - 1) = "
				<< phi_n << endl;
		}
	
	while (TRUE) {
		random_number_less_than_n(n, a);
		if (f_v) {
			cout << "choosing integer " << a
					<< " less than " << n << endl;
			}
		D.extended_gcd(a, phi_n, g, b, v, verbose_level - 2);
		if (g.is_one())
			break;
		if (f_v) {
			cout << "non trivial gcd: " << g
					<< " , repeating" << endl;
			}
		}		
	if (b.sign()) {
		if (f_v) {
			cout << "making b positive" << endl;
			}
		D.add(b, phi_n, v);
		v.assign_to(b);
		}
	if (f_v) {
		cout << "the public key is (a,n) = " << a << "," << n << endl;
		cout << "the private key is (b,n) = " << b << "," << n << endl;
		}
	
}


void longinteger_domain::matrix_product(
		longinteger_object *A, longinteger_object *B,
		longinteger_object *&C, int Am, int An, int Bn)
{
	int i, j, k;
	longinteger_object a, b, c;
	
	for (i = 0; i < Am; i++) {
		for (j = 0; j < Bn; j++) {
			c.create(0);
			for (k = 0; k < An; k++) {
				mult(A[i * An + k], B[k * Bn + j], a);
				add(a, c, b);
				b.assign_to(c);
				}
			c.assign_to(C[i * Bn + j]);
			}
		}
}

void longinteger_domain::matrix_entries_integral_division_exact(
	longinteger_object *A, longinteger_object &b,
	int Am, int An)
{
	int i, j;
	longinteger_object q, r;
	int verbose_level = 0;
	
	for (i = 0; i < Am; i++) {
		for (j = 0; j < An; j++) {
			integral_division(A[i * An + j], b, q, r, verbose_level);
			if (!r.is_zero()) {
				cout << "integral division: " << b
					<< " does not divide " << A[i * An + j] << endl;
				cout << "i=" << i << " j=" << j << endl;
				exit(1);
				}
			q.assign_to(A[i * An + j]);
			}
		}
}

void longinteger_domain::matrix_print_GAP(
		ostream &ost, longinteger_object *A,
		int Am, int An)
{
	int i, j;
	
	ost << "[";
	for (i = 0; i < Am; i++) {
		ost << "[";
		for (j = 0; j < An; j++) {
			ost << A[i * An + j];
			if (j < An - 1)
				ost << ",";
			ost << " ";
			}
		ost << "]";
		if (i < An - 1)
			ost << ",";
		ost << endl;
		}
	ost << "];" << endl;
}

void longinteger_domain::matrix_print_tex(
		ostream &ost, longinteger_object *A,
		int Am, int An)
{
	int i, j;
	
	ost << "\\begin{array}{*{" << An << "}{r}}" << endl;
	for (i = 0; i < Am; i++) {
		for (j = 0; j < An; j++) {
			ost << A[i * An + j];
			if (j < An - 1)
				ost << " & ";
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
}

void longinteger_domain::power_mod(
	char *aa, char *bb, char *nn,
	longinteger_object &result, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object a, b, c, d, n;
	
	a.create_from_base_10_string(aa, 0);
	c.create_from_base_10_string(aa, 0);
	b.create_from_base_10_string(bb, 0);
	d.create_from_base_10_string(bb, 0);
	n.create_from_base_10_string(nn, 0);
	power_longint_mod(c, d, n, verbose_level - 1);
	c.assign_to(result);
	if (f_v) {
		cout << "longinteger_domain::power_mod:" << aa
				<< " ^ " << bb << " == " << result
				<< " mod " << nn << endl;
		}
}


void longinteger_domain::factorial(
		longinteger_object &result, int n)
{
	int *x;
	int i;
	
	x = NEW_int(n);
	for (i = 0; i < n; i++) {
		x[i] = i + 1;
		}
	multiply_up(result, x, n, 0 /* verbose_level */);
	FREE_int(x);
}

void longinteger_domain::group_order_PGL(
		longinteger_object &result,
		int n, int q, int f_semilinear)
{
	int *x;
	int i, l;
	int p, e;
	number_theory_domain NT;
	
	NT.factor_prime_power(q, p, e);
	l = n;
	if (f_semilinear) {
		l++;
		}
	
	x = NEW_int(l);
	for (i = 0; i < n; i++) {
		x[i] = NT.i_power_j(q, n) - NT.i_power_j(q, i);
		if (i == 0) {
			x[i] = x[i] / (q - 1);
			}
		}
	if (f_semilinear) {
		x[n] = e;
		}

#if 0
	cout << "longinteger_domain::group_order_PGL "
			"factors of |PGL(n,q)| = ";
	int_vec_print(cout, x, l);
	cout << endl;
#endif

	multiply_up(result, x, l, 0 /* verbose_level */);
	FREE_int(x);
}

int longinteger_domain::singleton_bound_for_d(
		int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d;
	
	if (f_v) {
		cout << "longinteger_domain::singleton_bound_for_d" << endl;
		}
	d = n - k + 1;
	return d;
}


int longinteger_domain::hamming_bound_for_d(
		int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int e, d, t;
	longinteger_object qnmk, qm1, qm1_power, B, s, a, b;
	
	if (f_v) {
		cout << "longinteger_domain::hamming_bound_for_d" << endl;
		}
	qnmk.create(q);
	qm1.create(q - 1);
	power_int(qnmk, n - k);
	qm1_power.create(1);
	B.create(0);
	if (f_vv) {
		cout << "longinteger_domain::hamming_bound_for_d: "
			"q=" << q << " n=" << n << " k=" << k << " "
			<< q << "^" << n - k << " = " << qnmk << endl;
		}
	for (e = 0; ; e++) {
		binomial(b, n, e, FALSE);
		mult(b, qm1_power, s);
		add(B, s, a);
		a.assign_to(B);
		if (compare(B, qnmk) == 1) {
			// now the size of the Ball of radius e is bigger than q^{n-m}
			t = e - 1;
			d = 2 * t + 2;
			if (f_vv) {
				cout << "B=" << B << " t=" << t << " d=" << d << endl;
				}
			break;
			}
		if (f_vv) {
			cout << "e=" << e << " B=" << B << " is OK" << endl;
			}
		mult(qm1_power, qm1, s);
		s.assign_to(qm1_power);
		}
	if (f_v) {
		cout << "longinteger_domain::hamming_bound_for_d done" << endl;
		}
	return d;
}

int longinteger_domain::plotkin_bound_for_d(
		int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int d;
	longinteger_object qkm1, qk, qm1, a, b, c, Q, R;
	
	if (f_v) {
		cout << "longinteger_domain::plotkin_bound_for_d" << endl;
		}

	// d \le \frac{n q^{k-1}}{q^k-1}

	qkm1.create(q);
	power_int(qkm1, k - 1);
	a.create(n);
	mult(a, qkm1, b);
		// now b = n q^{k-1}

	a.create(q - 1);
	mult(b, a, c);
		// now c = n q^{k-1} (q - 1)
		

	a.create(q);
	mult(a, qkm1, qk);
		// now qk = q^k

	a.create(-1);
	add(qk, a, b);
		// now b = 2^k - 1

	if (f_vv) {
		cout << "longinteger_domain::plotkin_bound_for_d "
				"q=" << q << " n=" << n << " k=" << k << endl;
		}
	integral_division(c, b, Q, R, FALSE /* verbose_level */);
	d = Q.as_int();
	if (f_vv) {
		cout << c << " / " << b << " = " << d << endl;
		}
	if (f_v) {
		cout << "longinteger_domain::plotkin_bound_for_d" << endl;
		}
	return d;
}

int longinteger_domain::griesmer_bound_for_d(
		int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d, n1;
	
	if (f_v) {
		cout << "longinteger_domain::griesmer_bound_for_d" << endl;
		}
	for (d = 1; d <= n; d++) {
		n1 = griesmer_bound_for_n(k, d, q, verbose_level - 2);
		if (n1 > n) {
			d--;
			break;
			}
		}
	if (f_v) {
		cout << "longinteger_domain::griesmer_bound_for_d done" << endl;
		}
	return d;
}

int longinteger_domain::griesmer_bound_for_n(
		int k, int d, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, n;
	longinteger_object qq, qi, d1, S, Q, R, one, a, b;
	
	if (f_v) {
		cout << "longinteger_domain::griesmer_bound_for_n" << endl;
		}
	one.create(1);
	d1.create(d);
	qq.create(q);
	qi.create(1);
	S.create(0);
	if (f_vv) {
		cout << "griesmer_bound_for_n: q=" << q
				<< " d=" << d << " k=" << k << endl;
		}
	for (i = 0; i < k; i++) {
		integral_division(d1, qi, Q, R, FALSE /* verbose_level */);
		if (!R.is_zero()) {
			add(Q, one, a);
			add(S, a, b);
			}
		else {
			add(S, Q, b);
			}
		b.assign_to(S);
		mult(qi, qq, a);
		a.assign_to(qi);
		if (f_vv) {
			cout << "i=" << i << " S=" << S << endl;
			}
		}
	n = S.as_int();
	if (f_v) {
		cout << "longinteger_domain::griesmer_bound_for_n" << endl;
		}
	return n;
}






//##############################################################################
// global functions:
//##############################################################################

#if 0
void test_longinteger()
{
	longinteger_domain D;
	int x[] = {15, 14, 12, 8};
	longinteger_object a, b, q, r;
	int verbose_level = 0;
	
	D.multiply_up(a, x, 4);
	cout << "a=" << a << endl;
	b.create(2);
	while (!a.is_zero()) {
		D.integral_division(a, b, q, r, verbose_level);
		//cout << a << " = " << q << " * " << b << " + " << r << endl;
		cout << r << endl;
		q.assign_to(a);
		}
	
	D.multiply_up(a, x, 4);
	cout << "a=" << a << endl;
	
	int *rep, len;
	D.base_b_representation(a, 2, rep, len);
	b.create_from_base_b_representation(2, rep, len);
	cout << "b=" << b << endl;
	FREE_int(rep);
}

void test_longinteger2()
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

void test_longinteger3()
{
	int i, j;
	longinteger_domain D;
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

void test_longinteger4()
{
	int n = 6, q = 2, k, x, d = 3;
	longinteger_domain D;
	longinteger_object a;
	
	for (k = 0; k <= n; k++) {
		for (x = 0; x <= n; x++) {
			if (x > 0 && x < d)
				continue;
			if (q == 2 && EVEN(d) && ODD(x))
				continue;
			D.krawtchouk(a, n, q, k, x);
			a.print(cout);
			cout << " ";
			}
		cout << endl;
		}
}

void test_longinteger5()
{
	longinteger_domain D;
	longinteger_object a, b, u, v, g;
	int verbose_level = 2;
	
	a.create(9548);
	b.create(254774);
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

void test_longinteger6()
{
	int verbose_level = 2;
	longinteger_domain D;
	longinteger_object a, b;
	
	a.create(7411);
	b.create(9283);
	D.jacobi(a, b, verbose_level);


}

void test_longinteger7()
{
	longinteger_domain D;
	longinteger_object a, b;
	int i, j;
	int mult[15];
		
	a.create(15);
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

void test_longinteger8()
{
	int verbose_level = 2;
	longinteger_domain D;
	longinteger_object a, b, one;
	int nb_solovay_strassen_tests = 100;
	int f_miller_rabin_test = TRUE;
	
	one.create(1);
	a.create(197659);
	D.find_probable_prime_above(a, nb_solovay_strassen_tests, 
		f_miller_rabin_test, verbose_level);
}

void mac_williams_equations(longinteger_object *&M, int n, int k, int q)
{
	longinteger_domain D;
	int i, j;
	
	M = NEW_OBJECTS(longinteger_object, (n + 1) * (n + 1));
	
	for (i = 0; i <= n; i++) {
		for (j = 0; j <= n; j++) {
			D.krawtchouk(M[i * (n + 1) + j], n, q, i, j);
			}
		}
}

void determine_weight_enumerator()
{
	int n = 19, k = 7, q = 2;
	longinteger_domain D;
	longinteger_object *M, *A1, *A2, qk;
	int i;
	
	qk.create(q);
	D.power_int(qk, k);
	cout << q << "^" << k << " = " << qk << endl;
	
	mac_williams_equations(M, n, k, q);
	
	D.matrix_print_tex(cout, M, n + 1, n + 1);
	
	A1 = NEW_OBJECTS(longinteger_object, n + 1);
	A2 = NEW_OBJECTS(longinteger_object, n + 1);
	for (i = 0; i <= n; i++) {
		A1[i].create(0);
		}
	A1[0].create(1);
	A1[8].create(78);
	A1[12].create(48);
	A1[16].create(1);
	D.matrix_print_tex(cout, A1, n + 1, 1);
	
	D.matrix_product(M, A1, A2, n + 1, n + 1, 1);
	D.matrix_print_tex(cout, A2, n + 1, 1);

	D.matrix_entries_integral_division_exact(A2, qk, n + 1, 1);

	D.matrix_print_tex(cout, A2, n + 1, 1);
	
	FREE_OBJECTS(M);
	FREE_OBJECTS(A1);
	FREE_OBJECTS(A2);
}

void longinteger_collect_setup(int &nb_agos,
		longinteger_object *&agos, int *&multiplicities)
{
	nb_agos = 0;
	agos = NULL;
	multiplicities = NULL;
}

void longinteger_collect_free(int &nb_agos,
		longinteger_object *&agos, int *&multiplicities)
{
	if (nb_agos) {
		FREE_OBJECTS(agos);
		FREE_int(multiplicities);
		}
}

void longinteger_collect_add(int &nb_agos,
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

void longinteger_collect_print(ostream &ost,
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
#endif

void longinteger_free_global_data()
{
	cout << "longinteger_free_global_data" << endl;
	if (tab_binomials) {
		cout << "longinteger_free_global_data before "
				"FREE_OBJECTS(tab_binomials)" << endl;
		FREE_OBJECTS(tab_binomials);
		cout << "longinteger_free_global_data after "
				"FREE_OBJECTS(tab_binomials)" << endl;
		tab_binomials = NULL;
		tab_binomials_size = 0;
		}
	if (tab_q_binomials) {
		cout << "longinteger_free_global_data before "
				"FREE_OBJECTS(tab_q_binomials)" << endl;
		FREE_OBJECTS(tab_q_binomials);
		cout << "longinteger_free_global_data after "
				"FREE_OBJECTS(tab_q_binomials)" << endl;
		tab_q_binomials = NULL;
		tab_q_binomials_size = 0;
		}
	cout << "longinteger_free_global_data done" << endl;
}

void longinteger_print_digits(char *rep, int len)
{
	for (int h = 0; h < len; h++) cout << (char)('0' + rep[h]) << " ";
}

void longinteger_domain_free_tab_q_binomials()
{
	if (tab_q_binomials) {
		FREE_OBJECTS(tab_q_binomials);
		tab_q_binomials = NULL;
	}
}



}}

