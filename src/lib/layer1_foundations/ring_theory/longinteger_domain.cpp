// longinteger_domain.cpp
//
// Anton Betten
//
// started as longinteger.cpp:  October 26, 2002
// moved here: January 23, 2015



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace ring_theory {



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

int longinteger_domain::is_less_than(
		longinteger_object &a, longinteger_object &b)
{
	if (compare_unsigned(a, b) == -1)
		return true;
	else
		return false;
}

void longinteger_domain::subtract_signless(
		longinteger_object &a, longinteger_object &b,
		longinteger_object &c)
// c = a - b, assuming a > b
{
	int i;
	char ai, bi, carry;
	
	c.freeself();
	c.sign() = false;
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
		if (i < b.len()) {
			bi = b.rep()[i];
		}
		else {
			bi = 0;
		}
		bi += carry;
		ai = a.rep()[i];
		if (bi > ai) {
			ai += 10;
			carry = 1;
		}
		else {
			carry = 0;
		}
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
		if (i < a.len()) {
			ai = a.rep()[i];
		}
		else {
			ai = 0;
		}
		if (i < b.len()) {
			bi = b.rep()[i];
		}
		else {
			bi = 0;
		}
		ci = ai + bi + carry;
		if (ci >= 10) {
			carry = 1;
		}
		else {
			carry = 0;
		}
		c.rep()[i] = ci % 10;
	}
	c.normalize();
}

void longinteger_domain::add_mod(
		longinteger_object &a,
	longinteger_object &b, longinteger_object &c,
	longinteger_object &m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object d, q;

	if (f_v ) {
		cout << "longinteger_domain::add_mod "
				"a=" << a << " b=" << b << " m=" << m << endl;
	}
	add(a, b, d);
	integral_division(
		d, m,
		q, c,
		0 /*verbose_level*/);
	if (f_v ) {
		cout << "longinteger_domain::add_mod "
				"a=" << a << " +  b=" << b << " mod m=" << m << " is " << c << endl;
	}

}

void longinteger_domain::add_in_place(
		longinteger_object &a,
		longinteger_object &b)
// a := a + b
{
	longinteger_object C;
	
	add(a, b, C);
	C.assign_to(a);
}

void longinteger_domain::subtract_in_place(
		longinteger_object &a, longinteger_object &b)
// a := a - b
{
	longinteger_object c;

	if (is_less_than(a, b)) {
		subtract_signless(b, a, c);
		mult_integer_in_place(c, -1);
	}
	else {
		subtract_signless(a, b, c);
	}
	c.assign_to(a);
}

void longinteger_domain::add_int_in_place(
		longinteger_object &a, long int b)
// a := a + b
{
	longinteger_object C;
	longinteger_object B;

	B.create(b, __FILE__, __LINE__);
	add(a, B, C);
	C.assign_to(a);
}


void longinteger_domain::mult(
		longinteger_object &a, longinteger_object &b,
		longinteger_object &c)
{
	int i, j;
	char ai, bj, d, carry;
	int f_v = false;
	
	if (a.is_zero() || b.is_zero()) {
		c.create(0, __FILE__, __LINE__);
		return;
	}
	if ((a.sign() && !b.sign()) || (!a.sign() && b.sign())) {
		c.sign() = true;
	}
	else {
		c.sign() = false;
	}
	
	c.freeself();
	c.len() = a.len() + b.len() + 2;
	c.rep() = NEW_char(c.len());
	for (i = 0; i < c.len(); i++) {
		c.rep()[i] = 0;
	}
	
	if (f_v) {
		cout << "longinteger_domain::mult a=";
		print_digits(a.rep(), a.len());
		cout << "b=";
		print_digits(b.rep(), b.len());
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
		print_digits(c.rep(), c.len());
		cout << endl;
	}
	c.normalize();
	if (f_v) {
		cout << "longinteger_domain::mult after normalize, c=";
		print_digits(c.rep(), c.len());
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
	
	B.create(b, __FILE__, __LINE__);
	mult(a, B, C);
	C.assign_to(a);
}

static int do_division(
		longinteger_domain &D,
		longinteger_object &r, longinteger_object table[10])
{
	int i, cmp;
	
	for (i = 9; i >= 0; i--) {
		cmp = D.compare(r, table[i]);
		if (cmp >= 0) {
			return i;
		}
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
		c.create(0, __FILE__, __LINE__);
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
	if ((a.sign() && !b.sign()) || (!a.sign() && b.sign())) {
		c.sign() = true;
	}
	else {
		c.sign() = false;
	}

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
	for (i = 0; i < c.len(); i++) {
		c.rep()[i] = 0;
	}
	c.assign_to(c0);
	
	for (j = b.len() - 1; j >= 0; j--) {
		if (f_vv) {
			cout << "j=" << j << endl;
		}
		bj = b.rep()[j];
		carry = 0;
		for (i = 0; i < l; i++) {
			if (i < a.len()) {
				ai = a.rep()[i];
			}
			else {
				ai = 0;
			}
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
			for (; i < c.len(); i++) {
				c.rep()[i] = 0;
			}
		}
		else {
			for (i = 0; i < r.len(); i++) {
				if (i < c.len()) {
					c.rep()[i] = r.rep()[i];
				}
			}
			for (; i < c.len(); i++) {
				c.rep()[i] = 0;
			}
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
		longinteger_object &a, int *x, int len,
		int verbose_level)
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
		b.create(x[i], __FILE__, __LINE__);
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

void longinteger_domain::multiply_up_lint(
		longinteger_object &a, long int *x, int len,
		int verbose_level)
{
	longinteger_object b, c;
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "longinteger_domain::multiply_up_lint" << endl;
	}
	a.one();
	if (f_v) {
		cout << "longinteger_domain::multiply_up_lint "
				"a=" << a << endl;
	}
	for (i = 0; i < len; i++) {
		if (x[i] == 1) {
			continue;
		}
		b.create(x[i], __FILE__, __LINE__);
		if (f_v) {
			cout << "longinteger_domain::multiply_up_lint "
					"i=" << i << " x[i]=" << x[i]
					<< " b=" << b << endl;
		}
		mult(a, b, c);
		if (f_v) {
			cout << "longinteger_domain::multiply_up_lint "
					"i=" << i << " x[i]=" << x[i]
					<< " c=" << c << endl;
		}
		c.assign_to(a);
		if (f_v) {
			cout << "longinteger_domain::multiply_up_lint "
					"i=" << i << " x[i]=" << x[i]
					<< " a=" << a << endl;
		}
		//cout << "*" << x[i] << "=" << a << endl;
	}
	if (f_v) {
		cout << "longinteger_domain::multiply_up_lint done" << endl;
	}
}

int longinteger_domain::quotient_as_int(
		longinteger_object &a, longinteger_object &b)
{
	longinteger_object q, r;

	integral_division(a, b, q, r, 0);
	return q.as_int();
}

long int longinteger_domain::quotient_as_lint(
		longinteger_object &a, longinteger_object &b)
{
	longinteger_object q, r;

	integral_division(a, b, q, r, 0);
	return q.as_lint();
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
		c.create(i, __FILE__, __LINE__);
		mult(b, c, table[i]);
	}
	q.freeself();
	r.freeself();
	
	// load r with leading b.len() digits of a: 
	r.sign() = false;
	r.len() = b.len() + 1;
	r.rep() = NEW_char(r.len());
	l = a.len() - b.len();
	for (i = 0; i < b.len(); i++) {
		r.rep()[i] = a.rep()[l + i];
	}
	r.rep()[b.len()] = 0;
	
	// allocate q of length a.len() - b.len() + 1
	q.sign() = false;
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
		
		if (l == 0) {
			break;
		}
		
		// put c into r, shift up by one digit 
		// and append the next digit a[l-1].
		for (i = 0; i < c.len(); i++) {
			r.rep()[i + 1] = c.rep()[i];
		}
		for (i++ ; i < r.len(); i++) {
			r.rep()[i] = 0;
		}
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

	B.create(b, __FILE__, __LINE__);
	integral_division(a, B, q, R, verbose_level);
	r = R.as_int();
}

void longinteger_domain::integral_division_by_lint(
	longinteger_object &a,
	long int b, longinteger_object &q, long int &r)
{
	longinteger_object B, R;
	int verbose_level = 0;

	B.create(b, __FILE__, __LINE__);
	integral_division(a, B, q, R, verbose_level);
	r = R.as_lint();
}

void longinteger_domain::inverse_mod(
	longinteger_object &a,
	longinteger_object &m, longinteger_object &av, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object g, u, v;
	//int c;

	if (f_v) {
		cout << "longinteger_domain::inverse_mod "
				"a=" << a << " m=" << m << endl;
	}
	extended_gcd(
		a,
		m, g,
		u, v, 0 /* verbose_level */);
	if (f_v) {
		cout << "longinteger_domain::inverse_mod ";
		g.print(cout);
		cout << " = ";
		u.print(cout);
		cout << " * ";
		a.print(cout);
		cout << " + ";
		v.print(cout);
		cout << " * ";
		m.print(cout);
		cout << endl;
	}
	if (u.sign()) {
		subtract_signless(m, u, av);
	}
	else {
		u.assign_to(av);
	}
	if (f_v) {
		cout << "longinteger_domain::inverse_mod "
				"a=" << a << " m=" << m << " av=" << av << endl;
	}
}

void longinteger_domain::extended_gcd(
	longinteger_object &a,
	longinteger_object &b, longinteger_object &g, 
	longinteger_object &u, longinteger_object &v,
	int verbose_level)
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
		u.create(0, __FILE__, __LINE__);
		v.create(1, __FILE__, __LINE__);
		if (g.sign()) {
			g.negate();
			u.negate();
			v.negate();
		}
		return;
	}
	if (b.is_zero()) {
		a.assign_to(g);
		u.create(1, __FILE__, __LINE__);
		v.create(0, __FILE__, __LINE__);
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
	sm1.create(1, __FILE__, __LINE__);
	tm1.create(0, __FILE__, __LINE__);
	s.create(0, __FILE__, __LINE__);
	t.create(1, __FILE__, __LINE__);
	while (true) {
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
	longinteger_object b, c, d, n1, N;
	int r;
	
	a.assign_to(b);
	n.assign_to(N);
	c.one();
	if (f_v) {
		cout << "longinteger_domain::power_longint_mod" << endl;
		cout << "computing " << b << " to the power "
				<< n << " mod " << m << ":" << endl;
	}
	while (!N.is_zero()) {
		if (f_vv) {
			cout << "n=" << N << " : " << b << "^"
					<< N << " * " << c << endl;
		}
		integral_division_by_int(N, 2, n1, r);
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
		n1.assign_to(N);
	}
	c.assign_to(a);
}






void longinteger_domain::square_root(
		longinteger_object &a,
		longinteger_object &sqrt_a,
		int verbose_level)
{
	int i, la, len;
	char c1;
	int f_v = (verbose_level >= 1);
	longinteger_object a2;

	if (a.is_zero()) {
		sqrt_a.create(0, __FILE__, __LINE__);
		return;
	}
	if (a.sign()) {
		cout << "longinteger_domain::square_root a is negative" << endl;
		exit(1);
	}

	la = a.len();
	if (ODD(la)) {
		la++;
	}
	len = (la >> 1) + 1;

	sqrt_a.freeself();
	sqrt_a.len() = len;
	sqrt_a.rep() = NEW_char(sqrt_a.len());
	for (i = 0; i < sqrt_a.len(); i++) {
		sqrt_a.rep()[i] = 0;
	}

	if (f_v) {
		cout << "longinteger_domain::square_root a=";
		print_digits(a.rep(), a.len());
	}

	for (i = len - 1; i >= 0; i--) {
		for (c1 = 9; c1 >= 0; c1--) {
			if (f_v) {
				cout << "trying a[" << i << "]=" << (int) c1 << endl;
			}
			sqrt_a.rep()[i] = c1;
			if (f_v) {
				cout << "sqrt_a = " << sqrt_a << endl;
			}
			if (c1) {
				mult(sqrt_a, sqrt_a, a2);
				if (f_v) {
					cout << "a2 = " << a2 << endl;
				}
				if (compare_unsigned(a2, a) <= 0) {
					if (f_v) {
						cout << "success with digit " << i
								<< " : sqrt_a=" << sqrt_a << endl;
					}
					break;
				}
			}
		}
		if (f_v) {
			cout << "sqrt_a[" << i << "]=" << sqrt_a.rep()[i] << endl;
		}
	}

	if (f_v) {
		cout << "longinteger_domain::square_root c=";
		print_digits(sqrt_a.rep(), sqrt_a.len());
		cout << endl;
	}
	sqrt_a.normalize();
	if (f_v) {
		cout << "longinteger_domain::square_root after normalize, sqrt_a=";
		print_digits(sqrt_a.rep(), sqrt_a.len());
		cout << endl;
	}
}



int longinteger_domain::square_root_mod(
		int a, int p, int verbose_level)
// solves x^2 = a mod p. Returns x
{
	int f_v = (verbose_level >= 1);
	longinteger_object P;
	longinteger_object A, X, a2, a4, b, X2, Four, Two, mOne;
	int round;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "longinteger_domain::square_root_mod" << endl;
	}
	a = a % p;
	if (NT.Jacobi(a, p, 0 /* verbose_level*/) == -1) {
		cout << "longinteger_domain::square_root_mod a is not a square modulo p" << endl;
		cout << "a=" << a << endl;
		cout << "p=" << p << endl;
		exit(1);
	}
	Two.create(2, __FILE__, __LINE__);
	Four.create(4, __FILE__, __LINE__);
	A.create(a, __FILE__, __LINE__);
	P.create(p, __FILE__, __LINE__);
	mOne.create(p - 1, __FILE__, __LINE__);
	if (f_v) {
		cout << "longinteger_domain::square_root_mod A=" << A << endl;
		cout << "longinteger_domain::square_root_mod P=" << P << endl;
		cout << "longinteger_domain::square_root_mod mOne=" << mOne << endl;
	}
	if (p % 4 == 3) {
		A.assign_to(X);
		power_int_mod(X, (p + 1) >> 2, P);
		return X.as_int();
	}
	if (p % 8 == 5) {
		if (f_v) {
			cout << "longinteger_domain::square_root_mod p % 8 == 5" << endl;
		}
		A.assign_to(b);
		power_int_mod(b, (p - 1) >> 2, P);
		if (f_v) {
			cout << "longinteger_domain::square_root_mod a^((p-1)/4)=" << b << endl;
		}
		// cout << "A = " << A << endl;
		// cout << "b = A^(p-1)/4=" << b << endl;
		if (b.is_one()) {
			if (f_v) {
				cout << "longinteger_domain::square_root_mod a^((p-1)/4)=1" << endl;
			}
			A.assign_to(X);
			power_int_mod(X, (p + 3) >> 3, P);
			if (f_v) {
				cout << "longinteger_domain::square_root_mod done" << endl;
			}
			return X.as_int();
			}
		if (compare_unsigned(b, mOne) == 0) {
			if (f_v) {
				cout << "longinteger_domain::square_root_mod a^((p-1)/4)=1" << endl;
			}

			//a2.add_mod(A, A, P);
			//a4.add_mod(a2, a2, P);
			//a4.power_int_mod((p - 5) >> 3, P);
			//X.mult_mod(a2, a4, P);

			add_mod(A, A, a2, P, 0 /* verbose_level */);
			add_mod(a2, a2, a4, P, 0 /* verbose_level */);
			power_int_mod(a4, (p - 5) >> 3, P);
			mult_mod(a2, a4, X, P, 0 /* verbose_level */);
			if (f_v) {
				cout << "longinteger_domain::square_root_mod done" << endl;
			}
			return X.as_int();
			}
		else {
			cout << "longinteger_domain::square_root_mod p % 8 = 5 and power neq +-1" << endl;
			cout << "power = " << b << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "longinteger_domain::square_root_mod p % 8 == 1" << endl;
	}
	// now p % 8 == 1
	// Tonelli / Shanks algorithm:
	int n, r = 0, q, e, m;
	longinteger_object Z, N, Y, B, T, d, AB, Ypower, Bpower, Tmp1;

	if (f_v) {
		cout << "longinteger_domain::square_root_mod, Tonelli / Shanks:" << endl;
	}
	q = p - 1;
	if (f_v) {
		cout << "longinteger_domain::square_root_mod q=" << q << endl;
	}
	e = 0;
	while (EVEN(q)) {
		q >>= 1;
		e++;
	}
	if (f_v) {
		cout << "p - 1 = 2^" << e << " * " << q << endl;
	}

	// pick n to be a nonsquare mod p:
	for (n = 1; n < p - 1; n++) {
		r = NT.Legendre(n, p, verbose_level - 1);
		if (r == -1) {
			break;
		}
	}

	if (f_v) {
		cout << "n=" << n << " p=" << p << " Legendre(n,p)=" << r<< endl;
	}
	N.create(n, __FILE__, __LINE__);
	if (f_v) {
		cout << "longinteger_domain::square_root_mod N=" << N << endl;
	}
	N.assign_to(Z);
	power_int_mod(Z, q, P);
	Z.assign_to(Y);
	if (f_v) {
		cout << "longinteger_domain::square_root_mod Y=N^q=" << Y << endl;
	}
	r = e;
	A.assign_to(X);
	power_int_mod(X, (q - 1) >> 1, P);
	if (f_v) {
		cout << "longinteger_domain::square_root_mod X=" << X << endl;
	}
	mult_mod(X, X, d, P, 0 /* verbose_level */);
	mult_mod(A, d, B, P, 0 /* verbose_level */);
	mult_mod(A, X, Tmp1, P, 0 /* verbose_level */);
	Tmp1.assign_to(X);
	if (f_v) {
		cout << "initialization:" << endl;
	}
	round = 0;
	while (true) {
		if (f_v) {
			cout << "Y=" << Y << endl;
			cout << "r=" << r << endl;
			cout << "X=" << X << endl;
			cout << "B=" << B << endl;
		}


		mult_mod(X, X, X2, P, 0 /* verbose_level */);
		if (f_v) {
			cout << "longinteger_domain::square_root_mod X2=" << X2 << endl;
		}
		mult_mod(A, B, AB, P, 0 /* verbose_level */);
		if (f_v) {
			cout << "longinteger_domain::square_root_mod AB=" << AB << endl;
			cout << "B=" << B << endl;
		}

		if (compare_unsigned(AB, X2) != 0) {
			cout << "loop invariant violated: ab != x^2" << endl;
			cout << "ab=" << d << endl;
			cout << "x^2=" << X2 << endl;
			exit(1);
		}

		Y.assign_to(Ypower);
		power_int_mod(Ypower, 1 << (r - 1), P);
		if (f_v) {
			cout << "longinteger_domain::square_root_mod Ypower=" << Ypower << endl;
		}

		if (compare_unsigned(Ypower, mOne)) {
			cout << "loop invariant violated: Y^{2^{r-1}} != -1" << endl;
			exit(1);
		}



		B.assign_to(Bpower);
		if (f_v) {
			cout << "longinteger_domain::square_root_mod Bpower (before)=" << Bpower << endl;
		}
		power_int_mod(Bpower, 1 << (r - 1), P);
		if (f_v) {
			cout << "longinteger_domain::square_root_mod Bpower=" << Bpower << endl;
		}
		if (!Bpower.is_one()) {
			cout << "loop invariant violated: B^{2^{r-1}} != 1" << endl;
			exit(1);
		}





		if (remainder_mod_int(B, p) == 1) {
			m = -1;
		}
		else {
			for (m = 1; ; m++) {
				B.assign_to(d);
				power_int_mod(d, 1 << m, P);
				if (d.is_one()) {
					break;
				}
				if (m >= r) {
					cout << "sqrt_mod(), Tonelli / Shanks:" << endl;
					cout << "error: a is not a quadratic residue mod p" << endl;
					exit(1);
				}
			}
		}


		if (f_v) {
			cout << round << " & " << A << " & " << B << " & " << X << " & "
				<< X2 << " & " << Y << " & " << r << " & " << AB
				<< " & " << Ypower << " & " << Bpower << " & ";
		}

		if (m == -1) {
			if (f_v) {
				cout << " & & & & \\\\" << endl;
			}
		}
		else {
			if (f_v) {
				cout << m;
			}
		}

		//cout << "m=" << m << endl;

		if (m == -1) {
			if (f_v) {
				cout << "longinteger_domain::square_root_mod done" << endl;
			}
			return X.as_int();
		}

		if (f_v) {
			cout << "m=" << m << endl;
		}
		Y.assign_to(T);
		power_int_mod(T, 1 << (r - m - 1), P);
		mult_mod(T, T, Y, P, 0 /* verbose_level */);
		r = m; // integers

		mult_mod(X, T, Tmp1, P, 0 /* verbose_level */);
		Tmp1.assign_to(X);

		mult_mod(B, Y, Tmp1, P, 0 /* verbose_level */);
		Tmp1.assign_to(B);

		if (f_v) {
			cout << " & " << Y << " & " << X << " & " << B << " & " << r;
			cout << "\\\\" << endl;
		}
		round++;
		}
	if (f_v) {
		cout << "longinteger_domain::square_root_mod done" << endl;
	}
}

void longinteger_domain::create_q_to_the_n(
		longinteger_object &a, int q, int n)
// create (q^n - 1)
{
	a.create(q, __FILE__, __LINE__);
	power_int(a, n);
}


void longinteger_domain::create_qnm1(
		longinteger_object &a, int q, int n)
// create (q^n - 1)
{
	longinteger_object b, c;
	
	b.create(q, __FILE__, __LINE__);
	power_int(b, n);
	c.create(-1, __FILE__, __LINE__);
	add(b, c, a);
}

void longinteger_domain::create_Mersenne(longinteger_object &M, int n)
// $M_n = 2^n - 1$
{
	longinteger_object a, b;

	a.create(2, __FILE__, __LINE__);
	b.create(-1, __FILE__, __LINE__);
	power_int(a, n);
	add(a, b, M);
	// cout << "Mersenne number M_" << n << "=" << a << endl;
}

void longinteger_domain::create_Fermat(longinteger_object &F, int n)
// $F_n = 2^{2^n} + 1$
{
	longinteger_object a, b;
	int l;

	a.create(2, __FILE__, __LINE__);
	b.create(1, __FILE__, __LINE__);
	l = 1 << n;
	// cout << "l=" << l << endl;
	power_int(a, l);
	add(a, b, F);
	// cout << "Fermat number F_" << n << "=" << a << endl;
}



void longinteger_domain::Dedekind_number(longinteger_object &Dnq,
		int n, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;
	longinteger_object A, S, B;

	if (f_v) {
		cout << "longinteger_domain::Dedekind_number n=" << n << " q=" << q << endl;
	}
	int *primes;
	int *exponents;
	int len, cnt;
	long int i, N, h, d, d_prime;
	int *v;

	len = NT.factor_int(n, primes, exponents);

	v = NEW_int(len);

	S.create(0, __FILE__, __LINE__);

	N = NT.i_power_j(2, len);
	if (f_v) {
		cout << "\\frac{1}{" << n << "} \\Big( ";
	}

	int f_first = true;
	for (i = 0; i < N; i++) {
		if (false) {
			cout << "i=" << i << endl;
		}
		Gg.AG_element_unrank(2, v, 1, len, i);
		d = 1;
		cnt = 0;
		for (h = 0; h < len; h++) {
			if (v[h]) {
				d *= primes[h];
				cnt++;
			}
		}
		d_prime = n / d;
		//a = NT.i_power_j(q, d_prime);
		create_q_to_the_n(A, q, d_prime);

		if (false) {
			cout << "d=" << d << " d_prime=" << d_prime << " A=" << A << " S=" << S << endl;
		}

		if (ODD(cnt)) {
			if (f_v) {
				cout << " - ";
			}
			subtract_in_place(S, A);
		}
		else {
			if (f_v) {
				if (!f_first) {
					cout << " + ";
				}
			}
			add_in_place(S, A);
		}
		if (f_first) {
			f_first = false;
		}
		if (f_v) {
			cout << "q^{" << d_prime << "}";
		}
	}
	if (f_v) {
		cout << "\\Big) ";
	}
	if (f_v) {
		cout << endl;
	}
	B.create(n, __FILE__, __LINE__);

	if (f_v) {
		cout << "S=" << S << endl;
	}
	integral_division_exact(S, B, Dnq);



	FREE_int(v);
	FREE_int(primes);
	FREE_int(exponents);


	if (f_v) {
		cout << "longinteger_domain::Dedekind_number done" << endl;
	}
}



int longinteger_domain::is_even(longinteger_object &a)
{
	if (((a.rep()[0] % 2)) == 0)
		return true;
	else
		return false;
}

int longinteger_domain::is_odd(longinteger_object &a)
{
	if (is_even(a))
		return false;
	else
		return true;
}



int longinteger_domain::remainder_mod_int(
		longinteger_object &a, int p)
{
	int r;
	longinteger_object q;
	
	integral_division_by_int(a, p, q, r);
	return r;
}

int longinteger_domain::multiplicity_of_p(
		longinteger_object &a,
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

long int longinteger_domain::smallest_primedivisor(
	longinteger_object &a,
	int p_min, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object n, n1, q, pp;
	long int p, r, cnt = 0;
	
	if (f_v) {
		cout << "longinteger_domain::smallest_primedivisor " << a
				<< " p_min=" << p_min << endl;
	}
	a.assign_to(n);
	
	if (p_min == 0) {
		p_min = 2;
	}
	if (p_min < 0) {
		p_min = - p_min;
	}
	if (p_min <= 2) {
		if (is_even(n)) {
			return 2;
		}
		p_min = 3;
	}
	if (p_min <= 3) {
		if (remainder_mod_int(n, 3) == 0) {
			return 3;
		}
		p_min = 5;
	}
	if (EVEN(p_min)) {
		p_min--;
	}
	p = p_min;
	while (true) {
		cnt++;
		if (f_vv) {
			cout << "longinteger_domain::smallest_primedivisor n=" << n
					<< " trying p=" << p << endl;
		}
		n.assign_to(n1);
		integral_division_by_lint(n1, p, q, r);
		if (f_vv && (cnt % 1) == 0) {
			cout << "longinteger_domain::smallest_primedivisor n=" << n1
				<< " trying p=" << p << " q=" << q
				<< " r=" << r << endl;
			cnt = 0;
		}
		if (r == 0) {
			return p;
		}
		pp.create(p, __FILE__, __LINE__);
		if (compare(q, pp) < 0) {
			break;
		}
		p += 2;
	}
	if (f_v) {
		cout << "longinteger_domain::smallest_primedivisor "
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
	long int p, last_prime = 0;
	int i;
	number_theory::number_theory_domain NT;
	
	if (f_v) {
		cout << "longinteger_domain::factor_into_longintegers factoring " << a << endl;
	}
	if (a.is_zero()) {
		cout << "longinteger_domain::factor_into_longintegers a is zero" << endl;
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
		p = n.as_lint();
	}
	pp.create(p, __FILE__, __LINE__);
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
			cout << "longinteger_domain::factor_into_longintegers "
					"remaining factor: " << n << endl;
		}
		p = smallest_primedivisor(n, last_prime, verbose_level);
		// if p == 0: n is prime
		
		if (p == 0) {
			p = n.as_lint();
		}
		if (p == last_prime) {
			exponents[nb_primes - 1]++;
			pp.create(p, __FILE__, __LINE__);
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
				pp.create(p, __FILE__, __LINE__);
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
		cout << "longinteger_domain::factor_into_longintegers "
				"prime factorization of " << a << " = ";
		NT.print_longfactorization(nb_primes, primes, exponents);
		cout << endl;
	}
}

void longinteger_domain::factor(
		longinteger_object &a,
	int &nb_primes, int *&primes, int *&exponents, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object n, q;
	int p, last_prime = 2, i, r;
	number_theory::number_theory_domain NT;
	
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

int longinteger_domain::jacobi(
		longinteger_object &a,
	longinteger_object &m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object a1, m1, a2, m2, a3, m3;
	longinteger_object u, v, g, q, r, res, minus_one;
	int n, rr, r1, t1, t2;
	
	if (f_v) {
		cout << "longinteger_domain::jacobi(" << a << " over " << m << ")" << endl;
	}
	a.assign_to(a1);
	m.assign_to(m1);
	r1 = 1;
	
	minus_one.create(-1, __FILE__, __LINE__);
	extended_gcd(a1, m1, g, u, v, verbose_level - 2);
	if (!g.is_one_or_minus_one()) {
		return 0;
	}
	
	while (true) {
		// we now have
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
			// t = (m1 * m1 - 1) >> 3 = (m1 * m1 - 1) / 8
			// multiply r1 by  (-1) to the power t:
			rr = remainder_mod_int(m1, 8);
			if (rr == 3 || rr == 5) {
				r1 = -r1; // note ABS(r1) == 1
			}
		}
		if (f_vv) {
			cout << r1 << " * Jacobi(" << a1 << ", " << m1 << ")" << endl;
		}
		if (a1.is_one_or_minus_one()) {
			break;
		}
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
			cout << r1 << " * Jacobi(" << a1 << ", " << m1 << ")" << endl;
		}
	}
	if (f_v) {
		cout << "jacobi(" << a << ", " << m << ") = " << r1 << endl;
	}
	return r1;
}

void longinteger_domain::random_number_less_than_n(
	longinteger_object &n, longinteger_object &r)
{
	int i, l, rr;
	//char *n_rep;
	char *r_rep;
	orbiter_kernel_system::os_interface Os;
	
	l = n.len();
	n.assign_to(r);
	//n_rep = n.rep();
	r_rep = r.rep();
	while (true) {
		for (i = l - 1; i >= 0; i--) {
			rr = Os.random_integer(10);
			r_rep[i] = (char) rr;
		}
		r.normalize();
		if (compare_unsigned(r, n) < 0) {
			break;
		}
	}
}

void longinteger_domain::random_number_with_n_decimals(
	longinteger_object &R, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *str;
	int i;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "longinteger_domain::random_number_with_n_decimals" << endl;
	}
	str = NEW_char(n + 1);
	for (i = 0; i <= n; i++) {
		str[n - i] = '0' + Os.random_integer(10);
		if (i == n) {
			str[n - i] = '1' + Os.random_integer(9);
		}
	}
	str[n] = 0;


	if (f_v) {
		cout << "longinteger_domain::random_number_with_n_decimals "
				"random number = " << str << endl;
	}

	R.create_from_base_10_string(str);

	FREE_char(str);

	if (f_v) {
		cout << "longinteger_domain::random_number_with_n_decimals done" << endl;
	}
}


void longinteger_domain::matrix_product(
		longinteger_object *A,
		longinteger_object *B,
		longinteger_object *&C, int Am, int An, int Bn)
{
	int i, j, k;
	longinteger_object a, b, c;
	
	for (i = 0; i < Am; i++) {
		for (j = 0; j < Bn; j++) {
			c.create(0, __FILE__, __LINE__);
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
		std::ostream &ost, longinteger_object *A,
		int Am, int An)
{
	int i, j;
	
	ost << "[";
	for (i = 0; i < Am; i++) {
		ost << "[";
		for (j = 0; j < An; j++) {
			ost << A[i * An + j];
			if (j < An - 1) {
				ost << ",";
			}
			ost << " ";
		}
		ost << "]";
		if (i < An - 1) {
			ost << ",";
		}
		ost << endl;
	}
	ost << "];" << endl;
}

void longinteger_domain::matrix_print_tex(
		std::ostream &ost, longinteger_object *A,
		int Am, int An)
{
	int i, j;
	
	ost << "\\begin{array}{*{" << An << "}{r}}" << endl;
	for (i = 0; i < Am; i++) {
		for (j = 0; j < An; j++) {
			ost << A[i * An + j];
			if (j < An - 1) {
				ost << " & ";
			}
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
	long int *x;
	int i, l;
	int p, e;
	number_theory::number_theory_domain NT;
	
	NT.factor_prime_power(q, p, e);
	l = n;
	if (f_semilinear) {
		l++;
	}
	
	x = NEW_lint(l);
	for (i = 0; i < n; i++) {
		x[i] = NT.i_power_j_lint(q, n) - NT.i_power_j_lint(q, i);
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

	multiply_up_lint(result, x, l, 0 /* verbose_level */);
	FREE_lint(x);
}





void longinteger_domain::square_root_floor(
		longinteger_object &a,
		longinteger_object &x, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "longinteger_domain::square_root_floor" << endl;
	}
	x.freeself();
	longinteger_object Y, YY;
	int la, l, len;
	char c1;

	a.normalize();
	if (a.sign()) {
		cout << "longinteger_domain::square_root_floor "
				"no square root, the number is negative" << endl;
		exit(1);
	}
	if (a.is_zero()) {
		x.create(0, __FILE__, __LINE__);
		return;
	}

	la = a.len();
	if (ODD(la)) {
		la++;
	}
	len = (la >> 1) + 1;
	Y.sign() = false;
	Y.len() = len;
	Y.rep() = NEW_char(len);

	//Y.allocate_empty(len);
	//Y.s_sign() = false;
	for (l = 0; l < len; l++) {
		Y.ith(l) = (char) 0;
	}

	for (l = len - 1; l >= 0; l--) {
		for (c1 = 9; c1 >= 0; c1--) {
			Y.ith(l) = c1;
			mult(Y, Y, YY);

			if (compare_unsigned(YY, a) <= 0) {
				break;
			}
		}
	}
	Y.normalize();
	Y.swap_with(x);
	if (f_v) {
		cout << "longinteger_domain::square_root_floor done" << endl;
	}
}




void longinteger_domain::print_digits(char *rep, int len)
{
	for (int h = 0; h < len; h++) {
		cout << (char)('0' + rep[h]) << " ";
	}
}


void longinteger_domain::Chinese_Remainders(
		std::vector<long int> &Remainders,
		std::vector<long int> &Moduli,
		longinteger_object &x, longinteger_object &M,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "longinteger_domain::Chinese_Remainders" << endl;
	}

	longinteger_object k, mr1, m1v;
	longinteger_object r1, r2;
	longinteger_object m1, m2;
	longinteger_object q, r1r, c, e;
	int i;

	r1.create(Remainders[0], __FILE__, __LINE__);
	m1.create(Moduli[0], __FILE__, __LINE__);
	x.create(Remainders[0], __FILE__, __LINE__);


	for (i = 1; i < Remainders.size(); i++) {

		r2.create(Remainders[i], __FILE__, __LINE__);
		m2.create(Moduli[i], __FILE__, __LINE__);


		integral_division(
				r1, m2,
				q, r1r,
				0 /*verbose_level*/);

		subtract_signless(m2, r1r, mr1);
				// c = a - b, assuming a > b

		if (f_v) {
			cout << "longinteger_domain::Chinese_Remainders i=" << i << " -r1 mod m2=" << mr1 << endl;
		}

		//mr1 = int_negate(r1, m2);

		inverse_mod(
				m1,
				m2, m1v, verbose_level);

		if (f_v) {
			cout << "longinteger_domain::Chinese_Remainders i=" << i << " m1^-1 mod m2=" << m1v << endl;
		}
		//m1v = inverse_mod(m1, m2);

		//k = mult_mod(m1v, add_mod(r2, mr1, m2), m2);


		add_mod(r2,
				mr1, c,
				m2, 0 /* verbose_level*/);

		if (f_v) {
			cout << "longinteger_domain::Chinese_Remainders i=" << i << " r2-r1=" << c << endl;
		}


		mult_mod(m1v,
				c, k,
				m2, 0 /* verbose_level*/);

		if (f_v) {
			cout << "longinteger_domain::Chinese_Remainders i=" << i << " m1^-1 * (r2-r1) = " << k << endl;
		}


		//x = r1 + k * m1;

		mult(k, m1, e);


		if (f_v) {
			cout << "longinteger_domain::Chinese_Remainders i=" << i << " k * m1 = " << e << endl;
		}

		add_in_place(r1, e);


		if (f_v) {
			cout << "longinteger_domain::Chinese_Remainders i=" << i << " r1 = " << r1 << endl;
		}

		r1.assign_to(x);

		//r1 = x;
		//m1 *= m2;
		mult_in_place(m1, m2);

		if (f_v) {
			cout << "longinteger_domain::Chinese_Remainders i=" << i << " x=" << x << " m1=" << m1 << endl;
		}


	}

	m1.assign_to(M);
	//M = m1;

	if (f_v) {
		cout << "longinteger_domain::Chinese_Remainders" << endl;
	}
}



}}}

