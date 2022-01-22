// longinteger.cpp
//
// Anton Betten
// 19.11.1999
// moved from D2 to ORBI Nov 15, 2007

#include "foundations/foundations.h"
#include "discreta.h"

using namespace std;


#undef DEBUG_LONGINTEGER_DIVISION
#undef DEBUG_LONGINTEGER_COMPARE

namespace orbiter {
namespace layer2_discreta {


static void subtract_signless(longinteger &x, longinteger &y, longinteger &z);
static int do_division(longinteger& r, longinteger *d);

longinteger::longinteger()
{
	k = LONGINTEGER;
	clearself();
}

longinteger::longinteger(int a)
{
	int l;
	char *p;
	ostringstream s;
	
	s << a << ends;
	l = s.str().length();
	p = new char [l + 1];
	s.str().copy(p, l, 0);
	p[l] = 0;
	longinteger x(p);
	swap(x);
	delete [] p;
}

longinteger::longinteger(const char *s)
{
	// cout << "longinteger::longinteger(char *s)" << s << endl;
	k = LONGINTEGER;
	clearself();
	if (s[0] == '-')
		allocate(TRUE, s + 1);
	else
		allocate(FALSE, s);
	normalize_representation();
}

longinteger::longinteger(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "longinteger::copy constructor for object: "
			<< const_cast<discreta_base &>(x) << "\n";
	clearself();
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

longinteger& longinteger::operator = (const discreta_base &x)
	// copy assignment
{
	// cout << "longinteger::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void longinteger::settype_longinteger()
{
	new(this) longinteger;
	k = LONGINTEGER;
}

longinteger::~longinteger()
{
	// cout << "~longinteger()\n";
	freeself_longinteger();
}

void longinteger::freeself_longinteger()
{
	// cout << "longinteger::freeself_longinteger()\n";
	if (s_rep())
		delete (char *) s_rep();
	self.longinteger_rep = NULL;
}

kind longinteger::s_virtual_kind()
{
	return LONGINTEGER;
}

void longinteger::copyobject_to(discreta_base &x)
{
	// cout << "longinteger::copyobject_to()\n";
	x.freeself();

	longinteger &xx = x.change_to_longinteger();
	xx.allocate_internal(s_sign(), s_len(), &s_p(0));
	xx.normalize_representation();
}

ostream& longinteger::print(ostream& ost)
{
	int i, l;
	char c;
		
#ifdef PRINT_WITH_TYPE
	ost << "(LONGINTEGER, ";
#endif
	if (s_rep() == NULL) {
		ost << "NULL";
		}
	else {
		if (s_sign())
			ost << "-";
		l = s_len();
		for (i = l - 1; i >= 0; i--) {
			c = '0' + s_rep()->p[i];
			ost << c;
#ifdef LONGINTEGER_PRINT_DOTS
			if (i && (i % LONGINTEGER_DIGITS_FOR_DOT) == 0)
				ost << ".";
#endif
			}
		}
#ifdef PRINT_WITH_TYPE
	ost << ")";
#endif
	return ost;
}

LONGINTEGER_REPRESENTATION *longinteger::s_rep()
{
	return self.longinteger_rep;
}

int& longinteger::s_sign()
{
	return self.longinteger_rep->sign;
}

int& longinteger::s_len()
{
	return self.longinteger_rep->len;
}

char& longinteger::s_p(int i)
{
	return self.longinteger_rep->p[i];
}

void longinteger::allocate(int sign, const char *p)
{
	int i, l;
	
	l = strlen(p);
	if (s_kind() != LONGINTEGER) {
		cout << "longinteger::allocate() s_kind() != LONGINTEGER\n";
		exit(1);
		}
	allocate_empty(l);
	s_sign() = sign;
	s_len() = l;
	for (i = 0; i < l; i++) {
		s_p(i) = p[l - 1 - i] - '0';
		}
}

void longinteger::allocate_internal(int sign, int len, const char *p)
{
	int i;
	
	if (s_kind() != LONGINTEGER) {
		cout << "longinteger::allocate_internal() s_kind() != LONGINTEGER\n";
		exit(1);
		}
	allocate_empty(len);
	s_sign() = sign;
	for (i = 0; i < len; i++) {
		s_p(i) = p[i];
		}
}

void longinteger::allocate_empty(int len)
{
	int i;
	
	if (s_kind() != LONGINTEGER) {
		cout << "longinteger::allocate_empty() s_kind() != LONGINTEGER\n";
		exit(1);
		}
	self.longinteger_rep = (LONGINTEGER_REPRESENTATION *)
			new char[sizeof(LONGINTEGER_REPRESENTATION) + len];
	s_sign() = FALSE;
	s_len() = len;
	for (i = 0; i < len; i++) {
		s_p(i) = (char) 0;
		}
}

void longinteger::normalize_representation()
{
	int i, l;
	
	l = s_len();
	for (i = l - 1; i > 0; i--) {
		if (s_p(i) != 0)
			break;
		}
	s_len() = i + 1;
}


int longinteger::compare_with(discreta_base &b)
{
	int sa, sb;
	
	if (s_kind() != LONGINTEGER) {
		cout << "longinteger::compare_with() s_kind() != LONGINTEGER\n";
		exit(1);
		}
	if (b.s_kind() != LONGINTEGER) {
		if (b.s_kind() != INTEGER) {
			cout << "longinteger::compare_with() "
					"b is neither longinteger nor integer\n";
			exit(1);
			}
		longinteger b1;
		
		b1.homo_z(b.s_i_i());
		return compare_with(b1);
		}
	longinteger &B = b.as_longinteger();
	int r;
	
	sa = s_sign();
	sb = B.s_sign();
	if (sa != sb) {
		if (sa)
			return -1;
		else
			return 1;
		}
	r = compare_with_unsigned(B);
#ifdef DEBUG_LONGINTEGER_COMPARE
	cout << "longinteger::compare_with_unsigned() returns " << r << "\n";
#endif
	if (sa)
		return -1 * r;
	else
		return r;
}

int longinteger::compare_with_unsigned(longinteger &b)
{
	int la, lb, l, i, d1, d2;
	
	la = s_len();
	lb = b.s_len();
	l = MAXIMUM(la, lb);
#ifdef DEBUG_LONGINTEGER_COMPARE
	cout << "longinteger::compare_with_unsigned()\n";
	cout << "la=" << la << " lb=" << lb << endl;
#endif

	for (i = l - 1; i >= 0; i--) {
		if (i < la)
			d1 = s_p(i);
		else
			d1 = 0;
		if (i < lb)
			d2 = b.s_p(i);
		else
			d2 = 0;
#ifdef DEBUG_LONGINTEGER_COMPARE
		cout << "i=" << i << "d1=" << d1 << "d2=" << d2 << endl;
#endif
		if (d1 < d2)
			return -1;
		else if (d1 > d2)
			return 1;
		}
	return 0;
}

void longinteger::mult_to(discreta_base &x, discreta_base &y)
{
	longinteger Y;
	int len;
	
	if (s_kind() != LONGINTEGER) {
		cout << "longinteger::mult_to() this is not a longinteger\n";
		exit(1);
		}
	if (x.s_kind() == INTEGER) {
		longinteger x1;
		
		x1.homo_z(x.s_i_i());
		mult_to(x1, y);
		return;
		}
	if (x.s_kind() != LONGINTEGER) {
		cout << "longinteger::mult_to() x is not a longinteger\n";
		exit(1);
		}
	longinteger &X = x.as_longinteger();
	
	len = s_len() + X.s_len() + 2;
	Y.allocate_empty(len);
	if ((s_sign() && X.s_sign()) || (!s_sign() && !X.s_sign())) {
		Y.s_sign() = FALSE;
		}
	else {
		Y.s_sign() = TRUE;
		}
	
	int la, lb;
	char cb, c1, carry;
	
	for (lb = 0; lb < X.s_len(); lb++) {
		cb = X.s_p(lb);
		carry = 0;
		for (la = 0; la < s_len(); la++) {
			c1 = s_p(la) * cb + carry + Y.s_p(la + lb);
			if (c1 > 100) {
				cout << "longinteger:mult_to() error: c1 >= 100\n";
				exit(1);
				}
			carry = c1 / 10;
			Y.s_p(la + lb) = c1 % 10;
			}
		if (carry) {
			if (Y.s_p(lb + s_len()) != 0) {
				cout << "longinteger:mult_to() error: "
						"carry && Y.s_p(lb + s_len()) != 0\n";
				exit(1);
				}
			Y.s_p(lb + s_len()) = carry;
			}
		}
	Y.normalize_representation();
	y.freeself();
	Y.swap(y);
}

int longinteger::invert_to(discreta_base &x)
{	
	if (s_kind() != LONGINTEGER) {
		cout << "longinteger::invert_to() this is not a longinteger\n";
		exit(1);
		}
	cout << "longinteger::invert_to() not yet implemented\n";
	return FALSE;
}

void longinteger::add_to(discreta_base &x, discreta_base &y)
{
	longinteger Y;
	int len, sign_a, sign_b, cmp_a_b, l;
	char ca, cb, c1, carry;
	
	if (s_kind() != LONGINTEGER) {
		cout << "longinteger::add_to() this is not a longinteger\n";
		exit(1);
		}
	if (x.s_kind() == INTEGER) {
		longinteger x1;
		
		x1.homo_z(x.s_i_i());
		add_to(x1, y);
		return;
		}
	if (x.s_kind() != LONGINTEGER) {
		cout << "longinteger::add_to() x is not a longinteger\n";
		exit(1);
		}
	longinteger &X = x.as_longinteger();
	len = MAXIMUM(s_len(), X.s_len()) + 1;
	Y.allocate_empty(len);
	sign_a = s_sign();
	sign_b = X.s_sign();
	if ((sign_a && sign_b) || (!sign_a && !sign_b)) {
		Y.s_sign() = sign_a;
		}
	else {
		// mixed signs: subtraction */
		cmp_a_b = compare_with_unsigned(X);
		if (cmp_a_b < 0) {
			// |this| < |X|
			
			subtract_signless(X, *this, Y);
			Y.s_sign() = X.s_sign();
			goto l_exit;
			}
		else if (cmp_a_b > 0) {  
			// |this| > |X|
		
			subtract_signless(*this, X, Y);
			Y.s_sign() = s_sign();
			goto l_exit;
			}
		else {
			// |a| = |b|
			Y.s_len() = 1;
			Y.s_sign() = FALSE;
			Y.s_p(0) = 0;
			goto l_exit;
			}
		}
	carry = 0;
	for (l = 0; l < len; l++) {
		if (l < s_len())
			ca = s_p(l);
		else
			ca = 0;
		if (l < X.s_len())
			cb = X.s_p(l);
		else
			cb = 0;
		c1 = ca + cb + carry;
		if (c1 >= 10)
			carry = 1;
		else
			carry = 0;
		Y.s_p(l) = c1 % 10;
		}
l_exit:
	Y.normalize_representation();
	y.freeself();
	Y.swap(y);
}

static void subtract_signless(longinteger &x, longinteger &y, longinteger &z)
// z := x - y (signless)
// assumes |x| > |y|
{
	int len, l;
	char ca, cb, carry;
	
	len = x.s_len();
	z.freeself_longinteger();
	z.allocate_empty(len);
	z.s_sign() = FALSE;
	carry = 0;
	for (l = 0; l < len; l++) {
		if (l < y.s_len())
			cb = y.s_p(l);
		else
			cb = 0;
		cb += carry;
		ca = x.s_p(l);
		if (cb > ca) {
			ca += 10;
			carry = 1;
			}
		else
			carry = 0;
		z.s_p(l) = ca - cb;
		}
	z.normalize_representation();
}

void longinteger::negate_to(discreta_base &x)
{
	if (s_kind() != LONGINTEGER) {
		cout << "longinteger::negate_to() this not an integer\n";
		exit(1);
		}
	x.freeself();
	longinteger& X = x.as_longinteger();
	
	X = *this;
	if (X.is_zero())
		return;
	if (X.s_sign())
		X.s_sign() = FALSE;
	else
		X.s_sign() = TRUE;
}

void longinteger::zero()
{
	longinteger x("0");
	swap(x);
}

void longinteger::one()
{
	longinteger x("1");
	swap(x);
}

void longinteger::m_one()
{
	longinteger x("-1");
	swap(x);
}

void longinteger::homo_z(int z)
{
	char str[1000];
	
	snprintf(str, 1000, "%d", z);
	
	longinteger x(str);
	swap(x);
}

void longinteger::inc()
{
	longinteger x = ("1");
	*this += x;
}

void longinteger::dec()
{
	longinteger x = ("-1");
	*this += x;
}

int longinteger::is_zero()
{
	longinteger x = ("0");
	
	if (compare_with(x) == 0)
		return TRUE;
	return FALSE;
}

int longinteger::is_one()
{
	longinteger x = ("1");
	
	if (compare_with(x) == 0)
		return TRUE;
	return FALSE;
}

int longinteger::is_m_one()
{
	longinteger x = ("-1");
	
	if (compare_with(x) == 0)
		return TRUE;
	return FALSE;
}

int longinteger::is_even()
{
	int d = (int) s_p(0);
	
	return EVEN(d);
}

int longinteger::is_odd()
{
	int d = (int) s_p(0);
	
	return ODD(d);
}

int longinteger::compare_with_euclidean(discreta_base &b)
{
	if (s_kind() != LONGINTEGER) {
		cout << "longinteger::compare_with_euclidean "
				"s_kind() != LONGINTEGER\n";
		exit(1);
		}
	if (b.s_kind() != LONGINTEGER) {
		if (b.s_kind() != INTEGER) {
			cout << "longinteger::compare_with_euclidean "
					"b is neither longinteger nor integer\n";
			exit(1);
			}
		longinteger b1;
		
		b1.homo_z(b.s_i_i());
		return compare_with_euclidean(b1);
		}
	longinteger &B = b.as_longinteger();
	int r;
	r = compare_with_unsigned(B);
	// cout << "longinteger::compare_with_unsigned() returns " << r << "\n";
	return r;
}

void longinteger::integral_division(discreta_base &x,
		discreta_base &q, discreta_base &r, int verbose_level)
{
	if (s_kind() != LONGINTEGER) {
		cout << "longinteger::integral_division "
				"this is not a longinteger\n";
		exit(1);
		}
	if (x.s_kind() == INTEGER) {
		longinteger x1;
		
		x1.homo_z(x.s_i_i());
		integral_division(x1, q, r, verbose_level);
		return;
		}
	if (x.s_kind() != LONGINTEGER) {
		cout << "longinteger::integral_division "
				"x is not a longinteger\n";
		exit(1);
		}
	longinteger &X = x.as_longinteger();
	longinteger Q, R, r1, r2, d[10], dm[10];
	int len, sign_x, sign_r, i, l1, l, idx;
	
	normalize_representation();
	X.normalize_representation();
	len = s_len() - X.s_len() + 1;
	if (len <= 0) {
		Q.zero();
		R = *this;
		q.freeself();
		r.freeself();
		Q.swap(q);
		R.swap(r);
		return;
		}
	Q.allocate_empty(len);
	Q.s_sign() = FALSE;
	
	if (s_sign() == X.s_sign()) {
		Q.s_sign() = FALSE;
		sign_r = s_sign();
		}
	else {
		Q.s_sign() = TRUE;
		sign_r = s_sign();
		}
	
	sign_x = X.s_sign();
	X.s_sign() = FALSE;
	for (i = 0; i < 10; i++) {
#ifdef DEBUG_LONGINTEGER_DIVISION
		cout << "i = " << i << " ";
#endif
		r1.homo_z(i);
#ifdef DEBUG_LONGINTEGER_DIVISION
		cout << "r1 = " << r1 << " ";
#endif
		d[i].mult(X, r1);
#ifdef DEBUG_LONGINTEGER_DIVISION
		cout << "d[i] = " << d[i] << " ";
#endif
		d[i].negate_to(dm[i]);
#ifdef DEBUG_LONGINTEGER_DIVISION
		cout << "dm[i] = " << dm[i] << endl;
#endif
		}
	X.s_sign() = sign_x;
	
	// load r1 with leading X.s_len() digits of this: 
	len = X.s_len();
	r1.freeself();
	r1.allocate_empty(len);
	l1 = s_len() - len;
	for (l = 0; l < len; l++) {
		r1.s_p(l) = s_p(l1 + l);
		}
#ifdef DEBUG_LONGINTEGER_DIVISION
	cout << "r1 = " << r1 << endl;
#endif
	
	
	// main loop containing all divisions:
	for ( ; l1 >= 0; l1--) {
#ifdef DEBUG_LONGINTEGER_DIVISION
		cout << "dividing r1=" << r1 << endl;
#endif
		idx = do_division(r1, d);
#ifdef DEBUG_LONGINTEGER_DIVISION
		cout << "do_division, idx = " << idx << endl;
		cout << "Q[" << l1 << "]=" << idx << endl;
#endif
		
		Q.s_p(l1) = (char) idx;
#ifdef DEBUG_LONGINTEGER_DIVISION
		cout << "calling r2.add()" << endl;
#endif
		r2.add(r1, dm[idx]);
#ifdef DEBUG_LONGINTEGER_DIVISION
		cout << "r2=" << r2 << endl;
#endif
		if (l1 == 0)
			break;
		
		// put r2 into r1, shift up by one digit 
		// and append the next digit l1 - 1 of this.
		r1.freeself_longinteger();
		len = r2.s_len() + 1;
		r1.allocate_empty(len);
		for (l = 0; l < r2.s_len(); l++) {
			r1.s_p(l + 1) = r2.s_p(l);
			}
		r1.s_p(0) = s_p(l1 - 1);
		r2.freeself_longinteger();
		}
	Q.normalize_representation();
	r2.normalize_representation();
	r2.s_sign() = sign_r;
	q.freeself();
	r.freeself();
	Q.swap(q);
	r2.swap(r);
}

static int do_division(longinteger& r, longinteger *d)
{
	int i, cmp;
	
	for (i = 9; i >= 0; i--) {
#ifdef DEBUG_LONGINTEGER_DIVISION
		cout << "do_division, i = " << i << " r=" << r << " d[i]=" << d[i] << endl;
#endif
		cmp = r.compare_with(d[i]);
		if (cmp >= 0)
			return i;
		}
	cout << "longinteger do_division() not found\n";
	exit(1);
}

void longinteger::square_root_floor(discreta_base &x)
{
	if (s_kind() != LONGINTEGER) {
		cout << "longinteger::square_root_floor "
				"this is not a longinteger\n";
		exit(1);
		}
	x.freeself();
	longinteger &X = x.change_to_longinteger();
	longinteger Y, YY;
	int la, l, len;
	char c1;
	
	normalize_representation();
	if (s_sign()) {
		cout << "longinteger::square_root_floor "
				"no square root, the number is negative\n";
		exit(1); 
		}
	if (is_zero()) {
		X.allocate(FALSE, "0");
		return;
		}
	
	la = s_len();
	if (ODD(la))
		la++;
	len = (la >> 1) + 1;
	Y.allocate_empty(len);
	Y.s_sign() = FALSE;
	for (l = 0; l < len; l++) {
		Y.s_p(l) = (char) 0;
		}
	
	for (l = len - 1; l >= 0; l--) {
		for (c1 = 9; c1 >= 0; c1--) {
			Y.s_p(l) = c1;
			YY.mult(Y, Y);
			if (YY.compare_with(*this) <= 0)
				break;
			}
		}
	Y.normalize_representation();
	Y.swap(X);
}

longinteger& longinteger::Mersenne(int n)
// $M_n = 2^n - 1$
{
	longinteger a = "2", b = "-1";
	
	a.power_int(n);
	a += b;
	// cout << "Mersenne number M_" << n << "=" << a << endl;
	swap(a);
	return *this;
}

longinteger& longinteger::Fermat(int n)
// $F_n = 2^{2^n} + 1$
{
	longinteger a = "2", b = "1", l = "2";
	
	l.power_int(n);
	// cout << "l=" << l << endl;
	a.power_longinteger(l);
	a += b;
	// cout << "Fermat number F_" << n << "=" << a << endl;
	swap(a);
	return *this;
}

int longinteger::s_i()
{
	char *p;
	int x, l;
	ostringstream s;
	
	s << *this << ends;
	// cout << "str=(" << str << ")" << endl;
	l = s.str().length();
	p = new char [l + 1];
	s.str().copy(p, l, 0);
	p[l] = 0;
	sscanf(p, "%d", &x);
	delete [] p;
	return x;
	
}

int longinteger::retract_to_integer_if_possible(integer &x)
{
	if (s_len() < 6) {
		int i = s_i();
		x.m_i(i);
		return TRUE;
		}
	else
		return FALSE;
}

int longinteger::modp(int p)
{
	longinteger P, Q, R;
	
	P.homo_z(p);
	integral_division(P, Q, R, 0);
	return R.s_i();
}

int longinteger::ny_p(int p)
{
	longinteger P, Q, R;
	int n = 0;
	
	P.homo_z(p);
	while (TRUE) {
		integral_division(P, Q, R, 0);
		if (!R.is_zero())
			break;
		n++;
		swap(Q);
		}
	return n;
}

void longinteger::divide_out_int(int d)
{
	longinteger D, Q, R;
	
	D.homo_z(d);
	integral_division(D, Q, R, 0);
	swap(Q);
}

int longinteger::Lucas_test_Mersenne(int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	longinteger s("4"), m2("-2"), t;
	
	Mersenne(m);
	if (f_v) 
		cout << "s_0 = " << s << endl;
	for (i = 1; i <= m - 2; i++) {
		t.mult(s, s);
		t += m2;
		t.modulo(*this);
		s = t;
		if (f_v)
			cout << "s_" << i << " = " << s << endl;
		}
	if (s.is_zero())
		return TRUE;
	else
		return FALSE;
}

}}

