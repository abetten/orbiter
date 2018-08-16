// longinteger.C
//
// Anton Betten
//
// started:  October 26, 2002




#include "foundations.h"

INT longinteger_f_print_scientific = FALSE;

longinteger_object::longinteger_object()
{
	sgn = FALSE;
	l = 0;
	r = NULL;
}

longinteger_object::~longinteger_object()
{
	freeself();
}

void longinteger_object::freeself()
{
	INT f_v = FALSE;
	
	if (r) {
		if (f_v) {
			cout << "longinteger_object::freeself ";
			longinteger_print_digits(rep(), len());
			cout << endl;
			}
		FREE_BYTE(r);
		r = NULL;
		}
	if (f_v) {
		cout << "longinteger_object::freeself" << endl;
		}
}

void longinteger_object::create(INT i)
{
	INT ii, j, dj;
	INT f_v = FALSE;

	ii = i;
	freeself();
	if (i < 0) {
		sgn = TRUE;
		i = -i;
		}
	else {
		sgn = FALSE;
		}
	if (i == 0) {
		r = NEW_BYTE(1);
		r[0] = 0;
		l = 1;
		return;
		}
	l = (int) INT_log10(i);
	if (f_v) {
		cout << "longinteger_object::create i=" << i << " log =  " << l << endl;
		}
	r = NEW_BYTE(l);
	j = 0;
	while (i) {
		dj = i % 10;
		r[j] = dj;
		i /= 10;
		j++;
		}
	if (f_v) {
		cout << "longinteger_object::create i=" << ii << " created ";
		longinteger_print_digits(rep(), len());
		cout << " with j=" << j << " digits" << endl;
		}
}

void longinteger_object::create_product(INT nb_factors, INT *factors)
{
	longinteger_domain D;
	
	D.multiply_up(*this, factors, nb_factors);
}

void longinteger_object::create_power(INT a, INT e)
// creates a^e
{
	longinteger_domain D;
	INT *factors;
	INT i;

	factors = NEW_INT(e);
	for (i = 0; i < e; i++) {
		factors[i] = a;
		}
	
	D.multiply_up(*this, factors, e);

	FREE_INT(factors);
}

void longinteger_object::create_power_minus_one(INT a, INT e)
// creates a^e  - 1
{
	longinteger_domain D;
	longinteger_object A;
	INT *factors;
	INT i;

	A.create(-1);
	factors = NEW_INT(e);
	for (i = 0; i < e; i++) {
		factors[i] = a;
		}
	
	D.multiply_up(*this, factors, e);
	D.add_in_place(*this, A);

	FREE_INT(factors);
}

void longinteger_object::create_from_base_b_representation(INT b, INT *rep, INT len)
{
	longinteger_domain D;
	longinteger_object x, y, z, bb;
	INT i;
	
	x.zero();
	bb.create(b);
	for (i = len - 1; i >= 0; i--) {
		D.mult(x, bb, z);
		y.create(rep[i]);
		D.add(z, y, x);
		}
	x.assign_to(*this);
}

void longinteger_object::create_from_base_10_string(const BYTE *str, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object x, y, z, bb;
	INT i, len;
	
	len = strlen(str);
	x.zero();
	bb.create(10);
	for (i = len - 1; i >= 0; i--) {
		D.mult(x, bb, z);
		y.create(str[len - 1 - i] - '0');
		D.add(z, y, x);
		}
	if (f_v) {
		cout << "longinteger_object::create_from_base_10_string() str = " << str << endl;
		cout << "object = " << x << endl;
		}
	x.assign_to(*this);
}

void longinteger_object::create_from_base_10_string(const BYTE *str)
{
	create_from_base_10_string(str, 0);
}

INT longinteger_object::as_INT()
{
	INT i, x = 0;
	
	for (i = l - 1; i >= 0; i--) {
		x *= 10;
		x += r[i];
		}
	if (sgn) {
		x = -x;
		}
	return x;
}

void longinteger_object::assign_to(longinteger_object &b)
{
	INT i;
	INT f_v = FALSE;
	
	if (f_v) {
		cout << "longinteger_object::assign_to before b.freeself()" << endl;
		if (b.rep()) {
			cout << "this is what we free: ";
			longinteger_print_digits(b.rep(), b.len());
			cout << endl;
			}
		}
	b.freeself();
	if (f_v) {
		cout << "longinteger_object::assign_to after b.freeself()" << endl;
		}
	b.sgn = sgn;
	b.l = l;
	b.r = NEW_BYTE(b.l);
	for (i = 0; i < l; i++) {
		b.r[i] = r[i];
		}
	if (f_v) {
		cout << "after assign: ";
		longinteger_print_digits(b.rep(), b.len());
		cout << endl;
		cout << "longinteger_object::assign_to done" << endl;
		}
}

void longinteger_object::swap_with(longinteger_object &b)
{
	char s;
	INT length;
	char *rep;
	
	s = sgn;
	length = l;
	rep = r;
	sgn = b.sgn;
	l = b.l;
	r = b.r;
	b.sgn = s;
	b.l = (int) length;
	b.r = rep;
}

ostream& longinteger_object::print(ostream& ost)
{
	INT i;
	char c;
		
	if (r == NULL) {
		ost << "NULL";
		}
	else {
		if (sgn)
			ost << "-";
		for (i = l - 1; i >= 0; i--) {
			c = '0' + r[i];
			ost << c;
			}
		if (longinteger_f_print_scientific) {
			if (l > 5) {
				char c1, c2;

				c1 = '0' + r[l - 1];
				c2 = '0' + r[l - 2];
				ost << " = " << c1 << "." << c2 << " x 10^" << l - 1;
				}
			}
		}
	return ost;
}

ostream& longinteger_object::print_not_scientific(ostream& ost)
{
	INT i;
	char c;
		
	if (r == NULL) {
		ost << "NULL";
		}
	else {
		if (sgn)
			ost << "-";
		for (i = l - 1; i >= 0; i--) {
			c = '0' + r[i];
			ost << c;
			}
		}
	return ost;
}

INT longinteger_object::output_width()
{
	INT h;
	
	h = l;
	if (sgn)
		h++;
	return h;
}

void longinteger_object::print_width(ostream& ost, INT width)
{
	INT i, len, w;
	char c;
		
	if (r == NULL) {
		len = width - 4;
		for (i = 0; i < len; i++)
			ost << " ";
		ost << "NULL";
		return;
		}
	w = output_width();
	if (w > width) {
		len = width - 5;
		for (i = 0; i < len; i++)
			ost << " ";
		ost << "large";
		}
	else {
		len = width - w;
		for (i = 0; i < len; i++)
			ost << " ";
		if (sgn)
			ost << "-";
		for (i = l - 1; i >= 0; i--) {
			c = '0' + r[i];
			ost << c;
			}
		}
}

void longinteger_object::print_to_string(BYTE *str)
{
	INT i, j = 0;
	char c;
		
	if (r == NULL) {
		str[0] = 0;
		}
	else {
		if (sgn) {
			str[j++] = '-';
			}
		for (i = l - 1; i >= 0; i--) {
			c = '0' + r[i];
			str[j++] = c;
			}
		}
	str[j] = 0;
}

void longinteger_object::normalize()
{
	INT i;
	
	for (i = l - 1; i > 0; i--) {
		if (r[i] != 0)
			break;
		}
	l = (int) i + 1;
	if (l == 1 && r[0] == 0) {
		sgn = FALSE;
		}
}

void longinteger_object::negate()
{
	if (is_zero())
		return;
	if (sign())
		sign() = FALSE;
	else
		sign() = TRUE;
	
}

int longinteger_object::is_zero()
{
	normalize();
	if (l == 1 && r[0] == 0)
		return TRUE;
	else
		return FALSE;
}

void longinteger_object::zero()
{
	create(0);
}

int longinteger_object::is_one()
{
	normalize();
	if (!sgn && l == 1 && r[0] == 1)
		return TRUE;
	else
		return FALSE;
}

int longinteger_object::is_mone()
{
	normalize();
	if (sgn && l == 1 && r[0] == 1)
		return TRUE;
	else
		return FALSE;
}

int longinteger_object::is_one_or_minus_one()
{
	normalize();
	if (l == 1 && r[0] == 1)
		return TRUE;
	else
		return FALSE;
}

void longinteger_object::one()
{
	create(1);
}

void longinteger_object::increment()
{
	longinteger_object b, c;
	longinteger_domain D;
	
	b.create(1);
	D.add(*this, b, c);
	swap_with(c);
}

void longinteger_object::decrement()
{
	longinteger_object b, c;
	longinteger_domain D;
	
	b.create(-1);
	D.add(*this, b, c);
	swap_with(c);
}

void longinteger_object::add_INT(INT a)
{
	longinteger_object b, c;
	longinteger_domain D;
	
	b.create(a);
	D.add(*this, b, c);
	swap_with(c);
}

void longinteger_object::create_i_power_j(INT i, INT j)
{
	longinteger_domain D;
	
	create(i);
	D.power_int(*this, j);
}


ostream& operator<<(ostream& ost, longinteger_object& p)
{
	// cout << "operator<< starting\n";
	p.print(ost);
	// cout << "operator<< finished\n";
	return ost;
};

INT longinteger_object::compare_with_INT(INT a)
{
	longinteger_domain D;
	longinteger_object b;
	
	b.create(a);
	return D.compare(*this, b);
}


