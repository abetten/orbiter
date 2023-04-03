// integer.cpp
//
// Anton Betten
// 18.12.1998
// moved from D2 to ORBI Nov 15, 2007

#include "../layer2_discreta/discreta.h"
#include "layer1_foundations/foundations.h"

using namespace std;


namespace orbiter {
namespace layer2_discreta {
namespace typed_objects {


#undef INTEGER_M_I_VERBOSE

integer::integer()
{
	k = INTEGER;
	clearself();
}

integer::integer(char *p)
{
	int i = atoi(p);
	
	k = INTEGER;
	clearself();
	m_i((int) i);
}

integer::integer(long int i)
{
	k = INTEGER;
	clearself();
	m_i((int) i);
}

integer::integer(const discreta_base &x)
	// copy constructor:    this := x
{
	// cout << "integer::copy constructor for object: "
	//<< const_cast<discreta_base &>(x) << "\n";
	clearself();
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

integer& integer::operator = (const discreta_base &x)
	// copy assignment
{
	// cout << "integer::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void integer::settype_integer()
{
	// cout << "integer::settype_integer()\n";
	new(this) integer;
	k = INTEGER;
}

integer::~integer()
{
	// cout << "~integer()\n";
	freeself_integer();
}

void integer::freeself_integer()
{
	// cout << "integer::freeself_integer()\n";
	clearself();
}

kind integer::s_virtual_kind()
{
	return INTEGER;
}

void integer::copyobject_to(discreta_base &x)
{
	// cout << "integer::copyobject_to()\n";
	x.freeself();
	integer &xx = x.change_to_integer();
	xx.m_i( (int) s_i() );
}

ostream& integer::print(ostream& ost)
{
	domain *dom;
#ifdef PRINT_WITH_TYPE
	ost << "(INTEGER, ";
#endif
#if 0
	if (dom && dom->type == GFp) {
		ost << " mod " << dom->p.s_i_i();
		}
#endif
	if (is_GFq_domain(dom)) {
		unipoly a;
		domain *sub_domain;
		int p;
		
		sub_domain = dom->sub_domain();
		with w(sub_domain);
		p = sub_domain->order_int();
		
		a.numeric_polynomial(s_i(), p);
		ost << a;
		}
	else {
		ost << s_i();
		}

#ifdef PRINT_WITH_TYPE
	ost << ")";
#endif
	return ost;
}

integer& integer::m_i(long int i)
{
	if (s_kind() != INTEGER) {
		cout << "error: integer::m_i "
				"this not an integer, converting\n";
		exit(1);
		// settype_integer();
		}
	self.integer_value = i;
	return *this;
}

int integer::compare_with(discreta_base &a)
{
	long int i, j;
	//domain *dom;
	
	if (s_kind() != INTEGER) {
		return compare_with(a);
		}
	if (a.s_kind() != INTEGER) {
		if (a.s_kind() == LONGINTEGER) {
			int r = a.as_longinteger().compare_with(*this);
			return -r;
			}
		cout << "integer::compare_with "
				"a is neither integer nor longinteger\n";
		exit(1);
		}
#if 0
	if (is_GFp_domain(dom)) {
		m_i( remainder_mod(s_i(), dom->order_int()) );
		a.m_i_i( remainder_mod(a.s_i_i(), dom->order_int()) );
		}
#endif
	i = s_i();
	j = a.s_i_i();
	if (i < j) {
		return -1;
	}
	if (i > j) {
		return 1;
	}
	return 0;
}


void integer::mult_to(discreta_base &x, discreta_base &y, int verbose_level)
{
	domain *dom;
	
	if (x.s_kind() == INTEGER) {

		if (is_GFq_domain(dom)) {
			unipoly a, b, c;
			domain *sub_domain;
			int p, res;
		
			sub_domain = dom->sub_domain();
			with w(sub_domain);
			p = sub_domain->order_int();

			a.numeric_polynomial(s_i(), p);
			b.numeric_polynomial(x.s_i_i(), p);
			c.mult_mod(a, b, *dom->factor_poly(), verbose_level);
			res = c.polynomial_numeric(p);
			y.m_i_i(res);
			}

		else if (is_Orbiter_finite_field_domain(dom)) {
			y.m_i_i(dom->get_F()->mult(s_i(), x.s_i_i()));
		}
		else {


			int l1, l2, l3;
	
			l1 = log2();
			l2 = x.as_integer().log2();
			l3 = l1 + l2;
			if (l3 >= NB_BITS_THRESHOLD_FOR_LONGINTEGER) {
				longinteger a, b, c;
				
				a.homo_z(s_i());
				b.homo_z(x.s_i_i());
				a.mult_to(b, c, verbose_level);
				y = c;
				return;
				}
			else {
				if (is_GFp_domain(dom)) {
					// cout << "integer::mult() GFp domain" << endl;
					y.m_i_i( remainder_mod(s_i() * x.s_i_i(), dom->order_int()) );
					}
				else {
					y.m_i_i( s_i() * x.s_i_i() );
					}
				}
			}
		}
	else if (x.s_kind() == LONGINTEGER) {
		longinteger a, b, c;
			
		a.homo_z(s_i());
		b = x;
		a.mult_to(b, c, verbose_level);
		y = c;
		return;
		}
	else {
		cout << "integer::mult_to() objectkind of x:";
		x.printobjectkind(cout);
		cout << endl;
		exit(1);
		}
}

int integer::invert_to(discreta_base &x, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	domain *dom;
	
	if (f_v) {
		cout << "integer::invert_to" << endl;
	}
	if (s_kind() != INTEGER) {
		cout << "integer::invert_to() this not an integer" << endl;
		exit(1);
		}
	if (is_zero())
		return false;
	i = s_i();
	if (is_GFp_domain(dom)) {
		if (f_v) {
			cout << "integer::invert_to is_GFp_domain" << endl;
		}
		int a, p, av;

		a = s_i();
		if (f_v) {
			cout << "integer::invert_to a=" << a << endl;
		}
		p = dom->order_int();
		if (f_v) {
			cout << "integer::invert_to p=" << p << endl;
		}
		av = invert_mod_integer(a, p, verbose_level);
		if (f_v) {
			cout << "integer::invert_to av=" << av << endl;
		}
		x.m_i_i( av );
		return true;
		}
	else if (is_GFq_domain(dom)) {
		if (f_v) {
			cout << "integer::invert_to is_GFq_domain" << endl;
		}
		unipoly a;
		domain *sub_domain;
		int p, res;
		
		sub_domain = dom->sub_domain();
		with w(sub_domain);
		p = sub_domain->order_int();
	
		a.numeric_polynomial(s_i(), p);
#if 0
		// cout << "integer::invert_to() a=" << a << endl;
		// a.printobjectkind(cout);
		// cout << endl;
		a.invert_mod(*dom->factor_poly());
#else
		int q, l;
		
		q = dom->order_int();
		l = q - 2;
		a.power_int_mod(l, *dom->factor_poly(), verbose_level);
#endif
		res = a.polynomial_numeric(p);
		x.m_i_i(res);
		return true;
		}
	else if (is_Orbiter_finite_field_domain(dom)) {
		if (f_v) {
			cout << "integer::invert_to Orbiter_finite_field domain" << endl;
		}
		int a, av;
		a = s_i();
		av = dom->get_F()->inverse(a);
		x.m_i_i(av);
		return true;
		}
	if (i == 1 || i == -1) {
		x.m_i_i( i );
		return true;
		}
	else {
		cout << "integer::invert_to cannot invert " << *this << endl;
		exit(1);
		}
	return false;
}

void integer::add_to(discreta_base &x, discreta_base &y)
{
	domain *dom;
	
	if (x.s_kind() == INTEGER) {


		if (is_GFq_domain(dom)) {
			unipoly a, b, c;
			domain *sub_domain;
			int p, res;
		
			sub_domain = dom->sub_domain();
			with w(sub_domain);
			p = sub_domain->order_int();
	
			a.numeric_polynomial(s_i(), p);
			b.numeric_polynomial(x.s_i_i(), p);
			c.add(a, b);
			res = c.polynomial_numeric(p);
			y.m_i_i(res);
			}
		else if (is_Orbiter_finite_field_domain(dom)) {
			y.m_i_i(dom->get_F()->add(s_i(), x.s_i_i()));
		}
		else {
			int l1, l2, l3;
	
			l1 = log2();
			l2 = x.as_integer().log2();
			l3 = MAXIMUM(l1, l2) + 1;;
			if (l3 >= NB_BITS_THRESHOLD_FOR_LONGINTEGER) {
				longinteger a, b, c;
			
				a.homo_z(s_i());
				b.homo_z(x.s_i_i());
				a.add_to(b, c);
				y = c;
				return;
				}
			else {
				if (is_GFp_domain(dom)) {
					// cout << "integer::add_to() GFp domain" << endl;
					y.m_i_i( remainder_mod(s_i() + x.s_i_i(), dom->order_int()) );
					}
				else {
					y.m_i_i( s_i() + x.s_i_i() );
					}
				}
			}
		}
	else if (x.s_kind() == LONGINTEGER) {
		longinteger a, b, c;
			
		a.homo_z(s_i());
		b = x;
		a.add_to(b, c);
		y = c;
		return;
		}
	else {
		cout << "integer::add_to() objectkind of x:";
		x.printobjectkind(cout);
		cout << endl;
		exit(1);
		}
}

void integer::negate_to(discreta_base &x)
{
	long int i;
	domain *dom;
	
	if (s_kind() != INTEGER) {
		cout << "integer::negate_to() this not an integer\n";
		exit(1);
		}
	if (is_GFq_domain(dom)) {
		unipoly a;
		domain *sub_domain;
		int p, res;
		
		sub_domain = dom->sub_domain();
		with w(sub_domain);
		p = sub_domain->order_int();
	
		a.numeric_polynomial(s_i(), p);
		a.negate();
		res = a.polynomial_numeric(p);
		x.m_i_i(res);
		return;
		}
	else if (is_Orbiter_finite_field_domain(dom)) {
		x.m_i_i(dom->get_F()->negate(s_i()));
		return;
	}
	i = s_i();
	if (is_GFp_domain(dom)) {
		x.m_i_i( remainder_mod(-i, dom->order_int()));
		}
	else {
		x.m_i_i( - i );
		}
}

void integer::normalize(discreta_base &p)
{
	long int i, pp;
	
	i = s_i();
	pp = p.s_i_i();
	if (i < 0) {
		i *= -1;
		i %= pp;
		if (i == 0)
			m_i(0);
		else
			m_i((int)(pp - i));
		return;
		}
	i %= pp;
	m_i((int) i);
	return;
	
}

void integer::zero()
{
	domain *dom;
	
	if (is_GFp_domain(dom)) {
		m_i( 0 );
		}
	else if (is_GFq_domain(dom)) {
		m_i( 0 );
		}
	else if (is_Orbiter_finite_field_domain(dom)) {
		m_i( 0 );
	}
	else {
		m_i(0);
		}
}

void integer::one()
{
	domain *dom;
	
	if (is_GFp_domain(dom)) {
		m_i( 1 );
		}
	else if (is_GFq_domain(dom)) {
		m_i( 1 );
		}
	else if (is_Orbiter_finite_field_domain(dom)) {
		m_i( 1 );
	}
	else {
		m_i(1);
		}
}

void integer::m_one()
{
	one();
	negate();
}

void integer::homo_z(int z)
{
	domain *dom;
	
	if (is_GFp_domain(dom)) {
		m_i( remainder_mod(z, dom->order_int()));
		}
	else if (is_GFq_domain(dom)) {
		int p = finite_field_domain_characteristic(dom);
		cout << "homo_z in GFq, characteristic = " << p << endl;
		m_i( remainder_mod(z, p));
		// cout << "integer::homo_z() not allowed for GF(q) domain" << endl;
		// exit(1);
		}
	else if (is_Orbiter_finite_field_domain(dom)) {
		int p = finite_field_domain_characteristic(dom);
		m_i( remainder_mod(z, p));
	}
	else {
		m_i(z);
		}
}

void integer::inc()
{
	domain *dom;
	
	if (is_GFp_domain(dom)) {
		m_i( remainder_mod(s_i() + 1, dom->order_int()));
		}
	else if (is_GFq_domain(dom)) {
		cout << "integer::inc() not allowed for GF(q) domain" << endl;
		exit(1);
		}
	else if (is_Orbiter_finite_field_domain(dom)) {
		cout << "integer::inc() not allowed for finite_field domain" << endl;
		exit(1);
		}
	else {
		m_i( s_i() + 1);
		}
}

void integer::dec()
{
	domain *dom;
	
	if (is_GFp_domain(dom)) {
		m_i( remainder_mod(s_i() - 1, dom->order_int()));
		}
	else if (is_GFq_domain(dom)) {
		cout << "integer::dec() not allowed for GF(q) domain" << endl;
		exit(1);
		}
	else if (is_Orbiter_finite_field_domain(dom)) {
		cout << "integer::dec() not allowed for finite_field domain" << endl;
		exit(1);
		}
	else {
		m_i( s_i() - 1);
		}
}

int integer::is_zero()
{
	integer a; 
	
	a.zero();
	if (compare_with(a) == 0)
		return true;
	else
		return false;
}

int integer::is_one()
{
	integer a; 
	
	a.one();
	if (compare_with(a) == 0)
		return true;
	else
		return false;
}

int integer::is_m_one()
{
	integer a; 
	
	a.m_one();
	if (compare_with(a) == 0)
		return true;
	else
		return false;
}

int integer::compare_with_euclidean(discreta_base &a)
{
	int i, j;
	
	if (s_kind() != INTEGER) {
		return compare_with_euclidean(a);
		}
	if (a.s_kind() != INTEGER) {
		cout << "integer::compare_with_euclidean() a is not an integer\n";
		exit(1);
		}
	i = ABS(s_i());
	j = ABS(a.s_i_i());
	if (i < j)
		return -1;
	if (i > j)
		return 1;
	return 0;
}

void integer::integral_division(
		discreta_base &x,
		discreta_base &q, discreta_base &r,
		int verbose_level)
{
	int a, b, qq, rr;
	
	if (s_kind() != INTEGER) {
		cout << "integer::integral_division() this not an integer\n";
		exit(1);
		}
	if (x.s_kind() != INTEGER) {
		if (x.s_kind() == LONGINTEGER) {
			integer y;
			if (!x.as_longinteger().retract_to_integer_if_possible(y)) {
				cout << "integer::integral_division() "
						"longinteger x cannot be retracted to integer\n";
				cout << "x=" << x << endl;
				exit(1);
				}
			integral_division(y, q, r, verbose_level);
			return;
			}
		else {
			cout << "integer::integral_division() "
					"x is neither integer nor longinteger\n";
			exit(1);
			}
		}
	a = s_i();
	b = x.s_i_i();
	// cout << "integer::integral_division() a = " << a << ", b = " << b << "\n";
	if (b <= 0) {
		cout << "integer::integral_division() b = " << b << "\n";
		exit(1);
		}
	qq = a / b;
	rr = a - qq * b;
	q.m_i_i(qq);
	r.m_i_i(rr);
}

void integer::rand(int low, int high)
{
	int l = high + 1 - low;
	double r = (double) ::rand() * (double)l / RAND_MAX;
	
	m_i(low + (int) r);
}

int integer::log2()
{
	int a = ABS(s_i());
	int l = 0;
	
	while (a) {
		l++;
		a >>= 1;
		}
	return l;
}



}}}



