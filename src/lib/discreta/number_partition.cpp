// number_partition.cpp
//
// Anton Betten
// 23.03.2000
// moved from D2 to ORBI Nov 15, 2007

#include "foundations/foundations.h"
#include "discreta.h"


using namespace std;


#undef PARTITION_CHANGE_KIND_VERBOSE
#undef PARTITION_COPY_VERBOSE


namespace orbiter {
namespace layer2_discreta {

number_partition::number_partition() : Vector()
{
	k = NUMBER_PARTITION;
	self.vector_pointer = NULL;
}

number_partition::number_partition(int n) : Vector()
{
	k = NUMBER_PARTITION;
	self.vector_pointer = NULL;
	allocate_number_partition();
	first(n);
}

void number_partition::allocate_number_partition()
{
	// cout << "number_partition::allocate_number_partition()\n";
	// c_kind(VECTOR);
	Vector::m_l(2);
	c_kind(NUMBER_PARTITION);
	Vector::s_i(0).change_to_integer().m_i(PARTITION_TYPE_EXPONENT);
	Vector::s_i(1).change_to_vector();
}

number_partition::number_partition(const discreta_base &x)
	// copy constructor:    this := x
{
	// cout << "number_partition::copy constructor for object: " << const_cast<discreta_base &>(x) << "\n";
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

number_partition& number_partition::operator = (const discreta_base &x)
	// copy assignment
{
	// cout << "number_partition::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void number_partition::settype_number_partition()
{
	OBJECTSELF s;
	
	s = self;
	new(this) number_partition;
	self = s;
	k = NUMBER_PARTITION;
}

number_partition::~number_partition()
{
	freeself_number_partition();
}

void number_partition::freeself_number_partition()
{
	// cout << "partition::freeself_partition()\n";
	freeself_vector();
}

kind number_partition::s_virtual_kind()
{
	return NUMBER_PARTITION;
}

void number_partition::copyobject_to(discreta_base &x)
{
#ifdef PARTITION_COPY_VERBOSE
	cout << "number_partition::copyobject_to()\n";
	print_as_vector(cout);
#endif
	Vector::copyobject_to(x);
	x.as_number_partition().settype_number_partition();
#ifdef PARTITION_COPY_VERBOSE
	x.as_number_partition().print_as_vector(cout);
#endif
}

ostream& number_partition::print(ostream& ost)
{
	int i, l, a;
	
	l = s_self().s_l();
	if (s_type() == PARTITION_TYPE_VECTOR) {
		ost << "[";
		for (i = 0; i < l; i++) {
			a = s_i(i);
			ost << a;
			if (i < l)
				ost << ", ";
			}
		ost << "]";
		}
	else {
		ost << "(";
		int f_first = TRUE;
		for (i = 0; i < l; i++) {
			a = s_i(i);
			if (a == 0)
				continue;
			if (!f_first)
				ost << ", ";
			ost << (i + 1);
			if (a > 1) {
				if (current_printing_mode() == printing_mode_latex) {
					ost << "^" << a;
					}
				else {
					ost << "^{" << a << "}";
					}
				}
			f_first = FALSE;
			}
		ost << ")";
		}
	return ost;
}

void number_partition::first(int n)
{
	// cout << "number_partition::first()\n";
	allocate_number_partition();
	s_type() = PARTITION_TYPE_EXPONENT;
	m_l(n);
	s_i(n - 1) = 1;
}

int number_partition::next()
{
	// cout << "number_partition::next()\n";
	if (s_type() == PARTITION_TYPE_EXPONENT)
		return next_exponent();
	else
		return next_vector();
}

int number_partition::next_exponent()
{
	int i, j, n, a, s;
	// cout << "number_partition::next_exponent()\n";
	n = s_self().s_l();
	s = s_i(0);
	for (i = 1; i < n; i++) {
		a = s_i(i);
		if (a > 0) {
			a--;
			s += (i + 1);
			s_i(i) = a;
			for (j = i - 1; j >= 0; j--) {
				a = s / (j + 1);
				s -= a * (j + 1);
				s_i(j) = a;
				}
			return TRUE;
			}
		}
	return FALSE;
}

int number_partition::next_vector()
{
	// cout << "number_partition::next_vector()\n";
	return FALSE;
}

int number_partition::first_into_k_parts(int n, int k)
{
	int s;
	
	first(n);
	do {
		s = nb_parts();
		if (s == k)
			return TRUE;
		} while (next());
	return FALSE;
}

int number_partition::next_into_k_parts(int n, int k)
{
	int s;
	
	while (next()) {
		s = nb_parts();
		if (s == k)
			return TRUE;
		}
	return FALSE;
}

int number_partition::first_into_at_most_k_parts(int n, int k)
{
	int s;
	
	first(n);
	do {
		s = nb_parts();
		if (s <= k)
			return TRUE;
		} while (next());
	return FALSE;
}

int number_partition::next_into_at_most_k_parts(int n, int k)
{
	int s;
	
	while (next()) {
		s = nb_parts();
		if (s <= k)
			return TRUE;
		}
	return FALSE;
}

int number_partition::nb_parts()
{
	int i, n, s = 0;
	
	n = s_l();
	for (i = 0; i < n; i++) {
		// cout << "number_partition::nb_parts() i=" << i << ", s_i=" << s_i(i) << endl;
		s += s_i(i);
		}
	return s;
}

void number_partition::conjugate()
{
	Vector q;
	int i, ii = 0, n, s, a;
	
	n = s_l();
	q.m_l_n(n);
	s = nb_parts();
	// cout << "number_partition::conjugate() s=" << s << endl;
	for (i = 1; i <= n; i++) {
		a = s_i(i - 1);
		if (a) {
			q.m_ii(s - 1, i - ii);
			// cout << "partition::conjugate() q[" << s - 1 << "] = " << i - ii << endl;
			ii = i;
			s -= a;
			}
		}
	q.swap(s_self());
}

void number_partition::type(number_partition &q)
{
	int s, i, n, a;
	
	s = nb_parts();
	q.allocate_number_partition();
	q.m_l(s);
	n = s_l();
	for (i = 0; i < n; i++) {
		a = s_i(i);
		if (a)
			q.s_i(a - 1)++;
		}
}

void number_partition::multinomial(discreta_base &res, int f_v)
{
	discreta_base a, b, c;
	int i, n, m;
	
	n = s_l();
	a.factorial(n);
#if 0
	if (f_v) {
		cout << "multinomial() factorial(" << n << ")=" << a << endl;
		}
#endif
	b.m_i_i(1);
	for (i = 1; i <= n; i++) {
		m = s_i(i - 1);
		if (m == 0)
			continue;
		c.factorial(i);
		c.power_int(m);
		b *= c;
		}
	a.integral_division_exact(b, res); 
	if (f_v) {
		cout << "multinomial(" << *this << ") = " << res << endl;
		}
}

void number_partition::multinomial_ordered(discreta_base &res, int f_v)
{
	number_partition q;
	discreta_base a;
	
	type(q);
	
	multinomial(res, f_v);
	q.multinomial(a, f_v);
	res *= a;
	if (f_v) {
		cout << "multinomial_ordered(" << *this << ") = " << res << endl;
		}
}

int number_partition::sum_of_decreased_parts()
{
	int i, n, s;
	
	s = 0;
	if (s_type() != PARTITION_TYPE_EXPONENT) {
		cout << "number_partition::sum_of_decreased_parts() not of type exponent\n";
		exit(1);
		}
	n = s_l();
	for (i = 2; i <= n; i++) {
		s += s_i(i - 1) * (i - 1);
		}
	return s;
}


}}
