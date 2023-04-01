// base.cpp
//
// Anton Betten
// 18.12.1998
// moved from D2 to ORBI Nov 15, 2007


#include "../layer2_discreta/discreta.h"
#include "layer1_foundations/foundations.h"

#undef BASE_SETTYPE_VERBOSE

using namespace std;

namespace orbiter {
namespace layer2_discreta {
namespace typed_objects {


discreta_base::discreta_base()
{
	k = BASE;
	clearself();
}

discreta_base::discreta_base(const discreta_base &x)
	// copy constructor:    this := x
{
	// cout << "discreta_base::copy constructor for object: " << x << "\n";
	clearself();
#if 0
	if (x.k != x.s_virtual_kind()) {
		x.c_kind(k);
		}
#endif
	// cout << "discreta_base::copy constructor, calling copyobject_to()\n";
	const_cast<discreta_base &>(x).copyobject_to(*this);
	// cout << "discreta_base::copy constructor finished\n";
}

discreta_base& discreta_base::operator = (const discreta_base &x)
	// copy assignment
{
	// cout << "discreta_base::operator = (copy assignment)" << endl;
	// cout << "source=" << x << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

discreta_base::~discreta_base()
{
	// cout << "discreta_base::~discreta_base()\n";
	// printobjectkindln(cout);
	// cout << endl;
	freeself_kind(k); // virtual kind may be different form k ! */
}

void discreta_base::freeself_discreta_base()
{
	// cout << "discreta_base::freeself_discreta_base()\n";
	// printobjectkindln(cout);
	// cout << "self=" << self.vector_pointer << endl;
	if (s_kind() != BASE) {
		cout << "freeself() not implemented for class ";
		printobjectkindln(cout);
		exit(1);
		// freeself();
		// return;
		}
	clearself();
}

void discreta_base::freeself()
{
	freeself_kind(s_kind());
}

void discreta_base::freeself_kind(kind k)
{
	switch (k) {
		case BASE: freeself_discreta_base(); break;
		case INTEGER: as_integer().freeself_integer(); break;
		case VECTOR: as_vector().freeself_vector(); break;
		case NUMBER_PARTITION: as_number_partition().freeself_number_partition(); break;
		case PERMUTATION: as_permutation().freeself_permutation(); break;
		case MATRIX: as_matrix().freeself_matrix(); break;
		case LONGINTEGER: as_longinteger().freeself_longinteger(); break;
		case MEMORY: as_memory().freeself_memory(); break;
		//case PERM_GROUP: as_perm_group().freeself_perm_group(); break;
		//case PERM_GROUP_STAB_CHAIN: as_perm_group_stab_chain().freeself_perm_group_stab_chain(); break;
		case UNIPOLY: as_unipoly().freeself_unipoly(); break;
		//case SOLID: as_solid().freeself_solid(); break;
		//case BITMATRIX: as_bitmatrix().freeself_bitmatrix(); break;
		//case PC_PRESENTATION: as_pc_presentation().freeself_pc_presentation(); break;
		//case PC_SUBGROUP: as_pc_subgroup().freeself_pc_subgroup(); break;
		//case GROUP_WORD: as_group_word().freeself_group_word(); break;
		//case GROUP_TABLE: as_group_table().freeself_group_table(); break;
		// case ACTION: as_action().freeself_action(); break;
		//case GEOMETRY: as_geometry().freeself_geometry(); break;
		case HOLLERITH: as_hollerith().freeself_hollerith(); break;
		//case GROUP_SELECTION: as_group_selection().freeself_group_selection(); break;
		case BT_KEY: as_bt_key().freeself_bt_key(); break;
		case DATABASE: as_database().freeself_database(); break;
		case BTREE: as_btree().freeself_btree(); break;
		case DESIGN_PARAMETER_SOURCE: as_design_parameter_source().freeself_design_parameter_source(); break;
		case DESIGN_PARAMETER: as_design_parameter().freeself_design_parameter(); break;
		default: cout << "discreta_base::freeself_kind(), unknown kind: k= " << kind_ascii(k) << "\n";
		}
}

void discreta_base::settype_base()
{
#ifdef BASE_SETTYPE_VERBOSE
	if (s_kind() != BASE) {
		cout << "warning: base::settype_base converting from "
			<< kind_ascii(s_kind()) << " to BASE\n";
		}
#endif
	new(this) discreta_base;
}

kind discreta_base::s_kind()
{
	kind kv;
	
	kv = s_virtual_kind();
	if (k != kv) {
		cout << "discreta_base::s_kind "
				"kind != virtual kind\n";
		cout << "k=" << kind_ascii(k)
				<< ", virtual kind = " << kind_ascii(kv) << endl;
		exit(1);
		}
	return k;
}

kind discreta_base::s_virtual_kind()
{
	return BASE;
}

void discreta_base::c_kind(kind k)
{
	// cout << "discreta_base::c_kind(), k= " << kind_ascii(k) << "\n";
	switch (k) {
		case BASE: settype_base(); break;
		case INTEGER: as_integer().settype_integer(); break;
		case VECTOR: as_vector().settype_vector(); break;
		case NUMBER_PARTITION: as_number_partition().settype_number_partition(); break;
		case PERMUTATION: as_permutation().settype_permutation(); break;
		case MATRIX: as_matrix().settype_matrix(); break;
		case LONGINTEGER: as_longinteger().settype_longinteger(); break;
		case MEMORY: as_memory().settype_memory(); break;
		//case PERM_GROUP: as_perm_group().settype_perm_group(); break;
		//case PERM_GROUP_STAB_CHAIN: as_perm_group_stab_chain().settype_perm_group_stab_chain(); break;
		case UNIPOLY: as_unipoly().settype_unipoly(); break;
		//case SOLID: as_solid().settype_solid(); break;
		//case BITMATRIX: as_bitmatrix().settype_bitmatrix(); break;
		//case PC_PRESENTATION: as_pc_presentation().settype_pc_presentation(); break;
		//case PC_SUBGROUP: as_pc_subgroup().settype_pc_subgroup(); break;
		//case GROUP_WORD: as_group_word().settype_group_word(); break;
		//case GROUP_TABLE: as_group_table().settype_group_table(); break;
		// case ACTION: as_action().settype_action(); break;
		//case GEOMETRY: as_geometry().settype_geometry(); break;
		case HOLLERITH: as_hollerith().settype_hollerith(); break;
		//case GROUP_SELECTION: as_group_selection().settype_group_selection(); break;
		case BT_KEY: as_bt_key().settype_bt_key(); break;
		case DATABASE: as_database().settype_database(); break;
		case BTREE: as_btree().settype_btree(); break;
		case DESIGN_PARAMETER_SOURCE: as_design_parameter_source().settype_design_parameter_source(); break;
		case DESIGN_PARAMETER: as_design_parameter().settype_design_parameter(); break;
		default: cout << "discreta_base::c_kind(), k= " << kind_ascii(k) << " unknown\n";
		}
	if (s_kind() != k) {
		cout << "discreta_base::c_kind() did not work\n";
		}
	// cout << "discreta_base::c_kind() finished \n";
}

void discreta_base::swap(discreta_base &a)
{
	kind k, ka;
	OBJECTSELF s, sa;
	
	k = s_kind();
	ka = a.s_kind();
	s = self;
	sa = a.self;
	c_kind(ka);
	self = sa;
	a.c_kind(k);
	a.self = s;
}

void discreta_base::copyobject(discreta_base &x)
// this := x
{
	// cout << "discreta_base::copyobject\n";
	// cout << "source=" << x << endl;
	x.copyobject_to(*this);
}

void discreta_base::copyobject_to(discreta_base &x)
{
	kind k = s_kind();
	OBJECTSELF s = self;
	
	if (k != BASE) {
		cout << "error: discreta_base::copyobject_to "
				"for object of kind " << kind_ascii(k) << endl;
		exit(1);
		}
	cout << "warning: discreta_base::copyobject_to "
			"for object: " << *this << "\n";
	x.freeself();
	x.c_kind(k);
	x.self = s;
}

ostream& discreta_base::print(ostream& ost)
{
	ost << "object of kind BASE";
	return ost;
}

ostream& discreta_base::println(ostream &ost)
{
	print(ost) << endl;
	return ost;
}

void discreta_base::print_to_hollerith(hollerith& h)
{
	ostringstream s;
	int l;
	char *p;
	
	s << *this << ends;
	l = (int) s.str().length();
	p = new char [l + 1];
	s.str().copy(p, l, 0);
	p[l] = 0;
	h.init(p);
	delete [] p;
}

ostream& discreta_base::printobjectkind(ostream& ost)
{
	orbiter::layer2_discreta::typed_objects::printobjectkind(ost, s_kind());
	return ost;
}

ostream& discreta_base::printobjectkindln(ostream& ost)
{
	printobjectkind(ost) << "\n";
	return ost;
}

long int & discreta_base::s_i_i()
{
	if (s_kind() != INTEGER) {
		cout << "discreta_base::s_i_i not an integer, objectkind=";
		printobjectkindln(cout);
		exit(1);
		}
	return as_integer().s_i();
}

void discreta_base::m_i_i(long int i)
{
	change_to_integer().m_i(i);
}


int discreta_base::compare_with(discreta_base &a)
{
	if (s_kind() != BASE) {
		cout << "compare_with() not implemented for class ";
		printobjectkindln(cout);
		exit(1);
		// return compare_with(a);
		}
	NOT_EXISTING_FUNCTION("discreta_base::compare_with");
	exit(1);
	return 0;
}

int discreta_base::eq(discreta_base &a)
{
	int r = compare_with(a);
	if (r == 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int discreta_base::neq(discreta_base &a)
{
	int r = compare_with(a);
	if (r != 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int discreta_base::le(discreta_base &a)
{
	int r = compare_with(a);
	if (r <= 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int discreta_base::lt(discreta_base &a)
{
	//cout << "lt(): " << *this << ", " << a;
	int r = compare_with(a);
	//cout << " r=" << r << endl;
	if (r < 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int discreta_base::ge(discreta_base &a)
{
	int r = compare_with(a);
	if (r >= 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int discreta_base::gt(discreta_base &a)
{
	int r = compare_with(a);
	if (r > 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int discreta_base::is_even()
{
	discreta_base a, q, r;
	
	a.m_i_i(2);
	integral_division(a, q, r, 0);
	if (r.is_zero()) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int discreta_base::is_odd()
{
	if (is_even()) {
		return FALSE;
	}
	else {
		return TRUE;
	}
}



// mathematical functions:

void discreta_base::mult(discreta_base &x, discreta_base &y, int verbose_level)
{
	x.mult_to(y, *this, verbose_level);
}

void discreta_base::mult_mod(
		discreta_base &x,
		discreta_base &y, discreta_base &p, int verbose_level)
{
	discreta_base z;
	
	x.mult_to(y, z, verbose_level);
	z.modulo(p, verbose_level);
	swap(z);
}

void discreta_base::mult_to(discreta_base &x, discreta_base &y, int verbose_level)
{
	if (s_kind() != BASE) {
		cout << "mult_to() not implemented for class ";
		printobjectkindln(cout);
		exit(1);
		// mult_to(x, y);
		// return;
	}
	NOT_EXISTING_FUNCTION("discreta_base::mult_to");
	exit(1);
}

int discreta_base::invert(int verbose_level)
{
	discreta_base a;
	int ret;
	
	ret = invert_to(a, verbose_level);
	// cout << "discreta_base::invert() a="; a.println();
	// freeself();
	swap(a);
	return ret;
}

int discreta_base::invert_mod(discreta_base &p, int verbose_level)
{
	discreta_base u, v, g;
	
	// cout << "discreta_base::invert_mod this=" << *this << endl;
	extended_gcd(p, u, v, g, 0);
	// cout << "discreta_base::invert_mod "
	//"gcd = " << g << " = " << u << " * " << *this
	// << " + " << v << " * " << p << endl;
	if (!g.is_one()) {
		return FALSE;
	}
	swap(u);
	// normalize(p);
	return TRUE;
}

int discreta_base::invert_to(discreta_base &x, int verbose_level)
{
	if (s_kind() != BASE) {
		// cout << "invert_to() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		return invert_to(x, verbose_level);
	}
	NOT_EXISTING_FUNCTION("discreta_base::invert_to");
	exit(1);
}

void discreta_base::mult_apply(discreta_base &x, int verbose_level)
{
	discreta_base a;
	
	// cout << "discreta_base::mult_apply() calling mult_to()\n";
	mult_to(x, a, verbose_level);
	freeself();
	swap(a);
}

#if 1
discreta_base& discreta_base::power_int(int l, int verbose_level)
{
	discreta_base a, b;
	
	if (l < 0) {
		invert(verbose_level);
		l *= -1;
	}
	a = *this;
	a.one();
	b = *this;
	while (l) {
		if (EVEN(l)) {
			b *= b;
			l >>= 1;
		}
		if (ODD(l)) {
			a *= b;
			l--;
		}
	}
	*this = a;
	return *this;
}
#endif

#if 0
discreta_base& discreta_base::power_int(int l)
{
	discreta_base *a = callocobject(BASE);
	discreta_base *b = callocobject(BASE);
	
	*a = *this;
	a->one();
	*b = *this;
	while (l) {
		if (EVEN(l)) {
			*b *= *b;
			l >>= 1;
			}
		if (ODD(l)) {
			*a *= *b;
			l--;
			}
		}
	*this = *a;
	freeobject(a);
	freeobject(b);
	return *this;
}
#endif

discreta_base& discreta_base::power_int_mod(int l, discreta_base &p, int verbose_level)
{
	discreta_base a, b, c;
	
	a = *this;
	a.one();
	b = *this;
	// cout << "discreta_base:power_int_mod() x=" << *this << " l=" << l << " p=" << p << endl;
	while (l) {
		// cout << "= " << a << " * " << b << "^" << l << endl;
		if (EVEN(l)) {
			c.mult_mod(b, b, p, verbose_level);
			c.swap(b);
			l >>= 1;
		}
		// cout << "= " << a << " * " << b << "^" << l << endl;
		if (ODD(l)) {
			c.mult_mod(a, b, p, verbose_level);
			c.swap(a);
			l--;
		}
	}
	// cout << "= " << a << " * " << b << "^" << l << endl;
	*this = a;
	return *this;
}

discreta_base& discreta_base::power_longinteger(longinteger& l)
{
	discreta_base a, b, c;
	
	a = *this;
	a.one();
	b = *this;
	while (!l.is_zero()) {
		if (a.s_kind() == LONGINTEGER) {
			longinteger &B = b.as_longinteger();
			int d;
			
			d = B.s_len();
			cout << "l=" << l << " digits=" << d << endl;
		}
		if (l.is_even()) {
			b *= b;
			// c.mult(b, b);
			// b.swap(c);
			l.divide_out_int(2);
		}
		if (l.is_odd()) {
			a *= b;
			// c.mult(a, b);
			// a.swap(c);
			l.dec();
		}
	}
	*this = a;
	return *this;
}

discreta_base& discreta_base::power_longinteger_mod(
		longinteger& l, discreta_base &p, int verbose_level)
{
	discreta_base a, b, c;
	
	a = *this;
	a.one();
	b = *this;
	while (!l.is_zero()) {
		if (a.s_kind() == LONGINTEGER) {
			longinteger &B = a.as_longinteger();
			int d;
			
			d = B.s_len();
			cout << "l=" << l << " digits=" << d << endl;
		}
		if (l.is_even()) {
			c.mult_mod(b, b, p, verbose_level);
			c.swap(b);
			l.divide_out_int(2);
		}
		if (l.is_odd()) {
			c.mult_mod(a, b, p, verbose_level);
			c.swap(a);
			l.dec();
		}
	}
	*this = a;
	return *this;
}

discreta_base& discreta_base::commutator(discreta_base &x, discreta_base &y, int verbose_level)
{
	discreta_base xv, yv, a;
	
	x.invert_to(xv, verbose_level);
	y.invert_to(yv, verbose_level);
	a.mult(xv, yv, verbose_level);
	a *= x;
	a *= y;
	swap(a);
	xv.freeself();
	yv.freeself();
	return *this;
}

discreta_base& discreta_base::conjugate(discreta_base &x, discreta_base &y, int verbose_level)
{
	discreta_base yv, a;
	
	// cout << "discreta_base::conjugate: y.invert_to(yv)\n";
	y.invert_to(yv, verbose_level);
	// cout << "yv= " << yv << endl;
	// cout << "x= " << x << endl;
	// cout << "discreta_base::conjugate: a.mult(yv, x)\n";
	a.mult(yv, x, verbose_level);
	// cout << "a=" << a << endl;
	// cout << "discreta_base::conjugate: a *= y\n";
	a *= y;
	swap(a);
	return *this;
}

discreta_base& discreta_base::divide_by(discreta_base& x, int verbose_level)
{
	discreta_base q, r;
	integral_division(x, q, r, verbose_level);
	swap(q);
	return *this;
}

discreta_base& discreta_base::divide_by_exact(discreta_base& x, int verbose_level)
{
	discreta_base q;
	integral_division_exact(x, q, verbose_level);
	swap(q);
	return *this;
}

#undef DEBUG_ORDER

int discreta_base::order()
{
	discreta_base a, b;
	int i = 1;
	
	copyobject_to(a);
	copyobject_to(b);
	while (!b.is_one()) {
#ifdef DEBUG_ORDER
		cout << "discreta_base::order b^" << i << "=" << b << endl;
#endif
		b *= a;
		i++;
	}
#ifdef DEBUG_ORDER
	cout << "discreta_base::order b^" << i << "=" << b << " is one " << endl;
#endif
	return i;
}

int discreta_base::order_mod(discreta_base &p, int verbose_level)
{
	discreta_base a, b, c;
	int i = 1;
	
	copyobject_to(a);
	copyobject_to(b);
	while (!b.is_one()) {
		c.mult_mod(a, b, p, verbose_level);
		b.swap(c);
		i++;
	}
	return i;
}

void discreta_base::add(discreta_base &x, discreta_base &y)
{
	// cout << "discreta_base::add() x=" << x << ", y=" << y << endl;
	x.add_to(y, *this);
}

void discreta_base::add_mod(
		discreta_base &x, discreta_base &y, discreta_base &p, int verbose_level)
{
	discreta_base z;
	
	x.add_to(y, z);
	z.modulo(p, verbose_level);
	swap(z);
}

void discreta_base::add_to(discreta_base &x, discreta_base &y)
{
	if (s_kind() != BASE) {
		// cout << "add_to() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		add_to(x, y);
		return;
	}
	NOT_EXISTING_FUNCTION("discreta_base::add_to");
	exit(1);
}

void discreta_base::negate()
{
	discreta_base a;
	
	negate_to(a);
	swap(a);
}

void discreta_base::negate_to(discreta_base &x)
{
	if (s_kind() != BASE) {
		cout << "negate_to() not implemented for class ";
		printobjectkindln(cout);
		exit(1);
		// negate_to(x);
		// return;
	}
	NOT_EXISTING_FUNCTION("discreta_base::negate_to");
	exit(1);
}

void discreta_base::add_apply(discreta_base &x)
{
	discreta_base a;
	
	add_to(x, a);
	swap(a);
}

void discreta_base::normalize(discreta_base &p)
{
	if (s_kind() != BASE) {
		cout << "normalize() not implemented for class ";
		printobjectkindln(cout);
		exit(1);
		// normalize(p);
		// return;
	}
	NOT_EXISTING_FUNCTION("discreta_base::normalize");
	exit(1);
}

void discreta_base::zero()
{
	if (s_kind() != BASE) {
		// cout << "zero() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		zero();
		return;
	}
	NOT_EXISTING_FUNCTION("discreta_base::zero");
	exit(1);
}

void discreta_base::one()
{
	if (s_kind() != BASE) {
		// cout << "one() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		one();
		return;
	}
	NOT_EXISTING_FUNCTION("discreta_base::one");
	exit(1);
}

void discreta_base::m_one()
{
	if (s_kind() != BASE) {
		// cout << "m_one() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		m_one();
		return;
	}
	NOT_EXISTING_FUNCTION("discreta_base::m_one");
	exit(1);
}

void discreta_base::homo_z(int z)
{
	if (s_kind() != BASE) {
		// cout << "homo_z() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		homo_z(z);
		return;
	}
	NOT_EXISTING_FUNCTION("discreta_base::homo_z");
	exit(1);
}

void discreta_base::inc()
{
	if (s_kind() != BASE) {
		// cout << "inc() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		inc();
		return ;
	}
	NOT_EXISTING_FUNCTION("discreta_base::inc");
	exit(1);
}

void discreta_base::dec()
{
	if (s_kind() != BASE) {
		// cout << "dec() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		dec();
		return;
	}
	NOT_EXISTING_FUNCTION("base::dec");
	exit(1);
}

int discreta_base::is_zero()
{
	if (s_kind() != BASE) {
		// cout << "is_zero() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		return is_zero();
	}
	NOT_EXISTING_FUNCTION("discreta_base::is_zero");
	exit(1);
}

int discreta_base::is_one()
{
	if (s_kind() != BASE) {
		// cout << "is_one() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		return is_one();
	}
	NOT_EXISTING_FUNCTION("discreta_base::is_one");
	exit(1);
}

int discreta_base::is_m_one()
{
	if (s_kind() != BASE) {
		// cout << "is_m_one() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		return is_m_one();
	}
	NOT_EXISTING_FUNCTION("discreta_base::is_m_one");
	exit(1);
}

discreta_base& discreta_base::factorial(int z)
{
	discreta_base a, b;
	
	a.m_i_i(1);
	while (z) {
		b.m_i_i(z);
		a *= b;
		z--;
	}
	*this = a;
	return *this;
}

discreta_base& discreta_base::i_power_j(int i, int j)
{
	m_i_i(i);
	power_int(j, 0);
	return *this;
}

int discreta_base::compare_with_euclidean(
		discreta_base &a)
{
	if (s_kind() != BASE) {
		// cout << "compare_with_euclidean() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		return compare_with_euclidean(a);
	}
	NOT_EXISTING_FUNCTION("discreta_base::compare_with_euclidean");
	exit(1);
}

void discreta_base::integral_division(
		discreta_base &x, discreta_base &q, discreta_base &r,
		int verbose_level)
{
	if (s_kind() != BASE) {
		// cout << "integral_division() not implemented for class ";
		// printobjectkindln(cout);
		// exit(1);
		integral_division(x, q, r, verbose_level);
		return;
	}
	NOT_EXISTING_FUNCTION("discreta_base::integral_division");
	exit(1);
}

void discreta_base::integral_division_exact(
		discreta_base &x, discreta_base &q, int verbose_level)
{
	discreta_base r;
	
	if (s_kind() != BASE) {
		integral_division(x, q, r, verbose_level);
		if (r.is_zero()) {
			return;
		}
		cout << "integral_division_exact "
				"remainder not zero" << endl;
		cout << "this=" << *this << " divided by "
				<< x << " gives remainder " << r << endl;
		exit(1);
	}
	NOT_EXISTING_FUNCTION("discreta_base::integral_division");
	exit(1);
}

void discreta_base::integral_division_by_integer(
		int x, discreta_base &q, discreta_base &r, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "discreta_base::integral_division_by_integer" << endl;
	}
	discreta_base a;
	
	a.m_i_i(x);
	integral_division(a, q, r, 0);
	if (f_v) {
		cout << "discreta_base::integral_division_by_integer done" << endl;
	}
}

void discreta_base::integral_division_by_integer_exact(
		int x, discreta_base &q, int verbose_level)
{
	discreta_base a;
	
	a.m_i_i(x);
	integral_division_exact(a, q, verbose_level);
}

void discreta_base::integral_division_by_integer_exact_apply(int x, int verbose_level)
{
	discreta_base a, q;
	
	a.m_i_i(x);
	integral_division_exact(a, q, verbose_level);
	swap(q);
}

int discreta_base::is_divisor(discreta_base& y, int verbose_level)
{
	discreta_base q, r;
	
	y.integral_division(*this, q, r, verbose_level);
	if (r.is_zero())
		return TRUE;
	else
		return FALSE;
}

void discreta_base::modulo(discreta_base &p, int verbose_level)
{
	discreta_base q, r;
	
	integral_division(p, q, r, verbose_level);
	swap(r);
}

void discreta_base::extended_gcd(
		discreta_base &n,
		discreta_base &u, discreta_base &v, discreta_base &g,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	discreta_base sign1, sign2;
	int c;
	
	if (f_v) {
		cout << "discreta_base::extended_gcd "
				"m=" << *this << " n=" << n << endl;
		}
	c = compare_with_euclidean(n);
	if (c < 0) {
		n.extended_gcd(*this, v, u, g, verbose_level);
		return;
		}
	if (f_v) {
		cout << "discreta_base::extended_gcd "
				"m=" << *this << "(" << kind_ascii(s_kind()) << ")"
			<< "n=" << n << "(" << kind_ascii(n.s_kind()) << ")" << endl;
		}
	u = *this;
	v = n;
	if (/* c == 0 ||*/ n.is_zero()) {
		u.one();
		v.zero();
		g = *this;
		return;
		}

	if (s_kind() == INTEGER) {
		int a;
		a = s_i_i();
		if (a < 0) {
			sign1.m_i_i(-1);
			m_i_i(-a);
			}
		else
			sign1.m_i_i(1);
		
		a = n.s_i_i();
		if (a < 0) {
			sign2.m_i_i(-1);
			n.m_i_i(-a);
			}
		else
			sign2.m_i_i(1);
		
		}


	discreta_base M, N, Q, R;
	discreta_base u1, u2, u3, v1, v2, v3;
	
	M = *this;
	N = n;
	u1 = *this; u1.one();
	u2 = *this; u2.zero();
	v1 = n; v1.zero();
	v2 = n; v2.one();
	while (TRUE) {
		if (f_v) {
			cout << "loop:" << endl;
			cout << "M=" << M << "(" << kind_ascii(M.s_kind()) << ") N=" 
				<< N << "(" << kind_ascii(N.s_kind()) << ")" << endl;
			cout << "before integral_division" << endl;
			}
		M.integral_division(N, Q, R, verbose_level);
		if (f_v) {
			cout << "after integral_division" << endl;
			cout << "Q=" << Q << " R=" << R << endl;
			}
		if (R.is_zero()) {
			break;
			}
		// u3 := u1 - Q * u2
		u3 = u2;
		u3 *= Q;
		u3.negate();
		u3 += u1;
		
		// v3 := v1 - Q * v2
		v3 = v2;
		v3 *= Q;
		v3.negate();
		v3 += v1;
		
		M = N;
		N = R;
		u1 = u2;
		u2 = u3;
		v1 = v2;
		v2 = v3;
		}
	u = u2;
	v = v2;
	g = N;
	if (s_kind() == INTEGER) {
		// cout << "sign1=" << sign1 << endl;
		// cout << "sign2=" << sign2 << endl;
		int a;
		
		a = s_i_i();
		a *= sign1.s_i_i();
		m_i_i(a);

		a = u.s_i_i();
		a *= sign1.s_i_i();
		u.m_i_i(a);
		
		a = n.s_i_i();
		a *= sign2.s_i_i();
		n.m_i_i(a);
		
		a = v.s_i_i();
		a *= sign2.s_i_i();
		v.m_i_i(a);
		
		// *this *= sign1;
		// u *= sign1;
		// n *= sign2;
		// v *= sign2;
		}
	if (f_v) {
		cout << "g=" << g << " =" << u << " * "
				<< *this << " + " << v << " * " << n << endl;
		}
}

void discreta_base::write_memory(memory &m, int debug_depth)
{
	enum kind k;
	int i;
	char c;
	
	k = s_kind();
	i = (int) k;
	c = (char) k;
	if (!FITS_INTO_ONE_BYTE(i)) {
		cout << "write_memory(): kind not 1 byte" << endl;
		exit(1);
		}
	m.write_char(c);
	if (debug_depth > 0) {
		cout << "discreta_base::write_memory "
				"object of kind = " << kind_ascii(k) << endl;
		}
	switch (k) {
		case BASE:
			break;
		case INTEGER:
			m.write_int(s_i_i());
			break;
		case VECTOR:
			as_vector().write_mem(m, debug_depth);
			break;
		case NUMBER_PARTITION:
			as_number_partition().write_mem(m, debug_depth);
			break;
		case PERMUTATION:
			as_permutation().write_mem(m, debug_depth);
			break;
		case MATRIX:
			as_matrix().write_mem(m, debug_depth);
			break;
		case LONGINTEGER:
			// as_longinteger().write_mem(m, debug_depth);
			cout << "discreta_base::write_mem "
					"no write_mem for LONGINTEGER" << endl;
			break;
		case MEMORY:
			as_memory().write_mem(m, debug_depth);
			break;
		case HOLLERITH:
			as_hollerith().write_mem(m, debug_depth);
			break;
		//case PERM_GROUP:
			//as_perm_group().write_mem(m, debug_depth);
			//break;
		//case PERM_GROUP_STAB_CHAIN:
			//as_perm_group_stab_chain().write_mem(m, debug_depth);
			//break;
		case UNIPOLY:
			as_unipoly().write_mem(m, debug_depth);
			break;
		//case SOLID:
			//as_solid().write_mem(m, debug_depth);
			//break;
		//case BITMATRIX:
			//as_bitmatrix().write_mem(m, debug_depth);
			//break;
		//case PC_PRESENTATION:
			//as_pc_presentation().write_mem(m, debug_depth);
			//break;
		//case PC_SUBGROUP:
			//as_pc_subgroup().write_mem(m, debug_depth);
			//break;
		//case GROUP_WORD:
			//as_group_word().write_mem(m, debug_depth);
			//break;
		//case GROUP_TABLE:
			//as_group_table().write_mem(m, debug_depth);
			//break;
#if 0
		case ACTION:
			as_action().write_mem(m, debug_depth);
			break;
		case GEOMETRY:
			as_geometry().write_mem(m, debug_depth);
			break;
		case GROUP_SELECTION:
			as_group_selection().write_mem(m, debug_depth);
			break;
#endif
		case DESIGN_PARAMETER:
			as_design_parameter().write_mem(m, debug_depth);
			break;
		case DESIGN_PARAMETER_SOURCE:
			as_design_parameter_source().write_mem(m, debug_depth);
			break;
		default:
			cout << "discreta_base::write_memory "
					"no write_mem for " << kind_ascii(k) << endl;
			exit(1);
		}
}

void discreta_base::read_memory(memory &m, int debug_depth)
{
	enum kind k;
	int i;
	char c;
	
	m.read_char(&c);
	k = (enum kind) c;
	c_kind(k);
	switch (k) {
		case BASE:
			break;
		case INTEGER:
			m.read_int(&i);
			m_i_i(i);
			break;
		case VECTOR:
			as_vector().read_mem(m, debug_depth);
			break;
		case NUMBER_PARTITION:
			as_number_partition().read_mem(m, debug_depth);
			break;
		case PERMUTATION:
			as_permutation().read_mem(m, debug_depth);
			break;
		case MATRIX:
			as_matrix().read_mem(m, debug_depth);
			break;
		case LONGINTEGER:
			// as_longinteger().read_mem(m, debug_depth);
			cout << "discreta_base::read_mem "
					"no read_mem for LONGINTEGER" << endl;
			break;
		case MEMORY:
			as_memory().read_mem(m, debug_depth);
			break;
		case HOLLERITH:
			as_hollerith().read_mem(m, debug_depth);
			break;
		//case PERM_GROUP:
			//as_perm_group().read_mem(m, debug_depth);
			//break;
		//case PERM_GROUP_STAB_CHAIN:
			//as_perm_group_stab_chain().read_mem(m, debug_depth);
			//break;
		case UNIPOLY:
			as_unipoly().read_mem(m, debug_depth);
			break;
		//case SOLID:
			//as_vector().read_mem(m, debug_depth);
			//break;
//		case BITMATRIX:
//			as_bitmatrix().read_mem(m, debug_depth);
//			break;
		//case PC_PRESENTATION:
			//as_pc_presentation().read_mem(m, debug_depth);
			//break;
		//case PC_SUBGROUP:
			//as_pc_subgroup().read_mem(m, debug_depth);
			//break;
		//case GROUP_WORD:
			//as_group_word().read_mem(m, debug_depth);
			//break;
		//case GROUP_TABLE:
			//as_group_table().read_mem(m, debug_depth);
			//break;
#if 0
		case ACTION:
			as_action().read_mem(m, debug_depth);
			break;
		case GEOMETRY:
			as_geometry().read_mem(m, debug_depth);
			break;
		case GROUP_SELECTION:
			as_group_selection().read_mem(m, debug_depth);
			break;
#endif
		case DESIGN_PARAMETER:
			as_design_parameter().read_mem(m, debug_depth);
			break;
		case DESIGN_PARAMETER_SOURCE:
			as_design_parameter_source().read_mem(m, debug_depth);
			break;
		default:
			cout << "discreta_base::read_memory "
					"no read_mem for " << kind_ascii(k) << endl;
			exit(1);
		}
}

int discreta_base::calc_size_on_file()
{
	enum kind k;
	int i, size;
	//char c;
	
	k = s_kind();
	i = (int) k;
	//c = (char) k;
	if (!FITS_INTO_ONE_BYTE(i)) {
		cout << "write_memory "
				"kind not 1 byte" << endl;
		exit(1);
		}
	size = 1;
	switch (k) {
		case BASE:
			break;
		case INTEGER:
			size += 4;
			break;
		case VECTOR:
			size += as_vector().csf();
			break;
		case NUMBER_PARTITION:
			size += as_number_partition().csf();
			break;
		case PERMUTATION:
			size += as_permutation().csf();
			break;
		case MATRIX:
			size += as_matrix().csf();
			break;
		case LONGINTEGER:
			// size += as_longinteger().csf();
			cout << "discreta_base::write_mem "
					"no csf for LONGINTEGER" << endl;
			break;
		case MEMORY:
			size += as_memory().csf();
			break;
		case HOLLERITH:
			size += as_hollerith().csf();
			break;
		//case PERM_GROUP:
			//size += as_perm_group().csf();
			//break;
		//case PERM_GROUP_STAB_CHAIN:
			//size += as_perm_group_stab_chain().csf();
			//break;
		case UNIPOLY:
			size += as_unipoly().csf();
			break;
		//case SOLID:
			//size += as_vector().csf();
			//break;
//		case BITMATRIX:
//			size += as_bitmatrix().csf();
//			break;
		//case PC_PRESENTATION:
			//size += as_pc_presentation().csf();
			//break;
		//case PC_SUBGROUP:
			//size += as_pc_subgroup().csf();
			//break;
		//case GROUP_WORD:
			//size += as_group_word().csf();
			//break;
		//case GROUP_TABLE:
			//size += as_group_table().csf();
			//break;
#if 0
		case ACTION:
			size += as_action().csf();
			break;
		case GEOMETRY:
			size += as_geometry().csf();
			break;
		case GROUP_SELECTION:
			size += as_group_selection().csf();
			break;
#endif
		case DESIGN_PARAMETER:
			size += as_design_parameter().csf();
			break;
		case DESIGN_PARAMETER_SOURCE:
			size += as_design_parameter_source().csf();
			break;
		default:
			cout << "discreta_base::calc_size_on_file "
					"no csf for " << kind_ascii(k) << endl;
			exit(1);
		}
	return size;
}

void discreta_base::pack(memory & M, int verbose_level, int debug_depth)
// used to pack (i.e. to serialize) objects
// into (binary) strings in memory objects.
{
	int f_v = (verbose_level >= 1);
	int size, size0;
	
	if (f_v) {
		cout << "discreta_base::pack "
				"calculating memory size" << endl;
		}
	size0 = calc_size_on_file();
	// M.init(0, NULL);
	if (f_v) {
		cout << "discreta_base::pack "
				"allocating memory of size " << size0 << endl;
		}
	M.alloc(size0);
	M.used_length() = 0;
	if (f_v) {
		cout << "discreta_base::pack calling write_memory" << endl;
		}
	write_memory(M, debug_depth);
	size = M.used_length();
	if (size != size0) {
		cout << "discreta_base::pack "
				"WARNING!!!: size = " << size
				<< " != size0 = " << size0 << endl;
		}
}

void discreta_base::unpack(memory & M, int verbose_level, int debug_depth)
// unpacks an object from a binary representation in a memory object
{
	//int f_v = (verbose_level >= 1);
	read_memory(M, debug_depth);
}

void discreta_base::save_ascii(ostream & f)
// writes in ASCII text format (uuencoded like) into the stream f. 
{
	memory M;
	int f_v = FALSE, f_vv = FALSE;
	int size, debug_depth;
	int i;
	unsigned int a, a1, a2;
	uchar *pc, c1, c2;

	if (f_v) {
		cout << "discreta_base::save_ascii "
				"calculating memory size" << endl;
		}
	if (f_vv)
		debug_depth = 1;
	else
		debug_depth = 0;
	if (f_v) {
		cout << "discreta_base::save_ascii "
				"packing object" << endl;
		}
	pack(M, f_v, debug_depth);
#ifdef SAVE_ASCII_USE_COMPRESS
	if (f_v) {
		cout << "discreta_base::save_ascii "
				"compressing object" << endl;
		}
	M.compress(f_v);
#endif
	if (f_v) {
		cout << "discreta_base::save_ascii "
				"saving data" << endl;
		}
	size = M.used_length();
	pc = (uchar *) M.self.char_pointer;
	
	f << "ASCII " << size << endl;
	for (i = 0; i < size; i++) {
		a = (unsigned int) pc[i];
		a1 = a % (unsigned int) 16;
		a2 = a >> 4;
		c1 = '0' + a1;
		c2 = '0' + a2;
		f << c1 << c2;
		if ((i + 1) % 40 == 0)
			f << endl;
		}
	f << endl << "ASCIIEND" << endl;
}

//#define BUFSIZE 10000

void discreta_base::load_ascii(istream & f)
// reads ASCII style objects written with save-ascii
{
	memory M;
	char buf[BUFSIZE];
	char str[1024], *p;
	int f_v = TRUE;
	int f_vv = FALSE;
	int size, i, debug_depth;
	uchar *pc;
	uchar c;
	int a;
	unsigned int a1, a2;
	char cc;
	layer1_foundations::data_structures::string_tools ST;
		
	f.getline(buf, sizeof(buf));
	p = buf;
	ST.s_scan_token(&p, str);
	if (strcmp(str, "ASCII") != 0) {
		cout << "discreta_base::load_ascii "
				"error reading header: ASCII keyword not found" << endl;
		exit(1);
		}
	ST.s_scan_int(&p, &size);
	if (f_v) {
		cout << "discreta_base::load_ascii "
				"reading ASCII file of size " << size << endl;
		}
	M.alloc(size);
	pc = (uchar *) M.self.char_pointer;
	for (i = 0; i < size; i++) {
		while (TRUE) {
			if (f.eof()) {
				cout << "discreta_base::load_ascii "
						"primature EOF" << endl;
				exit(1);
				}
			f >> cc;
			if (cc == '\n')
				continue;
			break;
			}
		a1 = (unsigned int) cc;
		if (f.eof()) {
			cout << "discreta_base::load_ascii "
					"primature EOF" << endl;
			exit(1);
			}
		f >> cc;
		a2 = (unsigned int) cc;
		a1 = a1 - '0';
		a2 = a2 - '0';
		a = a2 << 4;
		a += a1;
		c = (uchar) a;
		pc[i] = c; 
		}
#if 1
	while (TRUE) {
		f.getline(buf, sizeof(buf));
		if (strlen(buf))
			break;
		// if (buf[0] != '\n') break;
		}
#endif
	// f.getline(buf, sizeof(buf));
	// cout << "discreta_base::load_ascii(): buf = " << buf << endl;
	p = buf;
	ST.s_scan_token(&p, str);
	if (strcmp(str, "ASCIIEND") != 0) {
		cout << "discreta_base::load_ascii "
				"error reading footer: ASCIIEND keyword not found" << endl;
		exit(1);
		}

			
	
	if (f_v) {
		cout << "file read." << endl;
		}
	M.used_length() = size;
#ifdef SAVE_ASCII_USE_COMPRESS
	M.decompress(TRUE /* f_verbose */);
#endif
	M.cur_pointer() = 0;
	if (f_vv)
		debug_depth = 1;
	else
		debug_depth = 0;
	unpack(M, f_v, debug_depth);
}

void discreta_base::save_file(const char *fname)
// writes in ASCII text format (uuencoded like) into the file.
{
	ofstream f(fname);
	save_ascii(f);
}
 
void discreta_base::load_file(const char *fname)
// read in ASCII text format (uuencoded like) from the file.
{
	ifstream f(fname);
	load_ascii(f);
}
 

}}}

