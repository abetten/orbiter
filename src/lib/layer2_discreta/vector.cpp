// vector.cpp
//
// Anton Betten
// 18.12.1998
// moved from D2 to ORBI Nov 15, 2007

#include "../layer2_discreta/discreta.h"
#include "layer1_foundations/foundations.h"

#undef VECTOR_COPY_VERBOSE
#undef VECTOR_CHANGE_KIND_VERBOSE

using namespace std;


namespace orbiter {
namespace layer2_discreta {


Vector::Vector()
{
	k = VECTOR;
	self.vector_pointer = NULL;
}

Vector::Vector(const discreta_base &x)
	// copy constructor:    this := x
{
	// cout << "Vector::copy constructor for object: " <<  << "\n";
	clearself();
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

Vector& Vector::operator = (const discreta_base &x)
	// copy assignment
{
	// cout << "Vector::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void Vector::settype_vector()
{
	OBJECTSELF s;
	
	s = self;
	new(this) Vector;
	self = s;
	k = VECTOR;
}

Vector::~Vector()
{
	// cout << "Vector::~Vector()\n";
	freeself_vector();
}

void Vector::freeself_vector()
{
	if (self.vector_pointer == NULL)
		return;
	// cout << "Vector::freeself_vector():"; cout << *this << endl;
	free_nobjects_plus_length(self.vector_pointer);
	self.vector_pointer = NULL;
}

kind Vector::s_virtual_kind()
{
	return VECTOR;
}

void Vector::copyobject_to(discreta_base &x)
{
	int i, l;
	
#ifdef VECTOR_COPY_VERBOSE
	cout << "in Vector::copyobject_to()\n";
#endif
	x.freeself();
	if (x.s_kind() != VECTOR) {
#ifdef VECTOR_CHANGE_KIND_VERBOSE
		cout << "warning: Vector::copyobject_to x not a vector\n";
#endif
		x.c_kind(VECTOR);
		x.clearself();
		// x.printobjectkindln();
		}
#ifdef VECTOR_COPY_VERBOSE
	cout << "source=" << *this << endl;
	cout << "target=" << x << endl;
#endif
	l = s_l();
#ifdef VECTOR_COPY_VERBOSE
	cout << "l=" << l << endl;
#endif
	Vector & xx = x.as_vector();
	xx.m_l(l);
#ifdef VECTOR_COPY_VERBOSE
	cout << "after xx.m_l(l)\n";
#endif
	for (i = 0; i < l; i++) {
#ifdef VECTOR_COPY_VERBOSE
		cout << "in Vector::copyobject_to() copy element " 
			<< i << "=" << s_i(i) << "\n";
#endif
		xx[i] = s_i(i);
		}
}

#undef PRINT_WITH_TYPE

ostream& Vector::Print(ostream& ost)
{
	int i, l;
	
	if (self.vector_pointer == NULL) {
		ost << "vector not allocated";
		}
	l = s_l();
#ifdef PRINT_WITH_TYPE
	ost << "(VECTOR of length " << l << ", \n";
#endif
	for (i = 0; i < l; i++) {
		s_i(i).print(ost);
		if (i < l - 1)
			ost << ", \n";
		}
#ifdef PRINT_WITH_TYPE
	ost << ")";
#endif
	ost << "\n";
	return ost;
}

ostream& Vector::print(ostream& ost)
{
	int i, l;
	
	// cout << "Vector::print()" << endl;
	if (self.vector_pointer == NULL) {
		ost << "vector not allocated";
		}
	l = s_l();
	if (current_printing_mode() == printing_mode_gap) {
		ost << "[";
		for (i = 0; i < l; i++) {
			s_i(i).print(ost);
			if (i < l - 1)
				ost << ", ";
			}
		ost << "]";
		}
	else {
#ifdef PRINT_WITH_TYPE
		ost << "(VECTOR of length " << l << ", ";
#else
		ost << "(";
#endif
		for (i = 0; i < l; i++) {
			s_i(i).print(ost);
			if (i < l - 1)
				ost << ", ";
			}
		ost << ")";
		}
	return ost;
}

ostream& Vector::print_unformatted(ostream& ost)
{
	int i, l;
	
	if (self.vector_pointer == NULL) {
		ost << "vector not allocated";
		}
	l = s_l();
	for (i = 0; i < l; i++) {
		s_i(i).print(ost);
		ost << " ";
		}
	return ost;
}

ostream& Vector::print_intvec(ostream& ost)
{
	int i, l;
	
	if (self.vector_pointer == NULL) {
		ost << "vector not allocated";
		}
	l = s_l();
	ost << "(";
	for (i = 0; i < l; i++) {
		ost << s_ii(i);
		if (i < l - 1)
			ost << " ";
		}
	ost << ")";
	return ost;
}

discreta_base & Vector::s_i(int i)
{
	int l;
	
	if (self.vector_pointer == NULL) {
		cout << "Vector::s_i() vector_pointer == NULL\n";
		exit(1);
		}
	l = self.vector_pointer[-1].s_i_i();
	if ( i < 0 || i >= l ) {
		cout << "Vector::s_i() addressing error, i = " << i << ", length = " << l << "\n";
		exit(1);		
		}
	return self.vector_pointer[i];
}

int Vector::s_l()
{
	if (self.vector_pointer == NULL)
		return 0;
	// cout << "Vector::s_l()" << endl;
	return self.vector_pointer[-1].s_i_i();
}

void Vector::m_l(int l)
{
	// cout << "vector::m_l() l=" << l << "\n";
	// printobjectkind(cout);
	// cout << *this << "\n";
	// cout << "calling freeself\n";
	freeself();
	// cout << "Vector::m_l(), calling calloc_nobjects_plus_length\n";
	self.vector_pointer = calloc_nobjects_plus_length(l, BASE);
}

void Vector::m_l_n(int l)
{
	int i;
	
	m_l(l);
	for (i = 0; i < l; i++) {
		s_i(i).m_i_i(0);
		}
}

void Vector::m_l_e(int l)
{
	int i;
	
	m_l(l);
	for (i = 0; i < l; i++) {
		s_i(i).m_i_i(1);
		}
}

void Vector::m_l_x(int l, discreta_base &x)
{
	int i;
	
	m_l(l);
	for (i = 0; i < l; i++) {
		s_i(i) = x;
		}
}

Vector& Vector::realloc(int l)
{
	Vector v;
	int i, ll;
	
	ll = s_l();
	v.m_l(l);
	for (i = 0; i < MINIMUM(l, ll); i++) {
		v.s_i(i).swap(s_i(i));
		}
	swap(v);
	return *this;
}

void Vector::mult_to(discreta_base &x, discreta_base &y)
{
	if (x.s_kind() == MATRIX) {
		y.change_to_vector();
		x.as_matrix().multiply_vector_from_left(*this, y.as_vector());
		}
	else if (x.s_kind() == VECTOR) {
		cout << "Vector::mult_to() error: cannot multiply vector with vector\n";
		exit(1);
		// Vector& px = x.as_vector();
		// vector_mult_to(px, y);
		}
	else {
		cout << "vector::mult_to() object x is of bad type\n";
		exit(1);
		}
}

void Vector::add_to(discreta_base &x, discreta_base &y)
{
	int i, l;
	
	y.freeself();
	if (s_kind() != VECTOR) {
		cout << "Vector::add_to() this is not a vector\n";
		exit(1);
		}
	if (x.s_kind() != VECTOR) {
		cout << "matrix::add_to() x is not a vector\n";
		exit(1);
		}
	Vector& px = x.as_vector();
	Vector py;
	
	l = s_l();
	if (l != px.s_l()) {
		cout << "vector::add_to() l != px.s_l()\n";
		exit(1);
		}
	py.m_l(l);
	for (i = 0; i < l; i++) {
		py[i].add(s_i(i), px[i]);
		}
	py.swap(y);
}

void Vector::inc()
{
	realloc(s_l() + 1);
}

void Vector::dec()
{
	int l = s_l();
	
	if (l == 0) {
		cout << "Vector::dec() length is zero\n";
		exit(1);
		}
	realloc(l - 1);
}

int Vector::compare_with(discreta_base &a)
{
	int l1, l2, i, c;
	
	if (s_kind() != VECTOR) {
		return compare_with(a);
		}
	if (a.s_kind() != VECTOR) {
		cout << "a is not a vector\n";
		exit(1);
		}
	Vector& v = a.as_vector();
	l1 = s_l();
	l2 = v.s_l();
	for (i = 0; i < l1; i++) {
		if (i < l2) {
			c = s_i(i).compare_with(v[i]);
			if (c != 0)
				return c;
			}
		else {
			return -1;
			}
		}
	if (l2 > l1)
		return 1;
	return 0;
}

void Vector::append_vector(Vector &v)
{
	Vector w;
	int i, l1, l2, l3;
	
	l1 = s_l();
	l2 = v.s_l();
	l3 = l1 + l2;
	w.m_l(l3);
	for (i = 0; i < l1; i++) {
		w[i].swap(s_i(i));
		}
	for (i = 0; i < l2; i++) {
		w[l1 + i].swap(v[i]);
		}
	swap(w);
}

Vector& Vector::append_integer(int a)
{
	int l;
	
	l = s_l();
	inc();
	m_ii(l, a);
	return *this;
}

Vector& Vector::append(discreta_base& a)
{
	int l;
	
	l = s_l();
	inc();
	s_i(l) = a;
	return *this;
}

Vector& Vector::insert_element(int i, discreta_base& x)
{
	int j, l;
	
	l = s_l();
	// cout << "Vector::insert_element(" << i << ", " << x << "), l=" << l << "\n";
	inc();
	for (j = l; j > i; j--) {
		s_i(j).swap(s_i(j - 1));
		}
	// cout << "before s_i(i) = x;\n";
	// cout << "s_i(i)=" << s_i(i) << endl;
	// cout << "x=" << x << endl;
	s_i(i) = x;
	return *this;
}

Vector& Vector::get_and_delete_element(int i, discreta_base& x)
{
	int l;
	
	l = s_l();
	if (i >= l) {
		cout << "Vector::get_and_delete_element() i >= l" << endl;
		exit(1);
		}
	x.swap(s_i(i));
	return delete_element(i);
}

Vector& Vector::delete_element(int i)
{
	int l, j;
	l = s_l();
	for (j = i + 1; j < l; j++) {
		s_i(j - 1).swap(s_i(j));
		}
	dec();
	return *this;
}

void Vector::get_first_and_remove(discreta_base & x)
{
	get_and_delete_element(0, x);
}

bool Vector::insert_sorted(discreta_base& x)
	// inserts x into the sorted Vector x.
	// ifthere are already occurences of x, the new x is added 
	// behind the x already there.
	// returns true if the element was already in the Vector.
{
	int idx;
	
	if (search(x, &idx)) {
		// cout << "insert_sorted() found element at " << idx << endl;
		idx++;
		insert_element(idx, x);
		return true;
		}
	else {
		// cout << "insert_sorted() element not found, inserting at " << idx << endl;
		insert_element(idx, x);
		return false;
		}
}

bool Vector::search(discreta_base& x, int *idx)
	// returns TRUE if the object x has been found. 
	// idx contains the position where the object which 
	// has been found lies. 
	// if there are more than one element equal to x in the Vector, 
	// the last one will be found. 
	// if the element has not been found, idx contains the position of 
	// the next larger element. 
	// This is the position to insert x if required.
{
	int l, r, m, res, len;
	bool f_found = false;
	
	len = s_l();
	if (len == 0) {
		*idx = 0;
		return false;
		}
	l = 0;
	r = len;
	// invariant:
	// p[i] <= v for i < l;
	// p[i] >  v for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle
		res = s_i(m).compare_with(x);
		// cout << "search l=" << l << " m=" << m << " r=" 
		// 	<< r << "res=" << res << endl;
		if (res <= 0) {
			l = m + 1;
			if (res == 0)
				f_found = true;
			}
		else
			r = m;
		}
	// now: l == r; 
	// and f_found is set accordingly */
	if (f_found)
		l--;
	*idx = l;
	return f_found;
}

static void quicksort(Vector& v, int left, int right);
static void quicksort_with_logging(Vector& v, permutation& p, int left, int right);
static void partition(Vector& v, int left, int right, int *middle);
static void partition_with_logging(Vector& v, permutation& p, int left, int right, int *middle);

Vector& Vector::sort()
{
	int l;
	
	l = s_l();
	quicksort(*this, 0, l - 1);
	return *this;
}

void Vector::sort_with_fellow(Vector &fellow)
{
	permutation p, pv;
	
	sort_with_logging(p);
	pv = p;
	pv.invert();
	fellow.apply_permutation(pv);
}

Vector& Vector::sort_with_logging(permutation& p)
	// the permutation p tells where the sorted elements 
	// lay before, i.e. p[i] is the position of the
	// sorted element i in the unsorted Vector.
{
	int l;
	
	l = s_l();
	p.m_l(l);
	p.one();
	quicksort_with_logging(*this, p, 0, l - 1);
	return *this;
}


static void quicksort(Vector& v, int left, int right)
{
	int middle;
	
	if (left < right) {
		partition(v, left, right, &middle);
		quicksort(v, left, middle - 1);
		quicksort(v, middle + 1, right);
		}
}

static void quicksort_with_logging(Vector& v, permutation& p, int left, int right)
{
	int middle;
	
	if (left < right) {
		partition_with_logging(v, p, left, right, &middle);
		quicksort_with_logging(v, p, left, middle - 1);
		quicksort_with_logging(v, p, middle + 1, right);
		}
}

static void partition(Vector& v, int left, int right, int *middle)
{
	int l, r, m, len, m1, res, pivot;
	
	// pivot strategy: take the element in the middle: 
	len = right + 1 - left;
	m1 = len >> 1;
	pivot = left;
	if (m1)
		v[pivot].swap(v[left + m1]);
	l = left;
	r = right;
	while (l < r) {
		while (TRUE) {
			if (l > right)
				break;
			res = v[l].compare_with(v[pivot]);
			if (res > 0)
				break;
			l++;
			}
		while (TRUE) {
			if (r < left)
				break;
			res = v[r].compare_with(v[pivot]);
			if (res <= 0)
				break;
			r--;
			}
		// now v[l] > v[pivot] and v[r] <= v[pivot] 
		if (l < r)
			v[l].swap(v[r]);
		}
	m = r;
	if (left != m)
		v[left].swap(v[m]);
	*middle = m;
}

static void partition_with_logging(Vector& v, permutation& p, int left, int right, int *middle)
{
	int l, r, m, len, m1, res, pivot;
	data_structures::algorithms Algo;
	
	// pivot strategy: take the element in the middle: 
	len = right + 1 - left;
	m1 = len >> 1;
	pivot = left;
	if (m1) {
		v[pivot].swap(v[left + m1]);
		Algo.lint_swap(p[pivot], p[left + m1]);
	}
	l = left;
	r = right;
	while (l < r) {
		while (TRUE) {
			if (l > right) {
				break;
			}
			res = v[l].compare_with(v[pivot]);
			if (res > 0) {
				break;
			}
			l++;
		}
		while (TRUE) {
			if (r < left) {
				break;
			}
			res = v[r].compare_with(v[pivot]);
			if (res <= 0) {
				break;
			}
			r--;
		}
		// now v[l] > v[pivot] and v[r] <= v[pivot] 
		if (l < r) {
			v[l].swap(v[r]);
			Algo.lint_swap(p[l], p[r]);
		}
	}
	m = r;
	if (left != m) {
		v[left].swap(v[m]);
		Algo.lint_swap(p[left], p[m]);
	}
	*middle = m;
}


void Vector::sum_of_all_entries(discreta_base &x)
{
	int l = s_l();
	int i;
	
	x = s_i(0);
	for (i = 1; i < l; i++) {
		x += s_i(i);
	}
}




void Vector::n_choose_k_first(int n, int k)
{
	int i;
	
	m_l_n(k);
	for (i = 0; i < k; i++) {
		m_ii(i, i);
	}
}

int Vector::n_choose_k_next(int n, int k)
{
	int i, ii, a;
	
	if (k != s_l()) {
		cout << "Vector::n_choose_k_next() k != s_l()";
		exit(1);
	}
	for (i = 0; i < k; i++) {
		a = s_ii(k - 1 - i);
		if (a < n - 1 - i) {
			m_ii(k - 1 - i, a + 1);
			for (ii = i - 1; ii >= 0; ii--) {
				m_ii(k - 1 - ii, s_ii(k - 1 - ii - 1) + 1);
			}
			return TRUE;
		}
	}
	return FALSE;
}

int Vector::next_lehmercode()
{
	int l = s_l();
	int i, j;
	
	for (i = l - 1, j = 0; i >= 0; i--, j++) {
		if (s_ii(i) < j) {
			s_i(i).inc();
			return TRUE;
		}
		else {
			m_ii(i, 0);
		}
	}
	return FALSE;
}

void Vector::lehmercode2perm(permutation& p)
//Computes the permutation $p$ defined by its lehmercode (this).
{
	int i, k, l;
	Vector list;
	
	l = s_l();
	p.m_l(l);
	list.m_l(l);

	// list := (0,1,2,...,l-1):
	for (i = 0; i < l; i++) {
		list.m_ii(i, i);
	}
	
	for (i = 0; i < l; i++) {
		k = s_ii(i);
		p[i] = list.s_ii(k);
		list.delete_element(k);
	}
}

void Vector::q_adic(int n, int q)
{
	int r, i = 0;
	
	m_l(0);
	do {
		inc();
		r = n % q;
		m_ii(i, r);
		n /= q;
		i++;
	} while(n);
}

int Vector::q_adic_as_int(int q)
{
	int r, n = 0, i, l;
	
	l = s_l();
	n = 0;
	for (i = l - 1; i >= 0; i--) {
		n *= q;
		r = s_ii(i);
		n += r;
	}
	return n;
}

void Vector::mult_scalar(discreta_base& a)
{
	int i, l;
	
	l = s_l();
	for (i = 0; i < l; i++) {
		s_i(i) *= a;
	}
}

void Vector::first_word(int n, int q)
{
	m_l_n(n);
}

int Vector::next_word(int q)
{
	int n, i;
	
	n = s_l();
	i = n - 1;
	while (s_ii(i) == q - 1) {
		m_ii(i, 0);
		i--;
		if (i < 0) {
			return FALSE;
		}
	}
	s_i(i).inc();
	return TRUE;
}

void Vector::first_regular_word(int n, int q)
{
	m_l_n(n);
}

int Vector::next_regular_word(int q)
{
	do {
		if (!next_word(q)) {
			return FALSE;
		}
	} while (!is_regular_word());
	return TRUE;
}

int Vector::is_regular_word()
// works correct only for Vectors over the integers
{
	int n, i, k, ipk, f_rg;
	
	n = s_l();
	if (n == 1) {
		return TRUE;
	}
	k = 1;
	do {
		i = 0;
		ipk = i + k;
		while (s_ii(ipk) == s_ii(i) && i < n - 1) {
			i++;
			if (ipk == n - 1) {
				ipk = 0;
			}
			else {
				ipk++;
			}
		}
		f_rg = (s_ii(ipk) < s_ii(i));
		k++;
	} while (f_rg && k <= n - 1);
	return f_rg;
}

void Vector::apply_permutation(permutation &p)
{
	int i, j, l;
	Vector v;
	
	l = s_l();
	v.m_l(l);
	for (i = 0; i < l; i++) {
		j = p.s_i(i);
		v[j].swap(s_i(i));
	}
	swap(v);
}

void Vector::apply_permutation_to_elements(permutation &p)
{
	int i, l, a, b;
	
	l = s_l();
	for (i = 0; i < l; i++) {
		a = s_ii(i);
		b = p.s_i(a);
		m_ii(i, b);
	}
}

void Vector::content(Vector & c, Vector & where)
{
	int i, l, idx;
	discreta_base x;
	Vector v;
	
	v.m_l(0);
	where.m_l(0);
	c.m_l(0);
	l = s_l();
	for (i = 0; i < l; i++) {
		x = s_i(i);
		if (c.search(x, &idx)) {
		}
		else {
			c.insert_element(idx, x);
			where.insert_element(idx, v);
		}
		where[idx].as_vector().append_integer(i);
	}
}

void Vector::content_multiplicities_only(Vector & c, Vector & mult)
{
	int i, l, idx;
	discreta_base x;
	integer int_ob;
	
	int_ob.m_i(0);
	mult.m_l(0);
	c.m_l(0);
	l = s_l();
	for (i = 0; i < l; i++) {
		x = s_i(i);
		if (c.search(x, &idx)) {
		}
		else {
			c.insert_element(idx, x);
			mult.insert_element(idx, int_ob);
		}
		mult[idx].inc();
	}
}

int Vector::hip()
// homogeneous integer Vector predicate
{
	int i, l;
	
	l = s_l();
	for (i = 0; i < l; i++) {
		if (s_i(i).s_kind() != INTEGER) {
			return FALSE;
		}
	}
	return TRUE;
}

int Vector::hip1()
// homogeneous integer Vector predicate, 
// test for 1 char numbers; 
// only to apply if hip TRUE. */
{
	int i, l, k;
	
	l = s_l();
	for (i = 0; i < l; i++) {
		if (s_i(i).s_kind() != INTEGER) {
			cout << "Vector::hip1(): object not of type INTEGER\n";
			exit(1);
		}
		k = s_ii(i);
		if (!ONE_char_int(k)) {
			return FALSE;
		}
	}
	return TRUE;
}

void Vector::write_mem(memory & m, int debug_depth)
{
	int i, l, k;
	char f_hip = 0, f_hip1 = 0;
	
	l = s_l();
	m.write_int(l);
	f_hip = (char) hip();
	if (f_hip) {
		f_hip1 = (char) hip1();
	}
	if (debug_depth > 0) {
		cout << "writing ";
		if (f_hip) {
			if (f_hip1)
				cout << "hip1 ";
			else
				cout << "hip ";
			}
		cout << "Vector of length " << l << endl;
		}
	m.write_char(f_hip);
	if (f_hip) {
		m.write_char(f_hip1);
		if (f_hip1) {
			for (i = 0; i < l; i++) {
				k = s_ii(i);
				m.write_char((char) k);
				}
			}
		else {
			for (i = 0; i < l; i++) {
				m.write_int(s_ii(i));
				}
			}
		}
	else {
		for (i = 0; i < l; i++) {
			if (debug_depth > 0) {
				cout << l << " ";
				if ((l % 20) == 0)
					cout << endl;
				}
			s_i(i).write_memory(m, debug_depth - 1);
			}
		}
}

void Vector::read_mem(memory & m, int debug_depth)
{
	int i, l, k;
	char c, f_hip = 0, f_hip1 = 0;
	
	m.read_int(&l);
	m_l(l);
	m.read_char(&f_hip);
	if (f_hip) {
		m.read_char(&f_hip1);
		}
	if (debug_depth > 0) {
		cout << "reading ";
		if (f_hip) {
			if (f_hip1)
				cout << "hip1 ";
			else
				cout << "hip ";
			}
		cout << "Vector of length " << l << endl;
		}
	if (f_hip) {
		if (f_hip1) {
			for (i = 0; i < l; i++) {
				m.read_char(&c);
				k = (int) c;
				m_ii(i, k);
				}
			}
		else {
			for (i = 0; i < l; i++) {
				m.read_int(&k);
				m_ii(i, k);
				}
			}
		}
	else {
		for (i = 0; i < l; i++) {
			if (debug_depth > 0) {
				cout << l << " ";
				if ((l % 20) == 0)
					cout << endl;
				}
			s_i(i).read_memory(m, debug_depth - 1);
			}
		}
}

int Vector::csf()
{
	int i, l;
	char f_hip, f_hip1;
	int size = 0;
	
	l = s_l();
	size += 4; /* l */
	f_hip = (char) hip();
	size += 1; /* f_hip */
	if (f_hip) {
		f_hip1 = (char) hip1();
		size += 1; /* f_hip1 */
		if (f_hip1)
			size += 1 * l;
		else
			size += 4 * l;
		}
	else {
		for (i = 0; i < l; i++)
			size += s_i(i).calc_size_on_file();
		}
	return size;
}

void Vector::conjugate(discreta_base & a)
{
	discreta_base av, b;
	int i, l;
	
	av = a;
	av.invert();
	l = s_l();
	for (i = 0; i < l; i++) {
		b = av;
		b *= s_i(i);
		b *= a;
		s_i(i) = b;
		}
}

void Vector::conjugate_with_inverse(discreta_base & a)
{
	discreta_base av;
	
	av = a;
	av.invert();
	conjugate(av);
}

void merge(Vector &v1, Vector &v2, Vector &v3)
{
	int l1, l2, l3, i1 = 0, i2 = 0, r;
	int f_add1; //, f_add2;
	
	l1 = v1.s_l();
	l2 = v2.s_l();
	l3 = l1 + l2;
	v3.m_l(l3);
	while (i1 < l1 || i2 < l2) {
		f_add1 = FALSE;
		//f_add2 = FALSE;
		if (i1 < l1 && i2 < l2) {
			r = v1[i1].compare_with(v2[i2]);
			if (r < 0)
				f_add1 = TRUE;
			//else
			//	f_add2 = TRUE;
			}
		else if (i1 < l1)
			f_add1 = TRUE;
		//else
		//	f_add2 = TRUE;
		if (f_add1) {
			v3[i1 + i2] = v1[i1];
			i1++;
			}
		else {
			v3[i1 + i2] = v2[i2];
			i2++;
			}
		}
}

void merge_with_fellows(Vector &v1, Vector &v1_fellow, 
	Vector &v2, Vector &v2_fellow, 
	Vector &v3, Vector &v3_fellow)
{
	int l1, l2, l3, i1 = 0, i2 = 0, r;
	int f_add1; //, f_add2;
	
	l1 = v1.s_l();
	l2 = v2.s_l();
	l3 = l1 + l2;
	v3.m_l(l3);
	v3_fellow.m_l(l3);
	while (i1 < l1 || i2 < l2) {
		f_add1 = FALSE;
		//f_add2 = FALSE;
		if (i1 < l1 && i2 < l2) {
			r = v1[i1].compare_with(v2[i2]);
			if (r < 0)
				f_add1 = TRUE;
			//else
			//	f_add2 = TRUE;
			}
		else if (i1 < l1)
			f_add1 = TRUE;
		//else
		//	f_add2 = TRUE;
		if (f_add1) {
			v3[i1 + i2] = v1[i1];
			v3_fellow[i1 + i2] = v1_fellow[i1];
			i1++;
			}
		else {
			v3[i1 + i2] = v2[i2];
			v3_fellow[i1 + i2] = v2_fellow[i2];
			i2++;
			}
		}
}

void merge_with_value(Vector &idx1, Vector &idx2, Vector &idx3, 
	Vector &val1, Vector &val2, Vector &val3)
{
	int i1, i2, l1, l2, a1, a2, f_add1, f_add2;
	Vector v;
	
	idx3.m_l(0);
	val3.m_l(0);
	i1 = 0;
	i2 = 0;
	l1 = idx1.s_l();
	l2 = idx2.s_l();
	while (i1 < l1 || i2 < l2) {
		f_add1 = FALSE;
		f_add2 = FALSE;
		if (i1 < l1 && i2 < l2) {
			a1 = idx1.s_ii(i1);
			a2 = idx2.s_ii(i2);
			if (a1 == a2) {
				v.m_l(2);
				v.m_ii(0, val1.s_ii(i1));
				v.m_ii(1, val2.s_ii(i2));
				idx3.append_integer(a1);
				val3.append(v);
				i1++;
				i2++;
				}
			else if (a1 < a2)
				f_add1 = TRUE;
			else 
				f_add2 = TRUE;
			}
		else {
			if (i1 < l1)
				f_add1 = TRUE;
			else
				f_add2 = TRUE;
			}
		if (f_add1) {
			a1 = idx1.s_ii(i1);
			v.m_l(2);
			v.m_ii(0, val1.s_ii(i1));
			v.m_ii(1, 0);
			idx3.append_integer(a1);
			val3.append(v);
			i1++;
			}
		if (f_add2) {
			a2 = idx2.s_ii(i2);
			v.m_l(2);
			v.m_ii(0, 0);
			v.m_ii(1, val2.s_ii(i2));
			idx3.append_integer(a2);
			val3.append(v);
			i2++;
			}
		
		}
}

void Vector::replace(Vector &v)
{
	int i, l, a, b;
	
	l = s_l();
	for (i = 0; i < l; i++) {
		a = s_ii(i);
		b = v.s_ii(a);
		m_ii(i, b);
		}
}

void Vector::vector_of_vectors_replace(Vector &v)
{
	int i, l;
	
	l = s_l();
	for (i = 0; i < l; i++) {
		s_i(i).as_vector().replace(v);
		}
}

void Vector::extract_subvector(Vector & v, int first, int len)
{
	int i;
	
	v.m_l(len);
	for (i = 0; i < len; i++) {
		v.s_i(i) = s_i(first + i);
		}
}


void Vector::PG_element_normalize()
// top (=highest) element which is different from zero becomes one
{
	int i, j, l;
	discreta_base a;
	
	l = s_l();
	for (i = l - 1; i >= 0; i--) {
		if (!s_i(i).is_zero()) {
			if (s_i(i).is_one())
				return;
			a = s_i(i);
			a.invert();
			for (j = i; j >= 0; j--) {
				s_i(j) *= a;
				}
			return;
			}
		}
	cout << "Vector::PG_element_normalize() zero vector()" << endl;
	exit(1);
}

void Vector::PG_element_rank(int &a)
{
	domain *d;
	int l, i, j, q, q_power_j, b;
	
	if (!is_finite_field_domain(d)) {
		cout << "Vector::PG_element_rank() no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	l = s_l();
	if (l <= 0) {
		cout << "Vector::PG_element_rank() vector not allocated()" << endl;
		exit(1);
		}
	PG_element_normalize();
	for (i = l - 1; i >= 0; i--) {
		if (!s_i(i).is_zero())
			break;
		}
	if (i < 0) {
		cout << "Vector::PG_element_rank() zero vector" << endl;
		exit(1);
		}
	if (!s_i(i).is_one()) {
		cout << "Vector::PG_element_rank() vector not normalized" << endl;
		exit(1);
		}

	b = 0;
	q_power_j = 1;
	for (j = 0; j < i; j++) {
		b += q_power_j;
		q_power_j *= q;
		}


	a = 0;
	for (j = i - 1; j >= 0; j--) {
		a += s_ii(j);
		if (j > 0)
			a *= q;
		}
	a += b;
}

void Vector::PG_element_rank_modified(int &a)
{
	domain *d;
	int l, i, j, q, q_power_j, b;
	
	if (!is_finite_field_domain(d)) {
		cout << "Vector::PG_element_rank_modified() no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	l = s_l();
	if (l <= 0) {
		cout << "Vector::PG_element_rank_modified() vector not allocated()" << endl;
		exit(1);
		}
	PG_element_normalize();
	for (i = 0; i < l; i++) {
		if (!s_i(i).is_zero())
			break;
		}
	if (i == l) {
		cout << "Vector::PG_element_rank_modified() zero vector" << endl;
		exit(1);
		}
	for (j = i + 1; j < l; j++) {
		if (!s_i(j).is_zero())
			break;
		}
	if (j == l) {
		// we have the unit vector vector e_i
		a = i;
		return;
		}
	
	for (i = l - 1; i >= 0; i--) {
		if (!s_i(i).is_zero())
			break;
		}
	if (i < 0) {
		cout << "Vector::PG_element_rank_modified() zero vector" << endl;
		exit(1);
		}
	if (!s_i(i).is_one()) {
		cout << "Vector::PG_element_rank_modified() vector not normalized" << endl;
		exit(1);
		}

	b = 0;
	q_power_j = 1;
	for (j = 0; j < i; j++) {
		b += q_power_j - 1;
		q_power_j *= q;
		}


	a = 0;
	for (j = i - 1; j >= 0; j--) {
		a += s_ii(j);
		if (j > 0)
			a *= q;
		}
	a += b;
	a += l - 1;
}

void Vector::PG_element_unrank(int a)
{
	domain *d;
	int q, n, l, qhl, k, j, r, a1 = a;
	
	if (!is_finite_field_domain(d)) {
		cout << "Vector::PG_element_unrank() no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	n = s_l();
	if (n <= 0) {
		cout << "Vector::PG_element_unrank() vector not allocated()" << endl;
		exit(1);
		}
	
	l = 0;
	qhl = 1;
	while (l < n) {
		if (a >= qhl) {
			a -= qhl;
			qhl *= q;
			l++;
			continue;
			}
		s_i(l).one();
		for (k = l + 1; k < n; k++) {
			s_i(k).zero();
			}
		j = 0;
		while (a != 0) {
			r = a % q;
			m_ii(j, r);
			j++;
			a -= r;
			a /= q;
			}
		for ( ; j < l; j++)
			m_ii(j, 0);
		return;
		}
	cout << "Vector::PG_element_unrank() a too large" << endl;
	cout << "n = " << n << endl;
	cout << "a = " << a1 << endl;
	exit(1);
}

void Vector::PG_element_unrank_modified(int a)
{
	domain *d;
	int q, n, l, qhl, k, j, r, a1 = a;
	
	if (!is_finite_field_domain(d)) {
		cout << "Vector::PG_element_unrank_modified() no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	n = s_l();
	if (n <= 0) {
		cout << "Vector::PG_element_unrank_modified() vector not allocated()" << endl;
		exit(1);
		}
	if (a < n) {
		for (k = 0; k < n; k++) {
			if (k == a)
				s_i(k).one();
			else
				s_i(k).zero();
			}
		return;
		}
	a -= (n - 1);	
	
	l = 0;
	qhl = 1;
	while (l < n) {
		if (a >= qhl) {
			a -= (qhl - 1);
			qhl *= q;
			l++;
			continue;
			}
		s_i(l).one();
		for (k = l + 1; k < n; k++) {
			s_i(k).zero();
			}
		j = 0;
		while (a != 0) {
			r = a % q;
			m_ii(j, r);
			j++;
			a -= r;
			a /= q;
			}
		for ( ; j < l; j++)
			m_ii(j, 0);
		return;
		}
	cout << "Vector::PG_element_unrank_modified() a too large" << endl;
	cout << "n = " << n << endl;
	cout << "a = " << a1 << endl;
	exit(1);
}

void Vector::AG_element_rank(int &a)
{
	domain *d;
	int q, l, i;
	
	if (!is_finite_field_domain(d)) {
		cout << "Vector::AG_element_rank() no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	l = s_l();
	if (l <= 0) {
		cout << "Vector::AG_element_rank() vector not allocated()" << endl;
		exit(1);
		}
	a = 0;
	for (i = l - 1; i >= 0; i--) {
		a += s_ii(i);
		if (i > 0)
			a *= q;
		}
}

void Vector::AG_element_unrank(int a)
{
	domain *d;
	int q, n, i, b;
	
	if (!is_finite_field_domain(d)) {
		cout << "Vector::AG_element_unrank() no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	n = s_l();
	if (n <= 0) {
		cout << "Vector::AG_element_unrank() vector not allocated()" << endl;
		exit(1);
		}
	for (i = 0; i < n; i++) {
		b = a % q;
		m_ii(i, b);
		a /= q;
		}
}

int Vector::hamming_weight()
{
	int i, l, w;
	
	w = 0;
	l = s_l();
	for (i = 0; i < l; i++) {
		if (!s_i(i).is_zero())
			w++;
		}
	return w;
}

void Vector::scalar_product(Vector &w, discreta_base & a)
{
	int l, i;
	discreta_base b;
	
	l = s_l();
	if (l != w.s_l()) {
		cout << "Vector::scalar_product() l != w.s_l()" << endl;
		exit(1);
		}
	for (i = 0; i < l; i++) {
		if (i == 0) {
			a.mult(s_i(i), w[i]);
			}
		else {
			b.mult(s_i(i), w[i]);
			a += b;
			}
		}
}

void Vector::hadamard_product(Vector &w)
{
	int l, i;
	discreta_base b;
	
	l = s_l();
	if (l != w.s_l()) {
		cout << "Vector::hadamard_product() l != w.s_l()" << endl;
		exit(1);
		}
	for (i = 0; i < l; i++) {
		s_i(i) *= w[i];
		}
}

void Vector::intersect(Vector& b, Vector &c)
{
	int l1 = s_l();
	int l2 = b.s_l();
	int l3 = 0;
	int i, idx;
	
	if (l2 < l1) {
		b.intersect(*this, c);
		return;
		}
	c.m_l(l1);
	for (i = 0; i < l1; i++) {
		if (b.search(s_i(i), &idx)) {
			c[l3++] = s_i(i);
			}
		}
	c.realloc(l3);
}


void intersection_of_vectors(Vector& V, Vector& v)
// V is a Vector of sorted Vectors, 
// v becomes the set of elements lying in all Vectors of V
{
	Vector vl;
	int l, i, j;
	permutation p;
	Vector vv;
	
	l = V.s_l();
	if (l == 0) {
		cout << "intersection_of_vectors() no vectors" << endl;
		exit(1);
		}
	vl.m_l_n(l);
	for (i = 0; i < l; i++) {
		vl.m_ii(i, V[i].as_vector().s_l());
		}
	vl.sort_with_logging(p);
	j = p[0];
	v = V[j].as_vector();
	for (i = 1; i < l; i++) {
		j = p[i];
		v.intersect(V[j].as_vector(), vv);
		vv.swap(v);
		}
}

int Vector::vector_of_vectors_overall_length()
{
	int i, l, s = 0;
	
	l = s_l();
	for (i = 0; i < l; i++) {
		Vector &v = s_i(i).as_vector();
		if (v.s_kind() != VECTOR) {
			cout << "vector::vector_of_vectors_overall_length element is not a vector" << endl;
			cout << *this << endl;
			exit(1);
			}
		s += v.s_l();
		}
	return s;
}

void Vector::first_divisor(Vector &exponents)
{
	int l = exponents.s_l();
	m_l_n(l);
}

int Vector::next_divisor(Vector &exponents)
{
	int n, i;
	
	n = s_l();
	i = n - 1;
	if (i < 0)
		return FALSE;
	while (s_ii(i) == exponents.s_ii(i)) {
		m_ii(i, 0);
		i--;
		if (i < 0)
			return FALSE;
		}
	s_i(i).inc();
	return TRUE;
}

int Vector::next_non_trivial_divisor(Vector &exponents)
{
	int n, i;
	
	n = s_l();
	i = n - 1;
	if (i < 0)
		return FALSE;
	while (s_ii(i) == exponents.s_ii(i)) {
		m_ii(i, 0);
		i--;
		if (i < 0)
			return FALSE;
		}
	s_i(i).inc();
	for (i = 0; i < n; i++) {
		if (s_ii(i) < exponents.s_ii(i))
			break;
		}
	if (i < n)
		return TRUE;
	else
		return FALSE;
}

void Vector::multiply_out(Vector &primes, discreta_base &x)
{
	int n, i;
	discreta_base a;
	
	x.m_i_i(1);
	n = s_l();
	for (i = 0; i < n; i++) {
		if (s_ii(i) == 0)
			continue;
		a = primes[i];
		a.power_int(s_ii(i));
		x *= a;
		}
}

int Vector::hash(int hash0)
{
	int h = hash0;
	int i, l;
	
	l = s_l();
	h = hash_int(h, s_l());
	for (i = 0; i < l; i++) {
		if (s_i(i).s_kind() != INTEGER) {
			cout << "Vector::hash() must be vector of integers" << endl;
			exit(1);
			}
		h = hash_int(h, s_ii(i));
		}
	return h;
}

int Vector::is_subset_of(Vector &w)
// w must be sorted
{
	int i, idx;
	
	for (i = 0; i < s_l(); i++) {
		if (!w.search(s_i(i), &idx))
			return FALSE;
		}
	return TRUE;
}

void Vector::concatenation(Vector &v1, Vector &v2)
{
	int l1, l2, l3, i, k;
	
	l1 = v1.s_l();
	l2 = v2.s_l();
	l3 = l1 + l2;
	m_l(l3);
	k = 0;
	for (i = 0; i < l1; i++) {
		s_i(k) = v1.s_i(i);
		k++;
		}
	for (i = 0; i < l2; i++) {
		s_i(k) = v2.s_i(i);
		k++;
		}
}

#undef DEBUG_PRint_WORD_NICELY

void Vector::print_word_nicely(ostream &ost, int f_generator_labels, Vector &generator_labels)
{
	if (f_generator_labels) {
		print_word_nicely_with_generator_labels(ost, generator_labels);
		}
	else {
		print_word_nicely2(ost);
		}
#ifdef DEBUG_PRint_WORD_NICELY
	cout << *this << " = ";
	if (f_generator_labels) {
		print_word_nicely_with_generator_labels(cout, generator_labels);
		}
	else {
		print_word_nicely2(cout);
		}
	cout << endl;
#endif
}

void Vector::print_word_nicely2(ostream &ost)
{
	int i, j, e, l;
	char c;
	
	l = s_l();
	// ost << "";
	for (i = 0; i < l; i += e) {
		for (j = i + 1; j < l; j++) {
			if (!s_i(j).eq(s_i(i)))
				break;
			}
		e = j - i;
		c = 'a' + s_ii(i);
		ost << c;
		if (e >= 10) {
			ost << "^{" << e << "} ";
			}
		else if (e >= 2) {
			ost << "^" << e << " ";
			}
		else
			ost << " ";
		}
	// ost << "";
}

void Vector::print_word_nicely_with_generator_labels(ostream &ost, Vector &generator_labels)
{
	int i, j, e, l;
	
	l = s_l();
	for (i = 0; i < l; i += e) {
		for (j = i + 1; j < l; j++) {
			if (!s_i(j).eq(s_i(i)))
				break;
			}
		e = j - i;
		hollerith &h = generator_labels[s_ii(i)].as_hollerith();
		ost << h.s();
		if (e >= 10) {
			ost << "^{" << e << "} ";
			}
		else if (e >= 2) {
			ost << "^" << e << " ";
			}
		else
			ost << " ";
		}
}

void Vector::vector_of_vectors_lengths(Vector &lengths)
{
	int i, l, ll;
	
	l = s_l();
	lengths.m_l_n(l);
	for (i = 0; i < l; i++) {
		ll = s_i(i).as_vector().s_l();
			lengths.m_ii(i, ll);
		}
}

void Vector::get_element_orders(Vector &vec_of_orders)
{
	int i, l, o;
	
	l = s_l();
	vec_of_orders.m_l_n(l);
	for (i = 0; i < l; i++) {
		o = s_i(i).order();
		vec_of_orders.m_ii(i, o);
		}
}

}}
