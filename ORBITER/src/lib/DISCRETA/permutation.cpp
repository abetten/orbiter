// permutation.cpp
//
// Anton Betten
// 10.11.1999
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"

using namespace std;



namespace orbiter {
namespace discreta {



#undef PERMUTATION_CHANGE_KIND_VERBOSE
#undef PERMUTATION_COPY_VERBOSE

enum permutation_print_type { 
	integer_from_zero,
	integer_from_one,
	PG_1_q_element_tex
	};

enum permutation_print_type current_permutation_print_type = integer_from_zero;
domain *current_permutation_print_type_dom = NULL;

void permutation::set_print_type_integer_from_zero()
{
	current_permutation_print_type = integer_from_zero;
}

void permutation::set_print_type_integer_from_one()
{
	current_permutation_print_type = integer_from_one;
}

void permutation::set_print_type_PG_1_q_element(domain *dom)
{
	current_permutation_print_type = PG_1_q_element_tex;
	current_permutation_print_type_dom = dom;
}

void permutation::convert_digit(int i, hollerith &a)
{
	a.init("");
	if (current_permutation_print_type == integer_from_zero) {
		if (current_printing_mode() == printing_mode_gap) {
			a.append_i(i + 1);
			}
		else {
			a.append_i(i);
			}
		}
	else if (current_permutation_print_type == integer_from_one) {
		a.append_i(i + 1);
		}
	else if (current_permutation_print_type == PG_1_q_element_tex) {
		Vector v;
		
		v.m_l_n(2);
		{
		with w(current_permutation_print_type_dom);
		v.PG_element_unrank(i);
		}
		if (v.s_ii(1) == 0)
			a.append("\\infty");
		else
			a.append_i(v.s_ii(0));
		}
	
}



permutation::permutation()
{
	k = PERMUTATION;
	self.vector_pointer = NULL;
}

permutation::permutation(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "permutation::copy constructor for object: " << const_cast<discreta_base &>(x) << "\n";
	clearself();
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

permutation& permutation::operator = (const discreta_base &x)
	// copy assignment
{
	// cout << "permutation::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void permutation::settype_permutation()
{
	OBJECTSELF s;
	
	s = self;
	new(this) permutation;
	self = s;
	k = PERMUTATION;
}

permutation::~permutation()
{
	freeself_permutation();
}

void permutation::freeself_permutation()
{
	// cout << "permutation::freeself_permutation()\n";
	if (self.vector_pointer == NULL) {
		// cout << "returning\n";
		return;
		}
	// cout << "free_nobjects_plus_length()\n";
	free_nobjects_plus_length(self.vector_pointer);
	self.vector_pointer = NULL;
}

kind permutation::s_virtual_kind()
{
	return PERMUTATION;
}

void permutation::copyobject_to(discreta_base &x)
{
	int i, l;
	
#ifdef PERMUTATION_COPY_VERBOSE
	cout << "permutation::copyobject_to()\n";
#endif
	x.freeself();
	if (x.s_kind() != PERMUTATION) {
		x.c_kind(PERMUTATION);
		x.self.vector_pointer = NULL;
		// x.printobjectkindln();
		}
	l = s_l();
#ifdef PERMUTATION_COPY_VERBOSE
	cout << "l=" << l << "\n";
#endif
	permutation & xx = x.as_permutation();
#ifdef PERMUTATION_COPY_VERBOSE
	cout << "calling xx.m_l()\n";
#endif
	xx.m_l(l);
	for (i = 0; i < l; i++) {
#ifdef PERMUTATION_COPY_VERBOSE
		cout << "copy " << i << ": " << s_i(i) << endl;
#endif
		xx[i] = s_i(i);
		}
}

ostream& permutation::print(ostream& ost)
{
	return print_cycle(ost);
}

ostream& permutation::print_list(ostream& ost)
{
	return as_vector().Vector::print(ost);
}

ostream& permutation::print_cycle(ostream& ost)
{
	if (current_printing_mode() == printing_mode_ascii || current_printing_mode() == printing_mode_latex) {
		Vector have_seen;
		int l, l1, first, next, len, n;
		int f_nothing_printed_at_all = TRUE;
		
		n = s_l();
		have_seen.m_l(n);
		for (l = 0; l < n; l++) {
			have_seen[l].m_i_i(FALSE);
			}
		l = 0;
		while (l < n) {
			if (have_seen[l].s_i_i()) {
				l++;
				continue;
				}
			/* Bearbeite Zyklus, beginnend mit l: */
			first = l;
			l1 = l;
			len = 1;
			while (TRUE) {
				have_seen[l1].m_i_i(TRUE);
				next = s_ii(l1);
				if (next > n) {
					cout << "permutation::print_cycle: next = " 
						<< next << " > n = " << n << endl;
					print_list(ost);
					exit(1);
					}
				if (next == first) {
					break;
					}
				if (have_seen[next].s_i_i()) {
					cout << "permutation::print_cycle: have_seen[next]\n"; 
					print_list(ost);
					exit(1);
					}
				l1 = next;
				len++;
				}
			if (len == 1)
				continue;
			f_nothing_printed_at_all = FALSE;
			/* Drucke Zyklus, beginnend mit first: */
			l1 = first;
			ost << "(";
			while (TRUE) {
				{
				hollerith a;
				convert_digit(l1, a);
				ost << a;
				}
				next = s_ii(l1);
				if (next == first) {
					break;
					}
				ost << ", ";
				l1 = next;
				}
			ost << ")";
			}
		if (f_nothing_printed_at_all) {
			ost << "id";
			}
		}
	
	else if (current_printing_mode() == printing_mode_gap) {
		Vector have_seen;
		int l, l1, first, next, len, n;
		int f_nothing_printed_at_all = TRUE;
		
		n = s_l();
		have_seen.m_l(n);
		for (l = 0; l < n; l++) {
			have_seen[l].m_i_i(FALSE);
			}
		l = 0;
		while (l < n) {
			if (have_seen[l].s_i_i()) {
				l++;
				continue;
				}
			/* Bearbeite Zyklus, beginnend mit l: */
			first = l;
			l1 = l;
			len = 1;
			while (TRUE) {
				have_seen[l1].m_i_i(TRUE);
				next = s_ii(l1);
				if (next > n) {
					cout << "permutation::print_cycle: next = " 
						<< next << " > n = " << n << endl;
					print_list(ost);
					exit(1);
					}
				if (next == first) {
					break;
					}
				if (have_seen[next].s_i_i()) {
					cout << "permutation::print_cycle: have_seen[next]\n"; 
					print_list(ost);
					exit(1);
					}
				l1 = next;
				len++;
				}
			if (len == 1)
				continue;
			f_nothing_printed_at_all = FALSE;
			/* Drucke Zyklus, beginnend mit first: */
			l1 = first;
			ost << "(";
			while (TRUE) {
				{
				hollerith a;
				convert_digit(l1, a);
				ost << a;
				}
				next = s_ii(l1);
				if (next == first) {
					break;
					}
				ost << ", ";
				l1 = next;
				}
			ost << ")";
		}
		if (f_nothing_printed_at_all) {
			ost << "(";
#if 0
				{
				hollerith a;
				convert_digit(n - 1, a);
				ost << a;
				}
#endif
			ost << ")";
			}
		}
			
	return ost;
}


void permutation::sscan(const char *s, int f_v)
{
	istringstream ins(s);
	scan(ins, f_v);
}

void permutation::scan(istream & is, int f_v)
//Scans a permutation from a stream.
{
	int l = 20;
	Vector cycle;
	permutation perm;
	int i, a_last, a, dig, ci;
	char s[10000], c;
	int si, largest_point = 0;
	
	//l = s_l();
	perm.m_l(l);
	cycle.m_l_n(l);
	perm.one();
	while (TRUE) {
		c = get_character(is, f_v);
		while (c == ' ' || c == '\t') {
			c = get_character(is, f_v);
			}
		ci = 0;
		if (c != '(') {
			break;
			}
		if (f_v) {
			cout << "opening parenthesis" << endl;
			}
		c = get_character(is, f_v);
		while (TRUE) {
			while (c == ' ' || c == '\t')
				c = get_character(is, f_v);
			
			si = 0;
			// read digits:
			while (c >= '0' && c <= '9') {
				s[si++] = c;
				c = get_character(is, f_v);
				}
			while (c == ' ' || c == '\t')
				c = get_character(is, f_v);
			if (c == ',')
				c = get_character(is, f_v);
			s[si] = 0;
			dig = atoi(s);
			if (dig > largest_point)
				largest_point = dig;
			if (f_v) {
				cout << "digit as string: " << s << ", numeric: " << dig << endl;
				}
			if (dig < 0) { 
				cout << "permutation::scan(): digit < 0" << endl;
				exit(1);
				}
			if (dig >= l) {
				permutation perm1;
				Vector cycle1;
				int l1, i;
				
				l1 = MAXIMUM(l + (l >> 1), largest_point + 1);
				cout << "permutation::scan(): digit = " << dig << " >= " << l << ", extending permutation degree to " << l1 << endl;
				perm1.m_l(l1);
				for (i = 0; i < l; i++) {
					perm1.m_ii(i, perm.s_i(i));
					}
				perm.swap(perm1);
				cycle1.m_l_n(l1);
				for (i = 0; i < l; i++) {
					cycle1.m_ii(i, cycle.s_ii(i));
					}
				cycle.swap(cycle1);
				l = l1;
				}
			si = 0;
			cycle.m_ii(ci, dig + 1);
			ci++;
			if (c == ')') {
				if (f_v) {
					cout << "closing parenthesis, cycle = ";
					for (i = 0; i < ci; i++)
						cout << cycle.s_ii(i) << " ";
					cout << endl;
					}
				for (i = 1; i < ci; i++) {
					a_last = cycle.s_ii(i - 1);
					a = cycle.s_ii(i);
					perm.m_ii(a_last - 1, a - 1);
					}
				if (ci > 1) {
					a_last = cycle.s_ii(ci - 1);
					a = cycle.s_ii(0);
					perm.m_ii(a_last - 1, a - 1);
					}
				ci = 0;
				if (!is)
					break;
				//c = get_character(is, f_v);
				break;
				}
			} // loop for one cycle
		if (!is)
			break;
		while (c == ' ' || c == '\t')
			c = get_character(is, f_v);
		ci = 0;
		} // end of loop over all cycles
	{
	permutation perm1;
	int i;
	
	perm1.m_l(largest_point + 1);
	for (i = 0; i <= largest_point; i++) {
		perm1.m_ii(i, perm.s_i(i));
		}
	perm.swap(perm1);
	}
	if (f_v) {
		cout << "read permutation " << perm;
		}
	swap(perm);
}

int & permutation::s_i(int i)
{
	int l;
	
	if (self.vector_pointer == NULL) {
		cout << "permutation::s_i() vector_pointer == NULL\n";
		exit(1);
		}
	l = self.vector_pointer[-1].s_i_i();
	if ( i < 0 || i >= l ) {
		cout << "permutation::s_i() addressing error, i = " << i << ", length = " << l << "\n";
		exit(1);		
		}
	return self.vector_pointer[i].self.integer_value;
}

void permutation::mult_to(discreta_base &x, discreta_base &y)
{
	permutation& px = x.as_permutation();
	permutation py;
	int i, l;
	
	if (s_kind() != PERMUTATION) {
		cout << "permutation::mult_to() this not a permutation\n";
		exit(1);
		}
	if (x.s_kind() != PERMUTATION) {
		cout << "permutation::mult_to() x is not a permutation\n";
		exit(1);
		}
	l = s_l();
	if (px.s_l() != l) {
		cout << "permutation::mult_to() px.s_l() != l\n";
		exit(1);
		}
	py.m_l(l);
	for (i = 0; i < l; i++) {
		py[i] = px[s_i(i)];
		}
	y.swap(py);
}

int permutation::invert_to(discreta_base &x)
{
	permutation px;
	int i, j, l;
	
	if (s_kind() != PERMUTATION) {
		cout << "permutation::invert_to() this is not a permutation\n";
		exit(1);
		}
	l = s_l();
	px.m_l(l);
	for (i = 0; i < l; i++) {
		j = s_ii(i);
		px[j] = i;
		}
	x.swap(px);
	return TRUE;
}

void permutation::one()
{
	int i, l;
	
	l = s_l();
	for (i = 0; i < l; i++)
		s_i(i) = i;
}

int permutation::is_one()
{
	int i, l;
	
	l = s_l();
	for (i = 0; i < l; i++)
		if (s_i(i) != i)
			return FALSE;
	return TRUE;
}

int permutation::compare_with(discreta_base &a)
{
	int l1, l2, i, c;
	
	if (s_kind() != PERMUTATION) {
		return compare_with(a);
		}
	if (a.s_kind() != PERMUTATION) {
		cout << "a is not a permutation\n";
		exit(1);
		}
	permutation& p = a.as_permutation();
	l1 = s_l();
	l2 = p.s_l();
	for (i = 0; i < l1; i++) {
		if (i < l2) {
			c = s_i(i) - p[i];
			if (c)
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

#if 0
#if TEXDOCU
void permutation::perm2lehmercode(Vector v)
#else
Computes the lehmercode of a given permutation
#endif
{
	int i, j, k;
	
	s_s()->copy(vec);
	for (i = 0; i < vec->s_li(); ) {
		k = vec->s_ii(i) - 1;
		vec->m_ii(i, k);
		i++;
		for (j = i; j < vec->s_li(); j++)
			if (vec->s_ii(j) > k)
				vec->s_i(j)->dec();
		}
}
#endif


void permutation::write_mem(memory & M, int debug_depth)
{
	int i, l, a;
	
	l = s_l();
	M.write_int(l);
	if (ONE_char_int(l)) {
		for (i = 0; i < l; i++) {
			a = s_i(i) + 1;
			M.write_char((char) a);
			}
		}
	else {
		for (i = 0; i < l; i++) {
			a = s_i(i) + 1;
			M.write_int(a);
			}
		}
}

void permutation::read_mem(memory & M, int debug_depth)
{
	int i, l, a;
	char c;
	
	M.read_int(&l);
	m_l(l);
	if (ONE_char_int(l)) {
		for (i = 0; i < l; i++) {
			M.read_char(&c);
			a = (int) c;
			a--;
			s_i(i) = a;
			}
		}
	else {
		for (i = 0; i < l; i++) {
			M.read_int(&a);
			a--;
			s_i(i) = a;
			}
		}
}

int permutation::csf()
{
	int l;
	int size = 0;
	
	l = s_l();
	size += 4; /* l */
	if (ONE_char_int(l))
		size += l * 1;
	else
		size += l * 4;
	return size;
}

void permutation::get_fixpoints(Vector &f)
{
	int i, l;
	
	l = s_l();
	f.m_l(0);
	for (i = 0; i < l; i++) {
		if (s_ii(i) == i) {
			f.append_integer(i);
			}
		}
}

void permutation::induce_action_on_blocks(permutation & gg, Vector & B)
//Computes the induced action on the blocks of a design. 
//this contains the point-permutation, B the sorted list 
//of blocks of the simple (no repeated blocks) design. 
//gg will contain the action of degree B$->$s\_li(); 
//Important: B is a sorted vector of sorted blocks !!!
//This routine works fine only for \lq simple\rq designs, 
//e.g. no block occurs twice 
//exception: empty blocks are treated correctly, 
//they lie in the first positions of B (due to the sorting).
{
	int i, j, l, b, a, aa;
	int nb_empty, idx;
	Vector b1;
	
	b = B.s_l();
	gg.m_l(b);
	nb_empty = 0;
	for (i = 0; i < b; i++) {
		Vector & bl = B.s_i(i).as_vector();
		l = bl.s_l();
		if (l == 0) {
			idx = nb_empty;
			nb_empty++;
			}
		else {
			b1.m_l(l);
			for (j = 0; j < l; j++) {
				a = bl.s_ii(j);
				aa = s_ii(a);
				b1.m_ii(j, aa);
				}
			b1.sort();
			if (!B.search(b1, &idx)) {
				cout << "permutation::induce_action_on_blocks, image block not found " << *this << endl;
				cout << "block: [" << bl << "]" << endl;
				cout << "image block: [" << b1 << "]" << endl;
				exit(1);
				}
			}
		gg.m_ii(i, idx);
		}
}

void permutation::induce3(permutation & b)
//induction on three sets. Gives a permutation of degree $(n choose 3)$ 
//where $n$ is the degree of the permutation in this.
{
	int k, l, n, n3, m1, m2, m3, bm1, bm2, bm3, x;

	n = s_l();
	n3 = n * (n - 1) * (n - 2);
	n3 /= 2;
	n3 /= 3;
	b.m_l(n3);
	k = 0;
	for (m1 = 1; m1 <= n; m1++) {
		for (m2 = m1 + 1; m2 <= n; m2++) {
			for (m3 = m2 + 1; m3 <= n; m3++) {
				bm1 = s_ii(m1 - 1) + 1;
				bm2 = s_ii(m2 - 1) + 1;
				bm3 = s_ii(m3 - 1) + 1;
				if (bm1 > bm2) {
					x = bm1; bm1 = bm2; bm2 = x;
					}
				if (bm2 > bm3) {
					x = bm2; bm2 = bm3; bm3 = x;
					}
				if (bm1 > bm2) {
					x = bm1; bm1 = bm2; bm2 = x;
					}
				x = bm3 - bm2;
				for (l = bm1 + 1; l < bm2; l++)
					x += n - l;
				for (l = 1; l < bm1; l++) {
					if (n - l > 1)
						x += ((n - l) * (n - l - 1)) >> 1;
					}
				b.m_ii(k, x - 1);
				k++;
				} // next m3
			} // next m2
		} // next m1
	if (k != n3) {
		cout << "permutation::induce3() k != n3" << endl;
		exit(1);
		}
}

void permutation::induce2(permutation & b)
//a is in fact only a permutation of $1, .. n$. 
//It computes the induced action of a on the pairs $(i,j)$, 
//$1 \le i < j \le n$, which are enumerated in the following way:
//$\{ \{1,2\}, \{1,3\}, \ldots,\{2,3\},\ldots,\{n-1,n\}\}$.
{
	int n;
	int i, j, k, i1, j1, k1, m;
	combinatorics_domain Combi;
	
	n = s_l();
	m = (n * (n - 1)) >> 1;
	b.m_l(m);
	for (i = 0; i < n - 1; i++) {
		i1 = s_ii(i);
		for (j = i + 1; j < n; j++) {
			j1 = s_ii(j);
			k = Combi.ij2k(i, j, n);
			k1 = Combi.ij2k(i1, j1, n);
			b.m_ii(k, k1);
			}
		}
}

void permutation::induce_on_2tuples(permutation & p, int f_injective)
//computes induction on two sets.
//tuple2\_rank and tuple2\_unrank are used.
{
	int n, m, i, j, rank, i1, j1, rank1;

	n = s_l();
	if (f_injective)
		m = n * (n - 1);
	else
		m = n * n;
	p.m_l(m);
	for (rank = 0; rank < m; rank++) {
		tuple2_rank(rank, i, j, n, f_injective);
		i1 = s_ii(i) - 1;
		j1 = s_ii(j) - 1;
		rank1 = tuple2_unrank(i1, j1, n, f_injective);
		p.m_ii(rank, rank1);
		}
}

void permutation::add_n_fixpoints_in_front(permutation & b, int n)
{
	int i, j, l;

	l = s_l();
	b.m_l(n + l);
	for (i = 0; i < n; i++)
		b.m_ii(i, i);
	for (i = 0; i < l; i++) {
		j = s_ii(i);
		b.m_ii(n + i, n + j);
		}
}

void permutation::add_n_fixpoints_at_end(permutation & b, int n)
{
	int i, j, l;

	l = s_l();
	b.m_l(n + l);
	for (i = 0; i < l; i++) {
		j = s_ii(i);
		b.m_ii(i, j);
		}
	for (i = 0; i < n; i++)
		b.m_ii(l + i, l + i);
}

void permutation::add_fixpoint_in_front(permutation & b)
//Adds a fixpoint as the new first point of the premutation. 
//All other points are shifted up by one element.
{
	add_n_fixpoints_in_front(b, 1);
	// cout << "add_n_fixpoints_in_front() gives " << b << endl;
}

void permutation::embed_at(permutation & b, int n, int at)
//adds at fixpoints at the beginning, 
//n - at - l fixpoints at the end. 
//l is the length of the this permutation.
//Result is b.
{
	permutation q;
	int l, m;

	l = s_l();
	if (n < l) {
		cout << "permutation::embed_at() n < l" << endl;
		exit(1);
		}
	if (at + l >= n) {
		cout << "permutation::embed_at() at + l >= n" << endl;
		exit(1);
		}
	m = n - at - l; // this is >= 0 !
	add_n_fixpoints_in_front(q, at);
	q.add_n_fixpoints_at_end(b, m);
}

void permutation::remove_fixpoint(permutation & b, int i)
//Attention: $0 \le i < n$ (before: $1 \le i \le n$)
{
	int j, k, l;

	l = s_l();
	if (s_ii(i) != i) {
		cout << "permutation::remove_fixpoint(): i is not a fixpoint" << endl;
		exit(1);
		}
	b.m_l(l - 1);
	for (j = 0; j < l; j++) {
		if (j == i)
			continue;
		k = s_ii(j);
		if (k > i)
			k--;
		if (j > i)
			b.m_ii(j - 1, k);
		else
			b.m_ii(j, k);
		}
}

void permutation::join(permutation & a, permutation & b)
//Let $a$ and $b$ act on disjoint sets, and put the resulting permutation into this.
{
	int i, j, l1, l2, l3;

	l1 = a.s_l();
	l2 = b.s_l();
	l3 = l1 + l2;
	m_l(l3);
	for (i = 0; i < l1; i++) {
		j = a.s_ii(i);
		m_ii(i, j);
		}
	for (i = 0; i < l2; i++) {
		j = b.s_ii(i);
		m_ii(l1 + i, l1 + j);
		}
}

void permutation::cartesian_product_action(permutation & a, permutation & b)
//a acts on the rows, b on the columns of the cartesian product.
{
	int n, m, nm, i, j, rank, i1, j1, rank1;

	n = a.s_l();
	m = b.s_l();
	nm = n * m;
	m_l(nm);
	for (rank = 0; rank < nm; rank++) {
		i = rank / m;
		j = rank % m;
		i1 = a.s_ii(i);
		j1 = b.s_ii(j);
		rank1 = i1 * m + j1;
		m_ii(rank, rank1);
		}
}


void permutation::Add2Cycle(int i0, int i1)
{
	int N;
	
	N = s_l();
	if (i0 < 0 || i0 >= N || i1 < 0 || i1 >= N) 
	{
		cout << "permutation::Add2Cycle \ni? < 0 || i? >= N" << endl;
		exit(1);
	}
	m_ii(i0, i1);
	m_ii(i1, i0);
}

void permutation::Add3Cycle(int i0, int i1, int i2)
{
	int N;
	
	N = s_l();
	if (i0 < 0 || i0 >= N || i1 < 0 || 
		i1 >= N || i2 < 0 || i2 >= N)
	{
		cout << "permutation::Add3Cycle \ni? < 0 || i? >= N" << endl;
		exit(1);
	}
	m_ii(i0, i1);
	m_ii(i1, i2);
	m_ii(i2, i0);
}

void permutation::Add4Cycle(int i0, int i1, int i2, int i3)
{
	int N;
	
	N = s_l();
	if (i0 < 0 || i0 >= N || i1 < 0 || 
		i1 >= N || i2 < 0 || i2 >= N || i3 < 0 || i3 >= N) 
	{
		cout << "permutation::Add4Cycle \ni? < 0 || i? >= N" << endl;
		exit(1);
	}
	m_ii(i0, i1);
	m_ii(i1, i2);
	m_ii(i2, i3);
	m_ii(i3, i0);
}

void permutation::Add5Cycle(int i0, int i1, int i2, int i3, int i4)
{
	int N;
	
	N = s_l();
	if (i0 < 0 || i0 >= N || i1 < 0 || 
		i1 >= N || i2 < 0 || i2 >= N || 
		i3 < 0 || i3 >= N || i4 < 0 || i4 >= N) 
	{
		cout << "permutation::Add5Cycle \ni? < 0 || i? >= N" << endl;
		exit(1);
	}
	m_ii(i0, i1);
	m_ii(i1, i2);
	m_ii(i2, i3);
	m_ii(i3, i4);
	m_ii(i4, i0);
}

void permutation::AddNCycle(int first, int len)
{
	int N, i;
	
	N = s_l();
	if (first < 1 || first + len - 1 > N) 
	{
		cout << "permutation::AddNCycle \nfirst < 1 || first+len-1 > N" << endl;
		exit(1);
	}
	for (i = 0; i < len; i++) 
	{
		if (i == len - 1)
			m_ii(first + i - 1, first);
		else
			m_ii(first + i - 1, first + i + 1);
	}
}

void permutation::cycle_type(Vector& type, int f_v)
{
	Vector have_seen;
	int l, l1, first, next, len, n;
	
	n = s_l();
	have_seen.m_l(n);
	type.m_l_n(n);
	for (l = 0; l < n; l++) {
		have_seen[l].m_i_i(FALSE);
		}
	l = 0;
	while (l < n) {
		if (have_seen[l].s_i_i()) {
			l++;
			continue;
			}
		/* Bearbeite Zyklus, beginnend mit l: */
		first = l;
		l1 = l;
		len = 1;
		while (TRUE) {
			have_seen[l1].m_i_i(TRUE);
			next = s_ii(l1);
			if (next > n) {
				cout << "permutation::cycle_type: next = " 
					<< next << " > n = " << n << endl;
				print_list(cout);
				exit(1);
				}
			if (next == first) {
				break;
				}
			if (have_seen[next].s_i_i()) {
				cout << "permutation::cycle_type: have_seen[next]\n"; 
				print_list(cout);
				exit(1);
				}
			l1 = next;
			len++;
			}
		type.s_ii(len - 1)++;
		}
	if (f_v) {
		cout << "the permutation " << *this << " has cycle type " << type << endl;
		}
}

int permutation::nb_of_inversions(int f_v)
{
	Vector type;
	int i, l, inv = 0, ai;
	
	l = s_l();
	cycle_type(type, f_v);
	for (i = 1; i <= l; i++) {
		ai = type.s_ii(i - 1);
		inv += ai * (i - 1);
		}
	return inv;
}

int permutation::signum(int f_v)
{
	int inv = nb_of_inversions(f_v);
	if (EVEN(inv))
		return 1;
	else
		return -1;
}

int permutation::is_even(int f_v)
{
	int inv = nb_of_inversions(f_v);
	if (EVEN(inv))
		return TRUE;
	else
		return FALSE;
}


void permutation::cycles(Vector &cycles)
{
	Vector have_seen, cycle;
	int l, l1, first, next, len, n;

	cycles.m_l(0);
	n = s_l();
	have_seen.m_l(n);
	for (l = 0; l < n; l++) {
		have_seen[l].m_i_i(FALSE);
		}
	l = 0;
	while (l < n) {
		if (have_seen[l].s_i_i()) {
			l++;
			continue;
			}
		/* Bearbeite Zyklus, beginnend mit l: */
		first = l;
		l1 = l;
		len = 1;
		cycle.m_l_n(n);
		cycle.m_ii(len - 1, l);
		while (TRUE) {
			have_seen[l1].m_i_i(TRUE);
			next = s_ii(l1);
			if (next > n) {
				cout << "permutation::cycles: next = " 
					<< next << " > n = " << n << endl;
				print_list(cout);
				exit(1);
				}
			if (next == first) {
				break;
				}
			if (have_seen[next].s_i_i()) {
				cout << "permutation::print_cycle: have_seen[next]\n"; 
				print_list(cout);
				exit(1);
				}
			l1 = next;
			cycle.m_ii(len, l1);
			len++;
			}
		cycle.realloc(len);
		cycles.append(cycle);
		}
}

void permutation::restrict_to_subset(permutation &q, int first, int len)
{
	int i, j;
	
	q.m_l(len);
	for (i = 0; i < len; i++) {
		j = s_i(first + i);
		if (j > first + len) {
			cout << "permutation::restrict_to_subset() j > first + len, subset not invariant" << endl;
			exit(1);
			}
		if (j < first) {
			cout << "permutation::restrict_to_subset() j < first, subset not invariant" << endl;
			exit(1);
			}
		j -= first;
		q.s_i(i) = j;
		}
}

void permutation::induce_on_lines_of_PG_k_q(int k, int q, permutation &per, int f_v, int f_vv)
{
	domain *dom;
	int nb_pts, nb_lines;
	int i, j, p1, p2, q1, q2;
	geometry_global Gg;
	
	dom = allocate_finite_field_domain(q, f_v);
	matrix L;
	nb_pts = Gg.nb_PG_elements(k, q);
	nb_lines = nb_PG_lines(k, q);
	if (f_v) {
		cout << "nb_pts=" << nb_pts << " nb_lines=" << nb_lines << endl;
		}
	if (s_l() != nb_pts) {
		cout << "permutation::induce_on_lines_of_PG_k_q() s_l() != nb_pts" << endl;
		exit(1);
		}
	per.m_l(nb_lines);
	L.m_mn_n(k + 1, 2);
	{
	with ww(dom);
	for (i = 0; i < nb_lines; i++) {
		L.PG_line_unrank(i); // a matrix of two columns
		if (f_vv) {
			cout << "L=\n" << L << endl;
			}
		L.PG_point_rank(0, 0, 1, 0, k + 1, p1);
		if (f_vv) {
			cout << p1 << endl;
			}
		L.PG_point_rank(0, 1, 1, 0, k + 1, p2);
		if (f_vv) {
			cout << p2 << endl;
			}
		q1 = s_i(p1);
		q2 = s_i(p2);
		L.PG_point_unrank(0, 0, 1, 0, k + 1, q1);
		L.PG_point_unrank(0, 1, 1, 0, k + 1, q2);
		if (f_vv) {
			cout << "L'=\n" << L << endl;
			}
		L.PG_line_rank(j, f_vv);
		if (f_vv) {
			cout << "j=" << j << endl;
			}
		per.s_i(i) = j;
		if (f_vv) {
			cout << i << "=("<< p1 << "," << p2 << ") -> (" << q1 << "," << q2 << ") = " << j << endl;
			}
		}
	if (f_v) {
		cout << "the permutation \n" << *this << endl;
		cout << "on projective lines\n" << per << endl;
		}
	}
}

void permutation::singer_cycle_on_points_of_projective_plane(int p, int f_modified, int f_v)
{
	unipoly a;
	matrix M;
	int l;
	int f_action_from_right = TRUE;
	geometry_global Gg;
	
	a.Singer(p, 3, FALSE, FALSE);
	cout << "permutation::singer_cycle_on_points_of_projective_plane(): primitive polynomial: " << a << endl;
	
	domain *dom = allocate_finite_field_domain(p, f_v);
	l = Gg.nb_PG_elements(2, p);
	m_l(l);
	{
	with ww(dom);
	M.m_mn_n(3, 3);
	M[1][0].one();
	M[2][1].one();
	M[0][2] = a[0];
	M[1][2] = a[1];
	M[2][2] = a[2];
	M[0][2].negate();
	M[1][2].negate();
	M[2][2].negate();
	M.transpose();
	cout << "as matrix:" << endl;
	cout << M << endl;
	M.PG_rep(*this, f_action_from_right, f_modified);
	}
	if (f_v) {
		cout << "singer cycle on points of projective plane of order "<< p << " : " << *this << endl;
		}
}

void permutation::Cn_in_Cnm(int n, int m)
{
	int i, nm = n * m;
	
	if (n < 1) {
		cout << "permutation::Cn_in_Cnm()n < 1" << endl;
		exit(1);
		}
	if (m < 1) {
		cout << "permutation::Cn_in_Cnm()m < 1" << endl;
		exit(1);
		}
	// the cycle (0,1, 2 ... ,nm-1):
	m_l(nm);
	one();
	for (i = 0; i < nm - 1; i++)
		m_ii(i, i + 1);
	m_ii(nm - 1, 0);
	power_int(m);
}

int permutation::preimage(int i)
{
	int j, l;
	
	l = s_l();
	for (j = 0; j < l; j++) {
		if (s_i(j) == i)
			return j;
		}
	cout << "permutation::preimage() error: not a permutation" << endl;
	exit(1);
}

void permutation::m_l(int l)
{
	int i;
	
	freeself();
	self.vector_pointer = calloc_nobjects_plus_length(l, INTEGER);
	for (i = 0; i < l; i++) {
		s_i(i) = i;
		}
}

void signum_map(discreta_base & x, discreta_base &d)
{
	int sgn;
	
	if (x.s_kind() != PERMUTATION) {
		cout << "signum_map() x must be a permutation" << endl;
		exit(1);
		}
	permutation & p = x.as_permutation();
	int f_v = FALSE;
	sgn = p.signum(f_v);
	d.change_to_integer();
	d.m_i_i(sgn);
}

#if 0
char get_character(istream & is, int f_v)
{
	char c;
	
	if (!is) {
		cout << "get_character() at end" << endl;
		exit(1);
		}
	is >> c;
	if (f_v) {
		cout << "get_character: \"" << c << "\", ascii=" << (int)c << endl;
		}
	return c;
}
#endif

}}


