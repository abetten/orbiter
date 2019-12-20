// perm_group_gens.cpp
//
// Anton Betten
// 27.07.2000
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace discreta {

static void schreier_trace(Vector & schreier, Vector & schreier_generator, 
	Vector & generators, int i, discreta_base & p);

int vec_generators_is_trivial_group(Vector & gen)
//TRUE if the generators are all the identity, FALSE otherwise.
{
	int i, l;
	
	l = gen.s_l();
	for (i = 0; i < l; i++) {
		if (!gen[i].is_one())
			return FALSE;
		}
	return TRUE;
}

int is_abelian(Vector & gen)
//True if the elements in the vector gen generate an abelian group.
{
	discreta_base a;
	int i, j, l;
	
	l = gen.s_l();
	for (i = 0; i < l; i++) {
		for (j = i + 1; j < l; j++) {
			a.commutator(gen[i], gen[j]);
			if (!a.is_one())
				return FALSE;
			}
		}
	return TRUE;
}


#if XML_AVAILABLE
#include <parser.h>
#endif

void read_file_of_generators_xml(Vector & gen, char *fname, int &f_cyclic_notation, int verbose_level)
//opens the file fname for reading and reads generators.
//A typical input file is:
//<GENERATORS NOTATION="CYCLE" DEGREE="6">
//<PERMUTATION>
//(0,1,2,3,4,5)
//</PERMUTATION>
//<PERMUTATION>
//(0,1)
//</PERMUTATION>
//</GENERATORS>
{
#if XML_AVAILABLE
	xmlDocPtr doc;
	xmlNodePtr cur;
	int i, j, size, degree;
	char *notation, *s;
	
	
	f_cyclic_notation = TRUE;
	size = file_size(fname);
	if (f_v) {
		cout << "reading file " << fname << " of size " << size << endl;
		}
	if (size <= 0) {
		cout << "the file " << fname << " cannot be opened!\n";
		exit(1);
		}
	gen.m_l(0);

	doc = xmlParseFile(fname);
	cur = doc->root;

	if (strcmp((const char *) cur->name, "GENERATORS")) { 
		cout << "error: file must begin with <GENERATORS>" << endl;
		exit(1);
		}
	notation = (char *) xmlGetProp(cur, (const xmlChar *) "NOTATION");
	degree = atoi((char *) xmlGetProp(cur, (const xmlChar *) "DEGREE"));
	
	if (f_v) {
		cout << "NOTATION=" << notation << endl;
		cout << "DEGREE=" << degree << endl;
		}
	if (notation == NULL)
		notation = "";
	if (strcmp(notation, "CYCLIC") == 0)
		f_cyclic_notation = TRUE;
	else if (strcmp(notation, "LIST") == 0)
		f_cyclic_notation = FALSE;
	else {
		cout << "unknown NOTATION, assuming LIST type (NOTATION can be either CYCLIC or LIST)" << endl;
		}
	cur = cur->childs;

	while (cur != NULL) {
		if (f_v) {
			cout << "scanning " << cur->name << endl;
			}
		if (!strcmp((const char *) cur->name, "PERMUTATION")) {
			permutation p;
			
			p.m_l(degree);
			// s = (char *) xmlNodeGetContent(cur);
			s = (char *) xmlNodeListGetString(doc, cur->childs, 1);
			if (f_v) {
				cout << "reading PERMUTATION: " << s << endl;
				}
			if (f_cyclic_notation) {
				istringstream ins(s, strlen(s) + 1);
				p.scan(ins, FALSE);
				if (p.s_l() < degree) {
					permutation p1;
					p.add_n_fixpoints_at_end(p1, degree - p.s_l());
					p = p1;
					}
				}
			else {
				istrstream ins(s, strlen(s) + 1);
				for (i = 0; i < degree; i++) {
					ins >> j;
					p.m_ii(i, j - 1);
					}
				}
			if (f_v) {
				cout << "read permutation " << p << endl;
				}
			gen.append(p);
			}
		cur = cur->next;
		}

	xmlFreeDoc(doc);
#else
	cout << "read_file_of_generators_xml(): XML library not available, quitting\n";
	exit(1);
#endif
}

void write_file_of_generators_xml_group_label(Vector & gen, char *group_label, int f_cyclic_notation)
//opens the file group_label.xml for writing and writes generators.
{
	hollerith fname;
	
	fname.init(group_label);
	fname.append(".xml");
	write_file_of_generators_xml(gen, fname.s(), f_cyclic_notation);
}

void write_file_of_generators_xml(Vector & gen, char *fname, int f_cyclic_notation)
//opens the file fname for writing and writes generators.
{
	int i, l, d;
	ofstream f(fname);
	
	l = gen.s_l();
	if (l == 0) {
		cout << "write_file_of_generators_xml() no generators" << endl;
		exit(1);
		}
	d = vec_generators_degree(gen);

	f << "<GENERATORS NOTATION=\"";
	if (f_cyclic_notation)
		f << "CYCLIC";
	else
		f << "LIST";
	f << "\" DEGREE=\"" << d << "\">\n";
	for (i = 0; i < l; i++) {
		f << "<PERMUTATION>\n";
		if (f_cyclic_notation) {
			f << gen[i].as_permutation() << endl;
			}
		else {
			gen[i].as_permutation().print_list(f);
			f << endl;
			}
		f << "</PERMUTATION>\n";
		}
	f << "</GENERATORS>" << endl;
}

void read_file_of_generators(Vector & G, char *fname)
// opens the file fname for reading and read generators via read\_generators().
{
	ifstream f(fname);
	
	if (!f) {
		cout << "read_file_of_generators(): can't open file '" << fname << "' for reading !\n";
		exit(1);
		}
	read_generators(G, f);
}

void read_generators(Vector & G, ifstream & f)
{
	int i, j, a, nb, deg;
	permutation p;
	
	f >> nb >> deg;
	G.m_l(nb);
	cout << "nb=" << nb << " deg=" << deg << endl;
	for (i = 0; i < nb; i++) {
		p.m_l(deg);
		for (j = 0; j < deg; j++) {
			f >> a;
			cout << a << " ";
			p.m_ii(j, a - 1);
			}
		cout << endl;
		G.s_i(i) = p;
		}
}

void write_file_of_generators_group_label(Vector & gen, char *group_label)
//opens the file group_label.txt for writing and writes generators.
{
	hollerith fname;
	
	fname.init(group_label);
	fname.append(".txt");
	write_file_of_generators(gen, fname.s());
}

void write_file_of_generators(Vector & G, char *fname)
// opens the file fname for writing and writes generators via write\_generators().
{
	ofstream f(fname);
	
	if (!f) {
		cout << "write_file_of_generators(): can't open file '" << fname << "' for writing !\n";
		exit(1);
		}
	write_generators(G, f);
}

void write_generators(Vector & G, ofstream & f)
{
	int i, j, a, nb, deg;
	
	nb = G.s_l();
	if (nb == 0) {
		cout << "write_generators() no generators" << endl;
		exit(1);
		}
	deg = G[0].as_permutation().s_l();
	f << nb << " " << deg << endl;
	for (i = 0; i < nb; i++) {
		permutation& p = G[i].as_permutation();
		for (j = 0; j < deg; j++) {
			a = p[j] + 1;
			f << a << " ";
			}
		f << endl;
		}
}

void write_file_of_generators_gap_group_label(Vector & gen, char *group_label)
//opens the file group_label.g for writing and writes generators.
{
	hollerith fname;
	
	fname.init(group_label);
	fname.append(".g");
	write_file_of_generators_gap(gen, fname.s());
}

void write_file_of_generators_gap(Vector & G, char *fname)
// opens the file fname for writing and writes generators via write\_generators\_gap().
{
	ofstream f(fname);
	
	if (!f) {
		cout << "write_file_of_generators_gap(): can't open file '" << fname << "' for writing !\n";
		exit(1);
		}
	write_generators_gap(G, f);
}

void write_generators_gap(Vector & G, ofstream & f)
{
	int i, nb;
	//permutation q;
	
	printing_mode pm(printing_mode_gap);

	//q.set_print_type_integer_from_one();
	nb = G.s_l();
	if (nb == 0) {
		cout << "write_generators_gap() no generators" << endl;
		exit(1);
		}
	f << "gens := [" << endl;
	for (i = 0; i < nb; i++) {
		permutation& p = G[i].as_permutation();
		f << p;
		if (i < nb - 1)
			f << ", " << endl;
		}
	f << "];";
	//q.set_print_type_integer_from_zero();
}

void vec_induced_group_on_subset(Vector & V, Vector & subset, Vector & W)
//Assume the elements of $V$ stabilize the set \lq subset\rq.
//Then $W$ becomes the restriction of these generators to this set.
//The entries in subset must be in $\{0,1,\ldots,deg-1\}$.
{
	permutation q;
	int i, j, k, a, b, r, l;

	r = V.s_l();
	W.m_l(r);
	l = subset.s_l();
	for (i = 0; i < r; i++) {
		permutation & p = V[i].as_permutation();
		q.m_l(l);
		for (j = 0; j < l; j++) {
			a = subset.s_ii(j);
			b = p.s_ii(a);
			for (k = 0; k < l; k++) {
				if (subset.s_ii(k) == b)
					break;
				}
			if (k == l) {
				cout << "vec_induced_group_on_subset(): the set is not invariant under the group" << endl;
				exit(1);
				}
			q.m_ii(j, k);
			}
		q.swap(W[i].as_permutation());
		}
}

void vec_subgroup_of_hol_of_cyclic_group(Vector & V, int n, int i)
//Computes generators for the subgroup of $Hol(C_n)$ of index $i$.
//$n$ must be a prime.
{
	long int m, m1, g, alpha, alpha1, j, jj;
	permutation p;
	int f_v = TRUE;
	number_theory_domain NT;
	
	if (!NT.is_prime(n)) {
		cout << "vec_subgroup_of_hol_of_cyclic_group() n must be a prime !" << endl;
		exit(1);
		}
	m = n - 1;
	m1 = m / i;
	g = NT.gcd_lint(m, i);
	if (m1 * i != m) {
		i = g;
		cout << "WARNING: vec_subgroup_of_hol_of_cyclic_group(): "
			"index " << i << " does not divide n - 1 = " << m << endl;
		cout << "setting index to " << g << endl;
		}
	m1 = m / g;
	cout << "vec_subgroup_of_hol_of_cyclic_group() creating group "
		"$" << n << " \\unlhd " << m1 << "$" << endl;

	alpha = NT.primitive_root(n, f_v);
	alpha1 = alpha;
	for (j = 2; j <= g; j++) {
		alpha1 = (alpha1 * alpha) % n;
		}
	cout << "primitive root alpha = " << alpha << endl;
	cout << "alpha^{" << g << "} = " << alpha1 << " mod " << n << endl;

	// the generator of the $n$-cycle:
	p.m_l(n);
	for (j = 0; j < n - 1; j++)
		p.m_ii(j, j + 1);
	p.m_ii(n - 1, 0);
	V.m_l(2);
	V[0] = p;
	cout << "generator of the cycle: " << p << endl;
	
	for (j = 0; j < n; j++) {
		jj = j * alpha1 % n;
		p.m_ii(j, jj);
		cout << j << " \\mapsto " << jj << "$" << endl;
		}
	V[1] = p;
	cout << "2nd generator: " << p << endl;
}

void vec_hol_of_cyclic_group(Vector & V, int n)
//Computes generators for the holomorph of $C_n$, 
//the cyclic group of $n$ elements. The vector V 
//becomes the vector of generators.
{
	Vector perm, perminv, red;
	permutation p;
	int flag, i, j, ii, a;

	perm.m_l_n(n + 1);
	perminv.m_l_n(n + 1);
	red.m_l_n(n + 1);

	p.m_l(n);
	for (i = 0; i < n - 1; i++)
		p.m_ii(i, i + 1);
	p.m_ii(n - 1, 0);
	V.m_l(0);
	V.append(p);
	for (a = 2; a < n; a++) {

		// we test the map: i \mapsto a * i mod n
		
		if (red.s_ii(a))
			continue;
		
		for (ii = 0; ii <= n; ii++)
			perminv.m_ii(ii, 0);
		flag = TRUE;
		
		for (i = 1; i < n; i++) {
			j = (i * a) % n;
			if (perminv.s_ii(j)) { // not an automorphism !
				flag = FALSE;
				break;
				}
			perminv.m_ii(j, i);
			perm.m_ii(i, j + 1);
	
			} // next i

		if (flag) { // we found an automorphism !
			i = 1;
			j = 0;
			do { // cycle of 1 under a:
				j++;
				i = (i * a) % n;
				red.m_ii(i, 1);
				} while (i > 1 && j < n);
			p.m_ii(0, 0);
			for (ii = 1; ii < n; ii++)
				p.m_ii(ii, perm.s_ii(ii) - 1);
			cout << "a=" << a << " p=" << p << endl;
			V.append(p);
			}
		
		} // next a
}

void vec_conjugate(Vector & gen, permutation & p)
//conjugates a vector of generators with the element p. 
//The new vector elements are $\{ p^{-1} g p \mid g \in gen\}$.
{
	permutation pv, tmp1;
	int i, l;

	pv = p;
	pv.invert();
	l = gen.s_l();
	for (i = 0; i < l; i++) {
		permutation & per = gen.s_i(i).as_permutation();
		if (per.s_kind() != PERMUTATION) {
			cout << "vec_conjugate() not a permutation" << endl;
			exit(1);
			}
		tmp1.mult(pv, per);
		per.mult(tmp1, p);
		}
}

void vec_induce_action_on_blocks(Vector & gen, Vector & B)
//calls induce\_action\_on\_blocks() for all elements in the vector gen.
//the result replaces the original vector gen.
{
	permutation q;
	int i, l;
	
	l = gen.s_l();
	for (i = 0; i < l; i++) {
		gen[i].as_permutation().induce_action_on_blocks(q, B);
		gen[i].swap(q);
		}
}

void vec_induce3(Vector & gen)
//calls induce3 for all elements in the vector gen.
{
	permutation q;
	int i, l;
	
	l = gen.s_l();
	for (i = 0; i < l; i++) {
		gen[i].as_permutation().induce3(q);
		gen[i].swap(q);
		}
}

void vec_induce2(Vector & gen)
//calls induce2 for all elements in the vector gen.
{
	permutation q;
	int i, l;
	
	l = gen.s_l();
	for (i = 0; i < l; i++) {
		gen[i].as_permutation().induce2(q);
		gen[i].swap(q);
		}
}

void vec_induce_on_2tuples(Vector & gen, int f_injective)
//calls induce_on_2tuples for all elements in the vector gen.
{
	permutation q;
	int i, l;
	
	l = gen.s_l();
	for (i = 0; i < l; i++) {
		gen[i].as_permutation().induce_on_2tuples(q, f_injective);
		gen[i].swap(q);
		}
}

void vec_add_fixpoint_in_front(Vector & gen)
//calls add\_fixpoint\_in\_front for all elements in the vector gen.
{
	permutation q;
	int i, l;
	
	l = gen.s_l();
	// cout << "vec_add_fixpoint_in_front() l=" << l << endl;
	for (i = 0; i < l; i++) {
		gen[i].as_permutation().add_fixpoint_in_front(q);
		gen[i].swap(q);
		}
}

void vec_add_fixpoint_at_end(Vector & gen)
//calls embed\_at(q, d + 1, 0); for all elements q in the vector gen.
{
	permutation q;
	int i, l;
	
	l = gen.s_l();
	for (i = 0; i < l; i++) {
		permutation &p = gen[i].as_permutation();
		int d = p.s_l();
		p.embed_at(q, d + 1, 0);
		gen[i].swap(q);
		}
}

int vec_generators_degree(Vector & a)
//Returns the degree of the (permutation-)group generated by the elements of a.
{
	int l, la;
	
	la = a.s_l();
	if (la == 0) {
		cout << "vec_generators_degree() la == 0" << endl;
		exit(1);
		}
	permutation & pa = a[0].as_permutation();
	if (pa.s_kind() != PERMUTATION) {
		cout << "vec_generators_degree() not applicable, not a vector of permutations !" << endl;
		exit(1);
		}
	l = pa.s_l();
	return l;
}

#if 0
void vec_generators_stabilize_point(Vector & a, Vector & b)
//Given a Permutation group $G$ acting on $X = \{0,1,\ldots, n-1\}$, 
//this routine computes generators for the stabilizer (in $G$) of 
//the first point: $G_0$.
//The first point is then removed.
//A Sims chain for $G$ is computed. Then, only generators 
//for $G_0$ are read out of this chain 
//(by a call to L.get_generators(1)).
{
	int i, l;
	Vector gen;

	l = a.s_l();	
	for (i = 0; i < l; i++) {
		if (a[i].as_permutation().s_i(0) != 0)
			break;
		}
	if (i < l) {
		perm_group G(a);
	
		G.get_generators(gen, 1);
		}
	else {
		gen = a;
		}
	l = gen.s_l();
	b.m_l(l);
	for (i = 0; i < l; i++) {
		b[i].change_to_permutation();
		gen[i].as_permutation().remove_fixpoint(b[i].as_permutation(), 0);
		}
}
#endif

#if 0
void vec_generators_group_order(Vector & gen, discreta_base & o)
{
	perm_group G(gen);
	
	G.group_order(o);
}
#endif

void vec_generators_remove_fixpoint(Vector & gen, int i)
//calls remove\_fixpoint(i) for all elements in the vector gen.
{
	permutation q;
	int ii, l;
	
	l = gen.s_l();
	for (ii = 0; ii < l; ii++) {
		gen[ii].as_permutation().remove_fixpoint(q, i);
		gen[ii].swap(q);
		}
}

void vec_generators_raise_to_nth_power(Vector & gen, int n)
//power_int(n) for all generators.
{
	int i, l;
	
	l = gen.s_l();
	for (i = 0; i < l; i++) {
		gen[i].power_int(n);
		}
}

void vec_generators_induce_on_lines_of_PG_k_q(Vector & gen, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;
	permutation pp;
	
	l = gen.s_l();
	for (i = 0; i < l; i++) {
		if (f_v) {
			cout << "i=" << i << " l=" << l << " : " << gen[i] << endl;
			}
		gen[i].as_permutation().induce_on_lines_of_PG_k_q(k, q, pp, verbose_level - 2);
		pp.swap(gen[i]);
		}
}

void vec_generators_trivial_group(Vector & gen, int deg)
//before: trivial_generators.
//Generator for the trivial permutation group on \lq deg\rq elements.
{
	permutation p;
	
	if (deg < 1) {
		cout << "vec_generators_trivial_group()|deg < 1" << endl;
		exit(1);
		}

	p.m_l(deg);
	p.one();
	gen.m_l(0);
	gen.append(p);
}

void vec_generators_cyclic_group(Vector & gen, int deg)
//Generator for the cyclic group on \lq deg\rq elements.
{
	permutation p;
	int i;
	
	if (deg < 1) {
		cout << "vec_generators_cyclic_group()|deg < 1" << endl;
		exit(1);
		}

	// the cycle (0,1, 2 ... ,deg-1):
	p.m_l(deg);
	p.one();
	for (i = 0; i < deg - 1; i++)
		p.m_ii(i, i + 1);
	p.m_ii(deg - 1, 0);
	gen.m_l(0);
	gen.append(p);
}

void vec_generators_Cn_in_Cnm(Vector & gen, int n, int m)
//Generator for the cyclic group $C_n$ as a subgroup of $C_{nm}$ (on $nm$ elements), 
//i.e. the $m$-th power of the generator of $C_{nm}.$
{
	gen.m_l(1);
	gen[0].change_to_permutation();
	gen[0].as_permutation().Cn_in_Cnm(n, m);
}

void vec_generators_AutCq_in_Cqm(Vector & gen, int q, int m)
//Generator for $Aut(C_n)$
//where q is a prime power, as a subgroup of $C_{qm}$ (on $qm$ elements).
{
	permutation per;
	int p, f;
	
	if (!factor_if_prime_power(q, &p, &f)) {
		cout << "vec_generators_AutCq_in_Cqm()|q must be a prime power" << endl;
		exit(1);
		}
	
	per.Cn_in_Cnm(q, m);

	if (p != 2) {
		
		}
	else {
		cout << "p = 2 not yet implemented" << endl;
		exit(1);
		}
	
	gen.m_l(1);
	gen[0] = per;
}

void vec_generators_symmetric_group(Vector & gen, int deg)
//Generators for the symmetric group on \lq deg\rq elements.
{
	permutation p;
	
	if (deg < 1) {
		cout << "vec_generators_symmetric_group()|deg < 1" << endl;
		exit(1);
		}

	vec_generators_cyclic_group(gen, deg);
	
	// the transposition (0,1):
	p.m_l(deg);
	p.one();
	p.m_ii(0, 1);
	p.m_ii(1, 0);
	gen.append(p);
}

void vec_generators_alternating_group(Vector & gen, int deg)
//Generators for the alternation group on \lq deg\rq elements.
{
	permutation p;
	int i, j, k;
	
	if (deg < 1) {
		cout << "vec_generators_alternating_group()|deg < 1" << endl;
		exit(1);
		}
	if (deg == 1 || deg == 2) {
		vec_generators_trivial_group(gen, deg);
		return;
		}

	gen.m_l(0);
	for (i = 0; i < deg; i++) {
		for (j = i + 1; j < deg; j++) {
			for (k = j + 1; k < deg; k++) {
				/* 3 cycle (i j k): */
				p.m_l(deg);
				p.one();
				p.m_ii(i, j); /* i -> j */
				p.m_ii(j, k); /* j -> k */
				p.m_ii(k, i); /* k -> i */
				gen.append(p);
				}
			}
		}	
}

void vec_generators_alternating_group_huppert(Vector & gen, int deg)
//Generators for the alternation group on \lq deg\rq elements.
//Generators as described in Huppert I, p 138.
{
	permutation p;
	int i;
	
	if (deg < 1) {
		cout << "vec_generators_alternating_group_huppert()|deg < 1" << endl;
		exit(1);
		}
	if (deg == 1 || deg == 2) {
		vec_generators_trivial_group(gen, deg);
		return;
		}

	gen.m_l(0);
	p.m_l(deg);
	p.one();
	// (0 1 2):
	p.s_i(0) = 1;
	p.s_i(1) = 2;
	p.s_i(2) = 0;
	gen.append(p);
	for (i = 1; i <= deg - 3; i++) {
		// (0 1)(i + 1, i + 2):
		p.m_l(deg);
		p.one();
		p.s_i(0) = 1;
		p.s_i(1) = 0;
		p.s_i(i + 1) = i + 2;
		p.s_i(i + 2) = i + 1;
		gen.append(p);
		}
}

void vec_generators_dihedral_group(Vector & gen, int deg)
//Generators for the dihedral group on \lq deg\rq elements.
{
	permutation p;
	int i, m;
	
	if (deg < 1) {
		cout << "vec_generators_dihedral_group() deg < 1" << endl;
		exit(1);
		}

	vec_generators_cyclic_group(gen, deg);

#if 0	
	// the involution (0,deg-1)(1,deg-2),...:
	p.m_l(deg);
	p.one();
	deg_2 = deg >> 1;
	for (i = 0; i < deg_2; i++) {
		p.m_ii(i, deg - i - 1);
		p.m_ii(deg - i - 1, i);
		}
#endif
	// the involution (1,deg-1)(2,deg-2),...:
	// (we prefer this because it is the point stabilizer of the point 0)
	p.m_l(deg);
	p.one();
	if (ODD(deg))
		m = (deg >> 1) + 1;
	else
		m = deg >> 1;
	for (i = 1; i < m; i++) {
		p.m_ii(i, deg - i);
		p.m_ii(deg - i, i);
		}
	gen.append(p);
}

void vec_generators_Mathieu_n(Vector & gen, int n)
//Returns generators for $M_n$ ($n \in \{11,12,23,24\}$).
//The generators are taken from Hall~\cite{Hall59}.
{
	if (n == 11)
		vec_generators_Mathieu_11(gen);
	else if (n == 12)
		vec_generators_Mathieu_12(gen);
	else if (n == 23)
		vec_generators_Mathieu_23(gen);
	else if (n == 24)
		vec_generators_Mathieu_24(gen);
	else {
		cout << "vec_generators_Mathieu_n() wrong n" << endl;
		exit(1);
		}
}


void vec_generators_Mathieu_11(Vector & gen)
//Returns generators for $M_11$.
//The generators are taken from Hall~\cite{Hall59}.
{
#if 0
	const char char *u[] = { "(1,2,3)(4,5,6)(7,8,9)", 
			"(2,4,3,7)(5,6,9,8)", 
			"(2,5,3,9)(4,8,7,6)", 
			"(1,10)(4,5)(6,8)(7,9)", 
			"(1,11)(4,6)(5,9)(7,8)" };`
#else
	const char *u[] = { "(0,1,2)(3,4,5)(6,7,8)(10)", 
			"(1,3,2,6)(4,5,8,7)(10)", 
			"(1,4,2,8)(3,7,6,5)(10)", 
			"(0,9)(3,4)(5,7)(6,8)(10)", 
			"(0,10)(3,5)(4,8)(6,7)" };
#endif
	permutation p;
	int f_v = FALSE;
	int i;
	
	gen.m_l(0);
	p.m_l(11);
	for (i = 0; i < 5; i++) {
		p.sscan(u[i], f_v);
		gen.append(p);
		}
	//vec_generators_remove_fixpoint(gen, 0);
}

void vec_generators_Mathieu_12(Vector & gen)
//Returns generators for $M_12$.
//$M_{12}$ is of order 95040.
//The generators are taken from Hall~\cite{Hall59}.
{
#if 0
	const char *u[] = { "(1,2,3)(4,5,6)(7,8,9)", 
			"(2,4,3,7)(5,6,9,8)", 
			"(2,5,3,9)(4,8,7,6)", 
			"(1,10)(4,5)(6,8)(7,9)", 
			"(1,11)(4,6)(5,9)(7,8)",
			"(1,12)(4,7)(5,6)(8,9)" };
#else
	const char *u[] = { "(0,1,2)(3,4,5)(6,7,8)(11)", 
			"(1,3,2,6)(4,5,8,7)(11)", 
			"(1,4,2,8)(3,7,6,5)(11)", 
			"(0,9)(3,4)(5,7)(6,8)(11)", 
			"(0,10)(3,5)(4,8)(6,7)(11)",
			"(0,11)(3,6)(4,5)(7,8)" };
#endif
	permutation p;
	int f_v = FALSE;
	int i;
	
	gen.m_l(0);
	p.m_l(12);
	for (i = 0; i < 6; i++) {
		p.sscan(u[i], f_v);
		gen.append(p);
		}
	//vec_generators_remove_fixpoint(gen, 0);
}

void vec_generators_Mathieu_23(Vector & gen)
//Returns generators for $M_{23}$.
//$M_{23}$ is 4-ply transitive of order 
//$23 \cdot 22 \cdot 21 \cdot 20 \cdot 16 \cdot 3 = 12926.008369.442488.320000$.
//It is the stabilizer of 24 in $M_{24}$.
{
#if 0
	const char *u[] = { "(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)", 
			"(3,17,10,7,9)(4,13,14,19,5)(8,18,11,12,23)(15,20,22,21,16)" };
#else
	const char *u[] = { "(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22)", 
			"(2,16,9,8,8)(3,12,13,18,4)(7,17,10,11,22)(14,19,21,20,15)" };
#endif
	permutation p;
	int f_v = FALSE;
	int i;
	
	gen.m_l(0);
	p.m_l(23);
	for (i = 0; i < 2; i++) {
		p.sscan(u[i], f_v);
		gen.append(p);
		}
	//vec_generators_remove_fixpoint(gen, 0);
}

void vec_generators_Mathieu_24(Vector & gen)
//Returns generators for $M_{24}$ of order $620448.401733.239439.360000$.
{
#if 0
	const char *u[] = { "(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)", 
			"(3,17,10,7,9)(4,13,14,19,5)(8,18,11,12,23)(15,20,22,21,16)",
			"(1,24)(2,23)(3,12)(4,16)(5,18)(6,10)(7,20)(8,14)(9,21)(11,17)(13,22)(15,19)" };
#else
	const char *u[] = { "(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22)(23)", 
			"(2,16,9,8,8)(3,12,13,18,4)(7,17,10,11,22)(14,19,21,20,15)(23)",
			"(0,23)(1,22)(2,11)(3,15)(4,17)(5,9)(6,19)(7,13)(8,20)(10,16)(12,21)(14,18)" };
#endif
	permutation p;
	int f_v = FALSE;
	int i;
	
	gen.m_l(0);
	p.m_l(24);
	for (i = 0; i < 3; i++) {
		p.sscan(u[i], f_v);
		gen.append(p);
		}
	//vec_generators_remove_fixpoint(gen, 0);
}

void vec_generators_diagonal_sum(Vector & a, Vector & b, Vector & c)
//given generators a and b for permutation groups $G$ and $H \simeq G$ on disjoint sets 
//$X$ and $Y$, this routine computes generators for the disjoint action of $G$ on 
//$X \cup Y$.
//Warning the generators for $G$ and $H$ must be in correspondence with 
//respect to an isomorphism from $G$ to $H$.
{
	permutation q;
	int i, la, lb;

	la = a.s_l();
	lb = b.s_l();
	if (la == 0) {
		cout << "vec_generators_diagonal_sum() la == 0" << endl;
		exit(1);
		}
	if (la != lb) {
		cout << "vec_generators_diagonal_sum() la != lb" << endl;
		exit(1);
		}
	
	c.m_l(0);
	for (i = 0; i < la; i++) {
		q.join(a[i].as_permutation(), b[i].as_permutation());
		c.append(q);
		}
}

void vec_generators_comma(Vector & a, Vector & b, Vector & c)
//Concatenates the lists (Vectors) of generators in a and b. 
//Result is c. Looks for the degree: if needed, one of the 
//generating sets is embedded to get the largest degree. 
//The fixepoints are added at the end.
{
	permutation q;
	int i, la, lb, da, db, dc;

	la = a.s_l();
	lb = b.s_l();
	da = vec_generators_degree(a);
	db = vec_generators_degree(b);
	dc = MAXIMUM(da, db);
	
	c.m_l(0);
	for (i = 0; i < la; i++) {
		a[i].as_permutation().add_n_fixpoints_at_end(q, dc - da);
		c.append(q);
		}
	for (i = 0; i < lb; i++) {
		b[i].as_permutation().add_n_fixpoints_at_end(q, dc - db);
		c.append(q);
		}
}

void vec_generators_direct_sum(Vector & a, Vector & b, Vector & c)
//given generators a and b for permutation groups $G$ and $H$ on disjoint sets 
//$X$ and $Y$, this routine computes generators for the disjoint action of 
//$X \cup Y$ (the action of the direct product of $G$ and $H$). 
{
	permutation q;
	int i, la, lb, da, db;

	la = a.s_l();
	lb = b.s_l();
	da = vec_generators_degree(a);
	db = vec_generators_degree(b);

	c.m_l(0);
	for (i = 0; i < la; i++) {
		a[i].as_permutation().add_n_fixpoints_at_end(q, db);
		c.append(q);
		}
	for (i = 0; i < lb; i++) {
		b[i].as_permutation().add_n_fixpoints_in_front(q, da);
		c.append(q);
		}
}

void vec_generators_direct_product(Vector & a, Vector & b, Vector & c)
//Given generators for permutation groups $G$ acting on $X$ 
//and $H$ acting of $Y$, this routine computes the action of 
//the direct product on $\{(x,y) \in X \times Y\}$ via 
//$(g \times h) \cdot (x,y) \mapsto (x^g, y^h)$.
//Each generator for $G$ or $H$ results in a generator  
//for $G \times H$ which moves only rows (or columns) of $X \times Y$.
{
	permutation id_n, id_m, q;
	int i, la, lb, n, m;

	la = a.s_l();
	lb = b.s_l();
	n = vec_generators_degree(a);
	m = vec_generators_degree(b);
	
	c.m_l(0);
	id_n.m_l(n);
	id_n.one();
	id_m.m_l(m);
	id_m.one();
	for (i = 0; i < la; i++) {
		q.cartesian_product_action(a[i].as_permutation(), id_m);
		c.append(q);
		}
	for (i = 0; i < lb; i++) {
		q.cartesian_product_action(id_n, b[i].as_permutation());
		c.append(q);
		}
}


void vec_generators_GL_n_q_as_matrices(Vector & gen,
		int n, domain *dom, int verbose_level)
{
	vec_generators_GL_n_q_subgroup_as_matrices(gen, n, 1, dom, verbose_level);
}

void vec_generators_GL_n_q_subgroup_as_matrices(Vector & gen,
		int n, int subgroup_index, domain *dom,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	with w(dom);
	discreta_matrix A;
	int i, j, alpha;
	
	if (f_v) {
		cout << "vec_generators_GL_n_q() n=" << n << endl;
		}
	alpha = finite_field_domain_primitive_root();
	if (f_v) {
		cout << "a primitive root is alpha=" << alpha << endl;
		cout << "a generating set is:" << endl;
		}
	if (subgroup_index > 1) {
		{
		with w(dom);
		
		integer a(alpha);

		a.power_int(subgroup_index);
		alpha = a.s_i();
		}
		if (f_v) {
			cout << "alpha^" << subgroup_index << " = " << alpha << endl;
			}
		}
	gen.m_l(0);
	for (i = 0; i < n; i++) {
		A.m_mn_n(n, n);
		A.one();
		A[i][i].as_integer().m_i((int) alpha);
		gen.append(A);
		if (f_v) {
			cout << A << endl;
			}
		}
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j)
				continue;
			A.m_mn_n(n, n);
			A.one();
			A[i][j].one();
			gen.append(A);
			if (f_v) {
				cout << A << endl;
				}
			}
		}
}

void vec_generators_SL_n_q_as_matrices(Vector & gen, int n, domain *dom, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	Vector GL_gen;
	
	vec_generators_GL_n_q_as_matrices(GL_gen, n, dom, verbose_level);

	with w(dom);
	
	kernel_of_homomorphism(GL_gen, gen, determinant_map, verbose_level);
}

void vec_generators_frobenius_in_PG(Vector & gen, int n, domain *dom, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	permutation p;
	
	frobenius_in_PG(dom, n, p);
	if (f_v) {
		cout << "frobenius_in_PG() n = " <<  n << ": \n" << p << endl;
		}
	gen.m_l(0);
	gen.append(p);
}

void vec_generators_frobenius_in_AG(Vector & gen, int n, domain *dom, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	permutation p;
	
	frobenius_in_AG(dom, n, p);
	if (f_v) {
		cout << "frobenius_in_AG() n = " <<  n << ": \n" << p << endl;
		}
	gen.m_l(0);
	gen.append(p);
}

void vec_generators_affine_translations(Vector & gen, int n, domain *dom, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	Vector b;
	with ww(dom);
	permutation p;
	int i, j, l;
	
	gen.m_l(0);
	finite_field_domain_base_over_subfield(b);
	l = b.s_l();
	if (f_v) {
		cout << "vec_generators_affine_translations(): discreta_base = " << b << endl;
		}
	for (i = 0; i < n; i++) {
		for (j = 0; j < l; j++) {
			translation_in_AG(dom, n, i, b[j], p);
			if (f_v) {
				cout << "i=" << i << " j=" << j << " translation: " << p << endl;
				}
			gen.append(p);
			}
		}
}

void vec_generators_affine_translations(Vector & gen, int n, int q, int verbose_level)
{
	domain *dom;
	
	dom = allocate_finite_field_domain(q, verbose_level);
	vec_generators_affine_translations(gen, n, dom, verbose_level);
	free_finite_field_domain(dom);
}

void vec_generators_projective_representation(domain *dom,
		Vector & a, Vector & b, int f_action_from_right, int f_modified,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;
	permutation q;
	
	if (f_v) {
		cout << "projective representation:" << endl;
		}
	b.m_l(0);
	l = a.s_l();
	for (i = 0; i < l; i++) {
		a[i].as_matrix().PG_rep(dom, q, f_action_from_right, f_modified);
		if (f_v) {
			{
			with ww(dom);
			cout << "the matrix \n" << a[i];
			}
			cout << "becomes \n" << q << endl << endl;
			}
		b.append(q);
		}
}

void vec_generators_affine_representation(domain *dom, Vector & a, Vector & b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;
	permutation q;
	
	if (f_v) {
		cout << "affine representation:" << endl;
		}
	b.m_l(0);
	l = a.s_l();
	for (i = 0; i < l; i++) {
		a[i].as_matrix().AG_rep(dom, q, TRUE /* f_action_from_right */);
		if (f_v) {
			{
			with ww(dom);
			cout << "the matrix \n" << a[i];
			}
			cout << "becomes \n" << q << endl << endl;
			}
		b.append(q);
		}
}

#define MY_DEBUG 0

void vec_generators_GL_n_q_projective_representation(Vector & gen,
		int n, int q, int f_special, int f_frobenius, int f_modified,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	domain *dom;
	Vector GL_gen, gen1, gen2;
	
	if (f_v) {
		cout << "vec_generators_GL_n_q_projective_representation" << endl;
	}
	dom = allocate_finite_field_domain(q, verbose_level);
	vec_generators_GL_n_q_as_matrices(GL_gen, n, dom, verbose_level);
#if MY_DEBUG
	{
	Vector gens;
	discreta_base o;
	vec_generators_projective_representation(dom, GL_gen, gens, f_v);
	vec_generators_group_order(gens, o);
	cout << "vec_generators_GL_n_q_projective_representation() group order GL=" << o << endl;
	write_file_of_generators_gap(gens, "GL_gens_debug.g");
	}
#endif
	if (f_special) {
		with w(dom);
	
		kernel_of_homomorphism(GL_gen, gen1, determinant_map, verbose_level);
		}
	else {
		gen1 = GL_gen;
		}
	vec_generators_projective_representation(dom, gen1, gen2, TRUE /* f_action_from_right */, f_modified, verbose_level);
#if MY_DEBUG
	{
	discreta_base o;
	vec_generators_group_order(gen2, o);
	cout << "vec_generators_GL_n_q_projective_representation() group order SL=" << o << endl;
	}
#endif
	if (f_frobenius) {
		Vector gen3;
		
		vec_generators_frobenius_in_PG(gen3, n - 1, dom, verbose_level);
		vec_generators_comma(gen2, gen3, gen);
		}
	else {
		gen2.swap(gen);
		}
	free_finite_field_domain(dom);
}

void vec_generators_GL_n_q_affine_representation(Vector & gen,
		int n, int q, int f_special, int f_frobenius, int f_translations,
		int verbose_level)
{
	vec_generators_GL_n_q_subgroup_affine_representation(gen,
			n, q, 1, f_special, f_frobenius, f_translations,
			verbose_level);
}

void vec_generators_GL_n_q_subgroup_affine_representation(Vector & gen,
		int n, int q, int subgroup_index,
		int f_special, int f_frobenius, int f_translations,
		int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	domain *dom;
	Vector GL_gen, gen1, gen2, gen4;
	
	dom = allocate_finite_field_domain(q, verbose_level - 2);
	vec_generators_GL_n_q_subgroup_as_matrices(GL_gen, n, subgroup_index, dom, verbose_level);
	if (f_special) {
		with w(dom);
	
		kernel_of_homomorphism(GL_gen, gen1, determinant_map, verbose_level);
		}
	else {
		gen1 = GL_gen;
		}
	vec_generators_affine_representation(dom, gen1, gen2, verbose_level);
	if (f_translations) {
		Vector gen3;
		
		vec_generators_affine_translations(gen3, n, dom, verbose_level);
		vec_generators_comma(gen2, gen3, gen4);
		}
	else {
		gen2.swap(gen4);
		}
	if (f_frobenius) {
		Vector gen3;
		
		vec_generators_frobenius_in_AG(gen3, n, dom, verbose_level);
		vec_generators_comma(gen4, gen3, gen);
		}
	else {
		gen4.swap(gen);
		}
	free_finite_field_domain(dom);
}

void kernel_of_homomorphism(Vector & gens, Vector & kernel_gens, 
	void (*hom)(discreta_base & x, discreta_base & image),
	int verbose_level)
//Computes generators for the kernel of a given homomorphism hom 
//given generators for a group.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	Vector rep, im_sorted, im_index;
	Vector schreier_back, schreier_gen;
	discreta_base g, h, ghv, a;
	integer int_ob;
	int nb_gen, i, j, k, next, next1, idx;
	
	// f_vv = TRUE;
	
	nb_gen = gens.s_l();
	if (nb_gen == 0) {
		cout << "kernel_of_homomorphism() no generators" << endl;
		exit(1);
		}

	kernel_gens.m_l(0);
	rep.m_l(0);
	im_sorted.m_l(0);
	rep.append(gens[0]);
	rep[0].one();
	hom(rep[0], a);
	if (!a.is_one()) {
		cout << "kernel_of_homomorphism() not homomorphic" << endl;
		exit(1);
		}
	im_sorted.append(a);
	im_index.m_l(1);
	im_index.m_ii(0, 0);
	schreier_back.m_l(1);
	schreier_gen.m_l(1);
	schreier_back.m_ii(0, -1);
	schreier_gen.m_ii(0, -1);
	i = 0;
	if (f_v) {
		cout << "in kernel_of_homomorphism():" << endl;
		}
	while (i < rep.s_l()) {
		for (j = 0; j < nb_gen; j++) {
			g.mult(rep[i], gens[j]);
			if (f_vv) {
				cout << "i=" << i << ", j=" << j << " rep[i]=" << rep[i] << ", gens[j]=" << gens[j] << ", g=rep[i]*gens[j] = " << g << endl;
				}
			hom(g, a);
			if (f_vv) {
				cout << "hom(g)=" << a << endl; 
				cout << "g=" << g << endl; 
				}
			if (!im_sorted.search(a, &next)) {
				idx = rep.s_l();
				rep.append(g);
				im_sorted.insert_element(next, a);
				int_ob.m_i(idx);
				im_index.insert_element(next, int_ob);
				int_ob.m_i(i);
				schreier_back.append(int_ob);
				int_ob.m_i(j);
				schreier_gen.append(int_ob);
				if (f_v) {
					cout << "new image element no " << rep.s_l() << ": " << a << endl;
					}
				if (f_vv) {
					domain dom(30000);
					with w(&dom);
					cout << "kernel_of_homomorphism(): " << endl;
					cout << "im_sorted = " << im_sorted << endl;
					cout << "im_index = " << im_index << endl;
					cout << "schreier_back = " << schreier_back << endl;
					cout << "schreier_gen = " << schreier_gen << endl;
					cout << "rep = " << rep << endl;
					cout << "kernel_gens = " << kernel_gens << endl;
					}
				}
			else {
				next1 = im_index.s_ii(next);
				schreier_trace(schreier_back, schreier_gen, gens, next1, h);
				if (f_vv) {
					cout << "g = " << g << endl;
					cout << "next = " << next << endl;
					cout << "next1 = " << next1 << endl;
					cout << "h = " << h << endl;
					}
				h.invert();
				if (f_vv) {
					cout << "h^{-1} = " << h << endl;
					}
				ghv.mult(g, h);
				if (f_vv) {
					cout << "generator = g*h^{-1} = " << ghv << endl;
					}
				if (!kernel_gens.search(ghv, &k)) {
					kernel_gens.insert_element(k, ghv);
					}
				}
			}
		
		i++;
		}
	if (f_v) {
		cout << "kernel_of_homomorphism() finished, number of generators = " << kernel_gens.s_l() << endl;
		for (i = 0; i < kernel_gens.s_l(); i++) {
			cout << "generator " << i << " = \n" << kernel_gens[i];
			hom(kernel_gens[i], a);
			cout << "maps to " << a << endl;
			}
		}
	if (f_vv) {
		domain dom(30000);
		with w(&dom);
		cout << "kernel_of_homomorphism(): " << endl;
		cout << "im_sorted = " << im_sorted << endl;
		cout << "im_index = " << im_index << endl;
		cout << "schreier_back = " << schreier_back << endl;
		cout << "schreier_gen = " << schreier_gen << endl;
		cout << "rep = " << rep << endl;
		cout << "kernel_gens = " << kernel_gens << endl;
		}
}

static void schreier_trace(Vector & schreier, Vector & schreier_generator, 
	Vector & generators, int i, discreta_base & p)
{
	discreta_base p1;
	int ii, prev, g;
	
	if (schreier.s_l() < i) {
		cout << "schreier_trace: schreier.s_l() < i" << endl;
		exit(1);
		}
	p = generators[0];
	p.one();
	ii = i;
	while (TRUE) {
		prev = schreier.s_ii(ii);
		g = schreier_generator.s_ii(ii);
		if (prev == -1)
			return;
		p1.mult(generators[g], p);
		p1.swap(p);
		ii = prev;
		}
}

void vec_generators_A5_in_PSL(Vector& G, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	hollerith a;
	int f_cyclic_notation;
	
	a.init("a5_in_psl.out ");
	a.append_i(q);
	if (f_v) {
		a.append(" -v");
		}
	system(a.s());
	
	a.init("A5_in_PSL_2_");
	a.append_i(q);
	a.append(".xml");

	read_file_of_generators_xml(G, a.s(), f_cyclic_notation, FALSE /* f_v */);
	
	cout << "vec_generators_A5_in_PSL() read file " << a << endl;
	cout << "found " << G.s_l() << " generators" << endl;
	if (G.s_l()) {
		cout << " of degree " << G[0].as_permutation().s_l() << endl;
		}
}


void vec_generators_S4_in_PSL(Vector& G, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	hollerith a;
	int f_cyclic_notation;
	
	a.init("s4_in_psl.out ");
	a.append_i(q);
	if (f_v) {
		a.append(" -v");
		}
	system(a.s());
	
	a.init("S4_in_PSL_2_");
	a.append_i(q);
	a.append(".xml");

	read_file_of_generators_xml(G, a.s(), f_cyclic_notation, FALSE /* f_v */);
	
	cout << "vec_generators_S4_in_PSL() read file " << a << endl;
	cout << "found " << G.s_l() << " generators" << endl;
	if (G.s_l()) {
		cout << " of degree " << G[0].as_permutation().s_l() << endl;
		}
}

void vec_generators_even_subgroup(Vector & gen, Vector & gen_even_subgroup, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	if (f_v) {
		cout << "computing generators for the subgroup of even elements:" << endl;
		cout << "(nb_gens=" << gen.s_l() << ")" << endl;
		}
	kernel_of_homomorphism(gen, gen_even_subgroup, signum_map, verbose_level);
	if (f_v) {
		cout << "generators for the subgroup of even elements are:" << endl;
		cout << gen_even_subgroup << endl;
		cout << "(nb_gens=" << gen_even_subgroup.s_l() << ")" << endl;
		}
}

#if 0
void vec_generators_on_conjugacy_class_of_subgroups_by_conjugation(perm_group &G, 
	Vector &LayerOrbit, int layer, int orbit, Vector &gens, Vector &induced_gens, int f_v, int f_vv)
{
	Vector action_data;
	action_data.m_l(1);
	action_data[0] = G;
	Vector gens_numeric;
	
	if (f_v) {
		cout << "vec_generators_on_conjugacy_class_of_subgroups_by_conjugation()" << endl;
		cout << "layer=" << layer << endl;
		cout << "orbit=" << orbit << endl;
		}
	G.rank_vector_of_elements(gens, gens_numeric);
	if (f_vv) {
		cout << "gens=" << gens << endl;
		cout << "gens_numeric=" << gens_numeric << endl;
		cout << "action_data = " << action_data << endl;
		}
	Vector &LO = LayerOrbit[layer].as_vector();
	Vector &orbit_data = LO[orbit].as_vector();
	Vector &Orbit = orbit_data[0].as_vector();
	if (f_vv) {
		cout << "Orbit=" << Orbit << endl;
		}
	
	induced_permutations_on_orbit(gens_numeric, induced_gens, Orbit, on_subset_of_group_elements_by_conjugation, action_data);
	cout << "induced action on that orbit: " << endl << induced_gens << endl;
}
#endif

void vec_generators_restrict_to_subset(Vector & gen, int first, int len)
//calls restrict_to_subset(first, len) for all elements in the vector gen.
{
	permutation q;
	int i, l;
	
	l = gen.s_l();
	for (i = 0; i < l; i++) {
		gen[i].as_permutation().restrict_to_subset(q, first, len);
		gen[i].swap(q);
		}
}

void wreath_embedding(permutation & g, int n, int m, permutation & q)
//utility function for computing wreath product generators.
{
	int i, j, ii, first, to;
	int nm;
	
	if (g.s_l() != m)
	{
		cout << "wreath_embedding() g.s_l() != m" << endl;
		exit(1);
	}
	nm = n * m;
	q.m_l(nm);
	for (i = 0; i < m; i++) {
		j = g.s_i(i);
		first = i * n;
		to = j * n;
		for (ii = 0; ii < n; ii++) {
			q.s_i(first + ii) = to + ii;
			}
		}
	cout << "leave wreath_embedding()" << endl;
}

void wreath_embedding_component(permutation & g, int n, int m, int j, permutation & q)
//utility function for computing wreath product generators.
{
	int i, i_im, first;
	int nm;
	
	if (g.s_l() != n)
	{
		cout << "wreath_embedding_component() g.s_l() != n" << endl;
		exit(1);
	}
	nm = n * m;
	q.m_l(nm);
	q.one();
	first = j * n;
	cout << "start for-loop in wreath_embedding_component():" << endl;
	for (i = 0; i < n; i++) {
		i_im = g.s_i(i);
		q.s_i(first + i) = first + i_im;
		}
	cout << "leave wreath_embedding_component()" << endl;
}

void vec_generators_wreath_product(Vector & G, Vector & H, Vector & W, int verbose_level)
//Computes generators for the wreath product $\la G \ra \wr \la H \ra$ into $W$.
{
	int f_v = (verbose_level >= 1);
	permutation per;
	int n, m, i, j, lG, lH, nb_gen;
	
	f_v = 1;

	lG = G.s_l();
	lH = H.s_l();
	n = G[0].as_permutation().s_l();
	cout << "G[0] = " << G[0] << endl;
	m = H[0].as_permutation().s_l();
	cout << "H[0] = " << H[0] << endl;
	nb_gen = 0;
	W.m_l(0);
	for (j = 0; j < m; j++) {
		if (f_v)
			cout << "embedding generators of G into " << j << "-component:" << endl;
		for (i = 0; i < lG; i++) {
			permutation & p = G[i].as_permutation();
			cout << "start wreath_embedding_component()" << endl;
			wreath_embedding_component(p, n, m, j, per);
			if (f_v)
				per.print_list(cout);
			W.inc();
			W[nb_gen] = per;
			nb_gen++;
			}
		}
	if (f_v)
		cout << "embedding generators of H:" << endl;
	for (i = 0; i < lH; i++) {
		permutation & p = H[i].as_permutation();
		cout << "start wreath_embedding()" << endl;
		wreath_embedding(p, n, m, per);
		if (f_v)
			per.print_list(cout);
		W.inc();
		W[nb_gen] = per;
		nb_gen++;
		}
}

void vec_generators_Sn_wreath_Sm(int n, int m, Vector & G)
//Computes generators for $S_n \wr S_m$ into $G$.
{
	permutation per;
	Vector Sn, Sm;
	int i, j, nb_gen_Sn, nb_gen_Sm, nb_gen;
	
	cout << "vec_generators_Sn_wreath_Sm(): n = " << n << " m = " << m << endl;
	if (n == 1) {
		return vec_generators_symmetric_group(G, m);
		}
	if (m == 1) {
		return vec_generators_symmetric_group(G, n);
		}
	vec_generators_symmetric_group(Sn, n);
	vec_generators_symmetric_group(Sm, m);
	nb_gen_Sn = Sn.s_l();
	nb_gen_Sm = Sm.s_l();
	nb_gen = 0;
	G.m_l(0);
	cout << "embedding generators of Sm:" << endl;
	for (i = 0; i < nb_gen_Sm; i++) {
		permutation & p = Sm[i].as_permutation();
		wreath_embedding(p, n, m, per);
		per.print_list(cout);
		G.inc();
		G[nb_gen] = per;
		nb_gen++;
		}
	for (j = 0; j < m; j++) {
		cout << "embedding generators of Sn into " << j << "-component:" << endl;
		for (i = 0; i < nb_gen_Sn; i++) {
			permutation & p = Sn[i].as_permutation();
			wreath_embedding_component(p, n, m, j, per);
			per.print_list(cout);
			G.inc();
			G[nb_gen] = per;
			nb_gen++;
			}
		}
}

void vec_generators_q1_q2(int q1, int q2, Vector & gen, hollerith &label, 
	int f_write_generators_to_file, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	Vector gen1, gen2;
	
	label.init("");
	label.append_i(q1);
	label.append("x");
	label.append_i(q2);
	
	domain *field_q1;
	domain *field_q2;
	
	field_q1 = allocate_finite_field_domain(q1, verbose_level - 2);
	field_q2 = allocate_finite_field_domain(q2, verbose_level - 2);
	
	vec_generators_affine_translations(gen1, 1 /* n */, field_q1, verbose_level);
	vec_generators_affine_translations(gen2, 1 /* n */, field_q2, verbose_level);

	free_finite_field_domain(field_q1);
	free_finite_field_domain(field_q2);

	if (f_vv) {
		cout << "generators for translations in GF(" << q1 << "):\n";
		gen1.Print(cout);
		cout << "generators for translations in GF(" << q2 << "):\n";
		gen2.Print(cout);
		cout << endl;
		}
	
	vec_generators_direct_product(gen1, gen2, gen);
	if (f_vv) {
		cout << "generators for direct product action:\n";
		gen.Print(cout);
		cout << endl;
		}
	if (f_write_generators_to_file) {
		int f_cyclic_notation = TRUE;
		write_file_of_generators_xml_group_label(gen, label.s(), f_cyclic_notation);
		write_file_of_generators_group_label(gen, label.s());
		write_file_of_generators_gap_group_label(gen, label.s());
		}
}

void vec_generators_q1_q2_aubv(int q1, int q2, int u, int v, Vector & G, hollerith &label, 
	int f_write_generators_to_file, int verbose_level)
{
	hollerith h;
	
	label.init("");
	label.append_i(q1);
	label.append("x");
	label.append_i(q2);
	label.append("_a");
	label.append_i(u);
	label.append("_b");
	label.append_i(v);
	
	vec_generators_q1_q2_au1bv1_au2bv2(q1, q2, u, v, 0, 0, G, h, FALSE, verbose_level);
	if (f_write_generators_to_file) {
		int f_cyclic_notation = TRUE;
		write_file_of_generators_xml_group_label(G, label.s(), f_cyclic_notation);
		write_file_of_generators_group_label(G, label.s());
		write_file_of_generators_gap_group_label(G, label.s());
		}
}

void vec_generators_q1_q2_au1bv1_au2bv2(int q1, int q2, int u1, int v1, int u2, int v2, 
	Vector & G, hollerith &label, int f_write_generators_to_file, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	Vector gen1, gen2, gen3, matrix_gen1;
	
	domain *field_q1;
	domain *field_q2;
	
	label.init("");
	label.append_i(q1);
	label.append("x");
	label.append_i(q2);
	label.append("_a");
	label.append_i(u1);
	label.append("_b");
	label.append_i(v1);
	label.append("_a");
	label.append_i(u2);
	label.append("_b");
	label.append_i(v2);


	field_q1 = allocate_finite_field_domain(q1, verbose_level);
	field_q2 = allocate_finite_field_domain(q2, verbose_level);
	
	vec_generators_affine_translations(gen1, 1 /* n */, field_q1, verbose_level);
	vec_generators_affine_translations(gen2, 1 /* n */, field_q2, verbose_level);

	int a, a1, a2, b, b1, b2;
	permutation perm_a1, perm_b1, perm_ab1;
	permutation perm_a2, perm_b2, perm_ab2;

	{
	with w(field_q1);
	a = finite_field_domain_primitive_root();

	if (f_v) {
		cout << "a primitive root in GF(" << q1 << ") is a=" << a << endl;
		}
	integer aa;
	aa.m_i(a);
	if (f_v) {
		cout << "a=" << aa << endl;
		}
	aa.power_int(u1);
	a1 = aa.s_i();
	if (f_v) {
		cout << "a^" << u1 << " = " << a1 << " = " << a1 << endl;
		}
	aa.m_i(a);
	aa.power_int(u2);
	a2 = aa.s_i();
	if (f_v) {
		cout << "a^" << u2 << " = " << a2 << " = " << a2 << endl;
		}
	
	discreta_matrix A;
	
	A.m_mn(1, 1);
	A.m_iji(0, 0, a1);
	A.AG_rep(field_q1, perm_a1, TRUE /* f_action_from_right */);
	if (f_vv) {
		cout << "as permutation: \nperm_a1=" << perm_a1 << endl;
		}
	A.m_iji(0, 0, a2);
	A.AG_rep(field_q1, perm_a2, TRUE /* f_action_from_right */);
	if (f_vv) {
		cout << "as permutation: \nperm_a2=" << perm_a2 << endl;
		}
	}
	{
	with w(field_q2);
	b = finite_field_domain_primitive_root();
	integer bb;
	
	if (f_v) {
		cout << "a primitive root in GF(" << q2 << ") is b=" << b << endl;
		}
	bb.m_i(b);
	bb.power_int(v1);
	b1 = bb.s_i();
	if (f_v) {
		cout << "b^" << v1 << " = " << b1 << " = " << b1 << endl;
		}
	bb.m_i(b);
	bb.power_int(v2);
	b2 = bb.s_i();
	if (f_v) {
		cout << "b^" << v2 << " = " << b2 << " = " << b2 << endl;
		}
	discreta_matrix A;
	
	A.m_mn(1, 1);
	A.m_iji(0, 0, b1);
	A.AG_rep(field_q2, perm_b1, TRUE /* f_action_from_right */);
	if (f_vv) {
		cout << "as permutation: \nperm_b1=" << perm_b1 << endl;
		}
	A.m_iji(0, 0, b2);
	A.AG_rep(field_q2, perm_b2, TRUE /* f_action_from_right */);
	if (f_vv) {
		cout << "as permutation: \nperm_b2=" << perm_b2 << endl;
		}
	}
	
	
	free_finite_field_domain(field_q1);
	free_finite_field_domain(field_q2);

	if (f_vv) {
		cout << "generators for translations in GF(" << q1 << "):\n";
		gen1.Print(cout);
		cout << "generators for translations in GF(" << q2 << "):\n";
		gen2.Print(cout);
		cout << endl;
		}
	
	vec_generators_direct_product(gen1, gen2, gen3);
	if (f_vv) {
		cout << "generators for direct product action:\n";
		gen3.Print(cout);
		cout << endl;
		}
	
	perm_ab1.cartesian_product_action(perm_a1, perm_b1);
	perm_ab2.cartesian_product_action(perm_a2, perm_b2);
	if (f_vv) {
		cout << "generators for direct product action perm_ab1:\n";
		cout << perm_ab1 << endl;
		cout << "generators for direct product action perm_ab2:\n";
		cout << perm_ab2 << endl;
		}
	
	gen3.append(perm_ab1);
	gen3.append(perm_ab2);

	G = gen3;
	if (f_write_generators_to_file) {
		int f_cyclic_notation = TRUE;
		write_file_of_generators_xml_group_label(G, label.s(), f_cyclic_notation);
		write_file_of_generators_group_label(G, label.s());
		write_file_of_generators_gap_group_label(G, label.s());
		}
	cout << "vec_generators_q1_q2_au1bv1_au2bv2() finished" << endl;
	//cout << "G=" << G << endl;
	cout << "label= " << label << endl;
}

void vec_generators_AGGL1q_subgroup(int q, int subgroup_index, 
	int f_special, int f_frobenius, int f_translations, int verbose_level)
{
	Vector gen;
	hollerith group_label;
	hollerith fname;
	
	group_label.init("A");
	if (f_special) {
		if (f_frobenius) {
			group_label.append("SS");
			}
		else {
			group_label.append("S");
			}
		}
	else {
		if (f_frobenius) {
			group_label.append("GG");
			}
		else {
			group_label.append("G");
			}
		}
	if (subgroup_index > 1) {
		group_label.append_i(subgroup_index);
		}
	group_label.append("L_1_");
	group_label.append_i(q);
	
	fname.init(group_label.s());
	fname.append(".xml");
	int f_cyclic_notation = TRUE;
	
	vec_generators_GL_n_q_subgroup_affine_representation(gen, 1, q, subgroup_index, 
		f_special, f_frobenius, f_translations, verbose_level);
	
	write_file_of_generators_xml(gen, fname.s(), f_cyclic_notation);
	
}

#if 0
void vec_generators_cycle_index(Vector &gen, Vector &C, int verbose_level)
{
	perm_group G(gen);
	
	cycle_index_perm_group(G, C, f_v, FALSE);
}
#endif

void vec_generators_singer_cycle_on_points_of_projective_plane(Vector &gen, int p, int f_modified, int verbose_level)
{
	gen.m_l(1);
	gen[0].change_to_permutation();
	gen[0].as_permutation().singer_cycle_on_points_of_projective_plane(p, f_modified, verbose_level);
}

}}
