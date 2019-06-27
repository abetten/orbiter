// unipoly.cpp
//
// Anton Betten
// 24.12.1999
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"

using namespace std;



namespace orbiter {
namespace discreta {


#undef CHANGE_KIND_VERBOSE
#undef COPY_VERBOSE


unipoly::unipoly() : Vector()
{
	k = UNIPOLY;
}

unipoly::unipoly(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "unipoly::copy constructor for object: " << const_cast<discreta_base &>(x) << "\n";
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

unipoly& unipoly::operator = (const discreta_base &x)
	// copy assignment
{
	//cout << "unipoly::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void unipoly::settype_unipoly()
{
	OBJECTSELF s;
	
	s = self;
	new(this) unipoly;
	self = s;
	k = UNIPOLY;
}

unipoly::~unipoly()
{
	freeself_unipoly();
}

void unipoly::freeself_unipoly()
{
	// cout << "unipoly::freeself_unipoly()\n";
	freeself_vector();
}

kind unipoly::s_virtual_kind()
{
	return UNIPOLY;
}

void unipoly::copyobject_to(discreta_base &x)
{
#ifdef COPY_VERBOSE
	cout << "unipoly::copyobject_to()\n";
	print_as_vector(cout);
#endif
	Vector::copyobject_to(x);
	x.as_unipoly().settype_unipoly();
#ifdef COPY_VERBOSE
	x.as_unipoly().print_as_vector(cout);
#endif
}

int my_unip_f_print_sub = FALSE;
int my_unip_f_use_variable_name = FALSE;
char my_unip_variable_name[128];


ostream& unipoly::print(ostream& ost)
{
	int d, i, f_print_k, k, f_prev = FALSE;
	discreta_base coef;
	const char *x, *y;
	int f_nothing_printed_at_all = TRUE;
	
	if (my_unip_f_use_variable_name)
		x = my_unip_variable_name;
	else
		x = "x";
	if (my_unip_f_print_sub)
		y = "_";
	else
		y = "^";
	d = degree();
	// ost << "(";
	for (i = d; i >= 0; i--) {
		coef = s_i(i);
		if (coef.s_kind() == INTEGER) {
			k = coef.s_i_i();
			if (k == 0) {
				if (i == 0 && f_nothing_printed_at_all) {
					ost << "0";
					}
				continue;
				}
			f_nothing_printed_at_all = FALSE;
			if (k < 0) {
				ost << " - ";
				}
			else if (f_prev) {
				ost << " + ";
				}
			if (k < 0)
				k = -k;
			if (k != 1 || (i == 0 && !my_unip_f_use_variable_name)) {
				ost << k;
				}
			}
		else {
			f_nothing_printed_at_all = FALSE;
			f_print_k = TRUE; 
			if (coef.is_one() && i > 0)
				f_print_k = FALSE;
			if (f_prev)
				ost << " + ";
			if (f_print_k) {
				ost << coef;
				} 
			}
		if (i == 0) {
			if (my_unip_f_use_variable_name) {
				ost << x;
				ost << y;
				ost << "0";
				}
			}
		else if (i == 1) {
			ost << x;
			if (my_unip_f_print_sub) {
				ost << y;
				ost << "1";
				}
			}
		else if (i > 1) {
			ost << x;
			ost << y;
			if (current_printing_mode() == printing_mode_latex) {
				if (i < 10) 
					ost << i;
				else
					ost << "{" << i << "}";
				}
			else {
				ost << i;
				}
			}
		f_prev = TRUE;
		}
	// ost << ")";
	return ost;
}

ostream& unipoly::print_as_vector(ostream& ost)
{
	// cout << "unipoly::print_as_vector()" << endl;
	return Vector::print(ost);
}

void unipoly::m_l(int l)
{
	// cout << "unipoly::m_l()\n";
	Vector::m_l_n(l);
	settype_unipoly();
}

int unipoly::degree()
{
	int l = s_l();
	int i;
	
	for (i = l - 1; i >= 0; i--) {
		if (!s_i(i).is_zero())
			return i;
		}
	return 0;
}

void unipoly::mult_to(discreta_base &x, discreta_base &y)
{
	unipoly& px = x.as_unipoly();
	unipoly py;
	discreta_base a;
	int d1, d2, d3, i, j, k;
	
	if (s_kind() != UNIPOLY) {
		cout << "unipoly::mult_to() this not a unipoly\n";
		exit(1);
		}
	if (x.s_kind() != UNIPOLY) {
		cout << "unipoly::mult_to() x is not a unipoly\n";
		exit(1);
		}
	d1 = degree();
	d2 = px.degree();
	d3 = d1 + d2;
	
	py.m_l(d3 + 1);
	a = s_i(0);
	a.zero();
	for (i = 0; i <= d3; i++) {
		py[i] = a;
		}
	for (i = 0; i <= d1; i++) {
		for (j = 0; j <= d2; j++) {
			k = i + j;
			a.mult(s_i(i), px.s_i(j));
			py[k] += a;
			}
		}
	py.swap(y);
}

void unipoly::add_to(discreta_base &x, discreta_base &y)
{
	unipoly& px = x.as_unipoly();
	unipoly py;
	discreta_base a;
	int d1, d2, d3, i;
	
	if (s_kind() != UNIPOLY) {
		cout << "unipoly::add_to() this not a unipoly\n";
		exit(1);
		}
	if (x.s_kind() != UNIPOLY) {
		cout << "unipoly::add_to() x is not a unipoly\n";
		exit(1);
		}
	d1 = degree();
	d2 = px.degree();
	d3 = MAXIMUM(d1, d2);
	
	py.m_l(d3 + 1);
	a = s_i(0);
	a.zero();
	for (i = 0; i <= d1; i++) {
		py[i] = s_i(i);
		}
	for (; i <= d3; i++) {
		py[i] = a;
		}
	for (i = 0; i <= d2; i++) {
		py[i] += px.s_i(i);
		}
	py.swap(y);
}

void unipoly::negate_to(discreta_base &x)
{
	unipoly px;
	int i, l;
	
	if (s_kind() != UNIPOLY) {
		cout << "unipoly::negate_to() this is not a unipoly\n";
		exit(1);
		}
	l = s_l();
	px.m_l(l);
	for (i = 0; i < l; i++) {
		s_i(i).negate_to(px.s_i(i));
		}
	x.swap(px);
}

void unipoly::one()
{
	m_l(1);
	s_i(0).one();
}

void unipoly::zero()
{
	m_l(1);
	s_i(0).zero();
}

void unipoly::x()
{
	m_l(2);
	s_i(0).zero();
	s_i(1).one();
}

void unipoly::x_to_the_i(int i)
{
	int j;
	
	m_l(i + 1);
	for (j = 0; j < i; j++) {
		s_i(j).zero();
		}
	s_i(i).one();
}

int unipoly::is_one()
{
	int d;
	
	d = degree();
	if (d > 0)
		return FALSE;
	return s_i(0).is_one();
}

int unipoly::is_zero()
{
	int d;
	
	d = degree();
	if (d > 0)
		return FALSE;
	return s_i(0).is_zero();
}

int unipoly::compare_with_euclidean(discreta_base &a)
{
	int d1, d2;
	unipoly &pa = a.as_unipoly();
	
	d1 = degree();
	d2 = pa.degree();
	if (d1 < d2)
		return -1;
	else if (d1 > d2)
		return 1;
	return 0;
}

void unipoly::integral_division(discreta_base &x,
		discreta_base &q, discreta_base &r, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int dm, dn, dq, i, j, ii, jj;
	discreta_base a, av, b, bav, c;
	unipoly &n = x.as_unipoly();
	unipoly qq, rr;
	
	if (f_v) {
		cout << "unipoly::integral_division" << endl;
		cout << "m=" << *this << endl;
		cout << "n=" << x << endl;
		}
	dm = degree();
	dn = n.degree();
	if (dn == 0) {
		if (n[0].is_zero()) {
			cout << "unipoly::integral_division(): division by zero" << endl;
			exit(1);
			}
		}
	if (dn > dm) {
		if (f_v) {
			cout << "unipoly::integral_division dn > dm, no division possible" << endl;
			}
		qq.zero();
		rr = *this;
		qq.swap(q);
		rr.swap(r);
		return;
		}
	dq = dm - dn;
	qq.m_l(dq + 1);
	rr = *this;
	// cout << "rr=" << rr << endl;
	a = n[dn];
	if (f_v) {
		cout << "unipoly::integral_division a=" << a << endl;
		}
	av = a;
	av.invert();
	if (f_v) {
		cout << "unipoly::integral_division av=" << av << endl;
		}
	for (i = dm, j = dq; i >= dn; i--, j--) {
		if (f_v) {
			cout << "unipoly::integral_division i=" << i << " j=" << j << endl;
			}

		b = rr[i];
		if (f_v) {
			cout << "unipoly::integral_division b=" << b << endl;
			cout << "unipoly::integral_division av=" << av << endl;
			cout << "unipoly::integral_division before mult" << endl;
			}

		bav.mult(b, av);
		qq[j] = bav;
		// cout << "i=" << i << " bav=" << bav << endl;
		for (ii = i, jj = dn; jj >= 0; ii--, jj--) {
			if (f_v) {
				cout << "unipoly::integral_division ii=" << ii << " jj=" << jj << endl;
				}
			c = n[jj];
			c *= bav;
			c.negate();
			rr[ii] += c; // rr[ii] -= bav * this[jj]
			if (ii == i && !rr[ii].is_zero()) {
				cout << "unipoly::integral_division(): ii == i && !rr[ii].is_zero()\n";
				exit(1);
				}
			// cout << "c=" << c << endl;
			// cout << "rr[ii]=" << rr[ii] << endl;
			}
		// cout << "i=" << i << " rr=" << rr << endl;
		}
	qq.swap(q);
	rr.swap(r);
	if (f_v) {
		cout << "q=" << q << endl;
		cout << "r=" << r << endl;
		}
}

void unipoly::derive()
{
	int i, d;
	discreta_base a;
	
	d = degree();
	for (i = 1; i <= d; i++) {
		a = s_i(i); // get the object type 
		a.homo_z(i);
		a *= s_i(i);
		s_i(i - 1) = a;
		}
	s_i(d).zero();
}

int unipoly::is_squarefree(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	unipoly a, u, v, g;
	int d;
	
	a = *this;
	a.derive();
	if (f_v) {
		cout << "unipoly::is_squarefree() derivative p' = " << a << endl;
		}
	extended_gcd(a, u, v, g, verbose_level - 2);
	if (f_v) {
		cout << "unipoly::is_squarefree() gcd(p, p') = " << g << endl;
		}
	d = g.degree();
	if (d >= 1)
		return FALSE;
	else
		return TRUE;
}

int unipoly::is_irreducible_GFp(int p, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	matrix B;
	domain d(p);
	with w(&d);
	int l, r;
	
	if (!is_squarefree(verbose_level))
		return FALSE;
	B.Berlekamp(*this, p, verbose_level - 1);
	l = B.s_m();
	r = B.rank();
	if (f_v) {
		cout << "has rank " << r << endl;
		}
	if (r == l - 1)
		return TRUE;
	else
		return FALSE;
}

int unipoly::is_irreducible(int q, int verbose_level)
{
	int f_v = (verbose_level = 1);
	matrix B;
	int l, r;
	
	if (!is_squarefree(verbose_level))
		return FALSE;
	B.Berlekamp(*this, q, verbose_level);
	l = B.s_m();
	r = B.rank();
	if (f_v) {
		cout << "has rank " << r << endl;
		}
	if (r == l - 1)
		return TRUE;
	else
		return FALSE;
}

int unipoly::is_primitive(int m, int p, Vector& vp, int verbose_level)
//Returns TRUE iff the polynomial $x$ has order $m$ 
//modulo the polynomial this (over GF(p)). 
//The prime factorization of $m$ must be given in vp (only the primes).
//A polynomial $a$ has order $m$ mod $q$ ($q = this$) iff 
//$a^m =1 mod q$ and $a^{m/p_i} \not= 1 mod q$ for all $p_i \mid m.$ 
//In this case, we have $a=x$ and we assume that $a^m = 1 mod q.$
{
	int l, i, m1;
	unipoly a;
	domain d(p);
	with w(&d);
	
	l = vp.s_l();
	for (i = 0; i < l; i++) {
		m1 = m / vp.s_ii(i);
		a.x();
		a.power_int_mod(m1, *this);
		if (a.is_one())
			return FALSE;
		}
	return TRUE;
}

void unipoly::numeric_polynomial(int n, int q)
{
	Vector v;
	int i, l;
	
	v.q_adic(n, q);
	l = v.s_l();
	m_l(l);
	for (i = 0; i < l; i++) {
		m_ii(i, v.s_ii(i));
		}
}

int unipoly::polynomial_numeric(int q)
{
	return q_adic_as_int(q);
}

void unipoly::singer_candidate(int p, int f, int b, int a)
{
	m_l(f + 1);
	s_i(f).one();
	s_i(f - 1).one();
	s_i(1).m_i_i(b);
	s_i(0).m_i_i(a);
}

void unipoly::Singer(int p, int f, int f_v, int f_vv)
{
	int m, i, a, b, low, high;
	Vector vp, ve;
	unipoly x;
	number_theory_domain NT;
	
	if (p <= 1) {
		cout << "unipoly::Singer(): p <= 1\n";
		exit(1);
		}
	if (!NT.is_prime(p)) {
		cout << "unipoly::Singer(): p not prime\n";
		exit(1);
		}
	m = NT.i_power_j(p, f) - 1;
	factor_integer(m, vp, ve);
	a = NT.primitive_root(p, f_v);
	for (b = 0; b < p; b++) {
		x.singer_candidate(p, f, b, a);
		if (f_v) {
			cout << "singer candidate " << x << endl;
			}
		if (x.is_irreducible_GFp(p, f_vv) && x.is_primitive(m, p, vp, f_vv)) {
			if (f_v) {
				cout << "OK" << endl;
				}
			swap(x);
			return;
			}
		}
	low = m + 1;
	high = low << 1;
	for (i = low; i <= high; i++) {
		x.numeric_polynomial(i, p);
		if (f_v) {
			cout << "candidate " << i - low + 1 << " : " << x << endl;
			}
		if (x.is_irreducible_GFp(p, f_vv) && x.is_primitive(m, p, vp, f_vv)) {
			if (f_v) {
				cout << "OK" << endl;
				}
			swap(x);
			return;
			}
		}
	cout << "ERROR: did not find an irrducible primitive polynomial, help!\n";
	exit(1);
}

void unipoly::get_an_irreducible_polynomial(int f, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int low, high, q, i;
	domain *d;
	unipoly x;
	number_theory_domain NT;
	
	if (f_v) {
		cout << "unipoly::get_an_irreducible_polynomial" << endl;
		}
	if (!is_finite_field_domain(d)) {
		cout << "unipoly::get_an_irreducible_polynomial() no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	if (f_v) {
		cout << "unipoly::get_an_irreducible_polynomial() \n"
			"searching for an irreducible polynomial of degree " << f <<
			" over GF(" << q << ")" << endl;
		}
	low = NT.i_power_j(q, f);
	high = low << 1; // only monic polynomials 
	for (i = low; i <= high; i++) {
		x.numeric_polynomial(i, q);
		if (f_vv) {
			cout << "candidate " << i - low + 1 << " : " << x << endl;
			}
		if (x.is_irreducible(q, verbose_level - 2)) {
			if (f_vv) {
				cout << "is irreducible" << endl;
				}
			if (f_v) {
				cout << "unipoly::get_an_irreducible_polynomial() " << x << endl;
				}
			swap(x);
			return;
			}
		}
	cout << "unipoly::get_an_irreducible_polynomial() no polynomial found" << endl;
	exit(1);
}

void unipoly::evaluate_at(discreta_base& x, discreta_base& y)
{
	int i, d;
	discreta_base z;
	
	d = degree();
	z = s_i(d);
	for (i = d - 1; i >= 0; i--) {
		z *= x;
		z += s_i(i);
		}
	y = z;
}

void unipoly::largest_divisor_prime_to(unipoly& q, unipoly& r)
//computes the monic polynomial $r$ with ($p$ is the polynomial in this)
//\begin{enumerate}
//\item
//$r \mid p$
//\item
//$\gcd(r,q) = 1$ and
//\item
//each irreducible polynomial dividing $p/r$ divides $q$.
//Lueneburg~\cite{Lueneburg87a}, p. 37.
//\end{enumerate}
//In other words, $r$ is the maximal divisor of $p$ which is prime to $q$.
{
	unipoly rr, g, u, v;
	//int d;
	
	//d = degree();
	r = *this;
	r.extended_gcd(q, u, v, g, 0);
	while (g.degree()) {
		r.integral_division_exact(g, rr);
		r.swap(rr);
		r.extended_gcd(q, u, v, g, 0);
		}
	r.monic();
}

void unipoly::monic()
{
	int d, i;
	discreta_base a;
	
	d = degree();
	a = s_i(d);
	if (a.is_one())
		return;
	a.invert();
	for (i = 0; i < d; i++) {
		s_i(i) *= a;
		}
}

void unipoly::normal_base(int p, matrix& F, matrix& N, int verbose_level)
// compare Lueneburg~\cite{Lueneburg87a} p. 106.
{
	int f_v = (verbose_level >= 1);
	domain d(p);
	with ww(&d);
	int i, f;
	Vector v, V;
	unipoly x, mue;
	
	f = degree();
	F.Frobenius(*this, p, FALSE);
	if (f_v) {
		cout << "unipoly::normal_base(): Frobenius:\n" << F << endl;
		}
	F.KX_cyclic_module_generator(v, mue, verbose_level - 1);
	V.m_l(f);
	V[0] = v;
	x.x();
	if (f_v) {
		cout << "unipoly::normal_base(): V[0]=" << v << endl;
		}
	for (i = 1; i < f; i++) {
		F.KX_module_apply(x, v);
		V[i] = v;
		if (f_v) {
			cout << "unipoly::normal_base(): V["<<i<<"]=" << v << endl;
			}
		}
	N.from_vector_of_columns(V);
	if (f_v) {
		cout << "unipoly::normal_base(): N=\n" << N << endl;
		}
	
#if 0
	matrix T, P, Pv, Q, Qv;
	integer a1;
	Vector v, vv, b, bb;

	T = F;
	T.elements_to_unipoly();
	T.minus_X_times_id();
	if (f_v) {
		cout << "F - x * Id=\n" << T << endl;
		}
	T.smith_normal_form(P, Pv, Q, Qv, f_vv, FALSE);
	
	if (f_v) {
		cout << "Q=\n" << Q << endl;
		}
	a1.m_i_i(1);
	Q.evaluate_at(a1);
	if (f_v) {
		cout << "Q(1)=\n" << Q << endl;
		cout << "F=\n" << F << endl;
		}
	Q.to_vector_of_columns(v);
	b = v.s_i(d - 1);
	vv.m_l(d);
	vv[1] = b;
	if (f_v) {
		cout << "N[0]=" << b << endl;
		}
	for (i = 1; i < d; i++) {
		bb.mult(F, b);
		b = bb;
		vv[i] = b;
		if (f_v) {
			cout << "N[" << i << "]=" << b << endl;
			}
		}
	N.from_vector_of_columns(vv);
	if (f_v) {
		cout << "N=" << N << endl;
		}
#endif

}

int unipoly::first_irreducible_polynomial(int p, unipoly& m, matrix& F, matrix& N, Vector &v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	domain d(p);
	with ww(&d);
	Vector w;
	unipoly a;
	int f, i;
	
	f = F.s_m();
	v.first_regular_word(f, p);
	if (!v.next_regular_word(p))
		return FALSE;
	
	if (f_v) {
		cout << "regular word:" << v << endl;
		}
	w.mult(N, v);
	a.m_l(f);
	for (i = 0; i < f; i++) {
		a[i] = w[i];
		}
	F.KX_module_minpol(a, m, *this, verbose_level - 1);
	return TRUE;
}

int unipoly::next_irreducible_polynomial(int p, unipoly& m, matrix& F, matrix& N, Vector &v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	domain d(p);
	with ww(&d);
	Vector w;
	unipoly a;
	int f, i;
	
	f = F.s_m();
	if (!v.next_regular_word(p))
		return FALSE;

	if (f_v) {
		cout << "regular word:" << v << endl;
		}
	w.mult(N, v);
	a.m_l(f);
	for (i = 0; i < f; i++) {
		a[i] = w[i];
		}
	F.KX_module_minpol(a, m, *this, verbose_level - 1);
	return TRUE;
}

void unipoly::normalize(discreta_base &p)
{
	if (p.s_kind() != UNIPOLY) {
		cout << "unipoly::normalize() p not an UNIPOLY" << endl;
		exit(1);
		}
	unipoly p1 = p.as_unipoly();
	unipoly q, r;
	
	integral_division(p1, q, r, 0); 
	swap(r);
}


void unipoly::Xnm1(int n)
{
	m_l_n(n + 1);
	s_i(n).one();
	s_i(0).one();
	s_i(0).negate();
}


static int multiply(Vector & vp, Vector & ve);

void unipoly::Phi(int n, int f_v)
{
	Vector vp, ve, vd;
	int i, j, l, mu, d, nd;
	unipoly p, q, r;
	
	if (f_v) {
		cout << "Phi(" << n << "):" << endl;
		}
	factor_integer(n, vp, ve);
	l = vp.s_l();
	vd.m_l_n(l);
	p.m_l(1);
	q.m_l(1);
	p.s_i(0).one();
	q.s_i(0).one();
	
	while (TRUE) {
		d = multiply(vp, vd);
		nd = n / d;
		mu = Moebius(nd);
		r.Xnm1(d);
		if (f_v) {
			cout << "d=" << d << " mu(d)=" << mu << " r=" << r << endl;
			cout << "p=" << p << endl;
			cout << "q=" << q << endl;
			}
		if (mu == 1) {
			p *= r;
			}
		else if (mu == -1) {
			q *= r;
			}
		
		for (i = 0; i < l; i++) {
			j = vd.s_ii(i);
			if (j < ve.s_ii(i)) {
				vd.m_ii(i, j + 1);
				break;
				}
			vd.m_ii(i, 0);
			}
		if (i == l)
			break;
		}
	p.integral_division_exact(q, *this);
}

static int multiply(Vector & vp, Vector & ve)
{
	int i, l, n, m;
	number_theory_domain NT;
	
	n = 1;
	l = vp.s_l();
	for (i = 0; i < l; i++) {
		m = NT.i_power_j(vp.s_ii(i), ve.s_ii(i));
		n *= m;
		}
	return n;
}


void unipoly::weight_enumerator_MDS_code(int n, int k, int q, int f_v, int f_vv, int f_vvv)
{
	int j, h, l;
	discreta_base a, b, c, d, e;
	
	m_l_n(n + 1);
	e.m_i_i(-1);
	s_i(0).m_i_i(1);
	for (j = n - k + 1; j <= n; j++) {
		l = k - n + j - 1;
		a.change_to_integer();
		b.change_to_integer();
		c.change_to_integer();
		d.change_to_integer();
		if (f_vvv) {
			cout << "j=" << j << endl;
			}
		Binomial(n, j, a);
		if (f_vvv) {
			cout << " \\binom{" << n << "}{" << j << "}=a=" << a << endl;
			}
		b.m_i_i(0);
		for (h = 0; h <= l; h++) {
			if (f_vvv) {
				cout << " h=" << h;
				}
			Binomial(j, h, c);
			if (f_vvv) {
				cout << "  {j \\choose h} = c=" << c << endl;
				}
			d.m_i_i(q);
			d.power_int(l - h + 1);
			if (f_vvv) {
				cout << "  q^" << l - h + 1 << "=" << d << endl;
				}
			d += e;
			if (f_vvv) {
				cout << "  minus 1, d=" << d << endl;
				}
			c *= d;
			if (ODD(h)) {
				c.negate();
				}
			if (f_vvv) {
				cout << "  c=" << c << endl;
				}
			b += c;
			if (f_vvv) {
				cout << "  new b=" << b << endl;
				}
			}
		b *= a;
		if (f_vv) {
			cout << " coeff=" << b << endl;
			}
		s_i(j) = b;
		}
	if (f_v) {
		cout << "unipoly::weight_enumerator_MDS_code() n=" << n << " k=" << k << " q=" << q << endl;
		cout << *this << endl;
		}
}

void unipoly::charpoly(int q, int size, int *mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	matrix M;
	int i, j, k, a, p, h;
	finite_field Fq;
	//unipoly_domain U;
	//unipoly_object char_poly;
	number_theory_domain NT;

	if (f_v) {
		cout << "unipoly::charpoly" << endl;
		}
	M.m_mn(size, size);
	k = 0;
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a = mtx[k++];
			M.m_iji(i, j, a);
			}
		}
	if (f_vv) {
		cout << "M=" << endl;
		cout << M << endl;
		}

	if (!NT.is_prime_power(q, p, h)) {
		cout << "q is not prime, we need a prime" << endl;
		exit(1);
		}
	Fq.init(q, verbose_level - 1);

	domain d(q);
	with w(&d);

#if 0

	matrix M2;
	M2 = M;
	for (i = 0; i < size; i++) {
		unipoly mue;
		M2.KX_module_order_ideal(i, mue, verbose_level - 1);
		cout << "order ideal " << i << ":" << endl;
		cout << mue << endl;
		}
#endif
	
	matrix M1, P, Pv, Q, Qv, S, T;
	
	M.elements_to_unipoly();
	M.minus_X_times_id();
	M1 = M;
	if (f_vv) {
		cout << "M - x * Id=\n" << M << endl;
		}
	M.smith_normal_form(P, Pv, Q, Qv, verbose_level - 2);

	if (f_vv) {
		cout << "the Smith normal form is:" << endl;
		cout << M << endl;
		}

	S.mult(P, Pv);
	if (f_vv) {
		cout << "P * Pv=\n" << S << endl;
		}

	S.mult(Q, Qv);
	if (f_vv) {
		cout << "Q * Qv=\n" << S << endl;
		}

	S.mult(P, M1);
	if (f_vv) {
		cout << "T.mult(S, Q):\n";
		}
	T.mult(S, Q);
	if (f_vv) {
		cout << "T=\n" << T << endl;
		}


	unipoly charpoly;
	int deg;
	int l, lv, b, c;

	charpoly = M.s_ij(size - 1, size - 1);
	
	if (f_vv) {
		cout << "characteristic polynomial:" << charpoly << endl;
		}
	deg = charpoly.degree();
	if (f_vv) {
		cout << "has degree " << deg << endl;
		}
	l = charpoly.s_ii(deg);
	if (f_vv) {
		cout << "leading coefficient " << l << endl;
		}
	lv = Fq.inverse(l);
	if (f_vv) {
		cout << "leading coefficient inverse " << lv << endl;
		}
	for (i = 0; i <= deg; i++) {
		b = charpoly.s_ii(i);
		c = Fq.mult(b, lv);
		charpoly.m_ii(i, c);
		}
	if (f_v) {
		cout << "monic characteristic polynomial:" << charpoly << endl;
		}
	swap(charpoly);
}

}}


