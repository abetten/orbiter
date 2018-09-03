// design_parameter.C
//
// Anton Betten
// 18.09.2000
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"

design_parameter::design_parameter() : Vector()
{
	k = DESIGN_PARAMETER;
}

design_parameter::design_parameter(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "design_parameter::copy constructor for object: " << const_cast<discreta_base &>(x) << "\n";
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

design_parameter& design_parameter::operator = (const discreta_base &x)
	// copy assignment
{
	cout << "design_parameter::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void design_parameter::settype_design_parameter()
{
	OBJECTSELF s;
	
	s = self;
	new(this) design_parameter;
	self = s;
	k = DESIGN_PARAMETER;
}

design_parameter::~design_parameter()
{
	freeself_design_parameter();
}

void design_parameter::freeself_design_parameter()
{
	// cout << "group_selection::freeself_design_parameter()\n";
	freeself_vector();
}

kind design_parameter::s_virtual_kind()
{
	return DESIGN_PARAMETER;
}

void design_parameter::copyobject_to(discreta_base &x)
{
#ifdef COPY_VERBOSE
	cout << "design_parameter::copyobject_to()\n";
	print_as_vector(cout);
#endif
	Vector::copyobject_to(x);
	x.as_design_parameter().settype_design_parameter();
#ifdef COPY_VERBOSE
	x.as_design_parameter().print_as_vector(cout);
#endif
}

ostream& design_parameter::print(ostream& ost)
{
	hollerith h;
	
	text(h);
	ost << h.s();
	return ost;
}

void design_parameter::init()
{
	m_l_n(6);
	c_kind(DESIGN_PARAMETER);
	id() = -1;
	t() = 0;
	v() = 0;
	K() = 0;
	lambda().m_i_i(0);
	s_i(5).change_to_vector();
	source().m_l(0);
}

void design_parameter::init(int t, int v, int k, int lambda)
{
	init();
	design_parameter::t() = t;
	design_parameter::v() = v;
	K() = k;
	design_parameter::lambda().m_i_i(lambda);
}

void design_parameter::init(int t, int v, int k, discreta_base& lambda)
{
	init();
	design_parameter::t() = t;
	design_parameter::v() = v;
	K() = k;
	design_parameter::lambda() = lambda;
}

void design_parameter::text(hollerith& h)
{
	hollerith hh;
	
	h.init("#");
	h.append_i(id());
	h.append(" ");
	
	text_parameter(hh);
	h.append(hh.s());
	h.append(" ");
	
	if (is_selfsupplementary()) {
		h.append("(selfsupplementary) ");
		}
	else {
		discreta_base ls;
		lambda_of_supplementary(ls);
		ls.print_to_hollerith(hh);
		
		h.append("(supplementary design has lambda=");
		h.append(hh.s());
		h.append(")");
		}
	for (int i = 0; i < source().s_l(); i++) {
		hollerith hh;
		
		source_i(i).text2(*this, hh);
		h.append("; ");
		h.append(hh.s());
		}
}

void design_parameter::text_parameter(hollerith& h)
{
	hollerith hh;
	
	h.init("");
	h.append_i(t());
	h.append("-(");
	h.append_i(v());
	h.append(",");
	h.append_i(K());
	h.append(",");
	
	lambda().print_to_hollerith(hh);
	h.append(hh.s());
	
	h.append(") ");
}

void design_parameter::reduced_t(design_parameter& p)
{
	discreta_base lambda_new;
	integer a, b;
	design_parameter_source S;

	// lambda_new = (lambda * (v - t + 1)) / (k - t + 1);
	a.m_i(v() - t() + 1);
	b.m_i(K() - t() + 1);
	lambda_new.mult(lambda(), a);
	lambda_new.divide_by_exact(b);
	
	p.init();
	p.v() = v();
	p.t() = t() - 1;
	p.K() = K();
	p.lambda() = lambda_new;
	S.init();
	S.prev() = id();
	S.rule() = rule_reduced_t;
	p.source().append(S);
}

int design_parameter::increased_t(design_parameter& p)
{
	discreta_base lambda_new;
	integer a, b, q, r;
	
	// lambda_new = (lambda * (k - t)) / (v - t);
	a.m_i(K() - t());
	b.m_i(v() - t());
	if (a.is_zero())
		return FALSE;
	if (b.is_zero())
		return FALSE;
	lambda_new.mult(lambda(), a);
	lambda_new.integral_division(b, q, r, 0);
	lambda_new.swap(q);
	if (!r.is_zero())
		return FALSE;
	p.init();
	p.v() = v();
	p.t() = t() + 1;
	p.K() = K();
	p.lambda() = lambda_new;
	return TRUE;
}

void design_parameter::supplementary_reduced_t(design_parameter& p)
{
	discreta_base lambda_new;
	integer a, b;
	design_parameter_source S;
	design_parameter q;

	supplementary(q);
	// lambda_new = (lambda * (v - t + 1)) / (k - t + 1);
	a.m_i(q.v() - q.t() + 1);
	b.m_i(q.K() - q.t() + 1);
	lambda_new.mult(q.lambda(), a);
	lambda_new.divide_by_exact(b);
	
	p.init();
	p.v() = q.v();
	p.t() = q.t() - 1;
	p.K() = q.K();
	p.lambda() = lambda_new;
	S.init();
	S.prev() = id();
	S.rule() = rule_supplementary_reduced_t;
	p.source().append(S);
}

void design_parameter::derived(design_parameter& p)
{
	design_parameter_source S;

	p.init();
	p.v() = v() - 1;
	p.t() = t() - 1;
	p.K() = K() - 1;
	p.lambda() = lambda();
	S.init();
	S.prev() = id();
	S.rule() = rule_derived;
	p.source().append(S);
}

int design_parameter::derived_inverse(design_parameter& p)
{
	design_parameter p1;
	
	p1.init();
	p1.v() = v() + 1;
	p1.t() = t() + 1;
	p1.K() = K() + 1;
	p1.lambda() = lambda();
	if (!design_parameters_admissible(p1.v(), p1.t(), p1.K(), p1.lambda()))
		return FALSE;
	p1.swap(p);
	return TRUE;
}

void design_parameter::supplementary_derived(design_parameter& p)
{
	design_parameter_source S;
	design_parameter q;

	supplementary(q);
	p.init();
	p.v() = q.v() - 1;
	p.t() = q.t() - 1;
	p.K() = q.K() - 1;
	p.lambda() = q.lambda();
	S.init();
	S.prev() = id();
	S.rule() = rule_supplementary_derived;
	p.source().append(S);
}

void design_parameter::residual(design_parameter& p)
{
	discreta_base lambda_new;
	integer a, b, c;
	design_parameter_source S;

	// lambda_new = (lambda * (v - t + 1)) / (k - t + 1) - lambda;
	a.m_i(v() - t() + 1);
	b.m_i(K() - t() + 1);
	c = lambda();
	c.negate();
	lambda_new.mult(lambda(), a);
	lambda_new.divide_by_exact(b);
	lambda_new += c;
	
	p.init();
	p.v() = v() - 1;
	p.t() = t() - 1;
	p.K() = K();
	p.lambda() = lambda_new;
	S.init();
	S.prev() = id();
	S.rule() = rule_residual;
	p.source().append(S);
}

int design_parameter::residual_inverse(design_parameter& p)
{
	discreta_base a, a1, b, q, r;
	design_parameter p1;
	
	p1.init();
	p1.v() = v() + 1;
	p1.t() = t() + 1;
	p1.K() = K();
	
	a1.m_i_i(K() - t());
	a.mult(lambda(), a1);
	b.m_i_i(v() + 1 - K());
	a.integral_division(b, q, r, 0);
	if (!r.is_zero())
		return FALSE;
	p1.lambda() = q;
	if (!design_parameters_admissible(p1.v(), p1.t(), p1.K(), p1.lambda()))
		return FALSE;
	p1.swap(p);
	return TRUE;
}

void design_parameter::ancestor(design_parameter& p, Vector & path, int f_v, int f_vv)
{
	design_parameter s, q;
	int f_special = FALSE;
	
	path.m_l_n(3);
	s = *this;
	if (f_v) {
		cout << "determining ancestor of " << s << endl;
		}
	if (s.lambda().is_one()) {
		if (s.t() + 1 == s.K()) {
			if (is_prime(s.v() - s.t())) {
				cout << "determining ancestor of steiner system " << s.t() << "(" << s.v() << "," << s.K() << "1) with v - t prime" << endl;
				f_special = TRUE;
				}
			}
		}
	if (f_special) {
		int v, t, k, n;
		
		v = s.v();
		t = s.t();
		k = s.K();
		n = v - 2 * t - 2;
		s.swap(p);
		p.t() = t + n;
		p.v() = v + n;
		p.K() = k + n;
		path.m_ii(1, n);
		}
	else {
		while (s.increased_t(q)) {
			if (f_vv) {
				cout << "ancestor, increasing t to " << q << endl;
				}
			s.swap(q);
			path.s_ii(0)++;
			}
		while (s.derived_inverse(q)) {
			if (f_vv) {
				cout << "ancestor, derived_inverse gives " << q << endl;
				}
			s.swap(q);
			path.s_ii(1)++;
			}
		while (s.residual_inverse(q)) {
			if (f_vv) {
				cout << "ancestor, residual_inverse gives " << q << endl;
				}
			s.swap(q);
			path.s_ii(2)++;
			}
		s.swap(p);
		}
	if (f_v) {
		cout << "ancestor of " << *this << endl << "is " << p << " " 
			<< path.s_ii(0) << " times reduced t, " 
			<< path.s_ii(1) << " times derived, " 
			<< path.s_ii(2) << " times residual." << endl;
		}
}

void design_parameter::supplementary_residual(design_parameter& p)
{
	discreta_base lambda_new;
	integer a, b, c;
	design_parameter_source S;
	design_parameter q;

	supplementary(q);
	// lambda_new = (lambda * (v - t + 1)) / (k - t + 1) - lambda;
	a.m_i(q.v() - q.t() + 1);
	b.m_i(q.K() - q.t() + 1);
	c = q.lambda();
	c.negate();
	lambda_new.mult(q.lambda(), a);
	lambda_new.divide_by_exact(b);
	lambda_new += c;
	
	p.init();
	p.v() = q.v() - 1;
	p.t() = q.t() - 1;
	p.K() = q.K();
	p.lambda() = lambda_new;
	S.init();
	S.prev() = id();
	S.rule() = rule_supplementary_residual;
	p.source().append(S);
}

int design_parameter::trung_complementary(design_parameter& p)
{
	discreta_base lambda_new;
	integer a, b;
	design_parameter_source S;

	if (v() != 2 * K() + 1)
		return FALSE;
	
	// lambda_new = (lambda * (2 * k + 2 - t)) / (k + 1 - t);
	a.m_i(2 * K() + 2 - t());
	b.m_i(K() - t() + 1);
	lambda_new.mult(lambda(), a);
	lambda_new.divide_by_exact(b);
	
	p.init();
	p.v() = v() + 1;
	p.t() = t();
	p.K() = K() + 1;
	p.lambda() = lambda_new;
	S.init();
	S.prev() = id();
	S.rule() = rule_trung_complementary;
	p.source().append(S);
	return TRUE;
}

int design_parameter::trung_left_partner(int& t1, int& v1, int& k1, discreta_base& lambda1,
	int& t_new, int& v_new, int& k_new, discreta_base& lambda_new)
{
	discreta_base a, q, r;
	integer b, c;
	
	c.m_i(K() - t());
	a.mult(lambda(), c);
	b.m_i(v() - K() + 1);
	if (b.is_zero())
		return FALSE;
	a.integral_division(b, q, r, 0);
	if (!r.is_zero())
		return FALSE;
	t1 = t();
	v1 = v();
	k1 = K() - 1;
	lambda1 = q;
	t_new = t();
	v_new = v() + 1;
	k_new = K();
	lambda_new.add(lambda(), lambda1);
	return TRUE;
}

int design_parameter::trung_right_partner(int& t1, int& v1, int& k1, discreta_base& lambda1,
	int& t_new, int& v_new, int& k_new, discreta_base& lambda_new)
{
	discreta_base a, q, r;
	integer b, c;
	
	c.m_i(v() - K());
	a.mult(lambda(), c);
	b.m_i(K() + 1 - t());
	if (b.is_zero())
		return FALSE;
	a.integral_division(b, q, r, 0);
	if (!r.is_zero())
		return FALSE;
	t1 = t();
	v1 = v();
	k1 = K() + 1;
	lambda1 = q;
	t_new = t();
	v_new = v() + 1;
	k_new = K() + 1;
	lambda_new.add(lambda(), lambda1);
	return TRUE;
}

int design_parameter::alltop(design_parameter& p)
// Alltop~\cite{Alltop75}.
// returns TRUE iff alltop could be applied;
// in this case, p contains the new design parameter set.
{
	design_parameter_source S;

	if (v() == 2 * K() + 1 && EVEN(t())) {
		p.init();
		p.v() = v() + 1;
		p.t() = t() + 1;
		p.K() = K() + 1;
		p.lambda() = lambda();
		S.init();
		S.prev() = id();
		S.rule() = rule_alltop;
		p.source().append(S);
		return TRUE;
		}
	if (v() == 2 * K() + 1 && ODD(t())) {
		discreta_base lmax, lmax_half, two, r;
		
		design_lambda_max(t(), v(), K(), lmax);
		two.m_i_i(2);
		lmax.integral_division(two, lmax_half, r, 0);
		r = lambda();
		r.negate();
		lmax_half += r;
		
		if (lmax_half.is_zero()) {
			p.init();
			p.v() = v() + 1;
			p.t() = t() + 1;
			p.K() = K() + 1;
			p.lambda() = lambda();
			S.init();
			S.prev() = id();
			S.rule() = rule_alltop;
			p.source().append(S);
			return TRUE;
			}
		}
	return FALSE;
}

void design_parameter::complementary(design_parameter& p)
{
	discreta_base lambda_new;
	design_parameter_source S;

	design_lambda_ijs(t(), v(), K(), lambda(), 1 /* s */, 0 /* i */, t() /* j */, lambda_new);
	
	p.init();
	p.v() = v();
	p.t() = t();
	p.K() = v() - K();
	p.lambda() = lambda_new;
	S.init();
	S.prev() = id();
	S.rule() = rule_complementary;
	p.source().append(S);
}

void design_parameter::supplementary(design_parameter& p)
{
	discreta_base lambda_new, a;
	design_parameter_source S;
	int num, denom, n, d, g, i;
	
	num = 1;
	denom = 1;
	n = v() - t();
	d = K() - t();
	for (i = 0; i < K() - t(); i++) {
		num *= n;
		denom *= d;
		n--;
		d--;
		g = gcd_int(num, denom);
		if (g != 1 && g != -1) {
			num /= g;
			denom /= g;
			}
		}
	if (denom != 1) {
		cout << "design_parameter::supplementary() error: denom != 1" << endl;
		exit(1);
		}
	lambda_new.m_i_i(num);

	a = lambda();
	a.negate();
	lambda_new += a;
	
	p.init();
	p.v() = v();
	p.t() = t();
	p.K() = K();
	p.lambda() = lambda_new;
	S.init();
	S.prev() = id();
	S.rule() = rule_supplementary;
	p.source().append(S);
}

int design_parameter::is_selfsupplementary()
{
	design_parameter q;
	discreta_base a, b, c;
	
	supplementary(q);
	a = q.lambda();
	b = lambda();
	b.negate();
	c.add(a, b);
	if (c.is_zero())
		return TRUE;
	else
		return FALSE;
}

void design_parameter::lambda_of_supplementary(discreta_base& lambda_supplementary)
{
	design_parameter q;

	supplementary(q);
	lambda_supplementary = q.lambda();
}

void design_parameter::init_database(database& D, char *path)
// path including trailing slash
/* 
 * btree #0: id, v, t, k, lambda
 * btree #1: v, t, k, lambda, id
 * btree #2: t, v, k, lambda, id
 * btree #3: lambda, v, t, k, id
 * btree #4: k, id
 */
{
	btree B;
	hollerith hh, h0, h1, h2, h3, h4;
	int f_compress = TRUE;
	int f_duplicatekeys = TRUE;
	
	hh.init(path);
	h0.init(path);
	h1.init(path);
	h2.init(path);
	h3.init(path);
	h4.init(path);
	hh.append("design_parameters.db");
	h0.append("design_parameters0.idx");
	h1.append("design_parameters1.idx");
	h2.append("design_parameters2.idx");
	h3.append("design_parameters3.idx");
	h4.append("design_parameters4.idx");
	
	int idx_id = 0;
	int idx_t = 1;
	int idx_v = 2;
	int idx_k = 3;
	int idx_lambda = 4;
	
	
	D.init(hh.s(), DESIGN_PARAMETER, f_compress);
	
	B.init(h0.s(), f_duplicatekeys, 0 /* btree_idx */);
	B.add_key_int4(idx_id, 0);
	B.add_key_int4(idx_v, 0);
	B.add_key_int4(idx_t, 0);
	B.add_key_int4(idx_k, 0);
	B.add_key_int4(idx_lambda, 0);
	D.btree_access().append(B);

	B.init(h1.s(), f_duplicatekeys, 1 /* btree_idx */);
	B.add_key_int4(idx_v, 0);
	B.add_key_int4(idx_t, 0);
	B.add_key_int4(idx_k, 0);
	B.add_key_int4(idx_lambda, 0);
	B.add_key_int4(idx_id, 0);
	D.btree_access().append(B);

	B.init(h2.s(), f_duplicatekeys, 2 /* btree_idx */);
	B.add_key_int4(idx_t, 0);
	B.add_key_int4(idx_v, 0);
	B.add_key_int4(idx_k, 0);
	B.add_key_int4(idx_lambda, 0);
	B.add_key_int4(idx_id, 0);
	D.btree_access().append(B);

	B.init(h3.s(), f_duplicatekeys, 3 /* btree_idx */);
	B.add_key_int4(idx_lambda, 0);
	B.add_key_int4(idx_v, 0);
	B.add_key_int4(idx_t, 0);
	B.add_key_int4(idx_k, 0);
	B.add_key_int4(idx_id, 0);
	D.btree_access().append(B);

	B.init(h4.s(), f_duplicatekeys, 4 /* btree_idx */);
	B.add_key_int4(idx_k, 0);
	B.add_key_int4(idx_id, 0);
	D.btree_access().append(B);
}

