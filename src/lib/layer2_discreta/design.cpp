// design.cpp
//
// Anton Betten
// 18.09.2000
// moved from D2 to ORBI Nov 15, 2007

#include "../layer2_discreta/discreta.h"
#include "layer1_foundations/foundations.h"

using namespace std;


namespace orbiter {
namespace layer2_discreta {
namespace typed_objects {


static void prepare_entry(
		Vector &entry, int i, int j,
		int h, int t, int v, int k, int lambda, int verbose_level);
static void determine_minimal_and_maximal_path(
		Vector &v,
		Vector & min_path, Vector & max_path, int & max_depth);
static void determine_dominating_ancestor(
		int t, int v, int k,
		discreta_base & lambda, Vector & path,
		design_parameter &dominating_ancestor, int verbose_level);
static void reduce_path(
		Vector &cmp, Vector &min_path);
static void family_report(
		database & D, ostream& fhtml,
		ostream &ftex, int t, int v, int k, discreta_base &lambda,
		Vector & cm, Vector & cmp, int minimal_t, int verbose_level);
static void f_m_j(
		int m, int j, discreta_base &a, int verbose_level);
static int max_m(
		int i, int j, int verbose_level);


int design_parameters_admissible(
		int v, int t, int k, discreta_base & lambda)
{
	int delta_lambda = calc_delta_lambda(v, t, k, false);
	discreta_base b, q, r;
	
	b.m_i_i(delta_lambda);
	lambda.integral_division(b, q, r, 0);
	if (!r.is_zero())
		return false;
	
	discreta_base lambda_max;
	design_lambda_max(t, v, k, lambda_max);
	if (lambda.gt(lambda_max))
		return false;
	return true;
}

int calc_delta_lambda(
		int v, int t, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	discreta_base lambda;
	int i;
	discreta_base a, b, a1, b1, g, rhs_a, rhs_b, delta_lambda, dl, a2, b2, gg;
	
	// f_v = true;

	lambda.m_i_i(1);
	if (f_v) {
		cout << "calc_delta_lambda: v=" << v << " t=" << t << " k="
				<< k << " lambda=" << lambda << endl;
		}
	for (i = t; i >= 0; i--) {
		if (i == t) {
			rhs_a = lambda;
			rhs_b.m_i_i(1);
			delta_lambda.m_i_i(1);
			}
		else {
			a1.m_i_i(v - i);
			b1.m_i_i(k - i);
			a.mult(rhs_a, a1, verbose_level);
			b.mult(rhs_b, b1, verbose_level);
			a.extended_gcd(b, a2, b2, g, 0);
			a.divide_by_exact(g, verbose_level);
			b.divide_by_exact(g, verbose_level);
			delta_lambda.extended_gcd(b, a2, b2, gg, 0);
			b1 = b;
			b1.divide_by_exact(gg, verbose_level);
			dl.mult(delta_lambda, b1, verbose_level);
			delta_lambda = dl;
			if (f_v) {
				cout << "t'=" << i << " lambda'=" << a << "/" << b
						<< " delta_lambda=" << delta_lambda << endl;
				}
			rhs_a = a;
			rhs_b = b;
			}
		}
	if (delta_lambda.s_kind() == INTEGER)
		return delta_lambda.s_i_i();
	else {
		cout << "calc_delta_lambda delta_lambda in longinteger" << endl;
		exit(1);
		}
}

void design_lambda_max(
		int t, int v, int k, discreta_base & lambda_max)
{
	Binomial(v - t, k - t, lambda_max);
}

void design_lambda_max_half(
		int t, int v, int k,
		discreta_base & lambda_max_half)
{
	discreta_base lambda_max, two, r;
	
	design_lambda_max(t, v, k, lambda_max);
	two.m_i_i(2);
	lambda_max.integral_division(two, lambda_max_half, r, 0);
}

void design_lambda_ijs_matrix(
		int t, int v, int k,
		discreta_base& lambda,
		int s, discreta_matrix & M, int verbose_level)
{
	int i, j;
	
	M.m_mn_n(t + 1, t + 1);
	for (i = 0; i <= t; i++) {
		for (j = 0; j <= t - i; j++) {
			design_lambda_ijs(t, v, k, lambda, s, i, j, M[i][j], verbose_level);
			}
		}
}

void design_lambda_ijs(
		int t, int v, int k,
		discreta_base& lambda, int s, int i, int j,
		discreta_base & lambda_ijs, int verbose_level)
//\lambda_{i,j}^{(s)} =
// \sum_{h=0}^j (-1)^h {j \choose h} {\lambda_{i+h} \choose s}
//cf. Wilson, Van Lint~\cite{VanLintWilson92}.
{
	discreta_base a, b, c;
	int h;
	
	lambda_ijs.m_i_i(0);
	for (h = 0; h <= j; h++) {
		Binomial(j, h, a);
		if (ODD(h))
			a.negate();
		design_lambda_ij(t, v, k, lambda, i + h, 0, b, verbose_level);
		N_choose_K(b, s, c, verbose_level);
		a *= c;
		lambda_ijs += a;
		}
}

void design_lambda_ij(
		int t, int v, int k,
		discreta_base& lambda, int i, int j,
		discreta_base & lambda_ij, int verbose_level)
//\lambda_{i,j} = \lambda \frac{{v-i-j \choose k-i}}{{v-t \choose k-t}}
//cf. Wilson, Van Lint~\cite{VanLintWilson92}.
{
	discreta_base a, b;
	
	Binomial(v - i - j, k - i, a);
	Binomial(v - t, k - t, b);
	lambda_ij = lambda;
	lambda_ij *= a;
	// cout << "design_lambda_ij() t=" << t << " v=" << v << " k=" << k
	//<< " lambda=" << lambda << " i=" << i << " j=" << j << endl;
	// cout << "design_lambda_ij() a=" << a << endl;
	// cout << "design_lambda_ij() b=" << b << endl;
	lambda_ij.divide_by_exact(b, verbose_level);
}

int is_trivial_clan(
		int t, int v, int k)
{
	discreta_base dl, lambda_max;
	
	int delta_lambda = calc_delta_lambda(v, t, k, false);
	dl.m_i_i(delta_lambda);
	design_lambda_max(t, v, k, lambda_max);
	if (dl.eq(lambda_max))
		return true;
	else
		return false;
}

void print_clan_tex_int(
		int t, int v, int k, int verbose_level)
{
	integer T(t), V(v), K(k);
	discreta_base lambda_max, m_max;
	
	int delta_lambda = calc_delta_lambda(v, t, k, false);
	design_lambda_max(t, v, k, lambda_max);
	lambda_max.integral_division_by_integer_exact(delta_lambda, m_max, verbose_level);
	print_clan_tex(T, V, K, delta_lambda, m_max);
}

void print_clan_tex_int(
		int t, int v, int k,
		int delta_lambda,
		discreta_base &m_max)
{
	integer T(t), V(v), K(k);
	print_clan_tex(T, V, K, delta_lambda, m_max);
}

void print_clan_tex(
		discreta_base &t, discreta_base &v,
		discreta_base &k,
		int delta_lambda, discreta_base &m_max)
{
	Vector vp, ve;
	
	factor_integer(m_max.s_i_i(), vp, ve);
	cout << t << "\\mbox{-}(" << v << "," << k << ", m \\cdot "
			<< delta_lambda << ")_{m \\le ";
	if (vp.s_l() > 1 || (vp.s_l() > 0 && ve.s_ii(0) > 1)) {
		{
		class printing_mode pm(printing_mode_latex);
		discreta_print_factorization(vp, ve, cout);
		}
		}
	else {
		cout << m_max;
		}
	cout << "}";
}

int is_ancestor(
		int t, int v, int k)
{
	int delta_lambda = calc_delta_lambda(v, t, k, false);
	return is_ancestor(t, v, k, delta_lambda);
}

int is_ancestor(
		int t, int v, int k, int delta_lambda)
{
	int c, T, V, K, Delta_lambda;
	
	if (calc_redinv(t, v, k, delta_lambda, c, T, V, K, Delta_lambda) && c == 1) {
		// cout << "is_ancestor(): " << t << " " << v << " " << k
		//<< " is not ancestor, red^-1 is possible for c=" << c << endl;
		return false;
		}
	if (calc_derinv(t, v, k, delta_lambda, c, T, V, K, Delta_lambda) && c == 1) {
		// cout << "is_ancestor(): " << t << " " << v << " " << k
		//<< " is not ancestor, der^-1 is possible for c=" << c << endl;
		return false;
		}
	if (calc_resinv(t, v, k, delta_lambda, c, T, V, K, Delta_lambda) && c == 1) {
		// cout << "is_ancestor(): " << t << " " << v << " " << k
		//<< " is not ancestor, res^-1 is possible for c=" << c << endl;
		return false;
		}
	return true;
}

int calc_redinv(
		int t, int v, int k, int delta_lambda,
		int &c, int &T, int &V, int &K, int &Delta_lambda)
{
	long int vt, kt, g, v1, k1, gg;
	algebra::number_theory::number_theory_domain NT;
	
	if (t == k)
		return false;
	T = t + 1;
	V = v;
	K = k;
	vt = v - t;
	kt = k - t;
	g = NT.gcd_lint(vt, kt);
	v1 = vt / g;
	k1 = kt / g;
	gg = NT.gcd_lint(delta_lambda, v1);
	c = v1 / gg;
	Delta_lambda = k1 * delta_lambda / gg;
	return true;
}

int calc_derinv(
		int t, int v, int k, int delta_lambda,
		int &c, int &T, int &V, int &K, int &Delta_lambda)
{
	T = t + 1;
	V = v + 1;
	K = k + 1;
	Delta_lambda = calc_delta_lambda(V, T, K, false);
	c = Delta_lambda / delta_lambda;
	return true;
}

int calc_resinv(
		int t, int v, int k, int delta_lambda,
		int &c, int &T, int &V, int &K, int &Delta_lambda)
{
	long int a, b, g;
	algebra::number_theory::number_theory_domain NT;
	
	if (t == k)
		return false;
	T = t + 1;
	V = v + 1;
	K = k;
	Delta_lambda = calc_delta_lambda(V, T, K, false);
	a = Delta_lambda * (v + 1 - k);
	b = delta_lambda * (k - t);
	g = NT.gcd_lint(a, b);
	c = a / g;
	return true;
}

void design_mendelsohn_coefficient_matrix(
		int t, int m, discreta_matrix & M)
//The Mendelsohn equations for any $t$-$(v,k,\lambda)$ design $\cD = (\cV, \cB)$  
//and any $m$-subset $M \subseteq \cV$ are for $s \ge 1$:
//\[
//\sum_{j=i}^m {m \choose j} \alpha_j^{(s)}(M) = 
//{\lambda_i \choose s} {m \choose i} \quad \text{for} i=0,\ldots,t 
//\]
//cf. Mendelsohn~\cite{Mendelsohn71}.
{
	int i, j;
	
	M.m_mn_n(t + 1, m + 1);
	for (i = 0; i <= t; i++) {
		for (j = i; j <= m; j++) {
			Binomial(j, i, M[i][j]);
			}
		}
}

void design_mendelsohn_rhs(
		int v, int t, int k,
		discreta_base& lambda,
		int m, int s, Vector & rhs, int verbose_level)
{
	int i;
	discreta_base a, b, c;
	
	rhs.m_l(t + 1);
	for (i = 0; i <= t; i++) {
		Binomial(m, i, a);
		design_lambda_ij(t, v, k, lambda, i, 0, b, verbose_level);
		N_choose_K(b, s, c, verbose_level);
		rhs[i].mult(a, c, verbose_level);
	}
}

int design_parameter_database_already_there(
		database &D,
		design_parameter &p, int& idx)
{
	int verbose_level = 0;
	btree& B_tvkl = D.btree_access_i(2);
	
	idx = B_tvkl.search_unique_int8_int8_int8_int8(
		p.t(), p.v(), p.K(), p.lambda().s_i_i(), verbose_level);
	if (idx == -1)
		return false;
	else
		return true;
}

void design_parameter_database_add_if_new(
		database &D,
		design_parameter &p,
		long int& highest_id, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;
	
	if (!design_parameter_database_already_there(D, p, idx)) {
		p.id() = ++highest_id;
		D.add_object(p, verbose_level - 2);
		if (f_v) {
			cout << p.id() << " added: " << p
					<< " new highest_id=" << highest_id << endl;
			}
		}
	else {
		int btree_idx = 2;
		btree& B_tvkl = D.btree_access_i(btree_idx);
	
		design_parameter p1;
		KEYTYPE key;
		DATATYPE data;
		
		B_tvkl.ith(idx, &key, &data, verbose_level - 1);
		D.get_object(&data, p1, verbose_level - 2);
		// D.ith_object(idx, btree_idx, p1, false, false);
		for (int i = 0; i < p.source().s_l(); i++) {
			p1.source().append(p.source_i(i));
			}
		D.delete_object(p1, data.datref, verbose_level - 2);
		D.add_object(p1, verbose_level - 2);
		if (f_v) {
			cout << p1.id() << " changed: " << p1 << endl;
			}
		}
}

void design_parameter_database_closure(
		database &D,
		int highest_id_already_closed,
		int minimal_t,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (!D.f_open()) {
		cout << "design_parameter_database_closure "
				"database not open" << endl;
		exit(1);
		}
	long int highest_id, old_highest_id, id;
	int btree_idx_id = 0;
	
	highest_id = D.get_highest_int8(btree_idx_id);
	old_highest_id = highest_id;
	if (f_v) {
		cout << "design_parameter_database_closure "
				"highest_id_already_closed=" << highest_id_already_closed
			<< " highest_id=" << highest_id << endl;
		}
	for (id = highest_id_already_closed + 1; id <= highest_id; id++) {
		design_parameter p, q;
		
		D.get_object_by_unique_int8(btree_idx_id, id, p, verbose_level);
		if (f_vv) {
			cout << "closure of design #" << id << " : " << p << endl;
			}
		
		if (f_vv) cout << "reduced_t:" << endl;
		p.reduced_t(q, verbose_level);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q,
					highest_id, verbose_level - 2);
			}
		
		if (f_vv) cout << "derived:" << endl;
		p.derived(q, verbose_level);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q,
					highest_id, verbose_level - 2);
			}
		
		if (f_vv) cout << "residual:" << endl;
		p.residual(q, verbose_level);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q,
					highest_id, verbose_level - 2);
			}
		
		if (p.trung_complementary(q, verbose_level)) {
			if (f_vv) cout << "trung_complementary:" << endl;
			if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
				design_parameter_database_add_if_new(D, q,
						highest_id, verbose_level - 2);
				}
			}
		
		if (p.alltop(q)) {
			if (f_vv) cout << "alltop:" << endl;
			if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
				design_parameter_database_add_if_new(D, q,
						highest_id, verbose_level - 2);
				}
			}
		
		if (p.v() == 2 * p.K() + 1) {
			if (f_vv) cout << "complementary design:" << endl;
			p.complementary(q, verbose_level);
			if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
				design_parameter_database_add_if_new(D, q,
						highest_id, verbose_level - 2);
				}
			}
		
#if 0
		if (f_vv) cout << "supplementary design:" << endl;
		p.supplementary(q);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q,
					highest_id, verbose_level - 2);
			}
#endif
		
		if (f_vv) cout << "supplementary_reduced_t:" << endl;
		p.supplementary_reduced_t(q, verbose_level);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q,
					highest_id, verbose_level - 2);
			}

		if (f_vv) cout << "supplementary_derived:" << endl;
		p.supplementary_derived(q, verbose_level);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q,
					highest_id, verbose_level - 2);
			}
		
		if (f_vv) cout << "supplementary_residual:" << endl;
		p.supplementary_residual(q, verbose_level);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q,
					highest_id, verbose_level - 2);
			}
		
		int t1, v1, k1;
		discreta_base lambda1;
		int t_new, v_new, k_new;
		discreta_base lambda_new;
		int idx;
		
		if (p.trung_left_partner(t1, v1, k1, lambda1, t_new, v_new,
				k_new, lambda_new, verbose_level) && lambda_new.s_kind() == INTEGER) {
			if (f_vv) cout << "trung_left_partner:" << endl;
			q.init();
			q.t() = t1;
			q.v() = v1;
			q.K() = k1;
			q.lambda() = lambda1;
			if (design_parameter_database_already_there(D, q, idx)) {
				q.t() = t_new;
				q.v() = v_new;
				q.K() = k_new;
				q.lambda() = lambda_new;
				design_parameter_source S;
				S.init();
				S.rule() = rule_trung_left;
				S.prev() = p.id();
				q.source().append(S);
				if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
					design_parameter_database_add_if_new(D, q,
							highest_id, verbose_level - 2);
					}
				}
			}
		
		if (p.trung_right_partner(t1, v1, k1, lambda1, t_new,
				v_new, k_new, lambda_new, verbose_level) && lambda_new.s_kind() == INTEGER) {
			if (f_vv) cout << "trung_right_partner:" << endl;
			q.init();
			q.t() = t1;
			q.v() = v1;
			q.K() = k1;
			q.lambda() = lambda1;
			if (design_parameter_database_already_there(D, q, idx)) {
				q.t() = t_new;
				q.v() = v_new;
				q.K() = k_new;
				q.lambda() = lambda_new;
				design_parameter_source S;
				S.init();
				S.rule() = rule_trung_right;
				S.prev() = p.id();
				q.source().append(S);
				if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
					design_parameter_database_add_if_new(D, q,
							highest_id, verbose_level - 2);
					}
				}
			}
		

		}
	if (f_v) {
		cout << "design_parameter_database_closure "
				"highest_id=" << highest_id
			<< ", i.e.  closuring yields=" << highest_id - old_highest_id
			<< " new parameter sets." << endl;
		}
}

//#define BUFSIZE 50000

void design_parameter_database_read_design_txt(
		char *fname_design_txt,
		char *path_db,
		int f_form_closure, int minimal_t, int verbose_level)
{
	char buf[BUFSIZE], *p_buf;
	char comment[BUFSIZE];
	int t, v, k, lambda;
	int btree_idx_id = 0;
	other::data_structures::string_tools ST;

	ifstream f(fname_design_txt);
	if (!f) {
		cout << "error opening file " << fname_design_txt << endl;
		exit(1);
		}
	design_parameter p;
	database D;
	
	p.init_database(D, path_db);
	D.open(verbose_level - 1);

	long int id = 0;
	long int highest_id_already_closed = -1;
	while (true) {
		if (f.eof()) {
			break;
			}
		f.getline(buf, sizeof(buf));
		p_buf = buf;
		if (buf[0] == '#')
			continue;
		ST.s_scan_int(&p_buf, &t);
		if (t == -1)
			break;
		ST.s_scan_int(&p_buf, &v);
		ST.s_scan_int(&p_buf, &k);
		ST.s_scan_int(&p_buf, &lambda);
		strcpy(comment, p_buf);
		// cout << "t=" << t << " v=" << v << " k=" << k
		//<< " lambda=" << lambda << " comment=" << comment << endl;
		
		p.init(t, v, k, lambda);
		if (strlen(comment)) {
			design_parameter_source S;
			
			S.init();
			S.comment().init(comment);
			p.source().append(S);
			}
		p.id() = id;
		cout << p << endl;
		
		
		// we check if the parameter set is admissible:
		{
		integer lambda_object(lambda);
		discreta_matrix M;
		
		design_lambda_ijs_matrix(t, v, k, lambda_object, 1 /* s */, M, verbose_level);
		}
		
		int idx;
		if (design_parameter_database_already_there(D, p, idx)) {
			cout << "already there, we are changing the dataset:" << endl;
			long int highest_id = -1;
				// highest_id is not used in the following routine 
				//as we know the dataset is already there:
			design_parameter_database_add_if_new(D, p,
					highest_id, verbose_level - 2);
			}
		else {
			D.add_object(p, verbose_level - 2);
			
			if (f_form_closure)
				design_parameter_database_closure(D,
						highest_id_already_closed, minimal_t,
						verbose_level - 2);
	
			highest_id_already_closed = D.get_highest_int8(btree_idx_id);
			id = highest_id_already_closed + 1;
			}
		}
	D.close(verbose_level - 1);
	
	// D.print(0, cout);


}

void design_parameter_database_export_tex(
		char *path_db)
{
	int verbose_level = 0;
	int btree_idx_id = 0;
	int btree_idx_tvkl = 2;
	
	design_parameter p;
	database D;
	
	p.init_database(D, path_db);
	D.open(verbose_level);

	long int id, highest_id;
	
	highest_id = D.get_highest_int8(btree_idx_id);

	cout << "design_parameter_database_export_tex() db_path=" << path_db
			<< " highest_id = " << highest_id << endl;



	int highest_page = highest_id / 100, i, page;
	Vector fname_page;
	
	fname_page.m_l(highest_page + 1);
	for (i = 0; i <= highest_page; i++) {
		hollerith h;
		
		h.init("design_id_ge_");
		h.append_i(i * 100);
		h.append(".html");
		fname_page.s_i(i) = h;
		}





	ofstream f("designs.tex", ios::trunc);
	other::l1_interfaces::latex_interface L;

	string title, author, extra_praeamble;

	title.assign("$t$-Designs");
	author.assign("DISCRETA");

	L.head(f, true /* f_book */, true /* f_title */,
		title, author, true /* f_toc */,
		false /* f_landscape */,
		true /* f_12pt */, 
		true /* f_enlarged_page */, 
		true /* f_pagenumbers */,
		extra_praeamble /* extra_praeamble */);
	printing_mode pm(printing_mode_latex);
	

	f << "\n\\chapter{Designs by $t, v, k, \\lambda$}\n\n";
	btree &B = D.btree_access_i(btree_idx_tvkl);
	int idx, len;
	long int t_min, t_max, t;
	
	len = B.length(verbose_level - 2);
	D.ith_object(0, btree_idx_tvkl, p, verbose_level - 2);
	t_min = p.t();
	D.ith_object(len - 1, btree_idx_tvkl, p, verbose_level - 2);
	t_max = p.t();


	hollerith fname_dir, h1, h2;
			
	fname_dir.init("designs.html");
	ofstream fhtml_dir(fname_dir.s());
			
			
	h1.init("t designs with small t");
	h2.init("t designs with small t");
		
	html_head(fhtml_dir, h1.s(), h2.s());	


	fhtml_dir << "<ul>" << endl;
	
	for (t = t_min; t <= t_max; t++) {
		int first, len;
		long int v, v_min, v_max;
		
		B.search_interval_int8(t, t, first, len, verbose_level);
		if (len == 0)
			continue;
		
		int nb_restricted = determine_restricted_number_of_designs_t(
				D, B, btree_idx_tvkl, t, first, len);


		f << "\\newpage\n\n";
		cout << "t=" << t << " number of designs: " << nb_restricted << endl;
		
		f << "\n\\section{Designs with $t=" << t << "$}\n\n";
		
		f << "There are alltogether " << nb_restricted << " parameter sets "
				"of designs with $t=" << t << "$.\\\\" << endl;
		
		fhtml_dir << "<li> t=" << t << " (" << nb_restricted << " parameter "
				"sets of designs)" << endl;
		
		D.ith_object(first, btree_idx_tvkl, p, verbose_level - 2);
		v_min = p.v();
		D.ith_object(first + len - 1, btree_idx_tvkl, p, verbose_level - 2);
		v_max = p.v();
		




		fhtml_dir << "<ul>" << endl;
		for (v = v_min; v <= v_max; v++) {
			int first, len;
			long int k, k_min, k_max;
		
			B.search_interval_int8_int8(t, t, v, v, first, len,
					verbose_level);
			if (len == 0)
				continue;
		
		
			f << "\n\\subsection{Designs with $t=" << t << "$, $v=" << v << "$}\n\n";
		
			int nb_restricted = determine_restricted_number_of_designs_t_v(D, B, btree_idx_tvkl, t, v, first, len);
			
			f << "There are alltogether " << nb_restricted << " parameter sets of designs with $t=" << t << "$ and $v=" << v << "$.\\\\" << endl;
		
		
			fhtml_dir << "<li> <a href=\"design_t" << t << "_v" << v << ".html\"> v=" << v << " (" << nb_restricted << " parameter sets of designs) </a>" << endl;
			
			D.ith_object(first, btree_idx_tvkl, p, verbose_level - 2);
			k_min = p.K();
			D.ith_object(first + len - 1, btree_idx_tvkl, p, verbose_level - 2);
			k_max = p.K();

			hollerith fname, h1, h2;
			
			fname.init("design_t");
			fname.append_i(t);
			fname.append("_v");
			fname.append_i(v);
			fname.append(".html");
			ofstream fhtml(fname.s());
			
			
			h1.init("t designs with t=");
			h1.append_i(t);
			h1.append(", v=");
			h1.append_i(v);
			h2.init("t designs with t=");
			h2.append_i(t);
			h2.append(", v=");
			h2.append_i(v);
		
			html_head(fhtml, h1.s(), h2.s());	



			for (k = k_min; k <= k_max; k++) {
				int first, len;
		
				B.search_interval_int8_int8_int8(t, t, v, v, k, k, first, len, verbose_level);
				if (len == 0)
					continue;
				
				discreta_base lambda_max, lambda_max_half;
				design_lambda_max(t, v, k, lambda_max);
				design_lambda_max_half(t, v, k, lambda_max_half);
				// cout << "t=" << t << " v=" << v << " k=" << k << " lambda_max=" << lambda_max << endl;
				int delta_lambda = calc_delta_lambda(v, t, k, false);





				Vector v_lambda, v_id;
				v_lambda.m_l(len);
				v_id.m_l_n(len);
				
				int l = 0;
				for (int i = 0; i < len; i++) {
					idx = first + i;
					D.ith_object(idx, btree_idx_tvkl, p, verbose_level - 2);
					
					if (p.lambda().s_i_i() > lambda_max_half.s_i_i())
						continue;
					v_lambda.s_i(l) = p.lambda();
					v_id.m_ii(l, p.id());
					l++;
					} // next i
				
				if (l) {
					if (l == 1) {
						hollerith link;
						int id = v_id.s_ii(0);
						prepare_link(link, id);
						f << "$" << t << "$-$(" << v << "," << k << ", " << v_lambda.s_i(0) << "_{\\#" << v_id.s_ii(0) << "})$" << endl;
						fhtml << "<a href=\"" << link.s() << "\">" << t << "-(" << v << "," << k << ", " << v_lambda.s_i(0) << ") </a><br>" << endl;
						}
					else {
						f << t << "-(" << v << "," << k << ",$\\lambda$) for $\\lambda \\in \\{";
						fhtml << t << "-(" << v << "," << k << ",lambda) for lambda in {";
						for (int ii = 0; ii < l; ii++) {
							hollerith link;
							int id = v_id.s_ii(ii);
							prepare_link(link, id);

							f << v_lambda.s_i(ii) << "_{\\#" << v_id.s_ii(ii) << "}";
							fhtml << " <a href=\"" << link.s() << "\">" << v_lambda.s_i(ii) << "</a>";
							if (ii < l - 1) {
								f << ",$ $";
								fhtml << ",";
								}
							if ((ii % 10) == 0) {
								f << endl;
								fhtml << endl;
								}
							}
						f << "\\}$ (" << l << " parameter sets)" << endl;
						fhtml << "} (" << l << " parameter sets)" << endl;
						}
					f << "$\\Delta \\lambda=" << delta_lambda << "$, $\\lambda_{max}=" << lambda_max << "$\\\\" << endl;
					fhtml << "delta lambda = "  << delta_lambda << ", lambda_max=" << lambda_max << "<br>" << endl;
					}
				} // next k
			html_foot(fhtml);
			
			} // next v
		fhtml_dir << "</ul>" << endl;

		} // next t
	fhtml_dir << "</ul>" << endl;
	
	fhtml_dir << "<p><hr><p>" << endl;
	
	fhtml_dir << "<a href=\"design_clans.html\"> design_clans </a>" << endl;
	
	fhtml_dir << "<p><hr><p>" << endl;
	
	fhtml_dir << "<ul>" << endl;
	for (page = 0; page <= highest_page; page++) {
		fhtml_dir << "<li> <a href=\"" << fname_page[page].as_hollerith().s() << "\"> id >= " << page * 100 << "</a>" << endl;
		}
	fhtml_dir << "</ul>" << endl;
	
	html_foot(fhtml_dir);
	
	
	f << "\n\\chapter{Designs by ID}\n\n";
	for (id = 0; id <= highest_id; id++) {
		if (id % 100 == 0) {
			f << "\n\\section{ID $\\ge " << id << "$}\n\n";
			cout << "ID >= " << id << endl;
			}
		if (!D.get_object_by_unique_int8_if_there(btree_idx_id, id, p, verbose_level))
			continue;
		// f << "\\subsection*{\\# " << id << "}\n";
		// f << "\\label{designID" << id << "}\n";
		
		hollerith h;
		p.text_parameter(h);
		f << "\\# " << p.id() << ": " << h.s() << endl;
			
		int j, l;
			
		design_parameter p1, ancestor;
		Vector path;
		
		p1 = p;
		p1.ancestor(ancestor, path, 0 /* verbose_level */);
		// cout << "ancestor=" << ancestor << endl;
		l = p.source().s_l();
		f << "\\begin{enumerate}\n";
		f << "\\item\n";
		f << "clan: " << ancestor.t() << "-(" 
			<< ancestor.v() << "," 
			<< ancestor.K() << "," 
			<< ancestor.lambda() << ")";
		if (path.s_ii(0)) {
			f << ", " << path.s_ii(0) << " $\\times$ reduced $t$";
			}
		if (path.s_ii(1)) {
			f << ", " << path.s_ii(1) << " $\\times$ derived";
			}
		if (path.s_ii(2)) {
			f << ", " << path.s_ii(2) << " $\\times$ residual";
			}
		f << endl; 
		
		for (j = 0; j < l; j++) {
			f << "\\item\n";
				
			hollerith s0, s1, s2;
				
			design_parameter_source& S = p.source_i(j);
			S.text012_extended(p, s0, s1, s2);
			f << s1.s();
			if (S.prev() != -1) {
				hollerith h;
				prepare_design_parameters_from_id(D, S.prev(), h);
				f << " " << h.s() << " (\\# " << S.prev() << ")";
				}
			f << s2.s() << endl;
			// S.text2(p, h);
			// f << h.s() << endl;
			}
		f << "\\end{enumerate}\n";
		f << "\\smallskip" << endl;
		}
	
	L.foot(f);

	for (page = 0; page <= highest_page; page++) {
		cout << "ID >= " << page * 100 << endl;
		ofstream fhtml(fname_page[page].as_hollerith().s());
		hollerith h1, h2;
		
		h1.init("t designs with small t, id ge ");
		h1.append_i(page * 100);
		h2.init("t designs with small t, id ge ");
		h2.append_i(page * 100);
		
		html_head(fhtml, h1.s(), h2.s());	

		for (id = page * 100; id <= MINIMUM((page + 1) * 100 - 1, highest_id); id++) {
			if (!D.get_object_by_unique_int8_if_there(btree_idx_id, id, p, verbose_level))
				continue;
		
			hollerith h;
			p.text_parameter(h);
			fhtml << "<a name=\"design"<< p.id() << "\"> # " << p.id() << ": " << h.s() << "</a>" << endl;
			
			int j, l;
			
			design_parameter ancestor, p1;
			Vector path;
			
			p1 = p;
			p1.ancestor(ancestor, path, 0 /* verbose_level */);
			
			l = p.source().s_l();
			fhtml << "<ul>\n";
			fhtml << "<li>clan: <a href=\"design_clan_" 
				<< ancestor.t() << "_" 
				<< ancestor.v() << "_" 
				<< ancestor.K() << ".html\"> " 
				<< ancestor.t() << "-(" 
				<< ancestor.v() << "," 
				<< ancestor.K() << "," 
				<< ancestor.lambda() << ")";
			if (path.s_ii(0)) {
				fhtml << ", " << path.s_ii(0) << " times reduced t";
				}
			if (path.s_ii(1)) {
				fhtml << ", " << path.s_ii(1) << " times derived";
				}
			if (path.s_ii(2)) {
				fhtml << ", " << path.s_ii(2) << " times residual";
				}
			fhtml << "</a>" << endl; 

			for (j = 0; j < l; j++) {
				fhtml << "<li>\n";
				
				hollerith s0, s1, s2;
					
				design_parameter_source& S = p.source_i(j);
				S.text012_extended(p, s0, s1, s2);
				fhtml << s1.s();
				if (S.prev() != -1) {
					hollerith link, h;
					prepare_link(link, S.prev());
					fhtml << " <a href=\"" << link.s() << "\">";
					prepare_design_parameters_from_id(D, S.prev(), h);
					fhtml << h.s() << " (# " << S.prev() << ") </a> ";
					}
				fhtml << s2.s() << endl;
				}
			fhtml << "</ul>\n";
			fhtml << "<p><hr><p>" << endl;
			}

		html_foot(fhtml);
		}
	D.close(verbose_level);

	
	
	
}

int determine_restricted_number_of_designs_t(
		database &D, btree &B,
	int btree_idx_tvkl, long int t, int first, int len)
{
	int verbose_level = 0;
	design_parameter p;
	long int v, v_min, v_max;
	int nb_restricted = 0;
	
	D.ith_object(first, btree_idx_tvkl, p, verbose_level - 2);
	v_min = p.v();
	D.ith_object(first + len - 1, btree_idx_tvkl, p, verbose_level - 2);
	v_max = p.v();

	for (v = v_min; v <= v_max; v++) {
		int first, len;
		
		B.search_interval_int8_int8(t, t, v, v, first, len, verbose_level);
		if (len == 0)
			continue;
		
		nb_restricted += determine_restricted_number_of_designs_t_v(D, B, 
			btree_idx_tvkl, t, v, first, len);
		}
	
	return nb_restricted;
}

int determine_restricted_number_of_designs_t_v(
		database &D, btree &B,
	int btree_idx_tvkl, long int t, long int v, int first, int len)
{
	int verbose_level = 0;
	design_parameter p;
	long int k, k_min, k_max;
	int nb_restricted = 0;
	
	D.ith_object(first, btree_idx_tvkl, p, verbose_level - 2);
	k_min = p.K();
	D.ith_object(first + len - 1, btree_idx_tvkl, p, verbose_level - 2);
	k_max = p.K();

	for (k = k_min; k <= k_max; k++) {
		int first, len;
		
		B.search_interval_int8_int8_int8(t, t, v, v, k, k, first, len, verbose_level);
		if (len == 0)
			continue;
				
		discreta_base lambda_max, lambda_max_half;
		design_lambda_max(t, v, k, lambda_max);
		design_lambda_max_half(t, v, k, lambda_max_half);
		// cout << "t=" << t << " v=" << v << " k=" << k << " lambda_max=" << lambda_max << endl;
		// int delta_lambda = calc_delta_lambda(v, t, k, false);

		int l = 0;
		for (int i = 0; i < len; i++) {
			int idx = first + i;
			D.ith_object(idx, btree_idx_tvkl, p, verbose_level - 2);
					
			if (p.lambda().s_i_i() > lambda_max_half.s_i_i())
				continue;
			l++;
			} // next i
		nb_restricted += l;
		}
	return nb_restricted;
}

void prepare_design_parameters_from_id(
		database &D, long int id, hollerith& h)
{
	int verbose_level = 0;
	int btree_idx_id = 0;
	design_parameter p;
	
	D.get_object_by_unique_int8(btree_idx_id, id, p, verbose_level);
	h.init("");
	h.append_i(p.t());
	h.append("-(");
	h.append_i(p.v());
	h.append(",");
	h.append_i(p.K());
	h.append(",");
	h.append_i(p.lambda().s_i_i());
	h.append(")");
}

void prepare_link(
		hollerith& link, int id)
{
	int page = id / 100;
	link.init("design_id_ge_");
	link.append_i(page * 100);
	link.append(".html#design");
	link.append_i(id);
}

#include <stdio.h>

void design_parameter_database_clans(
		char *path_db, int f_html, int verbose_level)
{
	//int verbose_level = 0;
	int btree_idx_id = 0;
	//int btree_idx_tvkl = 2;
	
	design_parameter p, q;
	database D;
	Vector ancestor, clan_lambda, clan_member, clan_member_path;
	
	p.init_database(D, path_db);
	D.open(verbose_level);

	long int id, highest_id;
	int idx1, idx2;
	
	highest_id = D.get_highest_int8(btree_idx_id);

	ancestor.m_l(0);
	clan_lambda.m_l(0);
	clan_member.m_l(0);
	clan_member_path.m_l(0);
	for (id = 0; id <= highest_id; id++) {

		if (!D.get_object_by_unique_int8_if_there(btree_idx_id, id, p, verbose_level))
			continue;
		
				
		discreta_base lambda_max_half;
		design_lambda_max_half(p.t(), p.v(), p.K(), lambda_max_half);
		if (p.lambda().s_i_i() > lambda_max_half.s_i_i())
			continue;
		
		
		Vector g, path;
		p.ancestor(q, path, verbose_level);
		
		g.m_l_n(3);
		g[0].m_i_i(q.t());
		g[1].m_i_i(q.v());
		g[2].m_i_i(q.K());
		//g[3] = q.lambda();
		
		if (ancestor.search(g, &idx1)) {
			cout << "clan found at " << idx1 << endl;
			Vector &CL = clan_lambda[idx1].as_vector();
			Vector &CM = clan_member[idx1].as_vector();
			Vector &CMP = clan_member_path[idx1].as_vector();
			if (CL.search(q.lambda(), &idx2)) {
				cout << "family found at " << idx2 << endl;
				Vector &cm = CM[idx2].as_vector();
				cm.append_integer(id);
				Vector &cmp = CMP[idx2].as_vector();
				cmp.append(path);
				}
			else {
				cout << "new family within the clan, inserting at " << idx2 << endl;
				CL.insert_element(idx2, q.lambda());
				Vector cm, cmp;
				cm.m_l(1);
				cm.m_ii(0, id);
				cmp.m_l(1);
				cmp[0] = path;
				CM.insert_element(idx2, cm);
				CMP.insert_element(idx2, cmp);
				}
			}
		else {
			cout << "new clan, inserting at " << idx1 << endl;
			ancestor.insert_element(idx1, g);
			Vector gf, cm, CM, cmp, CMP;
			gf.m_l(1);
			gf[0] = q.lambda();
			clan_lambda.insert_element(idx1, gf);
			cm.m_l(1);
			cm.m_ii(0, id);
			CM.m_l(0);
			CM.insert_element(0, cm);
			clan_member.insert_element(idx1, CM);
			cmp.m_l(1);
			cmp[0] = path;
			CMP.m_l(0);
			CMP.insert_element(0, cmp);
			clan_member_path.insert_element(idx1, CMP);
			}
		cout << "number of clans: " << ancestor.s_l() << endl;
		// cout << "clan = " << ancestor << endl;
		}
	
	int i, l, j, ll, h, lll;
	l = ancestor.s_l();
	cout << "there are " << l << " clans of design parameter sets:" << endl;
	for (i = 0; i < l; i++) {
		cout << "clan no " << i << " : ancestor = " << ancestor[i];
		Vector &g = ancestor[i].as_vector();
		int t = g.s_ii(0);
		int v = g.s_ii(1);
		int k = g.s_ii(2);
		
		int delta_lambda = calc_delta_lambda(v, t, k, false);
		cout << " delta_lambda = " << delta_lambda;
		discreta_base lambda_max, lambda_max_half;
		design_lambda_max(t, v, k, lambda_max);
		design_lambda_max_half(t, v, k, lambda_max_half);
		cout << " lambda_max = " << lambda_max;
		cout << " lambda_max_half = " << lambda_max_half << endl;
		}
	cout << endl;
	for (i = 0; i < l; i++) {
		cout << i << " & " << ancestor[i];
		Vector &g = ancestor[i].as_vector();
		int t = g.s_ii(0);
		int v = g.s_ii(1);
		int k = g.s_ii(2);
		
		int delta_lambda = calc_delta_lambda(v, t, k, false);
		cout << " & " << delta_lambda;
		discreta_base lambda_max, lambda_max_half;
		design_lambda_max(t, v, k, lambda_max);
		design_lambda_max_half(t, v, k, lambda_max_half);
		cout << " & " << lambda_max;
		Vector &CL = clan_lambda[i].as_vector();
		ll = CL.s_l();
		cout << " & $\\{  ";
		for (j = 0; j < ll; j++) {
			discreta_base dl, q;
			
			dl.m_i_i(delta_lambda);
			CL[j].integral_division_exact(dl, q, verbose_level);
			cout << q;
			if (j < ll - 1)
				cout << "$, $";
			}
		cout << " \\} $ ";
		cout << "\\\\" << endl;
		}
	cout << endl;
	
	
	for (i = 0; i < l; i++) {
		cout << "clan no " << i << " : ancestor = " << ancestor[i];
		Vector &g = ancestor[i].as_vector();
		int t = g.s_ii(0);
		int v = g.s_ii(1);
		int k = g.s_ii(2);
		
		int delta_lambda = calc_delta_lambda(v, t, k, false);
		cout << " delta_lambda = " << delta_lambda;
		discreta_base lambda_max, lambda_max_half;
		design_lambda_max(t, v, k, lambda_max);
		design_lambda_max_half(t, v, k, lambda_max_half);
		cout << " lambda_max = " << lambda_max;
		cout << " lambda_max_half = " << lambda_max_half << endl;
		
		Vector &CL = clan_lambda[i].as_vector();
		Vector &CM = clan_member[i].as_vector();
		ll = CL.s_l();
		cout << "containing " << ll << " families: " << endl;
		for (j = 0; j < ll; j++) {
			Vector &f = CM[j].as_vector();
			discreta_base &lambda = CL[j];
			lll = f.s_l();
			cout << "family " << j << ", lambda = " << lambda << " containing " << lll << " designs:" << endl;
			for (h = 0; h < lll; h++) {
				cout << "#" << f.s_ii(h) << " ";
				if (((h + 1) % 10) == 0)
					cout << endl;
				}
			cout << endl;
			}
		}
	D.close(verbose_level);
	
	if (f_html) {
		design_parameter_database_clan_report(path_db, ancestor, clan_lambda, clan_member, clan_member_path);
		}
}

void design_parameter_database_family_report(
		char *path_db, int t, int v, int k,
		int lambda, int minimal_t, int verbose_level)
{
	// int btree_idx_id = 0;
	int btree_idx_tvkl = 2;
	
	cout << "design_parameter_database_family_report() t=" << t << " v=" << v << " k=" << k << " lambda=" << lambda << endl;
	design_parameter p;
	Vector Layers;
	
	database D;
	
	p.init_database(D, path_db);
	D.open(verbose_level);
	
	btree& B_tvkl = D.btree_access_i(btree_idx_tvkl);
	
	int h, i, j, idx, id;
	
	Layers.m_l(t + 1);
	for (h = 0; h <= t; h++) {
		Layers[h].change_to_matrix();
		Layers[h].as_matrix().m_mn(h + 1, h + 1);
		}
	
	
	for (h = 0; h < t; h++) {
		if (t - h < minimal_t)
			continue;
		// cout << "h=" << h << endl;
		discreta_matrix &M = Layers[h].as_matrix();
		for (i = 0; i <= h; i++) {
			for (j = 0; j <= h - i; j++) {
				Vector entry;

				prepare_entry(entry, i, j, h, t, v, k, lambda, verbose_level);
				id = -1;
				if (entry.s_i(3).s_kind() == INTEGER) {
					idx = B_tvkl.search_unique_int8_int8_int8_int8(
							entry.s_ii(0), entry.s_ii(1), entry.s_ii(2),
							entry.s_ii(3), verbose_level);
					// idx is -1 if the dataset has not been found.
					if (idx != -1) {
						D.ith_object(idx, btree_idx_tvkl, p, verbose_level - 2);
						id = p.id();
						}
					}
				entry.m_ii(4, id);
				M.s_ij(i, j) = entry;
				} // next j
			} // next i
		} // next h
	
	D.close(verbose_level);
	
	
	for (h = 0; h < t; h++) {
		if (t - h < minimal_t)
			continue;
		discreta_matrix &M = Layers[h].as_matrix();
		cout << "h=" << h << endl;
		for (i = 0; i <= h; i++) {
			for (j = 0; j <= h; j++) {
				if (j <= h - i) {
					Vector &entry = M.s_ij(i, j).as_vector();
					cout << entry[0] << "-(" << entry[1] << "," << entry[2] << "," << entry[3] << ")";
					id = entry.s_ii(4);
					if (id != -1) {
						cout << "_{\\#" << id << "}";
						}
					}
				if (j < h)
					cout << " & ";
				} // next j
			cout << "\\\\" << endl;
			} // next i
		} // next h
}

static void prepare_entry(
		Vector &entry, int i, int j, int h, int t, int v, int k, int lambda, int verbose_level)
{
	design_parameter p, q;
	
	int h1 = h - i - j, u;
	if (h1 < 0) {
		cout << "prepare_entry() h1 < 0" << endl;
		exit(1);
		}
	
	p.init(t, v, k, lambda);
	for (u = 0; u < i; u++) {
		p.derived(q, verbose_level);
		p.swap(q);
		}
	for (u = 0; u < j; u++) {
		p.residual(q, verbose_level);
		p.swap(q);
		}
	for (u = 0; u < h1; u++) {
		p.reduced_t(q, verbose_level);
		p.swap(q);
		}
	entry.m_l(5);
	entry.m_ii(0, p.t());
	entry.m_ii(1, p.v());
	entry.m_ii(2, p.K());
	entry[3] = p.lambda();
	entry.m_ii(4, -1);
}

void design_parameter_database_clan_report(
		char *path_db,
		Vector &ancestor, Vector &clan_lambda,
		Vector & clan_member, Vector & clan_member_path)
{
	int verbose_level = 0;
	//int btree_idx_id = 0;
	//int btree_idx_tvkl = 2;
	
	design_parameter p, q;
	database D;
	
	p.init_database(D, path_db);
	D.open(verbose_level);

	//int highest_id;
	
	//highest_id = D.get_highest_int4(btree_idx_id);

	hollerith fname, fname_tex, fname_dir, h1, h2;

	fname_dir.init("design_clans.html");
	ofstream fhtml_dir(fname_dir.s());
			
			
	h1.init("t designs with small t by clans");
	h2.init("t designs with small t by clans");
		
	html_head(fhtml_dir, h1.s(), h2.s());	

	fhtml_dir << "in brackets: number of families / overall "
			"number of design parameter sets per clan<br>" << endl;

	fhtml_dir << "<ul>" << endl;
	int i, j, l, ll, s, lll;
	l = ancestor.s_l();
	for (i = 0; i < l; i++) {
		Vector &a = ancestor[i].as_vector();
		int t = a.s_ii(0);
		int v = a.s_ii(1);
		int k = a.s_ii(2);
		int delta_lambda = calc_delta_lambda(v, t, k, false);
		//cout << " delta_lambda = " << delta_lambda;
		discreta_base lambda_max, lambda_max_half, m_max, dl, r;
		dl.m_i_i(delta_lambda);
		design_lambda_max(t, v, k, lambda_max);
		design_lambda_max_half(t, v, k, lambda_max_half);
		// cout << " lambda_max = " << lambda_max;
		//cout << " lambda_max_half = " << lambda_max_half << endl;
		lambda_max_half.integral_division(dl, m_max, r, 0);
		ll = clan_lambda[i].as_vector().s_l();
		s = clan_member[i].as_vector().vector_of_vectors_overall_length();
		
		fhtml_dir << "<a href=\"design_clan_" << t << "_" << v
				<< "_" << k << ".html\">";
		fhtml_dir << t << "-(" << v << "," << k << "," << "m*"
				<< delta_lambda << ")</a>, 1 <= m <= " << m_max
			<< "; (" << ll << "/" << s << ") lambda_max=" << lambda_max
			<< ", lambda_max_half=" << lambda_max_half
			<< "<br>" << endl;
		}
	fhtml_dir << "</ul>" << endl;
	html_foot(fhtml_dir);
	
	
	for (i = 0; i < l; i++) {

		Vector &a = ancestor[i].as_vector();
		int t = a.s_ii(0);
		int v = a.s_ii(1);
		int k = a.s_ii(2);
		fname.init("design_clan_");
		fname.append_i(t);
		fname.append("_");
		fname.append_i(v);
		fname.append("_");
		fname.append_i(k);
		fname_tex = fname;
		fname_tex.append(".tex");
		fname.append(".html");
	
		ofstream fhtml(fname.s());
		ofstream ftex(fname_tex.s());
			
			
		h1.init("design clan: ");
		h1.append_i(t);
		h1.append("_");
		h1.append_i(v);
		h1.append("_");
		h1.append_i(k);
		h2.init("design clan: ");
		h2.append_i(t);
		h2.append("_");
		h2.append_i(v);
		h2.append("_");
		h2.append_i(k);
		
		html_head(fhtml, h1.s(), h2.s());	


		int delta_lambda = calc_delta_lambda(v, t, k, false);
		//cout << " delta_lambda = " << delta_lambda;
		discreta_base lambda_max, lambda_max_half, m_max, dl, r;
		dl.m_i_i(delta_lambda);
		design_lambda_max(t, v, k, lambda_max);
		design_lambda_max_half(t, v, k, lambda_max_half);
		// cout << " lambda_max = " << lambda_max;
		//cout << " lambda_max_half = " << lambda_max_half << endl;
		lambda_max_half.integral_division(dl, m_max, r, 0);
		ll = clan_lambda[i].as_vector().s_l();
		s = clan_member[i].as_vector().vector_of_vectors_overall_length();
		fhtml << t << "-(" << v << "," << k << "," << "m*"
				<< delta_lambda << "), 1 <= m <= " << m_max
			<< "; (" << ll << "/" << s << ") lambda_max=" << lambda_max
			<< ", lambda_max_half=" << lambda_max_half
			<< "<br>" << endl;
		ftex << "\\subsection*{Clan " << i << ": $" << t << "$-$(" << v
				<< "," << k
			<< ",m\\cdot " << delta_lambda << ")$}\n";
		ftex << "The clan contains " << ll << " families:\\\\" << endl;

		
		Vector &CL = clan_lambda[i].as_vector();
		Vector &CM = clan_member[i].as_vector();
		Vector &CMP = clan_member_path[i].as_vector();
		ll = CL.s_l();
		fhtml << "the clan contains " << ll << " families: " << endl;
		fhtml << "<ul>" << endl;
		for (j = 0; j < ll; j++) {
			Vector &cm = CM[j].as_vector();
			Vector &cmp = CMP[j].as_vector();
			discreta_base &lambda = CL[j];
			lll = cm.s_l();
			fhtml << "<li>family " << j << ", lambda = " << lambda
					<< " containing " << lll << " designs:" << endl;
			fhtml << "<br>" << endl;
			ftex << "\\subsubsection*{Family with $\\lambda="
					<< lambda << "$}" << endl;
			ftex << "The family contains " << lll
					<< " design parameter sets:\\\\" << endl;
#if 0
			int h;
			for (h = 0; h < lll; h++) {
				hollerith link, text1;
				int id = cm.s_ii(h);
				Vector &path = cmp.s_i(h).as_vector();
				prepare_link(link, id);
				fhtml << " <a href=\"" << link.s() << "\">";
				prepare_design_parameters_from_id(D, id, text1);
				fhtml << text1.s() << " (#" << id << "), path=" << path << " </a>, ";
				if (((h + 1) % 10) == 0)
					fhtml << "<br>" << endl;
				}
			fhtml << endl;
#endif
			Vector min_path, max_path;
			int max_depth, minimal_t;
		
			determine_minimal_and_maximal_path(cmp, min_path, max_path, max_depth);
			minimal_t = t - max_depth;
			
			fhtml << "<br>minpath=" << min_path << " minimal_t=" << minimal_t << endl;
			design_parameter dominating_ancestor;
			determine_dominating_ancestor(t, v, k, lambda, min_path, dominating_ancestor, verbose_level);
			// fhtml << "<br>dominating_ancestor: " << dominating_ancestor
			//<< " (path=" << min_path << ")" << endl;
			reduce_path(cmp, min_path);
			family_report(D, fhtml, ftex, dominating_ancestor.t(),
					dominating_ancestor.v(), dominating_ancestor.K(),
					dominating_ancestor.lambda(), cm, cmp, minimal_t, verbose_level);
			}		
		fhtml << "</ul>" << endl;

		html_foot(fhtml);
		}
	
	
	
	D.close(verbose_level);
}

static void determine_minimal_and_maximal_path(
		Vector &v,
		Vector & min_path, Vector & max_path, int & max_depth)
{
	int i, l, j, ll, depth;
	
	l = v.s_l();
	if (l == 0) {
		cout << "determine_minimal_and_maximal_path "
				"l == 0" << endl;
		exit(1);
		}
	ll = v[0].as_vector().s_l();
	min_path = v[0];
	max_path = v[0];
	max_depth = 0;
	for (i = 0; i < l; i++) {
		Vector & p = v[i].as_vector();
		if (p.s_l() != ll) {
			cout << "determine_minimal_and_maximal_path "
					"different lengths!" << endl;
			exit(1);
			}
		depth = p.s_ii(0) + p.s_ii(1) + p.s_ii(2);
		for (j = 0; j < ll; j++) {
			min_path.s_ii(j) = MINIMUM(min_path.s_ii(j), p.s_ii(j));
			max_path.s_ii(j) = MAXIMUM(max_path.s_ii(j), p.s_ii(j));
			max_depth = MAXIMUM(max_depth, depth);
			}
		}
}

static void determine_dominating_ancestor(
		int t, int v, int k,
		discreta_base & lambda, Vector & path,
		design_parameter &dominating_ancestor, int verbose_level)
{
	design_parameter p, q;
	int u;
	
	p.init(t, v, k, lambda);
	for (u = 0; u < path.s_ii(0); u++) {
		p.reduced_t(q, verbose_level);
		p.swap(q);
		}
	for (u = 0; u < path.s_ii(1); u++) {
		p.derived(q, verbose_level);
		p.swap(q);
		}
	for (u = 0; u < path.s_ii(2); u++) {
		p.residual(q, verbose_level);
		p.swap(q);
		}
	dominating_ancestor = p;
}

static void reduce_path(
		Vector &cmp, Vector &min_path)
{
	int i, l, j;
	
	l = cmp.s_l();
	for (i = 0; i < l; i++) {
		Vector &path = cmp[i].as_vector();
		for (j = 0; j < 3; j++) {
			path.s_ii(j) -= min_path.s_ii(j);
			}
		}
}

static void family_report(
		database & D, ostream& fhtml, ostream &ftex,
		int t, int v, int k, discreta_base &lambda, Vector & cm,
		Vector & cmp, int minimal_t, int verbose_level)
{
	int h, i, j, idx, idx1, id, nb_found = 0;
	Vector Layers;
	
	permutation per;
	cmp.sort_with_logging(per);
	
	Layers.m_l(t + 1);
	for (h = 0; h <= t; h++) {
		Layers[h].change_to_matrix();
		Layers[h].as_matrix().m_mn(h + 1, h + 1);
		}
	
	
	for (h = 0; h < t; h++) {
		if (t - h < minimal_t)
			continue;
		// cout << "h=" << h << endl;
		discreta_matrix &M = Layers[h].as_matrix();
		for (i = 0; i <= h; i++) {
			for (j = 0; j <= h - i; j++) {
				Vector path;

				path.m_l_n(3);
				path.m_ii(0, h - i - j);
				path.m_ii(1, i);
				path.m_ii(2, j);
				if (cmp.search(path, &idx)) {
					idx1 = per.s_i(idx);
					id = cm.s_ii(idx1);
					M.m_iji(i, j, id);
					nb_found++;
					}
				else {
					M.m_iji(i, j, -1);
					}
				} // next j
			} // next i
		} // next h
	if (nb_found != cm.s_l()) {
		cout << "family_report() nb_found != cm.s_l()" << endl;
		cout << "nb_found = " << nb_found << endl;
		cout << "nb of designs in the family = " << cm.s_l() << endl;
		exit(1);
		}
	fhtml << "<ul>" << endl;
	
	for (h = 0; h < t; h++) {
		if (t - h < minimal_t)
			continue;
		// cout << "h=" << h << endl;
		fhtml << "<li>" << endl;
		ftex << "\\begin{tabular}{*{" << h + 1 << "}{l}}" << endl;
		discreta_matrix &M = Layers[h].as_matrix();
		for (i = 0; i <= h; i++) {
			for (j = 0; j <= h - i; j++) {
				int id = M.s_iji(i, j);
				Vector path;
				
				path.m_l_n(3);
				path.m_ii(0, h - i - j);
				path.m_ii(1, i);
				path.m_ii(2, j);
				design_parameter p;
				determine_dominating_ancestor(t, v, k, lambda, path, p, verbose_level);
				if (id >= 0) {
					hollerith link, text1;
					
					prepare_link(link, id);
					fhtml << " <a href=\"" << link.s() << "\">";
					prepare_design_parameters_from_id(D, id, text1);
					fhtml << text1.s() << " (#" << id << ")</a> ";
					ftex << "$\\underline{\\mbox{" << text1.s() << "}}$";
					}
				else {
					fhtml << p.t() << "-(" << p.v() << "," << p.K()
							<< "," << p.lambda() << ") ";
					ftex << "$" << p.t() << "$-$(" << p.v() << ","
							<< p.K() << "," << p.lambda() << ")$";
					}
				if (j < h)
					ftex << " & ";
				} // next j
			for (; j < h; j++)
				ftex << " & ";
			ftex << "\\\\" << endl;
			
			fhtml << "<br>" << endl;
			} // next i
		fhtml << "<p>" << endl;
		ftex << "\\end{tabular}\\\\" << endl;
		}
	fhtml << "</ul>" << endl;
}

static void f_m_j(
		int m, int j, discreta_base &a, int verbose_level)
{
	int q = m / j;
	int r = m % j;
	if (q == 0) {
		a.m_i_i(0);
		return;
		}
	if (q == 1) {
		a.m_i_i(r);
		return;
		}
	discreta_base b, c, d, e, J, R, two;
	
	two.m_i_i(2);
	b.m_i_i(q);
	c.m_i_i(q - 1);
	d.mult(b, c, verbose_level);
	d.integral_division_exact(two, c, verbose_level);
	J.m_i_i(j);
	c *= J;
	R.m_i_i(r);
	R *= b;
	c += R;
	a = c;
}

static int max_m(
		int i, int j, int verbose_level)
{
	int m;
	discreta_base a, b, c, d, two;
	
	two.m_i_i(2);
	b.m_i_i(i);
	c.m_i_i(i - 1);
	d.mult(b, c, verbose_level);
	d.integral_division_exact(two, a, verbose_level);
	for (m = 0; ; m++) {
		f_m_j(m, j, b, verbose_level);
		if (b.gt(a)) {
			break;
			}
		}
	return m - 1;
}

int Maxfit(
		int i, int j, int verbose_level)
{
	int a, b, c;
	
	a = max_m(i, j, verbose_level);
	b = max_m(j, i, verbose_level);
	c = MINIMUM(a, b);
	return c;
}




}}}


