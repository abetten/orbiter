// design.C
//
// Anton Betten
// 18.09.2000
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"


static void prepare_entry(Vector &entry, INT i, INT j, INT h, INT t, INT v, INT k, INT lambda);
static void determine_minimal_and_maximal_path(Vector &v, Vector & min_path, Vector & max_path, INT & max_depth);
static void determine_dominating_ancestor(INT t, INT v, INT k, discreta_base & lambda, Vector & path, design_parameter &dominating_ancestor);
static void reduce_path(Vector &cmp, Vector &min_path);
static void family_report(database & D, ostream& fhtml, ostream &ftex, INT t, INT v, INT k, discreta_base &lambda, Vector & cm, Vector & cmp, INT minimal_t);


INT design_parameters_admissible(INT v, INT t, INT k, discreta_base & lambda)
{
	INT delta_lambda = calc_delta_lambda(v, t, k, FALSE);
	discreta_base b, q, r;
	
	b.m_i_i(delta_lambda);
	lambda.integral_division(b, q, r, 0);
	if (!r.is_zero())
		return FALSE;
	
	discreta_base lambda_max;
	design_lambda_max(t, v, k, lambda_max);
	if (lambda.gt(lambda_max))
		return FALSE;
	return TRUE;
}

INT calc_delta_lambda(INT v, INT t, INT k, INT f_v)
{
	discreta_base lambda;
	INT i;
	discreta_base a, b, a1, b1, g, rhs_a, rhs_b, delta_lambda, dl, a2, b2, gg;
	
	// f_v = TRUE;

	lambda.m_i_i(1);
	if (f_v) {
		cout << "calc_delta_lambda(): v=" << v << " t=" << t << " k=" << k << " lambda=" << lambda << endl;
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
			a.mult(rhs_a, a1);
			b.mult(rhs_b, b1);
			a.extended_gcd(b, a2, b2, g, 0);
			a.divide_by_exact(g);
			b.divide_by_exact(g);
			delta_lambda.extended_gcd(b, a2, b2, gg, 0);
			b1 = b;
			b1.divide_by_exact(gg);
			dl.mult(delta_lambda, b1);
			delta_lambda = dl;
			if (f_v) {
				cout << "t'=" << i << " lambda'=" << a << "/" << b << " delta_lambda=" << delta_lambda << endl;
				}
			rhs_a = a;
			rhs_b = b;
			}
		}
	if (delta_lambda.s_kind() == INTEGER)
		return delta_lambda.s_i_i();
	else {
		cout << "calc_delta_lambda() delta_lambda in longinteger" << endl;
		exit(1);
		}
}

void design_lambda_max(INT t, INT v, INT k, discreta_base & lambda_max)
{
	Binomial(v - t, k - t, lambda_max);
}

void design_lambda_max_half(INT t, INT v, INT k, discreta_base & lambda_max_half)
{
	discreta_base lambda_max, two, r;
	
	design_lambda_max(t, v, k, lambda_max);
	two.m_i_i(2);
	lambda_max.integral_division(two, lambda_max_half, r, 0);
}

void design_lambda_ijs_matrix(INT t, INT v, INT k, discreta_base& lambda, INT s, matrix & M)
{
	INT i, j;
	
	M.m_mn_n(t + 1, t + 1);
	for (i = 0; i <= t; i++) {
		for (j = 0; j <= t - i; j++) {
			design_lambda_ijs(t, v, k, lambda, s, i, j, M[i][j]);
			}
		}
}

void design_lambda_ijs(INT t, INT v, INT k, discreta_base& lambda, INT s, INT i, INT j, discreta_base & lambda_ijs)
//\lambda_{i,j}^{(s)} = \sum_{h=0}^j (-1)^h {j \choose h} {\lambda_{i+h} \choose s}
//cf. Wilson, Van Lint~\cite{VanLintWilson92}.
{
	discreta_base a, b, c;
	INT h;
	
	lambda_ijs.m_i_i(0);
	for (h = 0; h <= j; h++) {
		Binomial(j, h, a);
		if (ODD(h))
			a.negate();
		design_lambda_ij(t, v, k, lambda, i + h, 0, b);
		N_choose_K(b, s, c);
		a *= c;
		lambda_ijs += a;
		}
}

void design_lambda_ij(INT t, INT v, INT k, discreta_base& lambda, INT i, INT j, discreta_base & lambda_ij)
//\lambda_{i,j} = \lambda \frac{{v-i-j \choose k-i}}{{v-t \choose k-t}}
//cf. Wilson, Van Lint~\cite{VanLintWilson92}.
{
	discreta_base a, b;
	
	Binomial(v - i - j, k - i, a);
	Binomial(v - t, k - t, b);
	lambda_ij = lambda;
	lambda_ij *= a;
	// cout << "design_lambda_ij() t=" << t << " v=" << v << " k=" << k << " lambda=" << lambda << " i=" << i << " j=" << j << endl;
	// cout << "design_lambda_ij() a=" << a << endl;
	// cout << "design_lambda_ij() b=" << b << endl;
	lambda_ij.divide_by_exact(b);
}

INT is_trivial_clan(INT t, INT v, INT k)
{
	discreta_base dl, lambda_max;
	
	INT delta_lambda = calc_delta_lambda(v, t, k, FALSE);
	dl.m_i_i(delta_lambda);
	design_lambda_max(t, v, k, lambda_max);
	if (dl.eq(lambda_max))
		return TRUE;
	else
		return FALSE;
}

void print_clan_tex_INT(INT t, INT v, INT k)
{
	integer T(t), V(v), K(k);
	discreta_base lambda_max, m_max;
	
	INT delta_lambda = calc_delta_lambda(v, t, k, FALSE);
	design_lambda_max(t, v, k, lambda_max);
	lambda_max.integral_division_by_integer_exact(delta_lambda, m_max);
	print_clan_tex(T, V, K, delta_lambda, m_max);
}

void print_clan_tex_INT(INT t, INT v, INT k, INT delta_lambda, discreta_base &m_max)
{
	integer T(t), V(v), K(k);
	print_clan_tex(T, V, K, delta_lambda, m_max);
}

void print_clan_tex(discreta_base &t, discreta_base &v, discreta_base &k, INT delta_lambda, discreta_base &m_max)
{
	Vector vp, ve;
	
	factor_integer(m_max.s_i_i(), vp, ve);
	cout << t << "\\mbox{-}(" << v << "," << k << ", m \\cdot " << delta_lambda << ")_{m \\le ";
	if (vp.s_l() > 1 || (vp.s_l() > 0 && ve.s_ii(0) > 1)) {
		{
		class printing_mode pm(printing_mode_latex);
		print_factorization(vp, ve, cout);
		}
		}
	else {
		cout << m_max;
		}
	cout << "}";
}

INT is_ancestor(INT t, INT v, INT k)
{
	INT delta_lambda = calc_delta_lambda(v, t, k, FALSE);
	return is_ancestor(t, v, k, delta_lambda);
}

INT is_ancestor(INT t, INT v, INT k, INT delta_lambda)
{
	INT c, T, V, K, Delta_lambda;
	
	if (calc_redinv(t, v, k, delta_lambda, c, T, V, K, Delta_lambda) && c == 1) {
		// cout << "is_ancestor(): " << t << " " << v << " " << k << " is not ancestor, red^-1 is possible for c=" << c << endl;
		return FALSE;
		}
	if (calc_derinv(t, v, k, delta_lambda, c, T, V, K, Delta_lambda) && c == 1) {
		// cout << "is_ancestor(): " << t << " " << v << " " << k << " is not ancestor, der^-1 is possible for c=" << c << endl;
		return FALSE;
		}
	if (calc_resinv(t, v, k, delta_lambda, c, T, V, K, Delta_lambda) && c == 1) {
		// cout << "is_ancestor(): " << t << " " << v << " " << k << " is not ancestor, res^-1 is possible for c=" << c << endl;
		return FALSE;
		}
	return TRUE;
}

INT calc_redinv(INT t, INT v, INT k, INT delta_lambda, INT &c, INT &T, INT &V, INT &K, INT &Delta_lambda)
{
	INT vt, kt, g, v1, k1, gg;
	
	if (t == k)
		return FALSE;
	T = t + 1;
	V = v;
	K = k;
	vt = v - t;
	kt = k - t;
	g = gcd_INT(vt, kt);
	v1 = vt / g;
	k1 = kt / g;
	gg = gcd_INT(delta_lambda, v1);
	c = v1 / gg;
	Delta_lambda = k1 * delta_lambda / gg;
	return TRUE;
}

INT calc_derinv(INT t, INT v, INT k, INT delta_lambda, INT &c, INT &T, INT &V, INT &K, INT &Delta_lambda)
{
	T = t + 1;
	V = v + 1;
	K = k + 1;
	Delta_lambda = calc_delta_lambda(V, T, K, FALSE);
	c = Delta_lambda / delta_lambda;
	return TRUE;
}

INT calc_resinv(INT t, INT v, INT k, INT delta_lambda, INT &c, INT &T, INT &V, INT &K, INT &Delta_lambda)
{
	INT a, b, g;
	
	if (t == k)
		return FALSE;
	T = t + 1;
	V = v + 1;
	K = k;
	Delta_lambda = calc_delta_lambda(V, T, K, FALSE);
	a = Delta_lambda * (v + 1 - k);
	b = delta_lambda * (k - t);
	g = gcd_INT(a, b);
	c = a / g;
	return TRUE;
}

void design_mendelsohn_coefficient_matrix(INT t, INT m, matrix & M)
//The Mendelsohn equations for any $t$-$(v,k,\lambda)$ design $\cD = (\cV, \cB)$  
//and any $m$-subset $M \subseteq \cV$ are for $s \ge 1$:
//\[
//\sum_{j=i}^m {m \choose j} \alpha_j^{(s)}(M) = 
//{\lambda_i \choose s} {m \choose i} \quad \text{for} i=0,\ldots,t 
//\]
//cf. Mendelsohn~\cite{Mendelsohn71}.
{
	INT i, j;
	
	M.m_mn_n(t + 1, m + 1);
	for (i = 0; i <= t; i++) {
		for (j = i; j <= m; j++) {
			Binomial(j, i, M[i][j]);
			}
		}
}

void design_mendelsohn_rhs(INT v, INT t, INT k, discreta_base& lambda, INT m, INT s, Vector & rhs)
{
	INT i;
	discreta_base a, b, c;
	
	rhs.m_l(t + 1);
	for (i = 0; i <= t; i++) {
		Binomial(m, i, a);
		design_lambda_ij(t, v, k, lambda, i, 0, b);
		N_choose_K(b, s, c);
		rhs[i].mult(a, c);
		}
}

INT design_parameter_database_already_there(database &D, design_parameter &p, INT& idx)
{
	INT verbose_level = 0;
	btree& B_tvkl = D.btree_access_i(2);
	
	idx = B_tvkl.search_unique_INT4_INT4_INT4_INT4(
		p.t(), p.v(), p.K(), p.lambda().s_i_i(), verbose_level);
	if (idx == -1)
		return FALSE;
	else
		return TRUE;
}

void design_parameter_database_add_if_new(database &D, design_parameter &p, INT& highest_id, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT idx;
	
	if (!design_parameter_database_already_there(D, p, idx)) {
		p.id() = ++highest_id;
		D.add_object(p, verbose_level - 2);
		if (f_v) {
			cout << p.id() << " added: " << p << " new highest_id=" << highest_id << endl;
			}
		}
	else {
		INT btree_idx = 2;
		btree& B_tvkl = D.btree_access_i(btree_idx);
	
		design_parameter p1;
		KEYTYPE key;
		DATATYPE data;
		
		B_tvkl.ith(idx, &key, &data, verbose_level - 1);
		D.get_object(&data, p1, verbose_level - 2);
		// D.ith_object(idx, btree_idx, p1, FALSE, FALSE);
		for (INT i = 0; i < p.source().s_l(); i++) {
			p1.source().append(p.source_i(i));
			}
		D.delete_object(p1, data.datref, verbose_level - 2);
		D.add_object(p1, verbose_level - 2);
		if (f_v) {
			cout << p1.id() << " changed: " << p1 << endl;
			}
		}
}

void design_parameter_database_closure(database &D, INT highest_id_already_closed, INT minimal_t, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (!D.f_open()) {
		cout << "design_parameter_database_closure() database not open" << endl;
		exit(1);
		}
	INT highest_id, old_highest_id, id;
	INT btree_idx_id = 0;
	
	highest_id = D.get_highest_INT4(btree_idx_id);
	old_highest_id = highest_id;
	if (f_v) {
		cout << "design_parameter_database_closure() highest_id_already_closed=" << highest_id_already_closed 
			<< " highest_id=" << highest_id << endl;
		}
	for (id = highest_id_already_closed + 1; id <= highest_id; id++) {
		design_parameter p, q;
		
		D.get_object_by_unique_INT4(btree_idx_id, id, p, verbose_level);
		if (f_vv) {
			cout << "closure of design #" << id << " : " << p << endl;
			}
		
		if (f_vv) cout << "reduced_t:" << endl;
		p.reduced_t(q);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q, highest_id, verbose_level - 2);
			}
		
		if (f_vv) cout << "derived:" << endl;
		p.derived(q);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q, highest_id, verbose_level - 2);
			}
		
		if (f_vv) cout << "residual:" << endl;
		p.residual(q);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q, highest_id, verbose_level - 2);
			}
		
		if (p.trung_complementary(q)) {
			if (f_vv) cout << "trung_complementary:" << endl;
			if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
				design_parameter_database_add_if_new(D, q, highest_id, verbose_level - 2);
				}
			}
		
		if (p.alltop(q)) {
			if (f_vv) cout << "alltop:" << endl;
			if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
				design_parameter_database_add_if_new(D, q, highest_id, verbose_level - 2);
				}
			}
		
		if (p.v() == 2 * p.K() + 1) {
			if (f_vv) cout << "complementary design:" << endl;
			p.complementary(q);
			if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
				design_parameter_database_add_if_new(D, q, highest_id, verbose_level - 2);
				}
			}
		
#if 0
		if (f_vv) cout << "supplementary design:" << endl;
		p.supplementary(q);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q, highest_id, verbose_level - 2);
			}
#endif
		
		if (f_vv) cout << "supplementary_reduced_t:" << endl;
		p.supplementary_reduced_t(q);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q, highest_id, verbose_level - 2);
			}

		if (f_vv) cout << "supplementary_derived:" << endl;
		p.supplementary_derived(q);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q, highest_id, verbose_level - 2);
			}
		
		if (f_vv) cout << "supplementary_residual:" << endl;
		p.supplementary_residual(q);
		if (q.t() >= minimal_t && q.lambda().s_kind() == INTEGER) {
			design_parameter_database_add_if_new(D, q, highest_id, verbose_level - 2);
			}
		
		INT t1, v1, k1;
		discreta_base lambda1;
		INT t_new, v_new, k_new;
		discreta_base lambda_new;
		INT idx;
		
		if (p.trung_left_partner(t1, v1, k1, lambda1, t_new, v_new, k_new, lambda_new) && lambda_new.s_kind() == INTEGER) {
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
					design_parameter_database_add_if_new(D, q, highest_id, verbose_level - 2);
					}
				}
			}
		
		if (p.trung_right_partner(t1, v1, k1, lambda1, t_new, v_new, k_new, lambda_new) && lambda_new.s_kind() == INTEGER) {
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
					design_parameter_database_add_if_new(D, q, highest_id, verbose_level - 2);
					}
				}
			}
		

		}
	if (f_v) {
		cout << "design_parameter_database_closure() highest_id=" << highest_id 
			<< ", i.e.  closuring yields=" << highest_id - old_highest_id << " new parameter sets." << endl;
		}
}

//#define BUFSIZE 50000

void design_parameter_database_read_design_txt(BYTE *fname_design_txt, BYTE *path_db, INT f_form_closure, INT minimal_t, INT verbose_level)
{
	BYTE buf[BUFSIZE], *p_buf;
	BYTE comment[BUFSIZE];
	INT t, v, k, lambda;
	INT btree_idx_id = 0;

	ifstream f(fname_design_txt);
	if (!f) {
		cout << "error opening file " << fname_design_txt << endl;
		exit(1);
		}
	design_parameter p;
	database D;
	
	p.init_database(D, path_db);
	D.open(verbose_level - 1);

	INT id = 0;
	INT highest_id_already_closed = -1;
	while (TRUE) {
		if (f.eof()) {
			break;
			}
		f.getline(buf, sizeof(buf));
		p_buf = buf;
		if (buf[0] == '#')
			continue;
		s_scan_int(&p_buf, &t);
		if (t == -1)
			break;
		s_scan_int(&p_buf, &v);
		s_scan_int(&p_buf, &k);
		s_scan_int(&p_buf, &lambda);
		strcpy(comment, p_buf);
		// cout << "t=" << t << " v=" << v << " k=" << k << " lambda=" << lambda << " comment=" << comment << endl;
		
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
		matrix M;
		
		design_lambda_ijs_matrix(t, v, k, lambda_object, 1 /* s */, M);
		}
		
		INT idx;
		if (design_parameter_database_already_there(D, p, idx)) {
			cout << "already there, we are changing the dataset:" << endl;
			INT highest_id = -1;
				// highest_id is not used in the following routine 
				//as we know the dataset is already there:
			design_parameter_database_add_if_new(D, p, highest_id, verbose_level - 2);
			}
		else {
			D.add_object(p, verbose_level - 2);
			
			if (f_form_closure)
				design_parameter_database_closure(D, highest_id_already_closed, minimal_t, verbose_level - 2);
	
			highest_id_already_closed = D.get_highest_INT4(btree_idx_id);
			id = highest_id_already_closed + 1;
			}
		}
	D.close(verbose_level - 1);
	
	// D.print(0, cout);


}

void design_parameter_database_export_tex(BYTE *path_db)
{
	INT verbose_level = 0;
	INT btree_idx_id = 0;
	INT btree_idx_tvkl = 2;
	
	design_parameter p;
	database D;
	
	p.init_database(D, path_db);
	D.open(verbose_level);

	INT id, highest_id;
	
	highest_id = D.get_highest_INT4(btree_idx_id);

	cout << "design_parameter_database_export_tex() db_path=" << path_db << " highest_id = " << highest_id << endl;



	INT highest_page = highest_id / 100, i, page;
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

	latex_head(f, TRUE /* f_book */, TRUE /* f_title */, 
		"$t$-Designs", "DISCRETA", TRUE /* f_toc */, 
		FALSE /* f_landscape */,
		TRUE /* f_12pt */, 
		TRUE /* f_enlarged_page */, 
		TRUE /* f_pagenumbers */);
	printing_mode pm(printing_mode_latex);
	

	f << "\n\\chapter{Designs by $t, v, k, \\lambda$}\n\n";
	btree &B = D.btree_access_i(btree_idx_tvkl);
	INT idx, len;
	INT t_min, t_max, t;
	
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
		INT first, len;
		INT v, v_min, v_max;
		
		B.search_interval_INT4(t, t, first, len, verbose_level);
		if (len == 0)
			continue;
		
		INT nb_restricted = determine_restricted_number_of_designs_t(D, B, btree_idx_tvkl, t, first, len);


		f << "\\newpage\n\n";
		cout << "t=" << t << " number of designs: " << nb_restricted << endl;
		
		f << "\n\\section{Designs with $t=" << t << "$}\n\n";
		
		f << "There are alltogether " << nb_restricted << " parameter sets of designs with $t=" << t << "$.\\\\" << endl;
		
		fhtml_dir << "<li> t=" << t << " (" << nb_restricted << " parameter sets of designs)" << endl;
		
		D.ith_object(first, btree_idx_tvkl, p, verbose_level - 2);
		v_min = p.v();
		D.ith_object(first + len - 1, btree_idx_tvkl, p, verbose_level - 2);
		v_max = p.v();
		




		fhtml_dir << "<ul>" << endl;
		for (v = v_min; v <= v_max; v++) {
			INT first, len;
			INT k, k_min, k_max;
		
			B.search_interval_INT4_INT4(t, t, v, v, first, len, verbose_level);
			if (len == 0)
				continue;
		
		
			f << "\n\\subsection{Designs with $t=" << t << "$, $v=" << v << "$}\n\n";
		
			INT nb_restricted = determine_restricted_number_of_designs_t_v(D, B, btree_idx_tvkl, t, v, first, len);
			
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
				INT first, len;
		
				B.search_interval_INT4_INT4_INT4(t, t, v, v, k, k, first, len, verbose_level);
				if (len == 0)
					continue;
				
				discreta_base lambda_max, lambda_max_half;
				design_lambda_max(t, v, k, lambda_max);
				design_lambda_max_half(t, v, k, lambda_max_half);
				// cout << "t=" << t << " v=" << v << " k=" << k << " lambda_max=" << lambda_max << endl;
				INT delta_lambda = calc_delta_lambda(v, t, k, FALSE);





				Vector v_lambda, v_id;
				v_lambda.m_l(len);
				v_id.m_l_n(len);
				
				INT l = 0;
				for (INT i = 0; i < len; i++) {
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
						INT id = v_id.s_ii(0);
						prepare_link(link, id);
						f << "$" << t << "$-$(" << v << "," << k << ", " << v_lambda.s_i(0) << "_{\\#" << v_id.s_ii(0) << "})$" << endl;
						fhtml << "<a href=\"" << link.s() << "\">" << t << "-(" << v << "," << k << ", " << v_lambda.s_i(0) << ") </a><br>" << endl;
						}
					else {
						f << t << "-(" << v << "," << k << ",$\\lambda$) for $\\lambda \\in \\{";
						fhtml << t << "-(" << v << "," << k << ",lambda) for lambda in {";
						for (INT ii = 0; ii < l; ii++) {
							hollerith link;
							INT id = v_id.s_ii(ii);
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
		if (!D.get_object_by_unique_INT4_if_there(btree_idx_id, id, p, verbose_level))
			continue;
		// f << "\\subsection*{\\# " << id << "}\n";
		// f << "\\label{designID" << id << "}\n";
		
		hollerith h;
		p.text_parameter(h);
		f << "\\# " << p.id() << ": " << h.s() << endl;
			
		INT j, l;
			
		design_parameter p1, ancestor;
		Vector path;
		
		p1 = p;
		p1.ancestor(ancestor, path, FALSE, FALSE);
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
	
	latex_foot(f);

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
			if (!D.get_object_by_unique_INT4_if_there(btree_idx_id, id, p, verbose_level))
				continue;
		
			hollerith h;
			p.text_parameter(h);
			fhtml << "<a name=\"design"<< p.id() << "\"> # " << p.id() << ": " << h.s() << "</a>" << endl;
			
			INT j, l;
			
			design_parameter ancestor, p1;
			Vector path;
			
			p1 = p;
			p1.ancestor(ancestor, path, FALSE, FALSE);
			
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

INT determine_restricted_number_of_designs_t(database &D, btree &B, 
	INT btree_idx_tvkl, INT t, INT first, INT len)
{
	INT verbose_level = 0;
	design_parameter p;
	INT v, v_min, v_max;
	INT nb_restricted = 0;
	
	D.ith_object(first, btree_idx_tvkl, p, verbose_level - 2);
	v_min = p.v();
	D.ith_object(first + len - 1, btree_idx_tvkl, p, verbose_level - 2);
	v_max = p.v();

	for (v = v_min; v <= v_max; v++) {
		INT first, len;
		
		B.search_interval_INT4_INT4(t, t, v, v, first, len, verbose_level);
		if (len == 0)
			continue;
		
		nb_restricted += determine_restricted_number_of_designs_t_v(D, B, 
			btree_idx_tvkl, t, v, first, len);
		}
	
	return nb_restricted;
}

INT determine_restricted_number_of_designs_t_v(database &D, btree &B, 
	INT btree_idx_tvkl, INT t, INT v, INT first, INT len)
{
	INT verbose_level = 0;
	design_parameter p;
	INT k, k_min, k_max;
	INT nb_restricted = 0;
	
	D.ith_object(first, btree_idx_tvkl, p, verbose_level - 2);
	k_min = p.K();
	D.ith_object(first + len - 1, btree_idx_tvkl, p, verbose_level - 2);
	k_max = p.K();

	for (k = k_min; k <= k_max; k++) {
		INT first, len;
		
		B.search_interval_INT4_INT4_INT4(t, t, v, v, k, k, first, len, verbose_level);
		if (len == 0)
			continue;
				
		discreta_base lambda_max, lambda_max_half;
		design_lambda_max(t, v, k, lambda_max);
		design_lambda_max_half(t, v, k, lambda_max_half);
		// cout << "t=" << t << " v=" << v << " k=" << k << " lambda_max=" << lambda_max << endl;
		// INT delta_lambda = calc_delta_lambda(v, t, k, FALSE);

		INT l = 0;
		for (INT i = 0; i < len; i++) {
			INT idx = first + i;
			D.ith_object(idx, btree_idx_tvkl, p, verbose_level - 2);
					
			if (p.lambda().s_i_i() > lambda_max_half.s_i_i())
				continue;
			l++;
			} // next i
		nb_restricted += l;
		}
	return nb_restricted;
}

void prepare_design_parameters_from_id(database &D, INT id, hollerith& h)
{
	INT verbose_level = 0;
	INT btree_idx_id = 0;
	design_parameter p;
	
	D.get_object_by_unique_INT4(btree_idx_id, id, p, verbose_level);
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

void prepare_link(hollerith& link, INT id)
{
	INT page = id / 100;
	link.init("design_id_ge_");
	link.append_i(page * 100);
	link.append(".html#design");
	link.append_i(id);
}

#include <stdio.h>

void design_parameter_database_clans(BYTE *path_db, INT f_html, INT f_v, INT f_vv)
{
	INT verbose_level = 0;
	INT btree_idx_id = 0;
	//INT btree_idx_tvkl = 2;
	
	design_parameter p, q;
	database D;
	Vector ancestor, clan_lambda, clan_member, clan_member_path;
	
	p.init_database(D, path_db);
	D.open(verbose_level);

	INT id, highest_id, idx1, idx2;
	
	highest_id = D.get_highest_INT4(btree_idx_id);

	ancestor.m_l(0);
	clan_lambda.m_l(0);
	clan_member.m_l(0);
	clan_member_path.m_l(0);
	for (id = 0; id <= highest_id; id++) {

		if (!D.get_object_by_unique_INT4_if_there(btree_idx_id, id, p, verbose_level))
			continue;
		
				
		discreta_base lambda_max_half;
		design_lambda_max_half(p.t(), p.v(), p.K(), lambda_max_half);
		if (p.lambda().s_i_i() > lambda_max_half.s_i_i())
			continue;
		
		
		Vector g, path;
		p.ancestor(q, path, f_v, f_vv);
		
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
	
	INT i, l, j, ll, h, lll;
	l = ancestor.s_l();
	cout << "there are " << l << " clans of design parameter sets:" << endl;
	for (i = 0; i < l; i++) {
		cout << "clan no " << i << " : ancestor = " << ancestor[i];
		Vector &g = ancestor[i].as_vector();
		INT t = g.s_ii(0);
		INT v = g.s_ii(1);
		INT k = g.s_ii(2);
		
		INT delta_lambda = calc_delta_lambda(v, t, k, FALSE);
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
		INT t = g.s_ii(0);
		INT v = g.s_ii(1);
		INT k = g.s_ii(2);
		
		INT delta_lambda = calc_delta_lambda(v, t, k, FALSE);
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
			CL[j].integral_division_exact(dl, q);
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
		INT t = g.s_ii(0);
		INT v = g.s_ii(1);
		INT k = g.s_ii(2);
		
		INT delta_lambda = calc_delta_lambda(v, t, k, FALSE);
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

void design_parameter_database_family_report(BYTE *path_db, INT t, INT v, INT k, INT lambda, INT minimal_t)
{
	INT verbose_level = 0;
	// INT btree_idx_id = 0;
	INT btree_idx_tvkl = 2;
	
	cout << "design_parameter_database_family_report() t=" << t << " v=" << v << " k=" << k << " lambda=" << lambda << endl;
	design_parameter p;
	Vector Layers;
	
	database D;
	
	p.init_database(D, path_db);
	D.open(verbose_level);
	
	btree& B_tvkl = D.btree_access_i(btree_idx_tvkl);
	
	INT h, i, j, idx, id;
	
	Layers.m_l(t + 1);
	for (h = 0; h <= t; h++) {
		Layers[h].change_to_matrix();
		Layers[h].as_matrix().m_mn(h + 1, h + 1);
		}
	
	
	for (h = 0; h < t; h++) {
		if (t - h < minimal_t)
			continue;
		// cout << "h=" << h << endl;
		matrix &M = Layers[h].as_matrix();
		for (i = 0; i <= h; i++) {
			for (j = 0; j <= h - i; j++) {
				Vector entry;

				prepare_entry(entry, i, j, h, t, v, k, lambda);
				id = -1;
				if (entry.s_i(3).s_kind() == INTEGER) {
					idx = B_tvkl.search_unique_INT4_INT4_INT4_INT4(entry.s_ii(0), entry.s_ii(1), entry.s_ii(2), entry.s_ii(3), verbose_level);
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
		matrix &M = Layers[h].as_matrix();
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

static void prepare_entry(Vector &entry, INT i, INT j, INT h, INT t, INT v, INT k, INT lambda)
{
	design_parameter p, q;
	
	INT h1 = h - i - j, u;
	if (h1 < 0) {
		cout << "prepare_entry() h1 < 0" << endl;
		exit(1);
		}
	
	p.init(t, v, k, lambda);
	for (u = 0; u < i; u++) {
		p.derived(q);
		p.swap(q);
		}
	for (u = 0; u < j; u++) {
		p.residual(q);
		p.swap(q);
		}
	for (u = 0; u < h1; u++) {
		p.reduced_t(q);
		p.swap(q);
		}
	entry.m_l(5);
	entry.m_ii(0, p.t());
	entry.m_ii(1, p.v());
	entry.m_ii(2, p.K());
	entry[3] = p.lambda();
	entry.m_ii(4, -1);
}

void design_parameter_database_clan_report(BYTE *path_db, Vector &ancestor, Vector &clan_lambda, Vector & clan_member, Vector & clan_member_path)
{
	INT verbose_level = 0;
	INT btree_idx_id = 0;
	//INT btree_idx_tvkl = 2;
	
	design_parameter p, q;
	database D;
	
	p.init_database(D, path_db);
	D.open(verbose_level);

	INT highest_id;
	
	highest_id = D.get_highest_INT4(btree_idx_id);

	hollerith fname, fname_tex, fname_dir, h1, h2;

	fname_dir.init("design_clans.html");
	ofstream fhtml_dir(fname_dir.s());
			
			
	h1.init("t designs with small t by clans");
	h2.init("t designs with small t by clans");
		
	html_head(fhtml_dir, h1.s(), h2.s());	

	fhtml_dir << "in brackets: number of families / overall number of design parameter sets per clan<br>" << endl;

	fhtml_dir << "<ul>" << endl;
	INT i, j, l, ll, s, lll;
	l = ancestor.s_l();
	for (i = 0; i < l; i++) {
		Vector &a = ancestor[i].as_vector();
		INT t = a.s_ii(0);
		INT v = a.s_ii(1);
		INT k = a.s_ii(2);
		INT delta_lambda = calc_delta_lambda(v, t, k, FALSE);
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
		
		fhtml_dir << "<a href=\"design_clan_" << t << "_" << v << "_" << k << ".html\">";
		fhtml_dir << t << "-(" << v << "," << k << "," << "m*" << delta_lambda << ")</a>, 1 <= m <= " << m_max 
			<< "; (" << ll << "/" << s << ") lambda_max=" << lambda_max << ", lambda_max_half=" << lambda_max_half 
			<< "<br>" << endl;
		}
	fhtml_dir << "</ul>" << endl;
	html_foot(fhtml_dir);
	
	
	for (i = 0; i < l; i++) {

		Vector &a = ancestor[i].as_vector();
		INT t = a.s_ii(0);
		INT v = a.s_ii(1);
		INT k = a.s_ii(2);
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


		INT delta_lambda = calc_delta_lambda(v, t, k, FALSE);
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
		fhtml << t << "-(" << v << "," << k << "," << "m*" << delta_lambda << "), 1 <= m <= " << m_max 
			<< "; (" << ll << "/" << s << ") lambda_max=" << lambda_max << ", lambda_max_half=" << lambda_max_half 
			<< "<br>" << endl;
		ftex << "\\subsection*{Clan " << i << ": $" << t << "$-$(" << v << "," << k 
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
			fhtml << "<li>family " << j << ", lambda = " << lambda << " containing " << lll << " designs:" << endl;
			fhtml << "<br>" << endl;
			ftex << "\\subsubsection*{Family with $\\lambda=" << lambda << "$}" << endl;
			ftex << "The family contains " << lll << " design parameter sets:\\\\" << endl;
#if 0
			INT h;
			for (h = 0; h < lll; h++) {
				hollerith link, text1;
				INT id = cm.s_ii(h);
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
			INT max_depth, minimal_t;
		
			determine_minimal_and_maximal_path(cmp, min_path, max_path, max_depth);
			minimal_t = t - max_depth;
			
			fhtml << "<br>minpath=" << min_path << " minimal_t=" << minimal_t << endl;
			design_parameter dominating_ancestor;
			determine_dominating_ancestor(t, v, k, lambda, min_path, dominating_ancestor);
			// fhtml << "<br>dominating_ancestor: " << dominating_ancestor << " (path=" << min_path << ")" << endl;
			reduce_path(cmp, min_path);
			family_report(D, fhtml, ftex, dominating_ancestor.t(), dominating_ancestor.v(), dominating_ancestor.K(), dominating_ancestor.lambda(), cm, cmp, minimal_t);
			}		
		fhtml << "</ul>" << endl;

		html_foot(fhtml);
		}
	
	
	
	D.close(verbose_level);
}

static void determine_minimal_and_maximal_path(Vector &v, Vector & min_path, Vector & max_path, INT & max_depth)
{
	INT i, l, j, ll, depth;
	
	l = v.s_l();
	if (l == 0) {
		cout << "determine_minimal_and_maximal_path() l == 0" << endl;
		exit(1);
		}
	ll = v[0].as_vector().s_l();
	min_path = v[0];
	max_path = v[0];
	max_depth = 0;
	for (i = 0; i < l; i++) {
		Vector & p = v[i].as_vector();
		if (p.s_l() != ll) {
			cout << "determine_minimal_and_maximal_path() different lengths!" << endl;
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

static void determine_dominating_ancestor(INT t, INT v, INT k, discreta_base & lambda, Vector & path, design_parameter &dominating_ancestor)
{
	design_parameter p, q;
	INT u;
	
	p.init(t, v, k, lambda);
	for (u = 0; u < path.s_ii(0); u++) {
		p.reduced_t(q);
		p.swap(q);
		}
	for (u = 0; u < path.s_ii(1); u++) {
		p.derived(q);
		p.swap(q);
		}
	for (u = 0; u < path.s_ii(2); u++) {
		p.residual(q);
		p.swap(q);
		}
	dominating_ancestor = p;
}

static void reduce_path(Vector &cmp, Vector &min_path)
{
	INT i, l, j;
	
	l = cmp.s_l();
	for (i = 0; i < l; i++) {
		Vector &path = cmp[i].as_vector();
		for (j = 0; j < 3; j++) {
			path.s_ii(j) -= min_path.s_ii(j);
			}
		}
}

static void family_report(database & D, ostream& fhtml, ostream &ftex, INT t, INT v, INT k, discreta_base &lambda, Vector & cm, Vector & cmp, INT minimal_t)
{
	INT h, i, j, idx, idx1, id, nb_found = 0;
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
		matrix &M = Layers[h].as_matrix();
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
		matrix &M = Layers[h].as_matrix();
		for (i = 0; i <= h; i++) {
			for (j = 0; j <= h - i; j++) {
				INT id = M.s_iji(i, j);
				Vector path;
				
				path.m_l_n(3);
				path.m_ii(0, h - i - j);
				path.m_ii(1, i);
				path.m_ii(2, j);
				design_parameter p;
				determine_dominating_ancestor(t, v, k, lambda, path, p);
				if (id >= 0) {
					hollerith link, text1;
					
					prepare_link(link, id);
					fhtml << " <a href=\"" << link.s() << "\">";
					prepare_design_parameters_from_id(D, id, text1);
					fhtml << text1.s() << " (#" << id << ")</a> ";
					ftex << "$\\underline{\\mbox{" << text1.s() << "}}$";
					}
				else {
					fhtml << p.t() << "-(" << p.v() << "," << p.K() << "," << p.lambda() << ") ";
					ftex << "$" << p.t() << "$-$(" << p.v() << "," << p.K() << "," << p.lambda() << ")$";
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

static void f_m_j(INT m, INT j, discreta_base &a)
{
	INT q = m / j;
	INT r = m % j;
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
	d.mult(b, c);
	d.integral_division_exact(two, c);
	J.m_i_i(j);
	c *= J;
	R.m_i_i(r);
	R *= b;
	c += R;
	a = c;
}

static INT max_m(INT i, INT j)
{
	INT m;
	discreta_base a, b, c, d, two;
	
	two.m_i_i(2);
	b.m_i_i(i);
	c.m_i_i(i - 1);
	d.mult(b, c);
	d.integral_division_exact(two, a);
	for (m = 0; ; m++) {
		f_m_j(m, j, b);
		if (b.gt(a)) {
			return m - 1;
			}
		}
}

INT Maxfit(INT i, INT j)
{
	INT a, b, c;
	
	a = max_m(i, j);
	b = max_m(j, i);
	c = MINIMUM(a, b);
	return c;
}

#if 0

void create_all_masks(char *label, 
	int nb_row_partitions, char *row_partitions[], 
	int nb_col_partitions, char *col_partitions[])
{
	INT ci, cj;
	INT no;
	matrix number_of_masks;

	number_of_masks.m_mn_n(nb_row_partitions, nb_col_partitions);
	for (ci = 0; ci < nb_row_partitions; ci++) {
		for (cj = 0; cj < nb_col_partitions; cj++) {
			no = create_masks(label, 
				nb_row_partitions, row_partitions, 
				nb_col_partitions, col_partitions, ci, cj);
			number_of_masks.m_iji(ci, cj, no);
			}
		}
	cout << "number of masks:" << endl;
	cout << number_of_masks;
	
}

	
INT create_masks(char *label, 
	int nb_row_partitions, char *row_partitions[], 
	int nb_col_partitions, char *col_partitions[], 
	int ci, int cj)
{
	INT no = 0;
	geo_generate G;
	Vector v;
	Vector aut_generators;
	base ago;
	INT i, j, l, ll, h, idx, first, jj, l1, l2;
	Vector prev_row, prev_col;
	hollerith fname;
	matrix M;


	fname.init(label);
	fname.append("_");
	fname.append_i(ci + 1);
	fname.append("_");
	fname.append_i(cj + 1);
	fname.append(".masks");
		
	ofstream f(fname.s());
	G.init_from_string(row_partitions[ci], col_partitions[cj]);
		
	if (G.generate_first()) {
		while (TRUE) {
			no++;
			cout << no << endl;
			G.print();
			
			f << "mask " << no << endl;
			f << G.nb_rows << " " << G.nb_cols << endl;
			G.print(f);
			
			G.canonical_form(v, aut_generators, FALSE /* f_v */);
			perm_group A(aut_generators);
			A.group_order(ago);
			cout << "ago=" << ago << endl;
			f << aut_generators.s_l() << " " << G.nb_rows + G.nb_cols << endl;
			for (h = 0; h < aut_generators.s_l(); h++) {
				for (i = 0; i < G.nb_rows + G.nb_cols; i++) {
					j = aut_generators[h].as_permutation().s_ii(i);
					f << j << " ";
					}
				f << endl;
				}
			
			prev_row.m_l_n(G.nb_rows);
			prev_col.m_l_n(G.nb_cols);
			for (h = 0; h < G.nb_rows; h++)
				prev_row.m_ii(h, -1);
			for (h = 0; h < G.nb_cols; h++)
				prev_col.m_ii(h, -1);
			
			cout << "orbits:" << endl;
			l = A.tidx().s_l();
			for (i = 0; i < l; i++) {
				idx = A.tidx().s_ii(i);
				ll = A.Ti(idx).s_l();
				for (h = 0; h < ll; h++) {
					j = A.Tij(idx, h);
					if (h == 0) {
						first = j;
						if (first >= G.nb_rows)
							first -= G.nb_rows;
						}
					else {
						if (j >= G.nb_rows) {
							jj = j - G.nb_rows;
							prev_col.m_ii(jj, first);
							}
						else {
							prev_row.m_ii(j, first);
							}
						}
					cout << j << " ";
					}
				cout << endl;
				}
			cout << "prev_row: " << prev_row << endl;
			cout << "prev_col: " << prev_col << endl;
			l1 = 0;
			for (h = 0; h < G.nb_rows; h++) {
				if (prev_row.s_ii(h) != -1)
					l1++;
				}
			l2 = 0;
			for (h = 0; h < G.nb_cols; h++) {
				if (prev_col.s_ii(h) != -1)
					l2++;
				}
			f << l1 << endl;
			for (h = 0; h < G.nb_rows; h++) {
				if (prev_row.s_ii(h) != -1) {
					f << prev_row.s_ii(h) << " " << h << endl;
					}
				}
			f << l2 << endl;
			for (h = 0; h < G.nb_cols; h++) {
				if (prev_col.s_ii(h) != -1) {
					f << prev_col.s_ii(h) << " " << h << endl;
					}
				}
			f << endl;
			f << endl;
			
			G.get_reduced_incidence_matrix(M);
			cout << "reduced mask:" << endl << M << endl;
			{
				hollerith hh;
				INT f_row_decomp = FALSE;
				INT f_col_decomp = FALSE;
				Vector row_decomp, col_decomp;
				INT f_labelling_points = FALSE;
				INT f_labelling_blocks = FALSE;
				Vector point_labels, block_labels;
				
				hh.init(label);
				hh.append("_");
				hh.append_i(ci + 1);
				hh.append("_");
				hh.append_i(cj + 1);
				hh.append("_rm_");
				hh.append_i(no);
				hh.append(".tex");
				ofstream ff(hh.s());
				M.incma_print_latex(ff, 
					f_row_decomp, row_decomp, 
					f_col_decomp, col_decomp, 
					f_labelling_points, point_labels, 
					f_labelling_blocks, block_labels);
			}
			
			if (!G.generate_next())
				break;
			}
		}
	cout << fname << " : alltogether " << no << " masks" << endl;
	return no;
}
#endif

#if 0
void orbits_in_product_action(INT n1, INT n2, INT f_v, INT f_vv)
{
	char * s[] = { "C", "", "C", "", "X" };
	char s_n1[100];
	char s_n2[100];
	
	itoa(s_n1, 100, n1);
	itoa(s_n2, 100, n2);
	s[1] = s_n1;
	s[3] = s_n2;
	
	Vector gsel, gen;
	hollerith group_label, group_label_tex, acting_on;
	
	compose_gsel_from_strings(gsel, sizeof(s) / sizeof(char *), s);
	compose_group(gsel, gen, group_label, group_label_tex, acting_on, f_v);
	cout << "group " << group_label << ", " << group_label_tex << endl;
	cout << gen << endl;
	
	INT f_cyclic_notation = TRUE;
	write_file_of_generators_xml_group_label(gen, group_label.s(), f_cyclic_notation);
	write_file_of_generators_group_label(gen, group_label.s());
	write_file_of_generators_gap_group_label(gen, group_label.s());

	
	prepare_2_orbits_in_product_action(group_label.s(), gen, n1, n2, f_v, f_vv);
	
}

void orbits_in_product_action_D_CC(INT n1, INT p1, INT p2, INT f_v, INT f_vv)
{
	char * s[] = { "D", "", "C", "", "C", "", "X", "X" };
	char s_n1[100];
	char s_p1[100];
	char s_p2[100];
	
	itoa(s_n1, 100, n1);
	itoa(s_p1, 100, p1);
	itoa(s_p2, 100, p2);
	s[1] = s_n1;
	s[3] = s_p1;
	s[5] = s_p2;
	
	Vector gsel, gen;
	hollerith group_label, group_label_tex, acting_on;
	
	compose_gsel_from_strings(gsel, sizeof(s) / sizeof(char *), s);
	compose_group(gsel, gen, group_label, group_label_tex, acting_on, f_v);
	cout << "group " << group_label << ", " << group_label_tex << endl;
	cout << gen << endl;
	
	INT f_cyclic_notation = TRUE;
	write_file_of_generators_xml_group_label(gen, group_label.s(), f_cyclic_notation);
	write_file_of_generators_group_label(gen, group_label.s());
	write_file_of_generators_gap_group_label(gen, group_label.s());

	prepare_2_orbits_in_product_action(group_label.s(), gen, n1, p1 * p2, f_v, f_vv);
}

void orbits_in_product_action_CC_D(INT p1, INT p2, INT n2, INT f_v, INT f_vv)
{
	char * s[] = { "C", "", "C", "", "X", "D", "", "X" };
	char s_p1[100];
	char s_p2[100];
	char s_n2[100];
	
	itoa(s_p1, 100, p1);
	itoa(s_p2, 100, p2);
	itoa(s_n2, 100, n2);
	s[1] = s_p1;
	s[3] = s_p2;
	s[6] = s_n2;
	
	Vector gsel, gen;
	hollerith group_label, group_label_tex, acting_on;
	
	compose_gsel_from_strings(gsel, sizeof(s) / sizeof(char *), s);
	compose_group(gsel, gen, group_label, group_label_tex, acting_on, f_v);
	cout << "group " << group_label << ", " << group_label_tex << endl;
	cout << gen << endl;
	
	INT f_cyclic_notation = TRUE;
	write_file_of_generators_xml_group_label(gen, group_label.s(), f_cyclic_notation);
	write_file_of_generators_group_label(gen, group_label.s());
	write_file_of_generators_gap_group_label(gen, group_label.s());

	prepare_2_orbits_in_product_action(group_label.s(), gen, p1 * p2, n2, f_v, f_vv);
}

void orbits_in_product_action_extended(INT q1, INT q2, INT u, INT v, INT f_v, INT f_vv)
{
	Vector gen;
	hollerith label;
	INT f_write_generators_to_file = TRUE;
	vec_generators_q1_q2_aubv(q1, q2, u, v, gen, label, f_write_generators_to_file, f_v, f_vv);
	
	prepare_2_orbits_in_product_action(label.s(), gen, q1, q2, f_v, f_vv);
}

void orbits_in_product_action_extended_twice(INT q1, INT q2, INT u1, INT v1, INT u2, INT v2, 
	INT f_cycle_index, INT f_cycle_index_on_pairs, INT f_v, INT f_vv)
{
	hollerith label;
	Vector gen;
	INT f_write_generators_to_file = TRUE;
	
	cout << "design.C: orbits_in_product_action_extended_twice" << endl;
	
	// perm_group_gens.C:
	vec_generators_q1_q2_au1bv1_au2bv2(q1, q2, 
		u1, v1, u2, v2, gen, label, f_write_generators_to_file, f_v, f_vv);

	cout << "after vec_generators_q1_q2_au1bv1_au2bv2()" << endl;

	// orbit.C:
	cout << "calling prepare_2_orbits_in_product_action" << endl;
	
	prepare_2_orbits_in_product_action(label.s(), gen, q1, q2, f_v, f_vv);

#if 0

	if (f_cycle_index) {
		Vector C, C2;
		vec_generators_cycle_index(gen, C, TRUE);
		
		if (f_cycle_index_on_pairs) {
			cycle_index_on_pairs(C, C2, TRUE);
			}
		}
#endif

}

void extract_subgroup(INT q1, INT q2, INT u1, INT v1, INT f_cycle_index)
{
	Vector gen, gen1;
	INT f_v = FALSE;
	INT f_vv = FALSE;
	hollerith label;
	INT f_write_generators_to_file = FALSE;
	vec_generators_q1_q2_aubv(q1, q2, u1, v1, gen, label, f_write_generators_to_file, f_v, f_vv);
	INT l;
	
	gen1.m_l(1);
	l = gen.s_l();
	gen1[0] = gen[l - 2];
	gen1.Print(cout);

	hollerith h;
	h.init("a");
	h.append_i(u1);
	h.append("_b");
	h.append_i(v1);
	h.append(".txt");
	write_file_of_generators(gen1, h.s());
	
#if 0
	if (f_cycle_index) {
		Vector C;
		vec_generators_cycle_index(gen1, C, TRUE);
		cout << endl;
		}
#endif
}
#endif


