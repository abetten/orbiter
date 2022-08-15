/*
 * coding_theory_domain.cpp
 *
 *  Created on: Apr 21, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace coding_theory {


static void CRC_BCH256_771_divide(const char *in, char *out);


coding_theory_domain::coding_theory_domain()
{

}

coding_theory_domain::~coding_theory_domain()
{

}







void coding_theory_domain::make_mac_williams_equations(ring_theory::longinteger_object *&M,
		int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "coding_theory_domain::make_mac_williams_equations" << endl;
	}
	M = NEW_OBJECTS(ring_theory::longinteger_object, (n + 1) * (n + 1));

	for (i = 0; i <= n; i++) {
		for (j = 0; j <= n; j++) {
			Combi.krawtchouk(M[i * (n + 1) + j], n, q, i, j);
		}
	}

	{


		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		snprintf(str, 1000, "MacWilliams_n%d_k%d_q%d.tex", n, k, q);
		fname.assign(str);
		snprintf(str, 1000, "MacWilliams System for a $[%d,%d]_{%d}$ code", n, k, q);
		title.assign(str);


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "coding_theory_domain::make_mac_williams_equations "
						"before print_longinteger_matrix_tex" << endl;
			}

			orbiter_kernel_system::latex_interface Li;

			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.print_longinteger_matrix_tex(ost, M, n + 1, n + 1);
			ost << "\\right]" << endl;
			ost << "$$" << endl;

			if (f_v) {
				cout << "coding_theory_domain::make_mac_williams_equations "
						"after print_longinteger_matrix_tex" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "coding_theory_domain::make_mac_williams_equations done" << endl;
	}
}

void coding_theory_domain::make_table_of_bounds(
		int n_max, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n, k, d_S, d_H, d_P, d_G, d_GV;

	if (f_v) {
		cout << "coding_theory_domain::make_table_of_bounds" << endl;
	}
	vector<vector<long int>> Table;

	for (n = 2; n <= n_max; n++) {
		for (k = 1; k <= n; k++) {
			cout << "n=" << n << " k=" << k << " q=" << q << endl;
			d_S = singleton_bound_for_d(n, k, q, 0 /*verbose_level*/);
			cout << "d_S=" << d_S << endl;
			d_H = hamming_bound_for_d(n, k, q, 0 /*verbose_level*/);
			cout << "d_H=" << d_H << endl;
			d_P = plotkin_bound_for_d(n, k, q, 0 /*verbose_level*/);
			cout << "d_P=" << d_P << endl;
			d_G = griesmer_bound_for_d(n, k, q, 0 /*verbose_level*/);
			cout << "d_G=" << d_G << endl;
			d_GV = gilbert_varshamov_lower_bound_for_d(n, k, q, 0 /*verbose_level*/);
			cout << "d_GV=" << d_GV << endl;
			vector<long int> entry;

			entry.push_back(n);
			entry.push_back(k);
			entry.push_back(q);
			entry.push_back(d_GV);
			entry.push_back(d_S);
			entry.push_back(d_H);
			entry.push_back(d_P);
			entry.push_back(d_G);
			Table.push_back(entry);
		}
	}
	long int *T;
	int N;
	int i, j;
	int nb_cols = 8;

	N = Table.size();

	T = NEW_lint(N * nb_cols);
	for (i = 0; i < N; i++) {
		for (j = 0; j < nb_cols; j++) {
			T[i * nb_cols + j] = Table[i][j];
		}
	}
	orbiter_kernel_system::file_io Fio;
	std::string fname;
	char str[1000];

	sprintf(str, "_n%d_q%d", n_max, q);

	fname.assign("table_of_bounds");
	fname.append(str);
	fname.append(".csv");

	string *headers;

	headers = new string[8];

	headers[0].assign("n");
	headers[1].assign("k");
	headers[2].assign("q");
	headers[3].assign("GV");
	headers[4].assign("S");
	headers[5].assign("H");
	headers[6].assign("P");
	headers[7].assign("G");

	Fio.lint_matrix_write_csv_override_headers(fname, headers, T, N, nb_cols);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	FREE_lint(T);
	delete [] headers;


	if (f_v) {
		cout << "coding_theory_domain::make_table_of_bounds done" << endl;
	}
}

void coding_theory_domain::make_gilbert_varshamov_code(
		int n, int k, int d,
		field_theory::finite_field *F,
		int *&genma, int *&checkma,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code" << endl;
	}
	geometry::geometry_global Gg;
	int nmk;
	int N_points;
	long int *set;
	int *f_forbidden;


	nmk = n - k;

	set = NEW_lint(n);

	N_points = Gg.nb_PG_elements(nmk - 1, F->q);

	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code N_points = " << N_points << endl;
	}

	f_forbidden = NEW_int(N_points);
	Int_vec_zero(f_forbidden, N_points);

	make_gilbert_varshamov_code_recursion(F,
			n, k, d, N_points,
			set, f_forbidden, 0 /*level*/,
			verbose_level);



	cout << "coding_theory_domain::make_gilbert_varshamov_code found "
			"the following parity check matrix as projective set: ";
	Lint_vec_print(cout, set, n);
	cout << endl;

	int *M;
	M = NEW_int(n * n);

	matrix_from_projective_set(F,
			n, nmk, set,
			genma,
			verbose_level);

	cout << "coding_theory_domain::make_gilbert_varshamov_code parity check matrix:" << endl;
	Int_matrix_print(M, nmk, n);

	cout << "coding_theory_domain::make_gilbert_varshamov_code parity check matrix:" << endl;
	Int_vec_print_fully(cout, M, nmk * n);
	cout << endl;

	F->Linear_algebra->RREF_and_kernel(n, nmk, M, 0 /* verbose_level */);

	cout << "coding_theory_domain::make_gilbert_varshamov_code generator matrix:" << endl;
	Int_matrix_print(M + nmk * n, k, n);


	cout << "coding_theory_domain::make_gilbert_varshamov_code generator matrix:" << endl;
	Int_vec_print_fully(cout, M + nmk * n, k * n);
	cout << endl;


	genma = NEW_int(k * n);
	checkma = NEW_int(nmk * n);

	Int_vec_copy(M + nmk * n, genma, k * n);
	Int_vec_copy(M, checkma, nmk * n);


	//FREE_int(M);


	FREE_lint(set);
	FREE_int(f_forbidden);
	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code done" << endl;
	}
}

void coding_theory_domain::make_gilbert_varshamov_code_recursion(
		field_theory::finite_field *F,
		int n, int k, int d, long int N_points,
		long int *set, int *f_forbidden, int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code level = " << level << endl;
		cout << "coding_theory_domain::make_gilbert_varshamov_code set = ";
		Lint_vec_print(cout, set, level);
		cout << endl;
	}

	if (level == n) {
		cout << "done" << endl;
		return;
	}
	int a, b, i;

	for (a = 0; a < N_points; a++) {

		if (!f_forbidden[a]) {
			break;
		}
	}

	if (a == N_points) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code_recursion "
				"failure to construct the code" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code_recursion "
				"picking a=" << a << endl;
	}


	vector<int> add_set;
	int nmk;
	combinatorics::combinatorics_domain Combi;
	int cnt;

	nmk = n - k;
	set[level] = a;
	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code nmk = " << nmk << endl;
	}

	f_forbidden[a] = TRUE;
	add_set.push_back(a);
	cnt = 1;


	if (level) {

		int *subset;
		subset = NEW_int(level - 1);
		int s, N, h, u, c, e, t, f;
		int *v1;
		//int *v2;
		int *v3;

		v1 = NEW_int(nmk);
		//v2 = NEW_int(nmk);
		v3 = NEW_int(nmk);

		s = MINIMUM(level, d - 2);


		for (i = 1; i <= s; i++) {
			N = Combi.binomial_lint(level, i);
			if (f_v) {
				cout << "coding_theory_domain::make_gilbert_varshamov_code "
						"N_" << i << " = " << N << endl;
				cout << "set = ";
				Lint_vec_print(cout, set, level + 1);
				cout << endl;
				cout << "looping over all subsets of size " << i << ":" << endl;
			}
			for (h = 0; h < N; h++) {
				Combi.unrank_k_subset(h, subset, level, i);
				Int_vec_zero(v3, nmk);
				for (u = 0; u < i; u++) {
					c = subset[u];
					e = set[c];
					F->PG_element_unrank_modified(v1, 1, nmk, e);
					//P->unrank_point(v1, e);
					for (t = 0; t < nmk; t++) {
						v3[t] = F->add(v3[t], v1[t]);
					}
				}
				F->PG_element_unrank_modified(v1, 1, nmk, set[level]);
				//P->unrank_point(v1, set[level]);
				for (t = 0; t < nmk; t++) {
					v3[t] = F->add(v3[t], v1[t]);
				}
				F->PG_element_rank_modified(v3, 1, nmk, f);
				//f = P->rank_point(v3);
				if (f_v) {
					cout << "h=" << h << " / " << N << " : ";
					Int_vec_print(cout, subset, i);
					cout << " : ";
					Int_vec_print(cout, v3, nmk);
					cout << " : " << f;
				}
				if (!f_forbidden[f]) {
					f_forbidden[f] = TRUE;
					add_set.push_back(f);
					cnt++;
					if (f_v) {
						cout << " : is new forbidden point " << cnt;
					}
				}
				if (f_v) {
					cout << endl;
				}
			}
			//cout << endl;
		}
		FREE_int(subset);
		FREE_int(v1);
		//FREE_int(v2);
		FREE_int(v3);
	}
	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"level = " << level << " : cnt = " << cnt << " calling the recursion:" << endl;
	}
	make_gilbert_varshamov_code_recursion(F, n, k, d, N_points,
			set, f_forbidden, level + 1, verbose_level);
	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"level = " << level << " : cnt = " << cnt << " done with the recursion:" << endl;
	}

	for (i = 0; i < add_set.size(); i++) {
		b = add_set[i];
		f_forbidden[b] = FALSE;
	}


	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code done" << endl;
	}
}




int coding_theory_domain::gilbert_varshamov_lower_bound_for_d(int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	combinatorics::combinatorics_domain Combi;
	int i, d;
	ring_theory::longinteger_object qnmk, qm1, qm1_power, S, s, a, b;

	if (f_v) {
		cout << "coding_theory_domain::gilbert_varshamov_lower_bound_for_d" << endl;
	}
	qnmk.create(q, __FILE__, __LINE__);
	qm1.create(q - 1, __FILE__, __LINE__);
	D.power_int(qnmk, n - k);
	qm1_power.create(1, __FILE__, __LINE__);
	S.create(0, __FILE__, __LINE__);
	//cout << "gilbert_varshamov_lower_bound_for_d: q=" << q << " n=" << n << " k=" << k << " " << q << "^" << n - k << " = " << qnmk << endl;
	for (i = 0; ; i++) {
		Combi.binomial(b, n - 1, i, FALSE);
		D.mult(b, qm1_power, s);
		D.add(S, s, a);
		a.assign_to(S);
		if (D.compare(S, qnmk) >= 0) {
			d = i + 1;
			//cout << "S=" << S << " d=" << d << endl;
			break;
		}
		//cout << "i=" << i << " S=" << S << " is OK" << endl;
		D.mult(qm1_power, qm1, s);
		s.assign_to(qm1_power);
	}
	if (f_v) {
		cout << "coding_theory_domain::gilbert_varshamov_lower_bound_for_d done" << endl;
	}
	return d;
}


int coding_theory_domain::singleton_bound_for_d(
		int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d;

	if (f_v) {
		cout << "coding_theory_domain::singleton_bound_for_d" << endl;
	}
	d = n - k + 1;
	return d;
}


int coding_theory_domain::hamming_bound_for_d(
		int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int e, d, t;
	ring_theory::longinteger_object qnmk, qm1, qm1_power, B, s, a, b;
	ring_theory::longinteger_domain D;
	combinatorics::combinatorics_domain Combi;


	if (f_v) {
		cout << "coding_theory_domain::hamming_bound_for_d" << endl;
	}
	qnmk.create(q, __FILE__, __LINE__);
	qm1.create(q - 1, __FILE__, __LINE__);
	D.power_int(qnmk, n - k);
	qm1_power.create(1, __FILE__, __LINE__);
	B.create(0, __FILE__, __LINE__);
	if (f_vv) {
		cout << "coding_theory_domain::hamming_bound_for_d: "
			"q=" << q << " n=" << n << " k=" << k << " "
			<< q << "^" << n - k << " = " << qnmk << endl;
	}
	for (e = 0; ; e++) {
		Combi.binomial(b, n, e, FALSE);
		D.mult(b, qm1_power, s);
		D.add(B, s, a);
		a.assign_to(B);
		if (D.compare(B, qnmk) == 1) {
			// now the size of the Ball of radius e is bigger than q^{n-m}
			t = e - 1;
			d = 2 * t + 2;
			if (f_vv) {
				cout << "B=" << B << " t=" << t << " d=" << d << endl;
			}
			break;
		}
		if (f_vv) {
			cout << "e=" << e << " B=" << B << " is OK" << endl;
		}
		D.mult(qm1_power, qm1, s);
		s.assign_to(qm1_power);
	}
	if (f_v) {
		cout << "coding_theory_domain::hamming_bound_for_d done" << endl;
	}
	return d;
}

int coding_theory_domain::plotkin_bound_for_d(
		int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int d;
	ring_theory::longinteger_object qkm1, qk, qm1, a, b, c, Q, R;
	ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "coding_theory_domain::plotkin_bound_for_d" << endl;
	}

	// d \le \frac{n q^{k-1}}{q^k-1}

	qkm1.create(q, __FILE__, __LINE__);
	D.power_int(qkm1, k - 1);
	a.create(n, __FILE__, __LINE__);
	D.mult(a, qkm1, b);
		// now b = n q^{k-1}

	a.create(q - 1, __FILE__, __LINE__);
	D.mult(b, a, c);
		// now c = n q^{k-1} (q - 1)


	a.create(q, __FILE__, __LINE__);
	D.mult(a, qkm1, qk);
		// now qk = q^k

	a.create(-1, __FILE__, __LINE__);
	D.add(qk, a, b);
		// now b = 2^k - 1

	if (f_vv) {
		cout << "coding_theory_domain::plotkin_bound_for_d "
				"q=" << q << " n=" << n << " k=" << k << endl;
	}
	D.integral_division(c, b, Q, R, FALSE /* verbose_level */);
	d = Q.as_int();
	if (f_vv) {
		cout << c << " / " << b << " = " << d << endl;
	}
	if (f_v) {
		cout << "coding_theory_domain::plotkin_bound_for_d" << endl;
	}
	return d;
}

int coding_theory_domain::griesmer_bound_for_d(
		int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d, n1;

	if (f_v) {
		cout << "coding_theory_domain::griesmer_bound_for_d" << endl;
	}
	for (d = 1; d <= n; d++) {
		n1 = griesmer_bound_for_n(k, d, q, verbose_level - 2);
		if (n1 > n) {
			d--;
			break;
			}
	}
	if (f_v) {
		cout << "coding_theory_domain::griesmer_bound_for_d done" << endl;
	}
	return d;
}

int coding_theory_domain::griesmer_bound_for_n(
		int k, int d, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, n;
	ring_theory::longinteger_object qq, qi, d1, S, Q, R, one, a, b;
	ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "coding_theory_domain::griesmer_bound_for_n" << endl;
	}
	one.create(1, __FILE__, __LINE__);
	d1.create(d, __FILE__, __LINE__);
	qq.create(q, __FILE__, __LINE__);
	qi.create(1, __FILE__, __LINE__);
	S.create(0, __FILE__, __LINE__);
	if (f_vv) {
		cout << "coding_theory_domain::griesmer_bound_for_n q=" << q
				<< " d=" << d << " k=" << k << endl;
	}
	for (i = 0; i < k; i++) {
		D.integral_division(d1, qi, Q, R, FALSE /* verbose_level */);
		if (!R.is_zero()) {
			D.add(Q, one, a);
			D.add(S, a, b);
		}
		else {
			D.add(S, Q, b);
		}
		b.assign_to(S);
		D.mult(qi, qq, a);
		a.assign_to(qi);
		if (f_vv) {
			cout << "i=" << i << " S=" << S << endl;
		}
	}
	n = S.as_int();
	if (f_v) {
		cout << "coding_theory_domain::griesmer_bound_for_n" << endl;
	}
	return n;
}


void coding_theory_domain::do_make_macwilliams_system(
		int q, int n, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	coding_theory_domain C;
	ring_theory::longinteger_object *M;
	int i, j;

	if (f_v) {
		cout << "interface_coding_theory::do_make_macwilliams_system" << endl;
	}

	C.make_mac_williams_equations(M, n, k, q, verbose_level);

	cout << "\\begin{array}{r|*{" << n << "}{r}}" << endl;
	for (i = 0; i <= n; i++) {
		for (j = 0; j <= n; j++) {
			cout << M[i * (n + 1) + j];
			if (j < n) {
				cout << " & ";
			}
		}
		cout << "\\\\" << endl;
	}
	cout << "\\end{array}" << endl;

	cout << "[";
	for (i = 0; i <= n; i++) {
		cout << "[";
		for (j = 0; j <= n; j++) {
			cout << M[i * (n + 1) + j];
			if (j < n) {
				cout << ",";
			}
		}
		cout << "]";
		if (i < n) {
			cout << ",";
		}
	}
	cout << "]" << endl;


	if (f_v) {
		cout << "coding_theory_domain::do_make_macwilliams_system done" << endl;
	}
}



void coding_theory_domain::make_Hamming_graph_and_write_file(int n, int q,
		int f_projective, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int width, height;
	int *v;
	int *w;
	int *Table;
	geometry::geometry_global Gg;
	field_theory::finite_field *F = NULL;

	if (f_v) {
		cout << "coding_theory_domain::make_Hamming_graph_and_write_file" << endl;
	}

	v = NEW_int(n);
	w = NEW_int(n);

	if (f_projective) {
		width = height = Gg.nb_PG_elements(n - 1, q);
		F = NEW_OBJECT(field_theory::finite_field);
		F->finite_field_init(q, FALSE /* f_without_tables */, 0 /* verbose_level */);
	}
	else {
		width = height = Gg.nb_AG_elements(n, q);
	}

#if 0
	int N;
	N = width;
	if (f_graph) {
		Adj = NEW_int(N * N);
		int_vec_zero(Adj, N * N);
	}
#endif

	cout << "width=" << width << endl;

	int i, j, d, h;

	Table = NEW_int(height * width);
	for (i = 0; i < height; i++) {

		if (f_projective) {
			F->PG_element_unrank_modified(v, 1 /*stride*/, n, i);
		}
		else {
			Gg.AG_element_unrank(q, v, 1, n, i);
		}

		for (j = 0; j < width; j++) {

			if (f_projective) {
				F->PG_element_unrank_modified(w, 1 /*stride*/, n, j);
			}
			else {
				Gg.AG_element_unrank(q, w, 1, n, j);
			}

			d = 0;
			for (h = 0; h < n; h++) {
				if (v[h] != w[h]) {
					d++;
				}
			}

#if 0
			if (f_graph && d == 1) {
				Adj[i * N + j] = 1;
			}
#endif

			Table[i * width + j] = d;

		}
	}

	string fname;
	char str[1000];
	orbiter_kernel_system::file_io Fio;

	sprintf(str, "Hamming_n%d_q%d.csv", n, q);
	fname.assign(str);

	Fio.int_matrix_write_csv(fname, Table, height, width);

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "coding_theory_domain::make_Hamming_graph_and_write_file" << endl;
	}

}


void coding_theory_domain::compute_and_print_projective_weights(
		ostream &ost, field_theory::finite_field *F, int *M, int n, int k)
{
	int i;
	int *weights;

	weights = NEW_int(n + 1);
	code_projective_weight_enumerator(F, n, k,
		M, // [k * n]
		weights, // [n + 1]
		0 /*verbose_level*/);


	ost << "projective weights: " << endl;
	for (i = 0; i <= n; i++) {
		if (weights[i] == 0) {
			continue;
		}
		ost << i << " : " << weights[i] << endl;
	}
	FREE_int(weights);
}

int coding_theory_domain::code_minimum_distance(field_theory::finite_field *F, int n, int k,
		int *code, int verbose_level)
	// code[k * n]
{
	int f_v = (verbose_level >= 1);
	int *weight_enumerator;
	int i;

	if (f_v) {
		cout << "coding_theory_domain::code_minimum_distance" << endl;
	}
	weight_enumerator = NEW_int(n + 1);
	Int_vec_zero(weight_enumerator, n + 1);

	code_weight_enumerator_fast(F, n, k,
		code, // [k * n]
		weight_enumerator, // [n + 1]
		verbose_level);
	for (i = 1; i <= n; i++) {
		if (weight_enumerator[i]) {
			break;
		}
	}
	if (i == n + 1) {
		cout << "coding_theory_domain::code_minimum_distance "
				"the minimum weight is undefined" << endl;
		exit(1);
	}
	FREE_int(weight_enumerator);
	return i;
}

void coding_theory_domain::codewords_affine(field_theory::finite_field *F, int n, int k,
	int *code, // [k * n]
	long int *codewords, // q^k
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int N, h, rk;
	int *msg;
	int *word;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "coding_theory_domain::codewords_affine" << endl;
	}
	N = Gg.nb_AG_elements(k, F->q);
	if (f_v) {
		cout << N << " messages" << endl;
	}
	msg = NEW_int(k);
	word = NEW_int(n);

	for (h = 0; h < N; h++) {
		Gg.AG_element_unrank(F->q, msg, 1, k, h);
		F->Linear_algebra->mult_vector_from_the_left(msg, code, word, k, n);
		rk = Gg.AG_element_rank(F->q, word, 1, n);
		codewords[h] = rk;
	}
	FREE_int(msg);
	FREE_int(word);
	if (f_v) {
		cout << "coding_theory_domain::codewords_affine done" << endl;
	}
}

void coding_theory_domain::code_projective_weight_enumerator(field_theory::finite_field *F,
	int n, int k,
	int *code, // [k * n]
	int *weight_enumerator, // [n + 1]
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int N, h, wt, i;
	int *msg;
	int *word;
	int t0, t1, dt;
	geometry::geometry_global Gg;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "coding_theory_domain::code_projective_weight_enumerator" << endl;
	}
	N = Gg.nb_AG_elements(k, F->q);
	if (f_v) {
		cout << N << " messages" << endl;
	}
	msg = NEW_int(k);
	word = NEW_int(n);

	Int_vec_zero(weight_enumerator, n + 1);

	for (h = 0; h < N; h++) {
		if (f_v && (h % ONE_MILLION) == 0) {
			t1 = Os.os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			Os.time_check_delta(cout, dt);
			cout << endl;
			if (f_vv) {
				cout << "so far, the weight enumerator is:" << endl;
				for (i = 0; i <= n; i++) {
					if (weight_enumerator[i] == 0)
						continue;
					cout << setw(5) << i << " : " << setw(10)
							<< weight_enumerator[i] << endl;
				}
			}
		}
		Gg.AG_element_unrank(F->q, msg, 1, k, h);
		F->Linear_algebra->mult_vector_from_the_left(msg, code, word, k, n);
		wt = 0;
		for (i = 0; i < n; i++) {
			if (word[i]) {
				wt++;
			}
		}
		weight_enumerator[wt]++;
	}
	if (f_v) {
		cout << "the weight enumerator is:" << endl;
		for (i = 0; i <= n; i++) {
			if (weight_enumerator[i] == 0) {
				continue;
			}
			cout << setw(5) << i << " : " << setw(10)
					<< weight_enumerator[i] << endl;
		}
	}


	FREE_int(msg);
	FREE_int(word);
}

void coding_theory_domain::code_weight_enumerator(field_theory::finite_field *F,
	int n, int k,
	int *code, // [k * n]
	int *weight_enumerator, // [n + 1]
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int N, h, wt, i;
	int *msg;
	int *word;
	int t0, t1, dt;
	geometry::geometry_global Gg;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "coding_theory_domain::code_weight_enumerator" << endl;
	}
	N = Gg.nb_AG_elements(k, F->q);
	if (f_v) {
		cout << N << " messages" << endl;
	}
	msg = NEW_int(k);
	word = NEW_int(n);

	Int_vec_zero(weight_enumerator, n + 1);

	for (h = 0; h < N; h++) {
		if ((h % ONE_MILLION) == 0) {
			t1 = Os.os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			Os.time_check_delta(cout, dt);
			cout << endl;
			if (f_vv) {
				cout << "so far, the weight enumerator is:" << endl;
				for (i = 0; i <= n; i++) {
					if (weight_enumerator[i] == 0) {
						continue;
					}
					cout << setw(5) << i << " : " << setw(10)
							<< weight_enumerator[i] << endl;
				}
			}
		}
		Gg.AG_element_unrank(F->q, msg, 1, k, h);
		F->Linear_algebra->mult_vector_from_the_left(msg, code, word, k, n);
		wt = 0;
		for (i = 0; i < n; i++) {
			if (word[i]) {
				wt++;
			}
		}
		weight_enumerator[wt]++;
		}
	if (f_v) {
		cout << "the weight enumerator is:" << endl;
		for (i = 0; i <= n; i++) {
			if (weight_enumerator[i] == 0) {
				continue;
			}
			cout << setw(5) << i << " : " << setw(10)
					<< weight_enumerator[i] << endl;
		}
	}


	FREE_int(msg);
	FREE_int(word);
}


void coding_theory_domain::code_weight_enumerator_fast(field_theory::finite_field *F,
		int n, int k,
	int *code, // [k * n]
	int *weight_enumerator, // [n + 1]
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int N, h, wt, i;
	int *msg;
	int *word;
	int t0, t1, dt;
	geometry::geometry_global Gg;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "coding_theory_domain::code_weight_enumerator_fast" << endl;
	}
	N = Gg.nb_PG_elements(k - 1, F->q);
	if (f_v) {
		cout << N << " projective messages" << endl;
	}
	msg = NEW_int(k);
	word = NEW_int(n);


	Int_vec_zero(weight_enumerator, n + 1);

	for (h = 0; h < N; h++) {
		if (((h % ONE_MILLION) == 0) && h) {
			t1 = Os.os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			Os.time_check_delta(cout, dt);
			cout << endl;
			if (f_vv) {
				cout << "so far, the weight enumerator is:" << endl;
				for (i = 0; i <= n; i++) {
					if (weight_enumerator[i] == 0) {
						continue;
					}
					cout << setw(5) << i << " : " << setw(10)
							<< (F->q - 1) * weight_enumerator[i] << endl;
				}
			}
		}
		F->PG_element_unrank_modified(msg, 1, k, h);
		//AG_element_unrank(q, msg, 1, k, h);
		F->Linear_algebra->mult_vector_from_the_left(msg, code, word, k, n);
		wt = 0;
		for (i = 0; i < n; i++) {
			if (word[i]) {
				wt++;
			}
		}
		weight_enumerator[wt]++;
		if (f_vv) {
			cout << h << " / " << N << " msg: ";
			Int_vec_print(cout, msg, k);
			cout << " codeword ";
			Int_vec_print(cout, word, n);
			cout << " weight " << wt << endl;
		}
	}
	weight_enumerator[0] = 1;
	for (i = 1; i <= n; i++) {
		weight_enumerator[i] *= F->q - 1;
	}
	if (f_v) {
		cout << "the weight enumerator is:" << endl;
		for (i = 0; i <= n; i++) {
			if (weight_enumerator[i] == 0) {
				continue;
			}
			cout << setw(5) << i << " : " << setw(10)
					<< weight_enumerator[i] << endl;
		}
	}


	FREE_int(msg);
	FREE_int(word);
}

void coding_theory_domain::code_projective_weights(
		field_theory::finite_field *F,
	int n, int k,
	int *code, // [k * n]
	int *&weights, // will be allocated [N]
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 1);
	int N, h, wt, i;
	int *msg;
	int *word;
	int t0, t1, dt;
	geometry::geometry_global Gg;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "coding_theory_domain::code_projective_weights" << endl;
	}
	N = Gg.nb_PG_elements(k - 1, F->q);
	if (f_v) {
		cout << N << " projective messages" << endl;
	}
	weights = NEW_int(N);
	msg = NEW_int(k);
	word = NEW_int(n);

	for (h = 0; h < N; h++) {
		if ((h % ONE_MILLION) == 0) {
			t1 = Os.os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			Os.time_check_delta(cout, dt);
			cout << endl;
		}
		F->PG_element_unrank_modified(msg, 1, k, h);
		//AG_element_unrank(q, msg, 1, k, h);
		F->Linear_algebra->mult_vector_from_the_left(msg, code, word, k, n);
		wt = 0;
		for (i = 0; i < n; i++) {
			if (word[i]) {
				wt++;
			}
		}
		weights[h] = wt;
	}
	if (f_v) {
		cout << "coding_theory_domain::code_projective_weights done" << endl;
	}


	FREE_int(msg);
	FREE_int(word);
}

void coding_theory_domain::mac_williams_equations(ring_theory::longinteger_object *&M, int n, int k, int q)
{
	combinatorics::combinatorics_domain D;
	int i, j;

	M = NEW_OBJECTS(ring_theory::longinteger_object, (n + 1) * (n + 1));

	for (i = 0; i <= n; i++) {
		for (j = 0; j <= n; j++) {
			D.krawtchouk(M[i * (n + 1) + j], n, q, i, j);
		}
	}
}

void coding_theory_domain::determine_weight_enumerator()
{
	int n = 19, k = 7, q = 2;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object *M, *A1, *A2, qk;
	int i;

	qk.create(q, __FILE__, __LINE__);
	D.power_int(qk, k);
	cout << q << "^" << k << " = " << qk << endl;

	mac_williams_equations(M, n, k, q);

	D.matrix_print_tex(cout, M, n + 1, n + 1);

	A1 = NEW_OBJECTS(ring_theory::longinteger_object, n + 1);
	A2 = NEW_OBJECTS(ring_theory::longinteger_object, n + 1);
	for (i = 0; i <= n; i++) {
		A1[i].create(0, __FILE__, __LINE__);
	}
	A1[0].create(1, __FILE__, __LINE__);
	A1[8].create(78, __FILE__, __LINE__);
	A1[12].create(48, __FILE__, __LINE__);
	A1[16].create(1, __FILE__, __LINE__);
	D.matrix_print_tex(cout, A1, n + 1, 1);

	D.matrix_product(M, A1, A2, n + 1, n + 1, 1);
	D.matrix_print_tex(cout, A2, n + 1, 1);

	D.matrix_entries_integral_division_exact(A2, qk, n + 1, 1);

	D.matrix_print_tex(cout, A2, n + 1, 1);

	FREE_OBJECTS(M);
	FREE_OBJECTS(A1);
	FREE_OBJECTS(A2);
}


void coding_theory_domain::do_weight_enumerator(field_theory::finite_field *F,
		int *M, int m, int n,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *A;
	int *base_cols;
	int *weight_enumerator;
	int rk, i;
	orbiter_kernel_system::latex_interface Li;

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator" << endl;
	}

	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	weight_enumerator = NEW_int(n + 1);
	Int_vec_copy(M, A, m * n);

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator input matrix:" << endl;
		Int_matrix_print(A, m, n);

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.print_integer_matrix_tex(cout, A, m, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;

	}

	rk = F->Linear_algebra->Gauss_int(A,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, m, n, n,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator after RREF:" << endl;
		Int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;


		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.print_integer_matrix_tex(cout, A, rk, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;


		cout << "coding_theory_domain::do_weight_enumerator coefficients:" << endl;
		Int_vec_print(cout, A, rk * n);
		cout << endl;
	}

	code_weight_enumerator(F, n, rk,
		A /* code */, // [k * n]
		weight_enumerator, // [n + 1]
		verbose_level);

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator The weight enumerator is:" << endl;
		for (i = 0; i <= n; i++) {
			cout << i << " : " << weight_enumerator[i] << endl;
		}

		int f_first = TRUE;

		for (i = 0; i <= n; i++) {
			if (weight_enumerator[i] == 0) {
				continue;
			}
			if (f_first) {
				f_first = FALSE;
			}
			else {
				cout << " + ";
			}
			cout << weight_enumerator[i];
			if (i) {
				cout << "*";
				cout << "x";
				if (i > 1) {
					cout << "^";
					if (i < 10) {
						cout << i;
					}
					else {
						cout << "(" << i << ")";
					}
				}
			}
			if (n - i) {
				cout << "*";
				cout << "y";
				if (n - i > 1) {
					cout << "^";
					if (n - i < 10) {
						cout << n - i;
					}
					else {
						cout << "(" << n - i << ")";
					}
				}
			}

		}
		cout << endl;


		cout << "coding_theory_domain::do_weight_enumerator The weight enumerator is:" << endl;
		for (i = 0; i <= n; i++) {
			cout << i << " : " << weight_enumerator[i] << endl;
		}

		f_first = TRUE;

		for (i = 0; i <= n; i++) {
			if (weight_enumerator[i] == 0) {
				continue;
			}
			if (f_first) {
				f_first = FALSE;
			}
			else {
				cout << " + ";
			}
			cout << weight_enumerator[i];
			if (i) {
				//cout << "*";
				cout << "x";
				if (i > 1) {
					cout << "^";
					if (i < 10) {
						cout << i;
					}
					else {
						cout << "{" << i << "}";
					}
				}
			}
			if (n - i) {
				//cout << "*";
				cout << "y";
				if (n - i > 1) {
					cout << "^";
					if (n - i < 10) {
						cout << n - i;
					}
					else {
						cout << "{" << n - i << "}";
					}
				}
			}

		}
		cout << endl;

		cout << "weight enumerator:" << endl;
		Int_vec_print_fully(cout, weight_enumerator, n + 1);
		cout << endl;

	}


	if (f_normalize_from_the_left) {
		if (f_v) {
			cout << "coding_theory_domain::do_weight_enumerator normalizing from the left" << endl;
		}
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize_from_front(
					A + i * n, 1, n);
		}

		if (f_v) {
			cout << "coding_theory_domain::do_weight_enumerator after normalize from the left:" << endl;
			Int_matrix_print(A, rk, n);
			cout << "rk=" << rk << endl;
		}
	}

	if (f_normalize_from_the_right) {
		if (f_v) {
			cout << "coding_theory_domain::do_weight_enumerator normalizing from the right" << endl;
		}
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize(
					A + i * n, 1, n);
		}

		if (f_v) {
			cout << "coding_theory_domain::do_weight_enumerator after normalize from the right:" << endl;
			Int_matrix_print(A, rk, n);
			cout << "coding_theory_domain::do_weight_enumerator rk=" << rk << endl;
		}
	}


	FREE_int(A);
	FREE_int(base_cols);
	FREE_int(weight_enumerator);

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator done" << endl;
	}
}


void coding_theory_domain::do_minimum_distance(field_theory::finite_field *F,
		int *M, int m, int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *A;
	int *base_cols;
	//int *weight_enumerator;
	int rk, i;
	orbiter_kernel_system::latex_interface Li;
	orbiter_kernel_system::os_interface Os;
	long int t0, t1, dt, tps;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "coding_theory_domain::do_minimum_distance" << endl;
	}

	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	//weight_enumerator = NEW_int(n + 1);
	Int_vec_copy(M, A, m * n);

	if (f_v) {
		cout << "coding_theory_domain::do_minimum_distance input matrix:" << endl;
		Int_matrix_print(A, m, n);

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.print_integer_matrix_tex(cout, A, m, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;

	}

	rk = F->Linear_algebra->Gauss_int(A,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, m, n, n,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "coding_theory_domain::do_minimum_distance after RREF:" << endl;
		Int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;


		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.print_integer_matrix_tex(cout, A, rk, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;


		cout << "coding_theory_domain::do_minimum_distance coefficients:" << endl;
		Int_vec_print(cout, A, rk * n);
		cout << endl;
	}

#if 0
	code_weight_enumerator(F, n, rk,
		A /* code */, // [k * n]
		weight_enumerator, // [n + 1]
		verbose_level);
#endif

	int j;
	int d;
	int q = F->q;
	int idx_zero = 0;
	int idx_one = 1;
	int *add_table;
	int *mult_table;


	add_table = NEW_int(q * q);
	mult_table = NEW_int(q * q);
	for (i = 0; i < q; i++) {
		for (j = 0; j < q; j++) {
			add_table[i * q + j] = F->add(i, j);
		}
	}
	for (i = 0; i < q; i++) {
		for (j = 0; j < q; j++) {
			mult_table[i * q + j] = F->mult(i, j);
		}
	}

	d = mindist(n, m /* k */, q, A,
		verbose_level - 2, idx_zero, idx_one,
		add_table, mult_table);

	t1 = Os.os_ticks();

	dt = t1 - t0;

	tps = Os.os_ticks_per_second();
	//cout << "time_check_delta tps=" << tps << endl;

	int days, hours, minutes, seconds;

	Os.os_ticks_to_dhms(dt, tps, days, hours, minutes, seconds);


	if (f_v) {
		cout << "coding_theory_domain::do_minimum_distance The minimum distance is d = " << d << ", computed in " << days << " days, " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds" << endl;
	}


	FREE_int(add_table);
	FREE_int(mult_table);

	FREE_int(A);
	FREE_int(base_cols);
	//FREE_int(weight_enumerator);

	if (f_v) {
		cout << "coding_theory_domain::do_minimum_distance done" << endl;
	}
}




void coding_theory_domain::do_linear_code_through_basis(
		field_theory::finite_field *F,
		int n,
		long int *basis_set, int k,
		int f_embellish,
		int verbose_level)
{
	cout << "coding_theory_domain::linear_code_through_basis:" << endl;

	int i;
	int *genma;
	int *word;
	int *code_word;
	long int *set;
	int sz;
	geometry::geometry_global Gg;

	genma = NEW_int(k * n);
	word = NEW_int(k);
	code_word = NEW_int(n);

	for (i = 0; i < k; i++) {

		Gg.AG_element_unrank(2, genma + i * n, 1, n, basis_set[i]);
	}

	cout << "genma:" << endl;
	Int_matrix_print(genma, k, n);

	sz = 1 << k;
	set = NEW_lint(sz);

	//field_theory::finite_field *F;

	//F = NEW_OBJECT(field_theory::finite_field);
	//F->finite_field_init(2, FALSE /* f_without_tables */, 0);

	for (i = 0; i < sz; i++) {
		Gg.AG_element_unrank(2, word, 1, k, i);
		F->Linear_algebra->mult_matrix_matrix(word, genma,
				code_word, 1, k, n, 0 /* verbose_level*/);
		set[i] = Gg.AG_element_rank(2, code_word, 1, n);

		cout << i << " : ";
		Int_vec_print(cout, word, k);
		cout << " : ";
		Int_vec_print(cout, code_word, n);
		cout << " : " << set[i] << endl;
	}


	cout << "Codewords : ";
	Lint_vec_print(cout, set, sz);
	cout << endl;



	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		snprintf(str, 1000, "code_n%d_k%d_q%d.tex", n, k, F->q);
		fname.assign(str);
		snprintf(str, 1000, "Linear $[%d,%d]$ code over GF($%d$)", n, k, F->q);
		title.assign(str);



		{
			ofstream ost(fname);


			orbiter_kernel_system::latex_interface L;


			L.head(ost, FALSE /* f_book*/, TRUE /* f_title */,
				title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			ost << "$$" << endl;
			ost << "\\left[" << endl;
			L.int_matrix_print_tex(ost, genma, k, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;
			ost << "Codewords: ";
			ost << "$";
			Lint_vec_print(ost, set, sz);
			ost << "$\\\\";
			ost << endl;

			L.foot(ost);
		}

		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;

	}

	//investigate_code(set, sz, n, f_embellish, verbose_level);

	//FREE_OBJECT(F);
	FREE_int(genma);
	FREE_int(word);
	FREE_int(code_word);

}

void coding_theory_domain::matrix_from_projective_set(field_theory::finite_field *F,
		int n, int k, long int *columns_set_of_size_n,
		int *genma,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::matrix_from_projective_set" << endl;
	}
	int i, j;
	int *v;

	v = NEW_int(k);
	for (j = 0; j < n; j++) {

		F->PG_element_unrank_modified(v, 1, k, columns_set_of_size_n[j]);
		for (i = 0; i < k; i++) {
			genma[i * n + j] = v[i];
		}
	}
	FREE_int(v);

	if (f_v) {
		cout << "coding_theory_domain::matrix_from_projective_set done" << endl;
	}
}

void coding_theory_domain::do_linear_code_through_columns_of_parity_check_projectively(
		field_theory::finite_field *F,
		int n,
		long int *columns_set, int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::do_linear_code_through_columns_of_parity_check_projectively" << endl;
	}

	//field_theory::finite_field *F;
	int i, j;
	int *v;
	int *genma;
	int *word;
	int *code_word;
	geometry::geometry_global Gg;

	//F = NEW_OBJECT(field_theory::finite_field);
	//F->finite_field_init(2, FALSE /* f_without_tables */, 0);
	genma = NEW_int(k * n);
	v = NEW_int(k);
	word = NEW_int(k);
	code_word = NEW_int(n);
	for (j = 0; j < n; j++) {

		F->PG_element_unrank_modified(v, 1, k, columns_set[j]);
		for (i = 0; i < k; i++) {
			genma[i * n + j] = v[i];
		}
	}
	cout << "genma:" << endl;
	Int_matrix_print(genma, k, n);


	number_theory::number_theory_domain NT;
	long int *set;
	long int N;

	N = NT.i_power_j(2, k);
	set = NEW_lint(N);
	for (i = 0; i < N; i++) {
		Gg.AG_element_unrank(2, word, 1, k, i);
		F->Linear_algebra->mult_matrix_matrix(word, genma,
				code_word, 1, k, n, 0 /* verbose_level*/);
		set[i] = Gg.AG_element_rank(2, code_word, 1, n);

		cout << i << " : ";
		Int_vec_print(cout, word, k);
		cout << " : ";
		Int_vec_print(cout, code_word, n);
		cout << " : " << set[i] << endl;
	}


	cout << "Codewords : ";
	Lint_vec_print_fully(cout, set, N);
	cout << endl;

	char str[1000];
	string fname_csv;
	orbiter_kernel_system::file_io Fio;

	snprintf(str, 1000, "codewords_n%d_k%d_q%d.csv", n, k, F->q);
	fname_csv.assign(str);
	Fio.lint_matrix_write_csv(fname_csv, set, 1, N);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;



	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		snprintf(str, 1000, "code_n%d_k%d_q%d.tex", n, k, F->q);
		fname.assign(str);
		snprintf(str, 1000, "Linear $[%d,%d]$ code over GF($%d$)", n, k, F->q);
		title.assign(str);



		{
			ofstream ost(fname);


			orbiter_kernel_system::latex_interface L;


			L.head(ost, FALSE /* f_book*/, TRUE /* f_title */,
				title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			ost << "$$" << endl;
			ost << "\\left[" << endl;
			L.int_matrix_print_tex(ost, genma, k, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;
#if 0
			ost << "Codewords: ";
			ost << "$";
			lint_vec_print(ost, set, n);
			ost << "$\\\\";
#endif
			ost << endl;

			L.foot(ost);
		}

		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;

	}



	FREE_lint(set);
	FREE_int(genma);
	FREE_int(v);
	FREE_int(word);
	FREE_int(code_word);
	//FREE_OBJECT(F);

	if (f_v) {
		cout << "coding_theory_domain::do_linear_code_through_columns_of_parity_check_projectively done" << endl;
	}
}

void coding_theory_domain::do_linear_code_through_columns_of_parity_check(
		field_theory::finite_field *F,
		int n,
		long int *columns_set, int k,
		int *&genma,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::do_linear_code_through_columns_of_parity_check" << endl;
	}

	int i, j;
	int *v;
	int *word;
	int *code_word;
	geometry::geometry_global Gg;

	genma = NEW_int(k * n);
	v = NEW_int(k);
	word = NEW_int(k);
	code_word = NEW_int(n);

	for (j = 0; j < n; j++) {

		Gg.AG_element_unrank(2, v, 1, k, columns_set[j]);
		for (i = 0; i < k; i++) {
			genma[i * n + j] = v[i];
		}
	}

	if (f_v) {
		cout << "coding_theory_domain::do_linear_code_through_columns_of_parity_check genma:" << endl;
		Int_matrix_print(genma, k, n);
	}

#if 0
	number_theory::number_theory_domain NT;
	long int *set;
	long int N;

	N = NT.i_power_j(2, k);
	set = NEW_lint(N);
	for (i = 0; i < N; i++) {
		Gg.AG_element_unrank(2, word, 1, k, i);
		F->Linear_algebra->mult_matrix_matrix(word, genma,
				code_word, 1, k, n, 0 /* verbose_level*/);
		set[i] = Gg.AG_element_rank(2, code_word, 1, n);

		if (f_v) {
			cout << i << " : ";
			Int_vec_print(cout, word, k);
			cout << " : ";
			Int_vec_print(cout, code_word, n);
			cout << " : " << set[i] << endl;
		}
	}


	if (f_v) {
		cout << "Codewords : ";
		Lint_vec_print_fully(cout, set, N);
		cout << endl;
	}

	char str[1000];
	string fname_csv;
	orbiter_kernel_system::file_io Fio;

	snprintf(str, 1000, "codewords_n%d_k%d_q%d.csv", n, k, F->q);
	fname_csv.assign(str);
	Fio.lint_matrix_write_csv(fname_csv, set, 1, N);
	if (f_v) {
		cout << "written file " << fname_csv << " of size "
				<< Fio.file_size(fname_csv) << endl;
	}
#endif


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		snprintf(str, 1000, "code_n%d_k%d_q%d.tex", n, k, F->q);
		fname.assign(str);
		snprintf(str, 1000, "Linear $[%d,%d]$ code over GF($%d$)", n, k, F->q);
		title.assign(str);



		{
			ofstream ost(fname);


			orbiter_kernel_system::latex_interface L;


			L.head(ost, FALSE /* f_book*/, TRUE /* f_title */,
				title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			ost << "$$" << endl;
			ost << "\\left[" << endl;
			L.int_matrix_print_tex(ost, genma, k, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;
#if 0
			ost << "Codewords: ";
			ost << "$";
			lint_vec_print(ost, set, n);
			ost << "$\\\\";
#endif
			ost << endl;

			L.foot(ost);
		}

		orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}

	}



	//FREE_lint(set);
	//FREE_int(genma);
	FREE_int(v);
	FREE_int(word);
	FREE_int(code_word);
	//FREE_OBJECT(F);

	if (f_v) {
		cout << "coding_theory_domain::do_linear_code_through_columns_of_parity_check done" << endl;
	}
}

void coding_theory_domain::do_polynomial(
		int n,
		int polynomial_degree,
		int polynomial_nb_vars,
		std::string &polynomial_text,
		int f_embellish,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "reading polynomial " << polynomial_text << " of degree "
				<< polynomial_degree << " in "
				<< polynomial_nb_vars << " variables" << endl;
	}

	long int *poly_monomials;
	int poly_monomials_sz;
	ring_theory::homogeneous_polynomial_domain *Poly;
	field_theory::finite_field *Fq;
	int *mon;
	int *coeff;
	long int a;
	int i, j, b, idx;
	monomial_ordering_type Monomial_ordering_type = t_PART;

	Lint_vec_scan(polynomial_text, poly_monomials, poly_monomials_sz);
	cout << "polynomial after scan: ";
	Lint_vec_print(cout, poly_monomials, poly_monomials_sz);
	cout << endl;

	Poly = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Fq = NEW_OBJECT(field_theory::finite_field);

	Fq->finite_field_init(2, FALSE /* f_without_tables */, 0 /* verbose_level */);

	Poly->init(Fq, polynomial_nb_vars, polynomial_degree,
				Monomial_ordering_type,
				0 /* verbose_level */);
	mon = NEW_int(polynomial_nb_vars);
	coeff = NEW_int(Poly->get_nb_monomials());

	Int_vec_zero(coeff, Poly->get_nb_monomials());

	for (i = 0; i < poly_monomials_sz; i++) {
		Int_vec_zero(mon, polynomial_nb_vars);
		a = poly_monomials[i];
		j = 0;
		while (a) {
			b = a % 10;
			mon[b]++;
			a /= 10;
			j++;
		}
		mon[0] += polynomial_degree - j;
		idx = Poly->index_of_monomial(mon);
		coeff[idx] = Fq->add(coeff[idx], 1);
	}

	Poly->print_equation(cout, coeff);
	cout << endl;

	int *v;
	int *f;
	int h;
	long int *set;
	int set_sz = 0;
	geometry::geometry_global Gg;
	data_structures::sorting Sorting;
	long int N_points;

	N_points = Gg.nb_PG_elements(Poly->nb_variables - 1, Fq->q);

	v = NEW_int(polynomial_nb_vars);
	f = NEW_int(N_points);
	Poly->polynomial_function(coeff, f, verbose_level);


	set = NEW_lint(N_points);

	for (h = 0; h < N_points; h++) {
		Poly->unrank_point(v, h);
		cout << h << " : ";
		Int_vec_print(cout, v, polynomial_nb_vars);
		cout << " : " << f[h] << endl;
		if (f[h] == 1 && v[polynomial_nb_vars - 1] == 1) {
			a = Gg.AG_element_rank(2, v, 1, polynomial_nb_vars - 1);
			set[set_sz++] = a;
		}
	}
	FREE_int(v);

	Sorting.lint_vec_heapsort(set, set_sz);

	cout << "We found a set of size " << set_sz << " : " << endl;
	Lint_vec_print_fully(cout, set, set_sz);
	cout << endl;

	investigate_code(set, set_sz, n, f_embellish, verbose_level);

	FREE_int(f);

}

void coding_theory_domain::do_sylvester_hadamard(int n,
		int f_embellish,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::do_sylvester_hadamard" << endl;
	}
	int i;

	if (n % 4) {
		cout << "for Hadamard matrices, n must be divisible by 4." << endl;
		exit(1);
	}
	int m = n >> 2;
	int nb_factors, sz, sz1, j, a;
	int *M1;
	int *M2;
	int H2[4] = {1,1,1,2};
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;

	nb_factors = NT.int_log2(m);

	cout << "nb_factors = " << nb_factors << endl;

	if ((2 << nb_factors) != n) {
		cout << "for Sylvester type Hadamard matrices, n must be 4 times a power of two" << endl;
		exit(1);
	}

	M1 = NEW_int(2 * n * n);
	M2 = NEW_int(2 * n * n);
	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init(3, FALSE /* f_without_tables */, 0);
	Int_vec_copy(H2, M1, 4);
	sz = 2;
	for (i = 0; i < nb_factors; i++) {

		cout << "M1=" << endl;
		Int_matrix_print(M1, sz, sz);

		F->Linear_algebra->Kronecker_product_square_but_arbitrary(
				M1, H2,
				sz, 2, M2, sz1,
				verbose_level);
		Int_vec_copy(M2, M1, sz1 * sz1);

		sz = sz1;
	}
	cout << "Sylvester type Hadamard matrix:" << endl;
	Int_matrix_print(M1, sz, sz);
	for (i = 0; i < sz; i++) {
		for (j = 0; j < sz; j++) {
			a = M1[i * sz + j];
			M1[(sz + i) * sz + j] = F->negate(a);
		}
	}

	for (i = 0; i < 2 * sz; i++) {
		for (j = 0; j < sz; j++) {
			a = M1[i * sz + j];
			if (a == 2) {
				M1[i * sz + j] = 0;
			}
		}
	}
	cout << "Sylvester type Hadamard code:" << endl;
	Int_matrix_print(M1, 2 * sz, sz);


	long int *set;

	set = NEW_lint(2 * sz);
	for (i = 0; i < 2 * sz; i++) {
		set[i] = Gg.AG_element_rank(2, M1 + i * sz, 1, sz);
	}

	investigate_code(set, 2 * sz, n, f_embellish, verbose_level);

	FREE_lint(set);

	FREE_int(M1);
	FREE_int(M2);
	FREE_OBJECT(F);

}

void coding_theory_domain::code_diagram(
		std::string &label,
		long int *Words,
		int nb_words, int n, int f_metric_balls, int radius_of_metric_ball,
		int f_enhance, int radius,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_rows, nb_cols;
	int *v;
	int *M;
	int *M1;
	int *M2;
	int *M3; // the holes
	int N;
	int h, i, j;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "coding_theory_domain::code_diagram" << endl;
		cout << "n=" << n << endl;
		cout << "set:" << endl;
		Lint_vec_print(cout, Words, nb_words);
		cout << endl;
	}

	dimensions(n, nb_rows, nb_cols);

	if (f_v) {
		cout << "coding_theory_domain::code_diagram" << endl;
		cout << "nb_rows=" << nb_rows << endl;
		cout << "nb_cols=" << nb_cols << endl;
	}

	v = NEW_int(n);
	M1 = NEW_int(nb_rows * nb_cols);
	M2 = NEW_int(nb_rows * nb_cols);
	M3 = NEW_int(nb_rows * nb_cols);
	M = NEW_int(nb_rows * nb_cols);


	Int_vec_zero(M1, nb_rows * nb_cols);
	Int_vec_zero(M2, nb_rows * nb_cols);
	//Orbiter->Int_vec.zero(M3, nb_rows * nb_cols);
	for (h = 0; h < nb_rows * nb_cols; h++) {
		M3[h] = n + 1;
	}

	N = 1 << n;

	cout << "N=" << N << endl;

	cout << "placing codewords" << endl;
	for (h = 0; h < N; h++) {
		place_binary(h, i, j);
		M1[i * nb_cols + j] = h;
		//M2[i * nb_cols + j] = 1;
		}
	cout << "placing position values done" << endl;


	cout << "placing codewords" << endl;
	Int_vec_zero(M, nb_rows * nb_cols);
	for (h = 0; h < nb_words; h++) {
		convert_to_binary(n, Words[h], v);
		cout << "codeword " << h + 1 << " = " << setw(5) << Words[h];
		cout << " : ";
		print_binary(n, v);
		cout << endl;
		place_binary(Words[h], i, j);
		M[i * nb_cols + j] = h + 1;
		M2[i * nb_cols + j] = 1;
		M3[i * nb_cols + j] = 0; // distance is zero

		if (f_enhance) {
			embellish(M, nb_rows, nb_cols, i, j, h + 1 /* value */, radius);
		}
		if (f_enhance) {
			embellish(M2, nb_rows, nb_cols, i, j, 1 /* value */, radius);
		}
	}
	//int_matrix_print(M, nb_rows, nb_cols);
	cout << "placing codewords done" << endl;



	if (f_metric_balls) {
		int u, t, s, a;
		int *set_of_errors;

		set_of_errors = NEW_int(radius_of_metric_ball);

		for (h = 0; h < nb_words; h++) {
			convert_to_binary(n, Words[h], v);
			cout << "codeword " << h + 1 << " = " << setw(5) << Words[h];
			cout << " : ";
			print_binary(n, v);
			cout << endl;
			place_binary(Words[h], i, j);
			for (u = 1; u <= radius_of_metric_ball; u++) {
				combinatorics::combinatorics_domain Combi;

				N = Combi.int_n_choose_k(n, u);
				for (t = 0; t < N; t++) {
					Combi.unrank_k_subset(t, set_of_errors, n, u);
					convert_to_binary(n, Words[h], v);
					for (s = 0; s < u; s++) {
						a = set_of_errors[s];
						v[a] = (v[a] + 1) % 2;
					}
					place_binary(v, n, i, j);
					if (M[i * nb_cols + j]) {
						cout << "the metric balls overlap!" << endl;
						cout << "h=" << h << endl;
						cout << "t=" << t << endl;
						cout << "i=" << i << endl;
						cout << "j=" << j << endl;
						exit(1);
					}
					M[i * nb_cols + j] = h + 1;
				}
			}
		}
		FREE_int(set_of_errors);
	}

	int *Dist_from_code_enumerator;
	int d;
	int s, original_value, a;
	int *set_of_errors;

	set_of_errors = NEW_int(n);

	Dist_from_code_enumerator = NEW_int(n + 1);
	Int_vec_zero(Dist_from_code_enumerator, n + 1);

	for (d = 0; d < n; d++) {
		cout << "computing words of distance " << d + 1 << " from the code" << endl;
		for (h = 0; h < nb_rows * nb_cols; h++) {
			if (M3[h] == d) {
				Dist_from_code_enumerator[d]++;
				i = h / nb_cols;
				j = h % nb_cols;
				convert_to_binary(n, h, v);
				for (s = 0; s < n; s++) {
					original_value = v[s];
					v[s] = (v[s] + 1) % 2;
					place_binary(v, n, i, j);
					a = i * nb_cols + j;
					if (M3[a] > d + 1) {
						M3[a] = d + 1;
					}
					v[s] = original_value;
				}
			}
		}
		cout << "We found " << Dist_from_code_enumerator[d]
			<< " words at distance " << d << " from the code" << endl;

		if (Dist_from_code_enumerator[d] == 0) {
			break;
		}
	}
	cout << "d : # words at distance d from code" << endl;
	for (d = 0; d < n; d++) {
		cout << d << " : " << Dist_from_code_enumerator[d] << endl;
	}
	cout << endl;

	FREE_int(set_of_errors);

	{
		char str[1000];

		string fname;

		fname.assign(label);

		sprintf(str, "_%d_%d.tex", n, nb_words);
		fname.append(str);

		{
			ofstream fp(fname);
			orbiter_kernel_system::latex_interface L;

			L.head_easy(fp);
			fp << "$$" << endl;
			L.print_integer_matrix_tex(fp, M1, nb_rows, nb_cols);
			fp << "$$" << endl;



			fp << "$$" << endl;
			L.print_integer_matrix_tex(fp, M, nb_rows, nb_cols);
			fp << "$$" << endl;


			L.foot(fp);
		}
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	//cout << "M:" << endl;
	//int_matrix_print(M, nb_rows, nb_cols);

	{
		char str[1000];

		string fname;

		fname.assign(label);

		sprintf(str, "_diagram_%d_%d.csv", n, nb_words);
		fname.append(str);
		orbiter_kernel_system::file_io Fio;

		Fio.int_matrix_write_csv(fname, M, nb_rows, nb_cols);

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	//cout << "M2:" << endl;
	//int_matrix_print(M2, nb_rows, nb_cols);

	{
		char str[1000];

		string fname;

		fname.assign(label);

		sprintf(str, "_diagram_01_%d_%d.csv", n, nb_words);
		fname.append(str);
		orbiter_kernel_system::file_io Fio;

		Fio.int_matrix_write_csv(fname, M2, nb_rows, nb_cols);

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	{
		char str[1000];

		string fname;

		fname.assign(label);

		sprintf(str, "_holes_%d_%d.csv", n, nb_words);
		fname.append(str);
		orbiter_kernel_system::file_io Fio;

		Fio.int_matrix_write_csv(fname, M3, nb_rows, nb_cols);

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_int(v);
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(M3);
	FREE_int(M);

	if (f_v) {
		cout << "coding_theory_domain::code_diagram done" << endl;
	}
}

void coding_theory_domain::investigate_code(long int *Words,
		int nb_words, int n, int f_embellish, int verbose_level)
// creates a combinatorics::boolean_function_domain object
{
	int f_v = (verbose_level >= 1);
	int nb_rows, nb_cols;
	int *v;
	int *M;
	int *M1;
	int *M2;
	int N;
	//int D, d;
	int h, i, j;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "coding_theory_domain::investigate_code" << endl;
		cout << "n=" << n << endl;
		cout << "set:" << endl;
		Lint_vec_print(cout, Words, nb_words);
		cout << endl;
	}

	dimensions(n, nb_rows, nb_cols);

	if (f_v) {
		cout << "coding_theory_domain::investigate_code" << endl;
		cout << "nb_rows=" << nb_rows << endl;
		cout << "nb_cols=" << nb_cols << endl;
	}

	v = NEW_int(n);
	M1 = NEW_int(nb_rows * nb_cols);
	M2 = NEW_int(nb_rows * nb_cols);
	M = NEW_int(nb_rows * nb_cols);


	Int_vec_zero(M1, nb_rows * nb_cols);
	Int_vec_zero(M2, nb_rows * nb_cols);


	N = 1 << n;

	cout << "N=" << N << endl;

	cout << "placing codewords" << endl;
	for (h = 0; h < N; h++) {
		place_binary(h, i, j);
		M1[i * nb_cols + j] = h;
		//M2[i * nb_cols + j] = 1;
		}
	cout << "placing position values done" << endl;


#if 0
	colored_graph *C;

	C = new colored_graph;

	C->init_adjacency_no_colors(int nb_points, int *Adj, int verbose_level);
#endif


	char fname[1000];

	sprintf(fname, "code_%d_%d.tex", n, nb_words);

	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;

		L.head_easy(fp);
		fp << "$$" << endl;
		L.print_integer_matrix_tex(fp, M1, nb_rows, nb_cols);
		fp << "$$" << endl;



		cout << "placing codewords" << endl;
		Int_vec_zero(M, nb_rows * nb_cols);
		for (h = 0; h < nb_words; h++) {
			convert_to_binary(n, Words[h], v);
			cout << "codeword " << h + 1 << " = " << setw(5) << Words[h];
			cout << " : ";
			print_binary(n, v);
			cout << endl;
			place_binary(Words[h], i, j);
			M[i * nb_cols + j] = h + 1;
			M2[i * nb_cols + j] = 1;
			if (f_embellish) {
				embellish(M, nb_rows, nb_cols, i, j, h + 1, 3);
				}
			}
		//int_matrix_print(M, nb_rows, nb_cols);
		cout << "placing codewords done" << endl;

		fp << "$$" << endl;
		L.print_integer_matrix_tex(fp, M, nb_rows, nb_cols);
		fp << "$$" << endl;


		L.foot(fp);
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	cout << "M2:" << endl;
	Int_matrix_print(M2, nb_rows, nb_cols);

	{
		char str[1000];
		string fname;
		orbiter_kernel_system::file_io Fio;

		sprintf(str, "code_matrix_%d_%d.csv", nb_rows, nb_cols);
		fname.assign(str);
		Fio.int_matrix_write_csv(fname, M2, nb_rows, nb_cols);

	}


#if 0
	{
	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);

	cout << "before create_Levi_graph_from_incidence_matrix" << endl;
	CG->create_Levi_graph_from_incidence_matrix(M, nb_rows, nb_cols,
		FALSE /* f_point_labels */, NULL /* *point_labels */,
		verbose_level);

	cout << "after create_Levi_graph_from_incidence_matrix" << endl;

	char fname[1000];


	sprintf(fname, "code_%d_%d_Levi_%d_%d.bin",
		n, nb_words, nb_rows, nb_cols);
	CG->save(fname, verbose_level);
	delete CG;
	}
#endif


	combinatorics::boolean_function_domain *BF;
	int *coeff;
	int *f;
	int *g;

	BF = NEW_OBJECT(combinatorics::boolean_function_domain);


	cout << "before BF->init, n=" << n << endl;
	BF->init(n, 0 /*verbose_level*/);

	cout << "BF->Poly[n].get_nb_monomials()=" << BF->Poly[n].get_nb_monomials() << endl;
	coeff = NEW_int(BF->Poly[n].get_nb_monomials());
	f = NEW_int(N);
	g = NEW_int(N);

	Int_vec_zero(f, N);
	for (h = 0; h < nb_words; h++) {
		f[Words[h]] = 1;
	}

	cout << "computing the polynomial representation: " << endl;


	BF->compute_polynomial_representation(
			f /* func */, coeff, 0 /*verbose_level*/);
	//BF->search_for_bent_functions(verbose_level);



	cout << "The representation as polynomial is: ";

	cout << " : ";
	BF->Poly[BF->n].print_equation(cout, coeff);
	cout << " : ";
	//evaluate_projectively(poly, f_proj);
	BF->evaluate(coeff, g);
	//int_vec_print(cout, g, BF->Q);
	cout << endl;


	for (h = 0; h < BF->Q; h++) {
		if (f[h] != g[h]) {
			cout << "f[h] != g[h], h = " << h << ", an error has occured" << endl;
		}
	}


	FREE_OBJECT(BF);
	FREE_OBJECT(coeff);
	FREE_OBJECT(f);
	FREE_OBJECT(g);


#if 0
	D = INT_MAX;
	for (i = 0; i < nb_words; i++) {
		for (j = i + 1; j < nb_words; j++) {
			d = distance(n, Words[i], Words[j]);
			//cout << "The distance between word " << i << " and word " << j << " is " << d << endl;
			D = MINIMUM(D, d);
			}
		}

	cout << "minimum distance d = " << D << endl;
	cout << "attained for:" << endl;
	for (i = 0; i < nb_words; i++) {
		for (j = i + 1; j < nb_words; j++) {
			if (distance(n, Words[i], Words[j]) == D) {
				cout << i << ", " << j << endl;
				}
			}
		}
#endif

}

void coding_theory_domain::do_long_code(
		int n,
		std::vector<std::string> &long_code_generators_text,
		int f_nearest_codeword,
		std::string &nearest_codeword_text,
		int verbose_level)
// creates a combinatorics::boolean_function_domain object
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::do_long_code" << endl;
	}

	int i, j;
	int *genma;
	int k;

	k = long_code_generators_text.size();
	genma = NEW_int(k * n);
	Int_vec_zero(genma, k * n);
	for (i = 0; i < k; i++) {
		long int *set;
		int sz;


		orbiter_kernel_system::Orbiter->get_lint_vector_from_label(long_code_generators_text[i], set, sz, verbose_level);

		for (j = 0; j < sz; j++) {
			genma[i * n + set[j]] = 1;
		}
		FREE_lint(set);
	}

	cout << "genma:" << endl;
	Int_matrix_print(genma, k, n);

	{
		char str[1000];
		string fname;
		orbiter_kernel_system::file_io Fio;

		sprintf(str, "long_code_genma_n%d_k%d.csv", n, k);
		fname.assign(str);
		Fio.int_matrix_write_csv(fname, genma, k, n);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	int sz;
	int *message;
	int *code_word;
	int *M;
	int nb_rows, nb_cols;
	int h, r, c;
	geometry::geometry_global Gg;
	int *Wt;
	int wt;
	//int N;

	//N = 1 << n;
	dimensions_N(n, nb_rows, nb_cols);

	if (f_v) {
		cout << "n=" << n << endl;
		cout << "nb_rows=" << nb_rows << endl;
		cout << "nb_cols=" << nb_cols << endl;
	}

	sz = 1 << k;
	message = NEW_int(k);
	code_word = NEW_int(n);
	M = NEW_int(nb_rows * nb_cols);

	field_theory::finite_field *F;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init(2, FALSE /* f_without_tables */, 0);

	Wt = NEW_int(sz);
	Int_vec_zero(Wt, sz);

	for (i = 0; i < sz; i++) {
		Gg.AG_element_unrank(2, message, 1, k, i);
		F->Linear_algebra->mult_matrix_matrix(message, genma,
				code_word, 1, k, n, 0 /* verbose_level*/);

		Int_vec_zero(M, nb_rows * nb_cols);
		wt = 0;
		for (h = 0; h < n; h++) {
			if (code_word[h]) {
				wt++;
			}
		}
		Wt[i] = wt;
		for (h = 0; h < n; h++) {
			if (code_word[h]) {
				place_binary(h, r, c);
				M[r * nb_cols + c] = 1;
			}
		}
		{
			char str[1000];
			string fname;
			orbiter_kernel_system::file_io Fio;

			sprintf(str, "long_code_genma_n%d_k%d_codeword_%d.csv", n, k, i);
			fname.assign(str);
			Fio.int_matrix_write_csv(fname, M, nb_rows, nb_cols);
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}
	}

	{
		cout << "Weight distribution:";
		data_structures::tally C;

		C.init(Wt, sz, FALSE, 0);
		C.print_first(FALSE /* f_backwards */);
		cout << endl;

		cout << "i : weight of the i-th codeword" << endl;
		for (i = 0; i < sz; i++) {
			cout << i << " : " << Wt[i] << endl;
		}
	}


	combinatorics::boolean_function_domain *BF;
	int *f;
	int *g;
	int ln;

	BF = NEW_OBJECT(combinatorics::boolean_function_domain);


	ln = log2(n);

	cout << "before BF->init, ln=" << ln << endl;
	BF->init(ln, 0 /*verbose_level*/);

	f = NEW_int(n);
	g = NEW_int(n);


	if (f_nearest_codeword) {

		cout << "nearest codeword" << endl;

		long int *nearest_codeword_set;
		int nearest_codeword_sz;
		int *word;


		Lint_vec_scan(nearest_codeword_text,
				nearest_codeword_set, nearest_codeword_sz);


		word = NEW_int(n);
		Int_vec_zero(word, n);
		for (j = 0; j < nearest_codeword_sz; j++) {
			word[nearest_codeword_set[j]] = 1;
		}
		for (h = 0; h < n; h++) {
			if (word[h]) {
				f[h] = -1;
			}
			else {
				f[h] = 1;
			}
		}

		BF->apply_Walsh_transform(f, g);
		Int_vec_zero(M, nb_rows * nb_cols);
		for (h = 0; h < n; h++) {
			place_binary(h, r, c);
			M[r * nb_cols + c] = g[h];
		}
		{
			char str[1000];
			string fname;
			orbiter_kernel_system::file_io Fio;

			sprintf(str, "long_code_genma_n%d_k%d_nearest_codeword_fourier.csv", n, k);
			fname.assign(str);
			Fio.int_matrix_write_csv(fname, M, nb_rows, nb_cols);
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}

		Int_vec_zero(M, nb_rows * nb_cols);
		for (h = 0; h < n; h++) {
			if (word[h]) {
				place_binary(h, r, c);
				M[r * nb_cols + c] = 1;
			}
		}
		{
			char str[1000];
			string fname;
			orbiter_kernel_system::file_io Fio;

			sprintf(str, "long_code_genma_n%d_k%d_nearest_codeword.csv", n, k);
			fname.assign(str);
			Fio.int_matrix_write_csv(fname, M, nb_rows, nb_cols);
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}

		int d;
		int *D;

		D = NEW_int(sz);
		for (i = 0; i < sz; i++) {
			Gg.AG_element_unrank(2, message, 1, k, i);
			F->Linear_algebra->mult_matrix_matrix(message, genma,
					code_word, 1, k, n, 0 /* verbose_level*/);

			d = 0;
			for (h = 0; h < n; h++) {
				if (word[h] != code_word[h]) {
					d++;
				}
			D[i] = d;
			}
		}
		{
			cout << "distance distribution:";
			data_structures::tally C;

			C.init(D, sz, FALSE, 0);
			C.print_first(FALSE /* f_backwards */);
			cout << endl;

			cout << "i : distance from the i-th codeword" << endl;
			for (i = 0; i < sz; i++) {
				cout << i << " : " << D[i] << endl;
			}
		}
		FREE_int(D);


	}
	else {
		cout << "no nearest codeword option" << endl;
	}

	FREE_int(message);
	FREE_int(code_word);
	FREE_int(M);
	FREE_OBJECT(F);


}

void coding_theory_domain::embellish(int *M, int nb_rows, int nb_cols, int i0, int j0, int a, int rad)
{
	int i, j, u, v;

#if 0
	int ij[] = {
		-1, -1,
		-1, 0,
		-1, 1,
		0, -1,
		0, 1,
		1, -1,
		1, 0,
		1, 1,
		-2,-2,
		-2,-1,
		-2,0,
		-2,1,
		-2,2,
		-2,-2,
		-2,2,
		-1,-2,
		-1,2,
		0,-2,
		0,2,
		1,-2,
		1,2,
		2,-2,
		2,-1,
		2,0,
		2,1,
		2,2,s
		};
#endif
	for (u = -rad; u <= rad; u++) {
		for (v = -rad; v <= rad; v++) {
			i = i0 + u;
			j = j0 + v;
			place_entry(M, nb_rows, nb_cols, i, j, a);
			}
		}
#if 0
	for (h = 0; h < 8 + 18; h++) {
		i = i0 + ij[h * 2 + 0];
		j = j0 + ij[h * 2 + 1];
		place_entry(M, nb_rows, nb_cols, i, j, a);
		}
#endif
}

void coding_theory_domain::place_entry(int *M, int nb_rows, int nb_cols, int i, int j, int a)
{
	if (i < 0) {
		return;
	}
	if (j < 0) {
		return;
	}
	if (i >= nb_rows) {
		return;
	}
	if (j >= nb_cols) {
		return;
	}
	M[i * nb_cols + j] = a;
}

void coding_theory_domain::do_it(int n, int r, int a, int c, int seed, int verbose_level)
{
	int N;
	int i, j, h, s;
	int nb_rows, nb_cols;
	int *v;
	int *W;
	orbiter_kernel_system::latex_interface L;

	N = 1 << n;

	cout << "N=" << N << endl;

	for (h = 0; h < N; h++) {
		place_binary(h, i, j);
		cout << h << " : (" << i << "," << j << ")" << endl;
	}

	dimensions(n, nb_rows, nb_cols);
	int *M;
	int D, d;

	v = NEW_int(n);
	W = NEW_int(r);
	M = NEW_int(nb_rows * nb_cols);
	Int_vec_zero(M, nb_rows * nb_cols);

	s = seed;
	for (h = 0; h < r; h++) {
		W[h] = s;
		convert_to_binary(n, s, v);
		cout << "s = " << setw(5) << s;
		cout << " : ";
		print_binary(n, v);
		cout << endl;
		place_binary(s, i, j);
		M[i * nb_cols + j] = 1;
		s = (a * s + c) % N;
	}

	Int_matrix_print(M, nb_rows, nb_cols);

	L.print_integer_matrix_tex(cout, M, nb_rows, nb_cols);

	D = INT_MAX;
	for (i = 0; i < r; i++) {
		for (j = i + 1; j < r; j++) {
			d = distance(n, W[i], W[j]);
			cout << "The distance between word " << i << " and word " << j << " is " << d << endl;
			D = MINIMUM(D, d);
		}
	}

	cout << "minimum distance d = " << D << endl;
	cout << "attained for:" << endl;
	for (i = 0; i < r; i++) {
		for (j = i + 1; j < r; j++) {
			if (distance(n, W[i], W[j]) == D) {
				cout << i << ", " << j << endl;
			}
		}
	}
}

void coding_theory_domain::dimensions(int n, int &nb_rows, int &nb_cols)
{
	int i, j;

	place_binary((1 << n) - 1, i, j);
	nb_rows = i + 1;
	nb_cols = j + 1;
}

void coding_theory_domain::dimensions_N(int N, int &nb_rows, int &nb_cols)
{
	int i, j;
	long int a, b;
	number_theory::number_theory_domain NT;

	a = NT.int_log2(N);
	b = 1 << a;
	place_binary(b - 1, i, j);
	nb_rows = i + 1;
	nb_cols = j + 1;
}

void coding_theory_domain::print_binary(int n, int *v)
{
	int c;

	for (c = n - 1; c >= 0; c--) {
		cout << v[c];
	}
}

void coding_theory_domain::convert_to_binary(int n, long int h, int *v)
{
	int c;

	for (c = 0; c < n; c++) {
		if (h % 2) {
			v[c] = 1;
		}
		else {
			v[c] = 0;
		}
		h >>= 1;
	}
}

int coding_theory_domain::distance(int n, int a, int b)
{
	int c, d = 0;

	for (c = 0; c < n; c++) {
		if (a % 2 != b % 2) {
			d++;
		}
		a >>= 1;
		b >>= 1;
	}
	return d;
}

void coding_theory_domain::place_binary(long int h, int &i, int &j)
{
	int o[2];
	int c;

	o[0] = 1;
	o[1] = 0;
	i = 0;
	j = 0;
	for (c = 0; h; c++) {
		if (h % 2) {
			i += o[0];
			j += o[1];
		}
		h >>= 1;
		if (c % 2) {
			o[0] = o[1] << 1;
			o[1] = 0;
		}
		else {
			o[1] = o[0];
			o[0] = 0;
		}
	}
}

void coding_theory_domain::place_binary(int *v, int n, int &i, int &j)
{
	int o[2];
	int c;

	o[0] = 1;
	o[1] = 0;
	i = 0;
	j = 0;
	for (c = 0; c < n; c++) {
		if (v[c]) {
			i += o[0];
			j += o[1];
		}
		if (c % 2) {
			o[0] = o[1] << 1;
			o[1] = 0;
		}
		else {
			o[1] = o[0];
			o[0] = 0;
		}
	}
}




void coding_theory_domain::field_reduction(
		field_theory::finite_field *FQ, field_theory::finite_field *Fq,
		std::string &label,
		int m, int n, std::string &genma_text,
		int verbose_level)
// creates a field_theory::subfield_structure object
{
	int f_v = (verbose_level >= 1);
	int *M;
	int sz;
	int i;
	int *M2;

	if (f_v) {
		cout << "coding_theory_domain::field_reduction" << endl;
	}

	field_theory::subfield_structure *Sub;

	Sub = NEW_OBJECT(field_theory::subfield_structure);

	Sub->init(FQ, Fq, verbose_level);

	if (f_v) {
		Sub->print_embedding();
	}

	//Orbiter->Int_vec.scan(genma_text, M, sz);
	Get_int_vector_from_label(genma_text, M, sz, verbose_level);

	if (sz != m * n) {
		cout << "sz != m * n" << endl;
		exit(1);
	}

	M2 = NEW_int(Sub->s * m * Sub->s * n);

	for (i = 0; i < m; i++) {
		Sub->field_reduction(M + i * n, n, M2 + (i * Sub->s) * Sub->s * n,
				verbose_level);
	}


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		snprintf(str, 1000, "field_reduction_Q%d_q%d_%d_%d.tex", FQ->q, Fq->q, m, n);
		fname.assign(str);
		snprintf(str, 1000, "Field Reduction");
		title.assign(str);





		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);



			ost << "$$" << endl;
			ost << "\\left[" << endl;
			L.int_matrix_print_tex(ost, M2, m * Sub->s, Sub->s * n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;

			Int_vec_print_fully(ost, M2, m * Sub->s * Sub->s * n);
			ost << "\\\\" << endl;



			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "coding_theory_domain::field_reduction written "
				"file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

		string fname_csv;

		fname_csv.assign(label);
		fname_csv.append(".csv");

		Fio.int_matrix_write_csv(fname_csv, M2, m * Sub->s, Sub->s * n);
	}

	FREE_int(M2);
	FREE_OBJECT(Sub);

	if (f_v) {
		cout << "coding_theory_domain::field_reduction done" << endl;
	}
}

void coding_theory_domain::CRC_encode_text(field_theory::nth_roots *Nth,
		ring_theory::unipoly_object &CRC_poly,
	std::string &text, std::string &fname,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::CRC_encode_text e=" << Nth->F->e << endl;
	}
	int l, i, j, h, a;
	char c;
	int *encoding;
	int len;


	l = text.size();
	encoding = NEW_int(5 * l);
	j = 0;
	for (i = 0; i < l; i++) {
		c = text[i];
		if (c >= 'A' && c <= 'Z') {
			a = 3 + c - 'A';
		}
		else if (c >= 'a' && c <= 'z') {
			a = 3 + c - 'a';
		}
		else if (c == ' ') {
			a = 0;
		}
		else if (c == ',') {
			a = 1;
		}
		else if (c == '.') {
			a = 2;
		}
		else {
			//cout << "unknown character " << c << " skipping" << endl;
			//exit(1);
			continue;
		}
		for (h = 0; h < 5; h++) {
			encoding[j++] = a % 2;
			a >>= 1;
		}
	}

	len = j;

	int degree;

	degree = Nth->FX->degree(CRC_poly);
	if (f_v) {
		cout << "coding_theory_domain::CRC_encode_text degree=" << degree << endl;
	}

	int nb_rows;
	int nb_cols;
	int I, IP, IPq;
	int nb_bits;

	nb_rows = 80;
	nb_cols = 72;

	nb_bits = Nth->F->e;

	IP = nb_rows * nb_cols + nb_rows + nb_cols;

	IPq = IP / nb_bits;



	int *v;
	int *information;
	int *information_and_parity;
	int *information_and_parity_Fq;
	int *codeword_Fq;
	geometry::geometry_global GG;

	information = NEW_int(nb_rows * nb_cols);
	information_and_parity = NEW_int(nb_rows * nb_cols + nb_rows + nb_cols);
	information_and_parity_Fq = NEW_int(IPq);
	codeword_Fq = NEW_int(IPq + degree);


	int *row_parity;
	int *col_parity;

	row_parity = NEW_int(nb_rows);
	col_parity = NEW_int(nb_cols);

	v = NEW_int(nb_bits);


	for (I = 0; I * nb_rows * nb_cols < len; I++) {

		Int_vec_zero(information, nb_rows * nb_cols);

		for (j = 0; j < nb_rows * nb_cols; j++) {
			h = I * nb_rows * nb_cols + j;
			if (h < len) {
				information[j] = encoding[h];
			}
			else {
				information[j] = 0;
			}
		}
		orbiter_kernel_system::file_io Fio;
		string fname_base;
		string fname_out;
		data_structures::string_tools String;
		char str[1000];


		fname_base.assign(fname);
		String.chop_off_extension(fname_base);

		sprintf(str, "_word%d", I);
		fname_base.append(str);



		fname_out.assign(fname_base);
		fname_out.append("_information.csv");


		//Fio.int_vec_write_csv(encoding, 5 * l, fname, "encoding");
		Fio.int_matrix_write_csv(fname_out, information, nb_rows, nb_cols);
		cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;

		for (j = 0; j < nb_cols; j++) {
			a = 0;
			for (i = 0; i < nb_rows; i++) {
				a += information[i * nb_cols + j];
				a %= 2;
			}
			col_parity[j] = a;
		}


		fname_out.assign(fname_base);
		fname_out.append("_col_parity.csv");


		//Fio.int_vec_write_csv(encoding, 5 * l, fname, "encoding");
		Fio.int_matrix_write_csv(fname_out, col_parity, 1, nb_cols);
		cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;


		for (i = 0; i < nb_rows; i++) {
			a = 0;
			for (j = 0; j < nb_cols; j++) {
				a += information[i * nb_cols + j];
				a %= 2;
			}
			row_parity[i] = a;
		}

		fname_out.assign(fname_base);
		fname_out.append("_row_parity.csv");


		//Fio.int_vec_write_csv(encoding, 5 * l, fname, "encoding");
		Fio.int_matrix_write_csv(fname_out, row_parity, 1, nb_rows);
		cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;

		Int_vec_copy(information, information_and_parity, nb_rows * nb_cols);
		Int_vec_copy(row_parity, information_and_parity + nb_rows * nb_cols, nb_rows);
		Int_vec_copy(col_parity, information_and_parity + nb_rows * nb_cols + nb_rows, nb_cols);


		fname_out.assign(fname_base);
		fname_out.append("_IP.csv");


		//Fio.int_vec_write_csv(encoding, 5 * l, fname, "encoding");
		Fio.int_matrix_write_csv(fname_out, information_and_parity, 1, nb_rows * nb_cols + nb_rows + nb_cols);
		cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;


		for (i = 0; i < IPq; i++) {
			for (h = 0; h < nb_bits; h++) {
				v[h] = information_and_parity[i * nb_bits + h];
			}
			a = GG.AG_element_rank(2, v, 1, nb_bits);
			information_and_parity_Fq[i] = a;
		}

		fname_out.assign(fname_base);
		fname_out.append("_IPq.csv");

		Fio.int_matrix_write_csv(fname_out, information_and_parity_Fq, 1, IPq);
		cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;

		ring_theory::unipoly_object P;

		Nth->FX->create_object_of_degree(P, IPq + degree);
		for (i = 0; i < IPq; i++) {
			a = information_and_parity_Fq[i];
			Nth->FX->s_i(P, i + degree) = a;
		}

		cout << "P=";
		Nth->FX->print_object(P, cout);
		cout << endl;

		ring_theory::unipoly_object Q;
		ring_theory::unipoly_object R;

		Nth->FX->create_object_of_degree(Q, IPq + degree);
		Nth->FX->create_object_of_degree(R, degree);

		Nth->FX->division_with_remainder(P, CRC_poly, Q, R, verbose_level);

		cout << "R=";
		Nth->FX->print_object(R, cout);
		cout << endl;

		Int_vec_copy(information_and_parity_Fq, codeword_Fq + degree, IPq);
		for (i = 0; i < degree; i++) {
			a = Nth->FX->s_i(R, i);
			codeword_Fq[i] = a;
		}

		fname_out.assign(fname_base);
		fname_out.append("_codeword_Fq.csv");

		Fio.int_matrix_write_csv(fname_out, codeword_Fq, 1, IPq + degree);
		cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;


	}

	if (f_v) {
		cout << "coding_theory_domain::CRC_encode_text done" << endl;
	}
}

void coding_theory_domain::encode_text_5bits(std::string &text,
		std::string &fname, int verbose_level)
{
	int l, i, j, h, a;
	char c;
	long int *encoding;
	int len;


	l = text.size();
	encoding = NEW_lint(5 * l);
	j = 0;
	for (i = 0; i < l; i++) {
		c = text[i];
		if (c >= 'A' && c <= 'Z') {
			a = 3 + c - 'A';
		}
		else if (c >= 'a' && c <= 'z') {
			a = 3 + c - 'a';
		}
		else if (c == ' ') {
			a = 0;
		}
		else if (c == ',') {
			a = 1;
		}
		else if (c == '.') {
			a = 2;
		}
		else {
			cout << "unknown character " << c << " skipping" << endl;
			//exit(1);
			continue;
		}
		for (h = 0; h < 5; h++) {
			encoding[j++] = a % 2;
			a >>= 1;
		}
	}

	len = j;

	orbiter_kernel_system::file_io Fio;

	//Fio.int_vec_write_csv(encoding, 5 * l, fname, "encoding");
	Fio.lint_matrix_write_csv(fname, encoding, 1, len);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
}


void coding_theory_domain::field_induction(std::string &fname_in,
		std::string &fname_out, int nb_bits, int verbose_level)
{
	int i, h, len, len2;
	long int *M;
	long int a;
	long int *M2;
	int *v;
	int m, n;
	geometry::geometry_global GG;


	orbiter_kernel_system::file_io Fio;

	cout << "Reading file " << fname_in << " of size " << Fio.file_size(fname_in) << endl;
	Fio.lint_matrix_read_csv(fname_in, M, m, n, verbose_level);
	len = m * n;
	len2 = (len + nb_bits - 1) / nb_bits;
	v = NEW_int(nb_bits);
	M2 = NEW_lint(len2);
	for (i = 0; i < len2; i++) {
		for (h = 0; h < nb_bits; h++) {
			v[h] = M[i * nb_bits + h];
		}
		a = GG.AG_element_rank(2, v, 1, nb_bits);
		M2[i] = a;
	}
	Fio.lint_matrix_write_csv(fname_out, M2, 1, len2);
	cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;

}

int coding_theory_domain::Hamming_distance(int *v1, int *v2, int n)
{
	int i, d;

	d = 0;
	for (i = 0; i < n; i++) {
		if (v1[i] != v2[i]) {
			d++;
		}
	}
	return d;
}


int coding_theory_domain::Hamming_distance_binary(int a, int b, int n)
{
	int i, d, u, v;

	d = 0;
	for (i = 0; i < n; i++) {
		u = a % 2;
		v = b % 2;
		if (u != v) {
			d++;
		}
		a >>= 1;
		b >>= 1;
	}
	return d;
}


void coding_theory_domain::generator_matrix_cyclic_code(
		field_theory::finite_field *F,
		int n,
		std::string &poly_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::generator_matrix_cyclic_code" << endl;
	}

	int *coeffs;
	int sz;
	int k;
	int d;
	int *M;
	int *v;
	int i, j;
	long int rk;
	long int *Rk;

	Int_vec_scan(poly_coeffs, coeffs, sz);
	d = sz - 1;
	k = n - d;

	M = NEW_int(k * n);
	Int_vec_zero(M, k * n);
	for (i = 0; i < k; i++) {
		Int_vec_copy(coeffs, M + i * n + i, d + 1);
	}

	cout << "generator matrix:" << endl;
	Int_matrix_print(M, k, n);


	cout << "generator matrix:" << endl;
	Int_vec_print_fully(cout, M, k * n);
	cout << endl;

	v = NEW_int(k);
	Rk = NEW_lint(n);
	for (j = 0; j < n; j++) {
		for (i = 0; i < k; i++) {
			v[i] = M[i * n + j];
		}
		F->PG_element_rank_modified_lint(
				v, 1, k, rk);
		Rk[j] = rk;
	}

	cout << "generator matrix in projective points:" << endl;
	Lint_vec_print_fully(cout, Rk, n);
	cout << endl;

	if (f_v) {
		cout << "coding_theory_domain::generator_matrix_cyclic_code" << endl;
	}
}


/*
 * twocoef.cpp
 *
 *  Created on: Oct 22, 2020
 *      Author: alissabrown
 *
 *	Received a lot of help from Anton and the recursive function in the possibleC function is modeled after code found at
 *	https://www.geeksforgeeks.org/print-all-combinations-of-given-length/
 *
 *
 */

void coding_theory_domain::find_CRC_polynomials(
		field_theory::finite_field *F,
		int t, int da, int dc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::find_CRC_polynomials t=" << t
				<< " info=" << da << " check=" << dc << endl;
	}

	//int dc = 4; //dc is the number of parity bits & degree of g(x)
	//int da = 4; //da is the degree of the information polynomial
	int A[da + dc];
		// we have da information bits, which we can think of
		// as the coefficients of a polynomial.
		// After multiplying by x^dc,
		// A(x) has degree at most ad + dc - 1.
	long int nb_sol = 0;



	int C[dc + 1]; //Array C (what we divide by)
		// C(x) has the leading coefficient of one included,
		// hence we need one more array element

	int i = 0;

	for (i = 0; i <= dc; i++) {
		C[i] = 0;
	}


	std::vector<std::vector<int>> Solutions;

	if (F->q == 2) {
		search_for_CRC_polynomials_binary(t, da, A, dc, C, 0,
				nb_sol, Solutions, verbose_level - 1);
	}
	else {
		search_for_CRC_polynomials(t, da, A, dc, C, 0, F,
				nb_sol, Solutions, verbose_level - 1);
	}

	cout << "coding_theory_domain::find_CRC_polynomials info=" << da
			<< " check=" << dc << " nb_sol=" << nb_sol << endl;

	for (i = 0; i < Solutions.size(); i++) {
		cout << i << " : ";
		for (int j = dc; j >= 0; j--) {
			cout << Solutions[i][j];
		}
		cout << endl;
	}
	cout << "coding_theory_domain::find_CRC_polynomials info=" << da
			<< " check=" << dc << " nb_sol=" << nb_sol << endl;

}

void coding_theory_domain::search_for_CRC_polynomials(int t,
		int da, int *A, int dc, int *C,
		int i, field_theory::finite_field *F,
		long int &nb_sol,
		std::vector<std::vector<int> > &Solutions,
		int verbose_level)
{

	if (i == dc + 1) {

		int ret;

		if (t >= 2) {
			ret = test_all_two_bit_patterns(da, A, dc, C, F, verbose_level);
			if (ret && t >= 3) {
				ret = test_all_three_bit_patterns(da, A, dc, C, F, verbose_level);
			}
		}
		else {
			cout << "illegal value for t, t=" << t << endl;
			exit(1);
		}
		if (ret) {
			cout << "solution " << nb_sol << " is ";
			for (int j = dc; j >= 0; j--) {
				cout << C[j];
			}
			cout << endl;

			vector<int> sol;

			for (int j = 0; j <= dc; j++) {
				sol.push_back(C[j]);
			}
			Solutions.push_back(sol);


			nb_sol++;
		}

		return;
	}

	if (i == dc) {

		// C(x) has a leading coefficient of one:
		C[i] = 1;
		search_for_CRC_polynomials(t, da, A, dc, C,
				i + 1, F, nb_sol, Solutions, verbose_level);

	}
	else {
		int c;

		for (c = 0; c < F->q; c++) {

			C[i] = c;

			search_for_CRC_polynomials(t, da, A, dc, C,
					i + 1, F, nb_sol, Solutions, verbose_level);
		}
	}
}

void coding_theory_domain::search_for_CRC_polynomials_binary(int t,
		int da, int *A, int dc, int *C, int i,
		long int &nb_sol,
		std::vector<std::vector<int> > &Solutions,
		int verbose_level)
{

	if (i == dc + 1) {

		int ret;

		if (t >= 2) {
			ret = test_all_two_bit_patterns_binary(da, A, dc, C, verbose_level);
			if (ret && t >= 3) {
				ret = test_all_three_bit_patterns_binary(da, A, dc, C, verbose_level);
			}
		}
		else {
			cout << "illegal value for t, t=" << t << endl;
			exit(1);
		}
		if (ret) {
			cout << "solution " << nb_sol << " is ";
			for (int j = dc; j >= 0; j--) {
				cout << C[j];
			}
			cout << endl;

			vector<int> sol;

			for (int j = 0; j <= dc; j++) {
				sol.push_back(C[j]);
			}
			Solutions.push_back(sol);


			nb_sol++;
		}

		return;
	}

	if (i == dc) {

		C[i] = 1;
		search_for_CRC_polynomials_binary(t, da, A, dc, C,
				i + 1, nb_sol, Solutions, verbose_level);


	}
	else {
		int c;

		for (c = 0; c < 2; c++) {

			C[i] = c;

			search_for_CRC_polynomials_binary(t, da, A, dc, C,
					i + 1, nb_sol, Solutions, verbose_level);
		}
	}
}


int coding_theory_domain::test_all_two_bit_patterns(int da, int *A,
		int dc, int *C,
		field_theory::finite_field *F,
		int verbose_level)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{

	//cout << "choosetwo" << endl;

	int f_v = (verbose_level >= 1);

	int i;
	int j;
	int k;
	int ai, aj;
	int ret;
	int B[da + dc];

	if (f_v) {
		cout << "testing polynomial: ";
		for (k = dc; k >= 0; k--) {
			cout << C[k];
		}
		cout << endl;
	}

	for (i = 0; i < da; i++) {
		A[i] = 0;
	}

	for (i = 0; i < da; i++) {

		for (ai = 1; ai < F->q; ai++) {

			A[i] = ai;

			for (j = i + 1; j < da; j++) {

				for (aj = 1; aj < F->q; aj++) {

					A[j] = aj;

					for (k = 0; k < dc; k++) {
						B[k] = 0;
					}
					for (k = 0; k < da; k++) {
						B[dc + k] = A[k];
					}

					if (f_v) {
						cout << "testing error pattern: ";
						for (k = dc + da - 1; k >= 0; k--) {
							cout << B[k];
						}
					}



					ret = remainder_is_nonzero (da, B, dc, C, F);

					if (f_v) {
						cout << " : ";
						for (k = dc - 1; k >= 0; k--) {
							cout << B[k];
						}
						cout << endl;
					}

					if (!ret) {
						return false;
					}

				}
				A[j] = 0;
			}

		}
		A[i] = 0;
	}
	return true;
}

int coding_theory_domain::test_all_three_bit_patterns(int da, int *A,
		int dc, int *C,
		field_theory::finite_field *F,
		int verbose_level)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{

	//cout << "choosetwo" << endl;

	int f_v = (verbose_level >= 1);

	int i1, i2, i3;
	int k;
	int a1, a2, a3;
	int ret;
	int B[da + dc];

	if (f_v) {
		cout << "testing polynomial: ";
		for (k = dc; k >= 0; k--) {
			cout << C[k];
		}
		cout << endl;
	}

	for (int h = 0; h < da; h++) {
		A[h] = 0;
	}

	for (i1 = 0; i1 < da; i1++) {

		for (a1 = 1; a1 < F->q; a1++) {

			A[i1] = a1;

			for (i2 = i1 + 1; i2 < da; i2++) {

				for (a2 = 1; a2 < F->q; a2++) {

					A[i2] = a2;

					for (i3 = i2 + 1; i3 < da; i3++) {

						for (a3 = 1; a3 < F->q; a3++) {

							A[i3] = a3;

							for (int h = 0; h < dc; h++) {
								B[h] = 0;
							}
							for (int h = 0; h < da; h++) {
								B[dc + h] = A[h];
							}

							if (f_v) {
								cout << "testing error pattern: ";
								for (int h = dc + da - 1; h >= 0; h--) {
									cout << B[h];
								}
							}



							ret = remainder_is_nonzero (da, B, dc, C, F);

							if (f_v) {
								cout << " : ";
								for (int h = dc - 1; h >= 0; h--) {
									cout << B[h];
								}
								cout << endl;
							}

							if (!ret) {
								return false;
							}

						}
						A[i3] = 0;
					}
				}
				A[i2] = 0;
			}
		}
		A[i1] = 0;
	}
	return true;
}

int coding_theory_domain::test_all_two_bit_patterns_binary(int da, int *A,
		int dc, int *C,
		int verbose_level)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{

	//cout << "choosetwo" << endl;

	int f_v = (verbose_level >= 1);

	int i;
	int j;
	int k;
	int ret;
	int B[da + dc];

	if (f_v) {
		cout << "testing polynomial: ";
		for (k = dc; k >= 0; k--) {
			cout << C[k];
		}
		cout << endl;
	}

	for (i = 0; i < da; i++) {
		A[i] = 0;
	}

	for (i = 0; i < da; i++) {


		A[i] = 1;

		for (j = i + 1; j < da; j++) {


			A[j] = 1;

			for (k = 0; k < dc; k++) {
				B[k] = 0;
			}
			for (k = 0; k < da; k++) {
				B[dc + k] = A[k];
			}

			if (f_v) {
				cout << "testing error pattern: ";
				for (k = dc + da - 1; k >= 0; k--) {
					cout << B[k];
				}
			}



			ret = remainder_is_nonzero_binary(da, B, dc, C);

			if (f_v) {
				cout << " : ";
				for (k = dc - 1; k >= 0; k--) {
					cout << B[k];
				}
				cout << endl;
			}

			if (!ret) {
				return false;
			}

			A[j] = 0;
		}

		A[i] = 0;
	}
	return true;
}

int coding_theory_domain::test_all_three_bit_patterns_binary(int da, int *A,
		int dc, int *C,
		int verbose_level)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{

	//cout << "choosetwo" << endl;

	int f_v = (verbose_level >= 1);

	int i1, i2, i3;
	int k;
	int ret;
	int B[da + dc];

	if (f_v) {
		cout << "testing polynomial: ";
		for (k = dc; k >= 0; k--) {
			cout << C[k];
		}
		cout << endl;
	}

	for (int h = 0; h < da; h++) {
		A[h] = 0;
	}

	for (i1 = 0; i1 < da; i1++) {

		A[i1] = 1;

		for (i2 = i1 + 1; i2 < da; i2++) {


			A[i2] = 1;

			for (i3 = i2 + 1; i3 < da; i3++) {


				A[i3] = 1;

				for (int h = 0; h < dc; h++) {
					B[h] = 0;
				}
				for (int h = 0; h < da; h++) {
					B[dc + h] = A[h];
				}

				if (f_v) {
					cout << "testing error pattern: ";
					for (int h = dc + da - 1; h >= 0; h--) {
						cout << B[h];
					}
				}



				ret = remainder_is_nonzero_binary(da, B, dc, C);

				if (f_v) {
					cout << " : ";
					for (int h = dc - 1; h >= 0; h--) {
						cout << B[h];
					}
					cout << endl;
				}

				if (!ret) {
					return false;
				}

				A[i3] = 0;
			}
			A[i2] = 0;
		}

		A[i1] = 0;
	}
	return true;
}


int coding_theory_domain::remainder_is_nonzero(int da, int *A,
		int db, int *B, field_theory::finite_field *F)
// returns true if the remainder of A after division by B is nonzero
{

	int i, j, k, a, mav;

	for (i = da + db - 1; i >= db; i--) {

		a = A[i];

		if (a) {

			mav = F->negate(F->inverse(a));

			for (j = db, k = i; j >= 0; j--, k--) {

				//A[k] = (A[k] + B[j]) % 2;
				A[k] = F->add(A[k], F->mult(mav, B[j]));
					//A[k] = subtraction(A[k], B[j], p);
					//A[k]=(A[k]+2-B[j])%2;
			}
		}
	}


	for (int k = db - 1; k >= 0; k--) {
		if (A[k]) {
			return true;
		}
	}
	return false;
}


int coding_theory_domain::remainder_is_nonzero_binary(int da, int *A,
		int db, int *B)
// returns true if the remainder of A after division by B is nonzero
{

	int i, j, k, a;

	for (i = da + db - 1; i >= db; i--) {

		a = A[i];

		if (a) {

			//mav = F->negate(F->inverse(a));

			for (j = db, k = i; j >= 0; j--, k--) {

				A[k] = (A[k] + B[j]) % 2;
				//A[k] = F->add(A[k], F->mult(mav, B[j]));
					//A[k] = subtraction(A[k], B[j], p);
					//A[k]=(A[k]+2-B[j])%2;
			}
		}
	}


	for (int k = db - 1; k >= 0; k--) {
		if (A[k]) {
			return true;
		}
	}
	return false;
}

uint32_t coding_theory_domain::crc32(const char *s, size_t n)
// polynomial x^32 + x^26 + x^23 + x^22 + x^16 + x^12 + x^11
// + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
{
	uint32_t crc = 0xFFFFFFFF;

	for (size_t i = 0; i < n; i++) {
		char ch = s[i];
		for (size_t j = 0; j < 8; j++) {
			uint32_t b = (ch^crc) & 1;
			crc >>= 1;
			if (b) {
				crc = crc^0xEDB88320; // reversed polynomial
			}
			ch >>= 1;
		}
	}
	return ~crc;
}

void coding_theory_domain::crc32_test(int block_length, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::crc32_test block_length = " << block_length << endl;
	}
	//cout << "sizeof(int) = " << (int) sizeof(int) << endl;
	//cout << "sizeof(long int) = " << (int) sizeof(long int) << endl;
	unsigned int i;
	uint32_t crc;
	char *buffer;
	long int cnt = 0;
	vector<unsigned int> V;

	buffer = (char *) &i;
	for (i = 0; i < 0xFFFFFFFF; i++) {
		if ((i & 0xFFFFF) == 0) {
			cout << "i >> 20: " << (int) (i >> 20) << " cnt = " << cnt << endl;
		}
		crc = crc32(buffer, 4);
		if (crc == 0) {
			cout << cnt << " : " << i << endl;
			cnt++;
			V.push_back(i);
		}
	}
	data_structures::algorithms Algo;

	cout << "cnt = " << cnt << endl;
	for (i = 0; i < V.size(); i++) {
		cout << i << " : " << V[i] << " : ";

		Algo.print_uint32_hex(cout, V[i]);
		cout << " : ";
		Algo.print_uint32_binary(cout, V[i]);
		//Algo.print_uint32_hex(cout, ~V[i]);

		cout << endl;
	}

	if (f_v) {
		cout << "coding_theory_domain::crc32_test" << endl;
	}
}

void coding_theory_domain::crc256_test_k_subsets(int message_length, int R, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::crc256_test_k_subsets message_length in bytes = " << message_length << " R=" << R << " k=" << k << endl;
	}
	//cout << "sizeof(int) = " << (int) sizeof(int) << endl;
	//cout << "sizeof(long int) = " << (int) sizeof(long int) << endl;
	int i, h;
	char *input;
	char *check;
	int *message;
	int block_length_in_bits;
	int R8;
	int block_length;
	int message_length_in_bits;
	long int cnt;
	data_structures::bitvector B;
	int *set;
	vector<unsigned int> V;

	R8 = R * 8;
	block_length = message_length + R;
	block_length_in_bits = block_length * 8;
	message_length_in_bits = message_length * 8;

	set = NEW_int(k);
	check = NEW_char(R);
	message = NEW_int(message_length_in_bits);

	B.allocate(block_length_in_bits);

	input = (char *) B.get_data();
	cnt = 0;

	combinatorics::combinatorics_domain Combi;


	Int_vec_zero(message, message_length_in_bits);

	Combi.first_k_subset(set, message_length_in_bits, k);

	while (TRUE) {


		for (i = 0; i < k; i++) {
			message[set[i]] = 1;
		}

		B.zero();
		for (h = 0; h < message_length_in_bits; h++) {
			if (message[h]) {
				B.set_bit(h);
			}
		}


		if (R == 30) {
			CRC_BCH256_771_divide(input, check);
		}
		else if (R == 4) {
			uint32_t crc;
			char *p;

			p = (char *) &crc;
			crc = crc32(input, message_length);

			check[0] = p[0];
			check[1] = p[1];
			check[2] = p[2];
			check[3] = p[3];
		}
		else {
			cout << "coding_theory_domain::crc256_test_k_subsets I don't have a code of that length" << endl;
			exit(1);
		}


		for (h = 0; h < R; h++) {
			if (check[h]) {
				break;
			}
		}
		if ((cnt & 0xFFFFFF) == 0) {
			cout << cnt << " : ";
			Int_vec_print(cout, set, k);
			cout << endl;
			cnt++;
		}

		if (h == R) {
			V.push_back(cnt);
			cout << "remainder is zero, cnt=" << cnt;
			cout << " : ";
			Int_vec_print(cout, set, k);
			cout << endl;
		}


		for (i = 0; i < k; i++) {
			message[set[i]] = 0;
		}


		if (!Combi.next_k_subset(set, message_length_in_bits, k)) {
			break;
		}

		cnt++;



	}

	cout << "Number of undetected errors = " << V.size() << endl;



	if (f_v) {
		cout << "coding_theory_domain::crc256_test_k_subsets" << endl;
	}
}

void coding_theory_domain::crc32_remainders(int message_length, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::crc32_remainders "
				"message_length in bytes = " << message_length << endl;
	}

	uint32_t *Crc;
	uint32_t crc;
	int *Table;
	int message_length_in_bits;
	int i, j, a;
	int R = 4;

	message_length_in_bits = message_length * 8;


	crc32_remainders_compute(message_length, R, Crc, verbose_level);


	Table = NEW_int(message_length_in_bits * 32);

	for (i = 0; i < message_length_in_bits; i++) {

		crc = Crc[i];

		for (j = 0; j < 32; j++) {
			a = crc % 2;
			Table[i * 32 + j] = a;
			crc >>= 1;
		}
	}

	orbiter_kernel_system::file_io Fio;
	string fname;
	char str[1000];

	sprintf(str, "crc32_remainders_M%d.csv", message_length);
	fname.assign(str);

	Fio.int_matrix_write_csv(fname, Table, message_length_in_bits, 32);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "coding_theory_domain::crc32_remainders done" << endl;
	}

}


void coding_theory_domain::crc32_remainders_compute(int message_length, int R, uint32_t *&Crc, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::crc32_remainders_compute message_length in bytes = " << message_length << " R=" << R << endl;
	}
	//cout << "sizeof(int) = " << (int) sizeof(int) << endl;
	//cout << "sizeof(long int) = " << (int) sizeof(long int) << endl;
	int i, h;
	char *input;
	char *check;
	int *message;
	int block_length_in_bits;
	int R8;
	int block_length;
	int message_length_in_bits;
	long int cnt;
	data_structures::bitvector B;
	int *set;
	vector<unsigned int> V;
	int k = 1;

	R8 = R * 8;
	block_length = message_length + R;
	block_length_in_bits = block_length * 8;
	message_length_in_bits = message_length * 8;

	set = NEW_int(k);
	check = NEW_char(R);
	message = NEW_int(message_length_in_bits);

	B.allocate(block_length_in_bits);

	input = (char *) B.get_data();
	cnt = 0;

	combinatorics::combinatorics_domain Combi;


	Int_vec_zero(message, message_length_in_bits);


	Crc = (uint32_t *) NEW_int(message_length_in_bits);

	Combi.first_k_subset(set, message_length_in_bits, k);

	while (TRUE) {


		for (i = 0; i < k; i++) {
			message[set[i]] = 1;
		}

		B.zero();
		for (h = 0; h < message_length_in_bits; h++) {
			if (message[h]) {
				B.set_bit(h);
			}
		}


		if (R == 30) {
			CRC_BCH256_771_divide(input, check);
		}
		else if (R == 4) {
			uint32_t crc;
			char *p;

			p = (char *) &crc;
			crc = crc32(input, message_length);

			Crc[cnt] = crc;

			check[0] = p[0];
			check[1] = p[1];
			check[2] = p[2];
			check[3] = p[3];
		}
		else {
			cout << "coding_theory_domain::crc32_remainders_compute "
					"I don't have a code of that length" << endl;
			exit(1);
		}


		for (h = 0; h < R; h++) {
			if (check[h]) {
				break;
			}
		}
		if ((cnt & 0xFFFFFF) == 0) {
			cout << cnt << " : ";
			Int_vec_print(cout, set, k);
			cout << endl;
			cnt++;
		}

		if (h == R) {
			V.push_back(cnt);
			cout << "remainder is zero, cnt=" << cnt;
			cout << " : ";
			Int_vec_print(cout, set, k);
			cout << endl;
		}


		for (i = 0; i < k; i++) {
			message[set[i]] = 0;
		}


		if (!Combi.next_k_subset(set, message_length_in_bits, k)) {
			break;
		}

		cnt++;



	}

	if (cnt != message_length_in_bits) {
		cout << "coding_theory_domain::crc32_remainders_compute "
				"cnt != message_length_in_bits" << endl;
		exit(1);
	}



	cout << "Number of undetected errors = " << V.size() << endl;

	//FREE_int((int *) Crc);

	if (f_v) {
		cout << "coding_theory_domain::crc32_remainders_compute" << endl;
	}
}

// the size of the array B is  255 x 31
static const unsigned char B[] = {
  1, 26,210, 24,138,148,160, 58,108,199, 95, 56,  9,205,194,193,  3,248,110,150, 24,169,192,212,112,144, 97,109,174,253,  1,
  2, 52,185, 48,  9, 53, 93,116,216,147,190,112, 18,135,153,159,  6,237,220, 49, 48, 79,157,181,224, 61,194,218, 65,231,  2,
  3, 46,107, 40,131,161,253, 78,180, 84,225, 72, 27, 74, 91, 94,  5, 21,178,167, 40,230, 93, 97,144,173,163,183,239, 26,  3,
  4,104,111, 96, 18,106,186,232,173, 59, 97,224, 36, 19, 47, 35, 12,199,165, 98, 96,158, 39,119,221,122,153,169,130,211,  4,
  5,114,189,120,152,254, 26,210,193,252, 62,216, 45,222,237,226, 15, 63,203,244,120, 55,231,163,173,234,248,196, 44, 46,  5,
  6, 92,214, 80, 27, 95,231,156,117,168,223,144, 54,148,182,188, 10, 42,121, 83, 80,209,186,194, 61, 71, 91,115,195, 52,  6,
  7, 70,  4, 72,145,203, 71,166, 25,111,128,168, 63, 89,116,125,  9,210, 23,197, 72,120,122, 22, 77,215, 58, 30,109,201,  7,
  8,208,222,192, 36,212,105,205, 71,118,194,221, 72, 38, 94, 70, 24,147, 87,196,192, 33, 78,238,167,244, 47, 79, 25,187,  8,
  9,202, 12,216,174, 64,201,247, 43,177,157,229, 65,235,156,135, 27,107, 57, 82,216,136,142, 58,215,100, 78, 34,183, 70,  9,
 10,228,103,240, 45,225, 52,185,159,229,124,173, 90,161,199,217, 30,126,139,245,240,110,211, 91, 71,201,237,149, 88, 92, 10,
 11,254,181,232,167,117,148,131,243, 34, 35,149, 83,108,  5, 24, 29,134,229, 99,232,199, 19,143, 55, 89,140,248,246,161, 11,
 12,184,177,160, 54,190,211, 37,234, 77,163, 61,108, 53,113,101, 20, 84,242,166,160,191,105,153,122,142,182,230,155,104, 12,
 13,162, 99,184,188, 42,115, 31,134,138,252,  5,101,248,179,164, 23,172,156, 48,184, 22,169, 77, 10, 30,215,139, 53,149, 13,
 14,140,  8,144, 63,139,142, 81, 50,222, 29, 77,126,178,232,250, 18,185, 46,151,144,240,244, 44,154,179,116, 60,218,143, 14,
 15,150,218,136,181, 31, 46,107, 94, 25, 66,117,119,127, 42, 59, 17, 65, 64,  1,136, 89, 52,248,234, 35, 21, 81,116,114, 15,
 16,189,161,157, 72,181,210,135,142,236,153,167,144, 76,188,140, 48, 59,174,149,157, 66,156,193, 83,245, 94,158, 50,107, 16,
 17,167,115,133,194, 33,114,189,226, 43,198,159,153,129,126, 77, 51,195,192,  3,133,235, 92, 21, 35,101, 63,243,156,150, 17,
 18,137, 24,173, 65,128,143,243, 86,127, 39,215,130,203, 37, 19, 54,214,114,164,173, 13,  1,116,179,200,156, 68,115,140, 18,
 19,147,202,181,203, 20, 47,201, 58,184,120,239,139,  6,231,210, 53, 46, 28, 50,181,164,193,160,195, 88,253, 41,221,113, 19,
 20,213,206,253, 90,223,104,111, 35,215,248, 71,180, 95,147,175, 60,252, 11,247,253,220,187,182,142,143,199, 55,176,184, 20,
 21,207, 28,229,208, 75,200, 85, 79, 16,167,127,189,146, 81,110, 63,  4,101, 97,229,117,123, 98,254, 31,166, 90, 30, 69, 21,
 22,225,119,205, 83,234, 53, 27,251, 68, 70, 55,166,216, 10, 48, 58, 17,215,198,205,147, 38,  3,110,178,  5,237,241, 95, 22,
 23,251,165,213,217,126,149, 33,151,131, 25, 15,175, 21,200,241, 57,233,185, 80,213, 58,230,215, 30, 34,100,128, 95,162, 23,
 24,109,127, 93,108, 97,187, 74,201,154, 91,122,216,106,226,202, 40,168,249, 81, 93, 99,210, 47,244,  1,113,209, 43,208, 24,
 25,119,173, 69,230,245, 27,112,165, 93,  4, 66,209,167, 32, 11, 43, 80,151,199, 69,202, 18,251,132,145, 16,188,133, 45, 25,
 26, 89,198,109,101, 84,230, 62, 17,  9,229, 10,202,237,123, 85, 46, 69, 37, 96,109, 44, 79,154, 20, 60,179, 11,106, 55, 26,
 27, 67, 20,117,239,192, 70,  4,125,206,186, 50,195, 32,185,148, 45,189, 75,246,117,133,143, 78,100,172,210,102,196,202, 27,
 28,  5, 16, 61,126, 11,  1,162,100,161, 58,154,252,121,205,233, 36,111, 92, 51, 61,253,245, 88, 41,123,232,120,169,  3, 28,
 29, 31,194, 37,244,159,161,152,  8,102,101,162,245,180, 15, 40, 39,151, 50,165, 37, 84, 53,140, 89,235,137, 21,  7,254, 29,
 30, 49,169, 13,119, 62, 92,214,188, 50,132,234,238,254, 84,118, 34,130,128,  2, 13,178,104,237,201, 70, 42,162,232,228, 30,
 31, 43,123, 21,253,170,252,236,208,245,219,210,231, 51,150,183, 33,122,238,148, 21, 27,168, 57,185,214, 75,207, 70, 25, 31,
 32,103, 95, 39,144,119,185, 19,  1,197, 47, 83, 61,152,101,  5, 96,118, 65, 55, 39,132, 37,159,166,247,188, 33,100,214, 32,
 33,125,141, 63, 26,227, 25, 41,109,  2,112,107, 52, 85,167,196, 99,142, 47,161, 63, 45,229, 75,214,103,221, 76,202, 43, 33,
 34, 83,230, 23,153, 66,228,103,217, 86,145, 35, 47, 31,252,154,102,155,157,  6, 23,203,184, 42, 70,202,126,251, 37, 49, 34,
 35, 73, 52, 15, 19,214, 68, 93,181,145,206, 27, 38,210, 62, 91,101, 99,243,144, 15, 98,120,254, 54, 90, 31,150,139,204, 35,
 36, 15, 48, 71,130, 29,  3,251,172,254, 78,179, 25,139, 74, 38,108,177,228, 85, 71, 26,  2,232,123,141, 37,136,230,  5, 36,
 37, 21,226, 95,  8,137,163,193,192, 57, 17,139, 16, 70,136,231,111, 73,138,195, 95,179,194, 60, 11, 29, 68,229, 72,248, 37,
 38, 59,137,119,139, 40, 94,143,116,109,240,195, 11, 12,211,185,106, 92, 56,100,119, 85,159, 93,155,176,231, 82,167,226, 38,
 39, 33, 91,111,  1,188,254,181, 24,170,175,251,  2,193, 17,120,105,164, 86,242,111,252, 95,137,235, 32,134, 63,  9, 31, 39,
 40,183,129,231,180,163,208,222, 70,179,237,142,117,190, 59, 67,120,229, 22,243,231,165,107,113,  1,  3,147,110,125,109, 40,
 41,173, 83,255, 62, 55,112,228, 42,116,178,182,124,115,249,130,123, 29,120,101,255, 12,171,165,113,147,242,  3,211,144, 41,
 42,131, 56,215,189,150,141,170,158, 32, 83,254,103, 57,162,220,126,  8,202,194,215,234,246,196,225, 62, 81,180, 60,138, 42,
 43,153,234,207, 55,  2, 45,144,242,231, 12,198,110,244, 96, 29,125,240,164, 84,207, 67, 54, 16,145,174, 48,217,146,119, 43,
 44,223,238,135,166,201,106, 54,235,136,140,110, 81,173, 20, 96,116, 34,179,145,135, 59, 76,  6,220,121, 10,199,255,190, 44,
 45,197, 60,159, 44, 93,202, 12,135, 79,211, 86, 88, 96,214,161,119,218,221,  7,159,146,140,210,172,233,107,170, 81, 67, 45,
 46,235, 87,183,175,252, 55, 66, 51, 27, 50, 30, 67, 42,141,255,114,207,111,160,183,116,209,179, 60, 68,200, 29,190, 89, 46,
 47,241,133,175, 37,104,151,120, 95,220,109, 38, 74,231, 79, 62,113, 55,  1, 54,175,221, 17,103, 76,212,169,112, 16,164, 47,
 48,218,254,186,216,194,107,148,143, 41,182,244,173,212,217,137, 80, 77,239,162,186,198,185, 94,245,  2,226,191, 86,189, 48,
 49,192, 44,162, 82, 86,203,174,227,238,233,204,164, 25, 27, 72, 83,181,129, 52,162,111,121,138,133,146,131,210,248, 64, 49,
 50,238, 71,138,209,247, 54,224, 87,186,  8,132,191, 83, 64, 22, 86,160, 51,147,138,137, 36,235, 21, 63, 32,101, 23, 90, 50,
 51,244,149,146, 91, 99,150,218, 59,125, 87,188,182,158,130,215, 85, 88, 93,  5,146, 32,228, 63,101,175, 65,  8,185,167, 51,
 52,178,145,218,202,168,209,124, 34, 18,215, 20,137,199,246,170, 92,138, 74,192,218, 88,158, 41, 40,120,123, 22,212,110, 52,
 53,168, 67,194, 64, 60,113, 70, 78,213,136, 44,128, 10, 52,107, 95,114, 36, 86,194,241, 94,253, 88,232, 26,123,122,147, 53,
 54,134, 40,234,195,157,140,  8,250,129,105,100,155, 64,111, 53, 90,103,150,241,234, 23,  3,156,200, 69,185,204,149,137, 54,
 55,156,250,242, 73,  9, 44, 50,150, 70, 54, 92,146,141,173,244, 89,159,248,103,242,190,195, 72,184,213,216,161, 59,116, 55,
 56, 10, 32,122,252, 22,  2, 89,200, 95,116, 41,229,242,135,207, 72,222,184,102,122,231,247,176, 82,246,205,240, 79,  6, 56,
 57, 16,242, 98,118,130,162, 99,164,152, 43, 17,236, 63, 69, 14, 75, 38,214,240, 98, 78, 55,100, 34,102,172,157,225,251, 57,
 58, 62,153, 74,245, 35, 95, 45, 16,204,202, 89,247,117, 30, 80, 78, 51,100, 87, 74,168,106,  5,178,203, 15, 42, 14,225, 58,
 59, 36, 75, 82,127,183,255, 23,124, 11,149, 97,254,184,220,145, 77,203, 10,193, 82,  1,170,209,194, 91,110, 71,160, 28, 59,
 60, 98, 79, 26,238,124,184,177,101,100, 21,201,193,225,168,236, 68, 25, 29,  4, 26,121,208,199,143,140, 84, 89,205,213, 60,
 61,120,157,  2,100,232, 24,139,  9,163, 74,241,200, 44,106, 45, 71,225,115,146,  2,208, 16, 19,255, 28, 53, 52, 99, 40, 61,
 62, 86,246, 42,231, 73,229,197,189,247,171,185,211,102, 49,115, 66,244,193, 53, 42, 54, 77,114,111,177,150,131,140, 50, 62,
 63, 76, 36, 50,109,221, 69,255,209, 48,244,129,218,171,243,178, 65, 12,175,163, 50,159,141,166, 31, 33,247,238, 34,207, 63,
 64,206,190, 78, 61,238,111, 38,  2,151, 94,166,122, 45,202, 10,192,236,130,110, 78, 21, 74, 35, 81,243,101, 66,200,177, 64,
 65,212,108, 86,183,122,207, 28,110, 80,  1,158,115,224,  8,203,195, 20,236,248, 86,188,138,247, 33, 99,  4, 47,102, 76, 65,
 66,250,  7,126, 52,219, 50, 82,218,  4,224,214,104,170, 83,149,198,  1, 94, 95,126, 90,215,150,177,206,167,152,137, 86, 66,
 67,224,213,102,190, 79,146,104,182,195,191,238, 97,103,145, 84,197,249, 48,201,102,243, 23, 66,193, 94,198,245, 39,171, 67,
 68,166,209, 46, 47,132,213,206,175,172, 63, 70, 94, 62,229, 41,204, 43, 39, 12, 46,139,109, 84,140,137,252,235, 74, 98, 68,
 69,188,  3, 54,165, 16,117,244,195,107, 96,126, 87,243, 39,232,207,211, 73,154, 54, 34,173,128,252, 25,157,134,228,159, 69,
 70,146,104, 30, 38,177,136,186,119, 63,129, 54, 76,185,124,182,202,198,251, 61, 30,196,240,225,108,180, 62, 49, 11,133, 70,
 71,136,186,  6,172, 37, 40,128, 27,248,222, 14, 69,116,190,119,201, 62,149,171,  6,109, 48, 53, 28, 36, 95, 92,165,120, 71,
 72, 30, 96,142, 25, 58,  6,235, 69,225,156,123, 50, 11,148, 76,216,127,213,170,142, 52,  4,205,246,  7, 74, 13,209, 10, 72,
 73,  4,178,150,147,174,166,209, 41, 38,195, 67, 59,198, 86,141,219,135,187, 60,150,157,196, 25,134,151, 43, 96,127,247, 73,
 74, 42,217,190, 16, 15, 91,159,157,114, 34, 11, 32,140, 13,211,222,146,  9,155,190,123,153,120, 22, 58,136,215,144,237, 74,
 75, 48, 11,166,154,155,251,165,241,181,125, 51, 41, 65,207, 18,221,106,103, 13,166,210, 89,172,102,170,233,186, 62, 16, 75,
 76,118, 15,238, 11, 80,188,  3,232,218,253,155, 22, 24,187,111,212,184,112,200,238,170, 35,186, 43,125,211,164, 83,217, 76,
 77,108,221,246,129,196, 28, 57,132, 29,162,163, 31,213,121,174,215, 64, 30, 94,246,  3,227,110, 91,237,178,201,253, 36, 77,
 78, 66,182,222,  2,101,225,119, 48, 73, 67,235,  4,159, 34,240,210, 85,172,249,222,229,190, 15,203, 64, 17,126, 18, 62, 78,
 79, 88,100,198,136,241, 65, 77, 92,142, 28,211, 13, 82,224, 49,209,173,194,111,198, 76,126,219,187,208,112, 19,188,195, 79,
 80,115, 31,211,117, 91,189,161,140,123,199,  1,234, 97,118,134,240,215, 44,251,211, 87,214,226,  2,  6, 59,220,250,218, 80,
 81,105,205,203,255,207, 29,155,224,188,152, 57,227,172,180, 71,243, 47, 66,109,203,254, 22, 54,114,150, 90,177, 84, 39, 81,
 82, 71,166,227,124,110,224,213, 84,232,121,113,248,230,239, 25,246, 58,240,202,227, 24, 75, 87,226, 59,249,  6,187, 61, 82,
 83, 93,116,251,246,250, 64,239, 56, 47, 38, 73,241, 43, 45,216,245,194,158, 92,251,177,139,131,146,171,152,107, 21,192, 83,
 84, 27,112,179,103, 49,  7, 73, 33, 64,166,225,206,114, 89,165,252, 16,137,153,179,201,241,149,223,124,162,117,120,  9, 84,
 85,  1,162,171,237,165,167,115, 77,135,249,217,199,191,155,100,255,232,231, 15,171, 96, 49, 65,175,236,195, 24,214,244, 85,
 86, 47,201,131,110,  4, 90, 61,249,211, 24,145,220,245,192, 58,250,253, 85,168,131,134,108, 32, 63, 65, 96,175, 57,238, 86,
 87, 53, 27,155,228,144,250,  7,149, 20, 71,169,213, 56,  2,251,249,  5, 59, 62,155, 47,172,244, 79,209,  1,194,151, 19, 87,
 88,163,193, 19, 81,143,212,108,203, 13,  5,220,162, 71, 40,192,232, 68,123, 63, 19,118,152, 12,165,242, 20,147,227, 97, 88,
 89,185, 19, 11,219, 27,116, 86,167,202, 90,228,171,138,234,  1,235,188, 21,169, 11,223, 88,216,213, 98,117,254, 77,156, 89,
 90,151,120, 35, 88,186,137, 24, 19,158,187,172,176,192,177, 95,238,169,167, 14, 35, 57,  5,185, 69,207,214, 73,162,134, 90,
 91,141,170, 59,210, 46, 41, 34,127, 89,228,148,185, 13,115,158,237, 81,201,152, 59,144,197,109, 53, 95,183, 36, 12,123, 91,
 92,203,174,115, 67,229,110,132,102, 54,100, 60,134, 84,  7,227,228,131,222, 93,115,232,191,123,120,136,141, 58, 97,178, 92,
 93,209,124,107,201,113,206,190, 10,241, 59,  4,143,153,197, 34,231,123,176,203,107, 65,127,175,  8, 24,236, 87,207, 79, 93,
 94,255, 23, 67, 74,208, 51,240,190,165,218, 76,148,211,158,124,226,110,  2,108, 67,167, 34,206,152,181, 79,224, 32, 85, 94,
 95,229,197, 91,192, 68,147,202,210, 98,133,116,157, 30, 92,189,225,150,108,250, 91, 14,226, 26,232, 37, 46,141,142,168, 95,
 96,169,225,105,173,153,214, 53,  3, 82,113,245, 71,181,175, 15,160,154,195, 89,105,145,111,188,247,  4,217, 99,172,103, 96,
 97,179, 51,113, 39, 13,118, 15,111,149, 46,205, 78,120,109,206,163, 98,173,207,113, 56,175,104,135,148,184, 14,  2,154, 97,
 98,157, 88, 89,164,172,139, 65,219,193,207,133, 85, 50, 54,144,166,119, 31,104, 89,222,242,  9, 23, 57, 27,185,237,128, 98,
 99,135,138, 65, 46, 56, 43,123,183,  6,144,189, 92,255,244, 81,165,143,113,254, 65,119, 50,221,103,169,122,212, 67,125, 99,
100,193,142,  9,191,243,108,221,174,105, 16, 21, 99,166,128, 44,172, 93,102, 59,  9, 15, 72,203, 42,126, 64,202, 46,180,100,
101,219, 92, 17, 53,103,204,231,194,174, 79, 45,106,107, 66,237,175,165,  8,173, 17,166,136, 31, 90,238, 33,167,128, 73,101,
102,245, 55, 57,182,198, 49,169,118,250,174,101,113, 33, 25,179,170,176,186, 10, 57, 64,213,126,202, 67,130, 16,111, 83,102,
103,239,229, 33, 60, 82,145,147, 26, 61,241, 93,120,236,219,114,169, 72,212,156, 33,233, 21,170,186,211,227,125,193,174,103,
104,121, 63,169,137, 77,191,248, 68, 36,179, 40, 15,147,241, 73,184,  9,148,157,169,176, 33, 82, 80,240,246, 44,181,220,104,
105, 99,237,177,  3,217, 31,194, 40,227,236, 16,  6, 94, 51,136,187,241,250, 11,177, 25,225,134, 32, 96,151, 65, 27, 33,105,
106, 77,134,153,128,120,226,140,156,183, 13, 88, 29, 20,104,214,190,228, 72,172,153,255,188,231,176,205, 52,246,244, 59,106,
107, 87, 84,129, 10,236, 66,182,240,112, 82, 96, 20,217,170, 23,189, 28, 38, 58,129, 86,124, 51,192, 93, 85,155, 90,198,107,
108, 17, 80,201,155, 39,  5, 16,233, 31,210,200, 43,128,222,106,180,206, 49,255,201, 46,  6, 37,141,138,111,133, 55, 15,108,
109, 11,130,209, 17,179,165, 42,133,216,141,240, 34, 77, 28,171,183, 54, 95,105,209,135,198,241,253, 26, 14,232,153,242,109,
110, 37,233,249,146, 18, 88,100, 49,140,108,184, 57,  7, 71,245,178, 35,237,206,249, 97,155,144,109,183,173, 95,118,232,110,
111, 63, 59,225, 24,134,248, 94, 93, 75, 51,128, 48,202,133, 52,177,219,131, 88,225,200, 91, 68, 29, 39,204, 50,216, 21,111,
112, 20, 64,244,229, 44,  4,178,141,190,232, 82,215,249, 19,131,144,161,109,204,244,211,243,125,164,241,135,253,158, 12,112,
113, 14,146,236,111,184,164,136,225,121,183,106,222, 52,209, 66,147, 89,  3, 90,236,122, 51,169,212, 97,230,144, 48,241,113,
114, 32,249,196,236, 25, 89,198, 85, 45, 86, 34,197,126,138, 28,150, 76,177,253,196,156,110,200, 68,204, 69, 39,223,235,114,
115, 58, 43,220,102,141,249,252, 57,234,  9, 26,204,179, 72,221,149,180,223,107,220, 53,174, 28, 52, 92, 36, 74,113, 22,115,
116,124, 47,148,247, 70,190, 90, 32,133,137,178,243,234, 60,160,156,102,200,174,148, 77,212, 10,121,139, 30, 84, 28,223,116,
117,102,253,140,125,210, 30, 96, 76, 66,214,138,250, 39,254, 97,159,158,166, 56,140,228, 20,222,  9, 27,127, 57,178, 34,117,
118, 72,150,164,254,115,227, 46,248, 22, 55,194,225,109,165, 63,154,139, 20,159,164,  2, 73,191,153,182,220,142, 93, 56,118,
119, 82, 68,188,116,231, 67, 20,148,209,104,250,232,160,103,254,153,115,122,  9,188,171,137,107,233, 38,189,227,243,197,119,
120,196,158, 52,193,248,109,127,202,200, 42,143,159,223, 77,197,136, 50, 58,  8, 52,242,189,147,  3,  5,168,178,135,183,120,
121,222, 76, 44, 75,108,205, 69,166, 15,117,183,150, 18,143,  4,139,202, 84,158, 44, 91,125, 71,115,149,201,223, 41, 74,121,
122,240, 39,  4,200,205, 48, 11, 18, 91,148,255,141, 88,212, 90,142,223,230, 57,  4,189, 32, 38,227, 56,106,104,198, 80,122,
123,234,245, 28, 66, 89,144, 49,126,156,203,199,132,149, 22,155,141, 39,136,175, 28, 20,224,242,147,168, 11,  5,104,173,123,
124,172,241, 84,211,146,215,151,103,243, 75,111,187,204, 98,230,132,245,159,106, 84,108,154,228,222,127, 49, 27,  5,100,124,
125,182, 35, 76, 89,  6,119,173, 11, 52, 20, 87,178,  1,160, 39,135, 13,241,252, 76,197, 90, 48,174,239, 80,118,171,153,125,
126,152, 72,100,218,167,138,227,191, 96,245, 31,169, 75,251,121,130, 24, 67, 91,100, 35,  7, 81, 62, 66,243,193, 68,131,126,
127,130,154,124, 80, 51, 42,217,211,167,170, 39,160,134, 57,184,129,224, 45,205,124,138,199,133, 78,210,146,172,234,126,127,
128,129, 97,156,122,193,222, 76,  4, 51,188, 81,244, 90,137, 20,157,197, 25,220,156, 42,148, 70,162,251,202,132,141,127,128,
129,155,179,132,240, 85,126,118,104,244,227,105,253,151, 75,213,158, 61,119, 74,132,131, 84,146,210,107,171,233, 35,130,129,
130,181,216,172,115,244,131, 56,220,160,  2, 33,230,221, 16,139,155, 40,197,237,172,101,  9,243, 66,198,  8, 94,204,152,130,
131,175, 10,180,249, 96, 35,  2,176,103, 93, 25,239, 16,210, 74,152,208,171,123,180,204,201, 39, 50, 86,105, 51, 98,101,131,
132,233, 14,252,104,171,100,164,169,  8,221,177,208, 73,166, 55,145,  2,188,190,252,180,179, 49,127,129, 83, 45, 15,172,132,
133,243,220,228,226, 63,196,158,197,207,130,137,217,132,100,246,146,250,210, 40,228, 29,115,229, 15, 17, 50, 64,161, 81,133,
134,221,183,204, 97,158, 57,208,113,155, 99,193,194,206, 63,168,151,239, 96,143,204,251, 46,132,159,188,145,247, 78, 75,134,
135,199,101,212,235, 10,153,234, 29, 92, 60,249,203,  3,253,105,148, 23, 14, 25,212, 82,238, 80,239, 44,240,154,224,182,135,
136, 81,191, 92, 94, 21,183,129, 67, 69,126,140,188,124,215, 82,133, 86, 78, 24, 92, 11,218,168,  5, 15,229,203,148,196,136,
137, 75,109, 68,212,129, 23,187, 47,130, 33,180,181,177, 21,147,134,174, 32,142, 68,162, 26,124,117,159,132,166, 58, 57,137,
138,101,  6,108, 87, 32,234,245,155,214,192,252,174,251, 78,205,131,187,146, 41,108, 68, 71, 29,229, 50, 39, 17,213, 35,138,
139,127,212,116,221,180, 74,207,247, 17,159,196,167, 54,140, 12,128, 67,252,191,116,237,135,201,149,162, 70,124,123,222,139,
140, 57,208, 60, 76,127, 13,105,238,126, 31,108,152,111,248,113,137,145,235,122, 60,149,253,223,216,117,124, 98, 22, 23,140,
141, 35,  2, 36,198,235,173, 83,130,185, 64, 84,145,162, 58,176,138,105,133,236, 36, 60, 61, 11,168,229, 29, 15,184,234,141,
142, 13,105, 12, 69, 74, 80, 29, 54,237,161, 28,138,232, 97,238,143,124, 55, 75, 12,218, 96,106, 56, 72,190,184, 87,240,142,
143, 23,187, 20,207,222,240, 39, 90, 42,254, 36,131, 37,163, 47,140,132, 89,221, 20,115,160,190, 72,216,223,213,249, 13,143,
144, 60,192,  1, 50,116, 12,203,138,223, 37,246,100, 22, 53,152,173,254,183, 73,  1,104,  8,135,241, 14,148, 26,191, 20,144,
145, 38, 18, 25,184,224,172,241,230, 24,122,206,109,219,247, 89,174,  6,217,223, 25,193,200, 83,129,158,245,119, 17,233,145,
146,  8,121, 49, 59, 65, 81,191, 82, 76,155,134,118,145,172,  7,171, 19,107,120, 49, 39,149, 50, 17, 51, 86,192,254,243,146,
147, 18,171, 41,177,213,241,133, 62,139,196,190,127, 92,110,198,168,235,  5,238, 41,142, 85,230, 97,163, 55,173, 80, 14,147,
148, 84,175, 97, 32, 30,182, 35, 39,228, 68, 22, 64,  5, 26,187,161, 57, 18, 43, 97,246, 47,240, 44,116, 13,179, 61,199,148,
149, 78,125,121,170,138, 22, 25, 75, 35, 27, 46, 73,200,216,122,162,193,124,189,121, 95,239, 36, 92,228,108,222,147, 58,149,
150, 96, 22, 81, 41, 43,235, 87,255,119,250,102, 82,130,131, 36,167,212,206, 26, 81,185,178, 69,204, 73,207,105,124, 32,150,
151,122,196, 73,163,191, 75,109,147,176,165, 94, 91, 79, 65,229,164, 44,160,140, 73, 16,114,145,188,217,174,  4,210,221,151,
152,236, 30,193, 22,160,101,  6,205,169,231, 43, 44, 48,107,222,181,109,224,141,193, 73, 70,105, 86,250,187, 85,166,175,152,
153,246,204,217,156, 52,197, 60,161,110,184, 19, 37,253,169, 31,182,149,142, 27,217,224,134,189, 38,106,218, 56,  8, 82,153,
154,216,167,241, 31,149, 56,114, 21, 58, 89, 91, 62,183,242, 65,179,128, 60,188,241,  6,219,220,182,199,121,143,231, 72,154,
155,194,117,233,149,  1,152, 72,121,253,  6, 99, 55,122, 48,128,176,120, 82, 42,233,175, 27,  8,198, 87, 24,226, 73,181,155,
156,132,113,161,  4,202,223,238, 96,146,134,203,  8, 35, 68,253,185,170, 69,239,161,215, 97, 30,139,128, 34,252, 36,124,156,
157,158,163,185,142, 94,127,212, 12, 85,217,243,  1,238,134, 60,186, 82, 43,121,185,126,161,202,251, 16, 67,145,138,129,157,
158,176,200,145, 13,255,130,154,184,  1, 56,187, 26,164,221, 98,191, 71,153,222,145,152,252,171,107,189,224, 38,101,155,158,
159,170, 26,137,135,107, 34,160,212,198,103,131, 19,105, 31,163,188,191,247, 72,137, 49, 60,127, 27, 45,129, 75,203,102,159,
160,230, 62,187,234,182,103, 95,  5,246,147,  2,201,194,236, 17,253,179, 88,235,187,174,177,217,  4, 12,118,165,233,169,160,
161,252,236,163, 96, 34,199,101,105, 49,204, 58,192, 15, 46,208,254, 75, 54,125,163,  7,113, 13,116,156, 23,200, 71, 84,161,
162,210,135,139,227,131, 58, 43,221,101, 45,114,219, 69,117,142,251, 94,132,218,139,225, 44,108,228, 49,180,127,168, 78,162,
163,200, 85,147,105, 23,154, 17,177,162,114, 74,210,136,183, 79,248,166,234, 76,147, 72,236,184,148,161,213, 18,  6,179,163,
164,142, 81,219,248,220,221,183,168,205,242,226,237,209,195, 50,241,116,253,137,219, 48,150,174,217,118,239, 12,107,122,164,
165,148,131,195,114, 72,125,141,196, 10,173,218,228, 28,  1,243,242,140,147, 31,195,153, 86,122,169,230,142, 97,197,135,165,
166,186,232,235,241,233,128,195,112, 94, 76,146,255, 86, 90,173,247,153, 33,184,235,127, 11, 27, 57, 75, 45,214, 42,157,166,
167,160, 58,243,123,125, 32,249, 28,153, 19,170,246,155,152,108,244, 97, 79, 46,243,214,203,207, 73,219, 76,187,132, 96,167,
168, 54,224,123,206, 98, 14,146, 66,128, 81,223,129,228,178, 87,229, 32, 15, 47,123,143,255, 55,163,248, 89,234,240, 18,168,
169, 44, 50, 99, 68,246,174,168, 46, 71, 14,231,136, 41,112,150,230,216, 97,185, 99, 38, 63,227,211,104, 56,135, 94,239,169,
170,  2, 89, 75,199, 87, 83,230,154, 19,239,175,147, 99, 43,200,227,205,211, 30, 75,192, 98,130, 67,197,155, 48,177,245,170,
171, 24,139, 83, 77,195,243,220,246,212,176,151,154,174,233,  9,224, 53,189,136, 83,105,162, 86, 51, 85,250, 93, 31,  8,171,
172, 94,143, 27,220,  8,180,122,239,187, 48, 63,165,247,157,116,233,231,170, 77, 27, 17,216, 64,126,130,192, 67,114,193,172,
173, 68, 93,  3, 86,156, 20, 64,131,124,111,  7,172, 58, 95,181,234, 31,196,219,  3,184, 24,148, 14, 18,161, 46,220, 60,173,
174,106, 54, 43,213, 61,233, 14, 55, 40,142, 79,183,112,  4,235,239, 10,118,124, 43, 94, 69,245,158,191,  2,153, 51, 38,174,
175,112,228, 51, 95,169, 73, 52, 91,239,209,119,190,189,198, 42,236,242, 24,234, 51,247,133, 33,238, 47, 99,244,157,219,175,
176, 91,159, 38,162,  3,181,216,139, 26, 10,165, 89,142, 80,157,205,136,246,126, 38,236, 45, 24, 87,249, 40, 59,219,194,176,
177, 65, 77, 62, 40,151, 21,226,231,221, 85,157, 80, 67,146, 92,206,112,152,232, 62, 69,237,204, 39,105, 73, 86,117, 63,177,
178,111, 38, 22,171, 54,232,172, 83,137,180,213, 75,  9,201,  2,203,101, 42, 79, 22,163,176,173,183,196,234,225,154, 37,178,
179,117,244, 14, 33,162, 72,150, 63, 78,235,237, 66,196, 11,195,200,157, 68,217, 14, 10,112,121,199, 84,139,140, 52,216,179,
180, 51,240, 70,176,105, 15, 48, 38, 33,107, 69,125,157,127,190,193, 79, 83, 28, 70,114, 10,111,138,131,177,146, 89, 17,180,
181, 41, 34, 94, 58,253,175, 10, 74,230, 52,125,116, 80,189,127,194,183, 61,138, 94,219,202,187,250, 19,208,255,247,236,181,
182,  7, 73,118,185, 92, 82, 68,254,178,213, 53,111, 26,230, 33,199,162,143, 45,118, 61,151,218,106,190,115, 72, 24,246,182,
183, 29,155,110, 51,200,242,126,146,117,138, 13,102,215, 36,224,196, 90,225,187,110,148, 87, 14, 26, 46, 18, 37,182, 11,183,
184,139, 65,230,134,215,220, 21,204,108,200,120, 17,168, 14,219,213, 27,161,186,230,205, 99,246,240, 13,  7,116,194,121,184,
185,145,147,254, 12, 67,124, 47,160,171,151, 64, 24,101,204, 26,214,227,207, 44,254,100,163, 34,128,157,102, 25,108,132,185,
186,191,248,214,143,226,129, 97, 20,255,118,  8,  3, 47,151, 68,211,246,125,139,214,130,254, 67, 16, 48,197,174,131,158,186,
187,165, 42,206,  5,118, 33, 91,120, 56, 41, 48, 10,226, 85,133,208, 14, 19, 29,206, 43, 62,151, 96,160,164,195, 45, 99,187,
188,227, 46,134,148,189,102,253, 97, 87,169,152, 53,187, 33,248,217,220,  4,216,134, 83, 68,129, 45,119,158,221, 64,170,188,
189,249,252,158, 30, 41,198,199, 13,144,246,160, 60,118,227, 57,218, 36,106, 78,158,250,132, 85, 93,231,255,176,238, 87,189,
190,215,151,182,157,136, 59,137,185,196, 23,232, 39, 60,184,103,223, 49,216,233,182, 28,217, 52,205, 74, 92,  7,  1, 77,190,
191,205, 69,174, 23, 28,155,179,213,  3, 72,208, 46,241,122,166,220,201,182,127,174,181, 25,224,189,218, 61,106,175,176,191,
192, 79,223,210, 71, 47,177,106,  6,164,226,247,142,119, 67, 30, 93, 41,155,178,210, 63,222,101,243,  8,175,198, 69,206,192,
193, 85, 13,202,205,187, 17, 80,106, 99,189,207,135,186,129,223, 94,209,245, 36,202,150, 30,177,131,152,206,171,235, 51,193,
194,123,102,226, 78, 26,236, 30,222, 55, 92,135,156,240,218,129, 91,196, 71,131,226,112, 67,208, 19, 53,109, 28,  4, 41,194,
195, 97,180,250,196,142, 76, 36,178,240,  3,191,149, 61, 24, 64, 88, 60, 41, 21,250,217,131,  4, 99,165, 12,113,170,212,195,
196, 39,176,178, 85, 69, 11,130,171,159,131, 23,170,100,108, 61, 81,238, 62,208,178,161,249, 18, 46,114, 54,111,199, 29,196,
197, 61, 98,170,223,209,171,184,199, 88,220, 47,163,169,174,252, 82, 22, 80, 70,170,  8, 57,198, 94,226, 87,  2,105,224,197,
198, 19,  9,130, 92,112, 86,246,115, 12, 61,103,184,227,245,162, 87,  3,226,225,130,238,100,167,206, 79,244,181,134,250,198,
199,  9,219,154,214,228,246,204, 31,203, 98, 95,177, 46, 55, 99, 84,251,140,119,154, 71,164,115,190,223,149,216, 40,  7,199,
200,159,  1, 18, 99,251,216,167, 65,210, 32, 42,198, 81, 29, 88, 69,186,204,118, 18, 30,144,139, 84,252,128,137, 92,117,200,
201,133,211, 10,233,111,120,157, 45, 21,127, 18,207,156,223,153, 70, 66,162,224, 10,183, 80, 95, 36,108,225,228,242,136,201,
202,171,184, 34,106,206,133,211,153, 65,158, 90,212,214,132,199, 67, 87, 16, 71, 34, 81, 13, 62,180,193, 66, 83, 29,146,202,
203,177,106, 58,224, 90, 37,233,245,134,193, 98,221, 27, 70,  6, 64,175,126,209, 58,248,205,234,196, 81, 35, 62,179,111,203,
204,247,110,114,113,145, 98, 79,236,233, 65,202,226, 66, 50,123, 73,125,105, 20,114,128,183,252,137,134, 25, 32,222,166,204,
205,237,188,106,251,  5,194,117,128, 46, 30,242,235,143,240,186, 74,133,  7,130,106, 41,119, 40,249, 22,120, 77,112, 91,205,
206,195,215, 66,120,164, 63, 59, 52,122,255,186,240,197,171,228, 79,144,181, 37, 66,207, 42, 73,105,187,219,250,159, 65,206,
207,217,  5, 90,242, 48,159,  1, 88,189,160,130,249,  8,105, 37, 76,104,219,179, 90,102,234,157, 25, 43,186,151, 49,188,207,
208,242,126, 79, 15,154, 99,237,136, 72,123, 80, 30, 59,255,146,109, 18, 53, 39, 79,125, 66,164,160,253,241, 88,119,165,208,
209,232,172, 87,133, 14,195,215,228,143, 36,104, 23,246, 61, 83,110,234, 91,177, 87,212,130,112,208,109,144, 53,217, 88,209,
210,198,199,127,  6,175, 62,153, 80,219,197, 32, 12,188,102, 13,107,255,233, 22,127, 50,223, 17, 64,192, 51,130, 54, 66,210,
211,220, 21,103,140, 59,158,163, 60, 28,154, 24,  5,113,164,204,104,  7,135,128,103,155, 31,197, 48, 80, 82,239,152,191,211,
212,154, 17, 47, 29,240,217,  5, 37,115, 26,176, 58, 40,208,177, 97,213,144, 69, 47,227,101,211,125,135,104,241,245,118,212,
213,128,195, 55,151,100,121, 63, 73,180, 69,136, 51,229, 18,112, 98, 45,254,211, 55, 74,165,  7, 13, 23,  9,156, 91,139,213,
214,174,168, 31, 20,197,132,113,253,224,164,192, 40,175, 73, 46,103, 56, 76,116, 31,172,248,102,157,186,170, 43,180,145,214,
215,180,122,  7,158, 81, 36, 75,145, 39,251,248, 33, 98,139,239,100,192, 34,226,  7,  5, 56,178,237, 42,203, 70, 26,108,215,
216, 34,160,143, 43, 78, 10, 32,207, 62,185,141, 86, 29,161,212,117,129, 98,227,143, 92, 12, 74,  7,  9,222, 23,110, 30,216,
217, 56,114,151,161,218,170, 26,163,249,230,181, 95,208, 99, 21,118,121, 12,117,151,245,204,158,119,153,191,122,192,227,217,
218, 22, 25,191, 34,123, 87, 84, 23,173,  7,253, 68,154, 56, 75,115,108,190,210,191, 19,145,255,231, 52, 28,205, 47,249,218,
219, 12,203,167,168,239,247,110,123,106, 88,197, 77, 87,250,138,112,148,208, 68,167,186, 81, 43,151,164,125,160,129,  4,219,
220, 74,207,239, 57, 36,176,200, 98,  5,216,109,114, 14,142,247,121, 70,199,129,239,194, 43, 61,218,115, 71,190,236,205,220,
221, 80, 29,247,179,176, 16,242, 14,194,135, 85,123,195, 76, 54,122,190,169, 23,247,107,235,233,170,227, 38,211, 66, 48,221,
222,126,118,223, 48, 17,237,188,186,150,102, 29, 96,137, 23,104,127,171, 27,176,223,141,182,136, 58, 78,133,100,173, 42,222,
223,100,164,199,186,133, 77,134,214, 81, 57, 37,105, 68,213,169,124, 83,117, 38,199, 36,118, 92, 74,222,228,  9,  3,215,223,
224, 40,128,245,215, 88,  8,121,  7, 97,205,164,179,239, 38, 27, 61, 95,218,133,245,187,251,250, 85,255, 19,231, 33, 24,224,
225, 50, 82,237, 93,204,168, 67,107,166,146,156,186, 34,228,218, 62,167,180, 19,237, 18, 59, 46, 37,111,114,138,143,229,225,
226, 28, 57,197,222,109, 85, 13,223,242,115,212,161,104,191,132, 59,178,  6,180,197,244,102, 79,181,194,209, 61, 96,255,226,
227,  6,235,221, 84,249,245, 55,179, 53, 44,236,168,165,125, 69, 56, 74,104, 34,221, 93,166,155,197, 82,176, 80,206,  2,227,
228, 64,239,149,197, 50,178,145,170, 90,172, 68,151,252,  9, 56, 49,152,127,231,149, 37,220,141,136,133,138, 78,163,203,228,
229, 90, 61,141, 79,166, 18,171,198,157,243,124,158, 49,203,249, 50, 96, 17,113,141,140, 28, 89,248, 21,235, 35, 13, 54,229,
230,116, 86,165,204,  7,239,229,114,201, 18, 52,133,123,144,167, 55,117,163,214,165,106, 65, 56,104,184, 72,148,226, 44,230,
231,110,132,189, 70,147, 79,223, 30, 14, 77, 12,140,182, 82,102, 52,141,205, 64,189,195,129,236, 24, 40, 41,249, 76,209,231,
232,248, 94, 53,243,140, 97,180, 64, 23, 15,121,251,201,120, 93, 37,204,141, 65, 53,154,181, 20,242, 11, 60,168, 56,163,232,
233,226,140, 45,121, 24,193,142, 44,208, 80, 65,242,  4,186,156, 38, 52,227,215, 45, 51,117,192,130,155, 93,197,150, 94,233,
234,204,231,  5,250,185, 60,192,152,132,177,  9,233, 78,225,194, 35, 33, 81,112,  5,213, 40,161, 18, 54,254,114,121, 68,234,
235,214, 53, 29,112, 45,156,250,244, 67,238, 49,224,131, 35,  3, 32,217, 63,230, 29,124,232,117, 98,166,159, 31,215,185,235,
236,144, 49, 85,225,230,219, 92,237, 44,110,153,223,218, 87,126, 41, 11, 40, 35, 85,  4,146, 99, 47,113,165,  1,186,112,236,
237,138,227, 77,107,114,123,102,129,235, 49,161,214, 23,149,191, 42,243, 70,181, 77,173, 82,183, 95,225,196,108, 20,141,237,
238,164,136,101,232,211,134, 40, 53,191,208,233,205, 93,206,225, 47,230,244, 18,101, 75, 15,214,207, 76,103,219,251,151,238,
239,190, 90,125, 98, 71, 38, 18, 89,120,143,209,196,144, 12, 32, 44, 30,154,132,125,226,207,  2,191,220,  6,182, 85,106,239,
240,149, 33,104,159,237,218,254,137,141, 84,  3, 35,163,154,151, 13,100,116, 16,104,249,103, 59,  6, 10, 77,121, 19,115,240,
241,143,243,112, 21,121,122,196,229, 74, 11, 59, 42,110, 88, 86, 14,156, 26,134,112, 80,167,239,118,154, 44, 20,189,142,241,
242,161,152, 88,150,216,135,138, 81, 30,234,115, 49, 36,  3,  8, 11,137,168, 33, 88,182,250,142,230, 55,143,163, 82,148,242,
243,187, 74, 64, 28, 76, 39,176, 61,217,181, 75, 56,233,193,201,  8,113,198,183, 64, 31, 58, 90,150,167,238,206,252,105,243,
244,253, 78,  8,141,135, 96, 22, 36,182, 53,227,  7,176,181,180,  1,163,209,114,  8,103, 64, 76,219,112,212,208,145,160,244,
245,231,156, 16,  7, 19,192, 44, 72,113,106,219, 14,125,119,117,  2, 91,191,228, 16,206,128,152,171,224,181,189, 63, 93,245,
246,201,247, 56,132,178, 61, 98,252, 37,139,147, 21, 55, 44, 43,  7, 78, 13, 67, 56, 40,221,249, 59, 77, 22, 10,208, 71,246,
247,211, 37, 32, 14, 38,157, 88,144,226,212,171, 28,250,238,234,  4,182, 99,213, 32,129, 29, 45, 75,221,119,103,126,186,247,
248, 69,255,168,187, 57,179, 51,206,251,150,222,107,133,196,209, 21,247, 35,212,168,216, 41,213,161,254, 98, 54, 10,200,248,
249, 95, 45,176, 49,173, 19,  9,162, 60,201,230, 98, 72,  6, 16, 22, 15, 77, 66,176,113,233,  1,209,110,  3, 91,164, 53,249,
250,113, 70,152,178, 12,238, 71, 22,104, 40,174,121,  2, 93, 78, 19, 26,255,229,152,151,180, 96, 65,195,160,236, 75, 47,250,
251,107,148,128, 56,152, 78,125,122,175,119,150,112,207,159,143, 16,226,145,115,128, 62,116,180, 49, 83,193,129,229,210,251,
252, 45,144,200,169, 83,  9,219, 99,192,247, 62, 79,150,235,242, 25, 48,134,182,200, 70, 14,162,124,132,251,159,136, 27,252,
253, 55, 66,208, 35,199,169,225, 15,  7,168,  6, 70, 91, 41, 51, 26,200,232, 32,208,239,206,118, 12, 20,154,242, 38,230,253,
254, 25, 41,248,160,102, 84,175,187, 83, 73, 78, 93, 17,114,109, 31,221, 90,135,248,  9,147, 23,156,185, 57, 69,201,252,254,
255,  3,251,224, 42,242,244,149,215,148, 22,118, 84,220,176,172, 28, 37, 52, 17,224,160, 83,195,236, 41, 88, 40,103,  1,255,
};

static void CRC_BCH256_771_divide(const char *in, char *out)
{
	char R[771];
	int i, ii, jj;
	int x;

	for (i = 0; i < 771; i++) {
		R[i] = in[i];
	}

	for (i = 770; i >= 30; i--) {
		x = R[i];
		if (x == 0) {
			continue;
		}
		x--;
		for (ii = i, jj = 30; jj >= 0; ii--, jj--) {
			R[ii] ^= B[x * 31 + jj];
		}
	}

	for (i = 30; i >= 0; i--) {
		out[i] = R[i];
	}
}



void coding_theory_domain::introduce_errors(
		crc_options_description *Crc_options_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::introduce_errors " << endl;
		Crc_options_description->print();
	}

	data_structures::string_tools ST;
	string fname_error;
	//int information_length = block_length - 4;

	if (!Crc_options_description->f_input) {
		cout << "coding_theory_domain::introduce_errors please use -input <fname>" << endl;
		exit(1);
	}
	if (!Crc_options_description->f_output) {
		cout << "coding_theory_domain::introduce_errors please use -output <fname>" << endl;
		exit(1);
	}
	if (!Crc_options_description->f_block_length) {
		cout << "coding_theory_domain::introduce_errors please use -block_length <block_length>" << endl;
		exit(1);
	}
	if (!Crc_options_description->f_threshold) {
		cout << "coding_theory_domain::introduce_errors please use -threshold <threshold>" << endl;
		exit(1);
	}
#if 1
	fname_error.assign(Crc_options_description->output_fname);
	ST.chop_off_extension(fname_error);
	fname_error.append("_pattern.csv");
#endif

	int block_length;

	block_length = Crc_options_description->block_length;

	orbiter_kernel_system::file_io Fio;

	long int N, L, cnt;
	long int nb_blocks;
	char *buffer;

	N = Fio.file_size(Crc_options_description->input_fname);

	if (f_v) {
		cout << "coding_theory_domain::introduce_errors input file size = " << N << endl;
	}
	buffer = NEW_char(N);


	nb_blocks = (N + block_length - 1) / block_length;
	if (f_v) {
		cout << "coding_theory_domain::introduce_errors nb_blocks = " << nb_blocks << endl;
	}

	int a, b, c;
	//long int cnt;

	ifstream ist(Crc_options_description->input_fname, ios::binary);

	error_repository *Error_pattern; // [nb_blocks]
	//int *Nb_errors; // [nb_blocks]
	int nb_errors = 0;

	Error_pattern = NEW_OBJECTS(error_repository, nb_blocks);
	//Nb_errors = NEW_int(nb_blocks);
	for (cnt = 0; cnt < nb_blocks; cnt++) {
		Error_pattern[cnt].init(0);
	}

	//Int_vec_zero(Nb_errors, nb_blocks);


	//Error_pattern = NEW_lint(2 * nb_blocks * 3);
	//Lint_vec_zero(Error_pattern, 2 * nb_blocks * 3);

	ist.read(buffer, N);


	//cnt = 0;

	for (cnt = 0; cnt < nb_blocks; cnt++) {

		if ((cnt + 1) * block_length > N) {
			L = N - cnt * block_length;
			cout << "truncating, L = " << L << endl;
		}
		else {
			L = block_length;
		}


		orbiter_kernel_system::os_interface Os;

		a = Os.random_integer(1000000);
		if (a < Crc_options_description->threshold) {
			b = Os.random_integer(L);
			c = Os.random_integer(254) + 1;
			buffer[cnt * block_length + b] ^= c;

			Error_pattern[cnt].add_error(b, c, 0 /*verbose_level*/);
		}

	}


	if (Crc_options_description->f_file_based_error_generator) {
		cout << "f_file_based_error_generator" << endl;

		int threshold = Crc_options_description->file_based_error_generator_threshold;
		int nb_repeats;

		if (!Crc_options_description->f_nb_repeats) {
			nb_repeats = 1;
		}
		else {
			nb_repeats = Crc_options_description->nb_repeats;
		}

		int h;

		for (h = 0; h < nb_repeats; h++) {

			long int cnt1;

			for (cnt = 0; cnt < nb_blocks; cnt++) {

				orbiter_kernel_system::os_interface Os;

				a = Os.random_integer(1000000);
				if (a < threshold) {
					b = Os.random_integer(N);
					c = Os.random_integer(254) + 1;
					buffer[b] ^= c;
					cnt1 = b / block_length;
					Error_pattern[cnt1].add_error(b % block_length, c, 0 /*verbose_level*/);

				}
			}

		}
	}

	long int *Ep;
	int i, j;
	int nb_erroneous_blocks = 0;


	nb_errors = 0;
	for (cnt = 0; cnt < nb_blocks; cnt++) {
		nb_errors += Error_pattern[cnt].nb_errors;
	}

	Ep = NEW_lint(nb_errors * 3);
	j = 0;
	for (cnt = 0; cnt < nb_blocks; cnt++) {
		if (Error_pattern[cnt].nb_errors) {
			for (i = 0; i < Error_pattern[cnt].nb_errors; i++) {
				a = cnt;
				b = Error_pattern[cnt].Error_storage[i * 2 + 0];
				c = Error_pattern[cnt].Error_storage[i * 2 + 1];
				Ep[j * 3 + 0] = a;
				Ep[j * 3 + 1] = b;
				Ep[j * 3 + 2] = c;
				j++;
			}
			nb_erroneous_blocks++;
		}
		else {
		}
	}
	if (j != nb_errors) {
		cout << "j != nb_errors" << endl;
		exit(1);
	}

	cout << "nb_erroneous_blocks = " << nb_erroneous_blocks << " / " << nb_blocks << endl;
	cout << "nb_errors = " << nb_errors << endl;

	// write the modified output file:
	{
		ofstream ost(Crc_options_description->output_fname, ios::binary);
		ost.write(buffer, N);
	}

	cout << "Written file " << Crc_options_description->output_fname
			<< " of size " << Fio.file_size(Crc_options_description->output_fname) << endl;

	Fio.lint_matrix_write_csv(fname_error, Ep, nb_errors, 3);
	cout << "Written file " << fname_error << " of size " << Fio.file_size(fname_error) << endl;

	FREE_OBJECTS(Error_pattern);
	//FREE_lint(Error_pattern);
	FREE_lint(Ep);

	if (f_v) {
		cout << "coding_theory_domain::introduce_errors done" << endl;
	}

}


void coding_theory_domain::crc32_file_based(std::string &fname_in, std::string &fname_out,
		int block_length, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::crc32_file_based "
				"fname_in=" << fname_in << endl;
		cout << "coding_theory_domain::crc32_file_based "
				"block_length=" << block_length << endl;
	}

	data_structures::string_tools ST;
	//string fname_out;
	int information_length = block_length - 4;

#if 0
	fname_out.assign(fname_in);
	ST.chop_off_extension(fname_out);
	fname_out.append("_crc32.bin");
#endif

	orbiter_kernel_system::file_io Fio;

	long int N, L, nb_blocks, cnt;
	char *buffer;

	N = Fio.file_size(fname_in);

	if (f_v) {
		cout << "coding_theory_domain::crc32_file_based input file size = " << N << endl;
	}

	nb_blocks = (N + information_length - 1) / information_length;
	if (f_v) {
		cout << "coding_theory_domain::crc32_file_based nb_blocks = " << nb_blocks << endl;
	}

	buffer = NEW_char(block_length);


	ifstream ist(fname_in, ios::binary);

	{
		ofstream ost(fname_out, ios::binary);
		uint32_t crc;
		char *p_crc;

		p_crc = (char *) &crc;
		//C = 0;

		for (cnt = 0; cnt < nb_blocks; cnt++) {
		//while (C < N) {

			if ((cnt + 1) * information_length > N) {
				L = N - cnt * information_length;
			}
			else {
				L = information_length;
			}

			// read information_length bytes
			// (or less, in case we reached the end of he file)

			ist.read(buffer, L);

			// create 4 byte check and add to the block:

			crc = crc32(buffer, L);
			buffer[L + 0] = p_crc[0];
			buffer[L + 1] = p_crc[1];
			buffer[L + 2] = p_crc[2];
			buffer[L + 3] = p_crc[3];


			// write information_length + 4 bytes to file:
			// (or less in case we have reached the end of the input file):

			ost.write(buffer, L + 4);


			// count the bytes read,
			// so C is the position in the input file:

			//C += L;
		}

	}
	cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;



	if (f_v) {
		cout << "coding_theory_domain::crc32_file_based done" << endl;
	}

}


void coding_theory_domain::crc771_file_based(std::string &fname_in, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::crc771_file_based fname_in=" << fname_in << endl;
	}

	data_structures::string_tools ST;
	string fname_out;
	int block_length = 771;
	int redundancy = 30;
	int information_length = block_length - redundancy;

	fname_out.assign(fname_in);
	ST.chop_off_extension(fname_out);
	fname_out.append("_crc771.bin");

	orbiter_kernel_system::file_io Fio;

	long int N, C, L;
	char *buffer;

	N = Fio.file_size(fname_in);

	if (f_v) {
		cout << "coding_theory_domain::crc771_file_based input file size = " << N << endl;
	}
	buffer = NEW_char(block_length);


	ifstream ist(fname_in, ios::binary);

	{
		ofstream ost(fname_out, ios::binary);
		uint32_t crc;
		char *p_crc;
		int i;

		p_crc = (char *) &crc;
		C = 0;

		while (C < N) {

			if (C + information_length > N) {
				L = C + information_length - N;
			}
			else {
				L = information_length;
			}
			ist.read(buffer + redundancy, L);
			for (i = 0; i < redundancy; i++) {
				buffer[i] = 0;
			}
			for (i = L; i < block_length; i++) {
				buffer[i] = 0;
			}

			CRC_BCH256_771_divide(buffer, buffer);

			ost.write(buffer, block_length);


			C += information_length;
		}

	}
	cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;



	if (f_v) {
		cout << "coding_theory_domain::crc771_file_based done" << endl;
	}

}

void coding_theory_domain::check_errors(
		crc_options_description *Crc_options_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::check_errors " << endl;
	}

	if (!Crc_options_description->f_input) {
		cout << "coding_theory_domain::check_errors please use -input <fname>" << endl;
		exit(1);
	}
	if (!Crc_options_description->f_output) {
		cout << "coding_theory_domain::check_errors please use -output <fname>" << endl;
		exit(1);
	}
	if (!Crc_options_description->f_block_length) {
		cout << "coding_theory_domain::check_errors please use -block_length <block_length>" << endl;
		exit(1);
	}
	int block_length;
	int information_length;

	block_length = Crc_options_description->block_length;
	information_length = block_length - 4;
	if (f_v) {
		cout << "coding_theory_domain::check_errors block_length = " << block_length << endl;
		cout << "coding_theory_domain::check_errors information_length = " << information_length << endl;

	}

	if (!Crc_options_description->f_error_log) {
		cout << "coding_theory_domain::check_errors please use -error_log <fname>" << endl;
		exit(1);
	}


	std::string fname_coded;
	std::string fname_recovered;
	std::string fname_error_log;
	std::string fname_error_detected;
	std::string fname_error_undetected;


	data_structures::string_tools ST;
	//string fname_error;
	//int information_length = block_length - 4;


	fname_coded.assign(Crc_options_description->input_fname);
	fname_recovered.assign(Crc_options_description->output_fname);
	fname_error_log.assign(Crc_options_description->error_log_fname);

	fname_error_detected.assign(Crc_options_description->input_fname);
	ST.chop_off_extension(fname_error_detected);
	fname_error_detected.append("_err_detected.csv");

	fname_error_undetected.assign(Crc_options_description->input_fname);
	ST.chop_off_extension(fname_error_undetected);
	fname_error_undetected.append("_err_undetected.csv");

	orbiter_kernel_system::file_io Fio;

	long int N, L;
	long int nb_blocks;
	char *buffer;
	char *recovered_data;
	long int recovered_data_size = 0;

	N = Fio.file_size(fname_coded);

	if (f_v) {
		cout << "coding_theory_domain::check_errors input file size = " << N << endl;
	}
	buffer = NEW_char(block_length);
	recovered_data = NEW_char(N);


	nb_blocks = (N + block_length - 1) / block_length;
	if (f_v) {
		cout << "coding_theory_domain::check_errors nb_blocks = " << nb_blocks << endl;
	}

	int a, b, c;
	long int cnt;
	long int nb_error_detected, nb_error_undetected;

	ifstream ist(fname_coded, ios::binary);

	long int *Error_pattern;
	long int *Error_undetected;
	int nb_error = 0;
	int m;

	cout << "Reading file " << fname_error_log << " of size " << Fio.file_size(fname_error_log) << endl;
	Fio.lint_matrix_read_csv(fname_error_log, Error_pattern, nb_error, m, verbose_level);
	if (m != 3) {
		cout << "m != 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "coding_theory_domain::check_errors nb_error = " << nb_error << endl;
	}

	long int *Faulty_blocks;
	int cur_error;

	Faulty_blocks = NEW_lint(nb_blocks * 3);
	Error_undetected = NEW_lint(nb_blocks * 3);

	nb_error_detected = 0;
	nb_error_undetected = 0;
	cur_error = 0;
	cnt = 0;
	{
		uint32_t crc, crc_computed;
		char *p_crc;

		p_crc = (char *) &crc;

		for (cnt = 0; cnt < nb_blocks; cnt++) {

			if ((cnt + 1) * block_length > N) {
				L = N - cnt * block_length;
			}
			else {
				L = block_length;
			}

			// read information length + 4 bytes
			// (this includes the 4 byte check sum at the very end)

			ist.read(buffer, L);

			p_crc[0] = buffer[L - 4 + 0];
			p_crc[1] = buffer[L - 4 + 1];
			p_crc[2] = buffer[L - 4 + 2];
			p_crc[3] = buffer[L - 4 + 3];

			crc_computed = crc32(buffer, L - 4);


			if (crc_computed != crc) {
				Faulty_blocks[nb_error_detected * 3 + 0] = cnt;
				Faulty_blocks[nb_error_detected * 3 + 1] = crc;
				Faulty_blocks[nb_error_detected * 3 + 2] = crc_computed;

				//cout << "detected error " << nb_error_detected << " in block " << cnt << endl;
				//", crc=" << crc << " crc_computed=" << crc_computed << endl;
				nb_error_detected++;

				while (cur_error < nb_error && cnt == Error_pattern[cur_error * 3 + 0]) {
					//cout << "recovering error " << cur_error << " in block " << cnt << endl;
					a = cnt;
					b = Error_pattern[cur_error * 3 + 1];
					c = Error_pattern[cur_error * 3 + 2];
					buffer[b] ^= c;
					cur_error++;
				}
			}
			else {
				while (cur_error < nb_error && cnt == Error_pattern[cur_error * 3 + 0]) {
					cout << "undetected error in block " << cnt << endl;
					a = cnt;
					b = Error_pattern[cur_error * 3 + 1];
					c = Error_pattern[cur_error * 3 + 2];
					//buffer[b] ^= c;

					Error_undetected[nb_error_undetected * 3 + 0] = a;
					Error_undetected[nb_error_undetected * 3 + 1] = b;
					Error_undetected[nb_error_undetected * 3 + 2] = c;
					nb_error_undetected++;
					cur_error++;
				}
			}

			int i;
			for (i = 0; i < L - 4; i++) {
				recovered_data[cnt * information_length + i] = buffer[i];
			}
			recovered_data_size += L;

			if (cur_error < nb_error && cnt == Error_pattern[cur_error * 3 + 0]) {
				cur_error++;
			}

			//orbiter_kernel_system::os_interface Os;

#if 0
			a = Os.random_integer(1000000);
			if (a < threshold) {
				b = Os.random_integer(L);
				c = Os.random_integer(256);
				Error_pattern[nb_error * 3 + 0] = cnt;
				buffer[b] ^= c;
				Error_pattern[nb_error * 3 + 1] = b;
				Error_pattern[nb_error * 3 + 2] = c;
				nb_error++;
			}
#endif
		}

	}

	cout << "nb_error_detected = " << nb_error_detected << " / " << nb_error << endl;

	int nb_undetected_errors;


	nb_undetected_errors = nb_error - nb_error_detected;

	cout << "nb_undetected_errors = " << nb_error_undetected << endl;

#if 1
	Fio.lint_matrix_write_csv(fname_error_detected, Faulty_blocks, nb_error, 3);
	cout << "Written file " << fname_error_detected << " of size " << Fio.file_size(fname_error_detected) << endl;

	Fio.lint_matrix_write_csv(fname_error_undetected, Error_undetected, nb_error_undetected, 3);
	cout << "Written file " << fname_error_undetected << " of size " << Fio.file_size(fname_error_undetected) << endl;
#endif


	{
		ofstream ost(fname_recovered, ios::binary);

		ost.write(recovered_data, recovered_data_size);
	}
	cout << "Written file " << fname_recovered << " of size " << Fio.file_size(fname_recovered) << endl;

	FREE_lint(Error_pattern);
	FREE_lint(Error_undetected);

	if (f_v) {
		cout << "coding_theory_domain::check_errors done" << endl;
	}

}


void coding_theory_domain::extract_block(
		crc_options_description *Crc_options_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::extract_block " << endl;
	}

	if (!Crc_options_description->f_input) {
		cout << "coding_theory_domain::extract_block please use -input <fname>" << endl;
		exit(1);
	}
	if (!Crc_options_description->f_output) {
		cout << "coding_theory_domain::extract_block please use -output <fname>" << endl;
		exit(1);
	}
	if (!Crc_options_description->f_block_length) {
		cout << "coding_theory_domain::extract_block please use -block_length <block_length>" << endl;
		exit(1);
	}
	if (!Crc_options_description->f_selected_block) {
		cout << "coding_theory_domain::extract_block please use -selected_block <selected_block>" << endl;
		exit(1);
	}
	int block_length;
	int information_length;

	block_length = Crc_options_description->block_length;
	information_length = block_length - 4;
	if (f_v) {
		cout << "coding_theory_domain::extract_block block_length = " << block_length << endl;
		cout << "coding_theory_domain::extract_block information_length = " << information_length << endl;

	}

	std::string fname_coded;
	std::string fname_out;
	std::string fname_error_log;
	std::string fname_error_detected;
	std::string fname_error_undetected;


	data_structures::string_tools ST;
	//string fname_error;
	//int information_length = block_length - 4;


	fname_coded.assign(Crc_options_description->input_fname);
	fname_out.assign(Crc_options_description->output_fname);
	fname_error_log.assign(Crc_options_description->error_log_fname);

	fname_error_detected.assign(Crc_options_description->input_fname);
	ST.chop_off_extension(fname_error_detected);
	fname_error_detected.append("_err_detected.csv");

	fname_error_undetected.assign(Crc_options_description->input_fname);
	ST.chop_off_extension(fname_error_undetected);
	fname_error_undetected.append("_err_undetected.csv");

	orbiter_kernel_system::file_io Fio;

	long int N, L;
	long int nb_blocks;
	char *buffer;
	//char *recovered_data;
	//long int recovered_data_size = 0;

	N = Fio.file_size(fname_coded);

	if (f_v) {
		cout << "coding_theory_domain::check_errors input file size = " << N << endl;
	}
	buffer = NEW_char(block_length);
	//recovered_data = NEW_char(N);


	long int *Error_pattern;
	//long int *Error_undetected;
	int nb_error = 0;
	int m;

	cout << "Reading file " << fname_error_log << " of size " << Fio.file_size(fname_error_log) << endl;
	Fio.lint_matrix_read_csv(fname_error_log, Error_pattern, nb_error, m, verbose_level);
	if (m != 3) {
		cout << "m != 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "coding_theory_domain::check_errors nb_error = " << nb_error << endl;
	}


	nb_blocks = (N + block_length - 1) / block_length;
	if (f_v) {
		cout << "coding_theory_domain::check_errors nb_blocks = " << nb_blocks << endl;
	}

	{
		ifstream ist(fname_coded, ios::binary);



		long int a, b, c;
		long int cnt;
		long int cur_error;

		cur_error = 0;

		for (cnt = 0; cnt < nb_blocks; cnt++) {

			if ((cnt + 1) * block_length > N) {
				L = N - cnt * block_length;
			}
			else {
				L = block_length;
			}

			// read information length + 4 bytes
			// (this includes the 4 byte check sum at the very end)

			ist.read(buffer, L);



			if (cnt != Crc_options_description->selected_block) {

				while (cur_error < nb_error && cnt == Error_pattern[cur_error * 3 + 0]) {
					//cout << "recovering error " << cur_error << " in block " << cnt << endl;
					a = cnt;
					b = Error_pattern[cur_error * 3 + 1];
					c = Error_pattern[cur_error * 3 + 2];
					buffer[b] ^= c;
					cur_error++;
				}

				continue;
			}
			{
				ofstream ost(fname_out, ios::binary);

				ost.write(buffer, L);
			}
			cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;

			cout << "errors in block " << Crc_options_description->selected_block << ":" << endl;
			while (cur_error < nb_error && cnt == Error_pattern[cur_error * 3 + 0]) {
				//cout << "recovering error " << cur_error << " in block " << cnt << endl;
				a = cnt;
				b = Error_pattern[cur_error * 3 + 1];
				c = Error_pattern[cur_error * 3 + 2];

				cout << a << " : " << b << " : " << c << endl;
				cur_error++;
			}

		}
	}

	if (f_v) {
		cout << "coding_theory_domain::extract_block done" << endl;
	}
}




}}}


