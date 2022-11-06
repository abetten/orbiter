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

	snprintf(str, sizeof(str), "_n%d_q%d", n_max, q);

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

	snprintf(str, sizeof(str), "Hamming_n%d_q%d.csv", n, q);
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

void coding_theory_domain::codewords_affine(field_theory::finite_field *F,
		int n, int k,
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

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator before Gauss_int" << endl;
	}

	rk = F->Linear_algebra->Gauss_int(A,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, m, n, n,
		verbose_level);

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator after Gauss_int" << endl;
	}


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

		snprintf(str, sizeof(str), "_%d_%d.tex", n, nb_words);
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

		snprintf(str, sizeof(str), "_diagram_%d_%d.csv", n, nb_words);
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

		snprintf(str, sizeof(str), "_diagram_01_%d_%d.csv", n, nb_words);
		fname.append(str);
		orbiter_kernel_system::file_io Fio;

		Fio.int_matrix_write_csv(fname, M2, nb_rows, nb_cols);

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	{
		char str[1000];

		string fname;

		fname.assign(label);

		snprintf(str, sizeof(str), "_holes_%d_%d.csv", n, nb_words);
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

	snprintf(fname, sizeof(fname), "code_%d_%d.tex", n, nb_words);

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

		snprintf(str, sizeof(str), "code_matrix_%d_%d.csv", nb_rows, nb_cols);
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


	snprintf(fname, sizeof(fname), "code_%d_%d_Levi_%d_%d.bin",
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

		snprintf(str, sizeof(str), "long_code_genma_n%d_k%d.csv", n, k);
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

			snprintf(str, sizeof(str), "long_code_genma_n%d_k%d_codeword_%d.csv", n, k, i);
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

			snprintf(str, sizeof(str), "long_code_genma_n%d_k%d_nearest_codeword_fourier.csv", n, k);
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

			snprintf(str, sizeof(str), "long_code_genma_n%d_k%d_nearest_codeword.csv", n, k);
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


}}}


