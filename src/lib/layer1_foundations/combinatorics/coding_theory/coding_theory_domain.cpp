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
namespace combinatorics {
namespace coding_theory {




coding_theory_domain::coding_theory_domain()
{
	Record_birth();

}

coding_theory_domain::~coding_theory_domain()
{
	Record_death();

}




void coding_theory_domain::make_mac_williams_equations(
		algebra::ring_theory::longinteger_object *&M,
		int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "coding_theory_domain::make_mac_williams_equations" << endl;
	}
	M = NEW_OBJECTS(algebra::ring_theory::longinteger_object, (n + 1) * (n + 1));

	for (i = 0; i <= n; i++) {
		for (j = 0; j <= n; j++) {
			Combi.krawtchouk(M[i * (n + 1) + j], n, q, i, j);
		}
	}


	if (f_v) {
		cout << "coding_theory_domain::make_mac_williams_equations done" << endl;
	}
}

void coding_theory_domain::report_macwilliams_system(
		int q, int n, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	coding_theory_domain C;
	algebra::ring_theory::longinteger_object *M;
	int i, j;

	if (f_v) {
		cout << "interface_coding_theory::report_macwilliams_system" << endl;
	}

	C.make_mac_williams_equations(M, n, k, q, verbose_level);


	string fname;
	string author;
	string title;
	string extra_praeamble;



	fname = "macwilliams_n" + std::to_string(n) + "_k"
			+ std::to_string(k) + "_q" + std::to_string(q) + ".tex";
	title = "MacWilliams system for $[" + std::to_string(n) + ","
			+ std::to_string(k) + "]$ code over GF($" + std::to_string(q) + "$)";



	other::l1_interfaces::latex_interface L;
	int nb_rows, nb_cols;

	nb_rows = n + 1;
	nb_cols = n + 1;

	L.report_matrix_longinteger(
			fname,
			title,
			author,
			extra_praeamble,
			M, nb_rows, nb_cols, verbose_level);

#if 0
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
#endif

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
		cout << "coding_theory_domain::report_macwilliams_system done" << endl;
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

			if (f_v) {
				cout << "n=" << n << " k=" << k << " q=" << q << endl;
			}
			d_S = singleton_bound_for_d(n, k, q, 0 /*verbose_level*/);

			if (f_v) {
				cout << "d_S=" << d_S << endl;
			}
			d_H = hamming_bound_for_d(n, k, q, 0 /*verbose_level*/);

			if (f_v) {
				cout << "d_H=" << d_H << endl;
			}

			d_P = plotkin_bound_for_d(n, k, q, 0 /*verbose_level*/);
			if (f_v) {
				cout << "d_P=" << d_P << endl;
			}

			d_G = griesmer_bound_for_d(n, k, q, 0 /*verbose_level*/);
			if (f_v) {
				cout << "d_G=" << d_G << endl;
			}

			d_GV = gilbert_varshamov_lower_bound_for_d(n, k, q, 0 /*verbose_level*/);
			if (f_v) {
				cout << "d_GV=" << d_GV << endl;
			}

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
	other::orbiter_kernel_system::file_io Fio;
	std::string fname;

	fname= "table_of_bounds_n" + std::to_string(n_max)
			+ "_q" + std::to_string(q) + ".csv";

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

	Fio.Csv_file_support->lint_matrix_write_csv_override_headers(
			fname, headers, T, N, nb_cols);
	cout << "Written file " << fname
			<< " of size " << Fio.file_size(fname) << endl;

	FREE_lint(T);
	delete [] headers;


	if (f_v) {
		cout << "coding_theory_domain::make_table_of_bounds done" << endl;
	}
}

void coding_theory_domain::make_gilbert_varshamov_code(
		int n, int k, int d,
		algebra::field_theory::finite_field *F,
		int *&genma, int *&checkma,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code" << endl;
	}
	geometry::other_geometry::geometry_global Gg;
	int nmk;
	int N_points;
	long int *set;
	int *f_forbidden;


	nmk = n - k;

	set = NEW_lint(n);

	N_points = Gg.nb_PG_elements(nmk - 1, F->q);

	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"N_points = " << N_points << endl;
	}

	f_forbidden = NEW_int(N_points);
	Int_vec_zero(f_forbidden, N_points);

	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"before make_gilbert_varshamov_code_recursion" << endl;
	}
	make_gilbert_varshamov_code_recursion(F,
			n, k, d, N_points,
			set, f_forbidden, 0 /*level*/,
			verbose_level);
	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"after make_gilbert_varshamov_code_recursion" << endl;
	}



	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code found "
				"the following parity check matrix as projective set: " << endl;
		Lint_vec_print(cout, set, n);
		cout << endl;
	}

	int *M;
	M = NEW_int(n * n);

	matrix_from_projective_set(F,
			n, nmk, set,
			M,
			verbose_level);

	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"parity check matrix:" << endl;
		Int_matrix_print(M, nmk, n);
	}

	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"parity check matrix:" << endl;
		Int_vec_print_fully(cout, M, nmk * n);
		cout << endl;
	}

	F->Linear_algebra->RREF_and_kernel(
			n, nmk, M, 0 /* verbose_level */);

	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"generator matrix:" << endl;
		Int_matrix_print(M + nmk * n, k, n);
	}


	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"generator matrix:" << endl;
		Int_vec_print_fully(cout, M + nmk * n, k * n);
		cout << endl;
	}


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
		algebra::field_theory::finite_field *F,
		int n, int k, int d, long int N_points,
		long int *set, int *f_forbidden, int level,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"level = " << level << endl;
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"set = ";
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
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	int cnt;

	nmk = n - k;
	set[level] = a;
	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"nmk = " << nmk << endl;
	}

	f_forbidden[a] = true;
	add_set.push_back(a);
	cnt = 1;


	if (level) {

		int *subset;
		subset = NEW_int(level - 1);
		int s, N, h, u, c, e, t;
		long int f;
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
					F->Projective_space_basic->PG_element_unrank_modified(
							v1, 1, nmk, e);
					//P->unrank_point(v1, e);
					for (t = 0; t < nmk; t++) {
						v3[t] = F->add(v3[t], v1[t]);
					}
				}
				F->Projective_space_basic->PG_element_unrank_modified(
						v1, 1, nmk, set[level]);
				//P->unrank_point(v1, set[level]);
				for (t = 0; t < nmk; t++) {
					v3[t] = F->add(v3[t], v1[t]);
				}
				F->Projective_space_basic->PG_element_rank_modified(
						v3, 1, nmk, f);
				//f = P->rank_point(v3);
				if (f_v) {
					cout << "h=" << h << " / " << N << " : ";
					Int_vec_print(cout, subset, i);
					cout << " : ";
					Int_vec_print(cout, v3, nmk);
					cout << " : " << f;
				}
				if (!f_forbidden[f]) {
					f_forbidden[f] = true;
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
				"level = " << level << " : cnt = " << cnt
				<< " calling the recursion:" << endl;
	}
	make_gilbert_varshamov_code_recursion(
			F, n, k, d, N_points,
			set, f_forbidden, level + 1,
			verbose_level);
	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code "
				"level = " << level << " : cnt = " << cnt
				<< " done with the recursion:" << endl;
	}

	for (i = 0; i < add_set.size(); i++) {
		b = add_set[i];
		f_forbidden[b] = false;
	}


	if (f_v) {
		cout << "coding_theory_domain::make_gilbert_varshamov_code done" << endl;
	}
}




int coding_theory_domain::gilbert_varshamov_lower_bound_for_d(
		int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::ring_theory::longinteger_domain D;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	int i, d;
	algebra::ring_theory::longinteger_object qnmk, qm1, qm1_power, S, s, a, b;

	if (f_v) {
		cout << "coding_theory_domain::gilbert_varshamov_lower_bound_for_d" << endl;
	}
	qnmk.create(q);
	qm1.create(q - 1);
	D.power_int(qnmk, n - k);
	qm1_power.create(1);
	S.create(0);
	//cout << "gilbert_varshamov_lower_bound_for_d: q=" << q << " n=" << n << " k=" << k << " " << q << "^" << n - k << " = " << qnmk << endl;
	for (i = 0; ; i++) {
		Combi.binomial(b, n - 1, i, false);
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
		cout << "coding_theory_domain::gilbert_varshamov_lower_bound_for_d "
				"done" << endl;
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
	algebra::ring_theory::longinteger_object qnmk, qm1, qm1_power, B, s, a, b;
	algebra::ring_theory::longinteger_domain D;
	combinatorics::other_combinatorics::combinatorics_domain Combi;


	if (f_v) {
		cout << "coding_theory_domain::hamming_bound_for_d" << endl;
	}
	qnmk.create(q);
	qm1.create(q - 1);
	D.power_int(qnmk, n - k);
	qm1_power.create(1);
	B.create(0);
	if (f_vv) {
		cout << "coding_theory_domain::hamming_bound_for_d: "
			"q=" << q << " n=" << n << " k=" << k << " "
			<< q << "^" << n - k << " = " << qnmk << endl;
	}
	for (e = 0; ; e++) {
		Combi.binomial(b, n, e, false);
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
	algebra::ring_theory::longinteger_object qkm1, qk, qm1, a, b, c, Q, R;
	algebra::ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "coding_theory_domain::plotkin_bound_for_d" << endl;
	}

	// d \le \frac{n q^{k-1}}{q^k-1}

	qkm1.create(q);
	D.power_int(qkm1, k - 1);
	a.create(n);
	D.mult(a, qkm1, b);
		// now b = n q^{k-1}

	a.create(q - 1);
	D.mult(b, a, c);
		// now c = n q^{k-1} (q - 1)


	a.create(q);
	D.mult(a, qkm1, qk);
		// now qk = q^k

	a.create(-1);
	D.add(qk, a, b);
		// now b = 2^k - 1

	if (f_vv) {
		cout << "coding_theory_domain::plotkin_bound_for_d "
				"q=" << q << " n=" << n << " k=" << k << endl;
	}
	D.integral_division(c, b, Q, R, false /* verbose_level */);
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
	algebra::ring_theory::longinteger_object qq, qi, d1, S, Q, R, one, a, b;
	algebra::ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "coding_theory_domain::griesmer_bound_for_n" << endl;
	}
	one.create(1);
	d1.create(d);
	qq.create(q);
	qi.create(1);
	S.create(0);
	if (f_vv) {
		cout << "coding_theory_domain::griesmer_bound_for_n q=" << q
				<< " d=" << d << " k=" << k << endl;
	}
	for (i = 0; i < k; i++) {
		D.integral_division(d1, qi, Q, R, false /* verbose_level */);
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




void coding_theory_domain::make_Hamming_space_distance_matrix(
		int n, algebra::field_theory::finite_field *F,
		int f_projective, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int width, height;
	int *v;
	int *w;
	int *Table;
	geometry::other_geometry::geometry_global Gg;
	//field_theory::finite_field *F = NULL;

	if (f_v) {
		cout << "coding_theory_domain::make_Hamming_space_distance_matrix" << endl;
	}

	v = NEW_int(n);
	w = NEW_int(n);

	if (f_projective) {
		width = height = Gg.nb_PG_elements(n - 1, F->q);
	}
	else {
		width = height = Gg.nb_AG_elements(n, F->q);
	}


	if (f_v) {
		cout << "coding_theory_domain::make_Hamming_space_distance_matrix "
				"width=" << width << endl;
	}

	int i, j, d, h;

	Table = NEW_int(height * width);
	for (i = 0; i < height; i++) {

		if (f_projective) {
			F->Projective_space_basic->PG_element_unrank_modified(
					v, 1 /*stride*/, n, i);
		}
		else {
			Gg.AG_element_unrank(F->q, v, 1, n, i);
		}

		for (j = 0; j < width; j++) {

			if (f_projective) {
				F->Projective_space_basic->PG_element_unrank_modified(
						w, 1 /*stride*/, n, j);
			}
			else {
				Gg.AG_element_unrank(F->q, w, 1, n, j);
			}

			d = 0;
			for (h = 0; h < n; h++) {
				if (v[h] != w[h]) {
					d++;
				}
			}


			Table[i * width + j] = d;

		}
	}

	string fname;
	other::orbiter_kernel_system::file_io Fio;


	fname = "Hamming_n" + std::to_string(n)
			+ "_q" + std::to_string(F->q) + ".csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Table, height, width);

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "coding_theory_domain::make_Hamming_space_distance_matrix" << endl;
	}

}


void coding_theory_domain::compute_and_print_projective_weights(
		std::ostream &ost, algebra::field_theory::finite_field *F,
		int *M, int n, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::compute_and_print_projective_weights" << endl;
	}
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

	if (f_v) {
		cout << "coding_theory_domain::compute_and_print_projective_weights done" << endl;
	}
}

int coding_theory_domain::code_minimum_distance(
		algebra::field_theory::finite_field *F, int n, int k,
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

void coding_theory_domain::make_codewords_sorted(
		algebra::field_theory::finite_field *F,
		int n, int k,
		int *genma, // [k * n]
		long int *&codewords, // q^k
		long int &N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::make_codewords_sorted" << endl;
	}

	if (f_v) {
		cout << "coding_theory_domain::make_codewords_sorted "
				"before make_codewords" << endl;
	}
	make_codewords(F,
			n, k,
			genma, // [k * n]
			codewords, // q^k
			N,
			verbose_level);
	if (f_v) {
		cout << "coding_theory_domain::make_codewords_sorted "
				"after make_codewords" << endl;
	}

	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "coding_theory_domain::make_codewords_sorted "
				"before Sorting.lint_vec_heapsort" << endl;
	}
	Sorting.lint_vec_heapsort(codewords, N);
	if (f_v) {
		cout << "coding_theory_domain::make_codewords_sorted "
				"after Sorting.lint_vec_heapsort" << endl;
	}

	if (f_v) {
		cout << "coding_theory_domain::make_codewords_sorted "
				"N=" << N << endl;
		//Lint_vec_print_fully(cout, codewords, N);
		//cout << endl;
	}

	if (f_v) {
		cout << "coding_theory_domain::make_codewords_sorted done" << endl;
	}
}

void coding_theory_domain::make_codewords(
		algebra::field_theory::finite_field *F,
		int n, int k,
		int *genma, // [k * n]
		long int *&codewords, // q^k
		long int &N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::make_codewords" << endl;
	}

	algebra::number_theory::number_theory_domain NT;

	N = NT.i_power_j(F->q, k);

	codewords = NEW_lint(N);

	if (f_v) {
		cout << "coding_theory_domain::make_codewords "
				"before codewords_affine" << endl;
	}
	codewords_affine(F, n, k,
			genma, // [k * n]
			codewords, // q^k
			verbose_level);
	if (f_v) {
		cout << "coding_theory_domain::make_codewords "
				"after codewords_affine" << endl;
	}

	if (f_v) {
		cout << "coding_theory_domain::make_codewords done" << endl;
	}
}

void coding_theory_domain::codewords_affine(
		algebra::field_theory::finite_field *F,
		int n, int k,
	int *code, // [k * n]
	long int *codewords, // q^k
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int N, h, rk;
	int *msg;
	int *word;
	geometry::other_geometry::geometry_global Gg;

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

		F->Linear_algebra->mult_vector_from_the_left(
				msg, code, word, k, n);

		rk = Gg.AG_element_rank(F->q, word, 1, n);

		codewords[h] = rk;
	}
	FREE_int(msg);
	FREE_int(word);
	if (f_v) {
		cout << "coding_theory_domain::codewords_affine done" << endl;
	}
}

void coding_theory_domain::codewords_table(
		algebra::field_theory::finite_field *F,
		int n, int k,
	int *code, // [k * n]
	int *&codewords, // [q^k * n]
	long int &N, // q^k
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int h;
	int *msg;
	int *word;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "coding_theory_domain::codewords_table" << endl;
	}
	N = Gg.nb_AG_elements(k, F->q);
	if (f_v) {
		cout << N << " messages" << endl;
	}
	msg = NEW_int(k);
	word = NEW_int(n);
	codewords = NEW_int(N * n);

	for (h = 0; h < N; h++) {

		Gg.AG_element_unrank(F->q, msg, 1, k, h);

		F->Linear_algebra->mult_vector_from_the_left(
				msg, code, word, k, n);

		Int_vec_copy(word, codewords + h * n, n);
	}
	FREE_int(msg);
	FREE_int(word);
	if (f_v) {
		cout << "coding_theory_domain::codewords_affine done" << endl;
	}
}

void coding_theory_domain::code_projective_weight_enumerator(
		algebra::field_theory::finite_field *F,
	int n, int k,
	int *code, // [k * n]
	int *weight_enumerator, // [n + 1]
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	long int N, h, wt, i;
	int *msg;
	int *word;
	int t0, t1, dt;
	geometry::other_geometry::geometry_global Gg;
	other::orbiter_kernel_system::os_interface Os;

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

		F->Linear_algebra->mult_vector_from_the_left(
				msg, code, word, k, n);

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

void coding_theory_domain::code_weight_enumerator(
		algebra::field_theory::finite_field *F,
	int n, int k,
	int *code, // [k * n]
	int *weight_enumerator, // [n + 1]
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	long int N, N100, h, wt, i;
	int *msg;
	int *word;
	int t0, t1, dt;
	geometry::other_geometry::geometry_global Gg;
	other::orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "coding_theory_domain::code_weight_enumerator" << endl;
	}
	N = Gg.nb_AG_elements(k, F->q);
	if (f_v) {
		cout << "Number of codewords = " << N << endl;
	}

	N100 = N / 100;
	if (f_v) {
		cout << "1% = " << N100 << endl;
	}


	msg = NEW_int(k);
	word = NEW_int(n);

	Int_vec_zero(weight_enumerator, n + 1);

	for (h = 0; h < N; h++) {
		if ((h % N100) == 0 || (h % ONE_MILLION) == 0) {
			t1 = Os.os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " = " << h / N100 << "% : ";
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

		F->Linear_algebra->mult_vector_from_the_left(
				msg, code, word, k, n);


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


void coding_theory_domain::code_weight_enumerator_fast(
		algebra::field_theory::finite_field *F,
		int n, int k,
	int *code, // [k * n]
	int *weight_enumerator, // [n + 1]
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int N, h, wt, i;
	int *msg;
	int *word;
	int t0, t1, dt;
	geometry::other_geometry::geometry_global Gg;
	other::orbiter_kernel_system::os_interface Os;

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
			if (f_vv) {
				cout << setw(10) << h << " / " << setw(10) << N << " : ";
				Os.time_check_delta(cout, dt);
				cout << endl;

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

		F->Projective_space_basic->PG_element_unrank_modified(
				msg, 1, k, h);

		//AG_element_unrank(q, msg, 1, k, h);

		F->Linear_algebra->mult_vector_from_the_left(
				msg, code, word, k, n);

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
		algebra::field_theory::finite_field *F,
	int n, int k,
	int *code, // [k * n]
	int *&weights, // will be allocated [N]
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int N, h, wt, i;
	int *msg;
	int *word;
	int t0, t1, dt;
	geometry::other_geometry::geometry_global Gg;
	other::orbiter_kernel_system::os_interface Os;

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
		if (f_vv) {
			if ((h % ONE_MILLION) == 0) {
				t1 = Os.os_ticks();
				dt = t1 - t0;
				cout << setw(10) << h << " / " << setw(10) << N << " : ";
				Os.time_check_delta(cout, dt);
				cout << endl;
			}
		}

		F->Projective_space_basic->PG_element_unrank_modified(
				msg, 1, k, h);

		//AG_element_unrank(q, msg, 1, k, h);

		F->Linear_algebra->mult_vector_from_the_left(
				msg, code, word, k, n);

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

void coding_theory_domain::mac_williams_equations(
		algebra::ring_theory::longinteger_object *&M, int n, int k, int q)
{
	combinatorics::other_combinatorics::combinatorics_domain D;
	int i, j;

	M = NEW_OBJECTS(algebra::ring_theory::longinteger_object, (n + 1) * (n + 1));

	for (i = 0; i <= n; i++) {
		for (j = 0; j <= n; j++) {
			D.krawtchouk(M[i * (n + 1) + j], n, q, i, j);
		}
	}
}

void coding_theory_domain::determine_weight_enumerator()
{
	int n = 19, k = 7, q = 2;
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object *M, *A1, *A2, qk;
	int i;

	qk.create(q);
	D.power_int(qk, k);
	cout << q << "^" << k << " = " << qk << endl;

	mac_williams_equations(M, n, k, q);

	D.matrix_print_tex(cout, M, n + 1, n + 1);

	A1 = NEW_OBJECTS(algebra::ring_theory::longinteger_object, n + 1);
	A2 = NEW_OBJECTS(algebra::ring_theory::longinteger_object, n + 1);
	for (i = 0; i <= n; i++) {
		A1[i].create(0);
	}
	A1[0].create(1);
	A1[8].create(78);
	A1[12].create(48);
	A1[16].create(1);
	D.matrix_print_tex(cout, A1, n + 1, 1);

	D.matrix_product(M, A1, A2, n + 1, n + 1, 1);
	D.matrix_print_tex(cout, A2, n + 1, 1);

	D.matrix_entries_integral_division_exact(A2, qk, n + 1, 1);

	D.matrix_print_tex(cout, A2, n + 1, 1);

	FREE_OBJECTS(M);
	FREE_OBJECTS(A1);
	FREE_OBJECTS(A2);
}


void coding_theory_domain::do_weight_enumerator(
		algebra::field_theory::finite_field *F,
		int *M, int m, int n,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *A;
	int *base_cols;
	int *weight_enumerator;
	long int rk, i;
	other::l1_interfaces::latex_interface Li;

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator" << endl;
	}

	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	weight_enumerator = NEW_int(n + 1);
	Int_vec_copy(M, A, m * n);

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator "
				"input matrix:" << endl;
		Int_matrix_print(A, m, n);

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.print_integer_matrix_tex(cout, A, m, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;

	}

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator "
				"before Gauss_int" << endl;
	}

	rk = F->Linear_algebra->Gauss_int(
			A,
		false /* f_special */, true /* f_complete */, base_cols,
		false /* f_P */, NULL /*P*/, m, n, n,
		verbose_level);

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator "
				"after Gauss_int" << endl;
	}


	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator "
				"after RREF:" << endl;
		Int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;


		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.print_integer_matrix_tex(cout, A, rk, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;


		cout << "coding_theory_domain::do_weight_enumerator "
				"coefficients:" << endl;
		Int_vec_print(cout, A, rk * n);
		cout << endl;
	}

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator "
				"before code_weight_enumerator" << endl;
	}
	code_weight_enumerator(F, n, rk,
		A /* code */, // [k * n]
		weight_enumerator, // [n + 1]
		verbose_level);
	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator "
				"after code_weight_enumerator" << endl;
	}

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator "
				"The weight enumerator is:" << endl;
		for (i = 0; i <= n; i++) {
			cout << i << " : " << weight_enumerator[i] << endl;
		}

		int f_first = true;

		for (i = 0; i <= n; i++) {
			if (weight_enumerator[i] == 0) {
				continue;
			}
			if (f_first) {
				f_first = false;
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


		cout << "coding_theory_domain::do_weight_enumerator "
				"The weight enumerator is:" << endl;
		for (i = 0; i <= n; i++) {
			cout << i << " : " << weight_enumerator[i] << endl;
		}

		f_first = true;

		for (i = 0; i <= n; i++) {
			if (weight_enumerator[i] == 0) {
				continue;
			}
			if (f_first) {
				f_first = false;
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
			cout << "coding_theory_domain::do_weight_enumerator "
					"normalizing from the left" << endl;
		}
		for (i = 0; i < rk; i++) {
			F->Projective_space_basic->PG_element_normalize_from_front(
					A + i * n, 1, n);
		}

		if (f_v) {
			cout << "coding_theory_domain::do_weight_enumerator "
					"after normalize from the left:" << endl;
			Int_matrix_print(A, rk, n);
			cout << "rk=" << rk << endl;
		}
	}

	if (f_normalize_from_the_right) {
		if (f_v) {
			cout << "coding_theory_domain::do_weight_enumerator "
					"normalizing from the right" << endl;
		}
		for (i = 0; i < rk; i++) {
			F->Projective_space_basic->PG_element_normalize(
					A + i * n, 1, n);
		}

		if (f_v) {
			cout << "coding_theory_domain::do_weight_enumerator "
					"after normalize from the right:" << endl;
			Int_matrix_print(A, rk, n);
			cout << "coding_theory_domain::do_weight_enumerator "
					"rk=" << rk << endl;
		}
	}


	FREE_int(A);
	FREE_int(base_cols);
	FREE_int(weight_enumerator);

	if (f_v) {
		cout << "coding_theory_domain::do_weight_enumerator done" << endl;
	}
}


void coding_theory_domain::do_minimum_distance_Brouwer_Zimmermann(
		algebra::field_theory::finite_field *F,
		int *M, int m, int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *A;
	int *base_cols;
	//int *weight_enumerator;
	int rk, i;
	other::l1_interfaces::latex_interface Li;
	other::orbiter_kernel_system::os_interface Os;
	long int t0, t1, dt, tps;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "coding_theory_domain::do_minimum_distance_Brouwer_Zimmermann" << endl;
	}

	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	//weight_enumerator = NEW_int(n + 1);
	Int_vec_copy(M, A, m * n);

	if (f_v) {
		cout << "coding_theory_domain::do_minimum_distance_Brouwer_Zimmermann "
				"input matrix:" << endl;
		Int_matrix_print(A, m, n);

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.print_integer_matrix_tex(cout, A, m, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;

	}

	rk = F->Linear_algebra->Gauss_int(A,
		false /* f_special */, true /* f_complete */, base_cols,
		false /* f_P */, NULL /*P*/, m, n, n,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "coding_theory_domain::do_minimum_distance_Brouwer_Zimmermann "
				"after RREF:" << endl;
		Int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;


		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.print_integer_matrix_tex(cout, A, rk, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;


		cout << "coding_theory_domain::do_minimum_distance_Brouwer_Zimmermann "
				"coefficients:" << endl;
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

	if (f_v) {
		cout << "coding_theory_domain::do_minimum_distance_Brouwer_Zimmermann "
				"before mindist" << endl;
	}
	d = mindist(n, m /* k */, q, A,
		verbose_level - 2, idx_zero, idx_one,
		add_table, mult_table);
	if (f_v) {
		cout << "coding_theory_domain::do_minimum_distance_Brouwer_Zimmermann "
				"after mindist" << endl;
	}

	t1 = Os.os_ticks();

	dt = t1 - t0;

	tps = Os.os_ticks_per_second();
	//cout << "time_check_delta tps=" << tps << endl;

	int days, hours, minutes, seconds;

	Os.os_ticks_to_dhms(dt, tps, days, hours, minutes, seconds);


	if (f_v) {
		cout << "coding_theory_domain::do_minimum_distance_Brouwer_Zimmermann "
				"The minimum distance is d = " << d << ", "
				"computed in " << days << " days, " << hours << " hours, "
				<< minutes << " minutes, " << seconds << " seconds" << endl;
	}


	FREE_int(add_table);
	FREE_int(mult_table);

	FREE_int(A);
	FREE_int(base_cols);
	//FREE_int(weight_enumerator);

	if (f_v) {
		cout << "coding_theory_domain::do_minimum_distance_Brouwer_Zimmermann done" << endl;
	}
}




void coding_theory_domain::matrix_from_projective_set(
		algebra::field_theory::finite_field *F,
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

		F->Projective_space_basic->PG_element_unrank_modified(
				v, 1, k, columns_set_of_size_n[j]);

		// fill column j:

		for (i = 0; i < k; i++) {
			genma[i * n + j] = v[i];
		}

	}

	FREE_int(v);

	if (f_v) {
		cout << "coding_theory_domain::matrix_from_projective_set done" << endl;
	}
}


void coding_theory_domain::do_linear_code_through_columns_of_generator_matrix(
		algebra::field_theory::finite_field *F,
		int n,
		long int *columns_set, int k,
		int *&genma,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::do_linear_code_through_columns_of_generator_matrix" << endl;
	}

	int i, j;
	int *v;
	int *word;
	int *code_word;
	geometry::other_geometry::geometry_global Gg;

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
		cout << "coding_theory_domain::do_linear_code_through_columns_of_generator_matrix "
				"genma:" << endl;
		Int_matrix_print(genma, k, n);
	}


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;



		fname = "code_n" + std::to_string(n) + "_k"
				+ std::to_string(k) + "_q" + std::to_string(F->q) + ".tex";
		title = "Linear $[" + std::to_string(n) + ","
				+ std::to_string(k) + "]$ code over GF($" + std::to_string(F->q) + "$)";



		other::l1_interfaces::latex_interface L;
		int nb_rows, nb_cols;

		nb_rows = k;
		nb_cols = n;

		L.report_matrix(
					fname,
					title,
					author,
					extra_praeamble,
					genma, nb_rows, nb_cols);

#if 0
		{
			ofstream ost(fname);


			l1_interfaces::latex_interface L;


			L.head(ost, false /* f_book*/, true /* f_title */,
				title, author, false /* f_toc */, false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
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
#endif

		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname
					<< " of size " << Fio.file_size(fname) << endl;
		}

	}



	//FREE_lint(set);
	//FREE_int(genma);
	FREE_int(v);
	FREE_int(word);
	FREE_int(code_word);
	//FREE_OBJECT(F);

	if (f_v) {
		cout << "coding_theory_domain::do_linear_code_through_columns_of_generator_matrix done" << endl;
	}
}


void coding_theory_domain::do_polynomial(
		int n,
		int polynomial_degree,
		int polynomial_nb_vars,
		std::string &polynomial_text,
		//int f_embellish, int embellish_radius,
		int verbose_level)
// this function is not used anywhere
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::do_polynomial" << endl;
	}
	if (f_v) {
		cout << "reading polynomial " << polynomial_text << " of degree "
				<< polynomial_degree << " in "
				<< polynomial_nb_vars << " variables" << endl;
	}

	long int *poly_monomials;
	int poly_monomials_sz;
	algebra::ring_theory::homogeneous_polynomial_domain *Poly;
	algebra::field_theory::finite_field *Fq;
	int *mon;
	int *coeff;
	long int a;
	int i, j, b, idx;
	monomial_ordering_type Monomial_ordering_type = t_PART;

	Lint_vec_scan(polynomial_text, poly_monomials, poly_monomials_sz);
	cout << "polynomial after scan: ";
	Lint_vec_print(cout, poly_monomials, poly_monomials_sz);
	cout << endl;

	Poly = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);
	Fq = NEW_OBJECT(algebra::field_theory::finite_field);

	Fq->finite_field_init_small_order(
			2,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0 /* verbose_level */);

	Poly->init(
			Fq, polynomial_nb_vars, polynomial_degree,
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
	geometry::other_geometry::geometry_global Gg;
	other::data_structures::sorting Sorting;
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

#if 0
	if (f_v) {
		cout << "coding_theory_domain::do_polynomial "
				"before investigate_code" << endl;
	}
	investigate_code(Fq, set, set_sz, n,
			f_embellish, embellish_radius,
			verbose_level);
	if (f_v) {
		cout << "coding_theory_domain::do_polynomial "
				"after investigate_code" << endl;
	}
#endif

	FREE_int(f);

	if (f_v) {
		cout << "coding_theory_domain::do_polynomial done" << endl;
	}
}

void coding_theory_domain::do_sylvester_hadamard(
		algebra::field_theory::finite_field *F3,
		int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::do_sylvester_hadamard" << endl;
	}
	int i;

	if (F3->q != 3) {
		cout << "coding_theory_domain::do_sylvester_hadamard "
				"field should be of order 3." << endl;
		exit(1);

	}
	if (n % 4) {
		cout << "for Hadamard matrices, n must be divisible by 4." << endl;
		exit(1);
	}
	int m = n >> 2;
	int nb_factors, sz, sz1, j, a;
	int *M1;
	int *M2;
	int H2[4] = {1,1,1,2};
	algebra::number_theory::number_theory_domain NT;
	geometry::other_geometry::geometry_global Gg;

	nb_factors = NT.int_log2(m);

	if (f_v) {
		cout << "nb_factors = " << nb_factors << endl;
	}

	if ((2 << nb_factors) != n) {
		cout << "for Sylvester type Hadamard matrices, "
				"n must be 4 times a power of two" << endl;
		exit(1);
	}

	M1 = NEW_int(2 * n * n);
	M2 = NEW_int(2 * n * n);
	//field_theory::finite_field *F;

	//F = NEW_OBJECT(field_theory::finite_field);
	//F->finite_field_init(3, false /* f_without_tables */, 0);
	Int_vec_copy(H2, M1, 4);
	sz = 2;
	for (i = 0; i < nb_factors; i++) {

		if (f_v) {
			cout << "coding_theory_domain::do_sylvester_hadamard M1=" << endl;
			Int_matrix_print(M1, sz, sz);
		}

		F3->Linear_algebra->Kronecker_product_square_but_arbitrary(
				M1, H2,
				sz, 2, M2, sz1,
				verbose_level);
		Int_vec_copy(M2, M1, sz1 * sz1);

		sz = sz1;
	}
	if (f_v) {
		cout << "coding_theory_domain::do_sylvester_hadamard "
				"Sylvester type Hadamard matrix:" << endl;
		Int_matrix_print(M1, sz, sz);
	}
	for (i = 0; i < sz; i++) {
		for (j = 0; j < sz; j++) {
			a = M1[i * sz + j];
			M1[(sz + i) * sz + j] = F3->negate(a);
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
	if (f_v) {
		cout << "coding_theory_domain::do_sylvester_hadamard "
				"Sylvester type Hadamard code:" << endl;
		Int_matrix_print(M1, 2 * sz, sz);
	}

	{
		string fname;
		other::orbiter_kernel_system::file_io Fio;

		fname = "Sylvester_Hadamard_code_" + std::to_string(n) + ".csv";
		Fio.Csv_file_support->int_matrix_write_csv(
				fname, M1, 2 * sz, sz);
		if (f_v) {
			cout << "coding_theory_domain::do_sylvester_hadamard written file "
					<< fname << " of size " << Fio.file_size(fname) << endl;
		}

	}



	long int *set;

	set = NEW_lint(2 * sz);
	for (i = 0; i < 2 * sz; i++) {
		set[i] = Gg.AG_element_rank(2, M1 + i * sz, 1, sz);
	}

	{
		string fname;
		other::orbiter_kernel_system::file_io Fio;

		fname = "Sylvester_Hadamard_code_ranks_" + std::to_string(n) + ".csv";
		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, set, 2 * sz, 1);

		if (f_v) {
			cout << "coding_theory_domain::do_sylvester_hadamard written file "
					<< fname << " of size " << Fio.file_size(fname) << endl;
		}

	}


#if 0
	investigate_code(F, set, 2 * sz, n,
			f_embellish, embellish_radius,
			verbose_level);
#endif

	FREE_lint(set);

	FREE_int(M1);
	FREE_int(M2);
	//FREE_OBJECT(F);

}



void coding_theory_domain::field_reduction(
		algebra::field_theory::finite_field *FQ,
		algebra::field_theory::finite_field *Fq,
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

	algebra::field_theory::subfield_structure *Sub;

	Sub = NEW_OBJECT(algebra::field_theory::subfield_structure);

	Sub->init(FQ, Fq, verbose_level);

	if (f_v) {
		Sub->print_embedding();
	}

	Get_int_vector_from_label(genma_text, M, sz, verbose_level);

	if (sz != m * n) {
		cout << "sz != m * n" << endl;
		exit(1);
	}

	M2 = NEW_int(Sub->s * m * Sub->s * n);

	// field reduction of the m by n matrix.
	// The output will have size (s * m) x (s * n).

	for (i = 0; i < m; i++) {
		Sub->field_reduction(
				M + i * n, n,
				M2 + (i * Sub->s) * Sub->s * n,
				verbose_level);
	}


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = "field_reduction_Q" + std::to_string(FQ->q)
				+ "_q" + std::to_string(Fq->q)
				+ "_" + std::to_string(m) + "_" + std::to_string(n) + ".tex";
		title = "Field Reduction";




#if 0
		{
			ofstream ost(fname);
			l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);



			ost << "$$" << endl;
			ost << "\\left[" << endl;
			L.int_matrix_print_tex(
					ost, M2, m * Sub->s, Sub->s * n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;

			Int_vec_print_fully(
					ost, M2, m * Sub->s * Sub->s * n);
			ost << "\\\\" << endl;



			L.foot(ost);

		}
#endif

		other::l1_interfaces::latex_interface L;
		int nb_rows, nb_cols;

		nb_rows = m * Sub->s;
		nb_cols = Sub->s * n;

		L.report_matrix(
					fname,
					title,
					author,
					extra_praeamble,
					M2, nb_rows, nb_cols);


		other::orbiter_kernel_system::file_io Fio;

		cout << "coding_theory_domain::field_reduction written "
				"file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

		string fname_csv;

		fname_csv = label + ".csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname_csv, M2, m * Sub->s, Sub->s * n);
	}

	FREE_int(M2);
	FREE_OBJECT(Sub);

	if (f_v) {
		cout << "coding_theory_domain::field_reduction done" << endl;
	}
}



void coding_theory_domain::field_induction(
		std::string &fname_in,
		std::string &fname_out, int nb_bits,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::field_induction" << endl;
	}
	int i, h, len, len2;
	long int *M;
	long int a;
	long int *M2;
	int *v;
	int m, n;
	geometry::other_geometry::geometry_global GG;


	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Reading file " << fname_in << " of size "
				<< Fio.file_size(fname_in) << endl;
	}
	Fio.Csv_file_support->lint_matrix_read_csv(
			fname_in, M, m, n, verbose_level);
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
	Fio.Csv_file_support->lint_matrix_write_csv(
			fname_out, M2, 1, len2);
	if (f_v) {
		cout << "Written file " << fname_out << " of size "
				<< Fio.file_size(fname_out) << endl;
	}

	if (f_v) {
		cout << "coding_theory_domain::field_induction done" << endl;
	}
}

void coding_theory_domain::encode_text_5bits(
		std::string &text,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::encode_text_5bits" << endl;
	}
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

	other::orbiter_kernel_system::file_io Fio;

	//Fio.int_vec_write_csv(encoding, 5 * l, fname, "encoding");
	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, encoding, 1, len);
	cout << "Written file " << fname
			<< " of size " << Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "coding_theory_domain::encode_text_5bits done" << endl;
	}
}

int coding_theory_domain::Hamming_distance(
		int *v1, int *v2, int n)
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


int coding_theory_domain::Hamming_distance_binary(
		int a, int b, int n)
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

void coding_theory_domain::fixed_code(
		algebra::field_theory::finite_field *F,
		int n, int k, int *genma,
		long int *perm,
		int *&subcode_genma, int &subcode_k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::fixed_code "
				"n = " << n << endl;
	}


	long int t0, t1, dt;
	long int N;
	int *msg;
	int *word;
	geometry::other_geometry::geometry_global Gg;
	other::orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "coding_theory_domain::fixed_code" << endl;
	}
	N = Gg.nb_AG_elements(k, F->q);
	if (f_v) {
		cout << N << " messages" << endl;
	}
	msg = NEW_int(k);
	word = NEW_int(n);
	int h, i, a, j, b, cnt;
	vector<long int> V;

	cnt = 0;
	for (h = 0; h < N; h++) {
		if ((h % ONE_MILLION) == 0) {
			t1 = Os.os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			Os.time_check_delta(cout, dt);
			cout << endl;
		}
		Gg.AG_element_unrank(F->q, msg, 1, k, h);
		F->Linear_algebra->mult_vector_from_the_left(
				msg, genma, word, k, n);
		for (i = 0; i < n; i++) {
			a = word[i];
			j = perm[i];
			b = word[j];
			if (a != b) {
				break;
			}
		}
		if (i == n) {
			V.push_back(h);
			Int_vec_print(cout, word, n);
			cout << endl;
			cnt++;
		}
	}
	if (f_v) {
		cout << "coding_theory_domain::fixed_code "
				"we found " << cnt << " fixed words" << endl;
	}
	int *M;
	int rk;

	M = NEW_int(cnt * n);
	for (i = 0; i < N; i++) {
		Gg.AG_element_unrank(F->q, msg, 1, k, V[i]);
		F->Linear_algebra->mult_vector_from_the_left(
				msg, genma, word, k, n);
		Int_vec_copy(word, M + i * n, n);
	}
	rk = F->Linear_algebra->Gauss_easy(M, cnt, n);
	if (f_v) {
		cout << "coding_theory_domain::fixed_code "
				"The fix subcode has dimension " << rk << endl;
		Int_matrix_print(M, rk, n);
		cout << endl;
		Int_vec_print_fully(cout, M, rk * n);
		cout << endl;
	}

	subcode_k = rk;
	subcode_genma = NEW_int(subcode_k * n);
	Int_vec_copy(M, subcode_genma, subcode_k * n);


	FREE_int(M);

	if (f_v) {
		cout << "coding_theory_domain::fixed_code done" << endl;
	}
}



void coding_theory_domain::polynomial_representation_of_boolean_function(
		algebra::field_theory::finite_field *F,
		std::string &label_txt,
		long int *Words,
		int nb_words, int n,
		int verbose_level)
// computes the polynomial representation for the characteristic function
// of the set defined by the Words[] array of size nb_words.
// creates a combinatorics::boolean_function_domain object
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "coding_theory_domain::polynomial_representation_of_boolean_function" << endl;
	}


	combinatorics::special_functions::boolean_function_domain *BF;
	int *coeff;
	int *f;
	int *g;
	long int N;
	int h;


	N = 1 << n;

	if (f_v) {
		cout << "N=" << N << endl;
	}

	BF = NEW_OBJECT(combinatorics::special_functions::boolean_function_domain);


	if (f_v) {
		cout << "before BF->init, n=" << n << endl;
	}
	BF->init(F, n, 0 /*verbose_level*/);

	if (f_v) {
		cout << "BF->Poly[n].get_nb_monomials()="
				<< BF->Poly[n].get_nb_monomials() << endl;
	}
	coeff = NEW_int(BF->Poly[n].get_nb_monomials());

	// f is the characteristic function of the set defined by Words[nb_words]

	f = NEW_int(N);
	g = NEW_int(N);

	Int_vec_zero(f, N);
	for (h = 0; h < nb_words; h++) {
		f[Words[h]] = 1;
	}

	if (f_v) {
		cout << "computing the polynomial representation: " << endl;
	}


	BF->compute_polynomial_representation(
			f /* func */, coeff,
			0 /*verbose_level*/);
	//BF->search_for_bent_functions(verbose_level);



	if (f_v) {
		cout << "The representation as polynomial is: ";

		cout << " : ";
		BF->Poly[BF->n].print_equation(cout, coeff);
		cout << " : ";
		//evaluate_projectively(poly, f_proj);
		BF->evaluate(coeff, g);
		//int_vec_print(cout, g, BF->Q);
		cout << endl;
	}


	for (h = 0; h < BF->Q; h++) {
		if (f[h] != g[h]) {
			cout << "f[h] != g[h], h = " << h
					<< ", an error has occurred" << endl;
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

	if (f_v) {
		cout << "coding_theory_domain::polynomial_representation_of_boolean_function done" << endl;
	}
}



}}}}


