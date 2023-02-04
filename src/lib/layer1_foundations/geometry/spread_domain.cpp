/*
 * spread_domain.cpp
 *
 *  Created on: Aug 30, 2022
 *      Author: betten
 */





#include "foundations.h"


using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace geometry {

spread_domain::spread_domain()
{

	F = NULL;

	n = 0;
	k = 0;
	kn = 0;
	q = 0;

	nCkq = 0;
	nC1q = 0;
	kC1q = 0;

	qn = 0;
	qk = 0;

	order = 0;
	spread_size = 0;

	r = 0;
	nb_pts = 0;
	nb_points_total = 0;

	Grass = NULL;

	tmp_M1 = NULL;
	tmp_M2 = NULL;
	tmp_M3 = NULL;
	tmp_M4 = NULL;

	Klein = NULL;
	O = NULL;

	Data1 = NULL;
	Data2 = NULL;

}

spread_domain::~spread_domain()
{
	if (Grass) {
		FREE_OBJECT(Grass);
	}
	if (tmp_M1) {
		FREE_int(tmp_M1);
	}
	if (tmp_M2) {
		FREE_int(tmp_M2);
	}
	if (tmp_M3) {
		FREE_int(tmp_M3);
	}
	if (tmp_M4) {
		FREE_int(tmp_M4);
	}
	if (O) {
		FREE_OBJECT(O);
	}
	if (Klein) {
		FREE_OBJECT(Klein);
	}
	if (Data1) {
		FREE_int(Data1);
	}
	if (Data2) {
		FREE_int(Data2);
	}

}

void spread_domain::init(
		field_theory::finite_field *F,
		int n, int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	number_theory::number_theory_domain NT;
	combinatorics::combinatorics_domain Combi;


	if (f_v) {
		cout << "spread_domain::init" << endl;
		cout << "n=" << n << endl;
		cout << "k=" << k << endl;
		cout << "q=" << F->q << endl;
	}

	spread_domain::F = F;
	spread_domain::n = n;
	spread_domain::k = k;

	kn = k * n;
	q = F->q;

	nCkq = Combi.generalized_binomial(n, k, q);
	nC1q = Combi.generalized_binomial(n, 1, q);
	kC1q = Combi.generalized_binomial(k, 1, q);

	qn = NT.i_power_j(q, n);
	qk = NT.i_power_j(q, k);


	spread_size = (qn - 1) / (qk - 1);
		// (NT.i_power_j(q, n) - 1) / (NT.i_power_j(q, k) - 1);
	order = qk; //NT.i_power_j(q, k);

	r = kC1q; // Combi.generalized_binomial(k, 1, q);
	nb_points_total = nb_pts = nC1q;

	if (f_v) {
		cout << "spread_domain::init" << endl;
		cout << "q=" << q << endl;
		cout << "n=" << n << endl;
		cout << "k=" << k << endl;
		cout << "r=" << r << endl;
		cout << "order=" << order << endl;
		cout << "spread_size=" << spread_size << endl;
		cout << "nb_points_total=" << nb_points_total << endl;
	}

	tmp_M1 = NEW_int(n * n);
	tmp_M2 = NEW_int(n * n);
	tmp_M3 = NEW_int(n * n);
	tmp_M4 = NEW_int(n * n);

	Grass = NEW_OBJECT(geometry::grassmann);
	Grass->init(n, k, F, 0 /*MINIMUM(verbose_level - 1, 1)*/);


	if (f_v) {
		cout << "spread_domain::init "
				"nCkq = {n \\choose k}_q = " << nCkq << endl;
		cout << "spread_domain::init "
				"r = {k \\choose 1}_q = " << r << endl;
		cout << "spread_domain::init "
				"nb_pts = {n \\choose 1}_q = " << nb_pts << endl;
	}

	Data1 = NEW_int(spread_size * kn);
	Data2 = NEW_int(n * n);

	if (k == 2 && n == 4) {

		if (f_v) {
			cout << "spread_domain::init k == 2 and n == 4, "
					"initializing the Klein correspondence" << endl;
		}
		Klein = NEW_OBJECT(geometry::klein_correspondence);
		O = NEW_OBJECT(layer1_foundations::orthogonal_geometry::orthogonal);

		O->init(1 /* epsilon */, 6, F, 0 /* verbose_level*/);
		Klein->init(F, O, 0 /* verbose_level */);
	}
	else {
		if (f_v) {
			cout << "spread_domain::init we are not "
					"initializing the Klein correspondence" << endl;
		}
		O = NULL;
		Klein = NULL;
	}

	if (f_v) {
		cout << "spread_domain::init done" << endl;
	}
}


void spread_domain::unrank_point(int *v, long int a)
{
	F->Projective_space_basic->PG_element_unrank_modified_lint(
			v, 1, n, a);
}

long int spread_domain::rank_point(int *v)
{
	long int a;

	F->Projective_space_basic->PG_element_rank_modified_lint(
			v, 1, n, a);
	return a;
}

void spread_domain::unrank_subspace(int *M, long int a)
{
	Grass->unrank_lint_here(M, a, 0/*verbose_level - 4*/);
}

long int spread_domain::rank_subspace(int *M)
{
	long int a;

	a = Grass->rank_lint_here(M, 0 /*verbose_level*/);
	return a;
}

void spread_domain::print_points()
{
	int *v;
	int i;

	cout << "spread_domain::print_points" << endl;
	v = NEW_int(n);
	for (i = 0; i < nb_pts; i++) {
		unrank_point(v, i);
		cout << "point " << i << " : ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	FREE_int(v);
}

void spread_domain::print_points(long int *pts, int len)
{
	int *v;
	int h;
	long int a;

	cout << "spread_domain::print_points" << endl;
	v = NEW_int(n);
	for (h = 0; h < len; h++) {
		a = pts[h];
		unrank_point(v, a);
		cout << "point " << h << " : " << a << " : ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	FREE_int(v);
}

void spread_domain::print_elements()
{
	int i, j;
	int *M;

	M = NEW_int(kn);
	for (i = 0; i < nCkq; i++) {
		if (FALSE) {
			cout << i << ":" << endl;
		}
		unrank_subspace(M, i);
		if (FALSE) {
			Int_vec_print_integer_matrix_width(cout, M,
					k, n, n, F->log10_of_q + 1);
		}
		j = rank_subspace(M);
		if (j != i) {
			cout << "rank yields " << j << " != " << i << endl;
			exit(1);
		}
	}
	FREE_int(M);
}

void spread_domain::print_elements_and_points()
{
	int i, a, b;
	int *M, *v, *w;
	int *Line;

	cout << "spread_domain::print_elements_and_points" << endl;
	M = NEW_int(kn);
	v = NEW_int(k);
	w = NEW_int(n);
	Line = NEW_int(r);
	for (i = 0; i < nCkq; i++) {
		if (FALSE) {
			cout << i << ":" << endl;
		}
		unrank_subspace(M, i);
		for (a = 0; a < r; a++) {
			F->Projective_space_basic->PG_element_unrank_modified(
					v, 1, k, a);
			F->Linear_algebra->mult_matrix_matrix(
					v, M, w, 1, k, n,
					0 /* verbose_level */);
			b = rank_point(w);
			Line[a] = b;
		}
		cout << "line " << i << ":" << endl;
		Int_vec_print_integer_matrix_width(cout, M,
				k, n, n, F->log10_of_q + 1);
		cout << "points on subspace " << i << " : ";
		Int_vec_print(cout, Line, r);
		cout << endl;
	}
	FREE_int(M);
	FREE_int(v);
	FREE_int(w);
	FREE_int(Line);
}

void spread_domain::early_test_func(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
// for poset classification
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i0, i, j, rk;
	int *M;
	int *MM;
	int *B, *base_cols;

	if (f_v) {
		cout << "spread_domain::early_test_func checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		if (f_vv) {
			if (nb_candidates < 100) {
				for (i = 0; i < nb_candidates; i++) {
					Grass->unrank_lint(candidates[i], 0/*verbose_level - 4*/);
					cout << "candidate " << i << "="
							<< candidates[i] << ":" << endl;
					Int_vec_print_integer_matrix_width(cout,
							Grass->M, k, n, n, F->log10_of_q + 1);
				}
			}
			else {
				cout << "too many to print" << endl;
				f_vv = FALSE;
			}
		}
	}

	if (len + 1 > spread_size) {
		cout << "spread_domain::early_test_func len + 1 > spread_size" << endl;
		cout << "spread_domain::early_test_func len = " << len << endl;
		cout << "spread_domain::early_test_func spread_size = " << spread_size << endl;
		exit(1);
	}
	M = Data2; // [n * n]
	MM = Data1; // [(len + 1) * kn]
	B = tmp_M3;
	base_cols = tmp_M4;

	for (i = 0; i < len; i++) {
		unrank_subspace(MM + i * kn, S[i]);
	}
	if (f_v) {
		for (i = 0; i < len; i++) {
			cout << "p_" << i << "=" << S[i] << ":" << endl;
			Int_vec_print_integer_matrix_width(cout,
					MM + i * k * n, k, n, n, F->log10_of_q + 1);
		}
	}

	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {
		Grass->unrank_lint(candidates[j], 0/*verbose_level - 4*/);
		if (len == 0) {
			i0 = 0;
		}
		else {
			i0 = len - 1;
		}
		for (i = i0; i < len; i++) {
			Int_vec_copy(MM + i * kn, M, k * n);
			Int_vec_copy(Grass->M, M + kn, k * n);

			if (f_vv) {
				cout << "testing (p_" << i << ",candidates[" << j << "])="
						"(" << S[i] <<  "," << candidates[j] << ")" << endl;
				Int_vec_print_integer_matrix_width(cout, M,
						2 * k, n, n, F->log10_of_q + 1);
			}
			rk = F->Linear_algebra->rank_of_rectangular_matrix_memory_given(
					M, 2 * k, n, B, base_cols,
					FALSE /* f_complete */,
					0 /* verbose_level */);

			if (rk < 2 * k) {
				if (f_vv) {
					cout << "rank is " << rk << " which is bad" << endl;
				}
				break;
			}
			else {
				if (f_vv) {
					cout << "rank is " << rk << " which is OK" << endl;
				}
			}
		} // next i
		if (i == len) {
			good_candidates[nb_good_candidates++] = candidates[j];
		}
	} // next j

	if (f_v) {
		cout << "spread_domain::early_test_func we found " << nb_good_candidates
				<< " good candidates" << endl;
	}
	if (f_v) {
		cout << "spread_domain::early_test_func done" << endl;
	}
}

int spread_domain::check_function(
		int len, long int *S, int verbose_level)
// checks all {len \choose 2} pairs. This is very inefficient.
// This function should not be used for poset classification!
{
	int f_OK = TRUE;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, rk;
	int *M, *M1;
	int *B, *base_cols;

	if (f_v) {
		cout << "spread_domain::check_function checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	M1 = tmp_M1; // [kn]
	M = tmp_M2; // [n * n]
	B = tmp_M3;
	base_cols = tmp_M4;

	if (f_v) {
		for (i = 0; i < len; i++) {
			cout << "p_" << i << "=" << S[i] << ":" << endl;
			Grass->unrank_lint(S[i], 0/*verbose_level - 4*/);
			Int_vec_print_integer_matrix_width(cout, Grass->M,
					k, n, n, F->log10_of_q + 1);
		}
	}

	for (i = 0; i < len; i++) {
		unrank_subspace(M1, S[i]);
		for (j = i + 1; j < len; j++) {
			Int_vec_copy(M1, M, kn);
			unrank_subspace(M + kn, S[j]);

			if (f_vv) {
				cout << "testing (p_" << i << ",p_" << j << ")"
						"=(" << S[i] << "," << S[j] << ")" << endl;
				Int_vec_print_integer_matrix_width(cout, M,
						2 * k, n, n, F->log10_of_q + 1);
			}
			rk = F->Linear_algebra->rank_of_rectangular_matrix_memory_given(
					M, 2 * k, n, B, base_cols,
					FALSE /* f_complete */,
					0 /* verbose_level */);
			if (rk < 2 * k) {
				if (f_vv) {
					cout << "rank is " << rk << " which is bad" << endl;
				}
				f_OK = FALSE;
				break;
			}
			else {
				if (f_vv) {
					cout << "rank is " << rk << " which is OK" << endl;
				}
			}
		}
		if (f_OK == FALSE) {
			break;
		}
	}

	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
		}
		return TRUE;
	}
	else {
		if (f_v) {
			cout << "not OK" << endl;
		}
		return FALSE;
	}

}

int spread_domain::incremental_check_function(
		int len, long int *S, int verbose_level)
// checks the pairs (0,len-1),(1,len-1),\ldots,(len-2,len-1)
// for recoordinatize
{
	int f_OK = TRUE;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, rk;
	int *M, *M1;
	int *B, *base_cols;

	if (f_v) {
		cout << "spread_domain::incremental_check_function checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	if (len <= 1) {
		f_OK = TRUE;
		goto finish;
	}
	M1 = tmp_M1; // [kn]
	M = tmp_M2; // [n * n]
	B = tmp_M3;
	base_cols = tmp_M4;

	if (f_v) {
		for (i = 0; i < len; i++) {
			cout << "p_" << i << "=" << S[i] << ":" << endl;
			Grass->unrank_lint(S[i], 0/*verbose_level - 4*/);
			Int_vec_print_integer_matrix_width(cout,
					Grass->M, k, n, n, F->log10_of_q + 1);
		}
	}

	j = len - 1;

	unrank_subspace(M1, S[j]);
	for (i = 0; i < len - 1; i++) {
		unrank_subspace(M, S[i]);
		Int_vec_copy(M1, M + kn, kn);

		if (f_vv) {
			cout << "testing (p_" << i << ",p_" << j << ")"
					"=(" << S[i] <<  "," << S[j] << ")" << endl;
			Int_vec_print_integer_matrix_width(cout, M,
					2 * k, n, n, F->log10_of_q + 1);
		}
		rk = F->Linear_algebra->rank_of_rectangular_matrix_memory_given(
				M, 2 * k, n, B, base_cols,
				FALSE /* f_complete */,
				0 /* verbose_level */);
		if (rk < 2 * k) {
			if (f_vv) {
				cout << "rank is " << rk << " which is bad" << endl;
			}
			f_OK = FALSE;
			break;
		}
		else {
			if (f_vv) {
				cout << "rank is " << rk << " which is OK" << endl;
			}
		}
	}

finish:
	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
		}
		return TRUE;
	}
	else {
		if (f_v) {
			cout << "not OK" << endl;
		}
		return FALSE;
	}

}

void spread_domain::compute_dual_spread(
		int *spread,
		int *dual_spread, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_domain::compute_dual_spread" << endl;
	}

	Grass->compute_dual_spread(spread, dual_spread,
			spread_size, verbose_level - 1);

	if (f_v) {
		cout << "spread_domain::compute_dual_spread done" << endl;
	}
}

void spread_domain::print(
		std::ostream &ost, int len, long int *S)
{
	int i;
	int f_elements_exponential = FALSE;
	string symbol_for_print;

	symbol_for_print.assign("\\alpha");
	if (len == 0) {
		return;
	}
	for (i = 0; i < len; i++) {
		ost << "$S_{" << i + 1 << "}$ has rank " << S[i]
			<< " and is generated by\\\\" << endl;
		Grass->unrank_lint(S[i], 0);
		ost << "$$" << endl;
		ost << "\\left[" << endl;
		F->Io->latex_matrix(ost, f_elements_exponential, symbol_for_print,
			Grass->M, k, n);
		ost << "\\right]" << endl;
		ost << "$$" << endl << endl;
	}

}

void spread_domain::czerwinski_oakden(
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int sz = 26;
	long int data[26];
	int M[8];
	int h, u, i, a = 0, b = 0, c = 0, d = 0;
	orbiter_kernel_system::file_io Fio;
	int spreads[] =
		{
			// S1:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			3,0,0,3,
			4,0,0,4,
			4,3,3,0,
			0,3,3,1,
			2,1,1,4,
			4,4,4,2,
			2,3,3,3,
			3,4,4,1,
			1,1,1,3,
			3,2,2,2,
			0,1,1,2,
			1,2,2,0,
			0,2,2,4,
			2,4,4,0,
			0,4,4,3,
			3,1,1,0,
			1,4,4,4,
			4,2,2,3,
			3,3,3,4,
			4,1,1,1,
			1,3,3,2,
			2,2,2,1,

			//S2:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			3,0,0,3,
			4,0,0,4,
			1,2,2,0,
			0,2,2,4,
			4,3,3,0,
			0,3,3,1,
			2,1,3,3,
			4,4,4,1,
			2,3,1,3,
			3,4,2,2,
			1,1,1,4,
			3,2,4,2,
			0,1,1,2,
			2,4,4,0,
			0,4,4,3,
			3,1,1,0,
			1,4,4,4,
			4,2,2,3,
			3,3,3,4,
			4,1,1,1,
			1,3,3,2,
			2,2,2,1,

			// S3:
			0,0,0,0,
			1,0,0,1,
			1,2,2,0,
			0,2,2,4,
			4,0,0,4,
			4,3,3,0,
			0,3,3,1,
			2,1,3,3,
			4,4,4,1,
			2,3,1,3,
			3,4,2,2,
			1,1,1,4,
			3,2,4,2,
			0,1,4,0,
			2,0,4,3,
			2,4,0,3,
			0,4,1,0,
			3,0,1,2,
			3,1,0,2,
			1,4,4,4,
			4,2,2,3,
			3,3,3,4,
			4,1,1,1,
			1,3,3,2,
			2,2,2,1,

			//S4:
			0,0,0,0,
			1,0,0,1,
			1,2,2,0,
			0,2,2,4,
			4,0,0,4,
			4,3,3,0,
			0,3,3,1,
			2,1,3,3,
			4,4,4,1,
			2,3,1,3,
			3,4,2,2,
			1,1,1,4,
			3,2,4,2,
			0,1,1,2,
			2,0,0,2,
			2,4,4,0,
			0,4,4,3,
			3,0,0,3,
			3,1,1,0,
			1,4,3,4,
			4,2,1,1,
			3,3,3,2,
			4,1,2,1,
			1,3,4,4,
			2,2,2,3,

			// S5:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			3,0,0,3,
			4,0,0,4,
			0,2,1,0,
			0,3,4,0,
			1,1,3,1,
			1,4,2,1,
			4,1,3,4,
			4,4,2,4,
			1,2,1,4,
			1,3,4,4,
			2,1,2,0,
			2,2,4,3,
			2,3,3,3,
			2,4,1,3,
			3,1,4,2,
			3,2,2,2,
			3,3,1,2,
			3,4,3,0,
			4,2,1,1,
			4,3,4,1,
			0,1,2,3,
			0,4,3,2,

			// A1:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			3,0,0,3,
			4,0,0,4,
			1,1,3,1,
			4,4,2,4,
			1,4,2,1,
			4,1,3,4,
			3,2,1,3,
			2,2,1,2,
			3,3,4,3,
			2,3,4,2,
			4,2,4,4,
			1,3,1,1,
			3,4,3,0,
			2,1,2,0,
			0,4,3,3,
			0,1,2,2,
			1,2,4,0,
			4,3,1,0,
			2,4,3,2,
			3,1,2,3,
			0,2,4,1,
			0,3,1,4,

			//A2:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			3,0,0,3,
			4,0,0,4,
			2,2,1,2,
			0,3,4,0,
			0,1,3,0,
			4,1,3,4,
			2,4,2,2,
			3,4,2,3,
			3,2,3,1,
			4,3,2,1,
			3,1,4,2,
			2,1,4,1,
			4,4,1,0,
			0,4,1,1,
			1,1,1,3,
			1,4,4,4,
			2,3,3,3,
			1,3,3,2,
			0,2,2,4,
			1,2,2,0,
			4,2,4,3,
			3,3,1,4,

			//A3:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			3,0,0,3,
			4,0,0,4,
			1,3,4,1,
			2,3,4,2,
			4,2,1,4,
			2,4,4,2,
			1,4,2,1,
			2,2,3,0,
			3,2,3,1,
			3,3,2,0,
			0,1,4,4,
			4,1,4,3,
			4,4,2,3,
			0,4,2,4,
			1,1,3,2,
			1,2,1,3,
			0,2,1,2,
			2,1,1,0,
			3,1,1,1,
			3,4,4,0,
			0,3,3,4,
			4,3,3,3,

			// A4:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			3,0,0,3,
			4,0,0,4,
			2,2,1,2,
			0,3,4,0,
			4,1,3,4,
			3,4,2,3,
			3,2,3,1,
			3,1,4,2,
			2,1,4,1,
			4,4,1,0,
			1,4,4,4,
			2,3,3,3,
			0,2,2,4,
			1,2,2,0,
			4,2,4,3,
			3,3,1,4,
			2,4,2,1,
			0,1,1,3,
			4,3,2,2,
			1,1,3,0,
			1,3,1,1,
			0,4,3,2,

			// A5:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			3,0,0,3,
			4,0,0,4,
			0,1,3,0,
			1,2,1,4,
			2,3,3,3,
			3,4,1,2,
			1,3,4,4,
			4,3,4,2,
			0,4,2,4,
			3,1,3,4,
			0,2,4,0,
			3,2,4,3,
			2,1,2,2,
			1,4,3,1,
			4,1,2,1,
			1,1,2,3,
			2,2,4,1,
			0,3,1,1,
			2,4,1,3,
			4,4,1,0,
			3,3,2,0,
			4,2,3,2,

			//A6:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			3,0,0,3,
			4,0,0,4,
			4,2,1,1,
			1,3,4,4,
			3,4,2,2,
			3,2,1,0,
			0,2,1,2,
			2,2,2,3,
			3,3,3,2,
			1,1,1,4,
			2,3,3,1,
			4,3,3,3,
			0,1,3,0,
			0,4,2,0,
			0,3,4,0,
			4,4,2,4,
			1,4,2,1,
			2,4,1,3,
			3,1,4,2,
			1,2,3,4,
			2,1,4,1,
			4,1,4,3,

			//A7:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			3,0,0,3,
			4,0,0,4,
			2,2,2,3,
			1,3,3,0,
			1,4,4,3,
			2,1,1,0,
			4,2,1,1,
			4,3,4,2,
			0,4,2,4,
			3,1,3,4,
			2,3,3,3,
			1,2,2,0,
			1,1,1,3,
			2,4,4,0,
			3,4,1,2,
			0,1,4,1,
			3,3,2,1,
			0,2,3,2,
			4,4,3,1,
			4,1,2,2,
			0,3,1,4,
			3,2,4,4,

			//A8:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			3,0,0,3,
			4,0,0,4,
			1,3,4,4,
			4,3,4,2,
			4,1,2,1,
			2,1,2,4,
			0,2,4,0,
			4,4,3,4,
			2,4,1,3,
			3,3,2,0,
			3,1,3,3,
			4,2,1,4,
			1,4,1,0,
			0,3,2,3,
			1,2,2,2,
			2,3,3,1,
			3,4,4,1,
			0,1,1,2,
			3,2,1,1,
			1,1,3,0,
			0,4,3,2,
			2,2,4,3,

			// B1:
			0,0,0,0,
			1,0,0,1,
			3,0,0,3,
			2,2,4,2,
			2,4,3,2,
			0,3,1,0,
			1,2,3,4,
			2,0,1,2,
			2,3,2,1,
			4,3,0,2,
			0,4,3,1,
			0,2,4,1,
			1,3,4,4,
			4,4,2,3,
			3,4,2,4,
			3,1,1,1,
			1,1,1,3,
			0,1,2,2,
			4,0,1,4,
			3,2,3,3,
			2,1,0,4,
			4,1,4,0,
			4,2,3,0,
			3,3,4,3,
			1,4,2,0,

			// B2:
			0,0,0,0,
			1,0,0,1,
			4,0,0,4,
			3,4,3,3,
			3,3,1,3,
			4,1,2,4,
			1,4,1,2,
			0,4,4,1,
			1,2,2,3,
			2,1,0,3,
			3,1,0,2,
			3,2,2,2,
			4,2,1,4,
			2,2,1,0,
			1,1,3,1,
			2,4,3,4,
			0,2,2,1,
			2,3,2,0,
			4,4,4,0,
			2,0,4,3,
			3,0,4,2,
			0,3,1,1,
			4,3,3,2,
			0,1,3,0,
			1,3,4,4,

			// B3:
			0,0,0,0,
			1,0,0,1,
			4,0,0,4,
			4,4,3,4,
			2,2,4,2,
			4,1,2,4,
			0,4,3,0,
			0,2,4,4,
			4,2,4,0,
			1,2,2,3,
			3,1,2,0,
			2,1,3,3,
			1,3,1,4,
			1,1,3,2,
			2,4,2,1,
			0,1,1,1,
			1,4,4,3,
			4,3,1,0,
			3,2,3,1,
			2,3,0,3,
			2,0,1,3,
			3,0,1,2,
			0,3,2,2,
			3,4,0,2,
			3,3,4,1,

			// B4:
			0,0,0,0,
			1,0,0,1,
			4,0,0,4,
			3,4,3,3,
			3,1,2,3,
			1,3,1,1,
			1,2,4,1,
			0,1,1,2,
			2,3,2,0,
			2,0,3,2,
			2,4,0,2,
			1,1,3,0,
			0,4,2,1,
			2,2,1,0,
			0,3,4,2,
			4,3,1,3,
			3,3,1,4,
			4,4,4,0,
			0,2,3,4,
			2,1,4,4,
			4,2,2,2,
			3,0,4,3,
			3,2,0,3,
			1,4,2,4,
			4,1,3,1,

			// B5:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			4,3,1,4,
			1,2,4,1,
			3,3,1,3,
			1,1,2,1,
			1,4,1,2,
			2,3,3,1,
			3,0,4,4,
			4,2,0,3,
			0,4,1,1,
			1,3,3,0,
			0,3,2,2,
			2,1,1,0,
			3,4,0,4,
			4,0,3,3,
			0,2,4,2,
			2,2,4,0,
			4,4,2,0,
			0,1,3,4,
			4,1,4,3,
			3,2,2,4,
			2,4,2,3,
			3,1,3,2,

			// B6:
			0,0,0,0,
			1,0,0,1,
			4,2,2,3,
			1,2,2,0,
			3,2,2,2,
			2,2,2,4,
			3,3,3,1,
			1,4,4,2,
			2,4,1,1,
			1,1,1,0,
			3,4,0,4,
			0,1,4,4,
			2,0,4,3,
			1,3,3,0,
			3,1,4,1,
			4,0,3,4,
			4,1,0,3,
			0,4,1,3,
			0,2,2,1,
			2,1,3,2,
			4,4,4,0,
			2,3,0,2,
			0,3,3,3,
			3,0,1,2,
			4,3,1,4,

			//B7:
			0,0,0,0,
			1,0,0,1,
			2,2,4,2,
			3,2,4,3,
			1,2,4,1,
			3,3,1,3,
			2,4,3,2,
			1,1,1,0,
			0,3,2,1,
			3,1,0,2,
			2,0,2,3,
			4,1,0,3,
			3,0,2,4,
			4,4,4,0,
			0,2,3,4,
			4,3,3,0,
			0,4,1,4,
			3,4,3,1,
			0,1,4,4,
			1,4,3,3,
			4,2,2,0,
			2,3,0,4,
			4,0,1,2,
			1,3,2,2,
			2,1,1,1,

			// B8:
			0,0,0,0,
			1,0,0,1,
			2,0,0,2,
			3,0,0,3,
			4,2,2,0,
			0,2,2,3,
			4,3,3,0,
			0,4,4,3,
			4,4,4,2,
			4,0,1,4,
			4,1,0,4,
			3,2,4,4,
			3,4,2,4,
			0,1,1,2,
			2,3,1,1,
			2,1,3,1,
			1,4,1,0,
			1,1,4,0,
			3,1,3,3,
			2,2,2,1,
			3,3,1,3,
			2,4,4,1,
			1,2,3,2,
			1,3,2,2,
			0,3,3,4,

		};

	if (f_v) {
		cout << "spread_classify::czerwinski_oakden" << endl;
		}

	const char *label[] = {
		"S1",
		"S2",
		"S3",
		"S4",
		"S5",
		"A1",
		"A2",
		"A3",
		"A4",
		"A5",
		"A6",
		"A7",
		"A8",
		"B1",
		"B2",
		"B3",
		"B4",
		"B5",
		"B6",
		"B7",
		"B8",
		};
	char fname[] = "Czerwinski_Oakden.txt";
	string fname2;
	{
	ofstream fp(fname);

	for (h = 0; h < 21; h++) {
		for (u = 0; u < sz; u++) {
			for (i = 0; i < 8; i++) {
				M[i] = 0;
				}
			if (u == 0) {
				M[0 * 4 + 2] = 1;
				M[1 * 4 + 3] = 1;
				}
			else {
				M[0 * 4 + 0] = 1;
				M[1 * 4 + 1] = 1;
				a = spreads[h * 25 * 4 + (u - 1) * 4 + 0];
				b = spreads[h * 25 * 4 + (u - 1) * 4 + 1];
				c = spreads[h * 25 * 4 + (u - 1) * 4 + 2];
				d = spreads[h * 25 * 4 + (u - 1) * 4 + 3];
				M[0 * 4 + 2] = a;
				M[0 * 4 + 3] = b;
				M[1 * 4 + 2] = c;
				M[1 * 4 + 3] = d;
				}
			for (i = 0; i < 8; i++) {
				Grass->M[i] = M[i];
				}
			if (f_vv) {
				cout << "spread " << h << ", element " << u << ":" << endl;
				if (u) {
					cout << "a=" << a << " b=" << b
							<< " c=" << c << " d=" << d << endl;
					}
				}
			Int_matrix_print(Grass->M, 2, 4);
			data[u] = Grass->rank_lint(0);

			} // next u

		cout << "spread " << h << ":";
		Lint_vec_print(cout, data, sz);
		cout << endl;

		fp << "0 "; // a dummy
		for (i = 0; i < sz; i++) {
			fp << data[i] << " ";
			}
		fp << endl;


		fname2.assign("Czerwinski_Oakden_");
		fname2.append(label[h]);
		fname2.append(".txt");

		Fio.write_set_to_file(fname2, data, sz, 0/*verbose_level*/);
		cout << "Written file " << fname2 << " of size "
				<< Fio.file_size(fname2) << endl;
		} // next h
	fp << -1 << endl;
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
}

void spread_domain::write_spread_to_file(
		int type_of_spread, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *data;
	int sz = order + 1;
	char str[1000];

	if (f_v) {
		cout << "spread_classify::write_spread_to_file" << endl;
		}
	if (type_of_spread == SPREAD_OF_TYPE_FTWKB) {
		snprintf(str, sizeof(str), "spread_q%d_FTW.txt", q);
		}
	else if (type_of_spread == SPREAD_OF_TYPE_KANTOR) {
		snprintf(str, sizeof(str), "spread_q%d_Kantor.txt", q);
		}
	else if (type_of_spread == SPREAD_OF_TYPE_KANTOR2) {
		snprintf(str, sizeof(str), "spread_q%d_Kantor2.txt", q);
		}
	else if (type_of_spread == SPREAD_OF_TYPE_GANLEY) {
		snprintf(str, sizeof(str), "spread_q%d_Ganley.txt", q);
		}
	else if (type_of_spread == SPREAD_OF_TYPE_LAW_PENTTILA) {
		snprintf(str, sizeof(str), "spread_q%d_Law_Penttila.txt", q);
		}
	else if (type_of_spread == SPREAD_OF_TYPE_DICKSON_KANTOR) {
		snprintf(str, sizeof(str), "spread_q%d_DicksonKantor.txt", q);
		}
	else if (type_of_spread == SPREAD_OF_TYPE_HUDSON) {
		snprintf(str, sizeof(str), "spread_q%d_Hudson.txt", q);
		}

	string fname;

	fname.assign(str);

	orbiter_kernel_system::file_io Fio;

	data = NEW_lint(sz);
	if (type_of_spread == SPREAD_OF_TYPE_DICKSON_KANTOR ||
		type_of_spread == SPREAD_OF_TYPE_HUDSON) {
		make_spread(data, type_of_spread, verbose_level);
	}
	else {
		make_spread_from_q_clan(data, type_of_spread, verbose_level);
	}
	Fio.write_set_to_file(fname, data, sz, 0/*verbose_level*/);
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	FREE_lint(data);
}

void spread_domain::make_spread(
		long int *data,
		int type_of_spread,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int sz = order + 1;
	int M[8];
	int h, i, h1, s, t, sq, tq, x, y, w, z, eta, exponent;
	number_theory::number_theory_domain NT;
	int q1 = NT.i_power_j(F->p, (F->e >> 1));

	if (f_v) {
		cout << "spread_domain::make_spread q=" << q << " q1=" << q1 << endl;
	}
	if (n != 4) {
		cout << "spread_domain::make_spread n != 4" << endl;
		exit(1);
	}
	if (EVEN(q)) {
		cout << "spread_domain::make_spread need q odd" << endl;
		exit(1);
	}
	if (k != 2) {
		cout << "spread_domain::make_spread k != 2" << endl;
		exit(1);
	}
	for (eta = q1; eta < q; eta++) {
		if (F->negate(F->power(eta, q1)) == eta) {
			if (f_v) {
				cout << "spread_domain::make_spread eta=" << eta << endl;
			}
			break;
		}
	}
	exponent = (F->q + 2 * q1 - 1) >> 1;
	for (h = 0; h < sz; h++) {
		for (i = 0; i < 8; i++) {
			M[i] = 0;
		}
		if (h == 0) {
			M[0 * 4 + 2] = 1;
			M[1 * 4 + 3] = 1;
		}
		else {
			M[0 * 4 + 0] = 1;
			M[1 * 4 + 1] = 1;
			h1 = h - 1;
			s = h1 % q;
			t = (h1 - s) / q;
			x = s;
			y = t;
			sq = F->power(s, q1);
			tq = F->power(t, q1);
			if (type_of_spread == SPREAD_OF_TYPE_DICKSON_KANTOR) {
				w = F->add(sq, tq);
			}
			else {
				w = F->add(sq, F->power(t, exponent));
			}
			z = F->add(F->mult(eta, sq), tq);
			M[0 * 4 + 2] = x;
			M[0 * 4 + 3] = y;
			M[1 * 4 + 2] = w;
			M[1 * 4 + 3] = z;
		}
		for (i = 0; i < 8; i++) {
			Grass->M[i] = M[i];
		}
		if (f_vv) {
			cout << "spread element " << h << ":" << endl;
			Int_matrix_print(Grass->M, 2, 4);
		}
		data[h] = Grass->rank_lint(0);
	} // next h
	if (check_function(sz, data, verbose_level - 2)) {
		if (f_v) {
			cout << "spread_domain::make_spread The set is a spread" << endl;
		}
	}
	else {
		cout << "spread_domain::make_spread The set is NOT a spread" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "spread_domain::make_spread done" << endl;
	}
}


void spread_domain::make_spread_from_q_clan(
		long int *data,
		int type_of_spread, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int sz = order + 1;
	int M[8];
	int h, h1, i, s, t, t2, t3, t4, t5, t7, t9;
	int a_t = 0, b_t = 0, c_t = 0, x, y, w, z, r;
	int three, five, nonsquare = 0, minus_nonsquare = 0, nonsquare_inv = 0;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "spread_domain::make_spread_from_q_clan" << endl;
	}
	if (n != 4) {
		cout << "spread_domain::make_spread_from_q_clan n != 4" << endl;
		exit(1);
	}
	if (k != 2) {
		cout << "spread_domain::make_spread_from_q_clan k != 2" << endl;
		exit(1);
	}
	three = F->add3(1, 1, 1);
	five = F->add3(three, 1, 1);
	if (type_of_spread == SPREAD_OF_TYPE_KANTOR ||
		type_of_spread == SPREAD_OF_TYPE_GANLEY ||
		type_of_spread == SPREAD_OF_TYPE_LAW_PENTTILA) {
		nonsquare = F->alpha_power(1);
		nonsquare_inv = F->inverse(nonsquare);
		minus_nonsquare = F->negate(nonsquare);
	}
	for (h = 0; h < sz; h++) {
		for (i = 0; i < 8; i++) {
			M[i] = 0;
		}
		if (h == 0) {
			M[0 * 4 + 2] = 1;
			M[1 * 4 + 3] = 1;
		}
		else {
			M[0 * 4 + 0] = 1;
			M[1 * 4 + 1] = 1;
			h1 = h - 1;
			s = h1 % q;
			t = (h1 - s) / q;

			// create the q-clan:

			if (type_of_spread == SPREAD_OF_TYPE_FTWKB) {
				// Fisher Thas Walker Betten:
				// a_t = t, b_t = 3t^2, c_t = 3t^3
				// only when q \equiv 2 mod 3
				r = q % 3;
				if (r != 2) {
					cout << "FTWKB needs q equiv 2 mod 3" << endl;
					exit(1);
				}
				a_t = t;
				b_t = F->product3(three, t, t);
				c_t = F->mult(b_t, t);
			}
			else if (type_of_spread == SPREAD_OF_TYPE_KANTOR) {
				if (EVEN(q)) {
					cout << "KANTOR needs q to be odd" << endl;
					exit(1);
				}
				if (NT.is_prime(q)) {
					cout << "KANTOR needs q to be a prime power" << endl;
					exit(1);
				}
				a_t = t;
				b_t = 0;
				c_t = F->mult(minus_nonsquare, F->frobenius_power(t, 1));
			}
			else if (type_of_spread == SPREAD_OF_TYPE_KANTOR2) {
				if (EVEN(q)) {
					cout << "KANTOR2 needs q to be odd" << endl;
					exit(1);
				}
				if ((q % 5) != 2 && (q % 5) != 3) {
					cout << "KANTOR needs q congruent 2 or 3 mod 5" << endl;
					exit(1);
				}
				t3 = F->product3(t, t, t);
				t5 = F->product3(t3, t, t);
				a_t = t;
				b_t = F->mult(five, t3);
				c_t = F->mult(five, t5);
			}
			else if (type_of_spread == SPREAD_OF_TYPE_GANLEY) {
				int tmp1, tmp2;
				if (q % 3) {
					cout << "GANLEY, q needs to be "
							"a power of three" << endl;
					exit(1);
				}
				t3 = F->product3(t, t, t);
				t9 = F->product3(t3, t3, t3);
				a_t = t;
				b_t = t3;
				tmp1 = F->mult(nonsquare, t);
				tmp2 = F->mult(nonsquare_inv, t9);
				c_t = F->negate(F->add(tmp1, tmp2));
			}
			else if (type_of_spread == SPREAD_OF_TYPE_LAW_PENTTILA) {
				int tmp1, tmp2;
				int n2, n3;
				if (q % 3) {
					cout << "LAW_PENTTILA, q needs to "
							"be a power of three" << endl;
					exit(1);
				}
				t2 = F->mult(t, t);
				t3 = F->product3(t, t, t);
				t4 = F->mult(t2, t2);
				t7 = F->mult(t4, t3);
				t9 = F->product3(t3, t3, t3);
				n2 = F->mult(nonsquare, nonsquare);
				n3 = F->product3(nonsquare, nonsquare, nonsquare);
				a_t = t;
				b_t = F->negate(F->add(t4, F->mult(nonsquare, t2)));
				tmp1 = F->add(t7, F->mult(n2, t3));
				tmp2 = F->negate(F->add(F->mult(nonsquare_inv, t9),
						F->mult(n3, t)));
				c_t = F->add(tmp1, tmp2);
				if (h == 34) {
					cout << "s=" << s << endl;
					cout << "t=" << t << endl;
					cout << "n=" << nonsquare << endl;
					cout << "t2=" << t2 << endl;
					cout << "t3=" << t3 << endl;
					cout << "t4=" << t4 << endl;
					cout << "t7=" << t7 << endl;
					cout << "t9=" << t9 << endl;
					cout << "a_t=" << a_t << endl;
					cout << "b_t=" << b_t << endl;
					cout << "c_t=" << c_t << endl;
				}
			}


			// create the spread element
			// according to Gevaert-Johnson 1988:

			x = a_t;
			y = F->add(b_t, s);
			w = F->negate(s);
			z = c_t;
			M[0 * 4 + 2] = x;
			M[0 * 4 + 3] = y;
			M[1 * 4 + 2] = w;
			M[1 * 4 + 3] = z;
		}
		for (i = 0; i < 8; i++) {
			Grass->M[i] = M[i];
		}
		if (f_vv) {
			cout << "spread element " << h << ":" << endl;
			Int_matrix_print(Grass->M, 2, 4);
		}
		data[h] = Grass->rank_lint(0);
	}
	if (check_function(sz, data, verbose_level - 2)) {
		cout << "The set is a spread" << endl;
	}
	else {
		cout << "The set is NOT a spread" << endl;
		exit(1);
	}
}

void spread_domain::read_and_print_spread(
		std::string &fname, int verbose_level)
{
	long int *data;
	int sz;
	orbiter_kernel_system::file_io Fio;

	Fio.read_set_from_file(fname, data, sz, verbose_level);
	print_spread(cout, data, sz);
	FREE_lint(data);
}

void spread_domain::HMO(
		std::string &fname, int verbose_level)
// allocates a finite_field and a subfield_structure and a grassmann
// Lifts a spread from F_q to F_{q^2}
{
	int f_v = (verbose_level >= 1);
	long int *data;
	int sz, i, h, h1;
	int *G, *H;
	int *Ge, *He;
	int *GG, *HH;
	int alpha, beta, omega, x, y, tmp1, tmp2, f, z;
	int M[8];
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "spread_domain::HMO" << endl;
	}

	if (order != q * q) {
		cout << "spread_domain::HMO order != q * q" << endl;
		exit(1);
	}
	Fio.read_set_from_file(fname, data, sz, verbose_level);
	G = NEW_int(order);
	H = NEW_int(order);
	Ge = NEW_int(order);
	He = NEW_int(order);
	print_spread(cout, data, sz);


	if (f_v) {
		cout << "spread_domain::HMO before Grass->get_spread_matrices" << endl;
	}
	Grass->get_spread_matrices(G, H, data, verbose_level);
	if (f_v) {
		cout << "spread_domain::HMO after Grass->get_spread_matrices" << endl;
	}


	int q2;
	field_theory::finite_field *Fq2;
	field_theory::subfield_structure *Sub;

	q2 = q * q;

	Fq2 = NEW_OBJECT(field_theory::finite_field);

	Sub = NEW_OBJECT(field_theory::subfield_structure);

	if (f_v) {
		cout << "spread_domain::HMO before Fq2->finite_field_init_small_order" << endl;
	}
	Fq2->finite_field_init_small_order(q2,
			FALSE /* f_without_tables */,
			FALSE /* f_compute_related_fields */,
			verbose_level);
	if (f_v) {
		cout << "spread_domain::HMO after Fq2->finite_field_init_small_order" << endl;
	}

	Sub->init(Fq2, F, verbose_level);

	for (i = 0; i < q * q; i++) {
		Ge[i] = Sub->FQ_embedding[G[i]];
		He[i] = Sub->FQ_embedding[H[i]];
	}
	if (f_v) {
		cout << "spread_domain::HMO after embedding" << endl;
		cout << "Ge:" << endl;
		Int_matrix_print(Ge, q, q);
		cout << "He:" << endl;
		Int_matrix_print(He, q, q);
	}

	GG = NEW_int(q2 * q2);
	HH = NEW_int(q2 * q2);
	omega = Sub->Basis[1];
	if (f_v) {
		cout << "spread_domain::HMO omega=" << omega << endl;
	}
	for (alpha = 0; alpha < q2; alpha++) {
		for (beta = 0; beta < q2; beta++) {
			x = Sub->components[beta * 2 + 0];
			y = Sub->components[beta * 2 + 1];
			if (f_v) {
				cout << "spread_domain::HMO alpha=" << alpha << " beta=" << beta
						<< " x=" << x << " y=" << y << endl;
			}
			tmp1 = Ge[x * q + y];
			tmp2 = Fq2->negate(Fq2->mult(He[x * q + y], omega));
			f = Fq2->add(tmp1, tmp2);
			z = Fq2->frobenius_power(alpha, Fq2->e / 2);
			GG[alpha * q2 + beta] = f;
			HH[alpha * q2 + beta] = z;
		}
	}
	if (f_v) {
		cout << "spread_domain::HMO GG:" << endl;
		Int_matrix_print(GG, q2, q2);
		cout << "spread_domain::HMO HH:" << endl;
		Int_matrix_print(HH, q2, q2);
	}

	geometry::grassmann *Gq2;
	long int *Data2;
	int Sz;

	Gq2 = NEW_OBJECT(geometry::grassmann);
	Gq2->init(n, k, Fq2, verbose_level);

	Sz = q2 * q2 + 1;
	Data2 = NEW_lint(Sz);

	for (h = 0; h < Sz; h++) {
		for (i = 0; i < 8; i++) {
			M[i] = 0;
		}
		if (h == 1) {
			M[0 * 4 + 2] = 1;
			M[1 * 4 + 3] = 1;
		}
		else {
			if (h > 1) {
				h1 = h - 1;
			}
			else {
				h1 = 0;
			}
			M[0 * 4 + 0] = 1;
			M[1 * 4 + 1] = 1;
			y = h1 % q2;
			x = (h1 - y) / q2;
			if (f_v) {
				cout << "spread_domain::HMO h=" << h << " x=" << x << " y=" << y << endl;
			}
			M[0 * 4 + 2] = x;
			M[0 * 4 + 3] = y;
			M[1 * 4 + 2] = GG[x * q2 + y];
			M[1 * 4 + 3] = HH[x * q2 + y];
		}
		if (f_v) {
			cout << "spread_domain::HMO element " << h << ":" << endl;
			Int_matrix_print(M, 2, 4);
		}
#if 0
		for (i = 0; i < 8; i++) {
			Gq2->M[i] = M[i];
		}
#endif
		Data2[h] = Gq2->rank_lint_here(M, 0);
		if (f_v) {
			cout << "spread_domain::HMO has rank " << Data2[h] << endl;
		}
	}

	string fname2;

	fname2.assign("HMO_");
	fname2.append(fname);

	Fio.write_set_to_file(fname2, Data2, Sz, verbose_level);
	if (f_v) {
		cout << "spread_domain::HMO written file " << fname2
				<< " of size " << Fio.file_size(fname2) << endl;
	}

	FREE_lint(Data2);
	FREE_OBJECT(Gq2);
	FREE_OBJECT(Fq2);
	FREE_OBJECT(Sub);
	FREE_lint(data);
	FREE_int(G);
	FREE_int(H);
	FREE_int(Ge);
	FREE_int(He);
	FREE_int(GG);
	FREE_int(HH);
	if (f_v) {
		cout << "spread_domain::HMO done" << endl;
	}
}


void spread_domain::print_spread(
		std::ostream &ost, long int *data, int sz)
{
	//int sz = order + 1;
	int h;

	for (h = 0; h < sz; h++) {
		Grass->unrank_lint(data[h], 0);
		ost << "Spread element " << h << ":" << endl;
		Int_matrix_print_ost(ost, Grass->M, k, n);
	}
}




}}}


