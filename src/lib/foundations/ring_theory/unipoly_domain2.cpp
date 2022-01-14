/*
 * unipoly_domain2.cpp
 *
 *  Created on: Feb 17, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


void unipoly_domain::mult_easy(unipoly_object a,
		unipoly_object b, unipoly_object &c)
{
	int *ra = (int *) a;
	int *rb = (int *) b;
	int m = ra[0]; // degree of a
	int n = rb[0]; // degree of b
	int mn = m + n; // degree of a * b

	int *rc = (int *) c;
	FREE_int(rc);
	rc = NEW_int(mn + 2);
		// +1 since the number of coeffs is one more than the degree,
		// +1 since we allocate one unit for the degree itself

	int *A = ra + 1;
	int *B = rb + 1;
	int *C = rc + 1;
	int i, j, k, x, y;

	rc[0] = mn;
	for (i = 0; i <= mn; i++) {
		C[i] = 0;
	}
	for (i = m; i >= 0; i--) {
		for (j = n; j >= 0; j--) {
			k = i + j;
			x = C[k];
			y = F->mult(A[i], B[j]);
			if (x == 0) {
				C[k] = y;
			}
			else {
				C[k] = F->add(x, y);
			}
		}
	}
	c = (void *) rc;
}

void unipoly_domain::print_coeffs_top_down_assuming_one_character_per_digit(unipoly_object a, std::ostream &ost)
{
	int *ra = (int *) a;
	int m = ra[0]; // degree of a
	int *A = ra + 1;
	int i;

	for (i = m; i >= 0; i--) {
		ost << A[i];
	}
}

void unipoly_domain::print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(unipoly_object a, int m, std::ostream &ost)
{
	int *ra = (int *) a;
	//int m = ra[0]; // degree of a
	int *A = ra + 1;
	int i;
	int d = ra[0];

	for (i = m; i >= 0; i--) {
		if (i > d) {
			ost << "0";
		}
		else {
			ost << A[i];
		}
	}
}

void unipoly_domain::mult_easy_with_report(long int rk_a, long int rk_b, long int &rk_c,
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "unipoly_domain::mult_easy_with_report" << endl;
	}
	unipoly_object a;
	unipoly_object b;
	unipoly_object c;
	algorithms Algo;

	if (f_v) {
		cout << "unipoly_domain::mult_easy_with_report rk_a=" << rk_a << endl;
		cout << "unipoly_domain::mult_easy_with_report rk_b=" << rk_b << endl;
	}
	create_object_by_rank(a, rk_a,
				__FILE__, __LINE__, verbose_level);
	create_object_by_rank(b, rk_b,
				__FILE__, __LINE__, 0 /* verbose_level */);

	if (f_v) {
		cout << "unipoly_domain::mult_easy_with_report after create_object_by_rank" << endl;
	}

	int *ra = (int *) a;
	int *rb = (int *) b;
	int m = ra[0]; // degree of a
	int n = rb[0]; // degree of b
	int mn = m + n; // degree of a * b

	int *rc; // = (int *) c;
	//FREE_int(rc);
	if (f_v) {
		cout << "unipoly_domain::mult_easy_with_report before NEW_int, mn=" << mn << endl;
	}
	rc = NEW_int(mn + 2);
		// +1 since the number of coeffs is one more than the degree,
		// +1 since we allocate one unit for the degree itself

	int *A = ra + 1;
	int *B = rb + 1;
	int *C = rc + 1;
	int i, j, k, x, y;

	if (f_v) {
		cout << "unipoly_domain::mult_easy_with_report before rc[0] = mn;" << endl;
	}
	rc[0] = mn;
	for (i = 0; i <= mn; i++) {
		C[i] = 0;
	}

	ost << "\\begin{verbatim}" << endl;
	ost << setw(m + 1) << rk_a << " x " << setw(n + 1) << rk_b << " = " << endl;
	if (f_v) {
		cout << "unipoly_domain::mult_easy_with_report before print_coeffs_top_down_assuming_one_character_per_digit" << endl;
	}
	print_coeffs_top_down_assuming_one_character_per_digit(a, ost);
	ost << " x ";
	print_coeffs_top_down_assuming_one_character_per_digit(b, ost);
	ost << endl;
	Algo.print_repeated_character(ost, '=', m + 1 + 3 + n + 1);
	ost << endl;
	for (j = n; j >= 0; j--) {
		if (f_v) {
			cout << "unipoly_domain::mult_easy_with_report j=" << j << endl;
		}
		if (B[j] == 0) {
			continue;
		}
		Algo.print_repeated_character(ost, ' ', 4 + n - j);
		print_coeffs_top_down_assuming_one_character_per_digit(a, ost);
		ost << endl;
	}
	Algo.print_repeated_character(ost, ' ', 4);
	Algo.print_repeated_character(ost, '=', m + 1 + n);
	ost << endl;

	if (f_v) {
		cout << "unipoly_domain::mult_easy_with_report multiplying" << endl;
	}
	for (i = m; i >= 0; i--) {
		if (f_v) {
			cout << "unipoly_domain::mult_easy_with_report i=" << i << endl;
		}
		for (j = n; j >= 0; j--) {
			if (f_v) {
				cout << "unipoly_domain::mult_easy_with_report j=" << j << endl;
			}
			k = i + j;
			x = C[k];
			y = F->mult(A[i], B[j]);
			if (x == 0) {
				C[k] = y;
			}
			else {
				C[k] = F->add(x, y);
			}
		}
	}
	c = (void *) rc;
	Algo.print_repeated_character(ost, ' ', 4);
	print_coeffs_top_down_assuming_one_character_per_digit(c, ost);
	ost << endl;
	rk_c = rank(c);
	if (f_v) {
		cout << "unipoly_domain::mult_easy_with_report rk_c=" << rk_c << endl;
	}
	Algo.print_repeated_character(ost, ' ', 3);
	ost << "=" << setw(mn + 1) << rk_c << endl;
	ost << endl;
	ost << "\\end{verbatim}" << endl;

	delete_object(a);
	delete_object(b);
	delete_object(c);
	if (f_v) {
		cout << "unipoly_domain::mult_easy_with_report done" << endl;
	}

}

void unipoly_domain::division_with_remainder_from_file_with_report(
		std::string &input_fname, long int rk_b,
		long int &rk_q, long int &rk_r, std::ostream &ost, int verbose_level)
	//unipoly_object a, unipoly_object b,
	//unipoly_object &q, unipoly_object &r,
	//int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "unipoly_domain::division_with_remainder_from_file_with_report" << endl;
	}

	unipoly_object a;
	unipoly_object b;
	unipoly_object q;
	unipoly_object r;
	algorithms Algo;


	create_object_from_csv_file(
			a, input_fname,
			__FILE__, __LINE__,
			verbose_level);
	create_object_by_rank(b, rk_b,
				__FILE__, __LINE__, 0 /* verbose_level */);
	create_object_by_rank(q, 0,
				__FILE__, __LINE__, 0 /* verbose_level */);
	create_object_by_rank(r, 0,
				__FILE__, __LINE__, 0 /* verbose_level */);


	int da, db;
	int i;

	da = degree(a);
	db = degree(b);

	ost << "\\begin{verbatim}" << endl;
	Algo.print_repeated_character(ost, ' ', db + 1 + 3);
	ost << input_fname << " / " << setw(db + 1) << rk_b << " = " << endl;


	division_with_remainder_with_report(a, b, q, r, TRUE, ost, verbose_level);

	int *rr = (int *) r;
	int *R = rr + 1;
	for (i = db - 1; i >= 0; i--) {
		if (R[i]) {
			break;
		}
	}
	rk_q = rank(q);
	rk_r = rank(r);
	Algo.print_repeated_character(ost, ' ', db + 1 + 3);
	Algo.print_repeated_character(ost, ' ', da - i - 2);
	ost << "= " << setw(i + 1) << rk_r << endl;
	ost << endl;
	ost << "\\end{verbatim}" << endl;

	if (f_v) {
		cout << "unipoly_domain::division_with_remainder_from_file_with_report done" << endl;
	}

}


void unipoly_domain::division_with_remainder_from_file_all_k_bit_error_patterns(
		std::string &input_fname, long int rk_b, int k,
		long int *&rk_q, long int *&rk_r, int &n, int &N, std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "unipoly_domain::division_with_remainder_from_file_all_k_bit_error_patterns" << endl;
	}
	combinatorics_domain Combi;


	unipoly_object a;
	unipoly_object b;
	unipoly_object q;
	unipoly_object r;


	create_object_from_csv_file(
			a, input_fname,
			__FILE__, __LINE__,
			verbose_level);
	create_object_by_rank(b, rk_b,
				__FILE__, __LINE__, 0 /* verbose_level */);
	create_object_by_rank(q, 0,
				__FILE__, __LINE__, 0 /* verbose_level */);
	create_object_by_rank(r, 0,
				__FILE__, __LINE__, 0 /* verbose_level */);


	int da, db;
	int *aa = (int *) a;
	int i, h;

	da = aa[0]; //degree(a);
	db = degree(b);

	n = da + 1;
	N = Combi.int_n_choose_k(n, k);

	rk_q = NEW_lint(N);
	rk_r = NEW_lint(N);

	int *set;
	int j, u;

	set = NEW_int(k);


	ost << "\\begin{verbatim}" << endl;
	ost << "      : ";
	print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(a, n - 1, ost);
	ost << endl;

	for (h = 0; h < N; h++) {
		Orbiter->Int_vec.zero(set, k);
		Combi.unrank_k_subset(h, set, n, k);


		create_object_from_csv_file(
				a, input_fname,
				__FILE__, __LINE__,
				verbose_level);
		create_object_by_rank(b, rk_b,
					__FILE__, __LINE__, 0 /* verbose_level */);
		create_object_by_rank(q, 0,
					__FILE__, __LINE__, 0 /* verbose_level */);
		create_object_by_rank(r, 0,
					__FILE__, __LINE__, 0 /* verbose_level */);

		int *aa = (int *) a;
		int *A = aa + 1;

		for (j = 0; j < k; j++) {
			u = set[j];
			if (A[u]) {
				A[u] = 0;
			}
			else {
				A[u] = 1;
			}
		}

		ost << setw(5) << h;
		ost << " : ";
		print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(a, n - 1, ost);

		cout << setw(5) << h;
		cout << " : ";
		print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(a, n - 1, cout);
		cout << endl;


		division_with_remainder_with_report(a, b, q, r, FALSE, ost, verbose_level);

		ost << " : ";
		print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(r, db - 1, ost);

		int *rr = (int *) r;
		int *R = rr + 1;
		for (i = db - 1; i >= 0; i--) {
			if (R[i]) {
				break;
			}
		}
		rk_q[h] = rank(q);
		rk_r[h] = rank(r);
		ost << " : ";
		ost << setw(5) << rk_r[h];
		ost << " : ";
		print_object(r, ost);
		ost << endl;

	}
	ost << "\\end{verbatim}" << endl;
	FREE_int(set);

	if (f_v) {
		cout << "unipoly_domain::division_with_remainder_from_file_all_k_bit_error_patterns done" << endl;
	}

}


void unipoly_domain::division_with_remainder_numerically_with_report(long int rk_a, long int rk_b,
		long int &rk_q, long int &rk_r, std::ostream &ost, int verbose_level)
	//unipoly_object a, unipoly_object b,
	//unipoly_object &q, unipoly_object &r,
	//int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "unipoly_domain::division_with_remainder_numerically_with_report" << endl;
	}

	unipoly_object a;
	unipoly_object b;
	unipoly_object q;
	unipoly_object r;
	algorithms Algo;

	create_object_by_rank(a, rk_a,
				__FILE__, __LINE__, 0 /* verbose_level */);
	create_object_by_rank(b, rk_b,
				__FILE__, __LINE__, 0 /* verbose_level */);
	create_object_by_rank(q, 0,
				__FILE__, __LINE__, 0 /* verbose_level */);
	create_object_by_rank(r, 0,
				__FILE__, __LINE__, 0 /* verbose_level */);

	int da, db;
	int i;

	da = degree(a);
	db = degree(b);

	ost << "\\begin{verbatim}" << endl;
	Algo.print_repeated_character(ost, ' ', db + 1 + 3);
	ost << setw(da + 1) << rk_a << " / " << setw(db + 1) << rk_b << " = " << endl;



	division_with_remainder_with_report(a, b, q, r, TRUE, ost, verbose_level);


	int *rr = (int *) r;
	int *R = rr + 1;
	for (i = db - 1; i >= 0; i--) {
		if (R[i]) {
			break;
		}
	}
	rk_q = rank(q);
	rk_r = rank(r);
	Algo.print_repeated_character(ost, ' ', db + 1 + 3);
	Algo.print_repeated_character(ost, ' ', da - i - 2);
	ost << "= " << setw(i + 1) << rk_r << endl;
	ost << endl;
	ost << "\\end{verbatim}" << endl;
}


void unipoly_domain::division_with_remainder_with_report(unipoly_object &a, unipoly_object &b,
		unipoly_object &q, unipoly_object &r, int f_report, std::ostream &ost, int verbose_level)
	//unipoly_object a, unipoly_object b,
	//unipoly_object &q, unipoly_object &r,
	//int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "unipoly_domain::division_with_remainder_with_report" << endl;
	}



	//int *ra = (int *) a;
	int *rb = (int *) b;
	//int *A = ra + 1;
	int *B = rb + 1;

	int da, db;

	da = degree(a);
	db = degree(b);

	if (f_factorring) {
		cout << "unipoly_domain::division_with_remainder_with_report "
				"not good for a factorring" << endl;
		exit(1);
	}
	if (db == 0) {
		if (B[0] == 0) {
			cout << "unipoly_domain::division_with_remainder_with_report: "
					"division by zero" << endl;
			exit(1);
		}
	}
	if (db > da) {
		cout << "unipoly_domain::division_with_remainder_with_report db > da" << endl;
		int *rq = (int *) q;
		FREE_int(rq);
		rq = NEW_int(2);
		int *Q = rq + 1;
		Q[0] = 0;
		rq[0] = 0;
		assign(a, r, 0 /*verbose_level*/);
		q = rq;
		goto done;
	}

	{
		int dq = da - db;
		int *rq = (int *) q;
		FREE_int(rq);
		rq = NEW_int(dq + 2);
		rq[0] = dq;

		assign(a, r, 0 /*verbose_level*/);

		int *rr = (int *) r;

		int *Q = rq + 1;
		int *R = rr + 1;

		int i, j, ii, jj, pivot, pivot_inv, x, c, d;

		pivot = B[db];
		pivot_inv = F->inverse(pivot);

		Orbiter->Int_vec.zero(Q, dq + 1);
		algorithms Algo;

		if (f_report) {
			Algo.print_repeated_character(ost, ' ', db + 1 + 3);
			print_coeffs_top_down_assuming_one_character_per_digit(a, ost);
			ost << " / ";
			print_coeffs_top_down_assuming_one_character_per_digit(b, ost);
			ost << " = " << endl << endl;

		}
		// quotient should be here, so we need to do the computation


		for (i = da, j = dq; i >= db; i--, j--) {


			x = R[i];
			if (x == 0) {
				continue;
			}

			c = F->mult(x, pivot_inv);
			Q[j] = c;
			c = F->negate(c);
			//cout << "i=" << i << " c=" << c << endl;
			for (ii = i, jj = db; jj >= 0; ii--, jj--) {
				d = B[jj];
				d = F->mult(c, d);
				R[ii] = F->add(d, R[ii]);
			}
			if (R[i] != 0) {
				cout << "unipoly::division_with_remainder_with_report: R[i] != 0" << endl;
				exit(1);
			}
			//cout << "i=" << i << endl;
			//cout << "q="; print_object((unipoly_object)
			// rq, cout); cout << endl;
			//cout << "r="; print_object(r, cout); cout << endl;
		}
		rr[0] = MAXIMUM(db - 1, 0);
		q = rq;

		if (f_report) {
			Algo.print_repeated_character(ost, ' ', db + 1 + 3);
			print_coeffs_top_down_assuming_one_character_per_digit(q, ost);
		}

		assign(a, r, 0 /*verbose_level*/);


		if (f_report) {
			ost << endl;
			Algo.print_repeated_character(ost, ' ', db + 1 + 3);
			Algo.print_repeated_character(ost, '=', da + 1);
			ost << endl;
		}


		for (i = da, j = dq; i >= db; i--, j--) {


			x = R[i];
			if (x == 0) {
				continue;
			}

			if (f_report) {
				if (i == da) {
					print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(b, db, ost);
					ost << " | ";
				}
				else {
					algorithms Algo;
					Algo.print_repeated_character(ost, ' ', db + 1 + 3);
				}
				Algo.print_repeated_character(ost, ' ', da - i);
				print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(r, i, ost);
				ost << endl;
				Algo.print_repeated_character(ost, ' ', db + 1 + 3);
				Algo.print_repeated_character(ost, ' ', da - i);
				print_coeffs_top_down_assuming_one_character_per_digit(b, ost);
				ost << endl;
				Algo.print_repeated_character(ost, ' ', db + 1 + 3);
				Algo.print_repeated_character(ost, ' ', da - i);
				Algo.print_repeated_character(ost, '=', db + 1);
				ost << endl;
			}

			c = F->mult(x, pivot_inv);
			Q[j] = c;
			c = F->negate(c);
			//cout << "i=" << i << " c=" << c << endl;
			for (ii = i, jj = db; jj >= 0; ii--, jj--) {
				d = B[jj];
				d = F->mult(c, d);
				R[ii] = F->add(d, R[ii]);
			}
			if (R[i] != 0) {
				cout << "unipoly::division_with_remainder_with_report: R[i] != 0" << endl;
				exit(1);
			}
			//cout << "i=" << i << endl;
			//cout << "q="; print_object((unipoly_object)
			// rq, cout); cout << endl;
			//cout << "r="; print_object(r, cout); cout << endl;
		}
		q = rq;
		//cout << "q="; print_object(q, cout); cout << endl;
		//cout << "r="; print_object(r, cout); cout << endl;

		while (i > 0 && R[i] == 0) {
			i--;
		}
		rr[0] = i;
		if (rr[0] < 0) {
			rr[0] = 0;
		}
		if (f_report) {
			Algo.print_repeated_character(ost, ' ', db + 1 + 3);
			Algo.print_repeated_character(ost, ' ', da - i);
			print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(r, i, ost);
			ost << endl;
		}
	}
done:
	if (f_v) {
		cout << "unipoly_domain::division_with_remainder_with_report done" << endl;
	}
}


}}

