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

	for (i = m; i >= 0; i--) {
		ost << A[i];
	}
}

void unipoly_domain::mult_easy_with_report(long int rk_a, long int rk_b, long int &rk_c, std::ostream &ost)
{
	unipoly_object a;
	unipoly_object b;
	unipoly_object c;

	create_object_by_rank(a, rk_a,
				__FILE__, __LINE__, 0 /* verbose_level */);
	create_object_by_rank(b, rk_b,
				__FILE__, __LINE__, 0 /* verbose_level */);


	int *ra = (int *) a;
	int *rb = (int *) b;
	int m = ra[0]; // degree of a
	int n = rb[0]; // degree of b
	int mn = m + n; // degree of a * b

	int *rc; // = (int *) c;
	//FREE_int(rc);
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

	ost << "\\begin{verbatim}" << endl;
	ost << setw(m + 1) << rk_a << " x " << setw(n + 1) << rk_b << " = " << endl;
	print_coeffs_top_down_assuming_one_character_per_digit(a, ost);
	ost << " x ";
	print_coeffs_top_down_assuming_one_character_per_digit(b, ost);
	ost << endl;
	print_repeated_character(ost, '=', m + 1 + 3 + n + 1);
	ost << endl;
	for (j = n; j >= 0; j--) {
		if (B[j] == 0) {
			continue;
		}
		print_repeated_character(ost, ' ', 4 + n - j);
		print_coeffs_top_down_assuming_one_character_per_digit(a, ost);
		ost << endl;
	}
	print_repeated_character(ost, ' ', 4);
	print_repeated_character(ost, '=', m + 1 + n);
	ost << endl;

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
	print_repeated_character(ost, ' ', 4);
	print_coeffs_top_down_assuming_one_character_per_digit(c, ost);
	ost << endl;
	rk_c = rank(c);
	print_repeated_character(ost, ' ', 3);
	ost << "=" << setw(mn + 1) << rk_c << endl;
	ost << endl;
	ost << "\\end{verbatim}" << endl;

	delete_object(a);
	delete_object(b);
	delete_object(c);

}



void unipoly_domain::division_with_remainder_with_report(long int rk_a, long int rk_b,
		long int &rk_q, long int &rk_r, std::ostream &ost, int verbose_level)
	//unipoly_object a, unipoly_object b,
	//unipoly_object &q, unipoly_object &r,
	//int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "unipoly_domain::division_with_remainder_with_report" << endl;
	}

	unipoly_object a;
	unipoly_object b;
	unipoly_object q;
	unipoly_object r;

	create_object_by_rank(a, rk_a,
				__FILE__, __LINE__, 0 /* verbose_level */);
	create_object_by_rank(b, rk_b,
				__FILE__, __LINE__, 0 /* verbose_level */);
	create_object_by_rank(q, 0,
				__FILE__, __LINE__, 0 /* verbose_level */);
	create_object_by_rank(r, 0,
				__FILE__, __LINE__, 0 /* verbose_level */);


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

		int_vec_zero(Q, dq + 1);

		ost << "\\begin{verbatim}" << endl;
		print_repeated_character(ost, ' ', db + 1 + 3);
		ost << setw(da + 1) << rk_a << " / " << setw(db + 1) << rk_b << " = " << endl;
		print_repeated_character(ost, ' ', db + 1 + 3);
		print_coeffs_top_down_assuming_one_character_per_digit(a, ost);
		ost << " / ";
		print_coeffs_top_down_assuming_one_character_per_digit(b, ost);
		ost << " = " << endl << endl;

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

		print_repeated_character(ost, ' ', db + 1 + 3);
		print_coeffs_top_down_assuming_one_character_per_digit(q, ost);


		assign(a, r, 0 /*verbose_level*/);


		ost << endl;
		print_repeated_character(ost, ' ', db + 1 + 3);
		print_repeated_character(ost, '=', da + 1);
		ost << endl;


		for (i = da, j = dq; i >= db; i--, j--) {


			x = R[i];
			if (x == 0) {
				continue;
			}

			if (i == da) {
				print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(b, db, ost);
				ost << " | ";
			}
			else {
				print_repeated_character(ost, ' ', db + 1 + 3);
			}
			print_repeated_character(ost, ' ', da - i);
			print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(r, i, ost);
			ost << endl;
			print_repeated_character(ost, ' ', db + 1 + 3);
			print_repeated_character(ost, ' ', da - i);
			print_coeffs_top_down_assuming_one_character_per_digit(b, ost);
			ost << endl;
			print_repeated_character(ost, ' ', db + 1 + 3);
			print_repeated_character(ost, ' ', da - i);
			print_repeated_character(ost, '=', db + 1);
			ost << endl;

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
		//cout << "q="; print_object(q, cout); cout << endl;
		//cout << "r="; print_object(r, cout); cout << endl;

		while (i > 0 && R[i] == 0) {
			i--;
		}
		print_repeated_character(ost, ' ', db + 1 + 3);
		print_repeated_character(ost, ' ', da - i);
		print_coeffs_top_down_assuming_one_character_per_digit_with_degree_given(r, i, ost);
		ost << endl;
		rk_q = rank(q);
		rk_r = rank(r);
		print_repeated_character(ost, ' ', db + 1 + 3);
		print_repeated_character(ost, ' ', da - i - 2);
		ost << "= " << setw(i + 1) << rk_r << endl;
		ost << endl;
		ost << "\\end{verbatim}" << endl;
	}
done:
	if (f_v) {
		cout << "unipoly_domain::division_with_remainder_with_report done" << endl;
	}
}


}}

