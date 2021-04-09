/*
 * finite_field_applications.cpp
 *
 *  Created on: Apr 8, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {



void finite_field::make_all_irreducible_polynomials_of_degree_d(
		int d, std::vector<std::vector<int> > &Table,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int cnt;
	number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << q << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}

#if 0
	cnt = count_all_irreducible_polynomials_of_degree_d(F, d, verbose_level - 2);

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"cnt = " << cnt << endl;
	}

	nb = cnt;

	Table = NEW_int(nb * (d + 1));
#endif

	//NT.factor_prime_power(F->q, p, e);

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				" q=" << q << " p=" << p << " e=" << e << endl;
	}

	unipoly_domain FX(this);

	const char *poly;
	algebra_global Algebra;

	poly = Algebra.get_primitive_polynomial(q, d, 0 /* verbose_level */);

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial is " << poly << endl;
	}

	unipoly_object m;
	unipoly_object g;
	unipoly_object minpol;
	combinatorics_domain Combi;


	FX.create_object_by_rank_string(m, poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial m = ";
		FX.print_object(m, cout);
		cout << endl;
	}

	FX.create_object_by_rank(g, 0, __FILE__, __LINE__, 0 /* verbose_level */);
	FX.create_object_by_rank(minpol, 0, __FILE__, __LINE__, 0 /* verbose_level */);

	int *Frobenius;
	int *Normal_basis;
	int *v;
	int *w;

	//Frobenius = NEW_int(d * d);
	Normal_basis = NEW_int(d * d);
	v = NEW_int(d);
	w = NEW_int(d);

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"before FX.Frobenius_matrix" << endl;
	}
	FX.Frobenius_matrix(Frobenius, m, verbose_level - 2);
	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"Frobenius_matrix = " << endl;
		int_matrix_print(Frobenius, d, d);
		cout << endl;
	}

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"before compute_normal_basis" << endl;
	}
	FX.compute_normal_basis(d, Normal_basis, Frobenius, verbose_level - 1);

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"Normal_basis = " << endl;
		int_matrix_print(Normal_basis, d, d);
		cout << endl;
	}

	cnt = 0;

	Combi.int_vec_first_regular_word(v, d, q);
	while (TRUE) {
		if (f_vv) {
			cout << "finite_field::make_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : v = ";
			Orbiter->Int_vec.print(cout, v, d);
			cout << endl;
		}

		mult_vector_from_the_right(Normal_basis, v, w, d, d);
		if (f_vv) {
			cout << "finite_field::make_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : w = ";
			Orbiter->Int_vec.print(cout, w, d);
			cout << endl;
		}

		FX.delete_object(g);
		FX.create_object_of_degree(g, d - 1);
		for (i = 0; i < d; i++) {
			((int *) g)[1 + i] = w[i];
		}

		FX.minimum_polynomial_extension_field(g, m, minpol, d, Frobenius,
				verbose_level - 3);
		if (f_vv) {
			cout << "finite_field::make_all_irreducible_polynomials_"
					"of_degree_d regular word " << cnt << " : v = ";
			Orbiter->Int_vec.print(cout, v, d);
			cout << " irreducible polynomial = ";
			FX.print_object(minpol, cout);
			cout << endl;
		}



		std::vector<int> T;

		for (i = 0; i <= d; i++) {
			T.push_back(((int *)minpol)[1 + i]);
			//Table[cnt * (d + 1) + i] = ((int *)minpol)[1 + i];
		}
		Table.push_back(T);


		cnt++;


		if (!Combi.int_vec_next_regular_word(v, d, q)) {
			break;
		}

	}

	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"there are " << cnt
				<< " irreducible polynomials "
				"of degree " << d << " over " << "F_" << q << endl;
	}

	FREE_int(Frobenius);
	FREE_int(Normal_basis);
	FREE_int(v);
	FREE_int(w);
	FX.delete_object(m);
	FX.delete_object(g);
	FX.delete_object(minpol);


	if (f_v) {
		cout << "finite_field::make_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << q << " done" << endl;
	}
}

int finite_field::count_all_irreducible_polynomials_of_degree_d(int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int cnt;
	number_theory_domain NT;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << q << endl;
	}


	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d " << endl;
	}

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"p=" << p << " e=" << e << endl;
	}
	if (e > 1) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"e=" << e << " is greater than one" << endl;
	}

	unipoly_domain FX(this);

	const char *poly;
	algebra_global Algebra;

	poly = Algebra.get_primitive_polynomial(q, d, 0 /* verbose_level */);

	unipoly_object m;
	unipoly_object g;
	unipoly_object minpol;


	FX.create_object_by_rank_string(m, poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial m = ";
		FX.print_object(m, cout);
		cout << endl;
	}

	FX.create_object_by_rank(g, 0, __FILE__, __LINE__, 0 /* verbose_level */);
	FX.create_object_by_rank(minpol, 0, __FILE__, __LINE__, 0 /* verbose_level */);

	int *Frobenius;
	//int *F2;
	int *Normal_basis;
	int *v;
	int *w;

	//Frobenius = NEW_int(d * d);
	//F2 = NEW_int(d * d);
	Normal_basis = NEW_int(d * d);
	v = NEW_int(d);
	w = NEW_int(d);

	FX.Frobenius_matrix(Frobenius, m, verbose_level - 3);
	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"Frobenius_matrix = " << endl;
		int_matrix_print(Frobenius, d, d);
		cout << endl;
	}

#if 0
	F->mult_matrix_matrix(Frobenius, Frobenius, F2, d, d, d,
			0 /* verbose_level */);
	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"Frobenius^2 = " << endl;
		int_matrix_print(F2, d, d);
		cout << endl;
	}
#endif

	FX.compute_normal_basis(d, Normal_basis, Frobenius, verbose_level - 3);

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"Normal_basis = " << endl;
		int_matrix_print(Normal_basis, d, d);
		cout << endl;
	}

	cnt = 0;
	Combi.int_vec_first_regular_word(v, d, q);
	while (TRUE) {
		if (f_vv) {
			cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : v = ";
			Orbiter->Int_vec.print(cout, v, d);
			cout << endl;
		}

		mult_vector_from_the_right(Normal_basis, v, w, d, d);
		if (f_vv) {
			cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : w = ";
			Orbiter->Int_vec.print(cout, w, d);
			cout << endl;
		}

		FX.delete_object(g);
		FX.create_object_of_degree(g, d - 1);
		for (i = 0; i < d; i++) {
			((int *) g)[1 + i] = w[i];
		}

		FX.minimum_polynomial_extension_field(g, m, minpol, d,
				Frobenius, verbose_level - 3);
		if (f_vv) {
			cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : v = ";
			Orbiter->Int_vec.print(cout, v, d);
			cout << " irreducible polynomial = ";
			FX.print_object(minpol, cout);
			cout << endl;
		}
		if (FX.degree(minpol) != d) {
			cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
					"The polynomial does not have degree d"
					<< endl;
			FX.print_object(minpol, cout);
			cout << endl;
			exit(1);
		}
		if (!FX.is_irreducible(minpol, verbose_level)) {
			cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
					"The polynomial is not irreducible" << endl;
			FX.print_object(minpol, cout);
			cout << endl;
			exit(1);
		}


		cnt++;

		if (!Combi.int_vec_next_regular_word(v, d, q)) {
			break;
		}

	}

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d "
				"there are " << cnt << " irreducible polynomials "
				"of degree " << d << " over " << "F_" << q << endl;
	}

	FREE_int(Frobenius);
	//FREE_int(F2);
	FREE_int(Normal_basis);
	FREE_int(v);
	FREE_int(w);
	FX.delete_object(m);
	FX.delete_object(g);
	FX.delete_object(minpol);

	if (f_v) {
		cout << "finite_field::count_all_irreducible_polynomials_of_degree_d done" << endl;
	}
	return cnt;
}

void finite_field::polynomial_division(
		std::string &A_coeffs, std::string &B_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::polynomial_division" << endl;
	}

	//int q = F->q;

	int *data_A;
	int *data_B;
	int sz_A, sz_B;

	Orbiter->Int_vec.scan(A_coeffs, data_A, sz_A);
	Orbiter->Int_vec.scan(B_coeffs, data_B, sz_B);



	unipoly_domain FX(this);
	unipoly_object A, B, Q, R;


	int da = sz_A - 1;
	int db = sz_B - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		FX.s_i(A, i) = data_A[i];
	}

	FX.create_object_of_degree(B, da);

	for (i = 0; i <= db; i++) {
		FX.s_i(B, i) = data_B[i];
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;
	}


	if (f_v) {
		cout << "B(X)=";
		FX.print_object(B, cout);
		cout << endl;
		}

	FX.create_object_of_degree(Q, da);

	FX.create_object_of_degree(R, da);


	if (f_v) {
		cout << "finite_field::polynomial_division "
				"before FX.division_with_remainder" << endl;
	}

	FX.division_with_remainder(
		A, B,
		Q, R,
		verbose_level);

	if (f_v) {
		cout << "finite_field::polynomial_division "
				"after FX.division_with_remainder" << endl;
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;

		cout << "Q(X)=";
		FX.print_object(Q, cout);
		cout << endl;

		cout << "R(X)=";
		FX.print_object(R, cout);
		cout << endl;
	}

	if (f_v) {
		cout << "finite_field::polynomial_division done" << endl;
	}
}

void finite_field::extended_gcd_for_polynomials(
		std::string &A_coeffs, std::string &B_coeffs, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::extended_gcd_for_polynomials" << endl;
	}



	int *data_A;
	int *data_B;
	int sz_A, sz_B;

	Orbiter->Int_vec.scan(A_coeffs, data_A, sz_A);
	Orbiter->Int_vec.scan(B_coeffs, data_B, sz_B);

	number_theory_domain NT;




	unipoly_domain FX(this);
	unipoly_object A, B, U, V, G;


	int da = sz_A - 1;
	int db = sz_B - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= q) {
			data_A[i] = NT.mod(data_A[i], q);
		}
		FX.s_i(A, i) = data_A[i];
	}

	FX.create_object_of_degree(B, da);

	for (i = 0; i <= db; i++) {
		if (data_B[i] < 0 || data_B[i] >= q) {
			data_B[i] = NT.mod(data_B[i], q);
		}
		FX.s_i(B, i) = data_B[i];
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;


		cout << "B(X)=";
		FX.print_object(B, cout);
		cout << endl;
	}

	FX.create_object_of_degree(U, da);

	FX.create_object_of_degree(V, da);

	FX.create_object_of_degree(G, da);


	if (f_v) {
		cout << "finite_field::extended_gcd_for_polynomials before FX.extended_gcd" << endl;
	}

	{
		FX.extended_gcd(
			A, B,
			U, V, G, verbose_level);
	}

	if (f_v) {
		cout << "finite_field::extended_gcd_for_polynomials after FX.extended_gcd" << endl;
	}

	if (f_v) {
		cout << "U(X)=";
		FX.print_object(U, cout);
		cout << endl;

		cout << "V(X)=";
		FX.print_object(V, cout);
		cout << endl;

		cout << "G(X)=";
		FX.print_object(G, cout);
		cout << endl;

		cout << "deg G(X) = " << FX.degree(G) << endl;
	}

	if (FX.degree(G) == 0) {
		int c, cv, d;

		c = FX.s_i(G, 0);
		if (c != 1) {
			if (f_v) {
				cout << "normalization:" << endl;
			}
			cv = inverse(c);
			if (f_v) {
				cout << "cv=" << cv << endl;
			}

			d = FX.degree(U);
			for (i = 0; i <= d; i++) {
				FX.s_i(U, i) = mult(cv, FX.s_i(U, i));
			}

			d = FX.degree(V);
			for (i = 0; i <= d; i++) {
				FX.s_i(V, i) = mult(cv, FX.s_i(V, i));
			}

			d = FX.degree(G);
			for (i = 0; i <= d; i++) {
				FX.s_i(G, i) = mult(cv, FX.s_i(G, i));
			}



			if (f_v) {
				cout << "after normalization:" << endl;

				cout << "U(X)=";
				FX.print_object(U, cout);
				cout << endl;

				cout << "V(X)=";
				FX.print_object(V, cout);
				cout << endl;

				cout << "G(X)=";
				FX.print_object(G, cout);
				cout << endl;
			}
		}

	}

	if (f_v) {
		cout << "finite_field::extended_gcd_for_polynomials done" << endl;
	}
}


void finite_field::polynomial_mult_mod(
		std::string &A_coeffs, std::string &B_coeffs, std::string &M_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::polynomial_mult_mod" << endl;
	}

	int *data_A;
	int *data_B;
	int *data_M;
	int sz_A, sz_B, sz_M;

	Orbiter->Int_vec.scan(A_coeffs, data_A, sz_A);
	Orbiter->Int_vec.scan(B_coeffs, data_B, sz_B);
	Orbiter->Int_vec.scan(M_coeffs, data_M, sz_M);

	number_theory_domain NT;




	unipoly_domain FX(this);
	unipoly_object A, B, M, C;


	int da = sz_A - 1;
	int db = sz_B - 1;
	int dm = sz_M - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= q) {
			data_A[i] = NT.mod(data_A[i], q);
		}
		FX.s_i(A, i) = data_A[i];
	}

	FX.create_object_of_degree(B, da);

	for (i = 0; i <= db; i++) {
		if (data_B[i] < 0 || data_B[i] >= q) {
			data_B[i] = NT.mod(data_B[i], q);
		}
		FX.s_i(B, i) = data_B[i];
	}

	FX.create_object_of_degree(M, dm);

	for (i = 0; i <= dm; i++) {
		if (data_M[i] < 0 || data_M[i] >= q) {
			data_M[i] = NT.mod(data_M[i], q);
		}
		FX.s_i(M, i) = data_M[i];
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;


		cout << "B(X)=";
		FX.print_object(B, cout);
		cout << endl;

		cout << "M(X)=";
		FX.print_object(M, cout);
		cout << endl;
	}

	FX.create_object_of_degree(C, da + db);



	if (f_v) {
		cout << "finite_field::polynomial_mult_mod before FX.mult_mod" << endl;
	}

	{
		FX.mult_mod(A, B, C, M, verbose_level);
	}

	if (f_v) {
		cout << "finite_field::polynomial_mult_mod after FX.mult_mod" << endl;
	}

	if (f_v) {
		cout << "C(X)=";
		FX.print_object(C, cout);
		cout << endl;

		cout << "deg C(X) = " << FX.degree(C) << endl;
	}

	if (f_v) {
		cout << "finite_field::polynomial_mult_mod done" << endl;
	}
}

void finite_field::Berlekamp_matrix(
		std::string &Berlekamp_matrix_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::Berlekamp_matrix" << endl;
	}

	int *data_A;
	int sz_A;

	Orbiter->Int_vec.scan(Berlekamp_matrix_coeffs, data_A, sz_A);

	number_theory_domain NT;




	unipoly_domain FX(this);
	unipoly_object A;


	int da = sz_A - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= q) {
			data_A[i] = NT.mod(data_A[i], q);
		}
		FX.s_i(A, i) = data_A[i];
	}



	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;
	}


	int *B;
	int r;



	if (f_v) {
		cout << "finite_field::Berlekamp_matrix before FX.Berlekamp_matrix" << endl;
	}

	{
		FX.Berlekamp_matrix(B, A, verbose_level);
	}

	if (f_v) {
		cout << "finite_field::Berlekamp_matrix after FX.Berlekamp_matrix" << endl;
	}

	if (f_v) {
		cout << "B=" << endl;
		int_matrix_print(B, da, da);
		cout << endl;
	}

	r = rank_of_matrix(B, da, 0 /* verbose_level */);

	if (f_v) {
		cout << "The matrix B has rank " << r << endl;
	}

	FREE_int(B);

	if (f_v) {
		cout << "finite_field::Berlekamp_matrix done" << endl;
	}
}




void finite_field::compute_normal_basis(int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::compute_normal_basis "
				<< " q=" << q << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}


	unipoly_domain FX(this);

	const char *poly;
	algebra_global Algebra;

	poly = Algebra.get_primitive_polynomial(q, d, 0 /* verbose_level */);

	if (f_v) {
		cout << "finite_field::compute_normal_basis "
				"chosen irreducible polynomial is " << poly << endl;
	}

	unipoly_object m;
	unipoly_object g;
	unipoly_object minpol;
	combinatorics_domain Combi;


	FX.create_object_by_rank_string(m, poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "finite_field::compute_normal_basis "
				"chosen irreducible polynomial m = ";
		FX.print_object(m, cout);
		cout << endl;
	}

	FX.create_object_by_rank(g, 0, __FILE__, __LINE__, 0 /* verbose_level */);
	FX.create_object_by_rank(minpol, 0, __FILE__, __LINE__, 0 /* verbose_level */);

	int *Frobenius;
	int *Normal_basis;

	//Frobenius = NEW_int(d * d);
	Normal_basis = NEW_int(d * d);

	if (f_v) {
		cout << "finite_field::compute_normal_basis "
				"before FX.Frobenius_matrix" << endl;
	}
	FX.Frobenius_matrix(Frobenius, m, verbose_level - 2);
	if (f_v) {
		cout << "finite_field::compute_normal_basis "
				"Frobenius_matrix = " << endl;
		int_matrix_print(Frobenius, d, d);
		cout << endl;
	}

	if (f_v) {
		cout << "finite_field::compute_normal_basis "
				"before compute_normal_basis" << endl;
	}

	FX.compute_normal_basis(d, Normal_basis, Frobenius, verbose_level - 1);

	if (f_v) {
		cout << "finite_field::compute_normal_basis "
				"Normal_basis = " << endl;
		int_matrix_print(Normal_basis, d, d);
		cout << endl;
	}

	if (f_v) {
		cout << "finite_field::compute_normal_basis done" << endl;
	}
}


void finite_field::do_nullspace(
		int m, int n, std::string &text,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	int *A;
	int *base_cols;
	int len, rk, i, rk1;

	latex_interface Li;

	if (f_v) {
		cout << "finite_field::do_nullspace" << endl;
	}


	Orbiter->Int_vec.scan(text, M, len);
	if (len != m * n) {
		cout << "number of coordinates received differs from m * n" << endl;
		cout << "received " << len << endl;
		exit(1);
	}

	if (m > n) {
		cout << "nullspace needs m < n" << endl;
		exit(1);
	}

	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	Orbiter->Int_vec.copy(M, A, m * n);

	if (f_v) {
		cout << "finite_field::do_nullspace before F->perp_standard" << endl;
	}

	rk = perp_standard(n, m, A, 0 /*verbose_level*/);

	if (f_v) {
		cout << "finite_field::do_nullspace after F->perp_standard" << endl;
	}


	if (f_v) {
		cout << "finite_field::do_nullspace after perp_standard:" << endl;
		int_matrix_print(A, n, n);
		cout << "rk=" << rk << endl;
	}

	rk1 = Gauss_int(A + rk * n,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, n - rk, n, n,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "finite_field::do_nullspace after RREF" << endl;
		int_matrix_print(A + rk * n, rk1, n);
		cout << "rank of nullspace = " << rk1 << endl;

		cout << "finite_field::do_nullspace coefficients:" << endl;
		Orbiter->Int_vec.print(cout, A + rk * n, rk1 * n);
		cout << endl;

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;
	}

	if (f_normalize_from_the_left) {
		if (f_v) {
			cout << "finite_field::do_nullspace normalizing from the left" << endl;
		}
		for (i = rk; i < n; i++) {
			PG_element_normalize_from_front(A + i * n, 1, n);
		}

		if (f_v) {
			cout << "finite_field::do_nullspace after normalize from the left:" << endl;
			int_matrix_print(A, n, n);
			cout << "rk=" << rk << endl;

			cout << "$$" << endl;
			cout << "\\left[" << endl;
			Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
			cout << "\\right]" << endl;
			cout << "$$" << endl;
		}
	}

	if (f_normalize_from_the_right) {
		if (f_v) {
			cout << "finite_field::do_nullspace normalizing from the right" << endl;
		}
		for (i = rk; i < n; i++) {
			PG_element_normalize(A + i * n, 1, n);
		}

		if (f_v) {
			cout << "finite_field::do_nullspace after normalize from the right:" << endl;
			int_matrix_print(A, n, n);
			cout << "rk=" << rk << endl;

			cout << "$$" << endl;
			cout << "\\left[" << endl;
			Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
			cout << "\\right]" << endl;
			cout << "$$" << endl;
		}
	}


	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "nullspace_%d_%d.tex", m, n);
		fname.assign(str);
		snprintf(title, 1000, "Nullspace");
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "finite_field::do_nullspace before report" << endl;
			}
			//report(ost, verbose_level);

			ost << "\\noindent Input matrix:" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, M, m, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;

			ost << "RREF:" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, A, rk, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;

			ost << "Basis for Perp:" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, A + rk * n, rk1, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;


			if (f_v) {
				cout << "finite_field::do_nullspace after report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "finite_field::do_nullspace written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	FREE_int(M);
	FREE_int(A);
	FREE_int(base_cols);

	if (f_v) {
		cout << "finite_field::do_nullspace done" << endl;
	}
}

void finite_field::do_RREF(
		int m, int n, std::string &text,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	int *A;
	int *base_cols;
	int len, rk, i;
	latex_interface Li;

	if (f_v) {
		cout << "do_RREF" << endl;
	}


	Orbiter->Int_vec.scan(text, M, len);
	if (len != m * n) {
		cout << "finite_field::do_RREF "
				"number of coordinates received differs from m * n" << endl;
		cout << "received " << len << endl;
		exit(1);
	}


	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	Orbiter->Int_vec.copy(M, A, m * n);

	rk = Gauss_int(A,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, m, n, n,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "after RREF:" << endl;
		int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;

		cout << "coefficients:" << endl;
		Orbiter->Int_vec.print(cout, A, rk * n);
		cout << endl;

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.int_matrix_print_tex(cout, A, rk, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;
	}



	if (f_normalize_from_the_left) {
		if (f_v) {
			cout << "normalizing from the left" << endl;
		}
		for (i = 0; i < rk; i++) {
			PG_element_normalize_from_front(A + i * n, 1, n);
		}

		if (f_v) {
			cout << "after normalize from the left:" << endl;
			int_matrix_print(A, rk, n);
			cout << "rk=" << rk << endl;
		}
	}

	if (f_normalize_from_the_right) {
		if (f_v) {
			cout << "normalizing from the right" << endl;
		}
		for (i = 0; i < rk; i++) {
			PG_element_normalize(A + i * n, 1, n);
		}

		if (f_v) {
			cout << "after normalize from the right:" << endl;
			int_matrix_print(A, rk, n);
			cout << "rk=" << rk << endl;
		}
	}


	Orbiter->Int_vec.copy(M, A, m * n);

	RREF_demo(A, m, n, verbose_level);



	FREE_int(M);
	FREE_int(A);
	FREE_int(base_cols);

	if (f_v) {
		cout << "finite_field::do_RREF done" << endl;
	}
}

void finite_field::apply_Walsh_Hadamard_transform(
		std::string &fname_csv_in, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::apply_Walsh_Hadamard_transform" << endl;
	}


	boolean_function_domain *BF;

	BF = NEW_OBJECT(boolean_function_domain);

	BF->init(n, verbose_level);


	file_io Fio;
	int *M;
	int m, nb_cols;
	int len;
	string fname_csv_out;

	fname_csv_out.assign(fname_csv_in);
	chop_off_extension(fname_csv_out);
	fname_csv_out.append("_transformed.csv");

	Fio.int_matrix_read_csv(fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	if (len != BF->Q) {
		cout << "finite_field::apply_Walsh_Hadamard_transform len != BF->Q" << endl;
		exit(1);
	}
	BF->raise(M, BF->F);

	BF->apply_Walsh_transform(BF->F, BF->T);

	cout << " : ";
	Orbiter->Int_vec.print(cout, BF->T, BF->Q);
	cout << endl;

	if (EVEN(n)) {
		if (BF->is_bent(BF->T)) {
			cout << "is bent" << endl;
		}
		else {
			cout << "is not bent" << endl;
		}
	}
	else {
		if (BF->is_near_bent(BF->T)) {
			cout << "is near bent" << endl;
		}
		else {
			cout << "is not near bent" << endl;
		}

	}
	Fio.int_matrix_write_csv(fname_csv_out, BF->T, m, nb_cols);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);
	FREE_OBJECT(BF);

	if (f_v) {
		cout << "finite_field::apply_Walsh_Hadamard_transform done" << endl;
	}
}

void finite_field::algebraic_normal_form(
		std::string &fname_csv_in, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::algebraic_normal_form" << endl;
	}


	boolean_function_domain *BF;

	BF = NEW_OBJECT(boolean_function_domain);

	if (f_v) {
		cout << "finite_field::algebraic_normal_form before BF->init" << endl;
	}
	BF->init(n, verbose_level);
	if (f_v) {
		cout << "finite_field::algebraic_normal_form after BF->init" << endl;
	}


	file_io Fio;
	int *M;
	int m, nb_cols;
	int len;
	string fname_csv_out;

	fname_csv_out.assign(fname_csv_in);
	chop_off_extension(fname_csv_out);
	fname_csv_out.append("_alg_normal_form.csv");

	Fio.int_matrix_read_csv(fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	if (len != BF->Q) {
		cout << "finite_field::algebraic_normal_form len != BF->Q" << endl;
		exit(1);
	}

	int *coeff;
	int nb_coeff;

	nb_coeff = BF->Poly[n].get_nb_monomials();

	coeff = NEW_int(nb_coeff);

	if (f_v) {
		cout << "finite_field::algebraic_normal_form before BF->compute_polynomial_representation" << endl;
	}
	BF->compute_polynomial_representation(M, coeff, verbose_level);
	if (f_v) {
		cout << "finite_field::algebraic_normal_form after BF->compute_polynomial_representation" << endl;
	}

	cout << "algebraic normal form:" << endl;
	BF->Poly[n].print_equation(cout, coeff);
	cout << endl;

	cout << "algebraic normal form in tex:" << endl;
	BF->Poly[n].print_equation_tex(cout, coeff);
	cout << endl;

	cout << "algebraic normal form in numerical form:" << endl;
	BF->Poly[n].print_equation_numerical(cout, coeff);
	cout << endl;



	Fio.int_matrix_write_csv(fname_csv_out, coeff, 1, nb_coeff);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);
	FREE_OBJECT(BF);

	if (f_v) {
		cout << "finite_field::algebraic_normal_form done" << endl;
	}
}

void finite_field::apply_trace_function(
		std::string &fname_csv_in, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::apply_trace_function" << endl;
	}


	file_io Fio;
	int *M;
	int m, nb_cols;
	int len, i;
	string fname_csv_out;

	fname_csv_out.assign(fname_csv_in);
	chop_off_extension(fname_csv_out);
	fname_csv_out.append("_trace.csv");

	Fio.int_matrix_read_csv(fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	for (i = 0; i < len; i++) {
		M[i] = absolute_trace(M[i]);
	}
	Fio.int_matrix_write_csv(fname_csv_out, M, m, nb_cols);

	FREE_int(M);

	if (f_v) {
		cout << "finite_field::apply_trace_function done" << endl;
	}
}

void finite_field::apply_power_function(
		std::string &fname_csv_in, long int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::apply_power_function" << endl;
	}


	file_io Fio;
	int *M;
	int m, nb_cols;
	int len, i;
	string fname_csv_out;

	fname_csv_out.assign(fname_csv_in);
	chop_off_extension(fname_csv_out);

	char str[1000];

	sprintf(str, "_power_%ld.csv", d);
	fname_csv_out.append(str);

	Fio.int_matrix_read_csv(fname_csv_in, M, m, nb_cols, verbose_level);
	len = m * nb_cols;
	for (i = 0; i < len; i++) {
		M[i] = power(M[i], d);
	}
	Fio.int_matrix_write_csv(fname_csv_out, M, m, nb_cols);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);

	if (f_v) {
		cout << "finite_field::apply_power_function done" << endl;
	}
}

void finite_field::identity_function(
		std::string &fname_csv_out, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::identity_function" << endl;
	}


	file_io Fio;
	int *M;
	int i;

	M = NEW_int(q);
	for (i = 0; i < q; i++) {
		M[i] = i;
	}
	Fio.int_matrix_write_csv(fname_csv_out, M, 1, q);
	cout << "written file " << fname_csv_out << " of size "
			<< Fio.file_size(fname_csv_out) << endl;

	FREE_int(M);

	if (f_v) {
		cout << "finite_field::identity_function done" << endl;
	}
}

void finite_field::do_trace(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int s, t;
	int *T = NULL;
	int *T0 = NULL;
	int *T1 = NULL;
	int nb_T0 = 0;
	int nb_T1 = 0;


	if (f_v) {
		cout << "finite_field::do_trace" << endl;
	}


	T = NEW_int(q);
	T0 = NEW_int(q);
	T1 = NEW_int(q);

	for (s = 0; s < q; s++) {

#if 0
		int s2, /*s3,*/ s4, s8, s2p1, s2p1t7, s2p1t6, s2p1t4, f;

		s2 = F->mult(s, s);
		s2p1 = F->add(s2, 1);
		s2p1t7 = F->power(s2p1, 7);
		s2p1t6 = F->power(s2p1, 6);
		s2p1t4 = F->power(s2p1, 4);
		//s3 = F->power(s, 3);
		s4 = F->power(s, 4);
		s8 = F->power(s, 8);

		f = F->add4(F->mult(s, s2p1t7), F->mult(s2, s2p1t6), F->mult(s4, s2p1t4), s8);

		//f = F->mult(top, F->inverse(bot));
		t = F->absolute_trace(f);
#endif

		t = absolute_trace(s);
		T[s] = t;
		if (t == 1) {
			T1[nb_T1++] = s;
		}
		else {
			T0[nb_T0++] = s;
		}
	}



	cout << "Trace 0:" << endl;
	Orbiter->Int_vec.print_fully(cout, T0, nb_T0);
	cout << endl;

	cout << "Trace 1:" << endl;
	Orbiter->Int_vec.print_fully(cout, T1, nb_T1);
	cout << endl;

	char str[1000];
	string fname_csv;
	file_io Fio;

	snprintf(str, 1000, "F_q%d_trace.csv", q);
	fname_csv.assign(str);
	Fio.int_matrix_write_csv(fname_csv, T, 1, q);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	snprintf(str, 1000, "F_q%d_trace_0.csv", q);
	fname_csv.assign(str);
	Fio.int_vec_write_csv(T0, nb_T0,
			fname_csv, "Trace_0");
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;

	snprintf(str, 1000, "F_q%d_trace_1.csv", q);
	fname_csv.assign(str);
	Fio.int_vec_write_csv(T1, nb_T1,
			fname_csv, "Trace_1");
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;

	FREE_int(T);
	FREE_int(T0);
	FREE_int(T1);

	if (f_v) {
		cout << "finite_field::do_trace done" << endl;
	}
}

void finite_field::do_norm(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int s, t;
	int *T0 = NULL;
	int *T1 = NULL;
	int nb_T0 = 0;
	int nb_T1 = 0;

	if (f_v) {
		cout << "finite_field::do_norm" << endl;
	}

	T0 = NEW_int(q);
	T1 = NEW_int(q);

	for (s = 0; s < q; s++) {
		t = absolute_norm(s);
		if (t == 1) {
			T1[nb_T1++] = s;
		}
		else {
			T0[nb_T0++] = s;
		}
	}

	cout << "Norm 0:" << endl;
	Orbiter->Int_vec.print_fully(cout, T0, nb_T0);
	cout << endl;

	cout << "Norm 1:" << endl;
	Orbiter->Int_vec.print_fully(cout, T1, nb_T1);
	cout << endl;


	char str[1000];
	string fname_csv;
	file_io Fio;

	snprintf(str, 1000, "F_q%d_norm_0.csv", q);
	fname_csv.assign(str);
	Fio.int_vec_write_csv(T0, nb_T0,
			fname_csv, "Norm_0");
	cout << "written file " << fname_csv << " of size " << Fio.file_size(fname_csv) << endl;

	snprintf(str, 1000, "F_q%d_norm_1.csv", q);
	fname_csv.assign(str);
	Fio.int_vec_write_csv(T1, nb_T1,
			fname_csv, "Norm_1");
	cout << "written file " << fname_csv << " of size " << Fio.file_size(fname_csv) << endl;


	if (f_v) {
		cout << "finite_field::do_norm done" << endl;
	}
}

void finite_field::do_cheat_sheet_GF(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::do_cheat_sheet_GF q=" << q << endl;
	}

	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "%s.tex", label.c_str());
	snprintf(title, 1000, "Cheat Sheet $%s$", label_tex.c_str());
	//sprintf(author, "");
	author[0] = 0;



	addition_table_save_csv();

	multiplication_table_save_csv();

	addition_table_reordered_save_csv();

	multiplication_table_reordered_save_csv();


	{
		ofstream f(fname);


		latex_interface L;


		L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
			title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
				TRUE /* f_12pt */,
				TRUE /* f_enlarged_page */,
				TRUE /* f_pagenumbers */,
				NULL /* extra_praeamble */);


		cheat_sheet(f, verbose_level);

		cheat_sheet_main_table(f, verbose_level);

		cheat_sheet_addition_table(f, verbose_level);

		cheat_sheet_multiplication_table(f, verbose_level);

		cheat_sheet_power_table(f, TRUE, verbose_level);

		cheat_sheet_power_table(f, FALSE, verbose_level);





		L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "finite_field::do_cheat_sheet_GF q=" << q << " done" << endl;
	}
}


void finite_field::do_make_table_of_irreducible_polynomials(int deg, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::do_make_table_of_irreducible_polynomials" << endl;
		cout << "deg=" << deg << endl;
		cout << "q=" << q << endl;
	}

	int nb;
	std::vector<std::vector<int>> Table;

	make_all_irreducible_polynomials_of_degree_d(deg,
			Table, verbose_level);

	nb = Table.size();

	cout << "The " << nb << " irreducible polynomials of "
			"degree " << deg << " over F_" << q << " are:" << endl;

	Orbiter->Int_vec.vec_print(Table);


	int *T;
	int i, j;

	T = NEW_int(Table.size() * (deg + 1));
	for (i = 0; i < Table.size(); i++) {
		for (j = 0; j < deg + 1; j++) {
			T[i * (deg + 1) + j] = Table[i][j];
		}
	}



	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "Irred_q%d_d%d.tex", q, deg);
		fname.assign(str);
		snprintf(title, 1000, "Irreducible Polynomials of Degree %d over F%d", deg, q);
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;
			geometry_global GG;
			long int rk;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "finite_field::do_make_table_of_irreducible_polynomials before report" << endl;
			}
			//report(ost, verbose_level);

			ost << "There are " << Table.size() << " irreducible polynomials of "
					"degree " << deg << " over the field F" << q << ":\\\\" << endl;
			//ost << "\\begin{multicols}{2}" << endl;
			//ost << "\\noindent" << endl;
			for (i = 0; i < Table.size(); i++) {
				ost << i << " : $";
				for (j = deg; j>= 0; j--) {
					ost << T[i * (deg + 1) + j];
				}
				ost << " : ";
				rk = GG.AG_element_rank(q, T + i * (deg + 1), 1, deg + 1);
				ost << rk;
				ost << "$\\\\" << endl;
			}
			//ost << "\\end{multicols}" << endl;


			if (f_v) {
				cout << "finite_field::do_make_table_of_irreducible_polynomials after report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "finite_field::do_make_table_of_irreducible_polynomials written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	FREE_int(T);

	//int_matrix_print(Table, nb, deg + 1);

	//FREE_int(Table);

	if (f_v) {
		cout << "finite_field::do_make_table_of_irreducible_polynomials done" << endl;
	}
}

void finite_field::polynomial_find_roots(
		std::string &A_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::polynomial_find_roots" << endl;
	}

	int *data_A;
	int sz_A;

	Orbiter->Int_vec.scan(A_coeffs, data_A, sz_A);

	number_theory_domain NT;




	unipoly_domain FX(this);
	unipoly_object A;


	int da = sz_A - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= q) {
			data_A[i] = NT.mod(data_A[i], q);
		}
		FX.s_i(A, i) = data_A[i];
	}


	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;
	}



	if (f_v) {
		cout << "finite_field::polynomial_find_roots before FX.mult_mod" << endl;
	}

	{
		int a, b;

		for (a = 0; a < q; a++) {
			b = FX.substitute_scalar_in_polynomial(
				A, a, 0 /* verbose_level*/);
			if (b == 0) {
				cout << "a root is " << a << endl;
			}
		}
	}

	if (f_v) {
		cout << "finite_field::polynomial_find_roots after FX.mult_mod" << endl;
	}


	if (f_v) {
		cout << "finite_field::polynomial_find_roots done" << endl;
	}
}

void finite_field::sift_polynomials(long int rk0, long int rk1, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	unipoly_object p;
	longinteger_domain ZZ;
	long int rk;
	int f_is_irred, f_is_primitive;
	int f, i;
	long int len, idx;
	long int *Table;
	longinteger_object Q, m1, Qm1;

	if (f_v) {
		cout << "finite_field::sift_polynomials" << endl;
	}
	unipoly_domain D(this);

	len = rk1 - rk0;
	Table = NEW_lint(len * 3);
	for (rk = rk0; rk < rk1; rk++) {

		idx = rk - rk0;

		D.create_object_by_rank(p, rk,
				__FILE__, __LINE__, 0 /* verbose_level*/);
		if (f_v) {
			cout << "rk=" << rk << " poly=";
			D.print_object(p, cout);
			cout << endl;
		}


		int nb_primes;
		longinteger_object *primes;
		int *exponents;


		f = D.degree(p);
		Q.create(q, __FILE__, __LINE__);
		m1.create(-1, __FILE__, __LINE__);
		ZZ.power_int(Q, f);
		ZZ.add(Q, m1, Qm1);
		if (f_v) {
			cout << "finite_field::sift_polynomials Qm1 = " << Qm1 << endl;
		}
		ZZ.factor_into_longintegers(Qm1, nb_primes,
				primes, exponents, verbose_level - 2);
		if (f_v) {
			cout << "finite_field::get_a_primitive_polynomial after factoring "
					<< Qm1 << " nb_primes=" << nb_primes << endl;
			cout << "primes:" << endl;
			for (i = 0; i < nb_primes; i++) {
				cout << i << " : " << primes[i] << endl;
			}
		}

		f_is_irred = D.is_irreducible(p, verbose_level - 2);
		f_is_primitive = D.is_primitive(p,
				Qm1,
				nb_primes, primes,
				verbose_level);

		if (f_v) {
			cout << "rk=" << rk << " poly=";
			D.print_object(p, cout);
			cout << " is_irred=" << f_is_irred << " is_primitive=" << f_is_primitive << endl;
		}

		Table[idx * 3 + 0] = rk;
		Table[idx * 3 + 1] = f_is_irred;
		Table[idx * 3 + 2] = f_is_primitive;


		D.delete_object(p);

		FREE_int(exponents);
		FREE_OBJECTS(primes);
	}
	for (idx = 0; idx < len; idx++) {
		rk = Table[idx * 3 + 0];
		D.create_object_by_rank(p, rk,
				__FILE__, __LINE__, 0 /* verbose_level*/);
		f_is_irred = Table[idx * 3 + 1];
		f_is_primitive = Table[idx * 3 + 2];
		if (f_v) {
			cout << "rk=" << rk;
			cout << " is_irred=" << f_is_irred << " is_primitive=" << f_is_primitive;
			cout << " poly=";
			D.print_object(p, cout);
			cout << endl;
		}
		D.delete_object(p);
	}

	if (f_v) {
		cout << "finite_field::sift_polynomials done" << endl;
	}

}

void finite_field::mult_polynomials(long int rk0, long int rk1, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::mult_polynomials" << endl;
	}
	unipoly_domain D(this);

	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "polynomial_mult_%ld_%ld.tex", rk0, rk1);
		fname.assign(str);
		snprintf(title, 1000, "Polynomial Mult");
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "finite_field::mult_polynomials before report" << endl;
			}
			//report(ost, verbose_level);

			long int rk2;
			D.mult_easy_with_report(rk0, rk1, rk2, ost, verbose_level);
			ost << "$" << rk0 << " \\otimes " << rk1 << " = " << rk2 << "$\\\\" << endl;


			if (f_v) {
				cout << "finite_field::mult_polynomials after report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "finite_field::mult_polynomials written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "finite_field::mult_polynomials done" << endl;
	}

}

void finite_field::polynomial_division_from_file_with_report(
		std::string &input_file, long int rk1, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::polynomial_division_from_file_with_report" << endl;
	}
	unipoly_domain D(this);
	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "polynomial_division_file_%ld.tex", rk1);
		fname.assign(str);
		snprintf(title, 1000, "Polynomial Division");
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "finite_field::polynomial_division_from_file_with_report before division_with_remainder_numerically_with_report" << endl;
			}
			//report(ost, verbose_level);

			long int rk_q, rk_r;

			D.division_with_remainder_from_file_with_report(input_file, rk1,
					rk_q, rk_r, ost, verbose_level);

			ost << "$ / " << rk1 << " = " << rk_q << "$ Remainder $" << rk_r << "$\\\\" << endl;


			if (f_v) {
				cout << "finite_field::polynomial_division_from_file_with_report after division_with_remainder_numerically_with_report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "finite_field::polynomial_division_from_file_with_report written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "finite_field::polynomial_division_from_file_with_report done" << endl;
	}

}

void finite_field::polynomial_division_from_file_all_k_error_patterns_with_report(
		std::string &input_file, long int rk1, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::polynomial_division_from_file_all_k_error_patterns_with_report" << endl;
	}
	unipoly_domain D(this);
	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "polynomial_division_file_all_%d_error_patterns_%ld.tex", k, rk1);
		fname.assign(str);
		snprintf(title, 1000, "Polynomial Division");
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "finite_field::polynomial_division_from_file_all_k_error_patterns_with_report before division_with_remainder_numerically_with_report" << endl;
			}
			//report(ost, verbose_level);

			long int *rk_q, *rk_r;
			int N, n, h;
			combinatorics_domain Combi;

			int *set;

			set = NEW_int(k);



			D.division_with_remainder_from_file_all_k_bit_error_patterns(input_file,
					rk1, k,
					rk_q, rk_r, n, N, ost, verbose_level);

			ost << "$" << input_file << " / " << rk1 << "$\\\\" << endl;

			for (h = 0; h < N; h++) {
				Combi.unrank_k_subset(h, set, n, k);
				ost << h << " : ";
				Orbiter->Int_vec.print(ost, set, k);
				ost << " : ";
				ost << rk_r[h] << "\\\\" << endl;
			}

			FREE_lint(rk_q);
			FREE_lint(rk_r);

			if (f_v) {
				cout << "finite_field::polynomial_division_from_file_all_k_error_patterns_with_report after division_with_remainder_numerically_with_report" << endl;
			}

			FREE_int(set);

			L.foot(ost);

		}
		file_io Fio;

		cout << "finite_field::polynomial_division_from_file_all_k_error_patterns_with_report written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "finite_field::polynomial_division_from_file_with_report done" << endl;
	}

}

void finite_field::polynomial_division_with_report(long int rk0, long int rk1, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::polynomial_division_with_report" << endl;
	}
	unipoly_domain D(this);

	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "polynomial_division_%ld_%ld.tex", rk0, rk1);
		fname.assign(str);
		snprintf(title, 1000, "Polynomial Division");
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "finite_field::polynomial_division_with_report before division_with_remainder_numerically_with_report" << endl;
			}
			//report(ost, verbose_level);

			long int rk2, rk3;
			D.division_with_remainder_numerically_with_report(rk0, rk1, rk2, rk3, ost, verbose_level);
			ost << "$" << rk0 << " / " << rk1 << " = " << rk2 << "$ Remainder $" << rk3 << "$\\\\" << endl;


			if (f_v) {
				cout << "finite_field::polynomial_division_with_report after division_with_remainder_numerically_with_report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "finite_field::polynomial_division_with_report written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "finite_field::polynomial_division_with_report done" << endl;
	}

}


void finite_field::RREF_demo(int *A, int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::RREF_demo" << endl;
	}


	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "RREF_example_q%d_%d_%d.tex", q, m, n);
		fname.assign(str);
		snprintf(title, 1000, "RREF example $q=%d$", q);
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					FALSE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "finite_field::RREF_demo before report" << endl;
			}
			RREF_demo2(ost, A, m, n, verbose_level);
			if (f_v) {
				cout << "finite_field::RREF_demo after report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "finite_field::RREF_demo done" << endl;
	}
}

void finite_field::RREF_demo2(std::ostream &ost, int *A, int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *base_cols;
	int i, j, rk;
	latex_interface Li;
	int cnt = 0;

	if (f_v) {
		cout << "finite_field::RREF_demo2" << endl;
	}


	ost << "{\\bf \\Large" << endl;

	ost << endl;
	ost << "\\clearpage" << endl;
	ost << "\\vspace*{\\fill}" << endl;
	ost << endl;

	ost << "\\noindent A matrix over the field ${\\mathbb F}_{" << q << "}$\\\\" << endl;
	ost << "$$" << endl;
	ost << "\\left[" << endl;
	Li.int_matrix_print_tex(ost, A, m, n);
	ost << "\\right]" << endl;
	ost << "$$" << endl;
	cnt++;
	if ((cnt % 3) == 0) {
		ost << endl;
		ost << "\\clearpage" << endl;
		ost << endl;
	}

	base_cols = NEW_int(n);

	i = 0;
	j = 0;
	while (TRUE) {
		if (RREF_search_pivot(A, m, n,
			i, j, base_cols, verbose_level)) {
			ost << "\\noindent  i=" << i << " j=" << j << ", found pivot in column " << base_cols[i] << "\\\\" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, A, m, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;
			cnt++;
			if ((cnt % 3) == 0) {
				ost << endl;
				ost << "\\clearpage" << endl;
				ost << "\\vspace*{\\fill}" << endl;
				ost << endl;
			}


			RREF_make_pivot_one(A, m, n, i, j, base_cols, verbose_level);
			ost << "\\noindent After making pivot 1:\\\\" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, A, m, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;
			cnt++;
			if ((cnt % 3) == 0) {
				ost << endl;
				ost << "\\clearpage" << endl;
				ost << "\\vspace*{\\fill}" << endl;
				ost << endl;
			}


			RREF_elimination_below(A, m, n, i, j, base_cols, verbose_level);
			ost << "\\noindent After elimination below pivot:\\\\" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, A, m, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;
			cnt++;
			if ((cnt % 3) == 0) {
				ost << endl;
				ost << "\\clearpage" << endl;
				ost << "\\vspace*{\\fill}" << endl;
				ost << endl;
			}

		}
		else {
			rk = i;
			ost << "Did not find pivot. The rank is " << rk << "\\\\" << endl;
			break;
		}
	}
	for (i = rk - 1; i >= 0; i--) {
		RREF_elimination_above(A, m, n, i, base_cols, verbose_level);
		ost << "\\noindent After elimination above pivot " << i << ":\\\\" << endl;
		ost << "$$" << endl;
		ost << "\\left[" << endl;
		Li.int_matrix_print_tex(ost, A, m, n);
		ost << "\\right]" << endl;
		ost << "$$" << endl;
		cnt++;
		if ((cnt % 3) == 0) {
			ost << endl;
			ost << "\\clearpage" << endl;
			ost << "\\vspace*{\\fill}" << endl;
			ost << endl;
		}
	}

	Orbiter->Int_vec.print_fully(ost, A, m * n);
	ost << "\\\\" << endl;


	ost << "}" << endl;

	FREE_int(base_cols);

	if (f_v) {
		cout << "finite_field::RREF_demo2 done" << endl;
	}

}

void finite_field::gl_random_matrix(int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	int *M2;
	unipoly_object char_poly;

	if (f_v) {
		cout << "gl_random_matrix" << endl;
		}
	//F.finite_field_init(q, 0 /*verbose_level*/);
	M = NEW_int(k * k);
	M2 = NEW_int(k * k);

	random_invertible_matrix(M, k, verbose_level - 2);

	cout << "Random invertible matrix:" << endl;
	int_matrix_print(M, k, k);


	{
		unipoly_domain U(this);



		U.create_object_by_rank(char_poly, 0, __FILE__, __LINE__, verbose_level);

		U.characteristic_polynomial(M, k, char_poly, verbose_level - 2);

		cout << "The characteristic polynomial is ";
		U.print_object(char_poly, cout);
		cout << endl;

		U.substitute_matrix_in_polynomial(char_poly, M, M2, k, verbose_level);
		cout << "After substitution, the matrix is " << endl;
		int_matrix_print(M2, k, k);

		U.delete_object(char_poly);

	}
	FREE_int(M);
	FREE_int(M2);

}




}}

