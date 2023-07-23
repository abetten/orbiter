/*
 * linear_algebra_global.cpp
 *
 *  Created on: Jan 10, 2022
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace linear_algebra {


linear_algebra_global::linear_algebra_global()
{

}

linear_algebra_global::~linear_algebra_global()
{

}



void linear_algebra_global::Berlekamp_matrix(
		field_theory::finite_field *F,
		std::string &Berlekamp_matrix_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra_global::Berlekamp_matrix" << endl;
	}

	int *data_A;
	int sz_A;


	Get_int_vector_from_label(Berlekamp_matrix_label, data_A, sz_A, verbose_level);


	number_theory::number_theory_domain NT;




	ring_theory::unipoly_domain FX(F);
	ring_theory::unipoly_object A;


	int da = sz_A - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= F->q) {
			data_A[i] = NT.mod(data_A[i], F->q);
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
		cout << "linear_algebra_global::Berlekamp_matrix "
				"before FX.Berlekamp_matrix" << endl;
	}

	{
		FX.Berlekamp_matrix(B, A, verbose_level);
	}

	if (f_v) {
		cout << "linear_algebra_global::Berlekamp_matrix "
				"after FX.Berlekamp_matrix" << endl;
	}

	if (f_v) {
		cout << "B=" << endl;
		Int_matrix_print(B, da, da);
		cout << endl;
	}

	r = F->Linear_algebra->rank_of_matrix(B, da, 0 /* verbose_level */);

	if (f_v) {
		cout << "The matrix B has rank " << r << endl;
	}


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;

		fname = "Berlekamp_matrix_q" + std::to_string(F->q)
						+ "_d" + std::to_string(da)
				+ ".tex";
		title = "Berlekamp Matrix";



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


			if (f_v) {
				cout << "linear_algebra_global::Berlekamp_matrix "
						"before report" << endl;
			}
			//report(ost, verbose_level);

			ost << "\\noindent "
					"Berlekamp matrix: "
					"$q=" << F->q << "$ "
					<< endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			L.int_matrix_print_tex(ost, B, da, da);
			ost << "\\right]" << endl;
			ost << "$$" << endl;


			if (f_v) {
				cout << "linear_algebra_global::Berlekamp_matrix "
						"after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "linear_algebra_global::Berlekamp_matrix "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	FREE_int(B);

	if (f_v) {
		cout << "linear_algebra_global::Berlekamp_matrix done" << endl;
	}
}




void linear_algebra_global::compute_normal_basis(
		field_theory::finite_field *F,
		int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis "
				<< " q=" << F->q << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}


	field_theory::normal_basis *Nor;

	Nor = NEW_OBJECT(field_theory::normal_basis);

	Nor->init(F, d, verbose_level);



	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis "
				"Normal_basis = " << endl;
		Int_matrix_print(Nor->Normal_basis, d, d);
		cout << endl;
	}

	{

		string fname;
		string author;
		string title;
		string extra_praeamble;

		fname = "normal_basis_q" + std::to_string(F->q)
						+ "_d" + std::to_string(d)
				+ ".tex";
		title = "Normal Basis";



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


			if (f_v) {
				cout << "linear_algebra_global::compute_normal_basis "
						"before report" << endl;
			}
			//report(ost, verbose_level);


			Nor->report(ost);
#if 0
			ost << "\\noindent "
					"Normal Basis is in the columns of the following matrix: "
					"$q=" << F->q << "$ "
					<< endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			L.int_matrix_print_tex(ost, Nor->Normal_basis, d, d);
			ost << "\\right]" << endl;
			ost << "$$" << endl;
#endif

			if (f_v) {
				cout << "linear_algebra_global::compute_normal_basis "
						"after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "linear_algebra_global::compute_normal_basis "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis done" << endl;
	}
}

void linear_algebra_global::compute_normal_basis_with_given_polynomial(
		field_theory::finite_field *F,
		std::string &poly_encoded, int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis_with_given_polynomial "
				<< " q=" << F->q << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}


	field_theory::normal_basis *Nor;

	Nor = NEW_OBJECT(field_theory::normal_basis);

	Nor->init_with_polynomial_coded(
			F, poly_encoded, d,
			verbose_level);


	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis_with_given_polynomial "
				"Normal_basis = " << endl;
		Int_matrix_print(Nor->Normal_basis, d, d);
		cout << endl;
	}



	{

		string fname;
		string author;
		string title;
		string extra_praeamble;

		fname = "normal_basis_q" + std::to_string(F->q)
						+ "_d" + std::to_string(d)
						+ "_poly" + poly_encoded
				+ ".tex";
		title = "Normal Basis";



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


			if (f_v) {
				cout << "linear_algebra_global::compute_normal_basis_with_given_polynomial "
						"before report" << endl;
			}
			//report(ost, verbose_level);

			Nor->report(ost);
#if 0
			ost << "\\noindent "
					"Normal Basis is in the columns of the following matrix: "
					"$q=" << F->q << "$ "
					"poly=$" << poly_encoded << "$" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			L.int_matrix_print_tex(ost, Nor->Normal_basis, d, d);
			ost << "\\right]" << endl;
			ost << "$$" << endl;
#endif


			if (f_v) {
				cout << "linear_algebra_global::compute_normal_basis_with_given_polynomial "
						"after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "linear_algebra_global::compute_normal_basis_with_given_polynomial "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis_with_given_polynomial done" << endl;
	}
}


void linear_algebra_global::nullspace(
		field_theory::finite_field *F,
		int *M, int m, int n,
		int *&Nullspace, int &nullspace_m, int &nullspace_n,
		int *&A,
		int &rk_A,
		int *&base_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int *A;
	//int *base_cols;
	int i, rk1;

	if (f_v) {
		cout << "linear_algebra_global::nullspace" << endl;
	}
	A = NEW_int(MAXIMUM(m, n) * n);
	base_cols = NEW_int(n);
	Int_vec_copy(M, A, m * n);

	if (f_v) {
		cout << "linear_algebra_global::do_nullspace "
				"before Linear_algebra->perp_standard" << endl;
	}

	rk_A = F->Linear_algebra->perp_standard(n, m, A, 0 /*verbose_level*/);

	if (f_v) {
		cout << "linear_algebra_global::do_nullspace "
				"after Linear_algebra->perp_standard" << endl;
	}


	if (f_v) {
		cout << "linear_algebra_global::do_nullspace "
				"after perp_standard:" << endl;
		Int_matrix_print(A, n, n);
		cout << "rk=" << rk_A << endl;
	}

	if (f_v) {
		cout << "linear_algebra_global::nullspace "
				"before F->Linear_algebra->Gauss_int" << endl;
	}
	rk1 = F->Linear_algebra->Gauss_int(A + rk_A * n,
		false /* f_special */, true /* f_complete */, base_cols,
		false /* f_P */, NULL /*P*/, n - rk_A, n, n,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "linear_algebra_global::nullspace "
				"after F->Linear_algebra->Gauss_int" << endl;
	}


	if (f_v) {
		cout << "linear_algebra_global::do_nullspace "
				"after RREF" << endl;
		Int_matrix_print(A + rk_A * n, rk1, n);
		cout << "rank of nullspace = " << rk1 << endl;

		cout << "linear_algebra_global::do_nullspace "
				"coefficients:" << endl;
		Int_vec_print_fully(cout, A + rk_A * n, rk1 * n);
		cout << endl;

		l1_interfaces::latex_interface Li;


		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.int_matrix_print_tex(cout, A + rk_A * n, rk1, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;
	}

	nullspace_m = n - rk_A;
	nullspace_n = n;

	Nullspace = NEW_int(nullspace_m * nullspace_n);

	Int_vec_copy(A + rk_A * n, Nullspace, nullspace_m * nullspace_n);


	//FREE_int(A);
	//FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra_global::nullspace done" << endl;
	}

}


void linear_algebra_global::do_nullspace(
		field_theory::finite_field *F,
		std::string &input_matrix,
		int *&Nullspace, int &nullspace_m, int &nullspace_n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *A;
	int *base_cols;
	int rk_A;
	int *M;
	int m, n;

	if (f_v) {
		cout << "linear_algebra_global::do_nullspace" << endl;
	}

	Get_matrix(input_matrix, M, m, n);

	l1_interfaces::latex_interface Li;


	nullspace(
			F,
			M, m, n,
			Nullspace, nullspace_m, nullspace_n,
			A, rk_A, base_cols,
			verbose_level);

#if 0
	if (f_normalize_from_the_left) {
		if (f_v) {
			cout << "linear_algebra_global::do_nullspace "
					"normalizing from the left" << endl;
		}
		for (i = rk; i < n; i++) {
			F->Projective_space_basic->PG_element_normalize_from_front(
					A + i * n, 1, n);
		}

		if (f_v) {
			cout << "linear_algebra_global::do_nullspace "
					"after normalize from the left:" << endl;
			Int_matrix_print(A, n, n);
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
			cout << "linear_algebra_global::do_nullspace "
					"normalizing from the right" << endl;
		}
		for (i = rk; i < n; i++) {
			F->Projective_space_basic->PG_element_normalize(
					A + i * n, 1, n);
		}

		if (f_v) {
			cout << "linear_algebra_global::do_nullspace "
					"after normalize from the right:" << endl;
			Int_matrix_print(A, n, n);
			cout << "rk=" << rk << endl;

			cout << "$$" << endl;
			cout << "\\left[" << endl;
			Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
			cout << "\\right]" << endl;
			cout << "$$" << endl;
		}
	}
#endif


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;

		fname = input_matrix + "_nullspace.tex";
		title = "Right Nullspace";



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


			if (f_v) {
				cout << "linear_algebra_global::do_nullspace "
						"before report" << endl;
			}
			//report(ost, verbose_level);

			ost << "\\noindent Input matrix:" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, M, m, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;

			ost << "The matrix has rank " << rk_A << "\\\\" << endl;

			ost << "RREF:" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, A, rk_A, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;

			ost << "Basis for right Nullspace are the rows of the following matrix:" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, Nullspace, nullspace_m, nullspace_n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;

			ost << "The Nullspace has dimension " << nullspace_m << "\\\\" << endl;
			ost << "Basis columns for the Nullspace are ";
			Int_vec_print(ost, base_cols, nullspace_m);
			ost << "\\\\" << endl;



			if (f_v) {
				cout << "linear_algebra_global::do_nullspace "
						"after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "linear_algebra_global::do_nullspace "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	FREE_int(A);
	FREE_int(base_cols);

	FREE_int(M);

	if (f_v) {
		cout << "linear_algebra_global::do_nullspace done" << endl;
	}
}

void linear_algebra_global::do_RREF(
		field_theory::finite_field *F,
		std::string &input_matrix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *A;
	int *base_cols;
	int rk, i;
	int *M;
	int m, n;
	l1_interfaces::latex_interface Li;

	if (f_v) {
		cout << "linear_algebra_global::do_RREF" << endl;
	}

	Get_matrix(input_matrix, M, m, n);


	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	Int_vec_copy(M, A, m * n);
	if (f_v) {
		cout << "linear_algebra_global::do_RREF "
				"input matrix A of size " << m << " x " << n << endl;
		//Int_matrix_print(A, m, n);
	}

	rk = F->Linear_algebra->Gauss_int(A,
		false /* f_special */, true /* f_complete */, base_cols,
		false /* f_P */, NULL /*P*/, m, n, n,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "linear_algebra_global::do_RREF "
				"after RREF:" << endl;
		Int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;

		cout << "coefficients:" << endl;
		Int_vec_print(cout, A, rk * n);
		cout << endl;

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.int_matrix_print_tex(cout, A, rk, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;
	}


#if 0
	if (f_normalize_from_the_left) {
		if (f_v) {
			cout << "normalizing from the left" << endl;
		}
		for (i = 0; i < rk; i++) {
			F->Projective_space_basic->PG_element_normalize_from_front(
					A + i * n, 1, n);
		}

		if (f_v) {
			cout << "after normalize from the left:" << endl;
			Int_matrix_print(A, rk, n);
			cout << "rk=" << rk << endl;
		}
	}

	if (f_normalize_from_the_right) {
		if (f_v) {
			cout << "normalizing from the right" << endl;
		}
		for (i = 0; i < rk; i++) {
			F->Projective_space_basic->PG_element_normalize(
					A + i * n, 1, n);
		}

		if (f_v) {
			cout << "after normalize from the right:" << endl;
			Int_matrix_print(A, rk, n);
			cout << "rk=" << rk << endl;
		}
	}
#endif

	long int *Row, *Col;
	int *B;
	int j;

	B = NEW_int(rk * n);
	Int_vec_copy(A, B, rk * n);

	Row = NEW_lint(rk);
	Col = NEW_lint(n);
	for (i = 0; i < rk; i++) {
		F->Projective_space_basic->PG_element_rank_modified_lint(
				B + i * n, 1, n, Row[i]);
	}
	Int_vec_copy(A, B, rk * n);
	for (j = 0; j < n; j++) {
		for (i = 0; i < rk; i++) {
			if (B[i * n + j]) {
				break;
			}
		}
		if (i < rk) {
			F->Projective_space_basic->PG_element_rank_modified_lint(
					B + j, n, rk, Col[j]);
		}
		else {
			Col[j] = -1;
		}
	}

	FREE_int(B);

	cout << "The orbiter ranks of the row vectors are: ";
	Lint_vec_print(cout, Row, rk);
	cout << endl;
	cout << "The orbiter ranks of the column vectors are: ";
	Lint_vec_print(cout, Col, n);
	cout << endl;


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;

		fname = input_matrix + "_rref.tex";
		title = "RREF";



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


			if (f_v) {
				cout << "linear_algebra_global::do_RREF "
						"before report" << endl;
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


			ost << "The orbiter ranks of the row vectors are: ";
			Lint_vec_print(ost, Row, rk);
			ost << "\\\\";
			ost << endl;
			ost << "The orbiter ranks of the column vectors are: ";
			Lint_vec_print(ost, Col, n);
			ost << "\\\\";
			ost << endl;




			if (f_v) {
				cout << "linear_algebra_global::do_RREF "
						"after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "linear_algebra_global::do_RREF "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	FREE_int(M);
	FREE_int(A);
	FREE_int(base_cols);

	if (f_v) {
		cout << "linear_algebra_global::do_RREF done" << endl;
	}
}

void linear_algebra_global::RREF_demo(
		field_theory::finite_field *F,
		int *A, int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra_global::RREF_demo" << endl;
	}


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = "RREF_example_q" + std::to_string(F->q) + "_" + std::to_string(m) + "_" + std::to_string(n) + ".tex";
		title = "RREF example $q=" + std::to_string(F->q) + "$";


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
					false /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "linear_algebra_global::RREF_demo "
						"before RREF_with_steps_latex" << endl;
			}
			RREF_with_steps_latex(F, ost, A, m, n, verbose_level);
			if (f_v) {
				cout << "linear_algebra_global::RREF_demo "
						"after RREF_with_steps_latex" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "linear_algebra_global::RREF_demo done" << endl;
	}
}

void linear_algebra_global::RREF_with_steps_latex(
		field_theory::finite_field *F,
		std::ostream &ost, int *A, int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *base_cols;
	int i, j, rk;
	l1_interfaces::latex_interface Li;
	int cnt = 0;

	if (f_v) {
		cout << "linear_algebra_global::RREF_with_steps_latex" << endl;
	}


	ost << "{\\bf \\Large" << endl;

	ost << endl;
	//ost << "\\clearpage" << endl;
	//ost << "\\vspace*{\\fill}" << endl;
	ost << endl;

	ost << "\\noindent A matrix over the field "
			"${\\mathbb F}_{" << F->q << "}$\\\\" << endl;
	ost << "$$" << endl;
	ost << "\\left[" << endl;
	Li.int_matrix_print_tex(ost, A, m, n);
	ost << "\\right]" << endl;
	ost << "$$" << endl;
	cnt++;
	if ((cnt % 3) == 0) {
		ost << endl;
		//ost << "\\clearpage" << endl;
		ost << endl;
	}

	base_cols = NEW_int(n);

	i = 0;
	j = 0;
	while (true) {
		if (F->Linear_algebra->RREF_search_pivot(A, m, n,
			i, j, base_cols, verbose_level)) {
			ost << "\\noindent  Position $(i,j)=(" << i << "," << j << "),$ "
					"found pivot in column " << base_cols[i] << "\\\\" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, A, m, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;
			cnt++;
			if ((cnt % 3) == 0) {
				ost << endl;
				//ost << "\\clearpage" << endl;
				//ost << "\\vspace*{\\fill}" << endl;
				ost << endl;
			}


			F->Linear_algebra->RREF_make_pivot_one(A, m, n, i, j, base_cols, verbose_level);
			ost << "\\noindent After making pivot 1:\\\\" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, A, m, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;
			cnt++;
			if ((cnt % 3) == 0) {
				ost << endl;
				//ost << "\\clearpage" << endl;
				//ost << "\\vspace*{\\fill}" << endl;
				ost << endl;
			}


			F->Linear_algebra->RREF_elimination_below(A, m, n, i, j, base_cols, verbose_level);
			ost << "\\noindent After elimination below pivot:\\\\" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, A, m, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;
			cnt++;
			if ((cnt % 3) == 0) {
				ost << endl;
				//ost << "\\clearpage" << endl;
				//ost << "\\vspace*{\\fill}" << endl;
				ost << endl;
			}

		}
		else {
			rk = i;
			ost << "Did not find pivot. "
					"The rank of the matrix is " << rk << ".\\\\" << endl;
			break;
		}
	}
	for (i = rk - 1; i >= 0; i--) {
		F->Linear_algebra->RREF_elimination_above(A, m, n, i, base_cols, verbose_level);
		ost << "\\noindent After elimination above pivot " << i
				<< " in position (" << i << "," << base_cols[i] << "):\\\\" << endl;
		ost << "$$" << endl;
		ost << "\\left[" << endl;
		Li.int_matrix_print_tex(ost, A, m, n);
		ost << "\\right]" << endl;
		ost << "$$" << endl;
		cnt++;
		if ((cnt % 3) == 0) {
			ost << endl;
			//ost << "\\clearpage" << endl;
			//ost << "\\vspace*{\\fill}" << endl;
			ost << endl;
		}
	}

	Int_vec_print_fully(ost, A, m * n);
	ost << "\\\\" << endl;


	ost << "}" << endl;

	FREE_int(base_cols);

	if (f_v) {
		cout << "linear_algebra_global::RREF_with_steps_latex done" << endl;
	}

}


int linear_algebra_global::reverse_engineer_semilinear_map(
		field_theory::finite_field *F,
		int n,
	int *Elt, int *Mtx, int &frobenius,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int d = n + 1;
	int *v1, *v2, *v1_save;
	int *w1, *w2, *w1_save;
	int h, hh, l, e, frobenius_inv, lambda, rk, c, cv;
	long int i, j;
	int *system;
	int *base_cols;
	number_theory::number_theory_domain NT;


	if (f_v) {
		cout << "linear_algebra_global::reverse_engineer_semilinear_map" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "d=" << d << endl;
	}
	//F = P->F;
	//q = F->q;

	v1 = NEW_int(d);
	v2 = NEW_int(d);
	v1_save = NEW_int(d);
	w1 = NEW_int(d);
	w2 = NEW_int(d);
	w1_save = NEW_int(d);



	if (f_v) {
		cout << "linear_algebra_global::reverse_engineer_semilinear_map "
				"mapping unit vectors" << endl;
	}
	for (e = 0; e < d; e++) {
		// map the unit vector e_e
		// (with a one in position e and zeros elsewhere):
		for (h = 0; h < d; h++) {
			if (h == e) {
				v1[h] = 1;
			}
			else {
				v1[h] = 0;
			}
		}
		Int_vec_copy(v1, v1_save, d);
		F->Projective_space_basic->PG_element_rank_modified_lint(
				v1, 1, n + 1, i);
		//rank_point(v1);
			// Now, the value of i should be equal to e.
		//j = element_image_of(i, Elt, 0);
		j = Elt[i];
		//unrank_point(v2, j);
		F->Projective_space_basic->PG_element_unrank_modified_lint(
				v2, 1, n + 1, j);
		if (f_v) {
			cout << "linear_algebra_global::reverse_engineer_semilinear_map "
					"unit vector " << e << " has rank " << i
					<< " and maps to " << j << endl;
		}

#if 0
		if (f_vv) {
			print_from_to(d, i, j, v1_save, v2);
		}
#endif


		Int_vec_copy(v2, Mtx + e * d, d);
	}

	if (f_vv) {
		cout << "Mtx (before scaling):" << endl;
		Int_vec_print_integer_matrix_width(
				cout, Mtx, d, d, d, F->log10_of_q);
		cout << endl;
	}

	// map the vector (1,1,...,1):
	if (f_v) {
		cout << "linear_algebra_global::reverse_engineer_semilinear_map "
				"mapping the all-one vector"
				<< endl;
	}
	for (h = 0; h < d; h++) {
		v1[h] = 1;
	}
	Int_vec_copy(v1, v1_save, d);
	//i = rank_point(v1);

	F->Projective_space_basic->PG_element_rank_modified_lint(
			v1, 1, n + 1, i);

	//j = element_image_of(i, Elt, 0);
	j = Elt[i];
	//unrank_point(v2, j);
	F->Projective_space_basic->PG_element_unrank_modified_lint(
			v2, 1, n + 1, j);
	if (f_v) {
		cout << "linear_algebra_global::reverse_engineer_semilinear_map "
				"the all one vector has rank " << i
				<< " and maps to " << j << endl;
	}

#if 0
	if (f_vv) {
		print_from_to(d, i, j, v1_save, v2);
	}
#endif

	system = NEW_int(d * (d + 1));
	base_cols = NEW_int(d + 1);
	// coefficient matrix:
	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			system[i * (d + 1) + j] = Mtx[j * d + i];
		}
	}
	// RHS:
	for (i = 0; i < d; i++) {
		system[i * (d + 1) + d] = v2[i];
	}
	if (f_vv) {
		cout << "linear_algebra_global::reverse_engineer_semilinear_map "
				"linear system:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, system,
				d, d + 1, d + 1, F->log10_of_q);
		cout << endl;
	}
	rk = F->Linear_algebra->Gauss_simple(
			system, d, d + 1, base_cols,
			verbose_level - 4);
	if (rk != d) {
		cout << "linear_algebra_global::reverse_engineer_semilinear_map "
				"rk != d, fatal" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "linear_algebra_global::reverse_engineer_semilinear_map "
				"after Gauss_simple:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, system,
				d, d + 1, d + 1, F->log10_of_q);
		cout << endl;
	}
	for (i = 0; i < d; i++) {
		c = system[i * (d + 1) + d];
		if (c == 0) {
			cout << "linear_algebra_global::reverse_engineer_semilinear_map "
					"the input matrix does not have full rank" << endl;
			exit(1);
		}
		for (j = 0; j < d; j++) {
			Mtx[i * d + j] = F->mult(c, Mtx[i * d + j]);
		}
	}

	if (f_vv) {
		cout << "Mtx (after scaling):" << endl;
		Int_vec_print_integer_matrix_width(
				cout, Mtx, d, d, d, F->log10_of_q);
		cout << endl;
	}



	frobenius = 0;
	if (F->q != F->p) {

		// figure out the frobenius:
		if (f_v) {
			cout << "linear_algebra_global::reverse_engineer_semilinear_map "
					"figuring out the frobenius" << endl;
		}


		// create the vector (1,p,0,...,0)

		for (h = 0; h < d; h++) {
			if (h == 0) {
				v1[h] = 1;
			}
			else if (h == 1) {
				v1[h] = F->p;
			}
			else {
				v1[h] = 0;
			}
		}
		Int_vec_copy(v1, v1_save, d);
		//i = rank_point(v1);

		F->Projective_space_basic->PG_element_rank_modified_lint(
				v1, 1, n + 1, i);

		//j = element_image_of(i, Elt, 0);
		j = Elt[i];
		//unrank_point(v2, j);
		F->Projective_space_basic->PG_element_unrank_modified_lint(
				v2, 1, n + 1, j);


#if 0
		if (f_vv) {
			print_from_to(d, i, j, v1_save, v2);
		}
#endif


		// coefficient matrix:
		for (i = 0; i < d; i++) {
			for (j = 0; j < 2; j++) {
				system[i * 3 + j] = Mtx[j * d + i];
			}
		}
		// RHS:
		for (i = 0; i < d; i++) {
			system[i * 3 + 2] = v2[i];
		}
		rk = F->Linear_algebra->Gauss_simple(
				system,
				d, 3, base_cols, verbose_level - 4);
		if (rk != 2) {
			cout << "rk != 2, fatal" << endl;
			exit(1);
		}
		if (f_vv) {
			cout << "after Gauss_simple:" << endl;
			Int_vec_print_integer_matrix_width(cout,
					system, 2, 3, 3, F->log10_of_q);
			cout << endl;
		}

		c = system[0 * 3 + 2];
		if (c != 1) {
			cv = F->inverse(c);
			for (hh = 0; hh < 2; hh++) {
				system[hh * 3 + 2] = F->mult(
						cv, system[hh * 3 + 2]);
			}
		}
		if (f_vv) {
			cout << "after scaling the last column:" << endl;
			Int_vec_print_integer_matrix_width(cout,
					system, 2, 3, 3, F->log10_of_q);
			cout << endl;
		}
		lambda = system[1 * 3 + 2];
		if (f_vv) {
			cout << "lambda=" << lambda << endl;
		}


		l = F->log_alpha(lambda);
		if (f_vv) {
			cout << "l=" << l << endl;
		}
		for (i = 0; i < F->e; i++) {
			if (NT.i_power_j(F->p, i) == l) {
				frobenius = i;
				break;
			}
		}
		if (i == F->e) {
			cout << "linear_algebra_global::reverse_engineer_semilinear_map "
					"problem figuring out the Frobenius" << endl;
			exit(1);
		}

		frobenius_inv = (F->e - frobenius) % F->e;
		if (f_vv) {
			cout << "frobenius = " << frobenius << endl;
			cout << "frobenius_inv = " << frobenius_inv << endl;
		}
		for (hh = 0; hh < d * d; hh++) {
			Mtx[hh] = F->frobenius_power(Mtx[hh], frobenius_inv);
		}


	}
	else {
		frobenius = 0;
		frobenius_inv = 0;
	}


	if (f_v) {
		cout << "linear_algebra_global::reverse_engineer_semilinear_map "
				"we found the following map" << endl;
		cout << "Mtx:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				Mtx, d, d, d, F->log10_of_q);
		cout << endl;
		cout << "frobenius = " << frobenius << endl;
		cout << "frobenius_inv = " << frobenius_inv << endl;
	}



	FREE_int(v1);
	FREE_int(v2);
	FREE_int(v1_save);
	FREE_int(w1);
	FREE_int(w2);
	FREE_int(w1_save);
	FREE_int(system);
	FREE_int(base_cols);

	if (f_v) {
		cout << "linear_algebra_global::reverse_engineer_semilinear_map done" << endl;
	}

	return true;
}





}}}

