/*
 * linear_algebra_global.cpp
 *
 *  Created on: Jan 10, 2022
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {
namespace linear_algebra {


linear_algebra_global::linear_algebra_global()
{

}

linear_algebra_global::~linear_algebra_global()
{

}



void linear_algebra_global::Berlekamp_matrix(
		field_theory::finite_field *F,
		std::string &Berlekamp_matrix_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra_global::Berlekamp_matrix" << endl;
	}

	int *data_A;
	int sz_A;


	Orbiter->get_vector_from_label(Berlekamp_matrix_coeffs, data_A, sz_A, verbose_level);


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
		cout << "linear_algebra_global::Berlekamp_matrix before FX.Berlekamp_matrix" << endl;
	}

	{
		FX.Berlekamp_matrix(B, A, verbose_level);
	}

	if (f_v) {
		cout << "linear_algebra_global::Berlekamp_matrix after FX.Berlekamp_matrix" << endl;
	}

	if (f_v) {
		cout << "B=" << endl;
		Orbiter->Int_vec->matrix_print(B, da, da);
		cout << endl;
	}

	r = F->Linear_algebra->rank_of_matrix(B, da, 0 /* verbose_level */);

	if (f_v) {
		cout << "The matrix B has rank " << r << endl;
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
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis "
				<< " q=" << F->q << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}


	ring_theory::unipoly_domain FX(F);

	string poly;
	knowledge_base K;

	K.get_primitive_polynomial(poly, F->q, d, 0 /* verbose_level */);

	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis "
				"chosen irreducible polynomial is " << poly << endl;
	}

	ring_theory::unipoly_object m;
	ring_theory::unipoly_object g;
	ring_theory::unipoly_object minpol;
	combinatorics::combinatorics_domain Combi;


	FX.create_object_by_rank_string(m, poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis "
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
		cout << "linear_algebra_global::compute_normal_basis "
				"before FX.Frobenius_matrix" << endl;
	}
	FX.Frobenius_matrix(Frobenius, m, verbose_level - 2);
	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis "
				"Frobenius_matrix = " << endl;
		Orbiter->Int_vec->matrix_print(Frobenius, d, d);
		cout << endl;
	}

	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis "
				"before compute_normal_basis" << endl;
	}

	FX.compute_normal_basis(d, Normal_basis, Frobenius, verbose_level - 1);

	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis "
				"Normal_basis = " << endl;
		Orbiter->Int_vec->matrix_print(Normal_basis, d, d);
		cout << endl;
	}

	if (f_v) {
		cout << "linear_algebra_global::compute_normal_basis done" << endl;
	}
}


void linear_algebra_global::do_nullspace(
		field_theory::finite_field *F,
		int *M, int m, int n,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *A;
	int *base_cols;
	int rk, i, rk1;

	latex_interface Li;

	if (f_v) {
		cout << "linear_algebra_global::do_nullspace" << endl;
	}


	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	Orbiter->Int_vec->copy(M, A, m * n);

	if (f_v) {
		cout << "linear_algebra_global::do_nullspace before Linear_algebra->perp_standard" << endl;
	}

	rk = F->Linear_algebra->perp_standard(n, m, A, 0 /*verbose_level*/);

	if (f_v) {
		cout << "linear_algebra_global::do_nullspace after Linear_algebra->perp_standard" << endl;
	}


	if (f_v) {
		cout << "linear_algebra_global::do_nullspace after perp_standard:" << endl;
		Orbiter->Int_vec->matrix_print(A, n, n);
		cout << "rk=" << rk << endl;
	}

	rk1 = F->Linear_algebra->Gauss_int(A + rk * n,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, n - rk, n, n,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "linear_algebra_global::do_nullspace after RREF" << endl;
		Orbiter->Int_vec->matrix_print(A + rk * n, rk1, n);
		cout << "rank of nullspace = " << rk1 << endl;

		cout << "linear_algebra_global::do_nullspace coefficients:" << endl;
		Orbiter->Int_vec->print_fully(cout, A + rk * n, rk1 * n);
		cout << endl;

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;
	}

	if (f_normalize_from_the_left) {
		if (f_v) {
			cout << "linear_algebra_global::do_nullspace normalizing from the left" << endl;
		}
		for (i = rk; i < n; i++) {
			F->PG_element_normalize_from_front(A + i * n, 1, n);
		}

		if (f_v) {
			cout << "linear_algebra_global::do_nullspace after normalize from the left:" << endl;
			Orbiter->Int_vec->matrix_print(A, n, n);
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
			cout << "linear_algebra_global::do_nullspace normalizing from the right" << endl;
		}
		for (i = rk; i < n; i++) {
			F->PG_element_normalize(A + i * n, 1, n);
		}

		if (f_v) {
			cout << "linear_algebra_global::do_nullspace after normalize from the right:" << endl;
			Orbiter->Int_vec->matrix_print(A, n, n);
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
				cout << "linear_algebra_global::do_nullspace before report" << endl;
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
				cout << "linear_algebra_global::do_nullspace after report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "linear_algebra_global::do_nullspace written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	FREE_int(A);
	FREE_int(base_cols);

	if (f_v) {
		cout << "linear_algebra_global::do_nullspace done" << endl;
	}
}

void linear_algebra_global::do_RREF(
		field_theory::finite_field *F,
		int *M, int m, int n,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *A;
	int *base_cols;
	int rk, i;
	latex_interface Li;

	if (f_v) {
		cout << "linear_algebra_global::do_RREF" << endl;
	}



	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	Orbiter->Int_vec->copy(M, A, m * n);
	if (f_v) {
		cout << "linear_algebra_global::do_RREF input matrix A:" << endl;
		Orbiter->Int_vec->matrix_print(A, m, n);
	}

	rk = F->Linear_algebra->Gauss_int(A,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, m, n, n,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "linear_algebra_global::do_RREF after RREF:" << endl;
		Orbiter->Int_vec->matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;

		cout << "coefficients:" << endl;
		Orbiter->Int_vec->print(cout, A, rk * n);
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
			F->PG_element_normalize_from_front(A + i * n, 1, n);
		}

		if (f_v) {
			cout << "after normalize from the left:" << endl;
			Orbiter->Int_vec->matrix_print(A, rk, n);
			cout << "rk=" << rk << endl;
		}
	}

	if (f_normalize_from_the_right) {
		if (f_v) {
			cout << "normalizing from the right" << endl;
		}
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize(A + i * n, 1, n);
		}

		if (f_v) {
			cout << "after normalize from the right:" << endl;
			Orbiter->Int_vec->matrix_print(A, rk, n);
			cout << "rk=" << rk << endl;
		}
	}


	Orbiter->Int_vec->copy(M, A, m * n);

	RREF_demo(F, A, m, n, verbose_level);



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
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "RREF_example_q%d_%d_%d.tex", F->q, m, n);
		fname.assign(str);
		snprintf(title, 1000, "RREF example $q=%d$", F->q);
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
				cout << "linear_algebra_global::RREF_demo before RREF_with_steps_latex" << endl;
			}
			RREF_with_steps_latex(F, ost, A, m, n, verbose_level);
			if (f_v) {
				cout << "linear_algebra_global::RREF_demo after RREF_with_steps_latex" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

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
	latex_interface Li;
	int cnt = 0;

	if (f_v) {
		cout << "linear_algebra_global::RREF_with_steps_latex" << endl;
	}


	ost << "{\\bf \\Large" << endl;

	ost << endl;
	//ost << "\\clearpage" << endl;
	//ost << "\\vspace*{\\fill}" << endl;
	ost << endl;

	ost << "\\noindent A matrix over the field ${\\mathbb F}_{" << F->q << "}$\\\\" << endl;
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
	while (TRUE) {
		if (F->Linear_algebra->RREF_search_pivot(A, m, n,
			i, j, base_cols, verbose_level)) {
			ost << "\\noindent  Position $(i,j)=(" << i << "," << j << "),$ found pivot in column " << base_cols[i] << "\\\\" << endl;
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
			ost << "Did not find pivot. The rank of the matrix is " << rk << ".\\\\" << endl;
			break;
		}
	}
	for (i = rk - 1; i >= 0; i--) {
		F->Linear_algebra->RREF_elimination_above(A, m, n, i, base_cols, verbose_level);
		ost << "\\noindent After elimination above pivot " << i << " in position (" << i << "," << base_cols[i] << "):\\\\" << endl;
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

	Orbiter->Int_vec->print_fully(ost, A, m * n);
	ost << "\\\\" << endl;


	ost << "}" << endl;

	FREE_int(base_cols);

	if (f_v) {
		cout << "linear_algebra_global::RREF_with_steps_latex done" << endl;
	}

}



}}}

