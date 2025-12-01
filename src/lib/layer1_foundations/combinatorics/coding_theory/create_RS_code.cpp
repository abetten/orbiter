/*
 * create_RS_code.cpp
 *
 *  Created on: Dec 9, 2022
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace coding_theory {

create_RS_code::create_RS_code()
{
	Record_birth();
	n = 0;
	d = 0;
	F = NULL;

	FX = NULL;

	P = NULL;

	degree = 0;
	k = 0;
	Genma = NULL;
	generator_polynomial = NULL;
}

create_RS_code::~create_RS_code()
{
	Record_death();
}


void create_RS_code::init(
		algebra::field_theory::finite_field *F,
		int n, int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_RS_code::init" << endl;
	}
	if (n != F->q - 1) {
		cout << "create_RS_code::init n is not equal to q - 1" << endl;
		exit(1);
	}

	cyclic_codes Cyclic_codes;

	create_RS_code::n = n;
	create_RS_code::d = d;
	create_RS_code::F = F;

	FX = NEW_OBJECT(algebra::ring_theory::unipoly_domain);

	FX->init_basic(F, verbose_level);

	int i;
	algebra::ring_theory::unipoly_object Q, M;

	P = NEW_OBJECT(algebra::ring_theory::unipoly_object);
	FX->create_object_by_rank(*P, 1, 0 /*verbose_level*/);
	FX->create_object_by_rank(Q, 1, 0 /*verbose_level*/);

	for (i = 1; i < d; i++) {

		int coeff[2];

		coeff[0] = F->negate(F->alpha_power(i));
		coeff[1] = 1;
		FX->create_object_of_degree_with_coefficients(M, 1, coeff);

		if (f_v) {
			cout << "create_RS_code::init i=" << i << endl;
			cout << "create_RS_code::init P=";
			FX->print_object(*P, cout);
			cout << endl;
			cout << "create_RS_code::init M=";
			FX->print_object(M, cout);
			cout << endl;
		}
		FX->mult(*P, M, Q, verbose_level);
		if (f_v) {
			cout << "create_RS_code::init Q=";
			FX->print_object(Q, cout);
			cout << endl;
		}
		FX->assign(Q, *P, 0 /* verbose_level */);
	}


	degree = FX->recalculate_degree(*P);
	generator_polynomial = NEW_int(degree + 1);
	for (i = 0; i <= degree; i++) {
		generator_polynomial[i] = FX->s_i(*P, i);
	}

	Cyclic_codes.generator_matrix_cyclic_code(n,
				degree, generator_polynomial, Genma);

	k = n - degree;


	if (f_v) {
		cout << "create_RS_code::init done" << endl;
	}
}

void create_RS_code::do_report(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i;
	string label;
	coding_theory_domain Codes;
	other::l1_interfaces::latex_interface Li;

	if (f_v) {
		cout << "create_RS_code::do_report" << endl;
	}


	other::orbiter_kernel_system::file_io Fio;
	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = "RS_codes_q" + std::to_string(F->q) + "_n" + std::to_string(n) + "_d" + std::to_string(d);
		title = "RS codes";



		{
			ofstream ost(fname);
			algebra::number_theory::number_theory_domain NT;


			other::l1_interfaces::latex_interface L;

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


			report(ost, verbose_level);

			L.foot(ost);


		}

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	}

	if (f_v) {
		cout << "create_RS_code::do_report done" << endl;
	}
}

void create_RS_code::report(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	string label;
	coding_theory_domain Codes;
	other::l1_interfaces::latex_interface Li;

	if (f_v) {
		cout << "create_RS_code::report" << endl;
	}


	ost << "\\noindent" << endl;
	ost << "RS-code:\\\\" << endl;
	ost << "$n=" << n << ",$ $k=" << k << ",$ $d_0=" << d << ",$ $q=" << F->q << ",$ \\\\" << endl;
	ost << "$g(x) = ";
	for (i = 1; i < d; i++) {
		ost << "m_{" << i << "}";
	}
	ost << "=" << endl;
	//ost << "$" << endl;

	FX->print_object(*P, ost);
	ost << "$" << endl;
	ost << "\\\\" << endl;

#if 0
	ost << "\\noindent" << endl;
	ost << "Chosen cyclotomic sets:\\\\" << endl;
	Nth->Cyc->print_latex_with_selection(ost, Sel, nb_sel);
#endif

	ost << "\\bigskip" << endl;


	ost << "The generator polynomial has degree " << degree << endl;

	ost << "\\begin{verbatim}" << endl;
	ost << "-dense \"";
	FX->print_object_dense(*P, ost);
	ost << "\"" << endl;
	ost << endl;

	ost << "-sparse \"";
	FX->print_object_sparse(*P, ost);
	ost << "\"" << endl;
	ost << endl;
	ost << "\\end{verbatim}" << endl;


	if (n < 100) {

		ost << "The generator matrix is:" << endl;
		ost << "$$" << endl;
		ost << "\\left[" << endl;
		Li.int_matrix_print_tex(ost, Genma, k, n);
		ost << "\\right]" << endl;
		ost << "$$" << endl;

		ost << "The generator matrix is:" << endl;
		ost << "\\begin{verbatim}" << endl;
		Int_matrix_print_ost(ost, Genma, k, n);
		ost << "\\end{verbatim}" << endl;

		ost << "The generator matrix as a makefile variable is:" << endl;
		ost << "\\begin{verbatim}" << endl;
		ost << "CODE_RS_F" << F->q << "_N" << n << "_K" << k << "_D" << d << "_GENMA";
		other::orbiter_kernel_system::Orbiter->Int_vec->matrix_print_makefile_style_ost(ost, Genma, k, n);
		ost << "\\end{verbatim}" << endl;
	}
	else {
		ost << "The generator matrix is too big to print.\\\\" << endl;
	}

	if (f_v) {
		cout << "create_RS_code::report done" << endl;
	}
}





}}}}



