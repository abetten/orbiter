/*
 * create_BCH_code.cpp
 *
 *  Created on: Jan 13, 2022
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {
namespace coding_theory {

create_BCH_code::create_BCH_code()
{
	n = 0;
	d = 0;
	F = NULL;

	Nth = NULL;

	P = NULL;

	Selection = NULL;
	Sel = NULL;
	nb_sel = 0;

	degree = 0;
	k = 0;
	Genma = NULL;
	generator_polynomial = NULL;
}

create_BCH_code::~create_BCH_code()
{
}

void create_BCH_code::init(field_theory::finite_field *F, int n, int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_BCH_code::init" << endl;
	}

	coding_theory_domain Codes;

	create_BCH_code::n = n;
	create_BCH_code::d = d;
	create_BCH_code::F = F;

	Nth = NEW_OBJECT(field_theory::nth_roots);

	Nth->init(F, n, verbose_level);

	int i, j;


	Selection = NEW_int(Nth->Cyc->S->nb_sets);
	Sel = NEW_int(Nth->Cyc->S->nb_sets);

	for (i = 0; i < Nth->Cyc->S->nb_sets; i++) {
		Selection[i] = FALSE;
	}

	for (i = 0; i < d - 1; i++) {
		j = Nth->Cyc->Index[(1 + i) % n];
		Selection[j] = TRUE;
	}

	nb_sel = 0;
	for (i = 0; i < Nth->Cyc->S->nb_sets; i++) {
		if (Selection[i]) {
			Sel[nb_sel++] = i;
		}
	}

	if (f_v) {
		cout << "coding_theory_domain::make_BCH_code Sel=";
		Orbiter->Int_vec->print(cout, Sel, nb_sel);
		cout << endl;
	}

	ring_theory::unipoly_object Q;

	P = NEW_OBJECT(ring_theory::unipoly_object);
	Nth->FX->create_object_by_rank(*P, 1, __FILE__, __LINE__, 0 /*verbose_level*/);
	Nth->FX->create_object_by_rank(Q, 1, __FILE__, __LINE__, 0 /*verbose_level*/);

	for (i = 0; i < nb_sel; i++) {

		j = Sel[i];

		if (f_v) {
			cout << "coding_theory_domain::make_BCH_code P=";
			Nth->FX->print_object(*P, cout);
			cout << endl;
			cout << "j=" << j << endl;
			Nth->FX->print_object(Nth->generator_Fq[j], cout);
			cout << endl;
		}
		Nth->FX->mult(*P, Nth->generator_Fq[j], Q, verbose_level);
		if (f_v) {
			cout << "coding_theory_domain::make_BCH_code Q=";
			Nth->FX->print_object(Q, cout);
			cout << endl;
		}
		Nth->FX->assign(Q, *P, 0 /* verbose_level */);
	}


	degree = Nth->FX->degree(*P);
	generator_polynomial = NEW_int(degree + 1);
	for (i = 0; i <= degree; i++) {
		generator_polynomial[i] = Nth->FX->s_i(*P, i);
	}

	Codes.generator_matrix_cyclic_code(n,
				degree, generator_polynomial, Genma);

	k = n - degree;


	if (f_v) {
		cout << "create_BCH_code::init done" << endl;
	}
}


void create_BCH_code::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	string label;
	coding_theory_domain Codes;
	latex_interface Li;

	if (f_v) {
		cout << "create_BCH_code::report" << endl;
	}


	ost << "\\noindent" << endl;
	ost << "BCH-code:\\\\" << endl;
	ost << "$n=" << n << ",$ $k=" << k << ",$ $d_0=" << d << ",$ $q=" << F->q << ",$ \\\\" << endl;
	ost << "$g(x) = ";
	for (i = 0; i < nb_sel; i++) {
		ost << "m_{" << Sel[i] << "}";
	}
	ost << "=" << endl;
	//ost << "$" << endl;
	Nth->FX->print_object(*P, ost);
	ost << "$" << endl;
	ost << "\\\\" << endl;


	ost << "\\noindent" << endl;
	ost << "Chosen cyclotomic sets:\\\\" << endl;
	Nth->Cyc->print_latex_with_selection(ost, Sel, nb_sel);

	ost << "\\bigskip" << endl;


	ost << "The generator polynomial has degree " << degree << endl;

	ost << "\\begin{verbatim}" << endl;
	ost << "-dense \"";
	Nth->FX->print_object_dense(*P, ost);
	ost << "\"" << endl;
	ost << endl;

	ost << "-sparse \"";
	Nth->FX->print_object_sparse(*P, ost);
	ost << "\"" << endl;
	ost << endl;
	ost << "\\end{verbatim}" << endl;


	if (n < 22) {

		ost << "The generator matrix is:" << endl;
		ost << "$$" << endl;
		ost << "\\left[" << endl;
		Li.int_matrix_print_tex(ost, Genma, k, n);
		ost << "\\right]" << endl;
		ost << "$$" << endl;

	}

	if (f_v) {
		cout << "create_BCH_code::report done" << endl;
	}
}

}}}



