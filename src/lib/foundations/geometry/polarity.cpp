/*
 * polarity.cpp
 *
 *  Created on: Oct 18, 2021
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


polarity::polarity()
{
	P = NULL;
	Point_to_hyperplane = NULL;
	Hyperplane_to_point = NULL;
}

polarity::~polarity()
{
	if (Point_to_hyperplane) {
		FREE_int(Point_to_hyperplane);
	}
	if (Hyperplane_to_point) {
		FREE_int(Hyperplane_to_point);
	}
}

void polarity::init_standard_polarity(projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 1);
	int i, j;
	int *A;
	int a;
	int n, d;
	long int N_points;

	if (f_v) {
		cout << "polarity::init_standard_polarity" << endl;
	}
	polarity::P = P;
	n = P->n;
	d = n + 1;
	N_points = P->N_points;

	Point_to_hyperplane = NEW_int(N_points);
	Hyperplane_to_point = NEW_int(N_points);

	A = NEW_int(d * d);

	for (i = 0; i < P->N_points; i++) {
		P->Grass_hyperplanes->unrank_lint(i, 0 /*verbose_level - 4*/);
		for (j = 0; j < n * d; j++) {
			A[j] = P->Grass_hyperplanes->M[j];
		}
		if (f_vv) {
			cout << "hyperplane " << i << ":" << endl;
			Orbiter->Int_vec.print_integer_matrix_width(cout,
				A, n, d, d,
				P->F->log10_of_q + 1);
		}
		P->F->perp_standard(d, n, A, 0);
		if (FALSE) {
			Orbiter->Int_vec.print_integer_matrix_width(cout,
				A, d, d, d,
				P->F->log10_of_q + 1);
		}
		P->F->PG_element_rank_modified(A + n * d, 1, d, a);
		if (f_vv) {
			cout << "hyperplane " << i << " is perp of point ";
			Orbiter->Int_vec.print(cout, A + n * d, d);
			cout << " = " << a << endl;
		}
		Point_to_hyperplane[a] = i;
		Hyperplane_to_point[i] = a;
	}
	if (FALSE /* f_vv */) {
		cout << "i : pt_to_hyperplane[i] : hyperplane_to_pt[i]" << endl;
		for (i = 0; i < N_points; i++) {
			cout << setw(4) << i << " "
				<< setw(4) << Point_to_hyperplane[i] << " "
				<< setw(4) << Hyperplane_to_point[i] << endl;
		}
	}
	FREE_int(A);
	if (f_v) {
		cout << "polarity::init_standard_polarity done" << endl;
	}
}

void polarity::init_general_polarity(projective_space *P, int *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 1);
	int i;
	int *v;
	int *A;
	int a;
	int n, d;
	long int N_points;

	if (f_v) {
		cout << "polarity::init_general_polarity" << endl;
	}
	polarity::P = P;
	n = P->n;
	d = n + 1;
	N_points = P->N_points;

	Point_to_hyperplane = NEW_int(N_points);
	Hyperplane_to_point = NEW_int(N_points);

	v = NEW_int(d);
	A = NEW_int(d * d);

	for (i = 0; i < P->N_points; i++) {

		P->F->PG_element_unrank_modified(v, 1, d, i);


		P->F->mult_matrix_matrix(v, Mtx,
				A, 1, d, d, 0 /* verbose_level*/);


		if (f_vv) {
			cout << "point " << i << " * Mtx = " << endl;
			Orbiter->Int_vec.print_integer_matrix_width(cout,
				A, 1, d, d,
				P->F->log10_of_q + 1);
		}
		P->F->perp_standard(d, 1, A, 0);
		if (FALSE) {
			Orbiter->Int_vec.print_integer_matrix_width(cout,
				A, d, d, d,
				P->F->log10_of_q + 1);
		}
		a = P->Grass_hyperplanes->rank_lint_here(A + d, 0 /*verbose_level - 4*/);
		if (f_vv) {
			cout << "hyperplane " << i << " is perp of point ";
			Orbiter->Int_vec.print(cout, A + 2 * d, d);
			cout << " = " << a << endl;
		}
		Point_to_hyperplane[i] = a;
		Hyperplane_to_point[a] = i;
	}
	if (FALSE /* f_vv */) {
		cout << "i : pt_to_hyperplane[i] : hyperplane_to_pt[i]" << endl;
		for (i = 0; i < N_points; i++) {
			cout << setw(4) << i << " "
				<< setw(4) << Point_to_hyperplane[i] << " "
				<< setw(4) << Hyperplane_to_point[i] << endl;
		}
	}
	FREE_int(v);
	FREE_int(A);
	if (f_v) {
		cout << "polarity::init_general_polarity done" << endl;
	}
}

void polarity::init_reversal_polarity(projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Mtx;
	int n, d, i;

	if (f_v) {
		cout << "polarity::init_reversal_polarity" << endl;
	}
	polarity::P = P;
	n = P->n;
	d = n + 1;

	Mtx = NEW_int(d * d);
	Orbiter->Int_vec.zero(Mtx, d * d);

	for (i = 0; i < d; i++) {
		Mtx[i * d + d - 1 - i] = 1;
	}

	if (f_v) {
		cout << "polarity::init_reversal_polarity before init_general_polarity" << endl;
	}

	init_general_polarity(P, Mtx, verbose_level);

	if (f_v) {
		cout << "polarity::init_reversal_polarity after init_general_polarity" << endl;
	}

	FREE_int(Mtx);

	if (f_v) {
		cout << "polarity::init_reversal_polarity done" << endl;
	}
}

void polarity::report(std::ostream &f)
{
	int i;

	//f << "Polarity point $\\leftrightarrow$ hyperplane:\\\\" << endl;
	f << "\\begin{multicols}{4}" << endl;
	for (i = 0; i < P->N_points; i++) {
		f << "$" << i << " \\leftrightarrow " << Point_to_hyperplane[i] << "$\\\\" << endl;
	}
	f << "\\end{multicols}" << endl;
	f << "\\clearpage" << endl << endl;
}

}}

