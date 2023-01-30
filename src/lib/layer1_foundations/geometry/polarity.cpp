/*
 * polarity.cpp
 *
 *  Created on: Oct 18, 2021
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {


polarity::polarity()
{
	P = NULL;
	Point_to_hyperplane = NULL;
	Hyperplane_to_point = NULL;
	f_absolute = NULL;
	Line_to_line = NULL;
	f_absolute_line = NULL;
	nb_absolute_lines = 0;
	nb_self_dual_lines = 0;
}

polarity::~polarity()
{
	if (Point_to_hyperplane) {
		FREE_int(Point_to_hyperplane);
	}
	if (Hyperplane_to_point) {
		FREE_int(Hyperplane_to_point);
	}
	if (f_absolute) {
		FREE_int(f_absolute);
	}
	if (Line_to_line) {
		FREE_lint(Line_to_line);
	}
	if (f_absolute_line) {
		FREE_int(f_absolute_line);
	}
}

void polarity::init_standard_polarity(projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 1);
	long int i, j;
	int *A;
	long int a;
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

	if (d == 4) {
		Line_to_line = NEW_lint(P->N_lines);
	}
	A = NEW_int(d * d);

	for (i = 0; i < P->N_points; i++) {
		P->Grass_hyperplanes->unrank_lint(i, 0 /*verbose_level - 4*/);
		for (j = 0; j < n * d; j++) {
			A[j] = P->Grass_hyperplanes->M[j];
		}
		if (f_vv) {
			cout << "hyperplane " << i << ":" << endl;
			Int_vec_print_integer_matrix_width(cout,
				A, n, d, d,
				P->F->log10_of_q + 1);
		}
		P->F->Linear_algebra->perp_standard(d, n, A, 0);
		if (FALSE) {
			Int_vec_print_integer_matrix_width(cout,
				A, d, d, d,
				P->F->log10_of_q + 1);
		}
		P->F->Projective_space_basic->PG_element_rank_modified_lint(
				A + n * d, 1, d, a);
		if (f_vv) {
			cout << "hyperplane " << i << " is perp of point ";
			Int_vec_print(cout, A + n * d, d);
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


	if (d == 4) {
		for (i = 0; i < P->N_lines; i++) {
			P->Grass_lines->unrank_lint_here(A, i, 0 /*verbose_level - 4*/);
			if (f_vv) {
				cout << "line " << i << ":" << endl;
				Int_vec_print_integer_matrix_width(cout,
					A, 2, d, d,
					P->F->log10_of_q + 1);
			}
			P->F->Linear_algebra->perp_standard(d, 2, A, 0);
			if (FALSE) {
				Int_vec_print_integer_matrix_width(cout,
					A, d, d, d,
					P->F->log10_of_q + 1);
			}
			a = P->Grass_lines->rank_lint_here(A + 2 * d, 0 /*verbose_level - 4*/);
			if (f_vv) {
				cout << "perp of line " << i << " is " << a << ":";
				Int_vec_print(cout, A + 2 * d, d);
				cout << endl;
			}
			Line_to_line[i] = a;
		}

	}

	FREE_int(A);

	if (f_v) {
		cout << "polarity::init_standard_polarity "
				"before determine_absolute_points" << endl;
	}
	determine_absolute_points(f_absolute, verbose_level);
	if (f_v) {
		cout << "polarity::init_standard_polarity "
				"after determine_absolute_points" << endl;
	}

	if (d == 4) {
		if (f_v) {
			cout << "polarity::init_standard_polarity "
					"before determine_absolute_lines" << endl;
		}
		determine_absolute_lines(verbose_level);
		if (f_v) {
			cout << "polarity::init_standard_polarity "
					"after determine_absolute_lines" << endl;
		}

	}
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
	int *B;
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

	if (d == 4) {
		Line_to_line = NEW_lint(P->N_lines);
	}

	v = NEW_int(d);
	A = NEW_int(d * d);
	B = NEW_int(d * d);

	for (i = 0; i < P->N_points; i++) {

		P->F->Projective_space_basic->PG_element_unrank_modified(
				v, 1, d, i);


		P->F->Linear_algebra->mult_matrix_matrix(v, Mtx,
				A, 1, d, d, 0 /* verbose_level*/);


		if (f_vv) {
			cout << "point " << i << " * Mtx = " << endl;
			Int_vec_print_integer_matrix_width(cout,
				A, 1, d, d,
				P->F->log10_of_q + 1);
		}
		P->F->Linear_algebra->perp_standard(d, 1, A, 0);
		if (FALSE) {
			Int_vec_print_integer_matrix_width(cout,
				A, d, d, d,
				P->F->log10_of_q + 1);
		}
		a = P->Grass_hyperplanes->rank_lint_here(A + d, 0 /*verbose_level - 4*/);
		if (f_vv) {
			cout << "hyperplane " << i << " is perp of point ";
			Int_vec_print(cout, A + 2 * d, d);
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


	if (d == 4) {
		for (i = 0; i < P->N_lines; i++) {
			P->Grass_lines->unrank_lint_here(A, i, 0 /*verbose_level - 4*/);
			if (f_vv) {
				cout << "line " << i << ":" << endl;
				Int_vec_print_integer_matrix_width(cout,
					A, 2, d, d,
					P->F->log10_of_q + 1);
			}

			P->F->Linear_algebra->mult_matrix_matrix(A, Mtx,
					B, 2, d, d, 0 /* verbose_level*/);

			P->F->Linear_algebra->perp_standard(d, 2, B, 0);
			if (FALSE) {
				Int_vec_print_integer_matrix_width(cout,
					B, d, d, d,
					P->F->log10_of_q + 1);
			}
			a = P->Grass_lines->rank_lint_here(B + 2 * d, 0 /*verbose_level - 4*/);
			if (f_vv) {
				cout << "perp of line " << i << " is " << a << ":";
				Int_vec_print(cout, B + 2 * d, d);
				cout << endl;
			}
			Line_to_line[i] = a;
		}

	}



	FREE_int(v);
	FREE_int(A);
	FREE_int(B);

	if (f_v) {
		cout << "polarity::init_general_polarity "
				"before determine_absolute_points" << endl;
	}
	determine_absolute_points(f_absolute, verbose_level);
	if (f_v) {
		cout << "polarity::init_general_polarity "
				"after determine_absolute_points" << endl;
	}

	if (d == 4) {
		if (f_v) {
			cout << "polarity::init_general_polarity "
					"before determine_absolute_lines" << endl;
		}
		determine_absolute_lines(verbose_level);
		if (f_v) {
			cout << "polarity::init_general_polarity "
					"after determine_absolute_lines" << endl;
		}

	}

	if (f_v) {
		cout << "polarity::init_general_polarity done" << endl;
	}
}

void polarity::determine_absolute_points(int *&f_absolute, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int N_points;
	int N = 0;

	if (f_v) {
		cout << "polarity::determine_absolute_points" << endl;
	}

	if (P->n != 3) {
		cout << "polarity::determine_absolute_points "
				"we need n=3, skipping" << endl;
		return;
	}
	N_points = P->nb_rk_k_subspaces_as_lint(1 /* type_i */);
	f_absolute = NEW_int(N_points);

	for (i = 0; i < N_points; i++) {
		j = Point_to_hyperplane[i];
		f_absolute[i] = P->incidence_test_for_objects_of_type_ij(
			1 /* type_i */, P->n /* type_j */, i, j,
			0 /* verbose_level */);
		if (f_absolute[i]) {
			if (FALSE) {
				cout << "polarity::determine_absolute_points "
						"absolute point: " << i << endl;
			}
			N++;
		}
	}

	if (f_v) {
		cout << "polarity::determine_absolute_points "
				"The number of absolute points is " << N << endl;
	}

}

void polarity::determine_absolute_lines(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "polarity::determine_absolute_lines" << endl;
	}

	if (P->n != 3) {
		cout << "polarity::determine_absolute_lines "
				"we need n=3, skipping" << endl;
		return;
	}
	f_absolute_line = NEW_int(P->N_lines);
	nb_absolute_lines = 0;
	nb_self_dual_lines = 0;

	for (i = 0; i < P->N_lines; i++) {
		j = Line_to_line[i];
		if (P->test_if_lines_are_disjoint_from_scratch(i, j)) {
			f_absolute_line[i] = FALSE;
		}
		else {
			f_absolute_line[i] = TRUE;
		}
		if (f_absolute_line[i]) {
			if (FALSE) {
				cout << "polarity::determine_absolute_lines "
						"absolute line: " << i << endl;
			}
			nb_absolute_lines++;
		}
		if (j == i) {
			nb_self_dual_lines++;
		}
	}

	if (f_v) {
		cout << "polarity::determine_absolute_lines "
				"The number of absolute lines is " << nb_absolute_lines << endl;
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
	Int_vec_zero(Mtx, d * d);

	// the anti-diagonal matrix:

	for (i = 0; i < d; i++) {
		Mtx[i * d + d - 1 - i] = 1;
	}

	if (f_v) {
		cout << "polarity::init_reversal_polarity "
				"before init_general_polarity" << endl;
	}

	init_general_polarity(P, Mtx, verbose_level);

	if (f_v) {
		cout << "polarity::init_reversal_polarity "
				"after init_general_polarity" << endl;
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

	if (f_absolute == NULL) {
		cout << "polarity::report NULL pointer: f_absolute" << endl;
		return;
	}

	if (P->N_points >= 1000) {
		f << "Too many to list\\\\" << endl;
		return;
	}
	int N;
	N = 0;
	for (i = 0; i < P->N_points; i++) {
		if (f_absolute[i]) {
			N++;
		}
	}
	f << "There are " << N << " absolute points: \\\\" << endl;
	for (i = 0; i < P->N_points; i++) {
		if (f_absolute[i]) {
			f << "$" << i << " \\leftrightarrow " << Point_to_hyperplane[i] << "$\\\\" << endl;
		}
	}

	if (P->n + 1 == 4) {
		f << "Lines $\\leftrightarrow$ lines:\\\\" << endl;
		f << "\\begin{multicols}{4}" << endl;
		for (i = 0; i < P->N_lines; i++) {
			f << "$" << i << " \\leftrightarrow " << Line_to_line[i] << "$\\\\" << endl;
		}
		f << "\\end{multicols}" << endl;

	}
	f << "There are " << nb_absolute_lines << " absolute lines: \\\\" << endl;
	for (i = 0; i < P->N_lines; i++) {
		if (f_absolute_line[i]) {
			f << "$" << i << " \\leftrightarrow " << Line_to_line[i] << "$\\\\" << endl;
		}
	}
	f << "There are " << nb_self_dual_lines << " self dual lines: \\\\" << endl;
	for (i = 0; i < P->N_lines; i++) {
		if (Line_to_line[i] == i) {
			f << "$" << i << " \\leftrightarrow " << Line_to_line[i] << "$\\\\" << endl;
		}
	}


	f << "\\clearpage" << endl << endl;

}

}}}


