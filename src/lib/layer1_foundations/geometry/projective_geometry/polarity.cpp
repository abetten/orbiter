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
namespace projective_geometry {


polarity::polarity()
{
	//std::string label_txt;
	//std::string label_tex;

	//std::string degree_sequence_txt;
	//std::string degree_sequence_tex;

	P = NULL;
	Point_to_hyperplane = NULL;
	Hyperplane_to_point = NULL;
	f_absolute = NULL;
	Line_to_line = NULL;
	f_absolute_line = NULL;
	nb_absolute_lines = 0;
	nb_self_dual_lines = 0;

	nb_ranks = 0;
	rank_sequence = NULL;
	rank_sequence_opposite = NULL;
	nb_objects = NULL;
	offset = NULL;
	total_degree = 0;

	Mtx = NULL;


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
	if (rank_sequence) {
		FREE_int(rank_sequence);
	}
	if (rank_sequence_opposite) {
		FREE_int(rank_sequence_opposite);
	}
	if (nb_objects) {
		FREE_lint(nb_objects);
	}
	if (offset) {
		FREE_lint(offset);
	}
	if (Mtx) {
		FREE_int(Mtx);
	}
}

void polarity::init_standard_polarity(
		projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 1);
	long int i, j;
	int *A;
	long int a;
	int n, d;
	long int N_points;

	if (f_v) {
		cout << "polarity::init_standard_polarity" << endl;
	}
	polarity::P = P;
	n = P->Subspaces->n;
	d = n + 1;
	N_points = P->Subspaces->N_points;

	Point_to_hyperplane = NEW_int(N_points);
	Hyperplane_to_point = NEW_int(N_points);

	if (d == 4) {
		Line_to_line = NEW_lint(P->Subspaces->N_lines);
	}
	A = NEW_int(d * d);


	if (f_v) {
		cout << "polarity::init_standard_polarity before init_ranks" << endl;
	}
	init_ranks(verbose_level);
	if (f_v) {
		cout << "polarity::init_standard_polarity after init_ranks" << endl;
	}

	label_txt = "standard_polarity_" + stringify_rank_sequence();
	label_tex = "standard\\_polarity\\_" + stringify_rank_sequence();

	degree_sequence_txt = stringify_degree_sequence();
	degree_sequence_tex = stringify_degree_sequence();

	for (i = 0; i < P->Subspaces->N_points; i++) {
		P->Subspaces->Grass_hyperplanes->unrank_lint(
				i, 0 /*verbose_level - 4*/);
		for (j = 0; j < n * d; j++) {
			A[j] = P->Subspaces->Grass_hyperplanes->M[j];
		}
		if (f_vv) {
			cout << "hyperplane " << i << ":" << endl;
			Int_vec_print_integer_matrix_width(cout,
				A, n, d, d,
				P->Subspaces->F->log10_of_q + 1);
		}
		P->Subspaces->F->Linear_algebra->perp_standard(d, n, A, 0);
		if (false) {
			Int_vec_print_integer_matrix_width(cout,
				A, d, d, d,
				P->Subspaces->F->log10_of_q + 1);
		}
		P->Subspaces->F->Projective_space_basic->PG_element_rank_modified_lint(
				A + n * d, 1, d, a);
		if (f_vv) {
			cout << "hyperplane " << i << " is perp of point ";
			Int_vec_print(cout, A + n * d, d);
			cout << " = " << a << endl;
		}
		Point_to_hyperplane[a] = i;
		Hyperplane_to_point[i] = a;
	}
	if (false /* f_vv */) {
		cout << "i : pt_to_hyperplane[i] : hyperplane_to_pt[i]" << endl;
		for (i = 0; i < N_points; i++) {
			cout << setw(4) << i << " "
				<< setw(4) << Point_to_hyperplane[i] << " "
				<< setw(4) << Hyperplane_to_point[i] << endl;
		}
	}


	if (d == 4) {
		for (i = 0; i < P->Subspaces->N_lines; i++) {
			P->Subspaces->Grass_lines->unrank_lint_here(
					A, i, 0 /*verbose_level - 4*/);
			if (f_vv) {
				cout << "line " << i << ":" << endl;
				Int_vec_print_integer_matrix_width(cout,
					A, 2, d, d,
					P->Subspaces->F->log10_of_q + 1);
			}
			P->Subspaces->F->Linear_algebra->perp_standard(d, 2, A, 0);
			if (false) {
				Int_vec_print_integer_matrix_width(cout,
					A, d, d, d,
					P->Subspaces->F->log10_of_q + 1);
			}
			a = P->Subspaces->Grass_lines->rank_lint_here(
					A + 2 * d, 0 /*verbose_level - 4*/);
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


void polarity::init_general_polarity(
		projective_space *P, int *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 1);
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
	n = P->Subspaces->n;
	d = n + 1;
	N_points = P->Subspaces->N_points;

	Point_to_hyperplane = NEW_int(N_points);
	Hyperplane_to_point = NEW_int(N_points);

	if (d == 4) {
		Line_to_line = NEW_lint(P->Subspaces->N_lines);
	}

	v = NEW_int(d);
	A = NEW_int(d * d);
	B = NEW_int(d * d);

	if (f_v) {
		cout << "polarity::init_general_polarity before init_ranks" << endl;
	}
	init_ranks(verbose_level);
	if (f_v) {
		cout << "polarity::init_general_polarity after init_ranks" << endl;
	}

	label_txt = "general_polarity_" + stringify_rank_sequence();
	label_tex = "general\\_polarity\\_" + stringify_rank_sequence();

	degree_sequence_txt = stringify_degree_sequence();
	degree_sequence_tex = stringify_degree_sequence();

	for (i = 0; i < P->Subspaces->N_points; i++) {

		P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
				v, 1, d, i);


		P->Subspaces->F->Linear_algebra->mult_matrix_matrix(
				v, Mtx,
				A, 1, d, d, 0 /* verbose_level*/);


		if (f_vv) {
			cout << "point " << i << " * Mtx = " << endl;
			Int_vec_print_integer_matrix_width(cout,
				A, 1, d, d,
				P->Subspaces->F->log10_of_q + 1);
		}
		P->Subspaces->F->Linear_algebra->perp_standard(d, 1, A, 0);
		if (false) {
			Int_vec_print_integer_matrix_width(cout,
				A, d, d, d,
				P->Subspaces->F->log10_of_q + 1);
		}
		a = P->Subspaces->Grass_hyperplanes->rank_lint_here(
				A + d, 0 /*verbose_level - 4*/);
		if (f_vv) {
			cout << "hyperplane " << i << " is perp of point ";
			Int_vec_print(cout, A + 2 * d, d);
			cout << " = " << a << endl;
		}
		Point_to_hyperplane[i] = a;
		Hyperplane_to_point[a] = i;
	}
	if (false /* f_vv */) {
		cout << "i : pt_to_hyperplane[i] : hyperplane_to_pt[i]" << endl;
		for (i = 0; i < N_points; i++) {
			cout << setw(4) << i << " "
				<< setw(4) << Point_to_hyperplane[i] << " "
				<< setw(4) << Hyperplane_to_point[i] << endl;
		}
	}


	if (d == 4) {
		for (i = 0; i < P->Subspaces->N_lines; i++) {
			P->Subspaces->Grass_lines->unrank_lint_here(
					A, i, 0 /*verbose_level - 4*/);
			if (f_vv) {
				cout << "line " << i << ":" << endl;
				Int_vec_print_integer_matrix_width(cout,
					A, 2, d, d,
					P->Subspaces->F->log10_of_q + 1);
			}

			P->Subspaces->F->Linear_algebra->mult_matrix_matrix(
					A, Mtx,
					B, 2, d, d, 0 /* verbose_level*/);

			P->Subspaces->F->Linear_algebra->perp_standard(
					d, 2, B, 0);
			if (false) {
				Int_vec_print_integer_matrix_width(cout,
					B, d, d, d,
					P->Subspaces->F->log10_of_q + 1);
			}
			a = P->Subspaces->Grass_lines->rank_lint_here(
					B + 2 * d, 0 /*verbose_level - 4*/);
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

void polarity::init_ranks(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity::init_ranks" << endl;
	}
	int n, d;
	long int N_points;

	n = P->Subspaces->n;
	d = n + 1;
	N_points = P->Subspaces->N_points;

	Mtx = NEW_int(d * d);

	if (d == 4) {
		int i;

		nb_ranks = 3;
		rank_sequence = NEW_int(nb_ranks);
		rank_sequence_opposite = NEW_int(nb_ranks);
		nb_objects = NEW_lint(nb_ranks);
		offset = NEW_lint(nb_ranks);
		rank_sequence[0] = 1;
		rank_sequence[1] = 2;
		rank_sequence[2] = 3; // = d - 1;
		for (i = 0; i < nb_ranks; i++) {
			rank_sequence_opposite[i] = n - rank_sequence[i];
		}
		nb_objects[0] = N_points;
		nb_objects[1] = P->Subspaces->N_lines;
		nb_objects[2] = N_points;
		offset[0] = 0;
		for (i = 1; i < nb_ranks; i++) {
			offset[i] = offset[i - 1] + nb_objects[i - 1];
		}
		total_degree = offset[nb_ranks - 1] + nb_objects[nb_ranks - 1];

	}
	else {
		int i;

		nb_ranks = 2;
		rank_sequence = NEW_int(nb_ranks);
		rank_sequence_opposite = NEW_int(nb_ranks);
		nb_objects = NEW_lint(nb_ranks);
		offset = NEW_lint(nb_ranks);
		rank_sequence[0] = 1;
		rank_sequence[1] = d - 1;
		for (i = 0; i < nb_ranks; i++) {
			rank_sequence_opposite[i] = n - rank_sequence[i];
		}
		nb_objects[0] = N_points;
		nb_objects[1] = N_points;
		offset[0] = 0;
		for (i = 1; i < nb_ranks; i++) {
			offset[i] = offset[i - 1] + nb_objects[i - 1];
		}
		total_degree = offset[nb_ranks - 1] + nb_objects[nb_ranks - 1];
	}
	if (f_v) {
		cout << "polarity::init_ranks total_degree = " << total_degree << endl;
	}
	if (f_v) {
		cout << "polarity::init_ranks done" << endl;
	}
}

void polarity::determine_absolute_points(
		int *&f_absolute, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int N_points;
	int N = 0;

	if (f_v) {
		cout << "polarity::determine_absolute_points" << endl;
	}

	if (P->Subspaces->n != 3) {
		cout << "polarity::determine_absolute_points "
				"we need n=3, skipping" << endl;
		return;
	}
	N_points = P->Subspaces->nb_rk_k_subspaces_as_lint(1 /* type_i */);
	f_absolute = NEW_int(N_points);

	for (i = 0; i < N_points; i++) {
		j = Point_to_hyperplane[i];
		f_absolute[i] = P->Subspaces->incidence_test_for_objects_of_type_ij(
			1 /* type_i */, P->Subspaces->n /* type_j */, i, j,
			0 /* verbose_level */);
		if (f_absolute[i]) {
			if (false) {
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

void polarity::determine_absolute_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "polarity::determine_absolute_lines" << endl;
	}

	if (P->Subspaces->n != 3) {
		cout << "polarity::determine_absolute_lines "
				"we need n=3, skipping" << endl;
		return;
	}
	f_absolute_line = NEW_int(P->Subspaces->N_lines);
	nb_absolute_lines = 0;
	nb_self_dual_lines = 0;

	for (i = 0; i < P->Subspaces->N_lines; i++) {
		j = Line_to_line[i];
		if (P->Subspaces->test_if_lines_are_disjoint_from_scratch(i, j)) {
			f_absolute_line[i] = false;
		}
		else {
			f_absolute_line[i] = true;
		}
		if (f_absolute_line[i]) {
			if (false) {
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


void polarity::init_reversal_polarity(
		projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Mtx;
	int n, d, i;

	if (f_v) {
		cout << "polarity::init_reversal_polarity" << endl;
	}
	polarity::P = P;
	n = P->Subspaces->n;
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

	label_txt = "reversal_polarity_" + stringify_rank_sequence();
	label_tex = "reversal\\_polarity\\_" + stringify_rank_sequence();

	degree_sequence_txt = stringify_rank_sequence();
	degree_sequence_tex = stringify_rank_sequence();



	FREE_int(Mtx);





	if (f_v) {
		cout << "polarity::init_reversal_polarity done" << endl;
	}
}

long int polarity::image_of_element(
		int *Elt, int rho, long int a,
		projective_space *P,
		algebra::matrix_group *M,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int b, c;
	int r_idx, r, h, d;
	int *v;
	int *vA;

	if (f_v) {
		cout << "polarity::image_of_element" << endl;
		cout << "polarity::image_of_element a=" << a << endl;
	}
	d = P->Subspaces->n + 1;

	if (f_v) {
		cout << "polarity::image_of_element Elt = " << endl;
		M->Element->GL_print_easy(Elt, cout);
	}

	for (r_idx = 0; r_idx < nb_ranks; r_idx++) {
		if (a < nb_objects[r_idx]) {

			r = rank_sequence[r_idx];
			if (f_v) {
				cout << "polarity::image_of_element r = " << r << endl;
			}


			// for points, we use a different ranking
			// than the one provided by grassmann.

			if (r == 1) {
				// point
				M->GFq->Projective_space_basic->PG_element_unrank_modified_lint(
						P->Subspaces->Grass_stack[r]->M, 1, d, a);
			}
			else {
				// subspace of dimension at least 2
				P->Subspaces->Grass_stack[r]->unrank_lint(a, 0 /*verbose_level*/);
			}
			if (f_v) {
				cout << "polarity::image_of_element input = " << endl;
				Int_matrix_print(P->Subspaces->Grass_stack[r]->M, r, d);
			}

			for (h = 0; h < r; h++) {
				v = P->Subspaces->Grass_stack[r]->M + h * d;
				vA = Mtx + h * d;
				M->GFq->Linear_algebra->projective_action_from_the_right(
						M->f_semilinear,
						v, Elt, vA, M->n,
						0 /*verbose_level - 1*/);
			}
			// vA = (v * A)^{p^f}  if f_semilinear
			// (where f = A[n * n]),
			// vA = v * A otherwise

			if (f_v) {
				cout << "polarity::image_of_element output = " << endl;
				Int_matrix_print(Mtx, r, d);
			}


			// Again, take care of points separately:

			if (r == 1) {
				// point
				M->GFq->Projective_space_basic->PG_element_rank_modified_lint(Mtx, 1, d, b);

			}
			else {
				b = P->Subspaces->Grass_stack[r]->rank_lint_here(Mtx, 0 /*verbose_level*/);
			}

			if (f_v) {
				cout << "polarity::image_of_element a -> " << b << " (before polarity)" << endl;
			}


			// test if polarity is present:

			if (rho) {

				// polarity is present, so apply the polarity:

				int r2;
				int b2;

				r2 = rank_sequence_opposite[r_idx];
				if (r == 1) {
					b2 = Point_to_hyperplane[b];
				}
				else if (d == r + 1) {
					b2 = Hyperplane_to_point[b];
				}
				else if (d == 4 && r == 2) {
					b2 = Line_to_line[b];
				}
				else {
					cout << "polarity::image_of_element rank not yet implemented" << endl;
					cout << "d = " << d << endl;
					cout << "r = " << r << endl;
					exit(1);
				}
				c = offset[r2] + b2;
				if (f_v) {
					cout << "polarity::image_of_element a -> " << b << " ->  (polarity) " << b2 << " -> (offset) " << c << " (after polarity)" << endl;
				}
			}
			else {

				// polarity is not present:

				int r2 = r_idx;
				c = offset[r2] + b;
				if (f_v) {
					cout << "polarity::image_of_element a -> " << b << " -> " << c << " (no polarity)" << endl;
				}
			}



			break;
		}
		a -= nb_objects[r_idx];
	}
	if (r_idx == nb_ranks) {
		cout << "polarity::image_of_element illegal input value" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "polarity::image_of_element done" << endl;
	}
	return c;
}



void polarity::report(
		std::ostream &f)
{
	int i;

	//f << "Polarity point $\\leftrightarrow$ hyperplane:\\\\" << endl;
	f << "\\begin{multicols}{4}" << endl;
	for (i = 0; i < P->Subspaces->N_points; i++) {
		f << "$" << i << " \\leftrightarrow " << Point_to_hyperplane[i] << "$\\\\" << endl;
	}
	f << "\\end{multicols}" << endl;

	if (f_absolute == NULL) {
		cout << "polarity::report NULL pointer: f_absolute" << endl;
		return;
	}

	if (P->Subspaces->N_points >= 1000) {
		f << "Too many to list\\\\" << endl;
		return;
	}
	int N;
	N = 0;
	for (i = 0; i < P->Subspaces->N_points; i++) {
		if (f_absolute[i]) {
			N++;
		}
	}
	f << "There are " << N << " absolute points: \\\\" << endl;
	for (i = 0; i < P->Subspaces->N_points; i++) {
		if (f_absolute[i]) {
			f << "$" << i << " \\leftrightarrow " << Point_to_hyperplane[i] << "$\\\\" << endl;
		}
	}

	if (P->Subspaces->n + 1 == 4) {
		f << "Lines $\\leftrightarrow$ lines:\\\\" << endl;
		f << "\\begin{multicols}{4}" << endl;
		for (i = 0; i < P->Subspaces->N_lines; i++) {
			f << "$" << i << " \\leftrightarrow " << Line_to_line[i] << "$\\\\" << endl;
		}
		f << "\\end{multicols}" << endl;

	}
	f << "There are " << nb_absolute_lines << " absolute lines: \\\\" << endl;
	for (i = 0; i < P->Subspaces->N_lines; i++) {
		if (f_absolute_line[i]) {
			f << "$" << i << " \\leftrightarrow " << Line_to_line[i] << "$\\\\" << endl;
		}
	}
	f << "There are " << nb_self_dual_lines << " self dual lines: \\\\" << endl;
	for (i = 0; i < P->Subspaces->N_lines; i++) {
		if (Line_to_line[i] == i) {
			f << "$" << i << " \\leftrightarrow " << Line_to_line[i] << "$\\\\" << endl;
		}
	}


	f << "\\clearpage" << endl << endl;

}

std::string polarity::stringify_rank_sequence()
{
	string s;

#if 0
	int i;

	for (i = 0; i < nb_ranks; i++) {
		s += std::to_string(rank_sequence[i]);
		if (i < nb_ranks - 1) {
			s += ",";
		}
	}
#endif
	s = Int_vec_stringify(rank_sequence, nb_ranks);
	return s;
}

std::string polarity::stringify_degree_sequence()
{
	string s;

#if 0
	int i;

	for (i = 0; i < nb_ranks; i++) {
		s += std::to_string(nb_objects[i]);
		if (i < nb_ranks - 1) {
			s += ",";
		}
	}
#endif
	s = Lint_vec_stringify(nb_objects, nb_ranks);
	return s;
}




}}}}



