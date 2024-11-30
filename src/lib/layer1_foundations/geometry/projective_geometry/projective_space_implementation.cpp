/*
 * projective_space_implementation.cpp
 *
 *  Created on: Nov 2, 2021
 *      Author: betten
 */


#include "foundations.h"


using namespace std;


#define MAX_NUMBER_OF_LINES_FOR_INCIDENCE_MATRIX 100000
#define MAX_NUMBER_OF_LINES_FOR_LINE_TABLE 1000000
#define MAX_NUMBER_OF_POINTS_FOR_POINT_TABLE 1000000
#define MAX_NB_POINTS_FOR_LINE_THROUGH_TWO_POINTS_TABLE 10000
#define MAX_NB_POINTS_FOR_LINE_INTERSECTION_TABLE 10000


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace projective_geometry {


projective_space_implementation::projective_space_implementation()
{
	P = NULL;

	Bitmatrix = NULL;

	Line_through_two_points = NULL;
	Line_intersection = NULL;
	Lines = NULL;
	Lines_on_point = NULL;

	v = NULL;
	w = NULL;

	Mtx = NULL;
	Mtx2 = NULL;

}

projective_space_implementation::~projective_space_implementation()
{
	if (Bitmatrix) {
		FREE_OBJECT(Bitmatrix);
	}
	if (Line_through_two_points) {
		FREE_int(Line_through_two_points);
	}
	if (Line_intersection) {
		FREE_int(Line_intersection);
	}
	if (Lines) {
		FREE_int(Lines);
	}
	if (Lines_on_point) {
		FREE_int(Lines_on_point);
	}
	if (v) {
		FREE_int(v);
	}
	if (w) {
		FREE_int(w);
	}
	if (Mtx) {
		FREE_int(Mtx);
	}
	if (Mtx2) {
		FREE_int(Mtx2);
	}
}

void projective_space_implementation::init(
		projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	int i, j, a, i1, i2, j1, j2;
	long int b;
	long int N_points, N_lines;
	int k, r, n;

	if (f_v) {
		cout << "projective_space_implementation::init" << endl;
	}

	projective_space_implementation::P = P;

	N_points = P->Subspaces->N_points;
	N_lines = P->Subspaces->N_lines;
	k = P->Subspaces->k;
	r = P->Subspaces->r;
	n = P->Subspaces->n;

	v = NEW_int(n + 1);
	w = NEW_int(n + 1);
	Mtx = NEW_int(3 * (n + 1));
	Mtx2 = NEW_int(3 * (n + 1));

	if (N_lines < MAX_NUMBER_OF_LINES_FOR_INCIDENCE_MATRIX) {
		if (f_v) {
			cout << "projective_space_implementation::init "
					"allocating Incidence (bitvector)" << endl;
		}
		Bitmatrix = NEW_OBJECT(data_structures::bitmatrix);
		Bitmatrix->init(N_points, N_lines, verbose_level);
	}
	else {
		if (f_v) {
			cout << "projective_space_implementation::init: "
				"N_lines too big, we do not initialize the "
				"incidence matrix" << endl;
		}
		Bitmatrix = NULL;
	}


	if (N_lines < MAX_NUMBER_OF_LINES_FOR_LINE_TABLE) {
		if (f_v) {
			cout << "projective_space_implementation::init "
					"allocating Lines" << endl;
		}
		Lines = NEW_int(N_lines * k);
	}
	else {
		if (f_v) {
			cout << "projective_space_implementation::init: "
				"N_lines too big, we do not initialize "
				"the line table" << endl;
		}
		Lines = NULL;
	}




	if (N_points < MAX_NUMBER_OF_POINTS_FOR_POINT_TABLE &&
			N_lines < MAX_NUMBER_OF_LINES_FOR_LINE_TABLE)  {
		if (f_v) {
			cout << "projective_space_implementation::init "
					"allocating Lines_on_point" << endl;
			cout << "projective_space_implementation::init "
					"allocating N_points=" << N_points << endl;
			cout << "projective_space_implementation::init "
					"allocating r=" << r << endl;
		}
		Lines_on_point = NEW_int(N_points * r);
	}
	else {
		if (f_v) {
			cout << "projective_space_implementation::init: "
				"N_points too big, we do not initialize the "
				"Lines_on_point table" << endl;
		}
		Lines_on_point = NULL;
	}




	if ((long int) N_points < MAX_NB_POINTS_FOR_LINE_THROUGH_TWO_POINTS_TABLE) {

		if (f_v) {
			cout << "projective_space_implementation::init "
					"allocating Line_through_two_points" << endl;
		}
		Line_through_two_points = NEW_int((long int) N_points * (long int) N_points);
	}
	else {
		if (f_v) {
			cout << "projective_space_implementation::init: "
				"we do not initialize "
				"Line_through_two_points" << endl;
		}
		Line_through_two_points = NULL;
	}

	if ((long int) N_lines < MAX_NB_POINTS_FOR_LINE_INTERSECTION_TABLE) {
		if (f_v) {
			cout << "projective_space_implementation::init "
					"allocating Line_intersection" << endl;
		}
		Line_intersection = NEW_int((long int) N_lines * (long int) N_lines);
		Int_vec_zero(Line_through_two_points, (long int) N_points * (long int) N_points);
		for (i = 0; i < (long int) N_lines * (long int) N_lines; i++) {
			Line_intersection[i] = -1;
		}
	}
	else {
		if (f_v) {
			cout << "projective_space_implementation::init: "
				"we do not initialize "
				"Line_intersection" << endl;
		}
		Line_intersection = NULL;
	}


	if (f_v) {
		cout << "projective_space_implementation::init "
				"number of points = " << N_points << endl;
	}
	if (false) {
		for (i = 0; i < N_points; i++) {
			P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
					v, 1, n + 1, i);
			cout << "point " << i << " : ";
			Int_vec_print(cout, v, n + 1);
			cout << " = ";
			P->Subspaces->F->Io->int_vec_print_field_elements(cout, v, n + 1);

			P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
					v, 1, n + 1);
			cout << " = ";
			Int_vec_print(cout, v, n + 1);


			cout << " = ";
			P->Subspaces->F->Io->int_vec_print_field_elements(cout, v, n + 1);


			cout << endl;
		}
	}
	if (f_v) {
		cout << "projective_space_implementation::init "
				"number of lines = " << N_lines << endl;
	}



	if (Lines || Bitmatrix || Lines_on_point) {


		if (f_v) {
			cout << "projective_space_implementation::init "
					"computing lines..." << endl;
			if (Lines) {
				cout << "Lines is allocated" << endl;
			}
			if (Bitmatrix) {
				cout << "Bitmatrix is allocated" << endl;
			}
			if (Lines_on_point) {
				cout << "Lines_on_point is allocated" << endl;
			}
		}



		int *R;

		R = NEW_int(N_points);
		Int_vec_zero(R, N_points);

		for (i = 0; i < N_lines; i++) {
#if 0
			if ((i % 1000000) == 0) {
				cout << "projective_space_implementation::init "
						"Line " << i << " / " << N_lines << ":" << endl;
			}
#endif
			P->Subspaces->Grass_lines->unrank_lint(i, 0/*verbose_level - 4*/);
			if (false) {
				Int_vec_print_integer_matrix_width(cout,
						P->Subspaces->Grass_lines->M, 2, n + 1, n + 1,
						P->Subspaces->F->log10_of_q + 1);
			}


#if 0
			// testing of grassmann:

			j = Grass_lines->rank_int(0/*verbose_level - 4*/);
			if (j != i) {
				cout << "projective_space_implementation::init "
						"rank yields " << j << " != " << i << endl;
				exit(1);
			}
#endif



			for (a = 0; a < k; a++) {
				P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
						v, 1, 2, a);
				P->Subspaces->F->Linear_algebra->mult_matrix_matrix(
						v, P->Subspaces->Grass_lines->M, w, 1, 2, n + 1,
						0 /* verbose_level */);
				P->Subspaces->F->Projective_space_basic->PG_element_rank_modified(
						w, 1, n + 1, b);
				if (Bitmatrix) {
					Bitmatrix->m_ij(b, i, 1);
				}

				if (Lines) {
					Lines[i * k + a] = b;
				}
				if (Lines_on_point) {
					Lines_on_point[b * r + R[b]] = i;
				}
				R[b]++;
			}
			if (f_vv) {
				cout << "line " << i << ":" << endl;
				Int_vec_print_integer_matrix_width(cout,
						P->Subspaces->Grass_lines->M, 2, n + 1, n + 1,
						P->Subspaces->F->log10_of_q + 1);

				if (Lines) {
					cout << "points on line " << i << " : ";
					Int_vec_print(cout, Lines + i * k, k);
					cout << endl;
				}
			}

		}
		for (i = 0; i < N_points; i++) {
			if (R[i] != r) {
				cout << "R[i] != r" << endl;
				exit(1);
			}
		}

		FREE_int(R);
	}
	else {
		if (f_v) {
			cout << "projective_space_implementation::init "
					"computing lines skipped" << endl;
		}
	}



#if 0
	if (f_v) {
		cout << "computing Lines_on_point..." << endl;
	}
	for (i = 0; i < N_points; i++) {
		if ((i % 1000) == 0) {
			cout << "point " << i << " / " << N_points << ":" << endl;
		}
		a = 0;
		for (j = 0; j < N_lines; j++) {
			if (is_incident(i, j)) {
				Lines_on_point[i * r + a] = j;
				a++;
			}
		}
		if (false /*f_vv */) {
			cout << "lines on point " << i << " = ";
			PG_element_unrank_modified(*F, w, 1, n + 1, i);
			int_vec_print(cout, w, n + 1);
			cout << " : ";
			int_vec_print(cout, Lines_on_point + i * r, r);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "computing Lines_on_point done" << endl;
	}
#endif

	if (false) {
		//cout << "incidence matrix:" << endl;
		//print_integer_matrix_width(cout, Incidence, N_points, N_lines, N_lines, 1);
		cout << "projective_space::init_incidence_structure Lines:" << endl;
		Int_vec_print_integer_matrix_width(cout, Lines, N_lines, k, k, 3);
		cout << "projective_space::init_incidence_structure Lines_on_point:" << endl;
		Int_vec_print_integer_matrix_width(cout, Lines_on_point, N_points, r, r, 3);
	}


	if (Line_through_two_points && Lines && Lines_on_point) {
		if (f_v) {
			cout << "projective_space_implementation::init "
					"computing Line_through_two_points..." << endl;
		}
		for (i1 = 0; i1 < N_points; i1++) {
			for (a = 0; a < r; a++) {
				j = Lines_on_point[i1 * r + a];
				for (b = 0; b < k; b++) {
					i2 = Lines[j * k + b];
					if (i2 == i1) {
						continue;
					}
					Line_through_two_points[i1 * N_points + i2] = j;
					Line_through_two_points[i2 * N_points + i1] = j;
				}
			}
		}
		if (false) {
			cout << "line through points i,j is" << endl;
			for (i = 0; i < N_points; i++) {
				for (j = i + 1; j < N_points; j++) {
					cout << i << " , " << j << " : "
						<< Line_through_two_points[i * N_points + j] << endl;
				}
			}
			//cout << "Line_through_two_points:" << endl;
			//print_integer_matrix_width(cout,
			//Line_through_two_points, N_points, N_points, N_points, 2);
		}
	}
	else {
		if (f_v) {
			cout << "projective_space_implementation::init "
				"computing Line_through_two_points skipped" << endl;
		}
	}

	if (Line_intersection && Lines && Lines_on_point) {
		if (f_v) {
			cout << "projective_space_implementation::init "
				"computing Line_intersection..." << endl;
		}
		for (j1 = 0; j1 < N_lines; j1++) {
			for (a = 0; a < k; a++) {
				i = Lines[j1 * k + a];
				for (b = 0; b < r; b++) {
					j2 = Lines_on_point[i * r + b];
					if (j2 == j1) {
						continue;
					}
					Line_intersection[j1 * N_lines + j2] = i;
					Line_intersection[j2 * N_lines + j1] = i;
				}
			}
		}
		if (false) {
			cout << "projective_space_implementation::init "
					"point of intersection of lines i,j is" << endl;
			for (i = 0; i < N_lines; i++) {
				for (j = i + 1; j < N_lines; j++) {
					cout << i << " , " << j << " : "
						<< Line_intersection[i * N_lines + j] << endl;
				}
			}
			//cout << "Line_intersection:" << endl;
			//print_integer_matrix_width(cout,
			// Line_intersection, N_lines, N_lines, N_lines, 2);
		}
	}
	else {
		if (f_v) {
			cout << "projective_space_implementation::init "
				"computing Line_intersection skipped" << endl;
		}
	}
	if (f_v) {
		cout << "projective_space_implementation::init done" << endl;
	}
}

data_structures::bitmatrix *projective_space_implementation::get_Bitmatrix()
{
	return Bitmatrix;
}

int projective_space_implementation::has_lines()
{
	if (Lines) {
		return true;
	}
	else {
		return false;
	}
}

int projective_space_implementation::has_lines_on_point()
{
	if (Lines_on_point) {
		return true;
	}
	else {
		return false;
	}
}


int projective_space_implementation::find_point_on_line(
		int line_rk, int pt)
{
	data_structures::sorting Sorting;
	int *L;
	int k;
	int idx;

	k = P->Subspaces->k;

	L = Lines + line_rk * k;
	if (Sorting.int_vec_search(L, k, pt, idx)) {
		return true;
	}
	else {
		return false;
	}
}

int projective_space_implementation::lines(
		int i, int j)
{
	int k;


	k = P->Subspaces->k;

	return Lines[i * k + j];
}

int projective_space_implementation::lines_on_point(
		int i, int j)
{
	int r;


	r = P->Subspaces->r;

	return Lines_on_point[i * r + j];
}

int projective_space_implementation::line_through_two_points(
		int i, int j)
{

	int N_points;


	N_points = P->Subspaces->N_points;
	//N_lines = P->Subspaces->N_lines;

	return Line_through_two_points[i * N_points + j];
}

int projective_space_implementation::line_intersection(
		int i, int j)
{

	int N_lines;


	//N_points = P->Subspaces->N_points;
	N_lines = P->Subspaces->N_lines;

	return Line_intersection[i * N_lines + j];
}

void projective_space_implementation::line_intersection_type(
		long int *set, int set_size, int *type,
		int verbose_level)
	// type[N_lines]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_implementation::line_intersection_type" << endl;
	}

	int i, j, a, b;

	for (i = 0; i < P->Subspaces->N_lines; i++) {
		type[i] = 0;
	}
	for (i = 0; i < set_size; i++) {
		a = set[i];
		for (j = 0; j < P->Subspaces->r; j++) {
			b = Lines_on_point[a * P->Subspaces->r + j];
			type[b]++;
		}
	}

	if (f_v) {
		cout << "projective_space_implementation::line_intersection_type done" << endl;
	}
}

void projective_space_implementation::point_types_of_line_set(
	long int *set_of_lines, int set_size,
	int *type, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_implementation::point_types_of_line_set" << endl;
	}
	int i, j, a, b;

	for (i = 0; i < P->Subspaces->N_points; i++) {
		type[i] = 0;
	}
	for (i = 0; i < set_size; i++) {
		a = set_of_lines[i];
		for (j = 0; j < P->Subspaces->k; j++) {
			b = Lines[a * P->Subspaces->k + j];
			type[b]++;
		}
	}
	if (f_v) {
		cout << "projective_space_implementation::point_types_of_line_set done" << endl;
	}
}

void projective_space_implementation::point_types_of_line_set_int(
	int *set_of_lines, int set_size,
	int *type, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_implementation::point_types_of_line_set_int" << endl;
	}
	int i, j, a, b;

	for (i = 0; i < P->Subspaces->N_points; i++) {
		type[i] = 0;
	}
	for (i = 0; i < set_size; i++) {
		a = set_of_lines[i];
		for (j = 0; j < P->Subspaces->k; j++) {
			b = Lines[a * P->Subspaces->k + j];
			type[b]++;
		}
	}
	if (f_v) {
		cout << "projective_space_implementation::point_types_of_line_set_int done" << endl;
	}
}

int projective_space_implementation::is_incident(
		int pt, int line)
{
	int f_v = false;
	long int rk;
	int n;

	n = P->Subspaces->n;

	if (true /*incidence_bitvec == NULL*/) {
		P->Subspaces->Grass_lines->unrank_lint(line, 0/*verbose_level - 4*/);

		if (f_v) {
			cout << "projective_space_implementation::is_incident "
					"line=" << endl;
			Int_matrix_print(P->Subspaces->Grass_lines->M, 2, n + 1);
		}
		Int_vec_copy(P->Subspaces->Grass_lines->M, Mtx, 2 * (n + 1));
		P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
				Mtx + 2 * (n + 1), 1, n + 1, pt);
		if (f_v) {
			cout << "point:" << endl;
			Int_vec_print(cout, Mtx + 2 * (n + 1), n + 1);
			cout << endl;
		}

		rk = P->Subspaces->F->Linear_algebra->rank_of_rectangular_matrix_memory_given(
				Mtx,
				3, n + 1, Mtx2, v /* base_cols */,
				false /* f_complete */,
				0 /*verbose_level*/);
		if (f_v) {
			cout << "rk = " << rk << endl;
		}
		if (rk == 3) {
			return false;
		}
		else {
			return true;
		}
	}
	else {
		//a = (long int) pt * (long int) N_lines + (long int) line;
		//return bitvector_s_i(incidence_bitvec, a);
		return Bitmatrix->s_ij(pt, line);
	}
}

void projective_space_implementation::incidence_m_ii(
		int pt, int line, int a)
{
	//long int b;

	if (Bitmatrix == NULL) {
		cout << "projective_space_implementation::incidence_m_ii "
				"Bitmatrix == NULL" << endl;
		exit(1);
		}
	//b = (long int) pt * (long int) N_lines + (long int) line;
	//bitvector_m_ii(incidence_bitvec, b, a);

	Bitmatrix->m_ij(pt, P->Subspaces->N_lines, a);
}

int projective_space_implementation::test_if_lines_are_disjoint(
		long int l1, long int l2)
{
	data_structures::sorting Sorting;

	if (Lines) {
		int k;
		k = P->Subspaces->k;
		return Sorting.test_if_sets_are_disjoint_assuming_sorted(
				Lines + l1 * k,
				Lines + l2 * k, k, k);
	}
	else {
		return test_if_lines_are_disjoint_from_scratch(l1, l2);
	}
}

int projective_space_implementation::test_if_lines_are_disjoint_from_scratch(
		long int l1, long int l2)
{
	int *Mtx;
	int m, rk;

	m = P->Subspaces->n + 1;
	Mtx = NEW_int(4 * m);
	P->Subspaces->Grass_lines->unrank_lint_here(
			Mtx, l1, 0/*verbose_level - 4*/);
	P->Subspaces->Grass_lines->unrank_lint_here(
			Mtx + 2 * m, l2, 0/*verbose_level - 4*/);
	rk = P->Subspaces->F->Linear_algebra->Gauss_easy(Mtx, 4, m);
	FREE_int(Mtx);
	if (rk == 4) {
		return true;
	}
	else {
		return false;
	}
}



}}}}





