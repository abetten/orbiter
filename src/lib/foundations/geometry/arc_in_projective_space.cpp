/*
 * arc_in_projective_space.cpp
 *
 *  Created on: Nov 26, 2021
 *      Author: betten
 */




#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {


arc_in_projective_space::arc_in_projective_space()
{
	P = NULL;

}

arc_in_projective_space::~arc_in_projective_space()
{

}

void arc_in_projective_space::init(projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_in_projective_space::init" << endl;
	}

	arc_in_projective_space::P = P;

	if (f_v) {
		cout << "arc_in_projective_space::init done" << endl;
	}
}

void arc_in_projective_space::PG_2_8_create_conic_plus_nucleus_arc_1(
		long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int frame_data[] = {1,0,0, 0,1,0,  0,0,1,  1,1,1 };
	int frame[4];
	int i, j, b, h, idx;
	int L[3];
	int v[3];
	data_structures::sorting Sorting;

	if (P->n != 2) {
		cout << "arc_in_projective_space::PG_2_8_create_conic_plus_nucleus_arc_1 "
				"P->n != 2" << endl;
		exit(1);
	}
	if (P->q != 8) {
		cout << "arc_in_projective_space::PG_2_8_create_conic_plus_nucleus_arc_1 "
				"P->q != 8" << endl;
		exit(1);
	}
	for (i = 0; i < 4; i++) {
		frame[i] = P->rank_point(frame_data + i * 3);
	}

	if (f_v) {
		cout << "frame: ";
		Orbiter->Int_vec->print(cout, frame, 4);
		cout << endl;
	}

	L[0] = P->Implementation->Line_through_two_points[frame[0] * P->N_points + frame[1]];
	L[1] = P->Implementation->Line_through_two_points[frame[1] * P->N_points + frame[2]];
	L[2] = P->Implementation->Line_through_two_points[frame[2] * P->N_points + frame[0]];

	if (f_v) {
		cout << "l1=" << L[0] << " l2=" << L[1] << " l3=" << L[2] << endl;
	}

	size = 0;
	for (h = 0; h < 3; h++) {
		for (i = 0; i < P->r; i++) {
			b = P->Implementation->Lines[L[h] * P->r + i];
			if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
				continue;
			}
			for (j = size; j > idx; j--) {
				the_arc[j] = the_arc[j - 1];
			}
			the_arc[idx] = b;
			size++;
		}
	}
	if (f_v) {
		cout << "there are " << size << " points on the three lines: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}


	for (i = 1; i < P->q; i++) {
		v[0] = 1;
		v[1] = i;
		v[2] = P->F->mult(i, i);
		b = P->rank_point(v);
		if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
			continue;
		}
		for (j = size; j > idx; j--) {
			the_arc[j] = the_arc[j - 1];
		}
		the_arc[idx] = b;
		size++;

	}

	if (f_v) {
		cout << "arc_in_projective_space::PG_2_8_create_conic_plus_nucleus_arc_1: after adding the rest of the "
				"conic, there are " << size << " points on the arc: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}
}

void arc_in_projective_space::PG_2_8_create_conic_plus_nucleus_arc_2(
		long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int frame_data[] = {1,0,0, 0,1,0,  0,0,1,  1,1,1 };
	int frame[4];
	int i, j, b, h, idx;
	int L[3];
	int v[3];
	data_structures::sorting Sorting;

	if (P->n != 2) {
		cout << "arc_in_projective_space::PG_2_8_create_conic_plus_"
				"nucleus_arc_2 P->n != 2" << endl;
		exit(1);
	}
	if (P->q != 8) {
		cout << "arc_in_projective_space::PG_2_8_create_conic_plus_"
				"nucleus_arc_2 P->q != 8" << endl;
		exit(1);
	}
	for (i = 0; i < 4; i++) {
		frame[i] = P->rank_point(frame_data + i * 3);
	}

	if (f_v) {
		cout << "frame: ";
		Orbiter->Int_vec->print(cout, frame, 4);
		cout << endl;
	}

	L[0] = P->Implementation->Line_through_two_points[frame[0] * P->N_points + frame[2]];
	L[1] = P->Implementation->Line_through_two_points[frame[2] * P->N_points + frame[3]];
	L[2] = P->Implementation->Line_through_two_points[frame[3] * P->N_points + frame[0]];

	if (f_v) {
		cout << "l1=" << L[0] << " l2=" << L[1] << " l3=" << L[2] << endl;
	}

	size = 0;
	for (h = 0; h < 3; h++) {
		for (i = 0; i < P->r; i++) {
			b = P->Implementation->Lines[L[h] * P->r + i];
			if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
				continue;
			}
			for (j = size; j > idx; j--) {
				the_arc[j] = the_arc[j - 1];
			}
			the_arc[idx] = b;
			size++;
		}
	}
	if (f_v) {
		cout << "there are " << size << " points on the three lines: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}


	for (i = 0; i < P->q; i++) {
		if (i == 1) {
			v[0] = 0;
			v[1] = 1;
			v[2] = 0;
		}
		else {
			v[0] = 1;
			v[1] = i;
			v[2] = P->F->mult(i, i);
		}
		b = P->rank_point(v);
		if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
			continue;
		}
		for (j = size; j > idx; j--) {
			the_arc[j] = the_arc[j - 1];
		}
		the_arc[idx] = b;
		size++;

	}

	if (f_v) {
		cout << "arc_in_projective_space::PG_2_8_create_conic_plus_"
				"nucleus_arc_2: after adding the rest of the conic, "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}
}

void arc_in_projective_space::create_Maruta_Hamada_arc(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int data[] = {
		0,1,2, 0,1,3, 0,1,4, 0,1,5,
		1,0,7, 1,0,8, 1,0,9, 1,0,10,
		1,2,0, 1,3,0, 1,4,0, 1,5,0,
		1,7,5, 1,8,4, 1,9,3, 1,10,2,
		1,1,1, 1,1,10, 1,10,1,  1,4,4,
		1,12,0, 1,0,12
		 };
	int points[22];
	int i, j, b, h, idx;
	long int L[4];
	int v[3];
	data_structures::sorting Sorting;

	if (P->n != 2) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc "
				"n != 2" << endl;
		exit(1);
	}
	if (P->q != 13) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc "
				"q != 13" << endl;
		exit(1);
	}
	for (i = 0; i < 22; i++) {
		points[i] = P->rank_point(data + i * 3);
		cout << "point " << i << " has rank " << points[i] << endl;
	}

	if (f_v) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc "
				"points: ";
		Orbiter->Int_vec->print(cout, points, 22);
		cout << endl;
	}

	L[0] = P->Implementation->Line_through_two_points[1 * P->N_points + 2];
	L[1] = P->Implementation->Line_through_two_points[0 * P->N_points + 2];
	L[2] = P->Implementation->Line_through_two_points[0 * P->N_points + 1];
	L[3] = P->Implementation->Line_through_two_points[points[20] * P->N_points + points[21]];

	if (f_v) {
		cout << "L:";
		Orbiter->Lint_vec->print(cout, L, 4);
		cout << endl;
	}

	if (f_v) {
		for (h = 0; h < 4; h++) {
			cout << "h=" << h << " : L[h]=" << L[h] << " : " << endl;
			for (i = 0; i < P->r; i++) {
				b = P->Implementation->Lines[L[h] * P->r + i];
					cout << "point " << b << " = ";
					P->unrank_point(v, b);
					P->F->PG_element_normalize_from_front(v, 1, 3);
				Orbiter->Int_vec->print(cout, v, 3);
				cout << endl;
			}
			cout << endl;
		}
	}
	size = 0;
	for (h = 0; h < 4; h++) {
		for (i = 0; i < P->r; i++) {
			b = P->Implementation->Lines[L[h] * P->r + i];
			if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
				continue;
			}
			for (j = size; j > idx; j--) {
				the_arc[j] = the_arc[j - 1];
			}
			the_arc[idx] = b;
			size++;
		}
	}
	if (f_v) {
		cout << "there are " << size
				<< " points on the quadrilateral: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}


	// remove the first 16 points:
	for (i = 0; i < 16; i++) {
		if (f_v) {
			cout << "removing point " << i << " : "
				<< points[i] << endl;
		}
		if (!Sorting.lint_vec_search(the_arc, size, points[i], idx, 0)) {
			cout << "error, cannot find point to be removed" << endl;
			exit(1);
		}
		for (j = idx; j < size; j++) {
			the_arc[j] = the_arc[j + 1];
		}
		size--;
	}

	// add points 16-19:
	for (i = 16; i < 20; i++) {
		if (Sorting.lint_vec_search(the_arc, size, points[i], idx, 0)) {
			cout << "error, special point already there" << endl;
			exit(1);
		}
		for (j = size; j > idx; j--) {
			the_arc[j] = the_arc[j - 1];
		}
		the_arc[idx] = points[i];
		size++;
	}

	if (f_v) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc: "
				"after adding the special point, there are "
				<< size << " points on the arc: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}

}

void arc_in_projective_space::create_Maruta_Hamada_arc2(
		long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int data[] = {
		1,6,2, 1,11,4, 1,5,5, 1,2,6, 1,10,7, 1,12,8, 1,7,10, 1,4,11, 1,8,12,
		0,1,10, 0,1,12, 0,1,4, 1,0,1, 1,0,3, 1,0,9, 1,1,0, 1,3,0, 1,9,0,
		1,4,4, 1,4,12, 1,12,4, 1,10,10, 1,10,12, 1,12,10
		 };
	int points[24];
	int i, j, a;
	int L[9];

	if (P->n != 2) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc2 "
				"P->n != 2" << endl;
		exit(1);
	}
	if (P->q != 13) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc2 "
				"P->q != 13" << endl;
		exit(1);
	}
	for (i = 0; i < 24; i++) {
		points[i] = P->rank_point(data + i * 3);
		cout << "point " << i << " has rank " << points[i] << endl;
	}

	if (f_v) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc2 "
				"points: ";
		Orbiter->Int_vec->print(cout, points, 25);
		cout << endl;
	}
	for (i = 0; i < 9; i++) {
		L[i] = P->Standard_polarity->Point_to_hyperplane[points[i]];
	}
	size = 0;
	for (i = 0; i < 9; i++) {
		for (j = i + 1; j < 9; j++) {
			a = P->intersection_of_two_lines(L[i], L[j]);
			the_arc[size++] = a;
		}
	}
	for (i = 9; i < 24; i++) {
		the_arc[size++] = points[i];
	}
	if (f_v) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc2: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}
}


void arc_in_projective_space::create_pasch_arc(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int data[] = {1,1,1, 1,0,0, 0,1,1,  0,1,0,  1,0,1 };
	int points[5];
	int i, j, b, h, idx;
	int L[4];
	data_structures::sorting Sorting;

	if (P->n != 2) {
		cout << "arc_in_projective_space::create_pasch_arc "
				"P->n != 2" << endl;
		exit(1);
	}
#if 0
	if (q != 8) {
		cout << "arc_in_projective_space::create_pasch_arc "
				"q != 8" << endl;
		exit(1);
	}
#endif
	for (i = 0; i < 5; i++) {
		points[i] = P->rank_point(data + i * 3);
	}

	if (f_v) {
		cout << "arc_in_projective_space::create_pasch_arc() points: ";
		Orbiter->Int_vec->print(cout, points, 5);
		cout << endl;
	}

	L[0] = P->Implementation->Line_through_two_points[points[0] * P->N_points + points[1]];
	L[1] = P->Implementation->Line_through_two_points[points[0] * P->N_points + points[3]];
	L[2] = P->Implementation->Line_through_two_points[points[2] * P->N_points + points[3]];
	L[3] = P->Implementation->Line_through_two_points[points[1] * P->N_points + points[4]];

	if (f_v) {
		cout << "L:";
		Orbiter->Int_vec->print(cout, L, 4);
		cout << endl;
	}

	size = 0;
	for (h = 0; h < 4; h++) {
		for (i = 0; i < P->r; i++) {
			b = P->Implementation->Lines[L[h] * P->r + i];
			if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
				continue;
			}
			for (j = size; j > idx; j--) {
				the_arc[j] = the_arc[j - 1];
			}
			the_arc[idx] = b;
			size++;
		}
	}
	if (f_v) {
		cout << "there are " << size << " points on the pasch lines: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}


	P->v[0] = 1;
	P->v[1] = 1;
	P->v[2] = 0;
	b = P->rank_point(P->v);
	if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
		cout << "error, special point already there" << endl;
		exit(1);
	}
	for (j = size; j > idx; j--) {
		the_arc[j] = the_arc[j - 1];
	}
	the_arc[idx] = b;
	size++;

	if (f_v) {
		cout << "arc_in_projective_space::create_pasch_arc: after "
				"adding the special point, there are "
				<< size << " points on the arc: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}
}

void arc_in_projective_space::create_Cheon_arc(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int data[] = {1,0,0, 0,1,0, 0,0,1 };
	int points[3];
	int i, j, a, b, c, h, idx, t;
	int L[3];
	int pencil[9];
	int Pencil[21];
	data_structures::sorting Sorting;

	if (P->n != 2) {
		cout << "arc_in_projective_space::create_Cheon_arc P->n != 2" << endl;
		exit(1);
	}
#if 0
	if (P->q != 8) {
		cout << "arc_in_projective_space::create_Cheon_arc P->q != 8" << endl;
		exit(1);
	}
#endif
	for (i = 0; i < 9; i++) {
		pencil[i] = 0;
	}
	for (i = 0; i < 21; i++) {
		Pencil[i] = 0;
	}
	for (i = 0; i < 3; i++) {
		points[i] = P->rank_point(data + i * 3);
	}

	if (f_v) {
		cout << "points: ";
		Orbiter->Int_vec->print(cout, points, 5);
		cout << endl;
	}

	L[0] = P->Implementation->Line_through_two_points[points[0] * P->N_points + points[1]];
	L[1] = P->Implementation->Line_through_two_points[points[1] * P->N_points + points[2]];
	L[2] = P->Implementation->Line_through_two_points[points[2] * P->N_points + points[0]];

	if (f_v) {
		cout << "L:";
		Orbiter->Int_vec->print(cout, L, 3);
		cout << endl;
	}

	size = 0;
	for (h = 0; h < 3; h++) {
		for (i = 0; i < P->r; i++) {
			b = P->Implementation->Lines[L[h] * P->r + i];
			if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
				continue;
			}
			for (j = size; j > idx; j--) {
				the_arc[j] = the_arc[j - 1];
			}
			the_arc[idx] = b;
			size++;
		}
	}

	if (f_v) {
		cout << "arc_in_projective_space::create_Cheon_arc there are "
				<< size << " points on the 3 lines: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}




	for (h = 0; h < 3; h++) {

		if (f_v) {
			cout << "h=" << h << endl;
		}

		for (i = 0; i < P->r; i++) {
			pencil[i] = P->Implementation->Lines_on_point[points[h] * P->r + i];
		}


		j = 0;
		for (i = 0; i < P->r; i++) {
			b = pencil[i];
			if (b == L[0] || b == L[1] || b == L[2])
				continue;
			Pencil[h * 7 + j] = b;
			j++;
		}
		if (j != 7) {
			cout << "j=" << j << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "Pencil:" << endl;
		Orbiter->Int_vec->print_integer_matrix_width(cout, Pencil, 3, 7, 7, 4);
	}

	for (i = 0; i < 7; i++) {
		a = Pencil[0 * 7 + i];
		for (j = 0; j < 7; j++) {
			b = Pencil[1 * 7 + j];
			if (f_v) {
				cout << "i=" << i << " a=" << a << " j="
						<< j << " b=" << b << endl;
			}
			c = P->Implementation->Line_intersection[a * P->N_lines + b];
			if (f_v) {
				cout << "c=" << c << endl;
			}
			if (Sorting.lint_vec_search(the_arc, size, c, idx, 0)) {
				continue;
			}
			for (t = size; t > idx; t--) {
				the_arc[t] = the_arc[t - 1];
			}
			the_arc[idx] = c;
			size++;
#if 0
			if (size > 31) {
				cout << "create_Cheon_arc size=" << size << endl;
			}
#endif
		}
	}
	if (f_v) {
		cout << "there are " << size << " points on the Cheon lines: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}


}


void arc_in_projective_space::create_regular_hyperoval(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];

	if (P->n != 2) {
		cout << "arc_in_projective_space::create_regular_hyperoval "
				"P->n != 2" << endl;
		exit(1);
	}

	for (i = 0; i < P->q; i++) {
		v[0] = P->F->mult(i, i);
		v[1] = i;
		v[2] = 1;
		the_arc[i] = P->rank_point(v);
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[P->q] = P->rank_point(v);

	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[P->q + 1] = P->rank_point(v);

	size = P->q + 2;

	if (f_v) {
		cout << "arc_in_projective_space::create_regular_hyperoval: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}
}

void arc_in_projective_space::create_translation_hyperoval(
	long int *the_arc, int &size,
	int exponent, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];

	if (f_v) {
		cout << "arc_in_projective_space::create_translation_hyperoval" << endl;
		cout << "exponent = " << exponent << endl;
	}
	if (P->n != 2) {
		cout << "arc_in_projective_space::create_translation_hyperoval "
				"P->n != 2" << endl;
		exit(1);
	}

	for (i = 0; i < P->q; i++) {
		v[0] = P->F->frobenius_power(i, exponent);
		v[1] = i;
		v[2] = 1;
		the_arc[i] = P->rank_point(v);
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[P->q] = P->rank_point(v);

	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[P->q + 1] = P->rank_point(v);

	size = P->q + 2;

	if (f_v) {
		cout << "arc_in_projective_space::create_translation_hyperoval: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}
	if (f_v) {
		cout << "arc_in_projective_space::create_translation_hyperoval "
				"done" << endl;
	}
}

void arc_in_projective_space::create_Segre_hyperoval(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];

	if (P->n != 2) {
		cout << "arc_in_projective_space::create_Segre_hyperoval "
				"n != 2" << endl;
		exit(1);
	}

	for (i = 0; i < P->q; i++) {
		v[0] = P->F->power(i, 6);
		v[1] = i;
		v[2] = 1;
		the_arc[i] = P->rank_point(v);
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[P->q] = P->rank_point(v);

	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[P->q + 1] = P->rank_point(v);

	size = P->q + 2;

	if (f_v) {
		cout << "arc_in_projective_space::create_Segre_hyperoval: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}
}

void arc_in_projective_space::create_Payne_hyperoval(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, b, u, u2, g;
	int exponent;
	int one_sixth, one_half, five_sixth;

	if (f_v) {
		cout << "arc_in_projective_space::create_Payne_hyperoval" << endl;
	}
	if (P->n != 2) {
		cout << "arc_in_projective_space::create_Payne_hyperoval P->n != 2" << endl;
		exit(1);
	}
	exponent = P->q - 1;
	a.create(6, __FILE__, __LINE__);
	b.create(exponent, __FILE__, __LINE__);

	D.extended_gcd(a, b, g, u, u2, 0 /* verbose_level */);
	one_sixth = u.as_int();
	while (one_sixth < 0) {
		one_sixth += exponent;
	}
	if (f_v) {
		cout << "one_sixth = " << one_sixth << endl;
	}

	a.create(2, __FILE__, __LINE__);
	D.extended_gcd(a, b, g, u, u2, 0 /* verbose_level */);
	one_half = u.as_int();
	while (one_half < 0) {
		one_half += exponent;
	}
	if (f_v) {
		cout << "one_half = " << one_half << endl;
	}

	five_sixth = (5 * one_sixth) % exponent;
	if (f_v) {
		cout << "five_sixth = " << five_sixth << endl;
	}

	for (i = 0; i < P->q; i++) {
		v[0] = P->F->add3(
				P->F->power(i, one_sixth),
				P->F->power(i, one_half),
				P->F->power(i, five_sixth));
		v[1] = i;
		v[2] = 1;
		the_arc[i] = P->rank_point(v);
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[P->q] = P->rank_point(v);

	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[P->q + 1] = P->rank_point(v);

	size = P->q + 2;

	if (f_v) {
		cout << "arc_in_projective_space::create_Payne_hyperoval: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}
}

void arc_in_projective_space::create_Cherowitzo_hyperoval(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];
	int h;
	int sigma;
	int exponent, one_half, e1, e2, e3;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "arc_in_projective_space::create_Cherowitzo_hyperoval" << endl;
	}
	if (P->n != 2) {
		cout << "arc_in_projective_space::create_Cherowitzo_hyperoval "
				"P->n != 2" << endl;
		exit(1);
	}
	h = P->F->e;
	if (EVEN(h)) {
		cout << "arc_in_projective_space::create_Cherowitzo_hyperoval "
				"field degree must be odd" << endl;
		exit(1);
	}
	if (P->F->p != 2) {
		cout << "arc_in_projective_space::create_Cherowitzo_hyperoval "
				"needs characteristic 2" << endl;
		exit(1);
	}
	exponent = P->q - 1;
	one_half = (h + 1) >> 1;
	sigma = NT.i_power_j(2, one_half);
	e1 = sigma;
	e2 = (sigma + 2) % exponent;
	e3 = (3 * sigma + 4) % exponent;

	for (i = 0; i < P->q; i++) {
		v[0] = P->F->add3(
				P->F->power(i, e1),
				P->F->power(i, e2),
				P->F->power(i, e3));
		v[1] = i;
		v[2] = 1;
		the_arc[i] = P->rank_point(v);
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[P->q] = P->rank_point(v);

	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[P->q + 1] = P->rank_point(v);

	size = P->q + 2;

	if (f_v) {
		cout << "arc_in_projective_space::create_Cherowitzo_hyperoval: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}
}

void arc_in_projective_space::create_OKeefe_Penttila_hyperoval_32(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];

	if (f_v) {
		cout << "arc_in_projective_space::create_OKeefe_Penttila_hyperoval_32"
				<< endl;
	}
	if (P->n != 2) {
		cout << "arc_in_projective_space::create_OKeefe_Penttila_hyperoval_32 "
				"P->n != 2" << endl;
		exit(1);
	}
	if (P->F->q != 32) {
		cout << "arc_in_projective_space::create_OKeefe_Penttila_hyperoval_32 "
				"needs q=32" << endl;
		exit(1);
	}

	{
		arc_basic A;

		A.init(P->F, verbose_level - 1);

		for (i = 0; i < P->q; i++) {


			v[0] = A.OKeefe_Penttila_32(i);
			v[1] = i;
			v[2] = 1;
			the_arc[i] = P->rank_point(v);
		}
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[P->q] = P->rank_point(v);

	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[P->q + 1] = P->rank_point(v);

	size = P->q + 2;

	if (f_v) {
		cout << "arc_in_projective_space::create_OKeefe_Penttila_hyperoval_32: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec->print(cout, the_arc, size);
		cout << endl;
	}
}

void arc_in_projective_space::arc_lifting_diophant(
	long int *arc, int arc_sz,
	int target_sz, int target_d,
	solvers::diophant *&D,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	int *line_type;
	int i, j, h, pt;
	long int *free_points;
	int nb_free_points;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "arc_in_projective_space::arc_lifting_diophant" << endl;
	}

	free_points = NEW_lint(P->N_points);

	Combi.set_complement_lint(arc, arc_sz,
			free_points, nb_free_points, P->N_points);



	line_type = NEW_int(P->N_lines);
	P->line_intersection_type(arc, arc_sz,
			line_type, 0 /* verbose_level */);
	if (f_vv) {
		cout << "line_type: ";
		Orbiter->Int_vec->print_fully(cout, line_type, P->N_lines);
		cout << endl;
	}

	if (f_vv) {
		cout << "line type:" << endl;
		for (i = 0; i < P->N_lines; i++) {
			cout << i << " : " << line_type[i] << endl;
		}
	}

	tally C;
	C.init(line_type, P->N_lines, FALSE, 0);
	if (f_v) {
		cout << "arc_in_projective_space::arc_lifting_diophant line_type:";
		C.print_naked(TRUE);
		cout << endl;
		cout << "nb_free_points=" << nb_free_points << endl;
	}


	D = NEW_OBJECT(solvers::diophant);
	D->open(P->N_lines + 1, nb_free_points);
	//D->f_x_max = TRUE;
	for (j = 0; j < nb_free_points; j++) {
		D->x_min[j] = 0;
		D->x_max[j] = 1;
	}
	D->f_has_sum = TRUE;
	D->sum = target_sz - arc_sz;
	h = 0;
	for (i = 0; i < P->N_lines; i++) {
		if (line_type[i] > P->k) {
			cout << "arc_in_projective_space::arc_lifting_diophant "
					"line_type[i] > P->k" << endl;
			exit(1);
		}
	#if 0
		if (line_type[i] < P->k - 1) {
			continue;
		}
	#endif
		for (j = 0; j < nb_free_points; j++) {
			pt = free_points[j];
			if (P->is_incident(pt, i /* line */)) {
				D->Aij(h, j) = 1;
			}
			else {
				D->Aij(h, j) = 0;
			}
		}
		D->type[h] = t_LE;
		D->RHSi(h) = target_d - line_type[i];
		h++;
	}


	// add one extra row:
	for (j = 0; j < nb_free_points; j++) {
		D->Aij(h, j) = 1;
	}
	D->type[h] = t_EQ;
	D->RHSi(h) = target_sz - arc_sz;
	h++;

	D->m = h;

	D->init_var_labels(free_points, verbose_level);

	if (f_vv) {
		cout << "arc_in_projective_space::arc_lifting_diophant "
				"The system is:" << endl;
		D->print_tight();
	}

#if 0
	if (f_save_system) {
		cout << "arc_in_projective_space saving the system "
				"to file " << fname_system << endl;
		D->save_in_general_format(fname_system, 0 /* verbose_level */);
		cout << "arc_in_projective_space saving the system "
				"to file " << fname_system << " done" << endl;
		D->print();
		D->print_tight();
	}
#endif

	FREE_int(line_type);
	FREE_lint(free_points);

	if (f_v) {
		cout << "arc_in_projective_space::arc_lifting_diophant done" << endl;
	}

}

void arc_in_projective_space::arc_with_given_set_of_s_lines_diophant(
	long int *s_lines, int nb_s_lines,
	int target_sz, int arc_d, int arc_d_low, int arc_s,
	int f_dualize,
	solvers::diophant *&D,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	int i, j, h, a, line;
	long int *other_lines;
	int nb_other_lines;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "arc_in_projective_space::arc_with_given_set_of_s_lines_diophant" << endl;
	}

	other_lines = NEW_lint(P->N_points);

	Combi.set_complement_lint(s_lines, nb_s_lines,
			other_lines, nb_other_lines, P->N_lines);


	if (f_dualize) {
		if (P->Standard_polarity == NULL) {
			cout << "arc_in_projective_space::arc_with_given_set_of_s_lines_diophant "
					"Standard_polarity == NULL" << endl;
			exit(1);
		}
	}


	D = NEW_OBJECT(solvers::diophant);
	D->open(P->N_lines + 1, P->N_points);
	//D->f_x_max = TRUE;
	for (j = 0; j < P->N_points; j++) {
		D->x_min[j] = 0;
		D->x_max[j] = 1;
	}
	D->f_has_sum = TRUE;
	D->sum = target_sz;
	h = 0;
	for (i = 0; i < nb_s_lines; i++) {
		if (f_dualize) {
			line = P->Standard_polarity->Point_to_hyperplane[s_lines[i]];
		}
		else {
			line = s_lines[i];
		}
		for (j = 0; j < P->N_points; j++) {
			if (P->is_incident(j, line)) {
				a = 1;
			}
			else {
				a = 0;
			}
			D->Aij(h, j) = a;
		}
		D->type[h] = t_EQ;
		D->RHSi(h) = arc_s;
		h++;
	}
	for (i = 0; i < nb_other_lines; i++) {
		if (f_dualize) {
			line = P->Standard_polarity->Point_to_hyperplane[other_lines[i]];
		}
		else {
			line = other_lines[i];
		}
		for (j = 0; j < P->N_points; j++) {
			if (P->is_incident(j, line)) {
				a = 1;
			}
			else {
				a = 0;
			}
			D->Aij(h, j) = a;
		}
		D->type[h] = t_INT;
		D->RHSi(h) = arc_d;
		D->RHS_low_i(h) = arc_d_low;
		//D->type[h] = t_LE;
		//D->RHSi(h) = arc_d;
		h++;
	}


	// add one extra row:
	for (j = 0; j < P->N_points; j++) {
		D->Aij(h, j) = 1;
	}
	D->type[h] = t_EQ;
	D->RHSi(h) = target_sz;
	h++;

	D->m = h;

	//D->init_var_labels(N_points, verbose_level);

	if (f_vv) {
		cout << "arc_in_projective_space::arc_with_given_set_of_s_lines_diophant "
				"The system is:" << endl;
		D->print_tight();
	}

#if 0
	if (f_save_system) {
		cout << "do_arc_lifting saving the system "
				"to file " << fname_system << endl;
		D->save_in_general_format(fname_system, 0 /* verbose_level */);
		cout << "do_arc_lifting saving the system "
				"to file " << fname_system << " done" << endl;
		D->print();
		D->print_tight();
		}
#endif

	FREE_lint(other_lines);

	if (f_v) {
		cout << "arc_in_projective_space::arc_with_given_set_of_s_lines_diophant done" << endl;
	}

}


void arc_in_projective_space::arc_with_two_given_line_sets_diophant(
		long int *s_lines, int nb_s_lines, int arc_s,
		long int *t_lines, int nb_t_lines, int arc_t,
		int target_sz, int arc_d, int arc_d_low,
		int f_dualize,
		solvers::diophant *&D,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	int i, j, h, a, line;
	long int *other_lines;
	int nb_other_lines;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "arc_in_projective_space::arc_with_two_given_line_sets_diophant" << endl;
	}

	other_lines = NEW_lint(P->N_points);

	Orbiter->Lint_vec->copy(s_lines, other_lines, nb_s_lines);
	Orbiter->Lint_vec->copy(t_lines, other_lines + nb_s_lines, nb_t_lines);
	Sorting.lint_vec_heapsort(other_lines, nb_s_lines + nb_t_lines);

	Combi.set_complement_lint(other_lines, nb_s_lines + nb_t_lines,
			other_lines + nb_s_lines + nb_t_lines, nb_other_lines, P->N_lines);


	if (f_dualize) {
		if (P->Standard_polarity == NULL) {
			cout << "arc_in_projective_space::arc_with_two_given_line_sets_diophant "
					"Standard_polarity == NULL" << endl;
			exit(1);
		}
	}


	D = NEW_OBJECT(solvers::diophant);
	D->open(P->N_lines + 1, P->N_points);
	//D->f_x_max = TRUE;
	for (j = 0; j < P->N_points; j++) {
		D->x_min[j] = 0;
		D->x_max[j] = 1;
	}
	D->f_has_sum = TRUE;
	D->sum = target_sz;
	h = 0;
	for (i = 0; i < nb_s_lines; i++) {
		if (f_dualize) {
			line = P->Standard_polarity->Point_to_hyperplane[s_lines[i]];
		}
		else {
			line = s_lines[i];
		}
		for (j = 0; j < P->N_points; j++) {
			if (P->is_incident(j, line)) {
				a = 1;
			}
			else {
				a = 0;
			}
			D->Aij(h, j) = a;
		}
		D->type[h] = t_EQ;
		D->RHSi(h) = arc_s;
		h++;
	}
	for (i = 0; i < nb_t_lines; i++) {
		if (f_dualize) {
			line = P->Standard_polarity->Point_to_hyperplane[t_lines[i]];
		}
		else {
			line = t_lines[i];
		}
		for (j = 0; j < P->N_points; j++) {
			if (P->is_incident(j, line)) {
				a = 1;
			}
			else {
				a = 0;
			}
			D->Aij(h, j) = a;
		}
		D->type[h] = t_EQ;
		D->RHSi(h) = arc_t;
		h++;
	}
	for (i = 0; i < nb_other_lines; i++) {
		int l;

		l = other_lines[nb_s_lines + nb_t_lines + i];
		if (f_dualize) {
			line = P->Standard_polarity->Point_to_hyperplane[l];
		}
		else {
			line = l;
		}
		for (j = 0; j < P->N_points; j++) {
			if (P->is_incident(j, line)) {
				a = 1;
			}
			else {
				a = 0;
			}
			D->Aij(h, j) = a;
		}
		D->type[h] = t_INT;
		D->RHSi(h) = arc_d;
		D->RHS_low_i(h) = arc_d_low;
		//D->type[h] = t_LE;
		//D->RHSi(h) = arc_d;
		h++;
	}


	// add one extra row:
	for (j = 0; j < P->N_points; j++) {
		D->Aij(h, j) = 1;
	}
	D->type[h] = t_EQ;
	D->RHSi(h) = target_sz;
	h++;

	D->m = h;

	//D->init_var_labels(N_points, verbose_level);

	if (f_vv) {
		cout << "arc_in_projective_space::arc_with_two_given_line_sets_diophant "
				"The system is:" << endl;
		D->print_tight();
	}

#if 0
	if (f_save_system) {
		cout << "do_arc_lifting saving the system "
				"to file " << fname_system << endl;
		D->save_in_general_format(fname_system, 0 /* verbose_level */);
		cout << "do_arc_lifting saving the system "
				"to file " << fname_system << " done" << endl;
		D->print();
		D->print_tight();
	}
#endif

	FREE_lint(other_lines);

	if (f_v) {
		cout << "arc_in_projective_space::arc_with_two_given_line_sets_diophant done" << endl;
	}

}

void arc_in_projective_space::arc_with_three_given_line_sets_diophant(
		long int *s_lines, int nb_s_lines, int arc_s,
		long int *t_lines, int nb_t_lines, int arc_t,
		long int *u_lines, int nb_u_lines, int arc_u,
		int target_sz, int arc_d, int arc_d_low,
		int f_dualize,
		solvers::diophant *&D,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	int i, j, h, a, line;
	long int *other_lines;
	int nb_other_lines;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "arc_in_projective_space::arc_with_three_given_line_sets_diophant" << endl;
	}

	other_lines = NEW_lint(P->N_points);

	Orbiter->Lint_vec->copy(s_lines, other_lines, nb_s_lines);
	Orbiter->Lint_vec->copy(t_lines, other_lines + nb_s_lines, nb_t_lines);
	Orbiter->Lint_vec->copy(u_lines, other_lines + nb_s_lines + nb_t_lines, nb_u_lines);
	Sorting.lint_vec_heapsort(other_lines, nb_s_lines + nb_t_lines + nb_u_lines);

	Combi.set_complement_lint(other_lines, nb_s_lines + nb_t_lines + nb_u_lines,
			other_lines + nb_s_lines + nb_t_lines + nb_u_lines, nb_other_lines, P->N_lines);


	if (f_dualize) {
		if (P->Standard_polarity->Point_to_hyperplane == NULL) {
			cout << "arc_in_projective_space::arc_with_three_given_line_sets_diophant "
					"Polarity_point_to_hyperplane == NULL" << endl;
			exit(1);
		}
	}


	D = NEW_OBJECT(solvers::diophant);
	D->open(P->N_lines + 1, P->N_points);
	//D->f_x_max = TRUE;
	for (j = 0; j < P->N_points; j++) {
		D->x_min[j] = 0;
		D->x_max[j] = 1;
	}
	D->f_has_sum = TRUE;
	D->sum = target_sz;
	h = 0;
	for (i = 0; i < nb_s_lines; i++) {
		if (f_dualize) {
			line = P->Standard_polarity->Point_to_hyperplane[s_lines[i]];
		}
		else {
			line = s_lines[i];
		}
		for (j = 0; j < P->N_points; j++) {
			if (P->is_incident(j, line)) {
				a = 1;
			}
			else {
				a = 0;
			}
			D->Aij(h, j) = a;
		}
		D->type[h] = t_EQ;
		D->RHSi(h) = arc_s;
		h++;
	}
	for (i = 0; i < nb_t_lines; i++) {
		if (f_dualize) {
			line = P->Standard_polarity->Point_to_hyperplane[t_lines[i]];
		}
		else {
			line = t_lines[i];
		}
		for (j = 0; j < P->N_points; j++) {
			if (P->is_incident(j, line)) {
				a = 1;
			}
			else {
				a = 0;
			}
			D->Aij(h, j) = a;
		}
		D->type[h] = t_EQ;
		D->RHSi(h) = arc_t;
		h++;
	}
	for (i = 0; i < nb_u_lines; i++) {
		if (f_dualize) {
			line = P->Standard_polarity->Point_to_hyperplane[u_lines[i]];
		}
		else {
			line = u_lines[i];
		}
		for (j = 0; j < P->N_points; j++) {
			if (P->is_incident(j, line)) {
				a = 1;
			}
			else {
				a = 0;
			}
			D->Aij(h, j) = a;
		}
		D->type[h] = t_EQ;
		D->RHSi(h) = arc_u;
		h++;
	}
	for (i = 0; i < nb_other_lines; i++) {
		int l;

		l = other_lines[nb_s_lines + nb_t_lines + nb_u_lines + i];
		if (f_dualize) {
			line = P->Standard_polarity->Point_to_hyperplane[l];
		}
		else {
			line = l;
		}
		for (j = 0; j < P->N_points; j++) {
			if (P->is_incident(j, line)) {
				a = 1;
			}
			else {
				a = 0;
			}
			D->Aij(h, j) = a;
		}
		D->type[h] = t_INT;
		D->RHSi(h) = arc_d;
		D->RHS_low_i(h) = arc_d_low;
		h++;
	}


	// add one extra row:
	for (j = 0; j < P->N_points; j++) {
		D->Aij(h, j) = 1;
	}
	D->type[h] = t_EQ;
	D->RHSi(h) = target_sz;
	h++;

	D->m = h;

	//D->init_var_labels(N_points, verbose_level);

	if (f_vv) {
		cout << "arc_in_projective_space::arc_with_three_given_line_sets_diophant "
				"The system is:" << endl;
		D->print_tight();
	}

#if 0
	if (f_save_system) {
		cout << "arc_in_projective_space::arc_with_three_given_line_sets_diophant saving the system "
				"to file " << fname_system << endl;
		D->save_in_general_format(fname_system, 0 /* verbose_level */);
		cout << "arc_in_projective_space::arc_with_three_given_line_sets_diophant saving the system "
				"to file " << fname_system << " done" << endl;
		D->print();
		D->print_tight();
	}
#endif

	FREE_lint(other_lines);

	if (f_v) {
		cout << "arc_in_projective_space::arc_with_three_given_line_sets_diophant done" << endl;
	}

}


void arc_in_projective_space::maximal_arc_by_diophant(
		int arc_sz, int arc_d,
		std::string &secant_lines_text,
		std::string &external_lines_as_subset_of_secants_text,
		solvers::diophant *&D,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	int h, i, j, s;
	int a, line;
	int *secant_lines;
	int nb_secant_lines;
	int *Idx;
	int *external_lines;
	int nb_external_lines;
	int *other_lines;
	int nb_other_lines;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "arc_in_projective_space::maximal_arc_by_diophant" << endl;
	}

	other_lines = NEW_int(P->N_lines);

	Orbiter->Int_vec->scan(secant_lines_text, secant_lines, nb_secant_lines);
	Orbiter->Int_vec->scan(external_lines_as_subset_of_secants_text, Idx, nb_external_lines);

	Sorting.int_vec_heapsort(secant_lines, nb_secant_lines);

	Combi.set_complement(
			secant_lines, nb_secant_lines,
			other_lines, nb_other_lines,
			P->N_lines);


	external_lines = NEW_int(nb_external_lines);
	for (i = 0; i < nb_external_lines; i++) {
		external_lines[i] = secant_lines[Idx[i]];
	}
	h = 0;
	j = 0;
	for (i = 0; i <= nb_external_lines; i++) {
		if (i < nb_external_lines) {
			s = Idx[i];
		}
		else {
			s = nb_secant_lines;
		}
		for (; j < s; j++) {
			secant_lines[h] = secant_lines[j];
			h++;
		}
		j++;
	}
	if (h != nb_secant_lines - nb_external_lines) {
		cout << "h != nb_secant_lines - nb_external_lines" << endl;
		exit(1);
	}
	nb_secant_lines = h;

	int nb_slack1, nb_pencil_conditions;
	int slack1_start;
	int nb_eqns;
	int nb_vars;
	int *pencil_idx;
	int *pencil_sub_idx;
	int *nb_times_hit;
	int pt, sub_idx;

	pencil_idx = NEW_int(P->N_lines);
	pencil_sub_idx = NEW_int(P->N_lines);
	nb_times_hit = NEW_int(P->k);
	Orbiter->Int_vec->zero(nb_times_hit, P->k);

	pencil_idx[0] = -1;
	for (i = 1; i < P->N_lines; i++) {
		pt = P->intersection_of_two_lines(i, 0);
		if (pt > P->k + 2) {
			cout << "pt > k + 2" << endl;
			cout << "i=" << i << endl;
			cout << "pt=" << pt << endl;
			exit(1);
		}
		if (pt == 0 || pt == 1) {
			pencil_idx[i] = -1;
			pencil_sub_idx[i] = -1;
			continue;
		}
		if (pt < 4) {
			cout << "pt < 4" << endl;
			exit(1);
		}
		pt -= 4;
		pencil_idx[i] = pt;
		pencil_sub_idx[i] = nb_times_hit[pt];
		nb_times_hit[pt]++;
	}
	for (pt = 0; pt < P->k - 2; pt++) {
		if (nb_times_hit[pt] != P->k - 1) {
			cout << "nb_times_hit[pt] != k - 1" << endl;
			cout << "pt = " << pt << endl;
			cout << "nb_times_hit[pt] = " << nb_times_hit[pt] << endl;
			exit(1);
		}
	}

	if (nb_other_lines != (P->k - 2) * (P->k - 1)) {
		cout << "nb_other_lines != (k - 2) * (k - 1)" << endl;
		exit(1);
	}
	nb_slack1 = nb_other_lines;
	nb_pencil_conditions = P->k - 2;
	slack1_start = P->N_points;


	nb_eqns = P->N_lines + 1 + nb_pencil_conditions;
	nb_vars = P->N_points + nb_slack1;

	D = NEW_OBJECT(solvers::diophant);
	D->open(nb_eqns, nb_vars);
	//D->f_x_max = TRUE;
	for (j = 0; j < nb_vars; j++) {
		D->x_min[j] = 0;
		D->x_max[j] = 1;
	}
	D->f_has_sum = TRUE;
	D->sum = arc_sz + (arc_d - 1) * nb_pencil_conditions;
	h = 0;
	for (i = 0; i < nb_secant_lines; i++) {
		line = secant_lines[i];
		for (j = 0; j < P->N_points; j++) {
			if (P->is_incident(j, line)) {
				a = 1;
			}
			else {
				a = 0;
			}
			D->Aij(h, j) = a;
		}
		D->type[h] = t_EQ;
		D->RHSi(h) = arc_d;
		h++;
	}
	for (i = 0; i < nb_external_lines; i++) {
		line = external_lines[i];
		for (j = 0; j < P->N_points; j++) {
			if (P->is_incident(j, line)) {
				a = 1;
			}
			else {
				a = 0;
			}
			D->Aij(h, j) = a;
		}
		D->type[h] = t_EQ;
		D->RHSi(h) = 0;
		h++;
	}
	for (i = 0; i < nb_other_lines; i++) {
		line = other_lines[i];
		cout << "other line " << i << " / " << nb_other_lines << " is: " << line << " : ";
		for (j = 0; j < P->N_points; j++) {
			if (P->is_incident(j, line)) {
				a = 1;
			}
			else {
				a = 0;
			}
			D->Aij(h, j) = a;
		}
		pt = pencil_idx[line];
		sub_idx = pencil_sub_idx[line];
		cout << "pt=" << pt << " sub_idx=" << sub_idx << endl;
		if (pt < 0) {
			cout << "pt < 0" << endl;
			exit(1);
		}
		if (sub_idx < 0) {
			cout << "sub_idx < 0" << endl;
			exit(1);
		}
		D->Aij(h, slack1_start + pt * (P->k - 1) + sub_idx) = arc_d;

		D->type[h] = t_EQ;
		D->RHSi(h) = arc_d;
		h++;
	}
	for (i = 0; i < nb_pencil_conditions; i++) {
		D->type[h] = t_EQ;
		D->RHSi(h) = arc_d - 1;
		for (j = 0; j < P->k - 1; j++) {
			D->Aij(h, slack1_start + i * (P->k - 1) + j) = 1;
		}
		h++;
	}
	// add one extra row:
	for (j = 0; j < P->N_points; j++) {
		D->Aij(h, j) = 1;
		}
	D->type[h] = t_EQ;
	D->RHSi(h) = arc_sz;
	h++;

	if (h != nb_eqns) {
		cout << "arc_in_projective_space::maximal_arc_by_diophant h != nb_eqns" << endl;
		exit(1);
	}

	//D->m = h;

	//D->init_var_labels(N_points, verbose_level);

	if (f_vv) {
		cout << "arc_in_projective_space::maximal_arc_by_diophant "
				"The system is:" << endl;
		D->print_tight();
	}

	FREE_int(other_lines);
	FREE_int(external_lines);

	if (f_v) {
		cout << "arc_in_projective_space::maximal_arc_by_diophant done" << endl;
	}
}

void arc_in_projective_space::arc_lifting1(
		int arc_size,
		int arc_d,
		int arc_d_low,
		int arc_s,
		std::string arc_input_set,
		std::string arc_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_in_projective_space::arc_lifting1" << endl;
	}
	solvers::diophant *D = NULL;
	int f_save_system = TRUE;

	long int *the_set_in;
	int set_size_in;

	Orbiter->Lint_vec->scan(arc_input_set, the_set_in, set_size_in);

	arc_with_given_set_of_s_lines_diophant(
			the_set_in /*one_lines*/, set_size_in /* nb_one_lines */,
			arc_size /*target_sz*/, arc_d /* target_d */,
			arc_d_low, arc_s /* target_s */,
			TRUE /* f_dualize */,
			D,
			verbose_level);

	if (FALSE) {
		D->print_tight();
	}

	if (f_save_system) {

		string fname_system;

		fname_system.assign(arc_label);
		fname_system.append(".diophant");
		cout << "perform_job_for_one_set saving the system "
				"to file " << fname_system << endl;
		D->save_in_general_format(fname_system, 0 /* verbose_level */);
		cout << "perform_job_for_one_set saving the system "
				"to file " << fname_system << " done" << endl;
		//D->print();
		//D->print_tight();
	}

	long int nb_backtrack_nodes;
	long int *Sol;
	int nb_sol;

	D->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level);

	if (f_v) {
		cout << "before D->get_solutions" << endl;
	}
	D->get_solutions(Sol, nb_sol, verbose_level);
	if (f_v) {
		cout << "after D->get_solutions, nb_sol=" << nb_sol << endl;
	}
	string fname_solutions;

	fname_solutions.assign(arc_label);
	fname_solutions.append(".solutions");

	{
		ofstream fp(fname_solutions);
		int i, j, a;

		for (i = 0; i < nb_sol; i++) {
			fp << D->sum;
			for (j = 0; j < D->sum; j++) {
				a = Sol[i * D->sum + j];
				fp << " " << a;
				}
			fp << endl;
			}
		fp << -1 << " " << nb_sol << endl;
	}
	file_io Fio;

	cout << "Written file " << fname_solutions << " of size "
			<< Fio.file_size(fname_solutions) << endl;
	FREE_lint(Sol);
	FREE_lint(the_set_in);
	if (f_v) {
		cout << "arc_in_projective_space::arc_lifting1 done" << endl;
	}

}

void arc_in_projective_space::arc_lifting2(
		int arc_size,
		int arc_d,
		int arc_d_low,
		int arc_s,
		std::string arc_input_set,
		std::string arc_label,
		int arc_t,
		std::string t_lines_string,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_in_projective_space::arc_lifting2" << endl;
	}
	long int *t_lines;
	int nb_t_lines;

	Orbiter->Lint_vec->scan(t_lines_string, t_lines, nb_t_lines);

	cout << "The t-lines, t=" << arc_t << " are ";
	Orbiter->Lint_vec->print(cout, t_lines, nb_t_lines);
	cout << endl;


	long int *the_set_in;
	int set_size_in;

	Orbiter->Lint_vec->scan(arc_input_set, the_set_in, set_size_in);


	solvers::diophant *D = NULL;
	int f_save_system = TRUE;

	arc_with_two_given_line_sets_diophant(
			the_set_in /* s_lines */, set_size_in /* nb_s_lines */, arc_s,
			t_lines, nb_t_lines, arc_t,
			arc_size /*target_sz*/, arc_d, arc_d_low,
			TRUE /* f_dualize */,
			D,
			verbose_level);

	if (FALSE) {
		D->print_tight();
	}

	if (f_save_system) {
		string fname_system;

		fname_system.assign(arc_label);
		fname_system.append(".diophant");

		//sprintf(fname_system, "system_%d.diophant", back_end_counter);
		cout << "perform_job_for_one_set saving the system "
				"to file " << fname_system << endl;
		D->save_in_general_format(fname_system, 0 /* verbose_level */);
		cout << "perform_job_for_one_set saving the system "
				"to file " << fname_system << " done" << endl;
		//D->print();
	}

	long int nb_backtrack_nodes;
	long int *Sol;
	int nb_sol;

	D->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level);

	if (f_v) {
		cout << "before D->get_solutions" << endl;
	}
	D->get_solutions(Sol, nb_sol, verbose_level);
	if (f_v) {
		cout << "after D->get_solutions, nb_sol=" << nb_sol << endl;
	}
	string fname_solutions;

	fname_solutions.assign(arc_label);
	fname_solutions.append(".solutions");

	{
		ofstream fp(fname_solutions);
		int i, j, a;

		for (i = 0; i < nb_sol; i++) {
			fp << D->sum;
			for (j = 0; j < D->sum; j++) {
				a = Sol[i * D->sum + j];
				fp << " " << a;
			}
			fp << endl;
		}
		fp << -1 << " " << nb_sol << endl;
	}

	file_io Fio;

	cout << "Written file " << fname_solutions << " of size "
			<< Fio.file_size(fname_solutions) << endl;
	FREE_lint(Sol);
	FREE_lint(the_set_in);
	if (f_v) {
		cout << "arc_in_projective_space::arc_lifting2 done" << endl;
	}

}


void arc_in_projective_space::arc_lifting3(
		int arc_size,
		int arc_d,
		int arc_d_low,
		int arc_s,
		std::string arc_input_set,
		std::string arc_label,
		int arc_t,
		std::string t_lines_string,
		int arc_u,
		std::string u_lines_string,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_in_projective_space::arc_lifting3" << endl;
	}
	//int arc_size;
	//int arc_d;
	solvers::diophant *D = NULL;
	int f_save_system = TRUE;

	long int *t_lines;
	int nb_t_lines;
	long int *u_lines;
	int nb_u_lines;


	Orbiter->Lint_vec->scan(t_lines_string, t_lines, nb_t_lines);
	Orbiter->Lint_vec->scan(u_lines_string, u_lines, nb_u_lines);
	//lint_vec_print(cout, t_lines, nb_t_lines);
	//cout << endl;

	cout << "The t-lines, t=" << arc_t << " are ";
	Orbiter->Lint_vec->print(cout, t_lines, nb_t_lines);
	cout << endl;
	cout << "The u-lines, u=" << arc_u << " are ";
	Orbiter->Lint_vec->print(cout, u_lines, nb_u_lines);
	cout << endl;


	long int *the_set_in;
	int set_size_in;

	Orbiter->Lint_vec->scan(arc_input_set, the_set_in, set_size_in);


	arc_with_three_given_line_sets_diophant(
			the_set_in /* s_lines */, set_size_in /* nb_s_lines */, arc_s,
			t_lines, nb_t_lines, arc_t,
			u_lines, nb_u_lines, arc_u,
			arc_size /*target_sz*/, arc_d, arc_d_low,
			TRUE /* f_dualize */,
			D,
			verbose_level);

	if (FALSE) {
		D->print_tight();
	}
	if (f_save_system) {
		string fname_system;

		fname_system.assign(arc_label);
		fname_system.append(".diophant");

		cout << "arc_in_projective_space::perform_activity saving the system "
				"to file " << fname_system << endl;
		D->save_in_general_format(fname_system, 0 /* verbose_level */);
		cout << "arc_in_projective_space::perform_activity saving the system "
				"to file " << fname_system << " done" << endl;
		//D->print();
		//D->print_tight();
	}

	long int nb_backtrack_nodes;
	long int *Sol;
	int nb_sol;

	D->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level);

	if (f_v) {
		cout << "before D->get_solutions" << endl;
	}
	D->get_solutions(Sol, nb_sol, verbose_level);
	if (f_v) {
		cout << "after D->get_solutions, nb_sol=" << nb_sol << endl;
	}
	string fname_solutions;

	fname_solutions.assign(arc_label);
	fname_solutions.append(".solutions");

	{
		ofstream fp(fname_solutions);
		int i, j, a;

		for (i = 0; i < nb_sol; i++) {
			fp << D->sum;
			for (j = 0; j < D->sum; j++) {
				a = Sol[i * D->sum + j];
				fp << " " << a;
			}
			fp << endl;
		}
		fp << -1 << " " << nb_sol << endl;
	}
	file_io Fio;

	cout << "Written file " << fname_solutions << " of size "
			<< Fio.file_size(fname_solutions) << endl;
	FREE_lint(Sol);
	FREE_lint(the_set_in);
	if (f_v) {
		cout << "arc_in_projective_space::arc_lifting3 done" << endl;
	}

}

void arc_in_projective_space::create_hyperoval(
	int f_translation, int translation_exponent,
	int f_Segre, int f_Payne, int f_Cherowitzo, int f_OKeefe_Penttila,
	std::string &label_txt,
	std::string &label_tex,
	int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n = 2;
	int i, d;
	int *v;
	data_structures::sorting Sorting;
	char str[1000];
	char str2[1000];

	d = n + 1;

	if (f_v) {
		cout << "arc_in_projective_space::create_hyperoval" << endl;
	}


	v = NEW_int(d);
	Pts = NEW_lint(P->N_points);

	sprintf(str, "_q%d", P->F->q);
	sprintf(str2, "\\_q%d", P->F->q);

	if (f_translation) {
		P->Arc_in_projective_space->create_translation_hyperoval(Pts, nb_pts,
				translation_exponent, verbose_level - 0);
		label_txt.assign("hyperoval_translation");
		label_txt.append(str);
		label_tex.assign("hyperoval\\_translation");
		label_tex.append(str2);
	}
	else if (f_Segre) {
		P->Arc_in_projective_space->create_Segre_hyperoval(Pts, nb_pts, verbose_level - 2);
		label_txt.assign("hyperoval_Segre");
		label_txt.append(str);
		label_tex.assign("hyperoval\\_Segre");
		label_tex.append(str2);
	}
	else if (f_Payne) {
		P->Arc_in_projective_space->create_Payne_hyperoval(Pts, nb_pts, verbose_level - 2);
		label_txt.assign("hyperoval_Payne");
		label_txt.append(str);
		label_tex.assign("hyperoval\\_Payne");
		label_tex.append(str2);
	}
	else if (f_Cherowitzo) {
		P->Arc_in_projective_space->create_Cherowitzo_hyperoval(Pts, nb_pts, verbose_level - 2);
		label_txt.assign("hyperoval_Cherowitzo");
		label_txt.append(str);
		label_tex.assign("hyperoval\\_Cherowitzo");
		label_tex.append(str2);
	}
	else if (f_OKeefe_Penttila) {
		P->Arc_in_projective_space->create_OKeefe_Penttila_hyperoval_32(Pts, nb_pts,
				verbose_level - 2);
		label_txt.assign("hyperoval_OKeefe_Penttila");
		label_txt.append(str);
		label_tex.assign("hyperoval\\_OKeefe\\_Penttila");
		label_tex.append(str2);
	}
	else {
		P->Arc_in_projective_space->create_regular_hyperoval(Pts, nb_pts, verbose_level - 2);
		label_txt.assign("hyperoval_regular");
		label_txt.append(str);
		label_tex.assign("hyperoval\\_regular");
		label_tex.append(str2);
	}

	if (f_v) {
		cout << "i : point : projective rank" << endl;
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				Orbiter->Int_vec->print(cout, v, d);
				cout << endl;
			}
		}
	}

	if (!Sorting.test_if_set_with_return_value_lint(Pts, nb_pts)) {
		cout << "arc_in_projective_space::create_hyperoval the set is not a set, "
				"something is wrong" << endl;
		exit(1);
	}

	FREE_OBJECT(P);
	FREE_int(v);
	//FREE_int(L);
}

void arc_in_projective_space::create_subiaco_oval(
	int f_short,
	std::string &label_txt,
	std::string &label_tex,
	int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;
	char str[1000];
	char str2[1000];

	if (f_v) {
		cout << "arc_in_projective_space::create_subiaco_oval" << endl;
	}

	sprintf(str, "_q%d", P->F->q);
	sprintf(str2, "\\_q%d", P->F->q);
	{
		arc_basic A;

		A.init(P->F, verbose_level - 1);
		A.Subiaco_oval(Pts, nb_pts, f_short, verbose_level);
	}
	if (f_short) {
		label_txt.assign("oval_subiaco_short");
		label_txt.append(str);
		label_tex.assign("oval\\_subiaco\\_short");
		label_tex.append(str2);
	}
	else {
		label_txt.assign("oval_subiaco_long");
		label_txt.append(str);
		label_tex.assign("oval\\_subiaco\\_long");
		label_tex.append(str);
	}


	if (f_v) {
		int i;
		int n = 2, d = n + 1;
		int *v;

		v = NEW_int(d);

		cout << "i : point : projective rank" << endl;
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				Orbiter->Int_vec->print(cout, v, d);
				cout << endl;
			}
		}
		FREE_int(v);
		FREE_OBJECT(P);
	}

	if (!Sorting.test_if_set_with_return_value_lint(Pts, nb_pts)) {
		cout << "create_subiaco_oval the set is not a set, "
				"something is wrong" << endl;
		exit(1);
	}

}


void arc_in_projective_space::create_subiaco_hyperoval(
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;
	char str[1000];
	char str2[1000];

	if (f_v) {
		cout << "arc_in_projective_space::create_subiaco_hyperoval" << endl;
	}

	{
		arc_basic A;

		A.init(P->F, verbose_level - 1);
		A.Subiaco_hyperoval(Pts, nb_pts, verbose_level);
	}

	sprintf(str, "_q%d", P->F->q);
	sprintf(str2, "\\_q%d", P->F->q);

	label_txt.assign("subiaco_hyperoval");
	label_txt.append(str);
	label_tex.assign("subiaco\\_hyperoval");
	label_tex.append(str2);


	if (f_v) {
		int i;
		int n = 2, d = n + 1;
		int *v;

		v = NEW_int(d);
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				Orbiter->Int_vec->print(cout, v, d);
				cout << endl;
			}
		}
		FREE_int(v);
		FREE_OBJECT(P);
	}

	if (!Sorting.test_if_set_with_return_value_lint(Pts, nb_pts)) {
		cout << "arc_in_projective_space::create_subiaco_hyperoval "
				"the set is not a set, "
				"something is wrong" << endl;
		exit(1);
	}

}




}}

