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
namespace geometry {
namespace other_geometry {



arc_in_projective_space::arc_in_projective_space()
{
	P = NULL;

}

arc_in_projective_space::~arc_in_projective_space()
{

}

void arc_in_projective_space::init(
		projective_geometry::projective_space *P, int verbose_level)
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

void arc_in_projective_space::create_arc_1_BCKM(
		long int *&the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	long int b, rk_Q;
	int v[3];
	vector<long int> Pts;

	if (P->Subspaces->n != 2) {
		cout << "arc_in_projective_space::create_arc_1_BCKM "
				"P->n != 2" << endl;
		exit(1);
	}
	if (P->Subspaces->F->p != 2) {
		cout << "arc_in_projective_space::create_arc_1_BCKM "
				"we must be in characteristic 2" << endl;
		exit(1);
	}

	long int *Hyperoval;
	int hyperoval_size;
	int frob_power;
	number_theory::number_theory_domain NT;


	hyperoval_size = P->Subspaces->q + 2;
	Hyperoval = NEW_lint(hyperoval_size);

	for (frob_power = 1; frob_power < P->Subspaces->F->e; frob_power++) {
		if (NT.gcd_lint(frob_power, P->Subspaces->F->e) == 1) {
			break;
		}
	}
	if (frob_power == P->Subspaces->F->e) {
		cout << "arc_in_projective_space::create_arc_1_BCKM "
				"could not determine the Frobenius power e" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "arc_in_projective_space::create_arc_1_BCKM "
				"frob_power = " << frob_power << endl;
	}

	for (i = 0; i < P->Subspaces->q + 2; i++) {
		if (i == P->Subspaces->q) {
			v[0] = 1;
			v[1] = 0;
			v[2] = 0;
		}
		else if (i == P->Subspaces->q + 1) {
			v[0] = 0;
			v[1] = 1;
			v[2] = 0;
		}
		else {
			v[0] = P->Subspaces->F->frobenius_power(
					i, frob_power);
			v[1] = i;
			v[2] = 1;
		}
		b = P->rank_point(v);
		Hyperoval[i] = b;
	}


	int intersection_number = 0;
	std::vector<long int> External_lines;
	std::vector<long int> External_lines_without;

	P->Subspaces->find_lines_by_intersection_number(
			Hyperoval, hyperoval_size,
		intersection_number,
		External_lines,
		verbose_level);

	if (f_v) {
		cout << "arc_in_projective_space::create_arc_1_BCKM "
				"number of external lines = " << External_lines.size() << endl;
	}

	v[0] = 1;
	v[1] = 1;
	v[2] = 0;
	rk_Q = P->rank_point(v);

	long int *line_pencil;
	int line_pencil_size;

	line_pencil_size = P->Subspaces->q + 1;
	line_pencil = NEW_lint(line_pencil_size);



	P->Subspaces->create_lines_on_point(
			rk_Q,
			line_pencil, 0 /* verbose_level */);


	data_structures::algorithms Algorithms;

	Algorithms.set_minus(
			External_lines, line_pencil, line_pencil_size,
			External_lines_without,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "arc_in_projective_space::create_arc_1_BCKM "
				"number of external lines after removing the "
				"pencil of Q = " << External_lines_without.size() << endl;
	}

	long int line_at_infinity_rk = 0;
	long int pt;
	long int pt_N, pt_Nprime;
	long int line_rk;

	vector<long int> N1, N2;

	pt_N = 1; // (0,1,0)
	pt_Nprime = 0; // (1,0,0)


	for (i = 0; i < P->Subspaces->q; i++) {
		pt = Hyperoval[i];
		if (P->Subspaces->F->absolute_trace(i) == 0) {
			line_rk = P->Subspaces->line_through_two_points(
						pt, pt_N);
			N1.push_back(line_rk);
		}
	}

	if (f_v) {
		cout << "arc_in_projective_space::create_arc_1_BCKM "
				"size of N1 = " << N1.size() << endl;
	}

	for (i = 0; i < P->Subspaces->q; i++) {
		pt = Hyperoval[i];
		if (P->Subspaces->F->absolute_trace(i) == 1) {
			line_rk = P->Subspaces->line_through_two_points(
						pt, pt_Nprime);
			N2.push_back(line_rk);
		}
	}

	if (f_v) {
		cout << "arc_in_projective_space::create_arc_1_BCKM "
				"size of N2 = " << N2.size() << endl;
	}

	size = 1 + N1.size() + N2.size() + External_lines_without.size();
	vector<long int> A;

	A.push_back(line_at_infinity_rk);
	for (i = 0; i < N1.size(); i++) {
		A.push_back(N1[i]);
	}
	for (i = 0; i < N2.size(); i++) {
		A.push_back(N2[i]);
	}
	for (i = 0; i < External_lines_without.size(); i++) {
		A.push_back(External_lines_without[i]);
	}
	if (f_v) {
		cout << "arc_in_projective_space::create_arc_1_BCKM "
				"size of A = " << A.size() << endl;
	}
	if (A.size() != size) {
		cout << "arc_in_projective_space::create_arc_1_BCKM "
				"A.size() != size" << endl;
		exit(1);
	}

	the_arc = NEW_lint(size);
	for (i = 0; i < size; i++) {
		line_rk = A[i];
		pt = P->Subspaces->Standard_polarity->Hyperplane_to_point[line_rk];
		the_arc[i] = pt;
	}

	FREE_lint(line_pencil);


#if 0
	for (i = 0; i < 4; i++) {
		frame[i] = P->rank_point(frame_data + i * 3);
	}

	if (f_v) {
		cout << "frame: ";
		Int_vec_print(cout, frame, 4);
		cout << endl;
	}

	L[0] = P->Subspaces->Implementation->line_through_two_points(frame[0], frame[1]);
	L[1] = P->Subspaces->Implementation->line_through_two_points(frame[1], frame[2]);
	L[2] = P->Subspaces->Implementation->line_through_two_points(frame[2], frame[0]);

	if (f_v) {
		cout << "l1=" << L[0] << " l2=" << L[1] << " l3=" << L[2] << endl;
	}

	// collect the 3*q points on the projective triangle:

	for (h = 0; h < 3; h++) {
		for (i = 0; i < P->Subspaces->r; i++) {
			b = P->Subspaces->Implementation->lines(L[h], i);
			Pts.push_back(b);
		}
	}
	if (f_v) {
		cout << "there are " << Pts.size() << " points on the three lines: ";
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}


	// add the q points (1,t,t^2), t in F_q
	// one of them lies on the projective triangle.

	for (i = 1; i < P->Subspaces->q; i++) {
		v[0] = 1;
		v[1] = i;
		v[2] = P->Subspaces->F->mult(i, i);
		b = P->rank_point(v);
		Pts.push_back(b);
	}


	Algorithms.filter_duplicates_and_make_array_of_long_int(
			Pts, the_arc, size,
			0 /*verbose_level */);

#endif


	if (f_v) {
		cout << "arc_in_projective_space::create_arc_1_BCKM: "
				"created arc of size =  " << size << " : ";
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}
}

void arc_in_projective_space::create_arc_2_BCKM(
		long int *&the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int frame_data[] = {1,0,0, 0,1,0,  0,0,1,  1,1,1 };
	int frame[4];
	int i, h;
	long int b;
	int L[3];
	int v[3];
	vector<long int> Pts;

	if (P->Subspaces->n != 2) {
		cout << "arc_in_projective_space::create_arc_2_BCKM "
				"P->n != 2" << endl;
		exit(1);
	}
#if 0
	if (P->Subspaces->q != 8) {
		cout << "arc_in_projective_space::create_arc_2_BCKM "
				"P->q != 8" << endl;
		exit(1);
	}
#endif
	for (i = 0; i < 4; i++) {
		frame[i] = P->rank_point(frame_data + i * 3);
	}

	if (f_v) {
		cout << "frame: ";
		Int_vec_print(cout, frame, 4);
		cout << endl;
	}

	L[0] = P->Subspaces->Implementation->line_through_two_points(frame[0], frame[2]);
	L[1] = P->Subspaces->Implementation->line_through_two_points(frame[2], frame[3]);
	L[2] = P->Subspaces->Implementation->line_through_two_points(frame[3], frame[0]);

	if (f_v) {
		cout << "l1=" << L[0] << " l2=" << L[1] << " l3=" << L[2] << endl;
	}

	for (h = 0; h < 3; h++) {
		for (i = 0; i < P->Subspaces->r; i++) {
			b = P->Subspaces->Implementation->lines(L[h], i);
			Pts.push_back(b);
		}
	}
	if (f_v) {
		cout << "there are " << Pts.size() << " points on the three lines: ";
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}


	for (i = 0; i < P->Subspaces->q; i++) {
		if (i == 1) {
			v[0] = 0;
			v[1] = 1;
			v[2] = 0;
		}
		else {
			v[0] = 1;
			v[1] = i;
			v[2] = P->Subspaces->F->mult(i, i);
		}
		b = P->rank_point(v);
		Pts.push_back(b);

	}

	data_structures::algorithms Algorithms;

	Algorithms.filter_duplicates_and_make_array_of_long_int(
			Pts, the_arc, size,
			0 /*verbose_level */);


	if (f_v) {
		cout << "arc_in_projective_space::create_arc_2_BCKM: "
				"after adding the rest of the conic, "
				"there are " << size << " points on the arc: ";
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}
}

void arc_in_projective_space::create_Maruta_Hamada_arc(
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N;

	if (f_v) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc" << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc "
				"P->n != 2" << endl;
		exit(1);
	}

	N = P->Subspaces->N_points;
	Pts = NEW_lint(N);

	create_Maruta_Hamada_arc2(Pts, nb_pts, verbose_level);

	label_tex = "Maruta_Hamada_arc2_q" + std::to_string(P->Subspaces->q);
	label_tex = "Maruta\\_Hamada\\_arc2\\_q" + std::to_string(P->Subspaces->q);

	//FREE_int(Pts);
	if (f_v) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc done" << endl;
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

	if (P->Subspaces->n != 2) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc "
				"n != 2" << endl;
		exit(1);
	}
	if (P->Subspaces->q != 13) {
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
		Int_vec_print(cout, points, 22);
		cout << endl;
	}

	L[0] = P->Subspaces->Implementation->line_through_two_points(1, 2);
	L[1] = P->Subspaces->Implementation->line_through_two_points(0, 2);
	L[2] = P->Subspaces->Implementation->line_through_two_points(0, 1);
	L[3] = P->Subspaces->Implementation->line_through_two_points(points[20], points[21]);

	if (f_v) {
		cout << "L:";
		Lint_vec_print(cout, L, 4);
		cout << endl;
	}

	if (f_v) {
		for (h = 0; h < 4; h++) {
			cout << "h=" << h << " : L[h]=" << L[h] << " : " << endl;
			for (i = 0; i < P->Subspaces->r; i++) {
				b = P->Subspaces->Implementation->lines(L[h], i);
					cout << "point " << b << " = ";
					P->unrank_point(v, b);
					P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
							v, 1, 3);
				Int_vec_print(cout, v, 3);
				cout << endl;
			}
			cout << endl;
		}
	}
	size = 0;
	for (h = 0; h < 4; h++) {
		for (i = 0; i < P->Subspaces->r; i++) {
			b = P->Subspaces->Implementation->lines(L[h], i);
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
		Lint_vec_print(cout, the_arc, size);
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
		Lint_vec_print(cout, the_arc, size);
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

	if (P->Subspaces->n != 2) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc2 "
				"P->n != 2" << endl;
		exit(1);
	}
	if (P->Subspaces->q != 13) {
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
		Int_vec_print(cout, points, 25);
		cout << endl;
	}
	for (i = 0; i < 9; i++) {
		L[i] = P->Subspaces->Standard_polarity->Point_to_hyperplane[points[i]];
	}
	size = 0;
	for (i = 0; i < 9; i++) {
		for (j = i + 1; j < 9; j++) {
			a = P->Subspaces->intersection_of_two_lines(L[i], L[j]);
			the_arc[size++] = a;
		}
	}
	for (i = 9; i < 24; i++) {
		the_arc[size++] = points[i];
	}
	if (f_v) {
		cout << "arc_in_projective_space::create_Maruta_Hamada_arc2: "
				"there are " << size << " points on the arc: ";
		Lint_vec_print(cout, the_arc, size);
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

	if (f_v) {
		cout << "arc_in_projective_space::create_pasch_arc" << endl;
	}
	if (P->Subspaces->n != 2) {
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
		cout << "arc_in_projective_space::create_pasch_arc points: ";
		Int_vec_print(cout, points, 5);
		cout << endl;
	}

	L[0] = P->Subspaces->Implementation->line_through_two_points(points[0], points[1]);
	L[1] = P->Subspaces->Implementation->line_through_two_points(points[0], points[3]);
	L[2] = P->Subspaces->Implementation->line_through_two_points(points[2], points[3]);
	L[3] = P->Subspaces->Implementation->line_through_two_points(points[1], points[4]);

	if (f_v) {
		cout << "L:";
		Int_vec_print(cout, L, 4);
		cout << endl;
	}

	size = 0;
	for (h = 0; h < 4; h++) {
		for (i = 0; i < P->Subspaces->r; i++) {
			b = P->Subspaces->Implementation->lines(L[h], i);
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
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}


	P->Subspaces->v[0] = 1;
	P->Subspaces->v[1] = 1;
	P->Subspaces->v[2] = 0;
	b = P->rank_point(P->Subspaces->v);
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
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}
	if (f_v) {
		cout << "arc_in_projective_space::create_pasch_arc done" << endl;
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

	if (f_v) {
		cout << "arc_in_projective_space::create_Cheon_arc" << endl;
	}
	if (P->Subspaces->n != 2) {
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
		Int_vec_print(cout, points, 5);
		cout << endl;
	}

	L[0] = P->Subspaces->Implementation->line_through_two_points(points[0], points[1]);
	L[1] = P->Subspaces->Implementation->line_through_two_points(points[1], points[2]);
	L[2] = P->Subspaces->Implementation->line_through_two_points(points[2], points[0]);

	if (f_v) {
		cout << "L:";
		Int_vec_print(cout, L, 3);
		cout << endl;
	}

	size = 0;
	for (h = 0; h < 3; h++) {
		for (i = 0; i < P->Subspaces->r; i++) {
			b = P->Subspaces->Implementation->lines(L[h], i);
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
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}




	for (h = 0; h < 3; h++) {

		if (f_v) {
			cout << "h=" << h << endl;
		}

		for (i = 0; i < P->Subspaces->r; i++) {
			pencil[i] =
					P->Subspaces->Implementation->lines_on_point(points[h], i);
		}


		j = 0;
		for (i = 0; i < P->Subspaces->r; i++) {
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
		Int_vec_print_integer_matrix_width(cout, Pencil, 3, 7, 7, 4);
	}

	for (i = 0; i < 7; i++) {
		a = Pencil[0 * 7 + i];
		for (j = 0; j < 7; j++) {
			b = Pencil[1 * 7 + j];
			if (f_v) {
				cout << "i=" << i << " a=" << a << " j="
						<< j << " b=" << b << endl;
			}
			c = P->Subspaces->Implementation->line_intersection(a, b);
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
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}

	if (f_v) {
		cout << "arc_in_projective_space::create_Cheon_arc done" << endl;
	}

}


void arc_in_projective_space::create_regular_hyperoval(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];

	if (f_v) {
		cout << "arc_in_projective_space::create_regular_hyperoval" << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "arc_in_projective_space::create_regular_hyperoval "
				"P->n != 2" << endl;
		exit(1);
	}

	for (i = 0; i < P->Subspaces->q; i++) {
		v[0] = P->Subspaces->F->mult(i, i);
		v[1] = i;
		v[2] = 1;
		the_arc[i] = P->rank_point(v);
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[P->Subspaces->q] = P->rank_point(v);

	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[P->Subspaces->q + 1] = P->rank_point(v);

	size = P->Subspaces->q + 2;

	if (f_v) {
		cout << "arc_in_projective_space::create_regular_hyperoval: "
				"there are " << size << " points on the arc: ";
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}
	if (f_v) {
		cout << "arc_in_projective_space::create_regular_hyperoval done" << endl;
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
	if (P->Subspaces->n != 2) {
		cout << "arc_in_projective_space::create_translation_hyperoval "
				"P->Subspaces->n != 2" << endl;
		exit(1);
	}

	for (i = 0; i < P->Subspaces->q; i++) {
		v[0] = P->Subspaces->F->frobenius_power(i, exponent);
		v[1] = i;
		v[2] = 1;
		the_arc[i] = P->rank_point(v);
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[P->Subspaces->q] = P->rank_point(v);

	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[P->Subspaces->q + 1] = P->rank_point(v);

	size = P->Subspaces->q + 2;

	if (f_v) {
		cout << "arc_in_projective_space::create_translation_hyperoval: "
				"there are " << size << " points on the arc: ";
		Lint_vec_print(cout, the_arc, size);
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

	if (f_v) {
		cout << "arc_in_projective_space::create_Segre_hyperoval" << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "arc_in_projective_space::create_Segre_hyperoval "
				"n != 2" << endl;
		exit(1);
	}

	for (i = 0; i < P->Subspaces->q; i++) {
		v[0] = P->Subspaces->F->power(i, 6);
		v[1] = i;
		v[2] = 1;
		the_arc[i] = P->rank_point(v);
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[P->Subspaces->q] = P->rank_point(v);

	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[P->Subspaces->q + 1] = P->rank_point(v);

	size = P->Subspaces->q + 2;

	if (f_v) {
		cout << "arc_in_projective_space::create_Segre_hyperoval: "
				"there are " << size << " points on the arc: ";
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}
	if (f_v) {
		cout << "arc_in_projective_space::create_Segre_hyperoval done" << endl;
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
	if (P->Subspaces->n != 2) {
		cout << "arc_in_projective_space::create_Payne_hyperoval P->n != 2" << endl;
		exit(1);
	}
	exponent = P->Subspaces->q - 1;
	a.create(6);
	b.create(exponent);

	D.extended_gcd(a, b, g, u, u2, 0 /* verbose_level */);
	one_sixth = u.as_int();
	while (one_sixth < 0) {
		one_sixth += exponent;
	}
	if (f_v) {
		cout << "one_sixth = " << one_sixth << endl;
	}

	a.create(2);
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

	for (i = 0; i < P->Subspaces->q; i++) {
		v[0] = P->Subspaces->F->add3(
				P->Subspaces->F->power(i, one_sixth),
				P->Subspaces->F->power(i, one_half),
				P->Subspaces->F->power(i, five_sixth));
		v[1] = i;
		v[2] = 1;
		the_arc[i] = P->rank_point(v);
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[P->Subspaces->q] = P->rank_point(v);

	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[P->Subspaces->q + 1] = P->rank_point(v);

	size = P->Subspaces->q + 2;

	if (f_v) {
		cout << "arc_in_projective_space::create_Payne_hyperoval: "
				"there are " << size << " points on the arc: ";
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}
	if (f_v) {
		cout << "arc_in_projective_space::create_Payne_hyperoval done" << endl;
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
	if (P->Subspaces->n != 2) {
		cout << "arc_in_projective_space::create_Cherowitzo_hyperoval "
				"P->n != 2" << endl;
		exit(1);
	}
	h = P->Subspaces->F->e;
	if (EVEN(h)) {
		cout << "arc_in_projective_space::create_Cherowitzo_hyperoval "
				"field degree must be odd" << endl;
		exit(1);
	}
	if (P->Subspaces->F->p != 2) {
		cout << "arc_in_projective_space::create_Cherowitzo_hyperoval "
				"needs characteristic 2" << endl;
		exit(1);
	}
	exponent = P->Subspaces->q - 1;
	one_half = (h + 1) >> 1;
	sigma = NT.i_power_j(2, one_half);
	e1 = sigma;
	e2 = (sigma + 2) % exponent;
	e3 = (3 * sigma + 4) % exponent;

	for (i = 0; i < P->Subspaces->q; i++) {
		v[0] = P->Subspaces->F->add3(
				P->Subspaces->F->power(i, e1),
				P->Subspaces->F->power(i, e2),
				P->Subspaces->F->power(i, e3));
		v[1] = i;
		v[2] = 1;
		the_arc[i] = P->rank_point(v);
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[P->Subspaces->q] = P->rank_point(v);

	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[P->Subspaces->q + 1] = P->rank_point(v);

	size = P->Subspaces->q + 2;

	if (f_v) {
		cout << "arc_in_projective_space::create_Cherowitzo_hyperoval: "
				"there are " << size << " points on the arc: ";
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}
	if (f_v) {
		cout << "arc_in_projective_space::create_Cherowitzo_hyperoval done" << endl;
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
	if (P->Subspaces->n != 2) {
		cout << "arc_in_projective_space::create_OKeefe_Penttila_hyperoval_32 "
				"P->n != 2" << endl;
		exit(1);
	}
	if (P->Subspaces->F->q != 32) {
		cout << "arc_in_projective_space::create_OKeefe_Penttila_hyperoval_32 "
				"needs q=32" << endl;
		exit(1);
	}

	{
		arc_basic A;

		A.init(P->Subspaces->F, verbose_level - 1);

		for (i = 0; i < P->Subspaces->q; i++) {


			v[0] = A.OKeefe_Penttila_32(i);
			v[1] = i;
			v[2] = 1;
			the_arc[i] = P->rank_point(v);
		}
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[P->Subspaces->q] = P->rank_point(v);

	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[P->Subspaces->q + 1] = P->rank_point(v);

	size = P->Subspaces->q + 2;

	if (f_v) {
		cout << "arc_in_projective_space::create_OKeefe_Penttila_hyperoval_32: "
				"there are " << size << " points on the arc: ";
		Lint_vec_print(cout, the_arc, size);
		cout << endl;
	}
	if (f_v) {
		cout << "arc_in_projective_space::create_OKeefe_Penttila_hyperoval_32 done"
				<< endl;
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

	free_points = NEW_lint(P->Subspaces->N_points);

	Combi.set_complement_lint(arc, arc_sz,
			free_points, nb_free_points, P->Subspaces->N_points);



	line_type = NEW_int(P->Subspaces->N_lines);
	P->Subspaces->line_intersection_type(arc, arc_sz,
			line_type, 0 /* verbose_level */);
	if (f_vv) {
		cout << "line_type: ";
		Int_vec_print_fully(cout, line_type, P->Subspaces->N_lines);
		cout << endl;
	}

	if (f_vv) {
		cout << "line type:" << endl;
		for (i = 0; i < P->Subspaces->N_lines; i++) {
			cout << i << " : " << line_type[i] << endl;
		}
	}

	data_structures::tally C;
	C.init(line_type, P->Subspaces->N_lines, false, 0);
	if (f_v) {
		cout << "arc_in_projective_space::arc_lifting_diophant line_type:";
		C.print_bare(true);
		cout << endl;
		cout << "nb_free_points=" << nb_free_points << endl;
	}


	D = NEW_OBJECT(solvers::diophant);
	D->open(P->Subspaces->N_lines + 1, nb_free_points, verbose_level - 1);
	//D->f_x_max = true;
	for (j = 0; j < nb_free_points; j++) {
		D->x_min[j] = 0;
		D->x_max[j] = 1;
	}
	D->f_has_sum = true;
	D->sum = target_sz - arc_sz;
	h = 0;
	for (i = 0; i < P->Subspaces->N_lines; i++) {
		if (line_type[i] > P->Subspaces->k) {
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
			if (P->Subspaces->is_incident(pt, i /* line */)) {
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

void arc_in_projective_space::create_diophant_for_arc_lifting_with_given_set_of_s_lines(
	long int *s_lines, int nb_s_lines,
	int target_sz, int arc_d, int arc_d_low, int arc_s,
	int f_dualize,
	solvers::diophant *&D,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	int i, j, h, a, line;

	// other_lines is the complement of s_lines
	long int *other_lines;
	int nb_other_lines;

	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "arc_in_projective_space::create_diophant_for_arc_lifting_with_given_set_of_s_lines" << endl;
	}

	other_lines = NEW_lint(P->Subspaces->N_points);

	Combi.set_complement_lint(
			s_lines, nb_s_lines,
			other_lines, nb_other_lines,
			P->Subspaces->N_lines);


	if (f_dualize) {
		if (P->Subspaces->Standard_polarity == NULL) {
			cout << "arc_in_projective_space::create_diophant_for_arc_lifting_with_given_set_of_s_lines "
					"Standard_polarity == NULL" << endl;
			exit(1);
		}
	}


	D = NEW_OBJECT(solvers::diophant);
	D->open(P->Subspaces->N_lines + 1, P->Subspaces->N_points, verbose_level - 1);
	//D->f_x_max = true;
	for (j = 0; j < P->Subspaces->N_points; j++) {
		D->x_min[j] = 0;
		D->x_max[j] = 1;
	}
	D->f_has_sum = true;
	D->sum = target_sz;
	h = 0;
	for (i = 0; i < nb_s_lines; i++) {
		if (f_dualize) {
			line = P->Subspaces->Standard_polarity->Point_to_hyperplane[s_lines[i]];
		}
		else {
			line = s_lines[i];
		}
		for (j = 0; j < P->Subspaces->N_points; j++) {
			if (P->Subspaces->is_incident(j, line)) {
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
			line = P->Subspaces->Standard_polarity->Point_to_hyperplane[other_lines[i]];
		}
		else {
			line = other_lines[i];
		}
		for (j = 0; j < P->Subspaces->N_points; j++) {
			if (P->Subspaces->is_incident(j, line)) {
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
	for (j = 0; j < P->Subspaces->N_points; j++) {
		D->Aij(h, j) = 1;
	}
	D->type[h] = t_EQ;
	D->RHSi(h) = target_sz;
	h++;

	D->m = h;

	//D->init_var_labels(N_points, verbose_level);

	if (f_vv) {
		cout << "arc_in_projective_space::create_diophant_for_arc_lifting_with_given_set_of_s_lines "
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
		cout << "arc_in_projective_space::create_diophant_for_arc_lifting_with_given_set_of_s_lines done" << endl;
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

	other_lines = NEW_lint(P->Subspaces->N_points);

	Lint_vec_copy(s_lines, other_lines, nb_s_lines);
	Lint_vec_copy(t_lines, other_lines + nb_s_lines, nb_t_lines);
	Sorting.lint_vec_heapsort(other_lines, nb_s_lines + nb_t_lines);

	Combi.set_complement_lint(other_lines, nb_s_lines + nb_t_lines,
			other_lines + nb_s_lines + nb_t_lines, nb_other_lines, P->Subspaces->N_lines);


	if (f_dualize) {
		if (P->Subspaces->Standard_polarity == NULL) {
			cout << "arc_in_projective_space::arc_with_two_given_line_sets_diophant "
					"Standard_polarity == NULL" << endl;
			exit(1);
		}
	}


	D = NEW_OBJECT(solvers::diophant);
	D->open(P->Subspaces->N_lines + 1, P->Subspaces->N_points, verbose_level - 1);
	//D->f_x_max = true;
	for (j = 0; j < P->Subspaces->N_points; j++) {
		D->x_min[j] = 0;
		D->x_max[j] = 1;
	}
	D->f_has_sum = true;
	D->sum = target_sz;
	h = 0;
	for (i = 0; i < nb_s_lines; i++) {
		if (f_dualize) {
			line = P->Subspaces->Standard_polarity->Point_to_hyperplane[s_lines[i]];
		}
		else {
			line = s_lines[i];
		}
		for (j = 0; j < P->Subspaces->N_points; j++) {
			if (P->Subspaces->is_incident(j, line)) {
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
			line = P->Subspaces->Standard_polarity->Point_to_hyperplane[t_lines[i]];
		}
		else {
			line = t_lines[i];
		}
		for (j = 0; j < P->Subspaces->N_points; j++) {
			if (P->Subspaces->is_incident(j, line)) {
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
			line = P->Subspaces->Standard_polarity->Point_to_hyperplane[l];
		}
		else {
			line = l;
		}
		for (j = 0; j < P->Subspaces->N_points; j++) {
			if (P->Subspaces->is_incident(j, line)) {
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
	for (j = 0; j < P->Subspaces->N_points; j++) {
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

	other_lines = NEW_lint(P->Subspaces->N_points);

	Lint_vec_copy(s_lines, other_lines, nb_s_lines);
	Lint_vec_copy(t_lines, other_lines + nb_s_lines, nb_t_lines);
	Lint_vec_copy(u_lines, other_lines + nb_s_lines + nb_t_lines, nb_u_lines);
	Sorting.lint_vec_heapsort(other_lines, nb_s_lines + nb_t_lines + nb_u_lines);

	Combi.set_complement_lint(
			other_lines, nb_s_lines + nb_t_lines + nb_u_lines,
			other_lines + nb_s_lines + nb_t_lines + nb_u_lines, nb_other_lines,
			P->Subspaces->N_lines);


	if (f_dualize) {
		if (P->Subspaces->Standard_polarity->Point_to_hyperplane == NULL) {
			cout << "arc_in_projective_space::arc_with_three_given_line_sets_diophant "
					"Polarity_point_to_hyperplane == NULL" << endl;
			exit(1);
		}
	}


	D = NEW_OBJECT(solvers::diophant);
	D->open(P->Subspaces->N_lines + 1, P->Subspaces->N_points, verbose_level - 1);
	//D->f_x_max = true;
	for (j = 0; j < P->Subspaces->N_points; j++) {
		D->x_min[j] = 0;
		D->x_max[j] = 1;
	}
	D->f_has_sum = true;
	D->sum = target_sz;
	h = 0;
	for (i = 0; i < nb_s_lines; i++) {
		if (f_dualize) {
			line = P->Subspaces->Standard_polarity->Point_to_hyperplane[s_lines[i]];
		}
		else {
			line = s_lines[i];
		}
		for (j = 0; j < P->Subspaces->N_points; j++) {
			if (P->Subspaces->is_incident(j, line)) {
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
			line = P->Subspaces->Standard_polarity->Point_to_hyperplane[t_lines[i]];
		}
		else {
			line = t_lines[i];
		}
		for (j = 0; j < P->Subspaces->N_points; j++) {
			if (P->Subspaces->is_incident(j, line)) {
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
			line = P->Subspaces->Standard_polarity->Point_to_hyperplane[u_lines[i]];
		}
		else {
			line = u_lines[i];
		}
		for (j = 0; j < P->Subspaces->N_points; j++) {
			if (P->Subspaces->is_incident(j, line)) {
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
			line = P->Subspaces->Standard_polarity->Point_to_hyperplane[l];
		}
		else {
			line = l;
		}
		for (j = 0; j < P->Subspaces->N_points; j++) {
			if (P->Subspaces->is_incident(j, line)) {
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
	for (j = 0; j < P->Subspaces->N_points; j++) {
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

	other_lines = NEW_int(P->Subspaces->N_lines);

	Int_vec_scan(secant_lines_text, secant_lines, nb_secant_lines);
	Int_vec_scan(external_lines_as_subset_of_secants_text, Idx, nb_external_lines);

	Sorting.int_vec_heapsort(secant_lines, nb_secant_lines);

	Combi.set_complement(
			secant_lines, nb_secant_lines,
			other_lines, nb_other_lines,
			P->Subspaces->N_lines);


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

	pencil_idx = NEW_int(P->Subspaces->N_lines);
	pencil_sub_idx = NEW_int(P->Subspaces->N_lines);
	nb_times_hit = NEW_int(P->Subspaces->k);
	Int_vec_zero(nb_times_hit, P->Subspaces->k);

	pencil_idx[0] = -1;
	for (i = 1; i < P->Subspaces->N_lines; i++) {
		pt = P->Subspaces->intersection_of_two_lines(i, 0);
		if (pt > P->Subspaces->k + 2) {
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
	for (pt = 0; pt < P->Subspaces->k - 2; pt++) {
		if (nb_times_hit[pt] != P->Subspaces->k - 1) {
			cout << "nb_times_hit[pt] != k - 1" << endl;
			cout << "pt = " << pt << endl;
			cout << "nb_times_hit[pt] = " << nb_times_hit[pt] << endl;
			exit(1);
		}
	}

	if (nb_other_lines != (P->Subspaces->k - 2) * (P->Subspaces->k - 1)) {
		cout << "nb_other_lines != (k - 2) * (k - 1)" << endl;
		exit(1);
	}
	nb_slack1 = nb_other_lines;
	nb_pencil_conditions = P->Subspaces->k - 2;
	slack1_start = P->Subspaces->N_points;


	nb_eqns = P->Subspaces->N_lines + 1 + nb_pencil_conditions;
	nb_vars = P->Subspaces->N_points + nb_slack1;

	D = NEW_OBJECT(solvers::diophant);
	D->open(nb_eqns, nb_vars, verbose_level - 1);
	//D->f_x_max = true;
	for (j = 0; j < nb_vars; j++) {
		D->x_min[j] = 0;
		D->x_max[j] = 1;
	}
	D->f_has_sum = true;
	D->sum = arc_sz + (arc_d - 1) * nb_pencil_conditions;
	h = 0;
	for (i = 0; i < nb_secant_lines; i++) {
		line = secant_lines[i];
		for (j = 0; j < P->Subspaces->N_points; j++) {
			if (P->Subspaces->is_incident(j, line)) {
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
		for (j = 0; j < P->Subspaces->N_points; j++) {
			if (P->Subspaces->is_incident(j, line)) {
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
		for (j = 0; j < P->Subspaces->N_points; j++) {
			if (P->Subspaces->is_incident(j, line)) {
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
		D->Aij(h, slack1_start + pt * (P->Subspaces->k - 1) + sub_idx) = arc_d;

		D->type[h] = t_EQ;
		D->RHSi(h) = arc_d;
		h++;
	}
	for (i = 0; i < nb_pencil_conditions; i++) {
		D->type[h] = t_EQ;
		D->RHSi(h) = arc_d - 1;
		for (j = 0; j < P->Subspaces->k - 1; j++) {
			D->Aij(h, slack1_start + i * (P->Subspaces->k - 1) + j) = 1;
		}
		h++;
	}
	// add one extra row:
	for (j = 0; j < P->Subspaces->N_points; j++) {
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

#if 0
void arc_in_projective_space::write_solutions_as_index_set(
		std::string &fname_solutions, long int *Sol, int nb_sol, int sum,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_in_projective_space::write_solutions_as_index_set" << endl;
	}

	{
		ofstream fp(fname_solutions);
		int i, j;
		long int a;

		for (i = 0; i < nb_sol; i++) {
			fp << sum;
			for (j = 0; j < sum; j++) {
				a = Sol[i * sum + j];
				fp << " " << a;
			}
			fp << endl;
		}
		fp << -1 << " " << nb_sol << endl;
	}
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname_solutions << " of size "
				<< Fio.file_size(fname_solutions) << endl;
	}

	if (f_v) {
		cout << "arc_in_projective_space::write_solutions_as_index_set done" << endl;
	}
}
#endif


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
	int f_save_system = true;

	long int *the_set_in;
	int set_size_in;

	Lint_vec_scan(arc_input_set, the_set_in, set_size_in);

	if (f_v) {
		cout << "arc_in_projective_space::arc_lifting1 "
				"before create_diophant_for_arc_lifting_with_given_set_of_s_lines" << endl;
	}
	create_diophant_for_arc_lifting_with_given_set_of_s_lines(
			the_set_in /*one_lines*/, set_size_in /* nb_one_lines */,
			arc_size /*target_sz*/, arc_d /* target_d */,
			arc_d_low, arc_s /* target_s */,
			true /* f_dualize */,
			D,
			verbose_level - 1);
	if (f_v) {
		cout << "arc_in_projective_space::arc_lifting1 "
				"after create_diophant_for_arc_lifting_with_given_set_of_s_lines" << endl;
	}

	if (false) {
		D->print_tight();
	}

	if (f_save_system) {

		string fname_system;

		fname_system = arc_label + ".diophant";
		cout << "perform_job_for_one_set saving the system "
				"to file " << fname_system << endl;
		D->save_in_general_format(fname_system, 0 /* verbose_level */);
		cout << "perform_job_for_one_set saving the system "
				"to file " << fname_system << " done" << endl;
		//D->print();
		//D->print_tight();
	}

	long int nb_backtrack_nodes;
	int *Sol;
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

	fname_solutions = arc_label + ".solutions";

	orbiter_kernel_system::file_io Fio;

	Fio.write_solutions_as_index_set(
			fname_solutions, Sol, nb_sol, D->n, D->sum, verbose_level);

	FREE_int(Sol);
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

	Lint_vec_scan(t_lines_string, t_lines, nb_t_lines);

	cout << "The t-lines, t=" << arc_t << " are ";
	Lint_vec_print(cout, t_lines, nb_t_lines);
	cout << endl;


	long int *the_set_in;
	int set_size_in;

	Lint_vec_scan(arc_input_set, the_set_in, set_size_in);


	solvers::diophant *D = NULL;
	int f_save_system = true;

	arc_with_two_given_line_sets_diophant(
			the_set_in /* s_lines */, set_size_in /* nb_s_lines */, arc_s,
			t_lines, nb_t_lines, arc_t,
			arc_size /*target_sz*/, arc_d, arc_d_low,
			true /* f_dualize */,
			D,
			verbose_level);

	if (false) {
		D->print_tight();
	}

	if (f_save_system) {
		string fname_system;

		fname_system = arc_label + ".diophant";

		cout << "perform_job_for_one_set saving the system "
				"to file " << fname_system << endl;
		D->save_in_general_format(fname_system, 0 /* verbose_level */);
		cout << "perform_job_for_one_set saving the system "
				"to file " << fname_system << " done" << endl;
		//D->print();
	}

	long int nb_backtrack_nodes;
	int *Sol;
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

	fname_solutions = arc_label + ".solutions";

	orbiter_kernel_system::file_io Fio;

	Fio.write_solutions_as_index_set(
			fname_solutions, Sol, nb_sol, D->n, D->sum, verbose_level);

	FREE_int(Sol);
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
	int f_save_system = true;

	long int *t_lines;
	int nb_t_lines;
	long int *u_lines;
	int nb_u_lines;


	Lint_vec_scan(t_lines_string, t_lines, nb_t_lines);
	Lint_vec_scan(u_lines_string, u_lines, nb_u_lines);
	//lint_vec_print(cout, t_lines, nb_t_lines);
	//cout << endl;

	cout << "The t-lines, t=" << arc_t << " are ";
	Lint_vec_print(cout, t_lines, nb_t_lines);
	cout << endl;
	cout << "The u-lines, u=" << arc_u << " are ";
	Lint_vec_print(cout, u_lines, nb_u_lines);
	cout << endl;


	long int *the_set_in;
	int set_size_in;

	Lint_vec_scan(arc_input_set, the_set_in, set_size_in);


	arc_with_three_given_line_sets_diophant(
			the_set_in /* s_lines */, set_size_in /* nb_s_lines */, arc_s,
			t_lines, nb_t_lines, arc_t,
			u_lines, nb_u_lines, arc_u,
			arc_size /*target_sz*/, arc_d, arc_d_low,
			true /* f_dualize */,
			D,
			verbose_level);

	if (false) {
		D->print_tight();
	}
	if (f_save_system) {
		string fname_system;

		fname_system = arc_label + ".diophant";

		cout << "arc_in_projective_space::perform_activity saving the system "
				"to file " << fname_system << endl;
		D->save_in_general_format(fname_system, 0 /* verbose_level */);
		cout << "arc_in_projective_space::perform_activity saving the system "
				"to file " << fname_system << " done" << endl;
		//D->print();
		//D->print_tight();
	}

	long int nb_backtrack_nodes;
	int *Sol;
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

	fname_solutions = arc_label + ".solutions";

	orbiter_kernel_system::file_io Fio;

	Fio.write_solutions_as_index_set(
			fname_solutions, Sol, nb_sol, D->n, D->sum, verbose_level);

	FREE_int(Sol);
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

	d = n + 1;

	if (f_v) {
		cout << "arc_in_projective_space::create_hyperoval" << endl;
	}


	v = NEW_int(d);
	Pts = NEW_lint(P->Subspaces->N_points);


	if (f_translation) {
		P->Arc_in_projective_space->create_translation_hyperoval(
				Pts, nb_pts,
				translation_exponent, verbose_level - 0);
		label_txt = "hyperoval_translation_q" + std::to_string(P->Subspaces->F->q);
		label_tex = "hyperoval\\_translation\\_q" + std::to_string(P->Subspaces->F->q);
	}
	else if (f_Segre) {
		P->Arc_in_projective_space->create_Segre_hyperoval(
				Pts, nb_pts, verbose_level - 2);
		label_txt = "hyperoval_Segre_q" + std::to_string(P->Subspaces->F->q);
		label_tex = "hyperoval\\_Segre\\_q" + std::to_string(P->Subspaces->F->q);
	}
	else if (f_Payne) {
		P->Arc_in_projective_space->create_Payne_hyperoval(
				Pts, nb_pts, verbose_level - 2);
		label_txt = "hyperoval_Payne_q" + std::to_string(P->Subspaces->F->q);
		label_tex = "hyperoval\\_Payne\\_q" + std::to_string(P->Subspaces->F->q);
	}
	else if (f_Cherowitzo) {
		P->Arc_in_projective_space->create_Cherowitzo_hyperoval(
				Pts, nb_pts, verbose_level - 2);
		label_txt = "hyperoval_Cherowitzo_q" + std::to_string(P->Subspaces->F->q);
		label_tex = "hyperoval\\_Cherowitzo\\_q" + std::to_string(P->Subspaces->F->q);
	}
	else if (f_OKeefe_Penttila) {
		P->Arc_in_projective_space->create_OKeefe_Penttila_hyperoval_32(
				Pts, nb_pts,
				verbose_level - 2);
		label_txt = "hyperoval_OKeefe_Penttila_q" + std::to_string(P->Subspaces->F->q);
		label_tex = "hyperoval\\_OKeefe\\_Penttila\\_q" + std::to_string(P->Subspaces->F->q);
	}
	else {
		P->Arc_in_projective_space->create_regular_hyperoval(
				Pts, nb_pts, verbose_level - 2);
		label_txt = "hyperoval_regular_q" + std::to_string(P->Subspaces->F->q);
		label_tex = "hyperoval\\_regular\\_q" + std::to_string(P->Subspaces->F->q);
	}

	if (f_v) {
		cout << "i : point : projective rank" << endl;
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				Int_vec_print(cout, v, d);
				cout << endl;
			}
		}
	}

	if (!Sorting.test_if_set_with_return_value_lint(
			Pts, nb_pts)) {
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

	if (f_v) {
		cout << "arc_in_projective_space::create_subiaco_oval" << endl;
	}

	{
		arc_basic A;

		A.init(P->Subspaces->F, verbose_level - 1);
		A.Subiaco_oval(Pts, nb_pts, f_short, verbose_level);
	}

	if (f_short) {
		label_txt = "oval_subiaco_short_q" + std::to_string(P->Subspaces->F->q);
		label_tex = "oval\\_subiaco\\_short\\_q" + std::to_string(P->Subspaces->F->q);
	}
	else {
		label_txt = "oval_subiaco_long_q" + std::to_string(P->Subspaces->F->q);
		label_tex = "oval\\_subiaco\\_long\\_q" + std::to_string(P->Subspaces->F->q);
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
				Int_vec_print(cout, v, d);
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

	if (f_v) {
		cout << "arc_in_projective_space::create_subiaco_hyperoval" << endl;
	}

	{
		arc_basic A;

		A.init(P->Subspaces->F, verbose_level - 1);
		A.Subiaco_hyperoval(Pts, nb_pts, verbose_level);
	}

	label_txt = "subiaco_hyperoval_q" + std::to_string(P->Subspaces->F->q);
	label_tex = "subiaco\\_hyperoval\\_q" + std::to_string(P->Subspaces->F->q);


	if (f_v) {
		int i;
		int n = 2, d = n + 1;
		int *v;

		v = NEW_int(d);
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				Int_vec_print(cout, v, d);
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

int arc_in_projective_space::arc_test(
		long int *input_pts, int nb_pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Pts;
	int *Mtx;
	int set[3];
	int ret = true;
	int h, i, N;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "arc_in_projective_space::arc_test" << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "arc_in_projective_space::arc_test P->n != 2" << endl;
		exit(1);
	}
	Pts = NEW_int(nb_pts * 3);
	Mtx = NEW_int(3 * 3);
	for (i = 0; i < nb_pts; i++) {
		P->unrank_point(Pts + i * 3, input_pts[i]);
	}
	if (f_v) {
		cout << "arc_in_projective_space::arc_test Pts=" << endl;
		Int_matrix_print(Pts, nb_pts, 3);
	}
	N = Combi.int_n_choose_k(nb_pts, 3);
	for (h = 0; h < N; h++) {
		Combi.unrank_k_subset(h, set, nb_pts, 3);
		Int_vec_copy(Pts + set[0] * 3, Mtx, 3);
		Int_vec_copy(Pts + set[1] * 3, Mtx + 3, 3);
		Int_vec_copy(Pts + set[2] * 3, Mtx + 6, 3);
		if (P->Subspaces->F->Linear_algebra->rank_of_matrix(
				Mtx, 3, 0 /* verbose_level */) < 3) {
			if (f_v) {
				cout << "Points P_" << set[0] << ", P_" << set[1]
					<< " and P_" << set[2] << " are collinear" << endl;
			}
			ret = false;
		}
	}

	FREE_int(Pts);
	FREE_int(Mtx);
	if (f_v) {
		cout << "arc_in_projective_space::arc_test done" << endl;
	}
	return ret;
}

void arc_in_projective_space::compute_bisecants_and_conics(
	long int *arc6,
	int *&bisecants, int *&conics, int verbose_level)
// bisecants[15 * 3]
// conics[6 * 6]
{
	int f_v = (verbose_level >= 1);
	long int i, j, h, pi, pj, Line[2];
	long int arc5[5];
	int six_coeffs[6];

	if (f_v) {
		cout << "arc_in_projective_space::compute_bisecants_and_conics" << endl;
	}
	bisecants = NEW_int(15 * 3);
	conics = NEW_int(6 * 6);

	h = 0;
	for (i = 0; i < 6; i++) {
		pi = arc6[i];
		for (j = i + 1; j < 6; j++, h++) {
			pj = arc6[j];
			Line[0] = pi;
			Line[1] = pj;
			P->Plane->determine_line_in_plane(
					Line,
				bisecants + h * 3,
				0 /* verbose_level */);
			P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
				bisecants + h * 3, 1, 3);
		}
	}
	if (f_v) {
		cout << "arc_in_projective_space::compute_bisecants_and_conics "
				"bisecants:" << endl;
		Int_matrix_print(bisecants, 15, 3);
	}

	for (j = 0; j < 6; j++) {
		//int deleted_point;

		//deleted_point = arc6[j];
		Lint_vec_copy(arc6, arc5, j);
		Lint_vec_copy(arc6 + j + 1, arc5 + j, 5 - j);

#if 0
		cout << "deleting point " << j << " / 6:";
		int_vec_print(cout, arc5, 5);
		cout << endl;
#endif

		P->Plane->determine_conic_in_plane(
				arc5, 5,
				six_coeffs, 0 /* verbose_level */);
		P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
				six_coeffs, 1, 6);
		Int_vec_copy(six_coeffs, conics + j * 6, 6);
	}

	if (f_v) {
		cout << "arc_in_projective_space::compute_bisecants_and_conics "
				"conics:" << endl;
		Int_matrix_print(conics, 6, 6);
	}

	if (f_v) {
		cout << "arc_in_projective_space::compute_bisecants_and_conics "
				"done" << endl;
	}
}

}}}}


