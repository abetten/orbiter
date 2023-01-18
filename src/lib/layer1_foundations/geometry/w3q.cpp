// w3q.cpp
// 
// Anton Betten
//
// started: March 4, 2011
// 
//
//

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {



W3q::W3q()
{
	q = 0;

	P3 = NULL;
	Q4 = NULL;
	F = NULL;
	Basis = NULL;

	nb_lines = 0;
	Lines = NULL;

	Q4_rk = NULL;
	Line_idx = NULL;

	//int v5[5];
}


W3q::~W3q()
{
	if (P3) {
		FREE_OBJECT(P3);
	}
	if (Q4) {
		FREE_OBJECT(Q4);
	}
	if (Basis) {
		FREE_int(Basis);
	}
	if (Lines) {
		FREE_int(Lines);
	}
	if (Q4_rk) {
		FREE_int(Q4_rk);
	}
	if (Line_idx) {
		FREE_int(Line_idx);
	}
}

void W3q::init(
		field_theory::finite_field *F, int verbose_level)
// allocates a projective_space and a orthogonal
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int f_vvv = FALSE; //(verbose_level >= 3);
	int h, rk;

	W3q::F = F;
	W3q::q = F->q;

	if (f_v) {
		cout << "W3q::init" << endl;
	}
	P3 = NEW_OBJECT(projective_space);
	Q4 = NEW_OBJECT(orthogonal_geometry::orthogonal);
	Basis = NEW_int(2 * 4);
	
	P3->projective_space_init(3, F,
		FALSE /* f_init_incidence_structure */, 
		verbose_level - 1  /*MINIMUM(verbose_level - 1, 3)*/);
	F = P3->F;
	Q4->init(0, 5, F, verbose_level - 1);


	if (f_v) {
		cout << "W3q::init before find_lines" << endl;
	}
	find_lines(verbose_level);
	if (f_v) {
		cout << "W3q::init after find_lines" << endl;
	}


	if (f_v) {
		print_lines();
	}

#if 0
	cout << "They are" << endl;
	int_vec_print(cout, Lines, nb_lines);
	cout << endl;
#endif

	if (nb_lines != Q4->Hyperbolic_pair->nb_points) {
		cout << "W3q::init nb_lines != Q4->nb_points" << endl;
		exit(1);
	}
	Q4_rk = NEW_int(nb_lines);
	Line_idx = NEW_int(nb_lines);


	for (h = 0; h < nb_lines; h++) {
		P3->unrank_line(Basis, Lines[h]);
		if (f_vv) {
			cout << "Line " << h << " is " << Lines[h] << ":" << endl;
			Int_vec_print_integer_matrix_width(cout,
					Basis, 2, 4, 4, F->log10_of_q);
			cout << endl;
		}

		isomorphism_Q4q(Basis, Basis + 4, v5);

		if (f_vvv) {
			cout << "v5=";
			Int_vec_print(cout, v5, 5);
			cout << endl;
		}
		
		rk = Q4->Hyperbolic_pair->rank_point(v5, 1, 0);

		if (f_vvv) {
			cout << "orthogonal point rank " << rk << endl;
		}
		
		Q4_rk[h] = rk;
		Line_idx[rk] = h;
	}
	

}

void W3q::find_lines(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, c;

	if (f_v) {
		cout << "W3q::find_lines" << endl;
	}
	Lines = NEW_int(P3->N_lines);
	nb_lines = 0;
	for (h = 0; h < P3->N_lines; h++) {
		P3->unrank_line(Basis, h);
		c = evaluate_symplectic_form(Basis, Basis + 4);
		//c = F->evaluate_symmetric_form(2, Basis, Basis + 4);
		if (c) {
			continue;
		}
		Lines[nb_lines++] = h;
	}
	cout << "We found " << nb_lines << " absolute lines" << endl;
	if (f_v) {
		cout << "W3q::find_lines done" << endl;
	}
}

void W3q::print_lines()
{
	int h;

	cout << "the lines are:" << endl;
	for (h = 0; h < nb_lines; h++) {
		cout << setw(4) << h << " : ";
		cout << setw(4) << Lines[h] << " : " << endl;
		P3->unrank_line(Basis, Lines[h]);
		Int_matrix_print(Basis, 2, 4);
		cout << endl;
	}
}

int W3q::evaluate_symplectic_form(int *x4, int *y4)
{
	return F->Linear_algebra->evaluate_symplectic_form(4, x4, y4);

	/*F->add4(
			F->mult(x4[0], y4[1]), 
			F->negate(F->mult(x4[1], y4[0])), 
			F->mult(x4[2], y4[3]), 
			F->negate(F->mult(x4[3], y4[2]))
		);*/
}

void W3q::isomorphism_Q4q(int *x4, int *y4, int *v)
{
	v[0] = F->Linear_algebra->Pluecker_12(x4, y4);
	v[1] = F->negate(F->Linear_algebra->Pluecker_13(x4, y4));
	v[2] = F->Linear_algebra->Pluecker_42(x4, y4);
	v[3] = F->negate(F->Linear_algebra->Pluecker_14(x4, y4));
	v[4] = F->Linear_algebra->Pluecker_23(x4, y4);
}


void W3q::print_by_lines()
{
	int h;
	cout << "The isomorphism is:" << endl;
	cout << "h : Lines[h] : Q4_rk[h] : Line_idx[h] : "
			"x : y : point in Q(4,q)" << endl;
	cout << "Where x and y are a basis for the line" << endl;
	for (h = 0; h < nb_lines; h++) {
		cout << setw(4) << h << " : ";
		cout << setw(4) << Lines[h] << " : ";
		cout << setw(4) << Q4_rk[h] << " : ";
		cout << setw(4) << Line_idx[h] << " : ";
		P3->unrank_line(Basis, Lines[h]);
		Int_vec_print(cout, Basis, 4);
		cout << " : ";
		Int_vec_print(cout, Basis + 4, 4);
		Q4->Hyperbolic_pair->unrank_point(v5, 1, Q4_rk[h], 0);
		cout << " : ";
		Int_vec_print(cout, v5, 5);
		cout << endl;
	}
}

void W3q::print_by_points()
{
	int h;
	cout << "The isomorphism is:" << endl;
	cout << "h : Line_idx[h] : Lines[Line_idx[h]] "
			"x : y : point in Q(4,q)" << endl;
	cout << "Where x and y are a basis for the line" << endl;
	for (h = 0; h < nb_lines; h++) {
		cout << setw(4) << h << " : ";
		cout << setw(4) << Line_idx[h] << " : ";
		cout << setw(4) << Lines[Line_idx[h]] << " : ";
		P3->unrank_line(Basis, Lines[Line_idx[h]]);
		Int_vec_print(cout, Basis, 4);
		cout << " : ";
		Int_vec_print(cout, Basis + 4, 4);
		Q4->Hyperbolic_pair->unrank_point(v5, 1, h, 0);
		cout << " : ";
		Int_vec_print(cout, v5, 5);
		cout << endl;
	}
}

int W3q::find_line(int line)
{
	int idx;
	data_structures::sorting Sorting;

	if (!Sorting.int_vec_search(Lines, nb_lines, line, idx)) {
		cout << "W3q::find_line could not find the line" << endl;
		exit(1);
	}
	return idx;
}

}}}

