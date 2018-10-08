// w3q.C
// 
// Anton Betten
//
// started: March 4, 2011
// 
//
//

#include "foundations.h"


W3q::W3q()
{
	null();
}

W3q::~W3q()
{
	freeself();
}

void W3q::null()
{
	q = 0;
	nb_lines = 0;
	P3 = NULL;
	Q4 = NULL;
	F = NULL;
	Basis = NULL;
	Lines = NULL;
	Q4_rk = NULL;
	Line_idx = NULL;
}

void W3q::freeself()
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
	null();
}

void W3q::init(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int h, c, rk;

	W3q::F = F;
	W3q::q = F->q;

	if (f_v) {
		cout << "W3q::init" << endl;
		}
	P3 = NEW_OBJECT(projective_space);
	Q4 = NEW_OBJECT(orthogonal);
	Basis = NEW_int(2 * 4);
	
	P3->init(3, F, 
		//TRUE /* f_init_group */, 
		//FALSE /* f_line_action */, 
		FALSE /* f_init_incidence_structure */, 
		//TRUE /* f_semilinear */, 
		//TRUE /* f_basis */,
		verbose_level - 1  /*MINIMUM(verbose_level - 1, 3)*/);
	F = P3->F;
	Q4->init(0, 5, F, verbose_level - 1);

	Lines = NEW_int(P3->N_lines);
	nb_lines = 0;
	for (h = 0; h < P3->N_lines; h++) {
		P3->unrank_line(Basis, h);
		c = evaluate_symplectic_form(Basis, Basis + 4);
		if (c) {
			continue;
			}
		Lines[nb_lines++] = h;
		}
	cout << "We found " << nb_lines << " Lines, they are" << endl;
	int_vec_print(cout, Lines, nb_lines);
	cout << endl;

	if (nb_lines != Q4->nb_points) {
		cout << "nb_lines != Q4->nb_points" << endl;
		exit(1);
		}
	Q4_rk = NEW_int(nb_lines);
	Line_idx = NEW_int(nb_lines);


	for (h = 0; h < nb_lines; h++) {
		P3->unrank_line(Basis, Lines[h]);
		if (f_vv) {
			cout << "Line " << h << " is " << Lines[h] << ":" << endl;
			print_integer_matrix_width(cout,
					Basis, 2, 4, 4, F->log10_of_q);
			cout << endl;
			}

		isomorphism_Q4q(Basis, Basis + 4, v5);

		if (f_vvv) {
			cout << "v5=";
			int_vec_print(cout, v5, 5);
			cout << endl;
			}
		
		rk = Q4->rank_point(v5, 1, 0);

		if (f_vvv) {
			cout << "orthogonal point rank " << rk << endl;
			}
		
		Q4_rk[h] = rk;
		Line_idx[rk] = h;
		}
	

	if (f_v) {
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
			int_vec_print(cout, Basis, 4);
			cout << " : ";
			int_vec_print(cout, Basis + 4, 4);
			Q4->unrank_point(v5, 1, Q4_rk[h], 0);
			cout << " : ";
			int_vec_print(cout, v5, 5);
			cout << endl;
			}
		}
}

int W3q::evaluate_symplectic_form(int *x4, int *y4)
{
	return F->evaluate_symplectic_form(4, x4, y4);

	/*F->add4(
			F->mult(x4[0], y4[1]), 
			F->negate(F->mult(x4[1], y4[0])), 
			F->mult(x4[2], y4[3]), 
			F->negate(F->mult(x4[3], y4[2]))
		);*/
}

void W3q::isomorphism_Q4q(int *x4, int *y4, int *v)
{
	v[0] = F->Pluecker_12(x4, y4);
	v[1] = F->negate(F->Pluecker_13(x4, y4));
	v[2] = F->Pluecker_42(x4, y4);
	v[3] = F->negate(F->Pluecker_14(x4, y4));
	v[4] = F->Pluecker_23(x4, y4);
}


