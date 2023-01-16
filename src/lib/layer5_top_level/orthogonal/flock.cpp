/*
 * flock.cpp
 *
 *  Created on: Jan 14, 2023
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


flock::flock()
{
	BLT_set = NULL;

	point_idx = 0;

	Flock = NULL;
	Flock_reduced = NULL;
	Flock_affine = NULL;
	ABC = NULL;

	Table_of_ABC = NULL;

	func_f = NULL;
	func_g = NULL;

	PF = NULL;
}

flock::~flock()
{

	if (Flock) {
		FREE_lint(Flock);
	}
	if (Flock_reduced) {
		FREE_lint(Flock_reduced);
	}
	if (Flock_affine) {
		FREE_lint(Flock_affine);
	}
	if (ABC) {
		FREE_int(ABC);
	}
	if (Table_of_ABC) {
		FREE_OBJECT(Table_of_ABC);
	}
	if (func_f) {
		FREE_int(func_f);
	}
	if (func_g) {
		FREE_int(func_g);
	}
	if (PF) {
		FREE_OBJECT(PF);
	}

}

void flock::init(blt_set_with_action *BLT_set, int point_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flock::init" << endl;
	}

	geometry::geometry_global Gg;

	flock::BLT_set = BLT_set;
	flock::point_idx = point_idx;
	int i, j;
	long int plane, plane_reduced;
	int B[3 * 5];
	int C[3 * 4];
	int D[4 * 4];
	int E[4];
	int F[3];

	Flock = NEW_lint(BLT_set->Blt_set_domain->q);
	Flock_reduced = NEW_lint(BLT_set->Blt_set_domain->q);
	Flock_affine = NEW_lint(BLT_set->Blt_set_domain->q);
	ABC = NEW_int(BLT_set->Blt_set_domain->q * 3);


	j = 0;
	for (i = 0; i < BLT_set->Blt_set_domain->target_size; i++) {
		if (i == point_idx) {
			continue;
		}
		plane = BLT_set->Pi_ij[point_idx * BLT_set->Blt_set_domain->target_size + i];

		BLT_set->Blt_set_domain->G53->unrank_lint_here(B, plane, 0 /*verbose_level*/);

		if (f_v) {
			cout << "flock::init " << i << " / "
					<< BLT_set->Blt_set_domain->target_size << " B:" << endl;
			Int_matrix_print(B, 3, 5);
		}

		Flock[j] = plane;

		int u, v, w;

		for (u = 0; u < 3; u++) {
			w = 0;
			for (v = 0; v < 5; v++) {
				if (v == 2) {
					continue;
				}
				C[u * 4 + w] = B[u * 5 + v];
				w++;
			}
		}

		if (f_v) {
			cout << "flock::init " << i << " / "
					<< BLT_set->Blt_set_domain->target_size << " C:" << endl;
			Int_matrix_print(C, 3, 4);
		}
		Int_vec_copy(C, D, 3 * 4);


		plane_reduced = BLT_set->Blt_set_domain->G43->rank_lint_here(D, 0 /*verbose_level*/);
		Flock_reduced[j] = plane_reduced;

		Int_vec_copy(C, D, 3 * 4);

		BLT_set->Blt_set_domain->F->Linear_algebra->perp_standard(4, 3,
				D, 0 /* verbose_level */);

		Int_vec_copy(D + 3 * 4, E, 4);

		if (f_v) {
			cout << "flock::init " << i << " / "
					<< BLT_set->Blt_set_domain->target_size << " E:" << endl;
			Int_matrix_print(E, 1, 4);
		}

		BLT_set->Blt_set_domain->F->PG_element_normalize_from_a_given_position(
				E, 1, 4, 1);

		if (f_v) {
			cout << "flock::init " << i << " / "
					<< BLT_set->Blt_set_domain->target_size << " E:" << endl;
			Int_matrix_print(E, 1, 4);
		}


		F[0] = E[2]; // ai
		F[1] = E[3]; // bi
		F[2] = E[0]; // ci

		if (f_v) {
			cout << "flock::init " << i << " / "
					<< BLT_set->Blt_set_domain->target_size << " F:" << endl;
			Int_matrix_print(F, 1, 3);
		}

		ABC[j * 3 + 0] = F[0];
		ABC[j * 3 + 1] = F[1];
		ABC[j * 3 + 2] = F[2];

		Flock_affine[j] = Gg.AG_element_rank(BLT_set->Blt_set_domain->F->q, F, 1, 3);

		if (f_v) {
			cout << "flock::init " << i << " / "
					<< BLT_set->Blt_set_domain->target_size
					<< " Flock_affine=" << endl;
			cout << Flock_affine[j] << endl;
		}

		j++;
	}
	if (j != BLT_set->Blt_set_domain->q) {
		cout << "flock::init j != BLT_set->Blt_set_domain->q" << endl;
		exit(1);
	}

	Table_of_ABC = NEW_OBJECT(data_structures::int_matrix);

	if (f_v) {
		cout << "flock::init before Table_of_ABC->allocate_and_init" << endl;
	}
	Table_of_ABC->allocate_and_init(BLT_set->Blt_set_domain->F->q, 3, ABC);
	if (f_v) {
		cout << "flock::init after Table_of_ABC->allocate_and_init" << endl;
	}

	if (f_v) {
		cout << "flock::init before Table_of_ABC->sort_rows" << endl;
	}
	Table_of_ABC->sort_rows(verbose_level);
	if (f_v) {
		cout << "flock::init after Table_of_ABC->sort_rows" << endl;
	}

	func_f = NEW_int(BLT_set->Blt_set_domain->F->q);
	func_g = NEW_int(BLT_set->Blt_set_domain->F->q);

	for (i = 0; i < BLT_set->Blt_set_domain->F->q; i++) {
		func_f[i] = Table_of_ABC->M[i * 3 + 1];
		func_g[i] = Table_of_ABC->M[i * 3 + 2];
	}
	if (f_v) {
		cout << "flock::init Flock:" << endl;
		BLT_set->Blt_set_domain->G53->print_set(Flock, BLT_set->Blt_set_domain->q);

		cout << "flock::init Flock_reduced:" << endl;
		BLT_set->Blt_set_domain->G43->print_set(Flock_reduced, BLT_set->Blt_set_domain->q);

		cout << "flock::init ABC:" << endl;
		Int_matrix_print(ABC, BLT_set->Blt_set_domain->q, 3);

		cout << "flock::init Table_of_ABC:" << endl;
		Table_of_ABC->print();

		cout << "flock::init func_f=" << endl;
		Int_vec_print(cout, func_f, BLT_set->Blt_set_domain->F->q);
		cout << endl;
		cout << "flock::init func_g=" << endl;
		Int_vec_print(cout, func_g, BLT_set->Blt_set_domain->F->q);
		cout << endl;

	}


	PF = NEW_OBJECT(combinatorics::polynomial_function_domain);

	if (f_v) {
		cout << "flock::init before PF->init" << endl;
	}
	PF->init(BLT_set->Blt_set_domain->F, 1 /*n*/, verbose_level);
	if (f_v) {
		cout << "flock::init after PF->init" << endl;
	}

	int q;
	int *coeff;
	int nb_coeff;

	q = BLT_set->Blt_set_domain->F->q;


	if (f_v) {
		cout << "flock::init before PF->algebraic_normal_form" << endl;
	}
	PF->algebraic_normal_form(
			func_f, q,
			coeff, nb_coeff,
			verbose_level);
	if (f_v) {
		cout << "flock::init after PF->algebraic_normal_form" << endl;
	}

	if (f_v) {
		cout << "flock::init before PF->algebraic_normal_form" << endl;
	}
	PF->algebraic_normal_form(
			func_g, q,
			coeff, nb_coeff,
			verbose_level);
	if (f_v) {
		cout << "flock::init after PF->algebraic_normal_form" << endl;
	}





	test_flock_condition(verbose_level);

	if (f_v) {
		cout << "flock::init done" << endl;
	}
}

void flock::test_flock_condition(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flock::test_flock_condition" << endl;
	}

	field_theory::finite_field *F;
	int q;
	//field_theory::square_nonsquare *SN;
	int i, j;
	int ai, bi, ci;
	int aj, bj, cj;
	int a, b, c;
	int two, four;
	int x;


	F = BLT_set->Blt_set_domain->F;
	q = F->q;

	two = F->add(1, 1);
	four = F->add(two, two);


	//SN = NEW_OBJECT(field_theory::square_nonsquare);

	//SN->init(F, verbose_level);

	for (i = 0; i < q; i++) {
		ai = ABC[i * 3 + 0];
		bi = ABC[i * 3 + 1];
		ci = ABC[i * 3 + 2];
		for (j = i + 1; j < q; j++) {
			aj = ABC[j * 3 + 0];
			bj = ABC[j * 3 + 1];
			cj = ABC[j * 3 + 2];
			a = F->add(ai, F->negate(aj));
			b = F->add(bi, F->negate(bj));
			c = F->add(ci, F->negate(cj));
			x = F->add(F->mult(c, c), F->mult(four, F->mult(a, b)));
			if (F->is_square(x)) {
				cout << "i=" << i << ",j=" << j << ",x=" << x << " yes" << endl;
			}
			else {
				cout << "i=" << i << ",j=" << j << ",x=" << x << " no" << endl;
			}
		}
	}

	//FREE_OBJECT(SN);


}

}}}

