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

void flock::init(
		blt_set_with_action *BLT_set,
		int point_idx, int verbose_level)
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

		// delete column 2 from the generator matrix of the hyperplane B[]
		// and copy into C[]
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


		plane_reduced = BLT_set->Blt_set_domain->G43->
				rank_lint_here(D, 0 /*verbose_level*/);
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

		BLT_set->Blt_set_domain->F->Projective_space_basic->
			PG_element_normalize_from_a_given_position(
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

		Flock_affine[j] = Gg.AG_element_rank(
				BLT_set->Blt_set_domain->F->q, F, 1, 3);

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
	Table_of_ABC->allocate_and_init(
			BLT_set->Blt_set_domain->F->q, 3, ABC);
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

	test_flock_condition(BLT_set->Blt_set_domain->F,
			FALSE /* f_magic */, ABC, verbose_level);



	PF = NEW_OBJECT(combinatorics::polynomial_function_domain);

	if (f_v) {
		cout << "flock::init before PF->init" << endl;
	}
	PF->init(BLT_set->Blt_set_domain->F, 1 /*n*/, verbose_level);
	if (f_v) {
		cout << "flock::init after PF->init" << endl;
	}

	int q;
	int *coeff_f;
	int *coeff_g;
	int nb_coeff;
	int degree;


	degree = PF->max_degree;
	q = BLT_set->Blt_set_domain->F->q;
	if (f_v) {
		cout << "flock::init degree = " << degree << endl;
		cout << "flock::init q = " << q << endl;
	}


	if (f_v) {
		cout << "flock::init before PF->algebraic_normal_form" << endl;
	}
	PF->algebraic_normal_form(
			func_f, q,
			coeff_f, nb_coeff,
			verbose_level);
	if (f_v) {
		cout << "flock::init after PF->algebraic_normal_form" << endl;
	}

	if (f_v) {
		cout << "flock::init before PF->algebraic_normal_form" << endl;
	}
	PF->algebraic_normal_form(
			func_g, q,
			coeff_g, nb_coeff,
			verbose_level);
	if (f_v) {
		cout << "flock::init after PF->algebraic_normal_form" << endl;
	}


	if (f_v) {
		cout << "flock::init before quadratic_lift" << endl;
	}
	quadratic_lift(coeff_f, coeff_g, verbose_level);
	if (f_v) {
		cout << "flock::init after quadratic_lift" << endl;
	}


	if (f_v) {
		cout << "flock::init before cubic_lift" << endl;
	}
	cubic_lift(coeff_f, coeff_g, verbose_level);
	if (f_v) {
		cout << "flock::init after cubic_lift" << endl;
	}




	if (f_v) {
		cout << "flock::init done" << endl;
	}
}

void flock::test_flock_condition(
		field_theory::finite_field *F,
		int f_magic, int *ABC, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);

	if (f_v) {
		cout << "flock::test_flock_condition" << endl;
	}

	int q;
	int N;
	int i, j;
	int ai, bi, ci;
	int aj, bj, cj;
	int a, b, c;
	int two, four;
	int x;
	int *outcome;
	int cnt;


	//F = BLT_set->Blt_set_domain->F;
	q = F->q;

	two = F->add(1, 1);
	four = F->add(two, two);

	N = (q * (q - 1)) >> 1;
	outcome = NEW_int(N);

	if (f_magic) {

#if 1
		// magical preprocessing:

		for (i = 0; i < q; i++) {

			ai = ABC[i * 3 + 0];
			ABC[i * 3 + 0] = F->mult(ai, F->p);

			bi = ABC[i * 3 + 1];
			ABC[i * 3 + 1] = F->mult(F->negate(1), F->p);

		}
#endif
	}

	cnt = 0;
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
			outcome[cnt] = F->is_square(x);

			if (f_vv) {
				if (outcome[cnt]) {
					cout << "i=" << i << ",j=" << j << ",x=" << x << " yes" << endl;
				}
				else {
					cout << "i=" << i << ",j=" << j << ",x=" << x << " no" << endl;
				}
			}
			cnt++;
		}
	}
	data_structures::tally T;

	T.init(outcome, N, FALSE, 0);
	cout << "outcome : ";
	T.print_first(FALSE /*f_backwards*/);
	cout << endl;


	FREE_int(outcome);

}

void flock::quadratic_lift(
		int *coeff_f, int *coeff_g, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);

	if (f_v) {
		cout << "flock::quadratic_lift" << endl;
	}

	int q;
	int nb_coeff;
	int degree;


	degree = PF->max_degree;
	q = BLT_set->Blt_set_domain->F->q;

	field_theory::finite_field *F2;

	F2 = NEW_OBJECT(field_theory::finite_field);

	int Q;

	Q = q * q;

	F2->finite_field_init_small_order(Q,
			FALSE /* f_without_tables */,
			FALSE /* f_compute_related_fields */,
			verbose_level);


	ring_theory::homogeneous_polynomial_domain *Poly2;

	Poly2 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "flock::quadratic_lift before Poly2->init" << endl;
	}
	Poly2->init(F2, 2, degree,
				t_PART,
				0 /* verbose_level */);
	if (f_v) {
		cout << "flock::quadratic_lift after Poly2->init" << endl;
	}



	int *lifted_f;
	int *lifted_g;


	lifted_f = NEW_int(Q);
	lifted_g = NEW_int(Q);

	int v[2];
	int i;

	for (i = 0; i < Q; i++) {
		//Gg.AG_element_unrank(Q, v, 1, 1, i);
		v[0] = i;
		v[1] = 1;
		lifted_f[i] = Poly2->evaluate_at_a_point(coeff_f, v);
		lifted_g[i] = Poly2->evaluate_at_a_point(coeff_g, v);
	}

	if (f_v) {

		cout << "flock::quadratic_lift lifted_f:" << endl;
		Int_vec_print(cout, lifted_f, Q);
		cout << endl;

		cout << "flock::quadratic_lift lifted_g:" << endl;
		Int_vec_print(cout, lifted_g, Q);
		cout << endl;
	}

	int *ABC2;

	ABC2 = NEW_int(Q * 3);
	for (i = 0; i < Q; i++) {
		ABC2[i * 3 + 0] = i;
		ABC2[i * 3 + 1] = lifted_f[i];
		ABC2[i * 3 + 2] = lifted_g[i];
	}

	test_flock_condition(F2, TRUE /* f_magic */, ABC2, verbose_level);

	if (f_v) {
		cout << "flock::quadratic_lift done" << endl;
	}
}



void flock::cubic_lift(
		int *coeff_f, int *coeff_g, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);

	if (f_v) {
		cout << "flock::cubic_lift" << endl;
	}

	int q;
	int nb_coeff;
	int degree;


	degree = PF->max_degree;
	q = BLT_set->Blt_set_domain->F->q;

	field_theory::finite_field *FQ;

	FQ = NEW_OBJECT(field_theory::finite_field);

	int Q;

	Q = q * q * q;

	FQ->finite_field_init_small_order(Q,
			FALSE /* f_without_tables */,
			FALSE /* f_compute_related_fields */,
			verbose_level);


	ring_theory::homogeneous_polynomial_domain *Poly2;

	Poly2 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "flock::cubic_lift before Poly2->init" << endl;
	}
	Poly2->init(FQ, 2, degree,
				t_PART,
				0 /* verbose_level */);
	if (f_v) {
		cout << "flock::cubic_lift after Poly2->init" << endl;
	}



	int *lifted_f;
	int *lifted_g;


	lifted_f = NEW_int(Q);
	lifted_g = NEW_int(Q);

	int v[2];
	int i;

	for (i = 0; i < Q; i++) {
		//Gg.AG_element_unrank(Q, v, 1, 1, i);
		v[0] = i;
		v[1] = 1;
		lifted_f[i] = Poly2->evaluate_at_a_point(coeff_f, v);
		lifted_g[i] = Poly2->evaluate_at_a_point(coeff_g, v);
	}

	if (f_v) {

		cout << "flock::cubic_lift lifted_f:" << endl;
		Int_vec_print(cout, lifted_f, Q);
		cout << endl;

		cout << "flock::cubic_lift lifted_g:" << endl;
		Int_vec_print(cout, lifted_g, Q);
		cout << endl;
	}

	int *ABC2;

	ABC2 = NEW_int(Q * 3);
	for (i = 0; i < Q; i++) {
		ABC2[i * 3 + 0] = i;
		ABC2[i * 3 + 1] = lifted_f[i];
		ABC2[i * 3 + 2] = lifted_g[i];
	}

	test_flock_condition(FQ, FALSE /* f_magic */, ABC2, verbose_level);

	if (f_v) {
		cout << "flock::cubic_lift done" << endl;
	}
}


}}}

