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


flock_from_blt_set::flock_from_blt_set()
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

	q = 0;
	coeff_f = NULL;
	coeff_g = NULL;
	nb_coeff = 0;
	degree = 0;

}

flock_from_blt_set::~flock_from_blt_set()
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
	if (coeff_f) {
		FREE_int(coeff_f);
	}
	if (coeff_g) {
		FREE_int(coeff_g);
	}

}

void flock_from_blt_set::init(
		blt_set_with_action *BLT_set,
		int point_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flock_from_blt_set::init" << endl;
	}

	geometry::geometry_global Gg;


	flock_from_blt_set::BLT_set = BLT_set;
	flock_from_blt_set::point_idx = point_idx;

	q = BLT_set->Blt_set_domain_with_action->Blt_set_domain->F->q;
	if (f_v) {
		cout << "flock::init q = " << q << " point_idx = " << point_idx << endl;
	}

	int target_size;
	int i, j;
	long int plane, plane_reduced;
	int B[3 * 5]; // generator matrix for the plane \Pi_i
	int C[3 * 4]; // generator matrix of \Pi_i after column 2 has been deleted
	int D[4 * 4]; // to compute the dual coordinates of the plane C
	int E[4]; // the last row of D will contain the dual coordinates of the plane C.
	int F[3]; // the flock function values aj, bj, cj for one j corresponding to \Pi_i

	target_size = BLT_set->Blt_set_domain_with_action->Blt_set_domain->target_size;
	Flock = NEW_lint(q);
	Flock_reduced = NEW_lint(q);
	Flock_affine = NEW_lint(q);
	ABC = NEW_int(q * 3);


	j = 0;
	for (i = 0; i < target_size; i++) {
		if (i == point_idx) {
			continue;
		}
		plane = BLT_set->Pi_ij[point_idx * target_size + i];

		BLT_set->Blt_set_domain_with_action->Blt_set_domain->G53->unrank_lint_here(
				B, plane, 0 /*verbose_level*/);

		if (f_v) {
			cout << "flock_from_blt_set::init " << i << " / "
					<< target_size << " B:" << endl;
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
			cout << "flock_from_blt_set::init " << i << " / "
					<< target_size << " C:" << endl;
			Int_matrix_print(C, 3, 4);
		}
		Int_vec_copy(C, D, 3 * 4);


		plane_reduced = BLT_set->Blt_set_domain_with_action->Blt_set_domain->G43->rank_lint_here(
				D, 0 /*verbose_level*/);

		Flock_reduced[j] = plane_reduced;

		Int_vec_copy(C, D, 3 * 4);

		BLT_set->Blt_set_domain_with_action->Blt_set_domain->F->Linear_algebra->perp_standard(
				4, 3,
				D, 0 /* verbose_level */);

		Int_vec_copy(
				D + 3 * 4, E, 4);

		if (f_v) {
			cout << "flock_from_blt_set::init " << i << " / "
					<< target_size << " E:" << endl;
			Int_matrix_print(E, 1, 4);
		}

		BLT_set->Blt_set_domain_with_action->Blt_set_domain->F->Projective_space_basic->
			PG_element_normalize_from_a_given_position(
				E, 1, 4, 1);

		if (f_v) {
			cout << "flock_from_blt_set::init " << i << " / "
					<< target_size << " E:" << endl;
			Int_matrix_print(E, 1, 4);
		}


		F[0] = E[2]; // ai
		F[1] = E[3]; // bi
		F[2] = E[0]; // ci

		if (f_v) {
			cout << "flock_from_blt_set::init " << i << " / "
					<< target_size << " F:" << endl;
			Int_matrix_print(F, 1, 3);
		}

		ABC[j * 3 + 0] = F[0];
		ABC[j * 3 + 1] = F[1];
		ABC[j * 3 + 2] = F[2];

		Flock_affine[j] = Gg.AG_element_rank(q, F, 1, 3);

		if (f_v) {
			cout << "flock_from_blt_set::init " << i << " / "
					<< target_size
					<< " Flock_affine=" << endl;
			cout << Flock_affine[j] << endl;
		}

		j++;
	}
	if (j != q) {
		cout << "flock_from_blt_set::init j != q" << endl;
		exit(1);
	}

	Table_of_ABC = NEW_OBJECT(data_structures::int_matrix);

	if (f_v) {
		cout << "flock_from_blt_set::init "
				"before Table_of_ABC->allocate_and_init" << endl;
	}
	Table_of_ABC->allocate_and_init(
			q, 3, ABC);
	if (f_v) {
		cout << "flock_from_blt_set::init "
				"after Table_of_ABC->allocate_and_init" << endl;
	}

	if (f_v) {
		cout << "flock_from_blt_set::init "
				"before Table_of_ABC->sort_rows" << endl;
	}
	Table_of_ABC->sort_rows(verbose_level);
	if (f_v) {
		cout << "flock_from_blt_set::init "
				"after Table_of_ABC->sort_rows" << endl;
	}

	func_f = NEW_int(q);
	func_g = NEW_int(q);

	for (i = 0; i < q; i++) {
		func_f[i] = Table_of_ABC->M[i * 3 + 1];
		func_g[i] = Table_of_ABC->M[i * 3 + 2];
	}

	if (f_v) {
		cout << "flock_from_blt_set::init Flock:" << endl;
		BLT_set->Blt_set_domain_with_action->Blt_set_domain->G53->print_set(
				Flock, q);

		cout << "flock_from_blt_set::init Flock_reduced:" << endl;
		BLT_set->Blt_set_domain_with_action->Blt_set_domain->G43->print_set(
				Flock_reduced, q);

		cout << "flock_from_blt_set::init ABC:" << endl;
		Int_matrix_print(ABC, q, 3);

		cout << "flock_from_blt_set::init Table_of_ABC:" << endl;
		Table_of_ABC->print();

		cout << "flock_from_blt_set::init func_f=" << endl;
		Int_vec_print(cout, func_f, q);
		cout << endl;
		cout << "flock_from_blt_set::init func_g=" << endl;
		Int_vec_print(cout, func_g, q);
		cout << endl;

	}


	int *outcome;
	int N;


	BLT_set->Blt_set_domain_with_action->Blt_set_domain->test_flock_condition(
			BLT_set->Blt_set_domain_with_action->Blt_set_domain->F,
			ABC, outcome, N, verbose_level);


	data_structures::tally T;

	T.init(outcome, N, false, 0);
	cout << "outcome : ";
	T.print_first(false /*f_backwards*/);
	cout << endl;

	FREE_int(outcome);

	degree = BLT_set->Blt_set_domain_with_action->PF->max_degree;
	if (f_v) {
		cout << "flock_from_blt_set::init degree = " << degree << endl;
		cout << "flock_from_blt_set::init q = " << q << endl;
	}


	if (f_v) {
		cout << "flock_from_blt_set::init "
				"before PF->algebraic_normal_form" << endl;
	}
	BLT_set->Blt_set_domain_with_action->PF->algebraic_normal_form(
			func_f, q,
			coeff_f, nb_coeff,
			verbose_level);
	if (f_v) {
		cout << "flock_from_blt_set::init "
				"after PF->algebraic_normal_form" << endl;
	}

	if (f_v) {
		cout << "flock_from_blt_set::init "
				"before PF->algebraic_normal_form" << endl;
	}
	BLT_set->Blt_set_domain_with_action->PF->algebraic_normal_form(
			func_g, q,
			coeff_g, nb_coeff,
			verbose_level);
	if (f_v) {
		cout << "flock_from_blt_set::init "
				"after PF->algebraic_normal_form" << endl;
	}




	if (f_v) {
		cout << "flock_from_blt_set::init coeff_f = " << endl;
		Int_vec_print(cout, coeff_f, nb_coeff);
		cout << endl;
	}

	if (f_v) {
		cout << "flock_from_blt_set::init coeff_g = " << endl;
		Int_vec_print(cout, coeff_g, nb_coeff);
		cout << endl;
	}


	if (f_v) {
		cout << "flock_from_blt_set::init done" << endl;
	}
}

void flock_from_blt_set::report(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flock_from_blt_set::report" << endl;
	}

	ost << "\\subsection*{Flock from BLT-set}" << endl;

	ost << "Point index: " << point_idx << "\\\\" << endl;

	ost << "$f=";
	Int_vec_print_fully(ost, func_f, q);
	ost << "$\\\\" << endl;

	ost << "$g=";
	Int_vec_print_fully(ost, func_g, q);
	ost << "$\\\\" << endl;

	ost << "polynomial function associated with $f=";
	Int_vec_print_fully(ost, coeff_f, nb_coeff);
	ost << "$\\\\" << endl;

	ost << "polynomial function associated with $g=";
	Int_vec_print_fully(ost, coeff_g, nb_coeff);
	ost << "$\\\\" << endl;

	cout << "polynomial_function_domain::algebraic_normal_form "
			"algebraic normal form in tex:" << endl;

	ost << "polynomial function associated with $f$ is" << endl;
	ost << "$$" << endl;
	BLT_set->Blt_set_domain_with_action->PF->Poly[
				BLT_set->Blt_set_domain_with_action->PF->max_degree
				].print_equation_tex(ost, coeff_f);
	ost << endl;
	ost << "$$" << endl;


	ost << "polynomial function associated with $g$ is" << endl;
	ost << "$$" << endl;
	BLT_set->Blt_set_domain_with_action->PF->Poly[
				BLT_set->Blt_set_domain_with_action->PF->max_degree
				].print_equation_tex(ost, coeff_g);
	ost << endl;
	ost << "$$" << endl;

	int *outcome;
	int N;

	BLT_set->Blt_set_domain_with_action->Blt_set_domain->test_flock_condition(
			BLT_set->Blt_set_domain_with_action->Blt_set_domain->F,
			ABC, outcome, N, verbose_level - 2);

	data_structures::tally T;

	T.init(outcome, N, false, 0);
	ost << "flock condition : ";
	//ost << "$";
	T.print_file_tex(ost, false /*f_backwards*/);
	//ost << "$";
	ost << "\\\\";
	ost << endl;

	FREE_int(outcome);

	if (f_v) {
		cout << "flock_from_blt_set::report done" << endl;
	}
}



}}}

