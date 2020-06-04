/*
 * web_of_cubic_curves.cpp
 *
 *  Created on: Jun 3, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;



namespace orbiter {
namespace foundations {




web_of_cubic_curves::web_of_cubic_curves()
{

	Surf = NULL;

	nb_T = 0;
	T_idx = NULL;

	arc6[0] = 0;
	base_curves4[0] = 0;
	Web_of_cubic_curves = NULL;
	The_plane_equations = NULL;
	base_curves = NULL;
	The_plane_rank = NULL;
	The_plane_duals = NULL;
	Dual_point_ranks = NULL;
	Lines27[0] = 0;

}

web_of_cubic_curves::~web_of_cubic_curves()
{
	if (Web_of_cubic_curves) {
		FREE_int(Web_of_cubic_curves);
	}
	if (The_plane_equations) {
		FREE_int(The_plane_equations);
	}
	if (base_curves) {
		FREE_int(base_curves);
	}
	if (The_plane_rank) {
		FREE_lint(The_plane_rank);
	}
	if (The_plane_duals) {
		FREE_lint(The_plane_duals);
	}
	if (Dual_point_ranks) {
		FREE_lint(Dual_point_ranks);
	}
}

void web_of_cubic_curves::init(surface_domain *Surf,
		long int *arc6, int *base_curves4,
		int nb_T, int *T_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "web_of_cubic_curves::init" << endl;
	}
	web_of_cubic_curves::Surf = Surf;
	web_of_cubic_curves::nb_T = nb_T;
	web_of_cubic_curves::T_idx = T_idx;
	lint_vec_copy(arc6, web_of_cubic_curves::arc6, 6);
	int_vec_copy(base_curves4, web_of_cubic_curves::base_curves4, 4);

	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"Creating the web of cubic "
				"curves through the arc:" << endl;
	}
	Surf->create_web_of_cubic_curves_and_equations_based_on_four_tritangent_planes(
		arc6, base_curves4,
		Web_of_cubic_curves, The_plane_equations,
		0 /*verbose_level*/);

	if (f_v) {
		cout << "arc_lifting::lift_prepare "
				"Testing the web of cubic curves:" << endl;
	}

	int pt_vec[3];
	int i, j, c;

	for (i = 0; i < 45; i++) {
		//cout << i << " / " << 45 << ":" << endl;
		for (j = 0; j < 6; j++) {
			Surf->P2->unrank_point(pt_vec, arc6[j]);
			c = Surf->Poly3->evaluate_at_a_point(
					Web_of_cubic_curves + i * 10, pt_vec);
			if (c) {
				cout << "web_of_cubic_curves::init "
						"the cubic curve does not "
						"pass through the arc" << endl;
				exit(1);
			}
		}
	}
	if (f_v) {
		cout << "web_of_cubic_curves::init The cubic curves all pass "
				"through the arc" << endl;
	}

	if (f_v) {
		cout << "arc_lifting::lift_prepare "
				"Computing the ranks of 4-subsets:" << endl;
	}

	int *Rk;
	int N;

	Surf->web_of_cubic_curves_rank_of_foursubsets(
		Web_of_cubic_curves,
		Rk, N, 0 /*verbose_level*/);
	{
		classify C;
		C.init(Rk, N, FALSE, 0 /* verbose_level */);
		cout << "web_of_cubic_curves::init "
				"classification of ranks of 4-subsets:" << endl;
		C.print_naked_tex(cout, TRUE /* f_backwards */);
		cout << endl;
	}

	FREE_int(Rk);

	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"Web_of_cubic_curves:" << endl;
		int_matrix_print(Web_of_cubic_curves, 45, 10);
	}

	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"base_curves4=";
		int_vec_print(cout, base_curves4, 4);
		cout << endl;
	}

	base_curves = NEW_int(4 * 10);
	for (i = 0; i < 4; i++) {
		int_vec_copy(Web_of_cubic_curves + base_curves4[i] * 10,
				base_curves + i * 10, 10);
	}
	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"base_curves:" << endl;
		int_matrix_print(base_curves, 4, 10);
	}



	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"The_plane_equations:" << endl;
		int_matrix_print(The_plane_equations, 45, 4);
	}

	The_plane_rank = NEW_lint(45);
	The_plane_duals = NEW_lint(45);

	latex_interface L;

	int Basis[16];
	for (i = 0; i < 45; i++) {
		int_vec_copy(The_plane_equations + i * 4, Basis, 4);
		Surf->F->RREF_and_kernel(4, 1, Basis, 0 /* verbose_level */);
		The_plane_rank[i] = Surf->rank_plane(Basis + 4);
	}
	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"The_plane_ranks:" << endl;
		L.print_lint_matrix_with_standard_labels(cout,
				The_plane_rank, 45, 1, TRUE /* f_tex */);
	}

	for (i = 0; i < 45; i++) {
		The_plane_duals[i] = Surf->rank_point(
				The_plane_equations + i * 4);
	}

	Dual_point_ranks = NEW_lint(nb_T * 6);

	cout << "web_of_cubic_curves::init "
			"computing Dual_point_ranks:" << endl;
	for (i = 0; i < nb_T; i++) {
		//cout << "trihedral pair " << i << " / "
		//<< Surf->nb_trihedral_pairs << endl;

		int e[6];

		int_vec_copy(Surf->Trihedral_to_Eckardt + T_idx[i] * 6, e, 6);
		for (j = 0; j < 6; j++) {
			Dual_point_ranks[i * 6 + j] = The_plane_duals[e[j]];
		}
	}

	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"Dual_point_ranks:" << endl;
		lint_matrix_print(Dual_point_ranks, nb_T, 6);
	}


	if (f_v) {
		cout << "web_of_cubic_curves::init before "
				"Surf->create_lines_from_plane_equations" << endl;
	}
	Surf->create_lines_from_plane_equations(
			The_plane_equations, Lines27, verbose_level);
	if (f_v) {
		cout << "web_of_cubic_curves::init after "
				"Surf->create_lines_from_plane_equations" << endl;
	}




	if (f_v) {
		cout << "web_of_cubic_curves::init done" << endl;
	}
}

}}
