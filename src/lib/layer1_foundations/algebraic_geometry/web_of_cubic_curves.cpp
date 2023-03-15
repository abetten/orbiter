/*
 * web_of_cubic_curves.cpp
 *
 *  Created on: Jun 3, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


static void Web_of_cubic_curves_entry_print(int *p,
	int m, int n, int i, int j, int val,
	std::string &output, void *data);


web_of_cubic_curves::web_of_cubic_curves()
{

	Surf = NULL;

	nb_T = 0;
	T_idx = NULL;

	arc6[0] = 0;

	E = NULL;
	E_idx = NULL;
	T_idx = NULL;
	nb_T = 0;

	base_curves4[0] = 0;
	t_idx0 = 0;
	row_col_Eckardt_points[0] = 0;
	Web_of_cubic_curves = NULL;
	Tritangent_plane_equations = NULL;
	base_curves = NULL;
	The_plane_rank = NULL;
	The_plane_duals = NULL;
	Dual_point_ranks = NULL;
	Lines27[0] = 0;

}

web_of_cubic_curves::~web_of_cubic_curves()
{
	if (E) {
		FREE_OBJECT(E);
	}
	if (E_idx) {
		FREE_int(E_idx);
	}
	if (T_idx) {
		FREE_int(T_idx);
	}
	if (Web_of_cubic_curves) {
		FREE_int(Web_of_cubic_curves);
	}
	if (Tritangent_plane_equations) {
		FREE_int(Tritangent_plane_equations);
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
		long int *arc6, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h;

	if (f_v) {
		cout << "web_of_cubic_curves::init" << endl;
	}
	web_of_cubic_curves::Surf = Surf;

	Lint_vec_copy(arc6, web_of_cubic_curves::arc6, 6);


	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"before find_Eckardt_points" << endl;
	}
	find_Eckardt_points(verbose_level);
	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"after find_Eckardt_points" << endl;
	}

	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"before print_Eckardt_point_data" << endl;
	}
	print_Eckardt_point_data(cout, verbose_level);
	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"after print_Eckardt_point_data" << endl;
	}

	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"before find_trihedral_pairs" << endl;
	}
	find_trihedral_pairs(verbose_level);
	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"after find_trihedral_pairs" << endl;
	}

	//int t_idx0;
	//t_idx0 = T_idx[0];
	t_idx0 = T_idx[115];
	//t_idx = T_idx[0];
	//t_idx = T_idx[115];
	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"We choose trihedral pair t_idx0=" << t_idx0 << endl;
	}

	for (h = 0; h < 6; h++) {
		row_col_Eckardt_points[h] =
				Surf->Schlaefli->Trihedral_to_Eckardt[t_idx0 * 6 + h];
	}
	//int_vec_copy(Surf->Trihedral_to_Eckardt + t_idx0 * 6, row_col_Eckardt_points, 6);

#if 1
	base_curves4[0] = row_col_Eckardt_points[0];
	base_curves4[1] = row_col_Eckardt_points[1];
	base_curves4[2] = row_col_Eckardt_points[3];
	base_curves4[3] = row_col_Eckardt_points[4];
#else
	base_curves4[3] = row_col_Eckardt_points[0];
	base_curves4[0] = row_col_Eckardt_points[1];
	base_curves4[1] = row_col_Eckardt_points[3];
	base_curves4[2] = row_col_Eckardt_points[4];
#endif

	if (f_v) {
		cout << "web_of_cubic_curves::init base_curves4=";
		Int_vec_print(cout, base_curves4, 4);
		cout << endl;
	}

	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"Creating the web of cubic "
				"curves through the arc:" << endl;
	}
	create_web_and_equations_based_on_four_tritangent_planes(
		arc6, base_curves4,
		0 /*verbose_level*/);

	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"Testing the web of cubic curves:" << endl;
	}

	int pt_vec[3];
	int i, j, c;

	for (i = 0; i < 45; i++) {
		//cout << i << " / " << 45 << ":" << endl;

		for (j = 0; j < 6; j++) {

			Surf->P2->unrank_point(pt_vec, arc6[j]);

			c = Surf->PolynomialDomains->Poly3->evaluate_at_a_point(
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
		cout << "web_of_cubic_curves::init "
				"The cubic curves all pass through the arc" << endl;
	}

	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"Computing the ranks of 4-subsets:" << endl;
	}

	int *Rk;
	int N;

	rank_of_foursubsets(Rk, N, 0 /*verbose_level*/);
	{
		data_structures::tally C;
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
		Int_matrix_print(Web_of_cubic_curves, 45, 10);
	}

	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"base_curves4=";
		Int_vec_print(cout, base_curves4, 4);
		cout << endl;
	}

	base_curves = NEW_int(4 * 10);
	for (i = 0; i < 4; i++) {
		Int_vec_copy(Web_of_cubic_curves + base_curves4[i] * 10,
				base_curves + i * 10, 10);
	}
	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"base_curves:" << endl;
		Int_matrix_print(base_curves, 4, 10);
	}



	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"Tritangent_plane_equations:" << endl;
		Int_matrix_print(Tritangent_plane_equations, 45, 4);
	}

	The_plane_rank = NEW_lint(45);
	The_plane_duals = NEW_lint(45);

	orbiter_kernel_system::latex_interface L;

	int Basis[16];
	for (i = 0; i < 45; i++) {

		Int_vec_copy(Tritangent_plane_equations + i * 4, Basis, 4);

		Surf->F->Linear_algebra->RREF_and_kernel(4, 1, Basis, 0 /* verbose_level */);

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
				Tritangent_plane_equations + i * 4);
	}

	Dual_point_ranks = NEW_lint(nb_T * 6);

	cout << "web_of_cubic_curves::init "
			"computing Dual_point_ranks:" << endl;
	for (i = 0; i < nb_T; i++) {
		//cout << "trihedral pair " << i << " / "
		//<< Surf->nb_trihedral_pairs << endl;

		long int e[6];

		Lint_vec_copy(Surf->Schlaefli->Trihedral_to_Eckardt + T_idx[i] * 6, e, 6);
		for (j = 0; j < 6; j++) {
			Dual_point_ranks[i * 6 + j] = The_plane_duals[e[j]];
		}
	}

	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"Dual_point_ranks:" << endl;
		orbiter_kernel_system::Orbiter->Lint_vec->matrix_print(
				Dual_point_ranks, nb_T, 6);
	}


	if (f_v) {
		cout << "web_of_cubic_curves::init before "
				"Surf->create_lines_from_plane_equations" << endl;
	}
	Surf->create_lines_from_plane_equations(
			Tritangent_plane_equations, Lines27, verbose_level);
	if (f_v) {
		cout << "web_of_cubic_curves::init after "
				"Surf->create_lines_from_plane_equations" << endl;
	}




	if (f_v) {
		cout << "web_of_cubic_curves::init done" << endl;
	}
}


void web_of_cubic_curves::compute_web_of_cubic_curves(
		long int *arc6, int verbose_level)
// curves[45 * 10]
{
	int f_v = (verbose_level >= 1);
	int *bisecants;
	int *conics;
	int ten_coeff[10];
	int a, rk, i, j, k, l, m, n;
	int ij, kl, mn;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "web_of_cubic_curves::compute_web_of_cubic_curves" << endl;
	}
	Surf->P2->Arc_in_projective_space->compute_bisecants_and_conics(arc6,
		bisecants, conics, verbose_level);

	Web_of_cubic_curves = NEW_int(45 * 10);


	a = 0;

	// the first 30 curves:
	for (rk = 0; rk < 30; rk++, a++) {

		Combi.ordered_pair_unrank(rk, i, j, 6);

		ij = Combi.ij2k(i, j, 6);

		Surf->PolynomialDomains->multiply_conic_times_linear(conics + j * 6,
			bisecants + ij * 3,
			ten_coeff,
			0 /* verbose_level */);

		Int_vec_copy(ten_coeff, Web_of_cubic_curves + a * 10, 10);
	}

	// the next 15 curves:
	for (rk = 0; rk < 15; rk++, a++) {

		Combi.unordered_triple_pair_unrank(rk, i, j, k, l, m, n);

		ij = Combi.ij2k(i, j, 6);
		kl = Combi.ij2k(k, l, 6);
		mn = Combi.ij2k(m, n, 6);

		Surf->PolynomialDomains->multiply_linear_times_linear_times_linear(
			bisecants + ij * 3,
			bisecants + kl * 3,
			bisecants + mn * 3,
			ten_coeff,
			0 /* verbose_level */);

		Int_vec_copy(ten_coeff, Web_of_cubic_curves + a * 10, 10);
	}

	if (a != 45) {
		cout << "web_of_cubic_curves::compute_web_of_cubic_curves a != 45" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "The web of cubic curves is:" << endl;
		Int_matrix_print(Web_of_cubic_curves, 45, 10);
	}

	FREE_int(bisecants);
	FREE_int(conics);

	if (f_v) {
		cout << "web_of_cubic_curves::compute_web_of_cubic_curves done" << endl;
	}
}

void web_of_cubic_curves::rank_of_foursubsets(
	int *&rk, int &N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int set[4], i, j, a;
	int B[4 * 10];
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "web_of_cubic_curves::rank_of_foursubsets" << endl;
	}
	if (f_v) {
		cout << "web of cubic curves:" << endl;
		Int_matrix_print(Web_of_cubic_curves, 45, 10);
	}
	N = Combi.int_n_choose_k(45, 4);
	rk = NEW_int(N);
	for (i = 0; i < N; i++) {
		Combi.unrank_k_subset(i, set, 45, 4);
		if (f_v) {
			cout << "subset " << i << " / " << N << " is ";
			Int_vec_print(cout, set, 4);
			cout << endl;
		}
		for (j = 0; j < 4; j++) {
			a = set[j];
			Int_vec_copy(Web_of_cubic_curves + a * 10, B + j * 10, 10);
		}
		rk[i] = Surf->F->Linear_algebra->rank_of_rectangular_matrix(B,
			4, 10, 0 /* verbose_level */);
	}
	if (f_v) {
		cout << "web_of_cubic_curves::rank_of_foursubsets done" << endl;
	}
}

void web_of_cubic_curves::create_web_and_equations_based_on_four_tritangent_planes(
		long int *arc6, int *base_curves4,
		int verbose_level)
// Web_of_cubic_curves[45 * 10]
{
	int f_v = (verbose_level >= 1);
	int h, rk, idx;
	int *base_curves;
	int *curves;
	int *curves_t;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "web_of_cubic_curves::create_web_and_equations_based_on_four_tritangent_planes" << endl;
	}

	compute_web_of_cubic_curves(arc6, verbose_level);

	base_curves = NEW_int(5 * 10);
	curves = NEW_int(5 * 10);
	curves_t = NEW_int(10 * 5);



	for (h = 0; h < 4; h++) {
		Int_vec_copy(Web_of_cubic_curves + base_curves4[h] * 10,
			base_curves + h * 10, 10);
	}

	if (f_v) {
		cout << "base_curves:" << endl;
		Int_matrix_print(base_curves, 4, 10);
	}

	// find the plane equations:

	Tritangent_plane_equations = NEW_int(45 * 4);

	for (h = 0; h < 45; h++) {

		if (f_v) {
			cout << "h=" << h << " / " << 45 << ":" << endl;
		}

		if (Sorting.int_vec_search_linear(base_curves4, 4, h, idx)) {
			Int_vec_zero(Tritangent_plane_equations + h * 4, 4);
			Tritangent_plane_equations[h * 4 + idx] = 1;
		}
		else {
			Int_vec_copy(base_curves, curves, 4 * 10);
			Int_vec_copy(Web_of_cubic_curves + h * 10, curves + 4 * 10, 10);

			if (f_v) {
				cout << "h=" << h << " / " << 45
					<< " the system is:" << endl;
				Int_matrix_print(curves, 5, 10);
			}

			Surf->F->Linear_algebra->transpose_matrix(curves, curves_t, 5, 10);

			if (f_v) {
				cout << "after transpose:" << endl;
				Int_matrix_print(curves_t, 10, 5);
			}

			rk = Surf->F->Linear_algebra->RREF_and_kernel(5, 10, curves_t,
				0 /* verbose_level */);
			if (rk != 4) {
				cout << "web_of_cubic_curves::create_web_and_equations_based_on_four_tritangent_planes "
						"the rank of the "
						"system is not equal to 4" << endl;
				cout << "rk = " << rk << endl;
				exit(1);
			}
			if (curves_t[4 * 5 + 4] != Surf->F->negate(1)) {
				cout << "h=" << h << " / " << 2
					<< " curves_t[4 * 5 + 4] != -1" << endl;
				exit(1);
			}
			Int_vec_copy(curves_t + 4 * 5,
					Tritangent_plane_equations + h * 4, 4);

			Surf->F->Projective_space_basic->PG_element_normalize(
					Tritangent_plane_equations + h * 4, 1, 4);

		}
		if (f_v) {
			cout << "h=" << h << " / " << 45
				<< ": the plane equation is ";
			Int_vec_print(cout, Tritangent_plane_equations + h * 4, 4);
			cout << endl;
		}


	}
	if (f_v) {
		cout << "the plane equations are: " << endl;
		Int_matrix_print(Tritangent_plane_equations, 45, 4);
		cout << endl;
	}

	FREE_int(base_curves);
	FREE_int(curves);
	FREE_int(curves_t);

	if (f_v) {
		cout << "web_of_cubic_curves::create_web_and_equations_based_on_four_tritangent_planes done" << endl;
	}
}




void web_of_cubic_curves::find_Eckardt_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "web_of_cubic_curves::find_Eckardt_points" << endl;
	}
	int s;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "web_of_cubic_curves::find_Eckardt_points "
				"before Surf->P2->compute_eckardt_point_info" << endl;
	}
	E = Gg.compute_eckardt_point_info(Surf->P2, arc6, verbose_level);
	if (f_v) {
		cout << "web_of_cubic_curves::find_Eckardt_points "
				"after Surf->P2->compute_eckardt_point_info" << endl;
	}
	if (f_v) {
		cout << "web_of_cubic_curves::find_Eckardt_points "
				"We found " << E->nb_E
				<< " Eckardt points" << endl;
		for (s = 0; s < E->nb_E; s++) {
			cout << s << " / " << E->nb_E << " : ";
			E->E[s].print();
			cout << " = E_{" << s << "}";
			cout << endl;
		}
	}


	E_idx = NEW_int(E->nb_E);
	for (s = 0; s < E->nb_E; s++) {
		E_idx[s] = E->E[s].rank();
	}
	if (f_v) {
		cout << "by rank: ";
		Int_vec_print(cout, E_idx, E->nb_E);
		cout << endl;
	}
	if (f_v) {
		cout << "web_of_cubic_curves::find_Eckardt_points done" << endl;
	}
}

void web_of_cubic_curves::find_trihedral_pairs(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "web_of_cubic_curves::find_trihedral_pairs" << endl;
	}
#if 0
	Surf->find_trihedral_pairs_from_collinear_triples_of_Eckardt_points(
		E_idx, nb_E,
		T_idx, nb_T, verbose_level);
#else
	T_idx = NEW_int(120);
	nb_T = 120;
	for (i = 0; i < 120; i++) {
		T_idx[i] = i;
	}
#endif

	int t_idx;

	if (nb_T == 0) {
		cout << "nb_T == 0" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "List of special trihedral pairs:" << endl;
		for (i = 0; i < nb_T; i++) {
			t_idx = T_idx[i];
			cout << i << " / " << nb_T << ": T_{" << t_idx << "} =  T_{"
					<< Surf->Schlaefli->Trihedral_pair_labels[t_idx] << "}" << endl;
		}
	}

	if (f_v) {
		cout << "web_of_cubic_curves::find_trihedral_pairs done" << endl;
	}
}

void web_of_cubic_curves::create_surface_equation_from_trihedral_pair(
		long int *arc6,
	int t_idx, int *surface_equation,
	int &lambda,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *The_surface_equations;
	long int row_col_Eckardt_points[6];
	int The_six_plane_equations[6 * 4];
	int lambda_rk;

	if (f_v) {
		cout << "web_of_cubic_curves::create_surface_equation_from_trihedral_pair "
				"t_idx=" << t_idx << endl;
	}


	Lint_vec_copy(Surf->Schlaefli->Trihedral_to_Eckardt + t_idx * 6, row_col_Eckardt_points, 6);

	Int_vec_copy(Tritangent_plane_equations + row_col_Eckardt_points[0] * 4,
			The_six_plane_equations, 4);
	Int_vec_copy(Tritangent_plane_equations + row_col_Eckardt_points[1] * 4,
			The_six_plane_equations + 4, 4);
	Int_vec_copy(Tritangent_plane_equations + row_col_Eckardt_points[2] * 4,
			The_six_plane_equations + 8, 4);
	Int_vec_copy(Tritangent_plane_equations + row_col_Eckardt_points[3] * 4,
			The_six_plane_equations + 12, 4);
	Int_vec_copy(Tritangent_plane_equations + row_col_Eckardt_points[4] * 4,
			The_six_plane_equations + 16, 4);
	Int_vec_copy(Tritangent_plane_equations + row_col_Eckardt_points[5] * 4,
			The_six_plane_equations + 20, 4);


	The_surface_equations = NEW_int((Surf->q + 1) * 20);

	Surf->create_equations_for_pencil_of_surfaces_from_trihedral_pair(
		The_six_plane_equations, The_surface_equations,
		verbose_level - 2);

	create_lambda_from_trihedral_pair_and_arc(arc6,
		t_idx, lambda, lambda_rk,
		verbose_level - 2);

	Int_vec_copy(The_surface_equations + lambda_rk * 20,
		surface_equation, 20);

	FREE_int(The_surface_equations);

	if (f_v) {
		cout << "web_of_cubic_curves::create_surface_equation_from_trihedral_pair done" << endl;
	}
}

void web_of_cubic_curves::extract_six_curves_from_web(
	//int *row_col_Eckardt_points, int *six_curves,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "web_of_cubic_curves::extract_six_curves_from_web" << endl;
	}
	for (i = 0; i < 6; i++) {
		Int_vec_copy(Web_of_cubic_curves + row_col_Eckardt_points[i] * 10,
			six_curves + i * 10, 10);
	}

	if (f_v) {
		cout << "The six curves are:" << endl;
		Int_matrix_print(six_curves, 6, 10);
	}
	if (f_v) {
		cout << "web_of_cubic_curves::extract_six_curves_from_web done" << endl;
	}
}


void web_of_cubic_curves::create_lambda_from_trihedral_pair_and_arc(
	long int *arc6, int t_idx,
	int &lambda, int &lambda_rk,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int pt, f_point_was_found;
	int v[3];
	int w[2];
	int evals[6];
	int evals_for_point[6];
	int pt_on_surface[4];
	int a, b, ma, bv;

	if (f_v) {
		cout << "web_of_cubic_curves::create_lambda_from_trihedral_pair_and_arc "
				"t_idx=" << t_idx << endl;
	}

	if (f_v) {
		cout << "Trihedral pair T_{" << Surf->Schlaefli->Trihedral_pair_labels[t_idx] << "}"
			<< endl;
	}

	Lint_vec_copy(Surf->Schlaefli->Trihedral_to_Eckardt + t_idx * 6, row_col_Eckardt_points, 6);

	if (f_v) {
		cout << "row_col_Eckardt_points = ";
		Lint_vec_print(cout, row_col_Eckardt_points, 6);
		cout << endl;
	}



	extract_six_curves_from_web(/*row_col_Eckardt_points, six_curves,*/ verbose_level);

	if (f_v) {
		cout << "The six curves are:" << endl;
		Int_matrix_print(six_curves, 6, 10);
	}



	if (f_v) {
		cout << "web_of_cubic_curves::create_lambda_from_trihedral_pair_and_arc "
				"before find_point_not_on_six_curves" << endl;
	}
	find_point_not_on_six_curves(/*arc6,*/ /*six_curves,*/
		pt, f_point_was_found, verbose_level);
	if (!f_point_was_found) {
		cout << "web_of_cubic_curves::create_lambda_from_trihedral_pair_and_arc "
				"did not find point not on any of the six curves, "
				"picking lambda = 1" << endl;
		//exit(1);
		lambda = 1;
	}
	else {
		if (f_v) {
			cout << "web_of_cubic_curves::create_lambda_from_trihedral_pair_and_arc "
					"after find_point_not_on_six_curves" << endl;
			cout << "pt=" << pt << endl;
		}

		Surf->PolynomialDomains->Poly3->unrank_point(v, pt);
		for (i = 0; i < 6; i++) {
			evals[i] = Surf->PolynomialDomains->Poly3->evaluate_at_a_point(
					six_curves + i * 10, v);
		}

		if (f_v) {
			cout << "The point pt=" << pt << " = ";
			Int_vec_print(cout, v, 3);
			cout << " is nonzero on all plane sections of "
					"the trihedral pair. The values are ";
			Int_vec_print(cout, evals, 6);
			cout << endl;
		}

		if (f_v) {
			cout << "solving for lambda:" << endl;
		}
		a = Surf->F->mult3(evals[0], evals[1], evals[2]);
		b = Surf->F->mult3(evals[3], evals[4], evals[5]);
		ma = Surf->F->negate(a);
		bv = Surf->F->inverse(b);
		lambda = Surf->F->mult(ma, bv);

#if 1
		pt_on_surface[0] = evals[0];
		pt_on_surface[1] = evals[1];
		pt_on_surface[2] = evals[3];
		pt_on_surface[3] = evals[4];
#endif

		if (FALSE) {
			cout << "lambda = " << lambda << endl;
		}



		for (i = 0; i < 6; i++) {
			evals_for_point[i] =
					Surf->PolynomialDomains->Poly1_4->evaluate_at_a_point(
						Tritangent_plane_equations +
						row_col_Eckardt_points[i] * 4,
						pt_on_surface);
		}
		a = Surf->F->mult3(evals_for_point[0],
			evals_for_point[1],
			evals_for_point[2]);
		b = Surf->F->mult3(evals_for_point[3],
			evals_for_point[4],
			evals_for_point[5]);
		lambda = Surf->F->mult(Surf->F->negate(a), Surf->F->inverse(b));
		if (f_v) {
			cout << "lambda = " << lambda << endl;
		}
	}
	w[0] = 1;
	w[1] = lambda;
	Surf->F->Projective_space_basic->PG_element_rank_modified(
			w, 1, 2, lambda_rk);

	if (f_v) {
		cout << "web_of_cubic_curves::create_lambda_from_trihedral_pair_and_arc done" << endl;
	}
}

void web_of_cubic_curves::find_point_not_on_six_curves(
	int &pt, int &f_point_was_found,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int v[3];
	int i;
	int idx, a;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "web_of_cubic_curves::find_point_not_on_six_curves" << endl;
		cout << "web_of_cubic_curves::find_point_not_on_six_curves "
			"P2->N_points = " << Surf->P2->Subspaces->N_points << endl;
	}
	for (pt = 0; pt < Surf->P2->Subspaces->N_points; pt++) {
		if (Sorting.lint_vec_search_linear(arc6, 6, pt, idx)) {
			continue;
		}
		Surf->PolynomialDomains->Poly3->unrank_point(v, pt);
		for (i = 0; i < 6; i++) {
			a = Surf->PolynomialDomains->Poly3->evaluate_at_a_point(six_curves + i * 10, v);
			if (a == 0) {
				break;
			}
		}
		if (i == 6) {
			break;
		}
	}
	if (pt == Surf->P2->Subspaces->N_points) {
		cout << "web_of_cubic_curves::find_point_not_on_six_curves "
				"could not find a point which is not on "
				"any of the six curves" << endl;
		f_point_was_found = FALSE;
		pt = -1;
	}
	else {
		f_point_was_found = TRUE;
	}
	if (f_v) {
		cout << "web_of_cubic_curves::find_point_not_on_six_curves done" << endl;
	}
}

void web_of_cubic_curves::print_lines(std::ostream &ost)
{
	int i, a;
	int v[8];

	ost << "The 27 lines:\\\\";
	for (i = 0; i < 27; i++) {
		a = Lines27[i];
		ost << "$$" << endl;
		ost << "\\ell_{" << i << "} = "
				<< Surf->Schlaefli->Labels->Line_label_tex[i] << " = " << a << " = ";
		Surf->unrank_line(v, a);
		//ost << "\\left[ " << endl;
		Surf->Gr->print_single_generator_matrix_tex(ost, a);
		//ost << "\\right] ";
		ost << "$$" << endl;
	}
}

void web_of_cubic_curves::print_trihedral_plane_equations(std::ostream &ost)
{
	orbiter_kernel_system::latex_interface L;
	int i;

	ost << "The chosen abstract trihedral pair is no "
			<< t_idx0 << ":" << endl;
	ost << "$$" << endl;
	Surf->Schlaefli->latex_abstract_trihedral_pair(ost, t_idx0);
	ost << "$$" << endl;
	ost << "The six planes in the trihedral pair are:" << endl;
	ost << "$$" << endl;
	Lint_vec_print(ost, row_col_Eckardt_points, 6);
	ost << "$$" << endl;


	ost << "The six curves are:\\\\";
	for (i = 0; i < 6; i++) {
		ost << "$$" << endl;
		ost << "W_{" << Surf->Schlaefli->Eckard_point_label[row_col_Eckardt_points[i]];
		ost << "}=\\Phi\\big(\\pi_{" << row_col_Eckardt_points[i]
			<< "}\\big) = \\Phi\\big(\\pi_{"
			<< Surf->Schlaefli->Eckard_point_label[row_col_Eckardt_points[i]]
			<< "}\\big)=V\\Big(" << endl;
		Surf->PolynomialDomains->Poly3->print_equation(ost, six_curves + i * 10);
		ost << "\\Big)" << endl;
		ost << "$$" << endl;
	}


	ost << "The coefficients of the six curves are:\\\\";
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
			six_curves, 6, 10, TRUE /* f_tex*/);
	ost << "$$" << endl;



	ost << endl << "\\bigskip" << endl << endl;
	ost << "We choose planes $0,1,3,4$ for the base curves:" << endl;
	ost << "$$" << endl;
	Int_vec_print(ost, base_curves4, 4);
	ost << "$$" << endl;


	ost << "The four base curves are:\\\\";
	for (i = 0; i < 4; i++) {
		ost << "$$" << endl;
		ost << "W_{" << Surf->Schlaefli->Eckard_point_label[base_curves4[i]];
		ost << "}=\\Phi\\big(\\pi_{" << base_curves4[i]
			<< "}\\big) = \\Phi\\big(\\pi_{"
			<< Surf->Schlaefli->Eckard_point_label[base_curves4[i]]
			<< "}\\big)=V\\Big(" << endl;
		Surf->PolynomialDomains->Poly3->print_equation(ost, base_curves + i * 10);
		ost << "\\Big)" << endl;
		ost << "$$" << endl;
	}

	ost << "The coefficients of the four base curves are:\\\\";
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
			base_curves, 4, 10, TRUE /* f_tex*/);
	ost << "$$" << endl;

	ost << "The resulting tritangent plane equations are:\\\\";
	for (i = 0; i < 45; i++) {
		ost << "$\\pi_{" << i << "}=\\pi_{"
			<< Surf->Schlaefli->Eckard_point_label[i] << "}=V\\Big(";
		Surf->PolynomialDomains->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + i * 4);
		ost << "\\Big)$\\\\";
	}

	ost << "The dual coordinates of the plane equations are:\\\\";
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
			Tritangent_plane_equations, 15, 4, TRUE /* f_tex*/);
	ost << "\\;\\;" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
			Tritangent_plane_equations + 15 * 4, 15, 4, 15, 0, TRUE /* f_tex*/);
	ost << "\\;\\;" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
			Tritangent_plane_equations + 30 * 4, 15, 4, 30, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "The dual ranks are:\\\\";
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
		The_plane_duals, 15, 1, TRUE /* f_tex*/);
	ost << "\\;\\;" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		The_plane_duals + 15 * 1, 15, 1, 15, 0, TRUE /* f_tex*/);
	ost << "\\;\\;" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		The_plane_duals + 30 * 1, 15, 1, 30, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;

	print_lines(ost);
}

void web_of_cubic_curves::print_the_six_plane_equations(
	int *The_six_plane_equations,
	long int *plane6, std::ostream &ost)
{
	orbiter_kernel_system::latex_interface L;
	int i, h;

	ost << "The six plane equations are:" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
			The_six_plane_equations, 6, 4, TRUE /* f_tex*/);
	ost << "$$" << endl;

	ost << "The six plane equations are:\\\\";
	for (i = 0; i < 6; i++) {
		h = row_col_Eckardt_points[i];
		ost << "$\\pi_{" << h << "}=\\pi_{"
				<< Surf->Schlaefli->Eckard_point_label[h] << "}=V\\big(";
		Surf->PolynomialDomains->Poly1_4->print_equation(ost, Tritangent_plane_equations + h * 4);
		ost << "\\big)$\\\\";
	}
}

void web_of_cubic_curves::print_surface_equations_on_line(
	int *The_surface_equations,
	int lambda, int lambda_rk, std::ostream &ost)
{
	orbiter_kernel_system::latex_interface L;
	int i;
	int v[2];

	ost << "The $q+1$ equations on the line are:" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
			The_surface_equations, Surf->F->q + 1, 20, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$$" << endl;
	ost << "\\lambda = " << lambda << ", \\; \\mbox{in row} \\; "
			<< lambda_rk << endl;
	ost << "$$" << endl;

	ost << "The $q+1$ equations on the line are:\\\\" << endl;
	for (i = 0; i < Surf->F->q + 1; i++) {
		ost << "Row " << i << " : ";

		Surf->F->Projective_space_basic->PG_element_unrank_modified(
				v, 1, 2, i);
		Surf->F->Projective_space_basic->PG_element_normalize_from_front(
				v, 1, 2);

		ost << "$";
		ost << v[0] << " \\cdot ";
		ost << "\\big(";
		Surf->PolynomialDomains->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + row_col_Eckardt_points[0] * 4);
		ost << "\\big)";
		ost << "\\big(";
		Surf->PolynomialDomains->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + row_col_Eckardt_points[1] * 4);
		ost << "\\big)";
		ost << "\\big(";
		Surf->PolynomialDomains->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + row_col_Eckardt_points[2] * 4);
		ost << "\\big)";
		ost << "+";
		ost << v[1] << " \\cdot ";
		ost << "\\big(";
		Surf->PolynomialDomains->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + row_col_Eckardt_points[3] * 4);
		ost << "\\big)";
		ost << "\\big(";
		Surf->PolynomialDomains->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + row_col_Eckardt_points[4] * 4);
		ost << "\\big)";
		ost << "\\big(";
		Surf->PolynomialDomains->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + row_col_Eckardt_points[5] * 4);
		ost << "\\big)";
		ost << " = ";
		Surf->PolynomialDomains->Poly3_4->print_equation(ost,
				The_surface_equations + i * 20);
		ost << "$\\\\";
	}
}

void web_of_cubic_curves::print_dual_point_ranks(std::ostream &ost)
{
	orbiter_kernel_system::latex_interface L;

	ost << "Dual point ranks:\\\\";
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
			Dual_point_ranks, nb_T, 6, TRUE /* f_tex*/);
	ost << "$$" << endl;
}

void web_of_cubic_curves::print_Eckardt_point_data(
		std::ostream &ost, int verbose_level)
{
	E->print_bisecants(ost, verbose_level);
	E->print_intersections(ost, verbose_level);
	E->print_conics(ost, verbose_level);
}

void web_of_cubic_curves::report_basics(std::ostream &ost, int verbose_level)
{
	Surf->print_basics(ost);
#if 0
	Surf->print_polynomial_domains(ost);
	Surf->print_Schlaefli_labelling(ost);
	Surf->print_Steiner_and_Eckardt(ost);
#endif

}

void web_of_cubic_curves::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "web_of_cubic_curves::report" << endl;
	}


	ost << "Web of cubic curves:\\\\" << endl << endl;

	if (f_v) {
		cout << "web_of_cubic_curves::report before print_Eckardt_point_data" << endl;
	}
	print_Eckardt_point_data(ost, verbose_level);
	if (f_v) {
		cout << "web_of_cubic_curves::report after print_Eckardt_point_data" << endl;
	}

	if (f_v) {
		cout << "web_of_cubic_curves::report before E->print_Eckardt_points" << endl;
	}
	E->print_Eckardt_points(ost, verbose_level);
	if (f_v) {
		cout << "web_of_cubic_curves::report before print_web_of_cubic_curves" << endl;
	}
	print_web_of_cubic_curves(arc6, ost);


	if (f_v) {
		cout << "web_of_cubic_curves::report before print_plane_equations" << endl;
	}
	print_trihedral_plane_equations(ost);


	//cout << "web_of_cubic_curves::report before print_dual_point_ranks" << endl;
	//print_dual_point_ranks(ost);

	//ost << "Reporting web of cubic curves done.\\\\";


	if (f_v) {
		cout << "web_of_cubic_curves::report done" << endl;
	}

}

void web_of_cubic_curves::print_web_of_cubic_curves(long int *arc6, std::ostream &ost)
{
	combinatorics::combinatorics_domain Combi;
	orbiter_kernel_system::latex_interface L;

	ost << "The web of cubic curves is:\\\\" << endl;

#if 0
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
		Web_of_cubic_curves, 15, 10, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
		Web_of_cubic_curves + 15 * 10, 15, 10, 15, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
		Web_of_cubic_curves + 30 * 10, 15, 10, 30, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;
#endif

	int *bisecants;
	int *conics;

	int labels[15];
	int row_fst[1];
	int row_len[1];
	int col_fst[1];
	int col_len[1];
	row_fst[0] = 0;
	row_len[0] = 15;
	col_fst[0] = 0;
	col_len[0] = 10;
	char str[1000];
	int i, j, k, l, m, n, h, ij, kl, mn;

	Surf->P2->Arc_in_projective_space->compute_bisecants_and_conics(arc6,
			bisecants, conics, 0 /*verbose_level*/);

	for (h = 0; h < 45; h++) {
		ost << "$";
		snprintf(str, 1000, "W_{%s}=\\Phi\\big(\\pi_{%d}\\big) "
				"= \\Phi\\big(\\pi_{%s}\\big)",
				Surf->Schlaefli->Eckard_point_label[h].c_str(), h,
				Surf->Schlaefli->Eckard_point_label[h].c_str());
		ost << str;
		ost << " = ";
		if (h < 30) {
			Combi.ordered_pair_unrank(h, i, j, 6);
			ij = Combi.ij2k(i, j, 6);
			ost << "C_" << j + 1
				<< "P_{" << i + 1 << "}P_{" << j + 1 << "} = ";
			ost << "\\big(";
			Surf->PolynomialDomains->Poly2->print_equation(ost, conics + j * 6);
			ost << "\\big)";
			ost << "\\big(";
			Surf->PolynomialDomains->Poly1->print_equation(ost, bisecants + ij * 3);
			ost << "\\big)";
			//multiply_conic_times_linear(conics + j * 6,
			//bisecants + ij * 3, ten_coeff, 0 /* verbose_level */);
		}
		else {
			Combi.unordered_triple_pair_unrank(h - 30, i, j, k, l, m, n);
			ij = Combi.ij2k(i, j, 6);
			kl = Combi.ij2k(k, l, 6);
			mn = Combi.ij2k(m, n, 6);
			ost << "P_{" << i + 1 << "}P_{" << j + 1 << "},P_{"
					<< k + 1 << "}P_{" << l + 1 << "},P_{"
					<< m + 1 << "}P_{" << n + 1 << "} = ";
			ost << "\\big(";
			Surf->PolynomialDomains->Poly1->print_equation(ost, bisecants + ij * 3);
			ost << "\\big)";
			ost << "\\big(";
			Surf->PolynomialDomains->Poly1->print_equation(ost, bisecants + kl * 3);
			ost << "\\big)";
			ost << "\\big(";
			Surf->PolynomialDomains->Poly1->print_equation(ost, bisecants + mn * 3);
			ost << "\\big)";
			//multiply_linear_times_linear_times_linear(
			//bisecants + ij * 3, bisecants + kl * 3,
			//bisecants + mn * 3, ten_coeff, 0 /* verbose_level */);
		}
		ost << " = ";
		Surf->PolynomialDomains->Poly3->print_equation(ost, Web_of_cubic_curves + h * 10);
		ost << "$\\\\";
	}

	ost << "The coeffcients are:" << endl;
	for (i = 0; i < 15; i++) {
		labels[i] = i;
	}
	ost << "$$" << endl;
	L.int_matrix_print_with_labels_and_partition(ost,
			Web_of_cubic_curves, 15, 10,
		labels, labels,
		row_fst, row_len, 1,
		col_fst, col_len, 1,
		Web_of_cubic_curves_entry_print, (void *) this,
		TRUE /* f_tex */);
	ost << "$$" << endl;

	for (i = 0; i < 15; i++) {
		labels[i] = 15 + i;
	}
	ost << "$$" << endl;
	L.int_matrix_print_with_labels_and_partition(ost,
			Web_of_cubic_curves, 15, 10,
		labels, labels,
		row_fst, row_len, 1,
		col_fst, col_len, 1,
		Web_of_cubic_curves_entry_print, (void *) this,
		TRUE /* f_tex */);
	ost << "$$" << endl;

	for (i = 0; i < 15; i++) {
		labels[i] = 30 + i;
	}
	ost << "$$" << endl;
	L.int_matrix_print_with_labels_and_partition(ost,
			Web_of_cubic_curves, 15, 10,
		labels, labels,
		row_fst, row_len, 1,
		col_fst, col_len, 1,
		Web_of_cubic_curves_entry_print, (void *) this,
		TRUE /* f_tex */);
	ost << "$$" << endl;

	FREE_int(bisecants);
	FREE_int(conics);

}

static void Web_of_cubic_curves_entry_print(int *p,
	int m, int n, int i, int j, int val,
	std::string &output, void *data)
{
	web_of_cubic_curves *Web = (web_of_cubic_curves *) data;

	if (i == -1) {
		Web->Surf->PolynomialDomains->Poly3->print_monomial_latex(output, j);
	}
	else if (j == -1) {
		char str[1000];

		snprintf(str, sizeof(str), "\\pi_{%d}", i);
		output.append(str);
		output.append(" = \\pi_{");
		output.append(Web->Surf->Schlaefli->Eckard_point_label[i]);
		output.append("}");
		//snprintf(output, 1000, "\\pi_{%d} = \\pi_{%s}", i,
		//		Web->Surf->Schlaefli->Eckard_point_label[i].c_str());
	}
	else {
		char str[1000];

		snprintf(str, sizeof(str), "%d", i);
		output.append(str);
		//snprintf(output, 1000, "%d", val);
	}
}




}}}

