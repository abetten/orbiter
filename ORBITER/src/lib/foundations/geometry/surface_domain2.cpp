/*
 * surface_domain2.cpp
 *
 *  Created on: Nov 3, 2019
 *      Author: anton
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {



void surface_domain::multiply_conic_times_linear(int *six_coeff,
	int *three_coeff, int *ten_coeff,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx;
	int M[3];

	if (f_v) {
		cout << "surface_domain::multiply_conic_times_linear" << endl;
		}


	int_vec_zero(ten_coeff, 10);
	for (i = 0; i < 6; i++) {
		a = six_coeff[i];
		for (j = 0; j < 3; j++) {
			b = three_coeff[j];
			c = F->mult(a, b);
			int_vec_add(Poly2->Monomials + i * 3,
				Poly1->Monomials + j * 3, M, 3);
			idx = Poly3->index_of_monomial(M);
			if (idx >= 10) {
				cout << "surface_domain::multiply_conic_times_linear "
						"idx >= 10" << endl;
				exit(1);
				}
			ten_coeff[idx] = F->add(ten_coeff[idx], c);
			}
		}


	if (f_v) {
		cout << "surface_domain::multiply_conic_times_linear done" << endl;
		}
}

void surface_domain::multiply_linear_times_linear_times_linear(
	int *three_coeff1, int *three_coeff2, int *three_coeff3,
	int *ten_coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a, b, c, d, idx;
	int M[3];

	if (f_v) {
		cout << "surface_domain::multiply_linear_times_linear_"
				"times_linear" << endl;
		}

	int_vec_zero(ten_coeff, 10);
	for (i = 0; i < 3; i++) {
		a = three_coeff1[i];
		for (j = 0; j < 3; j++) {
			b = three_coeff2[j];
			for (k = 0; k < 3; k++) {
				c = three_coeff3[k];
				d = F->mult3(a, b, c);
				int_vec_add3(Poly1->Monomials + i * 3,
					Poly1->Monomials + j * 3,
					Poly1->Monomials + k * 3,
					M, 3);
				idx = Poly3->index_of_monomial(M);
				if (idx >= 10) {
					cout << "surface::multiply_linear_times_"
							"linear_times_linear idx >= 10" << endl;
					exit(1);
					}
				ten_coeff[idx] = F->add(ten_coeff[idx], d);
				}
			}
		}


	if (f_v) {
		cout << "surface_domain::multiply_linear_times_linear_"
				"times_linear done" << endl;
		}
}

void surface_domain::multiply_linear_times_linear_times_linear_in_space(
	int *four_coeff1, int *four_coeff2, int *four_coeff3,
	int *twenty_coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a, b, c, d, idx;
	int M[4];

	if (f_v) {
		cout << "surface_domain::multiply_linear_times_linear_"
				"times_linear_in_space" << endl;
		}

	int_vec_zero(twenty_coeff, 20);
	for (i = 0; i < 4; i++) {
		a = four_coeff1[i];
		for (j = 0; j < 4; j++) {
			b = four_coeff2[j];
			for (k = 0; k < 4; k++) {
				c = four_coeff3[k];
				d = F->mult3(a, b, c);
				int_vec_add3(Poly1_4->Monomials + i * 4,
					Poly1_4->Monomials + j * 4,
					Poly1_4->Monomials + k * 4,
					M, 4);
				idx = index_of_monomial(M);
				if (idx >= 20) {
					cout << "surface_domain::multiply_linear_times_linear_"
							"times_linear_in_space idx >= 20" << endl;
					exit(1);
					}
				twenty_coeff[idx] = F->add(twenty_coeff[idx], d);
				}
			}
		}


	if (f_v) {
		cout << "surface_domain::multiply_linear_times_linear_"
				"times_linear_in_space done" << endl;
		}
}

void surface_domain::multiply_Poly2_3_times_Poly2_3(
	int *input1, int *input2,
	int *result, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx;
	int M[3];

	if (f_v) {
		cout << "surface_domain::multiply_Poly2_3_times_Poly2_3" << endl;
		}

	int_vec_zero(result, Poly4_x123->nb_monomials);
	for (i = 0; i < Poly2->nb_monomials; i++) {
		a = input1[i];
		for (j = 0; j < Poly2->nb_monomials; j++) {
			b = input2[j];
			c = F->mult(a, b);
			int_vec_add(Poly2->Monomials + i * 3,
				Poly2->Monomials + j * 3,
				M, 3);
			idx = Poly4_x123->index_of_monomial(M);
			result[idx] = F->add(result[idx], c);
			}
		}


	if (f_v) {
		cout << "surface_domain::multiply_Poly2_3_times_Poly2_3 done" << endl;
		}
}

void surface_domain::multiply_Poly1_3_times_Poly3_3(int *input1, int *input2,
	int *result, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx;
	int M[3];

	if (f_v) {
		cout << "surface_domain::multiply_Poly1_3_times_Poly3_3" << endl;
		}

	int_vec_zero(result, Poly4_x123->nb_monomials);
	for (i = 0; i < Poly1->nb_monomials; i++) {
		a = input1[i];
		for (j = 0; j < Poly3->nb_monomials; j++) {
			b = input2[j];
			c = F->mult(a, b);
			int_vec_add(Poly1->Monomials + i * 3,
				Poly3->Monomials + j * 3, M, 3);
			idx = Poly4_x123->index_of_monomial(M);
			result[idx] = F->add(result[idx], c);
			}
		}

	if (f_v) {
		cout << "surface_domain::multiply_Poly1_3_times_Poly3_3 done" << endl;
		}
}

void surface_domain::web_of_cubic_curves(long int *arc6, int *&curves,
	int verbose_level)
// curves[45 * 10]
{
	int f_v = (verbose_level >= 1);
	int *bisecants;
	int *conics;
	int ten_coeff[10];
	int a, rk, i, j, k, l, m, n;
	int ij, kl, mn;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "surface::web_of_cubic_curves" << endl;
		}
	P2->compute_bisecants_and_conics(arc6,
		bisecants, conics, verbose_level);

	curves = NEW_int(45 * 10);


	a = 0;

	// the first 30 curves:
	for (rk = 0; rk < 30; rk++, a++) {
		Combi.ordered_pair_unrank(rk, i, j, 6);
		ij = Combi.ij2k(i, j, 6);
		multiply_conic_times_linear(conics + j * 6,
			bisecants + ij * 3,
			ten_coeff,
			0 /* verbose_level */);
		int_vec_copy(ten_coeff, curves + a * 10, 10);
		}

	// the next 15 curves:
	for (rk = 0; rk < 15; rk++, a++) {
		Combi.unordered_triple_pair_unrank(rk, i, j, k, l, m, n);
		ij = Combi.ij2k(i, j, 6);
		kl = Combi.ij2k(k, l, 6);
		mn = Combi.ij2k(m, n, 6);
		multiply_linear_times_linear_times_linear(
			bisecants + ij * 3,
			bisecants + kl * 3,
			bisecants + mn * 3,
			ten_coeff,
			0 /* verbose_level */);
		int_vec_copy(ten_coeff, curves + a * 10, 10);
		}

	if (a != 45) {
		cout << "surface_domain::web_of_cubic_curves a != 45" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "The web of cubic curves is:" << endl;
		int_matrix_print(curves, 45, 10);
		}

	FREE_int(bisecants);
	FREE_int(conics);

	if (f_v) {
		cout << "surface_domain::web_of_cubic_curves done" << endl;
		}
}

void surface_domain::web_of_cubic_curves_rank_of_foursubsets(
	int *Web_of_cubic_curves,
	int *&rk, int &N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int set[4], i, j, a;
	int B[4 * 10];
	combinatorics_domain Combi;

	if (f_v) {
		cout << "surface_domain::web_of_cubic_curves_rank_of_foursubsets" << endl;
		}
	if (f_v) {
		cout << "web of cubic curves:" << endl;
		int_matrix_print(Web_of_cubic_curves, 45, 10);
		}
	N = Combi.int_n_choose_k(45, 4);
	rk = NEW_int(N);
	for (i = 0; i < N; i++) {
		Combi.unrank_k_subset(i, set, 45, 4);
		if (f_v) {
			cout << "subset " << i << " / " << N << " is ";
			int_vec_print(cout, set, 4);
			cout << endl;
			}
		for (j = 0; j < 4; j++) {
			a = set[j];
			int_vec_copy(Web_of_cubic_curves + a * 10,
				B + j * 10, 10);
			}
		rk[i] = F->rank_of_rectangular_matrix(B,
			4, 10, 0 /* verbose_level */);
		}
	if (f_v) {
		cout << "surface_domain::web_of_cubic_curves_rank_of_foursubsets done" << endl;
		}
}

void
surface_domain::create_web_of_cubic_curves_and_equations_based_on_four_tritangent_planes(
		long int *arc6, int *base_curves4,
	int *&Web_of_cubic_curves, int *&The_plane_equations,
	int verbose_level)
// Web_of_cubic_curves[45 * 10]
{
	int f_v = (verbose_level >= 1);
	int h, rk, idx;
	int *base_curves;
	int *curves;
	int *curves_t;
	sorting Sorting;

	if (f_v) {
		cout << "surface_domain::create_web_of_cubic_curves_and_equations_based_"
				"on_four_tritangent_planes" << endl;
		}

	web_of_cubic_curves(arc6, Web_of_cubic_curves, verbose_level);

	base_curves = NEW_int(5 * 10);
	curves = NEW_int(5 * 10);
	curves_t = NEW_int(10 * 5);



	for (h = 0; h < 4; h++) {
		int_vec_copy(Web_of_cubic_curves + base_curves4[h] * 10,
			base_curves + h * 10, 10);
		}

	if (f_v) {
		cout << "base_curves:" << endl;
		int_matrix_print(base_curves, 4, 10);
		}

	// find the plane equations:

	The_plane_equations = NEW_int(45 * 4);

	for (h = 0; h < 45; h++) {

		if (f_v) {
			cout << "h=" << h << " / " << 45 << ":" << endl;
			}

		if (Sorting.int_vec_search_linear(base_curves4, 4, h, idx)) {
			int_vec_zero(The_plane_equations + h * 4, 4);
			The_plane_equations[h * 4 + idx] = 1;
			}
		else {
			int_vec_copy(base_curves, curves, 4 * 10);
			int_vec_copy(Web_of_cubic_curves + h * 10,
				curves + 4 * 10, 10);

			if (f_v) {
				cout << "h=" << h << " / " << 45
					<< " the system is:" << endl;
				int_matrix_print(curves, 5, 10);
				}

			F->transpose_matrix(curves, curves_t, 5, 10);

			if (f_v) {
				cout << "after transpose:" << endl;
				int_matrix_print(curves_t, 10, 5);
				}

			rk = F->RREF_and_kernel(5, 10, curves_t,
				0 /* verbose_level */);
			if (rk != 4) {
				cout << "surface::create_surface_and_planes_from_"
						"trihedral_pair_and_arc the rank of the "
						"system is not equal to 4" << endl;
				cout << "rk = " << rk << endl;
				exit(1);
				}
			if (curves_t[4 * 5 + 4] != F->negate(1)) {
				cout << "h=" << h << " / " << 2
					<< " curves_t[4 * 5 + 4] != -1" << endl;
				exit(1);
				}
			int_vec_copy(curves_t + 4 * 5,
				The_plane_equations + h * 4, 4);

			F->PG_element_normalize(
				The_plane_equations + h * 4, 1, 4);

			}
		if (f_v) {
			cout << "h=" << h << " / " << 45
				<< ": the plane equation is ";
			int_vec_print(cout, The_plane_equations + h * 4, 4);
			cout << endl;
			}


		}
	if (f_v) {
		cout << "the plane equations are: " << endl;
		int_matrix_print(The_plane_equations, 45, 4);
		cout << endl;
		}

	FREE_int(base_curves);
	FREE_int(curves);
	FREE_int(curves_t);

	if (f_v) {
		cout << "surface_domain::create_web_of_cubic_curves_and_equations_"
				"based_on_four_tritangent_planes done" << endl;
		}
}

void surface_domain::create_equations_for_pencil_of_surfaces_from_trihedral_pair(
	int *The_six_plane_equations, int *The_surface_equations,
	int verbose_level)
// The_surface_equations[(q + 1) * 20]
{
	int f_v = (verbose_level >= 1);
	int v[2];
	int l;
	int eqn_F[20];
	int eqn_G[20];
	int eqn_F2[20];
	int eqn_G2[20];

	if (f_v) {
		cout << "surface_domain::create_equations_for_pencil_of_surfaces_"
				"from_trihedral_pair" << endl;
		}


	for (l = 0; l < q + 1; l++) {
		F->PG_element_unrank_modified(v, 1, 2, l);

		multiply_linear_times_linear_times_linear_in_space(
			The_six_plane_equations + 0 * 4,
			The_six_plane_equations + 1 * 4,
			The_six_plane_equations + 2 * 4,
			eqn_F, FALSE /* verbose_level */);
		multiply_linear_times_linear_times_linear_in_space(
			The_six_plane_equations + 3 * 4,
			The_six_plane_equations + 4 * 4,
			The_six_plane_equations + 5 * 4,
			eqn_G, FALSE /* verbose_level */);

		int_vec_copy(eqn_F, eqn_F2, 20);
		F->scalar_multiply_vector_in_place(v[0], eqn_F2, 20);
		int_vec_copy(eqn_G, eqn_G2, 20);
		F->scalar_multiply_vector_in_place(v[1], eqn_G2, 20);
		F->add_vector(eqn_F2, eqn_G2,
			The_surface_equations + l * 20, 20);
		F->PG_element_normalize(
			The_surface_equations + l * 20, 1, 20);
		}

	if (f_v) {
		cout << "surface_domain::create_equations_for_pencil_of_surfaces_"
				"from_trihedral_pair done" << endl;
		}
}

void surface_domain::create_lambda_from_trihedral_pair_and_arc(
	long int *arc6,
	int *Web_of_cubic_curves,
	int *The_plane_equations, int t_idx,
	int &lambda, int &lambda_rk,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int row_col_Eckardt_points[6];
	int six_curves[6 * 10];
	int pt, f_point_was_found;
	int v[3];
	int w[2];
	int evals[6];
	int evals_for_point[6];
	int pt_on_surface[4];
	int a, b, ma, bv;

	if (f_v) {
		cout << "surface_domain::create_lambda_from_trihedral_pair_and_arc "
				"t_idx=" << t_idx << endl;
		}

	if (f_v) {
		cout << "Trihedral pair T_{" << Trihedral_pair_labels[t_idx] << "}"
			<< endl;
		}

	int_vec_copy(Trihedral_to_Eckardt + t_idx * 6,
		row_col_Eckardt_points, 6);

	if (f_v) {
		cout << "row_col_Eckardt_points = ";
		int_vec_print(cout, row_col_Eckardt_points, 6);
		cout << endl;
		}



	extract_six_curves_from_web(Web_of_cubic_curves,
		row_col_Eckardt_points, six_curves, verbose_level);

	if (f_v) {
		cout << "The six curves are:" << endl;
		int_matrix_print(six_curves, 6, 10);
		}



	if (f_v) {
		cout << "surface_domain::create_lambda_from_trihedral_pair_and_arc "
				"before find_point_not_on_six_curves" << endl;
		}
	find_point_not_on_six_curves(arc6, six_curves,
		pt, f_point_was_found, verbose_level);
	if (!f_point_was_found) {
		lambda = 1;
		}
	else {
		if (f_v) {
			cout << "surface_domain::create_lambda_from_trihedral_pair_and_arc "
					"after find_point_not_on_six_curves" << endl;
			cout << "pt=" << pt << endl;
			}

		Poly3->unrank_point(v, pt);
		for (i = 0; i < 6; i++) {
			evals[i] = Poly3->evaluate_at_a_point(
				six_curves + i * 10, v);
			}

		if (f_v) {
			cout << "The point pt=" << pt << " = ";
			int_vec_print(cout, v, 3);
			cout << " is nonzero on all plane sections of "
					"the trihedral pair. The values are ";
			int_vec_print(cout, evals, 6);
			cout << endl;
			}

		if (f_v) {
			cout << "solving for lambda:" << endl;
			}
		a = F->mult3(evals[0], evals[1], evals[2]);
		b = F->mult3(evals[3], evals[4], evals[5]);
		ma = F->negate(a);
		bv = F->inverse(b);
		lambda = F->mult(ma, bv);

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
				Poly1_4->evaluate_at_a_point(
				The_plane_equations +
					row_col_Eckardt_points[i] * 4,
				pt_on_surface);
			}
		a = F->mult3(evals_for_point[0],
			evals_for_point[1],
			evals_for_point[2]);
		b = F->mult3(evals_for_point[3],
			evals_for_point[4],
			evals_for_point[5]);
		lambda = F->mult(F->negate(a), F->inverse(b));
		if (f_v) {
			cout << "lambda = " << lambda << endl;
			}
		}
	w[0] = 1;
	w[1] = lambda;
	F->PG_element_rank_modified(w, 1, 2, lambda_rk);

	if (f_v) {
		cout << "surface_domain::create_lambda_from_trihedral_"
				"pair_and_arc done" << endl;
		}
}


void surface_domain::create_surface_equation_from_trihedral_pair(long int *arc6,
	int *Web_of_cubic_curves,
	int *The_plane_equations, int t_idx, int *surface_equation,
	int &lambda,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *The_surface_equations;
	int row_col_Eckardt_points[6];
	int The_six_plane_equations[6 * 4];
	int lambda_rk;

	if (f_v) {
		cout << "surface_domain::create_surface_equation_from_"
				"trihedral_pair t_idx=" << t_idx << endl;
		}


	int_vec_copy(Trihedral_to_Eckardt + t_idx * 6, row_col_Eckardt_points, 6);

	int_vec_copy(The_plane_equations + row_col_Eckardt_points[0] * 4,
			The_six_plane_equations, 4);
	int_vec_copy(The_plane_equations + row_col_Eckardt_points[1] * 4,
			The_six_plane_equations + 4, 4);
	int_vec_copy(The_plane_equations + row_col_Eckardt_points[2] * 4,
			The_six_plane_equations + 8, 4);
	int_vec_copy(The_plane_equations + row_col_Eckardt_points[3] * 4,
			The_six_plane_equations + 12, 4);
	int_vec_copy(The_plane_equations + row_col_Eckardt_points[4] * 4,
			The_six_plane_equations + 16, 4);
	int_vec_copy(The_plane_equations + row_col_Eckardt_points[5] * 4,
			The_six_plane_equations + 20, 4);


	The_surface_equations = NEW_int((q + 1) * 20);

	create_equations_for_pencil_of_surfaces_from_trihedral_pair(
		The_six_plane_equations, The_surface_equations,
		verbose_level - 2);

	create_lambda_from_trihedral_pair_and_arc(arc6,
		Web_of_cubic_curves,
		The_plane_equations, t_idx, lambda, lambda_rk,
		verbose_level - 2);

	int_vec_copy(The_surface_equations + lambda_rk * 20,
		surface_equation, 20);

	FREE_int(The_surface_equations);

	if (f_v) {
		cout << "surface_domain::create_surface_equation_from_"
				"trihedral_pair done" << endl;
		}
}

void surface_domain::extract_six_curves_from_web(
	int *Web_of_cubic_curves,
	int *row_col_Eckardt_points, int *six_curves,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "surface_domain::extract_six_curves_from_web" << endl;
		}
	for (i = 0; i < 6; i++) {
		int_vec_copy(Web_of_cubic_curves + row_col_Eckardt_points[i] * 10,
			six_curves + i * 10, 10);
		}

	if (f_v) {
		cout << "The six curves are:" << endl;
		int_matrix_print(six_curves, 6, 10);
		}
	if (f_v) {
		cout << "surface_domain::extract_six_curves_from_web done" << endl;
		}
}

void surface_domain::find_point_not_on_six_curves(long int *arc6,
	int *six_curves,
	int &pt, int &f_point_was_found,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int v[3];
	int i;
	int idx, a;
	sorting Sorting;

	if (f_v) {
		cout << "surface_domain::find_point_not_on_six_curves" << endl;
		cout << "surface_domain::find_point_not_on_six_curves "
			"P2->N_points="
			<< P2->N_points << endl;
		}
	pt = -1;
	for (pt = 0; pt < P2->N_points; pt++) {
		if (Sorting.lint_vec_search_linear(arc6, 6, pt, idx)) {
			continue;
			}
		Poly3->unrank_point(v, pt);
		for (i = 0; i < 6; i++) {
			a = Poly3->evaluate_at_a_point(six_curves + i * 10, v);
			if (a == 0) {
				break;
				}
			}
		if (i == 6) {
			break;
			}
		}
	if (pt == P2->N_points) {
		cout << "could not find a point which is not on "
				"any of the curve" << endl;
		f_point_was_found = FALSE;
		}
	else {
		f_point_was_found = TRUE;
		}
	if (f_v) {
		cout << "surface_domain::find_point_not_on_six_curves "
				"done" << endl;
		}
}

int surface_domain::plane_from_three_lines(long int *three_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[6 * 4];
	long int rk;

	if (f_v) {
		cout << "surface_domain::plane_from_three_lines" << endl;
		}
	unrank_lines(Basis, three_lines, 3);
	rk = F->Gauss_easy(Basis, 6, 4);
	if (rk != 3) {
		cout << "surface_domain::plane_from_three_lines rk != 3" << endl;
		exit(1);
		}
	rk = rank_plane(Basis);

	if (f_v) {
		cout << "surface_domain::plane_from_three_lines done" << endl;
		}
	return rk;
}

void surface_domain::Trihedral_pairs_to_planes(long int *Lines, long int *Planes,
	int verbose_level)
// Planes[nb_trihedral_pairs * 6]
{
	int f_v = (verbose_level >= 1);
	int t, i, j, rk;
	long int tritangent_plane[3];
	long int three_lines[3];
	latex_interface L;

	if (f_v) {
		cout << "surface_domain::Trihedral_pairs_to_planes" << endl;
		}
	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				tritangent_plane[j] =
					Trihedral_pairs[t * 9 + i * 3 + j];
				}
			for (j = 0; j < 3; j++) {
				three_lines[j] =
					Lines[tritangent_plane[j]];
				}
			rk = plane_from_three_lines(three_lines,
				0 /* verbose_level */);
			Planes[t * 6 + i] = rk;
			}
		for (j = 0; j < 3; j++) {
			for (i = 0; i < 3; i++) {
				tritangent_plane[i] =
					Trihedral_pairs[t * 9 + i * 3 + j];
				}
			for (i = 0; i < 3; i++) {
				three_lines[i] =
					Lines[tritangent_plane[i]];
				}
			rk = plane_from_three_lines(three_lines,
				0 /* verbose_level */);
			Planes[t * 6 + 3 + j] = rk;
			}
		}
	if (f_v) {
		cout << "Trihedral_pairs_to_planes:" << endl;
		L.print_lint_matrix_with_standard_labels(cout,
			Planes, nb_trihedral_pairs, 6, FALSE /* f_tex */);
		}
	if (f_v) {
		cout << "surface_domain::Trihedral_pairs_to_planes done" << endl;
		}
}

void surface_domain::create_surface_family_S(int a,
	long int *Lines27,
	int *equation20, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_family_S" << endl;
		}

	int nb_E = 0;
	int b = 1;
	int alpha, beta;

	if (f_v) {
		cout << "surface_domain::create_surface_family_S "
				"creating surface for a=" << a << ":" << endl;
		}

	create_surface_ab(a, b,
		equation20,
		Lines27,
		alpha, beta, nb_E,
		0 /* verbose_level */);

	if (f_v) {
		cout << "surface_domain::create_surface_family_S "
				"The double six is:" << endl;
		lint_matrix_print(Lines27, 2, 6);
		cout << "The lines are : ";
		lint_vec_print(cout, Lines27, 27);
		cout << endl;
		}

	if (f_v) {
		cout << "surface_domain::create_surface_family_S "
				"done" << endl;
		}
}

void surface_domain::compute_tritangent_planes(long int *Lines,
	int *&Tritangent_planes, int &nb_tritangent_planes,
	int *&Unitangent_planes, int &nb_unitangent_planes,
	int *&Lines_in_tritangent_plane,
	int *&Line_in_unitangent_plane,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Inc_lines_planes;
	int *The_plane_type;
	int nb_planes;
	int i, j, h, c;

	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes" << endl;
		}
	if (f_v) {
		cout << "Lines=" << endl;
		lint_vec_print(cout, Lines, 27);
		cout << endl;
		}
	P->line_plane_incidence_matrix_restricted(Lines, 27,
		Inc_lines_planes, nb_planes, 0 /* verbose_level */);

	The_plane_type = NEW_int(nb_planes);
	int_vec_zero(The_plane_type, nb_planes);

	for (j = 0; j < nb_planes; j++) {
		for (i = 0; i < 27; i++) {
			if (Inc_lines_planes[i * nb_planes + j]) {
				The_plane_type[j]++;
				}
			}
		}
	classify Plane_type;

	Plane_type.init(The_plane_type, nb_planes, FALSE, 0);
	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes The plane type is: ";
		Plane_type.print_naked(TRUE);
		cout << endl;
		}


	Plane_type.get_class_by_value(Tritangent_planes,
		nb_tritangent_planes, 3 /* value */, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes "
				"The tritangent planes are: ";
		int_vec_print(cout, Tritangent_planes, nb_tritangent_planes);
		cout << endl;
		}
	Lines_in_tritangent_plane = NEW_int(nb_tritangent_planes * 3);
	for (h = 0; h < nb_tritangent_planes; h++) {
		j = Tritangent_planes[h];
		c = 0;
		for (i = 0; i < 27; i++) {
			if (Inc_lines_planes[i * nb_planes + j]) {
				Lines_in_tritangent_plane[h * 3 + c++] = i;
				}
			}
		if (c != 3) {
			cout << "surface_domain::compute_tritangent_planes c != 3" << endl;
			exit(1);
			}
		}


	Plane_type.get_class_by_value(Unitangent_planes,
		nb_unitangent_planes, 1 /* value */, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes "
				"The unitangent planes are: ";
		int_vec_print(cout, Unitangent_planes, nb_unitangent_planes);
		cout << endl;
		}
	Line_in_unitangent_plane = NEW_int(nb_unitangent_planes);
	for (h = 0; h < nb_unitangent_planes; h++) {
		j = Unitangent_planes[h];
		c = 0;
		for (i = 0; i < 27; i++) {
			if (Inc_lines_planes[i * nb_planes + j]) {
				Line_in_unitangent_plane[h * 1 + c++] = i;
				}
			}
		if (c != 1) {
			cout << "surface_domain::compute_tritangent_planes c != 1" << endl;
			exit(1);
			}
		}

	FREE_int(Inc_lines_planes);
	FREE_int(The_plane_type);

	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes done" << endl;
		}
}

#if 0
void surface_domain::compute_external_lines_on_three_tritangent_planes(
	long int *Lines, long int *&External_lines, int &nb_external_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	latex_interface L;

	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_"
				"three_tritangent_planes" << endl;
		}

	int *Tritangent_planes;
	int nb_tritangent_planes;
	int *Lines_in_tritangent_plane; // [nb_tritangent_planes * 3]

	int *Unitangent_planes;
	int nb_unitangent_planes;
	int *Line_in_unitangent_plane; // [nb_unitangent_planes]

	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_"
				"three_tritangent_planes computing "
				"tritangent planes:" << endl;
		}
	compute_tritangent_planes(Lines,
		Tritangent_planes, nb_tritangent_planes,
		Unitangent_planes, nb_unitangent_planes,
		Lines_in_tritangent_plane,
		Line_in_unitangent_plane,
		verbose_level);

	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_"
				"three_tritangent_planes Lines_in_"
				"tritangent_plane: " << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			Lines_in_tritangent_plane, nb_tritangent_planes,
			3, FALSE);
		}

	int *Intersection_matrix;
		// [nb_tritangent_planes * nb_tritangent_planes]
	int *Plane_intersections;
	int *Plane_intersections_general;
	int rk, idx;
	sorting Sorting;



	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_"
				"three_tritangent_planes Computing intersection "
				"matrix of tritangent planes:" << endl;
		}

	P->plane_intersection_matrix_in_three_space(Tritangent_planes,
		nb_tritangent_planes, Intersection_matrix,
		0 /* verbose_level */);

	Plane_intersections =
			NEW_int(nb_tritangent_planes * nb_tritangent_planes);
	Plane_intersections_general =
			NEW_int(nb_tritangent_planes * nb_tritangent_planes);
	for (i = 0; i < nb_tritangent_planes; i++) {
		for (j = 0; j < nb_tritangent_planes; j++) {
			Plane_intersections[i * nb_tritangent_planes + j] = -1;
			Plane_intersections_general[i * nb_tritangent_planes + j] = -1;
			if (j != i) {
				rk = Intersection_matrix[i * nb_tritangent_planes + j];
				if (Sorting.lint_vec_search_linear(
					Lines, 27, rk, idx)) {
					Plane_intersections[i * nb_tritangent_planes + j] = idx;
					}
				else {
					Plane_intersections_general[
						i * nb_tritangent_planes + j] = rk;
					}
				}
			}
		}

	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_three_"
				"tritangent_planes The tritangent planes intersecting "
				"in surface lines:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			Plane_intersections, nb_tritangent_planes,
			nb_tritangent_planes, FALSE);
		}


	classify Plane_intersection_type;

	Plane_intersection_type.init(Plane_intersections,
		nb_tritangent_planes * nb_tritangent_planes, TRUE, 0);
	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_three_"
				"tritangent_planes The surface lines in terms "
				"of plane intersections are: ";
		Plane_intersection_type.print_naked(TRUE);
		cout << endl;
		}


	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_three_"
				"tritangent_planes The tritangent planes "
				"intersecting in general lines:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
				Plane_intersections_general, nb_tritangent_planes,
				nb_tritangent_planes, FALSE);
		}

	classify Plane_intersection_type2;

	Plane_intersection_type2.init(Plane_intersections_general,
		nb_tritangent_planes * nb_tritangent_planes, TRUE, 0);
	if (f_v) {
		cout << "The other lines in terms of plane intersections are: ";
		Plane_intersection_type2.print_naked(TRUE);
		cout << endl;
		}


	Plane_intersection_type2.get_data_by_multiplicity(
		External_lines, nb_external_lines, 6, 0 /* verbose_level */);

	Sorting.int_vec_heapsort(External_lines, nb_external_lines);

	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_three_"
				"tritangent_planes The non-surface lines which are on "
				"three tritangent planes are:" << endl;
		int_vec_print(cout, External_lines, nb_external_lines);
		cout << endl;
		cout << "surface_domain::compute_external_lines_on_three_"
				"tritangent_planes these lines are:" << endl;
		P->Grass_lines->print_set(External_lines, nb_external_lines);
		}

	FREE_int(Tritangent_planes);
	FREE_int(Lines_in_tritangent_plane);
	FREE_int(Unitangent_planes);
	FREE_int(Line_in_unitangent_plane);
	FREE_int(Intersection_matrix);
	FREE_int(Plane_intersections);
	FREE_int(Plane_intersections_general);

	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_three_"
				"tritangent_planes done" << endl;
		}
}
#endif

void surface_domain::init_double_sixes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, ij, u, v, l, m, n, h, a, b, c;
	int set[6];
	int size_complement;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "surface_domain::init_double_sixes" << endl;
		}
	Double_six = NEW_int(36 * 12);
	h = 0;
	// first type: D : a_1,..., a_6; b_1, ..., b_6
	for (i = 0; i < 12; i++) {
		Double_six[h * 12 + i] = i;
		}
	h++;

	// second type:
	// D_{ij} :
	// a_1, b_1, c_23, c_24, c_25, c_26;
	// a_2, b_2, c_13, c_14, c_15, c_16
	for (ij = 0; ij < 15; ij++, h++) {
		//cout << "second type " << ij << " / " << 15 << endl;
		Combi.k2ij(ij, i, j, 6);
		set[0] = i;
		set[1] = j;
		Combi.set_complement(set, 2 /* subset_size */, set + 2,
			size_complement, 6 /* universal_set_size */);
		//cout << "set : ";
		//int_vec_print(cout, set, 6);
		//cout << endl;
		Double_six[h * 12 + 0] = line_ai(i);
		Double_six[h * 12 + 1] = line_bi(i);
		for (u = 0; u < 4; u++) {
			Double_six[h * 12 + 2 + u] = line_cij(j, set[2 + u]);
			}
		Double_six[h * 12 + 6] = line_ai(j);
		Double_six[h * 12 + 7] = line_bi(j);
		for (u = 0; u < 4; u++) {
			Double_six[h * 12 + 8 + u] = line_cij(i, set[2 + u]);
			}
		}

	// third type: D_{ijk} :
	// a_1, a_2, a_3, c_56, c_46, c_45;
	// c_23, c_13, c_12, b_4, b_5, b_6
	for (v = 0; v < 20; v++, h++) {
		//cout << "third type " << v << " / " << 20 << endl;
		Combi.unrank_k_subset(v, set, 6, 3);
		Combi.set_complement(set, 3 /* subset_size */, set + 3,
			size_complement, 6 /* universal_set_size */);
		i = set[0];
		j = set[1];
		k = set[2];
		l = set[3];
		m = set[4];
		n = set[5];
		Double_six[h * 12 + 0] = line_ai(i);
		Double_six[h * 12 + 1] = line_ai(j);
		Double_six[h * 12 + 2] = line_ai(k);
		Double_six[h * 12 + 3] = line_cij(m, n);
		Double_six[h * 12 + 4] = line_cij(l, n);
		Double_six[h * 12 + 5] = line_cij(l, m);
		Double_six[h * 12 + 6] = line_cij(j, k);
		Double_six[h * 12 + 7] = line_cij(i, k);
		Double_six[h * 12 + 8] = line_cij(i, j);
		Double_six[h * 12 + 9] = line_bi(l);
		Double_six[h * 12 + 10] = line_bi(m);
		Double_six[h * 12 + 11] = line_bi(n);
		}

	if (h != 36) {
		cout << "surface_domain::init_double_sixes h != 36" << endl;
		exit(1);
		}

	Double_six_label_tex = NEW_pchar(36);
	char str[1000];

	for (i = 0; i < 36; i++) {
		if (i < 1) {
			sprintf(str, "D");
			}
		else if (i < 1 + 15) {
			ij = i - 1;
			Combi.k2ij(ij, a, b, 6);
			set[0] = a;
			set[1] = b;
			Combi.set_complement(set, 2 /* subset_size */, set + 2,
				size_complement, 6 /* universal_set_size */);
			sprintf(str, "D_{%d%d}", a + 1, b + 1);
			}
		else {
			v = i - 16;
			Combi.unrank_k_subset(v, set, 6, 3);
			Combi.set_complement(set, 3 /* subset_size */, set + 3,
				size_complement, 6 /* universal_set_size */);
			a = set[0];
			b = set[1];
			c = set[2];
			sprintf(str, "D_{%d%d%d}", a + 1, b + 1, c + 1);
			}
		if (f_v) {
			cout << "creating label " << str
				<< " for Double six " << i << endl;
			}
		l = strlen(str);
		Double_six_label_tex[i] = NEW_char(l + 1);
		strcpy(Double_six_label_tex[i], str);
		}

	if (f_v) {
		cout << "surface_domain::init_double_sixes done" << endl;
		}
}

void surface_domain::create_half_double_sixes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, ij, v, l;
	int set[6];
	int size_complement;
	combinatorics_domain Combi;
	sorting Sorting;

	if (f_v) {
		cout << "surface_domain::create_half_double_sixes" << endl;
		}
	Half_double_sixes = NEW_int(72 * 6);
	Half_double_six_to_double_six = NEW_int(72);
	Half_double_six_to_double_six_row = NEW_int(72);

	int_vec_copy(Double_six, Half_double_sixes, 36 * 12);
	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			Sorting.int_vec_heapsort(
				Half_double_sixes + (2 * i + j) * 6, 6);
			Half_double_six_to_double_six[2 * i + j] = i;
			Half_double_six_to_double_six_row[2 * i + j] = j;
			}
		}
	Half_double_six_label_tex = NEW_pchar(72);
	char str[1000];

	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			if (i < 1) {
				sprintf(str, "D");
				}
			else if (i < 1 + 15) {
				ij = i - 1;
				Combi.k2ij(ij, a, b, 6);
				set[0] = a;
				set[1] = b;
				Combi.set_complement(set, 2 /* subset_size */,
					set + 2, size_complement,
					6 /* universal_set_size */);
				sprintf(str, "D_{%d%d}", a + 1, b + 1);
				}
			else {
				v = i - 16;
				Combi.unrank_k_subset(v, set, 6, 3);
				Combi.set_complement(set, 3 /* subset_size */,
					set + 3, size_complement,
					6 /* universal_set_size */);
				a = set[0];
				b = set[1];
				c = set[2];
				sprintf(str, "D_{%d%d%d}",
					a + 1, b + 1, c + 1);
				}
			if (j == 0) {
				sprintf(str + strlen(str), "^\\top");
				}
			else {
				sprintf(str + strlen(str), "^\\bot");
				}
			if (f_v) {
				cout << "creating label " << str
					<< " for half double six "
					<< 2 * i + j << endl;
				}
			l = strlen(str);
			Half_double_six_label_tex[2 * i + j] = NEW_char(l + 1);
			strcpy(Half_double_six_label_tex[2 * i + j], str);
			}
		}

	if (f_v) {
		cout << "surface_domain::create_half_double_sixes done" << endl;
		}
}

int surface_domain::find_half_double_six(int *half_double_six)
{
	int i;
	sorting Sorting;

	Sorting.int_vec_heapsort(half_double_six, 6);
	for (i = 0; i < 72; i++) {
		if (int_vec_compare(half_double_six,
			Half_double_sixes + i * 6, 6) == 0) {
			return i;
			}
		}
	cout << "surface_domain::find_half_double_six did not find "
			"half double six" << endl;
	exit(1);
}

void surface_domain::ijklm2n(int i, int j, int k, int l, int m, int &n)
{
	int v[6];
	int size_complement;
	combinatorics_domain Combi;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	v[3] = l;
	v[4] = m;
	Combi.set_complement_safe(v, 5, v + 5, size_complement, 6);
	if (size_complement != 1) {
		cout << "surface_domain::ijklm2n size_complement != 1" << endl;
		exit(1);
		}
	n = v[5];
}

void surface_domain::ijkl2mn(int i, int j, int k, int l, int &m, int &n)
{
	int v[6];
	int size_complement;
	combinatorics_domain Combi;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	v[3] = l;
	Combi.set_complement_safe(v, 4, v + 4, size_complement, 6);
	if (size_complement != 2) {
		cout << "surface_domain::ijkl2mn size_complement != 2" << endl;
		exit(1);
		}
	m = v[4];
	n = v[5];
}

void surface_domain::ijk2lmn(int i, int j, int k, int &l, int &m, int &n)
{
	int v[6];
	int size_complement;
	combinatorics_domain Combi;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	cout << "surface_domain::ijk2lmn v=";
	int_vec_print(cout, v, 3);
	cout << endl;
	Combi.set_complement_safe(v, 3, v + 3, size_complement, 6);
	if (size_complement != 3) {
		cout << "surface_domain::ijk2lmn size_complement != 3" << endl;
		cout << "size_complement=" << size_complement << endl;
		exit(1);
		}
	l = v[3];
	m = v[4];
	n = v[5];
}

void surface_domain::ij2klmn(int i, int j, int &k, int &l, int &m, int &n)
{
	int v[6];
	int size_complement;
	combinatorics_domain Combi;

	v[0] = i;
	v[1] = j;
	Combi.set_complement_safe(v, 2, v + 2, size_complement, 6);
	if (size_complement != 4) {
		cout << "surface_domain::ij2klmn size_complement != 4" << endl;
		exit(1);
		}
	k = v[2];
	l = v[3];
	m = v[4];
	n = v[5];
}

void surface_domain::get_half_double_six_associated_with_Clebsch_map(
	int line1, int line2, int transversal,
	int hds[6],
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t1, t2, t3;
	int i, j, k, l, m, n;
	int i1, j1;
	int null;

	if (f_v) {
		cout << "surface_domain::get_half_double_six_associated_"
				"with_Clebsch_map" << endl;
		}

	if (line1 > line2) {
		cout << "surface_domain::get_half_double_six_associated_"
				"with_Clebsch_map line1 > line2" << endl;
		exit(1);
		}
	t1 = type_of_line(line1);
	t2 = type_of_line(line2);
	t3 = type_of_line(transversal);

	if (f_v) {
		cout << "t1=" << t1 << " t2=" << t2 << " t3=" << t3 << endl;
		}
	if (t1 == 0 && t2 == 0) { // ai and aj:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (t3 == 1) { // bk
			index_of_line(transversal, k, null);
			//cout << "i=" << i << " j=" << j << " k=" << k <<< endl;
			ijk2lmn(i, j, k, l, m, n);
			// bl, bm, bn, cij, cik, cjk
			hds[0] = line_bi(l);
			hds[1] = line_bi(m);
			hds[2] = line_bi(n);
			hds[3] = line_cij(i, j);
			hds[4] = line_cij(i, k);
			hds[5] = line_cij(j, k);
			}
		else if (t3 == 2) { // cij
			index_of_line(transversal, i1, j1);
				// test whether {i1,j1} =  {i,j}
			if ((i == i1 && j == j1) || (i == j1 && j == i1)) {
				ij2klmn(i, j, k, l, m, n);
				// bi, bj, bk, bl, bm, bn
				hds[0] = line_bi(i);
				hds[1] = line_bi(j);
				hds[2] = line_bi(k);
				hds[3] = line_bi(l);
				hds[4] = line_bi(m);
				hds[5] = line_bi(n);
				}
			else {
				cout << "surface_domain::get_half_doble_six_associated_"
						"with_Clebsch_map not {i,j} = {i1,j1}" << endl;
				exit(1);
				}
			}
		}
	else if (t1 == 1 && t2 == 1) { // bi and bj:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (t3 == 0) { // ak
			index_of_line(transversal, k, null);
			ijk2lmn(i, j, k, l, m, n);
			// al, am, an, cij, cik, cjk
			hds[0] = line_ai(l);
			hds[1] = line_ai(m);
			hds[2] = line_ai(n);
			hds[3] = line_cij(i, j);
			hds[4] = line_cij(i, k);
			hds[5] = line_cij(j, k);
			}
		else if (t3 == 2) { // cij
			index_of_line(transversal, i1, j1);
			if ((i == i1 && j == j1) || (i == j1 && j == i1)) {
				ij2klmn(i, j, k, l, m, n);
				// ai, aj, ak, al, am, an
				hds[0] = line_ai(i);
				hds[1] = line_ai(j);
				hds[2] = line_ai(k);
				hds[3] = line_ai(l);
				hds[4] = line_ai(m);
				hds[5] = line_ai(n);
				}
			else {
				cout << "surface_domain::get_half_doble_six_associated_"
						"with_Clebsch_map not {i,j} = {i1,j1}" << endl;
				exit(1);
				}
			}
		}
	else if (t1 == 0 && t2 == 1) { // ai and bi:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (j != i) {
			cout << "surface_domain::get_half_double_six_associated_"
					"with_Clebsch_map j != i" << endl;
			exit(1);
			}
		if (t3 != 2) {
			cout << "surface_domain::get_half_double_six_associated_"
					"with_Clebsch_map t3 != 2" << endl;
			exit(1);
			}
		index_of_line(transversal, i1, j1);
		if (i1 == i) {
			j = j1;
			}
		else {
			j = i1;
			}
		ij2klmn(i, j, k, l, m, n);
		// cik, cil, cim, cin, aj, bj
		hds[0] = line_cij(i, k);
		hds[1] = line_cij(i, l);
		hds[2] = line_cij(i, m);
		hds[3] = line_cij(i, n);
		hds[4] = line_ai(j);
		hds[5] = line_bi(j);
		}
	else if (t1 == 1 && t2 == 2) { // bi and cjk:
		index_of_line(line1, i, null);
		index_of_line(line2, j, k);
		if (t3 == 2) { // cil
			index_of_line(transversal, i1, j1);
			if (i1 == i) {
				l = j1;
				}
			else if (j1 == i) {
				l = i1;
				}
			else {
				cout << "surface_domain::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
				}
			ijkl2mn(i, j, k, l, m, n);
			// cin, cim, aj, ak, al, cnm
			hds[0] = line_cij(i, n);
			hds[1] = line_cij(i, m);
			hds[2] = line_ai(j);
			hds[3] = line_ai(k);
			hds[4] = line_ai(l);
			hds[5] = line_cij(n, m);
			}
		else if (t3 == 0) { // aj
			index_of_line(transversal, j1, null);
			if (j1 == k) {
				// swap k and j
				int tmp = k;
				k = j;
				j = tmp;
				}
			if (j1 != j) {
				cout << "surface_domain::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
				}
			ijk2lmn(i, j, k, l, m, n);
			// ak, cil, cim, cin, bk, cij
			hds[0] = line_ai(k);
			hds[1] = line_cij(i, l);
			hds[2] = line_cij(i, m);
			hds[3] = line_cij(i, n);
			hds[4] = line_bi(k);
			hds[5] = line_cij(i, j);
			}
		}
	else if (t1 == 0 && t2 == 2) { // ai and cjk:
		index_of_line(line1, i, null);
		index_of_line(line2, j, k);
		if (t3 == 2) { // cil
			index_of_line(transversal, i1, j1);
			if (i1 == i) {
				l = j1;
				}
			else if (j1 == i) {
				l = i1;
				}
			else {
				cout << "surface_domain::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
				}
			ijkl2mn(i, j, k, l, m, n);
			// cin, cim, bj, bk, bl, cnm
			hds[0] = line_cij(i, n);
			hds[1] = line_cij(i, m);
			hds[2] = line_bi(j);
			hds[3] = line_bi(k);
			hds[4] = line_bi(l);
			hds[5] = line_cij(n, m);
			}
		else if (t3 == 1) { // bj
			index_of_line(transversal, j1, null);
			if (j1 == k) {
				// swap k and j
				int tmp = k;
				k = j;
				j = tmp;
				}
			if (j1 != j) {
				cout << "surface_domain::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
				}
			ijk2lmn(i, j, k, l, m, n);
			// bk, cil, cim, cin, ak, cij
			hds[0] = line_bi(k);
			hds[1] = line_cij(i, l);
			hds[2] = line_cij(i, m);
			hds[3] = line_cij(i, n);
			hds[4] = line_ai(k);
			hds[5] = line_cij(i, j);
			}
		}
	else if (t1 == 2 && t2 == 2) { // cij and cik:
		index_of_line(line1, i, j);
		index_of_line(line2, i1, j1);
		if (i == i1) {
			k = j1;
			}
		else if (i == j1) {
			k = i1;
			}
		else if (j == i1) {
			j = i;
			i = i1;
			k = j1;
			}
		else if (j == j1) {
			j = i;
			i = j1;
			k = i1;
			}
		else {
			cout << "surface_domain::get_half_double_six_associated_"
					"with_Clebsch_map error" << endl;
			exit(1);
			}
		if (t3 == 0) { // ai
			index_of_line(transversal, i1, null);
			if (i1 != i) {
				cout << "surface_domain::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
				}
			ijk2lmn(i, j, k, l, m, n);
			// bi, clm, cnm, cln, bj, bk
			hds[0] = line_bi(i);
			hds[1] = line_cij(l, m);
			hds[2] = line_cij(n, m);
			hds[3] = line_cij(l, n);
			hds[4] = line_bi(j);
			hds[5] = line_bi(k);
			}
		else if (t3 == 1) { // bi
			index_of_line(transversal, i1, null);
			if (i1 != i) {
				cout << "surface_domain::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
				exit(1);
				}
			ijk2lmn(i, j, k, l, m, n);
			// ai, clm, cnm, cln, aj, ak
			hds[0] = line_ai(i);
			hds[1] = line_cij(l, m);
			hds[2] = line_cij(n, m);
			hds[3] = line_cij(l, n);
			hds[4] = line_ai(j);
			hds[5] = line_ai(k);
			}
		else if (t3 == 2) { // clm
			index_of_line(transversal, l, m);
			ijklm2n(i, j, k, l, m, n);
			// ai, bi, cmn, cln, ckn, cjn
			hds[0] = line_ai(i);
			hds[1] = line_bi(i);
			hds[2] = line_cij(m, n);
			hds[3] = line_cij(l, n);
			hds[4] = line_cij(k, n);
			hds[5] = line_cij(j, n);
			}
		}
	else {
		cout << "surface_domain::get_half_double_six_associated_"
				"with_Clebsch_map error" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface_domain::get_half_double_six_associated_"
				"with_Clebsch_map done" << endl;
		}
}

void surface_domain::prepare_clebsch_map(int ds, int ds_row,
	int &line1, int &line2, int &transversal,
	int verbose_level)
{
	int ij, i, j, k, l, m, n, size_complement;
	int set[6];
	combinatorics_domain Combi;

	if (ds == 0) {
		if (ds_row == 0) {
			line1 = line_bi(0);
			line2 = line_bi(1);
			transversal = line_cij(0, 1);
			return;
			}
		else {
			line1 = line_ai(0);
			line2 = line_ai(1);
			transversal = line_cij(0, 1);
			return;
			}
		}
	ds--;
	if (ds < 15) {
		ij = ds;
		Combi.k2ij(ij, i, j, 6);

		if (ds_row == 0) {
			line1 = line_ai(j);
			line2 = line_bi(j);
			transversal = line_cij(i, j);
			return;
			}
		else {
			line1 = line_ai(i);
			line2 = line_bi(i);
			transversal = line_cij(i, j);
			return;
			}
		}
	ds -= 15;
	Combi.unrank_k_subset(ds, set, 6, 3);
	Combi.set_complement(set, 3 /* subset_size */, set + 3,
		size_complement, 6 /* universal_set_size */);
	i = set[0];
	j = set[1];
	k = set[2];
	l = set[3];
	m = set[4];
	n = set[5];
	if (ds_row == 0) {
		line1 = line_bi(l);
		line2 = line_bi(m);
		transversal = line_ai(n);
		return;
		}
	else {
		line1 = line_ai(i);
		line2 = line_ai(j);
		transversal = line_bi(k);
		return;
		}

}

int surface_domain::clebsch_map(long int *Lines, long int *Pts, int nb_pts,
	int line_idx[2], long int plane_rk,
	long int *Image_rk, int *Image_coeff,
	int verbose_level)
// assuming:
// In:
// Lines[27]
// Pts[nb_pts]
// Out:
// Image_rk[nb_pts]  (image point in the plane in local coordinates)
//   Note Image_rk[i] is -1 if Pts[i] does not have an image.
// Image_coeff[nb_pts * 4] (image point in the plane in PG(3,q) coordinates)
{
	int f_v = (verbose_level >= 1);
	int Plane[4 * 4];
	int Line_a[2 * 4];
	int Line_b[2 * 4];
	int Dual_planes[4 * 4];
		// dual coefficients of three planes:
		// the first plane is line_a together with the surface point
		// the second plane is line_b together with the surface point
		// the third plane is the plane onto which we map.
		// the fourth row is for the image point.
	int M[4 * 4];
	int v[4];
	long int i, h, pt, r;
	int coefficients[3];
	int base_cols[4];

	if (f_v) {
		cout << "surface_domain::clebsch_map" << endl;
		}
	P->Grass_planes->unrank_lint_here(Plane, plane_rk,
			0 /* verbose_level */);
	r = F->Gauss_simple(Plane, 3, 4, base_cols,
			0 /* verbose_level */);
	if (f_v) {
		cout << "Plane rank " << plane_rk << " :" << endl;
		int_matrix_print(Plane, 3, 4);
		}

	F->RREF_and_kernel(4, 3, Plane, 0 /* verbose_level */);

	if (f_v) {
		cout << "Plane (3 basis vectors and dual coordinates):" << endl;
		int_matrix_print(Plane, 4, 4);
		cout << "base_cols: ";
		int_vec_print(cout, base_cols, r);
		cout << endl;
		}

	// make sure the two lines are not contained in
	// the plane onto which we map:

	// test line_a:
	P->Grass_lines->unrank_lint_here(Line_a,
		Lines[line_idx[0]], 0 /* verbose_level */);
	if (f_v) {
		cout << "Line a = " << Line_label_tex[line_idx[0]]
			<< " = " << Lines[line_idx[0]] << ":" << endl;
		int_matrix_print(Line_a, 2, 4);
		}
	for (i = 0; i < 2; i++) {
		if (F->dot_product(4, Line_a + i * 4, Plane + 3 * 4)) {
			break;
			}
		}
	if (i == 2) {
		cout << "surface_domain::clebsch_map Line a lies "
				"inside the hyperplane" << endl;
		return FALSE;
		}

	// test line_b:
	P->Grass_lines->unrank_lint_here(Line_b,
		Lines[line_idx[1]], 0 /* verbose_level */);
	if (f_v) {
		cout << "Line b = " << Line_label_tex[line_idx[1]]
			<< " = " << Lines[line_idx[1]] << ":" << endl;
		int_matrix_print(Line_b, 2, 4);
		}
	for (i = 0; i < 2; i++) {
		if (F->dot_product(4, Line_b + i * 4, Plane + 3 * 4)) {
			break;
			}
		}
	if (i == 2) {
		cout << "surface_domain::clebsch_map Line b lies "
				"inside the hyperplane" << endl;
		return FALSE;
		}

	// and now, map all surface points:
	for (h = 0; h < nb_pts; h++) {
		pt = Pts[h];

		unrank_point(v, pt);

		int_vec_zero(Image_coeff + h * 4, 4);
		if (f_v) {
			cout << "pt " << h << " / " << nb_pts << " is " << pt << " = ";
			int_vec_print(cout, v, 4);
			cout << ":" << endl;
			}

		// make sure the points do not lie on either line_a or line_b
		// because the map is undefined there:
		int_vec_copy(Line_a, M, 2 * 4);
		int_vec_copy(v, M + 2 * 4, 4);
		if (F->Gauss_easy(M, 3, 4) == 2) {
			if (f_v) {
				cout << "The point is on line_a" << endl;
				}
			Image_rk[h] = -1;
			continue;
			}
		int_vec_copy(Line_b, M, 2 * 4);
		int_vec_copy(v, M + 2 * 4, 4);
		if (F->Gauss_easy(M, 3, 4) == 2) {
			if (f_v) {
				cout << "The point is on line_b" << endl;
				}
			Image_rk[h] = -1;
			continue;
			}

		// The point is good:

		// Compute the first plane in dual coordinates:
		int_vec_copy(Line_a, M, 2 * 4);
		int_vec_copy(v, M + 2 * 4, 4);
		F->RREF_and_kernel(4, 3, M, 0 /* verbose_level */);
		int_vec_copy(M + 3 * 4, Dual_planes, 4);
		if (f_v) {
			cout << "First plane in dual coordinates: ";
			int_vec_print(cout, M + 3 * 4, 4);
			cout << endl;
			}

		// Compute the second plane in dual coordinates:
		int_vec_copy(Line_b, M, 2 * 4);
		int_vec_copy(v, M + 2 * 4, 4);
		F->RREF_and_kernel(4, 3, M, 0 /* verbose_level */);
		int_vec_copy(M + 3 * 4, Dual_planes + 4, 4);
		if (f_v) {
			cout << "Second plane in dual coordinates: ";
			int_vec_print(cout, M + 3 * 4, 4);
			cout << endl;
			}


		// The third plane is the image
		// plane, given by dual coordinates:
		int_vec_copy(Plane + 3 * 4, Dual_planes + 8, 4);
		if (f_v) {
			cout << "Dual coordinates for all three planes: " << endl;
			int_matrix_print(Dual_planes, 3, 4);
			cout << endl;
			}

		r = F->RREF_and_kernel(4, 3,
				Dual_planes, 0 /* verbose_level */);
		if (f_v) {
			cout << "Dual coordinates and perp: " << endl;
			int_matrix_print(Dual_planes, 4, 4);
			cout << endl;
			cout << "matrix of dual coordinates has rank " << r << endl;
			}


		if (r < 3) {
			if (f_v) {
				cout << "The line is contained in the plane" << endl;
				}
			Image_rk[h] = -1;
			continue;
			}
		F->PG_element_normalize(Dual_planes + 12, 1, 4);
		if (f_v) {
			cout << "intersection point normalized: ";
			int_vec_print(cout, Dual_planes + 12, 4);
			cout << endl;
			}
		int_vec_copy(Dual_planes + 12, Image_coeff + h * 4, 4);

		// compute local coordinates of the image point:
		F->reduce_mod_subspace_and_get_coefficient_vector(
			3, 4, Plane, base_cols,
			Dual_planes + 12, coefficients,
			0 /* verbose_level */);
		Image_rk[h] = P2->rank_point(coefficients);
		if (f_v) {
			cout << "pt " << h << " / " << nb_pts
				<< " is " << pt << " : image = ";
			int_vec_print(cout, Image_coeff + h * 4, 4);
			cout << " image = " << Image_rk[h] << endl;
			}
		}

	if (f_v) {
		cout << "surface_domain::clebsch_map done" << endl;
		}
	return TRUE;
}

void surface_domain::clebsch_cubics(int verbose_level)
{
	int f_v = (verbose_level >= 1);



	if (f_v) {
		cout << "surface_domain::clebsch_cubics" << endl;
		}

	if (!f_has_large_polynomial_domains) {
		cout << "surface::clebsch_cubics f_has_large_"
				"polynomial_domains is FALSE" << endl;
		exit(1);
		}
	int Monomial[27];

	int i, j, idx;

	Clebsch_Pij = NEW_int(3 * 4 * nb_monomials2);
	Clebsch_P = NEW_pint(3 * 4);
	Clebsch_P3 = NEW_pint(3 * 3);

	int_vec_zero(Clebsch_Pij, 3 * 4 * nb_monomials2);


	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			Clebsch_P[i * 4 + j] =
				Clebsch_Pij + (i * 4 + j) * nb_monomials2;
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Clebsch_P3[i * 3 + j] =
				Clebsch_Pij + (i * 4 + j) * nb_monomials2;
			}
		}
	int coeffs[] = {
		1, 15, 2, 11,
		1, 16, 2, 12,
		1, 17, 2, 13,
		1, 18, 2, 14,
		0, 3, 2, 19,
		0, 4, 2, 20,
		0, 5, 2, 21,
		0, 6, 2, 22,
		0, 23, 1, 7,
		0, 24, 1, 8,
		0, 25, 1, 9,
		0, 26, 1, 10
		};
	int c0, c1;

	if (f_v) {
		cout << "surface_domain::clebsch_cubics "
				"Setting up the matrix P:" << endl;
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			cout << "i=" << i << " j=" << j << endl;
			int_vec_zero(Monomial, 27);
			c0 = coeffs[(i * 4 + j) * 4 + 0];
			c1 = coeffs[(i * 4 + j) * 4 + 1];
			int_vec_zero(Monomial, 27);
			Monomial[c0] = 1;
			Monomial[c1] = 1;
			idx = Poly2_27->index_of_monomial(Monomial);
			Clebsch_P[i * 4 + j][idx] = 1;
			c0 = coeffs[(i * 4 + j) * 4 + 2];
			c1 = coeffs[(i * 4 + j) * 4 + 3];
			int_vec_zero(Monomial, 27);
			Monomial[c0] = 1;
			Monomial[c1] = 1;
			idx = Poly2_27->index_of_monomial(Monomial);
			Clebsch_P[i * 4 + j][idx] = 1;
			}
		}


	if (f_v) {
		cout << "surface_domain::clebsch_cubics the matrix "
				"Clebsch_P is:" << endl;
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			cout << "Clebsch_P_" << i << "," << j << ":";
			Poly2_27->print_equation(cout, Clebsch_P[i * 4 + j]);
			cout << endl;
			}
		}

	int *Cubics;
	int *Adjugate;
	int *Ad[3 * 3];
	int *C[4];
	int m1;


	if (f_v) {
		cout << "surface_domain::clebsch_cubics allocating cubics" << endl;
		}

	Cubics = NEW_int(4 * nb_monomials6);
	int_vec_zero(Cubics, 4 * nb_monomials6);

	Adjugate = NEW_int(3 * 3 * nb_monomials4);
	int_vec_zero(Adjugate, 3 * 3 * nb_monomials4);

	for (i = 0; i < 4; i++) {
		C[i] = Cubics + i * nb_monomials6;
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Ad[i * 3 + j] = Adjugate + (i * 3 + j) * nb_monomials4;
			}
		}

	if (f_v) {
		cout << "surface_domain::clebsch_cubics computing "
				"C[3] = the determinant" << endl;
		}
	// compute C[3] as the negative of the determinant
	// of the matrix of the first three columns:
	//int_vec_zero(C[3], nb_monomials6);
	m1 = F->negate(1);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 0],
			Clebsch_P[1 * 4 + 1],
			Clebsch_P[2 * 4 + 2], m1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 1],
			Clebsch_P[1 * 4 + 2],
			Clebsch_P[2 * 4 + 0], m1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 2],
			Clebsch_P[1 * 4 + 0],
			Clebsch_P[2 * 4 + 1], m1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 0],
			Clebsch_P[1 * 4 + 1],
			Clebsch_P[0 * 4 + 2], 1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 1],
			Clebsch_P[1 * 4 + 2],
			Clebsch_P[0 * 4 + 0], 1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 2],
			Clebsch_P[1 * 4 + 0],
			Clebsch_P[0 * 4 + 1], 1, C[3],
			0 /* verbose_level*/);

	int I[3];
	int J[3];
	int size_complement, scalar;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "surface_domain::clebsch_cubics computing adjugate" << endl;
		}
	// compute adjugate:
	for (i = 0; i < 3; i++) {
		I[0] = i;
		Combi.set_complement(I, 1, I + 1, size_complement, 3);
		for (j = 0; j < 3; j++) {
			J[0] = j;
			Combi.set_complement(J, 1, J + 1, size_complement, 3);

			if ((i + j) % 2) {
				scalar = m1;
				}
			else {
				scalar = 1;
				}
			minor22(Clebsch_P3, I[1], I[2], J[1], J[2], scalar,
					Ad[j * 3 + i], 0 /* verbose_level */);
			}
		}

	// multiply adjugate * last column:
	if (f_v) {
		cout << "surface_domain::clebsch_cubics multiply adjugate "
				"times last column" << endl;
		}

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			multiply42_and_add(Ad[i * 3 + j],
					Clebsch_P[j * 4 + 3], C[i], 0 /* verbose_level */);
			}
		}

	if (f_v) {
		cout << "surface_domain::clebsch_cubics We have "
				"computed the Clebsch cubics" << endl;
		}


	int Y[3];
	int M24[24];
	int h;

	Clebsch_coeffs = NEW_int(4 * Poly3->nb_monomials * nb_monomials3);
	int_vec_zero(Clebsch_coeffs,
			4 * Poly3->nb_monomials * nb_monomials3);
	CC = NEW_pint(4 * Poly3->nb_monomials);
	for (i = 0; i < 4; i++) {
		for (j = 0; j < Poly3->nb_monomials; j++) {
			CC[i * Poly3->nb_monomials + j] =
				Clebsch_coeffs +
					(i * Poly3->nb_monomials + j) * nb_monomials3;
			}
		}
	for (i = 0; i < Poly3->nb_monomials; i++) {
		int_vec_copy(Poly3->Monomials + i * 3, Y, 3);
		for (j = 0; j < nb_monomials6; j++) {
			if (int_vec_compare(Y, Poly6_27->Monomials + j * 27, 3) == 0) {
				int_vec_copy(Poly6_27->Monomials + j * 27 + 3, M24, 24);
				idx = Poly3_24->index_of_monomial(M24);
				for (h = 0; h < 4; h++) {
					CC[h * Poly3->nb_monomials + i][idx] =
						F->add(CC[h * Poly3->nb_monomials + i][idx],
								C[h][j]);
					}
				}
			}
		}

	if (f_v) {
		print_clebsch_cubics(cout);
		}

	FREE_int(Cubics);
	FREE_int(Adjugate);

	if (f_v) {
		cout << "surface_domain::clebsch_cubics done" << endl;
		}
}

void surface_domain::multiply_222_27_and_add(int *M1, int *M2, int *M3,
	int scalar, int *MM, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a, b, c, d, idx;
	int M[27];

	if (f_v) {
		cout << "surface_domain::multiply_222_27_and_add" << endl;
		}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_domain::multiply_222_27_and_add "
				"f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	//int_vec_zero(MM, nb_monomials6);
	for (i = 0; i < nb_monomials2; i++) {
		a = M1[i];
		if (a == 0) {
			continue;
			}
		for (j = 0; j < nb_monomials2; j++) {
			b = M2[j];
			if (b == 0) {
				continue;
				}
			for (k = 0; k < nb_monomials2; k++) {
				c = M3[k];
				if (c == 0) {
					continue;
					}
				d = F->mult3(a, b, c);
				int_vec_add3(Poly2_27->Monomials + i * 27,
					Poly2_27->Monomials + j * 27,
					Poly2_27->Monomials + k * 27,
					M, 27);
				idx = Poly6_27->index_of_monomial(M);
				if (idx >= nb_monomials6) {
					cout << "surface_domain::multiply_222_27_and_add "
							"idx >= nb_monomials6" << endl;
					exit(1);
					}
				d = F->mult(scalar, d);
				MM[idx] = F->add(MM[idx], d);
				}
			}
		}


	if (f_v) {
		cout << "surface_domain::multiply_222_27_and_add done" << endl;
		}
}

void surface_domain::minor22(int **P3, int i1, int i2, int j1, int j2,
	int scalar, int *Ad, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, d, idx;
	int M[27];

	if (f_v) {
		cout << "surface_domain::minor22" << endl;
		}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_domain::minor22 "
				"f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	int_vec_zero(Ad, nb_monomials4);
	for (i = 0; i < nb_monomials2; i++) {
		a = P3[i1 * 3 + j1][i];
		if (a == 0) {
			continue;
			}
		for (j = 0; j < nb_monomials2; j++) {
			b = P3[i2 * 3 + j2][j];
			if (b == 0) {
				continue;
				}
			d = F->mult(a, b);
			int_vec_add(Poly2_27->Monomials + i * 27,
				Poly2_27->Monomials + j * 27,
				M, 27);
			idx = Poly4_27->index_of_monomial(M);
			if (idx >= nb_monomials4) {
				cout << "surface_domain::minor22 "
						"idx >= nb_monomials4" << endl;
				exit(1);
				}
			d = F->mult(scalar, d);
			Ad[idx] = F->add(Ad[idx], d);
			}
		}
	for (i = 0; i < nb_monomials2; i++) {
		a = P3[i2 * 3 + j1][i];
		if (a == 0) {
			continue;
			}
		for (j = 0; j < nb_monomials2; j++) {
			b = P3[i1 * 3 + j2][j];
			if (b == 0) {
				continue;
				}
			d = F->mult(a, b);
			int_vec_add(Poly2_27->Monomials + i * 27,
				Poly2_27->Monomials + j * 27,
				M, 27);
			idx = Poly4_27->index_of_monomial(M);
			if (idx >= nb_monomials4) {
				cout << "surface_domain::minor22 "
						"idx >= nb_monomials4" << endl;
				exit(1);
				}
			d = F->mult(scalar, d);
			d = F->negate(d);
			Ad[idx] = F->add(Ad[idx], d);
			}
		}


	if (f_v) {
		cout << "surface_domain::minor22 done" << endl;
		}
}

void surface_domain::multiply42_and_add(int *M1, int *M2,
		int *MM, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, d, idx;
	int M[27];

	if (f_v) {
		cout << "surface_domain::multiply42_and_add" << endl;
		}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_domain::multiply42_and_add "
				"f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	for (i = 0; i < nb_monomials4; i++) {
		a = M1[i];
		if (a == 0) {
			continue;
			}
		for (j = 0; j < nb_monomials2; j++) {
			b = M2[j];
			if (b == 0) {
				continue;
				}
			d = F->mult(a, b);
			int_vec_add(Poly4_27->Monomials + i * 27,
				Poly2_27->Monomials + j * 27,
				M, 27);
			idx = Poly6_27->index_of_monomial(M);
			if (idx >= nb_monomials6) {
				cout << "surface_domain::multiply42_and_add "
						"idx >= nb_monomials6" << endl;
				exit(1);
				}
			MM[idx] = F->add(MM[idx], d);
			}
		}

	if (f_v) {
		cout << "surface_domain::multiply42_and_add done" << endl;
		}
}

void surface_domain::prepare_system_from_FG(int *F_planes, int *G_planes,
	int lambda, int *&system, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;


	if (f_v) {
		cout << "surface_domain::prepare_system_from_FG" << endl;
		}
	system = NEW_int(3 * 4 * 3);
	int_vec_zero(system, 3 * 4 * 3);
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			int *p = system + (i * 4 + j) * 3;
			if (i == 0) {
				p[0] = 0;
				p[1] = F->mult(lambda, G_planes[0 * 4 + j]);
				p[2] = F_planes[2 * 4 + j];
				}
			else if (i == 1) {
				p[0] = F_planes[0 * 4 + j];
				p[1] = 0;
				p[2] = G_planes[1 * 4 + j];
				}
			else if (i == 2) {
				p[0] = G_planes[2 * 4 + j];
				p[1] = F_planes[1 * 4 + j];
				p[2] = 0;
				}
			}
		}
	if (f_v) {
		cout << "surface_domain::prepare_system_from_FG done" << endl;
		}
}


void surface_domain::compute_nine_lines(int *F_planes, int *G_planes,
	long int *nine_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int Basis[16];

	if (f_v) {
		cout << "surface_domain::compute_nine_lines" << endl;
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			int_vec_copy(F_planes + i * 4, Basis, 4);
			int_vec_copy(G_planes + j * 4, Basis + 4, 4);
			F->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
			nine_lines[i * 3 + j] = Gr->rank_lint_here(
				Basis + 8, 0 /* verbose_level */);
			}
		}
	if (f_v) {
		cout << "The nine lines are: ";
		lint_vec_print(cout, nine_lines, 9);
		cout << endl;
		}
	if (f_v) {
		cout << "surface_domain::compute_nine_lines done" << endl;
		}
}

void surface_domain::compute_nine_lines_by_dual_point_ranks(
	long int *F_planes_rank,
	long int *G_planes_rank, long int *nine_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int F_planes[12];
	int G_planes[12];
	int Basis[16];

	if (f_v) {
		cout << "surface_domain::compute_nine_lines_by_dual_"
				"point_ranks" << endl;
		}
	for (i = 0; i < 3; i++) {
		P->unrank_point(F_planes + i * 4, F_planes_rank[i]);
		}
	for (i = 0; i < 3; i++) {
		P->unrank_point(G_planes + i * 4, G_planes_rank[i]);
		}

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			int_vec_copy(F_planes + i * 4, Basis, 4);
			int_vec_copy(G_planes + j * 4, Basis + 4, 4);
			F->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
			nine_lines[i * 3 + j] = Gr->rank_lint_here(
				Basis + 8, 0 /* verbose_level */);
			}
		}
	if (f_v) {
		cout << "The nine lines are: ";
		lint_vec_print(cout, nine_lines, 9);
		cout << endl;
		}
	if (f_v) {
		cout << "surface_domain::compute_nine_lines_by_dual_"
				"point_ranks done" << endl;
		}
}

void surface_domain::split_nice_equation(int *nice_equation,
	int *&f1, int *&f2, int *&f3, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::split_nice_equation" << endl;
		}
	int M[4];
	int i, a, idx;

	f1 = NEW_int(Poly1->nb_monomials);
	f2 = NEW_int(Poly2->nb_monomials);
	f3 = NEW_int(Poly3->nb_monomials);
	int_vec_zero(f1, Poly1->nb_monomials);
	int_vec_zero(f2, Poly2->nb_monomials);
	int_vec_zero(f3, Poly3->nb_monomials);

	for (i = 0; i < 20; i++) {
		a = nice_equation[i];
		if (a == 0) {
			continue;
			}
		int_vec_copy(Poly3_4->Monomials + i * 4, M, 4);
		if (M[0] == 3) {
			cout << "surface_domain::split_nice_equation the x_0^3 "
				"term is supposed to be zero" << endl;
			exit(1);
			}
		else if (M[0] == 2) {
			idx = Poly1->index_of_monomial(M + 1);
			f1[idx] = a;
			}
		else if (M[0] == 1) {
			idx = Poly2->index_of_monomial(M + 1);
			f2[idx] = a;
			}
		else if (M[0] == 0) {
			idx = Poly3->index_of_monomial(M + 1);
			f3[idx] = a;
			}
		}
	if (f_v) {
		cout << "surface_domain::split_nice_equation done" << endl;
		}
}

void surface_domain::assemble_tangent_quadric(
	int *f1, int *f2, int *f3,
	int *&tangent_quadric, int verbose_level)
// 2*x_0*f_1 + f_2
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::assemble_tangent_quadric" << endl;
		}
	int M[4];
	int i, a, idx, two;


	two = F->add(1, 1);
	tangent_quadric = NEW_int(Poly2_4->nb_monomials);
	int_vec_zero(tangent_quadric, Poly2_4->nb_monomials);

	for (i = 0; i < Poly1->nb_monomials; i++) {
		a = f1[i];
		if (a == 0) {
			continue;
			}
		int_vec_copy(Poly1->Monomials + i * 3, M + 1, 3);
		M[0] = 1;
		idx = Poly2_4->index_of_monomial(M);
		tangent_quadric[idx] = F->mult(two, a);
		}

	for (i = 0; i < Poly2->nb_monomials; i++) {
		a = f2[i];
		if (a == 0) {
			continue;
			}
		int_vec_copy(Poly2->Monomials + i * 3, M + 1, 3);
		M[0] = 0;
		idx = Poly2_4->index_of_monomial(M);
		tangent_quadric[idx] = a;
		}
	if (f_v) {
		cout << "surface_domain::assemble_tangent_quadric done" << endl;
		}
}

void surface_domain::tritangent_plane_to_trihedral_pair_and_position(
	int tritangent_plane_idx,
	int &trihedral_pair_idx, int &position, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	static int Table[] = {
		0, 2, // 0
		0, 5, // 1
		22, 0, // 2
		0, 1, // 3
		20, 0, // 4
		1, 1, // 5
		26, 0, //6
		5, 1, // 7
		32, 0, //8
		6, 1, //9
		0, 0, //10
		25, 0, // 11
		1, 0, // 12
		43, 0, //13
		2, 0, //14
		55, 0, // 15
		3, 0, // 16
		3, 3, //17
		4, 0, //18
		67, 0, // 19
		5, 0, // 20
		73, 0, // 21
		6, 0, // 22
		6, 3, // 23
		7, 0, // 24
		79, 0, // 25
		8, 0, // 26
		8, 3, // 27
		9, 0, // 28
		9, 3, // 29
		115, 0, // 30
		114, 0, // 31
		34, 2, // 32
		113, 0, // 33
		111, 0, // 34
		34, 5, // 35
		74, 2, // 36
		110, 0, // 37
		49, 2, // 38
		26, 5, // 39
		38, 5, // 40
		53, 5, // 41
		36, 5, // 42
		45, 5, // 43
		51, 5, // 44
		};

	if (f_v) {
		cout << "surface_domain::tritangent_plane_to_trihedral_"
				"pair_and_position" << endl;
		}
	trihedral_pair_idx = Table[2 * tritangent_plane_idx + 0];
	position = Table[2 * tritangent_plane_idx + 1];
	if (f_v) {
		cout << "surface_domain::tritangent_plane_to_trihedral_"
				"pair_and_position done" << endl;
		}
}

void surface_domain::do_arc_lifting_with_two_lines(
	long int *Arc6, int p1_idx, int p2_idx, int partition_rk,
	long int line1, long int line2,
	int *coeff20, long int *lines27,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int arc[6];
	long int P1, P2;


	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines" << endl;
		cout << "Arc6: ";
		lint_vec_print(cout, Arc6, 6);
		cout << endl;
		cout << "p1_idx=" << p1_idx << " p2_idx=" << p2_idx
				<< " partition_rk=" << partition_rk
				<< " line1=" << line1 << " line2=" << line2 << endl;
	}

	P1 = Arc6[p1_idx];
	P2 = Arc6[p2_idx];

	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines before "
				"P->rearrange_arc_for_lifting" << endl;
		}
	P->rearrange_arc_for_lifting(Arc6,
				P1, P2, partition_rk, arc,
				verbose_level);

	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines after "
				"P->rearrange_arc_for_lifting" << endl;
		cout << "arc: ";
		lint_vec_print(cout, arc, 6);
		cout << endl;
		}

	arc_lifting_with_two_lines *AL;

	AL = NEW_OBJECT(arc_lifting_with_two_lines);


	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines before "
				"AL->create_surface" << endl;
		}
	AL->create_surface(this, arc, line1, line2, verbose_level);
	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines after "
				"AL->create_surface" << endl;
		cout << "equation: ";
		int_vec_print(cout, AL->coeff, 20);
		cout << endl;
		cout << "lines: ";
		lint_vec_print(cout, AL->lines27, 27);
		cout << endl;
		}

	int_vec_copy(AL->coeff, coeff20, 20);
	lint_vec_copy(AL->lines27, lines27, 27);


	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines done" << endl;
	}
}


}}
