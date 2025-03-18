/*
 * special_functions_domain.cpp
 *
 *  Created on: Jan 21, 2025
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace special_functions {


special_functions_domain::special_functions_domain()
{
	Record_birth();

	Fq = NULL;
	q = 0;

	P = NULL;

	nb_vars = 0;
	max_degree = 0;

}

special_functions_domain::~special_functions_domain()
{
	Record_death();
}

void special_functions_domain::init(
		geometry::projective_geometry::projective_space *P,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "special_functions_domain::init" << endl;
	}


	special_functions_domain::P = P;
	Fq = P->Subspaces->F;
	nb_vars = P->Subspaces->n + 1;

	q = Fq->q;

	max_degree = (q - 1) * nb_vars;

	if (f_v) {
		cout << "special_functions_domain::init q = " << q << endl;
		cout << "special_functions_domain::init nb_vars = " << nb_vars << endl;
		cout << "special_functions_domain::init max_degree = " << max_degree << endl;
	}

	if (f_v) {
		cout << "special_functions_domain::init done" << endl;
	}
}


void special_functions_domain::make_polynomial_representation(
		long int *Pts, int nb_pts,
		std::string &poly_rep,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "special_functions_domain::make_polynomial_representation" << endl;
	}
	if (f_v) {
		cout << "special_functions_domain::make_polynomial_representation set = ";
		Lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
	}

	long int *Pts_complement;
	int N;

	N = P->Subspaces->N_points;

	Pts_complement = NEW_lint(N);

	if (nb_pts > P->Subspaces->N_points) {
		cout << "special_functions_domain::make_polynomial_representation "
				"nb_pts > P->Subspaces->N_points" << endl;
		exit(1);
	}

	other::data_structures::sorting Sorting;

	Lint_vec_copy(Pts, Pts_complement, nb_pts);

	Sorting.lint_vec_sort_and_remove_duplicates(
			Pts_complement, nb_pts);

	if (f_v) {
		cout << "special_functions_domain::make_polynomial_representation set sorted = ";
		Lint_vec_print(cout, Pts_complement, nb_pts);
		cout << endl;
	}

	Lint_vec_complement_to(
			Pts_complement, Pts_complement + nb_pts, N, nb_pts);



	int nb_pts_complement;

	nb_pts_complement = N - nb_pts;

	if (f_v) {
		cout << "special_functions_domain::make_polynomial_representation complementary set = ";
		Lint_vec_print(cout, Pts_complement + nb_pts, nb_pts_complement);
		cout << endl;
	}

	int *Pt_coords;
	int i;


	Pt_coords = NEW_int(nb_pts_complement * nb_vars);

	for (i = 0; i < nb_pts_complement; i++) {
		P->unrank_point(Pt_coords + i * nb_vars, Pts_complement[nb_pts + i]);
	}

	if (f_v) {
		cout << "special_functions_domain::make_polynomial_representation "
				"list of points of the complementary set in coordinates:" << endl;
		Int_matrix_print(Pt_coords, nb_pts_complement, nb_vars);
	}


	string s;
	int h, j;

	int *J;

	J = NEW_int(nb_pts_complement);
	for (h = 0; h < nb_pts_complement; h++) {
		for (j = nb_vars - 1; j >= 0; j--) {
			if (Pt_coords[h * nb_vars + j]) {
				break;
			}
		}
		if (j < 0) {
			cout << "special_functions_domain::make_polynomial_representation j < 0" << endl;
			exit(1);
		}
		J[h] = j;
	}
	if (f_v) {
		cout << "special_functions_domain::make_polynomial_representation J = ";
		Int_vec_print(cout, J, nb_pts_complement);
		cout << endl;
	}

	for (h = 0; h < nb_pts_complement; h++) {

		string sh, qm1, power_qm1;

		int *v;

		v = Pt_coords + h * nb_vars;

		j = J[h];
		if (q - 1 >= 10) {
			qm1 = "(" + std::to_string(q - 1) + ")";
		}
		else {
			qm1 = std::to_string(q - 1);
		}

		if (q > 2) {
			power_qm1 = "^" + qm1;
		}
		else {
			power_qm1 = "";
		}

		for (i = 0; i < nb_vars; i++) {

			string Xj, Xi;

			Xi = "X" + std::to_string(i);
			Xj = "X" + std::to_string(j);

			string si;

			if (v[i]) {
				if (v[i] == 1) {
					si = "(" + Xj + power_qm1 + "-(" + Xi + "-" + Xj + ")" + power_qm1 + ")";
				}
				else {
					si = "(" + Xj + power_qm1 + "-(" + Xi + "-1*" + std::to_string(v[i]) + "*" + Xj + ")" + power_qm1 + ")";
				}
			}
			else {
				si = "(" + Xj + power_qm1 + "-" + Xi + power_qm1 + ")";

			}
			if (i == 0) {
				sh = si;
			}
			else {
				sh = sh + " * " + si;
			}
		}

		if (h == 0) {
			s = sh;
		}
		else {
			s = s + " + " + sh;
		}
 	}


	poly_rep = s;


	FREE_lint(Pts_complement);
	FREE_int(Pt_coords);
	FREE_int(J);

	if (f_v) {
		cout << "special_functions_domain::make_polynomial_representation done" << endl;
	}
}


}}}}


