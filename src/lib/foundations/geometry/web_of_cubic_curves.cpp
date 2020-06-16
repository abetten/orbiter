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

	if (f_v) {
		cout << "web_of_cubic_curves::init" << endl;
	}
	web_of_cubic_curves::Surf = Surf;

	lint_vec_copy(arc6, web_of_cubic_curves::arc6, 6);


	if (f_v) {
		cout << "web_of_cubic_curves::init before find_Eckardt_points" << endl;
	}
	find_Eckardt_points(verbose_level);
	if (f_v) {
		cout << "web_of_cubic_curves::init after find_Eckardt_points" << endl;
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
		cout << "web_of_cubic_curves::init before find_trihedral_pairs" << endl;
	}
	find_trihedral_pairs(verbose_level);
	if (f_v) {
		cout << "web_of_cubic_curves::init after find_trihedral_pairs" << endl;
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

	int_vec_copy(Surf->Trihedral_to_Eckardt +
			t_idx0 * 6, row_col_Eckardt_points, 6);

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
		int_vec_print(cout, base_curves4, 4);
		cout << endl;
	}

	if (f_v) {
		cout << "web_of_cubic_curves::init "
				"Creating the web of cubic "
				"curves through the arc:" << endl;
	}
	Surf->create_web_of_cubic_curves_and_equations_based_on_four_tritangent_planes(
		arc6, base_curves4,
		Web_of_cubic_curves, Tritangent_plane_equations,
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
		cout << "web_of_cubic_curves::init "
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
				"Tritangent_plane_equations:" << endl;
		int_matrix_print(Tritangent_plane_equations, 45, 4);
	}

	The_plane_rank = NEW_lint(45);
	The_plane_duals = NEW_lint(45);

	latex_interface L;

	int Basis[16];
	for (i = 0; i < 45; i++) {
		int_vec_copy(Tritangent_plane_equations + i * 4, Basis, 4);
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
				Tritangent_plane_equations + i * 4);
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
			Tritangent_plane_equations, Lines27, verbose_level);
	if (f_v) {
		cout << "web_of_cubic_curves::init after "
				"Surf->create_lines_from_plane_equations" << endl;
	}




	if (f_v) {
		cout << "web_of_cubic_curves::init done" << endl;
	}
}

void web_of_cubic_curves::find_Eckardt_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "web_of_cubic_curves::find_Eckardt_points" << endl;
	}
	int s;

	if (f_v) {
		cout << "web_of_cubic_curves::find_Eckardt_points before Surf->P2->compute_eckardt_point_info" << endl;
	}
	E = Surf->P2->compute_eckardt_point_info(Surf, arc6, verbose_level);
	if (f_v) {
		cout << "web_of_cubic_curves::find_Eckardt_points after Surf->P2->compute_eckardt_point_info" << endl;
	}
	if (f_v) {
		cout << "web_of_cubic_curves::find_Eckardt_points We found " << E->nb_E
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
		int_vec_print(cout, E_idx, E->nb_E);
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
					<< Surf->Trihedral_pair_labels[t_idx] << "}" << endl;
		}
	}

	if (f_v) {
		cout << "web_of_cubic_curves::find_trihedral_pairs done" << endl;
	}
}

void web_of_cubic_curves::print_lines(ostream &ost)
{
	int i, a;
	int v[8];

	ost << "The 27 lines:\\\\";
	for (i = 0; i < 27; i++) {
		a = Lines27[i];
		ost << "$$" << endl;
		ost << "\\ell_{" << i << "} = "
				<< Surf->Line_label_tex[i] << " = " << a << " = ";
		Surf->unrank_line(v, a);
		//ost << "\\left[ " << endl;
		Surf->Gr->print_single_generator_matrix_tex(ost, a);
		//ost << "\\right] ";
		ost << "$$" << endl;
	}
}

void web_of_cubic_curves::print_trihedral_plane_equations(ostream &ost)
{
	latex_interface L;
	int i;

	ost << "The chosen abstract trihedral pair is no "
			<< t_idx0 << ":" << endl;
	ost << "$$" << endl;
	Surf->latex_abstract_trihedral_pair(ost, t_idx0);
	ost << "$$" << endl;
	ost << "The six planes in the trihedral pair are:" << endl;
	ost << "$$" << endl;
	int_vec_print(ost, row_col_Eckardt_points, 6);
	ost << "$$" << endl;
	ost << "We choose planes $0,1,3,4$ for the base curves:" << endl;
	ost << "$$" << endl;
	int_vec_print(ost, base_curves4, 4);
	ost << "$$" << endl;
	ost << "The four base curves are:\\\\";
	for (i = 0; i < 4; i++) {
		ost << "$$" << endl;
		ost << "W_{" << Surf->Eckard_point_label[base_curves4[i]];
		ost << "}=\\Phi\\big(\\pi_{" << base_curves4[i]
			<< "}\\big) = \\Phi\\big(\\pi_{"
			<< Surf->Eckard_point_label[base_curves4[i]]
			<< "}\\big)=V\\Big(" << endl;
		Surf->Poly3->print_equation(ost, base_curves + i * 10);
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
			<< Surf->Eckard_point_label[i] << "}=V\\Big(";
		Surf->Poly1_4->print_equation(ost,
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
	long int *plane6, ostream &ost)
{
	latex_interface L;
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
				<< Surf->Eckard_point_label[h] << "}=V\\big(";
		Surf->Poly1_4->print_equation(ost, Tritangent_plane_equations + h * 4);
		ost << "\\big)$\\\\";
	}
}

void web_of_cubic_curves::print_surface_equations_on_line(
	int *The_surface_equations,
	int lambda, int lambda_rk, ostream &ost)
{
	latex_interface L;
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

		Surf->F->PG_element_unrank_modified(v, 1, 2, i);
		Surf->F->PG_element_normalize_from_front(v, 1, 2);

		ost << "$";
		ost << v[0] << " \\cdot ";
		ost << "\\big(";
		Surf->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + row_col_Eckardt_points[0] * 4);
		ost << "\\big)";
		ost << "\\big(";
		Surf->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + row_col_Eckardt_points[1] * 4);
		ost << "\\big)";
		ost << "\\big(";
		Surf->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + row_col_Eckardt_points[2] * 4);
		ost << "\\big)";
		ost << "+";
		ost << v[1] << " \\cdot ";
		ost << "\\big(";
		Surf->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + row_col_Eckardt_points[3] * 4);
		ost << "\\big)";
		ost << "\\big(";
		Surf->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + row_col_Eckardt_points[4] * 4);
		ost << "\\big)";
		ost << "\\big(";
		Surf->Poly1_4->print_equation(ost,
				Tritangent_plane_equations + row_col_Eckardt_points[5] * 4);
		ost << "\\big)";
		ost << " = ";
		Surf->Poly3_4->print_equation(ost,
				The_surface_equations + i * 20);
		ost << "$\\\\";
	}
}

void web_of_cubic_curves::print_dual_point_ranks(ostream &ost)
{
	latex_interface L;

	ost << "Dual point ranks:\\\\";
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
			Dual_point_ranks, nb_T, 6, TRUE /* f_tex*/);
	ost << "$$" << endl;
}

void web_of_cubic_curves::print_Eckardt_point_data(ostream &ost, int verbose_level)
{
	E->print_bisecants(ost, verbose_level);
	E->print_intersections(ost, verbose_level);
	E->print_conics(ost, verbose_level);
}

void web_of_cubic_curves::report(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "web_of_cubic_curves::report" << endl;
	}

#if 1
	Surf->print_polynomial_domains(ost);
	Surf->print_line_labelling(ost);

	if (f_v) {
		cout << "web_of_cubic_curves::report before print_Steiner_and_Eckardt" << endl;
	}
	Surf->print_Steiner_and_Eckardt(ost);
	if (f_v) {
		cout << "web_of_cubic_curves::report after print_Steiner_and_Eckardt" << endl;
	}
#endif

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
	Surf->print_web_of_cubic_curves(arc6, Web_of_cubic_curves, ost);


	if (f_v) {
		cout << "web_of_cubic_curves::report before print_plane_equations" << endl;
	}
	print_trihedral_plane_equations(ost);


	//cout << "web_of_cubic_curves::report before print_dual_point_ranks" << endl;
	//print_dual_point_ranks(ost);


	if (f_v) {
		cout << "web_of_cubic_curves::report done" << endl;
	}

}

}}
