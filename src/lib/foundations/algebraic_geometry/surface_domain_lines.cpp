// surface_domain_lines.cpp
//
// Anton Betten
//
// moved here from surface.cpp: Dec 26, 2018
//
//
//
//

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {



void surface_domain::init_Schlaefli(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::init_Schlaefli" << endl;
	}

	Schlaefli = NEW_OBJECT(schlaefli);

	Schlaefli->init(this, verbose_level);


	if (f_v) {
		cout << "surface_domain::init_Schlaefli done" << endl;
	}
}



void surface_domain::unrank_line(int *v, long int rk)
{
	Gr->unrank_lint_here(v, rk, 0 /* verbose_level */);
}

void surface_domain::unrank_lines(int *v, long int *Rk, int nb)
{
	int i;

	for (i = 0; i < nb; i++) {
		Gr->unrank_lint_here(v + i * 8, Rk[i], 0 /* verbose_level */);
	}
}

long int surface_domain::rank_line(int *v)
{
	long int rk;

	rk = Gr->rank_lint_here(v, 0 /* verbose_level */);
	return rk;
}

void surface_domain::build_cubic_surface_from_lines(
	int len, long int *S,
	int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int r;
	int *System;
	int nb_rows;

	if (f_v) {
		cout << "surface_domain::build_cubic_surface_from_lines" << endl;
	}

	if (f_v) {
		cout << "surface_domain::build_cubic_surface_from_lines before create_system" << endl;
	}
	create_system(len, S, System, nb_rows, verbose_level);
	if (f_v) {
		cout << "surface_domain::build_cubic_surface_from_lines after create_system" << endl;
	}


	int base_cols[20];

	if (f_v) {
		cout << "surface_domain::build_cubic_surface_from_lines before F->Gauss_simple" << endl;
	}
	r = F->Linear_algebra->Gauss_simple(System, nb_rows, nb_monomials,
		base_cols, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::build_cubic_surface_from_lines after F->Gauss_simple" << endl;
	}

	if (FALSE) {
		cout << "surface_domain::create_system "
				"The system in RREF:" << endl;
		Orbiter->Int_vec->matrix_print(System, nb_rows, nb_monomials);
	}
	if (f_v) {
		cout << "surface_domain::create_system "
				"The system has rank " << r << endl;
	}


	if (r != nb_monomials - 1) {
		cout << "surface_domain::build_cubic_surface_from_lines "
				"r != nb_monomials - 1" << endl;
		cout << "r=" << r << endl;
		exit(1);
	}

	int kernel_m, kernel_n;

	if (f_v) {
		cout << "surface_domain::build_cubic_surface_from_lines before F->matrix_get_kernel" << endl;
	}
	F->Linear_algebra->matrix_get_kernel(System, r, nb_monomials, base_cols, r,
		kernel_m, kernel_n, coeff, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::build_cubic_surface_from_lines after F->matrix_get_kernel" << endl;
	}

	FREE_int(System);

	//cout << "kernel_m=" << kernel_m << endl;
	//cout << "kernel_n=" << kernel_n << endl;
	if (f_v) {
		cout << "surface_domain::build_cubic_surface_from_lines done" << endl;
	}
}

int surface_domain::rank_of_system(int len, long int *S,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *System;
	int nb_rows;
	int r;

	if (f_v) {
		cout << "surface_domain::rank_of_system" << endl;
	}
	create_system(len, S, System, nb_rows, verbose_level);


	int base_cols[20];

	r = F->Linear_algebra->Gauss_simple(System, nb_rows, nb_monomials,
		base_cols, 0 /* verbose_level */);


	FREE_int(System);

	if (f_v) {
		cout << "surface_domain::rank_of_system done" << endl;
	}

	return r;
}

void surface_domain::create_system(int len, long int *S,
		int *&System, int &nb_rows, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	long int a;
	int i, j;

	if (f_v) {
		cout << "surface_domain::create_system" << endl;
	}
	if (f_v) {
		cout << "surface_domain::create_system len = " << len << endl;
	}

	vector<long int> Pts;
	long int *pts_on_line;
	int *Pt_coords;


	pts_on_line = NEW_lint(P->k);
	//nb_pts = 0;
	for (i = 0; i < len; i++) {
		a = S[i];

		if (P->Implementation->Lines) {
			for (j = 0; j < P->k; j++) {
				Pts.push_back(P->Implementation->Lines[a * P->k + j]);
				//pt_list[nb_pts++] = P->Lines[a * P->k + j];
			}
		}
		else {
			if (f_v) {
				cout << "surface_domain::create_system before P->create_points_on_line" << endl;
			}
			P->create_points_on_line(a,
					pts_on_line, //pt_list + nb_pts,
					0 /* verbose_level */);
			if (f_v) {
				cout << "surface_domain::create_system after P->create_points_on_line" << endl;
			}
			//nb_pts += P->k;
			for (j = 0; j < P->k; j++) {
				Pts.push_back(pts_on_line[j]);
			}
		}
	}
	FREE_lint(pts_on_line);

#if 0
	if (nb_pts > max_pts) {
		cout << "surface_domain::create_system "
				"nb_pts > max_pts" << endl;
		exit(1);
		}
	if (FALSE) {
		cout << "surface_domain::create_system list of "
				"covered points by lines:" << endl;
		lint_matrix_print(pt_list, len, P->k);
		}
#endif

	if (f_v) {
		cout << "surface_domain::create_system list of "
				"covered points by lines has been created" << endl;
	}


	nb_rows = Pts.size();

	if (f_v) {
		cout << "surface_domain::create_system nb_rows = " << nb_rows << endl;
		cout << "surface_domain::create_system n = " << n << endl;
	}
	Pt_coords = NEW_int(nb_rows * n);

	for (i = 0; i < nb_rows; i++) {
		unrank_point(Pt_coords + i * n, Pts[i]);
	}

	if (f_v && FALSE) {
		cout << "surface_domain::create_system list of "
				"covered points in coordinates:" << endl;
		Orbiter->Int_vec->matrix_print(Pt_coords, nb_rows, n);
	}

	if (f_v) {
		cout << "surface_domain::create_system nb_rows = " << nb_rows << endl;
	}

	System = NEW_int(nb_rows * nb_monomials);

	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_monomials; j++) {
			System[i * nb_monomials + j] = Poly3_4->evaluate_monomial(j, Pt_coords + i * n);
		}
	}
	FREE_int(Pt_coords);


	if (f_v) {
		cout << "surface_domain::create_system "
				"The system has been created" << endl;
	}
	if (f_v && FALSE) {
		cout << "surface_domain::create_system "
				"The system:" << endl;
		Orbiter->Int_vec->matrix_print(System, nb_rows, nb_monomials);
	}

	if (f_v) {
		cout << "surface_domain::create_system done" << endl;
	}
}

void surface_domain::compute_intersection_points(int *Adj,
	long int *Lines, int nb_lines,
	long int *&Intersection_pt,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j1, j2, a1, a2, pt;

	if (f_v) {
		cout << "surface_domain::compute_intersection_points" << endl;
	}
	Intersection_pt = NEW_lint(nb_lines * nb_lines);
	Orbiter->Lint_vec->mone(Intersection_pt, nb_lines * nb_lines);
	for (j1 = 0; j1 < nb_lines; j1++) {
		a1 = Lines[j1];
		for (j2 = j1 + 1; j2 < nb_lines; j2++) {
			a2 = Lines[j2];
			if (Adj[j1 * nb_lines + j2]) {
				pt = P->intersection_of_two_lines(a1, a2);
				Intersection_pt[j1 * nb_lines + j2] = pt;
				Intersection_pt[j2 * nb_lines + j1] = pt;
			}
		}
	}
	if (f_v) {
		cout << "surface_domain::compute_intersection_points done" << endl;
	}
}

void surface_domain::compute_intersection_points_and_indices(int *Adj,
	long int *Points, int nb_points,
	long int *Lines, int nb_lines,
	int *&Intersection_pt, int *&Intersection_pt_idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j1, j2;
	long int a1, a2, pt;
	int idx;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "surface_domain::compute_intersection_points_and_indices" << endl;
	}
	Intersection_pt = NEW_int(nb_lines * nb_lines);
	Intersection_pt_idx = NEW_int(nb_lines * nb_lines);
	Orbiter->Int_vec->mone(Intersection_pt, nb_lines * nb_lines);
	for (j1 = 0; j1 < nb_lines; j1++) {
		a1 = Lines[j1];
		for (j2 = j1 + 1; j2 < nb_lines; j2++) {
			a2 = Lines[j2];
			if (Adj[j1 * nb_lines + j2]) {
				pt = P->intersection_of_two_lines(a1, a2);
				if (!Sorting.lint_vec_search(Points, nb_points,
					pt, idx, 0)) {
					cout << "surface_domain::compute_intersection_points_and_indices "
							"cannot find point in Points" << endl;
					cout << "Points:";
					Orbiter->Lint_vec->print_fully(cout, Points, nb_points);
					cout << endl;
					cout << "j1=" << j1 << endl;
					cout << "j2=" << j2 << endl;
					cout << "a1=" << a1 << endl;
					cout << "a2=" << a2 << endl;
					cout << "pt=" << pt << endl;
					exit(1);
				}
				Intersection_pt[j1 * nb_lines + j2] = pt;
				Intersection_pt[j2 * nb_lines + j1] = pt;
				Intersection_pt_idx[j1 * nb_lines + j2] = idx;
				Intersection_pt_idx[j2 * nb_lines + j1] = idx;
			}
		}
	}
	if (f_v) {
		cout << "surface_domain::compute_intersection_points_and_indices done" << endl;
	}
}

void surface_domain::lines_meet3_and_skew3(
	long int *lines_meet3, long int *lines_skew3,
	long int *&lines, int &nb_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int o_rank[6];
	int i, j;
	long int *perp;
	int perp_sz;

	if (f_v) {
		cout << "surface_domain::lines_meet3_and_skew3" << endl;
		cout << "The three lines we will meet are ";
		Orbiter->Lint_vec->print(cout, lines_meet3, 3);
		cout << endl;
		cout << "The three lines we will be skew to are ";
		Orbiter->Lint_vec->print(cout, lines_skew3, 3);
		cout << endl;
	}
	for (i = 0; i < 3; i++) {
		o_rank[i] = Klein->line_to_point_on_quadric(
				lines_meet3[i], 0 /* verbose_level*/);
	}
	for (i = 0; i < 3; i++) {
		o_rank[3 + i] = Klein->line_to_point_on_quadric(
				lines_skew3[i], 0 /* verbose_level*/);
	}

	O->perp_of_k_points(o_rank, 3, perp, perp_sz, verbose_level);

	lines = NEW_lint(perp_sz);
	nb_lines = 0;
	for (i = 0; i < perp_sz; i++) {
		for (j = 0; j < 3; j++) {
			if (O->evaluate_bilinear_form_by_rank(perp[i],
				o_rank[3 + j]) == 0) {
				break;
			}
		}
		if (j == 3) {
			lines[nb_lines++] = perp[i];
		}
	}

	FREE_lint(perp);

	for (i = 0; i < nb_lines; i++) {
		lines[i] = Klein->point_on_quadric_to_line(lines[i], 0 /* verbose_level*/);
	}

	if (f_v) {
		cout << "surface_domain::lines_meet3_and_skew3 done" << endl;
	}
}

void surface_domain::perp_of_three_lines(long int *three_lines,
		long int *&perp, int &perp_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int o_rank[3];
	int i;

	if (f_v) {
		cout << "surface_domain::perp_of_three_lines" << endl;
		cout << "The three lines are ";
		Orbiter->Lint_vec->print(cout, three_lines, 3);
		cout << endl;
	}
	for (i = 0; i < 3; i++) {
		o_rank[i] = Klein->line_to_point_on_quadric(
				three_lines[i], 0 /* verbose_level*/);
	}
	O->perp_of_k_points(o_rank, 3, perp, perp_sz, verbose_level);

	for (i = 0; i < perp_sz; i++) {
		perp[i] = Klein->point_on_quadric_to_line(
				perp[i], 0 /* verbose_level*/);
	}

	if (f_v) {
		cout << "surface_domain::perp_of_three_lines done" << endl;
	}
}

//! Given four general lines in four_lines[4], complete the two transversal lines.
/*!
 * Given four general lines in four_lines[4], complete the two transversal lines.
 * The function uses the perp on the Klein quadric.
 * Conversion to and from the Klein quadric is done automatically.
 */


int surface_domain::perp_of_four_lines(
		long int *four_lines, long int *trans12,
		int &perp_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int o_rank[4];
	int i;
	long int *Perp;
	int ret = TRUE;

	if (f_v) {
		cout << "surface_domain::perp_of_four_lines" << endl;
		cout << "The four lines are ";
		Orbiter->Lint_vec->print(cout, four_lines, 4);
		cout << endl;
	}
	for (i = 0; i < 4; i++) {
		o_rank[i] = Klein->line_to_point_on_quadric(
				four_lines[i], 0 /* verbose_level*/);
	}
	//Perp = NEW_int(O->alpha * (O->q + 1));
	O->perp_of_k_points(o_rank, 4, Perp, perp_sz, verbose_level);
	if (perp_sz != 2) {
		if (f_v) {
			cout << "perp_sz = " << perp_sz << " != 2" << endl;
		}
		ret = FALSE;
		goto finish;
	}

	trans12[0] = Klein->point_on_quadric_to_line(Perp[0], 0 /* verbose_level*/);
	trans12[1] = Klein->point_on_quadric_to_line(Perp[1], 0 /* verbose_level*/);

finish:
	FREE_lint(Perp);
	if (f_v) {
		cout << "surface_domain::perp_of_four_lines done" << endl;
	}
	return ret;
}

int surface_domain::rank_of_four_lines_on_Klein_quadric(
		long int *four_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int o_rank[4];
	int *coords;
	int i;
	long int rk;

	if (f_v) {
		cout << "surface_domain::rank_of_four_lines_on_Klein_quadric" << endl;
		cout << "The four lines are ";
		Orbiter->Lint_vec->print(cout, four_lines, 4);
		cout << endl;
	}
	for (i = 0; i < 4; i++) {
		o_rank[i] = Klein->line_to_point_on_quadric(
				four_lines[i], 0 /* verbose_level*/);
	}

	coords = NEW_int(4 * 6);
	for (i = 0; i < 4; i++) {
		O->unrank_point(coords + i * 6, 1,
			o_rank[i], 0 /* verbose_level */);
	}
	rk = F->Linear_algebra->Gauss_easy(coords, 4, 6);
	FREE_int(coords);
	if (f_v) {
		cout << "surface_domain::rank_of_four_lines_on_Klein_quadric done" << endl;
	}
	return rk;
}

//! Given a five-plus-one five_pts[5], complete the double-six.
/*!
 * Given a five-plus-one five_pts[5], complete the double-six. We assume that the
 * transversal line of the five lines is the line whose rank is 0.
 */

int surface_domain::create_double_six_from_five_lines_with_a_common_transversal(
	long int *five_pts, long int *double_six,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int o_rank[12];
	int i, j;
	int ret = TRUE;

	if (f_v) {
		cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal" << endl;
		cout << "The five lines are ";
		Orbiter->Lint_vec->print(cout, five_pts, 5);
		cout << endl;
	}
	for (i = 0; i < 5; i++) {
		o_rank[i] = Klein->line_to_point_on_quadric(
				five_pts[i], 0 /* verbose_level*/);
	}
	for (i = 0; i < 5; i++) {
		for (j = i + 1; j < 5; j++) {
			if (O->evaluate_bilinear_form_by_rank(
					o_rank[i], o_rank[j]) == 0) {
				cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal two of the given lines "
						"intersect, error" << endl;
				exit(1);
			}
		}
	}
	if (f_v) {
		cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal" << endl;
		cout << "The five lines as orthogonal points are ";
		Orbiter->Lint_vec->print(cout, o_rank, 5);
		cout << endl;
	}

	int nb_subsets;
	int subset[4];
	long int pts[4];
	int rk;
	long int **Perp;
	int *Perp_sz;
	long int lines[2];
	long int opposites[5];
	long int transversal = 0;
	combinatorics::combinatorics_domain Combi;

	nb_subsets = Combi.int_n_choose_k(5, 4);
	Perp = NEW_plint(nb_subsets);
	Perp_sz = NEW_int(nb_subsets);
	if (f_v) {
		cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal computing perp of 4-subsets" << endl;
	}
	for (rk = 0; rk < nb_subsets; rk++) {
		Combi.unrank_k_subset(rk, subset, 5, 4);
		for (i = 0; i < 4; i++) {
			pts[i] = o_rank[subset[i]];
		}

		if (f_v) {
			cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal subset " << rk << " / " << nb_subsets << " : " << endl;
		}
		O->perp_of_k_points(pts, 4,
			Perp[rk], Perp_sz[rk], 0/*verbose_level - 1*/);
		if (FALSE) {
			cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal the perp of the subset ";
			Orbiter->Int_vec->print(cout, subset, 4);
			cout << " has size " << Perp_sz[rk] << " : ";
			Orbiter->Lint_vec->print(cout, Perp[rk], Perp_sz[rk]);
			cout << endl;
		}
		if (Perp_sz[rk] != 2) {
			cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal "
					"Perp_opp_sz != 2, something is wrong" << endl;
			cout << "subset " << rk << " / " << nb_subsets << endl;
			exit(1);
			ret = FALSE;
			nb_subsets = rk + 1;
			goto finish;
		}
		if (rk == 0) {
			Orbiter->Lint_vec->copy(Perp[rk], lines, 2);
		}
		else if (rk == 1) {
			if (lines[0] == Perp[rk][0]) {
				transversal = lines[0];
				opposites[0] = lines[1];
				opposites[1] = Perp[rk][1];
			}
			else if (lines[0] == Perp[rk][1]) {
				transversal = lines[0];
				opposites[0] = lines[1];
				opposites[1] = Perp[rk][0];
			}
			else if (lines[1] == Perp[rk][0]) {
				transversal = lines[1];
				opposites[0] = lines[0];
				opposites[1] = Perp[rk][1];
			}
			else if (lines[1] == Perp[rk][1]) {
				transversal = lines[1];
				opposites[0] = lines[0];
				opposites[1] = Perp[rk][0];
			}
		}
		else {
			if (transversal == Perp[rk][0]) {
				opposites[rk] = Perp[rk][1];
			}
			else {
				opposites[rk] = Perp[rk][0];
			}
		}
	}
	if (FALSE) {
		cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal" << endl;
		cout << "opposites ";
		Orbiter->Lint_vec->print(cout, opposites, 5);
		cout << endl;
	}

	o_rank[11] = transversal;
	for (i = 0; i < 5; i++) {
		o_rank[10 - i] = opposites[i];
	}

	long int *Perp_opp;
	int Perp_opp_sz;
	long int transversal_opp;

	//Perp_opp = NEW_int(O->alpha * (O->q + 1));
	if (f_v) {
		cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal before O->perp_of_k_points" << endl;
	}
	O->perp_of_k_points(opposites, 4, Perp_opp, Perp_opp_sz,
			0/*verbose_level - 1*/);
	if (f_v) {
		cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal after O->perp_of_k_points" << endl;
	}
	if (FALSE) {
		cout << "the perp of the opposite subset ";
		Orbiter->Lint_vec->print(cout, opposites, 4);
		cout << " has size " << Perp_opp_sz << ":";
		Orbiter->Lint_vec->print(cout, Perp_opp, Perp_opp_sz);
		cout << endl;
	}
	if (Perp_opp_sz != 2) {
		ret = FALSE;
		cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal Perp_opp_sz != 2, "
				"something is wrong" << endl;
		exit(1);
		FREE_lint(Perp_opp);
		goto finish;
	}

	transversal_opp = -1;
	if (Perp_opp[0] == o_rank[0]) {
		transversal_opp = Perp_opp[1];
	}
	else if (Perp_opp[1] == o_rank[0]) {
		transversal_opp = Perp_opp[0];
	}
	else {
		cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal something is wrong "
				"with Perp_opp" << endl;
		exit(1);
	}

	o_rank[5] = transversal_opp;

	if (f_v) {
		cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal" << endl;
		cout << "o_rank ";
		Orbiter->Lint_vec->print(cout, o_rank, 12);
		cout << endl;
	}

	for (i = 0; i < 12; i++) {
		double_six[i] = Klein->point_on_quadric_to_line(o_rank[i], 0 /* verbose_level*/);
	}
	if (f_v) {
		cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal" << endl;
		cout << "double_six ";
		Orbiter->Lint_vec->print(cout, double_six, 12);
		cout << endl;
	}



	if (f_v) {
		for (i = 0; i < 12; i++) {
			for (j = 0; j < 12; j++) {
				if (O->evaluate_bilinear_form_by_rank(
					o_rank[i], o_rank[j]) == 0) {
					cout << "1";
				}
				else {
					cout << "0";
				}
			}
			cout << endl;
		}
	}


	FREE_lint(Perp_opp);

finish:
	for (i = 0; i < nb_subsets; i++) {
		FREE_lint(Perp[i]);
	}
	FREE_plint(Perp);
	FREE_int(Perp_sz);

	if (f_v) {
		cout << "surface_domain::create_double_six_from_five_lines_with_a_common_transversal done" << endl;
	}
	return ret;
}

//! Given a single six in single_six[6], compute the other 6 lines in a double-six.
/*!
 * Given a single six in single_six[6], compute the other 6 lines in a double-six.
 * The function uses the perp function on the Klein quadric.
 */


int surface_domain::create_double_six_from_six_disjoint_lines(
		long int *single_six,
		long int *double_six, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int o_rank[12];
	int i, j;
	int ret = FALSE;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "surface_domain::create_double_six_from_six_disjoint_lines" << endl;
	}
	for (i = 0; i < 6; i++) {
		o_rank[i] = Klein->line_to_point_on_quadric(single_six[i], 0 /* verbose_level*/);
	}

	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++) {
			if (O->evaluate_bilinear_form_by_rank(
				o_rank[i], o_rank[j]) == 0) {
				cout << "two of the given lines intersect, error" << endl;
				exit(1);
			}
		}
	}


	// compute the perp on the Klein quadric of each of the 6 given lines:
	long int **Perp_without_pt;
	int perp_sz = 0;
	int sz;

	sz = O->alpha * q;
	Perp_without_pt = NEW_plint(6);
	for (i = 0; i < 6; i++) {
		Perp_without_pt[i] = NEW_lint(sz);
		O->perp(o_rank[i], Perp_without_pt[i], perp_sz,
			0 /* verbose_level */);
		if (perp_sz != sz) {
			cout << "perp_sz != sz" << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "perp_sz=" << perp_sz << endl;
		for (i = 0; i < 6; i++) {
			Orbiter->Lint_vec->print(cout, Perp_without_pt[i], perp_sz);
			cout << endl;
		}
	}


	// compute the intersection of all perps, five at a time:

	long int **I2;
	int *I2_sz;
	long int **I3;
	int *I3_sz;
	long int **I4;
	int *I4_sz;
	long int **I5;
	int *I5_sz;
	long int six2, six3, six4, six5, rk, rk2;
	int subset[6];

	six2 = Combi.int_n_choose_k(6, 2);
	I2 = NEW_plint(six2);
	I2_sz = NEW_int(six2);
	for (rk = 0; rk < six2; rk++) {
		Combi.unrank_k_subset(rk, subset, 6, 2);
		Sorting.vec_intersect(
			Perp_without_pt[subset[0]],
			perp_sz,
			Perp_without_pt[subset[1]],
			perp_sz,
			I2[rk], I2_sz[rk]);
		if (f_v) {
			cout << "Perp_" << subset[0] << " \\cap Perp_" << subset[1]
				<< " of size " << I2_sz[rk] << " = ";
			Orbiter->Lint_vec->print(cout, I2[rk], I2_sz[rk]);
			cout << endl;
		}
	}
	six3 = Combi.int_n_choose_k(6, 3);
	I3 = NEW_plint(six3);
	I3_sz = NEW_int(six3);
	for (rk = 0; rk < six3; rk++) {
		Combi.unrank_k_subset(rk, subset, 6, 3);
		rk2 = Combi.rank_k_subset(subset, 6, 2);
		Combi.unrank_k_subset(rk, subset, 6, 3);
		Sorting.vec_intersect(I2[rk2], I2_sz[rk2],
			Perp_without_pt[subset[2]],
			perp_sz,
			I3[rk], I3_sz[rk]);
		if (f_v) {
			cout << "Perp_" << subset[0] << " \\cap Perp_" << subset[1]
				<< " \\cap Perp_" << subset[2] << " of size "
				<< I3_sz[rk] << " = ";
			Orbiter->Lint_vec->print(cout, I3[rk], I3_sz[rk]);
			cout << endl;
		}
	}

	six4 = Combi.int_n_choose_k(6, 4);
	I4 = NEW_plint(six4);
	I4_sz = NEW_int(six4);
	for (rk = 0; rk < six4; rk++) {
		Combi.unrank_k_subset(rk, subset, 6, 4);
		rk2 = Combi.rank_k_subset(subset, 6, 3);
		Combi.unrank_k_subset(rk, subset, 6, 4);
		Sorting.vec_intersect(I3[rk2], I3_sz[rk2],
			Perp_without_pt[subset[3]], perp_sz,
			I4[rk], I4_sz[rk]);
		if (f_v) {
			cout << rk << " / " << six4 << " : Perp_" << subset[0]
				<< " \\cap Perp_" << subset[1] << " \\cap Perp_"
				<< subset[2] << " \\cap Perp_" << subset[3]
				<< " of size " << I4_sz[rk] << " = ";
			Orbiter->Lint_vec->print(cout, I4[rk], I4_sz[rk]);
			cout << endl;
		}
	}

	six5 = Combi.int_n_choose_k(6, 5);
	I5 = NEW_plint(six5);
	I5_sz = NEW_int(six5);
	for (rk = 0; rk < six5; rk++) {
		Combi.unrank_k_subset(rk, subset, 6, 5);
		rk2 = Combi.rank_k_subset(subset, 6, 4);
		Combi.unrank_k_subset(rk, subset, 6, 5);
		Sorting.vec_intersect(I4[rk2], I4_sz[rk2],
			Perp_without_pt[subset[4]], perp_sz,
			I5[rk], I5_sz[rk]);
		if (f_v) {
			cout << rk << " / " << six5 << " : Perp_" << subset[0]
				<< " \\cap Perp_" << subset[1] << " \\cap Perp_"
				<< subset[2] << " \\cap Perp_" << subset[3]
				<< " \\cap Perp_" << subset[4] << " of size "
				<< I5_sz[rk] << " = ";
			Orbiter->Lint_vec->print(cout, I5[rk], I5_sz[rk]);
			cout << endl;
		}

		if (I5_sz[rk] != 1) {
			cout << "surface_domain::create_double_six I5_sz[rk] != 1" << endl;
			ret = FALSE;
			goto free_it;
		}
	}
	for (i = 0; i < 6; i++) {
		o_rank[6 + i] = I5[6 - 1 - i][0];
	}
	for (i = 0; i < 12; i++) {
		double_six[i] = Klein->point_on_quadric_to_line(o_rank[i], 0 /* verbose_level*/);
	}

	ret = TRUE;
free_it:
	for (i = 0; i < 6; i++) {
		FREE_lint(Perp_without_pt[i]);
	}
	FREE_plint(Perp_without_pt);
	//free_pint_all(Perp_without_pt, 6);



	for (i = 0; i < six2; i++) {
		FREE_lint(I2[i]);
	}
	FREE_plint(I2);
	//free_pint_all(I2, six2);

	FREE_int(I2_sz);


	for (i = 0; i < six3; i++) {
		FREE_lint(I3[i]);
	}
	FREE_plint(I3);
	//free_pint_all(I3, six3);

	FREE_int(I3_sz);


	for (i = 0; i < six4; i++) {
		FREE_lint(I4[i]);
	}
	FREE_plint(I4);

	//free_pint_all(I4, six4);

	FREE_int(I4_sz);


	for (i = 0; i < six5; i++) {
		FREE_lint(I5[i]);
	}
	FREE_plint(I5);

	//free_pint_all(I5, six5);

	FREE_int(I5_sz);


	if (f_v) {
		cout << "surface_domain::create_double_six_from_six_disjoint_lines done" << endl;
	}
	return ret;
}

//! Given a double six in double_six[12], compute the 15 remaining lines cij.
/*!
 * Given a double six in double_six[12], compute the 15 remaining lines cij.
 * The function uses Linear Algebra.
 */


void surface_domain::create_the_fifteen_other_lines(
	long int *double_six,
	long int *fifteen_other_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_the_fifteen_other_lines" << endl;
	}
	long int *Planes;
	long int *Lines;
	int h, k3;
	int i, j;

	Planes = NEW_lint(30);
	if (f_v) {
		cout << "creating the 30 planes:" << endl;
	}
	for (h = 0; h < 30; h++) {
		i = Schlaefli->Labels->Sets[h * 2 + 0];
		j = Schlaefli->Labels->Sets[h * 2 + 1];
		Gr->unrank_lint_here(Basis0, double_six[i],
				0/* verbose_level*/);
		Gr->unrank_lint_here(Basis0 + 8, double_six[j],
				0/* verbose_level*/);
		if (F->Linear_algebra->Gauss_easy(Basis0, 4, 4) != 3) {
			cout << "the rank is not 3" << endl;
			exit(1);
		}
		Planes[h] = Gr3->rank_lint_here(Basis0,
				0/* verbose_level*/);
		if (f_v) {
			cout << "plane " << h << " / " << 30
				<< " has rank " << Planes[h] << " and basis ";
			Orbiter->Int_vec->print(cout, Basis0, 12);
			cout << endl;
		}
	}
	Lines = NEW_lint(15);
	if (f_v) {
		cout << "creating the 15 lines:" << endl;
	}
	for (h = 0; h < 15; h++) {
		i = Schlaefli->Labels->Sets2[h * 2 + 0];
		j = Schlaefli->Labels->Sets2[h * 2 + 1];
		Gr3->unrank_lint_here(Basis1, Planes[i],
				0/* verbose_level*/);
		Gr3->unrank_lint_here(Basis2, Planes[j],
				0/* verbose_level*/);
		F->Linear_algebra->intersect_subspaces(4, 3, Basis1, 3, Basis2,
			k3, Basis0, 0 /* verbose_level */);
		if (k3 != 2) {
			cout << "the rank is not 2" << endl;
			exit(1);
		}
		Lines[h] = Gr->rank_lint_here(Basis0,
				0/* verbose_level*/);
		for (i = 0; i < 2; i++) {
			F->PG_element_normalize_from_front(
				Basis0 + i * 4, 1, 4);
		}
		if (f_v) {
			cout << "line " << h << " / " << 15
				<< " has rank " << Lines[h]
				<< " and basis ";
			Orbiter->Int_vec->print(cout, Basis0, 8);
			cout << endl;
		}
	}

	Orbiter->Lint_vec->copy(Lines, fifteen_other_lines, 15);


	FREE_lint(Planes);
	FREE_lint(Lines);

	if (f_v) {
		cout << "surface_domain::create_the_fifteen_other_lines done" << endl;
	}

}


//! Given a set of lines in S12[12], test the double six property.
/*!
 * Given a set of lines in S12[12], test the double six property.
 * The function first computes the adjacency matrix of the line intersection graph.
 */

int surface_domain::test_double_six_property(long int *S12, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int expect;
	int f_fail = FALSE;

	if (f_v) {
		cout << "surface_domain::test_double_six_property" << endl;
	}

	int *Adj;

	compute_adjacency_matrix_of_line_intersection_graph(
		Adj, S12, 12, 0 /*verbose_level*/);


	for (i = 0; i < 12; i++) {
		for (j = i + 1; j < 12; j++) {
			if (i < 6 && j < 6) {
				expect = 0;
			}
			else if (i >= 6 && j >= 6) {
				expect = 0;
			}
			else if (i < 6 && j >= 6) {
				if (i == j - 6) {
					expect = 0;
				}
				else {
					expect = 1;
				}
			}
			if (Adj[i * 12 + j] != expect) {
				cout << "surface_domain::test_double_six_property double six property is "
						"violated for " << Schlaefli->Labels->Line_label[i] << " and " << Schlaefli->Labels->Line_label[j] << endl;
				f_fail = TRUE;
			}
		}
	}

	FREE_int(Adj);

	if (f_v) {
		cout << "surface_domain::test_double_six_property done" << endl;
	}
	if (f_fail) {
		return FALSE;
	}
	else {
		return TRUE;
	}
}

//! Given a set of lines in S[n], compute the associated line intersection graph
/*!
 * Given a set of lines in S[n], compute the associated intersection graph.
 * The function uses the Klein quadric and the associated bilinear form.
 * The lines are first converted to points on the Klein quadric using Klein.
 * Then, the bilinear form is evaluated for all pairs.
 */

void surface_domain::compute_adjacency_matrix_of_line_intersection_graph(
	int *&Adj, long int *S, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int *o_rank;

	if (f_v) {
		cout << "surface_domain::compute_adjacency_matrix_of_line_intersection_graph" << endl;
	}

#if 0
	if (n > 27) {
		cout << "surface_domain::compute_adjacency_matrix_of_line_intersection_graph n > 27" << endl;
		exit(1);
	}
#endif

	o_rank = NEW_lint(n);
	for (i = 0; i < n; i++) {
		o_rank[i] = Klein->line_to_point_on_quadric(S[i], 0 /* verbose_level*/);
	}

	Adj = NEW_int(n * n);
	Orbiter->Int_vec->zero(Adj, n * n);
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (O->evaluate_bilinear_form_by_rank(
				o_rank[i], o_rank[j]) == 0) {
				Adj[i * n + j] = 1;
				Adj[j * n + i] = 1;
			}
		}
	}
	FREE_lint(o_rank);
	if (f_v) {
		cout << "surface_domain::compute_adjacency_matrix_of_line_intersection_graph done" << endl;
	}
}

//! Given a set of lines in S[n], compute the associated disjointness graph
/*!
 * Given a set of lines in S[n], compute the associated disjointness graph.
 * The function uses the Klein quadric and the associated bilinear form.
 * The lines are first converted to points on the Klein quadric using Klein.
 * Then, the bilinear form is evaluated for all pairs.
 */
void surface_domain::compute_adjacency_matrix_of_line_disjointness_graph(
	int *&Adj, long int *S, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int *o_rank;

	if (f_v) {
		cout << "surface_domain::compute_adjacency_matrix_of_line_disjointness_graph" << endl;
	}

#if 0
	if (n > 27) {
		cout << "surface_domain::compute_adjacency_matrix_of_line_disjointness_graph n > 27" << endl;
		exit(1);
	}
#endif

	o_rank = NEW_lint(n);
	for (i = 0; i < n; i++) {
		o_rank[i] = Klein->line_to_point_on_quadric(S[i], 0 /* verbose_level*/);
	}

	Adj = NEW_int(n * n);
	Orbiter->Int_vec->zero(Adj, n * n);
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (O->evaluate_bilinear_form_by_rank(
				o_rank[i], o_rank[j]) != 0) {
				Adj[i * n + j] = 1;
				Adj[j * n + i] = 1;
			}
		}
	}
	FREE_lint(o_rank);
	if (f_v) {
		cout << "surface_domain::compute_adjacency_matrix_of_line_disjointness_graph done" << endl;
	}
}

void surface_domain::compute_points_on_lines(
		long int *Pts_on_surface, int nb_points_on_surface,
		long int *Lines, int nb_lines,
		data_structures::set_of_sets *&pts_on_lines,
		int *&f_is_on_line,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, l, r;
	int *Surf_pt_coords;
	int Basis[8];
	int Mtx[12];

	if (f_v) {
		cout << "surface_domain::compute_points_on_lines" << endl;
	}
	f_is_on_line = NEW_int(nb_points_on_surface);
	Orbiter->Int_vec->zero(f_is_on_line, nb_points_on_surface);

	pts_on_lines = NEW_OBJECT(data_structures::set_of_sets);
	pts_on_lines->init_basic_constant_size(nb_points_on_surface,
		nb_lines, q + 1, 0 /* verbose_level */);
	Surf_pt_coords = NEW_int(nb_points_on_surface * 4);
	for (i = 0; i < nb_points_on_surface; i++) {
		P->unrank_point(Surf_pt_coords + i * 4, Pts_on_surface[i]);
	}

	Orbiter->Lint_vec->zero(pts_on_lines->Set_size, nb_lines);
	for (i = 0; i < nb_lines; i++) {
		l = Lines[i];
		P->unrank_line(Basis, l);
		//cout << "Line " << i << " basis=";
		//int_vec_print(cout, Basis, 8);
		//cout << " : ";
		for (j = 0; j < nb_points_on_surface; j++) {
			Orbiter->Int_vec->copy(Basis, Mtx, 8);
			Orbiter->Int_vec->copy(Surf_pt_coords + j * 4, Mtx + 8, 4);
			r = F->Linear_algebra->Gauss_easy(Mtx, 3, 4);
			if (r == 2) {
				pts_on_lines->add_element(i, j);
				f_is_on_line[j] = TRUE;
				//cout << j << " ";
			}
		}
		//cout << endl;
	}
	//cout << "the surface points on the set of " << nb_lines
	//<< " lines are:" << endl;
	//pts_on_lines->print_table();

	FREE_int(Surf_pt_coords);
	if (f_v) {
		cout << "surface_domain::compute_points_on_lines done" << endl;
	}
}

int surface_domain::compute_rank_of_any_four(
		long int *&Rk, int &nb_subsets,
		long int *lines, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int subset[4];
	long int four_lines[4];
	int i, rk;
	int ret = TRUE;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "surface_domain::compute_rank_of_any_four" << endl;
	}
	nb_subsets = Combi.int_n_choose_k(sz, 4);
	Rk = NEW_lint(nb_subsets);
	for (rk = 0; rk < nb_subsets; rk++) {
		Combi.unrank_k_subset(rk, subset, sz, 4);
		for (i = 0; i < 4; i++) {
			four_lines[i] = lines[subset[i]];
		}

		if (f_v) {
			cout << "testing subset " << rk << " / "
				<< nb_subsets << " : " << endl;
		}

		Rk[rk] = rank_of_four_lines_on_Klein_quadric(
			four_lines, 0 /* verbose_level */);
		if (Rk[rk] < 4) {
			ret = FALSE;
		}
	}
	if (f_v) {
		cout << "Ranks:" << endl;
		Orbiter->Lint_vec->print(cout, Rk, nb_subsets);
		cout << endl;
	}
	if (f_v) {
		cout << "surface_domain::compute_rank_of_any_four done" << endl;
	}
	return ret;
}

void surface_domain::rearrange_lines_according_to_a_given_double_six(long int *Lines,
		int *given_double_six,
		long int *New_lines,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Adj;
	int nb_lines = 27;
	int i, j, h;

	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_to_a_given_double_six" << endl;
	}
	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_to_a_given_double_six "
			"before compute_adjacency_matrix_of_line_intersection_graph" << endl;
	}
	compute_adjacency_matrix_of_line_intersection_graph(Adj,
		Lines, nb_lines, 0 /* verbose_level */);

	for (i = 0; i < 12; i++) {
		New_lines[i] = Lines[given_double_six[i]];
	}


	h = 0;
	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++, h++) {
			New_lines[12 + h] = compute_cij(New_lines /* double_six */, i, j,
				0 /* verbose_level */);
		}
	}
	if (f_v) {
		cout << "New_lines:";
		Orbiter->Lint_vec->print(cout, New_lines, 27);
		cout << endl;
	}



	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_to_a_given_double_six done" << endl;
	}
}

//! Picks a double six and rearranges the lines accordingly
/*!
 * Given 27 lines in Lines[27], compute any double six and rearrange
 * the lines accordingly.
 */
void surface_domain::rearrange_lines_according_to_double_six(long int *Lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Adj;
	int nb_lines = 27;
	long int New_lines[27];

	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_to_double_six" << endl;
	}
	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_to_double_six "
			"before compute_adjacency_matrix_of_line_"
			"intersection_graph" << endl;
	}
	compute_adjacency_matrix_of_line_intersection_graph(Adj,
		Lines, nb_lines, 0 /* verbose_level */);


	data_structures::set_of_sets *line_intersections;
	int *Starter_Table;
	int nb_starter;

	line_intersections = NEW_OBJECT(data_structures::set_of_sets);

	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_to_double_six "
			"before line_intersections->init_from_adjacency_matrix"
			<< endl;
	}
	line_intersections->init_from_adjacency_matrix(nb_lines, Adj,
		0 /* verbose_level */);

	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_to_double_six "
			"before list_starter_configurations" << endl;
	}
	list_starter_configurations(Lines, nb_lines,
		line_intersections, Starter_Table, nb_starter,
		0 /*verbose_level */);

	int l, line_idx, subset_idx;

	if (nb_starter == 0) {
		cout << "surface_domain::rearrange_lines_according_to_double_six "
				"nb_starter == 0" << endl;
		exit(1);
	}
	l = 0;
	line_idx = Starter_Table[l * 2 + 0];
	subset_idx = Starter_Table[l * 2 + 1];

	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_to_double_six "
			"before rearrange_lines_according_to_starter_"
			"configuration" << endl;
	}
	rearrange_lines_according_to_starter_configuration(
		Lines, New_lines,
		line_idx, subset_idx, Adj,
		line_intersections, 0 /*verbose_level*/);

	Orbiter->Lint_vec->copy(New_lines, Lines, 27);

	FREE_int(Adj);
	FREE_int(Starter_Table);
	FREE_OBJECT(line_intersections);
	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_"
				"to_double_six done" << endl;
	}
}

void surface_domain::rearrange_lines_according_to_starter_configuration(
	long int *Lines, long int *New_lines,
	int line_idx, int subset_idx, int *Adj,
	data_structures::set_of_sets *line_intersections,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int S3[6];
	int i, idx;
	int nb_lines = 27;
	data_structures::sorting Sorting;


	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_to_starter_configuration" << endl;
	}

	create_starter_configuration(line_idx, subset_idx,
		line_intersections, Lines, S3, 0 /* verbose_level */);


	if (f_v) {
		cout << "line_intersections:" << endl;
		line_intersections->print_table();
	}

	int Line_idx[27];
	for (i = 0; i < 6; i++) {
		if (!Sorting.lint_vec_search_linear(Lines, nb_lines, S3[i], idx)) {
			cout << "could not find the line" << endl;
			exit(1);
		}
		Line_idx[i] = idx;
	}

	if (f_v) {
		cout << "The 5+1 lines are ";
		Orbiter->Int_vec->print(cout, Line_idx, 6);
		cout << endl;
	}

	Line_idx[11] = Line_idx[5];
	Line_idx[5] = 0;
	Orbiter->Lint_vec->zero(New_lines, 27);
	Orbiter->Lint_vec->copy(S3, New_lines, 5);
	New_lines[11] = S3[5];

	if (f_v) {
		cout << "computing b_j:" << endl;
	}
	for (i = 0; i < 5; i++) {
		int four_lines[4];

		if (f_v) {
			cout << i << " / " << 5 << ":" << endl;
		}

		Orbiter->Int_vec->copy(Line_idx, four_lines, i);
		Orbiter->Int_vec->copy(Line_idx + i + 1, four_lines + i, 5 - i - 1);
		if (f_v) {
			cout << "four_lines=";
			Orbiter->Int_vec->print(cout, four_lines, 4);
			cout << endl;
		}

		Line_idx[6 + i] = intersection_of_four_lines_but_not_b6(
			Adj, four_lines, Line_idx[11], verbose_level);
		if (f_v) {
			cout << "b_" << i + 1 << " = "
				<< Line_idx[6 + i] << endl;
		}
	}

	int five_lines_idx[5];
	Orbiter->Int_vec->copy(Line_idx + 6, five_lines_idx, 5);
	Line_idx[5] = intersection_of_five_lines(Adj,
		five_lines_idx, verbose_level);
	if (f_v) {
		cout << "a_" << i + 1 << " = "
			<< Line_idx[5] << endl;
	}


	long int double_six[12];
	int h, j;

	for (i = 0; i < 12; i++) {
		double_six[i] = Lines[Line_idx[i]];
	}
	Orbiter->Lint_vec->copy(double_six, New_lines, 12);

	h = 0;
	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++, h++) {
			New_lines[12 + h] = compute_cij(
				double_six, i, j,
				0 /* verbose_level */);
		}
	}
	if (f_v) {
		cout << "New_lines:";
		Orbiter->Lint_vec->print(cout, New_lines, 27);
		cout << endl;
	}

	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_to_starter_configuration done" << endl;
	}
}

int surface_domain::intersection_of_four_lines_but_not_b6(int *Adj,
	int *four_lines_idx, int b6, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, i, j;

	if (f_v) {
		cout << "surface_domain::intersection_of_four_lines_but_not_b6" << endl;
	}
	for (i = 0; i < 27; i++) {
		if (i == b6) {
			continue;
		}
		for (j = 0; j < 4; j++) {
			if (Adj[i * 27 + four_lines_idx[j]] == 0) {
				break;
			}
		}
		if (j == 4) {
			a = i;
			break;
		}
	}
	if (i == 27) {
		cout << "surface_domain::intersection_of_four_lines_but_not_b6 could not find the line" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "surface_domain::intersection_of_four_lines_but_not_b6 done" << endl;
	}
	return a;
}

int surface_domain::intersection_of_five_lines(int *Adj,
	int *five_lines_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, i, j;

	if (f_v) {
		cout << "surface_domain::intersection_of_five_lines" << endl;
	}
	for (i = 0; i < 27; i++) {
		for (j = 0; j < 5; j++) {
			if (Adj[i * 27 + five_lines_idx[j]] == 0) {
				break;
			}
		}
		if (j == 5) {
			a = i;
			break;
		}
	}
	if (i == 27) {
		cout << "surface_domain::intersection_of_five_lines "
				"could not find the line" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "surface_domain::intersection_of_five_lines done" << endl;
	}
	return a;
}

void surface_domain::rearrange_lines_according_to_a_given_double_six(
	long int *Lines,
	long int *New_lines, long int *double_six, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;


	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_to_a_given_double_six" << endl;
	}
	for (i = 0; i < 12; i++) {
		New_lines[i] = Lines[double_six[i]];
	}
	h = 0;
	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++, h++) {
			New_lines[12 + h] = compute_cij(
				New_lines /*double_six */, i, j, 0 /* verbose_level */);
		}
	}
	if (f_v) {
		cout << "New_lines:";
		Orbiter->Lint_vec->print(cout, New_lines, 27);
		cout << endl;
	}

	if (f_v) {
		cout << "surface_domain::rearrange_lines_according_to_a_given_double_six done" << endl;
	}
}

void surface_domain::create_lines_from_plane_equations(
	int *The_plane_equations,
	long int *Lines27, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int line_idx, plane1, plane2;
	int Basis[16];

	if (f_v) {
		cout << "surface_domain::create_lines_from_plane_equations" << endl;
	}

	for (line_idx = 0; line_idx < 27; line_idx++) {
		Schlaefli->find_tritangent_planes_intersecting_in_a_line(
			line_idx, plane1, plane2, 0 /* verbose_level */);
		Orbiter->Int_vec->copy(The_plane_equations + plane1 * 4, Basis, 4);
		Orbiter->Int_vec->copy(The_plane_equations + plane2 * 4, Basis + 4, 4);
		F->Linear_algebra->perp_standard(4, 2, Basis, 0 /* verbose_level */);
		Lines27[line_idx] = rank_line(Basis + 8);
	}

	if (f_v) {
		cout << "surface_domain::create_lines_from_plane_equations done" << endl;
	}
}



void surface_domain::create_remaining_fifteen_lines(
	long int *double_six, long int *fifteen_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, h;

	if (f_v) {
		cout << "surface_domain::create_remaining_fifteen_lines" << endl;
	}
	h = 0;
	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++) {
			if (f_vv) {
				cout << "surface_domain::create_remaining_fifteen_lines "
						"creating line c_ij where i=" << i
						<< " j=" << j << ":" << endl;
			}
			fifteen_lines[h++] = compute_cij(double_six, i, j, 0 /*verbose_level*/);
		}
	}
	if (f_v) {
		cout << "surface_domain::create_remaining_fifteen_lines done" << endl;
	}
}

//! Computes cij, given a double six
/*!
 * Given a double six in double_six[12], and i,j,
 * compute c_ij = a_ib_j intersect a_jb_i
 */
long int surface_domain::compute_cij(long int *double_six,
		int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int ai, aj, bi, bj;
	int Basis1[16];
	int Basis2[16];
	int K1[16];
	int K2[16];
	int K[16];
	int base_cols1[4];
	int base_cols2[4];
	int kernel_m, kernel_n;
	long int cij;

	if (f_v) {
		cout << "surface_domain::compute_cij" << endl;
	}
	ai = double_six[i];
	aj = double_six[j];
	bi = double_six[6 + i];
	bj = double_six[6 + j];
	Gr->unrank_lint_here(Basis1, ai, 0 /* verbose_level */);
	Gr->unrank_lint_here(Basis1 + 2 * 4, bj, 0 /* verbose_level */);
	Gr->unrank_lint_here(Basis2, aj, 0 /* verbose_level */);
	Gr->unrank_lint_here(Basis2 + 2 * 4, bi, 0 /* verbose_level */);
	if (F->Linear_algebra->Gauss_simple(Basis1, 4, 4, base_cols1,
			0 /* verbose_level */) != 3) {
		cout << "The rank of Basis1 is not 3" << endl;
		exit(1);
	}
	if (F->Linear_algebra->Gauss_simple(Basis2, 4, 4, base_cols2,
			0 /* verbose_level */) != 3) {
		cout << "The rank of Basis2 is not 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "surface_domain::compute_cij before matrix_get_"
				"kernel Basis1" << endl;
	}
	F->Linear_algebra->matrix_get_kernel(Basis1, 3, 4, base_cols1, 3,
		kernel_m, kernel_n, K1, 0 /* verbose_level */);
	if (kernel_m != 4) {
		cout << "surface_domain::compute_cij kernel_m != 4 "
				"when computing K1" << endl;
		exit(1);
	}
	if (kernel_n != 1) {
		cout << "surface_domain::compute_cij kernel_1 != 1 "
				"when computing K1" << endl;
		exit(1);
	}
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < 4; i++) {
			K[j * 4 + i] = K1[i * kernel_n + j];
		}
	}
	if (f_v) {
		cout << "surface_domain::compute_cij before matrix_"
				"get_kernel Basis2" << endl;
	}
	F->Linear_algebra->matrix_get_kernel(Basis2, 3, 4, base_cols2, 3,
		kernel_m, kernel_n, K2, 0 /* verbose_level */);
	if (kernel_m != 4) {
		cout << "surface_domain::compute_cij kernel_m != 4 "
				"when computing K2" << endl;
		exit(1);
	}
	if (kernel_n != 1) {
		cout << "surface_domain::compute_cij kernel_1 != 1 "
				"when computing K2" << endl;
		exit(1);
	}
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < 4; i++) {
			K[(1 + j) * 4 + i] = K2[i * kernel_n + j];
		}
	}
	if (F->Linear_algebra->Gauss_simple(K, 2, 4, base_cols1,
			0 /* verbose_level */) != 2) {
		cout << "The rank of K is not 2" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "surface_domain::compute_cij before "
				"matrix_get_kernel K" << endl;
	}
	F->Linear_algebra->matrix_get_kernel(K, 2, 4, base_cols1, 2,
		kernel_m, kernel_n, K1, 0 /* verbose_level */);
	if (kernel_m != 4) {
		cout << "surface_domain::compute_cij kernel_m != 4 "
				"when computing final kernel" << endl;
		exit(1);
	}
	if (kernel_n != 2) {
		cout << "surface_domain::compute_cij kernel_n != 2 "
				"when computing final kernel" << endl;
		exit(1);
	}
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < n; i++) {
			Basis1[j * n + i] = K1[i * kernel_n + j];
		}
	}
	cij = Gr->rank_lint_here(Basis1, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::compute_cij done" << endl;
	}
	return cij;
}

int surface_domain::compute_transversals_of_any_four(
		long int *&Trans, int &nb_subsets,
		long int *lines, int sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int trans12[2];
	int subset[4];
	long int four_lines[4];
	int i, rk, perp_sz;
	int ret = TRUE;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "surface_domain::compute_transversals_of_any_four" << endl;
	}
	nb_subsets = Combi.int_n_choose_k(sz, 4);
	Trans = NEW_lint(nb_subsets * 2);
	for (rk = 0; rk < nb_subsets; rk++) {
		Combi.unrank_k_subset(rk, subset, sz, 4);
		for (i = 0; i < 4; i++) {
			four_lines[i] = lines[subset[i]];
		}

		if (f_v) {
			cout << "testing subset " << rk << " / "
				<< nb_subsets << " : " << endl;
		}
		if (!perp_of_four_lines(four_lines, trans12,
			perp_sz, 0 /*verbose_level*/)) {

			if (f_v) {
				cout << "The 4-subset does not lead "
						"to two transversal lines: ";
				Orbiter->Int_vec->print(cout, subset, 4);
				cout << " = ";
				Orbiter->Lint_vec->print(cout, four_lines, 4);
				cout << " perp_sz=" << perp_sz << endl;
			}
			ret = FALSE;
			//break;
			trans12[0] = -1;
			trans12[1] = -1;
		}
		Orbiter->Lint_vec->copy(trans12, Trans + rk * 2, 2);
	}
	if (f_v) {
		cout << "Transversals:" << endl;
		Orbiter->Lint_vec->matrix_print(Trans, nb_subsets, 2);
	}
	if (f_v) {
		cout << "surface_domain::compute_transversals_of_any_four done" << endl;
	}
	return ret;
}

int surface_domain::read_schlaefli_label(const char *p)
{
	if (strcmp(p, "a1") == 0) {
		return 0;
	}
	else if (strcmp(p, "a2") == 0) {
		return 1;
	}
	else if (strcmp(p, "a3") == 0) {
		return 2;
	}
	else if (strcmp(p, "a4") == 0) {
		return 3;
	}
	else if (strcmp(p, "a5") == 0) {
		return 4;
	}
	else if (strcmp(p, "a6") == 0) {
		return 5;
	}
	else if (strcmp(p, "b1") == 0) {
		return 6;
	}
	else if (strcmp(p, "b2") == 0) {
		return 7;
	}
	else if (strcmp(p, "b3") == 0) {
		return 8;
	}
	else if (strcmp(p, "b4") == 0) {
		return 9;
	}
	else if (strcmp(p, "b5") == 0) {
		return 10;
	}
	else if (strcmp(p, "b6") == 0) {
		return 11;
	}
	else if (strcmp(p, "c12") == 0) {
		return 12;
	}
	else if (strcmp(p, "c13") == 0) {
		return 13;
	}
	else if (strcmp(p, "c14") == 0) {
		return 14;
	}
	else if (strcmp(p, "c15") == 0) {
		return 15;
	}
	else if (strcmp(p, "c16") == 0) {
		return 16;
	}
	else if (strcmp(p, "c23") == 0) {
		return 17;
	}
	else if (strcmp(p, "c24") == 0) {
		return 18;
	}
	else if (strcmp(p, "c25") == 0) {
		return 19;
	}
	else if (strcmp(p, "c26") == 0) {
		return 20;
	}
	else if (strcmp(p, "c34") == 0) {
		return 21;
	}
	else if (strcmp(p, "c35") == 0) {
		return 22;
	}
	else if (strcmp(p, "c36") == 0) {
		return 23;
	}
	else if (strcmp(p, "c45") == 0) {
		return 24;
	}
	else if (strcmp(p, "c46") == 0) {
		return 25;
	}
	else if (strcmp(p, "c56") == 0) {
		return 26;
	}
	else {
		cout << "surface_domain::read_schlaefli_label unknown schlaefli label: " << p << endl;
		exit(1);
	}
}

void surface_domain::read_string_of_schlaefli_labels(std::string &str, int *&v, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;
	char **argv;
	int i;

	if (f_v) {
		cout << "surface_domain::read_string_of_schlaefli_labels" << endl;
	}

	ST.chop_string_comma_separated(str.c_str(), sz, argv);

	if (f_v) {
		cout << "surface_domain::read_string_of_schlaefli_labels reading:" << endl;
		for (i = 0; i < sz; i++) {
			cout << i << " : " << argv[i] << endl;
		}
	}

	v = NEW_int(sz);
	for (i = 0; i < sz; i++) {
		v[i] = read_schlaefli_label(argv[i]);
	}


	if (f_v) {
		cout << "surface_domain::read_string_of_schlaefli_labels done" << endl;
	}
}

}}}


