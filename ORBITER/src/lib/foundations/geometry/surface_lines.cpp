// surface_lines.cpp
//
// Anton Betten
//
// moved here from surface.cpp: Dec 26, 2018
//
//
//
//

#include "foundations.h"

namespace orbiter {


void surface::init_line_data(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int i, j, h, h2;

	if (f_v) {
		cout << "surface::init_line_data" << endl;
		}

	Sets = NEW_int(30 * 2);
	M = NEW_int(6 * 6);
	int_vec_zero(M, 6 * 6);

	h = 0;
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (j == i) {
				continue;
				}
			M[i * 6 + j] = h;
			Sets[h * 2 + 0] = i;
			Sets[h * 2 + 1] = 6 + j;
			h++;
			}
		}


	if (h != 30) {
		cout << "h != 30" << endl;
		exit(1);
		}


	if (f_v) {
		cout << "surface::init_line_data Sets:" << endl;
		print_integer_matrix_with_standard_labels(cout,
			Sets, 30, 2, FALSE /* f_tex */);
		//int_matrix_print(Sets, 30, 2);
		}


	Sets2 = NEW_int(15 * 2);
	h2 = 0;
	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++) {
			Sets2[h2 * 2 + 0] = M[i * 6 + j];
			Sets2[h2 * 2 + 1] = M[j * 6 + i];
			h2++;
			}
		}
	if (h2 != 15) {
		cout << "h2 != 15" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "Sets2:" << endl;
		print_integer_matrix_with_standard_labels(cout,
			Sets2, 15, 2, FALSE /* f_tex */);
		//int_matrix_print(Sets2, 15, 2);
		}

	Line_label = NEW_pchar(27);
	Line_label_tex = NEW_pchar(27);
	char str[1000];
	int a, b, c, l;

	for (i = 0; i < 27; i++) {
		if (i < 6) {
			sprintf(str, "a_%d", i + 1);
			}
		else if (i < 12) {
			sprintf(str, "b_%d", i - 6 + 1);
			}
		else {
			h = i - 12;
			c = Sets2[h * 2 + 0];
			a = Sets[c * 2 + 0] + 1;
			b = Sets[c * 2 + 1] - 6 + 1;
			sprintf(str, "c_{%d%d}", a, b);
			}
		if (f_v) {
			cout << "creating label " << str
				<< " for line " << i << endl;
			}
		l = strlen(str);
		Line_label[i] = NEW_char(l + 1);
		strcpy(Line_label[i], str);
		}

	for (i = 0; i < 27; i++) {
		if (i < 6) {
			sprintf(str, "a_{%d}", i + 1);
			}
		else if (i < 12) {
			sprintf(str, "b_{%d}", i - 6 + 1);
			}
		else {
			h = i - 12;
			c = Sets2[h * 2 + 0];
			a = Sets[c * 2 + 0] + 1;
			b = Sets[c * 2 + 1] - 6 + 1;
			sprintf(str, "c_{%d%d}", a, b);
			}
		if (f_v) {
			cout << "creating label " << str
				<< " for line " << i << endl;
			}
		l = strlen(str);
		Line_label_tex[i] = NEW_char(l + 1);
		strcpy(Line_label_tex[i], str);
		}

	if (f_v) {
		cout << "surface::init_line_data done" << endl;
		}
}

int surface::line_ai(int i)
{
	if (i >= 6) {
		cout << "surface::line_ai i >= 6" << endl;
		exit(1);
		}
	return i;
}

int surface::line_bi(int i)
{
	if (i >= 6) {
		cout << "surface::line_bi i >= 6" << endl;
		exit(1);
		}
	return 6 + i;
}

int surface::line_cij(int i, int j)
{
	int a;

	if (i > j) {
		return line_cij(j, i);
		}
	if (i == j) {
		cout << "surface::line_cij i==j" << endl;
		exit(1);
		}
	if (i >= 6) {
		cout << "surface::line_cij i >= 6" << endl;
		exit(1);
		}
	if (j >= 6) {
		cout << "surface::line_cij j >= 6" << endl;
		exit(1);
		}
	a = ij2k(i, j, 6);
	return 12 + a;
}

int surface::type_of_line(int line)
// 0 = a_i, 1 = b_i, 2 = c_ij
{
	if (line < 6) {
		return 0;
		}
	else if (line < 12) {
		return 1;
		}
	else if (line < 27) {
		return 2;
		}
	else {
		cout << "surface::type_of_line error" << endl;
		exit(1);
		}
}

void surface::index_of_line(int line, int &i, int &j)
// returns i for a_i, i for b_i and (i,j) for c_ij
{
	int a;

	if (line < 6) { // ai
		i = line;
		}
	else if (line < 12) { // bj
		i = line - 6;
		}
	else if (line < 27) { // c_ij
		a = line - 12;
		k2ij(a, i, j, 6);
		}
	else {
		cout << "surface::index_of_line error" << endl;
		exit(1);
		}
}

void surface::unrank_line(int *v, int rk)
{
	Gr->unrank_int_here(v, rk, 0 /* verbose_level */);
}

void surface::unrank_lines(int *v, int *Rk, int nb)
{
	int i;

	for (i = 0; i < nb; i++) {
		Gr->unrank_int_here(v + i * 8, Rk[i], 0 /* verbose_level */);
		}
}

int surface::rank_line(int *v)
{
	int rk;

	rk = Gr->rank_int_here(v, 0 /* verbose_level */);
	return rk;
}

void surface::build_cubic_surface_from_lines(
	int len, int *S,
	int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int r;

	if (f_v) {
		cout << "surface::build_cubic_surface_from_lines" << endl;
		}
	r = compute_system_in_RREF(len, S, verbose_level);
	if (r != nb_monomials - 1) {
		cout << "surface::build_cubic_surface_from_lines "
				"r != nb_monomials - 1" << endl;
		cout << "r=" << r << endl;
		exit(1);
		}

	int kernel_m, kernel_n;

	F->matrix_get_kernel(System, r, nb_monomials, base_cols, r,
		kernel_m, kernel_n, coeff);

	//cout << "kernel_m=" << kernel_m << endl;
	//cout << "kernel_n=" << kernel_n << endl;
	if (f_v) {
		cout << "surface::build_cubic_surface_from_lines done" << endl;
		}
}

int surface::compute_system_in_RREF(
		int len, int *S, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int i, j, nb_pts, a, r;

	if (f_v) {
		cout << "surface::compute_system_in_RREF" << endl;
		}
	if (len > 27) {
		cout << "surface::compute_system_in_RREF len > 27" << endl;
		exit(1);
		}

	nb_pts = 0;
	for (i = 0; i < len; i++) {
		a = S[i];

		if (P->Lines) {
			for (j = 0; j < P->k; j++) {
				pt_list[nb_pts++] = P->Lines[a * P->k + j];
				}
			}
		else {
			P->create_points_on_line(a,
				pt_list + nb_pts,
				0 /* verbose_level */);
			nb_pts += P->k;
			}
		}

	if (nb_pts > max_pts) {
		cout << "surface::compute_system_in_RREF "
				"nb_pts > max_pts" << endl;
		exit(1);
		}
	if (FALSE) {
		cout << "surface::compute_system_in_RREF list of "
				"covered points by lines:" << endl;
		int_matrix_print(pt_list, len, P->k);
		}
	for (i = 0; i < nb_pts; i++) {
		unrank_point(Pts + i * n, pt_list[i]);
		}
	if (f_v && FALSE) {
		cout << "surface::compute_system_in_RREF list of "
				"covered points in coordinates:" << endl;
		int_matrix_print(Pts, nb_pts, n);
		}

	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < nb_monomials; j++) {
			System[i * nb_monomials + j] =
				F->evaluate_monomial(
					Poly3_4->Monomials + j * n,
					Pts + i * n, n);
			}
		}
	if (f_v && FALSE) {
		cout << "surface::compute_system_in_RREF "
				"The system:" << endl;
		int_matrix_print(System, nb_pts, nb_monomials);
		}
	r = F->Gauss_simple(System, nb_pts, nb_monomials,
		base_cols, 0 /* verbose_level */);
	if (FALSE) {
		cout << "surface::compute_system_in_RREF "
				"The system in RREF:" << endl;
		int_matrix_print(System, nb_pts, nb_monomials);
		}
	if (f_v) {
		cout << "surface::compute_system_in_RREF "
				"The system has rank " << r << endl;
		}
	return r;
}

void surface::compute_intersection_points(int *Adj,
	int *Lines, int nb_lines,
	int *&Intersection_pt,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j1, j2, a1, a2, pt;

	if (f_v) {
		cout << "surface::compute_intersection_points" << endl;
		}
	Intersection_pt = NEW_int(nb_lines * nb_lines);
	int_vec_mone(Intersection_pt, nb_lines * nb_lines);
	for (j1 = 0; j1 < nb_lines; j1++) {
		a1 = Lines[j1];
		for (j2 = j1 + 1; j2 < nb_lines; j2++) {
			a2 = Lines[j2];
			if (Adj[j1 * nb_lines + j2]) {
				pt = P->line_intersection(a1, a2);
				Intersection_pt[j1 * nb_lines + j2] = pt;
				Intersection_pt[j2 * nb_lines + j1] = pt;
				}
			}
		}
	if (f_v) {
		cout << "surface::compute_intersection_points done" << endl;
		}
}

void surface::compute_intersection_points_and_indices(int *Adj,
	int *Points, int nb_points,
	int *Lines, int nb_lines,
	int *&Intersection_pt, int *&Intersection_pt_idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j1, j2, a1, a2, pt, idx;

	if (f_v) {
		cout << "surface::compute_intersection_points_"
				"and_indices" << endl;
		}
	Intersection_pt = NEW_int(nb_lines * nb_lines);
	Intersection_pt_idx = NEW_int(nb_lines * nb_lines);
	int_vec_mone(Intersection_pt, nb_lines * nb_lines);
	for (j1 = 0; j1 < nb_lines; j1++) {
		a1 = Lines[j1];
		for (j2 = j1 + 1; j2 < nb_lines; j2++) {
			a2 = Lines[j2];
			if (Adj[j1 * nb_lines + j2]) {
				pt = P->line_intersection(a1, a2);
				if (!int_vec_search(Points, nb_points,
					pt, idx)) {
					cout << "surface::compute_intersection_points_"
							"and_indices cannot find point "
							"in Points" << endl;
					cout << "Points:";
					int_vec_print_fully(cout,
						Points, nb_points);
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
		cout << "surface::compute_intersection_points_"
				"and_indices done" << endl;
		}
}

void surface::lines_meet3_and_skew3(
	int *lines_meet3, int *lines_skew3,
	int *&lines, int &nb_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int o_rank[6];
	int i, j;
	int *perp;
	int perp_sz;

	if (f_v) {
		cout << "surface::lines_meet3_and_skew3" << endl;
		cout << "The three lines we will meet are ";
		int_vec_print(cout, lines_meet3, 3);
		cout << endl;
		cout << "The three lines we will be skew to are ";
		int_vec_print(cout, lines_skew3, 3);
		cout << endl;
		}
	for (i = 0; i < 3; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[lines_meet3[i]];
		}
	for (i = 0; i < 3; i++) {
		o_rank[3 + i] = Klein->Line_to_point_on_quadric[lines_skew3[i]];
		}

	O->perp_of_k_points(o_rank, 3, perp, perp_sz, verbose_level);

	lines = NEW_int(perp_sz);
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

	FREE_int(perp);

	for (i = 0; i < nb_lines; i++) {
		lines[i] = Klein->Point_on_quadric_to_line[lines[i]];
		}

	if (f_v) {
		cout << "surface::lines_meet3_and_skew3 done" << endl;
		}
}

void surface::perp_of_three_lines(int *three_lines,
	int *&perp, int &perp_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int o_rank[3];
	int i;

	if (f_v) {
		cout << "surface::perp_of_three_lines" << endl;
		cout << "The three lines are ";
		int_vec_print(cout, three_lines, 3);
		cout << endl;
		}
	for (i = 0; i < 3; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[three_lines[i]];
		}
	O->perp_of_k_points(o_rank, 3, perp, perp_sz, verbose_level);

	for (i = 0; i < perp_sz; i++) {
		perp[i] = Klein->Point_on_quadric_to_line[perp[i]];
		}

	if (f_v) {
		cout << "surface::perp_of_three_lines done" << endl;
		}
}

int surface::perp_of_four_lines(
	int *four_lines, int *trans12,
	int &perp_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int o_rank[4];
	int i;
	int *Perp;
	int ret = TRUE;

	if (f_v) {
		cout << "surface::perp_of_four_lines" << endl;
		cout << "The four lines are ";
		int_vec_print(cout, four_lines, 4);
		cout << endl;
		}
	for (i = 0; i < 4; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[four_lines[i]];
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

	trans12[0] = Klein->Point_on_quadric_to_line[Perp[0]];
	trans12[1] = Klein->Point_on_quadric_to_line[Perp[1]];

finish:
	FREE_int(Perp);
	if (f_v) {
		cout << "surface::perp_of_four_lines done" << endl;
		}
	return ret;
}

int surface::rank_of_four_lines_on_Klein_quadric(
	int *four_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int o_rank[4];
	int *coords;
	int i;
	int rk;

	if (f_v) {
		cout << "surface::rank_of_four_lines_on_Klein_quadric" << endl;
		cout << "The four lines are ";
		int_vec_print(cout, four_lines, 4);
		cout << endl;
		}
	for (i = 0; i < 4; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[four_lines[i]];
		}

	coords = NEW_int(4 * 6);
	for (i = 0; i < 4; i++) {
		O->unrank_point(coords + i * 6, 1,
			o_rank[i], 0 /* verbose_level */);
		}
	rk = F->Gauss_easy(coords, 4, 6);
	FREE_int(coords);
	if (f_v) {
		cout << "surface::rank_of_four_lines_on_Klein_quadric done" << endl;
		}
	return rk;
}

int surface::create_double_six_from_five_lines_with_a_common_transversal(
	int *five_pts, int *double_six,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int o_rank[12];
	int i, j;
	int ret = TRUE;

	if (f_v) {
		cout << "surface::create_double_six_from_five_lines_with_"
				"a_common_transversal" << endl;
		cout << "The five lines are ";
		int_vec_print(cout, five_pts, 5);
		cout << endl;
		}
	for (i = 0; i < 5; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[five_pts[i]];
		}
	for (i = 0; i < 5; i++) {
		for (j = i + 1; j < 5; j++) {
			if (O->evaluate_bilinear_form_by_rank(
					o_rank[i], o_rank[j]) == 0) {
				cout << "surface::create_double_six_from_five_lines_with_"
						"a_common_transversal two of the given lines "
						"intersect, error" << endl;
				exit(1);
				}
			}
		}
	if (f_v) {
		cout << "surface::create_double_six_from_five_lines_with_"
				"a_common_transversal" << endl;
		cout << "The five lines as orthogonal points are ";
		int_vec_print(cout, o_rank, 5);
		cout << endl;
		}

	int nb_subsets;
	int subset[4];
	int pts[4];
	int rk;
	int **Perp;
	int *Perp_sz;
	int lines[2];
	int opposites[5];
	int transversal = 0;

	nb_subsets = int_n_choose_k(5, 4);
	Perp = NEW_pint(nb_subsets);
	Perp_sz = NEW_int(nb_subsets);
#if 0
	for (rk = 0; rk < nb_subsets; rk++) {
		Perp[rk] = NEW_int(O->alpha * (O->q + 1));
		}
#endif
	for (rk = 0; rk < nb_subsets; rk++) {
		unrank_k_subset(rk, subset, 5, 4);
		for (i = 0; i < 4; i++) {
			pts[i] = o_rank[subset[i]];
			}

		if (f_v) {
			cout << "subset " << rk << " / " << nb_subsets << " : " << endl;
			}
		O->perp_of_k_points(pts, 4,
			Perp[rk], Perp_sz[rk], verbose_level - 1);
		if (f_v) {
			cout << "the perp of the subset ";
			int_vec_print(cout, subset, 4);
			cout << " has size " << Perp_sz[rk] << " : ";
			int_vec_print(cout, Perp[rk], Perp_sz[rk]);
			cout << endl;
			}
		if (Perp_sz[rk] != 2) {
			cout << "surface::create_double_six_from_five_lines_with_"
					"a_common_transversal Perp_opp_sz != 2, "
					"something is wrong" << endl;
			cout << "subset " << rk << " / " << nb_subsets << endl;
			exit(1);
			ret = FALSE;
			nb_subsets = rk + 1;
			goto finish;
			}
		if (rk == 0) {
			int_vec_copy(Perp[rk], lines, 2);
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
	if (f_v) {
		cout << "surface::create_double_six_from_five_lines_with_"
				"a_common_transversal" << endl;
		cout << "opposites ";
		int_vec_print(cout, opposites, 5);
		cout << endl;
		}

	o_rank[11] = transversal;
	for (i = 0; i < 5; i++) {
		o_rank[10 - i] = opposites[i];
		}

	int *Perp_opp;
	int Perp_opp_sz;
	int transversal_opp;

	//Perp_opp = NEW_int(O->alpha * (O->q + 1));
	O->perp_of_k_points(opposites, 4, Perp_opp, Perp_opp_sz,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "the perp of the opposite subset ";
		int_vec_print(cout, opposites, 4);
		cout << " has size " << Perp_opp_sz << ":";
		int_vec_print(cout, Perp_opp, Perp_opp_sz);
		cout << endl;
		}
	if (Perp_opp_sz != 2) {
		ret = FALSE;
		cout << "surface::create_double_six_from_five_lines_with_"
				"a_common_transversal Perp_opp_sz != 2, "
				"something is wrong" << endl;
		exit(1);
		FREE_int(Perp_opp);
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
		cout << "surface::create_double_six_from_five_lines_with_"
				"a_common_transversal something is wrong "
				"with Perp_opp" << endl;
		exit(1);
		}

	o_rank[5] = transversal_opp;

	if (f_v) {
		cout << "surface::create_double_six_from_five_lines_with_"
				"a_common_transversal" << endl;
		cout << "o_rank ";
		int_vec_print(cout, o_rank, 12);
		cout << endl;
		}

	for (i = 0; i < 12; i++) {
		double_six[i] = Klein->Point_on_quadric_to_line[o_rank[i]];
		}
	if (f_v) {
		cout << "surface::create_double_six_from_five_lines_with_"
				"a_common_transversal" << endl;
		cout << "double_six ";
		int_vec_print(cout, double_six, 12);
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


	FREE_int(Perp_opp);

finish:
	for (i = 0; i < nb_subsets; i++) {
		FREE_int(Perp[i]);
	}
	FREE_pint(Perp);
	//free_pint_all(Perp, nb_subsets);
	FREE_int(Perp_sz);

	if (f_v) {
		cout << "surface::create_double_six_from_five_lines_with_"
				"a_common_transversal done" << endl;
		}
	return ret;
}


int surface::create_double_six_from_six_disjoint_lines(
	int *single_six,
	int *double_six, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int o_rank[12];
	int i, j;
	int ret = FALSE;

	if (f_v) {
		cout << "surface::create_double_six_from_six_disjoint_lines" << endl;
		}
	for (i = 0; i < 6; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[single_six[i]];
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
	int **Perp_without_pt;
	int perp_sz = 0;
	int sz;

	sz = O->alpha * q;
	Perp_without_pt = NEW_pint(6);
	for (i = 0; i < 6; i++) {
		Perp_without_pt[i] = NEW_int(sz);
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
			int_vec_print(cout, Perp_without_pt[i], perp_sz);
			cout << endl;
			}
		}


	// compute the intersection of all perps, five at a time:

	int **I2, *I2_sz;
	int **I3, *I3_sz;
	int **I4, *I4_sz;
	int **I5, *I5_sz;
	int six2, six3, six4, six5, rk, rk2;
	int subset[6];

	six2 = int_n_choose_k(6, 2);
	I2 = NEW_pint(six2);
	I2_sz = NEW_int(six2);
	for (rk = 0; rk < six2; rk++) {
		unrank_k_subset(rk, subset, 6, 2);
		int_vec_intersect(
			Perp_without_pt[subset[0]],
			perp_sz,
			Perp_without_pt[subset[1]],
			perp_sz,
			I2[rk], I2_sz[rk]);
		if (f_v) {
			cout << "Perp_" << subset[0] << " \\cap Perp_" << subset[1]
				<< " of size " << I2_sz[rk] << " = ";
			int_vec_print(cout, I2[rk], I2_sz[rk]);
			cout << endl;
			}
		}
	six3 = int_n_choose_k(6, 3);
	I3 = NEW_pint(six3);
	I3_sz = NEW_int(six3);
	for (rk = 0; rk < six3; rk++) {
		unrank_k_subset(rk, subset, 6, 3);
		rk2 = rank_k_subset(subset, 6, 2);
		unrank_k_subset(rk, subset, 6, 3);
		int_vec_intersect(I2[rk2], I2_sz[rk2],
			Perp_without_pt[subset[2]],
			perp_sz,
			I3[rk], I3_sz[rk]);
		if (f_v) {
			cout << "Perp_" << subset[0] << " \\cap Perp_" << subset[1]
				<< " \\cap Perp_" << subset[2] << " of size "
				<< I3_sz[rk] << " = ";
			int_vec_print(cout, I3[rk], I3_sz[rk]);
			cout << endl;
			}
		}

	six4 = int_n_choose_k(6, 4);
	I4 = NEW_pint(six4);
	I4_sz = NEW_int(six4);
	for (rk = 0; rk < six4; rk++) {
		unrank_k_subset(rk, subset, 6, 4);
		rk2 = rank_k_subset(subset, 6, 3);
		unrank_k_subset(rk, subset, 6, 4);
		int_vec_intersect(I3[rk2], I3_sz[rk2],
			Perp_without_pt[subset[3]], perp_sz,
			I4[rk], I4_sz[rk]);
		if (f_v) {
			cout << rk << " / " << six4 << " : Perp_" << subset[0]
				<< " \\cap Perp_" << subset[1] << " \\cap Perp_"
				<< subset[2] << " \\cap Perp_" << subset[3]
				<< " of size " << I4_sz[rk] << " = ";
			int_vec_print(cout, I4[rk], I4_sz[rk]);
			cout << endl;
			}
		}

	six5 = int_n_choose_k(6, 5);
	I5 = NEW_pint(six5);
	I5_sz = NEW_int(six5);
	for (rk = 0; rk < six5; rk++) {
		unrank_k_subset(rk, subset, 6, 5);
		rk2 = rank_k_subset(subset, 6, 4);
		unrank_k_subset(rk, subset, 6, 5);
		int_vec_intersect(I4[rk2], I4_sz[rk2],
			Perp_without_pt[subset[4]], perp_sz,
			I5[rk], I5_sz[rk]);
		if (f_v) {
			cout << rk << " / " << six5 << " : Perp_" << subset[0]
				<< " \\cap Perp_" << subset[1] << " \\cap Perp_"
				<< subset[2] << " \\cap Perp_" << subset[3]
				<< " \\cap Perp_" << subset[4] << " of size "
				<< I5_sz[rk] << " = ";
			int_vec_print(cout, I5[rk], I5_sz[rk]);
			cout << endl;
			}

		if (I5_sz[rk] != 1) {
			cout << "surface::create_double_six I5_sz[rk] != 1" << endl;
			ret = FALSE;
			goto free_it;
			}
		}
	for (i = 0; i < 6; i++) {
		o_rank[6 + i] = I5[6 - 1 - i][0];
		}
	for (i = 0; i < 12; i++) {
		double_six[i] = Klein->Point_on_quadric_to_line[o_rank[i]];
		}

	ret = TRUE;
free_it:
	for (i = 0; i < 6; i++) {
		FREE_int(Perp_without_pt[i]);
	}
	FREE_pint(Perp_without_pt);
	//free_pint_all(Perp_without_pt, 6);



	for (i = 0; i < six2; i++) {
		FREE_int(I2[i]);
	}
	FREE_pint(I2);
	//free_pint_all(I2, six2);

	FREE_int(I2_sz);


	for (i = 0; i < six3; i++) {
		FREE_int(I3[i]);
	}
	FREE_pint(I3);
	//free_pint_all(I3, six3);

	FREE_int(I3_sz);


	for (i = 0; i < six4; i++) {
		FREE_int(I4[i]);
	}
	FREE_pint(I4);

	//free_pint_all(I4, six4);

	FREE_int(I4_sz);


	for (i = 0; i < six5; i++) {
		FREE_int(I5[i]);
	}
	FREE_pint(I5);

	//free_pint_all(I5, six5);

	FREE_int(I5_sz);


	if (f_v) {
		cout << "surface::create_double_six_from_six_"
				"disjoint_lines done" << endl;
		}
	return ret;
}

void surface::create_the_fifteen_other_lines(
	int *double_six,
	int *fifteen_other_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_the_fifteen_other_lines" << endl;
		}
	int *Planes;
	int *Lines;
	int h, k3;
	int i, j;

	Planes = NEW_int(30);
	if (f_v) {
		cout << "creating the 30 planes:" << endl;
		}
	for (h = 0; h < 30; h++) {
		i = Sets[h * 2 + 0];
		j = Sets[h * 2 + 1];
		Gr->unrank_int_here(Basis0, double_six[i],
				0/* verbose_level*/);
		Gr->unrank_int_here(Basis0 + 8, double_six[j],
				0/* verbose_level*/);
		if (F->Gauss_easy(Basis0, 4, 4) != 3) {
			cout << "the rank is not 3" << endl;
			exit(1);
			}
		Planes[h] = Gr3->rank_int_here(Basis0,
				0/* verbose_level*/);
		if (f_v) {
			cout << "plane " << h << " / " << 30
				<< " has rank " << Planes[h] << " and basis ";
			int_vec_print(cout, Basis0, 12);
			cout << endl;
			}
		}
	Lines = NEW_int(15);
	if (f_v) {
		cout << "creating the 15 lines:" << endl;
		}
	for (h = 0; h < 15; h++) {
		i = Sets2[h * 2 + 0];
		j = Sets2[h * 2 + 1];
		Gr3->unrank_int_here(Basis1, Planes[i],
				0/* verbose_level*/);
		Gr3->unrank_int_here(Basis2, Planes[j],
				0/* verbose_level*/);
		F->intersect_subspaces(4, 3, Basis1, 3, Basis2,
			k3, Basis0, 0 /* verbose_level */);
		if (k3 != 2) {
			cout << "the rank is not 2" << endl;
			exit(1);
			}
		Lines[h] = Gr->rank_int_here(Basis0,
				0/* verbose_level*/);
		for (i = 0; i < 2; i++) {
			F->PG_element_normalize_from_front(
				Basis0 + i * 4, 1, 4);
			}
		if (f_v) {
			cout << "line " << h << " / " << 15
				<< " has rank " << Lines[h]
				<< " and basis ";
			int_vec_print(cout, Basis0, 8);
			cout << endl;
			}
		}

	int_vec_copy(Lines, fifteen_other_lines, 15);


	FREE_int(Planes);
	FREE_int(Lines);

	if (f_v) {
		cout << "create_the_fifteen_other_lines done" << endl;
		}

}

void surface::init_adjacency_matrix_of_lines(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, l;

	if (f_v) {
		cout << "surface::init_adjacency_matrix_of_lines" << endl;
		}

	adjacency_matrix_of_lines = NEW_int(27 * 27);
	int_vec_zero(adjacency_matrix_of_lines, 27 * 27);

	// the ai lines:
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (j == i) {
				continue;
			}
			set_adjacency_matrix_of_lines(line_ai(i), line_bi(j));
		}
		for (k = 0; k < 6; k++) {
			if (k == i) {
				continue;
			}
			set_adjacency_matrix_of_lines(line_ai(i), line_cij(i, k));
		}
	}


	// the bi lines:
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (j == i) {
				continue;
			}
			set_adjacency_matrix_of_lines(line_bi(i), line_ai(j));
		}
		for (k = 0; k < 6; k++) {
			if (k == i) {
				continue;
			}
			set_adjacency_matrix_of_lines(line_bi(i), line_cij(i, k));
		}
	}




	// the cij lines:
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (j == i) {
				continue;
			}
			for (k = 0; k < 6; k++) {
				if (k == i) {
					continue;
				}
				if (k == j) {
					continue;
				}
				for (l = 0; l < 6; l++) {
					if (l == i) {
						continue;
					}
					if (l == j) {
						continue;
					}
					if (k == l) {
						continue;
					}
					set_adjacency_matrix_of_lines(
							line_cij(i, j), line_cij(k, l));
				} // next l
			} // next k
		} // next j
	} // next i

	int r, c;

	for (i = 0; i < 27; i++) {
		r = 0;
		for (j = 0; j < 27; j++) {
			if (get_adjacency_matrix_of_lines(i, j)) {
				r++;
			}
		}
		if (r != 10) {
			cout << "surface::init_adjacency_matrix_of_lines "
					"row sum r != 10, r = " << r << " in row " << i << endl;
		}
	}

	for (j = 0; j < 27; j++) {
		c = 0;
		for (i = 0; i < 27; i++) {
			if (get_adjacency_matrix_of_lines(i, j)) {
				c++;
			}
		}
		if (c != 10) {
			cout << "surface::init_adjacency_matrix_of_lines "
					"col sum c != 10, c = " << c << " in col " << j << endl;
		}
	}

	if (f_v) {
		cout << "surface::init_adjacency_matrix_of_lines done" << endl;
		}
}

void surface::set_adjacency_matrix_of_lines(int i, int j)
{
	adjacency_matrix_of_lines[i * 27 + j] = 1;
	adjacency_matrix_of_lines[j * 27 + i] = 1;
}

int surface::get_adjacency_matrix_of_lines(int i, int j)
{
	return adjacency_matrix_of_lines[i * 27 + j];
}

void surface::compute_adjacency_matrix_of_line_intersection_graph(
	int *&Adj, int *S, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "surface::compute_adjacency_matrix_of_"
				"line_intersection_graph" << endl;
		}
	if (n > 27) {
		cout << "surface::compute_adjacency_matrix_of_"
				"line_intersection_graph n > 27" << endl;
		exit(1);
		}
	for (i = 0; i < n; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[S[i]];
		}

	Adj = NEW_int(n * n);
	int_vec_zero(Adj, n * n);
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (O->evaluate_bilinear_form_by_rank(
				o_rank[i], o_rank[j]) == 0) {
				Adj[i * n + j] = 1;
				Adj[j * n + i] = 1;
				}
			}
		}
	if (f_v) {
		cout << "surface::compute_adjacency_matrix_of_"
				"line_intersection_graph done" << endl;
		}
}

void surface::compute_adjacency_matrix_of_line_disjointness_graph(
	int *&Adj, int *S, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "surface::compute_adjacency_matrix_of_"
				"line_disjointness_graph" << endl;
		}
	if (n > 27) {
		cout << "surface::compute_adjacency_matrix_of_"
				"line_disjointness_graph n > 27" << endl;
		exit(1);
		}
	for (i = 0; i < n; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[S[i]];
		}

	Adj = NEW_int(n * n);
	int_vec_zero(Adj, n * n);
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (O->evaluate_bilinear_form_by_rank(
				o_rank[i], o_rank[j]) != 0) {
				Adj[i * n + j] = 1;
				Adj[j * n + i] = 1;
				}
			}
		}
	if (f_v) {
		cout << "surface::compute_adjacency_matrix_of_"
				"line_disjointness_graph done" << endl;
		}
}

void surface::compute_points_on_lines(
	int *Pts_on_surface, int nb_points_on_surface,
	int *Lines, int nb_lines,
	set_of_sets *&pts_on_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, l, r;
	int *Surf_pt_coords;
	int Basis[8];
	int Mtx[12];

	if (f_v) {
		cout << "surface::compute_points_on_lines" << endl;
		}
	pts_on_lines = NEW_OBJECT(set_of_sets);
	pts_on_lines->init_basic_constant_size(nb_points_on_surface,
		nb_lines, q + 1, 0 /* verbose_level */);
	Surf_pt_coords = NEW_int(nb_points_on_surface * 4);
	for (i = 0; i < nb_points_on_surface; i++) {
		P->unrank_point(Surf_pt_coords + i * 4, Pts_on_surface[i]);
		}

	int_vec_zero(pts_on_lines->Set_size, nb_lines);
	for (i = 0; i < nb_lines; i++) {
		l = Lines[i];
		P->unrank_line(Basis, l);
		//cout << "Line " << i << " basis=";
		//int_vec_print(cout, Basis, 8);
		//cout << " : ";
		for (j = 0; j < nb_points_on_surface; j++) {
			int_vec_copy(Basis, Mtx, 8);
			int_vec_copy(Surf_pt_coords + j * 4, Mtx + 8, 4);
			r = F->Gauss_easy(Mtx, 3, 4);
			if (r == 2) {
				pts_on_lines->add_element(i, j);
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
		cout << "surface::compute_points_on_lines done" << endl;
		}
}

int surface::compute_rank_of_any_four(
	int *&Rk, int &nb_subsets,
	int *lines, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int subset[4];
	int four_lines[4];
	int i, rk;
	int ret = TRUE;

	if (f_v) {
		cout << "surface::compute_rank_of_any_four" << endl;
		}
	nb_subsets = int_n_choose_k(sz, 4);
	Rk = NEW_int(nb_subsets);
	for (rk = 0; rk < nb_subsets; rk++) {
		unrank_k_subset(rk, subset, sz, 4);
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
		int_vec_print(cout, Rk, nb_subsets);
		cout << endl;
		}
	if (f_v) {
		cout << "surface::compute_rank_of_any_four done" << endl;
		}
	return ret;
}

void surface::rearrange_lines_according_to_double_six(int *Lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Adj;
	int nb_lines = 27;
	int New_lines[27];

	if (f_v) {
		cout << "surface::rearrange_lines_according_to_double_six" << endl;
		}
	if (f_v) {
		cout << "surface::rearrange_lines_according_to_double_six "
			"before compute_adjacency_matrix_of_line_"
			"intersection_graph" << endl;
		}
	compute_adjacency_matrix_of_line_intersection_graph(Adj,
		Lines, nb_lines, 0 /* verbose_level */);


	set_of_sets *line_intersections;
	int *Starter_Table;
	int nb_starter;

	line_intersections = NEW_OBJECT(set_of_sets);

	if (f_v) {
		cout << "surface::rearrange_lines_according_to_double_six "
			"before line_intersections->init_from_adjacency_matrix"
			<< endl;
		}
	line_intersections->init_from_adjacency_matrix(nb_lines, Adj,
		0 /* verbose_level */);

	if (f_v) {
		cout << "surface::rearrange_lines_according_to_double_six "
			"before list_starter_configurations" << endl;
		}
	list_starter_configurations(Lines, nb_lines,
		line_intersections, Starter_Table, nb_starter,
		0 /*verbose_level */);

	int l, line_idx, subset_idx;

	if (nb_starter == 0) {
		cout << "surface::rearrange_lines_according_to_double_six "
				"nb_starter == 0" << endl;
		exit(1);
		}
	l = 0;
	line_idx = Starter_Table[l * 2 + 0];
	subset_idx = Starter_Table[l * 2 + 1];

	if (f_v) {
		cout << "surface::rearrange_lines_according_to_double_six "
			"before rearrange_lines_according_to_starter_"
			"configuration" << endl;
		}
	rearrange_lines_according_to_starter_configuration(
		Lines, New_lines,
		line_idx, subset_idx, Adj,
		line_intersections, 0 /*verbose_level*/);

	int_vec_copy(New_lines, Lines, 27);

	FREE_int(Adj);
	FREE_int(Starter_Table);
	FREE_OBJECT(line_intersections);
	if (f_v) {
		cout << "surface::rearrange_lines_according_"
				"to_double_six done" << endl;
		}
}

void surface::rearrange_lines_according_to_starter_configuration(
	int *Lines, int *New_lines,
	int line_idx, int subset_idx, int *Adj,
	set_of_sets *line_intersections,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int S3[6];
	int i, idx;
	int nb_lines = 27;


	if (f_v) {
		cout << "surface::rearrange_lines_according_"
				"to_starter_configuration" << endl;
		}

	create_starter_configuration(line_idx, subset_idx,
		line_intersections, Lines, S3, 0 /* verbose_level */);


	if (f_v) {
		cout << "line_intersections:" << endl;
		line_intersections->print_table();
		}

	int Line_idx[27];
	for (i = 0; i < 6; i++) {
		if (!int_vec_search_linear(Lines, nb_lines, S3[i], idx)) {
			cout << "could not find the line" << endl;
			exit(1);
			}
		Line_idx[i] = idx;
		}

	if (f_v) {
		cout << "The 5+1 lines are ";
		int_vec_print(cout, Line_idx, 6);
		cout << endl;
		}

	Line_idx[11] = Line_idx[5];
	Line_idx[5] = 0;
	int_vec_zero(New_lines, 27);
	int_vec_copy(S3, New_lines, 5);
	New_lines[11] = S3[5];

	if (f_v) {
		cout << "computing b_j:" << endl;
		}
	for (i = 0; i < 5; i++) {
		int four_lines[4];

		if (f_v) {
			cout << i << " / " << 5 << ":" << endl;
			}

		int_vec_copy(Line_idx, four_lines, i);
		int_vec_copy(Line_idx + i + 1, four_lines + i, 5 - i - 1);
		if (f_v) {
			cout << "four_lines=";
			int_vec_print(cout, four_lines, 4);
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
	int_vec_copy(Line_idx + 6, five_lines_idx, 5);
	Line_idx[5] = intersection_of_five_lines(Adj,
		five_lines_idx, verbose_level);
	if (f_v) {
		cout << "a_" << i + 1 << " = "
			<< Line_idx[5] << endl;
		}


	int double_six[12];
	int h, j;

	for (i = 0; i < 12; i++) {
		double_six[i] = Lines[Line_idx[i]];
		}
	int_vec_copy(double_six, New_lines, 12);

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
		int_vec_print(cout, New_lines, 27);
		cout << endl;
		}

	if (f_v) {
		cout << "surface::rearrange_lines_according_"
				"to_starter_configuration done" << endl;
		}
}

int surface::intersection_of_four_lines_but_not_b6(int *Adj,
	int *four_lines_idx, int b6, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, i, j;

	if (f_v) {
		cout << "surface::intersection_of_four_given_"
				"line_intersections_but_not_b6" << endl;
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
		cout << "surface::intersection_of_four_lines_but_"
				"not_b6 could not find the line" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "surface::intersection_of_four_given_"
				"line_intersections_but_not_b6 done" << endl;
		}
	return a;
}

int surface::intersection_of_five_lines(int *Adj,
	int *five_lines_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, i, j;

	if (f_v) {
		cout << "surface::intersection_of_five_lines" << endl;
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
		cout << "surface::intersection_of_five_lines "
				"could not find the line" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "surface::intersection_of_five_lines done" << endl;
		}
	return a;
}

void surface::rearrange_lines_according_to_a_given_double_six(
	int *Lines,
	int *New_lines, int *double_six, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;


	if (f_v) {
		cout << "surface::rearrange_lines_according_to_a_"
				"given_double_six" << endl;
		}
	for (i = 0; i < 12; i++) {
		New_lines[i] = Lines[double_six[i]];
		}
	h = 0;
	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++, h++) {
			New_lines[12 + h] = compute_cij(
				New_lines /*double_six */,
				i, j, 0 /* verbose_level */);
			}
		}
	if (f_v) {
		cout << "New_lines:";
		int_vec_print(cout, New_lines, 27);
		cout << endl;
		}

	if (f_v) {
		cout << "surface::rearrange_lines_according_to_a_"
				"given_double_six done" << endl;
		}
}

void surface::create_lines_from_plane_equations(
	int *The_plane_equations,
	int *Lines27, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int line_idx, plane1, plane2;
	int Basis[16];

	if (f_v) {
		cout << "surface::create_lines_from_plane_equations" << endl;
		}

	for (line_idx = 0; line_idx < 27; line_idx++) {
		find_tritangent_planes_intersecting_in_a_line(
			line_idx, plane1, plane2, 0 /* verbose_level */);
		int_vec_copy(The_plane_equations + plane1 * 4, Basis, 4);
		int_vec_copy(The_plane_equations + plane2 * 4, Basis + 4, 4);
		F->perp_standard(4, 2, Basis, 0 /* verbose_level */);
		Lines27[line_idx] = rank_line(Basis + 8);
		}

	if (f_v) {
		cout << "surface::create_lines_from_plane_equations done" << endl;
		}
}

int surface::identify_two_lines(
		int *lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int iso = 0;
	int *Adj;

	if (f_v) {
		cout << "surface::identify_two_lines" << endl;
		}

	compute_adjacency_matrix_of_line_intersection_graph(
		Adj, lines, 2, 0 /* verbose_level */);
	if (Adj[0 * 2 + 1]) {
		iso = 1;
		}
	else {
		iso = 0;
		}
	FREE_int(Adj);
	if (f_v) {
		cout << "surface::identify_two_lines done" << endl;
		}
	return iso;
}


int surface::identify_three_lines(int *lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int iso = 0;
	int *Adj;
	int i, j, c;
	int a1, a2; //, a3;

	if (f_v) {
		cout << "surface::identify_three_lines" << endl;
		}

	compute_adjacency_matrix_of_line_intersection_graph(
		Adj, lines, 3, 0 /* verbose_level */);


	c = 0;
	for (i = 0; i < 3; i++) {
		for (j = i + 1; j < 3; j++) {
			if (Adj[i * 3 + j]) {
				c++;
				}
			}
		}
	if (c == 0) {
		iso = 6;
		}
	else if (c == 1) {
		iso = 4;
		}
	else if (c == 2) {
		iso = 5;
		}
	else if (c == 3) {
		int *Intersection_pt;
		int rk;

		compute_intersection_points(Adj,
			lines, 3, Intersection_pt,
			0 /*verbose_level */);
		a1 = Intersection_pt[0 * 3 + 1];
		a2 = Intersection_pt[0 * 3 + 2];
		//a3 = Intersection_pt[1 * 3 + 2];
		if (a1 == a2) {
			int Basis[3 * 8];

			for (i = 0; i < 3; i++) {
				Gr->unrank_int_here(Basis + i * 8,
					lines[i], 0 /* verbose_level */);
				}
			rk = F->Gauss_easy(Basis, 6, 4);
			if (rk == 3) {
				iso = 2;
				}
			else if (rk == 4) {
				iso = 1;
				}
			else {
				cout << "surface::identify_three_lines rk=" << rk << endl;
				exit(1);
				}
			}
		else {
			iso = 3;
			}
		FREE_int(Intersection_pt);
		}


	FREE_int(Adj);
	if (f_v) {
		cout << "surface::identify_three_lines done" << endl;
		}
	return iso;
}


void surface::create_remaining_fifteen_lines(
	int *double_six, int *fifteen_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, h;

	if (f_v) {
		cout << "surface::create_remaining_fifteen_lines" << endl;
		}
	h = 0;
	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++) {
			if (f_vv) {
				cout << "surface::create_remaining_fifteen_lines "
						"creating line c_ij where i=" << i
						<< " j=" << j << ":" << endl;
				}
			fifteen_lines[h++] = compute_cij(
				double_six, i, j, 0 /*verbose_level*/);
			}
		}
	if (f_v) {
		cout << "surface::create_remaining_fifteen_lines done" << endl;
		}
}

int surface::compute_cij(int *double_six,
		int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ai, aj, bi, bj;
	int Basis1[16];
	int Basis2[16];
	int K1[16];
	int K2[16];
	int K[16];
	int base_cols1[4];
	int base_cols2[4];
	int kernel_m, kernel_n, cij;

	if (f_v) {
		cout << "surface::compute_cij" << endl;
		}
	ai = double_six[i];
	aj = double_six[j];
	bi = double_six[6 + i];
	bj = double_six[6 + j];
	Gr->unrank_int_here(Basis1, ai,
			0 /* verbose_level */);
	Gr->unrank_int_here(Basis1 + 2 * 4, bj,
			0 /* verbose_level */);
	Gr->unrank_int_here(Basis2, aj,
			0 /* verbose_level */);
	Gr->unrank_int_here(Basis2 + 2 * 4, bi,
			0 /* verbose_level */);
	if (F->Gauss_simple(Basis1, 4, 4, base_cols1,
			0 /* verbose_level */) != 3) {
		cout << "The rank of Basis1 is not 3" << endl;
		exit(1);
		}
	if (F->Gauss_simple(Basis2, 4, 4, base_cols2,
			0 /* verbose_level */) != 3) {
		cout << "The rank of Basis2 is not 3" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface::compute_cij before matrix_get_"
				"kernel Basis1" << endl;
		}
	F->matrix_get_kernel(Basis1, 3, 4, base_cols1, 3,
		kernel_m, kernel_n, K1);
	if (kernel_m != 4) {
		cout << "surface::compute_cij kernel_m != 4 "
				"when computing K1" << endl;
		exit(1);
		}
	if (kernel_n != 1) {
		cout << "surface::compute_cij kernel_1 != 1 "
				"when computing K1" << endl;
		exit(1);
		}
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < 4; i++) {
			K[j * 4 + i] = K1[i * kernel_n + j];
			}
		}
	if (f_v) {
		cout << "surface::compute_cij before matrix_"
				"get_kernel Basis2" << endl;
		}
	F->matrix_get_kernel(Basis2, 3, 4, base_cols2, 3,
		kernel_m, kernel_n, K2);
	if (kernel_m != 4) {
		cout << "surface::compute_cij kernel_m != 4 "
				"when computing K2" << endl;
		exit(1);
		}
	if (kernel_n != 1) {
		cout << "surface::compute_cij kernel_1 != 1 "
				"when computing K2" << endl;
		exit(1);
		}
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < 4; i++) {
			K[(1 + j) * 4 + i] = K2[i * kernel_n + j];
			}
		}
	if (F->Gauss_simple(K, 2, 4, base_cols1,
			0 /* verbose_level */) != 2) {
		cout << "The rank of K is not 2" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface::compute_cij before "
				"matrix_get_kernel K" << endl;
		}
	F->matrix_get_kernel(K, 2, 4, base_cols1, 2,
		kernel_m, kernel_n, K1);
	if (kernel_m != 4) {
		cout << "surface::compute_cij kernel_m != 4 "
				"when computing final kernel" << endl;
		exit(1);
		}
	if (kernel_n != 2) {
		cout << "surface::compute_cij kernel_n != 2 "
				"when computing final kernel" << endl;
		exit(1);
		}
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < n; i++) {
			Basis1[j * n + i] = K1[i * kernel_n + j];
			}
		}
	cij = Gr->rank_int_here(Basis1, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface::compute_cij done" << endl;
		}
	return cij;
}

int surface::compute_transversals_of_any_four(
	int *&Trans, int &nb_subsets,
	int *lines, int sz,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int trans12[2];
	int subset[4];
	int four_lines[4];
	int i, rk, perp_sz;
	int ret = TRUE;

	if (f_v) {
		cout << "surface::compute_transversals_of_any_four" << endl;
		}
	nb_subsets = int_n_choose_k(sz, 4);
	Trans = NEW_int(nb_subsets * 2);
	for (rk = 0; rk < nb_subsets; rk++) {
		unrank_k_subset(rk, subset, sz, 4);
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
				int_vec_print(cout, subset, 4);
				cout << " = ";
				int_vec_print(cout, four_lines, 4);
				cout << " perp_sz=" << perp_sz << endl;
				}
			ret = FALSE;
			//break;
			trans12[0] = -1;
			trans12[1] = -1;
			}
		int_vec_copy(trans12, Trans + rk * 2, 2);
		}
	if (f_v) {
		cout << "Transversals:" << endl;
		int_matrix_print(Trans, nb_subsets, 2);
		}
	if (f_v) {
		cout << "surface::compute_transversals_of_any_four done" << endl;
		}
	return ret;
}

}

