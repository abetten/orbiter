/*
 * blt_set_domain.cpp
 *
 *  Created on: Apr 6, 2019
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {



blt_set_domain::blt_set_domain()
{
	F = NULL;
	f_semilinear = FALSE;
	epsilon = 0;
	n = 0;
	q = 0;
	target_size = 0;
	degree = 0;


	O = NULL;
	f_orthogonal_allocated = FALSE;
	Pts = NULL;
	Candidates = NULL;
	P = NULL;
	G53 = NULL;
	//null();
}

blt_set_domain::~blt_set_domain()
{
	freeself();
}

void blt_set_domain::null()
{
}

void blt_set_domain::freeself()
{
	int f_v = FALSE;

	if (f_v) {
		cout << "blt_set_domain::freeself" << endl;
	}
	if (f_orthogonal_allocated) {
		if (f_v) {
			cout << "blt_set_domain::freeself before O" << endl;
		}
		if (O) {
			delete O;
		}
		f_orthogonal_allocated = FALSE;
		O = NULL;
	}
	if (Pts) {
		FREE_int(Pts);
	}
	if (Candidates) {
		FREE_int(Candidates);
	}
	if (P) {
		FREE_OBJECT(P);
	}
	if (G53) {
		FREE_OBJECT(G53);
	}
	null();
	if (f_v) {
		cout << "blt_set_domain::freeself done" << endl;
	}
}



void blt_set_domain::init(orthogonal *O,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;

	if (f_v) {
		cout << "blt_set_domain::init" << endl;
		cout << "blt_set_domain::init "
				"verbose_level = " << verbose_level << endl;
	}


	blt_set_domain::O = O;
	f_orthogonal_allocated = FALSE;
	blt_set_domain::F = O->F;
	blt_set_domain::q = F->q;
	n = O->n; // vector space dimension
	epsilon = O->epsilon;


	target_size = q + 1;
	degree = O->nb_points;


	if (f_v) {
		cout << "blt_set_domain::init q=" << q
				<< " target_size = " << target_size << endl;
	}


	f_semilinear = TRUE;
	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
	}
	if (f_v) {
		cout << "blt_set_domain::init "
				"f_semilinear=" << f_semilinear << endl;
	}


	if (f_v) {
		cout << "blt_set_domain::init "
				"allocating Pts and Candidates" << endl;
	}


	Pts = NEW_int(target_size * n);

	Candidates = NEW_int(degree * n);


	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "blt_set_domain::init before P->init" << endl;
	}


	P->init(4, F,
		FALSE /* f_init_incidence_structure */,
		verbose_level);

	if (f_v) {
		cout << "blt_set_domain::init after P->init" << endl;
	}


	G53 = NEW_OBJECT(grassmann);

	G53->init(5, 3, F, 0 /*verbose_level - 2*/);

	if (f_v) {
		cout << "blt_set_domain::init finished" << endl;
	}
}

void blt_set_domain::compute_adjacency_list_fast(
	int first_point_of_starter,
	long int *points, int nb_points, int *point_color,
	uchar *&bitvector_adjacency,
	long int &bitvector_length_in_bits,
	long int &bitvector_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int L;
	long int i, j, k;
	int c1, c2;
	int *Pts;
	int *form_value;
	int v1[5];
	int m[5];
	int f12, f13, f23, d;
	uint cnt;
	int two;
	int *Pi, *Pj;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "blt_set_domain::compute_adjacency_list_fast" << endl;
		}
	L = ((long int) nb_points * ((long int) nb_points - 1)) >> 1;

	bitvector_length_in_bits = L;
	bitvector_length = (L + 7) >> 3;
	bitvector_adjacency = NEW_uchar(bitvector_length);
	for (i = 0; i < bitvector_length; i++) {
		bitvector_adjacency[i] = 0;
	}

	Pts = NEW_int(nb_points * 5);
	form_value = NEW_int(nb_points);
	O->unrank_point(v1, 1, first_point_of_starter, 0);
	if (f_v) {
		cout << "blt_set_domain::compute_adjacency_list_fast "
				"unranking points" << endl;
	}
	for (i = 0; i < nb_points; i++) {
		O->unrank_point(Pts + i * 5, 1, points[i], 0);
		form_value[i] = O->evaluate_bilinear_form(
				v1, Pts + i * 5, 1);
	}

	if (f_v) {
		cout << "blt_set_domain::compute_adjacency_list_fast "
				"computing adjacencies" << endl;
	}

	cnt = 0;
	two = F->add(1, 1);

	for (i = 0; i < nb_points; i++) {
		f12 = form_value[i];
		c1 = point_color[i];
		Pi = Pts + i * 5;
		m[0] = F->mult(Pi[0], two);
		m[1] = Pi[2];
		m[2] = Pi[1];
		m[3] = Pi[4];
		m[4] = Pi[3];

		for (j = i + 1; j < nb_points; j++, cnt++) {
			k = Combi.ij2k_lint(i, j, nb_points);

			if ((cnt & ((1 << 25) - 1)) == 0 && cnt) {
				cout << "blt_set_domain::compute_adjacency_list_fast "
						"nb_points=" << nb_points << " adjacency "
						<< cnt << " / " << L << " i=" << i
						<< " j=" << j << endl;
			}
			c2 = point_color[j];
			if (c1 == c2) {
				bitvector_m_ii(bitvector_adjacency, k, 0);
				continue;
			}
			f13 = form_value[j];
			Pj = Pts + j * 5;
			f23 = F->add5(
				F->mult(m[0], Pj[0]),
				F->mult(m[1], Pj[1]),
				F->mult(m[2], Pj[2]),
				F->mult(m[3], Pj[3]),
				F->mult(m[4], Pj[4])
				);
			d = F->product3(f12, f13, f23);
			if (d == 0) {
				bitvector_m_ii(bitvector_adjacency, k, 0);
			}
			else {
				if (O->f_is_minus_square[d]) {
					bitvector_m_ii(bitvector_adjacency, k, 0);
				}
				else {
					bitvector_m_ii(bitvector_adjacency, k, 1);
				}
			}

		} // next j
	} // next i



	FREE_int(Pts);
	FREE_int(form_value);
	if (f_v) {
		cout << "blt_set_domain::compute_adjacency_list_fast done" << endl;
	}
}



void blt_set_domain::compute_colors(int orbit_at_level,
	long int *starter, int starter_sz,
	long int special_line,
	long int *candidates, int nb_candidates,
	int *&point_color, int &nb_colors,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int p1, p2;
	int v1[5];
	int v2[5];
	int v3[5];
	long int *pts_on_special_line;
	int idx, i;
	sorting Sorting;


	if (f_v) {
		cout << "blt_set_domain::compute_colors" << endl;
	}
	O->unrank_line(p1, p2, special_line, 0/*verbose_level*/);
	if (f_vv) {
		cout << "after unrank_line " << special_line << ":" << endl;
		cout << "p1=" << p1 << " p2=" << p2 << endl;
	}
	O->unrank_point(v1, 1, p1, 0);
	O->unrank_point(v2, 1, p2, 0);
	if (f_vv) {
		cout << "p1=" << p1 << " ";
		int_vec_print(cout, v1, 5);
		cout << endl;
		cout << "p2=" << p2 << " ";
		int_vec_print(cout, v2, 5);
		cout << endl;
	}
	if (p1 != starter[0]) {
		cout << "p1 != starter[0]" << endl;
		exit(1);
	}

	pts_on_special_line = NEW_lint(q + 1);
	O->points_on_line(p1, p2, pts_on_special_line,
			0/*verbose_level*/);

	if (f_vv) {
		cout << "pts_on_special_line:" << endl;
		lint_vec_print(cout, pts_on_special_line, q + 1);
		cout << endl;
	}

	if (!Sorting.lint_vec_search(pts_on_special_line, q + 1, starter[0], idx, 0)) {
		cout << "cannot find the first point on the line" << endl;
		exit(1);
	}
	for (i = idx; i < q + 1; i++) {
		pts_on_special_line[i] = pts_on_special_line[i + 1];
	}
	if (f_vv) {
		cout << "pts_on_special_line without the first "
				"starter point:" << endl;
		lint_vec_print(cout, pts_on_special_line, q);
		cout << endl;
	}

	int a, b;
	int t, c, j, h;
	int *starter_t;

	starter_t = NEW_int(starter_sz);
	starter_t[0] = -1;
	for (i = 1; i < starter_sz; i++) {
		O->unrank_point(v3, 1, starter[i], 0);
		a = O->evaluate_bilinear_form(v1, v3, 1);
		b = O->evaluate_bilinear_form(v2, v3, 1);
		if (a == 0) {
			cout << "a == 0, this should not be" << endl;
			exit(1);
		}
		// <v3,t*v1+v2> = t*<v3,v1>+<v3,v2> = t*a+b = 0
		// Thus, t = -b/a
		t = O->F->mult(O->F->negate(b), O->F->inverse(a));
		starter_t[i] = t;
	}

	if (f_vv) {
		cout << "starter_t:" << endl;
		int_vec_print(cout, starter_t, starter_sz);
		cout << endl;
	}

	long int *free_pts;
	int *open_colors;
	int *open_colors_inv;

	free_pts = NEW_lint(q);
	open_colors = NEW_int(q);
	open_colors_inv = NEW_int(q);

	point_color = NEW_int(nb_candidates);

	nb_colors = q - starter_sz + 1;
	j = 0;
	for (i = 0; i < q; i++) {
		for (h = 1; h < starter_sz; h++) {
			if (starter_t[h] == i) {
				break;
			}
		}
		if (h == starter_sz) {
			free_pts[j] = pts_on_special_line[i];
			open_colors[j] = i;
			j++;
		}
	}
	if (j != nb_colors) {
		cout << "blt_set_domain::compute_colors error: j != nb_colors" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "The " << nb_colors << " free points are :" << endl;
		lint_vec_print(cout, free_pts, nb_colors);
		cout << endl;
		cout << "The " << nb_colors << " open colors are :" << endl;
		int_vec_print(cout, open_colors, nb_colors);
		cout << endl;
	}
	for ( ; j < q; j++) {
		open_colors[j] = starter_t[j - nb_colors + 1];
	}
	if (f_vv) {
		cout << "open_colors :" << endl;
		int_vec_print(cout, open_colors, q);
		cout << endl;
	}
	for (i = 0; i < q; i++) {
		j = open_colors[i];
		open_colors_inv[j] = i;
	}
	if (f_vv) {
		cout << "open_colors_inv :" << endl;
		int_vec_print(cout, open_colors_inv, q);
		cout << endl;
	}


	for (i = 0; i < nb_candidates; i++) {
		O->unrank_point(v3, 1, candidates[i], 0);
		if (f_vv) {
			cout << "candidate " << i << " / " << nb_candidates
					<< " is " << candidates[i] << " = ";
			int_vec_print(cout, v3, 5);
			cout << endl;
		}
		a = O->evaluate_bilinear_form(v1, v3, 1);
		b = O->evaluate_bilinear_form(v2, v3, 1);
		if (a == 0) {
			cout << "a == 0, this should not be" << endl;
			exit(1);
		}
		// <v3,t*v1+v2> = t*<v3,v1>+<v3,v2> = t*a+b = 0
		// Thus, t = -b/a
		t = O->F->mult(O->F->negate(b), O->F->inverse(a));
		c = open_colors_inv[t];
		if (c >= nb_colors) {
			cout << "c >= nb_colors" << endl;
			cout << "i=" << i << endl;
			cout << "candidates[i]=" << candidates[i] << endl;
			cout << "as vector: ";
			int_vec_print(cout, v3, 5);
			cout << endl;
			cout << "a=" << a << endl;
			cout << "b=" << b << endl;
			cout << "t=" << t << endl;
			cout << "c=" << c << endl;
			cout << "nb_colors=" << nb_colors << endl;

			exit(1);
		}
		point_color[i] = c;
	}

	if (f_vv) {
		cout << "point colors:" << endl;
		int_vec_print(cout, point_color, nb_candidates);
		cout << endl;
	}

	FREE_lint(pts_on_special_line);
	FREE_int(starter_t);
	FREE_lint(free_pts);
	FREE_int(open_colors);
	FREE_int(open_colors_inv);
	if (f_v) {
		cout << "blt_set_domain::compute_colors done" << endl;
	}
}



void blt_set_domain::early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, a;
	int f_OK;
	int v[5];
	int *v1, *v2, *v3;
	int m1[5];
	int m3[5];
	int two;
	int fxy, fxz, fyz;

	if (f_v) {
		cout << "blt_set_domain::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		if (f_vv) {
			for (i = 0; i < nb_candidates; i++) {
				O->unrank_point(v, 1, candidates[i],
						0/*verbose_level - 4*/);
				cout << "candidate " << i << "="
						<< candidates[i] << ": ";
				int_vec_print(cout, v, 5);
				cout << endl;
			}
		}
	}
	for (i = 0; i < len; i++) {
		O->unrank_point(Pts + i * 5, 1,
				S[i], 0/*verbose_level - 4*/);
	}
	for (i = 0; i < nb_candidates; i++) {
		O->unrank_point(Candidates + i * 5, 1, candidates[i],
				0/*verbose_level - 4*/);
	}

	two = O->F->add(1, 1);


	if (len == 0) {
		lint_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
	}
	else {
		nb_good_candidates = 0;

		if (f_vv) {
			cout << "blt_set_domain::early_test_func before testing" << endl;
		}
		for (j = 0; j < nb_candidates; j++) {


			if (f_vv) {
				cout << "blt_set::early_test_func "
						"testing " << j << " / "
						<< nb_candidates << endl;
			}

			v1 = Pts;
			v3 = Candidates + j * 5;

			m1[0] = O->F->mult(two, v1[0]);
			m1[1] = v1[2];
			m1[2] = v1[1];
			m1[3] = v1[4];
			m1[4] = v1[3];

			//fxz = evaluate_bilinear_form(v1, v3, 1);
			// too slow !!!
			fxz = O->F->add5(
					O->F->mult(m1[0], v3[0]),
					O->F->mult(m1[1], v3[1]),
					O->F->mult(m1[2], v3[2]),
					O->F->mult(m1[3], v3[3]),
					O->F->mult(m1[4], v3[4])
				);


			if (fxz == 0) {
				f_OK = FALSE;
			}
			else {
				m3[0] = O->F->mult(two, v3[0]);
				m3[1] = v3[2];
				m3[2] = v3[1];
				m3[3] = v3[4];
				m3[4] = v3[3];

				f_OK = TRUE;
				for (i = 1; i < len; i++) {
					//fxy = evaluate_bilinear_form(v1, v2, 1);

					v2 = Pts + i * 5;

					fxy = O->F->add5(
						O->F->mult(m1[0], v2[0]),
						O->F->mult(m1[1], v2[1]),
						O->F->mult(m1[2], v2[2]),
						O->F->mult(m1[3], v2[3]),
						O->F->mult(m1[4], v2[4])
						);

					//fyz = evaluate_bilinear_form(v2, v3, 1);
					fyz = O->F->add5(
							O->F->mult(m3[0], v2[0]),
							O->F->mult(m3[1], v2[1]),
							O->F->mult(m3[2], v2[2]),
							O->F->mult(m3[3], v2[3]),
							O->F->mult(m3[4], v2[4])
						);

					a = O->F->product3(fxy, fxz, fyz);

					if (a == 0) {
						f_OK = FALSE;
						break;
					}
					if (O->f_is_minus_square[a]) {
						f_OK = FALSE;
						break;
					}

				}
			}
			if (f_OK) {
				good_candidates[nb_good_candidates++] =
						candidates[j];
			}
		} // next j
	} // else
}

int blt_set_domain::pair_test(int a, int x, int y, int verbose_level)
// We assume that a is an element
// of a set S of size at least two such that
// S \cup \{ x \} is BLT and
// S \cup \{ y \} is BLT.
// In order to test if S \cup \{ x, y \}
// is BLT, we only need to test
// the triple \{ x,y,a\}
{
	int v1[5], v2[5], v3[5];
	int f12, f13, f23;
	int d;

	O->unrank_point(v1, 1, a, 0);
	O->unrank_point(v2, 1, x, 0);
	O->unrank_point(v3, 1, y, 0);
	f12 = O->evaluate_bilinear_form(v1, v2, 1);
	f13 = O->evaluate_bilinear_form(v1, v3, 1);
	f23 = O->evaluate_bilinear_form(v2, v3, 1);
	d = O->F->product3(f12, f13, f23);
	if (d == 0) {
		return FALSE;
	}
	if (O->f_is_minus_square[d]) {
		return FALSE;
	}
	else {
		return TRUE;
	}

}

int blt_set_domain::check_conditions(int len, long int *S, int verbose_level)
{
	int f_OK = TRUE;
	int f_BLT_test = FALSE;
	int f_collinearity_test = FALSE;
	//int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	//f_v = TRUE;
	//f_vv = TRUE;

	if (f_vv) {
		cout << "checking set ";
		print_set(cout, len, S);
	}
	if (!collinearity_test(S, len, verbose_level)) {
		f_OK = FALSE;
		f_collinearity_test = TRUE;
	}
	if (!O->BLT_test(len, S, verbose_level)) {
		f_OK = FALSE;
		f_BLT_test = TRUE;
	}


	if (f_OK) {
		if (f_vv) {
			cout << "OK" << endl;
		}
		return TRUE;
	}
	else {
		if (f_vv) {
			cout << "not OK because of ";
			if (f_BLT_test) {
				cout << "BLT test";
			}
			if (f_collinearity_test) {
				cout << "collinearity test";
			}
			cout << endl;
		}
		return FALSE;
	}
}

int blt_set_domain::collinearity_test(long int *S, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	long int x, y;
	int f_OK = TRUE;
	int fxy;

	if (f_v) {
		cout << "blt_set_domain::collinearity_test test for" << endl;
		for (i = 0; i < len; i++) {
			O->unrank_point(O->v1, 1, S[i], 0);
			int_vec_print(cout, O->v1, n);
			cout << endl;
		}
	}
	y = S[len - 1];
	O->unrank_point(O->v1, 1, y, 0);

	for (i = 0; i < len - 1; i++) {
		x = S[i];
		O->unrank_point(O->v2, 1, x, 0);
		fxy = O->evaluate_bilinear_form(O->v1, O->v2, 1);

		if (fxy == 0) {
			f_OK = FALSE;
			if (f_v) {
				cout << "not OK; ";
				cout << "{x,y}={" << x << ","
						<< y << "} are collinear" << endl;
				int_vec_print(cout, O->v1, n);
				cout << endl;
				int_vec_print(cout, O->v2, n);
				cout << endl;
				cout << "fxy=" << fxy << endl;
			}
			break;
		}
	}

	if (f_v) {
		if (!f_OK) {
			cout << "collinearity test fails" << endl;
		}
	}
	return f_OK;
}

void blt_set_domain::print(ostream &ost, long int *S, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		O->unrank_point(O->v1, 1, S[i], 0);
		int_vec_print(ost, O->v1, n);
		ost << endl;
	}
}


void blt_set_domain::find_free_points(long int *S, int S_sz,
	long int *&free_pts, int *&free_pt_idx, int &nb_free_pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *lines_on_pt;
	long int *Perp;
	long int i, j, a, b, h, f, fst, len, pt;
	classify C;

	if (f_v) {
		cout << "blt_set_domain::find_free_points" << endl;
	}
	lines_on_pt = NEW_lint(S_sz * (q + 1));
	for (i = 0; i < S_sz; i++) {
		O->lines_on_point_by_line_rank(S[i],
				lines_on_pt + i * (q + 1),
				0 /* verbose_level */);
	}

	if (f_vv) {
		cout << "blt_set_domain::find_free_points "
				"Lines on partial BLT set:" << endl;
		lint_matrix_print(lines_on_pt, S_sz, q + 1);
	}

	Perp = NEW_lint(S_sz * (q + 1) * (q + 1));
	for (i = 0; i < S_sz; i++) {
		for (j = 0; j < q + 1; j++) {
			a = lines_on_pt[i * (q + 1) + j];
			O->points_on_line_by_line_rank(a,
					Perp + i * (q + 1) * (q + 1) + j * (q + 1),
					0 /* verbose_level */);
		}
	}
	if (f_vv) {
		cout << "blt_set_domain::find_free_points Perp:" << endl;
		lint_matrix_print(Perp, S_sz * (q + 1), q + 1);
	}


	C.init_lint(Perp, S_sz * (q + 1) * (q + 1), TRUE, 0);

	C.print(FALSE /* f_reverse */);


	// find the points which are in Perp only once:
	f = C.second_type_first[0];
	nb_free_pts = C.second_type_len[0];
	if (f_v) {
		cout << "blt_set_domain::find_free_points nb_free_pts="
				<< nb_free_pts << endl;
	}
	free_pts = NEW_lint(nb_free_pts);
	free_pt_idx = NEW_int(O->nb_points);
	for (h = 0; h < O->nb_points; h++) {
		free_pt_idx[h] = -1;
	}

	for (h = 0; h < nb_free_pts; h++) {
		b = C.second_sorting_perm_inv[f + h];
		fst = C.type_first[b];
		len = C.type_len[b];
		if (len != 1) {
			cout << "blt_set_domain::find_free_points len != 1" << endl;
			exit(1);
		}
		pt = C.data_sorted[fst];
		//cout << "h=" << h << " b=" << b << " len="
		//<< len << " pt=" << pt << endl;
		free_pts[h] = pt;
		free_pt_idx[pt] = h;
	}

	FREE_lint(lines_on_pt);
	FREE_lint(Perp);

	if (f_v) {
		cout << "blt_set_domain::find_free_points "
				"There are " << nb_free_pts << " free points" << endl;
	}
	if (f_v) {
		cout << "blt_set_domain::find_free_points done" << endl;
	}
}

int blt_set_domain::create_graph(
	int case_number, int nb_cases_total,
	long int *Starter_set, int starter_size,
	long int *candidates, int nb_candidates,
	int f_eliminate_graphs_if_possible,
	colored_graph *&CG,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int special_line;
	int ret = TRUE;

	if (f_v) {
		cout << "blt_set_domain::create_graph" << endl;
	}
	int *point_color;
	int nb_colors;

	long int *lines_on_pt;

	lines_on_pt = NEW_lint(1 /*starter_size*/ * (q + 1));
	O->lines_on_point_by_line_rank(
			Starter_set[0] /*R->rep[0]*/,
			lines_on_pt, 0 /* verbose_level */);

	if (f_vv) {
		cout << "Case " << case_number /*orbit_at_level*/
				<< " Lines on partial BLT set:" << endl;
		lint_matrix_print(lines_on_pt, 1 /*starter_size*/, q + 1);
	}


	special_line = lines_on_pt[0];

	compute_colors(case_number,
			Starter_set, starter_size,
			special_line,
			candidates, nb_candidates,
			point_color, nb_colors,
			verbose_level);


	classify C;

	C.init(point_color, nb_candidates, FALSE, 0);
	if (f_vv) {
		cout << "blt_set_domain::create_graph Case " << case_number
				<< " / " << nb_cases_total /* R->nb_cases*/
				<< " point colors (1st classification): ";
		C.print(FALSE /* f_reverse */);
		cout << endl;
	}


	classify C2;

	C2.init(point_color, nb_candidates, TRUE, 0);
	if (f_vv) {
		cout << "blt_set_domain::create_graph Case " << case_number
				<< " / " << nb_cases_total
				<< " point colors (2nd classification): ";
		C2.print(FALSE /* f_reverse */);
		cout << endl;
	}



	int f, /*l,*/ idx;

	f = C2.second_type_first[0];
	//l = C2.second_type_len[0];
	idx = C2.second_sorting_perm_inv[f + 0];
#if 0
	if (C.type_len[idx] != minimal_type_multiplicity) {
		cout << "idx != minimal_type" << endl;
		cout << "idx=" << idx << endl;
		cout << "minimal_type=" << minimal_type << endl;
		cout << "C.type_len[idx]=" << C.type_len[idx] << endl;
		cout << "minimal_type_multiplicity="
				<< minimal_type_multiplicity << endl;
		exit(1);
	}
#endif
	int minimal_type, minimal_type_multiplicity;

	minimal_type = idx;
	minimal_type_multiplicity = C2.type_len[idx];

	if (f_vv) {
		cout << "blt_set_domain::create_graph Case " << case_number
				<< " / " << nb_cases_total << " minimal type is "
				<< minimal_type << endl;
		cout << "blt_set_domain::create_graph Case " << case_number
				<< " / " << nb_cases_total << " minimal_type_multiplicity "
				<< minimal_type_multiplicity << endl;
	}

	if (f_eliminate_graphs_if_possible) {
		if (minimal_type_multiplicity == 0) {
			cout << "blt_set_domain::create_graph Case " << case_number
					<< " / " << nb_cases_total << " Color class "
					<< minimal_type << " is empty, the case is "
							"eliminated" << endl;
			ret = FALSE;
			goto finish;
		}
	}



	if (f_vv) {
		cout << "blt_set_domain::create_graph Case " << case_number
				<< " / " << nb_cases_total << " Computing adjacency list, "
						"nb_points=" << nb_candidates << endl;
	}

	uchar *bitvector_adjacency;
	long int bitvector_length_in_bits;
	long int bitvector_length;

	compute_adjacency_list_fast(Starter_set[0],
			candidates, nb_candidates, point_color,
			bitvector_adjacency, bitvector_length_in_bits, bitvector_length,
			verbose_level - 2);

	if (f_vv) {
		cout << "blt_set_domain::create_graph Case " << case_number
				<< " / " << nb_cases_total << " Computing adjacency "
						"list done" << endl;
		cout << "blt_set_domain::create_graph Case " << case_number
				<< " / " << nb_cases_total << " bitvector_length="
				<< bitvector_length << endl;
	}


	if (f_v) {
		cout << "blt_set_domain::create_graph creating colored_graph" << endl;
		}

	CG = NEW_OBJECT(colored_graph);

	CG->init(nb_candidates /* nb_points */, nb_colors, 1 /* nb_colors_per_vertex */,
		point_color, bitvector_adjacency, TRUE, verbose_level - 2);
		// the adjacency becomes part of the colored_graph object

	int i;
	for (i = 0; i < nb_candidates; i++) {
		CG->points[i] = candidates[i];
	}
	CG->init_user_data(Starter_set, starter_size, verbose_level - 2);
	snprintf(CG->fname_base, 1000, "graph_BLT_%d_%d_%d",
			q, starter_size, case_number);


	if (f_v) {
		cout << "blt_set_domain::create_graph colored_graph created" << endl;
	}

finish:

	FREE_lint(lines_on_pt);
	FREE_int(point_color);
	if (f_v) {
		cout << "blt_set_domain::create_graph done" << endl;
	}
	return ret;
}


}}
