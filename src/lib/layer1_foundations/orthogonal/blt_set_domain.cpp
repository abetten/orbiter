/*
 * blt_set_domain.cpp
 *
 *  Created on: Apr 6, 2019
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orthogonal_geometry {



blt_set_domain::blt_set_domain()
{
	F = NULL;
	f_semilinear = false;
	epsilon = 0;
	n = 0;
	q = 0;
	target_size = 0;
	nb_points_on_quadric = 0;
	max_degree = 0;


	O = NULL;
	f_orthogonal_allocated = false;
	Pts = NULL;
	Candidates = NULL;
	P = NULL;
	G53 = NULL;
	G54 = NULL;
	G43 = NULL;

	Q2 = 0;
	F2 = NULL;
	Poly2 = NULL;

	Q3 = 0;
	F3 = NULL;
	Poly3 = NULL;

	//null();
}

blt_set_domain::~blt_set_domain()
{
	int f_v = false;

	if (f_v) {
		cout << "blt_set_domain::~blt_set_domain" << endl;
	}
	if (f_orthogonal_allocated) {
		if (f_v) {
			cout << "blt_set_domain::~blt_set_domain before O" << endl;
		}
		if (O) {
			FREE_OBJECT(O);
		}
		f_orthogonal_allocated = false;
		O = NULL;
	}
	if (Pts) {
		FREE_int(Pts);
	}
	if (Candidates) {
		FREE_int(Candidates);
	}
	if (G43) {
		FREE_OBJECT(G43);
	}

	if (F2) {
		FREE_OBJECT(F2);
	}
	if (Poly2) {
		FREE_OBJECT(Poly2);
	}
	if (F3) {
		FREE_OBJECT(F3);
	}
	if (Poly3) {
		FREE_OBJECT(Poly3);
	}
	if (f_v) {
		cout << "blt_set_domain::~blt_set_domain done" << endl;
	}
}



void blt_set_domain::init_blt_set_domain(
		orthogonal *O,
		geometry::projective_space *P4,
		int f_create_extension_fields,
	int verbose_level)
// creates a grassmann G43.
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "blt_set_domain::init_blt_set_domain" << endl;
		cout << "blt_set_domain::init_blt_set_domain "
				"verbose_level = " << verbose_level << endl;
	}


	blt_set_domain::O = O;
	f_orthogonal_allocated = false;
	blt_set_domain::F = O->F;
	blt_set_domain::q = F->q;
	n = O->Quadratic_form->n; // vector space dimension
	epsilon = O->Quadratic_form->epsilon;


	max_degree = 1 * (q - 1);


	prefix = "BLT_q" + std::to_string(q);


	target_size = q + 1;
	nb_points_on_quadric = O->Hyperbolic_pair->nb_points;


	if (f_v) {
		cout << "blt_set_domain::init_blt_set_domain q=" << q
				<< " target_size = " << target_size << endl;
	}


	f_semilinear = true;
	if (NT.is_prime(q)) {
		f_semilinear = false;
	}
	if (f_v) {
		cout << "blt_set_domain::init_blt_set_domain "
				"f_semilinear=" << f_semilinear << endl;
	}


	if (f_v) {
		cout << "blt_set_domain::init_blt_set_domain "
				"allocating Pts and Candidates" << endl;
	}


	Pts = NEW_int(target_size * n);

	Candidates = NEW_int(nb_points_on_quadric * n);


	P = P4;

	G53 = P->Subspaces->Grass_planes;
	G54 = P->Subspaces->Grass_hyperplanes;



	G43 = NEW_OBJECT(geometry::grassmann);

	if (f_v) {
		cout << "blt_set_domain::init_blt_set_domain "
				"before G43->init" << endl;
	}
	G43->init(4, 3, F, 0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "blt_set_domain::init_blt_set_domain "
				"after G43->init" << endl;
	}

	if (f_create_extension_fields) {

		if (f_v) {
			cout << "blt_set_domain::init_blt_set_domain "
					"before create_extension_fields" << endl;
		}
		create_extension_fields(verbose_level);
		if (f_v) {
			cout << "blt_set_domain::init_blt_set_domain "
					"after create_extension_fields" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "blt_set_domain::init_blt_set_domain "
					"we are not creating any extension fields" << endl;
		}

	}


	if (f_v) {
		cout << "blt_set_domain::init_blt_set_domain finished" << endl;
	}
}

void blt_set_domain::create_extension_fields(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "blt_set_domain::create_extension_fields" << endl;
	}

	Q2 = q * q;

	F2 = NEW_OBJECT(field_theory::finite_field);


	F2->finite_field_init_small_order(Q2,
			false /* f_without_tables */,
			true /* f_compute_related_fields */,
			verbose_level);




	Poly2 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "blt_set_domain::create_extension_fields "
				"before Poly2->init" << endl;
	}
	Poly2->init(F2, 2, max_degree,
				t_PART,
				verbose_level);
	if (f_v) {
		cout << "blt_set_domain::create_extension_fields "
				"after Poly2->init" << endl;
	}


	Q3 = q * q * q;

	F3 = NEW_OBJECT(field_theory::finite_field);


	F3->finite_field_init_small_order(Q3,
			false /* f_without_tables */,
			true /* f_compute_related_fields */,
			verbose_level);



	Poly3 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "blt_set_domain::create_extension_fields "
				"before Poly3->init" << endl;
	}
	Poly3->init(F3, 2, max_degree,
				t_PART,
				verbose_level);
	if (f_v) {
		cout << "blt_set_domain::create_extension_fields "
				"after Poly3->init" << endl;
	}



	if (f_v) {
		cout << "blt_set_domain::create_extension_fields done" << endl;
	}
}

long int blt_set_domain::intersection_of_hyperplanes(
		long int plane_rk1, long int plane_rk2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int plane_rk = 0;
	int B1[4 * 5];
	int B2[4 * 5];
	int C[5 * 5];
	int rk_int;

	if (f_v) {
		cout << "blt_set_domain::intersection_of_hyperplanes" << endl;
	}

	G54->unrank_lint_here(B1, plane_rk1, verbose_level);
	G54->unrank_lint_here(B2, plane_rk2, verbose_level);

	F->Linear_algebra->intersect_subspaces(5, 4,
		B1, 4, B2,
		rk_int, C, 0 /* verbose_level */);

	if (rk_int != 3) {
		cout << "blt_set_domain::intersection_of_hyperplanes "
				"rk_int != 3" << endl;
		exit(1);
	}

	plane_rk = G53->rank_lint_here(C, verbose_level);


	if (f_v) {
		cout << "blt_set_domain::intersection_of_hyperplanes done" << endl;
	}
	return plane_rk;
}


long int blt_set_domain::compute_tangent_hyperplane(
	long int pt,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int plane_rk = 0;

	if (f_v) {
		cout << "blt_set_domain::compute_tangent_hyperplane" << endl;
	}

	int B[5 * 5];

	O->Hyperbolic_pair->unrank_point(B, 1, pt, 0);

	if (f_v) {
		cout << "blt_set_domain::compute_tangent_hyperplane "
				"before F->Linear_algebra->perp" << endl;
	}
	F->Linear_algebra->perp(
			5, 1, B,
			O->Quadratic_form->Gram_matrix,
			0 /* verbose_level */);
	if (f_v) {
		cout << "blt_set_domain::compute_tangent_hyperplane "
				"after F->Linear_algebra->perp" << endl;
	}
	if (f_v) {
		cout << "blt_set_domain::compute_tangent_hyperplane "
				"the matrix B is:" << endl;
		Int_vec_print_integer_matrix(cout, B, 5, 5);
	}

	plane_rk = G54->rank_lint_here(B + 5, verbose_level);


	if (f_v) {
		cout << "blt_set_domain::compute_tangent_hyperplane done" << endl;
	}
	return plane_rk;
}


void blt_set_domain::report_given_point_set(
		std::ostream &ost,
		long int *Pts, int nb_pts, int verbose_level)
{
	long int pt_rk, hyperplane_rk;
	int i;

	ost << "A set of points of size " << nb_pts << "\\\\" << endl;
	ost << "The Points:\\\\" << endl;
	for (i = 0; i < nb_pts; i++) {
		pt_rk = Pts[i];


		O->Hyperbolic_pair->unrank_point(
				O->Hyperbolic_pair->v1, 1, pt_rk, 0 /*verbose_level*/);

		hyperplane_rk = compute_tangent_hyperplane(
			pt_rk,
			0 /* verbose_level */);

		ost << i << " : $P_{" << pt_rk << "} = ";
		Int_vec_print(ost, O->Hyperbolic_pair->v1, O->Quadratic_form->n);
		ost << ", T_{" << hyperplane_rk << "}$\\\\" << endl;
	}
	//ost << endl;
}




void blt_set_domain::compute_adjacency_list_fast(
	int first_point_of_starter,
	long int *points, int nb_points, int *point_color,
	data_structures::bitvector *&Bitvec,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int L;
	long int L100;
	long int i, j, k;
	int c1, c2;
	int *Pts;
	int *form_value;
	int v1[5];
	int m[5];
	int f12, f13, f23, d;
	//long int cnt;
	int two;
	int *Pi, *Pj;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "blt_set_domain::compute_adjacency_list_fast" << endl;
	}

	L = ((long int) nb_points * ((long int) nb_points - 1)) >> 1;
	L100 = (L / 100) + 1;

	Bitvec = NEW_OBJECT(data_structures::bitvector);
	Bitvec->allocate(L);

	Pts = NEW_int(nb_points * 5);
	form_value = NEW_int(nb_points);

	O->Hyperbolic_pair->unrank_point(
			v1, 1, first_point_of_starter, 0);

	if (f_v) {
		cout << "blt_set_domain::compute_adjacency_list_fast "
				"unranking points" << endl;
	}
	for (i = 0; i < nb_points; i++) {

		O->Hyperbolic_pair->unrank_point(
				Pts + i * 5, 1, points[i], 0);

		form_value[i] = O->Quadratic_form->evaluate_bilinear_form(
				v1, Pts + i * 5, 1);
	}

	if (f_v) {
		cout << "blt_set_domain::compute_adjacency_list_fast "
				"computing adjacency matrix" << endl;
	}

	//cnt = 0;
	two = F->add(1, 1);

	k = 0;

	for (i = 0; i < nb_points; i++) {
		f12 = form_value[i];
		c1 = point_color[i];
		Pi = Pts + i * 5;
		m[0] = F->mult(Pi[0], two);
		m[1] = Pi[2];
		m[2] = Pi[1];
		m[3] = Pi[4];
		m[4] = Pi[3];

		for (j = i + 1; j < nb_points; j++, /*cnt++,*/ k++) {
			//k = Combi.ij2k_lint(i, j, nb_points);

			if (f_vv && (k % L100) == 0 && k) {
				cout << "blt_set_domain::compute_adjacency_list_fast "
						"nb_points=" << nb_points << " progress: "
						<< k / L100
						<< " %" << endl;
			}
			c2 = point_color[j];
			if (c1 == c2) {
				Bitvec->m_i(k, 0);
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
				Bitvec->m_i(k, 0);
			}
			else {
				if (O->SN->f_is_minus_square[d]) {
					Bitvec->m_i(k, 0);
				}
				else {
					Bitvec->m_i(k, 1);
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



void blt_set_domain::compute_colors(
		int orbit_at_level,
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
	data_structures::sorting Sorting;


	if (f_v) {
		cout << "blt_set_domain::compute_colors" << endl;
	}
	O->Hyperbolic_pair->unrank_line(p1, p2,
			special_line, 0/*verbose_level*/);
	if (f_vv) {
		cout << "after unrank_line " << special_line << ":" << endl;
		cout << "p1=" << p1 << " p2=" << p2 << endl;
	}
	O->Hyperbolic_pair->unrank_point(v1, 1, p1, 0);
	O->Hyperbolic_pair->unrank_point(v2, 1, p2, 0);
	if (f_vv) {
		cout << "p1=" << p1 << " ";
		Int_vec_print(cout, v1, 5);
		cout << endl;
		cout << "p2=" << p2 << " ";
		Int_vec_print(cout, v2, 5);
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
		Lint_vec_print(cout, pts_on_special_line, q + 1);
		cout << endl;
	}

	if (!Sorting.lint_vec_search(
			pts_on_special_line,
			q + 1, starter[0], idx, 0)) {
		cout << "cannot find the first point on the line" << endl;
		exit(1);
	}
	for (i = idx; i < q + 1; i++) {
		pts_on_special_line[i] = pts_on_special_line[i + 1];
	}
	if (f_vv) {
		cout << "pts_on_special_line without the first "
				"starter point:" << endl;
		Lint_vec_print(cout, pts_on_special_line, q);
		cout << endl;
	}

	int a, b;
	int t, c, j, h;
	int *starter_t;

	starter_t = NEW_int(starter_sz);
	starter_t[0] = -1;
	for (i = 1; i < starter_sz; i++) {
		O->Hyperbolic_pair->unrank_point(
				v3, 1, starter[i], 0);
		a = O->Quadratic_form->evaluate_bilinear_form(
				v1, v3, 1);
		b = O->Quadratic_form->evaluate_bilinear_form(
				v2, v3, 1);
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
		Int_vec_print(cout, starter_t, starter_sz);
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
		cout << "blt_set_domain::compute_colors error: "
				"j != nb_colors" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "The " << nb_colors << " free points are :" << endl;
		Lint_vec_print(cout, free_pts, nb_colors);
		cout << endl;
		cout << "The " << nb_colors << " open colors are :" << endl;
		Int_vec_print(cout, open_colors, nb_colors);
		cout << endl;
	}
	for ( ; j < q; j++) {
		open_colors[j] = starter_t[j - nb_colors + 1];
	}
	if (f_vv) {
		cout << "open_colors :" << endl;
		Int_vec_print(cout, open_colors, q);
		cout << endl;
	}
	for (i = 0; i < q; i++) {
		j = open_colors[i];
		open_colors_inv[j] = i;
	}
	if (f_vv) {
		cout << "open_colors_inv :" << endl;
		Int_vec_print(cout, open_colors_inv, q);
		cout << endl;
	}


	for (i = 0; i < nb_candidates; i++) {
		O->Hyperbolic_pair->unrank_point(v3, 1, candidates[i], 0);
		if (f_vv) {
			cout << "candidate " << i << " / " << nb_candidates
					<< " is " << candidates[i] << " = ";
			Int_vec_print(cout, v3, 5);
			cout << endl;
		}
		a = O->Quadratic_form->evaluate_bilinear_form(
				v1, v3, 1);
		b = O->Quadratic_form->evaluate_bilinear_form(
				v2, v3, 1);
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
			Int_vec_print(cout, v3, 5);
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
		Int_vec_print(cout, point_color, nb_candidates);
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



void blt_set_domain::early_test_func(
		long int *S, int len,
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
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		if (f_vv) {
			for (i = 0; i < nb_candidates; i++) {
				O->Hyperbolic_pair->unrank_point(
						v, 1, candidates[i],
						0/*verbose_level - 4*/);
				cout << "candidate " << i << "="
						<< candidates[i] << ": ";
				Int_vec_print(cout, v, 5);
				cout << endl;
			}
		}
	}
	if (f_v) {
		cout << "blt_set_domain::early_test_func "
				"unranking points" << endl;
	}
	for (i = 0; i < len; i++) {
		O->Hyperbolic_pair->unrank_point(
				Pts + i * 5, 1,
				S[i], 0/*verbose_level - 4*/);
	}
	if (f_v) {
		cout << "blt_set_domain::early_test_func "
				"unranking candidates" << endl;
	}
	for (i = 0; i < nb_candidates; i++) {
		O->Hyperbolic_pair->unrank_point(
				Candidates + i * 5, 1, candidates[i],
				0/*verbose_level - 4*/);
	}
	if (f_v) {
		cout << "blt_set_domain::early_test_func "
				"unranking candidates done" << endl;
	}

	if (f_v) {
		cout << "blt_set_domain::early_test_func "
				"computing two" << endl;
	}
	two = O->F->add(1, 1);
	if (f_v) {
		cout << "blt_set_domain::early_test_func "
				"after computing two" << endl;
	}


	if (len == 0) {
		if (f_v) {
			cout << "blt_set_domain::early_test_func "
					"len == 0, copying candidates" << endl;
		}
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
		if (f_v) {
			cout << "blt_set_domain::early_test_func "
					"after copying candidates" << endl;
		}
		nb_good_candidates = nb_candidates;
	}
	else {
		nb_good_candidates = 0;

		if (f_vv) {
			cout << "blt_set_domain::early_test_func "
					"before testing" << endl;
		}
		for (j = 0; j < nb_candidates; j++) {


			if (f_vv) {
				cout << "blt_set_domain::early_test_func "
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
				f_OK = false;
			}
			else {
				m3[0] = O->F->mult(two, v3[0]);
				m3[1] = v3[2];
				m3[2] = v3[1];
				m3[3] = v3[4];
				m3[4] = v3[3];

				f_OK = true;
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
						f_OK = false;
						break;
					}
					if (O->SN->f_is_minus_square[a]) {
						f_OK = false;
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

int blt_set_domain::pair_test(
		int a, int x, int y, int verbose_level)
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

	O->Hyperbolic_pair->unrank_point(v1, 1, a, 0);
	O->Hyperbolic_pair->unrank_point(v2, 1, x, 0);
	O->Hyperbolic_pair->unrank_point(v3, 1, y, 0);
	f12 = O->Quadratic_form->evaluate_bilinear_form(v1, v2, 1);
	f13 = O->Quadratic_form->evaluate_bilinear_form(v1, v3, 1);
	f23 = O->Quadratic_form->evaluate_bilinear_form(v2, v3, 1);
	d = O->F->product3(f12, f13, f23);
	if (d == 0) {
		return false;
	}
	if (O->SN->f_is_minus_square[d]) {
		return false;
	}
	else {
		return true;
	}

}

int blt_set_domain::check_conditions(
		int len, long int *S, int verbose_level)
{
	int f_OK = true;
	int f_BLT_test = false;
	int f_collinearity_test = false;
	//int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	orthogonal_global OG;

	//f_v = true;
	//f_vv = true;

	if (f_vv) {
		cout << "checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	if (!collinearity_test(S, len, 0 /*verbose_level*/)) {
		f_OK = false;
		f_collinearity_test = true;
	}
	if (!OG.BLT_test(O, len, S, 0 /*verbose_level*/)) {
		f_OK = false;
		f_BLT_test = true;
	}


	if (f_OK) {
		if (f_vv) {
			cout << "OK" << endl;
		}
		return true;
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
		return false;
	}
}

int blt_set_domain::collinearity_test(
		long int *S, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	long int x, y;
	int f_OK = true;
	int fxy;

	if (f_v) {
		cout << "blt_set_domain::collinearity_test for" << endl;
		for (i = 0; i < len; i++) {
			O->Hyperbolic_pair->unrank_point(
					O->Hyperbolic_pair->v1, 1, S[i], 0);
			Int_vec_print(cout, O->Hyperbolic_pair->v1, n);
			cout << endl;
		}
	}
	y = S[len - 1];
	O->Hyperbolic_pair->unrank_point(
			O->Hyperbolic_pair->v1, 1, y, 0);

	for (i = 0; i < len - 1; i++) {

		x = S[i];

		O->Hyperbolic_pair->unrank_point(
				O->Hyperbolic_pair->v2, 1, x, 0);

		fxy = O->Quadratic_form->evaluate_bilinear_form(
				O->Hyperbolic_pair->v1,
				O->Hyperbolic_pair->v2, 1);

		if (fxy == 0) {

			f_OK = false;

			if (f_v) {
				cout << "not OK; ";
				cout << "{x,y}={" << x << ","
						<< y << "} are collinear" << endl;
				Int_vec_print(cout, O->Hyperbolic_pair->v1, n);
				cout << endl;
				Int_vec_print(cout, O->Hyperbolic_pair->v2, n);
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

void blt_set_domain::print(
		std::ostream &ost, long int *S, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		O->Hyperbolic_pair->unrank_point(
				O->Hyperbolic_pair->v1, 1, S[i], 0);
		Int_vec_print(ost, O->Hyperbolic_pair->v1, n);
		ost << endl;
	}
}


void blt_set_domain::find_free_points(
		long int *S, int S_sz,
	long int *&free_pts, int *&free_pt_idx, int &nb_free_pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *lines_on_pt;
	long int *Perp;
	long int i, j, a, b, h, f, fst, len, pt;
	data_structures::tally C;

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
		Lint_matrix_print(lines_on_pt, S_sz, q + 1);
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
		Lint_matrix_print(Perp, S_sz * (q + 1), q + 1);
	}


	C.init_lint(Perp, S_sz * (q + 1) * (q + 1), true, 0);

	C.print(false /* f_reverse */);


	// find the points which are in Perp only once:
	f = C.second_type_first[0];
	nb_free_pts = C.second_type_len[0];
	if (f_v) {
		cout << "blt_set_domain::find_free_points nb_free_pts="
				<< nb_free_pts << endl;
	}
	free_pts = NEW_lint(nb_free_pts);
	free_pt_idx = NEW_int(O->Hyperbolic_pair->nb_points);
	for (h = 0; h < O->Hyperbolic_pair->nb_points; h++) {
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
	graph_theory::colored_graph *&CG,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int special_line;
	int ret = true;

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
		Lint_matrix_print(lines_on_pt, 1 /*starter_size*/, q + 1);
	}


	special_line = lines_on_pt[0];

	if (f_v) {
		cout << "blt_set_domain::create_graph "
				"before compute_colors" << endl;
	}
	compute_colors(
			case_number,
			Starter_set, starter_size,
			special_line,
			candidates, nb_candidates,
			point_color, nb_colors,
			verbose_level);
	if (f_v) {
		cout << "blt_set_domain::create_graph "
				"after compute_colors" << endl;
	}


	data_structures::tally C;

	C.init(point_color, nb_candidates, false, 0);
	if (f_vv) {
		cout << "blt_set_domain::create_graph Case " << case_number
				<< " / " << nb_cases_total /* R->nb_cases*/
				<< " point colors (1st classification): ";
		C.print(false /* f_reverse */);
		cout << endl;
	}


	data_structures::tally C2;

	C2.init(point_color, nb_candidates, true, 0);
	if (f_vv) {
		cout << "blt_set_domain::create_graph Case " << case_number
				<< " / " << nb_cases_total
				<< " point colors (2nd classification): ";
		C2.print(false /* f_reverse */);
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
			ret = false;
			goto finish;
		}
	}



	if (f_vv) {
		cout << "blt_set_domain::create_graph Case " << case_number
				<< " / " << nb_cases_total
				<< " Computing adjacency list, "
						"nb_points=" << nb_candidates << endl;
	}

	data_structures::bitvector *Bitvec;

	if (f_v) {
		cout << "blt_set_domain::create_graph "
				"before compute_adjacency_list_fast" << endl;
	}
	compute_adjacency_list_fast(
			Starter_set[0],
			candidates, nb_candidates, point_color,
			Bitvec,
			verbose_level - 2);

	if (f_vv) {
		cout << "blt_set_domain::create_graph Case " << case_number
				<< " / " << nb_cases_total << " Computing adjacency "
						"list done" << endl;
#if 0
		cout << "blt_set_domain::create_graph Case " << case_number
				<< " / " << nb_cases_total << " bitvector_length="
				<< bitvector_length << endl;
#endif
	}


	if (f_v) {
		cout << "blt_set_domain::create_graph "
				"creating colored_graph" << endl;
		}

	CG = NEW_OBJECT(graph_theory::colored_graph);

	{
		string label, label_tex;

		label = "BLT_" + std::to_string(case_number);
		label_tex = "BLT\\_" + std::to_string(case_number);

		CG->init(
				nb_candidates /* nb_points */,
				nb_colors, 1 /* nb_colors_per_vertex */,
				point_color, Bitvec, true,
				label, label_tex,
				verbose_level - 2);
			// Bitvec becomes part of the colored_graph object
	}
	int i;
	for (i = 0; i < nb_candidates; i++) {
		CG->points[i] = candidates[i];
	}

	{

		std::string fname;

		fname = prefix + "_graph_" + std::to_string(starter_size) + "_" + std::to_string(case_number);
		CG->init_user_data(
				Starter_set, starter_size, verbose_level - 2);
		CG->fname_base.assign(fname);
	}


	if (f_v) {
		cout << "blt_set_domain::create_graph "
				"the colored_graph has been created" << endl;
	}

finish:

	FREE_lint(lines_on_pt);
	FREE_int(point_color);
	if (f_v) {
		cout << "blt_set_domain::create_graph done" << endl;
	}
	return ret;
}


void blt_set_domain::test_flock_condition(
		field_theory::finite_field *F,
		int *ABC,
		int *&outcome,
		int &N,
		int verbose_level)
// F is given because the field might be
// an extension field of the current field
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);

	if (f_v) {
		cout << "blt_set_domain::test_flock_condition" << endl;
	}

	int q;
	//int N;
	int i, j;
	int ai, bi, ci;
	int aj, bj, cj;
	int a, b, c;
	int two, four;
	int x;
	//int *outcome;
	int cnt;



	q = F->q;

	two = F->add(1, 1);
	four = F->add(two, two);

	N = (q * (q - 1)) >> 1;
	outcome = NEW_int(N);

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


	//FREE_int(outcome);
	if (f_v) {
		cout << "blt_set_domain::test_flock_condition done" << endl;
	}

}

void blt_set_domain::quadratic_lift(
		int *coeff_f, int *coeff_g, int nb_coeff,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 2);

	if (f_v) {
		cout << "blt_set_domain::quadratic_lift" << endl;
		cout << "blt_set_domain::quadratic_lift "
				"nb_coeff = " << nb_coeff << endl;
		cout << "blt_set_domain::quadratic_lift "
				"Poly2->get_nb_monomials() = "
				<< Poly2->get_nb_monomials() << endl;
	}


	if (f_v) {
		cout << "blt_set_domain::quadratic_lift Q2=" << Q2 << endl;
	}

	if (f_v) {

		cout << "blt_set_domain::quadratic_lift coeff_f:" << endl;
		Int_vec_print(cout, coeff_f, nb_coeff);
		cout << endl;

		cout << "blt_set_domain::quadratic_lift coeff_g:" << endl;
		Int_vec_print(cout, coeff_g, nb_coeff);
		cout << endl;

	}


	int *lifted_f;
	int *lifted_g;


	lifted_f = NEW_int(Q2);
	lifted_g = NEW_int(Q2);

	int v[2];
	int i;

	for (i = 0; i < Q2; i++) {
		//Gg.AG_element_unrank(Q, v, 1, 1, i);
		v[0] = i;
		v[1] = 1;
		lifted_f[i] = Poly2->evaluate_at_a_point(coeff_f, v);
		lifted_g[i] = Poly2->evaluate_at_a_point(coeff_g, v);
	}

	if (f_v) {

		cout << "blt_set_domain::quadratic_lift lifted_f:" << endl;
		Int_vec_print(cout, lifted_f, Q2);
		cout << endl;

		cout << "blt_set_domain::quadratic_lift lifted_g:" << endl;
		Int_vec_print(cout, lifted_g, Q2);
		cout << endl;
	}

	int *ABC2;

	ABC2 = NEW_int(Q2 * 3);
	for (i = 0; i < Q2; i++) {
		ABC2[i * 3 + 0] = i;
		ABC2[i * 3 + 1] = lifted_f[i];
		ABC2[i * 3 + 2] = lifted_g[i];
	}

	int *outcome;
	int N;

	test_flock_condition(F2, ABC2, outcome, N, verbose_level);

	data_structures::tally T;

	T.init(outcome, N, false, 0);
	cout << "outcome : ";
	T.print_first(false /*f_backwards*/);
	cout << endl;

	FREE_int(outcome);

	FREE_int(ABC2);
	FREE_int(lifted_f);
	FREE_int(lifted_g);


	if (f_v) {
		cout << "blt_set_domain::quadratic_lift done" << endl;
	}
}



void blt_set_domain::cubic_lift(
		int *coeff_f, int *coeff_g, int nb_coeff,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 2);

	if (f_v) {
		cout << "blt_set_domain::cubic_lift" << endl;
		cout << "blt_set_domain::cubic_lift "
				"nb_coeff = " << nb_coeff << endl;
		cout << "blt_set_domain::cubic_lift "
				"Poly3->get_nb_monomials() = "
				<< Poly3->get_nb_monomials() << endl;
	}



	int *lifted_f;
	int *lifted_g;


	lifted_f = NEW_int(Q3);
	lifted_g = NEW_int(Q3);

	int v[2];
	int i;

	for (i = 0; i < Q3; i++) {
		//Gg.AG_element_unrank(Q, v, 1, 1, i);
		v[0] = i;
		v[1] = 1;
		lifted_f[i] = Poly3->evaluate_at_a_point(coeff_f, v);
		lifted_g[i] = Poly3->evaluate_at_a_point(coeff_g, v);
	}

	if (f_v) {

		cout << "blt_set_domain::cubic_lift lifted_f:" << endl;
		Int_vec_print(cout, lifted_f, Q3);
		cout << endl;

		cout << "blt_set_domain::cubic_lift lifted_g:" << endl;
		Int_vec_print(cout, lifted_g, Q3);
		cout << endl;
	}

	int *ABC2;

	ABC2 = NEW_int(Q3 * 3);
	for (i = 0; i < Q3; i++) {
		ABC2[i * 3 + 0] = i;
		ABC2[i * 3 + 1] = lifted_f[i];
		ABC2[i * 3 + 2] = lifted_g[i];
	}

	int *outcome;
	int N;

	test_flock_condition(F3, ABC2, outcome, N, verbose_level);

	data_structures::tally T;

	T.init(outcome, N, false, 0);
	cout << "outcome : ";
	T.print_first(false /*f_backwards*/);
	cout << endl;

	FREE_int(outcome);


	string fname_csv;
	orbiter_kernel_system::file_io Fio;


	fname_csv = "ABC_q" + std::to_string(Q3) + ".csv";

	Fio.int_matrix_write_csv(fname_csv, ABC2, Q3, 3);

	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	FREE_int(ABC2);
	FREE_int(lifted_f);
	FREE_int(lifted_g);

	if (f_v) {
		cout << "blt_set_domain::cubic_lift done" << endl;
	}
}


}}}

