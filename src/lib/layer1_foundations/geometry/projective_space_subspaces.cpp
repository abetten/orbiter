/*
 * projective_space_subspaces.cpp
 *
 *  Created on: Mar 10, 2023
 *      Author: betten
 */






#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry {


projective_space_subspaces::projective_space_subspaces()
{
	P = NULL;

	Grass_lines = NULL;
	Grass_planes = NULL;
	Grass_hyperplanes = NULL;

	Grass_stack = NULL;

	F = NULL;
	//Go = NULL;
	n = 0;
	q = 0;

	N_points = 0;
	N_lines = 0;

	Nb_subspaces = NULL;

	r = 0;
	k = 0;

	v = NULL;
	w = NULL;
	Mtx = NULL;
	Mtx2 = NULL;

	Implementation = NULL;

	Standard_polarity = NULL;
	Reversal_polarity = NULL;

}

projective_space_subspaces::~projective_space_subspaces()
{
	int f_v = false;

	if (f_v) {
		cout << "projective_space_subspaces::~projective_space_subspaces" << endl;
	}

	P = NULL;

	if (Grass_lines) {
		if (f_v) {
			cout << "projective_space_subspaces::~projective_space_subspaces "
					"deleting Grass_lines" << endl;
		}
		FREE_OBJECT(Grass_lines);
	}
	if (Grass_planes) {
		if (f_v) {
			cout << "projective_space_subspaces::~projective_space_subspaces "
					"deleting Grass_planes" << endl;
		}
		FREE_OBJECT(Grass_planes);
	}
	if (Grass_hyperplanes) {
		if (f_v) {
			cout << "projective_space_subspaces::~projective_space_subspaces "
					"deleting Grass_hyperplanes" << endl;
		}
		FREE_OBJECT(Grass_hyperplanes);
	}
	if (Grass_stack) {
		int i;

		for (i = 1; i < n + 1; i++) {
			FREE_OBJECT(Grass_stack[i]);
		}
		FREE_pvoid((void **) Grass_stack);
	}
	if (Nb_subspaces) {
		FREE_lint(Nb_subspaces);
	}

	if (v) {
		if (f_v) {
			cout << "projective_space_subspaces::~projective_space_subspaces deleting v" << endl;
		}
		FREE_int(v);
	}
	if (w) {
		FREE_int(w);
	}
	if (Mtx) {
		FREE_int(Mtx);
	}
	if (Mtx2) {
		FREE_int(Mtx2);
	}
	if (Implementation) {
		FREE_OBJECT(Implementation);
	}

	if (Standard_polarity) {
		FREE_OBJECT(Standard_polarity);
	}
	if (Reversal_polarity) {
		FREE_OBJECT(Reversal_polarity);
	}

	if (f_v) {
		cout << "projective_space_subspaces::~projective_space_subspaces done" << endl;
	}
}


void projective_space_subspaces::init(
	projective_space *P,
	int n,
	field_theory::finite_field *F,
	int f_init_incidence_structure,
	int verbose_level)
// n is projective dimension
{

	int f_v = (verbose_level >= 1);
	int i;
	combinatorics::combinatorics_domain C;
	ring_theory::longinteger_object a;

	if (f_v) {
		cout << "projective_space_subspaces::init" << endl;
	}
	projective_space_subspaces::P = P;
	if (f_v) {
		cout << "projective_space_subspaces::init done" << endl;
	}

	projective_space_subspaces::n = n;
	projective_space_subspaces::F = F;
	projective_space_subspaces::q = F->q;

	if (f_v) {
		cout << "projective_space_subspaces::init "
				"PG(" << n << "," << q << ")" << endl;
		cout << "f_init_incidence_structure="
				<< f_init_incidence_structure << endl;
	}


	v = NEW_int(n + 1);
	w = NEW_int(n + 1);
	Mtx = NEW_int(3 * (n + 1));
	Mtx2 = NEW_int(3 * (n + 1));

	Grass_lines = NEW_OBJECT(grassmann);
	Grass_lines->init(n + 1, 2, F, verbose_level - 2);
	if (n > 2) {
		Grass_planes = NEW_OBJECT(grassmann);
		Grass_planes->init(n + 1, 3, F, verbose_level - 2);
	}
	if (n >= 2) {
		Grass_hyperplanes = NEW_OBJECT(grassmann);
		Grass_hyperplanes->init(n + 1, n, F, verbose_level - 2);
	}

	Grass_stack = (grassmann **) NEW_pvoid(n + 1);
	Grass_stack[0] = NULL;
	for (i = 1; i < n + 1; i++) {
		Grass_stack[i] = NEW_OBJECT(grassmann);
		if (f_v) {
			cout << "projective_space_subspaces::init "
					"before Grass_stack[i]->init i=" << i << endl;
		}
		Grass_stack[i]->init(n + 1, i, F, verbose_level - 2);
	}

	if (f_v) {
		cout << "projective_space_subspaces::init "
				"computing number of "
				"subspaces of each dimension:" << endl;
	}
	Nb_subspaces = NEW_lint(n + 1);
	if (n < 10) {
		for (i = 0; i <= n; i++) {
			if (f_v) {
				cout << "projective_space_subspaces::init "
						"computing number of "
						"subspaces of dimension " << i + 1 << endl;
			}
			C.q_binomial_no_table(
				a,
				n + 1, i + 1, q, 0 /*verbose_level - 2*/);
			Nb_subspaces[i] = a.as_lint();
			//Nb_subspaces[i] = generalized_binomial(n + 1, i + 1, q);
		}

		C.q_binomial_no_table(
			a,
			n, 1, q, 0 /*verbose_level - 2*/);
		r = a.as_int();
		//r = generalized_binomial(n, 1, q);
	}
	else {
		for (i = 0; i <= n; i++) {
			if (f_v) {
				cout << "projective_space_subspaces::init "
						"computing number of "
						"subspaces of dimension " << i + 1 << endl;
				}
			Nb_subspaces[i] = 0;
		}
		r = 0;
	}
	N_points = Nb_subspaces[0]; // generalized_binomial(n + 1, 1, q);
	if (f_v) {
		cout << "projective_space_subspaces::init N_points=" << N_points << endl;
	}
	N_lines = Nb_subspaces[1]; // generalized_binomial(n + 1, 2, q);
	if (f_v) {
		cout << "projective_space_subspaces::init N_lines=" << N_lines << endl;
	}
	if (f_v) {
		cout << "projective_space_subspaces::init r=" << r << endl;
	}
	k = q + 1; // number of points on a line
	if (f_v) {
		cout << "projective_space_subspaces::init k=" << k << endl;
	}

	if (f_init_incidence_structure) {
		if (f_v) {
			cout << "projective_space_subspaces::init calling "
					"init_incidence_structure" << endl;
		}
		init_incidence_structure(verbose_level);
		if (f_v) {
			cout << "projective_space_subspaces::init "
					"init_incidence_structure done" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "projective_space_subspaces::init we don't initialize "
					"the incidence structure data" << endl;
		}
	}



}

void projective_space_subspaces::init_incidence_structure(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_subspaces::init_incidence_structure" << endl;
	}

	Implementation = NEW_OBJECT(projective_space_implementation);


	if (f_v) {
		cout << "projective_space_subspaces::init_incidence_structure "
				"before projective_space_implementation->init" << endl;
	}
	Implementation->init(P, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_subspaces::init_incidence_structure "
				"after projective_space_implementation->init" << endl;
	}


	if (n >= 2) {
		if (N_points < 1000) {
			if (f_v) {
				cout << "projective_space_subspaces::init "
						"before init_polarity" << endl;
			}

			init_polarity(verbose_level);


			if (f_v) {
				cout << "projective_space_subspaces::init "
						"after init_polarity" << endl;
			}
		}
		else {
			if (f_v) {
				cout << "projective_space_subspaces::init "
						"skipping init_polarity" << endl;
			}

		}
	}



	if (f_v) {
		cout << "projective_space_subspaces::init_incidence_structure done" << endl;
	}
}

void projective_space_subspaces::init_polarity(int verbose_level)
// uses Grass_hyperplanes
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_subspaces::init_polarity" << endl;
	}
	Standard_polarity = NEW_OBJECT(polarity);

	if (f_v) {
		cout << "projective_space_subspaces::init_polarity "
				"before Standard_polarity->init_standard_polarity" << endl;
	}
	Standard_polarity->init_standard_polarity(P, verbose_level);
	if (f_v) {
		cout << "projective_space_subspaces::init_polarity "
				"after Standard_polarity->init_standard_polarity" << endl;
	}

	Reversal_polarity = NEW_OBJECT(polarity);

	if (f_v) {
		cout << "projective_space_subspaces::init_polarity "
				"before Standard_polarity->init_reversal_polarity" << endl;
	}
	Reversal_polarity->init_reversal_polarity(P, verbose_level);
	if (f_v) {
		cout << "projective_space_subspaces::init_polarity "
				"after Standard_polarity->init_reversal_polarity" << endl;
	}


	if (f_v) {
		cout << "projective_space_subspaces::init_polarity done" << endl;
	}

}

void projective_space_subspaces::intersect_with_line(
		long int *set, int set_sz,
		int line_rk,
		long int *intersection, int &sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, b, idx;
	long int a;
	int *L;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space_subspaces::intersect_with_line" << endl;
	}
	L = Implementation->Lines + line_rk * k;
	sz = 0;
	for (i = 0; i < set_sz; i++) {
		a = set[i];
		b = a;
		if (b != a) {
			cout << "projective_space_subspaces::intersect_with_line data loss" << endl;
			exit(1);
		}
		if (Sorting.int_vec_search(L, k, b, idx)) {
			intersection[sz++] = a;
		}
	}
	if (f_v) {
		cout << "projective_space_subspaces::intersect_with_line done" << endl;
	}
}

void projective_space_subspaces::create_points_on_line(
	long int line_rk, long int *line, int verbose_level)
// needs line[k]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_subspaces::create_points_on_line" << endl;
	}
	long int a, b;

	Grass_lines->unrank_lint(line_rk, 0/*verbose_level - 4*/);
	for (a = 0; a < k; a++) {

		F->Projective_space_basic->PG_element_unrank_modified(
				v, 1, 2, a);

		F->Linear_algebra->mult_matrix_matrix(
				v, Grass_lines->M, w, 1, 2, n + 1,
				0 /* verbose_level */);

		F->Projective_space_basic->PG_element_rank_modified(
				w, 1, n + 1, b);

		line[a] = b;
	}
	if (f_v) {
		cout << "projective_space_subspaces::create_points_on_line done" << endl;
	}
}

void projective_space_subspaces::create_lines_on_point(
	long int point_rk,
	long int *line_pencil, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, i, d;
	int *v;
	int *w;
	int *Basis;

	if (f_v) {
		cout << "projective_space_subspaces::create_lines_on_point" << endl;
	}
	d = n + 1;
	v = NEW_int(d);
	w = NEW_int(n);
	Basis = NEW_int(2 * d);

	F->Projective_space_basic->PG_element_unrank_modified(
			v, 1, d, point_rk);
	for (i = 0; i < n + 1; i++) {
		if (v[i]) {
			break;
		}
	}
	if (i == n + 1) {
		cout << "projective_space_subspaces::create_lines_on_point zero vector" << endl;
		exit(1);
	}
	for (a = 0; a < r; a++) {
		F->Projective_space_basic->PG_element_unrank_modified(
				w, 1, n, a);
		Int_vec_copy(v, Basis, d);
		Int_vec_copy(w, Basis + d, i);
		Basis[d + i] = 0;
		Int_vec_copy(w + i, Basis + d + i + 1, n - i);
		b = Grass_lines->rank_lint_here(Basis, 0 /*verbose_level*/);
		line_pencil[a] = b;
	}
	FREE_int(v);
	FREE_int(w);
	FREE_int(Basis);
	if (f_v) {
		cout << "projective_space_subspaces::create_lines_on_point done" << endl;
	}
}

void projective_space_subspaces::create_lines_on_point_but_inside_a_plane(
	long int point_rk, long int plane_rk,
	long int *line_pencil, int verbose_level)
// assumes that line_pencil[q + 1] has been allocated
{
	int f_v = (verbose_level >= 1);
	int a, b, idx, d, rk, i;
	int *v;
	int *w;
	int *Basis;
	int *Plane;
	int *M;

	if (f_v) {
		cout << "projective_space_subspaces::create_lines_on_point_but_inside_a_plane" << endl;
	}
	if (n < 3) {
		cout << "projective_space_subspaces::create_lines_on_point_but_inside_a_plane n < 3" << endl;
		exit(1);
	}
	d = n + 1;
	v = NEW_int(d);
	w = NEW_int(n);
	Basis = NEW_int(2 * d);
	Plane = NEW_int(3 * d);
	M = NEW_int(4 * d);

	Grass_planes->unrank_lint_here(Plane, plane_rk, 0 /*verbose_level*/);

	F->Projective_space_basic->PG_element_unrank_modified(
			v, 1, d, point_rk);
	for (idx = 0; idx < n + 1; idx++) {
		if (v[idx]) {
			break;
		}
	}
	if (idx == n + 1) {
		cout << "projective_space_subspaces::create_lines_on_point_but_inside_a_plane zero vector" << endl;
		exit(1);
	}
	i = 0;
	for (a = 0; a < r; a++) {
		F->Projective_space_basic->PG_element_unrank_modified(
				w, 1, n, a);
		Int_vec_copy(v, Basis, d);
		Int_vec_copy(w, Basis + d, idx);
		Basis[d + idx] = 0;
		Int_vec_copy(w + idx, Basis + d + idx + 1, n - idx);


		Int_vec_copy(Plane, M, 3 * d);
		Int_vec_copy(Basis + d, M + 3 * d, d);
		rk = F->Linear_algebra->rank_of_rectangular_matrix(M, 4, d, 0 /* verbose_level*/);
		if (rk == 3) {
			b = Grass_lines->rank_lint_here(Basis, 0 /*verbose_level*/);
			line_pencil[i++] = b;
		}

	}
	if (i != q + 1) {
		cout << "projective_space_subspaces::create_lines_on_point_but_inside_a_plane  i != q + 1" << endl;
		exit(1);
	}
	FREE_int(v);
	FREE_int(w);
	FREE_int(Basis);
	FREE_int(Plane);
	FREE_int(M);
	if (f_v) {
		cout << "projective_space_subspaces::create_lines_on_point_but_inside_a_plane done" << endl;
	}
}


int projective_space_subspaces::create_point_on_line(
		long int line_rk, long int pt_rk, int verbose_level)
// pt_rk is between 0 and q-1.
{
	int f_v = (verbose_level >= 1);
	long int b;
	int v[2];

	if (f_v) {
		cout << "projective_space_subspaces::create_point_on_line" << endl;
	}
	Grass_lines->unrank_lint(line_rk, 0/*verbose_level - 4*/);
	if (f_v) {
		cout << "projective_space_subspaces::create_point_on_line line:" << endl;
		Int_matrix_print(Grass_lines->M, 2, n + 1);
	}

	F->Projective_space_basic->PG_element_unrank_modified(
			v, 1, 2, pt_rk);
	if (f_v) {
		cout << "projective_space_subspaces::create_point_on_line v=" << endl;
		Int_vec_print(cout, v, 2);
		cout << endl;
	}

	F->Linear_algebra->mult_matrix_matrix(v, Grass_lines->M, w, 1, 2, n + 1,
			0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space_subspaces::create_point_on_line w=" << endl;
		Int_vec_print(cout, w, n + 1);
		cout << endl;
	}

	F->Projective_space_basic->PG_element_rank_modified(
			w, 1, n + 1, b);

	if (f_v) {
		cout << "projective_space_subspaces::create_point_on_line b = " << b << endl;
	}
	return b;
}

void projective_space_subspaces::make_incidence_matrix(
	int &m, int &n,
	int *&Inc, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, h, j;

	if (f_v) {
		cout << "projective_space_subspaces::make_incidence_matrix" << endl;
	}
	m = N_points;
	n = N_lines;
	Inc = NEW_int(m * n);
	Int_vec_zero(Inc, m * n);
	for (i = 0; i < N_points; i++) {
		for (h = 0; h < r; h++) {
			j = Implementation->Lines_on_point[i * r + h];
			Inc[i * n + j] = 1;
		}
	}
	if (f_v) {
		cout << "projective_space_subspaces::make_incidence_matrix "
				"done" << endl;
	}
}

void projective_space_subspaces::make_incidence_matrix(
	std::vector<int> &Pts, std::vector<int> &Lines,
	int *&Inc, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b;
	int nb_pts, nb_lines;

	if (f_v) {
		cout << "projective_space_subspaces::make_incidence_matrix" << endl;
	}

	nb_pts = Pts.size();
	nb_lines = Lines.size();
	Inc = NEW_int(nb_pts * nb_lines);
	Int_vec_zero(Inc, nb_pts * nb_lines);
	for (i = 0; i < nb_pts; i++) {
		a = Pts[i];
		for (j = 0; j < nb_lines; j++) {
			b = Lines[j];
			if (is_incident(a, b)) {
				Inc[i * nb_lines + j] = 1;
			}
		}
	}
	if (f_v) {
		cout << "projective_space_subspaces::make_incidence_matrix done" << endl;
	}
}

int projective_space_subspaces::is_incident(int pt, int line)
{
	int f_v = false;
	long int rk;

	if (true /*incidence_bitvec == NULL*/) {
		Grass_lines->unrank_lint(line, 0/*verbose_level - 4*/);

		if (f_v) {
			cout << "projective_space_subspaces::is_incident "
					"line=" << endl;
			Int_matrix_print(Grass_lines->M, 2, n + 1);
		}
		Int_vec_copy(Grass_lines->M, Mtx, 2 * (n + 1));
		F->Projective_space_basic->PG_element_unrank_modified(
				Mtx + 2 * (n + 1), 1, n + 1, pt);
		if (f_v) {
			cout << "point:" << endl;
			Int_vec_print(cout, Mtx + 2 * (n + 1), n + 1);
			cout << endl;
		}

		rk = F->Linear_algebra->rank_of_rectangular_matrix_memory_given(
				Mtx,
				3, n + 1, Mtx2, v /* base_cols */,
				false /* f_complete */,
				0 /*verbose_level*/);
		if (f_v) {
			cout << "rk = " << rk << endl;
		}
		if (rk == 3) {
			return false;
		}
		else {
			return true;
		}
	}
	else {
		//a = (long int) pt * (long int) N_lines + (long int) line;
		//return bitvector_s_i(incidence_bitvec, a);
		return Implementation->Bitmatrix->s_ij(pt, line);
	}
}

void projective_space_subspaces::incidence_m_ii(int pt, int line, int a)
{
	//long int b;

	if (Implementation->Bitmatrix == NULL) {
		cout << "projective_space_subspaces::incidence_m_ii "
				"Bitmatrix == NULL" << endl;
		exit(1);
		}
	//b = (long int) pt * (long int) N_lines + (long int) line;
	//bitvector_m_ii(incidence_bitvec, b, a);

	Implementation->Bitmatrix->m_ij(pt, N_lines, a);
}

void projective_space_subspaces::make_incidence_structure_and_partition(
	incidence_structure *&Inc,
	data_structures::partitionstack *&Stack,
	int verbose_level)
// points vs lines
{
	int f_v = (verbose_level >= 1);
	int *M;
	int i, j, h;

	if (f_v) {
		cout << "projective_space_subspaces::make_incidence_structure_and_partition" << endl;
		cout << "N_points=" << N_points << endl;
		cout << "N_lines=" << N_lines << endl;
	}
	Inc = NEW_OBJECT(incidence_structure);


	if (f_v) {
		cout << "projective_space_subspaces::make_incidence_structure_and_partition "
				"allocating M of size "
				<< N_points * N_lines << endl;
	}
	M = NEW_int(N_points * N_lines);
	if (f_v) {
		cout << "projective_space_subspaces::make_incidence_structure_and_partition "
				"after allocating M of size "
				<< N_points * N_lines << endl;
	}
	Int_vec_zero(M, N_points * N_lines);

	if (Implementation->Lines_on_point == NULL) {
		cout << "projective_space_subspaces::make_incidence_structure_and_partition "
				"Lines_on_point == NULL" << endl;
		exit(1);
	}
	for (i = 0; i < N_points; i++) {
		for (h = 0; h < r; h++) {
			j = Implementation->Lines_on_point[i * r + h];
			M[i * N_lines + j] = 1;
		}
	}
	if (f_v) {
		cout << "projective_space_subspaces::make_incidence_structure_and_partition "
				"before Inc->init_by_matrix" << endl;
	}
	Inc->init_by_matrix(N_points, N_lines, M, verbose_level - 1);
	if (f_v) {
		cout << "projective_space_subspaces::make_incidence_structure_and_partition "
				"after Inc->init_by_matrix" << endl;
	}
	FREE_int(M);


	Stack = NEW_OBJECT(data_structures::partitionstack);
	Stack->allocate(N_points + N_lines, 0 /* verbose_level */);
	Stack->subset_contiguous(N_points, N_lines);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();

	if (f_v) {
		cout << "projective_space_subspaces::make_incidence_structure_and_partition done" << endl;
	}
}

void projective_space_subspaces::incma_for_type_ij(
	int row_type, int col_type,
	int *&Incma, int &nb_rows, int &nb_cols,
	int verbose_level)
// row_type, col_type are the vector space
// dimensions of the objects
// indexing rows and columns.
{
	int f_v = (verbose_level >= 1);
	int i, j, rk;
	int *Basis;
	int *Basis2;
	int *base_cols;
	int d = n + 1;

	if (f_v) {
		cout << "projective_space_subspaces::incma_for_type_ij" << endl;
		cout << "row_type = " << row_type << endl;
		cout << "col_type = " << col_type << endl;
	}
	if (col_type < row_type) {
		cout << "projective_space_subspaces::incma_for_type_ij "
				"col_type < row_type" << endl;
		exit(1);
	}
	if (col_type < 0) {
		cout << "projective_space_subspaces::incma_for_type_ij "
				"col_type < 0" << endl;
		exit(1);
	}
	if (col_type > n + 1) {
		cout << "projective_space_subspaces::incma_for_type_ij "
				"col_type > P->n + 1" << endl;
		exit(1);
	}
	nb_rows = nb_rk_k_subspaces_as_lint(row_type);
	nb_cols = nb_rk_k_subspaces_as_lint(col_type);


	Basis = NEW_int(3 * d * d);
	Basis2 = NEW_int(3 * d * d);
	base_cols = NEW_int(d);

	Incma = NEW_int(nb_rows * nb_cols);
	Int_vec_zero(Incma, nb_rows * nb_cols);
	for (i = 0; i < nb_rows; i++) {
		if (row_type == 1) {
			unrank_point(Basis, i);
		}
		else if (row_type == 2) {
			unrank_line(Basis, i);
		}
		else if (row_type == 3) {
			unrank_plane(Basis, i);
		}
		else {
			cout << "projective_space_subspaces::incma_for_type_ij "
					"row_type " << row_type
				<< " not yet implemented" << endl;
			exit(1);
		}
		for (j = 0; j < nb_cols; j++) {
			if (col_type == 1) {
				unrank_point(Basis + row_type * d, j);
			}
			else if (col_type == 2){
				unrank_line(Basis + row_type * d, j);
			}
			else if (col_type == 3) {
				unrank_plane(Basis + row_type * d, j);
			}
			else {
				cout << "projective_space_subspaces::incma_for_type_ij "
						"col_type " << col_type
					<< " not yet implemented" << endl;
				exit(1);
			}
			rk = F->Linear_algebra->rank_of_rectangular_matrix_memory_given(Basis,
					row_type + col_type, d, Basis2, base_cols,
					false /* f_complete */,
					0 /*verbose_level*/);
			if (rk == col_type) {
				Incma[i * nb_cols + j] = 1;
			}
		}
	}

	FREE_int(Basis);
	FREE_int(Basis2);
	FREE_int(base_cols);
	if (f_v) {
		cout << "projective_space_subspaces::incma_for_type_ij done" << endl;
	}
}

int projective_space_subspaces::incidence_test_for_objects_of_type_ij(
	int type_i, int type_j, int i, int j,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk;
	int *Basis;
	int *Basis2;
	int *base_cols;
	int d = n + 1;
	//int nb_rows, nb_cols;
	int f_incidence = false;

	if (f_v) {
		cout << "projective_space_subspaces::incidence_test_for_objects_of_type_ij" << endl;
		cout << "type_i = " << type_i << endl;
		cout << "type_j = " << type_j << endl;
	}
	if (type_j < type_i) {
		cout << "projective_space_subspaces::incidence_test_for_objects_of_type_ij "
				"type_j < type_i" << endl;
		exit(1);
	}
	if (type_i < 0) {
		cout << "projective_space_subspaces::incidence_test_for_objects_of_type_ij "
				"type_i < 0" << endl;
		exit(1);
	}
	if (type_j > n + 1) {
		cout << "projective_space_subspaces::incidence_test_for_objects_of_type_ij "
				"type_j > n + 1" << endl;
		exit(1);
	}
	//nb_rows = nb_rk_k_subspaces_as_lint(type_i);
	//nb_cols = nb_rk_k_subspaces_as_lint(type_j);


	Basis = NEW_int(3 * d * d);
	Basis2 = NEW_int(3 * d * d);
	base_cols = NEW_int(d);

	if (type_i == 1) {
		unrank_point(Basis, i);
	}
	else if (type_i == 2) {
		unrank_line(Basis, i);
	}
	else if (type_i == 3) {
		unrank_plane(Basis, i);
	}
	else {
		cout << "projective_space_subspaces::incidence_test_for_objects_of_type_ij "
				"row_type " << type_i
			<< " not yet implemented" << endl;
		exit(1);
	}
	if (type_j == 1) {
		unrank_point(Basis + type_i * d, j);
	}
	else if (type_j == 2){
		unrank_line(Basis + type_i * d, j);
	}
	else if (type_j == 3) {
		unrank_plane(Basis + type_i * d, j);
	}
	else {
		cout << "projective_space_subspaces::incidence_test_for_objects_of_type_ij "
				"type_j " << type_j
			<< " not yet implemented" << endl;
		exit(1);
	}
	rk = F->Linear_algebra->rank_of_rectangular_matrix_memory_given(Basis,
			type_i + type_j, d, Basis2, base_cols,
			false /* f_complete */,
			0 /*verbose_level*/);
	if (rk == type_j) {
		f_incidence = true;
	}

	FREE_int(Basis);
	FREE_int(Basis2);
	FREE_int(base_cols);
	if (f_v) {
		cout << "projective_space_subspaces::incidence_test_for_objects_of_type_ij done" << endl;
	}
	return f_incidence;
}


void projective_space_subspaces::points_on_line(
		long int line_rk,
		long int *&the_points,
		int &nb_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_subspaces::points_on_line" << endl;
	}

	nb_points = Grass_lines->nb_points_covered(0 /* verbose_level */);

	Grass_lines->unrank_lint(line_rk, 0 /* verbose_level */);

	the_points = NEW_lint(nb_points);

	Grass_lines->points_covered(the_points, verbose_level);


	if (f_v) {
		cout << "projective_space_subspaces::points_on_line done" << endl;
	}
}


void projective_space_subspaces::points_covered_by_plane(
		long int plane_rk,
		long int *&the_points,
		int &nb_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_subspaces::points_covered_by_plane" << endl;
	}

	nb_points = Grass_planes->nb_points_covered(0 /* verbose_level */);

	Grass_planes->unrank_lint(plane_rk, 0 /* verbose_level */);

	the_points = NEW_lint(nb_points);

	Grass_planes->points_covered(the_points, verbose_level);


	if (f_v) {
		cout << "projective_space_subspaces::points_covered_by_plane done" << endl;
	}
}

void projective_space_subspaces::incidence_and_stack_for_type_ij(
	int row_type, int col_type,
	incidence_structure *&Inc,
	data_structures::partitionstack *&Stack,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Incma;
	int nb_rows, nb_cols;

	if (f_v) {
		cout << "projective_space_subspaces::incidence_and_stack_for_type_ij" << endl;
	}
	incma_for_type_ij(
		row_type, col_type,
		Incma, nb_rows, nb_cols,
		verbose_level);
	if (f_v) {
		cout << "projective_space_subspaces::incidence_and_stack_for_type_ij "
				"before Inc->init_by_matrix" << endl;
	}
	Inc = NEW_OBJECT(incidence_structure);
	Inc->init_by_matrix(
			nb_rows, nb_cols, Incma,
			verbose_level - 1);
	if (f_v) {
		cout << "projective_space_subspaces::incidence_and_stack_for_type_ij "
				"after Inc->init_by_matrix" << endl;
	}
	FREE_int(Incma);


	Stack = NEW_OBJECT(data_structures::partitionstack);
	Stack->allocate(nb_rows + nb_cols, 0 /* verbose_level */);
	Stack->subset_contiguous(nb_rows, nb_cols);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();

	if (f_v) {
		cout << "projective_space_subspaces::incidence_and_stack_for_type_ij done" << endl;
	}
}

long int projective_space_subspaces::nb_rk_k_subspaces_as_lint(int k)
{
	combinatorics::combinatorics_domain C;
	ring_theory::longinteger_object aa;
	long int N;
	int d = n + 1;

	C.q_binomial(aa, d, k, q, 0/*verbose_level*/);
	N = aa.as_lint();
	return N;
}

long int projective_space_subspaces::rank_point(int *v)
{
	long int b;
	int verbose_level = 0;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_subspaces::rank_point: v=";
		Int_vec_print(cout, v, n + 1);
		cout << endl;
	}

	F->Projective_space_basic->PG_element_rank_modified_lint(
			v, 1, n + 1, b);

	if (f_v) {
		cout << "projective_space_subspaces::rank_point: v=";
		Int_vec_print(cout, v, n + 1);
		cout << " has rank " << b << ", done" << endl;
	}
	return b;
}

void projective_space_subspaces::unrank_point(int *v, long int rk)
{
	F->Projective_space_basic->PG_element_unrank_modified_lint(
			v, 1, n + 1, rk);
}

void projective_space_subspaces::unrank_points(int *v, long int *Rk, int sz)
{
	int i;

	for (i = 0; i < sz; i++) {
		F->Projective_space_basic->PG_element_unrank_modified_lint(
				v + i * (n + 1), 1, n + 1, Rk[i]);
	}
}

long int projective_space_subspaces::rank_line(int *basis)
{
	long int b;

	b = Grass_lines->rank_lint_here(basis, 0/*verbose_level - 4*/);
	return b;
}

void projective_space_subspaces::unrank_line(int *basis, long int rk)
{
	Grass_lines->unrank_lint_here(basis, rk, 0/*verbose_level - 4*/);
}

void projective_space_subspaces::unrank_lines(int *v, long int *Rk, int nb)
{
	int i;

	for (i = 0; i < nb; i++) {
		Grass_lines->unrank_lint_here(
				v + i * 2 * (n + 1), Rk[i], 0 /* verbose_level */);
	}
}

long int projective_space_subspaces::rank_plane(int *basis)
{
	long int b;

	if (Grass_planes == NULL) {
		cout << "projective_space_subspaces::rank_plane "
				"Grass_planes == NULL" << endl;
		exit(1);
	}
	b = Grass_planes->rank_lint_here(basis, 0/*verbose_level - 4*/);
	return b;
}

void projective_space_subspaces::unrank_plane(int *basis, long int rk)
{
	if (Grass_planes == NULL) {
		cout << "projective_space_subspaces::unrank_plane "
				"Grass_planes == NULL" << endl;
		exit(1);
	}
	Grass_planes->unrank_lint_here(basis, rk, 0/*verbose_level - 4*/);
}




long int projective_space_subspaces::line_through_two_points(
		long int p1, long int p2)
{
	long int b;

	unrank_point(Grass_lines->M, p1);
	unrank_point(Grass_lines->M + n + 1, p2);
	b = Grass_lines->rank_lint(0/*verbose_level - 4*/);
	return b;
}

int projective_space_subspaces::test_if_lines_are_disjoint(
		long int l1, long int l2)
{
	data_structures::sorting Sorting;

	if (Implementation->Lines) {
		return Sorting.test_if_sets_are_disjoint_assuming_sorted(
				Implementation->Lines + l1 * k,
				Implementation->Lines + l2 * k, k, k);
	}
	else {
		return test_if_lines_are_disjoint_from_scratch(l1, l2);
	}
}

int projective_space_subspaces::test_if_lines_are_disjoint_from_scratch(
		long int l1, long int l2)
{
	int *Mtx;
	int m, rk;

	m = n + 1;
	Mtx = NEW_int(4 * m);
	Grass_lines->unrank_lint_here(
			Mtx, l1, 0/*verbose_level - 4*/);
	Grass_lines->unrank_lint_here(
			Mtx + 2 * m, l2, 0/*verbose_level - 4*/);
	rk = F->Linear_algebra->Gauss_easy(Mtx, 4, m);
	FREE_int(Mtx);
	if (rk == 4) {
		return true;
	}
	else {
		return false;
	}
}

int projective_space_subspaces::intersection_of_two_lines(
		long int l1, long int l2)
// formerly intersection_of_two_lines_in_a_plane
{
	int *Mtx1;
	int *Mtx3;
	int b, r;

	int d = n + 1;
	int D = 2 * d;


	Mtx1 = NEW_int(d * d);
	Mtx3 = NEW_int(D * d);

	Grass_lines->unrank_lint_here(Mtx1, l1, 0/*verbose_level - 4*/);
	F->Linear_algebra->perp_standard(d, 2, Mtx1, 0);
	Int_vec_copy(Mtx1 + 2 * d, Mtx3, (d - 2) * d);

	Grass_lines->unrank_lint_here(Mtx1, l2, 0/*verbose_level - 4*/);
	F->Linear_algebra->perp_standard(d, 2, Mtx1, 0);
	Int_vec_copy(Mtx1 + 2 * d, Mtx3 + (d - 2) * d, (d - 2) * d);

	r = F->Linear_algebra->Gauss_easy(Mtx3, 2 * (d - 2), d);
	if (r < d - 1) {
		cout << "projective_space_subspaces::intersection_of_two_lines r < d - 1, "
				"the lines do not intersect" << endl;
		exit(1);
	}
	if (r > d - 1) {
		cout << "projective_space_subspaces::intersection_of_two_lines r > d - 1, "
				"something is wrong" << endl;
		exit(1);
	}
	F->Linear_algebra->perp_standard(d, d - 1, Mtx3, 0);
	b = rank_point(Mtx3 + (d - 1) * d);

	FREE_int(Mtx1);
	FREE_int(Mtx3);

	return b;
}




void projective_space_subspaces::line_intersection_type(
	long int *set, int set_size, int *type, int verbose_level)
// type[N_lines]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_subspaces::line_intersection_type" << endl;
	}
	if (Implementation->Lines_on_point == NULL) {
		if (f_v) {
			cout << "projective_space_subspaces::line_intersection_type "
					"before line_intersection_type_basic" << endl;
		}
		line_intersection_type_basic(set, set_size, type, verbose_level);
		if (f_v) {
			cout << "projective_space_subspaces::line_intersection_type "
					"after line_intersection_type_basic" << endl;
		}
	}
	else {

		if (f_v) {
			cout << "projective_space_subspaces::line_intersection_type "
					"before Implementation->line_intersection_type" << endl;
		}
		Implementation->line_intersection_type(
				set, set_size, type, verbose_level);
		if (f_v) {
			cout << "projective_space_subspaces::line_intersection_type "
					"after Implementation->line_intersection_type" << endl;
		}

	}
	if (f_v) {
		cout << "projective_space_subspaces::line_intersection_type done" << endl;
	}
}

void projective_space_subspaces::line_intersection_type_basic(
	long int *set, int set_size, int *type, int verbose_level)
// type[N_lines]
{
	int f_v = (verbose_level >= 1);
	long int rk, h, d;
	int *M;

	if (f_v) {
		cout << "projective_space_subspaces::line_intersection_type_basic" << endl;
	}
	d = n + 1;
	M = NEW_int(3 * d);
	for (rk = 0; rk < N_lines; rk++) {

		type[rk] = 0;
		Grass_lines->unrank_lint(
				rk, 0 /* verbose_level */);

		for (h = 0; h < set_size; h++) {

			Int_vec_copy(Grass_lines->M, M, 2 * d);
			unrank_point(M + 2 * d, set[h]);

			if (F->Linear_algebra->rank_of_rectangular_matrix(
					M,
					3, d, 0 /*verbose_level*/) == 2) {
				type[rk]++;
			}
		} // next h
	} // next rk
	FREE_int(M);
	if (f_v) {
		cout << "projective_space_subspaces::line_intersection_type_basic done" << endl;
	}
}

void projective_space_subspaces::line_intersection_type_basic_given_a_set_of_lines(
		long int *lines_by_rank, int nb_lines,
	long int *set, int set_size,
	int *type, int verbose_level)
// type[nb_lines]
{
	int f_v = (verbose_level >= 1);
	long int rk, h, d, u;
	int *M;

	if (f_v) {
		cout << "projective_space_subspaces::line_intersection_type_basic_given_a_set_of_lines" << endl;
	}
	d = n + 1;
	M = NEW_int(3 * d);
	for (u = 0; u < nb_lines; u++) {
		rk = lines_by_rank[u];
		type[u] = 0;
		Grass_lines->unrank_lint(
				rk, 0 /* verbose_level */);

		for (h = 0; h < set_size; h++) {

			Int_vec_copy(Grass_lines->M, M, 2 * d);
			unrank_point(M + 2 * d, set[h]);

			if (F->Linear_algebra->rank_of_rectangular_matrix(
					M,
					3, d, 0 /*verbose_level*/) == 2) {
				type[u]++;
			}
		} // next h
	} // next rk
	FREE_int(M);
	if (f_v) {
		cout << "projective_space_subspaces::line_intersection_type_basic_given_a_set_of_lines done" << endl;
	}
}


void projective_space_subspaces::line_intersection_type_through_hyperplane(
	long int *set, int set_size, int *type, int verbose_level)
// type[N_lines]
{
	int f_v = (verbose_level >= 1);
	int f_vv = false;
	long int rk, h, i, j, d, cnt, i1;
	int *M;
	int *M2;
	int *Pts1;
	int *Pts2;
	long int *set1;
	long int *set2;
	int *cnt1;
	int sz1, sz2;
	int *f_taken;
	int nb_pts_in_hyperplane;
	int idx;
	geometry_global Gg;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space_subspaces::line_intersection_type_through_hyperplane "
				"set_size=" << set_size << endl;
	}
	d = n + 1;
	M = NEW_int(3 * d);
	M2 = NEW_int(3 * d);
	set1 = NEW_lint(set_size);
	set2 = NEW_lint(set_size);
	sz1 = 0;
	sz2 = 0;
	for (i = 0; i < set_size; i++) {
		unrank_point(M, set[i]);
		if (f_vv) {
			cout << set[i] << " : ";
			Int_vec_print(cout, M, d);
			cout << endl;
		}
		if (M[d - 1] == 0) {
			set1[sz1++] = set[i];
		}
		else {
			set2[sz2++] = set[i];
		}
	}

	Sorting.lint_vec_heapsort(set1, sz1);

	if (f_vv) {
		cout << "projective_space_subspaces::line_intersection_type_through_hyperplane "
				"sz1=" << sz1 << " sz2=" << sz2 << endl;
	}


	// do the line type in the hyperplane:

	if (f_v) {
		cout << "projective_space_subspaces::line_intersection_type_through_hyperplane "
				"before line_intersection_type_basic" << endl;
	}
	line_intersection_type_basic(set1, sz1, type, verbose_level);
	if (f_v) {
		cout << "projective_space_subspaces::line_intersection_type_through_hyperplane "
				"after line_intersection_type_basic" << endl;
	}

	nb_pts_in_hyperplane = Gg.nb_PG_elements(n - 1, q);
	if (f_vv) {
		cout << "projective_space_subspaces::line_intersection_type_through_hyperplane "
				"nb_pts_in_hyperplane="
				<< nb_pts_in_hyperplane << endl;
	}

	cnt1 = NEW_int(nb_pts_in_hyperplane);
	Pts1 = NEW_int(nb_pts_in_hyperplane * d);
	Pts2 = NEW_int(sz2 * d);

	Int_vec_zero(cnt1, nb_pts_in_hyperplane);
	for (i = 0; i < nb_pts_in_hyperplane; i++) {

		F->Projective_space_basic->PG_element_unrank_modified(
				Pts1 + i * d, 1, n, i);

		Pts1[i * d + d - 1] = 0;

		F->Projective_space_basic->PG_element_rank_modified_lint(
				Pts1 + i * d, 1, n + 1, i1);

		// i1 is the rank of the hyperplane point
		// inside the larger space:
		//unrank_point(Pts1 + i * d, set1[i]);
		if (Sorting.lint_vec_search(set1, sz1, i1, idx, 0)) {
			cnt1[i] = 1;
		}
	}
	for (i = 0; i < sz2; i++) {
		unrank_point(Pts2 + i * d, set2[i]);
	}

	f_taken = NEW_int(sz2);
	for (i = 0; i < nb_pts_in_hyperplane; i++) {
		if (f_vv) {
			cout << "projective_space_subspaces::line_intersection_type_through_hyperplane "
					"checking lines through point " << i
					<< " / " << nb_pts_in_hyperplane << ":" << endl;
		}
		Int_vec_zero(f_taken, sz2);
		for (j = 0; j < sz2; j++) {
			if (f_taken[j]) {
				continue;
			}
			if (f_vv) {
				cout << "projective_space_subspaces::line_intersection_type_through_hyperplane "
						"j=" << j << " / " << sz2 << ":" << endl;
			}
			Int_vec_copy(Pts1 + i * d, M, d);
			Int_vec_copy(Pts2 + j * d, M + d, d);
			f_taken[j] = true;
			if (f_vv) {
				Int_matrix_print(M, 2, d);
			}
			rk = Grass_lines->rank_lint_here(M, 0 /* verbose_level */);
			if (f_vv) {
				cout << "projective_space_subspaces::line_intersection_type_through_hyperplane "
						"line rk=" << rk << " cnt1="
						<< cnt1[rk] << ":" << endl;
			}
			cnt = 1 + cnt1[i];
			for (h = j + 1; h < sz2; h++) {
				Int_vec_copy(M, M2, 2 * d);
				Int_vec_copy(Pts2 + h * d, M2 + 2 * d, d);
				if (F->Linear_algebra->rank_of_rectangular_matrix(
						M2,
						3, d, 0 /*verbose_level*/) == 2) {
					cnt++;
					f_taken[h] = true;
				}
			}
			type[rk] = cnt;
		}
	}
	FREE_int(f_taken);
	FREE_int(M);
	FREE_int(M2);
	FREE_lint(set1);
	FREE_lint(set2);
	FREE_int(Pts1);
	FREE_int(Pts2);
	FREE_int(cnt1);

	if (f_v) {
		cout << "projective_space_subspaces::line_intersection_type_through_hyperplane "
				"done" << endl;
	}
}





void projective_space_subspaces::point_plane_incidence_matrix(
		long int *point_rks, int nb_points,
		long int *plane_rks, int nb_planes,
		int *&M, int verbose_level)
// M[nb_points * nb_planes]
{
	int f_v = (verbose_level >= 1);
	int rk, d, i, j;
	int *M1;
	int *M2;
	int *Pts;
	//grassmann *G;

	if (f_v) {
		cout << "projective_space_subspaces::plane_intersection_type_basic" << endl;
	}
	d = n + 1;
	M1 = NEW_int(4 * d);
	M2 = NEW_int(4 * d);
	Pts = NEW_int(nb_points * d);
	//G = NEW_OBJECT(grassmann);

	//G->init(d, 3, F, 0 /* verbose_level */);

	M = NEW_int(nb_points * nb_planes);
	Int_vec_zero(M, nb_points * nb_planes);

	// unrank all point here so we don't
	// have to do it again in the loop
	for (i = 0; i < nb_points; i++) {
		unrank_point(Pts + i * d, point_rks[i]);
	}

	for (j = 0; j < nb_planes; j++) {
		if (rk && (rk % ONE_MILLION) == 0) {
			cout << "projective_space_subspaces::plane_intersection_type_basic "
					"rk=" << rk << endl;
		}
		rk = plane_rks[j];
		Grass_planes->unrank_lint_here(
				M1, rk, 0 /* verbose_level */);

		// check which points are contained in the plane:
		for (i = 0; i < nb_points; i++) {

			Int_vec_copy(M1, M2, 3 * d);
			//unrank_point(M2 + 3 * d, set[h]);
			Int_vec_copy(Pts + i * d, M2 + 3 * d, d);

			if (F->Linear_algebra->rank_of_rectangular_matrix(
					M2,
					4, d, 0 /*verbose_level*/) == 3) {
				// the point lies in the plane,
				// increment the intersection count:
				M[i * nb_planes + j] = 1;
			}
		} // next h

	} // next rk
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(Pts);
	//FREE_OBJECT(G);
	if (f_v) {
		cout << "projective_space_subspaces::plane_intersection_type_basic done" << endl;
	}
}



void projective_space_subspaces::plane_intersection_type_basic(
	long int *set, int set_size,
	int *type, int verbose_level)
// set[set_size], type[N_planes]
// type[i] = number of points in the set lying on plane i
// counts the number of points from the given set lying on each plane.
{
	int f_v = (verbose_level >= 1);
	int rk, h, d, N_planes;
	int *M1;
	int *M2;
	int *Pts;
	//grassmann *G;

	if (f_v) {
		cout << "projective_space_subspaces::plane_intersection_type_basic" << endl;
	}
	d = n + 1;
	M1 = NEW_int(4 * d);
	M2 = NEW_int(4 * d);
	Pts = NEW_int(set_size * d);
	//G = NEW_OBJECT(grassmann);

	//G->init(d, 3, F, 0 /* verbose_level */);

	N_planes = nb_rk_k_subspaces_as_lint(3);
	if (f_v) {
		cout << "projective_space_subspaces::plane_intersection_type_basic "
				"N_planes=" << N_planes << endl;
	}

	// unrank all point here so we don't
	// have to do it again in the loop
	for (h = 0; h < set_size; h++) {
		unrank_point(Pts + h * d, set[h]);
	}

	for (rk = 0; rk < N_planes; rk++) {
		if (rk && (rk % ONE_MILLION) == 0) {
			cout << "projective_space_subspaces::plane_intersection_type_basic "
					"rk=" << rk << endl;
		}
		type[rk] = 0;
		Grass_planes->unrank_lint_here(
				M1, rk, 0 /* verbose_level */);

		// check which points are contained in the plane:
		for (h = 0; h < set_size; h++) {

			Int_vec_copy(M1, M2, 3 * d);
			//unrank_point(M2 + 3 * d, set[h]);
			Int_vec_copy(Pts + h * d, M2 + 3 * d, d);

			if (F->Linear_algebra->rank_of_rectangular_matrix(
					M2,
					4, d, 0 /*verbose_level*/) == 3) {
				// the point lies in the plane,
				// increment the intersection count:
				type[rk]++;
			}
		} // next h

	} // next rk
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(Pts);
	//FREE_OBJECT(G);
	if (f_v) {
		cout << "projective_space_subspaces::plane_intersection_type_basic done" << endl;
	}
}

void projective_space_subspaces::hyperplane_intersection_type_basic(
		long int *set, int set_size, int *type,
	int verbose_level)
// type[N_hyperplanes]
{
	int f_v = (verbose_level >= 1);
	int rk, h, d, N_hyperplanes;
	int *M;
	int *Pts;
	//grassmann *G;

	if (f_v) {
		cout << "projective_space_subspaces::hyperplane_intersection_type_basic" << endl;
	}
	d = n + 1;
	M = NEW_int(4 * d);
	Pts = NEW_int(set_size * d);
	//G = NEW_OBJECT(grassmann);

	//G->init(d, d - 1, F, 0 /* verbose_level */);

	N_hyperplanes = nb_rk_k_subspaces_as_lint(d - 1);

	// unrank all points here so we don't
	// have to do it again in the loop
	for (h = 0; h < set_size; h++) {
		unrank_point(Pts + h * d, set[h]);
	}

	for (rk = 0; rk < N_hyperplanes; rk++) {
		type[rk] = 0;
		Grass_hyperplanes->unrank_lint(rk, 0 /* verbose_level */);

		// check which points are contained in the hyperplane:
		for (h = 0; h < set_size; h++) {

			Int_vec_copy(Grass_hyperplanes->M, M, (d - 1) * d);
			//unrank_point(M + (d - 1) * d, set[h]);
			Int_vec_copy(Pts + h * d, M + (d - 1) * d, d);

			if (F->Linear_algebra->rank_of_rectangular_matrix(
					M,
					d, d, 0 /*verbose_level*/) == d - 1) {
				// the point lies in the hyperplane,
				// increment the intersection count:
				type[rk]++;
			}
		} // next h

	} // next rk
	FREE_int(M);
	FREE_int(Pts);
	//FREE_OBJECT(G);
	if (f_v) {
		cout << "projective_space_subspaces::hyperplane_intersection_type_basic done" << endl;
	}
}



void projective_space_subspaces::line_intersection_type_collected(
	long int *set, int set_size, int *type_collected,
	int verbose_level)
// type[set_size + 1]
{
	int f_v = (verbose_level >= 1);
	int rk, h, d, cnt;
	int *M;
	int *Pts;

	if (f_v) {
		cout << "projective_space_subspaces::line_intersection_type_collected" << endl;
	}
	d = n + 1;
	M = NEW_int(3 * d);
	Pts = NEW_int(set_size * d);
	Int_vec_zero(type_collected, set_size + 1);

	// unrank all point here so we don't
	// have to do it again in the loop
	for (h = 0; h < set_size; h++) {
		unrank_point(Pts + h * d, set[h]);
	}

	// loop over all lines:
	for (rk = 0; rk < N_lines; rk++) {
		Grass_lines->unrank_lint(rk, 0 /* verbose_level */);
		cnt = 0;

		// count the number of points in the set that lie on the line:
		for (h = 0; h < set_size; h++) {

			Int_vec_copy(Grass_lines->M, M, 2 * d);


			//unrank_point(M + 2 * d, set[h]);
			Int_vec_copy(Pts + h * d, M + 2 * d, d);

			// test if the point lies on the line:
			if (F->Linear_algebra->rank_of_rectangular_matrix(M,
					3, d, 0 /*verbose_level*/) == 2) {

				// yes, increment the counter
				cnt++;
			}
		} // next h

		// cnt is the number of points on the line:
		// increment the line type vector at cnt:
		type_collected[cnt]++;

	} // next rk
	FREE_int(M);
	FREE_int(Pts);
	if (f_v) {
		cout << "projective_space_subspaces::line_intersection_type_collected done" << endl;
	}
}

void projective_space_subspaces::find_external_lines(
	long int *set, int set_size,
	long int *external_lines, int &nb_external_lines,
	int verbose_level)
{
	int *type;
	int i;

	nb_external_lines = 0;
	type = NEW_int(N_lines);
	line_intersection_type(set, set_size, type, verbose_level);
	for (i = 0; i < N_lines; i++) {
		if (type[i]) {
			continue;
		}
		external_lines[nb_external_lines++] = i;
	}
	FREE_int(type);
}

void projective_space_subspaces::find_tangent_lines(
	long int *set, int set_size,
	long int *tangent_lines, int &nb_tangent_lines,
	int verbose_level)
{
	int *type;
	int i;

	nb_tangent_lines = 0;
	type = NEW_int(N_lines);
	line_intersection_type(set, set_size, type, verbose_level);
	for (i = 0; i < N_lines; i++) {
		if (type[i] != 1) {
			continue;
		}
		tangent_lines[nb_tangent_lines++] = i;
	}
	FREE_int(type);
}

void projective_space_subspaces::find_secant_lines(
	long int *set, int set_size,
	long int *secant_lines, int &nb_secant_lines,
	int verbose_level)
{
	int *type;
	int i;

	nb_secant_lines = 0;
	type = NEW_int(N_lines);
	line_intersection_type(set, set_size, type, verbose_level);
	for (i = 0; i < N_lines; i++) {
		if (type[i] != 2) {
			continue;
		}
		secant_lines[nb_secant_lines++] = i;
	}
	FREE_int(type);
}

void projective_space_subspaces::find_k_secant_lines(
	long int *set, int set_size, int k,
	long int *secant_lines, int &nb_secant_lines,
	int verbose_level)
{
	int *type;
	int i;

	nb_secant_lines = 0;
	type = NEW_int(N_lines);
	line_intersection_type(set, set_size, type, verbose_level);
	for (i = 0; i < N_lines; i++) {
		if (type[i] != k) {
			continue;
		}
		secant_lines[nb_secant_lines++] = i;
	}
	FREE_int(type);
}

void projective_space_subspaces::export_incidence_matrix_to_csv(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_subspaces::export_incidence_matrix_to_csv" << endl;
	}

	int i, j, k;
	int *T;
	orbiter_kernel_system::file_io Fio;

	T = NEW_int(N_points * N_lines);
	for (i = 0; i < N_points; i++) {
		for (j = 0; j < N_lines; j++) {
			if (is_incident(i, j)) {
				k = 1;
			}
			else {
				k = 0;
			}
			T[i * N_lines + j] = k;
		}
	}
	string fname;

	make_fname_incidence_matrix_csv(fname);

	Fio.int_matrix_write_csv(fname, T, N_points, N_lines);

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_int(T);
	if (f_v) {
		cout << "projective_space_subspaces::export_incidence_matrix_to_csv done" << endl;
	}
}

void projective_space_subspaces::export_restricted_incidence_matrix_to_csv(
		std::string &rows, std::string &cols, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_subspaces::export_restricted_incidence_matrix_to_csv" << endl;
	}

	int *Rows;
	int nb_rows;
	int *Cols;
	int nb_cols;

	Get_int_vector_from_label(rows, Rows, nb_rows, verbose_level);
	Get_int_vector_from_label(cols, Cols, nb_cols, verbose_level);


	int i, j, k, ii, jj;
	int *T;
	orbiter_kernel_system::file_io Fio;

	T = NEW_int(nb_rows * nb_cols);
	for (i = 0; i < nb_rows; i++) {
		ii = Rows[i];
		for (j = 0; j < nb_cols; j++) {
			jj = Cols[j];
			if (is_incident(ii, jj)) {
				k = 1;
			}
			else {
				k = 0;
			}
			T[i * nb_cols + j] = k;
		}
	}
	string fname;

	make_fname_incidence_matrix_csv(fname);

	Fio.int_matrix_write_csv(fname, T, nb_rows, nb_cols);

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_int(T);
	if (f_v) {
		cout << "projective_space_subspaces::export_restricted_incidence_matrix_to_csv done" << endl;
	}
}

void projective_space_subspaces::make_fname_incidence_matrix_csv(std::string &fname)
{

	fname = "PG_n" + std::to_string(n) + "_q" + std::to_string(q) + "_incidence_matrix.csv";
}



void projective_space_subspaces::compute_decomposition(
		data_structures::partitionstack *S1,
		data_structures::partitionstack *S2,
		incidence_structure *&Inc,
		data_structures::partitionstack *&Stack,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_subspaces::compute_decomposition" << endl;
	}
	if (f_v) {
		cout << "projective_space_subspaces::compute_decomposition "
				"before incidence_and_stack_for_type_ij" << endl;
	}
	incidence_and_stack_for_type_ij(
		1 /* row_type */, 2 /* col_type */,
		Inc,
		Stack,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_subspaces::compute_decomposition "
				"after incidence_and_stack_for_type_ij" << endl;
	}

	int i, j, sz;

	for (i = 1; i < S1->ht; i++) {
		if (f_v) {
			cout << "projective_space_subspaces::compute_decomposition "
					"before Stack->split_cell (S1) i=" << i << endl;
		}
		Stack->split_cell(
				S1->pointList + S1->startCell[i],
				S1->cellSize[i], verbose_level);
	}
	int *set;
	set = NEW_int(Inc->nb_rows + Inc->nb_cols);
	for (i = 1; i < S2->ht; i++) {
		sz = S2->cellSize[i];
		Int_vec_copy(S2->pointList + S2->startCell[i], set, sz);
		for (j = 0; j < sz; j++) {
			set[j] += Inc->nb_rows;
		}
		if (f_v) {
			cout << "projective_space_subspaces::compute_decomposition "
					"before Stack->split_cell (S2) i=" << i << endl;
		}
		Stack->split_cell(set, sz, 0 /*verbose_level*/);
	}
	FREE_int(set);
	if (f_v) {
		cout << "projective_space_subspaces::compute_decomposition done" << endl;
	}

}

void projective_space_subspaces::compute_decomposition_based_on_tally(
		data_structures::tally *T1,
		data_structures::tally *T2,
		incidence_structure *&Inc,
		data_structures::partitionstack *&Stack,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_subspaces::compute_decomposition_based_on_tally" << endl;
	}
	if (f_v) {
		cout << "projective_space_subspaces::compute_decomposition_based_on_tally "
				"before incidence_and_stack_for_type_ij" << endl;
	}
	incidence_and_stack_for_type_ij(
		1 /* row_type */, 2 /* col_type */,
		Inc,
		Stack,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_subspaces::compute_decomposition_based_on_tally "
				"after incidence_and_stack_for_type_ij" << endl;
	}

	int i, j, sz;

	for (i = T1->nb_types - 1; i >= 1; i--) {
		if (f_v) {
			cout << "projective_space_subspaces::compute_decomposition_based_on_tally "
					"before Stack->split_cell (S1) i=" << i << endl;
		}
		Stack->split_cell(
				T1->sorting_perm_inv + T1->type_first[i],
				T1->type_len[i], verbose_level);
	}
	int *set;
	set = NEW_int(Inc->nb_rows + Inc->nb_cols);
	for (i = T2->nb_types - 1; i >= 1; i--) {
		sz = T2->type_len[i];
		Int_vec_copy(T2->sorting_perm_inv + T2->type_first[i], set, sz);
		for (j = 0; j < sz; j++) {
			set[j] += Inc->nb_rows;
		}
		if (f_v) {
			cout << "projective_space_subspaces::compute_decomposition_based_on_tally "
					"before Stack->split_cell (S2) i=" << i << endl;
		}
		Stack->split_cell(set, sz, 0 /*verbose_level*/);
	}
	FREE_int(set);
	if (f_v) {
		cout << "projective_space_subspaces::compute_decomposition_based_on_tally done" << endl;
	}

}

void projective_space_subspaces::polarity_rank_k_subspace(int k,
		long int rk_in, long int &rk_out, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *A;
	int d;
	grassmann *Gr_k;
	grassmann *Gr_dmk;

	if (f_v) {
		cout << "projective_space_subspaces::polarity_rank_k_subspace, k=" << k << endl;
	}
	d = n + 1;
	Gr_k = Grass_stack[k];
	Gr_dmk = Grass_stack[d - k];

	A = NEW_int(d * d);

	Gr_k->unrank_lint_here(A, rk_in, 0 /*verbose_level - 4*/);
	if (f_vv) {
		cout << "projective_space_subspaces::polarity_rank_k_subspace "
				"subspace " << rk_in << ":" << endl;
		Int_vec_print_integer_matrix_width(cout,
			A, k, d, d,
			F->log10_of_q + 1);
	}
	F->Linear_algebra->perp_standard(d, k, A, 0);
	if (false) {
		cout << "projective_space_subspaces::polarity_rank_k_subspace "
				"after F->perp_standard:" << endl;
		Int_vec_print_integer_matrix_width(cout,
			A, d, d, d,
			F->log10_of_q + 1);
	}
	rk_out = Gr_dmk->rank_lint_here(A + k *d, 0 /*verbose_level - 4*/);
	if (f_vv) {
		cout << "projective_space_subspaces::polarity_rank_k_subspace "
				"perp is " << endl;
		Int_vec_print_integer_matrix_width(
				cout, A + k * d, d - k, d, d, F->log10_of_q + 1);
		cout << "which has rank " << rk_out << endl;
	}

	FREE_int(A);
	if (f_v) {
		cout << "projective_space_subspaces::polarity_rank_k_subspace done" << endl;
	}
}

void projective_space_subspaces::planes_through_a_line(
	long int line_rk, std::vector<long int> &plane_ranks,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int rk;
	int h, d, j, r;
	int *M1;
	int *M2;
	int *base_cols;
	int *embedding;
	int *w;
	int *v;
	int N;
	geometry_global Gg;

	if (f_v) {
		cout << "projective_space_subspaces::planes_through_a_line" << endl;
	}
	d = n + 1;
	M1 = NEW_int(3 * d);
	M2 = NEW_int(3 * d);
	base_cols = NEW_int(d);
	embedding = NEW_int(d);
	w = NEW_int(d);
	v = NEW_int(d);
	Grass_lines->unrank_lint_here(
			M1, line_rk, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space_subspaces::planes_through_a_line "
				"M1=" << endl;
		Int_matrix_print(M1, 2, d);
	}

	r = F->Linear_algebra->base_cols_and_embedding(
			2, d, M1,
			base_cols, embedding, 0 /* verbose_level */);
	if (r != 2) {
		cout << "projective_space_subspaces::planes_through_a_line r != 2" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "projective_space_subspaces::planes_through_a_line "
				"after RREF, M1=" << endl;
		Int_matrix_print(M1, 2, d);
	}
	N = Gg.nb_PG_elements(n - 2, F->q);

	for (h = 0; h < N; h++) {

		F->Projective_space_basic->PG_element_unrank_modified(
				w, 1, d - 2, h);
		Int_vec_zero(v, d);
		for (j = 0; j < d - 2; j++) {
			v[embedding[j]] = w[j];
		}
		Int_vec_copy(M1, M2, 2 * d);
		Int_vec_copy(v, M2 + 2 * d, d);
		if (false) {
			cout << "projective_space_subspaces::planes_through_a_line "
					"h = " << h << ", M2=" << endl;
			Int_matrix_print(M2, 3, d);
		}
		if (F->Linear_algebra->rank_of_rectangular_matrix(
				M2, 3, d, 0 /*verbose_level*/) == 3) {

			// here, rank means the rank in the sense of linear algebra

			if (f_v) {
				cout << "projective_space_subspaces::planes_through_a_line "
						"h = " << h << ", M2=" << endl;
				Int_matrix_print(M2, 3, d);
			}
			rk = Grass_planes->rank_lint_here(M2, 0 /* verbose_level */);

			// here rank is in the sense of indexing

			if (f_v) {
				cout << "projective_space_subspaces::planes_through_a_line "
						"h = " << h << " rk=" << rk << endl;
			}
			plane_ranks.push_back(rk);
		}
	} // next h
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(base_cols);
	FREE_int(embedding);
	FREE_int(w);
	FREE_int(v);
	if (f_v) {
		cout << "projective_space_subspaces::planes_through_a_line done" << endl;
	}
}

void projective_space_subspaces::planes_through_a_set_of_lines(
		long int *Lines, int nb_lines,
		long int *&Plane_ranks,
		int &nb_planes_on_one_line, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, d;
	int *M;

	if (f_v) {
		cout << "projective_space_subspaces::planes_through_a_set_of_lines" << endl;
	}

	d = n + 1;
	M = NEW_int(3 * d);

	for (i = 0; i < nb_lines; i++) {

		std::vector<long int> plane_ranks;

		planes_through_a_line(
				Lines[i], plane_ranks,
				0 /*verbose_level*/);

		nb_planes_on_one_line = plane_ranks.size();

		if (i == 0) {
			Plane_ranks = NEW_lint(nb_lines * nb_planes_on_one_line);
		}
		for (j = 0; j < plane_ranks.size(); j++) {
			Plane_ranks[i * nb_planes_on_one_line + j] = plane_ranks[j];
		}

		if (f_v) {
			cout << "planes through line " << Lines[i] << " : ";
			for (j = 0; j < plane_ranks.size(); j++) {
				cout << plane_ranks[j];
				if (j < plane_ranks.size() - 1) {
					cout << ",";
				}
			}
			cout << endl;

			cout << "planes through line " << Lines[i] << endl;
			for (j = 0; j < plane_ranks.size(); j++) {
				cout << j << " : " << plane_ranks[j] << " : " << endl;
				Grass_planes->unrank_lint_here(M, plane_ranks[j], 0 /* verbose_level */);
				Int_matrix_print(M, 3, d);

			}
			cout << endl;
		}


	}
	FREE_int(M);
	if (f_v) {
		cout << "projective_space_subspaces::planes_through_a_set_of_lines done" << endl;
	}

}

void projective_space_subspaces::plane_intersection(
		int plane_rank,
		long int *set, int set_size,
		std::vector<int> &point_indices,
		std::vector<int> &point_local_coordinates,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int r, i, u, d;

	int *Basis;
	int *Basis_save;
	int *Coords;
	int base_cols[3];
	int coefficients[3];
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space_subspaces::plane_intersection" << endl;
	}
	d = n + 1;
	// allocate temporary data:
	Basis = NEW_int(4 * d);
	Basis_save = NEW_int(4 * d);
	Coords = NEW_int(set_size * d);

	for (i = 0; i < set_size; i++) {
		unrank_point(Coords + i * d, set[i]);
	}
	if (f_vv) {
		cout << "projective_space_subspaces::plane_intersection "
				"Coords:" << endl;
		Int_matrix_print(Coords, set_size, d);
	}

	Grass_planes->unrank_lint_here(
			Basis_save, plane_rank, 0 /* verbose_level */);

	int nb_pts_on_plane = 0;
	long int local_rank;

	for (u = 0; u < set_size; u++) {
		Int_vec_copy(Basis_save, Basis, 3 * d);
		Int_vec_copy(Coords + u * d, Basis + 3 * d, d);
		r = F->Linear_algebra->rank_of_rectangular_matrix(
				Basis,
				4, d, 0 /* verbose_level */);
		if (r < 4) {
			nb_pts_on_plane++;
			point_indices.push_back(u);

			Int_vec_copy(Basis_save, Basis, 3 * d);
			Int_vec_copy(Coords + u * d, Basis + 3 * d, d);

			F->Linear_algebra->Gauss_simple(
					Basis, 3, d,
					base_cols, 0 /*verbose_level */);
			F->Linear_algebra->reduce_mod_subspace_and_get_coefficient_vector(
				3, d, Basis, base_cols,
				Basis + 3 * d, coefficients, verbose_level);
			F->Projective_space_basic->PG_element_rank_modified(
					coefficients, 1, 3, local_rank);
			point_local_coordinates.push_back(local_rank);
		}
	}

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(Coords);
	if (f_v) {
		cout << "projective_space_subspaces::plane_intersection "
				"done" << endl;
	}
}

void projective_space_subspaces::line_intersection(
		int line_rank,
		long int *set, int set_size,
		std::vector<int> &point_indices,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int r, i, u, d;

	int *Basis;
	int *Basis_save;
	int *Coords;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "projective_space_subspaces::line_intersection" << endl;
	}
	d = n + 1;
	// allocate temporary data:
	Basis = NEW_int(3 * d);
	Basis_save = NEW_int(3 * d);
	Coords = NEW_int(set_size * d);

	for (i = 0; i < set_size; i++) {
		unrank_point(Coords + i * d, set[i]);
	}
	if (f_vv) {
		cout << "projective_space_subspaces::line_intersection "
				"Coords:" << endl;
		Int_matrix_print(Coords, set_size, d);
	}

	Grass_lines->unrank_lint_here(
			Basis_save, line_rank,
			0 /* verbose_level */);

	for (u = 0; u < set_size; u++) {
		Int_vec_copy(Basis_save, Basis, 2 * d);
		Int_vec_copy(Coords + u * d, Basis + 2 * d, d);
		r = F->Linear_algebra->rank_of_rectangular_matrix(Basis,
				3, d, 0 /* verbose_level */);
		if (r < 3) {
			point_indices.push_back(u);
		}
	}

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(Coords);
	if (f_v) {
		cout << "projective_space_subspaces::line_intersection "
				"done" << endl;
	}
}



}}}


