// projective_space.cpp
//
// Anton Betten
// Jan 17, 2010

#include "foundations.h"


using namespace std;


#define MAX_NUMBER_OF_LINES_FOR_INCIDENCE_MATRIX 100000
#define MAX_NUMBER_OF_LINES_FOR_LINE_TABLE 1000000
#define MAX_NUMBER_OF_POINTS_FOR_POINT_TABLE 1000000
#define MAX_NB_POINTS_FOR_LINE_THROUGH_TWO_POINTS_TABLE 10000
#define MAX_NB_POINTS_FOR_LINE_INTERSECTION_TABLE 10000


namespace orbiter {
namespace foundations {


projective_space::projective_space()
{
	Grass_lines = NULL;
	Grass_planes = NULL;
	Grass_hyperplanes = NULL;
	F = NULL;
	Go = NULL;
	n = 0;
	q = 0;

	N_points = 0;
	N_lines = 0;

	Nb_subspaces = NULL;

	r = 0;
	k = 0;


	Implementation = NULL;

	Standard_polarity = NULL;
	Reversal_polarity = NULL;

	v = NULL;
	w = NULL;
	Mtx = NULL;
	Mtx2 = NULL;
}



projective_space::~projective_space()
{
	freeself();
}


void projective_space::freeself()
{
	int f_v = FALSE;

	if (f_v) {
		cout << "projective_space::freeself" << endl;
	}
	if (Go) {
		if (f_v) {
			cout << "projective_space::freeself deleting Go" << endl;
		}
		FREE_OBJECT(Go);
	}
	if (Nb_subspaces) {
		FREE_lint(Nb_subspaces);
	}
	if (v) {
		if (f_v) {
			cout << "projective_space::freeself deleting v" << endl;
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

	if (Grass_lines) {
		if (f_v) {
			cout << "projective_space::freeself "
					"deleting Grass_lines" << endl;
		}
		FREE_OBJECT(Grass_lines);
	}
	if (Grass_planes) {
		if (f_v) {
			cout << "projective_space::freeself "
					"deleting Grass_planes" << endl;
		}
		FREE_OBJECT(Grass_planes);
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
		cout << "projective_space::freeself done" << endl;
	}
}

void projective_space::init(int n, finite_field *F, 
	int f_init_incidence_structure, 
	int verbose_level)
// n is projective dimension
{
	int f_v = (verbose_level >= 1);
	int i;
	combinatorics_domain C;
	longinteger_object a;

	projective_space::n = n;
	projective_space::F = F;
	projective_space::q = F->q;
	
	if (f_v) {
		cout << "projective_space::init PG(" << n << "," << q << ")" << endl;
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

	if (f_v) {
		cout << "projective_space::init computing number of "
				"subspaces of each dimension:" << endl;
	}
	Nb_subspaces = NEW_lint(n + 1);
	if (n < 10) {
		for (i = 0; i <= n; i++) {
			if (f_v) {
				cout << "projective_space::init computing number of "
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
				cout << "projective_space::init computing number of "
						"subspaces of dimension " << i + 1 << endl;
				}
			Nb_subspaces[i] = 0;
		}
		r = 0;
	}
	N_points = Nb_subspaces[0]; // generalized_binomial(n + 1, 1, q);
	if (f_v) {
		cout << "projective_space::init N_points=" << N_points << endl;
	}
	N_lines = Nb_subspaces[1]; // generalized_binomial(n + 1, 2, q);
	if (f_v) {
		cout << "projective_space::init N_lines=" << N_lines << endl;
	}
	if (f_v) {
		cout << "projective_space::init r=" << r << endl;
	}
	k = q + 1; // number of points on a line
	if (f_v) {
		cout << "projective_space::init k=" << k << endl;
	}




	if (f_init_incidence_structure) {
		if (f_v) {
			cout << "projective_space::init calling "
					"init_incidence_structure" << endl;
		}
		init_incidence_structure(verbose_level);
		if (f_v) {
			cout << "projective_space::init "
					"init_incidence_structure done" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "projective_space::init we don't initialize "
					"the incidence structure data" << endl;
		}
	}
	if (f_v) {
		
		cout << "projective_space::init n=" << n
				<< " q=" << q << " done" << endl;
	}
}

void projective_space::init_incidence_structure(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::init_incidence_structure" << endl;
	}

	Implementation = NEW_OBJECT(projective_space_implementation);


	if (f_v) {
		cout << "projective_space::init_incidence_structure "
				"before projective_space_implementation->init" << endl;
	}
	Implementation->init(this, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space::init_incidence_structure "
				"after projective_space_implementation->init" << endl;
	}


	if (n >= 2) {
		if (f_v) {
			cout << "projective_space_implementation::init before init_polarity" << endl;
		}

		init_polarity(verbose_level);


		if (f_v) {
			cout << "projective_space_implementation::init after init_polarity" << endl;
		}
	}


	
	if (f_v) {
		cout << "projective_space::init_incidence_structure done" << endl;
	}
}

void projective_space::init_polarity(int verbose_level)
// uses Grass_hyperplanes
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::init_polarity" << endl;
	}
	Standard_polarity = NEW_OBJECT(polarity);

	if (f_v) {
		cout << "projective_space::init_polarity before Standard_polarity->init_standard_polarity" << endl;
	}
	Standard_polarity->init_standard_polarity(this, verbose_level);
	if (f_v) {
		cout << "projective_space::init_polarity after Standard_polarity->init_standard_polarity" << endl;
	}

	Reversal_polarity = NEW_OBJECT(polarity);

	if (f_v) {
		cout << "projective_space::init_polarity before Standard_polarity->init_reversal_polarity" << endl;
	}
	Reversal_polarity->init_reversal_polarity(this, verbose_level);
	if (f_v) {
		cout << "projective_space::init_polarity after Standard_polarity->init_reversal_polarity" << endl;
	}



	if (f_v) {
		cout << "projective_space::init_polarity done" << endl;
	}

}
void projective_space::intersect_with_line(long int *set, int set_sz,
		int line_rk, long int *intersection, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, b, idx;
	long int a;
	int *L;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::intersect_with_line" << endl;
	}
	L = Implementation->Lines + line_rk * k;
	sz = 0;
	for (i = 0; i < set_sz; i++) {
		a = set[i];
		b = a;
		if (b != a) {
			cout << "projective_space::intersect_with_line data loss" << endl;
			exit(1);
		}
		if (Sorting.int_vec_search(L, k, b, idx)) {
			intersection[sz++] = a;
		}
	}
	if (f_v) {
		cout << "projective_space::intersect_with_line done" << endl;
	}
}

void projective_space::create_points_on_line(
	long int line_rk, long int *line, int verbose_level)
// needs line[k]
{
	int a, b;
	
	Grass_lines->unrank_lint(line_rk, 0/*verbose_level - 4*/);
	for (a = 0; a < k; a++) {
		F->PG_element_unrank_modified(v, 1, 2, a);
		F->mult_matrix_matrix(v, Grass_lines->M, w, 1, 2, n + 1,
				0 /* verbose_level */);
		F->PG_element_rank_modified(w, 1, n + 1, b);
		line[a] = b;
	}
}

void projective_space::create_lines_on_point(
	long int point_rk, long int *line_pencil, int verbose_level)
{
	int a, b, i, d;
	int *v;
	int *w;
	int *Basis;

	d = n + 1;
	v = NEW_int(d);
	w = NEW_int(n);
	Basis = NEW_int(2 * d);

	F->PG_element_unrank_modified(v, 1, d, point_rk);
	for (i = 0; i < n + 1; i++) {
		if (v[i]) {
			break;
		}
	}
	if (i == n + 1) {
		cout << "projective_space::create_lines_on_point zero vector" << endl;
		exit(1);
	}
	for (a = 0; a < r; a++) {
		F->PG_element_unrank_modified(w, 1, n, a);
		Orbiter->Int_vec.copy(v, Basis, d);
		Orbiter->Int_vec.copy(w, Basis + d, i);
		Basis[d + i] = 0;
		Orbiter->Int_vec.copy(w + i, Basis + d + i + 1, n - i);
		b = Grass_lines->rank_lint_here(Basis, 0 /*verbose_level*/);
		line_pencil[a] = b;
	}
	FREE_int(v);
	FREE_int(w);
	FREE_int(Basis);
}

void projective_space::create_lines_on_point_but_inside_a_plane(
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
		cout << "projective_space::create_lines_on_point_but_inside_a_plane" << endl;
	}
	if (n < 3) {
		cout << "projective_space::create_lines_on_point_but_inside_a_plane n < 3" << endl;
		exit(1);
	}
	d = n + 1;
	v = NEW_int(d);
	w = NEW_int(n);
	Basis = NEW_int(2 * d);
	Plane = NEW_int(3 * d);
	M = NEW_int(4 * d);

	Grass_planes->unrank_lint_here(Plane, plane_rk, 0 /*verbose_level*/);

	F->PG_element_unrank_modified(v, 1, d, point_rk);
	for (idx = 0; idx < n + 1; idx++) {
		if (v[idx]) {
			break;
		}
	}
	if (idx == n + 1) {
		cout << "projective_space::create_lines_on_point_but_inside_a_plane zero vector" << endl;
		exit(1);
	}
	i = 0;
	for (a = 0; a < r; a++) {
		F->PG_element_unrank_modified(w, 1, n, a);
		Orbiter->Int_vec.copy(v, Basis, d);
		Orbiter->Int_vec.copy(w, Basis + d, idx);
		Basis[d + idx] = 0;
		Orbiter->Int_vec.copy(w + idx, Basis + d + idx + 1, n - idx);


		Orbiter->Int_vec.copy(Plane, M, 3 * d);
		Orbiter->Int_vec.copy(Basis + d, M + 3 * d, d);
		rk = F->rank_of_rectangular_matrix(M, 4, d, 0 /* verbose_level*/);
		if (rk == 3) {
			b = Grass_lines->rank_lint_here(Basis, 0 /*verbose_level*/);
			line_pencil[i++] = b;
		}

	}
	if (i != q + 1) {
		cout << "projective_space::create_lines_on_point_but_inside_a_plane  i != q + 1" << endl;
		exit(1);
	}
	FREE_int(v);
	FREE_int(w);
	FREE_int(Basis);
	FREE_int(Plane);
	FREE_int(M);
	if (f_v) {
		cout << "projective_space::create_lines_on_point_but_inside_a_plane done" << endl;
	}
}


int projective_space::create_point_on_line(
		long int line_rk, int pt_rk, int verbose_level)
// pt_rk is between 0 and q-1.
{
	int f_v = (verbose_level >= 1);
	int b;
	int v[2];

	if (f_v) {
		cout << "projective_space::create_point_on_line" << endl;
	}
	Grass_lines->unrank_lint(line_rk, 0/*verbose_level - 4*/);
	if (f_v) {
		cout << "projective_space::create_point_on_line line:" << endl;
		Orbiter->Int_vec.matrix_print(Grass_lines->M, 2, n + 1);
	}

	F->PG_element_unrank_modified(v, 1, 2, pt_rk);
	if (f_v) {
		cout << "projective_space::create_point_on_line v=" << endl;
		Orbiter->Int_vec.print(cout, v, 2);
		cout << endl;
	}

	F->mult_matrix_matrix(v, Grass_lines->M, w, 1, 2, n + 1,
			0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space::create_point_on_line w=" << endl;
		Orbiter->Int_vec.print(cout, w, n + 1);
		cout << endl;
	}

	F->PG_element_rank_modified(w, 1, n + 1, b);

	if (f_v) {
		cout << "projective_space::create_point_on_line b = " << b << endl;
	}
	return b;
}

void projective_space::make_incidence_matrix(
	int &m, int &n,
	int *&Inc, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, h, j;

	if (f_v) {
		cout << "projective_space::make_incidence_matrix" << endl;
	}
	m = N_points;
	n = N_lines;
	Inc = NEW_int(m * n);
	Orbiter->Int_vec.zero(Inc, m * n);
	for (i = 0; i < N_points; i++) {
		for (h = 0; h < r; h++) {
			j = Implementation->Lines_on_point[i * r + h];
			Inc[i * n + j] = 1;
		}
	}
	if (f_v) {
		cout << "projective_space::make_incidence_matrix "
				"done" << endl;
	}
}

void projective_space::make_incidence_matrix(
	std::vector<int> &Pts, std::vector<int> &Lines,
	int *&Inc, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b;
	int nb_pts, nb_lines;

	if (f_v) {
		cout << "projective_space::make_incidence_matrix" << endl;
	}

	nb_pts = Pts.size();
	nb_lines = Lines.size();
	Inc = NEW_int(nb_pts * nb_lines);
	Orbiter->Int_vec.zero(Inc, nb_pts * nb_lines);
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
		cout << "projective_space::make_incidence_matrix done" << endl;
	}
}

int projective_space::is_incident(int pt, int line)
{
	int f_v = FALSE;
	long int rk;

	if (TRUE /*incidence_bitvec == NULL*/) {
#if 0
		cout << "projective_space::is_incident "
				"incidence_bitvec == 0" << endl;
		exit(1);
#endif
		Grass_lines->unrank_lint(line, 0/*verbose_level - 4*/);

		if (f_v) {
			cout << "projective_space::is_incident "
					"line=" << endl;
			Orbiter->Int_vec.matrix_print(Grass_lines->M, 2, n + 1);
		}
		Orbiter->Int_vec.copy(Grass_lines->M, Mtx, 2 * (n + 1));
		F->PG_element_unrank_modified(Mtx + 2 * (n + 1), 1, n + 1, pt);
		if (f_v) {
			cout << "point:" << endl;
			Orbiter->Int_vec.print(cout, Mtx + 2 * (n + 1), n + 1);
			cout << endl;
		}

		rk = F->rank_of_rectangular_matrix_memory_given(Mtx,
				3, n + 1, Mtx2, v /* base_cols */,
				0 /*verbose_level*/);
		if (f_v) {
			cout << "rk = " << rk << endl;
		}
		if (rk == 3) {
			return FALSE;
		}
		else {
			return TRUE;
		}
	}
	else {
		//a = (long int) pt * (long int) N_lines + (long int) line;
		//return bitvector_s_i(incidence_bitvec, a);
		return Implementation->Bitmatrix->s_ij(pt, line);
	}
}

void projective_space::incidence_m_ii(int pt, int line, int a)
{
	//long int b;

	if (Implementation->Bitmatrix == NULL) {
		cout << "projective_space::incidence_m_ii "
				"Bitmatrix == NULL" << endl;
		exit(1);
		}
	//b = (long int) pt * (long int) N_lines + (long int) line;
	//bitvector_m_ii(incidence_bitvec, b, a);

	Implementation->Bitmatrix->m_ij(pt, N_lines, a);
}

void projective_space::make_incidence_structure_and_partition(
	incidence_structure *&Inc, 
	partitionstack *&Stack, int verbose_level)
// points vs lines
{
	int f_v = (verbose_level >= 1);
	int *M;
	int i, j, h;

	if (f_v) {
		cout << "projective_space::make_incidence_structure_and_partition" << endl;
		cout << "N_points=" << N_points << endl;
		cout << "N_lines=" << N_lines << endl;
	}
	Inc = NEW_OBJECT(incidence_structure);

	
	if (f_v) {
		cout << "projective_space::make_incidence_structure_and_partition "
				"allocating M of size "
				<< N_points * N_lines << endl;
	}
	M = NEW_int(N_points * N_lines);
	if (f_v) {
		cout << "projective_space::make_incidence_structure_and_partition "
				"after allocating M of size "
				<< N_points * N_lines << endl;
	}
	Orbiter->Int_vec.zero(M, N_points * N_lines);

	if (Implementation->Lines_on_point == NULL) {
		cout << "projective_space::make_incidence_structure_and_partition "
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
		cout << "projective_space::make_incidence_structure_and_partition "
				"before Inc->init_by_matrix" << endl;
	}
	Inc->init_by_matrix(N_points, N_lines, M, verbose_level - 1);
	if (f_v) {
		cout << "projective_space::make_incidence_structure_and_partition "
				"after Inc->init_by_matrix" << endl;
	}
	FREE_int(M);


	Stack = NEW_OBJECT(partitionstack);
	Stack->allocate(N_points + N_lines, 0 /* verbose_level */);
	Stack->subset_continguous(N_points, N_lines);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();
	
	if (f_v) {
		cout << "projective_space::make_incidence_structure_and_partition done" << endl;
	}
}

void projective_space::incma_for_type_ij(
	int row_type, int col_type,
	int *&Incma, int &nb_rows, int &nb_cols,
	int verbose_level)
// row_type, col_type are the vector space dimensions of the objects
// indexing rows and columns.
{
	int f_v = (verbose_level >= 1);
	int i, j, rk;
	int *Basis;
	int *Basis2;
	int *base_cols;
	int d = n + 1;

	if (f_v) {
		cout << "projective_space::incma_for_type_ij" << endl;
		cout << "row_type = " << row_type << endl;
		cout << "col_type = " << col_type << endl;
	}
	if (col_type < row_type) {
		cout << "projective_space::incma_for_type_ij "
				"col_type < row_type" << endl;
		exit(1);
	}
	if (col_type < 0) {
		cout << "projective_space::incma_for_type_ij "
				"col_type < 0" << endl;
		exit(1);
	}
	if (col_type > n + 1) {
		cout << "projective_space::incma_for_type_ij "
				"col_type > P->n + 1" << endl;
		exit(1);
	}
	nb_rows = nb_rk_k_subspaces_as_lint(row_type);
	nb_cols = nb_rk_k_subspaces_as_lint(col_type);


	Basis = NEW_int(3 * d * d);
	Basis2 = NEW_int(3 * d * d);
	base_cols = NEW_int(d);

	Incma = NEW_int(nb_rows * nb_cols);
	Orbiter->Int_vec.zero(Incma, nb_rows * nb_cols);
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
			cout << "projective_space::incma_for_type_ij "
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
				cout << "projective_space::incma_for_type_ij "
						"col_type " << col_type
					<< " not yet implemented" << endl;
				exit(1);
			}
			rk = F->rank_of_rectangular_matrix_memory_given(Basis,
					row_type + col_type, d, Basis2, base_cols,
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
		cout << "projective_space::incma_for_type_ij done" << endl;
	}
}

void projective_space::incidence_and_stack_for_type_ij(
	int row_type, int col_type,
	incidence_structure *&Inc,
	partitionstack *&Stack,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Incma;
	int nb_rows, nb_cols;

	if (f_v) {
		cout << "projective_space::incidence_and_stack_for_type_ij" << endl;
	}
	incma_for_type_ij(
		row_type, col_type,
		Incma, nb_rows, nb_cols,
		verbose_level);
	if (f_v) {
		cout << "projective_space::incidence_and_stack_for_type_ij "
				"before Inc->init_by_matrix" << endl;
	}
	Inc = NEW_OBJECT(incidence_structure);
	Inc->init_by_matrix(nb_rows, nb_cols, Incma, verbose_level - 1);
	if (f_v) {
		cout << "projective_space::incidence_and_stack_for_type_ij "
				"after Inc->init_by_matrix" << endl;
	}
	FREE_int(Incma);


	Stack = NEW_OBJECT(partitionstack);
	Stack->allocate(nb_rows + nb_cols, 0 /* verbose_level */);
	Stack->subset_continguous(nb_rows, nb_cols);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();

	if (f_v) {
		cout << "projective_space::incidence_and_stack_for_type_ij done" << endl;
	}
}

long int projective_space::nb_rk_k_subspaces_as_lint(int k)
{
	combinatorics_domain C;
	longinteger_object aa;
	long int N;
	int d = n + 1;

	C.q_binomial(aa, d, k, q, 0/*verbose_level*/);
	N = aa.as_lint();
	return N;
}

void projective_space::print_set_of_points(ostream &ost, long int *Pts, int nb_pts)
{
	int h, I;
	int *v;

	v = NEW_int(n + 1);

	for (I = 0; I < (nb_pts + 39) / 40; I++) {
		ost << "$$" << endl;
		ost << "\\begin{array}{|r|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "i & \\mbox{Rank} & \\mbox{Point} \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;
		for (h = 0; h < 40; h++) {
			if (I * 40 + h < nb_pts) {
				unrank_point(v, Pts[I * 40 + h]);
				ost << I * 40 + h << " & " << Pts[I * 40 + h] << " & ";
				Orbiter->Int_vec.print(ost, v, n + 1);
				ost << "\\\\" << endl;
			}
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
	}
	FREE_int(v);
}

void projective_space::print_all_points()
{
	int *v;
	int i;

	v = NEW_int(n + 1);
	cout << "All points in PG(" << n << "," << q << "):" << endl;
	for (i = 0; i < N_points; i++) {
		unrank_point(v, i);
		cout << setw(3) << i << " : ";
		Orbiter->Int_vec.print(cout, v, n + 1);
		cout << endl;
	}
	FREE_int(v);
}

long int projective_space::rank_point(int *v)
{
	long int b;
	int verbose_level = 0;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::rank_point" << endl;
	}
	
	F->PG_element_rank_modified_lint(v, 1, n + 1, b);

	if (f_v) {
		cout << "projective_space::rank_point done" << endl;
	}
	return b;
}

void projective_space::unrank_point(int *v, long int rk)
{	
	F->PG_element_unrank_modified_lint(v, 1, n + 1, rk);
}

void projective_space::unrank_points(int *v, long int *Rk, int sz)
{
	int i;

	for (i = 0; i < sz; i++) {
		F->PG_element_unrank_modified_lint(v + i * (n + 1), 1, n + 1, Rk[i]);
	}
}

long int projective_space::rank_line(int *basis)
{
	long int b;
	
	b = Grass_lines->rank_lint_here(basis, 0/*verbose_level - 4*/);
	return b;
}

void projective_space::unrank_line(int *basis, long int rk)
{	
	Grass_lines->unrank_lint_here(basis, rk, 0/*verbose_level - 4*/);
}

void projective_space::unrank_lines(int *v, long int *Rk, int nb)
{
	int i;
	
	for (i = 0; i < nb; i++) {
		Grass_lines->unrank_lint_here(
				v + i * 2 * (n + 1), Rk[i], 0 /* verbose_level */);
	}
}

long int projective_space::rank_plane(int *basis)
{
	long int b;

	if (Grass_planes == NULL) {
		cout << "projective_space::rank_plane "
				"Grass_planes == NULL" << endl;
		exit(1);
	}
	b = Grass_planes->rank_lint_here(basis, 0/*verbose_level - 4*/);
	return b;
}

void projective_space::unrank_plane(int *basis, long int rk)
{	
	if (Grass_planes == NULL) {
		cout << "projective_space::unrank_plane "
				"Grass_planes == NULL" << endl;
		exit(1);
	}
	Grass_planes->unrank_lint_here(basis, rk, 0/*verbose_level - 4*/);
}

long int projective_space::line_through_two_points(
		long int p1, long int p2)
{
	long int b;
	
	unrank_point(Grass_lines->M, p1);
	unrank_point(Grass_lines->M + n + 1, p2);
	b = Grass_lines->rank_lint(0/*verbose_level - 4*/);
	return b;
}

int projective_space::test_if_lines_are_disjoint(
		long int l1, long int l2)
{
	sorting Sorting;

	if (Implementation->Lines) {
		return Sorting.test_if_sets_are_disjoint_assuming_sorted(
				Implementation->Lines + l1 * k, Implementation->Lines + l2 * k, k, k);
	}
	else {
		return test_if_lines_are_disjoint_from_scratch(l1, l2);
	}
}

int projective_space::test_if_lines_are_disjoint_from_scratch(
		long int l1, long int l2)
{
	int *Mtx;
	int m, rk;

	m = n + 1;
	Mtx = NEW_int(4 * m);
	Grass_lines->unrank_lint_here(Mtx, l1, 0/*verbose_level - 4*/);
	Grass_lines->unrank_lint_here(Mtx + 2 * m, l2, 0/*verbose_level - 4*/);
	rk = F->Gauss_easy(Mtx, 4, m);
	FREE_int(Mtx);
	if (rk == 4) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int projective_space::intersection_of_two_lines(long int l1, long int l2)
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
	F->perp_standard(d, 2, Mtx1, 0);
	Orbiter->Int_vec.copy(Mtx1 + 2 * d, Mtx3, (d - 2) * d);
	Grass_lines->unrank_lint_here(Mtx1, l2, 0/*verbose_level - 4*/);
	F->perp_standard(d, 2, Mtx1, 0);
	Orbiter->Int_vec.copy(Mtx1 + 2 * d, Mtx3 + (d - 2) * d, (d - 2) * d);
	r = F->Gauss_easy(Mtx3, 2 * (d - 2), d);
	if (r < d - 1) {
		cout << "projective_space::intersection_of_two_lines r < d - 1, "
				"the lines do not intersect" << endl;
		exit(1);
	}
	if (r > d - 1) {
		cout << "projective_space::intersection_of_two_lines r > d - 1, "
				"something is wrong" << endl;
		exit(1);
	}
	F->perp_standard(d, d - 1, Mtx3, 0);
	b = rank_point(Mtx3 + (d - 1) * d);

	FREE_int(Mtx1);
	FREE_int(Mtx3);
	
	return b;
}

int projective_space::arc_test(long int *input_pts, int nb_pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Pts;
	int *Mtx;
	int set[3];
	int ret = TRUE;
	int h, i, N;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "projective_space::arc_test" << endl;
	}
	if (n != 2) {
		cout << "projective_space::arc_test n != 2" << endl;
		exit(1);
	}
	Pts = NEW_int(nb_pts * 3);
	Mtx = NEW_int(3 * 3);
	for (i = 0; i < nb_pts; i++) {
		unrank_point(Pts + i * 3, input_pts[i]);
	}
	if (f_v) {
		cout << "projective_space::arc_test Pts=" << endl;
		Orbiter->Int_vec.matrix_print(Pts, nb_pts, 3);
	}
	N = Combi.int_n_choose_k(nb_pts, 3);
	for (h = 0; h < N; h++) {
		Combi.unrank_k_subset(h, set, nb_pts, 3);
		Orbiter->Int_vec.copy(Pts + set[0] * 3, Mtx, 3);
		Orbiter->Int_vec.copy(Pts + set[1] * 3, Mtx + 3, 3);
		Orbiter->Int_vec.copy(Pts + set[2] * 3, Mtx + 6, 3);
		if (F->rank_of_matrix(Mtx, 3, 0 /* verbose_level */) < 3) {
			if (f_v) {
				cout << "Points P_" << set[0] << ", P_" << set[1]
					<< " and P_" << set[2] << " are collinear" << endl;
			}
			ret = FALSE;
		}
	}

	FREE_int(Pts);
	FREE_int(Mtx);
	if (f_v) {
		cout << "projective_space::arc_test done" << endl;
	}
	return ret;
}

int projective_space::determine_line_in_plane(
	long int *two_input_pts,
	int *three_coeffs, 
	int verbose_level)
// returns FALSE is the rank of the coefficient matrix is not 2.
// TRUE otherwise.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *coords; // [nb_pts * 3];
	int *system; // [nb_pts * 3];
	int kernel[3 * 3];
	int base_cols[3];
	int i, x, y, z, rk;
	int kernel_m, kernel_n;
	int nb_pts = 2;

	if (f_v) {
		cout << "projective_space::determine_line_in_plane" << endl;
	}
	if (n != 2) {
		cout << "projective_space::determine_line_in_plane "
				"n != 2" << endl;
		exit(1);
	}



	coords = NEW_int(nb_pts * 3);
	system = NEW_int(nb_pts * 3);
	for (i = 0; i < nb_pts; i++) {
		unrank_point(coords + i * 3, two_input_pts[i]);
	}
	if (f_vv) {
		cout << "projective_space::determine_line_in_plane "
				"points:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				coords, nb_pts, 3, 3, F->log10_of_q);
	}
	for (i = 0; i < nb_pts; i++) {
		x = coords[i * 3 + 0];
		y = coords[i * 3 + 1];
		z = coords[i * 3 + 2];
		system[i * 3 + 0] = x;
		system[i * 3 + 1] = y;
		system[i * 3 + 2] = z;
	}
	if (f_v) {
		cout << "projective_space::determine_line_in_plane system:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				system, nb_pts, 3, 3, F->log10_of_q);
	}



	rk = F->Gauss_simple(system,
			nb_pts, 3, base_cols, verbose_level - 2);
	if (rk != 2) {
		if (f_v) {
			cout << "projective_space::determine_line_in_plane "
					"system undetermined" << endl;
		}
		return FALSE;
	}
	F->matrix_get_kernel(system, 2, 3, base_cols, rk, 
		kernel_m, kernel_n, kernel, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space::determine_line_in_plane line:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, kernel, 1, 3, 3, F->log10_of_q);
	}
	for (i = 0; i < 3; i++) {
		three_coeffs[i] = kernel[i];
	}
	FREE_int(coords);
	FREE_int(system);
	return TRUE;
}

int projective_space::nonconical_six_arc_get_nb_Eckardt_points(
		long int *Arc6, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::nonconical_six_arc_get_nb_Eckardt_points" << endl;
	}
	eckardt_point_info *E;
	int nb_E;
	surface_domain *Surf = NULL;

	E = compute_eckardt_point_info(Surf, Arc6, 0/*verbose_level*/);



	nb_E = E->nb_E;

	FREE_OBJECT(E);
	return nb_E;
}

int projective_space::test_nb_Eckardt_points(surface_domain *Surf,
		long int *S, int len, int pt, int nb_E, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = TRUE;
	long int Arc6[6];

	if (f_v) {
		cout << "projective_space::test_nb_Eckardt_points" << endl;
	}
	if (len != 5) {
		return TRUE;
	}

	Orbiter->Lint_vec.copy(S, Arc6, 5);
	Arc6[5] = pt;

	eckardt_point_info *E;

	E = compute_eckardt_point_info(Surf, Arc6, 0/*verbose_level*/);


	if (E->nb_E != nb_E) {
		ret = FALSE;
	}

	FREE_OBJECT(E);

	if (f_v) {
		cout << "projective_space::test_nb_Eckardt_points done" << endl;
	}
	return ret;
}


int projective_space::conic_test(long int *S, int len, int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = TRUE;
	int subset[5];
	long int the_set[6];
	int six_coeffs[6];
	int i;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "projective_space::conic_test" << endl;
	}
	if (len < 5) {
		return TRUE;
	}
	Combi.first_k_subset(subset, len, 5);
	while (TRUE) {
		for (i = 0; i < 5; i++) {
			the_set[i] = S[subset[i]];
		}
		the_set[5] = pt;
		if (determine_conic_in_plane(
				the_set, 6, six_coeffs, 0 /*verbose_level*/)) {
			ret = FALSE;
			break;
		}

		if (!Combi.next_k_subset(subset, len, 5)) {
			ret = TRUE;
			break;
		}
	}
	if (f_v) {
		cout << "projective_space::conic_test done" << endl;
	}
	return ret;
}


int projective_space::test_if_conic_contains_point(int *six_coeffs, int pt)
{
	int v[3];
	int c[6];
	int x, y, z, s, i;

	unrank_point(v, pt);
	x = v[0];
	y = v[1];
	z = v[2];
	c[0] = F->mult(x, x);
	c[1] = F->mult(y, y);
	c[2] = F->mult(z, z);
	c[3] = F->mult(x, y);
	c[4] = F->mult(x, z);
	c[5] = F->mult(y, z);
	s = 0;
	for (i = 0; i < 6; i++) {
		s = F->add(s, F->mult(six_coeffs[i], c[i]));
	}
	if (s == 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
 }

int projective_space::determine_conic_in_plane(
	long int *input_pts, int nb_pts,
	int *six_coeffs, 
	int verbose_level)
// returns FALSE if the rank of the coefficient
// matrix is not 5. TRUE otherwise.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *coords; // [nb_pts * 3];
	int *system; // [nb_pts * 6];
	int kernel[6 * 6];
	int base_cols[6];
	int i, x, y, z, rk;
	int kernel_m, kernel_n;

	if (f_v) {
		cout << "projective_space::determine_conic_in_plane" << endl;
	}
	if (n != 2) {
		cout << "projective_space::determine_conic_in_plane "
				"n != 2" << endl;
		exit(1);
	}
	if (nb_pts < 5) {
		cout << "projective_space::determine_conic_in_plane "
				"need at least 5 points" << endl;
		exit(1);
	}

	if (!arc_test(input_pts, nb_pts, verbose_level)) {
		if (f_v) {
			cout << "projective_space::determine_conic_in_plane "
					"some 3 of the points are collinear" << endl;
		}
		return FALSE;
	}


	coords = NEW_int(nb_pts * 3);
	system = NEW_int(nb_pts * 6);
	for (i = 0; i < nb_pts; i++) {
		unrank_point(coords + i * 3, input_pts[i]);
	}
	if (f_vv) {
		cout << "projective_space::determine_conic_in_plane "
				"points:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				coords, nb_pts, 3, 3, F->log10_of_q);
	}
	for (i = 0; i < nb_pts; i++) {
		x = coords[i * 3 + 0];
		y = coords[i * 3 + 1];
		z = coords[i * 3 + 2];
		system[i * 6 + 0] = F->mult(x, x);
		system[i * 6 + 1] = F->mult(y, y);
		system[i * 6 + 2] = F->mult(z, z);
		system[i * 6 + 3] = F->mult(x, y);
		system[i * 6 + 4] = F->mult(x, z);
		system[i * 6 + 5] = F->mult(y, z);
	}
	if (f_v) {
		cout << "projective_space::determine_conic_in_plane "
				"system:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				system, nb_pts, 6, 6, F->log10_of_q);
	}



	rk = F->Gauss_simple(system, nb_pts,
			6, base_cols, verbose_level - 2);
	if (rk != 5) {
		if (f_v) {
			cout << "projective_space::determine_conic_in_plane "
					"system undetermined" << endl;
		}
		return FALSE;
	}
	F->matrix_get_kernel(system, 5, 6, base_cols, rk, 
		kernel_m, kernel_n, kernel, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space::determine_conic_in_plane "
				"conic:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				kernel, 1, 6, 6, F->log10_of_q);
	}
	for (i = 0; i < 6; i++) {
		six_coeffs[i] = kernel[i];
	}
	FREE_int(coords);
	FREE_int(system);
	if (f_v) {
		cout << "projective_space::determine_conic_in_plane done" << endl;
	}
	return TRUE;
}


int projective_space::determine_cubic_in_plane(
		homogeneous_polynomial_domain *Poly_3_3,
		int nb_pts, long int *Pts, int *coeff10,
		int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int i, j, r, d;
	int *Pt_coord;
	int *System;
	int *base_cols;

	if (f_v) {
		cout << "projective_space::determine_cubic_in_plane" << endl;
	}
	d = n + 1;
	Pt_coord = NEW_int(nb_pts * d);
	System = NEW_int(nb_pts * Poly_3_3->get_nb_monomials());
	base_cols = NEW_int(Poly_3_3->get_nb_monomials());

	if (f_v) {
		cout << "projective_space::determine_cubic_in_plane list of "
				"points:" << endl;
		Orbiter->Lint_vec.print(cout, Pts, nb_pts);
		cout << endl;
	}
	for (i = 0; i < nb_pts; i++) {
		unrank_point(Pt_coord + i * d, Pts[i]);
	}
	if (f_v) {
		cout << "projective_space::determine_cubic_in_plane matrix of "
				"point coordinates:" << endl;
		Orbiter->Int_vec.matrix_print(Pt_coord, nb_pts, d);
	}

	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < Poly_3_3->get_nb_monomials(); j++) {
			System[i * Poly_3_3->get_nb_monomials() + j] =
					Poly_3_3->evaluate_monomial(j, Pt_coord + i * d);
		}
	}
	if (f_v) {
		cout << "projective_space::determine_cubic_in_plane "
				"The system:" << endl;
		Orbiter->Int_vec.matrix_print(System, nb_pts, Poly_3_3->get_nb_monomials());
	}
	r = F->Gauss_simple(System, nb_pts, Poly_3_3->get_nb_monomials(),
		base_cols, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space::determine_cubic_in_plane "
				"The system in RREF:" << endl;
		Orbiter->Int_vec.matrix_print(System, r, Poly_3_3->get_nb_monomials());
	}
	if (f_v) {
		cout << "projective_space::determine_cubic_in_plane "
				"The system has rank " << r << endl;
	}

	if (r != 9) {
		cout << "r != 9" << endl;
		exit(1);
	}
	int kernel_m, kernel_n;

	F->matrix_get_kernel(System, r, Poly_3_3->get_nb_monomials(),
		base_cols, r,
		kernel_m, kernel_n, coeff10, 0 /* verbose_level */);


	FREE_int(Pt_coord);
	FREE_int(System);
	FREE_int(base_cols);
	if (f_v) {
		cout << "projective_space::determine_cubic_in_plane done" << endl;
	}
	return r;
}


void projective_space::determine_quadric_in_solid(
	long int *nine_pts_or_more,
	int nb_pts, int *ten_coeffs, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *coords; // [nb_pts * 4]
	int *system; // [nb_pts * 10]
	int kernel[10 * 10];
	int base_cols[10];
	int i, x, y, z, w, rk;
	int kernel_m, kernel_n;

	if (f_v) {
		cout << "projective_space::determine_quadric_in_solid" << endl;
	}
	if (n != 3) {
		cout << "projective_space::determine_quadric_in_solid "
				"n != 3" << endl;
		exit(1);
	}
	if (nb_pts < 9) {
		cout << "projective_space::determine_quadric_in_solid "
				"you need to give at least 9 points" << endl;
		exit(1);
	}
	coords = NEW_int(nb_pts * 4);
	system = NEW_int(nb_pts * 10);
	for (i = 0; i < nb_pts; i++) {
		unrank_point(coords + i * 4, nine_pts_or_more[i]);
	}
	if (f_vv) {
		cout << "projective_space::determine_quadric_in_solid "
				"points:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				coords, nb_pts, 4, 4, F->log10_of_q);
	}
	for (i = 0; i < nb_pts; i++) {
		x = coords[i * 4 + 0];
		y = coords[i * 4 + 1];
		z = coords[i * 4 + 2];
		w = coords[i * 4 + 3];
		system[i * 10 + 0] = F->mult(x, x);
		system[i * 10 + 1] = F->mult(y, y);
		system[i * 10 + 2] = F->mult(z, z);
		system[i * 10 + 3] = F->mult(w, w);
		system[i * 10 + 4] = F->mult(x, y);
		system[i * 10 + 5] = F->mult(x, z);
		system[i * 10 + 6] = F->mult(x, w);
		system[i * 10 + 7] = F->mult(y, z);
		system[i * 10 + 8] = F->mult(y, w);
		system[i * 10 + 9] = F->mult(z, w);
	}
	if (f_v) {
		cout << "projective_space::determine_quadric_in_solid "
				"system:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				system, nb_pts, 10, 10, F->log10_of_q);
	}



	rk = F->Gauss_simple(system,
			nb_pts, 10, base_cols, verbose_level - 2);
	if (rk != 9) {
		cout << "projective_space::determine_quadric_in_solid "
				"system underdetermined" << endl;
		cout << "rk=" << rk << endl;
		exit(1);
	}
	F->matrix_get_kernel(system, 9, 10, base_cols, rk, 
		kernel_m, kernel_n, kernel, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space::determine_quadric_in_solid "
				"conic:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				kernel, 1, 10, 10, F->log10_of_q);
	}
	for (i = 0; i < 10; i++) {
		ten_coeffs[i] = kernel[i];
	}
	if (f_v) {
		cout << "projective_space::determine_quadric_in_solid done" << endl;
	}
}

void projective_space::conic_points_brute_force(
	int *six_coeffs,
	long int *points, int &nb_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[3];
	int i, a;

	if (f_v) {
		cout << "projective_space::conic_points_brute_force" << endl;
	}
	nb_points = 0;
	for (i = 0; i < N_points; i++) {
		unrank_point(v, i);
		a = F->evaluate_conic_form(six_coeffs, v);
		if (f_vv) {
			cout << "point " << i << " = ";
			Orbiter->Int_vec.print(cout, v, 3);
			cout << " gives a value of " << a << endl;
		}
		if (a == 0) {
			if (f_vv) {
				cout << "point " << i << " = ";
				Orbiter->Int_vec.print(cout, v, 3);
				cout << " lies on the conic" << endl;
			}
			points[nb_points++] = i;
		}
	}
	if (f_v) {
		cout << "projective_space::conic_points_brute_force done, "
				"we found " << nb_points << " points" << endl;
	}
	if (f_vv) {
		cout << "They are : ";
		Orbiter->Lint_vec.print(cout, points, nb_points);
		cout << endl;
	}
	if (f_v) {
		cout << "projective_space::conic_points_brute_force done" << endl;
	}
}

void projective_space::quadric_points_brute_force(
	int *ten_coeffs,
	long int *points, int &nb_points, int verbose_level)
// quadric in PG(3,q)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[3];
	long int i, a;

	if (f_v) {
		cout << "projective_space::quadric_points_brute_force" << endl;
	}
	nb_points = 0;
	for (i = 0; i < N_points; i++) {
		unrank_point(v, i);
		a = F->evaluate_quadric_form_in_PG_three(ten_coeffs, v);
		if (f_vv) {
			cout << "point " << i << " = ";
			Orbiter->Int_vec.print(cout, v, 3);
			cout << " gives a value of " << a << endl;
		}
		if (a == 0) {
			if (f_vv) {
				cout << "point " << i << " = ";
				Orbiter->Int_vec.print(cout, v, 4);
				cout << " lies on the quadric" << endl;
			}
			points[nb_points++] = i;
		}
	}
	if (f_v) {
		cout << "projective_space::quadric_points_brute_force done, "
				"we found " << nb_points << " points" << endl;
	}
	if (f_vv) {
		cout << "They are : ";
		Orbiter->Lint_vec.print(cout, points, nb_points);
		cout << endl;
	}
	if (f_v) {
		cout << "projective_space::quadric_points_brute_force done" << endl;
	}
}

void projective_space::conic_points(
	long int *five_pts, int *six_coeffs,
	long int *points, int &nb_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int Gram_matrix[9];
	int Basis[9];
	int Basis2[9];
	int v[3], w[3];
	int i, j, l, a = 0, av, ma, b, bv, t;

	
	if (f_v) {
		cout << "projective_space::conic_points" << endl;
	}
	if (n != 2) {
		cout << "projective_space::conic_points n != 2" << endl;
		exit(1);
	}
	Gram_matrix[0 * 3 + 0] = F->add(six_coeffs[0], six_coeffs[0]);
	Gram_matrix[1 * 3 + 1] = F->add(six_coeffs[1], six_coeffs[1]);
	Gram_matrix[2 * 3 + 2] = F->add(six_coeffs[2], six_coeffs[2]);
	Gram_matrix[0 * 3 + 1] = Gram_matrix[1 * 3 + 0] = six_coeffs[3];
	Gram_matrix[0 * 3 + 2] = Gram_matrix[2 * 3 + 0] = six_coeffs[4];
	Gram_matrix[1 * 3 + 2] = Gram_matrix[2 * 3 + 1] = six_coeffs[5];
	if (f_vv) {
		cout << "projective_space::conic_points Gram matrix:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Gram_matrix, 3, 3, 3, F->log10_of_q);
	}
	
	unrank_point(Basis, five_pts[0]);
	for (i = 1; i < 5; i++) {
		unrank_point(Basis + 3, five_pts[i]);
		a = F->evaluate_bilinear_form(3, Basis, Basis + 3, Gram_matrix);
		if (a) {
			break;
		}
	}
	if (i == 5) {
		cout << "projective_space::conic_points did not "
				"find non-orthogonal vector" << endl;
		exit(1);
	}
	if (a != 1) {
		av = F->inverse(a);
		for (i = 0; i < 3; i++) {
			Basis[3 + i] = F->mult(av, Basis[3 + i]);
		}
	}
	if (f_v) {	
		cout << "projective_space::conic_points "
				"Hyperbolic pair:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Basis, 2, 3, 3, F->log10_of_q);
	}
	F->perp(3, 2, Basis, Gram_matrix, 0 /* verbose_level */);
	if (f_v) {	
		cout << "projective_space::conic_points perp:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Basis, 3, 3, 3, F->log10_of_q);
	}
	a = F->evaluate_conic_form(six_coeffs, Basis + 6);
	if (f_v) {	
		cout << "projective_space::conic_points "
				"form value = " << a << endl;
	}
	if (a == 0) {
		cout << "projective_space::conic_points "
				"the form is degenerate or we are in "
				"characteristic zero" << endl;
		exit(1);
	}
	l = F->log_alpha(a);
	if ((l % 2) == 0) {
		j = l / 2;
		b = F->alpha_power(j);
		bv = F->inverse(b);
		for (i = 0; i < 3; i++) {
			Basis[6 + i] = F->mult(bv, Basis[6 + i]);
		}
		a = F->evaluate_conic_form(six_coeffs, Basis + 6);
		if (f_v) {	
			cout << "form value = " << a << endl;
		}
	}
	for (i = 0; i < 3; i++) {
		Basis2[3 + i] = Basis[6 + i];
	}
	for (i = 0; i < 3; i++) {
		Basis2[0 + i] = Basis[0 + i];
	}
	for (i = 0; i < 3; i++) {
		Basis2[6 + i] = Basis[3 + i];
	}
	if (f_v) {	
		cout << "Basis2:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Basis2, 3, 3, 3, F->log10_of_q);
	}
	// Now the form is a^{-1}y_1^2 = y_0y_2 
	// (or, equivalently, a^{-1}y_1^2 - y_0y_2 = 0)
	// and  the quadratic form on (0,1,0) in y-coordinates is a.
	// 
	// In the y-coordinates, the points on this conic are 
	// (1,0,0) and (t^2,t,-a) for t \in GF(q).
	// In the x-coordinates, the points are 
	// (1,0,0) * Basis2 and (t^2,t,-a) * Basis2 for t \in GF(q).

	v[0] = 1;
	v[1] = 0;
	v[2] = 0;

	F->mult_vector_from_the_left(v, Basis2, w, 3, 3);
	if (f_v) {	
		cout << "vector corresponding to 100:" << endl;
		Orbiter->Int_vec.print(cout, w, 3);
		cout << endl;
	}
	b = rank_point(w);
	points[0] = b;
	nb_points = 1;

	ma = F->negate(a);
	
	for (t = 0; t < F->q; t++) {
		v[0] = F->mult(t, t);
		v[1] = t;
		v[2] = ma;
		F->mult_vector_from_the_left(v, Basis2, w, 3, 3);
		if (f_v) {	
			cout << "vector corresponding to t=" << t << ":" << endl;
			Orbiter->Int_vec.print(cout, w, 3);
			cout << endl;
		}
		b = rank_point(w);
		points[nb_points++] = b;
	}
	if (f_vv) {	
		cout << "projective_space::conic_points conic points:" << endl;
		Orbiter->Lint_vec.print(cout, points, nb_points);
		cout << endl;
	}
	if (f_v) {
		cout << "projective_space::conic_points done" << endl;
	}
}

void projective_space::find_tangent_lines_to_conic(
	int *six_coeffs,
	long int *points, int nb_points,
	long int *tangents, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int v[3];
	int Basis[9];
	int Gram_matrix[9];
	int i;
	
	if (f_v) {
		cout << "projective_space::find_tangent_lines_to_conic" << endl;
	}
	if (n != 2) {
		cout << "projective_space::find_tangent_lines_to_conic "
				"n != 2" << endl;
		exit(1);
	}
	Gram_matrix[0 * 3 + 0] = F->add(six_coeffs[0], six_coeffs[0]);
	Gram_matrix[1 * 3 + 1] = F->add(six_coeffs[1], six_coeffs[1]);
	Gram_matrix[2 * 3 + 2] = F->add(six_coeffs[2], six_coeffs[2]);
	Gram_matrix[0 * 3 + 1] = Gram_matrix[1 * 3 + 0] = six_coeffs[3];
	Gram_matrix[0 * 3 + 2] = Gram_matrix[2 * 3 + 0] = six_coeffs[4];
	Gram_matrix[1 * 3 + 2] = Gram_matrix[2 * 3 + 1] = six_coeffs[5];
	
	for (i = 0; i < nb_points; i++) {
		unrank_point(Basis, points[i]);
		F->perp(3, 1, Basis, Gram_matrix, 0 /* verbose_level */);
		if (f_vv) {	
			cout << "perp:" << endl;
			Orbiter->Int_vec.print_integer_matrix_width(cout,
					Basis, 3, 3, 3, F->log10_of_q);
		}
		tangents[i] = rank_line(Basis + 3);
		if (f_vv) {	
			cout << "tangent at point " << i << " is "
					<< tangents[i] << endl;
		}
	}
	if (f_v) {
		cout << "projective_space::find_tangent_lines_to_conic done" << endl;
	}
}

void projective_space::compute_bisecants_and_conics(
	long int *arc6,
	int *&bisecants, int *&conics, int verbose_level)
// bisecants[15 * 3]
// conics[6 * 6]
{
	int f_v = (verbose_level >= 1);
	long int i, j, h, pi, pj, Line[2];
	long int arc5[5];
	int six_coeffs[6];

	if (f_v) {
		cout << "projective_space::compute_bisecants_and_conics" << endl;
	}
	bisecants = NEW_int(15 * 3);
	conics = NEW_int(6 * 6);
	
	h = 0;
	for (i = 0; i < 6; i++) {
		pi = arc6[i];
		for (j = i + 1; j < 6; j++, h++) {
			pj = arc6[j];
			Line[0] = pi;
			Line[1] = pj;
			determine_line_in_plane(Line, 
				bisecants + h * 3, 
				0 /* verbose_level */);
			F->PG_element_normalize_from_front(
				bisecants + h * 3, 1, 3);
		}
	}
	if (f_v) {
		cout << "projective_space::compute_bisecants_and_conics "
				"bisecants:" << endl;
		Orbiter->Int_vec.matrix_print(bisecants, 15, 3);
	}

	for (j = 0; j < 6; j++) {
		//int deleted_point;
		
		//deleted_point = arc6[j];
		Orbiter->Lint_vec.copy(arc6, arc5, j);
		Orbiter->Lint_vec.copy(arc6 + j + 1, arc5 + j, 5 - j);

#if 0
		cout << "deleting point " << j << " / 6:";
		int_vec_print(cout, arc5, 5);
		cout << endl;
#endif

		determine_conic_in_plane(arc5, 5,
				six_coeffs, 0 /* verbose_level */);
		F->PG_element_normalize_from_front(six_coeffs, 1, 6);
		Orbiter->Int_vec.copy(six_coeffs, conics + j * 6, 6);
	}

	if (f_v) {
		cout << "projective_space::compute_bisecants_and_conics "
				"conics:" << endl;
		Orbiter->Int_vec.matrix_print(conics, 6, 6);
	}

	if (f_v) {
		cout << "projective_space::compute_bisecants_and_conics "
				"done" << endl;
	}
}

eckardt_point_info *projective_space::compute_eckardt_point_info(
	surface_domain *Surf, long int *arc6,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	eckardt_point_info *E;
	
	if (f_v) {
		cout << "projective_space::compute_eckardt_point_info" << endl;
	}
	if (n != 2) {
		cout << "projective_space::compute_eckardt_point_info "
				"n != 2" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "arc: ";
		Orbiter->Lint_vec.print(cout, arc6, 6);
		cout << endl;
	}

	E = NEW_OBJECT(eckardt_point_info);
	E->init(Surf, this, arc6, verbose_level);

	if (f_v) {
		cout << "projective_space::compute_eckardt_point_info done" << endl;
	}
	return E;
}

void projective_space::PG_2_8_create_conic_plus_nucleus_arc_1(
		long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int frame_data[] = {1,0,0, 0,1,0,  0,0,1,  1,1,1 };
	int frame[4];
	int i, j, b, h, idx;
	int L[3];
	int v[3];
	sorting Sorting;

	if (n != 2) {
		cout << "projective_space::PG_2_8_create_conic_plus_nucleus_arc_1 "
				"n != 2" << endl;
		exit(1);
	}
	if (q != 8) {
		cout << "projective_space::PG_2_8_create_conic_plus_nucleus_arc_1 "
				"q != 8" << endl;
		exit(1);
	}
	for (i = 0; i < 4; i++) {
		frame[i] = rank_point(frame_data + i * 3);
	}

	if (f_v) {
		cout << "frame: ";
		Orbiter->Int_vec.print(cout, frame, 4);
		cout << endl;
	}
	
	L[0] = Implementation->Line_through_two_points[frame[0] * N_points + frame[1]];
	L[1] = Implementation->Line_through_two_points[frame[1] * N_points + frame[2]];
	L[2] = Implementation->Line_through_two_points[frame[2] * N_points + frame[0]];
	
	if (f_v) {
		cout << "l1=" << L[0] << " l2=" << L[1] << " l3=" << L[2] << endl;
	}

	size = 0;	
	for (h = 0; h < 3; h++) {
		for (i = 0; i < r; i++) {
			b = Implementation->Lines[L[h] * r + i];
			if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
				continue;
			}
			for (j = size; j > idx; j--) {
				the_arc[j] = the_arc[j - 1];
			}
			the_arc[idx] = b;
			size++;
		}
	}
	if (f_v) {
		cout << "there are " << size << " points on the three lines: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}


	for (i = 1; i < q; i++) {
		v[0] = 1;
		v[1] = i;
		v[2] = F->mult(i, i);
		b = rank_point(v);
		if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
			continue;
		}
		for (j = size; j > idx; j--) {
			the_arc[j] = the_arc[j - 1];
		}
		the_arc[idx] = b;
		size++;
		
	}

	if (f_v) {
		cout << "projective_space::PG_2_8_create_conic_plus_nucleus_arc_1: after adding the rest of the "
				"conic, there are " << size << " points on the arc: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}
}

void projective_space::PG_2_8_create_conic_plus_nucleus_arc_2(
		long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int frame_data[] = {1,0,0, 0,1,0,  0,0,1,  1,1,1 };
	int frame[4];
	int i, j, b, h, idx;
	int L[3];
	int v[3];
	sorting Sorting;

	if (n != 2) {
		cout << "projective_space::PG_2_8_create_conic_plus_"
				"nucleus_arc_2 n != 2" << endl;
		exit(1);
	}
	if (q != 8) {
		cout << "projective_space::PG_2_8_create_conic_plus_"
				"nucleus_arc_2 q != 8" << endl;
		exit(1);
	}
	for (i = 0; i < 4; i++) {
		frame[i] = rank_point(frame_data + i * 3);
	}

	if (f_v) {
		cout << "frame: ";
		Orbiter->Int_vec.print(cout, frame, 4);
		cout << endl;
	}
	
	L[0] = Implementation->Line_through_two_points[frame[0] * N_points + frame[2]];
	L[1] = Implementation->Line_through_two_points[frame[2] * N_points + frame[3]];
	L[2] = Implementation->Line_through_two_points[frame[3] * N_points + frame[0]];
	
	if (f_v) {
		cout << "l1=" << L[0] << " l2=" << L[1] << " l3=" << L[2] << endl;
	}

	size = 0;	
	for (h = 0; h < 3; h++) {
		for (i = 0; i < r; i++) {
			b = Implementation->Lines[L[h] * r + i];
			if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
				continue;
			}
			for (j = size; j > idx; j--) {
				the_arc[j] = the_arc[j - 1];
			}
			the_arc[idx] = b;
			size++;
		}
	}
	if (f_v) {
		cout << "there are " << size << " points on the three lines: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}


	for (i = 0; i < q; i++) {
		if (i == 1) {
			v[0] = 0;
			v[1] = 1;
			v[2] = 0;
		}
		else {
			v[0] = 1;
			v[1] = i;
			v[2] = F->mult(i, i);
		}
		b = rank_point(v);
		if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
			continue;
		}
		for (j = size; j > idx; j--) {
			the_arc[j] = the_arc[j - 1];
		}
		the_arc[idx] = b;
		size++;
		
	}

	if (f_v) {
		cout << "projective_space::PG_2_8_create_conic_plus_"
				"nucleus_arc_2: after adding the rest of the conic, "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}
}

void projective_space::create_Maruta_Hamada_arc(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int data[] = { 
		0,1,2, 0,1,3, 0,1,4, 0,1,5,
		1,0,7, 1,0,8, 1,0,9, 1,0,10, 
		1,2,0, 1,3,0, 1,4,0, 1,5,0,
		1,7,5, 1,8,4, 1,9,3, 1,10,2, 
		1,1,1, 1,1,10, 1,10,1,  1,4,4,
		1,12,0, 1,0,12 
		 };
	int points[22];
	int i, j, b, h, idx;
	long int L[4];
	int v[3];
	sorting Sorting;

	if (n != 2) {
		cout << "projective_space::create_Maruta_Hamada_arc "
				"n != 2" << endl;
		exit(1);
	}
	if (q != 13) {
		cout << "projective_space::create_Maruta_Hamada_arc "
				"q != 13" << endl;
		exit(1);
	}
	for (i = 0; i < 22; i++) {
		points[i] = rank_point(data + i * 3);
		cout << "point " << i << " has rank " << points[i] << endl;
	}

	if (f_v) {
		cout << "projective_space::create_Maruta_Hamada_arc "
				"points: ";
		Orbiter->Int_vec.print(cout, points, 22);
		cout << endl;
	}
	
	L[0] = Implementation->Line_through_two_points[1 * N_points + 2];
	L[1] = Implementation->Line_through_two_points[0 * N_points + 2];
	L[2] = Implementation->Line_through_two_points[0 * N_points + 1];
	L[3] = Implementation->Line_through_two_points[points[20] * N_points + points[21]];
	
	if (f_v) {
		cout << "L:";
		Orbiter->Lint_vec.print(cout, L, 4);
		cout << endl;
	}

	if (f_v) {
		for (h = 0; h < 4; h++) {
			cout << "h=" << h << " : L[h]=" << L[h] << " : " << endl;
			for (i = 0; i < r; i++) {
				b = Implementation->Lines[L[h] * r + i];
					cout << "point " << b << " = ";
				unrank_point(v, b);
				F->PG_element_normalize_from_front(v, 1, 3);
				Orbiter->Int_vec.print(cout, v, 3);
				cout << endl;
			}
			cout << endl;
		}
	}
	size = 0;	
	for (h = 0; h < 4; h++) {
		for (i = 0; i < r; i++) {
			b = Implementation->Lines[L[h] * r + i];
			if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
				continue;
			}
			for (j = size; j > idx; j--) {
				the_arc[j] = the_arc[j - 1];
			}
			the_arc[idx] = b;
			size++;
		}
	}
	if (f_v) {
		cout << "there are " << size
				<< " points on the quadrilateral: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}


	// remove the first 16 points:
	for (i = 0; i < 16; i++) {
		if (f_v) {
			cout << "removing point " << i << " : "
				<< points[i] << endl;
		}
		if (!Sorting.lint_vec_search(the_arc, size, points[i], idx, 0)) {
			cout << "error, cannot find point to be removed" << endl;
			exit(1);
		}
		for (j = idx; j < size; j++) {
			the_arc[j] = the_arc[j + 1];
		}
		size--;
	}

	// add points 16-19:
	for (i = 16; i < 20; i++) {
		if (Sorting.lint_vec_search(the_arc, size, points[i], idx, 0)) {
			cout << "error, special point already there" << endl;
			exit(1);
		}
		for (j = size; j > idx; j--) {
			the_arc[j] = the_arc[j - 1];
		}
		the_arc[idx] = points[i];
		size++;
	}
		
	if (f_v) {
		cout << "projective_space::create_Maruta_Hamada_arc: "
				"after adding the special point, there are "
				<< size << " points on the arc: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}

}

void projective_space::create_Maruta_Hamada_arc2(
		long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int data[] = { 
		1,6,2, 1,11,4, 1,5,5, 1,2,6, 1,10,7, 1,12,8, 1,7,10, 1,4,11, 1,8,12, 
		0,1,10, 0,1,12, 0,1,4, 1,0,1, 1,0,3, 1,0,9, 1,1,0, 1,3,0, 1,9,0,
		1,4,4, 1,4,12, 1,12,4, 1,10,10, 1,10,12, 1,12,10 
		 };
	int points[24];
	int i, j, a;
	int L[9];

	if (n != 2) {
		cout << "projective_space::create_Maruta_Hamada_arc2 "
				"n != 2" << endl;
		exit(1);
	}
	if (q != 13) {
		cout << "projective_space::create_Maruta_Hamada_arc2 "
				"q != 13" << endl;
		exit(1);
	}
	for (i = 0; i < 24; i++) {
		points[i] = rank_point(data + i * 3);
		cout << "point " << i << " has rank " << points[i] << endl;
	}

	if (f_v) {
		cout << "projective_space::create_Maruta_Hamada_arc2 "
				"points: ";
		Orbiter->Int_vec.print(cout, points, 25);
		cout << endl;
	}
	for (i = 0; i < 9; i++) {
		L[i] = Standard_polarity->Point_to_hyperplane[points[i]];
	}
	size = 0;
	for (i = 0; i < 9; i++) {
		for (j = i + 1; j < 9; j++) {
			a = intersection_of_two_lines(L[i], L[j]);
			the_arc[size++] = a;
		}
	}
	for (i = 9; i < 24; i++) {
		the_arc[size++] = points[i];
	}
	if (f_v) {
		cout << "projective_space::create_Maruta_Hamada_arc2: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}
}


void projective_space::create_pasch_arc(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int data[] = {1,1,1, 1,0,0, 0,1,1,  0,1,0,  1,0,1 };
	int points[5];
	int i, j, b, h, idx;
	int L[4];
	sorting Sorting;

	if (n != 2) {
		cout << "projective_space::create_pasch_arc "
				"n != 2" << endl;
		exit(1);
	}
#if 0
	if (q != 8) {
		cout << "projective_space::create_pasch_arc "
				"q != 8" << endl;
		exit(1);
	}
#endif
	for (i = 0; i < 5; i++) {
		points[i] = rank_point(data + i * 3);
	}

	if (f_v) {
		cout << "projective_space::create_pasch_arc() points: ";
		Orbiter->Int_vec.print(cout, points, 5);
		cout << endl;
	}
	
	L[0] = Implementation->Line_through_two_points[points[0] * N_points + points[1]];
	L[1] = Implementation->Line_through_two_points[points[0] * N_points + points[3]];
	L[2] = Implementation->Line_through_two_points[points[2] * N_points + points[3]];
	L[3] = Implementation->Line_through_two_points[points[1] * N_points + points[4]];
	
	if (f_v) {
		cout << "L:";
		Orbiter->Int_vec.print(cout, L, 4);
		cout << endl;
	}

	size = 0;	
	for (h = 0; h < 4; h++) {
		for (i = 0; i < r; i++) {
			b = Implementation->Lines[L[h] * r + i];
			if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
				continue;
			}
			for (j = size; j > idx; j--) {
				the_arc[j] = the_arc[j - 1];
			}
			the_arc[idx] = b;
			size++;
		}
	}
	if (f_v) {
		cout << "there are " << size << " points on the pasch lines: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}

	
	v[0] = 1;
	v[1] = 1;
	v[2] = 0;
	b = rank_point(v);
	if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
		cout << "error, special point already there" << endl;
		exit(1);
	}
	for (j = size; j > idx; j--) {
		the_arc[j] = the_arc[j - 1];
	}
	the_arc[idx] = b;
	size++;
		
	if (f_v) {
		cout << "projective_space::create_pasch_arc: after "
				"adding the special point, there are "
				<< size << " points on the arc: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}
}

void projective_space::create_Cheon_arc(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int data[] = {1,0,0, 0,1,0, 0,0,1 };
	int points[3];
	int i, j, a, b, c, h, idx, t;
	int L[3];
	int pencil[9];
	int Pencil[21];
	sorting Sorting;

	if (n != 2) {
		cout << "projective_space::create_Cheon_arc n != 2" << endl;
		exit(1);
	}
#if 0
	if (q != 8) {
		cout << "projective_space::create_Cheon_arc q != 8" << endl;
		exit(1);
	}
#endif
	for (i = 0; i < 9; i++) {
		pencil[i] = 0;
	}
	for (i = 0; i < 21; i++) {
		Pencil[i] = 0;
	}
	for (i = 0; i < 3; i++) {
		points[i] = rank_point(data + i * 3);
	}

	if (f_v) {
		cout << "points: ";
		Orbiter->Int_vec.print(cout, points, 5);
		cout << endl;
	}
	
	L[0] = Implementation->Line_through_two_points[points[0] * N_points + points[1]];
	L[1] = Implementation->Line_through_two_points[points[1] * N_points + points[2]];
	L[2] = Implementation->Line_through_two_points[points[2] * N_points + points[0]];
	
	if (f_v) {
		cout << "L:";
		Orbiter->Int_vec.print(cout, L, 3);
		cout << endl;
	}

	size = 0;	
	for (h = 0; h < 3; h++) {
		for (i = 0; i < r; i++) {
			b = Implementation->Lines[L[h] * r + i];
			if (Sorting.lint_vec_search(the_arc, size, b, idx, 0)) {
				continue;
			}
			for (j = size; j > idx; j--) {
				the_arc[j] = the_arc[j - 1];
			}
			the_arc[idx] = b;
			size++;
		}
	}

	if (f_v) {
		cout << "projective_space::create_Cheon_arc there are "
				<< size << " points on the 3 lines: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}




	for (h = 0; h < 3; h++) {

		if (f_v) {
			cout << "h=" << h << endl;
		}

		for (i = 0; i < r; i++) {
			pencil[i] = Implementation->Lines_on_point[points[h] * r + i];
		}


		j = 0;
		for (i = 0; i < r; i++) {
			b = pencil[i];
			if (b == L[0] || b == L[1] || b == L[2])
				continue;
			Pencil[h * 7 + j] = b;
			j++;
		}
		if (j != 7) {
			cout << "j=" << j << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "Pencil:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, Pencil, 3, 7, 7, 4);
	}

	for (i = 0; i < 7; i++) {
		a = Pencil[0 * 7 + i];
		for (j = 0; j < 7; j++) {
			b = Pencil[1 * 7 + j];
			if (f_v) {
				cout << "i=" << i << " a=" << a << " j="
						<< j << " b=" << b << endl;
			}
			c = Implementation->Line_intersection[a * N_lines + b];
			if (f_v) {
				cout << "c=" << c << endl;
			}
			if (Sorting.lint_vec_search(the_arc, size, c, idx, 0)) {
				continue;
			}
			for (t = size; t > idx; t--) {
				the_arc[t] = the_arc[t - 1];
			}
			the_arc[idx] = c;
			size++;
#if 0
			if (size > 31) {
				cout << "create_Cheon_arc size=" << size << endl;
			}
#endif
		}
	}
	if (f_v) {
		cout << "there are " << size << " points on the Cheon lines: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}

	
}


void projective_space::create_regular_hyperoval(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];

	if (n != 2) {
		cout << "projective_space::create_regular_hyperoval "
				"n != 2" << endl;
		exit(1);
	}

	for (i = 0; i < q; i++) {
		v[0] = F->mult(i, i);
		v[1] = i;
		v[2] = 1;
		the_arc[i] = rank_point(v);		
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[q] = rank_point(v);
	
	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[q + 1] = rank_point(v);
	
	size = q + 2;

	if (f_v) {
		cout << "projective_space::create_regular_hyperoval: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}
}

void projective_space::create_translation_hyperoval(
	long int *the_arc, int &size,
	int exponent, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];

	if (f_v) {
		cout << "projective_space::create_translation_hyperoval" << endl;
		cout << "exponent = " << exponent << endl;
	}
	if (n != 2) {
		cout << "projective_space::create_translation_hyperoval "
				"n != 2" << endl;
		exit(1);
	}

	for (i = 0; i < q; i++) {
		v[0] = F->frobenius_power(i, exponent);
		v[1] = i;
		v[2] = 1;
		the_arc[i] = rank_point(v);		
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[q] = rank_point(v);
	
	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[q + 1] = rank_point(v);
	
	size = q + 2;

	if (f_v) {
		cout << "projective_space::create_translation_hyperoval: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}
	if (f_v) {
		cout << "projective_space::create_translation_hyperoval "
				"done" << endl;
	}
}

void projective_space::create_Segre_hyperoval(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];

	if (n != 2) {
		cout << "projective_space::create_Segre_hyperoval "
				"n != 2" << endl;
		exit(1);
	}

	for (i = 0; i < q; i++) {
		v[0] = F->power(i, 6);
		v[1] = i;
		v[2] = 1;
		the_arc[i] = rank_point(v);		
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[q] = rank_point(v);
	
	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[q + 1] = rank_point(v);
	
	size = q + 2;

	if (f_v) {
		cout << "projective_space::create_Segre_hyperoval: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}
}

void projective_space::create_Payne_hyperoval(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];
	longinteger_domain D;
	longinteger_object a, b, u, u2, g;
	int exponent;
	int one_sixth, one_half, five_sixth;

	if (f_v) {
		cout << "projective_space::create_Payne_hyperoval" << endl;
	}
	if (n != 2) {
		cout << "projective_space::create_Payne_hyperoval n != 2" << endl;
		exit(1);
	}
	exponent = q - 1;
	a.create(6, __FILE__, __LINE__);
	b.create(exponent, __FILE__, __LINE__);
	
	D.extended_gcd(a, b, g, u, u2, 0 /* verbose_level */);
	one_sixth = u.as_int();
	while (one_sixth < 0) {
		one_sixth += exponent;
	}
	if (f_v) {
		cout << "one_sixth = " << one_sixth << endl;
	}

	a.create(2, __FILE__, __LINE__);
	D.extended_gcd(a, b, g, u, u2, 0 /* verbose_level */);
	one_half = u.as_int();
	while (one_half < 0) {
		one_half += exponent;
	}
	if (f_v) {
		cout << "one_half = " << one_half << endl;
	}

	five_sixth = (5 * one_sixth) % exponent;
	if (f_v) {
		cout << "five_sixth = " << five_sixth << endl;
	}

	for (i = 0; i < q; i++) {
		v[0] = F->add3(
			F->power(i, one_sixth), 
			F->power(i, one_half), 
			F->power(i, five_sixth));
		v[1] = i;
		v[2] = 1;
		the_arc[i] = rank_point(v);		
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[q] = rank_point(v);
	
	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[q + 1] = rank_point(v);
	
	size = q + 2;

	if (f_v) {
		cout << "projective_space::create_Payne_hyperoval: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}
}

void projective_space::create_Cherowitzo_hyperoval(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];
	int h;
	int sigma;
	int exponent, one_half, e1, e2, e3;
	number_theory_domain NT;

	if (f_v) {
		cout << "projective_space::create_Cherowitzo_hyperoval" << endl;
	}
	if (n != 2) {
		cout << "projective_space::create_Cherowitzo_hyperoval "
				"n != 2" << endl;
		exit(1);
	}
	h = F->e;
	if (EVEN(h)) {
		cout << "projective_space::create_Cherowitzo_hyperoval "
				"field degree must be odd" << endl;
		exit(1);
	}
	if (F->p != 2) {
		cout << "projective_space::create_Cherowitzo_hyperoval "
				"needs characteristic 2" << endl;
		exit(1);
	}
	exponent = q - 1;
	one_half = (h + 1) >> 1;
	sigma = NT.i_power_j(2, one_half);
	e1 = sigma;
	e2 = (sigma + 2) % exponent;
	e3 = (3 * sigma + 4) % exponent;

	for (i = 0; i < q; i++) {
		v[0] = F->add3(
			F->power(i, e1), 
			F->power(i, e2), 
			F->power(i, e3));
		v[1] = i;
		v[2] = 1;
		the_arc[i] = rank_point(v);		
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[q] = rank_point(v);
	
	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[q + 1] = rank_point(v);
	
	size = q + 2;

	if (f_v) {
		cout << "projective_space::create_Cherowitzo_hyperoval: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}
}

void projective_space::create_OKeefe_Penttila_hyperoval_32(
	long int *the_arc, int &size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int v[3];

	if (f_v) {
		cout << "projective_space::create_OKeefe_Penttila_hyperoval_32"
				<< endl;
	}
	if (n != 2) {
		cout << "projective_space::create_OKeefe_Penttila_hyperoval_32 "
				"n != 2" << endl;
		exit(1);
	}
	if (F->q != 32) {
		cout << "projective_space::create_OKeefe_Penttila_hyperoval_32 "
				"needs q=32" << endl;
		exit(1);
	}

	for (i = 0; i < q; i++) {
		v[0] = F->OKeefe_Penttila_32(i);
		v[1] = i;
		v[2] = 1;
		the_arc[i] = rank_point(v);		
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	the_arc[q] = rank_point(v);
	
	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	the_arc[q + 1] = rank_point(v);
	
	size = q + 2;

	if (f_v) {
		cout << "projective_space::create_OKeefe_Penttila_hyperoval_32: "
				"there are " << size << " points on the arc: ";
		Orbiter->Lint_vec.print(cout, the_arc, size);
		cout << endl;
	}
}




void projective_space::line_intersection_type(
	long int *set, int set_size, int *type, int verbose_level)
// type[N_lines]
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b;

	if (f_v) {
		cout << "projective_space::line_intersection_type" << endl;
	}
	if (Implementation->Lines_on_point == NULL) {
		line_intersection_type_basic(set, set_size, type, verbose_level);
	}
	else {
		for (i = 0; i < N_lines; i++) {
			type[i] = 0;
		}
		for (i = 0; i < set_size; i++) {
			a = set[i];
			for (j = 0; j < r; j++) {
				b = Implementation->Lines_on_point[a * r + j];
				type[b]++;
			}
		}
	}
}

void projective_space::line_intersection_type_basic(
	long int *set, int set_size, int *type, int verbose_level)
// type[N_lines]
{
	int f_v = (verbose_level >= 1);
	long int rk, h, i, j, d;
	int *M;

	if (f_v) {
		cout << "projective_space::line_intersection_type_basic" << endl;
	}
	d = n + 1;
	M = NEW_int(3 * d);
	for (rk = 0; rk < N_lines; rk++) {
		type[rk] = 0;
		Grass_lines->unrank_lint(rk, 0 /* verbose_level */);
		for (h = 0; h < set_size; h++) {
			for (i = 0; i < 2; i++) {
				for (j = 0; j < d; j++) {
					M[i * d + j] = Grass_lines->M[i * d + j];
				}
			}
			unrank_point(M + 2 * d, set[h]);
			if (F->rank_of_rectangular_matrix(M,
					3, d, 0 /*verbose_level*/) == 2) {
				type[rk]++;
			}
		} // next h
	} // next rk
	FREE_int(M);
}

void projective_space::line_intersection_type_through_hyperplane(
	long int *set, int set_size, int *type, int verbose_level)
// type[N_lines]
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;
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
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::line_intersection_type_through_hyperplane "
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
			Orbiter->Int_vec.print(cout, M, d);
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
		cout << "projective_space::line_intersection_type_through_hyperplane "
				"sz1=" << sz1 << " sz2=" << sz2 << endl;
	}
	

	// do the line type in the hyperplane:
	line_intersection_type_basic(set1, sz1, type, verbose_level);
	nb_pts_in_hyperplane = Gg.nb_PG_elements(n - 1, q);
	if (f_vv) {
		cout << "projective_space::line_intersection_type_through_hyperplane "
				"nb_pts_in_hyperplane="
				<< nb_pts_in_hyperplane << endl;
	}

	cnt1 = NEW_int(nb_pts_in_hyperplane);
	Pts1 = NEW_int(nb_pts_in_hyperplane * d);
	Pts2 = NEW_int(sz2 * d);
	
	Orbiter->Int_vec.zero(cnt1, nb_pts_in_hyperplane);
	for (i = 0; i < nb_pts_in_hyperplane; i++) {
		F->PG_element_unrank_modified(Pts1 + i * d, 1, n, i);
		Pts1[i * d + d - 1] = 0;
		F->PG_element_rank_modified_lint(Pts1 + i * d, 1, n + 1, i1);

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
			cout << "projective_space::line_intersection_type_through_hyperplane "
					"checking lines through point " << i
					<< " / " << nb_pts_in_hyperplane << ":" << endl;
		}
		Orbiter->Int_vec.zero(f_taken, sz2);
		for (j = 0; j < sz2; j++) {
			if (f_taken[j]) {
				continue;
			}
			if (f_vv) {
				cout << "projective_space::line_intersection_type_through_hyperplane "
						"j=" << j << " / " << sz2 << ":" << endl;
			}
			Orbiter->Int_vec.copy(Pts1 + i * d, M, d);
			Orbiter->Int_vec.copy(Pts2 + j * d, M + d, d);
			f_taken[j] = TRUE;
			if (f_vv) {
				Orbiter->Int_vec.matrix_print(M, 2, d);
			}
			rk = Grass_lines->rank_lint_here(M, 0 /* verbose_level */);
			if (f_vv) {
				cout << "projective_space::line_intersection_type_through_hyperplane "
						"line rk=" << rk << " cnt1="
						<< cnt1[rk] << ":" << endl;
			}
			cnt = 1 + cnt1[i];
			for (h = j + 1; h < sz2; h++) {
				Orbiter->Int_vec.copy(M, M2, 2 * d);
				Orbiter->Int_vec.copy(Pts2 + h * d, M2 + 2 * d, d);
				if (F->rank_of_rectangular_matrix(M2,
						3, d, 0 /*verbose_level*/) == 2) {
					cnt++;
					f_taken[h] = TRUE;
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
		cout << "projective_space::line_intersection_type_through_hyperplane "
				"done" << endl;
	}
}

void projective_space::find_secant_lines(
		long int *set, int set_size,
		long int *lines, int &nb_lines, int max_lines,
		int verbose_level)
// finds the secant lines as an ordered set (secant variety).
// this is done by looping over all pairs of points and creating the
// line that is spanned by the two points.
{
	int f_v = (verbose_level >= 1);
	int i, j, rk, d, h, idx;
	int *M;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::find_secant_lines "
				"set_size=" << set_size << endl;
	}
	d = n + 1;
	M = NEW_int(2 * d);
	nb_lines = 0;
	for (i = 0; i < set_size; i++) {
		for (j = i + 1; j < set_size; j++) {
			unrank_point(M, set[i]);
			unrank_point(M + d, set[j]);
			rk = Grass_lines->rank_lint_here(M, 0 /* verbose_level */);

			if (!Sorting.lint_vec_search(lines, nb_lines, rk, idx, 0)) {
				if (nb_lines == max_lines) {
					cout << "projective_space::find_secant_lines "
							"nb_lines == max_lines" << endl;
					exit(1);
				}
				for (h = nb_lines; h > idx; h--) {
					lines[h] = lines[h - 1];
				}
				lines[idx] = rk;
				nb_lines++;
			}
		}
	}
	FREE_int(M);
	if (f_v) {
		cout << "projective_space::find_secant_lines done" << endl;
	}
}

void projective_space::find_lines_which_are_contained(
		std::vector<long int> &Points,
		std::vector<long int> &Lines,
		int verbose_level)
// finds all lines which are completely contained in the set of points
// First, finds all lines in the set which lie
// in the hyperplane x_d = 0.
// Then finds all remaining lines.
// The lines are not arranged according to a double six.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = FALSE;
	long int rk;
	long int h, i, j, d, a, b;
	int idx;
	int *M;
	int *M2;
	int *Pts1;
	int *Pts2;
	long int *set1;
	long int *set2;
	int sz1, sz2;
	int *f_taken;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::find_lines_which_are_contained "
				"set_size=" << Points.size() << endl;
	}
	//nb_lines = 0;
	d = n + 1;
	M = NEW_int(3 * d);
	M2 = NEW_int(3 * d);
	set1 = NEW_lint(Points.size());
	set2 = NEW_lint(Points.size());
	sz1 = 0;
	sz2 = 0;
	for (i = 0; i < Points.size(); i++) {
		unrank_point(M, Points[i]);
		if (f_vvv) {
			cout << Points[i] << " : ";
			Orbiter->Int_vec.print(cout, M, d);
			cout << endl;
		}
		if (M[d - 1] == 0) {
			set1[sz1++] = Points[i];
		}
		else {
			set2[sz2++] = Points[i];
		}
	}

	// set1 is the set of points whose last coordinate is zero.
	// set2 is the set of points whose last coordinate is nonzero.
	Sorting.lint_vec_heapsort(set1, sz1);
	Sorting.lint_vec_heapsort(set2, sz2);
	
	if (f_vv) {
		cout << "projective_space::find_lines_which_are_contained "
				"sz1=" << sz1 << " sz2=" << sz2 << endl;
	}
	

	// find all secants in the hyperplane:
	long int *secants;
	int n2, nb_secants;

	n2 = (sz1 * (sz1 - 1)) >> 1;
	// n2 is an upper bound on the number of secant lines

	secants = NEW_lint(n2);


	if (f_v) {
		cout << "projective_space::find_lines_which_are_contained "
				"before find_secant_lines" << endl;
	}

	find_secant_lines(set1, sz1,
			secants, nb_secants, n2,
			0/*verbose_level - 3*/);

	if (f_v) {
		cout << "projective_space::find_lines_which_are_contained "
				"after find_secant_lines" << endl;
	}

	if (f_vv) {
		cout << "projective_space::find_lines_which_are_contained "
				"we found " << nb_secants
				<< " secants in the hyperplane" << endl;
	}

	// first we test the secants and
	// find those which are lines on the surface:
	if (f_vv) {
		cout << "projective_space::find_lines_which_are_contained "
				"testing secants, nb_secants=" << nb_secants << endl;
	}

	//nb_lines = 0;
	for (i = 0; i < nb_secants; i++) {
		rk = secants[i];
		Grass_lines->unrank_lint_here(M, rk, 0 /* verbose_level */);
		if (f_vvv) {
			cout << "testing secant " << i << " / " << nb_secants
					<< " which is line " << rk << ":" << endl;
			Orbiter->Int_vec.matrix_print(M, 2, d);
		}

		int coeffs[2];

		// loop over all points on the line:
		for (a = 0; a < q + 1; a++) {

			// unrank a point on the projective line:
			F->PG_element_unrank_modified(coeffs, 1, 2, a);
			Orbiter->Int_vec.copy(M, M2, 2 * d);

			// map the point to the line at hand.
			// form the linear combination:
			// coeffs[0] * row0 of M2 + coeffs[1] * row1 of M2:
			for (h = 0; h < d; h++) {
				M2[2 * d + h] = F->add(
						F->mult(coeffs[0], M2[0 * d + h]),
						F->mult(coeffs[1], M2[1 * d + h]));
			}

			// rank the test point and see
			// if it belongs to the surface:
			F->PG_element_rank_modified_lint(M2 + 2 * d, 1, d, b);
			if (!Sorting.lint_vec_search(set1, sz1, b, idx, 0)) {
				break;
			}
		}
		if (a == q + 1) {
			// all q + 1 points of the secant line
			// belong to the surface, so we
			// found a line on the surface in the hyperplane.
			//lines[nb_lines++] = rk;
			if (f_vv) {
				cout << "secant " << i << " / " << nb_secants
						<< " of rank " << rk << " is contained, adding" << endl;
			}
			Lines.push_back(rk);
		}
	}
	FREE_lint(secants);

	if (f_v) {
		cout << "projective_space::find_lines_which_are_contained "
				"We found " << Lines.size() << " in the hyperplane" << endl;
		//lint_vec_print(cout, lines, nb_lines);
		cout << endl;
	}
	
	

	Pts1 = NEW_int(sz1 * d);
	Pts2 = NEW_int(sz2 * d);
	
	for (i = 0; i < sz1; i++) {
		unrank_point(Pts1 + i * d, set1[i]);
	}
	for (i = 0; i < sz2; i++) {
		unrank_point(Pts2 + i * d, set2[i]);
	}

	if (f_vv) {
		cout << "projective_space::find_lines_which_are_contained "
				"checking lines through points of the hyperplane, sz1=" << sz1 << endl;
	}

	f_taken = NEW_int(sz2);
	for (i = 0; i < sz1; i++) {
		if (f_vvv) {
			cout << "projective_space::find_lines_which_are_contained "
					"checking lines through hyperplane point " << i
					<< " / " << sz1 << ":" << endl;
		}

		// consider a point P1 on the surface and in the hyperplane

		Orbiter->Int_vec.zero(f_taken, sz2);
		for (j = 0; j < sz2; j++) {
			if (f_taken[j]) {
				continue;
			}
			if (f_vvv) {
				cout << "projective_space::find_lines_which_are_contained "
						"i=" << i << " j=" << j << " / " << sz2 << ":" << endl;
			}

			// consider a point P2 on the surface
			// but not in the hyperplane:

			Orbiter->Int_vec.copy(Pts1 + i * d, M, d);
			Orbiter->Int_vec.copy(Pts2 + j * d, M + d, d);

			f_taken[j] = TRUE;

			if (f_vvv) {
				Orbiter->Int_vec.matrix_print(M, 2, d);
			}

			rk = Grass_lines->rank_lint_here(M, 0 /* verbose_level */);
			if (f_vvv) {
				cout << "projective_space::find_lines_which_are_contained "
						"line rk=" << rk << ":" << endl;
			}

			// test the q-1 points on the line through the P1 and P2
			// (but excluding P1 and P2 themselves):
			for (a = 1; a < q; a++) {
				Orbiter->Int_vec.copy(M, M2, 2 * d);

				// form the linear combination P3 = P1 + a * P2:
				for (h = 0; h < d; h++) {
					M2[2 * d + h] =
						F->add(
								M2[0 * d + h],
								F->mult(a, M2[1 * d + h]));
				}
				// row 2 of M2 contains the coordinates of the point P3:
				F->PG_element_rank_modified_lint(M2 + 2 * d, 1, d, b);
				if (!Sorting.lint_vec_search(set2, sz2, b, idx, 0)) {
					break;
				}
				else {
					if (f_vvv) {
						cout << "eliminating point " << idx << endl;
					}
					// we don't need to consider this point for P2:
					f_taken[idx] = TRUE;
				}
			}
			if (a == q) {
				// The line P1P2 is contained in the surface.
				// Add it to lines[]
#if 0
				if (nb_lines == max_lines) {
					cout << "projective_space::find_lines_which_are_"
							"contained nb_lines == max_lines" << endl;
					exit(1);
				}
#endif
				//lines[nb_lines++] = rk;
				if (f_vvv) {
					cout << "adding line " << rk << " nb_lines="
							<< Lines.size() << endl;
				}
				Lines.push_back(rk);
				if (f_vvv) {
					cout << "adding line " << rk << " nb_lines="
							<< Lines.size() << " done" << endl;
				}
			}
		}
	}
	FREE_int(M);
	FREE_int(M2);
	FREE_lint(set1);
	FREE_lint(set2);
	FREE_int(Pts1);
	FREE_int(Pts2);
	FREE_int(f_taken);

	if (f_v) {
		cout << "projective_space::find_lines_which_are_contained done" << endl;
	}
}


void projective_space::point_plane_incidence_matrix(
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
	grassmann *G;

	if (f_v) {
		cout << "projective_space::plane_intersection_type_basic" << endl;
	}
	d = n + 1;
	M1 = NEW_int(4 * d);
	M2 = NEW_int(4 * d);
	Pts = NEW_int(nb_points * d);
	G = NEW_OBJECT(grassmann);

	G->init(d, 3, F, 0 /* verbose_level */);

	M = NEW_int(nb_points * nb_planes);
	Orbiter->Int_vec.zero(M, nb_points * nb_planes);

	// unrank all point here so we don't
	// have to do it again in the loop
	for (i = 0; i < nb_points; i++) {
		unrank_point(Pts + i * d, point_rks[i]);
	}

	for (j = 0; j < nb_planes; j++) {
		if (rk && (rk % ONE_MILLION) == 0) {
			cout << "projective_space::plane_intersection_type_basic "
					"rk=" << rk << endl;
		}
		rk = plane_rks[j];
		G->unrank_lint_here(M1, rk, 0 /* verbose_level */);

		// check which points are contained in the plane:
		for (i = 0; i < nb_points; i++) {

			Orbiter->Int_vec.copy(M1, M2, 3 * d);
			//unrank_point(M2 + 3 * d, set[h]);
			Orbiter->Int_vec.copy(Pts + i * d, M2 + 3 * d, d);

			if (F->rank_of_rectangular_matrix(M2,
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
	FREE_OBJECT(G);
	if (f_v) {
		cout << "projective_space::plane_intersection_type_basic done" << endl;
	}
}



void projective_space::plane_intersection_type_basic(
	long int *set, int set_size,
	int *type, int verbose_level)
// type[N_planes]
{
	int f_v = (verbose_level >= 1);
	int rk, h, d, N_planes;
	int *M1;
	int *M2;
	int *Pts;
	grassmann *G;

	if (f_v) {
		cout << "projective_space::plane_intersection_type_basic" << endl;
	}
	d = n + 1;
	M1 = NEW_int(4 * d);
	M2 = NEW_int(4 * d);
	Pts = NEW_int(set_size * d);
	G = NEW_OBJECT(grassmann);

	G->init(d, 3, F, 0 /* verbose_level */);

	N_planes = nb_rk_k_subspaces_as_lint(3);
	if (f_v) {
		cout << "projective_space::plane_intersection_type_basic "
				"N_planes=" << N_planes << endl;
	}

	// unrank all point here so we don't
	// have to do it again in the loop
	for (h = 0; h < set_size; h++) {
		unrank_point(Pts + h * d, set[h]);
	}

	for (rk = 0; rk < N_planes; rk++) {
		if (rk && (rk % ONE_MILLION) == 0) {
			cout << "projective_space::plane_intersection_type_basic "
					"rk=" << rk << endl;
		}
		type[rk] = 0;
		G->unrank_lint_here(M1, rk, 0 /* verbose_level */);

		// check which points are contained in the plane:
		for (h = 0; h < set_size; h++) {

			Orbiter->Int_vec.copy(M1, M2, 3 * d);
			//unrank_point(M2 + 3 * d, set[h]);
			Orbiter->Int_vec.copy(Pts + h * d, M2 + 3 * d, d);

			if (F->rank_of_rectangular_matrix(M2,
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
	FREE_OBJECT(G);
	if (f_v) {
		cout << "projective_space::plane_intersection_type_basic done" << endl;
	}
}

void projective_space::hyperplane_intersection_type_basic(
		long int *set, int set_size, int *type,
	int verbose_level)
// type[N_hyperplanes]
{
	int f_v = (verbose_level >= 1);
	int rk, h, d, N_hyperplanes;
	int *M;
	int *Pts;
	grassmann *G;

	if (f_v) {
		cout << "projective_space::hyperplane_intersection_type_basic" << endl;
	}
	d = n + 1;
	M = NEW_int(4 * d);
	Pts = NEW_int(set_size * d);
	G = NEW_OBJECT(grassmann);

	G->init(d, d - 1, F, 0 /* verbose_level */);

	N_hyperplanes = nb_rk_k_subspaces_as_lint(d - 1);
	
	// unrank all point here so we don't
	// have to do it again in the loop
	for (h = 0; h < set_size; h++) {
		unrank_point(Pts + h * d, set[h]);
	}

	for (rk = 0; rk < N_hyperplanes; rk++) {
		type[rk] = 0;
		G->unrank_lint(rk, 0 /* verbose_level */);

		// check which points are contained in the hyperplane:
		for (h = 0; h < set_size; h++) {

			Orbiter->Int_vec.copy(G->M, M, (d - 1) * d);
			//unrank_point(M + (d - 1) * d, set[h]);
			Orbiter->Int_vec.copy(Pts + h * d, M + (d - 1) * d, d);

			if (F->rank_of_rectangular_matrix(M,
					d, d, 0 /*verbose_level*/) == d - 1) {
				// the point lies in the hyperplane,
				// increment the intersection count:
				type[rk]++;
			}
		} // next h

	} // next rk
	FREE_int(M);
	FREE_int(Pts);
	FREE_OBJECT(G);
	if (f_v) {
		cout << "projective_space::hyperplane_intersection_type_basic done" << endl;
	}
}



void projective_space::line_intersection_type_collected(
	long int *set, int set_size, int *type_collected,
	int verbose_level)
// type[set_size + 1]
{
	int f_v = (verbose_level >= 1);
	int rk, h, d, cnt;
	int *M;
	int *Pts;

	if (f_v) {
		cout << "projective_space::line_intersection_type_collected" << endl;
	}
	d = n + 1;
	M = NEW_int(3 * d);
	Pts = NEW_int(set_size * d);
	Orbiter->Int_vec.zero(type_collected, set_size + 1);

	// unrank all point here so we don't
	// have to do it again in the loop
	for (h = 0; h < set_size; h++) {
		unrank_point(Pts + h * d, set[h]);
	}

	// loop over all lines:
	for (rk = 0; rk < N_lines; rk++) {
		Grass_lines->unrank_lint(rk, 0 /* verbose_level */);
		cnt = 0;

		// find which points in the set lie on the line:
		for (h = 0; h < set_size; h++) {

			Orbiter->Int_vec.copy(Grass_lines->M, M, 2 * d);


			//unrank_point(M + 2 * d, set[h]);
			Orbiter->Int_vec.copy(Pts + h * d, M + 2 * d, d);

			// test if the point lies on the line:
			if (F->rank_of_rectangular_matrix(M,
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
		cout << "projective_space::line_intersection_type_collected done" << endl;
	}
}

void projective_space::point_types_of_line_set(
	long int *set_of_lines, int set_size,
	int *type, int verbose_level)
{
	int i, j, a, b;

	for (i = 0; i < N_points; i++) {
		type[i] = 0;
	}
	for (i = 0; i < set_size; i++) {
		a = set_of_lines[i];
		for (j = 0; j < k; j++) {
			b = Implementation->Lines[a * k + j];
			type[b]++;
		}
	}
}

void projective_space::point_types_of_line_set_int(
	int *set_of_lines, int set_size,
	int *type, int verbose_level)
{
	int i, j, a, b;

	for (i = 0; i < N_points; i++) {
		type[i] = 0;
	}
	for (i = 0; i < set_size; i++) {
		a = set_of_lines[i];
		for (j = 0; j < k; j++) {
			b = Implementation->Lines[a * k + j];
			type[b]++;
		}
	}
}

void projective_space::find_external_lines(
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

void projective_space::find_tangent_lines(
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

void projective_space::find_secant_lines(
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

void projective_space::find_k_secant_lines(
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


void projective_space::Baer_subline(long int *pts3,
	long int *&pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int *M;
	int *Basis;
	int *N; // local coordinates w.r.t. basis
	int *base_cols;
	int *z;
	int rk;
	int len;
	int i, j;
	number_theory_domain NT;

	if (f_v) {
		cout << "projective_space::Baer_subline" << endl;
	}
	if (ODD(F->e)) {
		cout << "projective_space::Baer_subline field degree "
				"must be even (because we need a "
				"quadratic subfield)" << endl;
		exit(1);
	}
	len = n + 1;
	M = NEW_int(3 * len);
	base_cols = NEW_int(len);
	z = NEW_int(len);
	for (j = 0; j < 3; j++) {
		unrank_point(M + j * len, pts3[j]);
	}
	if (f_vv) {
		cout << "projective_space::Baer_subline" << endl;
		cout << "M=" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				M, 3, len, len, F->log10_of_q);
	}
	rk = F->Gauss_simple(M,
			3, len, base_cols, verbose_level - 3);
	if (f_vv) {
		cout << "projective_space::Baer_subline" << endl;
		cout << "has rank " << rk << endl;
		cout << "base_cols=";
		Orbiter->Int_vec.print(cout, base_cols, rk);
		cout << endl;
		cout << "basis:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				M, rk, len, len, F->log10_of_q);
	}

	if (rk != 2) {
		cout << "projective_space::Baer_subline: rk should "
				"be 2 (points are not collinear)" << endl;
		exit(1);
	}
	
	Basis = NEW_int(rk * len);
	for (j = 0; j < rk * len; j++) {
		Basis[j] = M[j];
	}
	if (f_vv) {
		cout << "projective_space::Baer_subline basis:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Basis, rk, len, len, F->log10_of_q);
	}
		
	N = NEW_int(3 * rk);
	for (j = 0; j < 3; j++) {
		unrank_point(M + j * len, pts3[j]);
		//cout << "M + j * len:";
		//int_vec_print(cout, M + j * len, len);
		//cout << endl;
		//cout << "basis:" << endl;
		//print_integer_matrix_width(cout,
		//Basis, rk, 5, 5, P4->F->log10_of_q);
		
		F->reduce_mod_subspace_and_get_coefficient_vector(
			rk, len, Basis, base_cols, 
			M + j * len, N + j * rk, verbose_level - 3);
	}
	//cout << "after reduce_mod_subspace_and_get_
	//coefficient_vector: M=" << endl;
	//print_integer_matrix_width(cout, M, 3, len, len, F->log10_of_q);
	//cout << "(should be all zeros)" << endl;
	if (f_vv) {
		cout << "projective_space::Baer_subline "
				"local coordinates in the subspace are N=" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				N, 3, rk, rk, F->log10_of_q);
	}
	int *Frame;
	int *base_cols2;
	int rk2, a;

	Frame = NEW_int(2 * 3);
	base_cols2 = NEW_int(3);
	for (j = 0; j < 3; j++) {
		for (i = 0; i < 2; i++) {
			Frame[i * 3 + j] = N[j * 2 + i];
		}
	}
	if (f_vv) {
		cout << "projective_space::Baer_subline "
				"Frame=" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Frame, 2, 3, 3, F->log10_of_q);
	}
	rk2 = F->Gauss_simple(Frame,
			2, 3, base_cols2, verbose_level - 3);
	if (rk2 != 2) {
		cout << "projective_space::Baer_subline: "
				"rk2 should be 2" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "projective_space::Baer_subline "
				"after Gauss Frame=" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Frame, 2, 3, 3, F->log10_of_q);
		cout << "projective_space::Baer_subline "
				"base_cols2=";
		Orbiter->Int_vec.print(cout, base_cols2, rk2);
		cout << endl;
	}
	for (i = 0; i < 2; i++) {
		a = Frame[i * 3 + 2];
		for (j = 0; j < 2; j++) {
			N[i * 2 + j] = F->mult(a, N[i * 2 + j]);
		}
	}
	if (f_vv) {
		cout << "projective_space::Baer_subline "
				"local coordinates in the subspace are N=" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, N, 3, rk, rk, F->log10_of_q);
	}

#if 0
	int *Local_pts;
	int *Local_pts_sorted;
	int w[2];


	Local_pts = NEW_int(nb_pts);
	Local_pts_sorted = NEW_int(nb_pts);

	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < 2; j++) {
			w[j] = N[i * 2 + j];
		}
		PG_element_rank_modified(*F, w, 1, 2, a);
		Local_pts[i] = a;
		Local_pts_sorted[i] = a;
	}
	int_vec_heapsort(Local_pts_sorted, nb_pts);
	if (f_vv) {
		cout << "Local_pts=" << endl;
		int_vec_print(cout, Local_pts, nb_pts);
		cout << endl;
		cout << "Local_pts_sorted=" << endl;
		int_vec_print(cout, Local_pts_sorted, nb_pts);
		cout << endl;
	}
#endif


	int q0, index, t;


	q0 = NT.i_power_j(F->p, F->e >> 1);
	index = (F->q - 1) / (q0 - 1);
	
	nb_pts = q0 + 1;
	pts = NEW_lint(nb_pts);


	if (f_vv) {
		cout << "projective_space::Baer_subline q0=" << q0 << endl;
		cout << "projective_space::Baer_subline index=" << index << endl;
		cout << "projective_space::Baer_subline nb_pts=" << nb_pts << endl;
	}

#if 0
	for (i = 0; i < 3; i++) {
		for (j = 0; j < len; j++) {
			if (i < 2) {
				z[j] = Basis[i * len + j];
			}
			else {
				z[j] = F->add(Basis[0 * len + j], Basis[1 * len + j]);
			}
		}
		pts[i] = rank_point(z);
	}
#endif
	for (t = 0; t < 3; t++) {
		if (f_vvv) {
			cout << "t=" << t << endl;
		}
		F->mult_vector_from_the_left(N + t * 2, Basis, z, 2, len);
		if (f_vvv) {
			cout << "z=w*Basis";
			Orbiter->Int_vec.print(cout, z, len);
			cout << endl;
		}
		a = rank_point(z);
		pts[t] = a;
	}
	for (t = 2; t < q0; t++) {
		a = F->alpha_power((t - 1) * index);
		if (f_vvv) {
			cout << "t=" << t << " a=" << a << endl;
		}
		for (j = 0; j < 2; j++) {
			w[j] = F->add(N[0 * 2 + j], F->mult(a, N[1 * 2 + j]));
		}
		if (f_vvv) {
			cout << "w=";
			Orbiter->Int_vec.print(cout, w, 2);
			cout << endl;
		}
		F->mult_vector_from_the_left(w, Basis, z, 2, len);
		if (f_vvv) {
			cout << "z=w*Basis";
			Orbiter->Int_vec.print(cout, z, len);
			cout << endl;
		}
		a = rank_point(z);
		pts[t + 1] = a;
		if (f_vvv) {
			cout << "rank=" << a << endl;
		}
#if 0
		PG_element_rank_modified(*F, w, 1, 2, a);
		pts[t] = a;
		if (!int_vec_search(Local_pts_sorted, nb_pts, a, idx)) {
			ret = FALSE;
			if (f_vv) {
				cout << "did not find this point in the list of "
						"points, hence not contained in Baer subline" << endl;
			}
			goto done;
		}
#endif
		
	}

	if (f_vv) {
		cout << "projective_space::Baer_subline The Baer subline is";
		Orbiter->Lint_vec.print(cout, pts, nb_pts);
		cout << endl;
		print_set(pts, nb_pts);
	}
	



	FREE_int(N);
	FREE_int(M);
	FREE_int(base_cols);
	FREE_int(Basis);
	FREE_int(Frame);
	FREE_int(base_cols2);
	FREE_int(z);
}

void projective_space::report_summary(ostream &ost)
{
	//ost << "\\parindent=0pt" << endl;
	ost << "$q = " << F->q << "$\\\\" << endl;
	ost << "$p = " << F->p << "$\\\\" << endl;
	ost << "$e = " << F->e << "$\\\\" << endl;
	ost << "$n = " << n << "$\\\\" << endl;
	ost << "Number of points = " << N_points << "\\\\" << endl;
	ost << "Number of lines = " << N_lines << "\\\\" << endl;
	ost << "Number of lines on a point = " << r << "\\\\" << endl;
	ost << "Number of points on a line = " << k << "\\\\" << endl;
}

void projective_space::report(ostream &ost,
		layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::report" << endl;
	}

	ost << "\\subsection*{The projective space ${\\rm \\PG}(" << n << "," << F->q << ")$}" << endl;
	ost << "\\noindent" << endl;
	ost << "\\arraycolsep=2pt" << endl;
	ost << "\\parindent=0pt" << endl;
	ost << "$q = " << F->q << "$\\\\" << endl;
	ost << "$p = " << F->p << "$\\\\" << endl;
	ost << "$e = " << F->e << "$\\\\" << endl;
	ost << "$n = " << n << "$\\\\" << endl;
	ost << "Number of points = " << N_points << "\\\\" << endl;
	ost << "Number of lines = " << N_lines << "\\\\" << endl;
	ost << "Number of lines on a point = " << r << "\\\\" << endl;
	ost << "Number of points on a line = " << k << "\\\\" << endl;

	//ost<< "\\clearpage" << endl << endl;
	//ost << "\\section{The Finite Field with $" << q << "$ Elements}" << endl;
	//F->cheat_sheet(ost, verbose_level);

#if 0
	if (f_v) {
		cout << "projective_space::report before incidence_matrix_save_csv" << endl;
	}
	incidence_matrix_save_csv();
	if (f_v) {
		cout << "projective_space::report after incidence_matrix_save_csv" << endl;
	}
#endif

	if (n == 2) {
		//ost << "\\clearpage" << endl << endl;
		ost << "\\subsection*{The plane}" << endl;
		string fname_base;
		char str[1000];
		long int *set;
		int i;
		//int rad = 17000;

		set = NEW_lint(N_points);
		for (i = 0; i < N_points; i++) {
			set[i] = i;
		}
		sprintf(str, "plane_of_order_%d", q);
		fname_base.assign(str);

		draw_point_set_in_plane(fname_base,
				O,
				set, N_points,
				TRUE /*f_point_labels*/,
				//FALSE /*f_embedded*/,
				//FALSE /*f_sideways*/,
				//rad,
				verbose_level);
		FREE_lint(set);
		ost << "{\\scriptsize" << endl;
		ost << "$$" << endl;
		ost << "\\input " << fname_base << "_draw.tex" << endl;
		ost << "$$" << endl;
		ost << "}%%" << endl;
	}

	//ost << "\\clearpage" << endl << endl;
	ost << "\\subsection*{The points of ${\\rm \\PG}(" << n << "," << F->q << ")$}" << endl;
	cheat_sheet_points(ost, verbose_level);

	//cheat_sheet_point_table(ost, verbose_level);


#if 0
	//ost << "\\clearpage" << endl << endl;
	cheat_sheet_points_on_lines(ost, verbose_level);

	//ost << "\\clearpage" << endl << endl;
	cheat_sheet_lines_on_points(ost, verbose_level);
#endif

	// report subspaces:
	int k;

	for (k = 1; k < n; k++) {
		//ost << "\\clearpage" << endl << endl;
		if (k == 1) {
			ost << "\\subsection*{The lines of ${\\rm \\PG}(" << n << "," << F->q << ")$}" << endl;
		}
		else if (k == 2) {
			ost << "\\subsection*{The planes of ${\\rm \\PG}(" << n << "," << F->q << ")$}" << endl;
		}
		else {
			ost << "\\subsection*{The subspaces of dimension " << k << " of ${\\rm \\PG}(" << n << "," << F->q << ")$}" << endl;
		}
		//ost << "\\section{Subspaces of dimension " << k << "}" << endl;
		if (f_v) {
			cout << "projective_space::report before cheat_sheet_subspaces, k=" << k << endl;
		}
		cheat_sheet_subspaces(ost, k, verbose_level);
		if (f_v) {
			cout << "projective_space::report after cheat_sheet_subspaces, k=" << k << endl;
		}
	}


#if 0
	if (n >= 2 && N_lines < 25) {
		//ost << "\\clearpage" << endl << endl;
		ost << "\\section*{Line intersections}" << endl;
		cheat_sheet_line_intersection(ost, verbose_level);
	}


	if (n >= 2 && N_points < 25) {
		//ost << "\\clearpage" << endl << endl;
		ost << "\\section*{Line through point-pairs}" << endl;
		cheat_sheet_line_through_pairs_of_points(ost, verbose_level);
	}
#endif

	homogeneous_polynomial_domain *Poly1;
	homogeneous_polynomial_domain *Poly2;
	homogeneous_polynomial_domain *Poly3;
	homogeneous_polynomial_domain *Poly4;

	Poly1 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly2 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly3 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly4 = NEW_OBJECT(homogeneous_polynomial_domain);

	ost << "\\subsection*{The polynomial rings associated "
			"with ${\\rm \\PG}(" << n << "," << F->q << ")$}" << endl;
	Poly1->init(F,
			n + 1 /* nb_vars */, 1 /* degree */,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);
	Poly2->init(F,
			n + 1 /* nb_vars */, 2 /* degree */,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);
	Poly3->init(F,
			n + 1 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);
	Poly4->init(F,
			n + 1 /* nb_vars */, 4 /* degree */,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);

	Poly1->print_monomial_ordering(ost);
	Poly2->print_monomial_ordering(ost);
	Poly3->print_monomial_ordering(ost);
	Poly4->print_monomial_ordering(ost);

	FREE_OBJECT(Poly1);
	FREE_OBJECT(Poly2);
	FREE_OBJECT(Poly3);
	FREE_OBJECT(Poly4);

	if (f_v) {
		cout << "projective_space::report done" << endl;
	}

}

void projective_space::incidence_matrix_save_csv()
{
	int i, j, k;
	int *T;
	file_io Fio;

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
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	FREE_int(T);
}

void projective_space::make_fname_incidence_matrix_csv(std::string &fname)
{
	char str[1000];

	sprintf(str, "PG_n%d_q%d", n, q);
	fname.assign(str);
	fname.append("_incidence_matrix.csv");
}


void projective_space::create_latex_report(
		layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space::create_latex_report" << endl;
	}

	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "PG_%d_%d.tex", n, F->q);
		fname.assign(str);
		snprintf(title, 1000, "Cheat Sheet PG($%d,%d$)", n, F->q);
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "projective_space::create_latex_report before P->report" << endl;
			}
			report(ost, O, verbose_level);
			if (f_v) {
				cout << "projective_space::create_latex_report after P->report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "projective_space::create_latex_report done" << endl;
	}
}

void projective_space::create_latex_report_for_Grassmannian(int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space::create_latex_report_for_Grassmannian" << endl;
	}

	{
		char str[1000];
		string fname;
		char title[1000];
		char author[1000];

		snprintf(str, 1000, "Gr_%d_%d_%d.tex", n + 1, k, F->q);
		fname.assign(str);
		snprintf(title, 1000, "Cheat Sheet ${\\rm Gr}_{%d,%d,%d}$", n + 1, k, F->q);
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "projective_space::create_latex_report_for_Grassmannian "
						"before cheat_sheet_subspaces, k=" << k << endl;
			}
			cheat_sheet_subspaces(ost, k - 1, verbose_level);
			if (f_v) {
				cout << "projective_space::create_latex_report_for_Grassmannian "
						"after cheat_sheet_subspaces, k=" << k << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "projective_space::create_latex_report_for_Grassmannian done" << endl;
	}
}

void projective_space::compute_decomposition(partitionstack *S1, partitionstack *S2,
		incidence_structure *&Inc, partitionstack *&Stack, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::compute_decomposition" << endl;
	}
	if (f_v) {
		cout << "projective_space::compute_decomposition "
				"before incidence_and_stack_for_type_ij" << endl;
	}
	incidence_and_stack_for_type_ij(
		1 /* row_type */, 2 /* col_type */,
		Inc,
		Stack,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space::compute_decomposition "
				"after incidence_and_stack_for_type_ij" << endl;
	}

	int i, j, sz;

	for (i = 1; i < S1->ht; i++) {
		if (f_v) {
			cout << "projective_space::compute_decomposition "
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
		Orbiter->Int_vec.copy(S2->pointList + S2->startCell[i], set, sz);
		for (j = 0; j < sz; j++) {
			set[j] += Inc->nb_rows;
		}
		if (f_v) {
			cout << "projective_space::compute_decomposition "
					"before Stack->split_cell (S2) i=" << i << endl;
		}
		Stack->split_cell(set, sz, 0 /*verbose_level*/);
	}
	FREE_int(set);
	if (f_v) {
		cout << "projective_space::compute_decomposition done" << endl;
	}

}

void projective_space::compute_decomposition_based_on_tally(tally *T1, tally *T2,
		incidence_structure *&Inc, partitionstack *&Stack, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::compute_decomposition_based_on_tally" << endl;
	}
	if (f_v) {
		cout << "projective_space::compute_decomposition_based_on_tally "
				"before incidence_and_stack_for_type_ij" << endl;
	}
	incidence_and_stack_for_type_ij(
		1 /* row_type */, 2 /* col_type */,
		Inc,
		Stack,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space::compute_decomposition_based_on_tally "
				"after incidence_and_stack_for_type_ij" << endl;
	}

	int i, j, sz;

	for (i = T1->nb_types - 1; i >= 1; i--) {
		if (f_v) {
			cout << "projective_space::compute_decomposition_based_on_tally "
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
		Orbiter->Int_vec.copy(T2->sorting_perm_inv + T2->type_first[i], set, sz);
		for (j = 0; j < sz; j++) {
			set[j] += Inc->nb_rows;
		}
		if (f_v) {
			cout << "projective_space::compute_decomposition_based_on_tally "
					"before Stack->split_cell (S2) i=" << i << endl;
		}
		Stack->split_cell(set, sz, 0 /*verbose_level*/);
	}
	FREE_int(set);
	if (f_v) {
		cout << "projective_space::compute_decomposition_based_on_tally done" << endl;
	}

}

object_in_projective_space *projective_space::create_object_from_string(
		int type, std::string &input_fname, int input_idx,
		std::string &set_as_string, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::create_object_from_string" << endl;
		cout << "type=" << type << endl;
	}


	object_in_projective_space *OiP;

	OiP = NEW_OBJECT(object_in_projective_space);

	OiP->init_object_from_string(this,
			type, input_fname, input_idx,
			set_as_string, verbose_level);


	if (f_v) {
		cout << "projective_space::create_object_from_string"
				" done" << endl;
	}
	return OiP;
}

object_in_projective_space * projective_space::create_object_from_int_vec(
		int type, std::string &input_fname, int input_idx,
		long int *the_set, int set_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space::create_object_from_int_vec" << endl;
		cout << "type=" << type << endl;
	}


	object_in_projective_space *OiP;

	OiP = NEW_OBJECT(object_in_projective_space);

	OiP->init_object_from_int_vec(this,
			type, input_fname, input_idx,
			the_set, set_sz, verbose_level);


	if (f_v) {
		cout << "projective_space::create_object_from_int_vec"
				" done" << endl;
	}
	return OiP;
}


}}



