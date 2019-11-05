// projective_space.cpp
//
// Anton Betten
// Jan 17, 2010

#include "foundations.h"


using namespace std;


#define MAX_NUMBER_OF_LINES_FOR_INCIDENCE_MATRIX 100000
#define MAX_NUMBER_OF_LINES_FOR_LINE_TABLE 1000000
#define MAX_NUMBER_OF_POINTS_FOR_POINT_TABLE 1000000


namespace orbiter {
namespace foundations {


projective_space::projective_space()
{
	null();
};

projective_space::~projective_space()
{
	freeself();
}

void projective_space::null()
{
	Grass_lines = NULL;
	Grass_planes = NULL;
	F = NULL;
	Go = NULL;
	Nb_subspaces = NULL;
	incidence_bitvec = NULL;
	Line_through_two_points = NULL;
	Line_intersection = NULL;
	Lines = NULL;
	Lines_on_point = NULL;
	Polarity_point_to_hyperplane = NULL;
	Polarity_hyperplane_to_point = NULL;
	v = NULL;
	w = NULL;
	Mtx = NULL;
	Mtx2 = NULL;
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
	if (incidence_bitvec) {
		FREE_uchar(incidence_bitvec);
		}
	if (Line_through_two_points) {
		FREE_int(Line_through_two_points);
		}
	if (Line_intersection) {
		FREE_int(Line_intersection);
		}
	if (Lines) {
		FREE_int(Lines);
		}
	if (Lines_on_point) {
		if (f_v) {
			cout << "projective_space::freeself "
					"deleting Lines_on_point" << endl;
			}
		FREE_int(Lines_on_point);
		}
	if (Polarity_point_to_hyperplane) {
		FREE_int(Polarity_point_to_hyperplane);
		}
	if (Polarity_hyperplane_to_point) {
		FREE_int(Polarity_hyperplane_to_point);
		}
	null();
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
	longinteger_domain D;
	longinteger_object a;

	projective_space::n = n;
	projective_space::F = F;
	projective_space::q = F->q;
	
	if (f_v) {
		cout << "projective_space::init "
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
			D.q_binomial_no_table(
				a,
				n + 1, i + 1, q, verbose_level - 2);
			Nb_subspaces[i] = a.as_lint();
			//Nb_subspaces[i] = generalized_binomial(n + 1, i + 1, q);
			}

		D.q_binomial_no_table(
			a,
			n, 1, q, verbose_level - 2);
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
	int f_vv = FALSE; //(verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	int i, j, a, b, i1, i2, j1, j2;
	
	if (f_v) {
		cout << "projective_space::init_incidence_structure" << endl;
		}


	if (N_lines < MAX_NUMBER_OF_LINES_FOR_INCIDENCE_MATRIX) {
		if (f_v) {
			cout << "projective_space::init_incidence_structure "
					"allocating Incidence (bitvector)" << endl;
			}
		int len = (N_points * N_lines + 7) >> 3;
		if (f_v) {
			cout << "projective_space::init_incidence_structure "
					"allocating Incidence (bitvector) "
					"len = " << len << endl;
			}
		//Incidence = NEW_int(N_points * N_lines);
		//int_vec_zero(Incidence, N_points * N_points);
		incidence_bitvec = NEW_uchar(len);
		for (i = 0; i < len; i++) {
			incidence_bitvec[i] = 0;
			}

		}
	else {
		cout << "projective_space::init_incidence_structure: "
				"N_lines too big, we do not initialize the "
				"incidence matrix" << endl;
		//return;
		incidence_bitvec = NULL;
		}




	if (N_lines < MAX_NUMBER_OF_LINES_FOR_LINE_TABLE) {
		if (f_v) {
			cout << "projective_space::init_incidence_structure "
					"allocating Lines" << endl;
			}
		Lines = NEW_int(N_lines * k);
		}
	else {
		cout << "projective_space::init_incidence_structure: "
				"N_lines too big, we do not initialize "
				"the line table" << endl;
		Lines = NULL;
		}




	if (N_points < MAX_NUMBER_OF_POINTS_FOR_POINT_TABLE &&
			N_lines < MAX_NUMBER_OF_LINES_FOR_LINE_TABLE)  {
		if (f_v) {
			cout << "projective_space::init_incidence_structure "
					"allocating Lines_on_point" << endl;
			cout << "projective_space::init_incidence_structure "
					"allocating N_points=" << N_points << endl;
			cout << "projective_space::init_incidence_structure "
					"allocating r=" << r << endl;
			}
		Lines_on_point = NEW_int(N_points * r);
		}
	else {
		cout << "projective_space::init_incidence_structure: "
				"N_points too big, we do not initialize the "
				"Lines_on_point table" << endl;
		Lines_on_point = NULL;
		}



	if ((long int) N_points * (long int) N_points < ONE_MILLION) {

		if (f_v) {
			cout << "projective_space::init_incidence_structure "
					"allocating Line_through_two_points" << endl;
			}
		Line_through_two_points = NEW_int((long int) N_points * (long int) N_points);
		}
	else {
		cout << "projective_space::init_incidence_structure: "
				"we do not initialize "
				"Line_through_two_points" << endl;
		Line_through_two_points = NULL;
		}

	if ((long int) N_lines * (long int) N_lines < ONE_MILLION) {
		if (f_v) {
			cout << "projective_space::init_incidence_structure "
					"allocating Line_intersection" << endl;
			}
		Line_intersection = NEW_int((long int) N_lines * (long int) N_lines);
		int_vec_zero(Line_through_two_points, (long int) N_points * (long int) N_points);
		for (i = 0; i < (long int) N_lines * (long int) N_lines; i++) {
			Line_intersection[i] = -1;
			}
		}
	else {
		cout << "projective_space::init_incidence_structure: "
				"we do not initialize "
				"Line_intersection" << endl;
		Line_intersection = NULL;
		}	

	
	if (f_v) {
		cout << "projective_space::init_incidence_structure "
				"number of points = " << N_points << endl;
		}
	if (f_vv) {
		for (i = 0; i < N_points; i++) {
			F->PG_element_unrank_modified(v, 1, n + 1, i);
			cout << "point " << i << " : ";
			int_vec_print(cout, v, n + 1);
			cout << " = ";
			F->int_vec_print(cout, v, n + 1);

			F->PG_element_normalize_from_front(v, 1, n + 1);
			cout << " = ";
			int_vec_print(cout, v, n + 1);

		
			cout << " = ";
			F->int_vec_print(cout, v, n + 1);

			
			cout << endl;
			}
		}
	if (f_v) {
		cout << "projective_space::init_incidence_structure "
				"number of lines = " << N_lines << endl;
		}



	if (Lines || incidence_bitvec || Lines_on_point) {


		if (f_v) {
			cout << "projective_space::init_incidence_structure "
					"computing lines..." << endl;
			if (Lines) {
				cout << "Lines is allocated" << endl;
			}
			if (incidence_bitvec) {
				cout << "incidence_bitvec is allocated" << endl;
			}
			if (Lines_on_point) {
				cout << "Lines_on_point is allocated" << endl;
			}
		}



		int *R;

		R = NEW_int(N_points);
		int_vec_zero(R, N_points);
	
		for (i = 0; i < N_lines; i++) {
#if 0
			if ((i % 1000000) == 0) {
				cout << "projective_space::init_incidence_structure "
						"Line " << i << " / " << N_lines << ":" << endl;
				}
#endif
			Grass_lines->unrank_lint(i, 0/*verbose_level - 4*/);
			if (FALSE) {
				print_integer_matrix_width(cout,
						Grass_lines->M, 2, n + 1, n + 1,
						F->log10_of_q + 1);
				}


#if 0
			// testing of grassmann:
			
			j = Grass_lines->rank_int(0/*verbose_level - 4*/);
			if (j != i) {
				cout << "projective_space::init_incidence_structure "
						"rank yields " << j << " != " << i << endl;
				exit(1);
				}
#endif



			for (a = 0; a < k; a++) {
				F->PG_element_unrank_modified(v, 1, 2, a);
				F->mult_matrix_matrix(v, Grass_lines->M, w, 1, 2, n + 1,
						0 /* verbose_level */);
				F->PG_element_rank_modified(w, 1, n + 1, b);
				if (incidence_bitvec) {
					incidence_m_ii(b, i, 1);
					}

				if (Lines) {
					Lines[i * k + a] = b;
					}
				if (Lines_on_point) {
					Lines_on_point[b * r + R[b]] = i;
					}
				R[b]++;
				}
			if (f_vv) {
				cout << "line " << i << ":" << endl;
				print_integer_matrix_width(cout, 
					Grass_lines->M, 2, n + 1, n + 1, 
					F->log10_of_q + 1);

				if (Lines) {
					cout << "points on line " << i << " : ";
					int_vec_print(cout, Lines + i * k, k);
					cout << endl;
					}
				}
		
			}
		for (i = 0; i < N_points; i++) {
			if (R[i] != r) {
				cout << "R[i] != r" << endl;
				exit(1);
				}
			}

		FREE_int(R);
		}
	else {
		if (f_v) {
			cout << "projective_space::init_incidence_structure "
					"computing lines skipped" << endl;
			}
		}

	if (n == 2) {
		if (f_v) {
			cout << "projective_space::init_incidence_structure "
				"computing polarity information..." << endl;
			}
		Polarity_point_to_hyperplane = NEW_int(N_points);
		Polarity_hyperplane_to_point = NEW_int(N_points);
		for (i = 0; i < N_lines; i++) {
			int *A, a;
			A = NEW_int((n + 1) * (n + 1));
			Grass_lines->unrank_lint(i, 0 /*verbose_level - 4*/);
			for (j = 0; j < 2 * (n + 1); j++) {
				A[j] = Grass_lines->M[j];
				}
			if (FALSE) {
				print_integer_matrix_width(cout, 
					A, 2, n + 1, n + 1, 
					F->log10_of_q + 1);
				}
			F->perp_standard(n + 1, 2, A, 0);
			if (FALSE) {
				print_integer_matrix_width(cout, 
					A, n + 1, n + 1, n + 1, 
					F->log10_of_q + 1);
				}
			F->PG_element_rank_modified(
				A + 2 * (n + 1), 1, n + 1, a);
			if (f_vv) {
				cout << "line " << i << " is ";
				int_vec_print(cout, A + 2 * (n + 1), n + 1);
				cout << "^\\perp = " << a << "^\\perp" << endl;
				}
			FREE_int(A);
			Polarity_point_to_hyperplane[a] = i;
			Polarity_hyperplane_to_point[i] = a;
			}
		if (FALSE /* f_vv */) {
			cout << "i : pt_to_hyperplane[i] : hyperplane_to_pt[i]" << endl;
			for (i = 0; i < N_lines; i++) {
				cout << setw(4) << i << " " 
					<< setw(4) << Polarity_point_to_hyperplane[i] << " " 
					<< setw(4) << Polarity_hyperplane_to_point[i] << endl;
				}
			}
		}
	


#if 0
	if (f_v) {
		cout << "computing Lines_on_point..." << endl;
		}
	for (i = 0; i < N_points; i++) {
		if ((i % 1000) == 0) {
			cout << "point " << i << " / " << N_points << ":" << endl;
			}
		a = 0;
		for (j = 0; j < N_lines; j++) {	
			if (is_incident(i, j)) {
				Lines_on_point[i * r + a] = j;
				a++;
				}
			}
		if (FALSE /*f_vv */) {
			cout << "lines on point " << i << " = ";
			PG_element_unrank_modified(*F, w, 1, n + 1, i);
			int_vec_print(cout, w, n + 1);
			cout << " : ";
			int_vec_print(cout, Lines_on_point + i * r, r);
			cout << endl;
			}
		}
	if (f_v) {
		cout << "computing Lines_on_point done" << endl;
		}
#endif

	if (FALSE) {
		//cout << "incidence matrix:" << endl;
		//print_integer_matrix_width(cout, Incidence, N_points, N_lines, N_lines, 1);
		cout << "projective_space::init_incidence_structure Lines:" << endl;
		print_integer_matrix_width(cout, Lines, N_lines, k, k, 3);
		cout << "projective_space::init_incidence_structure Lines_on_point:" << endl;
		print_integer_matrix_width(cout, Lines_on_point, N_points, r, r, 3);
		}
	

	if (Line_through_two_points && Lines && Lines_on_point) {
		if (f_v) {
			cout << "projective_space::init_incidence_structure "
					"computing Line_through_two_points..." << endl;
			}
		for (i1 = 0; i1 < N_points; i1++) {
			for (a = 0; a < r; a++) {
				j = Lines_on_point[i1 * r + a];
				for (b = 0; b < k; b++) {
					i2 = Lines[j * k + b];
					if (i2 == i1)
						continue;
					Line_through_two_points[i1 * N_points + i2] = j;
					Line_through_two_points[i2 * N_points + i1] = j;
					}
				}
			}
		if (FALSE) {
			cout << "line through points i,j is" << endl;
			for (i = 0; i < N_points; i++) {
				for (j = i + 1; j < N_points; j++) {
					cout << i << " , " << j << " : "
						<< Line_through_two_points[i * N_points + j] << endl;
					}
				}
			//cout << "Line_through_two_points:" << endl;
			//print_integer_matrix_width(cout,
			//Line_through_two_points, N_points, N_points, N_points, 2);
			}
		}
	else {
		if (f_v) {
			cout << "projective_space::init_incidence_structure "
				"computing Line_through_two_points skipped" << endl;
			}
		}

	if (Line_intersection && Lines && Lines_on_point) {
		if (f_v) {
			cout << "projective_space::init_incidence_structure "
				"computing Line_intersection..." << endl;
			}
		for (j1 = 0; j1 < N_lines; j1++) {
			for (a = 0; a < k; a++) {
				i = Lines[j1 * k + a];
				for (b = 0; b < r; b++) {
					j2 = Lines_on_point[i * r + b];
					if (j2 == j1)
						continue;
					Line_intersection[j1 * N_lines + j2] = i;
					Line_intersection[j2 * N_lines + j1] = i;
					}
				}
			}
		if (FALSE) {
			cout << "projective_space::init_incidence_structure "
					"point of intersection of lines i,j is" << endl;
			for (i = 0; i < N_lines; i++) {
				for (j = i + 1; j < N_lines; j++) {
					cout << i << " , " << j << " : "
						<< Line_intersection[i * N_lines + j] << endl;
					}
				}
			//cout << "Line_intersection:" << endl;
			//print_integer_matrix_width(cout,
			// Line_intersection, N_lines, N_lines, N_lines, 2);
			}
		}
	else {
		if (f_v) {
			cout << "projective_space::init_incidence_structure "
				"computing Line_intersection skipped" << endl;
			}
		}
	if (f_v) {
		cout << "projective_space::init_incidence_structure done" << endl;
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
	L = Lines + line_rk * k;
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
	long int line_rk, int *line, int verbose_level)
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
		int_matrix_print(Grass_lines->M, 2, n + 1);
		}

	F->PG_element_unrank_modified(v, 1, 2, pt_rk);
	if (f_v) {
		cout << "projective_space::create_point_on_line v=" << endl;
		int_vec_print(cout, v, 2);
		cout << endl;
		}

	F->mult_matrix_matrix(v, Grass_lines->M, w, 1, 2, n + 1,
			0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space::create_point_on_line w=" << endl;
		int_vec_print(cout, w, n + 1);
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
	int_vec_zero(Inc, m * n);
	for (i = 0; i < N_points; i++) {
		for (h = 0; h < r; h++) {
			j = Lines_on_point[i * r + h];
			Inc[i * n + j] = 1;
			}
		}
	if (f_v) {
		cout << "projective_space::make_incidence_matrix "
				"done" << endl;
		}
}

int projective_space::is_incident(int pt, int line)
{
	int f_v = FALSE;
	int a, rk;

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
			int_matrix_print(Grass_lines->M, 2, n + 1);
		}
		int_vec_copy(Grass_lines->M, Mtx, 2 * (n + 1));
		F->PG_element_unrank_modified(Mtx + 2 * (n + 1), 1, n + 1, pt);
		if (f_v) {
			cout << "point:" << endl;
			int_vec_print(cout, Mtx + 2 * (n + 1), n + 1);
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
	a = pt * N_lines + line;
	return bitvector_s_i(incidence_bitvec, a);
}

void projective_space::incidence_m_ii(int pt, int line, int a)
{
	int b;

	if (incidence_bitvec == 0) {
		cout << "projective_space::incidence_m_ii "
				"incidence_bitvec == 0" << endl;
		exit(1);
		}
	b = pt * N_lines + line;
	bitvector_m_ii(incidence_bitvec, b, a);
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
		cout << "projective_space::make_incidence_structure_"
				"and_partition" << endl;
		cout << "N_points=" << N_points << endl;
		cout << "N_lines=" << N_lines << endl;
		}
	Inc = NEW_OBJECT(incidence_structure);

	
	if (f_v) {
		cout << "projective_space::make_incidence_structure_and_"
				"partition allocating M of size "
				<< N_points * N_lines << endl;
		}
	M = NEW_int(N_points * N_lines);
	if (f_v) {
		cout << "projective_space::make_incidence_structure_and_"
				"partition after allocating M of size "
				<< N_points * N_lines << endl;
		}
	int_vec_zero(M, N_points * N_lines);

	if (Lines_on_point == NULL) {
		cout << "projective_space::make_incidence_structure_and_"
				"partition Lines_on_point == NULL" << endl;
		exit(1);
		}
	for (i = 0; i < N_points; i++) {
		for (h = 0; h < r; h++) {
			j = Lines_on_point[i * r + h];
			M[i * N_lines + j] = 1;
			}
		}
	if (f_v) {
		cout << "projective_space::make_incidence_structure_and_"
				"partition before Inc->init_by_matrix" << endl;
		}
	Inc->init_by_matrix(N_points, N_lines, M, verbose_level - 1);
	if (f_v) {
		cout << "projective_space::make_incidence_structure_and_"
				"partition after Inc->init_by_matrix" << endl;
		}
	FREE_int(M);


	Stack = NEW_OBJECT(partitionstack);
	Stack->allocate(N_points + N_lines, 0 /* verbose_level */);
	Stack->subset_continguous(N_points, N_lines);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();
	
	if (f_v) {
		cout << "projective_space::make_incidence_structure_and_"
				"partition done" << endl;
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
	int_vec_zero(Incma, nb_rows * nb_cols);
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
		cout << "projective_space::incidence_and_stack_for_type_ij "
				"done" << endl;
	}
}
long int projective_space::nb_rk_k_subspaces_as_lint(int k)
{
	longinteger_domain D;
	longinteger_object aa;
	long int N;
	int d = n + 1;

	D.q_binomial(aa, d, k, q, 0/*verbose_level*/);
	N = aa.as_lint();
	return N;
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
		int_vec_print(cout, v, n + 1);
		cout << endl;
		}
}

long int projective_space::rank_point(int *v)
{
	long int b;
	
	F->PG_element_rank_modified_lint(v, 1, n + 1, b);
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
	if (Lines) {
		return test_if_sets_are_disjoint_assuming_sorted(
				Lines + l1 * k, Lines + l2 * k, k, k);
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
	int_vec_copy(Mtx1 + 2 * d, Mtx3, (d - 2) * d);
	Grass_lines->unrank_lint_here(Mtx1, l2, 0/*verbose_level - 4*/);
	F->perp_standard(d, 2, Mtx1, 0);
	int_vec_copy(Mtx1 + 2 * d, Mtx3 + (d - 2) * d, (d - 2) * d);
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
	N = Combi.int_n_choose_k(nb_pts, 3);
	for (h = 0; h < N; h++) {
		Combi.unrank_k_subset(h, set, nb_pts, 3);
		int_vec_copy(Pts + set[0] * 3, Mtx, 3);
		int_vec_copy(Pts + set[1] * 3, Mtx + 3, 3);
		int_vec_copy(Pts + set[2] * 3, Mtx + 6, 3);
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
		print_integer_matrix_width(cout,
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
		print_integer_matrix_width(cout,
				system, nb_pts, 3, 3, F->log10_of_q);
		}



	rk = F->Gauss_simple(system,
			nb_pts, 3, base_cols, verbose_level - 2);
	if (rk != 2) {
		if (f_v) {
			cout << "projective_space::determine_line_in_plane "
					"system underdetermined" << endl;
			}
		return FALSE;
		}
	F->matrix_get_kernel(system, 2, 3, base_cols, rk, 
		kernel_m, kernel_n, kernel);
	if (f_v) {
		cout << "projective_space::determine_line_in_plane line:" << endl;
		print_integer_matrix_width(cout, kernel, 1, 3, 3, F->log10_of_q);
		}
	for (i = 0; i < 3; i++) {
		three_coeffs[i] = kernel[i];
		}
	FREE_int(coords);
	FREE_int(system);
	return TRUE;
}

int projective_space::determine_conic_in_plane(
	long int *input_pts, int nb_pts,
	int *six_coeffs, 
	int verbose_level)
// returns FALSE is the rank of the coefficient
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
		print_integer_matrix_width(cout,
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
		print_integer_matrix_width(cout,
				system, nb_pts, 6, 6, F->log10_of_q);
		}



	rk = F->Gauss_simple(system, nb_pts,
			6, base_cols, verbose_level - 2);
	if (rk != 5) {
		if (f_v) {
			cout << "projective_space::determine_conic_in_plane "
					"system underdetermined" << endl;
			}
		return FALSE;
		}
	F->matrix_get_kernel(system, 5, 6, base_cols, rk, 
		kernel_m, kernel_n, kernel);
	if (f_v) {
		cout << "projective_space::determine_conic_in_plane "
				"conic:" << endl;
		print_integer_matrix_width(cout,
				kernel, 1, 6, 6, F->log10_of_q);
		}
	for (i = 0; i < 6; i++) {
		six_coeffs[i] = kernel[i];
		}
	FREE_int(coords);
	FREE_int(system);
	return TRUE;
}


int projective_space::determine_cubic_in_plane(
		homogeneous_polynomial_domain *Poly_3_3,
		int nb_pts, int *Pts, int *coeff10,
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
	System = NEW_int(nb_pts * Poly_3_3->nb_monomials);
	base_cols = NEW_int(Poly_3_3->nb_monomials);

	if (f_v) {
		cout << "projective_space::determine_cubic_in_plane list of "
				"points:" << endl;
		int_vec_print(cout, Pts, nb_pts);
		cout << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		unrank_point(Pt_coord + i * d, Pts[i]);
		}
	if (f_v) {
		cout << "projective_space::determine_cubic_in_plane matrix of "
				"point coordinates:" << endl;
		int_matrix_print(Pt_coord, nb_pts, d);
		}

	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < Poly_3_3->nb_monomials; j++) {
			System[i * Poly_3_3->nb_monomials + j] =
				F->evaluate_monomial(
					Poly_3_3->Monomials + j * d,
					Pt_coord + i * d, d);
			}
		}
	if (f_v) {
		cout << "projective_space::determine_cubic_in_plane "
				"The system:" << endl;
		int_matrix_print(System, nb_pts, Poly_3_3->nb_monomials);
		}
	r = F->Gauss_simple(System, nb_pts, Poly_3_3->nb_monomials,
		base_cols, 0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space::determine_cubic_in_plane "
				"The system in RREF:" << endl;
		int_matrix_print(System, r, Poly_3_3->nb_monomials);
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

	F->matrix_get_kernel(System, r, Poly_3_3->nb_monomials,
		base_cols, r,
		kernel_m, kernel_n, coeff10);


	FREE_int(Pt_coord);
	FREE_int(System);
	FREE_int(base_cols);
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
		print_integer_matrix_width(cout,
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
		print_integer_matrix_width(cout,
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
		kernel_m, kernel_n, kernel);
	if (f_v) {
		cout << "projective_space::determine_quadric_in_solid "
				"conic:" << endl;
		print_integer_matrix_width(cout,
				kernel, 1, 10, 10, F->log10_of_q);
		}
	for (i = 0; i < 10; i++) {
		ten_coeffs[i] = kernel[i];
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
			int_vec_print(cout, v, 3);
			cout << " gives a value of " << a << endl;
			}
		if (a == 0) {
			if (f_vv) {
				cout << "point " << i << " = ";
				int_vec_print(cout, v, 3);
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
		lint_vec_print(cout, points, nb_points);
		cout << endl;
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
			int_vec_print(cout, v, 3);
			cout << " gives a value of " << a << endl;
			}
		if (a == 0) {
			if (f_vv) {
				cout << "point " << i << " = ";
				int_vec_print(cout, v, 4);
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
		lint_vec_print(cout, points, nb_points);
		cout << endl;
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
		print_integer_matrix_width(cout,
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
		print_integer_matrix_width(cout,
				Basis, 2, 3, 3, F->log10_of_q);
		}
	F->perp(3, 2, Basis, Gram_matrix);
	if (f_v) {	
		cout << "projective_space::conic_points perp:" << endl;
		print_integer_matrix_width(cout,
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
		print_integer_matrix_width(cout,
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
		int_vec_print(cout, w, 3);
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
			int_vec_print(cout, w, 3);
			cout << endl;
			}
		b = rank_point(w);
		points[nb_points++] = b;
		}
	if (f_vv) {	
		cout << "projective_space::conic_points conic points:" << endl;
		lint_vec_print(cout, points, nb_points);
		cout << endl;
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
		F->perp(3, 1, Basis, Gram_matrix);
		if (f_vv) {	
			cout << "perp:" << endl;
			print_integer_matrix_width(cout,
					Basis, 3, 3, 3, F->log10_of_q);
			}
		tangents[i] = rank_line(Basis + 3);
		if (f_vv) {	
			cout << "tangent at point " << i << " is "
					<< tangents[i] << endl;
			}
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
		int_matrix_print(bisecants, 15, 3);
		}

	for (j = 0; j < 6; j++) {
		//int deleted_point;
		
		//deleted_point = arc6[j];
		lint_vec_copy(arc6, arc5, j);
		lint_vec_copy(arc6 + j + 1, arc5 + j, 5 - j);

#if 0
		cout << "deleting point " << j << " / 6:";
		int_vec_print(cout, arc5, 5);
		cout << endl;
#endif

		determine_conic_in_plane(arc5, 5,
				six_coeffs, 0 /* verbose_level */);
		F->PG_element_normalize_from_front(six_coeffs, 1, 6);
		int_vec_copy(six_coeffs, conics + j * 6, 6);
		}

	if (f_v) {
		cout << "projective_space::compute_bisecants_and_conics "
				"conics:" << endl;
		int_matrix_print(conics, 6, 6);
		}

	if (f_v) {
		cout << "projective_space::compute_bisecants_and_conics "
				"done" << endl;
		}
}

eckardt_point_info *projective_space::compute_eckardt_point_info(
	long int *arc6,
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
		lint_vec_print(cout, arc6, 6);
		cout << endl;
		}

	E = NEW_OBJECT(eckardt_point_info);
	E->init(this, arc6, verbose_level);

	if (f_v) {
		cout << "projective_space::compute_eckardt_point_info done" << endl;
		}
	return E;
}

void projective_space::PG_2_8_create_conic_plus_nucleus_arc_1(
		long int *the_arc, int &size, int verbose_level)
{
	int frame_data[] = {1,0,0, 0,1,0,  0,0,1,  1,1,1 };
	int frame[4];
	int i, j, b, h, idx;
	int L[3];
	int v[3];
	sorting Sorting;

	if (n != 2) {
		cout << "projective_space::PG_2_8_create_conic_"
				"plus_nucleus_arc_1 n != 2" << endl;
		exit(1);
		}
	if (q != 8) {
		cout << "projective_space::PG_2_8_create_conic_"
				"plus_nucleus_arc_1 q != 8" << endl;
		exit(1);
		}
	for (i = 0; i < 4; i++) {
		frame[i] = rank_point(frame_data + i * 3);
		}

	cout << "frame: ";
	int_vec_print(cout, frame, 4);
	cout << endl;
	
	L[0] = Line_through_two_points[frame[0] * N_points + frame[1]];
	L[1] = Line_through_two_points[frame[1] * N_points + frame[2]];
	L[2] = Line_through_two_points[frame[2] * N_points + frame[0]];
	
	cout << "l1=" << L[0] << " l2=" << L[1] << " l3=" << L[2] << endl;

	size = 0;	
	for (h = 0; h < 3; h++) {
		for (i = 0; i < r; i++) {
			b = Lines[L[h] * r + i];
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
	cout << "there are " << size << " points on the three lines: ";
	lint_vec_print(cout, the_arc, size);
	cout << endl;


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

	cout << "projective_space::PG_2_8_create_conic_"
			"plus_nucleus_arc_1: after adding the rest of the "
			"conic, there are " << size << " points on the arc: ";
	lint_vec_print(cout, the_arc, size);
	cout << endl;
}

void projective_space::PG_2_8_create_conic_plus_nucleus_arc_2(
		long int *the_arc, int &size, int verbose_level)
{
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

	cout << "frame: ";
	int_vec_print(cout, frame, 4);
	cout << endl;
	
	L[0] = Line_through_two_points[frame[0] * N_points + frame[2]];
	L[1] = Line_through_two_points[frame[2] * N_points + frame[3]];
	L[2] = Line_through_two_points[frame[3] * N_points + frame[0]];
	
	cout << "l1=" << L[0] << " l2=" << L[1] << " l3=" << L[2] << endl;

	size = 0;	
	for (h = 0; h < 3; h++) {
		for (i = 0; i < r; i++) {
			b = Lines[L[h] * r + i];
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
	cout << "there are " << size << " points on the three lines: ";
	lint_vec_print(cout, the_arc, size);
	cout << endl;


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

	cout << "projective_space::PG_2_8_create_conic_plus_"
			"nucleus_arc_2: after adding the rest of the conic, "
			"there are " << size << " points on the arc: ";
	lint_vec_print(cout, the_arc, size);
	cout << endl;
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
		int_vec_print(cout, points, 22);
		cout << endl;
		}
	
	L[0] = Line_through_two_points[1 * N_points + 2];
	L[1] = Line_through_two_points[0 * N_points + 2];
	L[2] = Line_through_two_points[0 * N_points + 1];
	L[3] = Line_through_two_points[points[20] * N_points + points[21]];
	
	if (f_v) {
		cout << "L:";
		lint_vec_print(cout, L, 4);
		cout << endl;
		}

	for (h = 0; h < 4; h++) {
		cout << "h=" << h << " : L[h]=" << L[h] << " : " << endl;
		for (i = 0; i < r; i++) {
			b = Lines[L[h] * r + i];
			cout << "point " << b << " = ";
			unrank_point(v, b);
			F->PG_element_normalize_from_front(v, 1, 3);
			int_vec_print(cout, v, 3);
			cout << endl;
			}
		cout << endl;
		}
	size = 0;	
	for (h = 0; h < 4; h++) {
		for (i = 0; i < r; i++) {
			b = Lines[L[h] * r + i];
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
		lint_vec_print(cout, the_arc, size);
		cout << endl;
		}


	// remove the first 16 points:
	for (i = 0; i < 16; i++) {
		cout << "removing point " << i << " : "
				<< points[i] << endl;
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
		lint_vec_print(cout, the_arc, size);
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
		int_vec_print(cout, points, 25);
		cout << endl;
		}
	for (i = 0; i < 9; i++) {
		L[i] = Polarity_point_to_hyperplane[points[i]];
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
		lint_vec_print(cout, the_arc, size);
		cout << endl;
		}
}


void projective_space::create_pasch_arc(
	int *the_arc, int &size, int verbose_level)
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
		int_vec_print(cout, points, 5);
		cout << endl;
		}
	
	L[0] = Line_through_two_points[points[0] * N_points + points[1]];
	L[1] = Line_through_two_points[points[0] * N_points + points[3]];
	L[2] = Line_through_two_points[points[2] * N_points + points[3]];
	L[3] = Line_through_two_points[points[1] * N_points + points[4]];
	
	if (f_v) {
		cout << "L:";
		int_vec_print(cout, L, 4);
		cout << endl;
		}

	size = 0;	
	for (h = 0; h < 4; h++) {
		for (i = 0; i < r; i++) {
			b = Lines[L[h] * r + i];
			if (Sorting.int_vec_search(the_arc, size, b, idx)) {
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
		int_vec_print(cout, the_arc, size);
		cout << endl;
		}

	
	v[0] = 1;
	v[1] = 1;
	v[2] = 0;
	b = rank_point(v);
	if (Sorting.int_vec_search(the_arc, size, b, idx)) {
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
		int_vec_print(cout, the_arc, size);
		cout << endl;
		}

}

void projective_space::create_Cheon_arc(
	int *the_arc, int &size, int verbose_level)
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
		int_vec_print(cout, points, 5);
		cout << endl;
		}
	
	L[0] = Line_through_two_points[points[0] * N_points + points[1]];
	L[1] = Line_through_two_points[points[1] * N_points + points[2]];
	L[2] = Line_through_two_points[points[2] * N_points + points[0]];
	
	if (f_v) {
		cout << "L:";
		int_vec_print(cout, L, 3);
		cout << endl;
		}

	size = 0;	
	for (h = 0; h < 3; h++) {
		for (i = 0; i < r; i++) {
			b = Lines[L[h] * r + i];
			if (Sorting.int_vec_search(the_arc, size, b, idx)) {
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
		int_vec_print(cout, the_arc, size);
		cout << endl;
		}




	for (h = 0; h < 3; h++) {

		if (f_v) {
			cout << "h=" << h << endl;
			}

		for (i = 0; i < r; i++) {
			pencil[i] = Lines_on_point[points[h] * r + i];
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
		print_integer_matrix_width(cout, Pencil, 3, 7, 7, 4);
		}

	for (i = 0; i < 7; i++) {
		a = Pencil[0 * 7 + i];
		for (j = 0; j < 7; j++) {
			b = Pencil[1 * 7 + j];
			if (f_v) {
				cout << "i=" << i << " a=" << a << " j="
						<< j << " b=" << b << endl;
				}
			c = Line_intersection[a * N_lines + b];
			if (f_v) {
				cout << "c=" << c << endl;
				}
			if (Sorting.int_vec_search(the_arc, size, c, idx)) {
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
		int_vec_print(cout, the_arc, size);
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
		lint_vec_print(cout, the_arc, size);
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
		lint_vec_print(cout, the_arc, size);
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
		lint_vec_print(cout, the_arc, size);
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
	a.create(6);
	b.create(exponent);
	
	D.extended_gcd(a, b, g, u, u2, 0 /* verbose_level */);
	one_sixth = u.as_int();
	while (one_sixth < 0) {
		one_sixth += exponent;
		}
	if (f_v) {
		cout << "one_sixth = " << one_sixth << endl;
		}

	a.create(2);
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
		lint_vec_print(cout, the_arc, size);
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
		lint_vec_print(cout, the_arc, size);
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
		lint_vec_print(cout, the_arc, size);
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
	if (Lines_on_point == NULL) {
		line_intersection_type_basic(set, set_size, type, verbose_level);
		}
	else {
		for (i = 0; i < N_lines; i++) {
			type[i] = 0;
			}
		for (i = 0; i < set_size; i++) {
			a = set[i];
			for (j = 0; j < r; j++) {
				b = Lines_on_point[a * r + j];
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
		cout << "projective_space::line_intersection_type_through_"
				"hyperplane set_size=" << set_size << endl;
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
			int_vec_print(cout, M, d);
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
		cout << "projective_space::line_intersection_type_through_"
				"hyperplane sz1=" << sz1 << " sz2=" << sz2 << endl;
		}
	

	// do the line type in the hyperplane:
	line_intersection_type_basic(set1, sz1, type, verbose_level);
	nb_pts_in_hyperplane = Gg.nb_PG_elements(n - 1, q);
	if (f_vv) {
		cout << "projective_space::line_intersection_type_through_"
				"hyperplane nb_pts_in_hyperplane="
				<< nb_pts_in_hyperplane << endl;
		}

	cnt1 = NEW_int(nb_pts_in_hyperplane);
	Pts1 = NEW_int(nb_pts_in_hyperplane * d);
	Pts2 = NEW_int(sz2 * d);
	
	int_vec_zero(cnt1, nb_pts_in_hyperplane);
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
			cout << "projective_space::line_intersection_type_through_"
					"hyperplane checking lines through point " << i
					<< " / " << nb_pts_in_hyperplane << ":" << endl;
			}
		int_vec_zero(f_taken, sz2);
		for (j = 0; j < sz2; j++) {
			if (f_taken[j]) {
				continue;
				}
			if (f_vv) {
				cout << "projective_space::line_intersection_type_through_"
						"hyperplane j=" << j << " / " << sz2 << ":" << endl;
				}
			int_vec_copy(Pts1 + i * d, M, d);
			int_vec_copy(Pts2 + j * d, M + d, d);
			f_taken[j] = TRUE;
			if (f_vv) {
				int_matrix_print(M, 2, d);
				}
			rk = Grass_lines->rank_lint_here(M, 0 /* verbose_level */);
			if (f_vv) {
				cout << "projective_space::line_intersection_type_through_"
						"hyperplane line rk=" << rk << " cnt1="
						<< cnt1[rk] << ":" << endl;
				}
			cnt = 1 + cnt1[i];
			for (h = j + 1; h < sz2; h++) {
				int_vec_copy(M, M2, 2 * d);
				int_vec_copy(Pts2 + h * d, M2 + 2 * d, d);
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
		cout << "projective_space::line_intersection_type_through_"
				"hyperplane done" << endl;
		}
}

void projective_space::find_secant_lines(long int *set, int set_size,
	long int *lines, int &nb_lines, int max_lines, int verbose_level)
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
			//lines[nb_lines++] = rk;
			}
		}
	FREE_int(M);
	if (f_v) {
		cout << "projective_space::find_secant_lines done" << endl;
		}
}

void projective_space::find_lines_which_are_contained(
	long int *set, int set_size,
	long int *lines, int &nb_lines, int max_lines,
	int verbose_level)
// finds all lines which are completely contained in the set of points
// given in set[set_size].
// First, finds all lines in the set which lie
// in the hyperplane x_d = 0.
// Then finds all remaining lines.
// The lines are not arranged according to a double six.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
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
				"set_size=" << set_size << endl;
		}
	nb_lines = 0;
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
			int_vec_print(cout, M, d);
			cout << endl;
			}
		if (M[d - 1] == 0) {
			set1[sz1++] = set[i];
			}
		else {
			set2[sz2++] = set[i];
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
	find_secant_lines(set1, sz1,
			secants, nb_secants,
			n2,
			verbose_level);
	if (f_vv) {
		cout << "projective_space::find_lines_which_are_contained "
				"we found " << nb_lines
				<< " secants in the hyperplane" << endl;
		}

	// first we test the secants and
	// find those which are lines on the surface:

	nb_lines = 0;
	for (i = 0; i < nb_secants; i++) {
		rk = secants[i];
		Grass_lines->unrank_lint_here(M, rk, 0 /* verbose_level */);
		if (f_vv) {
			cout << "testing secant " << i << " / " << nb_secants
					<< " which is line " << rk << ":" << endl;
			int_matrix_print(M, 2, d);
			}

		int coeffs[2];

		// loop over all points on the line:
		for (a = 0; a < q + 1; a++) {

			// unrank a point on the projective line:
			F->PG_element_unrank_modified(coeffs, 1, 2, a);
			int_vec_copy(M, M2, 2 * d);

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
			lines[nb_lines++] = rk;
			}
		}
	FREE_lint(secants);

	if (f_v) {
		cout << "projective_space::find_lines_which_are_contained "
				"We found " << nb_lines << " in the hyperplane" << endl;
		lint_vec_print(cout, lines, nb_lines);
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

	f_taken = NEW_int(sz2);
	for (i = 0; i < sz1; i++) {
		if (f_vv) {
			cout << "projective_space::find_lines_which_are_contained "
					"checking lines through hyperplane point " << i
					<< " / " << sz1 << ":" << endl;
			}

		// consider a point P1 on the surface and in the hyperplane

		int_vec_zero(f_taken, sz2);
		for (j = 0; j < sz2; j++) {
			if (f_taken[j]) {
				continue;
				}
			if (f_vv) {
				cout << "projective_space::find_lines_which_are_contained "
						"i=" << i << " j=" << j << " / "
						<< sz2 << ":" << endl;
				}

			// consider a point P2 on the surface
			// but not in the hyperplane:

			int_vec_copy(Pts1 + i * d, M, d);
			int_vec_copy(Pts2 + j * d, M + d, d);

			f_taken[j] = TRUE;

			if (f_vv) {
				int_matrix_print(M, 2, d);
				}

			rk = Grass_lines->rank_lint_here(M, 0 /* verbose_level */);
			if (f_vv) {
				cout << "projective_space::find_lines_which_are_contained "
						"line rk=" << rk << ":" << endl;
				}

			// test the q-1 points on the line through the P1 and P2
			// (but excluding P1 and P2 themselves):
			for (a = 1; a < q; a++) {
				int_vec_copy(M, M2, 2 * d);

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
					if (f_vv) {
						cout << "eliminating point " << idx << endl;
						}
					// we don't need to consider this point for P2:
					f_taken[idx] = TRUE;
					}
				}
			if (a == q) {
				// The line P1P2 is contained in the surface.
				// Add it to lines[]
				if (nb_lines == max_lines) {
					cout << "projective_space::find_lines_which_are_"
							"contained nb_lines == max_lines" << endl;
					exit(1);
					}
				if (f_v) {
					cout << "adding line " << rk << " nb_lines="
							<< nb_lines << endl;
					}
				lines[nb_lines++] = rk;
				}
			}
		}
	FREE_int(M);
	FREE_int(M2);
	FREE_lint(set1);
	FREE_lint(set2);
	FREE_int(Pts1);
	FREE_int(Pts2);

	if (f_v) {
		cout << "projective_space::find_lines_which_are_"
				"contained done" << endl;
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

			int_vec_copy(M1, M2, 3 * d);
			//unrank_point(M2 + 3 * d, set[h]);
			int_vec_copy(Pts + h * d, M2 + 3 * d, d);

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
		cout << "projective_space::hyperplane_"
				"intersection_type_basic" << endl;
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

			int_vec_copy(G->M, M, (d - 1) * d);
			//unrank_point(M + (d - 1) * d, set[h]);
			int_vec_copy(Pts + h * d, M + (d - 1) * d, d);

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
		cout << "projective_space::line_intersection_"
				"type_collected" << endl;
		}
	d = n + 1;
	M = NEW_int(3 * d);
	Pts = NEW_int(set_size * d);
	int_vec_zero(type_collected, set_size + 1);

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

			int_vec_copy(Grass_lines->M, M, 2 * d);


			//unrank_point(M + 2 * d, set[h]);
			int_vec_copy(Pts + h * d, M + 2 * d, d);

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
			b = Lines[a * k + j];
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
			b = Lines[a * k + j];
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


void projective_space::Baer_subline(int *pts3,
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
		print_integer_matrix_width(cout,
				M, 3, len, len, F->log10_of_q);
		}
	rk = F->Gauss_simple(M,
			3, len, base_cols, verbose_level - 3);
	if (f_vv) {
		cout << "projective_space::Baer_subline" << endl;
		cout << "has rank " << rk << endl;
		cout << "base_cols=";
		int_vec_print(cout, base_cols, rk);
		cout << endl;
		cout << "basis:" << endl;
		print_integer_matrix_width(cout,
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
		print_integer_matrix_width(cout,
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
		print_integer_matrix_width(cout,
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
		print_integer_matrix_width(cout,
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
		print_integer_matrix_width(cout,
				Frame, 2, 3, 3, F->log10_of_q);
		cout << "projective_space::Baer_subline "
				"base_cols2=";
		int_vec_print(cout, base_cols2, rk2);
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
		print_integer_matrix_width(cout, N, 3, rk, rk, F->log10_of_q);
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
			int_vec_print(cout, z, len);
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
			int_vec_print(cout, w, 2);
			cout << endl;
			}
		F->mult_vector_from_the_left(w, Basis, z, 2, len);
		if (f_vvv) {
			cout << "z=w*Basis";
			int_vec_print(cout, z, len);
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
		lint_vec_print(cout, pts, nb_pts);
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







}}



