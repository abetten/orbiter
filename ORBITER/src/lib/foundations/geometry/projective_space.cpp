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
		FREE_int(Nb_subspaces);
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
	Nb_subspaces = NEW_int(n + 1);
	if (n < 10) {
		for (i = 0; i <= n; i++) {
			if (f_v) {
				cout << "projective_space::init computing number of "
						"subspaces of dimension " << i + 1 << endl;
				}
			D.q_binomial_no_table(
				a,
				n + 1, i + 1, q, verbose_level - 2);
			Nb_subspaces[i] = a.as_int();
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



	if (N_points * N_points < ONE_MILLION) {

		if (f_v) {
			cout << "projective_space::init_incidence_structure "
					"allocating Line_through_two_points" << endl;
			}
		Line_through_two_points = NEW_int(N_points * N_points);
		}
	else {
		Line_through_two_points = NULL;
		}

	if (N_lines * N_lines < ONE_MILLION) {
		if (f_v) {
			cout << "projective_space::init_incidence_structure "
					"allocating Line_intersection" << endl;
			}
		Line_intersection = NEW_int(N_lines * N_lines);
		int_vec_zero(Line_through_two_points, N_points * N_points);
		for (i = 0; i < N_lines * N_lines; i++) {
			Line_intersection[i] = -1;
			}
		}
	else {
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
			Grass_lines->unrank_int(i, 0/*verbose_level - 4*/);
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
			Grass_lines->unrank_int(i, 0 /*verbose_level - 4*/);
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

void projective_space::create_points_on_line(
	int line_rk, int *line, int verbose_level)
// needs line[k]
{
	int a, b;
	
	Grass_lines->unrank_int(line_rk, 0/*verbose_level - 4*/);
	for (a = 0; a < k; a++) {
		F->PG_element_unrank_modified(v, 1, 2, a);
		F->mult_matrix_matrix(v, Grass_lines->M, w, 1, 2, n + 1,
				0 /* verbose_level */);
		F->PG_element_rank_modified(w, 1, n + 1, b);
		line[a] = b;
		}
}

int projective_space::create_point_on_line(
	int line_rk, int pt_rk, int verbose_level)
// pt_rk is between 0 and q-1.
{
	int f_v = (verbose_level >= 1);
	int b;
	int v[2];

	if (f_v) {
		cout << "projective_space::create_point_on_line" << endl;
		}
	Grass_lines->unrank_int(line_rk, 0/*verbose_level - 4*/);
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
		Grass_lines->unrank_int(line, 0/*verbose_level - 4*/);

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
	nb_rows = nb_rk_k_subspaces_as_int(row_type);
	nb_cols = nb_rk_k_subspaces_as_int(col_type);


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
int projective_space::nb_rk_k_subspaces_as_int(int k)
{
	longinteger_domain D;
	longinteger_object aa;
	int N;
	int d = n + 1;

	D.q_binomial(aa, d, k, q, 0/*verbose_level*/);
	N = aa.as_int();
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

int projective_space::rank_point(int *v)
{
	int b;
	
	F->PG_element_rank_modified(v, 1, n + 1, b);
	return b;
}

void projective_space::unrank_point(int *v, int rk)
{	
	F->PG_element_unrank_modified(v, 1, n + 1, rk);
}

void projective_space::unrank_points(int *v, int *Rk, int sz)
{
	int i;

	for (i = 0; i < sz; i++) {
		F->PG_element_unrank_modified(v + i * (n + 1), 1, n + 1, Rk[i]);
	}
}

int projective_space::rank_line(int *basis)
{
	int b;
	
	b = Grass_lines->rank_int_here(basis, 0/*verbose_level - 4*/);
	return b;
}

void projective_space::unrank_line(int *basis, int rk)
{	
	Grass_lines->unrank_int_here(basis, rk, 0/*verbose_level - 4*/);
}

void projective_space::unrank_lines(int *v, int *Rk, int nb)
{
	int i;
	
	for (i = 0; i < nb; i++) {
		Grass_lines->unrank_int_here(
				v + i * 2 * (n + 1), Rk[i], 0 /* verbose_level */);
		}
}

int projective_space::rank_plane(int *basis)
{
	int b;

	if (Grass_planes == NULL) {
		cout << "projective_space::rank_plane "
				"Grass_planes == NULL" << endl;
		exit(1);
		}
	b = Grass_planes->rank_int_here(basis, 0/*verbose_level - 4*/);
	return b;
}

void projective_space::unrank_plane(int *basis, int rk)
{	
	if (Grass_planes == NULL) {
		cout << "projective_space::unrank_plane "
				"Grass_planes == NULL" << endl;
		exit(1);
		}
	Grass_planes->unrank_int_here(basis, rk, 0/*verbose_level - 4*/);
}

int projective_space::line_through_two_points(
		int p1, int p2)
{
	int b;
	
	unrank_point(Grass_lines->M, p1);
	unrank_point(Grass_lines->M + n + 1, p2);
	b = Grass_lines->rank_int(0/*verbose_level - 4*/);
	return b;
}

int projective_space::test_if_lines_are_disjoint(
		int l1, int l2)
{
	if (Lines) {
		return test_if_sets_are_disjoint(
				Lines + l1 * k, Lines + l2 * k, k, k);
		}
	else {
		return test_if_lines_are_disjoint_from_scratch(l1, l2);
		}
}

int projective_space::test_if_lines_are_disjoint_from_scratch(
		int l1, int l2)
{
	int *Mtx;
	int m, rk;

	m = n + 1;
	Mtx = NEW_int(4 * m);
	Grass_lines->unrank_int_here(Mtx, l1, 0/*verbose_level - 4*/);
	Grass_lines->unrank_int_here(Mtx + 2 * m, l2, 0/*verbose_level - 4*/);
	rk = F->Gauss_easy(Mtx, 4, m);
	FREE_int(Mtx);
	if (rk == 4) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}

int projective_space::intersection_of_two_lines(int l1, int l2)
// formerly intersection_of_two_lines_in_a_plane
{
	int *Mtx1;
	int *Mtx3;
	int b, r;
	
#if 0
	int i;
	
	if (n != 2) {
		cout << "projective_space::intersection_of_two_lines_in_a_plane n != 2" << endl;
		exit(1);
		}
	Mtx1 = NEW_int(3 * 3);
	Mtx3 = NEW_int(3 * 3);
	
	Grass_lines->unrank_int(l1, 0/*verbose_level - 4*/);
	for (i = 0; i < 2 * 3; i++) {
		Mtx1[i] = Grass_lines->M[i];
		}
	F->perp_standard(3, 2, Mtx1, 0);
	for (i = 0; i < 3; i++) {
		Mtx3[i] = Mtx1[6 + i];
		}
	
	Grass_lines->unrank_int(l2, 0/*verbose_level - 4*/);
	for (i = 0; i < 2 * 3; i++) {
		Mtx1[i] = Grass_lines->M[i];
		}
	F->perp_standard(3, 2, Mtx1, 0);
	for (i = 0; i < 3; i++) {
		Mtx3[3 + i] = Mtx1[6 + i];
		}
	F->perp_standard(3, 2, Mtx3, 0);
	b = rank_point(Mtx3 + 6);

	FREE_int(Mtx1);
	FREE_int(Mtx3);
#else
	int d = n + 1;
	int D = 2 * d;

	
	Mtx1 = NEW_int(d * d);
	Mtx3 = NEW_int(D * d);

	Grass_lines->unrank_int_here(Mtx1, l1, 0/*verbose_level - 4*/);
	F->perp_standard(d, 2, Mtx1, 0);
	int_vec_copy(Mtx1 + 2 * d, Mtx3, (d - 2) * d);
	Grass_lines->unrank_int_here(Mtx1, l2, 0/*verbose_level - 4*/);
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
	
#endif
	
	return b;
}

int projective_space::arc_test(int *input_pts, int nb_pts, 
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
	int *two_input_pts,
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
	int *input_pts, int nb_pts, 
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
	int *nine_pts_or_more,
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
	int *points, int &nb_points, int verbose_level)
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
		int_vec_print(cout, points, nb_points);
		cout << endl;
		}
}

void projective_space::quadric_points_brute_force(
	int *ten_coeffs,
	int *points, int &nb_points, int verbose_level)
// quadric in PG(3,q)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[3];
	int i, a;

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
		int_vec_print(cout, points, nb_points);
		cout << endl;
		}
}

void projective_space::conic_points(
	int *five_pts, int *six_coeffs,
	int *points, int &nb_points, int verbose_level)
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
		int_vec_print(cout, points, nb_points);
		cout << endl;
		}
	
}

void projective_space::find_tangent_lines_to_conic(
	int *six_coeffs,
	int *points, int nb_points, 
	int *tangents, int verbose_level)
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
	int *arc6,
	int *&bisecants, int *&conics, int verbose_level)
// bisecants[15 * 3]
// conics[6 * 6]
{
	int f_v = (verbose_level >= 1);
	int i, j, h, pi, pj, Line[2];
	int arc5[5];
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
		int_vec_copy(arc6, arc5, j);
		int_vec_copy(arc6 + j + 1, arc5 + j, 5 - j);

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
	int *arc6, 
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
		int_vec_print(cout, arc6, 6);
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
	int *the_arc, int &size, int verbose_level)
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
	cout << "there are " << size << " points on the three lines: ";
	int_vec_print(cout, the_arc, size);
	cout << endl;


	for (i = 1; i < q; i++) {
		v[0] = 1;
		v[1] = i;
		v[2] = F->mult(i, i);
		b = rank_point(v);
		if (Sorting.int_vec_search(the_arc, size, b, idx)) {
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
	int_vec_print(cout, the_arc, size);
	cout << endl;
}

void projective_space::PG_2_8_create_conic_plus_nucleus_arc_2(
	int *the_arc, int &size, int verbose_level)
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
	cout << "there are " << size << " points on the three lines: ";
	int_vec_print(cout, the_arc, size);
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
		if (Sorting.int_vec_search(the_arc, size, b, idx)) {
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
	int_vec_print(cout, the_arc, size);
	cout << endl;
}

void projective_space::create_Maruta_Hamada_arc(
	int *the_arc, int &size, int verbose_level)
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
	int L[4];
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
		int_vec_print(cout, L, 4);
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
		cout << "there are " << size
				<< " points on the quadrilateral: ";
		int_vec_print(cout, the_arc, size);
		cout << endl;
		}


	// remove the first 16 points:
	for (i = 0; i < 16; i++) {
		cout << "removing point " << i << " : "
				<< points[i] << endl;
		if (!Sorting.int_vec_search(the_arc, size, points[i], idx)) {
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
		if (Sorting.int_vec_search(the_arc, size, points[i], idx)) {
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
		int_vec_print(cout, the_arc, size);
		cout << endl;
		}

}

void projective_space::create_Maruta_Hamada_arc2(
	int *the_arc, int &size, int verbose_level)
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
		int_vec_print(cout, the_arc, size);
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
	int *the_arc, int &size, int verbose_level)
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
		int_vec_print(cout, the_arc, size);
		cout << endl;
		}
}

void projective_space::create_translation_hyperoval(
	int *the_arc, int &size, 
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
		int_vec_print(cout, the_arc, size);
		cout << endl;
		}
	if (f_v) {
		cout << "projective_space::create_translation_hyperoval "
				"done" << endl;
		}
}

void projective_space::create_Segre_hyperoval(
	int *the_arc, int &size, int verbose_level)
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
		int_vec_print(cout, the_arc, size);
		cout << endl;
		}
}

void projective_space::create_Payne_hyperoval(
	int *the_arc, int &size, int verbose_level)
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
		int_vec_print(cout, the_arc, size);
		cout << endl;
		}
}

void projective_space::create_Cherowitzo_hyperoval(
	int *the_arc, int &size, int verbose_level)
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
		int_vec_print(cout, the_arc, size);
		cout << endl;
		}
}

void projective_space::create_OKeefe_Penttila_hyperoval_32(
	int *the_arc, int &size, int verbose_level)
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
		int_vec_print(cout, the_arc, size);
		cout << endl;
		}
}




void projective_space::line_intersection_type(
	int *set, int set_size, int *type, int verbose_level)
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
	int *set, int set_size, int *type, int verbose_level)
// type[N_lines]
{
	int f_v = (verbose_level >= 1);
	int rk, h, i, j, d;
	int *M;

	if (f_v) {
		cout << "projective_space::line_intersection_type_basic" << endl;
		}
	d = n + 1;
	M = NEW_int(3 * d);
	for (rk = 0; rk < N_lines; rk++) {
		type[rk] = 0;
		Grass_lines->unrank_int(rk, 0 /* verbose_level */);
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
	int *set, int set_size, int *type, int verbose_level)
// type[N_lines]
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;
	int rk, h, i, j, d, cnt, i1;
	int *M;
	int *M2;
	int *Pts1;
	int *Pts2;
	int *set1;
	int *set2;
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
	set1 = NEW_int(set_size);
	set2 = NEW_int(set_size);
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

	Sorting.int_vec_heapsort(set1, sz1);
	
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
		F->PG_element_rank_modified(Pts1 + i * d, 1, n + 1, i1);

		// i1 is the rank of the hyperplane point
		// inside the larger space:
		//unrank_point(Pts1 + i * d, set1[i]);
		if (Sorting.int_vec_search(set1, sz1, i1, idx)) {
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
			rk = Grass_lines->rank_int_here(M, 0 /* verbose_level */);
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
	FREE_int(set1);
	FREE_int(set2);
	FREE_int(Pts1);
	FREE_int(Pts2);
	FREE_int(cnt1);

	if (f_v) {
		cout << "projective_space::line_intersection_type_through_"
				"hyperplane done" << endl;
		}
}

void projective_space::find_secant_lines(int *set, int set_size, 
	int *lines, int &nb_lines, int max_lines, int verbose_level)
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
			rk = Grass_lines->rank_int_here(M, 0 /* verbose_level */);

			if (!Sorting.int_vec_search(lines, nb_lines, rk, idx)) {
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
	int *set, int set_size, 
	int *lines, int &nb_lines, int max_lines, 
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
	int rk, h, i, j, d, a, b, idx;
	int *M;
	int *M2;
	int *Pts1;
	int *Pts2;
	int *set1;
	int *set2;
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
	set1 = NEW_int(set_size);
	set2 = NEW_int(set_size);
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
	Sorting.int_vec_heapsort(set1, sz1);
	Sorting.int_vec_heapsort(set2, sz2);
	
	if (f_vv) {
		cout << "projective_space::find_lines_which_are_contained "
				"sz1=" << sz1 << " sz2=" << sz2 << endl;
		}
	

	// find all secants in the hyperplane:
	int *secants;
	int n2, nb_secants;

	n2 = (sz1 * (sz1 - 1)) >> 1;
	// n2 is an upper bound on the number of secant lines

	secants = NEW_int(n2);
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
		Grass_lines->unrank_int_here(M, rk, 0 /* verbose_level */);
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
			F->PG_element_rank_modified(M2 + 2 * d, 1, d, b);
			if (!Sorting.int_vec_search(set1, sz1, b, idx)) {
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
	FREE_int(secants);

	if (f_v) {
		cout << "projective_space::find_lines_which_are_contained "
				"We found " << nb_lines << " in the hyperplane" << endl;
		int_vec_print(cout, lines, nb_lines);
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

			rk = Grass_lines->rank_int_here(M, 0 /* verbose_level */);
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
				F->PG_element_rank_modified(M2 + 2 * d, 1, d, b);
				if (!Sorting.int_vec_search(set2, sz2, b, idx)) {
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
	FREE_int(set1);
	FREE_int(set2);
	FREE_int(Pts1);
	FREE_int(Pts2);

	if (f_v) {
		cout << "projective_space::find_lines_which_are_"
				"contained done" << endl;
		}
}


void projective_space::plane_intersection_type_basic(
	int *set, int set_size, 
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

	N_planes = nb_rk_k_subspaces_as_int(3);
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
		G->unrank_int_here(M1, rk, 0 /* verbose_level */);

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
	int *set, int set_size, int *type, 
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

	N_hyperplanes = nb_rk_k_subspaces_as_int(d - 1);
	
	// unrank all point here so we don't
	// have to do it again in the loop
	for (h = 0; h < set_size; h++) {
		unrank_point(Pts + h * d, set[h]);
	}

	for (rk = 0; rk < N_hyperplanes; rk++) {
		type[rk] = 0;
		G->unrank_int(rk, 0 /* verbose_level */);

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
	int *set, int set_size, int *type_collected, 
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
		Grass_lines->unrank_int(rk, 0 /* verbose_level */);
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

void projective_space::point_types(
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
	int *set, int set_size,
	int *external_lines, int &nb_external_lines,
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
	int *set, int set_size,
	int *tangent_lines, int &nb_tangent_lines,
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
	int *set, int set_size,
	int *secant_lines, int &nb_secant_lines,
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
	int *set, int set_size, int k,
	int *secant_lines, int &nb_secant_lines,
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
	int *&pts, int &nb_pts, int verbose_level)
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
	pts = NEW_int(nb_pts);


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
		int_vec_print(cout, pts, nb_pts);
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




void projective_space::print_set_numerical(int *set, int set_size)
{
	int i, a;
	int *v;
	
	v = NEW_int(n + 1);
	for (i = 0; i < set_size; i++) {
		a = set[i];
		unrank_point(v, a);
		cout << setw(3) << i << " : " << setw(5) << a << " : ";
		int_vec_print(cout, v, n + 1);
		cout << "=";
		F->PG_element_normalize_from_front(v, 1, n + 1);
		int_vec_print(cout, v, n + 1);
		cout << endl;
		}
	FREE_int(v);
}

void projective_space::print_set(
		int *set, int set_size)
{
	int i, a;
	int *v;
	
	v = NEW_int(n + 1);
	for (i = 0; i < set_size; i++) {
		a = set[i];
		unrank_point(v, a);
		cout << setw(3) << i << " : " << setw(5) << a << " : ";
		F->int_vec_print(cout, v, n + 1);
		cout << "=";
		F->PG_element_normalize_from_front(v, 1, n + 1);
		F->int_vec_print(cout, v, n + 1);
		cout << endl;
		}
	FREE_int(v);
}

void projective_space::print_line_set_numerical(
		int *set, int set_size)
{
	int i, a;
	int *v;
	
	v = NEW_int(2 * (n + 1));
	for (i = 0; i < set_size; i++) {
		a = set[i];
		unrank_line(v, a);
		cout << setw(3) << i << " : " << setw(5) << a << " : ";
		int_vec_print(cout, v, 2 * (n + 1));
		cout << endl;
		}
	FREE_int(v);
}


int projective_space::is_contained_in_Baer_subline(
	int *pts, int nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *subline;
	int sz;
	int i, idx, a;
	int ret = TRUE;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::is_contained_in_Baer_subline "
				"pts=" << endl;
		int_vec_print(cout, pts, nb_pts);
		cout << endl;
		cout << "computing Baer subline determined by the "
				"first three points:" << endl;
		}
	Baer_subline(pts, subline, sz, verbose_level - 2);
	if (f_vv) {
		cout << "projective_space::is_contained_in_Baer_subline "
				"The Baer subline is:" << endl;
		int_vec_print(cout, subline, sz);
		cout << endl;
		}
	Sorting.int_vec_heapsort(subline, sz);
	for (i = 0; i < nb_pts; i++) {
		a = pts[i];
		if (!Sorting.int_vec_search(subline, sz, a, idx)) {
			ret = FALSE;
			if (f_vv) {
				cout << "did not find " << i << "-th point " << a
						<< " in the list of points, hence not "
						"contained in Baer subline" << endl;
				}
			goto done;
			}
		
		}
done:
	FREE_int(subline);
	
	return ret;
}

int projective_space::determine_hermitian_form_in_plane(
	int *pts, int nb_pts, int *six_coeffs, int verbose_level)
// there is a memory problem in this function
// detected 7/14/11
// solved June 17, 2012: 
// coords and system were not freed
// system was allocated too short
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *coords; //[nb_pts * 3];
	int *system; //[nb_pts * 9];
	int kernel[9 * 9];
	int base_cols[9];
	int i, x, y, z, xq, yq, zq, rk;
	int Q, q, little_e;
	int kernel_m, kernel_n;
	number_theory_domain NT;

	if (f_v) {
		cout << "projective_space::determine_hermitian_"
				"form_in_plane" << endl;
		}
	coords = NEW_int(nb_pts * 3);
	system = NEW_int(nb_pts * 9);
	Q = F->q;
	if (ODD(F->e)) {
		cout << "projective_space::determine_hermitian_"
				"form_in_plane field degree must be even" << endl;
		exit(1);
		}
	little_e = F->e >> 1;
	q = NT.i_power_j(F->p, little_e);
	if (f_v) {
		cout << "projective_space::determine_hermitian_"
				"form_in_plane Q=" << Q << " q=" << q << endl;
		}
	if (n != 2) {
		cout << "projective_space::determine_hermitian_"
				"form_in_plane n != 2" << endl;
		exit(1);
		}
	for (i = 0; i < nb_pts; i++) {
		unrank_point(coords + i * 3, pts[i]);
		}
	if (f_vv) {
		cout << "projective_space::determine_hermitian_"
				"form_in_plane points:" << endl;
		print_integer_matrix_width(cout,
				coords, nb_pts, 3, 3, F->log10_of_q);
		}
	for (i = 0; i < nb_pts; i++) {
		x = coords[i * 3 + 0];
		y = coords[i * 3 + 1];
		z = coords[i * 3 + 2];
		xq = F->frobenius_power(x, little_e);
		yq = F->frobenius_power(y, little_e);
		zq = F->frobenius_power(z, little_e);
		system[i * 9 + 0] = F->mult(x, xq);
		system[i * 9 + 1] = F->mult(y, yq);
		system[i * 9 + 2] = F->mult(z, zq);
		system[i * 9 + 3] = F->mult(x, yq);
		system[i * 9 + 4] = F->mult(y, xq);
		system[i * 9 + 5] = F->mult(x, zq);
		system[i * 9 + 6] = F->mult(z, xq);
		system[i * 9 + 7] = F->mult(y, zq);
		system[i * 9 + 8] = F->mult(z, yq);
		}
	if (f_v) {
		cout << "projective_space::determine_hermitian_"
				"form_in_plane system:" << endl;
		print_integer_matrix_width(cout,
				system, nb_pts, 9, 9, F->log10_of_q);
		}



	rk = F->Gauss_simple(system,
			nb_pts, 9, base_cols, verbose_level - 2);
	if (f_v) {
		cout << "projective_space::determine_hermitian_form_"
				"in_plane rk=" << rk << endl;
		print_integer_matrix_width(cout,
				system, rk, 9, 9, F->log10_of_q);
		}
#if 0
	if (rk != 8) {
		if (f_v) {
			cout << "projective_space::determine_hermitian_form_"
					"in_plane system underdetermined" << endl;
			}
		return FALSE;
		}
#endif
	F->matrix_get_kernel(system, MINIMUM(nb_pts, 9), 9, base_cols, rk, 
		kernel_m, kernel_n, kernel);
	if (f_v) {
		cout << "projective_space::determine_hermitian_form_"
				"in_plane kernel:" << endl;
		print_integer_matrix_width(cout, kernel,
				kernel_m, kernel_n, kernel_n, F->log10_of_q);
		}
	six_coeffs[0] = kernel[0 * kernel_n + 0];
	six_coeffs[1] = kernel[1 * kernel_n + 0];
	six_coeffs[2] = kernel[2 * kernel_n + 0];
	six_coeffs[3] = kernel[3 * kernel_n + 0];
	six_coeffs[4] = kernel[5 * kernel_n + 0];
	six_coeffs[5] = kernel[7 * kernel_n + 0];
	if (f_v) {
		cout << "projective_space::determine_hermitian_form_"
				"in_plane six_coeffs:" << endl;
		int_vec_print(cout, six_coeffs, 6);
		cout << endl;
		}
	FREE_int(coords);
	FREE_int(system);
	return TRUE;
}

void projective_space::circle_type_of_line_subset(
	int *pts, int nb_pts, int *circle_type, 
	int verbose_level)
// circle_type[nb_pts]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *subline;
	int subset[3];
	int idx_set[3];
	int sz;
	int i, idx, a, b;
	combinatorics_domain Combi;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::circle_type_of_line_subset "
				"pts=" << endl;
		int_vec_print(cout, pts, nb_pts);
		cout << endl;
		//cout << "computing Baer subline determined by
		//the first three points:" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		circle_type[i] = 0;
		}
	
	Combi.first_k_subset(idx_set, nb_pts, 3);
	do {
		for (i = 0; i < 3; i++) {
			subset[i] = pts[idx_set[i]];
			}
		Baer_subline(subset, subline, sz, verbose_level - 2);
		b = 0;
		Sorting.int_vec_heapsort(subline, sz);
		for (i = 0; i < nb_pts; i++) {
			a = pts[i];
			if (Sorting.int_vec_search(subline, sz, a, idx)) {
				b++;
				}
			}


		if (f_v) {
			cout << "projective_space::circle_type_of_line_subset "
					"The Baer subline determined by " << endl;
			int_vec_print(cout, subset, 3);
			cout << " is ";
			int_vec_print(cout, subline, sz);
			cout << " which intersects in " << b << " points" << endl;
			}



		FREE_int(subline);
		circle_type[b]++;
		} while (Combi.next_k_subset(idx_set, nb_pts, 3));

	if (f_vv) {
		cout << "projective_space::circle_type_of_line_subset "
				"circle_type before fixing =" << endl;
		int_vec_print(cout, circle_type, nb_pts);
		cout << endl;
		}
	for (i = 4; i < nb_pts; i++) {
		a = Combi.int_n_choose_k(i, 3);
		if (circle_type[i] % a) {
			cout << "projective_space::circle_type_of_line_subset "
					"circle_type[i] % a" << endl;
			exit(1);
			}
		circle_type[i] /= a;
		}
	if (f_vv) {
		cout << "projective_space::circle_type_of_line_subset "
				"circle_type after fixing =" << endl;
		int_vec_print(cout, circle_type, nb_pts);
		cout << endl;
		}
}

void projective_space::create_unital_XXq_YZq_ZYq(
	int *U, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	//finite_field *FQ;
	int *v;
	int e, i, a;
	int X, Y, Z, Xq, Yq, Zq;

	if (f_v) {
		cout << "projective_space::create_unital_XXq_YZq_ZYq" << endl;
		}
	if (n != 2) {
		cout << "projective_space::create_unital_XXq_YZq_ZYq "
				"n != 2" << endl;
		exit(1);
 		}
	if (ODD(F->e)) {
		cout << "projective_space::create_unital_XXq_YZq_ZYq "
				"ODD(F->e)" << endl;
		exit(1);
 		}
	//FQ = F;
	
	v = NEW_int(3);
	e = F->e >> 1;
	if (f_vv) {
		cout << "e=" << e << endl;
		}
	sz = 0;
	for (i = 0; i < N_points; i++) {
		unrank_point(v, i);
		if (f_vvv) {
			cout << "i=" << i << " : ";
			int_vec_print(cout, v, 3);
			//cout << endl;
			}
		X = v[0];
		Y = v[1];
		Z = v[2];
		Xq = F->frobenius_power(X, e);
		Yq = F->frobenius_power(Y, e);
		Zq = F->frobenius_power(Z, e);
		a = F->add3(F->mult(X, Xq), F->mult(Y, Zq), F->mult(Z, Yq));
		if (f_vvv) {
			cout << " a=" << a << endl;
			}
		if (a == 0) {
			//cout << "a=0, adding i=" << i << endl;
			U[sz++] = i;
			//int_vec_print(cout, U, sz);
			//cout << endl;
			}
		}
	if (f_vv) {
		cout << "we found " << sz << " points:" << endl;	
		int_vec_print(cout, U, sz);
		cout << endl;
		print_set(U, sz);
		}
	FREE_int(v);

	if (f_v) {
		cout << "projective_space::create_unital_XXq_YZq_ZYq "
				"done" << endl;
		}
}


void projective_space::intersection_of_subspace_with_point_set(
	grassmann *G, int rk, int *set, int set_size, 
	int *&intersection_set, int &intersection_set_size, 
	int verbose_level)
{
	int h;
	int d = n + 1;
	int k = G->k;
	int *M;

	intersection_set = NEW_int(set_size);
	M = NEW_int((k + 1) * d);
	intersection_set_size = 0;

	G->unrank_int(rk, 0 /* verbose_level */);

	for (h = 0; h < set_size; h++) {
		int_vec_copy(G->M, M, k * d);
#if 0
		for (i = 0; i < k; i++) {
			for (j = 0; j < d; j++) {
				M[i * d + j] = G->M[i * d + j];
				}
			}
#endif
		unrank_point(M + k * d, set[h]);
		if (F->rank_of_rectangular_matrix(M,
				k + 1, d, 0 /*verbose_level*/) == k) {
			intersection_set[intersection_set_size++] = set[h];
			}
		} // next h

	FREE_int(M);
}

void projective_space::intersection_of_subspace_with_point_set_rank_is_longinteger(
	grassmann *G, longinteger_object &rk,
	int *set, int set_size,
	int *&intersection_set, int &intersection_set_size, 
	int verbose_level)
{
	int h;
	int d = n + 1;
	int k = G->k;
	int *M;

	intersection_set = NEW_int(set_size);
	M = NEW_int((k + 1) * d);
	intersection_set_size = 0;

	G->unrank_longinteger(rk, 0 /* verbose_level */);

	for (h = 0; h < set_size; h++) {
		int_vec_copy(G->M, M, k * d);
		unrank_point(M + k * d, set[h]);
		if (F->rank_of_rectangular_matrix(M,
				k + 1, d, 0 /*verbose_level*/) == k) {
			intersection_set[intersection_set_size++] = set[h];
			}
		} // next h

	FREE_int(M);
}

void projective_space::plane_intersection_invariant(
	grassmann *G, 
	int *set, int set_size, 
	int *&intersection_type, int &highest_intersection_number, 
	int *&intersection_matrix, int &nb_planes, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object *R;
	int **Pts_on_plane;
	int *nb_pts_on_plane;
	int nb_planes_total;
	int i, j, a, u, f, l, ii;

	if (f_v) {
		cout << "projective_space::plane_intersection_invariant" << endl;
		}
	plane_intersection_type_fast(G, 
		set, set_size, 
		R, Pts_on_plane, nb_pts_on_plane, nb_planes_total, 
		verbose_level - 1);

	classify C;
	int f_second = FALSE;

	C.init(nb_pts_on_plane, nb_planes_total, f_second, 0);
	if (f_v) {
		cout << "projective_space::plane_intersection_invariant "
				"plane-intersection type: ";
		C.print(FALSE /* f_backwards*/);
		}

	if (f_v) {
		cout << "The plane intersection type is (";
		C.print_naked(FALSE /*f_backwards*/);
		cout << ")" << endl << endl;
		}
	f = C.type_first[C.nb_types - 1];
	highest_intersection_number = C.data_sorted[f];
	intersection_type = NEW_int(highest_intersection_number + 1);

	int_vec_zero(intersection_type, highest_intersection_number + 1);
	
	for (i = 0; i < C.nb_types; i++) {
		f = C.type_first[i];
		l = C.type_len[i];
		a = C.data_sorted[f];
		intersection_type[a] = l;
		}
	f = C.type_first[C.nb_types - 1];
	nb_planes = C.type_len[C.nb_types - 1];

	int *Incma, *Incma_t, *IIt, *ItI;
	
	Incma = NEW_int(set_size * nb_planes);
	Incma_t = NEW_int(nb_planes * set_size);
	IIt = NEW_int(set_size * set_size);
	ItI = NEW_int(nb_planes * nb_planes);


	for (i = 0; i < set_size * nb_planes; i++) {
		Incma[i] = 0;
		}
	for (i = 0; i < nb_planes; i++) {
		ii = C.sorting_perm_inv[f + i];
		for (j = 0; j < nb_pts_on_plane[ii]; j++) {
			a = Pts_on_plane[ii][j];
			Incma[a * nb_planes + i] = 1;
			}
		}
	if (f_vv) {
		cout << "Incidence matrix:" << endl;
		print_integer_matrix_width(cout,
				Incma, set_size, nb_planes, nb_planes, 1);
		}
	for (i = 0; i < set_size; i++) {
		for (j = 0; j < set_size; j++) {
			a = 0;
			for (u = 0; u < nb_planes; u++) {
				a += Incma[i * nb_planes + u] *
						Incma_t[u * set_size + j];
				}
			IIt[i * set_size + j] = a;
			}
		}
	if (f_vv) {
		cout << "I * I^\\top = " << endl;
		print_integer_matrix_width(cout,
				IIt, set_size, set_size, set_size, 2);
		}
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < nb_planes; j++) {
			a = 0;
			for (u = 0; u < set_size; u++) {
				a += Incma[u * nb_planes + i] *
						Incma[u * nb_planes + j];
				}
			ItI[i * nb_planes + j] = a;
			}
		}
	if (f_v) {
		cout << "I^\\top * I = " << endl;
		print_integer_matrix_width(cout,
				ItI, nb_planes, nb_planes, nb_planes, 3);
		}
	
	intersection_matrix = NEW_int(nb_planes * nb_planes);
	int_vec_copy(ItI,
			intersection_matrix, nb_planes * nb_planes);

	FREE_int(Incma);
	FREE_int(Incma_t);
	FREE_int(IIt);
	FREE_int(ItI);


	for (i = 0; i < nb_planes_total; i++) {
		FREE_int(Pts_on_plane[i]);
		}
	FREE_pint(Pts_on_plane);
	FREE_int(nb_pts_on_plane);
	FREE_OBJECTS(R);
}

void projective_space::plane_intersection_type(
	grassmann *G, 
	int *set, int set_size, 
	int *&intersection_type, int &highest_intersection_number, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object *R;
	int **Pts_on_plane;
	int *nb_pts_on_plane;
	int nb_planes;
	int i, f, l, a;

	if (f_v) {
		cout << "projective_space::plane_intersection_type" << endl;
		}
	plane_intersection_type_fast(G, 
		set, set_size, 
		R, Pts_on_plane, nb_pts_on_plane, nb_planes, 
		verbose_level - 1);

	classify C;
	int f_second = FALSE;

	C.init(nb_pts_on_plane, nb_planes, f_second, 0);
	if (f_v) {
		cout << "projective_space::plane_intersection_type "
				"plane-intersection type: ";
		C.print(FALSE /*f_backwards*/);
		}

	if (f_v) {
		cout << "The plane intersection type is (";
		C.print_naked(FALSE /*f_backwards*/);
		cout << ")" << endl << endl;
		}

	f = C.type_first[C.nb_types - 1];
	highest_intersection_number = C.data_sorted[f];
	intersection_type = NEW_int(highest_intersection_number + 1);
	int_vec_zero(intersection_type, highest_intersection_number + 1);
	for (i = 0; i < C.nb_types; i++) {
		f = C.type_first[i];
		l = C.type_len[i];
		a = C.data_sorted[f];
		intersection_type[a] = l;
		}

	for (i = 0; i < nb_planes; i++) {
		FREE_int(Pts_on_plane[i]);
		}
	FREE_pint(Pts_on_plane);
	FREE_int(nb_pts_on_plane);
	FREE_OBJECTS(R);

}

void projective_space::plane_intersections(
	grassmann *G, 
	int *set, int set_size, 
	longinteger_object *&R, set_of_sets &SoS, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int **Pts_on_plane;
	int *nb_pts_on_plane;
	int nb_planes;
	int i;
	
	if (f_v) {
		cout << "projective_space::plane_intersections" << endl;
		}
	plane_intersection_type_fast(G, 
		set, set_size, 
		R, Pts_on_plane, nb_pts_on_plane, nb_planes, 
		verbose_level - 1);
	if (f_v) {
		cout << "projective_space::plane_intersections "
				"before Sos.init" << endl;
		}
	SoS.init(set_size, nb_planes,
			Pts_on_plane, nb_pts_on_plane, verbose_level - 1);
	if (f_v) {
		cout << "projective_space::plane_intersections "
				"after Sos.init" << endl;
		}
	for (i = 0; i < nb_planes; i++) {
		FREE_int(Pts_on_plane[i]);
		}
	FREE_pint(Pts_on_plane);
	FREE_int(nb_pts_on_plane);
	if (f_v) {
		cout << "projective_space::plane_intersections done" << endl;
		}
}

void projective_space::plane_intersection_type_slow(
	grassmann *G, 
	int *set, int set_size, 
	longinteger_object *&R, 
	int **&Pts_on_plane, int *&nb_pts_on_plane, int &len, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v3 = (verbose_level >= 3);
	int r, rk, i, u, d, N_planes, l;

	int *Basis;
	int *Basis_save;
	int *Coords;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::plane_intersection_type_slow" << endl;
		}
	if (f_vv) {
		print_set_numerical(set, set_size);
		}
	if (!Sorting.test_if_set_with_return_value(set, set_size)) {
		cout << "projective_space::plane_intersection_type_slow "
				"the input set if not a set" << endl;
		exit(1);
		}
	d = n + 1;
	N_planes = nb_rk_k_subspaces_as_int(3);

	if (f_v) {
		cout << "N_planes=" << N_planes << endl;
		}
	// allocate data that is returned:
	R = NEW_OBJECTS(longinteger_object, N_planes);
	Pts_on_plane = NEW_pint(N_planes);
	nb_pts_on_plane = NEW_int(N_planes);

	// allocate temporary data:
	Basis = NEW_int(4 * d);
	Basis_save = NEW_int(4 * d);
	Coords = NEW_int(set_size * d);
	
	for (i = 0; i < set_size; i++) {
		unrank_point(Coords + i * d, set[i]);
		}
	if (f_vv) {
		cout << "projective_space::plane_intersection_type_slow "
				"Coords:" << endl;
		int_matrix_print(Coords, set_size, d);
		}

	l = 0;
	for (rk = 0; rk < N_planes; rk++) {

		if (N_planes > 1000000) {
			if ((rk % 250000) == 0) {
				cout << "projective_space::plane_intersection_type_slow "
						<< rk << " / " << N_planes << endl;
				}
			}
		G->unrank_int_here(Basis_save, rk, 0 /* verbose_level */);
		//int_vec_copy(G->M, Basis_save, 3 * d);
		int *pts_on_plane;
		int nb = 0;
	
		pts_on_plane = NEW_int(set_size);
			
		for (u = 0; u < set_size; u++) {
			int_vec_copy(Basis_save, Basis, 3 * d);
			int_vec_copy(Coords + u * d, Basis + 3 * d, d);
			r = F->rank_of_rectangular_matrix(Basis,
					4, d, 0 /* verbose_level */);
			if (r < 4) {
				pts_on_plane[nb++] = u;
				}
			}

		
		Pts_on_plane[l] = pts_on_plane;
		nb_pts_on_plane[l] = nb;
		R[l].create(rk);
		l++;
		} // rk
	len = l;
	
	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(Coords);
	if (f_v) {
		cout << "projective_space::plane_intersection_type_slow "
				"done" << endl;
		}
}

void projective_space::plane_intersection_type_fast(
	grassmann *G, 
	int *set, int set_size, 
	longinteger_object *&R, 
	int **&Pts_on_plane, int *&nb_pts_on_plane, int &len, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int r, rk, rr, h, i, j, a, d, N_planes, N, N2, idx, l;

	int *Basis;
	int *Basis_save;
	int *f_subset_done;
	int *rank_idx;
	int *Coords;

	int subset[3];
	int subset2[3];
	int subset3[3];
	longinteger_object plane_rk, aa;
	int *pts_on_plane;
	combinatorics_domain Combi;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::plane_intersection_type_fast" << endl;
		}
	if (f_vv) {
		print_set_numerical(set, set_size);
		}

	if (!Sorting.test_if_set_with_return_value(set, set_size)) {
		cout << "projective_space::plane_intersection_type_fast "
				"the input set if not a set" << endl;
		exit(1);
		}
	d = n + 1;
	N_planes = nb_rk_k_subspaces_as_int(3);
	N = Combi.int_n_choose_k(set_size, 3);

	if (f_v) {
		cout << "N_planes=" << N_planes << endl;
		cout << "N=number of 3-subsets of the set=" << N << endl;
		}
	
	// allocate data that is returned:
	R = NEW_OBJECTS(longinteger_object, N);
	Pts_on_plane = NEW_pint(N);
	nb_pts_on_plane = NEW_int(N);

	// allocate temporary data:
	Basis = NEW_int(4 * d);
	Basis_save = NEW_int(4 * d);
	rank_idx = NEW_int(N);
	f_subset_done = NEW_int(N);
	Coords = NEW_int(set_size * d);
	
	for (i = 0; i < N; i++) {
		f_subset_done[i] = FALSE;
		}
	for (i = 0; i < set_size; i++) {
		unrank_point(Coords + i * d, set[i]);
		}
	if (f_vv) {
		cout << "projective_space::plane_intersection_type_fast "
				"Coords:" << endl;
		int_matrix_print(Coords, set_size, d);
		}


	len = 0;
	for (rk = 0; rk < N; rk++) {
		Combi.unrank_k_subset(rk, subset, set_size, 3);
		if (f_v) {
			cout << rk << "-th subset ";
			int_vec_print(cout, subset, 3);
			cout << endl;
			}
		if (f_subset_done[rk]) {
			if (f_v) {
				cout << "skipping" << endl;
				}
			continue;
			}
		for (j = 0; j < 3; j++) {
			a = subset[j];
			//a = set[subset[j]];
			int_vec_copy(Coords + a * d, Basis + j * d, d);
			//unrank_point(Basis + j * d, a);
			}
		if (f_v3) {
			cout << "subset: ";
			int_vec_print(cout, subset, 3);
			cout << " corresponds to Basis:" << endl;
			int_matrix_print(Basis, 3, d);
			}
		r = F->rank_of_rectangular_matrix(
				Basis, 3, d, 0 /* verbose_level */);
		if (r < 3) {
			if (TRUE || f_v) {
				cout << "projective_space::plane_intersection_type_fast "
						"not independent, skip" << endl;
				cout << "subset: ";
				int_vec_print(cout, subset, 3);
				cout << endl;
				}
			rank_idx[rk] = -1;
			continue;
			}
		G->rank_longinteger_here(Basis, plane_rk, 0 /* verbose_level */);
		if (f_v) {
			cout << rk << "-th subset ";
			int_vec_print(cout, subset, 3);
			cout << " plane_rk=" << plane_rk << endl;
			}

		if (Sorting.longinteger_vec_search(R, len, plane_rk, idx)) {
			//rank_idx[rk] = idx;
			// this case should never happen:
			cout << "projective_space::plane_intersection_type_fast "
					"longinteger_vec_search(R, len, plane_rk, idx) "
					"is TRUE" << endl;
			exit(1);
			}
		else {
			if (f_v3) {
				cout << "plane_rk=" << plane_rk
						<< " was not found" << endl;
				}
			pts_on_plane = NEW_int(set_size);
			//if (f_v3) {
				//cout << "after allocating pts_on_plane,
				//plane_rk=" << plane_rk << endl;
				//}

			plane_rk.assign_to(aa);
			G->unrank_longinteger_here(Basis_save, aa,
					0 /* verbose_level */);
			if (f_v3) {
				cout << "after unrank " << plane_rk
						<< ", Basis:" << endl;
				int_matrix_print(Basis_save, 3, d);
				}
					
			l = 0;
			for (h = 0; h < set_size; h++) {
				if (FALSE && f_v3) {
					cout << "testing point " << h << ":" << endl;
					cout << "plane_rk=" << plane_rk << endl;
					}
#if 0
				plane_rk.assign_to(aa);
				G->unrank_longinteger(aa, 0 /* verbose_level */);
				if (f_v3) {
					cout << "after unrank " << plane_rk << ":" << endl;
					int_matrix_print(G->M, 3, d);
					}	
				for (j = 0; j < 3 * d; j++) {
					Basis[j] = G->M[j];
					}
#endif

				int_vec_copy(Basis_save, Basis, 3 * d);
				//a = set[h];

				int_vec_copy(Coords + h * d, Basis + 3 * d, d);

				//unrank_point(Basis + 3 * d, set[h]);
				if (FALSE && f_v3) {
					cout << "Basis and point:" << endl;
					int_matrix_print(Basis, 4, d);
					}	
				r = F->rank_of_rectangular_matrix(Basis,
						4, d, 0 /* verbose_level */);
				if (r == 3) {
					pts_on_plane[l++] = h;
					if (f_v3) {
						cout << "point " << h
								<< " is on the plane" << endl;
						}
					}
				else {
					if (FALSE && f_v3) {
						cout << "point " << h
								<< " is not on the plane" << endl;
						}
					}
				}
			if (f_v) {
				cout << "We found an " << l << "-plane, "
						"its rank is " << plane_rk << endl;
				cout << "The ranks of points on that plane are : ";
				int_vec_print(cout, pts_on_plane, l);
				cout << endl;
				}


			if (l >= 3) {
				for (j = len; j > idx; j--) {
					R[j].swap_with(R[j - 1]);
					Pts_on_plane[j] = Pts_on_plane[j - 1];
					nb_pts_on_plane[j] = nb_pts_on_plane[j - 1];
					}
				for (j = 0; j < N; j++) {
					if (f_subset_done[j] && rank_idx[j] >= idx) {
						rank_idx[j]++;
						}
					}
				plane_rk.assign_to(R[idx]);
				if (f_v3) {
					cout << "after assign_to, "
							"plane_rk=" << plane_rk << endl;
					}
				rank_idx[rk] = idx;
				len++;




				N2 = Combi.int_n_choose_k(l, 3);
				for (i = 0; i < N2; i++) {
					Combi.unrank_k_subset(i, subset2, l, 3);
					for (h = 0; h < 3; h++) {
						subset3[h] = pts_on_plane[subset2[h]];
						}
					rr = Combi.rank_k_subset(subset3, set_size, 3);
					if (f_v) {
						cout << i << "-th subset3 ";
						int_vec_print(cout, subset3, 3);
						cout << " rr=" << rr << endl;
						}
					if (!f_subset_done[rr]) {
						f_subset_done[rr] = TRUE;
						rank_idx[rr] = idx;
						}
					else if (rank_idx[rr] == -1) {
						rank_idx[rr] = idx;
						}
					else if (rank_idx[rr] != idx) {
						cout << "projective_space::plane_intersection_"
							"type_fast f_subset_done[rr] && "
							"rank_idx[rr] >= 0 && "
							"rank_idx[rr] != idx" << endl;
						exit(1);
						}
					}
				Pts_on_plane[idx] = pts_on_plane;
				nb_pts_on_plane[idx] = l;
				}
			else {
				// now l <= 2, we skip those planes:
				
				FREE_int(pts_on_plane);
				f_subset_done[rk] = TRUE;
				rank_idx[rk] = -2;
				}
			} // else
		} // next rk
	
	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(f_subset_done);
	FREE_int(rank_idx);
	FREE_int(Coords);
}

void projective_space::find_planes_which_intersect_in_at_least_s_points(
	int *set, int set_size,
	int s,
	vector<int> &plane_ranks,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v3 = (verbose_level >= 3);
	int r, rk, i, u, d, N_planes;

	int *Basis;
	int *Basis_save;
	int *Coords;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::find_planes_which_intersect_"
				"in_at_least_s_points" << endl;
		}
	if (f_vv) {
		print_set_numerical(set, set_size);
		}
	if (!Sorting.test_if_set_with_return_value(set, set_size)) {
		cout << "projective_space::find_planes_which_intersect_"
				"in_at_least_s_points "
				"the input set if not a set" << endl;
		exit(1);
		}
	d = n + 1;
	N_planes = nb_rk_k_subspaces_as_int(3);

	if (f_v) {
		cout << "N_planes=" << N_planes << endl;
		}

	// allocate temporary data:
	Basis = NEW_int(4 * d);
	Basis_save = NEW_int(4 * d);
	Coords = NEW_int(set_size * d);

	for (i = 0; i < set_size; i++) {
		unrank_point(Coords + i * d, set[i]);
		}
	if (f_vv) {
		cout << "projective_space::find_planes_which_intersect_"
				"in_at_least_s_points "
				"Coords:" << endl;
		int_matrix_print(Coords, set_size, d);
		}

	int one_percent = 0;

	if (N_planes > 1000000) {
		one_percent = N_planes / 100;
	}
	for (rk = 0; rk < N_planes; rk++) {

		if (one_percent > 0) {
			if ((rk % one_percent) == 0) {
				cout << "projective_space::find_planes_which_intersect_"
						"in_at_least_s_points "
						<< rk << " / " << N_planes << " which is "
						<< rk / one_percent << " percent done" << endl;
				}
			}
		Grass_planes->unrank_int_here(Basis_save, rk, 0 /* verbose_level */);
		//int_vec_copy(G->M, Basis_save, 3 * d);

		int nb_pts_on_plane = 0;

		for (u = 0; u < set_size; u++) {
			int_vec_copy(Basis_save, Basis, 3 * d);
			int_vec_copy(Coords + u * d, Basis + 3 * d, d);
			r = F->rank_of_rectangular_matrix(Basis,
					4, d, 0 /* verbose_level */);
			if (r < 4) {
				nb_pts_on_plane++;
				}
			}

		if (nb_pts_on_plane >= s) {
			plane_ranks.push_back(rk);
		}
	} // rk
	if (f_v) {
		cout << "projective_space::find_planes_which_intersect_"
				"in_at_least_s_points we found "
				<< plane_ranks.size() << " planes which intersect "
						"in at least " << s << " points" << endl;
		}

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(Coords);
	if (f_v) {
		cout << "projective_space::find_planes_which_intersect_"
				"in_at_least_s_points "
				"done" << endl;
		}
}

void projective_space::plane_intersection(int plane_rank,
		int *set, int set_size,
		vector<int> &point_indices,
		vector<int> &point_local_coordinates,
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
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::plane_intersection" << endl;
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
		cout << "projective_space::plane_intersection "
				"Coords:" << endl;
		int_matrix_print(Coords, set_size, d);
		}

	Grass_planes->unrank_int_here(Basis_save, plane_rank, 0 /* verbose_level */);

	int nb_pts_on_plane = 0;
	int local_rank;

	for (u = 0; u < set_size; u++) {
		int_vec_copy(Basis_save, Basis, 3 * d);
		int_vec_copy(Coords + u * d, Basis + 3 * d, d);
		r = F->rank_of_rectangular_matrix(Basis,
				4, d, 0 /* verbose_level */);
		if (r < 4) {
			nb_pts_on_plane++;
			point_indices.push_back(u);

			int_vec_copy(Basis_save, Basis, 3 * d);
			int_vec_copy(Coords + u * d, Basis + 3 * d, d);

			F->Gauss_simple(Basis, 3, d,
					base_cols, 0 /*verbose_level */);
			F->reduce_mod_subspace_and_get_coefficient_vector(
				3, d, Basis, base_cols,
				Basis + 3 * d, coefficients, verbose_level);
			F->PG_element_rank_modified(
					coefficients, 1, 3, local_rank);
			point_local_coordinates.push_back(local_rank);
		}
	}

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(Coords);
	if (f_v) {
		cout << "projective_space::plane_intersection "
				"done" << endl;
		}
}

void projective_space::line_intersection(int line_rank,
		int *set, int set_size,
		vector<int> &point_indices,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int r, i, u, d;

	int *Basis;
	int *Basis_save;
	int *Coords;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::line_intersection" << endl;
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
		cout << "projective_space::line_intersection "
				"Coords:" << endl;
		int_matrix_print(Coords, set_size, d);
		}

	Grass_lines->unrank_int_here(Basis_save, line_rank, 0 /* verbose_level */);

	for (u = 0; u < set_size; u++) {
		int_vec_copy(Basis_save, Basis, 2 * d);
		int_vec_copy(Coords + u * d, Basis + 2 * d, d);
		r = F->rank_of_rectangular_matrix(Basis,
				3, d, 0 /* verbose_level */);
		if (r < 3) {
			point_indices.push_back(u);
			}
		}

	FREE_int(Basis);
	FREE_int(Basis_save);
	FREE_int(Coords);
	if (f_v) {
		cout << "projective_space::line_intersection "
				"done" << endl;
		}
}
void projective_space::klein_correspondence(
	projective_space *P5, 
	int *set_in, int set_size, int *set_out, 
	int verbose_level)
// Computes the Pluecker coordinates
// for a line in PG(3,q) in the following order:
// (x_1,x_2,x_3,x_4,x_5,x_6) = 
// (Pluecker_12, Pluecker_34, Pluecker_13,
//    Pluecker_42, Pluecker_14, Pluecker_23)
// satisfying the quadratic form x_1x_2 + x_3x_4 + x_5x_6 = 0
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int d = n + 1;
	int h;
	int basis8[8];
	int v6[6];
	int *x4, *y4;
	int a, b, c;
	int f_elements_exponential = TRUE;
	const char *symbol_for_print = "\\alpha";


	if (f_v) {
		cout << "projective_space::klein_correspondence" << endl;
		}
	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		Grass_lines->unrank_int(a, 0 /* verbose_level */);
		if (f_vv) {
			cout << setw(5) << h << " : " << setw(5) << a << " :" << endl;
			F->latex_matrix(cout, f_elements_exponential, 
				symbol_for_print, Grass_lines->M, 2, 4);
			cout << endl;
			}
		int_vec_copy(Grass_lines->M, basis8, 8);
		if (f_vv) {
			int_matrix_print(basis8, 2, 4);
			}
		x4 = basis8;
		y4 = basis8 + 4;
		v6[0] = F->Pluecker_12(x4, y4);
		v6[1] = F->Pluecker_34(x4, y4);
		v6[2] = F->Pluecker_13(x4, y4);
		v6[3] = F->Pluecker_42(x4, y4);
		v6[4] = F->Pluecker_14(x4, y4);
		v6[5] = F->Pluecker_23(x4, y4);
		if (f_vv) {
			cout << "v6 : ";
			int_vec_print(cout, v6, 6);
			cout << endl;
			}
		a = F->mult(v6[0], v6[1]);
		b = F->mult(v6[2], v6[3]);
		c = F->mult(v6[4], v6[5]);
		d = F->add3(a, b, c);
		//cout << "a=" << a << " b=" << b << " c=" << c << endl;
		//cout << "d=" << d << endl;
		if (d) {
			cout << "d != 0" << endl;
			exit(1);
			}
		set_out[h] = P5->rank_point(v6);
		}
	if (f_v) {
		cout << "projective_space::klein_correspondence done" << endl;
		}
	
}

void projective_space::Pluecker_coordinates(
	int line_rk, int *v6, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int basis8[8];
	int *x4, *y4;
	int f_elements_exponential = FALSE;
	const char *symbol_for_print = "\\alpha";
	
	if (f_v) {
		cout << "projective_space::Pluecker_coordinates" << endl;
		}
	Grass_lines->unrank_int(line_rk, 0 /* verbose_level */);
	if (f_vv) {
		cout << setw(5) << line_rk << " :" << endl;
		F->latex_matrix(cout, f_elements_exponential, 
			symbol_for_print, Grass_lines->M, 2, 4);
		cout << endl;
		}
	int_vec_copy(Grass_lines->M, basis8, 8);
	if (f_vv) {
		int_matrix_print(basis8, 2, 4);
		}
	x4 = basis8;
	y4 = basis8 + 4;
	v6[0] = F->Pluecker_12(x4, y4);
	v6[1] = F->Pluecker_34(x4, y4);
	v6[2] = F->Pluecker_13(x4, y4);
	v6[3] = F->Pluecker_42(x4, y4);
	v6[4] = F->Pluecker_14(x4, y4);
	v6[5] = F->Pluecker_23(x4, y4);
	if (f_vv) {
		cout << "v6 : ";
		int_vec_print(cout, v6, 6);
		cout << endl;
		}
	if (f_v) {
		cout << "projective_space::Pluecker_coordinates done" << endl;
		}
}

void projective_space::klein_correspondence_special_model(
	projective_space *P5, 
	int *table, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int d = n + 1;
	int h;
	int basis8[8];
	int x6[6];
	int y6[6];
	int *x4, *y4;
	int a, b, c;
	int half;
	int f_elements_exponential = TRUE;
	const char *symbol_for_print = "\\alpha";
	//int *table;

	if (f_v) {
		cout << "projective_space::klein_correspondence" << endl;
		}
	half = F->inverse(F->add(1, 1));
	if (f_v) {
		cout << "half=" << half << endl;
		cout << "N_lines=" << N_lines << endl;
		}
	//table = NEW_int(N_lines);
	for (h = 0; h < N_lines; h++) {
		Grass_lines->unrank_int(h, 0 /* verbose_level */);
		if (f_vv) {
			cout << setw(5) << h << " :" << endl;
			F->latex_matrix(cout, f_elements_exponential, 
				symbol_for_print, Grass_lines->M, 2, 4);
			cout << endl;
			}
		int_vec_copy(Grass_lines->M, basis8, 8);
		if (f_vv) {
			int_matrix_print(basis8, 2, 4);
			}
		x4 = basis8;
		y4 = basis8 + 4;
		x6[0] = F->Pluecker_12(x4, y4);
		x6[1] = F->Pluecker_34(x4, y4);
		x6[2] = F->Pluecker_13(x4, y4);
		x6[3] = F->Pluecker_42(x4, y4);
		x6[4] = F->Pluecker_14(x4, y4);
		x6[5] = F->Pluecker_23(x4, y4);
		if (f_vv) {
			cout << "x6 : ";
			int_vec_print(cout, x6, 6);
			cout << endl;
			}
		a = F->mult(x6[0], x6[1]);
		b = F->mult(x6[2], x6[3]);
		c = F->mult(x6[4], x6[5]);
		d = F->add3(a, b, c);
		//cout << "a=" << a << " b=" << b << " c=" << c << endl;
		//cout << "d=" << d << endl;
		if (d) {
			cout << "d != 0" << endl;
			exit(1);
			}
		y6[0] = F->negate(x6[0]);
		y6[1] = x6[1];
		y6[2] = F->mult(half, F->add(x6[2], x6[3]));
		y6[3] = F->mult(half, F->add(x6[2], F->negate(x6[3])));
		y6[4] = x6[4];
		y6[5] = x6[5];
		if (f_vv) {
			cout << "y6 : ";
			int_vec_print(cout, y6, 6);
			cout << endl;
			}
		table[h] = P5->rank_point(y6);
		}

	cout << "lines in PG(3,q) to points in PG(5,q) "
			"in special model:" << endl;
	for (h = 0; h < N_lines; h++) {
		cout << setw(4) << h << " : " << setw(5) << table[h] << endl;
		}
	
	//FREE_int(table);
	if (f_v) {
		cout << "projective_space::klein_correspondence_"
				"special_model done" << endl;
		}
	
}

void projective_space::cheat_sheet_points(
		ostream &f, int verbose_level)
{
	int i, d;
	int *v;

	d = n + 1;

	v = NEW_int(d);

	f << "PG$(" << n << ", " << q << ")$ has " << N_points
			<< " points:\\\\" << endl;
	if (F->e == 1) {
		f << "\\begin{multicols}{4}" << endl;
		for (i = 0; i < N_points; i++) {
			F->PG_element_unrank_modified(v, 1, d, i);
			f << "$P_{" << i << "}=";
			int_vec_print(f, v, d);
			f << "$\\\\" << endl;
			}
		f << "\\end{multicols}" << endl;
		}
	else {
		f << "\\begin{multicols}{2}" << endl;
		for (i = 0; i < N_points; i++) {
			F->PG_element_unrank_modified(v, 1, d, i);
			f << "$P_{" << i << "}=";
			int_vec_print(f, v, d);
			f << "=";
			F->int_vec_print_elements_exponential(f, v, d, "\\alpha");
			f << "$\\\\" << endl;
			}
		f << "\\end{multicols}" << endl;
		}

	//f << "\\clearpage" << endl << endl;

	f << "Normalized from the left:\\\\" << endl;
	f << "\\begin{multicols}{4}" << endl;
	for (i = 0; i < N_points; i++) {
		F->PG_element_unrank_modified(v, 1, d, i);
		F->PG_element_normalize_from_front(v, 1, d);
		f << "$P_{" << i << "}=";
		int_vec_print(f, v, d);
		f << "$\\\\" << endl;
		}
	f << "\\end{multicols}" << endl;
	f << "\\clearpage" << endl << endl;

	FREE_int(v);
}

void projective_space::cheat_sheet_point_table(
		ostream &f, int verbose_level)
{
	int I, i, j, a, d, nb_rows, nb_cols = 5;
	int nb_rows_per_page = 40, nb_tables;
	int *v;

	d = n + 1;

	v = NEW_int(d);

	f << "PG$(" << n << ", " << q << ")$ has " << N_points
			<< " points:\\\\" << endl;

	nb_rows = (N_points + nb_cols - 1) / nb_cols;
	nb_tables = (nb_rows + nb_rows_per_page - 1) / nb_rows_per_page;

	for (I = 0; I < nb_tables; I++) {
		f << "$$" << endl;
		f << "\\begin{array}{r|*{" << nb_cols << "}{r}}" << endl;
		f << "P_{" << nb_cols << "\\cdot i+j}";
		for (j = 0; j < nb_cols; j++) {
			f << " & " << j;
			}
		f << "\\\\" << endl;
		f << "\\hline" << endl;
		for (i = 0; i < nb_rows_per_page; i++) {
			f << (I * nb_rows_per_page + i) * nb_cols;
			for (j = 0; j < nb_cols; j++) {
				a = (I * nb_rows_per_page + i) * nb_cols + j;
				f << " & ";
				if (a < N_points) {
					F->PG_element_unrank_modified(v, 1, d, a);
					int_vec_print(f, v, d);
					}
				}
			f << "\\\\" << endl;
			}
		f << "\\end{array}" << endl;
		f << "$$" << endl;
		}
	
	FREE_int(v);
}


void projective_space::cheat_sheet_points_on_lines(
	ostream &f, int verbose_level)
{

	
	f << "PG$(" << n << ", " << q << ")$ has " << N_lines
			<< " lines, each with " << k << " points:\\\\" << endl;
	if (Lines == NULL) {
		f << "Don't have Lines table\\\\" << endl;
		}
	else {
		int *row_labels;
		int *col_labels;
		int i, nb;

		row_labels = NEW_int(N_lines);
		col_labels = NEW_int(k);
		for (i = 0; i < N_lines; i++) {
			row_labels[i] = i;
			}
		for (i = 0; i < k; i++) {
			col_labels[i] = i;
			}
		//int_matrix_print_tex(f, Lines, N_lines, k);
		for (i = 0; i < N_lines; i += 40) {
			nb = MINIMUM(N_lines - i, 40);
			//f << "i=" << i << " nb=" << nb << "\\\\" << endl;
			f << "$$" << endl;
			print_integer_matrix_with_labels(f,
					Lines + i * k, nb, k, row_labels + i,
					col_labels, TRUE /* f_tex */);
			f << "$$" << endl;
			}
		FREE_int(row_labels);
		FREE_int(col_labels);
		}
}

void projective_space::cheat_sheet_lines_on_points(
	ostream &f, int verbose_level)
{
	f << "PG$(" << n << ", " << q << ")$ has " << N_points
			<< " points, each with " << r << " lines:\\\\" << endl;
	if (Lines_on_point == NULL) {
		f << "Don't have Lines\\_on\\_point table\\\\" << endl;
		}
	else {
		int *row_labels;
		int *col_labels;
		int i, nb;

		row_labels = NEW_int(N_points);
		col_labels = NEW_int(r);
		for (i = 0; i < N_points; i++) {
			row_labels[i] = i;
			}
		for (i = 0; i < r; i++) {
			col_labels[i] = i;
			}
		for (i = 0; i < N_points; i += 40) {
			nb = MINIMUM(N_points - i, 40);
			//f << "i=" << i << " nb=" << nb << "\\\\" << endl;
			f << "$$" << endl;
			print_integer_matrix_with_labels(f,
				Lines_on_point + i * r, nb, r,
				row_labels + i, col_labels, TRUE /* f_tex */);
			f << "$$" << endl;
			}
		FREE_int(row_labels);
		FREE_int(col_labels);

#if 0
		f << "$$" << endl;
		int_matrix_print_tex(f, Lines_on_point, N_points, r);
		f << "$$" << endl;
#endif
		}
}


void projective_space::cheat_sheet_subspaces(
	ostream &f, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	grassmann *Gr;
	int *v;
	int n1, k1;
	//int nb_points;
	int nb_k_subspaces;
	int i, j, u;
	int f_need_comma = FALSE;
	combinatorics_domain Combi;


	if (f_v) {
		cout << "projective_space::cheat_sheet_subspaces "
				"k=" << k << endl;
		}
	n1 = n + 1;
	k1 = k + 1;
	v = NEW_int(n1);

	if (F->q >= 10) {
		f_need_comma = TRUE;
		}

	Gr = NEW_OBJECT(grassmann);
	Gr->init(n1, k1, F, 0 /*verbose_level*/);


	//nb_points = N_points;
	nb_k_subspaces = Combi.generalized_binomial(n1, k1, q);


	f << "PG$(" << n << ", " << q << ")$ has "
			<< nb_k_subspaces << " $" << k
			<< "$-subspaces:\\\\" << endl;
	f << "\\begin{multicols}{5}" << endl;
	for (u = 0; u < nb_k_subspaces; u++) {
		Gr->unrank_int(u, 0 /* verbose_level*/);
		f << "$L_{" << u << "}=";
		f << "\\left[" << endl;
		f << "\\begin{array}{c}" << endl;
		for (i = 0; i < k1; i++) {
			for (j = 0; j < n1; j++) {
				f << Gr->M[i * n1 + j];
				if (f_need_comma && j < n1 - 1) {
					f << ", ";
					}
				}
			f << "\\\\" << endl;
			}
		f << "\\end{array}" << endl;
		f << "\\right]" << endl;

		if (n == 3 && k == 1) {
			int v6[6];

			Pluecker_coordinates(u, v6, 0 /* verbose_level */);
			f << "Pl=(" << v6[0] << "," << v6[1] << ","
					<< v6[2] << "," << v6[3] << "," << v6[4]
					<< "," << v6[5] << " ";
			f << ")" << endl;

			}
		f << "$\\\\" << endl;

		if (((u + 1) % 1000) == 0) {
			f << "\\clearpage" << endl << endl;
			}
		}
	f << "\\end{multicols}" << endl;

	f << "\\clearpage" << endl << endl;

	FREE_OBJECT(Gr);
	FREE_int(v);

	if (f_v) {
		cout << "projective_space::cheat_sheet_subspaces "
				"done" << endl;
		}
}

void projective_space::cheat_sheet_line_intersection(
	ostream &f, int verbose_level)
{
	int i, j, a;


	f << "intersection of 2 lines:" << endl;
	f << "$$" << endl;
	f << "\\begin{array}{|r|*{" << N_points << "}{r}|}" << endl;
	f << "\\hline" << endl;
	for (j = 0; j < N_points; j++) {
		f << "& " << j << endl;
		}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
	for (i = 0; i < N_points; i++) {
		f << i;
		for (j = 0; j < N_points; j++) {
			a = Line_intersection[i * N_lines + j];
			f << " & ";
			if (i != j)
				f << a;
			}
		f << "\\\\[-3pt]" << endl;
		}
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
	f << "$$" << endl;
	f << "\\clearpage" << endl;


}

void projective_space::cheat_sheet_line_through_pairs_of_points(
	ostream &f, int verbose_level)
{
	int i, j, a;



	f << "line through 2 points:" << endl;
	f << "$$" << endl;
	f << "\\begin{array}{|r|*{" << N_points << "}{r}|}" << endl;
	f << "\\hline" << endl;
	for (j = 0; j < N_points; j++) {
		f << "& " << j << endl;
		}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
	for (i = 0; i < N_points; i++) {
		f << i;
		for (j = 0; j < N_points; j++) {

			a = Line_through_two_points[i * N_points + j];
			f << " & ";
			if (i != j)
				f << a;
			}
		f << "\\\\[-3pt]" << endl;
		}
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
	f << "$$" << endl;
	f << "\\clearpage" << endl;


}

void projective_space::conic_type_randomized(int nb_times, 
	int *set, int set_size, 
	int **&Pts_on_conic, int *&nb_pts_on_conic, int &len, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int rk, h, i, j, a, /*d,*/ N, l, cnt;

	int input_pts[5];
	int six_coeffs[6];
	int vec[3];

	int subset[5];
	longinteger_object conic_rk, aa;
	int *pts_on_conic;
	int allocation_length;
	geometry_global Gg;
	combinatorics_domain Combi;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::conic_type_randomized" << endl;
		}
	if (n != 2) {
		cout << "projective_space::conic_type_randomized "
				"n != 2" << endl;
		exit(1);
		}
	if (f_vv) {
		print_set_numerical(set, set_size);
		}

	if (!Sorting.test_if_set_with_return_value(set, set_size)) {
		cout << "projective_space::conic_type_randomized "
				"the input set if not a set" << endl;
		exit(1);
		}
	//d = n + 1;
	N = Combi.int_n_choose_k(set_size, 5);

	if (f_v) {
		cout << "set_size=" << set_size << endl;
		cout << "N=number of 5-subsets of the set=" << N << endl;
		}
	
	// allocate data that is returned:
	allocation_length = 1024;
	Pts_on_conic = NEW_pint(allocation_length);
	nb_pts_on_conic = NEW_int(allocation_length);


	len = 0;
	for (cnt = 0; cnt < nb_times; cnt++) {

		rk = random_integer(N);
		Combi.unrank_k_subset(rk, subset, set_size, 5);
		if (cnt && ((cnt % 1000) == 0)) {
			cout << cnt << " / " << nb_times << " : ";
			int_vec_print(cout, subset, 5);
			cout << endl;
			}

		for (i = 0; i < len; i++) {
			if (Sorting.int_vec_is_subset_of(subset, 5,
					Pts_on_conic[i], nb_pts_on_conic[i])) {

#if 0
				cout << "The set ";
				int_vec_print(cout, subset, 5);
				cout << " is a subset of the " << i << "th conic ";
				int_vec_print(cout, Pts_on_conic[i], nb_pts_on_conic[i]);
				cout << endl;
#endif

				break;
				}
			}
		if (i < len) {
			continue;
			}
		for (j = 0; j < 5; j++) {
			a = subset[j];
			input_pts[j] = set[a];
			}
		if (FALSE /* f_v3 */) {
			cout << "subset: ";
			int_vec_print(cout, subset, 5);
			cout << "input_pts: ";
			int_vec_print(cout, input_pts, 5);
			}

		if (!determine_conic_in_plane(input_pts,
				5, six_coeffs, 0 /* verbose_level */)) {
			continue;
			}


		F->PG_element_normalize(six_coeffs, 1, 6);
		Gg.AG_element_rank_longinteger(F->q, six_coeffs, 1, 6, conic_rk);
		if (FALSE /* f_vv */) {
			cout << rk << "-th subset ";
			int_vec_print(cout, subset, 5);
			cout << " conic_rk=" << conic_rk << endl;
			}

		if (FALSE /* longinteger_vec_search(R, len, conic_rk, idx) */) {

#if 0
			cout << "projective_space::conic_type_randomized "
					"longinteger_vec_search(R, len, conic_rk, idx) "
					"is TRUE" << endl;
			cout << "The current set is ";
			int_vec_print(cout, subset, 5);
			cout << endl;
			cout << "conic_rk=" << conic_rk << endl;
			cout << "The set where it should be is ";
			int_vec_print(cout, Pts_on_conic[idx], nb_pts_on_conic[idx]);
			cout << endl;
			cout << "R[idx]=" << R[idx] << endl;
			cout << "This is the " << idx << "th conic" << endl;
			exit(1);
#endif

			}
		else {
			if (f_v3) {
				cout << "conic_rk=" << conic_rk << " was not found" << endl;
				}
			pts_on_conic = NEW_int(set_size);
			l = 0;
			for (h = 0; h < set_size; h++) {
				if (FALSE && f_v3) {
					cout << "testing point " << h << ":" << endl;
					cout << "conic_rk=" << conic_rk << endl;
					}
				
				unrank_point(vec, set[h]);
				a = F->evaluate_conic_form(six_coeffs, vec);

				
				if (a == 0) {
					pts_on_conic[l++] = h;
					if (f_v3) {
						cout << "point " << h << " is on the conic" << endl;
						}
					}
				else {
					if (FALSE && f_v3) {
						cout << "point " << h
								<< " is not on the conic" << endl;
						}
					}
				}
			if (FALSE /*f_v*/) {
				cout << "We found an " << l
						<< "-conic, its rank is " << conic_rk << endl;

				
				}


			if (l >= 8) {

				if (f_v) {
					cout << "We found an " << l << "-conic, "
							"its rank is " << conic_rk << endl;
					cout << "The " << l << " points on the "
							<< len << "th conic are: ";
					int_vec_print(cout, pts_on_conic, l);
					cout << endl;


				
					}


#if 0
				for (j = len; j > idx; j--) {
					R[j].swap_with(R[j - 1]);
					Pts_on_conic[j] = Pts_on_conic[j - 1];
					nb_pts_on_conic[j] = nb_pts_on_conic[j - 1];
					}
				conic_rk.assign_to(R[idx]);
				Pts_on_conic[idx] = pts_on_conic;
				nb_pts_on_conic[idx] = l;
#else

				//conic_rk.assign_to(R[len]);
				Pts_on_conic[len] = pts_on_conic;
				nb_pts_on_conic[len] = l;

#endif


				len++;
				if (f_v) {
					cout << "We now have found " << len
							<< " conics" << endl;


					classify C;
					int f_second = FALSE;

					C.init(nb_pts_on_conic, len, f_second, 0);

					if (f_v) {
						cout << "The conic intersection type is (";
						C.print_naked(FALSE /*f_backwards*/);
						cout << ")" << endl << endl;
						}



					}

				if (len == allocation_length) {
					int new_allocation_length = allocation_length + 1024;


					int **Pts_on_conic1;
					int *nb_pts_on_conic1;
					
					Pts_on_conic1 = NEW_pint(new_allocation_length);
					nb_pts_on_conic1 = NEW_int(new_allocation_length);
					for (i = 0; i < len; i++) {
						//R1[i] = R[i];
						Pts_on_conic1[i] = Pts_on_conic[i];
						nb_pts_on_conic1[i] = nb_pts_on_conic[i];
						}
					FREE_pint(Pts_on_conic);
					FREE_int(nb_pts_on_conic);
					Pts_on_conic = Pts_on_conic1;
					nb_pts_on_conic = nb_pts_on_conic1;
					allocation_length = new_allocation_length;
					} 




				}
			else {
				// we skip this conic:
				
				FREE_int(pts_on_conic);
				}
			} // else
		} // next rk
	
}

void projective_space::conic_intersection_type(
	int f_randomized, int nb_times, 
	int *set, int set_size, 
	int *&intersection_type, int &highest_intersection_number, 
	int f_save_largest_sets, set_of_sets *&largest_sets, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//longinteger_object *R;
	int **Pts_on_conic;
	int *nb_pts_on_conic;
	int nb_conics;
	int i, j, idx, f, l, a, t;

	if (f_v) {
		cout << "projective_space::conic_intersection_type" << endl;
		}

	if (f_randomized) {
		if (f_v) {
			cout << "projective_space::conic_intersection_type "
					"randomized" << endl;
			}
		conic_type_randomized(nb_times, 
			set, set_size, 
			Pts_on_conic, nb_pts_on_conic, nb_conics, 
			verbose_level - 1);
		}
	else {
		if (f_v) {
			cout << "projective_space::conic_intersection_type "
					"not randomized" << endl;
			}
		conic_type(
			set, set_size, 
			Pts_on_conic, nb_pts_on_conic, nb_conics, 
			verbose_level - 1);
		}

	classify C;
	int f_second = FALSE;

	C.init(nb_pts_on_conic, nb_conics, f_second, 0);
	if (f_v) {
		cout << "projective_space::conic_intersection_type "
				"conic-intersection type: ";
		C.print(FALSE /*f_backwards*/);
		}

	if (f_v) {
		cout << "The conic intersection type is (";
		C.print_naked(FALSE /*f_backwards*/);
		cout << ")" << endl << endl;
		}

	f = C.type_first[C.nb_types - 1];
	highest_intersection_number = C.data_sorted[f];
	intersection_type = NEW_int(highest_intersection_number + 1);
	int_vec_zero(intersection_type, highest_intersection_number + 1);
	for (i = 0; i < C.nb_types; i++) {
		f = C.type_first[i];
		l = C.type_len[i];
		a = C.data_sorted[f];
		intersection_type[a] = l;
		}

	if (f_save_largest_sets) {
		largest_sets = NEW_OBJECT(set_of_sets);
		t = C.nb_types - 1;
		f = C.type_first[t];
		l = C.type_len[t];
		largest_sets->init_basic_constant_size(set_size, l,
				highest_intersection_number, verbose_level);
		for (j = 0; j < l; j++) {
			idx = C.sorting_perm_inv[f + j];
			int_vec_copy(Pts_on_conic[idx],
					largest_sets->Sets[j],
					highest_intersection_number);
			}
		}

	for (i = 0; i < nb_conics; i++) {
		FREE_int(Pts_on_conic[i]);
		}
	FREE_pint(Pts_on_conic);
	FREE_int(nb_pts_on_conic);

}

void projective_space::conic_type(
	int *set, int set_size, 
	int **&Pts_on_conic, int *&nb_pts_on_conic, int &len, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int rk, h, i, j, a, /*d,*/ N, l;

	int input_pts[5];
	int six_coeffs[6];
	int vec[3];

	int subset[5];
	longinteger_object conic_rk, aa;
	int *pts_on_conic;
	int allocation_length;
	geometry_global Gg;
	combinatorics_domain Combi;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::conic_type" << endl;
		}
	if (n != 2) {
		cout << "projective_space::conic_type n != 2" << endl;
		exit(1);
		}
	if (f_vv) {
		print_set_numerical(set, set_size);
		}

	if (!Sorting.test_if_set_with_return_value(set, set_size)) {
		cout << "projective_space::conic_type the input "
				"set if not a set" << endl;
		exit(1);
		}
	//d = n + 1;
	N = Combi.int_n_choose_k(set_size, 5);

	if (f_v) {
		cout << "set_size=" << set_size << endl;
		cout << "N=number of 5-subsets of the set=" << N << endl;
		}
	
	// allocate data that is returned:
	allocation_length = 1024;
	Pts_on_conic = NEW_pint(allocation_length);
	nb_pts_on_conic = NEW_int(allocation_length);


	len = 0;
	for (rk = 0; rk < N; rk++) {

		Combi.unrank_k_subset(rk, subset, set_size, 5);
		if (f_v) {
			cout << "projective_space::conic_type rk=" << rk << " / " << N << " : ";
			int_vec_print(cout, subset, 5);
			cout << endl;
			}

		for (i = 0; i < len; i++) {
			if (Sorting.int_vec_is_subset_of(subset, 5,
					Pts_on_conic[i], nb_pts_on_conic[i])) {

#if 0
				cout << "The set ";
				int_vec_print(cout, subset, 5);
				cout << " is a subset of the " << i << "th conic ";
				int_vec_print(cout,
						Pts_on_conic[i], nb_pts_on_conic[i]);
				cout << endl;
#endif

				break;
				}
			}
		if (i < len) {
			continue;
			}
		for (j = 0; j < 5; j++) {
			a = subset[j];
			input_pts[j] = set[a];
			}
		if (f_v) {
			cout << "subset: ";
			int_vec_print(cout, subset, 5);
			cout << "input_pts: ";
			int_vec_print(cout, input_pts, 5);
			}

		if (!determine_conic_in_plane(input_pts, 5,
				six_coeffs, verbose_level - 2)) {
			if (f_v) {
				cout << "determine_conic_in_plane returns FALSE" << endl;
			}
			continue;
			}


		F->PG_element_normalize(six_coeffs, 1, 6);
		Gg.AG_element_rank_longinteger(F->q,
				six_coeffs, 1, 6, conic_rk);
		if (FALSE /* f_vv */) {
			cout << rk << "-th subset ";
			int_vec_print(cout, subset, 5);
			cout << " conic_rk=" << conic_rk << endl;
			}

		if (FALSE /* longinteger_vec_search(R, len, conic_rk, idx) */) {

#if 0
			cout << "projective_space::conic_type_randomized "
					"longinteger_vec_search(R, len, conic_rk, idx) "
					"is TRUE" << endl;
			cout << "The current set is ";
			int_vec_print(cout, subset, 5);
			cout << endl;
			cout << "conic_rk=" << conic_rk << endl;
			cout << "The set where it should be is ";
			int_vec_print(cout, Pts_on_conic[idx], nb_pts_on_conic[idx]);
			cout << endl;
			cout << "R[idx]=" << R[idx] << endl;
			cout << "This is the " << idx << "th conic" << endl;
			exit(1);
#endif

			}
		else {
			if (f_v3) {
				cout << "conic_rk=" << conic_rk
						<< " was not found" << endl;
				}
			pts_on_conic = NEW_int(set_size);
			l = 0;
			for (h = 0; h < set_size; h++) {
				if (FALSE && f_v3) {
					cout << "testing point " << h << ":" << endl;
					cout << "conic_rk=" << conic_rk << endl;
					}
				
				unrank_point(vec, set[h]);
				a = F->evaluate_conic_form(six_coeffs, vec);

				
				if (a == 0) {
					pts_on_conic[l++] = h;
					if (f_v3) {
						cout << "point " << h
								<< " is on the conic" << endl;
						}
					}
				else {
					if (FALSE && f_v3) {
						cout << "point " << h
								<< " is not on the conic" << endl;
						}
					}
				}
			if (FALSE /*f_v*/) {
				cout << "We found an " << l << "-conic, "
						"its rank is " << conic_rk << endl;

				
				}


			if (l >= 6) {

				if (f_v) {
					cout << "We found an " << l << "-conic, "
							"its rank is " << conic_rk << endl;
					cout << "The " << l << " points on the "
							<< len << "th conic are: ";
					int_vec_print(cout, pts_on_conic, l);
					cout << endl;
					}


#if 0
				for (j = len; j > idx; j--) {
					R[j].swap_with(R[j - 1]);
					Pts_on_conic[j] = Pts_on_conic[j - 1];
					nb_pts_on_conic[j] = nb_pts_on_conic[j - 1];
					}
				conic_rk.assign_to(R[idx]);
				Pts_on_conic[idx] = pts_on_conic;
				nb_pts_on_conic[idx] = l;
#else

				//conic_rk.assign_to(R[len]);
				Pts_on_conic[len] = pts_on_conic;
				nb_pts_on_conic[len] = l;

#endif


				len++;
				if (f_v) {
					cout << "We now have found " << len
							<< " conics" << endl;


					classify C;
					int f_second = FALSE;

					C.init(nb_pts_on_conic, len, f_second, 0);

					if (f_v) {
						cout << "The conic intersection type is (";
						C.print_naked(FALSE /*f_backwards*/);
						cout << ")" << endl << endl;
						}



					}

				if (len == allocation_length) {
					int new_allocation_length = allocation_length + 1024;


					int **Pts_on_conic1;
					int *nb_pts_on_conic1;
					
					Pts_on_conic1 = NEW_pint(new_allocation_length);
					nb_pts_on_conic1 = NEW_int(new_allocation_length);
					for (i = 0; i < len; i++) {
						//R1[i] = R[i];
						Pts_on_conic1[i] = Pts_on_conic[i];
						nb_pts_on_conic1[i] = nb_pts_on_conic[i];
						}
					FREE_pint(Pts_on_conic);
					FREE_int(nb_pts_on_conic);
					Pts_on_conic = Pts_on_conic1;
					nb_pts_on_conic = nb_pts_on_conic1;
					allocation_length = new_allocation_length;
					} 




				}
			else {
				// we skip this conic:
				
				FREE_int(pts_on_conic);
				}
			} // else
		} // next rk

	if (f_v) {
		cout << "projective_space::conic_type done" << endl;
		}
}

void projective_space::find_nucleus(
	int *set, int set_size, int &nucleus, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, l, sz, idx, t1, t2;
	int *Lines;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::find_nucleus" << endl;
		}

	if (n != 2) {
		cout << "projective_space::find_nucleus n != 2" << endl;
		exit(1);
		}
	if (set_size != F->q + 1) {
		cout << "projective_space::find_nucleus "
				"set_size != F->q + 1" << endl;
		exit(1);
		}

	if (Lines_on_point == NULL) {
		init_incidence_structure(verbose_level);
		}

	Lines = NEW_int(r);
	a = set[0];
	for (i = 0; i < r; i++) {
		Lines[i] = Lines_on_point[a * r + i];
		}
	sz = r;
	Sorting.int_vec_heapsort(Lines, r);

	for (i = 0; i < set_size - 1; i++) {
		b = set[1 + i];
		l = line_through_two_points(a, b);
		if (!Sorting.int_vec_search(Lines, sz, l, idx)) {
			cout << "projective_space::find_nucleus "
					"cannot find secant in pencil" << endl;
			exit(1);
			}
		for (j = idx + 1; j < sz; j++) {
			Lines[j - 1] = Lines[j];
			}
		sz--;
		}
	if (sz != 1) {
		cout << "projective_space::find_nucleus sz != 1" << endl;	
		exit(1);
		}
	t1 = Lines[0];
	if (f_v) {
		cout << "projective_space::find_nucleus t1 = " << t1 << endl;
		}
	


	a = set[1];
	for (i = 0; i < r; i++) {
		Lines[i] = Lines_on_point[a * r + i];
		}
	sz = r;
	Sorting.int_vec_heapsort(Lines, r);

	for (i = 0; i < set_size - 1; i++) {
		if (i == 0) {
			b = set[0];
			}
		else {
			b = set[1 + i];
			}
		l = line_through_two_points(a, b);
		if (!Sorting.int_vec_search(Lines, sz, l, idx)) {
			cout << "projective_space::find_nucleus "
					"cannot find secant in pencil" << endl;
			exit(1);
			}
		for (j = idx + 1; j < sz; j++) {
			Lines[j - 1] = Lines[j];
			}
		sz--;
		}
	if (sz != 1) {
		cout << "projective_space::find_nucleus sz != 1" << endl;	
		exit(1);
		}
	t2 = Lines[0];
	if (f_v) {
		cout << "projective_space::find_nucleus t2 = " << t2 << endl;
		}
	
	nucleus = intersection_of_two_lines(t1, t2);
	if (f_v) {
		cout << "projective_space::find_nucleus "
				"nucleus = " << nucleus << endl;
		int v[3];
		unrank_point(v, nucleus);
		cout << "nucleus = ";
		int_vec_print(cout, v, 3);
		cout << endl;
		}



	if (f_v) {
		cout << "projective_space::find_nucleus done" << endl;
		}
}

void projective_space::points_on_projective_triangle(
	int *&set, int &set_size, int *three_points, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int three_lines[3];
	int *Pts;
	int sz, h, i, a;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::points_on_projective_triangle" << endl;
		}
	set_size = 3 * (q - 1);
	set = NEW_int(set_size);
	sz = 3 * (q + 1);
	Pts = NEW_int(sz);
	three_lines[0] = line_through_two_points(
			three_points[0], three_points[1]);
	three_lines[1] = line_through_two_points(
			three_points[0], three_points[2]);
	three_lines[2] = line_through_two_points(
			three_points[1], three_points[2]);

	create_points_on_line(three_lines[0],
			Pts, 0 /* verbose_level */);
	create_points_on_line(three_lines[1],
			Pts + (q + 1), 0 /* verbose_level */);
	create_points_on_line(three_lines[2],
			Pts + 2 * (q + 1), 0 /* verbose_level */);
	h = 0;
	for (i = 0; i < sz; i++) {
		a = Pts[i];
		if (a == three_points[0]) {
			continue;
			}
		if (a == three_points[1]) {
			continue;
			}
		if (a == three_points[2]) {
			continue;
			}
		set[h++] = a;
		}
	if (h != set_size) {
		cout << "projective_space::points_on_projective_triangle "
				"h != set_size" << endl;
		exit(1);
		}
	Sorting.int_vec_heapsort(set, set_size);
	
	FREE_int(Pts);
	if (f_v) {
		cout << "projective_space::points_on_projective_triangle "
				"done" << endl;
		}
}

void projective_space::elliptic_curve_addition_table(
	int *A6, int *Pts, int nb_pts, int *&Table, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k;
	int pi, pj, pk;
	sorting Sorting;

	if (f_v) {
		cout << "projective_space::elliptic_curve_"
				"addition_table" << endl;
		}
	Table = NEW_int(nb_pts * nb_pts);
	for (i = 0; i < nb_pts; i++) {
		pi = Pts[i];
		for (j = 0; j < nb_pts; j++) {
			pj = Pts[j];
			pk = elliptic_curve_addition(A6, pi, pj,
					0 /* verbose_level */);
			if (!Sorting.int_vec_search(Pts, nb_pts, pk, k)) {
				cout << "projective_space::elliptic_curve_"
						"addition_table cannot find point pk" << endl;
				cout << "i=" << i << " pi=" << pi << " j=" << j
						<< " pj=" << pj << " pk=" << pk << endl;
				cout << "Pts: ";
				int_vec_print(cout, Pts, nb_pts);
				cout << endl;
				exit(1);
				}
			Table[i * nb_pts + j] = k;
			}
		}
	if (f_v) {
		cout << "projective_space::elliptic_curve_"
				"addition_table done" << endl;
		}
}

int projective_space::elliptic_curve_addition(
	int *A6, int p1_rk, int p2_rk,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int p1[3];
	int p2[3];
	int p3[3];
	int x1, y1, z1;
	int x2, y2, z2;
	int x3, y3, z3;
	int a1, a2, a3, a4, a6;
	int p3_rk;

	if (f_v) {
		cout << "projective_space::elliptic_curve_addition" << endl;
		}
	
	a1 = A6[0];
	a2 = A6[1];
	a3 = A6[2];
	a4 = A6[3];
	a6 = A6[5];
	
	unrank_point(p1, p1_rk);
	unrank_point(p2, p2_rk);
	F->PG_element_normalize(p1, 1, 3);
	F->PG_element_normalize(p2, 1, 3);
	
	x1 = p1[0];
	y1 = p1[1];
	z1 = p1[2];
	x2 = p2[0];
	y2 = p2[1];
	z2 = p2[2];
	if (f_vv) {
		cout << "projective_space::elliptic_curve_addition "
				"x1=" << x1 << " y1=" << y1 << " z1=" << z1 << endl;
		cout << "projective_space::elliptic_curve_addition "
				"x2=" << x2 << " y2=" << y2 << " z2=" << z2 << endl;
		}
	if (z1 == 0) {
		if (p1_rk != 1) {
			cout << "projective_space::elliptic_curve_addition "
					"z1 == 0 && p1_rk != 1" << endl;
			exit(1);
			}
		x3 = x2;
		y3 = y2;
		z3 = z2;
#if 0
		if (z2 == 0) {
			if (p2_rk != 1) {
				cout << "projective_space::elliptic_curve_addition "
						"z2 == 0 && p2_rk != 1" << endl;
				exit(1);
				}
			x3 = 0;
			y3 = 1;
			z3 = 0;
			}
		else {
			x3 = x2;
			y3 = F->negate(F->add3(y2, F->mult(a1, x2), a3));
			z3 = 1;
			}
#endif

		}
	else if (z2 == 0) {
		if (p2_rk != 1) {
			cout << "projective_space::elliptic_curve_addition "
					"z2 == 0 && p2_rk != 1" << endl;
			exit(1);
			}
		x3 = x1;
		y3 = y1;
		z3 = z1;

#if 0
		// at this point, we know that z1 is not zero.
		x3 = x1;
		y3 = F->negate(F->add3(y1, F->mult(a1, x1), a3));
		z3 = 1;
#endif

		}
	else {
		// now both points are affine.


		int lambda_top, lambda_bottom, lambda, nu_top, nu_bottom, nu;
		int three, two; //, m_one;
		int c;

		c = F->add4(y1, y2, F->mult(a1, x2), a3);

		if (x1 == x2 && c == 0) {
			x3 = 0;
			y3 = 1;
			z3 = 0;
			}
		else {

			two = F->add(1, 1);
			three = F->add(two, 1);
			//m_one = F->negate(1);
		
		
		
			if (x1 == x2) {

				// point duplication:
				lambda_top = F->add4(F->mult3(three, x1, x1),
						F->mult3(two, a2, x1), a4,
						F->negate(F->mult(a1, y1)));
				lambda_bottom = F->add3(F->mult(two, y1),
						F->mult(a1, x1), a3);

				nu_top = F->add4(F->negate(F->mult3(x1, x1, x1)),
						F->mult(a4, x1), F->mult(two, a6),
						F->negate(F->mult(a3, y1)));
				nu_bottom = F->add3(F->mult(two, y1),
						F->mult(a1, x1), a3);

				}
			else {
				// adding different points:
				lambda_top = F->add(y2, F->negate(y1));
				lambda_bottom = F->add(x2, F->negate(x1));

				nu_top = F->add(F->mult(y1, x2), F->negate(F->mult(y2, x1)));
				nu_bottom = lambda_bottom;
				}


			if (lambda_bottom == 0) {
				cout << "projective_space::elliptic_curve_addition "
						"lambda_bottom == 0" << endl;
				cout << "projective_space::elliptic_curve_addition "
						"x1=" << x1 << " y1=" << y1 << " z1=" << z1 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"x2=" << x2 << " y2=" << y2 << " z2=" << z2 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a1=" << a1 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a2=" << a2 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a3=" << a3 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a4=" << a4 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a6=" << a6 << endl;
				exit(1);
				}
			lambda = F->mult(lambda_top, F->inverse(lambda_bottom));

			if (nu_bottom == 0) {
				cout << "projective_space::elliptic_curve_addition "
						"nu_bottom == 0" << endl;
				exit(1);
				}
			nu = F->mult(nu_top, F->inverse(nu_bottom));

			if (f_vv) {
				cout << "projective_space::elliptic_curve_addition "
						"a1=" << a1 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a2=" << a2 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a3=" << a3 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a4=" << a4 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"a6=" << a6 << endl;
				cout << "projective_space::elliptic_curve_addition "
						"three=" << three << endl;
				cout << "projective_space::elliptic_curve_addition "
						"lambda_top=" << lambda_top << endl;
				cout << "projective_space::elliptic_curve_addition "
						"lambda=" << lambda << " nu=" << nu << endl;
				}
			x3 = F->add3(F->mult(lambda, lambda), F->mult(a1, lambda),
					F->negate(F->add3(a2, x1, x2)));
			y3 = F->negate(F->add3(F->mult(F->add(lambda, a1), x3), nu, a3));
			z3 = 1;
			}
		}
	p3[0] = x3;
	p3[1] = y3;
	p3[2] = z3;
	if (f_vv) {
		cout << "projective_space::elliptic_curve_addition "
				"x3=" << x3 << " y3=" << y3 << " z3=" << z3 << endl;
		}
	p3_rk = rank_point(p3);
	if (f_v) {
		cout << "projective_space::elliptic_curve_addition "
				"done" << endl;
		}
	return p3_rk;
}

void projective_space::draw_point_set_in_plane(
	const char *fname, int *Pts, int nb_pts, 
	int f_with_points, int f_point_labels, 
	int f_embedded, int f_sideways, int rad, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int xmax = 1000000;
	int ymax = 1000000;
	int q, i;
	int *Table;

	if (f_v) {
		cout << "projective_space::draw_point_set_in_plane" << endl;
		}
	if (n != 2) {
		cout << "projective_space::draw_point_set_in_plane n != 2" << endl;
		exit(1);
		}
	q = F->q;
	Table = NEW_int(nb_pts * 3);
	for (i = 0; i < nb_pts; i++) {
		unrank_point(Table + i * 3, Pts[i]);
		}
	if (f_point_labels) {
		char str[1000];
		char **Labels;

		Labels = NEW_pchar(nb_pts);
		for (i = 0; i < nb_pts; i++) {
			sprintf(str, "%d", Pts[i]);
			Labels[i] = NEW_char(strlen(str) + 1);
			strcpy(Labels[i], str);
			}
		if (f_v) {
			cout << "projective_space::draw_point_set_in_plane "
					"before projective_plane_draw_grid" << endl;
			}
		projective_plane_draw_grid(fname, xmax, ymax, f_with_points, rad, 
			q, Table, nb_pts, TRUE, Labels, 
			f_embedded, f_sideways, 
			verbose_level - 1);
		if (f_v) {
			cout << "projective_space::draw_point_set_in_plane "
					"after projective_plane_draw_grid" << endl;
			}
		for (i = 0; i < nb_pts; i++) {
			FREE_char(Labels[i]);
			}
		FREE_pchar(Labels);
		}
	else {
		if (f_v) {
			cout << "projective_space::draw_point_set_in_plane "
					"before projective_plane_draw_grid" << endl;
			}
		projective_plane_draw_grid(fname, xmax, ymax, f_with_points, rad, 
			q, Table, nb_pts, FALSE, NULL,
			f_embedded, f_sideways, verbose_level - 1);
		if (f_v) {
			cout << "projective_space::draw_point_set_in_plane "
					"after projective_plane_draw_grid" << endl;
			}
		}
	FREE_int(Table);
	if (f_v) {
		cout << "projective_space::draw_point_set_in_plane done" << endl;
		}
}

void projective_space::line_plane_incidence_matrix_restricted(
	int *Lines, int nb_lines, int *&M, int &nb_planes, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *the_lines;
	int line_sz;
	int *Basis; // 3 * (n + 1)
	int *Work; // 5 * (n + 1)
	int i, j;

	if (f_v) {
		cout << "projective_space::line_plane_incidence_matrix_"
				"restricted" << endl;
		}
	if (n <= 2) {
		cout << "projective_space::line_plane_incidence_matrix_"
				"restricted n <= 2" << endl;
		exit(1);
		}
	line_sz = 2 * (n + 1);
	nb_planes = Nb_subspaces[2];

	M = NEW_int(nb_lines * nb_planes);
	Basis = NEW_int(3 * (n + 1));
	Work = NEW_int(5 * (n + 1));
	the_lines = NEW_int(nb_lines * line_sz);


	int_vec_zero(M, nb_lines * nb_planes);
	for (i = 0; i < nb_lines; i++) {
		unrank_line(the_lines + i * line_sz, Lines[i]);
		}
	for (j = 0; j < nb_planes; j++) {
		unrank_plane(Basis, j);
		for (i = 0; i < nb_lines; i++) {
			int_vec_copy(Basis, Work, 3 * (n + 1));
			int_vec_copy(the_lines + i * line_sz,
					Work + 3 * (n + 1), line_sz);
			if (F->Gauss_easy(Work, 5, n + 1) == 3) {
				M[i * nb_planes + j] = 1;
				}
			}
		}
	FREE_int(Work);
	FREE_int(Basis);
	FREE_int(the_lines);
}

int projective_space::test_if_lines_are_skew(
	int line1, int line2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis1[4 * 4];
	int Basis2[4 * 4];
	int rk;
	int M[16];

	if (f_v) {
		cout << "projective_space::test_if_lines_are_skew" << endl;
		}
	if (n != 3) {
		cout << "projective_space::test_if_lines_are_skew "
				"n != 3" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "line1=" << line1 << " line2=" << line2 << endl;
		}
	unrank_line(Basis1, line1);
	if (f_v) {
		cout << "line1:" << endl;
		int_matrix_print(Basis1, 2, 4);
		}
	unrank_line(Basis2, line2);
	if (f_v) {
		cout << "line2:" << endl;
		int_matrix_print(Basis2, 2, 4);
		}
	F->intersect_subspaces(4, 2, Basis1, 2, Basis2, 
		rk, M, 0 /* verbose_level */);
	if (rk == 0) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}

int projective_space::point_of_intersection_of_a_line_and_a_line_in_three_space(
	int line1, int line2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis1[4 * 4];
	int Basis2[4 * 4];
	int rk, a;
	int M[16];

	if (f_v) {
		cout << "projective_space::point_of_intersection_of_a_"
				"line_and_a_line_in_three_space" << endl;
		}
	if (n != 3) {
		cout << "projective_space::point_of_intersection_of_a_"
				"line_and_a_line_in_three_space n != 3" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "line1=" << line1 << " line2=" << line2 << endl;
		}
	unrank_line(Basis1, line1);
	if (f_v) {
		cout << "line1:" << endl;
		int_matrix_print(Basis1, 2, 4);
		}
	unrank_line(Basis2, line2);
	if (f_v) {
		cout << "line2:" << endl;
		int_matrix_print(Basis2, 2, 4);
		}
	F->intersect_subspaces(4, 2, Basis1, 2, Basis2, 
		rk, M, 0 /* verbose_level */);
	if (rk != 1) {
		cout << "projective_space::point_of_intersection_of_a_"
				"line_and_a_line_in_three_space intersection "
				"is not a point" << endl;
		cout << "line1:" << endl;
		int_matrix_print(Basis1, 2, 4);
		cout << "line2:" << endl;
		int_matrix_print(Basis2, 2, 4);
		cout << "rk = " << rk << endl;
		exit(1);
		}
	if (f_v) {
		cout << "intersection:" << endl;
		int_matrix_print(M, 1, 4);
		}
	a = rank_point(M);
	if (f_v) {
		cout << "point rank = " << a << endl;
		}
	if (f_v) {
		cout << "projective_space::point_of_intersection_of_a_"
				"line_and_a_line_in_three_space done" << endl;
		}
	return a;
}

int projective_space::point_of_intersection_of_a_line_and_a_plane_in_three_space(
	int line, int plane, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis1[4 * 4];
	int Basis2[4 * 4];
	int rk, a;
	int M[16];

	if (f_v) {
		cout << "projective_space::point_of_intersection_of_a_"
				"line_and_a_plane_in_three_space" << endl;
		}
	if (n != 3) {
		cout << "projective_space::point_of_intersection_of_a_"
				"line_and_a_plane_in_three_space n != 3" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "line=" << line << " plane=" << plane << endl;
		}
	unrank_line(Basis1, line);
	if (f_v) {
		cout << "line:" << endl;
		int_matrix_print(Basis1, 2, 4);
		}
	unrank_plane(Basis2, plane);
	if (f_v) {
		cout << "plane:" << endl;
		int_matrix_print(Basis2, 3, 4);
		}
	F->intersect_subspaces(4, 2, Basis1, 3, Basis2, 
		rk, M, 0 /* verbose_level */);
	if (rk != 1) {
		cout << "projective_space::point_of_intersection_of_a_"
				"line_and_a_plane_in_three_space intersection "
				"is not a point" << endl;
		}
	if (f_v) {
		cout << "intersection:" << endl;
		int_matrix_print(M, 1, 4);
		}
	a = rank_point(M);
	if (f_v) {
		cout << "point rank = " << a << endl;
		}
	if (f_v) {
		cout << "projective_space::point_of_intersection_of_a_"
				"line_and_a_plane_in_three_space done" << endl;
		}
	return a;
}

int projective_space::line_of_intersection_of_two_planes_in_three_space(
	int plane1, int plane2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis1[3 * 4];
	int Basis2[3 * 4];
	int rk, a;
	int M[16];

	if (f_v) {
		cout << "projective_space::line_of_intersection_of_"
				"two_planes_in_three_space" << endl;
		}
	if (n != 3) {
		cout << "projective_space::line_of_intersection_of_"
				"two_planes_in_three_space n != 3" << endl;
		exit(1);
		}
	unrank_plane(Basis1, plane1);
	unrank_plane(Basis2, plane2);
	F->intersect_subspaces(4, 3, Basis1, 3, Basis2, 
		rk, M, 0 /* verbose_level */);
	if (rk != 2) {
		cout << "projective_space::line_of_intersection_of_"
				"two_planes_in_three_space intersection is not a line" << endl;
		}
	a = rank_line(M);
	if (f_v) {
		cout << "projective_space::line_of_intersection_of_"
				"two_planes_in_three_space done" << endl;
		}
	return a;
}

int projective_space::line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(
	int plane1, int plane2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Plane1[4];
	int Plane2[4];
	int Basis[16];
	int rk;

	if (f_v) {
		cout << "projective_space::line_of_intersection_of_"
				"two_planes_in_three_space_using_dual_coordinates" << endl;
		}
	if (n != 3) {
		cout << "projective_space::line_of_intersection_of_"
				"two_planes_in_three_space_using_dual_coordinates "
				"n != 3" << endl;
		exit(1);
		}

	unrank_point(Plane1, plane1);
	unrank_point(Plane2, plane2);

	int_vec_copy(Plane1, Basis, 4);
	int_vec_copy(Plane2, Basis + 4, 4);
	F->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
	rk = Grass_lines->rank_int_here(Basis + 8, 0 /* verbose_level */);
	return rk;
}

int projective_space::transversal_to_two_skew_lines_through_a_point(
	int line1, int line2, int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis1[4 * 4];
	int Basis2[4 * 4];
	int Basis3[4 * 4];
	int a;

	if (f_v) {
		cout << "projective_space::transversal_to_two_skew_lines_"
				"through_a_point" << endl;
		}
	if (n != 3) {
		cout << "projective_space::transversal_to_two_skew_lines_"
				"through_a_point "
				"n != 3" << endl;
		exit(1);
		}
	unrank_line(Basis1, line1);
	unrank_point(Basis1 + 8, pt);
	unrank_line(Basis2, line2);
	unrank_point(Basis2 + 8, pt);
	F->RREF_and_kernel(4, 3, Basis1, 0 /* verbose_level */);
	F->RREF_and_kernel(4, 3, Basis2, 0 /* verbose_level */);
	int_vec_copy(Basis1 + 12, Basis3, 4);
	int_vec_copy(Basis2 + 12, Basis3 + 4, 4);
	F->RREF_and_kernel(4, 2, Basis3, 0 /* verbose_level */);
	a = rank_line(Basis3 + 8);
	if (f_v) {
		cout << "projective_space::transversal_to_two_skew_lines_"
				"through_a_point "
				"done" << endl;
		}
	return a;
}



void projective_space::plane_intersection_matrix_in_three_space(
	int *Planes, int nb_planes, int *&Intersection_matrix, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, rk;

	if (f_v) {
		cout << "projective_space::plane_intersection_matrix_"
				"in_three_space" << endl;
		}
	Intersection_matrix = NEW_int(nb_planes * nb_planes);
	for (i = 0; i < nb_planes; i++) {
		a = Planes[i];
		for (j = i + 1; j < nb_planes; j++) {
			b = Planes[j];
			Intersection_matrix[i * nb_planes + j] = -1;
			rk = line_of_intersection_of_two_planes_in_three_space(
					a, b, 0 /* verbose_level */);
			Intersection_matrix[i * nb_planes + j] = rk;
			Intersection_matrix[j * nb_planes + i] = rk;
			}
		}
	for (i = 0; i < nb_planes; i++) {
		Intersection_matrix[i * nb_planes + i] = -1;
		}
	
	if (f_v) {
		cout << "projective_space::plane_intersection_matrix_"
				"in_three_space done" << endl;
		}
}

int projective_space::dual_rank_of_plane_in_three_space(
	int plane_rank, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[4 * 4];
	int rk, dual_rk;
	
	if (f_v) {
		cout << "projective_space::dual_rank_of_plane_"
				"in_three_space" << endl;
		}
	unrank_plane(Basis, plane_rank);
	rk = F->RREF_and_kernel(4, 3, Basis, 0 /* verbose_level*/);
	if (rk != 3) {
		cout << "projective_space::dual_rank_of_plane_"
				"in_three_space rk != 3" << endl;
		exit(1);
		}
	dual_rk = rank_point(Basis + 3 * 4);
	if (f_v) {
		cout << "projective_space::dual_rank_of_plane_"
				"in_three_space done" << endl;
		}
	return dual_rk;
}

void projective_space::plane_equation_from_three_lines_in_three_space(
	int *three_lines, int *plane_eqn4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[6 * 4];
	int rk;
	
	if (f_v) {
		cout << "projective_space::plane_equation_from_three_"
				"lines_in_three_space" << endl;
		}
	unrank_lines(Basis, three_lines, 3);
	rk = F->RREF_and_kernel(4, 6, Basis, 0 /* verbose_level*/);
	if (rk != 3) {
		cout << "projective_space::plane_equation_from_three_"
				"lines_in_three_space rk != 3" << endl;
		exit(1);
		}
	int_vec_copy(Basis + 3 * 4, plane_eqn4, 4);
	
	if (f_v) {
		cout << "projective_space::plane_equation_from_three_"
				"lines_in_three_space done" << endl;
		}
}

void projective_space::decomposition(
	int nb_subsets, int *sz, int **subsets, 
	incidence_structure *&Inc, 
	partitionstack *&Stack, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_pts, nb_lines;
	int *Mtx;
	int *part;
	int i, j, level;
		
	if (f_v) {
		cout << "projective_space::decomposition" << endl;
		}
	nb_pts = N_points;
	nb_lines = N_lines;
	if (f_v) {
		cout << "m = N_points = " << nb_pts << endl;
		cout << "n = N_lines = " << nb_lines << endl;
		}
	part = NEW_int(nb_subsets);
	Mtx = NEW_int(nb_pts * nb_lines);
	if (incidence_bitvec == NULL) {
		cout << "projective_space::decomposition "
				"incidence_bitvec == NULL" << endl;
		exit(1);
		}
	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < nb_lines; j++) {
			Mtx[i * nb_lines + j] = is_incident(i, j);
			}
		}
	Inc = NEW_OBJECT(incidence_structure);
	Inc->init_by_matrix(nb_pts, nb_lines, Mtx, verbose_level - 2);




	int /*ht0,*/ c, l;

	Stack = NEW_OBJECT(partitionstack);
	Stack->allocate(nb_pts + nb_lines, 0);
	Stack->subset_continguous(Inc->nb_points(), Inc->nb_lines());
	Stack->split_cell(0);



	for (level = 0; level < nb_subsets; level++) {


		//ht0 = Stack->ht;

		if (sz[level]) {
			c = Stack->cellNumber[Stack->invPointList[subsets[level][0]]];
			l = Stack->cellSize[c];
			if (sz[level] < l) {
				Stack->split_cell(subsets[level], sz[level], 0);
				part[level] = Stack->ht - 1;
				}
			else {
				part[level] = c;
				}
			}
		else {
			part[level] = -1;
			}


		if (f_v) {
			cout << "projective_space::decomposition level " << level
					<< " : partition stack after splitting:" << endl;
			Stack->print(cout);
			cout << "i : part[i]" << endl;
			for (i = 0; i < nb_subsets; i++) {
				cout << setw(3) << i << " : " << setw(3) << part[i] << endl;
				}
			}


#if 0
		int hash;
		int TDO_depth;
		int f_labeled = TRUE;
		int f_vv = (verbose_level >= 2);


		TDO_depth = nb_pts + nb_lines;
	
		if (f_vv) {
			cout << "projective_space::decomposition "
					"before compute_TDO" << endl;
			}
		hash = Inc->compute_TDO(*Stack, ht0, TDO_depth, verbose_level + 2);
		if (f_vv) {
			cout << "projective_space::decomposition "
					"after compute_TDO" << endl;
			}

		if (FALSE) {
			Inc->print_partitioned(cout, *Stack, f_labeled);
			}
		if (f_v) {
			Inc->get_and_print_decomposition_schemes(*Stack);
			Stack->print_classes(cout);
			}
#endif

		}


	FREE_int(part);
	FREE_int(Mtx);
	if (f_v) {
		cout << "projective_space::decomposition done" << endl;
		}
}

void projective_space::arc_lifting_diophant(
	int *arc, int arc_sz,
	int target_sz, int target_d,
	diophant *&D,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *line_type;
	int i, j, h, pt;
	int *free_points;
	int nb_free_points;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "projective_space::arc_lifting_diophant" << endl;
		}

	free_points = NEW_int(N_points);

	Combi.set_complement(arc, arc_sz,
			free_points, nb_free_points, N_points);



	line_type = NEW_int(N_lines);
	line_intersection_type(arc, arc_sz,
			line_type, 0 /* verbose_level */);
	if (f_vv) {
		cout << "line_type: ";
		int_vec_print_fully(cout, line_type, N_lines);
		cout << endl;
		}

	if (f_vv) {
		cout << "line type:" << endl;
		for (i = 0; i < N_lines; i++) {
			cout << i << " : " << line_type[i] << endl;
			}
		}

	classify C;
	C.init(line_type, N_lines, FALSE, 0);
	if (f_v) {
		cout << "projective_space::arc_lifting_diophant line_type:";
		C.print_naked(TRUE);
		cout << endl;
		cout << "nb_free_points=" << nb_free_points << endl;
		}


	D = NEW_OBJECT(diophant);
	D->open(N_lines + 1, nb_free_points);
	D->f_x_max = TRUE;
	for (j = 0; j < nb_free_points; j++) {
		D->x_max[j] = 1;
		}
	D->f_has_sum = TRUE;
	D->sum = target_sz - arc_sz;
	h = 0;
	for (i = 0; i < N_lines; i++) {
		if (line_type[i] > k) {
			cout << "projective_space::arc_lifting_diophant "
					"line_type[i] > k" << endl;
			exit(1);
			}
	#if 0
		if (line_type[i] < k - 1) {
			continue;
			}
	#endif
		for (j = 0; j < nb_free_points; j++) {
			pt = free_points[j];
			if (is_incident(pt, i /* line */)) {
				D->Aij(h, j) = 1;
				}
			else {
				D->Aij(h, j) = 0;
				}
			}
		D->type[h] = t_LE;
		D->RHSi(h) = target_d - line_type[i];
		h++;
		}


	// add one extra row:
	for (j = 0; j < nb_free_points; j++) {
		D->Aij(h, j) = 1;
		}
	D->type[h] = t_EQ;
	D->RHSi(h) = target_sz - arc_sz;
	h++;

	D->m = h;

	D->init_var_labels(free_points, verbose_level);

	if (f_vv) {
		cout << "projective_space::arc_lifting_diophant "
				"The system is:" << endl;
		D->print_tight();
		}

#if 0
	if (f_save_system) {
		cout << "do_arc_lifting saving the system "
				"to file " << fname_system << endl;
		D->save_in_general_format(fname_system, 0 /* verbose_level */);
		cout << "do_arc_lifting saving the system "
				"to file " << fname_system << " done" << endl;
		D->print();
		D->print_tight();
		}
#endif

	FREE_int(line_type);
	FREE_int(free_points);

	if (f_v) {
		cout << "projective_space::arc_lifting_diophant done" << endl;
		}

}


void projective_space::rearrange_arc_for_lifting(int *Arc6,
		int P1, int P2, int partition_rk, int *arc,
		int verbose_level)
// P1 and P2 are points on the arc.
// Find them and remove them
// so we can find the remaining four point of the arc:
{
	int f_v = (verbose_level >= 1);
	int i, a, h;
	int part[4];
	int pts[4];
	combinatorics_domain Combi;

	if (f_v) {
		cout << "projective_space::rearrange_arc_for_lifting" << endl;
		}
	arc[0] = P1;
	arc[1] = P2;
	h = 2;
	for (i = 0; i < 6; i++) {
		a = Arc6[i];
		if (a == P1 || a == P2) {
			continue;
		}
		arc[h++] = a;
	}
	if (h != 6) {
		cout << "projective_space::rearrange_arc_for_lifting "
				"h != 6" << endl;
		exit(1);
	}
	// now arc[2], arc[3], arc[4], arc[5] are the remaining four points
	// of the arc.

	Combi.set_partition_4_into_2_unrank(partition_rk, part);

	int_vec_copy(arc + 2, pts, 4);

	for (i = 0; i < 4; i++) {
		a = part[i];
		arc[2 + i] = pts[a];
	}


	if (f_v) {
		cout << "projective_space::rearrange_arc_for_lifting done" << endl;
		}
}

void projective_space::find_two_lines_for_arc_lifting(
		int P1, int P2, int &line1, int &line2,
		int verbose_level)
// P1 and P2 are points on the arc and in the plane W=0.
// We find two skew lines in space through P1 and P2, respectively.
{
	int f_v = (verbose_level >= 1);
	int Basis[16];
	int Basis2[16];
	int Basis_search[16];
	int Basis_search_copy[16];
	int base_cols[4];
	int i, N, rk;
	geometry_global Gg;

	if (f_v) {
		cout << "projective_space::find_two_lines_for_arc_lifting" << endl;
		}
	if (n != 3) {
		cout << "projective_space::find_two_lines_for_arc_lifting "
				"n != 3" << endl;
		exit(1);
	}
	// unrank points in the plane W=3:
	// Note the points are points in PG(3,q), not local coordinates.
	unrank_point(Basis, P1);
	unrank_point(Basis + 4, P2);
	if (Basis[3]) {
		cout << "projective_space::find_two_lines_for_arc_lifting "
				"Basis[3] != 0, the point P1 does not lie "
				"in the hyperplane W = 0" << endl;
		exit(1);
	}
	if (Basis[7]) {
		cout << "projective_space::find_two_lines_for_arc_lifting "
				"Basis[7] != 0, the point P2 does not lie "
				"in the hyperplane W = 0" << endl;
		exit(1);
	}
	int_vec_zero(Basis + 8, 8);

	N = Gg.nb_PG_elements(3, q);
	for (i = 0; i < N; i++) {
		int_vec_copy(Basis, Basis_search, 4);
		int_vec_copy(Basis + 4, Basis_search + 4, 4);
		F->PG_element_unrank_modified(
				Basis_search + 8, 1, 4, i);
		if (Basis_search[11] == 0) {
			continue;
		}
		int_vec_copy(Basis_search, Basis_search_copy, 12);
		rk = F->Gauss_easy_memory_given(Basis_search_copy, 3, 4, base_cols);
		if (rk == 3) {
			break;
		}
	}
	if (i == N) {
		cout << "projective_space::find_two_lines_for_arc_lifting "
				"i == N, could not fined line1" << endl;
		exit(1);
	}
	int p0, p1;

	p0 = i;

	for (i = p0 + 1; i < N; i++) {
		int_vec_copy(Basis, Basis_search, 4);
		int_vec_copy(Basis + 4, Basis_search + 4, 4);
		F->PG_element_unrank_modified(
				Basis_search + 8, 1, 4, p0);
		F->PG_element_unrank_modified(
				Basis_search + 12, 1, 4, i);
		if (Basis_search[15] == 0) {
			continue;
		}
		int_vec_copy(Basis_search, Basis_search_copy, 16);
		rk = F->Gauss_easy_memory_given(Basis_search_copy, 4, 4, base_cols);
		if (rk == 4) {
			break;
		}
	}
	if (i == N) {
		cout << "projective_space::find_two_lines_for_arc_lifting "
				"i == N, could not fined line2" << endl;
		exit(1);
	}
	p1 = i;

	if (f_v) {
		cout << "projective_space::find_two_lines_for_arc_lifting "
				"p0=" << p0 << " p1=" << p1 << endl;
	}
	F->PG_element_unrank_modified(
			Basis + 8, 1, 4, p0);
	F->PG_element_unrank_modified(
			Basis + 12, 1, 4, p1);
	if (f_v) {
		cout << "projective_space::find_two_lines_for_arc_lifting " << endl;
		cout << "Basis:" << endl;
		int_matrix_print(Basis, 4, 4);
	}
	int_vec_copy(Basis, Basis2, 4);
	int_vec_copy(Basis + 8, Basis2 + 4, 4);
	int_vec_copy(Basis + 4, Basis2 + 8, 4);
	int_vec_copy(Basis + 12, Basis2 + 12, 4);
	if (f_v) {
		cout << "projective_space::find_two_lines_for_arc_lifting "
				"Basis2:" << endl;
		int_matrix_print(Basis2, 4, 4);
	}
	line1 = rank_line(Basis2);
	line2 = rank_line(Basis2 + 8);
	if (f_v) {
		cout << "projective_space::find_two_lines_for_arc_lifting "
				"line1=" << line1 << " line2=" << line2 << endl;
	}
	if (f_v) {
		cout << "projective_space::find_two_lines_for_arc_lifting "
				"done" << endl;
		}
}

void projective_space::lifted_action_on_hyperplane_W0_fixing_two_lines(
		int *A3, int f_semilinear, int frobenius,
		int line1, int line2,
		int *A4,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Line1[8];
	int Line2[8];
	int P1A[3];
	int P2A[3];
	int A3t[9];
	int x[3];
	int y[3];
	int xmy[4];
	int Mt[16];
	int M[16];
	int Mv[16];
	int v[4];
	int w[4];
	int lmei[4];
	int m1;
	int M_tmp[16];
	int tmp_basecols[4];
	int lambda, mu; //, epsilon, iota;
	int abgd[4];
	int i, j;
	int f_swap; // does A3 swap P1 and P2?

	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines" << endl;
		}
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"A3:" << endl;
		int_matrix_print(A3, 3, 3);
		cout << "f_semilinear = " << f_semilinear << " frobenius=" << frobenius << endl;
	}
	m1 = F->negate(1);
	unrank_line(Line1, line1);
	unrank_line(Line2, line2);
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"input Line1:" << endl;
		int_matrix_print(Line1, 2, 4);
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"input Line2:" << endl;
		int_matrix_print(Line2, 2, 4);
	}
	F->Gauss_step_make_pivot_one(Line1 + 4, Line1,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line1[3] = 0 and Line1[7] = 1
	F->Gauss_step_make_pivot_one(Line2 + 4, Line2,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line2[3] = 0 and Line2[7] = 1
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"modified Line1:" << endl;
		int_matrix_print(Line1, 2, 4);
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"modified Line2:" << endl;
		int_matrix_print(Line2, 2, 4);
	}

	F->PG_element_normalize(Line1, 1, 4);
	F->PG_element_normalize(Line2, 1, 4);
	F->PG_element_normalize(Line1 + 4, 1, 4);
	F->PG_element_normalize(Line2 + 4, 1, 4);
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"P1 = first point on Line1:" << endl;
		int_matrix_print(Line1, 1, 4);
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"P2 = first point on Line2:" << endl;
		int_matrix_print(Line2, 1, 4);
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"x = second point on Line1:" << endl;
		int_matrix_print(Line1 + 4, 1, 4);
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"y = second point on Line2:" << endl;
		int_matrix_print(Line2 + 4, 1, 4);
	}
	// compute P1*A3 to figure out if A switches P1 and P2 or not:
	F->mult_vector_from_the_left(Line1, A3, P1A, 3, 3);
	F->mult_vector_from_the_left(Line2, A3, P2A, 3, 3);
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"P1 * A = " << endl;
		int_matrix_print(P1A, 1, 3);
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"P2 * A = " << endl;
		int_matrix_print(P2A, 1, 3);
	}
	if (f_semilinear) {
		if (f_v) {
			cout << "projective_space::lifted_action_on_"
					"hyperplane_W0_fixing_two_lines applying frobenius" << endl;
		}
		F->vector_frobenius_power_in_place(P1A, 3, frobenius);
		F->vector_frobenius_power_in_place(P2A, 3, frobenius);
	}
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"P1 * A ^Phi^frobenius = " << endl;
		int_matrix_print(P1A, 1, 3);
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"P2 * A ^Phi^frobenius = " << endl;
		int_matrix_print(P2A, 1, 3);
	}
	F->PG_element_normalize(P1A, 1, 3);
	F->PG_element_normalize(P2A, 1, 3);
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"normalized P1 * A = " << endl;
		int_matrix_print(P1A, 1, 3);
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"normalized P2 * A = " << endl;
		int_matrix_print(P2A, 1, 3);
	}
	if (int_vec_compare(P1A, Line1, 3) == 0) {
		f_swap = FALSE;
		if (int_vec_compare(P2A, Line2, 3)) {
			cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"We don't have a swap but A3 does not stabilize P2" << endl;
			exit(1);
		}
	}
	else if (int_vec_compare(P1A, Line2, 3) == 0) {
		f_swap = TRUE;
		if (int_vec_compare(P2A, Line1, 3)) {
			cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"We have a swap but A3 does not map P2 to P1" << endl;
			exit(1);
		}
	}
	else {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines cannot determine "
				"if we have a swap or not." << endl;
		exit(1);
	}

	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"f_swap=" << f_swap << endl;
	}

	int_vec_copy(Line1 + 4, x, 3);
	int_vec_copy(Line2 + 4, y, 3);
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"x:" << endl;
		int_matrix_print(x, 1, 3);
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"y:" << endl;
		int_matrix_print(y, 1, 3);
	}

	F->linear_combination_of_vectors(1, x, m1, y, xmy, 3);
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"xmy:" << endl;
		int_matrix_print(xmy, 1, 3);
	}

	F->transpose_matrix(A3, A3t, 3, 3);
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"A3t:" << endl;
		int_matrix_print(A3t, 3, 3);
	}


	F->mult_vector_from_the_right(A3t, xmy, v, 3, 3);
	if (f_semilinear) {
		F->vector_frobenius_power_in_place(v, 3, frobenius);
	}
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"v:" << endl;
		int_matrix_print(v, 1, 3);
	}
	F->mult_vector_from_the_right(A3t, x, w, 3, 3);
	if (f_semilinear) {
		F->vector_frobenius_power_in_place(w, 3, frobenius);
	}
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"w:" << endl;
		int_matrix_print(w, 1, 3);
	}

	if (f_swap) {
		int_vec_copy(Line2 + 4, Mt + 0, 4);
		int_vec_copy(Line2 + 0, Mt + 4, 4);
		int_vec_copy(Line1 + 4, Mt + 8, 4);
		int_vec_copy(Line1 + 0, Mt + 12, 4);
	}
	else {
		int_vec_copy(Line1 + 4, Mt + 0, 4);
		int_vec_copy(Line1 + 0, Mt + 4, 4);
		int_vec_copy(Line2 + 4, Mt + 8, 4);
		int_vec_copy(Line2 + 0, Mt + 12, 4);
	}

	F->negate_vector_in_place(Mt + 8, 8);
	F->transpose_matrix(Mt, M, 4, 4);
	//int_vec_copy(Mt, M, 16);
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"M:" << endl;
		int_matrix_print(M, 4, 4);
	}

	F->invert_matrix_memory_given(M, Mv, 4,
			M_tmp, tmp_basecols);
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"Mv:" << endl;
		int_matrix_print(Mv, 4, 4);
	}

	v[3] = 0;
	w[3] = 0;
	F->mult_vector_from_the_right(Mv, v, lmei, 4, 4);
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"lmei:" << endl;
		int_matrix_print(lmei, 1, 4);
	}
	lambda = lmei[0];
	mu = lmei[1];
	//epsilon = lmei[2];
	//iota = lmei[3];

	if (f_swap) {
		F->linear_combination_of_three_vectors(
				lambda, y, mu, Line2, m1, w, abgd, 3);
	}
	else {
		F->linear_combination_of_three_vectors(
				lambda, x, mu, Line1, m1, w, abgd, 3);
	}
	abgd[3] = lambda;
	if (f_semilinear) {
		F->vector_frobenius_power_in_place(abgd, 4, F->e - frobenius);
	}
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"abgd:" << endl;
		int_matrix_print(abgd, 1, 4);
	}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			A4[i * 4 + j] = A3[i * 3 + j];
		}
		A4[i * 4 + 3] = 0;
	}
	int_vec_copy(abgd, A4 + 4 * 3, 4);
	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines "
				"A4:" << endl;
		int_matrix_print(A4, 4, 4);
	}

	if (f_semilinear) {
		A4[16] = frobenius;
	}

	if (f_v) {
		cout << "projective_space::lifted_action_on_"
				"hyperplane_W0_fixing_two_lines done" << endl;
		}
}

void projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines(
		int line1_from, int line1_to,
		int line2_from, int line2_to,
		int *A4,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Line1_from[8];
	int Line2_from[8];
	int Line1_to[8];
	int Line2_to[8];
	int P1[4];
	int P2[4];
	int x[4];
	int y[4];
	int u[4];
	int v[4];
	int umv[4];
	int M[16];
	int Mv[16];
	int lmei[4];
	int m1;
	int M_tmp[16];
	int tmp_basecols[4];
	int lambda, mu; //, epsilon, iota;
	int abgd[4];
	int i, j;

	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines" << endl;
		}
	m1 = F->negate(1);

	unrank_line(Line1_from, line1_from);
	unrank_line(Line2_from, line2_from);
	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"input Line1_from:" << endl;
		int_matrix_print(Line1_from, 2, 4);
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"input Line2_from:" << endl;
		int_matrix_print(Line2_from, 2, 4);
	}
	F->Gauss_step_make_pivot_one(Line1_from + 4, Line1_from,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line1[3] = 0 and Line1[7] = 1
	F->Gauss_step_make_pivot_one(Line2_from + 4, Line2_from,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line2[3] = 0 and Line2[7] = 1
	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"modified Line1_from:" << endl;
		int_matrix_print(Line1_from, 2, 4);
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"modified Line2_from:" << endl;
		int_matrix_print(Line2_from, 2, 4);
	}

	F->PG_element_normalize(Line1_from, 1, 4);
	F->PG_element_normalize(Line2_from, 1, 4);
	F->PG_element_normalize(Line1_from + 4, 1, 4);
	F->PG_element_normalize(Line2_from + 4, 1, 4);
	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"P1 = first point on Line1_from:" << endl;
		int_matrix_print(Line1_from, 1, 4);
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"P2 = first point on Line2_from:" << endl;
		int_matrix_print(Line2_from, 1, 4);
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"u = second point on Line1_from:" << endl;
		int_matrix_print(Line1_from + 4, 1, 4);
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"v = second point on Line2_from:" << endl;
		int_matrix_print(Line2_from + 4, 1, 4);
	}
	int_vec_copy(Line1_from + 4, u, 4);
	int_vec_copy(Line1_from, P1, 4);
	int_vec_copy(Line2_from + 4, v, 4);
	int_vec_copy(Line2_from, P2, 4);


	unrank_line(Line1_to, line1_to);
	unrank_line(Line2_to, line2_to);
	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"input Line1_to:" << endl;
		int_matrix_print(Line1_to, 2, 4);
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"input Line2_to:" << endl;
		int_matrix_print(Line2_to, 2, 4);
	}
	F->Gauss_step_make_pivot_one(Line1_to + 4, Line1_to,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line1[3] = 0 and Line1[7] = 1
	F->Gauss_step_make_pivot_one(Line2_to + 4, Line2_to,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line2[3] = 0 and Line2[7] = 1
	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"modified Line1_to:" << endl;
		int_matrix_print(Line1_to, 2, 4);
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"modified Line2_to:" << endl;
		int_matrix_print(Line2_to, 2, 4);
	}

	F->PG_element_normalize(Line1_to, 1, 4);
	F->PG_element_normalize(Line2_to, 1, 4);
	F->PG_element_normalize(Line1_to + 4, 1, 4);
	F->PG_element_normalize(Line2_to + 4, 1, 4);
	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"P1 = first point on Line1_to:" << endl;
		int_matrix_print(Line1_to, 1, 4);
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"P2 = first point on Line2_to:" << endl;
		int_matrix_print(Line2_to, 1, 4);
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"x = second point on Line1_to:" << endl;
		int_matrix_print(Line1_to + 4, 1, 4);
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"y = second point on Line2_to:" << endl;
		int_matrix_print(Line2_to + 4, 1, 4);
	}


	int_vec_copy(Line1_to + 4, x, 4);
	//int_vec_copy(Line1_to, P1, 4);
	if (int_vec_compare(P1, Line1_to, 4)) {
		cout << "Line1_from and Line1_to must intersect in W=0" << endl;
		exit(1);
	}
	int_vec_copy(Line2_to + 4, y, 4);
	//int_vec_copy(Line2_to, P2, 4);
	if (int_vec_compare(P2, Line2_to, 4)) {
		cout << "Line2_from and Line2_to must intersect in W=0" << endl;
		exit(1);
	}


	F->linear_combination_of_vectors(1, u, m1, v, umv, 3);
	umv[3] = 0;
	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"umv:" << endl;
		int_matrix_print(umv, 1, 4);
	}

	int_vec_copy(x, M + 0, 4);
	int_vec_copy(P1, M + 4, 4);
	int_vec_copy(y, M + 8, 4);
	int_vec_copy(P2, M + 12, 4);

	F->negate_vector_in_place(M + 8, 8);
	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"M:" << endl;
		int_matrix_print(M, 4, 4);
	}

	F->invert_matrix_memory_given(M, Mv, 4, M_tmp, tmp_basecols);
	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"Mv:" << endl;
		int_matrix_print(Mv, 4, 4);
	}

	F->mult_vector_from_the_left(umv, Mv, lmei, 4, 4);
	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"lmei=" << endl;
		int_matrix_print(lmei, 1, 4);
	}
	lambda = lmei[0];
	mu = lmei[1];
	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"lambda=" << lambda << " mu=" << mu << endl;
	}

	F->linear_combination_of_three_vectors(
			lambda, x, mu, P1, m1, u, abgd, 3);

	abgd[3] = lambda;

	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"abgd:" << endl;
		int_matrix_print(abgd, 1, 4);
	}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			if (i == j) {
				A4[i * 4 + j] = 1;
			}
			else {
				A4[i * 4 + j] = 0;
			}
		}
	}
	int_vec_copy(abgd, A4 + 3 * 4, 4);
	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines "
				"A4:" << endl;
		int_matrix_print(A4, 4, 4);
	}

	if (f_v) {
		cout << "projective_space::find_matrix_fixing_hyperplane_and_moving_two_skew_lines done" << endl;
		}

}

void projective_space::andre_preimage(projective_space *P4,
	int *set2, int sz2, int *set4, int &sz4, int verbose_level)
// we must be a projective plane
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	finite_field *FQ;
	finite_field *Fq;
	int /*Q,*/ q;
	int *v, *w1, *w2, *w3, *v2;
	int *components;
	int *embedding;
	int *pair_embedding;
	int i, h, k, a, a0, a1, b, b0, b1, e, alpha;

	if (f_v) {
		cout << "projective_space::andre_preimage" << endl;
		}
	FQ = F;
	//Q = FQ->q;
	alpha = FQ->p;
	if (f_vv) {
		cout << "alpha=" << alpha << endl;
		//FQ->print(TRUE /* f_add_mult_table */);
		}


	Fq = P4->F;
	q = Fq->q;

	v = NEW_int(3);
	w1 = NEW_int(5);
	w2 = NEW_int(5);
	w3 = NEW_int(5);
	v2 = NEW_int(2);
	e = F->e >> 1;
	if (f_vv) {
		cout << "projective_space::andre_preimage e=" << e << endl;
		}

	FQ->subfield_embedding_2dimensional(*Fq,
		components, embedding, pair_embedding, verbose_level - 3);

		// we think of FQ as two dimensional vector space
		// over Fq with basis (1,alpha)
		// for i,j \in Fq, with x = i + j * alpha \in FQ, we have
		// pair_embedding[i * q + j] = x;
		// also,
		// components[x * 2 + 0] = i;
		// components[x * 2 + 1] = j;
		// also, for i \in Fq, embedding[i] is the element
		// in FQ that corresponds to i

		// components[Q * 2]
		// embedding[q]
		// pair_embedding[q * q]

	if (f_vv) {
		FQ->print_embedding(*Fq,
			components, embedding, pair_embedding);
		}


	sz4 = 0;
	for (i = 0; i < sz2; i++) {
		if (f_vv) {
			cout << "projective_space::andre_preimage "
					"input point " << i << " : ";
			}
		unrank_point(v, set2[i]);
		FQ->PG_element_normalize(v, 1, 3);
		if (f_vv) {
			int_vec_print(cout, v, 3);
			cout << " becomes ";
			}

		if (v[2] == 0) {

			// we are dealing with a point on the
			// line at infinity.
			// Such a point corresponds to a line of the spread.
			// We create the line and then create all
			// q + 1 points on that line.

			if (f_vv) {
				cout << endl;
				}
			// w1[4] is the GF(q)-vector corresponding
			// to the GF(q^2)-vector v[2]
			// w2[4] is the GF(q)-vector corresponding
			// to the GF(q^2)-vector v[2] * alpha
			// where v[2] runs through the points of PG(1,q^2).
			// That way, w1[4] and w2[4] are a GF(q)-basis for the
			// 2-dimensional subspace v[2] (when viewed over GF(q)),
			// which is an element of the regular spread.

			for (h = 0; h < 2; h++) {
				a = v[h];
				a0 = components[a * 2 + 0];
				a1 = components[a * 2 + 1];
				b = FQ->mult(a, alpha);
				b0 = components[b * 2 + 0];
				b1 = components[b * 2 + 1];
				w1[2 * h + 0] = a0;
				w1[2 * h + 1] = a1;
				w2[2 * h + 0] = b0;
				w2[2 * h + 1] = b1;
				}
			if (FALSE) {
				cout << "w1=";
				int_vec_print(cout, w1, 4);
				cout << "w2=";
				int_vec_print(cout, w2, 4);
				cout << endl;
				}

			// now we create all points on the line
			// spanned by w1[4] and w2[4]:
			// There are q + 1 of these points.
			// We make sure that the coordinate vectors
			// have a zero in the last spot.

			for (h = 0; h < q + 1; h++) {
				Fq->PG_element_unrank_modified(v2, 1, 2, h);
				if (FALSE) {
					cout << "v2=";
					int_vec_print(cout, v2, 2);
					cout << " : ";
					}
				for (k = 0; k < 4; k++) {
					w3[k] = Fq->add(Fq->mult(v2[0], w1[k]),
							Fq->mult(v2[1], w2[k]));
					}
				w3[4] = 0;
				if (f_vv) {
					cout << " ";
					int_vec_print(cout, w3, 5);
					}
				a = P4->rank_point(w3);
				if (f_vv) {
					cout << " rank " << a << endl;
					}
				set4[sz4++] = a;
				}
			}
		else {

			// we are dealing with an affine point:
			// We make sure that the coordinate vector
			// has a zero in the last spot.


			for (h = 0; h < 2; h++) {
				a = v[h];
				a0 = components[a * 2 + 0];
				a1 = components[a * 2 + 1];
				w1[2 * h + 0] = a0;
				w1[2 * h + 1] = a1;
				}
			w1[4] = 1;
			if (f_vv) {
				//cout << "w1=";
				int_vec_print(cout, w1, 5);
				}
			a = P4->rank_point(w1);
			if (f_vv) {
				cout << " rank " << a << endl;
				}
			set4[sz4++] = a;
			}
		}
	if (f_v) {
		cout << "projective_space::andre_preimage "
				"we found " << sz4 << " points:" << endl;
		int_vec_print(cout, set4, sz4);
		cout << endl;
		P4->print_set(set4, sz4);
		for (i = 0; i < sz4; i++) {
			cout << set4[i] << " ";
			}
		cout << endl;
		}


	FREE_int(components);
	FREE_int(embedding);
	FREE_int(pair_embedding);
	if (f_v) {
		cout << "projective_space::andre_preimage done" << endl;
		}
}



}}



