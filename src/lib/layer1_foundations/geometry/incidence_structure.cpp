// incidence_structure.cpp
//
// Anton Betten
//
// June 20, 2010



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {



// #############################################################################
// incidence_structure
// #############################################################################

incidence_structure::incidence_structure()
{
	//std::string label;

	nb_rows = nb_cols = 0;

	f_rowsums_constant = false;
	f_colsums_constant = false;
	r = k = 0;
	nb_lines_on_point = NULL;
	nb_points_on_line = NULL;
	max_r = min_r = max_k = min_k = 0;
	lines_on_point = NULL;
	points_on_line = NULL;

	realization_type = 0;

	M = NULL;
	O = NULL;
	H = NULL;
	P = NULL;

}


incidence_structure::~incidence_structure()
{
	if (M) {
		FREE_int(M);
	}
	if (O) {
		// do not destroy
	}
	if (H) {
		// do not destroy
	}
	if (nb_lines_on_point) {
		FREE_int(nb_lines_on_point);
	}
	if (nb_points_on_line) {
		FREE_int(nb_points_on_line);
	}
	if (lines_on_point) {
		FREE_int(lines_on_point);
	}
	if (points_on_line) {
		FREE_int(points_on_line);
	}
}

void incidence_structure::check_point_pairs(int verbose_level)
{
	int *Mtx;
	int i, j, nb;
	int *Lines;

	Lines = NEW_int(nb_cols);
	Mtx = NEW_int(nb_rows * nb_rows);
	for (i = 0; i < nb_rows; i++) {
		for (j = i; j < nb_rows; j++) {
			nb = lines_through_two_points(Lines, i, j, 0);
			Mtx[i * nb_rows + j] = nb;
			Mtx[j * nb_rows + i] = nb;
		}
	}
	cout << "nb of lines through two points:" << endl;
	Int_vec_print_integer_matrix_width(cout, Mtx, nb_rows, nb_rows, nb_rows, 1);
	
	FREE_int(Lines);
}

int incidence_structure::lines_through_two_points(
		int *lines,
		int p1, int p2, int verbose_level)
{
	int h1, h2, l1, l2, nb;
	
	nb = 0;
	for (h1 = 0; h1 < nb_lines_on_point[p1]; h1++) {
		l1 = lines_on_point[p1 * max_r + h1];
		for (h2 = 0; h2 < nb_lines_on_point[p2]; h2++) {
			l2 = lines_on_point[p2 * max_r + h2];
			if (l1 == l2) {
				lines[nb++] = l1;
				break;
			}
		}
	}
	return nb;
}

void incidence_structure::init_projective_space(
		projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "incidence_structure::init_projective_space" << endl;
	}
	realization_type = INCIDENCE_STRUCTURE_REALIZATION_BY_PROJECTIVE_SPACE;
	incidence_structure::P = P;
	nb_rows = P->Subspaces->N_points;
	nb_cols = P->Subspaces->N_lines;

	f_rowsums_constant = true;
	f_colsums_constant = true;
	r = P->Subspaces->r;
	k = P->Subspaces->k;
	nb_lines_on_point = NEW_int(nb_rows);
	nb_points_on_line = NEW_int(nb_cols);
	for (i = 0; i < nb_rows; i++) {
		nb_lines_on_point[i] = r;
	}
	for (i = 0; i < nb_cols; i++) {
		nb_points_on_line[i] = k;
	}
	//int *nb_lines_on_point;
	//int *nb_points_on_line;
	max_r = min_r = r;
	max_k = min_k = k;

	if (f_v) {
		cout << "incidence_structure::init_projective_space nb_rows=" << nb_rows << endl;
		cout << "incidence_structure::init_projective_space nb_cols=" << nb_cols << endl;
	}
}

void incidence_structure::init_hjelmslev(
		hjelmslev *H, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//algebra_global Algebra;

	if (f_v) {
		cout << "incidence_structure::init_hjelmslev" << endl;
	}
	realization_type = INCIDENCE_STRUCTURE_REALIZATION_BY_HJELMSLEV;
	incidence_structure::H = H;
	f_rowsums_constant = true;
	f_colsums_constant = true;
	//max_r = min_r = r = O->alpha;
	//max_k = min_k = k = O->q + 1;
	nb_rows = H->R->nb_PHG_elements(H->k);
	nb_cols = H->number_of_submodules();
	if (f_v) {
		cout << "incidence_structure::init_hjelmslev nb_rows=" << nb_rows << endl;
		cout << "incidence_structure::init_hjelmslev nb_cols=" << nb_cols << endl;
	}
	int *Mtx;
	int *Inc_Mtx;
	int *v;
	int *base_cols;
	int mtx_rk;
	int i, j, h, k, n;

	k = H->k;
	n = H->n;
	Mtx = NEW_int((k + 1) * n);
	Inc_Mtx = NEW_int(nb_rows * nb_cols);
	v = NEW_int(k);
	base_cols = NEW_int(n);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			cout << "i=" << i << " j=" << j << endl;
			H->R->PHG_element_unrank(Mtx, 1, n, i);
			H->unrank_lint(Mtx + n, j, 0);
			Int_vec_print_integer_matrix_width(cout, Mtx, k + 1, n, n, 1);
			mtx_rk = H->R->Gauss_int(
				Mtx, true, false, base_cols, false, NULL,
				k + 1, n, n, 2);
			cout << "after Gauss:" << endl;
			Int_vec_print_integer_matrix_width(cout, Mtx, k + 1, n, n, 1);
			cout << "the rank is " << mtx_rk << endl;

			for (h = 0; h < n; h++) {
				if (Mtx[k * n + h]) {
					break;
				}
			}
			if (h < n) {
				if (f_v) {
					cout << "the last row is nonzero, the point is "
							"not on the line" << endl;
				}
				Inc_Mtx[i * nb_cols + j] = 0;
			}
			else {
				if (f_v) {
					cout << "the last row is zero, the point is on "
							"the line" << endl;
				}
				Inc_Mtx[i * nb_cols + j] = 1;
			}

#if 0
			if (mtx_rk == H->k) {
				Inc_Mtx[i * nb_cols + j] = 1;
			}
			else {
				Inc_Mtx[i * nb_cols + j] = 0;
			}
#endif
		}
	}
	if (f_v) {
		cout << "incidence matrix" << endl;
		Int_vec_print_integer_matrix_width(cout,
				Inc_Mtx, nb_rows, nb_cols, nb_cols, 1);
	}

	init_by_matrix(nb_rows, nb_cols, Inc_Mtx, verbose_level);
	realization_type = INCIDENCE_STRUCTURE_REALIZATION_BY_HJELMSLEV;
	
	FREE_int(Mtx);
	FREE_int(Inc_Mtx);
	FREE_int(v);
	FREE_int(base_cols);
	if (f_v) {
		cout << "incidence_structure::init_hjelmslev done" << endl;
	}
}

void incidence_structure::init_orthogonal(
		orthogonal_geometry::orthogonal *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "incidence_structure::init_orthogonal" << endl;
	}
	realization_type = INCIDENCE_STRUCTURE_REALIZATION_BY_ORTHOGONAL;
	incidence_structure::O = O;
	f_rowsums_constant = true;
	f_colsums_constant = true;
	max_r = min_r = r = O->Hyperbolic_pair->alpha;
	max_k = min_k = k = O->Quadratic_form->q + 1;
	nb_rows = O->Hyperbolic_pair->nb_points;
	nb_cols = O->Hyperbolic_pair->nb_lines;
	nb_lines_on_point = NEW_int(nb_rows);
	nb_points_on_line = NEW_int(nb_cols);
	for (i = 0; i < nb_rows; i++) {
		nb_lines_on_point[i] = max_r;
	}
	for (i = 0; i < nb_cols; i++) {
		nb_points_on_line[i] = max_k;
	}
	if (f_v) {
		cout << "incidence_structure::init_orthogonal done" << endl;
	}
}

void incidence_structure::init_by_incidences(
		int m, int n,
		int nb_inc, int *X, int verbose_level)
{
	int *M;
	int h, a;

	M = NEW_int(m * n);
	Int_vec_zero(M, m * n);
	for (h = 0; h < nb_inc; h++) {
		a = X[h];
		M[a] = 1;
	}
	init_by_matrix(m, n, M, verbose_level);
	FREE_int(M);
}

void incidence_structure::init_by_R_and_X(
		int m, int n,
		int *R, int *X, int max_r, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	int i, j, h;

	if (f_v) {
		cout << "incidence_structure::init_by_R_and_X "
				"m=" << m << " n=" << n << endl;
	}
	M = NEW_int(m * n);
	Int_vec_zero(M, m * n);
	for (i = 0; i < m; i++) {
		for (h = 0; h < R[i]; h++) {
			j = X[i * max_r + h];
			M[i * n + j] = 1;
		}
	}
	init_by_matrix(m, n, M, verbose_level - 1);
}

void incidence_structure::init_by_set_of_sets(
		data_structures::set_of_sets *SoS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	int i, j, h, m, n;

	if (f_v) {
		cout << "incidence_structure::init_by_set_of_sets" << endl;
	}
	m = SoS->nb_sets;
	n = SoS->underlying_set_size;
	M = NEW_int(m * n);
	Int_vec_zero(M, m * n);
	for (i = 0; i < m; i++) {
		for (h = 0; h < SoS->Set_size[i]; h++) {
			j = SoS->Sets[i][h];
			M[i * n + j] = 1;
		}
	}
	init_by_matrix(m, n, M, verbose_level - 1);
	if (f_v) {
		cout << "incidence_structure::init_by_set_of_sets done" << endl;
	}
}

void incidence_structure::init_by_matrix(
		int m, int n, int *M, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "incidence_structure::init_by_matrix "
				"m=" << m << " n=" << n << endl;
	}
	realization_type = INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX;
	nb_rows = m;
	nb_cols = n;
	incidence_structure::M = NEW_int(m * n);
	nb_lines_on_point = NEW_int(nb_rows);
	nb_points_on_line = NEW_int(nb_cols);
	Int_vec_copy(M, incidence_structure::M, m * n);
	init_by_matrix2(verbose_level);
	if (f_v) {
		cout << "incidence_structure::init_by_matrix done" << endl;
	}
}

void incidence_structure::init_by_matrix_as_bitmatrix(
		int m, int n,
		data_structures::bitmatrix *Bitmatrix,
		int verbose_level)
{
	int i, j;
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "incidence_structure::init_by_matrix_as_bitvector "
				"m=" << m << " n=" << n << endl;
	}
	realization_type = INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX;
	nb_rows = m;
	nb_cols = n;
	M = NEW_int(m * n);
	nb_lines_on_point = NEW_int(nb_rows);
	nb_points_on_line = NEW_int(nb_cols);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			M[i] = Bitmatrix->s_ij(i, j);
		}
	}
	init_by_matrix2(verbose_level);

	if (f_v) {
		cout << "incidence_structure::init_by_matrix_as_bitvector "
				"done" << endl;
	}
}

void incidence_structure::init_by_matrix2(int verbose_level)
{
	int i, j, h;
	int f_v = (verbose_level >= 1);
	int m = nb_rows;
	int n = nb_cols;

	if (f_v) {
		cout << "incidence_structure::init_by_matrix2" << endl;
	}
	for (i = 0; i < m; i++) {
		nb_lines_on_point[i] = 0;
		for (j = 0; j < n; j++) {
			if (M[i * n + j]) {
				nb_lines_on_point[i]++;
			}
		}
	}
	for (j = 0; j < n; j++) {
		nb_points_on_line[j] = 0;
		for (i = 0; i < m; i++) {
			if (M[i * n + j]) {
				nb_points_on_line[j]++;
			}
		}
	}
	min_r = nb_lines_on_point[0];
	max_r = nb_lines_on_point[0];
	for (i = 1; i < m; i++) {
		if (nb_lines_on_point[i] > max_r) {
			max_r = nb_lines_on_point[i];
		}
		if (nb_lines_on_point[i] < min_r) {
			min_r = nb_lines_on_point[i];
		}
	}

	if (f_v) {
		cout << "incidence_structure::init_by_matrix2 "
				"min_r=" << min_r << " max_r=" << max_r << endl;
	}

	min_k = nb_points_on_line[0];
	max_k = nb_points_on_line[0];
	for (j = 1; j < n; j++) {
		if (nb_points_on_line[j] > max_k) {
			max_k = nb_points_on_line[j];
		}
		if (nb_points_on_line[j] < min_k) {
			min_k = nb_points_on_line[j];
		}
	}

	if (f_v) {
		cout << "incidence_structure::init_by_matrix2 "
				"min_k=" << min_k << " max_k=" << max_k << endl;
	}

	if (max_r == min_r) {
		f_rowsums_constant = true;
		r = max_r;
	}
	else {
		f_rowsums_constant = false;
	}
	if (max_k == min_k) {
		f_colsums_constant = true;
		k = max_k;
	}
	else {
		f_colsums_constant = false;
	}
	lines_on_point = NEW_int(m * max_r);
	points_on_line = NEW_int(n * max_k);
	Int_vec_zero(lines_on_point, m * max_r);
	Int_vec_zero(points_on_line, n * max_k);
#if 0
	for (i = 0; i < m * max_r; i++) {
		lines_on_point[i] = 0;
	}
	for (i = 0; i < n * max_k; i++) {
		points_on_line[i] = 0;
	}
#endif
	for (i = 0; i < m; i++) {
		h = 0;
		for (j = 0; j < n; j++) {
			if (M[i * n + j]) {
				lines_on_point[i * max_r + h++] = j;
			}
		}
	}
	for (j = 0; j < n; j++) {
		h = 0;
		for (i = 0; i < m; i++) {
			if (M[i * n + j]) {
				points_on_line[j * max_k + h++] = i;
			}
		}
	}
#if 0
	if (f_vv) {
		cout << "lines_on_point:" << endl;
		print_integer_matrix_width(cout,
				lines_on_point, m, max_r, max_r, 4);
		cout << "points_on_line:" << endl;
		print_integer_matrix_width(cout,
				points_on_line, n, max_k, max_k, 4);
	}
#endif
	if (f_v) {
		cout << "incidence_structure::init_by_matrix2 done" << endl;
	}
}


int incidence_structure::nb_points()
{
	return nb_rows;
}

int incidence_structure::nb_lines()
{
	return nb_cols;
}

int incidence_structure::get_ij(int i, int j)
{
	if (realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX || 
		realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_HJELMSLEV) {
		return M[i * nb_cols + j];
	}
	else if (realization_type ==
			INCIDENCE_STRUCTURE_REALIZATION_BY_ORTHOGONAL) {
		long int p1, p2, rk;
		int *v;
		int *base_cols;

		//cout << "incidence_structure::get_ij i=" << i << " j=" << j << endl;
		v = NEW_int(3 * O->Quadratic_form->n);
		base_cols = NEW_int(O->Quadratic_form->n);
		//cout << "before O->unrank_point(v, 1, i);" << endl;
		O->Hyperbolic_pair->unrank_point(v, 1, i, 0);
		//cout << "before O->unrank_line(v, p1, p2, j);" << endl;
		O->Hyperbolic_pair->unrank_line(p1, p2, j, 0 /* verbose_level */);
		//cout << "before O->unrnk_point(v + 1 * O->n, 1, p1);" << endl;
		O->Hyperbolic_pair->unrank_point(v + 1 * O->Quadratic_form->n, 1, p1, 0);
		//cout << "before O->unrank_point(v + 2 * O->n, 1, p2);" << endl;
		O->Hyperbolic_pair->unrank_point(v + 2 * O->Quadratic_form->n, 1, p2, 0);
		rk = O->F->Linear_algebra->Gauss_simple(v, 3, O->Quadratic_form->n, base_cols, 0/* verbose_level*/);

		FREE_int(v);
		FREE_int(base_cols);

		if (rk == 2) {
			return true;
		}
		if (rk == 3) {
			return false;
		}
		cout << "incidence_structure::get_ij fatal: rk=" << rk << endl;
		exit(1);
	}
	cout << "incidence_structure::get_ij: unknown realization_type";
	exit(1);
}

int incidence_structure::get_lines_on_point(
		int *data, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int r;

	if (f_v) {
		cout << "incidence_structure::get_lines_on_point" << endl;
	}
	if (i < 0 || i >= nb_rows) {
		cout << "incidence_structure::get_lines_on_point "
				"i=" << i << " is illegal" << endl;
		exit(1);
	}
	if (realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX || 
		realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_HJELMSLEV) {
		int h;
		for (h = 0; h < nb_lines_on_point[i]; h++) {
			data[h] = lines_on_point[i * max_r + h];
		}
		r = nb_lines_on_point[i];
	}
	else if (realization_type ==
			INCIDENCE_STRUCTURE_REALIZATION_BY_ORTHOGONAL) {
		O->lines_on_point_by_line_rank_must_fit_into_int(
				i, data, 0/*verbose_level - 2*/);
		r = O->Hyperbolic_pair->alpha;
	}
	else if (realization_type ==
			INCIDENCE_STRUCTURE_REALIZATION_BY_PROJECTIVE_SPACE) {
		long int *Data;
		int h;

		Data = NEW_lint(P->Subspaces->r);
		P->Subspaces->create_lines_on_point(
				i, Data, verbose_level);
		for (h = 0; h < P->Subspaces->r; h++) {
			data[h] = Data[h];
		}
		FREE_lint(Data);
		r = P->Subspaces->r;
	}
	else {
		cout << "incidence_structure::get_lines_on_point "
			"fatal: unknown realization_type";
		exit(1);
	}
	if (f_v) {
		cout << "incidence_structure::get_lines_on_point pt = " << i << " : ";
		Int_vec_print(cout, data, r);
		cout << endl;
		cout << "incidence_structure::get_lines_on_point done" << endl;
	}
	return r;
}

int incidence_structure::get_points_on_line(
		int *data, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int k;

	if (f_v) {
		cout << "incidence_structure::get_points_on_line" << endl;
	}
	if (realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX || 
		realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_HJELMSLEV) {
		int h;
		for (h = 0; h < nb_points_on_line[j]; h++) {
			data[h] = points_on_line[j * max_k + h];
		}
		k = nb_points_on_line[j];
	}
	else if (realization_type ==
			INCIDENCE_STRUCTURE_REALIZATION_BY_ORTHOGONAL) {
		long int *Data;
		int h;

		Data = NEW_lint(nb_points_on_line[j]);
		O->points_on_line_by_line_rank(j, Data, 0/* verbose_level - 2*/);
		for (h = 0; h < nb_points_on_line[j]; h++) {
			data[h] = Data[h];
		}
		FREE_lint(Data);
		k = O->Quadratic_form->q + 1;
	}
	else if (realization_type ==
			INCIDENCE_STRUCTURE_REALIZATION_BY_PROJECTIVE_SPACE) {
		long int *Data;
		int h;

		Data = NEW_lint(P->Subspaces->k);
		P->Subspaces->create_points_on_line(j, Data, 0 /*verbose_level*/);
		for (h = 0; h < P->Subspaces->k; h++) {
			data[h] = Data[h];
		}
		FREE_lint(Data);
		k = P->Subspaces->k;
	}
	else {
		cout << "incidence_structure::get_points_on_line "
				"fatal: unknown realization_type";
		exit(1);
	}
	if (f_v) {
		cout << "incidence_structure::get_points_on_line "
				"line = " << j << " : ";
		Int_vec_print(cout, data, k);
		cout << endl;
		cout << "incidence_structure::get_points_on_line done" << endl;
	}
	return k;
}

int incidence_structure::get_nb_inc()
{
	if (f_rowsums_constant) {
		return nb_rows * r;
	}
	else {
		int i, j, nb = 0;
		
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (get_ij(i, j)) {
					nb++;
				}
			}
		}
		return nb;
	}
}

void incidence_structure::save_inc_file(char *fname)
{
	int i, j, nb_inc;

	nb_inc = get_nb_inc();
	
	ofstream f(fname);
	f << nb_rows << " " << nb_cols << " " << nb_inc << endl;
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (get_ij(i, j)) {
				f << i * nb_cols + j << " ";
			}
		}
	}
	f << "0" << endl; // ago not known
	f << "-1" << endl;
}

void incidence_structure::save_row_by_row_file(char *fname)
{
    int i, j; //, nb_inc;
    int w;
    number_theory::number_theory_domain NT;

	//nb_inc = get_nb_inc();
	
	if (!f_rowsums_constant) {
		cout << "incidence_structure::save_row_by_row_file "
				"rowsums are not constant" << endl;
		exit(1);
	}
	ofstream f(fname);
	w = NT.int_log10(nb_cols) + 1;
	f << nb_rows << " " << nb_cols << " " << r << endl;
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < r; j++) {
			f << setw(w) << lines_on_point[i * r + j] << " ";
		}
		f << endl;
	}
	f << "-1" << endl;
}


void incidence_structure::print(std::ostream &ost)
{
	int i, j;

	if (nb_cols == 0) {
		cout << "incidence_structure::print nb_cols == 0" << endl;
		return;
	}
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (get_ij(i, j)) {
				ost << "1";
			}
			else {
				ost << ".";
			}
		}
		ost << endl;
	}
}


data_structures::bitvector *incidence_structure::encode_as_bitvector()
{
	int i, j, a;
	data_structures::bitvector *B;

	B = NEW_OBJECT(data_structures::bitvector);
	B->allocate(nb_rows * nb_cols);

	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (M[i * nb_cols + j]) {
				a = i * nb_cols + j;
				//bitvector_set_bit(bitvec, a);
				B->m_i(a, 1);
			}
		}
	}
	return B;
}


incidence_structure *incidence_structure::apply_canonical_labeling(
		long int *canonical_labeling, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Incma_out;
	int i, j, ii, jj;

	if (f_v) {
		cout << "incidence_structure::apply_canonical_labeling" << endl;
	}

	if (f_vv) {
		cout << "incidence_structure::apply_canonical_labeling labeling:" << endl;
		Lint_vec_print(cout, canonical_labeling, nb_rows + nb_cols);
		cout << endl;
	}

	Incma_out = NEW_int(nb_rows * nb_cols);
	//Orbiter->Int_vec.zero(Incma_out, nb_rows * nb_cols);
	for (i = 0; i < nb_rows; i++) {
		ii = canonical_labeling[i];

		for (j = 0; j < nb_cols; j++) {
			jj = canonical_labeling[nb_rows + j] - nb_rows;

			//cout << "i=" << i << " j=" << j << " ii=" << ii
			//<< " jj=" << jj << endl;
			Incma_out[i * nb_cols + j] = M[ii * nb_cols + jj];
		}
	}


	incidence_structure *Inc_out;

	Inc_out = NEW_OBJECT(incidence_structure);

	Inc_out->init_by_matrix(nb_rows, nb_cols, Incma_out, verbose_level);

	FREE_int(Incma_out);
	if (f_v) {
		cout << "incidence_structure::apply_canonical_labeling done" << endl;
	}
	return Inc_out;
}

void incidence_structure::save_as_csv(
		std::string &fname_csv, int verbose_level)
{
	orbiter_kernel_system::file_io Fio;

	Fio.Csv_file_support->int_matrix_write_csv(
			fname_csv, M, nb_rows, nb_cols);
}

#if 0
void incidence_structure::save_as_Levi_graph(std::string &fname_bin,
		int f_point_labels, long int *point_labels,
		int verbose_level)
{
	file_io Fio;
	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);

	CG->create_Levi_graph_from_incidence_matrix(
			M, nb_rows, nb_cols,
			f_point_labels, point_labels,
			verbose_level);
	CG->save(fname_bin, verbose_level);

	FREE_OBJECT(CG);
}
#endif

void incidence_structure::init_large_set(
		long int *blocks,
		int N_points, int design_b, int design_k,
		int partition_class_size,
		int *&partition, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "incidence_structure::init_large_set" << endl;
	}

	int *block;
	int *Incma;
	int nb_classes;
	int nb_rows;
	int nb_cols;
	int N;
	int u, i, j, t, a;
	combinatorics::combinatorics_domain Combi;


	nb_classes = design_b / partition_class_size;
	nb_rows = N_points + nb_classes;
	nb_cols = design_b + 1;
	N = nb_rows + nb_cols;
	if (f_v) {
		cout << "incidence_structure::init_large_set "
				"nb_rows=" << nb_rows << " nb_cols=" << nb_cols << " N=" << N << endl;
	}

	Incma = NEW_int(nb_rows * nb_cols);
	Int_vec_zero(Incma, nb_rows * nb_cols);
	block = NEW_int(design_k);

	for (u = 0; u < nb_classes; u++) {
		for (j = 0; j < partition_class_size; j++) {
			a = blocks[u * partition_class_size + j];
			Combi.unrank_k_subset(a, block, N_points, design_k);
			for (t = 0; t < design_k; t++) {
				i = block[t];
				Incma[i * nb_cols + u * partition_class_size + j] = 1;
			}
			Incma[(N_points + u) * nb_cols + u * partition_class_size + j] = 1;
		}
		Incma[(N_points + u) * nb_cols + nb_cols - 1] = 1;
	}

	init_by_matrix(nb_rows, nb_cols, Incma, verbose_level);
	if (f_v) {
		Int_matrix_print(Incma, nb_rows, nb_cols);
	}


	partition = NEW_int(N);
	for (i = 0; i < N; i++) {
		partition[i] = 1;
	}
	partition[N_points - 1] = 0;
	partition[nb_rows - 1] = 0;
	partition[nb_rows + design_b - 1] = 0;
	partition[nb_rows + nb_cols - 1] = 0;


	if (f_v) {
		data_structures::tally T;

		T.init(partition, N, false, 0);
		T.print_array_tex(cout, true);
	}


	FREE_int(Incma);
	FREE_int(block);

	if (f_v) {
		cout << "incidence_structure::init_large_set done" << endl;
	}
}


#if 0
void incidence_structure::latex_it(
		std::ostream &ost,
		data_structures::partitionstack &P)
{

	cout << "incidence_structure::latex_it" << endl;
	cout << "currently disabled" << endl;
	exit(1);

#if 0
	int nb_V, nb_B;
	int *Vi, *Bj;
	int *R;
	int *X;
	int f_v = true;
	latex_interface L;

	rearrange(Vi, nb_V, Bj, nb_B, R, X, P);

	L.incma_latex(ost,
		nb_points(), nb_lines(), 
		nb_V, nb_B, Vi, Bj, 
		R, X, max_r);

	FREE_int(Vi);
	FREE_int(Bj);
	FREE_int(R);
	FREE_int(X);
#endif


}

void incidence_structure::rearrange(
		int *&Vi, int &nb_V,
	int *&Bj, int &nb_B, int *&R, int *&X,
	data_structures::partitionstack &P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "incidence_structure::rearrange" << endl;
	}
	int *row_classes;
	int nb_row_classes;
	int *col_classes;
	int nb_col_classes;
	//int *Vi, *Bj;
	int *row_perm;
	int *row_perm_inv;
	int *col_perm;
	int *col_perm_inv;
	int i, j, ii, jj, c, a, h;
	//int *R;
	//int *X;
	data_structures::sorting Sorting;

	row_classes = NEW_int(nb_points() + nb_lines());
	col_classes = NEW_int(nb_points() + nb_lines());
	P.get_row_and_col_classes(row_classes, nb_row_classes, 
		col_classes, nb_col_classes, 0 /* verbose_level */);
	//ost << "nb_row_classes = " << nb_row_classes << endl;
	//ost << "nb_col_classes = " << nb_col_classes << endl;

	nb_V = nb_row_classes;
	nb_B = nb_col_classes;
	Vi = NEW_int(nb_row_classes);
	Bj = NEW_int(nb_col_classes);
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		a = P.cellSize[c];
		Vi[i] = a;
	}
	for (i = 0; i < nb_col_classes; i++) {
		c = col_classes[i];
		a = P.cellSize[c];
		Bj[i] = a;
	}

	row_perm = NEW_int(nb_points());
	row_perm_inv = NEW_int(nb_points());
	col_perm = NEW_int(nb_lines());
	col_perm_inv = NEW_int(nb_lines());
	
	P.get_row_and_col_permutation(
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		row_perm, row_perm_inv, 
		col_perm, col_perm_inv,
		verbose_level);
	
#if 0
	cout << "row_perm:" << endl;
	int_vec_print(cout, row_perm, nb_points());
	cout << endl;
	cout << "col_perm:" << endl;
	int_vec_print(cout, col_perm, nb_lines());
	cout << endl;
#endif

	R = NEW_int(nb_points());
	X = NEW_int(nb_points() * max_r);

	for (i = 0; i < nb_points(); i++) {
		ii = row_perm_inv[i];
		R[i] = nb_lines_on_point[ii];
		for (h = 0; h < nb_lines_on_point[ii]; h++) {
			jj = lines_on_point[ii * max_r + h];
			j = col_perm[jj];
			X[i * max_r + h] = j;
		}
		Sorting.int_vec_heapsort(X + i * max_r, nb_lines_on_point[ii]);
	}
	


	FREE_int(row_classes);
	FREE_int(col_classes);
	//FREE_int(Vi);
	//FREE_int(Bj);
	FREE_int(row_perm);
	FREE_int(row_perm_inv);
	FREE_int(col_perm);
	FREE_int(col_perm_inv);
	//FREE_int(R);
	//FREE_int(X);
	if (f_v) {
		cout << "incidence_structure::rearrange done" << endl;
	}
}



void incidence_structure::decomposition_print_tex(
		std::ostream &ost,
		data_structures::partitionstack &PStack,
		int f_row_tactical, int f_col_tactical,
	int f_detailed,
	int f_local_coordinates, int verbose_level)
{
	int *row_classes, *row_class_inv, nb_row_classes;
	int *col_classes, *col_class_inv, nb_col_classes;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "incidence_structure::decomposition_print_tex" << endl;
	}
	if (f_v) {
		cout << "incidence_structure::decomposition_print_tex "
				"before get decomposition" << endl;
	}
	PStack.allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		verbose_level);

	ost << "\\subsection*{Decomposition}" << endl;
	PStack.print_classes_of_decomposition_tex(ost, row_classes, nb_row_classes,
		col_classes, nb_col_classes);
	
	if (f_row_tactical) {
		int *row_scheme;
		row_scheme = NEW_int(nb_row_classes * nb_col_classes);
		get_row_decomposition_scheme(PStack, 
			row_classes, row_class_inv, nb_row_classes,
			col_classes, col_class_inv, nb_col_classes, 
			row_scheme, 0);
		if (f_v) {
			cout << "incidence_structure::decomposition_print_tex row_scheme:" << endl;
			cout << "row_scheme:" << endl;
			PStack.print_decomposition_scheme(cout, 
				row_classes, nb_row_classes,
				col_classes, nb_col_classes, 
				row_scheme, -1, -1);
		}

		if (f_detailed) {
			ost << "\\subsection*{Incidences by row-scheme}" << endl;
			print_row_tactical_decomposition_scheme_incidences_tex(
				PStack, 
				ost, false /* f_enter_math_mode */, 
				row_classes, row_class_inv, nb_row_classes,
				col_classes, col_class_inv, nb_col_classes, 
				f_local_coordinates, 0 /*verbose_level*/);
		}
		FREE_int(row_scheme);
	}
	if (f_col_tactical) {
		int *col_scheme;
		col_scheme = NEW_int(nb_row_classes * nb_col_classes);
		get_col_decomposition_scheme(PStack, 
			row_classes, row_class_inv, nb_row_classes,
			col_classes, col_class_inv, nb_col_classes, 
			col_scheme, 0);
		if (f_v) {
			cout << "incidence_structure::decomposition_"
					"print_tex col_scheme:" << endl;
			cout << "col_scheme:" << endl;
			PStack.print_decomposition_scheme(cout, 
				row_classes, nb_row_classes,
				col_classes, nb_col_classes, 
				col_scheme, -1, -1);
		}

		if (f_detailed) {
			ost << "\\subsection*{Incidences by col-scheme}" << endl;
			print_col_tactical_decomposition_scheme_incidences_tex(
				PStack, 
				ost, false /* f_enter_math_mode */, 
				row_classes, row_class_inv, nb_row_classes,
				col_classes, col_class_inv, nb_col_classes, 
				f_local_coordinates, 0 /*verbose_level*/);
		}
		FREE_int(col_scheme);
	}


	


	FREE_int(row_classes);
	FREE_int(row_class_inv);
	FREE_int(col_classes);
	FREE_int(col_class_inv);
	if (f_v) {
		cout << "incidence_structure::decomposition_print_tex done" << endl;
	}
}



void incidence_structure::do_tdo_high_level(
		data_structures::partitionstack &S,
	int f_tdo_steps, int f_tdo_depth, int tdo_depth, 
	int f_write_tdo_files, int f_pic, 
	int f_include_tdo_scheme, int f_include_tdo_extra,
	int f_write_tdo_class_files,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int TDO_depth;

	if (f_v) {
		cout << "incidence_structure::do_tdo_high_level" << endl;
	}

	if (f_tdo_depth) {
		TDO_depth = tdo_depth;
	}
	else {
		TDO_depth = nb_points() + nb_lines();
	}

	if (f_tdo_steps) {
		compute_tdo_stepwise(S, 
			TDO_depth, 
			f_write_tdo_files, 
			f_pic, 
			f_include_tdo_scheme, 
			f_include_tdo_extra, 
			verbose_level);
	}
	else {
		compute_tdo(S, 
			f_write_tdo_files, 
			f_pic, 
			f_include_tdo_scheme, 
			verbose_level);
	}
	if (f_write_tdo_class_files) {
		int *row_classes, *row_class_inv, nb_row_classes;
		int *col_classes, *col_class_inv, nb_col_classes;
		int i;

		S.allocate_and_get_decomposition(
			row_classes, row_class_inv, nb_row_classes,
			col_classes, col_class_inv, nb_col_classes, 
			verbose_level - 1);

		for (i = 0; i < nb_row_classes; i++) {
			string fname;
			fname = label + "_TDO_point_class_" + std::to_string(i) + ".txt";
			S.write_cell_to_file_points_or_lines(
					row_classes[i], fname, verbose_level - 1);
		}
		for (i = 0; i < nb_col_classes; i++) {
			string fname;
			fname = label + "_TDO_line_class_" + std::to_string(i) + ".txt";
			S.write_cell_to_file_points_or_lines(
					col_classes[i], fname, verbose_level - 1);
		}
		FREE_int(row_classes);
		FREE_int(row_class_inv);
		FREE_int(col_classes);
		FREE_int(col_class_inv);
	}
}





void incidence_structure::compute_tdo(
		data_structures::partitionstack &S,
	int f_write_tdo_files, 
	int f_pic, 
	int f_include_tdo_scheme, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	string fname;
	int f_list_incidences = false;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "incidence_structure_compute_tdo" << endl;
	}
	
	int N;
	

	if (f_v) {
		cout << "incidence_structure_compute_tdo "
				"initial partition:" << endl;
		S.print(cout);
		cout << endl;
	}
	N = nb_points() + nb_lines();

	int TDO_depth = N;
	int f_labeled = true;

	compute_TDO_safe(S, TDO_depth, verbose_level - 1);

	if (f_vv) {
		print_partitioned(cout, S, f_labeled);
	}

	if (f_v) {
		cout << "TDO:" << endl;
		if (TDO_depth < N) {
			if (EVEN(TDO_depth)) {
				get_and_print_row_decomposition_scheme(
						S, f_list_incidences, false, verbose_level);
			}
			else {
				get_and_print_col_decomposition_scheme(
						S, f_list_incidences, false, verbose_level);
			}
		}
		else {
			get_and_print_decomposition_schemes(S);
			S.print_classes_points_and_lines(cout);
		}
	}

	if (f_write_tdo_files) {
		fname = label + "_tdo_scheme.tex";
		{
			ofstream fp(fname);

				//fp << "$$" << endl;
				get_and_print_tactical_decomposition_scheme_tex(
						fp, true /* f_enter_math */, S);
				//fp << "$$" << endl;
		}
		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}

		fname = label + "_tdo.tex";
		{
			ofstream fp(fname);

			if (f_include_tdo_scheme) {
				fp << "$\\begin{array}{c}" << endl;
				if (f_pic) {
					latex_it(fp, S);
					fp << "\\\\" << endl;
				}
				get_and_print_tactical_decomposition_scheme_tex(
					fp, false /* f_enter_math */, S);
				fp << "\\end{array}$" << endl;
			}
			else {
				latex_it(fp, S);
			}
		}
		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}

}

void incidence_structure::compute_tdo_stepwise(
		data_structures::partitionstack &S,
	int TDO_depth, 
	int f_write_tdo_files, 
	int f_pic, 
	int f_include_tdo_scheme, 
	int f_include_extra, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	string fname;
	string fname_pic;
	string fname_scheme;
	string fname_extra;
	int step, f_refine, f_refine_prev, f_done;
	int f_local_coordinates = false;
	int f_list_incidences = false;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "incidence_structure::compute_tdo_stepwise" << endl;
	}
	

	compute_TDO_safe_first(S, 
		TDO_depth, step, f_refine, f_refine_prev, verbose_level);

	S.sort_cells();

	f_done = false;
	while (true) {
		if (f_v) {
			cout << "incidence_structure::compute_tdo_stepwise "
					"TDO step=" << step << " ht=" << S.ht << endl;
			if (step == 0) {
				print_non_tactical_decomposition_scheme_tex(
						cout, false /* f_enter_math */, S);
			}
			else if (EVEN(step - 1)) {

				get_and_print_col_decomposition_scheme(
						S, f_list_incidences, false, verbose_level);
				get_and_print_column_tactical_decomposition_scheme_tex(
					cout, true /* f_enter_math */,
					false /* f_print_subscripts */, S);
			}
			else {

				get_and_print_row_decomposition_scheme(
						S, f_list_incidences, false, verbose_level);
				get_and_print_row_tactical_decomposition_scheme_tex(
					cout, true /* f_enter_math */,
					false /* f_print_subscripts */, S);
			}
			S.print_classes_points_and_lines(cout);
		}
		if (f_write_tdo_files) {

			fname = label + "_tdo_step_" + std::to_string(step) + ".tex";

			fname_pic = label + "_tdo_step_" + std::to_string(step) + "_pic.tex";

			fname_scheme = label + "_tdo_step_" + std::to_string(step) + "_scheme.tex";

			fname_extra = label + "_tdo_step_" + std::to_string(step) + "_extra.tex";

			{
				ofstream fp(fname);
				ofstream fp_pic(fname_pic);
				ofstream fp_scheme(fname_scheme);
				ofstream fp_extra(fname_extra);

				if (f_include_tdo_scheme) {
					fp << "\\subsection*{Step $" << step
							<< "$, Height $" << S.ht << "$}" << endl;
					//fp << "$ht=" << S.ht << "$\\\\" << endl;
					//fp << "\\bigskip" << endl;
					fp << "$\\begin{array}{c}" << endl;

					if (f_pic) {
						fp << "\\input " << fname_pic << endl;
						latex_it(fp_pic, S);
						fp << "\\\\" << endl;
					}

					fp << "\\input " << fname_scheme << endl;
					if (step == 0) {
						print_non_tactical_decomposition_scheme_tex(
							fp_scheme, false /* f_enter_math */, S);
					}
					else if (EVEN(step - 1)) {
						get_and_print_column_tactical_decomposition_scheme_tex(
							fp_scheme, false /* f_enter_math */,
							false /* f_print_subscripts */, S);
					}
					else {
						get_and_print_row_tactical_decomposition_scheme_tex(
							fp_scheme, false /* f_enter_math */,
							false /* f_print_subscripts */, S);
					}
					fp << "\\end{array}$" << endl;

					if (f_include_extra) {
						if (step == 0) {
							decomposition_print_tex(fp_extra, S,
									false, false, false,
									f_local_coordinates, verbose_level);
							fp << "\\input " << fname_extra << endl;
						}
						else if (EVEN(step - 1)) {
							decomposition_print_tex(fp_extra, S,
									false, true, true,
									f_local_coordinates, verbose_level);
							fp << "\\input " << fname_extra << endl;
							//PStack.print_classes_points_and_lines(cout);
						}
						else {
							decomposition_print_tex(fp_extra, S,
									true, false, true,
									f_local_coordinates, verbose_level);
							fp << "\\input " << fname_extra << endl;
						}
					}
				}
				else {
					latex_it(fp_pic, S);
				}
			}
			if (f_v) {
				cout << "written file " << fname << " of size "
						<< Fio.file_size(fname) << endl;
				cout << "written file " << fname_pic << " of size "
						<< Fio.file_size(fname_pic) << endl;
				cout << "written file " << fname_scheme << " of size "
						<< Fio.file_size(fname_scheme) << endl;
				cout << "written file " << fname_extra << " of size "
						<< Fio.file_size(fname_extra) << endl;
			}
		}
		if (f_done) {
			break;
		}

		f_done = compute_TDO_safe_next(S, 
			TDO_depth, step, f_refine, f_refine_prev, verbose_level);
		S.sort_cells();
	}



	if (f_vv) {
		int f_labeled = false;
		
		print_partitioned(cout, S, f_labeled);
	}

	if (f_v) {
		cout << "TDO:" << endl;
		if (TDO_depth < nb_points() + nb_lines()) {
			if (EVEN(TDO_depth)) {
				get_and_print_row_decomposition_scheme(S,
						f_list_incidences, false, verbose_level);
			}
			else {
				get_and_print_col_decomposition_scheme(S,
						f_list_incidences, false, verbose_level);
			}
		}
		else {
			get_and_print_decomposition_schemes(S);
			S.print_classes_points_and_lines(cout);
		}
	}

	if (f_write_tdo_files) {
		fname = label + "_tdo.tex";

		fname_pic = label + "_tdo_pic.tex";

		fname_scheme = label + "_tdo_scheme.tex";

		{
			ofstream fp(fname);
			ofstream fp_pic(fname_pic);
			ofstream fp_scheme(fname_scheme);

			if (f_include_tdo_scheme) {
				fp << "\\subsection*{The TDO at Height $"
						<< S.ht << "$}" << endl;
				fp << "$\\begin{array}{c}" << endl;
				if (f_pic) {
					fp << "\\input " << fname_pic << endl;
					latex_it(fp_pic, S);
					fp << "\\\\" << endl;
				}
				fp << "\\input " << fname_scheme << endl;
				get_and_print_tactical_decomposition_scheme_tex(
					fp_scheme, false /* f_enter_math */, S);
				fp << "\\end{array}$" << endl;
			}
			else {
				latex_it(fp_pic, S);
			}
		}
		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
			cout << "written file " << fname_pic << " of size "
					<< Fio.file_size(fname_pic) << endl;
			cout << "written file " << fname_scheme << " of size "
					<< Fio.file_size(fname_scheme) << endl;
		}
	}

}

void incidence_structure::init_partitionstack_trivial(
		data_structures::partitionstack *S,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N;


	if (f_v) {
		cout << "incidence_structure::init_partitionstack_trivial" << endl;
	}
	N = nb_points() + nb_lines();
	
	S->allocate(N, 0);

	// split off the column class:
	S->subset_contiguous(nb_points(), nb_lines());
	S->split_cell(0);

}

#endif





}}}



