// incidence_structure.C
//
// Anton Betten
//
// June 20, 2010



#include "galois.h"


// ####################################################################################
// incidence_structure
// ####################################################################################

INT incidence_structure::cntr_new = 0;
INT incidence_structure::cntr_objects = 0;
INT incidence_structure::f_debug_memory = FALSE;

void *incidence_structure::operator new(size_t bytes)
{
	cntr_new++;
	cntr_objects++;
	if (f_debug_memory) {
		cout << "incidence_structure::operator new bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void *incidence_structure::operator new[](size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(incidence_structure);
	cntr_new++;
	cntr_objects += n;
	if (f_debug_memory) {
		cout << "incidence_structure::operator new[] n=" << n 
			<< " bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void incidence_structure::operator delete(void *ptr, size_t bytes)
{
	if (f_debug_memory) {
		cout << "incidence_structure::operator delete bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects--;
	return ::free(ptr);
}

void incidence_structure::operator delete[](void *ptr, size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(incidence_structure);
	if (f_debug_memory) {
		cout << "incidence_structure::operator delete[] n=" << n 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects -= n;
	return ::free(ptr);
}

incidence_structure::incidence_structure()
{
	null();
}

incidence_structure::~incidence_structure()
{
	freeself();
}

void incidence_structure::null()
{
	label[0] = 0;
	M = NULL;
	O = NULL;
	H = NULL;
	nb_lines_on_point = NULL;
	nb_points_on_line = NULL;
	lines_on_point = NULL;
	points_on_line = NULL;
}

void incidence_structure::freeself()
{
	if (M) {
		FREE_INT(M);
		}
	if (O) {
		// do not destroy
		}
	if (H) {
		// do not destroy
		}
	if (nb_lines_on_point) {
		FREE_INT(nb_lines_on_point);
		}
	if (nb_points_on_line) {
		FREE_INT(nb_points_on_line);
		}
	if (lines_on_point) {
		FREE_INT(lines_on_point);
		}
	if (points_on_line) {
		FREE_INT(points_on_line);
		}
	null();
}

void incidence_structure::check_point_pairs(INT verbose_level)
{
	INT *Mtx;
	INT i, j, nb;
	INT *Lines;

	Lines = NEW_INT(nb_cols);
	Mtx = NEW_INT(nb_rows * nb_rows);
	for (i = 0; i < nb_rows; i++) {
		for (j = i; j < nb_rows; j++) {
			nb = lines_through_two_points(Lines, i, j, 0);
			Mtx[i * nb_rows + j] = nb;
			Mtx[j * nb_rows + i] = nb;
			}
		}
	cout << "nb of lines through two points:" << endl;
	print_integer_matrix_width(cout, Mtx, nb_rows, nb_rows, nb_rows, 1);
	
	FREE_INT(Lines);
}

INT incidence_structure::lines_through_two_points(INT *lines, INT p1, INT p2, INT verbose_level)
{
	INT h1, h2, l1, l2, nb;
	
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

void incidence_structure::init_hjelmslev(hjelmslev *H, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "incidence_structure::init_hjelmslev" << endl;
		}
	realization_type = INCIDENCE_STRUCTURE_REALIZATION_BY_HJELMSLEV;
	incidence_structure::H = H;
	f_rowsums_constant = TRUE;
	f_colsums_constant = TRUE;
	//max_r = min_r = r = O->alpha;
	//max_k = min_k = k = O->q + 1;
	nb_rows = nb_PHG_elements(H->k, *H->R);
	nb_cols = H->number_of_submodules();
	if (f_v) {
		cout << "nb_rows=" << nb_rows << endl;
		cout << "nb_cols=" << nb_cols << endl;
		}
	INT *Mtx;
	INT *Inc_Mtx;
	INT *v;
	INT *base_cols;
	INT mtx_rk;
	INT i, j, h, k, n;

	k = H->k;
	n = H->n;
	Mtx = NEW_INT((k + 1) * n);
	Inc_Mtx = NEW_INT(nb_rows * nb_cols);
	v = NEW_INT(k);
	base_cols = NEW_INT(n);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			cout << "i=" << i << " j=" << j << endl;
			PHG_element_unrank(*H->R, Mtx, 1, n, i);
			H->unrank_INT(Mtx + n, j, 0);
			print_integer_matrix_width(cout, Mtx, k + 1, n, n, 1);
			mtx_rk = H->R->Gauss_INT(Mtx, TRUE, FALSE, base_cols, FALSE, NULL, 
				k + 1, n, n, 2);
			cout << "after Gauss:" << endl;
			print_integer_matrix_width(cout, Mtx, k + 1, n, n, 1);
			cout << "the rank is " << mtx_rk << endl;

			for (h = 0; h < n; h++) {
				if (Mtx[k * n + h])
					break;
				}
			if (h < n) {
				if (f_v) {
					cout << "the last row is nonzero, the point is not on the line" << endl;
					}
				Inc_Mtx[i * nb_cols + j] = 0;
				}
			else {
				if (f_v) {
					cout << "the last row is zero, the point is on the line" << endl;
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
		print_integer_matrix_width(cout, Inc_Mtx, nb_rows, nb_cols, nb_cols, 1);
		}

	init_by_matrix(nb_rows, nb_cols, Inc_Mtx, verbose_level);
	realization_type = INCIDENCE_STRUCTURE_REALIZATION_BY_HJELMSLEV;
	
	FREE_INT(Mtx);
	FREE_INT(Inc_Mtx);
	FREE_INT(v);
	FREE_INT(base_cols);
	if (f_v) {
		cout << "incidence_structure::init_hjelmslev done" << endl;
		}
}

void incidence_structure::init_orthogonal(orthogonal *O, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i;

	if (f_v) {
		cout << "incidence_structure::init_orthogonal" << endl;
		}
	realization_type = INCIDENCE_STRUCTURE_REALIZATION_BY_ORTHOGONAL;
	incidence_structure::O = O;
	f_rowsums_constant = TRUE;
	f_colsums_constant = TRUE;
	max_r = min_r = r = O->alpha;
	max_k = min_k = k = O->q + 1;
	nb_rows = O->nb_points;
	nb_cols = O->nb_lines;
	nb_lines_on_point = NEW_INT(nb_rows);
	nb_points_on_line = NEW_INT(nb_cols);
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

void incidence_structure::init_by_incidences(INT m, INT n, INT nb_inc, INT *X, INT verbose_level)
{
	INT *M;
	INT h, i, a;

	M = NEW_INT(m * n);
	for (i = 0; i < m * n; i++) {
		M[i] = 0;
		}
	for (h = 0; h < nb_inc; h++) {
		a = X[h];
		M[a] = 1;
		}
	init_by_matrix(m, n, M, verbose_level);
	FREE_INT(M);
}

void incidence_structure::init_by_R_and_X(INT m, INT n, INT *R, INT *X, INT max_r, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *M;
	INT i, j, h;

	if (f_v) {
		cout << "incidence_structure::init_by_R_and_X m=" << m << " n=" << n << endl;
		}
	M = NEW_INT(m * n);
	for (i = 0; i < m * n; i++) {
		M[i] = 0;
		}
	for (i = 0; i < m; i++) {
		for (h = 0; h < R[i]; h++) {
			j = X[i * max_r + h];
			M[i * n + j] = 1;
			}
		}
	init_by_matrix(m, n, M, verbose_level - 1);
}

void incidence_structure::init_by_set_of_sets(set_of_sets *SoS, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *M;
	INT i, j, h, m, n;

	if (f_v) {
		cout << "incidence_structure::init_by_set_of_sets" << endl;
		}
	m = SoS->nb_sets;
	n = SoS->underlying_set_size;
	M = NEW_INT(m * n);
	INT_vec_zero(M, m * n);
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

void incidence_structure::init_by_matrix(INT m, INT n, INT *M, INT verbose_level)
{
	INT i;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "incidence_structure::init_by_matrix m=" << m << " n=" << n << endl;
		}
	realization_type = INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX;
	nb_rows = m;
	nb_cols = n;
	incidence_structure::M = NEW_INT(m * n);
	nb_lines_on_point = NEW_INT(nb_rows);
	nb_points_on_line = NEW_INT(nb_cols);
	for (i = 0; i < m * n; i++) {
		incidence_structure::M[i] = M[i];
		}
	init_by_matrix2(verbose_level);
	if (f_v) {
		cout << "incidence_structure::init_by_matrix done" << endl;
		}
}

void incidence_structure::init_by_matrix_as_bitvector(INT m, INT n, UBYTE *M_bitvec, INT verbose_level)
{
	INT i, j;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "incidence_structure::init_by_matrix_as_bitvector m=" << m << " n=" << n << endl;
		}
	realization_type = INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX;
	nb_rows = m;
	nb_cols = n;
	incidence_structure::M = NEW_INT(m * n);
	nb_lines_on_point = NEW_INT(nb_rows);
	nb_points_on_line = NEW_INT(nb_cols);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			incidence_structure::M[i] = bitvector_s_i(M_bitvec, i * n + j);
			}
		}
	init_by_matrix2(verbose_level);

	if (f_v) {
		cout << "incidence_structure::init_by_matrix_as_bitvector done" << endl;
		}
}

void incidence_structure::init_by_matrix2(INT verbose_level)
{
	INT i, j, h;
	INT f_v = (verbose_level >= 1);
	INT m = nb_rows;
	INT n = nb_cols;

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
		cout << "incidence_structure::init_by_matrix2 min_r=" << min_r << " max_r=" << max_r << endl;
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
		cout << "incidence_structure::init_by_matrix2 min_k=" << min_k << " max_k=" << max_k << endl;
		}

	if (max_r == min_r) {
		f_rowsums_constant = TRUE;
		r = max_r;
		}
	else {
		f_rowsums_constant = FALSE;
		}
	if (max_k == min_k) {
		f_colsums_constant = TRUE;
		k = max_k;
		}
	else {
		f_colsums_constant = FALSE;
		}
	lines_on_point = NEW_INT(m * max_r);
	points_on_line = NEW_INT(n * max_k);
	for (i = 0; i < m * max_r; i++) {
		lines_on_point[i] = 0;
		}
	for (i = 0; i < n * max_k; i++) {
		points_on_line[i] = 0;
		}
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
		print_integer_matrix_width(cout, lines_on_point, m, max_r, max_r, 4);
		cout << "points_on_line:" << endl;
		print_integer_matrix_width(cout, points_on_line, n, max_k, max_k, 4);
		}
#endif
	if (f_v) {
		cout << "incidence_structure::init_by_matrix2 done" << endl;
		}
}


INT incidence_structure::nb_points()
{
	return nb_rows;
}

INT incidence_structure::nb_lines()
{
	return nb_cols;
}

INT incidence_structure::get_ij(INT i, INT j)
{
	if (realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX || 
		realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_HJELMSLEV) {
		return M[i * nb_cols + j];
		}
	else if (realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_ORTHOGONAL) {
		INT p1, p2, rk;
		INT *v;
		INT *base_cols;

		//cout << "incidence_structure::get_ij i=" << i << " j=" << j << endl;
		v = NEW_INT(3 * O->n);
		base_cols = NEW_INT(O->n);
		//cout << "before O->unrank_point(v, 1, i);" << endl;
		O->unrank_point(v, 1, i, 0);
		//cout << "before O->unrank_line(v, p1, p2, j);" << endl;
		O->unrank_line(p1, p2, j, 0 /* verbose_level */);
		//cout << "before O->unrank_point(v + 1 * O->n, 1, p1);" << endl;
		O->unrank_point(v + 1 * O->n, 1, p1, 0);
		//cout << "before O->unrank_point(v + 2 * O->n, 1, p2);" << endl;
		O->unrank_point(v + 2 * O->n, 1, p2, 0);
		rk = O->F->Gauss_simple(v, 3, O->n, base_cols, 0/* verbose_level*/);

		FREE_INT(v);
		FREE_INT(base_cols);

		if (rk == 2) {
			return TRUE;
			}
		if (rk == 3) {
			return FALSE;
			}
		cout << "incidence_structure::get_ij fatal: rk=" << rk << endl;
		exit(1);
		}
	cout << "incidence_structure::get_ij: unknown realization_type";
	exit(1);
}

INT incidence_structure::get_lines_on_point(INT *data, INT i)
{
	if (i < 0 || i >= nb_rows) {
		cout << "incidence_structure::get_lines_on_point  i=" << i << " is illegal" << endl;
		exit(1);
		}
	if (realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX || 
		realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_HJELMSLEV) {
		INT h;
		for (h = 0; h < nb_lines_on_point[i]; h++) {
			data[h] = lines_on_point[i * max_r + h];
			}
		return nb_lines_on_point[i];
		}
	else if (realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_ORTHOGONAL) {
		O->lines_on_point_by_line_rank(i, data, 0/*verbose_level - 2*/);
		return O->alpha;
		}
	cout << "incidence_structure::get_lines_on_point fatal: unknown realization_type";
	exit(1);
}

INT incidence_structure::get_points_on_line(INT *data, INT j)
{
	if (realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX || 
		realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_HJELMSLEV) {
		INT h;
		for (h = 0; h < nb_points_on_line[j]; h++) {
			data[h] = points_on_line[j * max_k + h];
			}
		return nb_points_on_line[j];
		}
	else if (realization_type == INCIDENCE_STRUCTURE_REALIZATION_BY_ORTHOGONAL) {
		O->points_on_line_by_line_rank(j, data, 0/* verbose_level - 2*/);
		return O->q + 1;
		}
	cout << "incidence_structure::get_points_on_line fatal: unknown realization_type";
	exit(1);
}

INT incidence_structure::get_nb_inc()
{
	if (f_rowsums_constant) {
		return nb_rows * r;
		}
	else {
		INT i, j, nb = 0;
		
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

void incidence_structure::save_inc_file(BYTE *fname)
{
	INT i, j, nb_inc;

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

void incidence_structure::save_row_by_row_file(BYTE *fname)
{
    INT i, j, nb_inc;
    int w;

	nb_inc = get_nb_inc();
	
	if (!f_rowsums_constant) {
		cout << "incidence_structure::save_row_by_row_file rowsums are not constant" << endl;
		exit(1);
		}
	ofstream f(fname);
	w = (int) INT_log10(nb_cols) + 1;
	f << nb_rows << " " << nb_cols << " " << r << endl;
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < r; j++) {
			f << setw(w) << lines_on_point[i * r + j] << " ";
			}
		f << endl;
		}
	f << "-1" << endl;
}


void incidence_structure::print(ostream &ost)
{
	INT i, j;

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

void incidence_structure::compute_TDO_safe_first(partitionstack &PStack, 
	INT depth, INT &step, INT &f_refine, INT &f_refine_prev, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "incidence_structure::compute_TDO_safe_first" << endl;
		}
	step = 0;
	f_refine_prev = TRUE;
}

INT incidence_structure::compute_TDO_safe_next(partitionstack &PStack, 
	INT depth, INT &step, INT &f_refine, INT &f_refine_prev, INT verbose_level)
// returns TRUE when we are done, FALSE otherwise
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "incidence_structure::compute_TDO_safe_next" << endl;
		}
	if (EVEN(step)) {
		f_refine = refine_column_partition_safe(PStack, verbose_level - 3);
		}
	else {
		f_refine = refine_row_partition_safe(PStack, verbose_level - 3);
		}

	if (f_vv) {
		cout << "incidence_structure::compute_TDO_safe_next step=" << step << " after refine" << endl;
		}
	if (f_vvv) {
		if (EVEN(step)) {
			INT f_list_incidences = FALSE;
			get_and_print_col_decomposition_scheme(PStack, f_list_incidences, FALSE);
			PStack.print_classes_points_and_lines(cout);
			}
		else {
			INT f_list_incidences = FALSE;
			get_and_print_row_decomposition_scheme(PStack, f_list_incidences, FALSE);
			PStack.print_classes_points_and_lines(cout);
			}
		}
	step++;
	if (!f_refine_prev && !f_refine) {
		return TRUE;
		}
	f_refine_prev = f_refine;
	return FALSE;
}

void incidence_structure::compute_TDO_safe(partitionstack &PStack, 
	INT depth, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT f_refine, f_refine_prev;
	INT i;
	
	if (f_v) {
		cout << "incidence_structure::compute_TDO_safe" << endl;
		}
	
	f_refine_prev = TRUE;
	for (i = 0; i < depth; i++) {
		if (EVEN(i)) {
			f_refine = refine_column_partition_safe(PStack, verbose_level - 3);
			}
		else {
			f_refine = refine_row_partition_safe(PStack, verbose_level - 3);
			}

		if (f_v) {
			cout << "incidence_structure::compute_TDO_safe i=" << i << " after refine" << endl;
			if (EVEN(i)) {
				INT f_list_incidences = FALSE;
				get_and_print_col_decomposition_scheme(PStack, f_list_incidences, FALSE);
				PStack.print_classes_points_and_lines(cout);
				}
			else {
				INT f_list_incidences = FALSE;
				get_and_print_row_decomposition_scheme(PStack, f_list_incidences, FALSE);
				PStack.print_classes_points_and_lines(cout);
				}
			}
		
		if (!f_refine_prev && !f_refine) {
			goto done;
			}
		f_refine_prev = f_refine;
		}

done:
	if (f_v) {
		cout << "incidence_structure::compute_TDO_safe done" << endl;
		}
	
}

INT incidence_structure::compute_TDO(partitionstack &PStack, INT ht0, 
	INT depth, INT verbose_level)
{
	INT h1, h2, ht1, remaining_depth = depth;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	
	ht1 = PStack.ht;
	h2 = 0;
	
	if (f_v) {
		cout << "incidence_structure::compute_TDO depth=" << depth << " ht=" << PStack.ht << endl;
		if (ht0 >= PStack.ht) {
			cout << "ht0 >= PStack.ht, fatal" << endl;
			exit(1);
			}	
		}
	while (remaining_depth) {

		if (f_v) {
			cout << "incidence_structure::compute_TDO remaining_depth=" << remaining_depth << endl;
			}
		if (remaining_depth) {
			h1 = compute_TDO_step(PStack, ht0, verbose_level);
			h2 = hashing(h1, h2);
			ht0 = ht1;
			ht1 = PStack.ht;
			remaining_depth--;
			}
		else {
			break;
			}
		if (f_v) {
			cout << "incidence_structure::compute_TDO after compute_TDO_step ht0=" << ht0 << " ht1=" << ht1 << endl;
			}
		if (ht0 == ht1) {
			break;
			}
		}
	return h2;
}

INT incidence_structure::compute_TDO_step(partitionstack &PStack, INT ht0, INT verbose_level)
{
	INT ht1, h1, h2;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT f_is_row_class;

	ht1 = PStack.ht;
	h2 = 0;
	if (f_v) {
		cout << "incidence_structure::compute_TDO_step ht=" << PStack.ht << " ht0=" << ht0 << endl;
		cout << "before PStack.is_row_class" << endl;
		}
	f_is_row_class = PStack.is_row_class(ht0);
	if (f_v) {
		cout << "after PStack.is_row_class" << endl;
		}
	if (f_is_row_class) {
		if (f_vv) {
			cout << "incidence_structure::compute_TDO before refine_column_partition ht0=" << ht0 << " ht1=" << ht1 << endl;
			}
		h1 = refine_column_partition(PStack, ht0, verbose_level - 3);
		//cout << "h1=" << h1 << endl;
		h2 = hashing(h1, h2);
		if (f_v) {
			cout << "incidence_structure::compute_TDO after refine_column_partition ht=" << PStack.ht << endl;
			}
		if (f_vv) {
			INT f_list_incidences = FALSE;
			get_and_print_col_decomposition_scheme(PStack, f_list_incidences, FALSE);
			}	
		if (f_vvv) {
			PStack.print_classes_points_and_lines(cout);
			}
		}
	else {
		if (f_vv) {
			cout << "incidence_structure::compute_TDO before refine_column_partition ht0=" << ht0 << " ht1=" << ht1 << endl;
			}
		h1 = refine_row_partition(PStack, ht0, verbose_level - 3);
		//cout << "h1=" << h1 << endl;
		h2 = hashing(h1, h2);
		//cout << "h2=" << h2 << endl;
		if (f_v) {
			cout << "incidence_structure::compute_TDO after refine_row_partition ht=" << PStack.ht << endl;
			}
		if (f_vv) {
			INT f_list_incidences = FALSE;
			get_and_print_row_decomposition_scheme(PStack, f_list_incidences, FALSE);
			}
		if (f_vvv) {
			PStack.print_classes_points_and_lines(cout);
			}
		}
	return h2;
}

#if 0
INT incidence_structure::compute_TDO(partitionstack &PStack, INT ht0, 
	INT marker1, INT marker2, INT depth, INT verbose_level)
{
	INT h1, h2, ht1, remaining_depth = depth;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	
	ht1 = PStack.ht;
	h2 = 0;
	
	if (f_v) {
		cout << "incidence_structure::compute_TDO depth=" << depth << " ht=" << PStack.ht << endl;
		}
	if (remaining_depth) {
		if (f_vv) {
			cout << "incidence_structure::compute_TDO before refine_column_partition ht0=" << ht0 << " ht1=" << ht1 << endl;
			}
		h1 = refine_column_partition(PStack, ht0, verbose_level - 3);
		//cout << "h1=" << h1 << endl;
		h2 = hashing(h1, h2);
		if (f_v) {
			cout << "incidence_structure::compute_TDO after refine_column_partition ht=" << PStack.ht << endl;
			}
		if (f_vv) {
			get_and_print_col_decomposition_scheme(PStack, marker1, marker2);
			}	
		if (f_vvv) {
			PStack.print_classes_points_and_lines(cout);
			}
		//cout << "h2=" << h2 << endl;

		//cout << "after refinement:" << endl;
		//cout << *this << endl;
		remaining_depth--;
		}

	while (remaining_depth) {

		if (remaining_depth) {
			ht0 = ht1;
			ht1 = PStack.ht;
		
			if (f_vv) {
				cout << "incidence_structure::compute_TDO before refine_column_partition ht0=" << ht0 << " ht1=" << ht1 << endl;
				}
			h1 = refine_row_partition(PStack, ht0, verbose_level - 3);
			//cout << "h1=" << h1 << endl;
			h2 = hashing(h1, h2);
			//cout << "h2=" << h2 << endl;
			if (f_v) {
				cout << "incidence_structure::compute_TDO after refine_row_partition ht=" << PStack.ht << endl;
				}
			if (f_vv) {
				get_and_print_row_decomposition_scheme(PStack, marker1, marker2);
				}
			if (f_vvv) {
				PStack.print_classes_points_and_lines(cout);
				}
			remaining_depth--;
			}
		
		if (PStack.ht == ht1 || remaining_depth == 0)
			break;

		if (remaining_depth) {
			ht0 = ht1;
			ht1 = PStack.ht;
		
			if (f_vv) {
				cout << "incidence_structure::compute_TDO before refine_column_partition ht0=" << ht0 << " ht1=" << ht1 << endl;
				}
			h1 = refine_column_partition(PStack, ht0, verbose_level - 3);
			//cout << "h1=" << h1 << endl;
			h2 = hashing(h1, h2);
			//cout << "h2=" << h2 << endl;
			if (f_v) {
				cout << "incidence_structure::compute_TDO after refine_column_partition ht=" << PStack.ht << endl;
				}
			if (f_vv) {
				get_and_print_col_decomposition_scheme(PStack, marker1, marker2);
				}
			if (f_vvv) {
				PStack.print_classes_points_and_lines(cout);
				}
			remaining_depth--;
		
			if (PStack.ht == ht1)
				break;
			}
		}

	//cout << "hash = " << h2 << endl;
	
	if (f_v) {
		cout << "incidence_structure::compute_TDO finished, hash=" << h2 << endl;
		//get_and_print_decomposition_schemes(O, marker1, marker2);
		}	
	return h2;
}
#endif

void incidence_structure::get_partition(partitionstack &PStack, 
	INT *row_classes, INT *row_class_idx, INT &nb_row_classes, 
	INT *col_classes, INT *col_class_idx, INT &nb_col_classes)
{
	INT i, c;
	
	PStack.get_row_and_col_classes(
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 0 /*verbose_level*/);
	for (i = 0; i < PStack.ht; i++) {
		row_class_idx[i] = col_class_idx[i] = -1;
		}
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		row_class_idx[c] = i;
		}
	for (i = 0; i < nb_col_classes; i++) {
		c = col_classes[i];
		col_class_idx[c] = i;
		}
}

INT incidence_structure::refine_column_partition_safe(partitionstack &PStack, INT verbose_level)
{

	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 5);
	INT i, j, c, h, I, ht, first, next, N;
	INT *row_classes;
	INT *row_class_idx;
	INT nb_row_classes;
	INT *col_classes;
	INT *col_class_idx;
	INT nb_col_classes;

	INT *data;
	INT *neighbors;

	if (f_v) {
		cout << "incidence_structure::refine_column_partition_safe" << endl;
		}
	row_classes = NEW_INT(PStack.ht);
	col_classes = NEW_INT(PStack.ht);
	row_class_idx = NEW_INT(PStack.ht);
	col_class_idx = NEW_INT(PStack.ht);

	get_partition(PStack, row_classes, row_class_idx, nb_row_classes, col_classes, col_class_idx, nb_col_classes);
	
	N = nb_points() + nb_lines();
	data = NEW_INT(N * nb_row_classes);
	for (i = 0; i < N * nb_row_classes; i++) {
		data[i] = 0;
		}
	
	neighbors = NEW_INT(max_k);

	for (j = 0; j < nb_lines(); j++) {
		get_points_on_line(neighbors, j);
		for (h = 0; h < nb_points_on_line[j]; h++) {
			i = neighbors[h];
			c = PStack.cellNumber[PStack.invPointList[i]];
			I = row_class_idx[c];
			if (I == -1) {
				cout << "incidence_structure::refine_column_partition_safe I == -1" << endl;
				exit(1);
				}
			data[(nb_points() + j) * nb_row_classes + I]++;
			}
		}
	if (f_vv) {
		cout << "incidence_structure::refine_column_partition_safe data:" << endl;
		print_integer_matrix_width(cout, data + nb_points() * nb_row_classes, 
			nb_lines(), nb_row_classes, nb_row_classes, 3);
		}

	ht = PStack.ht;
	
	for (c = 0; c < ht; c++) {
		if (PStack.is_row_class(c))
			continue;
			
		if (PStack.cellSize[c] == 1)
			continue;
		first = PStack.startCell[c];
		next = first + PStack.cellSize[c];

		PStack.radix_sort(first /* left */,
				   next - 1 /* right */,
				   data, nb_row_classes, 0 /*radix*/, FALSE);
		}

	FREE_INT(data);
	FREE_INT(neighbors);
	
	FREE_INT(row_classes);
	FREE_INT(col_classes);
	FREE_INT(row_class_idx);
	FREE_INT(col_class_idx);
	if (f_v) {
		cout << "incidence_structure::refine_column_partition_safe done" << endl;
		}
	if (PStack.ht == ht) {
		return FALSE;
		}
	else {
		return TRUE;
		}
}

INT incidence_structure::refine_row_partition_safe(partitionstack &PStack, INT verbose_level)
{

	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 5);
	INT i, j, c, h, J, ht, first, next;
	INT *row_classes;
	INT *row_class_idx;
	INT nb_row_classes;
	INT *col_classes;
	INT *col_class_idx;
	INT nb_col_classes;

	INT *data;
	INT *neighbors;

	if (f_v) {
		cout << "incidence_structure::refine_row_partition_safe" << endl;
		}
	row_classes = NEW_INT(PStack.ht);
	col_classes = NEW_INT(PStack.ht);
	row_class_idx = NEW_INT(PStack.ht);
	col_class_idx = NEW_INT(PStack.ht);

	get_partition(PStack, row_classes, row_class_idx, nb_row_classes, col_classes, col_class_idx, nb_col_classes);
	
	data = NEW_INT(nb_points() * nb_col_classes);
	for (i = 0; i < nb_points() * nb_col_classes; i++) {
		data[i] = 0;
		}
	
	neighbors = NEW_INT(max_r);

	for (i = 0; i < nb_points(); i++) {
		get_lines_on_point(neighbors, i);
		for (h = 0; h < nb_lines_on_point[i]; h++) {
			j = neighbors[h] + nb_points();
			c = PStack.cellNumber[PStack.invPointList[j]];
			J = col_class_idx[c];
			if (J == -1) {
				cout << "incidence_structure::refine_row_partition_safe J == -1" << endl;
				exit(1);
				}
			data[i * nb_col_classes + J]++;
			}
		}
	if (f_vv) {
		cout << "incidence_structure::refine_row_partition_safe data:" << endl;
		print_integer_matrix_width(cout, data, nb_points(), nb_col_classes, nb_col_classes, 3);
		}

	ht = PStack.ht;
	
	for (c = 0; c < ht; c++) {
		if (PStack.is_col_class(c))
			continue;
			
		if (PStack.cellSize[c] == 1)
			continue;
		first = PStack.startCell[c];
		next = first + PStack.cellSize[c];

		PStack.radix_sort(first /* left */,
				   next - 1 /* right */,
				   data, nb_col_classes, 0 /*radix*/, FALSE);
		}

	FREE_INT(data);
	FREE_INT(neighbors);
	
	FREE_INT(row_classes);
	FREE_INT(col_classes);
	FREE_INT(row_class_idx);
	FREE_INT(col_class_idx);
	if (f_v) {
		cout << "incidence_structure::refine_row_partition_safe done" << endl;
		}
	if (PStack.ht == ht) {
		return FALSE;
		}
	else {
		return TRUE;
		}
}

INT incidence_structure::refine_column_partition(partitionstack &PStack, INT ht0, INT verbose_level)
{
	//verbose_level += 3;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT row_cell, f, l, i, j, x, y, u, c, cell, N, first, next, ht1, depth, idx, nb;
	INT *data;
	INT *neighbors, h;
	
	N = nb_points() + nb_lines();
	ht1 = PStack.ht;
	depth = ht1 - ht0 + 1;
	if (f_v) {
		cout << "incidence_structure::refine_column_partition ht0=" << ht0 << " ht=" << PStack.ht << " depth=" << depth << endl;
		cout << "N=" << N << endl;		
		cout << "depth=" << depth << endl;		
		cout << "max_r=" << max_r << endl;		
		}
	data = NEW_INT(N * depth);
	for (i = 0; i < N * depth; i++) {
		data[i] = 0;
		}
	
	neighbors = NEW_INT(max_r);
	for (y = 0; y < nb_lines(); y++) {
		j = nb_points() + y;
		c = PStack.cellNumber[PStack.invPointList[j]];
		data[j * depth + 0] = c;
		}

	for (row_cell = ht0; row_cell < ht1; row_cell++) {
		idx = row_cell - ht0;
		f = PStack.startCell[row_cell];
		l = PStack.cellSize[row_cell];
		if (f_vvv) {
			cout << "incidence_structure::refine_column_partition idx=" << idx 
				<< " row_cell=" << row_cell 
				<< " f=" << f << " l=" << l << endl;
			}
		if (!PStack.is_row_class(row_cell)) {
			cout << "row_cell is not a row cell" << endl;
			cout << "row_cell=" << row_cell << endl;
			cout << "ht0=" << ht0 << endl;
			cout << "ht1=" << ht1 << endl;
			cout << PStack << endl;
			exit(1);
			}
	
	
		for (i = 0; i < l; i++) {
			x = PStack.pointList[f + i];
			//if (f_v) {cout << i << " : " << x << " : ";}
			if (f_vv) {
				cout << "before get_lines_on_point" << endl;
				}
			nb = get_lines_on_point(neighbors, x);
			if (f_vv) {
				cout << "after get_lines_on_point nb=" << nb << endl;
				}
			
			//O.lines_on_point_by_line_rank(x, neighbors, 0/*verbose_level - 2*/);
			//if (f_v) {INT_vec_print(cout, neighbors, O.alpha);cout << endl;}
			for (u = 0; u < nb; u++) {
				y = neighbors[u];
				j = nb_points() + y;
				data[j * depth + 1 + idx]++;
				}
			}
		}
#if 0
	if (f_vvv) {
		cout << "data:" << endl;
		for (y = 0; y < O.nb_lines; y++) {
			j = O.nb_points + y;
			cout << y << " : " << j << " : ";
			INT_vec_print(cout, data + j * depth, depth);
			cout << endl;
			}
		cout << endl;
		}
#endif
		
	ht0 = PStack.ht;
	for (cell = 0; cell < ht0; cell++) {
		if (PStack.is_row_class(cell))
			continue;
			
		if (PStack.cellSize[cell] == 1)
			continue;
		first = PStack.startCell[cell];
		next = first + PStack.cellSize[cell];

		PStack.radix_sort(first /* left */,
				   next - 1 /* right */,
				   data, depth, 0 /*radix*/, FALSE);
		}
	if (f_vv) {
		cout << "incidence_structure::refine_column_partition after sorting, with " << PStack.ht - ht0 << " new classes" << endl;
		cout << PStack << endl;
		}

	if (f_vvv) {
		PStack.print_column_refinement_info(ht0, data, depth);
		}
		
	h = PStack.hash_column_refinement_info(ht0, data, depth, 0);
	
	FREE_INT(data);
	FREE_INT(neighbors);
	return h;
}

INT incidence_structure::refine_row_partition(partitionstack &PStack, INT ht0, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT col_cell, f, l, i, j, x, y, u, c, cell, N, first, next, ht1, depth, idx, nb;
	INT *data;
	INT *neighbors, nb_neighbors, h;
	
	N = nb_points() + nb_lines();
	ht1 = PStack.ht;
	depth = ht1 - ht0 + 1;
	if (f_v) {
		cout << "incidence_structure::refine_row_partition ht0=" << ht0 << " ht=" << PStack.ht << " depth=" << depth << endl;
		}
	data = NEW_INT(N * depth);
	for (i = 0; i < N * depth; i++) 
		data[i] = 0;
	
	nb_neighbors = max_k;
	neighbors = NEW_INT(nb_neighbors);
	for (x = 0; x < nb_points(); x++) {
		i = x;
		c = PStack.cellNumber[PStack.invPointList[i]];
		data[i * depth + 0] = c;
		}

	for (col_cell = ht0; col_cell < ht1; col_cell++) {
		idx = col_cell - ht0;
		f = PStack.startCell[col_cell];
		l = PStack.cellSize[col_cell];
		if (f_vvv) {
			cout << "incidence_structure::refine_row_partition idx=" << idx 
				<< " col_cell=" << col_cell 
				<< " f=" << f 
				<< " l=" << l << endl;
			}

		if (!PStack.is_col_class(col_cell)) {
			cout << "col_cell is not a col cell" << endl;
			cout << "ht0=" << ht0 << endl;
			cout << "ht1=" << ht1 << endl;
			cout << PStack << endl;
			exit(1);
			}
	
	
	
		for (j = 0; j < l; j++) {
			y = PStack.pointList[f + j];
			//if (f_v) {cout << j << " : " << y << " : ";}
			
			//O.points_on_line_by_line_rank(y - O.nb_points, neighbors, 0/* verbose_level - 2*/);
			nb = get_points_on_line(neighbors, y - nb_points());
	
			//if (f_v) {INT_vec_print(cout, neighbors, O.alpha);cout << endl;}
			for (u = 0; u < nb; u++) {
				x = neighbors[u];
				i = x;
				data[i * depth + 1 + idx]++;
				}
			}
		}
#if 0
	if (f_vvv) {
		cout << "data:" << endl;
		for (i = 0; i < O.nb_lines; i++) {
			cout << i << " : ";
			INT_vec_print(cout, data + i * depth, depth);
			cout << endl;
			}
		cout << endl;
		}
#endif
		
	ht0 = PStack.ht;
	for (cell = 0; cell < ht0; cell++) {
		if (PStack.is_col_class(cell))
			continue;
			
		if (PStack.cellSize[cell] == 1)
			continue;
		first = PStack.startCell[cell];
		next = first + PStack.cellSize[cell];

		PStack.radix_sort(first /* left */,
				   next - 1 /* right */,
				   data, depth, 0 /*radix*/, FALSE);
		}
	if (f_vv) {
		cout << "incidence_structure::refine_row_partition after sorting, with " << PStack.ht - ht0 << " new classes" << endl;
		cout << PStack << endl;
		}

	if (f_vv) {
		PStack.print_row_refinement_info(ht0, data, depth);
		}

	h = PStack.hash_row_refinement_info(ht0, data, depth, 0);

	
	FREE_INT(data);
	FREE_INT(neighbors);
	return h;
}

void incidence_structure::print_row_tactical_decomposition_scheme_incidences_tex(
	partitionstack &PStack, 
	ostream &ost, INT f_enter_math_mode, 
	INT *row_classes, INT *row_class_inv, INT nb_row_classes,
	INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
	INT f_local_coordinates, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *incidences;
	INT c1, c2, f1, f2, l1, l2;
	INT i, j, rij;
	INT u, v, x, a, b, c, J;
	INT *row_scheme;
	INT *S;

	if (f_v) {
		cout << "incidence_structure::print_row_tactical_decomposition_scheme_incidences_tex" << endl;
		}

	row_scheme = NEW_INT(nb_row_classes * nb_col_classes);
	
	get_row_decomposition_scheme(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		row_scheme, verbose_level - 2);

	
	for (i = 0; i < nb_row_classes; i++) {
		c1 = row_classes[i];
		f1 = PStack.startCell[c1];
		l1 = PStack.cellSize[c1];
		
		for (j = 0; j < nb_col_classes; j++) {

			rij = row_scheme[i * nb_col_classes + j];
			
			if (rij == 0) {
				continue;
				}
			
			S = NEW_INT(rij);
			get_incidences_by_row_scheme(PStack, 
				row_classes, row_class_inv, nb_row_classes,
				col_classes, col_class_inv, nb_col_classes, 
				i, j, 
				rij, incidences, verbose_level - 2);

			c2 = col_classes[j];
			f2 = PStack.startCell[c2];
			l2 = PStack.cellSize[c2];

			ost << "\\subsubsection*{Row class " << i << " (cell " << c1 << ") vs. col class " << j << " (cell " << c2 << ")";
			if (f_local_coordinates) {
				ost << " (in local coordinates)";
				}
			ost << ", $r_{" << i << ", " << j << "}=" << rij << "$}" << endl;
			//ost << "f1=" << f1 << " l1=" << l1 << endl;
			//ost << "f2=" << f2 << " l2=" << l2 << endl;
			for (u = 0; u < l1; u++) {
				x = PStack.pointList[f1 + u];
				ost << setw(4) << u << " : $P_{" << setw(4) << x << "}$ is incident with ";
				for (v = 0; v < rij; v++) {
					a = incidences[u * rij + v];
					if (f_local_coordinates) {
						b = nb_points() + a;
						c = PStack.cellNumber[PStack.invPointList[b]];
						J = col_class_inv[c];
						if (J != j) {
							cout << "incidence_structure::print_row_tactical_decomposition_scheme_incidences_tex J != j" << endl;
							cout << "j=" << j << endl;
							cout << "J=" << J << endl;
							}
						a = PStack.invPointList[b] - f2;
						}
					S[v] = a;
					//ost << a << " ";
					}
				INT_vec_heapsort(S, rij);
				ost << "$\\{";
				for (v = 0; v < rij; v++) {
					ost << "\\ell_{" << setw(4) << S[v] << "}";
					if (v < rij - 1) {
						ost << ", ";
						}
					}
				ost << "\\}$";
				
				ost << "\\\\" << endl;
				}
			
			FREE_INT(incidences);
			FREE_INT(S);
			} // next j
		} // next i

	FREE_INT(row_scheme);
}

void incidence_structure::print_col_tactical_decomposition_scheme_incidences_tex(
	partitionstack &PStack, 
	ostream &ost, INT f_enter_math_mode, 
	INT *row_classes, INT *row_class_inv, INT nb_row_classes,
	INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
	INT f_local_coordinates, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *incidences;
	INT c1, c2, f1, f2, l1, l2;
	INT i, j, kij;
	INT u, v, y, a, b, c, I;
	INT *col_scheme;
	INT *S;

	if (f_v) {
		cout << "incidence_structure::print_col_tactical_decomposition_scheme_incidences_tex" << endl;
		}

	col_scheme = NEW_INT(nb_row_classes * nb_col_classes);
	
	get_col_decomposition_scheme(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		col_scheme, verbose_level - 2);

	
	for (i = 0; i < nb_row_classes; i++) {
		c1 = row_classes[i];
		f1 = PStack.startCell[c1];
		l1 = PStack.cellSize[c1];
		
		for (j = 0; j < nb_col_classes; j++) {

			kij = col_scheme[i * nb_col_classes + j];
			
			if (kij == 0) {
				continue;
				}
			
			S = NEW_INT(kij);
			get_incidences_by_col_scheme(PStack, 
				row_classes, row_class_inv, nb_row_classes,
				col_classes, col_class_inv, nb_col_classes, 
				i, j, 
				kij, incidences, verbose_level - 2);

			c2 = col_classes[j];
			f2 = PStack.startCell[c2];
			l2 = PStack.cellSize[c2];

			ost << "\\subsubsection*{Row class " << i << " (cell " << c1 << ") vs. col class " << j << " (cell " << c2 << ")";
			if (f_local_coordinates) {
				ost << " (in local coordinates)";
				}
			ost << ", $k_{" << i << ", " << j << "}=" << kij << "$}" << endl;
			//ost << "f1=" << f1 << " l1=" << l1 << endl;
			//ost << "f2=" << f2 << " l2=" << l2 << endl;
			for (u = 0; u < l2; u++) {
				y = PStack.pointList[f2 + u] - nb_points();
				ost << setw(4) << u << " : $\\ell_{" << setw(4) << y << "}$ is incident with ";
				for (v = 0; v < kij; v++) {
					a = incidences[u * kij + v];
					if (f_local_coordinates) {
						b = a;
						c = PStack.cellNumber[PStack.invPointList[b]];
						I = row_class_inv[c];
						if (I != i) {
							cout << "incidence_structure::print_col_tactical_decomposition_scheme_incidences_tex I != i" << endl;
							cout << "i=" << i << endl;
							cout << "I=" << I << endl;
							}
						a = PStack.invPointList[b] - f1;
						}
					S[v] = a;
					//ost << a << " ";
					}
				INT_vec_heapsort(S, kij);
				ost << "$\\{";
				for (v = 0; v < kij; v++) {
					ost << "P_{" << setw(4) << S[v] << "}";
					if (v < kij - 1) {
						ost << ", ";
						}
					}
				ost << "\\}$";
				
				ost << "\\\\" << endl;
				}
			
			FREE_INT(incidences);
			FREE_INT(S);
			} // next j
		} // next i

	FREE_INT(col_scheme);
}

void incidence_structure::get_incidences_by_row_scheme(partitionstack &PStack, 
	INT *row_classes, INT *row_class_inv, INT nb_row_classes,
	INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
	INT row_class_idx, INT col_class_idx, 
	INT rij, INT *&incidences, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT c1, c2, f1, f2, l1, l2, x, nb, u, y, c, i, j;
	INT *sz;
	INT *neighbors;
	
	if (f_v) {
		cout << "incidence_structure::get_incidences_by_row_scheme" << endl;
		}
	c1 = row_classes[row_class_idx];
	f1 = PStack.startCell[c1];
	l1 = PStack.cellSize[c1];
	c2 = col_classes[col_class_idx];
	f2 = PStack.startCell[c2];
	l2 = PStack.cellSize[c2];
	incidences = NEW_INT(l1 * rij);
	neighbors = NEW_INT(max_r);
	sz = NEW_INT(l1);
	for (i = 0; i < l1; i++) {
		sz[i] = 0;
		}
	for (i = 0; i < l1; i++) {
		x = PStack.pointList[f1 + i];
		nb = get_lines_on_point(neighbors, x);
		//O.lines_on_point_by_line_rank(x, neighbors, verbose_level - 2);
		for (u = 0; u < nb; u++) {
			y = neighbors[u];
			j = nb_points() + y;
			c = PStack.cellNumber[PStack.invPointList[j]];
			if (c == c2) {
				incidences[i * rij + sz[i]++] = y;
				}
			}
		} // next i
	
	for (i = 0; i < l1; i++) {
		if (sz[i] != rij) {
			cout << "sz[i] != rij" << endl;
			exit(1);
			}
		}

	FREE_INT(sz);
	FREE_INT(neighbors);
}

void incidence_structure::get_incidences_by_col_scheme(partitionstack &PStack, 
	INT *row_classes, INT *row_class_inv, INT nb_row_classes,
	INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
	INT row_class_idx, INT col_class_idx, 
	INT kij, INT *&incidences, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT c1, c2, f1, f2, l1, l2, x, nb, u, y, c, i, j;
	INT *sz;
	INT *neighbors;
	
	if (f_v) {
		cout << "incidence_structure::get_incidences_by_col_scheme" << endl;
		}
	c1 = row_classes[row_class_idx];
	f1 = PStack.startCell[c1];
	l1 = PStack.cellSize[c1];
	c2 = col_classes[col_class_idx];
	f2 = PStack.startCell[c2];
	l2 = PStack.cellSize[c2];
	incidences = NEW_INT(l2 * kij);
	neighbors = NEW_INT(max_k);
	sz = NEW_INT(l2);
	for (j = 0; j < l2; j++) {
		sz[j] = 0;
		}
	for (j = 0; j < l2; j++) {
		y = PStack.pointList[f2 + j] - nb_points();
		nb = get_points_on_line(neighbors, y);
		for (u = 0; u < nb; u++) {
			x = neighbors[u];
			i = x;
			c = PStack.cellNumber[PStack.invPointList[i]];
			if (c == c1) {
				incidences[j * kij + sz[j]++] = x;
				}
			}
		} // next j
	
	for (j = 0; j < l2; j++) {
		if (sz[j] != kij) {
			cout << "sz[j] != kij" << endl;
			exit(1);
			}
		}

	FREE_INT(sz);
	FREE_INT(neighbors);
}


void incidence_structure::get_row_decomposition_scheme(partitionstack &PStack, 
	INT *row_classes, INT *row_class_inv, INT nb_row_classes,
	INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
	INT *row_scheme, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT I, J, i, j, c1, f1, l1, x, y, u, c, nb;
	INT *neighbors;
	INT *data0;
	INT *data1;
	
	if (f_v) {
		cout << "incidence_structure::get_row_decomposition_scheme" << endl;
		}
	neighbors = NEW_INT(max_r);
	data0 = NEW_INT(nb_col_classes);
	data1 = NEW_INT(nb_col_classes);
	INT_vec_zero(row_scheme, nb_row_classes * nb_col_classes);
	for (I = 0; I < nb_row_classes; I++) {
		c1 = row_classes[I];
		f1 = PStack.startCell[c1];
		l1 = PStack.cellSize[c1];
		INT_vec_zero(data0, nb_col_classes);
		for (i = 0; i < l1; i++) {
			x = PStack.pointList[f1 + i];
			INT_vec_zero(data1, nb_col_classes);
			nb = get_lines_on_point(neighbors, x);
			for (u = 0; u < nb; u++) {
				y = neighbors[u];
				j = nb_points() + y;
				c = PStack.cellNumber[PStack.invPointList[j]];
				J = col_class_inv[c];
				data1[J]++;
				}
			if (i == 0) {
				INT_vec_copy(data1, data0, nb_col_classes);
				}
			else {
				for (J = 0; J < nb_col_classes; J++) {
					if (data0[J] != data1[J]) {
						cout << "incidence_structure::get_row_decomposition_scheme not row-tactical I=" << I << " i=" << i << " J=" << J << endl;
						}
					}
				}
			} // next i

		INT_vec_copy(data0, row_scheme + I * nb_col_classes, nb_col_classes);
		}
	FREE_INT(neighbors);
	FREE_INT(data0);
	FREE_INT(data1);
}

void incidence_structure::get_row_decomposition_scheme_if_possible(partitionstack &PStack, 
	INT *row_classes, INT *row_class_inv, INT nb_row_classes,
	INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
	INT *row_scheme, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT I, J, i, j, c1, f1, l1, x, y, u, c, nb;
	INT *neighbors;
	INT *data0;
	INT *data1;
	
	if (f_v) {
		cout << "incidence_structure::get_row_decomposition_scheme_if_possible" << endl;
		}
	neighbors = NEW_INT(max_r);
	data0 = NEW_INT(nb_col_classes);
	data1 = NEW_INT(nb_col_classes);
	for (i = 0; i < nb_row_classes * nb_col_classes; i++) {
		row_scheme[i] = 0;
		}
	for (I = 0; I < nb_row_classes; I++) {
		c1 = row_classes[I];
		f1 = PStack.startCell[c1];
		l1 = PStack.cellSize[c1];
		for (j = 0; j < nb_col_classes; j++) 
			data0[j] = 0;
		for (i = 0; i < l1; i++) {
			x = PStack.pointList[f1 + i];
			for (J = 0; J < nb_col_classes; J++) 
				data1[J] = 0;
			nb = get_lines_on_point(neighbors, x);
			//O.lines_on_point_by_line_rank(x, neighbors, verbose_level - 2);
			for (u = 0; u < nb; u++) {
				y = neighbors[u];
				j = nb_points() + y;
				c = PStack.cellNumber[PStack.invPointList[j]];
				J = col_class_inv[c];
				data1[J]++;
				}
			if (i == 0) {
				for (J = 0; J < nb_col_classes; J++) {
					data0[J] = data1[J];
					}
				}
			else {
				for (J = 0; J < nb_col_classes; J++) {
					if (data0[J] != data1[J]) {
						data0[J] = -1;
						//cout << "not row-tactical I=" << I << " i=" << i << " J=" << J << endl;
						}
					}
				}
			} // next i
		for (J = 0; J < nb_col_classes; J++) {
			row_scheme[I * nb_col_classes + J] = data0[J];
			}
		}
	FREE_INT(neighbors);
	FREE_INT(data0);
	FREE_INT(data1);
}

void incidence_structure::get_col_decomposition_scheme(partitionstack &PStack, 
	INT *row_classes, INT *row_class_inv, INT nb_row_classes,
	INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
	INT *col_scheme, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT I, J, i, j, c1, f1, l1, x, y, u, c, nb;
	INT *neighbors;
	INT *data0;
	INT *data1;
	
	if (f_v) {
		cout << "incidence_structure::get_col_decomposition_scheme" << endl;
		}
	neighbors = NEW_INT(max_k);
	data0 = NEW_INT(nb_row_classes);
	data1 = NEW_INT(nb_row_classes);
	for (i = 0; i < nb_row_classes * nb_col_classes; i++) {
		col_scheme[i] = 0;
		}
	for (J = 0; J < nb_col_classes; J++) {
		c1 = col_classes[J];
		f1 = PStack.startCell[c1];
		l1 = PStack.cellSize[c1];
		for (i = 0; i < nb_row_classes; i++) 
			data0[i] = 0;
		for (j = 0; j < l1; j++) {
			y = PStack.pointList[f1 + j] - nb_points();
			for (I = 0; I < nb_row_classes; I++) 
				data1[I] = 0;
			
			//O.points_on_line_by_line_rank(y, neighbors, verbose_level - 2);
			nb = get_points_on_line(neighbors, y);
	
			for (u = 0; u < nb; u++) {
				x = neighbors[u];
				c = PStack.cellNumber[PStack.invPointList[x]];
				I = row_class_inv[c];
				data1[I]++;
				}
			if (j == 0) {
				for (I = 0; I < nb_row_classes; I++) {
					data0[I] = data1[I];
					}
				}
			else {
				for (I = 0; I < nb_row_classes; I++) {
					if (data0[I] != data1[I]) {
						cout << "not col-tactical J=" << J << " j=" << j << " I=" << I << endl;
						}
					}
				}
			} // next j
		for (I = 0; I < nb_row_classes; I++) {
			col_scheme[I * nb_col_classes + J] = data0[I];
			}
		}
	FREE_INT(neighbors);
	FREE_INT(data0);
	FREE_INT(data1);
}

void incidence_structure::row_scheme_to_col_scheme(partitionstack &PStack, 
	INT *row_classes, INT *row_class_inv, INT nb_row_classes,
	INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
	INT *row_scheme, INT *col_scheme, INT verbose_level)
{
	INT I, J, c1, l1, c2, l2, a, b, c;
	
	for (I = 0; I < nb_row_classes; I++) {
		c1 = row_classes[I];
		l1 = PStack.cellSize[c1];
		for (J = 0; J < nb_col_classes; J++) {
			c2 = col_classes[J];
			l2 = PStack.cellSize[c2];
			a = row_scheme[I * nb_col_classes + J];
			b = a * l1;
			if (b % l2) {
				cout << "incidence_structure::row_scheme_to_col_scheme: cannot be tactical" << endl;
				exit(1);
				}
			c = b / l2;
			col_scheme[I * nb_col_classes + J] = c;
			}
		}
}

void incidence_structure::get_and_print_row_decomposition_scheme(
	partitionstack &PStack, 
	INT f_list_incidences, INT f_local_coordinates)
{
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT *row_scheme;

	cout << "incidence_structure::get_and_print_row_decomposition_scheme computing row scheme" << endl;
	PStack.allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		0);
	
	row_scheme = NEW_INT(nb_row_classes * nb_col_classes);

	get_row_decomposition_scheme(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, 0);

	//cout << *this << endl;
	
	cout << "row_scheme:" << endl;
	PStack.print_decomposition_scheme(cout, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		row_scheme, -1, -1);

	if (f_list_incidences) {
		cout << "incidences by row-scheme:" << endl;
		print_row_tactical_decomposition_scheme_incidences_tex(
			PStack, 
			cout, FALSE /* f_enter_math_mode */, 
			row_classes, row_class_inv, nb_row_classes,
			col_classes, col_class_inv, nb_col_classes, 
			f_local_coordinates, 0 /*verbose_level*/);
		}

	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	FREE_INT(row_scheme);
}

void incidence_structure::get_and_print_col_decomposition_scheme(
	partitionstack &PStack, 
	INT f_list_incidences, INT f_local_coordinates)
{
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT *col_scheme;
	INT verbose_level = 0;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "incidence_structure::get_and_print_col_decomposition_scheme computing col scheme" << endl;
		}
	PStack.allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		verbose_level);
	
	col_scheme = NEW_INT(nb_row_classes * nb_col_classes);

	get_col_decomposition_scheme(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		col_scheme, verbose_level);

	//cout << *this << endl;
	
	if (f_v) {
		cout << "col_scheme:" << endl;
		PStack.print_decomposition_scheme(cout, 
			row_classes, nb_row_classes,
			col_classes, nb_col_classes, 
			col_scheme, -1, -1);
		}

	if (f_list_incidences) {
		cout << "incidences by col-scheme:" << endl;
		print_col_tactical_decomposition_scheme_incidences_tex(
			PStack, 
			cout, FALSE /* f_enter_math_mode */, 
			row_classes, row_class_inv, nb_row_classes,
			col_classes, col_class_inv, nb_col_classes, 
			f_local_coordinates, 0 /*verbose_level*/);
		}

	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	FREE_INT(col_scheme);
}

void incidence_structure::get_and_print_decomposition_schemes(partitionstack &PStack)
{
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT *row_scheme, *col_scheme;
	INT f_v = FALSE;
	
	cout << "incidence_structure::get_and_print_decomposition_schemes computing both schemes" << endl;
	PStack.allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		f_v);
	
	row_scheme = NEW_INT(nb_row_classes * nb_col_classes);
	col_scheme = NEW_INT(nb_row_classes * nb_col_classes);

	get_row_decomposition_scheme(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, f_v);

	row_scheme_to_col_scheme(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, col_scheme, f_v);
	
	//cout << *this << endl;
	
	cout << "row_scheme:" << endl;
	PStack.print_decomposition_scheme(cout, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		row_scheme, -1, -1);

	cout << "col_scheme:" << endl;
	PStack.print_decomposition_scheme(cout, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		col_scheme, -1, -1);

	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	FREE_INT(row_scheme);
	FREE_INT(col_scheme);
}

void incidence_structure::get_and_print_decomposition_schemes_tex(partitionstack &PStack)
{
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT *row_scheme, *col_scheme;
	INT f_v = FALSE;
	
	cout << "incidence_structure::get_and_print_decomposition_schemes_tex computing both schemes" << endl;
	PStack.allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		f_v);
	
	row_scheme = NEW_INT(nb_row_classes * nb_col_classes);
	col_scheme = NEW_INT(nb_row_classes * nb_col_classes);

	get_row_decomposition_scheme(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, f_v);

	row_scheme_to_col_scheme(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, col_scheme, f_v);
	
	//cout << *this << endl;
	
	cout << "row_scheme:" << endl;
	PStack.print_decomposition_scheme_tex(cout, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		row_scheme);

	cout << "col_scheme:" << endl;
	PStack.print_decomposition_scheme_tex(cout, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		col_scheme);
	cout << "tactical scheme:" << endl;
	PStack.print_tactical_decomposition_scheme_tex(cout, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		row_scheme, col_scheme, FALSE /* f_print_subscripts */);

	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	FREE_INT(row_scheme);
	FREE_INT(col_scheme);
}

void incidence_structure::get_and_print_tactical_decomposition_scheme_tex(
	ostream &ost, INT f_enter_math, partitionstack &PStack)
{
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT *row_scheme, *col_scheme;
	INT f_v = FALSE;
	
	cout << "incidence_structure::get_and_print_tactical_decomposition_scheme_tex computing both schemes" << endl;
	PStack.allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		f_v);
	
	row_scheme = NEW_INT(nb_row_classes * nb_col_classes);
	col_scheme = NEW_INT(nb_row_classes * nb_col_classes);

	get_row_decomposition_scheme(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, f_v);

	row_scheme_to_col_scheme(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, col_scheme, f_v);
	
	PStack.print_tactical_decomposition_scheme_tex_internal(ost, f_enter_math, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		row_scheme, col_scheme, FALSE /* f_print_subscripts */);

	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	FREE_INT(row_scheme);
	FREE_INT(col_scheme);
}

void incidence_structure::get_scheme(
	INT *&row_classes, INT *&row_class_inv, INT &nb_row_classes,
	INT *&col_classes, INT *&col_class_inv, INT &nb_col_classes,
	INT *&scheme, INT f_row_scheme, partitionstack &PStack)
{
	INT verbose_level = 0;

	PStack.allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		verbose_level);
	
	scheme = NEW_INT(nb_row_classes * nb_col_classes);
	if (f_row_scheme) {
		get_row_decomposition_scheme(PStack, 
			row_classes, row_class_inv, nb_row_classes,
			col_classes, col_class_inv, nb_col_classes, 
			scheme, verbose_level);
		}
	else {
		get_col_decomposition_scheme(PStack, 
			row_classes, row_class_inv, nb_row_classes,
			col_classes, col_class_inv, nb_col_classes, 
			scheme, verbose_level);
		}
}

void incidence_structure::free_scheme(
	INT *row_classes, INT *row_class_inv, 
	INT *col_classes, INT *col_class_inv, 
	INT *scheme)
{
	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	FREE_INT(scheme);
}

void incidence_structure::get_and_print_row_tactical_decomposition_scheme_tex(
	ostream &ost, INT f_enter_math, INT f_print_subscripts, partitionstack &PStack)
{
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT *row_scheme; //, *col_scheme;
	INT verbose_level = 0;
	INT f_v = FALSE;
	
	if (f_v) {
		cout << "incidence_structure::get_and_print_row_tactical_decomposition_scheme_tex computing row scheme" << endl;
		}
	PStack.allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		verbose_level);
	
	row_scheme = NEW_INT(nb_row_classes * nb_col_classes);
	//col_scheme = NEW_INT(nb_row_classes * nb_col_classes);

	if (f_v) {
		cout << "incidence_structure::get_and_print_row_tactical_decomposition_scheme_tex before get_row_decomposition_scheme" << endl;
		}
	get_row_decomposition_scheme(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, verbose_level);

	
	if (f_v) {
		cout << "incidence_structure::get_and_print_row_tactical_decomposition_scheme_tex before PStack.print_row_tactical_decomposition_scheme_tex" << endl;
		}
	PStack.print_row_tactical_decomposition_scheme_tex(ost, f_enter_math, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		row_scheme, f_print_subscripts);

	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	FREE_INT(row_scheme);
	//FREE_INT(col_scheme);
}

void incidence_structure::get_and_print_column_tactical_decomposition_scheme_tex(
	ostream &ost, INT f_enter_math, INT f_print_subscripts, partitionstack &PStack)
{
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT *col_scheme;
	INT verbose_level = 0;
	INT f_v = FALSE;
	
	if (f_v) {
		cout << "incidence_structure::get_and_print_column_tactical_decomposition_scheme_tex computing column scheme" << endl;
		}
	PStack.allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		verbose_level);
	
	//row_scheme = NEW_INT(nb_row_classes * nb_col_classes);
	col_scheme = NEW_INT(nb_row_classes * nb_col_classes);

	get_col_decomposition_scheme(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		col_scheme, verbose_level);

	
	PStack.print_column_tactical_decomposition_scheme_tex(ost, f_enter_math, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		col_scheme, f_print_subscripts);

	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	//FREE_INT(row_scheme);
	FREE_INT(col_scheme);
}

void incidence_structure::print_non_tactical_decomposition_scheme_tex(
	ostream &ost, INT f_enter_math, partitionstack &PStack)
{
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT *row_scheme;
	INT verbose_level = 0;
	INT f_v = FALSE;
	
	if (f_v) {
		cout << "incidence_structure::print_non_tactical_decomposition_scheme_tex" << endl;
		}
	PStack.allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		verbose_level);
	
	row_scheme = NEW_INT(nb_row_classes * nb_col_classes);
	//col_scheme = NEW_INT(nb_row_classes * nb_col_classes);

	get_row_decomposition_scheme_if_possible(PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, verbose_level);

	
	PStack.print_row_tactical_decomposition_scheme_tex(ost, f_enter_math, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		row_scheme, FALSE /* f_print_subscripts */);

	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	FREE_INT(row_scheme);
}

void incidence_structure::print_line(ostream &ost, partitionstack &P, 
	INT row_cell, INT i, INT *col_classes, INT nb_col_classes, 
	INT width, INT f_labeled)
{
	INT f1, f2, e1, e2;
	INT J, j, h, l, cell;
	INT first_column_element = P.startCell[1];
	
	f1 = P.startCell[row_cell];
	e1 = P.pointList[f1 + i];
	if (f_labeled) {
		ost << setw((int) width) << e1;
		}
	for (J = 0; J <= nb_col_classes; J++) {
		ost << "|";
		if (J == nb_col_classes)
			break;
		cell = col_classes[J];
		f2 = P.startCell[cell];
		l = P.cellSize[cell];
		for (j = 0; j < l; j++) {
			e2 = P.pointList[f2 + j] - first_column_element;
			if (get_ij(e1, e2)) {
				if (f_labeled) {
					ost << setw((int) width) << e1 * nb_lines() + e2;
					}
				else {
					ost << "X";
					//ost << "1";
					}
				}
			else {
				if (f_labeled) {
					for (h = 0; h < width - 1; h++) {
						ost << " ";
						}
					ost << ".";
					}
				else {
					ost << ".";
					//ost << "0";
					}
				}
			}
		}
	//ost << endl;
}

void incidence_structure::print_column_labels(ostream &ost, partitionstack &P, 
	INT *col_classes, INT nb_col_classes, INT width)
{
	INT f2, e2;
	INT J, j, h, l, cell;
	INT first_column_element = P.startCell[1];
	
	for (h = 0; h < width; h++) {
		ost << " ";
		}
	for (J = 0; J <= nb_col_classes; J++) {
		ost << "|";
		if (J == nb_col_classes)
			break;
		cell = col_classes[J];
		f2 = P.startCell[cell];
		l = P.cellSize[cell];
		for (j = 0; j < l; j++) {
			e2 = P.pointList[f2 + j] - first_column_element;
			ost << setw((int) width) << e2 + first_column_element;
			}
		}
	ost << endl;
}

void incidence_structure::print_hline(ostream &ost, partitionstack &P, 
	INT *col_classes, INT nb_col_classes, INT width, INT f_labeled)
{
	INT J, j, h, l, cell;
	
	if (f_labeled) {
		for (h = 0; h < width; h++) {
			ost << "-";
			}
		}
	else {
		//ost << "-";
		}
	for (J = 0; J <= nb_col_classes; J++) {
		ost << "+";
		if (J == nb_col_classes)
			break;
		cell = col_classes[J];
		l = P.cellSize[cell];
		for (j = 0; j < l; j++) {
			if (f_labeled) {
				for (h = 0; h < width; h++) {
					ost << "-";
					}
				}
			else {
				ost << "-";
				}
			}
		}
	ost << endl;
}

void incidence_structure::print_partitioned(ostream &ost, partitionstack &P, INT f_labeled)
{
	//INT *A;
	INT *row_classes;
	INT nb_row_classes;
	INT *col_classes;
	INT nb_col_classes;
	INT I, i, cell, l;
	INT width;
	INT mn;
	
	mn = nb_points() * nb_lines();
	
	width = INT_log10(mn) + 1;
	
	//A = get_incidence_matrix();
		
	row_classes = NEW_INT(nb_points() + nb_lines());
	col_classes = NEW_INT(nb_points() + nb_lines());
	P.get_row_and_col_classes(row_classes, nb_row_classes, 
		col_classes, nb_col_classes, 0 /* verbose_level */);
	//ost << "nb_row_classes = " << nb_row_classes << endl;
	//ost << "nb_col_classes = " << nb_col_classes << endl;
	
	if (f_labeled) {
		print_column_labels(ost, P, 
			col_classes, nb_col_classes, width);
		}
	for (I = 0; I <= nb_row_classes; I++) {
		print_hline(ost, P, col_classes, nb_col_classes, width, f_labeled);
		cell = row_classes[I];
		if (I < nb_row_classes) {
			l = P.cellSize[cell];
			for (i = 0; i < l; i++) {
				print_line(ost, P, cell, i, col_classes, nb_col_classes, width, f_labeled);
				ost << endl;
				}
			}
		}
	FREE_INT(row_classes);
	FREE_INT(col_classes);
	//FREE_INT(A);
}

void incidence_structure::point_collinearity_graph(INT *Adj, INT verbose_level)
// G[nb_points() * nb_points()]
{
	INT f_v = (verbose_level >= 1);
	INT i, j, h, l, u;
	
	if (f_v) {
		cout << "incidence_structure::point_collinearity_graph" << endl;
		}
	for (i = 0; i < nb_points(); i++) {
		for (j = 0; j < nb_points(); j++) {
			Adj[i * nb_points() + j] = 0;
			}
		}
	for (i = 0; i < nb_points(); i++) {
		for (h = 0; h < nb_lines_on_point[i]; h++) {
			l = lines_on_point[i * max_r + h];
			for (u = 0; u < nb_points_on_line[l]; u++) {
				j = points_on_line[l * max_k + u];
				if (j == i) {
					continue;
					}
				Adj[i * nb_points() + j] = 1;
				Adj[j * nb_points() + i] = 1;
				}
			}
		}
	if (f_v) {
		cout << "incidence_structure::point_collinearity_graph the graph is:" << endl;
		print_integer_matrix_width(cout, Adj, nb_points(), nb_points(), nb_points(), 1);
		}
}

void incidence_structure::line_intersection_graph(INT *Adj, INT verbose_level)
// G[nb_lines() * nb_lines()]
{
	INT f_v = (verbose_level >= 1);
	INT i, j, h, l, m, u;
	
	if (f_v) {
		cout << "incidence_structure::line_intersection_graph" << endl;
		}
	for (i = 0; i < nb_lines(); i++) {
		for (j = 0; j < nb_lines(); j++) {
			Adj[i * nb_lines() + j] = 0;
			}
		}
	for (l = 0; l < nb_lines(); l++) {
		for (u = 0; u < nb_points_on_line[l]; u++) {
			i = points_on_line[l * max_k + u];
			for (h = 0; h < nb_lines_on_point[i]; h++) {
				m = lines_on_point[i * max_r + h];
				if (l == m) {
					continue;
					}
				Adj[l * nb_lines() + m] = 1;
				Adj[m * nb_lines() + l] = 1;
				}
			}
		}
	if (f_v) {
		cout << "incidence_structure::line_intersection_graph the graph is:" << endl;
		print_integer_matrix_width(cout, Adj, nb_lines(), nb_lines(), nb_lines(), 1);
		}
}

void incidence_structure::latex_it(ostream &ost, partitionstack &P)
{
	INT nb_V, nb_B;
	INT *Vi, *Bj;
	INT *R;
	INT *X;
	INT f_v = TRUE;

	if (f_v) {
		cout << "latex_it" << endl;
		}
	rearrange(Vi, nb_V, Bj, nb_B, R, X, P);

	incma_latex(ost, 
		nb_points(), nb_lines(), 
		nb_V, nb_B, Vi, Bj, 
		R, X, max_r);

	FREE_INT(Vi);
	FREE_INT(Bj);
	FREE_INT(R);
	FREE_INT(X);


}

void incidence_structure::rearrange(INT *&Vi, INT &nb_V, 
	INT *&Bj, INT &nb_B, INT *&R, INT *&X, partitionstack &P)
{
	INT *row_classes;
	INT nb_row_classes;
	INT *col_classes;
	INT nb_col_classes;
	//INT *Vi, *Bj;
	INT *row_perm;
	INT *row_perm_inv;
	INT *col_perm;
	INT *col_perm_inv;
	INT i, j, ii, jj, c, a, h;
	//INT *R;
	//INT *X;

	row_classes = NEW_INT(nb_points() + nb_lines());
	col_classes = NEW_INT(nb_points() + nb_lines());
	P.get_row_and_col_classes(row_classes, nb_row_classes, 
		col_classes, nb_col_classes, 0 /* verbose_level */);
	//ost << "nb_row_classes = " << nb_row_classes << endl;
	//ost << "nb_col_classes = " << nb_col_classes << endl;

	nb_V = nb_row_classes;
	nb_B = nb_col_classes;
	Vi = NEW_INT(nb_row_classes);
	Bj = NEW_INT(nb_col_classes);
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

	row_perm = NEW_INT(nb_points());
	row_perm_inv = NEW_INT(nb_points());
	col_perm = NEW_INT(nb_lines());
	col_perm_inv = NEW_INT(nb_lines());
	
	P.get_row_and_col_permutation(
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		row_perm, row_perm_inv, 
		col_perm, col_perm_inv);
	
#if 0
	cout << "row_perm:" << endl;
	INT_vec_print(cout, row_perm, nb_points());
	cout << endl;
	cout << "col_perm:" << endl;
	INT_vec_print(cout, col_perm, nb_lines());
	cout << endl;
#endif

	R = NEW_INT(nb_points());
	X = NEW_INT(nb_points() * max_r);

	for (i = 0; i < nb_points(); i++) {
		ii = row_perm_inv[i];
		R[i] = nb_lines_on_point[ii];
		for (h = 0; h < nb_lines_on_point[ii]; h++) {
			jj = lines_on_point[ii * max_r + h];
			j = col_perm[jj];
			X[i * max_r + h] = j;
			}
		INT_vec_heapsort(X + i * max_r, nb_lines_on_point[ii]);
		} 
	


	FREE_INT(row_classes);
	FREE_INT(col_classes);
	//FREE_INT(Vi);
	//FREE_INT(Bj);
	FREE_INT(row_perm);
	FREE_INT(row_perm_inv);
	FREE_INT(col_perm);
	FREE_INT(col_perm_inv);
	//FREE_INT(R);
	//FREE_INT(X);
}



void incidence_structure::decomposition_print_tex(ostream &ost, 
	partitionstack &PStack, INT f_row_tactical, INT f_col_tactical, 
	INT f_detailed, INT f_local_coordinates, INT verbose_level)
{
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "incidence_structure::decomposition_print_tex get decomposition" << endl;
		}
	PStack.allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		verbose_level);

	ost << "\\subsection*{Decomposition}" << endl;
	PStack.print_decomposition_tex(ost, row_classes, nb_row_classes, 
		col_classes, nb_col_classes);
	
	if (f_row_tactical) {
		INT *row_scheme;
		row_scheme = NEW_INT(nb_row_classes * nb_col_classes);
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
				ost, FALSE /* f_enter_math_mode */, 
				row_classes, row_class_inv, nb_row_classes,
				col_classes, col_class_inv, nb_col_classes, 
				f_local_coordinates, 0 /*verbose_level*/);
			}
		FREE_INT(row_scheme);
		}
	if (f_col_tactical) {
		INT *col_scheme;
		col_scheme = NEW_INT(nb_row_classes * nb_col_classes);
		get_col_decomposition_scheme(PStack, 
			row_classes, row_class_inv, nb_row_classes,
			col_classes, col_class_inv, nb_col_classes, 
			col_scheme, 0);
		if (f_v) {
			cout << "incidence_structure::decomposition_print_tex col_scheme:" << endl;
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
				ost, FALSE /* f_enter_math_mode */, 
				row_classes, row_class_inv, nb_row_classes,
				col_classes, col_class_inv, nb_col_classes, 
				f_local_coordinates, 0 /*verbose_level*/);
			}
		FREE_INT(col_scheme);
		}


	


	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
}



void incidence_structure::do_tdo_high_level(partitionstack &S, 
	INT f_tdo_steps, INT f_tdo_depth, INT tdo_depth, 
	INT f_write_tdo_files, INT f_pic, 
	INT f_include_tdo_scheme, INT f_include_tdo_extra, INT f_write_tdo_class_files, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT TDO_depth;

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
		BYTE fname[1000];
		INT *row_classes, *row_class_inv, nb_row_classes;
		INT *col_classes, *col_class_inv, nb_col_classes;
		INT i;

		S.allocate_and_get_decomposition(
			row_classes, row_class_inv, nb_row_classes,
			col_classes, col_class_inv, nb_col_classes, 
			verbose_level - 1);

		for (i = 0; i < nb_row_classes; i++) {
			sprintf(fname, "%s_TDO_point_class_%ld.txt", label, i);
			S.write_cell_to_file_points_or_lines(row_classes[i], fname, verbose_level - 1);
			}
		for (i = 0; i < nb_col_classes; i++) {
			sprintf(fname, "%s_TDO_line_class_%ld.txt", label, i);
			S.write_cell_to_file_points_or_lines(col_classes[i], fname, verbose_level - 1);
			}
		FREE_INT(row_classes);
		FREE_INT(row_class_inv);
		FREE_INT(col_classes);
		FREE_INT(col_class_inv);
		}
}





void incidence_structure::compute_tdo(partitionstack &S, 
	INT f_write_tdo_files, 
	INT f_pic, 
	INT f_include_tdo_scheme, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	BYTE fname[1000];
	INT f_list_incidences = FALSE;

	if (f_v) {
		cout << "incidence_structure_compute_tdo" << endl;
		}
	
	INT N;
	

	if (f_v) {
		cout << "incidence_structure_compute_tdo initial partition:" << endl;
		cout << S << endl;
		}
	N = nb_points() + nb_lines();

	INT TDO_depth = N;
	INT f_labeled = TRUE;

	compute_TDO_safe(S, TDO_depth, verbose_level - 1);

	if (f_vv) {
		print_partitioned(cout, S, f_labeled);
		}

	if (f_v) {
		cout << "TDO:" << endl;
		if (TDO_depth < N) {
			if (EVEN(TDO_depth)) {
				get_and_print_row_decomposition_scheme(S, f_list_incidences, FALSE);
				}
			else {
				get_and_print_col_decomposition_scheme(S, f_list_incidences, FALSE);
				}
			}
		else {
			get_and_print_decomposition_schemes(S);
			S.print_classes_points_and_lines(cout);
			}
		}

	if (f_write_tdo_files) {
		sprintf(fname, "%s_tdo_scheme.tex", label);
		{
		ofstream fp(fname);

			//fp << "$$" << endl;
			get_and_print_tactical_decomposition_scheme_tex(fp, TRUE /* f_enter_math */, S);
			//fp << "$$" << endl;
		}
		if (f_v) {
			cout << "written file " << fname << " of size " << file_size(fname) << endl;
			}

		sprintf(fname, "%s_tdo.tex", label);
		{
		ofstream fp(fname);

		if (f_include_tdo_scheme) {
			fp << "$\\begin{array}{c}" << endl;
			if (f_pic) {
				latex_it(fp, S);
				fp << "\\\\" << endl;
				}
			get_and_print_tactical_decomposition_scheme_tex(
				fp, FALSE /* f_enter_math */, S);
			fp << "\\end{array}$" << endl;
			}
		else {
			latex_it(fp, S);
			}
		}
		if (f_v) {
			cout << "written file " << fname << " of size " << file_size(fname) << endl;
			}
		}

}

void incidence_structure::compute_tdo_stepwise(partitionstack &S, 
	INT TDO_depth, 
	INT f_write_tdo_files, 
	INT f_pic, 
	INT f_include_tdo_scheme, 
	INT f_include_extra, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	BYTE fname[1000];
	BYTE fname_pic[1000];
	BYTE fname_scheme[1000];
	BYTE fname_extra[1000];
	INT step, f_refine, f_refine_prev, f_done;
	INT f_local_coordinates = FALSE;
	INT f_list_incidences = FALSE;

	if (f_v) {
		cout << "incidence_structure::compute_tdo_stepwise" << endl;
		}
	

	compute_TDO_safe_first(S, 
		TDO_depth, step, f_refine, f_refine_prev, verbose_level);

	S.sort_cells();

	f_done = FALSE;
	while (TRUE) {
		if (f_v) {
			cout << "incidence_structure::compute_tdo_stepwise TDO step=" << step << " ht=" << S.ht << endl;
			if (step == 0) {
				print_non_tactical_decomposition_scheme_tex(
						cout, FALSE /* f_enter_math */, S);
				}
			else if (EVEN(step - 1)) {

				get_and_print_col_decomposition_scheme(S, f_list_incidences, FALSE);
				get_and_print_column_tactical_decomposition_scheme_tex(
					cout, TRUE /* f_enter_math */, FALSE /* f_print_subscripts */, S);
				}
			else {

				get_and_print_row_decomposition_scheme(S, f_list_incidences, FALSE);
				get_and_print_row_tactical_decomposition_scheme_tex(
					cout, TRUE /* f_enter_math */, FALSE /* f_print_subscripts */, S);
				}
			S.print_classes_points_and_lines(cout);
			}
		if (f_write_tdo_files) {
			sprintf(fname, "%s_tdo_step_%ld.tex", label, step);
			sprintf(fname_pic, "%s_tdo_step_%ld_pic.tex", label, step);
			sprintf(fname_scheme, "%s_tdo_step_%ld_scheme.tex", label, step);
			sprintf(fname_extra, "%s_tdo_step_%ld_extra.tex", label, step);
			{
			ofstream fp(fname);
			ofstream fp_pic(fname_pic);
			ofstream fp_scheme(fname_scheme);
			ofstream fp_extra(fname_extra);

			if (f_include_tdo_scheme) {
				fp << "\\subsection*{Step $" << step << "$, Height $" << S.ht << "$}" << endl;
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
						fp_scheme, FALSE /* f_enter_math */, S);
					}
				else if (EVEN(step - 1)) {
					get_and_print_column_tactical_decomposition_scheme_tex(
						fp_scheme, FALSE /* f_enter_math */, FALSE /* f_print_subscripts */, S);
					}
				else {
					get_and_print_row_tactical_decomposition_scheme_tex(
						fp_scheme, FALSE /* f_enter_math */, FALSE /* f_print_subscripts */, S);
					}
				fp << "\\end{array}$" << endl;
				
				if (f_include_extra) {
					if (step == 0) {
						decomposition_print_tex(fp_extra, S, FALSE, FALSE, FALSE, f_local_coordinates, verbose_level);
						fp << "\\input " << fname_extra << endl;
						}
					else if (EVEN(step - 1)) {
						decomposition_print_tex(fp_extra, S, FALSE, TRUE, TRUE, f_local_coordinates, verbose_level);
						fp << "\\input " << fname_extra << endl;
						//PStack.print_classes_points_and_lines(cout);
						}
					else {
						decomposition_print_tex(fp_extra, S, TRUE, FALSE, TRUE, f_local_coordinates, verbose_level);
						fp << "\\input " << fname_extra << endl;
						}
					}
				}
			else {
				latex_it(fp_pic, S);
				}
			}
			if (f_v) {
				cout << "written file " << fname << " of size " << file_size(fname) << endl;
				cout << "written file " << fname_pic << " of size " << file_size(fname_pic) << endl;
				cout << "written file " << fname_scheme << " of size " << file_size(fname_scheme) << endl;
				cout << "written file " << fname_extra << " of size " << file_size(fname_extra) << endl;
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
		INT f_labeled = FALSE;
		
		print_partitioned(cout, S, f_labeled);
		}

	if (f_v) {
		cout << "TDO:" << endl;
		if (TDO_depth < nb_points() + nb_lines()) {
			if (EVEN(TDO_depth)) {
				get_and_print_row_decomposition_scheme(S, f_list_incidences, FALSE);
				}
			else {
				get_and_print_col_decomposition_scheme(S, f_list_incidences, FALSE);
				}
			}
		else {
			get_and_print_decomposition_schemes(S);
			S.print_classes_points_and_lines(cout);
			}
		}

	if (f_write_tdo_files) {
		sprintf(fname, "%s_tdo.tex", label);
		sprintf(fname_pic, "%s_tdo_pic.tex", label);
		sprintf(fname_scheme, "%s_tdo_scheme.tex", label);
		{
		ofstream fp(fname);
		ofstream fp_pic(fname_pic);
		ofstream fp_scheme(fname_scheme);

		if (f_include_tdo_scheme) {
			fp << "\\subsection*{The TDO at Height $" << S.ht << "$}" << endl;
			fp << "$\\begin{array}{c}" << endl;
			if (f_pic) {
				fp << "\\input " << fname_pic << endl;
				latex_it(fp_pic, S);
				fp << "\\\\" << endl;
				}
			fp << "\\input " << fname_scheme << endl;
			get_and_print_tactical_decomposition_scheme_tex(
				fp_scheme, FALSE /* f_enter_math */, S);
			fp << "\\end{array}$" << endl;
			}
		else {
			latex_it(fp_pic, S);
			}
		}
		if (f_v) {
			cout << "written file " << fname << " of size " << file_size(fname) << endl;
			cout << "written file " << fname_pic << " of size " << file_size(fname_pic) << endl;
			cout << "written file " << fname_scheme << " of size " << file_size(fname_scheme) << endl;
			}
		}

}

void incidence_structure::init_partitionstack_trivial(partitionstack *S, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT N;


	if (f_v) {
		cout << "incidence_structure::init_partitionstack_trivial" << endl;
		}
	N = nb_points() + nb_lines();
	
	S->allocate(N, 0);

	// split off the column class:
	S->subset_continguous(nb_points(), nb_lines());
	S->split_cell(0);

}

void incidence_structure::init_partitionstack(partitionstack *S, 
	INT f_row_part, INT nb_row_parts, INT *row_parts,
	INT f_col_part, INT nb_col_parts, INT *col_parts,
	INT nb_distinguished_point_sets, INT **distinguished_point_sets, INT *distinguished_point_set_size, 
	INT nb_distinguished_line_sets, INT **distinguished_line_sets, INT *distinguished_line_set_size, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_v3 = (verbose_level >= 3);
	INT N;
	INT i, j, a, h;


	if (f_v) {
		cout << "incidence_structure::init_partitionstack" << endl;
		}
	N = nb_points() + nb_lines();
	
	S->allocate(N, 0);

	// split off the column class:
	S->subset_continguous(nb_points(), nb_lines());
	S->split_cell(0);



	if (f_row_part) {
		a = row_parts[0];
		for (i = 1; i < nb_row_parts; i++) {
			S->subset_continguous(a, nb_points() - a);
			S->split_cell(0);
			a += row_parts[i];
			}
		}
	if (f_col_part) {
		a = nb_points() + col_parts[0];
		for (i = 1; i < nb_col_parts; i++) {
			S->subset_continguous(a, nb_points() + nb_lines() - a);
			S->split_cell(0);
			a += col_parts[i];
			}
		}



	for (j = 0; j < nb_distinguished_point_sets; j++) {
		if (f_v) {
			cout << "splitting off " << j << "-th distinguished point set of size " << distinguished_point_set_size[j] << endl;
			}
		if (f_v3) {
			cout << "which is the following set of size " << distinguished_point_set_size[j] << ":" << endl;
			INT_vec_print(cout, distinguished_point_sets[j], distinguished_point_set_size[j]);
			cout << endl;
			}
		S->split_multiple_cells(distinguished_point_sets[j], distinguished_point_set_size[j], TRUE, verbose_level);
		if (f_vv) {
			cout << "incidence_structure::init_partitionstack partition:" << endl;
			S->print_classes_points_and_lines(cout);
			}
		}



	for (j = 0; j < nb_distinguished_line_sets; j++) {
		if (f_v) {
			cout << "splitting off " << j << "-th distinguished line set of size " << distinguished_line_set_size[j] << endl;
			}
		if (f_v3) {
			cout << "which is the following set of size " << distinguished_line_set_size[j] << ":" << endl;
			INT_vec_print(cout, distinguished_line_sets[j], distinguished_line_set_size[j]);
			cout << endl;
			}
		
		INT *set;
		INT set_sz;

		set_sz = distinguished_line_set_size[j];
		set = NEW_INT(set_sz);
		for (h = 0; h < set_sz; h++) {
			set[h] = distinguished_line_sets[j][h] + nb_points();
			}


		if (f_vv) {
			cout << "incidence_structure::init_partitionstack After adding Inc->nb_points():" << endl;
			INT_vec_print(cout, set, set_sz);
			cout << endl;
			}


		S->split_multiple_cells(set, set_sz, TRUE, verbose_level);
		FREE_INT(set);
		if (f_vv) {
			cout << "incidence_structure::init_partitionstack partition:" << endl;
			S->print_classes_points_and_lines(cout);
			}
		}

	if (f_vv) {
		cout << "incidence_structure::init_partitionstack we have arrived at the following partition:" << endl;
		S->print_classes_points_and_lines(cout);
		}
	if (f_v) {
		cout << "incidence_structure::init_partitionstack done" << endl;
		}

}

void incidence_structure::shrink_aut_generators(
	INT nb_distinguished_point_sets, 
	INT nb_distinguished_line_sets, 
	INT Aut_counter, INT *Aut, INT *Base, INT Base_length, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, h;
	INT m, n;
	INT nb_rows, nb_cols, total;

	if (f_v) {
		cout << "incidence_structure::shrink_aut_generators" << endl;
		}
	m = nb_points();
	n = nb_lines();

	nb_rows = m;
	nb_cols = n;

	nb_rows += nb_distinguished_line_sets;
	nb_cols += nb_distinguished_point_sets;

	total = nb_rows + nb_cols;

	for (h = 0; h < Aut_counter; h++) {
		for (i = 0; i < m; i++) {
			Aut[h * (m + n) + i] = Aut[h * total + i];
			}
		for (j = 0; j < n; j++) {
			Aut[h * (m + n) + m + j] = Aut[h * total + nb_rows + j] - nb_distinguished_line_sets;
			}
		for (i = Base_length - 1; i >= 0; i--) {
			if (Base[i] > m) {
				if (Base[i] < nb_rows) {
					cout << "Base[i] > m && Base[i] < nb_rows" << endl;
					exit(1);
					}
				Base[i] -= nb_distinguished_line_sets;
				}
			}
		}

	if (f_v) {
		cout << "incidence_structure::shrink_aut_generators done" << endl;
		}
}

void incidence_structure::print_aut_generators(INT Aut_counter, INT *Aut, 
	INT Base_length, INT *Base, INT *Transversal_length)
{
	INT m, n, i, j, h;
	INT *AUT;
	
	m = nb_points();
	n = nb_lines();
	AUT = NEW_INT(m + n);
	
	cout << "incidence_structure::print_aut_generators base_length = " << Base_length << endl;
	cout << "The base is : " << endl;
	for (i = Base_length - 1; i >= 0; i--) {
		cout << Base[i] << " ";
		}
	cout << endl;
	cout << "The transversal lengths are: " << endl;
	for (i = 0; i < Base_length; i++) {
		cout << Transversal_length[i] << " ";
		}
	cout << endl;
	cout << "We have found " << Aut_counter << " gens:" << endl;
	for (h = 0; h < Aut_counter; h++) {
		for (i = 0; i < m; i++) {
			AUT[i] = Aut[h * (m + n) + i];
			}
		for (j = 0; j < n; j++) {
			AUT[m + j] = Aut[h * (m + n) + m + j] - m;
			}
		cout << h << " : " ;
		cout << "generator " << h << ":" << endl;
		for (i = 0; i < m + n; i++)
			cout << AUT[i] << " ";
		cout << endl;
		perm_print_product_action(cout, AUT, m + n, m, 0 /* offset */, FALSE /* f_cycle_length */);
		cout << endl;
		//for ( j = 0; j < m + n; j++ ){
			//cout << Aut[h * (m + n) + j] << " ";
		//}
		//cout << endl;
		}
	cout << endl;
	
	FREE_INT(AUT);
}

void incidence_structure::compute_extended_collinearity_graph(
	INT *&Adj, INT &v, INT *&partition, 
	INT f_row_part, INT nb_row_parts, INT *row_parts,
	INT f_col_part, INT nb_col_parts, INT *col_parts,
	INT nb_distinguished_point_sets, INT **distinguished_point_sets, INT *distinguished_point_set_size, 
	INT nb_distinguished_line_sets, INT **distinguished_line_sets, INT *distinguished_line_set_size, 
	INT verbose_level)
// side effect: the distinguished sets will be sorted afterwards
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_v4 = (verbose_level >= 4);
	INT m = nb_points();
	INT n = nb_lines();
	INT i, j, J, j0, l, u, k, h, h1, h2, i1, i2, idx, y, z;
	INT v1, v2, v3;
	INT my_nb_col_parts;
	INT *my_col_parts;

	if (f_v) {
		cout << "incidence_structure::compute_extended_collinearity_graph" << endl;
		}
	v = v1 = m;
	if (f_col_part) {
		v += nb_col_parts - 1;
		}
	v2 = v;
	v += nb_distinguished_point_sets;
	v3 = v;
	for (i = 0; i < nb_distinguished_line_sets; i++) {
		v += distinguished_line_set_size[i];
		}
	if (f_v) {
		cout << "incidence_structure::compute_extended_collinearity_graph v=" << v << endl;
		}
	
	for (i = 0; i < nb_distinguished_point_sets; i++) {
		INT_vec_heapsort(distinguished_point_sets[i], distinguished_point_set_size[i]);
		}
	for (i = 0; i < nb_distinguished_line_sets; i++) {
		INT_vec_heapsort(distinguished_line_sets[i], distinguished_line_set_size[i]);
		}
	partition = NEW_INT(v);
	for (i = 0; i < v; i++) {
		partition[i] = 1;
		}
	Adj = NEW_INT(v * v);
	for (i = 0; i < v * v; i++) {
		Adj[i] = 0;
		}


	if (f_col_part) {
		my_nb_col_parts = nb_col_parts;
		my_col_parts = NEW_INT(my_nb_col_parts);
		for (i = 0; i < nb_col_parts; i++) {
			my_col_parts[i] = col_parts[i];
			}
		}
	else {
		my_nb_col_parts = 1;
		my_col_parts = NEW_INT(my_nb_col_parts);
		my_col_parts[0] = n;
		}
	j0 = 0;
	for (J = 0; J < my_nb_col_parts; J++) {
		l = my_col_parts[J];
		for (u = 0; u < l; u++) {
			j = j0 + u;
			k = nb_points_on_line[j];
			// keep track of which column class block j belongs to
			// we do this for the all column classes but the first
			if (J) {
				for (h = 0; h < k; h++) {
					i = points_on_line[j * max_k + h];
					Adj[i * v + v1 + J - 1] = 1;
					Adj[(v1 + J - 1) * v + i] = 1;
					}
				}
			// record the collinearity information resulting from block j:
			for (h1 = 0; h1 < k; h1++) {
				i1 = points_on_line[j * max_k + h1];
				for (h2 = h1 + 1; h2 < k; h2++) {
					i2 = points_on_line[j * max_k + h2];
					Adj[i1 * v + i2] = 1;
					Adj[i2 * v + i1] = 1;
					}
				}

#if 0
			// keep track of whether block j is in a distinguished line set:
			for (h = 0; h < nb_distinguished_line_sets; h++) {
				if (INT_vec_search(distinguished_line_sets[h], distinguished_line_set_size[h], j, idx)) {
					Adj[j * v + v3 + h] = 1;
					Adj[(v3 + h) * v + j] = 1;
					}
				}
#endif

			}
		}

	// record the distinguished line sets:
	z = v2;
	for (h = 0; h < nb_distinguished_line_sets; h++) {
		for (u = 0; u < distinguished_line_set_size[h]; u++) {
			j = distinguished_line_sets[h][u];
			k = nb_points_on_line[j];
			for (y = 0; y < k; y++) {
				i = points_on_line[j * max_k + y];
				Adj[i * v + z + u] = 1;
				Adj[(z + u) * v + i] = 1;
				}
			}
		z += distinguished_line_set_size[h];
		}


	// finally, we record the distinguished point sets:
	for (i = 0; i < m; i++) {
		for (h = 0; h < nb_distinguished_point_sets; h++) {
			if (INT_vec_search(distinguished_point_sets[h], distinguished_point_set_size[h], i, idx)) {
				Adj[i * v + v2 + h] = 1;
				Adj[(v2 + h) * v + i] = 1;
				}
			}
		}

	// initialize the partition:
	z = 0;
	if (f_row_part) {
		for (h = 0; h < nb_row_parts; h++) {
			partition[z + row_parts[h] - 1] = 0;
			z += row_parts[h];
			}
		}
	else {
		partition[m - 1] = 0;
		}
	z = v1;
	for (h = 0; h < my_nb_col_parts; h++) {
		partition[z + h] = 0;
		}
	z = v2;
	for (h = 0; h < nb_distinguished_point_sets; h++) {
		partition[z + h] = 0;
		}
	z = v3;
	for (h = 0; h < nb_distinguished_line_sets; h++) {
		l = distinguished_line_set_size[h];
		partition[z + l - 1] = 0;
		z += l;
		}
	if (f_vv) {
		cout << "incidence_structure::compute_extended_collinearity_graph Adj=" << endl;
		print_integer_matrix_width(cout, Adj, v, v, v, 1);
		}
	

	if (f_v) {
		cout << "incidence_structure::compute_extended_collinearity_graph done" << endl;
		}

	FREE_INT(my_col_parts);
}

void incidence_structure::compute_extended_matrix(
	INT *&M, INT &nb_rows, INT &nb_cols, INT &total, INT *&partition, 
	INT f_row_part, INT nb_row_parts, INT *row_parts,
	INT f_col_part, INT nb_col_parts, INT *col_parts,
	INT nb_distinguished_point_sets, INT **distinguished_point_sets, INT *distinguished_point_set_size, 
	INT nb_distinguished_line_sets, INT **distinguished_line_sets, INT *distinguished_line_set_size, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_v4 = (verbose_level >= 4);
	INT m = nb_points();
	INT n = nb_lines();
	INT i, j, h, a;

	if (f_v) {
		cout << "incidence_structure::compute_extended_matrix" << endl;
		}
	
	nb_rows = m;
	nb_cols = n;

	nb_rows += nb_distinguished_line_sets;
	nb_cols += nb_distinguished_point_sets;

	total = nb_rows + nb_cols;
	
	if (f_vv) {
		cout << "nb_distinguished_line_sets=" << nb_distinguished_line_sets << endl;
		cout << "nb_distinguished_point_sets=" << nb_distinguished_point_sets << endl;
		cout << "m=" << m << endl;
		cout << "n=" << n << endl;
		cout << "nb_rows=" << nb_rows << endl;
		cout << "nb_cols=" << nb_cols << endl;
		cout << "total=" << total << endl;
		}

	partition = NEW_INT(total);
	M = NEW_INT(nb_rows * nb_cols);
	for (i = 0; i < nb_rows * nb_cols; i++) {
		M[i] = 0;
		}
	for (i = 0; i < m; i++) {
		for (h = 0; h < nb_lines_on_point[i]; h++) {
			a = lines_on_point[i * max_r + h];
			M[i * nb_cols + a] = 1;
			}
		}


	for (j = 0; j < nb_distinguished_point_sets; j++) {
		if (f_v) {
			cout << "The " << j << "-th distinguished point set is:" << endl;
			INT_vec_print(cout, distinguished_point_sets[j], distinguished_point_set_size[j]);
			cout << endl;
			}
		for (i = 0; i < distinguished_point_set_size[j]; i++) {
			a = distinguished_point_sets[j][i];
			M[a * nb_cols + n + j] = 1;
			}
		}


	for (j = 0; j < nb_distinguished_line_sets; j++) {
		if (f_v) {
			cout << "The " << j << "-th distinguished line set is:" << endl;
			INT_vec_print(cout, distinguished_line_sets[j], distinguished_line_set_size[j]);
			cout << endl;
			}
		for (i = 0; i < distinguished_line_set_size[j]; i++) {
			a = distinguished_line_sets[j][i];
			M[(m + j) * nb_cols + a] = 1;
			}
		}


	if (f_v4) {
		cout << "incidence_structure::compute_extended_matrix The extended incidence matrix is:" << endl;
		print_integer_matrix_width(cout, M, nb_rows, nb_cols, nb_cols, 1);
		}


	for (i = 0; i < total; i++) {
		partition[i] = 1;
		}

	if (f_row_part) {
		INT a;
		a = 0;
		for (i = 0; i < nb_row_parts; i++) {
			a += row_parts[i];
			partition[a - 1] = 0;
			}
		}
	else {
		partition[m - 1] = 0;
		}
	if (f_col_part) {
		INT a;
		a = nb_rows;
		for (i = 0; i < nb_col_parts; i++) {
			a += col_parts[i];
			partition[a - 1] = 0;
			}
		}
	else {
		partition[nb_rows + n - 1] = 0;
		}
#if 0
	for (i = 0; i < PB.P.ht; i++) {
		j = PB.P.startCell[i] + PB.P.cellSize[i] - 1;
		if (PB.P.startCell[i] >= m) {
			j += nb_distinguished_line_sets;
			}
		partition[j] = 0;
		}
#endif

	for (i = 0; i < nb_distinguished_line_sets; i++) {
		partition[m + i] = 0;
		}

	for (j = 0; j < nb_distinguished_point_sets; j++) {
		partition[nb_rows + n + j] = 0;
		}
	if (f_v4) {
		cout << "incidence_structure::compute_extended_matrix The partition is:" << endl;
		INT_vec_print(cout, partition, total);
		cout << endl;
		}

	
	if (f_v) {
		cout << "incidence_structure::compute_extended_matrix done" << endl;
		}
}




// ####################################################################################
// a few global functions:
// ####################################################################################



// two functions from DISCRETA1:
// adapted to use ostream instead of FILE pointer

void incma_latex_picture(ostream &fp, 
	INT width, INT width_10, 
	INT f_outline_thin, const BYTE *unit_length, 
	const BYTE *thick_lines, const BYTE *thin_lines, const BYTE *geo_line_width, 
	INT v, INT b, 
	INT V, INT B, INT *Vi, INT *Bj, 
	INT *R, INT *X, INT dim_X, 
	INT f_labelling_points, const BYTE **point_labels, 
	INT f_labelling_blocks, const BYTE **block_labels)
// width for one box in 0.1mm 
// width_10 is 1 10th of width
// example: width = 40, width_10 = 4 
{
	INT w, h, w1, h1;
	INT i, j, k, r, a;
	INT x0, y0, x1, y1;
	INT X0, Y0, X1, Y1;
	INT width_8, width_5;
	const BYTE *tdo_line_width = thick_lines; // "0.7mm";
	const BYTE *line_width = thin_lines; // "0.15mm";
	// BYTE *geo_line_width = "0.25mm";
	
	width_8 = width - 2 * width_10;
	width_5 = width >> 1;
	fp << "\\unitlength" << unit_length << endl;
	w = b * width;
	h = v * width;
	w1 = w;
	h1 = h;
	if (f_labelling_points) {
		w1 += 2 * width;
		}
	if (f_labelling_blocks) {
		h1 += 2 * width;
		}
	fp << "\\begin{picture}(" << w1 << "," << h1 << ")" << endl;

	// the grid: 
	fp << "\\linethickness{" << tdo_line_width << "}" << endl;
	k = 0;
	for (i = -1; i < B; i++) {
		if (i >= 0) {
			a = Bj[i];
			k += a;
			}
		if (f_outline_thin) {
			if (i == -1 || i == B - 1) {
				continue;
				}
			}
		fp << "\\put(" << k * width << ",0){\\line(0,1){" << h << "}}" << endl;
		}
	if (k != b) {
		cout << "incma_latex_picture: k != b" << endl;
		exit(1);
		}
	k = 0;
	for (i = -1; i < V; i++) {
		if (i >= 0) {
			a = Vi[i];
			k += a;
			}
		if (f_outline_thin) {
			if (i == -1 || i == V - 1) {
				continue;
				}
			}
		fp << "\\put(0," << h - k * width << "){\\line(1,0){" << w << "}}" << endl;
		}
	if (k != v) {
		cout << "incma_latex_picture: k != v" << endl;
		exit(1);
		}

	// labeling of points:
	if (f_labelling_points) {
		for (i = 0; i < v; i++) {
			fp << "\\put(0," << h - i * width - width_5 
				<< "){\\makebox(0,0)[r]{" << point_labels[i] << "$\\,$}}" << endl;
			}
		}
	else {
		for (i = 0; i < v; i++) {
			fp << "\\put(0," << h - i * width - width_5 
				<< "){\\makebox(0,0)[r]{}}" << endl;
			}
		}

	// labeling of blocks:
	if (f_labelling_blocks) {
		for (i = 0; i < b; i++) {
			fp << "\\put(" << i * width + width_5 << "," << h + width_5 
				<< "){\\makebox(0,0)[b]{" << block_labels[i] << "}}" << endl;
			}
		}
	else {
		for (i = 0; i < b; i++) {
			fp << "\\put(" << i * width + width_5 << "," << h + width_5 
				<< "){\\makebox(0,0)[b]{}}" << endl;
			}
		}

	// the grid:
	fp << "\\linethickness{" << line_width << "}" << endl;
	fp << "\\multiput(0,0)(" << width << ",0){" << b + 1 << "}{\\line(0,1){" << h 
		<< "}}" << endl;
	fp << "\\multiput(0,0)(0," << width << "){" << v + 1 << "}{\\line(1,0){" << w << "}}" << endl;

	// the incidence matrix itself: 
	fp << "\\linethickness{" << geo_line_width << "}" << endl;
	for (i = 0; i < v; i++) {
		y0 = h - i * width;
		y1 = h - (i + 1) * width;
		Y0 = y0 - width_10;
		Y1 = y1 + width_10;
		for (r = 0; r < R[i]; r++) {
			j = X[i * dim_X + r];
			// printf("%ld ", j);
			x0 = j * width;
			x1 = (j + 1) * width;
			X0 = x0 + width_10;
			X1 = x1 - width_10;
			// hor. lines: 
			fp << "\\put(" << X0 << "," << Y0 << "){\\line(1,0){" << width_8 << "}}" << endl;
			fp << "\\put(" << X0 << "," << Y1 << "){\\line(1,0){" << width_8 << "}}" << endl;

			// vert. lines: 
			fp << "\\put(" << X0 << "," << Y1 << "){\\line(0,1){" << width_8 << "}}" << endl;
			fp << "\\put(" << X1 << "," << Y1 << "){\\line(0,1){" << width_8 << "}}" << endl;

			}
		// printf("\n");
		}

	fp << "\\end{picture}" << endl;
}


static INT incma_latex_unit_length_nb = 0;
static const BYTE *incma_latex_unit_length[100];



void incma_latex(ostream &fp, 
	INT v, INT b, 
	INT V, INT B, INT *Vi, INT *Bj, 
	INT *R, INT *X, INT dim_X)
{
	const BYTE *unit_length;

	if (incma_latex_unit_length_nb == 0) {
		unit_length = "0.065mm";
		}
	else {
		unit_length = incma_latex_unit_length[incma_latex_unit_length_nb - 1];
		}
	incma_latex_picture(fp, 
		40 /* width */, 
		10 /* width_10 */,  
		FALSE /* f_outline_thin */, 
		unit_length /* unit_length */, 
		"0.5mm" /* thick_lines */ , 
		"0.15mm" /* thin_lines */ , 
		"0.25mm" /* geo_line_width */ , 
		v, b, V, B, Vi, Bj, R, X, dim_X, 
		FALSE /* f_labelling_points */, NULL, 
		FALSE /* f_labelling_blocks */, NULL);
}


void incma_latex_override_unit_length(const BYTE *override_unit_length)
{
	incma_latex_unit_length[incma_latex_unit_length_nb++] = override_unit_length;
}

void incma_latex_override_unit_length_drop()
{
	incma_latex_unit_length_nb--;
}





