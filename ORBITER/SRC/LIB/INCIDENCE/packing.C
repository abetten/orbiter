// packing.C
// Anton Betten
//
// started: December 2006
// moved here from refine: 1/3/10

#include "galois.h"
#include "incidence.h"

INT TDO_upper_bounds_v_max_init = 12;
INT TDO_upper_bounds_v_max = -1;
INT *TDO_upper_bounds_table = NULL;
INT *TDO_upper_bounds_table_source = NULL;
	// 0 = nothing
	// 1 = packing number
	// 2 = braun test
	// 3 = maxfit
INT TDO_upper_bounds_initial_data[] = {
3,3,1,
4,3,1,
5,3,2,
6,3,4,
7,3,7,
8,3,8,
9,3,12,
10,3,13,
11,3,17,
12,3,20,
4,4,1,
5,4,1,
6,4,1,
7,4,2,
8,4,2,
9,4,3,
10,4,5,
11,4,6,
12,4,9,
5,5,1,
6,5,1,
7,5,1,
8,5,1,
9,5,2,
10,5,2,
11,5,2,
12,5,3,
6,6,1,
7,6,1,
8,6,1,
9,6,1,
10,6,1,
11,6,2,
12,6,2,
7,7,1,
8,7,1,
9,7,1,
10,7,1,
11,7,1,
12,7,1,
8,8,1,
9,8,1,
10,8,1,
11,8,1,
12,8,1,
9,9,1,
10,9,1,
11,9,1,
12,9,1,
10,10,1,
11,10,1,
12,10,1,
11,11,1,
12,11,1,
12,12,1,
-1
};

INT &TDO_upper_bound(INT i, INT j)
{
	INT m, bound;
	
	if (i <= 0) {
		cout << "TDO_upper_bound i <= 0, i = " << i << endl;
		exit(1);
		}
	if (j <= 0) {
		cout << "TDO_upper_bound j <= 0, j = " << j << endl;
		exit(1);
		}
	m = MAXIMUM(i, j);
	if (TDO_upper_bounds_v_max == -1) {
		TDO_refine_init_upper_bounds(12);
		}
	if (m > TDO_upper_bounds_v_max) {
		//cout << "I need TDO_upper_bound " << i << "," << j << endl;
		TDO_refine_extend_upper_bounds(m);
		}
	if (TDO_upper_bound_source(i, j) != 1) {
		//cout << "I need TDO_upper_bound " << i << "," << j << endl;
		}
	bound = TDO_upper_bound_internal(i, j);
	if (bound == -1) {
		cout << "TDO_upper_bound = -1 i=" << i << " j=" << j << endl;
		exit(1);
		}
	//cout << "PACKING " << i << " " << j << " = " << bound << endl;
	return TDO_upper_bound_internal(i, j);
}

INT &TDO_upper_bound_internal(INT i, INT j)
{
	if (i > TDO_upper_bounds_v_max) {
		cout << "TDO_upper_bound i > v_max" << endl;
		cout << "i=" << i << endl;
		cout << "TDO_upper_bounds_v_max=" << TDO_upper_bounds_v_max << endl;
		exit(1);
		}
	if (i <= 0) {
		cout << "TDO_upper_bound_internal i <= 0, i = " << i << endl;
		exit(1);
		}
	if (j <= 0) {
		cout << "TDO_upper_bound_internal j <= 0, j = " << j << endl;
		exit(1);
		}
	return TDO_upper_bounds_table[(i - 1) * TDO_upper_bounds_v_max + j - 1];
}

INT &TDO_upper_bound_source(INT i, INT j)
{
	if (i > TDO_upper_bounds_v_max) {
		cout << "TDO_upper_bound_source i > v_max" << endl;
		cout << "i=" << i << endl;
		cout << "TDO_upper_bounds_v_max=" << TDO_upper_bounds_v_max << endl;
		exit(1);
		}
	if (i <= 0) {
		cout << "TDO_upper_bound_source i <= 0, i = " << i << endl;
		exit(1);
		}
	if (j <= 0) {
		cout << "TDO_upper_bound_source j <= 0, j = " << j << endl;
		exit(1);
		}
	return TDO_upper_bounds_table_source[(i - 1) * TDO_upper_bounds_v_max + j - 1];
}

INT braun_test_single_type(INT v, INT k, INT ak)
{
	INT i, l, s, m;
	
	i = 0;
	s = 0;
	for (l = 1; l <= ak; l++) {
		m = MAXIMUM(k - i, 0);
		s += m;
		if (s > v)
			return FALSE;
		i++;
		}
	return TRUE;
}

INT braun_test_upper_bound(INT v, INT k)
{
	INT n, bound, v2, k2;
	
	//cout << "braun_test_upper_bound v=" << v << " k=" << k << endl;
	if (k == 1) {
		bound = INT_MAX;
		}
	else if (k == 2) {
		bound = ((v * (v - 1)) >> 1);
		}
	else {
		v2 = (v * (v - 1)) >> 1;
		k2 = (k * (k - 1)) >> 1;
		for (n = 1; ; n++) {
			if (braun_test_single_type(v, k, n) == FALSE) {
				bound = n - 1;
				break;
				}
			if (n * k2 > v2) {
				bound = n - 1;
				break;
				}
			}
		}
	//cout << "braun_test_upper_bound v=" << v << " k=" << k << " bound=" << bound << endl;
	return bound;
}

void TDO_refine_init_upper_bounds(INT v_max)
{
	INT i, j, bound, bound_braun, bound_maxfit, u;
	//cout << "TDO_refine_init_upper_bounds v_max=" << v_max << endl;
	
	TDO_upper_bounds_table = NEW_INT(v_max * v_max);
	TDO_upper_bounds_table_source = NEW_INT(v_max * v_max);
	TDO_upper_bounds_v_max = v_max;
	for (i = 0; i < v_max * v_max; i++) {
		TDO_upper_bounds_table[i] = -1;
		TDO_upper_bounds_table_source[i] = 0;
		}
	for (u = 0;; u++) {
		if (TDO_upper_bounds_initial_data[u * 3 + 0] == -1)
			break;
		i = TDO_upper_bounds_initial_data[u * 3 + 0];
		j = TDO_upper_bounds_initial_data[u * 3 + 1];
		bound = TDO_upper_bounds_initial_data[u * 3 + 2];
		bound_braun = braun_test_upper_bound(i, j);
		if (bound < bound_braun) {
			//cout << "i=" << i << " j=" << j << " bound=" << bound << " bound_braun=" << bound_braun << endl;
			}
		TDO_upper_bound_internal(i, j) = bound;
		TDO_upper_bound_source(i, j) = 1;
		}
	for (i = 1; i <= v_max; i++) {
		for (j = 1; j <= i; j++) {
			if (TDO_upper_bound_internal(i, j) != -1)
				continue;
			bound_braun = braun_test_upper_bound(i, j);
			TDO_upper_bound_internal(i, j) = bound_braun;
			TDO_upper_bound_source(i, j) = 2;
			bound_maxfit = packing_number_via_maxfit(i, j);
			if (bound_maxfit < bound_braun) {
				//cout << "i=" << i << " j=" << j << " bound_braun=" << bound_braun 
				//	<< " bound_maxfit=" << bound_maxfit << endl;
				TDO_upper_bound_internal(i, j) = bound_maxfit;
				TDO_upper_bound_source(i, j) = 3;
				}
			}
		}
	//print_integer_matrix_width(cout, TDO_upper_bounds_table, v_max, v_max, v_max, 3);
	//print_integer_matrix_width(cout, TDO_upper_bounds_table_source, v_max, v_max, v_max, 3);
}

void TDO_refine_extend_upper_bounds(INT new_v_max)
{
	INT *new_upper_bounds;
	INT *new_upper_bounds_source;
	INT i, j, bound, bound_braun, bound_maxfit, src;
	INT v_max;

	//cout << "TDO_refine_extend_upper_bounds new_v_max=" << new_v_max << endl;
	v_max = TDO_upper_bounds_v_max;
	new_upper_bounds = NEW_INT(new_v_max * new_v_max);
	new_upper_bounds_source = NEW_INT(new_v_max * new_v_max);
	for (i = 0; i < new_v_max * new_v_max; i++) {
		new_upper_bounds[i] = -1;
		new_upper_bounds_source[i] = 0;
		}
	for (i = 1; i <= v_max; i++) {
		for (j = 1; j <= v_max; j++) {
			bound = TDO_upper_bound_internal(i, j);
			src = TDO_upper_bound_source(i, j);
			new_upper_bounds[(i - 1) * new_v_max + (j - 1)] = bound;
			new_upper_bounds_source[(i - 1) * new_v_max + (j - 1)] = src;
			}
		}
	FREE_INT(TDO_upper_bounds_table);
	FREE_INT(TDO_upper_bounds_table_source);
	TDO_upper_bounds_table = new_upper_bounds;
	TDO_upper_bounds_table_source = new_upper_bounds_source;
	TDO_upper_bounds_v_max = new_v_max;
	for (i = v_max + 1; i <= new_v_max; i++) {
		for (j = 1; j <= i; j++) {
			bound_braun = braun_test_upper_bound(i, j);
			TDO_upper_bound_internal(i, j) = bound_braun;
			TDO_upper_bound_source(i, j) = 2;
			bound_maxfit = packing_number_via_maxfit(i, j);
			if (bound_maxfit < bound_braun) {
				//cout << "i=" << i << " j=" << j << " bound_braun=" << bound_braun 
				//	<< " bound_maxfit=" << bound_maxfit << endl;
				TDO_upper_bound_internal(i, j) = bound_maxfit;
				TDO_upper_bound_source(i, j) = 3;
				}
			}
		}
	//print_integer_matrix_width(cout, TDO_upper_bounds_table, new_v_max, new_v_max, new_v_max, 3);
	//print_integer_matrix_width(cout, TDO_upper_bounds_table_source, new_v_max, new_v_max, new_v_max, 3);
	
}

INT braun_test_on_line_type(INT v, INT *type)
{
	INT i, k, ak, l, s, m;
	
	i = 0;
	s = 0;
	for (k = v; k >= 2; k--) {
		ak = type[k];
		for (l = 0; l < ak; l++) {
			m = MAXIMUM(k - i, 0);
			s += m;
			if (s > v) {
				return FALSE;
				}
			i++;
			}
		}
	return TRUE;
}

INT maxfit_table_v_max = -1;
INT *maxfit_table = NULL;

INT &maxfit(INT i, INT j)
{
	INT m;
	
	m = MAXIMUM(i, j);
	if (maxfit_table_v_max == -1) {
		maxfit_table_init(2 * m);
		}
	if (m > maxfit_table_v_max) {
		maxfit_table_reallocate(2 * m);
		}
	return maxfit_internal(i, j);
}

INT &maxfit_internal(INT i, INT j)
{
	if (i > maxfit_table_v_max) {
		cout << "maxfit_table_v_max i > v_max" << endl;
		cout << "i=" << i << endl;
		cout << "maxfit_table_v_max=" << maxfit_table_v_max << endl;
		exit(1);
		}
	if (j > maxfit_table_v_max) {
		cout << "maxfit_table_v_max j > v_max" << endl;
		cout << "j=" << j << endl;
		cout << "maxfit_table_v_max=" << maxfit_table_v_max << endl;
		exit(1);
		}
	if (i <= 0) {
		cout << "maxfit_table_v_max i <= 0, i = " << i << endl;
		exit(1);
		}
	if (j <= 0) {
		cout << "maxfit_table_v_max j <= 0, j = " << j << endl;
		exit(1);
		}
	return maxfit_table[(i - 1) * maxfit_table_v_max + j - 1];
}

void maxfit_table_init(INT v_max)
{
	//cout << "maxfit_table_init v_max=" << v_max << endl;
	
	maxfit_table = NEW_INT(v_max * v_max);
	maxfit_table_v_max = v_max;
	maxfit_table_compute();
	//print_integer_matrix_width(cout, maxfit_table, v_max, v_max, v_max, 3);
}

void maxfit_table_reallocate(INT v_max)
{
	cout << "maxfit_table_reallocate v_max=" << v_max << endl;
	
	FREE_INT(maxfit_table);
	maxfit_table = NEW_INT(v_max * v_max);
	maxfit_table_v_max = v_max;
	maxfit_table_compute();
	//print_integer_matrix_width(cout, maxfit_table, v_max, v_max, v_max, 3);
}

#define Choose2(x)   ((x*(x-1))/2)

void maxfit_table_compute()
{
	INT M = maxfit_table_v_max;
	INT *matrix = maxfit_table;
	INT m, i, j, inz, gki;
	
	//cout << "computing maxfit table v_max=" << maxfit_table_v_max << endl;
	for (i=0; i<M*M; i++) {
		matrix[i] = 0;
		}
	m = 0;
	for (i=1; i<=M; i++) {
		//cout << "i=" << i << endl;
		inz = i;
		j = 1;
		while (i>=j) {
			gki = inz/i;
			if (j*(j-1)/2 < i*Choose2(gki)+(inz % i)*gki) {
				j++;
				}
			if (j<=M) {
				//cout << "j=" << j << " inz=" << inz << endl;
				m = MAXIMUM(m, inz);
				matrix[(j-1) * M + i-1]=inz;
				matrix[(i-1) * M + j-1]=inz;
				}
			inz++;
			}
		//print_integer_matrix_width(cout, matrix, M, M, M, 3);
		} // next i
}

INT packing_number_via_maxfit(INT n, INT k)
{
	INT m;
	
	if (k == 1) {
		return INT_MAX;
		}
	//cout << "packing_number_via_maxfit n=" << n << " k=" << k << endl;
	m=1;
	while (maxfit(n, m) >= m*k) {
		m++;
		}
	//cout << "P(" << n << "," << k << ")=" << m - 1 << endl;
	return m - 1;
}


