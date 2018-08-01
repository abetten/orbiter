// andre_construction.C
// 
// Anton Betten
// May 30, 2013
//
//
// 
//
//

#include "galois.h"




andre_construction::andre_construction()
{
	null();
}

andre_construction::~andre_construction()
{
	freeself();
}

void andre_construction::null()
{
	Grass = NULL;
	spread_elements_numeric = NULL;
	spread_elements_numeric_sorted = NULL;
	spread_elements_perm = NULL;
	spread_elements_perm_inv = NULL;
	spread_elements_genma = NULL;
	pivot = NULL;
	non_pivot = NULL;
}

void andre_construction::freeself()
{
	if (Grass) {
		delete Grass;
		}
	if (spread_elements_numeric) {
		FREE_INT(spread_elements_numeric);
		}
	if (spread_elements_numeric_sorted) {
		FREE_INT(spread_elements_numeric_sorted);
		}
	if (spread_elements_perm) {
		FREE_INT(spread_elements_perm);
		}
	if (spread_elements_perm_inv) {
		FREE_INT(spread_elements_perm_inv);
		}
	if (spread_elements_genma) {
		FREE_INT(spread_elements_genma);
		}
	if (pivot) {
		FREE_INT(pivot);
		}
	if (non_pivot) {
		FREE_INT(non_pivot);
		}
	null();
}

void andre_construction::init(finite_field *F, INT k, INT *spread_elements_numeric, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a;

	if (f_v) {
		cout << "andre_construction::init" << endl;
		}
	andre_construction::F = F;
	q = F->q;
	andre_construction::k = k;
	order = i_power_j(q, k);
	spread_size = order + 1;
	n = 2 * k;
	N = order * order + order + 1;
	if (f_v) {
		cout << "andre_construction::init order=" << order << endl;
		cout << "andre_construction::init N=" << N << endl;
		}
	
	Grass = new grassmann;
	Grass->init(n, k, F, verbose_level);

	if (f_v) {
		cout << "andre_construction::init after Grass->init" << endl;
		}

	andre_construction::spread_elements_numeric = NEW_INT(spread_size);
	spread_elements_numeric_sorted = NEW_INT(spread_size);
	for (i = 0; i < spread_size; i++) {
		andre_construction::spread_elements_numeric[i] = spread_elements_numeric[i];
		spread_elements_numeric_sorted[i] = spread_elements_numeric[i];
		}

	if (f_v) {
		cout << "andre_construction::init allocating spread_elements_perm" << endl;
		}

	spread_elements_perm = NEW_INT(spread_size);
	spread_elements_perm_inv = NEW_INT(spread_size);
	for (i = 0; i < spread_size; i++) {
		spread_elements_perm_inv[i] = i;
		}
	INT_vec_heapsort_with_log(spread_elements_numeric_sorted, spread_elements_perm_inv, spread_size);
	if (f_v) {
		cout << "andre_construction::init INT_vec_heapsort_with_log" << endl;
		}
	for (i = 0; i < spread_size; i++) {
		j = spread_elements_perm_inv[i];
		spread_elements_perm[j] = i;
		}
	
	if (f_v) {
		cout << "andre_construction::init before genma" << endl;
		}

	spread_elements_genma = NEW_INT(spread_size * k * n);
	for (i = 0; i < spread_size; i++) {
		Grass->unrank_INT_here(spread_elements_genma + i * k * n, spread_elements_numeric[i], 0);
		}
	pivot = NEW_INT(spread_size * k);
	non_pivot = NEW_INT(spread_size * (n - k));
	for (i = 0; i < spread_size; i++) {
		F->Gauss_simple(spread_elements_genma + i * k * n, k, n, pivot + i * k, 0 /*verbose_level*/);
		set_complement(pivot + i * k, k, non_pivot + i * (n - k), a, n);
		if (a != n - k) {
			cout << "andre_construction::init problem in set_complement" << endl;
			}
		cout << "andre_construction::init spread element " << i << " is " << spread_elements_numeric[i] << ":" << endl;
		INT_matrix_print(spread_elements_genma + i * k * n, k, n);
		cout << "pivot: ";
		INT_vec_print(cout, pivot + i * k, k);
		cout << endl;
		cout << "non_pivot: ";
		INT_vec_print(cout, non_pivot + i * (n - k), n - k);
		cout << endl;
		}
	if (f_v) {
		cout << "andre_construction::init done" << endl;
		}
}

void andre_construction::points_on_line(andre_construction_line_element *Line, 
	INT *pts_on_line, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "andre_construction::points_on_line" << endl;
		}
	if (Line->f_is_at_infinity) {
		for (i = 0; i < spread_size; i++) {
			pts_on_line[i] = i;
			}
		}
	else {
		pts_on_line[0] = Line->parallel_class_idx;
		for (i = 0; i < order; i++) {
			pts_on_line[1 + i] = Line->make_affine_point(i, 0 /* verbose_level*/);
			}
		}
	if (f_v) {
		cout << "andre_construction::points_on_line done" << endl;
		}
}



