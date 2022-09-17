// andre_construction.cpp
// 
// Anton Betten
// May 30, 2013
//
//
// 
//
//

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {



andre_construction::andre_construction()
{
	order = 0;
	spread_size = 0;
	n = 0;
	k = 0;
	q = 0;
	N = 0;

	Grass = NULL;
	F = NULL;

	spread_elements_numeric = NULL;
	spread_elements_numeric_sorted = NULL;
	spread_elements_perm = NULL;
	spread_elements_perm_inv = NULL;
	spread_elements_genma = NULL;
	pivot = NULL;
	non_pivot = NULL;
}


andre_construction::~andre_construction()
{
	if (Grass) {
		FREE_OBJECT(Grass);
	}
	if (spread_elements_numeric) {
		FREE_lint(spread_elements_numeric);
	}
	if (spread_elements_numeric_sorted) {
		FREE_lint(spread_elements_numeric_sorted);
	}
	if (spread_elements_perm) {
		FREE_lint(spread_elements_perm);
	}
	if (spread_elements_perm_inv) {
		FREE_lint(spread_elements_perm_inv);
	}
	if (spread_elements_genma) {
		FREE_int(spread_elements_genma);
	}
	if (pivot) {
		FREE_int(pivot);
	}
	if (non_pivot) {
		FREE_int(non_pivot);
	}
}

void andre_construction::init(field_theory::finite_field *F,
	int k, long int *spread_elements_numeric,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;
	number_theory::number_theory_domain NT;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "andre_construction::init" << endl;
	}
	andre_construction::F = F;
	q = F->q;
	andre_construction::k = k;
	order = NT.i_power_j(q, k);
	spread_size = order + 1;
	n = 2 * k;
	N = order * order + order + 1;
	if (f_v) {
		cout << "andre_construction::init order=" << order << endl;
		cout << "andre_construction::init N=" << N << endl;
	}
	
	Grass = NEW_OBJECT(grassmann);
	Grass->init(n, k, F, verbose_level);

	if (f_v) {
		cout << "andre_construction::init after Grass->init" << endl;
	}

	andre_construction::spread_elements_numeric = NEW_lint(spread_size);
	spread_elements_numeric_sorted = NEW_lint(spread_size);

	for (i = 0; i < spread_size; i++) {
		andre_construction::spread_elements_numeric[i] =
				spread_elements_numeric[i];
		spread_elements_numeric_sorted[i] =
				spread_elements_numeric[i];
	}

	if (f_v) {
		cout << "andre_construction::init allocating "
				"spread_elements_perm" << endl;
	}

	spread_elements_perm = NEW_lint(spread_size);
	spread_elements_perm_inv = NEW_lint(spread_size);
	for (i = 0; i < spread_size; i++) {
		spread_elements_perm_inv[i] = i;
	}
	Sorting.lint_vec_heapsort_with_log(
			spread_elements_numeric_sorted,
			spread_elements_perm_inv,
			spread_size);
	if (f_v) {
		cout << "andre_construction::init "
				"after int_vec_heapsort_with_log" << endl;
	}
	for (i = 0; i < spread_size; i++) {
		j = spread_elements_perm_inv[i];
		spread_elements_perm[j] = i;
	}
	
	if (f_v) {
		cout << "andre_construction::init creating generator matrices" << endl;
	}

	spread_elements_genma = NEW_int(spread_size * k * n);
	for (i = 0; i < spread_size; i++) {
		if (f_v) {
			cout << "andre_construction::init i=" << i << endl;
		}
		Grass->unrank_lint_here(
			spread_elements_genma + i * k * n,
			spread_elements_numeric[i], 0);
	}
	pivot = NEW_int(spread_size * k);
	non_pivot = NEW_int(spread_size * (n - k));

	if (f_v) {
		cout << "andre_construction::init computing pivot and non_pivot" << endl;
	}


	for (i = 0; i < spread_size; i++) {
		if (f_v) {
			cout << "andre_construction::init i=" << i << endl;
		}
		F->Linear_algebra->Gauss_simple(spread_elements_genma + i * k * n,
				k, n, pivot + i * k, 0 /*verbose_level*/);
		Combi.set_complement(pivot + i * k, k, non_pivot + i * (n - k), a, n);
		if (a != n - k) {
			cout << "andre_construction::init "
					"problem in set_complement" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "andre_construction::init spread element " << i
					<< " is " << spread_elements_numeric[i] << ":" << endl;
			Int_matrix_print(spread_elements_genma + i * k * n, k, n);
			cout << "pivot: ";
			Int_vec_print(cout, pivot + i * k, k);
			cout << endl;
			cout << "non_pivot: ";
			Int_vec_print(cout, non_pivot + i * (n - k), n - k);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "andre_construction::init done" << endl;
	}
}

void andre_construction::points_on_line(
	andre_construction_line_element *Line,
	int *pts_on_line, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

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
			pts_on_line[1 + i] = Line->make_affine_point(
					i, 0 /* verbose_level*/);
		}
	}
	if (f_v) {
		cout << "andre_construction::points_on_line done" << endl;
	}
}

void andre_construction::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "andre_construction::report" << endl;
		cout << "andre_construction::report spread_size=" << spread_size << endl;
	}


	ost << "The spread: \\\\" << endl;

	if (f_v) {
		cout << "andre_construction::report before Grass->print_set_tex" << endl;
	}
	Grass->print_set_tex(ost, spread_elements_numeric, spread_size, verbose_level);
	if (f_v) {
		cout << "andre_construction::report after Grass->print_set_tex" << endl;
	}

	if (f_v) {
		cout << "andre_construction::report done" << endl;
	}
}


}}}



