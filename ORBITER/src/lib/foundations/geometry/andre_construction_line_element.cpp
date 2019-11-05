// andre_construction_line_element.cpp
// 
// Anton Betten
// May 31, 2013
//
//
// 
//
//

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {



andre_construction_line_element::andre_construction_line_element()
{
	null();
}

andre_construction_line_element::~andre_construction_line_element()
{
	freeself();
}

void andre_construction_line_element::null()
{
	pivots = NULL;
	non_pivots = NULL;
	coset = NULL;
	coordinates = NULL;
}

void andre_construction_line_element::freeself()
{
	if (pivots) {
		FREE_int(pivots);
		}
	if (non_pivots) {
		FREE_int(non_pivots);
		}
	if (coset) {
		FREE_int(coset);
		}
	if (coordinates) {
		FREE_int(coordinates);
		}
	null();
}

void andre_construction_line_element::init(
		andre_construction *Andre, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "andre_construction_line_element::init" << endl;
		}
	andre_construction_line_element::Andre = Andre;
	andre_construction_line_element::k = Andre->k;
	andre_construction_line_element::n = Andre->n;
	andre_construction_line_element::q = Andre->q;
	andre_construction_line_element::spread_size = Andre->spread_size;
	andre_construction_line_element::F = Andre->F;
	pivots = NEW_int(k);
	non_pivots = NEW_int(n - k);
	coset = NEW_int(n - k);
	coordinates = NEW_int((k + 1) * n);
	if (f_v) {
		cout << "andre_construction_line_element::init done" << endl;
		}
}

void andre_construction_line_element::unrank(
		int line_rank, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;
	geometry_global Gg;

	if (f_v) {
		cout << "andre_construction_line_element::unrank "
				"line_rank=" << line_rank << endl;
		}
	andre_construction_line_element::line_rank = line_rank;
	if (line_rank < 1) {
		f_is_at_infinity = TRUE;
		}
	else {
		f_is_at_infinity = FALSE;
		line_rank -= 1;
		coset_idx = line_rank % Andre->order;
		parallel_class_idx = line_rank / Andre->order;
		Gg.AG_element_unrank(q, coset, 1, n - k, coset_idx);
		int_vec_copy(
			Andre->spread_elements_genma + parallel_class_idx * k * n,
			coordinates, k * n);
		for (i = 0; i < n - k; i++) {
			non_pivots[i] = Andre->non_pivot[
					parallel_class_idx * (n - k) + i];
			}
		for (i = 0; i < k; i++) {
			pivots[i] = Andre->pivot[parallel_class_idx * k + i];
			}
		for (i = 0; i < n; i++) {
			coordinates[k * n + i] = 0;
			}
		for (i = 0; i < n - k; i++) {
			a = coset[i];
			j = non_pivots[i];
			coordinates[k * n + j] = a;
			}
		}
	if (f_v) {
		cout << "andre_construction_line_element::unrank done" << endl;
		}
}

int andre_construction_line_element::rank(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, rk, idx;
	geometry_global Gg;
	combinatorics_domain Combi;
	sorting Sorting;

	if (f_v) {
		cout << "andre_construction_line_element::rank" << endl;
		}
	line_rank = 0;
	if (f_is_at_infinity) {
		line_rank = 0;
		}
	else {
		line_rank = 1;

		F->Gauss_simple(coordinates,
				k, n, pivots,
				0 /* verbose_level */);
		Combi.set_complement(pivots, k, non_pivots, a, n);

		for (i = 0; i < k; i++) {
			F->Gauss_step(coordinates + i * n,
					coordinates + k * n, n, pivots[i],
					0 /* verbose_level */);
				// afterwards: v2[idx] = 0 and v1,v2
				// span the same space as before
				// v1 is not changed if v1[idx] is nonzero
			}
		for (i = 0; i < n - k; i++) {
			j = non_pivots[i];
			a = coordinates[k * n + j];
			coset[i] = a;
			}
		coset_idx = Gg.AG_element_rank(q, coset, 1, n - k);

		rk = Andre->Grass->rank_lint_here(
				coordinates, 0 /* verbose_level*/);
		if (!Sorting.int_vec_search(Andre->spread_elements_numeric_sorted,
				spread_size, rk, idx)) {
			cout << "andre_construction_line_element::rank cannot "
					"find the spread element in the sorted list" << endl;
			exit(1);
			}
		parallel_class_idx = Andre->spread_elements_perm_inv[idx];
		line_rank += parallel_class_idx * Andre->order + coset_idx;
		}
	if (f_v) {
		cout << "andre_construction_line_element::unrank done" << endl;
		}
	return line_rank;
}

int andre_construction_line_element::make_affine_point(
		int idx, int verbose_level)
// 0 \le idx \le order
{
	int f_v = (verbose_level >= 1);
	int *vec1;
	int *vec2;
	int point_rank, a;
	geometry_global Gg;

	if (f_v) {
		cout << "andre_construction_line_element::make_"
				"affine_point" << endl;
		}
	vec1 = NEW_int(k + 1);
	vec2 = NEW_int(n);
	Gg.AG_element_unrank(q, vec1, 1, k, idx);
	vec1[k] = 1;

	F->mult_vector_from_the_left(vec1, coordinates, vec2, k + 1, n);

	point_rank = spread_size;
	a = Gg.AG_element_rank(q, vec2, 1, n);
	point_rank += a;

	FREE_int(vec1);
	FREE_int(vec2);
	if (f_v) {
		cout << "andre_construction_line_element::make_"
				"affine_point done" << endl;
		}
	return point_rank;
}

}
}

