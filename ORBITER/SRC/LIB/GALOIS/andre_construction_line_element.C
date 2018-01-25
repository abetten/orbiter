// andre_construction_line_element.C
// 
// Anton Betten
// May 31, 2013
//
//
// 
//
//

#include "galois.h"




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
		FREE_INT(pivots);
		}
	if (non_pivots) {
		FREE_INT(non_pivots);
		}
	if (coset) {
		FREE_INT(coset);
		}
	if (coordinates) {
		FREE_INT(coordinates);
		}
	null();
}

void andre_construction_line_element::init(andre_construction *Andre, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "andre_construction_line_element::init" << endl;
		}
	andre_construction_line_element::Andre = Andre;
	andre_construction_line_element::k = Andre->k;
	andre_construction_line_element::n = Andre->n;
	andre_construction_line_element::q = Andre->q;
	andre_construction_line_element::spread_size = Andre->spread_size;
	andre_construction_line_element::F = Andre->F;
	pivots = NEW_INT(k);
	non_pivots = NEW_INT(n - k);
	coset = NEW_INT(n - k);
	coordinates = NEW_INT((k + 1) * n);
	if (f_v) {
		cout << "andre_construction_line_element::init done" << endl;
		}
}

void andre_construction_line_element::unrank(INT line_rank, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a;

	if (f_v) {
		cout << "andre_construction_line_element::unrank line_rank=" << line_rank << endl;
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
		AG_element_unrank(q, coset, 1, n - k, coset_idx);
		INT_vec_copy(Andre->spread_elements_genma + parallel_class_idx * k * n, coordinates, k * n);
		for (i = 0; i < n - k; i++) {
			non_pivots[i] = Andre->non_pivot[parallel_class_idx * (n - k) + i];
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

INT andre_construction_line_element::rank(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, rk, idx;

	if (f_v) {
		cout << "andre_construction_line_element::rank" << endl;
		}
	line_rank = 0;
	if (f_is_at_infinity) {
		line_rank = 0;
		}
	else {
		line_rank = 1;

		F->Gauss_simple(coordinates, k, n, pivots, 0 /* verbose_level */);
		set_complement(pivots, k, non_pivots, a, n);

		for (i = 0; i < k; i++) {
			F->Gauss_step(coordinates + i * n, coordinates + k * n, n, pivots[i], 0 /* verbose_level */);
				// afterwards: v2[idx] = 0 and v1,v2 span the same space as before
				// v1 is not changed if v1[idx] is nonzero
			}
		for (i = 0; i < n - k; i++) {
			j = non_pivots[i];
			a = coordinates[k * n + j];
			coset[i] = a;
			}
		AG_element_rank(q, coset, 1, n - k, coset_idx);

		rk = Andre->Grass->rank_INT_here(coordinates, 0 /* verbose_level*/);
		if (!INT_vec_search(Andre->spread_elements_numeric_sorted, spread_size, rk, idx)) {
			cout << "andre_construction_line_element::rank annot find the spread element in the sorted list" << endl;
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

INT andre_construction_line_element::make_affine_point(INT idx, INT verbose_level)
// 0 \le idx \le order
{
	INT f_v = (verbose_level >= 1);
	INT *vec1;
	INT *vec2;
	INT point_rank, a;

	if (f_v) {
		cout << "andre_construction_line_element::make_affine_point" << endl;
		}
	vec1 = NEW_INT(k + 1);
	vec2 = NEW_INT(n);
	AG_element_unrank(q, vec1, 1, k, idx);
	vec1[k] = 1;

	F->mult_vector_from_the_left(vec1, coordinates, vec2, k + 1, n);

	point_rank = spread_size;
	AG_element_rank(q, vec2, 1, n, a);
	point_rank += a;

	FREE_INT(vec1);
	FREE_INT(vec2);
	if (f_v) {
		cout << "andre_construction_line_element::make_affine_point done" << endl;
		}
	return point_rank;
}


