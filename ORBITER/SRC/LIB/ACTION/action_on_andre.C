// action_on_andre.C
//
// Anton Betten
// June 2, 2013

#include "galois.h"
#include "action.h"

action_on_andre::action_on_andre()
{
	null();
}

action_on_andre::~action_on_andre()
{
	free();
}

void action_on_andre::null()
{
	An = NULL;
	An1 = NULL;
	Andre = NULL;
	coords1 = NULL;
	coords2 = NULL;
	coords3 = NULL;
}

void action_on_andre::free()
{
	if (coords1) {
		FREE_INT(coords1);
		}
	if (coords2) {
		FREE_INT(coords2);
		}
	if (coords3) {
		FREE_INT(coords3);
		}
	null();
}

void action_on_andre::init(action *An, action *An1, andre_construction *Andre, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_andre::init" << endl;
		}
	action_on_andre::An = An;
	action_on_andre::An1 = An1;
	action_on_andre::Andre = Andre;
	action_on_andre::k = Andre->k;
	action_on_andre::n = Andre->n;
	action_on_andre::q = Andre->q;
	k1 = k + 1;
	n1 = n + 1;
	N = Andre->N;
	degree = Andre->N * 2;
	coords1 = NEW_INT(k1 * n1);
	coords2 = NEW_INT(k1 * n1);
	coords3 = NEW_INT(k * n);
	if (f_v) {
		cout << "action_on_andre::init degree=" << degree << endl;
		}
	if (f_v) {
		cout << "action_on_andre::init done" << endl;
		}
}

void action_on_andre::compute_image(INT *Elt, INT i, INT &j, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a;
	
	if (f_v) {
		cout << "action_on_andre::compute_image" << endl;
		}
	if (i < N) {
		a = compute_image_of_point(Elt, i, verbose_level);
		j = a;
		}
	else {
		a = compute_image_of_line(Elt, i - N, verbose_level);
		j = N + a;
		}
	if (f_v) {
		cout << "action_on_andre::compute_image done" << endl;
		}
}

INT action_on_andre::compute_image_of_point(INT *Elt, INT pt_idx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	andre_construction_point_element Pt;
	INT i, image, rk, idx, parallel_class_idx;
	
	if (f_v) {
		cout << "action_on_andre::compute_image_of_point" << endl;
		}
	Pt.init(Andre, 0 /* verbose_level*/);
	Pt.unrank(pt_idx, 0 /* verbose_level*/);
	if (Pt.f_is_at_infinity) {
		if (f_v) {
			cout << "action_on_andre::compute_image_of_point point is at infinity, at_infinity_idx=" << Pt.at_infinity_idx << endl;
			}
		for (i = 0; i < k; i++) {
			INT_vec_copy(Andre->spread_elements_genma + Pt.at_infinity_idx * k * n + i * n, coords1 + i * n1, n);
			coords1[i * n1 + n] = 0;
			}
		if (f_v) {
			cout << "Spread element embedded:" << endl;
			INT_matrix_print(coords1, k, n1);
			}
		for (i = 0; i < k; i++) {
			An1->element_image_of_low_level(coords1 + i * n1, coords2 + i * n1, Elt, verbose_level - 1);
			}
		if (f_v) {
			cout << "Image of spread element:" << endl;
			INT_matrix_print(coords2, k, n1);
			}
		for (i = 0; i < k; i++) {
			INT_vec_copy(coords2 + i * n1, coords3 + i * n, n);
			}
		if (f_v) {
			cout << "Reduced:" << endl;
			INT_matrix_print(coords3, k, n);
			}
		rk = Andre->Grass->rank_INT_here(coords3, 0 /* verbose_level*/);
		if (f_v) {
			cout << "rk=" << rk << endl;
			}
		if (!INT_vec_search(Andre->spread_elements_numeric_sorted, Andre->spread_size, rk, idx)) {
			cout << "andre_construction_line_element::rank annot find the spread element in the sorted list" << endl;
			exit(1);
			}
		if (f_v) {
			cout << "idx=" << idx << endl;
			}
		parallel_class_idx = Andre->spread_elements_perm_inv[idx];
		if (f_v) {
			cout << "parallel_class_idx=" << parallel_class_idx << endl;
			}
		image = parallel_class_idx;
		}
	else {
		INT_vec_copy(Pt.coordinates, coords1, n);
		coords1[n] = 1;

		An1->element_image_of_low_level(coords1, coords2, Elt, verbose_level - 1);

		PG_element_normalize(*Andre->F, coords2, 1, n1);
		INT_vec_copy(coords2, Pt.coordinates, n);
		image = Pt.rank(0 /* verbose_level*/);
		}
	
	if (f_v) {
		cout << "action_on_andre::compute_image_of_point done" << endl;
		}
	return image;
}

INT action_on_andre::compute_image_of_line(INT *Elt, INT line_idx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	andre_construction_line_element Line;
	INT i, j, image;
	
	if (f_v) {
		cout << "action_on_andre::compute_image_of_line" << endl;
		}
	Line.init(Andre, 0 /* verbose_level*/);
	Line.unrank(line_idx, 0 /* verbose_level*/);
	if (Line.f_is_at_infinity) {
		image = 0;
		}
	else {
		for (i = 0; i < k1; i++) {
			for (j = 0; j < n; j++) {
				coords1[i * n1 + j] = Line.coordinates[i * n + j];
				}
			if (i < k) {
				coords1[i * n1 + n] = 0;
				}
			else {
				coords1[i * n1 + n] = 1;
				}
			}

		for (i = 0; i < k1; i++) {
			An1->element_image_of_low_level(coords1 + i * n1, coords2 + i * n1, Elt, verbose_level - 1);
			}

		for (i = 0; i < k; i++) {
			if (coords2[i * n1 + n]) {
				cout << "action_on_andre::compute_image_of_line coords2[i * n1 + n]" << endl;
				exit(1);
				}
			}

		PG_element_normalize(*Andre->F, coords2 + k * n1, 1, n1);

		for (i = 0; i < k1; i++) {
			for (j = 0; j < n; j++) {
				Line.coordinates[i * n + j] = coords2[i * n1 + j];
				}
			}
		image = Line.rank(0 /* verbose_level*/);
		}
	
	if (f_v) {
		cout << "action_on_andre::compute_image_of_line done" << endl;
		}
	return image;
}



