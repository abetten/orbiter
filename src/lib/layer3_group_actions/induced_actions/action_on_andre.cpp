// action_on_andre.cpp
//
// Anton Betten
// June 2, 2013

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_andre::action_on_andre()
{
	An = NULL;
	An1 = NULL;
	Andre = NULL;
	k = n = q = 0;
	k1 = n1 = 0;
	N = 0;
	degree = 0;
	coords1 = NULL;
	coords2 = NULL;
	coords3 = NULL;
}


action_on_andre::~action_on_andre()
{
	if (coords1) {
		FREE_int(coords1);
		}
	if (coords2) {
		FREE_int(coords2);
		}
	if (coords3) {
		FREE_int(coords3);
		}
}

void action_on_andre::init(
		actions::action *An,
		actions::action *An1,
		geometry::finite_geometries::andre_construction *Andre,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
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
	coords1 = NEW_int(k1 * n1);
	coords2 = NEW_int(k1 * n1);
	coords3 = NEW_int(k * n);
	if (f_v) {
		cout << "action_on_andre::init degree=" << degree << endl;
		}
	if (f_v) {
		cout << "action_on_andre::init done" << endl;
		}
}

long int action_on_andre::compute_image(
		int *Elt, long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int a, j;
	
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
	return j;
}

long int action_on_andre::compute_image_of_point(
		int *Elt,
		long int pt_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry::finite_geometries::andre_construction_point_element Pt;
	long int i, image, rk, parallel_class_idx;
	int idx;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "action_on_andre::compute_image_of_point" << endl;
		}
	Pt.init(Andre, 0 /* verbose_level*/);
	Pt.unrank(pt_idx, 0 /* verbose_level*/);
	if (Pt.f_is_at_infinity) {
		if (f_v) {
			cout << "action_on_andre::compute_image_of_point "
					"point is at infinity, at_infinity_idx="
					<< Pt.at_infinity_idx << endl;
			}
		for (i = 0; i < k; i++) {
			Int_vec_copy(Andre->spread_elements_genma +
					Pt.at_infinity_idx * k * n + i * n, coords1 + i * n1, n);
			coords1[i * n1 + n] = 0;
			}
		if (f_v) {
			cout << "Spread element embedded:" << endl;
			Int_matrix_print(coords1, k, n1);
			}
		for (i = 0; i < k; i++) {
			An1->Group_element->element_image_of_low_level(coords1 + i * n1,
					coords2 + i * n1, Elt, verbose_level - 1);
			}
		if (f_v) {
			cout << "Image of spread element:" << endl;
			Int_matrix_print(coords2, k, n1);
			}
		for (i = 0; i < k; i++) {
			Int_vec_copy(coords2 + i * n1, coords3 + i * n, n);
			}
		if (f_v) {
			cout << "Reduced:" << endl;
			Int_matrix_print(coords3, k, n);
			}
		rk = Andre->Grass->rank_lint_here(coords3, 0 /* verbose_level*/);
		if (f_v) {
			cout << "rk=" << rk << endl;
			}
		if (!Sorting.lint_vec_search(
				Andre->spread_elements_numeric_sorted,
				Andre->spread_size, rk, idx, 0)) {
			cout << "andre_construction_line_element::rank "
					"cannot find the spread element in the sorted list" << endl;
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
		Int_vec_copy(Pt.coordinates, coords1, n);
		coords1[n] = 1;

		An1->Group_element->element_image_of_low_level(coords1, coords2,
				Elt, verbose_level - 1);

		Andre->F->Projective_space_basic->PG_element_normalize(
				coords2, 1, n1);
		Int_vec_copy(coords2, Pt.coordinates, n);
		image = Pt.rank(0 /* verbose_level*/);
		}
	
	if (f_v) {
		cout << "action_on_andre::compute_image_of_point done" << endl;
		}
	return image;
}

long int action_on_andre::compute_image_of_line(
		int *Elt,
		long int line_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry::finite_geometries::andre_construction_line_element Line;
	int i, j, image;
	
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
			An1->Group_element->element_image_of_low_level(coords1 + i * n1,
					coords2 + i * n1, Elt, verbose_level - 1);
			}

		for (i = 0; i < k; i++) {
			if (coords2[i * n1 + n]) {
				cout << "action_on_andre::compute_image_of_line "
						"coords2[i * n1 + n]" << endl;
				exit(1);
				}
			}

		Andre->F->Projective_space_basic->PG_element_normalize(
				coords2 + k * n1, 1, n1);

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

}}}


