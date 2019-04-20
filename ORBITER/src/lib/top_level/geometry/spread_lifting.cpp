// spread_lifting.C
// 
// Anton Betten
// April 1, 2018
//
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


spread_lifting::spread_lifting()
{
	null();
}

spread_lifting::~spread_lifting()
{
	freeself();
}

void spread_lifting::null()
{
	S = NULL;
	E = NULL;
	starter = NULL;
	points_covered_by_starter = NULL;
	free_point_list = NULL;
	point_idx = NULL;
	col_labels = NULL;

}

void spread_lifting::freeself()
{
	if (points_covered_by_starter) {
		FREE_int(points_covered_by_starter);
		}
	if (free_point_list) {
		FREE_int(free_point_list);
		}
	if (point_idx) {
		FREE_int(point_idx);
		}
	if (col_labels) {
		FREE_int(col_labels);
		}
	null();
}

void spread_lifting::init(
	spread *S, exact_cover *E,
	int *starter, int starter_size, 
	int starter_case_number, int starter_number_of_cases, 
	int *candidates, int nb_candidates,
	strong_generators *Strong_gens,
	int f_lex, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	longinteger_object go;
	
	
	if (f_v) {
		cout << "spread_lifting::init" << endl;
		}
	spread_lifting::S = S;
	spread_lifting::E = E;
	spread_lifting::starter = starter;
	spread_lifting::starter_size = starter_size;
	spread_lifting::starter_case_number = starter_case_number;
	spread_lifting::starter_number_of_cases = starter_number_of_cases;
	spread_lifting::candidates = candidates;
	spread_lifting::nb_candidates = nb_candidates;
	spread_lifting::Strong_gens = Strong_gens;
	spread_lifting::f_lex = f_lex;
	
	if (f_v) {
		cout << "spread_lifting::init "
				"before compute_points_covered_by_starter" << endl;
		}
	compute_points_covered_by_starter(verbose_level - 2);
	if (f_v) {
		cout << "spread_lifting::init "
				"after compute_points_covered_by_starter" << endl;
		}
	
	if (f_v) {
		cout << "spread_lifting::init "
				"before prepare_free_points" << endl;
		}
	prepare_free_points(verbose_level - 2);
	if (f_v) {
		cout << "spread_lifting::init "
				"after prepare_free_points" << endl;
		}

	nb_needed = S->spread_size - starter_size;
	if (f_v) {
		cout << "spread_lifting::init "
				"nb_needed=" << nb_needed << endl;
		cout << "spread_lifting::init "
				"nb_candidates=" << nb_candidates << endl;
		}


	col_labels = NEW_int(nb_candidates);
	int_vec_copy(candidates, col_labels, nb_candidates);
	nb_cols = nb_candidates;


	if (f_lex) {
		int nb_cols_before;

		nb_cols_before = nb_cols;
		E->lexorder_test(col_labels, nb_cols, Strong_gens->gens, 
			verbose_level - 2);
		if (f_v) {
			cout << "spread_lifting::init after lexorder test "
					"nb_candidates before: " << nb_cols_before
					<< " reduced to  " << nb_cols << " (deleted "
					<< nb_cols_before - nb_cols << ")" << endl;
			}
		}

	if (f_v) {
		cout << "spread_lifting::init after lexorder test" << endl;
		cout << "spread_lifting::init nb_cols=" << nb_cols << endl;
		}

	if (f_v) {
		cout << "spread_lifting::init done" << endl;
		}
}

void spread_lifting::compute_points_covered_by_starter(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int i, a;
	sorting Sorting;

	if (f_v) {
		cout << "spread_lifting::compute_points_"
				"covered_by_starter" << endl;
		}
	nb_points_covered_by_starter = starter_size * S->block_size;
	points_covered_by_starter = NEW_int(nb_points_covered_by_starter);

	for (i = 0; i < starter_size; i++) {
		int *point_list;
		int nb_points;

		a = starter[i];
		S->Grass->unrank_int(a, 0/*verbose_level - 4*/);
		S->F->all_PG_elements_in_subspace(
			S->Grass->M, S->k, S->n, point_list,
			nb_points, 0 /*verbose_level - 2*/);
			// in projective.C
		
		if (nb_points != S->block_size) {
			cout << "spread_lifting::compute_points_"
					"covered_by_starter nb_points != S->block_size" << endl;
			exit(1);
			}

		int_vec_copy(point_list,
				points_covered_by_starter + i * S->block_size,
				S->block_size);

		if (f_v3) {
			cout << "starter element " << i << " / "
					<< starter_size << " is " << a << ":" << endl;
			int_matrix_print(S->Grass->M, S->k, S->n);
			//cout << endl;
			cout << "points_covered_by_starter: " << endl;
			int_vec_print(cout,
					points_covered_by_starter +
						i * S->block_size, S->block_size);
			cout << endl;
			}

		FREE_int(point_list);
		}
	Sorting.int_vec_heapsort(points_covered_by_starter,
			nb_points_covered_by_starter);
	if (f_vv) {
		cout << "spread_lifting::compute_points_"
				"covered_by_starter covered points computed:" << endl;
		cout << "spread_lifting::compute_points_"
				"covered_by_starter nb_points_covered_by_starter="
				<< nb_points_covered_by_starter << endl;
		int_vec_print(cout, points_covered_by_starter,
				nb_points_covered_by_starter);
		cout << endl;
		}

	if (f_v) {
		cout << "spread_lifting::compute_points_"
				"covered_by_starter done" << endl;
		}
}

void spread_lifting::prepare_free_points(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int i, j, idx;
	sorting Sorting;
	
	if (f_v) {
		cout << "spread_lifting::prepare_free_points" << endl;
		}

	nb_free_points = S->nb_points_total - nb_points_covered_by_starter;
	if (f_vv) {
		cout << "spread_lifting::prepare_free_points "
				"nb_free_points=" << nb_free_points << endl;
		}
	free_point_list = NEW_int(nb_free_points);
	point_idx = NEW_int(S->nb_points_total);
	j = 0;
	for (i = 0; i < S->nb_points_total; i++) {
		if (Sorting.int_vec_search(points_covered_by_starter,
				nb_points_covered_by_starter, i, idx)) {
			point_idx[i] = -1;
			}
		else {
			free_point_list[j] = i;
			point_idx[i] = j;
			j++;
			}
		}
	if (j != nb_free_points) {
		cout << "spread_lifting::prepare_free_points "
				"j != nb_free_points" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "spread_lifting::prepare_free_points "
				"computed free points" << endl;
		}

	if (f_v3) {
		cout << "spread_lifting::prepare_free_points "
				"The " << nb_free_points << " free points:" << endl;
		int_vec_print(cout,
				free_point_list, nb_free_points);
		cout << endl;
		S->print_points(free_point_list, nb_free_points);
		}
	if (f_v) {
		cout << "spread_lifting::prepare_free_points done" << endl;
		}
}

diophant *spread_lifting::create_system(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v5 = (verbose_level >= 5);
	diophant *Dio;
	int nb_rows = nb_free_points;
	int i, j, a, b, h;

	if (f_v) {
		cout << "spread_lifting::create_system" << endl;
		}

	Dio = NEW_OBJECT(diophant);
	Dio->open(nb_rows, nb_cols);
	Dio->f_has_sum = TRUE;
	Dio->sum = nb_needed;

	for (i = 0; i < nb_rows; i++) {
		Dio->type[i] = t_EQ;
		Dio->RHS[i] = 1;
		}

	Dio->fill_coefficient_matrix_with(0);
	if (f_vv) {
		cout << "spread_lifting::create_system "
				"nb_rows = " << nb_rows << endl;
		cout << "spread_lifting::create_system "
				"nb_cols = " << nb_cols << endl;
		}
	for (j = 0; j < nb_cols; j++) {

		int *point_list;
		int nb_points;

		a = col_labels[j];
		S->Grass->unrank_int(a, 0/*verbose_level - 4*/);
		if (f_vv) {
			cout << "candidate " << j << " / "
					<< nb_cols << " is " << a << endl;
			}

		if (f_v5) {
			cout << "Which is " << endl;
			int_matrix_print(S->Grass->M, S->k, S->n);
			}
		S->F->all_PG_elements_in_subspace(
				S->Grass->M, S->k, S->n, point_list, nb_points,
				0 /*verbose_level*/);
		if (nb_points != S->block_size) {
			cout << "spread_lifting::create_system "
					"nb_points != S->block_size" << endl;
			exit(1);
			}
		if (FALSE /*f_vv*/) {
			cout << "List of points: ";
			int_vec_print(cout, point_list, nb_points);
			cout << endl;
			}




		for (h = 0; h < S->block_size; h++) {
			b = point_list[h];
			i = point_idx[b];
			if (i == -1) {
				cout << "spread_lifting::create_system "
						"candidate block contains point that "
						"is already covered" << endl;
				exit(1);
				}
			if (i < 0 || i >= nb_free_points) {
				cout << "spread_lifting::create_system "
						"i < 0 || i >= nb_free_points" << endl;
				exit(1);
				}
			Dio->Aij(i, j) = 1;
			}
		FREE_int(point_list);
		}

	if (FALSE) {
		cout << "spread_lifting::create_system "
				"coefficient matrix" << endl;
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				cout << Dio->Aij(i, j);
				}
			cout << endl;
			}
		}
	if (f_v) {
		cout << "spread_lifting::create_system done" << endl;
		}
	return Dio;
}

void spread_lifting::find_coloring(diophant *Dio, 
	int *&col_color, int &nb_colors, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *colors;
	int i, j, a, c;
	int *v;
	
	if (f_v) {
		cout << "spread_lifting::find_coloring" << endl;
		}
	v = NEW_int(S->n);
	colors = NEW_int(nb_free_points);
	nb_colors = 0;
	for (i = 0; i < nb_free_points; i++) {
		a = free_point_list[i];
		S->unrank_point(v, a);
		S->F->PG_element_normalize_from_front(
			v, 1, S->n);
		if (v[0] != 1) {
			continue;
			}
		for (j = 1; j < S->k; j++) {
			if (v[j] != 0) {
				break;
				}
			}
		if (j < S->k) {
			continue;
			}
		if (f_v) {
			cout << "found color " << nb_colors
					<< " : " << i << " = " << a << " = ";
			int_vec_print(cout, v, S->n);
			cout << endl;
			}
		colors[nb_colors++] = i;
		}
	if (f_v) {
		cout << "spread_lifting::find_coloring "
				"we found " << nb_colors << " colors" << endl;
		}
	if (nb_colors != nb_needed) {
		cout << "spread_lifting::find_coloring "
				"nb_colors != nb_needed" << endl;
		exit(1);
		}
	
	col_color = NEW_int(nb_cols);
	for (j = 0; j < nb_cols; j++) {
		for (c = 0; c < nb_colors; c++) {
			i = colors[c];
			if (Dio->Aij(i, j)) {
				col_color[j] = c;
				break;
				}
			}
		if (c == nb_colors) {
			cout << "spread_lifting::find_coloring "
					"c == nb_colors" << endl;
			exit(1);
			}
		}
	FREE_int(colors);
	FREE_int(v);
	if (f_v) {
		cout << "spread_lifting::find_coloring "
				"done" << endl;
		}
}

}}

