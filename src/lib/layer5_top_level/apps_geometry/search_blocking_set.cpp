// search_blocking_set.cpp
// 
// Anton Betten
// started in INC_CAN:  July 14, 2010
// moved to TOP_LEVEL: Nov 2, 2010
// added active_set: Nov 3, 2010
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_geometry {



search_blocking_set::search_blocking_set()
{
	Record_birth();
	Inc = NULL;
	A = NULL;
	Control = NULL;
	Poset = NULL;
	gen = NULL;


	Line_intersections = NULL;
	blocking_set = NULL;
	blocking_set_len = 0;
	sz = NULL;
	
	active_set = NULL;
	sz_active_set = NULL;
	

	nb_solutions = 0;
	f_find_only_one = false;
	f_blocking_set_size_desired = false;
	blocking_set_size_desired = 0;

	max_search_depth = 0;
	search_nb_candidates = NULL;
	search_cur = NULL;
	search_candidates = NULL;
	save_sz = NULL;
}

search_blocking_set::~search_blocking_set()
{
	Record_death();
	int i;
	
	if (Line_intersections) {
		FREE_OBJECTS(Line_intersections);
		}
	if (Control) {
		FREE_OBJECT(Control);
		}
	if (Poset) {
		FREE_OBJECT(Poset);
		}
	if (gen) {
		FREE_OBJECT(gen);
		}
	if (blocking_set) {
		FREE_lint(blocking_set);
		}
	if (sz) {
		FREE_int(sz);
		}
	if (active_set) {
		FREE_OBJECT(active_set);
		}
	if (sz_active_set) {
		FREE_int(sz_active_set);
		}
	if (search_candidates) {
		for (i = 0; i < max_search_depth; i++) {
			if (search_candidates[i]) {
				FREE_int(search_candidates[i]);
				search_candidates[i] = NULL;
				}
			}
		FREE_pint(search_candidates);
		}
	if (search_nb_candidates) {
		FREE_int(search_nb_candidates);
		}
	if (search_cur) {
		FREE_int(search_cur);
		}
	if (save_sz) {
		for (i = 0; i < max_search_depth; i++) {
			if (save_sz[i]) {
				FREE_int(save_sz[i]);
				save_sz[i] = NULL;
				}
			}
		FREE_pint(save_sz);
		}
}

void search_blocking_set::init(
		geometry::other_geometry::incidence_structure *Inc,
		actions::action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;
	
	if (f_v) {
		cout << "search_blocking_set::init" << endl;
		}
	search_blocking_set::Inc = Inc;
	search_blocking_set::A = A;

	Line_intersections = NEW_OBJECTS(other::data_structures::fancy_set, Inc->nb_cols);
	for (j = 0; j < Inc->nb_cols; j++) {
		Line_intersections[j].init(Inc->nb_rows, 0);
		}

	blocking_set = NEW_lint(Inc->nb_rows);
	sz = NEW_int(Inc->nb_cols);

	active_set = NEW_OBJECT(other::data_structures::fancy_set);
	active_set->init(Inc->nb_rows, 0);
	sz_active_set = NEW_int(Inc->nb_cols + 1);
}

void search_blocking_set::find_partial_blocking_sets(
		int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t0;
	other::orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();


	if (f_v) {
		cout << "search_blocking_set::find_partial_blocking_sets" << endl;
		}
	
	
	
	if (f_v) {
		cout << "find_blocking_sets calling gen->init" << endl;
		}

	if (!A->f_has_strong_generators) {
		cout << "find_partial_blocking_sets !A->f_has_strong_generators" << endl;
		exit(1);
		}
	Control = NEW_OBJECT(poset_classification::poset_classification_control);
	Control->f_depth = true;
	Control->depth = Inc->nb_rows;

	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subset_lattice(A, A,
			A->Strong_gens,
			verbose_level);

	gen = NEW_OBJECT(poset_classification::poset_classification);

	gen->initialize_and_allocate_root_node(Control, Poset,
			Inc->nb_rows, verbose_level);

#if 0
	// ToDo
	gen->init_check_func(
		callback_check_partial_blocking_set, 
		this /* candidate_check_data */);
#endif

	//gen->init_incremental_check_func(
		//check_mindist_incremental, 
		//this /* candidate_check_data */);

	//gen->f_its_OK_to_not_have_an_early_test_func = true;
	
#if 0
	gen->f_print_function = true;
	gen->print_function = print_set;
	gen->print_function_data = this;
#endif	

	int schreier_depth = Inc->nb_rows;
	int f_use_invariant_subset_if_available = true;
	int f_debug = false;
	
	
	if (f_v) {
		cout << "find_partial_blocking_sets: calling generator_main" << endl;
		}
	gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level - 1);
	
	if (f_v) {
		cout << "find_partial_blocking_sets: done with generator_main" << endl;
		}
}

int search_blocking_set::test_level(
		int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_OK;
	int nb_orbits, f, h;
	
	if (f_v) {
		cout << "search_blocking_set::test_level: testing all partial "
				"blocking sets at level " << depth << endl;
		}
	f = gen->first_node_at_level(depth);
	nb_orbits = gen->nb_orbits_at_level(depth);
	if (f_v) {
		cout << "search_blocking_set::test_level: we found " << nb_orbits
				<< " orbits on partial blocking sets "
				"of size " << depth << endl;
		}
	f_OK = false;
	for (h = 0; h < nb_orbits; h++) {
		gen->get_node(f + h)->store_set_to(gen, depth - 1, blocking_set);
		
		if (f_v) {
			cout << "testing set " << h << " / " << nb_orbits << " : ";
			Lint_vec_print(cout, blocking_set, depth);
			cout << endl;
			}
		
		blocking_set_len = depth;

		f_OK = test_blocking_set(depth, blocking_set, verbose_level);

		if (f_OK) {
			if (f_v) {
				cout << "found blocking set" << endl;
				}
			break;
			}
		else {
			if (f_v) {
				cout << endl;
				}
			}
		}
	if (f_OK) {
		return true;
		}
	return false;
}

int search_blocking_set::test_blocking_set(
		int len, long int *S, int verbose_level)
// computes all Line_intersections[] sets based on the set S[len],
// uses Inc->lines_on_point[]
// tests if Line_intersections[j] is greater than zero 
// but less than Inc->nb_points_on_line[j]  for all j
{
	int f_OK = true;
	int f_v = (verbose_level >= 1);
	int i, j, h, a;
	
	if (f_v) {
		cout << "search_blocking_set::test_blocking_set "
				"checking set of points ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		}

	for (j = 0; j < Inc->nb_cols; j++) {
		Line_intersections[j].k = 0;
		}
	for (h = 0; h < len; h++) {
		i = S[h];
		for (a = 0; a < Inc->nb_lines_on_point[i]; a++) {
			j = Inc->lines_on_point[i * Inc->max_r + a];
			Line_intersections[j].add_element(i);
			}
		}
	for (j = 0; j < Inc->nb_cols; j++) {
		sz[j] = Line_intersections[j].k;
		}

	if (f_v) {
		other::data_structures::tally C;

		C.init(sz, Inc->nb_cols, false, 0);

		cout << "the line type is:";
		C.print(false /*f_backwards*/);
		}
	
	for (j = 0; j < Inc->nb_cols; j++) {
		a = Line_intersections[j].k;
		if (a == 0) {
			f_OK = false;
			if (f_v) {
				cout << "not OK, line " << j << " is disjoint" << endl;
				}
			break;
			}
		if (a >= Inc->nb_points_on_line[j]) {
			f_OK = false;
			if (f_v) {
				cout << "not OK, line " << j
						<< " is completely contained" << endl;
				}
			goto done;
			}
		}
	for (h = 0; h < len; h++) {
		i = S[h];
		for (a = 0; a < Inc->nb_lines_on_point[i]; a++) {
			j = Inc->lines_on_point[i * Inc->max_r + a];
			if (Line_intersections[j].k == 1) {
				break;
				}
			}
		if (a == Inc->nb_lines_on_point[i]) {
			f_OK = false;
			if (f_v) {
				cout << "not OK, point S[" << h << "]=" << i
						<< " is not on a 1-line" << endl;
				}
			goto done;
			}
		}
done:
	return f_OK;
}

int search_blocking_set::test_blocking_set_upper_bound_only(
		int len, long int *S, int verbose_level)
{
	int f_OK = true;
	int f_v = (verbose_level >= 1);
	int i, j, h, a;
	
	if (f_v) {
		cout << "search_blocking_set::test_blocking_set_upper_bound_only "
				"set of points ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		}

	for (j = 0; j < Inc->nb_cols; j++) {
		Line_intersections[j].k = 0;
		}
	for (h = 0; h < len; h++) {
		i = S[h];
		//cout << "adding line pencil of point " << i << " of size "
		//<< Inc->nb_lines_on_point[i] << endl;
		for (a = 0; a < Inc->nb_lines_on_point[i]; a++) {
			j = Inc->lines_on_point[i * Inc->max_r + a];
			//cout << "adding point " << i << " to line " << j << endl;
			Line_intersections[j].add_element(i);
			}
		}
	for (j = 0; j < Inc->nb_cols; j++) {
		sz[j] = Line_intersections[j].k;
		}

	if (f_v) {
		other::data_structures::tally C;

		C.init(sz, Inc->nb_cols, false, 0);

		cout << "the line type is:";
		C.print(false /*f_backwards*/);
		}
	
	for (j = 0; j < Inc->nb_cols; j++) {
		a = Line_intersections[j].k;
		if (a >= Inc->nb_points_on_line[j]) {
			f_OK = false;
			if (f_v) {
				cout << "not OK, line " << j
						<< " is completely contained" << endl;
				}
			goto done;
			}
		}
	for (h = 0; h < len; h++) {
		i = S[h];
		for (a = 0; a < Inc->nb_lines_on_point[i]; a++) {
			j = Inc->lines_on_point[i * Inc->max_r + a];
			if (Line_intersections[j].k == 1) {
				break;
				}
			}
		if (a == Inc->nb_lines_on_point[i]) {
			f_OK = false;
			if (f_v) {
				cout << "not OK, point S[" << h << "]=" << i
						<< " is not on a 1-line" << endl;
				}
			goto done;
			}
		}
done:
	return f_OK;
}


void search_blocking_set::search_for_blocking_set(
		int input_no,
		int level, int f_all, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f, nb_orbits, h, u, i, a, b, j;
	
	if (f_v) {
		cout << "search_blocking_set::search_for_blocking_set: "
				"input_no=" << input_no << " testing all partial "
						"blocking sets at level " << level << endl;
		cout << "f_all=" << f_all << endl;
		}

	max_search_depth = Inc->nb_rows - level;
	search_candidates = NEW_pint(max_search_depth);
	search_nb_candidates = NEW_int(max_search_depth);
	search_cur = NEW_int(max_search_depth);
	save_sz = NEW_pint(max_search_depth);
	for (i = 0; i < max_search_depth; i++) {
		search_candidates[i] = NEW_int(Inc->nb_rows);
		save_sz[i] = NEW_int(Inc->nb_cols);
		}


	nb_solutions = 0;
	
	if (f_all) {
		f_find_only_one = false;
		}
	else {
		f_find_only_one = true;
		}

	f = gen->first_node_at_level(level);
	nb_orbits = gen->nb_orbits_at_level(level);
	if (f_v) {
		cout << "search_blocking_set::search_for_blocking_set: "
				"we found " << nb_orbits << " orbits on partial "
				"blocking sets of size" << level << endl;
		}
	for (h = 0; h < nb_orbits; h++) {
		gen->get_node(f + h)->store_set_to(gen, level - 1, blocking_set);
		
		if (f_v) {
			cout << "input_no " << input_no << " level " << level
					<< " testing set " << h << " / " << nb_orbits << " : ";
			Lint_vec_print(cout, blocking_set, level);
			cout << endl;
			}
		
		blocking_set_len = level;

		if (level) {
			b = blocking_set[level - 1];
			}
		else {
			b = -1;
			}
		for (i = b + 1; i < Inc->nb_rows; i++) {
			active_set->add_element(i);
			}
		sz_active_set[0] = active_set->k;
		if (f_v) {
			cout << "sz_active_set[0]=" << sz_active_set[0] << endl;
			}
		
		for (j = 0; j < Inc->nb_cols; j++) {
			Line_intersections[j].k = 0;
			}
		for (u = 0; u < level; u++) {
			i = blocking_set[u];
			//cout << "adding line pencil of point " << i << " of size "
			//<< Inc->nb_lines_on_point[i] << endl;
			for (a = 0; a < Inc->nb_lines_on_point[i]; a++) {
				j = Inc->lines_on_point[i * Inc->max_r + a];
				//cout << "adding point " << i << " to line " << j << endl;
				Line_intersections[j].add_element(i);
				}
			}


		recursive_search_for_blocking_set(input_no,
				level, 0, verbose_level - 4);

		if (f_v) {
			cout << "input_no " << input_no << " level " << level
					<< " testing set " << h << " / " << nb_orbits << " : ";
			cout << " done" << endl;
			}


		if (f_find_only_one && nb_solutions) {
			break;
			}
		}


	if (f_v) {
		cout << "search_blocking_set::search_for_blocking_set done, "
				"we found " << nb_solutions << " solutions" << endl;
		}
}

int search_blocking_set::recursive_search_for_blocking_set(
		int input_no, int starter_level, int level,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;
	int t0_first, t0_len, t, line_idx, i, a, b;

	if (f_v) {
		cout << "search_blocking_set::recursive_search_for_blocking_set "
				"input_no = " << input_no << " level = " << level
				<<  " sz_active_set = " << active_set->k << endl;
		Lint_vec_print(cout, blocking_set, starter_level + level);
		cout << endl;
		}
	if (f_blocking_set_size_desired) {
		if (starter_level + level > blocking_set_size_desired) {
			if (f_v) {
				cout << "we backtrack since we reached the "
						"desired size" << endl;
				}
			return true;
			}
		}

	for (j = 0; j < Inc->nb_cols; j++) {
		sz[j] = Line_intersections[j].k;
		}
	other::data_structures::tally C;

	C.init(sz, Inc->nb_cols, false, 0);

	if (f_v) {
		cout << "the current line type is:";
		C.print(false /*f_backwards*/);
		}
	for (j = 0; j < Inc->nb_cols; j++) {
		if (sz[j] == Inc->nb_points_on_line[j]) {
			// backtrack, since one line is contained in the blocking set
			if (f_v) {
				cout << "we backtrack since line " << j
						<< " is contained in the blocking set" << endl;
				}
			return true;
			}
		}

	t0_first = C.type_first[0];
	t0_len = C.type_len[0];
	t = C.data_sorted[t0_first];
	if (t) {
		cout << "found blocking set of size "
				<< starter_level + level << " : ";
		Lint_vec_print(cout, blocking_set, starter_level + level);
		cout << " line type = ";
		C.print(false /*f_backwards*/);
		cout << " : solution no " << nb_solutions + 1;
		cout << " : ";
		for (i = 0; i < level; i++) {
			cout << i << ":" << search_cur[i] << "/"
					<< search_nb_candidates[i] << " ";
			}
		cout << endl;


		vector<int> sol;

		sol.resize(starter_level + level);
		for (j = 0; j < starter_level + level; j++) {
			sol[j] = (int) blocking_set[j];
			}
		solutions.push_back(sol);

		nb_solutions++;
		
		if (f_find_only_one) {
			return false;
			}
		else {
			return true;
			}
		}
	else {
		if (f_v) {
			cout << "there are " << t0_len << " 0-lines" << endl;
			}
		}
	line_idx = C.sorting_perm_inv[t0_first];
	if (f_v) {
		cout << "line_idx=" << line_idx << endl;
		}
	if (Line_intersections[line_idx].k != 0) {
		cout << "Line_intersections[line_idx].k != 0" << endl;
		exit(1);
		}
	j = 0;
	for (i = 0; i < Inc->nb_points_on_line[line_idx]; i++) {
		a = Inc->points_on_line[line_idx * Inc->max_k + i];
		if (active_set->is_contained(a)) {
			search_candidates[level][j++] = a;
			}
		}
	search_nb_candidates[level] = j;
	
	for (search_cur[level] = 0;
			search_cur[level] < search_nb_candidates[level];
			search_cur[level]++) {


		save_line_intersection_size(level);


		a = search_candidates[level][search_cur[level]];

		blocking_set[starter_level + level] = a;

		for (b = 0; b < Inc->nb_lines_on_point[a]; b++) {
			j = Inc->lines_on_point[a * Inc->max_r + b];
			//cout << "adding point " << i << " to line " << j << endl;
			Line_intersections[j].add_element(a);
			}

		active_set->delete_element(a);
		sz_active_set[level + 1] = active_set->k;
		
		if (!recursive_search_for_blocking_set(input_no,
				starter_level, level + 1, verbose_level)) {
			return false;
			}
		else {
			active_set->k = sz_active_set[level + 1];
			}

		
		restore_line_intersection_size(level);
		
		}
	return true;
}

void search_blocking_set::save_line_intersection_size(
		int level)
{
	int j;

	for (j = 0; j < Inc->nb_cols; j++) {
		save_sz[level][j] = Line_intersections[j].k;
		}
}

void search_blocking_set::restore_line_intersection_size(
		int level)
{
	int j;

	for (j = 0; j < Inc->nb_cols; j++) {
		Line_intersections[j].k = save_sz[level][j];
		}
}


#if 0
int callback_check_partial_blocking_set(
		int len, int *S,
		void *data, int verbose_level)
{
	search_blocking_set *SBS = (search_blocking_set *) data;
	int f_OK = true;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "check_partial_blocking_set: checking set of points ";
		print_set(cout, len, S);
		cout << endl;
		}

	if (len && S[len - 1] >= SBS->Inc->nb_rows) {
		return false;
		}

	//cout << "before SBS->test_blocking_set_upper_bound_only" << endl;
	f_OK = SBS->test_blocking_set_upper_bound_only(len, S, verbose_level);



	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
			}
		return true;
		}
	else {
		return false;
		}
}
#endif

}}}

