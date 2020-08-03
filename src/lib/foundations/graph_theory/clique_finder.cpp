// clique_finder.cpp
//
// Anton Betten
// 11/6/07
//
// moved into TOP_LEVEL: Oct 13, 2011
//
// This is the clique finder.
// The main function is clique_finder::backtrack_search()
//
// this file was originally developed for the search of BLT-sets

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {



#undef SUSPICOUS


clique_finder::clique_finder()
{
	f_maxdepth = FALSE;
	f_write_tree = FALSE;
	fp_tree = NULL;

	f_has_bitmatrix = FALSE;

	f_has_row_by_row_adjacency_matrix = FALSE;
	row_by_row_adjacency_matrix = NULL;

	point_labels = NULL;
	point_is_suspicous = NULL;

	bitmatrix_adjacency = NULL;
	pt_list = NULL;
	pt_list_inv = NULL;
	nb_points = NULL;
	candidates = NULL;
	nb_candidates = NULL;
	current_choice = NULL;
	level_counter = NULL;
	f_level_mod = NULL;
	level_r = NULL;
	level_m = NULL;
	current_clique = NULL;

	call_back_clique_found = NULL;
	call_back_add_point = NULL;
	call_back_delete_point = NULL;
	call_back_find_candidates = NULL;
	call_back_is_adjacent = NULL;
	call_back_after_reduction = NULL;

	f_has_print_current_choice_function = FALSE;
	call_back_print_current_choice = NULL;
	print_current_choice_data = NULL;

	call_back_clique_found_data1 = NULL;
	call_back_clique_found_data2 = NULL;
	null();
}

clique_finder::~clique_finder()
{
	free();
}

void clique_finder::null()
{

}

void clique_finder::free()
{
	if (point_labels) {
		FREE_int(point_labels);
	}
	if (point_is_suspicous) {
		FREE_int(point_is_suspicous);
	}

	if (bitmatrix_adjacency) {
		//delete [] adjacency;
		FREE_uchar(bitmatrix_adjacency);
	}
	if (pt_list) {
		FREE_int(pt_list);
	}
	if (pt_list_inv) {
		FREE_int(pt_list_inv);
	}
	if (nb_points) {
		FREE_int(nb_points);
	}
	if (candidates) {
		FREE_int(candidates);
	}
	if (nb_candidates) {
		FREE_int(nb_candidates);
	}
	if (current_choice) {
		FREE_int(current_choice);
	}
	if (level_counter) {
		FREE_int(level_counter);
	}
	if (f_level_mod) {
		FREE_int(f_level_mod);
	}
	if (level_r) {
		FREE_int(level_r);
	}
	if (level_m) {
		FREE_int(level_m);
	}
	if (current_clique) {
		FREE_int(current_clique);
	}
	if (f_has_row_by_row_adjacency_matrix) {
		int i;
		for (i = 0; i < n; i++) {
			FREE_char(row_by_row_adjacency_matrix[i]);
		}
		FREE_pchar(row_by_row_adjacency_matrix);
	}

	null();
}




void clique_finder::init(const char *label, int n, 
	int target_depth, 
	int f_has_adj_list, int *adj_list_coded, 
	int f_has_bitvector, uchar *bitvector_adjacency, 
	int print_interval, 
	int f_maxdepth, int maxdepth, 
	int f_store_solutions, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	strcpy(clique_finder::label, label);
	
	clique_finder::f_store_solutions = f_store_solutions;
	clique_finder::n = n;
	clique_finder::target_depth = target_depth;
	clique_finder::verbose_level = verbose_level;
	clique_finder::f_maxdepth = f_maxdepth;
	clique_finder::maxdepth = maxdepth;
	clique_finder::print_interval = print_interval;

	clique_finder::f_has_adj_list = f_has_adj_list;
	clique_finder::adj_list_coded = adj_list_coded;
	clique_finder::f_has_bitvector = f_has_bitvector;
	clique_finder::bitvector_adjacency = bitvector_adjacency;

	f_has_row_by_row_adjacency_matrix = FALSE;
	row_by_row_adjacency_matrix = NULL;
	
	if (f_v) {
		cout << "clique_finder::init " << label << " n=" << n
				<< " target_depth=" << target_depth << endl;
		cout << "f_has_adj_list=" << f_has_adj_list << endl;
		cout << "f_has_bitvector=" << f_has_bitvector << endl;
	}


	if (f_v) {
		cout << "clique_finder::init before delinearize_adjacency_list" << endl;
	}
	delinearize_adjacency_list(verbose_level);
	if (f_v) {
		cout << "clique_finder::init before after delinearize_adjacency_list" << endl;
	}

	nb_sol = 0;

	pt_list = NEW_int(n);
	if (f_v) {
		cout << "clique_finder::init pt_list allocated" << endl;
	}
	pt_list_inv = NEW_int(n);
	if (f_v) {
		cout << "clique_finder::init pt_list_inv allocated" << endl;
	}
	nb_points = NEW_int(target_depth + 1);
	candidates = NEW_int((target_depth + 1) * n);
	nb_candidates = NEW_int(target_depth);
	current_choice = NEW_int(target_depth);
	level_counter = NEW_int(target_depth);
	f_level_mod = NEW_int(target_depth);
	level_r = NEW_int(target_depth);
	level_m = NEW_int(target_depth);
	current_clique = NEW_int(target_depth);

	int_vec_zero(level_counter, target_depth);
	int_vec_zero(f_level_mod, target_depth);
	int_vec_zero(level_r, target_depth);
	int_vec_zero(level_m, target_depth);


	for (i = 0; i < n; i++) {
		pt_list[i] = i;
		pt_list_inv[i] = i;
	}
	nb_points[0] = n;
	counter = 0;
	decision_step_counter = 0;

	//allocate_bitmatrix(verbose_level);

	if (f_v) {
		cout << "clique_finder::init finished" << endl;
	}
}

void clique_finder::init_restrictions(int *restrictions,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, i, r, m;

	if (f_v) {
		cout << "clique_finder::init_restrictions" << endl;
	}
	for (h = 0; ; h++) {
		if (restrictions[h * 3] == -1) {
			break;
		}
		i = restrictions[h * 3 + 0];
		r = restrictions[h * 3 + 1];
		m = restrictions[h * 3 + 2];
		if (i >= target_depth) {
			cout << "clique_finder::init_restrictions "
					"i >= target_depth" << endl;
			exit(1);
		}
		f_level_mod[i] = TRUE;
		level_r[i] = r;
		level_m[i] = m;
		cout << "clique_finder::init_restrictions level "
				<< i << " congruent " << r << " mod " << m << endl;
	}
	if (f_v) {
		cout << "clique_finder::init_restrictions done" << endl;
	}
}

void clique_finder::init_point_labels(int *pt_labels)
{
	point_labels = NEW_int(n);
	int_vec_copy(pt_labels, point_labels, n);
}

void clique_finder::init_suspicous_points(int nb, int *point_list)
{
	int i, j, idx;
	int *point_list_ordered;
	sorting Sorting;
	
	point_list_ordered = NEW_int(nb);
	for (i = 0; i < nb; i++) {
		point_list_ordered[i] = point_list[i];
	}
	Sorting.int_vec_heapsort(point_list_ordered, nb);
	point_is_suspicous = NEW_int(n);
	for (i = 0; i < n; i++) {
		point_is_suspicous[i] = FALSE;
	}
	for (i = 0; i < n; i++) {
		if (point_labels) {
			j = point_labels[i];
		}
		else {
			j = i;
		}
		if (Sorting.int_vec_search(point_list_ordered, nb, j, idx)) {
			point_is_suspicous[i] = TRUE;
		}
	}
	FREE_int(point_list_ordered);
}

void clique_finder::backtrack_search(int depth, int verbose_level)
{
	int nb_old, i, nb_new;
	int pt1, pt2, pt, pass, f_go;
	unsigned long int counter_save;
	int my_verbose_level;
	
	counter++;
	counter_save = counter;

	if (depth && nb_candidates[depth - 1] > 1) {
		decision_step_counter++;
	}
	if ((counter & ((1 << 23) - 1)) == 0) {
		my_verbose_level = 1;
	}
	else {
		my_verbose_level = verbose_level;
	}
	int f_v = (my_verbose_level >= 1);
	int f_vv = (my_verbose_level >= 2);
	//int f_vvv = (my_verbose_level >= 3);

	if (f_v) {
		cout << "clique_finder::backtrack_search : ";
		log_position(depth, counter_save, counter);
		cout << " nb_sol=" << nb_sol << " starting" << endl;
	}
	write_entry_to_tree_file(depth, verbose_level);

	if (depth == target_depth) {
	
		// We found a clique:

		if (f_v) {
			cout << "clique_finder::backtrack_search "
					"depth == target_depth" << endl;
		}
		if (f_store_solutions) {
			//cout << "storing solution" << endl;
			vector<int> sol;
			int j;
			sol.resize(depth);
			for (j = 0; j < depth; j++) {
				sol[j] = (int) current_clique[j];
				}
			solutions.push_back(sol);
			
		}
		nb_sol++;
		
		//cout << "clique_finder::backtrack_search before
		//call_back_clique_found" << endl;
		if (call_back_clique_found) {
			//cout << "calling call_back_clique_found" << endl;
			(*call_back_clique_found)(this, verbose_level);
		}
		//cout << "solution " << nb_sol << ", found a clique
		//of size target_depth" << endl;
		//cout << "clique";
		//int_set_print(cout, current_clique, depth);
		//cout << " depth = " << depth << endl;
		//exit(1);

		return;
	}


	if (f_maxdepth && depth == maxdepth) {
		return;
	}
	if (depth == 0)
		nb_old = n;
	else
		nb_old = nb_points[depth - 1];

#if 0
	if (f_v || (counter % print_interval) == 0) {
		log_position(depth, counter_save, counter);
		cout << endl;
	}

	if (f_v && depth) {
		log_position(depth, counter_save, counter);
		cout << " : # active points from previous level is "
				<< nb_old << endl;
		//cout << " : previous lvl_pt_list[" << depth - 1
		//<< "] of size " << nb_old << " : " << endl;
		////int_vec_print(cout, lvl_pt_list[depth - 1],
		//lvl_nb_points[depth - 1]);
		//print_point_set(depth, counter_save, counter,
		//nb_old, lvl_pt_list[depth - 1]);
		//cout << endl;
	}
#endif

	// first pass:
	// if depth > 0 and we are not using call_back_find_candidates, 
	// we apply the lexicographical ordering test.
	// the points are required to be greater than
	// the previous point in the clique.
	// this also eliminates the point 
	// that was added to the clique in the previous step from pt_list.
	
	if (f_v) {
		cout << "clique_finder::backtrack_search : ";
		log_position(depth, counter_save, counter);
		cout << " first pass" << endl;
	}

	if (depth && call_back_find_candidates == NULL) {
		// if we don't have a find_candidates function,
		// then we use the lexicographical ordering.
		// The idea is that we may force the clique to be 
		// constructed in increasing order of its points.
		// Hence, now we can eliminate all points that 
		// are smaller than the most recently added clique point:
		nb_new = 0;
		pt1 = current_clique[depth - 1];
		for (i = 0; i < nb_old; i++) {
			pt2 = pt_list[i];
			if (pt2 > pt1) {
				swap_point(nb_new, i);
				nb_new++;
			}
		}
	}
	else {
		nb_new = nb_old;
	}
	

	// second pass: find the points that are connected with the 
	// previously chosen clique point:
	
	nb_old = nb_new;	
	if (depth) {
		nb_new = 0;
		pt1 = current_clique[depth - 1];
		for (i = 0; i < nb_old; i++) {
			// go over all points in the old list:
			
			pt2 = pt_list[i];
			if (is_adjacent(depth, pt1, pt2)) {

				// point is adjacent, so we keep that point

				swap_point(nb_new, i);
				nb_new++;
			}
			
		}
	}
	else {
		nb_new = nb_old;
	}
	

	pass = 2;
	
	if (f_vv) {
		log_position(depth, counter_save, counter);
		cout << " : pass 2: ";
		print_suspicous_point_subset(nb_new, pt_list);
		cout << endl;
	}


#if 0
	// higher passes: 
	// find the points that have sufficiently high degree:
	
	do {
		nb_old = nb_new;
		nb_new = 0;
		for (i = 0; i < nb_old; i++) {
			d = degree_of_point(i, nb_old);
			if (d >= target_depth - depth - 1) {
				swap_point(nb_new, i);
				nb_new++;
			}
			else {
				if (point_is_suspicous && 
					point_is_suspicous[pt_list[i]]) {
					log_position(depth, counter_save, counter);
					cout << " : pass " << pass 
						<< ": suspicous point " << point_label(pt_list[i])
						<< " eliminated, d=" << d
						<< " is less than target_depth - depth - 1 = "
						<< target_depth - depth - 1 << endl;;
					degree_of_point_verbose(i, nb_old);
				}
			}
		}
		pass++;

		if (f_vv) {
			log_position(depth, counter_save, counter);
			cout << " : pass " << pass << ": ";
			print_suspicous_point_subset(nb_new, pt_list);
			cout << endl;
		}

	} while (nb_new < nb_old);
#endif

	nb_points[depth] = nb_new;


	
	if (f_v) {
		log_position(depth, counter_save, counter);
		cout << "after " << pass << " passes: nb_points = "
				<< nb_new << endl;
	}
	

	if (call_back_after_reduction) {
		if (f_v) {
			log_position(depth, counter_save, counter);
			cout << " before call_back_after_reduction" << endl;
		}
		(*call_back_after_reduction)(this, depth,
				nb_points[depth], verbose_level);
		if (f_v) {
			log_position(depth, counter_save, counter);
			cout << " after call_back_after_reduction" << endl;
		}
	}


	{
		//int i; //, nb_old;

		if (call_back_find_candidates) {
			int reduced_nb_points;

			if (f_v) {
				log_position(depth, counter_save, counter);
				cout << " before call_back_find_candidates" << endl;
			}
			nb_candidates[depth] = (*call_back_find_candidates)(this,
				depth, current_clique,
				nb_points[depth], reduced_nb_points, pt_list, pt_list_inv,
				candidates + depth * n,
				0/*verbose_level*/);

			if (f_v) {
				log_position(depth, counter_save, counter);
				cout << " after call_back_find_candidates nb_candidates="
						<< nb_candidates[depth] << endl;
			}
			// The set of candidates is stored in
			// candidates + depth * n.
			// The number of candidates is in nb_candidates[depth]


#ifdef SUSPICOUS
			if (f_vv) {
				if (point_is_suspicous) {
					cout << "candidate set of size "
						<< nb_candidates[depth] << endl;
					print_suspicous_point_subset(
					nb_candidates[depth],
						candidates + depth * n);
					cout << endl;
				}
			}
#endif
			nb_points[depth] = reduced_nb_points;
		}
		else {
			// If we don't have a find_candidates callback,
			// we take all the points into consideration:

			int_vec_copy(pt_list, candidates + depth * n, nb_points[depth]);
			nb_candidates[depth] = nb_points[depth];
		}
	}


	// added Dec 2014:
	if (f_has_print_current_choice_function) {
		(*call_back_print_current_choice)(this, depth,
				print_current_choice_data, verbose_level);
	}
	
	// Now we are ready to go in the backtrack search.
	// We'll try each of the points in candidates one by one:

	for (current_choice[depth] = 0;
			current_choice[depth] < nb_candidates[depth];
			current_choice[depth]++, level_counter[depth]++) {


		if (f_v) {
			log_position(depth, counter_save, counter);
			cout << " choice " << current_choice[depth] << " / "
					<< nb_candidates[depth] << endl;
		}

		f_go = TRUE;  // Whether we want to go in the recursion.

		if (f_level_mod[depth]) {
			if ((level_counter[depth] % level_m[depth]) != level_r[depth]) {
				f_go = FALSE;
			}
		}


		pt = candidates[depth * n + current_choice[depth]];


		if (f_vv) {
			log_position_and_choice(depth, counter_save, counter);
			cout << endl;
		}



		// We add a point under consideration:
		
		current_clique[depth] = pt;

		if (call_back_add_point) {
			if (FALSE /*f_v*/) {
				cout << "before call_back_add_point" << endl;
			}
			(*call_back_add_point)(this, depth, 
				current_clique, pt, 0/*verbose_level*/);
			if (FALSE /*f_v*/) {
				cout << "after call_back_add_point" << endl;
			}
		}

		if (point_is_suspicous) {
			if (point_is_suspicous[pt]) {
				log_position(depth, counter_save, counter);
				cout << " : considering clique ";
				print_set(depth + 1, current_clique);
				//int_set_print(cout, current_clique, depth);
				cout << " depth = " << depth << " nb_old="
						<< nb_old << endl;
				f_go = TRUE;
			}
			else {
				f_go = FALSE;
			}
		}
	

		// and now, let's do the recursion:

		if (f_go) {
			backtrack_search(depth + 1, verbose_level);
		} // if (f_go)





		// We delete the point:

		if (call_back_delete_point) {
			if (FALSE /*f_v*/) {
				cout << "before call_back_delete_point" << endl;
			}
			(*call_back_delete_point)(this, depth, 
				current_clique, pt, 0/*verbose_level*/);
			if (FALSE /*f_v*/) {
				cout << "after call_back_delete_point" << endl;
			}
		}

	} // for current_choice[depth]


	
	if (f_v) {
		cout << "backtrack_search : ";
		log_position(depth, counter_save, counter);
		cout << " nb_sol=" << nb_sol << " done" << endl;
	}
}

int clique_finder::solve_decision_problem(int depth, int verbose_level)
// returns TRUE if we found a solution
{
	int nb_old, i, nb_new;
	int pt1, pt2, pt, pass, f_go;
	unsigned long int counter_save;
	int my_verbose_level;
	
	counter++;
	counter_save = counter;

	if (depth && nb_candidates[depth - 1] > 1) {
		decision_step_counter++;
	}
	if ((counter & ((1 << 17) - 1)) == 0) {
		my_verbose_level = verbose_level + 1;
	}
	else {
		my_verbose_level = verbose_level;
	}
	int f_v = (my_verbose_level >= 1);
	int f_vv = (my_verbose_level >= 2);
	//int f_vvv = (my_verbose_level >= 3);

	if (f_v) {
		cout << "solve_decision_problem : ";
		log_position(depth, counter_save, counter);
		cout << " nb_sol=" << nb_sol << " starting" << endl;
	}
	write_entry_to_tree_file(depth, verbose_level);

	if (depth == target_depth) {
		nb_sol++;
		//cout << "clique_finder::backtrack_search before
		//call_back_clique_found" << endl;
		if (call_back_clique_found) {
			(*call_back_clique_found)(this, verbose_level);
		}
		//cout << "solution " << nb_sol << ", found a clique
		//of size target_depth" << endl;
		//cout << "clique";
		//int_set_print(cout, current_clique, depth);
		//cout << " depth = " << depth << endl;
		//exit(1);

		return TRUE;
	}


	if (f_maxdepth && depth == maxdepth) {
		return FALSE;
	}
	if (depth == 0)
		nb_old = n;
	else
		nb_old = nb_points[depth - 1];

#if 0
	if (f_v || (counter % print_interval) == 0) {
		log_position(depth, counter_save, counter);
		cout << endl;
	}

	if (f_v && depth) {
		log_position(depth, counter_save, counter);
		cout << " : # active points from previous level is "
				<< nb_old << endl;
		//cout << " : previous lvl_pt_list[" << depth - 1
		//<< "] of size " << nb_old << " : " << endl;
		////int_vec_print(cout, lvl_pt_list[depth - 1],
		// lvl_nb_points[depth - 1]);
		//print_point_set(depth, counter_save, counter,
		//nb_old, lvl_pt_list[depth - 1]);
		//cout << endl;
	}
#endif

	// first pass:
	// if depth > 0 and we are not using call_back_find_candidates, 
	// we apply the lexicographical ordering test.
	// the points are required to be greater than the
	// previous point in the clique.
	// this also eliminates the point 
	// that was added to the clique in the previous step from pt_list.
	
	if (depth && call_back_find_candidates == NULL) {
		nb_new = 0;
		pt1 = current_clique[depth - 1];
		for (i = 0; i < nb_old; i++) {
			pt2 = pt_list[i];
			if (pt2 > pt1) {
				swap_point(nb_new, i);
				nb_new++;
			}
		}
	}
	else {
		nb_new = nb_old;
	}
	

	// second pass: find the points that are connected with the 
	// previously chosen clique point:
	
	nb_old = nb_new;	
	if (depth) {
		nb_new = 0;
		pt1 = current_clique[depth - 1];
		for (i = 0; i < nb_old; i++) {
			pt2 = pt_list[i];
			if (is_adjacent(depth, pt1, pt2)) {
				swap_point(nb_new, i);
				nb_new++;
			}
		}
	}
	else {
		nb_new = nb_old;
	}
	

	pass = 2;
	
	if (f_vv) {
		log_position(depth, counter_save, counter);
		cout << " : pass 2: ";
		print_suspicous_point_subset(nb_new, pt_list);
		cout << endl;
	}


#if 0
	// higher passes: 
	// find the points that have sufficiently high degree:
	
	do {
		nb_old = nb_new;
		nb_new = 0;
		for (i = 0; i < nb_old; i++) {
			d = degree_of_point(i, nb_old);
			if (d >= target_depth - depth - 1) {
				swap_point(nb_new, i);
				nb_new++;
			}
			else {
				if (point_is_suspicous && 
					point_is_suspicous[pt_list[i]]) {
					log_position(depth, counter_save, counter);
					cout << " : pass " << pass 
						<< ": suspicous point "
						<< point_label(pt_list[i])
						<< " eliminated, d=" << d
						<< " is less than target_depth - depth - 1 = "
						<< target_depth - depth - 1 << endl;;
					degree_of_point_verbose(i, nb_old);
				}
			}
		}
		pass++;

		if (f_vv) {
			log_position(depth, counter_save, counter);
			cout << " : pass " << pass << ": ";
			print_suspicous_point_subset(nb_new, pt_list);
			cout << endl;
		}

	} while (nb_new < nb_old);
#endif

	nb_points[depth] = nb_new;


	
	if (f_v) {
		log_position(depth, counter_save, counter);
		cout << "after " << pass << " passes: "
				"nb_points = " << nb_new << endl;
	}
	

	if (call_back_after_reduction) {
		(*call_back_after_reduction)(this, depth,
				nb_points[depth], verbose_level);
	}


	{
		int i; //, nb_old;

		if (call_back_find_candidates) {
			int reduced_nb_points;

			nb_candidates[depth] = (*call_back_find_candidates)(this,
				depth, current_clique,
				nb_points[depth], reduced_nb_points,
				pt_list, pt_list_inv,
				candidates + depth * n,
				0/*verbose_level*/);
#ifdef SUSPICOUS
			if (f_vv) {
				if (point_is_suspicous) {
					cout << "candidate set of size "
						<< nb_candidates[depth] << endl;
					print_suspicous_point_subset(
					nb_candidates[depth],
						candidates + depth * n);
					cout << endl;
				}
			}
#endif
			nb_points[depth] = reduced_nb_points;
		}
		else {
			for (i = 0; i < nb_points[depth]; i++) {
				candidates[depth * n + i] = pt_list[i];
			}
			nb_candidates[depth] = nb_points[depth];
		}
	}




	for (current_choice[depth] = 0;
			current_choice[depth] < nb_candidates[depth];
			current_choice[depth]++, level_counter[depth]++) {

		pt = candidates[depth * n + current_choice[depth]];

		f_go = TRUE;

		if (f_vv) {
			log_position_and_choice(depth, counter_save, counter);
			cout << endl;
		}
		// add a point
		
		current_clique[depth] = pt;

		if (call_back_add_point) {
			if (FALSE /*f_v*/) {
				cout << "before call_back_add_point" << endl;
			}
			(*call_back_add_point)(this, depth, 
				current_clique, pt, 0/*verbose_level*/);
			if (FALSE /*f_v*/) {
				cout << "after call_back_add_point" << endl;
			}
		}

		if (point_is_suspicous) {
			if (point_is_suspicous[pt]) {
				log_position(depth, counter_save, counter);
				cout << " : considering clique ";
				print_set(depth + 1, current_clique);
				//int_set_print(cout, current_clique, depth);
				cout << " depth = " << depth << " nb_old="
						<< nb_old << endl;
				f_go = TRUE;
			}
			else {
				f_go = FALSE;
			}
		}
	
		if (f_go) {
			if (solve_decision_problem(depth + 1, verbose_level)) {
				return TRUE;
			}
		} // if (f_go)

		// delete a point:

		if (call_back_delete_point) {
			if (FALSE /*f_v*/) {
				cout << "before call_back_delete_point" << endl;
			}
			(*call_back_delete_point)(this, depth, 
				current_clique, pt, 0/*verbose_level*/);
			if (FALSE /*f_v*/) {
				cout << "after call_back_delete_point" << endl;
			}
		}

	} // for current_choice[depth]


	
	if (f_v) {
		cout << "solve_decision_problem : ";
		log_position(depth, counter_save, counter);
		cout << " nb_sol=" << nb_sol << " done" << endl;
	}
	return FALSE;
}

void all_cliques_of_given_size(int *Adj,
		int nb_pts, int clique_sz, int *&Sol, int &nb_sol,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *adj_list_coded;
	int n2;
	int i, j, h;
	clique_finder *C;
	const char *label = "all_cliques_of_given_size";
	int print_interval = 1000;
	int f_maxdepth = FALSE;
	int maxdepth = 0;

	if (f_v) {
		cout << "all_cliques_of_given_size" << endl;
	}
	n2 = (nb_pts * (nb_pts - 1)) >> 1;
	adj_list_coded = NEW_int(n2);
	h = 0;
	cout << "all_cliques_of_given_size: "
			"computing adj_list_coded" << endl;
	for (i = 0; i < nb_pts; i++) {
		for (j = i + 1; j < nb_pts; j++) {
			adj_list_coded[h++] = Adj[i * nb_pts + j];
		}
	}
	
	C = NEW_OBJECT(clique_finder);
	
	if (f_v) {
		cout << "all_cliques_of_given_size: before C->init" << endl;
	}
	C->init(label, nb_pts, 
		clique_sz, 
		TRUE, adj_list_coded, 
		FALSE, NULL, 
		print_interval, 
		f_maxdepth, maxdepth, 
		TRUE /* f_store_solutions */, 
		verbose_level);

	C->backtrack_search(0 /* depth */, 0 /* verbose_level */);

	if (f_v) {
		cout << "all_cliques_of_given_size done with search, "
				"we found " << C->nb_sol << " solutions" << endl;
	}

	int sz;
	C->get_solutions(Sol, nb_sol, sz, verbose_level);
	if (sz != clique_sz) {
		cout << "all_cliques_of_given_size sz != clique_sz" << endl;
		exit(1);
	}
	FREE_OBJECT(C);
	FREE_int(adj_list_coded);
	if (f_v) {
		cout << "all_cliques_of_given_size done" << endl;
	}
}


	static int *nb_old, *nb_new;
	static int *pt1, *pt2, *pt, *pass, *f_go;
	static unsigned long int *counter_save;

void clique_finder::backtrack_search_not_recursive(int verbose_level)
{
	int depth;
	int i;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	nb_old = NEW_int(target_depth);
	nb_new = NEW_int(target_depth);
	pt1 = NEW_int(target_depth);
	pt2 = NEW_int(target_depth);
	pt = NEW_int(target_depth);
	pass = NEW_int(target_depth);
	f_go = NEW_int(target_depth);
	counter_save = (unsigned long int *) NEW_lint(target_depth);
	

	depth = 0;

entrance_point:


	counter++;
	counter_save[depth] = counter;


	if (f_v) {
		cout << "clique_finder::backtrack_search : ";
		log_position(depth, counter_save[depth], counter);
		cout << " nb_sol=" << nb_sol << " starting" << endl;
	}
	write_entry_to_tree_file(depth, verbose_level);

	if (depth == target_depth) {
	
		// We found a clique:

		if (f_v) {
			cout << "clique_finder::backtrack_search "
					"depth == target_depth" << endl;
		}
		if (f_store_solutions) {
			//cout << "storing solution" << endl;
			vector<int> sol;
			int j;
			sol.resize(depth);
			for (j = 0; j < depth; j++) {
				sol[j] = (int) current_clique[j];
			}
			solutions.push_back(sol);
			
		}
		nb_sol++;
		
		//cout << "clique_finder::backtrack_search
		// before call_back_clique_found" << endl;
		if (call_back_clique_found) {
			//cout << "calling call_back_clique_found" << endl;
			(*call_back_clique_found)(this, verbose_level);
		}
		//cout << "solution " << nb_sol << ", found a clique
		// of size target_depth" << endl;
		//cout << "clique";
		//int_set_print(cout, current_clique, depth);
		//cout << " depth = " << depth << endl;
		//exit(1);

		goto continuation_point;
	}


	if (f_maxdepth && depth == maxdepth) {
		goto continuation_point;
	}
	if (depth == 0)
		nb_old[depth] = n;
	else
		nb_old[depth] = nb_points[depth - 1];

#if 0
	if (f_v || (counter % print_interval) == 0) {
		log_position(depth, counter_save, counter);
		cout << endl;
	}

	if (f_v && depth) {
		log_position(depth, counter_save, counter);
		cout << " : # active points from previous level is "
				<< nb_old << endl;
		//cout << " : previous lvl_pt_list[" << depth - 1
		// << "] of size " << nb_old << " : " << endl;
		////int_vec_print(cout, lvl_pt_list[depth - 1],
		// lvl_nb_points[depth - 1]);
		//print_point_set(depth, counter_save, counter,
		// nb_old, lvl_pt_list[depth - 1]);
		//cout << endl;
	}
#endif

	// first pass:
	// if depth > 0 and we are not using call_back_find_candidates, 
	// we apply the lexicographical ordering test.
	// the points are required to be greater than the
	// previous point in the clique.
	// this also eliminates the point 
	// that was added to the clique in the previous step from pt_list.
	
	if (f_v) {
		cout << "clique_finder::backtrack_search : ";
		log_position(depth, counter_save[depth], counter);
		cout << " first pass" << endl;
	}

	if (depth && call_back_find_candidates == NULL) {
		// if we don't have a find_candidates function,
		// then we use the lexicographical ordering.
		// The idea is that we may force the clique to be 
		// constructed in increasing order of its points.
		// Hence, now we can eliminate all points that 
		// are smaller than the most recently added clique point:
		nb_new[depth] = 0;
		pt1[depth] = current_clique[depth - 1];
		for (i = 0; i < nb_old[depth]; i++) {
			pt2[depth] = pt_list[i];
			if (pt2[depth] > pt1[depth]) {
				swap_point(nb_new[depth], i);
				nb_new[depth]++;
			}
		}
	}
	else {
		nb_new[depth] = nb_old[depth];
	}
	

	// second pass: find the points that are connected with the 
	// previously chosen clique point:
	
	nb_old[depth] = nb_new[depth];	
	if (depth) {
		nb_new[depth] = 0;
		pt1[depth] = current_clique[depth - 1];
		for (i = 0; i < nb_old[depth]; i++) {
			// go over all points in the old list:
			
			pt2[depth] = pt_list[i];
			if (is_adjacent(depth, pt1[depth], pt2[depth])) {

				// point is adjacent, so we keep that point

				swap_point(nb_new[depth], i);
				nb_new[depth]++;
			}
			
		}
	}
	else {
		nb_new[depth] = nb_old[depth];
	}
	

	pass[depth] = 2;
	
	if (f_vv) {
		log_position(depth, counter_save[depth], counter);
		cout << " : pass 2: ";
		print_suspicous_point_subset(nb_new[depth], pt_list);
		cout << endl;
	}


#if 0
	// higher passes: 
	// find the points that have sufficiently high degree:
	
	do {
		nb_old = nb_new;
		nb_new = 0;
		for (i = 0; i < nb_old; i++) {
			d = degree_of_point(i, nb_old);
			if (d >= target_depth - depth - 1) {
				swap_point(nb_new, i);
				nb_new++;
			}
			else {
				if (point_is_suspicous && 
					point_is_suspicous[pt_list[i]]) {
					log_position(depth, counter_save, counter);
					cout << " : pass " << pass 
						<< ": suspicous point "
						<< point_label(pt_list[i])
						<< " eliminated, d=" << d
						<< " is less than target_depth - depth - 1 = "
						<< target_depth - depth - 1 << endl;;
					degree_of_point_verbose(i, nb_old);
				}
			}
		}
		pass++;

		if (f_vv) {
			log_position(depth, counter_save, counter);
			cout << " : pass " << pass << ": ";
			print_suspicous_point_subset(nb_new, pt_list);
			cout << endl;
		}

	} while (nb_new < nb_old);
#endif

	nb_points[depth] = nb_new[depth];


	
	if (f_v) {
		log_position(depth, counter_save[depth], counter);
		cout << "after " << pass[depth] << " passes: "
				"nb_points = " << nb_new[depth] << endl;
	}
	

	if (call_back_after_reduction) {
		if (f_v) {
			log_position(depth, counter_save[depth], counter);
			cout << " before call_back_after_reduction" << endl;
		}
		(*call_back_after_reduction)(this, depth,
				nb_points[depth], verbose_level);
		if (f_v) {
			log_position(depth, counter_save[depth], counter);
			cout << " after call_back_after_reduction" << endl;
		}
	}


	{
		//int i; //, nb_old;

		if (call_back_find_candidates) {
			int reduced_nb_points;

			if (f_v) {
				log_position(depth, counter_save[depth], counter);
				cout << " before call_back_find_candidates" << endl;
			}
			nb_candidates[depth] = (*call_back_find_candidates)(this,
				depth, current_clique,
				nb_points[depth], reduced_nb_points, pt_list, pt_list_inv,
				candidates + depth * n,
				0/*verbose_level*/);

			if (f_v) {
				log_position(depth, counter_save[depth], counter);
				cout << " after call_back_find_candidates "
						"nb_candidates=" << nb_candidates[depth] << endl;
			}
			// The set of candidates is stored in
			// candidates + depth * n.
			// The number of candidates is in nb_candidates[depth]


#ifdef SUSPICOUS
			if (f_vv) {
				if (point_is_suspicous) {
					cout << "candidate set of size "
						<< nb_candidates[depth] << endl;
					print_suspicous_point_subset(
					nb_candidates[depth],
						candidates + depth * n);
					cout << endl;
				}
			}
#endif
			nb_points[depth] = reduced_nb_points;
		}
		else {
			// If we don't have a find_candidates callback,
			// we take all the points into consideration:

			int_vec_copy(pt_list, candidates + depth * n,
					nb_points[depth]);
			nb_candidates[depth] = nb_points[depth];
		}
	}


	// added Dec 2014:
	if (f_has_print_current_choice_function) {
		(*call_back_print_current_choice)(this, depth,
				print_current_choice_data, verbose_level);
	}
	
	// Now we are ready to go in the backtrack search.
	// We'll try each of the points in candidates one by one:

	for (current_choice[depth] = 0;
			current_choice[depth] < nb_candidates[depth];
			current_choice[depth]++, level_counter[depth]++) {


		if (f_v) {
			log_position(depth, counter_save[depth], counter);
			cout << " choice " << current_choice[depth]
				<< " / " << nb_candidates[depth] << endl;
		}

		f_go[depth] = TRUE;  // Whether we want to go in the recursion.

		if (f_level_mod[depth]) {
			if ((level_counter[depth] % level_m[depth]) != level_r[depth]) {
				f_go[depth] = FALSE;
			}
		}


		pt[depth] = candidates[depth * n + current_choice[depth]];


		if (f_vv) {
			log_position_and_choice(depth, counter_save[depth], counter);
			cout << endl;
		}



		// We add a point under consideration:
		
		current_clique[depth] = pt[depth];

		if (call_back_add_point) {
			if (FALSE /*f_v*/) {
				cout << "before call_back_add_point" << endl;
			}
			(*call_back_add_point)(this, depth, 
				current_clique, pt[depth], 0/*verbose_level*/);
			if (FALSE /*f_v*/) {
				cout << "after call_back_add_point" << endl;
			}
		}

		if (point_is_suspicous) {
			if (point_is_suspicous[pt[depth]]) {
				log_position(depth, counter_save[depth], counter);
				cout << " : considering clique ";
				print_set(depth + 1, current_clique);
				//int_set_print(cout, current_clique, depth);
				cout << " depth = " << depth << " nb_old=" << nb_old << endl;
				f_go[depth] = TRUE;
			}
			else {
				f_go[depth] = FALSE;
			}
		}
	

		// and now, let's do the recursion:

		if (f_go[depth]) {
			//backtrack_search(depth + 1, verbose_level);
			depth++;
			goto entrance_point;


continuation_point:
			depth--;
		} // if (f_go)





		// We delete the point:

		if (call_back_delete_point) {
			if (FALSE /*f_v*/) {
				cout << "before call_back_delete_point" << endl;
				}
			(*call_back_delete_point)(this, depth, 
				current_clique, pt[depth], 0/*verbose_level*/);
			if (FALSE /*f_v*/) {
				cout << "after call_back_delete_point" << endl;
			}
		}

	} // for current_choice[depth]

	if (depth) {
		goto continuation_point;
	}


	FREE_int(nb_old);
	FREE_int(nb_new);
	FREE_int(pt1);
	FREE_int(pt2);
	FREE_int(pt);
	FREE_int(pass);
	FREE_int(f_go);
	FREE_lint((long int *) counter_save);
	
	if (f_v) {
		cout << "backtrack_search : ";
		log_position(depth, counter_save[depth], counter);
		cout << " nb_sol=" << nb_sol << " done" << endl;
	}
}

void clique_finder::open_tree_file(const char *fname_base,
		int f_decision_nodes_only)
{
	f_write_tree = TRUE;
	clique_finder::f_decision_nodes_only = f_decision_nodes_only;
	snprintf(fname_tree, 1000, "%s.tree", fname_base);
	fp_tree = new ofstream;
	fp_tree->open(fname_tree);
}

void clique_finder::close_tree_file()
{
	file_io Fio;

	*fp_tree << -1 << endl;
	fp_tree->close();
	delete fp_tree;
	cout << "written file " << fname_tree << " of size "
			<< Fio.file_size(fname_tree) << endl;
}

void clique_finder::get_solutions(int *&Sol,
		int &nb_solutions, int &clique_sz, int verbose_level)
{
	int i, j;

	nb_solutions = nb_sol;
	//nb_sol = nb_sol;
	clique_sz = target_depth;
	Sol = NEW_int(nb_sol * target_depth);
	for (i = 0; i < nb_sol; i++) {
		for (j = 0; j < target_depth; j++) {
			Sol[i * target_depth + j] = solutions.front()[j];
		}
		solutions.pop_front();
	}
}

void clique_finder::print_suspicous_points()
{
	int i, j;

	cout << "suspicous points: ";
	for (i = 0; i < n; i++) {
		if (point_is_suspicous[i]) {
			if (point_labels) {
				j = point_labels[i];
			}
			else {
				j = i;
			}
			cout << j << " ";
		}
	}
	cout << endl;
}

void clique_finder::print_set(int size, int *set)
{
	int i, a, b;

	cout << "(";
	for (i = 0; i < size; i++) {
		a = set[i];
		b = point_label(a);
		cout << b;
		if (i < size - 1) {
			cout << ", ";
		}
	}
	cout << ")";
}

void clique_finder::print_suspicous_point_subset(int size, int *set)
{
	int i, a, b, cnt = 0;

	for (i = 0; i < size; i++) {
		a = set[i];
		if (!is_suspicous(a)) {
			continue;
		}
		cnt++;
	}
	cout << cnt << "(";
	for (i = 0; i < size; i++) {
		a = set[i];
		if (!is_suspicous(a)) {
			continue;
		}
		//if (point_is_suspicous && !point_is_suspicous[a])
			//continue;
		b = point_label(a);
		cout << b;
		if (i < size - 1) {
			cout << ", ";
		}
	}
	cout << ")";
}

void clique_finder::log_position_and_choice(int depth,
		unsigned long int counter_save, unsigned long int counter)
{
	cout << "node " << counter << " at depth " << depth << " : ";
	log_choice(depth + 1);
	cout << " nb_sol=" << nb_sol << " ";
	if (FALSE) {
		cout << " clique ";
		int_set_print(cout, current_clique, depth);
	}
}

void clique_finder::log_position(int depth,
		unsigned long int counter_save, unsigned long int counter)
{
	cout << "node " << counter << " at depth " << depth << " : ";
	log_choice(depth);
	if (FALSE) {
		cout << " clique ";
		int_set_print(cout, current_clique, depth);
	}
}

void clique_finder::log_choice(int depth)
{
	int i;

	cout << "choice ";
	for (i = 0; i < depth; i++) {
		cout << i << ": " << current_choice[i]
			<< "/" << nb_candidates[i] << "("
			<< nb_points[i] << ")";
		if (i < depth - 1) {
			cout << ", ";
		}
	}
	cout << " ";
}

void clique_finder::swap_point(int idx1, int idx2)
{
	sorting Sorting;

	Sorting.int_vec_swap_points(pt_list, pt_list_inv, idx1, idx2);
}

void clique_finder::degree_of_point_statistic(int depth,
		int nb_points, int verbose_level)
{
	int *D;
	int i;

	D = NEW_int(nb_points);
	for (i = 0; i < nb_points; i++) {
		D[i] = degree_of_point(depth, i, nb_points);
	}
	tally C;
	int f_second = FALSE;

	C.init(D, nb_points, f_second, verbose_level);
	C.print(FALSE /* f_backwards */);

	FREE_int(D);
}

int clique_finder::degree_of_point(int depth, int i, int nb_points)
{
	int pti, ptj, j, d;

	pti = pt_list[i];
	d = 0;
	for (j = 0; j < nb_points; j++) {
		if (j == i) {
			continue;
		}
		ptj = pt_list[j];
		if (is_adjacent(depth, pti, ptj)) {
			d++;
		}
	}
	return d;
}

int clique_finder::is_suspicous(int i)
{
	if (point_is_suspicous == NULL)
		return FALSE;
	return point_is_suspicous[i];
}

int clique_finder::point_label(int i)
{
	if (point_labels)
		return point_labels[i];
	else
		return i;
}

int clique_finder::is_adjacent(int depth, int i, int j)
{
	int a;


#if 0
	if (i == j) {
		return 0;
		}
#endif

	//a = adjacency[i * n + j];
	if (f_has_row_by_row_adjacency_matrix) {
		a = row_by_row_adjacency_matrix[i][j];
	}
	else {
		a = s_ij(i, j);
	}


#if 0
	if (a == -1) {
		a = (*call_back_is_adjacent)(this, i, j, 0/* verbose_level */);
		adjacency[i * n + j] = a;
		adjacency[j * n + i] = a;
		}
#endif
	return a;
}

void clique_finder::write_entry_to_tree_file(int depth,
		int verbose_level)
{
	int i;

	if (!f_write_tree) {
		return;
	}

#if 0
		*fp_tree << "# " << depth << " ";
		for (i = 0; i < depth; i++) {
			*fp_tree << current_clique[i] << " ";
			}
		*fp_tree << endl;
#endif

	if (f_decision_nodes_only && nb_candidates[depth - 1] == 1) {
		return;
	}
	if (f_decision_nodes_only && depth == 0) {
		return;
	}
	if (f_decision_nodes_only) {
		int d;

		d = 0;
		for (i = 0; i < depth; i++) {
			if (nb_candidates[i] > 1) {
				d++;
			}
		}
		*fp_tree << d << " ";
		for (i = 0; i < depth; i++) {
			if (nb_candidates[i] > 1) {
				*fp_tree << current_clique[i] << " ";
			}
		}
		*fp_tree << endl;
	}
	else {
		*fp_tree << depth << " ";
		for (i = 0; i < depth; i++) {
			*fp_tree << current_clique[i] << " ";
		}
		*fp_tree << endl;
	}
}

void clique_finder::m_iji(int i, int j, int a)
{
	int m, n; //, N; //, jj, bit;
	//uchar mask;

	m = bitmatrix_m;
	n = bitmatrix_n;
	//N = bitmatrix_N;
	if (i < 0 || i >= m) {
		cout << "clique_finder::m_iji addressing error, i = "
				<< i << ", m = " << m << endl;
		exit(1);
	}
	if (j < 0 || j >= n) {
		cout << "clique_finder::m_iji addressing error, j = "
				<< j << ", n = " << n << endl;
		exit(1);
	}

	bitvector_m_ii(bitmatrix_adjacency, (long int) i * (long int) n + (long int) j, a);

#if 0
	jj = j >> 3;
	bit = j & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitmatrix_adjacency[i * N + jj];
	if (a == 0) {
		uchar not_mask = ~mask;
		x &= not_mask;
		}
	else {
		x |= mask;
		}
#endif
}

int clique_finder::s_ij(int i, int j)
{
	long int k;
	int aij;
	combinatorics_domain Combi;

	if (i < 0 || i >= n) {
		cout << "clique_finder::s_ij addressing error, i = "
				<< i << ", n = " << n << endl;
		exit(1);
	}
	if (j < 0 || j >= n) {
		cout << "clique_finder::s_ij addressing error, j = "
				<< j << ", n = " << n << endl;
		exit(1);
	}
	if (i == j) {
		return 0;
	}

	if (f_has_row_by_row_adjacency_matrix) {
		return row_by_row_adjacency_matrix[i][j];
	}
	else if (f_has_bitmatrix) {
		return bitvector_s_i(bitmatrix_adjacency, i * n + j);
	}
	else if (f_has_adj_list) {
		if (i == j) {
			return 0;
		}
		k = Combi.ij2k_lint(i, j, n);
		aij = adj_list_coded[k];
		return aij;
	}
	else if (f_has_bitvector) {
		if (i == j) {
			return 0;
		}
		k = Combi.ij2k_lint(i, j, n);
		aij = bitvector_s_i(bitvector_adjacency, k);
		return aij;
	}
	else {
		cout << "clique_finder::s_ij we don't have a matrix" << endl;
		exit(1);
	}

#if 0
	//uchar mask;
	jj = j >> 3;
	bit = j & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitmatrix_adjacency[i * N + jj];
	if (x & mask)
		return 1;
	else
		return 0;
#endif
}

void clique_finder::delinearize_adjacency_list(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k;
	int aij;

	if (f_v) {
		cout << "clique_finder::delinearize_adjacency_list" << endl;
	}
	row_by_row_adjacency_matrix = NEW_pchar(n);
	for (i = 0; i < n; i++) {
		row_by_row_adjacency_matrix[i] = NEW_char(n);
		for (j = 0; j < n; j++) {
			row_by_row_adjacency_matrix[i][j] = 0;
		}
	}
	k = 0;
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (f_has_bitvector) {
				aij = bitvector_s_i(bitvector_adjacency, k);
			}
			else if (f_has_adj_list) {
				aij = adj_list_coded[k];
			}
			else {
				cout << "clique_finder::delinearize_adjacency_list "
						"we don't have bitvector or adjacency list" << endl;
				exit(1);
			}
			if (aij) {
				row_by_row_adjacency_matrix[i][j] = 1;
				row_by_row_adjacency_matrix[j][i] = 1;
			}
			k++;
		}
	}
	f_has_row_by_row_adjacency_matrix = TRUE;
	if (f_v) {
		cout << "clique_finder::delinearize_adjacency_list done" << endl;
	}
}

void clique_finder::allocate_bitmatrix(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k, size;

	if (f_v) {
		cout << "clique_finder::allocate_bitmatrix" << endl;
	}
	bitmatrix_N = (n + 7) >> 3; // 1 char = 8 bits = 2^3
	bitmatrix_m = n;
	bitmatrix_n = n;
	size = (long int) bitmatrix_m * (long int) bitmatrix_N;
	if (f_v) {
		cout << "clique_finder::allocate_bitmatrix "
				"allocating BITMATRIX of size " << size << endl;
	}
	bitmatrix_adjacency = NEW_uchar(size);
	f_has_bitmatrix = TRUE;

	if (f_v) {
		cout << "clique_finder::allocate_bitmatrix "
				"adjacency matrix allocated" << endl;
	}


	if (f_v) {
		cout << "clique_finder::allocate_bitmatrix "
				"initializing adjacency matrix:" << endl;
	}


	k = 0;
	for (i = 0; i < n; i++) {
		for (j = i; j < n; j++) {
			if (f_v && ((k % (1 << 19)) == 0)) {
				cout << k << " : i=" << i << " j=" << j << endl;
			}
			if (i == j) {
				//adjacency[i * n + j] = 0;
				m_iji(i, j, 0);
			}
			else {
				//adjacency[i * n + j] = -1;
				//adjacency[j * n + i] = -1;
				//k = ij2k(i, j, n);
				//adjacency[i * n + j] = adj_list_coded[k];
				//adjacency[j * n + i] = adj_list_coded[k];

				int aij = 0;

				if (f_has_adj_list) {
					aij = adj_list_coded[k];
				}
				else if (f_has_bitvector) {
					aij = bitvector_s_i(bitvector_adjacency, k);
				}
				m_iji(i, j, aij);
				m_iji(j, i, aij);
				k++;
			}
		}
	}
	if (f_v) {
		cout << "clique_finder::allocate_bitmatrix done" << endl;
	}
}




}
}
