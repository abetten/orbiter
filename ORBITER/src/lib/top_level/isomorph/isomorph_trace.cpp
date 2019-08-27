// isomorph_trace.cpp
// 
// Anton Betten
// Oct 21, 2008
//
// moved here from isomorph_database.cpp 5/24/12
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

int isomorph::identify_solution_relaxed(int *set, int *transporter, 
	int f_implicit_fusion, int &orbit_no,
	int &f_failure_to_find_point, int verbose_level)
// returns the orbit number corresponding to 
// the canonical version of set and the extension.
// Calls trace_set and find_extension_easy.
// Returns FALSE if f_failure_to_find_point is TRUE after trace_set.
// Returns FALSE if find_extension_easy returns FALSE.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, id, id0, orbit, case_nb;
	int *canonical_set, *data;
	int *Elt;
	
	f_failure_to_find_point = FALSE;
	canonical_set = tmp_set1;
	data = tmp_set2;
	Elt = tmp_Elt1;
	
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::identify_solution_relaxed: ";
		cout << endl;
		//int_vec_print(cout, set, size);
		//cout << endl;
		//cout << "verbose_level=" << verbose_level << endl;
		}
	
	int_vec_copy(set, canonical_set, size);
#if 0
	for (i = 0; i < size; i++) {
		canonical_set[i] = set[i];
		}
#endif
	A->element_one(transporter, FALSE);

	while (TRUE) { 
		// this while loop is not needed
		
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::identify_solution_relaxed "
							"calling trace : " << endl;
			}
		case_nb = trace_set(canonical_set, transporter, 
			f_implicit_fusion, f_failure_to_find_point,
			verbose_level - 2);

		if (f_failure_to_find_point) {
			if (f_v) {
				cout << "iso_node " << iso_nodes
						<< " isomorph::identify_solution_relaxed "
						"after trace: trace_set returns "
						"f_failure_to_find_point" << endl;
				}
			return FALSE;
			}
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::identify_solution_relaxed "
					"after trace: ";
			print_node_local(level, case_nb);
			cout << endl;
			//cout << "case_nb = " << case_nb << " : ";
			cout << "canonical_set:" << endl;
			int_vec_print(cout, canonical_set, size);
			cout << endl;
			for (i = 0; i < size; i++) {
				cout << setw(5) << i << " : " << setw(6)
						<< canonical_set[i] << endl;
				}
			//cout << "transporter:" << endl;
			//gen->A->print(cout, transporter);
			////gen->A->print_as_permutation(cout, transporter);
			//cout << endl;
			}
		if (find_extension_easy(canonical_set, case_nb, id,
				verbose_level - 2)) {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " isomorph::identify_solution_relaxed "
						"after trace: ";
				print_node_local(level, case_nb);
				cout << " : ";
				cout << "solution is identified as id=" << id << endl;
				}
			orbit_representative(id, id0,
					orbit, Elt, verbose_level - 2);
			orbit_no = orbit;
	
			A->mult_apply_from_the_right(transporter, Elt);
			if (f_vv) {
				//cout << "transporter:" << endl;
				//gen->A->print(cout, transporter);
				////gen->A->print_as_permutation(cout, transporter);
				//cout << endl;
				}
			break;
			}
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::identify_solution_relaxed "
					"after trace: ";
			print_node_local(level, case_nb);
			cout << " : ";
			cout << "did not find extension" << endl;
			}
		return FALSE;
		//make_set_smaller(case_nb, canonical_set, transporter,
		//verbose_level - 2);
		//cnt++;
		}

	load_solution(id0, data);
	if (f_vv) {
		//cout << "iso_node " << iso_nodes
		//<< " isomorph::identify_solution_relaxed, checking" << endl;
		}
	if (!A->check_if_transporter_for_set(transporter,
			size, set, data, verbose_level - 2)) {
		cout << "isomorph::identify_solution_relaxed, "
				"check fails, stop" << endl;
		int_vec_print(cout, set, size);
		cout << endl;
		int_vec_print(cout, data, size);
		cout << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::identify_solution_relaxed "
				"after trace: ";
		print_node_local(level, case_nb);
		cout << " : ";
		cout << "id0 = " << id0 << " orbit=" << orbit << endl;
		}
	if (id0 != orbit_perm[orbit_fst[orbit]]) {
		cout << "id0 != orbit_perm[orbit_fst[orbit]]" << endl;
		cout << "id0=" << id0 << endl;
		cout << "orbit=" << orbit << endl;
		cout << "orbit_fst[orbit]=" << orbit_fst[orbit] << endl;
		cout << "orbit_perm[orbit_fst[orbit]]="
				<< orbit_perm[orbit_fst[orbit]] << endl;
		exit(1);
		}
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::identify_solution_relaxed "
				"after trace: ";
		print_node_local(level, case_nb);
		cout << " : ";
		cout << "solution is identified as id=" << id << endl;
		}
	
	return TRUE;
}


int isomorph::identify_solution(int *set,
	int *transporter,
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
// returns the orbit number corresponding to 
// the canonical version of set and the extension.
// Calls trace_set and find_extension_easy.
// If needed, calls make_set_smaller
// Called from identify_database_is_open
{
	int *canonical_set, *data;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int id, id0, orbit, case_nb, cnt = 0;
	int *Elt;
	
	canonical_set = tmp_set1;
	data = tmp_set2;
	Elt = tmp_Elt1;
	
	if (f_v) {
		cout << "iso_node " << iso_nodes
			<< " isomorph::identify_solution: ";
		cout << endl;
		//int_vec_print(cout, set, size);
		//cout << endl;
		//cout << "verbose_level=" << verbose_level << endl;
		}
	
	int_vec_copy(set, canonical_set, size);
#if 0
	for (i = 0; i < size; i++) {
		canonical_set[i] = set[i];
		}
#endif
	A->element_one(transporter, FALSE);

	while (TRUE) {
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::identify_solution "
					"calling trace, cnt = " << cnt << " : " << endl;
			}
		case_nb = trace_set(canonical_set, transporter, 
			f_implicit_fusion, f_failure_to_find_point, verbose_level - 2);
		if (f_failure_to_find_point) {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " isomorph::identify_solution "
						"trace returns f_failure_to_find_point" << endl;
				}
			return -1;
			}
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::identify_solution "
					"after trace: ";
			print_node_local(level, case_nb);
			cout << endl;
			//cout << "case_nb = " << case_nb << " : ";
			//int_vec_print(cout, canonical_set, size);
			//cout << endl;
			//cout << "transporter:" << endl;
			//gen->A->print(cout, transporter);
			////gen->A->print_as_permutation(cout, transporter);
			//cout << endl;
			}
		if (find_extension_easy(canonical_set, case_nb, id,
				verbose_level - 2)) {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " isomorph::identify_solution "
						"after trace: ";
				print_node_local(level, case_nb);
				cout << " : ";
				cout << "solution is identified as id=" << id;
				cout << " (with " << cnt << " iterations)" << endl;
				}
			orbit_representative(id, id0, orbit, Elt, verbose_level);
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " isomorph::identify_solution "
						"after trace: ";
				print_node_local(level, case_nb);
				cout << " : orbit_representative = " << id0 << endl;
				}
	
			A->mult_apply_from_the_right(transporter, Elt);
			if (f_vv) {
				//cout << "transporter:" << endl;
				//gen->A->print(cout, transporter);
				////gen->A->print_as_permutation(cout, transporter);
				//cout << endl;
				}
			break;
			}
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::identify_solution "
					"after trace: ";
			print_node_local(level, case_nb);
			cout << " : ";
			cout << "did not find extension, we are now trying to "
					"make the set smaller (iteration " << cnt
					<< ")" << endl;
			}
		make_set_smaller(case_nb, canonical_set, transporter,
				verbose_level - 2);
		cnt++;
		}

	load_solution(id0, data);
	if (f_vv) {
		//cout << "iso_node " << iso_nodes
		//<< " isomorph::identify_solution,
		// checking" << endl;
		}
	if (!A->check_if_transporter_for_set(transporter, size, set,
			data, verbose_level - 2)) {
		cout << "isomorph::identify_solution, "
				"check fails, stop" << endl;
		int_vec_print(cout, set, size);
		cout << endl;
		int_vec_print(cout, data, size);
		cout << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::identify_solution after trace: ";
		print_node_local(level, case_nb);
		cout << " : ";
		cout << "id0 = " << id0 << " orbit=" << orbit << endl;
		}
	if (id0 != orbit_perm[orbit_fst[orbit]]) {
		cout << "id0 != orbit_perm[orbit_fst[orbit]]" << endl;
		cout << "id0=" << id0 << endl;
		cout << "orbit=" << orbit << endl;
		cout << "orbit_fst[orbit]=" << orbit_fst[orbit] << endl;
		cout << "orbit_perm[orbit_fst[orbit]]="
				<< orbit_perm[orbit_fst[orbit]] << endl;
		exit(1);
		}
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::identify_solution after trace: ";
		print_node_local(level, case_nb);
		cout << " : ";
		cout << "solution is identified as id=" << id
				<< " belonging to orbit " << orbit
				<< " with representative " << id0;
		cout << " (" << cnt << " iterations)" << endl;
		}
	
	return orbit;
}

int isomorph::trace_set(
	int *canonical_set, int *transporter,
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
// returns the case number of the canonical set
// (local orbit number)
// Called from identify_solution and identify_solution_relaxed
// calls trace_set_recursion
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int n, case_nb;
	
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_set" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "depth_completed=" << depth_completed << endl;
		cout << "size=" << size << endl;
		cout << "level=" << level << endl;
		}
	n = trace_set_recursion(0 /* cur_level */, 0 /* cur_node_global */,  
		canonical_set, transporter, 
		f_implicit_fusion, f_failure_to_find_point, verbose_level - 1);
	
	if (f_failure_to_find_point) {
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::trace_set "
					"failure to find point" << endl;
			}
		return -1;
		}
	case_nb = n - gen->first_poset_orbit_node_at_level[level];
	
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_set the set traces to ";
		print_node_global(level, n);
		cout << endl;
		}
	if (f_vv) {
		cout << "iso_node " << iso_nodes
			<< " isomorph::trace_set transporter:" << endl;
#if 0
		gen->A->print(cout, transporter);
		//gen->A->print_as_permutation(cout, transporter);
		cout << endl;
#endif
		}
	
	if (case_nb < 0) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_set, case_nb < 0, "
						"case_nb = " << case_nb << endl;
		exit(1);
		}
	return case_nb;
}

void isomorph::make_set_smaller(int case_nb_local, 
	int *set, int *transporter, int verbose_level)
// Called from identify_solution.
// The goal is to produce a set that is lexicographically 
// smaller than the current starter.
// To do this, we find an element that is less than 
// the largest element in the current starter.
// There are two ways to find such an element.
// Either, the set already contains such an element, 
// or one can produce such an element by applying an element in the 
// stabilizer of the current starter.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int *image_set = make_set_smaller_set;
	int *Elt1 = make_set_smaller_Elt1;
	int *Elt2 = make_set_smaller_Elt2;
	int i, j, n, m, a, b;
	//int set1[1000];
	sorting Sorting;
	
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::make_set_smaller: " << endl;
		int_vec_print(cout, set, size);
		cout << endl;
		}
	nb_times_make_set_smaller_called++;
	n = gen->first_poset_orbit_node_at_level[level] + case_nb_local;
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " make_set_smaller_database case_nb_local = "
				<< case_nb_local << " n = " << n << endl;
		}

	vector_ge gens;
	longinteger_object go;


	load_strong_generators(level, case_nb_local /* cur_node */, 
		gens, go, verbose_level);


	a = set[level - 1];
	m = int_vec_minimum(set + level, size - level);
	if (m < a) {
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< "isomorph::make_set_smaller a = " << a
					<< " m = " << m << endl;
			}
		if (gen->f_starter) {
			Sorting.int_vec_heapsort(set + gen->starter_size,
					size - gen->starter_size);
			}
		else {
			Sorting.int_vec_heapsort(set, size);
			}
		if (f_vv) {
			cout << "iso_node " << iso_nodes
				<< "isomorph::make_set_smaller the reordered set is ";
			int_vec_print(cout, set, size);
			cout << endl;
			}
		return;
		}
	
	for (j = level; j < size; j++) {
		a = set[j];
		b = A->least_image_of_point(gens, a, Elt1, verbose_level - 1);
		if (b < m) {
			A->map_a_set_and_reorder(set, image_set, size, Elt1, 0);
			int_vec_copy(image_set, set, size);
#if 0
			for (ii = 0; ii < size; ii++) {
				set[ii] = image_set[ii];
				}
#endif
			A->element_mult(transporter, Elt1, Elt2, FALSE);
			A->element_move(Elt2, transporter, FALSE);
			if (f_vv) {
				cout << "iso_node " << iso_nodes
					<< "isomorph::make_set_smaller "
					"the set is made smaller: " << endl;
				int_vec_print(cout, set, size);
				cout << endl;
				}
			return;
			}
		}
	
	cout << "isomorph::make_set_smaller: "
			"error, something is wrong" << endl;
	cout << "isomorph::make_set_smaller no stabilizer element maps "
			"any element to something smaller" << endl;
	int_vec_print(cout, set, size);
	cout << endl;
	cout << "j : set[j] : least image" << endl;
	for (j = 0; j < size; j++) {
		a = set[j];
		b = A->least_image_of_point(gens, a, Elt1, verbose_level - 1);
		cout << setw(4) << j << " " << setw(4) << a << " " 
			<< setw(4) << b << " ";
		if (b < a) {
			cout << "smaller" << endl;
			}
		else {
			cout << endl;
			}
		}
	cout << "case_nb_local = " << case_nb_local << endl;
	cout << "iso_node = " << iso_nodes << endl;
	cout << "level = " << level << endl;
	cout << "m = " << m << endl;
	for (i = 0; i < gens.len; i++) {
		cout << "isomorph::make_set_smaller "
				"generator " << i << ":" << endl;
		A->element_print(gens.ith(i), cout);
		cout << endl;
		A->element_print_as_permutation(gens.ith(i), cout);
		cout << endl;
		}
		
	int f, l, id, c;
	int data[1000];
	
	f = solution_first[case_nb_local];
	l = solution_len[case_nb_local];
	cout << "f=" << f << " l=" << l << endl;
	for (i = 0; i < l; i++) {
		id = f + i;
		load_solution(id, data);
		Sorting.int_vec_heapsort(data + level, size - level);
		c = int_vec_compare(set + level, data + level, size - level);
		cout << setw(4) << id << " : compare = " << c << " : ";
		int_vec_print(cout, data, size);
		cout << endl;
		}
	exit(1);
}

int isomorph::trace_set_recursion(
	int cur_level, int cur_node_global,
	int *canonical_set, int *transporter, 
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
// returns the node in the generator that corresponds 
// to the canonical_set.
// Called from trace_set.
// Calls trace_next_point and handle_extension.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int pt, pt0, ret;
	sorting Sorting;
	
	f_failure_to_find_point = FALSE;
	if (f_v) {
		cout << "iso_node "
				<< iso_nodes
				<< " isomorph::trace_set_recursion ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		//int_vec_print(cout, canonical_set, size);
		cout << endl;
		}
	if (cur_level == 0 && gen->f_starter) {
		//int *cur_set = gen->set[0];
		//int *next_set = gen->set[0 + gen->starter_size];
		//int *cur_transporter = gen->transporter->ith(0);
		//int *next_transporter = gen->transporter->ith(0 + gen->starter_size);
		int *next_set;
		int *next_transporter;

		next_set = NEW_int(size);
		next_transporter = NEW_int(gen->Poset->A->elt_size_in_int);

		gen->root[0].trace_starter(gen, size, 
			canonical_set, next_set,
			transporter, next_transporter, 
			0 /*verbose_level */);

		int_vec_copy(next_set, canonical_set, size);
#if 0
		for (u = 0; u < size; u++) {
			canonical_set[u] = next_set[u];
			}
#endif

		if (f_vv) {
			cout << "iso_node "
					<< iso_nodes
					<< " isomorph::trace_set_recursion ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "after trace_starter" << endl;
			int_vec_print(cout, canonical_set, size);
			cout << endl;
			}

		gen->Poset->A->element_move(next_transporter, transporter, 0);
		FREE_int(next_set);
		FREE_int(next_transporter);
		if (f_v) {
			cout << "iso_node "
					<< iso_nodes << " isomorph::trace_set_recursion ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "after trace_starter, calling trace_set_recursion "
					"for node " << gen->starter_size << endl;
			}
		return trace_set_recursion(gen->starter_size, gen->starter_size, 
			canonical_set, transporter, 
			f_implicit_fusion, f_failure_to_find_point, verbose_level); 
		}
	pt = canonical_set[cur_level];
	if (f_vv) {
		cout << "iso_node "
				<< iso_nodes << " isomorph::trace_set_recursion ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "tracing point " << pt << endl;
		cout << "calling trace_next_point" << endl;
		}
	ret = trace_next_point(cur_level, cur_node_global, 
		canonical_set, transporter, f_implicit_fusion,
		f_failure_to_find_point, verbose_level - 2);
	if (f_vv) {
		cout << "iso_node "
				<< iso_nodes << " isomorph::trace_set_recursion ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "after tracing point " << pt << endl;
		}


	if (f_failure_to_find_point) {
		return -1;
		}

	
	if (!ret) {
		
		// we need to sort and restart the trace:

		if (f_vv) {
			cout << "iso_node "
					<< iso_nodes << " isomorph::trace_set_recursion ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "trace_next_point returns FALSE" << endl;
			}
		if (gen->f_starter) {
			Sorting.int_vec_heapsort(canonical_set + gen->starter_size,
					cur_level + 1 - gen->starter_size);
			}
		else {
			Sorting.int_vec_heapsort(canonical_set, cur_level + 1);
			}

		if (f_vv) {
			cout << "iso_node "
					<< iso_nodes << " isomorph::trace_set_recursion ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "restarting the trace" << endl;
			//int_vec_print(cout, canonical_set, cur_level + 1);
			//cout << endl;
			}
		
		if (gen->f_starter) {
			return trace_set_recursion(gen->starter_size,
					gen->starter_size, canonical_set,
				transporter, f_implicit_fusion, f_failure_to_find_point,
				verbose_level);
			}
		else {
			return trace_set_recursion(0, 0, canonical_set, 
				transporter, f_implicit_fusion, f_failure_to_find_point,
				verbose_level);
			}
		}
	pt0 = canonical_set[cur_level];
	if (f_vv) {
		cout << "iso_node "
				<< iso_nodes << " isomorph::trace_set_recursion ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "point " << pt << " has been mapped to "
				<< pt0 << ", calling handle_extension" << endl;
		//int_vec_print(cout, canonical_set, size);
		}
	ret = handle_extension(cur_level, cur_node_global, 
		canonical_set, transporter, 
		f_implicit_fusion, f_failure_to_find_point, verbose_level);

	if (f_failure_to_find_point) {
		return -1;
		}


	if (f_vv) {
		cout << "iso_node "
				<< iso_nodes << " isomorph::trace_set_recursion ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "after handle_extension" << endl;
		//cout << "transporter:" << endl;
		//gen->A->print(cout, transporter);
		//gen->A->print_as_permutation(cout, transporter);
		//cout << endl;
		}
	return ret;
}

int isomorph::trace_next_point(int cur_level,
	int cur_node_global,
	int *canonical_set, int *transporter, 
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
// Called from trace_set_recursion
// Calls ::trace_next_point_in_place
// and (possibly) trace_next_point_database
// Returns FALSE is the set becomes lexicographically smaller
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret;
	//int f_failure_to_find_point;
	

	f_failure_to_find_point = FALSE;

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_next_point ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << endl;
		}
	if (cur_level <= depth_completed) {
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::trace_next_point ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "cur_level <= depth_completed, using oracle" << endl;
			}
		
		poset_orbit_node *O = &gen->root[cur_node_global];
		ret = O->trace_next_point_in_place(gen, 
			cur_level, cur_node_global, size, 
			canonical_set, trace_set_recursion_tmp_set1,
			transporter, trace_set_recursion_Elt1, 
			f_implicit_fusion, f_failure_to_find_point, verbose_level - 2);
		if (f_failure_to_find_point) {
			cout << "isomorph::trace_next_point "
					"f_failure_to_find_point" << endl;
			cout << "iso_node " << iso_nodes
					<< " isomorph::trace_next_point ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "cur_level <= depth_completed, using oracle" << endl;
			return FALSE;
			}
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::trace_next_point after "
							"O->trace_next_point_in_place, "
							"return value ret = " << ret << endl;
			}
		}
	else {
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::trace_next_point ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "cur_level is not <= depth_completed, "
					"using database" << endl;
			}
		ret = trace_next_point_database(cur_level, cur_node_global, 
			canonical_set, transporter, verbose_level);
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::trace_next_point ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "after trace_next_point_database, "
					"return value ret = " << ret << endl;
			}
		}
	if (f_v && !ret) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_next_point ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "returning FALSE" << endl;
		}
	if (f_vv) {
		//cout << "iso_node " << iso_nodes
		// << " trace_next_point, transporter:" << endl;
		//gen->A->print(cout, transporter);
		//gen->A->print_as_permutation(cout, transporter);
		//cout << endl;
		}
	return ret;
}





int isomorph::trace_next_point_database(
	int cur_level, int cur_node_global,
	int *canonical_set, int *Elt_transporter,
	int verbose_level)
// Returns FALSE is the set becomes lexicographically smaller
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int cur_node_local, i;
	int set[1000];
	//char *elt;
	int *tmp_ELT;
	int pt, image;

	if (f_v) {
		cout << "iso_node " << iso_nodes
			<< " isomorph::trace_next_point_database ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << endl;
		}


	prepare_database_access(cur_level, verbose_level);
	//elt = NEW_char(gen->A->coded_elt_size_in_char);
	tmp_ELT = NEW_int(gen->Poset->A->elt_size_in_int);
	
	cur_node_local =
			cur_node_global -
			gen->first_poset_orbit_node_at_level[cur_level];
	DB_level->ith_object(cur_node_local,
			0/* btree_idx*/, *v,
			verbose_level - 2);
	
	if (f_vvv) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_next_point_database ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "v=" << *v << endl;
		}
	for (i = 0; i < cur_level; i++) {
		set[i] = v->s_ii(2 + i);
		}
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_next_point_database ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "set: ";
		int_vec_print(cout, set, cur_level);
		cout << endl;
		}
	int nb_strong_generators;
	int pos, ref;
	pos = 2 + cur_level;
	nb_strong_generators = v->s_ii(pos++);
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_next_point_database ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "nb_strong_generators=" << nb_strong_generators << endl;
		}
	if (nb_strong_generators == 0) {
		goto final_check;
		}
	pos = v->s_l() - 1;
	ref = v->s_ii(pos++);
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_next_point_database ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "ref = " << ref << endl;
		}

	{
	vector_ge gens;

	gens.init(gen->Poset->A, verbose_level - 2);
	gens.allocate(nb_strong_generators, verbose_level - 2);

	//fseek(fp_ge, ref * gen->Poset->A->coded_elt_size_in_char, SEEK_SET);
	fp_ge->seekg(ref * gen->Poset->A->coded_elt_size_in_char, ios::beg);
	for (i = 0; i < nb_strong_generators; i++) {
		gen->Poset->A->element_read_file_fp(gens.ith(i),
				*fp_ge, 0/* verbose_level*/);
		}
	
	
	pt = canonical_set[cur_level];

	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_next_point_database ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "computing least_image_of_point "
				"for point " << pt << endl;
		}
	image = gen->Poset->A2->least_image_of_point(gens,
			pt, tmp_ELT, verbose_level - 3);
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_next_point_database ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "least_image_of_point for point "
				<< pt << " returns " << image << endl;
		}
	if (f_vvv) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_next_point_database ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "image = " << image << endl;
		}
	}
	
	if (image == pt) {
		goto final_check;
		}

	if (FALSE /*f_vvv*/) {
		cout << "applying:" << endl;
		gen->Poset->A->element_print(tmp_ELT, cout);
		cout << endl;
		}
		
	for (i = cur_level; i < size; i++) {
		canonical_set[i] = gen->Poset->A2->element_image_of(
				canonical_set[i], tmp_ELT, FALSE);
		}

	//gen->A->map_a_set(gen->set[lvl],
	//gen->set[lvl + 1], len + 1, cosetrep, 0);

	//int_vec_sort(len, gen->set[lvl + 1]);
		// we keep the last point extra

	gen->Poset->A->mult_apply_from_the_right(
			Elt_transporter, tmp_ELT);

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::trace_next_point_database ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		//cout << "iso_node " << iso_nodes
		//<< " trace_next_point_database: the set becomes ";
		//int_vec_print(cout, canonical_set, size);
		cout << "done" << endl;
		}

final_check:

	FREE_int(tmp_ELT);
	//FREE_char(elt);

#if 1
	// this is needed if implicit fusion nodes are used
	
	if (cur_level > 0 &&
			canonical_set[cur_level] < canonical_set[cur_level - 1]) {
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::trace_next_point_database ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "the set becomes lexicographically less, "
					"we return FALSE" << endl;
			}
		return FALSE;
		}
#endif
	return TRUE;
		
}

int isomorph::handle_extension(
	int cur_level, int cur_node_global,
	int *canonical_set, int *Elt_transporter, 
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int pt0, next_node_global;
	
	pt0 = canonical_set[cur_level];
	
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::handle_extension node ";
		print_node_global(cur_level, cur_node_global);
		cout << " taking care of point " << pt0 << endl;
		}
	
	if (cur_level <= depth_completed) {
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension calling "
							"handle_extension_oracle" << endl;
			}
		next_node_global = handle_extension_oracle(cur_level,
			cur_node_global,
			canonical_set, Elt_transporter, f_implicit_fusion,
			f_failure_to_find_point, verbose_level);
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension "
					" handle_extension_oracle returns "
						<< next_node_global << endl;
			}
		}
	else {
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension "
							"calling handle_extension_database" << endl;
			}
		next_node_global = handle_extension_database(
			cur_level, cur_node_global,
			canonical_set, Elt_transporter,
			f_implicit_fusion,
			f_failure_to_find_point,
			verbose_level);
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension "
							"handle_extension_database returns "
					<< next_node_global << endl;
			}
		}
	return next_node_global;
}

int isomorph::handle_extension_database(int cur_level,
	int cur_node_global,
	int *canonical_set, int *Elt_transporter, 
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, pt0, pt, /*orbit_len,*/ t = 0, d = 0;
	int pos, ref, nb_strong_generators, nb_extensions;
	int nb_fusion, next_node_global;
	sorting Sorting;


	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::handle_extension_database "
				" node ";
		print_node_global(cur_level, cur_node_global);
		cout << endl;
		}
	pt0 = canonical_set[cur_level];		
	pos = 2 + cur_level;
	nb_strong_generators = v->s_ii(pos++);
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::handle_extension_database "
				" node ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "nb_strong_generators = " << nb_strong_generators << endl;
		}
	if (nb_strong_generators) {
		pos += gen->Poset->A->base_len();
		}
	nb_extensions = v->s_ii(pos++);
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::handle_extension_database "
				" node ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "nb_extensions = " << nb_extensions << endl;
		}
	nb_fusion = 0;
	for (i = 0; i < nb_extensions; i++) {
		pt = v->s_ii(pos++);
		//orbit_len = v->s_ii(pos++);
		t = v->s_ii(pos++);
		d = v->s_ii(pos++);
		if (pt == pt0) {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " isomorph::handle_extension_database "
						" node ";
				print_node_global(cur_level, cur_node_global);
				cout << " : ";
				cout << "we are in extension " << i << endl;
				}
			break;
			}
		if (t == 2) {
			nb_fusion++;
			}
		}
	if (i == nb_extensions) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::handle_extension_database "
				" node ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "did not find point " << pt0
				<< " in the list of extensions" << endl;
		exit(1);
		}
	pos = v->s_l() - 1;
	ref = v->s_ii(pos++);
	if (f_vvv) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::handle_extension_database "
				" node ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "handle_extension_database ref = " << ref << endl;
		}
	ref += nb_strong_generators;
	ref += nb_fusion;
	if (t == 1) {
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension_database "
					" node ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			//int_vec_print(cout, canonical_set, size);
			cout << "point has been mapped to " << pt0
					<< ", next node is node " << d << endl;
			}
		if (cur_level + 1 == level) {
			return d;
			}
		else {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " isomorph::handle_extension_database "
						" node ";
				print_node_global(cur_level, cur_node_global);
				cout << " : ";
				cout << "calling trace_set_recursion for level "
						<< cur_level + 1 << " and node " << d << endl;
				}
			return trace_set_recursion(cur_level + 1, d,
				canonical_set,
				Elt_transporter, f_implicit_fusion,
				f_failure_to_find_point,
				verbose_level - 1);
			}
		}
	else if (t == 2) {
		// fusion node		
		apply_isomorphism_database(
			cur_level, cur_node_global,
			i, canonical_set, Elt_transporter, ref,  
			verbose_level - 1);

		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension_database "
					" node ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "current_extension = " << i 
				<< " : fusion element has been applied";
			//int_vec_print(cout, canonical_set, size);
			cout << endl;
			}


		Sorting.int_vec_heapsort(canonical_set, cur_level + 1);
		
		if (FALSE) {
			cout << "iso_node " << iso_nodes
				<< " handle_extension_database cur_level = " << cur_level
				<< " cur_node_global = " << cur_node_global << " : " 
				<< " current_extension = " << i 
				<< " : after sorting the initial part : ";
			int_vec_print(cout, canonical_set, size);
			cout << endl;
			}
		next_node_global = gen->find_poset_orbit_node_for_set(cur_level + 1,
				canonical_set, FALSE /*f_tolerant*/, 0);
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension_database "
					" node ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "next_node=" ;
			print_node_global(cur_level + 1, next_node_global);
			}
		return next_node_global;

#if 0
		// we need to restart the trace:
		return trace_set_recursion(0, 0, canonical_set, 
			Elt_transporter, 
			f_implicit_fusion, verbose_level - 1);
#endif
		}
	else {
		cout << "handle_extension_database: illegal value of t" << endl;
		exit(1);
		}
}

int isomorph::handle_extension_oracle(int cur_level,
	int cur_node_global,
	int *canonical_set, int *Elt_transporter, 
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
// Returns next_node_global at level cur_level + 1.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	poset_orbit_node *O = &gen->root[cur_node_global];
	int pt0, current_extension, t, d, next_node_global;
	sorting Sorting;
	
	f_failure_to_find_point = FALSE;
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::handle_extension_oracle "
				" node ";
		print_node_global(cur_level, cur_node_global);
		cout << endl;
		}
	pt0 = canonical_set[cur_level];
	current_extension = O->find_extension_from_point(gen, pt0, FALSE);
	if (current_extension < 0) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::handle_extension_oracle "
				" node ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "did not find point pt0=" << pt0 << endl;
		f_failure_to_find_point = TRUE;
		return -1;
		}
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::handle_extension_oracle "
				" node ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "current_extension = " << current_extension << endl;
		}
	t = O->E[current_extension].type;
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " isomorph::handle_extension_oracle "
				" node ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "type = " << t << endl;
		}
	if (t == 1) {
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension_oracle "
					" node ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "extension node" << endl;
			}
		// extension node
		d = O->E[current_extension].data;
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension_oracle "
					" node ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			//int_vec_print(cout, canonical_set, size);
			cout << " point has been mapped to " << pt0
					<< ", next node is node " << d << endl;
			}
		if (cur_level + 1 == level) {
			return d;
			}
		else {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " isomorph::handle_extension_oracle "
						" node ";
				print_node_global(cur_level, cur_node_global);
				cout << " : ";
				cout << "calling trace_set_recursion for level "
						<< cur_level + 1 << " and node " << d << endl;
				}
			return trace_set_recursion(cur_level + 1, d,
				canonical_set,
				Elt_transporter, f_implicit_fusion,
				f_failure_to_find_point,
				verbose_level);
			}

		}
	else if (t == 2) {
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension_oracle "
					" node ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "fusion node" << endl;
			}
		// fusion node		
		apply_isomorphism_oracle(cur_level, cur_node_global, 
			current_extension, canonical_set, Elt_transporter, 
			verbose_level - 2);

		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension_oracle "
					" node ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "fusion element has been applied";
			int_vec_print(cout, canonical_set, size);
			cout << endl;
			}

		Sorting.int_vec_heapsort(canonical_set, cur_level + 1);
		
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension_oracle "
					" node ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << " current_extension = " << current_extension 
				<< " : after sorting the initial part ";
			int_vec_print(cout, canonical_set, size);
			cout << endl;
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension_oracle "
					<< " before gen->find_oracle_node_for_set" << endl;
			}
		next_node_global = gen->find_poset_orbit_node_for_set(
				cur_level + 1, canonical_set, FALSE /*f_tolerant*/,
				verbose_level);
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " isomorph::handle_extension_oracle "
					" node ";
			print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "next_node=" ;
			print_node_global(cur_level + 1, next_node_global);
			}
#if 0
		return next_node_global;
#endif

		// added 7/28/2012 A Betten:
		if (cur_level + 1 == level) {
			return next_node_global;
			}
		else {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " isomorph::handle_extension_oracle "
						" node ";
				print_node_global(cur_level, cur_node_global);
				cout << " : ";
				cout << "calling trace_set_recursion for level "
						<< cur_level + 1 << " and node " << next_node_global << endl;
				}
			return trace_set_recursion(cur_level + 1,
				next_node_global, canonical_set,
				Elt_transporter, f_implicit_fusion,
				f_failure_to_find_point, verbose_level);
			}



#if 0
		if (f_starter) {
			}
		else {
			// we need to restart the trace:
			return trace_set_recursion(0, 0, canonical_set, 
				Elt_transporter, 
				f_implicit_fusion, verbose_level);
			}
#endif


		}
	cout << "iso_node " << iso_nodes
			<< " isomorph::handle_extension_oracle "
			" node ";
	print_node_global(cur_level, cur_node_global);
	cout << " : ";
	cout << "current_extension = " << current_extension << " : ";
	cout << "unknown type " << t << endl;
	exit(1);
}

void isomorph::apply_isomorphism_database(
	int cur_level, int cur_node_global,
	int current_extension, int *canonical_set,
	int *Elt_transporter, int ref,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sorting Sorting;
	
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< "isomorph::apply_isomorphism_database "
				<< " node ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "ref = " << ref << endl;
		}
	
	//fseek(fp_ge, ref * gen->Poset->A->coded_elt_size_in_char, SEEK_SET);
	fp_ge->seekg(ref * gen->Poset->A->coded_elt_size_in_char, ios::beg);
	gen->Poset->A->element_read_file_fp(
			gen->Elt1, *fp_ge, 0/* verbose_level*/);
	
	gen->Poset->A2->map_a_set(canonical_set,
			apply_fusion_tmp_set1, size, gen->Elt1, 0);

	//int_vec_heapsort(apply_fusion_tmp_set1, level);
	Sorting.int_vec_heapsort(apply_fusion_tmp_set1, cur_level + 1);

	gen->Poset->A->element_mult(Elt_transporter,
			gen->Elt1, apply_fusion_Elt1, FALSE);

	int_vec_copy(apply_fusion_tmp_set1, canonical_set, size);
	gen->Poset->A->element_move(apply_fusion_Elt1,
			Elt_transporter, FALSE);

}

void isomorph::apply_isomorphism_oracle(
	int cur_level, int cur_node_global,
	int current_extension, int *canonical_set, int *Elt_transporter, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	poset_orbit_node *O = &gen->root[cur_node_global];
	sorting Sorting;

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " apply_isomorphism_oracle node ";
		print_node_global(cur_level, cur_node_global);
		cout << " : " << endl;
		}
	
	gen->Poset->A->element_retrieve(
			O->E[current_extension].data,
			gen->Elt1, FALSE);
	
	gen->Poset->A2->map_a_set(canonical_set,
			apply_fusion_tmp_set1, size, gen->Elt1, 0);

	//int_vec_heapsort(apply_fusion_tmp_set1, level);
	Sorting.int_vec_heapsort(apply_fusion_tmp_set1, cur_level + 1);

	gen->Poset->A->element_mult(Elt_transporter,
			gen->Elt1, apply_fusion_Elt1, FALSE);

	int_vec_copy(apply_fusion_tmp_set1, canonical_set, size);
	gen->Poset->A->element_move(apply_fusion_Elt1,
			Elt_transporter, FALSE);

}

}}


