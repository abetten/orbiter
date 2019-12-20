// backtrack.cpp
//
// Anton Betten
// March 15, 2008

#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


#define AUTS_ALLOCATE_BLOCK_SIZE 100


namespace orbiter {
namespace group_actions {


typedef struct action_is_minimal_data action_is_minimal_data;

//! internal class for is_minimal backtracking used by class action

struct action_is_minimal_data {
	action *A;
	int backtrack_node;
	int size;
	long int *set;
	long int *the_set;
	int *choices; // [A.base_len * A.degree]
	int *nb_choices;
	int *current_choice;
	long int *witness;
	int *transporter_witness;
	int *coset_rep;
	partitionstack *Staborbits;
	int nb_auts;
	int nb_auts_allocated;
	int *aut_data; // [nb_auts_allocated * A->base_len]
	int first_moved;
	int f_automorphism_seen;
	int *is_minimal_base_point;  // [A->base_len]
		// is_minimal_base_point[i] = TRUE means that 
		// the i-th base point b_i is the first moved point 
		// under the (i-1)-th group in the stabilizer chain.
};

void action_is_minimal_reallocate_aut_data(action_is_minimal_data &D);
int action_is_minimal_recursion(action_is_minimal_data *D,
		int depth, int verbose_level);

void action_is_minimal_reallocate_aut_data(action_is_minimal_data &D)
{
	int nb_auts_allocated2;
	int *aut_data2;
	int i;
	
	nb_auts_allocated2 = D.nb_auts_allocated + AUTS_ALLOCATE_BLOCK_SIZE;
	aut_data2 = NEW_int(nb_auts_allocated2 * D.A->base_len());
	for (i = 0; i < D.nb_auts * D.A->base_len(); i++) {
		aut_data2[i] = D.aut_data[i];
		}
	FREE_int(D.aut_data);
	D.aut_data = aut_data2;
	D.nb_auts_allocated = nb_auts_allocated2;
}

int action_is_minimal_recursion(action_is_minimal_data *D,
		int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int transversal_length, base_point, image_point;
	long int *current_set;
	long int *next_set;
	int i, idx, coset, cmp, ret, a;
	action *A;
	sorting Sorting;
	
	D->backtrack_node++;
	A = D->A;
	current_set = D->the_set + depth * D->size;
	next_set = D->the_set + (depth + 1) * D->size;
	if (f_vv || ((D->backtrack_node % 10000) == 0)) {
		cout << "action_is_minimal_recursion NODE "
				<< D->backtrack_node << " depth = " << depth << " : ";
		for (i = 0; i < depth; i++) {
			cout << i << ":" << D->current_choice[i]
				<< "/" << D->nb_choices[i] << " ";
			}
		cout << endl;
		}
	if (f_vvv) {
		lint_vec_print(cout, current_set, D->size);
		cout << endl;
		}
	if (depth == A->base_len()) {
		cmp = lint_vec_compare(current_set, D->the_set, D->size);
		if (cmp == 0) {
			D->f_automorphism_seen = TRUE;
			if (D->nb_auts == D->nb_auts_allocated) {
				action_is_minimal_reallocate_aut_data(*D);
				}
			for (i = 0; i < A->base_len(); i++) {
				a = D->current_choice[i];
				image_point = D->choices[i * A->degree + a];
				coset = A->Sims->get_orbit_inv(i, image_point);
				D->aut_data[D->nb_auts * A->base_len() + i] = coset;
				}
#if 0
			for (i = 0; i < A->base_len; i++) {
				a = D->current_choice[i];
				if (a) {
					D->first_moved = i;
					break;
					}
				}
#endif
			if (f_v) {
				cout << "automorphism " << D->nb_auts
					<< " first_moved = " << D->first_moved
					<< " choice: ";
				int_vec_print(cout, D->current_choice, A->base_len());
				cout << " points: ";
				int_vec_print(cout, D->aut_data +
						D->nb_auts * A->base_len(), A->base_len());
				cout << endl;
				}
			for (i = 0; i < A->base_len(); i++) {
				coset = D->aut_data[D->nb_auts * A->base_len() + i];
				A->Sims->path[i] = coset;

					//Sims->orbit_inv[i][aut_data[h * base_len + i]];
				}
			A->Sims->element_from_path_inv(D->transporter_witness);
			if (!A->check_if_transporter_for_set(
					D->transporter_witness, D->size,
				D->the_set, D->the_set, verbose_level)) {
				cout << "action_is_minimal_recursion: "
						"error while checking automorphism" << endl;
				exit(1);
				}
			if (f_v && D->first_moved < A->base_len()) {
				int *Elt, a1, a2;
				Elt = NEW_int(A->elt_size_in_int);
				A->invert(D->transporter_witness, Elt);
				i = D->first_moved;
				a = A->Stabilizer_chain->base_i(i);
				a1 = A->image_of(D->transporter_witness, a);
				a2 = A->image_of(Elt, a);
				cout << setw(3) << i << " : " 
					<< setw(3) << a1 << " -> " 
					<< setw(3) << a << " -> " 
					<< setw(3) << a2 << endl;
				FREE_int(Elt);
				}
			D->nb_auts++;
			}
		return TRUE;
		}
	
	transversal_length = A->transversal_length_i(depth);
	base_point = A->base_i(depth);
	if (f_vv) {
		cout << "depth = " << depth << " : ";
		cout << "transversal_length=" << transversal_length
				<< " base_point=" << base_point << endl;
		}
	if (f_vvv) {
		lint_vec_print(cout, current_set, D->size);
		cout << endl;
		}
	D->nb_choices[depth] = 0;
	for (i = 0; i < transversal_length; i++) {
		int f_accept = FALSE;
		int base_point;
		
		base_point = A->orbit_ij(depth, 0);
		image_point = A->orbit_ij(depth, i);
		if (D->is_minimal_base_point[depth] &&
				Sorting.lint_vec_search(current_set, D->size, base_point, idx, 0)) {
			if (Sorting.lint_vec_search(current_set, D->size, image_point, idx, 0)) {
				f_accept = TRUE;
				}
			}
		else {
			f_accept = TRUE;
			}
		if (f_accept) {
			D->choices[depth * A->degree +
					   D->nb_choices[depth]] = image_point;
			D->nb_choices[depth]++;
			if (f_vvv) {
				cout << "coset " << i << " image_point = "
						<< image_point << " added, "
						"D->nb_choices[depth]="
						<< D->nb_choices[depth] << endl;
				}
			}
		else {
			if (f_vvv) {
				cout << "coset " << i << " image_point = "
						<< image_point << " skipped, "
						"D->nb_choices[depth]="
						<< D->nb_choices[depth] << endl;
				}
			}
		}
	if (f_vv) {
		cout << "choice set of size " << D->nb_choices[depth] << " : ";
		int_vec_print(cout, D->choices + depth * A->degree,
				D->nb_choices[depth]);
		cout << endl;
		}
	
	for (D->current_choice[depth] = 0;
			D->current_choice[depth] < D->nb_choices[depth];
			D->current_choice[depth]++) {
		if (D->current_choice[depth]) {
			if (D->first_moved < depth && D->f_automorphism_seen) {
				if (f_vv) {
					cout << "returning from level " << depth 
						<< " because current_choice = "
						<< D->current_choice[depth]
						<< " and first_moved = " << D->first_moved << endl;
					}
				return TRUE;
				}
			if (D->first_moved > depth) {
				D->first_moved = depth;
				}
			if (depth == D->first_moved) {
				D->f_automorphism_seen = FALSE;
				}
			}
		image_point = D->choices[depth * A->degree +
								 D->current_choice[depth]];
		coset = A->Sims->get_orbit_inv(depth, image_point);
		if (f_vv) {
			cout << "depth = " << depth;
			cout << " choice " << D->current_choice[depth]
				<< " image_point=" << image_point
				<< " coset=" << coset << endl;
			}
		if (f_vvv) {
			lint_vec_print(cout, current_set, D->size);
			cout << endl;
			}
		A->Sims->coset_rep_inv(D->coset_rep, depth, coset, 0 /*verbose_level*/);
		// result is in A->Sims->cosetrep
		if (FALSE /*f_vvv*/) {
			cout << "cosetrep:" << endl;
			A->element_print(D->coset_rep, cout);
			cout << endl;
			A->element_print_as_permutation(D->coset_rep, cout);
			cout << endl;
			}
		A->map_a_set(current_set, next_set, D->size, D->coset_rep, 0);
		if (FALSE /*f_vv*/) {
			cout << "image set: ";
			lint_vec_print(cout, next_set, D->size);
			cout << endl;
			}
		Sorting.lint_vec_quicksort_increasingly(next_set, D->size);
		if (f_vv) {
			cout << "sorted image : ";
			lint_vec_print(cout, next_set, D->size);
			cout << endl;
			}
		cmp = lint_vec_compare(next_set, D->the_set, D->size);
		if (f_vv) {
			cout << "compare yields " << cmp;
			cout << endl;
			}
		
		if (f_vv) {
			cout << "NODE " << setw(5) << D->backtrack_node << " depth ";
			cout << setw(2) << depth << " current_choice "
					<< D->current_choice[depth];
			cout << " image_point=" << image_point
					<< " coset=" << coset << endl;
			//cout << " next_set = ";
			//int_vec_print(cout, next_set, D->size);
			//cout << endl;
			}
		
		if (cmp < 0) {
			if (f_v) {
				cout << "the current set is less than the original set, "
						"so the original set was not minimal" << endl;
				lint_vec_print(cout, next_set, D->size);
				cout << endl;
				}
			lint_vec_copy(next_set, D->witness, D->size);
			int k, choice;
			int_vec_zero(A->Sims->path, A->base_len());
			for (k = 0; k <= depth; k++) {
				choice = D->choices[k * A->degree + D->current_choice[k]];
				A->Sims->path[k] = A->Sims->get_orbit_inv(k, choice);
				}
			A->Sims->element_from_path_inv(D->transporter_witness);
			
			if (!A->check_if_transporter_for_set(
					D->transporter_witness, D->size,
				D->the_set, D->witness, verbose_level)) {
				cout << "action_is_minimal_recursion: error in "
						"check_if_transporter_for_set for witness" << endl;
				exit(1);
				}

			return FALSE;
			}
		ret = action_is_minimal_recursion(D, depth + 1, verbose_level);
		if (f_vv) {
			cout << "depth = " << depth << " finished" << endl;
			//int_vec_print(cout, current_set, D->size);
			//cout << " : choice " << D->current_choice[depth]
			// << " finished, return value = " << ret << endl;
			}
		if (!ret) {
			return FALSE;
			}
		}
	
	if (f_vv) {
		cout << "depth = " << depth << " finished" << endl;
		//int_vec_print(cout, current_set, D->size);
		//cout << endl;
		}
	return TRUE;
}

int action::is_minimal(int size, long int *set,
	int &backtrack_level, int verbose_level)
{
	long int *witness;
	int *transporter_witness;
	int ret, backtrack_nodes;
	int f_get_automorphism_group = FALSE;
	sims Aut;
	
	witness = NEW_lint(size);
	transporter_witness = NEW_int(elt_size_in_int);
	
	ret = is_minimal_witness(size, set, backtrack_level,
		witness, transporter_witness, backtrack_nodes, 
		f_get_automorphism_group, Aut, 
		verbose_level);
	
	FREE_lint(witness);
	FREE_int(transporter_witness);
	return ret;
}

void action::make_canonical(int size, long int *set,
	long int *canonical_set, int *transporter,
	int &total_backtrack_nodes, 
	int f_get_automorphism_group, sims *Aut,
	int verbose_level)
{
	//verbose_level += 10;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt1, *Elt2, *Elt3;
	long int *set1;
	long int *set2;
	int backtrack_level, backtrack_nodes, cnt = 0;
	//int f_get_automorphism_group = TRUE;
	//sims Aut;
	
	total_backtrack_nodes = 0;
	if (f_v) {
		cout << "action::make_canonical" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	if (f_vv) {
		cout << "the input set is ";
		lint_vec_print(cout, set, size);
		cout << endl;
		}

	longinteger_object go;
	Sims->group_order(go);
	if (f_v) {
		cout << "action::make_canonical group order = " << go << endl;
		}
	
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	Elt3 = NEW_int(elt_size_in_int);
	set1 = NEW_lint(size);
	set2 = NEW_lint(size);
	
	lint_vec_copy(set, set1, size);
	element_one(Elt1, FALSE);
	
	while (TRUE) {
		cnt++;
		//if (cnt == 4) verbose_level += 10;
		if (f_v) {
			cout << "action::make_canonical iteration "
						<< cnt << " before is_minimal_witness" << endl;
			}
		if (is_minimal_witness(/*default_action,*/ size, set1, 
			backtrack_level, set2, Elt2, 
			backtrack_nodes, 
			f_get_automorphism_group, *Aut,
			0 /*verbose_level - 1*/)) {
			total_backtrack_nodes += backtrack_nodes;
			if (f_v) {
				cout << "action::make_canonical: is minimal, "
						"after iteration " << cnt << " with "
					<< backtrack_nodes << " backtrack nodes, total:"
					<< total_backtrack_nodes << endl;
				}
			break;
			}
		//if (cnt == 4) verbose_level -= 10;
		total_backtrack_nodes += backtrack_nodes;
		if (f_v) {
			cout << "action::make_canonical finished iteration " << cnt;
			if (f_vv) {
				lint_vec_print(cout, set2, size);
				}
			cout << " with " 
				<< backtrack_nodes << " backtrack nodes, total:"
				<< total_backtrack_nodes << endl;
			}
		lint_vec_copy(set2, set1, size);
		element_mult(Elt1, Elt2, Elt3, 0);
		element_move(Elt3, Elt1, 0);
		
		}
	lint_vec_copy(set1, canonical_set, size);
	element_move(Elt1, transporter, FALSE);
	
	if (!check_if_transporter_for_set(transporter,
			size, set, canonical_set, verbose_level - 3)) {
		exit(1);
		}
	if (f_v) {
		cout << "action::make_canonical succeeds in " << cnt
				<< " iterations, total_backtrack_nodes="
				<< total_backtrack_nodes << endl;
		longinteger_object go;
		Aut->group_order(go);
		cout << "the automorphism group has order " << go << endl;
		}
	if (f_vv) {
		cout << "the canonical set is ";
		lint_vec_print(cout, canonical_set, size);
		cout << endl;
		}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_lint(set1);
	FREE_lint(set2);
	//exit(1);
}

int action::is_minimal_witness(int size, long int *set,
	int &backtrack_level, long int *witness, int *transporter_witness,
	int &backtrack_nodes, 
	int f_get_automorphism_group, sims &Aut,
	int verbose_level)
{
	action A;
	action_is_minimal_data D;
	int ret = TRUE;
	int i;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 4);
	int f_vvv = (verbose_level >= 5);
	int f_vvvv = (verbose_level >= 7);
	sorting Sorting;

	if (f_vv) {
		cout << "action::is_minimal_witness" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	if (f_vvv) {
		cout << "the input set is ";
		lint_vec_print(cout, set, size);
		cout << endl;
		}
	
	backtrack_nodes = 0;
	backtrack_level = size - 1;
	//cout << "action::is_minimal_witness backtrack_level = "
	//"size - 1 = " << backtrack_level << endl;

	if (f_vvvv) {
		cout << "current base is ";
		print_base();
		cout << "doing base change" << endl;
		}
	A.base_change(this, size, set, MINIMUM(1, verbose_level - 4));
	//A.eliminate_redundant_base_points(verbose_level - 4); 
	// !!! A Betten July 10, 2014
	if (f_vv) {
		cout << "base changed to ";
		A.print_base();
		}
	
	//cout << "action::is_minimal_witness testing membership" << endl;
	
	//A.Sims->test_if_subgroup(Sims, 2);
	
	//A.Sims->print_all_group_elements();
	
#if 0
	if (f_vvvv) {
		cout << "action " << A.label << endl;
		cout << "we have the following strong generators:" << endl;
		A.strong_generators->print_as_permutation(cout);
		cout << "and Sims:" << endl;
		A.Sims->print_basic_orbits();
		}
#endif
	
#if 0
	if (f_vvv) {
		A.Sims->print_generators();
		A.Sims->print_generators_as_permutations();
		A.Sims->print_basic_orbits();
		}
#endif

	D.A = &A;
	D.size = size;
	D.set = set;


	D.nb_auts = 0;
	D.nb_auts_allocated = AUTS_ALLOCATE_BLOCK_SIZE;
	D.aut_data = NEW_int(D.nb_auts_allocated * A.base_len());
	D.first_moved = A.base_len();
	D.f_automorphism_seen = FALSE;
	
	if (f_vv) {
		cout << "computing stabilizer orbits" << endl;
		}
	
	A.compute_stabilizer_orbits(D.Staborbits, verbose_level - 4);
	
	if (f_vv) {
		cout << "computing stabilizer orbits finished" << endl;
		}

	D.the_set = NEW_lint((A.base_len() + 1) * size);
	lint_vec_copy(set, D.the_set, size);
	Sorting.lint_vec_quicksort_increasingly(D.the_set, size);
	
	D.backtrack_node = 0;
	D.choices = NEW_int(A.base_len() * A.degree);
	D.nb_choices = NEW_int(A.base_len());
	D.current_choice = NEW_int(A.base_len());
	D.witness = witness;
	D.coset_rep = NEW_int(A.elt_size_in_int);
	D.transporter_witness = transporter_witness;
	D.is_minimal_base_point = NEW_int(A.base_len());

	for (i = 0; i < A.base_len(); i++) {
		partitionstack *S;
		int b, c, f, l, j, p;
		
		b = A.base_i(i);
		if (i == size)
			break;
#if 0
		if (b != set[i]) {
			cout << i << "-th base point is " << b 
				<< " different from i-th point in the set "
				<< set[i] << endl;
			exit(1);
			}
#endif
		S = &D.Staborbits[i];
		c = S->cellNumber[S->invPointList[b]];
		f = S->startCell[c];
		l = S->cellSize[c];
		for (j = 0; j < l; j++) {
			p = S->pointList[f + j];
			if (p < b) {
				if (f_vv) {
					cout << "action::is_minimal_witness level " << i 
						<< ", orbit of base_point " << b 
						<< " contains " << p 
						<< " which is a smaller point" << endl;
					}
				if (FALSE) {
					cout << "partitionstack:" << endl;
					S->print(cout);
					S->print_raw();
					}
				int k;
				int_vec_zero(A.Sims->path, A.base_len());
#if 0
				for (k = 0; k < A.base_len; k++) {
					A.Sims->path[k] = 0;
					}
#endif
				A.Sims->path[i] = A.orbit_inv_ij(i, p);
				A.Sims->element_from_path(transporter_witness, 0);


				for (k = 0; k < size; k++) {
					if (b == set[k]) {
						break;
						}
					}
				if (k == size) {
					//cout << "action::is_minimal_witness "
					// "did not find base point" << endl;
					//exit(1);
					}
				backtrack_level = k;
				//cout << "action::is_minimal_witness backtrack_level "
				//"= k = " << backtrack_level << endl;
				ret = FALSE;
				goto finish;
				}
			}
		}
	// now we compute is_minimal_base_point array:
	for (i = 0; i < A.base_len(); i++) {
		int j, b, c, l;
		partitionstack *S;
		S = &D.Staborbits[i];
		b = A.base_i(i);
		for (j = 0; j < b; j++) {
			c = S->cellNumber[S->invPointList[j]];
			l = S->cellSize[c];
			if (l > 1) {
				break;
				}
			}
		if (j < b) {
			D.is_minimal_base_point[i] = FALSE;
			}
		else {
			D.is_minimal_base_point[i] = TRUE;
			}
		}
	if (f_v) {
		cout << "action::is_minimal_witness: D.is_minimal_base_point=";
		int_vec_print(cout, D.is_minimal_base_point, A.base_len());
		cout << endl;
		}
	
	if (f_vv) {
		cout << "action::is_minimal_witness calling "
				"action_is_minimal_recursion" << endl;
		}
	ret = action_is_minimal_recursion(&D,
			0 /* depth */, verbose_level /* -3 */);
	if (f_vv) {
		cout << "action::is_minimal_witness "
				"action_is_minimal_recursion returns " << ret << endl;
		}
	backtrack_nodes = D.backtrack_node;
finish:
	if (!ret) {
		if (f_vv) {
			cout << "computing witness" << endl;
			}
		for (i = 0; i < size; i++) {
			witness[i] = A.image_of(transporter_witness, set[i]);
			}
		//int_vec_sort(size, witness);
		Sorting.lint_vec_heapsort(witness, size);
		}
	

	if (ret && f_get_automorphism_group) {
		if (f_vv) {
			int j, /*image_point,*/ coset;
			
			cout << "automorphism generators:" << endl;
			for (i = 0; i < D.nb_auts; i++) {
				cout << setw(3) << i << " : (";
				for (j = 0; j < base_len(); j++) {
					coset = D.aut_data[i * base_len() + j];
					cout << coset;
					//image_point = Sims->orbit[i][coset];
					//cout << image_point;
					if (j < base_len() - 1) {
						cout << ", ";
						}
					}
				cout << ")" << endl;
				//int_vec_print(cout, D.aut_data + i * base_len, base_len);
				}
			}
		sims Aut2, K;
		longinteger_object go, go2;
		
		if (f_vv) {
			cout << "building up automorphism group" << endl;
			}
		A.build_up_automorphism_group_from_aut_data(
				D.nb_auts, D.aut_data,
			Aut2, verbose_level - 3);
		Aut2.group_order(go2);
		if (f_v) {
			cout << "automorphism group in changed base "
					"has order " << go2 << endl;
			}
		
		Aut.init(this, verbose_level - 2);
		Aut.init_trivial_group(verbose_level - 1);
		K.init(this, verbose_level - 2);
		K.init_trivial_group(verbose_level - 1);
		
		
		Aut.build_up_group_random_process(&K, &Aut2, go2, 
			FALSE /* f_override_choose_next_base_point */,
			NULL, 
			verbose_level - 4);	
		//Aut.build_up_group_random_process_no_kernel(&Aut2, verbose_level);
		Aut.group_order(go);
		if (f_v) {
			cout << "automorphism group has order " << go << endl;
			}
		}
	
	if (FALSE) {
		cout << "freeing memory" << endl;
		}
	
	FREE_int(D.aut_data);
	delete [] D.Staborbits;
	FREE_lint(D.the_set);
	FREE_int(D.choices);
	FREE_int(D.nb_choices);
	FREE_int(D.current_choice);
	FREE_int(D.is_minimal_base_point);
	FREE_int(D.coset_rep);

	return ret;
}

}}



