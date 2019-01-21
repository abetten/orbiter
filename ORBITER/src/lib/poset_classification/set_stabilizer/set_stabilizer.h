/*
 * set_stabilizer.h
 *
 *  Created on: Aug 9, 2018
 *      Author: Anton Betten
 *
 * started:  September 20, 2007
 * pulled out of snakesandladders.h: Aug 9, 2018
 */

namespace orbiter {


// #############################################################################
// set_stabilizer_compute.C:
// #############################################################################


//! wrapper to compute the set stabilizer with the class compute_stabilizer


class set_stabilizer_compute {

public:

	action *A;
	action *A2;

	int set_size;
	int *the_set;
	int *the_set_sorted;
	int *the_set_sorting_perm;
	int *the_set_sorting_perm_inv;

	poset_classification *gen;

	int overall_backtrack_nodes;

	set_stabilizer_compute();
	~set_stabilizer_compute();
	void init(poset *Poset,
		//action *A,
		int *set, int size, int verbose_level);
	void init_with_strong_generators(
		poset *Poset,
		//action *A, action *A0,
		//strong_generators *Strong_gens,
		int *set, int size, int verbose_level);
	void compute_set_stabilizer(int t0, int &nb_backtrack_nodes, 
		strong_generators *&Aut_gens, int verbose_level);
	void print_frequencies(int lvl, int *frequency, int nb_orbits);
	int handle_frequencies(int lvl, 
		int *frequency, int nb_orbits, int *isomorphism_type_of_subset, 
		int &counter, int n_choose_k, 
		strong_generators *&Aut_gens, int verbose_level);
	void print_interesting_subsets(int lvl, 
		int nb_interesting_subsets, int *interesting_subsets);
	void compute_frequencies(int level, 
		int *&frequency, int &nb_orbits, 
		int *&isomorphism_type_of_subset, 
		int &n_choose_k, int verbose_level);
};


// #############################################################################
// compute_stabilizer.C:
// #############################################################################

//! wrapper to compute the set stabilizer using the poset classification algorithm

class compute_stabilizer {

public:

	int set_size;
	int *the_set;

	action *A;
	action *A2;
	poset_classification *gen;

	action *A_on_the_set;
	
	sims *Stab;
	longinteger_object stab_order, n_e_w_stab_order;
	int nb_times_orbit_count_does_not_match_up;
	int backtrack_nodes_first_time;
	int backtrack_nodes_total_in_loop;

	int level;
	int interesting_orbit; // previously orb_idx

	int *interesting_subsets; // [nb_interesting_subsets]
	int nb_interesting_subsets;

	int first_at_level;
	int reduced_set_size; // = set_size - level


	// maintained by null1, allocate1, free1:
	int *reduced_set1; // [set_size]
	int *reduced_set2; // [set_size]
	int *reduced_set1_n_e_w_labels; // [set_size]
	int *reduced_set2_n_e_w_labels; // [set_size]
	int *canonical_set1; // [set_size]
	int *canonical_set2; // [set_size]
	int *elt1, *Elt1, *Elt1_inv, *n_e_w_automorphism, *Elt4;
	int *elt2, *Elt2;


	strong_generators *Strong_gens_G;
	group *G;
	longinteger_object go_G;

	schreier *Stab_orbits;
	int nb_orbits;
	int *orbit_count1; // [nb_orbits]
	int *orbit_count2; // [nb_orbits]

	int nb_interesting_orbits;
	int *interesting_orbits;
	int nb_interesting_points;
	int *interesting_points;
	int *interesting_orbit_first;
	int *interesting_orbit_len;
	int local_idx1, local_idx2;





	action *A_induced;
	longinteger_object induced_go, K_go;

	int *transporter_witness;
	int *transporter1;
	int *transporter2;
	int *T1, *T1v;
	int *T2;

	sims *Kernel_original;
	sims *K; // kernel for building up Stab



	sims *Aut;
	sims *Aut_original;
	longinteger_object ago;
	longinteger_object ago1;
	longinteger_object target_go;


	union_find_on_k_subsets *U;
	
	compute_stabilizer();
	~compute_stabilizer();
	void null();
	void freeself();
	void init(int *the_set, int set_size, poset_classification *gen,
		action *A, action *A2, 
		int level, int interesting_orbit, int frequency, 
		int *subset_ranks, int verbose_level);
	void init_U(int verbose_level);
	void compute_orbits(int verbose_level);
		// uses Strong_gens_G to compute orbits on points 
		// in action A2
	void restricted_action(int verbose_level);
	void main_loop(int verbose_level);
	void main_loop_handle_case(int cnt, int verbose_level);
	void map_the_first_set(int cnt, int verbose_level);
	void map_the_second_set(int cnt, int verbose_level);
	void update_stabilizer(int verbose_level);
	void add_automorphism(int verbose_level);
	void retrieve_automorphism(int verbose_level);
	void make_canonical_second_set(int verbose_level);
	int compute_second_reduced_set();
	int check_orbit_count();
	void print_orbit_count(int f_both);
	void null1();
	void allocate1();
	void free1();
};

}


