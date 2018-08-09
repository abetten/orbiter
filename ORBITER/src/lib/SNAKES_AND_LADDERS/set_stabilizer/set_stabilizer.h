/*
 * set_stabilizer.h
 *
 *  Created on: Aug 9, 2018
 *      Author: Anton Betten
 *
 * started:  September 20, 2007
 * pulled out of snakesandladders.h: Aug 9, 2018
 */

// #############################################################################
// set_stabilizer_compute.C:
// #############################################################################

class set_stabilizer_compute {

public:

	action *A;
	action *A2;

	INT set_size;
	INT *the_set;
	INT *the_set_sorted;
	INT *the_set_sorting_perm;
	INT *the_set_sorting_perm_inv;

	generator *gen;

	INT overall_backtrack_nodes;

	set_stabilizer_compute();
	~set_stabilizer_compute();
	void init(action *A, INT *set, INT size, INT verbose_level);
	void init_with_strong_generators(action *A, action *A0, 
		strong_generators *Strong_gens, 
		INT *set, INT size, INT verbose_level);
	void compute_set_stabilizer(INT t0, INT &nb_backtrack_nodes, 
		strong_generators *&Aut_gens, INT verbose_level);
	void print_frequencies(INT lvl, INT *frequency, INT nb_orbits);
	INT handle_frequencies(INT lvl, 
		INT *frequency, INT nb_orbits, INT *isomorphism_type_of_subset, 
		INT &counter, INT n_choose_k, 
		strong_generators *&Aut_gens, INT verbose_level);
	void print_interesting_subsets(INT lvl, 
		INT nb_interesting_subsets, INT *interesting_subsets);
	void compute_frequencies(INT level, 
		INT *&frequency, INT &nb_orbits, 
		INT *&isomorphism_type_of_subset, 
		INT &n_choose_k, INT verbose_level);
};


// #############################################################################
// compute_stabilizer.C:
// #############################################################################

class compute_stabilizer {

public:

	INT set_size;
	INT *the_set;

	action *A;
	action *A2;
	generator *gen;

	action *A_on_the_set;
	
	sims *Stab;
	longinteger_object stab_order, new_stab_order;
	INT nb_times_orbit_count_does_not_match_up;
	INT backtrack_nodes_first_time;
	INT backtrack_nodes_total_in_loop;

	INT level;
	INT interesting_orbit; // previously orb_idx

	INT *interesting_subsets; // [nb_interesting_subsets]
	INT nb_interesting_subsets;

	INT first_at_level;
	INT reduced_set_size; // = set_size - level


	// maintained by null1, allocate1, free1:
	INT *reduced_set1; // [set_size]
	INT *reduced_set2; // [set_size]
	INT *reduced_set1_new_labels; // [set_size]
	INT *reduced_set2_new_labels; // [set_size]
	INT *canonical_set1; // [set_size]
	INT *canonical_set2; // [set_size]
	INT *elt1, *Elt1, *Elt1_inv, *new_automorphism, *Elt4;
	INT *elt2, *Elt2;


	strong_generators *Strong_gens_G;
	group *G;
	longinteger_object go_G;

	schreier *Stab_orbits;
	INT nb_orbits;
	INT *orbit_count1; // [nb_orbits]
	INT *orbit_count2; // [nb_orbits]

	INT nb_interesting_orbits;
	INT *interesting_orbits;
	INT nb_interesting_points;
	INT *interesting_points;
	INT *interesting_orbit_first;
	INT *interesting_orbit_len;
	INT local_idx1, local_idx2;





	action *A_induced;
	longinteger_object induced_go, K_go;

	INT *transporter_witness;
	INT *transporter1;
	INT *transporter2;
	INT *T1, *T1v;
	INT *T2;

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
	void init(INT *the_set, INT set_size, generator *gen, 
		action *A, action *A2, 
		INT level, INT interesting_orbit, INT frequency, 
		INT *subset_ranks, INT verbose_level);
	void init_U(INT verbose_level);
	void compute_orbits(INT verbose_level);
		// uses Strong_gens_G to compute orbits on points 
		// in action A2
	void restricted_action(INT verbose_level);
	void main_loop(INT verbose_level);
	void main_loop_handle_case(INT cnt, INT verbose_level);
	void map_the_first_set(INT cnt, INT verbose_level);
	void map_the_second_set(INT cnt, INT verbose_level);
	void update_stabilizer(INT verbose_level);
	void add_automorphism(INT verbose_level);
	void retrieve_automorphism(INT verbose_level);
	void make_canonical_second_set(INT verbose_level);
	INT compute_second_reduced_set();
	INT check_orbit_count();
	void print_orbit_count(INT f_both);
	void null1();
	void allocate1();
	void free1();
};



