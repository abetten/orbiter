/*
 * set_stabilizer.h
 *
 *  Created on: Jun 27, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_GROUP_ACTIONS_SET_STABILIZER_SET_STABILIZER_H_
#define SRC_LIB_GROUP_ACTIONS_SET_STABILIZER_SET_STABILIZER_H_

namespace orbiter {
namespace layer4_classification {
namespace set_stabilizer {



// #############################################################################
// compute_stabilizer.cpp
// #############################################################################

//! to compute the set-stabilizer

class compute_stabilizer {

public:


	substructure_stats_and_selection *SubSt;

	actions::action *A_on_the_set;
		// only used to print the induced action on the set
		// of the set stabilizer

	groups::sims *Stab; // the stabilizer of the original set


	ring_theory::longinteger_object stab_order, new_stab_order;
	int nb_times_orbit_count_does_not_match_up;
	int backtrack_nodes_first_time;
	int backtrack_nodes_total_in_loop;

	stabilizer_orbits_and_types *Stab_orbits;







	actions::action *A_induced;
		// the action on Stab_orbits->interesting_points[]

	ring_theory::longinteger_object induced_go, K_go;

	int *transporter_witness;
	int *transporter1;
	int *transporter2;
	int *T1, *T1v;
	int *T2;

	groups::sims *Kernel_original;
	groups::sims *K; // kernel for building up Stab



	groups::sims *Aut;
	groups::sims *Aut_original;
	ring_theory::longinteger_object ago;
	ring_theory::longinteger_object ago1;
	ring_theory::longinteger_object target_go;


	//union_find_on_k_subsets *U;


	long int *Canonical_form_input;
		// [nb_interesting_subsets_reduced * reduced_set_size]
	long int *Canonical_forms;
		// [nb_interesting_subsets_reduced * reduced_set_size]
	int *Canonical_form_transporter;
		// [nb_interesting_subsets_reduced * A_induced->elt_size_in_int]

	int nb_interesting_subsets_rr;
	long int *interesting_subsets_rr;


	compute_stabilizer();
	~compute_stabilizer();

	void init(
			substructure_stats_and_selection *SubSt,
			long int *canonical_pts,
			int verbose_level);
	void compute_automorphism_group(
			int verbose_level);
	void compute_automorphism_group_handle_case(
			int cnt2, int verbose_level);
	void setup_stabilizer(
			groups::sims *Stab0, int verbose_level);
	void restricted_action_on_interesting_points(
			int verbose_level);
	void compute_canonical_form(
			int verbose_level);
	void compute_canonical_form_handle_case(
			int cnt, int verbose_level);
	void compute_canonical_set(
			long int *set_in, long int *set_out, int sz,
			int *transporter, int verbose_level);
	void compute_canonical_set_and_group(
			long int *set_in, long int *set_out, int sz,
			int *transporter,
			groups::sims *&stab, int verbose_level);
	void update_stabilizer(
			int verbose_level);
	void add_automorphism(
			int verbose_level);
	void retrieve_automorphism(
			int verbose_level);
	void make_canonical_second_set(
			int verbose_level);
	void report(
			std::ostream &ost);
	void print_canonical_sets();

};






// #############################################################################
// stabilizer_orbits_and_types.cpp
// #############################################################################




//! orbits of the stabilizer of the substructure and orbit types



class stabilizer_orbits_and_types {

public:
	compute_stabilizer *CS;

	groups::strong_generators *selected_set_stab_gens;
	groups::sims *selected_set_stab;


	int reduced_set_size; // = set_size - level




	long int *reduced_set1; // [set_size]
	long int *reduced_set2; // [set_size]
	long int *reduced_set1_new_labels; // [set_size]
	long int *reduced_set2_new_labels; // [set_size]
	long int *canonical_set1; // [set_size]
	long int *canonical_set2; // [set_size]

	int *elt1, *Elt1, *Elt1_inv, *new_automorphism, *Elt4;
	int *elt2, *Elt2;
	int *transporter0; // = elt1 * elt2

	ring_theory::longinteger_object go_G;

	groups::schreier *Schreier;
	int nb_orbits;
	int *orbit_count1; // [nb_orbits]
	int *orbit_count2; // [nb_orbits]


	int minimal_orbit_pattern_idx;
	int nb_interesting_subsets_reduced;
	long int *interesting_subsets_reduced;

	int *Orbit_patterns; // [nb_interesting_subsets * nb_orbits]


	int *orbit_to_interesting_orbit; // [nb_orbits]

	int nb_interesting_orbits;
	int *interesting_orbits; // [nb_interesting_orbits]

	int nb_interesting_points; // sum of orbit length of interesting orbits
	long int *interesting_points; // [nb_interesting_points]
	// Note: Interesting points are sorted within each orbit.
	// Otherwise, it would not be possible to compute a canonical form.

	int *interesting_orbit_first; // [nb_interesting_orbits]
	int *interesting_orbit_len; // [nb_interesting_orbits]

	int local_idx1, local_idx2;



	stabilizer_orbits_and_types();
	~stabilizer_orbits_and_types();
	void init(
			compute_stabilizer *CS,
			int verbose_level);
	void compute_stabilizer_orbits_and_find_minimal_pattern(
			int verbose_level);
	// uses selected_set_stab_gens to compute orbits on points in action A2
	void save_interesting_subsets_reduced(
			int stage, int verbose_level);
	void find_orbit_pattern(
			int cnt, int *transp, int verbose_level);
	// computes transporter to transp
	void find_interesting_orbits(
			int verbose_level);
	void compute_local_labels(
			long int *set_in, long int *set_out, int sz,
			int verbose_level);
	void map_subset_and_compute_local_labels(
			int cnt, int verbose_level);
	void map_reduced_set_and_do_orbit_counting(
			int cnt,
			long int subset_idx, int *transporter, int verbose_level);
		// computes orbit_count1[]
	int check_orbit_count();
	void print_orbit_count(
			int f_both);
	void print_minimal_orbit_pattern();

};


// #############################################################################
// substructure_classifier.cpp
// #############################################################################



//! classification of substructures




class substructure_classifier {
public:


	std::string fname_base_out;
	int substructure_size;

	poset_classification::poset_classification *PC;
	poset_classification::poset_classification_control *Control;
	actions::action *A;
	actions::action *A2;
	poset_classification::poset_with_group_action *Poset;
	int nb_orbits;


	substructure_classifier();
	~substructure_classifier();
	void classify_substructures(
			std::string &fname_base_out,
			actions::action *A,
			actions::action *A2,
			groups::strong_generators *gens,
			int substructure_size,
			int verbose_level);
	void set_stabilizer_in_any_space(
			actions::action *A, actions::action *A2,
			groups::strong_generators *Strong_gens,
			int intermediate_subset_size,
			std::string &fname_mask, int nb, std::string &column_label,
			std::string &fname_out,
			int verbose_level);
	void set_stabilizer_of_set(
			std::string &fname_out,
			int cnt, int nb, int row,
			long int *pts,
			int nb_pts,
			long int *canonical_pts,
			int verbose_level);
	void handle_orbit(
			substructure_stats_and_selection *SubSt,
			long int *canonical_pts,
			int *transporter_to_canonical_form,
			groups::strong_generators *&Gens_stabilizer_original_set,
			int verbose_level);


};





// #############################################################################
// substructure_stats_and_selection.cpp
// #############################################################################



//! analyzing the substructures of a given set




class substructure_stats_and_selection {
public:

	std::string fname_case_out;

	substructure_classifier *SubC;

	long int *Pts;
	int nb_pts;


	// computed by SubC->PC->trace_all_k_subsets_and_compute_frequencies:
	int nCk;
	int *isotype; // [nCk]
	int *orbit_frequencies; // [nb_orbits]
	int nb_orbits;
	data_structures::tally *T;
	// for orbit_frequencies[]


	data_structures::set_of_sets *SoS;
	int *types;
	int nb_types;
	int selected_type;
	int selected_orbit;
	int selected_frequency;

	long int *interesting_subsets;
		// [selected_frequency]
	int nb_interesting_subsets;
		// interesting_subsets are the lvl-subsets of the given set
		// which are of the chosen type.
		// There is nb_interesting_subsets of them.

	groups::strong_generators *gens;
		// generators for the selected canonical subset
	//int *transporter_to_canonical_form;
	//strong_generators *Gens_stabilizer_original_set;


	substructure_stats_and_selection();
	~substructure_stats_and_selection();
	void init(
			std::string &fname_case_out,
			substructure_classifier *SubC,
			long int *Pts,
			int nb_pts,
			int verbose_level);

};




}}}





#endif /* SRC_LIB_GROUP_ACTIONS_SET_STABILIZER_SET_STABILIZER_H_ */
