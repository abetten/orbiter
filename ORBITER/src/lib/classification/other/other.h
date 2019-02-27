/*
 * other.h
 *
 *  Created on: Aug 9, 2018
 *      Author: Anton Betten
 *
 * started:  September 20, 2007
 * pulled out of snakesandladders.h: Aug 9, 2018
 */

namespace orbiter {
namespace classification {


// #############################################################################
// snakes_and_ladders_global.C:
// #############################################################################

void read_orbit_rep_and_candidates_from_files_and_process(action *A, 
	char *prefix, 
	int level, int orbit_at_level, int level_of_candidates_file, 
	void (*early_test_func_callback)(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		void *data, int verbose_level), 
	void *early_test_func_callback_data, 
	int *&starter,
	int &starter_sz,
	sims *&Stab,
	strong_generators *&Strong_gens, 
	int *&candidates,
	int &nb_candidates,
	int &nb_cases, 
	int verbose_level);
void read_candidates_for_one_orbit_from_file(char *prefix,
		int level, int orbit_at_level, int level_of_candidates_file,
		int *S,
		void (*early_test_func_callback)(int *S, int len,
			int *candidates, int nb_candidates,
			int *good_candidates, int &nb_good_candidates,
			void *data, int verbose_level),
		void *early_test_func_callback_data,
		int *&candidates,
		int &nb_candidates,
		int verbose_level);
void read_orbit_rep_and_candidates_from_files(action *A, char *prefix, 
	int level, int orbit_at_level, int level_of_candidates_file, 
	int *&starter,
	int &starter_sz,
	sims *&Stab,
	strong_generators *&Strong_gens, 
	int *&candidates,
	int &nb_candidates,
	int &nb_cases, 
	int verbose_level);
int find_orbit_index_in_data_file(const char *prefix,
		int level_of_candidates_file, int *starter,
		int verbose_level);
void compute_orbits_on_subsets(poset_classification *&gen,
	int target_depth,
	const char *prefix, 
	int f_W, int f_w,
	poset *Poset,
	//action *A, action *A2,
	//strong_generators *Strong_gens,
#if 0
	void (*early_test_func_callback)(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		void *data, int verbose_level),
	void *early_test_func_data, 
	int (*candidate_incremental_check_func)(int len, int *S, 
		void *data, int verbose_level), 
	void *candidate_incremental_check_data, 
#endif
	int verbose_level);
void orbits_on_k_sets(
	poset *Poset,
	int k, int *&orbit_reps, int &nb_orbits, int verbose_level);
poset_classification *orbits_on_k_sets_compute(
	poset *Poset,
	int k, int verbose_level);
void print_extension_type(std::ostream &ost, int t);
const char *trace_result_as_text(trace_result r);
int trace_result_is_no_result(trace_result r);
void wedge_product_export_magma(poset_classification *Gen, int n, int q,
	int vector_space_dimension, int level, int verbose_level);


}}




