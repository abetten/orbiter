/*
 * other.h
 *
 *  Created on: Aug 9, 2018
 *      Author: Anton Betten
 *
 * started:  September 20, 2007
 * pulled out of snakesandladders.h: Aug 9, 2018
 */

// #############################################################################
// snakes_and_ladders_global.C:
// #############################################################################

void read_orbit_rep_and_candidates_from_files_and_process(action *A, 
	BYTE *prefix, 
	INT level, INT orbit_at_level, INT level_of_candidates_file, 
	void (*early_test_func_callback)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level), 
	void *early_test_func_callback_data, 
	INT *&starter,
	INT &starter_sz,
	sims *&Stab,
	strong_generators *&Strong_gens, 
	INT *&candidates,
	INT &nb_candidates,
	INT &nb_cases, 
	INT verbose_level);
void read_candidates_for_one_orbit_from_file(BYTE *prefix,
		INT level, INT orbit_at_level, INT level_of_candidates_file,
		INT *S,
		void (*early_test_func_callback)(INT *S, INT len,
			INT *candidates, INT nb_candidates,
			INT *good_candidates, INT &nb_good_candidates,
			void *data, INT verbose_level),
		void *early_test_func_callback_data,
		INT *&candidates,
		INT &nb_candidates,
		INT verbose_level);
void read_orbit_rep_and_candidates_from_files(action *A, BYTE *prefix, 
	INT level, INT orbit_at_level, INT level_of_candidates_file, 
	INT *&starter,
	INT &starter_sz,
	sims *&Stab,
	strong_generators *&Strong_gens, 
	INT *&candidates,
	INT &nb_candidates,
	INT &nb_cases, 
	INT verbose_level);
INT find_orbit_index_in_data_file(const BYTE *prefix,
		INT level_of_candidates_file, INT *starter,
		INT verbose_level);
void compute_orbits_on_subsets(generator *&gen, 
	INT target_depth,
	const BYTE *prefix, 
	INT f_W, INT f_w,
	action *A, action *A2, 
	strong_generators *Strong_gens, 
	void (*early_test_func_callback)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level),
	void *early_test_func_data, 
	INT (*candidate_incremental_check_func)(INT len, INT *S, 
		void *data, INT verbose_level), 
	void *candidate_incremental_check_data, 
	INT verbose_level);
void orbits_on_k_sets(action *A1, action *A2, 
	strong_generators *Strong_gens, 
	INT k, INT *&orbit_reps, INT &nb_orbits, INT verbose_level);
generator *orbits_on_k_sets_compute(action *A1, action *A2, 
	strong_generators *Strong_gens, 
	INT k, INT verbose_level);
void print_extension_type(ostream &ost, INT t);
const BYTE *trace_result_as_text(trace_result r);
INT trace_result_is_no_result(trace_result r);
void wedge_product_export_magma(generator *Gen, INT n, INT q, 
	INT vector_space_dimension, INT level, INT verbose_level);





