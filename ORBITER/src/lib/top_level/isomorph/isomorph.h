// isomorph.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09

// #############################################################################
// isomorph.C
// isomorph_testing.C
// isomorph_database.C
// isomorph_trace.C
// isomorph_files.C
// #############################################################################

//! hybrid algorithm to classify combinatorial bjects


class isomorph {
public:
	INT size;
	INT level;
	INT f_use_database_for_starter;
	INT depth_completed;
	INT f_use_implicit_fusion;
	

	char prefix[500];
	char prefix_invariants[500];
	char prefix_tex[500];

	char fname_staborbits[500];
	char fname_case_len[500];
	char fname_statistics[500];
	char fname_hash_and_datref[500];
	char fname_db1[500];
	char fname_db2[500];
	char fname_db3[500];
	char fname_db4[500];
	char fname_db5[500];
	char fname_db_level[500];
	char fname_db_level_idx1[500];
	char fname_db_level_idx2[500];
	char fname_db_level_ge[500];
	
	char event_out_fname[1000];
	char fname_orbits_of_stabilizer_csv[1000];

	INT nb_starter; 
		// the number of orbits at 'level', 
		// previously called nb_cases
	
	INT N;
		// the number of solutions, 
		// computed in init_cases_from_file
		// or read from file in read_case_len
	

	
	INT *solution_first;
		// [nb_starter + 1] the beginning of solutions 
		// belonging to a given starter
		// previously called case_first
	INT *solution_len;
		// [nb_starter + 1] the number of solutions 
		// belonging to a given starter
		// previously called case_len


		// solution_first and solution_len are initialized in 
		// isomorph_files.C:
		// isomorph::init_solutions
		// isomorph::count_solutions_from_clique_finder
		// isomorph::count_solutions

		// they are written to file in 
		// isomorph::write_solution_first_and_len()



	INT *starter_number;
		// [N]  starter_number[i] = j means that 
		// solution i belongs to starter j
		// previously called case_number





	// from the summary file (and only used in isomorph_files.C)
	INT *stats_nb_backtrack;
		// [nb_starter]
	INT *stats_nb_backtrack_decision;
		// [nb_starter]
	INT *stats_graph_size;
		// [nb_starter]
	INT *stats_time;
		// [nb_starter]

	
	action *A_base;
		// A Betten 3/18/2013
		// the action in which we represent the group
		// do not free
	action *A;
		// the action in which we act on the set
		// do not free


	poset_classification *gen;
		// do not free



	INT nb_orbits;
		// Number of flag orbits.
		// Overall number of orbits of stabilizers 
		// of starters on the solutions
		// computed in orbits_of_stabilizer, 
		// which in turn calls orbits_of_stabilizer_case 
		// for each starter
		// orbits_of_stabilizer_case takes the 
		// strong generators from the 
		// generator data structure
		// For computing the orbits, induced_action_on_sets
		// is used to establish an action on sets.

	INT orbit_no;
		// Used in isomorph_testing:
		// The flag orbit we are currently testing.
		// In the end, this will become the representative of a 
		// n e w isomorphism type

	INT *orbit_fst;
		// [nb_orbits + 1]
		// orbit_fst[i] is the beginning of solutions 
		// associated to the i-th flag orbit 
		// in the sorted list of solutions
		// allocated in isomorph::read_orbit_data()

	INT *orbit_len;
		// [nb_orbits]
		// orbit_len[i] is the length of the i-th flag orbit 
		// allocated in isomorph::read_orbit_data()


	INT *orbit_number;
		// [N]
		// orbit_number[i] is the flag orbit 
		// containing the i-th solution 
		// in the original labeling
		// allocated in isomorph::read_orbit_data()

 
	INT *orbit_perm;
		// [N]
		// orbit_perm[i] is the original label 
		// of the i-th solution in the ordered list
		// we often see id = orbit_perm[orbit_fst[orbit_no]];
		// this is the index of the first solution 
		// associated to flag orbit orbit_no, for instance
		// allocated in isomorph::read_orbit_data()

	INT *orbit_perm_inv;
		// [N]
		// orbit_perm_inv is the inverse of orbit_perm
		// allocated in isomorph::read_orbit_data()


	INT *schreier_vector; // [N]
		// allocated in isomorph::read_orbit_data()
	INT *schreier_prev; // [N]
		// allocated in isomorph::read_orbit_data()

	
	// added Dec 25, 2012:
	// computed in isomorph.C isomorph::orbits_of_stabilizer
	// these variables are not used ?? (Oct 30, 2014)
	// They are used, for instance in isomorph_testing.C 
	// isomorph::write_classification_graph (May 3, 2015)
	INT *starter_orbit_fst;
		// [nb_starter + 1]
		// the beginning of flag orbits 
		// belonging to a given starter
	INT *starter_nb_orbits;
		// [nb_starter]
		// the number of flag orbits belonging to a given starter




	representatives *Reps;

	INT iso_nodes;
	
	

	
	INT nb_open, nb_reps, nb_fused;
	
	// for iso_test:


	ofstream *fp_event_out;

	action *AA;
	action *AA_perm;
	action *AA_on_k_subsets;
	union_find *UF;
	vector_ge *gens_perm;
	
	INT subset_rank;
	INT *subset;
	INT *subset_witness;
	INT *rearranged_set;
	INT *rearranged_set_save;
	INT *canonical_set;
	INT *tmp_set;
	INT *Elt_transporter, *tmp_Elt, *Elt1, *transporter;

	INT cnt_minimal;
	INT NCK;
	longinteger_object stabilizer_group_order;

	INT stabilizer_nb_generators;
	INT **stabilizer_generators;
		// INT[stabilizer_nb_generators][size]


	INT *stabilizer_orbit;
	INT nb_is_minimal_called;
	INT nb_is_minimal;
	INT nb_sets_reached;
	

	// temporary data used in identify_solution
	INT f_tmp_data_has_been_allocated;
	INT *tmp_set1;
	INT *tmp_set2;
	INT *tmp_set3;
	INT *tmp_Elt1;
	INT *tmp_Elt2;
	INT *tmp_Elt3;
	// temporary data used in trace_set_recursion
	INT *trace_set_recursion_tmp_set1;
	INT *trace_set_recursion_Elt1;
	// temporary data used in trace_set_recursion
	INT *apply_fusion_tmp_set1;
	INT *apply_fusion_Elt1;
	// temporary data used in find_extension
	INT *find_extension_set1;
	// temporary data used in make_set_smaller
	INT *make_set_smaller_set;
	INT *make_set_smaller_Elt1;
	INT *make_set_smaller_Elt2;
	// temporary data used in orbit_representative
	INT *orbit_representative_Elt1;
	INT *orbit_representative_Elt2;
	// temporary data used in handle_automorphism
	INT *handle_automorphism_Elt1;


	// database access:
	database *D1, *D2;
	char fname_ge1[1000];
	char fname_ge2[1000];
	FILE *fp_ge1, *fp_ge2;
	Vector *v;
	database *DB_sol;
	INT *id_to_datref;
	INT *id_to_hash;
	INT *hash_vs_id_hash; // sorted
	INT *hash_vs_id_id;
	INT f_use_table_of_solutions;
	INT *table_of_solutions; // [N * size]

	// pointer only, do not free:
	database *DB_level;
	FILE *fp_ge;
	
	sims *stabilizer_recreated;
	

	void (*print_set_function)(isomorph *Iso, 
		INT iso_cnt, sims *Stab, schreier &Orb, 
		INT *data, void *print_set_data, INT verbose_level);
	void *print_set_data;
	

	// some statistics:
	INT nb_times_make_set_smaller_called;

	// isomorph.C:
	isomorph();
	void null();
	~isomorph();
	void free();
	void null_tmp_data();
	void allocate_tmp_data();
	void free_tmp_data();
	void init(const char *prefix, 
		action *A_base, action *A, poset_classification *gen,
		INT size, INT level, 
		INT f_use_database_for_starter, 
		INT f_implicit_fusion, INT verbose_level);
	void init_solution(INT verbose_level);
	void load_table_of_solutions(INT verbose_level);
	void init_starter_number(INT verbose_level);
	void list_solutions_by_starter();
	void list_solutions_by_orbit();
	void orbits_of_stabilizer(INT verbose_level);
	void orbits_of_stabilizer_case(INT the_case, 
		vector_ge &gens, INT verbose_level);
	void orbit_representative(INT i, INT &i0, 
		INT &orbit, INT *transporter, INT verbose_level);
		// slow because it calls load_strong_generators
	void test_orbit_representative(INT verbose_level);
	void test_identify_solution(INT verbose_level);
	void compute_stabilizer(sims *&Stab, INT verbose_level);
	void test_compute_stabilizer(INT verbose_level);
	void test_memory();
	void test_edges(INT verbose_level);
	INT test_edge(INT n1, INT *subset1, 
		INT *transporter, INT verbose_level);
	void read_data_files_for_starter(INT level, 
		const char *prefix, INT verbose_level);
		// Calls gen->read_level_file_binary 
		// for all levels i from 0 to level
		// Uses letter a files for i from 0 to level - 1
		// and letter b file for i = level.
		// If gen->f_starter is TRUE, 
		// we start from i = gen->starter_size instead.
		// Finally, it computes nb_starter.
	void compute_nb_starter(INT level, INT verbose_level);
	void print_node_local(INT level, INT node_local);
	void print_node_global(INT level, INT node_global);
	void test_hash(INT verbose_level);
	void compute_Ago_Ago_induced(longinteger_object *&Ago, 
		longinteger_object *&Ago_induced, INT verbose_level);
	void init_high_level(action *A, poset_classification *gen,
		INT size, char *prefix_classify, char *prefix, 
		INT level, INT verbose_level);
	


	// isomorph_testing.C:
	void iso_test_init(INT verbose_level);
	void iso_test_init2(INT verbose_level);
	void probe(INT flag_orbit, INT subset_rk, 
		INT f_implicit_fusion, INT verbose_level);
	void isomorph_testing(INT t0, INT f_play_back, 
		const char *play_back_file_name, 
		INT f_implicit_fusion, INT print_mod, INT verbose_level);
	void write_classification_matrix(INT verbose_level);
	void write_classification_graph(INT verbose_level);
	void decomposition_matrix(INT verbose_level);
	void compute_down_link(INT *&down_link, INT verbose_level);
	void do_iso_test(INT t0, sims *&Stab, 
		INT f_play_back, ifstream *play_back_file, 
		INT &f_eof, INT print_mod, 
		INT f_implicit_fusion, INT verbose_level);
	INT next_subset(INT t0, 
		INT &f_continue, sims *Stab, INT *data, 
		INT f_play_back, ifstream *play_back_file, INT &f_eof,
		INT verbose_level);
	void process_rearranged_set(sims *Stab, INT *data, 
		INT f_implicit_fusion, INT verbose_level);
	INT is_minimal(INT verbose_level);
	void stabilizer_action_exit();
	void stabilizer_action_init(INT verbose_level);
		// Computes the permutations of the set 
		// that are induced by the 
		// generators for the stabilizer in AA
	void stabilizer_action_add_generator(INT *Elt, INT verbose_level);
	void print_statistics_iso_test(INT t0, sims *Stab);
	INT identify(INT *set, INT f_implicit_fusion, 
		INT verbose_level);
	INT identify_database_is_open(INT *set, 
		INT f_implicit_fusion, INT verbose_level);
	void induced_action_on_set_basic(sims *S, 
		INT *set, INT verbose_level);
	void induced_action_on_set(sims *S, 
		INT *set, INT verbose_level);
	// Called by do_iso_test and print_isomorphism_types
	// Creates the induced action on the set from the given action.
	// The given action is gen->A2
	// The induced action is computed to AA
	// The set is in set[].
	// Allocates a n e w union_find data structure and initializes it
	// using the generators in S.
	// Calls action::induced_action_by_restriction()
	INT handle_automorphism(INT *set, 
		sims *Stab, INT *Elt, INT verbose_level);

	// isomorph_database.C:
	void setup_and_open_solution_database(INT verbose_level);
	void setup_and_create_solution_database(INT verbose_level);
	void close_solution_database(INT verbose_level);
	void setup_and_open_level_database(INT verbose_level);
	// Called from do_iso_test, identify and test_hash 
	// (Which are all in isomorph_testing.C)
	// Calls init_DB for D and D.open.
	// Calls init_DB_level for D1 and D2 and D1->open and D2->open.
	// Calls fopen for fp_ge1 and fp_ge2.
	void close_level_database(INT verbose_level);
	// Closes D1, D2, fp_ge1, fp_ge2.
	void prepare_database_access(INT cur_level, INT verbose_level);
	// sets D to be D1 or D2, depending on cur_level
	// Called from 
	// load_strong_generators
	// trace_next_point_database
	void init_DB_sol(INT verbose_level);
		// We assume that the starter is of size 5 and that 
		// fields 3-8 are the starter
	void add_solution_to_database(INT *data, 
		INT nb, INT id, INT no, 
		INT nb_solutions, INT h, UINT4 &datref, 
		INT print_mod, INT verbose_level);
	void load_solution(INT id, INT *data);
	void load_solution_by_btree(INT btree_idx, 
		INT idx, INT &id, INT *data);
	INT find_extension_easy(INT *set, 
		INT case_nb, INT &idx, INT verbose_level);
		// returns TRUE if found, FALSE otherwise
		// Called from identify_solution
		// Linear search through all solutions at a given starter.
		// calls load solution for each of the solutions 
		// stored with the case and compares the vectors.
	INT find_extension_search_interval(INT *set, 
		INT first, INT len, INT &idx, 
		INT f_btree_idx, INT btree_idx, 
		INT f_through_hash, INT verbose_level);
	INT find_extension_easy_old(INT *set, 
		INT case_nb, INT &idx, INT verbose_level);
	INT find_extension_easy_new(INT *set, 
		INT case_nb, INT &idx, INT verbose_level);
	INT open_database_and_identify_object(INT *set, 
		INT *transporter, 
		INT f_implicit_fusion, INT verbose_level);
	void init_DB_level(database &D, INT level, 
		INT verbose_level);
	void create_level_database(INT level, INT verbose_level);
	void load_strong_generators(INT cur_level, 
		INT cur_node_local, 
		vector_ge &gens, longinteger_object &go, 
		INT verbose_level);
		// Called from compute_stabilizer and 
		// from orbit_representative
	void load_strong_generators_oracle(INT cur_level, 
		INT cur_node_local, 
		vector_ge &gens, longinteger_object &go, 
		INT verbose_level);
	void load_strong_generators_database(INT cur_level, 
		INT cur_node_local, 
		vector_ge &gens, longinteger_object &go, 
		INT verbose_level);
		// Reads node cur_node_local (local index) 
		// from database D through btree 0
		// Reads generators from file fp_ge

	// isomorph_trace.C:
	INT identify_solution_relaxed(INT *set, INT *transporter, 
		INT f_implicit_fusion, INT &orbit_no, 
		INT &f_failure_to_find_point, 
		INT verbose_level);
	// returns the orbit number corresponding to 
	// the canonical version of set and the extension.
	// Calls trace_set and find_extension_easy.
	// Called from process_rearranged_set
	INT identify_solution(INT *set, INT *transporter, 
		INT f_implicit_fusion, INT &f_failure_to_find_point, 
		INT verbose_level);
		// returns the orbit number corresponding to 
		// the canonical version of set and the extension.
		// Calls trace_set and find_extension_easy.
		// If needed, calls make_set_smaller
	INT trace_set(INT *canonical_set, INT *transporter, 
		INT f_implicit_fusion, INT &f_failure_to_find_point, 
		INT verbose_level);
		// returns the case number of the canonical set 
		// (local orbit number)
		// Called from identify_solution and identify_solution_relaxed
		// calls trace_set_recursion
	void make_set_smaller(INT case_nb_local, 
		INT *set, INT *transporter, INT verbose_level);
		// Called from identify_solution.
		// The goal is to produce a set that is lexicographically 
		// smaller than the current starter.
		// To do this, we find an element that is less than 
		// the largest element in the current starter.
		// There are two ways to find such an element.
		// Either, the set already contains such an element, 
		// or one can produce such an element 
		// by applying an element in the 
		// stabilizer of the current starter.
	INT trace_set_recursion(INT cur_level, 
		INT cur_node_global, 
		INT *canonical_set, INT *transporter, 
		INT f_implicit_fusion, 
		INT &f_failure_to_find_point, INT verbose_level);
		// returns the node in the generator that corresponds 
		// to the canonical_set.
		// Called from trace_set.
		// Calls trace_next_point and handle_extension.
	INT trace_next_point(INT cur_level, 
		INT cur_node_global, 
		INT *canonical_set, INT *transporter, 
		INT f_implicit_fusion, 
		INT &f_failure_to_find_point, INT verbose_level);
		// Called from trace_set_recursion
		// Calls oracle::trace_next_point_in_place 
		// and (possibly) trace_next_point_database
		// Returns FALSE is the set becomes lexicographically smaller
	INT trace_next_point_database(INT cur_level, 
		INT cur_node_global, 
		INT *canonical_set, INT *Elt_transporter, 
		INT verbose_level);
		// Returns FALSE is the set becomes lexicographically smaller
	INT handle_extension(INT cur_level, 
		INT cur_node_global, 
		INT *canonical_set, INT *Elt_transporter, 
		INT f_implicit_fusion, 
		INT &f_failure_to_find_point, INT verbose_level);
	INT handle_extension_database(INT cur_level, 
		INT cur_node_global, 
		INT *canonical_set, INT *Elt_transporter, 
		INT f_implicit_fusion, 
		INT &f_failure_to_find_point, 
		INT verbose_level);
	INT handle_extension_oracle(INT cur_level, 
		INT cur_node_global, 
		INT *canonical_set, INT *Elt_transporter, 
		INT f_implicit_fusion, 
		INT &f_failure_to_find_point, 
		INT verbose_level);
	// Returns next_node_global at level cur_level + 1.
	void apply_fusion_element_database(INT cur_level, 
		INT cur_node_global, 
		INT current_extension, INT *canonical_set, 
		INT *Elt_transporter, INT ref, 
		INT verbose_level);
	void apply_fusion_element_oracle(INT cur_level, 
		INT cur_node_global, 
		INT current_extension, INT *canonical_set, 
		INT *Elt_transporter, 
		INT verbose_level);


	// isomorph_files.C:
	void init_solutions(INT **Solutions, 
		INT *Nb_sol, INT verbose_level);
	// Solutions[nb_starter], Nb_sol[nb_starter]
	void count_solutions_from_clique_finder_case_by_case(INT nb_files, 
		INT *list_of_cases, const char **fname, 
		INT verbose_level);
	void count_solutions_from_clique_finder(INT nb_files, 
		const char **fname, 
		INT verbose_level);
	void read_solutions_from_clique_finder_case_by_case(INT nb_files, 
		INT *list_of_cases, const char **fname, 
		INT verbose_level);
	void read_solutions_from_clique_finder(INT nb_files, 
		const char **fname, INT verbose_level);
	void add_solutions_to_database(INT *Solutions, 
		INT the_case, INT nb_solutions, INT nb_solutions_total, 
		INT print_mod, INT &no, 
		INT verbose_level);
	void build_up_database(INT nb_files, const char **fname, 
		INT f_has_final_test_function, 
		INT (*final_test_function)(INT *data, INT sz, 
			void *final_test_data, INT verbose_level),
		void *final_test_data, 
		INT verbose_level);
	void init_cases_from_file_modulus_and_build_up_database(
		INT modulus, INT level, 
		INT f_collated, INT base_split, 
		INT f_get_statistics, 
		INT f_has_final_test_function, 
		INT (*final_test_function)(INT *data, INT sz, 
			void *final_test_data, INT verbose_level),
		void *final_test_data, 
		INT verbose_level);
	void init_cases_from_file_mixed_modulus_and_build_up_database(
		INT nb_Mod, INT *Mod_r, INT *Mod_split, INT *Mod_base_split, 
		INT level, INT f_get_statistics, 
		INT f_has_final_test_function, 
		INT (*final_test_function)(INT *data, INT sz, 
			void *final_test_data, INT verbose_level),
		void *final_test_data, 
		INT verbose_level);
	void count_solutions(INT nb_files, 
		const char **fname, INT f_get_statistics, 
		INT f_has_final_test_function, 
		INT (*final_test_function)(INT *data, INT sz, 
			void *final_test_data, INT verbose_level),
		void *final_test_data, 
		INT verbose_level);
	void get_statistics(INT nb_files, const char **fname, 
		INT verbose_level);
	void write_statistics();
	void evaluate_statistics(INT verbose_level);
	void count_solutions2(INT nb_files, const char **fname, 
		INT &total_days, INT &total_hours, INT &total_minutes, 
		INT f_has_final_test_function, 
		INT (*final_test_function)(INT *data, INT sz, 
			void *final_test_data, INT verbose_level),
		void *final_test_data, 
		INT verbose_level);
	void write_solution_first_and_len(); // previously write_case_len
	void read_solution_first_and_len(); // previously read_case_len
	void write_starter_nb_orbits(INT verbose_level);
	void read_starter_nb_orbits(INT verbose_level);
	void write_hash_and_datref_file(INT verbose_level);
		// Writes the file 'fname_hash_and_datref' 
		// containing id_to_hash[] and id_to_datref[]
	void read_hash_and_datref_file(INT verbose_level);
		// Reads the file 'fname_hash_and_datref' 
		// containing id_to_hash[] and id_to_datref[]
		// Also initializes hash_vs_id_hash and hash_vs_id_id
		// Called from init_solution
	void write_orbit_data(INT verbose_level);
		// Writes the file 'fname_staborbits'
	void read_orbit_data(INT verbose_level);
		// Reads from the file 'fname_staborbits'
		// Reads nb_orbits, N, 
		// orbit_fst[nb_orbits + 1]
		// orbit_len[nb_orbits]
		// orbit_number[N]
		// orbit_perm[N]
		// schreier_vector[N]
		// schreier_prev[N]
		// and computed orbit_perm_inv[N]
	void print_isomorphism_types(INT f_select, 
		INT select_first, INT select_len, 
		INT verbose_level);
		// Calls print_set_function (if available)
	void induced_action_on_set_and_kernel(ostream &file, 
		action *A, sims *Stab, INT size, INT *set, 
		INT verbose_level);
	void handle_event_files(INT nb_event_files, 
		const char **event_file_name, INT verbose_level);
	void read_event_file(const char *event_file_name, 
		INT verbose_level);
	void skip_through_event_file(ifstream &f, 
		INT verbose_level);
	void skip_through_event_file1(ifstream &f, 
		INT case_no, INT orbit_no, INT verbose_level);
	void event_file_completed_cases(const char *event_file_name, 
		INT &nb_completed_cases, INT *completed_cases, 
		INT verbose_level);
	void event_file_read_case(const char *event_file_name, 
		INT case_no, INT verbose_level);
	void event_file_read_case1(ifstream &f, 
		INT case_no, INT verbose_level);
	INT next_subset_play_back(INT &subset_rank, 
		ifstream *play_back_file, 
		INT &f_eof, INT verbose_level);
	void read_everything_including_classification(
		const char *prefix_classify, INT verbose_level);


};


// #############################################################################
// isomorph_arguments.C
// #############################################################################

//! a helper class for isomorph


class isomorph_arguments {
public:
	INT f_init_has_been_called;
	
	INT f_build_db;
	INT f_read_solutions;
	INT f_read_solutions_from_clique_finder;
	INT f_read_solutions_from_clique_finder_list_of_cases;
	const char *fname_list_of_cases;
	INT f_read_solutions_after_split;
	INT read_solutions_split_m;
	
	INT f_read_statistics_after_split;
	INT read_statistics_split_m;

	INT f_compute_orbits;
	INT f_isomorph_testing;
	INT f_classification_graph;
	INT f_event_file; // -e <event file> option
	const char *event_file_name;
	INT print_mod;
	INT f_report;
	INT f_subset_orbits;
	INT f_subset_orbits_file;
	const char *subset_orbits_fname;
	INT f_eliminate_graphs_if_possible;
	INT f_down_orbits;

	const char *prefix_iso;

	action *A;
	action *A2;
	poset_classification *gen;
	INT target_size;
	const char *prefix_with_directory;
	exact_cover_arguments *ECA;
	
	void (*callback_report)(isomorph *Iso, void *data, 
		INT verbose_level);
	void (*callback_subset_orbits)(isomorph *Iso, void *data, 
		INT verbose_level);
	void *callback_data;

	INT f_has_final_test_function;
	INT (*final_test_function)(INT *data, INT sz, 
		void *final_test_data, INT verbose_level);
	void *final_test_data;
	
	isomorph_arguments();
	~isomorph_arguments();
	void null();
	void freeself();
	void read_arguments(int argc, const char **argv, 
		INT verbose_level);
	void init(action *A, action *A2, poset_classification *gen,
		INT target_size, const char *prefix_with_directory, 
		exact_cover_arguments *ECA, 
		void (*callback_report)(isomorph *Iso, void *data, 
			INT verbose_level), 
		void (*callback_subset_orbits)(isomorph *Iso, void *data, 
			INT verbose_level), 
		void *callback_data, 
		INT verbose_level);
	void execute(INT verbose_level);

};

//! auxiliary class to pass case specific data to the function isomorph_worker

struct isomorph_worker_data {
	INT *the_set;
	INT set_size;
	void *callback_data;
};




// #############################################################################
// isomorph_global.C:
// #############################################################################

void isomorph_read_statistic_files(action *A_base, 
	action *A, poset_classification *gen,
	INT size, const char *prefix_classify, 
	const char *prefix, INT level, 
	const char **fname, INT nb_files, INT verbose_level);
void isomorph_build_db(action *A_base, 
	action *A, poset_classification *gen,
	INT size, const char *prefix_classify, 
	const char *prefix_iso, INT level, INT verbose_level);
void isomorph_read_solution_files(action *A_base, 
	action *A, poset_classification *gen,
	INT size, const char *prefix_classify, 
	const char *prefix_iso, INT level, 
	const char **fname, INT nb_files, 
	INT f_has_final_test_function, 
	INT (*final_test_function)(INT *data, INT sz, 
		void *final_test_data, INT verbose_level),
	void *final_test_data, 
	INT verbose_level);
void isomorph_init_solutions_from_memory(action *A_base, 
	action *A, poset_classification *gen,
	INT size, const char *prefix_classify, 
	const char *prefix_iso, INT level, 
	INT **Solutions, INT *Nb_sol, INT verbose_level);
void isomorph_read_solution_files_from_clique_finder_case_by_case(
	action *A_base, action *A, poset_classification *gen,
	INT size, const char *prefix_classify, 
	const char *prefix_iso, INT level, 
	const char **fname, INT *list_of_cases, 
	INT nb_files, INT verbose_level);
void isomorph_read_solution_files_from_clique_finder(action *A_base, 
	action *A, poset_classification *gen,
	INT size, const char *prefix_classify, 
	const char *prefix_iso, INT level, 
	const char **fname, INT nb_files, INT verbose_level);
void isomorph_compute_orbits(action *A_base, 
	action *A, poset_classification *gen,
	INT size, const char *prefix_classify, 
	const char *prefix_iso, INT level, INT verbose_level);
void isomorph_testing(action *A_base, 
	action *A, poset_classification *gen,
	INT size, const char *prefix_classify, 
	const char *prefix_iso, INT level, 
	INT f_play_back, const char *old_event_file, 
	INT print_mod, INT verbose_level);
void isomorph_classification_graph(action *A_base, 
	action *A, poset_classification *gen,
	INT size, const char *prefix_classify, 
	const char *prefix_iso, INT level, 
	INT verbose_level);
void isomorph_identify(action *A_base, 
	action *A, poset_classification *gen,
	INT size, const char *prefix_classify, 
	const char *prefix_iso, INT level, 
	INT identify_nb_files, const char **fname, INT *Iso_type, 
	INT f_save, INT verbose_level);
void isomorph_identify_table(action *A_base, 
	action *A, poset_classification *gen,
	INT size, const char *prefix_classify, 
	const char *prefix_iso, INT level, 
	INT nb_rows, INT *Table, INT *Iso_type, 
	INT verbose_level);
	// Table[nb_rows * size]
void isomorph_worker(action *A_base, action *A,
	poset_classification *gen,
	INT size, const char *prefix_classify, const char *prefix_iso, 
	void (*work_callback)(isomorph *Iso, void *data, INT verbose_level), 
	void *work_data, 
	INT level, INT verbose_level);
void isomorph_compute_down_orbits(action *A_base, 
	action *A, poset_classification *gen,
	INT size, const char *prefix_classify, const char *prefix, 
	void *data, 
	INT level, INT verbose_level);
void isomorph_compute_down_orbits_worker(isomorph *Iso, 
	void *data, INT verbose_level);
void isomorph_compute_down_orbits_for_isomorphism_type(
	isomorph *Iso, INT orbit, 
	INT &cnt_orbits, INT &cnt_special_orbits, 
	INT *&special_orbit_identify, 
	INT verbose_level);
void isomorph_report_data_in_source_code_inside_tex(
	isomorph &Iso, const char *prefix, char *label_of_structure_plural, 
	ostream &F, INT verbose_level);
void isomorph_report_data_in_source_code_inside_tex_with_selection(
	isomorph &Iso, const char *prefix, char *label_of_structure_plural, 
	ostream &fp, INT selection_size, INT *selection, 
	INT verbose_level);



