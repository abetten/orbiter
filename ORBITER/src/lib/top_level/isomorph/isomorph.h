// isomorph.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09


namespace orbiter {


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
	int size;
	int level;
	int f_use_database_for_starter;
	int depth_completed;
	int f_use_implicit_fusion;
	

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

	int nb_starter; 
		// the number of orbits at 'level', 
		// previously called nb_cases
	
	int N;
		// the number of solutions, 
		// computed in init_cases_from_file
		// or read from file in read_case_len
	

	
	int *solution_first;
		// [nb_starter + 1] the beginning of solutions 
		// belonging to a given starter
		// previously called case_first
	int *solution_len;
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



	int *starter_number;
		// [N]  starter_number[i] = j means that 
		// solution i belongs to starter j
		// previously called case_number





	// from the summary file (and only used in isomorph_files.C)
	int *stats_nb_backtrack;
		// [nb_starter]
	int *stats_nb_backtrack_decision;
		// [nb_starter]
	int *stats_graph_size;
		// [nb_starter]
	int *stats_time;
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



	int nb_orbits;
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

	int orbit_no;
		// Used in isomorph_testing:
		// The flag orbit we are currently testing.
		// In the end, this will become the representative of a 
		// n e w isomorphism type

	int *orbit_fst;
		// [nb_orbits + 1]
		// orbit_fst[i] is the beginning of solutions 
		// associated to the i-th flag orbit 
		// in the sorted list of solutions
		// allocated in isomorph::read_orbit_data()

	int *orbit_len;
		// [nb_orbits]
		// orbit_len[i] is the length of the i-th flag orbit 
		// allocated in isomorph::read_orbit_data()


	int *orbit_number;
		// [N]
		// orbit_number[i] is the flag orbit 
		// containing the i-th solution 
		// in the original labeling
		// allocated in isomorph::read_orbit_data()

 
	int *orbit_perm;
		// [N]
		// orbit_perm[i] is the original label 
		// of the i-th solution in the ordered list
		// we often see id = orbit_perm[orbit_fst[orbit_no]];
		// this is the index of the first solution 
		// associated to flag orbit orbit_no, for instance
		// allocated in isomorph::read_orbit_data()

	int *orbit_perm_inv;
		// [N]
		// orbit_perm_inv is the inverse of orbit_perm
		// allocated in isomorph::read_orbit_data()


	int *schreier_vector; // [N]
		// allocated in isomorph::read_orbit_data()
	int *schreier_prev; // [N]
		// allocated in isomorph::read_orbit_data()

	
	// added Dec 25, 2012:
	// computed in isomorph.C isomorph::orbits_of_stabilizer
	// these variables are not used ?? (Oct 30, 2014)
	// They are used, for instance in isomorph_testing.C 
	// isomorph::write_classification_graph (May 3, 2015)
	int *starter_orbit_fst;
		// [nb_starter + 1]
		// the beginning of flag orbits 
		// belonging to a given starter
	int *starter_nb_orbits;
		// [nb_starter]
		// the number of flag orbits belonging to a given starter




	representatives *Reps;

	int iso_nodes;
	
	

	
	int nb_open, nb_reps, nb_fused;
	
	// for iso_test:


	ofstream *fp_event_out;

	action *AA;
	action *AA_perm;
	action *AA_on_k_subsets;
	union_find *UF;
	vector_ge *gens_perm;
	
	int subset_rank;
	int *subset;
	int *subset_witness;
	int *rearranged_set;
	int *rearranged_set_save;
	int *canonical_set;
	int *tmp_set;
	int *Elt_transporter, *tmp_Elt, *Elt1, *transporter;

	int cnt_minimal;
	int NCK;
	longinteger_object stabilizer_group_order;

	int stabilizer_nb_generators;
	int **stabilizer_generators;
		// int[stabilizer_nb_generators][size]


	int *stabilizer_orbit;
	int nb_is_minimal_called;
	int nb_is_minimal;
	int nb_sets_reached;
	

	// temporary data used in identify_solution
	int f_tmp_data_has_been_allocated;
	int *tmp_set1;
	int *tmp_set2;
	int *tmp_set3;
	int *tmp_Elt1;
	int *tmp_Elt2;
	int *tmp_Elt3;
	// temporary data used in trace_set_recursion
	int *trace_set_recursion_tmp_set1;
	int *trace_set_recursion_Elt1;
	// temporary data used in trace_set_recursion
	int *apply_fusion_tmp_set1;
	int *apply_fusion_Elt1;
	// temporary data used in find_extension
	int *find_extension_set1;
	// temporary data used in make_set_smaller
	int *make_set_smaller_set;
	int *make_set_smaller_Elt1;
	int *make_set_smaller_Elt2;
	// temporary data used in orbit_representative
	int *orbit_representative_Elt1;
	int *orbit_representative_Elt2;
	// temporary data used in handle_automorphism
	int *handle_automorphism_Elt1;


	// database access:
	database *D1, *D2;
	char fname_ge1[1000];
	char fname_ge2[1000];
	FILE *fp_ge1, *fp_ge2;
	Vector *v;
	database *DB_sol;
	int *id_to_datref;
	int *id_to_hash;
	int *hash_vs_id_hash; // sorted
	int *hash_vs_id_id;
	int f_use_table_of_solutions;
	int *table_of_solutions; // [N * size]

	// pointer only, do not free:
	database *DB_level;
	FILE *fp_ge;
	
	sims *stabilizer_recreated;
	

	void (*print_set_function)(isomorph *Iso, 
		int iso_cnt, sims *Stab, schreier &Orb, 
		int *data, void *print_set_data, int verbose_level);
	void *print_set_data;
	

	// some statistics:
	int nb_times_make_set_smaller_called;

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
		int size, int level, 
		int f_use_database_for_starter, 
		int f_implicit_fusion, int verbose_level);
	void init_solution(int verbose_level);
	void load_table_of_solutions(int verbose_level);
	void init_starter_number(int verbose_level);
	void list_solutions_by_starter();
	void list_solutions_by_orbit();
	void orbits_of_stabilizer(int verbose_level);
	void orbits_of_stabilizer_case(int the_case, 
		vector_ge &gens, int verbose_level);
	void orbit_representative(int i, int &i0, 
		int &orbit, int *transporter, int verbose_level);
		// slow because it calls load_strong_generators
	void test_orbit_representative(int verbose_level);
	void test_identify_solution(int verbose_level);
	void compute_stabilizer(sims *&Stab, int verbose_level);
	void test_compute_stabilizer(int verbose_level);
	void test_memory();
	void test_edges(int verbose_level);
	int test_edge(int n1, int *subset1, 
		int *transporter, int verbose_level);
	void read_data_files_for_starter(int level, 
		const char *prefix, int verbose_level);
		// Calls gen->read_level_file_binary 
		// for all levels i from 0 to level
		// Uses letter a files for i from 0 to level - 1
		// and letter b file for i = level.
		// If gen->f_starter is TRUE, 
		// we start from i = gen->starter_size instead.
		// Finally, it computes nb_starter.
	void compute_nb_starter(int level, int verbose_level);
	void print_node_local(int level, int node_local);
	void print_node_global(int level, int node_global);
	void test_hash(int verbose_level);
	void compute_Ago_Ago_induced(longinteger_object *&Ago, 
		longinteger_object *&Ago_induced, int verbose_level);
	void init_high_level(action *A, poset_classification *gen,
		int size, char *prefix_classify, char *prefix, 
		int level, int verbose_level);
	


	// isomorph_testing.C:
	void iso_test_init(int verbose_level);
	void iso_test_init2(int verbose_level);
	void probe(int flag_orbit, int subset_rk, 
		int f_implicit_fusion, int verbose_level);
	void isomorph_testing(int t0, int f_play_back, 
		const char *play_back_file_name, 
		int f_implicit_fusion, int print_mod, int verbose_level);
	void write_classification_matrix(int verbose_level);
	void write_classification_graph(int verbose_level);
	void decomposition_matrix(int verbose_level);
	void compute_down_link(int *&down_link, int verbose_level);
	void do_iso_test(int t0, sims *&Stab, 
		int f_play_back, ifstream *play_back_file, 
		int &f_eof, int print_mod, 
		int f_implicit_fusion, int verbose_level);
	int next_subset(int t0, 
		int &f_continue, sims *Stab, int *data, 
		int f_play_back, ifstream *play_back_file, int &f_eof,
		int verbose_level);
	void process_rearranged_set(sims *Stab, int *data, 
		int f_implicit_fusion, int verbose_level);
	int is_minimal(int verbose_level);
	void stabilizer_action_exit();
	void stabilizer_action_init(int verbose_level);
		// Computes the permutations of the set 
		// that are induced by the 
		// generators for the stabilizer in AA
	void stabilizer_action_add_generator(int *Elt, int verbose_level);
	void print_statistics_iso_test(int t0, sims *Stab);
	int identify(int *set, int f_implicit_fusion, 
		int verbose_level);
	int identify_database_is_open(int *set, 
		int f_implicit_fusion, int verbose_level);
	void induced_action_on_set_basic(sims *S, 
		int *set, int verbose_level);
	void induced_action_on_set(sims *S, 
		int *set, int verbose_level);
	// Called by do_iso_test and print_isomorphism_types
	// Creates the induced action on the set from the given action.
	// The given action is gen->A2
	// The induced action is computed to AA
	// The set is in set[].
	// Allocates a n e w union_find data structure and initializes it
	// using the generators in S.
	// Calls action::induced_action_by_restriction()
	int handle_automorphism(int *set, 
		sims *Stab, int *Elt, int verbose_level);

	// isomorph_database.C:
	void setup_and_open_solution_database(int verbose_level);
	void setup_and_create_solution_database(int verbose_level);
	void close_solution_database(int verbose_level);
	void setup_and_open_level_database(int verbose_level);
	// Called from do_iso_test, identify and test_hash 
	// (Which are all in isomorph_testing.C)
	// Calls init_DB for D and D.open.
	// Calls init_DB_level for D1 and D2 and D1->open and D2->open.
	// Calls fopen for fp_ge1 and fp_ge2.
	void close_level_database(int verbose_level);
	// Closes D1, D2, fp_ge1, fp_ge2.
	void prepare_database_access(int cur_level, int verbose_level);
	// sets D to be D1 or D2, depending on cur_level
	// Called from 
	// load_strong_generators
	// trace_next_point_database
	void init_DB_sol(int verbose_level);
		// We assume that the starter is of size 5 and that 
		// fields 3-8 are the starter
	void add_solution_to_database(int *data, 
		int nb, int id, int no, 
		int nb_solutions, int h, uint_4 &datref,
		int print_mod, int verbose_level);
	void load_solution(int id, int *data);
	void load_solution_by_btree(int btree_idx, 
		int idx, int &id, int *data);
	int find_extension_easy(int *set, 
		int case_nb, int &idx, int verbose_level);
		// returns TRUE if found, FALSE otherwise
		// Called from identify_solution
		// Linear search through all solutions at a given starter.
		// calls load solution for each of the solutions 
		// stored with the case and compares the vectors.
	int find_extension_search_interval(int *set, 
		int first, int len, int &idx, 
		int f_btree_idx, int btree_idx, 
		int f_through_hash, int verbose_level);
	int find_extension_easy_old(int *set, 
		int case_nb, int &idx, int verbose_level);
	int find_extension_easy_new(int *set, 
		int case_nb, int &idx, int verbose_level);
	int open_database_and_identify_object(int *set, 
		int *transporter, 
		int f_implicit_fusion, int verbose_level);
	void init_DB_level(database &D, int level, 
		int verbose_level);
	void create_level_database(int level, int verbose_level);
	void load_strong_generators(int cur_level, 
		int cur_node_local, 
		vector_ge &gens, longinteger_object &go, 
		int verbose_level);
		// Called from compute_stabilizer and 
		// from orbit_representative
	void load_strong_generators_oracle(int cur_level, 
		int cur_node_local, 
		vector_ge &gens, longinteger_object &go, 
		int verbose_level);
	void load_strong_generators_database(int cur_level, 
		int cur_node_local, 
		vector_ge &gens, longinteger_object &go, 
		int verbose_level);
		// Reads node cur_node_local (local index) 
		// from database D through btree 0
		// Reads generators from file fp_ge

	// isomorph_trace.C:
	int identify_solution_relaxed(int *set, int *transporter, 
		int f_implicit_fusion, int &orbit_no, 
		int &f_failure_to_find_point, 
		int verbose_level);
	// returns the orbit number corresponding to 
	// the canonical version of set and the extension.
	// Calls trace_set and find_extension_easy.
	// Called from process_rearranged_set
	int identify_solution(int *set, int *transporter, 
		int f_implicit_fusion, int &f_failure_to_find_point, 
		int verbose_level);
		// returns the orbit number corresponding to 
		// the canonical version of set and the extension.
		// Calls trace_set and find_extension_easy.
		// If needed, calls make_set_smaller
	int trace_set(int *canonical_set, int *transporter, 
		int f_implicit_fusion, int &f_failure_to_find_point, 
		int verbose_level);
		// returns the case number of the canonical set 
		// (local orbit number)
		// Called from identify_solution and identify_solution_relaxed
		// calls trace_set_recursion
	void make_set_smaller(int case_nb_local, 
		int *set, int *transporter, int verbose_level);
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
	int trace_set_recursion(int cur_level, 
		int cur_node_global, 
		int *canonical_set, int *transporter, 
		int f_implicit_fusion, 
		int &f_failure_to_find_point, int verbose_level);
		// returns the node in the generator that corresponds 
		// to the canonical_set.
		// Called from trace_set.
		// Calls trace_next_point and handle_extension.
	int trace_next_point(int cur_level, 
		int cur_node_global, 
		int *canonical_set, int *transporter, 
		int f_implicit_fusion, 
		int &f_failure_to_find_point, int verbose_level);
		// Called from trace_set_recursion
		// Calls oracle::trace_next_point_in_place 
		// and (possibly) trace_next_point_database
		// Returns FALSE is the set becomes lexicographically smaller
	int trace_next_point_database(int cur_level, 
		int cur_node_global, 
		int *canonical_set, int *Elt_transporter, 
		int verbose_level);
		// Returns FALSE is the set becomes lexicographically smaller
	int handle_extension(int cur_level, 
		int cur_node_global, 
		int *canonical_set, int *Elt_transporter, 
		int f_implicit_fusion, 
		int &f_failure_to_find_point, int verbose_level);
	int handle_extension_database(int cur_level, 
		int cur_node_global, 
		int *canonical_set, int *Elt_transporter, 
		int f_implicit_fusion, 
		int &f_failure_to_find_point, 
		int verbose_level);
	int handle_extension_oracle(int cur_level, 
		int cur_node_global, 
		int *canonical_set, int *Elt_transporter, 
		int f_implicit_fusion, 
		int &f_failure_to_find_point, 
		int verbose_level);
	// Returns next_node_global at level cur_level + 1.
	void apply_isomorphism_database(int cur_level, 
		int cur_node_global, 
		int current_extension, int *canonical_set, 
		int *Elt_transporter, int ref, 
		int verbose_level);
	void apply_isomorphism_oracle(int cur_level, 
		int cur_node_global, 
		int current_extension, int *canonical_set, 
		int *Elt_transporter, 
		int verbose_level);


	// isomorph_files.C:
	void init_solutions(int **Solutions, 
		int *Nb_sol, int verbose_level);
	// Solutions[nb_starter], Nb_sol[nb_starter]
	void count_solutions_from_clique_finder_case_by_case(int nb_files, 
		int *list_of_cases, const char **fname, 
		int verbose_level);
	void count_solutions_from_clique_finder(int nb_files, 
		const char **fname, 
		int verbose_level);
	void read_solutions_from_clique_finder_case_by_case(int nb_files, 
		int *list_of_cases, const char **fname, 
		int verbose_level);
	void read_solutions_from_clique_finder(int nb_files, 
		const char **fname, int verbose_level);
	void add_solutions_to_database(int *Solutions, 
		int the_case, int nb_solutions, int nb_solutions_total, 
		int print_mod, int &no, 
		int verbose_level);
	void build_up_database(int nb_files, const char **fname, 
		int f_has_final_test_function, 
		int (*final_test_function)(int *data, int sz, 
			void *final_test_data, int verbose_level),
		void *final_test_data, 
		int verbose_level);
	void init_cases_from_file_modulus_and_build_up_database(
		int modulus, int level, 
		int f_collated, int base_split, 
		int f_get_statistics, 
		int f_has_final_test_function, 
		int (*final_test_function)(int *data, int sz, 
			void *final_test_data, int verbose_level),
		void *final_test_data, 
		int verbose_level);
	void init_cases_from_file_mixed_modulus_and_build_up_database(
		int nb_Mod, int *Mod_r, int *Mod_split, int *Mod_base_split, 
		int level, int f_get_statistics, 
		int f_has_final_test_function, 
		int (*final_test_function)(int *data, int sz, 
			void *final_test_data, int verbose_level),
		void *final_test_data, 
		int verbose_level);
	void count_solutions(int nb_files, 
		const char **fname, int f_get_statistics, 
		int f_has_final_test_function, 
		int (*final_test_function)(int *data, int sz, 
			void *final_test_data, int verbose_level),
		void *final_test_data, 
		int verbose_level);
	void get_statistics(int nb_files, const char **fname, 
		int verbose_level);
	void write_statistics();
	void evaluate_statistics(int verbose_level);
	void count_solutions2(int nb_files, const char **fname, 
		int &total_days, int &total_hours, int &total_minutes, 
		int f_has_final_test_function, 
		int (*final_test_function)(int *data, int sz, 
			void *final_test_data, int verbose_level),
		void *final_test_data, 
		int verbose_level);
	void write_solution_first_and_len(); // previously write_case_len
	void read_solution_first_and_len(); // previously read_case_len
	void write_starter_nb_orbits(int verbose_level);
	void read_starter_nb_orbits(int verbose_level);
	void write_hash_and_datref_file(int verbose_level);
		// Writes the file 'fname_hash_and_datref' 
		// containing id_to_hash[] and id_to_datref[]
	void read_hash_and_datref_file(int verbose_level);
		// Reads the file 'fname_hash_and_datref' 
		// containing id_to_hash[] and id_to_datref[]
		// Also initializes hash_vs_id_hash and hash_vs_id_id
		// Called from init_solution
	void print_hash_vs_id();
	void write_orbit_data(int verbose_level);
		// Writes the file 'fname_staborbits'
	void read_orbit_data(int verbose_level);
		// Reads from the file 'fname_staborbits'
		// Reads nb_orbits, N, 
		// orbit_fst[nb_orbits + 1]
		// orbit_len[nb_orbits]
		// orbit_number[N]
		// orbit_perm[N]
		// schreier_vector[N]
		// schreier_prev[N]
		// and computed orbit_perm_inv[N]
	void print_isomorphism_types(int f_select, 
		int select_first, int select_len, 
		int verbose_level);
		// Calls print_set_function (if available)
	void induced_action_on_set_and_kernel(ostream &file, 
		action *A, sims *Stab, int size, int *set, 
		int verbose_level);
	void handle_event_files(int nb_event_files, 
		const char **event_file_name, int verbose_level);
	void read_event_file(const char *event_file_name, 
		int verbose_level);
	void skip_through_event_file(ifstream &f, 
		int verbose_level);
	void skip_through_event_file1(ifstream &f, 
		int case_no, int orbit_no, int verbose_level);
	void event_file_completed_cases(const char *event_file_name, 
		int &nb_completed_cases, int *completed_cases, 
		int verbose_level);
	void event_file_read_case(const char *event_file_name, 
		int case_no, int verbose_level);
	void event_file_read_case1(ifstream &f, 
		int case_no, int verbose_level);
	int next_subset_play_back(int &subset_rank, 
		ifstream *play_back_file, 
		int &f_eof, int verbose_level);
	void read_everything_including_classification(
		const char *prefix_classify, int verbose_level);


};


// #############################################################################
// isomorph_arguments.C
// #############################################################################

//! a helper class for isomorph


class isomorph_arguments {
public:
	int f_init_has_been_called;
	
	int f_build_db;
	int f_read_solutions;
	int f_read_solutions_from_clique_finder;
	int f_read_solutions_from_clique_finder_list_of_cases;
	const char *fname_list_of_cases;
	int f_read_solutions_after_split;
	int read_solutions_split_m;
	
	int f_read_statistics_after_split;
	int read_statistics_split_m;

	int f_compute_orbits;
	int f_isomorph_testing;
	int f_classification_graph;
	int f_event_file; // -e <event file> option
	const char *event_file_name;
	int print_mod;
	int f_report;
	int f_subset_orbits;
	int f_subset_orbits_file;
	const char *subset_orbits_fname;
	int f_eliminate_graphs_if_possible;
	int f_down_orbits;

	const char *prefix_iso;

	action *A;
	action *A2;
	poset_classification *gen;
	int target_size;
	const char *prefix_with_directory;
	exact_cover_arguments *ECA;
	
	void (*callback_report)(isomorph *Iso, void *data, 
		int verbose_level);
	void (*callback_subset_orbits)(isomorph *Iso, void *data, 
		int verbose_level);
	void *callback_data;

	int f_has_final_test_function;
	int (*final_test_function)(int *data, int sz, 
		void *final_test_data, int verbose_level);
	void *final_test_data;
	
	isomorph_arguments();
	~isomorph_arguments();
	void null();
	void freeself();
	void read_arguments(int argc, const char **argv, 
		int verbose_level);
	void init(action *A, action *A2, poset_classification *gen,
		int target_size, const char *prefix_with_directory, 
		exact_cover_arguments *ECA, 
		void (*callback_report)(isomorph *Iso, void *data, 
			int verbose_level), 
		void (*callback_subset_orbits)(isomorph *Iso, void *data, 
			int verbose_level), 
		void *callback_data, 
		int verbose_level);
	void execute(int verbose_level);

};

//! auxiliary class to pass case specific data to the function isomorph_worker

struct isomorph_worker_data {
	int *the_set;
	int set_size;
	void *callback_data;
};




// #############################################################################
// isomorph_global.C:
// #############################################################################

void isomorph_read_statistic_files(action *A_base, 
	action *A, poset_classification *gen,
	int size, const char *prefix_classify, 
	const char *prefix, int level, 
	const char **fname, int nb_files, int verbose_level);
void isomorph_build_db(action *A_base, 
	action *A, poset_classification *gen,
	int size, const char *prefix_classify, 
	const char *prefix_iso, int level, int verbose_level);
void isomorph_read_solution_files(action *A_base, 
	action *A, poset_classification *gen,
	int size, const char *prefix_classify, 
	const char *prefix_iso, int level, 
	const char **fname, int nb_files, 
	int f_has_final_test_function, 
	int (*final_test_function)(int *data, int sz, 
		void *final_test_data, int verbose_level),
	void *final_test_data, 
	int verbose_level);
void isomorph_init_solutions_from_memory(action *A_base, 
	action *A, poset_classification *gen,
	int size, const char *prefix_classify, 
	const char *prefix_iso, int level, 
	int **Solutions, int *Nb_sol, int verbose_level);
void isomorph_read_solution_files_from_clique_finder_case_by_case(
	action *A_base, action *A, poset_classification *gen,
	int size, const char *prefix_classify, 
	const char *prefix_iso, int level, 
	const char **fname, int *list_of_cases, 
	int nb_files, int verbose_level);
void isomorph_read_solution_files_from_clique_finder(action *A_base, 
	action *A, poset_classification *gen,
	int size, const char *prefix_classify, 
	const char *prefix_iso, int level, 
	const char **fname, int nb_files, int verbose_level);
void isomorph_compute_orbits(action *A_base, 
	action *A, poset_classification *gen,
	int size, const char *prefix_classify, 
	const char *prefix_iso, int level, int verbose_level);
void isomorph_testing(action *A_base, 
	action *A, poset_classification *gen,
	int size, const char *prefix_classify, 
	const char *prefix_iso, int level, 
	int f_play_back, const char *old_event_file, 
	int print_mod, int verbose_level);
void isomorph_classification_graph(action *A_base, 
	action *A, poset_classification *gen,
	int size, const char *prefix_classify, 
	const char *prefix_iso, int level, 
	int verbose_level);
void isomorph_identify(action *A_base, 
	action *A, poset_classification *gen,
	int size, const char *prefix_classify, 
	const char *prefix_iso, int level, 
	int identify_nb_files, const char **fname, int *Iso_type, 
	int f_save, int verbose_level);
void isomorph_identify_table(action *A_base, 
	action *A, poset_classification *gen,
	int size, const char *prefix_classify, 
	const char *prefix_iso, int level, 
	int nb_rows, int *Table, int *Iso_type, 
	int verbose_level);
	// Table[nb_rows * size]
void isomorph_worker(action *A_base, action *A,
	poset_classification *gen,
	int size, const char *prefix_classify, const char *prefix_iso, 
	void (*work_callback)(isomorph *Iso, void *data, int verbose_level), 
	void *work_data, 
	int level, int verbose_level);
void isomorph_compute_down_orbits(action *A_base, 
	action *A, poset_classification *gen,
	int size, const char *prefix_classify, const char *prefix, 
	void *data, 
	int level, int verbose_level);
void isomorph_compute_down_orbits_worker(isomorph *Iso, 
	void *data, int verbose_level);
void isomorph_compute_down_orbits_for_isomorphism_type(
	isomorph *Iso, int orbit, 
	int &cnt_orbits, int &cnt_special_orbits, 
	int *&special_orbit_identify, 
	int verbose_level);
void isomorph_report_data_in_source_code_inside_tex(
	isomorph &Iso, const char *prefix, char *label_of_structure_plural, 
	ostream &F, int verbose_level);
void isomorph_report_data_in_source_code_inside_tex_with_selection(
	isomorph &Iso, const char *prefix, char *label_of_structure_plural, 
	ostream &fp, int selection_size, int *selection, 
	int verbose_level);

}


