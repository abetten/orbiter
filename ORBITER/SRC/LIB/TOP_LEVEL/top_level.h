// top_level.h
//
// Anton Betten
//
// started:  September 23 2010
//
// based on global.h, which was taken from reader.h: 3/22/09

typedef class representatives representatives; // added 7/3/12
typedef class isomorph isomorph; // added 3/22/09
typedef class search_blocking_set search_blocking_set; // added Nov 2, 2010
typedef class choose_points_or_lines choose_points_or_lines; // added Nov 29, 2010
typedef class subspace_orbits subspace_orbits; // added March 29, 2012 (started Jan 25, 2010)
typedef struct factor_group factor_group;
typedef class orbit_rep orbit_rep;
typedef class orbit_of_sets orbit_of_sets; // added March 27, 2013
typedef class singer_cycle singer_cycle; // added March 27, 2013
typedef class exact_cover exact_cover; // added April 30, 2013
typedef class recoordinatize recoordinatize; // added November 2, 2013
typedef class translation_plane translation_plane; // added November 2, 2013
typedef class polar polar;
typedef class orbit_of_subspaces orbit_of_subspaces; // added April 10, 2014
typedef class young young; // added March 16, 2015
typedef class exact_cover_arguments exact_cover_arguments; // added January 12, 2016
typedef class translation_plane_via_andre_model translation_plane_via_andre_model; // added January 25, 2016
typedef class isomorph_arguments isomorph_arguments; // added January 27, 2016
typedef struct isomorph_worker_data isomorph_worker_data;
typedef class elliptic_curve elliptic_curve; // added November 19, 2014
//typedef class surface_classify surface_classify; // added August 28, 2016
typedef class surface_classify_wedge surface_classify_wedge; // added September 2, 2016
//typedef class surface_data surface_data; // added August 30, 2016
//typedef surface_data *p_surface_data;
typedef class arc_generator arc_generator; // moved here February 23, 2017
typedef class surface_with_action surface_with_action; // added March 22, 2017
typedef class surface_object_with_action surface_object_with_action; // added October 4, 2017
typedef class classify_trihedral_pairs classify_trihedral_pairs; // added October 9, 2017
typedef class classify_double_sixes classify_double_sixes; // added October 10, 2017
typedef class object_in_projective_space_with_action object_in_projective_space_with_action; // added December 30, 2017
typedef class surface_create_description surface_create_description; // added January 14, 2018
typedef class surface_create surface_create; // added January 14, 2018
typedef class arc_lifting arc_lifting; // added January 14, 2018


// ####################################################################################
// top_level_global.C
// ####################################################################################


INT callback_partial_ovoid_test(INT len, INT *S, void *data, INT verbose_level);



// ####################################################################################
// representatives.C
// ####################################################################################

class representatives {
public:
	action *A;

	BYTE prefix[1000];
	BYTE fname_rep[1000];
	BYTE fname_stabgens[1000];
	BYTE fname_fusion[1000];
	BYTE fname_fusion_ge[1000];



	// flag orbits:
	INT nb_objects;
	INT *fusion; // [nb_objects]
		// fusion[i] == -2 means that the flag orbit i 
		// has not yet been processed by the
		// isomorphism testing procedure.
		// fusion[i] = i means that flag orbit [i] 
		// in an orbit representative
		// Otherwise, fusion[i] is an earlier flag_orbit, 
		// and handle[i] is a group element that maps 
		// to it
	INT *handle; // [nb_objects]
		// handle[i] is only relevant if fusion[i] != i,
		// i.e., if flag orbit i is not a representative
		// of an isomorphism type.
		// In this case, handle[i] is the (handle of a) 
		// group element moving flag orbit i to flag orbit fusion[i]. 


	// classified objects:
	INT count;
	INT *rep; // [count] 
	sims **stab; // [count] 



	//BYTE *elt;
	INT *Elt1;
	INT *tl; // [A->base_len]

	INT nb_open;
	INT nb_reps;
	INT nb_fused;


	representatives();
	void null();
	~representatives();
	void free();
	void init(action *A, INT nb_objects, BYTE *prefix, INT verbose_level);
	void write_fusion(INT verbose_level);
	void read_fusion(INT verbose_level);
	void write_representatives_and_stabilizers(INT verbose_level);
	void read_representatives_and_stabilizers(INT verbose_level);
	void save(INT verbose_level);
	void load(INT verbose_level);
	void calc_fusion_statistics();
	void print_fusion_statistics();
};


// ####################################################################################
// isomorph.C
// isomorph_testing.C
// isomorph_database.C
// isomorph_trace.C
// isomorph_files.C
// ####################################################################################


class isomorph {
public:
	INT size;
	INT level;
	INT f_use_database_for_starter;
	INT depth_completed;
	INT f_use_implicit_fusion;
	

	BYTE prefix[500];
	BYTE prefix_invariants[500];
	BYTE prefix_tex[500];

	BYTE fname_staborbits[500];
	BYTE fname_case_len[500];
	BYTE fname_statistics[500];
	BYTE fname_hash_and_datref[500];
	BYTE fname_db1[500];
	BYTE fname_db2[500];
	BYTE fname_db3[500];
	BYTE fname_db4[500];
	BYTE fname_db5[500];
	BYTE fname_db_level[500];
	BYTE fname_db_level_idx1[500];
	BYTE fname_db_level_idx2[500];
	BYTE fname_db_level_ge[500];
	
	BYTE event_out_fname[1000];
	BYTE fname_orbits_of_stabilizer_csv[1000];

	INT nb_starter; 
		// the number of orbits at 'level', 
		// previously called nb_cases
	
	INT N;
		// the number of solutions, 
		// computed in init_cases_from_file
		// or read from file in read_case_len
	

	
	INT *solution_first;
		// [nb_starter + 1] the beginning of solutions belonging to a given starter
		// previously called case_first
	INT *solution_len;
		// [nb_starter + 1] the number of solutions belonging to a given starter
		// previously called case_len


		// solution_first and solution_len are initialized in 
		// isomorph_files.C:
		// isomorph::init_solutions
		// isomorph::count_solutions_from_clique_finder
		// isomorph::count_solutions

		// they are written to file in isomorph::write_solution_first_and_len()



	INT *starter_number;
		// [N]  starter_number[i] = j means that solution i belongs to starter j
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


	generator *gen;
		// do not free



	INT nb_orbits;
		// Number of flag orbits.
		// Overall number of orbits of stabilizers of starters on the solutions
		// computed in orbits_of_stabilizer, 
		// which in turn calls orbits_of_stabilizer_case for each starter
		// orbits_of_stabilizer_case takes the strong generators from the 
		// generator data structure
		// For computing the orbits, induced_action_on_sets
		// is used to establish an action on sets.

	INT orbit_no;
		// Used in isomorph_testing:
		// The flag orbit we are currently testing.
		// In the end, this will become the representative of a 
		// new isomorphism type

	INT *orbit_fst;
		// [nb_orbits + 1]
		// orbit_fst[i] is the beginning of solutions associated to the i-th flag orbit 
		// in the sorted list of solutions
		// allocated in isomorph::read_orbit_data()

	INT *orbit_len;
		// [nb_orbits]
		// orbit_len[i] is the length of the i-th flag orbit 
		// allocated in isomorph::read_orbit_data()


	INT *orbit_number;
		// [N]
		// orbit_number[i] is the flag orbit containing the i-th solution 
		// in the original labeling
		// allocated in isomorph::read_orbit_data()

 
	INT *orbit_perm;
		// [N]
		// orbit_perm[i] is the original label of the i-th solution in the ordered list
		// we often see id = orbit_perm[orbit_fst[orbit_no]];
		// this is the index of the first solution associated to flag orbit orbit_no, for instance
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
	// They are used, for instance in isomorph_testing.C isomorph::write_classification_graph (May 3, 2015)
	INT *starter_orbit_fst;
		// [nb_starter + 1]
		// the beginning of flag orbits belonging to a given starter
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
	INT **stabilizer_generators; // INT[stabilizer_nb_generators][size]


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
	BYTE fname_ge1[1000];
	BYTE fname_ge2[1000];
	FILE *fp_ge1, *fp_ge2;
	Vector *v;
	//INT first_node;
	//database *D;
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
	

	void (*print_set_function)(isomorph *Iso, INT iso_cnt, sims *Stab, schreier &Orb, 
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
	void init(const BYTE *prefix, 
		action *A_base, action *A, generator *gen, 
		INT size, INT level, 
		INT f_use_database_for_starter, 
		INT f_implicit_fusion, INT verbose_level);
	void init_solution(INT verbose_level); // previously init_cases
	void load_table_of_solutions(INT verbose_level);
	void init_starter_number(INT verbose_level); // previously init_case_number
	void list_solutions_by_starter();
	void list_solutions_by_orbit();
	void orbits_of_stabilizer(INT verbose_level);
	void orbits_of_stabilizer_case(INT the_case, vector_ge &gens, INT verbose_level);
	void orbit_representative(INT i, INT &i0, 
		INT &orbit, INT *transporter, INT verbose_level);
	// slow because it calls load_strong_generators
	void test_orbit_representative(INT verbose_level);
	void test_identify_solution(INT verbose_level);
	void compute_stabilizer(sims *&Stab, INT verbose_level);
	void test_compute_stabilizer(INT verbose_level);
	void test_memory();
	void test_edges(INT verbose_level);
	INT test_edge(INT n1, INT *subset1, INT *transporter, INT verbose_level);
#if 0
	void read_data_file(INT f_recompute_schreier, INT verbose_level);
	// Reads the data for starters.
	// First, it reads the data file whose name is fname_data_file,
	// which contains the data up to level - 1
	// and sets depth_completed to one level less, which is level - 2.
	// Then, the bottom two levels are read from the level databases 
	// through the object database D.
	// In this step, it calls init_DB_level.
#endif
	void read_data_files_for_starter(INT level, 
		const BYTE *prefix, INT verbose_level);
		// Calls gen->read_level_file_binary for all levels i from 0 to level
		// Uses letter a files for i from 0 to level - 1
		// and letter b file for i = level.
		// If gen->f_starter is TRUE, we start from i = gen->starter_size instead.
		// Finally, it computes nb_starter.
	void compute_nb_starter(INT level, INT verbose_level);
	void print_node_local(INT level, INT node_local);
	void print_node_global(INT level, INT node_global);
	void test_hash(INT verbose_level);
	void compute_Ago_Ago_induced(longinteger_object *&Ago, longinteger_object *&Ago_induced, INT verbose_level);
	void init_high_level(action *A, generator *gen, 
		INT size, BYTE *prefix_classify, BYTE *prefix, INT level, INT verbose_level);
	


	// isomorph_testing.C:
	void iso_test_init(INT verbose_level);
	void iso_test_init2(INT verbose_level);
	void probe(INT flag_orbit, INT subset_rk, INT f_implicit_fusion, INT verbose_level);
	void isomorph_testing(INT t0, INT f_play_back, const BYTE *play_back_file_name, 
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
		// Computes the permutations of the set that are induced by the 
		// generators for the stabilizer in AA
	void stabilizer_action_add_generator(INT *Elt, INT verbose_level);
	void print_statistics_iso_test(INT t0, sims *Stab);
	INT identify(INT *set, INT f_implicit_fusion, INT verbose_level);
	INT identify_database_is_open(INT *set, INT f_implicit_fusion, INT verbose_level);
	void induced_action_on_set_basic(sims *S, INT *set, INT verbose_level);
	void induced_action_on_set(sims *S, INT *set, INT verbose_level);
	// Called by do_iso_test and print_isomorphism_types
	// Creates the induced action on the set from the given action.
	// The given action is gen->A2
	// The induced action is computed to AA
	// The set is in set[].
	// Allocates a new union_find data structure and initializes it 
	// using the generators in S.
	// Calls action::induced_action_by_restriction()
	INT handle_automorphism(INT *set, sims *Stab, INT *Elt, INT verbose_level);

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
		INT nb, INT id, INT no, INT nb_solutions, INT h, UINT4 &datref, 
		INT print_mod, INT verbose_level);
	void load_solution(INT id, INT *data);
	void load_solution_by_btree(INT btree_idx, INT idx, INT &id, INT *data);
	INT find_extension_easy(INT *set, INT case_nb, INT &idx, INT verbose_level);
		// returns TRUE if found, FALSE otherwise
		// Called from identify_solution
		// Linear search through all solutions at a given starter.
		// calls load solution for each of the solutions 
		// stored with the case and compares the vectors.
	INT find_extension_search_interval(INT *set, 
		INT first, INT len, INT &idx, 
		INT f_btree_idx, INT btree_idx, 
		INT f_through_hash, INT verbose_level);
	INT find_extension_easy_old(INT *set, INT case_nb, INT &idx, INT verbose_level);
	INT find_extension_easy_new(INT *set, INT case_nb, INT &idx, INT verbose_level);
	INT open_database_and_identify_object(INT *set, INT *transporter, 
		INT f_implicit_fusion, INT verbose_level);
	void init_DB_level(database &D, INT level, INT verbose_level);
	void create_level_database(INT level, INT verbose_level);
	void load_strong_generators(INT cur_level, INT cur_node_local, 
		vector_ge &gens, longinteger_object &go, INT verbose_level);
		// Called from compute_stabilizer and from orbit_representative
	void load_strong_generators_oracle(INT cur_level, INT cur_node_local, 
		vector_ge &gens, longinteger_object &go, INT verbose_level);
	void load_strong_generators_database(INT cur_level, INT cur_node_local, 
		vector_ge &gens, longinteger_object &go, INT verbose_level);
		// Reads node cur_node_local (local index) from database D through btree 0
		// Reads generators from file fp_ge

	// isomorph_trace.C:
	INT identify_solution_relaxed(INT *set, INT *transporter, 
		INT f_implicit_fusion, INT &orbit_no, INT &f_failure_to_find_point, INT verbose_level);
	// returns the orbit number corresponding to 
	// the canonical version of set and the extension.
	// Calls trace_set and find_extension_easy.
	// Called from process_rearranged_set
	INT identify_solution(INT *set, INT *transporter, 
		INT f_implicit_fusion, INT &f_failure_to_find_point, INT verbose_level);
		// returns the orbit number corresponding to 
		// the canonical version of set and the extension.
		// Calls trace_set and find_extension_easy.
		// If needed, calls make_set_smaller
	INT trace_set(INT *canonical_set, INT *transporter, 
		INT f_implicit_fusion, INT &f_failure_to_find_point, INT verbose_level);
		// returns the case number of the canonical set (local orbit number)
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
	// or one can produce such an element by applying an element in the 
	// stabilizer of the current starter.
	INT trace_set_recursion(INT cur_level, INT cur_node_global, 
		INT *canonical_set, INT *transporter, 
		INT f_implicit_fusion, INT &f_failure_to_find_point, INT verbose_level);
		// returns the node in the generator that corresponds 
		// to the canonical_set.
		// Called from trace_set.
		// Calls trace_next_point and handle_extension.
	INT trace_next_point(INT cur_level, INT cur_node_global, 
		INT *canonical_set, INT *transporter, 
		INT f_implicit_fusion, INT &f_failure_to_find_point, INT verbose_level);
		// Called from trace_set_recursion
		// Calls oracle::trace_next_point_in_place 
		// and (possibly) trace_next_point_database
		// Returns FALSE is the set becomes lexicographically smaller
	INT trace_next_point_database(INT cur_level, INT cur_node_global, 
		INT *canonical_set, INT *Elt_transporter, INT verbose_level);
		// Returns FALSE is the set becomes lexicographically smaller
	INT handle_extension(INT cur_level, INT cur_node_global, 
		INT *canonical_set, INT *Elt_transporter, 
		INT f_implicit_fusion, INT &f_failure_to_find_point, INT verbose_level);
	INT handle_extension_database(INT cur_level, INT cur_node_global, 
		INT *canonical_set, INT *Elt_transporter, 
		INT f_implicit_fusion, INT &f_failure_to_find_point, INT verbose_level);
	INT handle_extension_oracle(INT cur_level, INT cur_node_global, 
		INT *canonical_set, INT *Elt_transporter, 
		INT f_implicit_fusion, INT &f_failure_to_find_point, INT verbose_level);
	// Returns next_node_global at level cur_level + 1.
	void apply_fusion_element_database(INT cur_level, INT cur_node_global, 
		INT current_extension, INT *canonical_set, INT *Elt_transporter, INT ref, 
		INT verbose_level);
	void apply_fusion_element_oracle(INT cur_level, INT cur_node_global, 
		INT current_extension, INT *canonical_set, INT *Elt_transporter, 
		INT verbose_level);


	// isomorph_files.C:
	void init_solutions(INT **Solutions, INT *Nb_sol, INT verbose_level);
	// Solutions[nb_starter], Nb_sol[nb_starter]
	void count_solutions_from_clique_finder_case_by_case(INT nb_files, INT *list_of_cases, const BYTE **fname, INT verbose_level);
	void count_solutions_from_clique_finder(INT nb_files, const BYTE **fname, INT verbose_level);
	void read_solutions_from_clique_finder_case_by_case(INT nb_files, INT *list_of_cases, const BYTE **fname, INT verbose_level);
	void read_solutions_from_clique_finder(INT nb_files, const BYTE **fname, INT verbose_level);
	void add_solutions_to_database(INT *Solutions, 
		INT the_case, INT nb_solutions, INT nb_solutions_total, INT print_mod, INT &no, 
		INT verbose_level);
	void build_up_database(INT nb_files, const BYTE **fname, 
		INT f_has_final_test_function, 
		INT (*final_test_function)(INT *data, INT sz, void *final_test_data, INT verbose_level),
		void *final_test_data, 
		INT verbose_level);
	void init_cases_from_file_modulus_and_build_up_database(
		INT modulus, INT level, 
		INT f_collated, INT base_split, 
		INT f_get_statistics, 
		INT f_has_final_test_function, 
		INT (*final_test_function)(INT *data, INT sz, void *final_test_data, INT verbose_level),
		void *final_test_data, 
		INT verbose_level);
	void init_cases_from_file_mixed_modulus_and_build_up_database(
		INT nb_Mod, INT *Mod_r, INT *Mod_split, INT *Mod_base_split, 
		INT level, INT f_get_statistics, 
		INT f_has_final_test_function, 
		INT (*final_test_function)(INT *data, INT sz, void *final_test_data, INT verbose_level),
		void *final_test_data, 
		INT verbose_level);
	void count_solutions(INT nb_files, const BYTE **fname, INT f_get_statistics, 
		INT f_has_final_test_function, 
		INT (*final_test_function)(INT *data, INT sz, void *final_test_data, INT verbose_level),
		void *final_test_data, 
		INT verbose_level);
	void get_statistics(INT nb_files, const BYTE **fname, INT verbose_level);
	void write_statistics();
	void evaluate_statistics(INT verbose_level);
	void count_solutions2(INT nb_files, const BYTE **fname, 
		INT &total_days, INT &total_hours, INT &total_minutes, 
		INT f_has_final_test_function, 
		INT (*final_test_function)(INT *data, INT sz, void *final_test_data, INT verbose_level),
		void *final_test_data, 
		INT verbose_level);
	void write_solution_first_and_len(); // previously write_case_len
	void read_solution_first_and_len(); // previously read_case_len
	void write_starter_nb_orbits(INT verbose_level);
	void read_starter_nb_orbits(INT verbose_level);
	void write_hash_and_datref_file(INT verbose_level);
	// Writes the file 'fname_hash_and_datref' containing id_to_hash[] and id_to_datref[]
	void read_hash_and_datref_file(INT verbose_level);
	// Reads the file 'fname_hash_and_datref' containing id_to_hash[] and id_to_datref[]
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
	void print_isomorphism_types(INT f_select, INT select_first, INT select_len, INT verbose_level);
	// Calls print_set_function (if available)
	void induced_action_on_set_and_kernel(ostream &file, action *A, sims *Stab, INT size, INT *set, INT verbose_level);
	void handle_event_files(INT nb_event_files, const BYTE **event_file_name, INT verbose_level);
	void read_event_file(const BYTE *event_file_name, INT verbose_level);
	void skip_through_event_file(ifstream &f, INT verbose_level);
	void skip_through_event_file1(ifstream &f, INT case_no, INT orbit_no, INT verbose_level);
	void event_file_completed_cases(const BYTE *event_file_name, 
		INT &nb_completed_cases, INT *completed_cases, INT verbose_level);
	void event_file_read_case(const BYTE *event_file_name, INT case_no, INT verbose_level);
	void event_file_read_case1(ifstream &f, INT case_no, INT verbose_level);
	INT next_subset_play_back(INT &subset_rank, ifstream *play_back_file, 
		INT &f_eof, INT verbose_level);
	void read_everything_including_classification(const BYTE *prefix_classify, INT verbose_level);


};


// isomorph_global.C:
void isomorph_read_statistic_files(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix, INT level, 
	const BYTE **fname, INT nb_files, INT verbose_level);
void isomorph_build_db(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix_iso, INT level, INT verbose_level);
void isomorph_read_solution_files(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix_iso, INT level, 
	const BYTE **fname, INT nb_files, 
	INT f_has_final_test_function, 
	INT (*final_test_function)(INT *data, INT sz, void *final_test_data, INT verbose_level),
	void *final_test_data, 
	INT verbose_level);
void isomorph_init_solutions_from_memory(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix_iso, INT level, 
	INT **Solutions, INT *Nb_sol, INT verbose_level);
void isomorph_read_solution_files_from_clique_finder_case_by_case(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix_iso, INT level, 
	const BYTE **fname, INT *list_of_cases, INT nb_files, INT verbose_level);
void isomorph_read_solution_files_from_clique_finder(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix_iso, INT level, 
	const BYTE **fname, INT nb_files, INT verbose_level);
void isomorph_compute_orbits(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix_iso, INT level, INT verbose_level);
void isomorph_testing(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix_iso, INT level, 
	INT f_play_back, const BYTE *old_event_file, INT print_mod, INT verbose_level);
void isomorph_classification_graph(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix_iso, INT level, 
	INT verbose_level);
void isomorph_identify(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix_iso, INT level, 
	INT identify_nb_files, const BYTE **fname, INT *Iso_type, 
	INT f_save, INT verbose_level);
void isomorph_identify_table(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix_iso, INT level, 
	INT nb_rows, INT *Table, INT *Iso_type, 
	INT verbose_level);
// Table[nb_rows * size]
void isomorph_worker(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix_iso, 
	void (*work_callback)(isomorph *Iso, void *data, INT verbose_level), 
	void *work_data, 
	INT level, INT verbose_level);
void isomorph_compute_down_orbits(action *A_base, action *A, generator *gen, 
	INT size, const BYTE *prefix_classify, const BYTE *prefix, 
	void *data, 
	INT level, INT verbose_level);
void isomorph_compute_down_orbits_worker(isomorph *Iso, void *data, INT verbose_level);
void isomorph_compute_down_orbits_for_isomorphism_type(isomorph *Iso, INT orbit, 
	INT &cnt_orbits, INT &cnt_special_orbits, INT *&special_orbit_identify, INT verbose_level);
void isomorph_report_data_in_source_code_inside_tex(isomorph &Iso, const BYTE *prefix, BYTE *label_of_structure_plural, ostream &F, INT verbose_level);
void isomorph_report_data_in_source_code_inside_tex_with_selection(isomorph &Iso, const BYTE *prefix, BYTE *label_of_structure_plural, ostream &f, INT selection_size, INT *selection, INT verbose_level);






// ####################################################################################
// extra.C:
// ####################################################################################


void isomorph_print_set(ostream &ost, INT len, INT *S, void *data);
void print_from_to(INT d, INT i, INT j, INT *v1, INT *v2);
sims *create_sims_for_stabilizer(action *A, 
	INT *set, INT set_size, INT verbose_level);
sims *create_sims_for_stabilizer_with_input_group(action *A, 
	action *A0, strong_generators *Strong_gens, 
	INT *set, INT set_size, INT verbose_level);

void solve_diophant(INT *Inc, INT nb_rows, INT nb_cols, INT nb_needed, 
	INT f_has_Rhs, INT *Rhs, 
	INT *&Solutions, INT &nb_sol, INT &nb_backtrack, INT &dt, 
	INT f_DLX, 
	INT f_draw_system, const BYTE *fname_system, 
	INT f_write_tree, const BYTE *fname_tree, 
	INT verbose_level);
// allocates Solutions[nb_sol * target_size]
// where target_size = starter_size + nb_needed
void compute_lifts(exact_cover_arguments *ECA, INT verbose_level);
void compute_lifts_new(
	action *A, action *A2, 
	void *user_data, 
	const BYTE *base_fname, 
	const BYTE *input_prefix, const BYTE *output_prefix, 
	const BYTE *solution_prefix, 
	INT starter_size, INT target_size, 
	INT f_lex, INT f_split, INT split_r, INT split_m, 
	INT f_solve, INT f_save, INT f_read_instead, 
	INT f_draw_system, const BYTE *fname_system, 
	INT f_write_tree, const BYTE *fname_tree,
	void (*prepare_function_new)(exact_cover *E, INT starter_case, 
		INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_label, 
		INT &f_ruled_out, 
		INT verbose_level), 
	void (*early_test_function)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level), 
	void *early_test_function_data,
	INT f_has_solution_test_function, 
	INT (*solution_test_func)(exact_cover *EC, INT *S, INT len, void *data, INT verbose_level), 
	void *solution_test_func_data,
	INT f_has_late_cleanup_function, 
	void (*late_cleanup_function)(exact_cover *EC, INT starter_case, INT verbose_level), 
	INT verbose_level);



// ####################################################################################
// incidence_structure.C:
// ####################################################################################


void incidence_structure_compute_tda(partitionstack &S, 
	incidence_structure *Inc, 
	action *A, 
	INT f_write_tda_files, 
	INT f_include_group_order, 
	INT f_pic, 
	INT f_include_tda_scheme, 
	INT verbose_level);
void incidence_structure_compute_TDA_general(partitionstack &S, 
	incidence_structure *Inc, 
	INT f_combined_action, 
	action *A, action *A_on_points, action *A_on_lines, 
	vector_ge *generators, 
	INT f_write_tda_files, 
	INT f_include_group_order, 
	INT f_pic, 
	INT f_include_tda_scheme, 
	INT verbose_level);
void incidence_structure_compute_TDO_TDA(incidence_structure *Inc, 
	INT f_tda_files, 
	INT f_tda_with_group_order, 
	INT f_tda_with_scheme, 
	INT f_pic, 
	INT &TDO_ht, INT &TDA_ht, 
	INT verbose_level);
INT incidence_structure_find_blocking_set(incidence_structure *Inc, INT input_no, 
	INT *blocking_set, INT &blocking_set_size, 
	INT blocking_set_starter_size, 
	INT f_all_blocking_sets, 
	INT f_blocking_set_size_desired, INT blocking_set_size_desired, 
	INT verbose_level);

// ####################################################################################
// projective_space.C:
// ####################################################################################


//void init_orthogonal(action *A, INT epsilon, INT n, finite_field *F, INT verbose_level);
void Hill_cap56(int argc, const char **argv, 
	BYTE *fname, INT &nb_Pts, INT *&Pts, 
	INT verbose_level);
void append_orbit_and_adjust_size(schreier *Orb, INT idx, INT *set, INT &sz);
INT test_if_arc(finite_field *Fq, INT *pt_coords, INT *set, INT set_sz, INT k, INT verbose_level);
void create_Buekenhout_Metz(
	finite_field *Fq, finite_field *FQ, 
	INT f_classical, INT f_Uab, INT parameter_a, INT parameter_b, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);


// ####################################################################################
// factor_group.C:
// ####################################################################################


struct factor_group {
	INT goi;
	action *A;
	sims *S;
	INT size_subgroup;
	INT *subgroup;
	INT *all_cosets;
	INT nb_cosets;
	action *ByRightMultiplication;
	action *FactorGroup;
	action *FactorGroupConjugated;
	INT goi_factor_group;
};

void create_factor_group(action *A, sims *S, INT goi, 
	INT size_subgroup, INT *subgroup, factor_group *F, INT verbose_level);


// in analyze_group.C:
void analyze_group(action *A, sims *S, vector_ge *SG, vector_ge *gens2, INT verbose_level);
void compute_regular_representation(action *A, sims *S, vector_ge *SG, INT *&perm, INT verbose_level);
void presentation(action *A, sims *S, INT goi, vector_ge *gens, INT *primes, INT verbose_level);




// ####################################################################################
// decomposition.C:
// ####################################################################################


void decomposition_projective_space(INT k, finite_field *F, INT nb_subsets, INT *sz, INT **subsets, 
	//INT f_semilinear, INT f_basis, 
	INT verbose_level);


// ####################################################################################
// search_blocking_set.C:
// ####################################################################################



class search_blocking_set {
public:
	incidence_structure *Inc; // do not free
	action *A; // do not free
	generator *gen;

	fancy_set *Line_intersections; // [Inc->nb_cols]
	INT *blocking_set;
	INT blocking_set_len;
	INT *sz; // [Inc->nb_cols]
	
	fancy_set *active_set;
	INT *sz_active_set; // [Inc->nb_cols + 1]

	deque<vector<int> > solutions;
	INT nb_solutions;
	INT f_find_only_one;
	INT f_blocking_set_size_desired;
	INT blocking_set_size_desired;

	INT max_search_depth;
	INT *search_nb_candidates;
	INT *search_cur;
	INT **search_candidates;
	INT **save_sz;

	
	search_blocking_set();
	~search_blocking_set();
	void null();
	void freeself();
	void init(incidence_structure *Inc, action *A, INT verbose_level);
	void find_partial_blocking_sets(INT depth, INT verbose_level);
	INT test_level(INT depth, INT verbose_level);
	INT test_blocking_set(INT len, INT *S, INT verbose_level);
	INT test_blocking_set_upper_bound_only(INT len, INT *S, INT verbose_level);
	void search_for_blocking_set(INT input_no, INT level, INT f_all, INT verbose_level);
	INT recursive_search_for_blocking_set(INT input_no, INT starter_level, INT level, INT verbose_level);
	void save_line_intersection_size(INT level);
	void restore_line_intersection_size(INT level);
};

INT callback_check_partial_blocking_set(INT len, INT *S, void *data, INT verbose_level);

// ####################################################################################
// choose_points_or_lines.C:
// ####################################################################################

class choose_points_or_lines {

public:
	BYTE label[1000];
	INT t0;
	
	void *data;

	action *A;
	action *A_lines;
	action *A2;
		// = A if f_choose_lines is FALSE
		// = A_lines if f_choose_lines is TRUE
	
	INT f_choose_lines;
		// TRUE if we are looking for a set of lines
		// FALSE if we are looking for a set of points
	INT nb_points_or_lines;
		// the size of the set we are looking for

	INT print_generators_verbose_level;


	INT *transporter;
		// maps the canonical rep to the favorite rep
	INT *transporter_inv;
		// maps the favorite rep to the canonical rep 


	INT (*check_function)(INT len, INT *S, void *data, INT verbose_level);
	generator *gen;

	INT nb_orbits;
	INT current_orbit;

	INT f_has_favorite;
	INT f_iso_test_only; // do not change to favorite
	INT *favorite;
	INT favorite_size;

	INT f_has_orbit_select;
	INT orbit_select;
	


	
	INT *representative; // [nb_points_or_lines]

	longinteger_object *stab_order;
	sims *stab;
	strong_generators *Stab_Strong_gens;


	choose_points_or_lines();
	~choose_points_or_lines();
	void null();
	void freeself();
	void null_representative();
	void free_representative();
	void init(const BYTE *label, void *data, 
		action *A, action *A_lines, 
		INT f_choose_lines, 
		INT nb_points_or_lines, 
		INT (*check_function)(INT len, INT *S, void *data, INT verbose_level), 
		INT t0, 
		INT verbose_level);
	void compute_orbits_from_sims(sims *G, INT verbose_level);
	void compute_orbits(strong_generators *Strong_gens, INT verbose_level);
	void choose_orbit(INT orbit_no, INT &f_hit_favorite, INT verbose_level);
	INT favorite_orbit_representative(INT *transporter, INT *transporter_inv, 
		INT *the_favorite_representative, 
		INT verbose_level);
	void print_rep();
	void print_stab();
	INT is_in_rep(INT a);
	
};


// ####################################################################################
// subspace_orbits.C
// ####################################################################################


class subspace_orbits {
public:

	linear_group *LG;
	INT n;
	finite_field *F;
	INT q;
	INT depth;

	INT f_print_generators;
	

	INT *tmp_M; // [n * n]
	INT *tmp_M2; // [n * n]
	INT *tmp_M3; // [n * n]
	INT *base_cols; // [n]
	INT *v; // [n]
	INT *w; // [n]
	INT *weights; // [n + 1]

	generator *Gen;

	INT schreier_depth;
	INT f_use_invariant_subset_if_available;
	INT f_implicit_fusion;
	INT f_debug;

	INT f_has_strong_generators;
	strong_generators *Strong_gens;

	INT f_has_extra_test_func;
	INT (*extra_test_func)(subspace_orbits *SubOrb, 
		INT len, INT *S, void *data, INT verbose_level);
	void *extra_test_func_data;

	INT test_dim;


	subspace_orbits();
	~subspace_orbits();
	void init(int argc, const char **argv, 
		linear_group *LG, INT depth, 
		INT verbose_level);
	void init_group(INT verbose_level);
	void compute_orbits(INT verbose_level);
	void unrank_set_to_M(INT len, INT *S);
	void unrank_set_to_matrix(INT len, INT *S, INT *M);
	void rank_set_from_matrix(INT len, INT *S, INT *M);
	void Kramer_Mesner_matrix(INT t, INT k, INT f_print_matrix, 
		INT f_read_solutions, const BYTE *solution_fname, INT verbose_level);
	void print_all_solutions(diophant *D, INT k, INT *Sol, INT nb_sol, 
		INT **Subspace_ranks, INT &nb_subspaces, INT verbose_level);
	void print_one_solution(diophant *D, INT k, INT *sol, 
		INT *&subspace_ranks, INT &nb_subspaces, INT verbose_level);
	INT test_dim_C_cap_Cperp_property(INT len, INT *S, INT d);
	INT compute_minimum_distance(INT len, INT *S);
	void print_set(INT len, INT *S);
	INT test_set(INT len, INT *S, INT verbose_level);
	INT test_minimum_distance(INT len, INT *S, INT mindist, INT verbose_level);
	INT test_if_self_orthogonal(INT len, INT *S, INT f_doubly_even, INT verbose_level);
};


INT subspace_orbits_rank_point_func(INT *v, void *data);
void subspace_orbits_unrank_point_func(INT *v, INT rk, void *data);
//INT subspace_orbits_test_func(ostream &ost, INT len, INT *S, void *data, INT verbose_level);
void subspace_orbits_early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);


// ####################################################################################
// orbit_rep.C
// ####################################################################################

class orbit_rep {
public:
	BYTE prefix[1000];
	action *A;
	void (*early_test_func_callback)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level);
	void *early_test_func_callback_data;
	
	INT level;
	INT orbit_at_level;
	INT nb_cases;

	INT *rep;

	sims *Stab;
	strong_generators *Strong_gens;

	longinteger_object *stab_go;
	INT *candidates;
	INT nb_candidates;


	orbit_rep();
	~orbit_rep();
	void null();
	void freeself();
	void init_from_file(action *A, BYTE *prefix, 
		INT level, INT orbit_at_level, INT level_of_candidates_file, 
		void (*early_test_func_callback)(INT *S, INT len, 
			INT *candidates, INT nb_candidates, 
			INT *good_candidates, INT &nb_good_candidates, 
			void *data, INT verbose_level), 
		void *early_test_func_callback_data, 
		INT verbose_level);
	
};


// ####################################################################################
// singer_cycle.C
// ####################################################################################

class singer_cycle {
public:	
	finite_field *F;
	action *A;
	action *A2;
	INT n;
	INT q;
	INT *poly_coeffs; // of degree n
	INT *Singer_matrix;
	INT *Elt;
	vector_ge *gens;
	projective_space *P;
	INT *singer_point_list;
	INT *singer_point_list_inv;
	schreier *Sch;
	INT nb_line_orbits;
	INT *line_orbit_reps;
	INT *line_orbit_len;
	INT *line_orbit_first;
	BYTE **line_orbit_label;
	BYTE **line_orbit_label_tex;
	INT *line_orbit;
	INT *line_orbit_inv;

	singer_cycle();
	~singer_cycle();
	void null();
	void freeself();
	void init(INT n, finite_field *F, action *A, action *A2, INT verbose_level);
	void init_lines(INT verbose_level);
};


// ####################################################################################
// orbit_of_sets.C
// ####################################################################################

class orbit_of_sets {
public:
	action *A;
	action *A2;
	vector_ge *gens;
	INT *set;
	INT sz;

	INT position_of_original_set;
	INT allocation_length;
	INT used_length;
	INT **Sets;

	orbit_of_sets();
	~orbit_of_sets();
	void null();
	void freeself();
	void init(action *A, action *A2, INT *set, INT sz, vector_ge *gens, INT verbose_level);
	void compute(INT verbose_level);
	void get_table_of_orbits(INT *&Table, INT &orbit_length, INT &set_size, INT verbose_level);
};

INT orbit_of_sets_compare_func(void *a, void *b, void *data);



// ####################################################################################
// exact_cover.C
// ####################################################################################

class exact_cover {
public:

	BYTE input_prefix[1000];
	BYTE output_prefix[1000];
	BYTE solution_prefix[1000];
	BYTE base_fname[1000];

	BYTE fname_solutions[1000];
	BYTE fname_statistics[1000];

	void *user_data;

	action *A_base;
	action *A_on_blocks;
	

	void (*prepare_function_new)(exact_cover *E, INT starter_case, 
		INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_label, 
		INT &f_ruled_out, 
		INT verbose_level);



	void (*early_test_func)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level);
	void *early_test_func_data;

	INT f_has_solution_test_func;
	INT (*solution_test_func)(exact_cover *EC, INT *S, INT len, void *data, INT verbose_level);
	void *solution_test_func_data;

	INT f_has_late_cleanup_function;
	void (*late_cleanup_function)(exact_cover *E, INT starter_case, INT verbose_level);


	INT target_size;
	
	INT f_lex;
	
	INT f_split;
	INT split_r;
	INT split_m;

	INT f_single_case;
	INT single_case;

	INT starter_size;
	INT starter_nb_cases;
	INT *starter; // [starter_size]

	INT f_randomized;
	const BYTE *random_permutation_fname;
	INT *random_permutation;

	exact_cover();
	~exact_cover();
	void null();
	void freeself();
	void init_basic(void *user_data, 
		action *A_base, action *A_on_blocks, 
		INT target_size, INT starter_size, 
		const BYTE *input_prefix, const BYTE *output_prefix, const BYTE *solution_prefix, const BYTE *base_fname, 
		INT f_lex, 
		INT verbose_level);

	void init_early_test_func(
		void (*early_test_func)(INT *S, INT len, 
			INT *candidates, INT nb_candidates, 
			INT *good_candidates, INT &nb_good_candidates, 
			void *data, INT verbose_level), 
		void *early_test_func_data,
		INT verbose_level);
	void init_prepare_function_new(
		void (*prepare_function_new)(exact_cover *E, INT starter_case, 
			INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
			diophant *&Dio, INT *&col_label, 
			INT &f_ruled_out, 
			INT verbose_level),
		INT verbose_level);
	void set_split(INT split_r, INT split_m, INT verbose_level);
	void set_single_case(INT single_case, INT verbose_level);
	void randomize(const BYTE *random_permutation_fname, INT verbose_level);
	void add_solution_test_function(
		INT (*solution_test_func)(exact_cover *EC, INT *S, INT len, void *data, INT verbose_level), 
		void *solution_test_func_data,
		INT verbose_level);
	void add_late_cleanup_function(
		void (*late_cleanup_function)(exact_cover *E, INT starter_case, INT verbose_level)
		);
	void compute_liftings_new(INT f_solve, INT f_save, INT f_read_instead, 
		INT f_draw_system, const BYTE *fname_system, 
		INT f_write_tree, const BYTE *fname_tree, INT verbose_level);
	void compute_liftings_single_case_new(INT starter_case, 
		INT f_solve, INT f_save, INT f_read_instead, 
		INT &nb_col, 
		INT *&Solutions, INT &sol_length, INT &nb_sol, INT &nb_backtrack, INT &dt, 
		INT f_draw_system, const BYTE *fname_system, 
		INT f_write_tree, const BYTE *fname_tree, 
		INT verbose_level);
	void lexorder_test(INT *live_blocks2, INT &nb_live_blocks2, vector_ge *stab_gens, 
		INT verbose_level);
};

// ####################################################################################
// recoordinatize.C
// ####################################################################################

class recoordinatize {
public:
	//translation_plane *T;
	INT n;
	INT k;
	INT q;
	grassmann *Grass;
	finite_field *F;
	action *A; // P Gamma L(n,q) 
	action *A2; // action of A on grassmannian of k-subspaces of V(n,q)
	INT f_projective;
	INT f_semilinear;
	INT nCkq; // n choose k in q
	INT (*check_function_incremental)(INT len, INT *S, void *check_function_incremental_data, INT verbose_level);
	void *check_function_incremental_data;


	INT f_data_is_allocated;
	INT *M;
	INT *M1;
	INT *AA;
	INT *AAv;
	INT *TT;
	INT *TTv;
	INT *B;
	INT *C;
	INT *N;
	INT *Elt;

	// initialized in compute_starter():
	INT starter_j1, starter_j2, starter_j3;
	action *A0;	// P Gamma L(k,q)
	action *A0_linear; // PGL(k,q), needed for compute_live_points
	vector_ge *gens2;

	INT *live_points;
	INT nb_live_points;


	recoordinatize();
	~recoordinatize();
	void null();
	void freeself();
	void init(INT n, INT k, finite_field *F, grassmann *Grass, action *A, action *A2, 
		INT f_projective, INT f_semilinear, 
		INT (*check_function_incremental)(INT len, INT *S, void *data, INT verbose_level), 
		void *check_function_incremental_data, 
		INT verbose_level);
	void do_recoordinatize(INT i1, INT i2, INT i3, INT verbose_level);
	void compute_starter(INT *&S, INT &size, 
		strong_generators *&Strong_gens, INT verbose_level);
	void stabilizer_of_first_three(strong_generators *&Strong_gens, INT verbose_level);
	void compute_live_points(INT verbose_level);
	void compute_live_points_low_level(INT *&live_points, INT &nb_live_points, INT verbose_level);
	void make_first_three(INT &j1, INT &j2, INT &j3, INT verbose_level);
};


// ####################################################################################
// translation_plane.C
// ####################################################################################

#define SPREAD_OF_TYPE_FTWKB 1
#define SPREAD_OF_TYPE_KANTOR 2
#define SPREAD_OF_TYPE_KANTOR2 3
#define SPREAD_OF_TYPE_GANLEY 4
#define SPREAD_OF_TYPE_LAW_PENTTILA 5
#define SPREAD_OF_TYPE_DICKSON_KANTOR 6
#define SPREAD_OF_TYPE_HUDSON 7


class translation_plane {
public:

	finite_field *F;

	int argc;
	const char **argv;

	INT order;
	INT spread_size; // = order + 1
	INT n; // = 2 * k
	INT k;
	INT kn; // = k * n
	INT q;
	INT nCkq; // n choose k in q
	INT r, nb_pts;
	INT nb_points_total; // = nb_pts = {n choose 1}_q
	INT block_size; // = r = {k choose 1}_q
	INT max_depth;

	INT f_print_generators;
	INT f_projective;
	INT f_semilinear;
	INT f_basis;
	INT f_induce_action;
	INT f_override_schreier_depth;
	INT override_schreier_depth;

	BYTE starter_directory_name[1000];
	BYTE prefix[1000];
	//BYTE prefix_with_directory[1000];
	INT starter_size;


	// allocated in init();
	action *A; // P Gamma L(n,q) 
	action *A2; // action of A on grassmannian of k-subspaces of V(n,q)
	action_on_grassmannian *AG;
	grassmann *Grass; // {n choose k}_q


	INT f_recoordinatize;
	recoordinatize *R;

	// if f_recoordinatize is TRUE:
	INT *Starter, Starter_size;
	strong_generators *Starter_Strong_gens;

	// for check_function_incremental:
	INT *tmp_M1;
	INT *tmp_M2;
	INT *tmp_M3;
	INT *tmp_M4;


	generator *gen; // allocated in init()


	singer_cycle *Sing;


	// if k = 2 only:
	klein_correspondence *Klein;
	orthogonal *O;


	INT Nb;
	INT *Data1; // [max_depth * kn], previously [Nb * n], which was too much
	INT *Data2; // [n * n]


	translation_plane();
	~translation_plane();
	void null();
	void freeself();
	void init(INT order, INT n, INT k, INT max_depth, 
		finite_field *F, INT f_recoordinatize, 
		const BYTE *input_prefix, 
		const BYTE *base_fname,
		INT starter_size,  
		int argc, const char **argv, 
		INT verbose_level);
	void print_points();
	void print_elements();
	void print_elements_and_points();
	void read_arguments(int argc, const char **argv);
	void init2(INT verbose_level);
	void compute(INT verbose_level);
	//void identify(INT *data, INT sz, INT *transporter, INT &orbit_at_level, INT verbose_level);
	//void test_identify(INT level, INT nb_times, INT verbose_level);
	void early_test_func(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	INT check_function(INT len, INT *S, INT verbose_level);
	INT check_function_incremental(INT len, INT *S, INT verbose_level);
	INT check_function_pair(INT rk1, INT rk2, INT verbose_level);
	void lifting_prepare_function_new(exact_cover *E, INT starter_case, 
		INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_labels, 
		INT &f_ruled_out, 
		INT verbose_level);
	void compute_dual_spread(INT *spread, INT *dual_spread, INT verbose_level);


	// translation_plane2.C:
	void print_isomorphism_type(isomorph *Iso, 
		INT iso_cnt, sims *Stab, schreier &Orb, 
		INT *data, INT verbose_level);
		// called from callback_print_isomorphism_type()
#if 0
	void klein_plane_intersections(INT *data, INT data_size, 
		longinteger_object *&R,
		INT **&Pts_on_plane, 
		INT *&nb_pts_on_plane, 
		INT &nb_planes, 
		INT verbose_level);
#endif
	void save_klein_invariants(BYTE *prefix, 
		INT iso_cnt, 
		INT *data, INT data_size, INT verbose_level);
	void klein(ofstream &ost, 
		isomorph *Iso, 
		INT iso_cnt, sims *Stab, schreier &Orb, 
		INT *data, INT data_size, INT verbose_level);
#if 0
	void test_plane_intersection_type_of_klein_image(
		const BYTE *fname, INT verbose_level);
#endif
	void plane_intersection_type_of_klein_image(
		projective_space *P3, 
		projective_space *P5, 
		grassmann *Gr, 
		INT *data, INT size, 
		INT *&intersection_type, INT &highest_intersection_number, 
		INT verbose_level);

	void czerwinski_oakden(INT level, INT verbose_level);
	void write_spread_to_file(INT type_of_spread, INT verbose_level);
	void make_spread(INT *data, INT type_of_spread, INT verbose_level);
	void make_spread_from_q_clan(INT *data, INT type_of_spread, INT verbose_level);
	void read_and_print_spread(const BYTE *fname, INT verbose_level);
	void HMO(const BYTE *fname, INT verbose_level);
	void get_spread_matrices(INT *F, INT *G, INT *data, INT verbose_level);
	void print_spread(INT *data, INT sz);
	void report2(isomorph &Iso, INT verbose_level);
	void all_cooperstein_thas_quotients(isomorph &Iso, INT verbose_level);
	void cooperstein_thas_quotients(isomorph &Iso, ofstream &f, INT h, INT &cnt, INT verbose_level);
	void orbit_info_short(ofstream &f, isomorph &Iso, INT h);
	void report_stabilizer(isomorph &Iso, ofstream &f, INT orbit, INT verbose_level);
	void print(INT len, INT *S);
};


void translation_plane_lifting_early_test_function(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
void translation_plane_lifting_prepare_function_new(exact_cover *EC, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level);
INT starter_canonize_callback(INT *Set, INT len, INT *Elt, void *data, INT verbose_level);
INT translation_plane_check_function_incremental(INT len, INT *S, void *data, INT verbose_level);


// translation_plane2.C:
void translation_plane_early_test_func_callback(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
INT translation_plane_check_function_callback(INT len, INT *S, void *data, INT verbose_level);
INT translation_plane_check_function_incremental_callback(INT len, INT *S, void *data, INT verbose_level);
INT translation_plane_check_conditions(INT len, INT *S, void *data, INT verbose_level);
void translation_plane_callback_report(isomorph *Iso, void *data, INT verbose_level);
void translation_plane_callback_make_quotients(isomorph *Iso, void *data, INT verbose_level);


// ####################################################################################
// polar.C: (added June 1, 2010)
// ####################################################################################

	

class polar {
public:
	INT epsilon;
	INT n; // vector space dimension
	INT k;
	INT q;
	INT depth;

	INT f_print_generators;

	action *A; // the orthogonal action

#if 0
	action_on_grassmannian *AG;
			// on k-dimensional subspaces of GF(q)^n,
			// not needed
#endif

	
	matrix_group *Mtx; // only a copy of a pointer, not to be freed
	orthogonal *O; // only a copy of a pointer, not to be freed
	finite_field *F; // only a copy of a pointer, not to be freed

	INT *tmp_M; // [n * n]
	INT *base_cols; // [n]

	generator *Gen;

	INT schreier_depth;
	INT f_use_invariant_subset_if_available;
	//INT f_implicit_fusion;
	INT f_debug;

	INT f_has_strong_generators;
	INT f_has_strong_generators_allocated;
	strong_generators *Strong_gens;

	INT first_node, nb_orbits, nb_elements;
	
	polar();
	~polar();
	void init_group_by_base_images(INT *group_generator_data, INT group_generator_size, 
		INT f_group_order_target, const BYTE *group_order_target, INT verbose_level);
	void init_group(INT *group_generator_data, INT group_generator_size, 
		INT f_group_order_target, const BYTE *group_order_target, INT verbose_level);
	void init(int argc, const char **argv, action *A, orthogonal *O, 
		INT epsilon, INT n, INT k, finite_field *F, INT depth, INT verbose_level);
	void init2(INT verbose_level);
	void compute_orbits(INT t0, INT verbose_level);
	void compute_cosets(INT depth, INT orbit_idx, INT verbose_level);
	void dual_polar_graph(INT depth, INT orbit_idx, 
		longinteger_object *&Rank_table, INT &nb_maximals, INT verbose_level);
	void show_stabilizer(INT depth, INT orbit_idx, INT verbose_level);
	void compute_Kramer_Mesner_matrix(INT t, INT k, INT verbose_level);
	INT test(INT *S, INT len, INT verbose_level);
		// test if totally isotropic, i.e. contained in its own perp
	void test_if_in_perp(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	void test_if_closed_under_cosets(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	void get_stabilizer(INT orbit_idx, group &G, longinteger_object &go_G);
	void get_orbit_length(INT orbit_idx, longinteger_object &length);
	INT get_orbit_length_as_INT(INT orbit_idx);
	void orbit_element_unrank(INT orbit_idx, INT rank, INT *set, INT verbose_level);
	void orbit_element_rank(INT &orbit_idx, INT &rank, INT *set, INT verbose_level);
	void unrank_point(INT *v, INT rk);
	INT rank_point(INT *v);
	void list_whole_orbit(INT depth, INT orbit_idx, INT f_limit, INT limit);
};


INT polar_callback_rank_point_func(INT *v, void *data);
void polar_callback_unrank_point_func(INT *v, INT rk, void *data);
INT polar_callback_test_func(INT len, INT *S, void *data, INT verbose_level);
void polar_callback_early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);


// ####################################################################################
// orbit_of_subspaces.C
// ####################################################################################

class orbit_of_subspaces {
public:
	action *A;
	action *A2;
	finite_field *F;
	vector_ge *gens;
	INT k;
	INT n;
	INT kn;
	INT sz; // = 1 + k + kn
	INT sz_for_compare; // = 1 + k + kn
	INT f_has_desired_pivots;
	INT *desired_pivots; // [k]
	INT *subspace_by_rank; // [k]

	INT f_has_rank_functions;
	void *rank_unrank_data;
	INT (*rank_vector_callback)(INT *v, INT n, void *data, INT verbose_level);
	void (*unrank_vector_callback)(INT rk, INT *v, INT n, void *data, INT verbose_level);
	void (*compute_image_of_vector_callback)(INT *v, INT *w, INT *Elt, void *data, INT verbose_level);
	void *compute_image_of_vector_callback_data;

	INT position_of_original_subspace;
	INT allocation_length;
	INT used_length;
	INT **Subspaces;
	INT *prev;
	INT *label;


	orbit_of_subspaces();
	~orbit_of_subspaces();
	void null();
	void freeself();
	void init(action *A, action *A2, finite_field *F, 
		INT *subspace, INT k, INT n, 
		INT f_has_desired_pivots, INT *desired_pivots, 
		INT f_has_rank_functions, void *rank_unrank_data, 
		INT (*rank_vector_callback)(INT *v, INT n, void *data, INT verbose_level), 
		void (*unrank_vector_callback)(INT rk, INT *v, INT n, void *data, INT verbose_level), 
		void (*compute_image_of_vector_callback)(INT *v, INT *w, INT *Elt, void *data, INT verbose_level), 
		void *compute_image_of_vector_callback_data, 
		vector_ge *gens, INT verbose_level);
	INT rank_vector(INT *v, INT verbose_level);
	void unrank_vector(INT rk, INT *v, INT verbose_level);
	void rref(INT *subspace, INT verbose_level);
	void rref_and_rank_and_hash(INT *subspace, INT verbose_level);
	void map_a_subspace(INT *subspace, INT *image_subspace, INT *Elt, INT verbose_level);
	void map_a_basis(INT *basis, INT *image_basis, INT *Elt, INT verbose_level);
	void print_orbit();
	void compute(INT verbose_level);
	void get_transporter(INT idx, INT *transporter, INT verbose_level);
	void get_random_schreier_generator(INT *Elt, INT verbose_level);
	void compute_stabilizer(action *default_action, longinteger_object &go, 
		sims *&Stab, INT verbose_level);
		// this function allocates a sims structure into Stab.
	INT search_data(INT *data, INT &idx);
};

INT orbit_of_subspaces_compare_func(void *a, void *b, void *data);

// ####################################################################################
// young.C
// ####################################################################################

class young {
public:
	INT n;
	action *A;
	sims *S;
	longinteger_object go;
	INT goi;
	INT *Elt;
	INT *v;

	action *Aconj;
	action_by_conjugation *ABC;
	schreier *Sch;
	strong_generators *SG;
	INT nb_classes;
	INT *class_size;
	INT *class_rep;
	a_domain *D;

	INT l1, l2;
	INT *row_parts;
	INT *col_parts;
	INT *Tableau;

	set_of_sets *Row_partition;
	set_of_sets *Col_partition;

	vector_ge *gens1, *gens2;
	sims *S1, *S2;


	young();
	~young();
	void null();
	void freeself();
	void init(INT n, INT verbose_level);
	void create_module(INT *h_alpha, 
		INT *&Base, INT *&base_cols, INT &rk, 
		INT verbose_level);
	void create_representations(INT *Base, INT *Base_inv, INT rk, INT verbose_level);
	void create_representation(INT *Base, INT *base_cols, INT rk, INT group_elt, INT *Mtx, INT verbose_level);
		// Mtx[rk * rk * D->size_of_instance_in_INT]
	void young_symmetrizer(INT *row_parts, INT nb_row_parts, 
		INT *tableau, 
		INT *elt1, INT *elt2, INT *elt3, 
		INT verbose_level);
	void compute_generators(INT &go1, INT &go2, INT verbose_level);
	void Maschke(INT *Rep, 
		INT dim_of_module, INT dim_of_submodule, 
		INT *&Mu, 
		INT verbose_level);
};

// ####################################################################################
// exact_cover_arguments.C:
// ####################################################################################



class exact_cover_arguments {
public:
	action *A;
	action *A2;
	void *user_data;
	INT f_has_base_fname;
	const BYTE *base_fname;
	INT f_has_input_prefix;
	const BYTE *input_prefix;
	INT f_has_output_prefix;
	const BYTE *output_prefix;
	INT f_has_solution_prefix;
	const BYTE *solution_prefix;
	INT f_lift;
	INT f_starter_size;
	INT starter_size;
	INT target_size;
	INT f_lex;
	INT f_split;
	INT split_r;
	INT split_m;
	INT f_solve;
	INT f_save;
	INT f_read;
	INT f_draw_system;
	const BYTE *fname_system;
	INT f_write_tree;
	const BYTE *fname_tree;
	void (*prepare_function_new)(exact_cover *E, INT starter_case, 
		INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_label, 
		INT &f_ruled_out, 
		INT verbose_level);
	void (*early_test_function)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level);
	void *early_test_function_data;
	INT f_has_solution_test_function;
	INT (*solution_test_func)(exact_cover *EC, INT *S, INT len, void *data, INT verbose_level);
	void *solution_test_func_data;
	INT f_has_late_cleanup_function;
	void (*late_cleanup_function)(exact_cover *EC, INT starter_case, INT verbose_level);

	INT f_randomized;
	const BYTE *random_permutation_fname;

	exact_cover_arguments();
	~exact_cover_arguments();
	void null();
	void freeself();
	void read_arguments(int argc, const char **argv, 
		INT verbose_level);
	void compute_lifts(INT verbose_level);
};



// ##################################################################################################
// translation_plane_via_andre_model.C
// ##################################################################################################


class translation_plane_via_andre_model {
public:
	finite_field *F;
	INT q;
	INT k;
	INT n;
	INT k1;
	INT n1;
	
	andre_construction *Andre;
	INT N; // number of points = number of lines
	INT twoN; // 2 * N
	INT f_semilinear;

	andre_construction_line_element *Line;
	INT *Incma;
	INT *pts_on_line;
	INT *Line_through_two_points; // [N * N]
	INT *Line_intersection; // [N * N]

	action *An;
	action *An1;

	action *OnAndre;

	strong_generators *strong_gens;

	incidence_structure *Inc;
	partitionstack *Stack;

	generator *arcs;

	translation_plane_via_andre_model();
	~translation_plane_via_andre_model();
	void null();
	void freeself();
	void init(INT *spread_elements_numeric, INT k, finite_field *F, 
		vector_ge *spread_stab_gens, longinteger_object &spread_stab_go, INT verbose_level);
	void classify_arcs(const BYTE *prefix, INT depth, INT verbose_level);
	void classify_subplanes(const BYTE *prefix, INT verbose_level);
	INT check_arc(INT *S, INT len, INT verbose_level);
	INT check_subplane(INT *S, INT len, INT verbose_level);
	INT check_if_quadrangle_defines_a_subplane(INT *S, INT *subplane7, INT verbose_level);
};


INT translation_plane_via_andre_model_check_arc(INT len, INT *S, void *data, INT verbose_level);
INT translation_plane_via_andre_model_check_subplane(INT len, INT *S, void *data, INT verbose_level);


// ##################################################################################################
// isomorph_arguments.C
// ##################################################################################################

class isomorph_arguments {
public:
	INT f_init_has_been_called;
	
	INT f_build_db;
	INT f_read_solutions;
	INT f_read_solutions_from_clique_finder;
	INT f_read_solutions_from_clique_finder_list_of_cases;
	const BYTE *fname_list_of_cases;
	INT f_read_solutions_after_split;
	INT read_solutions_split_m;
	
	INT f_read_statistics_after_split;
	INT read_statistics_split_m;

	INT f_compute_orbits;
	INT f_isomorph_testing;
	INT f_classification_graph;
	INT f_event_file; // -e <event file> option
	const BYTE *event_file_name;
	INT print_mod;
	INT f_report;
	INT f_subset_orbits;
	INT f_subset_orbits_file;
	const BYTE *subset_orbits_fname;
	INT f_eliminate_graphs_if_possible;
	INT f_down_orbits;

	const BYTE *prefix_iso;

	action *A;
	action *A2;
	generator *gen;
	INT target_size;
	const BYTE *prefix_with_directory;
	exact_cover_arguments *ECA;
	
	void (*callback_report)(isomorph *Iso, void *data, INT verbose_level);
	void (*callback_subset_orbits)(isomorph *Iso, void *data, INT verbose_level);
	void *callback_data;

	INT f_has_final_test_function;
	INT (*final_test_function)(INT *data, INT sz, void *final_test_data, INT verbose_level);
	void *final_test_data;
	
	isomorph_arguments();
	~isomorph_arguments();
	void null();
	void freeself();
	void read_arguments(int argc, const char **argv, 
		INT verbose_level);
	void init(action *A, action *A2, generator *gen, 
		INT target_size, const BYTE *prefix_with_directory, exact_cover_arguments *ECA, 
		void (*callback_report)(isomorph *Iso, void *data, INT verbose_level), 
		void (*callback_subset_orbits)(isomorph *Iso, void *data, INT verbose_level), 
		void *callback_data, 
		INT verbose_level);
	void execute(INT verbose_level);

};

struct isomorph_worker_data {
	INT *the_set;
	INT set_size;
	void *callback_data;
};

// ####################################################################################
// elliptic_curve.C:
// ####################################################################################


// needs sqrt_mod_involved from DISCRETA/global.C

class elliptic_curve {

public:
	INT q;
	INT p;
	INT e;
	INT b, c; // the equation of the curve is Y^2 = X^3 + bX + c mod p
	INT nb; // number of points
	INT *T; // [nb * 3] point coordinates
		// the point at inifinity is last
	INT *A; // [nb * nb] addition table
	finite_field *F;


	elliptic_curve();
	~elliptic_curve();
	void null();
	void freeself();
	void init(finite_field *F, INT b, INT c, INT verbose_level);
	void compute_points(INT verbose_level);
	void add_point_to_table(INT x, INT y, INT z);
	INT evaluate_RHS(INT x);
		// evaluates x^3 + bx + c
	void print_points();
	void print_points_affine();
	void addition(
		INT x1, INT x2, INT x3, 
		INT y1, INT y2, INT y3, 
		INT &z1, INT &z2, INT &z3, INT verbose_level);
	void draw_grid(char *fname, INT xmax, INT ymax, INT f_with_points, INT verbose_level);
	void draw_grid2(mp_graphics &G, INT f_with_points, INT verbose_level);
	void make_affine_point(INT x1, INT x2, INT x3, INT &a, INT &b, INT verbose_level);
	void compute_addition_table(INT verbose_level);
	void print_addition_table();
	INT index_of_point(INT x1, INT x2, INT x3);
	INT order_of_point(INT i);
	void print_all_powers(INT i);
};


#if 0
// ####################################################################################
// surface_classify.C
// ####################################################################################

class surface_classify {
public:
	finite_field *F;
	INT q;
	action *A;
	orthogonal *O;
	generator *gen;

	INT *u, *v; // temporary vectors of length 6

	klein_correspondence *Klein;
	surface *Surf;
	schreier *Sch;
	longinteger_object go;
	sims *Stab;
	strong_generators *stab_gens;
	INT pt;
	INT *Pts;
	INT nb_pts;
	action *A_on_neighbors;

	INT len; // number of orbits on 5-sets of lines
	INT *Idx;
	INT nb; // number of good orbits

	surface_data **The_surface;
	INT *Nb_points;
	INT *Nb_lines;
	INT *Nb_double_six;
	INT *Nb_E;
	INT *is_isomorphic_to;
	INT *Ago0;
	INT *Ago_nb_cosets;
	INT *Ago;
	INT *Orb;
	INT **Coeffs;
	INT nb_iso;
	INT nb_E;
	INT *Elt0;
	INT *Elt1;


	surface_classify();
	~surface_classify();
	void null();
	void freeself();
	void init(finite_field *F, action *A, orthogonal *O, 
		generator *gen, 
		INT verbose_level);
	INT surface_test(INT *S, INT len, INT verbose_level);
	void process_surfaces(INT nb_identify, 
		BYTE **Identify_label, 
		INT **Identify_coeff, 
		INT **Identify_monomial, 
		INT *Identify_length, 
		INT verbose_level);
	void test_orbits(INT verbose_level);
	void make_table_of_double_sixes(INT *Lines, INT nb_lines, 
		set_of_sets *SoS, INT *&Table, INT &N, INT verbose_level);
	void classify_surfaces(INT verbose_level);
	INT new_surface(INT orb, INT iso_type, INT verbose_level);
	void identify(INT nb_identify, 
		BYTE **Identify_label, 
		INT **Identify_coeff, 
		INT **Identify_monomial, 
		INT *Identify_length, 
		INT **&Label, 
		INT *&nb_Labels, 
		INT verbose_level);
	void allocate_data();
	void free_data();


};
#endif

// ####################################################################################
// surface_classify_wedge.C
// ####################################################################################

class surface_classify_wedge {
public:
	finite_field *F;
	INT q;
	linear_group *LG;

	INT f_semilinear;

	BYTE fname_base[1000];

	action *A; // the action of PGL(4,q) on points
	action *A2; // the action on the wedge product

	surface *Surf;
	surface_with_action *Surf_A;

	INT *Elt0;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;

	classify_double_sixes *Classify_double_sixes;






	// classification of surfaces:
	flag_orbits *Flag_orbits;

	classification *Surfaces;



	INT nb_identify;
	BYTE **Identify_label;
	INT **Identify_coeff;
	INT **Identify_monomial;
	INT *Identify_length;




	

#if 0
	// this is old stuff:

	vector_ge *Isomorphisms; // [nb]

	INT nb_iso;
	surface_data **The_surface; // [nb_iso]

	INT *is_isomorphic_to; // allocated [len], used as [nb], 
		// iso_type associated to this node
		// a transporter is stored in Isomorphisms
	
	INT *Orb; // [nb], used as [nb_iso], indexed by iso_type, 
		// lists the initial orbit (index into Idx) 
		// that led to the definition of the isomorphism type

	// old stuff until here
#endif



	surface_classify_wedge();
	~surface_classify_wedge();
	void null();
	void freeself();
	//void allocate_data();
	//void free_data();
	void read_arguments(int argc, const char **argv, 
		INT verbose_level);
	void init(finite_field *F, linear_group *LG, 
		int argc, const char **argv, 
		INT verbose_level);
	void classify_surfaces_from_double_sixes(INT verbose_level);
	void downstep(INT verbose_level);
	void upstep(INT verbose_level);
	void write_file(ofstream &fp, INT verbose_level);
	void read_file(ifstream &fp, INT verbose_level);

#if 0
	void do_classify_surfaces(INT verbose_level);
	INT new_surface(INT orb, INT iso_type, INT verbose_level);
	void store_isomorphism(INT *Elt, INT orb, INT verbose_level);
	void store_identity(INT orb, INT verbose_level);
	void generate_source_code(INT verbose_level);
	void print_surfaces();
	void print_surface(set_and_stabilizer *SaS, INT verbose_level);
	void starter_configurations_which_are_involved(INT iso_type, 
		INT *&Starter_configuration_idx, INT &nb_starter_conf, INT verbose_level);
	void derived_arcs(INT verbose_level);
#endif

	void identify_surfaces(INT verbose_level);
	void identify(INT nb_identify, 
		BYTE **Identify_label, 
		INT **Identify_coeff, 
		INT **Identify_monomial, 
		INT *Identify_length, 
		//INT **&Label, 
		//INT *&nb_Labels, 
		INT verbose_level);
	void identify_surface_command_line(INT cnt, 
		//INT **&Label, INT *&nb_Labels, 
		INT &isomorphic_to, INT *Elt_isomorphism, 
		INT verbose_level);
	void identify_Sa_and_print_table(INT verbose_level);
	void identify_Sa(INT *Iso_type, INT *Nb_E, INT verbose_level);
	void identify_surface(INT *coeff_of_given_surface, 
		INT &isomorphic_to, INT *Elt_isomorphism, 
		INT verbose_level);
	void latex_surfaces(ostream &ost);
	void report_surface(ostream &ost, INT orbit_index, INT verbose_level);
	void generate_source_code(INT verbose_level);
		// no longer produces nb_E[] and single_six[]

};


#if 0
// ####################################################################################
// surface_data.C
// ####################################################################################

class surface_data {
public:
	surface_classify_wedge *SC;
	action *A;
	finite_field *F;
	INT q;
	INT nb_monomials; // = SC->Surf->nb_monomials


	INT orb;
	INT idx;
	INT iso_type;
	
	INT S[5]; 
		// the starter configuration:
		// the set of 5 lisjoint lines as indices into Neighbor
	INT S2[6]; 
		// S2[0..4] is the 5 lines a_1,...,a_5 as lines in PG(3,q). 
		// S2[5] is the transversal b_6

	INT v[6];
	INT w[6];
	
	INT f_data_allocated;
	INT *D; // [6 * 8]; the unranked version of S2[6]
	INT *coeff; // [nb_monomials]

	// the points on the surface:
	INT *Surface;
	INT nb_points_on_surface;

	// the lines on the surface:
	INT *Lines; // [27]
	INT *Lines_wedge; // [27]
	INT *Lines_klein; // [27]
	INT nb_lines;

	set_of_sets *pts_on_lines;

	incidence_structure *IS;
	partitionstack *PStack;
	INT nb_E;

	INT *Adj;

	set_of_sets *line_intersections;
	INT nb_starter_configurations;
	INT *Table;

	INT ago0;
	INT nb_cosets;
	INT nb_aut;
	vector_ge *Aut_cosets;

	INT *Elt0;
	INT *Elt1;
	INT *Elt2;

	strong_generators *stab_gens;
	longinteger_object ago;


	surface_data();
	~surface_data();
	void null();
	void freeself();
	void allocate_data();
	void free_data();
	INT init(surface_classify_wedge *SC, INT orb, INT iso_type, INT verbose_level);
	void automorphisms_and_isomorphisms(INT verbose_level);
	void automorphisms_and_isomorphisms_with_spreadsheet(spreadsheet *&Sp, INT verbose_level);
	void compute_decomposition_schemes(INT verbose_level);
	void compute_adjacency_matrix_and_line_intersections(INT verbose_level);
	void print(INT iso_type);
	void store_automorphism_coset(INT *Elt, INT verbose_level);
};
#endif

// ####################################################################################
// arc_generator.C
// ####################################################################################


class arc_generator {

public:

	INT q;
	INT f_poly;
	const BYTE *poly;
	finite_field *F;
	int argc;
	const char **argv;

	exact_cover_arguments *ECA;
	isomorph_arguments *IA;

	INT verbose_level;
	INT f_starter;
	INT f_draw_poset;
	INT f_list;
	INT list_depth;
	INT f_simeon;
	INT simeon_s;


	INT nb_points_total;
	INT f_target_size;
	INT target_size;

	BYTE starter_directory_name[1000];
	BYTE prefix[1000];
	BYTE prefix_with_directory[1000];
	INT starter_size;


	INT f_recognize;
	const BYTE *recognize[1000];
	INT nb_recognize;



	INT f_no_arc_testing;


	INT f_semilinear;

	action *A;
	
	grassmann *Grass;
	action_on_grassmannian *AG;
	action *A_on_lines;
	
	projective_space *P; // projective n-space
	
	INT f_d;
	INT d;
	INT f_n;
	INT n;
	INT *line_type; // [P2->N_lines]

		
	generator *gen;

	


	arc_generator();
	~arc_generator();
	void null();
	void freeself();
	void read_arguments(int argc, const char **argv);
	void main(INT verbose_level);
	void init(finite_field *F,
		const BYTE *input_prefix, 
		const BYTE *base_fname,
		INT starter_size,  
		int argc, const char **argv, 
		INT verbose_level);
	void prepare_generator(INT verbose_level);
	void compute_starter(INT verbose_level);

	void early_test_func(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	//INT check_arc(INT *S, INT len, INT verbose_level);
	void print(INT len, INT *S);
	void print_set_in_affine_plane(INT len, INT *S);
	void point_unrank(INT *v, INT rk);
	INT point_rank(INT *v);
	void compute_line_type(INT *set, INT len, INT verbose_level);
	void lifting_prepare_function_new(exact_cover *E, INT starter_case, 
		INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_labels, 
		INT &f_ruled_out, 
		INT verbose_level);
	// compute the incidence matrix of tangent lines versus candidate points
	// extended by external lines versus candidate points
	INT arc_test(INT *S, INT len, INT verbose_level);
	void report(isomorph &Iso, INT verbose_level);
	void report_decompositions(isomorph &Iso, ofstream &f, INT orbit, 
		INT *data, INT verbose_level);
	void report_stabilizer(isomorph &Iso, ofstream &f, INT orbit, INT verbose_level);
	void simeon(INT len, INT *S, INT s, INT verbose_level);
#if 0
	INT simeon_matrix_entry(INT *Coord, INT *C, INT *E, INT *S, INT len, INT s, 
		INT *T, INT verbose_level);
#endif
};


INT callback_arc_test(exact_cover *EC, INT *S, INT len, void *data, INT verbose_level);
INT check_arc(INT len, INT *S, void *data, INT verbose_level);
INT placebo_test_function(INT len, INT *S, void *data, INT verbose_level);
void arc_generator_early_test_function(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
void placebo_early_test_function(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
void arc_generator_lifting_prepare_function_new(exact_cover *EC, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level);
void print_arc(INT len, INT *S, void *data);
void print_point(INT pt, void *data);
void callback_arc_report(isomorph *Iso, void *data, INT verbose_level);
void arc_print(INT len, INT *S, void *data);


// ####################################################################################
// surface_with_action.C:
// ####################################################################################


class surface_with_action {

public:

	INT q;
	finite_field *F; // do not free
	INT f_semilinear;
	
	surface *Surf; // do not free

	action *A; // linear group PGGL(4,q)
	action *A2; // linear group PGGL(4,q) acting on lines
	sims *S; // linear group PGGL(4,q)

	INT *Elt1;
	
	action_on_homogeneous_polynomials *AonHPD_3_4;


	classify_trihedral_pairs *Classify_trihedral_pairs;

	recoordinatize *Recoordinatize;
	INT *regulus; // [regulus_size]
	INT regulus_size; // q + 1


	surface_with_action();
	~surface_with_action();
	void null();
	void freeself();
	void init(surface *Surf, INT f_semilinear, INT verbose_level);
	void init_group(INT f_semilinear, INT verbose_level);
	INT create_double_six_safely(
		INT *five_lines, INT transversal_line, INT *double_six, INT verbose_level);
	INT create_double_six_from_five_lines_with_a_common_transversal(
		INT *five_lines, INT transversal_line, INT *double_six, INT verbose_level);
	void arc_lifting_and_classify(INT f_log_fp, ofstream &fp, 
		INT *Arc6, 
		const BYTE *arc_label, const BYTE *arc_label_short, 
		INT nb_surfaces, 
		arc_generator *Gen, 
		INT *Arc_identify_nb, 
		INT *Arc_identify, 
		INT *f_deleted, 
		INT *Not_on_conic_idx, 
		INT nb_arcs_not_on_conic, 
		INT verbose_level);

};




// ####################################################################################
// surface_object_with_action.C:
// ####################################################################################


class surface_object_with_action {

public:

	INT q;
	finite_field *F; // do not free

	surface *Surf; // do not free
	surface_with_action *Surf_A; // do not free

	surface_object *SO; // do not free
	strong_generators *Aut_gens;  // generators for the automorphism group

	action *A_on_points;
	action *A_on_Eckardt_points;
	action *A_on_Double_points;
	action *A_on_the_lines;
	action *A_single_sixes;
	action *A_on_tritangent_planes;
	action *A_on_trihedral_pairs;
	action *A_on_pts_not_on_lines;


	schreier *Orbits_on_points;
	schreier *Orbits_on_Eckardt_points;
	schreier *Orbits_on_Double_points;
	schreier *Orbits_on_lines;
	schreier *Orbits_on_single_sixes;
	schreier *Orbits_on_tritangent_planes;
	schreier *Orbits_on_trihedral_pairs;
	schreier *Orbits_on_points_not_on_lines;



	surface_object_with_action();
	~surface_object_with_action();
	void null();
	void freeself();
	INT init_equation(surface_with_action *Surf_A, INT *eqn, 
		strong_generators *Aut_gens, INT verbose_level);
	void init(surface_with_action *Surf_A, INT *Lines, INT *eqn, 
		strong_generators *Aut_gens, INT verbose_level);
	void init_surface_object(surface_with_action *Surf_A, surface_object *SO, 
		strong_generators *Aut_gens, INT verbose_level);
	void compute_orbits_of_automorphism_group(INT verbose_level);
	void init_orbits_on_points(INT verbose_level);
	void init_orbits_on_Eckardt_points(INT verbose_level);
	void init_orbits_on_Double_points(INT verbose_level);
	void init_orbits_on_lines(INT verbose_level);
	void init_orbits_on_half_double_sixes(INT verbose_level);
	void init_orbits_on_tritangent_planes(INT verbose_level);
	void init_orbits_on_trihedral_pairs(INT verbose_level);
	void init_orbits_on_points_not_on_lines(INT verbose_level);
	void print_automorphism_group(ostream &ost);
	void compute_quartic(INT pt_orbit, 
		INT &pt_A, INT &pt_B, INT *transporter, 
		INT *equation, INT *equation_nice, INT verbose_level);
	void quartic(ostream &ost, INT verbose_level);
	void cheat_sheet(ostream &ost, 
		const BYTE *label_txt, const BYTE *label_tex, INT verbose_level);
	void cheat_sheet_quartic_curve(ostream &ost, 
		const BYTE *label_txt, const BYTE *label_tex, INT verbose_level);
};


// ####################################################################################
// classify_trihedral_pairs.C:
// ####################################################################################


class classify_trihedral_pairs {

public:

	INT q;
	finite_field *F; // do not free
	action *A; // do not free

	surface_with_action *Surf_A; // do not free
	surface *Surf; // do not free

	strong_generators *gens_type1;
	strong_generators *gens_type2;

	generator *orbits_on_trihedra_type1;
	generator *orbits_on_trihedra_type2;

	INT nb_orbits_type1;
	INT nb_orbits_type2;
	INT nb_orbits_ordered_total;

	flag_orbits *Flag_orbits;

	INT nb_orbits_trihedral_pairs;

	classification *Trihedral_pairs;



	classify_trihedral_pairs();
	~classify_trihedral_pairs();
	void null();
	void freeself();
	void init(surface_with_action *Surf_A, INT verbose_level);

	void classify_orbits_on_trihedra(INT verbose_level);
	void list_orbits_on_trihedra_type1(ostream &ost);
	void list_orbits_on_trihedra_type2(ostream &ost);
	void early_test_func_type1(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	void early_test_func_type2(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	void identify_three_planes(INT p1, INT p2, INT p3, 
		INT &type, INT *transporter, INT verbose_level);
	void classify(INT verbose_level);
	void downstep(INT verbose_level);
	void upstep(INT verbose_level);
	void print_trihedral_pairs(ostream &ost);
	strong_generators *identify_trihedral_pair_and_get_stabilizer(
		INT *planes6, INT *transporter, INT &orbit_index, INT verbose_level);
	void identify_trihedral_pair(INT *planes6, 
		INT *transporter, INT &orbit_index, INT verbose_level);

};

void classify_trihedral_pairs_early_test_function_type1(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
void classify_trihedral_pairs_early_test_function_type2(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);


// ####################################################################################
// classify_double_sixes.C:
// ####################################################################################


class classify_double_sixes {

public:

	INT q;
	finite_field *F; // do not free
	action *A; // do not free

	linear_group *LG; // do not free

	surface_with_action *Surf_A; // do not free
	surface *Surf; // do not free


	// pulled from surface_classify_wedge:

	action *A2; // the action on the wedge product
	action_on_wedge_product *AW;
		// internal data structure for the wedge action

	INT *Elt0; // used in identify_five_plus_one
	INT *Elt1; // used in identify_five_plus_one 
	INT *Elt2; // used in upstep
	INT *Elt3; // used in upstep
	INT *Elt4; // used in upstep

	strong_generators *SG_line_stab;
		// stabilizer of the special line in PGL(4,q) 
		// this group acts on the set Neighbors[] in the wedge action


	INT l_min;
	INT short_orbit_idx;

	INT nb_neighbors;
		// = (q + 1) * q * (q + 1)

	INT *Neighbors; // [nb_neighbors] 
		// The lines which intersect the special line. 
		// In wedge ranks.
		// The array Neighbors is sorted.

	INT *Neighbor_to_line; // [nb_neighbors] 
		// The lines which intersect the special line. 
		// In grassmann (i.e., line) ranks.
	INT *Neighbor_to_klein; // [nb_neighbors] 
		// In orthogonal ranks (i.e., points on the Klein quadric).

	INT *Line_to_neighbor; // [Surf->nb_lines_PG_3]
	
	longinteger_object go, stab_go;
	sims *Stab;
	strong_generators *stab_gens;

	INT *orbit;
	INT orbit_len;

	INT pt0_idx_in_orbit;
	INT pt0_wedge;
	INT pt0_line;
	INT pt0_klein;


	INT Basis[8];
	INT *line_to_orbit; // [nb_lines_PG_3]
	INT *orbit_to_line; // [nb_lines_PG_3]

	INT *Pts_klein;
	INT *Pts_wedge;
	INT nb_pts;
	
	INT *Pts_wedge_to_line; // [nb_pts]
	INT *line_to_pts_wedge; // [nb_lines_PG_3]

	action *A_on_neighbors; // restricted action A2 on the set Neighbors[]

	generator *Five_plus_one; // orbits on five-plus-one configurations


	INT *u, *v, *w; // temporary vectors of length 6
	INT *u1, *v1; // temporary vectors of length 6

	INT len; // = gen->nb_orbits_at_level(5) = number of orbits on 5-sets of lines
	INT *Idx; // Idx[nb], list of orbits for which the system has rank 19
	INT nb; // number of good orbits
	INT *Po; // Po[Flag_orbits->nb_flag_orbits], list of orbits for which a double six exists

	
	flag_orbits *Flag_orbits;

	classification *Double_sixes;


	classify_double_sixes();
	~classify_double_sixes();
	void null();
	void freeself();
	void init(surface_with_action *Surf_A, linear_group *LG, 
		int argc, const char **argv, 
		INT verbose_level);
	void compute_neighbors(INT verbose_level);
	void make_spreadsheet_of_neighbors(spreadsheet *&Sp, INT verbose_level);
	void classify_partial_ovoids(INT verbose_level);
	INT partial_ovoid_test(INT *S, INT len, INT verbose_level);
	void test_orbits(INT verbose_level);
	void make_spreadsheet_of_fiveplusone_configurations(spreadsheet *&Sp, 
		/*INT f_with_fusion, INT *Fusion, vector_ge *transporter,*/ 
		INT verbose_level);
	void identify_five_plus_one(INT *five_lines, INT transversal_line, 
		INT *five_lines_out_as_neighbors, INT &orbit_index, 
		INT *transporter, INT verbose_level);
	void classify(INT verbose_level);
	void downstep(INT verbose_level);
	//void downstep2(INT verbose_level);
	void upstep(INT verbose_level);
	void print_five_plus_ones(ostream &ost);
	void identify_double_six(INT *double_six, 
		INT *transporter, INT &orbit_index, INT verbose_level);
	void write_file(ofstream &fp, INT verbose_level);
	void read_file(ifstream &fp, INT verbose_level);

};

// ####################################################################################
// object_in_projective_space_with_action.C:
// ####################################################################################


class object_in_projective_space_with_action {

public:

	object_in_projective_space *OiP; // do not free
	strong_generators *Aut_gens;  // generators for the automorphism group


	object_in_projective_space_with_action();
	~object_in_projective_space_with_action();
	void null();
	void freeself();
	void init(object_in_projective_space *OiP, 
		strong_generators *Aut_gens, INT verbose_level);
};



// ####################################################################################
// surface_create_description.C:
// ####################################################################################



class surface_create_description {

public:

	INT f_q;
	INT q;
	INT f_catalogue;
	INT iso;
	INT f_by_coefficients;
	const BYTE *coefficients_text;
	INT f_family_S;
	INT parameter_a;
	INT f_arc_lifting;
	const BYTE *arc_lifting_text;


	
	surface_create_description();
	~surface_create_description();
	void null();
	void freeself();
	INT read_arguments(int argc, const char **argv, 
		INT verbose_level);
};

// ####################################################################################
// surface_create.C:
// ####################################################################################



class surface_create {

public:
	surface_create_description *Descr;

	BYTE prefix[1000];

	INT q;
	finite_field *F;

	INT f_semilinear;
	
	surface *Surf;

	surface_with_action *Surf_A;
	

	INT coeffs[20];
	INT f_has_lines;
	INT Lines[27];
	INT f_has_group;
	strong_generators *Sg;
	


	
	surface_create();
	~surface_create();
	void null();
	void freeself();
	void init(surface_create_description *Descr, INT verbose_level);
	void apply_transformations(const BYTE **transform_coeffs, 
		INT *f_inverse_transform, INT nb_transform, INT verbose_level);
};




// ####################################################################################
// arc_lifting.C:
// ####################################################################################


class arc_lifting {

public:

	INT q;
	finite_field *F; // do not free

	surface *Surf; // do not free

	surface_with_action *Surf_A;

	INT *arc;
	INT arc_size;
	

	// data from projective_space::find_Eckardt_points_from_arc_not_on_conic_prepare_data:
		INT *bisecants; // [15]
		INT *Intersections; // [15 * 15]
		INT *B_pts; // [nb_B_pts]
		INT *B_pts_label; // [nb_B_pts * 3]
		INT nb_B_pts; // at most 15
		INT *E2; // [6 * 5 * 2] Eckardt points of the second type 
		INT nb_E2; // at most 30
		INT *conic_coefficients; // [6 * 6]


	eckardt_point *E;
	INT *E_idx;
	INT nb_E; // = nb_B_pts + nb_E2

	INT *T_idx;
	INT nb_T;

	INT t_idx0;


	INT *the_equation; // [20]
	INT *Web_of_cubic_curves; // [45 * 10]
	INT *The_plane_equations; // [45 * 4]
	INT *The_plane_rank; // [45]
	INT *The_plane_duals; // [45]
	INT base_curves4[4];
	INT row_col_Eckardt_points[6];
	INT *Dual_point_ranks; // [nb_T * 6]
	INT *base_curves; // [4 * 10]
	INT Lines27[27];


	INT The_six_plane_equations[6 * 4]; // [6 * 4]
	INT *The_surface_equations; // [(q + 1) * 20]
	INT planes6[6];
	INT lambda, lambda_rk;
	INT t_idx;

	strong_generators *stab_gens;
	strong_generators *gens_subgroup;
	longinteger_object stabilizer_of_trihedral_pair_go;
	action *A_on_equations;
	schreier *Orb;
	longinteger_object stab_order;
	INT trihedral_pair_orbit_index;
	vector_ge *cosets;

	vector_ge *coset_reps;
	INT nine_lines[9];
	INT *aut_T_index;
	INT *aut_coset_index;
	strong_generators *Aut_gens;


	INT F_plane[3 * 4];
	INT G_plane[3 * 4];
	INT *System; // [3 * 4 * 3]
	//INT nine_lines[9];

	INT *transporter0;
	INT *transporter;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	INT *Elt4;
	INT *Elt5;


	arc_lifting();
	~arc_lifting();
	void null();
	void freeself();
	void init(surface_with_action *Surf_A, INT *arc, INT arc_size, INT verbose_level);
	void find_Eckardt_points(INT verbose_level);
	void find_trihedral_pairs(INT verbose_level);
	void lift_prepare(INT verbose_level);
	void print(ostream &ost);
	void print_Eckardt_point_data(ostream &ost);
	void print_bisecants(ostream &ost);
	void print_intersections(ostream &ost);
	void print_conics(ostream &ost);
	void print_Eckardt_points(ostream &ost);
	void print_web_of_cubic_curves(ostream &ost);
	void print_trihedral_plane_equations(ostream &ost);
	void print_lines(ostream &ost);
	void print_dual_point_ranks(ostream &ost);
	void print_FG(ostream &ost);
	void print_the_six_plane_equations(INT *The_six_plane_equations, INT *plane6, ostream &ost);
	void print_surface_equations_on_line(INT *The_surface_equations, 
		INT lambda, INT lambda_rk, ostream &ost);
	void create_the_six_plane_equations(INT t_idx, INT *The_six_plane_equations, INT *planes6, INT verbose_level);
	void create_surface_from_trihedral_pair_and_arc(INT t_idx, INT *planes6, 
		INT *The_six_plane_equations, INT *The_surface_equations, 
		INT &lambda, INT &lambda_rk, INT verbose_level);
		// plane6[6]
		// The_six_plane_equations[6 * 4]
		// The_surface_equations[(q + 1) * 20]
	strong_generators *create_stabilizer_of_trihedral_pair(INT *planes6, 
		INT &trihedral_pair_orbit_index, INT verbose_level);
	void create_action_on_equations_and_compute_orbits(INT *The_surface_equations, 
		strong_generators *gens_for_stabilizer_of_trihedral_pair, 
		action *&A_on_equations, schreier *&Orb, 
		INT verbose_level);
	void create_clebsch_system(INT *The_six_plane_equations, INT lambda, INT verbose_level);
	void loop_over_trihedral_pairs(vector_ge *cosets, vector_ge *&coset_reps, 
		INT *&aut_T_index, INT *&aut_coset_index, INT verbose_level);
	void create_surface(surface_with_action *Surf_A, INT *Arc6, INT verbose_level);
};



