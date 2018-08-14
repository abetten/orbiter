/*
 * poset_classification.h
 *
 *  Created on: Aug 9, 2018
 *      Author: Anton Betten
 *
 * started:  September 20, 2007
 * pulled out of snakesandladders.h: Aug 9, 2018
 */

// #############################################################################
// extension.C:
// #############################################################################

#define NB_EXTENSION_TYPES 5

#define EXTENSION_TYPE_UNPROCESSED  0
#define EXTENSION_TYPE_EXTENSION 1
#define EXTENSION_TYPE_FUSION 2
#define EXTENSION_TYPE_PROCESSING 3
#define EXTENSION_TYPE_NOT_CANONICAL 4

class extension {

public:
	INT pt;
	INT orbit_len;
	INT type;
		// EXTENSION_TYPE_UNPROCESSED = unprocessed
		// EXTENSION_TYPE_EXTENSION = extension node
		// EXTENSION_TYPE_FUSION = fusion node
		// EXTENSION_TYPE_PROCESSING = currently processing
		// EXTENSION_TYPE_NOT_CANONICAL = no extension formed 
		// because it is not canonical
	INT data;
		// if EXTENSION_TYPE_EXTENSION: a handle to the next 
		//  oracle node
		// if EXTENSION_TYPE_FUSION: a handle to a fusion element
	INT data1;
		// if EXTENSION_TYPE_FUSION: node to which we are fusing
	INT data2;
		// if EXTENSION_TYPE_FUSION: extension within that 
		// node to which we are fusing

	extension();
	~extension();
};


// #############################################################################
// generator.C
// #############################################################################

class generator {

public:
	INT t0;

	BYTE problem_label[1000];
	
	action *A; // the action in which the group is given
	action *A2; // the action in which we do the search
	
	strong_generators *Strong_gens;
	longinteger_object go;

	// used as storage for the current set:
	INT *S; // [sz]
	
	INT sz;
		// the target depth
		
	
	INT *Elt_memory; // [5 * elt_size_in_INT]
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	INT *Elt4;
	INT *Elt5;
	
	INT *tmp_set_apply_fusion;
		// used in oracle_upstep.C oracle::apply_fusion_element

	INT *tmp_find_node_for_subspace_by_rank1;
		// [vector_space_dimension] used in generator_trace.C: 
		// find_node_for_subspace_by_rank
	INT *tmp_find_node_for_subspace_by_rank2;
		// [sz * vector_space_dimension] used in generator_trace.C: 
		// find_node_for_subspace_by_rank
	INT *tmp_find_node_for_subspace_by_rank3;
		// [vector_space_dimension] used in generator_trace.C: 
		// find_node_for_subspace_by_rank

	INT f_candidate_check_func;
	INT (*candidate_check_func)(INT len, INT *S, void *data, 
		INT verbose_level);
	void *candidate_check_data;
	
	INT f_candidate_incremental_check_func;
	INT (*candidate_incremental_check_func)(INT len, INT *S, 
		void *data, INT verbose_level);
	void *candidate_incremental_check_data;

	INT f_print_function;
	void (*print_function)(INT len, INT *S, void *data);
	void *print_function_data;
	
	INT nb_times_trace;
	INT nb_times_trace_was_saved;
	
	// data for find_automorphism_by_tracing:
	vector_ge *transporter; // [sz + 1]
	INT **set; // [sz + 1][sz]

	
	
	// the following is maintained by init_oracle / exit_oracle:
	INT nb_oracle_nodes_used;
	INT nb_oracle_nodes_allocated;
	INT oracle_nodes_increment;
	INT oracle_nodes_increment_last;
	
	oracle *root;
	
	INT *first_oracle_node_at_level;
	INT *set0; // [sz + 1] temporary storage
	INT *set1; // [sz + 1] temporary storage
	INT *set3; // [sz + 1] temporary storage
	
	INT *nb_extension_nodes_at_level_total;
	INT *nb_extension_nodes_at_level;
	INT *nb_fusion_nodes_at_level;
	INT *nb_unprocessed_nodes_at_level;


	// command line options:


	INT depth; // search depth
	INT f_w; // write output in level files (only last level)
	INT f_W; // write output in level files (each level)
	INT f_T; // draw tree file (each level)
	INT f_t; // draw tree file (only last level)
	INT f_Log; // log nodes (each level)
	INT f_log; // log nodes (only last level)
	INT f_print_only;
	INT f_find_group_order;
	INT find_group_order;
	
	INT verbose_level;
	INT verbose_level_group_theory;
	
	INT xmax, ymax, radius;
	
	INT f_recover;
	const BYTE *recover_fname;

	INT f_extend;
	INT extend_from, extend_to;
	INT extend_r, extend_m;
	BYTE extend_fname[1000];

	INT f_lex;
	INT f_max_depth;
	INT max_depth;

	BYTE fname_base[1000]; // = path + prefix
	BYTE prefix[1000]; // = fname_base without prefix
	BYTE path[1000];


	INT f_starter;
	INT starter_size;
	INT *starter;
	strong_generators *starter_strong_gens;
	INT *starter_live_points;
	INT starter_nb_live_points;
	void *starter_canonize_data;
	INT (*starter_canonize)(INT *Set, INT len, INT *Elt, 
		void *data, INT verbose_level);
	INT *starter_canonize_Elt;
	
	INT f_has_invariant_subset_for_root_node;
	INT *invariant_subset_for_root_node;
	INT invariant_subset_for_root_node_size;
	

	INT f_do_group_extension_in_upstep;
		// is TRUE by default

	INT f_allowed_to_show_group_elements;
	INT downstep_orbits_print_max_orbits;
	INT downstep_orbits_print_max_points_per_orbit;
	
	
	INT f_on_subspaces;
	INT vector_space_dimension;
	finite_field *F;
	INT (*rank_point_func)(INT *v, void *data);
	void (*unrank_point_func)(INT *v, INT rk, void *data);
	void *rank_point_data;
	INT *tmp_v1; // [vector_space_dimension]

	INT f_early_test_func;
	void (*early_test_func)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level);
	void *early_test_func_data;
	INT f_its_OK_to_not_have_an_early_test_func;

	INT nb_times_image_of_called0;
	INT nb_times_mult_called0;
	INT nb_times_invert_called0;
	INT nb_times_retrieve_called0;
	INT nb_times_store_called0;
	
	double progress_last_time;
	double progress_epsilon;

	INT f_draw_schreier_trees;
		BYTE schreier_tree_prefix[1000];
		INT schreier_tree_xmax; // = 1000000;
		INT schreier_tree_ymax; // =  500000;
		INT schreier_tree_f_circletext; // = TRUE;
		INT schreier_tree_rad; // = 25000;
		INT schreier_tree_f_embedded;
		INT schreier_tree_f_sideways;
		double schreier_tree_scale;
		double schreier_tree_line_width;



	// generator.C:
	INT nb_orbits_at_level(INT level);
	INT nb_flag_orbits_up_at_level(INT level);
	oracle *get_node_ij(INT level, INT node);
	INT poset_structure_is_contained(INT *set1, INT sz1, 
		INT *set2, INT sz2, INT verbose_level);
	void print_progress_by_extension(INT size, INT cur, 
		INT prev, INT cur_ex, INT nb_ext_cur, INT nb_fuse_cur);
	void print_progress(INT size, INT cur, INT prev, 
		INT nb_ext_cur, INT nb_fuse_cur);
	void print_progress(INT lvl, double progress);
	void print_progress_by_level(INT lvl);
	void print_orbit_numbers(INT depth);
	void print_statistic_on_callbacks_naked();
	void print_statistic_on_callbacks();
	set_and_stabilizer *get_set_and_stabilizer(INT level, 
		INT orbit_at_level, INT verbose_level);
	void get_set_by_level(INT level, INT node, INT *set);
	void get_set(INT node, INT *set, INT &size);
	void get_set(INT level, INT orbit, INT *set, INT &size);
	void print_set_verbose(INT node);
	void print_set_verbose(INT level, INT orbit);
	void print_set(INT node);
	void print_set(INT level, INT orbit);
	
	INT find_oracle_node_for_set(INT len, INT *set, 
		INT f_tolerant, INT verbose_level);
	INT find_oracle_node_for_set_basic(INT from, 
		INT node, INT len, INT *set, INT f_tolerant, 
		INT verbose_level);
	void oracle_depth_breadth_perm_and_inverse(INT max_depth, 
		INT *&perm, INT *&perm_inv, INT verbose_level);
	INT count_extension_nodes_at_level(INT lvl);
	double level_progress(INT lvl);
	void count_automorphism_group_orders(INT lvl, INT &nb_agos, 
		longinteger_object *&agos, INT *&multiplicities, 
		INT verbose_level);
	void compute_and_print_automorphism_group_orders(INT lvl, 
		ostream &ost);
	void stabilizer_order(INT node, longinteger_object &go);
	INT check_the_set(INT len, INT *S, INT verbose_level);
	INT check_the_set_incrementally(INT len, INT *S, INT verbose_level);

	void orbit_length(INT orbit_at_level, INT level, longinteger_object &len);
	void get_orbit_length_and_stabilizer_order(INT node, INT level, 
		longinteger_object &stab_order, longinteger_object &len);
	INT orbit_length_as_INT(INT orbit_at_level, INT level);
	void print_representatives_at_level(INT lvl);
	void print_lex_rank(INT *set, INT sz);
	void print_problem_label();
	void print_level_info(INT i, INT prev);
	void print_level_extension_info(INT i, 
		INT prev, INT cur_extension);
	void print_level_extension_coset_info(INT i, 
		INT prev, INT cur_extension, INT coset, INT nb_cosets);
	void recreate_schreier_vectors_up_to_level(INT lvl, 
		INT f_compact, INT verbose_level);
	void recreate_schreier_vectors_at_level(INT i, INT f_compact, 
		INT verbose_level);
	void print_node(INT node);
	void print_tree();
	void get_table_of_nodes(INT *&Table, INT &nb_rows, INT &nb_cols, 
		INT verbose_level);
	INT count_live_points(INT level, INT node_local, INT f_compact, 
		INT verbose_level);
	void find_automorphism_group_of_order(INT level, INT order);
	void get_stabilizer_order(INT level, INT orbit_at_level, 
		longinteger_object &go);
	void get_stabilizer_group(group *&G,  
		INT level, INT orbit_at_level, INT verbose_level);
	void get_stabilizer_generators(strong_generators *&gens,  
		INT level, INT orbit_at_level, INT verbose_level);
	void change_extension_type(INT level, INT node, INT cur_ext, 
		INT type, INT verbose_level);
	void orbit_element_unrank(INT depth, INT orbit_idx, 
		INT rank, INT *set, INT verbose_level);
	void orbit_element_rank(INT depth, INT &orbit_idx, 
		INT &rank, INT *set, 
		INT verbose_level);
		// used in M_SYSTEM/global_data::do_puzzle
		// and in UNITALS/test_lines_data::get_orbit_of_sets
	void coset_unrank(INT depth, INT orbit_idx, INT rank, 
		INT *Elt, INT verbose_level);
	INT coset_rank(INT depth, INT orbit_idx, INT *Elt, 
		INT verbose_level);
	void list_all_orbits_at_level(INT depth, 
		INT f_has_print_function, 
		void (*print_function)(INT len, INT *S, void *data), 
		void *print_function_data, 
		INT f_show_orbit_decomposition, INT f_show_stab, 
		INT f_save_stab, INT f_show_whole_orbit);
	void compute_integer_property_of_selected_list_of_orbits(
		INT depth, 
		INT nb_orbits, INT *Orbit_idx, 
		INT (*compute_function)(INT len, INT *S, void *data), 
		void *compute_function_data,
		INT *&Data);
	void list_selected_set_of_orbits_at_level(INT depth, 
		INT nb_orbits, INT *Orbit_idx, 
		INT f_has_print_function, 
		void (*print_function)(INT len, INT *S, void *data), 
		void *print_function_data, 
		INT f_show_orbit_decomposition, INT f_show_stab, 
		INT f_save_stab, INT f_show_whole_orbit);
	void test_property(INT depth, 
		INT (*test_property_function)(INT len, INT *S, void *data), 
		void *test_property_data, 
		INT &nb, INT *&Orbit_idx);
	void print_schreier_vectors_at_depth(INT depth, INT verbose_level);
	void print_schreier_vector(INT depth, INT orbit_idx, 
		INT verbose_level);
	void list_whole_orbit(INT depth, INT orbit_idx, 
		INT f_has_print_function, 
		void (*print_function)(INT len, INT *S, void *data), 
		void *print_function_data, 
		INT f_show_orbit_decomposition, INT f_show_stab, 
		INT f_save_stab, INT f_show_whole_orbit);
	void print_extensions_at_level(ostream &ost, INT lvl);
	void map_to_canonical_k_subset(INT *the_set, INT set_size, 
		INT subset_size, INT subset_rk, 
		INT *reduced_set, INT *transporter, INT &local_idx, 
		INT verbose_level);
		// fills reduced_set[set_size - subset_size], 
		// transporter and local_idx
		// local_idx is the index of the orbit that the 
		// subset belongs to 
		// (in the list of orbit of subsets of size subset_size)
	void get_representative_of_subset_orbit(INT *set, INT size, 
		INT local_orbit_no, 
		strong_generators *&Strong_gens, 
		INT verbose_level);
	void print_fusion_nodes(INT depth);
	void find_interesting_k_subsets(INT *the_set, INT n, INT k, 
		INT *&interesting_sets, INT &nb_interesting_sets, 
		INT &orbit_idx, INT verbose_level);
	void classify_k_subsets(INT *the_set, INT n, INT k, 
		classify *&C, INT verbose_level);
	void trace_all_k_subsets(INT *the_set, INT n, INT k, 
		INT &nCk, INT *&isotype, INT verbose_level);
	void get_orbit_representatives(INT level, INT &nb_orbits, 
		INT *&Orbit_reps, INT verbose_level);
	void unrank_point(INT *v, INT rk);
	INT rank_point(INT *v);
	void unrank_basis(INT *Basis, INT *S, INT len);
	void rank_basis(INT *Basis, INT *S, INT len);
	void generate_source_code(INT level, INT verbose_level);

	// generator_init.C:
	generator();
	~generator();
	void null();
	void freeself();
	void usage();
	void read_arguments(int argc, const char **argv, 
		INT verbose_level);
	void init(action *A, action *A2, 
		strong_generators *gens, 
		INT sz, INT verbose_level);
		// gens will not be copied !
	void initialize(action *A_base, action *A_use, 
		strong_generators *gens, 
		INT depth, 
		const BYTE *path, const BYTE *prefix, INT verbose_level);
	void initialize_with_starter(action *A_base, action *A_use, 
		strong_generators *gens, 
		INT depth, 
		BYTE *path, 
		BYTE *prefix, 
		INT starter_size, 
		INT *starter, 
		strong_generators *Starter_Strong_gens, 
		INT *starter_live_points, 
		INT starter_nb_live_points, 
		void *starter_canonize_data, 
		INT (*starter_canonize)(INT *Set, INT len, INT *Elt, 
			void *data, INT verbose_level), 
		INT verbose_level);
	void init_root_node_invariant_subset(
		INT *invariant_subset, INT invariant_subset_size, 
		INT verbose_level);
	void init_root_node(INT verbose_level);
	void init_oracle(INT nb_oracle_nodes, INT verbose_level);
	void exit_oracle();
	void reallocate();
	void reallocate_to(INT new_number_of_nodes, INT verbose_level);
	void init_check_func(
		INT (*candidate_check_func)(INT len, INT *S, void *data, 
		INT verbose_level), 
		void *candidate_check_data);
	void init_incremental_check_func(
		INT (*candidate_incremental_check_func)(INT len, INT *S, 
			void *data, INT verbose_level), 
		void *candidate_incremental_check_data);
	void init_starter(INT starter_size, 
		INT *starter, 
		strong_generators *starter_strong_gens, 
		INT *starter_live_points, 
		INT starter_nb_live_points, 
		void *starter_canonize_data, 
		INT (*starter_canonize)(INT *Set, INT len, INT *Elt, 
			void *data, INT verbose_level), 
		INT verbose_level);
		// Does not initialize the first starter nodes. 
		// This is done in init_root_node 
	void init_vector_space_action(INT vector_space_dimension, 
		finite_field *F, 
		INT (*rank_point_func)(INT *v, void *data), 
		void (*unrank_point_func)(INT *v, INT rk, void *data), 
		void *data, 
		INT verbose_level);
	void init_early_test_func(
		void (*early_test_func)(INT *S, INT len, 
			INT *candidates, INT nb_candidates, 
			INT *good_candidates, INT &nb_good_candidates, 
			void *data, INT verbose_level), 
		void *data,  
		INT verbose_level);

	// generator_classify.C
	INT compute_orbits(INT from_level, INT to_level, 
		INT f_write_candidate_file, 
		INT verbose_level);
		// returns TRUE if there is at least one orbit 
		// at level to_level, 
		// FALSE otherwise
	INT main(INT t0, 
		INT schreier_depth, 
		INT f_use_invariant_subset_if_available, 
		INT f_debug, 
		INT verbose_level);
		// f_use_invariant_subset_if_available is 
		// an option that affects the downstep.
		// if FALSE, the orbits of the stabilizer 
		// on all points are computed. 
		// if TRUE, the orbits of the stabilizer 
		// on the set of points that were 
		// possible in the previous level are computed only 
		// (using Schreier.orbits_on_invariant_subset_fast).
		// The set of possible points is stored 
		// inside the schreier vector data structure (sv).
	void extend_level(INT size, 
		INT f_create_schreier_vector, 
		INT f_compact, 
		INT f_use_invariant_subset_if_available, 
		INT f_debug, 
		INT f_write_candidate_file, 
		INT verbose_level);
		// calls downstep, upstep
	void downstep(INT size, INT f_create_schreier_vector, INT f_compact, 
		INT f_use_invariant_subset_if_available, 
		INT verbose_level);
		// calls root[prev].downstep_subspace_action 
		// or root[prev].downstep
	void upstep(INT size, INT f_debug, 
		INT verbose_level);
		// calls extend_node
	void extend_node(INT size, INT prev, INT &cur, 
		INT f_debug, 
		INT f_indicate_not_canonicals, FILE *fp,
		INT verbose_level);
		// called by generator::upstep
		// Uses an upstep_work structure to handle the work.
		// Calls upstep_work::handle_extension



	// generator_combinatorics.C:
	void Plesken_matrix_up(INT depth, 
		INT *&P, INT &N, INT verbose_level);
	void Plesken_matrix_down(INT depth, 
		INT *&P, INT &N, INT verbose_level);
	void Plesken_submatrix_up(INT i, INT j, 
		INT *&Pij, INT &N1, INT &N2, INT verbose_level);
	void Plesken_submatrix_down(INT i, INT j,
		INT *&Pij, INT &N1, INT &N2, INT verbose_level);
	INT count_incidences_up(INT lvl1, INT po1, 
		INT lvl2, INT po2, INT verbose_level);
	INT count_incidences_down(INT lvl1, 
		INT po1, INT lvl2, INT po2, INT verbose_level);
	void Asup_to_Ainf(INT t, INT k, INT *M_sup, 
		INT *M_inf, INT verbose_level);
	void test_for_multi_edge_in_classification_graph(
		INT depth, INT verbose_level);


	// generator_trace.C:
	INT find_isomorphism(INT *set1, INT *set2, INT sz, 
		INT *transporter, INT &orbit_idx, INT verbose_level);
	set_and_stabilizer *identify_and_get_stabilizer(
		INT *set, INT sz, INT *transporter, 
		INT &orbit_at_level, 
		INT verbose_level);
	void identify(INT *data, INT sz, INT *transporter, 
		INT &orbit_at_level, INT verbose_level);
	void test_identify(INT level, INT nb_times, INT verbose_level);
	void generator_apply_fusion_element_no_transporter(
		INT cur_level, INT size, INT cur_node, INT cur_ex, 
		INT *set_in, INT *set_out, 
		INT verbose_level);
	INT generator_apply_fusion_element(INT level, INT size, 
		INT current_node, INT current_extension, 
		INT *set_in, INT *set_out, INT *set_tmp, 
		INT *transporter_in, INT *transporter_out, 
		INT f_tolerant, 
		INT verbose_level);
	INT trace_set_recursion(INT cur_level, INT cur_node, 
		INT size, INT level, 
		INT *canonical_set, INT *tmp_set1, INT *tmp_set2, 
		INT *Elt_transporter, INT *tmp_Elt1, 
		INT f_tolerant, 
		INT verbose_level);
		// called by generator::trace_set
		// returns the node in the generator that corresponds 
		// to the canonical_set
		// or -1 if f_tolerant and the node could not be found
	INT trace_set(INT *set, INT size, INT level, 
		INT *canonical_set, INT *Elt_transporter, 
		INT verbose_level);
		// returns the case number of the canonical set
	INT find_node_for_subspace_by_rank(INT *set, INT len, 
		INT verbose_level);
	
	// generator_draw.C:
	void write_treefile_and_draw_tree(BYTE *fname_base, 
		INT lvl, INT xmax, INT ymax, INT rad, 
		INT f_embedded, INT verbose_level);
	INT write_treefile(BYTE *fname_base, INT lvl, 
		INT verbose_level);
	void draw_tree(BYTE *fname_base, INT lvl, 
		INT xmax, INT ymax, INT rad, INT f_embedded, 
		INT f_sideways, INT verbose_level);
	void draw_tree_low_level(BYTE *fname, INT nb_nodes, 
		INT *coord_xyw, INT *perm, INT *perm_inv, 
		INT f_draw_points, INT f_draw_extension_points, 
		INT f_draw_aut_group_order, 
		INT xmax, INT ymax, INT rad, INT f_embedded, 
		INT f_sideways, INT verbose_level);
	void draw_tree_low_level1(mp_graphics &G, INT nb_nodes, 
		INT *coords, INT *perm, INT *perm_inv, 
		INT f_draw_points, INT f_draw_extension_points, 
		INT f_draw_aut_group_order, 
		INT radius, INT verbose_level);
	void draw_poset_full(const BYTE *fname_base, INT depth, 
		INT data, INT f_embedded, INT f_sideways, 
		double x_stretch, INT verbose_level);
	void draw_poset(const BYTE *fname_base, INT depth, 
		INT data1, INT f_embedded, INT f_sideways, 
		INT verbose_level);
	void draw_level_graph(const BYTE *fname_base, INT depth, 
		INT data, INT level, INT f_embedded, INT f_sideways, 
		INT verbose_level);
	void make_full_poset_graph(INT depth, layered_graph *&LG, 
		INT data1, double x_stretch, 
		INT verbose_level);
	void make_auxiliary_graph(INT depth, 
		layered_graph *&LG, INT data1, 
		INT verbose_level);
	void make_graph(INT depth, layered_graph *&LG, 
		INT data1, INT f_tree, INT verbose_level);
	void make_level_graph(INT depth, layered_graph *&LG, 
		INT data1, INT level, INT verbose_level);
	void print_data_structure_tex(INT depth, INT verbose_level);
	void make_poset_graph_detailed(layered_graph *&LG, 
		INT data1, INT max_depth, INT verbose_level);


	// in generator_io.C:
	void housekeeping(INT i, INT f_write_files, INT t0, 
		INT verbose_level);
	void housekeeping_no_data_file(INT i, INT t0, INT verbose_level);
	INT test_sv_level_file_binary(INT level, BYTE *fname_base);
	void read_sv_level_file_binary(INT level, BYTE *fname_base, 
		INT f_split, INT split_mod, INT split_case, 
		INT f_recreate_extensions, INT f_dont_keep_sv, 
		INT verbose_level);
	void write_sv_level_file_binary(INT level, BYTE *fname_base, 
		INT f_split, INT split_mod, INT split_case, 
		INT verbose_level);
	void read_sv_level_file_binary2(INT level, FILE *fp, 
		INT f_split, INT split_mod, INT split_case, 
		INT f_recreate_extensions, INT f_dont_keep_sv, 
		INT verbose_level);
	void write_sv_level_file_binary2(INT level, FILE *fp, 
		INT f_split, INT split_mod, INT split_case, 
		INT verbose_level);
	void read_level_file_binary(INT level, BYTE *fname_base, 
		INT verbose_level);
	void write_level_file_binary(INT level, BYTE *fname_base, 
		INT verbose_level);
	void read_level_file_binary2(INT level, FILE *fp, 
		INT &nb_group_elements, INT verbose_level);
	void write_level_file_binary2(INT level, FILE *fp, 
		INT &nb_group_elements, INT verbose_level);
	INT calc_size_on_file(INT depth_completed, INT verbose_level);
	void make_fname_candidates_file_default(BYTE *fname, INT level);
	void write_candidates_binary_using_sv(BYTE *fname_base, 
		INT lvl, INT t0, INT verbose_level);
	void read_level_file(INT level, BYTE *fname, INT verbose_level);
	void read_data_file(INT &depth_completed, 
		const BYTE *fname, INT verbose_level);
	void write_data_file(INT depth_completed, 
		const BYTE *fname_base, INT verbose_level);
	void read_memory_object(INT &depth_completed, 
		memory_object *m, INT &nb_group_elements, INT verbose_level);
	void write_memory_object(INT depth_completed, 
		memory_object *m, INT &nb_group_elements, INT verbose_level);
	void recover(const BYTE *recover_fname, 
		INT &depth_completed, INT verbose_level);
	void write_lvl_file_with_candidates(BYTE *fname_base, 
		INT lvl, INT t0, INT verbose_level);
	void write_lvl_file(BYTE *fname_base, INT lvl, 
		INT t0, INT f_with_stabilizer_generators, INT f_long_version, 
		INT verbose_level);
	void write_lvl(ostream &f, INT lvl, INT t0, 
		INT f_with_stabilizer_generators, INT f_long_version, 
		INT verbose_level);
	void log_nodes_for_treefile(INT cur, INT depth, 
		ostream &f, INT f_recurse, INT verbose_level);
	void Log_nodes(INT cur, INT depth, ostream &f, 
		INT f_recurse, INT verbose_level);
	void log_current_node(ostream &f, INT size);
	void make_spreadsheet_of_orbit_reps(spreadsheet *&Sp, 
		INT max_depth);
	void make_spreadsheet_of_level_info(spreadsheet *&Sp, 
		INT max_depth);
	void write_file(ofstream &fp, INT depth_completed, 
		INT verbose_level);
	void read_file(ifstream &fp, INT &depth_completed, 
		INT verbose_level);

	// generator_recognize.C:
	void recognize_start_over(
		INT size, INT f_implicit_fusion,
		INT lvl, INT current_node,
		INT &final_node, INT verbose_level);
	// Called from oracle::find_automorphism_by_tracing_recursion
	// when trace_next_point returns FALSE
	// This can happen only if f_implicit_fusion is TRUE
	void recognize_recursion(
		INT size, INT f_implicit_fusion,
		INT lvl, INT current_node, INT &final_node,
		INT verbose_level);
	// this routine is called by upstep_work::find_automorphism_by_tracing
	// we are dealing with a set of size len + 1.
	// but we can only trace the first len points.
	// the tracing starts at lvl = 0 with current_node = 0
	void recognize(
		INT *the_set, INT size, INT *transporter, INT f_implicit_fusion,
		INT &final_node, INT verbose_level);

};

// in generator_io.C:
void generator_read_candidates_of_orbit(const BYTE *fname, INT orbit_at_level,
	INT *&candidates, INT &nb_candidates, INT verbose_level);




// #############################################################################
// oracle.C, oracle_io.C, oracle_upstep.C, oracle_upstep_subspace_action.C, 
// oracle_downstep.C, oracle_downstep_subspace_action.C:
// #############################################################################


class oracle {
public:
	INT node;
	INT prev;
	
	INT pt;
	INT nb_strong_generators;
	INT *hdl_strong_generators;
	INT *tl;
	
	INT nb_extensions;
	extension *E;
	
	INT *sv;
	
	// oracle.C:
	oracle();
	~oracle();
	void null();
	void freeself();
	void init_root_node(generator *gen, INT verbose_level);
		// copies gen->SG0 and gen->tl into the oracle 
		// structure using store_strong_generators
	void init_extension_node_prepare_G(generator *gen,
		INT prev, INT prev_ex, INT size, group &G,
		longinteger_object &go_G,
		INT verbose_level);
		// sets up the group G using the strong
		// generators that are stored
	void init_extension_node_prepare_H(generator *gen, 
		INT prev, INT prev_ex, INT size, 
		group &G, longinteger_object &go_G, 
		group &H, longinteger_object &go_H, 
		INT pt, INT pt_orbit_len, 
		INT verbose_level);
		// sets up the group H which is the stabilizer 
		// of the point pt in G
	void compute_point_stabilizer_in_subspace_setting(
		generator *gen,
		INT prev, INT prev_ex, INT size, 
		group &G, longinteger_object &go_G, 
		group &H, longinteger_object &go_H, 
		INT pt, INT pt_orbit_len, 
		INT verbose_level);
	void compute_point_stabilizer_in_standard_setting(
		generator *gen,
		INT prev, INT prev_ex, INT size, 
		group &G, longinteger_object &go_G, 
		group &H, /* longinteger_object &go_H, */
		INT pt, INT pt_orbit_len, 
		INT verbose_level);
	INT get_level(generator *gen);
	INT get_node_in_level(generator *gen);
	INT get_nb_of_live_points();
	INT get_nb_of_orbits_under_stabilizer();
	void get_stabilizer_order(generator *gen, longinteger_object &go);
	void get_stabilizer(generator *gen, 
		group &G, longinteger_object &go_G, 
		INT verbose_level);
	void get_stabilizer_generators(generator *gen, 
		strong_generators *&Strong_gens, 
		INT verbose_level);
	void oracle_depth_breadth_perm_and_inverse(generator *gen, 
		INT max_depth, 
		INT &idx, INT hdl, INT cur_depth, INT *perm, INT *perm_inv);
	INT find_extension_from_point(generator *gen, INT pt, 
		INT verbose_level);
	void print_extensions(ostream &ost);
	void store_strong_generators(generator *gen, 
		strong_generators *Strong_gens);
	void log_current_node_without_group(generator *gen, 
		INT s, ostream &f, INT verbose_level);
	void log_current_node(generator *gen, INT s, 
		ostream &f, INT f_with_stabilizer_generators, 
		INT verbose_level);
	void log_current_node_after_applying_group_element(
		generator *gen, INT s, ostream &f, INT hdl, 
		INT verbose_level);
	void log_current_node_with_candidates(generator *gen, 
		INT lvl, ostream &f, INT verbose_level);
	INT depth_of_node(generator *gen);
	void store_set(generator *gen, INT i);
		// stores a set of size i + 1 to gen->S
	void store_set_with_verbose_level(generator *gen, INT i, 
		INT verbose_level);
	// stores a set of size i + 1 to gen->S[]
	void store_set_to(generator *gen, INT i, INT *to);
	void store_set_to(generator *gen, INT *to);
	INT check_node_and_set_consistency(generator *gen, INT i, INT *set);
	void print_set_verbose(generator *gen);
	void print_set(generator *gen);
	void print_node(generator *gen);
	void print_extensions(generator *gen);
	void reconstruct_extensions_from_sv(generator *gen, 
		INT verbose_level);

	// in oracle_io.C:
	void read_memory_object(action *A, memory_object *m, 
		INT &nb_group_elements, INT verbose_level);
	void write_memory_object(action *A, memory_object *m, 
		INT &nb_group_elements, INT verbose_level);
	void sv_read_file(FILE *fp, INT verbose_level);
	void sv_write_file(FILE *fp, INT verbose_level);
	void read_file(action *A, FILE *fp, INT &nb_group_elements, 
		INT verbose_level);
	void write_file(action *A, FILE *fp, INT &nb_group_elements, 
		INT verbose_level);
	INT calc_size_on_file(action *A, INT verbose_level);


	// oracle_upstep.C:
	INT apply_fusion_element(generator *gen, 
		INT lvl, INT current_node, 
		INT current_extension, INT len, INT f_tolerant, 
		INT verbose_level);
		// returns next_node
	void install_fusion_node(generator *gen, 
		INT lvl, INT current_node, 
		INT my_node, INT my_extension, INT my_coset, 
		INT pt0, INT current_extension, 
		INT f_debug, INT f_implicit_fusion, 
		INT verbose_level);
		// Called from oracle::handle_last_level
	INT trace_next_point_wrapper(generator *gen, INT lvl, 
		INT current_node, 
		INT len, INT f_implicit_fusion, INT &f_failure_to_find_point, 
		INT verbose_level);
		// Called from upstep_work::find_automorphism_by_tracing_recursion
		// applies the permutation which maps the point with index lvl 
		// (i.e. the lvl+1-st point) to its orbit representative.
		// also maps all the other points under that permutation.
		// we are dealing with a set of size len + 1
		// returns FALSE if we are using implicit fusion nodes 
		// and the set becomes lexicographically
		// less than before, in which case trace has to be restarted.
	INT trace_next_point_in_place(generator *gen, 
		INT lvl, INT current_node, INT size, 
		INT *cur_set, INT *tmp_set,
		INT *cur_transporter, INT *tmp_transporter, 
		INT f_implicit_fusion, INT &f_failure_to_find_point, 
		INT verbose_level);
		// called by generator::trace_set_recursion
	void trace_starter(generator *gen, INT size, 
		INT *cur_set, INT *next_set,
		INT *cur_transporter, INT *next_transporter, 
		INT verbose_level);
	INT trace_next_point(generator *gen, 
		INT lvl, INT current_node, INT size, 
		INT *cur_set, INT *next_set,
		INT *cur_transporter, INT *next_transporter, 
		INT f_implicit_fusion, INT &f_failure_to_find_point, 
		INT verbose_level);
		// Called by oracle::trace_next_point_wrapper 
		// and by oracle::trace_next_point_in_place
		// returns FALSE only if f_implicit_fusion is TRUE and
		// the set becomes lexcographically less 
	INT orbit_representative_and_coset_rep_inv(generator *gen, 
		INT lvl, INT pt_to_trace, INT &pt0, INT *&cosetrep, 
		INT verbose_level);
		// called by oracle::trace_next_point
		// FALSE means the point to trace was not found. 
		// This can happen if nodes were eliminated due to clique_test

	// oracle_upstep_subspace_action.C:
	void orbit_representative_and_coset_rep_inv_subspace_action(
		generator *gen, 
		INT lvl, INT pt_to_trace, INT &pt0, INT *&cosetrep, 
		INT verbose_level);
		// called by oracle::trace_next_point
		

	// oracle_downstep.C
	// top level functions:
	void downstep(generator *gen, 
		INT lvl, 
		INT f_create_schreier_vector, INT f_compact, 
		INT f_use_invariant_subset_if_available, 
		INT f_implicit_fusion, 
		INT verbose_level);
		// Called from generator::downstep if we are acting on sets 
		// (i.e., not on subspaces).
		// Calls downstep_orbits, 
		// downstep_orbit
	void compute_schreier_vector(generator *gen, 
		INT lvl, INT f_compact, INT verbose_level);
		// called from generator::recreate_schreier_vectors_at_level
		// and from generator::count_live_points
		// calls downstep_apply_early_test
		// and check_orbits
		// and Schreier.get_schreier_vector

		// 1st level under downstep:
	void downstep_orbits(
		generator *gen, schreier &Schreier, action &AR, 
		INT lvl, 
		INT f_use_invariant_subset_if_available, 
		INT &f_using_invariant_subset, 
		INT &f_node_is_dead_because_of_clique_testing, 
		INT verbose_level);
		// calls downstep_get_invariant_subset, 
		// downstep_apply_early_test, 
		// and AR.induced_action_by_restriction
		// if f_use_invariant_subset_if_available and 
		// f_using_invariant_subset
		//
		// Sets up the schreier data structure Schreier 
		// If f_using_invariant_subset, we will use the 
		// restricted action AR, otherwise the action gen->A2
		// In this action, the orbits are computed using 
		// Schreier.compute_all_point_orbits
		// and possibly printed using downstep_orbits_print
	void downstep_orbit_test_and_schreier_vector(
		generator *gen, schreier &Schreier, action &AR, 
		INT lvl, 
		INT f_use_invariant_subset_if_available, 
		INT f_using_invariant_subset,
		INT f_create_schreier_vector,
		INT f_compact, 
		INT &nb_good_orbits, INT &nb_points, 
		INT verbose_level);
		// called from downstep once downstep_orbits is completed
		// Calls check_orbits_wrapper and 
		// create_schreier_vector_wrapper
		// The order in which these two functions are called matters.
	void downstep_implicit_fusion(
		generator *gen, schreier &Schreier, action &AR, 
		INT f_using_invariant_subset,
		INT lvl, 
		INT f_implicit_fusion, 
		INT good_orbits1, INT nb_points1, 
		INT verbose_level);
		// called from downstep, 
		// once downstep_orbit_test_and_schreier_vector is done
		// calls test_orbits_for_implicit_fusion
	void find_extensions(generator *gen, 
		schreier &O, action &AR, INT f_using_invariant_subset, 
		INT lvl, 
		INT verbose_level);
		// called by downstep
		// prepares all extension nodes and marks them as unprocessed.
		// we are at depth lvl, i.e., currently, 
		// we have a set of size lvl.


		// second level under downstep:
	INT downstep_get_invariant_subset(
		generator *gen, 
		INT lvl, 
		INT &n, INT *&subset, INT &f_subset_is_allocated, 
		INT verbose_level);
		// called from downstep_orbits
		// Gets the live points at the present node.
	void downstep_apply_early_test(
		generator *gen, 
		INT lvl, 
		INT n, INT *subset, 
		INT *candidates, INT &nb_candidates, 
		INT verbose_level);
		// called from downstep_orbits, compute_schreier_vector  
		// calls the callback early test function if available
		// and calls test_point_using_check_functions otherwise

	void check_orbits_wrapper(generator *gen, 
		schreier &Schreier, action &AR, INT f_using_invariant_subset, 
		INT lvl, 
		INT &nb_good_orbits1, INT &nb_points1, 
		INT f_use_incremental_test_func_if_available, 
		INT verbose_level);
		// called from downstep_orbit_test_and_schreier_vector
		// This function and create_schreier_vector_wrapper 
		// are used in pairs.
		// Except, the order in which the function is used matters.
		// Calls check_orbits
	void create_schreier_vector_wrapper(INT f_create_schreier_vector, 
		INT f_compact, 
		schreier &Schreier, INT verbose_level);
		// called from downstep_orbit_test_and_schreier_vector
		// calls Schreier.get_schreier_vector

	void test_orbits_for_implicit_fusion(generator *gen, 
		schreier &Schreier, action &AR, 
		INT f_using_invariant_subset, 
		INT lvl, INT verbose_level);
		// called from downstep_implicit_fusion
		// eliminates implicit fusion orbits from the 
		// Schreier data structure, 
	INT nb_extension_points();
		// sums up the lengths of orbits in all extensions
	void check_orbits(generator *gen, 
		schreier &Schreier, action &AR, 
		INT f_using_invariant_subset, 
		INT lvl, 
		INT f_use_incremental_test_func_if_available, 
		INT verbose_level);
		// called from compute_schreier_vector 
		// and check_orbits_wrapper (which is called from 
		// downstep_orbit_test_and_schreier_vector)
		// calls test_point_using_check_functions
		// eliminates bad orbits from the Schreier data structure, 
		// does not eliminate implicit fusion orbits
	INT test_point_using_check_functions(generator *gen, 
		INT lvl, INT rep, INT *the_set, 
		INT verbose_level);
		// called by check_orbits and downstep_apply_early_test 
		// Calls gen->check_the_set_incrementally 
		// (if gen->f_candidate_incremental_check_func).
		// Otherwise, calls gen->check_the_set 
		// (if gen->f_candidate_check_func).
		// Otherwise accepts any point.
	void relabel_schreier_vector(action &AR, INT verbose_level);
		// called from compute_schreier_vector, 
		// downstep_orbit_test_and_schreier_vector
	void downstep_orbits_print(generator *gen, 
		schreier &Schreier, action &AR, 
		INT lvl, 
		INT f_using_invariant_subset, INT f_print_orbits, 
		INT max_orbits, INT max_points_per_orbit);


	// oracle_downstep_subspace_action.C
	void setup_factor_space_action_light(generator *gen, 
		action_on_factor_space &AF, 
		INT lvl, INT verbose_level);
	void setup_factor_space_action_with_early_test(generator *gen, 
		action_on_factor_space &AF, action &A_factor_space, 
		INT lvl, INT verbose_level);
	void setup_factor_space_action(generator *gen, 
		action_on_factor_space &AF, action &A_factor_space, 
		INT lvl, INT f_compute_tables, 
		INT verbose_level);
	void downstep_subspace_action_print_orbits(
		generator *gen, schreier &Schreier, 
		INT lvl, 
		INT f_print_orbits, 
		INT verbose_level);
	void downstep_subspace_action(generator *gen, 
		INT lvl, 
		INT f_create_schreier_vector, INT f_compact, 
		INT f_use_invariant_subset_if_available, 
		INT f_implicit_fusion, 
		INT verbose_level);
	void downstep_orbits_subspace_action(
		generator *gen, schreier &Schreier, 
		INT lvl, 
		INT f_use_invariant_subset_if_available, 
		INT &f_using_invariant_subset, 
		INT verbose_level);
	void find_extensions_subspace_action(generator *gen, schreier &O, 
		action *A_factor_space, action_on_factor_space *AF, 
		INT lvl, INT f_implicit_fusion, INT verbose_level);
	void create_schreier_vector_wrapper_subspace_action(
		INT f_create_schreier_vector, INT f_compact, 
		schreier &Schreier, 
		action *A_factor_space, action_on_factor_space *AF, 
		INT verbose_level);
};


// oracle_downstep_subspace_action.C:
void schreier_vector_relabel_points(INT *sv, action_on_factor_space *AF, 
	INT f_compact, INT f_trivial_group, INT verbose_level);




// #############################################################################
// upstep_work.C:
// #############################################################################

typedef struct coset_table_entry coset_table_entry;

struct coset_table_entry {
	INT coset;
		// as in the loop in upstep_work::upstep_subspace_action
		// goes from 0 to degree - 1.

	INT node; // = final_node as computed by find_automorphism_by_tracing
	INT ex; // = final_ex as computed by find_automorphism_by_tracing
	INT type; // = return value of find_automorphism_by_tracing

	INT nb_times_image_of_called;
	INT nb_times_mult_called;
	INT nb_times_invert_called;
	INT nb_times_retrieve_called;
};

class upstep_work {
public:
	generator *gen;
	INT size;
	INT prev;
	INT prev_ex;
	INT cur;
	INT nb_fusion_nodes;
	INT nb_fuse_cur;
	INT nb_ext_cur;
	INT f_debug;
	INT f_implicit_fusion;
	INT f_indicate_not_canonicals;
	INT mod_for_printing;


	INT pt;
	INT pt_orbit_len;
	
	INT *path; // [size + 1] 
		// path[i] is the node that represents set[0,..,i-1]


	
	oracle *O_prev;
	oracle *O_cur;

	group *G;
	group *H;	
	longinteger_object go_G, go_H;

	INT coset;
	
	INT nb_cosets;
	INT nb_cosets_processed;
	coset_table_entry *coset_table;

	FILE *f;


	upstep_work();
	~upstep_work();
	void init(generator *gen, 
		INT size,
		INT prev,
		INT prev_ex,
		INT cur,
		INT f_debug,
		INT f_implicit_fusion,
		INT f_indicate_not_canonicals, 
		FILE *fp, 
		INT verbose_level);
		// called from generator::extend_node
	void handle_extension(INT &nb_fuse_cur, INT &nb_ext_cur, 
		INT verbose_level);
		// called from generator::extend_node
		// Calls handle_extension_fusion_type 
		// or handle_extension_unprocessed_type
		//
		// Handles the extension 'cur_ex' in node 'prev'.
		// We are extending a set of size 'size' 
		// to a set of size 'size' + 1. 
		// Calls oracle::init_extension_node for the 
		// new node that is (possibly) created
	void handle_extension_fusion_type(INT verbose_level);
		// called from upstep_work::handle_extension
		// Handles the extension 'cur_ex' in node 'prev'.
	void handle_extension_unprocessed_type(INT verbose_level);
		// called from upstep_work::handle_extension
		// calls init_extension_node
	INT init_extension_node(INT verbose_level);
		// Called from upstep_work::handle_extension_unprocessed_type
		// Calls upstep_subspace_action or upstep_for_sets, 
		// depending on the type of action
		// then changes the type of the extension to 
		// EXTENSION_TYPE_EXTENSION

		// Establishes a new node at depth 'size' 
		// (i.e., a set of size 'size') as an extension 
		// of a previous node (prev) at depth size - 1 
		// with respect to a given point (pt).
		// This function is to be called for the next 
		// free oracle node which will 
		// become the descendant of the previous node (prev).
		// the extension node corresponds to the point pt. 
		// returns FALSE if the set is not canonical 
		// (provided f_indicate_not_canonicals is TRUE)
	INT upstep_for_sets(INT verbose_level);
		// This routine is called from upstep_work::init_extension_node
		// It is testing a set of size 'size'. 
		// The newly added point is in gen->S[size - 1]
		// returns FALSE if the set is not canonical 
		// (provided f_indicate_not_canonicals is TRUE)
	void print_level_extension_info();
	void print_level_extension_coset_info();

	// upstep_work_subspace_action.C:
	INT upstep_subspace_action(INT verbose_level);
		// This routine is called from upstep_work::init_extension_node
		// It computes coset_table.
		// It is testing a set of size 'size'. 
		// The newly added point is in gen->S[size - 1]
		// The extension is initiated from node 'prev' 
		// and from extension 'prev_ex' 
		// The node 'prev' is at depth 'size' - 1 
		// returns FALSE if the set is not canonical 
		// (provided f_indicate_not_canonicals is TRUE)


	// upstep_work_trace.C:

	trace_result find_automorphism_by_tracing(
		INT &final_node, INT &final_ex, INT f_tolerant, 
		INT verbose_level);
	trace_result find_automorphism_by_tracing_recursion(
		INT lvl, INT current_node, INT &final_node, INT &final_ex, 
		INT f_tolerant, INT verbose_level);
	trace_result handle_last_level(
		INT lvl, INT current_node, INT current_extension, INT pt0, 
		INT &final_node, INT &final_ex,  
		INT verbose_level);
	trace_result start_over(
		INT lvl, INT current_node, 
		INT &final_node, INT &final_ex, 
		INT f_tolerant, INT verbose_level);
};

// in upstep_work.C:
void print_coset_table(coset_table_entry *coset_table, INT len);



