// data_structures.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005

namespace orbiter {

namespace group_actions {


// #############################################################################
// data_input_stream.cpp:
// #############################################################################


#define INPUT_TYPE_SET_OF_POINTS 1
#define INPUT_TYPE_SET_OF_LINES 2
#define INPUT_TYPE_SET_OF_PACKING 3
#define INPUT_TYPE_FILE_OF_POINTS 4
#define INPUT_TYPE_FILE_OF_LINES 5
#define INPUT_TYPE_FILE_OF_PACKINGS 6
#define INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE 7
#define INPUT_TYPE_FILE_OF_POINT_SET 8



//! description of input data for classification of geometric objects from the command line


class data_input_stream {
public:
	int nb_inputs;
	int input_type[1000];
	const char *input_string[1000];
	const char *input_string2[1000];

	data_input_stream();
	~data_input_stream();
	void null();
	void freeself();
	void read_arguments_from_string(
			const char *str, int verbose_level);
	int read_arguments(int argc, const char **argv,
		int verbose_level);
	int count_number_of_objects_to_test(
		int verbose_level);
};



// #############################################################################
// group.cpp
// #############################################################################


//! a container data structure for groups




class group {

public:
	action *A;

	int f_has_ascii_coding;
	char *ascii_coding;

	int f_has_strong_generators;
	vector_ge *SG;
	int *tl;
	
	int f_has_sims;
	sims *S;
	
	group();
	~group();
	void null();
	void freeself();
	group(action *A, int verbose_level);
	group(action *A, const char *ascii_coding, int verbose_level);
	group(action *A, vector_ge &SG, int *tl, int verbose_level);
	void init(action *A, int verbose_level);
	void init_ascii_coding_to_sims(const char *ascii_coding, int verbose_level);
	void init_ascii_coding(const char *ascii_coding, int verbose_level);
	void delete_ascii_coding();
	void delete_sims();
	void init_strong_generators_empty_set(int verbose_level);
	void init_strong_generators(vector_ge &SG, int *tl, int verbose_level);
	void init_strong_generators_by_hdl(int nb_gen, int *gen_hdl, 
		int *tl, int verbose_level);
	void delete_strong_generators();
	void require_ascii_coding();
	void require_strong_generators();
	void require_sims();
	void group_order(longinteger_object &go);
	void print_group_order(std::ostream &ost);
	void print_tl();
	void code_ascii(int verbose_level);
	void decode_ascii(int verbose_level);
	void schreier_sims(int verbose_level);
	void get_strong_generators(int verbose_level);
	void point_stabilizer(group &stab, int pt, int verbose_level);
	void point_stabilizer_with_action(action *A2, 
		group &stab, int pt, int verbose_level);
	void induced_action(action &induced_action, 
		group &H, group &K, int verbose_level);
	void extension(group &N, group &H, int verbose_level);
		// N needs to have strong generators, 
		// H needs to have sims
		// N and H may have different actions, 
		// the action of N is taken for the extension.
	void print_strong_generators(std::ostream &ost,
		int f_print_as_permutation);
	void print_strong_generators_with_different_action(
			std::ostream &ost, action *A2);
	void print_strong_generators_with_different_action_verbose(
			std::ostream &ost, action *A2, int verbose_level);

};

// #############################################################################
// object_in_projective_space_with_action.cpp
// #############################################################################



//! to represent an object in projective space


class object_in_projective_space_with_action {

public:

	object_in_projective_space *OiP;
		// do not free
	strong_generators *Aut_gens;
		// generators for the automorphism group
	int nb_rows, nb_cols;
	int *canonical_labeling;


	object_in_projective_space_with_action();
	~object_in_projective_space_with_action();
	void null();
	void freeself();
	void init(object_in_projective_space *OiP,
		strong_generators *Aut_gens,
		int nb_rows, int nb_cols,
		int *canonical_labeling,
		int verbose_level);
	void init_known_ago(
		object_in_projective_space *OiP,
		int known_ago,
		int nb_rows, int nb_cols,
		int *canonical_labeling,
		int verbose_level);
};


// #############################################################################
// orbit_rep.cpp
// #############################################################################


//! to hold one orbit after reading files from Orbiters poset classification


class orbit_rep {
public:
	char prefix[1000];
	action *A;
	void (*early_test_func_callback)(int *S, int len,
		int *candidates, int nb_candidates,
		int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level);
	void *early_test_func_callback_data;

	int level;
	int orbit_at_level;
	int nb_cases;

	int *rep;

	sims *Stab;
	strong_generators *Strong_gens;

	longinteger_object *stab_go;
	int *candidates;
	int nb_candidates;


	orbit_rep();
	~orbit_rep();
	void null();
	void freeself();
	void init_from_file(action *A, char *prefix,
		int level, int orbit_at_level, int level_of_candidates_file,
		void (*early_test_func_callback)(int *S, int len,
			int *candidates, int nb_candidates,
			int *good_candidates, int &nb_good_candidates,
			void *data, int verbose_level),
		void *early_test_func_callback_data,
		int verbose_level);

};




// #############################################################################
// orbit_transversal.cpp
// #############################################################################

//! a set of orbits using a vector of orbit representatives and stabilizers


class orbit_transversal {

public:
	action *A;
	action *A2;
	
	int nb_orbits;
	set_and_stabilizer *Reps;

	orbit_transversal();
	~orbit_transversal();
	void null();
	void freeself();
	void init_from_schreier(
			schreier *Sch,
			action *default_action,
			longinteger_object &full_group_order,
			int verbose_level);
	void read_from_file(action *A, action *A2, 
		const char *fname, int verbose_level);
	classify *get_ago_distribution(int *&ago,
			int verbose_level);
	void report_ago_distribution(std::ostream &ost);
	void print_table_latex(
			std::ostream &f,
			int f_has_callback,
			void (*callback_print_function)(
					std::stringstream &ost, void *data, void *callback_data),
			void *callback_data,
			int f_has_callback2,
			void (*callback_print_function2)(
					std::stringstream &ost, void *data, void *callback_data),
			void *callback_data2,
			int verbose_level);
	void export_data_in_source_code_inside_tex(
			const char *prefix,
			char *label_of_structure, std::ostream &ost,
			int verbose_level);
};



// #############################################################################
// orbit_type_repository.cpp
// #############################################################################





//! A collection invariants called orbit type assciated with a system of sets. The orbit types are based on the orbits of a given group.



class orbit_type_repository {

public:

	orbits_on_something *Oos;

	int nb_sets;
	int set_size;
	int *Sets; // [nb_sets * set_size]
		// A system of sets that is gicen
	int goi;

	int orbit_type_size;
		// the size of the invariant
	int *Type_repository; // [nb_sets * orbit_type_size]
		// for each set, the orbit invariant

		// The next items are related to the classification of the
		// orbit invariant:

	int nb_types;
		// the number of distinct types that appear in the Type_repository
	int *type_first; // [nb_types]
	int *type_len; // [nb_types]
	int *type; // [nb_sets]
		// type[i] is the index into the Type_representatives of the
		// invariant associated with the i-th set in Sets[]
	int *Type_representatives; // [nb_types]
		// The distinct types that appear in the Type_repository

	orbit_type_repository();
	~orbit_type_repository();
	void null();
	void freeself();
	void init(
			orbits_on_something *Oos,
			int nb_sets,
			int set_size,
			int *Sets,
			int goi,
			int verbose_level);
	void report(std::ostream &ost);
	void report_one_type(std::ostream &ost, int type_idx);

};


// #############################################################################
// projective_space_job_description.cpp
// #############################################################################





//! description of a job to be applied to a set in projective space PG(n,q)



class projective_space_job_description {

	int t0;
	finite_field *F;
	projective_space_with_action *PA;
	int back_end_counter;

public:

	int f_input;
	data_input_stream *Data;


	int f_fname_base_out;
	const char *fname_base_out;


	int f_q;
	int q;
	int f_n;
	int n;
	int f_poly;
	const char *poly;

	int f_embed;
		// follow up option for f_print:
		//f_orthogonal, orthogonal_epsilon

	int f_andre;
		// follow up option for f_andre:
		int f_Q;
		int Q;
		int f_poly_Q;
		const char *poly_Q;


	int f_print;
		// follow up option for f_print:
		int f_lines_in_PG;
		int f_points_in_PG;
		int f_points_on_grassmannian;
		int points_on_grassmannian_k;
		int f_orthogonal;
		int orthogonal_epsilon;
		int f_homogeneous_polynomials;
		int homogeneous_polynomials_degree;
		int f_homogeneous_polynomial_domain_has_been_allocated;
		homogeneous_polynomial_domain *HPD;


	//int f_group = FALSE;
	int f_list_group_elements;
	int f_line_type;
	int f_plane_type;
	int f_plane_type_failsafe;
	int f_conic_type;
		// follow up option for f_conic_type:
		int f_randomized;
		int nb_times;

	int f_hyperplane_type;
	// follow up option for f_hyperplane_type:
		int f_show;


	int f_cone_over;

	//int f_move_line = FALSE;
	//int from_line = 0, to_line = 0;

	int f_bsf3;
	int f_test_diagonals;
	const char *test_diagonals_fname;
	int f_klein;

	int f_draw_points_in_plane;
		const char *draw_points_in_plane_fname_base;
		// follow up option for f_draw_points_in_plane:

		int f_point_labels;
		int f_embedded;
		int f_sideways;

	int f_canonical_form;
	const char *canonical_form_fname_base;
	int f_ideal;
	int ideal_degree;
	//int f_find_Eckardt_points_from_arc = FALSE;

	int f_intersect_with_set_from_file;
	const char *intersect_with_set_from_file_fname;
	int intersect_with_set_from_file_set_has_beed_read;
	int *intersect_with_set_from_file_set;
	int intersect_with_set_from_file_set_size;

	int f_arc_with_given_set_as_s_lines_after_dualizing;
	int arc_size;
	int arc_d;
	int arc_d_low;
	int arc_s;

	int f_arc_with_two_given_sets_of_lines_after_dualizing;
	int arc_t;
	const char *t_lines_string;
	int *t_lines;
	int nb_t_lines;

	int f_arc_with_three_given_sets_of_lines_after_dualizing;
	int arc_u;
	const char *u_lines_string;
	int *u_lines;
	int nb_u_lines;




	projective_space_job_description();
	~projective_space_job_description();
	void read_arguments_from_string(
			const char *str, int verbose_level);
	int read_arguments(
		int argc, const char **argv,
		int verbose_level);
	void perform_job(int verbose_level);
	void back_end(int input_idx,
			object_in_projective_space *OiP,
			std::ostream &fp,
			std::ostream &fp_tex,
			int verbose_level);
	void perform_job_for_one_set(int input_idx,
		object_in_projective_space *OiP,
		int *&the_set_out,
		int &set_size_out,
		std::ostream &fp_tex,
		int verbose_level);
	void do_canonical_form(
		int *set, int set_size, int f_semilinear,
		const char *fname_base, int verbose_level);

};


// #############################################################################
// projective_space_with_action.cpp
// #############################################################################




//! projective space PG(n,q) with automorphism group PGGL(n+1,q)



class projective_space_with_action {

public:

	int n;
	int d; // n + 1
	int q;
	finite_field *F; // do not free
	int f_semilinear;
	int f_init_incidence_structure;

	projective_space *P;
	

	action *A; // linear group PGGL(d,q)
	action *A_on_lines; // linear group PGGL(d,q) acting on lines
	sims *S; // linear group PGGL(d,q)

	int *Elt1;


	projective_space_with_action();
	~projective_space_with_action();
	void null();
	void freeself();
	void init(finite_field *F, int n, int f_semilinear, 
		int f_init_incidence_structure, int verbose_level);
	void init_group(int f_semilinear, int verbose_level);
	strong_generators *set_stabilizer(
		int *set, int set_size, int &canonical_pt, 
		int *canonical_set_or_NULL, 
		int f_save_incma_in_and_out, 
		const char *save_incma_in_and_out_prefix, 
		int f_compute_canonical_form, 
		uchar *&canonical_form, int &canonical_form_len, 
		int verbose_level);
	void canonical_labeling(
		object_in_projective_space *OiP,
		int *canonical_labeling,
		int verbose_level);
	strong_generators *set_stabilizer_of_object(
		object_in_projective_space *OiP, 
		int f_save_incma_in_and_out, 
		const char *save_incma_in_and_out_prefix, 
		int f_compute_canonical_form, 
		uchar *&canonical_form, 
		int &canonical_form_len, 
		int *canonical_labeling,
		int verbose_level);
		// canonical_labeling[nb_rows + nb_cols]
		// where nb_rows and nb_cols is the encoding size,
		// which can be computed using
		// object_in_projective_space::encoding_size(
		//   int &nb_rows, int &nb_cols,
		//   int verbose_level)
	void report_fixed_objects_in_PG_3_tex(
		int *Elt, std::ostream &ost,
		int verbose_level);
	void report_orbits_in_PG_3_tex(
		int *Elt, std::ostream &ost,
		int verbose_level);
	void report_decomposition_by_single_automorphism(
		int *Elt, std::ostream &ost,
		int verbose_level);
	object_in_projective_space *
	create_object_from_string(
		int type, const char *input_fname, int input_idx,
		const char *set_as_string, int verbose_level);
	object_in_projective_space *
	create_object_from_int_vec(
		int type, const char *input_fname, int input_idx,
		int *the_set, int set_sz, int verbose_level);
	int process_object(
		classify_bitvectors *CB,
		object_in_projective_space *OiP,
		int f_save_incma_in_and_out, const char *prefix,
		int nb_objects_to_test,
		strong_generators *&SG,
		int *canonical_labeling,
		int verbose_level);
	void classify_objects_using_nauty(
		data_input_stream *Data,
		classify_bitvectors *CB,
		int f_save_incma_in_and_out, const char *prefix,
		int verbose_level);
	void save(const char *output_prefix,
			classify_bitvectors *CB,
			int verbose_level);
	void merge_packings(
			const char **fnames, int nb_files,
			const char *file_of_spreads,
			classify_bitvectors *&CB,
			int verbose_level);
	void select_packings(
			const char *fname,
			const char *file_of_spreads_original,
			spread_tables *Spread_tables,
			int f_self_dual,
			int f_ago, int select_ago,
			classify_bitvectors *&CB,
			int verbose_level);
	void select_packings_self_dual(
			const char *fname,
			const char *file_of_spreads_original,
			int f_split, int split_r, int split_m,
			spread_tables *Spread_tables,
			classify_bitvectors *&CB,
			int verbose_level);
	void latex_report(const char *fname,
			const char *prefix,
			classify_bitvectors *CB,
			int f_save_incma_in_and_out,
			int fixed_structure_order_list_sz,
			int *fixed_structure_order_list,
			int max_TDO_depth,
			int verbose_level);
};

//globals:
void OiPA_encode(void *extra_data,
	int *&encoding, int &encoding_sz, void *global_data);
void OiPA_group_order(void *extra_data,
	longinteger_object &go, void *global_data);
void print_summary_table_entry(int *Table,
	int m, int n, int i, int j, int val, char *output, void *data);
void compute_ago_distribution(
	classify_bitvectors *CB, classify *&C_ago, int verbose_level);
void compute_ago_distribution_permuted(
	classify_bitvectors *CB, classify *&C_ago, int verbose_level);
void compute_and_print_ago_distribution(std::ostream &ost,
	classify_bitvectors *CB, int verbose_level);
void compute_and_print_ago_distribution_with_classes(
		std::ostream &ost,
	classify_bitvectors *CB, int verbose_level);
int table_of_sets_compare_func(void *data, int i,
		int *search_object,
		void *extra_data);


// #############################################################################
// schreier_vector_handler.cpp:
// #############################################################################

//! manages access to schreier vectors


class schreier_vector_handler {
public:
	action *A;
	action *A2;
	int *cosetrep;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int f_check_image;
	int f_allow_failure;
	int nb_calls_to_coset_rep_inv;
	int nb_calls_to_coset_rep_inv_recursion;

	schreier_vector_handler();
	~schreier_vector_handler();
	void null();
	void freeself();
	void init(action *A, action *A2,
			int f_allow_failure,
			int verbose_level);
	int coset_rep_inv(
			schreier_vector *S,
			int pt, int &pt0,
			int verbose_level);
	int coset_rep_inv_recursion(
		schreier_vector *S,
		int pt, int &pt0,
		int verbose_level);
	schreier_vector *sv_read_file(
			int gen_hdl_first, int nb_gen,
			std::ifstream &fp, int verbose_level);
	void sv_write_file(schreier_vector *Sv,
			std::ofstream &fp, int verbose_level);
	set_of_sets *get_orbits_as_set_of_sets(schreier_vector *Sv,
			int verbose_level);

};

// #############################################################################
// schreier_vector.cpp:
// #############################################################################

//! compact storage of schreier vectors


class schreier_vector {
public:
	int gen_hdl_first;
	int nb_gen;
	int number_of_orbits;
	int *sv;
		// the length of sv is n+1 if the group is trivial
		// and 3*n + 1 otherwise.
		//
		// sv[0] = n = number of points in the set on which we act
		// the next n entries are the points of the set
		// the next 2*n entries only exist if the group is non-trivial:
		// the next n entries are the previous pointers
		// the next n entries are the labels
	int f_has_local_generators;
	vector_ge *local_gens;

	schreier_vector();
	~schreier_vector();
	void null();
	void freeself();
	void init(int gen_hdl_first, int nb_gen, int *sv,
			int verbose_level);
	void init_local_generators(
			vector_ge *gens,
			int verbose_level);
	void set_sv(int *sv, int verbose_level);
	int *points();
	int *prev();
	int *label();
	int get_number_of_points();
	int get_number_of_orbits();
	int count_number_of_orbits();
	void count_number_of_orbits_and_get_orbit_reps(
			int *&orbit_reps, int &nb_orbits);
	int determine_depth_recursion(
		int n, int *pts, int *prev,
		int *depth, int *ancestor, int pos);
	void relabel_points(
		action_on_factor_space *AF,
		int verbose_level);
	void orbit_stats(
			int &nb_orbits, int *&orbit_reps, int *&orbit_length, int *&total_depth,
			int verbose_level);
	void orbit_of_point(
			int pt, int *&orbit_elts, int &orbit_len,
			int verbose_level);
	void init_from_schreier(schreier *S,
		int f_trivial_group, int verbose_level);
	void init_shallow_schreier_forest(schreier *S,
		int f_trivial_group, int f_randomized,
		int verbose_level);
	// initializes local_gens
	void export_tree_as_layered_graph(
			int orbit_no, int orbit_rep,
			const char *fname_mask,
			int verbose_level);
	void trace_back(int pt, int &depth);
	void print();
};



// #############################################################################
// set_and_stabilizer.cpp
// #############################################################################


//! a set and its known set stabilizer



class set_and_stabilizer {

public:
	action *A;
	action *A2;
	int *data;
	int sz;
	longinteger_object target_go;
	strong_generators *Strong_gens;
	sims *Stab;

	set_and_stabilizer();
	~set_and_stabilizer();
	void null();
	void freeself();
	void init(action *A, action *A2, int verbose_level);
	void group_order(longinteger_object &go);
	int group_order_as_int();
	void init_everything(action *A, action *A2, int *Set, int set_sz, 
		strong_generators *gens, int verbose_level);
	void allocate_data(int sz, int verbose_level);
	set_and_stabilizer *create_copy(int verbose_level);
	void init_data(int *data, int sz, int verbose_level);
	void init_stab_from_data(int *data_gens, 
		int data_gens_size, int nb_gens, const char *ascii_target_go, 
		int verbose_level);
	void init_stab_from_file(const char *fname_gens, 
		int verbose_level);
	void print_set_tex(std::ostream &ost);
	void print_set_tex_for_inline_text(std::ostream &ost);
	void print_generators_tex(std::ostream &ost);
	//set_and_stabilizer *apply(int *Elt, int verbose_level);
	void apply_to_self(int *Elt, int verbose_level);
	void apply_to_self_inverse(int *Elt, int verbose_level);
	void apply_to_self_element_raw(int *Elt_data, int verbose_level);
	void apply_to_self_inverse_element_raw(int *Elt_data, 
		int verbose_level);
	void rearrange_by_orbits(int *&orbit_first, 
		int *&orbit_length, int *&orbit, 
		int &nb_orbits, int verbose_level);
	action *create_restricted_action_on_the_set(int verbose_level);
	void print_restricted_action_on_the_set(int verbose_level);
	void test_if_group_acts(int verbose_level);
	
	void init_surface(surface_domain *Surf, action *A, action *A2,
		int q, int no, int verbose_level);
};

// #############################################################################
// union_find.cpp
// #############################################################################


//! a union find data structure (used in the poset classification)




class union_find {

public:
	action *A;
	int *prev;


	union_find();
	~union_find();
	void freeself();
	void null();
	void init(action *A, int verbose_level);
	int ancestor(int i);
	int count_ancestors();
	int count_ancestors_above(int i0);
	void do_union(int a, int b);
	void print();
	void add_generators(vector_ge *gens, int verbose_level);
	void add_generator(int *Elt, int verbose_level);
};

// #############################################################################
// union_find_on_k_subsets.cpp
// #############################################################################

//! a union find data structure (used in the poset classification)



class union_find_on_k_subsets {

public:

	int *set;
	int set_sz;
	int k;

	sims *S;

	int *interesting_k_subsets;
	int nb_interesting_k_subsets;
	
	action *A_original;
	action *Ar; // restricted action on the set
	action *Ar_perm;
	action *Ark; // Ar_perm on k_subsets
	action *Arkr; // Ark restricted to interesting_k_subsets

	vector_ge *gens_perm;

	union_find *UF;


	union_find_on_k_subsets();
	~union_find_on_k_subsets();
	void freeself();
	void null();
	void init(action *A_original, sims *S, 
		int *set, int set_sz, int k, 
		int *interesting_k_subsets, int nb_interesting_k_subsets, 
		int verbose_level);
	int is_minimal(int rk, int verbose_level);
};


}}

