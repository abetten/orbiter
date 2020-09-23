// data_structures.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005


#ifndef ORBITER_SRC_LIB_GROUP_ACTIONS_DATA_STRUCTURES_DATA_STRUCTURES_H_
#define ORBITER_SRC_LIB_GROUP_ACTIONS_DATA_STRUCTURES_DATA_STRUCTURES_H_



namespace orbiter {

namespace group_actions {




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
	void init_strong_generators_by_handle_and_with_tl(
			std::vector<int> &gen_handle,
			std::vector<int> &tl, int verbose_level);
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
// incidence_structure_with_group.cpp
// #############################################################################



//! to represent an incidence structure and its group


class incidence_structure_with_group {

public:

	incidence_structure *Inc;
	int N; // Inc->nb_rows + Inc->nb_cols;

	int *partition;

	int f_has_canonical_form;
	uchar *canonical_form; // [canonical_form_len]
	int canonical_form_len;

	int f_has_canonical_labeling;
	long int *canonical_labeling;  // [nb_rows + nb_cols]

	action *A_perm; // degree = N

	incidence_structure_with_group();
	~incidence_structure_with_group();
	void null();
	void freeself();
	void init(incidence_structure *Inc,
		int *partition,
		int verbose_level);
	void print_canonical_form(std::ostream &ost);
	void set_stabilizer_and_canonical_form(
			int f_save_incma_in_and_out, const char *save_incma_in_and_out_prefix,
			int f_compute_canonical_form,
			int verbose_level);
};

// #############################################################################
// object_in_projective_space_with_action.cpp
// #############################################################################



//! to represent an object in projective space


class object_in_projective_space_with_action {

public:

	object_in_projective_space *OiP;
		// do not free
	//strong_generators *Aut_gens;
		// generators for the automorphism group
	long int ago;
	int nb_rows, nb_cols;
	long int *canonical_labeling;


	object_in_projective_space_with_action();
	~object_in_projective_space_with_action();
	void null();
	void freeself();
	void init(object_in_projective_space *OiP,
		//strong_generators *Aut_gens,
		long int ago,
		int nb_rows, int nb_cols,
		long int *canonical_labeling,
		int verbose_level);
#if 0
	void init_known_ago(
		object_in_projective_space *OiP,
		long int known_ago,
		int nb_rows, int nb_cols,
		long int *canonical_labeling,
		int verbose_level);
#endif
};


// #############################################################################
// orbit_rep.cpp
// #############################################################################


//! to hold one orbit after reading files from Orbiters poset classification


class orbit_rep {
public:
	char prefix[1000];
	action *A;
	void (*early_test_func_callback)(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level);
	void *early_test_func_callback_data;

	int level;
	int orbit_at_level;
	int nb_cases;

	long int *rep;

	sims *Stab;
	strong_generators *Strong_gens;

	longinteger_object *stab_go;
	long int *candidates;
	int nb_candidates;


	orbit_rep();
	~orbit_rep();
	void null();
	void freeself();
	void init_from_file(action *A, std::string &prefix,
		int level, int orbit_at_level, int level_of_candidates_file,
		void (*early_test_func_callback)(long int *S, int len,
			long int *candidates, int nb_candidates,
			long int *good_candidates, int &nb_good_candidates,
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
			std::string &fname, int verbose_level);
	void read_from_file_one_case_only(
			action *A, action *A2, std::string &fname,
			int case_nr, int verbose_level);
	tally *get_ago_distribution(long int *&ago,
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





//! A collection of invariants called orbit type associated with a system of sets. The orbit types are based on the orbits of a given group.



class orbit_type_repository {

public:

	orbits_on_something *Oos;

	int nb_sets;
	int set_size;
	long int *Sets; // [nb_sets * set_size]
		// A system of sets that is given
	long int goi;

	int orbit_type_size;
		// the size of the invariant
	long int *Type_repository; // [nb_sets * orbit_type_size]
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
	long int *Type_representatives; // [nb_types]
		// The distinct types that appear in the Type_repository

	orbit_type_repository();
	~orbit_type_repository();
	void null();
	void freeself();
	void init(
			orbits_on_something *Oos,
			int nb_sets,
			int set_size,
			long int *Sets,
			long int goi,
			int verbose_level);
	void report(std::ostream &ost);
	void report_one_type(std::ostream &ost, int type_idx);

};


// #############################################################################
// projective_space_job_description.cpp
// #############################################################################





//! description of a job to be applied to a set in projective space PG(n,q)



class projective_space_job_description {


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
	std::string poly;

	int f_embed;
		// follow up option for f_print:
		//f_orthogonal, orthogonal_epsilon

	int f_andre;
		// follow up option for f_andre:
		int f_Q;
		int Q;
		int f_poly_Q;
		std::string poly_Q;


	int f_print;
		// follow up option for f_print:
		int f_lines_in_PG;
		int f_points_in_PG;
		int f_points_on_grassmannian;
		int points_on_grassmannian_k;
		int f_orthogonal;
		int orthogonal_epsilon;
		int f_homogeneous_polynomials_LEX;
		int f_homogeneous_polynomials_PART;
		int homogeneous_polynomials_degree;



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
	std::string test_diagonals_fname;
	int f_klein;

	int f_draw_points_in_plane;
		std::string draw_points_in_plane_fname_base;
		// follow up option for f_draw_points_in_plane:

		int f_point_labels;
		int f_embedded;
		int f_sideways;

	int f_canonical_form;
	const char *canonical_form_fname_base;
	int f_ideal_LEX;
	int f_ideal_PART;
	int ideal_degree;
	//int f_find_Eckardt_points_from_arc = FALSE;

	int f_intersect_with_set_from_file;
	std::string intersect_with_set_from_file_fname;

	int f_arc_with_given_set_as_s_lines_after_dualizing;
	int arc_size;
	int arc_d;
	int arc_d_low;
	int arc_s;

	int f_arc_with_two_given_sets_of_lines_after_dualizing;
	int arc_t;
	const char *t_lines_string;

	int f_arc_with_three_given_sets_of_lines_after_dualizing;
	int arc_u;
	const char *u_lines_string;

	int f_dualize_hyperplanes_to_points;
	int f_dualize_points_to_hyperplanes;




	projective_space_job_description();
	~projective_space_job_description();
	void read_arguments_from_string(
			const char *str, int verbose_level);
	int read_arguments(
		int argc, const char **argv,
		int verbose_level);

};



// #############################################################################
// projective_space_job.cpp
// #############################################################################



//! perform a job for a set in projective space PG(n,q) as described by projective_space_job_description


class projective_space_job {


	int t0;
	finite_field *F;
	projective_space_with_action *PA;
	int back_end_counter;


public:

	projective_space_job_description *Descr;

	int f_homogeneous_polynomial_domain_has_been_allocated;
	homogeneous_polynomial_domain *HPD;

	int intersect_with_set_from_file_set_has_beed_read;
	long int *intersect_with_set_from_file_set;
	int intersect_with_set_from_file_set_size;

	long int *t_lines;
	int nb_t_lines;
	long int *u_lines;
	int nb_u_lines;


	projective_space_job();
	void perform_job(projective_space_job_description *Descr, int verbose_level);
	void back_end(int input_idx,
			object_in_projective_space *OiP,
			std::ostream &fp,
			std::ostream &fp_tex,
			int verbose_level);
	void perform_job_for_one_set(int input_idx,
		object_in_projective_space *OiP,
		long int *&the_set_out,
		int &set_size_out,
		std::ostream &fp_tex,
		int verbose_level);
	void do_canonical_form(
		long int *set, int set_size, int f_semilinear,
		const char *fname_base, int verbose_level);

};

// #############################################################################
// projective_space_object_classifier_description.cpp
// #############################################################################




//! description of a classification of objects using class projective_space_object_classifier



class projective_space_object_classifier_description {

public:

	int f_input;
	data_input_stream *Data;


	int f_save;
	std::string save_prefix;

	int f_report;
	std::string report_prefix;

	int fixed_structure_order_list_sz;
	int fixed_structure_order_list[1000];

	int f_max_TDO_depth;
	int max_TDO_depth;

	int f_classification_prefix;
	std::string classification_prefix;

	int f_save_incma_in_and_out;
	std::string save_incma_in_and_out_prefix;

	int f_save_canonical_labeling;

	int f_save_ago;

	int f_load_canonical_labeling;

	int f_load_ago;

	int f_save_cumulative_canonical_labeling;
	std::string cumulative_canonical_labeling_fname;

	int f_save_cumulative_ago;
	std::string cumulative_ago_fname;

	int f_save_cumulative_data;
	std::string cumulative_data_fname;

	int f_save_fibration;
	std::string fibration_fname;


	projective_space_object_classifier_description();
	~projective_space_object_classifier_description();
	int read_arguments(
		int argc, const char **argv,
		int verbose_level);

};


// #############################################################################
// projective_space_object_classifier.cpp
// #############################################################################




//! classification of objects in projective space PG(n,q) using graph a theoretic approach



class projective_space_object_classifier {

public:

	projective_space_object_classifier_description *Descr;

	projective_space_with_action *PA;

	int nb_objects_to_test;

	classify_bitvectors *CB;




	projective_space_object_classifier();
	~projective_space_object_classifier();
	void do_the_work(projective_space_object_classifier_description *Descr,
			projective_space_with_action *PA,
			int verbose_level);
	void classify_objects_using_nauty(
		int verbose_level);
	void process_multiple_objects_from_file(
			int file_type, int file_idx,
			std::string &input_data,
			std::string &input_data2,
			std::vector<std::vector<int> > &Cumulative_data,
			std::vector<long int> &Cumulative_Ago,
			std::vector<std::vector<int> > &Cumulative_canonical_labeling,
			std::vector<std::vector<std::pair<int, int> > > &Fibration,
			int verbose_level);
	void process_set_of_points(
			std::string &input_data,
			int verbose_level);
	void process_set_of_points_from_file(
			std::string &input_data,
			int verbose_level);
	void process_set_of_lines_from_file(
			std::string &input_data,
			int verbose_level);
	void process_set_of_packing(
			std::string &input_data,
			int verbose_level);
	int process_object(
		object_in_projective_space *OiP,
		strong_generators *&SG,
		long int *canonical_labeling, int &canonical_labeling_len,
		int &idx,
		int verbose_level);
	// returns f_found, which is TRUE if the object is rejected
	int process_object_with_known_canonical_labeling(
		object_in_projective_space *OiP,
		long int *canonical_labeling, int canonical_labeling_len,
		int &idx,
		int verbose_level);
	void save(
			std::string &output_prefix,
			int verbose_level);
	void latex_report(std::string &fname,
			std::string &prefix,
			int fixed_structure_order_list_sz,
			int *fixed_structure_order_list,
			int max_TDO_depth,
			int verbose_level);


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
	

	action *A; // linear group PGGL(d,q) in the action on points
	action *A_on_lines; // linear group PGGL(d,q) acting on lines
	//sims *S; // linear group PGGL(d,q)

	int *Elt1;


	projective_space_with_action();
	~projective_space_with_action();
	void null();
	void freeself();
	void init(finite_field *F, int n, int f_semilinear, 
		int f_init_incidence_structure, int verbose_level);
	void init_group(int f_semilinear, int verbose_level);
	void canonical_labeling(
		object_in_projective_space *OiP,
		int *canonical_labeling,
		int verbose_level);
	strong_generators *set_stabilizer_of_object(
		object_in_projective_space *OiP, 
		int f_save_incma_in_and_out, 
		std::string &save_incma_in_and_out_prefix,
		int f_compute_canonical_form, 
		uchar *&canonical_form, 
		int &canonical_form_len,
		long int *canonical_labeling, int &canonical_labeling_len,
		int verbose_level);
		// canonical_labeling[nb_rows + nb_cols] contains the canonical labeling
		// where nb_rows and nb_cols is the encoding size,
		// which can be computed using
		// object_in_projective_space::encoding_size(
		//   int &nb_rows, int &nb_cols,
		//   int verbose_level)
	void save_Levi_graph(std::string &prefix,
			const char *mask,
			int *Incma, int nb_rows, int nb_cols,
			long int *canonical_labeling, int canonical_labeling_len,
			int verbose_level);
	void report_fixed_objects_in_PG_3_tex(
		int *Elt, std::ostream &ost,
		int verbose_level);
	void report_orbits_in_PG_3_tex(
		int *Elt, std::ostream &ost,
		int verbose_level);
	void report_decomposition_by_single_automorphism(
		int *Elt, std::ostream &ost,
		int verbose_level);
	int process_object(
		classify_bitvectors *CB,
		object_in_projective_space *OiP,
		int f_save_incma_in_and_out, std::string &prefix,
		int nb_objects_to_test,
		strong_generators *&SG,
		long int *canonical_labeling,
		int verbose_level);
	void merge_packings(
			std::string *fnames, int nb_files,
			std::string &file_of_spreads,
			classify_bitvectors *&CB,
			int verbose_level);
	void select_packings(
			std::string &fname,
			std::string &file_of_spreads_original,
			spread_tables *Spread_tables,
			int f_self_dual,
			int f_ago, int select_ago,
			classify_bitvectors *&CB,
			int verbose_level);
	void select_packings_self_dual(
			std::string &fname,
			std::string &file_of_spreads_original,
			int f_split, int split_r, int split_m,
			spread_tables *Spread_tables,
			classify_bitvectors *&CB,
			int verbose_level);
	object_in_projective_space *create_object_from_string(
		int type, std::string &input_fname, int input_idx,
		std::string &set_as_string, int verbose_level);
	object_in_projective_space *create_object_from_int_vec(
		int type, std::string &input_fname, int input_idx,
		long int *the_set, int set_sz, int verbose_level);
};

//globals:
void OiPA_encode(void *extra_data,
	long int *&encoding, int &encoding_sz, void *global_data);
void OiPA_group_order(void *extra_data,
	longinteger_object &go, void *global_data);
void print_summary_table_entry(int *Table,
	int m, int n, int i, int j, int val, char *output, void *data);
void compute_ago_distribution(
	classify_bitvectors *CB, tally *&C_ago, int verbose_level);
void compute_ago_distribution_permuted(
	classify_bitvectors *CB, tally *&C_ago, int verbose_level);
void compute_and_print_ago_distribution(std::ostream &ost,
	classify_bitvectors *CB, int verbose_level);
void compute_and_print_ago_distribution_with_classes(
		std::ostream &ost,
	classify_bitvectors *CB, int verbose_level);
int table_of_sets_compare_func(void *data, int i,
		void *search_object,
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
	void print_info_and_generators(
			schreier_vector *S);
	int coset_rep_inv_lint(
			schreier_vector *S,
			long int pt, long int &pt0,
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
			int pt, long int *&orbit_elts, int &orbit_len,
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
	long int *data;
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
	long int group_order_as_lint();
	void init_everything(action *A, action *A2, long int *Set, int set_sz,
		strong_generators *gens, int verbose_level);
	void allocate_data(int sz, int verbose_level);
	set_and_stabilizer *create_copy(int verbose_level);
	void init_data(long int *data, int sz, int verbose_level);
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
	int find(long int pt);
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

	long int *set;
	int set_sz;
	int k;

	sims *S;

	long int *interesting_k_subsets;
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
		long int *set, int set_sz, int k,
		long int *interesting_k_subsets, int nb_interesting_k_subsets,
		int verbose_level);
	int is_minimal(int rk, int verbose_level);
};


}}


#endif /* ORBITER_SRC_LIB_GROUP_ACTIONS_DATA_STRUCTURES_DATA_STRUCTURES_H_ */



