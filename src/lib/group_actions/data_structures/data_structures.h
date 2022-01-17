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
// group_container.cpp
// #############################################################################


//! a container data structure for groups




class group_container {

public:
	action *A;

	int f_has_ascii_coding;
	char *ascii_coding;

	int f_has_strong_generators;
	vector_ge *SG;
	int *tl;
	
	int f_has_sims;
	sims *S;
	
	group_container();
	~group_container();
	void null();
	void freeself();
	group_container(action *A, int verbose_level);
	group_container(action *A, const char *ascii_coding, int verbose_level);
	group_container(action *A, vector_ge &SG, int *tl, int verbose_level);
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
	void point_stabilizer(group_container &stab, int pt, int verbose_level);
	void point_stabilizer_with_action(action *A2, 
			group_container &stab, int pt, int verbose_level);
	void induced_action(action &induced_action, 
			group_container &H, group_container &K, int verbose_level);
	void extension(group_container &N, group_container &H, int verbose_level);
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
	data_structures::bitvector *canonical_form;

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
	void set_stabilizer_and_canonical_form(
			int f_compute_canonical_form,
			incidence_structure *&Inc_out,
			int verbose_level);
};


// #############################################################################
// orbit_rep.cpp
// #############################################################################


//! to hold one orbit after reading files from Orbiters poset classification


class orbit_rep {
public:
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
			std::string &prefix,
			std::string &label_of_structure, std::ostream &ost,
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
	void create_latex_report(std::string &prefix, int verbose_level);
	void report(std::ostream &ost, int verbose_level);
	void report_one_type(std::ostream &ost, int type_idx, int verbose_level);

};






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
	data_structures::set_of_sets *get_orbits_as_set_of_sets(schreier_vector *Sv,
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
			int pt, long int *&orbit_elts, int &orbit_len, int &idx_of_root_node,
			int verbose_level);
	void init_from_schreier(schreier *S,
		int f_trivial_group, int verbose_level);
	void init_shallow_schreier_forest(schreier *S,
		int f_trivial_group, int f_randomized,
		int verbose_level);
	// initializes local_gens
	void export_tree_as_layered_graph(
			int orbit_no, int orbit_rep,
			std::string &fname_mask,
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
		int data_gens_size, int nb_gens, std::string &ascii_target_go,
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
	int find(long int pt);
};

// #############################################################################
// union_find.cpp
// #############################################################################


//! a union find data structure (used in the poset classification algorithm)




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

//! a union find data structure (used in the poset classification algorithm)



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

// #############################################################################
// vector_ge.cpp
// #############################################################################

//! to hold a vector of group elements


class vector_ge {

public:
	action *A;
	int *data;
	int len;

	vector_ge();
	vector_ge(action *A);
	~vector_ge();
	void null();
	void freeself();
	void init(action *A, int verbose_level);
	void copy(vector_ge *&vector_copy, int verbose_level);
	void init_by_hdl(action *A, int *gen_hdl, int nb_gen, int verbose_level);
	void init_single(action *A, int *Elt, int verbose_level);
	void init_double(action *A, int *Elt1, int *Elt2, int verbose_level);
	void init_from_permutation_representation(action *A, sims *S, int *data,
		int nb_elements, int verbose_level);
		// data[nb_elements * degree]
	void init_from_data(action *A, int *data,
		int nb_elements, int elt_size, int verbose_level);
	void init_conjugate_svas_of(vector_ge *v, int *Elt,
		int verbose_level);
	void init_conjugate_sasv_of(vector_ge *v, int *Elt,
		int verbose_level);
	int *ith(int i);
	void print(std::ostream &ost);
	void print_quick(std::ostream& ost);
	void print_tex(std::ostream &ost);
	void print_generators_tex(
			foundations::longinteger_object &go,
			std::ostream &ost);
	void print_as_permutation(std::ostream& ost);
	void allocate(int length, int verbose_level);
	void reallocate(int new_length, int verbose_level);
	void reallocate_and_insert_at(int position, int *elt, int verbose_level);
	void insert_at(int length_before, int position, int *elt, int verbose_level);
	void append(int *elt, int verbose_level);
	void copy_in(int i, int *elt);
	void copy_out(int i, int *elt);
	void conjugate_svas(int *Elt);
	void conjugate_sasv(int *Elt);
	void print_with_given_action(std::ostream &ost, action *A2);
	void print(std::ostream &ost, int f_print_as_permutation,
		int f_offset, int offset,
		int f_do_it_anyway_even_for_big_degree,
		int f_print_cycles_of_length_one);
	void print_for_make_element(std::ostream &ost);
	void write_to_memory_object(
		foundations::memory_object *m,
		int verbose_level);
	void read_from_memory_object(
		foundations::memory_object *m,
		int verbose_level);
	void write_to_file_binary(std::ofstream &fp,
		int verbose_level);
	void read_from_file_binary(std::ifstream &fp,
		int verbose_level);
	void write_to_csv_file_coded(std::string &fname, int verbose_level);
	void save_csv(std::string &fname, int verbose_level);
	void read_column_csv(std::string &fname, action *A, int col_idx, int verbose_level);
	void extract_subset_of_elements_by_rank_text_vector(
		const char *rank_vector_text, sims *S,
		int verbose_level);
	void extract_subset_of_elements_by_rank(int *rank_vector,
		int len, sims *S, int verbose_level);
	int test_if_all_elements_stabilize_a_point(action *A2, int pt);
	int test_if_all_elements_stabilize_a_set(action *A2,
		long int *set, int sz, int verbose_level);
	schreier *orbits_on_points_schreier(
			action *A_given, int verbose_level);
	void reverse_isomorphism_exterior_square(int verbose_level);
	void matrix_representation(
			action_on_homogeneous_polynomials *A_on_HPD, int *&M, int &nb_gens,
			int verbose_level);

};



}}


#endif /* ORBITER_SRC_LIB_GROUP_ACTIONS_DATA_STRUCTURES_DATA_STRUCTURES_H_ */



