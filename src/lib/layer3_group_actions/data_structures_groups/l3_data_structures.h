// data_structures.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005


#ifndef ORBITER_SRC_LIB_GROUP_ACTIONS_DATA_STRUCTURES_DATA_STRUCTURES_H_
#define ORBITER_SRC_LIB_GROUP_ACTIONS_DATA_STRUCTURES_DATA_STRUCTURES_H_



namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {



// #############################################################################
// export_group.cpp
// #############################################################################

//! a data structure to export and import groups internally

class export_group {

public:

	actions::action *A_base;
	actions::action *A_induced;
	groups::strong_generators *Strong_gens;

	export_group();
	~export_group();
	void init(
			actions::action *A_base,
			actions::action *A_induced,
			groups::strong_generators *Strong_gens,
			int verbose_level);


};


// #############################################################################
// group_container.cpp
// #############################################################################


//! a container data structure for groups




class group_container {

public:
	actions::action *A;

	int f_has_ascii_coding;
	std::string ascii_coding;

	int f_has_strong_generators;
	vector_ge *SG;
	int *tl;
	
	int f_has_sims;
	groups::sims *S;
	
	group_container();
	~group_container();
	void init(
			actions::action *A,
			int verbose_level);
	void init_ascii_coding_to_sims(
			std::string &ascii_coding, int verbose_level);
	void init_ascii_coding(
			std::string &ascii_coding, int verbose_level);
	void delete_ascii_coding();
	void delete_sims();
	void init_strong_generators_empty_set(
			int verbose_level);
	void init_strong_generators(
			vector_ge &SG,
			int *tl, int verbose_level);
	void init_strong_generators_by_handle_and_with_tl(
			std::vector<int> &gen_handle,
			std::vector<int> &tl, int verbose_level);
	void init_strong_generators_by_hdl(
			int nb_gen, int *gen_hdl,
		int *tl, int verbose_level);
	void delete_strong_generators();
	void require_ascii_coding();
	void require_strong_generators();
	void require_sims();
	void group_order(
			algebra::ring_theory::longinteger_object &go);
	void print_group_order(
			std::ostream &ost);
	void print_tl();
	void code_ascii(
			int verbose_level);
	void decode_ascii(
			int verbose_level);
	void schreier_sims(
			int verbose_level);
	void get_strong_generators(
			int verbose_level);
	void point_stabilizer(
			group_container &stab, int pt,
			int verbose_level);
	void point_stabilizer_with_action(
			actions::action *A2,
			group_container &stab, int pt, int verbose_level);
	void induced_action(
			actions::action &induced_action,
			group_container &H, group_container &K,
			int verbose_level);
	void extension(
			group_container &N,
			group_container &H, int verbose_level);
		// N needs to have strong generators, 
		// H needs to have sims
		// N and H may have different actions, 
		// the action of N is taken for the extension.
	void print_strong_generators(
			std::ostream &ost,
		int f_print_as_permutation);
	void print_strong_generators_with_different_action(
			std::ostream &ost, actions::action *A2);
	void print_strong_generators_with_different_action_verbose(
			std::ostream &ost,
			actions::action *A2,
			int verbose_level);

};



// #############################################################################
// group_table_and_generators.cpp
// #############################################################################


//! group table as matrix of integers and generator indices




class group_table_and_generators {

public:
	int *Table;
	int group_order;
	int *gens;
	int nb_gens;

	group_table_and_generators();
	~group_table_and_generators();
	void init(
			groups::sims *Sims,
			data_structures_groups::vector_ge *generators,
			int verbose_level);
	void init_basic(
			int *Table,
			int group_order,
			int *gens,
			int nb_gens,
			int verbose_level);


};



// #############################################################################
// hash_table_subgroups.cpp
// #############################################################################


//! fast access to an array of subgroups using a hash table




class hash_table_subgroups {

public:

	std::vector<void *> Subgroups;

	std::multimap<uint32_t, int> Hashing;
		// we store the pair (hash, idx)
		// where hash is the hash value of the set and idx is the
		// index in the table Sets where the set is stored.
		//
		// we use a multimap because the hash values are not unique
		// two sets may have the same hash value.
		// map cannot handle that.



	hash_table_subgroups();
	~hash_table_subgroups();
	int nb_groups();
	int add_subgroup(
			groups::subgroup *Subgroup,
			int verbose_level);
	int find_subgroup(
			groups::subgroup *Subgroup,
			int &pos, uint32_t &hash, int verbose_level);
	int find_subgroup_direct(
			int *Elements, int group_order,
			int &pos, uint32_t &hash, int verbose_level);
	groups::subgroup *get_subgroup(
			int group_idx);




};


// #############################################################################
// orbit_rep.cpp
// #############################################################################


//! to hold one orbit after reading files from Orbiters poset classification


class orbit_rep {
public:
	actions::action *A;
	void (*early_test_func_callback)(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level);
	void *early_test_func_callback_data;

	int level;
	int orbit_at_level;
	int nb_cases;

	long int *rep;

	groups::sims *Stab;
	groups::strong_generators *Strong_gens;

	algebra::ring_theory::longinteger_object *stab_go;
	long int *candidates;
	int nb_candidates;


	orbit_rep();
	~orbit_rep();
	void init_from_file(
			actions::action *A,
			std::string &prefix,
		int level, int orbit_at_level,
		int level_of_candidates_file,
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
	actions::action *A;
	actions::action *A2;
	
	int nb_orbits;
	set_and_stabilizer *Reps;

	orbit_transversal();
	~orbit_transversal();
	void init_from_schreier(
			groups::schreier *Sch,
			actions::action *default_action,
			algebra::ring_theory::longinteger_object &full_group_order,
			int verbose_level);
	void read_from_file(
			actions::action *A,
			actions::action *A2,
			std::string &fname, int verbose_level);
	void read_from_file_one_case_only(
			actions::action *A,
			actions::action *A2,
			std::string &fname,
			int case_nr,
			set_and_stabilizer *&Rep,
			int verbose_level);
	other::data_structures::tally *get_ago_distribution(
			long int *&ago,
			int verbose_level);
	void report_ago_distribution(
			std::ostream &ost);
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
// schreier_vector_handler.cpp:
// #############################################################################

//! manages access to schreier vectors


class schreier_vector_handler {
public:
	actions::action *A;
	actions::action *A2;
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
	void init(
			actions::action *A, actions::action *A2,
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
	void sv_write_file(
			schreier_vector *Sv,
			std::ofstream &fp, int verbose_level);
	other::data_structures::set_of_sets *get_orbits_as_set_of_sets(
			schreier_vector *Sv,
			int verbose_level);

};

// #############################################################################
// schreier_vector.cpp:
// #############################################################################

//! compact storage of schreier vectors for use in the poset classification data structure


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
	void init(
			int gen_hdl_first, int nb_gen, int *sv,
			int verbose_level);
	void init_local_generators(
			vector_ge *gens,
			int verbose_level);
	void set_sv(
			int *sv, int verbose_level);
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
			induced_actions::action_on_factor_space *AF,
		int verbose_level);
	void orbit_stats(
			int &nb_orbits, int *&orbit_reps,
			int *&orbit_length, int *&total_depth,
			int verbose_level);
	void orbit_of_point(
			int pt, long int *&orbit_elts,
			int &orbit_len, int &idx_of_root_node,
			int verbose_level);
	void init_from_schreier(
			groups::schreier *S,
		int f_trivial_group, int verbose_level);
	void init_shallow_schreier_forest(
			groups::schreier *S,
		int f_trivial_group, int f_randomized,
		int verbose_level);
	// initializes local_gens
	void export_tree_as_layered_graph(
			int orbit_no, int orbit_rep,
			std::string &fname_mask,
			int verbose_level);
	void trace_back(
			int pt, int &depth);
	void print();
};



// #############################################################################
// set_and_stabilizer.cpp
// #############################################################################


//! a set and its known set stabilizer



class set_and_stabilizer {

public:
	actions::action *A;
	actions::action *A2;
	long int *data;
	int sz;
	algebra::ring_theory::longinteger_object target_go;
	groups::strong_generators *Strong_gens;
	groups::sims *Stab;

	set_and_stabilizer();
	~set_and_stabilizer();
	void init(
			actions::action *A,
			actions::action *A2,
			int verbose_level);
	void group_order(
			algebra::ring_theory::longinteger_object &go);
	long int group_order_as_lint();
	void init_everything(
			actions::action *A,
			actions::action *A2,
			long int *Set, int set_sz,
			groups::strong_generators *gens,
			int verbose_level);
	void allocate_data(
			int sz, int verbose_level);
	set_and_stabilizer *create_copy(
			int verbose_level);
	void init_data(
			long int *data, int sz, int verbose_level);
	void init_stab_from_data(
			int *data_gens,
		int data_gens_size, int nb_gens,
		std::string &ascii_target_go,
		int verbose_level);
	void init_stab_from_file(
			std::string &fname_gens,
		int verbose_level);
	void print_set_tex(
			std::ostream &ost);
	void print_set_tex_for_inline_text(
			std::ostream &ost);
	void print_generators_tex(
			std::ostream &ost);
	void apply_to_self(
			int *Elt, int verbose_level);
	void apply_to_self_inverse(
			int *Elt, int verbose_level);
	void apply_to_self_element_raw(
			int *Elt_data, int verbose_level);
	void apply_to_self_inverse_element_raw(
			int *Elt_data,
		int verbose_level);
	void rearrange_by_orbits(
			int *&orbit_first,
		int *&orbit_length, int *&orbit, 
		int &nb_orbits,
		int verbose_level);
	actions::action *create_restricted_action_on_the_set(
			int verbose_level);
	void print_restricted_action_on_the_set(
			int verbose_level);
	void test_if_group_acts(
			int verbose_level);
	int find(
			long int pt);
};




// #############################################################################
// union_find.cpp
// #############################################################################


//! a union find data structure (used in the poset classification algorithm)




class union_find {

public:
	actions::action *A;
	int *prev;


	union_find();
	~union_find();
	void init(
			actions::action *A, int verbose_level);
	int ancestor(
			int i);
	int count_ancestors();
	int count_ancestors_above(
			int i0);
	void do_union(
			int a, int b);
	void print();
	void add_generators(
			vector_ge *gens, int verbose_level);
	void add_generator(
			int *Elt, int verbose_level);
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

	groups::sims *S;

	long int *interesting_k_subsets;
	int nb_interesting_k_subsets;
	
	actions::action *A_original;
	actions::action *Ar; // restricted action on the set
	actions::action *Ar_perm;
	actions::action *Ark; // Ar_perm on k_subsets
	actions::action *Arkr; // Ark restricted to interesting_k_subsets

	vector_ge *gens_perm;

	union_find *UF;


	union_find_on_k_subsets();
	~union_find_on_k_subsets();
	void init(
			actions::action *A_original, groups::sims *S,
		long int *set, int set_sz, int k,
		long int *interesting_k_subsets,
		int nb_interesting_k_subsets,
		int verbose_level);
	int is_minimal(
			int rk, int verbose_level);
};



// #############################################################################
// vector_ge_description.cpp
// #############################################################################



//! to define a vector of group elements


class vector_ge_description {

public:

	// TABLES/vector_ge_builder.tex

	int f_action;
	std::string action_label;

	int f_read_csv;
	std::string read_csv_fname;
	std::string read_csv_column_label;


	int f_vector_data;
	std::string vector_data_label;

	vector_ge_description();
	~vector_ge_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();
};




// #############################################################################
// vector_ge.cpp
// #############################################################################

//! a vector of group elements


class vector_ge {

public:
	actions::action *A;
	int *data;
	int len;

	vector_ge();
	~vector_ge();
	void null();
	void init(
			actions::action *A, int verbose_level);
	void copy(
			vector_ge *&vector_copy, int verbose_level);
	void init_by_hdl(
			actions::action *A,
			int *gen_hdl, int nb_gen, int verbose_level);
	void init_by_hdl(
			actions::action *A,
			std::vector<int> &gen_hdl, int verbose_level);
	void init_single(
			actions::action *A,
			int *Elt, int verbose_level);
	void init_double(
			actions::action *A,
			int *Elt1, int *Elt2, int verbose_level);
	void init_from_permutation_representation(
			actions::action *A, groups::sims *S, int *data,
		int nb_elements, int verbose_level);
		// data[nb_elements * degree]
	void init_from_data(
			actions::action *A, int *data,
		int nb_elements, int elt_size, int verbose_level);
	void init_transposed(
			vector_ge *v,
			int verbose_level);
	void init_conjugate_svas_of(
			vector_ge *v, int *Elt,
		int verbose_level);
	void init_conjugate_sasv_of(
			vector_ge *v, int *Elt,
		int verbose_level);
	vector_ge *make_inverses(
			int verbose_level);
	int *ith(
			int i);
	void print(
			std::ostream &ost);
	void print_quick(
			std::ostream& ost);
	void print_tex(
			std::ostream &ost);
	void report_elements(
			std::string &label,
			int f_with_permutation,
			int f_override_action, actions::action *A_special,
			int verbose_level);
	void report_elements_coded(
			std::string &label,
			int f_override_action, actions::action *A_special,
			int verbose_level);
	void print_generators_tex(
			algebra::ring_theory::longinteger_object &go,
			std::ostream &ost);
	void print_as_permutation(
			std::ostream& ost);
	void allocate(
			int length, int verbose_level);
	void reallocate(
			int new_length, int verbose_level);
	void reallocate_and_insert_at(
			int position, int *elt, int verbose_level);
	void insert_at(
			int length_before,
			int position, int *elt, int verbose_level);
	void append(
			int *elt, int verbose_level);
	void copy_in(
			int i, int *elt);
	void copy_out(
			int i, int *elt);
	void conjugate_svas(
			int *Elt);
	void conjugate_sasv(
			int *Elt);
	void print_with_given_action(
			std::ostream &ost, actions::action *A2);
	void print(
			std::ostream &ost,
			int f_print_as_permutation,
		int f_offset, int offset,
		int f_do_it_anyway_even_for_big_degree,
		int f_print_cycles_of_length_one,
		int verbose_level);
	void print_for_make_element(
			std::ostream &ost);
	void write_to_memory_object(
			other::orbiter_kernel_system::memory_object *m,
		int verbose_level);
	void read_from_memory_object(
			other::orbiter_kernel_system::memory_object *m,
		int verbose_level);
	void write_to_file_binary(
			std::ofstream &fp,
		int verbose_level);
	void read_from_file_binary(
			std::ifstream &fp,
		int verbose_level);
	void write_to_csv_file_coded(
			std::string &fname, int verbose_level);
	void save_csv(
			std::string &fname, int verbose_level);
	void export_inversion_graphs(
			std::string &fname, int verbose_level);
	void read_column_csv(
			std::string &fname,
			actions::action *A, int col_idx,
			int verbose_level);
	void read_column_csv_using_column_label(
			std::string &fname,
			actions::action *A,
			std::string &column_label,
			int verbose_level);
	void extract_subset_of_elements_by_rank_text_vector(
			std::string &rank_vector_text, groups::sims *S,
		int verbose_level);
	void extract_subset_of_elements_by_rank(
			int *rank_vector,
		int len, groups::sims *S, int verbose_level);
	void compute_rank_vector(
			long int *&rank_vector, groups::sims *Sims,
			int verbose_level);
	int test_if_all_elements_stabilize_a_point(
			actions::action *A2, int pt);
	int test_if_all_elements_stabilize_a_set(
			actions::action *A2,
		long int *set, int sz,
		int verbose_level);
	groups::schreier *compute_all_point_orbits_schreier(
			actions::action *A_given, int verbose_level);
	void reverse_isomorphism_exterior_square(
			int verbose_level);
	void matrix_representation(
			induced_actions::action_on_homogeneous_polynomials *A_on_HPD,
			int *&M, int &nb_gens,
			int verbose_level);
	void stab_BLT_set_from_catalogue(
			actions::action *A,
			algebra::field_theory::finite_field *F, int iso,
		std::string &target_go_text,
		int verbose_level);
	int test_if_in_set_stabilizer(
			actions::action *A,
			long int *set, int sz, int verbose_level);
	void print_generators_gap(
			std::ostream &ost, int verbose_level);
	void print_generators_gap_in_different_action(
			std::ostream &ost, actions::action *A2, int verbose_level);
	void print_generators_compact(
			std::ostream &ost, int verbose_level);
	void multiply_with(
			vector_ge **V, int nb_with, vector_ge *&result, int verbose_level);
	void conjugate_svas_to(
			int *Elt, vector_ge *&result, int verbose_level);
	void conjugate_sasv_to(
			int *Elt, vector_ge *&result, int verbose_level);
	void field_reduction(
			int subfield_index,
			int verbose_level);
	void rational_normal_form(
			int verbose_level);

};



}}}


#endif /* ORBITER_SRC_LIB_GROUP_ACTIONS_DATA_STRUCTURES_DATA_STRUCTURES_H_ */



