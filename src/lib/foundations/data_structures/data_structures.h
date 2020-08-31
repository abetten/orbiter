// data_structures.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


#ifndef ORBITER_SRC_LIB_FOUNDATIONS_DATA_STRUCTURES_DATA_STRUCTURES_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_DATA_STRUCTURES_DATA_STRUCTURES_H_



namespace orbiter {
namespace foundations {

// #############################################################################
// bitmatrix.cpp
// #############################################################################

//! matrices over GF(2) stored as bitvectors

class bitmatrix {
public:
	int m;
	int n;
	int N;
	uint32_t *data;

	bitmatrix();
	~bitmatrix();
	void init(int m, int n, int verbose_level);
	void unrank_PG_elements_in_columns_consecutively(
			finite_field *F, long int start_value, int verbose_level);
	void rank_PG_elements_in_columns(
			finite_field *F, int *perms, unsigned int *PG_ranks, int verbose_level);
	void print();
	void zero_out();
	int s_ij(int i, int j);
	void m_iji(int i, int j, int a);
	void mult_int_matrix_from_the_left(int *A, int Am, int An,
			bitmatrix *Out, int verbose_level);

};

// #############################################################################
// classify_bitvectors.cpp
// #############################################################################

//! classification of 0/1 matrices using canonical forms

class classify_bitvectors {
public:

	int nb_types;
		// the number of isomorphism types

	int rep_len;
		// the number of uchar we need to store the canonical form of
		// one object


	uchar **Type_data;
		// Type_data[N][rep_len]
		// the canonical form of the i-th representative is
		// Type_data[i][rep_len]
	int *Type_rep;
		// Type_rep[N]
		// Type_rep[i] is the index of the candidate which
		// has been chosen as representative
		// for the i-th isomorphism type
	int *Type_mult;
		// Type_mult[N]
		// Type_mult[i] gives the number of candidates so far which
		// are isomorphic to the i-th isomorphism class representative
	void **Type_extra_data;
		// Type_extra_data[N]
		// Type_extra_data[i] is a pointer that is stored with the
		// i-th isomorphism class representative

	int N;
		// number of candidates (or objects) that we will test
	int n;
		// number of candidates that we have already tested

	int *type_of;
		// type_of[N]
		// type_of[i] is the isomorphism type of the i-th candidate

	tally *C_type_of;
		// the classification of type_of[N]
		// this will be computed in finalize()

	int *perm;
		// the permutation which lists the orbit
		// representative in the order
		// in which they appear in the list of candidates

	classify_bitvectors();
	~classify_bitvectors();
	void null();
	void freeself();
	void init(int N, int rep_len, int verbose_level);
	int search(uchar *data, int &idx, int verbose_level);
	void search_and_add_if_new(uchar *data,
			void *extra_data, int &f_found, int &idx, int verbose_level);
	//int add(uchar *data, void *extra_data, int verbose_level);
	void finalize(int verbose_level);
	void print_reps();
	void print_table();
	void save(
			std::string &prefix,
		void (*encode_function)(void *extra_data,
			long int *&encoding, int &encoding_sz, void *global_data),
		void (*get_group_order_or_NULL)(void *extra_data,
			longinteger_object &go, void *global_data),
		void *global_data,
		int verbose_level);

};

int compare_func_for_bitvectors(void *a, void *b, void *data);


// #############################################################################
// data_file.cpp
// #############################################################################

//! to read data files from the poset classification algorithm


class data_file {
	
	public:

	std::string fname;
	int nb_cases;
	int *set_sizes;
	long int **sets;
	int *casenumbers;
	char **Ago_ascii;
	char **Aut_ascii;

	int f_has_candidates;
	int *nb_candidates;
	int **candidates;

	data_file();
	~data_file();
	void null();
	void freeself();
	void read(std::string &fname, int f_casenumbers, int verbose_level);
	void read_candidates(std::string &candidates_fname, int verbose_level);
};

// #############################################################################
// fancy_set.cpp
// #############################################################################

//! subset of size k of a set of size n


class fancy_set {
	
	public:

	int n;
	int k;
	long int *set;
	long int *set_inv;

	fancy_set();
	~fancy_set();
	void null();
	void freeself();
	void init(int n, int verbose_level);
	void init_with_set(int n, int k, int *subset, int verbose_level);
	void print();
	void println();
	void swap(int pos, int a);
	int is_contained(int a);
	void copy_to(fancy_set *to);
	void add_element(int elt);
	void add_elements(int *elts, int nb);
	void delete_elements(int *elts, int nb);
	void delete_element(int elt);
	void select_subset(int *elts, int nb);
	void intersect_with(int *elts, int nb);
	void subtract_set(fancy_set *set_to_subtract);
	void sort();
	int compare_lexicographically(fancy_set *second_set);
	void complement(fancy_set *compl_set);
	int is_subset(fancy_set *set2);
	int is_equal(fancy_set *set2);

};

// #############################################################################
// int_vector.cpp
// #############################################################################

//! vector of ints

class int_vector {
public:

	long int *M;
	int m;
	int alloc_length;

	int_vector();
	~int_vector();
	void null();
	void freeself();
	void allocate(int len);
	void allocate_and_init(int len, long int *V);
	void allocate_and_init_int(int len, int *V);
	void init_permutation_from_string(const char *s);
	void read_ascii_file(const char *fname);
	void read_binary_file_int4(const char *fname);
	long int &s_i(int i);
	int &length();
	void print(std::ostream &ost);
	void zero();
	int search(int a, int &idx);
	void sort();
	void make_space();
	void append(int a);
	void insert_at(int a, int idx);
	void insert_if_not_yet_there(int a);
	void sort_and_remove_duplicates();
	void write_to_ascii_file(const char *fname);
	void write_to_binary_file_int4(const char *fname);
	void write_to_csv_file(const char *fname, const char *label);
	int hash();
	int minimum();
	int maximum();



};

// #############################################################################
// page_storage.cpp
// #############################################################################

//! bulk storage of group elements in compressed form

class page_storage {

public:
	long int overall_length;

	long int entry_size; // in char
	long int page_length_log; // number of bits
	long int page_length; // entries per page
	long int page_size; // size in char of one page
	long int allocation_table_length;
		// size in char of one allocation table

	long int page_ptr_used;
	long int page_ptr_allocated;
	long int page_ptr_oversize;

	uchar **pages;
	uchar **allocation_tables;

	long int next_free_entry;
	long int nb_free_entries;

	int f_elt_print_function;
	void (* elt_print)(void *p, void *data, std::ostream &ost);
	void *elt_print_data;


	void init(int entry_size, int page_length_log,
		int verbose_level);
	void add_elt_print_function(
		void (* elt_print)(void *p, void *data, std::ostream &ost),
		void *elt_print_data);
	void print();
	uchar *s_i_and_allocate(long int i);
	uchar *s_i_and_deallocate(long int i);
	uchar *s_i(long int i);
	uchar *s_i_and_allocation_bit(long int i, int &f_allocated);
	void check_allocation_table();
	long int store(uchar *elt);
	void dispose(long int hdl);
	void check_free_list();
	page_storage();
	~page_storage();
	void print_storage_used();

};

void test_page_storage(int verbose_level);


// #############################################################################
// partitionstack.cpp
// #############################################################################


std::ostream& operator<<(std::ostream& ost, partitionstack& p);


//! partitionstack for set partitions following Jeffrey Leon


class partitionstack {
	public:

	// data structure for the partition stack,
	// following Leon:
		int n;
		int ht;
		int ht0;

		int *pointList, *invPointList;
		int *cellNumber;

		int *startCell;
		int *cellSize;
		int *parent;


	// for matrix canonization:
	// int first_column_element;

	// subset to be chosen by classify_by_..._extract_subset():
	// used as input for split_cell()
		//
	// used if SPLIT_MULTIPLY is defined:
		int nb_subsets;
		int *subset_first;
		int *subset_length;
		int *subsets;
		//
	// used if SPLIT_MULTIPLY is not defined:
		int *subset;
		int subset_size;

	partitionstack();
	~partitionstack();
	void allocate(int n, int verbose_level);
	void free();
	int parent_at_height(int h, int cell);
	int is_discrete();
	int smallest_non_discrete_cell();
	int biggest_non_discrete_cell();
	int smallest_non_discrete_cell_rows_preferred();
	int biggest_non_discrete_cell_rows_preferred();
	int nb_partition_classes(int from, int len);
	int is_subset_of_cell(int *set, int size, int &cell_idx);
	void sort_cells();
	void sort_cell(int cell);
	void reverse_cell(int cell);
	void check();
	void print_raw();
	void print_class(std::ostream& ost, int idx);
	void print_classes_tex(std::ostream& ost);
	void print_class_tex(std::ostream& ost, int idx);
	void print_class_point_or_line(std::ostream& ost, int idx);
	void print_classes(std::ostream& ost);
	void print_classes_points_and_lines(std::ostream& ost);
	std::ostream& print(std::ostream& ost);
	void print_cell(int i);
	void print_cell_latex(std::ostream &ost, int i);
	void print_subset();
	void write_cell_to_file(int i, char *fname, 
		int verbose_level);
	void write_cell_to_file_points_or_lines(int i, 
		char *fname, int verbose_level);
	void refine_arbitrary_set_lint(int size, long int *set, int verbose_level);
	void refine_arbitrary_set(int size, int *set, int verbose_level);
	void split_cell(int verbose_level);
	void split_multiple_cells(int *set, int set_size, 
		int f_front, int verbose_level);
	void split_line_cell_front_or_back(int *set, int set_size, 
		int f_front, int verbose_level);
	void split_cell_front_or_back(int *set, int set_size, 
		int f_front, int verbose_level);
	void split_cell(int *set, int set_size, int verbose_level);
	void join_cell();
	void reduce_height(int ht0);
	void isolate_point(int pt);
	void subset_continguous(int from, int len);
	int is_row_class(int c);
	int is_col_class(int c);
	void allocate_and_get_decomposition(
		int *&row_classes, int *&row_class_inv, 
			int &nb_row_classes,
		int *&col_classes, int *&col_class_inv, 
			int &nb_col_classes, 
		int verbose_level);
	void get_row_and_col_permutation(
		int *row_classes, int nb_row_classes,
		int *col_classes, int nb_col_classes, 
		int *row_perm, int *row_perm_inv, 
		int *col_perm, int *col_perm_inv);
	void get_row_and_col_classes(int *row_classes, 
		int &nb_row_classes,
		int *col_classes, int &nb_col_classes, 
		int verbose_level);
	void initial_matrix_decomposition(int nbrows, 
		int nbcols,
		int *V, int nb_V, int *B, int nb_B, 
		int verbose_level);
	int is_descendant_of(int cell, int ancestor_cell, 
		int verbose_level);
	int is_descendant_of_at_level(int cell, int ancestor_cell, 
	int level, int verbose_level);
	int cellSizeAtLevel(int cell, int level);

	// TDO for orthogonal:
	int compute_TDO(orthogonal &O, int ht0, 
		int marker1, int marker2, int depth, int verbose_level);
	void get_and_print_row_decomposition_scheme(orthogonal &O, 
		int marker1, int marker2);
	void get_and_print_col_decomposition_scheme(orthogonal &O, 
		int marker1, int marker2);
	void get_and_print_decomposition_schemes(orthogonal &O, 
		int marker1, int marker2);
	void print_decomposition_tex(std::ostream &ost,
		int *row_classes, int nb_row_classes,
		int *col_classes, int nb_col_classes);
	void print_decomposition_scheme(std::ostream &ost,
		int *row_classes, int nb_row_classes,
		int *col_classes, int nb_col_classes, 
		int *scheme, int marker1, int marker2);
	void print_decomposition_scheme_tex(std::ostream &ost,
		int *row_classes, int nb_row_classes,
		int *col_classes, int nb_col_classes, 
		int *scheme);
	void print_tactical_decomposition_scheme_tex_internal(
			std::ostream &ost, int f_enter_math_mode,
		int *row_classes, int nb_row_classes,
		int *col_classes, int nb_col_classes, 
		int *row_scheme, int *col_scheme, int f_print_subscripts);
	void print_tactical_decomposition_scheme_tex(std::ostream &ost,
		int *row_classes, int nb_row_classes,
		int *col_classes, int nb_col_classes, 
		int *row_scheme, int *col_scheme, int f_print_subscripts);
	void print_row_tactical_decomposition_scheme_tex(
			std::ostream &ost, int f_enter_math_mode,
		int *row_classes, int nb_row_classes,
		int *col_classes, int nb_col_classes, 
		int *row_scheme, int f_print_subscripts);
	void print_column_tactical_decomposition_scheme_tex(
			std::ostream &ost, int f_enter_math_mode,
		int *row_classes, int nb_row_classes,
		int *col_classes, int nb_col_classes, 
		int *col_scheme, int f_print_subscripts);
	void print_non_tactical_decomposition_scheme_tex(
			std::ostream &ost, int f_enter_math_mode,
		int *row_classes, int nb_row_classes,
		int *col_classes, int nb_col_classes, 
		int f_print_subscripts);
	void row_scheme_to_col_scheme(orthogonal &O, 
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int *row_scheme, int *col_scheme, int verbose_level);
	void get_row_decomposition_scheme(orthogonal &O, 
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int *row_scheme, int verbose_level);
	void get_col_decomposition_scheme(orthogonal &O, 
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int *col_scheme, int verbose_level);
	int refine_column_partition(orthogonal &O, int ht0, int verbose_level);
	int refine_row_partition(orthogonal &O, int ht0, int verbose_level);
	int hash_column_refinement_info(int ht0, int *data, int depth, 
		int hash0);
	int hash_row_refinement_info(int ht0, int *data, int depth, int hash0);
	void print_column_refinement_info(int ht0, int *data, int depth);
	void print_row_refinement_info(int ht0, int *data, int depth);
	void radix_sort(int left, int right, int *C, 
		int length, int radix, int verbose_level);
	void radix_sort_bits(int left, int right, 
		int *C, int length, int radix, int mask, int verbose_level);
	void swap_ij(int *perm, int *perm_inv, int i, int j);
	int my_log2(int m);
	void split_by_orbit_partition(int nb_orbits, 
		int *orbit_first, int *orbit_len, int *orbit,
		int offset, 
		int verbose_level);
};

// #############################################################################
// set_of_sets_lint.cpp
// #############################################################################

//! set of sets with entries over long int


class set_of_sets_lint {

public:

	long int underlying_set_size;
	int nb_sets;
	long int **Sets;
	int *Set_size;


	set_of_sets_lint();
	~set_of_sets_lint();
	void null();
	void freeself();
	void init_simple(long int underlying_set_size,
			int nb_sets, int verbose_level);
	void init(long int underlying_set_size,
			int nb_sets, long int **Pts, int *Sz, int verbose_level);
	void init_basic(long int underlying_set_size,
			int nb_sets, int *Sz, int verbose_level);
	void init_set(int idx_of_set,
			long int *set, int sz, int verbose_level);
};




// #############################################################################
// set_of_sets.cpp
// #############################################################################

//! set of sets


class set_of_sets {

public:
	
	int underlying_set_size;
	int nb_sets;
	long int **Sets;
	long int *Set_size;


	set_of_sets();
	~set_of_sets();
	void null();
	void freeself();
	set_of_sets *copy();
	void init_simple(int underlying_set_size, 
		int nb_sets, int verbose_level);
	void init_from_adjacency_matrix(int n, int *Adj, 
		int verbose_level);
	void init(int underlying_set_size, int nb_sets, 
		long int **Pts, long int *Sz, int verbose_level);
	void init_with_Sz_in_int(int underlying_set_size,
			int nb_sets, long int **Pts, int *Sz, int verbose_level);
	void init_basic(int underlying_set_size, 
		int nb_sets, long int *Sz, int verbose_level);
	void init_basic_with_Sz_in_int(int underlying_set_size,
			int nb_sets, int *Sz, int verbose_level);
	void init_basic_constant_size(int underlying_set_size, 
		int nb_sets, int constant_size, int verbose_level);
	void init_from_file(int underlying_set_size, 
		const char *fname, int verbose_level);
	void init_from_csv_file(int underlying_set_size, 
		const char *fname, int verbose_level);
	void init_from_orbiter_file(int underlying_set_size, 
		const char *fname, int verbose_level);
	void init_set(int idx_of_set, int *set, int sz, 
		int verbose_level);
		// Stores a copy of the given set.
	void init_cycle_structure(int *perm, int n, int verbose_level);
	int total_size();
	long int &element(int i, int j);
	void add_element(int i, long int a);
	void print();
	void print_table();
	void print_table_tex(std::ostream &ost);
	void dualize(set_of_sets *&S, int verbose_level);
	void remove_sets_of_given_size(int k, 
		set_of_sets &S, int *&Idx, 
		int verbose_level);
	void extract_largest_sets(set_of_sets &S, 
		int *&Idx, int verbose_level);
	void intersection_matrix(
		int *&intersection_type, int &highest_intersection_number, 
		int *&intersection_matrix, int &nb_big_sets, 
		int verbose_level);
	void compute_incidence_matrix(int *&Inc, int &m, int &n, 
		int verbose_level);
	void compute_and_print_tdo_row_scheme(std::ofstream &file,
		int verbose_level);
	void compute_and_print_tdo_col_scheme(std::ofstream &file,
		int verbose_level);
	void init_decomposition(decomposition *&D, int verbose_level);
	void compute_tdo_decomposition(decomposition &D, 
		int verbose_level);
	int is_member(int i, int a, int verbose_level);
	void sort_all(int verbose_level);
	void all_pairwise_intersections(set_of_sets *&Intersections, 
		int verbose_level);
	void pairwise_intersection_matrix(int *&M, int verbose_level);
	void all_triple_intersections(set_of_sets *&Intersections, 
		int verbose_level);
	int has_constant_size_property();
	int largest_set_size();
	void save_csv(const char *fname, 
		int f_make_heading, int verbose_level);
	void save_constant_size_csv(const char *fname,
			int verbose_level);
	int find_common_element_in_two_sets(int idx1, int idx2, 
		int &common_elt);
	void sort();
	void sort_big(int verbose_level);
	void compute_orbits(int &nb_orbits, int *&orbit, int *&orbit_inv, 
		int *&orbit_first, int *&orbit_len, 
		void (*compute_image_function)(set_of_sets *S, 
			void *compute_image_data, int elt_idx, int gen_idx, 
			int &idx_of_image, int verbose_level), 
		void *compute_image_data, 
		int nb_gens, 
		int verbose_level);
	int number_of_eckardt_points(int verbose_level);
	void get_eckardt_points(int *&E, int &nb_E, int verbose_level);
	void evaluate_function_and_store(set_of_sets *&Function_values,
			int (*evaluate_function)(int a, int i, int j, void *evaluate_data, int verbose_level),
			void *evaluate_data,
			int verbose_level);
};

int set_of_sets_compare_func(void *data, int i, int j, void *extra_data);
void set_of_sets_swap_func(void *data, int i, int j, void *extra_data);


// #############################################################################
// sorting.cpp
// #############################################################################

//! a collection of functions related to sorted vectors


class sorting {

public:
	sorting();
	~sorting();

	void int_vec_search_vec(int *v, int len, int *A, int A_sz, int *Idx);
	void lint_vec_search_vec(
			long int *v, int len, long int *A, int A_sz, long int *Idx);
	void int_vec_search_vec_linear(int *v, int len, int *A, int A_sz, int *Idx);
	void lint_vec_search_vec_linear(
			long int *v, int len, long int *A, int A_sz, long int *Idx);
	int int_vec_is_subset_of(int *set, int sz, int *big_set, int big_set_sz);
	int lint_vec_is_subset_of(int *set, int sz, long int *big_set, int big_set_sz);
	void int_vec_swap_points(int *list, int *list_inv, int idx1, int idx2);
	int int_vec_is_sorted(int *v, int len);
	void int_vec_sort_and_remove_duplicates(int *v, int &len);
	void lint_vec_sort_and_remove_duplicates(long int *v, int &len);
	int int_vec_sort_and_test_if_contained(int *v1, int len1, int *v2, int len2);
	int lint_vec_sort_and_test_if_contained(
			long int *v1, int len1, long int *v2, int len2);
	int int_vecs_are_disjoint(int *v1, int len1, int *v2, int len2);
	int int_vecs_find_common_element(int *v1, int len1,
		int *v2, int len2, int &idx1, int &idx2);
	int lint_vecs_find_common_element(long int *v1, int len1,
		long int *v2, int len2, int &idx1, int &idx2);
	void int_vec_insert_and_reallocate_if_necessary(int *&vec,
		int &used_length, int &alloc_length, int a, int verbose_level);
	void int_vec_append_and_reallocate_if_necessary(int *&vec,
		int &used_length, int &alloc_length, int a, int verbose_level);
	void lint_vec_append_and_reallocate_if_necessary(long int *&vec,
			int &used_length, int &alloc_length, long int a,
			int verbose_level);
	int int_vec_is_zero(int *v, int len);
	int test_if_sets_are_equal(int *set1, int *set2, int set_size);
	void test_if_set(int *set, int set_size);
	int test_if_set_with_return_value(int *set, int set_size);
	int test_if_set_with_return_value_lint(long int *set, int set_size);
	void rearrange_subset(int n, int k, int *set,
		int *subset, int *rearranged_set, int verbose_level);
	void rearrange_subset_lint(int n, int k,
		long int *set, int *subset, long int *rearranged_set,
		int verbose_level);
	void rearrange_subset_lint_all(int n, int k,
		long int *set, long int *subset, long int *rearranged_set,
		int verbose_level);
	int int_vec_search_linear(int *v, int len, int a, int &idx);
	int lint_vec_search_linear(long int *v, int len, long int a, int &idx);
	void int_vec_intersect(int *v1, int len1, int *v2, int len2,
		int *&v3, int &len3);
	void vec_intersect(long int *v1, int len1,
		long int *v2, int len2, long int *&v3, int &len3);
	void int_vec_intersect_sorted_vectors(int *v1, int len1,
		int *v2, int len2, int *v3, int &len3);
	void lint_vec_intersect_sorted_vectors(long int *v1, int len1,
			long int *v2, int len2, long int *v3, int &len3);
	void int_vec_sorting_permutation(int *v, int len, int *perm,
		int *perm_inv, int f_increasingly);
	// perm and perm_inv must be allocated to len elements
	void int_vec_quicksort(int *v, int (*compare_func)(int a, int b),
		int left, int right);
	void lint_vec_quicksort(long int *v,
		int (*compare_func)(long int a, long int b), int left, int right);
	void int_vec_quicksort_increasingly(int *v, int len);
	void int_vec_quicksort_decreasingly(int *v, int len);
	void lint_vec_quicksort_increasingly(long int *v, int len);
	void lint_vec_quicksort_decreasingly(long int *v, int len);
	void quicksort_array(int len, void **v,
		int (*compare_func)(void *a, void *b, void *data), void *data);
	void quicksort_array_with_perm(int len, void **v, int *perm,
		int (*compare_func)(void *a, void *b, void *data), void *data);
	//void int_vec_sort(int len, int *p);
	int vec_search(void **v, int (*compare_func)(void *a, void *b, void *data),
		void *data_for_compare,
		int len, void *a, int &idx, int verbose_level);
	int vec_search_general(void *vec,
		int (*compare_func)(void *vec, void *a, int b, void *data_for_compare),
		void *data_for_compare,
		int len, void *a, int &idx, int verbose_level);
	int int_vec_search_and_insert_if_necessary(int *v, int &len, int a);
	int int_vec_search_and_remove_if_found(int *v, int &len, int a);
	int int_vec_search(int *v, int len, int a, int &idx);
		// This function finds the last occurence of the element a.
		// If a is not found, it returns in idx the position
		// where it should be inserted if
		// the vector is assumed to be in increasing order.
	int lint_vec_search(long int *v, int len, long int a,
			int &idx, int verbose_level);
		// This function finds the last occurence of the element a.
		// If a is not found, it returns in idx the position
		// where it should be inserted if
		// the vector is assumed to be in increasing order.
	int int_vec_search_first_occurence(int *v, int len, int a, int &idx,
			int verbose_level);
		// This function finds the first occurence of the element a.
	int longinteger_vec_search(longinteger_object *v, int len,
		longinteger_object &a, int &idx);
	void int_vec_classify_and_print(std::ostream &ost, int *v, int l);
	void int_vec_values(int *v, int l, int *&w, int &w_len);
	void int_vec_multiplicities(int *v, int l, int *&w, int &w_len);
	void int_vec_values_and_multiplicities(int *v, int l,
		int *&val, int *&mult, int &nb_values);
	void int_vec_classify(int length, int *the_vec, int *&the_vec_sorted,
		int *&sorting_perm, int *&sorting_perm_inv,
		int &nb_types, int *&type_first, int *&type_len);
	void int_vec_classify_with_arrays(int length,
		int *the_vec, int *the_vec_sorted,
		int *sorting_perm, int *sorting_perm_inv,
		int &nb_types, int *type_first, int *type_len);
	void int_vec_sorted_collect_types(int length, int *the_vec_sorted,
		int &nb_types, int *type_first, int *type_len);
	void int_vec_print_classified(std::ostream &ost, int *vec, int len);
	void int_vec_print_types(std::ostream &ost,
		int f_backwards, int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void int_vec_print_types_naked_stringstream(std::stringstream &sstr,
		int f_backwards, int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void int_vec_print_types_naked(std::ostream &ost, int f_backwards,
		int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void int_vec_print_types_naked_tex(std::ostream &ost, int f_backwards,
		int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void int_vec_print_types_naked_tex_we_are_in_math_mode(std::ostream &ost,
		int f_backwards, int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void Heapsort(void *v, int len, int entry_size_in_chars,
		int (*compare_func)(void *v1, void *v2));
	void Heapsort_general(void *data, int len,
		int (*compare_func)(void *data, int i, int j, void *extra_data),
		void (*swap_func)(void *data, int i, int j, void *extra_data),
		void *extra_data);
	int search_general(void *data, int len, void *search_object, int &idx,
		int (*compare_func)(void *data, int i, void *search_object,
		void *extra_data),
		void *extra_data, int verbose_level);
		// This function finds the last occurence of the element a.
		// If a is not found, it returns in idx the position
		// where it should be inserted if
		// the vector is assumed to be in increasing order.
	void int_vec_heapsort(int *v, int len);
	void lint_vec_heapsort(long int *v, int len);
	void int_vec_heapsort_with_log(int *v, int *w, int len);
	void lint_vec_heapsort_with_log(long int *v, long int *w, int len);
	void heapsort_make_heap(int *v, int len);
	void lint_heapsort_make_heap(long int *v, int len);
	void heapsort_make_heap_with_log(int *v, int *w, int len);
	void lint_heapsort_make_heap_with_log(long int *v, long int *w, int len);
	void Heapsort_make_heap(void *v, int len, int entry_size_in_chars,
		int (*compare_func)(void *v1, void *v2));
	void Heapsort_general_make_heap(void *data, int len,
		int (*compare_func)(void *data, int i, int j, void *extra_data),
		void (*swap_func)(void *data, int i, int j, void *extra_data),
		void *extra_data);
	void heapsort_sift_down(int *v, int start, int end);
	void lint_heapsort_sift_down(long int *v, int start, int end);
	void heapsort_sift_down_with_log(int *v, int *w, int start, int end);
	void lint_heapsort_sift_down_with_log(
			long int *v, long int *w, int start, int end);
	void Heapsort_sift_down(void *v, int start, int end, int entry_size_in_chars,
		int (*compare_func)(void *v1, void *v2));
	void Heapsort_general_sift_down(void *data, int start, int end,
		int (*compare_func)(void *data, int i, int j, void *extra_data),
		void (*swap_func)(void *data, int i, int j, void *extra_data),
		void *extra_data);
	void heapsort_swap(int *v, int i, int j);
	void lint_heapsort_swap(long int *v, int i, int j);
	void Heapsort_swap(void *v, int i, int j, int entry_size_in_chars);
	void find_points_by_multiplicity(int *data, int data_sz, int multiplicity,
		int *&pts, int &nb_pts);
	void int_vec_bubblesort_increasing(int len, int *p);
	int integer_vec_compare(int *p, int *q, int len);
	void schreier_vector_compute_depth_and_ancestor(
		int n, int *pts, int *prev, int f_use_pts_inv, int *pts_inv,
		int *&depth, int *&ancestor, int verbose_level);
	int schreier_vector_determine_depth_recursion(
		int n, int *pts, int *prev, int f_use_pts_inv, int *pts_inv,
		int *depth, int *ancestor, int pos);
	void schreier_vector_tree(
		int n, int *pts, int *prev, int f_use_pts_inv, int *pts_inv,
		std::string &fname_base,
		layered_graph *&LG,
		int f_embedded, int f_sideways,
		int verbose_level);
	int compare_sets(int *set1, int *set2, int sz1, int sz2);
	int compare_sets_lint(long int *set1, long int *set2, int sz1, int sz2);
	int test_if_sets_are_disjoint(long int *set1, long int *set2, int sz1, int sz2);
	void d_partition(double *v, int left, int right, int *middle);
	void d_quicksort(double *v, int left, int right);
	void d_quicksort_array(int len, double *v);
};

int int_compare_increasingly(void *a, void *b, void *data);
int int_compare_decreasingly(void *a, void *b, void *data);
int compare_increasingly_int(int a, int b);
int compare_decreasingly_int(int a, int b);
int compare_increasingly_lint(long int a, long int b);
int compare_decreasingly_lint(long int a, long int b);
int int_vec_compare(int *p, int *q, int len);
//int int_vec_compare(int *p, int *q, int len);
int lint_vec_compare(long int *p, long int *q, int len);
int int_vec_compare_stride(int *p, int *q, int len, int stride);


// #############################################################################
// spreadsheet.cpp
// #############################################################################

//! for reading and writing of csv files


class spreadsheet {

public:

	char **tokens;
	int nb_tokens;

	int *line_start, *line_size;
	int nb_lines;

	int nb_rows, nb_cols;
	int *Table;


	spreadsheet();
	~spreadsheet();
	void null();
	void freeself();
	void init_set_of_sets(set_of_sets *S, int f_make_heading);
	void init_int_matrix(int nb_rows, int nb_cols, int *A);
	void init_empty_table(int nb_rows, int nb_cols);
	void fill_entry_with_text(int row_idx, 
		int col_idx, const char *text);
	void fill_column_with_text(int col_idx, const char **text, 
		const char *heading);
	void fill_column_with_int(int col_idx, int *data, 
		const char *heading);
	void fill_column_with_lint(int col_idx,
			long int *data, const char *heading);
	void fill_column_with_row_index(int col_idx, 
		const char *heading);
	void add_token(const char *label);
	void save(const char *fname, int verbose_level);
	void read_spreadsheet(const char *fname, int verbose_level);
	void print_table(std::ostream &ost, int f_enclose_in_parentheses);
	void print_table_latex_all_columns(std::ostream &ost,
		int f_enclose_in_parentheses);
	void print_table_latex(std::ostream &ost, int *f_column_select,
		int f_enclose_in_parentheses);
	void print_table_row(int row, int f_enclose_in_parentheses, 
			std::ostream &ost);
	void print_table_row_latex(int row, int *f_column_select, 
		int f_enclose_in_parentheses, std::ostream &ost);
	void print_table_row_detailed(int row, std::ostream &ost);
	void print_table_with_row_selection(int *f_selected, 
			std::ostream &ost);
	void print_table_sorted(std::ostream &ost, const char *sort_by);
	void add_column_with_constant_value(const char *label, char *value);
	void add_column_with_int(const char *label, int *Value);
	void add_column_with_text(const char *label, char **Value);
	void reallocate_table();
	void reallocate_table_add_row();
	int find_by_column(const char *join_by);
	void tokenize(const char *fname, 
		char **&tokens, int &nb_tokens, int verbose_level);
	void remove_quotes(int verbose_level);
	void remove_rows(const char *drop_column, const char *drop_label, 
		int verbose_level);
	void remove_rows_where_field_is_empty(const char *drop_column, 
		int verbose_level);
	void find_rows(int verbose_level);
	void get_value_double_or_NA(int i, int j, double &val, int &f_NA);
	char *get_string(int i, int j);
	int get_int(int i, int j);
	double get_double(int i, int j);
	void join_with(spreadsheet *S2, int by1, int by2, 
		int verbose_level);
	void patch_with(spreadsheet *S2, char *join_by);


};


// #############################################################################
// super_fast_hash.cpp
// #############################################################################

uint32_t SuperFastHash (const char * data, int len);


// #############################################################################
// vector_hashing.cpp
// #############################################################################

//! hash tables


class vector_hashing {

public:
	int data_size;
	int N;
	int bit_length;
	int *vector_data;
	int *H;
	int *H_sorted;
	int *perm;
	int *perm_inv;
	int nb_types;
	int *type_first;
	int *type_len;
	int *type_value;


	vector_hashing();
	~vector_hashing();
	void allocate(int data_size, int N, int bit_length);
	void compute_tables(int verbose_level);
	void print();
	int rank(int *data);
	void unrank(int rk, int *data);
};

}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_DATA_STRUCTURES_DATA_STRUCTURES_H_ */



