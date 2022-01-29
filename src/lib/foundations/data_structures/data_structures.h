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
namespace layer1_foundations {
namespace data_structures {


// #############################################################################
// algorithms.cpp
// #############################################################################



//! catch all class for algorithms


class algorithms {
public:

	algorithms();
	~algorithms();
	int hashing(int hash0, int a);
	int hashing_fixed_width(int hash0, int a, int bit_length);
	void uchar_print_bitwise(std::ostream &ost, uchar u);
	void uchar_move(uchar *p, uchar *q, int len);
	void int_swap(int& x, int& y);
	void print_pointer_hex(std::ostream &ost, void *p);
	void print_hex_digit(std::ostream &ost, int digit);
	void print_repeated_character(std::ostream &ost, char c, int n);
	uint32_t root_of_tree_uint32_t (uint32_t* S, uint32_t i);
	void solve_diophant(int *Inc,
		int nb_rows, int nb_cols, int nb_needed,
		int f_has_Rhs, int *Rhs,
		long int *&Solutions, int &nb_sol, long int &nb_backtrack, int &dt,
		int f_DLX,
		int verbose_level);
	// allocates Solutions[nb_sol * nb_needed]
	uint32_t SuperFastHash (const char * data, int len);

};

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
			field_theory::finite_field *F, long int start_value, int verbose_level);
	void rank_PG_elements_in_columns(
			field_theory::finite_field *F,
			int *perms, unsigned int *PG_ranks, int verbose_level);
	void print();
	void zero_out();
	int s_ij(int i, int j);
	void m_ij(int i, int j, int a);
	void mult_int_matrix_from_the_left(int *A, int Am, int An,
			bitmatrix *Out, int verbose_level);

};

// #############################################################################
// bitvector.cpp
// #############################################################################

//! compact storage of 0/1-data as bitvectors

class bitvector {

private:
	uchar *data; // [allocated_length]
	long int length; // number of bits used
	long int allocated_length;


public:

	bitvector();
	~bitvector();
	void allocate(long int length);
	long int get_length();
	long int get_allocated_length();
	uchar *get_data();
	void m_i(long int i, int a);
	void set_bit(long int i);
	int s_i(long int i);
	void save(std::ofstream &fp);
	void load(std::ifstream &fp);
	uint32_t compute_hash();
	void print();

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
		// the number of char we need to store the canonical form of
		// one object


	uchar **Type_data;
		// Type_data[nb_types][rep_len]
		// the canonical form of the i-th representative is
		// Type_data[i][rep_len]
	int *Type_rep;
		// Type_rep[nb_types]
		// Type_rep[i] is the index of the candidate which
		// has been chosen as representative
		// for the i-th isomorphism type
	int *Type_mult;
		// Type_mult[nb_types]
		// Type_mult[i] gives the number of candidates which
		// are isomorphic to the i-th isomorphism class representative
	void **Type_extra_data;
		// Type_extra_data[nb_types]
		// Type_extra_data[i] is a pointer that is stored with the
		// i-th isomorphism class representative

	int N;
		// number of candidates (or objects) that we will test
	int n;
		// number of candidates that we have already tested

	int *type_of;
		// type_of[nb_types]
		// type_of[i] is the isomorphism type of the i-th candidate

	tally *C_type_of;
		// the classification of type_of[nb_types]
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
	int compare_at(uchar *data, int idx);
	void add_at_idx(uchar *data,
			void *extra_data, int idx, int verbose_level);
	void finalize(int verbose_level);
	void print_reps();
	void print_table();
	void save(
			std::string &prefix,
		void (*encode_function)(void *extra_data,
			long int *&encoding, int &encoding_sz, void *global_data),
		void (*get_group_order_or_NULL)(void *extra_data,
				ring_theory::longinteger_object &go, void *global_data),
		void *global_data,
		int verbose_level);

};




// #############################################################################
// classify_using_canonical_forms.cpp
// #############################################################################

//! classification of objects using canonical forms

class classify_using_canonical_forms {
public:


	int nb_input_objects;


	std::vector<bitvector *> B;
	std::vector<void *> Objects;
	std::vector<long int> Ago;
	std::vector<int> input_index;

	std::multimap<uint32_t, int> Hashing;
		// we store the pair (hash, idx)
		// where hash is the hash value of the set and idx is the
		// index in the table Sets where the set is stored.
		//
		// we use a multimap because the hash values are not unique
		// it happens that two sets have the same hash value.
		// map cannot handle that.


	//std::vector<void *> Input_objects;
	//std::vector<int> orbit_rep_of_input_object;

	classify_using_canonical_forms();
	~classify_using_canonical_forms();
	void orderly_test(geometry::object_with_canonical_form *OwCF,
			int &f_accept, int verbose_level);
	void find_object(geometry::object_with_canonical_form *OwCF,
			int &f_found, int &idx,
			nauty_output *&NO,
			bitvector *&Canonical_form,
			int verbose_level);
		// if f_found is TRUE, B[idx] agrees with the given object
	void add_object(geometry::object_with_canonical_form *OwCF,
			int &f_new_object,
			int verbose_level);

};

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
// data_input_stream_description_element.cpp:
// #############################################################################


//! describes one element in an input stream of combinatorial objects


class data_input_stream_description_element {
public:
	enum data_input_stream_type input_type;
	std::string input_string;
	std::string input_string2;

	// for t_data_input_stream_file_of_designs:
	int input_data1; // N_points
	int input_data2; // b = number of blocks
	int input_data3; // k = block size
	int input_data4; // partition class size

	data_input_stream_description_element();
	~data_input_stream_description_element();
	void print();
	void init_set_of_points(std::string &a);
	void init_set_of_lines(std::string &a);
	void init_set_of_points_and_lines(std::string &a, std::string &b);
	void init_packing(std::string &a, int q);
	void init_file_of_points(std::string &a);
	void init_file_of_lines(std::string &a);
	void init_file_of_packings(std::string &a);
	void init_file_of_packings_through_spread_table(
			std::string &a, std::string &b, int q);
	void init_file_of_point_set(std::string &a);
	void init_file_of_designs(std::string &a,
				int N_points, int b, int k, int partition_class_size);
	void init_file_of_incidence_geometries(std::string &a,
				int v, int b, int f);
	void init_file_of_incidence_geometries_by_row_ranks(
			std::string &a,
				int v, int b, int r);
	void init_incidence_geometry(std::string &a,
				int v, int b, int f);
	void init_incidence_geometry_by_row_ranks(std::string &a,
				int v, int b, int r);
	void init_from_parallel_search(std::string &fname_mask,
			int nb_cases, std::string &cases_fname);

};


// #############################################################################
// data_input_stream_description.cpp:
// #############################################################################


//! description of input data for classification of geometric objects from the command line


class data_input_stream_description {
public:


	int nb_inputs;

	std::vector<data_input_stream_description_element> Input;

	data_input_stream_description();
	~data_input_stream_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();
	void print_item(int i);


};


// #############################################################################
// data_input_stream.cpp:
// #############################################################################


//! input data for classification of geometric objects from the command line


class data_input_stream {
public:

	data_input_stream_description *Descr;

	int nb_objects_to_test;

	std::vector<void *> Objects;

	data_input_stream();
	~data_input_stream();
	void init(data_input_stream_description *Descr, int verbose_level);
	int count_number_of_objects_to_test(
		int verbose_level);
	void read_objects(int verbose_level);


};


// #############################################################################
// data_structures_global.cpp:
// #############################################################################


//! a catch-all container class for everything related to data structures


class data_structures_global {
public:
	data_structures_global();
	~data_structures_global();
	void bitvector_m_ii(uchar *bitvec, long int i, int a);
	void bitvector_set_bit(uchar *bitvec, long int i);
	int bitvector_s_i(uchar *bitvec, long int i);
	uint32_t int_vec_hash(int *data, int len);
	uint32_t lint_vec_hash(long int *data, int len);
	uint32_t char_vec_hash(char *data, int len);
	int int_vec_hash_after_sorting(int *data, int len);
	int lint_vec_hash_after_sorting(long int *data, int len);

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
	void save(std::string &fname, int verbose_level);

};

// #############################################################################
// int_matrix.cpp:
// #############################################################################


//! matrices over int


class int_matrix {
public:

	int *M;
	int m;
	int n;

	int_matrix();
	~int_matrix();
	void null();
	void freeself();
	void allocate(int m, int n);
	void allocate_and_init(int m, int n, int *Mtx);
	int &s_ij(int i, int j);
	int &s_m();
	int &s_n();
	void print();

};


// #############################################################################
// int_vec.cpp:
// #############################################################################


//! int arrays

class int_vec {
public:

	int_vec();
	~int_vec();
	void add(int *v1, int *v2, int *w, int len);
	void add3(int *v1, int *v2, int *v3, int *w, int len);
	void apply(int *from, int *through, int *to, int len);
	void apply_lint(int *from, long int *through, long int *to, int len);
	int is_constant_on_subset(int *v,
		int *subset, int sz, int &value);
	void take_away(int *v, int &len,
			int *take_away, int nb_take_away);
		// v must be sorted
	int count_number_of_nonzero_entries(int *v, int len);
	int find_first_nonzero_entry(int *v, int len);
	void zero(int *v, long int len);
	int is_zero(int *v, long int len);
	void mone(int *v, long int len);
	void copy(int *from, int *to, long int len);
	void copy_to_lint(int *from, long int *to, long int len);
	void swap(int *v1, int *v2, long int len);
	void delete_element_assume_sorted(int *v,
		int &len, int a);
	void complement(int *v, int n, int k);
	// computes the complement to v + k (v must be allocated to n elements)
	// the first k elements of v[] must be in increasing order.
	void complement(int *v, int *w, int n, int k);
	// computes the complement of v[k] in the set {0,...,n-1} to w[n - k]
	void init5(int *v, int a0, int a1, int a2, int a3, int a4);
	int minimum(int *v, int len);
	int maximum(int *v, int len);
	void copy(int len, int *from, int *to);
	int first_difference(int *p, int *q, int len);
	int vec_max_log_of_entries(std::vector<std::vector<int> > &p);
	void vec_print(std::vector<std::vector<int> > &p);
	void vec_print(std::vector<std::vector<int> > &p, int w);
	void distribution_compute_and_print(std::ostream &ost,
		int *v, int v_len);
	void distribution(int *v,
		int len_v, int *&val, int *&mult, int &len);
	void print(std::ostream &ost, std::vector<int> &v);
	void print(std::ostream &ost, int *v, int len);
	void print_str(std::stringstream &ost, int *v, int len);
	void print_str_naked(std::stringstream &ost, int *v, int len);
	void print_as_table(std::ostream &ost, int *v, int len, int width);
	void print_fully(std::ostream &ost, std::vector<int> &v);
	void print_fully(std::ostream &ost, int *v, int len);
	void print_dense(std::ostream &ost, int *v, int len);
	void print_Cpp(std::ostream &ost, int *v, int len);
	void print_GAP(std::ostream &ost, int *v, int len);
	void print_classified(int *v, int len);
	void print_classified_str(std::stringstream &sstr,
			int *v, int len, int f_backwards);
	void scan(std::string &s, int *&v, int &len);
	void scan(const char *s, int *&v, int &len);
	void scan_from_stream(std::istream & is, int *&v, int &len);
	void print_to_str(char *str, int *data, int len);
	void print_to_str_naked(char *str, int *data, int len);
	void print(int *v, int len);
	void print_integer_matrix(std::ostream &ost,
		int *p, int m, int n);
	void print_integer_matrix_width(std::ostream &ost,
		int *p, int m, int n, int dim_n, int w);
	void print_integer_matrix_in_C_source(std::ostream &ost,
		int *p, int m, int n);
	void matrix_make_block_matrix_2x2(int *Mtx,
		int k, int *A, int *B, int *C, int *D);
	void matrix_delete_column_in_place(int *Mtx,
		int k, int n, int pivot);
	int matrix_max_log_of_entries(int *p, int m, int n);
	void matrix_print_ost(std::ostream &ost, int *p, int m, int n);
	void matrix_print(int *p, int m, int n);
	void matrix_print_tight(int *p, int m, int n);
	void matrix_print_ost(std::ostream &ost, int *p, int m, int n, int w);
	void matrix_print(int *p, int m, int n, int w);
	void matrix_print_bitwise(int *p, int m, int n);
	void distribution_print(std::ostream &ost,
		int *val, int *mult, int len);
	void set_print(std::ostream &ost, int *v, int len);
	void integer_vec_print(std::ostream &ost, int *v, int len);
	int hash(int *v, int len, int bit_length);
	void create_string_with_quotes(std::string &str, int *v, int len);
	void transpose(int *M, int m, int n, int *Mt);

};



// #############################################################################
// int_vector.cpp
// #############################################################################

//! vector of int

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
	void read_ascii_file(std::string &fname);
	void read_binary_file_int4(std::string &fname);
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
	void write_to_ascii_file(std::string &fname);
	void write_to_binary_file_int4(std::string &fname);
	void write_to_csv_file(std::string &fname, const char *label);
	uint32_t hash();
	int minimum();
	int maximum();



};


// #############################################################################
// lint_vec.cpp:
// #############################################################################


//! long int arrays

class lint_vec {
public:

	lint_vec();
	~lint_vec();
	void apply(long int *from, long int *through, long int *to, int len);
	void take_away(long int *v, int &len,
			long int *take_away, int nb_take_away);
	void zero(long int *v, long int len);
	void mone(long int *v, long int len);
	void copy(long int *from, long int *to, long int len);
	void copy_to_int(long int *from, int *to, long int len);
	void complement(long int *v, long int *w, int n, int k);
	long int minimum(long int *v, int len);
	long int maximum(long int *v, int len);
	void matrix_print_width(std::ostream &ost,
		long int *p, int m, int n, int dim_n, int w);
	void set_print(long int *v, int len);
	void set_print(std::ostream &ost, long int *v, int len);
	void print(std::ostream &ost, long int *v, int len);
	void print(std::ostream &ost, std::vector<long int> &v);
	void print_as_table(std::ostream &ost, long int *v, int len, int width);
	void print_bare_fully(std::ostream &ost, long int *v, int len);
	void print_fully(std::ostream &ost, long int *v, int len);
	void print_fully(std::ostream &ost, std::vector<long int> &v);
	int matrix_max_log_of_entries(long int *p, int m, int n);
	void matrix_print(long int *p, int m, int n);
	void matrix_print(long int *p, int m, int n, int w);
	void scan(std::string &s, long int *&v, int &len);
	void scan(const char *s, long int *&v, int &len);
	void scan_from_stream(std::istream & is, long int *&v, int &len);
	void print_to_str(char *str, long int *data, int len);
	void print_to_str_naked(char *str, long int *data, int len);
	void create_string_with_quotes(std::string &str, long int *v, int len);


};



// #############################################################################
// nauty_output.cpp:
// #############################################################################


//! output data created by a run of nauty

class nauty_output {
public:


	int N;
	int *Aut;
	int Aut_counter;
	int *Base;
	int Base_length;
	long int *Base_lint;
	int *Transversal_length;
	ring_theory::longinteger_object *Ago;

	int *canonical_labeling; // [N]

	long int nb_firstpathnode;
	long int nb_othernode;
	long int nb_processnode;
	long int nb_firstterminal;

	nauty_output();
	~nauty_output();
	void allocate(int N, int verbose_level);
	void print();
	void print_stats();
	int belong_to_the_same_orbit(int a, int b, int verbose_level);

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



// #############################################################################
// partitionstack.cpp
// #############################################################################



//! data structure for set partitions following Jeffrey Leon


class partitionstack {
	public:

	// data structure for the partition stack,
	// following Leon:
		int n; // size of the set that is partitioned
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
	void free();
	void allocate(int n, int verbose_level);
	void allocate_with_two_classes(int n, int v, int b, int verbose_level);
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
	void get_cell(int i, int *&cell, int &cell_sz, int verbose_level);
	void get_cell_lint(int i, long int *&cell, int &cell_sz, int verbose_level);
	void get_row_classes(set_of_sets *&Sos, int verbose_level);
	void get_column_classes(set_of_sets *&Sos, int verbose_level);
	void write_cell_to_file(int i,
			std::string &fname, int verbose_level);
	void write_cell_to_file_points_or_lines(int i, 
			std::string &fname, int verbose_level);
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
// set_builder_description.cpp
// #############################################################################



//! to define a set of integers for class set_builder


class set_builder_description {
public:

	int f_index_set_loop;
	int index_set_loop_low;
	int index_set_loop_upper_bound;
	int index_set_loop_increment;

	int f_affine_function;
	int affine_function_a;
	int affine_function_b;

	int f_clone_with_affine_function;
	int clone_with_affine_function_a;
	int clone_with_affine_function_b;

	int f_set_builder;
	set_builder_description *Descr;

	int f_here;
	std::string here_text;

	int f_file;
	std::string file_name;

	set_builder_description();
	~set_builder_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();
};



// #############################################################################
// set_builder.cpp
// #############################################################################



//! to create a set of integers from class set_builder_description


class set_builder {
public:

	set_builder_description *Descr;

	long int *set;
	int sz;

	set_builder();
	~set_builder();
	void init(set_builder_description *Descr, int verbose_level);
	long int process_transformations(long int x);
	long int clone_with_affine_function(long int x);
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
	void init_from_file(int &underlying_set_size,
			std::string &fname, int verbose_level);
	void init_from_csv_file(int underlying_set_size, 
			std::string &fname, int verbose_level);
	void init_from_orbiter_file(int underlying_set_size, 
			std::string &fname, int verbose_level);
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
	void print_table_latex_simple(std::ostream &ost);
	void print_table_latex_simple_with_selection(std::ostream &ost, int *Selection, int nb_sel);
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
	void compute_and_print_tdo_row_scheme(std::ostream &file,
		int verbose_level);
	void compute_and_print_tdo_col_scheme(std::ostream &file,
		int verbose_level);
	void init_decomposition(geometry::decomposition *&D, int verbose_level);
	void compute_tdo_decomposition(geometry::decomposition &D,
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
	void save_csv(std::string &fname,
		int f_make_heading, int verbose_level);
	void save_constant_size_csv(std::string &fname,
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
	void evaluate_function_and_store(data_structures::set_of_sets *&Function_values,
			int (*evaluate_function)(int a, int i, int j, void *evaluate_data, int verbose_level),
			void *evaluate_data,
			int verbose_level);
	int find_smallest_class();
};



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
	int lint_vec_is_subset_of(int *set, int sz, long int *big_set, int big_set_sz, int verbose_level);
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
	int test_if_sets_are_disjoint(long int *set1, int sz1, long int *set2, int sz2);
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
		// This function finds the last occurrence of the element a.
		// If a is not found, it returns in idx the position
		// where it should be inserted if
		// the vector is assumed to be in increasing order.
	int lint_vec_search(long int *v, int len, long int a,
			int &idx, int verbose_level);
		// This function finds the last occurrence of the element a.
		// If a is not found, it returns in idx the position
		// where it should be inserted if
		// the vector is assumed to be in increasing order.
	int vector_lint_search(std::vector<long int> &v,
			long int a, int &idx, int verbose_level);
	int int_vec_search_first_occurence(int *v, int len, int a, int &idx,
			int verbose_level);
		// This function finds the first occurrence of the element a.
	int longinteger_vec_search(ring_theory::longinteger_object *v, int len,
			ring_theory::longinteger_object &a, int &idx);
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
	void Heapsort_general_with_log(void *data, int *w, int len,
		int (*compare_func)(void *data,
				int i, int j, void *extra_data),
		void (*swap_func)(void *data,
				int i, int j, void *extra_data),
		void *extra_data);
	int search_general(void *data, int len, void *search_object, int &idx,
		int (*compare_func)(void *data, int i, void *search_object,
		void *extra_data),
		void *extra_data, int verbose_level);
		// This function finds the last occurrence of the element a.
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
	void Heapsort_general_make_heap_with_log(void *data, int *w, int len,
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
	void Heapsort_general_sift_down_with_log(void *data, int *w, int start, int end,
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
	int lint_vec_compare(long int *p, long int *q, int len);
	void schreier_vector_compute_depth_and_ancestor(
		int n, int *pts, int *prev, int f_use_pts_inv, int *pts_inv,
		int *&depth, int *&ancestor, int verbose_level);
	int schreier_vector_determine_depth_recursion(
		int n, int *pts, int *prev, int f_use_pts_inv, int *pts_inv,
		int *depth, int *ancestor, int pos);
	void schreier_vector_tree(
		int n, int *pts, int *prev, int f_use_pts_inv, int *pts_inv,
		std::string &fname_base,
		graphics::layered_graph_draw_options *LG_Draw_options,
		graph_theory::layered_graph *&LG,
		int verbose_level);
	int compare_sets(int *set1, int *set2, int sz1, int sz2);
	int compare_sets_lint(long int *set1, long int *set2, int sz1, int sz2);
	int test_if_sets_are_disjoint(long int *set1, long int *set2, int sz1, int sz2);
	void d_partition(double *v, int left, int right, int *middle);
	void d_quicksort(double *v, int left, int right);
	void d_quicksort_array(int len, double *v);
	int test_if_sets_are_disjoint_assuming_sorted(int *set1, int *set2, int sz1, int sz2);
	int test_if_sets_are_disjoint_assuming_sorted_lint(long int *set1, long int *set2, int sz1, int sz2);
	int uchar_vec_compare(uchar *p, uchar *q, int len);
	int test_if_sets_are_disjoint_not_assuming_sorted(long int *v, long int *w, int len);
	int int_vec_compare(int *p, int *q, int len);
	int int_vec_compare_stride(int *p, int *q, int len, int stride);
	void sorted_vec_get_first_and_length(int *v, int len,
			int *class_first, int *class_len, int &nb_classes);
	// we assume that the vector v is sorted.

};


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
	void fill_entry_with_text(int row_idx,
			int col_idx, std::string &text);
	void set_entry_lint(int row_idx,
			int col_idx, long int val);
	void fill_column_with_text(int col_idx, const char **text, 
		const char *heading);
	void fill_column_with_int(int col_idx, int *data, 
		const char *heading);
	void fill_column_with_lint(int col_idx,
			long int *data, const char *heading);
	void fill_column_with_row_index(int col_idx, 
		const char *heading);
	void add_token(const char *label);
	void save(std::string &fname, int verbose_level);
	void read_spreadsheet(std::string &fname, int verbose_level);
	void print_table(std::ostream &ost, int f_enclose_in_parentheses);
	void print_table_latex_all_columns(std::ostream &ost,
		int f_enclose_in_parentheses);
	void print_table_latex(std::ostream &ost,
			int *f_column_select,
			int f_enclose_in_parentheses,
			int nb_lines_per_table);
	void print_table_row(int row, int f_enclose_in_parentheses, 
			std::ostream &ost);
	void print_table_row_latex(int row, int *f_column_select, 
		int f_enclose_in_parentheses, std::ostream &ost);
	void print_table_row_detailed(int row, std::ostream &ost);
	void print_table_row_with_column_selection(int row,
			int f_enclose_in_parentheses,
			int *Col_selection, int nb_cols_selected, std::ostream &ost);
	void print_table_with_row_selection(int *f_selected, 
			std::ostream &ost);
	void print_table_sorted(std::ostream &ost, const char *sort_by);
	void add_column_with_constant_value(const char *label, char *value);
	void add_column_with_int(const char *label, int *Value);
	void add_column_with_text(const char *label, char **Value);
	void reallocate_table();
	void reallocate_table_add_row();
	int find_column(std::string &column_label);
	int find_by_column(const char *join_by);
	void tokenize(std::string &fname,
		char **&tokens, int &nb_tokens, int verbose_level);
	void remove_quotes(int verbose_level);
	void remove_rows(const char *drop_column, const char *drop_label, 
		int verbose_level);
	void remove_rows_where_field_is_empty(const char *drop_column, 
		int verbose_level);
	void find_rows(int verbose_level);
	void get_value_double_or_NA(int i, int j, double &val, int &f_NA);
	//void get_string_entry(std::string &entry, int i, int j);
	void get_string(std::string &str, int i, int j);
	long int get_int(int i, int j);
	double get_double(int i, int j);
	void join_with(spreadsheet *S2, int by1, int by2, 
		int verbose_level);
	void patch_with(spreadsheet *S2, char *join_by);


};


// #############################################################################
// string_tools.cpp
// #############################################################################

//! functions related to strings and character arrays


class string_tools {

public:

	string_tools();
	~string_tools();
	int is_csv_file(const char *fname);
	int is_inc_file(const char *fname);
	int is_xml_file(const char *fname);
	int s_scan_int(char **s, int *i);
	int s_scan_lint(char **s, long int *i);
	int s_scan_double(char **s, double *d);
	int s_scan_token(char **s, char *str);
	int s_scan_token_arbitrary(char **s, char *str);
	int s_scan_str(char **s, char *str);
	int s_scan_token_comma_separated(const char **s, char *str);
	void scan_permutation_from_string(const char *s,
		int *&perm, int &degree, int verbose_level);
	void scan_permutation_from_stream(std::istream & is,
		int *&perm, int &degree, int verbose_level);
	void chop_string(const char *str, int &argc, char **&argv);
	void chop_string_comma_separated(const char *str, int &argc, char **&argv);
	void convert_arguments(int &argc, const char **argv, std::string *&Argv);
	char get_character(std::istream & is, int verbose_level);
	void replace_extension_with(char *p, const char *new_ext);
	void replace_extension_with(std::string &p, const char *new_ext);
	void chop_off_extension(char *p);
	void chop_off_extension_and_path(std::string &p);
	void chop_off_extension(std::string &p);
	void chop_off_path(std::string &p);
	void chop_off_extension_if_present(std::string &p, const char *ext);
	void chop_off_extension_if_present(char *p, const char *ext);
	void get_fname_base(const char *p, char *fname_base);
	void get_extension(std::string &p, std::string &ext);
	void get_extension_if_present(const char *p, char *ext);
	void get_extension_if_present_and_chop_off(char *p, char *ext);
	void string_fix_escape_characters(std::string &str);
	void remove_specific_character(std::string &str, char c);
	void create_comma_separated_list(std::string &output, long int *input, int input_sz);
	int is_all_whitespace(const char *str);
	void text_to_three_double(std::string &text, double *d);
	int strcmp_with_or_without(char *p, char *q);
	int starts_with_a_number(std::string &str);
	int stringcmp(std::string &str, const char *p);
	int strtoi(std::string &str);
	int str2int(std::string &str);
	long int strtolint(std::string &str);
	double strtof(std::string &str);


};

int string_tools_compare_strings(void *a, void *b, void *data);


// #############################################################################
// tally.cpp
// #############################################################################


//! a statistical analysis of data consisting of single integers



class tally {

public:

	int data_length;

	int f_data_ownership;
	int *data;
	int *data_sorted;
	int *sorting_perm;
		// computed using int_vec_sorting_permutation
	int *sorting_perm_inv;
		// perm_inv[i] is the index in data
		// of the element in data_sorted[i]
	int nb_types;
	int *type_first;
	int *type_len;

	int f_second;
	int *second_data_sorted;
	int *second_sorting_perm;
	int *second_sorting_perm_inv;
	int second_nb_types;
	int *second_type_first;
	int *second_type_len;

	tally();
	~tally();
	void init(int *data, int data_length,
		int f_second, int verbose_level);
	void init_lint(long int *data, int data_length,
		int f_second, int verbose_level);
	void sort_and_classify();
	void sort_and_classify_second();
	int class_of(int pt_idx);
	void print(int f_backwards);
	void print_no_lf(int f_backwards);
	void print_tex_no_lf(int f_backwards);
	void print_first(int f_backwards);
	void print_second(int f_backwards);
	void print_first_tex(int f_backwards);
	void print_second_tex(int f_backwards);
	void print_file(std::ostream &ost, int f_backwards);
	void print_file_tex(std::ostream &ost, int f_backwards);
	void print_file_tex_we_are_in_math_mode(std::ostream &ost, int f_backwards);
	void print_naked_stringstream(std::stringstream &sstr, int f_backwards);
	void print_naked(int f_backwards);
	void print_naked_tex(std::ostream &ost, int f_backwards);
	void print_types_naked_tex(std::ostream &ost, int f_backwards,
		int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void print_array_tex(std::ostream &ost, int f_backwards);
	double average();
	double average_of_non_zero_values();
	void get_data_by_multiplicity(int *&Pts, int &nb_pts,
		int multiplicity, int verbose_level);
	void get_data_by_multiplicity_as_lint(
			long int *&Pts, int &nb_pts, int multiplicity, int verbose_level);
	int determine_class_by_value(int value);
	int get_value_of_class(int class_idx);
	int get_largest_value();
	void get_class_by_value(int *&Pts, int &nb_pts, int value,
		int verbose_level);
	void get_class_by_value_lint(
			long int *&Pts, int &nb_pts, int value, int verbose_level);
	data_structures::set_of_sets *get_set_partition_and_types(int *&types,
		int &nb_types, int verbose_level);
	void save_classes_individually(std::string &fname);
};


// #############################################################################
// tally_vector_data.cpp
// #############################################################################


//! a statistical analysis of data consisting of vectors of ints



class tally_vector_data {

public:

	int data_set_sz;
	int data_length;

	int *data; // [data_length * data_set_sz]

	int *rep_idx; // [data_length], rep_idx[i] is the index into Rep of data[i * data_set_sz]
	int *Reps; // [data_length * data_set_sz], used [nb_types * data_set_sz]
	int *Frequency; // [data_length], used [nb_types]
	int *sorting_perm;
		// computed using int_vec_sorting_permutation
	int *sorting_perm_inv;
		// perm_inv[i] is the index in data
		// of the element in data_sorted[i]


	std::multimap<uint32_t, int> Hashing;
		// we store the pair (hash, idx)
		// where hash is the hash value of the set and idx is the
		// index in the table Sets where the set is stored.
		//
		// we use a multimap because the hash values are not unique
		// it happens that two sets have the same hash value.
		// map cannot handle that.

	int nb_types;
	int *type_first;
	//int *type_len; same as Frequency[]

	int **Reps_in_lex_order; // [nb_types]
	int *Frequency_in_lex_order; // [nb_types]


	tally_vector_data();
	~tally_vector_data();
	void init(int *data, int data_length, int data_set_sz,
		int verbose_level);
	int hash_and_find(int *data,
			int &idx, uint32_t &h, int verbose_level);
	void print();
	void save_classes_individually(std::string &fname, int verbose_level);
	void get_transversal(
			int *&transversal, int *&frequency, int &nb_types, int verbose_level);
	void print_classes_bigger_than_one(int verbose_level);

};




// #############################################################################
// vector_builder_description.cpp
// #############################################################################



//! to define a vector of field elements


class vector_builder_description {
public:

	int f_field;
	std::string field_label;

	int f_dense;
	std::string dense_text;

	int f_compact;
	std::string compact_text;

	int f_repeat;
	std::string repeat_text;
	int repeat_length;

	int f_format;
	int format_k;

	int f_file;
	std::string file_name;

	int f_sparse;
	int sparse_len;
	std::string sparse_pairs;

	int f_concatenate;
	std::vector<std::string> concatenate_list;

	int f_loop;
	int loop_start;
	int loop_upper_bound;
	int loop_increment;

	vector_builder_description();
	~vector_builder_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();
};



// #############################################################################
// vector_builder.cpp
// #############################################################################



//! to create a vector of field elements from class vector_builder_description


class vector_builder {
public:

	vector_builder_description *Descr;

	field_theory::finite_field *F;

	int *v;
	int len;

	int f_has_k;
	int k;

	vector_builder();
	~vector_builder();
	void init(vector_builder_description *Descr, field_theory::finite_field *F, int verbose_level);
};



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

}}}



#endif /* ORBITER_SRC_LIB_FOUNDATIONS_DATA_STRUCTURES_DATA_STRUCTURES_H_ */



