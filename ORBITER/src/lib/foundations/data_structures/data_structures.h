// data_structures.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


// #############################################################################
// INT_vector.C:
// #############################################################################

//! vector on INTs

class INT_vector {
public:

	INT *M;
	INT m;
	INT alloc_length;

	INT_vector();
	~INT_vector();
	void null();
	void freeself();
	void allocate(INT len);
	void allocate_and_init(INT len, INT *V);
	void init_permutation_from_string(const char *s);
	void read_ascii_file(const BYTE *fname);
	void read_binary_file_INT4(const BYTE *fname);
	INT &s_i(INT i);
	INT &length();
	void print(ostream &ost);
	void zero();
	INT search(INT a, INT &idx);
	void sort();
	void make_space();
	void append(INT a);
	void insert_at(INT a, INT idx);
	void insert_if_not_yet_there(INT a);
	void sort_and_remove_duplicates();
	void write_to_ascii_file(const BYTE *fname);
	void write_to_binary_file_INT4(const BYTE *fname);
	void write_to_csv_file(const BYTE *fname, const BYTE *label);
	INT hash();
	INT minimum();
	INT maximum();

	

};

// #############################################################################
// data_file.C:
// #############################################################################

//! to read files of classifications from the poset classification algorithm


class data_file {
	
	public:

	BYTE fname[1000];
	INT nb_cases;
	INT *set_sizes;
	INT **sets;
	INT *casenumbers;
	BYTE **Ago_ascii;
	BYTE **Aut_ascii;

	INT f_has_candidates;
	INT *nb_candidates;
	INT **candidates;

	data_file();
	~data_file();
	void null();
	void freeself();
	void read(const BYTE *fname, INT f_casenumbers, INT verbose_level);
	void read_candidates(const BYTE *candidates_fname, INT verbose_level);
};

// #############################################################################
// fancy_set.C:
// #############################################################################

//! to store a subset of size k of a set of size n


class fancy_set {
	
	public:

	INT n;
	INT k;
	INT *set;
	INT *set_inv;

	fancy_set();
	~fancy_set();
	void null();
	void freeself();
	void init(INT n, INT verbose_level);
	void init_with_set(INT n, INT k, INT *subset, INT verbose_level);
	void print();
	void println();
	void swap(INT pos, INT a);
	INT is_contained(INT a);
	void copy_to(fancy_set *to);
	void add_element(INT elt);
	void add_elements(INT *elts, INT nb);
	void delete_elements(INT *elts, INT nb);
	void delete_element(INT elt);
	void select_subset(INT *elts, INT nb);
	void intersect_with(INT *elts, INT nb);
	void subtract_set(fancy_set *set_to_subtract);
	void sort();
	INT compare_lexicographically(fancy_set *second_set);
	void complement(fancy_set *compl_set);
	INT is_subset(fancy_set *set2);
	INT is_equal(fancy_set *set2);

};


// #############################################################################
// partitionstack.C
// #############################################################################


ostream& operator<<(ostream& ost, partitionstack& p);


//! Leon type partitionstack class


class partitionstack {
	public:

	// data structure for the partition stack,
	// following Leon:
		INT n;
		INT ht;
		INT ht0;

		INT *pointList, *invPointList;
		INT *cellNumber;

		INT *startCell;
		INT *cellSize;
		INT *parent;


	// for matrix canonization:
	// INT first_column_element;

	// subset to be chosen by classify_by_..._extract_subset():
	// used as input for split_cell()
		//
	// used if SPLIT_MULTIPLY is defined:
		INT nb_subsets;
		INT *subset_first;
		INT *subset_length;
		INT *subsets;
		//
	// used if SPLIT_MULTIPLY is not defined:
		INT *subset;
		INT subset_size;

	partitionstack();
	~partitionstack();
	void allocate(INT n, INT verbose_level);
	void free();
	INT parent_at_height(INT h, INT cell);
	INT is_discrete();
	INT smallest_non_discrete_cell();
	INT biggest_non_discrete_cell();
	INT smallest_non_discrete_cell_rows_preferred();
	INT biggest_non_discrete_cell_rows_preferred();
	INT nb_partition_classes(INT from, INT len);
	INT is_subset_of_cell(INT *set, INT size, INT &cell_idx);
	void sort_cells();
	void sort_cell(INT cell);
	void reverse_cell(INT cell);
	void check();
	void print_raw();
	void print_class(ostream& ost, INT idx);
	void print_classes_tex(ostream& ost);
	void print_class_tex(ostream& ost, INT idx);
	void print_class_point_or_line(ostream& ost, INT idx);
	void print_classes(ostream& ost);
	void print_classes_points_and_lines(ostream& ost);
	ostream& print(ostream& ost);
	void print_cell(INT i);
	void print_cell_latex(ostream &ost, INT i);
	void print_subset();
	void write_cell_to_file(INT i, BYTE *fname, 
		INT verbose_level);
	void write_cell_to_file_points_or_lines(INT i, 
		BYTE *fname, INT verbose_level);
	void refine_arbitrary_set(INT size, INT *set, 
		INT verbose_level);
	void split_cell(INT verbose_level);
	void split_multiple_cells(INT *set, INT set_size, 
		INT f_front, INT verbose_level);
	void split_line_cell_front_or_back(INT *set, INT set_size, 
		INT f_front, INT verbose_level);
	void split_cell_front_or_back(INT *set, INT set_size, 
		INT f_front, INT verbose_level);
	void split_cell(INT *set, INT set_size, INT verbose_level);
	void join_cell();
	void reduce_height(INT ht0);
	void isolate_point(INT pt);
	void subset_continguous(INT from, INT len);
	INT is_row_class(INT c);
	INT is_col_class(INT c);
	void allocate_and_get_decomposition(
		INT *&row_classes, INT *&row_class_inv, 
			INT &nb_row_classes,
		INT *&col_classes, INT *&col_class_inv, 
			INT &nb_col_classes, 
		INT verbose_level);
	void get_row_and_col_permutation(
		INT *row_classes, INT nb_row_classes,
		INT *col_classes, INT nb_col_classes, 
		INT *row_perm, INT *row_perm_inv, 
		INT *col_perm, INT *col_perm_inv);
	void get_row_and_col_classes(INT *row_classes, 
		INT &nb_row_classes,
		INT *col_classes, INT &nb_col_classes, 
		INT verbose_level);
	void initial_matrix_decomposition(INT nbrows, 
		INT nbcols,
		INT *V, INT nb_V, INT *B, INT nb_B, 
		INT verbose_level);
	INT is_descendant_of(INT cell, INT ancestor_cell, 
		INT verbose_level);
	INT is_descendant_of_at_level(INT cell, INT ancestor_cell, 
	INT level, INT verbose_level);
	INT cellSizeAtLevel(INT cell, INT level);

	// TDO for orthogonal:
	INT compute_TDO(orthogonal &O, INT ht0, 
		INT marker1, INT marker2, INT depth, INT verbose_level);
	void get_and_print_row_decomposition_scheme(orthogonal &O, 
		INT marker1, INT marker2);
	void get_and_print_col_decomposition_scheme(orthogonal &O, 
		INT marker1, INT marker2);
	void get_and_print_decomposition_schemes(orthogonal &O, 
		INT marker1, INT marker2);
	void print_decomposition_tex(ostream &ost, 
		INT *row_classes, INT nb_row_classes,
		INT *col_classes, INT nb_col_classes);
	void print_decomposition_scheme(ostream &ost, 
		INT *row_classes, INT nb_row_classes,
		INT *col_classes, INT nb_col_classes, 
		INT *scheme, INT marker1, INT marker2);
	void print_decomposition_scheme_tex(ostream &ost, 
		INT *row_classes, INT nb_row_classes,
		INT *col_classes, INT nb_col_classes, 
		INT *scheme);
	void print_tactical_decomposition_scheme_tex_internal(
		ostream &ost, INT f_enter_math_mode, 
		INT *row_classes, INT nb_row_classes,
		INT *col_classes, INT nb_col_classes, 
		INT *row_scheme, INT *col_scheme, INT f_print_subscripts);
	void print_tactical_decomposition_scheme_tex(ostream &ost, 
		INT *row_classes, INT nb_row_classes,
		INT *col_classes, INT nb_col_classes, 
		INT *row_scheme, INT *col_scheme, INT f_print_subscripts);
	void print_row_tactical_decomposition_scheme_tex(
		ostream &ost, INT f_enter_math_mode, 
		INT *row_classes, INT nb_row_classes,
		INT *col_classes, INT nb_col_classes, 
		INT *row_scheme, INT f_print_subscripts);
	void print_column_tactical_decomposition_scheme_tex(
		ostream &ost, INT f_enter_math_mode, 
		INT *row_classes, INT nb_row_classes,
		INT *col_classes, INT nb_col_classes, 
		INT *col_scheme, INT f_print_subscripts);
	void print_non_tactical_decomposition_scheme_tex(
		ostream &ost, INT f_enter_math_mode, 
		INT *row_classes, INT nb_row_classes,
		INT *col_classes, INT nb_col_classes, 
		INT f_print_subscripts);
	void row_scheme_to_col_scheme(orthogonal &O, 
		INT *row_classes, INT *row_class_inv, INT nb_row_classes,
		INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
		INT *row_scheme, INT *col_scheme, INT verbose_level);
	void get_row_decomposition_scheme(orthogonal &O, 
		INT *row_classes, INT *row_class_inv, INT nb_row_classes,
		INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
		INT *row_scheme, INT verbose_level);
	void get_col_decomposition_scheme(orthogonal &O, 
		INT *row_classes, INT *row_class_inv, INT nb_row_classes,
		INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
		INT *col_scheme, INT verbose_level);
	INT refine_column_partition(orthogonal &O, INT ht0, INT verbose_level);
	INT refine_row_partition(orthogonal &O, INT ht0, INT verbose_level);
	INT hash_column_refinement_info(INT ht0, INT *data, INT depth, 
		INT hash0);
	INT hash_row_refinement_info(INT ht0, INT *data, INT depth, INT hash0);
	void print_column_refinement_info(INT ht0, INT *data, INT depth);
	void print_row_refinement_info(INT ht0, INT *data, INT depth);
	void radix_sort(INT left, INT right, INT *C, 
		INT length, INT radix, INT verbose_level);
	void radix_sort_bits(INT left, INT right, 
		INT *C, INT length, INT radix, INT mask, INT verbose_level);
	void swap_ij(INT *perm, INT *perm_inv, INT i, INT j);
	INT my_log2(INT m);
	void split_by_orbit_partition(INT nb_orbits, 
		INT *orbit_first, INT *orbit_len, INT *orbit,
		INT offset, 
		INT verbose_level);
};

// #############################################################################
// set_of_sets.C:
// #############################################################################

//! to store a set of sets


class set_of_sets {

public:
	
	INT underlying_set_size;
	INT nb_sets;
	INT **Sets;
	INT *Set_size;


	set_of_sets();
	~set_of_sets();
	void null();
	void freeself();
	set_of_sets *copy();
	void init_simple(INT underlying_set_size, 
		INT nb_sets, INT verbose_level);
	void init_from_adjacency_matrix(INT n, INT *Adj, 
		INT verbose_level);
	void init(INT underlying_set_size, INT nb_sets, 
		INT **Pts, INT *Sz, INT verbose_level);
	void init_basic(INT underlying_set_size, 
		INT nb_sets, INT *Sz, INT verbose_level);
	void init_basic_constant_size(INT underlying_set_size, 
		INT nb_sets, INT constant_size, INT verbose_level);
	void init_from_file(INT underlying_set_size, 
		const BYTE *fname, INT verbose_level);
	void init_from_csv_file(INT underlying_set_size, 
		const BYTE *fname, INT verbose_level);
	void init_from_orbiter_file(INT underlying_set_size, 
		const BYTE *fname, INT verbose_level);
	void init_set(INT idx_of_set, INT *set, INT sz, 
		INT verbose_level);
		// Stores a copy of the given set.
	void init_cycle_structure(INT *perm, INT n, INT verbose_level);
	INT total_size();
	INT &element(INT i, INT j);
	void add_element(INT i, INT a);
	void print();
	void print_table();
	void print_table_tex(ostream &ost);
	void dualize(set_of_sets *&S, INT verbose_level);
	void remove_sets_of_given_size(INT k, 
		set_of_sets &S, INT *&Idx, 
		INT verbose_level);
	void extract_largest_sets(set_of_sets &S, 
		INT *&Idx, INT verbose_level);
	void intersection_matrix(
		INT *&intersection_type, INT &highest_intersection_number, 
		INT *&intersection_matrix, INT &nb_big_sets, 
		INT verbose_level);
	void compute_incidence_matrix(INT *&Inc, INT &m, INT &n, 
		INT verbose_level);
	void compute_and_print_tdo_row_scheme(ofstream &file, 
		INT verbose_level);
	void compute_and_print_tdo_col_scheme(ofstream &file, 
		INT verbose_level);
	void init_decomposition(decomposition *&D, INT verbose_level);
	void compute_tdo_decomposition(decomposition &D, 
		INT verbose_level);
	INT is_member(INT i, INT a, INT verbose_level);
	void sort_all(INT verbose_level);
	void all_pairwise_intersections(set_of_sets *&Intersections, 
		INT verbose_level);
	void pairwise_intersection_matrix(INT *&M, INT verbose_level);
	void all_triple_intersections(set_of_sets *&Intersections, 
		INT verbose_level);
	INT has_constant_size_property();
	INT largest_set_size();
	void save_csv(const BYTE *fname, 
		INT f_make_heading, INT verbose_level);
	INT find_common_element_in_two_sets(INT idx1, INT idx2, 
		INT &common_elt);
	void sort();
	void sort_big(INT verbose_level);
	void compute_orbits(INT &nb_orbits, INT *&orbit, INT *&orbit_inv, 
		INT *&orbit_first, INT *&orbit_len, 
		void (*compute_image_function)(set_of_sets *S, 
			void *compute_image_data, INT elt_idx, INT gen_idx, 
			INT &idx_of_image, INT verbose_level), 
		void *compute_image_data, 
		INT nb_gens, 
		INT verbose_level);
	INT number_of_eckardt_points(INT verbose_level);
	void get_eckardt_points(INT *&E, INT &nb_E, INT verbose_level);
};

INT set_of_sets_compare_func(void *data, INT i, INT j, void *extra_data);
void set_of_sets_swap_func(void *data, INT i, INT j, void *extra_data);

// #############################################################################
// sorting.C:
// #############################################################################

void INT_vec_search_vec(INT *v, INT len, INT *A, INT A_sz, INT *Idx);
void INT_vec_search_vec_linear(INT *v, INT len, INT *A, INT A_sz, INT *Idx);
INT INT_vec_is_subset_of(INT *set, INT sz, INT *big_set, INT big_set_sz);
void INT_vec_swap_points(INT *list, INT *list_inv, INT idx1, INT idx2);
INT INT_vec_is_sorted(INT *v, INT len);
void INT_vec_sort_and_remove_duplicates(INT *v, INT &len);
INT INT_vec_sort_and_test_if_contained(INT *v1, INT len1, INT *v2, INT len2);
INT INT_vecs_are_disjoint(INT *v1, INT len1, INT *v2, INT len2);
INT INT_vecs_find_common_element(INT *v1, INT len1, 
	INT *v2, INT len2, INT &idx1, INT &idx2);
void INT_vec_insert_and_reallocate_if_necessary(INT *&vec, 
	INT &used_length, INT &alloc_length, INT a, INT verbose_level);
void INT_vec_append_and_reallocate_if_necessary(INT *&vec, 
	INT &used_length, INT &alloc_length, INT a, INT verbose_level);
INT INT_vec_is_zero(INT *v, INT len);
INT test_if_sets_are_equal(INT *set1, INT *set2, INT set_size);
void test_if_set(INT *set, INT set_size);
INT test_if_set_with_return_value(INT *set, INT set_size);
void rearrange_subset(INT n, INT k, INT *set, 
	INT *subset, INT *rearranged_set, INT verbose_level);
INT INT_vec_search_linear(INT *v, INT len, INT a, INT &idx);
void INT_vec_intersect(INT *v1, INT len1, INT *v2, INT len2, 
	INT *&v3, INT &len3);
void INT_vec_intersect_sorted_vectors(INT *v1, INT len1, 
	INT *v2, INT len2, INT *v3, INT &len3);
void INT_vec_sorting_permutation(INT *v, INT len, INT *perm, 
	INT *perm_inv, INT f_increasingly);
// perm and perm_inv must be allocated to len elements
INT INT_compare_increasingly(void *a, void *b, void *data);
INT INT_compare_decreasingly(void *a, void *b, void *data);
void INT_vec_quicksort(INT *v, INT (*compare_func)(INT a, INT b), 
	INT left, INT right);
INT compare_increasingly_INT(INT a, INT b);
INT compare_decreasingly_INT(INT a, INT b);
void INT_vec_quicksort_increasingly(INT *v, INT len);
void INT_vec_quicksort_decreasingly(INT *v, INT len);
void quicksort_array(INT len, void **v, 
	INT (*compare_func)(void *a, void *b, void *data), void *data);
void quicksort_array_with_perm(INT len, void **v, INT *perm, 
	INT (*compare_func)(void *a, void *b, void *data), void *data);
void INT_vec_sort(INT len, INT *p);
int int_vec_compare(int *p, int *q, int len);
INT INT_vec_compare(INT *p, INT *q, INT len);
INT INT_vec_compare_stride(INT *p, INT *q, INT len, INT stride);
INT vec_search(void **v, INT (*compare_func)(void *a, void *b, void *data), 
	void *data_for_compare, 
	INT len, void *a, INT &idx, INT verbose_level);
INT vec_search_general(void *vec, 
	INT (*compare_func)(void *vec, void *a, INT b, void *data_for_compare), 
	void *data_for_compare, 
	INT len, void *a, INT &idx, INT verbose_level);
INT INT_vec_search_and_insert_if_necessary(INT *v, INT &len, INT a);
INT INT_vec_search_and_remove_if_found(INT *v, INT &len, INT a);
INT INT_vec_search(INT *v, INT len, INT a, INT &idx);
	// This function finds the last occurence of the element a.
	// If a is not found, it returns in idx the position 
	// where it should be inserted if 
	// the vector is assumed to be in increasing order.
INT INT_vec_search_first_occurence(INT *v, INT len, INT a, INT &idx);
	// This function finds the first occurence of the element a.
INT longinteger_vec_search(longinteger_object *v, INT len, 
	longinteger_object &a, INT &idx);
void INT_vec_classify_and_print(ostream &ost, INT *v, INT l);
void INT_vec_values(INT *v, INT l, INT *&w, INT &w_len);
void INT_vec_multiplicities(INT *v, INT l, INT *&w, INT &w_len);
void INT_vec_values_and_multiplicities(INT *v, INT l, 
	INT *&val, INT *&mult, INT &nb_values);
void INT_vec_classify(INT length, INT *the_vec, INT *&the_vec_sorted, 
	INT *&sorting_perm, INT *&sorting_perm_inv, 
	INT &nb_types, INT *&type_first, INT *&type_len);
void INT_vec_classify_with_arrays(INT length, 
	INT *the_vec, INT *the_vec_sorted, 
	INT *sorting_perm, INT *sorting_perm_inv, 
	INT &nb_types, INT *type_first, INT *type_len);
void INT_vec_sorted_collect_types(INT length, INT *the_vec_sorted, 
	INT &nb_types, INT *type_first, INT *type_len);
void INT_vec_print_classified(ostream &ost, INT *vec, INT len);
void INT_vec_print_types(ostream &ost, 
	INT f_backwards, INT *the_vec_sorted, 
	INT nb_types, INT *type_first, INT *type_len);
void INT_vec_print_types_naked(ostream &ost, INT f_backwards, 
	INT *the_vec_sorted, 
	INT nb_types, INT *type_first, INT *type_len);
void INT_vec_print_types_naked_tex(ostream &ost, INT f_backwards, 
	INT *the_vec_sorted, 
	INT nb_types, INT *type_first, INT *type_len);
void Heapsort(void *v, INT len, INT entry_size_in_bytes, 
	INT (*compare_func)(void *v1, void *v2));
void Heapsort_general(void *data, INT len, 
	INT (*compare_func)(void *data, INT i, INT j, void *extra_data), 
	void (*swap_func)(void *data, INT i, INT j, void *extra_data), 
	void *extra_data);
INT search_general(void *data, INT len, INT *search_object, INT &idx, 
	INT (*compare_func)(void *data, INT i, INT *search_object, 
	void *extra_data), 
	void *extra_data, INT verbose_level);
	// This function finds the last occurence of the element a.
	// If a is not found, it returns in idx the position 
	// where it should be inserted if 
	// the vector is assumed to be in increasing order.
void INT_vec_heapsort(INT *v, INT len);
void INT_vec_heapsort_with_log(INT *v, INT *w, INT len);
void heapsort_make_heap(INT *v, INT len);
void heapsort_make_heap_with_log(INT *v, INT *w, INT len);
void Heapsort_make_heap(void *v, INT len, INT entry_size_in_bytes, 
	INT (*compare_func)(void *v1, void *v2));
void Heapsort_general_make_heap(void *data, INT len, 
	INT (*compare_func)(void *data, INT i, INT j, void *extra_data), 
	void (*swap_func)(void *data, INT i, INT j, void *extra_data), 
	void *extra_data);
void heapsort_sift_down(INT *v, INT start, INT end);
void heapsort_sift_down_with_log(INT *v, INT *w, INT start, INT end);
void Heapsort_sift_down(void *v, INT start, INT end, INT entry_size_in_bytes, 
	INT (*compare_func)(void *v1, void *v2));
void Heapsort_general_sift_down(void *data, INT start, INT end, 
	INT (*compare_func)(void *data, INT i, INT j, void *extra_data), 
	void (*swap_func)(void *data, INT i, INT j, void *extra_data), 
	void *extra_data);
void heapsort_swap(INT *v, INT i, INT j);
void Heapsort_swap(void *v, INT i, INT j, INT entry_size_in_bytes);
INT is_all_digits(BYTE *p);
void find_points_by_multiplicity(INT *data, INT data_sz, INT multiplicity, 
	INT *&pts, INT &nb_pts);

// #############################################################################
// spreadsheet.C:
// #############################################################################

//! for reading and writing of csv files


class spreadsheet {

public:

	BYTE **tokens;
	INT nb_tokens;

	INT *line_start, *line_size;
	INT nb_lines;

	INT nb_rows, nb_cols;
	INT *Table;


	spreadsheet();
	~spreadsheet();
	void null();
	void freeself();
	void init_set_of_sets(set_of_sets *S, INT f_make_heading);
	void init_INT_matrix(INT nb_rows, INT nb_cols, INT *A);
	void init_empty_table(INT nb_rows, INT nb_cols);
	void fill_entry_with_text(INT row_idx, 
		INT col_idx, const BYTE *text);
	void fill_column_with_text(INT col_idx, const BYTE **text, 
		const BYTE *heading);
	void fill_column_with_INT(INT col_idx, INT *data, 
		const BYTE *heading);
	void fill_column_with_row_index(INT col_idx, 
		const BYTE *heading);
	void add_token(BYTE *label);
	void save(const BYTE *fname, INT verbose_level);
	void read_spreadsheet(const BYTE *fname, INT verbose_level);
	void print_table(ostream &ost, INT f_enclose_in_parentheses);
	void print_table_latex_all_columns(ostream &ost, 
		INT f_enclose_in_parentheses);
	void print_table_latex(ostream &ost, INT *f_column_select, 
		INT f_enclose_in_parentheses);
	void print_table_row(INT row, INT f_enclose_in_parentheses, 
		ostream &ost);
	void print_table_row_latex(INT row, INT *f_column_select, 
		INT f_enclose_in_parentheses, ostream &ost);
	void print_table_row_detailed(INT row, ostream &ost);
	void print_table_with_row_selection(INT *f_selected, 
		ostream &ost);
	void print_table_sorted(ostream &ost, const BYTE *sort_by);
	void add_column_with_constant_value(BYTE *label, BYTE *value);
	void reallocate_table();
	void reallocate_table_add_row();
	INT find_by_column(const BYTE *join_by);
	void tokenize(const BYTE *fname, 
		BYTE **&tokens, INT &nb_tokens, INT verbose_level);
	void remove_quotes(INT verbose_level);
	void remove_rows(const BYTE *drop_column, const BYTE *drop_label, 
		INT verbose_level);
	void remove_rows_where_field_is_empty(const BYTE *drop_column, 
		INT verbose_level);
	void find_rows(INT verbose_level);
	void get_value_double_or_NA(INT i, INT j, double &val, INT &f_NA);
	BYTE *get_string(INT i, INT j);
	INT get_INT(INT i, INT j);
	double get_double(INT i, INT j);
	void join_with(spreadsheet *S2, INT by1, INT by2, 
		INT verbose_level);
	void patch_with(spreadsheet *S2, BYTE *join_by);


};

INT my_atoi(BYTE *str);
INT compare_strings(void *a, void *b, void *data);

// #############################################################################
// super_fast_hash.C:
// #############################################################################

uint32_t SuperFastHash (const char * data, int len);


// #############################################################################
// vector_hashing.C:
// #############################################################################

//! hash tables


class vector_hashing {

public:
	INT data_size;
	INT N;
	INT bit_length;
	INT *vector_data;
	INT *H;
	INT *H_sorted;
	INT *perm;
	INT *perm_inv;
	INT nb_types;
	INT *type_first;
	INT *type_len;
	INT *type_value;


	vector_hashing();
	~vector_hashing();
	void allocate(INT data_size, INT N, INT bit_length);
	void compute_tables(INT verbose_level);
	void print();
	INT rank(INT *data);
	void unrank(INT rk, INT *data);
};






