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
namespace other {
namespace data_structures {


// #############################################################################
// algorithms.cpp
// #############################################################################



//! catch all class for algorithms


class algorithms {
public:

	algorithms();
	~algorithms();
	int hashing(
			int hash0, int a);
	int hashing_fixed_width(
			int hash0, int a, int bit_length);
	void uchar_print_bitwise(
			std::ostream &ost, unsigned char u);
	void uchar_move(
			const unsigned char *p, unsigned char *q, int len);
	void uchar_expand_4(
			const unsigned char *p, unsigned char *q, int len);
	void uchar_compress_4(
			const unsigned char *p, unsigned char *q, int len);
	void uchar_zero(
			unsigned char *p, int len);
	void uchar_xor(
			unsigned char *in1, unsigned char *in2, unsigned char *out, int len);
	int uchar_compare(
			unsigned char *in1, unsigned char *in2, int len);
	int uchar_is_zero(
			unsigned char *in, int len);
	void int_swap(
			int& x, int& y);
	void lint_swap(
			long int & x, long int & y);
	void print_pointer_hex(
			std::ostream &ost, void *p);
	void print_uint32_hex(
			std::ostream &ost, uint32_t val);
	void print_hex(
			std::ostream &ost, unsigned char *p, int len);
	void print_binary(
			std::ostream &ost, unsigned char *p, int len);
	void print_uint32_binary(
			std::ostream &ost, uint32_t val);
	void print_hex_digit(
			std::ostream &ost, int digit);
	void print_bits(
			std::ostream &ost, char *data, int data_size);
	unsigned long int make_bitword(
			char *data, int data_size);
	void read_hex_data(
			std::string &str,
			char *&data, int &data_size, int verbose_level);
	unsigned char read_hex_digit(
			char digit);
	void print_repeated_character(
			std::ostream &ost, char c, int n);
	uint32_t root_of_tree_uint32_t(
			uint32_t* S, uint32_t i);
	void solve_diophant(
			int *Inc,
		int nb_rows, int nb_cols, int nb_needed,
		int f_has_Rhs, int *Rhs,
		int *&Solutions, int &nb_sol,
		long int &nb_backtrack, int &dt,
		int f_DLX,
		int verbose_level);
	// allocates Solutions[nb_sol * nb_needed]
	uint32_t SuperFastHash(
			const char * data, int len);
	uint32_t SuperFastHash_uint(
			const unsigned int * p, int sz);
	void union_of_sets(
			std::string &fname_set_of_sets,
			std::string &fname_input,
			std::string &fname_output, int verbose_level);
	void dot_product_of_columns(
			std::string &label, int verbose_level);
	void dot_product_of_rows(
			std::string &label, int verbose_level);
	void matrix_multiply_over_Z(
			std::string &label1, std::string &label2,
			int verbose_level);
	void matrix_rowspan_over_R(
			std::string &label, int verbose_level);
	int binary_logarithm(
			int m);
	char make_single_hex_digit(
			int c);
	void process_class_list(
			std::vector<std::vector<std::string> > &Classes_parsed,
			std::string &fname_cross_ref,
			int verbose_level);
	void filter_duplicates_and_make_array_of_long_int(
			std::vector<long int> &In, long int *&Out, int &size,
			int verbose_level);
	void set_minus(
			std::vector<long int> &In, long int *subtract_this, int size,
			std::vector<long int> &Out,
			int verbose_level);
	void create_layered_graph_from_tree(
			int degree,
			int *orbit_first,
			int *orbit_len,
			int *orbit,
			int *orbit_inv,
			int *prev,
			int *label,
			int orbit_no,
			combinatorics::graph_theory::layered_graph *&LG,
			int verbose_level);
	void tree_trace_back(
			int *orbit_inv,
			int *prev,
			int i, int &j);
	void make_layered_graph_for_schreier_vector_tree(
		int n, int *pts, int *prev,
		int f_use_pts_inv, int *pts_inv,
		std::string &fname_base,
		combinatorics::graph_theory::layered_graph *&LG,
		int verbose_level);
	// called from sims_io.cpp
	void make_and_draw_tree(
			std::string &fname_base,
			int n, int *pts, int *prev, int f_use_pts_inv, int *pts_inv,
			other::graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
	void schreier_vector_compute_depth_and_ancestor(
		int n, int *pts, int *prev, int f_prev_is_point_index, int *pts_inv,
		int *&depth, int *&ancestor, int verbose_level);
	int schreier_vector_determine_depth_recursion(
		int n, int *pts, int *prev, int f_use_pts_inv,
		int *pts_inv, int *depth, int *ancestor, int pos);

};


// #############################################################################
// ancestry_family.cpp
// #############################################################################



//! genealogy family record


class ancestry_family {
public:

	ancestry_tree *Tree;
	int idx;

	int start;
	int length;
	std::string id;

	std::string husband;
	int husband_index;
	int husband_family_index;

	std::string wife;
	int wife_index;
	int wife_family_index;

	std::vector<std::string> child;
	std::vector<int> child_index;
	std::vector<std::vector<int> > child_family_index;

	std::vector<int> topo_downlink;


	ancestry_family();
	~ancestry_family();
	void init(
			ancestry_tree *Tree,
			int idx,
			int start, int length,
			std::vector<std::vector<std::string> > &Data,
			int verbose_level);
	void get_connnections(
			int verbose_level);
	std::string get_initials(
			int verbose_level);
	void topo_sort_prepare(
			int verbose_level);
	int topo_rank_of_parents(
			int *topo_rank, int verbose_level);

};


// #############################################################################
// ancestry_indi.cpp
// #############################################################################



//! genealogy individual record


class ancestry_indi {
public:

	ancestry_tree *Tree;
	int idx;

	int start;
	int length;
	std::string id;
	std::string name;
	std::string given_name;
	std::string sur_name;
	std::string sex;
	std::string famc;
	std::string fams;
	std::string birth_date;
	std::string death_date;

	std::vector<int> family_index;

	ancestry_indi();
	~ancestry_indi();
	void init(
			ancestry_tree *Tree,
			int idx,
			int start, int length,
			std::vector<std::vector<std::string> > &Data,
			int verbose_level);
	std::string initials(
			int verbose_level);

};



// #############################################################################
// ancestry_tree.cpp
// #############################################################################



//! genealogy family tree


class ancestry_tree {
public:

	std::string fname_gedcom;
	std::string fname_base;

	std::vector<std::vector<std::string> > Data;
	std::vector<std::vector<int> > Indi;
	std::vector<std::vector<int> > Fam;

	int nb_indi, nb_fam;
	data_structures::ancestry_family **Family;
	data_structures::ancestry_indi **Individual;

	ancestry_tree();
	~ancestry_tree();
	void read_gedcom(
			std::string &fname_gedcom, int verbose_level);
	void topo_sort(
			int *&topo_rank, int &rk_max, int verbose_level);
	void create_poset(
			combinatorics::graph_theory::layered_graph *&L, int verbose_level);
	void topo_sort_prepare(
			int verbose_level);
	void get_connnections(
			int verbose_level);
	int find_individual(
			std::string &id, int verbose_level);
	int find_in_family_as_child(
			int indi_idx);
	std::vector<int> find_in_family_as_parent(
			int indi_idx);
	void register_individual(
			int individual_index, int family_idx,
			int verbose_level);

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
	void init(
			int m, int n, int verbose_level);
	void unrank_PG_elements_in_columns_consecutively(
			algebra::field_theory::finite_field *F,
			long int start_value, int verbose_level);
	void rank_PG_elements_in_columns(
			algebra::field_theory::finite_field *F,
			int *perms, unsigned int *PG_ranks,
			int verbose_level);
	void print();
	void zero_out();
	int s_ij(
			int i, int j);
	void m_ij(
			int i, int j, int a);
	void mult_int_matrix_from_the_left(
			int *A, int Am, int An,
			bitmatrix *Out, int verbose_level);

};

// #############################################################################
// bitvector.cpp
// #############################################################################

//! compact storage of 0/1-data as bitvectors

class bitvector {

private:
	unsigned char *data; // [allocated_length]
	long int length; // number of bits used
	long int allocated_length;


public:

	bitvector();
	~bitvector();
	void allocate(
			long int length);
	void zero();
	long int get_length();
	long int get_allocated_length();
	unsigned char *get_data();
	void m_i(
			long int i, int a);
	void set_bit(
			long int i);
	int s_i(
			long int i);
	void save(
			std::ofstream &fp);
	void load(
			std::ifstream &fp);
	uint32_t compute_hash();
	void print();

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
	void read(
			std::string &fname, int f_casenumbers,
			int verbose_level);
	void read_candidates(
			std::string &candidates_fname,
			int verbose_level);
};



// #############################################################################
// data_structures_global.cpp:
// #############################################################################


//! a catch-all container class for everything related to data structures


class data_structures_global {
public:
	data_structures_global();
	~data_structures_global();
	void bitvector_m_ii(
			unsigned char *bitvec, long int i, int a);
	void bitvector_set_bit(
			unsigned char *bitvec, long int i);
	void bitvector_set_bit_reversed(
			unsigned char *bitvec, long int i);
	int bitvector_s_i(
			unsigned char *bitvec, long int i);
	uint32_t int_vec_hash(
			int *data, int len);
	uint64_t lint_vec_hash(
			long int *data, int len);
	uint32_t char_vec_hash(
			char *data, int len);
	int int_vec_hash_after_sorting(
			int *data, int len);
	long int lint_vec_hash_after_sorting(
			long int *data, int len);

};


// #############################################################################
// fancy_set.cpp
// #############################################################################

//! subset of size k of a set of size n


class fancy_set {
	
	public:

	int n;
	int k;
	long int *set; // [n]
	long int *set_inv; // [n]

	fancy_set();
	~fancy_set();
	void init(
			int n, int verbose_level);
	void init_with_set(
			int n, int k, int *subset,
			int verbose_level);
	void print();
	void println();
	void swap(
			int pos, int a);
	int is_contained(
			int a);
	void copy_to(
			fancy_set *to);
	void add_element(
			int elt);
	void add_elements(
			int *elts, int nb);
	void delete_elements(
			int *elts, int nb);
	void delete_element(
			int elt);
	void select_subset(
			int *elts, int nb);
	void intersect_with(
			int *elts, int nb);
	void subtract_set(
			fancy_set *set_to_subtract);
	void sort();
	int compare_lexicographically(
			fancy_set *second_set);
	void complement(
			fancy_set *compl_set);
	int is_subset(
			fancy_set *set2);
	int is_equal(
			fancy_set *set2);
	void save(
			std::string &fname, int verbose_level);

};


// #############################################################################
// forest.cpp
// #############################################################################

//! A forest representation of a set of group orbits. One orbit is one tree.


class forest {

	public:

	int degree;


	// suggested new class: schreier_forest:
	// long int degree
	int *orbit; // [A->degree]
	int *orbit_inv; // [A->degree]

		// prev and label are indexed
		// by the points in the order as listed in orbit.

	int *prev; // [A->degree]
	int *label; // [A->degree]

		// prev[coset] is the point which maps
		// to orbit[coset] under generator label[coset]

	//int *orbit_no; // [A->degree]
		// to find out which orbit point a lies in,
		// use orbit_number(pt).
		// It used to be orbit_no[orbit_inv[a]]
	// from extend_orbits:
	//prev[total] = cur_pt;
	//label[total] = i;

	int *orbit_first;  // [A->degree + 1]
	int *orbit_len;  // [A->degree]
	int nb_orbits;

	// end schreier_forest

	forest();
	~forest();
	void init(
			int degree, int verbose_level);
	void allocate_tables(
			int verbose_level);
	void initialize_tables(
			int verbose_level);
	void swap_points(
			int i, int j, int verbose_level);
	void move_point_here(
			int here, int pt);
	void move_point_set_lint_to_here(
			long int *subset, int len);
	int orbit_representative(
			int pt);
	int depth_in_tree(
			int j);
	int sum_up_orbit_lengths();
	void get_path_and_labels(
			std::vector<int> &path,
			std::vector<int> &labels,
			int i, int verbose_level);
	void trace_back(
			int i, int &j);
	void trace_back_and_record_path(
			int *path, int i, int &j);
	void intersection_vector(
			int *set, int len,
		int *intersection_cnt);
	void get_orbit_partition_of_points_and_lines(
			other::data_structures::partitionstack &S,
			int verbose_level);
	void get_orbit_partition(
			other::data_structures::partitionstack &S,
		int verbose_level);
	void get_orbit_in_order(
			std::vector<int> &Orb,
		int orbit_idx, int verbose_level);
	void get_orbit(
			int orbit_idx, long int *set, int &len,
		int verbose_level);
	void compute_orbit_statistic(
			int *set, int set_size,
		int *orbit_count, int verbose_level);
	void compute_orbit_statistic_lint(
			long int *set, int set_size,
		int *orbit_count, int verbose_level);
	other::data_structures::set_of_sets *get_set_of_sets(
			int verbose_level);
	void orbits_as_set_of_sets(
			other::data_structures::set_of_sets *&S,
			int verbose_level);
	void get_orbit_reps(
			int *&Reps, int &nb_reps, int verbose_level);
	int find_shortest_orbit_if_unique(
			int &idx);
	void elements_in_orbit_of(
			int pt, int *orb, int &nb,
		int verbose_level);
	void get_orbit_length(
			int *&orbit_length, int verbose_level);
	void get_orbit_lengths_once_each(
			int *&orbit_lengths,
		int &nb_orbit_lengths);
	int orbit_number(
			int pt);
	void get_orbit_number_and_position(
			int pt, int &orbit_idx, int &orbit_pos,
			int verbose_level);
	void get_orbit_decomposition_scheme_of_graph(
		int *Adj, int n, int *&Decomp_scheme,
		int verbose_level);
	void create_point_list_sorted(
			int *&point_list, int &point_list_length);
	int get_num_points();
		// This function returns the number of points in the
		// schreier forest
	double get_average_word_length();
		// This function returns the average word length of the forest.
	double get_average_word_length(
			int orbit_idx);


	// forest_io.cpp:
	void print_orbit(
			int orbit_no);
	void print_orbit_using_labels(
			int orbit_no, long int *labels);
	void print_orbit(
			std::ostream &ost, int orbit_no);
	void print_orbit_with_original_labels(
			std::ostream &ost, int orbit_no);
	void print_orbit_tex(
			std::ostream &ost, int orbit_no);
	void print_orbit_sorted_tex(
			std::ostream &ost,
			int orbit_no, int f_truncate, int max_length);
	void get_orbit_sorted(
			int *&v, int &len, int orbit_no);
	void print_and_list_orbits_sorted_by_length(
			std::ostream &ost);
	void print_orbit_sorted_with_original_labels_tex(
			std::ostream &ost,
			int orbit_no, int f_truncate, int max_length);
	void print_orbit_using_labels(
			std::ostream &ost, int orbit_no, long int *labels);
	void print_orbit_using_callback(
			std::ostream &ost, int orbit_no,
		void (*print_point)(
				std::ostream &ost, int pt, void *data),
		void *data);
	void print_orbit_type(
			int f_backwards);
	void list_all_orbits_tex(
			std::ostream &ost);
	void print_orbit_through_labels(
			std::ostream &ost,
		int orbit_no, long int *point_labels);
	void print_orbit_sorted(
			std::ostream &ost, int orbit_no);
	void print_orbit(
			int cur, int last);
	void print_tree(
			int orbit_no);
	void draw_forest(
			std::string &fname_mask,
			other::graphics::layered_graph_draw_options *Opt,
			int f_has_point_labels, long int *point_labels,
			int verbose_level);
	void get_orbit_by_levels(
			int orbit_no,
			other::data_structures::set_of_sets *&SoS,
			int verbose_level);
	void export_tree_as_layered_graph_and_save(
			int orbit_no,
			std::string &fname_mask,
			int verbose_level);
	void export_tree_as_layered_graph(
			int orbit_no,
			combinatorics::graph_theory::layered_graph *&LG,
			int verbose_level);
	void draw_tree(
			std::string &fname,
			other::graphics::layered_graph_draw_options *Opt,
			int orbit_no,
			int f_has_point_labels, long int *point_labels,
			int verbose_level);
	void draw_tree2(
			std::string &fname,
			other::graphics::layered_graph_draw_options *Opt,
			int *weight, int *placement_x, int max_depth,
			int i, int last,
			int f_has_point_labels, long int *point_labels,
			int verbose_level);
	void subtree_draw_lines(
			other::graphics::mp_graphics &G,
			other::graphics::layered_graph_draw_options *Opt,
			int parent_x, int parent_y, int *weight,
			int *placement_x, int max_depth, int i, int last,
			int y_max,
			int verbose_level);
	void subtree_draw_vertices(
			other::graphics::mp_graphics &G,
			other::graphics::layered_graph_draw_options *Opt,
			int parent_x, int parent_y, int *weight,
			int *placement_x, int max_depth, int i, int last,
			int f_has_point_labels, long int *point_labels,
			int y_max,
			int verbose_level);
	void subtree_place(
			int *weight, int *placement_x,
		int left, int right, int i, int last);
	int subtree_calc_weight(
			int *weight, int &max_depth,
		int i, int last);
	int subtree_depth_first(
			std::ostream &ost, int *path, int i, int last);
	void print_path(
			std::ostream &ost, int *path, int l);
	void write_to_file_csv(
			std::string &fname_csv, int verbose_level);

	void write_to_file_binary(
			std::ofstream &fp, int verbose_level);
	void read_from_file_binary(
			std::ifstream &fp, int verbose_level);
	void write_file_binary(
			std::string &fname, int verbose_level);
	void read_file_binary(
			std::string &fname, int verbose_level);

	void print_orbit_lengths(
			std::ostream &ost);
	void print_orbit_lengths_tex(
			std::ostream &ost);
	void print_fixed_points_tex(
			std::ostream &ost);
	void print_orbit_length_distribution(
			std::ostream &ost);
	void print_orbit_length_distribution_to_string(
			std::string &str);
	void print_orbit_reps(
			std::ostream &ost);
	void print(
			std::ostream &ost);
	void make_orbit_trees(
			std::ostream &ost,
			std::string &fname_mask,
			other::graphics::layered_graph_draw_options *Opt,
			int verbose_level);
	void print_and_list_orbit_tex(
			int i, std::ostream &ost);
	void print_and_list_orbits(
			std::ostream &ost);
	void print_and_list_orbits_with_original_labels(
			std::ostream &ost);
	void print_and_list_orbits_tex(
			std::ostream &ost);
	void print_and_list_non_trivial_orbits_tex(
			std::ostream &ost);
	void print_and_list_orbits_sorted_by_length(
		std::ostream &ost, int f_tex);
	void print_and_list_orbits_of_given_length(
		std::ostream &ost, int len);

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

	int *perm_inv;
	int *perm;

	int_matrix();
	~int_matrix();
	void allocate(
			int m, int n);
	void allocate_and_initialize_with_zero(
			int m, int n);
	void allocate_and_init(
			int m, int n, int *Mtx);
	int &s_ij(
			int i, int j);
	int &s_m();
	int &s_n();
	void print();
	void sort_rows(
			int verbose_level);
	void sort_rows_in_reverse(
			int verbose_level);
	void remove_duplicates(
			int verbose_level);
	void check_that_entries_are_distinct(
			int verbose_level);
	int search(
			int *entry, int &idx, int verbose_level);
	int search_first_column_only(
			int value, int &idx, int verbose_level);
	void write_csv(
			std::string &fname, int verbose_level);
	void write_index_set_csv(
			std::string &fname, int verbose_level);

};


// #############################################################################
// int_vec.cpp:
// #############################################################################


//! int arrays

class int_vec {
public:

	int_vec();
	~int_vec();
	void add(
			int *v1, int *v2, int *w, int len);
	void add3(
			int *v1, int *v2, int *v3, int *w, int len);
	void apply(
			int *from, int *through, int *to, int len);
	void apply_lint(
			int *from, long int *through, long int *to, int len);
	int is_constant(
			int *v,  int len);
	int is_constant_on_subset(
			int *v,
		int *subset, int sz, int &value);
	void take_away(
			int *v, int &len,
			int *take_away, int nb_take_away);
		// v must be sorted
	int count_number_of_nonzero_entries(
			int *v, int len);
	int find_first_nonzero_entry(
			int *v, int len);
	void zero(
			int *v, long int len);
	int is_zero(
			int *v, long int len);
	void one(
			int *v, long int len);
	void mone(
			int *v, long int len);
	int is_Hamming_weight_one(
			int *v, int &idx_nonzero, long int len);
	void copy(
			int *from, int *to, long int len);
	void copy_to_lint(
			int *from, long int *to, long int len);
	void swap(
			int *v1, int *v2, long int len);
	void delete_element_assume_sorted(
			int *v,
		int &len, int a);
	void complement(
			int *v, int n, int k);
	// computes the complement to v + k (v must be allocated to n elements)
	// the first k elements of v[] must be in increasing order.
	void complement(
			int *v, int *w, int n, int k);
	// computes the complement of v[k] in the set {0,...,n-1} to w[n - k]
	void init5(
			int *v, int a0, int a1, int a2, int a3, int a4);
	int minimum(
			int *v, int len);
	int maximum(
			int *v, int len);
	void copy(
			int len, int *from, int *to);
	int first_difference(
			int *p, int *q, int len);
	int vec_max_log_of_entries(
			std::vector<std::vector<int> > &p);
	void vec_print(
			std::vector<std::vector<int> > &p);
	void vec_print(
			std::vector<std::vector<int> > &p, int w);
	void distribution_compute_and_print(
			std::ostream &ost,
		int *v, int v_len);
	void distribution(
			int *v,
		int len_v, int *&val, int *&mult, int &len);
	void print(
			std::ostream &ost, std::vector<int> &v);
	void print_stl(
			std::ostream &ost, std::vector<int> &v);
	void print(
			std::ostream &ost, int *v, int len);
	void print_str(
			std::stringstream &ost, int *v, int len);
	void print_bare_str(
			std::stringstream &ost, int *v, int len);
	void print_as_table(
			std::ostream &ost, int *v, int len, int width);
	void print_fully(
			std::ostream &ost, std::vector<int> &v);
	void print_stl_fully(
			std::ostream &ost, std::vector<int> &v);
	void print_fully(
			std::ostream &ost, int *v, int len);
	void print_bare_fully(
			std::ostream &ost, int *v, int len);
	void print_dense(
			std::ostream &ost, int *v, int len);
	void print_Cpp(
			std::ostream &ost, int *v, int len);
	void print_GAP(
			std::ostream &ost, int *v, int len);
	void print_classified(
			int *v, int len);
	void print_classified_str(
			std::stringstream &sstr,
			int *v, int len, int f_backwards);
	void scan(
			std::string &s, int *&v, int &len);
	void scan_from_stream(
			std::istream & is, int *&v, int &len);
	void print_to_str(
			std::string &s, int *data, int len);
	void print_to_str_bare(
			std::string &s, int *data, int len);
	void print(
			int *v, int len);
	void print_integer_matrix(
			std::ostream &ost,
		int *p, int m, int n);
	void print_integer_matrix_width(
			std::ostream &ost,
		int *p, int m, int n, int dim_n, int w);
	void print_integer_matrix_in_C_source(
			std::ostream &ost,
		int *p, int m, int n);
	void matrix_make_block_matrix_2x2(
			int *Mtx,
		int k, int *A, int *B, int *C, int *D);
	void matrix_delete_column_in_place(
			int *Mtx,
		int k, int n, int pivot);
	int matrix_max_log_of_entries(
			int *p, int m, int n);
	void matrix_print_ost(
			std::ostream &ost, int *p, int m, int n);
	void matrix_print_makefile_style_ost(
			std::ostream &ost, int *p, int m, int n);
	void matrix_print(
			int *p, int m, int n);
	void matrix_print_comma_separated(
			int *p, int m, int n);
	void matrix_print_tight(
			int *p, int m, int n);
	void matrix_print_ost(
			std::ostream &ost, int *p, int m, int n, int w);
	void matrix_print_makefile_style_ost(
			std::ostream &ost, int *p, int m, int n, int w);
	void matrix_print(
			int *p, int m, int n, int w);
	void matrix_print_comma_separated(
			int *p, int m, int n, int w);
	void matrix_print_nonzero_entries(
			int *p, int m, int n);
	void matrix_print_bitwise(
			int *p, int m, int n);
	void distribution_print(
			std::ostream &ost,
		int *val, int *mult, int len);
	void distribution_print_to_string(
			std::string &str, int *val, int *mult, int len);
	void set_print(
			std::ostream &ost, int *v, int len);
	void integer_vec_print(
			std::ostream &ost, int *v, int len);
	int hash(
			int *v, int len, int bit_length);
	void create_string_with_quotes(
			std::string &str, int *v, int len);
	void transpose(
			int *M, int m, int n, int *Mt);
	void print_as_polynomial_in_algebraic_notation(
			std::ostream &ost, int *coeff_vector, int len);
	int compare(
			int *p, int *q, int len);
	std::string stringify(
			int *v, int len);

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
	void allocate(
			int len);
	void allocate_and_init(
			int len, long int *V);
	void allocate_and_init_int(
			int len, int *V);
	void init_permutation_from_string(
			std::string &s);
	void read_ascii_file(
			std::string &fname);
	void read_binary_file_int4(
			std::string &fname);
	long int &s_i(
			int i);
	int &length();
	void print(
			std::ostream &ost);
	void zero();
	int search(
			int a, int &idx);
	void sort();
	void make_space();
	void append(
			int a);
	void insert_at(
			int a, int idx);
	void insert_if_not_yet_there(
			int a);
	void sort_and_remove_duplicates();
	void write_to_ascii_file(
			std::string &fname);
	void write_to_binary_file_int4(
			std::string &fname);
	void write_to_csv_file(
			std::string &fname, std::string &label);
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
	void apply(
			long int *from, long int *through, long int *to, int len);
	void take_away(
			long int *v, int &len,
			long int *take_away, int nb_take_away);
	void zero(
			long int *v, long int len);
	void one(
			long int *v, long int len);
	void mone(
			long int *v, long int len);
	void copy(
			long int *from, long int *to, long int len);
	void copy_to_int(
			long int *from, int *to, long int len);
	void complement(
			long int *v, long int *w, int n, int k);
	long int minimum(
			long int *v, int len);
	long int maximum(
			long int *v, int len);
	void matrix_print_width(
			std::ostream &ost,
		long int *p, int m, int n, int dim_n, int w);
	void matrix_print_nonzero_entries(
			long int *p, int m, int n);
	void set_print(
			long int *v, int len);
	void set_print(
			std::ostream &ost, long int *v, int len);
	void print(
			std::ostream &ost, long int *v, int len);
	void print(
			std::ostream &ost, std::vector<long int> &v);
	void print_stl(
			std::ostream &ost, std::vector<long int> &v);
	void print_as_table(
			std::ostream &ost, long int *v, int len, int width);
	void print_bare_fully(
			std::ostream &ost, long int *v, int len);
	void print_fully(
			std::ostream &ost, long int *v, int len);
	void print_fully(
			std::ostream &ost, std::vector<long int> &v);
	void print_stl_fully(
			std::ostream &ost, std::vector<long int> &v);
	int matrix_max_log_of_entries(
			long int *p, int m, int n);
	void matrix_print(
			long int *p, int m, int n);
	void matrix_print(
			long int *p, int m, int n, int w);
	void scan(
			std::string &s, long int *&v, int &len);
	void scan_from_stream(
			std::istream & is, long int *&v, int &len);
	void print_to_str(
			std::string &s, long int *data, int len);
	void print_to_str_bare(
			std::string &s, long int *data, int len);
	void create_string_with_quotes(
			std::string &str, long int *v, int len);
	int compare(
			long int *p, long int *q, int len);
	std::string stringify(
			long int *v, int len);


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


	void init(
			int entry_size, int page_length_log,
		int verbose_level);
	void add_elt_print_function(
		void (* elt_print)(void *p, void *data, std::ostream &ost),
		void *elt_print_data);
	void print();
	uchar *s_i_and_allocate(
			long int i);
	uchar *s_i_and_deallocate(
			long int i);
	uchar *s_i(
			long int i);
	uchar *s_i_and_allocation_bit(
			long int i, int &f_allocated);
	void check_allocation_table();
	long int store(
			uchar *elt);
	void dispose(
			long int hdl);
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

		int *pointList; // [n]
		int *invPointList; // [n]
		int *cellNumber; // [n]

		int *startCell; // [n + 1]
		int *cellSize; // [n + 1]
		int *parent; // [n + 1]


	// for matrix canonization:
	// int first_column_element;

	// subset to be chosen by classify_by_..._extract_subset():
	// used as input for split_cell()
		//
	// used if SPLIT_MULTIPLY is defined:
		int nb_subsets;
		int *subset_first; // [n + 1]
		int *subset_length; // [n + 1]
		int *subsets; // [n + 1]
		//
	// used if SPLIT_MULTIPLY is not defined:
		int *subset;  // [n + 1]
		int subset_size;

	partitionstack();
	~partitionstack();
	void free();
	partitionstack *copy(
			int verbose_level);
	void allocate(
			int n, int verbose_level);
	void allocate_with_two_classes(
			int n, int v, int b, int verbose_level);
	int parent_at_height(
			int h, int cell);
	int is_discrete();
	int smallest_non_discrete_cell();
	int biggest_non_discrete_cell();
	int smallest_non_discrete_cell_rows_preferred();
	int biggest_non_discrete_cell_rows_preferred();
	int nb_partition_classes(
			int from, int len);
	int is_subset_of_cell(
			int *set, int size, int &cell_idx);
	void sort_cells();
	void sort_cell(
			int cell);
	void reverse_cell(
			int cell);
	void check();
	void print_raw();
	void print_class(
			std::ostream& ost, int idx);
	void print_classes_tex(
			std::ostream& ost);
	void print_class_tex(
			std::ostream& ost, int idx);
	void print_class_point_or_line(
			std::ostream& ost, int idx);
	void print_classes(
			std::ostream& ost);
	void print_classes_points_and_lines(
			std::ostream& ost);
	std::ostream& print(
			std::ostream& ost);
	void print_cell(
			int i);
	void print_cell_latex(
			std::ostream &ost, int i);
	void print_subset();
	void get_cell(
			int i,
			int *&cell, int &cell_sz, int verbose_level);
	void get_cell_lint(
			int i,
			long int *&cell, int &cell_sz,
			int verbose_level);
	void write_cell_to_file(
			int i,
			std::string &fname, int verbose_level);
	void write_cell_to_file_points_or_lines(
			int i,
			std::string &fname, int verbose_level);
	void refine_arbitrary_set_lint(
			int size, long int *set, int verbose_level);
	void refine_arbitrary_set(
			int size, int *set, int verbose_level);
	void split_cell(
			int verbose_level);
	void split_multiple_cells(
			int *set, int set_size,
		int f_front, int verbose_level);
	void split_line_cell_front_or_back_lint(
			long int *set, int set_size, int f_front,
			int verbose_level);
	void split_line_cell_front_or_back(
			int *set, int set_size,
		int f_front, int verbose_level);
	void split_cell_front_or_back_lint(
			long int *set, int set_size, int f_front,
			int verbose_level);
	void split_cell_front_or_back(
			int *set, int set_size,
		int f_front, int verbose_level);
	void split_cell(
			int *set, int set_size, int verbose_level);
	void join_cell();
	void reduce_height(
			int ht0);
	void isolate_point(
			int pt);
	void subset_contiguous(
			int from, int len);
	int is_row_class(
			int c);
	int is_col_class(
			int c);
	void initial_matrix_decomposition(
			int nbrows, int nbcols,
		int *V, int nb_V, int *B, int nb_B, 
		int verbose_level);
	int is_descendant_of(
			int cell, int ancestor_cell,
		int verbose_level);
	int is_descendant_of_at_level(
			int cell, int ancestor_cell,
			int level, int verbose_level);
	int cellSizeAtLevel(
			int cell, int level);



	int hash_column_refinement_info(
			int ht0, int *data, int depth,
		int hash0);
	int hash_row_refinement_info(
			int ht0, int *data, int depth, int hash0);
	void print_column_refinement_info(
			int ht0, int *data, int depth);
	void print_row_refinement_info(
			int ht0, int *data, int depth);
	void radix_sort(
			int left, int right, int *C,
		int length, int radix, int verbose_level);
	void radix_sort_bits(
			int left, int right,
		int *C, int length, int radix, int mask,
		int verbose_level);
	void swap_ij(
			int *perm, int *perm_inv, int i, int j);
	void split_by_orbit_partition(
			int nb_orbits,
		int *orbit_first, int *orbit_len, int *orbit,
		int offset,
		int verbose_level);

	void get_row_and_col_permutation(
			combinatorics::tactical_decompositions::row_and_col_partition *RC,
		int *row_perm, int *row_perm_inv,
		int *col_perm, int *col_perm_inv,
		int verbose_level);
	void get_row_and_col_classes(
			combinatorics::tactical_decompositions::row_and_col_partition *&RC,
			int verbose_level);
	void get_row_and_col_classes_old_fashioned(
			int *row_classes, int &nb_row_classes,
			int *col_classes, int &nb_col_classes,
		int verbose_level);
	void get_row_classes(
			set_of_sets *&Sos, int verbose_level);
	void get_column_classes(
			set_of_sets *&Sos, int verbose_level);

};

// #############################################################################
// set_builder_description.cpp
// #############################################################################



//! to define a set of integers for class set_builder


class set_builder_description {
public:

	// TABLES/set_builder.tex

	int f_loop;
	int loop_low;
	int loop_upper_bound;
	int loop_increment;

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

	int f_file_orbiter_format;
	std::string file_orbiter_format_name;

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
	void init(
			set_builder_description *Descr, int verbose_level);
	long int process_transformations(
			long int x);
	long int clone_with_affine_function(
			long int x);
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
	void init_simple(
			long int underlying_set_size,
			int nb_sets, int verbose_level);
	void init(
			long int underlying_set_size,
			int nb_sets, long int **Pts, int *Sz, int verbose_level);
	void init_from_set_of_sets(
			set_of_sets *SoS, int verbose_level);
	void init_single(
			long int underlying_set_size,
			long int *Pts, int sz, int verbose_level);
	void init_basic(
			long int underlying_set_size,
			int nb_sets, int *Sz, int verbose_level);
	void init_set(
			int idx_of_set,
			long int *set, int sz, int verbose_level);
	std::string stringify_set(
			int idx);

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
	set_of_sets *copy();
	void init_simple(
			int underlying_set_size,
		int nb_sets, int verbose_level);
	void init_from_adjacency_matrix(
			int n, int *Adj,
		int verbose_level);
	void init(
			int underlying_set_size, int nb_sets,
		long int **Pts, long int *Sz, int verbose_level);
	void add_constant_everywhere(
			int c,
			int verbose_level);
	void init_with_Sz_in_int(
			int underlying_set_size,
			int nb_sets, long int **Pts, int *Sz,
			int verbose_level);
	void init_basic(
			int underlying_set_size,
		int nb_sets, long int *Sz, int verbose_level);
	void init_basic_with_Sz_in_int(
			int underlying_set_size,
			int nb_sets, int *Sz, int verbose_level);
	void init_basic_constant_size(
			int underlying_set_size,
		int nb_sets, int constant_size, int verbose_level);
	void init_single(
			int underlying_set_size,
			long int *Pts, int sz, int verbose_level);
	void init_from_file(
			int &underlying_set_size,
			std::string &fname,
			std::string &col_heading,
			int verbose_level);
	// two modes: one for reading csv files, one for reading inc files.
	void init_from_csv_file(
			int underlying_set_size,
			std::string &fname, int verbose_level);
	// outdated.
	// Better use Fio.Csv_file_support->read_column_as_set_of_sets instead
	void init_from_orbiter_file(
			int underlying_set_size,
			std::string &fname, int verbose_level);
	void init_set(
			int idx_of_set, int *set, int sz,
		int verbose_level);
		// Stores a copy of the given set.
	void init_cycle_structure(
			int *perm, int n, int verbose_level);
	int total_size();
	long int &element(
			int i, int j);
	void add_element(
			int i, long int a);
	void print();
	void print_table();
	void print_table_tex(
			std::ostream &ost);
	void print_table_latex_simple(
			std::ostream &ost);
	void print_table_latex_simple_with_selection(
			std::ostream &ost, int *Selection, int nb_sel);
	void dualize(
			set_of_sets *&S, int verbose_level);
	void remove_sets_of_given_size(
			int k,
		set_of_sets &S, int *&Idx, 
		int verbose_level);
	void extract_largest_sets(
			set_of_sets &S,
		int *&Idx, int verbose_level);
	void intersection_matrix(
		int *&intersection_type, int &highest_intersection_number, 
		int *&intersection_matrix, int &nb_big_sets, 
		int verbose_level);
	void compute_incidence_matrix(
			int *&Inc, int &m, int &n,
		int verbose_level);
	void init_decomposition(
			combinatorics::tactical_decompositions::decomposition *&D,
			int verbose_level);
	void compute_tdo_decomposition(
			combinatorics::tactical_decompositions::decomposition &D,
		int verbose_level);
	int is_member(
			int i, int a, int verbose_level);
	void sort_all(
			int verbose_level);
	void all_pairwise_intersections(
			set_of_sets *&Intersections,
		int verbose_level);
	void pairwise_intersection_matrix(
			int *&M, int verbose_level);
	void all_triple_intersections(
			set_of_sets *&Intersections,
		int verbose_level);
	int has_constant_size_property();
	int get_constant_size();
	int largest_set_size();
	void save_csv(
			std::string &fname,
		int verbose_level);
	void save_constant_size_csv(
			std::string &fname,
			int verbose_level);
	int find_common_element_in_two_sets(
			int idx1, int idx2,
		int &common_elt);
	void sort();
	void sort_big(
			int verbose_level);
	void compute_orbits(
			int &nb_orbits,
			int *&orbit, int *&orbit_inv,
		int *&orbit_first, int *&orbit_len, 
		void (*compute_image_function)(set_of_sets *S, 
			void *compute_image_data, int elt_idx, int gen_idx, 
			int &idx_of_image, int verbose_level), 
		void *compute_image_data, 
		int nb_gens, 
		int verbose_level);
	int number_of_eckardt_points(
			int verbose_level);
	void get_eckardt_points(
			int *&E, int &nb_E, int verbose_level);
	void evaluate_function_and_store(
			data_structures::set_of_sets *&Function_values,
			int (*evaluate_function)(int a, int i, int j,
					void *evaluate_data, int verbose_level),
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

	void int_vec_search_vec(
			int *v, int len, int *A, int A_sz, int *Idx);
	void lint_vec_search_vec(
			long int *v, int len,
			long int *A, int A_sz, long int *Idx);
	void int_vec_search_vec_linear(
			int *v, int len, int *A, int A_sz, int *Idx);
	void lint_vec_search_vec_linear(
			long int *v, int len,
			long int *A, int A_sz, long int *Idx);
	int int_vec_is_subset_of(
			int *set, int sz, int *big_set, int big_set_sz);
	int lint_vec_is_subset_of(
			int *set, int sz,
			long int *big_set, int big_set_sz, int verbose_level);
	int lint_vec_is_subset_of_lint_vec(
			long int *set, int sz,
			long int *big_set, int big_set_sz,
			int verbose_level);
	void int_vec_swap_points(
			int *list, int *list_inv, int idx1, int idx2);
	int int_vec_is_sorted(
			int *v, int len);
	void int_vec_sort_and_remove_duplicates(
			int *v, int &len);
	int lint_vec_is_sorted(
			long int *v, int len);
	void lint_vec_sort_and_remove_duplicates(
			long int *v, int &len);
	int int_vec_sort_and_test_if_contained(
			int *v1, int len1, int *v2, int len2);
	int lint_vec_sort_and_test_if_contained(
			long int *v1, int len1, long int *v2, int len2);
	int int_vecs_are_disjoint(
			int *v1, int len1, int *v2, int len2);
	int int_vecs_find_common_element(
			int *v1, int len1,
		int *v2, int len2, int &idx1, int &idx2);
	int lint_vecs_find_common_element(
			long int *v1, int len1,
		long int *v2, int len2, int &idx1, int &idx2);
	void int_vec_insert_and_reallocate_if_necessary(
			int *&vec,
		int &used_length, int &alloc_length, int a,
		int verbose_level);
	void int_vec_append_and_reallocate_if_necessary(
			int *&vec,
		int &used_length, int &alloc_length, int a,
		int verbose_level);
	void lint_vec_append_and_reallocate_if_necessary(
			long int *&vec,
			int &used_length, int &alloc_length, long int a,
			int verbose_level);
	int int_vec_is_zero(
			int *v, int len);
	int test_if_sets_are_equal(
			int *set1, int *set2, int set_size);
	int test_if_sets_are_disjoint(
			long int *set1, int sz1, long int *set2, int sz2);
	// This function copies the sets and sorts them. It also allocates temporary memory.
	// This is inefficient if the sets are already sorted!
	void test_if_set(
			int *set, int set_size);
	int lint_vec_test_if_set(
			long int *set, int set_size);
	int test_if_set_with_return_value(
			int *set, int set_size);
	int test_if_set_with_return_value_lint(
			long int *set, int set_size);
	void rearrange_subset(
			int n, int k, int *set,
		int *subset, int *rearranged_set, int verbose_level);
	void rearrange_subset_lint(
			int n, int k,
		long int *set, int *subset, long int *rearranged_set,
		int verbose_level);
	void rearrange_subset_lint_all(
			int n, int k,
		long int *set, long int *subset, long int *rearranged_set,
		int verbose_level);
	int int_vec_search_linear(
			int *v, int len, int a, int &idx);
	int lint_vec_search_linear(
			long int *v, int len, long int a, int &idx);
	void int_vec_intersect(
			int *v1, int len1, int *v2, int len2,
		int *&v3, int &len3);
	void vec_intersect(
			long int *v1, int len1,
		long int *v2, int len2, long int *&v3, int &len3);
	void int_vec_intersect_sorted_vectors(
			int *v1, int len1,
		int *v2, int len2, int *v3, int &len3);
	void lint_vec_intersect_sorted_vectors(
			long int *v1, int len1,
			long int *v2, int len2, long int *v3, int &len3);
	void int_vec_sorting_permutation(
			int *v, int len, int *perm,
		int *perm_inv, int f_increasingly);
	void lint_vec_sorting_permutation(
			long int *v, int len,
		int *perm, int *perm_inv, int f_increasingly);
	// perm and perm_inv must be allocated to len elements
	void int_vec_quicksort(
			int *v, int (*compare_func)(int a, int b),
		int left, int right);
	void lint_vec_quicksort(
			long int *v,
		int (*compare_func)(long int a, long int b), int left, int right);
	void int_vec_quicksort_increasingly(
			int *v, int len);
	void int_vec_quicksort_decreasingly(
			int *v, int len);
	void lint_vec_quicksort_increasingly(
			long int *v, int len);
	void lint_vec_quicksort_decreasingly(
			long int *v, int len);
	void quicksort_array(
			int len, void **v,
		int (*compare_func)(void *a, void *b, void *data), void *data);
	void quicksort_array_with_perm(
			int len, void **v, int *perm,
		int (*compare_func)(void *a, void *b, void *data), void *data);
	int vec_search(
			void **v, int (*compare_func)(void *a, void *b, void *data),
		void *data_for_compare,
		int len, void *a, int &idx, int verbose_level);
	int vec_search_general(
			void *vec,
		int (*compare_func)(void *vec,
				void *a, int b, void *data_for_compare),
		void *data_for_compare,
		int len, void *a, int &idx, int verbose_level);
	int int_vec_search_and_insert_if_necessary(
			int *v, int &len, int a);
	int int_vec_search_and_remove_if_found(
			int *v, int &len, int a);
	int int_vec_search(
			int *v, int len, int a, int &idx);
		// This function finds the last occurrence of the element a.
		// If a is not found, it returns in idx the position
		// where it should be inserted if
		// the vector is assumed to be in increasing order.
	int lint_vec_search(
			long int *v, int len, long int a,
			int &idx, int verbose_level);
		// This function finds the last occurrence of the element a.
		// If a is not found, it returns in idx the position
		// where it should be inserted if
		// the vector is assumed to be in increasing order.
	int vector_lint_search(
			std::vector<long int> &v,
			long int a, int &idx, int verbose_level);
	int int_vec_search_first_occurrence(
			int *v, int len, int a, int &idx,
			int verbose_level);
		// This function finds the first occurrence of the element a.
	int lint_vec_search_first_occurrence(
			long int *v,
			int len, long int a, int &idx,
			int verbose_level);
		// This function finds the first occurrence of the element a.
	int longinteger_vec_search(
			algebra::ring_theory::longinteger_object *v, int len,
			algebra::ring_theory::longinteger_object &a, int &idx);
	void int_vec_classify_and_print(
			std::ostream &ost, int *v, int l);
	void int_vec_values(
			int *v, int l, int *&w, int &w_len);
	void int_vec_multiplicities(
			int *v, int l, int *&w, int &w_len);
	void int_vec_values_and_multiplicities(
			int *v, int l,
		int *&val, int *&mult, int &nb_values);
	void int_vec_classify(
			int length, int *the_vec, int *&the_vec_sorted,
		int *&sorting_perm, int *&sorting_perm_inv,
		int &nb_types, int *&type_first, int *&type_len);
	void int_vec_classify_with_arrays(
			int length,
		int *the_vec, int *the_vec_sorted,
		int *sorting_perm, int *sorting_perm_inv,
		int &nb_types, int *type_first, int *type_len);
	void int_vec_sorted_collect_types(
			int length, int *the_vec_sorted,
		int &nb_types, int *type_first, int *type_len);
	void lint_vec_sorted_collect_types(
			int length,
		long int *the_vec_sorted,
		int &nb_types, int *type_first, int *type_len);
	void int_vec_print_classified(
			std::ostream &ost, int *vec, int len);
	void int_vec_print_types(
			std::ostream &ost,
		int f_backwards, int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void lint_vec_print_types(
			std::ostream &ost,
		int f_backwards, long int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void int_vec_print_types_bare_stringstream(
			std::stringstream &sstr,
		int f_backwards, int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void lint_vec_print_types_bare_stringstream(
			std::stringstream &sstr,
		int f_backwards, long int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void int_vec_print_types_bare(
			std::ostream &ost, int f_backwards,
		int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	std::string int_vec_stringify_types_bare(
			std::ostream &ost,
		int f_backwards, int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void lint_vec_print_types_bare(
			std::ostream &ost,
		int f_backwards, long int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void int_vec_print_types_bare_tex(
			std::ostream &ost, int f_backwards,
		int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void lint_vec_print_types_bare_tex(
			std::ostream &ost,
		int f_backwards, long int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void int_vec_print_types_bare_tex_we_are_in_math_mode(
			std::ostream &ost,
		int f_backwards, int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void lint_vec_print_types_bare_tex_we_are_in_math_mode(
			std::ostream &ost,
		int f_backwards, long int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void Heapsort(
			void *v, int len, int entry_size_in_chars,
		int (*compare_func)(void *v1, void *v2));
	void Heapsort_general(
			void *data, int len,
		int (*compare_func)(void *data, int i, int j, void *extra_data),
		void (*swap_func)(void *data, int i, int j, void *extra_data),
		void *extra_data);
	void Heapsort_general_with_log(
			void *data, int *w, int len,
		int (*compare_func)(void *data,
				int i, int j, void *extra_data),
		void (*swap_func)(void *data,
				int i, int j, void *extra_data),
		void *extra_data);
	int search_general(
			void *data, int len, void *search_object, int &idx,
		int (*compare_func)(void *data, int i, void *search_object,
		void *extra_data),
		void *extra_data, int verbose_level);
		// This function finds the last occurrence of the element a.
		// If a is not found, it returns in idx the position
		// where it should be inserted if
		// the vector is assumed to be in increasing order.
	void int_vec_heapsort(
			int *v, int len);
	void lint_vec_heapsort(
			long int *v, int len);
	void int_vec_heapsort_with_log(
			int *v, int *w, int len);
	void lint_vec_heapsort_with_log(
			long int *v, long int *w, int len);
	void heapsort_make_heap(
			int *v, int len);
	void lint_heapsort_make_heap(
			long int *v, int len);
	void heapsort_make_heap_with_log(
			int *v, int *w, int len);
	void lint_heapsort_make_heap_with_log(
			long int *v, long int *w, int len);
	void Heapsort_make_heap(
			void *v, int len, int entry_size_in_chars,
		int (*compare_func)(void *v1, void *v2));
	void Heapsort_general_make_heap(
			void *data, int len,
		int (*compare_func)(void *data, int i, int j, void *extra_data),
		void (*swap_func)(void *data, int i, int j, void *extra_data),
		void *extra_data);
	void Heapsort_general_make_heap_with_log(
			void *data, int *w, int len,
		int (*compare_func)(void *data, int i, int j, void *extra_data),
		void (*swap_func)(void *data, int i, int j, void *extra_data),
		void *extra_data);
	void heapsort_sift_down(
			int *v, int start, int end);
	void lint_heapsort_sift_down(
			long int *v, int start, int end);
	void heapsort_sift_down_with_log(
			int *v, int *w, int start, int end);
	void lint_heapsort_sift_down_with_log(
			long int *v, long int *w, int start, int end);
	void Heapsort_sift_down(
			void *v, int start, int end, int entry_size_in_chars,
		int (*compare_func)(void *v1, void *v2));
	void Heapsort_general_sift_down(
			void *data, int start, int end,
		int (*compare_func)(void *data, int i, int j, void *extra_data),
		void (*swap_func)(void *data, int i, int j, void *extra_data),
		void *extra_data);
	void Heapsort_general_sift_down_with_log(
			void *data, int *w, int start, int end,
		int (*compare_func)(void *data, int i, int j, void *extra_data),
		void (*swap_func)(void *data, int i, int j, void *extra_data),
		void *extra_data);
	void heapsort_swap(
			int *v, int i, int j);
	void lint_heapsort_swap(
			long int *v, int i, int j);
	void Heapsort_swap(
			void *v, int i, int j, int entry_size_in_chars);
	void find_points_by_multiplicity(
			int *data, int data_sz, int multiplicity,
		int *&pts, int &nb_pts);
	void int_vec_bubblesort_increasing(
			int len, int *p);
	int integer_vec_compare(
			int *p, int *q, int len);
	int integer_vec_std_compare(
			const std::vector<unsigned int> &p,
			const std::vector<unsigned int> &q);

	int compare_sets(
			int *set1, int *set2, int sz1, int sz2);
	int compare_sets_lint(
			long int *set1, long int *set2, int sz1, int sz2);
	int test_if_sets_are_disjoint(
			long int *set1, long int *set2, int sz1, int sz2);
	// Assumes that the sets are sorted.
	// Not to be confused with a function with the same name but different order of parameters above.
	void d_partition(
			double *v, int left, int right, int *middle);
	void d_quicksort(
			double *v, int left, int right);
	void d_quicksort_array(
			int len, double *v);
	int test_if_sets_are_disjoint_assuming_sorted(
			int *set1, int *set2, int sz1, int sz2);
	int test_if_sets_are_disjoint_assuming_sorted_lint(
			long int *set1, long int *set2, int sz1, int sz2);
	int uchar_vec_compare(
			uchar *p, uchar *q, int len);
	int test_if_sets_are_disjoint_not_assuming_sorted(
			long int *v, long int *w, int len);
	int int_vec_compare(
			int *p, int *q, int len);
	int lint_vec_compare(
			long int *p, long int *q, int len);
	int uint_vec_compare(
			unsigned int *p, unsigned int *q, int len);
	int int_vec_compare_stride(
			int *p, int *q, int len, int stride);
	void sorted_vec_get_first_and_length(
			int *v, int len,
			int *class_first, int *class_len, int &nb_classes);
	// we assume that the vector v is sorted.

};


// #############################################################################
// spreadsheet.cpp
// #############################################################################

//! for reading and writing of csv files


class spreadsheet {

public:
	int nb_rows, nb_cols;

private:
	char **tokens;
	int nb_tokens;

	int *line_start, *line_size;
	int nb_lines;

	int *Table;

public:

	spreadsheet();
	~spreadsheet();
	void init_set_of_sets(
			set_of_sets *S, int f_make_heading);
	void init_int_matrix(
			int nb_rows, int nb_cols, int *A);
	void init_empty_table(
			int nb_rows, int nb_cols);
	void fill_entry_with_text(
			int row_idx,
		int col_idx, const char *text);
	void fill_entry_with_text(
			int row_idx,
			int col_idx, std::string &text);
	void set_entry_lint(
			int row_idx,
			int col_idx, long int val);
	void fill_column_with_text(
			int col_idx, std::string *text,
		const char *heading);
	void fill_column_with_int(
			int col_idx, int *data,
		const char *heading);
	void fill_column_with_lint(
			int col_idx,
			long int *data, const char *heading);
	void fill_column_with_row_index(
			int col_idx,
		const char *heading);
	void add_token(
			const char *label);
	void save(
			std::string &fname, int verbose_level);
	void read_spreadsheet(
			std::string &fname, int verbose_level);
	void print_table(
			std::ostream &ost, int f_enclose_in_parentheses);
	void print_table_latex_all_columns(
			std::ostream &ost,
		int f_enclose_in_parentheses);
	void print_table_latex(
			std::ostream &ost,
			int *f_column_select,
			int f_enclose_in_parentheses,
			int nb_lines_per_table);
	void print_table_row(
			int row, int f_enclose_in_parentheses,
			std::ostream &ost);
	void print_table_row_latex(
			int row, int *f_column_select,
		int f_enclose_in_parentheses, std::ostream &ost);
	void print_table_row_detailed(
			int row, std::ostream &ost);
	void print_table_row_with_column_selection(
			int row,
			int f_enclose_in_parentheses,
			int *Col_selection, int nb_cols_selected,
			std::ostream &ost, int verbose_level);
	void print_table_with_row_selection(
			int *f_selected,
			std::ostream &ost);
	void print_table_sorted(
			std::ostream &ost, const char *sort_by);
	void add_column_with_constant_value(
			const char *label, char *value);
	void add_column_with_int(
			std::string &heading, int *Value);
	void add_column_with_text(
			std::string &heading, std::string *Value);
	void reallocate_table();
	void reallocate_table_add_row();
	int find_column(
			std::string &column_label);
	int find_by_column(
			const char *join_by);
	void tokenize(
			std::string &fname,
		char **&tokens, int &nb_tokens, int verbose_level);
	void remove_quotes(
			int verbose_level);
	void remove_rows(
			const char *drop_column, const char *drop_label,
		int verbose_level);
	void remove_rows_where_field_is_empty(
			const char *drop_column,
		int verbose_level);
	void find_rows(
			int verbose_level);
	void get_value_double_or_NA(
			int i, int j, double &val, int &f_NA);
	std::string get_entry_ij(
			int i, int j);
	void get_string(
			std::string &str, int i, int j);
	long int get_lint(
			int i, int j);
	double get_double(
			int i, int j);
	void join_with(
			spreadsheet *S2, int by1, int by2,
		int verbose_level);
	void patch_with(
			spreadsheet *S2, char *join_by);
	void compare_columns(
			std::string &col1_label, std::string &col2_label,
			int verbose_level);
	void stringify(
			std::string *&Header_rows, std::string *&Header_cols, std::string *&T,
			int &nb_r, int &nb_c,
			int verbose_level);
	void stringify_but_keep_first_column(
			std::string *&Header_rows, std::string *&Header_cols, std::string *&T,
			int &nb_r, int &nb_c,
			int verbose_level);


};


// #############################################################################
// string_tools.cpp
// #############################################################################

//! functions related to strings and character arrays


class string_tools {

public:

	string_tools();
	~string_tools();
	int is_csv_file(
			const char *fname);
	int is_inc_file(
			const char *fname);
	int is_xml_file(
			const char *fname);
	int s_scan_int(
			char **s, int *i);
	int s_scan_lint(
			char **s, long int *i);
	int s_scan_double(
			char **s, double *d);
	int s_scan_token(
			char **s, char *str);
	int s_scan_token_arbitrary(
			char **s, char *str);
	int s_scan_str(
			char **s, char *str);
	int s_scan_token_comma_separated(
			const char **s, char *str, int verbose_level);
	void scan_permutation_from_string(
			std::string &s,
		int *&perm, int &degree, int verbose_level);
	void scan_permutation_from_stream(
			std::istream & is,
		int *&perm, int &degree, int verbose_level);
	void chop_string(
			const char *str, int &argc, char **&argv);
	void chop_string_comma_separated(
			const char *str, int &argc, char **&argv);
	void convert_arguments(
			int &argc, const char **argv, std::string *&Argv);
	char get_character(
			std::istream & is, int verbose_level);
	void replace_extension_with(
			char *p, const char *new_ext);
	void replace_extension_with(
			std::string &p, const char *new_ext);
	void chop_off_extension(
			char *p);
	void chop_off_extension_and_path(
			std::string &p);
	void chop_off_extension(
			std::string &p);
	std::string without_extension(
			std::string &p);
	void chop_off_path(
			std::string &p);
	void chop_off_extension_if_present(
			std::string &p, const char *ext);
	void chop_off_extension_if_present(
			char *p, const char *ext);
	void get_fname_base(
			const char *p, char *fname_base);
	void get_extension(
			std::string &p, std::string &ext);
	void get_extension_if_present(
			const char *p, char *ext);
	void get_extension_if_present_and_chop_off(
			char *p, char *ext);
	void fix_escape_characters(
			std::string &str);
	void remove_specific_character(
			std::string &str, char c);
	void create_comma_separated_list(
			std::string &output, long int *input, int input_sz);
	void parse_comma_separated_list(
			std::string &input_text, std::vector<std::string> &output,
			int verbose_level);
	int is_all_whitespace(
			const char *str);
	void text_to_three_double(
			std::string &text, double *d);
	int strcmp_with_or_without(
			char *p, char *q);
	int starts_with_a_number(
			std::string &str);
	int compare_string_string(
			std::string &str1, std::string &str2);
	int stringcmp(
			std::string &str, const char *p);
	int strtoi(
			std::string &str);
	int str2int(
			std::string &str);
	long int strtolint(
			std::string &str);
	double strtof(
			std::string &str);
	void parse_value_pairs(
			std::map<std::string, std::string> &symbol_table,
			std::string &evaluate_text, int verbose_level);
	void parse_comma_separated_values(
			std::vector<std::string> &symbol_table,
			std::string &evaluate_text, int verbose_level);
	void drop_quotes(
			std::string &in, std::string &out);
	void parse_comma_separated_strings(
			std::string &in, std::vector<std::string> &out);
	int read_schlaefli_label(
			const char *p);
	void read_string_of_schlaefli_labels(
			std::string &str, int *&v, int &sz, int verbose_level);
	void name_of_group_projective(
			std::string &label_txt,
			std::string &label_tex,
			int n, int q, int f_semilinear, int f_special,
			int verbose_level);
	void name_of_group_affine(
			std::string &label_txt,
			std::string &label_tex,
			int n, int q, int f_semilinear, int f_special,
			int verbose_level);
	void name_of_group_general_linear(
			std::string &label_txt,
			std::string &label_tex,
			int n, int q, int f_semilinear, int f_special,
			int verbose_level);
	void name_of_orthogonal_group(
			std::string &label_txt,
			std::string &label_tex,
			int epsilon, int n, int q, int f_semilinear, int verbose_level);
	void name_of_orthogonal_space(
			std::string &label_txt,
			std::string &label_tex,
			int epsilon, int n, int q, int verbose_level);
	void name_of_BLT_set(
			std::string &label_txt,
			std::string &label_tex,
			int q, int ocn, int f_embedded, int verbose_level);
	void make_latex_friendly_vector(
			std::vector<std::string> &S, int verbose_level);
	void make_latex_friendly_string(
			std::string &in, std::string &out, int verbose_level);
	std::string printf_d(
			std::string &format, int value);
	std::string printf_dd(
			std::string &format, int value1, int value2);
	std::string printf_s(
			std::string &format, std::string &replacement);
	std::string printf_ss(
			std::string &format, std::string &replacement1, std::string &replacement2);
	void parse_RHS_command(
			std::string &command,
			int &mult, diophant_equation_type &type,
			int &data1, int &data2, int verbose_level);
	int find_string_in_array(
			std::string *String_array, int nb_strings,
			std::string &str_to_find, int &pos);
	std::string texify(
			std::string &input_string);


};

int string_tools_compare_strings(
		void *a, void *b, void *data);





// #############################################################################
// tally_lint.cpp
// #############################################################################


//! a statistical analysis of data consisting of long integers



class tally_lint {

public:

	int data_length;

	int f_data_ownership;
	long int *data;
	long int *data_sorted;
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

	tally_lint();
	~tally_lint();
	void init(
			long int *data, int data_length,
		int f_second, int verbose_level);
	void init_vector_lint(
			std::vector<long int> &data,
			int f_second, int verbose_level);
	void sort_and_classify();
	void sort_and_classify_second();
	int class_of(
			int pt_idx);
	void print(
			int f_backwards);
	void print_no_lf(
			int f_backwards);
	void print_tex_no_lf(
			int f_backwards);
	void print_first(
			int f_backwards);
	void print_second(
			int f_backwards);
	void print_first_tex(
			int f_backwards);
	void print_second_tex(
			int f_backwards);
	void print_file(
			std::ostream &ost, int f_backwards);
	void print_file_tex(
			std::ostream &ost, int f_backwards);
	void print_file_tex_we_are_in_math_mode(
			std::ostream &ost, int f_backwards);
	void print_bare_stringstream(
			std::stringstream &sstr, int f_backwards);
	void print_bare(
			int f_backwards);
	void print_bare_tex(
			std::ostream &ost, int f_backwards);
	void print_types_bare_tex(
			std::ostream &ost, int f_backwards,
		int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void print_lint_types_bare_tex(
		std::ostream &ost, int f_backwards, long int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void print_array_tex(
			std::ostream &ost, int f_backwards);
	double average();
	double average_of_non_zero_values();
	void get_data_by_multiplicity(
			int *&Pts, int &nb_pts,
		int multiplicity, int verbose_level);
	void get_data_by_multiplicity_as_lint(
			long int *&Pts, int &nb_pts,
			int multiplicity, int verbose_level);
	int determine_class_by_value(
			int value);
	int get_value_of_class(
			int class_idx);
	int get_largest_value();
	void get_class_by_value(
			int *&Pts, int &nb_pts, int value,
		int verbose_level);
	void get_class_by_value_lint(
			long int *&Pts, int &nb_pts, int value, int verbose_level);
	data_structures::set_of_sets *get_set_partition_and_types(
			int *&types,
		int &nb_types, int verbose_level);
	void save_classes_individually(
			std::string &fname);
};



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

	// added 09/28/2022:
	data_structures::set_of_sets *Set_partition;
	//data_structures::set_of_sets *Orbits_classified;

	int *data_values; // [nb_types]

	//int *Orbits_classified_length; // [Orbits_classified_nb_types]
	//int Orbits_classified_nb_types;



	tally();
	~tally();
	void init(
			int *data, int data_length,
		int f_second, int verbose_level);
	void print_types();
	void init_lint(
			long int *data, int data_length,
		int f_second, int verbose_level);
	void sort_and_classify();
	void sort_and_classify_second();
	int class_of(
			int pt_idx);
	void print(
			int f_backwards);
	void print_no_lf(
			int f_backwards);
	void print_tex_no_lf(
			int f_backwards);
	void print_first(
			int f_backwards);
	void print_second(
			int f_backwards);
	void print_first_tex(
			int f_backwards);
	void print_second_tex(
			int f_backwards);
	void print_file(
			std::ostream &ost, int f_backwards);
	void print_file_tex(
			std::ostream &ost, int f_backwards);
	void print_file_tex_we_are_in_math_mode(
			std::ostream &ost, int f_backwards);
	void print_bare_stringstream(
			std::stringstream &sstr, int f_backwards);
	void print_bare(
			int f_backwards);
	std::string stringify_bare(
			int f_backwards);
	void print_bare_tex(
			std::ostream &ost, int f_backwards);
	std::string stringify_bare_tex(
			int f_backwards);
	void print_types_bare_tex(
			std::ostream &ost, int f_backwards,
		int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	std::string stringify_types_bare_tex(
			int f_backwards, int *the_vec_sorted,
		int nb_types, int *type_first, int *type_len);
	void print_array_tex(
			std::ostream &ost, int f_backwards);
	double average();
	double average_of_non_zero_values();
	void get_data_by_multiplicity(
			int *&Pts, int &nb_pts,
		int multiplicity, int verbose_level);
	void get_data_by_multiplicity_as_lint(
			long int *&Pts, int &nb_pts,
			int multiplicity, int verbose_level);
	int determine_class_by_value(
			int value);
	int get_value_of_class(
			int class_idx);
	int get_largest_value();
	void get_class_by_value(
			int *&Pts, int &nb_pts, int value,
		int verbose_level);
	void get_class_by_value_lint(
			long int *&Pts, int &nb_pts, int value, int verbose_level);
	data_structures::set_of_sets *get_set_partition_and_types(
			int *&types,
		int &nb_types, int verbose_level);
	void save_classes_individually(
			std::string &fname);
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

	int *rep_idx;
		// [data_length],
		// rep_idx[i] is the index into Rep of data[i * data_set_sz]
	int *Reps;
		// [data_length * data_set_sz],
		// used [nb_types * data_set_sz]
	int *Frequency;
		// [data_length], used [nb_types]
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
	void init(
			int *data, int data_length, int data_set_sz,
		int verbose_level);
	int hash_and_find(
			int *data,
			int &idx, uint32_t &h, int verbose_level);
	void print();
	void save_classes_individually(
			std::string &fname, int verbose_level);
	void get_transversal(
			int *&transversal, int *&frequency,
			int &nb_types, int verbose_level);
	void print_classes_bigger_than_one(
			int verbose_level);
	data_structures::set_of_sets *get_set_partition(
			int verbose_level);

};


// #############################################################################
// text_builder_description.cpp
// #############################################################################



//! to define a text object


class text_builder_description {
public:

	int f_here;
	std::string here_text;

	// TABLES/text_builder.tex

	text_builder_description();
	~text_builder_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};



// #############################################################################
// text_builder.cpp
// #############################################################################



//! to create a text object from class text_builder_description


class text_builder {
public:

	text_builder_description *Descr;

	int f_has_text;
	std::string text;

	text_builder();
	~text_builder();
	void init(
			text_builder_description *Descr,
			int verbose_level);
	void print(
			std::ostream &ost);

};




// #############################################################################
// vector_builder_description.cpp
// #############################################################################



//! to define a vector of field elements


class vector_builder_description {
public:


	// TABLES/vector_builder.tex


	int f_field;
	std::string field_label;

	int f_allow_negatives;

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

	int f_file_column;
	std::string file_column_name;
	std::string file_column_label;

	int f_load_csv_no_border;
	std::string load_csv_no_border_fname;

	int f_load_csv_data_column;
	std::string load_csv_data_column_fname;
	int load_csv_data_column_idx;

	int f_sparse;
	int sparse_len;
	std::string sparse_pairs;

	int f_concatenate;
	std::string concatenate_list;

	int f_loop;
	int loop_start;
	int loop_upper_bound;
	int loop_increment;

	int f_index_of_support;
	std::string index_of_support_input;

	int f_permutation_matrix;
	std::string permutation_matrix_data;

	int f_permutation_matrix_inverse;
	std::string permutation_matrix_inverse_data;

	// for use inside code:
	int f_binary_data_lint;
	int binary_data_lint_sz;
	long int *binary_data_lint;

	// for use inside code:
	int f_binary_data_int;
	int binary_data_int_sz;
	int *binary_data_int;

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

	algebra::field_theory::finite_field *F;

	long int *v;
	int len;

	int f_has_k;
	int k;

	vector_builder();
	~vector_builder();
	void init(
			vector_builder_description *Descr,
			algebra::field_theory::finite_field *F,
			int verbose_level);
	void print(
			std::ostream &ost);

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
	void allocate(
			int data_size, int N, int bit_length);
	void compute_tables(
			int verbose_level);
	void print();
	int rank(
			int *data);
	void unrank(
			int rk, int *data);
};

}}}}




#endif /* ORBITER_SRC_LIB_FOUNDATIONS_DATA_STRUCTURES_DATA_STRUCTURES_H_ */



