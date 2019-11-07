// combinatorics.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005

namespace orbiter {
namespace foundations {


// #############################################################################
// brick_domain.cpp
// #############################################################################

//! a problem of Neil Sloane

class brick_domain {

public:
	finite_field *F;
	int q;
	int nb_bricks;

	brick_domain();
	~brick_domain();
	void null();
	void freeself();
	void init(finite_field *F, int verbose_level);
	void unrank(int rk, int &f_vertical, 
		int &x0, int &y0, int verbose_level);
	int rank(int f_vertical, int x0, int y0, int verbose_level);
	void unrank_coordinates(int rk, 
		int &x1, int &y1, int &x2, int &y2, 
		int verbose_level);
	int rank_coordinates(int x1, int y1, int x2, int y2, 
		int verbose_level);
};

void brick_test(int q, int verbose_level);

// #############################################################################
// combinatorics_domain.cpp
// #############################################################################

//! a collection of combinatorial functions

class combinatorics_domain {

public:
	combinatorics_domain();
	~combinatorics_domain();
	int Hamming_distance_binary(int a, int b, int n);
	int int_factorial(int a);
	int Kung_mue_i(int *part, int i, int m);
	void partition_dual(int *part, int *dual_part, int n, int verbose_level);
	void make_all_partitions_of_n(int n, int *&Table, int &nb, int verbose_level);
	int count_all_partitions_of_n(int n);
	int partition_first(int *v, int n);
	int partition_next(int *v, int n);
	void partition_print(std::ostream &ost, int *v, int n);
	int int_vec_is_regular_word(int *v, int len, int q);
		// Returns TRUE if the word v of length len is regular, i.~e.
		// lies in an orbit of length $len$ under the action of the cyclic group
		// $C_{len}$ acting on the coordinates.
		// Lueneburg~\cite{Lueneburg87a} p. 118.
		// v is a vector over $\{0, 1, \ldots , q-1\}$
	int int_vec_first_regular_word(int *v, int len, int Q, int q);
	int int_vec_next_regular_word(int *v, int len, int Q, int q);
	int is_subset_of(int *A, int sz_A, int *B, int sz_B);
	int set_find(int *elts, int size, int a);
	void set_complement(int *subset, int subset_size, int *complement,
		int &size_complement, int universal_set_size);
	void set_complement_lint(long int *subset, int subset_size, long int *complement,
		int &size_complement, int universal_set_size);
	void set_complement_safe(int *subset, int subset_size, int *complement,
		int &size_complement, int universal_set_size);
	// subset does not need to be in increasing order
	void set_add_elements(int *elts, int &size,
		int *elts_to_add, int nb_elts_to_add);
	void set_add_element(int *elts, int &size, int a);
	void set_delete_elements(int *elts, int &size,
		int *elts_to_delete, int nb_elts_to_delete);
	void set_delete_element(int *elts, int &size, int a);
	int compare_lexicographically(int a_len, long int *a, int b_len, long int *b);
	long int int_n_choose_k(int n, int k);
	void make_t_k_incidence_matrix(int v, int t, int k, int &m, int &n, int *&M,
		int verbose_level);
	void print_k_subsets_by_rank(std::ostream &ost, int v, int k);
	int f_is_subset_of(int v, int t, int k, int rk_t_subset, int rk_k_subset);
	int rank_subset(int *set, int sz, int n);
	void rank_subset_recursion(int *set, int sz, int n, int a0, int &r);
	void unrank_subset(int *set, int &sz, int n, int r);
	void unrank_subset_recursion(int *set, int &sz, int n, int a0, int &r);
	int rank_k_subset(int *set, int n, int k);
	void unrank_k_subset(int rk, int *set, int n, int k);
	int first_k_subset(int *set, int n, int k);
	int next_k_subset(int *set, int n, int k);
	int next_k_subset_at_level(int *set, int n, int k, int backtrack_level);
	void subset_permute_up_front(int n, int k, int *set, int *k_subset_idx,
		int *permuted_set);
	int ordered_pair_rank(int i, int j, int n);
	void ordered_pair_unrank(int rk, int &i, int &j, int n);
	void set_partition_4_into_2_unrank(int rk, int *v);
	int set_partition_4_into_2_rank(int *v);
	int unordered_triple_pair_rank(int i, int j, int k, int l, int m, int n);
	void unordered_triple_pair_unrank(int rk, int &i, int &j, int &k,
		int &l, int &m, int &n);
	long int ij2k_lint(long int i, long int j, long int n);
	void k2ij_lint(long int k, long int & i, long int & j, long int n);
	int ij2k(int i, int j, int n);
	void k2ij(int k, int & i, int & j, int n);
	int ijk2h(int i, int j, int k, int n);
	void h2ijk(int h, int &i, int &j, int &k, int n);
	void random_permutation(int *random_permutation, int n);
	void perm_move(int *from, int *to, int n);
	void perm_identity(int *a, int n);
	int perm_is_identity(int *a, int n);
	void perm_elementary_transposition(int *a, int n, int f);
	void perm_mult(int *a, int *b, int *c, int n);
	void perm_conjugate(int *a, int *b, int *c, int n);
	// c := a^b = b^-1 * a * b
	void perm_inverse(int *a, int *b, int n);
	// b := a^-1
	void perm_raise(int *a, int *b, int e, int n);
	// b := a^e (e >= 0)
	void perm_direct_product(int n1, int n2, int *perm1, int *perm2, int *perm3);
	void perm_print_list(std::ostream &ost, int *a, int n);
	void perm_print_list_offset(std::ostream &ost, int *a, int n, int offset);
	void perm_print_product_action(std::ostream &ost, int *a, int m_plus_n, int m,
		int offset, int f_cycle_length);
	void perm_print(std::ostream &ost, int *a, int n);
	void perm_print_with_print_point_function(
			std::ostream &ost,
			int *a, int n,
			void (*point_label)(std::stringstream &sstr, long int pt, void *data),
			void *point_label_data);
	void perm_print_with_cycle_length(std::ostream &ost, int *a, int n);
	void perm_print_counting_from_one(std::ostream &ost, int *a, int n);
	void perm_print_offset(std::ostream &ost,
		int *a, int n,
		int offset,
		int f_print_cycles_of_length_one,
		int f_cycle_length,
		int f_max_cycle_length,
		int max_cycle_length,
		int f_orbit_structure,
		void (*point_label)(std::stringstream &sstr, long int pt, void *data),
		void *point_label_data);
	void perm_cycle_type(int *perm, int degree, int *cycles, int &nb_cycles);
	int perm_order(int *a, int n);
	int perm_signum(int *perm, int n);
	int is_permutation(int *perm, int n);
	void first_lehmercode(int n, int *v);
	int next_lehmercode(int n, int *v);
	void lehmercode_to_permutation(int n, int *code, int *perm);
	int disjoint_binary_representation(int u, int v);
	int hall_test(int *A, int n, int kmax, int *memo, int verbose_level);
	int philip_hall_test(int *A, int n, int k, int *memo, int verbose_level);
	// memo points to free memory of n int's
	int philip_hall_test_dual(int *A, int n, int k, int *memo, int verbose_level);
	// memo points to free memory of n int's
	void print_01_matrix_with_stars(std::ostream &ost, int *A, int m, int n);
	void print_int_matrix(std::ostream &ost, int *A, int m, int n);
	int create_roots_H4(finite_field *F, int *roots);
	long int generalized_binomial(int n, int k, int q);
	void print_tableau(int *Tableau, int l1, int l2,
		int *row_parts, int *col_parts);
	int ijk_rank(int i, int j, int k, int n);
	void ijk_unrank(int &i, int &j, int &k, int n, int rk);
	int largest_binomial2_below(int a2);
	int largest_binomial3_below(int a3);
	int binomial2(int a);
	int binomial3(int a);
	int minus_one_if_positive(int i);
	//int int_ij2k(int i, int j, int n);
	//void int_k2ij(int k, int & i, int & j, int n);
	void compute_adjacency_matrix(
			int *Table, int nb_sets, int set_size,
			const char *prefix_for_graph,
			uchar *&bitvector_adjacency,
			int &bitvector_length,
			int verbose_level);
};

// combinatorics.cpp:
long int callback_ij2k(long int i, long int j, int n);


// #############################################################################
// geo_parameter.cpp
// #############################################################################


#define MODE_UNDEFINED 0
#define MODE_SINGLE 1
#define MODE_STACK 2

#define UNKNOWNTYPE 0
#define POINTTACTICAL 1
#define BLOCKTACTICAL 2
#define POINTANDBLOCKTACTICAL 3

#define FUSE_TYPE_NONE 0
#define FUSE_TYPE_SIMPLE 1
#define FUSE_TYPE_DOUBLE 2
//#define FUSE_TYPE_MULTI 3
//#define FUSE_TYPE_TDO 4

//! decomposition stack of a linear space or incidence geometry



class geo_parameter {
public:
	int decomposition_type;
	int fuse_type;
	int v, b;

	int mode;
	char label[1000];

	// for MODE_SINGLE
	int nb_V, nb_B;
	int *V, *B;
	int *scheme;
	int *fuse;

	// for MODE_STACK
	int nb_parts, nb_entries;

	int *part;
	int *entries;
	int part_nb_alloc;
	int entries_nb_alloc;


	//vector<int> part;
	//vector<int> entries;

	int lambda_level;
	int row_level, col_level;
	int extra_row_level, extra_col_level;

	geo_parameter();
	~geo_parameter();
	void append_to_part(int a);
	void append_to_entries(int a1, int a2, int a3, int a4);
	void write(std::ofstream &aStream, char *label);
	void write_mode_single(std::ofstream &aStream, char *label);
	void write_mode_stack(std::ofstream &aStream, char *label);
	void convert_single_to_stack(int verbose_level);
	int partition_number_row(int row_idx);
	int partition_number_col(int col_idx);
	int input(std::ifstream &aStream);
	int input_mode_single(std::ifstream &aStream);
	int input_mode_stack(std::ifstream &aStream, int verbose_level);
	void init_tdo_scheme(tdo_scheme &G, int verbose_level);
	void print_schemes(tdo_scheme &G);
	void print_schemes_tex(tdo_scheme &G);
	void print_scheme_tex(std::ostream &ost, tdo_scheme &G, int h);
	void print_C_source();
	void convert_single_to_stack_fuse_simple_pt(int verbose_level);
	void convert_single_to_stack_fuse_simple_bt(int verbose_level);
	void convert_single_to_stack_fuse_double_pt(int verbose_level);
	void cut_off_two_lines(geo_parameter &GP2,
		int *&part_relabel, int *&part_length,
		int verbose_level);
	void cut_off(geo_parameter &GP2, int w,
		int *&part_relabel, int *&part_length,
		int verbose_level);
	void copy(geo_parameter &GP2);
	void print_schemes();
};

void int_vec_classify(int *v, int len,
		int *class_first, int *class_len, int &nb_classes);
int tdo_scheme_get_row_class_length_fused(tdo_scheme &G,
		int h, int class_first, int class_len);
int tdo_scheme_get_col_class_length_fused(tdo_scheme &G,
		int h, int class_first, int class_len);

// #############################################################################
// tdo_data.cpp TDO parameter refinement
// #############################################################################

//! a class related to the class tdo_scheme

class tdo_data {
public:
	int *types_first;
	int *types_len;
	int *only_one_type;
	int nb_only_one_type;
	int *multiple_types;
	int nb_multiple_types;
	int *types_first2;
	diophant *D1;
	diophant *D2;

	tdo_data();
	~tdo_data();
	void free();
	void allocate(int R);
	int solve_first_system(int verbose_level,
		int *&line_types, int &nb_line_types,
		int &line_types_allocated);
	void solve_second_system_omit(int verbose_level,
		int *classes_len,
		int *&line_types, int &nb_line_types,
		int *&distributions, int &nb_distributions,
		int omit);
	void solve_second_system_with_help(int verbose_level,
		int f_use_mckay_solver, int f_once,
		int *classes_len, int f_scale, int scaling,
		int *&line_types, int &nb_line_types,
		int *&distributions, int &nb_distributions,
		int cnt_second_system, solution_file_data *Sol);
	void solve_second_system_from_file(int verbose_level,
		int *classes_len, int f_scale, int scaling,
		int *&line_types, int &nb_line_types,
		int *&distributions, int &nb_distributions,
		char *solution_file_name);
	void solve_second_system(int verbose_level,
		int f_use_mckay_solver, int f_once,
		int *classes_len, int f_scale, int scaling,
		int *&line_types, int &nb_line_types,
		int *&distributions, int &nb_distributions);
};


// #############################################################################
// tdo_refinement.cpp
// #############################################################################



//! refinement of the parameters of a linear space

class tdo_refinement {
	public:

	int t0;
	int cnt;
	char *p_buf;
	char str[1000];
	char ext[1000];
	char *fname_in;
	char fname_out[1000];
	int f_lambda3;
	int lambda3, block_size;
	int f_scale;
	int scaling;
	int f_range;
	int range_first, range_len;
	int f_select;
	char *select_label;
	int f_omit1;
	int omit1;
	int f_omit2;
	int omit2;
	int f_D1_upper_bound_x0;
	int D1_upper_bound_x0;
	int f_reverse;
	int f_reverse_inverse;
	int f_use_packing_numbers;
	int f_dual_is_linear_space;
	int f_do_the_geometric_test;
	int f_once;
	int f_use_mckay_solver;


	geo_parameter GP;

	geo_parameter GP2;



	int f_doit;
	int nb_written, nb_written_tactical, nb_tactical;
	int cnt_second_system;
	solution_file_data *Sol;

	tdo_refinement();
	~tdo_refinement();
	void init(int verbose_level);
	void read_arguments(int argc, char **argv);
	void main_loop(int verbose_level);
	void do_it(std::ofstream &g, int verbose_level);
	void do_row_refinement(std::ofstream &g, tdo_scheme &G, partitionstack &P, int verbose_level);
	void do_col_refinement(std::ofstream &g, tdo_scheme &G, partitionstack &P, int verbose_level);
	void do_all_row_refinements(char *label_in, std::ofstream &g, tdo_scheme &G,
		int *point_types, int nb_point_types, int point_type_len,
		int *distributions, int nb_distributions, int &nb_tactical, int verbose_level);
	void do_all_column_refinements(char *label_in, std::ofstream &g, tdo_scheme &G,
		int *line_types, int nb_line_types, int line_type_len,
		int *distributions, int nb_distributions, int &nb_tactical, int verbose_level);
	int do_row_refinement(int t, char *label_in, std::ofstream &g, tdo_scheme &G,
		int *point_types, int nb_point_types, int point_type_len,
		int *distributions, int nb_distributions, int verbose_level);
		// returns TRUE or FALSE depending on whether the
		// refinement gave a tactical decomposition
	int do_column_refinement(int t, char *label_in, std::ofstream &g, tdo_scheme &G,
		int *line_types, int nb_line_types, int line_type_len,
		int *distributions, int nb_distributions, int verbose_level);
		// returns TRUE or FALSE depending on whether the
		// refinement gave a tactical decomposition
};

void print_distribution(std::ostream &ost,
	int *types, int nb_types, int type_len,
	int *distributions, int nb_distributions);
int compare_func_int_vec(void *a, void *b, void *data);
int compare_func_int_vec_inverse(void *a, void *b, void *data);
void distribution_reverse_sorting(int f_increasing,
	int *types, int nb_types, int type_len,
	int *distributions, int nb_distributions);



// #############################################################################
// tdo_scheme.cpp
// #############################################################################


#define MAX_SOLUTION_FILE 100

#define NUMBER_OF_SCHEMES 5
#define ROW 0
#define COL 1
#define LAMBDA 2
#define EXTRA_ROW 3
#define EXTRA_COL 4


//! internal class related to tdo_data


struct solution_file_data {
	int nb_solution_files;
	int system_no[MAX_SOLUTION_FILE];
	char *solution_file[MAX_SOLUTION_FILE];
};

//! canonical tactical decomposition of an incidence structure

class tdo_scheme {

public:

	// the following is needed by the TDO process:
	// allocated in init_partition_stack
	// freed in exit_partition_stack

	//partition_backtrack PB;

	partitionstack *P;

	int part_length;
	int *part;
	int nb_entries;
	int *entries;
	int row_level;
	int col_level;
	int lambda_level;
	int extra_row_level;
	int extra_col_level;

	int mn; // m + n
	int m; // # of rows
	int n; // # of columns

	int level[NUMBER_OF_SCHEMES];
	int *row_classes[NUMBER_OF_SCHEMES], nb_row_classes[NUMBER_OF_SCHEMES];
	int *col_classes[NUMBER_OF_SCHEMES], nb_col_classes[NUMBER_OF_SCHEMES];
	int *row_class_index[NUMBER_OF_SCHEMES];
	int *col_class_index[NUMBER_OF_SCHEMES];
	int *row_classes_first[NUMBER_OF_SCHEMES];
	int *row_classes_len[NUMBER_OF_SCHEMES];
	int *row_class_no[NUMBER_OF_SCHEMES];
	int *col_classes_first[NUMBER_OF_SCHEMES];
	int *col_classes_len[NUMBER_OF_SCHEMES];
	int *col_class_no[NUMBER_OF_SCHEMES];

	int *the_row_scheme;
	int *the_col_scheme;
	int *the_extra_row_scheme;
	int *the_extra_col_scheme;
	int *the_row_scheme_cur; // [m * nb_col_classes[ROW]]
	int *the_col_scheme_cur; // [n * nb_row_classes[COL]]
	int *the_extra_row_scheme_cur; // [m * nb_col_classes[EXTRA_ROW]]
	int *the_extra_col_scheme_cur; // [n * nb_row_classes[EXTRA_COL]]

	// end of TDO process data

	tdo_scheme();
	~tdo_scheme();

	void init_part_and_entries(
			int *part, int *entries, int verbose_level);
	void init_part_and_entries_int(
			int *part, int *entries, int verbose_level);
	void init_TDO(int *Part, int *Entries,
		int Row_level, int Col_level,
		int Extra_row_level, int Extra_col_level,
		int Lambda_level, int verbose_level);
	void exit_TDO();
	void init_partition_stack(int verbose_level);
	void exit_partition_stack();
	void get_partition(int h, int l, int verbose_level);
	void free_partition(int h);
	void complete_partition_info(int h, int verbose_level);
	void get_row_or_col_scheme(int h, int l, int verbose_level);
	void get_column_split_partition(int verbose_level, partitionstack &P);
	void get_row_split_partition(int verbose_level, partitionstack &P);
	void print_all_schemes();
	void print_scheme(int h, int f_v);
	void print_scheme_tex(std::ostream &ost, int h);
	void print_scheme_tex_fancy(std::ostream &ost,
			int h, int f_label, char *label);
	void compute_whether_first_inc_must_be_moved(
			int *f_first_inc_must_be_moved, int verbose_level);
	int count_nb_inc_from_row_scheme(int verbose_level);
	int count_nb_inc_from_extra_row_scheme(int verbose_level);


	int geometric_test_for_row_scheme(partitionstack &P,
		int *point_types, int nb_point_types, int point_type_len,
		int *distributions, int nb_distributions,
		int f_omit1, int omit1, int verbose_level);
	int geometric_test_for_row_scheme_level_s(partitionstack &P, int s,
		int *point_types, int nb_point_types, int point_type_len,
		int *distribution,
		int *non_zero_blocks, int nb_non_zero_blocks,
		int f_omit1, int omit1,
		int verbose_level);


	int refine_rows(int verbose_level,
		int f_use_mckay, int f_once,
		partitionstack &P,
		int *&point_types, int &nb_point_types, int &point_type_len,
		int *&distributions, int &nb_distributions,
		int &cnt_second_system, solution_file_data *Sol,
		int f_omit1, int omit1, int f_omit2, int omit2,
		int f_use_packing_numbers,
		int f_dual_is_linear_space,
		int f_do_the_geometric_test);
	int refine_rows_easy(int verbose_level,
		int *&point_types, int &nb_point_types, int &point_type_len,
		int *&distributions, int &nb_distributions,
		int &cnt_second_system);
	int refine_rows_hard(partitionstack &P, int verbose_level,
		int f_use_mckay, int f_once,
		int *&point_types, int &nb_point_types, int &point_type_len,
		int *&distributions, int &nb_distributions,
		int &cnt_second_system,
		int f_omit1, int omit1, int f_omit, int omit,
		int f_use_packing_numbers, int f_dual_is_linear_space);
	void row_refinement_L1_L2(partitionstack &P, int f_omit, int omit,
		int &L1, int &L2, int verbose_level);
	int tdo_rows_setup_first_system(int verbose_level,
		tdo_data &T, int r, partitionstack &P,
		int f_omit, int omit,
		int *&point_types, int &nb_point_types);
	int tdo_rows_setup_second_system(int verbose_level,
		tdo_data &T, partitionstack &P,
		int f_omit, int omit,
		int f_use_packing_numbers,
		int f_dual_is_linear_space,
		int *&point_types, int &nb_point_types);
	int tdo_rows_setup_second_system_eqns_joining(int verbose_level,
		tdo_data &T, partitionstack &P,
		int f_omit, int omit, int f_dual_is_linear_space,
		int *point_types, int nb_point_types,
		int eqn_offset);
	int tdo_rows_setup_second_system_eqns_counting(int verbose_level,
		tdo_data &T, partitionstack &P,
		int f_omit, int omit,
		int *point_types, int nb_point_types,
		int eqn_offset);
	int tdo_rows_setup_second_system_eqns_packing(int verbose_level,
		tdo_data &T, partitionstack &P,
		int f_omit, int omit,
		int *point_types, int nb_point_types,
		int eqn_start, int &nb_eqns_used);

	int refine_columns(int verbose_level, int f_once, partitionstack &P,
		int *&line_types, int &nb_line_types, int &line_type_len,
		int *&distributions, int &nb_distributions,
		int &cnt_second_system, solution_file_data *Sol,
		int f_omit1, int omit1, int f_omit, int omit,
		int f_D1_upper_bound_x0, int D1_upper_bound_x0,
		int f_use_mckay_solver,
		int f_use_packing_numbers);
	int refine_cols_hard(partitionstack &P,
		int verbose_level, int f_once,
		int *&line_types, int &nb_line_types, int &line_type_len,
		int *&distributions, int &nb_distributions,
		int &cnt_second_system, solution_file_data *Sol,
		int f_omit1, int omit1, int f_omit, int omit,
		int f_D1_upper_bound_x0, int D1_upper_bound_x0,
		int f_use_mckay_solver,
		int f_use_packing_numbers);
	void column_refinement_L1_L2(partitionstack &P,
		int f_omit, int omit,
		int &L1, int &L2, int verbose_level);
	int tdo_columns_setup_first_system(int verbose_level,
		tdo_data &T, int r, partitionstack &P,
		int f_omit, int omit,
		int *&line_types, int &nb_line_types);
	int tdo_columns_setup_second_system(int verbose_level,
		tdo_data &T, partitionstack &P,
		int f_omit, int omit,
		int f_use_packing_numbers,
		int *&line_types, int &nb_line_types);
	int tdo_columns_setup_second_system_eqns_joining(int verbose_level,
		tdo_data &T, partitionstack &P,
		int f_omit, int omit,
		int *line_types, int nb_line_types,
		int eqn_start);
	void tdo_columns_setup_second_system_eqns_counting(int verbose_level,
		tdo_data &T, partitionstack &P,
		int f_omit, int omit,
		int *line_types, int nb_line_types,
		int eqn_start);
	int tdo_columns_setup_second_system_eqns_upper_bound(int verbose_level,
		tdo_data &T, partitionstack &P,
		int f_omit, int omit,
		int *line_types, int nb_line_types,
		int eqn_start, int &nb_eqns_used);


	int td3_refine_rows(int verbose_level, int f_once,
		int lambda3, int block_size,
		int *&point_types, int &nb_point_types, int &point_type_len,
		int *&distributions, int &nb_distributions);
	int td3_rows_setup_first_system(int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T, int r, partitionstack &P,
		int &nb_vars,int &nb_eqns,
		int *&point_types, int &nb_point_types);
	int td3_rows_setup_second_system(int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T,
		int nb_vars, int &Nb_vars, int &Nb_eqns,
		int *&point_types, int &nb_point_types);
	int td3_rows_counting_flags(int verbose_level,
		int lambda3, int block_size, int lambda2, int &S,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&point_types, int &nb_point_types, int eqn_offset);
	int td3_refine_columns(int verbose_level, int f_once,
		int lambda3, int block_size, int f_scale, int scaling,
		int *&line_types, int &nb_line_types, int &line_type_len,
		int *&distributions, int &nb_distributions);
	int td3_columns_setup_first_system(int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T, int r, partitionstack &P,
		int &nb_vars, int &nb_eqns,
		int *&line_types, int &nb_line_types);
	int td3_columns_setup_second_system(int verbose_level,
		int lambda3, int block_size, int lambda2, int f_scale, int scaling,
		tdo_data &T,
		int nb_vars, int &Nb_vars, int &Nb_eqns,
		int *&line_types, int &nb_line_types);
	int td3_columns_triples_same_class(int verbose_level,
		int lambda3, int block_size,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&line_types, int &nb_line_types, int eqn_offset);
	int td3_columns_pairs_same_class(int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&line_types, int &nb_line_types, int eqn_offset);
	int td3_columns_counting_flags(int verbose_level,
		int lambda3, int block_size, int lambda2, int &S,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&line_types, int &nb_line_types, int eqn_offset);
	int td3_columns_lambda2_joining_pairs_from_different_classes(
		int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&line_types, int &nb_line_types, int eqn_offset);
	int td3_columns_lambda3_joining_triples_2_1(int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&line_types, int &nb_line_types, int eqn_offset);
	int td3_columns_lambda3_joining_triples_1_1_1(int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&line_types, int &nb_line_types, int eqn_offset);


};







}}
