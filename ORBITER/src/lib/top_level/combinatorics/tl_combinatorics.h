/*
 * tl_combinatorics.h
 *
 *  Created on: Oct 27, 2019
 *      Author: betten
 */

#ifndef ORBITER_SRC_LIB_TOP_LEVEL_COMBINATORICS_TL_COMBINATORICS_H_
#define ORBITER_SRC_LIB_TOP_LEVEL_COMBINATORICS_TL_COMBINATORICS_H_


namespace orbiter {
namespace top_level {



// #############################################################################
// cayley_graph_search.cpp
// #############################################################################

//! for a problem of Ferdinand Ihringer


class cayley_graph_search {

public:

	int level;
	int group;
	int subgroup;

	int ord;
	int degree;
	int data_size;

	long int go;
	int go_subgroup;
	int nb_involutions;
	int *f_has_order2;
	int *f_subgroup; // [go]
	int *list_of_elements; // [go]
	int *list_of_elements_inverse; // [go]

	action *A;
	finite_field *F;
	int target_depth;

	int *Elt1;
	int *Elt2;
	vector_ge *gens;
	vector_ge *gens_subgroup;
	longinteger_object target_go, target_go_subgroup;
	strong_generators *Strong_gens;
	strong_generators *Strong_gens_subgroup;

	sims *S;
	sims *S_subgroup;

	int *Table;
	int *generators;
	int nb_generators;

	char fname_base[1000];
	char prefix[1000];
	char fname[1000];
	char fname_graphs[1000];

	strong_generators *Aut_gens;
	longinteger_object Aut_order;
	action *Aut;
	action *A2;
	poset *Poset;
	poset_classification *gen;


	void init(int level, int group, int subgroup, int verbose_level);
	void init_group(int verbose_level);
	void init_group2(int verbose_level);
	void init_group_level_3(int verbose_level);
	void init_group_level_4(int verbose_level);
	void init_group_level_5(int verbose_level);
	int incremental_check_func(int len, long int *S, int verbose_level);
	void classify_subsets(int verbose_level);
	void write_file(int verbose_level);
	void create_Adjacency_list(long int *Adj,
		long int *connection_set, int connection_set_sz,
		int verbose_level);
	// Adj[go * connection_set_sz]
	void create_additional_edges(
		long int *Additional_neighbor,
		int *Additional_neighbor_sz,
		long int connection_element,
		int verbose_level);
	// Additional_neighbor[go], Additional_neighbor_sz[go]

};



// #############################################################################
// delandtsheer_doyen.cpp
// #############################################################################



#define MAX_MASK_TESTS 1000



//! searching for line transitive point imprimitive designs preserving a grid structure on points


class delandtsheer_doyen {
public:
	int argc;
	const char **argv;

	int d1;
	int d2;
	int q1;
	int q2;

	const char *group_label;

	finite_field *F1;
	finite_field *F2;


	int DELANDTSHEER_DOYEN_X;
	int DELANDTSHEER_DOYEN_Y;
	int K;

	int Xsize; // = D = q1 = # of rows
	int Ysize; // = C = q2 = # of cols

	int V; // = Xsize * Ysize
	int b;
	long int *line;        // [K];
	int *row_sum; // [Xsize]
	int *col_sum; // [Ysize]


	matrix_group *M1;
	matrix_group *M2;
	action *A1;
	action *A2;

	action *A;
	action *A0;

	strong_generators *SG;
	longinteger_object go;
	direct_product *P;
	poset *Poset_pairs;
	poset *Poset_search;
	poset_classification *Pairs;
	poset_classification *Gen;

	// orbits on pairs:
	int f_subgroup;
	int *pair_orbit; // [V * V]
	int nb_orbits;
	int *transporter;
	int *tmp_Elt;
	int *orbit_length; 		// [nb_orbits]
	int *orbit_covered; 		// [nb_orbits]
	int *orbit_covered_max; 	// [nb_orbits]
		// orbit_covered_max[i] = orbit_length[i] / b;
	int *orbits_covered; 		// [K * K]


	// intersection type tests:

	int inner_pairs_in_rows;
	int inner_pairs_in_cols;

	// row intersection type
	int f_R;
	int nb_row_types;
	int *row_type;     		// [nb_row_types + 1]
	int *row_type_cur; 		// [nb_row_types + 1]
	int *row_type_this_or_bigger; 	// [nb_row_types + 1]

	// col intersection type
	int f_C;
	int nb_col_types;
	int *col_type;     		// [nb_col_types + 1]
	int *col_type_cur; 		// [nb_col_types + 1]
	int *col_type_this_or_bigger; 	// [nb_col_types + 1]


	// mask related test:
	int nb_mask_tests;
	int mask_test_level[MAX_MASK_TESTS];
	int mask_test_who[MAX_MASK_TESTS];
		// 1 = x
		// 2 = y
		// 3 = x+y
		// 4 = singletons
	int mask_test_what[MAX_MASK_TESTS];
		// 1 = eq
		// 2 = ge
		// 3 = le
	int mask_test_value[MAX_MASK_TESTS];

	// a file where we print the solution, it has the extension bblt
	// for "base block line transitive" design
	//FILE *fp_sol;
	//char fname_solution_file[1000];

	// for testing the mask:
	int *f_row_used; // [Xsize];
	int *f_col_used; // [Ysize];
	int *row_idx; // [Xsize];
	int *col_idx; // [Ysize];
	int *singletons; // [K];

	// temporary data
	int *row_col_idx; // [Xsize];
	int *col_row_idx; // [Ysize];

	long int *live_points; // [V]
	int nb_live_points;

	delandtsheer_doyen();
	~delandtsheer_doyen();
	void init(int argc, const char **argv,
			int d1, int q1, int d2, int q2,
			int f_subgroup, const char *subgroup_gens_text,
			const char *subgroup_order_text,
			const char *group_label,
			int depth,
			int verbose_level);
	void create_graph(
			long int *line0, int len, int verbose_level);
	void testing(strong_generators *SG, int verbose_level);
	int find_pair_orbit(int i, int j, int verbose_level);
	int find_pair_orbit_by_tracing(int i, int j, int verbose_level);
	void compute_pair_orbit_table(int verbose_level);
	void write_pair_orbit_file(int verbose_level);
	void print_mask_test_i(std::ostream &ost, int i);
	int check_conditions(long int *S, int len, int verbose_level);
	int check_orbit_covering(long int *line,
			int len, int verbose_level);
	int check_row_sums(long int *line,
			int len, int verbose_level);
	int check_col_sums(long int *line,
			int len, int verbose_level);
	int check_mask(long int *line,
			int len, int verbose_level);
	void get_mask_core_and_singletons(
		long int *line, int len,
		int &nb_rows_used, int &nb_cols_used,
		int &nb_singletons, int verbose_level);
};


int delandtsheer_doyen_check_conditions(int len, long int *S,
		void *data, int verbose_level);


// #############################################################################
// design_create_description.cpp
// #############################################################################

//! to describe the construction of a known design from the command line



class design_create_description {

public:

	int f_q;
	int q;
	//int f_k;
	//int k;
	int f_catalogue;
	int iso;
	int f_family;
	const char *family_name;



	design_create_description();
	~design_create_description();
	void null();
	void freeself();
	int read_arguments(int argc, const char **argv,
		int verbose_level);
	int get_q();
};



// #############################################################################
// design_create.cpp
// #############################################################################

//! to create a known design using a description from class design_create_description



class design_create {

public:
	design_create_description *Descr;

	char prefix[1000];
	char label_txt[1000];
	char label_tex[1000];

	int q;
	finite_field *F;
	int k;

	//int f_semilinear;

	action *A;
	action *A2;

	int degree;

	long int *set;
	int sz;

	int f_has_group;
	strong_generators *Sg;


	projective_space *P;

	int *block; // [k]


	design_create();
	~design_create();
	void null();
	void freeself();
	void init(design_create_description *Descr, int verbose_level);
	void create_design_PG_2_q(finite_field *F,
			long int *&set, int &sz, int &k, int verbose_level);
	void unrank_block_in_PG_2_q(int *block,
			int rk, int verbose_level);
	int rank_block_in_PG_2_q(int *block,
			int verbose_level);
	int get_nb_colors_as_two_design(int verbose_level);
	int get_color_as_two_design_assume_sorted(long int *design, int verbose_level);
};




// #############################################################################
// difference_set_in_heisenberg_group.cpp
// #############################################################################


//! to find difference sets in Heisenberg groups following Tao


class difference_set_in_heisenberg_group {

public:
	char fname_base[1000];

	int n;
	int q;
	finite_field *F;
	heisenberg *H;
	int *Table;
	int *Table_abv;
	int *gens;
	int nb_gens;

#if 0
	int *N_gens;
	int N_nb_gens;
	int N_go;
#endif
	action *A;
	strong_generators *Aut_gens;
	longinteger_object Aut_order;

	int given_base_length; // = nb_gens
	int *given_base; // = gens
	int *base_image;
	int *base_image_elts;
	int *E1;
	int rk_E1;

	char prefix[1000];
	char fname_magma_out[1000];
	sims *Aut;
	sims *U;
	longinteger_object U_go;
	vector_ge *U_gens;
	schreier *Sch;


	// N = normalizer of U in Aut
	int *N_gens;
	int N_nb_gens, N_go;
	action *N;
	longinteger_object N_order;

	action *N_on_orbits;
	int *Paired_with;
	int nb_paired_orbits;
	long int *Pairs;
	int *Pair_orbit_length;


	int *Pairs_of_type1;
	int nb_pairs_of_type1;
	int *Pairs_of_type2;
	int nb_pairs_of_type2;
	int *Sets1;
	int *Sets2;


	long int *Short_pairs;
	long int *Long_pairs;

	int *f_orbit_select;
	int *Short_orbit_inverse;

	action *A_on_short_orbits;
	int nb_short_orbits;
	int nb_long_orbits;

	poset_classification *gen;




	void init(int n, finite_field *F, int verbose_level);
	void do_n2q3(int verbose_level);
	void check_overgroups_of_order_nine(int verbose_level);
	void create_minimal_overgroups(int verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);

};

void difference_set_in_heisenberg_group_early_test_func(
		long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level);


// #############################################################################
// graph_classify.cpp
// #############################################################################

//! classification of graphs and tournaments


class graph_classify {

public:

	poset *Poset;
	poset_classification *gen;

	action *A_base; // symmetric group on n vertices
	action *A_on_edges; // action on pairs

	int f_n;
	int n; // number of vertices
	int n2; // n choose 2

	int *adjacency; // [n * n]

	//int f_lex;

	int f_regular;
	int regularity;
	int *degree_sequence; // [n]

	int f_girth;
	int girth;
	int *neighbor; // [n]
	int *neighbor_idx; // [n]
	int *distance; // [n]

	int f_list; // list whole orbits in the end
	int f_list_all; // list whole orbits in the end
	int f_draw_graphs;
	int f_embedded;
	int f_sideways;
	int f_draw_graphs_at_level;
	int level;
	double scale;
	int f_x_stretch;
	double x_stretch;

	int f_depth;
	int depth;

	long int *S1; // [n2]


	int f_tournament;
	int f_no_superking;

	int f_draw_level_graph;
	int level_graph_level;
	int f_test_multi_edge;

	int f_draw_poset;
	int f_draw_full_poset;
	int f_plesken;

	int f_identify;
	long int identify_data[1000];
	int identify_data_sz;




	graph_classify();
	~graph_classify();
	void read_arguments(int argc, const char **argv);
	void init(int argc, const char **argv);
	int check_conditions(int len, long int *S, int verbose_level);
	int check_conditions_tournament(int len, long int *S,
			int verbose_level);
	int check_regularity(long int *S, int len,
			int verbose_level);
	int compute_degree_sequence(long int *S, int len);
	int girth_check(long int *S, int len, int verbose_level);
	int girth_test_vertex(long int *S, int len,
			int vertex, int girth, int verbose_level);
	void get_adjacency(long int *S, int len, int verbose_level);
	void print(std::ostream &ost, long int *S, int len);
	void print_score_sequences(int level, int verbose_level);
	void score_sequence(int n, long int *set, int sz, long int *score, int verbose_level);
	void draw_graphs(int level, double scale,
			int xmax_in, int ymax_in, int xmax, int ymax,
			int f_embedded, int f_sideways, int verbose_level);

};

void graph_classify_test_function(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level);
void graph_classify_print_set(std::ostream &ost,
		int len, long int *S, void *data);



// #############################################################################
// hadamard_classify.cpp
// #############################################################################

//! classification of Hadamard matrices




class hadamard_classify {

public:
	int n;
	int N, N2;
	int bitvector_length;
	uchar *bitvector_adjacency;
	colored_graph *CG;

	action *A;

	int *v;

	poset_classification *gen;
	int nb_orbits;

	void init(int n, int f_draw, int verbose_level, int verbose_level_clique);
	int clique_test(long int *set, int sz);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int dot_product(int a, int b, int n);
};


void hadamard_classify_early_test_function(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);



// #############################################################################
// large_set_classify.cpp
// #############################################################################

//! classification of large sets of designs

class large_set_classify {
public:
	design_create *DC;
	int design_size;
	int nb_points;
	int nb_lines;
	int search_depth;

	char starter_directory_name[1000];
	char prefix[1000];
	char path[1000];
	char prefix_with_directory[1000];


	int f_lexorder_test;
	int size_of_large_set;


	long int *Design_table;
	const char *design_table_prefix;
	int nb_designs;
	int nb_colors;
	int *design_color_table;

	action *A_on_designs;


	uchar *bitvector_adjacency;
	int bitvector_length;
	int *degree;

	poset *Poset;
	poset_classification *gen;

	int nb_needed;

	long int *Design_table_reduced;
	long int *Design_table_reduced_idx;
	int nb_reduced;
	int nb_remaining_colors;
	int *reduced_design_color_table;

	action *A_reduced;
	schreier *Orbits_on_reduced;
	int *color_of_reduced_orbits;

	orbits_on_something *OoS;
	int selected_type_idx;


	large_set_classify();
	~large_set_classify();
	void null();
	void freeself();
	void init(design_create *DC,
			const char *input_prefix, const char *base_fname,
			int search_depth,
			int f_lexorder_test,
			const char *design_table_prefix,
			int verbose_level);
	void init_designs(orbit_of_sets *SetOrb,
			int verbose_level);
	void compute(int verbose_level);
	void read_classification(orbit_transversal *&T,
			int level, int verbose_level);
	void read_classification_single_case(set_and_stabilizer *&Rep,
			int level, int case_nr, int verbose_level);
	void make_reduced_design_table(
			long int *set, int set_sz,
			long int *&Design_table_out, long int *&Design_table_out_idx, int &nb_out,
			int verbose_level);
	void compute_colors(
			long int *Design_table, int nb_designs, int *&design_color_table,
			int verbose_level);
	void compute_reduced_colors(
			long int *set, int set_sz,
			int verbose_level);
	int designs_are_disjoint(int i, int j);
	void process_starter_case(set_and_stabilizer *Rep,
			strong_generators *SG, const char *prefix,
			char *group_label, int orbit_length,
			int f_read_solution_file, const char *solution_file_name,
			long int *&Large_sets, int &nb_large_sets,
			int verbose_level);
	int test_orbit(long int *orbit, int orbit_length);
	int test_pair_of_orbits(
			long int *orbit1, int orbit_length1,
			long int *orbit2, int orbit_length2);

};

int large_set_design_test_orbit(long int *orbit, int orbit_length,
		void *extra_data);
int large_set_design_test_pair_of_orbits(long int *orbit1, int orbit_length1,
		long int *orbit2, int orbit_length2, void *extra_data);
int large_set_design_compare_func_for_invariants(void *data, int i, int j, void *extra_data);
void large_set_swap_func_for_invariants(void *data, int i, int j, void *extra_data);
int large_set_design_compare_func(void *data, int i, int j, void *extra_data);
void large_set_swap_func(void *data, int i, int j, void *extra_data);
void large_set_early_test_function(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
int large_set_compute_color_of_reduced_orbits_callback(schreier *Sch,
		int orbit_idx, void *data, int verbose_level);



// #############################################################################
// regular_ls_classify.cpp
// #############################################################################


//! classification of regular linear spaces




class regular_ls_classify {

public:
	int m;
	int n;
	int k;
	int r;

	//int onk;
	//int onr;
	int starter_size;
	int target_size;
	int *initial_pair_covering;

	char starter_directory_name[1000];
	char prefix[1000];
	char prefix_with_directory[1000];

	int m2;
	int *v1; // [k]

	poset *Poset;
	poset_classification *gen;
	action *A;
	action *A2;
	action_on_k_subsets *Aonk; // only a pointer, do not free

	int *row_sum; // [m]
	int *pairs; // [m2]
	int *open_rows; // [m]
	int *open_row_idx; // [m]
	int *open_pairs; // [m2]
	int *open_pair_idx; // [m2]

	void init_basic(int argc, const char **argv,
		const char *input_prefix, const char *base_fname,
		int starter_size,
		int verbose_level);
	void read_arguments(int argc, const char **argv);
	regular_ls_classify();
	~regular_ls_classify();
	void null();
	void freeself();
	void init_group(int verbose_level);
	void init_action_on_k_subsets(int onk, int verbose_level);
	void init_generator(
		int f_has_initial_pair_covering, int *initial_pair_covering,
		strong_generators *Strong_gens,
		int verbose_level);
	void compute_starter(
		int f_draw_poset, int f_embedded, int f_sideways, int verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void print(std::ostream &ost, long int *S, int len);
	void lifting_prepare_function_new(exact_cover *E, int starter_case,
		long int *candidates, int nb_candidates, strong_generators *Strong_gens,
		diophant *&Dio, long int *&col_labels,
		int &f_ruled_out,
		int verbose_level);
};




void regular_ls_classify_print_set(std::ostream &ost, int len, long int *S, void *data);
void regular_ls_classify_early_test_function(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
int regular_ls_classify_check_function_incremental_callback(int len, int *S,
		void *data, int verbose_level);
void regular_ls_classify_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates, strong_generators *Strong_gens,
	diophant *&Dio, long int *&col_labels,
	int &f_ruled_out,
	int verbose_level);



// #############################################################################
// tactical_decomposition.cpp
// #############################################################################

//! tactical decomposition of an incidence structure with respect to a given group



class tactical_decomposition {
public:

	int set_size;
	int nb_blocks;
	incidence_structure *Inc;
	int f_combined_action;
	action *A;
	action *A_on_points;
	action *A_on_lines;
	strong_generators * gens;
	partitionstack *Stack;
	schreier *Sch;
	schreier *Sch_points;
	schreier *Sch_lines;

	tactical_decomposition();
	~tactical_decomposition();
	void init(int nb_rows, int nb_cols,
			incidence_structure *Inc,
			int f_combined_action,
			action *Aut,
			action *A_on_points,
			action *A_on_lines,
			strong_generators * gens,
			int verbose_level);
	void report(int f_enter_math, std::ostream &ost);

};





}}


#endif /* ORBITER_SRC_LIB_TOP_LEVEL_COMBINATORICS_TL_COMBINATORICS_H_ */
