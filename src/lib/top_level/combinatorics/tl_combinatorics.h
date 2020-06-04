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
// boolean_function.cpp
// #############################################################################


//! boolean function


class boolean_function {
public:
	int n;
	int n2; // n / 2
	int Q; // 2^n
	int Q2; // 2^{n/2}
	//int NN;
	longinteger_object NN; // 2^Q
	int N; // size of PG(n,2)

	finite_field *Fq; // the field F2
	finite_field *FQ; // the field of order 2^n

	homogeneous_polynomial_domain *Poly;
		// Poly[i] = polynomial of degree i in n + 1 variables.
		// i = 1,..,n
	int **A_poly;
	int **B_poly;
	int *Kernel;
	int dim_kernel;

	action *A;
	vector_ge *nice_gens;

	action_on_homogeneous_polynomials *AonHPD;
	strong_generators *SG;
	longinteger_object go;

	long int *affine_points; // [Q]
	action *A_affine; // restricted action on affine points


	int *v; // [n]
	int *v1; // [n + 1]
	int *w; // [n]
	int *f; // [Q]
	int *f2; // [Q]
	int *F; // [Q]
	int *T; // [Q]
	int *W; // [Q * Q]
	int *f_proj;
	int *f_proj2;



	boolean_function();
	~boolean_function();
	void init(int n, int verbose_level);
	void init_group(int verbose_level);
	void setup_polynomial_rings(int verbose_level);
	void compute_polynomial_representation(int *func, int *coeff, int verbose_level);
	void evaluate_projectively(int *coeff, int *f);
	void evaluate(int *coeff, int *f);
	void raise(int *in, int *out);
	void apply_Walsh_transform(int *in, int *out);
	int is_bent(int *T);
	void search_for_bent_functions(int verbose_level);
};


void boolean_function_print_function(int *poly, int sz, void *data);
void boolean_function_reduction_function(int *poly, void *data);


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
	poset_classification_control *Control;
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
// create_graph_description.cpp
// #############################################################################


//! a description of a graph read from the command line


class create_graph_description {
public:

	int f_load_from_file;
	const char *fname;

	int f_edge_list;
	int n;
	const char *edge_list_text;

	int f_edges_as_pairs;
	const char *edges_as_pairs_text;


	int f_Johnson;
	int Johnson_n;
	int Johnson_k;
	int Johnson_s;

	int f_Paley;
	int Paley_q;

	int f_Sarnak;
	int Sarnak_p;
	int Sarnak_q;

	int f_Schlaefli;
	int Schlaefli_q;

	int f_Shrikhande;

	int f_Winnie_Li;
	int Winnie_Li_q;
	int Winnie_Li_index;

	int f_Grassmann;
	int Grassmann_n;
	int Grassmann_k;
	int Grassmann_q;
	int Grassmann_r;

	int f_coll_orthogonal;
	int coll_orthogonal_epsilon;
	int coll_orthogonal_d;
	int coll_orthogonal_q;


	create_graph_description();
	void read_arguments_from_string(
			const char *str, int verbose_level);
	int read_arguments(
		int argc, const char **argv,
		int verbose_level);


};


// #############################################################################
// create_graph.cpp
// #############################################################################


//! creates a graph from a description with create_graph_description


class create_graph {
public:

	create_graph_description *description;

	int f_has_CG;
	colored_graph *CG;

	int N;
	int *Adj;

	char label[1000];
	char label_tex[1000];

	create_graph();
	~create_graph();
	void init(
			create_graph_description *description,
			int verbose_level);
	void create_Johnson(int &N, int *&Adj, int n, int k, int s, int verbose_level);
	void create_Paley(int &N, int *&Adj, int q, int verbose_level);
	void create_Sarnak(int &N, int *&Adj, int p, int q, int verbose_level);
	void create_Schlaefli(int &N, int *&Adj, int q, int verbose_level);
	void create_Shrikhande(int &N, int *&Adj, int verbose_level);
	void create_Winnie_Li(int &N, int *&Adj, int q, int index, int verbose_level);
	void create_Grassmann(int &N, int *&Adj,
			int n, int k, int q, int r, int verbose_level);
	void create_coll_orthogonal(int &N, int *&Adj,
			int epsilon, int d, int q, int verbose_level);

};

// #############################################################################
// delandtsheer_doyen_description.cpp
// #############################################################################

#define MAX_MASK_TESTS 1000


//! description of the problem for delandtsheer_doyen


class delandtsheer_doyen_description {
public:

	int f_depth;
	int depth;

	int f_d1;
	int d1;

	int f_d2;
	int d2;

	int f_q1;
	int q1;

	int f_q2;
	int q2;

	int f_group_label;
	const char *group_label;

	int f_mask_label;
	const char *mask_label;



	int DELANDTSHEER_DOYEN_X;
	int DELANDTSHEER_DOYEN_Y;
	int f_K;
	int K;

	int f_pair_search_control;
	poset_classification_control *Pair_search_control;

	int f_search_control;
	poset_classification_control *Search_control;

	// row intersection type
	int f_R;
	int nb_row_types;
	int *row_type;     		// [nb_row_types + 1]

	// col intersection type
	int f_C;
	int nb_col_types;
	int *col_type;     		// [nb_col_types + 1]


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

	int f_singletons;
	int f_subgroup;
	const char *subgroup_gens;
	const char *subgroup_order;

	delandtsheer_doyen_description();
	~delandtsheer_doyen_description();
	int read_arguments(
		int argc, const char **argv,
		int verbose_level);

};



// #############################################################################
// delandtsheer_doyen.cpp
// #############################################################################



//! search for line transitive point imprimitive linear spaces as described by Delandtsheer and Doyen



class delandtsheer_doyen {
public:

	delandtsheer_doyen_description *Descr;

	finite_field *F1;
	finite_field *F2;

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
	int *row_type_cur; 		// [nb_row_types + 1]
	int *row_type_this_or_bigger; 	// [nb_row_types + 1]

	// col intersection type
	int *col_type_cur; 		// [nb_col_types + 1]
	int *col_type_this_or_bigger; 	// [nb_col_types + 1]


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
	void init(delandtsheer_doyen_description *Descr, int verbose_level);
	void show_generators(int verbose_level);
	void search_singletons(int verbose_level);
	void search_starter(int verbose_level);
	void compute_orbits_on_pairs(strong_generators *Strong_gens, int verbose_level);
	strong_generators *scan_subgroup_generators(int verbose_level);
	void create_monomial_group(int verbose_level);
	void create_action(int verbose_level);
	void create_graph(long int *line0, int len, int verbose_level);
	int find_pair_orbit(int i, int j, int verbose_level);
	int find_pair_orbit_by_tracing(int i, int j, int verbose_level);
	void compute_pair_orbit_table(int verbose_level);
	void write_pair_orbit_file(int verbose_level);
	void print_mask_test_i(std::ostream &ost, int i);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int check_conditions(long int *S, int len, int verbose_level);
	int check_orbit_covering(long int *line, int len, int verbose_level);
	int check_row_sums(long int *line, int len, int verbose_level);
	int check_col_sums(long int *line, int len, int verbose_level);
	int check_mask(long int *line, int len, int verbose_level);
	void get_mask_core_and_singletons(
		long int *line, int len,
		int &nb_rows_used, int &nb_cols_used,
		int &nb_singletons, int verbose_level);
};


void delandtsheer_doyen_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
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

	action *A; // Sym(degree)
	action *A2; // Sym(degree), in the action on k-subsets


	action *Aut;
	// PGGL(3,q) in case of PG_2_q with q not prime
	// PGL(3,q) in case of PG_2_q withq  prime
	action *Aut_on_lines; // Aut induced on lines

	int degree;

	long int *set;
	int sz; // = b, the number of blocks

	int f_has_group;
	strong_generators *Sg;


	projective_space_with_action *PA;
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
	long int *given_base; // = gens
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

	int f_control;
	poset_classification_control *Control;

	int regularity;
	int *degree_sequence; // [n]

	int f_girth;
	int girth;
	int *neighbor; // [n]
	int *neighbor_idx; // [n]
	int *distance; // [n]

	//int f_list; // list whole orbits in the end
	//int f_list_all; // list whole orbits in the end
	int f_draw_graphs;
	//int f_embedded;
	//int f_sideways;
	int f_draw_graphs_at_level;
	int level;
	//double scale;
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

	//int f_draw_poset;
	//int f_draw_full_poset;
	//int f_plesken;

	int f_identify;
	long int identify_data[1000];
	int identify_data_sz;




	graph_classify();
	~graph_classify();
	void read_arguments(int argc, const char **argv, int verbose_level);
	void init(int argc, const char **argv, int verbose_level);
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
// graph_theoretic_activity_description.cpp
// #############################################################################

//! description of an activity for graphs


class graph_theoretic_activity_description {

public:

	int f_find_cliques;
	clique_finder_control *Clique_finder_control;
	int f_export_magma;
	int f_export_maple;
	int f_print;
	int f_sort_by_colors;
	int f_split;
	const char *split_file;


	graph_theoretic_activity_description();
	~graph_theoretic_activity_description();
	void null();
	void freeself();
	void read_arguments_from_string(
			const char *str, int verbose_level);
	int read_arguments(
		int argc, const char **argv,
		int verbose_level);


};


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
// hall_system_classify.cpp
// #############################################################################

//! classification of Hall systems


class hall_system_classify {
public:
	//int e;
	int n; // 3^e
	int nm1; // n-1
		// number of points different from the reflection point
	int nb_pairs;
		// nm1 / 2
		// = number of lines (=triples) through the reflection point
		// = number of lines (=triples) through any point
	int nb_pairs2; // = nm1 choose 2
		// number of pairs of points different from the reflection point.
		// these are the pairs of points that are covered by the
		// triples that we will choose.
		// The other pairs have been covered by the lines through
		// the reflection point, so they are fine because we assume that
		// these lines exist.
	int nb_blocks_overall; // {n \choose 2} / 6
	int nb_blocks_needed; // nb_blocks_overall - (n - 1) / 2
	int nb_orbits_needed; // nb_blocks_needed / 2
	int depth;
	int N;
		// {nb_pairs choose 3} * 8
		// {nb_pairs choose 3} counts the number of ways to shoose three lines
		// through the reflection point.
		// the times 8 is because every triple of lines through the
		// reflection point has 2^3 ways of choosing one point on each line.
	int N0; // {nb_pairs choose 3} * 4
	int *row_sum; // [nm1]
		// this is where we test whether each of the
		// points different from the reflection point lies on
		// the right number of triples.
		// The right number is nb_pairs
	int *pair_covering; // [nb_pairs2]
		// this is where we test whether each of the
		// pairs of points not including the reflection point
		// is covered once


	long int *triples; // [N0 * 6]
		// a table of all triples so that
		// we can induce the group action on to them.


	action *A;
		// The symmetric group on nm1 points.
	action *A_on_triples;
		// the induced action on unordered triples as stored in triples[].
	strong_generators *Strong_gens_Hall_reflection;
		// the involution which switches the
		// points on every line through the center (other than the center).
	strong_generators *Strong_gens_normalizer;
		// Strong generators for the normalizer of the involution.
	sims *S;
		// The normalizer of the involution

	char prefix[1000];
	char fname_orbits_on_triples[1000];
	schreier *Orbits_on_triples;
		// Orbits of the reflection group on triples.
	action *A_on_orbits;
		// Induced action of A_on_triples
		// on the orbit of the reflection group.
	int f_play_it_safe;

	poset_classification_control *Control;
	poset *Poset;
		// subset lattice for action A_on_orbits
	poset_classification *PC;
		// Classification of subsets in the action A_on_orbits


	hall_system_classify();
	~hall_system_classify();
	void null();
	void freeself();
	void init(int argc, const char **argv,
			int n, int depth,
			int verbose_level);
	void orbits_on_triples(int verbose_level);
	void print(std::ostream &ost, long int *S, int len);
	void unrank_triple(long int *T, int rk);
	void unrank_triple_pair(long int *T1, long int *T2, int rk);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
};


// in hall_system.cpp:
void hall_system_print_set(std::ostream &ost, int len, long int *S, void *data);
void hall_system_early_test_function(long int *S, int len,
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
	int design_size; // = DC->sz = b, the number of blocks in the design
	int nb_points; // = DC->A->degree
	int nb_lines; // = DC->A2->degree
	int search_depth;

	char starter_directory_name[1000];
	char prefix[1000];
	char path[1000];
	char prefix_with_directory[1000];


	int f_lexorder_test;
	int size_of_large_set; // = nb_lines / design_size


	long int *Design_table; // [nb_designs * design_size]
	const char *design_table_prefix;
	int nb_designs; // = SetOrb->used_length;
	int nb_colors; // = DC->get_nb_colors_as_two_design(0 /* verbose_level */);
	int *design_color_table; // [nb_designs]

	action *A_on_designs; // action on designs in Design_table


	uchar *bitvector_adjacency;
	int bitvector_length;
	int *degree;

	poset_classification_control *Control;
	poset *Poset;
	poset_classification *gen;

	int nb_needed;

	// reduced designs are those which are compatible
	// with all the designs in the chosen set
	long int *Design_table_reduced; // [nb_reduced * design_size]
	long int *Design_table_reduced_idx; // [nb_reduced], index into Design_table[]
	int nb_reduced;
	int nb_remaining_colors; // = nb_colors - set_sz; // we assume that k = 4
	int *reduced_design_color_table; // [nb_reduced]
		// colors of the reduced designs after throwing away
		// the colors covered by the designs in the chosen set.
		// The remaining colors are relabeled consecutively.

	action *A_reduced;
		// reduced action A_on_designs based on Design_table_reduced_idx[]
	schreier *Orbits_on_reduced;
	int *color_of_reduced_orbits;

	orbits_on_something *OoS;
		// in action A_reduced
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
			long int *chosen_set, int chosen_set_sz,
			int verbose_level);
	int designs_are_disjoint(int i, int j);
	void process_starter_case(
			long int *starter_set, int starter_set_sz,
			strong_generators *SG, const char *prefix,
			const char *group_label, int orbit_length,
			int f_read_solution_file, const char *solution_file_name,
			long int *&Large_sets, int &nb_large_sets,
			int f_compute_normalizer_orbits, strong_generators *N_gens,
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
// pentomino_puzzle.cpp
// #############################################################################


#define NB_PIECES 18



//! generate all solutions of the pentomino puzzle


class pentomino_puzzle {

	public:
	int *S[NB_PIECES];
	int S_length[NB_PIECES];
	int *O[NB_PIECES];
	int O_length[NB_PIECES];
	int *T[NB_PIECES];
	int T_length[NB_PIECES];
	int *R[NB_PIECES];
	int R_length[NB_PIECES];
	int Rotate[4 * 25];
	int Rotate6[4 * 36];
	int var_start[NB_PIECES + 1];
	int var_length[NB_PIECES + 1];


	void main(int verbose_level);
	int has_interlocking_Ps(long int *set);
	int has_interlocking_Pprime(long int *set);
	int has_interlocking_Ls(long int *set);
	int test_if_interlocking_Ps(int a1, int a2);
	int has_interlocking_Lprime(long int *set);
	int test_if_interlocking_Ls(int a1, int a2);
	int number_of_pieces_of_type(int t, long int *set);
	int does_it_contain_an_I(long int *set);
	void decode_assembly(long int *set);
	// input set[5]
	void decode_piece(int j, int &h, int &r, int &t);
	// h is the kind of piece
	// r is the rotation index
	// t is the translation index
	// to get the actual rotation and translation, use
	// R[h][r] and T[h][t].
	int code_piece(int h, int r, int t);
	void draw_it(std::ostream &ost, long int *sol);
	void compute_image_function(set_of_sets *S,
			int elt_idx,
			int gen_idx, int &idx_of_image, int verbose_level);
	void turn_piece(int &h, int &r, int &t, int verbose_level);
	void flip_piece(int &h, int &r, int &t, int verbose_level);
	void setup_pieces();
	void setup_rotate();
	void setup_var_start();
	void make_coefficient_matrix(diophant *D);

};


void pentomino_puzzle_compute_image_function(set_of_sets *S,
		void *compute_image_data, int elt_idx,
		int gen_idx, int &idx_of_image, int verbose_level);
int pentomino_puzzle_compare_func(void *vec, void *a, int b, void *data_for_compare);


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

	poset_classification_control *Control;
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
