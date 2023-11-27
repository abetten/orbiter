/*
 * graph_theory.h
 *
 *  Created on: Mar 30, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_GRAPH_THEORY_GRAPH_THEORY_H_
#define SRC_LIB_TOP_LEVEL_GRAPH_THEORY_GRAPH_THEORY_H_




namespace orbiter {
namespace layer5_applications {
namespace apps_graph_theory {


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

	actions::action *A;
	field_theory::finite_field *F;
	int target_depth;

	int *Elt1;
	int *Elt2;
	data_structures_groups::vector_ge *gens;
	data_structures_groups::vector_ge *gens_subgroup;
	ring_theory::longinteger_object target_go, target_go_subgroup;
	groups::strong_generators *Strong_gens;
	groups::strong_generators *Strong_gens_subgroup;

	groups::sims *S;
	groups::sims *S_subgroup;

	int *Table;
	int *generators;
	int nb_generators;

	std::string fname_base;
	std::string prefix;
	std::string fname;
	std::string fname_graphs;

	groups::strong_generators *Aut_gens;
	ring_theory::longinteger_object Aut_order;
	actions::action *Aut;
	actions::action *A2;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification_control *Control;
	poset_classification::poset_classification *gen;


	void init(
			int level, int group, int subgroup, int verbose_level);
	void init_group(
			int verbose_level);
	void init_group2(
			int verbose_level);
	void init_group_level_3(
			int verbose_level);
	void init_group_level_4(
			int verbose_level);
	void init_group_level_5(
			int verbose_level);
	int incremental_check_func(
			int len, long int *S, int verbose_level);
	void classify_subsets(
			int verbose_level);
	void write_file(
			int verbose_level);
	void create_Adjacency_list(
			long int *Adj,
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


//! a description of a graph using command line arguments


class create_graph_description {
public:

	int f_load;
	std::string fname;

	int f_load_csv_no_border;

	int f_load_adjacency_matrix_from_csv_and_select_value;
	std::string load_adjacency_matrix_from_csv_and_select_value_fname;
	int load_adjacency_matrix_from_csv_and_select_value_value;

	int f_load_dimacs;

	int f_edge_list;
	int n;
	std::string edge_list_text;

	int f_edges_as_pairs;
	std::string edges_as_pairs_text;

	int f_cycle;
	int cycle_n;

	int f_inversion_graph;
	std::string inversion_graph_text;

	int f_Hamming;
	int Hamming_n;
	int Hamming_q;

	int f_Johnson;
	int Johnson_n;
	int Johnson_k;
	int Johnson_s;

	int f_Paley;
	std::string Paley_label_Fq;

	int f_Sarnak;
	int Sarnak_p;
	int Sarnak_q;

	int f_Schlaefli;
	std::string Schlaefli_label_Fq;

	int f_Shrikhande;

	int f_Winnie_Li;
	std::string Winnie_Li_label_Fq;
	int Winnie_Li_index;

	int f_Grassmann;
	int Grassmann_n;
	int Grassmann_k;
	std::string Grassmann_label_Fq;
	int Grassmann_r;

	int f_coll_orthogonal;
	std::string coll_orthogonal_space_label;
	std::string coll_orthogonal_set_of_points_label;

	int f_tritangent_planes_disjointness_graph;
	int f_trihedral_pair_disjointness_graph;

	int f_non_attacking_queens_graph;
	int non_attacking_queens_graph_n;

	int f_subset;
	std::string subset_label;
	std::string subset_label_tex;
	std::string subset_text;

	int f_disjoint_sets_graph;
	std::string disjoint_sets_graph_fname;

	int f_orbital_graph;
	std::string orbital_graph_group;
	int orbital_graph_orbit_idx;

	int f_collinearity_graph;
	std::string collinearity_graph_matrix;

	int f_chain_graph;
	std::string chain_graph_partition_1;
	std::string chain_graph_partition_2;

	int f_Cayley_graph;
	std::string Cayley_graph_group;
	std::string Cayley_graph_gens;

	std::vector<graph_modification_description> Modifications;

	create_graph_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};


// #############################################################################
// create_graph.cpp
// #############################################################################


//! creates a graph from a description with create_graph_description


class create_graph {
public:

	create_graph_description *description;

	int f_has_CG;
	graph_theory::colored_graph *CG;

	int N;
	int *Adj;


	std::string label;
	std::string label_tex;

	create_graph();
	~create_graph();
	void init(
			create_graph_description *description,
			int verbose_level);
	void create_cycle(
			int n, int verbose_level);
	void create_inversion_graph(
			std::string &perm_text, int verbose_level);
	void create_Hamming(
			int n, int q, int verbose_level);
	void create_Johnson(
			int n, int k, int s, int verbose_level);
	void create_Paley(
			std::string &label_Fq, int verbose_level);
	void create_Sarnak(
			int p, int q,
			int verbose_level);
	void create_Schlaefli(
			std::string &label_Fq, int verbose_level);
	void create_Shrikhande(
			int verbose_level);
	void create_Winnie_Li(
			std::string &label_Fq, int index, int verbose_level);
	void create_Grassmann(
			int n, int k, std::string &label_Fq,
			int r, int verbose_level);
	void create_coll_orthogonal(
			std::string &orthogonal_space_label,
			std::string &set_of_points_label,
			int verbose_level);
	void make_orbital_graph(
			apps_algebra::any_group *AG, int orbit_idx,
			int verbose_level);
	void make_collinearity_graph(
			int *Inc, int nb_rows, int nb_cols,
			int verbose_level);
	void make_chain_graph(
			int *part1, int sz1,
			int *part2, int sz2,
			int verbose_level);

};


// #############################################################################
// graph_classification_activity_description.cpp
// #############################################################################

//! an activity for a classification of graphs and tournaments


class graph_classification_activity_description {

public:
	int f_draw_level_graph;
	int draw_level_graph_level;

	int f_draw_graphs;

	int f_list_graphs_at_level;
	int list_graphs_at_level_level_min;
	int list_graphs_at_level_level_max;

	int f_draw_graphs_at_level;
	int draw_graphs_at_level_level;

	int f_draw_options;
	graphics::layered_graph_draw_options *draw_options;

	int f_recognize_graphs_from_adjacency_matrix_csv;
	std::string recognize_graphs_from_adjacency_matrix_csv_fname;


	graph_classification_activity_description();
	~graph_classification_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// graph_classification_activity.cpp
// #############################################################################

//! an activity for a classification of graphs and tournaments


class graph_classification_activity {

public:

	graph_classification_activity_description *Descr;
	graph_classify *GC;

	graph_classification_activity();
	~graph_classification_activity();
	void init(
			graph_classification_activity_description *Descr,
			graph_classify *GC,
			int verbose_level);
	void perform_activity(int verbose_level);


};


// #############################################################################
// graph_classify_description.cpp
// #############################################################################

//! classification of graphs and tournaments


class graph_classify_description {

public:
	int f_n;
	int n; // number of vertices

	int f_regular;

	int f_control;
	poset_classification::poset_classification_control *Control;

	int regularity;

	int f_girth;
	int girth;

	int f_depth;
	int depth;

	int f_tournament;
	int f_no_superking;

	int f_test_multi_edge;


	int f_identify;
	long int identify_data[1000]; // ToDo
	int identify_data_sz;


	graph_classify_description();
	~graph_classify_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// graph_classify.cpp
// #############################################################################

//! classification of graphs and tournaments


class graph_classify {

public:

	graph_classify_description *Descr;

	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *gen;

	actions::action *A_base; // symmetric group on n vertices
	actions::action *A_on_edges; // action on pairs

	int n2; // n choose 2

	int *adjacency; // [n * n]

	int *degree_sequence; // [n]

	int *neighbor; // [n]
	int *neighbor_idx; // [n]
	int *distance; // [n]

	long int *S1; // [n2]






	graph_classify();
	~graph_classify();
	void init(
			graph_classify_description *Descr, int verbose_level);
	int check_conditions(
			int len, long int *S, int verbose_level);
	int check_conditions_tournament(
			int len, long int *S,
			int verbose_level);
	int check_regularity(
			long int *S, int len,
			int verbose_level);
	int compute_degree_sequence(
			long int *S, int len);
	int girth_check(
			long int *S, int len, int verbose_level);
	int girth_test_vertex(
			long int *S, int len,
			int vertex, int girth, int verbose_level);
	void get_adjacency(
			long int *S, int len, int verbose_level);
	void print(
			std::ostream &ost, long int *S, int len);
	void print_score_sequences(
			int level, int verbose_level);
	void score_sequence(
			int n, long int *set, int sz, long int *score,
			int verbose_level);
	void list_graphs(
			int level_min, int level_max, int verbose_level);
	void draw_graphs(
			int level,
			graphics::layered_graph_draw_options *draw_options,
			int verbose_level);
	void recognize_graph_from_adjacency_list(
			int *Adj, int N2,
			int &iso_type,
			int verbose_level);
	int number_of_orbits();

};



// #############################################################################
// graph_modification_description.cpp
// #############################################################################

//! unary operators to modify graphs and tournaments


class graph_modification_description {

public:

	int f_complement;

	int f_distance_2;


	graph_modification_description();
	~graph_modification_description();
	int check_and_parse_argument(
		int argc, int &i, std::string *argv,
		int verbose_level);
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();
	void apply(
			graph_theory::colored_graph *&CG, int verbose_level);

};


// #############################################################################
// graph_theoretic_activity_description.cpp
// #############################################################################

//! description of an activity for graphs


class graph_theoretic_activity_description {

public:

	int f_find_cliques;
	graph_theory::clique_finder_control *Clique_finder_control;

	int f_find_subgraph;
	std::string find_subgraph_label;

	int f_export_magma;
	int f_export_maple;
	int f_export_csv;
	int f_export_graphviz;
	int f_print;
	int f_sort_by_colors;

	int f_split;
	std::string split_input_fname;
	std::string split_by_file;

	int f_split_by_starters;
	std::string split_by_starters_fname_reps;
	std::string split_by_starters_col_label;

	int f_split_by_clique;
	std::string split_by_clique_label;
	std::string split_by_clique_set;

	int f_save;
	int f_automorphism_group;

	int f_properties;
	int f_eigenvalues;
	int f_draw;


	graph_theoretic_activity_description();
	~graph_theoretic_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};


// #############################################################################
// graph_theoretic_activity.cpp
// #############################################################################

//! an activity for graphs


class graph_theoretic_activity {

public:

	graph_theoretic_activity_description *Descr;
	graph_theory::colored_graph *CG;


	graph_theoretic_activity();
	~graph_theoretic_activity();
	void init(
			graph_theoretic_activity_description *Descr,
			graph_theory::colored_graph *CG,
			int verbose_level);
	void perform_activity(
			int verbose_level);


};




// #############################################################################
// graph_theory_apps.cpp
// #############################################################################

//! applications in graph theory involving groups


class graph_theory_apps {

public:

	graph_theory_apps();
	~graph_theory_apps();
	void automorphism_group(
			graph_theory::colored_graph *CG, int verbose_level);
	void expander_graph(
			int p, int q,
			int f_special,
			field_theory::finite_field *F,
			actions::action *A,
			int *&Adj, int &N,
			int verbose_level);


};




}}}


#endif /* SRC_LIB_TOP_LEVEL_GRAPH_THEORY_GRAPH_THEORY_H_ */
