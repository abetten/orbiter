// sajeeb.h


#include <iostream>
#include <fstream>
//#include <sstream>
#include <iomanip>

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include <map>
#include <vector>
#include <deque>



#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif


#define SYSTEMUNIX



typedef class file_output file_output;
typedef class clique_finder clique_finder;
typedef class colored_graph colored_graph;
typedef class rainbow_cliques rainbow_cliques;



using namespace std;

typedef int *pint;
typedef int **ppint;
typedef unsigned int uint;
typedef uint *puint;
typedef long LONG;
typedef LONG *PLONG;
typedef unsigned long ULONG;
typedef ULONG *PULONG;
typedef short SHORT;
typedef SHORT *PSHORT;
typedef char *pchar;
typedef unsigned char uchar;
typedef uchar *puchar;
typedef char SCHAR;
typedef SCHAR *PSCHAR;
typedef float FLOAT;
typedef FLOAT *PFLOAT;
typedef char TSTRING;
typedef int *pint;
typedef void *pvoid;
typedef int int4;


void read_set_from_file(const char *fname, int *&the_set, int &set_size, int verbose_level);
void replace_extension_with(char *p, const char *new_ext);
int file_size(const char *name);
void int_vec_copy(int *from, int *to, int len);
void int_vec_zero(int *v, int len);
void int_vec_print(ostream &ost, int *v, int len);
void int_vec_print_fully(ostream &ost, int *v, int len);
void int_set_print(int *v, int len);
void int_set_print(ostream &ost, int *v, int len);
void print_set(ostream &ost, int size, int *set);
void int_vec_swap_points(int *list, int *list_inv, int idx1, int idx2);
uchar *bitvector_allocate(int length);
uchar *bitvector_allocate_and_coded_length(int length, int &coded_length);
void bitvector_m_ii(uchar *bitvec, int i, int a);
void bitvector_set_bit(uchar *bitvec, int i);
int bitvector_s_i(uchar *bitvec, int i);
// returns 0 or 1
int ij2k(int i, int j, int n);
void k2ij(int k, int & i, int & j, int n);
void get_extension_if_present(const char *p, char *ext);
void chop_off_extension_if_present(char *p, const char *ext);
void fwrite_int4(FILE *fp, int a);
int4 fread_int4(FILE *fp);
void fwrite_uchars(FILE *fp, uchar *p, int len);
void fread_uchars(FILE *fp, uchar *p, int len);
void colored_graph_all_cliques_list_of_cases(int *list_of_cases, int nb_cases, int f_output_solution_raw, 
	const char *fname_template, 
	const char *fname_sol, const char *fname_stats, 
	int f_maxdepth, int maxdepth, 
	int f_prefix, const char *prefix, 
	int print_interval, 
	int verbose_level);
void call_back_clique_found_using_file_output(clique_finder *CF, int verbose_level);


// ##################################################################################################
// colored_graph.C
// ##################################################################################################


class colored_graph {
public:

	char fname_base[1000];
	
	int nb_points;
	int nb_colors;
	
	int bitvector_length;
	int L;
	
	int *points;
	int *point_color;
	

	int user_data_size;
	int *user_data;

	int f_ownership_of_bitvec;
	uchar *bitvector_adjacency;

	int f_has_list_of_edges;
	int nb_edges;
	int *list_of_edges;

	colored_graph();
	~colored_graph();
	void null();
	void freeself();
	void compute_edges(int verbose_level);
	int is_adjacent(int i, int j);
	void set_adjacency(int i, int j, int a);
	void print();
	void init(int nb_points, int nb_colors, 
		int *colors, uchar *bitvec, int f_ownership_of_bitvec, 
		int verbose_level);
	void init_with_point_labels(int nb_points, int nb_colors, 
		int *colors, uchar *bitvec, int f_ownership_of_bitvec, 
		int *point_labels, 
		int verbose_level);
	void init_no_colors(int nb_points, uchar *bitvec, int f_ownership_of_bitvec, 
		int verbose_level);
	void init_adjacency(int nb_points, int nb_colors, 
		int *colors, int *Adj, int verbose_level);
	void init_adjacency_no_colors(int nb_points, int *Adj, int verbose_level);
	void init_user_data(int *data, int data_size, int verbose_level);
	void load(const char *fname, int verbose_level);
	void all_cliques_of_size_k_ignore_colors(int target_depth, 
		int &nb_sol, int &decision_step_counter, int verbose_level);
	void all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file(int target_depth, 
		const char *fname, 
		int f_restrictions, int *restrictions, 
		int &nb_sol, int &decision_step_counter, 
		int verbose_level);
	void all_rainbow_cliques(ofstream *fp, int f_output_solution_raw, 
		int f_maxdepth, int maxdepth, 
		int f_restrictions, int *restrictions, 
		int f_tree, int f_decision_nodes_only, const char *fname_tree,  
		int print_interval, 
		int &search_steps, int &decision_steps, int &nb_sol, int &dt, 
		int verbose_level);
	void all_rainbow_cliques_with_additional_test_function(ofstream *fp, int f_output_solution_raw, 
		int f_maxdepth, int maxdepth, 
		int f_restrictions, int *restrictions, 
		int f_tree, int f_decision_nodes_only, const char *fname_tree,  
		int print_interval, 
		int f_has_additional_test_function,
		void (*call_back_additional_test_function)(rainbow_cliques *R, void *user_data, 
			int current_clique_size, int *current_clique, 
			int nb_pts, int &reduced_nb_pts, 
			int *pt_list, int *pt_list_inv, 
			int verbose_level), 
		int f_has_print_current_choice_function,
		void (*call_back_print_current_choice)(clique_finder *CF, 
			int depth, void *user_data, int verbose_level), 
		void *user_data, 
		int &search_steps, int &decision_steps, int &nb_sol, int &dt, 
		int verbose_level);
	void export_to_file(const char *fname, int verbose_level);
	void early_test_func_for_clique_search(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);

};

// ##################################################################################################
// file_output.C:
// ##################################################################################################


class file_output {
public:
	char fname[1000];
	int f_file_is_open;
	ofstream *fp;
	void *user_data;
	
	file_output();
	~file_output();
	void null();
	void freeself();
	void open(const char *fname, void *user_data, int verbose_level);
	void close();
	void write_line(int nb, int *data, int verbose_level);
};


// ##################################################################################################
// clique_finder.C
// ##################################################################################################


class clique_finder {
public:
	char label[1000];
	int n; // number of points
	
	int print_interval;
	
	int f_write_tree;
	int f_decision_nodes_only;
	char fname_tree[1000];
	ofstream *fp_tree;

	
	int f_maxdepth;
	int maxdepth;
	
	int *point_labels;
	int *point_is_suspicous;
	
	int target_depth;
	int verbose_level;
	

	int f_has_bitmatrix;
	int bitmatrix_m;
	int bitmatrix_n;
	int bitmatrix_N;
	uchar *bitmatrix_adjacency;

	int f_has_adj_list;
	int *adj_list_coded;
	int f_has_bitvector;
	uchar *bitvector_adjacency;

	int *pt_list;
	int *pt_list_inv;
	int *nb_points;
	int *candidates; // [max_depth * n]
	int *nb_candidates; // [max_depth]
	int *current_choice; // [max_depth]
	int *level_counter; // [max_depth] (added Nov 8, 2014)
	int *f_level_mod; // [max_depth] (added Nov 8, 2014)
	int *level_r; // [max_depth] (added Nov 8, 2014)
	int *level_m; // [max_depth] (added Nov 8, 2014)

	int *current_clique; // [max_depth]

	uint counter; // number of backtrack nodes
	uint decision_step_counter; // number of backtrack nodes that are decision nodes

	// solution storage:
	int f_store_solutions;
	deque<vector<int> > solutions;
	int nb_sol;


	// callbacks:
	void (*call_back_clique_found)(clique_finder *CF, int verbose_level);
	
	// added May 26, 2009:
	void (*call_back_add_point)(clique_finder *CF, 
		int current_clique_size, int *current_clique, 
		int pt, int verbose_level);
	void (*call_back_delete_point)(clique_finder *CF, 
		int current_clique_size, int *current_clique, 
		int pt, int verbose_level);
	int (*call_back_find_candidates)(clique_finder *CF, 
		int current_clique_size, int *current_clique, 
		int nb_pts, int &reduced_nb_pts, 
		int *pt_list, int *pt_list_inv, 
		int *candidates, int verbose_level);
		// Jan 2, 2012: added reduced_nb_pts and pt_list_inv

	int (*call_back_is_adjacent)(clique_finder *CF, 
		int pt1, int pt2, int verbose_level);
	// added Oct 2011:
	void (*call_back_after_reduction)(clique_finder *CF, 
		int depth, int nb_points, int verbose_level);

	// added Nov 2014:
	int f_has_print_current_choice_function;
	void (*call_back_print_current_choice)(clique_finder *CF, 
		int depth, void *user_data, int verbose_level);
	void *print_current_choice_data;
	
	void *call_back_clique_found_data;
	
	
	void open_tree_file(const char *fname_base, int f_decision_nodes_only);
	void close_tree_file();
	void init(const char *label, int n, 
		int target_depth, 
		int f_has_adj_list, int *adj_list_coded, 
		int f_has_bitvector, uchar *bitvector_adjacency, 
		int print_interval, 
		int f_maxdepth, int maxdepth, 
		int f_store_solutions, 
		int verbose_level);
	void allocate_bitmatrix(int verbose_level);
	void init_restrictions(int *restrictions, int verbose_level);
	clique_finder();
	~clique_finder();
	void null();
	void free();
	void init_point_labels(int *pt_labels);
	void print_set(int size, int *set);
	void log_position_and_choice(int depth, int counter_save, int counter);
	void log_position(int depth, int counter_save, int counter);
	void log_choice(int depth);
	void swap_point(int idx1, int idx2);
	int degree_of_point(int depth, int i, int nb_points);
	//int degree_of_point_verbose(int i, int nb_points);
	int is_suspicous(int i);
	int point_label(int i);
	int is_adjacent(int depth, int i, int j);
	int is_viable(int depth, int pt);
	void write_entry_to_tree_file(int depth, int verbose_level);
	void m_iji(int i, int j, int a);
	int s_ij(int i, int j);
	void backtrack_search(int depth, int verbose_level);
	int solve_decision_problem(int depth, int verbose_level);
		// returns TRUE if we found a solution
	void get_solutions(int *&Sol, int &nb_solutions, int &clique_sz, int verbose_level);
};

void all_cliques_of_given_size(int *Adj, int nb_pts, int clique_sz, int *&Sol, int &nb_sol, int verbose_level);

// ##################################################################################################
// rainbow_cliques.C
// ##################################################################################################


class rainbow_cliques {
public:

	rainbow_cliques();
	~rainbow_cliques();
	void null();
	void freeself();

	ofstream *fp_sol;
	int f_output_solution_raw;
	
	colored_graph *graph;
	clique_finder *CF;
	int *f_color_satisfied;
	int *color_chosen_at_depth;
	int *color_frequency;
	int target_depth;

	// added November 5, 2014:
	int f_has_additional_test_function;
	void (*call_back_additional_test_function)(rainbow_cliques *R, void *user_data, 
		int current_clique_size, int *current_clique, 
		int nb_pts, int &reduced_nb_pts, 
		int *pt_list, int *pt_list_inv, 
		int verbose_level);
	void *user_data;


	void search(colored_graph *graph, ofstream *fp_sol, int f_output_solution_raw, 
		int f_maxdepth, int maxdepth, 
		int f_restrictions, int *restrictions, 
		int f_tree, int f_decision_nodes_only, const char *fname_tree,  
		int print_interval, 
		int &search_steps, int &decision_steps, int &nb_sol, int &dt, 
		int verbose_level);
	void search_with_additional_test_function(colored_graph *graph, ofstream *fp_sol, int f_output_solution_raw, 
		int f_maxdepth, int maxdepth, 
		int f_restrictions, int *restrictions, 
		int f_tree, int f_decision_nodes_only, const char *fname_tree,  
		int print_interval, 
		int f_has_additional_test_function,
		void (*call_back_additional_test_function)(rainbow_cliques *R, void *user_data, 
			int current_clique_size, int *current_clique, 
			int nb_pts, int &reduced_nb_pts, 
			int *pt_list, int *pt_list_inv, 
			int verbose_level), 
		int f_has_print_current_choice_function,
		void (*call_back_print_current_choice)(clique_finder *CF, 
			int depth, void *user_data, int verbose_level), 
		void *user_data, 
		int &search_steps, int &decision_steps, int &nb_sol, int &dt, 
		int verbose_level);
	int find_candidates(
		int current_clique_size, int *current_clique, 
		int nb_pts, int &reduced_nb_pts, 
		int *pt_list, int *pt_list_inv, 
		int *candidates, int verbose_level);
	void clique_found(int *current_clique, int verbose_level);
	void clique_found_record_in_original_labels(int *current_clique, int verbose_level);

};

void call_back_colored_graph_clique_found(clique_finder *CF, int verbose_level);
void call_back_colored_graph_add_point(clique_finder *CF, 
	int current_clique_size, int *current_clique, 
	int pt, int verbose_level);
void call_back_colored_graph_delete_point(clique_finder *CF, 
	int current_clique_size, int *current_clique, 
	int pt, int verbose_level);
int call_back_colored_graph_find_candidates(clique_finder *CF, 
	int current_clique_size, int *current_clique, 
	int nb_pts, int &reduced_nb_pts, 
	int *pt_list, int *pt_list_inv, 
	int *candidates, int verbose_level);


