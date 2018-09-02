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

typedef long int INT;
typedef INT *PINT;
typedef INT **PPINT;
typedef unsigned long UINT;
typedef UINT *PUINT;
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
typedef int INT4;


void read_set_from_file(const char *fname, INT *&the_set, INT &set_size, INT verbose_level);
void replace_extension_with(char *p, const char *new_ext);
INT file_size(const char *name);
void INT_vec_copy(INT *from, INT *to, INT len);
void INT_vec_zero(INT *v, INT len);
void INT_vec_print(ostream &ost, INT *v, INT len);
void INT_vec_print_fully(ostream &ost, INT *v, INT len);
void INT_set_print(INT *v, INT len);
void INT_set_print(ostream &ost, INT *v, INT len);
void print_set(ostream &ost, INT size, INT *set);
void INT_vec_swap_points(INT *list, INT *list_inv, INT idx1, INT idx2);
uchar *bitvector_allocate(INT length);
uchar *bitvector_allocate_and_coded_length(INT length, INT &coded_length);
void bitvector_m_ii(uchar *bitvec, INT i, INT a);
void bitvector_set_bit(uchar *bitvec, INT i);
INT bitvector_s_i(uchar *bitvec, INT i);
// returns 0 or 1
INT ij2k(INT i, INT j, INT n);
void k2ij(INT k, INT & i, INT & j, INT n);
void get_extension_if_present(const char *p, char *ext);
void chop_off_extension_if_present(char *p, const char *ext);
void fwrite_INT4(FILE *fp, INT a);
INT4 fread_INT4(FILE *fp);
void fwrite_uchars(FILE *fp, uchar *p, INT len);
void fread_uchars(FILE *fp, uchar *p, INT len);
void colored_graph_all_cliques_list_of_cases(INT *list_of_cases, INT nb_cases, INT f_output_solution_raw, 
	const char *fname_template, 
	const char *fname_sol, const char *fname_stats, 
	INT f_maxdepth, INT maxdepth, 
	INT f_prefix, const char *prefix, 
	INT print_interval, 
	INT verbose_level);
void call_back_clique_found_using_file_output(clique_finder *CF, INT verbose_level);


// ##################################################################################################
// colored_graph.C
// ##################################################################################################


class colored_graph {
public:

	char fname_base[1000];
	
	INT nb_points;
	INT nb_colors;
	
	INT bitvector_length;
	INT L;
	
	INT *points;
	INT *point_color;
	

	INT user_data_size;
	INT *user_data;

	INT f_ownership_of_bitvec;
	uchar *bitvector_adjacency;

	INT f_has_list_of_edges;
	INT nb_edges;
	INT *list_of_edges;

	colored_graph();
	~colored_graph();
	void null();
	void freeself();
	void compute_edges(INT verbose_level);
	INT is_adjacent(INT i, INT j);
	void set_adjacency(INT i, INT j, INT a);
	void print();
	void init(INT nb_points, INT nb_colors, 
		INT *colors, uchar *bitvec, INT f_ownership_of_bitvec, 
		INT verbose_level);
	void init_with_point_labels(INT nb_points, INT nb_colors, 
		INT *colors, uchar *bitvec, INT f_ownership_of_bitvec, 
		INT *point_labels, 
		INT verbose_level);
	void init_no_colors(INT nb_points, uchar *bitvec, INT f_ownership_of_bitvec, 
		INT verbose_level);
	void init_adjacency(INT nb_points, INT nb_colors, 
		INT *colors, INT *Adj, INT verbose_level);
	void init_adjacency_no_colors(INT nb_points, INT *Adj, INT verbose_level);
	void init_user_data(INT *data, INT data_size, INT verbose_level);
	void load(const char *fname, INT verbose_level);
	void all_cliques_of_size_k_ignore_colors(INT target_depth, 
		INT &nb_sol, INT &decision_step_counter, INT verbose_level);
	void all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file(INT target_depth, 
		const char *fname, 
		INT f_restrictions, INT *restrictions, 
		INT &nb_sol, INT &decision_step_counter, 
		INT verbose_level);
	void all_rainbow_cliques(ofstream *fp, INT f_output_solution_raw, 
		INT f_maxdepth, INT maxdepth, 
		INT f_restrictions, INT *restrictions, 
		INT f_tree, INT f_decision_nodes_only, const char *fname_tree,  
		INT print_interval, 
		INT &search_steps, INT &decision_steps, INT &nb_sol, INT &dt, 
		INT verbose_level);
	void all_rainbow_cliques_with_additional_test_function(ofstream *fp, INT f_output_solution_raw, 
		INT f_maxdepth, INT maxdepth, 
		INT f_restrictions, INT *restrictions, 
		INT f_tree, INT f_decision_nodes_only, const char *fname_tree,  
		INT print_interval, 
		INT f_has_additional_test_function,
		void (*call_back_additional_test_function)(rainbow_cliques *R, void *user_data, 
			INT current_clique_size, INT *current_clique, 
			INT nb_pts, INT &reduced_nb_pts, 
			INT *pt_list, INT *pt_list_inv, 
			INT verbose_level), 
		INT f_has_print_current_choice_function,
		void (*call_back_print_current_choice)(clique_finder *CF, 
			INT depth, void *user_data, INT verbose_level), 
		void *user_data, 
		INT &search_steps, INT &decision_steps, INT &nb_sol, INT &dt, 
		INT verbose_level);
	void export_to_file(const char *fname, INT verbose_level);
	void early_test_func_for_clique_search(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);

};

// ##################################################################################################
// file_output.C:
// ##################################################################################################


class file_output {
public:
	char fname[1000];
	INT f_file_is_open;
	ofstream *fp;
	void *user_data;
	
	file_output();
	~file_output();
	void null();
	void freeself();
	void open(const char *fname, void *user_data, INT verbose_level);
	void close();
	void write_line(INT nb, INT *data, INT verbose_level);
};


// ##################################################################################################
// clique_finder.C
// ##################################################################################################


class clique_finder {
public:
	char label[1000];
	INT n; // number of points
	
	INT print_interval;
	
	INT f_write_tree;
	INT f_decision_nodes_only;
	char fname_tree[1000];
	ofstream *fp_tree;

	
	INT f_maxdepth;
	INT maxdepth;
	
	INT *point_labels;
	INT *point_is_suspicous;
	
	INT target_depth;
	INT verbose_level;
	

	INT f_has_bitmatrix;
	INT bitmatrix_m;
	INT bitmatrix_n;
	INT bitmatrix_N;
	uchar *bitmatrix_adjacency;

	INT f_has_adj_list;
	INT *adj_list_coded;
	INT f_has_bitvector;
	uchar *bitvector_adjacency;

	INT *pt_list;
	INT *pt_list_inv;
	INT *nb_points;
	INT *candidates; // [max_depth * n]
	INT *nb_candidates; // [max_depth]
	INT *current_choice; // [max_depth]
	INT *level_counter; // [max_depth] (added Nov 8, 2014)
	INT *f_level_mod; // [max_depth] (added Nov 8, 2014)
	INT *level_r; // [max_depth] (added Nov 8, 2014)
	INT *level_m; // [max_depth] (added Nov 8, 2014)

	INT *current_clique; // [max_depth]

	UINT counter; // number of backtrack nodes
	UINT decision_step_counter; // number of backtrack nodes that are decision nodes

	// solution storage:
	INT f_store_solutions;
	deque<vector<int> > solutions;
	INT nb_sol;


	// callbacks:
	void (*call_back_clique_found)(clique_finder *CF, INT verbose_level);
	
	// added May 26, 2009:
	void (*call_back_add_point)(clique_finder *CF, 
		INT current_clique_size, INT *current_clique, 
		INT pt, INT verbose_level);
	void (*call_back_delete_point)(clique_finder *CF, 
		INT current_clique_size, INT *current_clique, 
		INT pt, INT verbose_level);
	INT (*call_back_find_candidates)(clique_finder *CF, 
		INT current_clique_size, INT *current_clique, 
		INT nb_pts, INT &reduced_nb_pts, 
		INT *pt_list, INT *pt_list_inv, 
		INT *candidates, INT verbose_level);
		// Jan 2, 2012: added reduced_nb_pts and pt_list_inv

	INT (*call_back_is_adjacent)(clique_finder *CF, 
		INT pt1, INT pt2, INT verbose_level);
	// added Oct 2011:
	void (*call_back_after_reduction)(clique_finder *CF, 
		INT depth, INT nb_points, INT verbose_level);

	// added Nov 2014:
	INT f_has_print_current_choice_function;
	void (*call_back_print_current_choice)(clique_finder *CF, 
		INT depth, void *user_data, INT verbose_level);
	void *print_current_choice_data;
	
	void *call_back_clique_found_data;
	
	
	void open_tree_file(const char *fname_base, INT f_decision_nodes_only);
	void close_tree_file();
	void init(const char *label, INT n, 
		INT target_depth, 
		INT f_has_adj_list, INT *adj_list_coded, 
		INT f_has_bitvector, uchar *bitvector_adjacency, 
		INT print_interval, 
		INT f_maxdepth, INT maxdepth, 
		INT f_store_solutions, 
		INT verbose_level);
	void allocate_bitmatrix(INT verbose_level);
	void init_restrictions(INT *restrictions, INT verbose_level);
	clique_finder();
	~clique_finder();
	void null();
	void free();
	void init_point_labels(INT *pt_labels);
	void print_set(INT size, INT *set);
	void log_position_and_choice(INT depth, INT counter_save, INT counter);
	void log_position(INT depth, INT counter_save, INT counter);
	void log_choice(INT depth);
	void swap_point(INT idx1, INT idx2);
	INT degree_of_point(INT depth, INT i, INT nb_points);
	//INT degree_of_point_verbose(INT i, INT nb_points);
	INT is_suspicous(INT i);
	INT point_label(INT i);
	INT is_adjacent(INT depth, INT i, INT j);
	INT is_viable(INT depth, INT pt);
	void write_entry_to_tree_file(INT depth, INT verbose_level);
	void m_iji(INT i, INT j, INT a);
	INT s_ij(INT i, INT j);
	void backtrack_search(INT depth, INT verbose_level);
	INT solve_decision_problem(INT depth, INT verbose_level);
		// returns TRUE if we found a solution
	void get_solutions(INT *&Sol, INT &nb_solutions, INT &clique_sz, INT verbose_level);
};

void all_cliques_of_given_size(INT *Adj, INT nb_pts, INT clique_sz, INT *&Sol, INT &nb_sol, INT verbose_level);

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
	INT f_output_solution_raw;
	
	colored_graph *graph;
	clique_finder *CF;
	INT *f_color_satisfied;
	INT *color_chosen_at_depth;
	INT *color_frequency;
	INT target_depth;

	// added November 5, 2014:
	INT f_has_additional_test_function;
	void (*call_back_additional_test_function)(rainbow_cliques *R, void *user_data, 
		INT current_clique_size, INT *current_clique, 
		INT nb_pts, INT &reduced_nb_pts, 
		INT *pt_list, INT *pt_list_inv, 
		INT verbose_level);
	void *user_data;


	void search(colored_graph *graph, ofstream *fp_sol, INT f_output_solution_raw, 
		INT f_maxdepth, INT maxdepth, 
		INT f_restrictions, INT *restrictions, 
		INT f_tree, INT f_decision_nodes_only, const char *fname_tree,  
		INT print_interval, 
		INT &search_steps, INT &decision_steps, INT &nb_sol, INT &dt, 
		INT verbose_level);
	void search_with_additional_test_function(colored_graph *graph, ofstream *fp_sol, INT f_output_solution_raw, 
		INT f_maxdepth, INT maxdepth, 
		INT f_restrictions, INT *restrictions, 
		INT f_tree, INT f_decision_nodes_only, const char *fname_tree,  
		INT print_interval, 
		INT f_has_additional_test_function,
		void (*call_back_additional_test_function)(rainbow_cliques *R, void *user_data, 
			INT current_clique_size, INT *current_clique, 
			INT nb_pts, INT &reduced_nb_pts, 
			INT *pt_list, INT *pt_list_inv, 
			INT verbose_level), 
		INT f_has_print_current_choice_function,
		void (*call_back_print_current_choice)(clique_finder *CF, 
			INT depth, void *user_data, INT verbose_level), 
		void *user_data, 
		INT &search_steps, INT &decision_steps, INT &nb_sol, INT &dt, 
		INT verbose_level);
	INT find_candidates(
		INT current_clique_size, INT *current_clique, 
		INT nb_pts, INT &reduced_nb_pts, 
		INT *pt_list, INT *pt_list_inv, 
		INT *candidates, INT verbose_level);
	void clique_found(INT *current_clique, INT verbose_level);
	void clique_found_record_in_original_labels(INT *current_clique, INT verbose_level);

};

void call_back_colored_graph_clique_found(clique_finder *CF, INT verbose_level);
void call_back_colored_graph_add_point(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT pt, INT verbose_level);
void call_back_colored_graph_delete_point(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT pt, INT verbose_level);
INT call_back_colored_graph_find_candidates(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level);


