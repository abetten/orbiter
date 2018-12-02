// codes.h
//
// Anton Betten
// December 30, 2003
//

#include "orbiter.h"

typedef class code_generator code_generator;

extern int t0; // the system time when the program started

int main(int argc, const char **argv);

// #############################################################################
// code_generator:
// #############################################################################


//! classification of optimal linear codes over Fq



class code_generator {

public:

	int verbose_level;

	int f_nmk;
	int n;
	int q;
	int d;

	int f_report_schreier_trees;
	int f_report;

	int f_read_data_file;
	const char *fname_data_file;
	int depth_completed;


	char directory_path[1000];
	char prefix[1000];


	finite_field *F; // F_q

	
	int f_linear;
	int k; // for linear codes
	int nmk; // n - k
	rank_checker rc;
	int *v1; // [nmk], used by Hamming distance
	int *v2; // [nmk], used by Hamming distance
	



	int f_nonlinear;
	int N; // number of codewords for nonlinear codes
	linear_group_description *description;
	linear_group *L;

	
	strong_generators *Strong_gens;
	action *A; // PGL(n - k, q) if f_linear

	poset *Poset;
	poset_classification *gen;
			

	int f_irreducibility_test;
	int f_semilinear;
	int f_list;
	int f_table_of_nodes;


	int schreier_depth; // = 1000;
	int f_use_invariant_subset_if_available; // = TRUE;
	int f_debug; // = FALSE;
	//int f_lex; // = FALSE;
	
	int f_draw_poset;
	int f_print_data_structure;
	int f_draw_schreier_trees;

	void read_arguments(int argc, const char **argv);
	code_generator();
	~code_generator();
	void null();
	void freeself();
	void init(int argc, const char **argv);
	void main(int verbose_level);
	void print(ostream &ost, int len, int *S);
#if 0
	void early_test_func_by_using_group(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
#endif
	int Hamming_distance(int a, int b);
};

void print_code(ostream &ost, int len, int *S, void *data);




