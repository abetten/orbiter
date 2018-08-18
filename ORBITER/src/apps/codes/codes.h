// codes.h
//
// Anton Betten
// December 30, 2003
//

#include "orbiter.h"

typedef class code_generator code_generator;

extern INT t0; // the system time when the program started

int main(int argc, const char **argv);

// #############################################################################
// code_generator:
// #############################################################################

class code_generator {

public:

	INT verbose_level;

	INT f_nmk;
	INT n;
	INT q;
	INT d;

	BYTE directory_path[1000];
	BYTE prefix[1000];


	finite_field *F; // F_q

	
	INT f_linear;
	INT k; // for linear codes
	INT nmk; // n - k
	rank_checker rc;
	INT *v1; // [nmk], used by Hamming distance
	INT *v2; // [nmk], used by Hamming distance
	



	INT f_nonlinear;
	INT N; // number of codewords for nonlinear codes
	linear_group_description *description;
	linear_group *L;

	
	strong_generators *Strong_gens;
	action *A; // PGL(n - k, q) if f_linear

	generator *gen;
			

	INT f_irreducibility_test;
	INT f_semilinear;
	INT f_list;
	INT f_table_of_nodes;


	INT schreier_depth; // = 1000;
	INT f_use_invariant_subset_if_available; // = TRUE;
	INT f_debug; // = FALSE;
	//INT f_lex; // = FALSE;
	
	INT f_draw_poset;
	INT f_print_data_structure;
	INT f_draw_schreier_trees;

	void read_arguments(int argc, const char **argv);
	code_generator();
	~code_generator();
	void null();
	void freeself();
	void init(int argc, const char **argv);
	void print(INT len, INT *S);
	void main(INT verbose_level);
	void early_test_func_by_using_group(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	INT Hamming_distance(INT a, INT b);
};

void check_mindist_early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, 
	INT verbose_level);
INT check_mindist(INT len, INT *S, void *data, INT verbose_level);
INT check_mindist_incremental(INT len, INT *S, 
	void *data, INT verbose_level);
void print_code(INT len, INT *S, void *data);




