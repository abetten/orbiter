// solver.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09

// #############################################################################
// exact_cover.C
// #############################################################################

class exact_cover {
public:

	BYTE input_prefix[1000];
	BYTE output_prefix[1000];
	BYTE solution_prefix[1000];
	BYTE base_fname[1000];

	BYTE fname_solutions[1000];
	BYTE fname_statistics[1000];

	void *user_data;

	action *A_base;
	action *A_on_blocks;
	

	void (*prepare_function_new)(exact_cover *E, INT starter_case, 
		INT *candidates, INT nb_candidates, 
		strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_label, 
		INT &f_ruled_out, 
		INT verbose_level);



	void (*early_test_func)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level);
	void *early_test_func_data;

	INT f_has_solution_test_func;
	INT (*solution_test_func)(exact_cover *EC, INT *S, INT len, 
		void *data, INT verbose_level);
	void *solution_test_func_data;

	INT f_has_late_cleanup_function;
	void (*late_cleanup_function)(exact_cover *E, 
		INT starter_case, INT verbose_level);


	INT target_size;
	
	INT f_lex;
	
	INT f_split;
	INT split_r;
	INT split_m;

	INT f_single_case;
	INT single_case;

	INT starter_size;
	INT starter_nb_cases;
	INT *starter; // [starter_size]

	INT f_randomized;
	const BYTE *random_permutation_fname;
	INT *random_permutation;

	exact_cover();
	~exact_cover();
	void null();
	void freeself();
	void init_basic(void *user_data, 
		action *A_base, action *A_on_blocks, 
		INT target_size, INT starter_size, 
		const BYTE *input_prefix, const BYTE *output_prefix, 
		const BYTE *solution_prefix, const BYTE *base_fname, 
		INT f_lex, 
		INT verbose_level);

	void init_early_test_func(
		void (*early_test_func)(INT *S, INT len, 
			INT *candidates, INT nb_candidates, 
			INT *good_candidates, INT &nb_good_candidates, 
			void *data, INT verbose_level), 
		void *early_test_func_data,
		INT verbose_level);
	void init_prepare_function_new(
		void (*prepare_function_new)(exact_cover *E, INT starter_case, 
			INT *candidates, INT nb_candidates, 
			strong_generators *Strong_gens, 
			diophant *&Dio, INT *&col_label, 
			INT &f_ruled_out, 
			INT verbose_level),
		INT verbose_level);
	void set_split(INT split_r, INT split_m, INT verbose_level);
	void set_single_case(INT single_case, INT verbose_level);
	void randomize(const BYTE *random_permutation_fname, INT verbose_level);
	void add_solution_test_function(
		INT (*solution_test_func)(exact_cover *EC, INT *S, INT len, 
		void *data, INT verbose_level), 
		void *solution_test_func_data,
		INT verbose_level);
	void add_late_cleanup_function(
		void (*late_cleanup_function)(exact_cover *E, 
		INT starter_case, INT verbose_level)
		);
	void compute_liftings_new(INT f_solve, INT f_save, INT f_read_instead, 
		INT f_draw_system, const BYTE *fname_system, 
		INT f_write_tree, const BYTE *fname_tree, INT verbose_level);
	void compute_liftings_single_case_new(INT starter_case, 
		INT f_solve, INT f_save, INT f_read_instead, 
		INT &nb_col, 
		INT *&Solutions, INT &sol_length, INT &nb_sol, 
		INT &nb_backtrack, INT &dt, 
		INT f_draw_system, const BYTE *fname_system, 
		INT f_write_tree, const BYTE *fname_tree, 
		INT verbose_level);
	void lexorder_test(INT *live_blocks2, INT &nb_live_blocks2, 
		vector_ge *stab_gens, 
		INT verbose_level);
};




// #############################################################################
// exact_cover_arguments.C:
// #############################################################################



class exact_cover_arguments {
public:
	action *A;
	action *A2;
	void *user_data;
	INT f_has_base_fname;
	const BYTE *base_fname;
	INT f_has_input_prefix;
	const BYTE *input_prefix;
	INT f_has_output_prefix;
	const BYTE *output_prefix;
	INT f_has_solution_prefix;
	const BYTE *solution_prefix;
	INT f_lift;
	INT f_starter_size;
	INT starter_size;
	INT target_size;
	INT f_lex;
	INT f_split;
	INT split_r;
	INT split_m;
	INT f_solve;
	INT f_save;
	INT f_read;
	INT f_draw_system;
	const BYTE *fname_system;
	INT f_write_tree;
	const BYTE *fname_tree;
	void (*prepare_function_new)(exact_cover *E, INT starter_case, 
		INT *candidates, INT nb_candidates, 
		strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_label, 
		INT &f_ruled_out, 
		INT verbose_level);
	void (*early_test_function)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level);
	void *early_test_function_data;
	INT f_has_solution_test_function;
	INT (*solution_test_func)(exact_cover *EC, INT *S, INT len, 
		void *data, INT verbose_level);
	void *solution_test_func_data;
	INT f_has_late_cleanup_function;
	void (*late_cleanup_function)(exact_cover *EC, 
		INT starter_case, INT verbose_level);

	INT f_randomized;
	const BYTE *random_permutation_fname;

	exact_cover_arguments();
	~exact_cover_arguments();
	void null();
	void freeself();
	void read_arguments(int argc, const char **argv, 
		INT verbose_level);
	void compute_lifts(INT verbose_level);
};


