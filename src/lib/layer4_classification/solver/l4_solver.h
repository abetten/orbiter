// solver.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09



#ifndef ORBITER_SRC_LIB_TOP_LEVEL_SOLVER_SOLVER_H_
#define ORBITER_SRC_LIB_TOP_LEVEL_SOLVER_SOLVER_H_


namespace orbiter {
namespace layer4_classification {
namespace solvers_package {


// #############################################################################
// exact_cover.cpp
// #############################################################################


//! exact cover problems arising with the lifting of combinatorial objects


class exact_cover {
public:

	std::string input_prefix;
	std::string output_prefix;
	std::string solution_prefix;
	std::string base_fname;

	std::string fname_solutions;
	std::string fname_statistics;

	void *user_data;

	actions::action *A_base;
	actions::action *A_on_blocks;
	

	void (*prepare_function_new)(
			exact_cover *E, int starter_case,
		long int *candidates, int nb_candidates,
		groups::strong_generators *Strong_gens,
		solvers::diophant *&Dio, long int *&col_label,
		int &f_ruled_out, 
		int verbose_level);



	void (*early_test_func)(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level);
	void *early_test_func_data;

	int f_has_solution_test_func;
	int (*solution_test_func)(
			exact_cover *EC, long int *S, int len,
		void *data, int verbose_level);
	void *solution_test_func_data;

	int f_has_late_cleanup_function;
	void (*late_cleanup_function)(
			exact_cover *E,
		int starter_case, int verbose_level);


	int target_size;
	
	int f_lex;
	
	int f_split;
	int split_r;
	int split_m;

	int f_single_case;
	int single_case;

	int starter_size;
	int starter_nb_cases;
	long int *starter; // [starter_size]

	int f_randomized;
	std::string random_permutation_fname;
	int *random_permutation;

	exact_cover();
	~exact_cover();
	void null();
	void freeself();
	void init_basic(
			void *user_data,
			actions::action *A_base,
			actions::action *A_on_blocks,
		int target_size, int starter_size, 
		std::string &input_prefix,
		std::string &output_prefix,
		std::string &solution_prefix,
		std::string &base_fname,
		int f_lex, 
		int verbose_level);

	void init_early_test_func(
		void (*early_test_func)(long int *S, int len,
				long int *candidates, int nb_candidates,
				long int *good_candidates, int &nb_good_candidates,
			void *data, int verbose_level), 
		void *early_test_func_data,
		int verbose_level);
	void init_prepare_function_new(
		void (*prepare_function_new)(
				solvers_package::exact_cover *E, int starter_case,
				long int *candidates, int nb_candidates,
				groups::strong_generators *Strong_gens,
			solvers::diophant *&Dio, long int *&col_label,
			int &f_ruled_out, 
			int verbose_level),
		int verbose_level);
	void set_split(
			int split_r, int split_m, int verbose_level);
	void set_single_case(
			int single_case, int verbose_level);
	void randomize(
			std::string &random_permutation_fname,
			int verbose_level);
	void add_solution_test_function(
		int (*solution_test_func)(
				exact_cover *EC, long int *S, int len,
		void *data, int verbose_level), 
		void *solution_test_func_data,
		int verbose_level);
	void add_late_cleanup_function(
		void (*late_cleanup_function)(
				exact_cover *E,
		int starter_case, int verbose_level)
		);
	void compute_liftings_new(
			int f_solve, int f_save, int f_read_instead,
		int f_draw_system, std::string &draw_options_label,
		std::string &fname_system,
		int f_write_tree,
		std::string &fname_tree, int verbose_level);
	void compute_liftings_single_case_new(
			int starter_case,
		int f_solve, int f_save, int f_read_instead, 
		int &nb_col, 
		int *&Solutions, int &sol_length, int &nb_sol,
		int &nb_backtrack, int &dt, 
		int f_draw_system, std::string &draw_options_label,
		std::string &fname_system,
		int f_write_tree, std::string &fname_tree,
		int verbose_level);
	void lexorder_test(
			long int *live_blocks2, int &nb_live_blocks2,
			data_structures_groups::vector_ge *stab_gens,
		int verbose_level);
};




// #############################################################################
// exact_cover_arguments.cpp
// #############################################################################


//! command line arguments to control the lifting via exact cover



class exact_cover_arguments {
public:
	actions::action *A;
	actions::action *A2;
	void *user_data;
	int f_has_base_fname;
	std::string base_fname;
	int f_has_input_prefix;
	std::string input_prefix;
	int f_has_output_prefix;
	std::string output_prefix;
	int f_has_solution_prefix;
	std::string solution_prefix;
	int f_lift;
	int f_starter_size;
	int starter_size;
	int target_size;
	int f_lex;
	int f_split;
	int split_r;
	int split_m;
	int f_solve;
	int f_save;
	int f_read;
	int f_draw_system;
	std::string draw_options;
	std::string fname_system;
	int f_write_tree;
	std::string fname_tree;
	void (*prepare_function_new)(
			exact_cover *E, int starter_case,
		long int *candidates, int nb_candidates,
		groups::strong_generators *Strong_gens,
		solvers::diophant *&Dio, long int *&col_label,
		int &f_ruled_out, 
		int verbose_level);
	void (*early_test_function)(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level);
	void *early_test_function_data;
	int f_has_solution_test_function;
	int (*solution_test_func)(
			exact_cover *EC, long int *S, int len,
		void *data, int verbose_level);
	void *solution_test_func_data;
	int f_has_late_cleanup_function;
	void (*late_cleanup_function)(exact_cover *EC, 
		int starter_case, int verbose_level);

	int f_randomized;
	std::string random_permutation_fname;

	exact_cover_arguments();
	~exact_cover_arguments();
	void null();
	void freeself();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void compute_lifts(
			int verbose_level);
};

}}}



#endif /* ORBITER_SRC_LIB_TOP_LEVEL_SOLVER_SOLVER_H_ */



