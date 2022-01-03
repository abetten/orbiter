// solvers.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


#ifndef ORBITER_SRC_LIB_FOUNDATIONS_SOLVERS_SOLVERS_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_SOLVERS_SOLVERS_H_


namespace orbiter {
namespace foundations {

// #############################################################################
// diophant_activity_description.cpp
// #############################################################################


//! to describe an activity for a diophantine system from command line arguments

class diophant_activity_description {
public:

	int f_input_file;
	std::string input_file;
	int f_print;
	int f_solve_mckay;
	int f_solve_standard;
	int f_draw_as_bitmap;
	int box_width;
	int bit_depth; // 8 or 24
	int f_draw;
	int f_perform_column_reductions;

	int f_project_to_single_equation_and_solve;
	int eqn_idx;
	int solve_case_idx;

	int f_project_to_two_equations_and_solve;
	int eqn1_idx;
	int eqn2_idx;
	int solve_case_idx_r;
	int solve_case_idx_m;

	int f_test_single_equation;
	int max_number_of_coefficients;



	diophant_activity_description();
	~diophant_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();



};


// #############################################################################
// diophant_activity.cpp
// #############################################################################


//! to perform an activity for a diophantine system using diophant_activity_description

class diophant_activity {
public:

	diophant_activity_description *Descr;

	diophant_activity();
	~diophant_activity();
	void init_from_file(diophant_activity_description *Descr,
			int verbose_level);
	void perform_activity(diophant_activity_description *Descr, diophant *Dio,
			int verbose_level);


};



// #############################################################################
// diophant_create.cpp
// #############################################################################


//! to create a diophantine system from diophant_description

class diophant_create {
public:

	diophant_description *Descr;

	diophant *D;

	diophant_create();
	~diophant_create();
	void init(
			diophant_description *Descr,
			int verbose_level);

};



// #############################################################################
// diophant_description.cpp
// #############################################################################


//! to describe a diophantine system from command line arguments

class diophant_description {
public:
	int f_q;
	int input_q;
	int f_override_polynomial;
	std::string override_polynomial;

	int f_maximal_arc;
	int maximal_arc_sz;
	int maximal_arc_d;
	std::string maximal_arc_secants_text;
	std::string external_lines_as_subset_of_secants_text;

	int f_label;
	std::string label;

	int f_coefficient_matrix;
	int coefficient_matrix_m;
	int coefficient_matrix_n;
	std::string coefficient_matrix_text;

	int f_problem_of_Steiner_type;
	int problem_of_Steiner_type_nb_t_orbits;
	std::string problem_of_Steiner_type_covering_matrix_fname;

	int f_coefficient_matrix_csv;
	std::string coefficient_matrix_csv;


	int f_RHS;
	std::string RHS_text;

	int f_RHS_csv;
	std::string RHS_csv_text;

	int f_RHS_constant;
	std::string RHS_constant_text;

	int f_x_max_global;
	int x_max_global;

	int f_x_min_global;
	int x_min_global;

	int f_x_bounds;
	std::string x_bounds_text;

	int f_x_bounds_csv;
	std::string x_bounds_csv;

	int f_has_sum;
	int has_sum;


	diophant_description();
	~diophant_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// diophant.cpp
// #############################################################################


//! diophantine systems of equations (i.e., linear systems over the integers)

//! there are three types of conditions:
//! t_EQ: equality, the sum in row i on the left must equal RHS[i]
//! t_LE: inequality, the sum in row i on the left must
//!         be less than or equal to RHS[i]
//! t_INT: interval, the sum in row i on the left must
//!         be in the closed interval RHS_low[i] and RHS[i].
//! t_ZOR: Zero or otherwise: the sum in row i on the left must
//!         be zero or equal to RHS[i]
//! Here, the sum on the left in row i means
//! the value \sum_{j=0}^{n-1} A(i,j) * x[j].



class diophant {
public:
	std::string label;
	int m; // number of equations or inequalities
	int n; // number of indeterminates
	int f_has_sum;
	int sum; // constraint: sum(i=0..(n-1); x[i]) = sum 
	int sum1;
	//int f_x_max;
	// with constraints: x[i] <= x_max[i] for i=0..(n-1) 
	
	int *A; // [m * n] the coefficient matrix
	int *G; // [m * n] matrix of gcd values
	int *x_max; // [n] upper bounds for x
	int *x_min; // [n] lower bounds for x
	int *x; // [n]  current value of x
	int *RHS; // [m] the right hand sides
	int *RHS_low; // [m] the minimum for the right hand side
		// this is used only in the McKay solver
	int *RHS1;
		// [m] he current values of the RHS 
		// (=RHS - what is chosen on the left
	diophant_equation_type *type;
	char **eqn_label; // [m] a label for each equation / inequality

	int f_has_var_labels;
	int *var_labels; // [n]


	// the following vectors are used by diophant::test_solution
	int *X; // [n]
	int *Y; // [m]
	
	std::deque<std::vector<int> > _results;
	int _maxresults;
	int _resultanz;
	int _cur_result;
	long int nb_steps_betten;
	int f_max_time;
	int f_broken_off_because_of_maxtime;
	int max_time_in_sec;
	int max_time_in_ticks;
	int t0;


	diophant();
	~diophant();
	void null();
	void freeself();
	
	void open(int m, int n);
	void init_var_labels(long int *labels, int verbose_level);
	void join_problems(diophant *D1, diophant *D2, int verbose_level);
	void init_partition_problem(
		int *weights, int nb_weights, int target_value,
		int verbose_level);
	void init_partition_problem_with_bounds(
		int *weights, int *bounds, int nb_weights, int target_value,
		int verbose_level);
	void init_problem_of_Steiner_type_with_RHS(int nb_rows, 
		int nb_cols, int *Inc, int nb_to_select, 
		int *Rhs, int verbose_level);
	void init_problem_of_Steiner_type(int nb_rows, int nb_cols, 
		int *Inc, int nb_to_select, int verbose_level);
	void init_RHS(int RHS_value, int verbose_level);
	void init_clique_finding_problem(int *Adj, int nb_pts, 
		int nb_to_select, int verbose_level);
	void fill_coefficient_matrix_with(int a);
	void set_x_min_constant(int a);
	void set_x_max_constant(int a);
	int &Aij(int i, int j);
	int &Gij(int i, int j);
	int &RHSi(int i);
	int &RHS_low_i(int i);
	void init_eqn_label(int i, char *label);
	void print();
	void print_tight();
	void print_dense();
	void print2(int f_with_gcd);
	void print_compressed();
	void print_eqn(int i, int f_with_gcd);
	void print_eqn_compressed(int i);
	void print_eqn_dense(int i);
	void print_x_long();
	void print_x(int header);
	int RHS_ge_zero();
	int solve_first(int verbose_level);
	int solve_next();
	int solve_first_mckay(int f_once, int verbose_level);
	//void draw_solutions(std::string &fname_base, int verbose_level);
	void write_solutions(std::string &fname, int verbose_level);
	void read_solutions_from_file(std::string &fname_sol,
		int verbose_level);
	void get_solutions(long int *&Sol, int &nb_sol, int verbose_level);
	void get_solutions_full_length(int *&Sol, int &nb_sol, 
		int verbose_level);
	void test_solution_full_length(int *sol, int verbose_level);
	int solve_all_DLX(int f_write_tree, const char *fname_tree, 
		int verbose_level);
	int solve_all_DLX_with_RHS(int f_write_tree, const char *fname_tree, 
		int verbose_level);
	int solve_all_DLX_with_RHS_and_callback(int f_write_tree, 
		const char *fname_tree, 
		void (*user_callback_solution_found)(int *sol, int len, 
			int nb_sol, void *data), 
		int verbose_level);
	int solve_all_mckay(long int &nb_backtrack_nodes, int maxresults, int verbose_level);
	int solve_once_mckay(int verbose_level);
	int solve_all_betten(int verbose_level);
	int solve_all_betten_with_conditions(int verbose_level, 
		int f_max_sol, int max_sol, 
		int f_max_time, int max_time_in_seconds);
	int solve_first_betten(int verbose_level);
	int solve_next_mckay(int verbose_level);
	int solve_next_betten(int verbose_level);
	int j_fst(int j, int verbose_level);
	int j_nxt(int j, int verbose_level);
	void solve_mckay(const char *label, int maxresults, 
		long int &nb_backtrack_nodes, int &nb_sol, int verbose_level);
	void solve_mckay_override_minrhs_in_inequalities(const char *label, 
		int maxresults, long int &nb_backtrack_nodes,
		int minrhs, int &nb_sol, int verbose_level);
	void latex_it();
	void latex_it(std::ostream &ost);
	void trivial_row_reductions(int &f_no_solution, int verbose_level);
	diophant *trivial_column_reductions(int verbose_level);
	int count_non_zero_coefficients_in_row(int i);
	void coefficient_values_in_row(int i, int &nb_values, 
		int *&values, int *&multiplicities, int verbose_level);
	int maximum_number_of_non_zero_coefficients_in_row();
	void get_coefficient_matrix(int *&M, int &nb_rows, int &nb_cols, 
		int verbose_level);
	void save_in_general_format(std::string &fname, int verbose_level);
	void read_general_format(std::string &fname, int verbose_level);
	void eliminate_zero_rows_quick(int verbose_level);
	void eliminate_zero_rows(int *&eqn_number, int verbose_level);
	int is_zero_outside(int first, int len, int i);
	void project(diophant *D, int first, int len, int *&eqn_number, 
		int &nb_eqns_replaced, int *&eqns_replaced, 
		int verbose_level);
	void split_by_equation(int eqn_idx, int f_solve_case, int solve_case_idx, int verbose_level);
	void split_by_two_equations(int eqn1_idx, int eqn2_idx,
			int f_solve_case, int solve_case_idx_r, int solve_case_idx_m,
			int verbose_level);
	void project_to_single_equation_and_solve(
			int max_number_of_coefficients,
			int verbose_level);
	void project_to_single_equation(diophant *D, int eqn_idx,
			int verbose_level);
	void project_to_two_equations(diophant *D, int eqn1_idx, int eqn2_idx,
			int verbose_level);
	void multiply_A_x_to_RHS1();
	void write_xml(std::ostream &ost, const char *label);
	void read_xml(std::ifstream &f, char *label, int verbose_level);
		// label will be set to the label that is in the file
		// therefore, label must point to sufficient memory
	void append_equation();
	void delete_equation(int I);
	void write_gurobi_binary_variables(const char *fname);
	void draw_as_bitmap(std::string &fname,
			int f_box_width, int box_width, int bit_depth, int verbose_level);
	void draw_it(
			std::string &fname_base,
			layered_graph_draw_options *Draw_options,
			//int xmax_in, int ymax_in, int xmax_out, int ymax_out,
			int verbose_level);
	void draw_partitioned(
			std::string &fname_base,
			layered_graph_draw_options *Draw_options,
		//int xmax_in, int ymax_in, int xmax_out, int ymax_out,
		int f_solution, int *solution, int solution_sz, 
		int verbose_level);
	int test_solution(int *sol, int len, int verbose_level);
	void get_columns(int *col, int nb_col, set_of_sets *&S, 
		int verbose_level);
	void test_solution_file(std::string &solution_file,
		int verbose_level);
	void analyze(int verbose_level);
	int is_of_Steiner_type();
	void make_clique_graph_adjacency_matrix(bitvector *&Adj,
		int verbose_level);
	void make_clique_graph(colored_graph *&CG, int verbose_level);
	void make_clique_graph_and_save(std::string &clique_graph_fname,
		int verbose_level);
	void test_if_the_last_solution_is_unique();


	int solve_first_mckay_once_option(int f_once, int verbose_level);


};

void diophant_callback_solution_found(int *sol, 
	int len, int nb_sol, void *data);
void solve_diophant(int *Inc, int nb_rows, int nb_cols, int nb_needed, 
	int f_has_Rhs, int *Rhs, 
	long int *&Solutions, int &nb_sol, long int &nb_backtrack, int &dt,
	int f_DLX, 
	//int f_draw_system, std::string &fname_system,
	int f_write_tree, std::string &fname_tree,
	int verbose_level);
// allocates Solutions[nb_sol * target_size]
// where target_size = starter_size + nb_needed


// #############################################################################
// dlx.cpp
// #############################################################################

extern int *DLX_Cur_col;

void install_callback_solution_found(
	void (*callback_solution_found)(int *solution, int len, int nb_sol, 
		void *data),
	void *callback_solution_found_data);
void de_install_callback_solution_found();
void DlxTest();
void DlxTransposeAppendAndSolve(int *Data, int nb_rows, int nb_cols, 
	int &nb_sol, int &nb_backtrack, 
	int f_write_file, const char *solution_fname, 
	int f_write_tree_file, const char *tree_fname, 
	int verbose_level);
void DlxTransposeAndSolveRHS(int *Data, int nb_rows, int nb_cols, 
	int *RHS, int f_has_type, diophant_equation_type *type, 
	int &nb_sol, int &nb_backtrack, 
	int f_write_file, const char *solution_fname, 
	int f_write_tree_file, const char *tree_fname, 
	int verbose_level);
void DlxAppendRowAndSolve(int *Data, int nb_rows, int nb_cols, 
	int &nb_sol, int &nb_backtrack, 
	int f_write_file, const char *solution_fname, 
	int f_write_tree_file, const char *tree_fname, 
	int verbose_level);
void DlxAppendRowAndSolveRHS(int *Data, int nb_rows, int nb_cols, 
	int *RHS, int f_has_type, diophant_equation_type *type, 
	int &nb_sol, int &nb_backtrack, 
	int f_write_file, const char *solution_fname, 
	int f_write_tree_file, const char *tree_fname, 
	int verbose_level);
void DlxSolve(int *Data, int nb_rows, int nb_cols, 
	int &nb_sol, int &nb_backtrack, 
	int f_write_file, const char *solution_fname, 
	int f_write_tree_file, const char *tree_fname, 
	int verbose_level);
void DlxSolve_with_RHS(int *Data, int nb_rows, int nb_cols, 
	int *RHS, int f_has_type, diophant_equation_type *type, 
	int &nb_sol, int &nb_backtrack, 
	int f_write_file, const char *solution_fname, 
	int f_write_tree_file, const char *tree_fname, 
	int verbose_level);
void DlxSearchRHS(int k, int verbose_level);






// #############################################################################
// mckay.cpp
// #############################################################################


#define MCKAY_DEBUG


//! a solver for systems of diophantine equations


namespace mckay {
	// we use the MCKAY algorithm for now...


	#include <stdio.h>
	#include <math.h>

	/* bigger gets more diagnostic output */
	//#define VERBOSE 0

	//#undef MCKAY_DEBUG
	#define INTERVAL_IN_SECONDS 1

	//! a term in a diophantine system of type tMCKAY

	typedef struct {int var,coeff;} term;
	typedef std::vector<term> equation;

	//! solves diophantine systems according to McKay

	class tMCKAY {
	public:
		tMCKAY();
		void Init(diophant *lgs, const char *label, 
			int aEqnAnz, int aVarAnz);
		void possolve(std::vector<int> &lo, std::vector<int> &hi,
				std::vector<equation> &eqn,
				std::vector<int> &lorhs, std::vector<int> &hirhs,
				std::vector<int> &neqn, int numeqn, int numvar,
			int verbose_level);

		long int nb_calls_to_solve;
		int first_moved;
		int second_moved;
		const char *problem_label;

	protected:
		bool subtract(int eqn1, equation &e1, int l1, int lors1, 
			int hirs1, int eqn2, equation &e2, int *pl2, 
			int *plors2, int *phirs2, int verbose_level);
		void pruneqn(std::vector<int> &lo, std::vector<int> &hi,
				int numvar,
				std::vector<int> &lorhs, std::vector<int> &hirhs,
				std::vector<equation> &eqn, std::vector<int> &neqn,
				int numeqn, int verbose_level);
		void varprune(std::vector<int> &lo, std::vector<int> &hi,
				std::vector<int> &lorhs, std::vector<int> &hirhs,
				std::vector<equation> &eqn, std::vector<int> &neqn,
			int numeqn, int verbose_level);
		void puteqns(std::vector<int> &lo, std::vector<int> &hi,
			int numvar, 
			std::vector<int> &lorhs, std::vector<int> &hirhs,
			std::vector<equation> &eqn, std::vector<int> &neqn,
			int numeqn);
		int divideeqns(std::vector<int> &lorhs, std::vector<int> &hirhs,
				std::vector<equation> &eqn, std::vector<int> &neqn,
			int numeqn);
		int gcd(int n1,int n2);
		void solve(int level, 
				std::vector<int> &alo, std::vector<int> &ahi,
				std::vector<bool> &aactive, int numvar,
				std::vector<int> &lorhs, std::vector<int> &hirhs,
				std::vector<equation> &eqn, std::vector<int> &neqn,
			int numeqn, int verbose_level);
		int restrict_variables(int level, 
				std::vector<int> &lo, std::vector<int> &hi,
				std::vector<bool> &active, int numvar,
				std::vector<int> &lorhs, std::vector<int> &hirhs,
				std::vector<equation> &eqn, std::vector<int> &neqn,
			int numeqn, int &f_restriction_made, 
			int verbose_level);
		void log_12l(long int current_node, int level);

		int _eqnanz;
		int _varanz;
		std::vector<bool> unitcoeffs;
		std::vector<bool> active;
		int rekurs;
		bool _break;

		diophant *D;
		//tLGS *_lgs;

	#ifdef MCKAY_DEBUG
		std::vector<int> range,split,branch;
		int ticks0;
	#endif

	};
}


}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_SOLVERS_SOLVERS_H_ */





