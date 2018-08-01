// solvers.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005

// #############################################################################
// diophant.C:
// #############################################################################



class diophant {
public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;

	BYTE label[1000];
	INT m; // number of equations or inequalities
	INT n; // number of indeterminates
	INT sum; // constraint: sum(i=0..(n-1); x[i]) = sum 
	INT sum1;
	INT f_x_max;
	// with constraints: x[i] <= x_max[i] for i=0..(n-1) 
	
	INT *A; // [m][n] the coefficient matrix
	INT *G; // [m][n] matrix of gcd values
	INT *x_max; // [n] upper bounds for x
	INT *x; // [n]  current value of x
	INT *RHS; // [m] the right hand sides
	INT *RHS1;
		// [m] he current values of the RHS 
		// (=RHS - what is chosen on the left
	diophant_equation_type *type;
	BYTE **eqn_label; // [m] a label for each equation / inequality


	// the following vectors are used by diophant::test_solution
	INT *X; // [n]
	INT *Y; // [m]
	
	deque<vector<int> > _results;
	int _maxresults;
	int _resultanz;
	int _cur_result;
	INT nb_steps_betten;
	INT f_max_time;
	INT f_broken_off_because_of_maxtime;
	INT max_time_in_sec;
	INT max_time_in_ticks;
	INT t0;


	diophant();
	~diophant();
	void null();
	void freeself();
	
	void open(INT m, INT n);
	void join_problems(diophant *D1, diophant *D2, INT verbose_level);
	void init_problem_of_Steiner_type_with_RHS(INT nb_rows, 
		INT nb_cols, INT *Inc, INT nb_to_select, 
		INT *Rhs, INT verbose_level);
	void init_problem_of_Steiner_type(INT nb_rows, INT nb_cols, 
		INT *Inc, INT nb_to_select, INT verbose_level);
	void init_RHS(INT RHS_value, INT verbose_level);
	void init_clique_finding_problem(INT *Adj, INT nb_pts, 
		INT nb_to_select, INT verbose_level);
	void fill_coefficient_matrix_with(INT a);
	INT &Aij(INT i, INT j);
	INT &Gij(INT i, INT j);
	INT &RHSi(INT i);
	void init_eqn_label(INT i, BYTE *label);
	void print();
	void print_tight();
	void print_dense();
	void print2(INT f_with_gcd);
	void print_compressed();
	void print_eqn(INT i, INT f_with_gcd);
	void print_eqn_compressed(INT i);
	void print_eqn_dense(INT i);
	void print_x_long();
	void print_x(INT header);
	INT RHS_ge_zero();
	INT solve_first(INT verbose_level);
	INT solve_next();
	INT solve_first_wassermann(INT verbose_level);
	INT solve_first_mckay(INT f_once, INT verbose_level);
	void draw_solutions(const BYTE *fname, INT verbose_level);
	void write_solutions(const BYTE *fname, INT verbose_level);
	void read_solutions_from_file(const BYTE *fname_sol, 
		INT verbose_level);
	void get_solutions(INT *&Sol, INT &nb_sol, INT verbose_level);
	void get_solutions_full_length(INT *&Sol, INT &nb_sol, 
		INT verbose_level);
	void test_solution_full_length(INT *sol, INT verbose_level);
	INT solve_all_DLX(INT f_write_tree, const BYTE *fname_tree, 
		INT verbose_level);
	INT solve_all_DLX_with_RHS(INT f_write_tree, const BYTE *fname_tree, 
		INT verbose_level);
	INT solve_all_DLX_with_RHS_and_callback(INT f_write_tree, 
		const BYTE *fname_tree, 
		void (*user_callback_solution_found)(INT *sol, INT len, 
			INT nb_sol, void *data), 
		INT verbose_level);
	INT solve_all_mckay(INT &nb_backtrack_nodes, INT verbose_level);
	INT solve_once_mckay(INT verbose_level);
	INT solve_all_betten(INT verbose_level);
	INT solve_all_betten_with_conditions(INT verbose_level, 
		INT f_max_sol, INT max_sol, 
		INT f_max_time, INT max_time_in_seconds);
	INT solve_first_betten(INT verbose_level);
	INT solve_next_mckay(INT verbose_level);
	INT solve_next_betten(INT verbose_level);
	INT j_fst(INT j, INT verbose_level);
	INT j_nxt(INT j, INT verbose_level);
	void solve_mckay(const BYTE *label, INT maxresults, 
		INT &nb_backtrack_nodes, INT &nb_sol, INT verbose_level);
	void solve_mckay_override_minrhs_in_inequalities(const BYTE *label, 
		INT maxresults, INT &nb_backtrack_nodes, 
		INT minrhs, INT &nb_sol, INT verbose_level);
	void latex_it();
	void latex_it(ostream &ost);
	void trivial_row_reductions(INT &f_no_solution, INT verbose_level);
	INT count_non_zero_coefficients_in_row(INT i);
	void coefficient_values_in_row(INT i, INT &nb_values, 
		INT *&values, INT *&multiplicities, INT verbose_level);
	INT maximum_number_of_non_zero_coefficients_in_row();
	void get_coefficient_matrix(INT *&M, INT &nb_rows, INT &nb_cols, 
		INT verbose_level);
	void save_as_Levi_graph(const BYTE *fname, INT verbose_level);
	void save_in_compact_format(const BYTE *fname, INT verbose_level);
	void read_compact_format(const BYTE *fname, INT verbose_level);
	void save_in_general_format(const BYTE *fname, INT verbose_level);
	void read_general_format(const BYTE *fname, INT verbose_level);
	void save_in_wassermann_format(const BYTE *fname, INT verbose_level);
	void solve_wassermann(INT verbose_level);
	void eliminate_zero_rows_quick(INT verbose_level);
	void eliminate_zero_rows(INT *&eqn_number, INT verbose_level);
	INT is_zero_outside(INT first, INT len, INT i);
	void project(diophant *D, INT first, INT len, INT *&eqn_number, 
		INT &nb_eqns_replaced, INT *&eqns_replaced, 
		INT verbose_level);
	void multiply_A_x_to_RHS1();
	void write_xml(ostream &ost, const BYTE *label);
	void read_xml(ifstream &f, BYTE *label);
		// label will be set to the label that is in the file
		// therefore, label must point to sufficient memory
	void append_equation();
	void delete_equation(INT I);
	void write_gurobi_binary_variables(const BYTE *fname);
	void draw_it(const BYTE *fname_base, INT xmax_in, INT ymax_in, 
		INT xmax_out, INT ymax_out);
	void draw_partitioned(const BYTE *fname_base, 
		INT xmax_in, INT ymax_in, 
		INT xmax_out, INT ymax_out, 
		INT f_solution, INT *solution, INT solution_sz, 
		INT verbose_level);
	INT test_solution(INT *sol, INT len, INT verbose_level);
	void get_columns(INT *col, INT nb_col, set_of_sets *&S, 
		INT verbose_level);
	void test_solution_file(const BYTE *solution_file, 
		INT verbose_level);
	void analyze(INT verbose_level);
	INT is_of_Steiner_type();
	void make_clique_graph_adjacency_matrix(UBYTE *&Adj, 
		INT verbose_level);
	void make_clique_graph(colored_graph *&CG, INT verbose_level);
	void make_clique_graph_and_save(const BYTE *clique_graph_fname, 
		INT verbose_level);
	void test_if_the_last_solution_is_unique();
};

void diophant_callback_solution_found(INT *sol, 
	INT len, INT nb_sol, void *data);
INT diophant_solve_first_mckay(diophant *Dio, INT f_once, INT verbose_level);
INT diophant_solve_all_mckay(diophant *Dio, INT &nb_backtrack_nodes, INT verbose_level);
INT diophant_solve_once_mckay(diophant *Dio, INT verbose_level);
INT diophant_solve_next_mckay(diophant *Dio, INT verbose_level);
void diophant_solve_mckay(diophant *Dio, const BYTE *label, INT maxresults, INT &nb_backtrack_nodes, INT &nb_sol, INT verbose_level);
void diophant_solve_mckay_override_minrhs_in_inequalities(diophant *Dio, const BYTE *label, 
	INT maxresults, INT &nb_backtrack_nodes, 
	INT minrhs, INT &nb_sol, INT verbose_level);
void solve_diophant(INT *Inc, INT nb_rows, INT nb_cols, INT nb_needed, 
	INT f_has_Rhs, INT *Rhs, 
	INT *&Solutions, INT &nb_sol, INT &nb_backtrack, INT &dt, 
	INT f_DLX, 
	INT f_draw_system, const BYTE *fname_system, 
	INT f_write_tree, const BYTE *fname_tree, 
	INT verbose_level);
// allocates Solutions[nb_sol * target_size]
// where target_size = starter_size + nb_needed


// #############################################################################
// dlx.C:
// #############################################################################

extern INT *DLX_Cur_col;

void install_callback_solution_found(
	void (*callback_solution_found)(INT *solution, INT len, INT nb_sol, 
		void *data),
	void *callback_solution_found_data);
void de_install_callback_solution_found();
void DlxTest();
void DlxTransposeAppendAndSolve(INT *Data, INT nb_rows, INT nb_cols, 
	INT &nb_sol, INT &nb_backtrack, 
	INT f_write_file, const BYTE *solution_fname, 
	INT f_write_tree_file, const BYTE *tree_fname, 
	INT verbose_level);
void DlxTransposeAndSolveRHS(INT *Data, INT nb_rows, INT nb_cols, 
	INT *RHS, INT f_has_type, diophant_equation_type *type, 
	INT &nb_sol, INT &nb_backtrack, 
	INT f_write_file, const BYTE *solution_fname, 
	INT f_write_tree_file, const BYTE *tree_fname, 
	INT verbose_level);
void DlxAppendRowAndSolve(INT *Data, INT nb_rows, INT nb_cols, 
	INT &nb_sol, INT &nb_backtrack, 
	INT f_write_file, const BYTE *solution_fname, 
	INT f_write_tree_file, const BYTE *tree_fname, 
	INT verbose_level);
void DlxAppendRowAndSolveRHS(INT *Data, INT nb_rows, INT nb_cols, 
	INT *RHS, INT f_has_type, diophant_equation_type *type, 
	INT &nb_sol, INT &nb_backtrack, 
	INT f_write_file, const BYTE *solution_fname, 
	INT f_write_tree_file, const BYTE *tree_fname, 
	INT verbose_level);
void DlxSolve(INT *Data, INT nb_rows, INT nb_cols, 
	INT &nb_sol, INT &nb_backtrack, 
	INT f_write_file, const BYTE *solution_fname, 
	INT f_write_tree_file, const BYTE *tree_fname, 
	INT verbose_level);
void DlxSolve_with_RHS(INT *Data, INT nb_rows, INT nb_cols, 
	INT *RHS, INT f_has_type, diophant_equation_type *type, 
	INT &nb_sol, INT &nb_backtrack, 
	INT f_write_file, const BYTE *solution_fname, 
	INT f_write_tree_file, const BYTE *tree_fname, 
	INT verbose_level);
void DlxSearchRHS(INT k, INT verbose_level);






// #############################################################################
// mckay.C: solver for systems of diophantine equations
// #############################################################################

namespace mckay {
	// we use the MCKAY algorithm for now...


	#include <stdio.h>
	#include <math.h>

	/* bigger gets more diagnostic output */
	//#define VERBOSE 0

	#define MCKAY_DEBUG
	#define INTERVAL_IN_SECONDS 1

	typedef struct {int var,coeff;} term;
	typedef vector<term> equation;

	class tMCKAY {
	public:
		tMCKAY();
		void Init(diophant *lgs, const char *label, 
			int aEqnAnz, int aVarAnz);
		void possolve(vector<int> &lo, vector<int> &hi, 
			vector<equation> &eqn, 
			vector<int> &lorhs, vector<int> &hirhs, 
			vector<int> &neqn, int numeqn, int numvar, 
			int verbose_level);

		INT nb_calls_to_solve;
		INT first_moved;
		INT second_moved;
		const char *problem_label;

	protected:
		bool subtract(INT eqn1, equation &e1, int l1, int lors1, 
			int hirs1, INT eqn2, equation &e2, int *pl2, 
			int *plors2, int *phirs2, INT verbose_level);
		void pruneqn(vector<int> &lorhs, vector<int> &hirhs, 
			vector<equation> &eqn, vector<int> &neqn, 
			int numeqn, INT verbose_level);
		void varprune(vector<int> &lo, vector<int> &hi, 
			vector<int> &lorhs, vector<int> &hirhs, 
			vector<equation> &eqn, vector<int> &neqn, 
			int numeqn, INT verbose_level);
		void puteqns(vector<int> &lo, vector<int> &hi, 
			int numvar, 
			vector<int> &lorhs, vector<int> &hirhs, 
			vector<equation> &eqn, vector<int> &neqn, 
			int numeqn);
		int divideeqns(vector<int> &lorhs, vector<int> &hirhs, 
			vector<equation> &eqn, vector<int> &neqn, 
			int numeqn);
		int gcd(int n1,int n2);
		void solve(int level, 
			vector<int> &alo, vector<int> &ahi, 
			vector<bool> &aactive, int numvar, 
			vector<int> &lorhs, vector<int> &hirhs, 
			vector<equation> &eqn, vector<int> &neqn, 
			int numeqn, int verbose_level);
		INT restrict_variables(int level, 
			vector<int> &lo, vector<int> &hi, 
			vector<bool> &active, int numvar, 
			vector<int> &lorhs, vector<int> &hirhs, 
			vector<equation> &eqn, vector<int> &neqn, 
			int numeqn, INT &f_restriction_made, 
			int verbose_level);
		void log_12l(INT current_node, int level);

		int _eqnanz;
		int _varanz;
		vector<bool> unitcoeffs;
		vector<bool> active;
		int rekurs;
		bool _break;

		diophant *D;
		//tLGS *_lgs;

	#ifdef MCKAY_DEBUG
		vector<int> range,split,branch;
		INT ticks0;
	#endif

	};
}





