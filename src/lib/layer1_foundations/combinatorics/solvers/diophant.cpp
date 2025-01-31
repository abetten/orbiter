// diophant.cpp
//
// Anton Betten
// September 18, 2000
//
// moved to GALOIS: April 16, 2015

#include "foundations.h"


using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace solvers {



static void (*diophant_user_callback_solution_found)(int *sol,
	int len, int nb_sol, void *data) = NULL;

static void diophant_callback_solution_found(
		int *sol,
	int len, int nb_sol, void *data);


diophant::diophant()
{
	Record_birth();
	//std::string label;

	m = 0;
	n = 0;
	f_has_sum = false;
	sum = sum1 = 0;

	//f_x_max = false;

	A = NULL;
	G = NULL;
	x_max = NULL;
	x_min = NULL;
	x = NULL;
	RHS = NULL;
	RHS_low = NULL;
	RHS1 = NULL;
	type = NULL;
	eqn_label = NULL;

	f_has_var_labels = false;
	var_labels = NULL;

	X = false;
	Y = false;

	// results
	_maxresults = 0;
	_resultanz = 0;
	_cur_result = 0;
	nb_steps_betten = 0;
	f_max_time = false;
	f_broken_off_because_of_maxtime = false;
	max_time_in_sec = 0;
	max_time_in_ticks = 0;
	t0 = 0;

	//null();
}


diophant::~diophant()
{
	Record_death();
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::~diophant" << endl;
	}
	if (A) {
		FREE_int(A);
	}
	if (G) {
		FREE_int(G);
	}
	if (x) {
		FREE_int(x);
	}
	if (x_max) {
		FREE_int(x_max);
	}
	if (x_min) {
		FREE_int(x_min);
	}
	if (f_v) {
		cout << "diophant::~diophant before RHS" << endl;
	}
	if (RHS) {
		FREE_int(RHS);
	}
	if (f_v) {
		cout << "diophant::~diophant before RHS_low" << endl;
	}
	if (RHS_low) {
		FREE_int(RHS_low);
	}
	if (f_v) {
		cout << "diophant::~diophant before RHS1" << endl;
	}
	if (RHS1) {
		FREE_int(RHS1);
	}
	if (type) {
		FREE_OBJECT(type);
	}
	if (f_v) {
		cout << "diophant::~diophant before eqn_label" << endl;
	}
	if (eqn_label) {
		delete [] eqn_label;
		eqn_label = NULL;
	}
	if (X) {
		FREE_int(X);
	}
	if (Y) {
		FREE_int(Y);
	}
	if (f_has_var_labels) {
		FREE_int(var_labels);
	}
	if (f_v) {
		cout << "diophant::~diophant done" << endl;
	}
}

void diophant::open(
		int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::open" << endl;
	}
	int i;
	
	A = NEW_int(m * n);
	Int_vec_zero(A, m * n);
	G = NEW_int(m * n);
	Int_vec_zero(G, m * n);
	x = NEW_int(n);
	Int_vec_zero(x, n);
	x_max = NEW_int(n);
	Int_vec_zero(x_max, n);
	x_min = NEW_int(n);
	Int_vec_zero(x_min, n);
	RHS = NEW_int(m);
	Int_vec_zero(RHS, m);
	RHS_low = NEW_int(m);
	RHS1 = NEW_int(m);
	type = NEW_OBJECTS(diophant_equation_type, m);
	eqn_label = new string[m];
	X = NEW_int(n);
	Y = NEW_int(m);
	
	for (i = 0; i < n; i++) {
		x_max[i] = 0;
	}
	label[0] = 0;
	diophant::m = m;
	diophant::n = n;
	f_has_sum = false;
	//f_x_max = false;
	f_max_time = false;
	f_has_var_labels = false;
}

void diophant::init_var_labels(
		long int *labels, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::init_var_labels" << endl;
	}
	var_labels = NEW_int(n);
	f_has_var_labels = true;
	Lint_vec_copy_to_int(labels, var_labels, n);
	if (f_v) {
		cout << "diophant::init_var_labels done" << endl;
	}

}
void diophant::join_problems(
		diophant *D1, diophant *D2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_rows, nb_cols;
	int nb_r1, nb_r2;
	int i, j;

	if (f_v) {
		cout << "diophant::join_problems" << endl;
	}
	if (D1->n != D2->n) {
		cout << "D1->n != D2->n" << endl;
		exit(1);
	}
	if (D1->sum != D2->sum) {
		cout << "D1->sum != D2->sum" << endl;
		exit(1);
	}
	if (!D1->f_has_sum) {
		cout << "!D1->f_has_sum" << endl;
		exit(1);
	}
	if (!D2->f_has_sum) {
		cout << "!D2->f_has_sum" << endl;
		exit(1);
	}
#if 0
	if (D1->f_x_max != D2->f_x_max) {
		cout << "D1->f_x_max != D2->f_x_max" << endl;
		exit(1);
	}
#endif
	nb_cols = D1->n;
	nb_r1 = D1->m;
	nb_r2 = D2->m;
	nb_rows = nb_r1 + nb_r2;
	if (f_v) {
		cout << "diophant::join_problems before open" << endl;
	}
	open(nb_rows, nb_cols, verbose_level - 1);
	if (f_v) {
		cout << "diophant::join_problems after open" << endl;
	}
	f_has_sum = true;
	sum = D1->sum;
	//f_x_max = D1->f_x_max;
#if 0
	if (f_x_max) {
		for (i = 0; i < nb_cols; i++) {
			if (D1->x_max[i] != D2->x_max[i]) {
				cout << "D1->x_max[i] != D2->x_max[i]" << endl;
				exit(1);
			}
			x_max[i] = D1->x_max[i];
		}
	}
#else
	for (i = 0; i < nb_cols; i++) {
		x_max[i] = MINIMUM(D1->x_max[i], D2->x_max[i]);
		x_min[i] = MAXIMUM(D1->x_min[i], D2->x_min[i]);
	}
#endif
	for (i = 0; i < nb_r1; i++) {
		for (j = 0; j < nb_cols; j++) {
			Aij(i, j) = D1->Aij(i, j);
		}
		type[i] = D1->type[i];
		RHSi(i) = D1->RHSi(i);
		RHS_low_i(i) = D1->RHS_low_i(i);
	}
	for (i = 0; i < nb_r2; i++) {
		for (j = 0; j < nb_cols; j++) {
			Aij(nb_r1 + i, j) = D2->Aij(i, j);
		}
		type[nb_r1 + i] = D2->type[i];
		RHSi(nb_r1 + i) = D2->RHSi(i);
		RHS_low_i(nb_r1 + i) = D2->RHS_low_i(nb_r1 + i);
	}
	if (f_v) {
		cout << "diophant::join_problems done" << endl;
	}
}

void diophant::init_partition_problem(
	int *weights, int nb_weights, int target_value,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;

	if (f_v) {
		cout << "diophant::init_partition_problem" << endl;
	}
	open(1, nb_weights, verbose_level - 1);
	for (j = 0; j < nb_weights; j++) {
		x_max[j] = target_value / weights[j];
		x_min[j] = 0;
	}
	//f_x_max = true;
	f_has_sum = false;
	//sum = nb_to_select;
	for (j = 0; j < nb_weights; j++) {
		Aij(0, j) = weights[j];
	}
	RHSi(0) = target_value;
	RHS_low_i(0) = 0; // not used
	if (f_v) {
		cout << "diophant::init_partition_problem" << endl;
	}
}

void diophant::init_partition_problem_with_bounds(
	int *weights, int *bounds,
	int nb_weights, int target_value,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;

	if (f_v) {
		cout << "diophant::init_partition_problem_with_bounds" << endl;
	}
	open(1, nb_weights, verbose_level - 1);
	for (j = 0; j < nb_weights; j++) {
		x_max[j] = bounds[j]; // target_value / weights[j];
		x_min[j] = 0;
	}
	//f_x_max = true;
	f_has_sum = false;
	//sum = nb_to_select;
	for (j = 0; j < nb_weights; j++) {
		Aij(0, j) = weights[j];
	}
	RHSi(0) = target_value;
	RHS_low_i(0) = 0; // not used
	if (f_v) {
		cout << "diophant::init_partition_problem_with_bounds" << endl;
	}
}



void diophant::init_problem_of_Steiner_type_with_RHS(
	int nb_rows, int nb_cols, int *Inc, int nb_to_select,
	int *Rhs, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "diophant::init_problem_of_Steiner_type_with_RHS" << endl;
	}
	open(nb_rows, nb_cols, verbose_level - 1);
	for (j = 0; j < nb_cols; j++) {
		x_max[j] = 1;
		x_min[j] = 0;
	}
	//f_x_max = true;
	f_has_sum = true;
	sum = nb_to_select;
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Aij(i, j) = Inc[i * nb_cols + j];
		}
		RHSi(i) = Rhs[i];
		RHS_low_i(i) = 0; // not used
	}
	if (f_v) {
		cout << "diophant::init_problem_of_Steiner_type_with_RHS done" << endl;
	}
}

void diophant::init_problem_of_Steiner_type(
	int nb_rows, int nb_cols, int *Inc, int nb_to_select,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "diophant::init_problem_of_Steiner_type" << endl;
	}
	open(nb_rows, nb_cols, verbose_level - 1);
	for (j = 0; j < nb_cols; j++) {
		x_max[j] = 1;
		x_min[j] = 0;
	}
	//f_x_max = true;
	f_has_sum = true;
	sum = nb_to_select;
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Aij(i, j) = Inc[i * nb_cols + j];
		}
		RHSi(i) = 1;
		RHS_low_i(i) = 0; // not used
	}
	if (f_v) {
		cout << "diophant::init_problem_of_Steiner_type done" << endl;
	}
}

void diophant::init_RHS(
		int RHS_value, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "diophant::init_RHS" << endl;
	}
	for (i = 0; i < m; i++) {
		RHSi(i) = RHS_value;
	}
	if (f_v) {
		cout << "diophant::init_RHS done" << endl;
	}
}

void diophant::init_clique_finding_problem(
		int *Adj, int nb_pts,
	int nb_to_select, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, i1, i2;
	int nb_zeros = 0, nb_ones = 0, total;

	if (f_v) {
		cout << "diophant::init_clique_finding_problem" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		for (j = i + 1; j < nb_pts; j++) {
			if (Adj[i * nb_pts + j]) {
				nb_ones++;
			}
			else {
				nb_zeros++;
			}
		}
	}
	total = nb_ones + nb_zeros;
	if (f_v) {
		cout << "nb_zeros=" << nb_zeros << endl;
		cout << "nb_ones =" << nb_ones << endl;
		total = nb_zeros + nb_ones;
		cout << "edge density = " <<
				(double)nb_ones / (double)total << endl;
	}
	open(nb_zeros, nb_pts, verbose_level - 1);
	for (j = 0; j < nb_pts; j++) {
		x_max[j] = 1;
		x_min[j] = 0;
	}
	//f_x_max = true;
	f_has_sum = true;
	sum = nb_to_select;
	i = 0;
	for (i1 = 0; i1 < nb_pts; i1++) {
		for (i2 = i1 + 1; i2 < nb_pts; i2++) {
			if (Adj[i1 * nb_pts + i2] == 0) {
				Aij(i, i1) = 1;
				Aij(i, i2) = 1;
				type[i] = t_LE;
				RHSi(i) = 1;
				RHS_low_i(i) = 0; // not used
				i++;
			}
		}
	}
	if (f_v) {
		cout << "diophant::init_clique_finding_problem done" << endl;
	}
}


void diophant::fill_coefficient_matrix_with(
		int a)
{
	int i;
	
	for (i = 0; i < m * n; i++) {
		A[i] = a;
	}
}

void diophant::set_x_min_constant(
		int a)
{
	int j;

	for (j = 0; j < n; j++) {
		x_min[j] = a;
	}
}

void diophant::set_x_max_constant(
		int a)
{
	int j;

	for (j = 0; j < n; j++) {
		x_max[j] = a;
	}
}

int &diophant::Aij(
		int i, int j)
{
	if (i >= m) {
		cout << "diophant::Aij i >= m" << endl;
		cout << "i=" << i << endl;
		cout << "m=" << m << endl;
		exit(1);
	}
	if (j >= n) {
		cout << "diophant::Aij j >= n" << endl;
		cout << "j=" << j << endl;
		cout << "n=" << n << endl;
		exit(1);
	}
	return A[i * n + j];
}

int &diophant::Gij(
		int i, int j)
{
	if (i >= m) {
		cout << "diophant::Gij i >= m" << endl;
		cout << "i=" << i << endl;
		cout << "m=" << m << endl;
		exit(1);
	}
	if (j >= n) {
		cout << "diophant::Gij j >= n" << endl;
		cout << "j=" << j << endl;
		cout << "n=" << n << endl;
		exit(1);
	}
	return G[i * n + j];
}

int &diophant::RHSi(
		int i)
{
	if (i >= m) {
		cout << "diophant::RHSi i >= m" << endl;
		exit(1);
	}
	return RHS[i];
}

int &diophant::RHS_low_i(
		int i)
{
	if (i >= m) {
		cout << "diophant::RHS_low_i i >= m" << endl;
		exit(1);
	}
	return RHS_low[i];
}

void diophant::init_eqn_label(
		int i, std::string &label)
{
	if (i >= m) {
		cout << "diophant::init_eqn_label i >= m" << endl;
		cout << "label: " << label << endl;
		cout << "i=" << i << endl;
		exit(1);
	}
	eqn_label[i] = label;
}

void diophant::print()
{
	print2(false);
}

void diophant::print_tight()
{
	int i, j, s, c;
	for (i = 0; i < m; i++) {
		s = 0;
		for (j = 0; j < n; j++) {
			c = Aij(i, j);
			s += c;
			cout << setw(1) << c;
		}
		if (type[i] == t_EQ) {
			cout << " = " << RHS[i];
		}
		else if (type[i] == t_LE) {
			cout << " <= " << RHS[i];
		}
		else if (type[i] == t_INT) {
			cout << " in [" << RHS_low_i(i) << "," << RHS[i] << "]";
		}
		else if (type[i] == t_ZOR) {
			cout << " ZOR " << RHS[i];
		}
		cout << " (rowsum=" << s << ")" << endl;
	}
	if (f_has_sum) {
		cout << "sum = " << sum << endl;
	}
	else {
		cout << "there is no restriction on the sum" << endl;
	}
}

void diophant::print2(
		int f_with_gcd)
{
	int i, j;
	
	cout << "diophant with m=" << m << " n=" << n << endl;
	for (i = 0; i < m; i++) {
		print_eqn(i, f_with_gcd);
	}
	//if (f_x_max) {
		for (j = 0; j < n; j++) {
			cout << x_min[j] << " \\le x_{" << j << "} \\le " << x_max[j] << ", ";
		}
		cout << endl;
	//}
	if (f_has_sum) {
		cout << "sum = " << sum << endl;
	}
	else {
		cout << "there is no condition on the sum of x_i" << endl;
	}
	//latex_it();
}

void diophant::print_dense()
{
	int i, j;
	
	cout << "diophant with m=" << m << " n=" << n << endl;
	for (i = 0; i < m; i++) {
		print_eqn_dense(i);
	}
	//if (f_x_max) {
		for (j = 0; j < n; j++) {
			cout << x_min[j];
			cout << x_max[j];
		}
		cout << endl;
	//}
	if (f_has_sum) {
		cout << "sum = " << sum << endl;
	}
	else {
		cout << "there is no condition on the sum of x_i" << endl;
	}
	//latex_it();
}

void diophant::print_compressed()
{
	int i, j;
	
	cout << "diophant with m=" << m << " n=" << n << endl;
	for (i = 0; i < m; i++) {
		print_eqn_compressed(i);
	}
	//if (f_x_max) {
		for (j = 0; j < n; j++) {
			cout << x_min[j] << " \\le x_{" << j << "} \\le " << x_max[j] << ", ";
		}
		cout << endl;
	//}
	if (f_has_sum) {
		cout << "sum = " << sum << endl;
	}
	else {
		cout << "there is no condition on the sum of x_i" << endl;
	}
}


void diophant::print_eqn(
		int i, int f_with_gcd)
{
	int j;
	
	for (j = 0; j < n; j++) {
		cout << setw(3) << Aij(i, j) << " ";
		if (f_with_gcd) {
			cout << "|" << setw(3) << Gij(i, j) << " ";
		}
	}
	if (type[i] == t_EQ) {
		cout << " = " << setw(3) << RHSi(i);
	}
	else if (type[i] == t_LE) {
		cout << " <= " << setw(3) << RHSi(i);
	}
	else if (type[i] == t_INT) {
		cout << " in [" << RHS_low_i(i) << ", " << setw(3) << RHSi(i) << "]";
	}
	else if (type[i] == t_ZOR) {
		cout << " ZOR " << setw(3) << RHSi(i);
	}
	cout << " ";
	cout << eqn_label[i];
	cout << endl;
}

void diophant::print_eqn_compressed(
		int i)
{
	int j;
	
	for (j = 0; j < n; j++) {
		if (Aij(i, j) == 0) {
			continue;
		}
		if (Aij(i, j) == 1) {
			cout << " + x_{" << j << "} ";
		}
		else {
			cout << " + " << setw(3) 
				<< Aij(i, j) << " * x_{" << j << "} ";
		}
	}
	if (type[i] == t_EQ) {
		cout << " = ";
	}
	else if (type[i] == t_LE) {
		cout << " <= ";
	}
	else if (type[i] == t_INT) {
		cout << " in [,]";
	}
	else if (type[i] == t_ZOR) {
		cout << " ZOR ";
	}
	cout << setw(3) << RHSi(i) << " ";
	cout << eqn_label[i];
	cout << endl;
}

void diophant::print_eqn_dense(
		int i)
{
	int j;
	
	for (j = 0; j < n; j++) {
		cout << Aij(i, j);
	}
	if (type[i] == t_EQ) {
		cout << " = " << setw(3) << RHSi(i);
	}
	else if (type[i] == t_LE) {
		cout << " <= " << setw(3) << RHSi(i);
	}
	else if (type[i] == t_INT) {
		cout << " in [" << RHS_low_i(i) << "," << setw(3) << RHSi(i) << "]";
	}
	else if (type[i] == t_ZOR) {
		cout << " ZOR " << setw(3) << RHSi(i);
	}
	cout << " ";
	cout << eqn_label[i];
	cout << endl;
}

void diophant::print_x_long()
{
	int j;
	
	for (j = 0; j < n; j++) {
		cout << "x_{" << j << "} = " << x[j] << endl;
	}
}

void diophant::print_x(
		int header)
{
	int j;
	
	cout << setw(5) << header << " : ";
	for (j = 0; j < n; j++) {
		cout << setw(3) << x[j] << " ";
	}
	cout << endl;
}

int diophant::RHS_ge_zero()
{
	int k;
	
	for (k = 0; k < m; k++) {
		if (RHS1[k] < 0) {
			return false;
		}
	}
	return true;
}


int diophant::solve_first(
		int verbose_level)
{

#if 0
	if (false/*n >= 50*/) {
		return solve_first_wassermann(verbose_level);
	}
#endif

	if (true) {
		return solve_first_betten(verbose_level);
	}
	else {
		//cout << "diophant::solve_first
		//solve_first_mckay is disabled" << endl;
		return solve_first_mckay(false, verbose_level);
	}
}

int diophant::solve_next()
{
	return solve_next_betten(0);
	//return solve_next_mckay();
}


#if 0
int diophant::solve_first_wassermann(int verbose_level)
{
	solve_wassermann(verbose_level);
	exit(1);
}
#endif


int diophant::solve_first_mckay(
		int f_once, int verbose_level)
{
	int f_v = true;//(verbose_level >= 1);
	int j;
	//int maxresults = 10000000;
	int maxresults = INT_MAX;
	vector<int> res;
	long int nb_backtrack_nodes;
	int nb_sol;

	//verbose_level = 4;
	if (f_v) {
		cout << "diophant::solve_first_mckay "
				"calling solve_mckay" << endl;
	}
	if (f_once) {
		maxresults = 1;
	}

	string label;

	solve_mckay(
			label, maxresults, nb_backtrack_nodes, nb_sol,
			verbose_level - 2);
	if (f_v) {
		cout << "diophant::solve_first_mckay found " << _resultanz
			<< " solutions, using " << nb_backtrack_nodes
			<< " backtrack nodes" << endl;
	}
	_cur_result = 0;
	if (_resultanz == 0) {
		return false;
	}
	res = _results[_cur_result]; //.front();
	for (j = 0; j < n; j++) {
		x[j] = res[j];
	}
	//_results.pop_front();
	_cur_result++;
	if (f_v) {
		cout << "diophant::solve_first_mckay done" << endl;
	}
	return true;
}


void diophant::write_solutions_full_length(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "diophant::write_solutions_full_length" << endl;
	}


	int *Sol;
	int nb_sol, width;

	width = n;

	get_solutions(
			Sol,
			nb_sol, 0 /*verbose_level*/);

	Fio.Csv_file_support->int_matrix_write_csv(fname, Sol, nb_sol, width);


	if (f_v) {
		cout << "diophant::write_solutions_full_length "
				"written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "diophant::write_solutions_full_length done" << endl;
	}
}

void diophant::write_solutions_index_set(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "diophant::write_solutions_index_set" << endl;
	}


	int *Sol;
	int nb_sol, width;

	width = n;

	get_solutions(
			Sol,
			nb_sol, 0 /*verbose_level*/);

	//Fio.Csv_file_support->int_matrix_write_csv(fname, Sol, nb_sol, width);

	Fio.write_solutions_as_index_set(
			fname, Sol, nb_sol, width, sum,
			verbose_level);

	if (f_v) {
		cout << "diophant::write_solutions_index_set "
				"written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "diophant::write_solutions_index_set done" << endl;
	}
}


void diophant::read_solutions_from_file(
		std::string &fname_sol,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	vector<int> res;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "diophant::read_solutions_from_file" << endl;
	}
	if (f_v) {
		cout << "diophant::read_solutions_from_file reading file "
				<< fname_sol << " of size "
				<< Fio.file_size(fname_sol) << endl;
	}
	if (Fio.file_size(fname_sol) <= 0) {
		cout << "diophant::read_solutions_from_file file "
				<< fname_sol << " does not exist" << endl;
		exit(1);
	}

	{
		ifstream fp(fname_sol);
		int N, s, h;

		fp >> N >> s;

		sum = s;
		cout << "diophant::read_solutions_from_file reading " << N
				<< " solutions of length " << sum << endl;
		_resultanz = 0;
		for (i = 0; i < N; i++) {
			res.resize(n);
			for (j = 0; j < n; j++) {
				res[j] = 0;
			}
			for (h = 0; h < n; h++) {
				fp >> res[h];
			}
			_results.push_back(res);
			_resultanz++;
		}
	
	}
	if (f_v) {
		cout << "diophant::read_solutions_from_file read " << _resultanz
			<< " solutions from file " << fname_sol << " of size "
			<< Fio.file_size(fname_sol) << endl;
	}
}

void diophant::get_solutions_index_set(
		int *&Sol, int &nb_sol, int verbose_level)
// allocates Sol[nb_sol * sum]
{
	int f_v = (verbose_level >= 1);
	int i, j, cnt;
	vector<int> res;

	if (f_v) {
		cout << "diophant::get_solutions" << endl;
		cout << "nb_sol = " << _resultanz << endl;
		cout << "sum = " << sum << endl;
	}
	if (!f_has_sum) {
		cout << "diophant::get_solutions !f_has_sum" << endl;
		exit(1);
	}
	nb_sol = _resultanz;
	Sol = NEW_int(nb_sol * sum);
	for (i = 0; i < _resultanz; i++) {
		res = _results[i];
		cnt = 0;
		for (j = 0; j < n; j++) {
			if (res[j]) {
				Sol[i * sum + cnt] = j;
				cnt++;
			}
		}
		if (cnt != sum) {
			cout << "diophant::get_solutions_index_set cnt != sum" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "diophant::get_solutions done" << endl;
	}
}

void diophant::get_solutions(
		int *&Sol, int &nb_sol, int verbose_level)
// allocates Sol[nb_sol * n]
{
	int f_v = (verbose_level >= 1);
	int i, j;
	vector<int> res;

	if (f_v) {
		cout << "diophant::get_solutions" << endl;
		cout << "nb_sol = " << _resultanz << endl;
		cout << "sum = " << sum << endl;
	}
#if 0
	if (!f_has_sum) {
		cout << "diophant::get_solutions !f_has_sum" << endl;
		exit(1);
	}
#endif
	nb_sol = _resultanz;
	Sol = NEW_int(nb_sol * n);
	for (i = 0; i < _resultanz; i++) {
		res = _results[i];
		for (j = 0; j < n; j++) {
			Sol[i * n + j] = res[j];
		}
	}
	if (f_v) {
		cout << "diophant::get_solutions done" << endl;
	}
}

#if 0
void diophant::get_solutions_full_length(
		int *&Sol,
		int &nb_sol, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	vector<int> res;

	if (f_v) {
		cout << "diophant::get_solutions_full_length" << endl;
		cout << "nb_sol = " << _resultanz << endl;
		cout << "sum = " << sum << endl;
	}
#if 0
	if (!f_has_sum) {
		cout << "diophant::get_solutions_full_length !f_has_sum" << endl;
		exit(1);
	}
#endif
	nb_sol = _resultanz;
	Sol = NEW_int(nb_sol * n);
	for (i = 0; i < _resultanz; i++) {
		res = _results[i]; //.front();
		for (j = 0; j < n; j++) {
			Sol[i * n + j] = res[j];
		}
		//_results.pop_front();
	}
	if (f_v) {
		cout << "diophant::get_solutions_full_length done" << endl;
	}
}
#endif

void diophant::test_solution_full_length(
		int *sol, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, s;

	if (f_v) {
		cout << "diophant::test_solution_full_length" << endl;
	}
	s = 0;
	for (j = 0; j < n; j++) {
		s += sol[j];
	}
	cout << "diophant::test_solution_full_length s=" << s << endl;
	if (!f_has_sum) {
		cout << "diophant::get_solutions_full_length !f_has_sum" << endl;
		exit(1);
	}
	if (s != sum) {
		cout << "diophant::test_solution_full_length s != sum" << endl;
		exit(1);
	}
	for (i = 0; i < m; i++) {
		s = 0;
		for (j = 0; j < n; j++) {
			s += Aij(i, j) * sol[j];
		}
		cout << "diophant::test_solution_full_length condition " << i
				<< " / " << m << ":" << endl;
		if (type[i] == t_EQ) {
			if (s != RHSi(i)) {
				cout << "diophant::test_solution_full_length condition "
						<< i << " / " << m << ": s=" << s
						<< " is not equal to " << RHSi(i) << endl;
				exit(1);
			}
		}
		else if (type[i] == t_LE) {
			if (s >= RHSi(i)) {
				cout << "diophant::test_solution_full_length condition "
						<< i << " / " << m << ": s=" << s
						<< " is larger than " << RHSi(i) << endl;
				exit(1);
			}
		}
		else if (type[i] == t_INT) {
			if (s >= RHSi(i)) {
				cout << "diophant::test_solution_full_length condition "
						<< i << " / " << m << ": s=" << s
						<< " is larger than " << RHSi(i) << endl;
				exit(1);
			}
			if (s < RHS_low_i(i)) {
				cout << "diophant::test_solution_full_length condition "
						<< i << " / " << m << ": s=" << s
						<< " is less than " << RHS_low_i(i) << endl;
				exit(1);
			}
		}
		else if (type[i] == t_ZOR) {
			if (s != 0 && s != RHSi(i)) {
				cout << "diophant::test_solution_full_length condition "
						<< i << " / " << m << ": s=" << s
						<< " is not equal to " << RHSi(i)
						<< " or zero" << endl;
				exit(1);
			}
		}
	}
}

int diophant::solve_all_DLX(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::solve_all_DLX verbose_level="
				<< verbose_level << endl;
	}
	//install_callback_solution_found(diophant_callback_solution_found, this);

	int *Inc;
	int i, j;
	//int nb_sol, nb_backtrack;

	Inc = NEW_int(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			Inc[i * n + j] = Aij(i, j);
		}
	}

	_resultanz = 0;

	dlx_problem_description Descr;

	Descr.f_label_txt = true;
	Descr.label_txt.assign(label);
	Descr.f_label_tex = true;
	Descr.label_tex.assign(label);

	Descr.f_data_matrix = true;
	Descr.data_matrix = Inc;
	Descr.data_matrix_m = m;
	Descr.data_matrix_n = n;

	dlx_solver Solver;

	if (f_v) {
		cout << "diophant::solve_all_DLX before Solver.init" << endl;
	}
	Solver.init(&Descr, verbose_level);
	if (f_v) {
		cout << "diophant::solve_all_DLX after Solver.init" << endl;
	}
	Solver.install_callback_solution_found(diophant_callback_solution_found, this);


	if (f_v) {
		cout << "diophant::solve_all_DLX before Solver.Solve" << endl;
	}
	Solver.Solve(verbose_level);
	if (f_v) {
		cout << "diophant::solve_all_DLX after Solver.Solve" << endl;
	}

	
#if 0
	DlxTransposeAppendAndSolve(Inc, m, n, nb_sol, nb_backtrack, 
		false, "", 
		verbose_level - 1);

	nb_steps_betten = nb_backtrack;
#endif

	nb_steps_betten = Solver.nb_backtrack_nodes;
	

	FREE_int(Inc);
	if (f_v) {
		cout << "diophant::solve_all_DLX done found " << _resultanz
			<< " solutions with " << nb_steps_betten
			<< " backtrack steps" << endl;
	}

	return _resultanz;
}

int diophant::solve_all_DLX_with_RHS(
		int f_write_tree,
		const char *fname_tree, int verbose_level)
{
	cout << "diophant::solve_all_DLX_with_RHS disabled" << endl;
#if 0
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "diophant::solve_all_DLX_with_RHS "
				"verbose_level=" << verbose_level << endl;
	}
	if (f_v) {
		cout << "diophant::solve_all_DLX_with_RHS "
				"m=" << m << " n=" << n << endl;
	}
	install_callback_solution_found(
		diophant_callback_solution_found,
		this);

	int *Inc;
	int *my_RHS;
	int f_has_type;
	diophant_equation_type *my_type;
	int i, j;
	int nb_sol, nb_backtrack;

	Inc = NEW_int(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			Inc[i * n + j] = Aij(i, j);
		}
	}
	if (f_vv) {
		cout << "diophant::solve_all_DLX_with_RHS "
				"the system:" << endl;
		Int_matrix_print(Inc, m, n);
	}
	f_has_type = true;
	my_RHS = NEW_int(m);
	my_type = NEW_OBJECTS(diophant_equation_type, m);
	for (i = 0; i < m; i++) {
		my_RHS[i] = RHS[i];
		my_type[i] = type[i];
	}
	if (f_vv) {
		cout << "diophant::solve_all_DLX_with_RHS  RHS:" << endl;
		Int_matrix_print(my_RHS, m, 1);
		//cout << diophant::solve_all_DLX_with_RHS  type:" << endl;
		//int_matrix_print(my_type, m, 1);
	}

	_resultanz = 0;
	
	DlxTransposeAndSolveRHS(Inc, m, n, 
		my_RHS, f_has_type, my_type, 
		nb_sol, nb_backtrack, 
		false, "", 
		f_write_tree, fname_tree, 
		verbose_level - 1);
	
	nb_steps_betten = nb_backtrack;
	FREE_int(Inc);
	FREE_int(my_RHS);
	FREE_OBJECTS(my_type);
	if (f_v) {
		cout << "diophant::solve_all_DLX_with_RHS done found "
			<< _resultanz << " solutions with " << nb_backtrack
			<< " backtrack steps" << endl;
	}
#endif
	return _resultanz;
}

int diophant::solve_all_DLX_with_RHS_and_callback(
	int f_write_tree, const char *fname_tree,
	void (*user_callback_solution_found)(int *sol,
			int len, int nb_sol, void *data),
	int verbose_level)
{
	cout << "diophant::solve_all_DLX_with_RHS_and_callback disabled" << endl;

#if 0
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "diophant::solve_all_DLX_with_RHS "
				"verbose_level=" << verbose_level << endl;
	}
	if (f_v) {
		cout << "diophant::solve_all_DLX_with_RHS "
				"m=" << m << " n=" << n << endl;
	}
	diophant_user_callback_solution_found = user_callback_solution_found;
	
	install_callback_solution_found(
		diophant_callback_solution_found,
		this);

	int *Inc;
	int *my_RHS;
	int f_has_type;
	diophant_equation_type *my_type;
	int i, j;
	int nb_sol, nb_backtrack;

	Inc = NEW_int(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			Inc[i * n + j] = Aij(i, j);
		}
	}
	if (f_vv) {
		cout << "diophant::solve_all_DLX_with_RHS "
				"the system:" << endl;
		Int_matrix_print(Inc, m, n);
	}
	f_has_type = true;
	my_RHS = NEW_int(m);
	my_type = NEW_OBJECTS(diophant_equation_type, m);
	//my_f_le = NEW_int(m);
	for (i = 0; i < m; i++) {
		my_RHS[i] = RHS[i];
		my_type[i] = type[i];
		//my_f_le[i] = f_le[i];
	}
	if (f_vv) {
		cout << "diophant::solve_all_DLX_with_RHS  RHS:" << endl;
		Int_matrix_print(my_RHS, m, 1);
		//cout << diophant::solve_all_DLX_with_RHS  type:" << endl;
		//int_matrix_print(my_type, m, 1);
	}

	_resultanz = 0;
	
	DlxTransposeAndSolveRHS(Inc, m, n, 
		my_RHS, f_has_type, my_type, 
		nb_sol, nb_backtrack, 
		false, "", 
		f_write_tree, fname_tree, 
		verbose_level - 1);
	
	nb_steps_betten = nb_backtrack;
	FREE_int(Inc);
	FREE_int(my_RHS);
	FREE_OBJECTS(my_type);
	if (f_v) {
		cout << "diophant::solve_all_DLX_with_RHS done found "
				<< _resultanz << " solutions with " << nb_backtrack
				<< " backtrack steps" << endl;
	}
#endif
	return _resultanz;
}

int diophant::solve_all_mckay(
		long int &nb_backtrack_nodes, int maxresults,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int maxresults = 10000;
	//long int nb_backtrack_nodes;
	int nb_sol;
	
	if (f_v) {
		cout << "diophant::solve_all_mckay before solve_mckay, "
				"verbose_level=" << verbose_level << endl;
	}
	solve_mckay(label, maxresults,
			nb_backtrack_nodes, nb_sol,
			verbose_level - 2);
	if (f_v) {
		cout << "diophant::solve_all_mckay found " << _resultanz
				<< " solutions in " << nb_backtrack_nodes
				<< " backtrack nodes" << endl;
	}
	return _resultanz;
}

int diophant::solve_once_mckay(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int maxresults = 1;
	long int nb_backtrack_nodes;
	int nb_sol;

	solve_mckay(label, maxresults,
			nb_backtrack_nodes, nb_sol, verbose_level - 2);
	if (f_v) {
		cout << "diophant::solve_once_mckay found " << _resultanz
				<< " solutions in " << nb_backtrack_nodes
				<< " backtrack nodes" << endl;
	}
	return _resultanz;
}


int diophant::solve_all_betten(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "diophant::solve_all_betten" << endl;
	}

	int j;
	vector<int> lo;
	//int maxresults = 10000000;
	_resultanz = 0;
	_cur_result = 0;
	
	if (solve_first_betten(verbose_level - 2)) {
		lo.resize(n);
		for (j = 0; j < n; j++) {
			lo[j] = (int) x[j];
		}
		_results.push_back(lo);
		_resultanz++;
		while (solve_next_betten(verbose_level - 2)) {
			lo.resize(n);
			for (j = 0; j < n; j++) {
				lo[j] = (int) x[j];
			}
			if ((_resultanz % 100000) == 0) {
				if (f_v) {
					cout << "diophant::solve_all_betten nb_sol=" << _results.size() << endl;
				}
			}
			_results.push_back(lo);
			_resultanz++;
		}
	}
	//solve_mckay(maxresults, verbose_level - 2);
	if (f_v) {
		cout << "diophant::solve_all_betten found " << _resultanz 
			<< " solutions in " << nb_steps_betten << " steps" << endl;
	}
	return _resultanz;
}

int diophant::solve_all_betten_with_conditions(
		int verbose_level,
	int f_max_sol, int max_sol, 
	int f_max_time, int max_time_in_seconds)
{
	int f_v = (verbose_level >= 1);
	int j;
	vector<int> lo;
	other::orbiter_kernel_system::os_interface Os;


	//int maxresults = 10000000;
	_resultanz = 0;
	_cur_result = 0;
	
	if (f_max_time) {
		diophant::f_max_time = true;
		diophant::max_time_in_sec = max_time_in_seconds;
		f_broken_off_because_of_maxtime = false;
		t0 = Os.os_ticks();
		max_time_in_ticks = max_time_in_seconds * Os.os_ticks_per_second();
		if (true || f_v) {
			cout << "solve_all_betten_with_conditions maxtime "
					"max_time_in_sec=" << max_time_in_sec << endl;
		}
	}
	t0 = Os.os_ticks();
	if (solve_first_betten(verbose_level - 2)) {
		lo.resize(n);
		for (j = 0; j < n; j++) {
			lo[j] = (int) x[j];
		}
		_results.push_back(lo);
		_resultanz++;
		if (f_max_sol && _resultanz == max_sol) {
			return true;
		}
		while (solve_next_betten(verbose_level - 2)) {
			lo.resize(n);
			for (j = 0; j < n; j++) {
				lo[j] = (int) x[j];
			}
			_results.push_back(lo);
			_resultanz++;
			if (f_max_sol && _resultanz == max_sol) {
				return true;
			}
		}
	}
	if (f_broken_off_because_of_maxtime) {
		return true;
	}
	//solve_mckay(maxresults, verbose_level - 2);
	if (f_v) {
		cout << "diophant::solve_all_betten found " << _resultanz 
			<< " solutions in " << nb_steps_betten << " steps" << endl;
	}
	return false;
}

int diophant::solve_first_betten(
		int verbose_level)
{
	int i, j, g;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int k, total_max;
	algebra::number_theory::number_theory_domain NT;
	other::orbiter_kernel_system::os_interface Os;

	if (!f_has_sum) {
		cout << "diophant::solve_first_betten !f_has_sum" << endl;
		exit(1);
	}
	nb_steps_betten = 0;
	if (m <= 0) {
		if (f_v) {
			cout << "diophant::solve_first_betten: m <= 0" << endl;
		}
		return true;
		}
	if (n == 0) {
		//cout << "diophant::solve_first_betten: n == 0" << endl;
		for (i = 0; i < m; i++) {
			if (type[i] == t_EQ) {
				if (RHS[i]) {
					if (f_v) {
						cout << "diophant::solve_first_betten no solution "
							"in equation " << i << " because n=0 and "
							"RHS=" << RHS[i] << " and not an inequality"
							<< endl;
					}
					return false;
				}
			}
		}
		return true;
	}
	for (i = 0; i < m; i++) {
		RHS1[i] = RHS[i];
	}
	sum1 = sum;
	//if (f_x_max) {
		total_max = 0;
		for (k = 0; k < n; k++) {
			total_max += x_max[k];
		}
		if (total_max < sum) {
			if (f_v) {
				cout << "diophant::solve_first_betten() total_max "
					<< total_max << " < sum = " << sum
					<< ", no solution" << endl;
			}
			return false;
		}
	//}
	
	// 
	// compute gcd: 
	//
	for (i = 0; i < m; i++) {
		if (type[i] == t_EQ) {
			if (n >= 2) {
				j = n - 2;
				g = Aij(i, n - 1);
				Gij(i, j) = g;
				j--;
				for (; j >= 0; j--) {
					g = NT.gcd_lint(Aij(i, j + 1), g);
					Gij(i, j) = g;
				}
			}
			Gij(i, n - 1) = 0;
				// in the last step: RHS1 cong 0 mod 0 means: RHS1 == 0 
		}
		else {
			for (j = 0; j < n; j++) {
				Gij(i, j) = 0;
			}
		}
	}
	
	for (j = 0; j < n; j++) {
		x[j] = 0;
	}
	if (f_vv) {
		cout << "diophant::solve_first_betten: gcd computed:" << endl;
		print2(true);
	}

	j = 0;
	while (true) {
		while (true) {
			if (j >= n) {
				if (f_v) {
					cout << "diophant::solve_first_betten solution" << endl;
					print_x(nb_steps_betten);
				}
				return true;
			}
			nb_steps_betten++;
			if ((nb_steps_betten % 1000000) == 0) {
				cout << "diophant::solve_first_betten nb_steps_betten="
					<< nb_steps_betten << " sol=" << _resultanz << " ";
				if (f_max_time) {
					int t1 = Os.os_ticks();
					int dt = t1 - t0;
					int t = dt / Os.os_ticks_per_second();
					cout << "time in seconds: " << t;
				}
				cout << endl;
				print_x(nb_steps_betten);
				if (f_max_time) {
					int t1 = Os.os_ticks();
					int dt = t1 - t0;
					if (dt > max_time_in_ticks) {
						f_broken_off_because_of_maxtime = true;
						return false;
					}
				}
			}
#if 0
			if (nb_steps_betten == 51859) {
				verbose_level = 4;
				f_v = (verbose_level >= 1);
				f_vv = (verbose_level >= 2);
			}
#endif
			if (f_vv) {
				cout << "diophant::solve_first_betten j=" << j
					<< " sum1=" << sum1 << " x:" << endl;
				print_x(nb_steps_betten);
				cout << endl;
			}
			if (!j_fst(j, verbose_level - 2)) {
				break;
			}
			j++;
		}
		while (true) {
			if (j == 0) {
				if (f_v) {
					cout << "diophant::solve_first_betten "
							"no solution" << endl;
				}
				return false;
			}
			j--;
			nb_steps_betten++;
			if ((nb_steps_betten % 1000000) == 0) {
				cout << "diophant::solve_first_betten nb_steps_betten="
						<< nb_steps_betten << " sol=" << _resultanz << " ";
				if (f_max_time) {
					int t1 = Os.os_ticks();
					int dt = t1 - t0;
					int t = dt / Os.os_ticks_per_second();
					cout << "time in seconds: " << t;
				}
				cout << endl;
				print_x(nb_steps_betten);
				if (f_max_time) {
					int t1 = Os.os_ticks();
					int dt = t1 - t0;
					if (dt > max_time_in_ticks) {
						f_broken_off_because_of_maxtime = true;
						return false;
					}
				}
			}
#if 0
			if (nb_steps_betten == 51859) {
				verbose_level = 4;
				f_v = (verbose_level >= 1);
				f_vv = (verbose_level >= 2);
			}
#endif
			if (f_vv) {
				cout << "diophant::solve_first_betten j=" << j
						<< " sum1=" << sum1 << " x:" << endl;
				print_x(nb_steps_betten);
				cout << endl;
			}
			if (j_nxt(j, verbose_level - 2)) {
				break;
			}
		}
		j++;
	}
}

int diophant::solve_next_mckay(
		int verbose_level)
{
	int j;
	if (_cur_result < _resultanz) {
		for (j = 0; j < n; j++) {
			x[j] = _results[_cur_result][j]; //.front()[j];
		}
		//_results.pop_front();
		_cur_result++;
		return true;
	}
	else {
		return false;
	}
}

int diophant::solve_next_betten(
		int verbose_level)
{
	int j;
	other::orbiter_kernel_system::os_interface Os;
	
	if (!f_has_sum) {
		cout << "diophant::solve_next_betten !f_has_sum" << endl;
		exit(1);
	}
	if (m == 0) {
		return false;
		}
	if (n == 0) {
		return false;
	}
	j = n - 1;
	while (true) {
		while (true) {
			nb_steps_betten++;
			if ((nb_steps_betten % 100000) == 0) {
				cout << "diophant::solve_next_betten nb_steps_betten="
						<< nb_steps_betten << " sol=" << _resultanz << " ";
				if (f_max_time) {
					int t1 = Os.os_ticks();
					int dt = t1 - t0;
					int t = dt / Os.os_ticks_per_second();
					cout << "time in seconds: " << t;
				}
				cout << endl;
				print_x(nb_steps_betten);
				if (f_max_time) {
					int t1 = Os.os_ticks();
					int dt = t1 - t0;
					if (dt > max_time_in_ticks) {
						f_broken_off_because_of_maxtime = true;
						return false;
					}
				}
			}
			if (j_nxt(j, verbose_level)) {
				break;
			}
			if (j == 0) {
				return false;
			}
			j--;
		}
		while (true) {
			if (j >= n - 1) {
				return true;
			}
			j++;
			nb_steps_betten++;
			if ((nb_steps_betten % 1000000) == 0) {
				cout << "diophant::solve_next_betten nb_steps_betten="
						<< nb_steps_betten << " sol=" << _resultanz << " ";
				if (f_max_time) {
					int t1 = Os.os_ticks();
					int dt = t1 - t0;
					int t = dt / Os.os_ticks_per_second();
					cout << "time in seconds: " << t;
				}
				cout << endl;
				if (f_max_time) {
					int t1 = Os.os_ticks();
					int dt = t1 - t0;
					if (dt > max_time_in_ticks) {
						f_broken_off_because_of_maxtime = true;
						return false;
					}
				}
			}
			if (!j_fst(j, verbose_level)) {
				break;
			}
		}
		j--;
	}
}

int diophant::j_fst(
		int j, int verbose_level)
// if return value is false, 
// x[j] is 0 and RHS1[i] unchanged;
// otherwise RHS1[i] := RHS1[i] - lf->x[j] * lf->a[i][j] 
// and RHS1[i] divisible by g[i][j] 
// (or RHS1 == 0 if g[j] == 0)
// for all 0 <= i < n. 
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, ii, g, a, b;
	
	if (f_v) {
		cout << "j_fst node=" << nb_steps_betten << " j=" << j << endl;
	}
	// max value for x[j]: 
	x[j] = sum1;
	//if (f_x_max) {
		// with restriction: 
		x[j] = MINIMUM(x[j], x_max[j]);
	//}
	if (f_vv) {
		cout << "diophant::j_fst j=" << j << " trying x[j]=" << x[j] << endl;
	}
	for (i = 0; i < m; i++) {
		if (x[j] == 0) {
			break;
		}
		a = Aij(i, j);
		if (a == 0) {
			continue;
		}
		// x[j] = MINIMUM(x[j], (RHS1[i] / a));
		b = RHS1[i] / a;
		if (b < x[j]) {
			if (f_vv) {
				cout << "diophant::j_fst j=" << j << " reducing x[j] "
					"from " << x[j] << " to " << b
					<< " because of equation " << i << " = "
					<< eqn_label[i] << endl;
				cout << "RHS1[i]=" << RHS1[i] << endl;
				cout << "a=" << a << endl;
			}
			x[j] = b;
		}
	}
	if (f_vv) {
		cout << "diophant::j_fst j=" << j << " trying "
				"x[j]=" << x[j] << endl;
	}
	
	sum1 -= x[j];
	for (i = 0; i < m; i++) {
		RHS1[i] -= x[j] * A[i * n + j];
	}
	for (i = 0; i < m; i++) {
		if (RHS1[i] < 0) {
			cout << "diophant::j_fst(): RHS1[i] < 0" << endl;
			exit(1);
		}
	}
	// RHS1[] non negative now 
	if (f_vv) {
		cout << "diophant::j_fst: x[" << j << "] = " << x[j] << endl;
	}

	if (j == n - 1) {
		// now have to check if the 
		// current vector x[] is in fact a 
		// solution;
		// this means:
		// a) if eqn i is an inequation: 
		//          no restriction
		// b) if eqn i is an equation: 
		//          RHS[i] must be 0
		//
		for (i = 0; i < m; i++) {
			if (type[i] == t_LE) {
				continue;
			}
			if (RHS1[i] != 0) {
				break; // no solution 
			}
		}
		if (i < m || sum1 > 0) {
			// cout << "no solution !" << endl;

			// not passed, go back */
			// NOTE: we CAN go back here 
			// in any case; reason: 
			// if we decrement x[n - 1]
			// than sum1 will be positive 
			// and this cannot be a solution. 
			for (ii = 0; ii < m; ii++) {
				RHS1[ii] += x[j] * A[ii * n + j];
			}
			sum1 += x[j];
			x[j] = 0;
			if (f_vv) {
				cout << "diophant::j_fst no solution b/c RHS[i] "
						"nonzero || sum1 > 0" << endl;
				cout << "i=" << i << endl;
				cout << "j=" << j << " = n - 1" << endl;
				cout << "n=" << n << endl;
				cout << "RHS1[i]=" << RHS1[i] << endl;
				cout << "sum1=" << sum1 << endl;
				cout << "m=" << m << endl;
				print_x(nb_steps_betten);
			}
			return false;
		}
		return true;
	}
	
	while (true) {
		// check gcd restrictions: 
		for (i = 0; i < m; i++) {
			if (type[i] == t_LE) {
				continue;
				// it is an inequality, hence no gcd condition
			}
			g = G[i * n + j];
			if (g == 0 && RHS1[i] != 0) {
				if (f_vv) {
					cout << "diophant::j_fst g == 0 && RHS1[i] != 0 in "
							"eqn i=" << i << " = " << eqn_label[i] << endl;
					cout << "g=" << g << endl;
					cout << "i=" << i << endl;
					cout << "j=" << j << " != n - 1" << endl;
					cout << "n=" << n << endl;
					cout << "RHS1[i]=" << RHS1[i] << endl;
					print_x(nb_steps_betten);
				}
				break;
			}
			if (g == 0) {
				continue;
			}
			if (g == 1) {
				// no restriction
				continue;
			}
			if ((RHS1[i] % g) != 0) {
				if (f_vv) {
					cout << "diophant::j_fst (RHS1[i] % g) != 0 in "
							"equation i=" << i << " = " << eqn_label[i] << endl;
					cout << "g=" << g << endl;
					cout << "i=" << i << endl;
					cout << "j=" << j << " != n - 1" << endl;
					cout << "n=" << n << endl;
					cout << "RHS1[i]=" << RHS1[i] << endl;
					print_x(nb_steps_betten);
				}
				break;
			}
		}
		if (i == m) { // OK
			break;
		}
		
		if (f_vv) {
			cout << "gcd test failed !" << endl;
		}
		// was not OK
		if (x[j] == 0) {
			if (f_vv) {
				cout << "diophant::j_fst no solution b/c gcd test "
						"failed in equation " << i << " = " << eqn_label[i] << endl;
				cout << "j=" << j << endl;
				cout << "x[j]=" << x[j] << endl;
				cout << "RHS1[i]=" << RHS1[i] << endl;
				cout << "Gij(i,j)=" << Gij(i,j) << endl;
				print_x(nb_steps_betten);
			}
			return false;
		}
		x[j]--;
		sum1++;
		for (ii = 0; ii < m; ii++) {
			RHS1[ii] += A[ii * n + j];
		}
		if (f_vv) {
			cout << "diophant::j_fst decrementing to: x[" << j
					<< "] = " << x[j] << endl;
		}
	}
	return true;
}

int diophant::j_nxt(
		int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, ii, g;

	if (f_v) {
		cout << "j_nxt node=" << nb_steps_betten << " j=" << j <<  endl;
	}
	if (j == n - 1) {
		for (ii = 0; ii < m; ii++) {
			RHS1[ii] += x[j] * A[ii * n + j];
		}
		sum1 += x[j];
		x[j] = 0;
		if (f_vv) {
			cout << "diophant::j_nxt no solution b/c j == n - 1" << endl;
			cout << "j=" << j << endl;
			cout << "n=" << n << endl;
			print_x(nb_steps_betten);
		}
		return false;
	}
	
	while (x[j] > 0) {
		x[j]--;
		if (f_vv) {
			cout << "diophant::j_nxt decrementing to: x[" << j
					<< "] = " << x[j] << endl;
		}
		sum1++;
		for (ii = 0; ii < m; ii++) {
			RHS1[ii] += A[ii * n + j];
		}
		
		// check gcd restrictions: 
		for (i = 0; i < m; i++) {
			if (type[i] == t_LE) {
				continue;
				// it is an inequality, hence no gcd condition
			}
			g = G[i * n + j];
			if (g == 0 && RHS1[i] != 0) {
				break;
			}
			if (g == 0) {
				continue;
			}
			if (g == 1) {
				// no restriction
				continue;
			}
			if ((RHS1[i] % g) != 0) {
				break;
			}
		}
		if (i == m) {
			// OK
			return true;
		}
		if (f_vv) {
			cout << "diophant::j_nxt() gcd restriction failed in "
					"eqn " << i << " = " << eqn_label[i] << endl;
		}
	}
	if (f_vv) {
		cout << "diophant::j_nxt no solution b/c gcd test failed" << endl;
		cout << "j=" << j << endl;
		print_x(nb_steps_betten);
	}
	return false;
}

void diophant::solve_mckay(
		std::string &label,
		int maxresults,
	long int &nb_backtrack_nodes, int &nb_sol,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::solve_mckay" << endl;
	}
	solve_mckay_override_minrhs_in_inequalities(label, 
		maxresults, nb_backtrack_nodes, 0 /* minrhs */, nb_sol, 
		verbose_level);
	if (f_v) {
		cout << "diophant::solve_mckay done" << endl;
	}
}

void diophant::solve_mckay_override_minrhs_in_inequalities(
	std::string &label,
	int maxresults, long int &nb_backtrack_nodes,
	int minrhs, int &nb_sol, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, nb;
	vector<int> minres, maxres, fanz;
	mckay::tMCKAY lgs;
	vector<mckay::equation> eqn;
	map<int, int>::iterator it;
	vector<int> minvarvalue;
	vector<int> maxvarvalue;

	if (f_v) {
		cout << "diophant::solve_mckay_override_minrhs_in_inequalities "
				<< label << ", a system "
				"of size " << m << " x " << n << endl;
	}

	int nb_eqns;

	if (f_has_sum) {
		nb_eqns= m + 1;
	}
	else {
		nb_eqns = m;
	}

	lgs.Init(this, label, nb_eqns, n);
	minres.resize(nb_eqns);
	maxres.resize(nb_eqns);
	fanz.resize(m + 1);
	eqn.resize(m + 1);
	
	for (i = 0; i < m; i++) {
		// the RHS:
		if (type[i] == t_EQ) {
			minres[i] = (int) RHS[i];
			maxres[i] = (int) RHS[i];
		}
		else if (type[i] == t_LE) {
			minres[i] = (int) minrhs;
			maxres[i] = (int) RHS[i];
		}
		else if (type[i] == t_INT) {
			minres[i] = (int) RHS_low[i];
			maxres[i] = (int) RHS[i];
		}
		else if (type[i] == t_ZOR) {
			minres[i] = (int) 0;
			maxres[i] = (int) RHS[i];
		}
		else {
			cout << "diophant::solve_mckay_override_minrhs_in_inequalities "
					"we cannot do this type of "
					"condition, equation " << i << endl;
			exit(1);
		}
		
		// count the number of nonzero coefficients:
		nb = 0;
		for (j = 0; j < n; j++) {
			if (A[i * n + j]) {
				nb++;
			}
		}
		
		// initialize coefficients:
		fanz[i] = nb;
		eqn[i].resize(nb);
		nb = 0;
		for (j = 0; j < n; j++) {
			if (A[i * n + j]) {
				eqn[i][nb].var = j;
				eqn[i][nb].coeff = (int) A[i * n + j];
				nb++;
			}
		}
	} // next i
	
	if (f_has_sum) {
		// one more equation for \sum x_j = sum
		i = (int) m;
		fanz[i] = (int) n;
		eqn[i].resize(n);
		for (j = 0; j < n; j++) {
			eqn[i][j].var = j;
			eqn[i][j].coeff = 1;
		}
		minres[i] = (int) sum;
		maxres[i] = (int) sum;
	}
	
	// now the bounds on the x_j
	minvarvalue.resize(n);
	maxvarvalue.resize(n);

	if (f_v) {
		cout << "diophant::solve_mckay_override_minrhs_in_inequalities "
				"f_x_max=true" << endl;
		cout << "x_max=";
		Int_vec_print(cout, x_max, n);
		cout << endl;
		cout << "x_min=";
		Int_vec_print(cout, x_min, n);
		cout << endl;
	}
	for (j = 0; j < n; j++) {
		minvarvalue[j] = (int) x_min[j];
		maxvarvalue[j] = (int) x_max[j];
	}

#if 0
	else {
		if (f_v) {
			cout << "diophant::solve_mckay_override_minrhs_in_inequalities "
					"f_x_max=false initializing with "  << INT_MAX << endl;
		}
		for (j = 0; j < n; j++) {
			minvarvalue[j] = 0;
			maxvarvalue[j] = (int) INT_MAX;
		}
	}
#endif

	// trying to restrict maxvarvalue[]:
	int b;
	for (i = 0; i < m; i++) {
		if (f_v) {
			cout << "trying to restrict using of equation " << i << endl;
		}
		for (j = 0; j < n; j++) {
			if (A[i * n + j]) {

				b = RHS[i] / A[i * n + j];

				if (b < maxvarvalue[j]) {
					if (f_v) {
						cout << "lowering maxvarvalue[" << j << "] from "
								<< maxvarvalue[j] << " to " << b
								<< " because of equation " << i << endl;
					}
					maxvarvalue[j] = b;
				}
			}
		}
	}
	if (f_v) {
		cout << "after restrictions:" << endl;
		print_dense();
	}

	if (f_v) {
		cout << "diophant::solve_mckay_override_minrhs_in_inequalities "
				"maxvarvalue=" << endl;
		for (j = 0; j < n; j++) {
			cout << j << " : " << maxvarvalue[j] << endl;
		}
	}
	_resultanz = 0;
	_maxresults = (int) maxresults;
	
	lgs.possolve(minvarvalue, maxvarvalue, 
		eqn, minres, maxres, fanz, 
		nb_eqns, n,
		verbose_level);
	nb_backtrack_nodes = lgs.nb_calls_to_solve;
	nb_sol = _resultanz;
	if (f_v) {
		cout << "diophant::solve_mckay_override_minrhs_in_inequalities "
			<< label << " finished, "
			"number of solutions = " << _resultanz
			<< " nb_backtrack_nodes=" << nb_backtrack_nodes << endl;
	}
}

void diophant::latex_it()
{
	latex_it(cout);
}

void diophant::latex_it(
		std::ostream &ost)
{
	int i, j, a;
	
	ost << "\\begin{array}{|*{" << n << "}{r}|r|l|}" << endl;
#if 0
	ost << "\\hline" << endl;
	//ost << "   & ";
	for (j = 0; j < n; j++) {
		ost << setw(2) << (int)(j / 10) << " & ";
		}
	ost << " & & \\\\" << endl;
	ost << "   & ";
	for (j = 0; j < n; j++) {
		ost << setw(2) << j % 10 << " & ";
		}
	ost << " & & \\\\" << endl;
	if (f_x_max) {
		//ost << "   & ";
		for (j = 0; j < n; j++) {
			ost << setw(2) << (int)(x_max[j] / 10) << " & ";
			}
		ost << " & & \\\\" << endl;
		ost << "   & ";
		for (j = 0; j < n; j++) {
			ost << setw(2) << x_max[j] % 10 << " & ";
			}
		ost << " & & \\\\" << endl;
		}
	ost << "\\hline" << endl;
#endif
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		//ost << setw(2) << i << " & ";
		for (j = 0; j < n; j++) {
			a = Aij(i, j);
			ost << setw(2) << a << " & ";
		}
		if (type[i] == t_EQ) {
			ost << " =  " << setw(2) << RHS[i] << " & ";
		}
		else if (type[i] == t_LE) {
			ost << "  \\le   " << setw(2) << RHS[i] << " & ";
		}
		else if (type[i] == t_INT) {
			ost << "  \\in   [" << setw(2) << RHS_low[i] << "," << setw(2) << RHS[i] << "] & ";
		}
		else if (type[i] == t_ZOR) {
			ost << "  ZOR   " << setw(2) << RHS[i] << " & ";
		}
		ost << eqn_label[i];
		ost << "\\\\" << endl;
	}
	ost << "\\hline" << endl;
	//if (f_x_max) {
		ost << "\\multicolumn{" << n + 2 << "}{|c|}{" << endl;
		ost << "\\mbox{subject to:}" << endl;
		ost << "}\\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\multicolumn{" << n + 2 << "}{|l|}{" << endl;
		for (j = 0; j < n; j++) {
			ost << x_min[j] << " \\le x_{" << j + 1 << "} \\le " << x_max[j] << "\\," << endl;
		}
		if (f_has_sum) {
			ost << "\\sum_{i=1}^{" << n << "} x_i=" << sum << endl;
		}
		ost << "}\\\\" << endl;
		ost << "\\hline" << endl;
	//}
	ost << "\\end{array}" << endl;
}

void diophant::trivial_row_reductions(
	int &f_no_solution, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, d, m1;
	int f_trivial;

	if (f_v) {
		cout << "diophant::trivial_row_reductions" << endl;
	}
	m1 = m;
	f_no_solution = false;
	for (i = m - 1; i >= 0; i--) {
		f_trivial = false;
		d = count_non_zero_coefficients_in_row(i);
		if (type[i] == t_LE) {
			if (d <= RHS[i]) {
				f_trivial = false; // this is only valid if x_max = 1
			}
		}
		else if (type[i] == t_INT) {
			if (d <= RHS[i]) {
				f_trivial = false; // this is only valid if x_max = 1
			}
		}
		else if (type[i] == t_EQ) {
			if (RHS[i] > d) {
				f_no_solution = false; // this is only valid if x_max = 1
			}
		}
		if (f_trivial) {
			delete_equation(i);
		}
	}
	if (f_v) {
		cout << "diophant::trivial_row_reductions done, eliminated "
				<< m1 - m << " equations" << endl;
	}
}

diophant *diophant::trivial_column_reductions(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h, a, rhs;
	int *f_deleted;
	int *col_idx;
	int nb_deleted;

	if (f_v) {
		cout << "diophant::trivial_column_reductions" << endl;
	}
	f_deleted = NEW_int(n);
	col_idx = NEW_int(n);
	Int_vec_zero(f_deleted, n);
	Int_vec_zero(col_idx, n);
	for (j = 0; j < n; j++) {
		col_idx[j] = -1;
	}
	for (i = 0; i < m; i++) {
		if (type[i] == t_EQ) {
			rhs = RHS[i];
			for (j = 0; j < n; j++) {
				a = Aij(i, j);
				if (a > rhs) {
					f_deleted[j] = true;
				}
			}
		}
	}
	for (j = 0; j < n; j++) {
		cout << f_deleted[j];
	}
	cout << endl;
	nb_deleted = 0;
	h = 0;
	for (j = 0; j < n; j++) {
		if (f_deleted[j]) {
			nb_deleted++;
		}
		else {
			col_idx[j] = h;
			h++;
		}
	}
	if (f_v) {
		cout << "diophant::trivial_column_reductions nb_deleted = " << nb_deleted << endl;
		cout << "col_idx=";
		Int_vec_print(cout, col_idx, n);
		cout << endl;
	}


	diophant *D2;

	D2 = NEW_OBJECT(diophant);
	D2->open(m, n - nb_deleted, verbose_level - 1);
	D2->f_has_sum = f_has_sum;
	D2->sum = sum;
	//D2->f_x_max = f_x_max;
	D2->f_has_var_labels = true;
	D2->var_labels = NEW_int(n - nb_deleted);
	for (i = 0; i < m; i++) {
		D2->type[i] = type[i];
		D2->RHS[i] = RHS[i];
		for (j = 0; j < n; j++) {
			if (f_deleted[j]) {
			}
			else {
				h = col_idx[j];
				D2->Aij(i, h) = Aij(i, j);
			}
		}
	}
	for (j = 0; j < n; j++) {
		if (f_deleted[j]) {
		}
		else {
			h = col_idx[j];
			D2->var_labels[h] = j;
			D2->x_max[h] = x_max[j];
			D2->x_min[h] = x_min[j];
		}
	}
	FREE_int(f_deleted);
	FREE_int(col_idx);

	return D2;

}

int diophant::count_non_zero_coefficients_in_row(
		int i)
{
	int j, d, a;
	
	d = 0;
	for (j = 0; j < n; j++) {
		a = Aij(i, j);
		if (a) {
			d++;
		}
	}
	return d;
}

void diophant::coefficient_values_in_row(
		int i, int &nb_values,
	int *&values, int *&multiplicities, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, j, k, idx;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "diophant::coefficient_values_in_row" << endl;
	}
	nb_values = 0;
	values = NEW_int(n);
	multiplicities = NEW_int(n);
	for (j = 0; j < n; j++) {
		a = Aij(i, j);
		if (a) {
			if (!Sorting.int_vec_search(values, nb_values, a, idx)) {
				for (k = nb_values; k > idx; k--) {
					values[k] = values[k - 1];
					multiplicities[k] = multiplicities[k - 1];
				}
				values[idx] = a;
				multiplicities[idx] = 1;
				nb_values++;
			}
			else {
				multiplicities[idx]++;
			}
		}
	}
	
}

int diophant::maximum_number_of_non_zero_coefficients_in_row()
{
	int i, d_max = 0, d;
	
	for (i = 0; i < m; i++) {
		d = count_non_zero_coefficients_in_row(i);
		d_max = MAXIMUM(d, d_max);
	}
	return d_max;
}

void diophant::get_coefficient_matrix(
		int *&M,
	int &nb_rows, int &nb_cols, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	
	if (f_v) {
		cout << "diophant::get_coefficient_matrix" << endl;
	}
	nb_rows = m;
	nb_cols = n;
	M = NEW_int(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			M[i * n + j] = Aij(i, j);
		}
	}
	if (f_v) {
		cout << "diophant::get_coefficient_matrix done" << endl;
	}
}

#if 0
void diophant::save_as_Levi_graph(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "diophant::save_as_Levi_graph" << endl;
		}

	{
		colored_graph *CG;
		int *M;
		int nb_rows, nb_cols;

		if (f_v) {
			cout << "diophant::save_as_Levi_graph before create_Levi_graph_"
					"from_coefficient_matrix" << endl;
		}
	
		get_coefficient_matrix(M, nb_rows, nb_cols, verbose_level - 1);

		CG = NEW_OBJECT(colored_graph);

		CG->create_Levi_graph_from_incidence_matrix(
			M, nb_rows, nb_cols,
			false /* f_point_labels */,
			NULL /* *point_labels */,
			verbose_level);
		if (f_v) {
			cout << "diophant::save_as_Levi_graph after create_Levi_graph_"
					"from_coefficient_matrix" << endl;
		}
		CG->save(fname, verbose_level);
		FREE_OBJECT(CG);
	}

	if (f_v) {
		cout << "diophant::save_as_Levi_graph done" << endl;
	}
}
#endif

#if 0
void diophant::save_in_compact_format(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, d;
	file_io Fio;
	
	if (f_v) {
		cout << "diophant::save_in_compact_format" << endl;
	}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = Aij(i, j);
			if (a > 1) {
				cout << "diophant::save_in_compact_format coefficient "
					"matrix must be 0/1" << endl;
				exit(1);
			}
		}
	}
	{
		ofstream fp(fname);

		fp << "% " << fname << endl;
		fp << m << " " << n << " " << f_has_sum << " " << sum << endl;
		for (i = 0; i < m; i++) {
			fp << i << " ";
			if (type[i] == t_EQ) {
				fp << "EQ" << " " << RHS[i];
				}
			else if (type[i] == t_LE) {
				fp << "LE" << " " << RHS[i];
				}
			else if (type[i] == t_INT) {
				fp << "INT" << " " << RHS_low[i] << " " << RHS[i];
				}
			else if (type[i] == t_ZOR) {
				fp << "ZOR" << " " << RHS[i];
				}

			d = count_non_zero_coefficients_in_row(i);
	
			fp << " " << d;
			for (j = 0; j < n; j++) {
				a = Aij(i, j);
				if (a) {
					fp << " " << j;
					}
				}
			fp << endl;
			}
		fp << "END" << endl;
	}
	if (f_v) {
		cout << "diophant::save_in_compact_format done, " << endl;
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
}

void diophant::read_compact_format(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m, n, s;
	int cnt, i, d, h, a;
	int f_has_sum1;
	
	if (f_v) {
		cout << "diophant::read_compact_format" << endl;
		}
	string line;
	string EQ("EQ");
	string LE("LE");
	string INT("INT");
	string ZOR("ZOR");
	{
	ifstream myfile (fname);
	if (myfile.is_open()) {
		getline (myfile, line); // file name
		getline (myfile, line); // m n sum

		i = line.find(" ");
		
		string str = line.substr(0, i);
		string remainder = line.substr(i + 1);

		//cout << "substring ='" << str << "'" << endl;
		m = atoi(str.c_str()); // stoi(str) is C++11
		//cout << "remainder ='" << remainder << "'" << endl;
		i = remainder.find(" ");
		str = remainder.substr(0, i);
		//cout << "substring ='" << str << "'" << endl;
		n = atoi(str.c_str());
		string remainder2 = remainder.substr(i + 1);

		i = remainder2.find(" ");
		str = remainder2.substr(0, i);
		//cout << "substring ='" << str << "'" << endl;
		f_has_sum1 = atoi(str.c_str());

		str = remainder2.substr(i + 1);
		s = atoi(str.c_str());
		//cout << "diophant::read_compact_format m=" << m << " n=" << n << " sum=" << s << endl;


		open(m, n);
		f_has_sum = f_has_sum1;
		sum = s;


		for (cnt = 0; cnt < m; cnt++) {
			getline (myfile, line);
			i = line.find(" ");
			remainder = line.substr(i + 1);
			line = remainder;
			//cout << "remainder = '" << remainder << "'" << endl;
			i = line.find(" ");
			str = line.substr(0, i);
			remainder = line.substr(i + 1);
			line = remainder;
			if (str.compare(EQ) == 0) {
				//cout << "equal" << endl;
				type[cnt] = t_EQ;
				}
			else if (str.compare(LE) == 0) {
				//cout << "less than or equal" << endl;
				type[cnt] = t_LE;
				}
			else if (str.compare(INT) == 0) {
				//cout << "interval" << endl;
				type[cnt] = t_INT;
				}
			else if (str.compare(ZOR) == 0) {
				//cout << "less than or equal" << endl;
				type[cnt] = t_ZOR;
				}
			else {
				cout << "cannot find EQ or LE or ZOR" << endl;
				exit(1);
				}
			//cout << "remainder = '" << line << "'" << endl;
			i = line.find(" ");
			str = line.substr(0, i);
			remainder = line.substr(i + 1);
			line = remainder;
			RHSi(cnt) = atoi(str.c_str());
			//cout << "rhs = " << RHS[cnt] << endl;
			//cout << "remainder = '" << line << "'" << endl;

			if (type[cnt] == t_INT) {
				RHS_low_i(cnt) = RHSi(cnt);
				i = line.find(" ");
				str = line.substr(0, i);
				remainder = line.substr(i + 1);
				line = remainder;
				RHSi(cnt) = atoi(str.c_str());
			}

			i = line.find(" ");
			str = line.substr(0, i);
			d = atoi(str.c_str());
			remainder = line.substr(i + 1);
			line = remainder;

			//cout << "d = " << d << endl;
			for (h = 0; h < d; h++) {
				i = line.find(" ");
				str = line.substr(0, i);
				a = atoi(str.c_str());
				remainder = line.substr(i + 1);
				line = remainder;
				Aij(cnt, a) = 1;
				}
			

			} // next cnt
		//cout << "read " << cnt << " lines" << endl;
		
#if 0
		while ( getline (myfile, line) ) {
			cout << line << '\n';
			
			}
#endif
		myfile.close();
		}
	else {
		cout << "Cannot open file " << fname << endl;
		exit(1);
		}

	if (f_v) {
		cout << "diophant::read_compact_format read system with " << m
			<< " rows and " << n << " columns and f_has_sum = " << f_has_sum
			<< " and sum " << sum << endl;
		}
	}
}
#endif

void diophant::save_in_general_format(
		std::string &fname, int verbose_level)
// ToDo this does not save the values of x_min[] and x_max[]
{
	int f_v = (verbose_level >= 1);
	int i, j, a, d, h, val;
	other::orbiter_kernel_system::file_io Fio;
	other::data_structures::string_tools ST;
	
	if (f_v) {
		cout << "diophant::save_in_general_format" << endl;
		}

#if 0
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = Aij(i, j);
			if (a > 1) {
				cout << "diophant::save_in_general_format coefficient "
						"matrix must be 0/1" << endl;
				exit(1);
				}
			}
		}
#endif

	std::string fname_coeff;
	std::string fname_RHS;
	std::string fname_x_bounds;

	fname_coeff.assign(fname);
	fname_RHS.assign(fname);
	fname_x_bounds.assign(fname);

	ST.replace_extension_with(fname_coeff, "_coeff_matrix.csv");
	ST.replace_extension_with(fname_RHS, "_RHS.csv");
	ST.replace_extension_with(fname_x_bounds, "_x_bounds.csv");

	Fio.Csv_file_support->int_matrix_write_csv(
			fname_coeff, A, m, n);
	if (f_v) {
		cout << "diophant::save_in_general_format "
				"written file " << fname_coeff << " of size "
				<< Fio.file_size(fname_coeff) << endl;
	}

	int *RHS_coded;
	RHS_coded = NEW_int(m * 3);
	for (i = 0; i < m; i++) {
		RHS_coded[3 * i + 0] = RHS_low[i];
		RHS_coded[3 * i + 1] = RHS[i];
		if (type[i] == t_EQ) {
			RHS_coded[3 * i + 0] = RHS[i];
			RHS_coded[3 * i + 2] = 1;
		}
		else if (type[i] == t_LE) {
			RHS_coded[3 * i + 2] = 2;
		}
		else if (type[i] == t_INT) {
			RHS_coded[3 * i + 2] = 3;
		}
		else if (type[i] == t_ZOR) {
			RHS_coded[3 * i + 2] = 4;
		}
		else {
			cout << "type[i] is not recognized" << endl;
			exit(1);
		}
	}
	Fio.Csv_file_support->int_matrix_write_csv(
			fname_RHS, RHS_coded, m, 3);
	if (f_v) {
		cout << "diophant::save_in_general_format "
				"written file " << fname_RHS << " of size "
				<< Fio.file_size(fname_RHS) << endl;
	}
	FREE_int(RHS_coded);

	int *X_bounds;
	X_bounds = NEW_int(n * 2);
	for (j = 0; j < n; j++) {
		X_bounds[2 * j + 0] = x_min[j];
		X_bounds[2 * j + 1] = x_max[j];
	}

	Fio.Csv_file_support->int_matrix_write_csv(
			fname_x_bounds, X_bounds, n, 2);
	FREE_int(X_bounds);
	if (f_v) {
		cout << "diophant::save_in_general_format "
				"written file " << fname_x_bounds << " of size "
				<< Fio.file_size(fname_x_bounds) << endl;
	}

	{
		ofstream fp(fname);

		fp << "% diophantine system in general format " << fname << endl;
		fp << m << " " << n << " " << f_has_sum << " " << sum << " "
				<< f_has_var_labels << endl;
		if (f_has_var_labels) {
			for (j = 0; j < n; j++) {
				fp << var_labels[j] << " ";
			}
			fp << endl;
		}
		for (i = 0; i < m; i++) {
			fp << i << " ";
			if (type[i] == t_EQ) {
				fp << "EQ" << " " << RHS[i];
			}
			else if (type[i] == t_LE) {
				fp << "LE" << " " << RHS[i];
			}
			else if (type[i] == t_INT) {
				fp << "INT" << " " << RHS_low[i] << " " << RHS[i];
			}
			else if (type[i] == t_ZOR) {
				fp << "ZOR" << " " << RHS[i];
			}


			int nb_values;
			int *values, *multiplicities;
			int d1;


			coefficient_values_in_row(i, nb_values,
				values, multiplicities, 0 /*verbose_level*/);

	
			//d = count_non_zero_coefficients_in_row(i);
	
			fp << " " << nb_values;
			for (h = 0; h < nb_values; h++) {
				val = values[h];
				d = multiplicities[h];
				fp << " " << val;
				fp << " " << d;
				d1 = 0;
				for (j = 0; j < n; j++) {
					a = Aij(i, j);
					if (a == val) {
						fp << " " << j;
						d1++;
					}
				}
				if (d1 != d) {
					cout << "d1 != d" << endl;
					cout << "i=" << i << endl;
					cout << "val=" << val << endl;
					cout << "d=" << d << endl;
					cout << "d1=" << d1 << endl;
					exit(1);
				}
			}
			fp << endl;

			FREE_int(values);
			FREE_int(multiplicities);

		}
		for (j = 0; j < n; j++) {
			fp << x_min[j] << " " << x_max[j] << endl;
		}
		fp << "END" << endl;
	}
	if (f_v) {
		cout << "diophant::save_in_general_format done, " << endl;
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
}

void diophant::read_general_format(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 1);
	int m, n, s;
	int cnt, i, j, d, h, a, nb_types, t, val;
	int f_has_sum1;
	int f_has_var_labels_save = false;
	
	if (f_v) {
		cout << "diophant::read_general_format" << endl;
	}
	string line;
	string str_EQ; str_EQ.assign("EQ");
	string str_LE; str_LE.assign("LE");
	string str_INT; str_INT.assign("INT");
	string str_ZOR; str_ZOR.assign("ZOR");
	{
	ifstream myfile (fname);
	if (myfile.is_open()) {
		if (f_vv) {
			cout << "diophant::read_general_format" << endl;
		}
		getline (myfile, line); // file name


		//cout << "diophant::read_general_format parsing '" << line << "'" << endl;
		getline (myfile, line); // m n sum

		i = line.find(" ");
		
		string str = line.substr(0, i);
		string remainder = line.substr(i + 1);

		//cout << "substring ='" << str << "'" << endl;
		m = atoi(str.c_str()); // stoi(str) is C++11
		//cout << "remainder ='" << remainder << "'" << endl;
		i = remainder.find(" ");
		str = remainder.substr(0, i);
		//cout << "substring ='" << str << "'" << endl;
		n = atoi(str.c_str());
		string remainder2 = remainder.substr(i + 1);

		i = remainder2.find(" ");
		str = remainder2.substr(0, i);
		//cout << "substring ='" << str << "'" << endl;
		f_has_sum1 = atoi(str.c_str());

		string remainder3 = remainder2.substr(i + 1);
		i = remainder3.find(" ");
		str = remainder3.substr(0, i);
		s = atoi(str.c_str());
		string remainder4 = remainder3.substr(i + 1);
		f_has_var_labels_save = atoi(remainder4.c_str());


		//cout << "diophant::read_general_format "
		//		"m=" << m << " n=" << n << " remainder3=" << remainder3 << endl;

		//str = remainder3.substr(i + 1);
		if (f_vv) {
			cout << "diophant::read_general_format "
					"m=" << m << " n=" << n << " sum=" << s
					<< " f_has_var_labels=" << f_has_var_labels_save << endl;
		}

		open(m, n, verbose_level - 1);
		f_has_var_labels = f_has_var_labels_save;
		f_has_sum = f_has_sum1;
		sum = s;

		if (f_has_var_labels) {

			if (f_vv) {
				cout << "reading var labels" << endl;
			}
			var_labels = NEW_int(n);
			getline (myfile, line);
			for (j = 0; j < n; j++) {

				// read the value:
				i = line.find(" ");
				str = line.substr(0, i);
				var_labels[j] = atoi(str.c_str());
				remainder = line.substr(i + 1);
				line = remainder;
			}
		}
		else {
			if (f_vv) {
				cout << "not reading var labels" << endl;
			}
		}

		for (cnt = 0; cnt < m; cnt++) {
			if (f_v) {
				cout << "reading equation " << cnt << " / " << m << ":" << endl;
			}
			getline (myfile, line);
			i = line.find(" ");
			remainder = line.substr(i + 1);
			line = remainder;
			if (f_v) {
				cout << "remainder = '" << remainder << "'" << endl;
			}
			i = line.find(" ");
			str = line.substr(0, i);
			remainder = line.substr(i + 1);
			line = remainder;
			if (str.compare(str_EQ) == 0) {
				//cout << "equal" << endl;
				type[cnt] = t_EQ;
			}
			else if (str.compare(str_LE) == 0) {
				//cout << "less than or equal" << endl;
				type[cnt] = t_LE;
			}
			else if (str.compare(str_INT) == 0) {
				//cout << "less than or equal" << endl;
				type[cnt] = t_INT;
			}
			else if (str.compare(str_ZOR) == 0) {
				//cout << "less than or equal" << endl;
				type[cnt] = t_ZOR;
			}
			else {
				cout << "cannot find EQ or LE or ZOR" << endl;
				exit(1);
			}
			//cout << "remainder = '" << line << "'" << endl;

			// read the RHS:

			i = line.find(" ");
			str = line.substr(0, i);
			remainder = line.substr(i + 1);
			line = remainder;
			RHSi(cnt) = atoi(str.c_str());
			//cout << "rhs = " << RHS[cnt] << endl;
			//cout << "remainder = '" << line << "'" << endl;
			if (type[cnt] == t_INT) {
				RHS_low_i(cnt) = RHSi(cnt);
				i = line.find(" ");
				str = line.substr(0, i);
				remainder = line.substr(i + 1);
				line = remainder;
				RHSi(cnt) = atoi(str.c_str());
				//cout << "rhs_low = " << RHS_low[cnt] << endl;
				//cout << "rhs = " << RHS[cnt] << endl;
				//cout << "remainder = '" << line << "'" << endl;
			}

			// read nb_types:
			i = line.find(" ");
			str = line.substr(0, i);
			nb_types = atoi(str.c_str());
			remainder = line.substr(i + 1);
			line = remainder;

			for (t = 0; t < nb_types; t++) {

				// read the value:
				i = line.find(" ");
				str = line.substr(0, i);
				val = atoi(str.c_str());
				remainder = line.substr(i + 1);
				line = remainder;

				// read the multiplicity:
				i = line.find(" ");
				str = line.substr(0, i);
				d = atoi(str.c_str());
				remainder = line.substr(i + 1);
				line = remainder;

				// read the coefficients:

				//cout << "d = " << d << endl;
				for (h = 0; h < d; h++) {
					i = line.find(" ");
					str = line.substr(0, i);
					a = atoi(str.c_str());
					remainder = line.substr(i + 1);
					line = remainder;
					Aij(cnt, a) = val;
				
				}
			}
			

		} // next cnt
		//cout << "read " << cnt << " lines" << endl;
		
#if 0
		while ( getline (myfile, line) ) {
			cout << line << '\n';
			
		}
#endif
		x_min = NEW_int(n);
		x_max = NEW_int(n);
		for (j = 0; j < n; j++) {
			getline (myfile, line);

			// read the value:
			i = line.find(" ");
			str = line.substr(0, i);
			x_min[j] = atoi(str.c_str());
			remainder = line.substr(i + 1);
			line = remainder;
			x_max[j] = atoi(remainder.c_str());
		}

		myfile.close();
	}
	else {
		cout << "Cannot open file " << fname << endl;
		exit(1);
	}

	if (f_v) {
		cout << "diophant::read_general_format read system with " << m
			<< " rows and " << n << " columns and f_has_sum=" << f_has_sum
			<< " sum " << sum << endl;
		}
	}
}

void diophant::eliminate_zero_rows_quick(
		int verbose_level)
{
	int *eqn_number;
	eliminate_zero_rows(eqn_number, verbose_level);
	FREE_int(eqn_number);
}

void diophant::eliminate_zero_rows(
		int *&eqn_number, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, mm;
	int f_delete = false;
	
	eqn_number = NEW_int(m);
	mm = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (Aij(i, j)) {
				break;
			}
		}
		if (j < n) {
			f_delete = false;
		}
		else {
			f_delete = true;
		}
		if (f_delete && type[i] == t_EQ && RHS[i]) {
			f_delete = false;
		}
		if (!f_delete) {
			eqn_number[mm] = i;
			if (i != mm) {
				for (j = 0; j < n; j++) {
					Aij(mm, j) = Aij(i, j);
				}
				RHS[mm] = RHS[i];
				RHS_low[mm] = RHS_low[i];
				type[mm] = type[i];
				eqn_label[mm].assign(eqn_label[i]);
			}
			mm++;
		}
		else {
		}
	}
	if (f_v) {
		cout << "eliminate_zero_rows: eliminated " << m - mm
				<< " zero rows" << endl;
	}
	m = mm;
}

int diophant::is_zero_outside(
		int first, int len, int i)
{
	int j;
	
	for (j = 0; j < n; j++) {
		if (j >= first && j < first + len) {
			continue;
		}
		if (Aij(i, j)) {
			return false;
		}
	}
	return true;
}

void diophant::project(
		diophant *D, int first, int len,
		int *&eqn_number, int &nb_eqns_replaced, int *&eqns_replaced,
		int verbose_level)
{
	int i, j, f_zo;
	
	D->open(m, len, verbose_level - 1);
	nb_eqns_replaced = 0;
	eqns_replaced = NEW_int(m);
	for (i = 0; i < m; i++) {
		f_zo = is_zero_outside(first, len, i);
		if (f_zo) {
			eqns_replaced[nb_eqns_replaced++] = i;
		}
		for (j = 0; j < len; j++) {
			D->Aij(i, j) = Aij(i, first + j);
		}
		D->RHS_low[i] = RHS_low[i];
		D->RHS[i] = RHS[i];
		D->type[i] = type[i];
		if (!f_zo) {
			D->type[i] = t_LE;
		}
		D->init_eqn_label(i, eqn_label[i]);
	}
	//D->f_x_max = f_x_max;
	//if (f_x_max) {
		for (j = 0; j < len; j++) {
			D->x_max[j] = x_max[first + j];
			D->x_min[j] = x_min[first + j];
		}
	//}
	D->f_has_sum = f_has_sum;
	D->sum = sum;
	D->eliminate_zero_rows(eqn_number, 0);
}

void diophant::split_by_equation(
		int eqn_idx,
		int f_solve_case,
		int solve_case_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::split_by_equation" << endl;
	}
	diophant *D;
	long int nb_backtrack_nodes;
	int max_results = INT_MAX;
	int N, a;

	D = NEW_OBJECT(diophant);
	a = count_non_zero_coefficients_in_row(eqn_idx);
	project_to_single_equation(D, eqn_idx, verbose_level);
	if (f_v) {
		cout << "equation " << eqn_idx << " has " << a
				<< " nonzero coefficients:" << endl;
		D->print_dense();
	}
	D->solve_all_mckay(nb_backtrack_nodes, max_results, 0 /* verbose_level */);
	N = D->_resultanz;
	if (f_v) {
		cout << "equation " << eqn_idx << " has " << N << " solutions." << endl;
		D->print_dense();
	}

	int j;
	int *Sol;
	int nb_sol;

	D->get_solutions(Sol, nb_sol, verbose_level);

	if (f_solve_case) {
		if (f_v) {
			cout << "Solution " << solve_case_idx << " is ";
			for (j = 0; j < n; j++) {
				if (Sol[solve_case_idx * n + j] == 0) {
					continue;
				}
				cout << j << ", ";
			}
			cout << endl;
		}
		cout << "Solution " << solve_case_idx << " is ";
		for (j = 0; j < n; j++) {
			if (D->x_min[j] != D->x_max[j]) {
				cout << "x_" << j << " = " << Sol[solve_case_idx * n + j] << ", ";
			}
		}
		cout << endl;

		for (j = 0; j < n; j++) {
			if (D->x_min[j] != D->x_max[j]) {
				x_min[j] = x_max[j] = Sol[solve_case_idx * n + j];
			}
		}
		cout << "solving case " << solve_case_idx << endl;
		solve_all_mckay(nb_backtrack_nodes, INT_MAX, 0 /*verbose_level */);
		cout << "solved case " << solve_case_idx
				<< " with " << _resultanz << " solutions" << endl;

	}
	FREE_int(Sol);

	FREE_OBJECT(D);

}

void diophant::split_by_two_equations(
		int eqn1_idx, int eqn2_idx,
		int f_solve_case,
		int solve_case_idx_r, int solve_case_idx_m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::split_by_two_equations" << endl;
	}
	diophant *D;
	long int nb_backtrack_nodes;
	int max_results = INT_MAX;
	int N;

	D = NEW_OBJECT(diophant);
	project_to_two_equations(D, eqn1_idx, eqn2_idx, verbose_level);
	if (f_v) {
		cout << "equations (" << eqn1_idx << ", " << eqn2_idx << "):" << endl;
		D->print_dense();
	}
	D->solve_all_mckay(nb_backtrack_nodes, max_results, 0 /* verbose_level */);
	N = D->_resultanz;
	if (f_v) {
		cout << "equations (" << eqn1_idx << ", " << eqn2_idx << ") "
				"have " << N << " solutions." << endl;
		D->print_dense();
	}

	int j;
	int *Sol;
	int nb_sol;
	int idx;

	D->get_solutions(Sol, nb_sol, verbose_level);

	if (f_solve_case) {
		for (idx = 0; idx < nb_sol; idx++) {
			if ((idx % solve_case_idx_m) != solve_case_idx_r) {
				continue;
			}
			if (f_v) {
				cout << "Solution " << idx << " is ";
				for (j = 0; j < n; j++) {
					if (Sol[idx * n + j] == 0) {
						continue;
					}
					cout << j << ", ";
				}
				cout << endl;
			}
			cout << "Solution " << idx << " is ";
			for (j = 0; j < n; j++) {
				if (D->x_min[j] != D->x_max[j]) {
					cout << "x_" << j << " = " << Sol[idx * n + j] << ", ";
				}
			}
			cout << endl;

			for (j = 0; j < n; j++) {
				if (D->x_min[j] != D->x_max[j]) {
					x_min[j] = x_max[j] = Sol[idx * n + j];
				}
			}
			cout << "solving case " << idx << endl;
			solve_all_mckay(nb_backtrack_nodes, max_results, 0 /*verbose_level */);
			cout << "solved case " << idx << " with " << _resultanz << " solutions" << endl;
		}
	}
	FREE_int(Sol);

	FREE_OBJECT(D);

}

void diophant::project_to_single_equation_and_solve(
		int max_number_of_coefficients,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::project_to_single_equation_and_solve" << endl;
		cout << "max_number_of_coefficients = " << max_number_of_coefficients << endl;
	}
	diophant *D;
	long int *nb_sol_in_eqn;
	long int nb_backtrack_nodes;
	int max_results = 10000;
	int *eqn_idx;
	int i, h, a;

	nb_sol_in_eqn = NEW_lint(m);
	eqn_idx = NEW_int(m);
	h = 0;
	for (i = 0; i < m; i++) {
		a = count_non_zero_coefficients_in_row(i);
		if (f_v) {
			cout << "equation " << i << " has " << a << " nonzero coefficients" << endl;
		}
		if (a < max_number_of_coefficients) {
			D = NEW_OBJECT(diophant);
			project_to_single_equation(D, i, verbose_level);
			if (f_v) {
				cout << "equation " << i << " has " << a << " nonzero coefficients:" << endl;
				D->print_dense();
			}
			D->solve_all_mckay(nb_backtrack_nodes, max_results, 0 /* verbose_level */);
			eqn_idx[h] = i;
			nb_sol_in_eqn[h] = D->_resultanz;
			h++;
			FREE_OBJECT(D);
		}
		else {
			if (f_v) {
				cout << "ignoring equation " << i << endl;
			}
		}
	}

	cout << "number of solutions of individual equations:" << endl;
	for (i = 0; i < h; i++) {
		cout << setw(3) << eqn_idx[i] << " : " << setw(4) << nb_sol_in_eqn[i] << endl;
	}

	other::data_structures::tally C;

	C.init_lint(nb_sol_in_eqn, h, false, 0);
	cout << "number of solutions of individual equations classified:" << endl;
	C.print(true);

	FREE_int(eqn_idx);
	FREE_lint(nb_sol_in_eqn);

	if (f_v) {
		cout << "diophant::project_to_single_equation_and_solve done" << endl;
	}
}

void diophant::project_to_single_equation(
		diophant *D, int eqn_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;

	if (f_v) {
		cout << "diophant::project_to_single_equation" << endl;
	}

	D->open(1, n, verbose_level - 1);

	for (j = 0; j < n; j++) {
		D->Aij(0, j) = Aij(eqn_idx, j);
	}
	D->RHS[0] = RHS[eqn_idx];
	D->RHS_low[0] = RHS_low[eqn_idx];
	D->type[0] = type[eqn_idx];

	for (j = 0; j < n; j++) {
		D->x_max[j] = x_max[j];
		D->x_min[j] = x_min[j];

		if (D->Aij(0, j) == 0) {
			D->x_max[j] = 0;
		}
	}

	D->f_has_sum = false;
	D->sum = 0;

	if (f_v) {
		cout << "diophant::project_to_single_equation done" << endl;
	}
}

void diophant::project_to_two_equations(
		diophant *D, int eqn1_idx, int eqn2_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;

	if (f_v) {
		cout << "diophant::project_to_two_equations" << endl;
	}

	D->open(2, n, verbose_level - 1);

	for (j = 0; j < n; j++) {
		D->Aij(0, j) = Aij(eqn1_idx, j);
		D->Aij(1, j) = Aij(eqn2_idx, j);
	}
	D->RHS[0] = RHS[eqn1_idx];
	D->RHS[1] = RHS[eqn2_idx];
	D->RHS_low[0] = RHS_low[eqn1_idx];
	D->RHS_low[1] = RHS_low[eqn2_idx];
	D->type[0] = type[eqn1_idx];
	D->type[1] = type[eqn2_idx];

	for (j = 0; j < n; j++) {
		D->x_max[j] = x_max[j];
		D->x_min[j] = x_min[j];

		if (D->Aij(0, j) == 0 && D->Aij(1, j) == 0) {
			D->x_max[j] = 0;
		}
	}

	D->f_has_sum = false;
	D->sum = 0;

	if (f_v) {
		cout << "diophant::project_to_two_equations done" << endl;
	}
}

void diophant::multiply_A_x_to_RHS1()
{
	int i, j, a;
	
	for (i = 0; i < m; i++) {
		a = 0;
		for (j = 0; j < n; j++) {
			a += Aij(i, j) * x[j];
		}
		RHS1[i] = a;
	}
}

void diophant::write_xml(std::ostream &ost, std::string &label)
{
	int i, j;
	string equation_label;
	
	ost << "<DIOPHANT label=\"" << label << "\" num_eqns=" << m
			<< " num_vars=" << n << " f_has_sum=" << f_has_sum
			<< " sum=" << sum << " >" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0;j < n; j++) {
			ost << setw(4) << Aij(i, j) << " ";
		}
		equation_label.assign(eqn_label[i]);
		if (type[i] == t_EQ) {
			ost << setw(2) << 0 << " " << setw(4) << RHS[i];
		}
		else if (type[i] == t_LE) {
			ost << setw(2) << 1 << " " << setw(4) << RHS[i];
		}
		else if (type[i] == t_INT) {
			ost << setw(2) << 2 << " " << setw(4) << RHS_low_i(i) << " " << setw(4) << RHS[i];
		}
		else if (type[i] == t_ZOR) {
			ost << setw(2) << 3 << " " << setw(4) << RHS[i];
		}
		ost << " \"" << equation_label << "\"" << endl;
	}
	for (j = 0; j < n; j++) {
		ost << setw(4) << x_min[j] << " " << setw(4) << x_max[j] << endl;
	}
	ost << "</DIOPHANT>" << endl;
	
}


void diophant::read_xml(ifstream &f, std::string &label, int verbose_level)
{
#ifdef SYSTEMUNIX
	int f_v = (verbose_level >= 1);
	string str, mapkey, mapval;
	bool brk;
	int eqpos, l, M = 0, N = 0, F_has_sum = 0, Sum = 0, i, j, a;
	char c;
	other::data_structures::string_tools ST;


	if (f_v) {
		cout << "diophant::read_xml" << endl;
	}
	f.ignore(INT_MAX, '<');
	f >> str;
	brk = false;
	if (str != "DIOPHANT") {
		cout << "not a DIOPHANT object: str=" << str << endl;
		exit(1);
		}
	while (!brk) {
		f >> str;
		if (str.substr(str.size() - 1, 1) == ">") {
			str = str.substr(0, str.size() - 1);
			brk = true;
			}
		eqpos = (int) str.find("=");
		if (eqpos > 0) {
			mapkey = str.substr(0, eqpos);
			mapval = str.substr(eqpos + 1, str.size() - eqpos - 1);
			if (mapkey == "label") {
				label.assign(mapval);
			}
			else if (mapkey == "num_eqns") {
				M = ST.str2int(mapval);
				if (f_v) {
					cout << "diophant::read_xml "
							"num_eqns = " << M << endl;
				}
			}
			else if (mapkey == "num_vars") {
				N = ST.str2int(mapval);
				if (f_v) {
					cout << "diophant::read_xml "
							"num_vars = " << N << endl;
				}
			}
			else if (mapkey == "f_has_sum") {
				F_has_sum = ST.str2int(mapval);
				if (f_v) {
					cout << "diophant::read_xml "
							"F_has_sum = " << F_has_sum << endl;
				}
			}
			else if (mapkey == "sum") {
				Sum = ST.str2int(mapval);
				if (f_v) {
					cout << "diophant::read_xml "
							"Sum = " << Sum << endl;
				}
			}
		}
		brk = brk || f.eof();
	}
	if (f_v) {
		cout << "diophant::read_xml M=" << M << " N=" << N << endl;
	}
	open(M, N, verbose_level - 1);
	f_has_sum = F_has_sum;
	sum = Sum;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			f >> a;
			Aij(i, j) = a;
		}
		int t;
		f >> t;
		if (t == 0) {
			type[i] = t_EQ;
		}
		else if (t == 1) {
			type[i] = t_LE;
		}
		else if (t == 2) {
			type[i] = t_INT;
		}
		else if (t == 3) {
			type[i] = t_ZOR;
		}
		f >> RHS[i];
		if (type[i] == t_INT) {
			RHS_low[i] = RHS[i];
			f >> RHS[i];
		}

		//f.ignore(INT_MAX, '\"');
		while (true) {
			f >> c;
			if (c == '\"') {
				break;
			}
		}
		char tmp[1000];
		l = 0;
		while (true) {
			f >> c;
			if (c == '\"') {
				break;
			}
			tmp[l] = c;
			l++;
		}
		tmp[l] = 0;
		eqn_label[i].assign(tmp);
	}
	if (f_v) {
		cout << "diophant::read_xml reading x_max[]" << endl;
	}
	for (j = 0; j < n; j++) {
		f >> x_min[j] >> x_max[j];
		if (f_v) {
			cout << "diophant::read_xml reading x_max[" << j << "]="
					<< x_max[j] << endl;
		}
	}
	if (f_v) {
		cout << "diophant::read_xml read the following file:" << endl;
		write_xml(cout, label);
	}
#endif
#ifdef SYSTEMWINDOWS
	cout << "diophant::read_xml has a problem under windows"<< endl;
	exit(1);
#endif
}

void diophant::append_equation()
{
	int *AA, *R_low, *R, *R1, *Y1;
	diophant_equation_type *type1;
	int m1 = m + 1;
	int i, j;

	AA = NEW_int(m1 * n);
	R_low = NEW_int(m1);
	R = NEW_int(m1);
	R1 = NEW_int(m1);
	type1 = NEW_OBJECTS(diophant_equation_type, m1);
	Y1 = NEW_int(m1);
	
	for (i = 0; i < m; i++) {

		for (j = 0; j < n; j++) {
			AA[i * n + j] = Aij(i, j);
		}
		R_low[i] = RHS_low[i];
		R[i] = RHS[i];
		R1[i] = RHS1[i];
		type1[i] = type[i];
	}

	FREE_int(A);
	FREE_int(RHS_low);
	FREE_int(RHS);
	FREE_int(RHS1);
	FREE_OBJECTS(type);
	FREE_int(Y);

	A = AA;
	RHS_low = R_low;
	RHS = R;
	RHS1 = R1;
	type = type1;
	Y = Y1;

	Int_vec_zero(A + m * n, n);
	RHS_low[m] = 0;
	RHS[m] = 0;
	RHS1[m] = 0;
	type[m] = t_EQ;

	m++;
	
}

void diophant::delete_equation(
		int I)
{
	int i, j;
	
	for (i = I; i < m - 1; i++) {
		eqn_label[i].assign(eqn_label[i + 1]);
		type[i] = type[i + 1];
		RHS_low[i] = RHS_low[i + 1];
		RHS[i] = RHS[i + 1];
		for (j = 0; j < n; j++) {
			Aij(i, j) = Aij(i + 1, j);
		}
	}
	m--;
}

void diophant::write_gurobi_binary_variables(
		std::string &fname)
{
	int i, j, a;
	other::orbiter_kernel_system::file_io Fio;

	{
		ofstream f(fname);
		f << "Maximize" << endl;
		f << "  ";
		for (j = 0; j < n; j++) {
			f << " + 0 x" << j;
		}
		f << endl;
		f << "  Subject to" << endl;
		for (i = 0; i < m; i++) {
			f << "  ";
			for (j = 0; j < n; j++) {
				a = Aij(i, j);
				if (a == 0) {
					continue;
				}
				f << " + " << a << " x"<< j;
			}
			f << " = " << RHSi(i) << endl;
		}
		f << "Binary" << endl;
		for (i = 0; i < n; i++) {
			f << "x" << i << endl;
		}
		f << "End" << endl;
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
}

void diophant::draw_as_bitmap(
		std::string &fname,
		int f_box_width, int box_width,
		int bit_depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::draw_as_bitmap" << endl;
	}

	int *M;
	int m1 = m + 1;
	int n1 = n + 1;
	int i, j, a;

	M = NEW_int(m1 * n1);
	for (i = 0; i < m1; i++) {
		for (j = 0; j < n1; j++) {
			if (i == m) {
				if (j == n) {
					a = 0;
				}
				else {
					a = x_max[j];
				}
			}
			else {
				if (j == n) {
					a = RHS[i];
				}
				else {
					a = Aij(i, j);
				}
			}
			M[i * n1 + j] = a;
		}
	}


	other::l1_interfaces::easy_BMP_interface BMP;
	other::graphics::draw_bitmap_control Draw_bitmap_control;

	Draw_bitmap_control.M = M;
	Draw_bitmap_control.m = m1;
	Draw_bitmap_control.n = n1;
	Draw_bitmap_control.f_box_width = f_box_width;
	Draw_bitmap_control.box_width = box_width;
	Draw_bitmap_control.bit_depth = bit_depth;
	Draw_bitmap_control.f_invert_colors = false;


	BMP.draw_bitmap(&Draw_bitmap_control, verbose_level);

	FREE_int(M);

	if (f_v) {
		cout << "diophant::draw_as_bitmap done" << endl;
	}

}

void diophant::draw_it(
		std::string &fname_base,
		other::graphics::layered_graph_draw_options *Draw_options,
		int verbose_level)
{
	int f_dots = false;
	int f_partition = false;
	int f_bitmatrix = false;
	int f_row_grid = false;
	int f_col_grid = false;
	combinatorics::graph_theory::graph_theory_domain Graph;


	Graph.draw_bitmatrix(
			fname_base,
			Draw_options,
			f_dots,
		f_partition, 0, NULL, 0, NULL, 
		f_row_grid, f_col_grid, 
		f_bitmatrix, NULL, A, 
		m, n,
		false, NULL,
		verbose_level);
}

void diophant::draw_partitioned(
		std::string &fname_base,
		other::graphics::layered_graph_draw_options *Draw_options,
	int f_solution, int *solution, int solution_sz, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_dots = false;
	int f_bitmatrix = false;
	int i, ii, j, jj;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	combinatorics::graph_theory::graph_theory_domain Graph;
	
	
	if (f_v) {
		cout << "diophant::draw_partitioned" << endl;
	}
	
	int *T;
	int *A2;
	int a;

	T = NEW_int(m);
	A2 = NEW_int(m * n);
	Int_vec_zero(A2, m * n);

	for (i = 0; i < m; i++) {
		if (type[i] == t_EQ) {
			T[i] = -RHS[i];
		}
		else if (type[i] == t_LE) {
			T[i] = 2;
		}
		else if (type[i] == t_INT) {
			T[i] = 3;
		}
		else if (type[i] == t_ZOR) {
			T[i] = 4;
		}
	}

	other::data_structures::tally C;

	C.init(T, m, false, 0);
	if (f_v) {
		cout << "diophant::draw_partitioned "
				"we found " << C.nb_types
				<< " classes according to type[]" << endl;
	}
	
	int *part;
	int *part_col;	
	int *col_perm;
	int size_complement;
	int col_part_size;

	part = NEW_int(C.nb_types + 1);
	for (i = 0; i < C.nb_types; i++) {
		part[i] = C.type_first[i];
	}
	part[C.nb_types] = m;

	col_perm = NEW_int(n);

	


	if (f_solution) {
		part_col = NEW_int(3);
		part_col[0] = 0;
		part_col[1] = n - solution_sz;
		part_col[2] = n;

		Int_vec_copy(solution, col_perm + n - solution_sz, solution_sz);
		Combi.set_complement(solution, solution_sz, col_perm, size_complement, n);
		
		if (size_complement != n - solution_sz) {
			cout << "diophant::draw_partitioned size_complement "
					"!= n - solution_sz" << endl;
			exit(1);
		}
		col_part_size = 2;
	}
	else {
		part_col = NEW_int(2);
		part_col[0] = 0;
		part_col[1] = n;
		for (j = 0; j < n; j++) {
			col_perm[j] = j;
		}
		col_part_size = 1;
	}

#if 0
	if (f_v) {
		cout << "row_perm:";
		int_vec_print(cout, C.sorting_perm_inv, m);
		cout << endl;
	}
	if (f_v) {
		cout << "col_perm:";
		int_vec_print(cout, col_perm, n);
		cout << endl;
	}
#endif


	for (i = 0; i < m; i++) {
		ii = C.sorting_perm_inv[i];
		//cout << "i=" << i << " ii=" << ii << endl;
		for (j = 0; j < n; j++) {
			jj = col_perm[j];
			a = Aij(ii, jj);
			A2[i * n + j] = a;
		}
	}
	if (false) {
		cout << "diophant::draw_partitioned A2=" << endl;
		Int_matrix_print(A2, m, n);
	}

	int f_row_grid = false;
	int f_col_grid = false;


	Graph.draw_bitmatrix(
			fname_base,
			Draw_options,
			f_dots,
		true /* f_partition */, C.nb_types, part, col_part_size, part_col, 
		f_row_grid, f_col_grid, 
		f_bitmatrix, NULL,
		A2, m, n,
		false, NULL, verbose_level - 1);


	FREE_int(T);
	FREE_int(A2);
	FREE_int(part);
	FREE_int(part_col);
	FREE_int(col_perm);
	if (f_v) {
		cout << "diophant::draw_partitioned done" << endl;
	}
}

int diophant::test_solution(
		int *sol, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, b, c, ret;

	if (f_v) {
		cout << "diophant::test_solution" << endl;
	}
	if (false) {
		Int_vec_print(cout, sol, len);
		cout << endl;
		other::data_structures::set_of_sets *S;

		get_columns(sol, len, S, 0 /* verbose_level */);
		S->print_table();

		FREE_OBJECT(S);
	}
	Int_vec_zero(Y, m);
	Int_vec_zero(X, n);
	for (j = 0; j < len; j++) {
		X[sol[j]] = 1;
	}
	for (i = 0; i < m; i++) {
		b = 0;
		for (j = 0; j < n; j++) {
			c = Aij(i, j) * X[j];
			b += c;
		}
		Y[i] = b;
	}
	if (false) {
		cout << "Y=";
		Int_vec_print_fully(cout, Y, m);
		cout << endl;
	}
	ret = true;
	for (i = 0; i < m; i++) {
		if (type[i] == t_EQ) {
			if (Y[i] != RHS[i]) {
				cout << "diophant::test_solution "
						"t_EQ and Y[i] != RHS[i]" << endl;
				ret = false;
				break;
			}
		}
		else if (type[i] == t_LE) {
			if (Y[i] > RHS[i]) {
				cout << "diophant::test_solution "
						"t_LE and Y[i] > RHS[i]" << endl;
				ret = false;
				break;
			}

		}
		else if (type[i] == t_INT) {
			if (Y[i] > RHS[i]) {
				cout << "diophant::test_solution "
						"t_INT and Y[i] > RHS[i]" << endl;
				ret = false;
				break;
			}
			if (Y[i] < RHS_low[i]) {
				cout << "diophant::test_solution "
						"t_INT and Y[i] < RHS_low[i]" << endl;
				ret = false;
				break;
			}

		}
		else if (type[i] == t_ZOR) {
			if (Y[i] != 0 && Y[i] != RHS[i]) {
				ret = false;
				break;
			}
		}
		else {
			cout << "diophant::test_solution unknown type" << endl;
			exit(1);
		}
	}
	
	
	
	if (f_v) {
		cout << "diophant::test_solution done" << endl;
	}
	return ret;
}


void diophant::get_columns(
		int *col, int nb_col,
		other::data_structures::set_of_sets *&S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h, d;

	if (f_v) {
		cout << "diophant::get_columns" << endl;
	}
	S = NEW_OBJECT(other::data_structures::set_of_sets);

	S->init_simple(m, nb_col, 0 /* verbose_level */);
	for (h = 0; h < nb_col; h++) {
		j = col[h];
		d = 0;
		for (i = 0; i < m; i++) {
			if (Aij(i, j)) {
				d++;
			}
		}
		S->Sets[h] = NEW_lint(d);
		S->Set_size[h] = d;
		d = 0;
		for (i = 0; i < m; i++) {
			if (Aij(i, j)) {
				S->Sets[h][d] = i;
				d++;
			}
		}
	}
}

void diophant::test_solution_file(
		std::string &solution_file,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Sol;
	int nb_sol, sol_length;
	int i;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "diophant::test_solution_file" << endl;
	}
	Fio.int_matrix_read_text(solution_file, Sol, nb_sol, sol_length);
	
	for (i = 0; i < nb_sol; i++) {
		if (f_vv) {
			cout << "diophant::test_solution_file testing solution "
					<< i << " / " << nb_sol << ":" << endl;
		}
		if (!test_solution(Sol + i * sol_length, sol_length, verbose_level)) {
			cout << "solution " << i << " / " << nb_sol << " is bad" << endl;
		}
		else {
			cout << "solution " << i << " / " << nb_sol << " is OK" << endl;
		}
		cout << "Y=";
		Int_vec_print(cout, Y, m);
		cout << endl;

		other::data_structures::tally C;

		C.init(Y, m, false, 0);
		cout << "classification: ";
		C.print_bare(false);
		cout << endl;
	}
	if (f_v) {
		cout << "diophant::test_solution_file done" << endl;
	}
}

void diophant::analyze(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, h, val, d;
	
	if (f_v) {
		cout << "diophant::analyze" << endl;
	}

	for (i = 0; i < m; i++) {

	
		int nb_values;
		int *values, *multiplicities;


		coefficient_values_in_row(i, nb_values, 
			values, multiplicities, 0 /*verbose_level*/);


		cout << "row " << i << ": ";
		for (h = 0; h < nb_values; h++) {
			val = values[h];
			d = multiplicities[h];
			cout << val << "^" << d;
			if (h < nb_values - 1) {
				cout << ", ";
			}
		}
		cout << endl;

		FREE_int(values);
		FREE_int(multiplicities);
		
	}

	if (f_v) {
		cout << "diophant::analyze done" << endl;
	}
}

int diophant::is_of_Steiner_type()
{
	int i;

	for (i = 0; i < m; i++) {
		if (type[i] != t_EQ || type[i] != t_EQ) {
			return false;
		}
		if (RHSi(i) != 1) {
			return false;
		}
	}
	return true;
}

void diophant::make_clique_graph_adjacency_matrix(
		other::data_structures::bitvector *&Adj,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j1, j2, L, k, i;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "diophant::make_clique_graph_adjacency_matrix" << endl;
	}
#if 0
	if (!is_of_Steiner_type()) {
		cout << "diophant::make_clique_graph_adjacency_matrix "
				"the system is not of Steiner type" << endl;
		exit(1);
		}
#endif
	L = (n * (n - 1)) >> 1;
	if (f_v) {
		cout << "diophant::make_clique_graph_adjacency_matrix "
				"L=" << L << endl;
	}


	Adj = NEW_OBJECT(other::data_structures::bitvector);

#if 0
	//length = (L + 7) >> 3;
	Adj = bitvector_allocate(L);
	for (i = 0; i < L; i++) {
		bitvector_m_ii(Adj, i, 1);
	}
#else
	if (f_v) {
		cout << "diophant::make_clique_graph_adjacency_matrix "
				"before Adj->allocate" << endl;
	}
	Adj->allocate(L);
	if (f_v) {
		cout << "diophant::make_clique_graph_adjacency_matrix "
				"after Adj->allocate" << endl;
	}
	if (f_v) {
		cout << "diophant::make_clique_graph_adjacency_matrix "
				"setting array to one" << endl;
	}
	for (i = 0; i < L; i++) {
		Adj->m_i(i, 1);
	}
#endif
	for (i = 0; i < m; i++) {
		if (false) {
			cout << "diophant::make_clique_graph_adjacency_matrix "
					"i=" << i << endl;
		}
		for (j1 = 0; j1 < n; j1++) {
			if (Aij(i, j1) == 0) {
				continue;
			}
			for (j2 = j1 + 1; j2 < n; j2++) {
				if (Aij(i, j2) == 0) {
					continue;
				}
				// now: j1 and j2 do not go together
				k = Combi.ij2k(j1, j2, n);
				Adj->m_i(k, 0);
			}
		}
	}
	if (f_v) {
		cout << "diophant::make_clique_graph_adjacency_matrix done" << endl;
	}
}


void diophant::make_clique_graph(
		combinatorics::graph_theory::colored_graph *&CG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::bitvector *Adj;

	if (f_v) {
		cout << "diophant::make_clique_graph" << endl;
	}
	make_clique_graph_adjacency_matrix(Adj, verbose_level - 1);


	CG = NEW_OBJECT(combinatorics::graph_theory::colored_graph);

	string label, label_tex;

	label.assign("clique_graph");
	label_tex.assign("clique\\_graph");

	CG->init_no_colors(n, Adj, true, label, label_tex, verbose_level - 1);
	
	
	if (f_v) {
		cout << "diophant::make_clique_graph" << endl;
	}
}

void diophant::make_clique_graph_and_save(
		std::string &clique_graph_fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::make_clique_graph_and_save" << endl;
	}

	combinatorics::graph_theory::colored_graph *CG;

	make_clique_graph(CG, verbose_level - 1);
	CG->save(clique_graph_fname, verbose_level - 1);

	FREE_OBJECT(CG);
	if (f_v) {
		cout << "diophant::make_clique_graph_and_save done" << endl;
	}
}


void diophant::test_if_the_last_solution_is_unique()
{
	vector<int> last;
	vector<int> cur;
	int l, i, j;

	l = _results.size();
	last = _results.at(l - 1);
	for (i = 0; i < l - 1; i++) {
		cur = _results.at(i);
		for (j = 0; j < n; j++) {
			if (cur[j] != last[j]) {
				break;
			}
		}
		if (j == n) {
			cout << "The last solution " << l - 1
					<< " is the same as solution " << i << endl;
			exit(1);
		}
	}
}




int diophant::solve_first_mckay_once_option(
		int f_once, int verbose_level)
{
	int f_v = true;//(verbose_level >= 1);
	int j;
	int maxresults = 10000000;
	vector<int> res;
	long int nb_backtrack_nodes;
	int nb_sol;

	//verbose_level = 4;
	if (f_v) {
		cout << "diophant::solve_first_mckay calling "
				"solve_mckay" << endl;
	}
	if (f_once) {
		maxresults = 1;
	}


	string label;

	solve_mckay(label,
			maxresults, nb_backtrack_nodes, nb_sol,
			verbose_level - 2);

	if (f_v) {
		cout << "diophant::solve_first_mckay found " << _resultanz
			<< " solutions, using " << nb_backtrack_nodes
			<< " backtrack nodes" << endl;
	}
	_cur_result = 0;
	if (_resultanz == 0) {
		return false;
	}
	res = _results[_cur_result]; //.front();
	for (j = 0; j < n; j++) {
		x[j] = res[j];
	}
	//_results.pop_front();
	_cur_result++;
	if (f_v) {
		cout << "diophant::solve_first_mckay done" << endl;
	}
	return true;
}


// #############################################################################
// callbacks and globals
// #############################################################################


static void diophant_callback_solution_found(
		int *sol, int len,
	int nb_sol, void *data)
{
	int f_v = false;
	diophant *D = (diophant *) data;
	vector<int> lo;
	int i, j;

	if ((nb_sol % 1000) == 0) {
		f_v = true;
	}
	if (f_v) {
		cout << "diophant_callback_solution_found "
				"recording solution "
				<< nb_sol << " len = " << len << " : ";
		Int_vec_print(cout, sol, len);
		cout << endl;
		cout << "D->_resultanz=" << D->_resultanz << endl;

		for (i = 0; i < len; i++) {
			cout << 0 /*DLX_Cur_col[i]*/ << "/" << sol[i];
			if (i < len - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}

	if (!D->test_solution(sol, len, 0 /* verbose_level */)) {
		cout << "diophant_callback_solution_found "
				"the solution fails the test" << endl;
		exit(1);
		//return;
	}
	else {
		if (f_v) {
			cout << "diophant_callback_solution_found "
					"the solution passes the test" << endl;
		}
	}


	lo.resize(D->n);
	for (j = 0; j < D->n; j++) {
		lo[j] = 0;
	}
	for (j = 0; j < len; j++) {
		lo[sol[j]] = 1;
	}
	D->_results.push_back(lo);
	D->_resultanz++;

	if (diophant_user_callback_solution_found) {
		(*diophant_user_callback_solution_found)(sol, len, nb_sol, data);
	}
	//D->test_if_the_last_solution_is_unique();
}


}}}}



