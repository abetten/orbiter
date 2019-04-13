// diophant.C
//
// Anton Betten
// September 18, 2000
//
// moved to GALOIS: April 16, 2015

#include "foundations.h"


using namespace std;

namespace orbiter {
namespace foundations {



void (*diophant_user_callback_solution_found)(int *sol, 
	int len, int nb_sol, void *data) = NULL;


diophant::diophant()
{
	null();
}


diophant::~diophant()
{
	freeself();
}

void diophant::null()
{
	A = NULL;
	G = NULL;
	x = NULL;
	x_max = NULL;
	RHS = NULL;
	RHS1 = NULL;
	type = NULL;
	eqn_label = NULL;
	m = 0;
	n = 0;
	f_max_time = FALSE;
	X = FALSE;
	Y = FALSE;
	f_has_sum = FALSE;
	f_x_max = FALSE;
	f_has_var_labels = FALSE;
	var_labels = NULL;
}

void diophant::freeself()
{
	int i;

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
	if (RHS) {
		FREE_int(RHS);
		}
	if (RHS1) {
		FREE_int(RHS1);
		}
	if (type) {
		FREE_OBJECT(type);
		}
	if (eqn_label) {
		for (i = 0; i < m; i++) {
			if (eqn_label[i]) {
				FREE_char(eqn_label[i]);
				}
			}
		FREE_pchar(eqn_label);
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
	null();
}

void diophant::open(int m, int n)
{
	int i;
	
	A = NEW_int(m * n);
	G = NEW_int(m * n);
	x = NEW_int(n);
	x_max = NEW_int(n);
	RHS = NEW_int(m);
	RHS1 = NEW_int(m);
	type = NEW_OBJECTS(diophant_equation_type, m);
	eqn_label = NEW_pchar(m);
	X = NEW_int(n);
	Y = NEW_int(m);
	
	for (i = 0; i < n; i++) {
		x_max[i] = 0;
		}
	label[0] = 0;
	diophant::m = m;
	diophant::n = n;
	for (i = 0; i < m; i++) {
		type[i] = t_EQ;
		eqn_label[i] = NULL;
		}
	f_has_sum = FALSE;
	f_x_max = FALSE;
	f_max_time = FALSE;
	f_has_var_labels = FALSE;
}

void diophant::init_var_labels(int *labels, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::init_var_labels" << endl;
	}
	var_labels = NEW_int(n);
	f_has_var_labels = TRUE;
	int_vec_copy(labels, var_labels, n);
	if (f_v) {
		cout << "diophant::init_var_labels done" << endl;
	}

}
void diophant::join_problems(
		diophant *D1, diophant *D2, int verbose_level)
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
	if (D1->f_x_max != D2->f_x_max) {
		cout << "D1->f_x_max != D2->f_x_max" << endl;
		exit(1);
		}
	nb_cols = D1->n;
	nb_r1 = D1->m;
	nb_r2 = D2->m;
	nb_rows = nb_r1 + nb_r2;
	open(nb_rows, nb_cols);
	f_has_sum = TRUE;
	sum = D1->sum;
	f_x_max = D1->f_x_max;
	if (f_x_max) {
		for (i = 0; i < nb_cols; i++) {
			if (D1->x_max[i] != D2->x_max[i]) {
				cout << "D1->x_max[i] != D2->x_max[i]" << endl;
				exit(1);
				}
			x_max[i] = D1->x_max[i];
			}
		}
	for (i = 0; i < nb_r1; i++) {
		for (j = 0; j < nb_cols; j++) {
			Aij(i, j) = D1->Aij(i, j);
			}
		type[i] = D1->type[i];
		RHSi(i) = D1->RHSi(i);
		}
	for (i = 0; i < nb_r2; i++) {
		for (j = 0; j < nb_cols; j++) {
			Aij(nb_r1 + i, j) = D2->Aij(i, j);
			}
		type[nb_r1 + i] = D2->type[i];
		RHSi(nb_r1 + i) = D2->RHSi(i);
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
	open(1, nb_weights);
	for (j = 0; j < nb_weights; j++) {
		x_max[j] = target_value / weights[j];
		}
	f_x_max = TRUE;
	f_has_sum = FALSE;
	//sum = nb_to_select;
	for (j = 0; j < nb_weights; j++) {
		Aij(0, j) = weights[j];
	}
	RHSi(0) = target_value;
	if (f_v) {
		cout << "diophant::init_partition_problem" << endl;
		}
}


void diophant::init_problem_of_Steiner_type_with_RHS(
	int nb_rows, int nb_cols, int *Inc, int nb_to_select,
	int *Rhs, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "diophant::init_problem_of_Steiner_"
				"type_with_RHS" << endl;
		}
	open(nb_rows, nb_cols);
	for (i = 0; i < nb_cols; i++) {
		x_max[i] = 1;
		}
	f_x_max = TRUE;
	f_has_sum = TRUE;
	sum = nb_to_select;
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Aij(i, j) = Inc[i * nb_cols + j];
			}
		RHSi(i) = Rhs[i];
		}
	if (f_v) {
		cout << "diophant::init_problem_of_Steiner_"
				"type_with_RHS done" << endl;
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
	open(nb_rows, nb_cols);
	for (i = 0; i < nb_cols; i++) {
		x_max[i] = 1;
		}
	f_x_max = TRUE;
	f_has_sum = TRUE;
	sum = nb_to_select;
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Aij(i, j) = Inc[i * nb_cols + j];
			}
		RHSi(i) = 1;
		}
	if (f_v) {
		cout << "diophant::init_problem_of_Steiner_type done" << endl;
		}
}

void diophant::init_RHS(int RHS_value, int verbose_level)
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

void diophant::init_clique_finding_problem(int *Adj, int nb_pts, 
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
	open(nb_zeros, nb_pts);
	for (i = 0; i < nb_pts; i++) {
		x_max[i] = 1;
		}
	f_x_max = TRUE;
	f_has_sum = TRUE;
	sum = nb_to_select;
	i = 0;
	for (i1 = 0; i1 < nb_pts; i1++) {
		for (i2 = i1 + 1; i2 < nb_pts; i2++) {
			if (Adj[i1 * nb_pts + i2] == 0) {
				Aij(i, i1) = 1;
				Aij(i, i2) = 1;
				type[i] = t_LE;
				RHSi(i) = 1;
				i++;
				}
			}
		}
	if (f_v) {
		cout << "diophant::init_clique_finding_problem done" << endl;
		}
}


void diophant::fill_coefficient_matrix_with(int a)
{
	int i;
	
	for (i = 0; i < m * n; i++) {
		A[i] = a;
		}
}

int &diophant::Aij(int i, int j)
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

int &diophant::Gij(int i, int j)
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

int &diophant::RHSi(int i)
{
	if (i >= m) {
		cout << "diophant::RHSi i >= m" << endl;
		exit(1);
		}
	return RHS[i];
}

void diophant::init_eqn_label(int i, char *label)
{
	int l;
	
	if (i >= m) {
		cout << "diophant::init_eqn_label i >= m" << endl;
		cout << "label: " << label << endl;
		cout << "i=" << i << endl;
		exit(1);
		}
	if (eqn_label[i]) {
		FREE_char(eqn_label[i]);
		eqn_label[i] = NULL;
		}
	l = strlen(label) + 1;
	eqn_label[i] = NEW_char(l);
	strcpy(eqn_label[i], label);
}

void diophant::print()
{
	print2(FALSE);
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
			cout << " = ";
			}
		else if (type[i] == t_LE) {
			cout << " <= ";
			}
		else if (type[i] == t_ZOR) {
			cout << " ZOR ";
			}
		cout << RHS[i] << " (rowsum=" << s << ")" << endl;
		}
	cout << "sum = " << sum << endl;
}

void diophant::print2(int f_with_gcd)
{
	int i, j;
	
	cout << "diophant with m=" << m << " n=" << n << endl;
	for (i = 0; i < m; i++) {
		print_eqn(i, f_with_gcd);
		}
	if (f_x_max) {
		for (j = 0; j < n; j++) {
			cout << "x_{" << j << "} \\le " << x_max[j] << endl;
			}
		}
	if (f_has_sum) {
		cout << "sum = " << sum << endl;
	} else {
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
	if (f_x_max) {
		for (j = 0; j < n; j++) {
			cout << x_max[j];
			}
		cout << endl;
		}
	if (f_has_sum) {
		cout << "sum = " << sum << endl;
	} else {
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
	if (f_x_max) {
		for (j = 0; j < n; j++) {
			cout << "x_{" << j << "} \\le " << x_max[j] << endl;
			}
		}
	if (f_has_sum) {
		cout << "sum = " << sum << endl;
	} else {
		cout << "there is no condition on the sum of x_i" << endl;
	}
}


void diophant::print_eqn(int i, int f_with_gcd)
{
	int j;
	
	for (j = 0; j < n; j++) {
		cout << setw(3) << Aij(i, j) << " ";
		if (f_with_gcd) {
			cout << "|" << setw(3) << Gij(i, j) << " ";
			}
		}
	if (type[i] == t_EQ) {
		cout << " = ";
		}
	else if (type[i] == t_LE) {
		cout << " <= ";
		}
	else if (type[i] == t_ZOR) {
		cout << " ZOR ";
		}
	cout << setw(3) << RHSi(i) << " ";
	if (eqn_label[i]) {
		cout << eqn_label[i];
		}
	cout << endl;
}

void diophant::print_eqn_compressed(int i)
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
	else if (type[i] == t_ZOR) {
		cout << " ZOR ";
		}
	cout << setw(3) << RHSi(i) << " ";
	if (eqn_label[i]) {
		cout << eqn_label[i];
		}
	cout << endl;
}

void diophant::print_eqn_dense(int i)
{
	int j;
	
	for (j = 0; j < n; j++) {
		cout << Aij(i, j);
		}
	if (type[i] == t_EQ) {
		cout << " = ";
		}
	else if (type[i] == t_LE) {
		cout << " <= ";
		}
	else if (type[i] == t_ZOR) {
		cout << " ZOR ";
		}
	cout << setw(3) << RHSi(i) << " ";
	if (eqn_label[i]) {
		cout << eqn_label[i];
		}
	cout << endl;
}

void diophant::print_x_long()
{
	int j;
	
	for (j = 0; j < n; j++) {
		cout << "x_{" << j << "} = " << x[j] << endl;
		}
}

void diophant::print_x(int header)
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
		if (RHS1[k] < 0)
			return FALSE;
		}
	return TRUE;
}

int diophant::solve_first(int verbose_level)
{
	if (FALSE/*n >= 50*/) {
		return solve_first_wassermann(verbose_level);
		}
	else if (TRUE) {
		return solve_first_betten(verbose_level);
		}
	else {
		//cout << "diophant::solve_first
		//solve_first_mckay is disabled" << endl;
		return solve_first_mckay(FALSE, verbose_level);
		}
}

int diophant::solve_next()
{
	return solve_next_betten(0);
	//return solve_next_mckay();
}

int diophant::solve_first_wassermann(int verbose_level)
{
	solve_wassermann(verbose_level);
	exit(1);
}

int diophant::solve_first_mckay(int f_once, int verbose_level)
{
	int f_v = TRUE;//(verbose_level >= 1);
	int j;
	int maxresults = 10000000;
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
	solve_mckay("",
			maxresults, nb_backtrack_nodes, nb_sol, verbose_level - 2);
	if (f_v) {
		cout << "diophant::solve_first_mckay found " << _resultanz
			<< " solutions, using " << nb_backtrack_nodes
			<< " backtrack nodes" << endl;
		}
	_cur_result = 0;
	if (_resultanz == 0)
		return FALSE;
	res = _results.front();
	for (j = 0; j < n; j++) {
		x[j] = res[j];
		}
	_results.pop_front();
	_cur_result++;
	if (f_v) {
		cout << "diophant::solve_first_mckay done" << endl;
		}
	return TRUE;
}

void diophant::draw_solutions(const char *fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	vector<int> res;
	int *solution;
	int solution_sz;

	if (f_v) {
		cout << "diophant::draw_solutions" << endl;
		}
	solution = NEW_int(n);

	for (i = 0; i < _resultanz; i++) {
		res = _results.front();
		solution_sz = 0;
		for (j = 0; j < n; j++) {
			if (res[j]) {
				solution[solution_sz++] = j;
				}
			}
		
		char fname_base2[1000];
		
		sprintf(fname_base2, "%s_sol_%d", fname_base, i);
		
		int xmax_in = ONE_MILLION;
		int ymax_in = ONE_MILLION;
		int xmax_out = ONE_MILLION;
		int ymax_out = ONE_MILLION;

		draw_partitioned(fname_base2, xmax_in, ymax_in, xmax_out, ymax_out, 
			TRUE /*f_solution*/, solution, solution_sz, 
			verbose_level);
		_results.pop_front();
		}

	FREE_int(solution);
	if (f_v) {
		cout << "diophant::draw_solutions done" << endl;
		}
}

void diophant::write_solutions(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;
	vector<int> res;

	if (f_v) {
		cout << "diophant::write_solutions" << endl;
		}

	{
	ofstream fp(fname);

		fp << _resultanz << " " << n << endl;
		for (i = 0; i < _resultanz; i++) {
			res = _results.front();
			h = 0;
			for (j = 0; j < n; j++) {
				fp << res[j] << " ";
				h += res[j];
#if 0
				if (res[j]) {
					fp << j << " ";
					h++;
					}
#endif
				}
			if (h != sum) {
				cout << "diophant::write_solutions h != sum" << endl;
				cout << "nb_sol = " << _resultanz << endl;
				cout << "sum = " << sum << endl;
				cout << "h = " << h << endl;
				cout << "res=" << endl;
				for (j = 0; j < n; j++) {
					cout << j << " : " << res[j] << endl;
					}
				exit(1);
				}
			fp << endl;
			_results.pop_front();
			}
			fp << "-1" << endl;
	}
	if (f_v) {
		cout << "diophant::write_solutions written file " << fname
				<< " of size " << file_size(fname) << endl;
		}
}

void diophant::read_solutions_from_file(const char *fname_sol,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	vector<int> res;

	if (f_v) {
		cout << "diophant::read_solutions_from_file" << endl;
		}
	if (f_v) {
		cout << "diophant::read_solutions_from_file reading file "
				<< fname_sol << " of size " << file_size(fname_sol) << endl;
		}
	if (file_size(fname_sol) <= 0) {
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
			<< file_size(fname_sol) << endl;
		}
}


void diophant::get_solutions(int *&Sol, int &nb_sol, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;
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
		res = _results.front();
		h = 0;
		for (j = 0; j < n; j++) {
			//x[j] = res[j];
			if (res[j]) {
				Sol[i * sum + h] = j;
				h++;
				}
			}
		if (h != sum) {
			cout << "diophant::get_solutions h != sum" << endl;
			exit(1);
			}
		_results.pop_front();
		}
	if (f_v) {
		cout << "diophant::get_solutions done" << endl;
		}
}

void diophant::get_solutions_full_length(int *&Sol,
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
	if (!f_has_sum) {
		cout << "diophant::get_solutions_full_length !f_has_sum" << endl;
		exit(1);
	}
	nb_sol = _resultanz;
	Sol = NEW_int(nb_sol * n);
	for (i = 0; i < _resultanz; i++) {
		res = _results.front();
		for (j = 0; j < n; j++) {
			Sol[i * n + j] = res[j];
			}
		_results.pop_front();
		}
	if (f_v) {
		cout << "diophant::get_solutions_full_length done" << endl;
		}
}

void diophant::test_solution_full_length(int *sol, int verbose_level)
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
		if (type[i] == t_LE) {
			if (s >= RHSi(i)) {
				cout << "diophant::test_solution_full_length condition "
						<< i << " / " << m << ": s=" << s
						<< " is larger than " << RHSi(i) << endl;
				exit(1);
				}
			}
		else if (type[i] == t_EQ) {
			if (s != RHSi(i)) {
				cout << "diophant::test_solution_full_length condition "
						<< i << " / " << m << ": s=" << s
						<< " is not equal to " << RHSi(i) << endl;
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

int diophant::solve_all_DLX(int f_write_tree,
		const char *fname_tree, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::solve_all_DLX verbose_level="
				<< verbose_level << endl;
		}
	install_callback_solution_found(
		diophant_callback_solution_found,
		this);
	int *Inc;
	int i, j;
	int nb_sol, nb_backtrack;

	Inc = NEW_int(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			Inc[i * n + j] = Aij(i, j);
			}
		}

	_resultanz = 0;
	
	DlxTransposeAppendAndSolve(Inc, m, n, nb_sol, nb_backtrack, 
		FALSE, "", 
		f_write_tree, fname_tree, 
		verbose_level - 1);
		// GALOIS/dlx.C
	
	nb_steps_betten = nb_backtrack;
	FREE_int(Inc);
	if (f_v) {
		cout << "diophant::solve_all_DLX done found " << _resultanz
			<< " solutions with " << nb_backtrack
			<< " backtrack steps" << endl;
		}
	return _resultanz;
}

int diophant::solve_all_DLX_with_RHS(int f_write_tree,
		const char *fname_tree, int verbose_level)
{
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
		int_matrix_print(Inc, m, n);
		}
	f_has_type = TRUE;
	my_RHS = NEW_int(m);
	my_type = NEW_OBJECTS(diophant_equation_type, m);
	for (i = 0; i < m; i++) {
		my_RHS[i] = RHS[i];
		my_type[i] = type[i];
		}
	if (f_vv) {
		cout << "diophant::solve_all_DLX_with_RHS  RHS:" << endl;
		int_matrix_print(my_RHS, m, 1);
		//cout << diophant::solve_all_DLX_with_RHS  type:" << endl;
		//int_matrix_print(my_type, m, 1);
		}

	_resultanz = 0;
	
	DlxTransposeAndSolveRHS(Inc, m, n, 
		my_RHS, f_has_type, my_type, 
		nb_sol, nb_backtrack, 
		FALSE, "", 
		f_write_tree, fname_tree, 
		verbose_level - 1);
		// GALOIS/dlx.C
	
	nb_steps_betten = nb_backtrack;
	FREE_int(Inc);
	FREE_int(my_RHS);
	FREE_OBJECTS(my_type);
	if (f_v) {
		cout << "diophant::solve_all_DLX_with_RHS done found "
			<< _resultanz << " solutions with " << nb_backtrack
			<< " backtrack steps" << endl;
		}
	return _resultanz;
}

int diophant::solve_all_DLX_with_RHS_and_callback(
	int f_write_tree, const char *fname_tree,
	void (*user_callback_solution_found)(int *sol,
			int len, int nb_sol, void *data),
	int verbose_level)
{
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
		int_matrix_print(Inc, m, n);
		}
	f_has_type = TRUE;
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
		int_matrix_print(my_RHS, m, 1);
		//cout << diophant::solve_all_DLX_with_RHS  type:" << endl;
		//int_matrix_print(my_type, m, 1);
		}

	_resultanz = 0;
	
	DlxTransposeAndSolveRHS(Inc, m, n, 
		my_RHS, f_has_type, my_type, 
		nb_sol, nb_backtrack, 
		FALSE, "", 
		f_write_tree, fname_tree, 
		verbose_level - 1);
		// GALOIS/dlx.C
	
	nb_steps_betten = nb_backtrack;
	FREE_int(Inc);
	FREE_int(my_RHS);
	FREE_OBJECTS(my_type);
	if (f_v) {
		cout << "diophant::solve_all_DLX_with_RHS done found "
				<< _resultanz << " solutions with " << nb_backtrack
				<< " backtrack steps" << endl;
		}
	return _resultanz;
}

int diophant::solve_all_mckay(long int &nb_backtrack_nodes, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int maxresults = 10000;
	//long int nb_backtrack_nodes;
	int nb_sol;
	
	if (f_v) {
		cout << "diophant::solve_all_mckay before solve_mckay, "
				"verbose_level=" << verbose_level << endl;
		}
	solve_mckay(label, maxresults,
			nb_backtrack_nodes, nb_sol, verbose_level - 2);
	if (f_v) {
		cout << "diophant::solve_all_mckay found " << _resultanz
				<< " solutions in " << nb_backtrack_nodes
				<< " backtrack nodes" << endl;
		}
	return _resultanz;
}

int diophant::solve_once_mckay(int verbose_level)
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


int diophant::solve_all_betten(int verbose_level)
{
	int f_v = (verbose_level >= 1);
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

int diophant::solve_all_betten_with_conditions(int verbose_level, 
	int f_max_sol, int max_sol, 
	int f_max_time, int max_time_in_seconds)
{
	int f_v = (verbose_level >= 1);
	int j;
	vector<int> lo;
	//int maxresults = 10000000;
	_resultanz = 0;
	_cur_result = 0;
	
	if (f_max_time) {
		diophant::f_max_time = TRUE;
		diophant::max_time_in_sec = max_time_in_seconds;
		f_broken_off_because_of_maxtime = FALSE;
		t0 = os_ticks();
		max_time_in_ticks = max_time_in_seconds * os_ticks_per_second();
		if (TRUE || f_v) {
			cout << "solve_all_betten_with_conditions maxtime "
					"max_time_in_sec=" << max_time_in_sec << endl;
			}
		}
	t0 = os_ticks();
	if (solve_first_betten(verbose_level - 2)) {
		lo.resize(n);
		for (j = 0; j < n; j++) {
			lo[j] = (int) x[j];
			}
		_results.push_back(lo);
		_resultanz++;
		if (f_max_sol && _resultanz == max_sol) {
			return TRUE;
			}
		while (solve_next_betten(verbose_level - 2)) {
			lo.resize(n);
			for (j = 0; j < n; j++) {
				lo[j] = (int) x[j];
				}
			_results.push_back(lo);
			_resultanz++;
			if (f_max_sol && _resultanz == max_sol) {
				return TRUE;
				}
			}
		}
	if (f_broken_off_because_of_maxtime) {
		return TRUE;
		}
	//solve_mckay(maxresults, verbose_level - 2);
	if (f_v) {
		cout << "diophant::solve_all_betten found " << _resultanz 
			<< " solutions in " << nb_steps_betten << " steps" << endl;
		}
	return FALSE;
}

int diophant::solve_first_betten(int verbose_level)
{
	int i, j, g;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int k, total_max;

	if (!f_has_sum) {
		cout << "diophant::solve_first_betten !f_has_sum" << endl;
		exit(1);
	}
	nb_steps_betten = 0;
	if (m <= 0) {
		if (f_v) {
			cout << "diophant::solve_first_betten(): m <= 0" << endl;
			}
		return TRUE;
		}
	if (n == 0) {
		//cout << "diophant::solve_first_betten(): n == 0" << endl;
		for (i = 0; i < m; i++) {
			if (type[i] == t_EQ) {
				if (RHS[i]) {
					if (f_v) {
						cout << "diophant::solve_first_betten no solution "
							"in equation " << i << " because n=0 and "
							"RHS=" << RHS[i] << " and not an inequality"
							<< endl;
						}
					return FALSE;
					}
				}
			}
		return TRUE;
		}
	for (i = 0; i < m; i++) {
		RHS1[i] = RHS[i];
		}
	sum1 = sum;
	if (f_x_max) {
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
			return FALSE;
			}
		}
	
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
					g = gcd_int(Aij(i, j + 1), g);
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
		print2(TRUE);
		}

	j = 0;
	while (TRUE) {
		while (TRUE) {
			if (j >= n) {
				if (f_v) {
					cout << "diophant::solve_first_betten solution" << endl;
					print_x(nb_steps_betten);
					}
				return TRUE;
				}
			nb_steps_betten++;
			if ((nb_steps_betten % 1000000) == 0) {
				cout << "diophant::solve_first_betten nb_steps_betten="
					<< nb_steps_betten << " sol=" << _resultanz << " ";
				if (f_max_time) {
					int t1 = os_ticks();
					int dt = t1 - t0;
					int t = dt / os_ticks_per_second();
					cout << "time in seconds: " << t;
					}
				cout << endl;
				print_x(nb_steps_betten);
				if (f_max_time) {
					int t1 = os_ticks();
					int dt = t1 - t0;
					if (dt > max_time_in_ticks) {
						f_broken_off_because_of_maxtime = TRUE;
						return FALSE;
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
		while (TRUE) {
			if (j == 0) {
				if (f_v) {
					cout << "diophant::solve_first_betten "
							"no solution" << endl;
					}
				return FALSE;
				}
			j--;
			nb_steps_betten++;
			if ((nb_steps_betten % 1000000) == 0) {
				cout << "diophant::solve_first_betten nb_steps_betten="
						<< nb_steps_betten << " sol=" << _resultanz << " ";
				if (f_max_time) {
					int t1 = os_ticks();
					int dt = t1 - t0;
					int t = dt / os_ticks_per_second();
					cout << "time in seconds: " << t;
					}
				cout << endl;
				print_x(nb_steps_betten);
				if (f_max_time) {
					int t1 = os_ticks();
					int dt = t1 - t0;
					if (dt > max_time_in_ticks) {
						f_broken_off_because_of_maxtime = TRUE;
						return FALSE;
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
			if (j_nxt(j, verbose_level - 2))
				break;
			}
		j++;
		}
}

int diophant::solve_next_mckay(int verbose_level)
{
	int j;
	if (_cur_result < _resultanz) {
		for (j = 0; j < n; j++) {
			x[j] = _results.front()[j];
			}
		_results.pop_front();
		_cur_result++;
		return TRUE;
		}
	else {
		return FALSE;
		}
}

int diophant::solve_next_betten(int verbose_level)
{
	int j;
	
	if (!f_has_sum) {
		cout << "diophant::solve_next_betten !f_has_sum" << endl;
		exit(1);
	}
	if (m == 0) {
		return FALSE;
		}
	if (n == 0) {
		return FALSE;
		}
	j = n - 1;
	while (TRUE) {
		while (TRUE) {
			nb_steps_betten++;
			if ((nb_steps_betten % 1000000) == 0) {
				cout << "diophant::solve_next_betten nb_steps_betten="
						<< nb_steps_betten << " sol=" << _resultanz << " ";
				if (f_max_time) {
					int t1 = os_ticks();
					int dt = t1 - t0;
					int t = dt / os_ticks_per_second();
					cout << "time in seconds: " << t;
					}
				cout << endl;
				print_x(nb_steps_betten);
				if (f_max_time) {
					int t1 = os_ticks();
					int dt = t1 - t0;
					if (dt > max_time_in_ticks) {
						f_broken_off_because_of_maxtime = TRUE;
						return FALSE;
						}
					}
				}
			if (j_nxt(j, verbose_level))
				break;
			if (j == 0)
				return FALSE;
			j--;
			}
		while (TRUE) {
			if (j >= n - 1) {
				return TRUE;
				}
			j++;
			nb_steps_betten++;
			if ((nb_steps_betten % 1000000) == 0) {
				cout << "diophant::solve_next_betten nb_steps_betten="
						<< nb_steps_betten << " sol=" << _resultanz << " ";
				if (f_max_time) {
					int t1 = os_ticks();
					int dt = t1 - t0;
					int t = dt / os_ticks_per_second();
					cout << "time in seconds: " << t;
					}
				cout << endl;
				if (f_max_time) {
					int t1 = os_ticks();
					int dt = t1 - t0;
					if (dt > max_time_in_ticks) {
						f_broken_off_because_of_maxtime = TRUE;
						return FALSE;
						}
					}
				}
			if (!j_fst(j, verbose_level))
				break;
			}
		j--;
		}
}

int diophant::j_fst(int j, int verbose_level)
// if return value is FALSE, 
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
	if (f_x_max) {
		// with restriction: 
		x[j] = MINIMUM(x[j], x_max[j]);
		}
	if (f_vv) {
		cout << "diophant::j_fst j=" << j << " trying x[j]=" << x[j] << endl;
		}
	for (i = 0; i < m; i++) {
		if (x[j] == 0)
			break;
		a = Aij(i, j);
		if (a == 0)
			continue;
		// x[j] = MINIMUM(x[j], (RHS1[i] / a));
		b = RHS1[i] / a;
		if (b < x[j]) {
			if (f_vv) {
				char *label;
				
				if (eqn_label[i]) {
					label = eqn_label[i];
					}
				else {
					label = NEW_char(1);
					label[0] = 0;
					}
				cout << "diophant::j_fst j=" << j << " reducing x[j] "
					"from " << x[j] << " to " << b
					<< " because of equation " << i << " = "
					<< label << endl;
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
			if (type[i] == t_LE)
				continue;
			if (RHS1[i] != 0)
				break; // no solution 
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
			return FALSE;
			}
		return TRUE;
		}
	
	while (TRUE) {
		// check gcd restrictions: 
		for (i = 0; i < m; i++) {
			if (type[i] == t_LE)
				continue;
				// it is an inequality, hence no gcd condition 
			g = G[i * n + j];
			if (g == 0 && RHS1[i] != 0) {
				if (f_vv) {
					char *label;
				
					if (eqn_label[i])
						label = eqn_label[i];
					else {
						label = (char *) "";
						}
					cout << "diophant::j_fst g == 0 && RHS1[i] != 0 in "
							"eqn i=" << i << " = " << label << endl;
					cout << "g=" << g << endl;
					cout << "i=" << i << endl;
					cout << "j=" << j << " != n - 1" << endl;
					cout << "n=" << n << endl;
					cout << "RHS1[i]=" << RHS1[i] << endl;
					print_x(nb_steps_betten);
					}
				break;
				}
			if (g == 0)
				continue;
			if (g == 1) // no restriction 
				continue;
			if ((RHS1[i] % g) != 0) {
				if (f_vv) {
					char *label;
				
					if (eqn_label[i])
						label = eqn_label[i];
					else {
						label = (char *) "";
						}
					cout << "diophant::j_fst (RHS1[i] % g) != 0 in "
							"equation i=" << i << " = " << label << endl;
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
		if (i == m) // OK 
			break;
		
		if (f_vv) {
			cout << "gcd test failed !" << endl;
			}
		// was not OK
		if (x[j] == 0) {
			if (f_vv) {
				char *label;
				
				if (eqn_label[i])
					label = eqn_label[i];
				else {
					label = (char *) "";
					}
				cout << "diophant::j_fst no solution b/c gcd test "
						"failed in equation " << i << " = " << label << endl;
				cout << "j=" << j << endl;
				cout << "x[j]=" << x[j] << endl;
				cout << "RHS1[i]=" << RHS1[i] << endl;
				cout << "Gij(i,j)=" << Gij(i,j) << endl;
				print_x(nb_steps_betten);
				}
			return FALSE;
			}
		x[j]--;
		sum1++;
		for (ii = 0; ii < m; ii++) {
			RHS1[ii] += A[ii * n + j];
			}
		if (f_vv) {
			cout << "diophant::j_fst() decrementing to: x[" << j
					<< "] = " << x[j] << endl;
			}
		}
	return TRUE;
}

int diophant::j_nxt(int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, ii, g;

	if (f_v) {
		cout << "j_nxt node=" << nb_steps_betten << " j=" << j <<  endl;
		}
	if (j == n - 1) {
		for (ii = 0; ii < m; ii++)
			RHS1[ii] += x[j] * A[ii * n + j];
		sum1 += x[j];
		x[j] = 0;
		if (f_vv) {
			cout << "diophant::j_nxt no solution b/c j == n - 1" << endl;
			cout << "j=" << j << endl;
			cout << "n=" << n << endl;
			print_x(nb_steps_betten);
			}
		return FALSE;
		}
	
	while (x[j] > 0) {
		x[j]--;
		if (f_vv) {
			cout << "diophant::j_nxt() decrementing to: x[" << j
					<< "] = " << x[j] << endl;
			}
		sum1++;
		for (ii = 0; ii < m; ii++)
			RHS1[ii] += A[ii * n + j];
		
		// check gcd restrictions: 
		for (i = 0; i < m; i++) {
			if (type[i] == t_LE)
				continue;
				// it is an inequality, hence no gcd condition
			g = G[i * n + j];
			if (g == 0 && RHS1[i] != 0)
				break;
			if (g == 0)
				continue;
			if (g == 1) // no restriction 
				continue;
			if ((RHS1[i] % g) != 0)
				break;
			}
		if (i == m) // OK 
			return TRUE;
		if (f_vv) {
			char *label;
				
			if (eqn_label[i])
				label = eqn_label[i];
			else {
				label = (char *) "";
				}
			cout << "diophant::j_nxt() gcd restriction failed in "
					"eqn " << i << " = " << label << endl;
			}
		}
	if (f_vv) {
		cout << "diophant::j_nxt no solution b/c gcd test failed" << endl;
		cout << "j=" << j << endl;
		print_x(nb_steps_betten);
		}
	return FALSE;
}

void diophant::solve_mckay(const char *label, int maxresults, 
	long int &nb_backtrack_nodes, int &nb_sol, int verbose_level)
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
	const char *label,
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
	lgs.Init(this, label, (int) m + 1, (int) n);
	minres.resize(m + 1);
	maxres.resize(m + 1);
	fanz.resize(m + 1);
	eqn.resize(m + 1);
	
	for (i = 0; i < m; i++) {
		// the RHS:
		if (type[i] == t_LE) {
			minres[i] = (int) minrhs;
			maxres[i] = (int) RHS[i];
			}
		else if (type[i] == t_EQ) {
			minres[i] = (int) RHS[i];
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
			if (A[i * n + j])
				nb++;
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
		}
	
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
	if (f_x_max) {
		if (f_v) {
			cout << "diophant::solve_mckay_override_minrhs_in_inequalities "
					"f_x_max=true" << endl;
			cout << "x_max=";
			int_vec_print(cout, x_max, n);
			cout << endl;
		}
		for (j = 0; j < n; j++) {
			minvarvalue[j] = 0;
			maxvarvalue[j] = (int) x_max[j];
			}
		}
	else {
		if (f_v) {
			cout << "diophant::solve_mckay_override_minrhs_in_inequalities "
					"f_x_max=false sum="  << sum << endl;
		}
		for (j = 0; j < n; j++) {
			minvarvalue[j] = 0;
			maxvarvalue[j] = (int) sum;
			}
		}
	if (f_v) {
		cout << "diophant::solve_mckay_override_minrhs_in_inequalities "
				"maxvarvalue=" << endl;
		for (j = 0; j < n; j++) {
			cout << j << " : " << maxvarvalue[j] << endl;
		}
	}
#if 0
  for (j=1; j<=_eqnanz; j++) {
    minres[j-1] = _eqns[j-1].GetMinResult();
    maxres[j-1] = _eqns[j-1].GetMaxResult();
    fanz[j-1] = _eqns[j-1].GetVarAnz();
    eqn[j-1].resize(_eqns[j-1].GetVarAnz());
    it = _eqns[j-1].GetFaktoren().begin();
    for (i=1; i<=_eqns[j-1].GetVarAnz(); i++) {
      eqn[j-1][i-1].var=it->first-1;
      eqn[j-1][i-1].coeff=it->second;
      it++;
    }
  }
#endif
	_resultanz = 0;
	_maxresults = (int) maxresults;
	
	lgs.possolve(minvarvalue, maxvarvalue, 
		eqn, minres, maxres, fanz, 
		(int) m + 1, (int) n, 
		(int) verbose_level);
	nb_backtrack_nodes = lgs.nb_calls_to_solve;
	nb_sol = _resultanz;
	if (f_v) {
		cout << "diophant::solve_mckay_override_minrhs_in_inequalities "
			<< label << " finished, "
			"number of solutions = " << _resultanz
			<< " nb_backtrack_nodes=" << nb_backtrack_nodes << endl;
		}
}


static int cnt_wassermann = 0;

#define BUFSIZE_WASSERMANN 1000000

void diophant::latex_it()
{
	latex_it(cout);
}

void diophant::latex_it(ostream &ost)
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
			ost << " =  ";
			}
		else if (type[i] == t_LE) {
			ost << "  \\le   ";
			}
		else if (type[i] == t_ZOR) {
			ost << "  ZOR   ";
			}
		ost << setw(2) << RHS[i] << " & ";
		if (eqn_label[i])
			ost << eqn_label[i];
		ost << "\\\\" << endl;
		}
	ost << "\\hline" << endl;
	if (f_x_max) {
		ost << "\\multicolumn{" << n + 2 << "}{|c|}{" << endl;
		ost << "\\mbox{subject to:}" << endl;
		ost << "}\\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\multicolumn{" << n + 2 << "}{|l|}{" << endl;
		for (j = 0; j < n; j++) {
			ost << "x_{" << j + 1 << "} \\le " << x_max[j] << "\\," << endl;
			}
		if (f_has_sum) {
			ost << "\\sum_{i=1}^{" << n << "} x_i=" << sum << endl;
		}
		ost << "}\\\\" << endl;
		ost << "\\hline" << endl;
		}
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
	f_no_solution = FALSE;
	for (i = m - 1; i >= 0; i--) {
		f_trivial = FALSE;
		d = count_non_zero_coefficients_in_row(i);
		if (type[i] == t_LE) {
			if (d <= RHS[i]) {
				f_trivial = TRUE;
				}
			}
		else if (type[i] == t_EQ) {
			if (RHS[i] > d) {
				f_no_solution = TRUE;
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

int diophant::count_non_zero_coefficients_in_row(int i)
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

void diophant::coefficient_values_in_row(int i, int &nb_values, 
	int *&values, int *&multiplicities, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, j, k, idx;

	if (f_v) {
		cout << "diophant::coefficient_values_in_row" << endl;
		}
	nb_values = 0;
	values = NEW_int(n);
	multiplicities = NEW_int(n);
	for (j = 0; j < n; j++) {
		a = Aij(i, j);
		if (a) {
			if (!int_vec_search(values, nb_values, a, idx)) {
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

void diophant::get_coefficient_matrix(int *&M, 
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

void diophant::save_as_Levi_graph(const char *fname, int verbose_level)
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
		FALSE /* f_point_labels */, 
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

void diophant::save_in_compact_format(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, d;
	
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
			fp << "EQ";
			}
		else if (type[i] == t_LE) {
			fp << "LE";
			}
		else if (type[i] == t_ZOR) {
			fp << "ZOR";
			}
		fp << " " << RHS[i];

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
				<< file_size(fname) << endl;
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
		s = atoi(remainder2.c_str());
		//cout << "m=" << m << " n=" << n << " sum=" << s << endl;


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

void diophant::save_in_general_format(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, d, h, val;
	
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
			fp << "EQ";
			}
		else if (type[i] == t_LE) {
			fp << "LE";
			}
		else if (type[i] == t_ZOR) {
			fp << "ZOR";
			}
		fp << " " << RHS[i];

	
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
	fp << "END" << endl;
	}
	if (f_v) {
		cout << "diophant::save_in_general_format done, " << endl;
		cout << "written file " << fname << " of size "
				<< file_size(fname) << endl;
		}
}

void diophant::read_general_format(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m, n, s;
	int cnt, i, j, d, h, a, nb_types, t, val;
	int f_has_sum1;
	int f_has_var_labels_save = FALSE;
	
	if (f_v) {
		cout << "diophant::read_general_format" << endl;
		}
	string line;
	string EQ("EQ");
	string LE("LE");
	string ZOR("ZOR");
	{
	ifstream myfile (fname);
	if (myfile.is_open()) {
		cout << "diophant::read_general_format" << endl;
		getline (myfile, line); // file name


		cout << "diophant::read_general_format parsing '" << line << "'" << endl;
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
		cout << "diophant::read_general_format "
				"m=" << m << " n=" << n << " sum=" << s
				<< " f_has_var_labels=" << f_has_var_labels_save << endl;

		open(m, n);
		f_has_var_labels = f_has_var_labels_save;
		f_has_sum = f_has_sum1;
		sum = s;

		if (f_has_var_labels) {

			cout << "reading var labels" << endl;
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
			cout << "not reading var labels" << endl;
		}

		for (cnt = 0; cnt < m; cnt++) {
			cout << "reading equation " << cnt << " / " << m << ":" << endl;
			getline (myfile, line);
			i = line.find(" ");
			remainder = line.substr(i + 1);
			line = remainder;
			cout << "remainder = '" << remainder << "'" << endl;
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
			else if (str.compare(ZOR) == 0) {
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

void diophant::save_in_wassermann_format(
		const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int cnt_inequalities = 0, cur_inequality, f_need_slack;
	int i, j, a;
	
	for (i = 0; i < m; i++) {
		if (type[i] == t_LE) {
			cnt_inequalities++;
			}
		}
	if (f_v) {
		cout << "save_in_wassermann_format cnt_inequalities = "
				<< cnt_inequalities << endl;
		}
	{
	ofstream f(fname);
	
	f << "% " << fname << endl;
	f << m << " " << n + cnt_inequalities << " " << 1 << endl;
	cur_inequality = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = Aij(i, j);
			f << setw(2) << a << " ";
			}
		if (type[i] == t_LE) {
			f_need_slack = TRUE;
			}
		else {
			f_need_slack = FALSE;
			}
		if (f_need_slack) {
			cout << "equation " << i << " is inequality" << endl;
			cout << "cur_inequality = " << cur_inequality << endl;
			}
		for (j = 0; j < cnt_inequalities; j++) {
			if (f_need_slack && j == cur_inequality) {
				f << setw(2) << 1 << " ";
				}
			else {
				f << setw(2) << 0 << " ";
				}
			}
		if (f_need_slack)
			cur_inequality++;
		f << setw(2) << RHS[i] << endl;
		}
	}
	cout << "written file " << fname << " of size "
			<< file_size(fname) << endl;
}

void diophant::solve_wassermann(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname[1000];
	char *fname_solutions = (char *) "solutions";
	char cmd[1000];
	char *buf;
	//int lambda, c0_factor = 20, beta = 120;
	//int p = 14, xmax, silence_level = 0;
	
	f_v = TRUE;
	if (f_v) {
		cout << "diophant::solve_wassermann " << cnt_wassermann << endl;
		}
	sprintf(fname, "wassermann_input_%d.txt", cnt_wassermann);

	save_in_wassermann_format(fname, verbose_level);

	cnt_wassermann++;

#if 1
	sprintf(cmd, "../ALFRED/LLL_ENUM/BIN/discreta_lll_with "
			"30 10 1 40 %s 0", fname);
	cout << "executing: " << cmd << endl;
	system(cmd);
	cout << "found file " << fname_solutions << " of size "
			<< file_size(fname_solutions) << endl;
	if (file_size(fname_solutions) < 0) {
		cout << "did not find solution file" << endl;
		exit(1);
		}
	_resultanz = 0;
	buf = NEW_char(BUFSIZE_WASSERMANN);
	{
	ifstream f(fname_solutions);
	while (!f.eof()) {
		f.getline(buf, BUFSIZE_WASSERMANN, '\n');
		cout << buf << endl;
		}
	}
	FREE_char(buf);
	if (f_v) {
		cout << "diophant::solve_wassermann "
				<< cnt_wassermann - 1 << " finished" << endl;
		}
	exit(1);
#endif
}

void diophant::eliminate_zero_rows_quick(int verbose_level)
{
	int *eqn_number;
	eliminate_zero_rows(eqn_number, verbose_level);
	FREE_int(eqn_number);
}

void diophant::eliminate_zero_rows(int *&eqn_number, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, mm;
	int f_delete = FALSE;
	
	eqn_number = NEW_int(m);
	mm = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (Aij(i, j))
				break;
			}
		if (j < n) {
			f_delete = FALSE;
			}
		else {
			f_delete = TRUE;
			}
		if (f_delete && type[i] == t_EQ && RHS[i]) {
			f_delete = FALSE;
			}
		if (!f_delete) {
			eqn_number[mm] = i;
			if (i != mm) {
				for (j = 0; j < n; j++) {
					Aij(mm, j) = Aij(i, j);
					}
				RHS[mm] = RHS[i];
				type[mm] = type[i];
				eqn_label[mm] = eqn_label[i];
				}
			mm++;
			}
		else {
			if (eqn_label[i]) {
				FREE_char(eqn_label[i]);
				eqn_label[i] = NULL;
				}
			}
		}
	if (f_v) {
		cout << "eliminate_zero_rows: eliminated " << m - mm
				<< " zero rows" << endl;
		}
	m = mm;
}

int diophant::is_zero_outside(int first, int len, int i)
{
	int j;
	
	for (j = 0; j < n; j++) {
		if (j >= first && j < first + len)
			continue;
		if (Aij(i, j))
			return FALSE;
		}
	return TRUE;
}

void diophant::project(diophant *D, int first, int len,
		int *&eqn_number, int &nb_eqns_replaced, int *&eqns_replaced,
		int verbose_level)
{
	int i, j, f_zo;
	
	D->open(m, len);	
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
		D->RHS[i] = RHS[i];
		D->type[i] = type[i];
		if (!f_zo) {
			D->type[i] = t_LE;
			}
		if (eqn_label[i]) {
			D->init_eqn_label(i, eqn_label[i]);
			}
		}
	D->f_x_max = f_x_max;
	if (f_x_max) {
		for (j = 0; j < len; j++) {
			D->x_max[j] = x_max[first + j];
			}
		}
	D->f_has_sum = f_has_sum;
	D->sum = sum;
	D->eliminate_zero_rows(eqn_number, 0);
}

void diophant::multiply_A_x_to_RHS1()
{
	int i, j, a;
	
	for (i = 0; i < m; i++) {
		a = 0;
		for (j = 0; j < n; j++) {
			a+= Aij(i, j) * x[j];
			}
		RHS1[i] = a;
		}
}

void diophant::write_xml(ostream &ost, const char *label)
{
	int i, j;
	char *lbl;
	
	ost << "<DIOPHANT label=\"" << label << "\" num_eqns=" << m
			<< " num_vars=" << n << " f_has_sum=" << f_has_sum
			<< " sum=" << sum << " f_x_max="
			<< f_x_max << ">" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0;j < n; j++) {
			ost << setw(4) << Aij(i, j) << " ";
			}
		if (eqn_label[i]) {
			lbl = eqn_label[i];
			}
		else {
			lbl = (char *) "";
			}
		if (type[i] == t_EQ) {
			ost << setw(2) << 0;
			}
		else if (type[i] == t_LE) {
			ost << setw(2) << 1;
			}
		else if (type[i] == t_ZOR) {
			ost << setw(2) << 2;
			}
		ost << setw(4) << RHS[i] << " \"" << lbl << "\"" << endl;
		}
	if (f_x_max) {
		ost << endl;
		for (j = 0;j < n; j++) {
			ost << setw(4) << x_max[j] << " ";
			}
		ost << endl;
		}
	ost << "</DIOPHANT>" << endl;
	
}


void diophant::read_xml(ifstream &f, char *label, int verbose_level)
{
#ifdef SYSTEMUNIX
	int f_v = (verbose_level >= 1);
	string str, mapkey, mapval;
	bool brk;
	int eqpos, l, M = 0, N = 0, F_has_sum = 0, Sum = 0, F_x_max = 0, i, j, a;
	char tmp[1000], c;


	if (f_v) {
		cout << "diophant::read_xml" << endl;
	}
	label[0] = 0;
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
				l = (int) mapval.size();
				for (i = 1; i < l; i++) {
					label[i - 1] = mapval[i];
					}
				label[l - 2] = 0;
				}
			else if (mapkey == "num_eqns") {
				M = str2int(mapval);
				}
			else if (mapkey == "num_vars") {
				N = str2int(mapval);
				}
			else if (mapkey == "f_has_sum") {
				F_has_sum = str2int(mapval);
				}
			else if (mapkey == "sum") {
				Sum = str2int(mapval);
				}
			else if (mapkey == "f_x_max") {
				F_x_max = str2int(mapval);
				}
			}
		brk = brk || f.eof();
		}
	cout << "M=" << M << " N=" << N << endl;
	open(M, N);
	f_has_sum = F_has_sum;
	sum = Sum;
	f_x_max = F_x_max;
	if (f_v) {
		cout << "diophant::read_xml f_x_max=" << f_x_max << endl;
	}
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
			type[i] = t_ZOR;
			}
		f >> RHS[i];

		//f.ignore(INT_MAX, '\"');
		while (TRUE) {
			f >> c;
			if (c == '\"') {
				break;
				}
			}
		l = 0;
		while (TRUE) {
			f >> c;
			if (c == '\"') {
				break;
				}
			tmp[l] = c;
			l++;
			}
		tmp[l] = 0;
		eqn_label[i] = NEW_char(l + 1);
		for (j = 0; j < l; j++) {
			eqn_label[i][j] = tmp[j];
			}
		eqn_label[i][l] = 0;
		}
	if (f_x_max) {
		if (f_v) {
			cout << "diophant::read_xml reading x_max[]" << endl;
		}
		for (j = 0; j < n; j++) {
			f >> x_max[j];
			if (f_v) {
				cout << "diophant::read_xml reading x_max[" << j << "]="
						<< x_max[j] << endl;
			}
		}
	}
	write_xml(cout, label);
#endif
#ifdef SYSTEMWINDOWS
	cout << "diophant::read_xml has a problem under windows"<< endl;
	exit(1);
#endif
}

void diophant::append_equation()
{
	int *AA, *R, *R1, *Y1;
	diophant_equation_type *type1;
	char **L;
	int m1 = m + 1;
	int i, j;

	AA = NEW_int(m1 * n);
	R = NEW_int(m1);
	R1 = NEW_int(m1);
	type1 = NEW_OBJECTS(diophant_equation_type, m1);
	L = NEW_pchar(m1);
	Y1 = NEW_int(m1);
	
	for (i = 0; i < m; i++) {

		for (j = 0; j < n; j++) {
			AA[i * n + j] = Aij(i, j);
			}
		R[i] = RHS[i];
		R1[i] = RHS1[i];
		type1[i] = type[i];
		L[i] = eqn_label[i];
		}

	FREE_int(A);
	FREE_int(RHS);
	FREE_int(RHS1);
	FREE_OBJECTS(type);
	FREE_pchar(eqn_label);
	FREE_int(Y);

	A = AA;
	RHS = R;
	RHS1 = R1;
	type = type1;
	eqn_label = L;
	Y = Y1;

	int_vec_zero(A + m * n, n);
	RHS[m] = 0;
	RHS1[m] = 0;
	type[m] = t_EQ;
	eqn_label[m] = NULL;

	m++;
	
}

void diophant::delete_equation(int I)
{
	int i, j;
	
	if (eqn_label[I]) {
		FREE_char(eqn_label[I]);
		eqn_label[I] = NULL;
		}
	for (i = I; i < m - 1; i++) {
		eqn_label[i] = eqn_label[i + 1];
		eqn_label[i + 1] = NULL;
		type[i] = type[i + 1];
		RHS[i] = RHS[i + 1];
		for (j = 0; j < n; j++) {
			Aij(i, j) = Aij(i + 1, j);
			}
		}
	m--;
}

void diophant::write_gurobi_binary_variables(const char *fname)
{
	int i, j, a;
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
			if (a == 0)
				continue;
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
			<< file_size(fname) << endl;
}

void diophant::draw_it(const char *fname_base,
		int xmax_in, int ymax_in, int xmax_out, int ymax_out)
{
	int f_dots = FALSE;
	int f_partition = FALSE;
	int f_bitmatrix = FALSE;
	int f_row_grid = FALSE;
	int f_col_grid = FALSE;
	double scale = 0.5;
	double line_width = 0.5;
	

	draw_bitmatrix(fname_base, f_dots, 
		f_partition, 0, NULL, 0, NULL, 
		f_row_grid, f_col_grid, 
		f_bitmatrix, NULL, A, 
		m, n, xmax_in, ymax_in, xmax_out, ymax_out, 
		scale, line_width, 
		FALSE, NULL);
		// in draw.C
}

void diophant::draw_partitioned(const char *fname_base, 
	int xmax_in, int ymax_in, int xmax_out, int ymax_out, 
	int f_solution, int *solution, int solution_sz, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_dots = FALSE;
	int f_bitmatrix = FALSE;
	double scale = 0.5;
	double line_width = 0.5;
	int i, ii, j, jj;
	
	
	if (f_v) {
		cout << "diophant::draw_partitioned" << endl;
		}
	
	int *T;
	int *A2;
	int a;

	T = NEW_int(m);
	A2 = NEW_int(m * n);
	int_vec_zero(A2, m * n);

	for (i = 0; i < m; i++) {
		if (type[i] == t_EQ) {
			T[i] = 1;
			}
		else if (type[i] == t_LE) {
			T[i] = 2;
			}
		else if (type[i] == t_ZOR) {
			T[i] = 3;
			}
		}

	classify C;

	C.init(T, m, FALSE, 0);
	if (f_v) {
		cout << "diophant::draw_partitioned we found " << C.nb_types
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

		int_vec_copy(solution, col_perm + n - solution_sz, solution_sz);
		set_complement(solution, solution_sz, col_perm, size_complement, n);
		
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
	if (FALSE) {
		cout << "diophant::draw_partitioned A2=" << endl;
		int_matrix_print(A2, m, n);
		}

	int f_row_grid = FALSE;
	int f_col_grid = FALSE;


	draw_bitmatrix(fname_base, f_dots, 
		TRUE /* f_partition */, C.nb_types, part, col_part_size, part_col, 
		f_row_grid, f_col_grid, 
		f_bitmatrix, NULL, A2, 
		m, n, xmax_in, ymax_in, xmax_out, ymax_out, 
		scale, line_width, 
		FALSE, NULL);
		// in draw.C

	FREE_int(T);
	FREE_int(A2);
	FREE_int(part);
	FREE_int(part_col);
	FREE_int(col_perm);
	if (f_v) {
		cout << "diophant::draw_partitioned done" << endl;
		}
}

int diophant::test_solution(int *sol, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, b, c, ret;

	if (f_v) {
		cout << "diophant::test_solution" << endl;
		}
	if (FALSE) {
		int_vec_print(cout, sol, len);
		cout << endl;
		set_of_sets *S;

		get_columns(sol, len, S, 0 /* verbose_level */);
		S->print_table();

		FREE_OBJECT(S);
		}
	int_vec_zero(Y, m);
	int_vec_zero(X, n);
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
	if (FALSE) {
		cout << "Y=";
		int_vec_print_fully(cout, Y, m);
		cout << endl;
		}
	ret = TRUE;
	for (i = 0; i < m; i++) {
		if (type[i] == t_EQ) {
			if (Y[i] != RHS[i]) {
				cout << "diophant::test_solution Y[i] != RHS[i]" << endl;
				exit(1);
				}
			}
		else if (type[i] == t_LE) {
			if (Y[i] > RHS[i]) {
				cout << "diophant::test_solution Y[i] > RHS[i]" << endl;
				exit(1);
				}

			}
		else if (type[i] == t_ZOR) {
			if (Y[i] != 0 && Y[i] != RHS[i]) {
				ret = FALSE;
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


void diophant::get_columns(int *col, int nb_col,
		set_of_sets *&S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h, d;

	if (f_v) {
		cout << "diophant::get_columns" << endl;
		}
	S = NEW_OBJECT(set_of_sets);

	S->init_simple(m, nb_col, 0 /* verbose_level */);
	for (h = 0; h < nb_col; h++) {
		j = col[h];
		d = 0;
		for (i = 0; i < m; i++) {
			if (Aij(i, j)) {
				d++;
				}
			}
		S->Sets[h] = NEW_int(d);
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

void diophant::test_solution_file(const char *solution_file,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Sol;
	int nb_sol, sol_length;
	int i;

	if (f_v) {
		cout << "diophant::test_solution_file" << endl;
		}
	int_matrix_read_text(solution_file, Sol, nb_sol, sol_length);
	
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
		int_vec_print(cout, Y, m);
		cout << endl;

		classify C;

		C.init(Y, m, FALSE, 0);
		cout << "classification: ";
		C.print_naked(FALSE);
		cout << endl;
		}
	if (f_v) {
		cout << "diophant::test_solution_file done" << endl;
		}
}

void diophant::analyze(int verbose_level)
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
			return FALSE;
			}
		if (RHSi(i) != 1) {
			return FALSE;
			}
		}
	return TRUE;
}

void diophant::make_clique_graph_adjacency_matrix(uchar *&Adj,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j1, j2, L, k, i;
	//int length;

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
	//length = (L + 7) >> 3;
	Adj = bitvector_allocate(L);
	for (i = 0; i < L; i++) {
		bitvector_m_ii(Adj, i, 1);
		}
	for (i = 0; i < m; i++) {
		for (j1 = 0; j1 < n; j1++) {
			if (Aij(i, j1) == 0) {
				continue;
				}
			for (j2 = j1 + 1; j2 < n; j2++) {
				if (Aij(i, j2) == 0) {
					continue;
					}
				// now: j1 and j2 do not go together
				k = ij2k(j1, j2, n);
				bitvector_m_ii(Adj, k, 0);
				}
			}
		}
	if (f_v) {
		cout << "diophant::make_clique_graph_adjacency_"
				"matrix done" << endl;
		}
}


void diophant::make_clique_graph(colored_graph *&CG, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	uchar *Adj;

	if (f_v) {
		cout << "diophant::make_clique_graph" << endl;
		}
	make_clique_graph_adjacency_matrix(Adj, verbose_level - 1);


	CG = NEW_OBJECT(colored_graph);

	CG->init_no_colors(n, Adj, TRUE, verbose_level - 1);
	
	
	if (f_v) {
		cout << "diophant::make_clique_graph" << endl;
		}
}

void diophant::make_clique_graph_and_save(
		const char *clique_graph_fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant::make_clique_graph_and_save" << endl;
		}

	colored_graph *CG;

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

// #############################################################################
// callbacks and globals
// #############################################################################



void diophant_callback_solution_found(int *sol, int len, 
	int nb_sol, void *data)
{
	int f_v = FALSE;
	diophant *D = (diophant *) data;
	vector<int> lo;
	int i, j;

	if ((nb_sol % 1000) == 0) {
		f_v = TRUE;
		}
	if (f_v) {
		cout << "diophant_callback_solution_found recording solution "
				<< nb_sol << " len = " << len << " : ";
		int_vec_print(cout, sol, len);
		cout << endl;
		cout << "D->_resultanz=" << D->_resultanz << endl;

		for (i = 0; i < len; i++) {
			cout << DLX_Cur_col[i] << "/" << sol[i];
			if (i < len - 1) {
				cout << ", ";
				}
			}
		cout << endl;
		}

	if (!D->test_solution(sol, len, 0 /* verbose_level */)) {
		cout << "diophant_callback_solution_found the solution "
				"is not a solution" << endl;
		exit(1);
		return;
		}
	else {
		if (f_v) {
			cout << "diophant_callback_solution_found D->test_solution "
					"returns TRUE" << endl;
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


int diophant_solve_first_mckay(diophant *Dio,
		int f_once, int verbose_level)
{
	int f_v = TRUE;//(verbose_level >= 1);
	int j;
	int maxresults = 10000000;
	vector<int> res;
	int nb_backtrack_nodes;
	int nb_sol;

	//verbose_level = 4;
	if (f_v) {
		cout << "diophant::solve_first_mckay calling solve_mckay" << endl;
		}
	if (f_once) {
		maxresults = 1;
		}
	diophant_solve_mckay(Dio, "",
			maxresults, nb_backtrack_nodes, nb_sol,
			verbose_level - 2);
	if (f_v) {
		cout << "diophant_solve_first_mckay found " << Dio->_resultanz
			<< " solutions, using " << nb_backtrack_nodes
			<< " backtrack nodes" << endl;
		}
	Dio->_cur_result = 0;
	if (Dio->_resultanz == 0)
		return FALSE;
	res = Dio->_results.front();
	for (j = 0; j < Dio->n; j++) {
		Dio->x[j] = res[j];
		}
	Dio->_results.pop_front();
	Dio->_cur_result++;
	if (f_v) {
		cout << "diophant_solve_first_mckay done" << endl;
		}
	return TRUE;
}

int diophant_solve_all_mckay(diophant *Dio,
		int &nb_backtrack_nodes, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int maxresults = 10000000;
	//int nb_backtrack_nodes;
	int nb_sol;
	
	diophant_solve_mckay(Dio, Dio->label,
			maxresults, nb_backtrack_nodes, nb_sol,
			verbose_level);
	if (f_v) {
		cout << "diophant_solve_all_mckay found " << Dio->_resultanz
				<< " solutions in " << nb_backtrack_nodes
				<< " backtrack nodes" << endl;
		}
	return Dio->_resultanz;
}

int diophant_solve_once_mckay(diophant *Dio, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int maxresults = 1;
	int nb_backtrack_nodes;
	int nb_sol;

	diophant_solve_mckay(Dio, Dio->label,
			maxresults, nb_backtrack_nodes, nb_sol,
			verbose_level - 2);
	if (f_v) {
		cout << "diophant_solve_once_mckay found " << Dio->_resultanz
				<< " solutions in " << nb_backtrack_nodes
				<< " backtrack nodes" << endl;
		}
	return Dio->_resultanz;
}


int diophant_solve_next_mckay(diophant *Dio, int verbose_level)
{
	int j;
	if (Dio->_cur_result < Dio->_resultanz) {
		for (j = 0; j < Dio->n; j++) {
			Dio->x[j] = Dio->_results.front()[j];
			}
		Dio->_results.pop_front();
		Dio->_cur_result++;
		return TRUE;
		}
	else {
		return FALSE;
		}
}


void diophant_solve_mckay(diophant *Dio,
		const char *label, int maxresults,
		int &nb_backtrack_nodes, int &nb_sol,
		int verbose_level)
{
	diophant_solve_mckay_override_minrhs_in_inequalities(Dio,
			label, maxresults, nb_backtrack_nodes,
			0 /* minrhs */, nb_sol, verbose_level);
}

void diophant_solve_mckay_override_minrhs_in_inequalities(
	diophant *Dio, const char *label,
	int maxresults, int &nb_backtrack_nodes, 
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
		cout << "diophant_solve_mckay_override_minrhs_in_inequalities "
			<< label << ", a system of size " << Dio->m
			<< " x " << Dio->n << endl;
		}
	lgs.Init(Dio, label, (int) Dio->m + 1, (int) Dio->n);
	minres.resize(Dio->m + 1);
	maxres.resize(Dio->m + 1);
	fanz.resize(Dio->m + 1);
	eqn.resize(Dio->m + 1);
	
	for (i = 0; i < Dio->m; i++) {
		// the RHS:
		if (Dio->type[i] == t_LE) {
			minres[i] = (int) minrhs;
			maxres[i] = (int) Dio->RHS[i];
			}
		else {
			minres[i] = (int) Dio->RHS[i];
			maxres[i] = (int) Dio->RHS[i];
			}
		
		// count the number of nonzero coefficients:
		nb = 0;
		for (j = 0; j < Dio->n; j++) {
			if (Dio->A[i * Dio->n + j])
				nb++;
			}
		
		// initialize coefficients:
		fanz[i] = (int) nb;
		eqn[i].resize(nb);
		nb = 0;
		for (j = 0; j < Dio->n; j++) {
			if (Dio->A[i * Dio->n + j]) {
				eqn[i][nb].var = (int) j;
				eqn[i][nb].coeff = (int) Dio->A[i * Dio->n + j];
				nb++;
				}
			}
		}
	
	// one more equation for \sum x_j = sum
	i = Dio->m;
	fanz[i] = (int) Dio->n;
	eqn[i].resize(Dio->n);
	for (j = 0; j < Dio->n; j++) {
		eqn[i][j].var = (int) j;
		eqn[i][j].coeff = 1;
		}
	minres[i] = (int) Dio->sum;
	maxres[i] = (int) Dio->sum;
	
	// now the bounds on the x_j
	minvarvalue.resize(Dio->n);
	maxvarvalue.resize(Dio->n);
	if (Dio->f_x_max) {
		for (j = 0; j < Dio->n; j++) {
			minvarvalue[j] = 0;
			maxvarvalue[j] = (int) Dio->x_max[j];
			}
		}
	else {
		for (j = 0; j < Dio->n; j++) {
			minvarvalue[j] = 0;
			maxvarvalue[j] = (int) Dio->sum;
			}
		}
#if 0
  for (j=1; j<=_eqnanz; j++) {
    minres[j-1] = _eqns[j-1].GetMinResult();
    maxres[j-1] = _eqns[j-1].GetMaxResult();
    fanz[j-1] = _eqns[j-1].GetVarAnz();
    eqn[j-1].resize(_eqns[j-1].GetVarAnz());
    it = _eqns[j-1].GetFaktoren().begin();
    for (i=1; i<=_eqns[j-1].GetVarAnz(); i++) {
      eqn[j-1][i-1].var=it->first-1;
      eqn[j-1][i-1].coeff=it->second;
      it++;
    }
  }
#endif
	Dio->_resultanz = 0;
	Dio->_maxresults = (int) maxresults;
	
	lgs.possolve(minvarvalue, maxvarvalue, eqn, minres, maxres, fanz, 
		(int) Dio->m + 1, (int) Dio->n, (int) verbose_level);
	nb_backtrack_nodes = lgs.nb_calls_to_solve;
	nb_sol = Dio->_resultanz;
	if (f_v) {
		cout << "diophant_solve_mckay_override_minrhs_"
			"in_inequalities " << label
			<< " finished, number of solutions = " << Dio->_resultanz 
			<< " nb_backtrack_nodes=" << nb_backtrack_nodes << endl;
		}
}


void solve_diophant(int *Inc,
	int nb_rows, int nb_cols, int nb_needed,
	int f_has_Rhs, int *Rhs, 
	int *&Solutions, int &nb_sol, int &nb_backtrack, int &dt, 
	int f_DLX, 
	int f_draw_system, const char *fname_system, 
	int f_write_tree, const char *fname_tree, 
	int verbose_level)
// allocates Solutions[nb_sol * nb_needed]
{
	int f_v = (verbose_level >= 1);
	//int f_v4 = FALSE; //(verbose_level >= 2);
	//int i, j;
	diophant *Dio;
	int t0 = os_ticks();

	if (f_v) {
		cout << "solve_diophant nb_rows=" << nb_rows << " nb_cols="
			<< nb_cols << " f_has_Rhs=" << f_has_Rhs
			<< " verbose_level=" << verbose_level << endl;
		cout << "f_write_tree=" << f_write_tree << endl;
		cout << "f_DLX=" << f_DLX << endl;
		//int_matrix_print(Inc, nb_rows, nb_cols);
		}
	Dio = NEW_OBJECT(diophant);

	if (f_has_Rhs) {
		Dio->init_problem_of_Steiner_type_with_RHS(nb_rows, 
			nb_cols, Inc, nb_needed, 
			Rhs, 
			0 /* verbose_level */);
		}
	else {
		Dio->init_problem_of_Steiner_type(nb_rows, 
			nb_cols, Inc, nb_needed, 
			0 /* verbose_level */);
		}

	if (f_draw_system) {
		int xmax_in = 1000000;
		int ymax_in = 1000000;
		int xmax_out = 1000000;
		int ymax_out = 1000000;
		
		if (f_v) {
			cout << "solve_diophant drawing the system" << endl;
			}
		Dio->draw_it(fname_system, xmax_in, ymax_in, xmax_out, ymax_out);
		if (f_v) {
			cout << "solve_diophant drawing the system done" << endl;
			}
		}

	if (FALSE /*f_v4*/) {
		Dio->print();
		}

	if (f_DLX && !f_has_Rhs) {
		Dio->solve_all_DLX(f_write_tree, fname_tree, 0 /* verbose_level*/);
		nb_backtrack = Dio->nb_steps_betten;
		}
	else {
		diophant_solve_all_mckay(Dio, nb_backtrack, verbose_level - 2);
		}

	nb_sol = Dio->_resultanz;
	if (nb_sol) {
		Dio->get_solutions(Solutions, nb_sol, 1 /* verbose_level */);
		if (FALSE /*f_v4*/) {
			cout << "Solutions:" << endl;
			int_matrix_print(Solutions, nb_sol, nb_needed);
			}
		}
	else {
		Solutions = NULL;
		}
	FREE_OBJECT(Dio);
	int t1 = os_ticks();
	dt = t1 - t0;
	if (f_v) {
		cout << "solve_diophant done nb_sol=" << nb_sol
				<< " nb_backtrack=" << nb_backtrack << " dt=" << dt << endl;
		}
}

}
}



