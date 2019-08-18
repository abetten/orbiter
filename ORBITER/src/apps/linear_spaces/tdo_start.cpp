// tdo_start.cpp
//
// Anton Betten
//
// started:  Dec 26 2006

#include "orbiter.h"

using namespace std;



using namespace orbiter;


int t0;

const char *version = "tdo_start Jan 30 2008";

char buf[BUFSIZE];

void print_usage();
int main(int argc, char **argv);
void create_all_linetypes(char *label_base, int m, int verbose_level);
void write_tdo_line_type(ofstream &g, char *label_base, int m, int n, 
	int nb_line_types, int *lines, int *multiplicities);
void write_row_scheme(ofstream &g, char *label_base, int nb_V, int nb_B, 
	int *V, int *B, int *the_scheme);
void write_col_scheme(ofstream &g, char *label_base, int nb_V, int nb_B, 
	int *V, int *B, int *the_scheme);
void write_td_scheme(ofstream &g, char *label_base, int nb_V, int nb_B, 
	int *V, int *B, int *the_scheme);

void print_usage()
{
	cout << "usage: tdo_start.out [options] <tdo_file>\n";
	cout << "where options can be:\n";
	cout << "-v <n>" << endl;
	cout << "  verbose level <n>" << endl;
	cout << "-conf <m> <n> <r> <k>" << endl;
	cout << "  create a TDO for a configuration m_r n_k" << endl;
	cout << "-linearspace <n> <i_1> <a_1> <i_2> <a_2> ... -1 <file_name>" << endl;
	cout << "  create TDO file for linear spaces on n points" << endl;
	cout << "  with a_j (> 0) lines of size i_j." << endl;
	cout << "  Note that \\sum_{j}a_j{i_j \\choose 2} = {n \\choose 2} " << endl;
	cout << "  is required. The output is written into the specified file," << endl;
	cout << "  as <file_name>.tdo" << endl;
	cout << "-all <n>" << endl;
	cout << "  Create a TDO-file with all possible line cases for" << endl;
	cout << "  linear spaces on <n> points" << endl;
	cout << "" << endl;
	cout << "-rowscheme <m> <n> <V_1> ... <V_m> <B_1> ... <B_n> <r_{1,1}> <r_{1,2}> ... <r_{m,n}>" << endl;
	cout << "-colscheme <m> <n> <V_1> ... <V_m> <B_1> ... <B_n> <k_{1,1}> <k_{1,2}> ... <k_{m,n}>" << endl;
	cout << "-tdscheme <m> <n> <V_1> ... <V_m> <B_1> ... <B_n> <r_{1,1}> <r_{1,2}> ... <r_{m,n}>" << endl;
}

int main(int argc, char **argv)
{
	char fname_out[1000];
	t0 = os_ticks();
	int verbose_level = 0;
	int f_conf = FALSE;
	int f_all = FALSE;
	int f_linearspace = FALSE;
	int f_rowscheme = FALSE;
	int f_colscheme = FALSE;
	int f_tdscheme = FALSE;
	int nb_V, nb_B;
	int *V, *B, *the_scheme;
	int row_level, col_level, lambda_level, extra_row_level, extra_col_level;
	int i, m, n, r, k, a, m2, a2, ii, jj;
	int nb_lines;
	int line_size[1000];
	int line_multiplicity[1000];
	char *label_base;
	char label[1000];
	combinatorics_domain Combi;

	cout << version << endl;
	if (argc <= 1) {
		print_usage();
		exit(1);
		}
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		if (strcmp(argv[i], "-conf") == 0) {
			f_conf = TRUE;
			m = atoi(argv[++i]);
			n = atoi(argv[++i]);
			r = atoi(argv[++i]);
			k = atoi(argv[++i]);
			cout << "-conf " << m << " " << n << " " << r << " " << k << endl;
		}
		if (strcmp(argv[i], "-linearspace") == 0) {
			f_linearspace = TRUE;
			m = atoi(argv[++i]);
			n = 0;
			for (nb_lines = 0; ; nb_lines++) {
				a = atoi(argv[++i]);
				if (a == -1)
					break;
				line_size[nb_lines] = a;
				a = atoi(argv[++i]);
				line_multiplicity[nb_lines] = a;
				n += a;
				}
			cout << "-linearspace " << m << " " << n << endl;
		}
		if (strcmp(argv[i], "-all") == 0) {
			f_all = TRUE;
			m = atoi(argv[++i]);
			cout << "-all " << m << endl;
		}
		if (strcmp(argv[i], "-rowscheme") == 0) {
			f_rowscheme = TRUE;
			nb_V = atoi(argv[++i]);
			nb_B = atoi(argv[++i]);
			V = new int[nb_V];
			B = new int[nb_B];
			the_scheme = new int[nb_V * nb_B];
			for (ii = 0; ii < nb_V; ii++) {
				V[ii] = atoi(argv[++i]);
				}
			for (jj = 0; jj < nb_B; jj++) {
				B[jj] = atoi(argv[++i]);
				}
			for (ii = 0; ii < nb_V; ii++) {
				for (jj = 0; jj < nb_B; jj++) {
					the_scheme[ii * nb_B + jj] = atoi(argv[++i]);
					}
				}
		}
		if (strcmp(argv[i], "-colscheme") == 0) {
			f_colscheme = TRUE;
			nb_V = atoi(argv[++i]);
			nb_B = atoi(argv[++i]);
			cout << "-colscheme " << nb_V << " " << nb_B << endl;
			V = new int[nb_V];
			B = new int[nb_B];
			the_scheme = new int[nb_V * nb_B];
			for (ii = 0; ii < nb_V; ii++) {
				V[ii] = atoi(argv[++i]);
				}
			cout << "V:" << endl;
			for (ii = 0; ii < nb_V; ii++) {
				cout << V[ii] << " ";
				}
			cout << endl;
			cout << "B:" << endl;
			for (jj = 0; jj < nb_B; jj++) {
				B[jj] = atoi(argv[++i]);
				}
			for (jj = 0; jj < nb_B; jj++) {
				cout << B[jj] << " ";
				}
			cout << endl;
			for (ii = 0; ii < nb_V; ii++) {
				for (jj = 0; jj < nb_B; jj++) {
					the_scheme[ii * nb_B + jj] = atoi(argv[++i]);
					}
				}
			cout << "scheme:" << endl;
			for (ii = 0; ii < nb_V; ii++) {
				for (jj = 0; jj < nb_B; jj++) {
					cout << setw(3) << the_scheme[ii * nb_B + jj];
					}
				cout << endl;
				}
			}
		if (strcmp(argv[i], "-tdscheme") == 0) {
			f_tdscheme = TRUE;
			nb_V = atoi(argv[++i]);
			nb_B = atoi(argv[++i]);
			V = new int[nb_V];
			B = new int[nb_B];
			the_scheme = new int[nb_V * nb_B];
			for (ii = 0; ii < nb_V; ii++) {
				V[ii] = atoi(argv[++i]);
				}
			for (jj = 0; jj < nb_B; jj++) {
				B[jj] = atoi(argv[++i]);
				}
			for (ii = 0; ii < nb_V; ii++) {
				for (jj = 0; jj < nb_B; jj++) {
					the_scheme[ii * nb_B + jj] = atoi(argv[++i]);
					}
				}
		}
	}
	label_base = argv[argc - 1];
	sprintf(fname_out, "%s.tdo", label_base);
	{
	
	if (f_linearspace) {
		m2 = Combi.binomial2(m);
		for (i = 0; i < nb_lines; i++) {
			a = line_size[i];
			a2 = Combi.binomial2(a);
			a2 *= line_multiplicity[i];
			m2 -= a2;
			}
		if (m2 < 0) {
			cout << "error in the line type" << endl;
			exit(1);
			}
		if (m2 > 0) {
			cout << "need " << m2 << " additional 2-lines" << endl;
			exit(1);
			}
		}
	
	if (f_conf) {
		cout << "opening file " << fname_out << " for writing" << endl;
		ofstream g(fname_out);
		sprintf(label, "%s.0", label_base);
		g << label << " " << m + n << " " << m << " " << -1 << " " 
			<< 2 << " " << 0 << " " << 1 << " " << r << " " 
			<< 2 << " " << 1 << " " << 0 << " " << k << " " 
			<< -1 << " ";
		row_level = col_level = lambda_level = 2;
		extra_row_level = -1;
		extra_col_level = -1;
		g << row_level << " " 
			<< col_level << " " 
			<< lambda_level << " "
			<< extra_row_level << " "
			<< extra_col_level << " "
			<< endl;
		g << -1 << endl;
		}
	if (f_linearspace) {
		cout << "opening file " << fname_out << " for writing" << endl;
		ofstream g(fname_out);
		write_tdo_line_type(g, label_base, m, n, 
			nb_lines, line_size, line_multiplicity);
		g << -1 << endl;
		}
	
	if (f_all) {
		create_all_linetypes(label_base, m, verbose_level);
		}
	if (f_rowscheme) {
		cout << "opening file " << fname_out << " for writing" << endl;
		ofstream g(fname_out);
		write_row_scheme(g, label_base, nb_V, nb_B, V, B, the_scheme);
		g << -1 << endl;
		}
	if (f_colscheme) {
		cout << "opening file " << fname_out << " for writing" << endl;
		ofstream g(fname_out);
		write_col_scheme(g, label_base, nb_V, nb_B, V, B, the_scheme);
		g << -1 << endl;
		}
	if (f_tdscheme) {
		cout << "opening file " << fname_out << " for writing" << endl;
		ofstream g(fname_out);
		write_td_scheme(g, label_base, nb_V, nb_B, V, B, the_scheme);
		g << -1 << endl;
		}
	
	}
	cout << "time: ";
	time_check(cout, t0);
	cout << endl;
}

void create_all_linetypes(char *label_base, int m, int verbose_level)
{
	char fname[1000];
	char label[1000];
	int nb_line_types;
	int *lines;
	int *multiplicities;
	int *types;
	int nb_eqns = 2, nb_vars = m;
	int nb_sol, m2, i, /* k, */ j, a, n, Nb_sol;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	combinatorics_domain Combi;
	
	cout << "create all line types of linear spaces on " << m << " points" << endl;
	lines = new int[m];
	multiplicities = new int[m];
	
	diophant D;
	
	D.open(nb_eqns, nb_vars);
	
	for (i = 2; i <= m; i++) {
		//cout << "i = " << i << " : " << 0 * nb_vars + i - 2 << " : " << binomial2(i) << endl;
		D.A[0 * nb_vars + nb_vars - i] = Combi.binomial2(i);
		}
	D.A[0 * nb_vars + m - 1] = 0;
	for (i = 0; i < nb_vars; i++) {
		D.A[1 * nb_vars + i] = 1;
		}
	m2 = Combi.binomial2(m);
	D.RHS[0] = m2;
	D.RHS[1] = m2;
	D.type[0] = t_EQ;
	D.type[1] = t_EQ;
	D.sum = m2;
	
	if (f_v) {
		D.print();
		}
	nb_sol = 0;
	if (D.solve_first_betten(f_vvv)) {
	
		while (TRUE) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < nb_vars; i++) {
					cout << " " << D.x[i];
					}
				cout << endl;
				}
			nb_sol++;
			if (!D.solve_next_betten(f_vvv))
				break;
			}
		}
	if (f_v) {
		cout << "found " << nb_sol << " line types" << endl;
		}
	Nb_sol = nb_sol;
	types = new int[Nb_sol * nb_vars];
	nb_sol = 0;
	if (D.solve_first_betten(f_vvv)) {
	
		while (TRUE) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < nb_vars; i++) {
					cout << " " << D.x[i];
					}
				cout << endl;
				}
			for (i = 0; i < nb_vars; i++) {
				types[nb_sol * nb_vars + i] = D.x[i];
				}
				
			nb_sol++;
			if (!D.solve_next_betten(f_vvv))
				break;
			}
		}
	
	//diophant_close(D);
	
	sprintf(fname, "%s.tdo", label_base);
	{
		cout << "opening file " << fname << " for writing" << endl;
		ofstream g(fname);


		for (i = 0; i < Nb_sol; i++) {
			//k = Nb_sol - 1 - i;
			nb_line_types = 0;
			for (j = 0; j < nb_vars - 1; j++) {
				a = types[i * nb_vars + j];
				if (a == 0) 
					continue;
				lines[nb_line_types] = m - j;
				multiplicities[nb_line_types] = a;
				nb_line_types++;
				}
			n = m2 - types[i * nb_vars + nb_vars - 1];
			sprintf(label, "%s.%d", label_base, i + 1);
			write_tdo_line_type(g, label, m, n, 
				nb_line_types, lines, multiplicities);
			}
		g << -1 << endl;
	}
	
	
	delete [] lines;
	delete [] multiplicities;
	delete [] types;
}

void write_tdo_line_type(ofstream &g, char *label_base, int m, int n, 
	int nb_line_types, int *lines, int *multiplicities)
{
	int a, j, row_level, col_level, lambda_level, extra_row_level, extra_col_level;
	
	g << label_base << " " << m + n << " " << m << " ";
	a = multiplicities[0];
	for (j = 1; j < nb_line_types; j++) {
		g << m + a << " ";
		a += multiplicities[j];
		}
	g << -1 << " ";
	col_level = 2 + nb_line_types - 1;
	row_level = 1;
	lambda_level = 2;
	extra_row_level = -1;
	extra_col_level = -1;
	for (j = 0; j < nb_line_types; j++) {
		g << col_level << " " << 1 + j << " " << 0 << " " << lines[j] << " ";
		}
	g << -1 << " ";
	g << row_level << " " 
		<< col_level << " " 
		<< lambda_level << " " 
		<< extra_row_level << " " 
		<< extra_col_level << " " 
		<< endl;
}

void write_row_scheme(ofstream &g, char *label_base, int nb_V, int nb_B, 
	int *V, int *B, int *the_scheme)
{
	int i, j, ii, jj, m, n, f, row_level, col_level, lambda_level;
	int extra_row_level = -1;
	int extra_col_level = -1;
	
	m = 0;
	for (i = 0; i < nb_V; i++) {
		m += V[i];
		}
	n = 0;
	for (j = 0; j < nb_B; j++) {
		n += B[j];
		}
	g << label_base << " " << m + n << " " << m << " ";
	f = 0;
	for (i = 1; i < nb_V; i++) {
		f += V[i - 1];
		g << f << " ";
		}
	f = m;
	for (j = 1; j < nb_B; j++) {
		f += B[j - 1];
		g << f << " ";
		}
	g << -1 << " ";
	col_level = 2;
	row_level = nb_V + nb_B;
	lambda_level = 2;
	for (i = 0; i < nb_V; i++) {
		if (i == 0)
			ii = 0;
		else 
			ii = i + 1;
		for (j = 0; j < nb_B; j++) {
			if (j == 0)
				jj = 1;
			else 
				jj = nb_V + j;
			g << row_level << " " 
				<< ii << " " << jj << " " 
				<< the_scheme[i * nb_B + j] << " ";
			}
		}
	g << -1 << " ";
	g << row_level << " " 
		<< col_level << " " 
		<< lambda_level << " " 
		<< extra_row_level << " " 
		<< extra_col_level << " " 
		<< endl;
}

void write_col_scheme(ofstream &g, char *label_base, int nb_V, int nb_B, 
	int *V, int *B, int *the_scheme)
{
	int i, j, ii, jj, m, n, f, row_level, col_level, lambda_level;
	int extra_row_level = -1;
	int extra_col_level = -1;
	
	m = 0;
	for (i = 0; i < nb_V; i++) {
		m += V[i];
		}
	n = 0;
	for (j = 0; j < nb_B; j++) {
		n += B[j];
		}
	g << label_base << " " << m + n << " " << m << " ";
	f = 0;
	for (i = 1; i < nb_V; i++) {
		f += V[i - 1];
		g << f << " ";
		}
	f = m;
	for (j = 1; j < nb_B; j++) {
		f += B[j - 1];
		g << f << " ";
		}
	g << -1 << " ";
	col_level = nb_V + nb_B;
	row_level = 2;
	lambda_level = 2;
	for (i = 0; i < nb_V; i++) {
		if (i == 0)
			ii = 0;
		else 
			ii = i + 1;
		for (j = 0; j < nb_B; j++) {
			if (j == 0)
				jj = 1;
			else 
				jj = nb_V + j;
			g << col_level << " " 
				<< jj << " " << ii << " " 
				<< the_scheme[i * nb_B + j] << " ";
			}
		}
	g << -1 << " ";
	g << row_level << " " 
		<< col_level << " " 
		<< lambda_level << " " 
		<< extra_row_level << " " 
		<< extra_col_level << " " 
		<< endl;
}

void write_td_scheme(ofstream &g, char *label_base, int nb_V, int nb_B, 
	int *V, int *B, int *the_scheme)
{
	int i, j, ii, jj, m, n, f, row_level, col_level, lambda_level, a, b, c;
	int extra_row_level = -1;
	int extra_col_level = -1;
	
	m = 0;
	for (i = 0; i < nb_V; i++) {
		m += V[i];
		}
	n = 0;
	for (j = 0; j < nb_B; j++) {
		n += B[j];
		}
	g << label_base << " " << m + n << " " << m << " ";
	f = 0;
	for (i = 1; i < nb_V; i++) {
		f += V[i - 1];
		g << f << " ";
		}
	f = m;
	for (j = 1; j < nb_B; j++) {
		f += B[j - 1];
		g << f << " ";
		}
	g << -1 << " ";
	col_level = nb_V + nb_B;
	row_level = nb_V + nb_B;
	lambda_level = 2;
	for (i = 0; i < nb_V; i++) {
		if (i == 0)
			ii = 0;
		else 
			ii = i + 1;
		for (j = 0; j < nb_B; j++) {
			if (j == 0)
				jj = 1;
			else 
				jj = nb_V + j;
			g << row_level << " " 
				<< ii << " " << jj << " " 
				<< the_scheme[i * nb_B + j] << " ";
			}
		}
	for (i = 0; i < nb_V; i++) {
		if (i == 0)
			ii = 0;
		else 
			ii = i + 1;
		for (j = 0; j < nb_B; j++) {
			if (j == 0)
				jj = 1;
			else 
				jj = nb_V + j;
			a = V[i];
			b = B[j];
			c = (a * the_scheme[i * nb_B + j]) / b;
			if (b * c != (a * the_scheme[i * nb_B + j])) {
				cout << "not tactical in (" << i << "," << j << ")-spot" << endl;
				exit(1);
				}
			g << col_level << " " 
				<< jj << " " << ii << " " 
				<< c << " ";
			}
		}
	g << -1 << " ";
	g << row_level << " " << col_level << " " 
		<< lambda_level << " " 
		<< extra_row_level << " " 
		<< extra_col_level << " " 
		<< endl;
}

