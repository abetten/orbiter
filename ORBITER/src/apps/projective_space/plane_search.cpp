// plane_search.C
// 
// Anton Betten
// Oct 6, 2010
//
//
// 
//
//

#include "orbiter.h"

using namespace orbiter;



// global data:

int t0; // the system time when the program started

int main(int argc, char **argv);
void plane_search(int q, int verbose_level);
void backtrack(finite_field *F, int row, int verbose_level);
void compute_row(finite_field *F, int row);
int check_row(finite_field *F, int row, int verbose_level);
void remove_row(finite_field *F, int row);
void clear_row(finite_field *F, int row);

int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_q = FALSE;
	int q;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		}
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}
	plane_search(q, verbose_level);
	
	the_end(t0);
}

int nb_sol = 0;
int *lambda; // [q]
int *M; // [q * q]
int *f_column_entry; // [q * q]

void plane_search(int q, int verbose_level)
{
	int i;
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);
	
	M = NEW_int(q * q);
	for (i = 0; i < q * q; i++) {
		M[i] = 0;
		}
	f_column_entry = NEW_int(q * q);
	for (i = 0; i < q * q; i++) {
		f_column_entry[i] = FALSE;
		}
	lambda = NEW_int(q);
	lambda[0] = 0;
	compute_row(F, 0);
	if (!check_row(F, 0, verbose_level)) {
		cout << "row 0 does not fit" << endl;
		exit(1);
		}
	lambda[1] = 0;
	compute_row(F, 1);
	if (!check_row(F, 1, verbose_level)) {
		cout << "row 1 does not fit" << endl;
		exit(1);
		}

	backtrack(F, 2, verbose_level);
	cout << "nb_sol=" << nb_sol << endl;
	FREE_OBJECT(F);
}

void backtrack(finite_field *F, int row, int verbose_level)
{
	int q = F->q;
	int f_v = (verbose_level >= 1);
	int f = F->e;

	if (f_v) {
		cout << "backtrack row=" << row << endl;
				cout << "lambda=";
				int_vec_print(cout, lambda, row);
				cout << endl;
				cout << "M=" << endl;
				int_matrix_print(M, q, q);
				cout << endl;
				cout << "f_column_entry=" << endl;
				int_matrix_print(f_column_entry, q, q);
				cout << endl;
		}
	if (row == q) {
		cout << "solution " << nb_sol << " : ";
		int_vec_print(cout, lambda, q);
		cout << endl;
		nb_sol++;
		return;
		}
	for (lambda[row] = 0; lambda[row] < f; lambda[row]++) {
		if (f_v) {
			cout << "backtrack row=" << row << " lambda=" << lambda[row] << endl;
			}
		compute_row(F, row);
		if (check_row(F, row, verbose_level)) {
			if (f_v) {
				cout << "accepted" << endl;
				cout << "lambda=";
				int_vec_print(cout, lambda, row + 1);
				cout << endl;
				cout << "M=" << endl;
				int_matrix_print(M, q, q);
				cout << endl;
				cout << "f_column_entry=" << endl;
				int_matrix_print(f_column_entry, q, q);
				cout << endl;
				}
			backtrack(F, row + 1, verbose_level);
			remove_row(F, row);
			if (f_v) {
				cout << "after remove_row:" << endl;
				cout << "lambda=";
				int_vec_print(cout, lambda, row + 1);
				cout << endl;
				cout << "M=" << endl;
				int_matrix_print(M, q, q);
				cout << endl;
				cout << "f_column_entry=" << endl;
				int_matrix_print(f_column_entry, q, q);
				cout << endl;
				}
			}
		else {
			if (f_v) {
				cout << "rejected" << endl;
				cout << "lambda=";
				int_vec_print(cout, lambda, row + 1);
				cout << endl;
				cout << "M=" << endl;
				int_matrix_print(M, q, q);
				cout << endl;
				cout << "f_column_entry=" << endl;
				int_matrix_print(f_column_entry, q, q);
				cout << endl;
				}
			}
		clear_row(F, row);
		}
}


void compute_row(finite_field *F, int row)
{
	int x, x1, y, l;
	int q = F->q;

	l = lambda[row];
	for (x = 1; x < q; x++) {
		x1 = F->frobenius_power(x, l);
		y = F->mult(x1, row);
		M[row * q + x] = y;
		}
}

int check_row(finite_field *F, int row, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int x, y, x1;
	int q = F->q;

	for (x = 1; x < q; x++) {
		y = M[row * q + x];
		if (f_column_entry[x * q + y]) {
			for (x1 = 1; x1 < x; x1++) {
				y = M[row * q + x1];
				f_column_entry[x1 * q + y] = FALSE;
				}
			if (f_v) {
				cout << "check_row row=" << row
						<< " x=" << x << " y=" << y << " fails" << endl;
				}
			return FALSE;
			}
		f_column_entry[x * q + y] = TRUE;
		}
	return TRUE;
}

void remove_row(finite_field *F, int row)
{
	int x, y;
	int q = F->q;
	
	for (x = 1; x < q; x++) {
		y = M[row * q + x];
		if (!f_column_entry[x * q + y]) {
			cout << "remove_row row=" << row
					<< " x=" << x << " y=" << y
					<< " entry in f_column_entry is not there" << endl;
			exit(1);
			}
		f_column_entry[x * q + y] = FALSE;
		}
}

void clear_row(finite_field *F, int row)
{
	int x, y;
	int q = F->q;
	
	for (x = 1; x < q; x++) {
		y = M[row * q + x];
		M[row * q + x] = 0;
		}
}


