// analyze_projective_code.C
// 
// Anton Betten
// 7/16/2011
//
//
// 
//
//
//

#include "orbiter.h"

#define MY_MAX_SET_SIZE 2000

// global data:

int t0; // the system time when the program started


void do_analyze_projective_code(int n, finite_field *F, 
	int *the_set, int set_size, int verbose_level);


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_n = FALSE;
	int n = 0;
	int f_q = FALSE;
	int q;
	int f_file = FALSE;
	char *file_name;
	int f_poly = FALSE;
	const char *poly = "";
	int *the_set;
	int set_size = 0;

	
 	t0 = os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			file_name = argv[++i];
			cout << "-file " << file_name << endl;
			}
		}
	if (!f_n) {
		cout << "please use -n option" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q option" << endl;
		exit(1);
		}

	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(q, poly, 0);

	if (f_file) {
		read_set_from_file(file_name, the_set, set_size, verbose_level - 2);
		if (set_size > MY_MAX_SET_SIZE) {
			cout << "set is too big, please increase MY_MAX_SET_SIZE" << endl;
			exit(1);
			}
		}
	else {
		cout << "please use option -file <fname> to specify the file" << endl;
		exit(1);
		}

	do_analyze_projective_code(n, F, the_set, set_size, verbose_level);

	FREE_OBJECT(F);
	the_end(t0);
}

void do_analyze_projective_code(int n, finite_field *F, 
	int *the_set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	//int f_with_group = FALSE;
	//int f_semilinear = FALSE;
	//int f_basis = FALSE;
	int *genma;
	int *v;
	int d, i, j;
	int q = F->q;
	int f_elements_exponential = TRUE;
	const char *symbol_for_print = "\\alpha";

	if (f_v) {
		cout << "analyzing projective code of length " << set_size << endl;
		cout << "in PG(" << n << "," << q << ")" << endl;
		}

	P = NEW_OBJECT(projective_space);

	P->init(n, F, 
		//f_with_group, 
		//FALSE /* f_line_action */, 
		TRUE /* f_init_incidence_structure */, 
		//f_semilinear, 
		//f_basis, 
		0 /*verbose_level*/);

	
	d = n + 1;
	v = NEW_int(d);
	genma = NEW_int(d * set_size);
	
	for (j = 0; j < set_size; j++) {
		P->unrank_point(v, the_set[j]);
		for (i = 0; i < d; i++) {
			genma[i * set_size + j] = v[i];
			}
		}

	if (f_v) {
		cout << "generator matrix m=" << d << " n=" << set_size << endl;
		F->latex_matrix(cout,
				f_elements_exponential, symbol_for_print,
				genma, d, set_size);
		print_integer_matrix_width(cout, genma,
				d, set_size, set_size, 2);
		}

	int N;
	int min_d;
	int *weights;
	F->code_projective_weights(set_size, d,
			genma, weights, verbose_level);

	N = nb_PG_elements(n, q);
	min_d = weights[0];
	for (i = 1; i < N; i++) {
		if (weights[i] < min_d) {
			min_d = weights[i];
			}
		}
	cout << "minimum distance " << min_d << endl;

	classify C;

	C.init(weights, N, FALSE, 0);
	cout << "projective weights: ";
	C.print(FALSE /*f_backwards*/);

	FREE_OBJECT(P);
}




