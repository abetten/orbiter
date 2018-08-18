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

INT t0; // the system time when the program started


void do_analyze_projective_code(INT n, finite_field *F, 
	INT *the_set, INT set_size, INT verbose_level);


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_n = FALSE;
	INT n = 0;
	INT f_q = FALSE;
	INT q;
	INT f_file = FALSE;
	BYTE *file_name;
	INT f_poly = FALSE;
	const BYTE *poly = "";
	INT *the_set;
	INT set_size = 0;

	
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

	F = new finite_field;
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

	the_end(t0);
}

void do_analyze_projective_code(INT n, finite_field *F, 
	INT *the_set, INT set_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	//INT f_with_group = FALSE;
	//INT f_semilinear = FALSE;
	//INT f_basis = FALSE;
	INT *genma;
	INT *v;
	INT d, i, j;
	INT q = F->q;
	INT f_elements_exponential = TRUE;
	const BYTE *symbol_for_print = "\\alpha";

	if (f_v) {
		cout << "analyzing projective code of length " << set_size << endl;
		cout << "in PG(" << n << "," << q << ")" << endl;
		}

	P = new projective_space;

	P->init(n, F, 
		//f_with_group, 
		//FALSE /* f_line_action */, 
		TRUE /* f_init_incidence_structure */, 
		//f_semilinear, 
		//f_basis, 
		0 /*verbose_level*/);

	
	d = n + 1;
	v = NEW_INT(d);
	genma = NEW_INT(d * set_size);
	
	for (j = 0; j < set_size; j++) {
		P->unrank_point(v, the_set[j]);
		for (i = 0; i < d; i++) {
			genma[i * set_size + j] = v[i];
			}
		}

	if (f_v) {
		cout << "generator matrix m=" << d << " n=" << set_size << endl;
		F->latex_matrix(cout, f_elements_exponential, symbol_for_print, genma, d, set_size);
		print_integer_matrix_width(cout, genma, d, set_size, set_size, 2);
		}

	INT N;
	INT min_d;
	INT *weights;
	F->code_projective_weights(set_size, d, genma, weights, verbose_level);

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

	delete P;
}




