// desarguesian_spread.cpp
// 
// Anton Betten
// July 21, 2014
//
//
//

#include <orbiter.h>

using namespace std;


using namespace orbiter;


// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);



int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;
	int f_n = FALSE;
	int n;
	int f_s = FALSE;
	int s;
	int f_q = FALSE;
	int q;
	int f_poly_q = FALSE;
	const char *poly_q = NULL;
	int f_poly_Q = FALSE;
	const char *poly_Q = NULL;

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
		else if (strcmp(argv[i], "-s") == 0) {
			f_s = TRUE;
			s = atoi(argv[++i]);
			cout << "-s " << s << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-poly_q") == 0) {
			f_poly_q = TRUE;
			poly_q = argv[++i];
			cout << "-poly_q " << poly_q << endl;
			}
		else if (strcmp(argv[i], "-poly_Q") == 0) {
			f_poly_Q = TRUE;
			poly_Q = argv[++i];
			cout << "-poly_Q " << poly_Q << endl;
			}
		}
	if (!f_n) {
		cout << "please use -n option" << endl;
		exit(1);
		}
	if (!f_s) {
		cout << "please use -s option" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q option" << endl;
		exit(1);
		}
	int f_v = (verbose_level >= 1);

	int Q;
	int m;
	number_theory_domain NT;

	m = n / s;
	if (m * s != n) {
		cout << "m * s != n" << endl;
		exit(1);
		}

	Q = NT.i_power_j(q, s);
	
	finite_field *Fq;
	finite_field *FQ;
	subfield_structure *SubS;
	desarguesian_spread *D;

	Fq = NEW_OBJECT(finite_field);
	if (f_v) {
		cout << "before Fq->init" << endl;
		}
	Fq->init_override_polynomial(q, poly_q, 0);

	FQ = NEW_OBJECT(finite_field);
	if (f_v) {
		cout << "before FQ->init" << endl;
		}
	FQ->init_override_polynomial(Q, poly_Q, 0);

	SubS = NEW_OBJECT(subfield_structure);
	if (f_v) {
		cout << "before SubS->init" << endl;
		}
	SubS->init(FQ, Fq, verbose_level);

	if (f_v) {
		cout << "Field-basis: ";
		int_vec_print(cout, SubS->Basis, s);
		cout << endl;
		}
	
	D = NEW_OBJECT(desarguesian_spread);
	if (f_v) {
		cout << "before D->init" << endl;
		}
	D->init(n, m, s, 
		SubS, 
		verbose_level);
	
	D->print_spread_element_table_tex();

	
	the_end(t0);
}

