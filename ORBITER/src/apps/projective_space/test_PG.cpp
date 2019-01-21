// test_PG.C
// 
// Anton Betten
// 1/17/2010
//
//
// 
//
//

#include "orbiter.h"


using namespace orbiter;



// global data:

int t0; // the system time when the program started

void test1(int n, int q, int verbose_level);

int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_n = FALSE;
	int n = 0;
	int f_q = FALSE;
	int q;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc - 1; i++) {
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
		}
	if (!f_n) {
		cout << "please use -n option" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q option" << endl;
		exit(1);
		}
	test1(n, q, verbose_level);
	
	the_end(t0);
}


void test1(int n, int q, int verbose_level)
{
	finite_field *F;
	projective_space *P;

	F = NEW_OBJECT(finite_field);
	P = NEW_OBJECT(projective_space);

	F->init(q, 0);
	P->init(n, F, 
		TRUE /* f_init_incidence_structure */, 
		verbose_level);

	FREE_OBJECT(P);
	FREE_OBJECT(F);
}

