// cheat_sheet_PG.cpp
// 
// Anton Betten
// 3/14/2010
//
//
// 
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;


// global data:

int t0; // the system time when the program started



int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_override_poly = FALSE;
	char *my_override_poly = NULL;
	int f_n = FALSE;
	int n = 0;
	int f_q = FALSE;
	int q = 0;
	int f_surface = FALSE;
	os_interface Os;

 	t0 = Os.os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_override_poly = TRUE;
			my_override_poly = argv[++i];
			cout << "-poly " << my_override_poly << endl;
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
		else if (strcmp(argv[i], "-surface") == 0) {
			f_surface = TRUE;
			cout << "-surface " << endl;
			}
		}
	
	if (!f_n) {
		cout << "please use -n option to specify n" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q option to specify q" << endl;
		exit(1);
		}
	finite_field *F;

	F = NEW_OBJECT(finite_field);

	F->init_override_polynomial(q, my_override_poly, 0);

	F->cheat_sheet_PG(n, f_surface, verbose_level);
	
	FREE_OBJECT(F);

	the_end(t0);
}

