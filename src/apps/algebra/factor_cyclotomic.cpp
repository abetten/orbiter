// factor_cyclotomic.cpp
//
// Anton Betten
// June 2007

#include "orbiter.h"

using namespace std;


using namespace orbiter;

void print_usage();


void print_usage()
{
	cout << "usage: factor_cyclotomic [options] "
			"n q d a_d a_d-1 ... a_0" << endl;
	cout << "where options can be:" << endl;
	cout << "-v  <n>                "
			": verbose level <n>" << endl;
	cout << "-poly  <m>             "
			": use polynomial <m> to create the field GF(q)" << endl;
}

int main(int argc, char **argv)
{
	//int t0 = os_ticks();
	int verbose_level = 0;
	int f_poly = FALSE;
	char *poly = NULL;
	int i, j;
	int n, q, d;

	if (argc <= 4) {
		print_usage();
		exit(1);
		}
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-f_poly " << poly << endl;
			}
		else if (argv[i][0] != '-')
			break;
		}
	
	n = atoi(argv[i]);
	q = atoi(argv[++i]);
	d = atoi(argv[++i]);
	int *coeffs;
	
	coeffs = NEW_int(d + 1);
	for (j = d; j >= 0; j--) {
		coeffs[j] = atoi(argv[++i]);
		}
	
	algebra_global AG;

	AG.factor_cyclotomic(n, q, d, coeffs, f_poly, poly, verbose_level);
	
	FREE_int(coeffs);
}

