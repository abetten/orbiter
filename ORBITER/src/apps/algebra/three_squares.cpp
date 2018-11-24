// three_squares.cpp
//
// Anton Betten
// Nov 23, 2018

#include "orbiter.h"


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i, h;
	int f_n = FALSE;
	int n;
	int *set;
	int N;

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
		}

	if (!f_n) {
		cout << "Please use option -n <n>" << endl;
		exit(1);
		}

	int *squares;
	int a, b;
	int k = 3;

	squares = NEW_int(n + 1);
	squares[0] = 0;
	for (i = 1; i <= n; i++) {
		squares[i] = i * i;
	}
	set = NEW_int(k);

	N = int_n_choose_k(n, k);


	for (h = 0; h < N; h++) {
		unrank_k_subset(h, set, n, k);
		a = squares[set[0] + 1] + squares[set[1] + 1] + squares[set[2] + 1];
		b = sqrt(a);
		if (b * b == a) {
			cout << set[0] + 1 << "^2 + " << set[1] + 1 << "^2 + "
					<< set[2] + 1 << "^2 = " << b << "^2" << endl;
		}
	}
	FREE_int(set);
}
