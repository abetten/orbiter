// three_squares.cpp
//
// Anton Betten
// Nov 23, 2018

#include "orbiter.h"

using namespace std;


using namespace orbiter;


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_n = FALSE;
	int n;
	//int *set;
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
	//set = NEW_int(k);

	N = int_n_choose_k(n, k);


#if 0
	for (h = 0; h < N; h++) {
		unrank_k_subset(h, set, n, k);
		a = squares[set[0] + 1] + squares[set[1] + 1] + squares[set[2] + 1];
		b = sqrt(a);
		if (b * b == a) {
			cout << set[0] + 1 << "^2 + " << set[1] + 1 << "^2 + "
					<< set[2] + 1 << "=^2 = "  << "=" << b*b << b << "^2" << endl;
		}
	}
#else
	int i1, i2, i3;

	for (i1 = 1; i1 <= n; i1++) {
		for (i2 = i1; i2 <= n; i2++) {
			for (i3 = i2; i3 <= n; i3++) {
				a = squares[i1] + squares[i2] + squares[i3];
				b = sqrt(a);
				if (b * b == a) {
					cout << i1 << "^2 + " << i2 << "^2 + "
							<< i3 << "^2 = " << a << " = " << b << "^2" << endl;
				}
			}
		}
	}
#endif
	//FREE_int(set);
}
