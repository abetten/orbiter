// matrix_rank.C
// 
// Anton Betten
//
// Sept 29, 2015
//
//


#include "orbiter.h"

using namespace std;


using namespace orbiter;

// global data:

int t0; // the system time when the program started


void count(int n, int k, finite_field *F, int verbose_level);



int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_n = FALSE;
	int n = 0;
	int f_k = FALSE;
	int k = 0;
	int f_q = FALSE;
	int q = 0;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		}
	if (!f_k) {
		cout << "please use option -k <k>" << endl;
		exit(1);
		}
	if (!f_n) {
		cout << "please use option -n <n>" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}

	finite_field *F;

	F = NEW_OBJECT(finite_field);

	F->init(q, 0);


	count(n, k, F, verbose_level);


	FREE_OBJECT(F);
	
	the_end(t0);
}

void count(int n, int k, finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q, h;
	int kn = k * n;
	int m, N, r;
	int *M;
	int *Rk;

	if (f_v) {
		cout << "count" << endl;
		}
	
	m = MAXIMUM(k, n);
	q = F->q;
	N = i_power_j(q, kn);
	M = NEW_int(kn);
	Rk = NEW_int(m + 1);
	int_vec_zero(Rk, m + 1);

	for (h = 0; h < N; h++) {
		AG_element_unrank(q, M, 1, kn, h);
		r = F->rank_of_rectangular_matrix(M, k, n, 0 /* verbose_level */);
		Rk[r]++;
		}
	
	cout << "Rank distribution:" << endl;
	for (h = 0; h <= m; h++) {
		cout << h << " : " << Rk[h] << endl;
		}
	cout << "N=" << N << endl;

	if (f_v) {
		cout << "count done" << endl;
		}
	
}


