// matrix_rank.C
// 
// Anton Betten
//
// Sept 29, 2015
//
//


#include "orbiter.h"


// global data:

INT t0; // the system time when the program started


void count(INT n, INT k, finite_field *F, INT verbose_level);



int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_n = FALSE;
	INT n = 0;
	INT f_k = FALSE;
	INT k = 0;
	INT f_q = FALSE;
	INT q = 0;
	
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

	F = new finite_field;

	F->init(q, 0);


	count(n, k, F, verbose_level);


	delete F;
	
	the_end(t0);
}

void count(INT n, INT k, finite_field *F, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT q, h;
	INT kn = k * n;
	INT m, N, r;
	INT *M;
	INT *Rk;

	if (f_v) {
		cout << "count" << endl;
		}
	
	m = MAXIMUM(k, n);
	q = F->q;
	N = i_power_j(q, kn);
	M = NEW_INT(kn);
	Rk = NEW_INT(m + 1);
	INT_vec_zero(Rk, m + 1);

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


