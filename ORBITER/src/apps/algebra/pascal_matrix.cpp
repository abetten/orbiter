// pascal_matrix.C
// 
// Anton Betten
// January 12, 2018
//
//
// 
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;

void do_it(int q, int k, int verbose_level);
int entry_ij(finite_field *F, int m, int n);

int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_q = FALSE;
	int q = 0;
	int f_k = FALSE;
	int k = 0;
		
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
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		}
	
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}
	if (!f_k) {
		cout << "please use option -k <k>" << endl;
		exit(1);
		}

	do_it(q, k, verbose_level);
}

void do_it(int q, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *P;
	int i, j;

	if (f_v) {
		cout << "do_it() q=" << q << " k=" << k << endl;
		}
	finite_field *F;

	F = NEW_OBJECT(finite_field);

	cout << "initializing finite field of order " << q << ":" << endl;
	F->init(q, 0);
	
	P = NEW_int(k * q);
	for (i = 0; i < k; i++) {
		for (j = 0; j < q; j++) {
			P[i * q + j] = entry_ij(F, i, j);
			}
		}
	cout << "P_{" << k << "," << q << "}=" << endl;
	int_matrix_print(P, k, q);

	int *set;
	int *v;
	int a;

	set = NEW_int(q + 1);
	v = NEW_int(k);

	for (j = 0; j < q; j++) {
		for (i = 0; i < k; i++) {
			v[i] = P[i * q + j];
			}
		F->PG_element_rank_modified(v, 1, k, a);

		set[j] = a;
		}
	int_vec_zero(v, k);
	v[k - 1] = 1;
	F->PG_element_rank_modified(v, 1, k, a);
	set[q] = a;

	char fname[1000];
	//char label[1000];
	file_io Fio;

	//sprintf(label, "Pascal_%d_%d", k, q);
	sprintf(fname, "Pascal_%d_%d.csv", k, q);

	Fio.int_matrix_write_csv(fname, set, 1, q + 1);
	//write_set_to_file(fname, set, q, 0 /*verbose_level*/);
	//int_vec_write_csv(set, q, fname, label);
	cout << "Written file " << fname
			<< " of size " << Fio.file_size(fname) << endl;
}

int entry_ij(finite_field *F, int m, int n)
{
	int a = 1;
	int top, bot, b, i;


	for (i = 1; i <= m; i++) {
		top = F->add(n, F->negate(i - 1));
		bot = i;
		b = F->mult(top, F->inverse(bot));
		a = F->mult(a, b);
		}
	return a;
}


