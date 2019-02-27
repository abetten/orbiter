// sequences.C
// 
// Anton Betten
// August 6, 2015
//
// creades codes of length n over an alphabet of size q 
// with minimum distance at least d.
// Following an idea outlined in the paper by Kokkala and Oestergaard
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;

int distance(int n, int *seq1, int *seq2);


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i, j, N, s;
	int *Adj;
	int *seq1;
	int *seq2;
	int *seq3;
	int f_n = FALSE;
	int n; // sequence length
	int f_q = FALSE;
	int q; // alphabet size
	int f_d = FALSE;
	int d; // minimum distance
	int f_set = FALSE;
	int sz = 0;
	int set[1000];

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
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
			}
		else if (strcmp(argv[i], "-set") == 0) {
			f_set = TRUE;
			sz = atoi(argv[++i]);
			for (j = 0; j < sz; j++) {
				set[j] = atoi(argv[++i]);
				}
			cout << "-set " << sz;
			int_vec_print(cout, set, sz);
			cout << endl;
			}
		}

	if (!f_n) {
		cout << "Please use option -n <n>" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "Please use option -q <q>" << endl;
		exit(1);
		}
	if (!f_d) {
		cout << "Please use option -d <d>" << endl;
		exit(1);
		}
#if 0
	if (!f_set) {
		cout << "Please use option -set <sz> <s_1> ... <s_sz>" << endl;
		exit(1);
		}
#endif

	N = i_power_j(q, n);
	

	Adj = NEW_int(N * N);
	int_vec_zero(Adj, N * N);

	seq1 = NEW_int(n);
	seq2 = NEW_int(n);
	seq3 = NEW_int(n);
	


	cout << "There are " << N << " words of length " << n << " over an alphabet of size " << q << ":" << endl;
	for (i = 0; i < N; i++) {
		AG_element_unrank(q, seq1, 1, n, i);
		cout << i << " : ";
		int_vec_print(cout, seq1, n);
		cout << endl;
		}

	for (i = 0; i < N; i++) {
		AG_element_unrank(q, seq1, 1, n, i);
		for (j = i + 1; j < N; j++) {
			AG_element_unrank(q, seq2, 1, n, j);

			s = distance(n, seq1, seq2);
			if (s >= d) {
				Adj[i * N + j] = 1;
				Adj[j * N + 1] = 1;
				}
			}
		}

	colored_graph *CG;
	char fname[1000];

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency_no_colors(N, Adj, verbose_level);

	sprintf(fname, "Sequences_%d_%d_%d.colored_graph", n, q, d);

	CG->save(fname, verbose_level);

	FREE_OBJECT(CG);



	if (f_set) {
		cout << "The set of size " << sz << " is ";
		int_vec_print(cout, set, sz);
		cout << endl;


		int a, b;


		for (i = 0; i < sz; i++) {
			a = set[i];
			AG_element_unrank(q, seq1, 1, n, a);
			cout << i << " : " << a << " : ";
			int_vec_print(cout, seq1, n);
			cout << endl;
			}

		int *D, *Adj2;
		
		D = NEW_int(sz * sz);
		Adj2 = NEW_int(sz * sz);
		int_vec_zero(D, sz * sz);
		int_vec_zero(Adj2, sz * sz);
		
		for (i = 0; i < sz; i++) {
			a = set[i];
			AG_element_unrank(q, seq1, 1, n, a);
			for (j = 0; j < sz; j++) {
				if (j == i) {
					continue;
					}
				b = set[j];
				AG_element_unrank(q, seq2, 1, n, b);

				s = distance(n, seq1, seq2);
				D[i * sz + j] = s;
				}
			}
		cout << "The distance matrix is:" << endl;
		int_matrix_print(D, sz, sz);
		cout << endl;

		for (i = 0; i < sz; i++) {
			for (j = 0; j < sz; j++) {
				if (j == i) {
					continue;
					}
				if (D[i * sz + j] > d) {
					Adj2[i * sz + j] = 1;
					}
				}
			}

		cout << "The adjacency matrix is:" << endl;
		int_matrix_print(Adj2, sz, sz);
		cout << endl;


		action *Aut2;
		nauty_interface Nauty;

		Aut2 = Nauty.create_automorphism_group_of_graph(Adj2, sz, verbose_level);
		cout << "The automorphism group of the distance graph has order ";
		longinteger_object ago2;
		Aut2->group_order(ago2);
		cout << ago2 << endl;

		colored_graph *CG;
		char fname[1000];

		CG = NEW_OBJECT(colored_graph);
		CG->init_adjacency_no_colors(sz, Adj2, verbose_level);

		sprintf(fname, "Sequences_%d_%d_%d_set_%d.colored_graph", n, q, d, sz);

		CG->save(fname, verbose_level);

		FREE_OBJECT(CG);


		}




	FREE_int(Adj);
	FREE_int(seq1);
	FREE_int(seq2);
	FREE_int(seq3);
}

int distance(int n, int *seq1, int *seq2)
{
	int h, s;
	
	s = 0;
	for (h = 0; h < n; h++) {
		if (seq1[h] != seq2[h]) {
			s++;
			}
		}
	return s;
}

