// johnson.C
// 
// Anton Betten
// January 20, 2015

#include "orbiter.h"

using namespace std;


using namespace orbiter;

int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i, j, N, sz;
	int *Adj;
	int *set1;
	int *set2;
	int *set3;
	int f_n_max = FALSE;
	int n_max;
	int n, k, n2, s;
	combinatorics_domain Combi;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n_max") == 0) {
			f_n_max = TRUE;
			n_max = atoi(argv[++i]);
			cout << "-n_max " << n_max << endl;
			}
		}

	if (!f_n_max) {
		cout << "Please use option -n_max <n_max>" << endl;
		exit(1);
		}

	
	for (n = 3; n <= n_max; n++) {

		cout << "n=" << n << endl;
		n2 = n >> 1;
		
		for (k = 2; k <= n2; k++) {

			cout << "n=" << n << " k=" << k << endl;

			for (s = 0; s < k; s++) {
			
				cout << "n=" << n << " k=" << k << " s=" << s << endl;
				
				N = Combi.int_n_choose_k(n, k);

				Adj = NEW_int(N * N);
				int_vec_zero(Adj, N * N);

				set1 = NEW_int(k);
				set2 = NEW_int(k);
				set3 = NEW_int(k);
	
				for (i = 0; i < N; i++) {
					Combi.unrank_k_subset(i, set1, n, k);
					for (j = i + 1; j < N; j++) {
						Combi.unrank_k_subset(j, set2, n, k);

						int_vec_intersect_sorted_vectors(
								set1, k, set2, k, set3, sz);
						if (sz == s) {
							Adj[i * N + j] = 1;
							Adj[j * N + 1] = 1;
							}
						}
					}

				action *Aut;
				longinteger_object ago;
				longinteger_object a;
				int b;
				longinteger_domain D;
				nauty_interface Nauty;

				Aut = Nauty.create_automorphism_group_of_graph(
						Adj, N, 0/*verbose_level*/);
				Aut->group_order(ago);
				

				D.factorial(a, n);
				cout << "ago = " << ago << endl;
				cout << "n factorial = " << a << endl;

				b = D.quotient_as_int(ago, a);

				cout << "n=" << n << " k=" << k << " s=" << s
						<< " ago_quotient = " << b << endl;

				delete Aut;
				FREE_int(Adj);
				FREE_int(set1);
				FREE_int(set2);
				FREE_int(set3);
				
				} // next s
			} // next k

		} // next n

#if 0
	colored_graph *CG;
	char fname[1000];

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency_no_colors(N, Adj, verbose_level);

	sprintf(fname, "Johnson_%d_%d_%d.colored_graph", n, k, s);

	CG->save(fname, verbose_level);

	FREE_OBJECT(CG);
#endif

}

