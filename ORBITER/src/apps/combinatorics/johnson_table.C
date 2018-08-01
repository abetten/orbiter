// johnson.C
// 
// Anton Betten
// January 20, 2015

#include "orbiter.h"


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i, j, N, sz;
	INT *Adj;
	INT *set1;
	INT *set2;
	INT *set3;
	INT f_n_max = FALSE;
	INT n_max;
	INT n, k, n2, s;

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
				
				N = INT_n_choose_k(n, k);

				Adj = NEW_INT(N * N);
				INT_vec_zero(Adj, N * N);

				set1 = NEW_INT(k);
				set2 = NEW_INT(k);
				set3 = NEW_INT(k);
	
				for (i = 0; i < N; i++) {
					unrank_k_subset(i, set1, n, k);
					for (j = i + 1; j < N; j++) {
						unrank_k_subset(j, set2, n, k);

						INT_vec_intersect_sorted_vectors(set1, k, set2, k, set3, sz);
						if (sz == s) {
							Adj[i * N + j] = 1;
							Adj[j * N + 1] = 1;
							}
						}
					}

				action *Aut;
				longinteger_object ago;
				longinteger_object a;
				INT b;
				longinteger_domain D;

				Aut = create_automorphism_group_of_graph(Adj, N, 0/*verbose_level*/);
				Aut->group_order(ago);
				

				D.factorial(a, n);
				cout << "ago = " << ago << endl;
				cout << "n factorial = " << a << endl;

				b = D.quotient_as_INT(ago, a);

				cout << "n=" << n << " k=" << k << " s=" << s << " ago_quotient = " << b << endl;

				delete Aut;
				FREE_INT(Adj);
				FREE_INT(set1);
				FREE_INT(set2);
				FREE_INT(set3);
				
				} // next s
			} // next k

		} // next n

#if 0
	colored_graph *CG;
	BYTE fname[1000];

	CG = new colored_graph;
	CG->init_adjacency_no_colors(N, Adj, verbose_level);

	sprintf(fname, "Johnson_%ld_%ld_%ld.colored_graph", n, k, s);

	CG->save(fname, verbose_level);

	delete CG;
#endif

}

