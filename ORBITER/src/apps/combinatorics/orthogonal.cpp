// orthogonal.cpp
// 
// Anton Betten
// November 22, 2015

#include "orbiter.h"

using namespace std;


using namespace orbiter;

int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i, j;
	int *Adj;
	int f_epsilon = FALSE;
	int epsilon;
	int f_d = FALSE;
	int d;
	int f_q = FALSE;
	int q;
	int f_list_points = FALSE;
	finite_field *F;
	int n, N, a, nb_e, nb_inc;
	int c1 = 0, c2 = 0, c3 = 0;
	int *v, *v2;
	int *Gram; // Gram matrix
	geometry_global Gg;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-epsilon") == 0) {
			f_epsilon = TRUE;
			epsilon = atoi(argv[++i]);
			cout << "-epsilon " << epsilon << endl;
			}
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-list_points") == 0) {
			f_list_points = TRUE;
			cout << "-list_points" << endl;
			}
		}

	if (!f_epsilon) {
		cout << "Please use option -epsilon <epsilon>" << endl;
		exit(1);
		}
	if (!f_d) {
		cout << "Please use option -d <d>" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "Please use option -q <q>" << endl;
		exit(1);
		}

	n = d - 1; // projective dimension

	v = NEW_int(d);
	v2 = NEW_int(d);
	Gram = NEW_int(d * d);
	
	cout << "epsilon=" << epsilon << " n=" << n << " q=" << q << endl;
	
	N = Gg.nb_pts_Qepsilon(epsilon, n, q);
	
	cout << "number of points = " << N << endl;
	
	F = NEW_OBJECT(finite_field);
	
	F->init(q, verbose_level - 1);
	F->print();
	
	if (epsilon == 0) {
		c1 = 1;
		}
	else if (epsilon == -1) {
		F->choose_anisotropic_form(c1, c2, c3, verbose_level - 2);
		//cout << "incma.cpp: epsilon == -1, need irreducible polynomial" << endl;
		//exit(1);
		}
	F->Gram_matrix(epsilon, n, c1, c2, c3, Gram);
	cout << "Gram matrix" << endl;
	print_integer_matrix_width(cout, Gram, d, d, d, 2);
	
	if (f_list_points) {
		for (i = 0; i < N; i++) {
			F->Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i, 0 /* verbose_level */);
			cout << i << " : ";
			int_vec_print(cout, v, n + 1);
			j = F->Q_epsilon_rank(v, 1, epsilon, n, c1, c2, c3, 0 /* verbose_level */);
			cout << " : " << j << endl;
		
			}
		}

	
	cout << "allocating adjacency matrix" << endl;
	Adj = NEW_int(N * N);
	cout << "allocating adjacency matrix was successful" << endl;
	nb_e = 0;
	nb_inc = 0;
	for (i = 0; i < N; i++) {
		//cout << i << " : ";
		F->Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i, 0 /* verbose_level */);
		for (j = i + 1; j < N; j++) {
			F->Q_epsilon_unrank(v2, 1, epsilon, n, c1, c2, c3, j, 0 /* verbose_level */);
			a = F->evaluate_bilinear_form(v, v2, n + 1, Gram);
			if (a == 0) {
				//cout << j << " ";
				//k = ij2k(i, j, N);
				//cout << k << ", ";
				nb_e++;
				//if ((nb_e % 50) == 0)
					//cout << endl;
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
				}
			else {
				Adj[i * N + j] = 0;
				Adj[j * N + i] = 0;
				; //cout << " 0";
				nb_inc++;
				}
			}
		//cout << endl;
		Adj[i * N + i] = 0;
		}
	cout << endl;
	cout << "The adjacency matrix of the collinearity graph has been computed" << endl;

	colored_graph *CG;
	char fname[1000];

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency_no_colors(N, Adj, verbose_level);

	sprintf(fname, "O_%d_%d_%d.colored_graph", epsilon, d, q);

	CG->save(fname, verbose_level);

	FREE_OBJECT(CG);
	FREE_int(Adj);
	FREE_int(v);
	FREE_int(v2);
	FREE_int(Gram);
	FREE_OBJECT(F);
}


