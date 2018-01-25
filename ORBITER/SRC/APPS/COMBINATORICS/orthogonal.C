// orthogonal.C
// 
// Anton Betten
// November 22, 2015

#include "orbiter.h"


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i, j;
	INT *Adj;
	INT f_epsilon = FALSE;
	INT epsilon;
	INT f_d = FALSE;
	INT d;
	INT f_q = FALSE;
	INT q;
	INT f_list_points = FALSE;
	finite_field *F;
	INT n, N, a, nb_e, nb_inc;
	INT c1 = 0, c2 = 0, c3 = 0;
	INT *v, *v2;
	INT *Gram; // Gram matrix

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

	v = new INT[d];
	v2 = new INT[d];
	Gram = new INT[d * d];
	
	cout << "epsilon=" << epsilon << " n=" << n << " q=" << q << endl;
	
	N = nb_pts_Qepsilon(epsilon, n, q);
	
	cout << "number of points = " << N << endl;
	
	F = new finite_field;
	
	F->init(q, verbose_level - 1);
	F->print(TRUE);
	
	if (epsilon == 0) {
		c1 = 1;
		}
	else if (epsilon == -1) {
		choose_anisotropic_form(*F, c1, c2, c3, verbose_level - 2);
		//cout << "incma.C: epsilon == -1, need irreducible polynomial" << endl;
		//exit(1);
		}
	Gram_matrix(*F, epsilon, n, c1, c2, c3, Gram);
	cout << "Gram matrix" << endl;
	print_integer_matrix_width(cout, Gram, d, d, d, 2);
	
	if (f_list_points) {
		for (i = 0; i < N; i++) {
			Q_epsilon_unrank(*F, v, 1, epsilon, n, c1, c2, c3, i);
			cout << i << " : ";
			INT_vec_print(cout, v, n + 1);
			j = Q_epsilon_rank(*F, v, 1, epsilon, n, c1, c2, c3);
			cout << " : " << j << endl;
		
			}
		}

	
	cout << "allocating adjacency matrix" << endl;
	Adj = NEW_INT(N * N);
	cout << "allocating adjacency matrix was successfull" << endl;
	nb_e = 0;
	nb_inc = 0;
	for (i = 0; i < N; i++) {
		//cout << i << " : ";
		Q_epsilon_unrank(*F, v, 1, epsilon, n, c1, c2, c3, i);
		for (j = i + 1; j < N; j++) {
			Q_epsilon_unrank(*F, v2, 1, epsilon, n, c1, c2, c3, j);
			a = evaluate_bilinear_form(*F, v, v2, n + 1, Gram);
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
	BYTE fname[1000];

	CG = new colored_graph;
	CG->init_adjacency_no_colors(N, Adj, verbose_level);

	sprintf(fname, "O_%ld_%ld_%ld.colored_graph", epsilon, d, q);

	CG->save(fname, verbose_level);

	delete CG;
	FREE_INT(Adj);
	FREE_INT(v);
	FREE_INT(v2);
	FREE_INT(Gram);
	delete F;
}


