// ovoid.C
// 
// Anton Betten
// 3/19/2011
//
// 
//
// creates the elliptic quadric ovoid in PG(3,q)
//

#include "orbiter.h"





// global data:

INT t0; // the system time when the program started


void create_ovoid(finite_field *F, INT verbose_level);


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT q;
	INT f_poly = FALSE;
	char *poly = NULL;
	
	t0 = os_ticks();



 	

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		}
	
	finite_field *F;

	F = new finite_field;
	F->init_override_polynomial(q, poly, 0);
	create_ovoid(F, verbose_level);

	delete F;
	the_end(t0);
}

void create_ovoid(finite_field *F, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	INT n = 3, epsilon = -1;
	INT c1 = 1, c2 = 0, c3 = 0;
	INT N, i, j, d, h;
	INT *v, *w, *L;
	INT q = F->q;

	d = n + 1;
	P = new projective_space;

	
	P->init(n, F, 
		//TRUE /* f_init_group */, 
		//FALSE /* f_line_action */, 
		TRUE /* f_init_incidence_structure */, 
		//TRUE /* f_semilinear */, 
		//TRUE /* f_basis */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	N = nb_pts_Qepsilon(epsilon, n, q);

	v = NEW_INT(n + 1);
	w = NEW_INT(n + 1);
	L = NEW_INT(P->N_points);

	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	choose_anisotropic_form(*F, c1, c2, c3, verbose_level);
	for (i = 0; i < N; i++) {
		Q_epsilon_unrank(*F, v, 1, epsilon, n, c1, c2, c3, i);
		for (h = 0; h < d; h++) {
			w[h] = v[h];
			}
		j = P->rank_point(w);
		L[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			INT_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}

	cout << "list of points on the ovoid:" << endl;
	cout << N << endl;
	for (i = 0; i < N; i++) {
		cout << L[i] << " ";
		}
	cout << endl;

	char fname[1000];
	sprintf(fname, "ovoid_%ld.txt", q);
	write_set_to_file(fname, L, N, verbose_level);

	delete P;
	FREE_INT(v);
	FREE_INT(w);
	FREE_INT(L);
}


