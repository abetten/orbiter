// test_arc.cpp
//
// Anton Betten
// July 24, 2017

#include "orbiter.h"

using namespace std;


using namespace orbiter;

// global data:

int t0; // the system time when the program started

int main(int argc, char **argv);

int main(int argc, char **argv)
{
	int verbose_level = 0;
	int f_k = FALSE;
	int k = 0;
	int f_q = FALSE;
	int q = 0;
	int f_poly = FALSE;
	const char *poly = NULL;
	int f_arc = FALSE;
	const char *arc_text = NULL;
	int i;
	os_interface Os;

	t0 = Os.os_ticks();
	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-arc") == 0) {
			f_arc = TRUE;
			arc_text = argv[++i];
			cout << "-arc " << arc_text << endl;
			}
		}
	
	if (!f_k) {
		cout << "please use option -k <k>" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}

	int f_v = (verbose_level >= 1);
	
	cout << "k=" << k << endl;
	cout << "q=" << q << endl;
	cout << "poly=";
	if (f_poly) {
		cout << poly;
		}
	else {
		cout << endl;
		}
	
	finite_field *F;
	projective_space *P;

	if (f_v) {
		cout << "creating finite field:" << endl;
		}
	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(q, poly, verbose_level);


	if (f_v) {
		cout << "creating projective space "
				"PG(" << k - 1 << ", " << q << ")" << endl;
		}


	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "before P->init" << endl;
		}
	P->init(k - 1, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /*verbose_level - 2*/);


	

	if (f_arc) {
		int *the_arc;
		int the_arc_sz;
		int *Coord;
		int a, j, nb_c, rk;
		int *Roth_Lempel;
		latex_interface L;
		
		int_vec_scan(arc_text, the_arc, the_arc_sz);
		cout << "input arc = ";
		int_vec_print(cout, the_arc, the_arc_sz);
		cout << endl;

		
		Coord = NEW_int(the_arc_sz * k);
		for (i = 0; i < the_arc_sz; i++) {
			a = the_arc[i];
			F->projective_point_unrank(k - 1, Coord + i * k, a);
			}
		for (i = 0; i < the_arc_sz; i++) {
			cout << the_arc[i] << " : ";
			int_vec_print(cout, Coord + i * k, k);
			cout << endl;
			}

		nb_c = the_arc_sz - k;
		Roth_Lempel = NEW_int(k * nb_c);
		for (i = k; i < the_arc_sz; i++) {
			for (j = 0; j < k; j++) {
				a = Coord[i * k + j];
				if (a == 0) {
					cout << "a is zero" << endl;
					exit(1);
					}
				Roth_Lempel[j * nb_c + i - k] = F->inverse(a);
				}
			}
		cout << "Roth_Lempel:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
				Roth_Lempel, k, nb_c, TRUE /* f_tex*/);
		rk = F->Gauss_easy(Roth_Lempel, k, nb_c);
		cout << "The matrix has rank " << rk << endl;
		

		FREE_int(Coord);
		}

	FREE_OBJECT(P);
	FREE_OBJECT(F);

	the_end(t0);
}


