// test_arc.C
//
// Anton Betten
// July 24, 2017

#include "orbiter.h"


// global data:

INT t0; // the system time when the program started

int main(int argc, char **argv);

int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT f_k = FALSE;
	INT k = 0;
	INT f_q = FALSE;
	INT q = 0;
	INT f_poly = FALSE;
	const BYTE *poly = NULL;
	INT f_arc = FALSE;
	const BYTE *arc_text = NULL;
	INT i;

	t0 = os_ticks();
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

	INT f_v = (verbose_level >= 1);
	
	cout << "k=" << k << endl;
	cout << "q=" << q << endl;
	cout << "poly=";
	if (poly) {
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
	F = new finite_field;
	F->init_override_polynomial(q, poly, verbose_level);


	if (f_v) {
		cout << "creating projective space PG(" << k - 1 << ", " << q << ")" << endl;
		}


	P = new projective_space;

	if (f_v) {
		cout << "before P->init" << endl;
		}
	P->init(k - 1, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /*verbose_level - 2*/);


	

	if (f_arc) {
		INT *the_arc;
		INT the_arc_sz;
		INT *Coord;
		INT a, j, nb_c, rk;
		INT *Roth_Lempel;
		
		INT_vec_scan(arc_text, the_arc, the_arc_sz);
		cout << "input arc = ";
		INT_vec_print(cout, the_arc, the_arc_sz);
		cout << endl;

		
		Coord = NEW_INT(the_arc_sz * k);
		for (i = 0; i < the_arc_sz; i++) {
			a = the_arc[i];
			F->projective_point_unrank(k - 1, Coord + i * k, a);
			}
		for (i = 0; i < the_arc_sz; i++) {
			cout << the_arc[i] << " : ";
			INT_vec_print(cout, Coord + i * k, k);
			cout << endl;
			}

		nb_c = the_arc_sz - k;
		Roth_Lempel = NEW_INT(k * nb_c);
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
		print_integer_matrix_with_standard_labels(cout, Roth_Lempel, k, nb_c, TRUE /* f_tex*/);
		rk = F->Gauss_easy(Roth_Lempel, k, nb_c);
		cout << "The matrix has rank " << rk << endl;
		

		FREE_INT(Coord);
		}

	delete P;
	delete F;

	the_end(t0);
}


