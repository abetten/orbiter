/*
 * surfaces_arc_lifting_main.cpp
 *
 *  Created on: Jan 9, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


int t0 = 0;


int main(int argc, const char **argv);


int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int q = 0;
	int i;
	int f_linear4 = FALSE;
	linear_group_description *Descr4;
	linear_group *LG4;
	int f_linear3 = FALSE;
	linear_group_description *Descr3;
	linear_group *LG3;
	int f_draw_poset = FALSE;
	int f_draw_poset_full = FALSE;

	t0 = os_ticks();


	surface_domain *Surf;
	surface_with_action *Surf_A;
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	Surf = NEW_OBJECT(surface_domain);
	Surf_A = NEW_OBJECT(surface_with_action);


	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-verbose_level " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-linear4") == 0) {
			f_linear4 = TRUE;
			Descr4 = NEW_OBJECT(linear_group_description);
			i += Descr4->read_arguments(argc - (i - 1),
				argv + i, verbose_level);

			cout << "-linear4" << endl;
			}
		else if (strcmp(argv[i], "-linear3") == 0) {
			f_linear3 = TRUE;
			Descr3 = NEW_OBJECT(linear_group_description);
			i += Descr3->read_arguments(argc - (i - 1),
				argv + i, verbose_level);

			cout << "-linear3" << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset" << endl;
			}
		else if (strcmp(argv[i], "-draw_poset_full") == 0) {
			f_draw_poset_full = TRUE;
			cout << "-draw_poset_full" << endl;
			}
		}


	if (!f_linear4) {
		cout << "please use option -linear4 ..." << endl;
		exit(1);
		}

	if (!f_linear3) {
		cout << "please use option -linear3 ..." << endl;
		exit(1);
		}


	int f_v = (verbose_level >= 1);

	q = Descr4->input_q;
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	Descr4->F = F;


	int f_semilinear;

	if (is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}

	LG4 = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "surface_classify before LG4->init, "
				"creating the group" << endl;
		}

	LG4->init(Descr4, verbose_level - 1);

	if (f_v) {
		cout << "surface_classify after LG4->init" << endl;
		}

	if (Descr3->input_q != q) {
		cout << "the two groups need to have the same field" << endl;
		exit(1);
	}
	Descr3->F = F;
	LG3 = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "surface_classify before LG3->init, "
				"creating the group" << endl;
		}

	LG3->init(Descr3, verbose_level - 1);

	if (f_v) {
		cout << "surface_classify after LG3->init" << endl;
		}


	if (f_v) {
		cout << "surface_classify before Surf->init" << endl;
		}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "surface_classify after Surf->init" << endl;
		}



#if 0
	if (f_v) {
		cout << "before Surf->init_large_polynomial_domains" << endl;
		}
	Surf->init_large_polynomial_domains(0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf->init_large_polynomial_domains" << endl;
		}
#endif


	if (f_v) {
		cout << "before Surf_A->init" << endl;
		}
	Surf_A->init(Surf, f_semilinear, 0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf_A->init" << endl;
		}


#if 0
	if (f_v) {
		cout << "before Surf_A->Classify_trihedral_"
				"pairs->classify" << endl;
		}
	Surf_A->Classify_trihedral_pairs->classify(0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf_A->Classify_trihedral_"
				"pairs->classify" << endl;
		}
#endif


	surfaces_arc_lifting *SAL;

	SAL = NEW_OBJECT(surfaces_arc_lifting);

	if (f_v) {
		cout << "before SAL->init" << endl;
		}
	SAL->init(F, LG4, LG3,
			f_semilinear, Surf_A,
			argc, argv,
			verbose_level);
	if (f_v) {
		cout << "after SAL->init" << endl;
		}


	FREE_OBJECT(SAL);

	the_end(t0);
	//the_end_quietly(t0);

}


