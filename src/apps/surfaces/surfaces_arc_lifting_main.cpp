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
	int f_draw_poset_of_six_arcs = FALSE;
	//int f_draw_poset_full = FALSE;
	int f_report = FALSE;
	os_interface Os;

	t0 = Os.os_ticks();


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
		else if (strcmp(argv[i], "-draw_poset_of_six_arcs") == 0) {
			f_draw_poset_of_six_arcs = TRUE;
			cout << "-draw_poset_of_six_arcs" << endl;
		}
#if 0
		else if (strcmp(argv[i], "-draw_poset_full") == 0) {
			f_draw_poset_full = TRUE;
			cout << "-draw_poset_full" << endl;
		}
#endif
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report" << endl;
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
	if (Descr4->f_override_polynomial) {
		cout << "creating finite field of order q=" << Descr4->input_q
				<< " using override polynomial " << Descr4->override_polynomial << endl;
		F->init_override_polynomial(Descr4->input_q,
				Descr4->override_polynomial, verbose_level);
	}
	else {
		cout << "creating finite field of order q=" << Descr4->input_q << endl;
		F->init(Descr4->input_q, 0);
	}

	Descr4->F = F;


#if 0
	int f_semilinear;
	number_theory_domain NT;

	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}
#endif

	LG4 = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "surface_classify before LG4->init, "
				"creating the group" << endl;
	}

	LG4->init(Descr4, verbose_level - 10);

	if (f_v) {
		cout << "surface_classify after LG4->init" << endl;
	}

	int f_semilinear4;


	f_semilinear4 = LG4->A_linear->is_semilinear_matrix_group();
	if (f_v) {
		cout << "surface_classify f_semilinear4 = " << f_semilinear4 << endl;
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

	LG3->init(Descr3, verbose_level - 10);

	if (f_v) {
		cout << "surface_classify after LG3->init" << endl;
	}

	int f_semilinear3;


	f_semilinear3 = LG3->A_linear->is_semilinear_matrix_group();
	if (f_v) {
		cout << "surface_classify f_semilinear3 = " << f_semilinear3 << endl;
	}

	if (f_semilinear3 != f_semilinear4) {
		cout << "the grups must both be semilinear or both not be semilinear" << endl;
		exit(1);
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
	Surf_A->init(Surf, LG4, 0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf_A->init" << endl;
	}


#if 0
	Classify_trihedral_pairs = NEW_OBJECT(classify_trihedral_pairs);
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
	group_theoretic_activity *GTA;

	SAL = NEW_OBJECT(surfaces_arc_lifting);
	GTA = NEW_OBJECT(group_theoretic_activity);

	if (f_v) {
		cout << "before SAL->init" << endl;
	}

	SAL->init(GTA, F, LG4, LG3,
			f_semilinear4, Surf_A,
			verbose_level);
	if (f_v) {
		cout << "after SAL->init" << endl;
	}

	if (f_draw_poset_of_six_arcs) {
		SAL->draw_poset_of_six_arcs(verbose_level);
	}
	if (f_report) {
		SAL->report(verbose_level);
	}

	FREE_OBJECT(SAL);

	the_end(t0);
	//the_end_quietly(t0);

}


