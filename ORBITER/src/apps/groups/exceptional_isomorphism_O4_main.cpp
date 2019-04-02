/*
 * exceptional_isomorphism_O4_main.cpp
 *
 *  Created on: Mar 31, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

using namespace orbiter;

// global data:

int t0; // the system time when the program started

int main(int argc, char **argv);
void do_it(int q, int verbose_level);


int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_q = FALSE;
	int q = 0;

	t0 = os_ticks();

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		}
	if (!f_q) {
		cout << "please use -q <q>" << endl;
		exit(1);
		}
	do_it(q, verbose_level);

	the_end_quietly(t0);
}

void do_it(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	action *A5;
	action *A4;
	action *A2;
	int f_semilinear = FALSE;
	int f_basis = TRUE;
	int p, h, i, j;

	if (f_v) {
		cout << "do_it q=" << q << endl;
	}
	A5 = NEW_OBJECT(action);
	A4 = NEW_OBJECT(action);
	A2 = NEW_OBJECT(action);
	is_prime_power(q, p, h);
	if (h > 1)
		f_semilinear = TRUE;
	else
		f_semilinear = FALSE;


	F = NEW_OBJECT(finite_field);

	F->init(q, 0);

	if (f_v) {
		cout << "do_it before A5->init_orthogonal_group" << endl;
	}
	A5->init_orthogonal_group(0 /*epsilon*/, 5, F,
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level);

	if (f_v) {
		cout << "do_it before A4->init_orthogonal_group" << endl;
	}
	A4->init_orthogonal_group(1 /*epsilon*/, 4, F,
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level);


	vector_ge *nice_gens;
	nice_gens = NEW_OBJECT(vector_ge);

	if (f_v) {
		cout << "do_it before A2->init_projective_group" << endl;
	}
	A2->init_projective_group(2, F, f_semilinear,
		TRUE /*f_basis*/,
		nice_gens,
		verbose_level);


	if (!A5->f_has_strong_generators) {
		cout << "action A5 does not have strong generators" << endl;
		exit(1);
		}

	exceptional_isomorphism_O4 *Iso;

	Iso = NEW_OBJECT(exceptional_isomorphism_O4);

	if (f_v) {
		cout << "do_it before Iso->init" << endl;
	}
	Iso->init(F, A2, A4, A5, verbose_level);


	sims *G;

	G = A4->Sims;
	cout << "checking all group elements in O^+(4,q):" << endl;

	sims *G5;

	G5 = A5->Sims;

	longinteger_object go;
	int goi;
	longinteger_object go5;
	int go5i;
	int f_switch;
	int *E5;
	int *E4a;
	int *E4b;
	int *E2a;
	int *E2b;

	E5 = NEW_int(A5->elt_size_in_int);
	E4a = NEW_int(A4->elt_size_in_int);
	E4b = NEW_int(A4->elt_size_in_int);
	E2a = NEW_int(A2->elt_size_in_int);
	E2b = NEW_int(A2->elt_size_in_int);

	G->group_order(go);
	goi = go.as_int();
	G5->group_order(go5);
	go5i = go5.as_int();

	cout << "group order O^+(4," << q << ") is " << go << " as int " << goi << endl;
	cout << "group order O(5," << q << ") is " << go5 << " as int " << go5i << endl;


	for (i = 0; i < goi; i++) {
		cout << "i=" << i << " / " << goi << endl;
		G->element_unrank_int(i, E4a);

		cout << "E4a=" << endl;
		A4->element_print_quick(E4a, cout);


		Iso->apply_4_to_2(E4a, f_switch, E2a, E2b, 0 /*verbose_level*/);
		cout << "after isomorphism:" << endl;
		cout << "f_switch=" << f_switch << endl;
		cout << "E2a=" << endl;
		A2->element_print_quick(E2a, cout);
		cout << "E2b=" << endl;
		A2->element_print_quick(E2b, cout);

		Iso->apply_2_to_4(f_switch, E2a, E2b, E4b, 0 /*verbose_level*/);
		j = G->element_rank_int(E4b);
		cout << "rank returns j=" << j << endl;
		if (j != i) {
			cout << "j != i" << endl;
			cout << "i=" << i << endl;
			cout << "j=" << j << endl;
			exit(1);
		}
		Iso->apply_4_to_5(E4a, E5, 0 /*verbose_level*/);
		cout << "E5=" << endl;
		A5->element_print_quick(E5, cout);
		j = G5->element_rank_int(E5);
		cout << "rank in O(5,q)=" << j << endl;
	}

}

