/*
 * nauty_output.cpp
 *
 *  Created on: Aug 21, 2021
 *      Author: betten
 */


#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {
namespace data_structures {


nauty_output::nauty_output()
{
	N = 0;
	Aut = NULL;
	Aut_counter = 0;
	Base = NULL;
	Base_length = 0;
	Base_lint = NULL;
	Transversal_length = NULL;
	Ago = NULL;

	canonical_labeling = NULL;

	nb_firstpathnode = 0;
	nb_othernode = 0;
	nb_processnode = 0;
	nb_firstterminal = 0;
}

nauty_output::~nauty_output()
{
	//cout << "nauty_output::~nauty_output" << endl;
	if (Aut) {
		//cout << "before FREE_int(Aut);" << endl;
		FREE_int(Aut);
	}
	if (Base) {
		//cout << "before FREE_int(Base);" << endl;
		FREE_int(Base);
	}
	if (Base_lint) {
		//cout << "before FREE_lint(Base_lint);" << endl;
		FREE_lint(Base_lint);
	}
	if (Transversal_length) {
		//cout << "before FREE_int(Transversal_length);" << endl;
		FREE_int(Transversal_length);
	}
	if (Ago) {
		//cout << "before FREE_OBJECT(Ago);" << endl;
		FREE_OBJECT(Ago);
	}
	if (canonical_labeling) {
		//cout << "before FREE_int(canonical_labeling);" << endl;
		FREE_int(canonical_labeling);
	}
	//cout << "nauty_output::~nauty_output done" << endl;
}

void nauty_output::allocate(int N, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "nauty_output::allocate" << endl;
	}
	nauty_output::N = N;

	Aut = NEW_int(N * N);
	Base = NEW_int(N);
	Base_lint = NEW_lint(N);
	Transversal_length = NEW_int(N);
	Ago = NEW_OBJECT(ring_theory::longinteger_object);
	canonical_labeling = NEW_int(N);

	int i;

	for (i = 0; i < N; i++) {
		canonical_labeling[i] = i;
	}
}

void nauty_output::print()
{
		cout << "nauty_output::print" << endl;
		cout << "N=" << N << endl;
}

void nauty_output::print_stats()
{
	cout << "nb_backtrack1 = " << nb_firstpathnode << endl;
	cout << "nb_backtrack2 = " << nb_othernode << endl;
	cout << "nb_backtrack3 = " << nb_processnode << endl;
	cout << "nb_backtrack4 = " << nb_firstterminal << endl;

}

int nauty_output::belong_to_the_same_orbit(int a, int b, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "nauty_output::belong_to_the_same_orbit" << endl;
	}

	int *prev;
	int *orbit;
	int *Q;
	int Q_len;
	int orbit_len;
	int c, d;
	int i, j;
	int nb_gen;

	nb_gen = Aut_counter;
	prev = NEW_int(N);
	orbit = NEW_int(N);
	Q = NEW_int(N);
	Q[0] = a;
	Q_len = 1;
	orbit_len = 0;
	prev[a] = a;

	for (i = 0; i < N; i++) {
		prev[i] = -1;
	}

	while (Q_len) {
		c = Q[0];
		for (i = 1; i < Q_len; i++) {
			Q[i - 1] = Q[i];
		}
		Q_len--;
		orbit[orbit_len++] = c;
		for (j = 0; j < nb_gen; j++) {
			d = Aut[j * N + c];
			if (prev[d] == -1) {
				prev[d] = c;
				Q[Q_len++] = d;
				if (d == b) {
					FREE_int(prev);
					FREE_int(orbit);
					FREE_int(Q);
					return TRUE;
				}
			}
		}
	}

	if (f_v) {
		cout << "nauty_output::belong_to_the_same_orbit done" << endl;
	}
	return FALSE;
}




}}}

