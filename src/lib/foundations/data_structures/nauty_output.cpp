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


nauty_output::nauty_output()
{
	N = 0;
	Aut = NULL;
	Aut_counter = 0;
	Base = NULL;
	Base_length = 0;
	Base_lint = NULL;
	Transversal_length = NULL;
	//longinteger_object Ago;
}

nauty_output::~nauty_output()
{
	if (Aut) {
		FREE_int(Aut);
	}
	if (Base) {
		FREE_int(Base);
	}
	if (Base_lint) {
		FREE_lint(Base_lint);
	}
	if (Transversal_length) {
		FREE_int(Transversal_length);
	}
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
}



}}
