/*
 * sylow_structure.cpp
 *
 *  Created on: Mar 24, 2019
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {

sylow_structure::sylow_structure()
{
	null();
}

sylow_structure::~sylow_structure()
{
	freeself();
}

void sylow_structure::null()
{
	primes = NULL;
	exponents = NULL;
	nb_primes = 0;
	S = NULL;
	Sub = NULL;
}

void sylow_structure::freeself()
{
	if (primes) {
		FREE_int(primes);
		}
	if (exponents) {
		FREE_int(exponents);
		}
	if (Sub) {
		FREE_OBJECTS(Sub);
		}
	null();
}

void sylow_structure::init(sims *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;

	if (f_v) {
		cout << "sylow_structure::init" << endl;
	}

	longinteger_domain D;
	int i;

	sylow_structure::S = S;
	S->group_order(go);

	D.factor(go, nb_primes,
			primes, exponents,
			verbose_level);

	if (f_v) {
		cout << "sylow_structure::init group order = " << go << " = ";
		for (i = 0; i < nb_primes; i++) {
			cout << primes[i] << "^" << exponents[i];
			if (i < nb_primes - 1) {
				cout << " * ";
			}
		}
		cout << endl;
	}

	Sub = NEW_OBJECTS(subgroup, nb_primes);

	for (idx = 0; idx < nb_primes; idx++) {
		strong_generators *SG;
		sims *P;

		P = NEW_OBJECT(sims);
		SG = NEW_OBJECT(strong_generators);
		if (f_v) {
			cout << "sylow_structure::init "
					"computing Sylow subgroup for prime "
					<< primes[idx] << ": " << endl;
		}
		S->sylow_subgroup(primes[idx], P, verbose_level);
		SG->init_from_sims(P, verbose_level);
		Sub[idx].init_from_sims(S, P, SG, verbose_level);
	}

	if (f_v) {
		cout << "sylow_structure::init done" << endl;
	}
}

void sylow_structure::report(ostream &ost)
{
	int idx;

	for (idx = 0; idx < nb_primes; idx++) {
		ost << "The " << primes[idx] << "-Sylow groups have order $"
				<< primes[idx] << "^{" << exponents[idx] << "}$\\\\" << endl;
	}
	for (idx = 0; idx < nb_primes; idx++) {
		ost << "One " << primes[idx] << "-Sylow group has "
				"the following generators:\\\\" << endl;
		Sub[idx].report(ost);
	}
}



}}

