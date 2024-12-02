/*
 * sylow_structure.cpp
 *
 *  Created on: Mar 24, 2019
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {



sylow_structure::sylow_structure()
{
	Record_birth();
	primes = NULL;
	exponents = NULL;
	nb_primes = 0;
	S = NULL;
	Subgroup_lattice = NULL;
	Sub = NULL;
}

sylow_structure::~sylow_structure()
{
	Record_death();
	int verbose_level = 0;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sylow_structure::~sylow_structure" << endl;
	}
	if (primes) {
		FREE_int(primes);
	}
	if (exponents) {
		FREE_int(exponents);
	}
	if (Subgroup_lattice) {
		if (f_v) {
			cout << "sylow_structure::~sylow_structure before FREE_OBJECT(Subgroup_lattice)" << endl;
		}
		FREE_OBJECT(Subgroup_lattice);
		if (f_v) {
			cout << "sylow_structure::~sylow_structure after FREE_OBJECT(Subgroup_lattice)" << endl;
		}
	}
	if (Sub) {
		if (f_v) {
			cout << "sylow_structure::~sylow_structure before FREE_OBJECTS(Sub)" << endl;
		}
		FREE_OBJECTS(Sub);
		if (f_v) {
			cout << "sylow_structure::~sylow_structure after FREE_OBJECTS(Sub)" << endl;
		}
	}
	if (f_v) {
		cout << "sylow_structure::~sylow_structure done" << endl;
	}
}

void sylow_structure::init(
		sims *Sims,
		std::string &label_txt,
		std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;

	if (f_v) {
		cout << "sylow_structure::init" << endl;
	}

	algebra::ring_theory::longinteger_domain D;
	int i;

	sylow_structure::S = Sims;
	S->group_order(go);
	if (f_v) {
		cout << "sylow_structure::init "
				"group_order = " << go << endl;
	}

	if (f_v) {
		cout << "sylow_structure::init "
				"before D.factor" << endl;
	}
	D.factor(go, nb_primes,
			primes, exponents,
			verbose_level);
	if (f_v) {
		cout << "sylow_structure::init "
				"after D.factor" << endl;
	}

	if (f_v) {
		cout << "sylow_structure::init "
				"group order = " << go << " = ";
		for (i = 0; i < nb_primes; i++) {
			cout << primes[i] << "^" << exponents[i];
			if (i < nb_primes - 1) {
				cout << " * ";
			}
		}
		cout << endl;
	}

	groups::strong_generators *SG;

	SG = NEW_OBJECT(groups::strong_generators);

	if (f_v) {
		cout << "sylow_structure::init before SG->init_from_sims" << endl;
	}
	SG->init_from_sims(
			S, verbose_level - 2);
	if (f_v) {
		cout << "sylow_structure::init after SG->init_from_sims" << endl;
	}

	Subgroup_lattice = NEW_OBJECT(subgroup_lattice);

	if (f_v) {
		cout << "sylow_structure::init "
				"before Subgroup_lattice->init_basic" << endl;
	}
	Subgroup_lattice->init_basic(
			S->A, S,
			label_txt,
			label_tex,
			SG, //S->A->Strong_gens,
			verbose_level - 1);
	if (f_v) {
		cout << "sylow_structure::init "
				"after Subgroup_lattice->init_basic" << endl;
	}

	Sub = NEW_OBJECTS(subgroup, nb_primes);

	for (idx = 0; idx < nb_primes; idx++) {
		sims *P;

		P = NEW_OBJECT(sims);
		if (f_v) {
			cout << "sylow_structure::init "
					"computing Sylow subgroup for prime "
					<< primes[idx] << ": " << endl;
		}
		S->sylow_subgroup(
				primes[idx], P, verbose_level);
		if (f_v) {
			cout << "sylow_structure::init "
					"Sylow subgroup for prime "
					<< primes[idx] << " has order "
					<< P->group_order_lint() << endl;
		}
		if (f_v) {
			cout << "sylow_structure::init "
					"before Sub[idx].init_from_sims" << endl;
		}

		groups::strong_generators *SG1;

		SG1 = NEW_OBJECT(groups::strong_generators);

		if (f_v) {
			cout << "sylow_structure::init "
					"before SG1->init_from_sims" << endl;
		}
		SG1->init_from_sims(
				P, verbose_level - 2);
		if (f_v) {
			cout << "sylow_structure::init "
					"after SG1->init_from_sims" << endl;
		}

		Sub[idx].init_from_sims(
				Subgroup_lattice, P, SG1, verbose_level);
		if (f_v) {
			cout << "sylow_structure::init "
					"after Sub[idx].init_from_sims" << endl;
		}
		// don't free the objects SG1 and P, as they are now part of Sub[idx].
		//FREE_OBJECT(SG1);
		//FREE_OBJECT(P);
		if (f_v) {
			cout << "sylow_structure::init "
					"computing Sylow subgroup for prime "
					<< primes[idx] << " done" << endl;
		}

	}


	if (f_v) {
		cout << "sylow_structure::init done" << endl;
	}
}

void sylow_structure::report(
		std::ostream &ost)
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



}}}

