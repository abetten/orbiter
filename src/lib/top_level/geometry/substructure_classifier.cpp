/*
 * substructure_classifier.cpp
 *
 *  Created on: Jun 9, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace top_level {


substructure_classifier::substructure_classifier()
{
	//PA = NULL;
	substructure_size = 0;
	PC = NULL;
	Control = NULL;
	A = NULL;
	A2 = NULL;
	Poset = NULL;
	nb_orbits = 0;
}


substructure_classifier::~substructure_classifier()
{

}


void substructure_classifier::classify_substructures(
		action *A,
		action *A2,
		strong_generators *gens,
		int substructure_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "substructure_classifier::classify_substructures, substructure_size=" << substructure_size << endl;
		cout << "substructure_classifier::classify_substructures, action A=";
		A->print_info();
		cout << endl;
		cout << "substructure_classifier::classify_substructures, action A2=";
		A2->print_info();
		cout << endl;
		cout << "substructure_classifier::classify_substructures generators:" << endl;
		gens->print_generators_tex(cout);
	}

	//substructure_classifier::PA = PA;
	substructure_classifier::A = A;
	substructure_classifier::A2 = A2;
	substructure_classifier::substructure_size = substructure_size;

	Poset = NEW_OBJECT(poset_with_group_action);


	Control = NEW_OBJECT(poset_classification_control);

	Control->f_depth = TRUE;
	Control->depth = substructure_size;


	if (f_v) {
		cout << "substructure_classifier::classify_substructures control=" << endl;
		Control->print();
	}


	Poset->init_subset_lattice(A, A2,
			gens,
			verbose_level);

	if (f_v) {
		cout << "substructure_classifier::classify_substructures "
				"before Poset->orbits_on_k_sets_compute" << endl;
	}
	PC = Poset->orbits_on_k_sets_compute(
			Control,
			substructure_size,
			verbose_level);
	if (f_v) {
		cout << "substructure_classifier::classify_substructures "
				"after Poset->orbits_on_k_sets_compute" << endl;
	}

	nb_orbits = PC->nb_orbits_at_level(substructure_size);

	cout << "We found " << nb_orbits << " orbits at level " << substructure_size << ":" << endl;

	int j;

	for (j = 0; j < nb_orbits; j++) {


		strong_generators *Strong_gens;

		PC->get_stabilizer_generators(
				Strong_gens,
				substructure_size, j, 0 /* verbose_level*/);

		longinteger_object go;

		Strong_gens->group_order(go);

		FREE_OBJECT(Strong_gens);

		cout << j << " : " << go << endl;


	}

	if (f_v) {
		cout << "substructure_classifier::classify_substructures done" << endl;
	}



}

}}

