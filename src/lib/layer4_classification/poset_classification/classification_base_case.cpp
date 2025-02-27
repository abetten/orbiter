/*
 * classification_base_case.cpp
 *
 *  Created on: Jun 2, 2020
 *      Author: betten
 */



#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


classification_base_case::classification_base_case()
{
	Record_birth();
		Poset = NULL;

		size = 0;
		orbit_rep = NULL; // [size]
		Stab_gens = NULL;
		live_points = NULL;
		nb_live_points = 0;
		recognition_function_data = NULL;
		recognition_function = NULL;
		Elt = NULL;
}

classification_base_case::~classification_base_case()
{
	Record_death();
	if (Elt) {
		FREE_int(Elt);
	}
}

void classification_base_case::init(
		poset_with_group_action *Poset,
		int size, long int *orbit_rep,
		long int *live_points, int nb_live_points,
		groups::strong_generators *Stab_gens,
		void *recognition_function_data,
		int (*recognition_function)(long int *Set, int len,
				int *Elt, void *data, int verbose_level),
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_base_case::init" << endl;
		cout << "classification_base_case::init size=" << size << endl;
		cout << "classification_base_case::init nb_live_points=" << nb_live_points << endl;
	}

	classification_base_case::Poset = Poset;

	classification_base_case::size = size;
	classification_base_case::orbit_rep = orbit_rep;
	classification_base_case::Stab_gens = Stab_gens;
	classification_base_case::live_points = live_points;
	classification_base_case::nb_live_points = nb_live_points;
	classification_base_case::recognition_function_data = recognition_function_data;
	classification_base_case::recognition_function = recognition_function;
	Elt = NEW_int(Poset->A->elt_size_in_int);


	if (f_v) {
		cout << "classification_base_case::init done" << endl;
	}
}


int classification_base_case::invoke_recognition(
		long int *Set, int len,
			int *Elt, int verbose_level)
{
	return (*recognition_function)(Set, len, Elt, recognition_function_data, verbose_level);
}


}}}



