/*
 * factor_poset.cpp
 *
 *  Created on: Feb 14, 2026
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace graph_theory {


factor_poset::factor_poset()
{
	Record_birth();

	nb_layers = 0;
	Nb_elements = NULL;
	Fst = NULL;
	Nb_orbits = NULL;
	Fst_element_per_orbit = NULL;
	Orbit_len = NULL;

	LG = NULL;

}


factor_poset::~factor_poset()
{
	Record_death();

	if (Nb_elements) {
		FREE_int(Nb_elements);
	}
#if 0
	if (Nb_orbits) {
		FREE_int(Nb_orbits);
	}
#endif
	if (Fst) {
		FREE_int(Fst);
	}
	if (Fst_element_per_orbit) {
		int i;
		for (i = 0; i < nb_layers; i++) {
			FREE_int(Fst_element_per_orbit[i]);
		}
		FREE_pint(Fst_element_per_orbit);
	}

	if (Nb_orbits) {
		FREE_int(Nb_orbits);
	}
	if (Orbit_len) {
		int i;
		for (i = 0; i < nb_layers; i++) {
			FREE_int(Orbit_len[i]);
		}
		FREE_pint(Orbit_len);
	}

	if (LG) {
		FREE_OBJECT(LG);
	}

}

void factor_poset::init(
		int depth,
		int *Nb_orbits,
		int **Orbit_len,
		int data1,
		double x_stretch,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "factor_poset::init" << endl;
	}

	int i, j;

	nb_layers = depth + 1;
	if (f_v) {
		cout << "factor_poset::init nb_layers = " << nb_layers << endl;
		cout << "factor_poset::init x_stretch = " << x_stretch << endl;
	}

	Nb_elements = NEW_int(nb_layers);
	factor_poset::Nb_orbits = Nb_orbits; // = NEW_int(nb_layers);

	Fst = NEW_int(nb_layers + 1);
	Fst_element_per_orbit = NEW_pint(nb_layers);
	factor_poset::Orbit_len = Orbit_len; // = NEW_pint(nb_layers);

	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {

		//Nb_orbits[i] = nb_orbits_at_level(i);
		Fst_element_per_orbit[i] = NEW_int(Nb_orbits[i] + 1);
		//Orbit_len[i] = NEW_int(Nb_orbits[i]);
		Nb_elements[i] = 0;

		Fst_element_per_orbit[i][0] = 0;

		for (j = 0; j < Nb_orbits[i]; j++) {

			//Orbit_len[i][j] = orbit_length_as_int(j, i);

			Nb_elements[i] += Orbit_len[i][j];

			Fst_element_per_orbit[i][j + 1] =
					Fst_element_per_orbit[i][j] + Orbit_len[i][j];
		}
		Fst[i + 1] = Fst[i] + Nb_elements[i];
	}


	int lvl;


	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);
	LG->add_data1(data1, 0/*verbose_level*/);

	if (f_v) {
		cout << "poset_classification::make_full_poset_graph "
				"before LG->init" << endl;
		cout << "nb_layers=" << nb_layers << endl;
		for (lvl = 0; lvl < depth; lvl++) {
			cout << "Nb_elements[" << lvl << "]=" << Nb_elements[lvl] << endl;
		}
	}
	string dummy;
	dummy.assign("");
	LG->init(nb_layers, Nb_elements, dummy, verbose_level);

	if (f_v) {
		cout << "poset_classification::make_full_poset_graph "
				"after LG->init" << endl;
	}
	if (f_v) {
		cout << "poset_classification::make_full_poset_graph "
				"before LG->place_with_grouping" << endl;
	}
	LG->place_with_grouping(Orbit_len, Nb_orbits, x_stretch, verbose_level);
	//LG->place(verbose_level);
	if (f_v) {
		cout << "poset_classification::make_full_poset_graph "
				"after LG->place" << endl;
	}



	if (f_v) {
		cout << "factor_poset::init done" << endl;
	}
}



}}}}


