/*
 * orbit_of_elements.cpp
 *
 *  Created on: Feb 25, 2025
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


orbit_of_elements::orbit_of_elements()
{
	Record_birth();

	Class = NULL;

	idx = 0;

	go_P = 0;

	Element = NULL;
	Element_rk = 0;
	Elements_P = NULL;
	Orbits_P = NULL;

	orbit_length = 0;
	Table_of_elements = NULL;
}


orbit_of_elements::~orbit_of_elements()
{
	Record_death();

	if (Element) {
		FREE_int(Element);
	}
	if (Elements_P) {
		FREE_OBJECT(Elements_P);
	}
	if (Orbits_P) {
		FREE_OBJECT(Orbits_P);
	}
	if (Table_of_elements) {
		FREE_lint(Table_of_elements);
	}
}

void orbit_of_elements::init(
		groups::any_group *Any_group,
		groups::sims *Sims_G,
		actions::action *A_conj,
		interfaces::conjugacy_classes_and_normalizers *Classes,
		int idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_elements::init" << endl;
	}

	orbit_of_elements::idx = idx;


	other::data_structures::sorting Sorting;


	//long int rk;

	//Sims_P = Classes->Conjugacy_class[idx]->gens->create_sims(verbose_level);
	go_P = 1; //Sims_P->group_order_lint();

	Element = NEW_int(Classes->A->elt_size_in_int);
	Classes->A->Group_element->element_move(
			Classes->Conjugacy_class[idx]->nice_gens->ith(0),
			Element, 0);
	Element_rk = Sims_G->element_rank_lint(Element);
	Elements_P = NEW_lint(go_P);

	Elements_P[0] = Element_rk;
	Sorting.lint_vec_heapsort(Elements_P, go_P);

	if (f_v) {
		cout << "orbit_of_elements::init "
				"before Any_group->A->create_induced_action_by_conjugation" << endl;
	}



	Orbits_P = NEW_OBJECT(orbits_schreier::orbit_of_sets);


	if (f_v) {
		cout << "orbit_of_elements::init "
				"before Orbits_P->init" << endl;
	}

	Orbits_P->init(
			Any_group->A,
			A_conj,
			Elements_P, go_P,
			Any_group->Subgroup_gens->gens,
			verbose_level - 4);

	if (f_v) {
		cout << "orbit_of_elements::init "
				"after Orbits_P->init" << endl;
	}


	int set_size;

	Orbits_P->get_table_of_orbits(
			Table_of_elements,
			orbit_length, set_size, verbose_level - 4);

	Sorting.lint_vec_heapsort(Table_of_elements, orbit_length);

	if (f_v) {
		cout << "orbit_of_elements::init "
				"table of elements:" << endl;
		Lint_vec_print(cout, Table_of_elements, orbit_length);
		cout << endl;
		cout << "orbit_of_elements::init "
				" idx=" << idx
				<< " number of elements = " << orbit_length << endl;
	}




	if (f_v) {
		cout << "orbit_of_elements::init done" << endl;
	}



}


}}}

