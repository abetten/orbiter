/*
 * group_theory_global.cpp
 *
 *  Created on: Sep 27, 2024
 *      Author: betten
 */






#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {


group_theory_global::group_theory_global()
{

}


group_theory_global::~group_theory_global()
{

}


void group_theory_global::strong_generators_conjugate_avGa(
		strong_generators *SG_in,
		int *Elt_a,
		strong_generators *&SG_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//actions::action *A;
	data_structures_groups::vector_ge *gens;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_avGa" << endl;
	}

	//A = SG_in->A;

	SG_in->group_order(go);
	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_avGa "
				"go=" << go << endl;
	}
	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_avGa "
				"before gens->init_conjugate_svas_of" << endl;
	}
	gens->init_conjugate_svas_of(
			SG_in->gens, Elt_a, verbose_level);
	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_avGa "
				"after gens->init_conjugate_svas_of" << endl;
	}

	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_avGa "
				"before generators_to_strong_generators" << endl;
	}
	SG_in->A->generators_to_strong_generators(
		true /* f_target_go */, go,
		gens, SG_out,
		0 /*verbose_level*/);

	FREE_OBJECT(gens);

	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_avGa done" << endl;
	}
}


void group_theory_global::strong_generators_conjugate_aGav(
		strong_generators *SG_in,
		int *Elt_a,
		strong_generators *&SG_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::vector_ge *gens;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_aGav" << endl;
	}

	SG_in->group_order(go);
	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_aGav "
				"go=" << go << endl;
	}
	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	gens->init_conjugate_sasv_of(
			SG_in->gens, Elt_a, 0 /* verbose_level */);



	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_aGav "
				"before generators_to_strong_generators" << endl;
	}
	SG_in->A->generators_to_strong_generators(
		true /* f_target_go */, go,
		gens, SG_out,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_aGav "
				"after generators_to_strong_generators" << endl;
	}

	FREE_OBJECT(gens);

	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_aGav done" << endl;
	}
}

void group_theory_global::set_of_coset_representatives(
		groups::strong_generators *Subgroup_gens_H,
		groups::strong_generators *Subgroup_gens_G,
		data_structures_groups::vector_ge *&coset_reps,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_global::set_of_coset_representatives" << endl;
	}

	//actions::action *A1;
	//actions::action *A2;

	//A1 = A;
	//A2 = AG_secondary->A;


	groups::sims *S;

	S = Subgroup_gens_G->create_sims(verbose_level);
	// the large group

	if (f_v) {
		cout << "group_theory_global::set_of_coset_representatives "
				"before Subgroup_gens_H->set_of_coset_representatives" << endl;
	}
	Subgroup_gens_H->set_of_coset_representatives(
			S,
			coset_reps,
			verbose_level);
	if (f_v) {
		cout << "group_theory_global::set_of_coset_representatives "
				"after Subgroup_gens_H->set_of_coset_representatives" << endl;
		cout << "group_theory_global::set_of_coset_representatives "
				"number of coset reps = " << coset_reps->len << endl;
	}



	FREE_OBJECT(S);

	if (f_v) {
		cout << "group_theory_global::set_of_coset_representatives done" << endl;
	}
}




}}}


