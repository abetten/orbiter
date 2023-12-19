/*
 * canonical_form_nauty.cpp
 *
 *  Created on: Apr 17, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



canonical_form_nauty::canonical_form_nauty()
{

	Classifier = NULL;

	Variety = NULL;

	nb_rows = 0;
	nb_cols = 0;
	Canonical_form = NULL;
	canonical_labeling = NULL;
	canonical_labeling_len = 0;


	Set_stab = NULL;

	Orb = NULL;

	Stab_gens_variety = NULL;

	f_found_canonical_form = false;
	idx_canonical_form = 0;
	idx_equation = 0;
	f_found_eqn = false;

	//std::vector<std::string> NO_stringified;
}

canonical_form_nauty::~canonical_form_nauty()
{
	if (canonical_labeling) {
		FREE_lint(canonical_labeling);
	}
	if (Set_stab) {
		FREE_OBJECT(Set_stab);
	}
	if (Orb) {
		FREE_OBJECT(Orb);
	}
	if (Stab_gens_variety) {
		FREE_OBJECT(Stab_gens_variety);
	}
}

void canonical_form_nauty::init(
		canonical_form_classifier *Classifier,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	canonical_form_nauty::Classifier = Classifier;

	if (f_v) {
		cout << "canonical_form_nauty::init" << endl;
	}
}


void canonical_form_nauty::compute_canonical_form_of_variety(
		canonical_form_of_variety *Variety,
		int verbose_level)
// Computes the canonical labeling of the graph associated with
// the set of rational points of the curve.
// Computes the stabilizer of the set of rational points of the curve.
// Computes the orbit of the equation under the stabilizer of the set.
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_nauty::compute_canonical_form_of_variety" << endl;
	}


	canonical_form_nauty::Variety = Variety;

	if (f_v) {
		Variety->Vo->Variety_object->print(cout);
	}


	if (f_v) {
		cout << "canonical_form_nauty::compute_canonical_form_of_variety "
				"before Nau.set_stabilizer_in_projective_space_using_nauty" << endl;
	}

	interfaces::nauty_interface_with_group Nau;

	Nau.set_stabilizer_in_projective_space_using_nauty(
			Classifier->PA->P,
			Classifier->PA->A,
			Variety->Vo->Variety_object->Point_sets->Sets[0],
			Variety->Vo->Variety_object->Point_sets->Set_size[0],
			Set_stab,
			Canonical_form,
			canonical_labeling, canonical_labeling_len,
			NO_stringified,
			verbose_level);

	if (f_v) {
		cout << "canonical_form_nauty::compute_canonical_form_of_variety "
				"after Nau.set_stabilizer_in_projective_space_using_nauty" << endl;
	}


	// The order of the set stabilizer is needed
	// in order to be able to compute the subgroup
	// which fixes the canonical equation.


	ring_theory::longinteger_object set_stab_order;

	Set_stab->group_order(set_stab_order);
	if (f_v) {
		cout << "canonical_form_nauty::compute_canonical_form_of_variety "
				"set_stab_order = " << set_stab_order << endl;
	}


	// compute the orbit of the equation
	// under the stabilizer of the set of points:




	if (f_v) {
		cout << "canonical_form_nauty::compute_canonical_form_of_variety "
				"before orbit_of_equation_under_set_stabilizer" << endl;
	}
	orbit_of_equation_under_set_stabilizer(verbose_level - 1);
	if (f_v) {
		cout << "canonical_form_nauty::compute_canonical_form_of_variety "
				"after orbit_of_equation_under_set_stabilizer" << endl;
	}



	if (f_v) {
		cout << "canonical_form_nauty::compute_canonical_form_of_variety done" << endl;
	}
}


void canonical_form_nauty::orbit_of_equation_under_set_stabilizer(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_nauty::orbit_of_equation_under_set_stabilizer" << endl;
	}


	Orb = NEW_OBJECT(orbits_schreier::orbit_of_equations);

	ring_theory::longinteger_object set_stab_order;

	Set_stab->group_order(set_stab_order);

#if 1
	if (f_v) {
		cout << "canonical_form_nauty::orbit_of_equation_under_set_stabilizer "
				"before Orb->init" << endl;
	}
	Orb->init(
			Classifier->PA->A,
			Classifier->PA->F,
			Classifier->AonHPD,
			Set_stab /* A->Strong_gens*/,
			Variety->Vo->Variety_object->eqn,
		verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::orbit_of_equation_under_set_stabilizer "
				"after Orb->init" << endl;
		cout << "canonical_form_nauty::orbit_of_equation_under_set_stabilizer "
				"found an orbit of length " << Orb->used_length << endl;
	}


	// Compute the canonical form
	// and get the stabilizer of the canonical form to
	// gens_stab_of_canonical_equation

	if (f_v) {
		cout << "canonical_form_nauty::orbit_of_equation_under_set_stabilizer "
				"before Orb->get_canonical_form" << endl;
	}
	Orb->get_canonical_form(
			Variety->canonical_equation,
			Variety->transporter_to_canonical_form,
			Variety->gens_stab_of_canonical_equation,
			set_stab_order,
				verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::orbit_of_equation_under_set_stabilizer "
				"after Orb->get_canonical_form" << endl;
	}

	if (f_v) {
		cout << "canonical_form_nauty::orbit_of_equation_under_set_stabilizer "
				"before Orb->stabilizer_orbit_rep" << endl;
	}
	Stab_gens_variety = Orb->stabilizer_orbit_rep(
			set_stab_order, verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::orbit_of_equation_under_set_stabilizer "
				"after Orb->stabilizer_orbit_rep" << endl;
	}
	if (f_v) {
		ring_theory::longinteger_object go;

		Stab_gens_variety->group_order(go);
		cout << "The stabilizer of the variety is a group of order " << go << endl;
		Stab_gens_variety->print_generators_tex(cout);
	}
#endif


	if (f_v) {
		cout << "canonical_form_nauty::orbit_of_equation_under_set_stabilizer done" << endl;
	}
}

void canonical_form_nauty::report(std::ostream &ost)
{
	ost << "Number of equations with the same set of points "
			<< Orb->used_length << "\\\\" << endl;

	ost << endl;
	ost << "\\bigskip" << endl;
	ost << endl;



	ost << "Automorphism group: \\\\" << endl;
	Stab_gens_variety->print_generators_tex(ost);


	if (Stab_gens_variety->group_order_as_lint() < 50) {

		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;

		ost << "List of all elements of the automorphism group: \\\\" << endl;
		Stab_gens_variety->print_elements_ost(ost);
	}

}

}}}


