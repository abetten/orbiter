/*
 * variety_stabilizer_compute.cpp
 *
 *  Created on: Apr 17, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



variety_stabilizer_compute::variety_stabilizer_compute()
{

	Ring_with_action = NULL;

	Variety_object_with_action = NULL;

	nb_rows = 0;
	nb_cols = 0;
	Canonical_form = NULL;
	NO = NULL;


	Set_stab = NULL;

	Orb = NULL;

	Stab_gens_variety = NULL;

}

variety_stabilizer_compute::~variety_stabilizer_compute()
{
	if (NO) {
		FREE_OBJECT(NO);
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

void variety_stabilizer_compute::init(
		projective_geometry::ring_with_action *Ring_with_action,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	variety_stabilizer_compute::Ring_with_action = Ring_with_action;

	if (f_v) {
		cout << "variety_stabilizer_compute::init" << endl;
	}
}


void variety_stabilizer_compute::compute_canonical_form_of_variety(
		variety_object_with_action *Variety_object_with_action,
		int f_save_nauty_input_graphs,
		int verbose_level)
// Computes the canonical labeling of the graph associated with
// the set of rational points of the curve.
// Computes the stabilizer of the set of rational points of the curve.
// Computes the orbit of the equation under the stabilizer of the set.
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "variety_stabilizer_compute::compute_canonical_form_of_variety" << endl;
	}


	variety_stabilizer_compute::Variety_object_with_action = Variety_object_with_action;

	if (f_v) {
		//Variety->Vo->Variety_object->print(cout);
		Variety_object_with_action->print(cout);
		//NO_N,NO_ago,NO_base_len,NO_aut_cnt,NO_base,NO_tl,NO_aut,NO_cl,NO_stats
		//int f_has_nauty_output;
		//int nauty_output_column_index_start;
		//std::vector<std::string> Carrying_through;
	}



	interfaces::nauty_interface_with_group Nau;

	if (Variety_object_with_action->f_has_nauty_output) {
		if (f_v) {
			cout << "variety_stabilizer_compute::compute_canonical_form_of_variety "
					"f_has_nauty_output" << endl;
		}

		if (f_v) {
			cout << "variety_stabilizer_compute::compute_canonical_form_of_variety "
					"before Nau.set_stabilizer_in_projective_space_using_precomputed_nauty_data" << endl;
		}
		Nau.set_stabilizer_in_projective_space_using_precomputed_nauty_data(
				Ring_with_action->PA->P,
				Ring_with_action->PA->A,
				Variety_object_with_action->Variety_object->Point_sets->Sets[0],
				Variety_object_with_action->Variety_object->Point_sets->Set_size[0],
				f_save_nauty_input_graphs,
				Variety_object_with_action->nauty_output_index_start,
				Variety_object_with_action->Carrying_through,
				Set_stab,
				Canonical_form,
				NO,
				verbose_level);
		if (f_v) {
			cout << "variety_stabilizer_compute::compute_canonical_form_of_variety "
					"after Nau.set_stabilizer_in_projective_space_using_precomputed_nauty_data" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "variety_stabilizer_compute::compute_canonical_form_of_variety "
					"before Nau.set_stabilizer_in_projective_space_using_nauty" << endl;
		}

		Nau.set_stabilizer_in_projective_space_using_nauty(
				Ring_with_action->PA->P,
				Ring_with_action->PA->A,
				Variety_object_with_action->Variety_object->Point_sets->Sets[0],
				Variety_object_with_action->Variety_object->Point_sets->Set_size[0],
				f_save_nauty_input_graphs,
				Set_stab,
				Canonical_form,
				NO,
				verbose_level);

		if (f_v) {
			cout << "variety_stabilizer_compute::compute_canonical_form_of_variety "
					"after Nau.set_stabilizer_in_projective_space_using_nauty" << endl;
		}
	}


	// The order of the set stabilizer is needed
	// in order to be able to compute the subgroup
	// which fixes the canonical equation.


	ring_theory::longinteger_object set_stab_order;

	Set_stab->group_order(set_stab_order);
	if (f_v) {
		cout << "variety_stabilizer_compute::compute_canonical_form_of_variety "
				"set_stab_order = " << set_stab_order << endl;
	}

	if (set_stab_order.is_zero()) {
		cout << "variety_stabilizer_compute::compute_canonical_form_of_variety "
				"set_stab_order = " << set_stab_order << " is zero, this cannot be right. " << endl;
		exit(1);
	}


	// compute the orbit of the equation
	// under the stabilizer of the set of points:




	if (f_v) {
		cout << "variety_stabilizer_compute::compute_canonical_form_of_variety "
				"before orbit_of_equation_under_set_stabilizer" << endl;
	}
	orbit_of_equation_under_set_stabilizer(verbose_level - 1);
	if (f_v) {
		cout << "variety_stabilizer_compute::compute_canonical_form_of_variety "
				"after orbit_of_equation_under_set_stabilizer" << endl;
	}



	if (f_v) {
		cout << "variety_stabilizer_compute::compute_canonical_form_of_variety done" << endl;
	}
}


void variety_stabilizer_compute::orbit_of_equation_under_set_stabilizer(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "variety_stabilizer_compute::orbit_of_equation_under_set_stabilizer" << endl;
	}


	Orb = NEW_OBJECT(orbits_schreier::orbit_of_equations);


#if 1
	if (f_v) {
		cout << "variety_stabilizer_compute::orbit_of_equation_under_set_stabilizer "
				"before Orb->init" << endl;
	}
	Orb->init(
			Ring_with_action->PA->A,
			Ring_with_action->PA->F,
			Ring_with_action->AonHPD,
			Set_stab /* A->Strong_gens*/,
			Variety_object_with_action->Variety_object->eqn,
		verbose_level);
	if (f_v) {
		cout << "variety_stabilizer_compute::orbit_of_equation_under_set_stabilizer "
				"after Orb->init" << endl;
		cout << "variety_stabilizer_compute::orbit_of_equation_under_set_stabilizer "
				"found an orbit of length " << Orb->used_length << endl;
	}


	ring_theory::longinteger_object set_stab_order;

	Set_stab->group_order(set_stab_order);

	// Compute the canonical form
	// and get the stabilizer of the canonical form to
	// gens_stab_of_canonical_equation

#if 0
	if (f_v) {
		cout << "variety_stabilizer_compute::orbit_of_equation_under_set_stabilizer "
				"before Orb->get_canonical_form" << endl;
	}
	Orb->get_canonical_form(
			Variety->canonical_equation,
			Variety->transporter_to_canonical_form,
			Variety->gens_stab_of_canonical_equation,
			set_stab_order,
				verbose_level);
	if (f_v) {
		cout << "variety_stabilizer_compute::orbit_of_equation_under_set_stabilizer "
				"after Orb->get_canonical_form" << endl;
	}
#endif

	if (f_v) {
		cout << "variety_stabilizer_compute::orbit_of_equation_under_set_stabilizer "
				"before Orb->stabilizer_orbit_rep" << endl;
	}
	Stab_gens_variety = Orb->stabilizer_orbit_rep(
			set_stab_order, verbose_level);
	if (f_v) {
		cout << "variety_stabilizer_compute::orbit_of_equation_under_set_stabilizer "
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
		cout << "variety_stabilizer_compute::orbit_of_equation_under_set_stabilizer done" << endl;
	}
}

void variety_stabilizer_compute::report(
		std::ostream &ost)
{
	ost << "Number of equations with the same set of points "
			<< Orb->used_length << "\\\\" << endl;

	long int nauty_complexity;

	nauty_complexity = NO->nauty_complexity();

	ost << "Nauty complexity = "
			<< nauty_complexity << "\\\\" << endl;

	ost << endl;
	ost << "\\bigskip" << endl;
	ost << endl;

	Orb->print_orbit_as_equations_tex(ost);

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


