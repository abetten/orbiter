/*
 * stabilizer_of_set_of_rational_points.cpp
 *
 *  Created on: Apr 17, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



stabilizer_of_set_of_rational_points::stabilizer_of_set_of_rational_points()
{

	Classifier = NULL;

	Variety = NULL;

	nb_rows = 0;
	nb_cols = 0;
	Canonical_form = NULL;
	NO = NULL;


	Set_stab = NULL;

	Orb = NULL;

	Stab_gens_variety = NULL;

	f_found_canonical_form = false;
	idx_canonical_form = 0;
	idx_equation = 0;
	f_found_eqn = false;

}

stabilizer_of_set_of_rational_points::~stabilizer_of_set_of_rational_points()
{
	if (NO) {
		FREE_OBJECT(NO);
	}
#if 0
	if (canonical_labeling) {
		FREE_lint(canonical_labeling);
	}
#endif
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

void stabilizer_of_set_of_rational_points::init(
		canonical_form_classifier *Classifier,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	stabilizer_of_set_of_rational_points::Classifier = Classifier;

	if (f_v) {
		cout << "stabilizer_of_set_of_rational_points::init" << endl;
	}
}


void stabilizer_of_set_of_rational_points::compute_canonical_form_of_variety(
		canonical_form_of_variety *Variety,
		int verbose_level)
// Computes the canonical labeling of the graph associated with
// the set of rational points of the curve.
// Computes the stabilizer of the set of rational points of the curve.
// Computes the orbit of the equation under the stabilizer of the set.
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "stabilizer_of_set_of_rational_points::compute_canonical_form_of_variety" << endl;
	}


	stabilizer_of_set_of_rational_points::Variety = Variety;

	if (f_v) {
		Variety->Vo->Variety_object->print(cout);
		//NO_N,NO_ago,NO_base_len,NO_aut_cnt,NO_base,NO_tl,NO_aut,NO_cl,NO_stats
		//int f_has_nauty_output;
		//int nauty_output_column_index_start;
		//std::vector<std::string> Carrying_through;
	}



	interfaces::nauty_interface_with_group Nau;

	if (Variety->Vo->f_has_nauty_output) {
		if (f_v) {
			cout << "stabilizer_of_set_of_rational_points::compute_canonical_form_of_variety "
					"f_has_nauty_output" << endl;
		}
#if 0
		void set_stabilizer_in_projective_space_using_precomputed_nauty_data(
				geometry::projective_space *P,
				actions::action *A,
				long int *Pts, int sz,
				int nauty_output_index_start,
				std::vector<std::string> &Carrying_through,
				groups::strong_generators *&Set_stab,
				data_structures::bitvector *&Canonical_form,
				l1_interfaces::nauty_output *&NO,
				int verbose_level);
#endif

		if (f_v) {
			cout << "stabilizer_of_set_of_rational_points::compute_canonical_form_of_variety "
					"before Nau.set_stabilizer_in_projective_space_using_precomputed_nauty_data" << endl;
		}
		Nau.set_stabilizer_in_projective_space_using_precomputed_nauty_data(
				Classifier->PA->P,
				Classifier->PA->A,
				Variety->Vo->Variety_object->Point_sets->Sets[0],
				Variety->Vo->Variety_object->Point_sets->Set_size[0],
				Variety->Vo->nauty_output_index_start,
				Variety->Vo->Carrying_through,
				Set_stab,
				Canonical_form,
				NO,
				verbose_level);
		if (f_v) {
			cout << "stabilizer_of_set_of_rational_points::compute_canonical_form_of_variety "
					"after Nau.set_stabilizer_in_projective_space_using_precomputed_nauty_data" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "stabilizer_of_set_of_rational_points::compute_canonical_form_of_variety "
					"before Nau.set_stabilizer_in_projective_space_using_nauty" << endl;
		}

		Nau.set_stabilizer_in_projective_space_using_nauty(
				Classifier->PA->P,
				Classifier->PA->A,
				Variety->Vo->Variety_object->Point_sets->Sets[0],
				Variety->Vo->Variety_object->Point_sets->Set_size[0],
				Set_stab,
				Canonical_form,
				NO,
				verbose_level);

		if (f_v) {
			cout << "stabilizer_of_set_of_rational_points::compute_canonical_form_of_variety "
					"after Nau.set_stabilizer_in_projective_space_using_nauty" << endl;
		}
	}


	// The order of the set stabilizer is needed
	// in order to be able to compute the subgroup
	// which fixes the canonical equation.


	ring_theory::longinteger_object set_stab_order;

	Set_stab->group_order(set_stab_order);
	if (f_v) {
		cout << "stabilizer_of_set_of_rational_points::compute_canonical_form_of_variety "
				"set_stab_order = " << set_stab_order << endl;
	}


	// compute the orbit of the equation
	// under the stabilizer of the set of points:




	if (f_v) {
		cout << "stabilizer_of_set_of_rational_points::compute_canonical_form_of_variety "
				"before orbit_of_equation_under_set_stabilizer" << endl;
	}
	orbit_of_equation_under_set_stabilizer(verbose_level - 1);
	if (f_v) {
		cout << "stabilizer_of_set_of_rational_points::compute_canonical_form_of_variety "
				"after orbit_of_equation_under_set_stabilizer" << endl;
	}



	if (f_v) {
		cout << "stabilizer_of_set_of_rational_points::compute_canonical_form_of_variety done" << endl;
	}
}


void stabilizer_of_set_of_rational_points::orbit_of_equation_under_set_stabilizer(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "stabilizer_of_set_of_rational_points::orbit_of_equation_under_set_stabilizer" << endl;
	}


	Orb = NEW_OBJECT(orbits_schreier::orbit_of_equations);

	ring_theory::longinteger_object set_stab_order;

	Set_stab->group_order(set_stab_order);

#if 1
	if (f_v) {
		cout << "stabilizer_of_set_of_rational_points::orbit_of_equation_under_set_stabilizer "
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
		cout << "stabilizer_of_set_of_rational_points::orbit_of_equation_under_set_stabilizer "
				"after Orb->init" << endl;
		cout << "stabilizer_of_set_of_rational_points::orbit_of_equation_under_set_stabilizer "
				"found an orbit of length " << Orb->used_length << endl;
	}


	// Compute the canonical form
	// and get the stabilizer of the canonical form to
	// gens_stab_of_canonical_equation

	if (f_v) {
		cout << "stabilizer_of_set_of_rational_points::orbit_of_equation_under_set_stabilizer "
				"before Orb->get_canonical_form" << endl;
	}
	Orb->get_canonical_form(
			Variety->canonical_equation,
			Variety->transporter_to_canonical_form,
			Variety->gens_stab_of_canonical_equation,
			set_stab_order,
				verbose_level);
	if (f_v) {
		cout << "stabilizer_of_set_of_rational_points::orbit_of_equation_under_set_stabilizer "
				"after Orb->get_canonical_form" << endl;
	}

	if (f_v) {
		cout << "stabilizer_of_set_of_rational_points::orbit_of_equation_under_set_stabilizer "
				"before Orb->stabilizer_orbit_rep" << endl;
	}
	Stab_gens_variety = Orb->stabilizer_orbit_rep(
			set_stab_order, verbose_level);
	if (f_v) {
		cout << "stabilizer_of_set_of_rational_points::orbit_of_equation_under_set_stabilizer "
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
		cout << "stabilizer_of_set_of_rational_points::orbit_of_equation_under_set_stabilizer done" << endl;
	}
}

void stabilizer_of_set_of_rational_points::report(
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


