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
namespace projective_geometry {



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

	Stab_gens_quartic = NULL;

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
	if (Stab_gens_quartic) {
		FREE_OBJECT(Stab_gens_quartic);
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


void canonical_form_nauty::canonical_form_of_quartic_curve(
		canonical_form_of_variety *Variety,
		int verbose_level)
// Computes the canonical labeling of the graph associated with
// the set of rational points of the curve.
// Computes the stabilizer of the set of rational points of the curve.
// Computes the orbit of the equation under the stabilizer of the set.
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve" << endl;
	}


	canonical_form_nauty::Variety = Variety;

	if (f_v) {
		cout << "equation is:";
		Classifier->Poly_ring->print_equation_simple(
				cout, Variety->Qco->Quartic_curve_object->eqn15);
		cout << endl;
	}


	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"before set_stabilizer_using_nauty" << endl;
	}

	set_stabilizer_using_nauty(verbose_level - 1);

	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"after set_stabilizer_using_nauty" << endl;
	}


	// The order of the set stabilizer is needed
	// in order to be able to compute the subgroup
	// which fixes the canonical equation.


	ring_theory::longinteger_object set_stab_order;

	Set_stab->group_order(set_stab_order);
	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"set_stab_order = " << set_stab_order << endl;
	}


	// compute the orbit of the equation
	// under the stabilizer of the set of points:




	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"before orbit_of_equation_under_set_stabilizer" << endl;
	}
	orbit_of_equation_under_set_stabilizer(verbose_level - 1);
	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"after orbit_of_equation_under_set_stabilizer" << endl;
	}



	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve done" << endl;
	}
}

void canonical_form_nauty::set_stabilizer_using_nauty(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_nauty::set_stabilizer_using_nauty" << endl;
	}


	geometry::object_with_canonical_form *OwCF = NULL;


	OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

	if (f_v) {
		cout << "canonical_form_nauty::set_stabilizer_using_nauty "
				"before OwCF->init_point_set" << endl;
	}
	OwCF->init_point_set(
			Variety->Qco->Quartic_curve_object->Pts,
			Variety->Qco->Quartic_curve_object->nb_pts,
			verbose_level - 1);
	if (f_v) {
		cout << "canonical_form_nauty::set_stabilizer_using_nauty "
				"after OwCF->init_point_set" << endl;
	}
	OwCF->P = Classifier->PA->P;

	int nb_rows, nb_cols;

	OwCF->encoding_size(
				nb_rows, nb_cols,
				verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::set_stabilizer_using_nauty "
				"nb_rows = " << nb_rows << endl;
		cout << "canonical_form_nauty::set_stabilizer_using_nauty "
				"nb_cols = " << nb_cols << endl;
	}


	interfaces::nauty_interface_with_group Nau;
	l1_interfaces::nauty_output *NO;
	combinatorics::encoded_combinatorial_object *Enc;

	NO = NEW_OBJECT(l1_interfaces::nauty_output);
	NO->nauty_output_allocate(nb_rows + nb_cols,
			0,
			nb_rows + nb_cols,
			verbose_level);


	if (f_v) {
		cout << "canonical_form_nauty::set_stabilizer_using_nauty "
				"before Nau.set_stabilizer_of_object" << endl;
	}
	Set_stab = Nau.set_stabilizer_of_object(
			OwCF,
			Classifier->PA->A,
		true /* f_compute_canonical_form */,
		Canonical_form,
		NO,
		Enc,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "canonical_form_nauty::set_stabilizer_using_nauty "
				"after Nau.set_stabilizer_of_object" << endl;
	}


	if (f_v) {
		cout << "canonical_form_nauty::set_stabilizer_using_nauty "
				"order of set stabilizer = " << *NO->Ago << endl;

		NO->print_stats();
	}

	NO->stringify_as_vector(
			NO_stringified,
			verbose_level);



	canonical_labeling = NEW_lint(NO->N);
	canonical_labeling_len = NO->N;

	Int_vec_copy_to_lint(
			NO->canonical_labeling,
			canonical_labeling,
			canonical_labeling_len);

	FREE_OBJECT(NO);
	FREE_OBJECT(Enc);
	FREE_OBJECT(OwCF);


	if (f_v) {
		cout << "canonical_form_nauty::set_stabilizer_using_nauty done" << endl;
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
			Variety->Qco->Quartic_curve_object->eqn15,
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
	Stab_gens_quartic = Orb->stabilizer_orbit_rep(
			set_stab_order, verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::orbit_of_equation_under_set_stabilizer "
				"after Orb->stabilizer_orbit_rep" << endl;
	}
	if (f_v) {
		ring_theory::longinteger_object go;

		Stab_gens_quartic->group_order(go);
		cout << "The stabilizer is a group of order " << go << endl;
		Stab_gens_quartic->print_generators_tex(cout);
	}
#endif


	if (f_v) {
		cout << "canonical_form_nauty::orbit_of_equation_under_set_stabilizer done" << endl;
	}
}

}}}


