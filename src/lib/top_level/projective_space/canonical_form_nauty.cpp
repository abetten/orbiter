/*
 * canonical_form_nauty.cpp
 *
 *  Created on: Apr 17, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



canonical_form_nauty::canonical_form_nauty()
{
	idx = 0;
	eqn = NULL;
	sz = 0;

	Pts_on_curve = NULL;
	sz_curve = 0;

	bitangents = NULL;
	nb_bitangents = 0;

	nb_rows = 0;
	nb_cols = 0;
	Canonical_form = NULL;
	canonical_labeling = NULL;
	canonical_labeling_len = 0;


	SG_pt_stab = NULL;

	Orb = NULL;

	Stab_gens_quartic = NULL;
}

canonical_form_nauty::~canonical_form_nauty()
{
}

void canonical_form_nauty::quartic_curve(
		projective_space_with_action *PA,
		homogeneous_polynomial_domain *Poly4_x123,
		action_on_homogeneous_polynomials *AonHPD,
		int idx, int *eqn, int sz,
		long int *Pts_on_curve, int sz_curve,
		long int *bitangents, int nb_bitangents,
		int *canonical_equation,
		int *transporter_to_canonical_form,
		strong_generators *&gens_stab_of_canonical_equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve" << endl;
	}

	canonical_form_nauty::idx = idx;
	canonical_form_nauty::eqn = eqn;
	canonical_form_nauty::sz = sz;
	canonical_form_nauty::Pts_on_curve = Pts_on_curve;
	canonical_form_nauty::sz_curve = sz_curve;
	canonical_form_nauty::bitangents = bitangents;
	canonical_form_nauty::nb_bitangents = nb_bitangents;

	if (f_v) {
		cout << "equation is:";
		Poly4_x123->print_equation_simple(cout, eqn);
		cout << endl;
	}

	longinteger_object pt_stab_order;
	object_in_projective_space *OiP = NULL;

	int f_compute_canonical_form = TRUE;


	OiP = NEW_OBJECT(object_in_projective_space);

	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve before OiP->init_point_set" << endl;
	}
	OiP->init_point_set(PA->P,
			Pts_on_curve, sz_curve,
			verbose_level - 1);
	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve after OiP->init_point_set" << endl;
	}

	int nb_rows, nb_cols;

	OiP->encoding_size(
				nb_rows, nb_cols,
				verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve nb_rows = " << nb_rows << endl;
		cout << "canonical_form_nauty::quartic_curve nb_cols = " << nb_cols << endl;
	}

	//canonical_labeling = NEW_lint(nb_rows + nb_cols);

	nauty_interface_with_group Nau;
	nauty_output *NO;

	NO = NEW_OBJECT(nauty_output);
	NO->allocate(nb_rows + nb_cols, verbose_level);


	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve before Nau.set_stabilizer_of_object" << endl;
	}
	SG_pt_stab = Nau.set_stabilizer_of_object(
		OiP,
		PA->A,
		f_compute_canonical_form, Canonical_form,
		//canonical_labeling, canonical_labeling_len,
		NO,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve after Nau.set_stabilizer_of_object" << endl;
	}


	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve "
				"go = " << *NO->Ago << endl;

		NO->print_stats();



	}

	FREE_OBJECT(NO);

	SG_pt_stab->group_order(pt_stab_order);
	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve "
				"pt_stab_order = " << pt_stab_order << endl;
	}

	FREE_OBJECT(OiP);
	//FREE_lint(canonical_labeling);





	// compute the orbit of the equation under the stabilizer of the set of points:


	//orbit_of_equations *Orb;

	Orb = NEW_OBJECT(orbit_of_equations);


#if 1
	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve "
				"before Orb->init" << endl;
	}
	Orb->init(PA->A, PA->F,
		AonHPD,
		SG_pt_stab /* A->Strong_gens*/, eqn,
		verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve "
				"after Orb->init" << endl;
		cout << "canonical_form_nauty::quartic_curve "
				"found an orbit of length " << Orb->used_length << endl;
	}


	// ToDo: we need to compute the canonical form!

	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve "
				"before Orb->get_canonical_form" << endl;
	}
	Orb->get_canonical_form(
				canonical_equation,
				transporter_to_canonical_form,
				gens_stab_of_canonical_equation,
				pt_stab_order,
				verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve "
				"after Orb->get_canonical_form" << endl;
	}

	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve "
				"before Orb->stabilizer_orbit_rep" << endl;
	}
	Stab_gens_quartic = Orb->stabilizer_orbit_rep(
			pt_stab_order, verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve "
				"after Orb->stabilizer_orbit_rep" << endl;
	}
	if (f_v) {
		longinteger_object go;

		Stab_gens_quartic->group_order(go);
		cout << "The stabilizer is a group of order " << go << endl;
		Stab_gens_quartic->print_generators_tex(cout);
	}
#endif

	//FREE_OBJECT(SG_pt_stab);
	//FREE_OBJECT(Orb);


	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve done" << endl;
	}
}




}}

