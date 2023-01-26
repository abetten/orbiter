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
#if 0
	idx = 0;
	eqn = NULL;
	sz = 0;

	Pts_on_curve = NULL;
	sz_curve = 0;

	bitangents = NULL;
	nb_bitangents = 0;
#endif

	Classifier = NULL;

	Qco = NULL;

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
		quartic_curve_object *Qco,
		int *canonical_equation,
		int *transporter_to_canonical_form,
		groups::strong_generators *&gens_stab_of_canonical_equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve" << endl;
	}

#if 0
	canonical_form_nauty::idx = Qco->cnt;
	canonical_form_nauty::eqn = Qco->eqn;
	canonical_form_nauty::sz = Qco->sz;
	canonical_form_nauty::Pts_on_curve = Qco->pts;
	canonical_form_nauty::sz_curve = Qco->nb_pts;
	canonical_form_nauty::bitangents = Qco->bitangents;
	canonical_form_nauty::nb_bitangents = Qco->nb_bitangents;
#endif

	canonical_form_nauty::Qco = Qco;

	if (f_v) {
		cout << "equation is:";
		Classifier->Poly_ring->print_equation_simple(cout, Qco->eqn);
		cout << endl;
	}

	ring_theory::longinteger_object pt_stab_order;
	geometry::object_with_canonical_form *OwCF = NULL;

	int f_compute_canonical_form = TRUE;


	OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"before OwCF->init_point_set" << endl;
	}
	OwCF->init_point_set(
			Qco->pts, Qco->nb_pts,
			verbose_level - 1);
	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"after OwCF->init_point_set" << endl;
	}
	OwCF->P = Classifier->Descr->PA->P;

	int nb_rows, nb_cols;

	OwCF->encoding_size(
				nb_rows, nb_cols,
				verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"nb_rows = " << nb_rows << endl;
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"nb_cols = " << nb_cols << endl;
	}


	actions::nauty_interface_with_group Nau;
	data_structures::nauty_output *NO;

	NO = NEW_OBJECT(data_structures::nauty_output);
	NO->allocate(nb_rows + nb_cols, verbose_level);


	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"before Nau.set_stabilizer_of_object" << endl;
	}
	SG_pt_stab = Nau.set_stabilizer_of_object(
			OwCF,
			Classifier->Descr->PA->A,
		f_compute_canonical_form, Canonical_form,
		NO,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"after Nau.set_stabilizer_of_object" << endl;
	}


	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"go = " << *NO->Ago << endl;

		NO->print_stats();



	}

	int i;

	canonical_labeling = NEW_lint(NO->N);
	for (i = 0; i < NO->N; i++) {
		canonical_labeling[i] = NO->canonical_labeling[i];
	}

	canonical_labeling_len = NO->N;

	FREE_OBJECT(NO);

	SG_pt_stab->group_order(pt_stab_order);
	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"pt_stab_order = " << pt_stab_order << endl;
	}

	FREE_OBJECT(OwCF);





	// compute the orbit of the equation under the stabilizer of the set of points:


	//orbit_of_equations *Orb;

	Orb = NEW_OBJECT(orbits_schreier::orbit_of_equations);


#if 1
	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"before Orb->init" << endl;
	}
	Orb->init(
			Classifier->Descr->PA->A,
			Classifier->Descr->PA->F,
			Classifier->AonHPD,
		SG_pt_stab /* A->Strong_gens*/, Qco->eqn,
		verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"after Orb->init" << endl;
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"found an orbit of length " << Orb->used_length << endl;
	}


	// ToDo: we need to compute the canonical form!

	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"before Orb->get_canonical_form" << endl;
	}
	Orb->get_canonical_form(
			canonical_equation,
			transporter_to_canonical_form,
			gens_stab_of_canonical_equation,
				pt_stab_order,
				verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"after Orb->get_canonical_form" << endl;
	}

	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"before Orb->stabilizer_orbit_rep" << endl;
	}
	Stab_gens_quartic = Orb->stabilizer_orbit_rep(
			pt_stab_order, verbose_level);
	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve "
				"after Orb->stabilizer_orbit_rep" << endl;
	}
	if (f_v) {
		ring_theory::longinteger_object go;

		Stab_gens_quartic->group_order(go);
		cout << "The stabilizer is a group of order " << go << endl;
		Stab_gens_quartic->print_generators_tex(cout);
	}
#endif

	//FREE_OBJECT(SG_pt_stab);
	//FREE_OBJECT(Orb);


	if (f_v) {
		cout << "canonical_form_nauty::canonical_form_of_quartic_curve done" << endl;
	}
}




}}}


