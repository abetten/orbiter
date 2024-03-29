/*
 * automorphism_group_of_variety.cpp
 *
 *  Created on: Mar 29, 2024
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



automorphism_group_of_variety::automorphism_group_of_variety()
{
	PA = NULL;

	HPD = NULL;

	AonHPD = NULL;


	equation = NULL;
	Pts_on_object = NULL;
	nb_pts = 0;

	OwCF = NULL;

	nb_rows = nb_cols = 0;

	Canonical_form = NULL;

	NO = NULL;

	Enc = NULL;

	SG_pt_stab = NULL;
	//ring_theory::longinteger_object pt_stab_order;

	Orb = NULL;

	Stab_gens_quartic = NULL;


#if 0
	projective_geometry::projective_space_with_action *PA;

	ring_theory::homogeneous_polynomial_domain *HPD;

	induced_actions::action_on_homogeneous_polynomials *AonHPD;

	int *equation;

	canonical_form_classification::object_with_canonical_form *OwCF;

	data_structures::bitvector *Canonical_form;

	l1_interfaces::nauty_output *NO;

	canonical_form_classification::encoded_combinatorial_object *Enc;

	groups::strong_generators *SG_pt_stab;
		// the stabilizer of the set of rational points
	ring_theory::longinteger_object pt_stab_order;
		// order of stabilizer of the set of rational points

	orbits_schreier::orbit_of_equations *Orb;

	groups::strong_generators *Stab_gens_quartic;
		// stabilizer of quartic curve obtained by doing an orbit algorithm
#endif

}

automorphism_group_of_variety::~automorphism_group_of_variety()
{
	if (equation) {
		FREE_int(equation);
	}
	if (OwCF) {
		FREE_OBJECT(OwCF);
	}
	if (Canonical_form) {
		FREE_OBJECT(Canonical_form);
	}
	if (NO) {
		FREE_OBJECT(NO);
	}
	if (Enc) {
		FREE_OBJECT(Enc);
	}
	if (SG_pt_stab) {
		FREE_OBJECT(SG_pt_stab);
	}
	if (Orb) {
		FREE_OBJECT(Orb);
	}
	if (Stab_gens_quartic) {
		FREE_OBJECT(Stab_gens_quartic);
	}

}

void automorphism_group_of_variety::init(
		projective_geometry::projective_space_with_action *PA,
		induced_actions::action_on_homogeneous_polynomials *AonHPD,
		int *equation,
		long int *Pts_on_object,
		int nb_pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "automorphism_group_of_variety::init" << endl;
	}
	automorphism_group_of_variety::PA = PA;
	automorphism_group_of_variety::AonHPD = AonHPD;
	HPD = AonHPD->HPD;
	automorphism_group_of_variety::equation = NEW_int(HPD->get_nb_monomials());
	Int_vec_copy(equation, automorphism_group_of_variety::equation, HPD->get_nb_monomials());
	automorphism_group_of_variety::Pts_on_object = Pts_on_object;
	automorphism_group_of_variety::nb_pts = nb_pts;

	OwCF = NEW_OBJECT(canonical_form_classification::object_with_canonical_form);

	if (f_v) {
		cout << "automorphism_group_of_variety::init "
				"before OwCF->init_point_set" << endl;
	}
	OwCF->init_point_set(
			Pts_on_object, nb_pts,
			verbose_level - 1);
	if (f_v) {
		cout << "automorphism_group_of_variety::init "
				"after Quartic_curve_from_surface->OwCF->init_point_set" << endl;
	}
	OwCF->P = PA->P;


	OwCF->encoding_size(
				nb_rows, nb_cols,
				verbose_level);
	if (f_v) {
		cout << "automorphism_group_of_variety::init "
				"nb_rows = " << nb_rows << endl;
		cout << "automorphism_group_of_variety::init "
				"nb_cols = " << nb_cols << endl;
	}


	interfaces::nauty_interface_with_group Nau;


	NO = NEW_OBJECT(l1_interfaces::nauty_output);

	if (f_v) {
		cout << "automorphism_group_of_variety::init "
				"before NO->nauty_output_allocate" << endl;
	}

	NO->nauty_output_allocate(nb_rows + nb_cols,
			0,
			nb_rows + nb_cols,
			0 /* verbose_level */);

	if (f_v) {
		cout << "automorphism_group_of_variety::init "
				"after NO->nauty_output_allocate" << endl;
	}

	int f_compute_canonical_form = false;


	if (f_v) {
		cout << "automorphism_group_of_variety::init "
				"before Nau.set_stabilizer_of_object" << endl;
	}

	SG_pt_stab = Nau.set_stabilizer_of_object(
			OwCF,
		PA->A,
		f_compute_canonical_form, Canonical_form,
		NO,
		Enc,
		verbose_level);

	if (f_v) {
		cout << "automorphism_group_of_variety::init "
				"after Nau.set_stabilizer_of_object" << endl;
	}

#if 0
	string file_name_encoding;

	file_name_encoding = Quartic_curve_from_surface->SOA->SO->label_txt + "_encoding";
	Quartic_curve_from_surface->Enc->save_incma(file_name_encoding, verbose_level);
#endif

	if (f_v) {
		NO->print_stats();
	}

	//FREE_OBJECT(NO);
	//FREE_OBJECT(Enc);

	SG_pt_stab->group_order(pt_stab_order);
	if (f_v) {
		cout << "automorphism_group_of_variety::init "
				"pt_stab_order = " << pt_stab_order << endl;
	}

	//FREE_OBJECT(OwCF);



	// compute the orbit of the equation under the stabilizer of the set of points:



	Orb = NEW_OBJECT(orbits_schreier::orbit_of_equations);


#if 1
	if (f_v) {
		cout << "automorphism_group_of_variety::init "
				"before Orb->init" << endl;
	}
	Orb->init(
			PA->A,
			PA->F,
			AonHPD /* AonHPD */,
			SG_pt_stab /* A->Strong_gens*/,
			equation,
			verbose_level);
	if (f_v) {
		cout << "automorphism_group_of_variety::init "
				"after Orb->init" << endl;
		cout << "automorphism_group_of_variety::init "
				"found an orbit of length " << Orb->used_length << endl;
	}




	if (f_v) {
		cout << "automorphism_group_of_variety::init "
				"before Orb->stabilizer_orbit_rep" << endl;
	}
	Stab_gens_quartic = Orb->stabilizer_orbit_rep(
			pt_stab_order, verbose_level);
	if (f_v) {
		cout << "automorphism_group_of_variety::init "
				"after Orb->stabilizer_orbit_rep" << endl;
	}
	Stab_gens_quartic->print_generators_tex(cout);
#endif

	if (f_v) {
		cout << "automorphism_group_of_variety::init done" << endl;
	}
}

}}}

