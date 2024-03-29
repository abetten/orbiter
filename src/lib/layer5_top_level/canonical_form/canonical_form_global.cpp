/*
 * canonical_form_global.cpp
 *
 *  Created on: Mar 29, 2024
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



canonical_form_global::canonical_form_global()
{
}

canonical_form_global::~canonical_form_global()
{
}

void canonical_form_global::compute_stabilizer_of_quartic_curve(
		applications_in_algebraic_geometry::quartic_curves::quartic_curve_from_surface
			*Quartic_curve_from_surface,
		automorphism_group_of_variety *&Aut_of_variety,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve" << endl;
	}


	Aut_of_variety = NEW_OBJECT(automorphism_group_of_variety);

	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"before Aut_of_variety->init" << endl;
	}

	Aut_of_variety->init(
			Quartic_curve_from_surface->SOA->Surf_A->PA->PA2,
			Quartic_curve_from_surface->SOA->Surf_A->AonHPD_4_3,
			Quartic_curve_from_surface->curve,
			Quartic_curve_from_surface->Pts_on_curve, Quartic_curve_from_surface->sz_curve,
			verbose_level);

	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"after Aut_of_variety->init" << endl;
	}

#if 0
	//applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action *Surf_A;

	//Surf_A = Quartic_curve_from_surface->SOA->Surf_A;

	Quartic_curve_from_surface->OwCF = NEW_OBJECT(canonical_form_classification::object_with_canonical_form);

	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"before Quartic_curve_from_surface->OwCF->init_point_set" << endl;
	}
	Quartic_curve_from_surface->OwCF->init_point_set(
			Quartic_curve_from_surface->Pts_on_curve, Quartic_curve_from_surface->sz_curve,
			verbose_level - 1);
	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"after Quartic_curve_from_surface->OwCF->init_point_set" << endl;
	}
	Quartic_curve_from_surface->OwCF->P = Surf_A->PA->PA2->P;

	int nb_rows, nb_cols;

	Quartic_curve_from_surface->OwCF->encoding_size(
				nb_rows, nb_cols,
				verbose_level);
	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"nb_rows = " << nb_rows << endl;
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"nb_cols = " << nb_cols << endl;
	}


	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"before Nau.set_stabilizer_of_object" << endl;
	}

	interfaces::nauty_interface_with_group Nau;


	Quartic_curve_from_surface->NO = NEW_OBJECT(l1_interfaces::nauty_output);

	Quartic_curve_from_surface->NO->nauty_output_allocate(nb_rows + nb_cols,
			0,
			nb_rows + nb_cols,
			0 /* verbose_level */);

	int f_compute_canonical_form = false;


	Quartic_curve_from_surface->SG_pt_stab = Nau.set_stabilizer_of_object(
			Quartic_curve_from_surface->OwCF,
		Surf_A->PA->PA2->A,
		f_compute_canonical_form, Quartic_curve_from_surface->Canonical_form,
		Quartic_curve_from_surface->NO,
		Quartic_curve_from_surface->Enc,
		verbose_level);

	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"after Nau.set_stabilizer_of_object" << endl;
	}


	string file_name_encoding;

	file_name_encoding = Quartic_curve_from_surface->SOA->SO->label_txt + "_encoding";
	Quartic_curve_from_surface->Enc->save_incma(file_name_encoding, verbose_level);

	if (f_v) {
		Quartic_curve_from_surface->NO->print_stats();
	}

	//FREE_OBJECT(NO);
	//FREE_OBJECT(Enc);

	Quartic_curve_from_surface->SG_pt_stab->group_order(Quartic_curve_from_surface->pt_stab_order);
	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"pt_stab_order = " << Quartic_curve_from_surface->pt_stab_order << endl;
	}

	//FREE_OBJECT(OwCF);



	// compute the orbit of the equation under the stabilizer of the set of points:



	Quartic_curve_from_surface->Orb = NEW_OBJECT(orbits_schreier::orbit_of_equations);


#if 1
	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"before Quartic_curve_from_surface->Orb->init" << endl;
	}
	Quartic_curve_from_surface->Orb->init(
			Surf_A->PA->PA2->A,
			Surf_A->PA->F,
			Surf_A->AonHPD_4_3 /* AonHPD */,
			Quartic_curve_from_surface->SG_pt_stab /* A->Strong_gens*/,
		Quartic_curve_from_surface->curve,
		verbose_level);
	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"after Orb->init" << endl;
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"found an orbit of length " << Quartic_curve_from_surface->Orb->used_length << endl;
	}




	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"before Orb->stabilizer_orbit_rep" << endl;
	}
	Quartic_curve_from_surface->Stab_gens_quartic = Quartic_curve_from_surface->Orb->stabilizer_orbit_rep(
			Quartic_curve_from_surface->pt_stab_order, verbose_level);
	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve "
				"after Orb->stabilizer_orbit_rep" << endl;
	}
	Quartic_curve_from_surface->Stab_gens_quartic->print_generators_tex(cout);
#endif

	//FREE_OBJECT(SG_pt_stab);
	//FREE_OBJECT(Orb);
#endif

	if (f_v) {
		cout << "canonical_form_global::compute_stabilizer_of_quartic_curve" << endl;
	}
}


}}}



