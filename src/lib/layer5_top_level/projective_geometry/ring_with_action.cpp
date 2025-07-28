/*
 * ring_with_action.cpp
 *
 *  Created on: Sep 20, 2024
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace projective_geometry {



ring_with_action::ring_with_action()
{
	Record_birth();
	PA = NULL;

	Poly_ring = NULL;

	AonHPD = NULL;
}


ring_with_action::~ring_with_action()
{
	Record_death();
	if (AonHPD) {
		FREE_OBJECT(AonHPD);
	}
}


void ring_with_action::ring_with_action_init(
		projective_geometry::projective_space_with_action *PA,
		algebra::ring_theory::homogeneous_polynomial_domain *Poly_ring,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_with_action::ring_with_action_init "
				"verbose_level=" << verbose_level << endl;
	}

	ring_with_action::PA = PA;
	ring_with_action::Poly_ring = Poly_ring;
	//ring_with_action::AonHPD = AonHPD;

	AonHPD = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "ring_with_action::ring_with_action_init "
				"before AonHPD->init" << endl;
	}
	AonHPD->init(
			PA->A, Poly_ring, verbose_level - 3);
	if (f_v) {
		cout << "ring_with_action::ring_with_action_init "
				"after AonHPD->init" << endl;
	}


	if (f_v) {
		cout << "ring_with_action::ring_with_action_init done" << endl;
	}
}

void ring_with_action::lift_mapping(
		int *gamma, int *Elt, int verbose_level)
// turn the permutation gamma into a semilinear mapping
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_with_action::lift_mapping" << endl;
	}



	int d;
	int *Mtx; // [d * d + 1]


	d = PA->P->Subspaces->n + 1;
	Mtx = NEW_int(d * d + 1);


	int frobenius;
	algebra::linear_algebra::linear_algebra_global LA;

	if (f_v) {
		cout << "ring_with_action::lift_mapping "
				"before LA.reverse_engineer_semilinear_map" << endl;
	}

	// works for any dimension:

	LA.reverse_engineer_semilinear_map(
			PA->P->Subspaces->F,
			PA->P->Subspaces->n,
			gamma, Mtx, frobenius,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "ring_with_action::lift_mapping "
				"after LA.reverse_engineer_semilinear_map" << endl;
	}

	Mtx[d * d] = frobenius;

	PA->A->Group_element->make_element(
			Elt, Mtx,
			0 /* verbose_level*/);


	FREE_int(Mtx);

	if (f_v) {
		cout << "ring_with_action::lift_mapping done" << endl;
	}
}

void ring_with_action::apply(
		int *Elt, int *eqn_in, int *eqn_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_with_action::apply" << endl;
	}
	if (f_v) {
		cout << "ring_with_action::apply "
				"before substitute_semilinear" << endl;
	}

	int d, frobenius, frobenius_inv;
	int *Mtx; // [d * d + 1]


	d = PA->P->Subspaces->n + 1;
	Mtx = Elt;
	frobenius = Mtx[d * d];

	if (frobenius) {
		frobenius_inv = Poly_ring->get_F()->e - frobenius;
	}
	else {
		frobenius_inv = frobenius;
	}

	Poly_ring->substitute_semilinear(
			eqn_in,
			eqn_out,
			PA->A->is_semilinear_matrix_group(),
			frobenius_inv, Mtx,
			0/*verbose_level*/);

	if (f_v) {
		cout << "ring_with_action::apply "
				"after substitute_semilinear" << endl;
	}

	if (f_v) {
		cout << "ring_with_action::apply done" << endl;
	}
}

void ring_with_action::nauty_interface(
		canonical_form::variety_object_with_action *Variety_object_with_action,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		groups::strong_generators *&Set_stab,
		other::data_structures::bitvector *&Canonical_form,
		other::l1_interfaces::nauty_output *&NO,
		int verbose_level)
// called from variety_stabilizer_compute::compute_canonical_form_of_variety
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_with_action::nauty_interface" << endl;
		cout << "ring_with_action::nauty_interface verbose_level = " << verbose_level << endl;
	}

	if (Variety_object_with_action->f_has_nauty_output) {

		if (f_v) {
			cout << "ring_with_action::nauty_interface f_has_nauty_output" << endl;
		}

		if (f_v) {
			cout << "ring_with_action::nauty_interface "
					"before nauty_interface_with_precomputed_data" << endl;
		}

		nauty_interface_with_precomputed_data(
				Variety_object_with_action,
				Nauty_control,
				Set_stab,
				Canonical_form,
				NO,
				verbose_level - 2);

		if (f_v) {
			cout << "ring_with_action::nauty_interface "
					"after nauty_interface_with_precomputed_data" << endl;
		}
	}
	else {

		if (f_v) {
			cout << "ring_with_action::nauty_interface f_has_nauty_output is false" << endl;
		}

		if (f_v) {
			cout << "ring_with_action::nauty_interface "
					"before nauty_interface_from_scratch" << endl;
		}

		nauty_interface_from_scratch(
				Variety_object_with_action,
				Nauty_control,
				Set_stab,
				Canonical_form,
				NO,
				verbose_level - 2);

		if (f_v) {
			cout << "ring_with_action::nauty_interface "
					"after nauty_interface_from_scratch" << endl;
		}

	}

	if (f_v) {
		cout << "ring_with_action::nauty_interface done" << endl;
	}
}

void ring_with_action::nauty_interface_with_precomputed_data(
		canonical_form::variety_object_with_action *Variety_object_with_action,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		groups::strong_generators *&Set_stab,
		other::data_structures::bitvector *&Canonical_form,
		other::l1_interfaces::nauty_output *&NO,
		int verbose_level)
// Nauty interface with precomputed data
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_with_action::nauty_interface_with_precomputed_data" << endl;
	}

	interfaces::nauty_interface_with_group Nau;


	if (f_v) {
		cout << "ring_with_action::nauty_interface_with_precomputed_data "
				"before Nau.set_stabilizer_in_projective_space_using_precomputed_nauty_data" << endl;
	}


	//
	// Nauty interface with precomputed data:
	//

	Nau.set_stabilizer_in_projective_space_using_precomputed_nauty_data(
			PA->P,
			PA->A,
			Variety_object_with_action->Variety_object->Point_sets->Sets[0],
			Variety_object_with_action->Variety_object->Point_sets->Set_size[0],
			Nauty_control,
			Variety_object_with_action->nauty_output_index_start,
			Variety_object_with_action->Carrying_through,
			Set_stab,
			Canonical_form,
			NO,
			verbose_level);




	if (f_v) {
		cout << "ring_with_action::nauty_interface_with_precomputed_data done" << endl;
	}

}

void ring_with_action::nauty_interface_from_scratch(
		canonical_form::variety_object_with_action *Variety_object_with_action,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		groups::strong_generators *&Set_stab,
		other::data_structures::bitvector *&Canonical_form,
		other::l1_interfaces::nauty_output *&NO,
		int verbose_level)
// Nauty interface without precomputed data
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_with_action::nauty_interface_from_scratch" << endl;
	}

	interfaces::nauty_interface_with_group Nau;


	if (f_v) {
		cout << "ring_with_action::nauty_interface_from_scratch "
				"before Nau.set_stabilizer_in_projective_space_using_nauty" << endl;
	}


	//
	// Nauty interface without precomputed data:
	//


	Nau.set_stabilizer_in_projective_space_using_nauty(
			PA->P,
			PA->A,
			Variety_object_with_action->Variety_object->Point_sets->Sets[0],
			Variety_object_with_action->Variety_object->Point_sets->Set_size[0],
			Nauty_control,
			Set_stab,
			Canonical_form,
			NO,
			verbose_level - 2);

	if (f_v) {
		cout << "ring_with_action::nauty_interface_from_scratch "
				"after Nau.set_stabilizer_in_projective_space_using_nauty" << endl;
	}


	if (f_v) {
		cout << "ring_with_action::nauty_interface_from_scratch done" << endl;
	}

}




}}}

