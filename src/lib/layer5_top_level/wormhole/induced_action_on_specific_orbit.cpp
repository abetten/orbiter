/*
 * induced_action_on_specific_orbit.cpp
 *
 *  Created on: Sep 23, 2026
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace wormhole {


static long int induced_action_on_specific_orbit_compute_image(
		void *Wormhole,
		int *Elt,
		long int i, int verbose_level);


induced_action_on_specific_orbit::induced_action_on_specific_orbit()
{
	Record_birth();


	//std::string label;

	Orbits = NULL;

	//On_polynomials = NULL;
	//Schreier = NULL;

	Orb = NULL;
	Orbit_of_equations = NULL;

	A = NULL;

	AonHPD = NULL;

	Wormhole_action = NULL;

	A_induced = NULL;

}



induced_action_on_specific_orbit::~induced_action_on_specific_orbit()
{
	Record_death();
}


void induced_action_on_specific_orbit::init_action_on_polynomial_orbit(
		std::string &orbit_object_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "induced_action_on_specific_orbit::init_action_on_polynomial_orbit" << endl;
	}

	label = orbit_object_label;

	Orbits = Get_orbits(orbit_object_label);

#if 0
	orbits::orbits_create *Orbits;

	int f_has_On_polynomials;
	orbits_on_polynomials *On_polynomials;

	int f_has_Of_One_polynomial;
	orbits_on_polynomials *Of_One_polynomial;

#endif


#if 0
	if (!Orbits->f_has_On_polynomials) {
		cout << "induced_action_on_specific_orbit::init_action_on_polynomial_orbit must have orbits on polynomials" << endl;
		exit(1);
	}




	On_polynomials = Orbits->On_polynomials;


	if (!On_polynomials->f_has_Sch) {
		cout << "induced_action_on_specific_orbit::init_action_on_polynomial_orbit we don't have a schreier tree" << endl;
		exit(1);

	}
#endif


	Orb = Orbits->Of_One_polynomial;



	// initialized by orbit_of_one_polynomial:
	//int f_has_Orb;
	//layer4_classification::orbits_schreier::orbit_of_equations *Orb;


	if (!Orb->f_has_Orb) {
		cout << "induced_action_on_specific_orbit::init_action_on_polynomial_orbit must have orbits on one polynomial" << endl;
		exit(1);

	}

	Orbit_of_equations = Orb->Orb;



	A = Orbit_of_equations->A;

	AonHPD = Orbit_of_equations->AonHPD;

	if (f_v) {
		cout << "induced_action_on_specific_orbit::init_action_on_polynomial_orbit "
				"found the orbit_of_equations object for the orbits" << endl;
	}



	Wormhole_action = NEW_OBJECT(layer3_group_actions::induced_actions::wormhole_action);


	Wormhole_action->init(
			A,
			this /*void *Wormhole*/,
			induced_action_on_specific_orbit_compute_image,
			verbose_level);


	if (f_v) {
		cout << "induced_action_on_specific_orbit::init_action_on_polynomial_orbit "
				"before compute_induced_action" << endl;
	}

	A_induced = compute_induced_action(
			A,
			verbose_level);

	if (f_v) {
		cout << "induced_action_on_specific_orbit::init_action_on_polynomial_orbit "
				"after compute_induced_action" << endl;
	}

	Wormhole_action->perm_degree = A_induced->degree;

	if (f_v) {
		cout << "induced_action_on_specific_orbit::init_action_on_polynomial_orbit "
				"created an action of degree " << Wormhole_action->perm_degree << endl;
	}


	if (f_v) {
		cout << "induced_action_on_specific_orbit::init_action_on_polynomial_orbit done" << endl;
	}
}


actions::action *induced_action_on_specific_orbit::compute_induced_action(
		actions::action *A_old,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	actions::action *A;

	if (f_v) {
		cout << "induced_action_on_specific_orbit::compute_induced_action" << endl;
	}
	if (f_v) {
		cout << "induced_action_on_specific_orbit::compute_induced_action A_old=" << A_old->label << endl;
	}
	A = NEW_OBJECT(actions::action);


	A->label = A_old->label + "_OnPolyOrbit";
	A->label_tex = A_old->label_tex + "{\\rm OnPolyOrbit}";


	if (f_v) {
		cout << "the old_action " << A_old->label
				<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	if (A_old->type_G != matrix_group_t) {
		cout << "induced_action_on_specific_orbit::compute_induced_action "
				"old action not of matrix group type" << endl;
		cout << "symmetry group type: ";
		actions::action_global AG;
		AG.action_print_symmetry_group_type(
					cout, A_old->type_G);
		cout << endl;
		exit(1);
	}

	A->type_G = action_by_wormhole_t;
	A->G.Wormhole_action = Wormhole_action;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = 0;

	A->f_has_strong_generators = false;

	A->degree = Orbit_of_equations->used_length;
	//A->base_len = 0;
	if (f_v) {
		cout << "induced_action_on_specific_orbit::compute_induced_action "
				"before init_function_pointers_induced_action" << endl;
	}
	A->ptr = NEW_OBJECT(actions::action_pointer_table);
	A->ptr->init_function_pointers_induced_action();



	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->f_is_linear = false;
	A->dimension = 0;
	//A->dimension = Orbit_of_equations->nb_monomials;

	if (f_v) {
		cout << "induced_action_on_specific_orbit::compute_induced_action "
				"before A->allocate_element_data" << endl;
	}
	A->Group_element->allocate_element_data();


	if (f_v) {
		cout << "induced_action_on_specific_orbit::compute_induced_action "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		A->print_info();
	}
	return A;
}




static long int induced_action_on_specific_orbit_compute_image(
		void *Wormhole,
		int *Elt,
		long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "induced_action_on_specific_orbit_compute_image" << endl;
	}

	induced_action_on_specific_orbit *Induced_action_on_specific_orbit;

	Induced_action_on_specific_orbit = (induced_action_on_specific_orbit *) Wormhole;

	long int j;

	j = Induced_action_on_specific_orbit->Orbit_of_equations->compute_image_of(
			i, Elt, 0 /*verbose_level */);


	if (f_v) {
		cout << "induced_action_on_specific_orbit_compute_image done" << endl;
	}
	return j;
}

}}}


