/*
 * action_on_forms_activity.cpp
 *
 *  Created on: Oct 24, 2022
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


action_on_forms_activity::action_on_forms_activity()
{
	Descr = NULL;

	AF = NULL;


}

action_on_forms_activity::~action_on_forms_activity()
{

}

void action_on_forms_activity::init(
		action_on_forms_activity_description *Descr,
		apps_algebra::action_on_forms *AF,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_forms_activity::init" << endl;
	}

	action_on_forms_activity::Descr = Descr;
	action_on_forms_activity::AF = AF;

	if (f_v) {
		cout << "action_on_forms_activity::init done" << endl;
	}
}


void action_on_forms_activity::perform_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_forms_activity::perform_activity" << endl;
	}

	if (Descr->f_algebraic_normal_form) {
		do_algebraic_normal_form(verbose_level);
	}
	else if (Descr->f_orbits_on_functions) {
		do_orbits_on_functions(verbose_level);
	}
	else if (Descr->f_associated_set_in_plane) {
		do_associated_set_in_plane(verbose_level);
	}
	else if (Descr->f_differential_uniformity) {
		do_differential_uniformity(verbose_level);
	}


	if (f_v) {
		cout << "action_on_forms_activity::perform_activity done" << endl;
	}
}

void action_on_forms_activity::do_algebraic_normal_form(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_forms_activity::do_algebraic_normal_form" << endl;
	}

	int *func;
	int len;

	int *coeff;
	int nb_coeff;

	Get_int_vector_from_label(Descr->algebraic_normal_form_input, func, len, 0 /* verbose_level */);

	AF->PF->algebraic_normal_form(
			func, len,
			coeff, nb_coeff,
			verbose_level);

	if (f_v) {
		cout << "action_on_forms_activity::do_algebraic_normal_form done" << endl;
	}
}

void action_on_forms_activity::do_orbits_on_functions(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_forms_activity::do_orbits_on_functions" << endl;
	}

	int *The_functions;
	int nb_functions;
	int len;

	Get_matrix(Descr->orbits_on_functions_input, The_functions, nb_functions, len);

	if (f_v) {
		cout << "action_on_forms_activity::do_orbits_on_functions before AF->orbits_on_functions" << endl;
	}
	AF->orbits_on_functions(The_functions, nb_functions, len,
			verbose_level);
	if (f_v) {
		cout << "action_on_forms_activity::do_orbits_on_functions after AF->orbits_on_functions" << endl;
	}

	if (f_v) {
		cout << "action_on_forms_activity::do_orbits_on_functions done" << endl;
	}

}

void action_on_forms_activity::do_associated_set_in_plane(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_forms_activity::do_associated_set_in_plane" << endl;
	}

	int *func;
	int len;
	long int *Rk;

	Get_int_vector_from_label(Descr->associated_set_in_plane_input, func, len, 0 /* verbose_level */);

	AF->associated_set_in_plane(func, len,
			Rk, verbose_level);

	if (f_v) {
		cout << "action_on_forms_activity::do_associated_set_in_plane done" << endl;
	}
}

void action_on_forms_activity::do_differential_uniformity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_forms_activity::do_differential_uniformity" << endl;
	}

	int *func;
	int len;

	Get_int_vector_from_label(Descr->differential_uniformity_input, func, len, 0 /* verbose_level */);

	AF->differential_uniformity(func, len, verbose_level);

	if (f_v) {
		cout << "action_on_forms_activity::do_differential_uniformity done" << endl;
	}
}



}}}




