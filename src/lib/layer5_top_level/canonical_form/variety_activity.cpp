/*
 * variety_activity.cpp
 *
 *  Created on: Jul 15, 2024
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {


variety_activity::variety_activity()
{
	Descr = NULL;

	nb_input_Vo = 0;

	Input_Vo = NULL;

}

variety_activity::~variety_activity()
{

}

void variety_activity::init(
		variety_activity_description *Descr,
		int nb_input_Vo,
		canonical_form::variety_object_with_action *Input_Vo,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::init" << endl;
	}
	if (f_v) {
		cout << "variety_activity::init nb_input_Vo = " << nb_input_Vo << endl;
	}

	variety_activity::Descr = Descr;

	variety_activity::nb_input_Vo = nb_input_Vo;
	variety_activity::Input_Vo = Input_Vo;

	if (f_v) {
		cout << "variety_activity::init done" << endl;
	}
}


void variety_activity::perform_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::perform_activity" << endl;
	}

	if (Descr->f_compute_group) {
		do_compute_group(verbose_level);
	}
	if (Descr->f_report) {
		do_report(verbose_level);
	}


	if (f_v) {
		cout << "variety_activity::perform_activity done" << endl;
	}
}

void variety_activity::do_compute_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::do_compute_group" << endl;
	}
	if (f_v) {
		cout << "variety_activity::do_compute_group nb_input_Vo = " << nb_input_Vo << endl;
	}

	if (nb_input_Vo == 0) {
		cout << "variety_activity::do_compute_group nb_input_Vo == 0" << endl;
		exit(1);
	}

	canonical_form::canonical_form_classifier *Classifier;

	Classifier = NEW_OBJECT(canonical_form::canonical_form_classifier);

	if (f_v) {
		cout << "variety_activity::do_compute_group before getting PA" << endl;
	}
	projective_geometry::projective_space_with_action *PA = Input_Vo[0].PA;
	if (f_v) {
		cout << "variety_activity::do_compute_group after getting PA" << endl;
	}

	if (f_v) {
		cout << "variety_activity::do_compute_group before getting Poly_ring" << endl;
	}
	ring_theory::homogeneous_polynomial_domain *Poly_ring = Input_Vo[0].Variety_object->Ring;
	if (f_v) {
		cout << "variety_activity::do_compute_group after getting Poly_ring" << endl;
	}

	if (f_v) {
		cout << "variety_activity::do_compute_group before Classifier->init_direct" << endl;
	}

	Classifier->init_direct(
			PA,
			Poly_ring,
			nb_input_Vo,
			Input_Vo,
			verbose_level);

	if (f_v) {
		cout << "variety_activity::do_compute_group after Classifier->init_direct" << endl;
	}


	canonical_form::classification_of_varieties_nauty *Nauty;

	Nauty = NEW_OBJECT(canonical_form::classification_of_varieties_nauty);



	Classifier->Classification_of_varieties_nauty = Nauty;

	//Classifier->Output_nauty = Nauty;

	std::string fname_base;


	fname_base = Input_Vo[0].Variety_object->label_txt + "_c";

	if (f_v) {
		cout << "variety_activity::do_compute_group before Nauty->init" << endl;
	}
	Nauty->init(
			nb_input_Vo,
			Input_Vo,
			fname_base,
			Classifier,
			verbose_level);
	if (f_v) {
		cout << "variety_activity::do_compute_group after Nauty->init" << endl;
	}


	if (f_v) {
		cout << "variety_activity::do_compute_group before Nauty->classify_nauty" << endl;
	}
	Nauty->classify_nauty(verbose_level);
	if (f_v) {
		cout << "variety_activity::do_compute_group after Nauty->classify_nauty" << endl;
	}


	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"before Nauty->write_classification_by_nauty_csv" << endl;
	}
	Nauty->write_classification_by_nauty_csv(
			fname_base,
			verbose_level);
	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"after Nauty->write_classification_by_nauty_csv" << endl;
	}


	FREE_OBJECT(Classifier);

	if (f_v) {
		cout << "variety_activity::do_compute_group done" << endl;
	}
}


void variety_activity::do_report(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::do_report" << endl;
	}

	if (f_v) {
		cout << "variety_activity::do_report done" << endl;
	}
}


}}}
