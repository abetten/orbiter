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
		do_compute_group(
				Descr->f_output_fname_base,
				Descr->output_fname_base,
				verbose_level);
	}
	if (Descr->f_report) {
		Input_Vo[0].do_report(verbose_level);
	}
	if (Descr->f_singular_points) {
		do_singular_points(verbose_level);
	}


	if (f_v) {
		cout << "variety_activity::perform_activity done" << endl;
	}
}

void variety_activity::do_compute_group(
		int f_has_output_fname_base,
		std::string &output_fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::do_compute_group" << endl;
	}
	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"nb_input_Vo = " << nb_input_Vo << endl;
	}

	if (nb_input_Vo == 0) {
		cout << "variety_activity::do_compute_group "
				"nb_input_Vo == 0" << endl;
		exit(1);
	}

	std::string fname_base;

	if (f_has_output_fname_base) {
		fname_base = output_fname_base;
	}
	else {
		fname_base = Input_Vo[0].Variety_object->label_txt + "_c";
	}

	canonical_form::canonical_form_classifier *Classifier;

	Classifier = NEW_OBJECT(canonical_form::canonical_form_classifier);


	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"before Classifier->init_direct" << endl;
	}

	Classifier->init_direct(
			nb_input_Vo,
			Input_Vo,
			fname_base,
			verbose_level);

	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"after Classifier->init_direct" << endl;
	}



	canonical_form::canonical_form_global Canonical_form_global;

	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"before Canonical_form_global.compute_group_and_tactical_decomposition" << endl;
	}
	Canonical_form_global.compute_group_and_tactical_decomposition(
			Classifier,
			Input_Vo,
			fname_base,
			verbose_level);
	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"after Canonical_form_global.compute_group_and_tactical_decomposition" << endl;
	}

	FREE_OBJECT(Classifier);

	if (f_v) {
		cout << "variety_activity::do_compute_group done" << endl;
	}
}



void variety_activity::do_singular_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::do_singular_points" << endl;
	}

	geometry::projective_space *P;


	P = Input_Vo[0].Variety_object->Projective_space;

	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"before getting Poly_ring" << endl;
	}
	ring_theory::homogeneous_polynomial_domain *Poly_ring = Input_Vo[0].Variety_object->Ring;
	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"after getting Poly_ring" << endl;
	}


	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"before Poly_ring->compute_singular_points_projectively" << endl;
	}
	Poly_ring->compute_singular_points_projectively(
			P,
			Input_Vo[0].Variety_object->eqn,
			Input_Vo[0].Variety_object->Singular_points,
			verbose_level);
	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"after Poly_ring->compute_singular_points_projectively" << endl;
	}

	Input_Vo[0].Variety_object->f_has_singular_points = true;

	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"number of singular points = " << Input_Vo[0].Variety_object->Singular_points.size() << endl;
	}
	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"The singular points are: " << endl;
		Lint_vec_stl_print_fully(cout, Input_Vo[0].Variety_object->Singular_points);
		cout << endl;
	}

	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = Input_Vo[0].Variety_object->label_txt + "_singular_pts.csv";

	Fio.Csv_file_support->vector_lint_write_csv(
			fname,
			Input_Vo[0].Variety_object->Singular_points);

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "variety_activity::do_singular_points done" << endl;
	}
}


}}}
