/*
 * classification_of_cubic_surfaces_with_double_sixes_activity.cpp
 *
 *  Created on: Apr 1, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_double_sixes {


classification_of_cubic_surfaces_with_double_sixes_activity::classification_of_cubic_surfaces_with_double_sixes_activity()
{
	Descr = NULL;
	SCW = NULL;
}

classification_of_cubic_surfaces_with_double_sixes_activity::~classification_of_cubic_surfaces_with_double_sixes_activity()
{
}

void classification_of_cubic_surfaces_with_double_sixes_activity::init(
		classification_of_cubic_surfaces_with_double_sixes_activity_description
			*Descr,
		surface_classify_wedge *SCW,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::init" << endl;
	}

	classification_of_cubic_surfaces_with_double_sixes_activity::Descr = Descr;
	classification_of_cubic_surfaces_with_double_sixes_activity::SCW = SCW;

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::init done" << endl;
	}
}

void classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity" << endl;
	}
	if (Descr->f_report) {
		cout << "f_report" << endl;
		cout << "SCW->Surf->n = " << SCW->Surf->n << endl;
		report(Descr->report_options, verbose_level);
	}
	else if (Descr->f_identify_Eckardt) {
		do_surface_identify_Eckardt(verbose_level);
	}
	else if (Descr->f_identify_F13) {
		do_surface_identify_F13(verbose_level);
	}
	else if (Descr->f_identify_Bes) {
		do_surface_identify_Bes(verbose_level);
	}
	else if (Descr->f_identify_general_abcd) {
		do_surface_identify_general_abcd(verbose_level);
	}
	else if (Descr->f_isomorphism_testing) {
		do_surface_isomorphism_testing(
				Descr->isomorphism_testing_surface1,
				Descr->isomorphism_testing_surface2,
				verbose_level);
	}
	else if (Descr->f_recognize) {
		do_recognize(Descr->recognize_surface, verbose_level);
	}
	else if (Descr->f_create_source_code) {
		do_write_source_code(verbose_level);
	}
	else if (Descr->f_sweep_Cayley) {
		do_sweep_Cayley(verbose_level);
	}

}


void classification_of_cubic_surfaces_with_double_sixes_activity::report(
		poset_classification::poset_classification_report_options
			*report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::report" << endl;
	}

	if (!orbiter_kernel_system::Orbiter->f_draw_options) {
		cout << "for a report of the surfaces, please use -draw_options ... -end" << endl;
		exit(1);
	}

	int f_with_stabilizers = true;

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::report "
				"before SCW->create_report" << endl;
		cout << "SCW->Surf->n = " << SCW->Surf->n << endl;
	}
	SCW->create_report(
			f_with_stabilizers,
			orbiter_kernel_system::Orbiter->draw_options,
			report_options,
			verbose_level - 1);
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::report "
				"after SCW->create_report" << endl;
	}
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::report done" << endl;
	}
}

void classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_Eckardt(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_Eckardt" << endl;
	}


	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_Eckardt "
				"before SCW->identify_Eckardt_and_print_table" << endl;
	}
	SCW->identify_Eckardt_and_print_table(verbose_level);
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_Eckardt "
				"after SCW->identify_Eckardt_and_print_table" << endl;
	}


	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_Eckardt done" << endl;
	}
}

void classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_F13(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_F13" << endl;
	}


	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_F13 "
				"before SCW->identify_F13_and_print_table" << endl;
	}
	SCW->identify_F13_and_print_table(verbose_level);
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_F13 "
				"after SCW->identify_F13_and_print_table" << endl;
	}

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_F13 done" << endl;
	}
}

void classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_Bes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_Bes" << endl;
	}

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_Bes "
				"before SCW->identify_Bes_and_print_table" << endl;
	}
	SCW->identify_Bes_and_print_table(verbose_level);
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_Bes "
				"after SCW->identify_Bes_and_print_table" << endl;
	}


	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_Bes done" << endl;
	}
}

void classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_general_abcd(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_general_abcd" << endl;
	}

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_general_abcd "
				"before SCW->identify_general_abcd_and_print_table" << endl;
	}
	SCW->identify_general_abcd_and_print_table(verbose_level);
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_general_abcd "
				"after SCW->identify_general_abcd_and_print_table" << endl;
	}

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_identify_general_abcd done" << endl;
	}
}

void classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_isomorphism_testing(
		cubic_surfaces_in_general::surface_create_description
			*surface_descr_isomorph1,
		cubic_surfaces_in_general::surface_create_description
			*surface_descr_isomorph2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_isomorphism_testing" << endl;
	}


	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_isomorphism_testing "
				"before SCW->test_isomorphism" << endl;
	}
	SCW->test_isomorphism(
			surface_descr_isomorph1,
			surface_descr_isomorph2,
			verbose_level);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_isomorphism_testing "
				"after SCW->test_isomorphism" << endl;
	}
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_surface_isomorphism_testing done" << endl;
	}
}

void classification_of_cubic_surfaces_with_double_sixes_activity::do_recognize(
		cubic_surfaces_in_general::surface_create_description
			*surface_descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_recognize" << endl;
	}


	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_recognize "
				"before SCW->recognition" << endl;
	}
	SCW->recognition(
			surface_descr,
			verbose_level);
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_recognize "
				"after SCW->recognition" << endl;
	}
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_recognize done" << endl;
	}
}

void classification_of_cubic_surfaces_with_double_sixes_activity::do_write_source_code(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_write_source_code" << endl;
	}


	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_write_source_code "
				"before SCW->Surface_repository->generate_source_code" << endl;
	}
	SCW->Surface_repository->generate_source_code(verbose_level);
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_write_source_code "
				"after SCW->Surface_repository->generate_source_code" << endl;
	}
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_write_source_code done" << endl;
	}
}

void classification_of_cubic_surfaces_with_double_sixes_activity::do_sweep_Cayley(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_sweep_Cayley" << endl;
	}


	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_sweep_Cayley "
				"before SCW->sweep_Cayley" << endl;
	}
	SCW->sweep_Cayley(verbose_level);
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_sweep_Cayley "
				"after SCW->sweep_Cayley" << endl;
	}
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::do_sweep_Cayley done" << endl;
	}
}




}}}}



