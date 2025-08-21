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
	Record_birth();
	Descr = NULL;
	SCW = NULL;
}

classification_of_cubic_surfaces_with_double_sixes_activity::~classification_of_cubic_surfaces_with_double_sixes_activity()
{
	Record_death();
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
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"-report" << endl;
			cout << "SCW->Surf->n = " << SCW->Surf->n << endl;
		}
		report(Descr->report_options, verbose_level);
	}
	else if (Descr->f_stats) {
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"-stats" << endl;
		}
		SCW->stats(Descr->stats_prefix);
	}
	else if (Descr->f_identify_Eckardt) {
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"before SCW->identify_Eckardt_and_print_table" << endl;
		}
		SCW->identify_Eckardt_and_print_table(verbose_level);
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"after SCW->identify_Eckardt_and_print_table" << endl;
		}

	}
	else if (Descr->f_identify_F13) {
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"before SCW->identify_F13_and_print_table" << endl;
		}
		SCW->identify_F13_and_print_table(verbose_level);
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"after SCW->identify_F13_and_print_table" << endl;
		}
	}
	else if (Descr->f_identify_Bes) {
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"before SCW->identify_Bes_and_print_table" << endl;
		}
		SCW->identify_Bes_and_print_table(verbose_level);
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"after SCW->identify_Bes_and_print_table" << endl;
		}
	}
	else if (Descr->f_identify_general_abcd) {
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"before SCW->identify_general_abcd_and_print_table" << endl;
		}
		SCW->identify_general_abcd_and_print_table(verbose_level);
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"after SCW->identify_general_abcd_and_print_table" << endl;
		}
	}
	else if (Descr->f_isomorphism_testing) {

		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"before SCW->test_isomorphism" << endl;
		}
		SCW->test_isomorphism(
				Descr->isomorphism_testing_surface1_label,
				Descr->isomorphism_testing_surface2_label,
				verbose_level);

		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"after SCW->test_isomorphism" << endl;
		}


	}
	else if (Descr->f_recognize) {
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"before SCW->recognition" << endl;
		}
		SCW->recognition(
				Descr->recognize_surface_label,
				verbose_level);
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"after SCW->recognition" << endl;
		}

	}
	else if (Descr->f_create_source_code) {
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"before SCW->Surface_repository->generate_source_code" << endl;
		}
		SCW->Surface_repository->generate_source_code(verbose_level);
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"after SCW->Surface_repository->generate_source_code" << endl;
		}
	}
	else if (Descr->f_sweep_Cayley) {
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"before SCW->sweep_Cayley" << endl;
		}
		SCW->sweep_Cayley(verbose_level);
		if (f_v) {
			cout << "classification_of_cubic_surfaces_with_double_sixes_activity::perform_activity "
					"after SCW->sweep_Cayley" << endl;
		}
	}

}


void classification_of_cubic_surfaces_with_double_sixes_activity::report(
		std::string &options,
		//poset_classification::poset_classification_report_options
		//	*report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::report" << endl;
	}


	//poset_classification::poset_classification_report_options *report_options = NULL;
	//int f_with_stabilizers = true;

	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::report "
				"before SCW->create_report" << endl;
		cout << "SCW->Surf->n = " << SCW->Surf->n << endl;
	}
	SCW->create_report(
			options,
			//f_with_stabilizers,
			//report_options,
			verbose_level - 1);
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::report "
				"after SCW->create_report" << endl;
	}
	if (f_v) {
		cout << "classification_of_cubic_surfaces_with_double_sixes_activity::report done" << endl;
	}
}








}}}}



