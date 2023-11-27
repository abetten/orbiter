/*
 * packing_was_activity.cpp
 *
 *  Created on: Apr 3, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {


packing_was_activity::packing_was_activity()
{
	Descr = NULL;
	PW = NULL;

}

packing_was_activity::~packing_was_activity()
{

}



void packing_was_activity::init(
		packing_was_activity_description *Descr,
		packing_was *PW,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_activity::init" << endl;
	}

	packing_was_activity::Descr = Descr;
	packing_was_activity::PW = PW;

	if (f_v) {
		cout << "packing_was_activity::init done" << endl;
	}
}

void packing_was_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (Descr->f_report) {


		if (f_v) {
			cout << "packing_was_activity::perform_activity before PW->report" << endl;
		}

		PW->report(0 /* verbose_level */);

		if (f_v) {
			cout << "packing_was_activity::perform_activity after PW->report" << endl;
		}


	}
	else if (Descr->f_export_reduced_spread_orbits) {


		if (f_v) {
			cout << "packing_was_activity::perform_activity before PW->export_reduced_spread_orbits_csv" << endl;
		}

		int f_original_spread_numbers = true;

		PW->export_reduced_spread_orbits_csv(Descr->export_reduced_spread_orbits_fname_base,
				f_original_spread_numbers, verbose_level);

		if (f_v) {
			cout << "packing_was_activity::perform_activity after PW->export_reduced_spread_orbits_csv" << endl;
		}

	}
	else if (Descr->f_create_graph_on_mixed_orbits) {


		if (f_v) {
			cout << "packing_was_activity::perform_activity f_create_graph_on_mixed_orbits" << endl;
		}

		if (f_v) {
			cout << "packing_was_activity::perform_activity before PW->create_graph_on_mixed_orbits_and_save_to_file" << endl;
		}

		PW->create_graph_on_mixed_orbits_and_save_to_file(
				Descr->create_graph_on_mixed_orbits_orbit_lengths,
				false  /* f_has_user_data */, NULL /* long int *user_data */, 0 /* int user_data_size */,
				verbose_level);

		if (f_v) {
			cout << "packing_was_activity::perform_activity after PW->create_graph_on_mixed_orbits_and_save_to_file" << endl;
		}


		if (f_v) {
			cout << "packing_was_activity::perform_activity f_create_graph_on_mixed_orbits done" << endl;
		}

	}




	if (f_v) {
		cout << "packing_was_activity::perform_activity" << endl;
	}

}

}}}


