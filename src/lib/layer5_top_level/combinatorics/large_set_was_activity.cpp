/*
 * large_set_was_activity.cpp
 *
 *  Created on: May 27, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {



large_set_was_activity::large_set_was_activity()
{
	Descr = NULL;
	LSW = NULL;
}


large_set_was_activity::~large_set_was_activity()
{
}


void large_set_was_activity::perform_activity(
		large_set_was_activity_description *Descr,
		large_set_was *LSW, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "large_set_was_activity::perform_activity" << endl;
	}

	large_set_was_activity::Descr = Descr;
	large_set_was_activity::LSW = LSW;


	if (Descr->f_normalizer_on_orbits_of_a_given_length) {

		LSW->do_normalizer_on_orbits_of_a_given_length(
				Descr->normalizer_on_orbits_of_a_given_length_length,
				Descr->normalizer_on_orbits_of_a_given_length_nb_orbits,
				Descr->normalizer_on_orbits_of_a_given_length_control,
				verbose_level);
	}

	if (Descr->f_create_graph_on_orbits_of_length) {

		LSW->create_graph_on_orbits_of_length(
				Descr->create_graph_on_orbits_of_length_fname,
				Descr->create_graph_on_orbits_of_length_length,
				verbose_level);
	}

	if (Descr->f_create_graph_on_orbits_of_length_based_on_N_orbits) {

		LSW->create_graph_on_orbits_of_length_based_on_N_orbits(
				Descr->create_graph_on_orbits_of_length_based_on_N_orbits_fname_mask,
				Descr->create_graph_on_orbits_of_length_based_on_N_orbits_length,
				Descr->create_graph_on_orbits_of_length_based_on_N_nb_N_orbits_preselected,
				Descr->create_graph_on_orbits_of_length_based_on_N_orbits_r,
				Descr->create_graph_on_orbits_of_length_based_on_N_orbits_m,
				verbose_level);

	}

	if (Descr->f_read_solution_file) {

		long int *starter_set = NULL;
		int starter_set_sz = 0;

		LSW->read_solution_file(
				Descr->read_solution_file_name,
				starter_set,
				starter_set_sz,
				Descr->read_solution_file_orbit_length,
				verbose_level);
	}


	if (f_v) {
		cout << "large_set_was_activity::perform_activity done" << endl;
	}

}



}}}


