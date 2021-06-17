/*
 * large_set_was_activity_description.cpp
 *
 *  Created on: May 27, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


large_set_was_activity_description::large_set_was_activity_description()
{
	f_normalizer_on_orbits_of_a_given_length = FALSE;
	normalizer_on_orbits_of_a_given_length_length = 0;
	normalizer_on_orbits_of_a_given_length_nb_orbits = 0;
	normalizer_on_orbits_of_a_given_length_control = NULL;

	f_create_graph_on_orbits_of_length = FALSE;
	//std::string create_graph_on_orbits_of_length_fname;
	create_graph_on_orbits_of_length_length = 0;

	f_create_graph_on_orbits_of_length_based_on_N_orbits = FALSE;
	//std::string create_graph_on_orbits_of_length_based_on_N_orbits_fname;
	create_graph_on_orbits_of_length_based_on_N_orbits_length = 0;

	f_read_solution_file = FALSE;
	read_solution_file_orbit_length = 0;
	//std::string read_solution_file_name;
}

large_set_was_activity_description::~large_set_was_activity_description()
{

}


int large_set_was_activity_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "large_set_was_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {


		if (stringcmp(argv[i], "-normalizer_on_orbits_of_a_given_length") == 0) {
			f_normalizer_on_orbits_of_a_given_length = TRUE;
			normalizer_on_orbits_of_a_given_length_length = strtoi(argv[++i]);
			normalizer_on_orbits_of_a_given_length_nb_orbits = strtoi(argv[++i]);

			cout << "-normalizer_on_orbits_of_a_given_length reading poset_classification_control options" << endl;

			normalizer_on_orbits_of_a_given_length_control = NEW_OBJECT(poset_classification_control);
			i += normalizer_on_orbits_of_a_given_length_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}


			cout << "-normalizer_on_orbits_of_a_given_length " << normalizer_on_orbits_of_a_given_length_length
					<< " " << normalizer_on_orbits_of_a_given_length_nb_orbits
					<< endl;
		}
		else if (stringcmp(argv[i], "-create_graph_on_orbits_of_length") == 0) {
			f_create_graph_on_orbits_of_length = TRUE;
			create_graph_on_orbits_of_length_fname.assign(argv[++i]);
			create_graph_on_orbits_of_length_length = strtoi(argv[++i]);
			cout << "-create_graph_on_orbits_of_length "
					<< " " << create_graph_on_orbits_of_length_fname
					<< " " << create_graph_on_orbits_of_length_length
					<< endl;
		}
		else if (stringcmp(argv[i], "-create_graph_on_orbits_of_length_based_on_N_orbits") == 0) {
			f_create_graph_on_orbits_of_length_based_on_N_orbits = TRUE;
			create_graph_on_orbits_of_length_based_on_N_orbits_fname.assign(argv[++i]);
			create_graph_on_orbits_of_length_based_on_N_orbits_length = strtoi(argv[++i]);
			cout << "-create_graph_on_orbits_of_length_based_on_N_orbits "
					<< " " << create_graph_on_orbits_of_length_based_on_N_orbits_fname
					<< " " << create_graph_on_orbits_of_length_based_on_N_orbits_length
					<< endl;
		}
		else if (stringcmp(argv[i], "-read_solution_file") == 0) {
			f_read_solution_file = TRUE;
			read_solution_file_orbit_length = strtoi(argv[++i]);
			read_solution_file_name.assign(argv[++i]);
			cout << "-read_solution_file "
					<< read_solution_file_orbit_length
					<< " " << read_solution_file_name
					<< endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			break;
		}
	} // next i
	cout << "large_set_was_activity_description::read_arguments done" << endl;
	return i + 1;
}




}}




