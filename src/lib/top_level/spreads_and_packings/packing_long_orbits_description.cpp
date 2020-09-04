/*
 * packing_long_orbits_description.cpp
 *
 *  Created on: Sep 3, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


packing_long_orbits_description::packing_long_orbits_description()
{
	f_split = FALSE;
	split_r = 0;
	split_m = 0;


	f_orbit_length = FALSE;
	orbit_length = 0;

	f_clique_size = FALSE;
	clique_size = 0;

	f_list_of_cases_from_file = FALSE;
	//std::string process_list_of_cases_from_file_fname;

	f_solution_path = FALSE;
	//std::string solution_path;

	f_create_graphs = FALSE;

	f_solve = FALSE;

	f_read_solutions = FALSE;
}

packing_long_orbits_description::~packing_long_orbits_description()
{
}

int packing_long_orbits_description::read_arguments(int argc, const char **argv,
	int verbose_level)
{
	int i;



	cout << "packing_long_orbits_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {


		if (strcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_r = atoi(argv[++i]);
			split_m = atoi(argv[++i]);
			cout << "-split " << split_r << " " << split_m << " " << endl;
		}


		else if (strcmp(argv[i], "-orbit_length") == 0) {
			f_orbit_length = TRUE;
			orbit_length = atoi(argv[++i]);
			cout << "-orbit_length " << orbit_length << " " << endl;
		}

		else if (strcmp(argv[i], "-clique_size") == 0) {
			f_clique_size = TRUE;
			clique_size = atoi(argv[++i]);
			cout << "-clique_size " << clique_size << " " << endl;
		}

		else if (strcmp(argv[i], "-list_of_cases_from_file") == 0) {
			f_list_of_cases_from_file = TRUE;
			list_of_cases_from_file_fname.assign(argv[++i]);
			cout << "-list_of_cases_from_file "
				<< list_of_cases_from_file_fname << " "
				<< endl;
		}




		else if (strcmp(argv[i], "-solution_path") == 0) {
			f_solution_path = TRUE;
			solution_path.assign(argv[++i]);
			cout << "-solution_path " << solution_path << endl;
		}

		else if (strcmp(argv[i], "-create_graphs") == 0) {
			f_create_graphs = TRUE;
			cout << "-create_graphs " << endl;
		}

		else if (strcmp(argv[i], "-solve") == 0) {
			f_solve = TRUE;
			cout << "-solve " << endl;
		}

		else if (strcmp(argv[i], "-read_solutions") == 0) {
			f_read_solutions = TRUE;
			cout << "-read_solutions " << endl;
		}

		else if (strcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "ignoring argument " << argv[i] << endl;
		}
	} // next i


	cout << "packing_long_orbits_description::read_arguments done" << endl;
	return i;
}



}}

