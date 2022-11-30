/*
 * packing_long_orbits_description.cpp
 *
 *  Created on: Sep 3, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {


packing_long_orbits_description::packing_long_orbits_description()
{
	f_split = FALSE;
	split_r = 0;
	split_m = 0;


	f_orbit_length = FALSE;
	orbit_length = 0;

	f_mixed_orbits = FALSE;
	//std::string mixed_orbits_length_text;


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

int packing_long_orbits_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;



	cout << "packing_long_orbits_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {


		if (ST.stringcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_r = ST.strtoi(argv[++i]);
			split_m = ST.strtoi(argv[++i]);
			cout << "-split " << split_r << " " << split_m << " " << endl;
		}


		else if (ST.stringcmp(argv[i], "-orbit_length") == 0) {
			f_orbit_length = TRUE;
			orbit_length = ST.strtoi(argv[++i]);
			cout << "-orbit_length " << orbit_length << " " << endl;
		}

		else if (ST.stringcmp(argv[i], "-mixed_orbits") == 0) {
			f_mixed_orbits = TRUE;
			mixed_orbits_length_text.assign(argv[++i]);
			cout << "-mixed_orbits " << mixed_orbits_length_text << " " << endl;
		}


		else if (ST.stringcmp(argv[i], "-list_of_cases_from_file") == 0) {
			f_list_of_cases_from_file = TRUE;
			list_of_cases_from_file_fname.assign(argv[++i]);
			cout << "-list_of_cases_from_file "
				<< list_of_cases_from_file_fname << " "
				<< endl;
		}




		else if (ST.stringcmp(argv[i], "-solution_path") == 0) {
			f_solution_path = TRUE;
			solution_path.assign(argv[++i]);
			cout << "-solution_path " << solution_path << endl;
		}

		else if (ST.stringcmp(argv[i], "-create_graphs") == 0) {
			f_create_graphs = TRUE;
			cout << "-create_graphs " << endl;
		}

		else if (ST.stringcmp(argv[i], "-solve") == 0) {
			f_solve = TRUE;
			cout << "-solve " << endl;
		}

		else if (ST.stringcmp(argv[i], "-read_solutions") == 0) {
			f_read_solutions = TRUE;
			cout << "-read_solutions " << endl;
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "packing_long_orbits_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i


	cout << "packing_long_orbits_description::read_arguments done" << endl;
	return i + 1;
}

void packing_long_orbits_description::print()
{
	if (f_split) {
		cout << "-split " << split_r << " " << split_m << " " << endl;
	}


	if (f_orbit_length) {
		cout << "-orbit_length " << orbit_length << " " << endl;
	}

	if (f_mixed_orbits) {
		cout << "-mixed_orbits " << mixed_orbits_length_text << " " << endl;
	}


	if (f_list_of_cases_from_file) {
		cout << "-list_of_cases_from_file "
			<< list_of_cases_from_file_fname << " "
			<< endl;
	}




	if (f_solution_path) {
		cout << "-solution_path " << solution_path << endl;
	}

	if (f_create_graphs) {
		cout << "-create_graphs " << endl;
	}

	if (f_solve) {
		cout << "-solve " << endl;
	}

	if (f_read_solutions) {
		cout << "-read_solutions " << endl;
	}

}



}}}

