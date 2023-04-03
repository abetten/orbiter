/*
 * large_set_was_description.cpp
 *
 *  Created on: May 27, 2021
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


large_set_was_description::large_set_was_description()
{

	f_H = false;
	//std::string H_go;
	//std::string H_generators_text;

	f_N = false;
	//std::string N_go;
	//std::string N_generators_text;

	f_report = false;

	f_prefix = false;
	//std::string prefix;

	f_selected_orbit_length = false;
	selected_orbit_length = 0;

}



large_set_was_description::~large_set_was_description()
{
}

int large_set_was_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;



	cout << "large_set_was_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-H") == 0) {
			f_H = true;
			H_go.assign(argv[++i]);
			H_generators_text.assign(argv[++i]);
			cout << "-H " << H_go
					<< " " << H_generators_text
					<< endl;
		}

		else if (ST.stringcmp(argv[i], "-N") == 0) {
			f_N = true;
			N_go.assign(argv[++i]);
			N_generators_text.assign(argv[++i]);
			cout << "-N " << N_go
					<< " " << N_generators_text
					<< endl;
		}
		else if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			cout << "-report " << endl;
		}
		else if (ST.stringcmp(argv[i], "-prefix") == 0) {
			f_prefix = true;
			prefix.assign(argv[++i]);
			cout << "-prefix " << prefix << endl;
		}
		else if (ST.stringcmp(argv[i], "-selected_orbit_length") == 0) {
			f_selected_orbit_length = true;
			selected_orbit_length = ST.strtoi(argv[++i]);
			cout << "-selected_orbit_length " << selected_orbit_length << endl;
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "ignoring argument " << argv[i] << endl;
		}
	} // next i

	cout << "large_set_was_description::read_arguments done" << endl;
	return i + 1;
}

void large_set_was_description::print()
{
	if (f_H) {
		cout << "-H " << H_go
				<< " " << H_generators_text
				<< endl;
	}

	if (f_N) {
		cout << "-N " << N_go
				<< " " << N_generators_text
				<< endl;
	}
	if (f_report) {
		cout << "-report " << endl;
	}
	if (f_prefix) {
		cout << "-prefix " << prefix << endl;
	}
	if (f_selected_orbit_length) {
		cout << "-selected_orbit_length " << selected_orbit_length << endl;
	}
}



}}}
