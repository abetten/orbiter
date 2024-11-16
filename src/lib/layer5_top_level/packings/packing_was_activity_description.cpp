/*
 * packing_was_activity_description.cpp
 *
 *  Created on: Apr 3, 2021
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {



packing_was_activity_description::packing_was_activity_description()
{
	f_report = false;

	f_export_reduced_spread_orbits = false;
	//std::string export_reduced_spread_orbits_fname_base;

	f_create_graph_on_mixed_orbits = false;
	//std::string create_graph_on_mixed_orbits_orbit_lengths;
}

packing_was_activity_description::~packing_was_activity_description()
{

}

int packing_was_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	cout << "packing_was_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		cout << "packing_was_activity_description::read_arguments, "
				"next argument is " << argv[i] << endl;


		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			cout << "-report" << endl;
		}
		else if (ST.stringcmp(argv[i], "-export_reduced_spread_orbits") == 0) {
			f_export_reduced_spread_orbits = true;
			export_reduced_spread_orbits_fname_base.assign(argv[++i]);
			cout << "-export_reduced_spread_orbits "
					<< export_reduced_spread_orbits_fname_base << endl;
		}
		else if (ST.stringcmp(argv[i], "-create_graph_on_mixed_orbits") == 0) {
			f_create_graph_on_mixed_orbits = true;
			create_graph_on_mixed_orbits_orbit_lengths.assign(argv[++i]);
			cout << "-create_graph_on_mixed_orbits "
					<< create_graph_on_mixed_orbits_orbit_lengths << endl;
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "packing_was_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "packing_was_activity_description::read_arguments looping, i=" << i << endl;
	} // next i
	cout << "packing_was_activity_description::read_arguments done" << endl;
	return i + 1;
}

void packing_was_activity_description::print()
{
	if (f_report) {
		cout << "-report" << endl;
	}
	if (f_export_reduced_spread_orbits) {
		cout << "-export_reduced_spread_orbits "
				<< export_reduced_spread_orbits_fname_base << endl;
	}
	if (f_create_graph_on_mixed_orbits) {
		cout << "-create_graph_on_mixed_orbits "
				<< create_graph_on_mixed_orbits_orbit_lengths << endl;
	}
}




}}}



