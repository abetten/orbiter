/*
 * quartic_curve_activity_description.cpp
 *
 *  Created on: May 21, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace quartic_curves {


quartic_curve_activity_description::quartic_curve_activity_description()
{

	f_report = false;

	f_export_something = false;
	//std::string export_something_what;

	f_create_surface = false;

	f_extract_orbit_on_bitangents_by_length = false;
	extract_orbit_on_bitangents_by_length_length = 0;

	f_extract_specific_orbit_on_bitangents_by_length = false;
	extract_specific_orbit_on_bitangents_by_length_length = 0;
	extract_specific_orbit_on_bitangents_by_length_index = 0;


	f_extract_specific_orbit_on_kovalevski_points_by_length = false;
	extract_specific_orbit_on_kovalevski_points_by_length_length = 0;
	extract_specific_orbit_on_kovalevski_points_by_length_index = 0;

}

quartic_curve_activity_description::~quartic_curve_activity_description()
{
}

int quartic_curve_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	cout << "quartic_curve_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			if (f_v) {
				cout << "-report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_something") == 0) {
			f_export_something = true;
			export_something_what.assign(argv[++i]);
			if (f_v) {
				cout << "-export_something " << export_something_what << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-create_surface") == 0) {
			f_create_surface = true;
			if (f_v) {
				cout << "-create_surface " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-extract_orbit_on_bitangents_by_length") == 0) {
			f_extract_orbit_on_bitangents_by_length = true;
			extract_orbit_on_bitangents_by_length_length = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-extract_orbit_on_bitangents_by_length " << extract_orbit_on_bitangents_by_length_length << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-extract_specific_orbit_on_bitangents_by_length") == 0) {
			f_extract_specific_orbit_on_bitangents_by_length = true;
			extract_specific_orbit_on_bitangents_by_length_length = ST.strtoi(argv[++i]);
			extract_specific_orbit_on_bitangents_by_length_index = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-extract_specific_orbit_on_bitangents_by_length "
						<< extract_specific_orbit_on_bitangents_by_length_length << " "
						<< extract_specific_orbit_on_bitangents_by_length_index << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-extract_specific_orbit_on_kovalevski_points_by_length") == 0) {
			f_extract_specific_orbit_on_kovalevski_points_by_length = true;
			extract_specific_orbit_on_kovalevski_points_by_length_length = ST.strtoi(argv[++i]);
			extract_specific_orbit_on_kovalevski_points_by_length_index = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-extract_specific_orbit_on_kovalevski_points_by_length "
						<< extract_specific_orbit_on_kovalevski_points_by_length_length << " "
						<< extract_specific_orbit_on_kovalevski_points_by_length_index << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "quartic_curve_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "quartic_curve_activity_description::read_arguments looping, i=" << i << endl;
	} // next i

	cout << "quartic_curve_activity_description::read_arguments done" << endl;
	return i + 1;
}

void quartic_curve_activity_description::print()
{
	if (f_report) {
		cout << "-report " << endl;
	}
	if (f_export_something) {
		cout << "-export_something " << export_something_what << endl;
	}
	if (f_create_surface) {
		cout << "-create_surface " << endl;
	}
	if (f_extract_orbit_on_bitangents_by_length) {
		cout << "-extract_orbit_on_bitangents_by_length " << extract_orbit_on_bitangents_by_length_length << endl;
	}
	if (f_extract_specific_orbit_on_bitangents_by_length) {
		cout << "-extract_specific_orbit_on_bitangents_by_length "
				<< extract_specific_orbit_on_bitangents_by_length_length << " "
				<< extract_specific_orbit_on_bitangents_by_length_index << endl;
	}
}





}}}}

