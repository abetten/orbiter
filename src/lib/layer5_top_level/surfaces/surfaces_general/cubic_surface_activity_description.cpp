/*
 * cubic_surface_activity_description.cpp
 *
 *  Created on: Mar 18, 2021
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_in_general {


cubic_surface_activity_description::cubic_surface_activity_description()
{
	f_report = false;

	f_report_group_elements = false;
	//std::string report_group_elements_csv_file;
	//std::string report_group_elements_heading;

	f_export_something = false;
	//std::string export_something_what;

	f_export_gap = false;

	f_all_quartic_curves = false;

	f_export_all_quartic_curves = false;

	f_export_something_with_group_element = false;
	//std::string export_something_with_group_element_what;
	//std::string export_something_with_group_element_label;

	f_action_on_module = false;
	//std::string action_on_module_type;
	//std::string action_on_module_basis;
	//std::string action_on_module_gens;

	f_Clebsch_map_up = false;
	Clebsch_map_up_line_1_idx = -1;
	Clebsch_map_up_line_2_idx = -1;

	f_Clebsch_map_up_single_point = false;
	Clebsch_map_up_single_point_input_point = -1;
	Clebsch_map_up_single_point_line_1_idx = -1;
	Clebsch_map_up_single_point_line_2_idx = -1;

}

cubic_surface_activity_description::~cubic_surface_activity_description()
{

}


int cubic_surface_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "cubic_surface_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			if (f_v) {
				cout << "-report " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-report_group_elements") == 0) {
			f_report_group_elements = true;
			report_group_elements_csv_file.assign(argv[++i]);
			report_group_elements_heading.assign(argv[++i]);
			if (f_v) {
				cout << "-report_group_elements "
						<< report_group_elements_csv_file
						<< " " << report_group_elements_heading << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_something") == 0) {
			f_export_something = true;
			export_something_what.assign(argv[++i]);
			if (f_v) {
				cout << "-export_something " << export_something_what << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_gap") == 0) {
			f_export_gap = true;
			if (f_v) {
				cout << "-export_gap " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-all_quartic_curves") == 0) {
			f_all_quartic_curves = true;
			if (f_v) {
				cout << "-all_quartic_curves " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_all_quartic_curves") == 0) {
			f_export_all_quartic_curves = true;
			if (f_v) {
				cout << "-export_all_quartic_curves " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_something_with_group_element") == 0) {
			f_export_something_with_group_element = true;
			export_something_with_group_element_what.assign(argv[++i]);
			export_something_with_group_element_label.assign(argv[++i]);
			if (f_v) {
				cout << "-export_something_with_group_element "
						<< export_something_with_group_element_what << " "
						<< export_something_with_group_element_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-action_on_module") == 0) {
			f_action_on_module = true;
			action_on_module_type.assign(argv[++i]);
			action_on_module_basis.assign(argv[++i]);
			action_on_module_gens.assign(argv[++i]);
			if (f_v) {
				cout << "-action_on_module "
						<< action_on_module_type << " "
						<< action_on_module_basis << " "
						<< action_on_module_gens << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Clebsch_map_up") == 0) {
			f_Clebsch_map_up = true;
			Clebsch_map_up_line_1_idx = ST.strtoi(argv[++i]);
			Clebsch_map_up_line_2_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Clebsch_map_up "
						<< Clebsch_map_up_line_1_idx << " "
						<< Clebsch_map_up_line_2_idx << " "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Clebsch_map_up_single_point") == 0) {
			f_Clebsch_map_up_single_point = true;
			Clebsch_map_up_single_point_input_point = ST.strtoi(argv[++i]);
			Clebsch_map_up_single_point_line_1_idx = ST.strtoi(argv[++i]);
			Clebsch_map_up_single_point_line_2_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Clebsch_map_up_single_point "
						<< Clebsch_map_up_single_point_input_point << " "
						<< Clebsch_map_up_single_point_line_1_idx << " "
						<< Clebsch_map_up_single_point_line_2_idx << " "
						<< endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "cubic_surface_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		if (f_v) {
			cout << "cubic_surface_activity_description::read_arguments looping, i=" << i << endl;
		}
	} // next i

	if (f_v) {
		cout << "cubic_surface_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void cubic_surface_activity_description::print()
{
	if (f_report) {
		cout << "-report " << endl;
	}
	if (f_report_group_elements) {
		cout << "-report_group_elements "
				<< report_group_elements_csv_file
				<< " " << report_group_elements_heading << endl;
	}
	if (f_export_something) {
		cout << "-export_something " << export_something_what << endl;
	}
	if (f_export_gap) {
		cout << "-export_gap " << endl;
	}
	if (f_all_quartic_curves) {
		cout << "-all_quartic_curves " << endl;
	}
	if (f_export_all_quartic_curves) {
		cout << "-export_all_quartic_curves " << endl;
	}
	if (f_export_something_with_group_element) {
		cout << "-export_something_with_group_element "
				<< export_something_with_group_element_what << " "
				<< export_something_with_group_element_label << endl;
	}
	if (f_action_on_module) {
		cout << "-action_on_module "
				<< action_on_module_type << " "
				<< action_on_module_gens << endl;
	}
	if (f_Clebsch_map_up) {
		cout << "-Clebsch_map_up "
				<< Clebsch_map_up_line_1_idx << " "
				<< Clebsch_map_up_line_2_idx << " "
				<< endl;
	}
	if (f_Clebsch_map_up_single_point) {
		cout << "-Clebsch_map_up_single_point "
				<< Clebsch_map_up_single_point_input_point << " "
				<< Clebsch_map_up_single_point_line_1_idx << " "
				<< Clebsch_map_up_single_point_line_2_idx << " "
				<< endl;
	}
}



}}}}


