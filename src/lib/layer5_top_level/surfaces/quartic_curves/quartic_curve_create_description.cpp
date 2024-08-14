/*
 * quartic_curve_create_description.cpp
 *
 *  Created on: May 20, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace quartic_curves {

quartic_curve_create_description::quartic_curve_create_description()
{

	f_space = false;
	//std::string space_label;

	f_space_pointer = false;
	space_pointer = NULL;

	f_label_txt = false;
	//label_txt

	f_label_tex = false;
	//label_tex

	f_label_for_summary = false;
	//label_for_summary


	f_catalogue = false;
	iso = 0;
	f_by_coefficients = false;
	//coefficients_text = NULL;


	f_by_equation = false;
	//std::string equation_name_of_formula;
	//std::string equation_name_of_formula_tex;
	//std::string equation_text;
	//std::string equation_parameters;
	//std::string equation_parameters_tex;
	//std::string equation_parameter_values;

	f_by_symbolic_object = false;
	//std::string by_symbolic_object_ring_label;
	//std::string by_symbolic_object_name_of_formula;


	f_from_cubic_surface = false;
	//std::string from_cubic_surface_label;
	from_cubic_surface_point_orbit_idx = 0;

	f_from_variety = false;
	//std::string from_variety_label;

	f_override_group = false;
	//std::string override_group_order;
	override_group_nb_gens = 0;
	//std::string override_group_gens;


	//std::vector<std::string> transform_coeffs;
	//std::vector<int> f_inverse_transform;

}

quartic_curve_create_description::~quartic_curve_create_description()
{
}

int quartic_curve_create_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "quartic_curve_create_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-space") == 0) {
			f_space = true;
			space_label.assign(argv[++i]);
			if (f_v) {
				cout << "-space " << space_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_txt") == 0) {
			f_label_txt = true;
			label_txt.assign(argv[++i]);
			if (f_v) {
				cout << "-label_txt " << label_txt << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_tex") == 0) {
			f_label_tex = true;
			label_tex.assign(argv[++i]);
			if (f_v) {
				cout << "-label_tex " << label_tex << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_for_summary") == 0) {
			f_label_for_summary = true;
			label_for_summary.assign(argv[++i]);
			if (f_v) {
				cout << "-label_for_summary " << label_for_summary << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-catalogue") == 0) {
			f_catalogue = true;
			iso = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-catalogue " << iso << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-by_coefficients") == 0) {
			f_by_coefficients = true;
			coefficients_text.assign(argv[++i]);
			if (f_v) {
				cout << "-by_coefficients " << coefficients_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-by_equation") == 0) {
			f_by_equation = true;
			equation_name_of_formula.assign(argv[++i]);
			equation_name_of_formula_tex.assign(argv[++i]);
			equation_text.assign(argv[++i]);
			equation_parameters.assign(argv[++i]);
			equation_parameters_tex.assign(argv[++i]);
			equation_parameter_values.assign(argv[++i]);
			if (f_v) {
				cout << "-by_equation "
						<< equation_name_of_formula << " "
						<< equation_name_of_formula_tex << " "
						<< equation_text << " "
						<< equation_parameters << " "
						<< equation_parameters_tex << " "
						<< equation_parameter_values << " "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-by_symbolic_object") == 0) {
			f_by_symbolic_object = true;
			by_symbolic_object_ring_label.assign(argv[++i]);
			by_symbolic_object_name_of_formula.assign(argv[++i]);
			if (f_v) {
				cout << "-by_symbolic_object " << by_symbolic_object_ring_label
						<< " " << by_symbolic_object_name_of_formula << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-from_cubic_surface") == 0) {
			f_from_cubic_surface = true;

			from_cubic_surface_label.assign(argv[++i]);
			from_cubic_surface_point_orbit_idx = ST.strtoi(argv[++i]);

			if (f_v) {
				cout << "-from_cubic_surface "
						<< from_cubic_surface_label
						<< " " << from_cubic_surface_point_orbit_idx
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-from_variety") == 0) {
			f_from_variety = true;

			from_variety_label.assign(argv[++i]);

			if (f_v) {
				cout << "-from_variety "
						<< from_variety_label
						<< endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-override_group") == 0) {
			f_override_group = true;

			override_group_order.assign(argv[++i]);
			override_group_nb_gens = ST.strtoi(argv[++i]);
			override_group_gens.assign(argv[++i]);

			if (f_v) {
				cout << "-override_group "
					<< override_group_order
					<< " " << override_group_nb_gens
					<< " " << override_group_gens
					<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-transform") == 0) {

			string s;

			s.assign(argv[++i]);
			transform_coeffs.push_back(s);
			f_inverse_transform.push_back(false);
			if (f_v) {
				cout << "-transform " << transform_coeffs[transform_coeffs.size() - 1]
					<< " " << f_inverse_transform[transform_coeffs.size() - 1] << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-transform_inverse") == 0) {

			string s;

			s.assign(argv[++i]);
			transform_coeffs.push_back(s);
			f_inverse_transform.push_back(true);
			if (f_v) {
				cout << "-transform_inverse " << transform_coeffs[transform_coeffs.size() - 1]
					<< " " << f_inverse_transform[transform_coeffs.size() - 1] << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "quartic_curve_create_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "quartic_curve_create_description::read_arguments done" << endl;
	}
	return i + 1;
}


void quartic_curve_create_description::print()
{
	if (f_space) {
		cout << "-space " << space_label << endl;
	}
	if (f_label_txt) {
		cout << "-label_txt " << label_txt << endl;
	}
	if (f_label_tex) {
		cout << "-label_tex " << label_tex << endl;
	}
	if (f_label_for_summary) {
		cout << "-label_for_summary " << label_for_summary << endl;
	}
	if (f_catalogue) {
		cout << "-catalogue " << iso << endl;
	}
	if (f_by_coefficients) {
		cout << "-by_coefficients " << coefficients_text << endl;
	}
	if (f_by_equation) {
		cout << "-by_equation "
				<< equation_name_of_formula << " "
				<< equation_name_of_formula_tex << " "
				<< equation_text << " "
				<< equation_parameters << " "
				<< equation_parameters_tex << " "
				<< equation_parameter_values << " "
				<< endl;
	}

	if (f_by_symbolic_object) {
		cout << "-by_symbolic_object " << by_symbolic_object_ring_label
				<< " " << by_symbolic_object_name_of_formula << endl;
	}

	if (f_from_cubic_surface) {

		cout << "-from_cubic_surface "
				<< from_cubic_surface_label
				<< " " << from_cubic_surface_point_orbit_idx
				<< endl;
	}
	if (f_from_variety) {
		cout << "-from_variety "
				<< from_variety_label
				<< endl;
	}



	if (f_override_group) {
		cout << "-override_group "
				<< override_group_order
				<< " " << override_group_nb_gens
				<< " " << override_group_gens
				<< endl;
	}

	for (int i = 0; i < transform_coeffs.size(); i++) {
		if (f_inverse_transform[i]) {
			cout << "-transform_inverse " << transform_coeffs[i] << endl;
		}
		else {
			cout << "-transform " << transform_coeffs[i] << endl;
		}
	}

}




}}}}


