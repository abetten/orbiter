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

	f_space = FALSE;
	//std::string space_label;

	f_space_pointer = FALSE;
	space_pointer = NULL;

	f_label_txt = FALSE;
	//label_txt

	f_label_tex = FALSE;
	//label_tex

	f_label_for_summary = FALSE;
	//label_for_summary


	f_catalogue = FALSE;
	iso = 0;
	f_by_coefficients = FALSE;
	//coefficients_text = NULL;


	f_by_equation = FALSE;
	//std::string equation_name_of_formula;
	//std::string equation_name_of_formula_tex;
	//std::string equation_managed_variables;
	//std::string equation_text;
	//std::string equation_parameters;
	//std::string equation_parameters_tex;



	//nb_select_double_six = 0;
	//select_double_six_string[];

	f_from_cubic_surface = FALSE;
	//std::string from_cubic_surface_label;
	from_cubic_surface_point_orbit_idx = 0;

	f_override_group = FALSE;
	//std::string override_group_order;
	override_group_nb_gens = 0;
	//std::string override_group_gens;


	//std::vector<std::string> transform_coeffs;
	//std::vector<int> f_inverse_transform;

}

quartic_curve_create_description::~quartic_curve_create_description()
{
}

int quartic_curve_create_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	cout << "quartic_curve_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-space") == 0) {
			f_space = TRUE;
			space_label.assign(argv[++i]);
			cout << "-space " << space_label << endl;
		}
		else if (ST.stringcmp(argv[i], "-label_txt") == 0) {
			f_label_txt = TRUE;
			label_txt.assign(argv[++i]);
			cout << "-label_txt " << label_txt << endl;
		}
		else if (ST.stringcmp(argv[i], "-label_tex") == 0) {
			f_label_tex = TRUE;
			label_tex.assign(argv[++i]);
			cout << "-label_tex " << label_tex << endl;
		}
		else if (ST.stringcmp(argv[i], "-label_for_summary") == 0) {
			f_label_for_summary = TRUE;
			label_for_summary.assign(argv[++i]);
			cout << "-label_for_summary " << label_for_summary << endl;
		}
		else if (ST.stringcmp(argv[i], "-catalogue") == 0) {
			f_catalogue = TRUE;
			iso = ST.strtoi(argv[++i]);
			cout << "-catalogue " << iso << endl;
		}
		else if (ST.stringcmp(argv[i], "-by_coefficients") == 0) {
			f_by_coefficients = TRUE;
			coefficients_text.assign(argv[++i]);
			cout << "-by_coefficients " << coefficients_text << endl;
		}
		else if (ST.stringcmp(argv[i], "-by_equation") == 0) {
			f_by_equation = TRUE;
			equation_name_of_formula.assign(argv[++i]);
			equation_name_of_formula_tex.assign(argv[++i]);
			equation_managed_variables.assign(argv[++i]);
			equation_text.assign(argv[++i]);
			equation_parameters.assign(argv[++i]);
			equation_parameters_tex.assign(argv[++i]);
			cout << "-by_equation "
					<< equation_name_of_formula << " "
					<< equation_name_of_formula_tex << " "
					<< equation_managed_variables << " "
					<< equation_text << " "
					<< equation_parameters << " "
					<< equation_parameters_tex << " "
					<< endl;
		}
		else if (ST.stringcmp(argv[i], "-from_cubic_surface") == 0) {
			f_from_cubic_surface = TRUE;

			from_cubic_surface_label.assign(argv[++i]);
			from_cubic_surface_point_orbit_idx = ST.strtoi(argv[++i]);

			cout << "-from_cubic_surface "
					<< from_cubic_surface_label
					<< " " << from_cubic_surface_point_orbit_idx
					<< endl;
		}
		else if (ST.stringcmp(argv[i], "-override_group") == 0) {
			f_override_group = TRUE;

			override_group_order.assign(argv[++i]);
			override_group_nb_gens = ST.strtoi(argv[++i]);
			override_group_gens.assign(argv[++i]);

			cout << "-override_group "
					<< override_group_order
					<< " " << override_group_nb_gens
					<< " " << override_group_gens
					<< endl;
		}

		else if (ST.stringcmp(argv[i], "-transform") == 0) {

			string s;

			s.assign(argv[++i]);
			transform_coeffs.push_back(s);
			f_inverse_transform.push_back(FALSE);
			cout << "-transform " << transform_coeffs[transform_coeffs.size() - 1]
					<< " " << f_inverse_transform[transform_coeffs.size() - 1] << endl;
		}
		else if (ST.stringcmp(argv[i], "-transform_inverse") == 0) {

			string s;

			s.assign(argv[++i]);
			transform_coeffs.push_back(s);
			f_inverse_transform.push_back(TRUE);
			cout << "-transform_inverse " << transform_coeffs[transform_coeffs.size() - 1]
					<< " " << f_inverse_transform[transform_coeffs.size() - 1] << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "quartic_curve_create_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "quartic_curve_create_description::read_arguments done" << endl;
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
				<< equation_managed_variables << " "
				<< equation_text << " "
				<< equation_parameters << " "
				<< equation_parameters_tex << " "
				<< endl;
	}

	if (f_from_cubic_surface) {

		cout << "-from_cubic_surface "
				<< from_cubic_surface_label
				<< " " << from_cubic_surface_point_orbit_idx
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


