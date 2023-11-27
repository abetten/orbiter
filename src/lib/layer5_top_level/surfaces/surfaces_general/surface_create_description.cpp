// surface_create_description.cpp
// 
// Anton Betten
//
// December 8, 2017
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_in_general {


surface_create_description::surface_create_description()
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

	f_by_rank = false;
	//std::string rank_text;
	rank_defining_q = 0;


	f_family_Eckardt = false;
	family_Eckardt_a = 0;
	family_Eckardt_b = 0;

	f_family_G13 = false;
	family_G13_a = 0;

	f_family_F13 = false;
	family_F13_a = 0;

	f_family_bes = false;
	family_bes_a = 0;
	family_bes_c = 0;


	f_family_general_abcd = false;
	family_general_abcd_a = 0;
	family_general_abcd_b = 0;
	family_general_abcd_c = 0;
	family_general_abcd_d = 0;

	f_arc_lifting = false;
	//arc_lifting_text = NULL;
	//arc_lifting_two_lines_text = NULL;
	f_arc_lifting_with_two_lines = false;

	f_Cayley_form = false;
	Cayley_form_k = 0;
	Cayley_form_l = 0;
	Cayley_form_m = 0;
	Cayley_form_n = 0;


	f_by_equation = false;
	//std::string equation_ring_label;
	//std::string equation_name_of_formula;
	//std::string equation_name_of_formula_tex;
	//std::string equation_managed_variables;
	//std::string equation_text;
	//std::string equation_parameters;
	//std::string equation_parameters_tex;
	//std::string equation_parameter_values;


	f_by_symbolic_object = false;
	//std::string by_symbolic_object_ring_label;
	//std::string by_symbolic_object_name_of_formula;


	f_by_double_six = false;
	//std::string by_double_six_label;
	//std::string by_double_six_label_tex;
	//std::string by_double_six_text;


	f_by_skew_hexagon = false;
	//std::string by_skew_hexagon_label;
	//std::string by_skew_hexagon_label_tex

	//nb_select_double_six = 0;
	//select_double_six_string[];

	f_override_group = false;
	//std::string override_group_order;
	override_group_nb_gens = 0;
	//std::string override_group_gens;

	f_random = false;


	//std::vector<std::string> transform_coeffs;
	//std::vector<int> f_inverse_transform;

}

surface_create_description::~surface_create_description()
{
}

int surface_create_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	cout << "surface_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-space") == 0) {
			f_space = true;
			space_label.assign(argv[++i]);
			cout << "-space " << space_label << endl;
		}
		else if (ST.stringcmp(argv[i], "-label_txt") == 0) {
			f_label_txt = true;
			label_txt.assign(argv[++i]);
			cout << "-label_txt " << label_txt << endl;
		}
		else if (ST.stringcmp(argv[i], "-label_tex") == 0) {
			f_label_tex = true;
			label_tex.assign(argv[++i]);
			cout << "-label_tex " << label_tex << endl;
		}
		else if (ST.stringcmp(argv[i], "-label_for_summary") == 0) {
			f_label_for_summary = true;
			label_for_summary.assign(argv[++i]);
			cout << "-label_for_summary " << label_for_summary << endl;
		}
		else if (ST.stringcmp(argv[i], "-catalogue") == 0) {
			f_catalogue = true;
			iso = ST.strtoi(argv[++i]);
			cout << "-catalogue " << iso << endl;
		}
		else if (ST.stringcmp(argv[i], "-by_coefficients") == 0) {
			f_by_coefficients = true;
			coefficients_text.assign(argv[++i]);
			cout << "-by_coefficients " << coefficients_text << endl;
		}
		else if (ST.stringcmp(argv[i], "-by_rank") == 0) {
			f_by_rank = true;
			rank_text.assign(argv[++i]);
			rank_defining_q = ST.strtoi(argv[++i]);
			cout << "-by_rank " << rank_text << " " << rank_defining_q << endl;
		}
		else if (ST.stringcmp(argv[i], "-family_Eckardt") == 0) {
			f_family_Eckardt = true;
			family_Eckardt_a = ST.strtoi(argv[++i]);
			family_Eckardt_b = ST.strtoi(argv[++i]);
			cout << "-family_Eckardt " << family_Eckardt_a << " " << family_Eckardt_b << endl;
		}
		else if (ST.stringcmp(argv[i], "-family_G13") == 0) {
			f_family_G13 = true;
			family_G13_a = ST.strtoi(argv[++i]);
			cout << "-family_G13 " << family_G13_a << endl;
		}
		else if (ST.stringcmp(argv[i], "-family_F13") == 0) {
			f_family_F13 = true;
			family_F13_a = ST.strtoi(argv[++i]);
			cout << "-family_F13 " << family_F13_a << endl;
		}
		else if (ST.stringcmp(argv[i], "-family_bes") == 0) {
			f_family_bes = true;
			family_bes_a = ST.strtoi(argv[++i]);
			family_bes_c = ST.strtoi(argv[++i]);
			cout << "-family_bes " << family_bes_a << " " << family_bes_c << endl;
		}
		else if (ST.stringcmp(argv[i], "-family_general_abcd") == 0) {
			f_family_general_abcd = true;
			family_general_abcd_a = ST.strtoi(argv[++i]);
			family_general_abcd_b = ST.strtoi(argv[++i]);
			family_general_abcd_c = ST.strtoi(argv[++i]);
			family_general_abcd_d = ST.strtoi(argv[++i]);
			cout << "-family_general_abcd "
					<< family_general_abcd_a << " " << family_general_abcd_b << " "
					<< family_general_abcd_c << " " << family_general_abcd_d
					<< endl;
		}
		else if (ST.stringcmp(argv[i], "-arc_lifting") == 0) {
			f_arc_lifting = true;
			arc_lifting_text.assign(argv[++i]);
			cout << "-arc_lifting " << arc_lifting_text << endl;
		}
		else if (ST.stringcmp(argv[i], "-arc_lifting_with_two_lines") == 0) {
			f_arc_lifting_with_two_lines = true;
			arc_lifting_text.assign(argv[++i]);
			arc_lifting_two_lines_text.assign(argv[++i]);
			cout << "-arc_lifting_with_two_lines " << arc_lifting_text
					<< " " << arc_lifting_two_lines_text << endl;
		}
		else if (ST.stringcmp(argv[i], "-Cayley_form") == 0) {
			f_Cayley_form = true;
			Cayley_form_k = ST.strtoi(argv[++i]);
			Cayley_form_l = ST.strtoi(argv[++i]);
			Cayley_form_m = ST.strtoi(argv[++i]);
			Cayley_form_n = ST.strtoi(argv[++i]);
			cout << "-Cayley_form "
					<< Cayley_form_k << " " << Cayley_form_l << " "
					<< Cayley_form_m << " " << Cayley_form_n
					<< endl;
		}
		else if (ST.stringcmp(argv[i], "-by_equation") == 0) {
			f_by_equation = true;
			equation_ring_label.assign(argv[++i]);
			equation_name_of_formula.assign(argv[++i]);
			equation_name_of_formula_tex.assign(argv[++i]);
			equation_managed_variables.assign(argv[++i]);
			equation_text.assign(argv[++i]);
			equation_parameters.assign(argv[++i]);
			equation_parameters_tex.assign(argv[++i]);
			equation_parameter_values.assign(argv[++i]);
			cout << "-by_equation "
					<< equation_ring_label << " "
					<< equation_name_of_formula << " "
					<< equation_name_of_formula_tex << " "
					<< equation_managed_variables << " "
					<< equation_text << " "
					<< equation_parameters << " "
					<< equation_parameters_tex << " "
					<< equation_parameter_values << " "
					<< endl;
		}
		else if (ST.stringcmp(argv[i], "-by_symbolic_object") == 0) {
			f_by_symbolic_object = true;
			by_symbolic_object_ring_label.assign(argv[++i]);
			by_symbolic_object_name_of_formula.assign(argv[++i]);
			cout << "-by_equation "
					<< by_symbolic_object_ring_label << " "
					<< by_symbolic_object_name_of_formula << " "
					<< endl;
		}
		else if (ST.stringcmp(argv[i], "-by_double_six") == 0) {
			f_by_double_six = true;
			by_double_six_label.assign(argv[++i]);
			by_double_six_label_tex.assign(argv[++i]);
			by_double_six_text.assign(argv[++i]);
			cout << "-by_double_six "
					<< by_double_six_label << " "
					<< by_double_six_label_tex << " "
					<< by_double_six_text << " "
					<< endl;
		}
		else if (ST.stringcmp(argv[i], "-by_skew_hexagon") == 0) {
			f_by_skew_hexagon = true;

			by_skew_hexagon_label.assign(argv[++i]);
			by_skew_hexagon_label_tex.assign(argv[++i]);
			cout << "-by_double_six "
					<< by_skew_hexagon_label << " "
					<< by_skew_hexagon_label_tex << " "
					<< endl;
		}

		else if (ST.stringcmp(argv[i], "-select_double_six") == 0) {
			//f_select_double_six = true;
			string s;

			s.assign(argv[++i]);
			select_double_six_string.push_back(s);
			cout << "-select_double_six "
					<< select_double_six_string[select_double_six_string.size() - 1] << endl;
		}


		else if (ST.stringcmp(argv[i], "-override_group") == 0) {
			f_override_group = true;

			override_group_order.assign(argv[++i]);
			override_group_nb_gens = ST.strtoi(argv[++i]);
			override_group_gens.assign(argv[++i]);

			cout << "-override_group "
					<< override_group_order
					<< " " << override_group_nb_gens
					<< " " << override_group_gens
					<< endl;
		}

		else if (ST.stringcmp(argv[i], "-random") == 0) {
			f_random = true;

			cout << "-random " << endl;
		}


		else if (ST.stringcmp(argv[i], "-transform") == 0) {

			string s;

			s.assign(argv[++i]);
			transform_coeffs.push_back(s);
			f_inverse_transform.push_back(false);
			cout << "-transform " << transform_coeffs[transform_coeffs.size() - 1]
					<< " " << f_inverse_transform[transform_coeffs.size() - 1] << endl;
		}
		else if (ST.stringcmp(argv[i], "-transform_inverse") == 0) {

			string s;

			s.assign(argv[++i]);
			transform_coeffs.push_back(s);
			f_inverse_transform.push_back(true);
			cout << "-transform_inverse " << transform_coeffs[transform_coeffs.size() - 1]
					<< " " << f_inverse_transform[transform_coeffs.size() - 1] << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "surface_create_description::read_arguments unknown command " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "surface_create_description::read_arguments done" << endl;
	return i + 1;
}

void surface_create_description::print()
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
	if (f_by_rank) {
		cout << "-by_rank " << rank_text << " " << rank_defining_q << endl;
	}
	if (f_family_Eckardt) {
		cout << "-family_Eckardt " << family_Eckardt_a << " " << family_Eckardt_b << endl;
	}
	if (f_family_G13) {
		cout << "-family_G13 " << family_G13_a << endl;
	}
	if (f_family_F13) {
		cout << "-family_F13 " << family_F13_a << endl;
	}
	if (f_family_bes) {
		cout << "-family_bes " << family_bes_a << " " << family_bes_c << endl;
	}
	if (f_family_general_abcd) {
		cout << "-family_general_abcd "
				<< family_general_abcd_a << " " << family_general_abcd_b << " "
				<< family_general_abcd_c << " " << family_general_abcd_d
				<< endl;
	}
	if (f_arc_lifting) {
		cout << "-arc_lifting " << arc_lifting_text << endl;
	}
	if (f_arc_lifting_with_two_lines) {
		cout << "-arc_lifting_with_two_lines " << arc_lifting_text
				<< " " << arc_lifting_two_lines_text << endl;
	}
	if (f_Cayley_form) {
		cout << "-Cayley_form "
				<< Cayley_form_k << " " << Cayley_form_l << " "
				<< Cayley_form_m << " " << Cayley_form_n
				<< endl;
	}
	if (f_by_equation) {
		cout << "-by_equation "
				<< equation_ring_label << " "
				<< equation_name_of_formula << " "
				<< equation_name_of_formula_tex << " "
				<< equation_managed_variables << " "
				<< equation_text << " "
				<< equation_parameters << " "
				<< equation_parameters_tex << " "
				<< equation_parameter_values << " "
				<< endl;
	}
	if (f_by_symbolic_object) {
		cout << "-by_equation "
				<< by_symbolic_object_ring_label << " "
				<< by_symbolic_object_name_of_formula << " "
				<< endl;
	}
	if (f_by_double_six) {
		cout << "-by_double_six "
				<< by_double_six_label << " "
				<< by_double_six_label_tex << " "
				<< by_double_six_text << " "
				<< endl;
	}
	if (f_by_skew_hexagon) {
		cout << "-by_skew_hexagon "
				<< by_skew_hexagon_label << " "
				<< by_skew_hexagon_label_tex << " "
				<< endl;
	}

	for (int i = 0; i < select_double_six_string.size(); i++) {
		cout << "-select_double_six " << select_double_six_string[i] << endl;
	}


	if (f_override_group) {
		cout << "-override_group "
				<< override_group_order
				<< " " << override_group_nb_gens
				<< " " << override_group_gens
				<< endl;
	}
	if (f_random) {

		cout << "-random " << endl;
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




