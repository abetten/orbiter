/*
 * polynomial_ring_activity_description.cpp
 *
 *  Created on: Feb 26, 2022
 *      Author: betten
 */




#include "foundations.h"


using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace ring_theory {



polynomial_ring_activity_description::polynomial_ring_activity_description()
{
	f_cheat_sheet = FALSE;

	f_ideal = FALSE;
	//ideal_label;
	//ideal_label_txt
	//std::string ideal_point_set_label;

	f_apply_transformation = FALSE;
	//std::string apply_transformation_Eqn_in_label;
	//std::string apply_transformation_vector_ge_label;

	f_set_variable_names = FALSE;
	//std::string set_variable_names_txt;
	//std::string set_variable_names_tex;

	f_print_equation = FALSE;
	//std::string print_equation_input;

}


polynomial_ring_activity_description::~polynomial_ring_activity_description()
{

}


int polynomial_ring_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "polynomial_ring_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-cheat_sheet") == 0) {
			f_cheat_sheet = TRUE;
			if (f_v) {
				cout << "-cheat_sheet " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-ideal") == 0) {
			f_ideal = TRUE;

			ideal_label_txt.assign(argv[++i]);
			ideal_label_tex.assign(argv[++i]);
			ideal_point_set_label.assign(argv[++i]);

			if (f_v) {
				cout << "-ideal "
					<< ideal_label_txt << " "
					<< ideal_label_tex << " "
					<< ideal_point_set_label << " "
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-apply_transformation") == 0) {
			f_apply_transformation = TRUE;
			apply_transformation_Eqn_in_label.assign(argv[++i]);
			apply_transformation_vector_ge_label.assign(argv[++i]);

			if (f_v) {
				cout << "-apply_transformation "
						<< apply_transformation_Eqn_in_label << " "
						<< apply_transformation_vector_ge_label << " "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-set_variable_names") == 0) {
			f_set_variable_names = TRUE;
			set_variable_names_txt.assign(argv[++i]);
			set_variable_names_tex.assign(argv[++i]);

			if (f_v) {
				cout << "-set_variable_names "
						<< set_variable_names_txt << " "
						<< set_variable_names_tex << " "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_equation") == 0) {
			f_print_equation = TRUE;
			print_equation_input.assign(argv[++i]);

			if (f_v) {
				cout << "-print_equation "
						<< print_equation_input << " "
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
			cout << "polynomial_ring_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "polynomial_ring_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void polynomial_ring_activity_description::print()
{
	if (f_cheat_sheet) {
		cout << "-cheat_sheet " << endl;
	}
	if (f_ideal) {

		cout << "-ideal "
				<< ideal_label_txt << " "
				<< ideal_label_tex << " "
				<< ideal_point_set_label << " "
				<< endl;
	}
	if (f_apply_transformation) {
		cout << "-apply_transformation "
				<< apply_transformation_Eqn_in_label << " "
				<< apply_transformation_vector_ge_label << " "
				<< endl;
	}
	if (f_set_variable_names) {
		cout << "-set_variable_names "
				<< set_variable_names_txt << " "
				<< set_variable_names_tex << " "
				<< endl;
	}
	if (f_print_equation) {
		cout << "-print_equation "
				<< print_equation_input << " "
				<< endl;
	}
}



}}}
