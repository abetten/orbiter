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
	//std::string ideal_ring_label;
	//ideal_label;
	//ideal_label_txt
	//std::string ideal_point_set_label;

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
			ideal_ring_label.assign(argv[++i]);

			ideal_label_txt.assign(argv[++i]);
			ideal_label_tex.assign(argv[++i]);
			ideal_point_set_label.assign(argv[++i]);

			cout << "-ideal "
					<< ideal_ring_label << " "
					<< ideal_label_txt << " "
					<< ideal_label_tex << " "
					<< ideal_point_set_label << " "
					<< endl;
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
				<< ideal_ring_label << " "
				<< ideal_label_txt << " "
				<< ideal_label_tex << " "
				<< ideal_point_set_label << " "
				<< endl;
	}
}



}}}
