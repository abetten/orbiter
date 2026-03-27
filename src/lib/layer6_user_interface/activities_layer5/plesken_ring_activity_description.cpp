/*
 * plesken_ring_activity_description.cpp
 *
 *  Created on: Mar 3, 2026
 *      Author: betten
 */




#include "orbiter_user_interface.h"

using namespace std;


namespace orbiter {
namespace layer6_user_interface {
namespace activities_layer5 {


plesken_ring_activity_description::plesken_ring_activity_description()
{
	Record_birth();

	f_report = false;
	//std::string report_draw_options_label;

	f_evaluate_join = false;
	//std::string evaluate_join_ring_label;
	//std::string evaluate_join_formula_label;

	f_evaluate_meet = false;
	//std::string evaluate_meet_ring_label;
	//std::string evaluate_meet_formula_label;

}

plesken_ring_activity_description::~plesken_ring_activity_description()
{
	Record_death();

}



int plesken_ring_activity_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "plesken_ring_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {


		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;
			report_draw_options_label.assign(argv[++i]);
			if (f_v) {
				cout << "-report " << report_draw_options_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-evaluate_join") == 0) {
			f_evaluate_join = true;
			evaluate_join_ring_label.assign(argv[++i]);
			evaluate_join_formula_label.assign(argv[++i]);
			if (f_v) {
				cout << "-evaluate_join " << evaluate_join_ring_label << " " << evaluate_join_formula_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-evaluate_meet") == 0) {
			f_evaluate_meet = true;
			evaluate_meet_ring_label.assign(argv[++i]);
			evaluate_meet_formula_label.assign(argv[++i]);
			if (f_v) {
				cout << "-evaluate_meet " << evaluate_meet_ring_label << " " << evaluate_meet_formula_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "plesken_ring_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "plesken_ring_activity_description::read_arguments done" << endl;
	return i + 1;
}


void plesken_ring_activity_description::print()
{
	if (f_report) {
		cout << "-report " << report_draw_options_label << endl;
	}
	if (f_evaluate_join) {
		cout << "-evaluate_join " << evaluate_join_ring_label
				<< " " << evaluate_join_formula_label << endl;
	}
	if (f_evaluate_meet) {
		cout << "-evaluate_meet " << evaluate_meet_ring_label
				<< " " << evaluate_meet_formula_label << endl;
	}
}




}}}





