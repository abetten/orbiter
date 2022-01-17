/*
 * draw_mod_n_description.cpp
 *
 *  Created on: Apr 15, 2021
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


draw_mod_n_description::draw_mod_n_description()
{
	f_n = FALSE;
	n = 0;

	f_mod_s = FALSE;
	mod_s = 0;

	f_divide_out_by = FALSE;
	divide_out_by = 0;

	f_file = FALSE;
	//std::string fname;
	f_inverse = 0;
	f_additive_inverse = 0;

	f_power_cycle = 0;
	power_cycle_base = 0;

	f_cyclotomic_sets = FALSE;
	cyclotomic_sets_q = 0;
	//std::string cyclotomic_sets_reps;

	f_cyclotomic_sets_thickness = FALSE;
	cyclotomic_sets_thickness = 100;

}

draw_mod_n_description::~draw_mod_n_description()
{

}



int draw_mod_n_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "draw_mod_n_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			cout << "-n " << n << endl;
		}
		else if (ST.stringcmp(argv[i], "-mod_s") == 0) {
			f_mod_s = TRUE;
			mod_s = ST.strtoi(argv[++i]);
			cout << "-mod_s " << mod_s << endl;
		}
		else if (ST.stringcmp(argv[i], "-divide_out_by") == 0) {
			f_divide_out_by = TRUE;
			divide_out_by = ST.strtoi(argv[++i]);
			cout << "-divide_out_by " << divide_out_by << endl;
		}
		else if (ST.stringcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname.assign(argv[++i]);
			cout << "-file " << fname << endl;
		}
		else if (ST.stringcmp(argv[i], "-inverse") == 0) {
			f_inverse = TRUE;
			cout << "-inverse " << endl;
		}
		else if (ST.stringcmp(argv[i], "-additive_inverse") == 0) {
			f_additive_inverse = TRUE;
			cout << "-additive_inverse " << endl;
		}
		else if (ST.stringcmp(argv[i], "-power_cycle") == 0) {
			f_power_cycle = TRUE;
			power_cycle_base = ST.strtoi(argv[++i]);
			cout << "-power_cycle " << power_cycle_base << endl;
		}
		else if (ST.stringcmp(argv[i], "-cyclotomic_sets") == 0) {
			f_cyclotomic_sets = TRUE;
			cyclotomic_sets_q = ST.strtoi(argv[++i]);
			cyclotomic_sets_reps.assign(argv[++i]);
			cout << "-cyclotomic_sets " << cyclotomic_sets_q << " " << cyclotomic_sets_reps << endl;
		}
		else if (ST.stringcmp(argv[i], "-cyclotomic_sets_thickness") == 0) {
			f_cyclotomic_sets_thickness = TRUE;
			cyclotomic_sets_thickness = ST.strtoi(argv[++i]);
			cout << "-cyclotomic_sets_thickness " << cyclotomic_sets_thickness << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "draw_mod_n_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	if (f_v) {
		cout << "draw_mod_n_description::read_arguments done" << endl;
	}
	return i + 1;
}

void draw_mod_n_description::print()
{
	if (f_n) {
		cout << "-n " << n << endl;
	}
	if (f_mod_s) {
		cout << "-mod_s " << mod_s << endl;
	}
	if (f_divide_out_by) {
		cout << "-divide_out_by " << divide_out_by << endl;
	}
	if (f_file) {
		cout << "-file " << fname << endl;
	}
	if (f_inverse) {
		cout << "-inverse " << endl;
	}
	if (f_additive_inverse) {
		cout << "-additive_inverse " << endl;
	}
	if (f_power_cycle) {
		cout << "-power_cycle " << power_cycle_base << endl;
	}
	if (f_cyclotomic_sets) {
		cout << "-cyclotomic_sets " << cyclotomic_sets_q << " " << cyclotomic_sets_reps << endl;
	}
	if (f_cyclotomic_sets_thickness) {
		cout << "-cyclotomic_sets_thickness " << cyclotomic_sets_thickness << endl;
	}
}





}}

