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
	f_file = FALSE;
	//std::string fname;
	f_inverse = 0;
	f_additive_inverse = 0;

	f_power_cycle = 0;
	power_cycle_base = 0;

	f_cyclotomic_sets = FALSE;
	cyclotomic_sets_q = 0;
	//std::string cyclotomic_sets_reps;

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

	if (f_v) {
		cout << "draw_mod_n_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = strtoi(argv[++i]);
			cout << "-n " << n << endl;
		}
		else if (stringcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname.assign(argv[++i]);
			cout << "-file " << fname << endl;
		}
		else if (stringcmp(argv[i], "-inverse") == 0) {
			f_inverse = TRUE;
			cout << "-inverse " << endl;
		}
		else if (stringcmp(argv[i], "-additive_inverse") == 0) {
			f_additive_inverse = TRUE;
			cout << "-additive_inverse " << endl;
		}
		else if (stringcmp(argv[i], "-power_cycle") == 0) {
			f_power_cycle = TRUE;
			power_cycle_base = strtoi(argv[++i]);
			cout << "-power_cycle " << power_cycle_base << endl;
		}
		else if (stringcmp(argv[i], "-cyclotomic_sets") == 0) {
			f_cyclotomic_sets = TRUE;
			cyclotomic_sets_q = strtoi(argv[++i]);
			cyclotomic_sets_reps.assign(argv[++i]);
			cout << "-cyclotomic_sets " << cyclotomic_sets_q << " " << cyclotomic_sets_reps << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
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



}}

