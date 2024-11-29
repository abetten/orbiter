/*
 * draw_mod_n_description.cpp
 *
 *  Created on: Apr 15, 2021
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace graphics {


draw_mod_n_description::draw_mod_n_description()
{
	f_n = false;
	n = 0;

	f_mod_s = false;
	mod_s = 0;

	f_divide_out_by = false;
	divide_out_by = 0;

	f_file = false;
	//std::string fname;

	f_label_nodes = false;
	f_inverse = 0;
	f_additive_inverse = 0;

	f_power_cycle = 0;
	power_cycle_base = 0;

	f_cyclotomic_sets = false;
	cyclotomic_sets_q = 0;
	//std::string cyclotomic_sets_reps;

	f_cyclotomic_sets_thickness = false;
	cyclotomic_sets_thickness = 100;

	f_eigenvalues = false;
	//double eigenvalues_A[4];

	f_draw_options = false;
	//std::string draw_options_label;

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
			f_n = true;
			n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-n " << n << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-mod_s") == 0) {
			f_mod_s = true;
			mod_s = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-mod_s " << mod_s << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-divide_out_by") == 0) {
			f_divide_out_by = true;
			divide_out_by = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-divide_out_by " << divide_out_by << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-file") == 0) {
			f_file = true;
			fname.assign(argv[++i]);
			if (f_v) {
				cout << "-file " << fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_nodes") == 0) {
			f_label_nodes = true;
			if (f_v) {
				cout << "-label_nodes " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-dont_label_nodes") == 0) {
			f_label_nodes = false;
			if (f_v) {
				cout << "-dont_label_nodes " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-inverse") == 0) {
			f_inverse = true;
			if (f_v) {
				cout << "-inverse " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-additive_inverse") == 0) {
			f_additive_inverse = true;
			if (f_v) {
				cout << "-additive_inverse " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-power_cycle") == 0) {
			f_power_cycle = true;
			power_cycle_base = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-power_cycle " << power_cycle_base << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-cyclotomic_sets") == 0) {
			f_cyclotomic_sets = true;
			cyclotomic_sets_q = ST.strtoi(argv[++i]);
			cyclotomic_sets_reps.assign(argv[++i]);
			if (f_v) {
				cout << "-cyclotomic_sets " << cyclotomic_sets_q << " " << cyclotomic_sets_reps << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-cyclotomic_sets_thickness") == 0) {
			f_cyclotomic_sets_thickness = true;
			cyclotomic_sets_thickness = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-cyclotomic_sets_thickness " << cyclotomic_sets_thickness << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-eigenvalues") == 0) {
			f_eigenvalues = true;
			eigenvalues_A[0] = ST.strtof(argv[++i]);
			eigenvalues_A[1] = ST.strtof(argv[++i]);
			eigenvalues_A[2] = ST.strtof(argv[++i]);
			eigenvalues_A[3] = ST.strtof(argv[++i]);
			if (f_v) {
				cout << "-eigenvalues "
					<< eigenvalues_A[0] << " "
					<< eigenvalues_A[1] << " "
					<< eigenvalues_A[2] << " "
					<< eigenvalues_A[3] << " "
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_options") == 0) {
			f_draw_options = true;
			draw_options_label.assign(argv[++i]);
			if (f_v) {
				cout << "-draw_options " << draw_options_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
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
	if (f_label_nodes) {
		cout << "-label_nodes " << endl;
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
	if (f_eigenvalues) {
		cout << "-eigenvalues "
				<< eigenvalues_A[0] << " "
				<< eigenvalues_A[1] << " "
				<< eigenvalues_A[2] << " "
				<< eigenvalues_A[3] << " "
				<< endl;
	}
	if (f_draw_options) {
		cout << "-draw_options " << draw_options_label << endl;
	}
}





}}}


