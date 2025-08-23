/*
 * symbolic_object_activity_description.cpp
 *
 *  Created on: Feb 4, 2022
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace expression_parser {


symbolic_object_activity_description::symbolic_object_activity_description()
{
	Record_birth();
	f_export = false;

#if 0
	f_evaluate = false;
	//std::string evaluate_assignment;
#endif

	f_print = false;

	f_latex = false;

	f_evaluate_affine = false;

	f_collect_monomials_binary = false;

#if 0
	f_sweep = false;
	//std::string sweep_variables;

	f_sweep_affine = false;
	//std::string sweep_affine_variables;
#endif
}

symbolic_object_activity_description::~symbolic_object_activity_description()
{
	Record_death();

}

int symbolic_object_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "symbolic_object_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-export") == 0) {
			f_export = true;
			if (f_v) {
				cout << "-export " << endl;
			}
		}
#if 0
		else if (ST.stringcmp(argv[i], "-evaluate") == 0) {
			f_evaluate = true;
			evaluate_assignment.assign(argv[++i]);
			if (f_v) {
				cout << "-evaluate "
						<< evaluate_assignment << endl;
			}
		}
#endif
		else if (ST.stringcmp(argv[i], "-print") == 0) {
			f_print = true;
			if (f_v) {
				cout << "-print " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-latex") == 0) {
			f_latex = true;
			if (f_v) {
				cout << "-latex " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-evaluate_affine") == 0) {
			f_evaluate_affine = true;
			if (f_v) {
				cout << "-evaluate_affine " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-collect_monomials_binary") == 0) {
			f_collect_monomials_binary = true;
			if (f_v) {
				cout << "-collect_monomials_binary " << endl;
			}
		}



#if 0
		else if (ST.stringcmp(argv[i], "-sweep") == 0) {
			f_sweep = true;
			sweep_variables.assign(argv[++i]);
			if (f_v) {
				cout << "-sweep "
						<< " " << sweep_variables << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-sweep_affine") == 0) {
			f_sweep_affine = true;
			sweep_affine_variables.assign(argv[++i]);
			if (f_v) {
				cout << "-sweep "
						<< " " << sweep_affine_variables << endl;
			}
		}
#endif
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "symbolic_object_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "symbolic_object_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void symbolic_object_activity_description::print()
{
	if (f_export) {
		cout << "-export " << endl;
	}
	if (f_print) {
		cout << "-print " << endl;
	}
	if (f_latex) {
		cout << "-latex " << endl;
	}
	if (f_evaluate_affine) {
		cout << "-evaluate_affine " << endl;
	}
	if (f_collect_monomials_binary) {
		cout << "-collect_monomials_binary " << endl;
	}

#if 0
	if (f_sweep) {
		cout << "-sweep "
				<< " " << sweep_variables << endl;
	}
	if (f_sweep_affine) {
		cout << "-sweep "
				<< " " << sweep_affine_variables << endl;
	}
#endif
}


}}}}





