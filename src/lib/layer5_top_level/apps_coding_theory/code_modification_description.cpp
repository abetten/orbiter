/*
 * code_modification_description.cpp
 *
 *  Created on: Aug 10, 2022
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_coding_theory {



code_modification_description::code_modification_description()
{
	Record_birth();
	f_dual = false;

}

code_modification_description::~code_modification_description()
{
	Record_death();
}


int code_modification_description::check_and_parse_argument(
	int argc, int &i, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "code_modification_description::check_and_parse_argument" << endl;
	}
	if (ST.stringcmp(argv[i], "-dual") == 0) {
		f_dual = true;
		i++;
		if (f_v) {
			cout << "-dual " << endl;
		}
		return true;
	}
	if (f_v) {
		cout << "code_modification_description::read_arguments done" << endl;
	}
	return false;
}

int code_modification_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "code_modification_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-dual") == 0) {
			f_dual = true;
			if (f_v) {
				cout << "-complement " << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "code_modification_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "code_modification_description::read_arguments done" << endl;
	}
	return i + 1;
}

void code_modification_description::print()
{
	if (f_dual) {
		cout << "-dual " << endl;
	}
}

void code_modification_description::apply(
		apps_coding_theory::create_code *Code, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "code_modification_description::apply" << endl;
	}
	if (f_dual) {
		Code->dual_code(verbose_level);
	}
	if (f_v) {
		cout << "code_modification_description::apply done" << endl;
	}

}



}}}


