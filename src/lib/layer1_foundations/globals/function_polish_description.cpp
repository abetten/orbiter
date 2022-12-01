/*
 * function_polish_description.cpp
 *
 *  Created on: Apr 18, 2020
 *      Author: betten
 */


#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {


function_polish_description::function_polish_description()
{
	nb_constants = 0;
	//const_names = NULL;
	//const_values = NULL;
	nb_variables = 0;
	//variable_names = NULL;
	code_sz = 0;
	//code = NULL;
}

function_polish_description::~function_polish_description()
{
}


int function_polish_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, i0, h;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "function_polish_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-const") == 0) {
			if (f_v) {
				cout << "-const" << endl;
			}
			i0 = i + 1;
			for (++i; i < argc; i++) {
				if (ST.stringcmp(argv[i], "-const_end") == 0) {
					break;
				}
			}
			if (i < argc) {
				nb_constants = (i - i0) >> 1;

				for (h = 0; h < nb_constants; h++) {

					string str;

					str.assign(argv[i0 + 2 * h + 0]);

					const_names.push_back(str);

					str.assign(argv[i0 + 2 * h + 1]);

					const_values.push_back(str);

				}
				if (f_v) {
					cout << "read " << nb_constants << " constants" << endl;
				}
			}
			else {
				cout << "cannot find -cost_end command" << endl;
				exit(1);
			}
		}
		else if (ST.stringcmp(argv[i], "-var") == 0) {
			if (f_v) {
				cout << "-var" << endl;
			}
			i0 = i + 1;
			for (++i; i < argc; i++) {
				if (ST.stringcmp(argv[i], "-var_end") == 0) {
					break;
				}
			}
			if (i < argc) {
				nb_variables = i - i0;


				for (h = 0; h < nb_variables; h++) {

					string str;

					str.assign(argv[i0 + h]);

					variable_names.push_back(str);

				}
				if (f_v) {
					cout << "read " << nb_variables << " variables" << endl;
				}
			}
			else {
				cout << "cannot find -var_end command" << endl;
				exit(1);
			}
		}
		else if (ST.stringcmp(argv[i], "-code") == 0) {
			if (f_v) {
				cout << "-code" << endl;
			}
			i0 = i + 1;
			for (++i; i < argc; i++) {
				if (ST.stringcmp(argv[i], "-code_end") == 0) {
					break;
				}
			}
			if (i < argc) {
				code_sz = i - i0;
				for (h = 0; h < code_sz; h++) {

					string str;

					str.assign(argv[i0 + h]);

					code.push_back(str);
				}
				if (f_v) {
					cout << "read " << code_sz << " code items" << endl;
					for (h = 0; h < code_sz; h++) {
						cout << h << " : " << code[h] << endl;
					}
				}
			}
			else {
				cout << "cannot find -code_end command" << endl;
				exit(1);
			}
		}
		else if (ST.stringcmp(argv[i], "-function_end") == 0) {
			if (f_v) {
				cout << "-function_end" << endl;
			}
			break;
		}
		else {
			cout << "function_polish_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "function_polish_description::read_arguments done" << endl;
	}
	return i + 1;
}

void function_polish_description::print()
{
	cout << "-function_polish_description" << endl;
}


}}


