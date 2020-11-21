/*
 * function_polish_description.cpp
 *
 *  Created on: Apr 18, 2020
 *      Author: betten
 */


#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


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


#if 0
void function_polish_description::read_arguments_from_string(
		const char *str, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	int argc;
	char **argv;
	int i;

	if (f_v) {
		cout << "function_polish_description::read_arguments_from_string" << endl;
	}
	chop_string(str, argc, argv);

	if (f_vv) {
		cout << "argv:" << endl;
		for (i = 0; i < argc; i++) {
			cout << i << " : " << argv[i] << endl;
		}
	}


	read_arguments(
		argc, (const char **) argv,
		verbose_level);

	for (i = 0; i < argc; i++) {
		FREE_char(argv[i]);
	}
	FREE_pchar(argv);
	if (f_v) {
		cout << "function_polish_description::read_arguments_from_string "
				"done" << endl;
	}
}
#endif

int function_polish_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i, i0, h;

	cout << "function_polish_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-const") == 0) {
			cout << "-const" << endl;
			i0 = i + 1;
			for (++i; i < argc; i++) {
				if (stringcmp(argv[i], "-const_end") == 0) {
					break;
				}
			}
			if (i < argc) {
				nb_constants = (i - i0) >> 1;
				//const_names = NEW_pchar(nb_constants);
				//const_values = NEW_pchar(nb_constants);

				for (h = 0; h < nb_constants; h++) {

					string str;

					str.assign(argv[i0 + 2 * h + 0]);

					const_names.push_back(str);

					str.assign(argv[i0 + 2 * h + 1]);

					const_values.push_back(str);

#if 0
					l = strlen(argv[i0 + 2 * h + 0]);
					const_names[h] = NEW_char(l + 1);
					strcpy(const_names[h], argv[i0 + 2 * h + 0]);
					l = strlen(argv[i0 + 2 * h + 1]);
					const_values[h] = NEW_char(l + 1);
					strcpy(const_values[h], argv[i0 + 2 * h + 1]);
#endif
				}
				cout << "read " << nb_constants << " constants" << endl;
			}
			else {
				cout << "cannod find -cost_end command" << endl;
				exit(1);
			}
		}
		else if (stringcmp(argv[i], "-var") == 0) {
			cout << "-var" << endl;
			i0 = i + 1;
			for (++i; i < argc; i++) {
				if (stringcmp(argv[i], "-var_end") == 0) {
					break;
				}
			}
			if (i < argc) {
				nb_variables = i - i0;


				//variable_names = NEW_pchar(nb_variables);
				for (h = 0; h < nb_variables; h++) {

					string str;

					str.assign(argv[i0 + h]);

					variable_names.push_back(str);

#if 0
					l = strlen(argv[i0 + h]);
					variable_names[h] = NEW_char(l + 1);
					strcpy(variable_names[h], argv[i0 + h]);
#endif
				}
				cout << "read " << nb_variables << " variables" << endl;
			}
			else {
				cout << "cannod find -var_end command" << endl;
				exit(1);
			}
		}
		else if (stringcmp(argv[i], "-code") == 0) {
			cout << "-code" << endl;
			i0 = i + 1;
			for (++i; i < argc; i++) {
				if (stringcmp(argv[i], "-code_end") == 0) {
					break;
				}
			}
			if (i < argc) {
				code_sz = i - i0;
				//code = NEW_pchar(code_sz);
				for (h = 0; h < code_sz; h++) {

					string str;

					str.assign(argv[i0 + h]);

					code.push_back(str);
#if 0
					l = strlen(argv[i0 + h]);
					code[h] = NEW_char(l + 1);
					strcpy(code[h], argv[i0 + h]);
#endif
				}
				cout << "read " << code_sz << " code items" << endl;
				for (h = 0; h < code_sz; h++) {
					cout << h << " : " << code[h] << endl;
				}
			}
			else {
				cout << "cannod find -code_end command" << endl;
				exit(1);
			}
		}
		else if (stringcmp(argv[i], "-function_end") == 0) {
			cout << "-function_end" << endl;
			break;
		}
		else {
			cout << "function_polish_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "function_polish_description::read_arguments done" << endl;
	return i + 1;
}

}}


