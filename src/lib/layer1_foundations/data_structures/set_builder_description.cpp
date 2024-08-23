/*
 * set_builder_description.cpp
 *
 *  Created on: Nov 7, 2020
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


set_builder_description::set_builder_description()
{
	f_loop = 0;
	loop_low = 0;
	loop_upper_bound = 0;
	loop_increment = 0;

	f_clone_with_affine_function = false;
	clone_with_affine_function_a = false;
	clone_with_affine_function_b = false;

	f_affine_function = false;
	affine_function_a = 0;
	affine_function_b = 0;

	f_set_builder = false;
	Descr = NULL;

	f_here = false;
	//std::string here_text;

	f_file = false;
	//std::string file_name;

	f_file_orbiter_format = false;
	//std::string file_orbiter_format_name;

}

set_builder_description::~set_builder_description()
{
	if (Descr) {
		FREE_OBJECT(Descr);
	}
}


int set_builder_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	string_tools ST;

	cout << "set_builder_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-loop") == 0) {
			f_loop = true;
			loop_low = ST.strtoi(argv[++i]);
			loop_upper_bound = ST.strtoi(argv[++i]);
			loop_increment = ST.strtoi(argv[++i]);
			cout << "-loop " << loop_low << " "
					<< loop_upper_bound << " "
					<< loop_increment << endl;
		}
		else if (ST.stringcmp(argv[i], "-affine_function") == 0) {
			f_affine_function = true;
			affine_function_a = ST.strtoi(argv[++i]);
			affine_function_b = ST.strtoi(argv[++i]);
			cout << "-affine_function " << affine_function_a << " " << affine_function_b << endl;
		}
		else if (ST.stringcmp(argv[i], "-clone_with_affine_function") == 0) {
			f_clone_with_affine_function = true;
			clone_with_affine_function_a = ST.strtoi(argv[++i]);
			clone_with_affine_function_b = ST.strtoi(argv[++i]);
			cout << "-clone_with_affine_function " << clone_with_affine_function_a << " " << clone_with_affine_function_b << endl;
		}
		else if (ST.stringcmp(argv[i], "-set_builder") == 0) {
			f_set_builder = true;
			Descr = NEW_OBJECT(set_builder_description);
			cout << "reading -set_builder" << endl;
			i += Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-set_builder" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-set_builder " << endl;
			Descr->print();
		}
		else if (ST.stringcmp(argv[i], "-here") == 0) {
			f_here = true;
			here_text.assign(argv[++i]);
			cout << "-here " << here_text << endl;
		}
		else if (ST.stringcmp(argv[i], "-file") == 0) {
			f_file = true;
			file_name.assign(argv[++i]);
			cout << "-file " << file_name << endl;
		}
		else if (ST.stringcmp(argv[i], "-file_orbiter_format") == 0) {
			f_file_orbiter_format = true;
			file_orbiter_format_name.assign(argv[++i]);
			cout << "-file_orbiter_format " << file_orbiter_format_name << endl;
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "set_builder_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "set_builder_description::read_arguments done" << endl;
	return i + 1;
}

void set_builder_description::print()
{
	cout << "set_builder_description:" << endl;
	if (f_loop) {
		cout << "-loop " << loop_low << " "
				<< loop_upper_bound << " " << loop_increment << endl;
	}
	if (f_affine_function) {
		cout << "-affine_function " << affine_function_a << " " << affine_function_b << endl;
	}
	if (f_clone_with_affine_function) {
		cout << "-clone_with_affine_function " << clone_with_affine_function_a << " " << clone_with_affine_function_b << endl;
	}
	if (f_set_builder) {
		cout << "-set_builder" << endl;
		Descr->print();
		cout << "-end" << endl;
	}
	if (f_here) {
		cout << "-here " << here_text << endl;
	}
	if (f_file) {
		cout << "-file " << file_name << endl;
	}
	if (f_file_orbiter_format) {
		cout << "-file_orbiter_format " << file_orbiter_format_name << endl;
	}
}




}}}



