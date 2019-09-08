/*
 * data_input_stream.cpp
 *
 *  Created on: Jan 18, 2019
 *      Author: betten
 */


#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {

data_input_stream::data_input_stream()
{
	null();
}

data_input_stream::~data_input_stream()
{
	freeself();
}

void data_input_stream::null()
{
	nb_inputs = 0;
}

void data_input_stream::freeself()
{
	null();
}

void data_input_stream::read_arguments_from_string(
		const char *str, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	int argc;
	char **argv;
	int i;

	if (f_v) {
		cout << "data_input_stream::read_arguments_from_string" << endl;
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
		cout << "data_input_stream::read_arguments_from_string "
				"done" << endl;
	}
}

int data_input_stream::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "data_input_stream::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

#if 0
		if (argv[i][0] != '-') {
			continue;
			}
#endif

		if (strcmp(argv[i], "-set_of_points") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_SET_OF_POINTS;
			input_string[nb_inputs] = argv[++i];
			input_string2[nb_inputs] = NULL;
			cout << "data_input_stream::read_arguments -set_of_points " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (strcmp(argv[i], "-set_of_lines") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_SET_OF_LINES;
			input_string[nb_inputs] = argv[++i];
			input_string2[nb_inputs] = NULL;
			cout << "data_input_stream::read_arguments -set_of_lines " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (strcmp(argv[i], "-set_of_packing") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_SET_OF_PACKING;
			input_string[nb_inputs] = argv[++i];
			input_string2[nb_inputs] = NULL;
			cout << "data_input_stream::read_arguments -set_of_packing " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (strcmp(argv[i], "-file_of_points") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_FILE_OF_POINTS;
			input_string[nb_inputs] = argv[++i];
			input_string2[nb_inputs] = NULL;
			cout << "data_input_stream::read_arguments -file_of_points " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (strcmp(argv[i], "-file_of_lines") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_FILE_OF_LINES;
			input_string[nb_inputs] = argv[++i];
			input_string2[nb_inputs] = NULL;
			cout << "data_input_stream::read_arguments -file_of_lines " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (strcmp(argv[i], "-file_of_packings") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_FILE_OF_PACKINGS;
			input_string[nb_inputs] = argv[++i];
			input_string2[nb_inputs] = NULL;
			cout << "data_input_stream::read_arguments -file_of_packings " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (strcmp(argv[i],
				"-file_of_packings_through_spread_table") == 0) {
			input_type[nb_inputs] =
					INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE;
			input_string[nb_inputs] = argv[++i];
			input_string2[nb_inputs] = argv[++i];
			cout << "data_input_stream::read_arguments -file_of_packings_through_spread_table "
				<< input_string[nb_inputs] << " "
				<< input_string2[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (strcmp(argv[i], "-file_of_point_set") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_FILE_OF_POINT_SET;
			input_string[nb_inputs] = argv[++i];
			input_string2[nb_inputs] = NULL;
			cout << "data_input_stream::read_arguments -file_of_point_set " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "data_input_stream::read_arguments -end" << endl;
			return i;
			}
		else {
			cout << "data_input_stream::read_arguments unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "data_input_stream::read_arguments done" << endl;
	return i + 1;
}

int data_input_stream::count_number_of_objects_to_test(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int input_idx, nb_obj;
	int nb_objects_to_test;
	file_io Fio;

	if (f_v) {
		cout << "data_input_stream::count_number_of_objects_to_test" << endl;
		}
	nb_objects_to_test = 0;
	for (input_idx = 0; input_idx < nb_inputs; input_idx++) {
		cout << "input " << input_idx << " / " << nb_inputs
			<< " is:" << endl;

		if (input_type[input_idx] == INPUT_TYPE_SET_OF_POINTS) {
			if (f_v) {
				cout << "input set of points "
						<< input_string[input_idx] << ":" << endl;
				}

			nb_objects_to_test++;

			}
		else if (input_type[input_idx] == INPUT_TYPE_SET_OF_LINES) {
			if (f_v) {
				cout << "input set of lines "
						<< input_string[input_idx] << ":" << endl;
				}

			nb_objects_to_test++;

			}
		else if (input_type[input_idx] == INPUT_TYPE_SET_OF_PACKING) {
			if (f_v) {
				cout << "input set of packing "
						<< input_string[input_idx] << ":" << endl;
				}

			nb_objects_to_test++;

			}
		else if (input_type[input_idx] == INPUT_TYPE_FILE_OF_POINTS) {
			if (f_v) {
				cout << "input sets of points from file "
						<< input_string[input_idx] << ":" << endl;
				}
			nb_obj = Fio.count_number_of_orbits_in_file(
					input_string[input_idx], 0 /* verbose_level*/);
			if (f_v) {
				cout << "The file " << input_string[input_idx]
					<< " has " << nb_obj << " objects" << endl;
				}

			nb_objects_to_test += nb_obj;
			}
		else if (input_type[input_idx] == INPUT_TYPE_FILE_OF_LINES) {
			if (f_v) {
				cout << "input sets of lines from file "
					<< input_string[input_idx] << ":" << endl;
				}
			nb_obj = Fio.count_number_of_orbits_in_file(
				input_string[input_idx], 0 /* verbose_level*/);
			if (f_v) {
				cout << "The file " << input_string[input_idx]
					<< " has " << nb_obj << " objects" << endl;
				}

			nb_objects_to_test += nb_obj;
			}
		else if (input_type[input_idx] == INPUT_TYPE_FILE_OF_PACKINGS) {
			if (f_v) {
				cout << "input sets of packings from file "
					<< input_string[input_idx] << ":" << endl;
				}
			nb_obj = Fio.count_number_of_orbits_in_file(
				input_string[input_idx], 0 /* verbose_level*/);
			if (f_v) {
				cout << "The file " << input_string[input_idx]
					<< " has " << nb_obj << " objects" << endl;
				}

			nb_objects_to_test += nb_obj;
			}
		else if (input_type[input_idx] ==
				INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
			if (f_v) {
				cout << "input sets of packings from file "
					<< input_string[input_idx] << endl;
				cout << "through spread table "
					<< input_string2[input_idx] << " :" << endl;
				}
			nb_obj = Fio.count_number_of_orbits_in_file(
				input_string[input_idx], 0 /* verbose_level*/);
			if (f_v) {
				cout << "The file " << input_string[input_idx]
					<< " has " << nb_obj << " objects" << endl;
				}

			nb_objects_to_test += nb_obj;
			}
		else if (input_type[input_idx] == INPUT_TYPE_FILE_OF_POINT_SET) {
			if (f_v) {
				cout << "input set of points from file "
						<< input_string[input_idx] << ":" << endl;
				}
			nb_obj = 1;
			if (f_v) {
				cout << "The file " << input_string[input_idx]
					<< " has " << nb_obj << " objects" << endl;
				}

			nb_objects_to_test += nb_obj;
			}
		else {
			cout << "unknown input type" << endl;
			exit(1);
			}
		}

	if (f_v) {
		cout << "count_number_of_objects_to_test done" << endl;
		}
	return nb_objects_to_test;
}


}}

