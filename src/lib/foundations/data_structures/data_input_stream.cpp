/*
 * data_input_stream.cpp
 *
 *  Created on: Jan 18, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {

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

int data_input_stream::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "data_input_stream::read_arguments" << endl;
	if (argc) {
		cout << "data_input_stream::read_arguments next argument is " << argv[0] << endl;
	}
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-set_of_points") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_SET_OF_POINTS;

			//input_string[nb_inputs].assign(argv[++i]);

			os_interface Os;

			i++;
			Os.get_string_from_command_line(input_string[nb_inputs], argc, argv,
					i, verbose_level);
			i--;


			input_string2[nb_inputs].assign("");
			cout << "-set_of_points " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (stringcmp(argv[i], "-set_of_lines") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_SET_OF_LINES;
			input_string[nb_inputs].assign(argv[++i]);
			input_string2[nb_inputs].assign("");
			cout << "-set_of_lines " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (stringcmp(argv[i], "-set_of_packing") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_SET_OF_PACKING;
			input_string[nb_inputs].assign(argv[++i]);
			input_string2[nb_inputs].assign("");
			cout << "-set_of_packing " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (stringcmp(argv[i], "-file_of_points") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_FILE_OF_POINTS;
			input_string[nb_inputs].assign(argv[++i]);
			input_string2[nb_inputs].assign("");
			cout << "-file_of_points " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (stringcmp(argv[i], "-file_of_lines") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_FILE_OF_LINES;
			input_string[nb_inputs].assign(argv[++i]);
			input_string2[nb_inputs].assign("");
			cout << "-file_of_lines " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (stringcmp(argv[i], "-file_of_packings") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_FILE_OF_PACKINGS;
			input_string[nb_inputs].assign(argv[++i]);
			input_string2[nb_inputs].assign("");
			cout << "-file_of_packings " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (stringcmp(argv[i],
				"-file_of_packings_through_spread_table") == 0) {
			input_type[nb_inputs] =
					INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE;
			input_string[nb_inputs].assign(argv[++i]);
			input_string2[nb_inputs].assign(argv[++i]);
			cout << "-file_of_packings_through_spread_table "
				<< input_string[nb_inputs] << " "
				<< input_string2[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (stringcmp(argv[i], "-file_of_point_set") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_FILE_OF_POINT_SET;
			input_string[nb_inputs].assign(argv[++i]);
			input_string2[nb_inputs].assign("");
			cout << "-file_of_point_set " << input_string[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (stringcmp(argv[i], "-file_of_designs") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_FILE_OF_DESIGNS;
			input_string[nb_inputs].assign(argv[++i]);
			input_data1[nb_inputs] = strtoi(argv[++i]); // N_points
			input_data2[nb_inputs] = strtoi(argv[++i]); // b = number of blocks
			input_data3[nb_inputs] = strtoi(argv[++i]); // k = block size
			input_data4[nb_inputs] = strtoi(argv[++i]); // partition class size

			cout << "-file_of_designs " << input_string[nb_inputs]
					<< " " << input_data1[nb_inputs]
					<< " " << input_data2[nb_inputs]
					<< " " << input_data3[nb_inputs]
					<< " " << input_data4[nb_inputs]
					<< endl;
			nb_inputs++;
			}
		else if (stringcmp(argv[i], "-file_of_incidence_geometries") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_FILE_OF_INCIDENCE_GEOMETRIES;
			input_string[nb_inputs].assign(argv[++i]);
			input_data1[nb_inputs] = strtoi(argv[++i]); // v = number of points
			input_data2[nb_inputs] = strtoi(argv[++i]); // b = number of blocks
			input_data3[nb_inputs] = strtoi(argv[++i]); // f = numbr of flags
			cout << "-file_of_incidence_geometries " << input_string[nb_inputs]
				<< " " << input_data1[nb_inputs]
				<< " " << input_data2[nb_inputs]
				<< " " << input_data3[nb_inputs]
				<< endl;
			nb_inputs++;
			}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
			}
		else {
			cout << "data_input_stream::read_arguments unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "data_input_stream::read_arguments done" << endl;
	return i + 1;
}

void data_input_stream::print()
{
	int i;

	for (i = 0; i < nb_inputs; i++) {
		print_item(i);
	}
}

void data_input_stream::print_item(int i)
{
	if (input_type[i] == INPUT_TYPE_SET_OF_POINTS) {
		cout << "-set_of_points " << input_string[i] << endl;
	}
	else if (input_type[i] == INPUT_TYPE_SET_OF_LINES) {
		cout << "-set_of_lines " << input_string[i] << endl;
	}
	else if (input_type[i] == INPUT_TYPE_SET_OF_PACKING) {
		cout << "-set_of_packing " << input_string[i] << " " << input_string2[i] << endl;
	}
	else if (input_type[i] == INPUT_TYPE_FILE_OF_POINTS) {
		cout << "-file_of_points " << input_string[i] << endl;
	}
	else if (input_type[i] == INPUT_TYPE_FILE_OF_LINES) {
		cout << "-file_of_lines " << input_string[i] << endl;
	}
	else if (input_type[i] == INPUT_TYPE_FILE_OF_PACKINGS) {
		cout << "-file_of_packings " << input_string[i] << " " << input_string2[i] << endl;
	}
	else if (input_type[i] == INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
		cout << "-file_of_packings_through_spread_table "
			<< input_string[i] << " "
			<< input_string2[i] << endl;
	}
	else if (input_type[i] == INPUT_TYPE_FILE_OF_POINT_SET) {
		cout << "-file_of_point_set " << input_string[i] << " " << input_string2[i] << endl;
	}
	else if (input_type[i] == INPUT_TYPE_FILE_OF_DESIGNS) {
		cout << "-file_of_designs " << input_string[i]
				<< " " << input_data1[i]
				<< " " << input_data2[i]
				<< " " << input_data3[i]
				<< " " << input_data4[i]
				<< endl;
	}
	else if (input_type[i] == INPUT_TYPE_FILE_OF_INCIDENCE_GEOMETRIES) {
		cout << "-file_of_designs " << input_string[i]
				<< " " << input_data1[i]
				<< " " << input_data2[i]
				<< " " << input_data3[i]
				<< endl;
	}
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
		else if (input_type[input_idx] == INPUT_TYPE_FILE_OF_DESIGNS) {
			if (f_v) {
				cout << "input designs from file "
						<< input_string[input_idx] << ":" << endl;
			}
			{
			set_of_sets *SoS;
			int nck;
			combinatorics_domain Combi;

			nck = Combi.int_n_choose_k(input_data1[input_idx], input_data3[input_idx]);
			SoS = NEW_OBJECT(set_of_sets);

			cout << "classify_objects_using_nauty Reading the file " << input_string[input_idx]
				<<  " which contains designs on " << input_data1[input_idx] << " points, nck=" << nck << endl;
			SoS->init_from_file(
					nck /* underlying_set_size */,
					input_string[input_idx], verbose_level);
			cout << "Read the file " << input_string[input_idx] << endl;
			nb_obj = SoS->nb_sets;
			FREE_OBJECT(SoS);
			}
			if (f_v) {
				cout << "The file " << input_string[input_idx]
					<< " has " << nb_obj << " objects" << endl;
				}

			nb_objects_to_test += nb_obj;
			}
		else if (input_type[input_idx] == INPUT_TYPE_FILE_OF_INCIDENCE_GEOMETRIES) {
			if (f_v) {
				cout << "input incidence geometries from file "
						<< input_string[input_idx] << ":" << endl;
			}
			file_io Fio;
			int m, n, nb_flags;

			std::vector<std::vector<int> > Geos;

			Fio.read_incidence_file(Geos, m, n, nb_flags, input_string[input_idx], verbose_level);
			if (f_v) {
				cout << "input incidence geometries from file "
						"the file contains " << Geos.size() << "incidence geometries" << endl;
			}
			nb_objects_to_test += Geos.size();
			if (input_data1[input_idx] != m) {
				cout << "v does not match" << endl;
				exit(1);
			}
			if (input_data2[input_idx] != n) {
				cout << "b does not match" << endl;
				exit(1);
			}
			if (input_data3[input_idx] != nb_flags) {
				cout << "f does not match" << endl;
				exit(1);
			}
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

