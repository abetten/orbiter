/*
 * data_input_stream_description.cpp
 *
 *  Created on: Jan 18, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {

data_input_stream_description::data_input_stream_description()
{
	nb_inputs = 0;
	//int input_type[1000];
	//std::string input_string[1000];
	//std::string input_string2[1000];

	// for INPUT_TYPE_FILE_OF_DESIGNS:
	//int input_data1[1000]; // N_points
	//int input_data2[1000]; // b = number of blocks
	//int input_data3[1000]; // k = block size
	//int input_data4[1000]; // partition class size
}

data_input_stream_description::~data_input_stream_description()
{
}

int data_input_stream_description::read_arguments(
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

			input_string[nb_inputs].assign(argv[++i]);

#if 0
			os_interface Os;

			i++;
			Os.get_string_from_command_line(input_string[nb_inputs], argc, argv,
					i, verbose_level);
			i--;
#endif

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
		else if (stringcmp(argv[i], "-set_of_points_and_lines") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_SET_OF_POINTS_AND_LINES;
			input_string[nb_inputs].assign(argv[++i]);
			input_string2[nb_inputs].assign(argv[++i]);
			cout << "-set_of_points_and_lines " << input_string[nb_inputs]
				<< " " << input_string2[nb_inputs] << endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i], "-set_of_packing") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_SET_OF_PACKING;
			input_string[nb_inputs].assign(argv[++i]);
			input_data1[nb_inputs] = strtoi(argv[++i]); // q
			cout << "-set_of_packing " << input_string[nb_inputs]
				<< " " << input_data1[nb_inputs] << endl;
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
			input_data1[nb_inputs] = strtoi(argv[++i]); // q
			cout << "-file_of_packings_through_spread_table "
				<< input_string[nb_inputs]
				<< " " << input_string2[nb_inputs]
				<< " " << input_data1[nb_inputs]
				<< endl;
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
			input_string[nb_inputs].assign(argv[++i]); // the file name
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
			input_string[nb_inputs].assign(argv[++i]); // the file name
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
		else if (stringcmp(argv[i], "-incidence_geometry") == 0) {
			input_type[nb_inputs] = INPUT_TYPE_INCIDENCE_GEOMETRY;
			input_string[nb_inputs].assign(argv[++i]); // the flags
			input_data1[nb_inputs] = strtoi(argv[++i]); // v = number of points
			input_data2[nb_inputs] = strtoi(argv[++i]); // b = number of blocks
			input_data3[nb_inputs] = strtoi(argv[++i]); // f = numbr of flags
			cout << "-incidence_geometry " << input_string[nb_inputs]
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

void data_input_stream_description::print()
{
	int i;

	for (i = 0; i < nb_inputs; i++) {
		print_item(i);
	}
}

void data_input_stream_description::print_item(int i)
{
	if (input_type[i] == INPUT_TYPE_SET_OF_POINTS) {
		cout << "-set_of_points " << input_string[i] << endl;
	}
	else if (input_type[i] == INPUT_TYPE_SET_OF_LINES) {
		cout << "-set_of_lines " << input_string[i] << endl;
	}
	else if (input_type[nb_inputs] == INPUT_TYPE_SET_OF_POINTS_AND_LINES) {
		cout << "-set_of_points_and_lines " << input_string[nb_inputs] << " " << input_string2[nb_inputs] << endl;
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
			<< input_string[nb_inputs]
			<< " " << input_string2[nb_inputs]
			<< " " << input_data1[nb_inputs]
			<< endl;
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
	else if (input_type[nb_inputs] == INPUT_TYPE_INCIDENCE_GEOMETRY) {
		cout << "-incidence_geometry " << input_string[nb_inputs]
			<< " " << input_data1[nb_inputs]
			<< " " << input_data2[nb_inputs]
			<< " " << input_data3[nb_inputs]
			<< endl;
	}
}



}}

