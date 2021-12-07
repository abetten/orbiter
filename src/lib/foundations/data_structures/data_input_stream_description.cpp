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

			string a;

			a.assign(argv[++i]);
			add_set_of_points(a);

			cout << "-set_of_points " << input_string[nb_inputs] << endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i], "-set_of_lines") == 0) {

			string a;

			a.assign(argv[++i]);
			add_set_of_lines(a);

			cout << "-set_of_lines " << input_string[nb_inputs] << endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i], "-set_of_points_and_lines") == 0) {

			string a, b;

			a.assign(argv[++i]);
			b.assign(argv[++i]);
			add_set_of_points_and_lines(a, b);

			cout << "-set_of_points_and_lines " << input_string[nb_inputs]
				<< " " << input_string2[nb_inputs] << endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i], "-set_of_packing") == 0) {

			string a;
			int q;

			a.assign(argv[++i]);
			q = strtoi(argv[++i]);

			add_packing(a, q);

			cout << "-set_of_packing " << input_string[nb_inputs]
				<< " " << input_data1[nb_inputs] << endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i], "-file_of_points") == 0) {

			string a;

			a.assign(argv[++i]);

			add_file_of_points(a);

			cout << "-file_of_points " << input_string[nb_inputs] << endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i], "-file_of_lines") == 0) {

			string a;

			a.assign(argv[++i]);

			add_file_of_lines(a);

			cout << "-file_of_lines " << input_string[nb_inputs] << endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i], "-file_of_packings") == 0) {

			string a;

			a.assign(argv[++i]);

			add_file_of_packings(a);

			cout << "-file_of_packings " << input_string[nb_inputs] << endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i],
				"-file_of_packings_through_spread_table") == 0) {

			string a, b;
			int q;

			a.assign(argv[++i]);
			b.assign(argv[++i]);
			q = strtoi(argv[++i]);

			add_file_of_packings_through_spread_table(a, b, q);

			cout << "-file_of_packings_through_spread_table "
				<< input_string[nb_inputs]
				<< " " << input_string2[nb_inputs]
				<< " " << input_data1[nb_inputs]
				<< endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i], "-file_of_point_set") == 0) {

			string a;

			a.assign(argv[++i]);

			add_file_of_point_set(a);


			cout << "-file_of_point_set " << input_string[nb_inputs] << endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i], "-file_of_designs") == 0) {

			std::string a;
			int N_points, b, k, partition_class_size;

			a.assign(argv[++i]);
			N_points = strtoi(argv[++i]);
			b = strtoi(argv[++i]);
			k = strtoi(argv[++i]);
			partition_class_size = strtoi(argv[++i]);

			add_file_of_designs(a,
							N_points, b, k, partition_class_size);


			cout << "-file_of_designs " << input_string[nb_inputs]
					<< " " << input_data1[nb_inputs]
					<< " " << input_data2[nb_inputs]
					<< " " << input_data3[nb_inputs]
					<< " " << input_data4[nb_inputs]
					<< endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i], "-file_of_incidence_geometries") == 0) {

			std::string a;
			int v, b, f;

			a.assign(argv[++i]);
			v = strtoi(argv[++i]);
			b = strtoi(argv[++i]);
			f = strtoi(argv[++i]);

			add_file_of_incidence_geometries(a, v, b, f);
			cout << "-file_of_incidence_geometries " << input_string[nb_inputs]
				<< " " << input_data1[nb_inputs]
				<< " " << input_data2[nb_inputs]
				<< " " << input_data3[nb_inputs]
				<< endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i], "-incidence_geometry") == 0) {

			std::string a;
			int v, b, f;

			a.assign(argv[++i]);
			v = strtoi(argv[++i]);
			b = strtoi(argv[++i]);
			f = strtoi(argv[++i]);

			add_incidence_geometry(a, v, b, f);

			cout << "-incidence_geometry " << input_string[nb_inputs]
				<< " " << input_data1[nb_inputs]
				<< " " << input_data2[nb_inputs]
				<< " " << input_data3[nb_inputs]
				<< endl;
			nb_inputs++;
		}
		else if (stringcmp(argv[i], "-from_parallel_search") == 0) {

			string fname_mask;
			int nb_cases;
			string cases_fname;

			fname_mask.assign(argv[++i]);
			nb_cases = strtoi(argv[++i]);
			cases_fname.assign(argv[++i]);
			add_from_parallel_search(fname_mask, nb_cases, cases_fname);

			cout << "-from_parallel_search"
					<< " " << input_string[nb_inputs]
					<< " " << input_string2[nb_inputs]
					<< " " << input_data1[nb_inputs]
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
		cout << "-set_of_points_and_lines "
				<< input_string[i]
				<< " " << input_string2[i] << endl;
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
			<< input_string[i]
			<< " " << input_string2[i]
			<< " " << input_data1[i]
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
		cout << "-incidence_geometry " << input_string[i]
			<< " " << input_data1[i]
			<< " " << input_data2[i]
			<< " " << input_data3[i]
			<< endl;
	}
	else if (input_type[nb_inputs] == INPUT_TYPE_FROM_PARALLEL_SEARCH) {
		cout << "-from_parallel_search"
				<< " " << input_string[i]
				<< " " << input_string2[i]
				<< " " << input_data1[i]
			<< endl;
	}
}

void data_input_stream_description::add_set_of_points(std::string &a)
{
	input_type[nb_inputs] = INPUT_TYPE_SET_OF_POINTS;

	input_string[nb_inputs].assign(a);

}

void data_input_stream_description::add_set_of_lines(std::string &a)
{
	input_type[nb_inputs] = INPUT_TYPE_SET_OF_LINES;

	input_string[nb_inputs].assign(a);

}

void data_input_stream_description::add_set_of_points_and_lines(std::string &a, std::string &b)
{
	input_type[nb_inputs] = INPUT_TYPE_SET_OF_POINTS_AND_LINES;

	input_string[nb_inputs].assign(a);
	input_string2[nb_inputs].assign(b);

}

void data_input_stream_description::add_packing(std::string &a, int q)
{
	input_type[nb_inputs] = INPUT_TYPE_SET_OF_PACKING;

	input_string[nb_inputs].assign(a);
	input_data1[nb_inputs] = q;

}

void data_input_stream_description::add_file_of_points(std::string &a)
{
	input_type[nb_inputs] = INPUT_TYPE_FILE_OF_POINTS;

	input_string[nb_inputs].assign(a);

}

void data_input_stream_description::add_file_of_lines(std::string &a)
{
	input_type[nb_inputs] = INPUT_TYPE_FILE_OF_LINES;

	input_string[nb_inputs].assign(a);

}

void data_input_stream_description::add_file_of_packings(std::string &a)
{
	input_type[nb_inputs] = INPUT_TYPE_FILE_OF_PACKINGS;

	input_string[nb_inputs].assign(a);

}

void data_input_stream_description::add_file_of_packings_through_spread_table(
		std::string &a, std::string &b, int q)
{
	input_type[nb_inputs] = INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE;

	input_string[nb_inputs].assign(a);
	input_string2[nb_inputs].assign(b);
	input_data1[nb_inputs] = q;
}

void data_input_stream_description::add_file_of_point_set(std::string &a)
{
	input_type[nb_inputs] = INPUT_TYPE_FILE_OF_POINT_SET;

	input_string[nb_inputs].assign(a);

}

void data_input_stream_description::add_file_of_designs(std::string &a,
			int N_points, int b, int k, int partition_class_size)
{
	input_type[nb_inputs] = INPUT_TYPE_FILE_OF_DESIGNS;

	input_string[nb_inputs].assign(a);
	input_data1[nb_inputs] = N_points;
	input_data2[nb_inputs] = b;
	input_data3[nb_inputs] = k;
	input_data4[nb_inputs] = partition_class_size;

}

void data_input_stream_description::add_file_of_incidence_geometries(std::string &a,
			int v, int b, int f)
{
	input_type[nb_inputs] = INPUT_TYPE_FILE_OF_INCIDENCE_GEOMETRIES;

	input_string[nb_inputs].assign(a);
	input_data1[nb_inputs] = v;
	input_data2[nb_inputs] = b;
	input_data3[nb_inputs] = f;

}

void data_input_stream_description::add_incidence_geometry(std::string &a,
			int v, int b, int f)
{
	input_type[nb_inputs] = INPUT_TYPE_INCIDENCE_GEOMETRY;

	input_string[nb_inputs].assign(a);
	input_data1[nb_inputs] = v;
	input_data2[nb_inputs] = b;
	input_data3[nb_inputs] = f;

}

void data_input_stream_description::add_from_parallel_search(std::string &fname_mask,
		int nb_cases, std::string &cases_fname)
{
	input_type[nb_inputs] = INPUT_TYPE_FROM_PARALLEL_SEARCH;

	input_string[nb_inputs].assign(fname_mask);
	input_string2[nb_inputs].assign(cases_fname);
	input_data1[nb_inputs] = nb_cases;

}



}}

