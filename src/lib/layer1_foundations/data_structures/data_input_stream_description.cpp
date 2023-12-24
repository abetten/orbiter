/*
 * data_input_stream_description.cpp
 *
 *  Created on: Jan 18, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


data_input_stream_description::data_input_stream_description()
{
	f_label = false;
	//std::string label_txt;
	//std::string label_tex;

	nb_inputs = 0;
	//std::vector<data_input_stream_description_element> Input;
}

data_input_stream_description::~data_input_stream_description()
{
}


int data_input_stream_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	string_tools ST;

	if (f_v) {
		cout << "data_input_stream::read_arguments" << endl;
	}
	if (argc) {
		if (f_v) {
			cout << "data_input_stream::read_arguments "
				"next argument is " << argv[0] << endl;
		}
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-label") == 0) {
			f_label = true;
			label_txt.assign(argv[++i]);
			label_tex.assign(argv[++i]);
			if (f_v) {
				cout << "-label " << label_txt << " " << label_tex << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-set_of_points") == 0) {

			data_input_stream_description_element E;
			string a;

			a.assign(argv[++i]);
			E.init_set_of_points(a);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}
			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-set_of_lines") == 0) {

			data_input_stream_description_element E;
			string a;

			a.assign(argv[++i]);
			E.init_set_of_lines(a);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-set_of_points_and_lines") == 0) {

			data_input_stream_description_element E;
			string a, b;

			a.assign(argv[++i]);
			b.assign(argv[++i]);
			E.init_set_of_points_and_lines(a, b);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-set_of_packing") == 0) {

			data_input_stream_description_element E;
			string a;
			int q;

			a.assign(argv[++i]);
			q = ST.strtoi(argv[++i]);

			E.init_packing(a, q);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-file_of_points") == 0) {

			data_input_stream_description_element E;
			string a;

			a.assign(argv[++i]);

			E.init_file_of_points(a);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-file_of_points_csv") == 0) {

			data_input_stream_description_element E;
			string a, b;

			a.assign(argv[++i]);
			b.assign(argv[++i]);

			E.init_file_of_points_csv(a, b);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-file_of_lines") == 0) {

			data_input_stream_description_element E;
			string a;

			a.assign(argv[++i]);

			E.init_file_of_lines(a);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-file_of_packings") == 0) {

			data_input_stream_description_element E;
			string a;

			a.assign(argv[++i]);

			E.init_file_of_packings(a);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i],
				"-file_of_packings_through_spread_table") == 0) {

			data_input_stream_description_element E;
			string a, b;
			int q;

			a.assign(argv[++i]);
			b.assign(argv[++i]);
			q = ST.strtoi(argv[++i]);

			E.init_file_of_packings_through_spread_table(a, b, q);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-file_of_point_set") == 0) {

			data_input_stream_description_element E;
			string a;

			a.assign(argv[++i]);

			E.init_file_of_point_set(a);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-file_of_designs") == 0) {

			data_input_stream_description_element E;
			std::string a;
			int N_points, b, k, partition_class_size;

			a.assign(argv[++i]);
			N_points = ST.strtoi(argv[++i]);
			b = ST.strtoi(argv[++i]);
			k = ST.strtoi(argv[++i]);
			partition_class_size = ST.strtoi(argv[++i]);

			E.init_file_of_designs(a,
							N_points, b, k, partition_class_size);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-file_of_incidence_geometries") == 0) {

			data_input_stream_description_element E;
			std::string a;
			int v, b, f;

			a.assign(argv[++i]);
			v = ST.strtoi(argv[++i]);
			b = ST.strtoi(argv[++i]);
			f = ST.strtoi(argv[++i]);

			E.init_file_of_incidence_geometries(a, v, b, f);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-file_of_incidence_geometries_by_row_ranks") == 0) {

			data_input_stream_description_element E;
			std::string a;
			int v, b, r;

			a.assign(argv[++i]);
			v = ST.strtoi(argv[++i]);
			b = ST.strtoi(argv[++i]);
			r = ST.strtoi(argv[++i]);

			E.init_file_of_incidence_geometries_by_row_ranks(a, v, b, r);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-incidence_geometry") == 0) {

			data_input_stream_description_element E;
			std::string a;
			int v, b, f;

			a.assign(argv[++i]);
			v = ST.strtoi(argv[++i]);
			b = ST.strtoi(argv[++i]);
			f = ST.strtoi(argv[++i]);

			E.init_incidence_geometry(a, v, b, f);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-incidence_geometry_by_row_ranks") == 0) {

			data_input_stream_description_element E;
			std::string a;
			int v, b, r;

			a.assign(argv[++i]);
			v = ST.strtoi(argv[++i]);
			b = ST.strtoi(argv[++i]);
			r = ST.strtoi(argv[++i]);

			E.init_incidence_geometry_by_row_ranks(a, v, b, r);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-from_parallel_search") == 0) {

			data_input_stream_description_element E;
			string fname_mask;
			int nb_cases;
			string cases_fname;

			fname_mask.assign(argv[++i]);
			nb_cases = ST.strtoi(argv[++i]);
			cases_fname.assign(argv[++i]);
			E.init_from_parallel_search(fname_mask, nb_cases, cases_fname);
			Input.push_back(E);

			if (f_v) {
				E.print();
			}

			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "data_input_stream::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "data_input_stream::read_arguments done" << endl;
	}
	return i + 1;
}

void data_input_stream_description::print()
{
	int i;

	if (f_label) {
		cout << "-label " << label_txt << " " << label_tex << endl;
	}
	for (i = 0; i < nb_inputs; i++) {
		print_item(i);
	}
}

void data_input_stream_description::print_item(
		int i)
{
	Input[i].print();
}




}}}


