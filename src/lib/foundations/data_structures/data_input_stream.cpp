/*
 * data_input_stream.cpp
 *
 *  Created on: Nov 27, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {

data_input_stream::data_input_stream()
{
	Descr = NULL;

	nb_objects_to_test = 0;

	// Objects;

}

data_input_stream::~data_input_stream()
{
}

void data_input_stream::init(data_input_stream_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "data_input_stream::init" << endl;
	}

	data_input_stream::Descr = Descr;

	nb_objects_to_test = count_number_of_objects_to_test(verbose_level);

	if (f_v) {
		cout << "data_input_stream::init nb_objects_to_test=" << nb_objects_to_test << endl;
	}

	if (f_v) {
		cout << "data_input_stream::init before read_objects" << endl;
	}

	read_objects(verbose_level);

	if (f_v) {
		cout << "data_input_stream::init after read_objects" << endl;
	}

	if (f_v) {
		cout << "data_input_stream::init done" << endl;
	}
}

int data_input_stream::count_number_of_objects_to_test(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int input_idx, nb_obj;
	int nb_objects_to_test;
	file_io Fio;

	if (f_v) {
		cout << "data_input_stream::count_number_of_objects_to_test" << endl;
	}
	nb_objects_to_test = 0;
	for (input_idx = 0; input_idx < Descr->nb_inputs; input_idx++) {
		cout << "input " << input_idx << " / " << Descr->nb_inputs
			<< " is:" << endl;

		if (Descr->input_type[input_idx] == INPUT_TYPE_SET_OF_POINTS) {
			if (f_v) {
				cout << "input set of points "
						<< Descr->input_string[input_idx] << ":" << endl;
			}

			nb_objects_to_test++;

		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_SET_OF_LINES) {
			if (f_v) {
				cout << "input set of lines "
						<< Descr->input_string[input_idx] << ":" << endl;
			}

			nb_objects_to_test++;

		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_SET_OF_POINTS_AND_LINES) {
			if (f_v) {
				cout << "input set of points and lines "
						<< Descr->input_string[input_idx] << " "
						<< Descr->input_string2[input_idx] << ":" << endl;
			}

			nb_objects_to_test++;

		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_SET_OF_PACKING) {
			if (f_v) {
				cout << "input set of packing "
						<< Descr->input_string[input_idx] << ":" << endl;
			}

			nb_objects_to_test++;

		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINTS) {
			if (f_v) {
				cout << "input sets of points from file "
						<< Descr->input_string[input_idx] << ":" << endl;
			}
			nb_obj = Fio.count_number_of_orbits_in_file(
					Descr->input_string[input_idx], 0 /* verbose_level*/);
			if (f_v) {
				cout << "The file " << Descr->input_string[input_idx]
					<< " has " << nb_obj << " objects" << endl;
			}

			nb_objects_to_test += nb_obj;
		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_LINES) {
			if (f_v) {
				cout << "input sets of lines from file "
					<< Descr->input_string[input_idx] << ":" << endl;
			}
			nb_obj = Fio.count_number_of_orbits_in_file(
					Descr->input_string[input_idx], 0 /* verbose_level*/);
			if (f_v) {
				cout << "The file " << Descr->input_string[input_idx]
					<< " has " << nb_obj << " objects" << endl;
			}

			nb_objects_to_test += nb_obj;
		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_PACKINGS) {
			if (f_v) {
				cout << "input sets of packings from file "
					<< Descr->input_string[input_idx] << ":" << endl;
			}
			nb_obj = Fio.count_number_of_orbits_in_file(
					Descr->input_string[input_idx], 0 /* verbose_level*/);
			if (f_v) {
				cout << "The file " << Descr->input_string[input_idx]
					<< " has " << nb_obj << " objects" << endl;
			}

			nb_objects_to_test += nb_obj;
		}
		else if (Descr->input_type[input_idx] ==
				INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
			if (f_v) {
				cout << "input sets of packings from file "
					<< Descr->input_string[input_idx] << endl;
				cout << "through spread table "
					<< Descr->input_string2[input_idx] << " :" << endl;
			}
			nb_obj = Fio.count_number_of_orbits_in_file(
					Descr->input_string[input_idx], 0 /* verbose_level*/);
			if (f_v) {
				cout << "The file " << Descr->input_string[input_idx]
					<< " has " << nb_obj << " objects" << endl;
			}

			nb_objects_to_test += nb_obj;
		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINT_SET) {
			if (f_v) {
				cout << "input set of points from file "
						<< Descr->input_string[input_idx] << ":" << endl;
			}
			nb_obj = 1;
			if (f_v) {
				cout << "The file " << Descr->input_string[input_idx]
					<< " has " << nb_obj << " objects" << endl;
			}

			nb_objects_to_test += nb_obj;
		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_DESIGNS) {
			if (f_v) {
				cout << "input designs from file "
						<< Descr->input_string[input_idx] << ":" << endl;
			}
			{
				set_of_sets *SoS;
				int nck;
				combinatorics_domain Combi;

				nck = Combi.int_n_choose_k(Descr->input_data1[input_idx], Descr->input_data3[input_idx]);
				SoS = NEW_OBJECT(set_of_sets);

				cout << "classify_objects_using_nauty Reading the file " << Descr->input_string[input_idx]
					<<  " which contains designs on " << Descr->input_data1[input_idx] << " points, nck=" << nck << endl;
				SoS->init_from_file(
						nck /* underlying_set_size */,
						Descr->input_string[input_idx], verbose_level);
				cout << "Read the file " << Descr->input_string[input_idx] << endl;
				nb_obj = SoS->nb_sets;
				FREE_OBJECT(SoS);
			}
			if (f_v) {
				cout << "The file " << Descr->input_string[input_idx]
					<< " has " << nb_obj << " objects" << endl;
			}

			nb_objects_to_test += nb_obj;
		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_INCIDENCE_GEOMETRIES) {
			if (f_v) {
				cout << "input incidence geometries from file "
						<< Descr->input_string[input_idx] << ":" << endl;
			}
			file_io Fio;
			int m, n, nb_flags;

			std::vector<std::vector<int> > Geos;

			Fio.read_incidence_file(Geos, m, n, nb_flags, Descr->input_string[input_idx], verbose_level);
			if (f_v) {
				cout << "input incidence geometries from file "
						"the file contains " << Geos.size() << "incidence geometries" << endl;
			}
			nb_objects_to_test += Geos.size();
			if (Descr->input_data1[input_idx] != m) {
				cout << "v does not match" << endl;
				exit(1);
			}
			if (Descr->input_data2[input_idx] != n) {
				cout << "b does not match" << endl;
				exit(1);
			}
			if (Descr->input_data3[input_idx] != nb_flags) {
				cout << "nb_flags does not match" << endl;
				exit(1);
			}
		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_INCIDENCE_GEOMETRY) {
			if (f_v) {
				cout << "input incidence geometry directly "
						<< Descr->input_string[input_idx] << ":" << endl;
			}
			nb_objects_to_test++;
		}
		else {
			cout << "unknown input type" << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "data_input_stream::count_number_of_objects_to_test done" << endl;
	}
	return nb_objects_to_test;
}

void data_input_stream::read_objects(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int input_idx;


	if (f_v) {
		cout << "data_input_stream::read_objects" << endl;
	}

	for (input_idx = 0; input_idx < Descr->nb_inputs; input_idx++) {
		if (f_v) {
			cout << "data_input_stream::read_objects "
					"input " << input_idx << " / " << Descr->nb_inputs
					<< " is:" << endl;
		}

		if (Descr->input_type[input_idx] == INPUT_TYPE_SET_OF_POINTS) {

			if (f_v) {
				cout << "data_input_stream::read_objects "
						"input set of points "
					<< Descr->input_string[input_idx] << ":" << endl;
			}

			object_with_canonical_form *OwCF;


			OwCF = NEW_OBJECT(object_with_canonical_form);

			OwCF->init_point_set_from_string(
					Descr->input_string[input_idx] /*set_text*/,
					verbose_level);

			Objects.push_back(OwCF);

		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINT_SET) {

			file_io Fio;
			long int *the_set;
			int set_size;
			object_with_canonical_form *OwCF;

			Fio.read_set_from_file(Descr->input_string[input_idx], the_set, set_size, verbose_level);

			OwCF = NEW_OBJECT(object_with_canonical_form);

			OwCF->init_point_set(the_set, set_size, verbose_level);

			FREE_lint(the_set);

			Objects.push_back(OwCF);

		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_SET_OF_LINES) {

			object_with_canonical_form *OwCF;

			OwCF = NEW_OBJECT(object_with_canonical_form);

			OwCF->init_line_set_from_string(
					Descr->input_string[input_idx] /*set_text*/,
					verbose_level);

			Objects.push_back(OwCF);

		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_SET_OF_POINTS_AND_LINES) {

			object_with_canonical_form *OwCF;


			OwCF = NEW_OBJECT(object_with_canonical_form);

			OwCF->init_points_and_lines_from_string(
					Descr->input_string[input_idx] /*set_text*/,
					Descr->input_string2[input_idx] /*set2_text*/,
					verbose_level);

			Objects.push_back(OwCF);

		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_SET_OF_PACKING) {

			object_with_canonical_form *OwCF;
			int q;

			q = Descr->input_data1[input_idx];


			OwCF = NEW_OBJECT(object_with_canonical_form);

			OwCF->init_packing_from_string(
					Descr->input_string[input_idx] /*packing_text*/,
					q,
					verbose_level);

			Objects.push_back(OwCF);

		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_INCIDENCE_GEOMETRY) {

			object_with_canonical_form *OwCF;


			OwCF = NEW_OBJECT(object_with_canonical_form);

			OwCF->init_incidence_geometry_from_string(
					Descr->input_string[input_idx],
					Descr->input_data1[input_idx] /*v*/,
					Descr->input_data2[input_idx] /*b*/,
					Descr->input_data3[input_idx] /*nb_flags*/,
					verbose_level);

			Objects.push_back(OwCF);

		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_INCIDENCE_GEOMETRIES) {
			if (f_v) {
				cout << "input incidence geometries from file "
						<< Descr->input_string[input_idx] << ":" << endl;
			}
			file_io Fio;
			int m, n, nb_flags;

			std::vector<std::vector<int> > Geos;

			Fio.read_incidence_file(Geos, m, n, nb_flags, Descr->input_string[input_idx], verbose_level);
			if (f_v) {
				cout << "input incidence geometries from file "
						"the file contains " << Geos.size() << "incidence geometries" << endl;
			}
			int h;

			if (Descr->input_data1[input_idx] != m) {
				cout << "v does not match" << endl;
				exit(1);
			}
			if (Descr->input_data2[input_idx] != n) {
				cout << "b does not match" << endl;
				exit(1);
			}
			if (Descr->input_data3[input_idx] != nb_flags) {
				cout << "f does not match" << endl;
				exit(1);
			}

			for (h = 0; h < Geos.size(); h++) {
				object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(object_with_canonical_form);

				OwCF->init_incidence_geometry_from_vector(
						Geos[h],
						Descr->input_data1[input_idx] /*v*/,
						Descr->input_data2[input_idx] /*b*/,
						Descr->input_data3[input_idx] /*nb_flags*/,
						verbose_level);

				Objects.push_back(OwCF);

			}
		}
		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_DESIGNS) {

			int v, b, k, design_sz;

			v = Descr->input_data1[input_idx];
			b = Descr->input_data2[input_idx];
			k = Descr->input_data3[input_idx];
			design_sz = Descr->input_data4[input_idx];

			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			int underlying_set_size = 0;

			if (f_v) {
				cout << "data_input_stream::read_objects "
						"Reading the file " << Descr->input_string[input_idx] << endl;
			}
			SoS->init_from_file(
					underlying_set_size,
					Descr->input_string[input_idx], verbose_level);
			if (f_v) {
				cout << "Read the file " << Descr->input_string[input_idx] << ", underlying_set_size=" << underlying_set_size << endl;
			}

			int h;

			for (h = 0; h < SoS->nb_sets; h++) {


				object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(object_with_canonical_form);

				OwCF->init_large_set(
						SoS->Sets[h], SoS->Set_size[h], v, b, k, design_sz,
						verbose_level);

				Objects.push_back(OwCF);
			}

			FREE_OBJECT(SoS);
		}

		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {

			int q;

			q = Descr->input_data1[input_idx];

			file_io Fio;
			long int *Spread_table;
			int nb_spreads;
			int spread_size;

			if (f_v) {
				cout << "data_input_stream::read_objects "
						"Reading spread table from file "
					<< Descr->input_string2[input_idx] << endl;
			}

			Fio.lint_matrix_read_csv(Descr->input_string2[input_idx],
					Spread_table, nb_spreads, spread_size,
					0 /* verbose_level */);

			if (f_v) {
				cout << "Reading spread table from file "
						<< Descr->input_string2[input_idx] << " done" << endl;
				cout << "The spread table contains " << nb_spreads
						<< " spreads" << endl;
			}


			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			int underlying_set_size = 0;

			if (f_v) {
				cout << "projective_space_object_classifier::process_multiple_objects_from_file "
						"Reading the file " << Descr->input_string[input_idx] << endl;
			}
			SoS->init_from_file(
					underlying_set_size,
					Descr->input_string[input_idx], verbose_level);
			if (f_v) {
				cout << "Read the file " << Descr->input_string[input_idx] << ", underlying_set_size=" << underlying_set_size << endl;
			}

			int h;

			for (h = 0; h < SoS->nb_sets; h++) {


				object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(object_with_canonical_form);

				OwCF->init_packing_from_spread_table(
						SoS->Sets[h],
						Spread_table, nb_spreads, spread_size,
						q,
						verbose_level);

				Objects.push_back(OwCF);
			}
			FREE_lint(Spread_table);

			FREE_OBJECT(SoS);

		}

		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_PACKINGS) {

			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			int underlying_set_size = 0;

			if (f_v) {
				cout << "projective_space_object_classifier::process_multiple_objects_from_file "
						"Reading the file " << Descr->input_string[input_idx] << endl;
			}
			SoS->init_from_file(
					underlying_set_size,
					Descr->input_string[input_idx], verbose_level);
			if (f_v) {
				cout << "Read the file " << Descr->input_string[input_idx] << ", underlying_set_size=" << underlying_set_size << endl;
			}


			int h;

			for (h = 0; h < SoS->nb_sets; h++) {


				object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(object_with_canonical_form);

				OwCF->init_packing_from_set(
						SoS->Sets[h], SoS->Set_size[h], verbose_level);


				Objects.push_back(OwCF);
			}

			FREE_OBJECT(SoS);

		}


		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_LINES) {

			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			int underlying_set_size = 0;

			if (f_v) {
				cout << "projective_space_object_classifier::process_multiple_objects_from_file "
						"Reading the file " << Descr->input_string[input_idx] << endl;
			}
			SoS->init_from_file(
					underlying_set_size,
					Descr->input_string[input_idx], verbose_level);
			if (f_v) {
				cout << "Read the file " << Descr->input_string[input_idx] << ", underlying_set_size=" << underlying_set_size << endl;
			}


			int h;

			for (h = 0; h < SoS->nb_sets; h++) {


				object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(object_with_canonical_form);

				OwCF->init_line_set(
						SoS->Sets[h], SoS->Set_size[h], verbose_level);


				Objects.push_back(OwCF);
			}

			FREE_OBJECT(SoS);

		}


		else if (Descr->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINTS) {

			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			int underlying_set_size = 0;

			if (f_v) {
				cout << "projective_space_object_classifier::process_multiple_objects_from_file "
						"Reading the file " << Descr->input_string[input_idx] << endl;
			}
			SoS->init_from_file(
					underlying_set_size,
					Descr->input_string[input_idx], verbose_level);
			if (f_v) {
				cout << "Read the file " << Descr->input_string[input_idx] << ", underlying_set_size=" << underlying_set_size << endl;
			}


			int h;

			for (h = 0; h < SoS->nb_sets; h++) {


				object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(object_with_canonical_form);

				OwCF->init_point_set(
						SoS->Sets[h], SoS->Set_size[h], verbose_level);


				Objects.push_back(OwCF);
			}

			FREE_OBJECT(SoS);

		}

		else {
			cout << "data_input_stream::read_objects "
					"unknown input type " << Descr->input_type[input_idx] << endl;
			exit(1);
		}
	}

	if (nb_objects_to_test != Objects.size()) {
		cout << "data_input_stream::read_objects nb_objects_to_test != Objects.size()" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "data_input_stream::read_objects done" << endl;
	}

}

}}
