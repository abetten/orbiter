/*
 * data_input_stream.cpp
 *
 *  Created on: Nov 27, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


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
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "data_input_stream::count_number_of_objects_to_test" << endl;
	}
	nb_objects_to_test = 0;
	for (input_idx = 0; input_idx < Descr->nb_inputs; input_idx++) {
		cout << "input " << input_idx << " / " << Descr->nb_inputs
			<< " is:" << endl;

		if (Descr->Input[input_idx].input_type == t_data_input_stream_set_of_points) {
			if (f_v) {
				cout << "input set of points "
						<< Descr->Input[input_idx].input_string << ":" << endl;
			}

			nb_objects_to_test++;

		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_set_of_lines) {
			if (f_v) {
				cout << "input set of lines "
						<< Descr->Input[input_idx].input_string << ":" << endl;
			}

			nb_objects_to_test++;

		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_set_of_points_and_lines) {
			if (f_v) {
				cout << "input set of points and lines "
						<< Descr->Input[input_idx].input_string << " "
						<< Descr->Input[input_idx].input_string2 << ":" << endl;
			}

			nb_objects_to_test++;

		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_set_of_packing) {
			if (f_v) {
				cout << "input set of packing "
						<< Descr->Input[input_idx].input_string << ":" << endl;
			}

			nb_objects_to_test++;

		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_points) {
			if (f_v) {
				cout << "input sets of points from file "
						<< Descr->Input[input_idx].input_string << ":" << endl;
			}



			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			int underlying_set_size = 0;

			if (f_v) {
				cout << "data_input_stream::read_objects "
						"Reading the file " << Descr->Input[input_idx].input_string << endl;
			}
			SoS->init_from_file(
					underlying_set_size,
					Descr->Input[input_idx].input_string, verbose_level);
			if (f_v) {
				cout << "Read the file " << Descr->Input[input_idx].input_string << ", underlying_set_size=" << underlying_set_size << endl;
				cout << "number of sets = " << SoS->nb_sets << endl;
			}

			nb_objects_to_test += SoS->nb_sets;

			FREE_OBJECT(SoS);

#if 0
			nb_obj = Fio.count_number_of_orbits_in_file(
					Descr->Input[input_idx].input_string, 0 /* verbose_level*/);
			if (f_v) {
				cout << "The file " << Descr->Input[input_idx].input_string
					<< " has " << nb_obj << " objects" << endl;
			}
			nb_objects_to_test += nb_obj;
#endif

		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_lines) {
			if (f_v) {
				cout << "input sets of lines from file "
					<< Descr->Input[input_idx].input_string << ":" << endl;
			}
			nb_obj = Fio.count_number_of_orbits_in_file(
					Descr->Input[input_idx].input_string, 0 /* verbose_level*/);
			if (f_v) {
				cout << "The file " << Descr->Input[input_idx].input_string
					<< " has " << nb_obj << " objects" << endl;
			}

			nb_objects_to_test += nb_obj;
		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_packings) {
			if (f_v) {
				cout << "input sets of packings from file "
					<< Descr->Input[input_idx].input_string << ":" << endl;
			}
			nb_obj = Fio.count_number_of_orbits_in_file(
					Descr->Input[input_idx].input_string, 0 /* verbose_level*/);
			if (f_v) {
				cout << "The file " << Descr->Input[input_idx].input_string
					<< " has " << nb_obj << " objects" << endl;
			}

			nb_objects_to_test += nb_obj;
		}
		else if (Descr->Input[input_idx].input_type ==
				t_data_input_stream_file_of_packings_through_spread_table) {
			if (f_v) {
				cout << "input sets of packings from file "
					<< Descr->Input[input_idx].input_string << endl;
				cout << "through spread table "
					<< Descr->Input[input_idx].input_string2 << " :" << endl;
			}
			//nb_obj = Fio.count_number_of_orbits_in_file(
			//		Descr->input_string[input_idx], 0 /* verbose_level*/);

			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			int underlying_set_size = 0;

			if (f_v) {
				cout << "data_input_stream::count_number_of_objects_to_test "
						"Reading the file " << Descr->Input[input_idx].input_string << endl;
			}
			SoS->init_from_file(
					underlying_set_size,
					Descr->Input[input_idx].input_string, verbose_level);
			if (f_v) {
				cout << "Read the file " << Descr->Input[input_idx].input_string << ", underlying_set_size=" << underlying_set_size << endl;
			}

			nb_obj = SoS->nb_sets;

			FREE_OBJECT(SoS);


			if (f_v) {
				cout << "The file " << Descr->Input[input_idx].input_string
					<< " has " << nb_obj << " objects" << endl;
			}

			nb_objects_to_test += nb_obj;
		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_point_set) {
			if (f_v) {
				cout << "input set of points from file "
						<< Descr->Input[input_idx].input_string << ":" << endl;
			}
			nb_obj = 1;
			if (f_v) {
				cout << "The file " << Descr->Input[input_idx].input_string
					<< " has " << nb_obj << " objects" << endl;
			}

			nb_objects_to_test += nb_obj;
		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_designs) {
			if (f_v) {
				cout << "input designs from file "
						<< Descr->Input[input_idx].input_string << ":" << endl;
			}
			{
				set_of_sets *SoS;
				int nck;
				combinatorics::combinatorics_domain Combi;

				nck = Combi.int_n_choose_k(Descr->Input[input_idx].input_data1, Descr->Input[input_idx].input_data3);
				SoS = NEW_OBJECT(set_of_sets);

				cout << "classify_objects_using_nauty Reading the file " << Descr->Input[input_idx].input_string
					<<  " which contains designs on " << Descr->Input[input_idx].input_data1 << " points, nck=" << nck << endl;
				SoS->init_from_file(
						nck /* underlying_set_size */,
						Descr->Input[input_idx].input_string, verbose_level);
				cout << "Read the file " << Descr->Input[input_idx].input_string << endl;
				nb_obj = SoS->nb_sets;
				FREE_OBJECT(SoS);
			}
			if (f_v) {
				cout << "The file " << Descr->Input[input_idx].input_string
					<< " has " << nb_obj << " objects" << endl;
			}

			nb_objects_to_test += nb_obj;
		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_incidence_geometries) {
			if (f_v) {
				cout << "input incidence geometries from file "
						<< Descr->Input[input_idx].input_string << ":" << endl;
			}
			orbiter_kernel_system::file_io Fio;
			int m, n, nb_flags;

			std::vector<std::vector<int> > Geos;

			Fio.read_incidence_file(Geos, m, n, nb_flags, Descr->Input[input_idx].input_string, verbose_level);
			if (f_v) {
				cout << "input incidence geometries from file "
						"the file contains " << Geos.size() << "incidence geometries" << endl;
			}
			nb_objects_to_test += Geos.size();
			if (Descr->Input[input_idx].input_data1 != m) {
				cout << "v does not match" << endl;
				exit(1);
			}
			if (Descr->Input[input_idx].input_data2 != n) {
				cout << "b does not match" << endl;
				exit(1);
			}
			if (Descr->Input[input_idx].input_data3 != nb_flags) {
				cout << "nb_flags does not match" << endl;
				exit(1);
			}
		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_incidence_geometries_by_row_ranks) {
			if (f_v) {
				cout << "input incidence geometries by row ranks from file "
						<< Descr->Input[input_idx].input_string << ":" << endl;
			}
			orbiter_kernel_system::file_io Fio;
			int m, n, r;

			std::vector<std::vector<int> > Geos;

			Fio.read_incidence_by_row_ranks_file(
					Geos, m, n, r, Descr->Input[input_idx].input_string, verbose_level);
			if (f_v) {
				cout << "input incidence geometries from file "
						"the file contains " << Geos.size() << "incidence geometries" << endl;
			}
			nb_objects_to_test += Geos.size();
			if (Descr->Input[input_idx].input_data1 != m) {
				cout << "v does not match" << endl;
				exit(1);
			}
			if (Descr->Input[input_idx].input_data2 != n) {
				cout << "b does not match" << endl;
				exit(1);
			}
			if (Descr->Input[input_idx].input_data3 != r) {
				cout << "r does not match" << endl;
				exit(1);
			}
		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_incidence_geometry) {
			if (f_v) {
				cout << "input incidence geometry directly "
						<< Descr->Input[input_idx].input_string << ":" << endl;
			}
			nb_objects_to_test++;
		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_incidence_geometry_by_row_ranks) {
			if (f_v) {
				cout << "input incidence geometry directly "
						<< Descr->Input[input_idx].input_string << ":" << endl;
			}
			nb_objects_to_test++;
		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_from_parallel_search) {
			if (f_v) {
				cout << "input from parallel search" << endl;
			}

			int nb_cases = Descr->Input[input_idx].input_data1;
			string mask;
			string cases_fname;

			mask.assign(Descr->Input[input_idx].input_string);
			cases_fname.assign(Descr->Input[input_idx].input_string2);

			if (f_v) {
				cout << "input from parallel search, mask=" << mask << endl;
				cout << "input from parallel search, nb_cases=" << nb_cases << endl;
				cout << "input from parallel search, cases_fname=" << cases_fname << endl;
			}

			orbiter_kernel_system::file_io Fio;
			int i;


			for (i = 0; i < nb_cases; i++) {


				char str[1000];
				string fname;


				snprintf(str, sizeof(str), mask.c_str(), i);
				fname.assign(str);

				set_of_sets *SoS;
				int underlying_set_size = 0;
				int nb_sol;

				SoS = NEW_OBJECT(set_of_sets);
				SoS->init_from_orbiter_file(underlying_set_size,
						fname, 0 /*verbose_level*/);
				nb_sol = SoS->nb_sets;
				if (f_v) {
					cout << "objects from file " << i << " / " << nb_cases <<
							" the file contains " << nb_sol << " sets" << endl;
				}
				nb_objects_to_test += nb_sol;
				FREE_OBJECT(SoS);
			}
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

		if (Descr->Input[input_idx].input_type == t_data_input_stream_set_of_points) {

			if (f_v) {
				cout << "data_input_stream::read_objects "
						"input set of points "
					<< Descr->Input[input_idx].input_string << ":" << endl;
			}

			geometry::object_with_canonical_form *OwCF;


			OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

			OwCF->init_point_set_from_string(
					Descr->Input[input_idx].input_string /*set_text*/,
					verbose_level);

			Objects.push_back(OwCF);

		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_point_set) {

			orbiter_kernel_system::file_io Fio;
			long int *the_set;
			int set_size;
			geometry::object_with_canonical_form *OwCF;

			Fio.read_set_from_file(Descr->Input[input_idx].input_string, the_set, set_size, verbose_level);

			OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

			OwCF->init_point_set(the_set, set_size, verbose_level);

			FREE_lint(the_set);

			Objects.push_back(OwCF);

		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_set_of_lines) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

			OwCF->init_line_set_from_string(
					Descr->Input[input_idx].input_string /*set_text*/,
					verbose_level);

			Objects.push_back(OwCF);

		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_set_of_points_and_lines) {

			geometry::object_with_canonical_form *OwCF;


			OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

			OwCF->init_points_and_lines_from_string(
					Descr->Input[input_idx].input_string /*set_text*/,
					Descr->Input[input_idx].input_string2 /*set2_text*/,
					verbose_level);

			Objects.push_back(OwCF);

		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_set_of_packing) {

			geometry::object_with_canonical_form *OwCF;
			int q;

			q = Descr->Input[input_idx].input_data1;


			OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

			OwCF->init_packing_from_string(
					Descr->Input[input_idx].input_string /*packing_text*/,
					q,
					verbose_level);

			Objects.push_back(OwCF);

		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_incidence_geometry) {

			geometry::object_with_canonical_form *OwCF;


			OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

			OwCF->init_incidence_geometry_from_string(
					Descr->Input[input_idx].input_string,
					Descr->Input[input_idx].input_data1 /*v*/,
					Descr->Input[input_idx].input_data2 /*b*/,
					Descr->Input[input_idx].input_data3 /*nb_flags*/,
					verbose_level);

			Objects.push_back(OwCF);

		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_incidence_geometry_by_row_ranks) {

			geometry::object_with_canonical_form *OwCF;


			OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

			OwCF->init_incidence_geometry_from_string_of_row_ranks(
					Descr->Input[input_idx].input_string,
					Descr->Input[input_idx].input_data1 /*v*/,
					Descr->Input[input_idx].input_data2 /*b*/,
					Descr->Input[input_idx].input_data3 /*r*/,
					verbose_level);

			Objects.push_back(OwCF);

		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_incidence_geometries) {
			if (f_v) {
				cout << "input incidence geometries from file "
						<< Descr->Input[input_idx].input_string << ":" << endl;
			}
			orbiter_kernel_system::file_io Fio;
			int m, n, nb_flags;

			std::vector<std::vector<int> > Geos;

			Fio.read_incidence_file(Geos, m, n, nb_flags, Descr->Input[input_idx].input_string, verbose_level);
			if (f_v) {
				cout << "input incidence geometries from file "
						"the file contains " << Geos.size() << "incidence geometries" << endl;
			}
			int h;

			if (Descr->Input[input_idx].input_data1 != m) {
				cout << "v does not match" << endl;
				exit(1);
			}
			if (Descr->Input[input_idx].input_data2 != n) {
				cout << "b does not match" << endl;
				exit(1);
			}
			if (Descr->Input[input_idx].input_data3 != nb_flags) {
				cout << "f does not match" << endl;
				exit(1);
			}

			for (h = 0; h < Geos.size(); h++) {
				geometry::object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

				OwCF->init_incidence_geometry_from_vector(
						Geos[h],
						Descr->Input[input_idx].input_data1 /*v*/,
						Descr->Input[input_idx].input_data2 /*b*/,
						Descr->Input[input_idx].input_data3 /*nb_flags*/,
						verbose_level);

				Objects.push_back(OwCF);

			}
		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_incidence_geometries_by_row_ranks) {
			if (f_v) {
				cout << "input incidence geometries from file "
						<< Descr->Input[input_idx].input_string << " by row ranks:" << endl;
			}
			orbiter_kernel_system::file_io Fio;
			int m, n, r;

			std::vector<std::vector<int> > Geos;

			Fio.read_incidence_by_row_ranks_file(Geos, m, n, r, Descr->Input[input_idx].input_string, verbose_level);
			if (f_v) {
				cout << "input incidence geometries from file "
						"the file contains " << Geos.size() << "incidence geometries" << endl;
			}
			int h;

			if (Descr->Input[input_idx].input_data1 != m) {
				cout << "v does not match" << endl;
				exit(1);
			}
			if (Descr->Input[input_idx].input_data2 != n) {
				cout << "b does not match" << endl;
				exit(1);
			}
			if (Descr->Input[input_idx].input_data3 != r) {
				cout << "r does not match" << endl;
				exit(1);
			}

			for (h = 0; h < Geos.size(); h++) {
				geometry::object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

				OwCF->init_incidence_geometry_from_vector(
						Geos[h],
						Descr->Input[input_idx].input_data1 /*v*/,
						Descr->Input[input_idx].input_data2 /*b*/,
						Geos[h].size() /*nb_flags*/,
						verbose_level);

				Objects.push_back(OwCF);

			}
		}
		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_designs) {

			int v, b, k, design_sz;

			v = Descr->Input[input_idx].input_data1;
			b = Descr->Input[input_idx].input_data2;
			k = Descr->Input[input_idx].input_data3;
			design_sz = Descr->Input[input_idx].input_data4;

			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			int underlying_set_size = 0;

			if (f_v) {
				cout << "data_input_stream::read_objects "
						"Reading the file " << Descr->Input[input_idx].input_string << endl;
			}
			SoS->init_from_file(
					underlying_set_size,
					Descr->Input[input_idx].input_string, verbose_level);
			if (f_v) {
				cout << "Read the file " << Descr->Input[input_idx].input_string << ", underlying_set_size=" << underlying_set_size << endl;
			}

			int h;

			for (h = 0; h < SoS->nb_sets; h++) {

				if ((h % 1000) == 0) {
					cout << "data_input_stream::read_objects " << h << " / " << SoS->nb_sets << endl;
				}

				geometry::object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

				OwCF->init_large_set(
						SoS->Sets[h], SoS->Set_size[h], v, b, k, design_sz,
						0 /*verbose_level*/);

				Objects.push_back(OwCF);
			}

			FREE_OBJECT(SoS);
		}

		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_packings_through_spread_table) {

			int q;

			q = Descr->Input[input_idx].input_data1;

			orbiter_kernel_system::file_io Fio;
			long int *Spread_table;
			int nb_spreads;
			int spread_size;

			if (f_v) {
				cout << "data_input_stream::read_objects "
						"Reading spread table from file "
					<< Descr->Input[input_idx].input_string2 << endl;
			}

			Fio.lint_matrix_read_csv(Descr->Input[input_idx].input_string2,
					Spread_table, nb_spreads, spread_size,
					0 /* verbose_level */);

			if (f_v) {
				cout << "Reading spread table from file "
						<< Descr->Input[input_idx].input_string2 << " done" << endl;
				cout << "The spread table contains " << nb_spreads
						<< " spreads" << endl;
			}


			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			int underlying_set_size = 0;

			if (f_v) {
				cout << "data_input_stream::read_objects "
						"Reading the file " << Descr->Input[input_idx].input_string << endl;
			}
			SoS->init_from_file(
					underlying_set_size,
					Descr->Input[input_idx].input_string, verbose_level);
			if (f_v) {
				cout << "Read the file " << Descr->Input[input_idx].input_string << ", underlying_set_size=" << underlying_set_size << endl;
			}

			int h;

			for (h = 0; h < SoS->nb_sets; h++) {


				geometry::object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

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


		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_packings) {

			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			int underlying_set_size = 0;

			if (f_v) {
				cout << "data_input_stream::read_objects "
						"Reading the file " << Descr->Input[input_idx].input_string << endl;
			}
			SoS->init_from_file(
					underlying_set_size,
					Descr->Input[input_idx].input_string, verbose_level);
			if (f_v) {
				cout << "Read the file " << Descr->Input[input_idx].input_string << ", underlying_set_size=" << underlying_set_size << endl;
			}

			if (f_v) {
				cout << "set of sets:" << endl;
				SoS->print_table();
			}

			int h;

			for (h = 0; h < SoS->nb_sets; h++) {


				geometry::object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

				if (f_v) {
					cout << "before OwCF->init_packing_from_set " << h << " / " << SoS->nb_sets << endl;
				}

				OwCF->init_packing_from_set(
						SoS->Sets[h], SoS->Set_size[h], verbose_level);


				Objects.push_back(OwCF);
			}

			FREE_OBJECT(SoS);

		}


		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_lines) {

			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			int underlying_set_size = 0;

			if (f_v) {
				cout << "data_input_stream::read_objects "
						"Reading the file " << Descr->Input[input_idx].input_string << endl;
			}
			SoS->init_from_file(
					underlying_set_size,
					Descr->Input[input_idx].input_string, verbose_level);
			if (f_v) {
				cout << "Read the file " << Descr->Input[input_idx].input_string << ", underlying_set_size=" << underlying_set_size << endl;
			}


			int h;

			for (h = 0; h < SoS->nb_sets; h++) {


				geometry::object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

				OwCF->init_line_set(
						SoS->Sets[h], SoS->Set_size[h], verbose_level);


				Objects.push_back(OwCF);
			}

			FREE_OBJECT(SoS);

		}


		else if (Descr->Input[input_idx].input_type == t_data_input_stream_file_of_points) {

			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			int underlying_set_size = 0;

			if (f_v) {
				cout << "data_input_stream::read_objects "
						"Reading the file " << Descr->Input[input_idx].input_string << endl;
			}
			SoS->init_from_file(
					underlying_set_size,
					Descr->Input[input_idx].input_string, verbose_level);
			if (f_v) {
				cout << "Read the file " << Descr->Input[input_idx].input_string << ", underlying_set_size=" << underlying_set_size << endl;
				cout << "number of sets = " << SoS->nb_sets << endl;
			}


			int h;

			for (h = 0; h < SoS->nb_sets; h++) {


				geometry::object_with_canonical_form *OwCF;


				OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

				OwCF->init_point_set(
						SoS->Sets[h], SoS->Set_size[h], verbose_level);


				Objects.push_back(OwCF);
			}

			FREE_OBJECT(SoS);

		}

		else if (Descr->Input[input_idx].input_type == t_data_input_stream_from_parallel_search) {
			if (f_v) {
				cout << "input from parallel search" << endl;
			}

			int nb_cases = Descr->Input[input_idx].input_data1;
			string mask;
			string cases_fname;

			mask.assign(Descr->Input[input_idx].input_string);
			cases_fname.assign(Descr->Input[input_idx].input_string2);

			if (f_v) {
				cout << "input from parallel search, mask=" << mask << endl;
				cout << "input from parallel search, nb_cases=" << nb_cases << endl;
				cout << "input from parallel search, cases_fname=" << cases_fname << endl;
			}

			orbiter_kernel_system::file_io Fio;
			int c;

			set_of_sets *Reps;
			string col_label;
			int prefix_sz;

			col_label.assign("REP");


			Fio.read_column_and_parse(cases_fname,
					col_label,
					Reps, verbose_level);
			if (!Reps->has_constant_size_property()) {
				cout << "data_input_stream::read_objects the sets have different sizes" << endl;
				exit(1);
			}
			prefix_sz = Reps->Set_size[0];

			for (c = 0; c < nb_cases; c++) {


				if (f_v) {
					cout << "case " << c << " / " << nb_cases << " prefix=";
					Lint_vec_print(cout, Reps->Sets[c], prefix_sz);
				}

				char str[1000];
				string fname;


				snprintf(str, sizeof(str), mask.c_str(), c);
				fname.assign(str);

				set_of_sets *SoS;
				int underlying_set_size = 0;
				int nb_sol;
				int sol_width;

				SoS = NEW_OBJECT(set_of_sets);
				SoS->init_from_orbiter_file(underlying_set_size,
						fname, 0 /*verbose_level*/);
				nb_sol = SoS->nb_sets;
				if (f_v) {
					cout << "objects from file " << c << " / " << nb_cases <<
							" the file contains " << nb_sol << " sets" << endl;
				}
				if (nb_sol) {
					if (!SoS->has_constant_size_property()) {
						cout << "data_input_stream::read_objects the sets have different sizes" << endl;
						exit(1);
					}
					sol_width = SoS->Set_size[0];
					long int *Sol_idx;
					int i, j;
					long int *set;

					Sol_idx = NEW_lint(nb_sol * sol_width);
					set = NEW_lint(prefix_sz + sol_width);
					for (i = 0; i < nb_sol; i++) {
						for (j = 0; j < sol_width; j++) {
							Sol_idx[i * sol_width + j] = SoS->Sets[i][j];
						}
					}

					for (i = 0; i < nb_sol; i++) {
						geometry::object_with_canonical_form *OwCF;


						Lint_vec_copy(Reps->Sets[c], set, prefix_sz);
						Lint_vec_copy(Sol_idx + i * sol_width, set + prefix_sz, sol_width);

						OwCF = NEW_OBJECT(geometry::object_with_canonical_form);

						OwCF->init_point_set(
								set, prefix_sz + sol_width,
								0 /*verbose_level*/);

						Objects.push_back(OwCF);
					}
					FREE_lint(Sol_idx);
					FREE_lint(set);
				}
				//nb_objects_to_test += nb_sol;
				FREE_OBJECT(SoS);
			}
		}

		else {
			cout << "data_input_stream::read_objects "
					"unknown input type " << Descr->Input[input_idx].input_type << endl;
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

}}}

