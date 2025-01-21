/*
 * data_input_stream_description_element.cpp
 *
 *  Created on: Dec 31, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace canonical_form_classification {


data_input_stream_description_element::data_input_stream_description_element()
{
	Record_birth();
	input_type = t_data_input_stream_unknown;
	//std::string input_string;
	//std::string input_string2;
	input_data1 = 0;
	input_data2 = 0;
	input_data3 = 0;
	input_data4 = 0;

	//GOC = NULL;

}

data_input_stream_description_element::~data_input_stream_description_element()
{
	Record_death();
}



void data_input_stream_description_element::print()
{
	if (input_type == t_data_input_stream_unknown) {
		cout << "unknown type " << endl;
	}
	else if (input_type == t_data_input_stream_set_of_points) {
		cout << "-set_of_points " << input_string << endl;
	}
	else if (input_type == t_data_input_stream_set_of_lines) {
		cout << "-set_of_lines " << input_string << endl;
	}
	else if (input_type == t_data_input_stream_set_of_points_and_lines) {
		cout << "-set_of_points_and_lines "
				<< input_string
				<< " " << input_string2 << endl;
	}
	else if (input_type == t_data_input_stream_set_of_packing) {
		cout << "-set_of_packing " << input_string << " " << input_string2 << endl;
	}
	else if (input_type == t_data_input_stream_file_of_points) {
		cout << "-file_of_points " << input_string << endl;
	}
	else if (input_type == t_data_input_stream_file_of_points_csv) {
		cout << "-file_of_points_csv " << input_string << endl;
	}
	else if (input_type == t_data_input_stream_file_of_lines) {
		cout << "-file_of_lines " << input_string << endl;
	}
	else if (input_type == t_data_input_stream_file_of_packings) {
		cout << "-file_of_packings " << input_string << " " << input_string2 << endl;
	}
	else if (input_type == t_data_input_stream_file_of_packings_through_spread_table) {
		cout << "-file_of_packings_through_spread_table "
			<< input_string
			<< " " << input_string2
			<< " " << input_data1
			<< endl;
	}
	else if (input_type == t_data_input_stream_file_of_designs_through_block_orbits) {
		cout << "-file_of_designs_through_block_orbits "
			<< input_string
			<< " " << input_string2
			<< " " << input_data1
			<< " " << input_data2
			<< endl;
	}
	else if (input_type == t_data_input_stream_file_of_designs_through_blocks) {
		cout << "-file_of_designs_through_blocks "
				<< input_string
				<< input_string2
			<< " " << input_data1
			<< " " << input_data2
			<< " " << input_data3
			<< endl;
	}

	else if (input_type == t_data_input_stream_file_of_point_set) {
		cout << "-file_of_point_set " << input_string << " " << input_string2 << endl;
	}
	else if (input_type == t_data_input_stream_file_of_designs) {
		cout << "-file_of_designs " << input_string
				<< " " << input_data1
				<< " " << input_data2
				<< " " << input_data3
				<< " " << input_data4
				<< endl;
	}
	else if (input_type == t_data_input_stream_file_of_incidence_geometries) {
		cout << "-file_of_incidence_geometries " << input_string
				<< " " << input_data1
				<< " " << input_data2
				<< " " << input_data3
				<< endl;
	}
	else if (input_type == t_data_input_stream_file_of_incidence_geometries_by_row_ranks) {
		cout << "-file_of_incidence_geometries_by_row_ranks " << input_string
				<< " " << input_data1
				<< " " << input_data2
				<< " " << input_data3
				<< endl;
	}
	else if (input_type == t_data_input_stream_incidence_geometry) {
		cout << "-incidence_geometry " << input_string
			<< " " << input_data1
			<< " " << input_data2
			<< " " << input_data3
			<< endl;
	}
	else if (input_type == t_data_input_stream_incidence_geometry_by_row_ranks) {
		cout << "-incidence_geometry_by_row_ranks " << input_string
			<< " " << input_data1
			<< " " << input_data2
			<< " " << input_data3
			<< endl;
	}
	else if (input_type == t_data_input_stream_from_parallel_search) {
		cout << "-from_parallel_search"
				<< " " << input_string
				<< " " << input_string2
				<< " " << input_data1
			<< endl;
	}
	else if (input_type == t_data_input_stream_orbiter_file) {
		cout << "-orbiter_file"
				<< " " << input_string
			<< endl;
	}
	else if (input_type == t_data_input_stream_csv_file) {
		cout << "-csv_file"
				<< " " << input_string
				<< " " << input_string2
			<< endl;
	}
	else if (input_type == t_data_input_stream_graph_by_adjacency_matrix) {
		cout << "-graph_by_adjacency_matrix"
				<< " " << input_string
				<< " " << input_data1
			<< endl;
	}
	else if (input_type == t_data_input_stream_graph_object) {
		cout << "-graph_object"
				<< " " << input_string
			<< endl;
	}
	else if (input_type == t_data_input_stream_graph_by_adjacency_matrix_from_file) {
		cout << "-graph_by_adjacency_matrix_from_file"
				<< " " << input_string
				<< " " << input_string2
				<< " " << input_data1
			<< endl;
	}
	else if (input_type == t_data_input_stream_multi_matrix) {
		cout << "-multi_matrix"
				<< " " << input_string
				<< " " << input_string2
			<< endl;
	}
	else if (input_type == t_data_input_stream_geometric_object) {
		cout << "-geometric_object"
				<< " " << input_string
			<< endl;
	}
	else {
		cout << "data_input_stream_description_element::print unknown type" << endl;
		exit(1);
	}
}


void data_input_stream_description_element::init_set_of_points(
		std::string &a)
{
	input_type = t_data_input_stream_set_of_points;

	input_string.assign(a);

}

void data_input_stream_description_element::init_set_of_lines(
		std::string &a)
{
	input_type = t_data_input_stream_set_of_lines;

	input_string.assign(a);

}

void data_input_stream_description_element::init_set_of_points_and_lines(
		std::string &a, std::string &b)
{
	input_type = t_data_input_stream_set_of_points_and_lines;

	input_string.assign(a);
	input_string2.assign(b);

}

void data_input_stream_description_element::init_packing(
		std::string &a, int q)
{
	input_type = t_data_input_stream_set_of_packing;

	input_string.assign(a);
	input_data1 = q;

}


void data_input_stream_description_element::init_file_of_points(
		std::string &a)
{
	input_type = t_data_input_stream_file_of_points;

	input_string.assign(a);

}

void data_input_stream_description_element::init_file_of_points_csv(
		std::string &a, std::string &b)
{
	input_type = t_data_input_stream_file_of_points_csv;

	input_string.assign(a);
	input_string2.assign(b);

}

void data_input_stream_description_element::init_file_of_lines(
		std::string &a)
{
	input_type = t_data_input_stream_file_of_lines;

	input_string.assign(a);

}

void data_input_stream_description_element::init_file_of_packings(std::string &a)
{
	input_type = t_data_input_stream_file_of_packings;

	input_string.assign(a);

}

void data_input_stream_description_element::init_file_of_packings_through_spread_table(
		std::string &a, std::string &b, int q)
{
	input_type = t_data_input_stream_file_of_packings_through_spread_table;

	input_string.assign(a);
	input_string2.assign(b);
	input_data1 = q;
}

void data_input_stream_description_element::init_file_of_designs_through_block_orbits(
		std::string &a, std::string &b, int v, int k)
{
	input_type = t_data_input_stream_file_of_designs_through_block_orbits;

	input_string.assign(a);
	input_string2.assign(b);
	input_data1 = v;
	input_data2 = k;
}

void data_input_stream_description_element::init_file_of_designs_through_blocks(
		std::string &fname_blocks,
		std::string &col_label, int v, int b, int k)
{
	input_type = t_data_input_stream_file_of_designs_through_blocks;

	input_string.assign(fname_blocks);
	input_string2.assign(col_label);
	input_data1 = v;
	input_data2 = b;
	input_data3 = k;
}




void data_input_stream_description_element::init_file_of_point_set(
		std::string &a)
{
	input_type = t_data_input_stream_file_of_point_set;

	input_string.assign(a);

}

void data_input_stream_description_element::init_file_of_designs(
		std::string &a,
			int N_points, int b, int k, int partition_class_size)
{
	input_type = t_data_input_stream_file_of_designs;

	input_string.assign(a);
	input_data1 = N_points;
	input_data2 = b;
	input_data3 = k;
	input_data4 = partition_class_size;

}

void data_input_stream_description_element::init_file_of_incidence_geometries(
		std::string &a,
			int v, int b, int f)
{
	input_type = t_data_input_stream_file_of_incidence_geometries;

	input_string.assign(a);
	input_data1 = v;
	input_data2 = b;
	input_data3 = f;

}

void data_input_stream_description_element::init_file_of_incidence_geometries_by_row_ranks(
		std::string &a,
			int v, int b, int r)
{
	input_type = t_data_input_stream_file_of_incidence_geometries_by_row_ranks;

	input_string.assign(a);
	input_data1 = v;
	input_data2 = b;
	input_data3 = r;

}

void data_input_stream_description_element::init_incidence_geometry(
		std::string &a,
			int v, int b, int f)
{
	input_type = t_data_input_stream_incidence_geometry;

	input_string.assign(a);
	input_data1 = v;
	input_data2 = b;
	input_data3 = f;

}

void data_input_stream_description_element::init_incidence_geometry_by_row_ranks(
		std::string &a,
			int v, int b, int r)
{
	input_type = t_data_input_stream_incidence_geometry_by_row_ranks;

	input_string.assign(a);
	input_data1 = v;
	input_data2 = b;
	input_data3 = r;

}


void data_input_stream_description_element::init_from_parallel_search(
		std::string &fname_mask,
		int nb_cases, std::string &cases_fname)
{
	input_type = t_data_input_stream_from_parallel_search;

	input_string.assign(fname_mask);
	input_string2.assign(cases_fname);
	input_data1 = nb_cases;

}

void data_input_stream_description_element::init_orbiter_file(
		std::string &fname)
{
	input_type = t_data_input_stream_orbiter_file;

	input_string.assign(fname);

}

void data_input_stream_description_element::init_csv_file(
		std::string &fname, std::string &column_heading)
{
	input_type = t_data_input_stream_csv_file;

	input_string.assign(fname);
	input_string2.assign(column_heading);

}

void data_input_stream_description_element::init_graph_by_adjacency_matrix(
		std::string &adjacency_matrix,
			int N)
{
	input_type = t_data_input_stream_graph_by_adjacency_matrix;

	input_string.assign(adjacency_matrix);
	input_data1 = N;
	input_data2 = 0;
	input_data3 = 0;

}

void data_input_stream_description_element::init_graph_object(
		std::string &object_label)
{
	input_type = t_data_input_stream_graph_object;

	input_string.assign(object_label);
	input_data1 = 0;
	input_data2 = 0;
	input_data3 = 0;

}

void data_input_stream_description_element::init_graph_by_adjacency_matrix_from_file(
		std::string &fname,
		std::string &col_label,
			int N)
{
	input_type = t_data_input_stream_graph_by_adjacency_matrix_from_file;

	input_string.assign(fname);
	input_string2.assign(col_label);
	input_data1 = N;
	input_data2 = 0;
	input_data3 = 0;

}

void data_input_stream_description_element::init_multi_matrix(
		std::string &a, std::string &b)
{
	input_type = t_data_input_stream_multi_matrix;

	input_string.assign(a);
	input_string2.assign(b);
}

void data_input_stream_description_element::init_geometric_object(
		std::string &label)
{
	input_type = t_data_input_stream_geometric_object;

	input_string.assign(label);
}

}}}}


