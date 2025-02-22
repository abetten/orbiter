/*
 * orthogonal_space_activity_description.cpp
 *
 *  Created on: Jan 12, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


orthogonal_space_activity_description::orthogonal_space_activity_description()
{
	Record_birth();

	f_cheat_sheet_orthogonal = false;
	//std::string cheat_sheet_orthogonal_draw_options_label;

	f_print_points = false;
	//std::string print_points_label;

	f_print_lines = false;
	//std::string print_lines_label;

	f_unrank_line_through_two_points = false;
	//std::string unrank_line_through_two_points_p1;
	//std::string unrank_line_through_two_points_p2;

	f_lines_on_point = false;
	lines_on_point_rank = 0;

	f_perp = false;
	//std::string perp_text;

	f_set_stabilizer = false;
	set_stabilizer_intermediate_set_size = 0;
	//std::string set_stabilizer_fname_mask;
	set_stabilizer_nb = 0;
	//std::string set_stabilizer_column_label;
	//std::string set_stabilizer_fname_out;


	f_export_point_line_incidence_matrix = false;

	f_intersect_with_subspace = false;
	//std::string intersect_with_subspace_label;

	f_table_of_blt_sets = false;

	f_create_orthogonal_reflection = false;
	//std::string create_orthogonal_reflection_points;

	f_create_orthogonal_reflection_6_and_4 = false;
	//std::string create_orthogonal_reflection_6_and_4_points;
	//std::string create_orthogonal_reflection_6_and_4_A4;

}

orthogonal_space_activity_description::~orthogonal_space_activity_description()
{
	Record_death();

}


int orthogonal_space_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "orthogonal_space_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-cheat_sheet_orthogonal") == 0) {
			f_cheat_sheet_orthogonal = true;
			cheat_sheet_orthogonal_draw_options_label.assign(argv[++i]);
			if (f_v) {
				cout << "-cheat_sheet_orthogonal " << cheat_sheet_orthogonal_draw_options_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_points") == 0) {
			f_print_points = true;
			print_points_label.assign(argv[++i]);
			if (f_v) {
				cout << "-print_points " << print_points_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_lines") == 0) {
			f_print_lines = true;
			print_lines_label.assign(argv[++i]);
			if (f_v) {
				cout << "-print_lines " << print_lines_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-unrank_line_through_two_points") == 0) {
			f_unrank_line_through_two_points = true;
			unrank_line_through_two_points_p1.assign(argv[++i]);
			unrank_line_through_two_points_p2.assign(argv[++i]);
			if (f_v) {
				cout << "-unrank_line_through_two_points " << unrank_line_through_two_points_p1
					<< " " << unrank_line_through_two_points_p2 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-lines_on_point") == 0) {
			f_lines_on_point = true;
			lines_on_point_rank = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-lines_on_point " << lines_on_point_rank << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-perp") == 0) {
			f_perp = true;
			perp_text.assign(argv[++i]);
			if (f_v) {
				cout << "-perp " << perp_text << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-set_stabilizer") == 0) {
			f_set_stabilizer = true;
			set_stabilizer_intermediate_set_size = ST.strtoi(argv[++i]);
			set_stabilizer_fname_mask.assign(argv[++i]);
			set_stabilizer_nb = ST.strtoi(argv[++i]);
			set_stabilizer_column_label.assign(argv[++i]);
			set_stabilizer_fname_out.assign(argv[++i]);
			if (f_v) {
				cout << "-set_stabilizer "
						<< set_stabilizer_intermediate_set_size << " "
						<< set_stabilizer_fname_mask << " "
						<< set_stabilizer_nb << " "
						<< set_stabilizer_column_label << " "
						<< set_stabilizer_fname_out << " "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_point_line_incidence_matrix") == 0) {
			f_export_point_line_incidence_matrix = true;
			if (f_v) {
				cout << "-export_point_line_incidence_matrix " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-intersect_with_subspace") == 0) {
			f_intersect_with_subspace = true;
			intersect_with_subspace_label.assign(argv[++i]);
			if (f_v) {
				cout << "-intersect_with_subspace " << intersect_with_subspace_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-table_of_blt_sets") == 0) {
			f_table_of_blt_sets = true;
			if (f_v) {
				cout << "-table_of_blt_sets " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-create_orthogonal_reflection") == 0) {
			f_create_orthogonal_reflection = true;
			create_orthogonal_reflection_points.assign(argv[++i]);
			if (f_v) {
				cout << "-create_orthogonal_reflection " << create_orthogonal_reflection_points << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-create_orthogonal_reflection_6_and_4") == 0) {
			f_create_orthogonal_reflection_6_and_4 = true;
			create_orthogonal_reflection_6_and_4_points.assign(argv[++i]);
			create_orthogonal_reflection_6_and_4_A4.assign(argv[++i]);
			if (f_v) {
				cout << "-f_create_orthogonal_reflection_6_and_4 "
						<< create_orthogonal_reflection_6_and_4_points
						<< " " << create_orthogonal_reflection_6_and_4_A4 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "orthogonal_space_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		if (f_v) {
			cout << "orthogonal_space_activity_description::read_arguments looping, i=" << i << endl;
		}
	} // next i

	if (f_v) {
		cout << "orthogonal_space_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}


void orthogonal_space_activity_description::print()
{
	if (f_cheat_sheet_orthogonal) {
		cout << "-cheat_sheet_orthogonal " << cheat_sheet_orthogonal_draw_options_label << endl;
	}
	if (f_print_points) {
		cout << "-print_points " << print_points_label << endl;
	}
	if (f_print_lines) {
		cout << "-print_lines " << print_lines_label << endl;
	}
	if (f_unrank_line_through_two_points) {
		cout << "-unrank_line_through_two_points " << unrank_line_through_two_points_p1
				<< " " << unrank_line_through_two_points_p2 << endl;
	}
	if (f_lines_on_point) {
		cout << "-lines_on_point " << lines_on_point_rank << endl;
	}

	if (f_perp) {
		cout << "-perp " << perp_text << endl;
	}

	if (f_set_stabilizer) {
		cout << "-set_stabilizer "
				<< set_stabilizer_intermediate_set_size << " "
				<< set_stabilizer_fname_mask << " "
				<< set_stabilizer_nb << " "
				<< set_stabilizer_column_label << " "
				<< set_stabilizer_fname_out << " "
				<< endl;
	}
	if (f_export_point_line_incidence_matrix) {
		cout << "-export_point_line_incidence_matrix " << endl;
	}
	if (f_intersect_with_subspace) {
		cout << "-intersect_with_subspace " << intersect_with_subspace_label << endl;
	}
	if (f_table_of_blt_sets) {
		cout << "-table_of_blt_sets " << endl;
	}
	if (f_create_orthogonal_reflection) {
		cout << "-create_orthogonal_reflection " << create_orthogonal_reflection_points << endl;
	}
	if (f_create_orthogonal_reflection_6_and_4) {
		cout << "-f_create_orthogonal_reflection_6_and_4 "
				<< create_orthogonal_reflection_6_and_4_points
				<< " " << create_orthogonal_reflection_6_and_4_A4 << endl;
	}
}



}}}

