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

	//f_input = FALSE;
	//Data = NULL;

	f_create_BLT_set = FALSE;
	BLT_Set_create_description = NULL;

#if 0
	f_BLT_set_starter = FALSE;
	BLT_set_starter_size = 0;
	BLT_set_starter_control = NULL;

	f_BLT_set_graphs = FALSE;
	BLT_set_graphs_starter_size = 0;
	BLT_set_graphs_r = 0;
	BLT_set_graphs_m = 0;

	f_fname_base_out = FALSE;
	//fname_base_out;
#endif

	f_cheat_sheet_orthogonal = FALSE;

	f_print_points = FALSE;
	//std::string print_points_label;

	f_print_lines = FALSE;
	//std::string print_lines_label;

	f_unrank_line_through_two_points = FALSE;
	//std::string unrank_line_through_two_points_p1;
	//std::string unrank_line_through_two_points_p2;

	f_lines_on_point = FALSE;
	lines_on_point_rank = 0;

	f_perp = FALSE;
	//std::string perp_text;

	f_set_stabilizer = FALSE;
	set_stabilizer_intermediate_set_size = 0;
	//std::string set_stabilizer_fname_mask;
	set_stabilizer_nb = 0;
	//std::string set_stabilizer_column_label;
	//std::string set_stabilizer_fname_out;


	f_export_point_line_incidence_matrix = FALSE;

	f_intersect_with_subspace = FALSE;
	//std::string intersect_with_subspace_label;


}

orthogonal_space_activity_description::~orthogonal_space_activity_description()
{

}


int orthogonal_space_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "orthogonal_space_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

#if 0
		if (ST.stringcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data = NEW_OBJECT(data_structures::data_input_stream_description);
			if (f_v) {
				cout << "-input" << endl;
			}
			i += Data->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
			if (f_v) {
				cout << "orthogonal_space_activity_description::read_arguments finished reading -input" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
#endif

		if (ST.stringcmp(argv[i], "-create_BLT_set") == 0) {
			f_create_BLT_set = TRUE;
			BLT_Set_create_description = NEW_OBJECT(BLT_set_create_description);
			if (f_v) {
				cout << "-create_BLT_set" << endl;
			}
			i += BLT_Set_create_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
			if (f_v) {
				cout << "orthogonal_space_activity_description::read_arguments finished reading -create_BLT_set" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}

#if 0
		else if (ST.stringcmp(argv[i], "-BLT_set_starter") == 0) {
			f_BLT_set_starter = TRUE;
			BLT_set_starter_size = ST.strtoi(argv[++i]);

			BLT_set_starter_control = NEW_OBJECT(poset_classification::poset_classification_control);

			i += BLT_set_starter_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}


				cout << "-BLT_set_starter " << BLT_set_starter_size << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-BLT_set_graphs") == 0) {
			f_BLT_set_graphs = TRUE;
			BLT_set_graphs_starter_size = ST.strtoi(argv[++i]);
			BLT_set_graphs_r = ST.strtoi(argv[++i]);
			BLT_set_graphs_m = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-BLT_set_graphs " << BLT_set_graphs_starter_size << " " << BLT_set_graphs_r << " " << BLT_set_graphs_m << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-fname_base_out") == 0) {
			f_fname_base_out = TRUE;
			fname_base_out.assign(argv[++i]);
			if (f_v) {
				cout << "-fname_base_out " << fname_base_out << endl;
			}
		}
#endif

		else if (ST.stringcmp(argv[i], "-cheat_sheet_orthogonal") == 0) {
			f_cheat_sheet_orthogonal = TRUE;
			if (f_v) {
				cout << "-cheat_sheet_orthogonal "<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_points") == 0) {
			f_print_points = TRUE;
			print_points_label.assign(argv[++i]);
			if (f_v) {
				cout << "-print_points " << print_points_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_lines") == 0) {
			f_print_lines = TRUE;
			print_lines_label.assign(argv[++i]);
			if (f_v) {
				cout << "-print_lines " << print_lines_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-unrank_line_through_two_points") == 0) {
			f_unrank_line_through_two_points = TRUE;
			unrank_line_through_two_points_p1.assign(argv[++i]);
			unrank_line_through_two_points_p2.assign(argv[++i]);
			if (f_v) {
				cout << "-unrank_line_through_two_points " << unrank_line_through_two_points_p1
					<< " " << unrank_line_through_two_points_p2 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-lines_on_point") == 0) {
			f_lines_on_point = TRUE;
			lines_on_point_rank = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-lines_on_point " << lines_on_point_rank << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-perp") == 0) {
			f_perp = TRUE;
			perp_text.assign(argv[++i]);
			if (f_v) {
				cout << "-perp " << perp_text << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-set_stabilizer") == 0) {
			f_set_stabilizer = TRUE;
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
			f_export_point_line_incidence_matrix = TRUE;
			if (f_v) {
				cout << "-export_point_line_incidence_matrix " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-intersect_with_subspace") == 0) {
			f_intersect_with_subspace = TRUE;
			intersect_with_subspace_label.assign(argv[++i]);
			if (f_v) {
				cout << "-intersect_with_subspace " << intersect_with_subspace_label << endl;
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
#if 0
	if (f_input) {
		cout << "-input" << endl;
	}
#endif
	if (f_create_BLT_set) {
		cout << "-create_BLT_set ";
		BLT_Set_create_description->print();
	}
#if 0
	if (f_BLT_set_starter) {
		cout << "-BLT_set_starter " << BLT_set_starter_size << endl;
	}
	if (f_BLT_set_graphs) {
		cout << "-BLT_set_graphs " << BLT_set_graphs_starter_size << " " << BLT_set_graphs_r << " " << BLT_set_graphs_m << endl;
	}
	if (f_fname_base_out) {
		cout << "-fname_base_out " << fname_base_out << endl;
	}
#endif
	if (f_cheat_sheet_orthogonal) {
		f_cheat_sheet_orthogonal = TRUE;
		cout << "-cheat_sheet_orthogonal "<< endl;
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
}



}}}

