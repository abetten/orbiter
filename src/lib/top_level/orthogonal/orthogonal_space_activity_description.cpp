/*
 * orthogonal_space_activity_description.cpp
 *
 *  Created on: Jan 12, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


orthogonal_space_activity_description::orthogonal_space_activity_description()
{

	f_input = FALSE;
	Data = NULL;

	f_create_BLT_set = FALSE;
	BLT_Set_create_description = NULL;

	f_BLT_set_starter = FALSE;
	BLT_set_starter_size = 0;
	BLT_set_starter_control = NULL;

	f_BLT_set_graphs = FALSE;
	BLT_set_graphs_starter_size = 0;
	BLT_set_graphs_r = 0;
	BLT_set_graphs_m = 0;

	f_fname_base_out = FALSE;
	//fname_base_out;

	f_cheat_sheet_orthogonal = FALSE;

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

	if (f_v) {
		cout << "orthogonal_space_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data = NEW_OBJECT(data_input_stream);
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
		else if (stringcmp(argv[i], "-create_BLT_set") == 0) {
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
		else if (stringcmp(argv[i], "-BLT_set_starter") == 0) {
			f_BLT_set_starter = TRUE;
			BLT_set_starter_size = strtoi(argv[++i]);

			BLT_set_starter_control = NEW_OBJECT(poset_classification_control);

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
		else if (stringcmp(argv[i], "-BLT_set_graphs") == 0) {
			f_BLT_set_graphs = TRUE;
			BLT_set_graphs_starter_size = strtoi(argv[++i]);
			BLT_set_graphs_r = strtoi(argv[++i]);
			BLT_set_graphs_m = strtoi(argv[++i]);
			if (f_v) {
				cout << "-BLT_set_graphs " << BLT_set_graphs_starter_size << " " << BLT_set_graphs_r << " " << BLT_set_graphs_m << endl;
			}
		}
		else if (stringcmp(argv[i], "-fname_base_out") == 0) {
			f_fname_base_out = TRUE;
			fname_base_out.assign(argv[++i]);
			if (f_v) {
				cout << "-fname_base_out " << fname_base_out << endl;
			}
		}
		else if (stringcmp(argv[i], "-cheat_sheet_orthogonal") == 0) {
			f_cheat_sheet_orthogonal = TRUE;
			if (f_v) {
				cout << "-cheat_sheet_orthogonal "<< endl;
			}
		}
		else if (stringcmp(argv[i], "-unrank_line_through_two_points") == 0) {
			f_unrank_line_through_two_points = TRUE;
			unrank_line_through_two_points_p1.assign(argv[++i]);
			unrank_line_through_two_points_p2.assign(argv[++i]);
			if (f_v) {
				cout << "-unrank_line_through_two_points " << unrank_line_through_two_points_p1
					<< " " << unrank_line_through_two_points_p2 << endl;
			}
		}
		else if (stringcmp(argv[i], "-lines_on_point") == 0) {
			f_lines_on_point = TRUE;
			lines_on_point_rank = strtoi(argv[++i]);
			if (f_v) {
				cout << "-lines_on_point " << lines_on_point_rank << endl;
			}
		}

		else if (stringcmp(argv[i], "-perp") == 0) {
			f_perp = TRUE;
			perp_text.assign(argv[++i]);
			if (f_v) {
				cout << "-perp " << perp_text << endl;
			}
		}

		else if (stringcmp(argv[i], "-set_stabilizer") == 0) {
			f_set_stabilizer = TRUE;
			set_stabilizer_intermediate_set_size = strtoi(argv[++i]);
			set_stabilizer_fname_mask.assign(argv[++i]);
			set_stabilizer_nb = strtoi(argv[++i]);
			set_stabilizer_column_label.assign(argv[++i]);
			if (f_v) {
				cout << "-set_stabilizer "
						<< set_stabilizer_intermediate_set_size << " "
						<< set_stabilizer_fname_mask << " "
						<< set_stabilizer_nb << " "
						<< set_stabilizer_column_label << " "
						<< endl;
			}
		}

		else if (stringcmp(argv[i], "-end") == 0) {
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
	if (f_input) {
		cout << "-input" << endl;
	}
	if (f_create_BLT_set) {
		cout << "-create_BLT_set ";
		BLT_Set_create_description->print();
	}
	if (f_BLT_set_starter) {
		cout << "-BLT_set_starter " << BLT_set_starter_size << endl;
	}
	if (f_BLT_set_graphs) {
		cout << "-BLT_set_graphs " << BLT_set_graphs_starter_size << " " << BLT_set_graphs_r << " " << BLT_set_graphs_m << endl;
	}
	if (f_fname_base_out) {
		cout << "-fname_base_out " << fname_base_out << endl;
	}
	if (f_cheat_sheet_orthogonal) {
		f_cheat_sheet_orthogonal = TRUE;
		cout << "-cheat_sheet_orthogonal "<< endl;
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
				<< endl;
	}
}



}}
