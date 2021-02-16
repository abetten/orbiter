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

	f_create_BLT_set = FALSE;
	BLT_Set_create_description = NULL;

}

orthogonal_space_activity_description::~orthogonal_space_activity_description()
{

}


int orthogonal_space_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "orthogonal_space_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data = NEW_OBJECT(data_input_stream);
			cout << "-input" << endl;
			i += Data->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
			cout << "orthogonal_space_activity_description::read_arguments finished reading -input" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-create_BLT_set") == 0) {
			f_create_BLT_set = TRUE;
			BLT_Set_create_description = NEW_OBJECT(BLT_set_create_description);
			cout << "-create_BLT_set" << endl;
			i += BLT_Set_create_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
			cout << "orthogonal_space_activity_description::read_arguments finished reading -create_BLT_set" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-fname_base_out") == 0) {
			f_fname_base_out = TRUE;
			fname_base_out.assign(argv[++i]);
			cout << "-fname_base_out " << fname_base_out << endl;
		}
		else if (stringcmp(argv[i], "-cheat_sheet_orthogonal") == 0) {
			f_cheat_sheet_orthogonal = TRUE;
			//cheat_sheet_orthogonal_epsilon = strtoi(argv[++i]);
			//cheat_sheet_orthogonal_n = strtoi(argv[++i]);
			cout << "-cheat_sheet_orthogonal "<< endl;
		}
		else if (stringcmp(argv[i], "-unrank_line_through_two_points") == 0) {
			f_unrank_line_through_two_points = TRUE;
			unrank_line_through_two_points_p1.assign(argv[++i]);
			unrank_line_through_two_points_p2.assign(argv[++i]);
			cout << "-unrank_line_through_two_points " << unrank_line_through_two_points_p1
					<< " " << unrank_line_through_two_points_p2 << endl;
		}
		else if (stringcmp(argv[i], "-lines_on_point") == 0) {
			f_lines_on_point = TRUE;
			lines_on_point_rank = strtoi(argv[++i]);
			cout << "-lines_on_point " << lines_on_point_rank << endl;
		}

		else if (stringcmp(argv[i], "-perp") == 0) {
			f_perp = TRUE;
			perp_text.assign(argv[++i]);
			cout << "-perp " << perp_text << endl;
		}

		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "orthogonal_space_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "orthogonal_space_activity_description::read_arguments looping, i=" << i << endl;
	} // next i

	cout << "orthogonal_space_activity_description::read_arguments done" << endl;
	return i + 1;
}


}}
