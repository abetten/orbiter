/*
 * design_activity_description.cpp
 *
 *  Created on: May 26, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


design_activity_description::design_activity_description()
{

	f_load_table = FALSE;
	//std::string load_table_label;
	//std::string load_table_group;

	//std::string load_table_H_label;
	//std::string load_table_H_group_order;
	//std::string load_table_H_gens;
	load_table_selected_orbit_length = 0;

	f_canonical_form = FALSE;
	Canonical_form_Descr = NULL;


	f_extract_solutions_by_index_csv = FALSE;
	f_extract_solutions_by_index_txt = FALSE;
	//std::string extract_solutions_by_index_label;
	//std::string extract_solutions_by_index_group;
	//std::string extract_solutions_by_index_fname_solutions_in;
	//std::string extract_solutions_by_index_fname_solutions_out;
	//std::string extract_solutions_by_index_prefix;

	f_export_inc = FALSE;
	f_intersection_matrix = FALSE;
	f_export_blocks = FALSE;
	f_row_sums = FALSE;
	f_tactical_decomposition = FALSE;
}

design_activity_description::~design_activity_description()
{

}


int design_activity_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "design_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-load_table") == 0) {
			f_load_table = TRUE;
			load_table_label.assign(argv[++i]);
			load_table_group.assign(argv[++i]);
			load_table_H_label.assign(argv[++i]);
			load_table_H_group_order.assign(argv[++i]);
			load_table_H_gens.assign(argv[++i]);
			load_table_selected_orbit_length = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-load_table " << load_table_label
						<< " " << load_table_group
						<< " " << load_table_H_label
						<< " " << load_table_H_group_order
						<< " " << load_table_H_gens
						<< " " << load_table_selected_orbit_length
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-canonical_form") == 0) {
			f_canonical_form = TRUE;
			if (f_v) {
				cout << "-canonical_form, reading extra arguments" << endl;
			}

			Canonical_form_Descr = NEW_OBJECT(combinatorics::classification_of_objects_description);

			i += Canonical_form_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			if (f_v) {
				cout << "done reading -f_canonical_form " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
		}
		else if (ST.stringcmp(argv[i], "-extract_solutions_by_index_csv") == 0) {
			f_extract_solutions_by_index_csv = TRUE;
			extract_solutions_by_index_label.assign(argv[++i]);
			extract_solutions_by_index_group.assign(argv[++i]);
			extract_solutions_by_index_fname_solutions_in.assign(argv[++i]);
			extract_solutions_by_index_fname_solutions_out.assign(argv[++i]);
			extract_solutions_by_index_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-extract_solutions_by_index_csv "
						<< extract_solutions_by_index_label << " "
						<< extract_solutions_by_index_group << " "
						<< extract_solutions_by_index_fname_solutions_in << " "
						<< extract_solutions_by_index_fname_solutions_out << " "
						<< extract_solutions_by_index_prefix << " "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-extract_solutions_by_index_txt") == 0) {
			f_extract_solutions_by_index_txt = TRUE;
			extract_solutions_by_index_label.assign(argv[++i]);
			extract_solutions_by_index_group.assign(argv[++i]);
			extract_solutions_by_index_fname_solutions_in.assign(argv[++i]);
			extract_solutions_by_index_fname_solutions_out.assign(argv[++i]);
			extract_solutions_by_index_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-extract_solutions_by_index_txt "
						<< extract_solutions_by_index_label << " "
						<< extract_solutions_by_index_group << " "
						<< extract_solutions_by_index_fname_solutions_in << " "
						<< extract_solutions_by_index_fname_solutions_out << " "
						<< extract_solutions_by_index_prefix << " "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_inc") == 0) {
			f_export_inc = TRUE;
			if (f_v) {
				cout << "-export_inc " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-intersection_matrix") == 0) {
			f_intersection_matrix = TRUE;
			if (f_v) {
				cout << "-intersection_matrix " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_blocks") == 0) {
			f_export_blocks = TRUE;
			if (f_v) {
				cout << "-export_blocks " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-row_sums") == 0) {
			f_row_sums = TRUE;
			if (f_v) {
				cout << "-row_sums " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-tactical_decomposition") == 0) {
			f_tactical_decomposition = TRUE;
			if (f_v) {
				cout << "-tactical_decomposition " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "design_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "design_activity_description::read_arguments done" << endl;
	return i + 1;
}

void design_activity_description::print()
{
	if (f_canonical_form) {
		cout << "-canonical_form " << endl;
		Canonical_form_Descr->print();
	}
	if (f_load_table) {
		cout << "-load_table " << load_table_label
				<< " " << load_table_group
				<< " " << load_table_H_label
				<< " " << load_table_H_group_order
				<< " " << load_table_H_gens
				<< " " << load_table_selected_orbit_length
				<< endl;
	}
	if (f_extract_solutions_by_index_csv) {
		cout << "-extract_solutions_by_index_csv "
				<< extract_solutions_by_index_label << " "
				<< extract_solutions_by_index_group << " "
				<< extract_solutions_by_index_fname_solutions_in << " "
				<< extract_solutions_by_index_fname_solutions_out << " "
				<< extract_solutions_by_index_prefix << " "
				<< endl;
	}
	if (f_extract_solutions_by_index_txt) {
		cout << "-extract_solutions_by_index_txt "
				<< extract_solutions_by_index_label << " "
				<< extract_solutions_by_index_group << " "
				<< extract_solutions_by_index_fname_solutions_in << " "
				<< extract_solutions_by_index_fname_solutions_out << " "
				<< extract_solutions_by_index_prefix << " "
				<< endl;
	}
	if (f_export_inc) {
		cout << "-export_inc " << endl;
	}
	if (f_intersection_matrix) {
		cout << "-intersection_matrix " << endl;
	}
	if (f_export_blocks) {
		cout << "-export_blocks " << endl;
	}
	if (f_row_sums) {
		cout << "-row_sums " << endl;
	}
	if (f_tactical_decomposition) {
		cout << "-tactical_decomposition " << endl;
	}
}



}}}


